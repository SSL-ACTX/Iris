// src/mailbox.rs
//! High-performance mailbox implementation with cache-line alignment and batching support.

use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::mpsc;

/// Aligned atomic for counter to avoid false sharing.
#[repr(align(64))]
struct AlignedCounter(AtomicUsize);

/// Underlying sender type for user messages; either unbounded or bounded.
#[derive(Clone)]
enum UserSender {
    Unbounded(mpsc::UnboundedSender<Message>),
    Bounded(mpsc::Sender<Message>),
}

/// Underlying receiver type for user messages.
enum UserReceiver {
    Unbounded(mpsc::UnboundedReceiver<Message>),
    Bounded(mpsc::Receiver<Message>),
}

/// Message is an envelope that can be either a user payload (binary blob)
/// or a system message (e.g., exit notifications).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExitReason {
    Normal,
    Panic,
    Timeout,
    Killed,
    Oom,
    Disconnected,
    RemotePanic,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExitInfo {
    pub from: u64,
    pub reason: ExitReason,
    pub metadata: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackpressureLevel {
    Normal,
    High,
    Critical,
}

impl BackpressureLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackpressureLevel::Normal => "NORMAL",
            BackpressureLevel::High => "HIGH",
            BackpressureLevel::Critical => "CRITICAL",
        }
    }
}

const HIGH_ENTER_PCT: u64 = 70;
const HIGH_EXIT_PCT: u64 = 60;
const CRITICAL_ENTER_PCT: u64 = 90;
const CRITICAL_EXIT_PCT: u64 = 80;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SystemMessage {
    Exit(ExitInfo),
    /// Hot Swap signal containing the raw pointer (usize)
    /// to the new handler function / closure.
    HotSwap(usize),
    /// Instruction for a bounded mailbox to drop its oldest user message.
    DropOld,
    /// Heartbeat signal to verify actor/node responsiveness.
    Ping,
    /// Response to a heartbeat signal.
    Pong,
    /// Backpressure notification from runtime.
    Backpressure(BackpressureLevel),
}

/// Strategy for handling overflow in a bounded mailbox.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OverflowPolicy {
    DropNew,
    DropOld,
    Block,
    /// Redirect overflowing messages to a fallback PID (payload is sent there instead).
    Redirect(u64),
    /// Send a copy to the fallback PID and still queue the message.
    Spill(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    User(Bytes),
    Request { reply_to: u64, payload: Bytes },
    System(SystemMessage),
}

/// Sender half of a mailbox. Optimized for high-throughput concurrent sends.
#[derive(Clone)]
#[repr(align(64))]
pub struct MailboxSender {
    tx_user: UserSender,
    tx_sys: mpsc::UnboundedSender<SystemMessage>,
    /// Count of user messages currently queued. Aligned to avoid false sharing.
    counter: Arc<AlignedCounter>,
}

/// Receiver half of a mailbox.
#[repr(align(64))]
pub struct MailboxReceiver {
    rx_user: UserReceiver,
    rx_sys: mpsc::UnboundedReceiver<SystemMessage>,
    stash: VecDeque<Message>,
    deferred_systems: usize,
    counter: Arc<AlignedCounter>,
}

/// Create a new unbounded mailbox channel (sender, receiver).
pub fn channel() -> (MailboxSender, MailboxReceiver) {
    let (tx_user, rx_user) = mpsc::unbounded_channel();
    let (tx_sys, rx_sys) = mpsc::unbounded_channel();
    let counter = Arc::new(AlignedCounter(AtomicUsize::new(0)));
    (
        MailboxSender {
            tx_user: UserSender::Unbounded(tx_user),
            tx_sys,
            counter: counter.clone(),
        },
        MailboxReceiver {
            rx_user: UserReceiver::Unbounded(rx_user),
            rx_sys,
            stash: VecDeque::new(),
            deferred_systems: 0,
            counter: counter.clone(),
        },
    )
}

/// Create a bounded mailbox channel with given capacity.
pub fn bounded_channel(capacity: usize) -> (MailboxSender, MailboxReceiver) {
    let (tx_user, rx_user) = mpsc::channel(capacity);
    let (tx_sys, rx_sys) = mpsc::unbounded_channel();
    let counter = Arc::new(AlignedCounter(AtomicUsize::new(0)));
    (
        MailboxSender {
            tx_user: UserSender::Bounded(tx_user),
            tx_sys,
            counter: counter.clone(),
        },
        MailboxReceiver {
            rx_user: UserReceiver::Bounded(rx_user),
            rx_sys,
            stash: VecDeque::new(),
            deferred_systems: 0,
            counter: counter.clone(),
        },
    )
}

impl MailboxSender {
    /// Send a message into the mailbox. Fast path for concurrent producers.
    #[inline(always)]
    pub fn send(&self, msg: Message) -> Result<(), Message> {
        match msg {
            Message::User(_) | Message::Request { .. } => {
                // Fetch-add is generally fast on modern CPUs
                self.counter.0.fetch_add(1, Ordering::Relaxed);
                let res = match &self.tx_user {
                    UserSender::Unbounded(tx) => tx.send(msg).map_err(|e| e.0),
                    UserSender::Bounded(tx) => match tx.try_send(msg) {
                        Ok(()) => Ok(()),
                        Err(mpsc::error::TrySendError::Full(m)) => Err(m),
                        Err(mpsc::error::TrySendError::Closed(m)) => Err(m),
                    },
                };
                if res.is_err() {
                    self.counter.0.fetch_sub(1, Ordering::Relaxed);
                }
                res
            }
            Message::System(s) => match self.tx_sys.send(s) {
                Ok(()) => Ok(()),
                Err(e) => Err(Message::System(e.0)),
            },
        }
    }

    /// Batch send to reduce atomic overhead and task wakeups (if supported by underlying channel).
    /// Currently emulated but prepared for lower-level batching.
    pub fn send_batch(&self, msgs: Vec<Message>) -> usize {
        let mut count = 0;
        for msg in msgs {
            if self.send(msg).is_ok() {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Convenience: send user bytes directly.
    #[inline(always)]
    pub fn send_user_bytes(&self, b: Bytes) -> Result<(), Bytes> {
        self.counter.0.fetch_add(1, Ordering::Relaxed);
        let msg = Message::User(b);
        let res = match &self.tx_user {
            UserSender::Unbounded(tx) => tx.send(msg).map_err(|e| match e.0 {
                Message::User(b) => b,
                _ => unreachable!(),
            }),
            UserSender::Bounded(tx) => match tx.try_send(msg) {
                Ok(()) => Ok(()),
                Err(err) => match err {
                    mpsc::error::TrySendError::Full(m) => Err(match m {
                        Message::User(b) => b,
                        _ => unreachable!(),
                    }),
                    mpsc::error::TrySendError::Closed(m) => Err(match m {
                        Message::User(b) => b,
                        _ => unreachable!(),
                    }),
                },
            },
        };
        if res.is_err() {
            self.counter.0.fetch_sub(1, Ordering::Relaxed);
        }
        res
    }

    /// Convenience: send system message directly.
    pub fn send_system(&self, s: SystemMessage) -> Result<(), SystemMessage> {
        match self.tx_sys.send(s) {
            Ok(()) => Ok(()),
            Err(e) => Err(e.0),
        }
    }

    pub fn len(&self) -> usize {
        self.counter.0.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn backpressure_level(&self, capacity: Option<usize>) -> BackpressureLevel {
        if let Some(cap) = capacity {
            let len = self.len();
            if cap == 0 {
                return BackpressureLevel::Critical;
            }
            if len >= cap {
                BackpressureLevel::Critical
            } else {
                let pct = (len as u64).saturating_mul(100) / (cap as u64);
                if pct >= 90 {
                    BackpressureLevel::Critical
                } else if pct >= 70 {
                    BackpressureLevel::High
                } else {
                    BackpressureLevel::Normal
                }
            }
        } else {
            BackpressureLevel::Normal
        }
    }

    pub fn backpressure_level_with_hysteresis(
        &self,
        capacity: Option<usize>,
        previous: BackpressureLevel,
    ) -> BackpressureLevel {
        let Some(cap) = capacity else {
            return BackpressureLevel::Normal;
        };
        if cap == 0 {
            return BackpressureLevel::Critical;
        }

        let len = self.len();
        if len >= cap {
            return BackpressureLevel::Critical;
        }

        let pct = (len as u64).saturating_mul(100) / (cap as u64);
        match previous {
            BackpressureLevel::Normal => {
                if pct >= CRITICAL_ENTER_PCT {
                    BackpressureLevel::Critical
                } else if pct >= HIGH_ENTER_PCT {
                    BackpressureLevel::High
                } else {
                    BackpressureLevel::Normal
                }
            }
            BackpressureLevel::High => {
                if pct >= CRITICAL_ENTER_PCT {
                    BackpressureLevel::Critical
                } else if pct < HIGH_EXIT_PCT {
                    BackpressureLevel::Normal
                } else {
                    BackpressureLevel::High
                }
            }
            BackpressureLevel::Critical => {
                if pct < HIGH_EXIT_PCT {
                    BackpressureLevel::Normal
                } else if pct < CRITICAL_EXIT_PCT {
                    BackpressureLevel::High
                } else {
                    BackpressureLevel::Critical
                }
            }
        }
    }
}

impl MailboxReceiver {
    fn drop_oldest_user_queued(&mut self) -> bool {
        if let Some(pos) = self
            .stash
            .iter()
            .position(|m| matches!(m, Message::User(_) | Message::Request { .. }))
        {
            let _ = self.stash.remove(pos);
            self.counter.0.fetch_sub(1, Ordering::Relaxed);
            return true;
        }

        let dropped = match &mut self.rx_user {
            UserReceiver::Unbounded(rx) => rx.try_recv().ok(),
            UserReceiver::Bounded(rx) => rx.try_recv().ok(),
        };
        if let Some(m) = dropped {
            if matches!(m, Message::User(_) | Message::Request { .. }) {
                self.counter.0.fetch_sub(1, Ordering::SeqCst);
            }
            true
        } else {
            false
        }
    }

    pub async fn recv(&mut self) -> Option<Message> {
        loop {
            if self.deferred_systems > 0 {
                if let Some(pos) = self
                    .stash
                    .iter()
                    .position(|m| matches!(m, Message::System(_)))
                {
                    self.deferred_systems = self.deferred_systems.saturating_sub(1);
                    if let Some(Message::System(SystemMessage::DropOld)) = self.stash.get(pos) {
                        let _ = self.stash.remove(pos);
                        let _ = self.drop_oldest_user_queued();
                        continue;
                    }
                    return self.stash.remove(pos);
                }
                self.deferred_systems = 0;
            }

            if let Ok(sys) = self.rx_sys.try_recv() {
                if matches!(sys, SystemMessage::DropOld) {
                    let _ = self.drop_oldest_user_queued();
                    continue;
                }
                return Some(Message::System(sys));
            }

            if let Some(front) = self.stash.pop_front() {
                if matches!(front, Message::User(_) | Message::Request { .. }) {
                    self.counter.0.fetch_sub(1, Ordering::SeqCst);
                }
                return Some(front);
            }

            tokio::select! {
                biased;
                sys = self.rx_sys.recv() => {
                    match sys {
                        Some(SystemMessage::DropOld) => {
                            let _ = self.drop_oldest_user_queued();
                            continue;
                        }
                        Some(s) => return Some(Message::System(s)),
                        None => return None,
                    }
                }
                user = {
                    async {
                        match &mut self.rx_user {
                            UserReceiver::Unbounded(rx) => rx.recv().await,
                            UserReceiver::Bounded(rx) => rx.recv().await,
                        }
                    }
                } => {
                    if let Some(m) = user {
                        if matches!(m, Message::User(_) | Message::Request { .. }) {
                            self.counter.0.fetch_sub(1, Ordering::SeqCst);
                        }
                        return Some(m);
                    } else {
                        return None;
                    }
                }
            }
        }
    }

    pub fn try_recv(&mut self) -> Option<Message> {
        loop {
            if self.deferred_systems > 0 {
                if let Some(pos) = self
                    .stash
                    .iter()
                    .position(|m| matches!(m, Message::System(_)))
                {
                    self.deferred_systems = self.deferred_systems.saturating_sub(1);
                    if let Some(Message::System(SystemMessage::DropOld)) = self.stash.get(pos) {
                        let _ = self.stash.remove(pos);
                        let _ = self.drop_oldest_user_queued();
                        continue;
                    }
                    return self.stash.remove(pos);
                }
                self.deferred_systems = 0;
            }

            if let Ok(sys) = self.rx_sys.try_recv() {
                if matches!(sys, SystemMessage::DropOld) {
                    let _ = self.drop_oldest_user_queued();
                    continue;
                }
                return Some(Message::System(sys));
            }

            if let Some(front) = self.stash.pop_front() {
                if matches!(front, Message::User(_) | Message::Request { .. }) {
                    self.counter.0.fetch_sub(1, Ordering::SeqCst);
                }
                return Some(front);
            }

            let opt = match &mut self.rx_user {
                UserReceiver::Unbounded(rx) => rx.try_recv().ok(),
                UserReceiver::Bounded(rx) => rx.try_recv().ok(),
            };
            return opt.map(|m| {
                if matches!(m, Message::User(_) | Message::Request { .. }) {
                    self.counter.0.fetch_sub(1, Ordering::SeqCst);
                }
                m
            });
        }
    }

    /// Try to receive up to `max` messages without awaiting.
    pub fn try_recv_batch(&mut self, max: usize) -> Vec<Message> {
        let mut batch = Vec::with_capacity(max);
        while batch.len() < max {
            if let Some(msg) = self.try_recv() {
                batch.push(msg);
            } else {
                break;
            }
        }
        batch
    }

    pub async fn selective_recv<F>(&mut self, mut matcher: F) -> Option<Message>
    where
        F: FnMut(&Message) -> bool,
    {
        while let Some(idx) = self
            .stash
            .iter()
            .position(|m| matches!(m, Message::System(SystemMessage::DropOld)))
        {
            self.deferred_systems = self.deferred_systems.saturating_sub(1);
            let _ = self.stash.remove(idx);
            let _ = self.drop_oldest_user_queued();
        }

        if let Some(idx) = self.stash.iter().position(&mut matcher) {
            let m = self.stash.remove(idx);
            if let Some(m_ref) = m.as_ref() {
                if matches!(m_ref, Message::User(_) | Message::Request { .. }) {
                    self.counter.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            return m;
        }

        loop {
            if let Ok(sys) = self.rx_sys.try_recv() {
                if matches!(sys, SystemMessage::DropOld) {
                    let _ = self.drop_oldest_user_queued();
                    continue;
                }
                let m = Message::System(sys);
                if matcher(&m) {
                    return Some(m);
                } else {
                    self.stash.push_back(m);
                    continue;
                }
            }

            tokio::select! {
                biased;
                sys = self.rx_sys.recv() => {
                    match sys {
                        Some(SystemMessage::DropOld) => {
                            let _ = self.drop_oldest_user_queued();
                            continue;
                        }
                        Some(s) => {
                            let m = Message::System(s);
                            if matcher(&m) {
                                return Some(m);
                            } else {
                                self.stash.push_back(m);
                                self.deferred_systems = self.deferred_systems.saturating_add(1);
                                continue;
                            }
                        }
                        None => return None,
                    }
                }
                user = {
                    async {
                        match &mut self.rx_user {
                            UserReceiver::Unbounded(rx) => rx.recv().await,
                            UserReceiver::Bounded(rx) => rx.recv().await,
                        }
                    }
                } => {
                    match user {
                        Some(m) => {
                            if matcher(&m) {
                                if matches!(m, Message::User(_) | Message::Request { .. }) {
                                    self.counter.0.fetch_sub(1, Ordering::SeqCst);
                                }
                                return Some(m);
                            } else {
                                self.stash.push_back(m);
                                continue;
                            }
                        }
                        None => return None,
                    }
                }
            }
        }
    }
}
