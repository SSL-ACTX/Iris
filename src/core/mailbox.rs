// src/mailbox.rs
//! Minimal mailbox implementation (unbounded, binary messages)

use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::mpsc;

/// Underlying sender type for user messages; either unbounded or bounded.
#[derive(Clone)]
enum UserSender {
    Unbounded(mpsc::UnboundedSender<Bytes>),
    Bounded(mpsc::Sender<Bytes>),
}

/// Underlying receiver type for user messages.
enum UserReceiver {
    Unbounded(mpsc::UnboundedReceiver<Bytes>),
    Bounded(mpsc::Receiver<Bytes>),
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
    System(SystemMessage),
}

/// Sender half of a mailbox. Internally we keep separate channels for
/// system messages and user messages so receivers can prioritize system
/// messages (e.g., Exit notifications) over user payloads.
#[derive(Clone)]
pub struct MailboxSender {
    tx_user: UserSender,
    tx_sys: mpsc::UnboundedSender<SystemMessage>,
    /// Count of user messages currently queued (including those in channel and stash).
    counter: Arc<AtomicUsize>,
}

/// Receiver half of a mailbox.
pub struct MailboxReceiver {
    rx_user: UserReceiver,
    rx_sys: mpsc::UnboundedReceiver<SystemMessage>,
    stash: VecDeque<Message>,
    deferred_systems: usize,
    counter: Arc<AtomicUsize>,
}

/// Create a new mailbox channel (sender, receiver).
/// Create a new unbounded mailbox channel (sender, receiver).
pub fn channel() -> (MailboxSender, MailboxReceiver) {
    let (tx_user, rx_user) = mpsc::unbounded_channel();
    let (tx_sys, rx_sys) = mpsc::unbounded_channel();
    let counter = Arc::new(AtomicUsize::new(0));
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

/// Create a bounded mailbox channel with given capacity. If the queue is
/// full, `send` will return Err(msg) (drop-new policy).
pub fn bounded_channel(capacity: usize) -> (MailboxSender, MailboxReceiver) {
    let (tx_user, rx_user) = mpsc::channel(capacity);
    let (tx_sys, rx_sys) = mpsc::unbounded_channel();
    let counter = Arc::new(AtomicUsize::new(0));
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
    /// Send a message into the mailbox.
    /// For bounded user queues, policy is drop-new: error returned when full.
    pub fn send(&self, msg: Message) -> Result<(), Message> {
        match msg {
            Message::User(b) => {
                // increment counter before enqueue attempt
                self.counter.fetch_add(1, Ordering::Relaxed);
                let res = match &self.tx_user {
                    UserSender::Unbounded(tx) => tx.send(b).map_err(|e| Message::User(e.0)),
                    UserSender::Bounded(tx) => match tx.try_send(b) {
                        Ok(()) => Ok(()),
                        Err(mpsc::error::TrySendError::Full(b)) => Err(Message::User(b)),
                        Err(mpsc::error::TrySendError::Closed(b)) => Err(Message::User(b)),
                    },
                };
                if res.is_err() {
                    // rollback counter
                    self.counter.fetch_sub(1, Ordering::Relaxed);
                }
                res
            }
            Message::System(s) => match self.tx_sys.send(s) {
                Ok(()) => Ok(()),
                Err(e) => Err(Message::System(e.0)),
            },
        }
    }

    /// Convenience: send user bytes directly.
    pub fn send_user_bytes(&self, b: Bytes) -> Result<(), Bytes> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        let res = match &self.tx_user {
            UserSender::Unbounded(tx) => tx.send(b).map_err(|e| e.0),
            UserSender::Bounded(tx) => match tx.try_send(b) {
                Ok(()) => Ok(()),
                Err(err) => match err {
                    mpsc::error::TrySendError::Full(b) => Err(b),
                    mpsc::error::TrySendError::Closed(b) => Err(b),
                },
            },
        };
        if res.is_err() {
            self.counter.fetch_sub(1, Ordering::Relaxed);
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

    /// Return the number of user messages currently queued for this mailbox.
    pub fn len(&self) -> usize {
        self.counter.load(Ordering::Relaxed)
    }

    /// Return true if the mailbox has no user messages.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compute backpressure level given an optional configured capacity.
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

    /// Compute backpressure using hysteresis to avoid level flapping near thresholds.
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
            .position(|m| matches!(m, Message::User(_)))
        {
            let _ = self.stash.remove(pos);
            self.counter.fetch_sub(1, Ordering::Relaxed);
            return true;
        }

        let dropped = match &mut self.rx_user {
            UserReceiver::Unbounded(rx) => rx.try_recv().ok().is_some(),
            UserReceiver::Bounded(rx) => rx.try_recv().ok().is_some(),
        };
        if dropped {
            self.counter.fetch_sub(1, Ordering::SeqCst);
        }
        dropped
    }

    /// Await a message from the mailbox, prioritizing any already-enqueued
    /// system messages.
    pub async fn recv(&mut self) -> Option<Message> {
        loop {
            // Prefer any system messages already in the stash.
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

            // If there are deferred user messages, deliver them before awaiting new ones.
            if let Some(front) = self.stash.pop_front() {
                if matches!(front, Message::User(_)) {
                    self.counter.fetch_sub(1, Ordering::SeqCst);
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
                            UserReceiver::Unbounded(rx) => rx.recv().await.map(Message::User),
                            UserReceiver::Bounded(rx) => rx.recv().await.map(Message::User),
                        }
                    }
                } => {
                    if let Some(m) = user {
                        self.counter.fetch_sub(1, Ordering::SeqCst);
                        return Some(m);
                    } else {
                        return None;
                    }
                }
            }
        }
    }

    /// Try to receive without awaiting; system messages are preferred.
    pub fn try_recv(&mut self) -> Option<Message> {
        loop {
            // Prefer any system messages already in the stash.
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

            // Deliver deferred user messages first, then try underlying channel.
            if let Some(front) = self.stash.pop_front() {
                if matches!(front, Message::User(_)) {
                    self.counter.fetch_sub(1, Ordering::SeqCst);
                }
                return Some(front);
            }

            let opt = match &mut self.rx_user {
                UserReceiver::Unbounded(rx) => rx.try_recv().ok(),
                UserReceiver::Bounded(rx) => rx.try_recv().ok(),
            };
            return opt.map(|b| {
                self.counter.fetch_sub(1, Ordering::SeqCst);
                Message::User(b)
            });
        }
    }

    /// Selective receive: await until a message matching `matcher` arrives.
    /// Messages that don't match are stashed and will be delivered by subsequent
    /// `recv()`/`try_recv()` calls in the order they were encountered.
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

        // First, search stash for a matching message (preserve ordering).
        if let Some(idx) = self.stash.iter().position(&mut matcher) {
            let m = self.stash.remove(idx);
            if let Some(Message::User(_)) = m.as_ref() {
                self.counter.fetch_sub(1, Ordering::SeqCst);
            }
            return m;
        }

        loop {
            // Prefer any immediately available system messages.
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
                            UserReceiver::Unbounded(rx) => rx.recv().await.map(Message::User),
                            UserReceiver::Bounded(rx) => rx.recv().await.map(Message::User),
                        }
                    }
                } => {
                    match user {
                        Some(m) => {
                            if matcher(&m) {
                                self.counter.fetch_sub(1, Ordering::SeqCst);
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
