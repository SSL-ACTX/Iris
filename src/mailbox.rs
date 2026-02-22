// src/mailbox.rs
//! Minimal mailbox implementation (unbounded, binary messages)

use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SystemMessage {
    Exit(ExitInfo),
    /// Hot Swap signal containing the raw pointer (usize)
    /// to the new handler function / closure.
    HotSwap(usize),
    /// Heartbeat signal to verify actor/node responsiveness.
    Ping,
    /// Response to a heartbeat signal.
    Pong,
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
    counter: Arc<AtomicUsize>,
}

/// Create a new mailbox channel (sender, receiver).
/// Create a new unbounded mailbox channel (sender, receiver).
pub fn channel() -> (MailboxSender, MailboxReceiver) {
    let (tx_user, rx_user) = mpsc::unbounded_channel();
    let (tx_sys, rx_sys) = mpsc::unbounded_channel();
    let counter = Arc::new(AtomicUsize::new(0));
    (
        MailboxSender { tx_user: UserSender::Unbounded(tx_user), tx_sys, counter: counter.clone() },
        MailboxReceiver {
            rx_user: UserReceiver::Unbounded(rx_user),
            rx_sys,
            stash: VecDeque::new(),
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
        MailboxSender { tx_user: UserSender::Bounded(tx_user), tx_sys, counter: counter.clone() },
        MailboxReceiver {
            rx_user: UserReceiver::Bounded(rx_user),
            rx_sys,
            stash: VecDeque::new(),
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
                self.counter.fetch_add(1, Ordering::SeqCst);
                let backup = Message::User(b.clone());
                let res = match &self.tx_user {
                    UserSender::Unbounded(tx) => tx.send(b).map_err(|_| backup.clone()),
                    UserSender::Bounded(tx) => match tx.try_send(b) {
                        Ok(()) => Ok(()),
                        Err(_e) => Err(backup.clone()),
                    },
                };
                if res.is_err() {
                    // rollback counter
                    self.counter.fetch_sub(1, Ordering::SeqCst);
                }
                res
            }
            Message::System(s) => {
                let backup = Message::System(s.clone());
                match self.tx_sys.send(s) {
                    Ok(()) => Ok(()),
                    Err(_) => Err(backup),
                }
            }
        }
    }

    /// Convenience: send user bytes directly.
    pub fn send_user_bytes(&self, b: Bytes) -> Result<(), Bytes> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        let backup = b.clone();
        let res = match &self.tx_user {
            UserSender::Unbounded(tx) => tx.send(b).map_err(|_e| backup.clone()),
            UserSender::Bounded(tx) => match tx.try_send(b) {
                Ok(()) => Ok(()),
                Err(err) => match err {
                    mpsc::error::TrySendError::Full(_) => Err(backup.clone()),
                    mpsc::error::TrySendError::Closed(_) => Err(backup.clone()),
                },
            },
        };
        if res.is_err() {
            self.counter.fetch_sub(1, Ordering::SeqCst);
        }
        res
    }

    /// Convenience: send system message directly.
    pub fn send_system(&self, s: SystemMessage) -> Result<(), SystemMessage> {
        let backup = s.clone();
        match self.tx_sys.send(s) {
            Ok(()) => Ok(()),
            Err(_) => Err(backup),
        }
    }

    /// Return the number of user messages currently queued for this mailbox.
    pub fn len(&self) -> usize {
        self.counter.load(Ordering::SeqCst)
    }
}

impl MailboxReceiver {
    /// Await a message from the mailbox, prioritizing any already-enqueued
    /// system messages.
    pub async fn recv(&mut self) -> Option<Message> {
        // Prefer any system messages already in the stash.
        if let Some(pos) = self
            .stash
            .iter()
            .position(|m| matches!(m, Message::System(_)))
        {
            return self.stash.remove(pos);
        }

        if let Ok(sys) = self.rx_sys.try_recv() {
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
                    Some(s) => Some(Message::System(s)),
                    None => None,
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
                    Some(m)
                } else {
                    None
                }
            }
        }
    }

    /// Try to receive without awaiting; system messages are preferred.
    pub fn try_recv(&mut self) -> Option<Message> {
        // Prefer any system messages already in the stash.
        if let Some(pos) = self
            .stash
            .iter()
            .position(|m| matches!(m, Message::System(_)))
        {
            return self.stash.remove(pos);
        }

        if let Ok(sys) = self.rx_sys.try_recv() {
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
        opt.map(|b| {
            self.counter.fetch_sub(1, Ordering::SeqCst);
            Message::User(b)
        })
    }

    /// Selective receive: await until a message matching `matcher` arrives.
    /// Messages that don't match are stashed and will be delivered by subsequent
    /// `recv()`/`try_recv()` calls in the order they were encountered.
    pub async fn selective_recv<F>(&mut self, mut matcher: F) -> Option<Message>
    where
        F: FnMut(&Message) -> bool,
    {
        // First, search stash for a matching message (preserve ordering).
        if let Some(idx) = self.stash.iter().position(|m| matcher(m)) {
            let m = self.stash.remove(idx);
            if let Some(Message::User(_)) = m.as_ref() {
                self.counter.fetch_sub(1, Ordering::SeqCst);
            }
            return m;
        }

        loop {
            // Prefer any immediately available system messages.
            if let Ok(sys) = self.rx_sys.try_recv() {
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
                        Some(s) => {
                            let m = Message::System(s);
                            if matcher(&m) {
                                return Some(m);
                            } else {
                                self.stash.push_back(m);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn send_and_recv() {
        let (tx, mut rx) = channel();
        tx.send(Message::User(Bytes::from_static(b"hello")))
            .unwrap();
        let got = rx.recv().await.expect("should receive");
        match got {
            Message::User(buf) => assert_eq!(buf.as_ref(), b"hello"),
            _ => panic!("expected user message"),
        }
    }

    #[tokio::test]
    async fn bounded_mailbox_drop_new() {
        // This test will fail until bounded mailbox is implemented.
        let (tx, mut rx) = bounded_channel(2);
        tx.send(Message::User(Bytes::from_static(b"m1")))
            .unwrap();
        tx.send(Message::User(Bytes::from_static(b"m2")))
            .unwrap();
        // third send should be rejected because capacity is 2
        assert!(tx
            .send(Message::User(Bytes::from_static(b"m3")))
            .is_err());

        let first = rx.recv().await.expect("first");
        let second = rx.recv().await.expect("second");
        match first {
            Message::User(b) => assert_eq!(b.as_ref(), b"m1"),
            _ => panic!(),
        }
        match second {
            Message::User(b) => assert_eq!(b.as_ref(), b"m2"),
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn selective_receive_defers_and_preserves_order() {
        let (tx, mut rx) = channel();

        tx.send(Message::User(Bytes::from_static(b"m1"))).unwrap();
        tx.send(Message::User(Bytes::from_static(b"target")))
            .unwrap();
        tx.send(Message::User(Bytes::from_static(b"m3"))).unwrap();

        // Selectively receive the message whose bytes == b"target"
        let got = rx
            .selective_recv(|m| match m {
                Message::User(b) => b.as_ref() == b"target",
                _ => false,
            })
            .await
            .expect("should find target");

        match got {
            Message::User(b) => assert_eq!(b.as_ref(), b"target"),
            _ => panic!("expected user message"),
        }

        // After selective receive, deferred messages should be delivered in order.
        let first = rx.recv().await.expect("first deferred");
        let second = rx.recv().await.expect("second deferred");

        match first {
            Message::User(b) => assert_eq!(b.as_ref(), b"m1"),
            _ => panic!("expected user message"),
        }

        match second {
            Message::User(b) => assert_eq!(b.as_ref(), b"m3"),
            _ => panic!("expected user message"),
        }
    }
}
