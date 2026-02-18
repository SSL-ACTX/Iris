// src/mailbox.rs
//! Minimal mailbox implementation (unbounded, binary messages)

use tokio::sync::mpsc;
use bytes::Bytes;
use std::collections::VecDeque;

/// Message is an envelope that can be either a user payload (binary blob)
/// or a system message (e.g., exit notifications).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SystemMessage {
    Exit(u64),
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
    tx_user: mpsc::UnboundedSender<Bytes>,
    tx_sys: mpsc::UnboundedSender<SystemMessage>,
}

/// Receiver half of a mailbox.
pub struct MailboxReceiver {
    rx_user: mpsc::UnboundedReceiver<Bytes>,
    rx_sys: mpsc::UnboundedReceiver<SystemMessage>,
    stash: VecDeque<Message>,
}

/// Create a new mailbox channel (sender, receiver).
pub fn channel() -> (MailboxSender, MailboxReceiver) {
    let (tx_user, rx_user) = mpsc::unbounded_channel();
    let (tx_sys, rx_sys) = mpsc::unbounded_channel();
    (
        MailboxSender { tx_user, tx_sys },
        MailboxReceiver {
            rx_user,
            rx_sys,
            stash: VecDeque::new(),
        },
    )
}

impl MailboxSender {
    /// Send a message into the mailbox.
    pub fn send(&self, msg: Message) -> Result<(), Message> {
        match msg {
            Message::User(b) => {
                let backup = Message::User(b.clone());
                match self.tx_user.send(b) {
                    Ok(()) => Ok(()),
                    Err(_) => Err(backup),
                }
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
        let backup = b.clone();
        match self.tx_user.send(b) {
            Ok(()) => Ok(()),
            Err(_) => Err(backup),
        }
    }

    /// Convenience: send system message directly.
    pub fn send_system(&self, s: SystemMessage) -> Result<(), SystemMessage> {
        let backup = s.clone();
        match self.tx_sys.send(s) {
            Ok(()) => Ok(()),
            Err(_) => Err(backup),
        }
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
            return Some(front);
        }

        tokio::select! {
            biased;
            sys = self.rx_sys.recv() => {
                sys.map(Message::System)
            }
            user = self.rx_user.recv() => {
                user.map(Message::User)
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
            return Some(front);
        }

        self.rx_user.try_recv().ok().map(Message::User)
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
            return self.stash.remove(idx);
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
                user = self.rx_user.recv() => {
                    match user {
                        Some(b) => {
                            let m = Message::User(b);
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