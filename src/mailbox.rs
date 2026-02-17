// src/mailbox.rs
//! Minimal mailbox implementation (unbounded, binary messages)

use tokio::sync::mpsc;
use bytes::Bytes;

/// Message is an envelope that can be either a user payload (binary blob)
/// or a system message (e.g., exit notifications).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SystemMessage {
    Exit(u64),
    /// Phase 4: Hot Swap signal containing the raw pointer (usize)
    /// to the new handler function / closure.
    HotSwap(usize),
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
}

/// Create a new mailbox channel (sender, receiver).
pub fn channel() -> (MailboxSender, MailboxReceiver) {
    let (tx_user, rx_user) = mpsc::unbounded_channel();
    let (tx_sys, rx_sys) = mpsc::unbounded_channel();
    (
        MailboxSender { tx_user, tx_sys },
     MailboxReceiver { rx_user, rx_sys },
    )
}

impl MailboxSender {
    /// Send a message into the mailbox.
    ///
    /// Clone the payload so we can return the original `Message` on failure
    /// without relying on internals of `SendError`.
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
    /// system messages. If no system message is immediately available we await
    /// on both channels and return whichever arrives first.
    pub async fn recv(&mut self) -> Option<Message> {
        // Fast-path: prefer any already-queued system messages.
        if let Ok(sys) = self.rx_sys.try_recv() {
            return Some(Message::System(sys));
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
        if let Ok(sys) = self.rx_sys.try_recv() {
            return Some(Message::System(sys));
        }
        self.rx_user.try_recv().ok().map(Message::User)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn send_and_recv() {
        let (tx, mut rx) = channel();
        tx.send(Message::User(Bytes::from_static(b"hello"))).unwrap();
        let got = rx.recv().await.expect("should receive");
        match got {
            Message::User(buf) => assert_eq!(buf.as_ref(), b"hello"),
            _ => panic!("expected user message"),
        }
    }
}
