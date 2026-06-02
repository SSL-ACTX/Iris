use crate::mailbox::Message;
use std::sync::{Arc, Mutex};

/// A shared, append-only immutable log of messages.
pub struct LogStream {
    pub(crate) messages: Mutex<Vec<Message>>,
}

impl LogStream {
    pub fn new() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
        }
    }

    /// Append a message to the stream.
    pub fn append(&self, msg: Message) -> usize {
        let mut guard = self.messages.lock().unwrap();
        guard.push(msg);
        guard.len() - 1 // Return offset
    }

    /// Read messages from a specific offset to the end.
    pub fn read_from(&self, offset: usize) -> Vec<Message> {
        let guard = self.messages.lock().unwrap();
        if offset >= guard.len() {
            Vec::new()
        } else {
            guard[offset..].to_vec()
        }
    }

    pub fn len(&self) -> usize {
        self.messages.lock().unwrap().len()
    }
}

/// Represents an actor's position (pointer) in a LogStream.
pub struct ProjectionPointer {
    pub stream: Arc<LogStream>,
    pub current_offset: Mutex<usize>,
}

impl ProjectionPointer {
    pub fn new(stream: Arc<LogStream>, start_offset: usize) -> Self {
        Self {
            stream,
            current_offset: Mutex::new(start_offset),
        }
    }

    /// Fetch new messages and advance the pointer.
    pub fn poll(&self) -> Vec<Message> {
        let mut offset = self.current_offset.lock().unwrap();
        let msgs = self.stream.read_from(*offset);
        *offset += msgs.len();
        msgs
    }
}
