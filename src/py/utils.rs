// src/py/utils.rs
//! Shared helpers for converting between Rust and Python messages.
#![allow(non_local_definitions)]

use crate::mailbox;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Python-friendly structured system message used during conversions.
#[pyclass]
#[derive(Clone)]
pub struct PySystemMessage {
    #[pyo3(get)]
    pub type_name: String,
    #[pyo3(get)]
    pub target_pid: Option<u64>,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub metadata: Option<String>,
}

/// Convert a Rust `Message` into a Python object suitable
/// for passing back to the interpreter.
pub(crate) fn message_to_py(py: Python, msg: mailbox::Message) -> PyObject {
    match msg {
        mailbox::Message::User(b) => PyBytes::new(py, &b).into_py(py),
        mailbox::Message::System(mailbox::SystemMessage::Exit(info)) => {
            let reason = match info.reason {
                mailbox::ExitReason::Normal => "normal".to_string(),
                mailbox::ExitReason::Panic => "panic".to_string(),
                mailbox::ExitReason::Timeout => "timeout".to_string(),
                mailbox::ExitReason::Killed => "killed".to_string(),
                mailbox::ExitReason::Oom => "oom".to_string(),
                mailbox::ExitReason::Other(ref s) => s.clone(),
            };

            PySystemMessage {
                type_name: "EXIT".to_string(),
                target_pid: Some(info.from),
                reason,
                metadata: info.metadata.clone(),
            }
            .into_py(py)
        }
        mailbox::Message::System(mailbox::SystemMessage::HotSwap(_)) => {
            PySystemMessage {
                type_name: "HOT_SWAP".to_string(),
                target_pid: None,
                reason: "".to_string(),
                metadata: None,
            }
            .into_py(py)
        }
        mailbox::Message::System(mailbox::SystemMessage::Ping) => PySystemMessage {
            type_name: "PING".to_string(),
            target_pid: None,
            reason: "".to_string(),
            metadata: None,
        }
        .into_py(py),
        mailbox::Message::System(mailbox::SystemMessage::Pong) => PySystemMessage {
            type_name: "PONG".to_string(),
            target_pid: None,
            reason: "".to_string(),
            metadata: None,
        }
        .into_py(py),
    }
}

/// Run a Python matcher callback against a Rust message.
pub(crate) fn run_python_matcher(
    py: Python,
    matcher: &PyObject,
    msg: &mailbox::Message,
) -> bool {
    match msg {
        mailbox::Message::User(b) => match matcher.call1(py, (PyBytes::new(py, &b),)) {
            Ok(val) => val.extract::<bool>(py).unwrap_or(false),
            Err(_) => false,
        },
        mailbox::Message::System(s) => match s {
            mailbox::SystemMessage::Exit(info) => {
                let reason = match info.reason {
                    mailbox::ExitReason::Normal => "normal".to_string(),
                    mailbox::ExitReason::Panic => "panic".to_string(),
                    mailbox::ExitReason::Timeout => "timeout".to_string(),
                    mailbox::ExitReason::Killed => "killed".to_string(),
                    mailbox::ExitReason::Oom => "oom".to_string(),
                    mailbox::ExitReason::Other(ref s) => s.clone(),
                };
                let obj = PySystemMessage {
                    type_name: "EXIT".to_string(),
                    target_pid: Some(info.from),
                    reason,
                    metadata: info.metadata.clone(),
                };
                match matcher.call1(py, (obj.into_py(py),)) {
                    Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                    Err(_) => false,
                }
            }
            mailbox::SystemMessage::HotSwap(_) => {
                let obj = PySystemMessage {
                    type_name: "HOT_SWAP".to_string(),
                    target_pid: None,
                    reason: "".to_string(),
                    metadata: None,
                };
                match matcher.call1(py, (obj.into_py(py),)) {
                    Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                    Err(_) => false,
                }
            }
            mailbox::SystemMessage::Ping => {
                let obj = PySystemMessage {
                    type_name: "PING".to_string(),
                    target_pid: None,
                    reason: "".to_string(),
                    metadata: None,
                };
                match matcher.call1(py, (obj.into_py(py),)) {
                    Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                    Err(_) => false,
                }
            }
            mailbox::SystemMessage::Pong => {
                let obj = PySystemMessage {
                    type_name: "PONG".to_string(),
                    target_pid: None,
                    reason: "".to_string(),
                    metadata: None,
                };
                match matcher.call1(py, (obj.into_py(py),)) {
                    Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                    Err(_) => false,
                }
            }
        },
    }
}
