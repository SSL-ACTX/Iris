// src/py/utils.rs
//! Shared helpers for converting between Rust and Python messages.
#![allow(non_local_definitions)]

use crate::mailbox;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Execute a Python callback while safely catching Rust panics and
/// Python exceptions. Returns `true` if the callback completed normally.
pub(crate) fn run_python_callback_py<F>(py: Python, f: F) -> bool
where
    F: FnOnce(Python) -> PyResult<()>,
{
    let result = catch_unwind(AssertUnwindSafe(|| match f(py) {
        Ok(()) => Ok(()),
        Err(err) => {
            eprintln!("[Iris] Python actor exception: {}", err);
            // PyErr::print() calls CPython's PyErr_Print, which terminates the process
            // if the error is SystemExit. We must completely avoid it for SystemExit.
            if !err.is_instance_of::<pyo3::exceptions::PySystemExit>(py) {
                err.print(py);
            }
            Err(())
        }
    }));

    match result {
        Ok(Ok(())) => true,
        Ok(Err(())) => false,
        Err(payload) => {
            eprintln!("[Iris] Python actor unwind: {:?}", payload);
            false
        }
    }
}

/// Execute a Python callback while safely catching Rust panics and
/// Python exceptions. Returns `true` if the callback completed normally.
pub(crate) fn run_python_callback<F>(f: F) -> bool
where
    F: FnOnce(Python) -> PyResult<()>,
{
    Python::with_gil(|py| run_python_callback_py(py, f))
}

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
        mailbox::Message::System(mailbox::SystemMessage::HotSwap(_)) => PySystemMessage {
            type_name: "HOT_SWAP".to_string(),
            target_pid: None,
            reason: "".to_string(),
            metadata: None,
        }
        .into_py(py),
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
        mailbox::Message::System(mailbox::SystemMessage::Backpressure(level)) => PySystemMessage {
            type_name: "BACKPRESSURE".to_string(),
            target_pid: None,
            reason: level.as_str().to_string(),
            metadata: None,
        }
        .into_py(py),
        mailbox::Message::System(mailbox::SystemMessage::DropOld) => py.None(),
    }
}

/// Run a Python matcher callback against a Rust message.
pub(crate) fn run_python_matcher(py: Python, matcher: &PyObject, msg: &mailbox::Message) -> bool {
    match msg {
        mailbox::Message::User(b) => match matcher.call1(py, (PyBytes::new(py, b),)) {
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
            mailbox::SystemMessage::Backpressure(level) => {
                let obj = PySystemMessage {
                    type_name: "BACKPRESSURE".to_string(),
                    target_pid: None,
                    reason: level.as_str().to_string(),
                    metadata: None,
                };
                match matcher.call1(py, (obj.into_py(py),)) {
                    Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                    Err(_) => false,
                }
            }
            mailbox::SystemMessage::DropOld => false,
        },
    }
}
