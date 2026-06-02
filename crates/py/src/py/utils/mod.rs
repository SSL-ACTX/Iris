// src/py/utils.rs
//! Shared helpers for converting between Rust and Python messages.
#![allow(non_local_definitions)]

use crate::mailbox;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Execute a Python callback while safely catching Rust panics and
/// Python exceptions. Returns `true` if the callback completed normally.
pub(crate) fn run_python_callback_py(
    py: Python<'_>,
    f: impl FnOnce(Python<'_>) -> PyResult<()>,
) -> bool {
    let result = catch_unwind(AssertUnwindSafe(|| match f(py) {
        Ok(()) => Ok(()),
        Err(err) => {
            eprintln!("[Iris] Python actor exception: {}", err);
            // In modern PyO3, we use write_unraisable or similar to avoid SystemExit termination
            err.write_unraisable(py, None);
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
pub(crate) fn run_python_callback(f: impl FnOnce(Python<'_>) -> PyResult<()>) -> bool {
    Python::attach(|py| run_python_callback_py(py, f))
}

/// Python-friendly structured system message used during conversions.
#[pyclass(from_py_object)]
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

#[pyclass(from_py_object)]
pub struct PyRequest {
    #[pyo3(get)]
    pub reply_to: u64,
    #[pyo3(get)]
    pub payload: Py<PyAny>,
}

impl Clone for PyRequest {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            reply_to: self.reply_to,
            payload: self.payload.clone_ref(py),
        })
    }
}

/// Convert a Rust `Message` into a Python object suitable
/// for passing back to the interpreter.
pub(crate) fn message_to_py(py: Python<'_>, msg: mailbox::Message) -> Py<PyAny> {
    match msg {
        mailbox::Message::User(b) => PyBytes::new(py, &b).into_any().unbind(),
        mailbox::Message::Request { reply_to, payload } => {
            let obj = PyRequest {
                reply_to,
                payload: PyBytes::new(py, &payload).unbind().into_any(),
            };
            Bound::new(py, obj)
                .expect("Failed to create PyRequest")
                .into_any()
                .unbind()
        }
        mailbox::Message::System(mailbox::SystemMessage::Exit(info)) => {
            let reason = match info.reason {
                mailbox::ExitReason::Normal => "normal".to_string(),
                mailbox::ExitReason::Panic => "panic".to_string(),
                mailbox::ExitReason::Timeout => "timeout".to_string(),
                mailbox::ExitReason::Killed => "killed".to_string(),
                mailbox::ExitReason::Oom => "oom".to_string(),
                mailbox::ExitReason::Disconnected => "disconnected".to_string(),
                mailbox::ExitReason::RemotePanic => "remote_panic".to_string(),
                mailbox::ExitReason::Other(ref s) => s.clone(),
            };

            let obj = PySystemMessage {
                type_name: "EXIT".to_string(),
                target_pid: Some(info.from),
                reason,
                metadata: info.metadata.clone(),
            };
            Bound::new(py, obj)
                .expect("Failed to create PySystemMessage")
                .into_any()
                .unbind()
        }
        mailbox::Message::System(mailbox::SystemMessage::HotSwap(_)) => {
            let obj = PySystemMessage {
                type_name: "HOT_SWAP".to_string(),
                target_pid: None,
                reason: "".to_string(),
                metadata: None,
            };
            Bound::new(py, obj)
                .expect("Failed to create PySystemMessage")
                .into_any()
                .unbind()
        }
        mailbox::Message::System(mailbox::SystemMessage::Ping) => {
            let obj = PySystemMessage {
                type_name: "PING".to_string(),
                target_pid: None,
                reason: "".to_string(),
                metadata: None,
            };
            Bound::new(py, obj)
                .expect("Failed to create PySystemMessage")
                .into_any()
                .unbind()
        }
        mailbox::Message::System(mailbox::SystemMessage::Pong) => {
            let obj = PySystemMessage {
                type_name: "PONG".to_string(),
                target_pid: None,
                reason: "".to_string(),
                metadata: None,
            };
            Bound::new(py, obj)
                .expect("Failed to create PySystemMessage")
                .into_any()
                .unbind()
        }
        mailbox::Message::System(mailbox::SystemMessage::Backpressure(level)) => {
            let obj = PySystemMessage {
                type_name: "BACKPRESSURE".to_string(),
                target_pid: None,
                reason: level.as_str().to_string(),
                metadata: None,
            };
            Bound::new(py, obj)
                .expect("Failed to create PySystemMessage")
                .into_any()
                .unbind()
        }
        mailbox::Message::System(mailbox::SystemMessage::DropOld) => py.None(),
    }
}

/// Run a Python matcher callback against a Rust message.
pub(crate) fn run_python_matcher(
    py: Python<'_>,
    matcher: &Bound<'_, PyAny>,
    msg: &mailbox::Message,
) -> bool {
    match msg {
        mailbox::Message::User(b) => {
            let py_bytes = PyBytes::new(py, b);
            match matcher.call1((py_bytes,)) {
                Ok(val) => val.extract::<bool>().unwrap_or(false),
                Err(_) => false,
            }
        }
        mailbox::Message::Request { reply_to, payload } => {
            let obj = PyRequest {
                reply_to: *reply_to,
                payload: PyBytes::new(py, payload).unbind().into_any(),
            };
            let py_obj = Bound::new(py, obj).expect("Failed to create PyRequest");
            match matcher.call1((py_obj,)) {
                Ok(val) => val.extract::<bool>().unwrap_or(false),
                Err(_) => false,
            }
        }
        mailbox::Message::System(s) => match s {
            mailbox::SystemMessage::Exit(info) => {
                let reason = match info.reason {
                    mailbox::ExitReason::Normal => "normal".to_string(),
                    mailbox::ExitReason::Panic => "panic".to_string(),
                    mailbox::ExitReason::Timeout => "timeout".to_string(),
                    mailbox::ExitReason::Killed => "killed".to_string(),
                    mailbox::ExitReason::Oom => "oom".to_string(),
                    mailbox::ExitReason::Disconnected => "disconnected".to_string(),
                    mailbox::ExitReason::RemotePanic => "remote_panic".to_string(),
                    mailbox::ExitReason::Other(ref s) => s.clone(),
                };

                let obj = PySystemMessage {
                    type_name: "EXIT".to_string(),
                    target_pid: Some(info.from),
                    reason,
                    metadata: info.metadata.clone(),
                };
                let py_obj = Bound::new(py, obj).expect("Failed to create PySystemMessage");
                match matcher.call1((py_obj,)) {
                    Ok(val) => val.extract::<bool>().unwrap_or(false),
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
                let py_obj = Bound::new(py, obj).expect("Failed to create PySystemMessage");
                match matcher.call1((py_obj,)) {
                    Ok(val) => val.extract::<bool>().unwrap_or(false),
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
                let py_obj = Bound::new(py, obj).expect("Failed to create PySystemMessage");
                match matcher.call1((py_obj,)) {
                    Ok(val) => val.extract::<bool>().unwrap_or(false),
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
                let py_obj = Bound::new(py, obj).expect("Failed to create PySystemMessage");
                match matcher.call1((py_obj,)) {
                    Ok(val) => val.extract::<bool>().unwrap_or(false),
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
                let py_obj = Bound::new(py, obj).expect("Failed to create PySystemMessage");
                match matcher.call1((py_obj,)) {
                    Ok(val) => val.extract::<bool>().unwrap_or(false),
                    Err(_) => false,
                }
            }
            mailbox::SystemMessage::DropOld => false,
        },
    }
}
