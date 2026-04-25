// src/py/mailbox.rs
//! Python mailbox wrapper providing blocking recv/selective_recv.
#![allow(non_local_definitions)]

use crate::py::utils::{message_to_py, run_python_matcher};
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// A wrapper around a live MailboxReceiver for Python actors.
/// Revamped: Now purely blocking/synchronous to Python, running in dedicated threads.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyMailbox {
    pub(crate) inner: Arc<TokioMutex<crate::mailbox::MailboxReceiver>>,
}

#[pymethods]
impl PyMailbox {
    /// Receive the next message (Blocking).
    /// Releases the GIL while waiting. Checks for Python signals cleanly to allow Ctrl+C escapes.
    #[pyo3(signature = (timeout=None))]
    fn recv(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Py<PyAny>> {
        let rx = self.inner.clone();
        let start = std::time::Instant::now();
        let timeout_dur = timeout.map(std::time::Duration::from_secs_f64);

        loop {
            // Respect interrupt signals (Ctrl-C) to prevent hangs
            py.check_signals()?;

            let wait_time = std::time::Duration::from_millis(100);
            let actual_wait = if let Some(t) = timeout_dur {
                let elapsed = start.elapsed();
                if elapsed >= t {
                    return Ok(py.None());
                }
                std::cmp::min(wait_time, t.saturating_sub(elapsed))
            } else {
                wait_time
            };

            // Release GIL to allow other threads to run while we block on the channel
            let res = py.detach(|| {
                crate::RUNTIME.block_on(async {
                    let fut = async {
                        let mut guard = rx.lock().await;
                        guard.recv().await
                    };
                    tokio::time::timeout(actual_wait, fut).await
                })
            });

            match res {
                Ok(Some(msg)) => return Ok(message_to_py(py, msg)),
                Ok(None) => return Ok(py.None()),
                Err(_) => {
                    // Check if it's the end of user's requested timeout
                    if let Some(t) = timeout_dur {
                        if start.elapsed() >= t {
                            return Ok(py.None());
                        }
                    }
                    // Else loop back up to check_signals
                }
            }
        }
    }

    /// Selectively receive a message matching a Python predicate (Blocking).
    /// Releases the GIL while waiting. Checks for Python signals cleanly.
    #[pyo3(signature = (matcher, timeout=None))]
    fn selective_recv(
        &self,
        py: Python<'_>,
        matcher: Py<PyAny>,
        timeout: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let rx = self.inner.clone();
        let start = std::time::Instant::now();
        let timeout_dur = timeout.map(std::time::Duration::from_secs_f64);

        loop {
            py.check_signals()?;

            let wait_time = std::time::Duration::from_millis(100);
            let actual_wait = if let Some(t) = timeout_dur {
                let elapsed = start.elapsed();
                if elapsed >= t {
                    return Ok(py.None());
                }
                std::cmp::min(wait_time, t.saturating_sub(elapsed))
            } else {
                wait_time
            };

            let res = py.detach(|| {
                crate::RUNTIME.block_on(async {
                    let fut = async {
                        let mut guard = rx.lock().await;
                        guard
                            .selective_recv(|msg| {
                                Python::attach(|py| {
                                    Ok::<bool, PyErr>(run_python_matcher(py, matcher.bind(py), msg))
                                })
                                .unwrap_or(false)
                            })
                            .await
                    };
                    tokio::time::timeout(actual_wait, fut).await
                })
            });

            match res {
                Ok(Some(msg)) => return Ok(message_to_py(py, msg)),
                Ok(None) => return Ok(py.None()),
                Err(_) => {
                    if let Some(t) = timeout_dur {
                        if start.elapsed() >= t {
                            return Ok(py.None());
                        }
                    }
                }
            }
        }
    }
}
