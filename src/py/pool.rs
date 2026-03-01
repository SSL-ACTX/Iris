// src/py/pool.rs
//! GIL release and worker pool helpers for Python actors.
#![allow(non_local_definitions)]

use crossbeam_channel as cb_channel;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use crate::Runtime;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use pyo3::PyObject;
use pyo3::types::PyBytes;

/// Task variants sent to dedicated or pooled GIL workers.
#[cfg(feature = "pyo3")]
pub(crate) enum PoolTask {
    Execute {
        behavior: Arc<parking_lot::RwLock<PyObject>>,
        bytes: bytes::Bytes,
    },
    HotSwap {
        behavior: Arc<parking_lot::RwLock<PyObject>>,
        ptr: usize,
    },
}

#[cfg(feature = "pyo3")]
pub(crate) struct GilPool {
    pub(crate) sender: cb_channel::Sender<PoolTask>,
}

#[cfg(feature = "pyo3")]
pub(crate) static GIL_WORKER_POOL: OnceLock<Arc<GilPool>> = OnceLock::new();

#[cfg(feature = "pyo3")]
impl GilPool {
    fn new(size: usize) -> Self {
        let (tx, rx) = cb_channel::unbounded::<PoolTask>();
        for _ in 0..size {
            let rx = rx.clone();
            std::thread::spawn(move || {
                loop {
                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                        break;
                    }
                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(task) => match task {
                            PoolTask::Execute { behavior, bytes } => {
                                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                    break;
                                }
                                Python::with_gil(|py| {
                                    let guard = behavior.read();
                                    let cb = guard.as_ref(py);
                                    let pybytes = PyBytes::new(py, &bytes);
                                    if let Err(e) = cb.call1((pybytes,)) {
                                        eprintln!("[Iris] Python actor exception: {}", e);
                                        e.print(py);
                                    }
                                });
                            }
                            PoolTask::HotSwap { behavior, ptr } => {
                                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                    break;
                                }
                                Python::with_gil(|py| unsafe {
                                    let new_obj = PyObject::from_owned_ptr(
                                        py,
                                        ptr as *mut pyo3::ffi::PyObject,
                                    );
                                    *behavior.write() = new_obj;
                                });
                            }
                        },
                        Err(cb_channel::RecvTimeoutError::Timeout) => continue,
                        Err(cb_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            });
        }
        GilPool { sender: tx }
    }
}

/// Create a channel to offload Python callback execution when `release` is true.
/// Returns `None` if the release flag is false or we are falling back to the
/// shared pool. On strict mode, exceeding the per-runtime thread limit
/// produces an error.
#[cfg(feature = "pyo3")]
pub(crate) fn make_release_gil_channel(
    rt: &Runtime,
    release: bool,
    behavior: Arc<parking_lot::RwLock<PyObject>>,
) -> PyResult<Option<cb_channel::Sender<PoolTask>>> {
    if !release {
        return Ok(None);
    }

    static RELEASE_GIL_THREADS: AtomicUsize = AtomicUsize::new(0);
    const DEFAULT_MAX_RELEASE_GIL_THREADS: usize = 256;
    const DEFAULT_GIL_POOL_SIZE: usize = 8;

    let (max_threads, pool_size) = rt.get_release_gil_limits();
    let strict = rt.is_release_gil_strict();

    let prev = RELEASE_GIL_THREADS.fetch_add(1, Ordering::SeqCst);
    if prev >= max_threads {
        RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
        if strict {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "release_gil thread limit exceeded",
            ));
        }
        let _ = GIL_WORKER_POOL
            .get_or_init(|| Arc::new(GilPool::new(pool_size)))
            .clone();
        return Ok(None);
    }

    let (tx, rx) = cb_channel::unbounded::<PoolTask>();
    let _b_thread = behavior.clone();
    std::thread::spawn(move || {
        loop {
            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
                break;
            }
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(task) => match task {
                    PoolTask::Execute { behavior, bytes } => {
                        if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                            continue;
                        }
                        Python::with_gil(|py| {
                            let guard = behavior.read();
                            let cb = guard.as_ref(py);
                            let pybytes = PyBytes::new(py, &bytes);
                            if let Err(e) = cb.call1((pybytes,)) {
                                eprintln!("[Iris] Python actor exception: {}", e);
                                e.print(py);
                            }
                        });
                    }
                    PoolTask::HotSwap { behavior, ptr } => {
                        if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                            continue;
                        }
                        Python::with_gil(|py| unsafe {
                            let new_obj = PyObject::from_owned_ptr(
                                py,
                                ptr as *mut pyo3::ffi::PyObject,
                            );
                            *behavior.write() = new_obj;
                        });
                    }
                },
                Err(cb_channel::RecvTimeoutError::Timeout) => continue,
                Err(cb_channel::RecvTimeoutError::Disconnected) => {
                    RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
                    break;
                }
            }
        }
    });
    Ok(Some(tx))
}
