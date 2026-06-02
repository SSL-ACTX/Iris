// src/py/pool.rs
//! GIL release and worker pool helpers for Python actors.
#![allow(non_local_definitions)]

use crossbeam_channel as cb_channel;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use crate::Runtime;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Task variants sent to dedicated or pooled GIL workers.
pub(crate) enum PoolTask {
    Execute {
        behavior: Arc<parking_lot::RwLock<Py<PyAny>>>,
        bytes: bytes::Bytes,
        pid_holder: Arc<AtomicU64>,
        rt: Arc<Runtime>,
    },
    HotSwap {
        behavior: Arc<parking_lot::RwLock<Py<PyAny>>>,
        ptr: usize,
    },
}

/// Lightweight task variants for dedicated per-actor GIL threads.
pub(crate) enum DedicatedPoolTask {
    Execute(bytes::Bytes),
    HotSwap(usize),
}

pub(crate) struct GilPool {
    pub(crate) sender: cb_channel::Sender<PoolTask>,
}

pub(crate) static GIL_WORKER_POOL: OnceLock<Arc<GilPool>> = OnceLock::new();

impl GilPool {
    fn new(size: usize) -> Self {
        let (tx, rx) = cb_channel::unbounded::<PoolTask>();
        for _ in 0..size {
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                    break;
                }
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(task) => match task {
                        PoolTask::Execute {
                            behavior,
                            bytes,
                            pid_holder,
                            rt,
                        } => {
                            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                break;
                            }
                            let success = crate::py::utils::run_python_callback(|py| {
                                let guard = behavior.read();
                                let cb = guard.bind(py);
                                let pybytes = PyBytes::new(py, &bytes);
                                cb.call1((pybytes,)).map(|_| ())
                            });
                            if !success {
                                let pid = pid_holder.load(Ordering::SeqCst);
                                if pid != 0 {
                                    rt.stop(pid);
                                }
                            }
                        }
                        PoolTask::HotSwap { behavior, ptr } => {
                            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                break;
                            }
                            Python::attach(|py| unsafe {
                                let new_obj =
                                    Bound::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject)
                                        .unbind();
                                *behavior.write() = new_obj;
                                Ok::<(), PyErr>(())
                            })
                            .unwrap();
                        }
                    },
                    Err(cb_channel::RecvTimeoutError::Timeout) => continue,
                    Err(cb_channel::RecvTimeoutError::Disconnected) => break,
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
pub(crate) fn make_release_gil_channel(
    rt: &Runtime,
    release: bool,
    behavior: Arc<parking_lot::RwLock<Py<PyAny>>>,
    pid_holder: Arc<AtomicU64>,
    rt_arc: Arc<Runtime>,
) -> PyResult<Option<cb_channel::Sender<DedicatedPoolTask>>> {
    if !release {
        return Ok(None);
    }

    static RELEASE_GIL_THREADS: AtomicUsize = AtomicUsize::new(0);

    let (max_threads, pool_size) = rt.get_release_gil_limits();
    let strict = rt.is_release_gil_strict();

    // Explicit pooled mode: no dedicated per-actor GIL threads.
    // This is the preferred mode for high-throughput short-lived actors.
    if max_threads == 0 {
        let _ = GIL_WORKER_POOL
            .get_or_init(|| Arc::new(GilPool::new(pool_size)))
            .clone();
        return Ok(None);
    }

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

    let (tx, rx) = cb_channel::unbounded::<DedicatedPoolTask>();
    let _b_thread = behavior.clone();
    let pid_holder_thread = pid_holder.clone();
    let rt_thread = rt_arc.clone();
    std::thread::spawn(move || loop {
        if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
            RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
            break;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(task) => match task {
                DedicatedPoolTask::Execute(bytes) => {
                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                        continue;
                    }
                    let success = crate::py::utils::run_python_callback(|py| {
                        let guard = _b_thread.read();
                        let cb = guard.bind(py);
                        let pybytes = PyBytes::new(py, &bytes);
                        cb.call1((pybytes,)).map(|_| ())
                    });
                    if !success {
                        let pid = pid_holder_thread.load(Ordering::SeqCst);
                        if pid != 0 {
                            rt_thread.stop(pid);
                        }
                    }
                }
                DedicatedPoolTask::HotSwap(ptr) => {
                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                        continue;
                    }
                    Python::attach(|py| unsafe {
                        let new_obj =
                            Bound::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject).unbind();
                        *_b_thread.write() = new_obj;
                        Ok::<(), PyErr>(())
                    })
                    .unwrap();
                }
            },
            Err(cb_channel::RecvTimeoutError::Timeout) => continue,
            Err(cb_channel::RecvTimeoutError::Disconnected) => {
                RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
                break;
            }
        }
    });
    Ok(Some(tx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_threads_zero_forces_shared_pool_even_in_strict_mode() {
        let rt = Runtime::new();
        rt.set_release_gil_limits(0, 1);
        rt.set_release_gil_strict(true);

        Python::attach(|py| {
            let behavior = Arc::new(parking_lot::RwLock::new(py.None()));
            let ch = make_release_gil_channel(
                &rt,
                true,
                behavior,
                Arc::new(AtomicU64::new(0)),
                Arc::new(rt.clone()),
            )
            .expect("max_threads=0 should force shared pool mode without strict error");
            assert!(ch.is_none());
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
}
