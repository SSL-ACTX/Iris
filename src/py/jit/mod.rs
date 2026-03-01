// src/py/jit/mod.rs
//! Python JIT/offload support for the Iris runtime.
//!
//! This module provides the low-level bindings that power the `@iris.offload`
//! decorator in Python.  Initially the implementation is a no-op stub, but it
//! gives us a dedicated home for future JIT or actor‑routing logic.

#![allow(non_local_definitions)]

use std::sync::Arc;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyTuple};

// Offload actor pool ---------------------------------------------------------

/// A task describing a Python call to execute.
struct OffloadTask {
    func: Py<PyAny>,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    resp: std::sync::mpsc::Sender<Result<PyObject, PyErr>>,
}

struct OffloadPool {
    sender: crossbeam_channel::Sender<OffloadTask>,
}

impl OffloadPool {
    fn new(size: usize) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<OffloadTask>();

        for _ in 0..size {
            let rx = rx.clone();
            std::thread::spawn(move || {
                loop {
                    match rx.recv() {
                        Ok(task) => {
                            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                break;
                            }
                            Python::with_gil(|py| {
                                let func = task.func.as_ref(py);
                                let args = task.args.as_ref(py);
                                let kwargs = task.kwargs.as_ref().map(|k: &Py<PyDict>| k.as_ref(py));
                                let result = func.call(args, kwargs).map(|obj| obj.into_py(py));
                                let _ = task.resp.send(result.map_err(|e| e));
                            });
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        OffloadPool { sender: tx }
    }
}

// shared singleton
static OFFLOAD_POOL: once_cell::sync::OnceCell<Arc<OffloadPool>> = once_cell::sync::OnceCell::new();

fn get_offload_pool() -> Arc<OffloadPool> {
    OFFLOAD_POOL
        .get_or_init(|| Arc::new(OffloadPool::new(num_cpus::get())))
        .clone()
}

// Python bindings -----------------------------------------------------------

/// Initialize the Python submodule (called from `wrappers.populate_module`).
#[cfg(feature = "pyo3")]
pub(crate) fn init_py(m: &PyModule) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(register_offload, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(offload_call, m)?)?;
    Ok(())
}

/// Register a Python function for offloading.
///
/// This is the Rust-side hook invoked by the decorator.  For now we simply
/// return the original callable back to Python unmodified, but we log for
/// inspection and ensure the actor pool is initialized when strategy=actor.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn register_offload(
    func: PyObject,
    strategy: Option<String>,
    return_type: Option<String>,
) -> PyResult<PyObject> {
    if let Some(ref s) = strategy {
        if s == "actor" {
            // ensure pool exists
            let _ = get_offload_pool();
        }
    }
    eprintln!(
        "[Iris][jit] register_offload called strategy={:?} return_type={:?}",
        strategy, return_type
    );
    Ok(func)
}

/// Execute a Python callable on the offload actor pool, blocking until result.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn offload_call(
    py: Python,
    func: PyObject,
    args: &PyTuple,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    // ensure pool exists
    let pool = get_offload_pool();

    let (tx, rx) = std::sync::mpsc::channel();
    let task = OffloadTask {
        func: func.into_py(py),
        args: args.into_py(py),
        kwargs: kwargs.map(|d: &PyDict| d.into_py(py)),
        resp: tx,
    };

    pool.sender
        .send(task)
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("offload queue closed"))?;

    // wait for response without holding GIL; blocking recv on std channel is safe
    let result = py.allow_threads(move || {
        match rx.recv() {
            Ok(res) => res,
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "offload task canceled",
            )),
        }
    });

    result
}
