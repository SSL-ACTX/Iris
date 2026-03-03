// src/py/jit/mod.rs
//! Python JIT/offload support for the Iris runtime.
//!
//! This module provides the low-level bindings that power the `@iris.offload`
//! decorator in Python.  It also exposes the public API surface used by the
//! various JIT submodules (parser, codegen, heuristics) which live in their
//! own files for clarity.

#![allow(non_local_definitions)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyTuple};

use cranelift::prelude::*;
use cranelift_native;
use pyo3::AsPyPointer;

pub(crate) mod parser;
pub(crate) mod codegen;
pub(crate) mod heuristics;

// re-export helpers for convenience within this module
use crate::py::jit::codegen::{compile_jit, register_jit, lookup_jit, execute_jit_func};

static JIT_LOG_OVERRIDE: AtomicI8 = AtomicI8::new(-1); // -1 env, 0 off, 1 on
static JIT_LOG_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();

fn jit_log_env_var() -> &'static RwLock<String> {
    JIT_LOG_ENV_VAR.get_or_init(|| RwLock::new("IRIS_JIT_LOG".to_string()))
}

fn parse_bool_env(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "debug"
    )
}

pub(crate) fn jit_logging_enabled() -> bool {
    match JIT_LOG_OVERRIDE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let env_name = jit_log_env_var().read().unwrap().clone();
            std::env::var(env_name)
                .ok()
                .map(|v| parse_bool_env(&v))
                .unwrap_or(false)
        }
    }
}

pub(crate) fn jit_log<F>(msg: F)
where
    F: FnOnce() -> String,
{
    if jit_logging_enabled() {
        eprintln!("{}", msg());
    }
}

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
                                let kwargs = task
                                    .kwargs
                                    .as_ref()
                                    .map(|k: &Py<PyDict>| k.as_ref(py));
                                let result = func.call(args, kwargs).map(|obj| obj.into_py(py));
                                let _ = task.resp.send(result);
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
static OFFLOAD_POOL: once_cell::sync::OnceCell<Arc<OffloadPool>> =
    once_cell::sync::OnceCell::new();

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
    m.add_function(pyo3::wrap_pyfunction!(call_jit, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_jit_logging, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(is_jit_logging_enabled, m)?)?;
    Ok(())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_jit_logging(enabled: Option<bool>, env_var: Option<String>) -> PyResult<bool> {
    if let Some(name) = env_var {
        *jit_log_env_var().write().unwrap() = name;
    }
    match enabled {
        Some(true) => JIT_LOG_OVERRIDE.store(1, Ordering::Relaxed),
        Some(false) => JIT_LOG_OVERRIDE.store(0, Ordering::Relaxed),
        None => JIT_LOG_OVERRIDE.store(-1, Ordering::Relaxed),
    }
    Ok(jit_logging_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn is_jit_logging_enabled() -> PyResult<bool> {
    Ok(jit_logging_enabled())
}

/// Register a Python function for offloading.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn register_offload(
    func: PyObject,
    strategy: Option<String>,
    return_type: Option<String>,
    source_expr: Option<String>,
    arg_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    if let Some(ref s) = strategy {
        if s == "actor" {
            let _ = get_offload_pool();
        } else if s == "jit" {
            if let (Some(expr), Some(args)) = (source_expr.clone(), arg_names.clone()) {
                if let Some(entry) = compile_jit(&expr, &args) {
                    let key = func.as_ptr() as usize;
                    register_jit(key, entry);
                    jit_log(|| format!("[Iris][jit] compiled JIT for function ptr={}", key));
                } else {
                    jit_log(|| format!("[Iris][jit] failed to compile expr: {}", expr));
                }
            }
        }
    }
    jit_log(|| {
        format!(
            "[Iris][jit] register_offload called strategy={:?} return_type={:?} source={:?} args={:?}",
            strategy, return_type, source_expr, arg_names
        )
    });
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
    let key = func.as_ptr() as usize;
    if let Some(entry) = lookup_jit(key) {
        if let Ok(res) = execute_jit_func(py, &entry, args) {
            return Ok(res);
        }
    }

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

    let result = py.allow_threads(move || match rx.recv() {
        Ok(res) => res,
        Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "offload task canceled",
        )),
    });

    result
}

/// Directly invoke the JIT-compiled version of a Python function.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit(
    py: Python,
    func: PyObject,
    args: &PyTuple,
    _kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let key = func.as_ptr() as usize;
    if let Some(entry) = lookup_jit(key) {
        return execute_jit_func(py, &entry, args);
    }
    Err(pyo3::exceptions::PyRuntimeError::new_err("no JIT entry found"))
}

// ------- tests ------------------------------------------------------------

#[cfg(test)]
mod tests;
