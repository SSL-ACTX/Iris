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
use std::{any::Any, panic::{catch_unwind, AssertUnwindSafe}};

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
use crate::py::jit::codegen::{
    compile_jit,
    compile_jit_quantum,
    execute_registered_jit,
    lookup_jit,
    register_jit,
    register_quantum_jit,
    register_named_jit,
};

static JIT_LOG_OVERRIDE: AtomicI8 = AtomicI8::new(-1); // -1 env, 0 off, 1 on
static JIT_LOG_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_OVERRIDE: AtomicI8 = AtomicI8::new(-1); // -1 env, 0 off, 1 on
static JIT_QUANTUM_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();

fn jit_log_env_var() -> &'static RwLock<String> {
    JIT_LOG_ENV_VAR.get_or_init(|| RwLock::new("IRIS_JIT_LOG".to_string()))
}

fn jit_quantum_env_var() -> &'static RwLock<String> {
    JIT_QUANTUM_ENV_VAR.get_or_init(|| RwLock::new("IRIS_JIT_QUANTUM".to_string()))
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

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[cfg(feature = "pyo3")]
fn execute_registered_jit_guarded(
    py: Python,
    func_key: usize,
    args: &PyTuple,
) -> Option<PyResult<PyObject>> {
    match catch_unwind(AssertUnwindSafe(|| execute_registered_jit(py, func_key, args))) {
        Ok(res) => res,
        Err(payload) => {
            let msg = panic_payload_to_string(payload);
            Some(Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "jit panic: {}",
                msg
            ))))
        }
    }
}

pub(crate) fn quantum_speculation_enabled() -> bool {
    match JIT_QUANTUM_OVERRIDE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let env_name = jit_quantum_env_var().read().unwrap().clone();
            std::env::var(env_name)
                .ok()
                .map(|v| parse_bool_env(&v))
                .unwrap_or(false)
        }
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
    m.add_function(pyo3::wrap_pyfunction!(call_jit_step_loop_f64, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_jit_logging, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(is_jit_logging_enabled, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_speculation, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(is_quantum_speculation_enabled, m)?)?;
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

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_speculation(enabled: Option<bool>, env_var: Option<String>) -> PyResult<bool> {
    if let Some(name) = env_var {
        *jit_quantum_env_var().write().unwrap() = name;
    }
    match enabled {
        Some(true) => JIT_QUANTUM_OVERRIDE.store(1, Ordering::Relaxed),
        Some(false) => JIT_QUANTUM_OVERRIDE.store(0, Ordering::Relaxed),
        None => JIT_QUANTUM_OVERRIDE.store(-1, Ordering::Relaxed),
    }
    Ok(quantum_speculation_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn is_quantum_speculation_enabled() -> PyResult<bool> {
    Ok(quantum_speculation_enabled())
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
                let key = func.as_ptr() as usize;
                let func_name = Python::with_gil(|py| {
                    func.as_ref(py)
                        .getattr("__name__")
                        .ok()
                        .and_then(|n| n.extract::<String>().ok())
                });
                if quantum_speculation_enabled() {
                    let entries = match catch_unwind(AssertUnwindSafe(|| compile_jit_quantum(&expr, &args))) {
                        Ok(entries) => entries,
                        Err(payload) => {
                            let msg = panic_payload_to_string(payload);
                            jit_log(|| format!("[Iris][jit] panic while compiling quantum variants '{}': {}", expr, msg));
                            Vec::new()
                        }
                    };
                    if !entries.is_empty() {
                        if let Some(name) = func_name.as_deref() {
                            register_named_jit(name, entries[0].clone());
                        }
                        register_quantum_jit(key, entries);
                        jit_log(|| format!("[Iris][jit] compiled quantum JIT variants for ptr={}", key));
                    } else {
                        jit_log(|| format!("[Iris][jit] failed to compile quantum variants: {}", expr));
                    }
                } else {
                    let maybe_entry = match catch_unwind(AssertUnwindSafe(|| compile_jit(&expr, &args))) {
                        Ok(entry) => entry,
                        Err(payload) => {
                            let msg = panic_payload_to_string(payload);
                            jit_log(|| format!("[Iris][jit] panic while compiling expr '{}': {}", expr, msg));
                            None
                        }
                    };
                    if let Some(entry) = maybe_entry {
                        if let Some(name) = func_name.as_deref() {
                            register_named_jit(name, entry.clone());
                        }
                        register_jit(key, entry);
                        jit_log(|| format!("[Iris][jit] compiled JIT for function ptr={}", key));
                    } else {
                        jit_log(|| format!("[Iris][jit] failed to compile expr: {}", expr));
                    }
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
    if let Some(res) = execute_registered_jit_guarded(py, key, args) {
        if let Ok(obj) = res {
            return Ok(obj);
        }
        if let Err(err) = &res {
            jit_log(|| format!("[Iris][jit] guarded execution failed in offload_call; falling back: {}", err));
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
    if let Some(res) = execute_registered_jit_guarded(py, key, args) {
        return res;
    }
    Err(pyo3::exceptions::PyRuntimeError::new_err("no JIT entry found"))
}

/// Execute a registered scalar 2-arg JIT step function in a Rust loop.
///
/// This is used by Python wrappers for recurrence kernels to avoid Python↔Rust
/// crossing overhead on each iteration.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit_step_loop_f64(func: PyObject, seed: f64, count: usize) -> PyResult<f64> {
    let key = func.as_ptr() as usize;
    let entry = lookup_jit(key).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("no JIT entry found")
    })?;

    if entry.arg_count != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "step loop requires a 2-argument scalar JIT entry",
        ));
    }

    let run = || {
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let mut values = [seed, 0.0_f64];
        let mut state = seed;
        for i in 0..count {
            values[0] = state;
            values[1] = i as f64;
            state = f(values.as_ptr());
        }
        state
    };

    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(out) => Ok(out),
        Err(payload) => {
            let msg = panic_payload_to_string(payload);
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "jit panic: {}",
                msg
            )))
        }
    }
}

// ------- tests ------------------------------------------------------------

#[cfg(test)]
mod tests;
