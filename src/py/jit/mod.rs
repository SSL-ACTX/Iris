// src/py/jit/mod.rs
//! Python JIT/offload support for the Iris runtime.
//!
//! This module provides the low-level bindings that power the `@iris.offload`
//! decorator in Python.  It also exposes the public API surface used by the
//! various JIT submodules (parser, codegen, heuristics) which live in their
//! own files for clarity.

#![allow(non_local_definitions)]

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyTuple};

pub(crate) mod codegen;
pub(crate) mod config;
pub(crate) mod heuristics;
pub(crate) mod parser;
pub(crate) mod quantum;
pub(crate) mod simd;

pub(crate) use crate::py::jit::config::{
    jit_log, jit_logging_enabled, quantum_compile_budget_ns, quantum_compile_window_ns,
    quantum_cooldown_base_ns, quantum_cooldown_max_ns, quantum_log_threshold_ns,
    quantum_speculation_enabled, quantum_speculation_threshold_ns, quantum_stability_min_runs,
    quantum_stability_min_score, set_jit_logging_env_var, set_jit_logging_override,
    set_quantum_speculation_env_var, set_quantum_speculation_override,
};

#[cfg(test)]
pub(crate) use crate::py::jit::config::{jit_log_clear_hook, jit_log_hook};
use crate::py::jit::config::{
    jit_quantum_compile_budget_env_var, jit_quantum_compile_window_env_var,
    jit_quantum_cooldown_base_env_var, jit_quantum_cooldown_max_env_var, jit_quantum_log_env_var,
    jit_quantum_speculation_env_var, now_ns, panic_payload_to_string, parse_return_type,
};

pub(crate) use crate::py::jit::quantum::{
    maybe_rearm_quantum_compile, quantum_compile_may_run, record_quantum_compile_attempt,
    register_quantum_rearm_plan,
};

#[cfg(test)]
pub(crate) use crate::py::jit::quantum::{
    clear_quantum_rearm_plan_for_test, register_quantum_rearm_plan_for_test,
    reset_quantum_control_state,
};

// re-export helpers for convenience within this module
use crate::py::jit::codegen::{
    compile_jit_quantum, compile_jit_quantum_variant, compile_jit_with_return_type,
    execute_registered_jit, lookup_jit, quantum_has_seed_hint, quantum_profile_snapshot,
    quantum_seed_preferred_index, register_jit, register_named_jit, register_quantum_jit,
    seed_quantum_profile as seed_quantum_profile_state, QuantumProfileSeed,
};

// Offload actor pool ---------------------------------------------------------

#[cfg(feature = "pyo3")]
fn execute_registered_jit_guarded(
    py: Python,
    func_key: usize,
    args: &PyTuple,
) -> Option<PyResult<PyObject>> {
    match catch_unwind(AssertUnwindSafe(|| {
        execute_registered_jit(py, func_key, args)
    })) {
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
                while let Ok(task) = rx.recv() {
                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                        break;
                    }
                    Python::with_gil(|py| {
                        let func = task.func.as_ref(py);
                        let args = task.args.as_ref(py);
                        let kwargs = task.kwargs.as_ref().map(|k: &Py<PyDict>| k.as_ref(py));
                        let result = func.call(args, kwargs).map(|obj| obj.into_py(py));
                        let _ = task.resp.send(result);
                    });
                }
            });
        }

        OffloadPool { sender: tx }
    }
}

// shared singleton
static OFFLOAD_POOL: once_cell::sync::OnceCell<Arc<OffloadPool>> = once_cell::sync::OnceCell::new();
static OFFLOAD_STRATEGY: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<usize, String>>> =
    once_cell::sync::OnceCell::new();

fn get_offload_pool() -> Arc<OffloadPool> {
    OFFLOAD_POOL
        .get_or_init(|| Arc::new(OffloadPool::new(num_cpus::get())))
        .clone()
}

fn set_offload_strategy(func_key: usize, strategy: &str) {
    let map = OFFLOAD_STRATEGY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    if let Ok(mut guard) = map.lock() {
        guard.insert(func_key, strategy.to_ascii_lowercase());
    }
}

fn get_offload_strategy(func_key: usize) -> Option<String> {
    OFFLOAD_STRATEGY.get().and_then(|map| {
        map.lock()
            .ok()
            .and_then(|guard| guard.get(&func_key).cloned())
    })
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
    m.add_function(pyo3::wrap_pyfunction!(
        configure_quantum_speculation_threshold,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        get_quantum_speculation_threshold,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        set_quantum_speculation_threshold,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_log_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_log_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(set_quantum_log_threshold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_compile_budget, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_compile_budget, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_quantum_cooldown, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_cooldown, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(get_quantum_profile, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(seed_quantum_profile, m)?)?;
    Ok(())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_jit_logging(enabled: Option<bool>, env_var: Option<String>) -> PyResult<bool> {
    if let Some(name) = env_var {
        set_jit_logging_env_var(name);
    }
    set_jit_logging_override(enabled);
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
        set_quantum_speculation_env_var(name);
    }
    set_quantum_speculation_override(enabled);
    Ok(quantum_speculation_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn is_quantum_speculation_enabled() -> PyResult<bool> {
    Ok(quantum_speculation_enabled())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_speculation_threshold(
    threshold_ns: Option<u64>,
    env_var: Option<String>,
) -> PyResult<u64> {
    if let Some(name) = env_var {
        *jit_quantum_speculation_env_var().write().unwrap() = name;
    }
    if let Some(ns) = threshold_ns {
        let env_name = jit_quantum_speculation_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok(quantum_speculation_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_speculation_threshold() -> PyResult<u64> {
    Ok(quantum_speculation_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn set_quantum_speculation_threshold(
    threshold_ns: Option<u64>,
    env_var: Option<String>,
) -> PyResult<u64> {
    configure_quantum_speculation_threshold(threshold_ns, env_var)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_log_threshold(
    threshold_ns: Option<u64>,
    env_var: Option<String>,
) -> PyResult<u64> {
    if let Some(name) = env_var {
        *jit_quantum_log_env_var().write().unwrap() = name;
    }
    if let Some(ns) = threshold_ns {
        let env_name = jit_quantum_log_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok(quantum_log_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_log_threshold() -> PyResult<u64> {
    Ok(quantum_log_threshold_ns())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn set_quantum_log_threshold(threshold_ns: Option<u64>, env_var: Option<String>) -> PyResult<u64> {
    configure_quantum_log_threshold(threshold_ns, env_var)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_compile_budget(
    budget_ns: Option<u64>,
    window_ns: Option<u64>,
    budget_env_var: Option<String>,
    window_env_var: Option<String>,
) -> PyResult<(u64, u64)> {
    if let Some(name) = budget_env_var {
        *jit_quantum_compile_budget_env_var().write().unwrap() = name;
    }
    if let Some(name) = window_env_var {
        *jit_quantum_compile_window_env_var().write().unwrap() = name;
    }
    if let Some(ns) = budget_ns {
        let env_name = jit_quantum_compile_budget_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    if let Some(ns) = window_ns {
        let env_name = jit_quantum_compile_window_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok((quantum_compile_budget_ns(), quantum_compile_window_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_compile_budget() -> PyResult<(u64, u64)> {
    Ok((quantum_compile_budget_ns(), quantum_compile_window_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_cooldown(
    base_ns: Option<u64>,
    max_ns: Option<u64>,
    base_env_var: Option<String>,
    max_env_var: Option<String>,
) -> PyResult<(u64, u64)> {
    if let Some(name) = base_env_var {
        *jit_quantum_cooldown_base_env_var().write().unwrap() = name;
    }
    if let Some(name) = max_env_var {
        *jit_quantum_cooldown_max_env_var().write().unwrap() = name;
    }
    if let Some(ns) = base_ns {
        let env_name = jit_quantum_cooldown_base_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    if let Some(ns) = max_ns {
        let env_name = jit_quantum_cooldown_max_env_var().read().unwrap().clone();
        std::env::set_var(env_name, ns.to_string());
    }
    Ok((quantum_cooldown_base_ns(), quantum_cooldown_max_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_cooldown() -> PyResult<(u64, u64)> {
    Ok((quantum_cooldown_base_ns(), quantum_cooldown_max_ns()))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_profile(func: PyObject) -> PyResult<Vec<(usize, f64, u64, u64)>> {
    let key = func.as_ptr() as usize;
    let points = quantum_profile_snapshot(key).unwrap_or_default();
    Ok(points
        .into_iter()
        .map(|point| (point.index, point.ewma_ns, point.runs, point.failures))
        .collect())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn seed_quantum_profile(func: PyObject, rows: Vec<(usize, f64, u64, u64)>) -> PyResult<bool> {
    let key = func.as_ptr() as usize;
    let seeds = rows
        .into_iter()
        .map(|(index, ewma_ns, runs, failures)| QuantumProfileSeed {
            index,
            ewma_ns,
            runs,
            failures,
        })
        .collect::<Vec<_>>();
    Ok(seed_quantum_profile_state(key, &seeds))
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
    let key = func.as_ptr() as usize;
    if let Some(ref s) = strategy {
        set_offload_strategy(key, s);
        if s == "actor" {
            let _ = get_offload_pool();
        } else if s == "jit" {
            if let (Some(expr), Some(args)) = (source_expr.clone(), arg_names.clone()) {
                let func_name = Python::with_gil(|py| {
                    func.as_ref(py)
                        .getattr("__name__")
                        .ok()
                        .and_then(|n| n.extract::<String>().ok())
                });
                let return_type = parse_return_type(return_type.as_deref());
                register_quantum_rearm_plan(key, &expr, &args, return_type);
                let warm_seed_hint = quantum_has_seed_hint(key);

                let compile_single = || {
                    let maybe_entry = match catch_unwind(AssertUnwindSafe(|| {
                        compile_jit_with_return_type(&expr, &args, return_type)
                    })) {
                        Ok(entry) => entry,
                        Err(payload) => {
                            let msg = panic_payload_to_string(payload);
                            jit_log(|| {
                                format!(
                                    "[Iris][jit] panic while compiling expr '{}': {}",
                                    expr, msg
                                )
                            });
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
                };

                let compile_single_quantum = || {
                    let preferred_variant = quantum_seed_preferred_index(key).unwrap_or(0);
                    let started = Instant::now();
                    let maybe_entry = match catch_unwind(AssertUnwindSafe(|| {
                        compile_jit_quantum_variant(&expr, &args, return_type, preferred_variant)
                    })) {
                        Ok(entry) => entry,
                        Err(payload) => {
                            let msg = panic_payload_to_string(payload);
                            jit_log(|| {
                                format!(
                                    "[Iris][jit] panic while compiling warm quantum expr '{}': {}",
                                    expr, msg
                                )
                            });
                            None
                        }
                    };

                    let elapsed = started.elapsed().as_nanos() as u64;
                    let success = maybe_entry.is_some();
                    record_quantum_compile_attempt(key, now_ns(), elapsed, success);

                    if let Some(entry) = maybe_entry {
                        if let Some(name) = func_name.as_deref() {
                            register_named_jit(name, entry.clone());
                        }
                        register_jit(key, entry.clone());
                        register_quantum_jit(key, vec![entry]);
                        jit_log(|| {
                            format!(
                                "[Iris][jit] warm-seeded single-variant compile for ptr={} variant={}",
                                key, preferred_variant
                            )
                        });
                    } else {
                        jit_log(|| {
                            format!(
                                "[Iris][jit] failed warm-seeded single-variant compile: {}",
                                expr
                            )
                        });
                        compile_single();
                    }
                };

                let compile_budget_gated_quantum = || {
                    let started = Instant::now();
                    let entries = match catch_unwind(AssertUnwindSafe(|| {
                        let mut partial = Vec::new();
                        if let Some(auto) =
                            compile_jit_quantum_variant(&expr, &args, return_type, 0)
                        {
                            partial.push(auto);
                        }
                        if let Some(scalar) =
                            compile_jit_quantum_variant(&expr, &args, return_type, 1)
                        {
                            partial.push(scalar);
                        }
                        partial
                    })) {
                        Ok(entries) => entries,
                        Err(payload) => {
                            let msg = panic_payload_to_string(payload);
                            jit_log(|| {
                                format!("[Iris][jit] panic while compiling reduced quantum variants '{}': {}", expr, msg)
                            });
                            Vec::new()
                        }
                    };

                    let now = now_ns();
                    let elapsed = started.elapsed().as_nanos() as u64;
                    let success = !entries.is_empty();
                    record_quantum_compile_attempt(key, now, elapsed, success);

                    if success {
                        if let Some(name) = func_name.as_deref() {
                            register_named_jit(name, entries[0].clone());
                        }
                        register_quantum_jit(key, entries);
                        jit_log(|| {
                            format!(
                                "[Iris][jit] compiled reduced quantum variants for ptr={}",
                                key
                            )
                        });
                    } else {
                        jit_log(|| format!("[Iris][jit] failed reduced quantum compile: {}", expr));
                        compile_single();
                    }
                };

                if quantum_speculation_enabled() {
                    if warm_seed_hint {
                        compile_single_quantum();
                    } else {
                        let now = now_ns();
                        if quantum_compile_may_run(key, now) {
                            let started = Instant::now();
                            let entries = match catch_unwind(AssertUnwindSafe(|| {
                                compile_jit_quantum(&expr, &args, return_type)
                            })) {
                                Ok(entries) => entries,
                                Err(payload) => {
                                    let msg = panic_payload_to_string(payload);
                                    jit_log(|| {
                                        format!("[Iris][jit] panic while compiling quantum variants '{}': {}", expr, msg)
                                    });
                                    Vec::new()
                                }
                            };
                            let elapsed = started.elapsed().as_nanos() as u64;
                            let success = !entries.is_empty();
                            record_quantum_compile_attempt(key, now, elapsed, success);

                            if success {
                                if let Some(name) = func_name.as_deref() {
                                    register_named_jit(name, entries[0].clone());
                                }
                                register_quantum_jit(key, entries);
                                jit_log(|| {
                                    format!(
                                        "[Iris][jit] compiled quantum JIT variants for ptr={}",
                                        key
                                    )
                                });
                            } else {
                                jit_log(|| {
                                    format!(
                                        "[Iris][jit] failed to compile quantum variants: {}",
                                        expr
                                    )
                                });
                                compile_single();
                            }
                        } else {
                            jit_log(|| {
                                format!("[Iris][jit] quantum compile gated by cooldown/budget for ptr={}", key)
                            });
                            compile_budget_gated_quantum();
                        }
                    }
                } else {
                    compile_single();
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
    let use_jit = !matches!(get_offload_strategy(key).as_deref(), Some("actor"));
    if use_jit {
        if let Some(res) = execute_registered_jit_guarded(py, key, args) {
            if let Ok(obj) = res {
                return Ok(obj);
            }
            if let Err(err) = &res {
                jit_log(|| {
                    format!(
                        "[Iris][jit] guarded execution failed in offload_call; falling back: {}",
                        err
                    )
                });
            }
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

    py.allow_threads(move || match rx.recv() {
        Ok(res) => res,
        Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "offload task canceled",
        )),
    })
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
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "no JIT entry found",
    ))
}

/// Execute a registered scalar 2-arg JIT step function in a Rust loop.
///
/// This is used by Python wrappers for recurrence kernels to avoid Python↔Rust
/// crossing overhead on each iteration.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit_step_loop_f64(func: PyObject, seed: f64, count: usize) -> PyResult<f64> {
    let key = func.as_ptr() as usize;
    let entry = lookup_jit(key)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no JIT entry found"))?;

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

#[cfg(test)]
mod tests;
