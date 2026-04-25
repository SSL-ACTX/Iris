// src/py/jit_stub.rs
//! Python JIT API stubs when the `jit` feature is disabled.

#![allow(non_local_definitions)]

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyTuple};
#[cfg(feature = "pyo3")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "pyo3")]
#[pyfunction]
fn register_offload(
    func: Py<PyAny>,
    _strategy: Option<String>,
    _return_type: Option<String>,
    _source_expr: Option<String>,
    _arg_names: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    Ok(func)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn offload_call(
    py: Python<'_>,
    func: Py<PyAny>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    func.bind(py).call(args, kwargs).map(|v| v.unbind())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit(
    _py: Python<'_>,
    _func: Py<PyAny>,
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "JIT feature is disabled in this build",
    ))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit_step_loop_f64(_func: Py<PyAny>, _seed: f64, _count: usize) -> PyResult<f64> {
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "JIT feature is disabled in this build",
    ))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_jit_logging(_enabled: Option<bool>, _env_var: Option<String>) -> PyResult<bool> {
    Ok(false)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn is_jit_logging_enabled() -> PyResult<bool> {
    Ok(false)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_speculation(
    _enabled: Option<bool>,
    _env_var: Option<String>,
) -> PyResult<bool> {
    Ok(false)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn is_quantum_speculation_enabled() -> PyResult<bool> {
    Ok(false)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_profile(_func: Py<PyAny>) -> PyResult<Vec<(usize, f64, u64, u64)>> {
    Ok(Vec::new())
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn seed_quantum_profile(_func: Py<PyAny>, _rows: Vec<(usize, f64, u64, u64)>) -> PyResult<bool> {
    Ok(false)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_speculation_threshold(
    _threshold_ns: Option<u64>,
    _env_var: Option<String>,
) -> PyResult<u64> {
    Ok(1_000_000)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_speculation_threshold() -> PyResult<u64> {
    Ok(1_000_000)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_log_threshold(
    _threshold_ns: Option<u64>,
    _env_var: Option<String>,
) -> PyResult<u64> {
    Ok(1_000_000)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_log_threshold() -> PyResult<u64> {
    Ok(1_000_000)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_compile_budget(
    _budget_ns: Option<u64>,
    _window_ns: Option<u64>,
    _budget_env_var: Option<String>,
    _window_env_var: Option<String>,
) -> PyResult<(u64, u64)> {
    Ok((50_000_000, 1_000_000_000))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_compile_budget() -> PyResult<(u64, u64)> {
    Ok((50_000_000, 1_000_000_000))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn configure_quantum_cooldown(
    _base_ns: Option<u64>,
    _max_ns: Option<u64>,
    _base_env_var: Option<String>,
    _max_env_var: Option<String>,
) -> PyResult<(u64, u64)> {
    Ok((5_000_000, 1_000_000_000))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
fn get_quantum_cooldown() -> PyResult<(u64, u64)> {
    Ok((5_000_000, 1_000_000_000))
}

#[cfg(feature = "pyo3")]
pub(crate) fn init_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_offload, m)?)?;
    m.add_function(wrap_pyfunction!(offload_call, m)?)?;
    m.add_function(wrap_pyfunction!(call_jit, m)?)?;
    m.add_function(wrap_pyfunction!(call_jit_step_loop_f64, m)?)?;
    m.add_function(wrap_pyfunction!(configure_jit_logging, m)?)?;
    m.add_function(wrap_pyfunction!(is_jit_logging_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(configure_quantum_speculation, m)?)?;
    m.add_function(wrap_pyfunction!(is_quantum_speculation_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(get_quantum_profile, m)?)?;
    m.add_function(wrap_pyfunction!(seed_quantum_profile, m)?)?;
    m.add_function(wrap_pyfunction!(
        configure_quantum_speculation_threshold,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(get_quantum_speculation_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(configure_quantum_log_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(get_quantum_log_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(configure_quantum_compile_budget, m)?)?;
    m.add_function(wrap_pyfunction!(get_quantum_compile_budget, m)?)?;
    m.add_function(wrap_pyfunction!(configure_quantum_cooldown, m)?)?;
    m.add_function(wrap_pyfunction!(get_quantum_cooldown, m)?)?;
    Ok(())
}
