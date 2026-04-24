pub mod callbacks;
pub mod model;
pub mod state;
pub mod telemetry;

pub use callbacks::{
    get_tracing_stats, instruction_callback, is_tracing_active, jump_callback, py_return_callback,
    py_start_callback, start_tracing, stop_tracing,
};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn init_py(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(stop_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(instruction_callback, m)?)?;
    m.add_function(wrap_pyfunction!(py_start_callback, m)?)?;
    m.add_function(wrap_pyfunction!(jump_callback, m)?)?;
    m.add_function(wrap_pyfunction!(py_return_callback, m)?)?;
    m.add_function(wrap_pyfunction!(get_tracing_stats, m)?)?;
    m.add_function(wrap_pyfunction!(is_tracing_active, m)?)?;
    Ok(())
}
