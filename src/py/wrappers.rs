// src/py/wrappers.rs
//! Python-facing helper functions and module initialization.
#![allow(non_local_definitions)]

use crate::buffer::{global_registry, BufferId};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use std::os::raw::{c_char, c_void};

use super::runtime::PyRuntime;

extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if capsule.is_null() {
        return;
    }
    unsafe {
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, std::ptr::null());
        if !ptr.is_null() {
            let id_ptr = ptr as *mut BufferId;
            let id = std::ptr::read(id_ptr);
            global_registry().free(id);
            let _ = Box::from_raw(id_ptr);
        }
    }
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn allocate_buffer(py: Python, size: usize) -> PyResult<PyObject> {
    let id = global_registry().allocate(size);
    let (ptr, len) = global_registry()
    .ptr_len(id)
    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("failed to allocate"))?;

    unsafe {
        let mv = pyo3::ffi::PyMemoryView_FromMemory(
            ptr as *mut c_char,
            len as isize,
            pyo3::ffi::PyBUF_WRITE,
        );
        if mv.is_null() {
            global_registry().free(id);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "failed to create memoryview",
            ));
        }

        let boxed = Box::new(id);
        let capsule = pyo3::ffi::PyCapsule_New(
            Box::into_raw(boxed) as *mut c_void,
                                               std::ptr::null(),
                                               Some(capsule_destructor),
        );
        if capsule.is_null() {
            pyo3::ffi::Py_DecRef(mv as *mut pyo3::ffi::PyObject);
            global_registry().free(id);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "failed to create capsule",
            ));
        }

        let memobj = PyObject::from_owned_ptr(py, mv as *mut pyo3::ffi::PyObject);
        let idobj = id.into_py(py);
        let capobj = PyObject::from_owned_ptr(py, capsule as *mut pyo3::ffi::PyObject);
        Ok(PyTuple::new(py, &[idobj, memobj, capobj]).into())
    }
}

// Path-based registry helpers (module-level convenience wrappers)
#[pyfunction]
fn register_path(rt: PyRef<PyRuntime>, path: String, pid: u64) -> PyResult<()> {
    rt.inner.register_path(path, pid);
    Ok(())
}

#[pyfunction]
fn unregister_path(rt: PyRef<PyRuntime>, path: String) -> PyResult<()> {
    rt.inner.unregister_path(&path);
    Ok(())
}

#[pyfunction]
fn whereis_path(rt: PyRef<PyRuntime>, path: String) -> PyResult<Option<u64>> {
    Ok(rt.inner.whereis_path(&path))
}

#[pyfunction]
fn list_children(rt: PyRef<PyRuntime>, prefix: String) -> PyResult<Vec<(String, u64)>> {
    Ok(rt.inner.list_children(&prefix))
}

#[pyfunction]
fn spawn_with_path_observed(rt: PyRef<PyRuntime>, budget: usize, path: String) -> PyResult<u64> {
    Ok(rt.inner.spawn_with_path_observed(budget, path))
}

#[pyfunction]
fn list_children_direct(rt: PyRef<PyRuntime>, prefix: String) -> PyResult<Vec<(String, u64)>> {
    Ok(rt.inner.list_children_direct(&prefix))
}

#[pyfunction]
fn watch_path(rt: PyRef<PyRuntime>, prefix: String) -> PyResult<()> {
    rt.inner.watch_path(&prefix);
    Ok(())
}

#[pyfunction]
fn create_path_supervisor(rt: PyRef<PyRuntime>, path: String) -> PyResult<()> {
    rt.inner.create_path_supervisor(&path);
    Ok(())
}

#[pyfunction]
fn remove_path_supervisor(rt: PyRef<PyRuntime>, path: String) -> PyResult<()> {
    rt.inner.remove_path_supervisor(&path);
    Ok(())
}

#[pyfunction]
fn path_supervisor_watch(rt: PyRef<PyRuntime>, path: String, pid: u64) -> PyResult<()> {
    rt.inner.path_supervisor_watch(&path, pid);
    Ok(())
}

#[pyfunction]
fn path_supervisor_children(rt: PyRef<PyRuntime>, path: String) -> PyResult<Vec<u64>> {
    Ok(rt.inner.path_supervisor_children(&path))
}

#[cfg(feature = "pyo3")]
fn populate_module(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<PyRuntime>()?;
    m.add_class::<super::utils::PySystemMessage>()?;
    m.add_class::<super::mailbox::PyMailbox>()?;
    #[cfg(feature = "pyo3")]
    m.add_function(wrap_pyfunction!(allocate_buffer, m)?)?;
    // Path-based registry helpers (module-level convenience wrappers)
    m.add_function(wrap_pyfunction!(register_path, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_path, m)?)?;
    m.add_function(wrap_pyfunction!(whereis_path, m)?)?;
    m.add_function(wrap_pyfunction!(list_children, m)?)?;
    m.add_function(wrap_pyfunction!(spawn_with_path_observed, m)?)?;
    m.add_function(wrap_pyfunction!(list_children_direct, m)?)?;
    m.add_function(wrap_pyfunction!(watch_path, m)?)?;
    m.add_function(wrap_pyfunction!(create_path_supervisor, m)?)?;
    m.add_function(wrap_pyfunction!(remove_path_supervisor, m)?)?;
    m.add_function(wrap_pyfunction!(path_supervisor_watch, m)?)?;
    m.add_function(wrap_pyfunction!(path_supervisor_children, m)?)?;
    Ok(())
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn iris(_py: Python, m: &PyModule) -> PyResult<()> {
    populate_module(m)
}

#[cfg(feature = "pyo3")]
pub fn make_module(py: Python) -> PyResult<Py<PyModule>> {
    let m = PyModule::new(py, "iris")?;
    populate_module(m)?;
    Ok(m.into())
}

#[cfg(feature = "pyo3")]
pub fn init() {}
