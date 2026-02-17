// src/py.rs
//! Minimal PyO3 membrane entry
//! Optimized for asynchronous service discovery and structured system messages.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::wrap_pyfunction;
use std::os::raw::{c_void, c_char};
use pyo3::types::{PyTuple, PyBytes};
use pyo3::PyObject;
use crate::buffer::{global_registry, BufferId};
use std::sync::Arc;

#[cfg(feature = "pyo3")]
use pyo3_asyncio::tokio::future_into_py;

#[cfg(feature = "pyo3")]
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "pyo3")]
fn populate_module(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<PyRuntime>()?;
    m.add_class::<PySystemMessage>()?;
    #[cfg(feature = "pyo3")]
    m.add_function(wrap_pyfunction!(allocate_buffer, m)?)?;
    Ok(())
}

extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    if capsule.is_null() { return; }
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
fn allocate_buffer(py: Python, size: usize) -> PyResult<PyObject> {
    let id = global_registry().allocate(size);
    let (ptr, len) = global_registry().ptr_len(id).ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("failed to allocate"))?;

    unsafe {
        let mv = pyo3::ffi::PyMemoryView_FromMemory(ptr as *mut c_char, len as isize, pyo3::ffi::PyBUF_WRITE);
        if mv.is_null() {
            global_registry().free(id);
            return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create memoryview"));
        }

        let boxed = Box::new(id);
        let capsule = pyo3::ffi::PyCapsule_New(Box::into_raw(boxed) as *mut c_void, std::ptr::null(), Some(capsule_destructor));
        if capsule.is_null() {
            pyo3::ffi::Py_DecRef(mv as *mut pyo3::ffi::PyObject);
            global_registry().free(id);
            return Err(pyo3::exceptions::PyRuntimeError::new_err("failed to create capsule"));
        }

        let memobj = PyObject::from_owned_ptr(py, mv as *mut pyo3::ffi::PyObject);
        let idobj = id.into_py(py);
        let capobj = PyObject::from_owned_ptr(py, capsule as *mut pyo3::ffi::PyObject);
        Ok(PyTuple::new(py, &[idobj, memobj, capobj]).into())
    }
}

/// Phase 7: Structured System Message for Python
#[pyclass]
#[derive(Clone)]
pub struct PySystemMessage {
    #[pyo3(get)]
    pub type_name: String,
    #[pyo3(get)]
    pub target_pid: Option<u64>,
}

#[pyclass]
struct PyRuntime {
    inner: std::sync::Arc<crate::Runtime>,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        Self { inner: std::sync::Arc::new(crate::Runtime::new()) }
    }

    /// Phase 6: Register a human-readable name for a PID.
    fn register(&self, name: String, pid: u64) -> PyResult<()> {
        self.inner.register(name, pid);
        Ok(())
    }

    /// Phase 6: Resolve a name to its PID locally.
    fn resolve(&self, name: String) -> PyResult<Option<u64>> {
        Ok(self.inner.resolve(&name))
    }

    /// Phase 7: Resolve a name on a remote node (Synchronous/Blocking).
    /// Detects if an active runtime exists. If so, uses block_in_place to avoid panics.
    fn resolve_remote(&self, addr: String, name: String) -> PyResult<Option<u64>> {
        let rt = self.inner.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // Must use block_in_place if we are already in a runtime thread
            Ok(tokio::task::block_in_place(|| {
                handle.block_on(rt.resolve_remote_async(addr, name))
            }))
        } else {
            Ok(crate::RUNTIME.block_on(rt.resolve_remote_async(addr, name)))
        }
    }

    /// Phase 7: Resolve a name on a remote node (Asynchronous).
    /// Returns a Python Awaitable (Future) for use in asyncio loops.
    fn resolve_remote_py<'py>(&self, py: Python<'py>, addr: String, name: String) -> PyResult<&'py PyAny> {
        let rt = self.inner.clone();
        future_into_py(py, async move {
            let pid = rt.resolve_remote_async(addr, name).await;
            Ok(pid)
        })
    }

    /// Phase 5: Start a TCP listener on the specified address for remote message passing.
    fn listen(&self, addr: String) -> PyResult<()> {
        self.inner.listen(addr);
        Ok(())
    }

    /// Phase 5: Send a binary payload to a PID on a remote node.
    fn send_remote(&self, addr: String, pid: u64, data: &PyBytes) -> PyResult<()> {
        let bytes = bytes::Bytes::copy_from_slice(data.as_bytes());
        self.inner.send_remote(addr, pid, bytes);
        Ok(())
    }

    /// Phase 5: Monitor a remote PID.
    fn monitor_remote(&self, addr: String, pid: u64) -> PyResult<()> {
        self.inner.monitor_remote(addr, pid);
        Ok(())
    }

    /// Quick network probe to check if a node is reachable.
    /// Returns a boolean directly from the future to avoid type inference issues.
    fn is_node_up(&self, addr: String) -> PyResult<bool> {
        let fut = async {
            match tokio::net::TcpStream::connect(&addr).await {
                Ok(_) => true,
                Err(_) => false,
            }
        };

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // Must use block_in_place to prevent "runtime within runtime" panic
            Ok(tokio::task::block_in_place(|| {
                handle.block_on(fut)
            }))
        } else {
            Ok(crate::RUNTIME.block_on(fut))
        }
    }

    fn join(&self, py: Python, pid: u64) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.wait(pid);
        });
        Ok(())
    }

    fn stop(&self, pid: u64) -> PyResult<()> {
        self.inner.stop(pid);
        Ok(())
    }

    fn hot_swap(&self, pid: u64, new_handler: PyObject) -> PyResult<()> {
        let ptr = new_handler.into_ptr();
        self.inner.hot_swap(pid, ptr as usize);
        Ok(())
    }

    fn spawn_observed_handler(&self, budget: usize) -> u64 {
        self.inner.spawn_observed_handler(budget)
    }

    fn send_buffer(&self, pid: u64, buffer_id: u64) -> PyResult<bool> {
        if let Some(vec) = crate::buffer::global_registry().take(buffer_id) {
            let b = bytes::Bytes::from(vec);
            Ok(self.inner.send(pid, crate::mailbox::Message::User(b)).is_ok())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("invalid buffer id or already taken"))
        }
    }

    fn spawn_py_handler(&self, py_callable: PyObject, budget: usize) -> PyResult<u64> {
        let behavior = Arc::new(parking_lot::RwLock::new(py_callable));

        let handler = move |msg: crate::mailbox::Message| {
            let b = behavior.clone();
            async move {
                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                    return;
                }

                match msg {
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                        Python::with_gil(|py| {
                            unsafe {
                                let new_obj = PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject);
                                *b.write() = new_obj;
                            }
                        });
                    }
                    crate::mailbox::Message::User(bytes) => {
                        Python::with_gil(|py| {
                            let guard = b.read();
                            let cb = guard.as_ref(py);
                            let pybytes = PyBytes::new(py, &bytes);
                            if let Err(e) = cb.call1((pybytes,)) {
                                eprintln!("[Myrmidon] Python actor exception: {}", e);
                                e.print(py);
                            }
                        });
                    }
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::Exit(pid)) => {
                        let _ = pid;
                    }
                }
            }
        };
        Ok(self.inner.spawn_handler_with_budget(handler, budget))
    }

    fn send(&self, pid: u64, data: &PyBytes) -> PyResult<bool> {
        let msg = bytes::Bytes::copy_from_slice(data.as_bytes());
        Ok(self.inner.send(pid, crate::mailbox::Message::User(msg)).is_ok())
    }

    /// Retrieves messages from an observed actor.
    /// Now maps Message::System to PySystemMessage objects.
    fn get_messages(&self, py: Python, pid: u64) -> PyResult<Vec<PyObject>> {
        if let Some(vec) = self.inner.get_observed_messages(pid) {
            let out = vec.into_iter().map(|m| match m {
                crate::mailbox::Message::User(b) => PyBytes::new(py, &b).into_py(py),
                                          crate::mailbox::Message::System(crate::mailbox::SystemMessage::Exit(target)) => {
                                              PySystemMessage {
                                                  type_name: "EXIT".to_string(),
                                          target_pid: Some(target),
                                              }.into_py(py)
                                          }
                                          crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(_)) => {
                                              PySystemMessage {
                                                  type_name: "HOT_SWAP".to_string(),
                                          target_pid: None,
                                              }.into_py(py)
                                          }
            }).collect();
            Ok(out)
        } else {
            Ok(Vec::new())
        }
    }

    fn is_alive(&self, pid: u64) -> bool {
        self.inner.is_alive(pid)
    }

    fn children_count(&self) -> usize {
        self.inner.supervisor().children_count()
    }

    fn child_pids(&self) -> Vec<u64> {
        self.inner.supervisor().child_pids()
    }

    fn watch(&self, pid: u64, strategy: &str) -> PyResult<()> {
        use crate::supervisor::RestartStrategy;
        use crate::supervisor::ChildSpec;
        use std::sync::Arc;

        let strat = match strategy.to_lowercase().as_str() {
            "restartone" | "restart_one" | "one" => RestartStrategy::RestartOne,
            "restartall" | "restart_all" | "all" => RestartStrategy::RestartAll,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid strategy")),
        };

        let spec = ChildSpec { factory: Arc::new(move || Ok(pid)), strategy: strat };
        self.inner.supervisor().add_child(pid, spec);
        Ok(())
    }

    fn supervise_with_factory(&self, pid: u64, py_factory: PyObject, strategy: &str) -> PyResult<()> {
        use std::sync::Arc;

        let strat = match strategy.to_lowercase().as_str() {
            "restartone" | "restart_one" | "one" => crate::supervisor::RestartStrategy::RestartOne,
            "restartall" | "restart_all" | "all" => crate::supervisor::RestartStrategy::RestartAll,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid strategy")),
        };

        let _initial_pid = Python::with_gil(|py| {
            let obj = py_factory.as_ref(py);
            let called = obj.call0()?;
            let pid: u64 = called.extract()?;
            Ok::<u64, pyo3::PyErr>(pid)
        })?;

        let factory_py = py_factory.clone();
        let factory_closure: Arc<dyn Fn() -> Result<crate::pid::Pid, String> + Send + Sync> = Arc::new(move || {
            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                return Err("Interpreter shutting down".to_string());
            }
            Python::with_gil(|py| {
                let obj = factory_py.as_ref(py);
                match obj.call0() {
                    Ok(v) => match v.extract::<u64>() {
                        Ok(pid) => Ok(pid),
                             Err(e) => Err(e.to_string())
                    },
                    Err(e) => Err(e.to_string())
                }
            })
        });

        self.inner.supervise(pid, factory_closure, strat);
        Ok(())
    }
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn myrmidon(_py: Python, m: &PyModule) -> PyResult<()> {
    populate_module(m)
}

#[cfg(feature = "pyo3")]
pub fn make_module(py: Python) -> PyResult<Py<PyModule>> {
    let m = PyModule::new(py, "myrmidon")?;
    populate_module(m)?;
    Ok(m.into())
}

#[cfg(feature = "pyo3")]
pub fn init() {
}
