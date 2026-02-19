// src/py.rs
//! Minimal PyO3 membrane entry
//! Optimized for asynchronous service discovery and structured system messages.
#![allow(non_local_definitions)]

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::wrap_pyfunction;
use std::os::raw::{c_void, c_char};
use pyo3::types::{PyTuple, PyBytes};
use pyo3::PyObject;
use crate::buffer::{global_registry, BufferId};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use crossbeam_channel as cb_channel;

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
    m.add_class::<PyMailbox>()?;
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

/// Helper to convert a Rust `Message` to a Python object.
fn message_to_py(py: Python, msg: crate::mailbox::Message) -> PyObject {
    match msg {
        crate::mailbox::Message::User(b) => PyBytes::new(py, &b).into_py(py),
        crate::mailbox::Message::System(crate::mailbox::SystemMessage::Exit(target)) => {
            PySystemMessage { type_name: "EXIT".to_string(), target_pid: Some(target) }.into_py(py)
        }
        crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(_)) =>
            PySystemMessage { type_name: "HOT_SWAP".to_string(), target_pid: None }.into_py(py),
        crate::mailbox::Message::System(crate::mailbox::SystemMessage::Ping) =>
            PySystemMessage { type_name: "PING".to_string(), target_pid: None }.into_py(py),
        crate::mailbox::Message::System(crate::mailbox::SystemMessage::Pong) =>
            PySystemMessage { type_name: "PONG".to_string(), target_pid: None }.into_py(py),
    }
}

/// Run a Python matcher against a Rust message.
fn run_python_matcher(py: Python, matcher: &PyObject, msg: &crate::mailbox::Message) -> bool {
    match msg {
        crate::mailbox::Message::User(b) => {
            match matcher.call1(py, (PyBytes::new(py, &b),)) {
                Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                Err(_) => false,
            }
        }
        crate::mailbox::Message::System(s) => {
            match s {
                crate::mailbox::SystemMessage::Exit(target) => {
                    let obj = PySystemMessage { type_name: "EXIT".to_string(), target_pid: Some(*target) };
                    match matcher.call1(py, (obj.into_py(py),)) {
                        Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                        Err(_) => false,
                    }
                }
                crate::mailbox::SystemMessage::HotSwap(_) => {
                    let obj = PySystemMessage { type_name: "HOT_SWAP".to_string(), target_pid: None };
                    match matcher.call1(py, (obj.into_py(py),)) {
                        Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                        Err(_) => false,
                    }
                }
                crate::mailbox::SystemMessage::Ping => {
                    let obj = PySystemMessage { type_name: "PING".to_string(), target_pid: None };
                    match matcher.call1(py, (obj.into_py(py),)) {
                        Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                        Err(_) => false,
                    }
                }
                crate::mailbox::SystemMessage::Pong => {
                    let obj = PySystemMessage { type_name: "PONG".to_string(), target_pid: None };
                    match matcher.call1(py, (obj.into_py(py),)) {
                        Ok(val) => val.extract::<bool>(py).unwrap_or(false),
                        Err(_) => false,
                    }
                }
            }
        }
    }
}

/// A wrapper around a live MailboxReceiver for Python actors.
/// Revamped: Now purely blocking/synchronous to Python, running in dedicated threads.
#[pyclass]
#[derive(Clone)]
pub struct PyMailbox {
    inner: Arc<TokioMutex<crate::mailbox::MailboxReceiver>>,
}

#[pymethods]
impl PyMailbox {
    /// Receive the next message (Blocking).
    /// Releases the GIL while waiting.
    fn recv(&self, py: Python, timeout: Option<f64>) -> PyResult<PyObject> {
        let rx = self.inner.clone();
        
        // Release GIL to allow other threads to run while we block on the channel
        py.allow_threads(move || {
            let rt = tokio::runtime::Handle::current();
            
            // We are likely in a dedicated blocking thread, so we block_on the async work.
            let fut = async {
                let mut guard = rx.lock().await;
                guard.recv().await
            };

            let res = if let Some(sec) = timeout {
                rt.block_on(async {
                    match tokio::time::timeout(Duration::from_secs_f64(sec), fut).await {
                        Ok(val) => val,
                        Err(_) => None, // Timeout
                    }
                })
            } else {
                rt.block_on(fut)
            };

            // Re-acquire GIL to return result
            Python::with_gil(|py| {
                match res {
                    Some(msg) => Ok(message_to_py(py, msg)),
                    None => Ok(py.None()),
                }
            })
        })
    }

    /// Selectively receive a message matching a Python predicate (Blocking).
    /// Releases the GIL while waiting.
    fn selective_recv(&self, py: Python, matcher: PyObject, timeout: Option<f64>) -> PyResult<PyObject> {
        let rx = self.inner.clone();

        py.allow_threads(move || {
            let rt = tokio::runtime::Handle::current();
            
            let fut = async {
                let mut guard = rx.lock().await;
                guard.selective_recv(|msg| {
                    Python::with_gil(|py| run_python_matcher(py, &matcher, msg))
                }).await
            };

            let res = if let Some(sec) = timeout {
                rt.block_on(async {
                    match tokio::time::timeout(Duration::from_secs_f64(sec), fut).await {
                        Ok(val) => val,
                        Err(_) => None, // Timeout
                    }
                })
            } else {
                rt.block_on(fut)
            };

            Python::with_gil(|py| {
                match res {
                    Some(msg) => Ok(message_to_py(py, msg)),
                    None => Ok(py.None()),
                }
            })
        })
    }
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

    // --- Phase 6: Name Registry ---

    /// Register a human-readable name for a PID.
    /// If the name is already taken, it is overwritten.
    fn register(&self, name: String, pid: u64) -> PyResult<()> {
        self.inner.register(name, pid);
        Ok(())
    }

    /// Unregister a named PID.
    /// Does nothing if the name is not registered.
    fn unregister(&self, name: String) -> PyResult<()> {
        self.inner.unregister(&name);
        Ok(())
    }

    /// Resolve a name to its PID locally.
    /// Returns None if the name is not found.
    fn resolve(&self, name: String) -> PyResult<Option<u64>> {
        Ok(self.inner.resolve(&name))
    }

    /// Alias for resolve (Erlang style).
    fn whereis(&self, name: String) -> PyResult<Option<u64>> {
        Ok(self.inner.resolve(&name))
    }

    // --- End Registry ---

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

    /// Configure release_gil runtime limits programmatically.
    fn set_release_gil_limits(&self, max_threads: usize, pool_size: usize) -> PyResult<()> {
        self.inner.set_release_gil_limits(max_threads, pool_size);
        Ok(())
    }

    /// Enable or disable strict failure mode for release_gil (error on limit exceeded).
    fn set_release_gil_strict(&self, strict: bool) -> PyResult<()> {
        self.inner.set_release_gil_strict(strict);
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

    /// Spawns a push-based actor (original behavior).
    /// The `py_callable` is called with each message as an argument.
    /// If `release_gil` is true, the Python callback and hot-swap are executed
    /// in `tokio::task::spawn_blocking` so the actor's async loop doesn't hold the GIL.
    fn spawn_py_handler(&self, py_callable: PyObject, budget: usize, release_gil: Option<bool>) -> PyResult<u64> {
        let release = release_gil.unwrap_or(false);
        let behavior = Arc::new(parking_lot::RwLock::new(py_callable));
        // Safety guard: limit number of dedicated GIL-release threads to avoid OOM
        static RELEASE_GIL_THREADS: AtomicUsize = AtomicUsize::new(0);
        const DEFAULT_MAX_RELEASE_GIL_THREADS: usize = 256;
        const DEFAULT_GIL_POOL_SIZE: usize = 8;

        // Global shared GIL worker pool (initialized on demand)
        static GIL_WORKER_POOL: OnceLock<Arc<GilPool>> = OnceLock::new();

        // Pool task description used by shared workers
        enum PoolTask {
            Execute { behavior: Arc<parking_lot::RwLock<PyObject>>, bytes: bytes::Bytes },
            HotSwap { behavior: Arc<parking_lot::RwLock<PyObject>>, ptr: usize },
        }

        struct GilPool {
            sender: cb_channel::Sender<PoolTask>,
        }

        impl GilPool {
            fn new(size: usize) -> Self {
                let (tx, rx) = cb_channel::unbounded::<PoolTask>();
                for _ in 0..size {
                    let rx = rx.clone();
                    std::thread::spawn(move || {
                        while let Ok(task) = rx.recv() {
                            match task {
                                PoolTask::Execute { behavior, bytes } => {
                                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                        continue;
                                    }
                                    Python::with_gil(|py| {
                                        let guard = behavior.read();
                                        let cb = guard.as_ref(py);
                                        let pybytes = PyBytes::new(py, &bytes);
                                        if let Err(e) = cb.call1((pybytes,)) {
                                            eprintln!("[Myrmidon] Python actor exception: {}", e);
                                            e.print(py);
                                        }
                                    });
                                }
                                PoolTask::HotSwap { behavior, ptr } => {
                                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                        continue;
                                    }
                                    Python::with_gil(|py| unsafe {
                                        let new_obj = PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject);
                                        *behavior.write() = new_obj;
                                    });
                                }
                            }
                        }
                    });
                }
                GilPool { sender: tx }
            }
        }

        // If release is enabled, attempt to create a dedicated OS thread per-actor that
        // owns the Python interaction loop. We forward incoming messages to
        // that thread via a std::sync::mpsc channel to avoid per-message
        // spawn_blocking overhead and blocking-pool saturation. If we exceed the
        // dedicated-thread limit we fall back to a shared GIL worker pool.
        let maybe_tx = if release {
                // Read configurable limits from the Runtime instance
                let (max_threads, pool_size) = self.inner.get_release_gil_limits();
                let strict = self.inner.is_release_gil_strict();

                // Enforce global limit for dedicated threads
                let prev = RELEASE_GIL_THREADS.fetch_add(1, Ordering::SeqCst);
                if prev >= max_threads {
                    // Reached limit: decrement counter
                    RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
                    if strict {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err("release_gil thread limit exceeded"));
                    }
                    // Use shared pool as a fallback
                    let _ = GIL_WORKER_POOL.get_or_init(|| Arc::new(GilPool::new(pool_size))).clone();
                    None
                } else {
                let (tx, rx) = cb_channel::unbounded::<crate::mailbox::Message>();
                let b_thread = behavior.clone();

                std::thread::spawn(move || {
                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                        RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
                        return;
                    }

                    while let Ok(msg) = rx.recv() {
                        match msg {
                            crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                    continue;
                                }
                                Python::with_gil(|py| unsafe {
                                    let new_obj = PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject);
                                    *b_thread.write() = new_obj;
                                });
                            }
                            crate::mailbox::Message::User(bytes) => {
                                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                    continue;
                                }
                                Python::with_gil(|py| {
                                    let guard = b_thread.read();
                                    let cb = guard.as_ref(py);
                                    let pybytes = PyBytes::new(py, &bytes);
                                    if let Err(e) = cb.call1((pybytes,)) {
                                        eprintln!("[Myrmidon] Python actor exception: {}", e);
                                        e.print(py);
                                    }
                                });
                            }
                            crate::mailbox::Message::System(crate::mailbox::SystemMessage::Exit(_)) => {
                                break;
                            }
                            crate::mailbox::Message::System(crate::mailbox::SystemMessage::Ping) |
                            crate::mailbox::Message::System(crate::mailbox::SystemMessage::Pong) => {}
                        }
                    }
                    // Thread is exiting — decrement global counter
                    RELEASE_GIL_THREADS.fetch_sub(1, Ordering::SeqCst);
                });

                Some(tx)
            }
        } else {
            None
        };

        let handler = move |msg: crate::mailbox::Message| {
            let b = behavior.clone();
            let tx = maybe_tx.clone();
            let release_gil = release;
            async move {
                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                    return;
                }

                match msg {
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(_)) if tx.is_some() => {
                        // Forward hot-swap to the dedicated Python thread to perform
                        // the pointer->PyObject conversion under the GIL.
                        if let Some(tx) = tx {
                            let _ = tx.send(msg);
                        }
                    }
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) if release_gil => {
                        // No dedicated thread available — use shared pool if present, else fallback inline
                        if let Some(pool) = GIL_WORKER_POOL.get() {
                            let task = PoolTask::HotSwap { behavior: b.clone(), ptr };
                            let _ = pool.sender.send(task);
                        } else {
                            Python::with_gil(|py| unsafe {
                                let new_obj = PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject);
                                *b.write() = new_obj;
                            });
                        }
                    }
                    crate::mailbox::Message::User(_) if tx.is_some() => {
                        if let Some(tx) = tx {
                            let _ = tx.send(msg);
                        }
                    }
                    crate::mailbox::Message::User(bytes) if release_gil => {
                        if let Some(pool) = GIL_WORKER_POOL.get() {
                            let task = PoolTask::Execute { behavior: b.clone(), bytes: bytes.clone() };
                            let _ = pool.sender.send(task);
                        } else {
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
                    }
                    // No dedicated thread: execute inline under the GIL.
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                        Python::with_gil(|py| unsafe {
                            let new_obj = PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject);
                            *b.write() = new_obj;
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
                        // If we have a dedicated thread, dropping tx will cause it to exit.
                        drop(tx);
                    }
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::Ping) |
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::Pong) => {}
                }
            }
        };

        Ok(self.inner.spawn_handler_with_budget(handler, budget))
    }

    /// Spawns a pull-based actor.
    /// The `py_callable` is called ONCE with a `PyMailbox` object in a dedicated OS thread.
    /// This mimics Erlang/Go style blocking actors without needing Python asyncio.
    fn spawn_with_mailbox(&self, py_callable: PyObject, budget: usize) -> PyResult<u64> {
        let pid = self.inner.spawn_actor_with_budget(move |rx| async move {
            let mailbox = PyMailbox {
                inner: Arc::new(TokioMutex::new(rx))
            };
            
            // We are currently in a Tokio worker thread (async).
            // We need to spawn a dedicated OS thread (blocking) for the Python synchronous loop
            // to avoid blocking the Tokio runtime.
            let handle = tokio::task::spawn_blocking(move || {
                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                    return;
                }
                
                Python::with_gil(|py| {
                    // Just call the function. It is expected to block on mailbox.recv()
                    if let Err(e) = py_callable.call1(py, (mailbox,)) {
                        eprintln!("[Myrmidon] Python mailbox actor exception: {}", e);
                        e.print(py);
                    }
                });
            });

            // Await the thread's completion. This keeps the actor "alive" in the system
            // until the Python function returns.
            let _ = handle.await;
        }, budget);

        Ok(pid)
    }

    fn send(&self, pid: u64, data: &PyBytes) -> PyResult<bool> {
        let msg = bytes::Bytes::copy_from_slice(data.as_bytes());
        Ok(self.inner.send(pid, crate::mailbox::Message::User(msg)).is_ok())
    }

    /// Await selectively on observed messages for `pid` using a Python callable.
    fn selective_recv_observed_py<'py>(&self, py: Python<'py>, pid: u64, matcher: PyObject, timeout: Option<f64>) -> PyResult<&'py PyAny> {
        let rt = self.inner.clone();
        future_into_py(py, async move {
            let op = async {
                loop {
                    // Attempt to take a matching observed message atomically.
                    if let Some(m) = rt.take_observed_message_matching(pid, |msg| {
                        // Call into Python matcher to decide.
                        Python::with_gil(|py| run_python_matcher(py, &matcher, msg))
                    }) {
                        // Convert the message into a Python object before returning.
                        return Python::with_gil(|py| message_to_py(py, m));
                    }

                    // Not found yet — yield a bit and try again.
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            };

            if let Some(sec) = timeout {
                match tokio::time::timeout(Duration::from_secs_f64(sec), op).await {
                    Ok(val) => Ok(val),
                    Err(_) => Ok(Python::with_gil(|py| py.None()))
                }
            } else {
                Ok(op.await)
            }
        })
    }

    /// Retrieves messages from an observed actor.
    fn get_messages(&self, py: Python, pid: u64) -> PyResult<Vec<PyObject>> {
        if let Some(vec) = self.inner.get_observed_messages(pid) {
            let out = vec.into_iter().map(|m| message_to_py(py, m)).collect();
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
