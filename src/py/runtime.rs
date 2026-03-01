// src/py/runtime.rs
//! Python-facing runtime wrapper and associated methods.
#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_asyncio::tokio::future_into_py;
use std::sync::Arc;
use std::time::Duration;
use bytes;
use tokio::sync::Mutex as TokioMutex;

use crate::Runtime;

use super::pool::{make_release_gil_channel, PoolTask, GIL_WORKER_POOL};
use super::utils::{message_to_py, run_python_matcher};
use super::mailbox::PyMailbox;

#[pyclass]
pub struct PyRuntime {
    pub(crate) inner: std::sync::Arc<Runtime>,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(crate::Runtime::new()),
        }
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

    /// Register a hierarchical path for an actor PID.
    fn register_path(&self, path: String, pid: u64) -> PyResult<()> {
        self.inner.register_path(path, pid);
        Ok(())
    }

    fn unregister_path(&self, path: String) -> PyResult<()> {
        self.inner.unregister_path(&path);
        Ok(())
    }

    fn whereis_path(&self, path: String) -> PyResult<Option<u64>> {
        Ok(self.inner.whereis_path(&path))
    }

    fn list_children(&self, prefix: String) -> PyResult<Vec<(String, u64)>> {
        Ok(self.inner.list_children(&prefix))
    }

    fn list_children_direct(&self, prefix: String) -> PyResult<Vec<(String, u64)>> {
        Ok(self.inner.list_children_direct(&prefix))
    }

    fn watch_path(&self, prefix: String) -> PyResult<()> {
        self.inner.watch_path(&prefix);
        Ok(())
    }

    fn spawn_with_path_observed(&self, budget: usize, path: String) -> PyResult<u64> {
        Ok(self.inner.spawn_with_path_observed(budget, path))
    }

    fn create_path_supervisor(&self, path: String) -> PyResult<()> {
        self.inner.create_path_supervisor(&path);
        Ok(())
    }

    fn remove_path_supervisor(&self, path: String) -> PyResult<()> {
        self.inner.remove_path_supervisor(&path);
        Ok(())
    }

    fn path_supervisor_watch(&self, path: String, pid: u64) -> PyResult<()> {
        self.inner.path_supervisor_watch(&path, pid);
        Ok(())
    }

    fn path_supervisor_children(&self, path: String) -> PyResult<Vec<u64>> {
        Ok(self.inner.path_supervisor_children(&path))
    }

    // --- End Registry ---

    /// Phase 7: Resolve a name on a remote node (Synchronous/Blocking).
    /// Detects if an active runtime exists. If so, uses block_in_place to avoid panics.
    fn resolve_remote(&self, py: Python, addr: String, name: String) -> PyResult<Option<u64>> {
        let rt = self.inner.clone();
        py.allow_threads(|| {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                Ok(tokio::task::block_in_place(|| {
                    handle.block_on(rt.resolve_remote_async(addr, name))
                }))
            } else {
                Ok(crate::RUNTIME.block_on(rt.resolve_remote_async(addr, name)))
            }
        })
    }

    /// Phase 7: Resolve a name on a remote node (Asynchronous).
    /// Returns a Python Awaitable (Future) for use in asyncio loops.
    fn resolve_remote_py<'py>(
        &self,
        py: Python<'py>,
        addr: String,
        name: String,
    ) -> PyResult<&'py PyAny> {
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
    fn is_node_up(&self, py: Python, addr: String) -> PyResult<bool> {
        let fut = async {
            match tokio::net::TcpStream::connect(&addr).await {
                Ok(_) => true,
                Err(_) => false,
            }
        };

        py.allow_threads(|| {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                // Must use block_in_place to prevent "runtime within runtime" panic
                Ok(tokio::task::block_in_place(|| handle.block_on(fut)))
            } else {
                Ok(crate::RUNTIME.block_on(fut))
            }
        })
    }

    fn join(&self, py: Python, pid: u64) -> PyResult<()> {
        loop {
            // Allows Python signal handlers to jump in and kill hanging processes
            py.check_signals()?;

            let is_alive = py.allow_threads(|| {
                let alive = self.inner.is_alive(pid);
                if alive {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                alive
            });

            if !is_alive {
                break;
            }
        }
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
            Ok(self
            .inner
            .send(pid, crate::mailbox::Message::User(b))
            .is_ok())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "invalid buffer id or already taken",
            ))
        }
    }

    /// Spawns a push-based actor (original behavior).
    /// The `py_callable` is called with each message as an argument.
    /// If `release_gil` is true, the Python callback and hot-swap are executed
    /// in `tokio::task::spawn_blocking` so the actor's async loop doesn't hold the GIL.
    fn spawn_py_handler(
        &self,
        py_callable: PyObject,
        budget: usize,
        release_gil: Option<bool>,
    ) -> PyResult<u64> {
        let release = release_gil.unwrap_or(false);
        let behavior = Arc::new(parking_lot::RwLock::new(py_callable));
        // compute maybe_tx using shared helper (propagates error on strict limit)
        let maybe_tx = make_release_gil_channel(&self.inner, release, behavior.clone())?;

        let handler = move |msg: crate::mailbox::Message| {
            let b = behavior.clone();
            let tx = maybe_tx.clone();
            let release_gil = release;
            async move {
                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                    return;
                }

                if let Some(tx) = &tx {
                    // dedicated-thread path: translate to PoolTask
                    match msg {
                        crate::mailbox::Message::User(bytes) => {
                            let task = PoolTask::Execute {
                                behavior: b.clone(),
                                bytes: bytes.clone(),
                            };
                            let _ = tx.send(task);
                        }
                        crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                            let task = PoolTask::HotSwap {
                                behavior: b.clone(),
                                ptr,
                            };
                            let _ = tx.send(task);
                        }
                        _ => {}
                    }
                } else if release_gil {
                    // shared-pool fallback when release requested but no dedicated thread
                    match msg {
                        crate::mailbox::Message::User(bytes) => {
                            if let Some(pool) = GIL_WORKER_POOL.get() {
                                let task = PoolTask::Execute {
                                    behavior: b.clone(),
                                    bytes: bytes.clone(),
                                };
                                let _ = pool.sender.send(task);
                            } else {
                                Python::with_gil(|py| {
                                    let guard = b.read();
                                    let cb = guard.as_ref(py);
                                    let pybytes = PyBytes::new(py, &bytes);
                                    if let Err(e) = cb.call1((pybytes,)) {
                                        eprintln!("[Iris] Python actor exception: {}", e);
                                        e.print(py);
                                    }
                                });
                            }
                        }
                        crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                            if let Some(pool) = GIL_WORKER_POOL.get() {
                                let task = PoolTask::HotSwap {
                                    behavior: b.clone(),
                                    ptr,
                                };
                                let _ = pool.sender.send(task);
                            } else {
                                Python::with_gil(|py| unsafe {
                                    let new_obj =
                                        PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject);
                                    *b.write() = new_obj;
                                });
                            }
                        }
                        _ => {}
                    }
                } else {
                    // pure inline path
                    match msg {
                        crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                            Python::with_gil(|py| unsafe {
                                let new_obj =
                                    PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject);
                                *b.write() = new_obj;
                            });
                        }
                        crate::mailbox::Message::User(bytes) => {
                            Python::with_gil(|py| {
                                let guard = b.read();
                                let cb = guard.as_ref(py);
                                let pybytes = PyBytes::new(py, &bytes);
                                if let Err(e) = cb.call1((pybytes,)) {
                                    eprintln!("[Iris] Python actor exception: {}", e);
                                    e.print(py);
                                }
                            });
                        }
                        crate::mailbox::Message::System(crate::mailbox::SystemMessage::Exit(_info)) => {
                            // nothing special
                        }
                        crate::mailbox::Message::System(crate::mailbox::SystemMessage::Ping)
                        | crate::mailbox::Message::System(crate::mailbox::SystemMessage::Pong) => {}
                    }
                }
            }
        };

        Ok(self.inner.spawn_handler_with_budget(handler, budget))
    }

    /// Bounded mailbox variant of spawn_py_handler.
    fn spawn_py_handler_bounded(
        &self,
        py_callable: PyObject,
        budget: usize,
        capacity: usize,
        release_gil: Option<bool>,
    ) -> PyResult<u64> {
        let release = release_gil.unwrap_or(false);
        let behavior = Arc::new(parking_lot::RwLock::new(py_callable));
        let maybe_tx = make_release_gil_channel(&self.inner, release, behavior.clone())?;

        let handler = move |mut rx: crate::mailbox::MailboxReceiver| {
            let maybe_tx = maybe_tx.clone();
            let behavior = behavior.clone();
            async move {
                while let Some(msg) = rx.recv().await {
                    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                        return;
                    }

                    if let Some(tx) = &maybe_tx {
                        match msg {
                            crate::mailbox::Message::User(bytes) => {
                                let task = PoolTask::Execute { behavior: behavior.clone(), bytes: bytes.clone() };
                                let _ = tx.send(task);
                            }
                            crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                                let task = PoolTask::HotSwap { behavior: behavior.clone(), ptr };
                                let _ = tx.send(task);
                            }
                            _ => {}
                        }
                    } else {
                        match msg {
                            crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                                Python::with_gil(|py| unsafe {
                                    let new_obj = PyObject::from_owned_ptr(
                                        py,
                                        ptr as *mut pyo3::ffi::PyObject,
                                    );
                                    let mut guard = behavior.write();
                                    *guard = new_obj;
                                });
                            }
                            crate::mailbox::Message::User(bytes) => {
                                Python::with_gil(|py| {
                                    let guard = behavior.read();
                                    let cb = guard.as_ref(py);
                                    let pybytes = PyBytes::new(py, &bytes);
                                    if let Err(e) = cb.call1((pybytes,)) {
                                        eprintln!("[Iris] Python actor exception: {}", e);
                                        e.print(py);
                                    }
                                });
                            }
                            _ => {}
                        }
                    }
                }
            }
        };

        Ok(self.inner.spawn_actor_with_budget_bounded(handler, budget, capacity))
    }

    /// Spawn a child actor; lifetime is tied to `parent`.
    fn spawn_child(
        &self,
        parent: u64,
        py_callable: PyObject,
        budget: Option<u32>,
        release_gil: Option<bool>,
    ) -> PyResult<u64> {
        let bud = budget.unwrap_or(100) as usize;
        // forward to the helper, propagating errors
        self.spawn_child_py_handler(parent, py_callable, bud, release_gil)
    }

    /// Variant of `spawn_py_handler` that ties the new actor to a `parent` PID.
    /// Structured concurrency: child is automatically stopped when the parent exits.
    fn spawn_child_py_handler(
        &self,
        parent: u64,
        py_callable: PyObject,
        budget: usize,
        release_gil: Option<bool>,
    ) -> PyResult<u64> {
        // Implementation mirrors spawn_py_handler but uses parent API.
        let release = release_gil.unwrap_or(false);
        let behavior = Arc::new(parking_lot::RwLock::new(py_callable));
        let maybe_tx = make_release_gil_channel(&self.inner, release, behavior.clone())?;

        let handler = move |msg: crate::mailbox::Message| {
            let maybe_tx = maybe_tx.clone();
            let behavior = behavior.clone();
            async move {
                if let Some(tx) = maybe_tx {
                    // blocking GIL thread path
                    match msg {
                        crate::mailbox::Message::User(bytes) => {
                            let task = PoolTask::Execute { behavior: behavior.clone(), bytes: bytes.clone() };
                            let _ = tx.send(task);
                        }
                        crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                            let task = PoolTask::HotSwap { behavior: behavior.clone(), ptr };
                            let _ = tx.send(task);
                        }
                        _ => {}
                    }
                } else {
                    match msg {
                        crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                            unsafe {
                                let new_obj = Python::with_gil(|py| {
                                PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject)
                            });
                                let mut guard = behavior.write();
                                *guard = new_obj;
                            }
                        }
                        crate::mailbox::Message::User(bytes) => {
                            Python::with_gil(|py| {
                                let guard = behavior.read();
                                let cb = guard.as_ref(py);
                                let pybytes = PyBytes::new(py, &bytes);
                                if let Err(e) = cb.call1((pybytes,)) {
                                    eprintln!("[Iris] Python actor exception: {}", e);
                                    e.print(py);
                                }
                            });
                        }
                        _ => {}
                    }
                }
            }
        };

        Ok(self.inner.spawn_child_handler_with_budget(parent, handler, budget))
    }

    /// Spawns a pull-based actor.
    /// The `py_callable` is called ONCE with a `PyMailbox` object in a dedicated OS thread.
    /// This mimics Erlang/Go style blocking actors without needing Python asyncio.
    fn spawn_with_mailbox(&self, py_callable: PyObject, budget: usize) -> PyResult<u64> {
        let pid = self.inner.spawn_actor_with_budget(
            move |rx| async move {
                let mailbox = PyMailbox {
                    inner: Arc::new(TokioMutex::new(rx)),
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
                            eprintln!("[Iris] Python mailbox actor exception: {}", e);
                            e.print(py);
                        }
                    });
                });

                // Await the thread's completion. This keeps the actor "alive" in the system
                // until the Python function returns.
                let _ = handle.await;
            },
            budget,
        );

        Ok(pid)
    }

    /// Spawn a child actor that uses a blocking Python mailbox loop.
    fn spawn_child_with_mailbox(&self, parent: u64, py_callable: PyObject, budget: usize) -> PyResult<u64> {
        let pid = self.inner.spawn_child_with_budget(parent, move |rx| async move {
            let mailbox = PyMailbox {
                inner: Arc::new(TokioMutex::new(rx)),
            };

            let handle = tokio::task::spawn_blocking(move || {
                if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                    return;
                }

                Python::with_gil(|py| {
                    if let Err(e) = py_callable.call1(py, (mailbox,)) {
                        eprintln!("[Iris] Python mailbox actor exception: {}", e);
                        e.print(py);
                    }
                });
            });

            let _ = handle.await;
        }, budget);

        Ok(pid)
    }

    fn send(&self, pid: u64, data: &PyBytes) -> PyResult<bool> {
        let msg = bytes::Bytes::copy_from_slice(data.as_bytes());
        Ok(self
        .inner
        .send(pid, crate::mailbox::Message::User(msg))
        .is_ok())
    }

    /// Schedule a one-shot send from Python. Returns a numeric timer id.
    fn send_after(&self, pid: u64, delay_ms: u64, data: &PyBytes) -> PyResult<u64> {
        let msg = bytes::Bytes::copy_from_slice(data.as_bytes());
        let id = self
        .inner
        .send_after(pid, delay_ms, crate::mailbox::Message::User(msg));
        Ok(id)
    }

    /// Schedule a repeating interval send from Python. Returns a numeric timer id.
    fn send_interval(&self, pid: u64, interval_ms: u64, data: &PyBytes) -> PyResult<u64> {
        let msg = bytes::Bytes::copy_from_slice(data.as_bytes());
        let id = self
        .inner
        .send_interval(pid, interval_ms, crate::mailbox::Message::User(msg));
        Ok(id)
    }

    /// Cancel a previously scheduled timer/interval. Returns True if cancelled.
    fn cancel_timer(&self, timer_id: u64) -> PyResult<bool> {
        Ok(self.inner.cancel_timer(timer_id))
    }

    /// Await selectively on observed messages for `pid` using a Python callable.
    fn selective_recv_observed_py<'py>(
        &self,
        py: Python<'py>,
        pid: u64,
        matcher: PyObject,
        timeout: Option<f64>,
    ) -> PyResult<&'py PyAny> {
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
                       Err(_) => Ok(Python::with_gil(|py| py.None())),
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

    fn mailbox_size(&self, pid: u64) -> PyResult<Option<usize>> {
        Ok(self.inner.mailbox_size(pid))
    }

    fn children_count(&self) -> usize {
        self.inner.supervisor().children_count()
    }

    fn child_pids(&self) -> Vec<u64> {
        self.inner.supervisor().child_pids()
    }

    fn link(&self, a: u64, b: u64) -> PyResult<()> {
        self.inner.link(a, b);
        Ok(())
    }

    fn unlink(&self, a: u64, b: u64) -> PyResult<()> {
        self.inner.unlink(a, b);
        Ok(())
    }

    fn watch(&self, pid: u64, strategy: &str) -> PyResult<()> {
        use crate::supervisor::ChildSpec;
        use crate::supervisor::RestartStrategy;
        use std::sync::Arc;

        let strat = match strategy.to_lowercase().as_str() {
            "restartone" | "restart_one" | "one" => RestartStrategy::RestartOne,
            "restartall" | "restart_all" | "all" => RestartStrategy::RestartAll,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid strategy")),
        };

        let spec = ChildSpec {
            factory: Arc::new(move || Ok(pid)),
            strategy: strat,
        };
        self.inner.supervisor().add_child(pid, spec);
        Ok(())
    }

    fn supervise_with_factory(
        &self,
        pid: u64,
        py_factory: PyObject,
        strategy: &str,
    ) -> PyResult<()> {
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
        let factory_closure: Arc<dyn Fn() -> Result<crate::pid::Pid, String> + Send + Sync> =
        Arc::new(move || {
            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                return Err("Interpreter shutting down".to_string());
            }
            Python::with_gil(|py| {
                let obj = factory_py.as_ref(py);
                match obj.call0() {
                    Ok(v) => match v.extract::<u64>() {
                        Ok(pid) => Ok(pid),
                             Err(e) => Err(e.to_string()),
                    },
                    Err(e) => Err(e.to_string()),
                }
            })
        });

        self.inner.supervise(pid, factory_closure, strat);
        Ok(())
    }

    /// Attach a Python factory to a path-scoped supervisor.
    fn path_supervise_with_factory(
        &self,
        path: String,
        pid: u64,
        py_factory: PyObject,
        strategy: &str,
    ) -> PyResult<()> {
        use std::sync::Arc;

        let strat = match strategy.to_lowercase().as_str() {
            "restartone" | "restart_one" | "one" => {
                crate::supervisor::RestartStrategy::RestartOne
            }
            "restartall" | "restart_all" | "all" => {
                crate::supervisor::RestartStrategy::RestartAll
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid strategy")),
        };

        // Validate we can call the factory once to obtain an initial pid
        let _initial_pid = Python::with_gil(|py| {
            let obj = py_factory.as_ref(py);
            let called = obj.call0()?;
            let pid: u64 = called.extract()?;
            Ok::<u64, pyo3::PyErr>(pid)
        })?;

        let factory_py = py_factory.clone();
        let factory_closure: Arc<dyn Fn() -> Result<crate::pid::Pid, String> + Send + Sync> =
        Arc::new(move || {
            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                return Err("Interpreter shutting down".to_string());
            }
            Python::with_gil(|py| {
                let obj = factory_py.as_ref(py);
                match obj.call0() {
                    Ok(v) => match v.extract::<u64>() {
                        Ok(pid) => Ok(pid),
                             Err(e) => Err(e.to_string()),
                    },
                    Err(e) => Err(e.to_string()),
                }
            })
        });

        self.inner
        .path_supervise_with_factory(&path, pid, factory_closure, strat);
        Ok(())
    }
}
