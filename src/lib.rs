// src/lib.rs
//! Iris — core runtime (Phase 1-7)
//!
//! This crate contains the core logic for PID allocation, mailboxes,
//! cooperative scheduling, distributed networking, name registration,
//! and remote service discovery.

pub mod buffer;
pub mod mailbox;
pub mod network;
pub mod pid;
pub mod registry;
pub mod supervisor;

#[cfg(feature = "pyo3")]
pub mod py;

#[cfg(feature = "node")]
pub mod node;

use crate::pid::Pid;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::time::Duration;

/// A global, multi-threaded Tokio runtime shared by all Iris instances.
static RUNTIME: Lazy<TokioRuntime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .build()
    .expect("Failed to create Iris Tokio Runtime")
});

/// Lightweight runtime for spawning actors and managing distributed nodes.
#[derive(Clone)]
pub struct Runtime {
    slab: Arc<Mutex<pid::SlabAllocator>>,
    mailboxes: Arc<DashMap<Pid, mailbox::MailboxSender>>,
    supervisor: Arc<supervisor::Supervisor>,
    observers: Arc<DashMap<Pid, Arc<Mutex<Vec<mailbox::Message>>>>>,
    network: Arc<Mutex<Option<network::NetworkManager>>>,
    // network configuration (timeouts/limits/backoff)
    network_io_timeout: Arc<Mutex<Duration>>,
    network_max_payload: Arc<Mutex<usize>>,
    network_max_name_len: Arc<Mutex<usize>>,
    monitor_backoff_factor: Arc<Mutex<f64>>,
    monitor_backoff_max: Arc<Mutex<Duration>>,
    monitor_failure_threshold: Arc<Mutex<usize>>,
    registry: Arc<registry::NameRegistry>,
    /// Optional per-path supervisors (shallow supervisors keyed by path).
    path_supervisors: Arc<DashMap<String, Arc<supervisor::Supervisor>>>,
    /// Maps a child PID to its parent PID for structured concurrency.
    parent_of: Arc<DashMap<Pid, Pid>>,
    /// Tracks direct children of each parent PID.
    children_by_parent: Arc<DashMap<Pid, Vec<Pid>>>,
    // Runtime-configurable limits for Python GIL-release behavior
    release_gil_max_threads: Arc<Mutex<usize>>,
    gil_pool_size: Arc<Mutex<usize>>,
    release_gil_strict: Arc<Mutex<bool>>,
    // Timers: map from timer id -> cancellation sender
    timers: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<()>>>>,
    timer_counter: Arc<AtomicU64>,
}

impl Runtime {
    /// Create a new runtime instance and initialize the networking and registry sub-systems.
    pub fn new() -> Self {
        #[cfg(feature = "pyo3")]
        {
            pyo3::prepare_freethreaded_python();
        }

        let rt = Runtime {
            slab: Arc::new(Mutex::new(pid::SlabAllocator::new())),
            mailboxes: Arc::new(DashMap::new()),
            supervisor: Arc::new(supervisor::Supervisor::new()),
            observers: Arc::new(DashMap::new()),
            network: Arc::new(Mutex::new(None)),
            registry: Arc::new(registry::NameRegistry::new()),
            path_supervisors: Arc::new(DashMap::new()),
            parent_of: Arc::new(DashMap::new()),
            children_by_parent: Arc::new(DashMap::new()),
            release_gil_max_threads: Arc::new(Mutex::new(256)),
            gil_pool_size: Arc::new(Mutex::new(8)),
            release_gil_strict: Arc::new(Mutex::new(false)),
            timers: Arc::new(Mutex::new(HashMap::new())),
            timer_counter: Arc::new(AtomicU64::new(0)),
            network_io_timeout: Arc::new(Mutex::new(Duration::from_secs(5))),
            network_max_payload: Arc::new(Mutex::new(1024 * 1024)),
            network_max_name_len: Arc::new(Mutex::new(1024)),
            monitor_backoff_factor: Arc::new(Mutex::new(2.0)),
            monitor_backoff_max: Arc::new(Mutex::new(Duration::from_secs(60))),
            monitor_failure_threshold: Arc::new(Mutex::new(1)),
        };

        let net_manager = network::NetworkManager::new(Arc::new(rt.clone()));
        *rt.network.lock().unwrap() = Some(net_manager);

        rt
    }

    /// Schedule a one-shot message to be sent after `delay_ms` milliseconds.
    /// Returns a timer id that can be used to cancel the pending send.
    pub fn send_after(&self, pid: Pid, delay_ms: u64, msg: mailbox::Message) -> u64 {
        let id = self.timer_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        self.timers.lock().unwrap().insert(id, tx);

        let rt_clone = self.clone();
        RUNTIME.spawn(async move {
            let sleep = tokio::time::sleep(std::time::Duration::from_millis(delay_ms));
            tokio::select! {
                _ = sleep => {
                    let _ = rt_clone.send(pid, msg);
                }
                _ = rx => {
                    // cancelled
                }
            }
            let _ = rt_clone.timers.lock().unwrap().remove(&id);
        });

        id
    }

    /// Schedule a repeating interval that sends `msg` every `interval_ms` milliseconds.
    /// Returns a timer id that can be used to cancel the interval.
    pub fn send_interval(&self, pid: Pid, interval_ms: u64, msg: mailbox::Message) -> u64 {
        let id = self.timer_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        self.timers.lock().unwrap().insert(id, tx);

        let rt_clone = self.clone();
        RUNTIME.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(interval_ms));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let _ = rt_clone.send(pid, msg.clone());
                    }
                    _ = &mut rx => {
                        break;
                    }
                }
            }
            let _ = rt_clone.timers.lock().unwrap().remove(&id);
        });

        id
    }

    /// Cancel a scheduled timer/interval. Returns true if a timer was cancelled.
    pub fn cancel_timer(&self, timer_id: u64) -> bool {
        if let Some(tx) = self.timers.lock().unwrap().remove(&timer_id) {
            let _ = tx.send(());
            true
        } else {
            false
        }
    }

    /// Set runtime limits for GIL release handling.
    pub fn set_release_gil_limits(&self, max_threads: usize, pool_size: usize) {
        *self.release_gil_max_threads.lock().unwrap() = max_threads;
        *self.gil_pool_size.lock().unwrap() = pool_size;
    }

    /// Enable or disable strict failure mode: when true, spawning an actor with
    /// `release_gil=true` will return an error if the dedicated-thread limit is exceeded.
    pub fn set_release_gil_strict(&self, strict: bool) {
        *self.release_gil_strict.lock().unwrap() = strict;
    }

    /// Get the current release_gil limits (max_threads, pool_size).
    pub fn get_release_gil_limits(&self) -> (usize, usize) {
        (
            *self.release_gil_max_threads.lock().unwrap(),
         *self.gil_pool_size.lock().unwrap(),
        )
    }

    /// Returns whether strict failure mode is enabled.
    pub fn is_release_gil_strict(&self) -> bool {
        *self.release_gil_strict.lock().unwrap()
    }

    // --- Name Registry ---

    /// Register a name for an actor locally.
    pub fn register(&self, name: String, pid: Pid) {
        self.registry.register(name, pid);
    }

    /// Unregister a named actor locally.
    /// This was missing and caused the compilation error.
    pub fn unregister(&self, name: &str) {
        self.registry.unregister(name);
    }

    /// Resolve a human-readable name to a PID.
    pub fn resolve(&self, name: &str) -> Option<Pid> {
        self.registry.resolve(name)
    }

    /// Set the network I/O timeout used by all operations.
    pub fn set_network_io_timeout(&self, t: Duration) {
        *self.network_io_timeout.lock().unwrap() = t;
    }

    /// Get configured network I/O timeout.
    pub fn get_network_io_timeout(&self) -> Duration {
        *self.network_io_timeout.lock().unwrap()
    }

    /// Adjust maximum allowed payload length for send_remote (bytes).
    pub fn set_network_max_payload(&self, bytes: usize) {
        *self.network_max_payload.lock().unwrap() = bytes;
    }

    /// Get current payload limit.
    pub fn get_network_max_payload(&self) -> usize {
        *self.network_max_payload.lock().unwrap()
    }

    /// Adjust maximum allowed name length for remote resolve.
    pub fn set_network_max_name_len(&self, bytes: usize) {
        *self.network_max_name_len.lock().unwrap() = bytes;
    }

    /// Get current name length limit.
    pub fn get_network_max_name_len(&self) -> usize {
        *self.network_max_name_len.lock().unwrap()
    }

    /// Configure exponential backoff parameters for `monitor_remote`.
    ///
    /// `factor` is multiplied after each failure, capped by `max`.
    /// `failure_threshold` is how many consecutive failures must occur before
    /// the supervisor is notified.
    pub fn set_monitor_backoff(&self, factor: f64, max: Duration, failure_threshold: usize) {
        *self.monitor_backoff_factor.lock().unwrap() = factor;
        *self.monitor_backoff_max.lock().unwrap() = max;
        *self.monitor_failure_threshold.lock().unwrap() = failure_threshold;
    }

    pub fn get_monitor_backoff_factor(&self) -> f64 {
        *self.monitor_backoff_factor.lock().unwrap()
    }
    pub fn get_monitor_backoff_max(&self) -> Duration {
        *self.monitor_backoff_max.lock().unwrap()
    }
    pub fn get_monitor_failure_threshold(&self) -> usize {
        *self.monitor_failure_threshold.lock().unwrap()
    }

    /// Register a hierarchical path for an actor PID.
    pub fn register_path(&self, path: String, pid: Pid) {
        self.registry.register(path, pid);
    }

    /// Unregister a hierarchical path.
    pub fn unregister_path(&self, path: &str) {
        self.registry.unregister(path);
    }

    /// Resolve a path to a PID (exact match).
    pub fn whereis_path(&self, path: &str) -> Option<Pid> {
        self.registry.resolve(path)
    }

    /// Create a path-scoped supervisor for `path`.
    pub fn create_path_supervisor(&self, path: &str) {
        self.path_supervisors
        .entry(path.to_string())
        .or_insert_with(|| Arc::new(supervisor::Supervisor::new()));
    }

    /// Remove a path-scoped supervisor if present.
    pub fn remove_path_supervisor(&self, path: &str) {
        self.path_supervisors.remove(path);
    }

    /// Watch a specific pid under a path-scoped supervisor if it exists,
    /// otherwise fall back to the global supervisor.
    pub fn path_supervisor_watch(&self, path: &str, pid: Pid) {
        if let Some(entry) = self.path_supervisors.get(path) {
            entry.watch(pid);
        } else {
            self.supervisor().watch(pid);
        }
    }

    /// Return child PIDs supervised by the path-scoped supervisor, if any.
    pub fn path_supervisor_children(&self, path: &str) -> Vec<Pid> {
        if let Some(entry) = self.path_supervisors.get(path) {
            entry.child_pids()
        } else {
            Vec::new()
        }
    }

    /// List registered entries under a path prefix.
    pub fn list_children(&self, prefix: &str) -> Vec<(String, Pid)> {
        self.registry.list_children(prefix)
    }

    /// List only direct children one level below `prefix`.
    pub fn list_children_direct(&self, prefix: &str) -> Vec<(String, Pid)> {
        self.registry.list_direct_children(prefix)
    }

    /// Watch all direct children under `prefix` (shallow watch).
    /// This is a convenience to register existing PIDs with the supervisor.
    pub fn watch_path(&self, prefix: &str) {
        let children = self.list_children_direct(prefix);
        for (_path, pid) in children {
            self.supervisor.watch(pid);
        }
    }

    /// Spawn an observed handler and register it under `path`.
    pub fn spawn_with_path_observed(&self, budget: usize, path: String) -> Pid {
        let pid = self.spawn_observed_handler(budget);
        self.register_path(path, pid);
        pid
    }

    /// Send a message to an actor by its registered name.
    pub fn send_named(&self, name: &str, msg: mailbox::Message) -> Result<(), String> {
        if let Some(pid) = self.resolve(name) {
            self.send(pid, msg).map_err(|_| "Send failed".to_string())
        } else {
            Err(format!("Name '{}' not found", name))
        }
    }

    // --- Distributed Networking ---

    /// Enable the node to receive remote messages on the specified TCP address.
    pub fn listen(&self, addr: String) {
        let rt_handle = Arc::new(self.clone());
        RUNTIME.spawn(async move {
            let manager = network::NetworkManager::new(rt_handle);
            match manager.start_server(&addr).await {
                Ok(actual) => tracing::info!(%actual, "node is now listening for remote messages"),
                Err(e) => eprintln!("[Iris] Network Server Error: {}", e),
            }
        });
    }

    /// Resolve a name on a remote node.
    /// This is an async call that queries the remote node's registry.
    pub async fn resolve_remote_async(&self, addr: String, name: String) -> Option<Pid> {
        let manager = network::NetworkManager::new(Arc::new(self.clone()));
        match manager.resolve_remote(&addr, &name).await {
            Ok(0) => None, // Node returned 0, meaning not found
            Ok(pid) => Some(pid),
            Err(e) => {
                eprintln!("[Iris] Remote Resolve Error: {}", e);
                None
            }
        }
    }

    /// Send a binary payload to a PID on a specific remote node.
    pub fn send_remote(&self, addr: String, pid: Pid, data: bytes::Bytes) {
        let rt_handle = Arc::new(self.clone());
        RUNTIME.spawn(async move {
            let manager = network::NetworkManager::new(rt_handle);
            if let Err(e) = manager.send_remote(&addr, pid, data).await {
                eprintln!("[Iris] Remote Send Error: {}", e);
            }
        });
    }

    /// Remote Monitoring with Heartbeat support.
    /// Periodically probes the remote node (default 1s interval) to detect failures.
    pub fn monitor_remote(&self, addr: String, pid: Pid) {
        let rt_handle = Arc::new(self.clone());
        RUNTIME.spawn(async move {
            let manager = network::NetworkManager::new(rt_handle.clone());
            // Probes node health at a 1000ms interval for silent failure detection
            manager.monitor_remote(addr, pid, 1000).await;
        });
    }

    // --- Lifecycle & Core Logic ---

    /// Stop an actor by closing its mailbox.
    pub fn stop(&self, pid: Pid) {
        self.mailboxes.remove(&pid);
    }

    /// Block the current thread until the actor with `pid` fully exits.
    pub fn wait(&self, pid: Pid) {
        RUNTIME.block_on(async {
            while self.is_alive(pid) {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        });
    }

    /// Send a Hot Swap signal to the actor.
    pub fn hot_swap(&self, pid: Pid, handler_ptr: usize) {
        if let Some(sender) = self.mailboxes.get(&pid) {
            let _ = sender.send_system(mailbox::SystemMessage::HotSwap(handler_ptr));
        }
    }

    pub fn spawn_actor<H, Fut>(&self, handler: H) -> Pid
    where
    H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(handler(rx));
            let res = actor_handle.await;

            // Determine exit reason and metadata
            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                      Err(e) => {
                          if e.is_panic() {
                              (crate::mailbox::ExitReason::Panic, Some(format!("join_error: {:?}", e)))
                          } else {
                              (crate::mailbox::ExitReason::Other("join_error".to_string()), Some(format!("join_error: {:?}", e)))
                          }
                      }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            // Notify any path-scoped supervisors that supervise this pid
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            // structured concurrency cleanup
            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo { from: pid, reason: reason.clone(), metadata: meta.clone() };
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    pub fn spawn_actor_with_budget<H, Fut>(&self, handler: H, _budget: usize) -> Pid
    where
    H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();
        let fut = handler(rx);

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(fut);
            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                      Err(e) => {
                          if e.is_panic() {
                              (crate::mailbox::ExitReason::Panic, Some(format!("join_error: {:?}", e)))
                          } else {
                              (crate::mailbox::ExitReason::Other("join_error".to_string()), Some(format!("join_error: {:?}", e)))
                          }
                      }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo { from: pid, reason: reason.clone(), metadata: meta.clone() };
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    pub fn spawn_handler_with_budget<H, Fut>(&self, handler: H, budget: usize) -> Pid
    where
    H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());

        let handler = std::sync::Arc::new(handler);
        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let h_loop = handler.clone();
            let actor_handle = tokio::spawn(async move {
                let mut processed = 0;
                while let Some(msg) = rx.recv().await {
                    let h = h_loop.clone();
                    (h)(msg).await;
                    
                    processed += 1;
                    if processed >= budget {
                        processed = 0;
                        tokio::task::yield_now().await;
                    }
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                      Err(e) => {
                          if e.is_panic() {
                              (crate::mailbox::ExitReason::Panic, Some(format!("join_error: {:?}", e)))
                          } else {
                              (crate::mailbox::ExitReason::Other("join_error".to_string()), Some(format!("join_error: {:?}", e)))
                          }
                      }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            // structured concurrency cleanup
            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo { from: pid, reason: reason.clone(), metadata: meta.clone() };
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    /// Spawn a new message-handler actor tied to `parent`.
    pub fn spawn_child_handler_with_budget<H, Fut>(&self, parent: Pid, handler: H, budget: usize) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_handler_with_budget(handler, budget);
        self.parent_of.insert(pid, parent);
        self.children_by_parent
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(pid);
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    pub fn spawn_observed_handler(&self, _budget: usize) -> Pid {
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        let vec = Arc::new(Mutex::new(Vec::new()));
        self.observers.insert(pid, vec.clone());

        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        RUNTIME.spawn(async move {
            let v_clone = vec.clone();
            let actor_handle = tokio::spawn(async move {
                while let Some(msg) = rx.recv().await {
                    {
                        let mut guard = v_clone.lock().unwrap();
                        guard.push(msg);
                    }
                    tokio::task::yield_now().await;
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                      Err(e) => {
                          if e.is_panic() {
                              (crate::mailbox::ExitReason::Panic, Some(format!("join_error: {:?}", e)))
                          } else {
                              (crate::mailbox::ExitReason::Other("join_error".to_string()), Some(format!("join_error: {:?}", e)))
                          }
                      }
            };

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            for entry in path_supervisors2.iter() {
                let sup = entry.value();
                if sup.contains_child(pid) {
                    sup.notify_exit(pid);
                }
            }
            slab2.lock().unwrap().deallocate(pid);

            rt_exit_clone.handle_exit_internal(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let info = crate::mailbox::ExitInfo { from: pid, reason: reason.clone(), metadata: meta.clone() };
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
                }
            }
        });

        pid
    }

    pub fn get_observed_messages(&self, pid: Pid) -> Option<Vec<mailbox::Message>> {
        self.observers
        .get(&pid)
        .map(|entry| entry.value().lock().unwrap().clone())
    }

    /// Remove and return a single observed message matching the predicate.
    /// Used by FFI helpers to implement selective receive for observed actors.
    pub fn take_observed_message_matching<F>(
        &self,
        pid: Pid,
        mut matcher: F,
    ) -> Option<mailbox::Message>
    where
    F: FnMut(&mailbox::Message) -> bool,
    {
        if let Some(entry) = self.observers.get(&pid) {
            let mut guard = entry.value().lock().unwrap();
            if let Some(pos) = guard.iter().position(|m| matcher(m)) {
                return Some(guard.remove(pos));
            }
        }
        None
    }

    pub fn send(&self, pid: Pid, msg: mailbox::Message) -> Result<(), mailbox::Message> {
        if let Some(sender) = self.mailboxes.get(&pid) {
            sender.send(msg)
        } else {
            Err(msg)
        }
    }

    /// Return the number of queued user messages for the actor with `pid`.
    pub fn mailbox_size(&self, pid: Pid) -> Option<usize> {
        self.mailboxes.get(&pid).map(|s| s.len())
    }

    pub fn is_alive(&self, pid: Pid) -> bool {
        let slab = self.slab.lock().unwrap();
        slab.is_valid(pid)
    }

    pub fn supervisor(&self) -> Arc<supervisor::Supervisor> {
        self.supervisor.clone()
    }

    pub fn supervise(
        &self,
        pid: Pid,
        factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>,
                     strategy: supervisor::RestartStrategy,
    ) {
        let spec = supervisor::ChildSpec { factory, strategy };
        self.supervisor.add_child(pid, spec);
    }

    /// Spawn an actor whose lifetime is tied to `parent`.
    /// When the parent PID exits (normal or crash) the child will be
    /// automatically stopped as well.  The returned PID behaves just like
    /// one created with `spawn_actor`.
    pub fn spawn_child<H, Fut>(&self, parent: Pid, handler: H) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_actor(handler);
        // record relationships for structured concurrency
        self.parent_of.insert(pid, parent);
        self.children_by_parent
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(pid);
        // if parent is already dead, immediately stop the child
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    /// Same as `spawn_child` but accepts a budget for cooperative scheduling.
    pub fn spawn_child_with_budget<H, Fut>(&self, parent: Pid, handler: H, budget: usize) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_actor_with_budget(handler, budget);
        self.parent_of.insert(pid, parent);
        self.children_by_parent
            .entry(parent)
            .or_insert_with(Vec::new)
            .push(pid);
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    /// Attach a factory-based child spec to a path-scoped supervisor.
    pub fn path_supervise_with_factory(
        &self,
        path: &str,
        pid: Pid,
        factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>,
                                       strategy: supervisor::RestartStrategy,
    ) {
        let spec = supervisor::ChildSpec { factory, strategy };
        let entry = self
        .path_supervisors
        .entry(path.to_string())
        .or_insert_with(|| Arc::new(supervisor::Supervisor::new()));
        entry.add_child(pid, spec);
    }

    pub fn link(&self, a: Pid, b: Pid) {
        self.supervisor.link(a, b);
    }

    /// Internal helper invoked when any actor exits to maintain parent/child
    /// state and to enforce structured concurrency.  This is called from
    /// each spawn_* helper after the actor has torn down its mailbox and been
    /// deallocated.
    fn handle_exit_internal(&self, pid: Pid) {
        // remove the pid from its parent's child list (if any)
        if let Some((_, parent)) = self.parent_of.remove(&pid) {
            if let Some(mut entry) = self.children_by_parent.get_mut(&parent) {
                entry.retain(|&p| p != pid);
                if entry.is_empty() {
                    self.children_by_parent.remove(&parent);
                }
            }
        }

        // if this pid itself is a parent, kill its current children
        if let Some((_, children)) = self.children_by_parent.remove(&pid) {
            for child in children {
                // drop reverse mapping and close mailbox to stop actor
                let _ = self.parent_of.remove(&child);
                self.mailboxes.remove(&child);
            }
        }
    }

    pub fn unlink(&self, a: Pid, b: Pid) {
        self.supervisor.unlink(a, b);
    }
}
