// src/lib.rs
//! Myrmidon — core runtime (Phase 1-7)
//!
//! This crate contains the core logic for PID allocation, mailboxes,
//! cooperative scheduling, distributed networking, name registration,
//! and remote service discovery.

pub mod pid;
pub mod mailbox;
pub mod scheduler;
pub mod supervisor;
pub mod buffer;
pub mod network;
pub mod registry;

#[cfg(feature = "pyo3")]
pub mod py;

#[cfg(feature = "node")]
pub mod node;

use crate::pid::Pid;
use dashmap::DashMap;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use tokio::runtime::Runtime as TokioRuntime;

/// A global, multi-threaded Tokio runtime shared by all Myrmidon instances.
static RUNTIME: Lazy<TokioRuntime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .build()
    .expect("Failed to create Myrmidon Tokio Runtime")
});

/// Lightweight runtime for spawning actors and managing distributed nodes.
#[derive(Clone)]
pub struct Runtime {
    slab: Arc<Mutex<pid::SlabAllocator>>,
    mailboxes: Arc<DashMap<Pid, mailbox::MailboxSender>>,
    supervisor: Arc<supervisor::Supervisor>,
    observers: Arc<DashMap<Pid, Arc<Mutex<Vec<mailbox::Message>>>>>,
    network: Arc<Mutex<Option<network::NetworkManager>>>,
    registry: Arc<registry::NameRegistry>,
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
        };

        let net_manager = network::NetworkManager::new(Arc::new(rt.clone()));
        *rt.network.lock().unwrap() = Some(net_manager);

        rt
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
            if let Err(e) = manager.start_server(&addr).await {
                eprintln!("[Myrmidon] Network Server Error: {}", e);
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
                eprintln!("[Myrmidon] Remote Resolve Error: {}", e);
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
                eprintln!("[Myrmidon] Remote Send Error: {}", e);
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

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(handler(rx));
            let _ = actor_handle.await;

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            slab2.lock().unwrap().deallocate(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(pid)));
                }
            }
        });

        pid
    }

    pub fn spawn_actor_with_budget<H, Fut>(&self, handler: H, budget: usize) -> Pid
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
        let fut = handler(rx);
        let limited = crate::scheduler::ReductionLimiter::new(fut, budget);

        RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(limited);
            let _ = actor_handle.await;

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            slab2.lock().unwrap().deallocate(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(pid)));
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

        RUNTIME.spawn(async move {
            let h_loop = handler.clone();
            let actor_handle = tokio::spawn(async move {
                while let Some(msg) = rx.recv().await {
                    let h = h_loop.clone();
                    let fut = (h)(msg);
                    let limited = crate::scheduler::ReductionLimiter::new(fut, budget);
                    limited.await;
                }
            });

            let _ = actor_handle.await;

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            slab2.lock().unwrap().deallocate(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(pid)));
                }
            }
        });

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

            let _ = actor_handle.await;

            mailboxes2.remove(&pid);
            supervisor2.notify_exit(pid);
            slab2.lock().unwrap().deallocate(pid);

            let linked = supervisor2.linked_pids(pid);
            for lp in linked {
                if let Some(sender) = mailboxes2.get(&lp) {
                    let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(pid)));
                }
            }
        });

        pid
    }

    pub fn get_observed_messages(&self, pid: Pid) -> Option<Vec<mailbox::Message>> {
        self.observers.get(&pid).map(|entry| entry.value().lock().unwrap().clone())
    }

    /// Remove and return a single observed message matching the predicate.
    /// Used by FFI helpers to implement selective receive for observed actors.
    pub fn take_observed_message_matching<F>(&self, pid: Pid, mut matcher: F) -> Option<mailbox::Message>
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

    pub fn is_alive(&self, pid: Pid) -> bool {
        let slab = self.slab.lock().unwrap();
        slab.is_valid(pid)
    }

    pub fn supervisor(&self) -> Arc<supervisor::Supervisor> {
        self.supervisor.clone()
    }

    pub fn supervise(&self, pid: Pid, factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>, strategy: supervisor::RestartStrategy) {
        let spec = supervisor::ChildSpec { factory, strategy };
        self.supervisor.add_child(pid, spec);
    }

    pub fn link(&self, a: Pid, b: Pid) {
        self.supervisor.link(a, b);
    }

    pub fn unlink(&self, a: Pid, b: Pid) {
        self.supervisor.unlink(a, b);
    }
}
