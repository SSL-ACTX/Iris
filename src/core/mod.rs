use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::time::Duration;

pub mod buffer;
pub mod logging;
pub mod mailbox;
pub mod network;
pub mod pid;
pub mod registry;
pub mod supervisor;

pub mod behavior;
pub mod send;
pub mod spawn;
pub mod types;

#[cfg(feature = "vortex")]
pub mod vortex;
#[cfg(feature = "vortex")]
pub mod vortex_rt;

use pid::Pid;
use types::VirtualActorSpec;

#[cfg(feature = "vortex")]
use vortex::{VortexEngine, VortexGhostPolicy};

/// A global, multi-threaded Tokio runtime shared by all Iris instances.
pub(crate) static RUNTIME: Lazy<TokioRuntime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Iris Tokio Runtime")
});

/// Lightweight runtime for spawning actors and managing distributed nodes.
#[derive(Clone)]
pub struct Runtime {
    pub(crate) slab: Arc<Mutex<pid::SlabAllocator>>,
    pub(crate) mailboxes: Arc<DashMap<Pid, mailbox::MailboxSender>>,
    pub(crate) supervisor: Arc<supervisor::Supervisor>,
    pub(crate) observers: Arc<DashMap<Pid, Arc<Mutex<Vec<mailbox::Message>>>>>,
    pub(crate) network: Arc<Mutex<Option<network::NetworkManager>>>,
    // network configuration (timeouts/limits/backoff)
    pub(crate) network_io_timeout: Arc<Mutex<Duration>>,
    pub(crate) network_max_payload: Arc<Mutex<usize>>,
    pub(crate) network_max_name_len: Arc<Mutex<usize>>,
    pub(crate) monitor_backoff_factor: Arc<Mutex<f64>>,
    pub(crate) monitor_backoff_max: Arc<Mutex<Duration>>,
    pub(crate) monitor_failure_threshold: Arc<Mutex<usize>>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_engine: Option<Arc<Mutex<VortexEngine>>>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_watcher: Option<Arc<vortex::VortexWatcher>>,
    pub(crate) registry: Arc<registry::NameRegistry>,
    /// Mapping for locally‑spawned proxies that forward to remote actors.
    pub(crate) remote_proxies: Arc<DashMap<Pid, (String, Pid)>>,
    /// Reverse lookup from (address, remote_pid) -> local proxy PID.
    pub(crate) proxy_by_remote: Arc<DashMap<(String, Pid), Pid>>,
    /// Behavior version per actor PID (starts at 1).
    pub(crate) behavior_versions: Arc<DashMap<Pid, u64>>,
    /// Recent hot-swapped pointers used for rollback (capped).
    pub(crate) behavior_history: Arc<DashMap<Pid, Vec<usize>>>,
    /// Optional per-path supervisors (shallow supervisors keyed by path).
    pub(crate) path_supervisors: Arc<DashMap<String, Arc<supervisor::Supervisor>>>,
    /// Maps a child PID to its parent PID for structured concurrency.
    pub(crate) parent_of: Arc<DashMap<Pid, Pid>>,
    /// Tracks direct children of each parent PID.
    pub(crate) children_by_parent: Arc<DashMap<Pid, Vec<Pid>>>,
    /// Capacity for bounded mailboxes.
    pub(crate) bounded_capacity: Arc<DashMap<Pid, usize>>,
    /// Overflow policies for bounded mailboxes; default is DropNew if absent.
    pub(crate) overflow_policy: Arc<DashMap<Pid, mailbox::OverflowPolicy>>,
    /// Lazy/virtual actor specs reserved by PID and activated on first send.
    pub(crate) virtual_specs: Arc<DashMap<Pid, VirtualActorSpec>>,
    /// Per-virtual-actor activation lock to prevent duplicate activation races.
    pub(crate) virtual_activate_locks: Arc<DashMap<Pid, Arc<Mutex<()>>>>,
    /// Track last known backpressure level for each pid, to emit signals on change.
    pub(crate) backpressure_state: Arc<DashMap<Pid, mailbox::BackpressureLevel>>,
    // Runtime-configurable limits for Python GIL-release behavior
    pub(crate) release_gil_max_threads: Arc<Mutex<usize>>,
    pub(crate) gil_pool_size: Arc<Mutex<usize>>,
    pub(crate) release_gil_strict: Arc<Mutex<bool>>,
    // Timers: map from timer id -> cancellation sender
    pub(crate) timers: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<()>>>>,
    pub(crate) timer_counter: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_ghost_counter: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_auto_replay_count: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_auto_primary_wins: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_auto_ghost_wins: Arc<AtomicU64>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_auto_policy: Arc<Mutex<VortexGhostPolicy>>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_genetic_budgeting_enabled: Arc<Mutex<bool>>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_genetic_thresholds: Arc<Mutex<(f64, f64)>>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_isolation_disallowed_ops: Arc<Mutex<std::collections::HashSet<u8>>>,
    #[cfg(feature = "vortex")]
    pub(crate) vortex_genetic_history: Arc<DashMap<Pid, (usize, usize)>>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    /// Create a new runtime instance and initialize the networking and registry sub-systems.
    pub fn new() -> Self {
        logging::init_logger();

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
            bounded_capacity: Arc::new(DashMap::new()),
            overflow_policy: Arc::new(DashMap::new()),
            backpressure_state: Arc::new(DashMap::new()),
            virtual_specs: Arc::new(DashMap::new()),
            virtual_activate_locks: Arc::new(DashMap::new()),
            release_gil_max_threads: Arc::new(Mutex::new(0)),
            gil_pool_size: Arc::new(Mutex::new(8)),
            release_gil_strict: Arc::new(Mutex::new(false)),
            timers: Arc::new(Mutex::new(HashMap::new())),
            timer_counter: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_ghost_counter: Arc::new(AtomicU64::new(1)),
            #[cfg(feature = "vortex")]
            vortex_auto_replay_count: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_auto_primary_wins: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_auto_ghost_wins: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "vortex")]
            vortex_auto_policy: Arc::new(Mutex::new(VortexGhostPolicy::FirstSafePointWins)),
            #[cfg(feature = "vortex")]
            vortex_genetic_budgeting_enabled: Arc::new(Mutex::new(false)),
            #[cfg(feature = "vortex")]
            vortex_genetic_thresholds: Arc::new(Mutex::new((0.4, 0.7))),
            #[cfg(feature = "vortex")]
            vortex_isolation_disallowed_ops: Arc::new(Mutex::new(std::collections::HashSet::new())),
            #[cfg(feature = "vortex")]
            vortex_genetic_history: Arc::new(DashMap::new()),
            network_io_timeout: Arc::new(Mutex::new(Duration::from_secs(5))),
            network_max_payload: Arc::new(Mutex::new(1024 * 1024)),
            network_max_name_len: Arc::new(Mutex::new(1024)),
            monitor_backoff_factor: Arc::new(Mutex::new(2.0)),
            monitor_backoff_max: Arc::new(Mutex::new(Duration::from_secs(60))),
            monitor_failure_threshold: Arc::new(Mutex::new(1)),
            remote_proxies: Arc::new(DashMap::new()),
            proxy_by_remote: Arc::new(DashMap::new()),
            behavior_versions: Arc::new(DashMap::new()),
            behavior_history: Arc::new(DashMap::new()),
            #[cfg(feature = "vortex")]
            vortex_engine: Some(Arc::new(Mutex::new(VortexEngine::new()))),
            #[cfg(feature = "vortex")]
            vortex_watcher: Some(Arc::new(vortex::VortexWatcher::new())),
        };

        let net_manager = network::NetworkManager::new(Arc::new(rt.clone()));
        *rt.network.lock().unwrap() = Some(net_manager);

        rt
    }

    /// Set runtime limits for GIL release handling.
    ///
    /// `max_threads = 0` forces pooled mode (no dedicated thread per actor).
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
}
