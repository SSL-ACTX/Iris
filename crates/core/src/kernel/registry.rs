use crate::mailbox;
use crate::pid::Pid;
use crate::supervisor;
use crate::Runtime;
use dashmap::DashMap;
use std::sync::Arc;

pub struct NameRegistry {
    /// Mapping of human-readable names to PIDs.
    names: DashMap<String, Pid>,
}

impl NameRegistry {
    /// Create a new, empty name registry.
    pub fn new() -> Self {
        Self {
            names: DashMap::new(),
        }
    }
}

impl Default for NameRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NameRegistry {
    /// Register a PID under a specific name.
    /// If the name already exists, it will be overwritten.
    pub fn register(&self, name: String, pid: Pid) {
        self.names.insert(name, pid);
    }

    /// Retrieve the PID associated with a name.
    pub fn resolve(&self, name: &str) -> Option<Pid> {
        self.names.get(name).map(|p| *p)
    }

    /// Remove a name mapping.
    pub fn unregister(&self, name: &str) {
        self.names.remove(name);
    }

    /// List registered entries whose path begins with `prefix`.
    /// If `prefix` is empty or "/" this returns all entries.
    pub fn list_children(&self, prefix: &str) -> Vec<(String, Pid)> {
        let mut out = Vec::new();
        let norm = if prefix.is_empty() || prefix == "/" {
            String::new()
        } else if prefix.ends_with('/') {
            prefix.trim_end_matches('/').to_string()
        } else {
            prefix.to_string()
        };

        let matcher = if norm.is_empty() {
            None
        } else {
            Some(format!("{}/", norm))
        };

        for r in self.names.iter() {
            let key = r.key();
            if let Some(ref m) = matcher {
                if key.starts_with(m) {
                    out.push((key.clone(), *r.value()));
                }
            } else {
                out.push((key.clone(), *r.value()));
            }
        }

        out
    }

    /// List only direct children one level below `prefix`.
    /// For prefix `/a/b` entries returned will be `/a/b/child` but not `/a/b/child/grand`.
    pub fn list_direct_children(&self, prefix: &str) -> Vec<(String, Pid)> {
        let mut out = Vec::new();
        let norm = if prefix.is_empty() || prefix == "/" {
            String::new()
        } else if prefix.ends_with('/') {
            prefix.trim_end_matches('/').to_string()
        } else {
            prefix.to_string()
        };

        let matcher = if norm.is_empty() {
            String::new()
        } else {
            format!("{}/", norm)
        };

        for r in self.names.iter() {
            let key = r.key();
            if !matcher.is_empty() {
                if !key.starts_with(&matcher) {
                    continue;
                }
                let tail = &key[matcher.len()..];
                if tail.contains('/') {
                    continue; // deeper than one level
                }
                out.push((key.clone(), *r.value()));
            } else {
                // root: direct children are top-level entries without additional '/'
                let tail = key.strip_prefix('/').unwrap_or(key);
                if tail.contains('/') {
                    continue;
                }
                out.push((key.clone(), *r.value()));
            }
        }

        out
    }
}

impl Runtime {
    // --- Name Registry ---

    /// Register a name for an actor locally.
    pub fn register(&self, name: String, pid: Pid) {
        self.registry.0.register(name, pid);
    }

    /// Unregister a named actor locally.
    pub fn unregister(&self, name: &str) {
        self.registry.0.unregister(name);
    }

    /// Resolve a human-readable name to a PID.
    pub fn resolve(&self, name: &str) -> Option<Pid> {
        self.registry.0.resolve(name)
    }

    /// Register a hierarchical path for an actor PID.
    pub fn register_path(&self, path: String, pid: Pid) {
        self.registry.0.register(path, pid);
    }

    /// Unregister a hierarchical path.
    pub fn unregister_path(&self, path: &str) {
        self.registry.0.unregister(path);
    }

    /// Resolve a path to a PID (exact match).
    pub fn whereis_path(&self, path: &str) -> Option<Pid> {
        self.registry.0.resolve(path)
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
            entry.value().watch(pid);
        } else {
            self.supervisor.0.clone().watch(pid);
        }
    }

    /// Return child PIDs supervised by the path-scoped supervisor, if any.
    pub fn path_supervisor_children(&self, path: &str) -> Vec<Pid> {
        if let Some(entry) = self.path_supervisors.get(path) {
            entry.value().child_pids()
        } else {
            Vec::new()
        }
    }

    /// List registered entries under a path prefix.
    pub fn list_children(&self, prefix: &str) -> Vec<(String, Pid)> {
        self.registry.0.list_children(prefix)
    }

    /// List only direct children one level below `prefix`.
    pub fn list_children_direct(&self, prefix: &str) -> Vec<(String, Pid)> {
        self.registry.0.list_direct_children(prefix)
    }

    /// Watch all direct children under `prefix` (shallow watch).
    /// This is a convenience to register existing PIDs with the supervisor.
    pub fn watch_path(&self, prefix: &str) {
        let children = self.list_children_direct(prefix);
        for (_path, pid) in children {
            self.supervisor.0.watch(pid);
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
        entry.value().add_child(pid, spec);
    }
}
