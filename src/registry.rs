// src/registry.rs
//! Phase 6: Name Registry
//! Allows actors to be referenced by string names instead of raw PIDs.

use crate::pid::Pid;
use dashmap::DashMap;

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
}
