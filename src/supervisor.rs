// supervisor.rs
//! Supervisor
//!
//! Adds small, testable supervision behaviors used by the runtime. Each child
//! can be registered with a `ChildSpec` (factory + restart strategy). When a
//! watched child exits the supervisor may restart the single child (one-for-one)
//! or restart the whole supervised group (one-for-all).

use dashmap::DashMap;
use crate::pid::Pid;
use std::sync::{Arc, Mutex};

/// Restart strategies supported in Phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Restart only the failed child.
    RestartOne,
    /// Restart all children supervised by this supervisor.
    RestartAll,
}

/// A child specification holds a factory used to (re)spawn the child and the
/// restart strategy to apply when the child exits.
#[derive(Clone)]
pub struct ChildSpec {
    /// The factory may fail; we return Result<Pid, String> so callers can
    /// surface human-friendly error messages when a factory invocation fails.
    pub factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>,
    pub strategy: RestartStrategy,
}

/// Supervisor behavior notes:
/// - Factories are fallible and return `Result<Pid,String>`; when a factory
///   fails during a restart we log the failure and skip restarting that child.
/// - This design prevents panics during supervisor restarts caused by Python
///   or other foreign code used as factories; callers should ensure factories
///   return informative error strings to ease debugging.

/// Supervisor stores child specs keyed by `Pid`.
#[derive(Default)]
pub struct Supervisor {
    // Wrapped in Arc so they can be shared with background restart tasks
    children: Arc<DashMap<Pid, ChildSpec>>,
    /// Recent errors recorded while attempting to restart children.
    errors: Arc<Mutex<Vec<String>>>,
    /// Bidirectional links between PIDs. If A is linked to B, and A exits,
    /// B should receive an exit signal (delivered by the Runtime).
    links: Arc<DashMap<Pid, Vec<Pid>>>,
}

impl Supervisor {
    /// Create a supervisor instance.
    pub fn new() -> Self {
        Supervisor {
            children: Arc::new(DashMap::new()),
            errors: Arc::new(Mutex::new(Vec::new())),
            links: Arc::new(DashMap::new()),
        }
    }

    /// Add a child with an explicit child spec (factory + strategy).
    pub fn add_child(&self, pid: Pid, spec: ChildSpec) {
        self.children.insert(pid, spec);
    }

    /// Remove a child from supervision.
    pub fn remove_child(&self, pid: Pid) {
        self.children.remove(&pid);
        // cleanup any links involving this pid
        if let Some((_, v)) = self.links.remove(&pid) {
            for other in v {
                if let Some(mut entry) = self.links.get_mut(&other) {
                    entry.retain(|&p| p != pid);
                }
            }
        }
    }

    /// Remove a bidirectional link between two PIDs.
    pub fn unlink(&self, a: Pid, b: Pid) {
        if let Some(mut entry) = self.links.get_mut(&a) {
            entry.retain(|&p| p != b);
        }
        if let Some(mut entry) = self.links.get_mut(&b) {
            entry.retain(|&p| p != a);
        }
    }

    /// Backwards-compatible `watch` that simply inserts a default ChildSpec.
    /// Useful for tests / simple use-cases.
    pub fn watch(&self, pid: Pid) {
        let spec = ChildSpec { factory: Arc::new(move || Ok(pid)), strategy: RestartStrategy::RestartOne };
        self.children.insert(pid, spec);
    }

    /// Establish a bidirectional link between two PIDs.
    pub fn link(&self, a: Pid, b: Pid) {
        self.links.entry(a).or_insert_with(Vec::new).push(b);
        self.links.entry(b).or_insert_with(Vec::new).push(a);
    }

    /// Return linked pids for `pid` (snapshot).
    pub fn linked_pids(&self, pid: Pid) -> Vec<Pid> {
        self.links.get(&pid).map(|v| v.clone()).unwrap_or_default()
    }

    /// Stop watching a pid.
    pub fn unwatch(&self, pid: Pid) {
        self.children.remove(&pid);
    }

    /// Query helpers for tests/observability.
    pub fn contains_child(&self, pid: Pid) -> bool {
        self.children.contains_key(&pid)
    }

    pub fn children_count(&self) -> usize {
        self.children.len()
    }

    pub fn child_pids(&self) -> Vec<Pid> {
        self.children.iter().map(|kv| *kv.key()).collect()
    }

    /// Return a snapshot of recent supervisor error messages.
    pub fn errors(&self) -> Vec<String> {
        self.errors.lock().unwrap().clone()
    }

    /// Called by the runtime when a child exits. Applies the restart strategy
    /// recorded in the child's `ChildSpec` (if any).
    pub fn notify_exit(&self, pid: Pid) {
        // If the PID is not in our map, it might be a stale notification from
        // an actor that was already replaced during a RestartAll event.
        let spec = match self.children.get(&pid) {
            Some(s) => s.clone(),
            None => return,
        };
        tracing::info!("[supervisor] notify_exit(pid={}) strategy={:?}", pid, spec.strategy);

        // Perform restart logic asynchronously to avoid blocking the caller and
        // to allow introducing backoff/retries for flaky factories.
        let children = self.children.clone();
        let errors = self.errors.clone();
        let spec_clone = spec.clone();
        let links = self.links.clone();

        tokio::spawn(async move {
            // retry policy: up to 3 attempts with exponential backoff starting at 100ms
            let mut attempts: u32;
            let max_attempts = 3u32;
            let mut backoff_ms = 100u64;

            // Helper to clean up links for a dead PID
            let cleanup_links = |dead_pid: Pid, links_map: &DashMap<Pid, Vec<Pid>>| {
                if let Some((_, v)) = links_map.remove(&dead_pid) {
                    for other in v {
                        if let Some(mut entry) = links_map.get_mut(&other) {
                            entry.retain(|&p| p != dead_pid);
                        }
                    }
                }
            };

            match spec_clone.strategy {
                RestartStrategy::RestartAll => {
                    // Capture current specs/pids
                    let all: Vec<(Pid, ChildSpec)> = children.iter().map(|kv| (*kv.key(), kv.value().clone())).collect();
                    // clear map to avoid nested restarts
                    for (k, _) in &all { children.remove(k); }

                    for (orig_pid, s) in all {
                        // Cleanup links for the old PID as it is now dead
                        cleanup_links(orig_pid, &links);

                        attempts = 0;
                        loop {
                            attempts += 1;
                            match (s.factory)() {
                                Ok(new_pid) => { children.insert(new_pid, s.clone()); break; }
                                Err(err) => {
                                    tracing::error!("[supervisor] factory failed during RestartAll attempt={} err={}", attempts, err);
                                    {
                                        let mut guard = errors.lock().unwrap();
                                        guard.push(err.clone());
                                    } // drop guard before awaiting
                                    if attempts >= max_attempts { break; }
                                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                                    backoff_ms = backoff_ms.saturating_mul(2);
                                }
                            }
                        }
                    }
                }
                RestartStrategy::RestartOne => {
                    // Remove the specific failed PID (if present) and attempt restart with retries
                    if let Some((_, s)) = children.remove(&pid) {
                        // Cleanup links for the dead PID
                        cleanup_links(pid, &links);

                        attempts = 0;
                        loop {
                            attempts += 1;
                            match (s.factory)() {
                                Ok(new_pid) => { children.insert(new_pid, s.clone()); break; }
                                Err(err) => {
                                    tracing::error!("[supervisor] factory failed during RestartOne attempt={} err={}", attempts, err);
                                    {
                                        let mut guard = errors.lock().unwrap();
                                        guard.push(err.clone());
                                    } // drop guard before awaiting
                                    if attempts >= max_attempts { break; }
                                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                                    backoff_ms = backoff_ms.saturating_mul(2);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber;

    #[test]
    fn watch_and_unwatch() {
        let _ = tracing_subscriber::fmt::try_init();
        let s = Supervisor::new();
        s.watch(1);
        assert!(s.children.contains_key(&1));
        s.unwatch(1);
        assert!(!s.children.contains_key(&1));
    }

    #[tokio::test]
    async fn factory_failure_skips_restart() {
        let _ = tracing_subscriber::fmt::try_init();
        let s = Supervisor::new();
        // Insert a child whose factory always fails
        let bad_factory = Arc::new(move || Err::<Pid, String>("boom".to_string()));
        let spec = ChildSpec { factory: bad_factory, strategy: RestartStrategy::RestartOne };
        s.add_child(42, spec);

        // Notify exit: factory should fail and the children map should be empty
        s.notify_exit(42);

        // Wait for the supervisor task to process the failure.
        let mut attempts = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            attempts += 1;

            let no_children = s.children_count() == 0;
            let has_errors = !s.errors().is_empty();

            if no_children && has_errors {
                break;
            }

            if attempts > 30 {
                panic!("Timeout waiting for supervisor: children_count={} errors={}", s.children_count(), s.errors().len());
            }
        }

        // ensure the failure was recorded correctly
        let errs = s.errors();
        assert!(errs[0].contains("boom"));
    }
}
