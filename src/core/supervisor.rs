//! Supervisor
//!
//! Adds small, testable supervision behaviors used by the runtime. Each child
//! can be registered with a `ChildSpec` (factory + restart strategy). When a
//! watched child exits the supervisor may restart the single child (one-for-one)
//! or restart the whole supervised group (one-for-all).

use super::Runtime;
use crate::core::mailbox;
use crate::core::pid::{self, Pid};
use dashmap::{DashMap, DashSet};
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
///
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
    /// Tracks PIDs currently undergoing a restart to debounce duplicate exit signals
    /// and prevent cascading `RestartAll` loops.
    restarting: Arc<DashSet<Pid>>,
}

impl Supervisor {
    fn push_unique_link(links: &DashMap<Pid, Vec<Pid>>, a: Pid, b: Pid) {
        let mut entry = links.entry(a).or_default();
        if !entry.contains(&b) {
            entry.push(b);
        }
    }

    /// Create a supervisor instance.
    pub fn new() -> Self {
        Supervisor {
            children: Arc::new(DashMap::new()),
            errors: Arc::new(Mutex::new(Vec::new())),
            links: Arc::new(DashMap::new()),
            restarting: Arc::new(DashSet::new()),
        }
    }

    /// Add a child with an explicit child spec (factory + strategy).
    pub fn add_child(&self, pid: Pid, spec: ChildSpec) {
        self.children.insert(pid, spec);
    }

    /// Remove a child from supervision.
    pub fn remove_child(&self, pid: Pid) {
        self.children.remove(&pid);
        self.restarting.remove(&pid);
        self.cleanup_links_internal(pid);
    }

    /// Remove a bidirectional link between two PIDs.
    pub fn unlink(&self, a: Pid, b: Pid) {
        if let Some(mut entry) = self.links.get_mut(&a) {
            entry.retain(|&p| p != b);
            if entry.is_empty() {
                drop(entry);
                self.links.remove(&a);
            }
        }
        if let Some(mut entry) = self.links.get_mut(&b) {
            entry.retain(|&p| p != a);
            if entry.is_empty() {
                drop(entry);
                self.links.remove(&b);
            }
        }
    }

    /// Backwards-compatible `watch` that simply inserts a default ChildSpec.
    /// Useful for tests / simple use-cases.
    pub fn watch(&self, pid: Pid) {
        let spec = ChildSpec {
            factory: Arc::new(move || Ok(pid)),
            strategy: RestartStrategy::RestartOne,
        };
        self.children.insert(pid, spec);
    }

    /// Establish a bidirectional link between two PIDs.
    pub fn link(&self, a: Pid, b: Pid) {
        Self::push_unique_link(&self.links, a, b);
        Self::push_unique_link(&self.links, b, a);
    }

    /// Retrieve and remove the PIDs linked to `pid`.
    ///
    /// This method is destructive: it assumes the actor `pid` is dead or dying.
    /// It removes `pid` from the links map and also removes `pid` from the
    /// link lists of all its peers to prevent memory leaks and stale references.
    pub fn linked_pids(&self, pid: Pid) -> Vec<Pid> {
        if let Some((_, linked_peers)) = self.links.remove(&pid) {
            for peer in &linked_peers {
                if let Some(mut entry) = self.links.get_mut(peer) {
                    entry.retain(|&p| p != pid);
                }
            }
            linked_peers
        } else {
            Vec::new()
        }
    }

    /// Internal helper to cleanup links without returning them.
    fn cleanup_links_internal(&self, pid: Pid) {
        if let Some((_, linked_peers)) = self.links.remove(&pid) {
            for peer in linked_peers {
                if let Some(mut entry) = self.links.get_mut(&peer) {
                    entry.retain(|&p| p != pid);
                }
            }
        }
    }

    /// Stop watching a pid.
    pub fn unwatch(&self, pid: Pid) {
        self.children.remove(&pid);
        self.restarting.remove(&pid);
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
    pub fn notify_exit(&self, pid: Pid, reason: &mailbox::ExitReason) {
        // Only restart if the exit was not a normal shutdown.
        if matches!(reason, mailbox::ExitReason::Normal) {
            self.children.remove(&pid);
            self.restarting.remove(&pid);
            return;
        }

        // Debounce: If we are already restarting this PID, safely ignore the duplicate exit signal.
        if !self.restarting.insert(pid) {
            return;
        }

        let spec = match self.children.get(&pid) {
            Some(s) => s.clone(),
            None => {
                self.restarting.remove(&pid);
                return;
            }
        };

        tracing::info!(
            "[supervisor] notify_exit(pid={}) reason={:?} strategy={:?}",
            pid,
            reason,
            spec.strategy
        );

        let children = self.children.clone();
        let errors = self.errors.clone();
        let links = self.links.clone();
        let restarting = self.restarting.clone();

        match spec.strategy {
            RestartStrategy::RestartAll => {
                let all: Vec<(Pid, ChildSpec)> = children
                    .iter()
                    .map(|kv| (*kv.key(), kv.value().clone()))
                    .collect();

                // Mark the entire group as restarting to prevent cascaded exit signals
                // from spawning redundant RestartAll waves.
                for (p, _) in &all {
                    restarting.insert(*p);
                }

                // Spawn concurrent restart tasks without dropping the supervisor's registry count.
                for (orig_pid, s) in all {
                    let children_clone = children.clone();
                    let errors_clone = errors.clone();
                    let links_clone = links.clone();
                    let restarting_clone = restarting.clone();

                    tokio::spawn(async move {
                        let mut attempts = 0;
                        let max_attempts = 3;
                        let mut backoff_ms = 100;

                        loop {
                            attempts += 1;
                            match (s.factory)() {
                                Ok(new_pid) => {
                                    // Atomic swap: insert the new PID, then clean up the old one.
                                    children_clone.insert(new_pid, s.clone());
                                    children_clone.remove(&orig_pid);

                                    if let Some((_, v)) = links_clone.remove(&orig_pid) {
                                        for other in v {
                                            if let Some(mut entry) = links_clone.get_mut(&other) {
                                                entry.retain(|&p| p != orig_pid);
                                            }
                                        }
                                    }
                                    restarting_clone.remove(&orig_pid);
                                    break;
                                }
                                Err(err) => {
                                    tracing::error!("[supervisor] factory failed during RestartAll attempt={} err={}", attempts, err);
                                    {
                                        let mut guard = errors_clone.lock().unwrap();
                                        guard.push(err.clone());
                                    }

                                    if attempts >= max_attempts {
                                        tracing::error!("[supervisor] child permanently dropped after exhausting retries (RestartAll) err={}", err);
                                        children_clone.remove(&orig_pid);
                                        if let Some((_, v)) = links_clone.remove(&orig_pid) {
                                            for other in v {
                                                if let Some(mut entry) = links_clone.get_mut(&other)
                                                {
                                                    entry.retain(|&p| p != orig_pid);
                                                }
                                            }
                                        }
                                        restarting_clone.remove(&orig_pid);
                                        break;
                                    }

                                    tokio::time::sleep(std::time::Duration::from_millis(
                                        backoff_ms,
                                    ))
                                    .await;
                                    backoff_ms = backoff_ms.saturating_mul(2);
                                }
                            }
                        }
                    });
                }
            }
            RestartStrategy::RestartOne => {
                let children_clone = children.clone();
                let errors_clone = errors.clone();
                let links_clone = links.clone();
                let restarting_clone = restarting.clone();

                tokio::spawn(async move {
                    let mut attempts = 0;
                    let max_attempts = 3;
                    let mut backoff_ms = 100;

                    loop {
                        attempts += 1;
                        match (spec.factory)() {
                            Ok(new_pid) => {
                                // Atomic swap
                                children_clone.insert(new_pid, spec.clone());
                                children_clone.remove(&pid);

                                if let Some((_, v)) = links_clone.remove(&pid) {
                                    for other in v {
                                        if let Some(mut entry) = links_clone.get_mut(&other) {
                                            entry.retain(|&p| p != pid);
                                        }
                                    }
                                }
                                restarting_clone.remove(&pid);
                                break;
                            }
                            Err(err) => {
                                tracing::error!("[supervisor] factory failed during RestartOne attempt={} err={}", attempts, err);
                                {
                                    let mut guard = errors_clone.lock().unwrap();
                                    guard.push(err.clone());
                                }

                                if attempts >= max_attempts {
                                    tracing::error!("[supervisor] child permanently dropped after exhausting retries (RestartOne) err={}", err);
                                    children_clone.remove(&pid);
                                    if let Some((_, v)) = links_clone.remove(&pid) {
                                        for other in v {
                                            if let Some(mut entry) = links_clone.get_mut(&other) {
                                                entry.retain(|&p| p != pid);
                                            }
                                        }
                                    }
                                    restarting_clone.remove(&pid);
                                    break;
                                }

                                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms))
                                    .await;
                                backoff_ms = backoff_ms.saturating_mul(2);
                            }
                        }
                    }
                });
            }
        }
    }
}

impl Runtime {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize_actor_exit(
        mailboxes: &DashMap<Pid, mailbox::MailboxSender>,
        supervisor: &Arc<Supervisor>,
        slab: &Arc<Mutex<pid::SlabAllocator>>,
        path_supervisors: &DashMap<String, Arc<Supervisor>>,
        rt_exit: &Runtime,
        pid: Pid,
        reason: mailbox::ExitReason,
        meta: Option<String>,
    ) {
        let linked = supervisor.linked_pids(pid);
        mailboxes.remove(&pid);
        supervisor.notify_exit(pid, &reason);
        for entry in path_supervisors.iter() {
            let sup = entry.value();
            if sup.contains_child(pid) {
                sup.notify_exit(pid, &reason);
            }
        }
        slab.lock().unwrap().deallocate(pid);

        match reason {
            mailbox::ExitReason::Normal => {
                rt_exit
                    .telemetry()
                    .log_event(crate::core::telemetry::TelemetryEvent::ActorStopped { pid });
            }
            _ => {
                rt_exit.telemetry().log_event(
                    crate::core::telemetry::TelemetryEvent::ActorCrashed {
                        pid,
                        reason: format!("{:?}", reason),
                    },
                );
            }
        }

        // structured concurrency cleanup + runtime metadata cleanup
        rt_exit.handle_exit_internal(pid);

        for lp in linked {
            if let Some(sender) = mailboxes.get(&lp) {
                let info = mailbox::ExitInfo {
                    from: pid,
                    reason: reason.clone(),
                    metadata: meta.clone(),
                };
                let _ = sender.send(mailbox::Message::System(mailbox::SystemMessage::Exit(info)));
            }
        }
    }

    /// Stop an actor by closing its mailbox.
    pub fn stop(&self, pid: Pid) {
        // if this pid is a proxy, clear both maps so lookups don't return
        // stale entries.  DashMap::remove returns the key and value pair.
        if let Some((_key, (addr, rpid))) = self.remote_proxies.remove(&pid) {
            self.proxy_by_remote.remove(&(addr.clone(), rpid));
        }
        self.behavior_versions.remove(&pid);
        self.behavior_history.remove(&pid);

        if self.mailboxes.remove(&pid).is_some() {
            self.handle_exit_internal(pid);
        } else if self.virtual_specs.remove(&pid).is_some() {
            self.virtual_activate_locks.remove(&pid);
            self.supervisor
                .notify_exit(pid, &mailbox::ExitReason::Normal);
            self.slab.lock().unwrap().deallocate(pid);
            self.handle_exit_internal(pid);
        }
    }

    /// Block the current thread until the actor with `pid` fully exits.
    pub fn wait(&self, pid: Pid) {
        super::RUNTIME.block_on(async {
            while self.is_alive(pid) {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        });
    }

    pub fn supervisor(&self) -> Arc<Supervisor> {
        self.supervisor.clone()
    }

    pub fn supervise(
        &self,
        pid: Pid,
        factory: Arc<dyn Fn() -> Result<Pid, String> + Send + Sync>,
        strategy: RestartStrategy,
    ) {
        let spec = ChildSpec { factory, strategy };
        self.supervisor.add_child(pid, spec);
    }

    pub fn link(&self, a: Pid, b: Pid) {
        self.supervisor.link(a, b);
    }

    pub fn unlink(&self, a: Pid, b: Pid) {
        self.supervisor.unlink(a, b);
    }

    /// Internal helper invoked when any actor exits to maintain parent/child
    /// state and to enforce structured concurrency.  This is called from
    /// each spawn_* helper after the actor has torn down its mailbox and been
    /// deallocated.
    pub(crate) fn handle_exit_internal(&self, pid: Pid) {
        // DashMap::remove returns (key, value); clean up if this was a proxy
        if let Some((_key, (addr, rpid))) = self.remote_proxies.remove(&pid) {
            self.proxy_by_remote.remove(&(addr.clone(), rpid));
        }
        self.backpressure_state.remove(&pid);
        self.bounded_capacity.remove(&pid);
        self.overflow_policy.remove(&pid);
        self.behavior_versions.remove(&pid);
        self.behavior_history.remove(&pid);
        self.observers.remove(&pid);
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.remove(&pid);
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
                self.backpressure_state.remove(&child);
                self.bounded_capacity.remove(&child);
                self.overflow_policy.remove(&child);
                self.behavior_versions.remove(&child);
                self.behavior_history.remove(&child);
                self.observers.remove(&child);
            }
        }
    }
}
