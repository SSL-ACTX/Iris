// src/core/vortex/runtime.rs
use crate::core::vortex::{VortexEngine, VortexGhostPolicy, VortexGhostResolution, VortexVioCall};
use crate::core::Runtime;
use crate::pid::Pid;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

impl Runtime {
    pub fn get_vortex_engine(&self) -> Option<VortexEngine> {
        self.vortex_engine
            .as_ref()
            .and_then(|engine: &Arc<Mutex<VortexEngine>>| {
                engine
                    .lock()
                    .ok()
                    .map(|guard: std::sync::MutexGuard<'_, VortexEngine>| guard.clone())
            })
    }

    pub fn start_vortex_transaction_with_checkpoint(
        &self,
        id: u64,
        locals: HashMap<String, Vec<u8>>,
    ) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => {
                guard.start_transaction_with_checkpoint(id, locals);
                true
            }
            Err(_) => false,
        }
    }

    pub fn stage_vortex_transaction_vio(&self, op: String, payload: Vec<u8>) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => guard.stage_transaction_vio(op, payload),
            Err(_) => false,
        }
    }

    pub fn commit_vortex_transaction(&self) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => guard.commit_transaction(),
            Err(_) => false,
        }
    }

    pub fn take_vortex_committed_transaction_vio(&self) -> Option<Vec<VortexVioCall>> {
        let engine = self.vortex_engine.as_ref()?;
        match engine.lock() {
            Ok(mut guard) => Some(guard.take_committed_transaction_vio()),
            Err(_) => None,
        }
    }

    pub fn start_vortex_ghost_transaction_with_checkpoint(
        &self,
        id: u64,
        locals: HashMap<String, Vec<u8>>,
    ) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => {
                guard.start_ghost_transaction_with_checkpoint(id, locals);
                true
            }
            Err(_) => false,
        }
    }

    pub fn stage_vortex_ghost_transaction_vio(
        &self,
        ghost_id: u64,
        op: String,
        payload: Vec<u8>,
    ) -> bool {
        let Some(engine) = self.vortex_engine.as_ref() else {
            return false;
        };
        match engine.lock() {
            Ok(mut guard) => guard.stage_ghost_transaction_vio(ghost_id, op, payload),
            Err(_) => false,
        }
    }

    pub fn resolve_vortex_primary_ghost_race(
        &self,
        ghost_id: u64,
        winner_id: u64,
        policy: VortexGhostPolicy,
    ) -> Option<VortexGhostResolution> {
        let engine = self.vortex_engine.as_ref()?;
        match engine.lock() {
            Ok(mut guard) => guard.resolve_primary_ghost_race(ghost_id, winner_id, policy),
            Err(_) => None,
        }
    }

    pub fn replay_vortex_committed_vio_calls<F>(
        &self,
        calls: &[VortexVioCall],
        executor: F,
    ) -> Option<usize>
    where
        F: FnMut(&VortexVioCall) -> bool,
    {
        let engine = self.vortex_engine.as_ref()?;
        match engine.lock() {
            Ok(guard) => Some(guard.replay_committed_vio_calls(calls, executor)),
            Err(_) => None,
        }
    }

    pub fn get_vortex_auto_replay_count(&self) -> u64 {
        self.vortex_auto_replay_count.load(Ordering::Relaxed)
    }

    pub fn set_vortex_auto_ghost_policy(&self, policy: VortexGhostPolicy) -> bool {
        match self.vortex_auto_policy.lock() {
            Ok(mut guard) => {
                *guard = policy;
                true
            }
            Err(_) => false,
        }
    }

    pub fn get_vortex_auto_ghost_policy(&self) -> Option<VortexGhostPolicy> {
        self.vortex_auto_policy.lock().ok().map(|guard| *guard)
    }

    pub fn get_vortex_auto_resolution_counts(&self) -> (u64, u64) {
        (
            self.vortex_auto_primary_wins.load(Ordering::Relaxed),
            self.vortex_auto_ghost_wins.load(Ordering::Relaxed),
        )
    }

    pub fn reset_vortex_auto_telemetry(&self) {
        self.vortex_auto_replay_count.store(0, Ordering::Relaxed);
        self.vortex_auto_primary_wins.store(0, Ordering::Relaxed);
        self.vortex_auto_ghost_wins.store(0, Ordering::Relaxed);
        self.vortex_ghost_counter.store(1, Ordering::Relaxed);
    }

    pub fn set_vortex_genetic_budgeting(&self, enabled: bool) -> bool {
        match self.vortex_genetic_budgeting_enabled.lock() {
            Ok(mut guard) => {
                *guard = enabled;
                true
            }
            Err(_) => false,
        }
    }

    pub fn is_vortex_genetic_budgeting_enabled(&self) -> Option<bool> {
        self.vortex_genetic_budgeting_enabled
            .lock()
            .ok()
            .map(|guard| *guard)
    }

    pub fn set_vortex_genetic_thresholds(&self, low: f64, high: f64) -> bool {
        if low < 0.0 || high < 0.0 || low >= high || high > 1.0 {
            return false;
        }
        match self.vortex_genetic_thresholds.lock() {
            Ok(mut guard) => {
                *guard = (low, high);
                true
            }
            Err(_) => false,
        }
    }

    pub fn get_vortex_genetic_thresholds(&self) -> Option<(f64, f64)> {
        self.vortex_genetic_thresholds
            .lock()
            .ok()
            .map(|guard| *guard)
    }

    pub fn set_vortex_isolation_disallowed_ops(&self, ops: Vec<u8>) -> bool {
        match self.vortex_isolation_disallowed_ops.lock() {
            Ok(mut guard) => {
                guard.clear();
                for op in ops {
                    guard.insert(op);
                }
                true
            }
            Err(_) => false,
        }
    }

    pub fn get_vortex_isolation_disallowed_ops(&self) -> Option<Vec<u8>> {
        self.vortex_isolation_disallowed_ops
            .lock()
            .ok()
            .map(|guard| guard.iter().copied().collect::<Vec<_>>())
    }

    pub fn enable_vortex_watchdog(&self) -> bool {
        if let Some(watcher) = self.vortex_watcher.as_ref() {
            watcher.enable();
            true
        } else {
            false
        }
    }

    pub fn disable_vortex_watchdog(&self) -> bool {
        if let Some(watcher) = self.vortex_watcher.as_ref() {
            watcher.disable();
            true
        } else {
            false
        }
    }

    pub fn is_vortex_watchdog_enabled(&self) -> Option<bool> {
        self.vortex_watcher.as_ref().map(|w| w.is_enabled())
    }

    pub fn get_vortex_genetic_history(&self, pid: Pid) -> Option<(usize, usize)> {
        self.vortex_genetic_history.get(&pid).map(|r| *r)
    }

    pub fn get_all_vortex_genetic_history(&self) -> Vec<(Pid, usize, usize)> {
        self.vortex_genetic_history
            .iter()
            .map(|entry| (*entry.key(), entry.value().0, entry.value().1))
            .collect::<Vec<_>>()
    }

    pub fn reset_vortex_genetic_history(&self) {
        self.vortex_genetic_history.clear();
    }

    pub(crate) fn auto_checkpoint_and_replay_on_suspend(&self, pid: Pid, budget: usize) {
        let Some(engine_arc) = self.vortex_engine.as_ref() else {
            return;
        };

        let primary_id = self.vortex_ghost_counter.fetch_add(1, Ordering::Relaxed);
        let ghost_id = self.vortex_ghost_counter.fetch_add(1, Ordering::Relaxed);

        let mut primary_locals = HashMap::new();
        primary_locals.insert("pid".to_string(), pid.to_le_bytes().to_vec());
        primary_locals.insert("budget".to_string(), (budget as u64).to_le_bytes().to_vec());

        let mut ghost_locals = HashMap::new();
        ghost_locals.insert("pid".to_string(), pid.to_le_bytes().to_vec());
        ghost_locals.insert("budget".to_string(), (budget as u64).to_le_bytes().to_vec());

        let Ok(mut guard) = engine_arc.lock() else {
            return;
        };

        guard.start_transaction_with_checkpoint(primary_id, primary_locals);
        let _ =
            guard.stage_transaction_vio("suspend_primary".to_string(), pid.to_le_bytes().to_vec());

        guard.start_ghost_transaction_with_checkpoint(ghost_id, ghost_locals);
        let _ = guard.stage_ghost_transaction_vio(
            ghost_id,
            "suspend_ghost".to_string(),
            pid.to_le_bytes().to_vec(),
        );

        let policy = self
            .vortex_auto_policy
            .lock()
            .map(|guard| *guard)
            .unwrap_or(VortexGhostPolicy::FirstSafePointWins);

        if let Some(resolution) = guard.resolve_primary_ghost_race(ghost_id, ghost_id, policy) {
            if resolution.winner_id == primary_id {
                self.vortex_auto_primary_wins
                    .fetch_add(1, Ordering::Relaxed);
            } else if resolution.winner_id == ghost_id {
                self.vortex_auto_ghost_wins.fetch_add(1, Ordering::Relaxed);
            }

            let applied = guard.replay_committed_vio_calls(&resolution.committed_vio, |_| true);
            if applied > 0 {
                self.vortex_auto_replay_count
                    .fetch_add(applied as u64, Ordering::Relaxed);
            }
        }
    }
}
