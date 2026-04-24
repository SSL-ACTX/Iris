// src/vortex/watcher.rs
//! Simple experimental Vortex watchdog.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

pub struct VortexWatcher {
    enabled: Arc<AtomicBool>,
}

impl Default for VortexWatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl VortexWatcher {
    pub fn new() -> Self {
        VortexWatcher {
            enabled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn health(&self) -> &'static str {
        "vortex watcher healthy"
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn enable(&self) {
        if self
            .enabled
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let enabled = self.enabled.clone();
            tokio::spawn(async move {
                while enabled.load(Ordering::Relaxed) {
                    // In a full implementation, this would inspect actor liveness + preemption counters,
                    // and potentially escalate via OS-level signal (SIGVTALRM) or internal throttling.
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
        }
    }

    pub fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }
}
