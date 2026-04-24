// src/vortex/scheduler.rs
//! Experimental Vortex scheduler placeholder.

pub struct VortexScheduler;

impl Default for VortexScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl VortexScheduler {
    pub fn new() -> Self {
        VortexScheduler
    }

    pub fn describe(&self) -> &'static str {
        "vortex scheduler (stub)"
    }
}
