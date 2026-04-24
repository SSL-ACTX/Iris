// src/vortex/rescue_pool.rs
//! Experimental rescue pool to isolate stalled C-bound threads stub.

#[derive(Debug, Clone)]
pub struct RescuePool {
    pub active_count: usize,
}

impl RescuePool {
    pub fn new() -> Self {
        RescuePool { active_count: 0 }
    }
}

impl Default for RescuePool {
    fn default() -> Self {
        Self::new()
    }
}

impl RescuePool {
    pub fn run_blocking<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }

    pub fn detach_thread(&mut self) {
        self.active_count += 1;
    }

    pub fn reclaim_thread(&mut self) {
        if self.active_count > 0 {
            self.active_count -= 1;
        }
    }
}
