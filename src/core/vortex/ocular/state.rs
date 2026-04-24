use crate::vortex::ocular::model::TraceEvent;
use crossbeam_queue::ArrayQueue;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::thread;
use std::time::Instant;

pub static START_TIME: OnceLock<Instant> = OnceLock::new();
pub static EVENT_QUEUE: OnceLock<ArrayQueue<Vec<TraceEvent>>> = OnceLock::new();
pub static FREE_QUEUE: OnceLock<ArrayQueue<Vec<TraceEvent>>> = OnceLock::new();
pub static IS_RUNNING: AtomicBool = AtomicBool::new(false);
pub static IS_PRECISE: AtomicBool = AtomicBool::new(true);
pub static WORKER_THREAD: Mutex<Option<thread::JoinHandle<()>>> = Mutex::new(None);

pub static TSC_FREQ: OnceLock<f64> = OnceLock::new();
pub static START_TSC: OnceLock<u64> = OnceLock::new();

pub static DEINSTRUMENT_THRESHOLD: AtomicU32 = AtomicU32::new(500);
pub static PROCESSED_EVENTS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Default)]
pub struct TraceSummary {
    pub processed_events: u64,
    pub instruction_events: u64,
    pub unique_instruction_sites: usize,
    pub loop_trace_count: usize,
}

pub static LAST_SUMMARY: Mutex<TraceSummary> = Mutex::new(TraceSummary {
    processed_events: 0,
    instruction_events: 0,
    unique_instruction_sites: 0,
    loop_trace_count: 0,
});

pub static OBSERVED_OFFSETS_BY_CODE: Lazy<Mutex<HashMap<usize, Vec<i32>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Default)]
pub struct PatternSet {
    pub include: Vec<String>,
    pub exclude: Vec<String>,
}

pub static FILTER_PATTERNS: OnceLock<RwLock<PatternSet>> = OnceLock::new();

pub fn set_exclude_patterns(patterns: Vec<String>) {
    let rw = FILTER_PATTERNS.get_or_init(|| RwLock::new(PatternSet::default()));
    if let Ok(mut guard) = rw.write() {
        guard.exclude = patterns;
    }
}

pub fn get_exclude_patterns() -> Vec<String> {
    if let Some(rw) = FILTER_PATTERNS.get() {
        if let Ok(guard) = rw.read() {
            return guard.exclude.clone();
        }
    }
    Vec::new()
}

pub fn clear_exclude_patterns() {
    if let Some(rw) = FILTER_PATTERNS.get() {
        if let Ok(mut guard) = rw.write() {
            guard.exclude.clear();
        }
    }
}

pub fn set_include_patterns(patterns: Vec<String>) {
    let rw = FILTER_PATTERNS.get_or_init(|| RwLock::new(PatternSet::default()));
    if let Ok(mut guard) = rw.write() {
        guard.include = patterns;
    }
}

pub fn get_include_patterns() -> Vec<String> {
    if let Some(rw) = FILTER_PATTERNS.get() {
        if let Ok(guard) = rw.read() {
            return guard.include.clone();
        }
    }
    Vec::new()
}

pub fn clear_include_patterns() {
    if let Some(rw) = FILTER_PATTERNS.get() {
        if let Ok(mut guard) = rw.write() {
            guard.include.clear();
        }
    }
}

pub fn with_pattern_set<R>(f: impl FnOnce(&PatternSet) -> R) -> Option<R> {
    if let Some(rw) = FILTER_PATTERNS.get() {
        if let Ok(guard) = rw.read() {
            return Some(f(&guard));
        }
    }
    None
}

pub fn reset_stats() {
    PROCESSED_EVENTS.store(0, Ordering::Relaxed);
    if let Ok(mut summary) = LAST_SUMMARY.lock() {
        *summary = TraceSummary::default();
    }
    if let Ok(mut observed) = OBSERVED_OFFSETS_BY_CODE.lock() {
        observed.clear();
    }
}

pub fn set_summary(summary: TraceSummary) {
    if let Ok(mut s) = LAST_SUMMARY.lock() {
        *s = summary;
    }
}

pub fn get_summary() -> TraceSummary {
    LAST_SUMMARY.lock().map(|s| s.clone()).unwrap_or_default()
}

pub fn set_observed_offsets_by_code(new_map: HashMap<usize, Vec<i32>>) {
    if let Ok(mut observed) = OBSERVED_OFFSETS_BY_CODE.lock() {
        *observed = new_map;
    }
}

pub fn get_observed_offsets_for_code(code_ptr: usize) -> Vec<i32> {
    OBSERVED_OFFSETS_BY_CODE
        .lock()
        .ok()
        .and_then(|m| m.get(&code_ptr).cloned())
        .unwrap_or_default()
}

#[inline(always)]
pub fn read_tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut val: u64;
        unsafe { std::arch::asm!("mrs {}, cntvct_el0", out(reg) val) };
        val
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        0
    }
}

pub fn init_tsc_calibration() {
    START_TSC.get_or_init(|| {
        let tsc1 = read_tsc();
        let t1 = Instant::now();
        thread::sleep(std::time::Duration::from_millis(5));
        let tsc2 = read_tsc();
        let t2 = Instant::now();
        let elapsed_us = t2.duration_since(t1).as_micros() as f64;
        if elapsed_us > 0.0 {
            let _ = TSC_FREQ.set((tsc2.saturating_sub(tsc1)) as f64 / elapsed_us);
        }
        let _ = START_TIME.set(t2);
        tsc2
    });
}

pub fn get_ts() -> u64 {
    if let (Some(&start_tsc), Some(&freq)) = (START_TSC.get(), TSC_FREQ.get()) {
        let current_tsc = read_tsc();
        if current_tsc > start_tsc {
            ((current_tsc - start_tsc) as f64 / freq) as u64
        } else {
            0
        }
    } else {
        0
    }
}
