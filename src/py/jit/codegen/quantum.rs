// src/py/jit/codegen/quantum.rs
//! Quantum variant selection/profile state for the JIT.

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::OnceCell;

use crate::py::jit::codegen::jit_types::{
    JitEntry, QuantumProfilePoint, QuantumProfileSeed, QuantumVariantStrategy,
};
use crate::py::jit::config::{quantum_variant_failure_limit, quantum_variant_promotion_min_runs};

#[derive(Clone, Default)]
pub(crate) struct QuantumStats {
    pub(crate) ewma_ns: f64,
    pub(crate) runs: u64,
    pub(crate) failures: u64,
}

#[derive(Clone)]
pub(crate) struct QuantumState {
    pub(crate) entries: Vec<JitEntry>,
    pub(crate) stats: Vec<QuantumStats>,
    pub(crate) active: Vec<bool>,
    pub(crate) baseline_idx: usize,
    pub(crate) round_robin: usize,
    pub(crate) total_runs: u64,
}

pub(crate) static QUANTUM_REGISTRY: OnceCell<Mutex<HashMap<usize, QuantumState>>> = OnceCell::new();
pub(crate) static QUANTUM_PENDING_SEEDS: OnceCell<Mutex<HashMap<usize, Vec<QuantumProfileSeed>>>> =
    OnceCell::new();

fn apply_quantum_seeds(state: &mut QuantumState, seeds: &[QuantumProfileSeed]) {
    for seed in seeds {
        if let Some((pos, _)) = state.entries.iter().enumerate().find(|(_, entry)| {
            quantum_variant_index_for_strategy(entry.variant_strategy) == seed.index
        }) {
            if let Some(stats) = state.stats.get_mut(pos) {
                stats.ewma_ns = if seed.ewma_ns.is_finite() && seed.ewma_ns >= 0.0 {
                    seed.ewma_ns
                } else {
                    0.0
                };
                stats.runs = seed.runs;
                stats.failures = seed.failures;
            }
        }
    }
}

pub fn register_quantum_jit(func_key: usize, mut entries: Vec<JitEntry>) {
    if entries.is_empty() {
        return;
    }
    let entry_count = entries.len();
    // prefer optimized candidate (first) as baseline fallback mapping
    crate::py::jit::codegen::register_jit(func_key, entries[0].clone());
    let stats = vec![QuantumStats::default(); entries.len()];
    let mut state = QuantumState {
        entries: std::mem::take(&mut entries),
        stats,
        active: vec![true; entry_count],
        baseline_idx: 0,
        round_robin: 0,
        total_runs: 0,
    };

    if let Some(pending_map) = QUANTUM_PENDING_SEEDS.get() {
        if let Some(seeds) = pending_map.lock().unwrap().remove(&func_key) {
            apply_quantum_seeds(&mut state, &seeds);
        }
    }

    let map = QUANTUM_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    map.lock().unwrap().insert(func_key, state);
}

pub fn quantum_has_seed_hint(func_key: usize) -> bool {
    if let Some(map) = QUANTUM_PENDING_SEEDS.get() {
        let guard = map.lock().unwrap();
        return guard
            .get(&func_key)
            .map(|rows| !rows.is_empty())
            .unwrap_or(false);
    }

    false
}

fn quantum_variant_index_for_strategy(strategy: QuantumVariantStrategy) -> usize {
    match strategy {
        QuantumVariantStrategy::Auto => 0,
        QuantumVariantStrategy::ScalarFallback => 1,
        QuantumVariantStrategy::FastTrigExperiment => 2,
    }
}

pub fn quantum_seed_preferred_index(func_key: usize) -> Option<usize> {
    if let Some(map) = QUANTUM_PENDING_SEEDS.get() {
        let guard = map.lock().unwrap();
        if let Some(seeds) = guard.get(&func_key) {
            return preferred_seed_variant_index(seeds);
        }
    }

    None
}

pub(crate) fn quantum_score(stats: &QuantumStats) -> f64 {
    let penalty = 1.0 + (stats.failures as f64);
    if stats.ewma_ns > 0.0 {
        stats.ewma_ns * penalty
    } else {
        f64::MAX / 4.0
    }
}

pub(crate) fn quantum_seed_score(seed: &QuantumProfileSeed) -> Option<f64> {
    if !seed.ewma_ns.is_finite() || seed.ewma_ns <= 0.0 {
        return None;
    }

    let run_penalty = if seed.runs == 0 { 4.0 } else { 1.0 };
    let failure_penalty = 1.0 + (seed.failures as f64 * 2.0);
    Some(seed.ewma_ns * run_penalty * failure_penalty)
}

pub(crate) fn preferred_seed_variant_index(seeds: &[QuantumProfileSeed]) -> Option<usize> {
    const MIN_CONFIDENT_RUNS: u64 = 3;

    let scalar_seed = seeds.iter().find(|seed| seed.index == 1).cloned();
    let all_thin_samples = seeds
        .iter()
        .filter(|seed| quantum_seed_score(seed).is_some())
        .all(|seed| seed.runs < MIN_CONFIDENT_RUNS);

    if all_thin_samples {
        if let Some(seed) = scalar_seed {
            if quantum_seed_score(&seed).is_some() && seed.failures == 0 {
                return Some(1);
            }
        }
    }

    let mut best: Option<(usize, f64, u64)> = None;
    for seed in seeds {
        let Some(score) = quantum_seed_score(seed) else {
            continue;
        };

        match best {
            None => best = Some((seed.index, score, seed.runs)),
            Some((best_idx, best_score, best_runs)) => {
                if score < best_score
                    || ((score - best_score).abs() < f64::EPSILON && seed.runs > best_runs)
                    || ((score - best_score).abs() < f64::EPSILON
                        && seed.runs == best_runs
                        && seed.index < best_idx)
                {
                    best = Some((seed.index, score, seed.runs));
                }
            }
        }
    }

    best.map(|(idx, _, _)| idx)
}

pub(crate) fn active_indices(state: &QuantumState) -> Vec<usize> {
    state
        .active
        .iter()
        .enumerate()
        .filter_map(|(idx, is_active)| if *is_active { Some(idx) } else { None })
        .collect()
}

pub(crate) fn quantum_stability_score(state: &QuantumState) -> f64 {
    let active = active_indices(state);
    if active.len() <= 1 {
        return 1.0;
    }

    let mut total_runs = 0u64;
    let mut total_failures = 0u64;
    let mut min_ewma: f64 = f64::MAX;
    let mut max_ewma: f64 = 0.0;
    let mut ewma_count = 0u64;

    for idx in active {
        let s = &state.stats[idx];
        total_runs = total_runs.saturating_add(s.runs);
        total_failures = total_failures.saturating_add(s.failures);
        if s.runs > 0 && s.ewma_ns > 0.0 {
            ewma_count = ewma_count.saturating_add(1);
            min_ewma = min_ewma.min(s.ewma_ns);
            max_ewma = max_ewma.max(s.ewma_ns);
        }
    }

    let min_runs = crate::py::jit::quantum_stability_min_runs();
    if total_runs < min_runs {
        return 0.0;
    }

    let denom = (total_runs.saturating_add(total_failures)) as f64;
    let failure_rate = if denom > 0.0 {
        (total_failures as f64 / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let dispersion = if ewma_count >= 2 && max_ewma > 0.0 && min_ewma < f64::MAX {
        ((max_ewma - min_ewma) / max_ewma).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (1.0 - failure_rate) * (1.0 - dispersion)
}

pub(crate) fn reconcile_quantum_lifecycle_state(state: &mut QuantumState) {
    if state.entries.is_empty() {
        return;
    }

    let fail_limit = quantum_variant_failure_limit();
    let promotion_min_runs = quantum_variant_promotion_min_runs();
    let baseline = state.baseline_idx.min(state.entries.len() - 1);
    state.baseline_idx = baseline;

    for idx in 0..state.entries.len() {
        if idx == baseline {
            state.active[idx] = true;
            continue;
        }
        if !state.active[idx] {
            continue;
        }
        let s = &state.stats[idx];
        if s.failures >= fail_limit && s.runs < promotion_min_runs {
            state.active[idx] = false;
        }
    }

    let baseline_score = quantum_score(&state.stats[baseline]);
    let mut best_idx = baseline;
    let mut best_score = baseline_score;
    for idx in 0..state.entries.len() {
        if !state.active[idx] {
            continue;
        }
        let s = &state.stats[idx];
        if s.runs < promotion_min_runs {
            continue;
        }
        let score = quantum_score(s);
        if score < best_score * 0.90 {
            best_score = score;
            best_idx = idx;
        }
    }
    state.baseline_idx = best_idx;
    state.active[state.baseline_idx] = true;
}

#[allow(dead_code)]
pub(crate) fn reconcile_quantum_lifecycle(func_key: usize) -> bool {
    if let Some(map) = QUANTUM_REGISTRY.get() {
        let mut guard = map.lock().unwrap();
        if let Some(state) = guard.get_mut(&func_key) {
            reconcile_quantum_lifecycle_state(state);
            return true;
        }
    }
    false
}

#[allow(dead_code)]
pub(crate) fn quantum_active_variant_count(func_key: usize) -> Option<usize> {
    QUANTUM_REGISTRY.get().and_then(|map| {
        map.lock()
            .unwrap()
            .get(&func_key)
            .map(|state| state.active.iter().filter(|a| **a).count())
    })
}

#[allow(dead_code)]
pub(crate) fn quantum_stability_for(func_key: usize) -> Option<f64> {
    QUANTUM_REGISTRY.get().and_then(|map| {
        map.lock()
            .unwrap()
            .get(&func_key)
            .map(quantum_stability_score)
    })
}

pub(crate) fn choose_quantum_index(state: &mut QuantumState) -> usize {
    for (idx, s) in state.stats.iter().enumerate() {
        if !state.active.get(idx).copied().unwrap_or(false) {
            continue;
        }
        if s.runs == 0 {
            return idx;
        }
    }

    let active = active_indices(state);
    if active.is_empty() {
        return state
            .baseline_idx
            .min(state.entries.len().saturating_sub(1));
    }

    state.total_runs = state.total_runs.saturating_add(1);
    if state.total_runs.is_multiple_of(16) {
        let rr = state.round_robin % active.len();
        state.round_robin = (rr + 1) % active.len();
        return active[rr];
    }
    let mut best_idx = state
        .baseline_idx
        .min(state.entries.len().saturating_sub(1));
    let mut best_score = f64::MAX;
    for idx in active {
        let s = &state.stats[idx];
        let score = quantum_score(s);
        if score < best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

pub(crate) fn has_unrun_active_variant(state: &QuantumState) -> bool {
    state
        .stats
        .iter()
        .enumerate()
        .any(|(idx, stats)| state.active.get(idx).copied().unwrap_or(false) && stats.runs == 0)
}

pub(crate) fn should_use_quantum_dispatch(
    active_count: usize,
    best_ewma_ns: f64,
    stability: f64,
    min_stability: f64,
    speculation_threshold_ns: u64,
    has_unrun_variant: bool,
) -> bool {
    if active_count <= 1 {
        return false;
    }
    if has_unrun_variant {
        return true;
    }
    best_ewma_ns >= (speculation_threshold_ns as f64) && stability >= min_stability
}

pub(crate) fn update_quantum_stats(func_key: usize, idx: usize, elapsed_ns: u64, success: bool) {
    if let Some(map) = QUANTUM_REGISTRY.get() {
        if let Some(state) = map.lock().unwrap().get_mut(&func_key) {
            if let Some(stats) = state.stats.get_mut(idx) {
                if success {
                    stats.runs = stats.runs.saturating_add(1);
                    let sample = elapsed_ns as f64;
                    if stats.ewma_ns <= 0.0 {
                        stats.ewma_ns = sample;
                    } else {
                        stats.ewma_ns = stats.ewma_ns * 0.80 + sample * 0.20;
                    }
                } else {
                    stats.failures = stats.failures.saturating_add(1);
                }
            }
            reconcile_quantum_lifecycle_state(state);
        }
    }
}

pub fn quantum_profile_snapshot(func_key: usize) -> Option<Vec<QuantumProfilePoint>> {
    QUANTUM_REGISTRY.get().and_then(|map| {
        map.lock().unwrap().get(&func_key).map(|state| {
            state
                .stats
                .iter()
                .enumerate()
                .map(|(index, stats)| QuantumProfilePoint {
                    index: state
                        .entries
                        .get(index)
                        .map(|entry| quantum_variant_index_for_strategy(entry.variant_strategy))
                        .unwrap_or(index),
                    ewma_ns: stats.ewma_ns,
                    runs: stats.runs,
                    failures: stats.failures,
                })
                .collect()
        })
    })
}

pub fn seed_quantum_profile(func_key: usize, seeds: &[QuantumProfileSeed]) -> bool {
    if let Some(map) = QUANTUM_REGISTRY.get() {
        let mut guard = map.lock().unwrap();
        if let Some(state) = guard.get_mut(&func_key) {
            apply_quantum_seeds(state, seeds);
            return true;
        }
    }

    if seeds.is_empty() {
        return false;
    }

    let pending = QUANTUM_PENDING_SEEDS.get_or_init(|| Mutex::new(HashMap::new()));
    pending.lock().unwrap().insert(func_key, seeds.to_vec());
    true
}
