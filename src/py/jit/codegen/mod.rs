// src/py/jit/codegen/mod.rs
//! Core JIT compilation logic, including registry and Cranelift codegen.

use std::time::Instant;

// Internal submodules
mod buffer;
mod compiler;
mod eval;
mod exec;
mod gen_expr;
mod jit_module;
mod jit_types;
mod lowering;
mod math;
mod quantum;
mod registry;

// Re-exports
pub use compiler::*;
pub use jit_types::*;
pub use quantum::*;
pub use registry::*;

pub(crate) use jit_module::*;
pub(crate) use lowering::*;

// Internal helper exports for submodules
pub(crate) use buffer::*;
pub(crate) use eval::*;
pub(crate) use exec::*;
pub(crate) use gen_expr::gen_expr;
pub(crate) use math::*;

pub const BREAK_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0b01;
pub const CONTINUE_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0c01;

pub(crate) fn jit_dump_clif_enabled() -> bool {
    match std::env::var("IRIS_JIT_DUMP_CLIF") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(feature = "pyo3")]
pub fn execute_registered_jit(
    py: pyo3::Python<'_>,
    func_key: usize,
    args: &pyo3::Bound<'_, pyo3::types::PyTuple>,
) -> Option<pyo3::PyResult<pyo3::Py<pyo3::PyAny>>> {
    if crate::py::jit::quantum_speculation_enabled() {
        if let Some(map) = QUANTUM_REGISTRY.get() {
            let (entry, idx, fallback_entries, should_use_quantum, active_count) = {
                let mut guard = map.lock().unwrap();
                if let Some(state) = guard.get_mut(&func_key) {
                    if state.entries.is_empty() {
                        (None, 0usize, Vec::new(), false, 0usize)
                    } else {
                        let speculation_threshold_ns =
                            crate::py::jit::quantum_speculation_threshold_ns();
                        let best_ewma = state
                            .stats
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| state.active.get(*idx).copied().unwrap_or(false))
                            .map(|(_, s)| s)
                            .filter(|s| s.runs > 0)
                            .map(|s| s.ewma_ns)
                            .fold(f64::MAX, |a: f64, b| a.min(b));
                        let best_ewma = if best_ewma == f64::MAX {
                            0.0
                        } else {
                            best_ewma
                        };
                        let stability = quantum_stability_score(state);
                        let min_stability = crate::py::jit::quantum_stability_min_score();
                        let active_count = active_indices(state).len();
                        let has_unrun = has_unrun_active_variant(state);
                        let use_quantum = should_use_quantum_dispatch(
                            active_count,
                            best_ewma,
                            stability,
                            min_stability,
                            speculation_threshold_ns,
                            has_unrun,
                        );

                        if use_quantum {
                            let idx = choose_quantum_index(state);
                            let entry = state.entries[idx].clone();
                            if !entry.is_valid() {
                                if idx < state.active.len() {
                                    state.active[idx] = false;
                                }
                                let baseline_idx = state.baseline_idx.min(state.entries.len() - 1);
                                let baseline_entry = state.entries[baseline_idx].clone();
                                if !baseline_entry.is_valid() {
                                    (None, 0usize, Vec::new(), false, 0usize)
                                } else {
                                    (
                                        Some(baseline_entry),
                                        baseline_idx,
                                        Vec::new(),
                                        false,
                                        active_count,
                                    )
                                }
                            } else {
                                let mut fallbacks = Vec::new();
                                for (i, e) in state.entries.iter().enumerate() {
                                    if i != idx
                                        && state.active.get(i).copied().unwrap_or(false)
                                        && e.is_valid()
                                    {
                                        fallbacks.push((i, e.clone()));
                                    }
                                }
                                (Some(entry), idx, fallbacks, true, active_count)
                            }
                        } else {
                            let baseline_idx = state.baseline_idx.min(state.entries.len() - 1);
                            let baseline_entry = state.entries[baseline_idx].clone();
                            if !baseline_entry.is_valid() {
                                (None, 0usize, Vec::new(), false, active_count)
                            } else {
                                (
                                    Some(baseline_entry),
                                    baseline_idx,
                                    Vec::new(),
                                    false,
                                    active_count,
                                )
                            }
                        }
                    }
                } else {
                    (None, 0usize, Vec::new(), false, 0usize)
                }
            };

            if let Some(entry) = entry {
                let quantum_total = active_count.max(1usize);
                let chosen_strategy = entry.variant_strategy;
                let log_threshold_ns = crate::py::jit::quantum_log_threshold_ns();

                let log_if_slow = |chosen: usize, used: usize, elapsed_ns: u64, note: &str| {
                    if crate::py::jit::jit_logging_enabled() && elapsed_ns >= log_threshold_ns {
                        crate::py::jit::jit_log(|| {
                            format!(
                                "[Iris][jit][quantum] func_key={} chosen={}/{} used={} elapsed={}ns {} active={} strategy={:?}",
                                func_key, chosen, quantum_total, used, elapsed_ns, note, active_count, chosen_strategy
                            )
                        });
                    }
                };

                let start = Instant::now();
                match execute_jit_func(py, &entry, args) {
                    Ok(obj) => {
                        let elapsed = start.elapsed().as_nanos() as u64;
                        update_quantum_stats(func_key, idx, elapsed, true);
                        if !should_use_quantum {
                            let _ = crate::py::jit::maybe_rearm_quantum_compile(
                                func_key,
                                elapsed,
                                active_count,
                            );
                        }
                        log_if_slow(
                            idx,
                            idx,
                            elapsed,
                            if should_use_quantum {
                                "success"
                            } else {
                                "baseline"
                            },
                        );
                        return Some(Ok(obj));
                    }
                    Err(primary_err) => {
                        let elapsed = start.elapsed().as_nanos() as u64;
                        update_quantum_stats(func_key, idx, elapsed, false);
                        if !should_use_quantum {
                            let _ = crate::py::jit::maybe_rearm_quantum_compile(
                                func_key,
                                elapsed,
                                active_count,
                            );
                        }
                        log_if_slow(idx, idx, elapsed, "primary_failed");
                        if should_use_quantum {
                            for (fb_idx, fb_entry) in fallback_entries {
                                let start_fb = Instant::now();
                                match execute_jit_func(py, &fb_entry, args) {
                                    Ok(obj) => {
                                        let elapsed_fb = start_fb.elapsed().as_nanos() as u64;
                                        update_quantum_stats(func_key, fb_idx, elapsed_fb, true);
                                        log_if_slow(idx, fb_idx, elapsed_fb, "fallback_success");
                                        return Some(Ok(obj));
                                    }
                                    Err(_) => {
                                        let elapsed_fb = start_fb.elapsed().as_nanos() as u64;
                                        update_quantum_stats(func_key, fb_idx, elapsed_fb, false);
                                        log_if_slow(idx, fb_idx, elapsed_fb, "fallback_failed");
                                    }
                                }
                            }
                        }
                        return Some(Err(primary_err));
                    }
                }
            }
        }
    }

    lookup_jit(func_key).map(|entry| execute_jit_func(py, &entry, args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::py::jit::parser;

    #[test]
    fn detect_lowered_kernel_for_trig_unary() {
        let mut p = parser::Parser::new(parser::tokenize("math.sin(x)"));
        let expr = p.parse_expr().expect("parse trig");
        let args = vec!["x".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        assert_eq!(
            kernel,
            Some(LoweredKernel::Unary {
                op: LoweredUnaryKernel::Sin,
                input: 0,
            })
        );
    }

    #[test]
    fn detect_lowered_kernel_for_binary_arith() {
        let mut p = parser::Parser::new(parser::tokenize("a * b"));
        let expr = p.parse_expr().expect("parse binary");
        let args = vec!["a".to_string(), "b".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        assert_eq!(
            kernel,
            Some(LoweredKernel::Binary {
                op: LoweredBinaryKernel::Mul,
                lhs: 0,
                rhs: 1,
            })
        );
    }

    #[test]
    fn detect_lowered_kernel_for_complex_trig_exp_expr() {
        let src = "((math.sin(x) * 0.5 + x * 1.2) / (1.0 + math.exp(-abs(x) * 0.001)))";
        let mut p = parser::Parser::new(parser::tokenize(src));
        let expr = p.parse_expr().expect("parse complex lowered expr");
        let args = vec!["x".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        match kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected LoweredKernel::Expr(_), got {:?}", other),
        }
    }

    #[test]
    fn compile_sumover_complex_expr_selects_lowered_kernel() {
        let src = "sum(((math.sin(x) * 0.5 + x * 1.2) / (1.0 + math.exp(-abs(x) * 0.001)) for x in data))";
        let args = vec!["data".to_string()];
        let entry = compile_jit_impl(src, &args, true, JitReturnType::Float)
            .expect("compile sumover expression");
        match entry.lowered_kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected lowered expression kernel, got {:?}", other),
        }
    }

    #[test]
    fn lowered_unary_eval_pair_supports_wide_vector_math() {
        let result = lowered_unary_eval_pair(
            LoweredUnaryKernel::Sin,
            0.5_f64,
            1.0_f64,
            SimdMathMode::FastApprox,
        );
        assert!(result.is_some());
        let (y0, y1) = result.unwrap();
        assert!((y0 - 0.4794).abs() < 0.001);
        assert!((y1 - 0.84147).abs() < 0.001);
    }

    #[test]
    fn detect_lowered_kernel_for_filtered_ternary_expr() {
        let src = "((x % 2 == 0 or x > 75.0) and (not x < 10.0)) ? (x > 50.0 ? x * math.sin(x) : x * math.cos(x)) : 0.0";
        let mut p = parser::Parser::new(parser::tokenize(src));
        let expr = p.parse_expr().expect("parse filtered ternary expr");
        let args = vec!["x".to_string()];
        let kernel = detect_lowered_kernel(&expr, &args);
        match kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected LoweredKernel::Expr(_), got {:?}", other),
        }
    }

    #[test]
    fn compile_sumover_filtered_ternary_selects_lowered_kernel() {
        let src = "sum((x * math.sin(x) if x > 50.0 else x * math.cos(x) for x in data if (x % 2 == 0 or x > 75.0) and (not x < 10.0)))";
        let args = vec!["data".to_string()];
        let entry = compile_jit_impl(src, &args, true, JitReturnType::Float)
            .expect("compile filtered ternary sumover expression");
        match entry.lowered_kernel {
            Some(LoweredKernel::Expr(_)) => {}
            other => panic!("expected lowered expression kernel, got {:?}", other),
        }
    }
}
