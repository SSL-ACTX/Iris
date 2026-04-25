// src/py/jit/codegen/exec.rs
//! Execution logic for compiled JIT entries, handling vectorization, unpacking, and PyO3 bridging.

use std::collections::HashSet;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::IntoPyObjectExt;

use crate::py::jit::simd;

use super::{
    lowered_binary_eval, lowered_expr_eval, lowered_expr_eval_pair, lowered_unary_eval,
    lowered_unary_eval_pair, open_typed_buffer, read_buffer_f64, BufferElemType, BufferView,
    JitEntry, JitExecProfile, JitReturnType, LoweredBinaryKernel, LoweredKernel,
    LoweredUnaryKernel, ReductionMode, BREAK_SENTINEL_BITS, CONTINUE_SENTINEL_BITS,
    TLS_JIT_TYPE_PROFILE,
};

static LOWERED_EXEC_LOGGED: once_cell::sync::OnceCell<std::sync::Mutex<HashSet<usize>>> =
    once_cell::sync::OnceCell::new();
static UNROLL_EXEC_LOGGED: once_cell::sync::OnceCell<std::sync::Mutex<HashSet<usize>>> =
    once_cell::sync::OnceCell::new();
static SIMD_MATH_EXEC_LOGGED: once_cell::sync::OnceCell<std::sync::Mutex<HashSet<usize>>> =
    once_cell::sync::OnceCell::new();

fn log_lowered_exec_once(entry: &JitEntry, len: usize) {
    let Some(kernel) = entry.lowered_kernel.as_ref() else {
        return;
    };

    let set = LOWERED_EXEC_LOGGED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut guard = set.lock().unwrap();
    if !guard.insert(entry.func_ptr) {
        return;
    }

    let math_mode = entry.math_mode_for_entry();
    crate::py::jit::jit_log(|| {
        format!(
            "[Iris][jit][lower] execute kernel={:?} reduction={:?} len={} math_mode={:?} variant={:?}",
            kernel, entry.reduction, len, math_mode, entry.variant_strategy
        )
    });
}

fn log_unroll_exec_once(entry: &JitEntry, len: usize, unroll: usize, path: &'static str) {
    if unroll <= 1 || len < unroll {
        return;
    }
    let set = UNROLL_EXEC_LOGGED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut guard = set.lock().unwrap();
    if !guard.insert(entry.func_ptr) {
        return;
    }
    crate::py::jit::jit_log(|| {
        format!(
            "[Iris][jit][unroll] execute path={} len={} unroll={} variant={:?}",
            path, len, unroll, entry.variant_strategy
        )
    });
}

fn log_simd_math_exec_once(entry: &JitEntry, op: LoweredUnaryKernel) {
    let set = SIMD_MATH_EXEC_LOGGED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut guard = set.lock().unwrap();
    if !guard.insert(entry.func_ptr) {
        return;
    }

    crate::py::jit::jit_log(|| {
        format!(
            "[Iris][jit][simd-math] backend=neon-f64x2 op={:?} variant={:?}",
            op, entry.variant_strategy
        )
    });
}

enum LoweredVectorResult {
    Vector(Vec<f64>),
    Reduced(f64),
}

#[inline(always)]
fn try_execute_lowered_vector_kernel(
    entry: &JitEntry,
    views: &[BufferView],
    len: usize,
    unroll: usize,
) -> Option<LoweredVectorResult> {
    if !entry.allows_lowered_kernel() {
        return None;
    }
    let mode = entry.math_mode_for_entry();

    let kernel = entry.lowered_kernel.clone()?;

    let eval_unary = |op: LoweredUnaryKernel, view: &BufferView, idx: usize| {
        let x = unsafe { read_buffer_f64(view, idx) };
        lowered_unary_eval(op, x, mode)
    };
    let eval_binary =
        |op: LoweredBinaryKernel, lhs_view: &BufferView, rhs_view: &BufferView, idx: usize| {
            let l = unsafe { read_buffer_f64(lhs_view, idx) };
            let r = unsafe { read_buffer_f64(rhs_view, idx) };
            lowered_binary_eval(op, l, r)
        };

    if entry.reduction != ReductionMode::None {
        let lanes = unroll.clamp(1, 4);
        match kernel {
            LoweredKernel::Unary { op, input } => {
                let input_view = views.get(input)?;
                match entry.reduction {
                    ReductionMode::Sum => {
                        let mut lane_acc = [0.0_f64; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            let mut lane = 0;
                            while lane < lanes {
                                if lane + 1 < lanes {
                                    let x0 = unsafe { read_buffer_f64(input_view, i + lane) };
                                    let x1 = unsafe { read_buffer_f64(input_view, i + lane + 1) };
                                    if let Some((y0, y1)) =
                                        lowered_unary_eval_pair(op, x0, x1, mode)
                                    {
                                        log_simd_math_exec_once(entry, op);
                                        lane_acc[lane] += y0;
                                        lane_acc[lane + 1] += y1;
                                        lane += 2;
                                        continue;
                                    }
                                }
                                lane_acc[lane] += eval_unary(op, input_view, i + lane);
                                lane += 1;
                            }
                            i += lanes;
                        }
                        let mut acc = lane_acc[..lanes].iter().copied().sum::<f64>();
                        while i < len {
                            acc += eval_unary(op, input_view, i);
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(acc));
                    }
                    ReductionMode::Any => {
                        let mut lane_any = [false; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for (lane, item) in lane_any.iter_mut().enumerate().take(lanes) {
                                *item |= eval_unary(op, input_view, i + lane) != 0.0;
                            }
                            if lane_any[..lanes].iter().any(|v| *v) {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_unary(op, input_view, i) != 0.0 {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(0.0));
                    }
                    ReductionMode::All => {
                        let mut lane_all = [true; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for (lane, item) in lane_all.iter_mut().enumerate().take(lanes) {
                                *item &= eval_unary(op, input_view, i + lane) != 0.0;
                            }
                            if lane_all[..lanes].iter().any(|v| !*v) {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_unary(op, input_view, i) == 0.0 {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(1.0));
                    }
                    ReductionMode::None => {}
                }
            }
            LoweredKernel::Binary { op, lhs, rhs } => {
                let lhs_view = views.get(lhs)?;
                let rhs_view = views.get(rhs)?;
                match entry.reduction {
                    ReductionMode::Sum => {
                        let mut lane_acc = [0.0_f64; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for (lane, item) in lane_acc.iter_mut().enumerate().take(lanes) {
                                *item += eval_binary(op, lhs_view, rhs_view, i + lane);
                            }
                            i += lanes;
                        }
                        let mut acc = lane_acc[..lanes].iter().copied().sum::<f64>();
                        while i < len {
                            acc += eval_binary(op, lhs_view, rhs_view, i);
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(acc));
                    }
                    ReductionMode::Any => {
                        let mut lane_any = [false; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for (lane, item) in lane_any.iter_mut().enumerate().take(lanes) {
                                *item |= eval_binary(op, lhs_view, rhs_view, i + lane) != 0.0;
                            }
                            if lane_any[..lanes].iter().any(|v| *v) {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_binary(op, lhs_view, rhs_view, i) != 0.0 {
                                return Some(LoweredVectorResult::Reduced(1.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(0.0));
                    }
                    ReductionMode::All => {
                        let mut lane_all = [true; 4];
                        let mut i = 0;
                        while i + lanes <= len {
                            for (lane, item) in lane_all.iter_mut().enumerate().take(lanes) {
                                *item &= eval_binary(op, lhs_view, rhs_view, i + lane) != 0.0;
                            }
                            if lane_all[..lanes].iter().any(|v| !*v) {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += lanes;
                        }
                        while i < len {
                            if eval_binary(op, lhs_view, rhs_view, i) == 0.0 {
                                return Some(LoweredVectorResult::Reduced(0.0));
                            }
                            i += 1;
                        }
                        return Some(LoweredVectorResult::Reduced(1.0));
                    }
                    ReductionMode::None => {}
                }
            }
            LoweredKernel::Expr(expr) => match entry.reduction {
                ReductionMode::Sum => {
                    let mut lane_acc = [0.0_f64; 4];
                    let mut i = 0;
                    while i + lanes <= len {
                        let mut lane = 0;
                        while lane < lanes {
                            if lane + 1 < lanes {
                                let idx0 = i + lane;
                                let idx1 = i + lane + 1;
                                let (v0, v1) =
                                    lowered_expr_eval_pair(&expr, views, idx0, idx1, mode)?;
                                lane_acc[lane] += v0;
                                lane_acc[lane + 1] += v1;
                                lane += 2;
                                continue;
                            }
                            lane_acc[lane] += lowered_expr_eval(&expr, views, i + lane, mode)?;
                            lane += 1;
                        }
                        i += lanes;
                    }
                    let mut acc = lane_acc[..lanes].iter().copied().sum::<f64>();
                    while i < len {
                        acc += lowered_expr_eval(&expr, views, i, mode)?;
                        i += 1;
                    }
                    return Some(LoweredVectorResult::Reduced(acc));
                }
                ReductionMode::Any => {
                    let mut lane_any = [false; 4];
                    let mut i = 0;
                    while i + lanes <= len {
                        for (lane, item) in lane_any.iter_mut().enumerate().take(lanes) {
                            *item |= lowered_expr_eval(&expr, views, i + lane, mode)? != 0.0;
                        }
                        if lane_any[..lanes].iter().any(|v| *v) {
                            return Some(LoweredVectorResult::Reduced(1.0));
                        }
                        i += lanes;
                    }
                    while i < len {
                        if lowered_expr_eval(&expr, views, i, mode)? != 0.0 {
                            return Some(LoweredVectorResult::Reduced(1.0));
                        }
                        i += 1;
                    }
                    return Some(LoweredVectorResult::Reduced(0.0));
                }
                ReductionMode::All => {
                    let mut lane_all = [true; 4];
                    let mut i = 0;
                    while i + lanes <= len {
                        for (lane, item) in lane_all.iter_mut().enumerate().take(lanes) {
                            *item &= lowered_expr_eval(&expr, views, i + lane, mode)? != 0.0;
                        }
                        if lane_all[..lanes].iter().any(|v| !*v) {
                            return Some(LoweredVectorResult::Reduced(0.0));
                        }
                        i += lanes;
                    }
                    while i < len {
                        if lowered_expr_eval(&expr, views, i, mode)? == 0.0 {
                            return Some(LoweredVectorResult::Reduced(0.0));
                        }
                        i += 1;
                    }
                    return Some(LoweredVectorResult::Reduced(1.0));
                }
                ReductionMode::None => {}
            },
        }
        return None;
    }

    let mut results = Vec::with_capacity(len);

    match kernel {
        LoweredKernel::Unary { op, input } => {
            let input_view = views.get(input)?;
            let mut i = 0;
            while i + unroll <= len {
                let mut lane = 0;
                while lane < unroll {
                    if lane + 1 < unroll {
                        let idx = i + lane;
                        let x0 = unsafe { read_buffer_f64(input_view, idx) };
                        let x1 = unsafe { read_buffer_f64(input_view, idx + 1) };
                        if let Some((y0, y1)) = lowered_unary_eval_pair(op, x0, x1, mode) {
                            log_simd_math_exec_once(entry, op);
                            results.push(y0);
                            results.push(y1);
                            lane += 2;
                            continue;
                        }
                    }
                    let idx = i + lane;
                    results.push(eval_unary(op, input_view, idx));
                    lane += 1;
                }
                i += unroll;
            }
            while i < len {
                results.push(eval_unary(op, input_view, i));
                i += 1;
            }
            Some(LoweredVectorResult::Vector(results))
        }
        LoweredKernel::Binary { op, lhs, rhs } => {
            let lhs_view = views.get(lhs)?;
            let rhs_view = views.get(rhs)?;
            let mut i = 0;
            while i + unroll <= len {
                for lane in 0..unroll {
                    let idx = i + lane;
                    results.push(eval_binary(op, lhs_view, rhs_view, idx));
                }
                i += unroll;
            }
            while i < len {
                results.push(eval_binary(op, lhs_view, rhs_view, i));
                i += 1;
            }
            Some(LoweredVectorResult::Vector(results))
        }
        LoweredKernel::Expr(expr) => {
            let mut i = 0;
            while i + unroll <= len {
                let mut lane = 0;
                while lane < unroll {
                    if lane + 1 < unroll {
                        let idx0 = i + lane;
                        let idx1 = i + lane + 1;
                        let (v0, v1) = lowered_expr_eval_pair(&expr, views, idx0, idx1, mode)?;
                        results.push(v0);
                        results.push(v1);
                        lane += 2;
                        continue;
                    }
                    results.push(lowered_expr_eval(&expr, views, i + lane, mode)?);
                    lane += 1;
                }
                i += unroll;
            }
            while i < len {
                results.push(lowered_expr_eval(&expr, views, i, mode)?);
                i += 1;
            }
            Some(LoweredVectorResult::Vector(results))
        }
    }
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn lookup_exec_profile(func_ptr: usize) -> Option<JitExecProfile> {
    TLS_JIT_TYPE_PROFILE.with(|m| m.borrow().get(&func_ptr).cloned())
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn store_exec_profile(func_ptr: usize, profile: JitExecProfile) {
    TLS_JIT_TYPE_PROFILE.with(|m| {
        m.borrow_mut().insert(func_ptr, profile);
    });
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn reduction_identity(mode: ReductionMode) -> f64 {
    match mode {
        ReductionMode::None => 0.0,
        ReductionMode::Sum => 0.0,
        ReductionMode::Any => 0.0,
        ReductionMode::All => 1.0,
    }
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn reduction_step(mode: ReductionMode, acc: &mut f64, value: f64) -> bool {
    let bits = value.to_bits();
    if bits == BREAK_SENTINEL_BITS {
        return true;
    }
    if bits == CONTINUE_SENTINEL_BITS {
        return false;
    }

    match mode {
        ReductionMode::None => false,
        ReductionMode::Sum => {
            *acc += value;
            false
        }
        ReductionMode::Any => {
            if value != 0.0 {
                *acc = 1.0;
                true
            } else {
                false
            }
        }
        ReductionMode::All => {
            if value == 0.0 {
                *acc = 0.0;
                true
            } else {
                false
            }
        }
    }
}

#[inline(always)]
pub(crate) fn simd_unroll_factor_for_plan(plan: simd::SimdPlan) -> usize {
    if !plan.auto_vectorize {
        return 1;
    }
    let lanes = (plan.lane_bytes / std::mem::size_of::<f64>()).max(1);
    lanes.min(4)
}

#[inline(always)]
pub(crate) fn simd_unroll_factor() -> usize {
    simd_unroll_factor_for_plan(simd::auto_vectorization_plan())
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn jit_return_to_py(py: Python<'_>, value: f64, return_type: JitReturnType) -> Py<PyAny> {
    match return_type {
        JitReturnType::Float => value.into_py_any(py).unwrap(),
        JitReturnType::Int => (value as i64).into_py_any(py).unwrap(),
        JitReturnType::Bool => (value != 0.0).into_py_any(py).unwrap(),
    }
}

#[cfg(feature = "pyo3")]
#[inline(always)]
fn vec_f64_to_py(py: Python<'_>, values: &[f64], return_type: JitReturnType) -> Py<PyAny> {
    use pyo3::types::PyBytes;
    use pyo3::types::PyList;
    match return_type {
        JitReturnType::Float => {
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    std::mem::size_of_val(values),
                )
            };
            let py_bytes = PyBytes::new(py, byte_slice);
            let array_mod = py.import(pyo3::ffi::c_str!("array")).unwrap();
            let array_obj = array_mod
                .getattr("array")
                .unwrap()
                .call1(("d", py_bytes))
                .unwrap();
            array_obj.unbind()
        }
        JitReturnType::Int => {
            let list = PyList::new(py, values.iter().map(|v| *v as i64)).unwrap();
            list.unbind().into_any()
        }
        JitReturnType::Bool => {
            let list = PyList::new(py, values.iter().map(|v| *v != 0.0)).unwrap();
            list.unbind().into_any()
        }
    }
}

/// Shared execution helper for vectorized buffer evaluation (both generic and profiled).
#[cfg(feature = "pyo3")]
#[allow(clippy::too_many_arguments)]
fn execute_views(
    py: Python<'_>,
    entry: &JitEntry,
    f: extern "C" fn(*const f64) -> f64,
    loop_unroll: usize,
    views: &[BufferView],
    len: usize,
    arg_count: usize,
    log_path: &'static str,
) -> PyResult<Py<PyAny>> {
    log_unroll_exec_once(entry, len, loop_unroll, log_path);
    if let Some(lowered) = try_execute_lowered_vector_kernel(entry, views, len, loop_unroll) {
        log_lowered_exec_once(entry, len);
        match lowered {
            LoweredVectorResult::Vector(results) => {
                return Ok(vec_f64_to_py(py, &results, entry.return_type));
            }
            LoweredVectorResult::Reduced(value) => {
                return Ok(jit_return_to_py(py, value, entry.return_type));
            }
        }
    }

    if entry.reduction != ReductionMode::None {
        let mut acc = reduction_identity(entry.reduction);
        const MAX_FAST_ARGS: usize = 8;
        if arg_count <= MAX_FAST_ARGS {
            let mut i = 0;
            let mut should_break = false;
            while i + loop_unroll <= len {
                for lane in 0..loop_unroll {
                    let idx = i + lane;
                    let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                    for (j, view) in views.iter().enumerate().take(arg_count) {
                        iter_args[j] = unsafe { read_buffer_f64(view, idx) };
                    }
                    let val = f(iter_args.as_ptr());
                    if reduction_step(entry.reduction, &mut acc, val) {
                        should_break = true;
                        break;
                    }
                }
                if should_break {
                    break;
                }
                i += loop_unroll;
            }
            while i < len {
                let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                for (j, view) in views.iter().enumerate().take(arg_count) {
                    iter_args[j] = unsafe { read_buffer_f64(view, i) };
                }
                let val = f(iter_args.as_ptr());
                if reduction_step(entry.reduction, &mut acc, val) {
                    break;
                }
                i += 1;
            }
        } else {
            let mut i = 0;
            let mut should_break = false;
            while i + loop_unroll <= len {
                for lane in 0..loop_unroll {
                    let idx = i + lane;
                    let mut iter_args = Vec::with_capacity(arg_count);
                    for view in views.iter().take(arg_count) {
                        iter_args.push(unsafe { read_buffer_f64(view, idx) });
                    }
                    let val = f(iter_args.as_ptr());
                    if reduction_step(entry.reduction, &mut acc, val) {
                        should_break = true;
                        break;
                    }
                }
                if should_break {
                    break;
                }
                i += loop_unroll;
            }
            while i < len {
                let mut iter_args = Vec::with_capacity(arg_count);
                for view in views.iter().take(arg_count) {
                    iter_args.push(unsafe { read_buffer_f64(view, i) });
                }
                let val = f(iter_args.as_ptr());
                if reduction_step(entry.reduction, &mut acc, val) {
                    break;
                }
                i += 1;
            }
        }
        return Ok(jit_return_to_py(py, acc, entry.return_type));
    }

    let mut results = Vec::with_capacity(len);
    const MAX_FAST_ARGS: usize = 8;
    if arg_count <= MAX_FAST_ARGS {
        let mut i = 0;
        while i + loop_unroll <= len {
            for lane in 0..loop_unroll {
                let idx = i + lane;
                let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                for (j, view) in views.iter().enumerate().take(arg_count) {
                    iter_args[j] = unsafe { read_buffer_f64(view, idx) };
                }
                results.push(f(iter_args.as_ptr()));
            }
            i += loop_unroll;
        }
        while i < len {
            let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
            for (j, view) in views.iter().enumerate().take(arg_count) {
                iter_args[j] = unsafe { read_buffer_f64(view, i) };
            }
            results.push(f(iter_args.as_ptr()));
            i += 1;
        }
    } else {
        let mut i = 0;
        while i + loop_unroll <= len {
            for lane in 0..loop_unroll {
                let idx = i + lane;
                let mut iter_args = Vec::with_capacity(arg_count);
                for view in views.iter().take(arg_count) {
                    iter_args.push(unsafe { read_buffer_f64(view, idx) });
                }
                results.push(f(iter_args.as_ptr()));
            }
            i += loop_unroll;
        }
        while i < len {
            let mut iter_args = Vec::with_capacity(arg_count);
            for view in views.iter().take(arg_count) {
                iter_args.push(unsafe { read_buffer_f64(view, i) });
            }
            results.push(f(iter_args.as_ptr()));
            i += 1;
        }
    }
    Ok(vec_f64_to_py(py, &results, entry.return_type))
}

#[cfg(feature = "pyo3")]
fn try_exec_trailing_count(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
    f: extern "C" fn(*const f64) -> f64,
    loop_unroll: usize,
    arg_count: usize,
) -> PyResult<Option<Py<PyAny>>> {
    if arg_count != entry.arg_count + 1 {
        return Ok(None);
    }

    let Ok(count_item) = args.get_item(arg_count - 1) else {
        return Ok(None);
    };

    let Ok(count_i64) = count_item.extract::<i64>() else {
        return Ok(None);
    };

    if count_i64 < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "count must be non-negative",
        ));
    }

    let count = count_i64 as usize;
    const MAX_FAST_ARGS: usize = 8;
    let mut results = Vec::with_capacity(count);
    log_unroll_exec_once(entry, count, loop_unroll, "trailing-count");

    if entry.arg_count <= MAX_FAST_ARGS {
        let mut stack_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
        for (i, item) in stack_args.iter_mut().enumerate().take(entry.arg_count) {
            *item = args.get_item(i)?.extract::<f64>()?;
        }
        let mut produced = 0;
        while produced + loop_unroll <= count {
            for _ in 0..loop_unroll {
                results.push(f(stack_args.as_ptr()));
            }
            produced += loop_unroll;
        }
        while produced < count {
            results.push(f(stack_args.as_ptr()));
            produced += 1;
        }
    } else {
        let mut heap_args = Vec::with_capacity(entry.arg_count);
        for i in 0..entry.arg_count {
            let val = args.get_item(i)?.extract::<f64>()?;
            heap_args.push(val);
        }
        let mut produced = 0;
        while produced + loop_unroll <= count {
            for _ in 0..loop_unroll {
                results.push(f(heap_args.as_ptr()));
            }
            produced += loop_unroll;
        }
        while produced < count {
            results.push(f(heap_args.as_ptr()));
            produced += 1;
        }
    }

    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            results.as_ptr() as *const u8,
            results.len() * std::mem::size_of::<f64>(),
        )
    };
    let py_bytes = pyo3::types::PyBytes::new(py, byte_slice);
    let array_mod = py.import(pyo3::ffi::c_str!("array"))?;
    let array_obj = array_mod.getattr("array")?.call1(("d", py_bytes))?;
    Ok(Some(array_obj.unbind()))
}

#[cfg(feature = "pyo3")]
fn try_exec_profiled(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
    f: extern "C" fn(*const f64) -> f64,
    loop_unroll: usize,
    arg_count: usize,
) -> PyResult<Option<Py<PyAny>>> {
    let Some(profile) = lookup_exec_profile(entry.func_ptr) else {
        return Ok(None);
    };

    match profile {
        JitExecProfile::PackedBuffer {
            arg_count: expected,
            elem,
        } => {
            if arg_count == 1 && entry.arg_count == expected {
                if let Ok(item) = args.get_item(0) {
                    if let Some(view) = unsafe { open_typed_buffer(&item) } {
                        if view.elem_type == elem && view.len == expected {
                            if elem == BufferElemType::F64 && view.is_aligned_for_f64() {
                                let res = f(view.as_ptr_f64());
                                return Ok(Some(jit_return_to_py(py, res, entry.return_type)));
                            }
                            let mut converted = Vec::with_capacity(expected);
                            for i in 0..expected {
                                converted.push(unsafe { read_buffer_f64(&view, i) });
                            }
                            let res = f(converted.as_ptr());
                            return Ok(Some(jit_return_to_py(py, res, entry.return_type)));
                        }
                    }
                }
            }
        }
        JitExecProfile::VectorizedBuffers {
            arg_count: expected,
            elem_types,
        } => {
            if arg_count == expected && expected == elem_types.len() {
                let mut views = Vec::with_capacity(expected);
                let mut common_len: Option<usize> = None;
                let mut matched = true;
                for (i, expected_elem_type) in elem_types.iter().enumerate().take(expected) {
                    let Ok(item) = args.get_item(i) else {
                        matched = false;
                        break;
                    };
                    let Some(view) = (unsafe { open_typed_buffer(&item) }) else {
                        matched = false;
                        break;
                    };
                    if view.elem_type != *expected_elem_type {
                        matched = false;
                        break;
                    }
                    if let Some(len) = common_len {
                        if view.len != len {
                            matched = false;
                            break;
                        }
                    } else {
                        common_len = Some(view.len);
                    }
                    views.push(view);
                }

                if matched {
                    let len = common_len.unwrap_or(0);
                    let result = execute_views(
                        py,
                        entry,
                        f,
                        loop_unroll,
                        &views,
                        len,
                        expected,
                        "profiled-vector-buffers",
                    )?;
                    return Ok(Some(result));
                }
            }
        }
        JitExecProfile::ScalarArgs => {
            if arg_count == entry.arg_count {
                const MAX_FAST_ARGS: usize = 8;
                if arg_count <= MAX_FAST_ARGS {
                    let mut stack_args = [0.0_f64; MAX_FAST_ARGS];
                    let mut scalar_mismatch = false;
                    for (i, arg) in stack_args.iter_mut().enumerate().take(arg_count) {
                        let item = unsafe { pyo3::ffi::PyTuple_GetItem(args.as_ptr(), i as isize) };
                        let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
                        if (val == -1.0) && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
                            unsafe { pyo3::ffi::PyErr_Clear() };
                            scalar_mismatch = true;
                            break;
                        }
                        *arg = val;
                    }
                    if !scalar_mismatch {
                        let res = f(stack_args.as_ptr());
                        return Ok(Some(jit_return_to_py(py, res, entry.return_type)));
                    }
                }
            }
        }
    }
    Ok(None)
}

#[cfg(feature = "pyo3")]
fn try_exec_single_packed_buffer(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
    f: extern "C" fn(*const f64) -> f64,
    arg_count: usize,
) -> PyResult<Option<Py<PyAny>>> {
    if arg_count == 1 && entry.arg_count > 1 {
        if let Ok(item) = args.get_item(0) {
            if let Some(view) = unsafe { open_typed_buffer(&item) } {
                let len = view.len;
                if len == entry.arg_count {
                    let res = if view.elem_type == BufferElemType::F64 {
                        if view.is_aligned_for_f64() {
                            f(view.as_ptr_f64())
                        } else {
                            let mut converted = Vec::with_capacity(len);
                            for i in 0..len {
                                converted.push(unsafe { read_buffer_f64(&view, i) });
                            }
                            f(converted.as_ptr())
                        }
                    } else {
                        let mut converted = Vec::with_capacity(len);
                        for i in 0..len {
                            converted.push(unsafe { read_buffer_f64(&view, i) });
                        }
                        f(converted.as_ptr())
                    };
                    store_exec_profile(
                        entry.func_ptr,
                        JitExecProfile::PackedBuffer {
                            arg_count: entry.arg_count,
                            elem: view.elem_type,
                        },
                    );
                    return Ok(Some(jit_return_to_py(py, res, entry.return_type)));
                }
            }
        }
    }
    Ok(None)
}

#[cfg(feature = "pyo3")]
fn try_exec_vectorized_buffers(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
    f: extern "C" fn(*const f64) -> f64,
    loop_unroll: usize,
    arg_count: usize,
) -> PyResult<Option<Py<PyAny>>> {
    if arg_count == entry.arg_count && arg_count > 0 {
        let mut views = Vec::with_capacity(arg_count);
        let mut common_len = None;
        let mut all_buffers = true;

        for i in 0..arg_count {
            if let Ok(item) = args.get_item(i) {
                if let Some(view) = unsafe { open_typed_buffer(&item) } {
                    let len = view.len;
                    if let Some(c_len) = common_len {
                        if len != c_len {
                            all_buffers = false;
                            break;
                        }
                    } else {
                        common_len = Some(len);
                    }
                    views.push(view);
                } else {
                    all_buffers = false;
                    break;
                }
            } else {
                all_buffers = false;
                break;
            }
        }

        if all_buffers {
            if let Some(len) = common_len {
                let elem_types = views.iter().map(|v| v.elem_type).collect::<Vec<_>>();
                store_exec_profile(
                    entry.func_ptr,
                    JitExecProfile::VectorizedBuffers {
                        arg_count,
                        elem_types,
                    },
                );

                let result = execute_views(
                    py,
                    entry,
                    f,
                    loop_unroll,
                    &views,
                    len,
                    arg_count,
                    "generic-vector-buffers",
                )?;
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

#[cfg(feature = "pyo3")]
fn try_exec_sequence_fallback(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
    f: extern "C" fn(*const f64) -> f64,
    loop_unroll: usize,
    arg_count: usize,
) -> PyResult<Option<Py<PyAny>>> {
    if arg_count == 1 && entry.arg_count == 1 {
        if let Ok(item) = args.get_item(0) {
            let is_text_like = item.is_instance_of::<pyo3::types::PyString>()
                || item.is_instance_of::<pyo3::types::PyBytes>()
                || item.is_instance_of::<pyo3::types::PyByteArray>();

            if !is_text_like {
                if let Ok(len) = item.len() {
                    log_unroll_exec_once(entry, len, loop_unroll, "sequence-fallback");
                    if entry.reduction != ReductionMode::None {
                        let mut acc = reduction_identity(entry.reduction);
                        let mut i = 0;
                        while i + loop_unroll <= len {
                            for lane in 0..loop_unroll {
                                let idx = i + lane;
                                let elem = item.get_item(idx)?;
                                let val: f64 = elem.extract()?;
                                let arg0 = [val];
                                let out = f(arg0.as_ptr());
                                if reduction_step(entry.reduction, &mut acc, out) {
                                    return Ok(Some(jit_return_to_py(py, acc, entry.return_type)));
                                }
                            }
                            i += loop_unroll;
                        }
                        while i < len {
                            let elem = item.get_item(i)?;
                            let val: f64 = elem.extract()?;
                            let arg0 = [val];
                            let out = f(arg0.as_ptr());
                            if reduction_step(entry.reduction, &mut acc, out) {
                                break;
                            }
                            i += 1;
                        }
                        return Ok(Some(jit_return_to_py(py, acc, entry.return_type)));
                    }

                    let mut results = Vec::with_capacity(len);
                    let mut i = 0;
                    while i + loop_unroll <= len {
                        for lane in 0..loop_unroll {
                            let idx = i + lane;
                            let elem = item.get_item(idx)?;
                            let val: f64 = elem.extract()?;
                            let arg0 = [val];
                            results.push(f(arg0.as_ptr()));
                        }
                        i += loop_unroll;
                    }
                    while i < len {
                        let elem = item.get_item(i)?;
                        let val: f64 = elem.extract()?;
                        let arg0 = [val];
                        results.push(f(arg0.as_ptr()));
                        i += 1;
                    }
                    return Ok(Some(vec_f64_to_py(py, &results, entry.return_type)));
                }
            }
        }
    }
    Ok(None)
}

#[cfg(feature = "pyo3")]
fn try_exec_reduction_iterator(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
    f: extern "C" fn(*const f64) -> f64,
    arg_count: usize,
) -> PyResult<Option<Py<PyAny>>> {
    if arg_count == 1 && entry.arg_count == 1 && entry.reduction != ReductionMode::None {
        if let Ok(item) = args.get_item(0) {
            if let Ok(iter) = item.try_iter() {
                let mut acc = reduction_identity(entry.reduction);
                let mut buf = [0.0_f64; 1];
                for obj_res in iter {
                    let obj: Bound<'_, PyAny> = obj_res?;
                    let val: f64 = obj.extract()?;
                    buf[0] = val;
                    let out = f(buf.as_ptr());
                    if reduction_step(entry.reduction, &mut acc, out) {
                        break;
                    }
                }
                return Ok(Some(jit_return_to_py(py, acc, entry.return_type)));
            }
        }
    }
    Ok(None)
}

#[cfg(feature = "pyo3")]
fn exec_scalar_args(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
    f: extern "C" fn(*const f64) -> f64,
    arg_count: usize,
) -> PyResult<Py<PyAny>> {
    if arg_count != entry.arg_count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "wrong argument count for JIT function",
        ));
    }

    const MAX_FAST_ARGS: usize = 8;
    if arg_count <= MAX_FAST_ARGS {
        let mut stack_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
        for (i, arg) in stack_args.iter_mut().enumerate().take(arg_count) {
            *arg = args.get_item(i)?.extract::<f64>()?;
        }
        store_exec_profile(entry.func_ptr, JitExecProfile::ScalarArgs);
        let res = f(stack_args.as_ptr());
        return Ok(jit_return_to_py(py, res, entry.return_type));
    }

    let mut heap_args = Vec::with_capacity(arg_count);
    for i in 0..arg_count {
        let val = args.get_item(i)?.extract::<f64>()?;
        heap_args.push(val);
    }
    store_exec_profile(entry.func_ptr, JitExecProfile::ScalarArgs);
    let res = f(heap_args.as_ptr());
    Ok(jit_return_to_py(py, res, entry.return_type))
}

#[cfg(feature = "pyo3")]
#[inline(always)]
pub fn execute_jit_func(
    py: Python<'_>,
    entry: &JitEntry,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    if entry.func_ptr == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "JIT function pointer is invalid or not yet compiled",
        ));
    }

    let arg_count = args.len();
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let loop_unroll = entry.loop_unroll_for_entry();

    if let Some(res) = try_exec_trailing_count(py, entry, args, f, loop_unroll, arg_count)? {
        return Ok(res);
    }
    if let Some(res) = try_exec_profiled(py, entry, args, f, loop_unroll, arg_count)? {
        return Ok(res);
    }
    if let Some(res) = try_exec_single_packed_buffer(py, entry, args, f, arg_count)? {
        return Ok(res);
    }
    if let Some(res) = try_exec_vectorized_buffers(py, entry, args, f, loop_unroll, arg_count)? {
        return Ok(res);
    }
    if let Some(res) = try_exec_sequence_fallback(py, entry, args, f, loop_unroll, arg_count)? {
        return Ok(res);
    }
    if let Some(res) = try_exec_reduction_iterator(py, entry, args, f, arg_count)? {
        return Ok(res);
    }

    exec_scalar_args(py, entry, args, f, arg_count)
}

#[cfg(test)]
mod simd_unroll_tests {
    use super::*;

    #[test]
    fn scalar_plan_uses_unroll_1() {
        let plan = simd::SimdPlan {
            backend: simd::SimdBackend::Scalar,
            lane_bytes: 8,
            auto_vectorize: false,
        };
        assert_eq!(simd_unroll_factor_for_plan(plan), 1);
    }

    #[test]
    fn neon_plan_uses_unroll_2() {
        let plan = simd::SimdPlan {
            backend: simd::SimdBackend::ArmNeon,
            lane_bytes: 16,
            auto_vectorize: true,
        };
        assert_eq!(simd_unroll_factor_for_plan(plan), 2);
    }

    #[test]
    fn wide_simd_plans_cap_at_unroll_4() {
        let sve = simd::SimdPlan {
            backend: simd::SimdBackend::ArmSve,
            lane_bytes: 32,
            auto_vectorize: true,
        };
        let avx2 = simd::SimdPlan {
            backend: simd::SimdBackend::X86Avx2,
            lane_bytes: 32,
            auto_vectorize: true,
        };
        assert_eq!(simd_unroll_factor_for_plan(sve), 4);
        assert_eq!(simd_unroll_factor_for_plan(avx2), 4);
    }

    #[test]
    fn narrow_vector_plan_stays_unroll_1() {
        let plan = simd::SimdPlan {
            backend: simd::SimdBackend::WasmSimd128,
            lane_bytes: 8,
            auto_vectorize: true,
        };
        assert_eq!(simd_unroll_factor_for_plan(plan), 1);
    }
}
