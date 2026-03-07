// src/py/jit/codegen.rs
//! Core JIT compilation logic, including registry and Cranelift codegen.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::py::jit::parser::Expr;
use crate::py::jit::heuristics;

// cranelift imports
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use pyo3::AsPyPointer;
use pyo3::IntoPy;

const BREAK_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0b01;
const CONTINUE_SENTINEL_BITS: u64 = 0x7ff8_0000_0000_0c01;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SymbolAlias {
    Identity,
    Rename(&'static str),
}

pub(crate) fn resolve_symbol_alias(symbol: &str, arg_count: usize) -> Option<SymbolAlias> {
    match (symbol, arg_count) {
        ("float", 1) => Some(SymbolAlias::Identity),
        ("int", 1) => Some(SymbolAlias::Rename("trunc")),
        ("round", 1) => Some(SymbolAlias::Rename("round")),
        _ => None,
    }
}

/// A compiled function entry returned by the JIT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionMode {
    None,
    Sum,
    Any,
    All,
}

#[derive(Clone)]
pub struct JitEntry {
    pub func_ptr: usize,
    pub arg_count: usize,
    pub reduction: ReductionMode,
}

static JIT_FUNC_COUNTER: once_cell::sync::Lazy<AtomicUsize> =
    once_cell::sync::Lazy::new(|| AtomicUsize::new(0));

static JIT_REGISTRY: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<usize, JitEntry>>> =
    once_cell::sync::OnceCell::new();

#[derive(Clone)]
struct QuantumStats {
    ewma_ns: f64,
    runs: u64,
    failures: u64,
}

impl Default for QuantumStats {
    fn default() -> Self {
        Self {
            ewma_ns: 0.0,
            runs: 0,
            failures: 0,
        }
    }
}

#[derive(Clone)]
struct QuantumState {
    entries: Vec<JitEntry>,
    stats: Vec<QuantumStats>,
    round_robin: usize,
    total_runs: u64,
}

static QUANTUM_REGISTRY: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<usize, QuantumState>>> =
    once_cell::sync::OnceCell::new();

pub fn register_jit(func_key: usize, entry: JitEntry) {
    let map = JIT_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard.insert(func_key, entry);
}

pub fn lookup_jit(func_key: usize) -> Option<JitEntry> {
    JIT_REGISTRY
        .get()
        .and_then(|map| map.lock().unwrap().get(&func_key).cloned())
}

pub fn register_quantum_jit(func_key: usize, mut entries: Vec<JitEntry>) {
    if entries.is_empty() {
        return;
    }
    // prefer optimized candidate (first) as baseline fallback mapping
    register_jit(func_key, entries[0].clone());
    let stats = vec![QuantumStats::default(); entries.len()];
    let state = QuantumState {
        entries: std::mem::take(&mut entries),
        stats,
        round_robin: 0,
        total_runs: 0,
    };
    let map = QUANTUM_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    map.lock().unwrap().insert(func_key, state);
}

fn choose_quantum_index(state: &mut QuantumState) -> usize {
    for (idx, s) in state.stats.iter().enumerate() {
        if s.runs == 0 {
            return idx;
        }
    }
    state.total_runs = state.total_runs.saturating_add(1);
    if state.total_runs % 16 == 0 {
        state.round_robin = (state.round_robin + 1) % state.entries.len();
        return state.round_robin;
    }
    let mut best_idx = 0usize;
    let mut best_score = f64::MAX;
    for (idx, s) in state.stats.iter().enumerate() {
        let penalty = 1.0 + (s.failures as f64 * 0.25);
        let score = if s.ewma_ns > 0.0 { s.ewma_ns * penalty } else { f64::MAX / 4.0 };
        if score < best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

fn update_quantum_stats(func_key: usize, idx: usize, elapsed_ns: u64, success: bool) {
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
        }
    }
}

#[cfg(feature = "pyo3")]
pub fn execute_registered_jit(
    py: pyo3::Python,
    func_key: usize,
    args: &pyo3::types::PyTuple,
) -> Option<pyo3::PyResult<pyo3::PyObject>> {
    if crate::py::jit::quantum_speculation_enabled() {
        if let Some(map) = QUANTUM_REGISTRY.get() {
            let (entry, idx, fallback_entries) = {
                let mut guard = map.lock().unwrap();
                if let Some(state) = guard.get_mut(&func_key) {
                    if state.entries.is_empty() {
                        (None, 0usize, Vec::new())
                    } else {
                        let idx = choose_quantum_index(state);
                        let entry = Some(state.entries[idx].clone());
                        let mut fallbacks = Vec::new();
                        for (i, e) in state.entries.iter().enumerate() {
                            if i != idx {
                                fallbacks.push((i, e.clone()));
                            }
                        }
                        (entry, idx, fallbacks)
                    }
                } else {
                    (None, 0usize, Vec::new())
                }
            };

            if let Some(entry) = entry {
                let start = Instant::now();
                match execute_jit_func(py, &entry, args) {
                    Ok(obj) => {
                        update_quantum_stats(func_key, idx, start.elapsed().as_nanos() as u64, true);
                        return Some(Ok(obj));
                    }
                    Err(primary_err) => {
                        update_quantum_stats(func_key, idx, start.elapsed().as_nanos() as u64, false);
                        for (fb_idx, fb_entry) in fallback_entries {
                            let start_fb = Instant::now();
                            match execute_jit_func(py, &fb_entry, args) {
                                Ok(obj) => {
                                    update_quantum_stats(
                                        func_key,
                                        fb_idx,
                                        start_fb.elapsed().as_nanos() as u64,
                                        true,
                                    );
                                    return Some(Ok(obj));
                                }
                                Err(_) => {
                                    update_quantum_stats(
                                        func_key,
                                        fb_idx,
                                        start_fb.elapsed().as_nanos() as u64,
                                        false,
                                    );
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

thread_local! {
    static TLS_JIT_MODULE: std::cell::RefCell<Option<JITModule>> =
        std::cell::RefCell::new(None);
}

thread_local! {
    static TLS_JIT_TYPE_PROFILE: std::cell::RefCell<HashMap<usize, JitExecProfile>> =
        std::cell::RefCell::new(HashMap::new());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferElemType {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum JitExecProfile {
    ScalarArgs,
    PackedBuffer {
        arg_count: usize,
        elem: BufferElemType,
    },
    VectorizedBuffers {
        arg_count: usize,
        elem_types: Vec<BufferElemType>,
    },
}

fn with_jit_module<F, R>(f: F) -> R
where
    F: FnOnce(&mut JITModule) -> R,
{
    TLS_JIT_MODULE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            // lazily construct the module the first time for this thread
            let mut flag_builder = settings::builder();
            flag_builder.set("use_colocated_libcalls", "false").unwrap();
            if cfg!(target_arch = "aarch64") {
                flag_builder.set("is_pic", "false").unwrap();
            } else {
                flag_builder.set("is_pic", "true").unwrap();
            }
            let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
                panic!("host machine is not supported: {}", msg);
            });
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .expect("failed to create ISA");
            let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
            *opt = Some(JITModule::new(builder));
        }
        let module = opt.as_mut().unwrap();
        f(module)
    })
}

fn compile_jit_impl(expr_str: &str, arg_names: &[String], optimize_ast: bool) -> Option<JitEntry> {
    // tokenize and parse
    let tokens = crate::py::jit::parser::tokenize(expr_str);
    let mut parser = crate::py::jit::parser::Parser::new(tokens);
    let mut expr = parser.parse_expr()?;
    crate::py::jit::jit_log(|| format!("[Iris][jit] parsed AST for '{}': {:?}", expr_str, expr));
    // detect generator-style loop over a container and convert to body-only
    // expression.  Python wrapper will pass the container buffer and the
    // JIT runtime will vectorize across it; the compiled function gets a
    // single scalar argument representing each element.
    let mut adjusted_args = arg_names.to_vec();
    let mut reduction = ReductionMode::None;
    // use a cloned copy when destructuring to release borrow immediately
    if let Expr::SumOver { iter_var, container, body, pred } = expr.clone() {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!("[Iris][jit] converting SumOver '{}' in {}", iter_var, cont_name)
                });
                expr = if let Some(p) = pred {
                    Expr::Ternary(p, body, Box::new(Expr::Const(0.0)))
                } else {
                    *body.clone()
                };
                adjusted_args = vec![iter_var.clone()];
                reduction = ReductionMode::Sum;
            }
        }
    }
    if let Expr::AnyOver { iter_var, container, body, pred } = expr.clone() {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!("[Iris][jit] converting AnyOver '{}' in {}", iter_var, cont_name)
                });
                expr = if let Some(p) = pred {
                    Expr::Ternary(p, body, Box::new(Expr::Const(0.0)))
                } else {
                    *body.clone()
                };
                adjusted_args = vec![iter_var.clone()];
                reduction = ReductionMode::Any;
            }
        }
    }
    if let Expr::AllOver { iter_var, container, body, pred } = expr.clone() {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!("[Iris][jit] converting AllOver '{}' in {}", iter_var, cont_name)
                });
                expr = if let Some(p) = pred {
                    Expr::Ternary(p, body, Box::new(Expr::Const(1.0)))
                } else {
                    *body.clone()
                };
                adjusted_args = vec![iter_var.clone()];
                reduction = ReductionMode::All;
            }
        }
    }
    if optimize_ast {
        expr = heuristics::optimize(expr);
        crate::py::jit::jit_log(|| format!("[Iris][jit] optimized AST: {:?}", expr));
    } else {
        crate::py::jit::jit_log(|| format!("[Iris][jit] baseline AST (no-opt): {:?}", expr));
    }
    let arg_count = adjusted_args.len();

    // perform compilation using the thread-local module instance;
    // the closure returns the resulting pointer so we can pass it back.
    with_jit_module(|module| {
        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(types::I64));
        ctx.func.signature.returns.push(AbiParam::new(types::F64));

        let mut func_ctx = FunctionBuilderContext::new();
        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry = fb.create_block();
            fb.append_block_params_for_function_params(entry);
            fb.switch_to_block(entry);
            fb.seal_block(entry);
            let ptr_val = fb.block_params(entry)[0];
            let locals = HashMap::new();
            let val = gen_expr(&expr, &mut fb, ptr_val, arg_names, module, &locals);
            fb.ins().return_(&[val]);
            fb.finalize();
        }

        let idx = JIT_FUNC_COUNTER.fetch_add(1, Ordering::SeqCst);
        let func_name = format!("jit_func_{}", idx);
        let id = module
            .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
            .ok();
        if id.is_none() {
            return None;
        }
        let id = id.unwrap();
        if let Err(err) = module.define_function(id, &mut ctx) {
            crate::py::jit::jit_log(|| format!("[Iris][jit] define_function failed: {:?}", err));
            return None;
        }
        module.clear_context(&mut ctx);
        module.finalize_definitions();

        let code_ptr = module.get_finalized_function(id) as usize;
        Some(JitEntry {
            func_ptr: code_ptr,
            arg_count,
            reduction,
        })
    })
}

/// Compile an expression string into a native function entry.
pub fn compile_jit(expr_str: &str, arg_names: &[String]) -> Option<JitEntry> {
    compile_jit_impl(expr_str, arg_names, true)
}

/// Compile multiple speculative versions of the same expression in parallel.
pub fn compile_jit_quantum(expr_str: &str, arg_names: &[String]) -> Vec<JitEntry> {
    let expr_opt = expr_str.to_string();
    let expr_base = expr_str.to_string();
    let args_opt = arg_names.to_vec();
    let args_base = arg_names.to_vec();

    let (optimized, baseline) = std::thread::scope(|scope| {
        let h_opt = scope.spawn(move || compile_jit_impl(&expr_opt, &args_opt, true));
        let h_base = scope.spawn(move || compile_jit_impl(&expr_base, &args_base, false));
        (
            h_opt.join().ok().flatten(),
            h_base.join().ok().flatten(),
        )
    });

    let mut out = Vec::new();
    if let Some(e) = optimized {
        out.push(e);
    }
    if let Some(e) = baseline {
        out.push(e);
    }
    out
}

fn gen_expr(
    expr: &Expr,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {

    #[derive(Clone, Copy)]
    enum LoopControl<'a> {
        None,
        Break { cond: &'a Expr, value: &'a Expr, invert_cond: bool },
        Continue { cond: &'a Expr, value: &'a Expr, invert_cond: bool },
    }

    fn detect_loop_control(expr: &Expr) -> LoopControl<'_> {
        match expr {
            Expr::Call(name, args) if args.len() == 2 => {
                let symbol = name.rsplit('.').next().unwrap_or(name.as_str());
                match symbol {
                    "break_if" | "loop_break_if" | "break_when" | "loop_break_when" => LoopControl::Break {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: false,
                    },
                    "break_unless" | "loop_break_unless" => LoopControl::Break {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: true,
                    },
                    "continue_if" | "loop_continue_if" | "continue_when" | "loop_continue_when" => LoopControl::Continue {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: false,
                    },
                    "continue_unless" | "loop_continue_unless" => LoopControl::Continue {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: true,
                    },
                    _ => LoopControl::None,
                }
            }
            _ => LoopControl::None,
        }
    }

    match expr {
        Expr::Const(n) => fb.ins().f64const(*n),
        Expr::Var(name) => {
            if let Some(v) = locals.get(name) {
                return *v;
            }
            // treat named constants
            match name.as_str() {
                "pi" => return fb.ins().f64const(std::f64::consts::PI),
                "e" => return fb.ins().f64const(std::f64::consts::E),
                _ => {}
            }
            let idx = arg_names.iter().position(|n| n == name).unwrap_or(0);
            let offset = (idx as i64) * 8;
            let offset_const = fb.ins().iconst(types::I64, offset);
            let addr1 = fb.ins().iadd(ptr, offset_const);
            fb.ins().load(types::F64, MemFlags::new(), addr1, 0)
        }
        Expr::BinOp(lhs, op, rhs) => {
            let l = gen_expr(lhs, fb, ptr, arg_names, module, locals);
            let r = gen_expr(rhs, fb, ptr, arg_names, module, locals);
            match op.as_str() {
                "+" => fb.ins().fadd(l, r),
                "-" => fb.ins().fsub(l, r),
                "*" => fb.ins().fmul(l, r),
                "/" => fb.ins().fdiv(l, r),
                "%" => {
                    let mut sig = module.make_signature();
                    sig.params.push(AbiParam::new(types::F64));
                    sig.params.push(AbiParam::new(types::F64));
                    sig.returns.push(AbiParam::new(types::F64));
                    let fid = module
                        .declare_function("fmod", Linkage::Import, &sig)
                        .expect("failed to declare fmod");
                    let local = module.declare_func_in_func(fid, &mut fb.func);
                    let call = fb.ins().call(local, &[l, r]);
                    fb.inst_results(call)[0]
                }
                "**" => {
                    // existing exponent handling (unchanged)
                    if let Expr::Const(n) = **rhs {
                        if n == 1.0 {
                            return l;
                        }
                        if n == -1.0 {
                            let one = fb.ins().f64const(1.0);
                            return fb.ins().fdiv(one, l);
                        }
                        if n == 0.5 {
                            let mut sig = module.make_signature();
                            sig.params.push(AbiParam::new(types::F64));
                            sig.returns.push(AbiParam::new(types::F64));
                            let fid = module
                                .declare_function("sqrt", Linkage::Import, &sig)
                                .expect("failed to declare sqrt");
                            let local = module.declare_func_in_func(fid, &mut fb.func);
                            let call = fb.ins().call(local, &[l]);
                            return fb.inst_results(call)[0];
                        }
                        if n.fract() == 0.0 {
                            let exp = n as i64;
                            if exp == 0 {
                                return fb.ins().f64const(1.0);
                            } else if exp > 0 {
                                let mut e = exp as u64;
                                let mut base_val = l;
                                let mut acc = fb.ins().f64const(1.0);
                                while e > 0 {
                                    if e & 1 == 1 {
                                        acc = fb.ins().fmul(acc, base_val);
                                    }
                                    e >>= 1;
                                    if e > 0 {
                                        base_val = fb.ins().fmul(base_val, base_val);
                                    }
                                }
                                return acc;
                            }
                        }
                    }
                    let mut sig = module.make_signature();
                    sig.params.push(AbiParam::new(types::F64));
                    sig.params.push(AbiParam::new(types::F64));
                    sig.returns.push(AbiParam::new(types::F64));
                    let fid = module
                        .declare_function("pow", Linkage::Import, &sig)
                        .expect("failed to declare pow");
                    let local = module.declare_func_in_func(fid, &mut fb.func);
                    let call = fb.ins().call(local, &[l, r]);
                    fb.inst_results(call)[0]
                }
                "<" | ">" | "<=" | ">=" | "==" | "!=" => {
                    // comparison produces 1.0/0.0
                    let cc = match op.as_str() {
                        "<" => FloatCC::LessThan,
                        ">" => FloatCC::GreaterThan,
                        "<=" => FloatCC::LessThanOrEqual,
                        ">=" => FloatCC::GreaterThanOrEqual,
                        "==" => FloatCC::Equal,
                        "!=" => FloatCC::NotEqual,
                        _ => unreachable!(),
                    };
                    let boolv = fb.ins().fcmp(cc, l, r);
                    let intv = fb.ins().bint(types::I64, boolv);
                    fb.ins().fcvt_from_sint(types::F64, intv)
                }
                "and" | "or" => {
                    let zero = fb.ins().f64const(0.0);
                    let l_true = fb.ins().fcmp(FloatCC::NotEqual, l, zero);
                    let r_true = fb.ins().fcmp(FloatCC::NotEqual, r, zero);
                    let boolv = if op == "and" {
                        fb.ins().band(l_true, r_true)
                    } else {
                        fb.ins().bor(l_true, r_true)
                    };
                    let intv = fb.ins().bint(types::I64, boolv);
                    fb.ins().fcvt_from_sint(types::F64, intv)
                }
                _ => fb.ins().fadd(l, r),
            }
        }
        Expr::UnaryOp(op, sub) => {
            let v = gen_expr(sub, fb, ptr, arg_names, module, locals);
            match op {
                '-' => {
                    let zero = fb.ins().f64const(0.0);
                    fb.ins().fsub(zero, v)
                }
                '!' => {
                    let zero = fb.ins().f64const(0.0);
                    let boolv = fb.ins().fcmp(FloatCC::Equal, v, zero);
                    let intv = fb.ins().bint(types::I64, boolv);
                    fb.ins().fcvt_from_sint(types::F64, intv)
                }
                '+' => v,
                _ => v,
            }
        }
        Expr::Call(name, args) => {
            let symbol = name.rsplit('.').next().unwrap().to_string();
            if (symbol == "any_while" || symbol == "all_while") && args.len() == 5 {
                if let Expr::Var(iter_name) = &args[0] {
                    let init_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                    let cond_expr = &args[2];
                    let step_expr = &args[3];
                    let body_expr_raw = &args[4];

                    let zero = fb.ins().f64const(0.0);
                    let one = fb.ins().f64const(1.0);
                    let false_mask = fb.ins().fcmp(FloatCC::Equal, zero, one);
                    let true_mask = fb.ins().fcmp(FloatCC::Equal, zero, zero);
                    let acc_init = if symbol == "any_while" { zero } else { one };

                    let loop_block = fb.create_block();
                    let body_block = fb.create_block();
                    let continue_block = fb.create_block();
                    let exit_block = fb.create_block();
                    fb.append_block_param(loop_block, types::F64); // iter
                    fb.append_block_param(loop_block, types::F64); // acc
                    fb.append_block_param(loop_block, types::I64); // budget
                    fb.append_block_param(continue_block, types::F64); // next_iter
                    fb.append_block_param(continue_block, types::F64); // next_acc
                    fb.append_block_param(continue_block, types::I64); // next_budget
                    fb.append_block_param(exit_block, types::F64); // result

                    let max_iters = fb.ins().iconst(types::I64, 1_000_000);
                    fb.ins().jump(loop_block, &[init_val, acc_init, max_iters]);

                    fb.switch_to_block(loop_block);
                    let iter_val = fb.block_params(loop_block)[0];
                    let acc_val = fb.block_params(loop_block)[1];
                    let budget_val = fb.block_params(loop_block)[2];
                    let budget_exhausted = fb.ins().icmp_imm(IntCC::Equal, budget_val, 0);
                    let budget_ok_block = fb.create_block();
                    let budget_exit_block = fb.create_block();
                    fb.ins().brnz(budget_exhausted, budget_exit_block, &[]);
                    fb.ins().jump(budget_ok_block, &[]);

                    fb.switch_to_block(budget_exit_block);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(budget_ok_block);
                    let mut while_locals = locals.clone();
                    while_locals.insert(iter_name.clone(), iter_val);
                    let cond_val = gen_expr(cond_expr, fb, ptr, arg_names, module, &while_locals);
                    let cond_true = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                    fb.ins().brnz(cond_true, body_block, &[]);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(body_block);
                    let ctrl = detect_loop_control(body_expr_raw);
                    let body_expr = match ctrl {
                        LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                        LoopControl::None => body_expr_raw,
                    };

                    let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                        let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(break_true)
                        } else {
                            break_true
                        }
                    } else {
                        break_true
                    };

                    let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                        let continue_cond_val =
                            gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, continue_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(continue_true)
                        } else {
                            continue_true
                        }
                    } else {
                        continue_true
                    };

                    let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &while_locals);
                    let step_val = gen_expr(step_expr, fb, ptr, arg_names, module, &while_locals);
                    let budget_next = fb.ins().iadd_imm(budget_val, -1);
                    let body_bits = fb.ins().bitcast(types::I64, body_val);
                    let is_break_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, BREAK_SENTINEL_BITS as i64);
                    let is_continue_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, CONTINUE_SENTINEL_BITS as i64);
                    let stop_now = fb.ins().bor(break_true, is_break_sentinel);
                    let skip_body = fb.ins().bor(continue_true, is_continue_sentinel);

                    let body_true = fb.ins().fcmp(FloatCC::NotEqual, body_val, zero);
                    if symbol == "any_while" {
                        let effective_true = fb.ins().select(skip_body, false_mask, body_true);
                        let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
                        let next_true = fb.ins().bor(acc_true, effective_true);
                        let stop_any = fb.ins().bor(stop_now, next_true);
                        let any_exit_block = fb.create_block();
                        fb.ins().brnz(stop_any, any_exit_block, &[]);
                        fb.ins().jump(continue_block, &[step_val, zero, budget_next]);

                        fb.switch_to_block(any_exit_block);
                        let exit_val = fb.ins().select(stop_now, acc_val, one);
                        fb.ins().jump(exit_block, &[exit_val]);
                        fb.seal_block(any_exit_block);
                    } else {
                        let effective_true = fb.ins().select(skip_body, true_mask, body_true);
                        let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
                        let next_true = fb.ins().band(acc_true, effective_true);
                        let next_false = fb.ins().bnot(next_true);
                        let stop_all = fb.ins().bor(stop_now, next_false);
                        let all_exit_block = fb.create_block();
                        fb.ins().brnz(stop_all, all_exit_block, &[]);
                        fb.ins().jump(continue_block, &[step_val, one, budget_next]);

                        fb.switch_to_block(all_exit_block);
                        let exit_val = fb.ins().select(stop_now, acc_val, zero);
                        fb.ins().jump(exit_block, &[exit_val]);
                        fb.seal_block(all_exit_block);
                    }

                    fb.switch_to_block(continue_block);
                    let next_iter = fb.block_params(continue_block)[0];
                    let next_acc = fb.block_params(continue_block)[1];
                    let next_budget = fb.block_params(continue_block)[2];
                    fb.ins().jump(loop_block, &[next_iter, next_acc, next_budget]);

                    fb.seal_block(body_block);
                    fb.seal_block(budget_exit_block);
                    fb.seal_block(budget_ok_block);
                    fb.seal_block(continue_block);
                    fb.seal_block(loop_block);
                    fb.switch_to_block(exit_block);
                    fb.seal_block(exit_block);
                    return fb.block_params(exit_block)[0];
                }
            }

            if symbol == "sum_while" && args.len() == 5 {
                if let Expr::Var(iter_name) = &args[0] {
                    let init_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                    let cond_expr = &args[2];
                    let step_expr = &args[3];
                    let body_expr_raw = &args[4];

                    let zero = fb.ins().f64const(0.0);
                    let one = fb.ins().f64const(1.0);
                    let false_mask = fb.ins().fcmp(FloatCC::Equal, zero, one);
                    let acc_init = zero;

                    let loop_block = fb.create_block();
                    let body_block = fb.create_block();
                    let continue_block = fb.create_block();
                    let exit_block = fb.create_block();
                    fb.append_block_param(loop_block, types::F64); // iter
                    fb.append_block_param(loop_block, types::F64); // acc
                    fb.append_block_param(loop_block, types::I64); // budget
                    fb.append_block_param(continue_block, types::F64); // next_iter
                    fb.append_block_param(continue_block, types::F64); // next_acc
                    fb.append_block_param(continue_block, types::I64); // next_budget
                    fb.append_block_param(exit_block, types::F64); // result

                    let max_iters = fb.ins().iconst(types::I64, 1_000_000);
                    fb.ins().jump(loop_block, &[init_val, acc_init, max_iters]);

                    fb.switch_to_block(loop_block);
                    let iter_val = fb.block_params(loop_block)[0];
                    let acc_val = fb.block_params(loop_block)[1];
                    let budget_val = fb.block_params(loop_block)[2];
                    let budget_exhausted = fb.ins().icmp_imm(IntCC::Equal, budget_val, 0);
                    let budget_ok_block = fb.create_block();
                    let budget_exit_block = fb.create_block();
                    fb.ins().brnz(budget_exhausted, budget_exit_block, &[]);
                    fb.ins().jump(budget_ok_block, &[]);

                    fb.switch_to_block(budget_exit_block);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(budget_ok_block);
                    let mut while_locals = locals.clone();
                    while_locals.insert(iter_name.clone(), iter_val);
                    let cond_val = gen_expr(cond_expr, fb, ptr, arg_names, module, &while_locals);
                    let cond_true = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                    fb.ins().brnz(cond_true, body_block, &[]);
                    fb.ins().jump(exit_block, &[acc_val]);

                    fb.switch_to_block(body_block);
                    let ctrl = detect_loop_control(body_expr_raw);
                    let body_expr = match ctrl {
                        LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                        LoopControl::None => body_expr_raw,
                    };

                    let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                        let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(break_true)
                        } else {
                            break_true
                        }
                    } else {
                        break_true
                    };

                    let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                        let continue_cond_val =
                            gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
                        fb.ins().fcmp(FloatCC::NotEqual, continue_cond_val, zero)
                    } else {
                        false_mask
                    };
                    let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                        if invert_cond {
                            fb.ins().bnot(continue_true)
                        } else {
                            continue_true
                        }
                    } else {
                        continue_true
                    };

                    let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &while_locals);
                    let step_val = gen_expr(step_expr, fb, ptr, arg_names, module, &while_locals);
                    let budget_next = fb.ins().iadd_imm(budget_val, -1);
                    let body_bits = fb.ins().bitcast(types::I64, body_val);
                    let is_break_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, BREAK_SENTINEL_BITS as i64);
                    let is_continue_sentinel =
                        fb.ins().icmp_imm(IntCC::Equal, body_bits, CONTINUE_SENTINEL_BITS as i64);
                    let stop_now = fb.ins().bor(break_true, is_break_sentinel);
                    let skip_body = fb.ins().bor(continue_true, is_continue_sentinel);

                    let effective_body = fb.ins().select(skip_body, zero, body_val);
                    let next_acc = fb.ins().fadd(acc_val, effective_body);
                    let sum_break_block = fb.create_block();
                    fb.ins().brnz(stop_now, sum_break_block, &[]);
                    fb.ins().jump(continue_block, &[step_val, next_acc, budget_next]);
                    fb.switch_to_block(sum_break_block);
                    fb.ins().jump(exit_block, &[acc_val]);
                    fb.seal_block(sum_break_block);

                    fb.switch_to_block(continue_block);
                    let next_iter = fb.block_params(continue_block)[0];
                    let next_acc = fb.block_params(continue_block)[1];
                    let next_budget = fb.block_params(continue_block)[2];
                    fb.ins().jump(loop_block, &[next_iter, next_acc, next_budget]);

                    fb.seal_block(body_block);
                    fb.seal_block(budget_exit_block);
                    fb.seal_block(budget_ok_block);
                    fb.seal_block(continue_block);
                    fb.seal_block(loop_block);
                    fb.switch_to_block(exit_block);
                    fb.seal_block(exit_block);
                    return fb.block_params(exit_block)[0];
                }
            }

            if (symbol == "break_on_nan" || symbol == "loop_break_on_nan"
                || symbol == "continue_on_nan" || symbol == "loop_continue_on_nan")
                && args.len() == 1
            {
                let value_val = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let is_nan = fb.ins().fcmp(FloatCC::Unordered, value_val, value_val);
                let sentinel = if symbol == "break_on_nan" || symbol == "loop_break_on_nan" {
                    fb.ins().f64const(f64::from_bits(BREAK_SENTINEL_BITS))
                } else {
                    fb.ins().f64const(f64::from_bits(CONTINUE_SENTINEL_BITS))
                };
                return fb.ins().select(is_nan, sentinel, value_val);
            }

            if symbol == "if_else" && args.len() == 3 {
                let cond_val = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let then_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let else_val = gen_expr(&args[2], fb, ptr, arg_names, module, locals);
                let zero = fb.ins().f64const(0.0);
                let cond_true = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                return fb.ins().select(cond_true, then_val, else_val);
            }

            if let Some(alias) = resolve_symbol_alias(&symbol, args.len()) {
                match alias {
                    SymbolAlias::Identity => {
                        return gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                    }
                    SymbolAlias::Rename(target) => {
                        let mut arg_vals = Vec::with_capacity(args.len());
                        for a in args {
                            arg_vals.push(gen_expr(a, fb, ptr, arg_names, module, locals));
                        }
                        let mut sig = module.make_signature();
                        for _ in 0..arg_vals.len() {
                            sig.params.push(AbiParam::new(types::F64));
                        }
                        sig.returns.push(AbiParam::new(types::F64));
                        let func_id = module
                            .declare_function(target, Linkage::Import, &sig)
                            .expect("failed to declare external function");
                        let local = module.declare_func_in_func(func_id, &mut fb.func);
                        let call = fb.ins().call(local, &arg_vals);
                        return fb.inst_results(call)[0];
                    }
                }
            }

            if symbol == "min" && args.len() == 2 {
                let a = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let b = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let cond = fb.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
                return fb.ins().select(cond, a, b);
            }

            if symbol == "max" && args.len() == 2 {
                let a = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let b = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let cond = fb.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
                return fb.ins().select(cond, a, b);
            }

            if (symbol == "break_if" || symbol == "loop_break_if"
                || symbol == "break_when" || symbol == "loop_break_when"
                || symbol == "break_unless" || symbol == "loop_break_unless"
                || symbol == "continue_if" || symbol == "loop_continue_if"
                || symbol == "continue_when" || symbol == "loop_continue_when"
                || symbol == "continue_unless" || symbol == "loop_continue_unless")
                && args.len() == 2
            {
                let cond_val = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
                let value_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
                let zero = fb.ins().f64const(0.0);
                let cond_true = if symbol == "break_unless" || symbol == "loop_break_unless"
                    || symbol == "continue_unless" || symbol == "loop_continue_unless"
                {
                    fb.ins().fcmp(FloatCC::Equal, cond_val, zero)
                } else {
                    fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero)
                };
                let sentinel = if symbol == "break_if" || symbol == "loop_break_if"
                    || symbol == "break_when" || symbol == "loop_break_when"
                    || symbol == "break_unless" || symbol == "loop_break_unless"
                {
                    fb.ins().f64const(f64::from_bits(BREAK_SENTINEL_BITS))
                } else {
                    fb.ins().f64const(f64::from_bits(CONTINUE_SENTINEL_BITS))
                };
                return fb.ins().select(cond_true, sentinel, value_val);
            }

            let mut arg_vals = Vec::with_capacity(args.len());
            for a in args {
                arg_vals.push(gen_expr(a, fb, ptr, arg_names, module, locals));
            }
            let mut symbol = symbol;
            if symbol == "abs" {
                symbol = "fabs".to_string();
            }
            let mut sig = module.make_signature();
            for _ in 0..arg_vals.len() {
                sig.params.push(AbiParam::new(types::F64));
            }
            sig.returns.push(AbiParam::new(types::F64));
            let func_id = module
                .declare_function(&symbol, Linkage::Import, &sig)
                .expect("failed to declare external function");
            let local = module.declare_func_in_func(func_id, &mut fb.func);
            let call = fb.ins().call(local, &arg_vals);
            fb.inst_results(call)[0]
        }
        Expr::Ternary(cond, then_expr, else_expr) => {
            let cond_val = gen_expr(cond, fb, ptr, arg_names, module, locals);
            let zero = fb.ins().f64const(0.0);
            let cond_bool = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
            let then_val = gen_expr(then_expr, fb, ptr, arg_names, module, locals);
            let else_val = gen_expr(else_expr, fb, ptr, arg_names, module, locals);
            fb.ins().select(cond_bool, then_val, else_val)
        }
        Expr::SumOver { .. } => {
            panic!("SumOver should have been transformed before codegen");
        }
        Expr::AnyOver { .. } => {
            panic!("AnyOver should have been transformed before codegen");
        }
        Expr::AllOver { .. } => {
            panic!("AllOver should have been transformed before codegen");
        }
        Expr::AnyFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => {
            let start_val = gen_expr(start, fb, ptr, arg_names, module, locals);
            let end_val = gen_expr(end, fb, ptr, arg_names, module, locals);
            let step_val = if let Some(st) = step {
                gen_expr(st, fb, ptr, arg_names, module, locals)
            } else {
                fb.ins().f64const(1.0)
            };
            let zero = fb.ins().f64const(0.0);
            let one = fb.ins().f64const(1.0);

            let loop_block = fb.create_block();
            let body_block = fb.create_block();
            let short_true_block = fb.create_block();
            let continue_block = fb.create_block();
            let exit_block = fb.create_block();
            fb.append_block_param(loop_block, types::F64); // i
            fb.append_block_param(loop_block, types::F64); // acc (0/1)
            fb.append_block_param(continue_block, types::F64); // next acc (0/1)
            fb.append_block_param(exit_block, types::F64); // result
            fb.ins().jump(loop_block, &[start_val, zero]);

            fb.switch_to_block(loop_block);
            let i_val = fb.block_params(loop_block)[0];
            let acc_val = fb.block_params(loop_block)[1];
            let step_pos = fb.ins().fcmp(FloatCC::GreaterThan, step_val, zero);
            let step_neg = fb.ins().fcmp(FloatCC::LessThan, step_val, zero);
            let cond_lt = fb.ins().fcmp(FloatCC::LessThan, i_val, end_val);
            let cond_gt = fb.ins().fcmp(FloatCC::GreaterThan, i_val, end_val);
            let run_pos = fb.ins().band(step_pos, cond_lt);
            let run_neg = fb.ins().band(step_neg, cond_gt);
            let cond = fb.ins().bor(run_pos, run_neg);
            fb.ins().brnz(cond, body_block, &[]);
            fb.ins().jump(exit_block, &[acc_val]);

            fb.switch_to_block(body_block);
            let mut body_locals = locals.clone();
            body_locals.insert(iter_var.clone(), i_val);
            let ctrl = detect_loop_control(body);
            let body_expr = match ctrl {
                LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                LoopControl::None => body,
            };
            let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(break_true)
                } else {
                    break_true
                }
            } else {
                break_true
            };
            let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                let cont_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, cont_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(continue_true)
                } else {
                    continue_true
                }
            } else {
                continue_true
            };

            let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &body_locals);
            let body_true = fb.ins().fcmp(FloatCC::NotEqual, body_val, zero);
            let effective_true = if let Some(pred_expr) = pred {
                let pred_val = gen_expr(pred_expr, fb, ptr, arg_names, module, &body_locals);
                let pred_true = fb.ins().fcmp(FloatCC::NotEqual, pred_val, zero);
                fb.ins().band(pred_true, body_true)
            } else {
                body_true
            };
            let false_mask = fb.ins().fcmp(FloatCC::Equal, zero, one);
            let effective_true = fb.ins().select(continue_true, false_mask, effective_true);
            let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
            let next_true = fb.ins().bor(acc_true, effective_true);
            let next_acc_base = fb.ins().select(next_true, one, zero);
            let next_acc = fb.ins().select(break_true, acc_val, next_acc_base);
            fb.ins().brnz(next_true, short_true_block, &[]);
            fb.ins().jump(continue_block, &[next_acc]);

            fb.switch_to_block(continue_block);
            let next_acc = fb.block_params(continue_block)[0];
            let next_i_base = fb.ins().fadd(i_val, step_val);
            let next_i = fb.ins().select(break_true, end_val, next_i_base);
            fb.ins().jump(loop_block, &[next_i, next_acc]);

            fb.switch_to_block(short_true_block);
            fb.ins().jump(exit_block, &[one]);

            fb.seal_block(body_block);
            fb.seal_block(short_true_block);
            fb.seal_block(continue_block);
            fb.seal_block(loop_block);
            fb.switch_to_block(exit_block);
            fb.seal_block(exit_block);
            fb.block_params(exit_block)[0]
        }
        Expr::AllFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => {
            let start_val = gen_expr(start, fb, ptr, arg_names, module, locals);
            let end_val = gen_expr(end, fb, ptr, arg_names, module, locals);
            let step_val = if let Some(st) = step {
                gen_expr(st, fb, ptr, arg_names, module, locals)
            } else {
                fb.ins().f64const(1.0)
            };
            let zero = fb.ins().f64const(0.0);
            let one = fb.ins().f64const(1.0);
            let true_mask = fb.ins().fcmp(FloatCC::Equal, zero, zero);

            let loop_block = fb.create_block();
            let body_block = fb.create_block();
            let short_false_block = fb.create_block();
            let continue_block = fb.create_block();
            let exit_block = fb.create_block();
            fb.append_block_param(loop_block, types::F64); // i
            fb.append_block_param(loop_block, types::F64); // acc (0/1)
            fb.append_block_param(continue_block, types::F64); // next acc (0/1)
            fb.append_block_param(exit_block, types::F64); // result
            fb.ins().jump(loop_block, &[start_val, one]);

            fb.switch_to_block(loop_block);
            let i_val = fb.block_params(loop_block)[0];
            let acc_val = fb.block_params(loop_block)[1];
            let step_pos = fb.ins().fcmp(FloatCC::GreaterThan, step_val, zero);
            let step_neg = fb.ins().fcmp(FloatCC::LessThan, step_val, zero);
            let cond_lt = fb.ins().fcmp(FloatCC::LessThan, i_val, end_val);
            let cond_gt = fb.ins().fcmp(FloatCC::GreaterThan, i_val, end_val);
            let run_pos = fb.ins().band(step_pos, cond_lt);
            let run_neg = fb.ins().band(step_neg, cond_gt);
            let cond = fb.ins().bor(run_pos, run_neg);
            fb.ins().brnz(cond, body_block, &[]);
            fb.ins().jump(exit_block, &[acc_val]);

            fb.switch_to_block(body_block);
            let mut body_locals = locals.clone();
            body_locals.insert(iter_var.clone(), i_val);
            let ctrl = detect_loop_control(body);
            let body_expr = match ctrl {
                LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                LoopControl::None => body,
            };
            let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(break_true)
                } else {
                    break_true
                }
            } else {
                break_true
            };
            let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                let cont_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, cont_cond_val, zero)
            } else {
                fb.ins().fcmp(FloatCC::Equal, zero, one)
            };
            let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(continue_true)
                } else {
                    continue_true
                }
            } else {
                continue_true
            };

            let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &body_locals);
            let body_true = fb.ins().fcmp(FloatCC::NotEqual, body_val, zero);
            let effective_true = if let Some(pred_expr) = pred {
                let pred_val = gen_expr(pred_expr, fb, ptr, arg_names, module, &body_locals);
                let pred_true = fb.ins().fcmp(FloatCC::NotEqual, pred_val, zero);
                fb.ins().select(pred_true, body_true, true_mask)
            } else {
                body_true
            };
            let effective_true = fb.ins().select(continue_true, true_mask, effective_true);
            let acc_true = fb.ins().fcmp(FloatCC::NotEqual, acc_val, zero);
            let next_true = fb.ins().band(acc_true, effective_true);
            let next_acc_base = fb.ins().select(next_true, one, zero);
            let next_acc = fb.ins().select(break_true, acc_val, next_acc_base);
            fb.ins().brz(next_true, short_false_block, &[]);
            fb.ins().jump(continue_block, &[next_acc]);

            fb.switch_to_block(continue_block);
            let next_acc = fb.block_params(continue_block)[0];
            let next_i_base = fb.ins().fadd(i_val, step_val);
            let next_i = fb.ins().select(break_true, end_val, next_i_base);
            fb.ins().jump(loop_block, &[next_i, next_acc]);

            fb.switch_to_block(short_false_block);
            fb.ins().jump(exit_block, &[zero]);

            fb.seal_block(body_block);
            fb.seal_block(short_false_block);
            fb.seal_block(continue_block);
            fb.seal_block(loop_block);
            fb.switch_to_block(exit_block);
            fb.seal_block(exit_block);
            fb.block_params(exit_block)[0]
        }
        Expr::SumFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => {
            let start_val = gen_expr(start, fb, ptr, arg_names, module, locals);
            let end_val = gen_expr(end, fb, ptr, arg_names, module, locals);
            let step_val = if let Some(st) = step {
                gen_expr(st, fb, ptr, arg_names, module, locals)
            } else {
                fb.ins().f64const(1.0)
            };
            let zero_acc = fb.ins().f64const(0.0);
            let loop_block = fb.create_block();
            let body_block = fb.create_block();
            let exit_block = fb.create_block();
            fb.append_block_param(loop_block, types::F64); // i
            fb.append_block_param(loop_block, types::F64); // acc
            fb.append_block_param(exit_block, types::F64); // result
            fb.ins().jump(loop_block, &[start_val, zero_acc]);
            fb.switch_to_block(loop_block);
            let i_val = fb.block_params(loop_block)[0];
            let acc_val = fb.block_params(loop_block)[1];
            // runtime-aware step direction:
            // step > 0 => i < end
            // step < 0 => i > end
            // step == 0 => do not enter loop
            let zero = fb.ins().f64const(0.0);
            let step_pos = fb.ins().fcmp(FloatCC::GreaterThan, step_val, zero);
            let step_neg = fb.ins().fcmp(FloatCC::LessThan, step_val, zero);
            let cond_lt = fb.ins().fcmp(FloatCC::LessThan, i_val, end_val);
            let cond_gt = fb.ins().fcmp(FloatCC::GreaterThan, i_val, end_val);
            let run_pos = fb.ins().band(step_pos, cond_lt);
            let run_neg = fb.ins().band(step_neg, cond_gt);
            let cond = fb.ins().bor(run_pos, run_neg);
            fb.ins().brnz(cond, body_block, &[]);
            fb.ins().jump(exit_block, &[acc_val]);
            fb.switch_to_block(body_block);
            let mut body_locals = locals.clone();
            body_locals.insert(iter_var.clone(), i_val);
            let ctrl = detect_loop_control(body);
            let body_expr = match ctrl {
                LoopControl::Break { value, .. } | LoopControl::Continue { value, .. } => value,
                LoopControl::None => body,
            };
            let break_true = if let LoopControl::Break { cond, .. } = ctrl {
                let break_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, break_cond_val, zero)
            } else {
                let one_const = fb.ins().f64const(1.0);
                fb.ins().fcmp(FloatCC::Equal, zero, one_const)
            };
            let break_true = if let LoopControl::Break { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(break_true)
                } else {
                    break_true
                }
            } else {
                break_true
            };
            let continue_true = if let LoopControl::Continue { cond, .. } = ctrl {
                let cont_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &body_locals);
                fb.ins().fcmp(FloatCC::NotEqual, cont_cond_val, zero)
            } else {
                let one_const = fb.ins().f64const(1.0);
                fb.ins().fcmp(FloatCC::Equal, zero, one_const)
            };
            let continue_true = if let LoopControl::Continue { invert_cond, .. } = ctrl {
                if invert_cond {
                    fb.ins().bnot(continue_true)
                } else {
                    continue_true
                }
            } else {
                continue_true
            };

            let body_val = gen_expr(body_expr, fb, ptr, arg_names, module, &body_locals);
            let body_val = if let Some(pred_expr) = pred {
                let cond_val = gen_expr(pred_expr, fb, ptr, arg_names, module, &body_locals);
                let zero = fb.ins().f64const(0.0);
                let mask = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
                fb.ins().select(mask, body_val, zero)
            } else {
                body_val
            };
            let body_val = fb.ins().select(continue_true, zero, body_val);
            let next_acc_base = fb.ins().fadd(acc_val, body_val);
            let next_acc = fb.ins().select(break_true, acc_val, next_acc_base);
            let next_i_base = fb.ins().fadd(i_val, step_val);
            let next_i = fb.ins().select(break_true, end_val, next_i_base);
            fb.ins().jump(loop_block, &[next_i, next_acc]);
            fb.seal_block(body_block);
            fb.seal_block(loop_block);
            fb.switch_to_block(exit_block);
            fb.seal_block(exit_block);
            fb.block_params(exit_block)[0]
        }
    }
}

// helper for zero-copy buffer access used by the JIT runner
struct BufferView {
    view: pyo3::ffi::Py_buffer,
    elem_type: BufferElemType,
    len: usize,
}

impl BufferView {
    #[inline(always)]
    fn as_ptr_u8(&self) -> *const u8 {
        self.view.buf as *const u8
    }

    #[inline(always)]
    fn as_ptr_f64(&self) -> *const f64 {
        self.view.buf as *const f64
    }

    #[inline(always)]
    fn is_aligned_for_f64(&self) -> bool {
        (self.view.buf as usize) % std::mem::align_of::<f64>() == 0
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        unsafe { pyo3::ffi::PyBuffer_Release(&mut self.view) };
    }
}

#[cfg(feature = "pyo3")]
unsafe fn parse_buffer_elem_type(view: &pyo3::ffi::Py_buffer) -> Option<BufferElemType> {
    if view.itemsize <= 0 {
        return None;
    }
    let itemsize = view.itemsize as usize;

    fn expected_size_for_code(code: char) -> Option<usize> {
        match code {
            'd' => Some(std::mem::size_of::<f64>()),
            'f' => Some(std::mem::size_of::<f32>()),
            'q' => Some(std::mem::size_of::<i64>()),
            'i' => Some(std::mem::size_of::<i32>()),
            'h' => Some(std::mem::size_of::<i16>()),
            'b' => Some(std::mem::size_of::<i8>()),
            'Q' => Some(std::mem::size_of::<u64>()),
            'I' => Some(std::mem::size_of::<u32>()),
            'H' => Some(std::mem::size_of::<u16>()),
            'B' => Some(std::mem::size_of::<u8>()),
            '?' => Some(std::mem::size_of::<u8>()),
            _ => None,
        }
    }

    fn to_elem_type(code: char, itemsize: usize) -> Option<BufferElemType> {
        if code == 'l' {
            return match itemsize {
                8 => Some(BufferElemType::I64),
                4 => Some(BufferElemType::I32),
                _ => None,
            };
        }
        if code == 'L' {
            return match itemsize {
                8 => Some(BufferElemType::U64),
                4 => Some(BufferElemType::U32),
                _ => None,
            };
        }
        if let Some(expected) = expected_size_for_code(code) {
            if expected != itemsize {
                return None;
            }
        }
        match code {
            'd' => Some(BufferElemType::F64),
            'f' => Some(BufferElemType::F32),
            'q' => Some(BufferElemType::I64),
            'i' => Some(BufferElemType::I32),
            'h' => Some(BufferElemType::I16),
            'b' => Some(BufferElemType::I8),
            'Q' => Some(BufferElemType::U64),
            'I' => Some(BufferElemType::U32),
            'H' => Some(BufferElemType::U16),
            'B' => Some(BufferElemType::U8),
            '?' => Some(BufferElemType::Bool),
            _ => None,
        }
    }

    if view.format.is_null() {
        return match itemsize {
            8 => Some(BufferElemType::F64),
            4 => Some(BufferElemType::F32),
            2 => Some(BufferElemType::I16),
            1 => Some(BufferElemType::U8),
            _ => None,
        };
    }

    let fmt = CStr::from_ptr(view.format).to_str().ok()?;
    let code = fmt
        .chars()
        .rev()
        .find(|ch| ch.is_ascii_alphabetic() || *ch == '?')?;
    to_elem_type(code, itemsize)
}

#[cfg(feature = "pyo3")]
unsafe fn open_typed_buffer(obj: &pyo3::PyAny) -> Option<BufferView> {
    let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
    let flags = pyo3::ffi::PyBUF_C_CONTIGUOUS | pyo3::ffi::PyBUF_FORMAT;
    if pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, flags) != 0 {
        pyo3::ffi::PyErr_Clear();
        return None;
    }

    let itemsize = view.itemsize as usize;
    if itemsize == 0 {
        pyo3::ffi::PyBuffer_Release(&mut view);
        return None;
    }

    let elem_type = match parse_buffer_elem_type(&view) {
        Some(elem) => elem,
        None => {
            pyo3::ffi::PyBuffer_Release(&mut view);
            return None;
        }
    };

    let total_bytes = view.len as usize;
    if total_bytes % itemsize != 0 {
        pyo3::ffi::PyBuffer_Release(&mut view);
        return None;
    }

    let len = total_bytes / itemsize;
    Some(BufferView {
        view,
        elem_type,
        len,
    })
}

#[cfg(feature = "pyo3")]
#[inline(always)]
unsafe fn read_buffer_f64(view: &BufferView, index: usize) -> f64 {
    let base = view.as_ptr_u8();
    match view.elem_type {
        BufferElemType::F64 => {
            let p = base.add(index * std::mem::size_of::<f64>()) as *const f64;
            std::ptr::read_unaligned(p)
        }
        BufferElemType::F32 => {
            let p = base.add(index * std::mem::size_of::<f32>()) as *const f32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I64 => {
            let p = base.add(index * std::mem::size_of::<i64>()) as *const i64;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I32 => {
            let p = base.add(index * std::mem::size_of::<i32>()) as *const i32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I16 => {
            let p = base.add(index * std::mem::size_of::<i16>()) as *const i16;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I8 => {
            let p = base.add(index * std::mem::size_of::<i8>()) as *const i8;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U64 => {
            let p = base.add(index * std::mem::size_of::<u64>()) as *const u64;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U32 => {
            let p = base.add(index * std::mem::size_of::<u32>()) as *const u32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U16 => {
            let p = base.add(index * std::mem::size_of::<u16>()) as *const u16;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U8 => {
            let p = base.add(index * std::mem::size_of::<u8>()) as *const u8;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::Bool => {
            let p = base.add(index) as *const u8;
            if std::ptr::read_unaligned(p) == 0 {
                0.0
            } else {
                1.0
            }
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

/// Highly optimized helper to execute a JIT compiled function.
/// Handles zero-copy buffers (including multi-argument vectorization) and scalar argument unpacking via stack.
#[cfg(feature = "pyo3")]
#[inline(always)]
pub fn execute_jit_func(py: pyo3::Python, entry: &JitEntry, args: &pyo3::types::PyTuple) -> pyo3::PyResult<pyo3::PyObject> {
    let arg_count = args.len();
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

    // Speculative fast path using cached type profile.
    if let Some(profile) = lookup_exec_profile(entry.func_ptr) {
        match profile {
            JitExecProfile::PackedBuffer { arg_count: expected, elem } => {
                if arg_count == 1 && entry.arg_count == expected {
                    if let Ok(item) = args.get_item(0) {
                        if let Some(view) = unsafe { open_typed_buffer(item) } {
                            if view.elem_type == elem && view.len == expected {
                                if elem == BufferElemType::F64 {
                                    if view.is_aligned_for_f64() {
                                        let res = f(view.as_ptr_f64());
                                        return Ok(res.into_py(py));
                                    }
                                }
                                let mut converted = Vec::with_capacity(expected);
                                for i in 0..expected {
                                    converted.push(unsafe { read_buffer_f64(&view, i) });
                                }
                                let res = f(converted.as_ptr());
                                return Ok(res.into_py(py));
                            }
                        }
                    }
                }
            }
            JitExecProfile::VectorizedBuffers { arg_count: expected, elem_types } => {
                if arg_count == expected && expected == elem_types.len() {
                    let mut views = Vec::with_capacity(expected);
                    let mut common_len: Option<usize> = None;
                    let mut matched = true;
                    for i in 0..expected {
                        let Ok(item) = args.get_item(i) else {
                            matched = false;
                            break;
                        };
                        let Some(view) = (unsafe { open_typed_buffer(item) }) else {
                            matched = false;
                            break;
                        };
                        if view.elem_type != elem_types[i] {
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
                        if entry.reduction != ReductionMode::None {
                            let mut acc = reduction_identity(entry.reduction);
                            const MAX_FAST_ARGS: usize = 8;
                            if expected <= MAX_FAST_ARGS {
                                for i in 0..len {
                                    let mut iter_args = [0.0_f64; MAX_FAST_ARGS];
                                    for j in 0..expected {
                                        iter_args[j] = unsafe { read_buffer_f64(&views[j], i) };
                                    }
                                    let val = f(iter_args.as_ptr());
                                    if reduction_step(entry.reduction, &mut acc, val) {
                                        break;
                                    }
                                }
                            } else {
                                for i in 0..len {
                                    let mut iter_args = Vec::with_capacity(expected);
                                    for j in 0..expected {
                                        iter_args.push(unsafe { read_buffer_f64(&views[j], i) });
                                    }
                                    let val = f(iter_args.as_ptr());
                                    if reduction_step(entry.reduction, &mut acc, val) {
                                        break;
                                    }
                                }
                            }
                            return Ok(acc.into_py(py));
                        }

                        let mut results = Vec::with_capacity(len);
                        const MAX_FAST_ARGS: usize = 8;
                        if expected <= MAX_FAST_ARGS {
                            for i in 0..len {
                                let mut iter_args = [0.0_f64; MAX_FAST_ARGS];
                                for j in 0..expected {
                                    iter_args[j] = unsafe { read_buffer_f64(&views[j], i) };
                                }
                                results.push(f(iter_args.as_ptr()));
                            }
                        } else {
                            for i in 0..len {
                                let mut iter_args = Vec::with_capacity(expected);
                                for j in 0..expected {
                                    iter_args.push(unsafe { read_buffer_f64(&views[j], i) });
                                }
                                results.push(f(iter_args.as_ptr()));
                            }
                        }
                        let byte_slice = unsafe {
                            std::slice::from_raw_parts(
                                results.as_ptr() as *const u8,
                                results.len() * std::mem::size_of::<f64>(),
                            )
                        };
                        let py_bytes = pyo3::types::PyBytes::new(py, byte_slice);
                        let array_mod = py.import("array")?;
                        let array_obj = array_mod.getattr("array")?.call1(("d", py_bytes))?;
                        return Ok(array_obj.into_py(py));
                    }
                }
            }
            JitExecProfile::ScalarArgs => {
                if arg_count == entry.arg_count {
                    const MAX_FAST_ARGS: usize = 8;
                    if arg_count <= MAX_FAST_ARGS {
                        let mut stack_args = [0.0_f64; MAX_FAST_ARGS];
                        for i in 0..arg_count {
                            let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
                            let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
                            if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
                                return Err(pyo3::PyErr::fetch(py));
                            }
                            stack_args[i] = val;
                        }
                        let res = f(stack_args.as_ptr());
                        return Ok(res.into_py(py));
                    }
                }
            }
        }
    }

    // 1. Single buffer acting as the entire argument array for a multi-argument function
    if arg_count == 1 && entry.arg_count > 1 {
        if let Ok(item) = args.get_item(0) {
            if let Some(view) = unsafe { open_typed_buffer(item) } {
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
                    return Ok(res.into_py(py));
                }
            }
        }
    }

    // 2. Vectorization: 1 or more arguments, all of which must be buffers of the same length
    if arg_count == entry.arg_count && arg_count > 0 {
        let mut views = Vec::with_capacity(arg_count);
        let mut common_len = None;
        let mut all_buffers = true;

        for i in 0..arg_count {
            if let Ok(item) = args.get_item(i) {
                if let Some(view) = unsafe { open_typed_buffer(item) } {
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
                if entry.reduction != ReductionMode::None {
                    let mut acc = reduction_identity(entry.reduction);
                    const MAX_FAST_ARGS: usize = 8;
                    if arg_count <= MAX_FAST_ARGS {
                        for i in 0..len {
                            let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                            for j in 0..arg_count {
                                iter_args[j] = unsafe { read_buffer_f64(&views[j], i) };
                            }
                            let val = f(iter_args.as_ptr());
                            if reduction_step(entry.reduction, &mut acc, val) {
                                break;
                            }
                        }
                    } else {
                        for i in 0..len {
                            let mut iter_args = Vec::with_capacity(arg_count);
                            for j in 0..arg_count {
                                iter_args.push(unsafe { read_buffer_f64(&views[j], i) });
                            }
                            let val = f(iter_args.as_ptr());
                            if reduction_step(entry.reduction, &mut acc, val) {
                                break;
                            }
                        }
                    }

                    store_exec_profile(
                        entry.func_ptr,
                        JitExecProfile::VectorizedBuffers {
                            arg_count,
                            elem_types: views.iter().map(|v| v.elem_type).collect::<Vec<_>>(),
                        },
                    );
                    return Ok(acc.into_py(py));
                }

                let mut results = Vec::with_capacity(len);
                let elem_types = views.iter().map(|v| v.elem_type).collect::<Vec<_>>();
                let all_f64 = elem_types.iter().all(|k| *k == BufferElemType::F64);

                const MAX_FAST_ARGS: usize = 8;
                if arg_count <= MAX_FAST_ARGS {
                    for i in 0..len {
                        let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                        for j in 0..arg_count {
                            iter_args[j] = if all_f64 {
                                unsafe { read_buffer_f64(&views[j], i) }
                            } else {
                                unsafe { read_buffer_f64(&views[j], i) }
                            };
                        }
                        results.push(f(iter_args.as_ptr()));
                    }
                } else {
                    for i in 0..len {
                        let mut iter_args = Vec::with_capacity(arg_count);
                        for j in 0..arg_count {
                            let val = if all_f64 {
                                unsafe { read_buffer_f64(&views[j], i) }
                            } else {
                                unsafe { read_buffer_f64(&views[j], i) }
                            };
                            iter_args.push(val);
                        }
                        results.push(f(iter_args.as_ptr()));
                    }
                }

                store_exec_profile(
                    entry.func_ptr,
                    JitExecProfile::VectorizedBuffers {
                        arg_count,
                        elem_types,
                    },
                );

                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        results.as_ptr() as *const u8,
                        results.len() * std::mem::size_of::<f64>(),
                    )
                };
                let py_bytes = pyo3::types::PyBytes::new(py, byte_slice);
                let array_mod = py.import("array")?;
                let array_obj = array_mod.getattr("array")?.call1(("d", py_bytes))?;
                return Ok(array_obj.into_py(py));
            }
        }
    }

    // 2.5 Sequence fallback for reduction-style single-arg kernels
    if arg_count == 1 && entry.arg_count == 1 && entry.reduction != ReductionMode::None {
        if let Ok(item) = args.get_item(0) {
            if let Ok(iter) = item.iter() {
                let mut acc = reduction_identity(entry.reduction);
                let mut buf = [0.0_f64; 1];
                for obj_res in iter {
                    let obj = obj_res?;
                    let val: f64 = obj.extract()?;
                    buf[0] = val;
                    let out = f(buf.as_ptr());
                    if reduction_step(entry.reduction, &mut acc, out) {
                        break;
                    }
                }
                return Ok(acc.into_py(py));
            }
        }
    }

    if arg_count != entry.arg_count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "wrong argument count for JIT function",
        ));
    }

    // 3. Fast path for small number of standard Python scalar arguments
    const MAX_FAST_ARGS: usize = 8;
    if arg_count <= MAX_FAST_ARGS {
        let mut stack_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
        for i in 0..arg_count {
            let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
            let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
            if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
                return Err(pyo3::PyErr::fetch(py));
            }
            stack_args[i] = val;
        }
        store_exec_profile(entry.func_ptr, JitExecProfile::ScalarArgs);
        let res = f(stack_args.as_ptr());
        return Ok(res.into_py(py));
    }

    // 4. Fallback for > 8 scalar args: heap allocation
    let mut heap_args = Vec::with_capacity(arg_count);
    for i in 0..arg_count {
        let item = unsafe { pyo3::ffi::PyTuple_GET_ITEM(args.as_ptr(), i as isize) };
        let val = unsafe { pyo3::ffi::PyFloat_AsDouble(item) };
        if val == -1.0 && !unsafe { pyo3::ffi::PyErr_Occurred() }.is_null() {
            return Err(pyo3::PyErr::fetch(py));
        }
        heap_args.push(val);
    }
    store_exec_profile(entry.func_ptr, JitExecProfile::ScalarArgs);
    let res = f(heap_args.as_ptr());
    Ok(res.into_py(py))
}
