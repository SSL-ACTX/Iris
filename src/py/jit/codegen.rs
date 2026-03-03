// src/py/jit/codegen.rs
//! Core JIT compilation logic, including registry and Cranelift codegen.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::py::jit::parser::Expr;
use crate::py::jit::heuristics;

// cranelift imports
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use pyo3::AsPyPointer;
use pyo3::IntoPy;

/// A compiled function entry returned by the JIT.
#[derive(Clone)]
pub struct JitEntry {
    pub func_ptr: usize,
    pub arg_count: usize,
}

static JIT_FUNC_COUNTER: once_cell::sync::Lazy<AtomicUsize> =
    once_cell::sync::Lazy::new(|| AtomicUsize::new(0));

static JIT_REGISTRY: once_cell::sync::OnceCell<std::sync::Mutex<HashMap<usize, JitEntry>>> =
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

thread_local! {
    static TLS_JIT_MODULE: std::cell::RefCell<Option<JITModule>> =
        std::cell::RefCell::new(None);
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

/// Compile an expression string into a native function entry.
pub fn compile_jit(expr_str: &str, arg_names: &[String]) -> Option<JitEntry> {
    // tokenize and parse
    let tokens = crate::py::jit::parser::tokenize(expr_str);
    let mut parser = crate::py::jit::parser::Parser::new(tokens);
    let mut expr = parser.parse_expr()?;
    eprintln!("[Iris][jit] parsed AST for '{}': {:?}", expr_str, expr);
    // detect generator-style loop over a container and convert to body-only
    // expression.  Python wrapper will pass the container buffer and the
    // JIT runtime will vectorize across it; the compiled function gets a
    // single scalar argument representing each element.
    let mut adjusted_args = arg_names.to_vec();
    // use a cloned copy when destructuring to release borrow immediately
    if let Expr::SumOver { iter_var, container, body } = expr.clone() {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                eprintln!("[Iris][jit] converting SumOver '{}' in {}", iter_var, cont_name);
                expr = (*body.clone());
                adjusted_args = vec![iter_var.clone()];
            }
        }
    }
    expr = heuristics::optimize(expr);
    eprintln!("[Iris][jit] optimized AST: {:?}", expr);
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
            eprintln!("[Iris][jit] define_function failed: {:?}", err);
            return None;
        }
        module.clear_context(&mut ctx);
        module.finalize_definitions();

        let code_ptr = module.get_finalized_function(id) as usize;
        Some(JitEntry {
            func_ptr: code_ptr,
            arg_count,
        })
    })
}

fn gen_expr(
    expr: &Expr,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
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
                '+' => v,
                _ => v,
            }
        }
        Expr::Call(name, args) => {
            let mut arg_vals = Vec::with_capacity(args.len());
            for a in args {
                arg_vals.push(gen_expr(a, fb, ptr, arg_names, module, locals));
            }
            let mut symbol = name.rsplit('.').next().unwrap().to_string();
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
        Expr::SumFor {
            iter_var,
            start,
            end,
            body,
        } => {
            let start_val = gen_expr(start, fb, ptr, arg_names, module, locals);
            let end_val = gen_expr(end, fb, ptr, arg_names, module, locals);
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
            let cond = fb.ins().fcmp(FloatCC::LessThan, i_val, end_val);
            fb.ins().brnz(cond, body_block, &[]);
            fb.ins().jump(exit_block, &[acc_val]);
            fb.switch_to_block(body_block);
            let mut body_locals = locals.clone();
            body_locals.insert(iter_var.clone(), i_val);
            let body_val = gen_expr(body, fb, ptr, arg_names, module, &body_locals);
            let next_acc = fb.ins().fadd(acc_val, body_val);
            let one = fb.ins().f64const(1.0);
            let next_i = fb.ins().fadd(i_val, one);
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
/// If the given Python object exposes a contiguous buffer of `f64`, return a
/// pointer/length pair without copying. The buffer view is released before
/// the function returns.
#[cfg(feature = "pyo3")]
unsafe fn buffer_ptr_len(obj: &pyo3::PyAny) -> Option<(*const f64, usize)> {
    let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
    if pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, pyo3::ffi::PyBUF_SIMPLE) != 0 {
        pyo3::ffi::PyErr_Clear();
        return None;
    }
    let itemsize = view.itemsize as usize;
    let total_bytes = view.len as usize;
    if itemsize != std::mem::size_of::<f64>() {
        pyo3::ffi::PyBuffer_Release(&mut view);
        return None;
    }
    let len = total_bytes / itemsize;
    let ptr = view.buf as *const f64;
    pyo3::ffi::PyBuffer_Release(&mut view);
    Some((ptr, len))
}

/// Highly optimized helper to execute a JIT compiled function.
/// Handles zero-copy buffers (including multi-argument vectorization) and scalar argument unpacking via stack.
#[cfg(feature = "pyo3")]
#[inline(always)]
pub fn execute_jit_func(py: pyo3::Python, entry: &JitEntry, args: &pyo3::types::PyTuple) -> pyo3::PyResult<pyo3::PyObject> {
    let arg_count = args.len();

    // 1. Single buffer acting as the entire argument array for a multi-argument function
    if arg_count == 1 && entry.arg_count > 1 {
        if let Ok(item) = args.get_item(0) {
            if let Some((ptr, len)) = unsafe { buffer_ptr_len(item) } {
                if len == entry.arg_count {
                    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
                    let res = f(ptr);
                    return Ok(res.into_py(py));
                }
            }
        }
    }

    // 2. Vectorization: 1 or more arguments, all of which must be buffers of the same length
    if arg_count == entry.arg_count && arg_count > 0 {
        let mut ptrs = Vec::with_capacity(arg_count);
        let mut common_len = None;
        let mut all_buffers = true;

        for i in 0..arg_count {
            if let Ok(item) = args.get_item(i) {
                if let Some((ptr, len)) = unsafe { buffer_ptr_len(item) } {
                    if let Some(c_len) = common_len {
                        if len != c_len {
                            all_buffers = false;
                            break;
                        }
                    } else {
                        common_len = Some(len);
                    }
                    ptrs.push(ptr);
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
                let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
                let mut results = Vec::with_capacity(len);

                const MAX_FAST_ARGS: usize = 8;
                if arg_count <= MAX_FAST_ARGS {
                    for i in 0..len {
                        let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                        for j in 0..arg_count {
                            iter_args[j] = unsafe { *ptrs[j].add(i) };
                        }
                        results.push(f(iter_args.as_ptr()));
                    }
                } else {
                    for i in 0..len {
                        let mut iter_args = Vec::with_capacity(arg_count);
                        for j in 0..arg_count {
                            iter_args.push(unsafe { *ptrs[j].add(i) });
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
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
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
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let res = f(heap_args.as_ptr());
    Ok(res.into_py(py))
}
