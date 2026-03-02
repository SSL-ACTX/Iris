// src/py/jit/mod.rs
//! Python JIT/offload support for the Iris runtime.
//!
//! This module provides the low-level bindings that power the `@iris.offload`
//! decorator in Python.  Initially the implementation is a no-op stub, but it
//! gives us a dedicated home for future JIT or actor‑routing logic.

#![allow(non_local_definitions)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::{PyDict, PyTuple, PyBytes};

// cranelift imports
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use cranelift_native;
use pyo3::AsPyPointer;

// Offload actor pool ---------------------------------------------------------

/// A task describing a Python call to execute.
struct OffloadTask {
    func: Py<PyAny>,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    resp: std::sync::mpsc::Sender<Result<PyObject, PyErr>>,
}

struct OffloadPool {
    sender: crossbeam_channel::Sender<OffloadTask>,
}

impl OffloadPool {
    fn new(size: usize) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<OffloadTask>();

        for _ in 0..size {
            let rx = rx.clone();
            std::thread::spawn(move || {
                loop {
                    match rx.recv() {
                        Ok(task) => {
                            if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
                                break;
                            }
                            Python::with_gil(|py| {
                                let func = task.func.as_ref(py);
                                let args = task.args.as_ref(py);
                                let kwargs = task.kwargs.as_ref().map(|k: &Py<PyDict>| k.as_ref(py));
                                let result = func.call(args, kwargs).map(|obj| obj.into_py(py));
                                let _ = task.resp.send(result);
                            });
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        OffloadPool { sender: tx }
    }
}

// shared singleton
static OFFLOAD_POOL: once_cell::sync::OnceCell<Arc<OffloadPool>> = once_cell::sync::OnceCell::new();

fn get_offload_pool() -> Arc<OffloadPool> {
    OFFLOAD_POOL
        .get_or_init(|| Arc::new(OffloadPool::new(num_cpus::get())))
        .clone()
}

// JIT registry -------------------------------------------------------------

#[derive(Clone)]
struct JitEntry {
    func_ptr: usize,
    arg_count: usize,
}

static JIT_REGISTRY: once_cell::sync::OnceCell<Mutex<HashMap<usize, JitEntry>>> =
    once_cell::sync::OnceCell::new();

// Each thread gets its own JITModule instance.  This sidesteps the
// `Sync` requirements that prevented us from keeping a single global
// module in a `static`, and it still guarantees that the first
// compilation on a given thread will reserve the address space for all
// later functions.  Since the module is stored in thread-local storage
// it will live for the duration of the thread (i.e. the lifetime of the
// Python interpreter thread used during tests).
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

fn register_jit(func_key: usize, entry: JitEntry) {
    let map = JIT_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    guard.insert(func_key, entry);
}

fn lookup_jit(func_key: usize) -> Option<JitEntry> {
    JIT_REGISTRY
        .get()
        .and_then(|map| map.lock().unwrap().get(&func_key).cloned())
}

// simple expression AST for compiler
#[derive(Debug, Clone)]
enum Expr {
    Const(f64),
    Var(String),
    BinOp(Box<Expr>, String, Box<Expr>),
    Call(String, Vec<Expr>),
    UnaryOp(char, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
}

// parser helpers
fn tokenize(expr: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut chars = expr.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            continue;
        }
        // handle two-character operators
        if c == '*' && chars.peek() == Some(&'*') {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            chars.next();
            tokens.push("**".to_string());
            continue;
        }
        if (c == '<' || c == '>' || c == '=' || c == '!') && chars.peek() == Some(&'=') {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            let mut op = c.to_string();
            op.push('=');
            chars.next();
            tokens.push(op);
            continue;
        }
        if "+-*/(),%<>=!".contains(c) {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            tokens.push(c.to_string());
            continue;
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

// Pratt parser implementation with exponent precedence and comparisons
struct Parser {
    tokens: Vec<String>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<String>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn next(&mut self) -> Option<String> {
        if self.pos < self.tokens.len() {
            let s = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(s)
        } else {
            None
        }
    }

    fn parse_expr(&mut self) -> Option<Expr> {
        let mut node = self.parse_relation()?;
        if let Some(tok) = self.peek() {
            if tok == "if" {
                self.next();
                let cond = self.parse_expr()?;
                if self.next()? != "else" {
                    return None;
                }
                let alt = self.parse_expr()?;
                node = Expr::Ternary(Box::new(cond), Box::new(node), Box::new(alt));
            }
        }
        Some(node)
    }

    /// parse relational comparisons which have the lowest precedence
    /// (below arithmetic).  We call `parse_sum` so that `a + b < c` is
    /// interpreted as `(a + b) < c`.
    fn parse_relation(&mut self) -> Option<Expr> {
        let mut node = self.parse_sum()?;
        while let Some(op) = self.peek() {
            if op == "<" || op == ">" || op == "<=" || op == ">=" || op == "==" || op == "!=" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_sum()?;
                node = Expr::BinOp(Box::new(node), op_str, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    /// parse addition and subtraction
    fn parse_sum(&mut self) -> Option<Expr> {
        let mut node = self.parse_term()?;
        while let Some(op) = self.peek() {
            if op == "+" || op == "-" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_term()?;
                node = Expr::BinOp(Box::new(node), op_str, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_power(&mut self) -> Option<Expr> {
        let mut node = self.parse_factor()?;
        while let Some(op) = self.peek() {
            if op == "**" {
                self.next();
                let rhs = self.parse_power()?; // right-associative
                node = Expr::BinOp(Box::new(node), "**".to_string(), Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_term(&mut self) -> Option<Expr> {
        let mut node = self.parse_power()?;
        while let Some(op) = self.peek() {
            if op == "*" || op == "/" || op == "%" {
                let op_str = self.next().unwrap();
                let rhs = self.parse_power()?;
                node = Expr::BinOp(Box::new(node), op_str, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_factor(&mut self) -> Option<Expr> {
        if let Some(tok) = self.peek() {
            if tok == "(" {
                self.next();
                let expr = self.parse_expr();
                self.next(); // consume ')'
                return expr;
            }
            if tok == "-" {
                // unary minus
                self.next();
                if let Some(e) = self.parse_factor() {
                    return Some(Expr::UnaryOp('-', Box::new(e)));
                }
            }
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Option<Expr> {
        if let Some(tok) = self.next() {
            if let Ok(num) = tok.parse::<f64>() {
                return Some(Expr::Const(num));
            }
            // identifier or function call
            if let Some(peek) = self.peek() {
                if peek == "(" {
                    // function call
                    self.next(); // consume '('
                    let mut args = Vec::new();
                    while let Some(p) = self.peek() {
                        if p == ")" {
                            self.next();
                            break;
                        }
                        if let Some(expr) = self.parse_expr() {
                            args.push(expr);
                        }
                        if let Some(comma) = self.peek() {
                            if comma == "," {
                                self.next();
                            }
                        }
                    }
                    return Some(Expr::Call(tok, args));
                }
            }
            return Some(Expr::Var(tok));
        }
        None
    }
}

// compile the simple expression to native code using cranelift
use std::sync::atomic::{AtomicUsize, Ordering};

static JIT_FUNC_COUNTER: once_cell::sync::Lazy<AtomicUsize> =
    once_cell::sync::Lazy::new(|| AtomicUsize::new(0));

fn compile_jit(expr_str: &str, arg_names: &[String]) -> Option<JitEntry> {
    // tokenize and parse
    let tokens = tokenize(expr_str);
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    let arg_count = arg_names.len();

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
            let val = gen_expr(&expr, &mut fb, ptr_val, arg_names, module);
            fb.ins().return_(&[val]);
            fb.finalize();
        }

        let idx = JIT_FUNC_COUNTER.fetch_add(1, Ordering::SeqCst);
        let func_name = format!("jit_func_{}", idx);
        let id = module
            .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
            .ok();
        // propagate failure by returning None from outer function
        if id.is_none() {
            return None;
        }
        let id = id.unwrap();
        module.define_function(id, &mut ctx).ok()?;
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
) -> Value {
    match expr {
        Expr::Const(n) => fb.ins().f64const(*n),
        Expr::Var(name) => {
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
            let l = gen_expr(lhs, fb, ptr, arg_names, module);
            let r = gen_expr(rhs, fb, ptr, arg_names, module);
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
                    // if exponent is a floating‑point constant that is actually
                    // an integer, we can generate multiplications directly using
                    // exponentiation by squaring.  This avoids the overhead of a
                    // `pow` call for common small exponents.  Negative integers
                    // are *not* optimized to avoid introducing divide-by-zero
                    // behaviour; they fall through to `pow`.
                    if let Expr::Const(n) = **rhs {
                        if n == 1.0 {
                            return l;
                        }
                        if n == -1.0 {
                            let one = fb.ins().f64const(1.0);
                            return fb.ins().fdiv(one, l);
                        }
                        // common sqrt form used in vector norms
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
                                // squaring loop
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
                    // fallback: call native pow
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
                    // produce 1.0 for true, 0.0 for false
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
                    // convert signed integer to float
                    fb.ins().fcvt_from_sint(types::F64, intv)
                }
                _ => fb.ins().fadd(l, r),
            }
        }
        Expr::UnaryOp(op, sub) => {
            let v = gen_expr(sub, fb, ptr, arg_names, module);
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
            // evaluate arguments
            let mut arg_vals = Vec::with_capacity(args.len());
            for a in args {
                arg_vals.push(gen_expr(a, fb, ptr, arg_names, module));
            }

            // strip module prefixes (e.g. math.sin -> sin)
            let mut symbol = name.rsplit('.').next().unwrap().to_string();
            // map certain Python convenience names to C counterparts
            if symbol == "abs" {
                symbol = "fabs".to_string();
            }

            // Build signature and import
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
            // evaluate condition and produce a boolean
            let cond_val = gen_expr(cond, fb, ptr, arg_names, module);
            let zero = fb.ins().f64const(0.0);
            // any nonzero float is truthy
            let cond_bool = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);

            let then_val = gen_expr(then_expr, fb, ptr, arg_names, module);
            let else_val = gen_expr(else_expr, fb, ptr, arg_names, module);
            // select uses a B1 condition directly
            fb.ins().select(cond_bool, then_val, else_val)
        }
    }
}

// Python bindings -----------------------------------------------------------

/// Initialize the Python submodule (called from `wrappers.populate_module`).
#[cfg(feature = "pyo3")]
pub(crate) fn init_py(m: &PyModule) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(register_offload, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(offload_call, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(call_jit, m)?)?;
    Ok(())
}

/// Register a Python function for offloading.
///
/// This is the Rust-side hook invoked by the decorator.  For now we simply
/// return the original callable back to Python unmodified, but we log for
/// inspection and ensure the actor pool is initialized when strategy=actor.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn register_offload(
    func: PyObject,
    strategy: Option<String>,
    return_type: Option<String>,
    source_expr: Option<String>,
    arg_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    if let Some(ref s) = strategy {
        if s == "actor" {
            let _ = get_offload_pool();
        } else if s == "jit" {
            if let (Some(expr), Some(args)) = (source_expr.clone(), arg_names.clone()) {
                if let Some(entry) = compile_jit(&expr, &args) {
                    // store compiled entry keyed by python function pointer
                    let key = func.as_ptr() as usize;
                    register_jit(key, entry);
                    eprintln!("[Iris][jit] compiled JIT for function ptr={}", key);
                } else {
                    eprintln!("[Iris][jit] failed to compile expr: {}", expr);
                }
            }
        }
    }
    eprintln!(
        "[Iris][jit] register_offload called strategy={:?} return_type={:?} source={:?} args={:?}",
        strategy, return_type, source_expr, arg_names
    );
    Ok(func)
}

/// If the given Python object exposes a contiguous buffer of `f64`, return a
/// pointer/length pair without copying. The buffer view is released before
/// the function returns.
unsafe fn buffer_ptr_len(obj: &PyAny) -> Option<(*const f64, usize)> {
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
fn execute_jit_func(py: Python, entry: &JitEntry, args: &PyTuple) -> PyResult<PyObject> {
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
                        // Gather non-contiguous arguments into a stack array for Cranelift
                        let mut iter_args: [f64; MAX_FAST_ARGS] = [0.0; MAX_FAST_ARGS];
                        for j in 0..arg_count {
                            iter_args[j] = unsafe { *ptrs[j].add(i) };
                        }
                        results.push(f(iter_args.as_ptr()));
                    }
                } else {
                    // Fallback for > 8 array arguments
                    for i in 0..len {
                        let mut iter_args = Vec::with_capacity(arg_count);
                        for j in 0..arg_count {
                            iter_args.push(unsafe { *ptrs[j].add(i) });
                        }
                        results.push(f(iter_args.as_ptr()));
                    }
                }

                // Zero-copy output: construct a Python array.array directly from our memory bytes
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        results.as_ptr() as *const u8,
                        results.len() * std::mem::size_of::<f64>(),
                    )
                };
                let py_bytes = PyBytes::new(py, byte_slice);
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
                return Err(PyErr::fetch(py));
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
            return Err(PyErr::fetch(py));
        }
        heap_args.push(val);
    }
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let res = f(heap_args.as_ptr());
    Ok(res.into_py(py))
}

/// Execute a Python callable on the offload actor pool, blocking until result.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn offload_call(
    py: Python,
    func: PyObject,
    args: &PyTuple,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let key = func.as_ptr() as usize;
    if let Some(entry) = lookup_jit(key) {
        if let Ok(res) = execute_jit_func(py, &entry, args) {
            return Ok(res);
        }
    }

    let pool = get_offload_pool();

    let (tx, rx) = std::sync::mpsc::channel();
    let task = OffloadTask {
        func: func.into_py(py),
        args: args.into_py(py),
        kwargs: kwargs.map(|d: &PyDict| d.into_py(py)),
        resp: tx,
    };

    pool.sender
        .send(task)
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("offload queue closed"))?;

    // wait for response without holding GIL; blocking recv on std channel is safe
    let result = py.allow_threads(move || {
        match rx.recv() {
            Ok(res) => res,
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "offload task canceled",
            )),
        }
    });

    result
}

/// Directly invoke the JIT-compiled version of a Python function.
#[cfg(feature = "pyo3")]
#[pyfunction]
fn call_jit(
    py: Python,
    func: PyObject,
    args: &PyTuple,
    _kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let key = func.as_ptr() as usize;
    if let Some(entry) = lookup_jit(key) {
        return execute_jit_func(py, &entry, args);
    }
    Err(pyo3::exceptions::PyRuntimeError::new_err("no JIT entry found"))
}

// ------- tests ------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_jit_basic_math() {
        let args = vec!["a".to_string(), "b".to_string()];
        let entry = compile_jit("a + b", &args).expect("should compile");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let values = [1.5, 2.5];
        let result = f(values.as_ptr());
        assert_eq!(result, 4.0);
    }

    #[test]
    fn jit_builder_pic_flag_behavior() {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        if cfg!(target_arch = "aarch64") {
            flag_builder.set("is_pic", "false").unwrap();
        } else {
            flag_builder.set("is_pic", "true").unwrap();
        }
        let isa_builder = cranelift_native::builder().unwrap();
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        assert_eq!(isa.flags().is_pic(), !cfg!(target_arch = "aarch64"));
    }

    #[test]
    fn compile_jit_math_functions() {
        let args = vec!["x".to_string()];
        // trigonometry
        let entry = compile_jit("sin(x)", &args).expect("should compile sin");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [std::f64::consts::PI / 2.0];
        assert!((f(vals.as_ptr()) - 1.0).abs() < 1e-12);

        let entry2 = compile_jit("cos(x)", &args).expect("should compile cos");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        let vals2 = [0.0];
        assert!((g(vals2.as_ptr()) - 1.0).abs() < 1e-12);

        // exponentials / logs
        let entry3 = compile_jit("exp(x)", &args).expect("should compile exp");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
        let vals3 = [1.0];
        assert!((h(vals3.as_ptr()) - std::f64::consts::E).abs() < 1e-12);

        let entry4 = compile_jit("log(x)", &args).expect("should compile log");
        let k: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry4.func_ptr) };
        let vals4 = [std::f64::consts::E];
        assert!((k(vals4.as_ptr()) - 1.0).abs() < 1e-12);

        // square root and tangent
        let entry5 = compile_jit("sqrt(x)", &args).expect("should compile sqrt");
        let s: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry5.func_ptr) };
        let vals5 = [16.0];
        assert!((s(vals5.as_ptr()) - 4.0).abs() < 1e-12);

        let entry6 = compile_jit("tan(x)", &args).expect("should compile tan");
        let t: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry6.func_ptr) };
        let vals6 = [0.0];
        assert!((t(vals6.as_ptr()) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn compile_jit_pow_unary() {
        let args = vec!["a".to_string(), "b".to_string()];
        let entry = compile_jit("pow(a, b)", &args).expect("should compile pow");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [2.0, 3.0];
        assert_eq!(f(vals.as_ptr()), 8.0);
    }

    #[test]
    fn compile_jit_unary_minus() {
        let args = vec!["x".to_string()];
        let entry = compile_jit("-x", &args).expect("should compile unary minus");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [3.5];
        assert_eq!(f(vals.as_ptr()), -3.5);
    }

    #[test]
    fn compile_jit_constants_and_mod() {
        let args = vec!["a".to_string(), "b".to_string()];
        let entry = compile_jit("a % b", &args).expect("should handle mod");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [5.0, 2.0];
        assert_eq!(f(vals.as_ptr()), 1.0);

        // constant names
        let entry2 = compile_jit("pi + e", &[]).expect("consts");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        let empty: [f64; 0] = [];
        assert!((g(empty.as_ptr()) - (std::f64::consts::PI + std::f64::consts::E)).abs() < 1e-12);
    }

    #[test]
    fn compile_jit_relation_and_conditional() {
        let args = vec!["x".to_string(), "y".to_string()];
        let entry = compile_jit("x < y", &args).expect("compare");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [1.0, 2.0];
        assert_eq!(f(vals.as_ptr()), 1.0);

        let entry2 = compile_jit("x >= y", &args).expect("compare2");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        assert_eq!(g(vals.as_ptr()), 0.0);

        // conditional expression (ternary)
        let entry3 = compile_jit("x if x < y else y", &args).expect("ternary");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
        assert_eq!(h(vals.as_ptr()), 1.0);
    }

    #[test]
    fn compile_jit_python_api_call() {
        // ensure execute_jit_func behaves the same as the direct-call path
        Python::with_gil(|py| {
            let args = vec!["x".to_string(), "y".to_string()];
            let entry = compile_jit("x < y", &args).expect("compare");
            let tuple = PyTuple::new(py, &[1.0_f64, 2.0_f64]);
            let res_obj = execute_jit_func(py, &entry, tuple).expect("exec");
            let res: f64 = res_obj.extract(py).unwrap();
            assert_eq!(res, 1.0);
        });
    }

    #[tokio::test]
    async fn compile_jit_python_api_call_tokio() {
        // same as above but run inside tokio's async test harness
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let args = vec!["x".to_string(), "y".to_string()];
            let entry = compile_jit("x < y", &args).expect("compare");
            let tuple = PyTuple::new(py, &[1.0_f64, 2.0_f64]);
            // sanity check tuple contents using safe API
            let a: f64 = tuple.get_item(0).unwrap().extract().unwrap();
            let b: f64 = tuple.get_item(1).unwrap().extract().unwrap();
            assert_eq!(a, 1.0);
            assert_eq!(b, 2.0);
            let res_obj = execute_jit_func(py, &entry, tuple).expect("exec");
            let res: f64 = res_obj.extract(py).unwrap();
            assert_eq!(res, 1.0);
        });
    }

    #[test]
    fn compile_jit_power_op() {
        // simple power
        let entry = compile_jit("2 ** 3", &[]).expect("const power");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let empty: [f64; 0] = [];
        assert_eq!(f(empty.as_ptr()), 8.0);

        // right-associative
        let entry2 = compile_jit("2 ** 3 ** 2", &[]).expect("assoc");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 512.0); // 2^(3^2)

        // strength reduction path should handle small integer exponents
        let entry3 = compile_jit("5 ** 4", &[]).expect("strength");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
        assert_eq!(h(empty.as_ptr()), 625.0);

        // negative constants still use pow (result should be correct)
        let entry4 = compile_jit("2 ** -2", &[]).expect("neg exp");
        let k: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry4.func_ptr) };
        assert!((k(empty.as_ptr()) - 0.25).abs() < 1e-12);

        // fast sqrt rewrite for exponent 0.5
        let entry5 = compile_jit("9 ** 0.5", &[]).expect("sqrt rewrite");
        let q: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry5.func_ptr) };
        assert!((q(empty.as_ptr()) - 3.0).abs() < 1e-12);

        // reciprocal rewrite for exponent -1
        let entry6 = compile_jit("8 ** -1", &[]).expect("reciprocal rewrite");
        let r: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry6.func_ptr) };
        assert!((r(empty.as_ptr()) - 0.125).abs() < 1e-12);
    }

    #[test]
    fn compile_jit_dotted_and_abs() {
        let args = vec!["x".to_string()];
        let entry = compile_jit("math.sin(x)", &args).expect("dotted sin");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [std::f64::consts::PI / 2.0];
        assert!((f(vals.as_ptr()) - 1.0).abs() < 1e-12);

        let entry2 = compile_jit("abs(x)", &args).expect("abs maps to fabs");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        let vals2 = [-4.0];
        assert_eq!(h(vals2.as_ptr()), 4.0);
    }
}
