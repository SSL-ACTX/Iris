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
    BinOp(Box<Expr>, char, Box<Expr>),
    Call(String, Vec<Expr>),
    UnaryOp(char, Box<Expr>),
}

// parser helpers
fn tokenize(expr: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for c in expr.chars() {
        if c.is_whitespace() {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
        } else if "+-*/(),".contains(c) {
            if !cur.is_empty() {
                tokens.push(cur.clone());
                cur.clear();
            }
            tokens.push(c.to_string());
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

// Pratt parser implementation
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
        let mut node = self.parse_term()?;
        while let Some(op) = self.peek() {
            if op == "+" || op == "-" {
                let op = self.next().unwrap().chars().next().unwrap();
                let rhs = self.parse_term()?;
                node = Expr::BinOp(Box::new(node), op, Box::new(rhs));
                continue;
            }
            break;
        }
        Some(node)
    }

    fn parse_term(&mut self) -> Option<Expr> {
        let mut node = self.parse_factor()?;
        while let Some(op) = self.peek() {
            if op == "*" || op == "/" {
                let op = self.next().unwrap().chars().next().unwrap();
                let rhs = self.parse_factor()?;
                node = Expr::BinOp(Box::new(node), op, Box::new(rhs));
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
fn compile_jit(expr_str: &str, arg_names: &[String]) -> Option<JitEntry> {
    // tokenize and parse
    let tokens = tokenize(expr_str);
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    let arg_count = arg_names.len();

    // create a new JIT module for each compilation (avoids sync issues)
    // On aarch64 we avoid PIC so that cranelift-jit does not try to emit
    // PLT entries (the backend currently panics with an assert when PLT
    // generation is attempted).  Disabling PIC is safe for the few small
    // functions we emit because the module is only used locally and we
    // never relocate the code.
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
    let mut module = JITModule::new(builder);
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
        let val = gen_expr(&expr, &mut fb, ptr_val, arg_names, &mut module);
        fb.ins().return_(&[val]);
        fb.finalize();
    }

    let id = module
        .declare_function("jit_func", Linkage::Local, &ctx.func.signature)
        .ok()?;
    module.define_function(id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    module.finalize_definitions();

    let code_ptr = module.get_finalized_function(id) as usize;
    Some(JitEntry {
        func_ptr: code_ptr,
        arg_count,
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
            let idx = arg_names.iter().position(|n| n == name).unwrap_or(0);
            let offset = (idx as i64) * 8;
            let offset_const = fb.ins().iconst(types::I64, offset);
            let addr1 = fb.ins().iadd(ptr, offset_const);
            fb.ins().load(types::F64, MemFlags::new(), addr1, 0)
        }
        Expr::BinOp(lhs, op, rhs) => {
            let l = gen_expr(lhs, fb, ptr, arg_names, module);
            let r = gen_expr(rhs, fb, ptr, arg_names, module);
            match op {
                '+' => fb.ins().fadd(l, r),
                '-' => fb.ins().fsub(l, r),
                '*' => fb.ins().fmul(l, r),
                '/' => fb.ins().fdiv(l, r),
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

            // All of our external math helpers use f64->f64 or (f64,f64)->f64.
            // Build a signature accordingly and import the symbol by name.
            let mut sig = module.make_signature();
            for _ in 0..arg_vals.len() {
                sig.params.push(AbiParam::new(types::F64));
            }
            sig.returns.push(AbiParam::new(types::F64));
            let func_id = module
                .declare_function(name, Linkage::Import, &sig)
                .expect("failed to declare external function");
            let local = module.declare_func_in_func(func_id, &mut fb.func);
            let call = fb.ins().call(local, &arg_vals);
            fb.inst_results(call)[0]
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
        let entry = compile_jit("sin(x)", &args).expect("should compile sin");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [std::f64::consts::PI / 2.0];
        assert!((f(vals.as_ptr()) - 1.0).abs() < 1e-12);

        let entry2 = compile_jit("cos(x)", &args).expect("should compile cos");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        let vals2 = [0.0];
        assert!((g(vals2.as_ptr()) - 1.0).abs() < 1e-12);
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
}
