// src/py/jit/codegen/compiler.rs
//! Top-level JIT compilation functions for Iris.

use std::collections::HashMap;

use cranelift::prelude::*;
use cranelift_module::{Linkage, Module};

use crate::py::jit::heuristics;
use crate::py::jit::parser::Expr;

use super::*;

/// Compile an expression string into a native function entry.
#[allow(dead_code)]
pub fn compile_jit(expr_str: &str, arg_names: &[String]) -> Option<JitEntry> {
    compile_jit_with_return_type(expr_str, arg_names, JitReturnType::Float)
}

pub fn compile_jit_with_return_type(
    expr_str: &str,
    arg_names: &[String],
    return_type: JitReturnType,
) -> Option<JitEntry> {
    compile_jit_impl(expr_str, arg_names, true, return_type)
}

/// Compile multiple speculative versions of the same expression in parallel.
pub fn compile_jit_quantum(
    expr_str: &str,
    arg_names: &[String],
    return_type: JitReturnType,
) -> Vec<JitEntry> {
    let optimized = compile_jit_impl(expr_str, arg_names, true, return_type);
    let baseline = compile_jit_impl(expr_str, arg_names, false, return_type);

    let mut out = Vec::new();
    if let Some(e) = optimized {
        out.push(e.with_strategy(QuantumVariantStrategy::Auto));
    }
    if let Some(e) = baseline {
        out.push(e.with_strategy(QuantumVariantStrategy::ScalarFallback));
    }

    if let Some(exp) = compile_jit_impl(expr_str, arg_names, true, return_type)
        .map(|e| e.with_strategy(QuantumVariantStrategy::FastTrigExperiment))
    {
        out.push(exp);
    }

    crate::py::jit::jit_log(|| format!("[Iris][jit][quantum] built {} variants", out.len()));
    out
}

pub fn compile_jit_quantum_variant(
    expr_str: &str,
    arg_names: &[String],
    return_type: JitReturnType,
    variant_index: usize,
) -> Option<JitEntry> {
    match variant_index {
        0 => compile_jit_impl(expr_str, arg_names, true, return_type)
            .map(|entry| entry.with_strategy(QuantumVariantStrategy::Auto)),
        1 => compile_jit_impl(expr_str, arg_names, false, return_type)
            .map(|entry| entry.with_strategy(QuantumVariantStrategy::ScalarFallback)),
        2 => compile_jit_impl(expr_str, arg_names, true, return_type)
            .map(|entry| entry.with_strategy(QuantumVariantStrategy::FastTrigExperiment)),
        _ => None,
    }
}

pub(crate) fn compile_jit_impl(
    expr_str: &str,
    arg_names: &[String],
    optimize_ast: bool,
    return_type: JitReturnType,
) -> Option<JitEntry> {
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
    if let Expr::SumOver {
        iter_var,
        container,
        body,
        pred,
    } = expr.clone()
    {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!(
                        "[Iris][jit] converting SumOver '{}' in {}",
                        iter_var, cont_name
                    )
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
    if let Expr::AnyOver {
        iter_var,
        container,
        body,
        pred,
    } = expr.clone()
    {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!(
                        "[Iris][jit] converting AnyOver '{}' in {}",
                        iter_var, cont_name
                    )
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
    if let Expr::AllOver {
        iter_var,
        container,
        body,
        pred,
    } = expr.clone()
    {
        if let Expr::Var(ref cont_name) = *container {
            if adjusted_args.len() == 1 && adjusted_args[0] == *cont_name {
                crate::py::jit::jit_log(|| {
                    format!(
                        "[Iris][jit] converting AllOver '{}' in {}",
                        iter_var, cont_name
                    )
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
    let lowered_kernel = detect_lowered_kernel(&expr, &adjusted_args);
    if let Some(kernel) = lowered_kernel.as_ref() {
        crate::py::jit::jit_log(|| format!("[Iris][jit][lower] selected kernel={:?}", kernel));
    }

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

        if jit_dump_clif_enabled() {
            crate::py::jit::jit_log(|| format!("[Iris][jit][clif] {}", ctx.func.display()));
        }

        let idx = next_jit_func_id();
        let func_name = format!("jit_func_{}", idx);
        let id = module
            .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
            .ok()?;
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
            return_type,
            lowered_kernel,
            variant_strategy: QuantumVariantStrategy::Auto,
        })
    })
}
