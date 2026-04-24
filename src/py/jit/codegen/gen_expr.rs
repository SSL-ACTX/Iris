// src/py/jit/codegen/gen_expr.rs
//! Expression-to-Cranelift IR lowering and helpers.

use std::collections::HashMap;

use crate::py::jit::parser::Expr;

use cranelift::prelude::*;
use cranelift_jit::JITModule;
use cranelift_module::{Linkage, Module};

use super::{
    lookup_named_jit, resolve_symbol_alias, SymbolAlias, BREAK_SENTINEL_BITS,
    CONTINUE_SENTINEL_BITS,
};

#[derive(Clone, Copy)]
enum LoopControl<'a> {
    None,
    Break {
        cond: &'a Expr,
        value: &'a Expr,
        invert_cond: bool,
    },
    Continue {
        cond: &'a Expr,
        value: &'a Expr,
        invert_cond: bool,
    },
}

fn detect_loop_control(expr: &Expr) -> LoopControl<'_> {
    match expr {
        Expr::Call(name, args) if args.len() == 2 => {
            let symbol = name.rsplit('.').next().unwrap_or(name.as_str());
            match symbol {
                "break_if" | "loop_break_if" | "break_when" | "loop_break_when" => {
                    LoopControl::Break {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: false,
                    }
                }
                "break_unless" | "loop_break_unless" => LoopControl::Break {
                    cond: &args[0],
                    value: &args[1],
                    invert_cond: true,
                },
                "continue_if" | "loop_continue_if" | "continue_when" | "loop_continue_when" => {
                    LoopControl::Continue {
                        cond: &args[0],
                        value: &args[1],
                        invert_cond: false,
                    }
                }
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

/// Lower a parsed `Expr` into Cranelift IR.
pub(crate) fn gen_expr(
    expr: &Expr,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
    match expr {
        Expr::Const(n) => fb.ins().f64const(*n),
        Expr::Var(name) => gen_var(name, fb, ptr, arg_names, locals),
        Expr::BinOp(lhs, op, rhs) => gen_binop(lhs, op, rhs, fb, ptr, arg_names, module, locals),
        Expr::UnaryOp(op, sub) => gen_unaryop(*op, sub, fb, ptr, arg_names, module, locals),
        Expr::Call(name, args) => gen_call(name, args, fb, ptr, arg_names, module, locals),
        Expr::Ternary(cond, then_expr, else_expr) => gen_ternary(
            cond, then_expr, else_expr, fb, ptr, arg_names, module, locals,
        ),
        Expr::SumOver { .. } => panic!("SumOver should have been transformed before codegen"),
        Expr::AnyOver { .. } => panic!("AnyOver should have been transformed before codegen"),
        Expr::AllOver { .. } => panic!("AllOver should have been transformed before codegen"),
        Expr::AnyFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => gen_any_for(
            iter_var, start, end, step, body, pred, fb, ptr, arg_names, module, locals,
        ),
        Expr::AllFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => gen_all_for(
            iter_var, start, end, step, body, pred, fb, ptr, arg_names, module, locals,
        ),
        Expr::SumFor {
            iter_var,
            start,
            end,
            step,
            body,
            pred,
        } => gen_sum_for(
            iter_var, start, end, step, body, pred, fb, ptr, arg_names, module, locals,
        ),
    }
}

fn gen_var(
    name: &str,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    locals: &HashMap<String, Value>,
) -> Value {
    if let Some(v) = locals.get(name) {
        return *v;
    }
    if let Some(idx) = arg_names.iter().position(|n| n == name) {
        let offset = (idx as i64) * 8;
        let offset_const = fb.ins().iconst(types::I64, offset);
        let addr1 = fb.ins().iadd(ptr, offset_const);
        return fb.ins().load(types::F64, MemFlags::new(), addr1, 0);
    }
    // treat named constants
    match name {
        "pi" => return fb.ins().f64const(std::f64::consts::PI),
        "e" => return fb.ins().f64const(std::f64::consts::E),
        _ => {}
    }
    let idx = 0;
    let offset = (idx as i64) * 8;
    let offset_const = fb.ins().iconst(types::I64, offset);
    let addr1 = fb.ins().iadd(ptr, offset_const);
    fb.ins().load(types::F64, MemFlags::new(), addr1, 0)
}

#[allow(clippy::too_many_arguments)]
fn gen_binop(
    lhs: &Expr,
    op: &str,
    rhs: &Expr,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
    let l = gen_expr(lhs, fb, ptr, arg_names, module, locals);
    let r = gen_expr(rhs, fb, ptr, arg_names, module, locals);
    match op {
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
            let local = module.declare_func_in_func(fid, fb.func);
            let call = fb.ins().call(local, &[l, r]);
            fb.inst_results(call)[0]
        }
        "**" => {
            if let Expr::Const(n) = rhs {
                if *n == 1.0 {
                    return l;
                }
                if *n == -1.0 {
                    let one = fb.ins().f64const(1.0);
                    return fb.ins().fdiv(one, l);
                }
                if *n == 0.5 {
                    let mut sig = module.make_signature();
                    sig.params.push(AbiParam::new(types::F64));
                    sig.returns.push(AbiParam::new(types::F64));
                    let fid = module
                        .declare_function("sqrt", Linkage::Import, &sig)
                        .expect("failed to declare sqrt");
                    let local = module.declare_func_in_func(fid, fb.func);
                    let call = fb.ins().call(local, &[l]);
                    return fb.inst_results(call)[0];
                }
                if n.fract() == 0.0 {
                    let exp = *n as i64;
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
            let local = module.declare_func_in_func(fid, fb.func);
            let call = fb.ins().call(local, &[l, r]);
            fb.inst_results(call)[0]
        }
        "<" | ">" | "<=" | ">=" | "==" | "!=" => {
            let cc = match op {
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

fn gen_unaryop(
    op: char,
    sub: &Expr,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
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

#[allow(clippy::too_many_arguments)]
fn gen_ternary(
    cond: &Expr,
    then_expr: &Expr,
    else_expr: &Expr,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
    let cond_val = gen_expr(cond, fb, ptr, arg_names, module, locals);
    let zero = fb.ins().f64const(0.0);
    let cond_bool = fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero);
    let then_val = gen_expr(then_expr, fb, ptr, arg_names, module, locals);
    let else_val = gen_expr(else_expr, fb, ptr, arg_names, module, locals);
    fb.ins().select(cond_bool, then_val, else_val)
}

fn gen_call(
    name: &str,
    args: &[Expr],
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
    let symbol = name.rsplit('.').next().unwrap().to_string();

    if (symbol == "any_while" || symbol == "all_while") && args.len() == 5 {
        return gen_any_all_while(&symbol, args, fb, ptr, arg_names, module, locals);
    }

    if symbol == "sum_while" && args.len() == 5 {
        return gen_sum_while(args, fb, ptr, arg_names, module, locals);
    }

    if (symbol == "break_on_nan"
        || symbol == "loop_break_on_nan"
        || symbol == "continue_on_nan"
        || symbol == "loop_continue_on_nan")
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

    if symbol == "let_bind" && args.len() == 3 {
        if let Expr::Var(var_name) = &args[0] {
            let v = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
            let mut new_locals = locals.clone();
            new_locals.insert(var_name.clone(), v);
            return gen_expr(&args[2], fb, ptr, arg_names, module, &new_locals);
        }
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
                let local = module.declare_func_in_func(func_id, fb.func);
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

    if let Some(named) = lookup_named_jit(&symbol) {
        if named.arg_count == args.len() {
            let helper_name = match args.len() {
                0 => Some("iris_jit_invoke_0"),
                1 => Some("iris_jit_invoke_1"),
                2 => Some("iris_jit_invoke_2"),
                3 => Some("iris_jit_invoke_3"),
                4 => Some("iris_jit_invoke_4"),
                5 => Some("iris_jit_invoke_5"),
                6 => Some("iris_jit_invoke_6"),
                7 => Some("iris_jit_invoke_7"),
                8 => Some("iris_jit_invoke_8"),
                9 => Some("iris_jit_invoke_9"),
                10 => Some("iris_jit_invoke_10"),
                11 => Some("iris_jit_invoke_11"),
                12 => Some("iris_jit_invoke_12"),
                13 => Some("iris_jit_invoke_13"),
                14 => Some("iris_jit_invoke_14"),
                15 => Some("iris_jit_invoke_15"),
                16 => Some("iris_jit_invoke_16"),
                _ => None,
            };

            if let Some(helper_name) = helper_name {
                let mut arg_vals = Vec::with_capacity(args.len() + 1);
                arg_vals.push(fb.ins().iconst(types::I64, named.func_ptr as i64));
                for a in args {
                    arg_vals.push(gen_expr(a, fb, ptr, arg_names, module, locals));
                }

                let mut sig = module.make_signature();
                sig.params.push(AbiParam::new(types::I64));
                for _ in 0..args.len() {
                    sig.params.push(AbiParam::new(types::F64));
                }
                sig.returns.push(AbiParam::new(types::F64));
                let func_id = module
                    .declare_function(helper_name, Linkage::Import, &sig)
                    .expect("failed to declare named jit invoke helper");
                let local = module.declare_func_in_func(func_id, fb.func);
                let call = fb.ins().call(local, &arg_vals);
                return fb.inst_results(call)[0];
            }
        }
    }

    if (symbol == "break_if"
        || symbol == "loop_break_if"
        || symbol == "break_when"
        || symbol == "loop_break_when"
        || symbol == "break_unless"
        || symbol == "loop_break_unless"
        || symbol == "continue_if"
        || symbol == "loop_continue_if"
        || symbol == "continue_when"
        || symbol == "loop_continue_when"
        || symbol == "continue_unless"
        || symbol == "loop_continue_unless")
        && args.len() == 2
    {
        let cond_val = gen_expr(&args[0], fb, ptr, arg_names, module, locals);
        let value_val = gen_expr(&args[1], fb, ptr, arg_names, module, locals);
        let zero = fb.ins().f64const(0.0);
        let cond_true = if symbol == "break_unless"
            || symbol == "loop_break_unless"
            || symbol == "continue_unless"
            || symbol == "loop_continue_unless"
        {
            fb.ins().fcmp(FloatCC::Equal, cond_val, zero)
        } else {
            fb.ins().fcmp(FloatCC::NotEqual, cond_val, zero)
        };
        let sentinel = if symbol == "break_if"
            || symbol == "loop_break_if"
            || symbol == "break_when"
            || symbol == "loop_break_when"
            || symbol == "break_unless"
            || symbol == "loop_break_unless"
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
    let local = module.declare_func_in_func(func_id, fb.func);
    let call = fb.ins().call(local, &arg_vals);
    fb.inst_results(call)[0]
}

fn gen_any_all_while(
    symbol: &str,
    args: &[Expr],
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
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
            let continue_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
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
            fb.ins()
                .icmp_imm(IntCC::Equal, body_bits, BREAK_SENTINEL_BITS as i64);
        let is_continue_sentinel =
            fb.ins()
                .icmp_imm(IntCC::Equal, body_bits, CONTINUE_SENTINEL_BITS as i64);
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
            fb.ins()
                .jump(continue_block, &[step_val, zero, budget_next]);

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
        fb.ins()
            .jump(loop_block, &[next_iter, next_acc, next_budget]);
        fb.seal_block(body_block);
        fb.seal_block(budget_exit_block);
        fb.seal_block(budget_ok_block);
        fb.seal_block(continue_block);
        fb.seal_block(loop_block);
        fb.switch_to_block(exit_block);
        fb.seal_block(exit_block);
        return fb.block_params(exit_block)[0];
    }
    fb.ins().f64const(0.0) // Fallback if format is wrong
}

fn gen_sum_while(
    args: &[Expr],
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
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
            let continue_cond_val = gen_expr(cond, fb, ptr, arg_names, module, &while_locals);
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
            fb.ins()
                .icmp_imm(IntCC::Equal, body_bits, BREAK_SENTINEL_BITS as i64);
        let is_continue_sentinel =
            fb.ins()
                .icmp_imm(IntCC::Equal, body_bits, CONTINUE_SENTINEL_BITS as i64);
        let stop_now = fb.ins().bor(break_true, is_break_sentinel);
        let skip_body = fb.ins().bor(continue_true, is_continue_sentinel);

        let effective_body = fb.ins().select(skip_body, zero, body_val);
        let next_acc = fb.ins().fadd(acc_val, effective_body);
        let sum_break_block = fb.create_block();
        fb.ins().brnz(stop_now, sum_break_block, &[]);
        fb.ins()
            .jump(continue_block, &[step_val, next_acc, budget_next]);
        fb.switch_to_block(sum_break_block);
        fb.ins().jump(exit_block, &[acc_val]);
        fb.seal_block(sum_break_block);

        fb.switch_to_block(continue_block);
        let next_iter = fb.block_params(continue_block)[0];
        let next_acc = fb.block_params(continue_block)[1];
        let next_budget = fb.block_params(continue_block)[2];
        fb.ins()
            .jump(loop_block, &[next_iter, next_acc, next_budget]);

        fb.seal_block(body_block);
        fb.seal_block(budget_exit_block);
        fb.seal_block(budget_ok_block);
        fb.seal_block(continue_block);
        fb.seal_block(loop_block);
        fb.switch_to_block(exit_block);
        fb.seal_block(exit_block);
        return fb.block_params(exit_block)[0];
    }
    fb.ins().f64const(0.0) // Fallback
}

#[allow(clippy::too_many_arguments)]
fn gen_any_for(
    iter_var: &str,
    start: &Expr,
    end: &Expr,
    step: &Option<Box<Expr>>,
    body: &Expr,
    pred: &Option<Box<Expr>>,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
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
    body_locals.insert(iter_var.to_string(), i_val);

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

#[allow(clippy::too_many_arguments)]
fn gen_all_for(
    iter_var: &str,
    start: &Expr,
    end: &Expr,
    step: &Option<Box<Expr>>,
    body: &Expr,
    pred: &Option<Box<Expr>>,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
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
    body_locals.insert(iter_var.to_string(), i_val);

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

#[allow(clippy::too_many_arguments)]
fn gen_sum_for(
    iter_var: &str,
    start: &Expr,
    end: &Expr,
    step: &Option<Box<Expr>>,
    body: &Expr,
    pred: &Option<Box<Expr>>,
    fb: &mut FunctionBuilder,
    ptr: Value,
    arg_names: &[String],
    module: &mut JITModule,
    locals: &HashMap<String, Value>,
) -> Value {
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
    body_locals.insert(iter_var.to_string(), i_val);

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
