// src/py/jit/tests.rs
//! Unit tests formerly embedded in `mod.rs`.  Now separated for clarity.

use super::*;
use crate::py::jit::codegen::execute_jit_func;

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
fn compile_jit_quantum_variants() {
    let args = vec!["x".to_string(), "y".to_string()];
    let entries = compile_jit_quantum("x + y", &args);
    assert!(!entries.is_empty(), "quantum compile should produce at least one variant");
    for entry in entries {
        assert_eq!(entry.arg_count, 2);
    }
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
fn compile_jit_nested_parens_generator() {
    let args = vec!["n".to_string()];
    let entry = compile_jit("sum((i * i for i in range(int(n))))", &args);
    assert!(entry.is_some(), "nested parens generator should compile");
}

#[test]
fn compile_jit_vector_generator_should_compile() {
    let args = vec!["x".to_string()];
    let entry = compile_jit("sum((x_i * x_i for x_i in x))", &args)
        .expect("vector generator should now compile");
    // entry.arg_count should equal 1 (element argument)
    assert_eq!(entry.arg_count, 1);
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

    // hyperbolics
    let entryh = compile_jit("sinh(x)", &args).expect("should compile sinh");
    let sh: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entryh.func_ptr) };
    let vsh = [1.0];
    assert!((sh(vsh.as_ptr()) - vsh[0].sinh()).abs() < 1e-12);

    let entryh2 = compile_jit("cosh(x)", &args).expect("should compile cosh");
    let ch: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entryh2.func_ptr) };
    let vch = [0.0];
    assert!((ch(vch.as_ptr()) - vch[0].cosh()).abs() < 1e-12);

    let entryh3 = compile_jit("tanh(x)", &args).expect("should compile tanh");
    let th: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entryh3.func_ptr) };
    let vth = [0.0];
    assert!((th(vth.as_ptr()) - vth[0].tanh()).abs() < 1e-12);

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

    let entry7 = compile_jit("int(x) + float(1)", &args).expect("should compile int/float casts");
    let u: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry7.func_ptr) };
    let vals7 = [2.9];
    assert!((u(vals7.as_ptr()) - 3.0).abs() < 1e-12);
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

    #[test]
    fn compile_jit_sum_range_loop() {
        let args = vec!["n".to_string()];
        let entry = compile_jit("sum(i for i in range(n))", &args).expect("sum range loop");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [5.0];
        assert_eq!(f(vals.as_ptr()), 10.0);
    }

    #[test]
    fn compile_jit_sum_range_loop_with_body_expr() {
        let args = vec!["x".to_string(), "n".to_string()];
        let entry = compile_jit("sum(i * x for i in range(n))", &args).expect("sum range with body expr");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [2.0, 4.0];
        assert_eq!(f(vals.as_ptr()), 12.0);
    }

    #[test]
    fn compile_jit_sum_range_negative_step() {
        let entry = compile_jit("sum(i for i in range(5, 0, -1))", &vec![]).expect("negative step");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let empty: [f64; 0] = [];
        assert_eq!(f(empty.as_ptr()), 15.0);
    }

    #[test]
    fn compile_jit_sum_range_negative_step_dynamic() {
        let args = vec!["a".to_string(), "b".to_string(), "s".to_string()];
        let entry = compile_jit("sum(i for i in range(a, b, s))", &args).expect("dynamic negative step");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [5.0, 0.0, -1.0];
        assert_eq!(f(vals.as_ptr()), 15.0);
    }

    #[test]
    fn compile_jit_sum_container_with_predicate() {
        let args = vec!["x".to_string()];
        let entry = compile_jit("sum(x_i for x_i in x if x_i > 0)", &args)
            .expect("container generator with predicate should compile");
        assert_eq!(entry.arg_count, 1);
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

        let positive = [3.0];
        assert_eq!(f(positive.as_ptr()), 3.0);

        let negative = [-2.0];
        assert_eq!(f(negative.as_ptr()), 0.0);
    }

    #[test]
    fn compile_jit_any_container_with_predicate() {
        let args = vec!["x".to_string()];
        let entry = compile_jit("any(x_i > 0 for x_i in x if x_i != 0)", &args)
            .expect("container any with predicate should compile");
        assert_eq!(entry.arg_count, 1);
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

        let positive = [2.0];
        assert_eq!(f(positive.as_ptr()), 1.0);

        let zero = [0.0];
        assert_eq!(f(zero.as_ptr()), 0.0);

        let negative = [-3.0];
        assert_eq!(f(negative.as_ptr()), 0.0);
    }

    #[test]
    fn compile_jit_all_container_with_predicate() {
        let args = vec!["x".to_string()];
        let entry = compile_jit("all(x_i > 0 for x_i in x if x_i != 0)", &args)
            .expect("container all with predicate should compile");
        assert_eq!(entry.arg_count, 1);
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

        let positive = [2.0];
        assert_eq!(f(positive.as_ptr()), 1.0);

        let zero = [0.0];
        assert_eq!(f(zero.as_ptr()), 1.0);

        let negative = [-3.0];
        assert_eq!(f(negative.as_ptr()), 0.0);
    }

    #[test]
    fn compile_jit_any_range_generator() {
        let entry = compile_jit("any(i > 3 for i in range(5))", &vec![]).expect("any range");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let empty: [f64; 0] = [];
        assert_eq!(f(empty.as_ptr()), 1.0);

        let entry2 = compile_jit("any(i > 10 for i in range(5))", &vec![]).expect("any range false");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 0.0);

        let entry3 = compile_jit("any(i > 0 for i in range(0))", &vec![]).expect("any empty");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
        assert_eq!(h(empty.as_ptr()), 0.0);
    }

    #[test]
    fn compile_jit_all_range_generator() {
        let entry = compile_jit("all(i < 5 for i in range(5))", &vec![]).expect("all range true");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let empty: [f64; 0] = [];
        assert_eq!(f(empty.as_ptr()), 1.0);

        let entry2 = compile_jit("all(i < 3 for i in range(5))", &vec![]).expect("all range false");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 0.0);

        let entry3 = compile_jit("all(i > 0 for i in range(0))", &vec![]).expect("all empty");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
        assert_eq!(h(empty.as_ptr()), 1.0);
    }

    #[test]
    fn compile_jit_any_all_with_predicate() {
        let empty: [f64; 0] = [];

        let any_pred = compile_jit("any(i > 3 for i in range(6) if i % 2 == 0)", &vec![])
            .expect("any with predicate");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_pred.func_ptr) };
        assert_eq!(f(empty.as_ptr()), 1.0); // included set: {0,2,4}

        let all_pred = compile_jit("all(i % 2 == 0 for i in range(6) if i < 5)", &vec![])
            .expect("all with predicate");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_pred.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 0.0); // included set: {0,1,2,3,4}
    }

    #[test]
    fn compile_jit_sum_with_break_continue_intrinsics() {
        let empty: [f64; 0] = [];

        let cont = compile_jit("sum(continue_if(i % 2 == 0, i) for i in range(6))", &vec![])
            .expect("sum continue_if");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(cont.func_ptr) };
        assert_eq!(f(empty.as_ptr()), 9.0); // 1+3+5

        let brk = compile_jit("sum(break_if(i >= 4, i) for i in range(10))", &vec![])
            .expect("sum break_if");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(brk.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 6.0); // 0+1+2+3
    }

    #[test]
    fn compile_jit_any_all_with_break_continue_intrinsics() {
        let empty: [f64; 0] = [];

        let any_cont = compile_jit("any(continue_if(i < 3, i > 5) for i in range(8))", &vec![])
            .expect("any continue_if");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_cont.func_ptr) };
        assert_eq!(f(empty.as_ptr()), 1.0);

        let any_break = compile_jit("any(break_if(i >= 3, i > 10) for i in range(8))", &vec![])
            .expect("any break_if");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_break.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 0.0);

        let all_cont = compile_jit("all(continue_if(i < 3, i < 10) for i in range(6))", &vec![])
            .expect("all continue_if");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_cont.func_ptr) };
        assert_eq!(h(empty.as_ptr()), 1.0);

        let all_break = compile_jit("all(break_if(i >= 4, i < 10) for i in range(6))", &vec![])
            .expect("all break_if");
        let q: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_break.func_ptr) };
        assert_eq!(q(empty.as_ptr()), 1.0);
    }

    #[test]
    fn compile_jit_with_break_continue_unless_intrinsics() {
        let empty: [f64; 0] = [];

        let sum_break_unless = compile_jit("sum(break_unless(i < 4, i) for i in range(10))", &vec![])
            .expect("sum break_unless");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_break_unless.func_ptr) };
        assert_eq!(f(empty.as_ptr()), 6.0);

        let sum_continue_unless = compile_jit("sum(continue_unless(i % 2 == 1, i) for i in range(6))", &vec![])
            .expect("sum continue_unless");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_continue_unless.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 9.0);

        let any_break_unless = compile_jit("any(break_unless(i < 3, i > 10) for i in range(8))", &vec![])
            .expect("any break_unless");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_break_unless.func_ptr) };
        assert_eq!(h(empty.as_ptr()), 0.0);

        let all_continue_unless = compile_jit("all(continue_unless(i % 2 == 0, i < 10) for i in range(6))", &vec![])
            .expect("all continue_unless");
        let q: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_continue_unless.func_ptr) };
        assert_eq!(q(empty.as_ptr()), 1.0);

        let sum_break_when = compile_jit("sum(break_when(i >= 4, i) for i in range(10))", &vec![])
            .expect("sum break_when");
        let r: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_break_when.func_ptr) };
        assert_eq!(r(empty.as_ptr()), 6.0);

        let sum_continue_when = compile_jit("sum(continue_when(i % 2 == 0, i) for i in range(6))", &vec![])
            .expect("sum continue_when");
        let s: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_continue_when.func_ptr) };
        assert_eq!(s(empty.as_ptr()), 9.0);

    }

    #[test]
    fn compile_jit_if_else_control_flow_function() {
        let args = vec!["x".to_string(), "y".to_string()];
        let entry = compile_jit("if_else(x < y, x, y)", &args).expect("if_else compile");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [2.0, 5.0];
        assert_eq!(f(vals.as_ptr()), 2.0);

        let entry2 = compile_jit("sum(if_else(i % 2 == 0, i, 0) for i in range(6))", &vec![])
            .expect("if_else in reduction compile");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        let empty: [f64; 0] = [];
        assert_eq!(g(empty.as_ptr()), 6.0);
    }

    #[test]
    fn compile_jit_while_reductions() {
        let args = vec!["n".to_string()];

        let sum_entry = compile_jit("sum_while(i, 0, i < n, i + 1, i)", &args)
            .expect("sum_while compile");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_entry.func_ptr) };
        let vals = [5.0];
        assert_eq!(f(vals.as_ptr()), 10.0);

        let sum_entry2 = compile_jit("sum_while(i, 1, i <= n, i + 1, i)", &args)
            .expect("sum_while inclusive compile");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_entry2.func_ptr) };
        assert_eq!(g(vals.as_ptr()), 15.0);
    }

    #[test]
    fn compile_jit_while_reductions_with_loop_control() {
        let empty: [f64; 0] = [];

        let sum_break = compile_jit(
            "sum_while(i, 0, i < 10, i + 1, break_if(i >= 4, i))",
            &vec![],
        )
        .expect("sum_while break_if compile");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_break.func_ptr) };
        assert_eq!(f(empty.as_ptr()), 6.0);

        let sum_continue = compile_jit(
            "sum_while(i, 0, i < 10, i + 1, continue_if(i % 2 == 0, i))",
            &vec![],
        )
        .expect("sum_while continue_if compile");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_continue.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 25.0);

        let sum_break_nan = compile_jit(
            "sum_while(i, 0, i < 10, i + 1, break_on_nan((i - i) / (i - i)))",
            &vec![],
        )
        .expect("sum_while break_on_nan compile");
        let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_break_nan.func_ptr) };
        assert_eq!(h(empty.as_ptr()), 0.0);

        let any_while = compile_jit(
            "any_while(i, 0, i < 8, i + 1, i >= 6)",
            &vec![],
        )
        .expect("any_while compile");
        let q: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_while.func_ptr) };
        assert_eq!(q(empty.as_ptr()), 1.0);

        let all_while = compile_jit(
            "all_while(i, 0, i < 8, i + 1, i < 8)",
            &vec![],
        )
        .expect("all_while compile");
        let r: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_while.func_ptr) };
        assert_eq!(r(empty.as_ptr()), 1.0);

        let any_continue_nan = compile_jit(
            "any_while(i, 0, i < 5, i + 1, continue_on_nan((i - i) / (i - i)))",
            &vec![],
        )
        .expect("any_while continue_on_nan compile");
        let s: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_continue_nan.func_ptr) };
        assert_eq!(s(empty.as_ptr()), 0.0);
    }

    #[test]
    fn compile_jit_function_inlining_min_max() {
        let args = vec!["x".to_string(), "y".to_string()];
        let entry = compile_jit("max(x, y) - min(x, y)", &args).expect("min/max compile");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [2.0, 7.0];
        assert_eq!(f(vals.as_ptr()), 5.0);

        let entry2 = compile_jit("sum(max(i, 2) - min(i, 2) for i in range(5))", &vec![])
            .expect("min/max in reduction compile");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        let empty: [f64; 0] = [];
        assert_eq!(g(empty.as_ptr()), 6.0);
    }

    #[test]
    fn compile_jit_range_step_and_predicate() {
        let entry = compile_jit("sum(i for i in range(0,10,2))", &vec![]).expect("step");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let empty: [f64; 0] = [];
        assert_eq!(f(empty.as_ptr()), 20.0);

        let entry2 = compile_jit("sum(i for i in range(5) if i % 2 == 0)", &vec![]).expect("pred");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        assert_eq!(g(empty.as_ptr()), 6.0);
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
    fn compile_jit_boolean_and_or() {
        let args = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let entry = compile_jit("x < y and y < z", &args).expect("and compare");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals_true = [1.0, 2.0, 3.0];
        let vals_false = [3.0, 2.0, 1.0];
        assert_eq!(f(vals_true.as_ptr()), 1.0);
        assert_eq!(f(vals_false.as_ptr()), 0.0);

        let entry2 = compile_jit("x > y or y < z", &args).expect("or compare");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        assert_eq!(g(vals_true.as_ptr()), 1.0);
        assert_eq!(g(vals_false.as_ptr()), 1.0);
    }

    #[test]
    fn compile_jit_boolean_not() {
        let args = vec!["x".to_string(), "y".to_string()];
        let entry = compile_jit("not x < y", &args).expect("not compare");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals_true = [3.0, 2.0];
        let vals_false = [1.0, 2.0];
        assert_eq!(f(vals_true.as_ptr()), 1.0);
        assert_eq!(f(vals_false.as_ptr()), 0.0);
    }

    #[test]
    fn compile_jit_boolean_literals() {
        let args = vec!["x".to_string(), "y".to_string()];

        let entry = compile_jit("x if True else y", &args).expect("ternary true literal");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals = [3.0, 9.0];
        assert_eq!(f(vals.as_ptr()), 3.0);

        let entry2 = compile_jit("x if False else y", &args).expect("ternary false literal");
        let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
        assert_eq!(g(vals.as_ptr()), 9.0);
    }

    #[test]
    fn compile_jit_comparison_chain() {
        let args = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let entry = compile_jit("x < y < z", &args).expect("comparison chain");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
        let vals_true = [1.0, 2.0, 3.0];
        let vals_false = [3.0, 2.0, 1.0];
        assert_eq!(f(vals_true.as_ptr()), 1.0);
        assert_eq!(f(vals_false.as_ptr()), 0.0);
    }

    #[test]
    fn compile_jit_mixed_chain_not_stress() {
        let args = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let entry = compile_jit("not x <= y < z and z >= y", &args).expect("mixed chain/not");
        let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

        // x<=y<z is true here, so `not ...` is false; false and true => false
        let vals_false = [1.0, 2.0, 3.0];
        assert_eq!(f(vals_false.as_ptr()), 0.0);

        // x<=y<z is false here, so `not ...` is true; true and true => true
        let vals_true = [3.0, 2.0, 2.0];
        assert_eq!(f(vals_true.as_ptr()), 1.0);
    }

#[test]
fn execute_jit_accepts_mixed_scalar_types() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let args = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let entry = compile_jit("x + y + z", &args).expect("compile mixed scalar test");
        let tuple = PyTuple::new(
            py,
            vec![1_i64.into_py(py), true.into_py(py), 2_i32.into_py(py)],
        );
        let result = execute_jit_func(py, &entry, tuple).expect("execute mixed scalars");
        let out: f64 = result.extract(py).unwrap();
        assert_eq!(out, 4.0);
    });
}

#[test]
fn execute_jit_vectorizes_non_f64_buffers() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let array_mod = py.import("array").unwrap();

        let args = vec!["x".to_string()];
        let mul_entry = compile_jit("x * 2", &args).expect("compile f32 buffer test");
        let f32_in = array_mod
            .getattr("array")
            .unwrap()
            .call1(("f", vec![1.5_f32, 2.0_f32, -3.0_f32]))
            .unwrap();
        let f32_tuple = PyTuple::new(py, &[f32_in]);
        let f32_out = execute_jit_func(py, &mul_entry, f32_tuple).expect("execute f32 buffer");
        let f32_vals: Vec<f64> = f32_out
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(f32_vals, vec![3.0, 4.0, -6.0]);

        let add_entry = compile_jit("x + 1", &args).expect("compile i32 buffer test");
        let i32_in = array_mod
            .getattr("array")
            .unwrap()
            .call1(("i", vec![1_i32, 2_i32, 7_i32]))
            .unwrap();
        let i32_tuple = PyTuple::new(py, &[i32_in]);
        let i32_out = execute_jit_func(py, &add_entry, i32_tuple).expect("execute i32 buffer");
        let i32_vals: Vec<f64> = i32_out
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(i32_vals, vec![2.0, 3.0, 8.0]);
    });
}

#[test]
fn execute_jit_handles_unaligned_f64_buffer_vectorized() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            "import struct\n\
buf=bytearray(1 + 8*3)\n\
struct.pack_into('ddd', buf, 1, 1.0, 2.0, 3.0)\n\
mv=memoryview(buf)[1:].cast('d')",
            None,
            Some(locals),
        )
        .unwrap();
        let mv = locals.get_item("mv").unwrap();

        let args = vec!["x".to_string()];
        let entry = compile_jit("x * 2", &args).expect("compile unaligned vectorized test");
        let tuple = PyTuple::new(py, &[mv]);
        let out_obj = execute_jit_func(py, &entry, tuple).expect("execute unaligned vectorized");
        let out: Vec<f64> = out_obj
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    });
}

#[test]
fn execute_jit_handles_unaligned_f64_buffer_packed_args() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            "import struct\n\
buf=bytearray(1 + 8*2)\n\
struct.pack_into('dd', buf, 1, 1.0, 2.0)\n\
mv=memoryview(buf)[1:].cast('d')",
            None,
            Some(locals),
        )
        .unwrap();
        let mv = locals.get_item("mv").unwrap();

        let args = vec!["a".to_string(), "b".to_string()];
        let entry = compile_jit("a + b", &args).expect("compile unaligned packed test");
        let tuple = PyTuple::new(py, &[mv]);
        let out_obj = execute_jit_func(py, &entry, tuple).expect("execute unaligned packed");
        let out: f64 = out_obj.extract(py).unwrap();
        assert_eq!(out, 3.0);
    });
}

#[test]
fn execute_jit_container_reductions_with_python_lists() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let locals = pyo3::types::PyDict::new(py);
        let list_obj = py.eval("[1.0, -2.0, 3.0, 0.0]", None, Some(locals)).unwrap();

        let sum_entry = compile_jit("sum(x_i * x_i for x_i in x)", &vec!["x".to_string()])
            .expect("sum container compile");
        let sum_tuple = PyTuple::new(py, &[list_obj]);
        let sum_obj = execute_jit_func(py, &sum_entry, sum_tuple).expect("sum container execute");
        let sum_val: f64 = sum_obj.extract(py).unwrap();
        assert_eq!(sum_val, 14.0);

        let any_entry = compile_jit("any(x_i > 2 for x_i in x if x_i != 0)", &vec!["x".to_string()])
            .expect("any container compile");
        let any_tuple = PyTuple::new(py, &[list_obj]);
        let any_obj = execute_jit_func(py, &any_entry, any_tuple).expect("any container execute");
        let any_val: f64 = any_obj.extract(py).unwrap();
        assert_eq!(any_val, 1.0);

        let all_entry = compile_jit("all(x_i >= -2 for x_i in x if x_i != 0)", &vec!["x".to_string()])
            .expect("all container compile");
        let all_tuple = PyTuple::new(py, &[list_obj]);
        let all_obj = execute_jit_func(py, &all_entry, all_tuple).expect("all container execute");
        let all_val: f64 = all_obj.extract(py).unwrap();
        assert_eq!(all_val, 1.0);
    });
}

#[test]
fn execute_jit_container_reductions_with_loop_control_intrinsics() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let locals = pyo3::types::PyDict::new(py);
        let list_obj = py.eval("[1.0, 2.0, 3.0, 4.0, 5.0]", None, Some(locals)).unwrap();

        let sum_break = compile_jit(
            "sum(break_if(x_i >= 4, x_i) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum break container compile");
        let sum_break_obj = execute_jit_func(py, &sum_break, PyTuple::new(py, &[list_obj]))
            .expect("sum break container execute");
        let sum_break_val: f64 = sum_break_obj.extract(py).unwrap();
        assert_eq!(sum_break_val, 6.0);

        let sum_continue = compile_jit(
            "sum(continue_if(x_i % 2 == 0, x_i) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum continue container compile");
        let sum_continue_obj = execute_jit_func(py, &sum_continue, PyTuple::new(py, &[list_obj]))
            .expect("sum continue container execute");
        let sum_continue_val: f64 = sum_continue_obj.extract(py).unwrap();
        assert_eq!(sum_continue_val, 9.0);

        let any_break = compile_jit(
            "any(break_if(x_i > 0, x_i > 10) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("any break container compile");
        let any_break_obj = execute_jit_func(py, &any_break, PyTuple::new(py, &[list_obj]))
            .expect("any break container execute");
        let any_break_val: f64 = any_break_obj.extract(py).unwrap();
        assert_eq!(any_break_val, 0.0);

        let all_continue = compile_jit(
            "all(continue_if(x_i < 4, x_i > 0) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("all continue container compile");
        let all_continue_obj = execute_jit_func(py, &all_continue, PyTuple::new(py, &[list_obj]))
            .expect("all continue container execute");
        let all_continue_val: f64 = all_continue_obj.extract(py).unwrap();
        assert_eq!(all_continue_val, 1.0);

        let sum_break_unless = compile_jit(
            "sum(break_unless(x_i < 4, x_i) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum break_unless container compile");
        let sum_break_unless_obj = execute_jit_func(py, &sum_break_unless, PyTuple::new(py, &[list_obj]))
            .expect("sum break_unless container execute");
        let sum_break_unless_val: f64 = sum_break_unless_obj.extract(py).unwrap();
        assert_eq!(sum_break_unless_val, 6.0);

        let sum_continue_unless = compile_jit(
            "sum(continue_unless(x_i % 2 == 1, x_i) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum continue_unless container compile");
        let sum_continue_unless_obj = execute_jit_func(py, &sum_continue_unless, PyTuple::new(py, &[list_obj]))
            .expect("sum continue_unless container execute");
        let sum_continue_unless_val: f64 = sum_continue_unless_obj.extract(py).unwrap();
        assert_eq!(sum_continue_unless_val, 9.0);

        let sum_break_when = compile_jit(
            "sum(break_when(x_i >= 4, x_i) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum break_when container compile");
        let sum_break_when_obj = execute_jit_func(py, &sum_break_when, PyTuple::new(py, &[list_obj]))
            .expect("sum break_when container execute");
        let sum_break_when_val: f64 = sum_break_when_obj.extract(py).unwrap();
        assert_eq!(sum_break_when_val, 6.0);

        let sum_continue_when = compile_jit(
            "sum(continue_when(x_i % 2 == 0, x_i) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum continue_when container compile");
        let sum_continue_when_obj = execute_jit_func(py, &sum_continue_when, PyTuple::new(py, &[list_obj]))
            .expect("sum continue_when container execute");
        let sum_continue_when_val: f64 = sum_continue_when_obj.extract(py).unwrap();
        assert_eq!(sum_continue_when_val, 9.0);

        let sum_break_on_nan = compile_jit(
            "sum(break_on_nan((x_i - x_i) / (x_i - x_i)) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum break_on_nan container compile");
        let sum_break_on_nan_obj = execute_jit_func(py, &sum_break_on_nan, PyTuple::new(py, &[list_obj]))
            .expect("sum break_on_nan container execute");
        let sum_break_on_nan_val: f64 = sum_break_on_nan_obj.extract(py).unwrap();
        assert_eq!(sum_break_on_nan_val, 0.0);

        let sum_continue_on_nan = compile_jit(
            "sum(continue_on_nan((x_i - x_i) / (x_i - x_i)) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("sum continue_on_nan container compile");
        let sum_continue_on_nan_obj = execute_jit_func(py, &sum_continue_on_nan, PyTuple::new(py, &[list_obj]))
            .expect("sum continue_on_nan container execute");
        let sum_continue_on_nan_val: f64 = sum_continue_on_nan_obj.extract(py).unwrap();
        assert_eq!(sum_continue_on_nan_val, 0.0);

        let if_else_container = compile_jit(
            "sum(if_else(x_i > 0, x_i, 0) for x_i in x)",
            &vec!["x".to_string()],
        )
        .expect("if_else container compile");
        let if_else_container_obj = execute_jit_func(py, &if_else_container, PyTuple::new(py, &[list_obj]))
            .expect("if_else container execute");
        let if_else_container_val: f64 = if_else_container_obj.extract(py).unwrap();
        assert_eq!(if_else_container_val, 15.0);
    });
}
