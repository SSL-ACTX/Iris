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
