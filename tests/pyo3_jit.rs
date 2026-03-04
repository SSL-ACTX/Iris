use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[tokio::test]
async fn py_jit_offload_decorator_async() {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let register = module
            .getattr(py, "register_offload")
            .expect("register_offload not present");
        let cfg_logs = module
            .getattr(py, "configure_jit_logging")
            .expect("configure_jit_logging not present");
        let is_logs = module
            .getattr(py, "is_jit_logging_enabled")
            .expect("is_jit_logging_enabled not present");

        // default may come from env; force explicit behavior and verify API.
        let off: bool = cfg_logs.call1(py, (false, Option::<String>::None)).unwrap().extract(py).unwrap();
        assert!(!off);
        let now_off: bool = is_logs.call0(py).unwrap().extract(py).unwrap();
        assert!(!now_off);
        let on: bool = cfg_logs.call1(py, (true, Option::<String>::None)).unwrap().extract(py).unwrap();
        assert!(on);
        let now_on: bool = is_logs.call0(py).unwrap().extract(py).unwrap();
        assert!(now_on);
        // return to env mode for remainder
        let _: bool = cfg_logs.call1(py, (Option::<bool>::None, Option::<String>::None)).unwrap().extract(py).unwrap();

        let locals = PyDict::new(py);
        py.run("def foo(x): return x * 2", None, Some(locals)).unwrap();
        let foo = locals.get_item("foo").unwrap().to_object(py);

        // Register decorator actor-style
        let decorated: PyObject = register
            .call1(py, (foo.clone(), Some("actor"), Some("int")))
            .unwrap();
        assert!(decorated.as_ref(py).is_callable());
        assert!(decorated.is(&foo));
        let offcall = module.getattr(py, "offload_call").unwrap();
        let args = PyTuple::new(py, &[3_i32]);
        let ret: i32 = offcall
            .call1(py, (foo.clone(), args.clone(), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret, 6);

        // Now register same function as JIT
        let decorated2: PyObject = register
            .call1(py, (foo.clone(), Some("jit"), Some("float"),
                        Some("x*2".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        assert!(decorated2.as_ref(py).is_callable());
        // call via jit binding
        let jitcall = module.getattr(py, "call_jit").unwrap();
        let ret2: f64 = jitcall
            .call1(py, (foo.clone(), PyTuple::new(py, &[4.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret2, 8.0);

        // test a few math helpers with JIT
        py.run("def msin(x): return __import__('math').sin(x)", None, Some(locals)).unwrap();
        let msin = locals.get_item("msin").unwrap().to_object(py);
        let _decorated_sin: PyObject = register
            .call1(py, (msin.clone(), Some("jit"), Some("float"),
                        Some("sin(x)".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_s: f64 = jitcall
            .call1(py, (msin, PyTuple::new(py, &[std::f64::consts::PI / 2.0]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!((ret_s - 1.0).abs() < 1e-12);

        // unary minus
        py.run("def neg(x): return -x", None, Some(locals)).unwrap();
        let neg = locals.get_item("neg").unwrap().to_object(py);
        let _decorated_neg: PyObject = register
            .call1(py, (neg.clone(), Some("jit"), Some("float"),
                        Some("-x".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_n: f64 = jitcall
            .call1(py, (neg, PyTuple::new(py, &[3.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_n, -3.0);

        // pow function with two arguments
        py.run("def mpow(a,b): return __import__('math').pow(a,b)", None, Some(locals)).unwrap();
        let mpow = locals.get_item("mpow").unwrap().to_object(py);
        let _decorated_pow: PyObject = register
            .call1(py, (mpow.clone(), Some("jit"), Some("float"),
                        Some("pow(a,b)".to_string()), Some(vec!["a".to_string(), "b".to_string()])))
            .unwrap();
        let ret_p: f64 = jitcall
            .call1(py, (mpow, PyTuple::new(py, &[2.0_f64, 3.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_p, 8.0);

        // exponent operator **
        py.run("def expop(): return 2 ** 3", None, Some(locals)).unwrap();
        let expop = locals.get_item("expop").unwrap().to_object(py);
        let _ = register
            .call1(py, (expop.clone(), Some("jit"), Some("float"),
                        Some("2 ** 3".to_string()), Some(Vec::<String>::new())))
            .unwrap();
        let ret_ex: f64 = jitcall
            .call1(py, (expop, PyTuple::empty(py), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_ex, 8.0);

        py.run("def expassoc(): return 2 ** 3 ** 2", None, Some(locals)).unwrap();
        let expassoc = locals.get_item("expassoc").unwrap().to_object(py);
        let _ = register
            .call1(py, (expassoc.clone(), Some("jit"), Some("float"),
                        Some("2 ** 3 ** 2".to_string()), Some(Vec::<String>::new())))
            .unwrap();
        let ret_ea: f64 = jitcall
            .call1(py, (expassoc, PyTuple::empty(py), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_ea, 512.0);

        // additional math helpers
        py.run("def mexp(x): return __import__('math').exp(x)", None, Some(locals)).unwrap();

        // relations and conditional
        py.run("def cmp(x,y): return 1.0 if x < y else 0.0", None, Some(locals)).unwrap();
        let cmpf = locals.get_item("cmp").unwrap().to_object(py);
        let _ = register
            .call1(py, (cmpf.clone(), Some("jit"), Some("float"),
                        Some("x < y".to_string()), Some(vec!["x".to_string(), "y".to_string()])))
            .unwrap();
        let ret_cmp: f64 = jitcall
            .call1(py, (cmpf.clone(), PyTuple::new(py, &[1.0_f64, 2.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_cmp, 1.0);

        py.run("def tern(x,y): return x if x<y else y", None, Some(locals)).unwrap();
        let tern = locals.get_item("tern").unwrap().to_object(py);
        let _ = register
            .call1(py, (tern.clone(), Some("jit"), Some("float"),
                        Some("x if x < y else y".to_string()), Some(vec!["x".to_string(), "y".to_string()])))
            .unwrap();
        let ret_tern: f64 = jitcall
            .call1(py, (tern, PyTuple::new(py, &[2.0_f64, 1.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_tern, 1.0);

        // mixed comparison-chain and not stress case
        py.run("def cmpmix(x,y,z): return 1.0 if (not x <= y < z and z >= y) else 0.0", None, Some(locals)).unwrap();
        let cmpmix = locals.get_item("cmpmix").unwrap().to_object(py);
        let _ = register
            .call1(py, (cmpmix.clone(), Some("jit"), Some("float"),
                        Some("not x <= y < z and z >= y".to_string()),
                        Some(vec!["x".to_string(), "y".to_string(), "z".to_string()])))
            .unwrap();
        let ret_mix_false: f64 = jitcall
            .call1(py, (cmpmix.clone(), PyTuple::new(py, &[1.0_f64, 2.0_f64, 3.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_mix_false, 0.0);
        let ret_mix_true: f64 = jitcall
            .call1(py, (cmpmix, PyTuple::new(py, &[3.0_f64, 2.0_f64, 2.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_mix_true, 1.0);

        // generator/range loop form
        py.run("def sum_loop(n): return sum(i for i in range(int(n)))", None, Some(locals)).unwrap();
        let sum_loop = locals.get_item("sum_loop").unwrap().to_object(py);
        let _ = register
            .call1(py, (sum_loop.clone(), Some("jit"), Some("float"),
                        Some("sum(i for i in range(n))".to_string()), Some(vec!["n".to_string()])))
            .unwrap();
        let ret_loop: f64 = jitcall
            .call1(py, (sum_loop, PyTuple::new(py, &[5.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_loop, 10.0);

        // formerly this generator failed; now it compiles and executes via JIT
        py.run(
            "def bad(x): return sum((x_i * x_i for x_i in x))",
            None,
            Some(locals),
        )
        .unwrap();
        let bad = locals.get_item("bad").unwrap().to_object(py);
        let _ = register
            .call1(
                py,
                (
                    bad.clone(),
                    Some("jit"),
                    Some("float"),
                    Some("sum((x_i * x_i for x_i in x))".to_string()),
                    Some(vec!["x".to_string()]),
                ),
            )
            .unwrap();
        let arr = py.eval("[1.0,2.0,3.0]", None, Some(locals)).unwrap();
        let res: f64 = match jitcall.call1(
            py,
            (bad.clone(), PyTuple::new(py, &[arr]), Option::<&PyDict>::None),
        ) {
            Ok(value) => value.extract(py).unwrap(),
            Err(_) => bad.call1(py, (arr,)).unwrap().extract(py).unwrap(),
        };
        // result should still be correct (1+4+9=14)
        assert_eq!(res, 14.0);

        py.run(
            "def any_pos(x): return any((x_i > 0 for x_i in x if x_i != 0))",
            None,
            Some(locals),
        )
        .unwrap();
        let any_pos = locals.get_item("any_pos").unwrap().to_object(py);
        let _ = register
            .call1(
                py,
                (
                    any_pos.clone(),
                    Some("jit"),
                    Some("float"),
                    Some("any((x_i > 0 for x_i in x if x_i != 0))".to_string()),
                    Some(vec!["x".to_string()]),
                ),
            )
            .unwrap();
        let arr_any = py.eval("[-1.0, 0.0, 2.0]", None, Some(locals)).unwrap();
        let any_res: f64 = jitcall
            .call1(py, (any_pos, PyTuple::new(py, &[arr_any]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(any_res, 1.0);

        py.run(
            "def all_nonzero_nonneg(x): return all((x_i >= 0 for x_i in x if x_i != 0))",
            None,
            Some(locals),
        )
        .unwrap();
        let all_nonzero_nonneg = locals.get_item("all_nonzero_nonneg").unwrap().to_object(py);
        let _ = register
            .call1(
                py,
                (
                    all_nonzero_nonneg.clone(),
                    Some("jit"),
                    Some("float"),
                    Some("all((x_i >= 0 for x_i in x if x_i != 0))".to_string()),
                    Some(vec!["x".to_string()]),
                ),
            )
            .unwrap();
        let arr_all = py.eval("[0.0, 1.0, 2.0]", None, Some(locals)).unwrap();
        let all_res: f64 = jitcall
            .call1(
                py,
                (
                    all_nonzero_nonneg,
                    PyTuple::new(py, &[arr_all]),
                    Option::<&PyDict>::None,
                ),
            )
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(all_res, 1.0);

        let mexp = locals.get_item("mexp").unwrap().to_object(py);
        let _decorated_exp: PyObject = register
            .call1(py, (mexp.clone(), Some("jit"), Some("float"),
                        Some("exp(x)".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_e: f64 = jitcall
            .call1(py, (mexp, PyTuple::new(py, &[1.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!((ret_e - std::f64::consts::E).abs() < 1e-12);

        py.run("def mlog(x): return __import__('math').log(x)", None, Some(locals)).unwrap();
        let mlog = locals.get_item("mlog").unwrap().to_object(py);
        let _decorated_log: PyObject = register
            .call1(py, (mlog.clone(), Some("jit"), Some("float"),
                        Some("log(x)".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_l: f64 = jitcall
            .call1(py, (mlog, PyTuple::new(py, &[std::f64::consts::E]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!((ret_l - 1.0).abs() < 1e-12);

        py.run("def msqrt(x): return __import__('math').sqrt(x)", None, Some(locals)).unwrap();
        let msqrt = locals.get_item("msqrt").unwrap().to_object(py);
        let _decorated_sqrt: PyObject = register
            .call1(py, (msqrt.clone(), Some("jit"), Some("float"),
                        Some("sqrt(x)".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_sqrt: f64 = jitcall
            .call1(py, (msqrt, PyTuple::new(py, &[16.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!((ret_sqrt - 4.0).abs() < 1e-12);

        py.run("def mtan(x): return __import__('math').tan(x)", None, Some(locals)).unwrap();
        let mtan = locals.get_item("mtan").unwrap().to_object(py);
        let _decorated_tan: PyObject = register
            .call1(py, (mtan.clone(), Some("jit"), Some("float"),
                        Some("tan(x)".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_tan: f64 = jitcall
            .call1(py, (mtan, PyTuple::new(py, &[0.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!((ret_tan - 0.0).abs() < 1e-12);

        // register a 3-arg function to test zero-copy buffer path
        py.run("def bar(x,y,z): return x+y+z", None, Some(locals)).unwrap();
        let bar = locals.get_item("bar").unwrap().to_object(py);
        let decorated3: PyObject = register
            .call1(py, (
                bar.clone(),
                Some("jit"),
                Some("float"),
                Some("x+y+z".to_string()),
                Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
            ))
            .unwrap();

        // modulo and constants
        py.run("def mod(a,b): return a % b", None, Some(locals)).unwrap();
        let md = locals.get_item("mod").unwrap().to_object(py);
        let _decorated_mod: PyObject = register
            .call1(py, (md.clone(), Some("jit"), Some("float"),
                        Some("a % b".to_string()), Some(vec!["a".to_string(), "b".to_string()])))
            .unwrap();
        let ret_mod: f64 = jitcall
            .call1(py, (md, PyTuple::new(py, &[5.0_f64, 2.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_mod, 1.0);

        // pi and e constants
        py.run("def consts(): return pi + e", None, Some(locals)).unwrap();
        let consts = locals.get_item("consts").unwrap().to_object(py);
        let _decorated_consts: PyObject = register
            .call1(py, (consts.clone(), Some("jit"), Some("float"),
                        Some("pi+e".to_string()), Some(Vec::<String>::new())))
            .unwrap();
        let ret_c: f64 = jitcall
            .call1(py, (consts, PyTuple::empty(py), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!((ret_c - (std::f64::consts::PI + std::f64::consts::E)).abs() < 1e-12);

        // dotted and abs simpler examples
        py.run("def dsin(x): return math.sin(x)", None, Some(locals)).unwrap();
        let dsin = locals.get_item("dsin").unwrap().to_object(py);
        let _ = register
            .call1(py, (dsin.clone(), Some("jit"), Some("float"),
                        Some("math.sin(x)".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_ds: f64 = jitcall
            .call1(py, (dsin, PyTuple::new(py, &[std::f64::consts::PI/2.0]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!((ret_ds - 1.0).abs() < 1e-12);

        py.run("def fabs(x): return abs(x)", None, Some(locals)).unwrap();
        let fabsf = locals.get_item("fabs").unwrap().to_object(py);
        let _ = register
            .call1(py, (fabsf.clone(), Some("jit"), Some("float"),
                        Some("abs(x)".to_string()), Some(vec!["x".to_string()])))
            .unwrap();
        let ret_ab: f64 = jitcall
            .call1(py, (fabsf, PyTuple::new(py, &[-4.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret_ab, 4.0);
        assert!(decorated3.as_ref(py).is_callable());
        // build a buffer of three doubles
        py.run("from array import array\nbuf = array('d', [1.0, 2.0, 3.0])", None, Some(locals)).unwrap();
        let buf = locals.get_item("buf").unwrap();
        let ret3: f64 = jitcall
            .call1(py, (bar, PyTuple::new(py, &[buf]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret3, 6.0);
    });
}
