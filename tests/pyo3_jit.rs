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
            .call1(py, (foo, PyTuple::new(py, &[4.0_f64]), Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();
        assert_eq!(ret2, 8.0);

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
