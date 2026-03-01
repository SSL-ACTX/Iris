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

        // Register decorator
        let decorated: PyObject = register
            .call1(py, (foo.clone(), Some("actor"), Some("int")))
            .unwrap();

        assert!(decorated.as_ref(py).is_callable());
        assert!(decorated.is(&foo));

        // Test offload_call
        let offcall = module.getattr(py, "offload_call").unwrap();
        let args = PyTuple::new(py, &[3_i32]);
        let ret: i32 = offcall
            .call1(py, (foo, args, Option::<&PyDict>::None))
            .unwrap()
            .extract(py)
            .unwrap();

        assert_eq!(ret, 6);
    });
}
