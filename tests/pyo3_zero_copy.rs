#![cfg(feature = "pyo3")]

use pyo3::prelude::*;

#[tokio::test]
async fn py_zero_copy_send() {
    let rt_py = Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let runtime_type = module.as_ref(py).getattr("PyRuntime").expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // spawn observed handler
    let pid: u64 = Python::with_gil(|py| {
        rt_py.as_ref(py).call_method1("spawn_observed_handler", (1usize,)).unwrap().extract().unwrap()
    });

    // allocate a Rust-owned buffer and write into it from Python
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let rv = module.as_ref(py).call_method1("allocate_buffer", (5usize,)).unwrap();
        let (id, mem, cap): (u64, pyo3::PyObject, pyo3::PyObject) = rv.extract().unwrap();
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("mem", mem.as_ref(py)).unwrap();
        locals.set_item("cap", cap.as_ref(py)).unwrap();
        py.run("mem[:5] = b'hello'", None, Some(locals)).unwrap();
        // send the buffer without copying
        rt_py.as_ref(py).call_method1("send_buffer", (pid, id)).unwrap();
    });

    // allow the tokio tasks to run
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let msgs: Vec<Vec<u8>> = Python::with_gil(|py| {
        rt_py.as_ref(py).call_method1("get_messages", (pid,)).unwrap().extract().unwrap()
    });

    assert_eq!(msgs.len(), 1);
    assert_eq!(&msgs[0], b"hello");
}
