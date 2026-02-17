#![cfg(feature = "pyo3")]

use pyo3::prelude::*;

#[tokio::test]
async fn py_runtime_spawn_and_send() {
    // create a single PyRuntime instance and keep it alive across await points
    let rt_py = Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let runtime_type = module.as_ref(py).getattr("PyRuntime").expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // spawn an observed handler and send a message (call into the same PyRuntime)
    let pid: u64 = Python::with_gil(|py| {
        rt_py.as_ref(py).call_method1("spawn_observed_handler", (1usize,)).unwrap().extract().unwrap()
    });

    let sent: bool = Python::with_gil(|py| {
        rt_py.as_ref(py).call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"hello"))).unwrap().extract().unwrap()
    });
    assert!(sent, "send failed");

    // allow the tokio tasks to run
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // inspect recorded messages for the same runtime/pid
    let msgs: Vec<Vec<u8>> = Python::with_gil(|py| {
        rt_py.as_ref(py).call_method1("get_messages", (pid,)).unwrap().extract().unwrap()
    });

    assert_eq!(msgs.len(), 1);
    assert_eq!(&msgs[0], b"hello");

    // --- NEW: spawn a Python-backed handler that appends to a Python list ---
    let lst_obj: pyo3::PyObject = Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let runtime_type = module.as_ref(py).getattr("PyRuntime").expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");

        // create a Python list and a callback that appends bytes to it
        let lst = pyo3::types::PyList::empty(py);
        let lst_obj: pyo3::PyObject = lst.into_py(py);
        let locals = pyo3::types::PyDict::new(py);
        // build a factory that returns a callback which closes over `lst`
        py.run("def make_cb(lst):\n    def cb(b): lst.append(b)\n    return cb\n", None, Some(locals)).unwrap();
        let make_cb = locals.get_item("make_cb").unwrap();
        let cb: pyo3::PyObject = make_cb.call1((lst_obj.as_ref(py),)).unwrap().into();

        // spawn a Python handler and send a message
        let pid: u64 = rt_obj.call_method1("spawn_py_handler", (cb, 1usize)).unwrap().extract().unwrap();
        rt_obj.call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"pycall"))).unwrap();

        lst_obj
    });

    // allow the tokio tasks to run (outside the GIL)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // verify Python list got the bytes
    Python::with_gil(|py| {
        let lst = lst_obj.as_ref(py).downcast::<pyo3::types::PyList>().unwrap();
        let got: Vec<&[u8]> = lst.extract().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], b"pycall");
    });

    // --- NEW: register a Python factory with the supervisor and validate ---
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let runtime_type = module.as_ref(py).getattr("PyRuntime").expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");

        // spawn observed handler to supervise
        let pid: u64 = rt_obj.call_method1("spawn_observed_handler", (1usize,)).unwrap().extract().unwrap();

        // create a factory that calls back into this same runtime instance
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("rt", rt_obj.into_py(py)).unwrap();
        py.run("def factory(rt=rt):\n    return rt.spawn_observed_handler(1)", None, Some(locals)).unwrap();
        let factory: pyo3::PyObject = locals.get_item("factory").unwrap().into();

        rt_obj.call_method1("supervise_with_factory", (pid, factory, "RestartOne")).unwrap();
        let count: usize = rt_obj.call_method0("children_count").unwrap().extract().unwrap();
        assert_eq!(count, 1);

        let pids: Vec<u64> = rt_obj.call_method0("child_pids").unwrap().extract().unwrap();
        assert_eq!(pids.len(), 1);
        assert_eq!(pids[0], pid);
    });
}
