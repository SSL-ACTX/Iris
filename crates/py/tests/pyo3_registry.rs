use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_registry_integration_py() {
    let results_py = Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();
        let results = PyList::empty(py);

        let locals = PyDict::new(py);
        locals.set_item("rt", &rt).unwrap();
        locals.set_item("results", &results).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"def named_handler(msg, results=results):
    results.append(msg.decode())
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        let handler = locals.get_item("named_handler").unwrap().unwrap();
        let pid: u64 = rt
            .call_method1("spawn", (handler,))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("register", ("my_service", pid)).unwrap();

        let resolved: u64 = rt
            .call_method1("resolve", ("my_service",))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(resolved, pid);

        rt.call_method1("send", (pid, b"ping")).unwrap();
        Ok::<Py<PyList>, PyErr>(results.unbind())
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(150)).await;

    Python::attach(|py| {
        let res: Vec<String> = results_py.bind(py).extract().unwrap();
        assert!(res.contains(&"ping".to_string()));
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_registry_whereis_and_unregister() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", &rt).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"import time

def dummy_service(mailbox):
    mailbox.recv()
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        let handler = locals.get_item("dummy_service").unwrap().unwrap();
        let pid: u64 = rt
            .call_method1("spawn", (handler,))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("register", ("test_service", pid)).unwrap();
        let found: Option<u64> = rt
            .call_method1("whereis", ("test_service",))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(found, Some(pid));

        rt.call_method1("unregister", ("test_service",)).unwrap();
        let not_found: Option<u64> = rt
            .call_method1("whereis", ("test_service",))
            .unwrap()
            .extract()
            .unwrap();
        assert!(not_found.is_none());

        rt.call_method1("stop", (pid,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_registry_ghost_lookups() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", &rt).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"import time

def logger_actor(mailbox):
    msg = mailbox.recv(timeout=1.0)
    if msg:
        print(f"Log: {msg}")

pid = rt.spawn(logger_actor)
rt.register("logger", pid)

time.sleep(0.2)
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"assert rt.resolve("ghost_service") is None
assert rt.whereis("ghost_service") is None

rt.unregister("ghost_service")
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_registry_none_lookups() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", &rt).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"assert rt.resolve("ghost_service") is None
assert rt.whereis("ghost_service") is None

rt.unregister("ghost_service")
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        Ok::<(), PyErr>(())
    })
    .unwrap();
}
