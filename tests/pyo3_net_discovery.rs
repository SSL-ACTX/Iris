// tests/pyo3_discovery.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::time::Duration;

#[tokio::test]
async fn test_remote_name_discovery() {
    let addr = "127.0.0.1:9090";

    // 1. Setup Node A (The Provider)
    let (rt_a, pid_a, results): (PyObject, u64, PyObject) = Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        rt.call_method1("listen", (addr,)).unwrap();

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", results).unwrap();

        py.run(r#"
def auth_handler(msg, results=results):
    results.append(f"Auth:{msg.decode()}")
"#, None, Some(locals)).unwrap();

        let handler = locals.get_item("auth_handler").unwrap();
        let pid: u64 = rt.call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap().extract().unwrap();

        // Register the name on Node A
        rt.call_method1("register", ("auth-service", pid)).unwrap();

        (rt.into_py(py), pid, results.into_py(py))
    });

    // Wait for server A to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 2. Setup Node B (The Client)
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt_b = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        // Node B resolves the name on Node A's address
        let resolved_pid: Option<u64> = rt_b.call_method1("resolve_remote", (addr, "auth-service"))
            .unwrap().extract().unwrap();

        assert_eq!(resolved_pid, Some(pid_a), "Node B failed to resolve Node A's service name");

        // Node B sends a message to the resolved PID
        let payload = pyo3::types::PyBytes::new(py, b"login_request");
        rt_b.call_method1("send_remote", (addr, resolved_pid.unwrap(), payload)).unwrap();
    });

    // 3. Verification: Did Node A receive the message?
    let mut success = false;
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        success = Python::with_gil(|py| {
            let res: Vec<String> = results.extract(py).unwrap();
            res.contains(&"Auth:login_request".to_string())
        });
        if success { break; }
    }

    assert!(success, "Remote message via discovered name never arrived");

    // Cleanup
    Python::with_gil(|py| {
        rt_a.call_method1(py, "stop", (pid_a,)).unwrap();
    });
}
