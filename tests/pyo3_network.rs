// tests/pyo3_network.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::time::Duration;

#[tokio::test]
async fn test_distributed_messaging() {
    let addr = "127.0.0.1:9999";

    // 1. Setup Node A (The Receiver)
    let (rt_a, pid_a, results): (PyObject, u64, PyObject) = Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt = rt_type.call0().unwrap();

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", results).unwrap();

        // Start listening
        rt.call_method1("listen", (addr,)).unwrap();

        // Spawn handler that records received messages
        py.run(r#"
def remote_handler(msg, results=results):
    results.append(msg.decode())
"#, None, Some(locals)).unwrap();

        let handler = locals.get_item("remote_handler").unwrap();
        let pid: u64 = rt.call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        (rt.into_py(py), pid, results.into_py(py))
    });

    // Wait for server to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 2. Setup Node B (The Sender) and transmit
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt_b = rt_type.call0().unwrap();

        let payload = pyo3::types::PyBytes::new(py, b"Hello from Node B");
        // Send to Node A's address and PID
        rt_b.call_method1("send_remote", (addr, pid_a, payload)).unwrap();
    });

    // 3. Verification: Did it arrive on Node A?
    let mut success = false;
    for _ in 0..20 { // Poll for up to 1 second
        tokio::time::sleep(Duration::from_millis(50)).await;
        success = Python::with_gil(|py| {
            let res: Vec<String> = results.extract(py).unwrap();
            res.contains(&"Hello from Node B".to_string())
        });
        if success { break; }
    }

    assert!(success, "Remote message never arrived at Node A");

    // Cleanup
    Python::with_gil(|py| {
        rt_a.call_method1(py, "stop", (pid_a,)).unwrap();
    });
}
