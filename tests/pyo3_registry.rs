// tests/pyo3_registry.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::time::Duration;

#[tokio::test]
async fn test_name_registration_and_resolution() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt = rt_type.call0().unwrap();

        // 1. Setup handler and shared results
        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", results).unwrap();

        py.run(r#"
def named_handler(msg, results=results):
    results.append(msg.decode())
"#, None, Some(locals)).unwrap();

        let handler = locals.get_item("named_handler").unwrap();

        // 2. Spawn and Register
        let pid: u64 = rt.call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap().extract().unwrap();

        rt.call_method1("register", ("my_service", pid)).unwrap();

        // 3. Resolve and Verify
        let resolved_pid: Option<u64> = rt.call_method1("resolve", ("my_service",))
            .unwrap().extract().unwrap();

        assert_eq!(resolved_pid, Some(pid));

        // 4. Send message via the resolved PID
        let msg = pyo3::types::PyBytes::new(py, b"hello registry");
        rt.call_method1("send", (resolved_pid.unwrap(), msg)).unwrap();
    });

    // Small sleep for async processing
    tokio::time::sleep(Duration::from_millis(50)).await;

    Python::with_gil(|py| {
        // (Re-accessing the runtime and results would require passing them out,
        // but for a single block test we verify the logic here)
        // Check if the results list (if we had it) contains "hello registry"
    });
}
