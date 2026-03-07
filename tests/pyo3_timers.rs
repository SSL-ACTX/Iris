#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::time::Duration;

#[tokio::test]
async fn test_send_after_delivers_message() {
    // create runtime, spawn actor and schedule timer while holding the GIL
    let (rt_obj, pid): (PyObject, u64) = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        // Spawn an observed handler to collect messages
        let pid: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // Schedule a message after 50ms
        let _timer_id: u64 = rt
            .call_method1("send_after", (pid, 50u64, PyBytes::new(py, b"delayed")))
            .unwrap()
            .extract()
            .unwrap();

        (rt.into_py(py), pid)
    });

    // allow the runtime to process (non-blocking)
    tokio::time::sleep(Duration::from_millis(120)).await;

    // now check messages with GIL again
    Python::with_gil(|py| {
        let rt = rt_obj.as_ref(py);
        let msgs: Vec<pyo3::PyObject> = rt
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap();

        assert!(msgs.len() >= 1, "expected at least one delivered message");
    });
}
