#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::test]
async fn test_telemetry_and_introspection() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // 1. Check initial metrics
    Python::with_gil(|py| {
        let metrics: HashMap<String, u64> = rt_py
            .as_ref(py)
            .call_method0("get_metrics")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(metrics.get("actor_count"), Some(&0));
    });

    // 2. Spawn actors and check metrics
    let pid: u64 = Python::with_gil(|py| {
        let p = rt_py
            .as_ref(py)
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        let metrics: HashMap<String, u64> = rt_py
            .as_ref(py)
            .call_method0("get_metrics")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(metrics.get("actor_count"), Some(&1));
        p
    });

    // 3. Send messages and check metrics
    Python::with_gil(|py| {
        for _ in 0..5 {
            rt_py
                .as_ref(py)
                .call_method1("send", (pid, PyBytes::new(py, b"hello")))
                .unwrap();
        }
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    Python::with_gil(|py| {
        let metrics: HashMap<String, u64> = rt_py
            .as_ref(py)
            .call_method0("get_metrics")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(metrics.get("messages_sent"), Some(&5));
        // observed handler processes messages immediately
        assert_eq!(metrics.get("messages_received"), Some(&5));
    });

    // 4. Check actor_info
    Python::with_gil(|py| {
        let info: HashMap<String, String> = rt_py
            .as_ref(py)
            .call_method1("actor_info", (pid,))
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(info.get("pid"), Some(&pid.to_string()));
        assert_eq!(info.get("is_proxy"), Some(&"false".to_string()));
    });

    // 5. Check list_actors
    Python::with_gil(|py| {
        let actors: Vec<u64> = rt_py
            .as_ref(py)
            .call_method0("list_actors")
            .unwrap()
            .extract()
            .unwrap();

        assert!(actors.contains(&pid));
        assert_eq!(actors.len(), 1);
    });

    // 6. Stop actor and check metrics
    Python::with_gil(|py| {
        rt_py.as_ref(py).call_method1("stop", (pid,)).unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    Python::with_gil(|py| {
        let metrics: HashMap<String, u64> = rt_py
            .as_ref(py)
            .call_method0("get_metrics")
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(metrics.get("actor_count"), Some(&0));

        let actors: Vec<u64> = rt_py
            .as_ref(py)
            .call_method0("list_actors")
            .unwrap()
            .extract()
            .unwrap();
        assert!(actors.is_empty());
    });
}
