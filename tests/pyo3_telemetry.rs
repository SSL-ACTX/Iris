#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::test]
async fn test_telemetry_and_introspection() {
    let rt_py = Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module.getattr("PyRuntime").unwrap();
        let rt = runtime_type.call0().unwrap();
        Ok::<Py<PyAny>, PyErr>(rt.unbind())
    })
    .unwrap();

    // 1. Check initial metrics
    Python::attach(|py| {
        let metrics: HashMap<String, u64> = rt_py
            .bind(py)
            .call_method1("metrics", ())
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(metrics.get("active_actors").cloned().unwrap_or(0), 0);
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // 2. Spawn actors and check metrics
    let pid: u64 = Python::attach(|py| {
        let p = rt_py
            .bind(py)
            .call_method1(
                "spawn",
                (py.eval(pyo3::ffi::c_str!("lambda m: None"), None, None)
                    .unwrap(),),
            )
            .unwrap()
            .extract()
            .unwrap();

        let metrics: HashMap<String, u64> = rt_py
            .bind(py)
            .call_method1("metrics", ())
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(metrics.get("active_actors").cloned().unwrap_or(0), 1);
        Ok::<u64, PyErr>(p)
    })
    .unwrap();

    // 3. Send messages and check metrics
    Python::attach(|py| {
        for _ in 0..5 {
            rt_py.bind(py).call_method1("send", (pid, b"test")).unwrap();
        }
        Ok::<(), PyErr>(())
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    Python::attach(|py| {
        let metrics: HashMap<String, u64> = rt_py
            .bind(py)
            .call_method1("metrics", ())
            .unwrap()
            .extract()
            .unwrap();
        // At least 5 messages processed
        assert!(metrics.get("messages_processed").cloned().unwrap_or(0) >= 5);
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // 4. Check actor_info
    Python::attach(|py| {
        let info: HashMap<String, String> = rt_py
            .bind(py)
            .call_method1("actor_info", (pid,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(info.contains_key("pid"));
        assert_eq!(info.get("pid").unwrap(), &pid.to_string());
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // 5. Check list_actors
    Python::attach(|py| {
        let actors: Vec<u64> = rt_py
            .bind(py)
            .call_method1("list_actors", ())
            .unwrap()
            .extract()
            .unwrap();
        assert!(actors.contains(&pid));
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // 6. Stop actor and check metrics
    Python::attach(|py| {
        rt_py.bind(py).call_method1("stop", (pid,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    Python::attach(|py| {
        let metrics: HashMap<String, u64> = rt_py
            .bind(py)
            .call_method1("metrics", ())
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(metrics.get("active_actors").cloned().unwrap_or(0), 0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}
