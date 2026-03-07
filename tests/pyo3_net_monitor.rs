// tests/pyo3_monitor.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use std::time::Duration;

#[tokio::test]
async fn test_remote_monitoring_failure() {
    let addr = "127.0.0.1:9998";

    // 1. Setup Node A (The Target)
    let (rt_a, pid_a) = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module
            .as_ref(py)
            .getattr("PyRuntime")
            .unwrap()
            .call0()
            .unwrap();

        rt.call_method1("listen", (addr,)).unwrap();

        py.run("def target(msg): pass", None, None).unwrap();
        let handler = py.eval("target", None, None).unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        // register under a well-known name so the other node can look it up
        rt.call_method1("register", ("target", pid)).unwrap();

        (rt.into_py(py), pid)
    });

    // 2. Setup Node B (The Guardian)
    let mut proxy_pid: u64 = 0;
    let rt_b = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module
            .as_ref(py)
            .getattr("PyRuntime")
            .unwrap()
            .call0()
            .unwrap();

        // Node B first resolves the PID; this returns a local proxy.
        let resolved: Option<u64> = rt
            .call_method1("resolve_remote", (addr, "target"))
            .unwrap()
            .extract()
            .unwrap();
        assert!(resolved.is_some());
        proxy_pid = resolved.unwrap();

        // Tell Node B to monitor the proxy; this will in turn watch the
        // real remote actor and shut down the proxy if the node disappears.
        rt.call_method1("monitor_remote", (addr, proxy_pid)).unwrap();
        rt.into_py(py)
    });

    // 3. Simulate Failure: Kill Node A's actor and stop listening
    Python::with_gil(|py| {
        rt_a.call_method1(py, "stop", (pid_a,)).unwrap();
    });

    // Give time for the network task in Node B to realize the connection is gone/refused
    tokio::time::sleep(Duration::from_millis(200)).await;

    // 4. Verification: proxy pid should no longer be alive
    Python::with_gil(|py| {
        let alive: bool = rt_b
            .call_method1(py, "is_alive", (proxy_pid,))
            .unwrap()
            .extract(py)
            .unwrap();
        assert!(!alive, "proxy should have been shut down after remote failure");
    });
}
