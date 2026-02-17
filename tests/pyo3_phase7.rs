// tests/pyo3_phase7.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Duration;

// Fix: Enable multi-thread runtime so block_in_place works in src/py.rs
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_phase7_features() {
    let addr = "127.0.0.1:9096";

    // 1. Setup Node A (The Provider)
    let (rt_a, pid_a) = Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        rt.call_method1("listen", (addr,)).unwrap();

        py.run("def handler(msg): pass", None, None).unwrap();
        let handler = py.eval("handler", None, None).unwrap();

        let pid: u64 = rt.call_method1("spawn_py_handler", (handler, 10usize))
            .unwrap().extract().unwrap();

        rt.call_method1("register", ("discovery-service", pid)).unwrap();
        (rt.into_py(py), pid)
    });

    // Wait for listener
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 2. Test Membrane Hardening: is_node_up (Synchronous call from inside tokio)
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt_b = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        let up: bool = rt_b.call_method1("is_node_up", (addr,)).unwrap().extract().unwrap();
        assert!(up, "is_node_up failed to detect live node");

        let down: bool = rt_b.call_method1("is_node_up", ("127.0.0.1:1234",)).unwrap().extract().unwrap();
        assert!(!down, "is_node_up reported dead node as up");
    });

    // 3. Test Async Service Discovery: resolve_remote_py (Awaitable)
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt_b = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", rt_b).unwrap();
        locals.set_item("addr", addr).unwrap();

        // Use asyncio to await the resolve_remote_py future
        py.run(r#"
import asyncio

async def run_discovery(rt, addr):
    # This calls the resolve_remote_py method which returns an Awaitable
    fut = rt.resolve_remote_py(addr, "discovery-service")
    return await fut

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_discovery(rt, addr))
"#, None, Some(locals)).unwrap();

        let resolved_pid: Option<u64> = locals.get_item("result").unwrap().extract().unwrap();
        assert_eq!(resolved_pid, Some(pid_a), "Async discovery failed to resolve PID");
    });

    // 4. Test Structured System Messages: EXIT mapping
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        // Spawn an observed handler to watch another actor
        let observer_pid: u64 = rt.call_method1("spawn_observed_handler", (10usize,)).unwrap().extract().unwrap();

        py.run("def target(msg): pass", None, None).unwrap();
        let handler = py.eval("target", None, None).unwrap();
        let target_pid: u64 = rt.call_method1("spawn_py_handler", (handler, 10usize)).unwrap().extract().unwrap();

        // Link them: in Myrmidon, linking causes an EXIT message to be sent when one dies
        rt.call_method1("watch", (target_pid, "one")).unwrap();

        // Force the target to stop
        rt.call_method1("stop", (target_pid,)).unwrap();
    });

    // Allow time for exit signal propagation
    tokio::time::sleep(Duration::from_millis(100)).await;

    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.as_ref(py).getattr("PyRuntime").unwrap().call0().unwrap();

        // We need to find the observer_pid again or pass it out. For simplicity in this test,
        // we check the child_pids of the runtime which should contain our observer.
        let pids: Vec<u64> = rt.call_method0("child_pids").unwrap().extract().unwrap();
        for pid in pids {
            let msgs: Vec<PyObject> = rt.call_method1("get_messages", (pid,)).unwrap().extract().unwrap();
            for msg in msgs {
                // Check if the message is an instance of PySystemMessage
                let type_name: String = msg.as_ref(py).getattr("type_name").unwrap().extract().unwrap();
                if type_name == "EXIT" {
                    let target: Option<u64> = msg.as_ref(py).getattr("target_pid").unwrap().extract().unwrap();
                    assert!(target.is_some(), "EXIT message missing target_pid");
                    return; // Success
                }
            }
        }
    });

    // Cleanup
    Python::with_gil(|py| {
        rt_a.call_method1(py, "stop", (pid_a,)).unwrap();
    });
}
