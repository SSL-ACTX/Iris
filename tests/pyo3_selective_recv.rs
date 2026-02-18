// tests/pyo3_selective_recv.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyDict;

// Needs multi-thread tokio runtime so Python blocking helpers work.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_observed_py() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        // Instantiate the runtime directly via the class constructor exposed in the module
        let rt_class = module.getattr(py, "PyRuntime").unwrap();
        let rt = rt_class.call0(py).unwrap();

        // Spawn an observed handler which stores incoming messages for inspection.
        let observer_pid: u64 = rt.call_method1(py, "spawn_observed_handler", (10usize,)).unwrap().extract(py).unwrap();

        // Send messages: m1, target, m3
        rt.call_method1(py, "send", (observer_pid, pyo3::types::PyBytes::new(py, b"m1"))).unwrap();
        rt.call_method1(py, "send", (observer_pid, pyo3::types::PyBytes::new(py, b"target"))).unwrap();
        rt.call_method1(py, "send", (observer_pid, pyo3::types::PyBytes::new(py, b"m3"))).unwrap();

        // Run an asyncio loop to await the selective receive
        let locals = PyDict::new(py);
        // FIX: Clone rt here so it isn't moved, allowing us to use it again below.
        locals.set_item("rt", rt.clone()).unwrap(); 
        locals.set_item("pid", observer_pid).unwrap();
        // Provide builtins so the executed code can define functions and use globals
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
import asyncio

def matcher(msg):
    return isinstance(msg, (bytes, bytearray)) and msg == b"target"

async def run_selective(rt, pid):
    # No timeout specified here
    fut = rt.selective_recv_observed_py(pid, matcher)
    return await fut

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_selective(rt, pid))
"#, Some(locals), Some(locals)).unwrap();

        // Verify result equals b"target"
        let result: Vec<u8> = locals.get_item("result").unwrap().extract().unwrap();
        assert_eq!(result, b"target".to_vec());

        // Verify remaining messages are m1 and m3 in order
        // This call was failing previously because rt had been moved
        let msgs: Vec<PyObject> = rt.call_method1(py, "get_messages", (observer_pid,)).unwrap().extract(py).unwrap();
        assert_eq!(msgs.len(), 2);
        let first: Vec<u8> = msgs[0].as_ref(py).extract().unwrap();
        let second: Vec<u8> = msgs[1].as_ref(py).extract().unwrap();
        assert_eq!(first, b"m1".to_vec());
        assert_eq!(second, b"m3".to_vec());
    });
}


// Test matching system EXIT messages produced when a watched actor stops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_system_message() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt_class = module.getattr(py, "PyRuntime").unwrap();
        let rt = rt_class.call0(py).unwrap();

        // Spawn an observed handler which stores incoming messages for inspection.
        let observer_pid: u64 = rt.call_method1(py, "spawn_observed_handler", (10usize,)).unwrap().extract(py).unwrap();

        // Send a HotSwap system message to the observer to test system-message matching.
        rt.call_method1(py, "hot_swap", (observer_pid, py.None())).unwrap();

        // Now await a HOT_SWAP system message using selective_recv (async path)
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("pid", observer_pid).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
import asyncio

def matcher(msg):
    # System messages are delivered as PySystemMessage types
    try:
        t = getattr(msg, "type_name")
        return t == "HOT_SWAP"
    except Exception:
        return False

async def run_selective(rt, pid):
    fut = rt.selective_recv_observed_py(pid, matcher)
    return await fut

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_selective(rt, pid))
"#, Some(locals), Some(locals)).unwrap();

        // Verify result is a PySystemMessage with type_name == 'HOT_SWAP'
        let result = locals.get_item("result").unwrap();
        let type_name: String = result.getattr("type_name").unwrap().extract().unwrap();
        assert_eq!(type_name, "HOT_SWAP");
    });
}

// Test timeout functionality: ensure it returns None if the message never arrives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_timeout() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt_class = module.getattr(py, "PyRuntime").unwrap();
        let rt = rt_class.call0(py).unwrap();

        let observer_pid: u64 = rt.call_method1(py, "spawn_observed_handler", (10usize,)).unwrap().extract(py).unwrap();

        // We do NOT send any messages.

        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("pid", observer_pid).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
import asyncio

def matcher(msg):
    return msg == b"never_arrives"

async def run_with_timeout(rt, pid):
    # Wait for 0.1 seconds, then timeout
    return await rt.selective_recv_observed_py(pid, matcher, 0.1)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_with_timeout(rt, pid))
"#, Some(locals), Some(locals)).unwrap();

        let result = locals.get_item("result").unwrap();
        assert!(result.is_none(), "Expected None result after timeout");
    });
}

// Test timeout functionality: ensure it returns the message if it exists (before timeout).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_success_with_timeout() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt_class = module.getattr(py, "PyRuntime").unwrap();
        let rt = rt_class.call0(py).unwrap();

        let observer_pid: u64 = rt.call_method1(py, "spawn_observed_handler", (10usize,)).unwrap().extract(py).unwrap();

        // Send the message immediately
        rt.call_method1(py, "send", (observer_pid, pyo3::types::PyBytes::new(py, b"exists"))).unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("pid", observer_pid).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
import asyncio

def matcher(msg):
    return msg == b"exists"

async def run_with_timeout(rt, pid):
    # Wait for 1.0 second (plenty of time), should return immediately
    return await rt.selective_recv_observed_py(pid, matcher, 1.0)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_with_timeout(rt, pid))
"#, Some(locals), Some(locals)).unwrap();

        let result: Vec<u8> = locals.get_item("result").unwrap().extract().unwrap();
        assert_eq!(result, b"exists".to_vec());
    });
}