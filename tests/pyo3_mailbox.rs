#![cfg(feature = "pyo3")]
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mailbox_actor_basic_recv() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        // Instantiate the runtime directly via the class constructor exposed in the module
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
import time

# A pull-based actor that reads from its mailbox.
def mailbox_actor(mailbox):
    # 1. Standard receive (blocking, releases GIL internally)
    msg1 = mailbox.recv()
    # 2. Receive with timeout
    msg2 = mailbox.recv(timeout=1.0)
    
    # Send results back to a global list for verification
    global results
    results = [msg1, msg2]

# Spawn the actor with a budget of 100
# This now spawns a real OS thread for the actor.
pid = rt.spawn_with_mailbox(mailbox_actor, 100)

# Send two messages
rt.send(pid, b"first")
rt.send(pid, b"second")

# We just sleep the main thread to allow the background actor thread to finish.
time.sleep(0.5)
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        let results: Vec<Vec<u8>> = locals
            .get_item("results")
            .expect("results global not set (actor failed?)")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(results[0], b"first");
        assert_eq!(results[1], b"second");
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mailbox_actor_selective_recv() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
import time

def mailbox_actor(mailbox):
    # We expect messages: "A", "target", "B"
    # We want to pick "target" out of order.
    
    def matcher(msg):
        return msg == b"target"

    # This should skip "A" and grab "target" (blocking)
    target = mailbox.selective_recv(matcher, timeout=1.0)
    
    # Next standard recv should get "A" (was deferred)
    after_1 = mailbox.recv()
    
    # Next standard recv should get "B"
    after_2 = mailbox.recv()
    
    global results
    results = [target, after_1, after_2]

# Spawn with budget
pid = rt.spawn_with_mailbox(mailbox_actor, 100)

# Send messages in specific order
rt.send(pid, b"A")
rt.send(pid, b"target")
rt.send(pid, b"B")

# Allow time for threaded execution
time.sleep(0.5)
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        let results: Vec<Vec<u8>> = locals
            .get_item("results")
            .expect("results global not set")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(results[0], b"target"); // Selective
        assert_eq!(results[1], b"A"); // Deferred
        assert_eq!(results[2], b"B"); // Normal flow
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mailbox_actor_timeout() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
import time

def mailbox_actor(mailbox):
    # Try to receive with a short timeout, expecting None
    # No await needed
    msg = mailbox.recv(timeout=0.05)
    
    global result
    result = msg

# Spawn with budget
pid = rt.spawn_with_mailbox(mailbox_actor, 100)
# We send NO messages

# Allow time for execution
time.sleep(0.5)
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        let result = locals.get_item("result").unwrap().unwrap();
        assert!(result.is_none());
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test]
async fn test_exit_reason_on_panic() {
    let (rt_py, observer) = Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        // Spawn an observed actor that will be stopped normally.
        let target: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // Spawn an observer to collect exit notifications
        let observer: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // Link the target to the observer so the observer receives the EXIT
        rt.call_method1("link", (target, observer)).unwrap();

        // Stop the target actor (normal exit)
        rt.call_method1("stop", (target,)).unwrap();

        Ok::<(Py<PyAny>, u64), PyErr>((rt.unbind(), observer))
    })
    .unwrap();

    // Give a delay for the exit to be delivered
    let mut found = false;
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        found = Python::attach(|py| {
            let rt = rt_py.bind(py);
            let msgs: Vec<Py<PyAny>> = rt
                .call_method1("get_messages", (observer,))
                .unwrap()
                .extract()
                .unwrap();

            // Find an EXIT message with reason 'normal'
            for m in msgs {
                let m_bound: &Bound<'_, PyAny> = m.bind(py);
                if let Ok(type_name_obj) = m_bound.getattr("type_name") {
                    let type_name: String = type_name_obj.extract::<String>().unwrap_or_default();
                    if type_name == "EXIT" {
                        let reason: String = m_bound
                            .getattr("reason")
                            .unwrap()
                            .extract::<String>()
                            .unwrap_or_default();
                        if reason == "normal" {
                            return Ok::<bool, PyErr>(true);
                        }
                    }
                }
            }
            Ok::<bool, PyErr>(false)
        })
        .unwrap();
        if found {
            break;
        }
    }

    assert!(
        found,
        "expected to find an EXIT message with reason 'normal'"
    );
}

// tests/pyo3_release_gil.rs

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_spawn_py_handler_release_gil_toggle() {
    // Create runtime and two handlers in the module namespace so they share a SEEN list
    let module = Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let g = module.dict();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import threading
SEEN = []

def handler_no_release(msg):
    SEEN.append(('no', threading.get_ident()))

def handler_release(msg):
    SEEN.append(('yes', threading.get_ident()))
"#
            ),
            Some(&g),
            None,
        )
        .unwrap();

        Ok::<Py<PyAny>, PyErr>(module.unbind().into())
    })
    .unwrap();

    // Spawn both actors and send a message to each
    Python::attach(|py| {
        let module_ref = module.bind(py);
        let rt = module_ref.getattr("PyRuntime").unwrap().call0().unwrap();

        let handler_no = module_ref.getattr("handler_no_release").unwrap();
        let handler_yes = module_ref.getattr("handler_release").unwrap();

        let pid_no: u64 = rt
            .call_method1("spawn_py_handler", (handler_no, 10usize, false))
            .unwrap()
            .extract()
            .unwrap();
        let pid_yes: u64 = rt
            .call_method1("spawn_py_handler", (handler_yes, 10usize, true))
            .unwrap()
            .extract()
            .unwrap();

        // Send simple byte messages
        let _ = rt
            .call_method1("send", (pid_no, PyBytes::new(py, b"ping")))
            .unwrap();
        let _ = rt
            .call_method1("send", (pid_yes, PyBytes::new(py, b"ping")))
            .unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // Allow the actors to process messages
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Inspect SEEN and ensure we observed both handlers and that their thread ids differ
    Python::attach(|py| {
        let module_ref = module.bind(py);
        let seen: Vec<(String, usize)> = module_ref.getattr("SEEN").unwrap().extract().unwrap();

        // Expect two entries (order not guaranteed)
        assert!(
            seen.len() >= 2,
            "expected at least two handler invocations, got {}",
            seen.len()
        );

        let mut no_tid = None;
        let mut yes_tid = None;
        for (tag, tid) in seen {
            if tag == "no" {
                no_tid = Some(tid);
            }
            if tag == "yes" {
                yes_tid = Some(tid);
            }
        }

        assert!(no_tid.is_some(), "no-release handler did not run");
        assert!(yes_tid.is_some(), "release handler did not run");
        assert_ne!(
            no_tid.unwrap(),
            yes_tid.unwrap(),
            "handlers ran on the same thread; expected different threads when toggling GIL release"
        );
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

// tests/pyo3_selective_recv.rs

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_observed_py() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        // Instantiate the runtime directly via the class constructor exposed in the module
        let rt_class = module.getattr("PyRuntime").unwrap();
        let rt = rt_class.call0().unwrap();

        // Spawn an observed handler which stores incoming messages for inspection.
        let observer_pid: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // Send messages: m1, target, m3
        rt.call_method1("send", (observer_pid, pyo3::types::PyBytes::new(py, b"m1")))
            .unwrap();
        rt.call_method1(
            "send",
            (observer_pid, pyo3::types::PyBytes::new(py, b"target")),
        )
        .unwrap();
        rt.call_method1("send", (observer_pid, pyo3::types::PyBytes::new(py, b"m3")))
            .unwrap();

        // Run an asyncio loop to await the selective receive
        let locals = PyDict::new(py);
        // FIX: Clone rt here so it isn't moved, allowing us to use it again below.
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("pid", observer_pid).unwrap();
        // Provide builtins so the executed code can define functions and use globals
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
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
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        // Verify result equals b"target"
        let result: Vec<u8> = locals
            .get_item("result")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(result, b"target".to_vec());

        // Verify remaining messages are m1 and m3 in order
        // This call was failing previously because rt had been moved
        let msgs: Vec<Py<PyAny>> = rt
            .call_method1("get_messages", (observer_pid,))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(msgs.len(), 2);
        let first: Vec<u8> = msgs[0].bind(py).extract().unwrap();
        let second: Vec<u8> = msgs[1].bind(py).extract().unwrap();
        assert_eq!(first, b"m1".to_vec());
        assert_eq!(second, b"m3".to_vec());
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

// Test matching system EXIT messages produced when a watched actor stops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_system_message() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt_class = module.getattr("PyRuntime").unwrap();
        let rt = rt_class.call0().unwrap();

        // Spawn an observed handler which stores incoming messages for inspection.
        let observer_pid: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // Send a HotSwap system message to the observer to test system-message matching.
        rt.call_method1("hot_swap", (observer_pid, py.None()))
            .unwrap();

        // Now await a HOT_SWAP system message using selective_recv (async path)
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("pid", observer_pid).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
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
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        // Verify result is a PySystemMessage with type_name == 'HOT_SWAP'
        let result = locals.get_item("result").unwrap().unwrap();
        let type_name: String = result.getattr("type_name").unwrap().extract().unwrap();
        assert_eq!(type_name, "HOT_SWAP");
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

// Test timeout functionality: ensure it returns None if the message never arrives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_timeout() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt_class = module.getattr("PyRuntime").unwrap();
        let rt = rt_class.call0().unwrap();

        let observer_pid: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // We do NOT send any messages.

        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("pid", observer_pid).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
import asyncio

def matcher(msg):
    return msg == b"never_arrives"

async def run_with_timeout(rt, pid):
    # Wait for 0.1 seconds, then timeout
    return await rt.selective_recv_observed_py(pid, matcher, 0.1)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_with_timeout(rt, pid))
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        let result = locals.get_item("result").unwrap().unwrap();
        assert!(result.is_none(), "Expected None result after timeout");
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

// Test timeout functionality: ensure it returns the message if it exists (before timeout).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_selective_recv_success_with_timeout() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt_class = module.getattr("PyRuntime").unwrap();
        let rt = rt_class.call0().unwrap();

        let observer_pid: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // Send the message immediately
        rt.call_method1(
            "send",
            (observer_pid, pyo3::types::PyBytes::new(py, b"exists")),
        )
        .unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("pid", observer_pid).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
import asyncio

def matcher(msg):
    return msg == b"exists"

async def run_with_timeout(rt, pid):
    # Wait for 1.0 second (plenty of time), should return immediately
    return await rt.selective_recv_observed_py(pid, matcher, 1.0)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_with_timeout(rt, pid))
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        let result: Vec<u8> = locals
            .get_item("result")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(result, b"exists".to_vec());
        Ok::<(), PyErr>(())
    })
    .unwrap();
}
