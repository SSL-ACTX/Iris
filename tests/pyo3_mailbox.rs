// tests/pyo3_mailbox.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mailbox_actor_basic_recv() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        // Instantiate the runtime directly via the class constructor exposed in the module
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
import time

# A pull-based actor that reads from its mailbox.
# NOTE: No longer 'async def'. This runs in a dedicated thread.
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
"#, Some(locals), Some(locals)).unwrap();

        let results: Vec<Vec<u8>> = locals.get_item("results").expect("results global not set (actor failed?)").extract().unwrap();
        assert_eq!(results[0], b"first");
        assert_eq!(results[1], b"second");
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mailbox_actor_selective_recv() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
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
"#, Some(locals), Some(locals)).unwrap();

        let results: Vec<Vec<u8>> = locals.get_item("results").expect("results global not set").extract().unwrap();
        assert_eq!(results[0], b"target"); // Selective
        assert_eq!(results[1], b"A");      // Deferred
        assert_eq!(results[2], b"B");      // Normal flow
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_mailbox_actor_timeout() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
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
"#, Some(locals), Some(locals)).unwrap();

        let result = locals.get_item("result").expect("result global not set");
        assert!(result.is_none());
    });
}
