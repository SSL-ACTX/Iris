// tests/pyo3_registry.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_registry_lifecycle() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
import time

# 1. Spawn a simple mailbox actor
def dummy_service(mailbox):
    # Just keep the channel open so the PID is valid
    mailbox.recv() 

# Spawn
pid = rt.spawn_with_mailbox(dummy_service, 100)
assert pid > 0

# 2. Register "my_service"
rt.register("my_service", pid)

# 3. Resolve it back
found_pid = rt.resolve("my_service")
assert found_pid == pid, f"Resolve failed: expected {pid}, got {found_pid}"

# 4. Test 'whereis' alias
alias_pid = rt.whereis("my_service")
assert alias_pid == pid, f"Whereis failed: expected {pid}, got {alias_pid}"

# 5. Overwrite registration (if supported, DashMap usually allows overwrite)
# Re-registering same name with same PID shouldn't error
rt.register("my_service", pid)

# 6. Unregister
rt.unregister("my_service")

# 7. Verify it is gone
gone = rt.resolve("my_service")
assert gone is None, f"Expected None after unregister, got {gone}"

# Clean up
rt.stop(pid)
"#, Some(locals), Some(locals)).unwrap();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_communication_via_name() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt).unwrap();
        locals.set_item("__builtins__", py.import("builtins").unwrap()).unwrap();

        py.run(r#"
import time

# Actor that receives a message and writes it to a global
def logger_actor(mailbox):
    msg = mailbox.recv(timeout=1.0)
    global log_result
    log_result = msg

# 1. Spawn
logger_pid = rt.spawn_with_mailbox(logger_actor, 100)

# 2. Register globally
rt.register("central_logger", logger_pid)

# 3. Resolve and Send
# (Simulates another actor finding this service)
target = rt.resolve("central_logger")

if target:
    rt.send(target, b"log_this_data")
else:
    raise Exception("Could not find central_logger")

# Wait for actor to process
time.sleep(0.2)
"#, Some(locals), Some(locals)).unwrap();

        // Verify the actor received the message via the looked-up PID
        let result: Vec<u8> = locals.get_item("log_result")
            .expect("log_result not set - actor did not receive message")
            .extract()
            .unwrap();
        
        assert_eq!(result, b"log_this_data");
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_register_non_existent_returns_none() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).unwrap();
        let rt = module.getattr(py, "PyRuntime").unwrap().call0(py).unwrap();
        let locals = PyDict::new(py);
        locals.set_item("rt", rt).unwrap();
        
        py.run(r#"
# resolving a name that was never registered should return None
assert rt.resolve("ghost_service") is None
assert rt.whereis("ghost_service") is None

# unregistering a non-existent name should be safe (no-op)
rt.unregister("ghost_service")
"#, Some(locals), Some(locals)).unwrap();
    });
}
