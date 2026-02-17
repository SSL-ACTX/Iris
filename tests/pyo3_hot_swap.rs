// tests/pyo3_hot_swap.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use std::time::Duration;

#[tokio::test]
async fn test_hot_swap_flow() {
    // 1. Setup: Create Runtime and shared results list
    let (rt, pid, results, handler_b): (PyObject, u64, PyObject, PyObject) = Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let runtime_type = module.as_ref(py).getattr("PyRuntime").expect("no PyRuntime type");
        let rt = runtime_type.call0().expect("construct PyRuntime");

        // A shared list to capture output from the actor
        let results = PyList::empty(py);

        // Define two different behaviors in the Python environment
        let locals = PyDict::new(py);
        locals.set_item("results", results).unwrap();

        // [FIX] We bind 'results=results' in the function definition.
        // This ensures the function object captures the specific 'results' list
        // at definition time, preventing NameErrors when running in background threads.
        py.run(r#"
def handler_a(msg, results=results):
    results.append(f"A:{msg.decode()}")

def handler_b(msg, results=results):
    results.append(f"B:{msg.decode()}")
"#, None, Some(locals)).unwrap();

        let handler_a = locals.get_item("handler_a").unwrap();
        let handler_b = locals.get_item("handler_b").unwrap();

        // Spawn the actor with Behavior A
        let pid: u64 = rt.call_method1("spawn_py_handler", (handler_a, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        (rt.into_py(py), pid, results.into_py(py), handler_b.into_py(py))
    });

    // 2. Execution: Send message to Behavior A
    Python::with_gil(|py| {
        let msg = pyo3::types::PyBytes::new(py, b"1");
        rt.call_method1(py, "send", (pid, msg)).unwrap();
    });

    // Allow async processing (Behavior A runs)
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 3. The Twist: Hot Swap to Behavior B
    Python::with_gil(|py| {
        // Call the hot_swap API
        rt.call_method1(py, "hot_swap", (pid, &handler_b)).unwrap();

        // Send message to the *same* PID, which should now use Behavior B
        let msg = pyo3::types::PyBytes::new(py, b"2");
        rt.call_method1(py, "send", (pid, msg)).unwrap();
    });

    // Allow async processing (Hot Swap + Behavior B runs)
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 4. Verification: Check the timeline of events
    Python::with_gil(|py| {
        let res: Vec<String> = results.extract(py).unwrap();

        // We expect [ "A:1", "B:2" ]
        assert_eq!(res.len(), 2, "Expected 2 messages, got {:?}", res);
        assert_eq!(res[0], "A:1");
        assert_eq!(res[1], "B:2");

        // Clean up
        rt.call_method1(py, "stop", (pid,)).unwrap();
    });

    // Final wait for stop
    tokio::time::sleep(Duration::from_millis(10)).await;
}
