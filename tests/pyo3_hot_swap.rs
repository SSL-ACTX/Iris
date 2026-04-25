// tests/pyo3_hot_swap.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Duration;

#[tokio::test]
async fn test_hot_swap_flow() {
    // 1. Setup: Create Runtime and shared results list
    let (rt, pid, results, handler_b): (Py<PyAny>, u64, Py<PyAny>, Py<PyAny>) =
        Python::attach(|py| {
            let module = iris::py::make_module(py).expect("make_module");
            let runtime_type = module.getattr("PyRuntime").expect("no PyRuntime type");
            let rt = runtime_type.call0().expect("construct PyRuntime");

            // A shared list to capture output from the actor
            let results = PyList::empty(py);

            // Define two different behaviors in the Python environment
            let locals = PyDict::new(py);
            locals.set_item("results", &results).unwrap();

            // [FIX] We bind 'results=results' in the function definition.
            // This ensures the function object captures the specific 'results' list
            // at definition time, preventing NameErrors when running in background threads.
            py.run(
                pyo3::ffi::c_str!(
                    r#"
def handler_a(msg, results=results):
    results.append(f"A:{msg.decode()}")

def handler_b(msg, results=results):
    results.append(f"B:{msg.decode()}")
"#
                ),
                None,
                Some(&locals),
            )
            .unwrap();

            let handler_a = locals.get_item("handler_a").unwrap().unwrap();
            let handler_b = locals.get_item("handler_b").unwrap().unwrap();

            // Spawn the actor with Behavior A
            let pid: u64 = rt
                .call_method1("spawn_py_handler", (handler_a, 10usize))
                .unwrap()
                .extract()
                .unwrap();

            Ok::<(Py<PyAny>, u64, Py<PyAny>, Py<PyAny>), PyErr>((
                rt.unbind(),
                pid,
                results.unbind().into(),
                handler_b.unbind(),
            ))
        })
        .unwrap();

    // 2. Execution: Send message to Behavior A
    Python::attach(|py| {
        let msg = pyo3::types::PyBytes::new(py, b"1");
        rt.bind(py).call_method1("send", (pid, msg)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // Allow async processing (Behavior A runs)
    tokio::time::sleep(Duration::from_millis(50)).await;

    // sanity check: ensure the first message was handled by A before swapping
    Python::attach(|py| {
        let res: Vec<String> = results.bind(py).extract().unwrap();
        assert!(
            res.first().map(|s| s == "A:1").unwrap_or(false),
            "first message was not processed by A before hot swap: {:?}",
            res
        );
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // 3. The Twist: Hot Swap to Behavior B
    Python::attach(|py| {
        // Call the hot_swap API
        rt.bind(py)
            .call_method1("hot_swap", (pid, &handler_b))
            .unwrap();

        // Send message to the *same* PID, which should now use Behavior B
        let msg = pyo3::types::PyBytes::new(py, b"2");
        rt.bind(py).call_method1("send", (pid, msg)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // Allow async processing (Hot Swap + Behavior B runs)
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 4. Verification: Check the timeline of events
    Python::attach(|py| {
        let res: Vec<String> = results.bind(py).extract().unwrap();

        // We expect [ "A:1", "B:2" ]
        assert_eq!(res.len(), 2, "Expected 2 messages, got {:?}", res);
        assert_eq!(res[0], "A:1");
        assert_eq!(res[1], "B:2");

        // Clean up
        rt.bind(py).call_method1("stop", (pid,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // Final wait for stop
    tokio::time::sleep(Duration::from_millis(10)).await;
}

#[tokio::test]
async fn test_behavior_versioning_and_rollback() {
    let (rt, pid, results, handler_b, handler_c): (
        Py<PyAny>,
        u64,
        Py<PyAny>,
        Py<PyAny>,
        Py<PyAny>,
    ) = Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module.getattr("PyRuntime").expect("no PyRuntime type");
        let rt = runtime_type.call0().expect("construct PyRuntime");

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", &results).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
def handler_a(msg, results=results):
    results.append(f"A:{msg.decode()}")

def handler_b(msg, results=results):
    results.append(f"B:{msg.decode()}")

def handler_c(msg, results=results):
    results.append(f"C:{msg.decode()}")
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        let handler_a = locals.get_item("handler_a").unwrap().unwrap();
        let handler_b = locals.get_item("handler_b").unwrap().unwrap();
        let handler_c = locals.get_item("handler_c").unwrap().unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler", (handler_a, 10usize))
            .unwrap()
            .extract()
            .unwrap();

        Ok::<(Py<PyAny>, u64, Py<PyAny>, Py<PyAny>, Py<PyAny>), PyErr>((
            rt.unbind(),
            pid,
            results.unbind().into(),
            handler_b.unbind(),
            handler_c.unbind(),
        ))
    })
    .unwrap();

    Python::attach(|py| {
        let ver: u64 = rt
            .bind(py)
            .call_method1("behavior_version", (pid,))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ver, 1);

        rt.bind(py)
            .call_method1("hot_swap", (pid, &handler_b))
            .unwrap();
        rt.bind(py)
            .call_method1("hot_swap", (pid, &handler_c))
            .unwrap();

        let ver_after: u64 = rt
            .bind(py)
            .call_method1("behavior_version", (pid,))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ver_after, 3);

        let rolled: u64 = rt
            .bind(py)
            .call_method1("rollback_behavior", (pid, 1usize))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(rolled, 2);

        let msg = pyo3::types::PyBytes::new(py, b"x");
        rt.bind(py).call_method1("send", (pid, msg)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(60)).await;

    Python::attach(|py| {
        let res: Vec<String> = results.bind(py).extract().unwrap();
        assert!(
            res.iter().any(|s| s == "B:x"),
            "rollback should reactivate B behavior: {:?}",
            res
        );
        rt.bind(py).call_method1("stop", (pid,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // Final wait for stop
    tokio::time::sleep(Duration::from_millis(10)).await;
}
