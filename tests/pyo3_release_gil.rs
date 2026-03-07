// tests/pyo3_release_gil.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_spawn_py_handler_release_gil_toggle() {
    // Create runtime and two handlers in the module namespace so they share a SEEN list
    let module = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let g = module.as_ref(py).dict();
        py.run(
            r#"
import threading
SEEN = []

def handler_no_release(msg):
    SEEN.append(('no', threading.get_ident()))

def handler_release(msg):
    SEEN.append(('yes', threading.get_ident()))
"#,
            Some(g),
            None,
        )
        .unwrap();

        module.into_py(py)
    });

    // Spawn both actors and send a message to each
    Python::with_gil(|py| {
        let module_ref = module.as_ref(py);
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
    });

    // Allow the actors to process messages
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Inspect SEEN and ensure we observed both handlers and that their thread ids differ
    Python::with_gil(|py| {
        let module_ref = module.as_ref(py);
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
    });
}
