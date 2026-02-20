// tests/pyo3_restart_all_path_supervisors.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use std::time::Duration;

#[tokio::test]
async fn test_path_supervisor_restart_all() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt = rt_type.call0().unwrap();

        // spawn two observed actors under distinct child paths but same prefix
        let pid1: u64 = rt
            .call_method1("spawn_with_path_observed", (10usize, "/svc/restart/all/one"))
            .unwrap()
            .extract()
            .unwrap();

        let pid2: u64 = rt
            .call_method1("spawn_with_path_observed", (10usize, "/svc/restart/all/two"))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("create_path_supervisor", ("/svc/restart/all",)).unwrap();

        // factories that spawn new observed actors under the same child paths
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("rt", rt).unwrap();
        py.run(
            r#"
def f1():
    return rt.spawn_with_path_observed(10, "/svc/restart/all/one")

def f2():
    return rt.spawn_with_path_observed(10, "/svc/restart/all/two")
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();

        let f1 = locals.get_item("f1").unwrap();
        let f2 = locals.get_item("f2").unwrap();

        rt.call_method1(
            "path_supervise_with_factory",
            ("/svc/restart/all", pid1, f1, "restartall"),
        )
        .unwrap();

        rt.call_method1(
            "path_supervise_with_factory",
            ("/svc/restart/all", pid2, f2, "restartall"),
        )
        .unwrap();

        // kill one pid — RestartAll should restart both children
        rt.call_method1("stop", (pid1,)).unwrap();

        // Wait for supervisor to perform restart attempts and then inspect children
        let mut attempts = 0;
        let mut seen_ok = false;
        while attempts < 40 {
            let children: Vec<u64> = rt
                .call_method1("path_supervisor_children", ("/svc/restart/all",))
                .unwrap()
                .extract()
                .unwrap();
            if children.len() == 2 && (!children.contains(&pid1) || !children.contains(&pid2)) {
                seen_ok = true;
                break;
            }
            attempts += 1;

            // Release the GIL while sleeping so the background supervisor task
            // can acquire it and run the Python factories
            py.allow_threads(|| {
                std::thread::sleep(std::time::Duration::from_millis(50));
            });
        }

        assert!(seen_ok, "supervisor did not restart children within timeout");
    });

    tokio::time::sleep(Duration::from_millis(300)).await;
}
