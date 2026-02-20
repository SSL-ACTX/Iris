// tests/pyo3_restart_path_supervisors.rs
#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use std::time::Duration;

#[tokio::test]
async fn test_path_supervisor_restart_one() {
    Python::with_gil(|py| {
        let module = myrmidon::py::make_module(py).expect("make_module");
        let rt_type = module.as_ref(py).getattr("PyRuntime").unwrap();
        let rt = rt_type.call0().unwrap();

        // spawn an observed actor and register it under a hierarchical path
        let pid: u64 = rt
            .call_method1("spawn_with_path_observed", (10usize, "/svc/restart/one"))
            .unwrap()
            .extract()
            .unwrap();

        // create a path supervisor for the prefix
        rt.call_method1("create_path_supervisor", ("/svc/restart",)).unwrap();

        // Python factory that spawns a new observed actor under same path
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            r#"
def factory():
    import myrmidon
    rt = myrmidon.Runtime()
    return rt.spawn_with_path_observed(10, "/svc/restart/one")
"#,
            None,
            Some(locals),
        )
        .unwrap();

        let factory = locals.get_item("factory").unwrap();

        // Attach factory-based supervision to the path (expected API)
        rt.call_method1(
            "path_supervise_with_factory",
            ("/svc/restart", pid, factory, "restartone"),
        )
        .unwrap();

        // Kill the original pid
        rt.call_method1("stop", (pid,)).unwrap();

        // After a short delay the supervisor should restart the child
        // and the children list should contain a pid (possibly new)
        let children: Vec<u64> = rt
            .call_method1("path_supervisor_children", ("/svc/restart",))
            .unwrap()
            .extract()
            .unwrap();

        assert!(!children.is_empty());
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
}
