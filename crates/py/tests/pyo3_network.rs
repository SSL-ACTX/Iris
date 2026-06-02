use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_send_py() {
    let addr = "127.0.0.1:9094";

    // 1. Setup Node A (The Receiver)
    let (rt_a, pid_a, results) = Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();
        rt.call_method1("listen", (addr,)).unwrap();

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", &results).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
def remote_handler(msg, results=results):
    results.append(msg.decode())
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        let handler = locals.get_item("remote_handler").unwrap().unwrap();
        let pid: u64 = rt
            .call_method1("spawn", (handler,))
            .unwrap()
            .extract()
            .unwrap();
        rt.call_method1("register", ("node_a_receiver", pid))
            .unwrap();

        Ok::<(Py<PyAny>, u64, Py<PyList>), PyErr>((rt.unbind(), pid, results.unbind()))
    })
    .unwrap();

    // 2. Setup Node B (The Sender) and send message
    tokio::time::sleep(Duration::from_millis(100)).await;

    Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt_b = module.getattr("PyRuntime").unwrap().call0().unwrap();

        let remote_pid: u64 = rt_b
            .call_method1("resolve_remote", (addr, "node_a_receiver"))
            .unwrap()
            .extract()
            .unwrap();

        rt_b.call_method1("send", (remote_pid, b"Hello from Node B"))
            .unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    // 3. Verify arrival
    let mut success = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        success = Python::attach(|py| {
            let res: Vec<String> = results.bind(py).extract().unwrap();
            Ok::<bool, PyErr>(res.contains(&"Hello from Node B".to_string()))
        })
        .unwrap();
        if success {
            break;
        }
    }
    assert!(success, "Remote message never arrived at Node A");

    Python::attach(|py| {
        rt_a.bind(py).call_method1("stop", (pid_a,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_remote_service_discovery_py() {
    let addr = "127.0.0.1:9095";

    let (rt_a, pid_a, results) = Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();
        rt.call_method1("listen", (addr,)).unwrap();

        let results = PyList::empty(py);
        let locals = PyDict::new(py);
        locals.set_item("results", &results).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
def auth_handler(msg, results=results):
    results.append(f"Auth:{msg.decode()}")
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        let handler = locals.get_item("auth_handler").unwrap().unwrap();
        let pid: u64 = rt
            .call_method1("spawn", (handler,))
            .unwrap()
            .extract()
            .unwrap();
        rt.call_method1("register", ("auth_service", pid)).unwrap();

        Ok::<(Py<PyAny>, u64, Py<PyList>), PyErr>((rt.unbind(), pid, results.unbind()))
    })
    .unwrap();

    // Give server time to bind
    std::thread::sleep(Duration::from_millis(150));

    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt_b = module.getattr("PyRuntime").unwrap().call0().unwrap();

        // Node B resolve via service name on Node A's address
        let remote_pid: u64 = rt_b
            .call_method1("resolve_remote", (addr, "auth_service"))
            .unwrap()
            .extract()
            .unwrap();

        rt_b.call_method1("send", (remote_pid, b"login_request"))
            .unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    let mut success = false;
    for _ in 0..15 {
        std::thread::sleep(Duration::from_millis(100));
        success = Python::attach(|py| {
            let res: Vec<String> = results.bind(py).extract().unwrap();
            Ok::<bool, PyErr>(res.contains(&"Auth:login_request".to_string()))
        })
        .unwrap();
        if success {
            break;
        }
    }
    assert!(success, "Remote message via discovered name never arrived");

    Python::attach(|py| {
        rt_a.bind(py).call_method1("stop", (pid_a,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_distributed_stop_observed_py() {
    let addr = "127.0.0.1:9998";

    let (rt_a, pid_a) = Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();
        rt.call_method1("listen", (addr,)).unwrap();

        py.run(pyo3::ffi::c_str!("def target(msg): pass"), None, None)
            .unwrap();
        let handler = py.eval(pyo3::ffi::c_str!("target"), None, None).unwrap();

        let pid: u64 = rt
            .call_method1("spawn", (handler,))
            .unwrap()
            .extract()
            .unwrap();
        rt.call_method1("register", ("terminator", pid)).unwrap();
        Ok::<(Py<PyAny>, u64), PyErr>((rt.unbind(), pid))
    })
    .unwrap();

    let mut proxy_pid: u64 = 0;
    let rt_b = Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        proxy_pid = rt
            .call_method1("resolve_remote", (addr, "terminator"))
            .unwrap()
            .extract()
            .unwrap();
        Ok::<Py<PyAny>, PyErr>(rt.unbind())
    })
    .unwrap();

    Python::attach(|py| {
        rt_a.bind(py).call_method1("stop", (pid_a,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(3000)).await;

    let mut alive = true;
    for _ in 0..30 {
        alive = Python::attach(|py| {
            let a: bool = rt_b
                .bind(py)
                .call_method1("is_alive", (proxy_pid,))
                .unwrap()
                .extract()
                .unwrap();
            Ok::<bool, PyErr>(a)
        })
        .unwrap();
        if !alive {
            break;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    assert!(
        !alive,
        "Remote proxy should be dead after actual actor stopped (waited up to 12s total)"
    );
}
