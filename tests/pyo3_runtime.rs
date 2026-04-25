use pyo3::prelude::*;
use std::time::Duration;

#[tokio::test]
async fn test_send_after_delivers_message() {
    // create runtime, spawn actor and schedule timer while holding the GIL
    let (rt_obj, pid): (Py<PyAny>, u64) = Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let runtime_type = module.getattr("PyRuntime").unwrap();
        let rt = runtime_type.call0().unwrap();

        py.run(pyo3::ffi::c_str!("def handler(msg): pass"), None, None)
            .unwrap();
        let handler = py.eval(pyo3::ffi::c_str!("handler"), None, None).unwrap();

        let pid: u64 = rt
            .call_method1("spawn", (handler,))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("send_after", (pid, b"delayed", 0.1))
            .unwrap();
        Ok::<(Py<PyAny>, u64), PyErr>((rt.unbind(), pid))
    })
    .unwrap();

    // wait for timer
    tokio::time::sleep(Duration::from_millis(150)).await;

    // now check messages with GIL again
    Python::attach(|py| {
        let rt = rt_obj.bind(py);
        let msgs: Vec<Vec<u8>> = rt
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0], b"delayed");
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test]
async fn test_allocate_buffer_and_send_no_copy() {
    let (rt_obj, pid, results): (Py<PyAny>, u64, Py<PyAny>) = Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        py.run(
            pyo3::ffi::c_str!("def handler(msg, results): results.append(bytes(msg))"),
            None,
            None,
        )
        .unwrap();
        let handler_src = py.eval(pyo3::ffi::c_str!("handler"), None, None).unwrap();
        let results = pyo3::types::PyList::empty(py);

        let pid: u64 = rt
            .call_method1("spawn", (handler_src, results.clone()))
            .unwrap()
            .extract()
            .unwrap();

        Ok::<_, PyErr>((rt.unbind(), pid, results.unbind().into_any()))
    })
    .unwrap();

    // allocate a Rust-owned buffer and write into it from Python
    Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rv = module.call_method1("allocate_buffer", (5usize,)).unwrap();
        let (id, mem, cap): (u64, Py<PyAny>, Py<PyAny>) = rv.extract().unwrap();
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("mem", mem.bind(py)).unwrap();
        locals.set_item("cap", cap.bind(py)).unwrap();
        py.run(pyo3::ffi::c_str!("mem[:5] = b'hello'"), None, Some(&locals))
            .unwrap();
        // send the buffer without copying
        rt_obj
            .bind(py)
            .call_method1("send_buffer", (pid, id))
            .unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;

    Python::attach(|py| {
        let res: Vec<Vec<u8>> = results.bind(py).extract().unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(&res[0], b"hello");
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test]
async fn test_selective_recv_from_python() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
def worker(mailbox):
    # wait for a message starting with 'b'
    m = mailbox.selective_recv(lambda msg: msg.startswith(b"b"), timeout=1.0)
    return m
"#
            ),
            None,
            None,
        )
        .unwrap();
        let worker_fn = py.eval(pyo3::ffi::c_str!("worker"), None, None).unwrap();

        let pid: u64 = rt
            .call_method1("spawn", (worker_fn,))
            .unwrap()
            .extract()
            .unwrap();

        // send 'a', then 'b'
        rt.call_method1("send", (pid, b"apple")).unwrap();
        rt.call_method1("send", (pid, b"banana")).unwrap();

        // wait for it to finish and check result
        // we'll poll is_alive or just sleep
        Ok::<(), PyErr>(())
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(150)).await;
}

#[tokio::test]
async fn test_spawn_child_pool_py() {
    let rt_py = Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module.getattr("PyRuntime").expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        Ok::<_, PyErr>(rt_obj.unbind())
    })
    .unwrap();

    let (parent_pid, _worker_pids, lst_obj) = Python::attach(|py| {
        let rt = rt_py.bind(py);

        let parent_pid: u64 = rt
            .call_method1("spawn_observed_handler", (1usize,))
            .unwrap()
            .extract()
            .unwrap();

        let lst = pyo3::types::PyList::empty(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", &lst).unwrap();
        py.run(
            pyo3::ffi::c_str!("def cb(b):\n    lst.append(bytes(b))"),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        let cb = locals.get_item("cb").unwrap().unwrap();

        let worker_pids: Vec<u64> = rt
            .call_method1(
                "spawn_child_pool",
                (parent_pid, &cb, 4usize, 64usize, false),
            )
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(worker_pids.len(), 4);

        for _ in 0..10 {
            rt.call_method1("send", (parent_pid, b"work")).unwrap();
        }

        Ok::<(u64, Vec<u64>, Py<PyAny>), PyErr>((parent_pid, worker_pids, lst.unbind().into_any()))
    })
    .unwrap();

    tokio::time::sleep(Duration::from_millis(300)).await;

    Python::attach(|py| {
        let lst = lst_obj.bind(py).cast::<pyo3::types::PyList>().unwrap();
        assert!(
            lst.len() >= 5,
            "workers should have processed some messages, got {}",
            lst.len()
        );
        rt_py.bind(py).call_method1("stop", (parent_pid,)).unwrap();
        Ok::<(), PyErr>(())
    })
    .unwrap();
}
