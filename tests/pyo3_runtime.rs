#![cfg(feature = "pyo3")]

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::time::Duration;

#[tokio::test]
async fn test_send_after_delivers_message() {
    // create runtime, spawn actor and schedule timer while holding the GIL
    let (rt_obj, pid): (PyObject, u64) = Python::with_gil(|py| {
        let module = iris::py::make_module(py).unwrap();
        let rt = module
            .as_ref(py)
            .getattr("PyRuntime")
            .unwrap()
            .call0()
            .unwrap();

        // Spawn an observed handler to collect messages
        let pid: u64 = rt
            .call_method1("spawn_observed_handler", (10usize,))
            .unwrap()
            .extract()
            .unwrap();

        // Schedule a message after 50ms
        let _timer_id: u64 = rt
            .call_method1("send_after", (pid, 50u64, PyBytes::new(py, b"delayed")))
            .unwrap()
            .extract()
            .unwrap();

        (rt.into_py(py), pid)
    });

    // allow the runtime to process (non-blocking)
    tokio::time::sleep(Duration::from_millis(120)).await;

    // now check messages with GIL again
    Python::with_gil(|py| {
        let rt = rt_obj.as_ref(py);
        let msgs: Vec<pyo3::PyObject> = rt
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap();

        assert!(!msgs.is_empty(), "expected at least one delivered message");
    });
}

#[tokio::test]
async fn py_zero_copy_send() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // spawn observed handler
    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_observed_handler", (1usize,))
            .unwrap()
            .extract()
            .unwrap()
    });

    // allocate a Rust-owned buffer and write into it from Python
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rv = module
            .as_ref(py)
            .call_method1("allocate_buffer", (5usize,))
            .unwrap();
        let (id, mem, cap): (u64, pyo3::PyObject, pyo3::PyObject) = rv.extract().unwrap();
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("mem", mem.as_ref(py)).unwrap();
        locals.set_item("cap", cap.as_ref(py)).unwrap();
        py.run("mem[:5] = b'hello'", None, Some(locals)).unwrap();
        // send the buffer without copying
        rt_py
            .as_ref(py)
            .call_method1("send_buffer", (pid, id))
            .unwrap();
    });

    // allow the tokio tasks to run
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let msgs: Vec<Vec<u8>> = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap()
    });

    assert_eq!(msgs.len(), 1);
    assert_eq!(&msgs[0], b"hello");
}

#[tokio::test]
async fn py_send_accepts_bytes_like_objects() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_observed_handler", (8usize,))
            .unwrap()
            .extract()
            .unwrap()
    });

    Python::with_gil(|py| {
        let bytearray_obj = py
            .eval("bytearray(b'hello-bytearray')", None, None)
            .unwrap();
        let memoryview_obj = py
            .eval("memoryview(b'hello-memoryview')", None, None)
            .unwrap();

        let sent_ba: bool = rt_py
            .as_ref(py)
            .call_method1("send", (pid, bytearray_obj))
            .unwrap()
            .extract()
            .unwrap();
        assert!(sent_ba);

        let sent_mv: bool = rt_py
            .as_ref(py)
            .call_method1("send", (pid, memoryview_obj))
            .unwrap()
            .extract()
            .unwrap();
        assert!(sent_mv);
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let msgs: Vec<Vec<u8>> = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap()
    });

    assert_eq!(msgs.len(), 2);
    assert_eq!(&msgs[0], b"hello-bytearray");
    assert_eq!(&msgs[1], b"hello-memoryview");
}

#[tokio::test]
async fn py_send_many_accepts_bytes_like_objects() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_observed_handler", (16usize,))
            .unwrap()
            .extract()
            .unwrap()
    });

    let accepted: usize = Python::with_gil(|py| {
        let payloads = py
            .eval(
                "[b'one', bytearray(b'two'), memoryview(b'three')]",
                None,
                None,
            )
            .unwrap();
        rt_py
            .as_ref(py)
            .call_method1("send_many", (pid, payloads))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert_eq!(accepted, 3);

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let msgs: Vec<Vec<u8>> = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap()
    });

    assert_eq!(
        msgs,
        vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]
    );
}

#[tokio::test]
async fn py_send_push_loop_does_not_yield_gil_mid_burst_regression() {
    let (rt_py, pid, list_obj, old_switch): (PyObject, u64, PyObject, f64) = Python::with_gil(
        |py| {
            let module = iris::py::make_module(py).expect("make_module");
            let runtime_type = module
                .as_ref(py)
                .getattr("PyRuntime")
                .expect("no PyRuntime type");
            let rt = runtime_type.call0().expect("construct PyRuntime");

            let locals = pyo3::types::PyDict::new(py);
            py.run(
                "import sys\nold = sys.getswitchinterval()\nsys.setswitchinterval(1.0)\nitems = []\ndef cb(b):\n    items.append(b)\n",
                Some(locals),
                Some(locals),
            )
            .unwrap();

            let cb = locals.get_item("cb").unwrap().unwrap();
            let list_obj: PyObject = locals.get_item("items").unwrap().unwrap().into_py(py);
            let old_switch: f64 = locals.get_item("old").unwrap().unwrap().extract().unwrap();

            let pid: u64 = rt
                .call_method1("spawn_py_handler", (cb, 100usize, false))
                .unwrap()
                .extract()
                .unwrap();

            for _ in 0..2000usize {
                let ok: bool = rt
                    .call_method1("send", (pid, PyBytes::new(py, b"x")))
                    .unwrap()
                    .extract()
                    .unwrap();
                assert!(ok);
            }

            let len_during_burst: usize = list_obj
                .as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len();
            assert_eq!(
                len_during_burst, 0,
                "callbacks ran during send burst (send likely yielded GIL)"
            );

            (rt.into_py(py), pid, list_obj, old_switch)
        },
    );

    tokio::time::sleep(Duration::from_millis(250)).await;

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let _ = rt; // Keep runtime alive while assertions run.

        let items = list_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        assert!(
            !items.is_empty(),
            "expected messages to be eventually processed after burst"
        );

        let sys = py.import("sys").unwrap();
        sys.call_method1("setswitchinterval", (old_switch,))
            .unwrap();

        // Cleanup spawned actor.
        let _ = rt.call_method1("stop", (pid,));
    });
}

#[tokio::test]
async fn py_runtime_spawn_and_send() {
    // create a single PyRuntime instance and keep it alive across await points
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // spawn an observed handler and send a message (call into the same PyRuntime)
    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_observed_handler", (1usize,))
            .unwrap()
            .extract()
            .unwrap()
    });

    let sent: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"hello")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(sent, "send failed");

    // allow the tokio tasks to run
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // inspect recorded messages for the same runtime/pid
    let msgs: Vec<Vec<u8>> = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("get_messages", (pid,))
            .unwrap()
            .extract()
            .unwrap()
    });

    assert_eq!(msgs.len(), 1);
    assert_eq!(&msgs[0], b"hello");

    // --- NEW: spawn a Python-backed handler that appends to a Python list ---
    let lst_obj: pyo3::PyObject = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");

        // create a Python list and a callback that appends bytes to it
        let lst = pyo3::types::PyList::empty(py);
        let lst_obj: pyo3::PyObject = lst.into_py(py);
        let locals = pyo3::types::PyDict::new(py);
        // build a factory that returns a callback which closes over `lst`
        py.run(
            "def make_cb(lst):\n    def cb(b): lst.append(b)\n    return cb\n",
            None,
            Some(locals),
        )
        .unwrap();
        let make_cb = locals.get_item("make_cb").unwrap().unwrap();
        let cb: pyo3::PyObject = make_cb.call1((lst_obj.as_ref(py),)).unwrap().into();

        // spawn a Python handler and send a message
        let pid: u64 = rt_obj
            .call_method1("spawn_py_handler", (cb, 1usize))
            .unwrap()
            .extract()
            .unwrap();
        rt_obj
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"pycall")))
            .unwrap();

        lst_obj
    });

    // allow the tokio tasks to run (outside the GIL)
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // verify Python list got the bytes
    Python::with_gil(|py| {
        let lst = lst_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<&[u8]> = lst.extract().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], b"pycall");
    });

    // --- NEW: register a Python factory with the supervisor and validate ---
    Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");

        // spawn observed handler to supervise
        let pid: u64 = rt_obj
            .call_method1("spawn_observed_handler", (1usize,))
            .unwrap()
            .extract()
            .unwrap();

        // create a factory that calls back into this same runtime instance
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("rt", rt_obj.into_py(py)).unwrap();
        py.run(
            "def factory(rt=rt):\n    return rt.spawn_observed_handler(1)",
            None,
            Some(locals),
        )
        .unwrap();
        let factory: pyo3::PyObject = locals.get_item("factory").unwrap().unwrap().into();

        rt_obj
            .call_method1("supervise_with_factory", (pid, factory, "RestartOne"))
            .unwrap();
        let count: usize = rt_obj
            .call_method0("children_count")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(count, 1);

        let pids: Vec<u64> = rt_obj
            .call_method0("child_pids")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(pids.len(), 1);
        assert_eq!(pids[0], pid);
    });
}

// simple bounded mailbox send drop-new test
#[tokio::test]
async fn py_bounded_mailbox_drop_new() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let msgs: pyo3::PyObject = Python::with_gil(|py| pyo3::types::PyList::empty(py).into_py(py));

    let cb = Python::with_gil(|py| {
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("msgs", msgs.as_ref(py)).unwrap();
        py.run(
            r#"import time
def cb(msg, msgs=msgs):
    time.sleep(0.01)
    msgs.append(bytes(msg))
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();
        locals.get_item("cb").unwrap().unwrap().to_object(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1(
                "spawn_py_handler_bounded",
                (cb.clone_ref(py), 100usize, 1usize, false),
            )
            .unwrap()
            .extract()
            .unwrap()
    });

    let send_results: Vec<bool> = Python::with_gil(|py| {
        let mut out = Vec::with_capacity(128);
        for i in 0..128u16 {
            let payload = [((i % 251) as u8)];
            let ok: bool = rt_py
                .as_ref(py)
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, &payload)))
                .unwrap()
                .extract()
                .unwrap();
            out.push(ok);
        }
        out
    });

    let accepted = send_results.iter().filter(|ok| **ok).count();
    let dropped = send_results.len() - accepted;
    assert!(accepted > 0, "expected at least one accepted message");
    assert!(
        dropped > 0,
        "expected drop-new to reject some messages under pressure"
    );

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let current_len: usize = Python::with_gil(|py| {
            msgs.as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len()
        });
        if current_len >= accepted {
            break;
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "timeout waiting for accepted messages to drain: accepted={}, observed={}",
                accepted, current_len
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    let final_len: usize = Python::with_gil(|py| {
        msgs.as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap()
            .len()
    });
    assert_eq!(
        final_len, accepted,
        "processed count should match accepted sends"
    );
}

// ---------- structured concurrency tests ----------

#[tokio::test]
async fn py_structured_concurrency_normal_and_crash() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    // helper to spawn a simple handler that stores messages in a list
    let make_handler = |py: Python<'_>| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            "def make_cb(lst):\n    def cb(b): lst.append(b)\n    return cb\n",
            None,
            Some(locals),
        )
        .unwrap();
        let lst = pyo3::types::PyList::empty(py);
        let make_cb = locals.get_item("make_cb").unwrap().unwrap();
        let cb = make_cb.call1((lst,)).unwrap().into_py(py);
        (lst.into_py(py), cb)
    };

    // normal-exit scenario
    let (_parent_list, parent_cb): (pyo3::PyObject, pyo3::PyObject) =
        Python::with_gil(make_handler);
    let parent_pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_py_handler", (parent_cb.clone(), 1usize))
            .unwrap()
            .extract()
            .unwrap()
    });

    // debug: list available methods on PyRuntime
    Python::with_gil(|py| {
        let obj = rt_py.as_ref(py);
        let dirlist: Vec<String> = obj
            .dir()
            .iter()
            .map(|item| item.extract::<String>().unwrap())
            .collect();
        eprintln!("PyRuntime attributes: {:?}", dirlist);
    });
    let child_pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1(
                "spawn_child",
                (parent_pid, parent_cb.clone(), 1usize, false),
            )
            .unwrap()
            .extract()
            .unwrap()
    });

    // send a message to the parent; the handler itself doesn't exit
    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (parent_pid, pyo3::types::PyBytes::new(py, b"ok")))
            .unwrap();
    });

    // now explicitly stop the parent (normal shutdown) to exercise structured concurrency
    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("stop", (parent_pid,))
            .unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let alive: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("is_alive", (child_pid,))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(!alive, "child should die when parent exits normally");

    // crash scenario: spawn parent that panics
    let crash_cb: pyo3::PyObject = Python::with_gil(|py| {
        let src = "def cb(_):\n    raise Exception('crash')\n";
        let locals = pyo3::types::PyDict::new(py);
        py.run(src, None, Some(locals)).unwrap();
        locals.get_item("cb").unwrap().unwrap().into_py(py)
    });
    let parent_crash: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_py_handler", (crash_cb.clone(), 1usize))
            .unwrap()
            .extract()
            .unwrap()
    });
    let child_crash: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_child", (parent_crash, crash_cb, 1usize, false))
            .unwrap()
            .extract()
            .unwrap()
    });

    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (parent_crash, pyo3::types::PyBytes::new(py, b"go")))
            .unwrap();
    });

    // wait a bit for Python exception to be logged
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // explicitly terminate the parent after crash
    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("stop", (parent_crash,))
            .unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let alive2: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("is_alive", (child_crash,))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(!alive2, "child should die when parent crashes");
}

#[tokio::test]
async fn py_spawn_child_pool_reuses_workers_under_parent() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    let (parent_pid, _worker_pids, lst_obj): (u64, Vec<u64>, pyo3::PyObject) =
        Python::with_gil(|py| {
            let rt = rt_py.as_ref(py);

            let parent_pid: u64 = rt
                .call_method1("spawn_observed_handler", (1usize,))
                .unwrap()
                .extract()
                .unwrap();

            let lst = pyo3::types::PyList::empty(py);
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("lst", lst).unwrap();
            py.run(
                "def cb(b):\n    lst.append(bytes(b))",
                Some(locals),
                Some(locals),
            )
            .unwrap();
            let cb: pyo3::PyObject = locals.get_item("cb").unwrap().unwrap().into();

            let worker_pids: Vec<u64> = rt
                .call_method1("spawn_child_pool", (parent_pid, cb, 4usize, 64usize, false))
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(worker_pids.len(), 4);

            for (i, pid) in worker_pids.iter().enumerate() {
                let payload = format!("m{}", i);
                let ok: bool = rt
                    .call_method1(
                        "send",
                        (*pid, pyo3::types::PyBytes::new(py, payload.as_bytes())),
                    )
                    .unwrap()
                    .extract()
                    .unwrap();
                assert!(ok);
            }

            (parent_pid, worker_pids, lst.into_py(py))
        });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    Python::with_gil(|py| {
        let lst = lst_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<Vec<u8>> = lst.extract().unwrap();
        assert_eq!(got.len(), 4);
    });

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        rt.call_method1("stop", (parent_pid,)).unwrap();
    });
}

// (Old overflow tests removed — replaced by deterministic `tests/overflow_policies.rs`)

#[tokio::test]
async fn py_virtual_actor_activates_on_first_send() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let list_obj: pyo3::PyObject =
        Python::with_gil(|py| pyo3::types::PyList::empty(py).into_py(py));

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", list_obj.as_ref(py)).unwrap();
        let handler = py
            .eval(
                "lambda m, lst=lst: lst.append(bytes(m))",
                None,
                Some(locals),
            )
            .unwrap();

        rt.call_method1(
            "spawn_virtual_py_handler",
            (handler, 16usize, Option::<u64>::None),
        )
        .unwrap()
        .extract()
        .unwrap()
    });

    let sent: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"lazy")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(sent);

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    Python::with_gil(|py| {
        let lst = list_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap();
        let got: Vec<Vec<u8>> = lst.extract().unwrap();
        assert_eq!(got, vec![b"lazy".to_vec()]);
    });
}

#[tokio::test]
async fn py_virtual_actor_idle_timeout() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let handler = py.eval("lambda m: None", None, None).unwrap();
        rt.call_method1("spawn_virtual_py_handler", (handler, 8usize, Some(50u64)))
            .unwrap()
            .extract()
            .unwrap()
    });

    let sent: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"ping")))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(sent);

    tokio::time::sleep(std::time::Duration::from_millis(220)).await;

    let alive: bool = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("is_alive", (pid,))
            .unwrap()
            .extract()
            .unwrap()
    });
    assert!(!alive, "virtual actor should stop after idle timeout");
}

#[tokio::test]
async fn py_handler_resilient_to_all_exit_types() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    let (pid_ok, pid_sys_exit, pid_exception) = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);

        // A regular actor that survives
        let cb_ok = py.eval("lambda m: None", None, None).unwrap();
        let pid_ok: u64 = rt
            .call_method1("spawn_py_handler", (cb_ok, 1usize))
            .unwrap()
            .extract()
            .unwrap();

        // An actor that does sys.exit(1)
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            "import sys\ndef sys_ext(m):\n    sys.exit(1)\n",
            None,
            Some(locals),
        )
        .unwrap();
        let cb_sys = locals.get_item("sys_ext").unwrap().unwrap();
        let pid_sys_exit: u64 = rt
            .call_method1("spawn_py_handler", (cb_sys, 1usize))
            .unwrap()
            .extract()
            .unwrap();

        // An actor that raises an exception
        let locals2 = pyo3::types::PyDict::new(py);
        py.run(
            "def exc(m):\n    raise ValueError('boom')\n",
            None,
            Some(locals2),
        )
        .unwrap();
        let cb_exc = locals2.get_item("exc").unwrap().unwrap();
        let pid_exception: u64 = rt
            .call_method1("spawn_py_handler", (cb_exc, 1usize))
            .unwrap()
            .extract()
            .unwrap();

        (pid_ok, pid_sys_exit, pid_exception)
    });

    // Send messages to trigger them
    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let msg = pyo3::types::PyBytes::new(py, b"trigger");
        rt.call_method1("send", (pid_ok, msg)).unwrap();
        rt.call_method1("send", (pid_sys_exit, msg)).unwrap();
        rt.call_method1("send", (pid_exception, msg)).unwrap();
    });

    // Let the Tokio runtime churn and execute the actors
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Verify properties
    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);

        let is_ok_alive: bool = rt
            .call_method1("is_alive", (pid_ok,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(is_ok_alive, "Normal actor should still be alive");

        let is_sys_alive: bool = rt
            .call_method1("is_alive", (pid_sys_exit,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(!is_sys_alive, "Actor that called sys.exit() should be dead");

        let is_exc_alive: bool = rt
            .call_method1("is_alive", (pid_exception,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(
            !is_exc_alive,
            "Actor that raised an exception should be dead"
        );
    });
}

#[tokio::test]
async fn test_py_handler_robust_exit_supervision() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");
        rt_obj.into_py(py)
    });

    let pids: Vec<u64> = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);

        // A monitor actor to capture EXIT messages
        let monitor_pid: u64 = rt
            .call_method1("spawn_observed_handler", (20usize,))
            .unwrap()
            .extract()
            .unwrap();

        // 1. sys.exit(2)
        // 2. KeyboardInterrupt
        // 3. BaseException directly
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            "def cb_sys(m):\n    import sys\n    sys.exit(2)\n\
             def cb_kbd(m):\n    raise KeyboardInterrupt()\n\
             def cb_base(m):\n    raise BaseException('base')\n",
            None,
            Some(locals),
        )
        .unwrap();

        let mut workers = Vec::new();
        for cb_name in &["cb_sys", "cb_kbd", "cb_base"] {
            let cb = locals.get_item(cb_name).unwrap().unwrap();
            let pid: u64 = rt
                .call_method1("spawn_py_handler", (cb, 10usize))
                .unwrap()
                .extract()
                .unwrap();

            // Link to monitor so it receives EXIT messages
            rt.call_method1("link", (monitor_pid, pid)).unwrap();
            workers.push(pid);

            // Trigger failure
            let bytes = pyo3::types::PyBytes::new(py, b"die");
            rt.call_method1("send", (pid, bytes)).unwrap();
        }

        workers.push(monitor_pid);
        workers
    });

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);

        let monitor_pid = pids.last().unwrap();
        for pid in pids.iter().take(pids.len() - 1) {
            let alive: bool = rt
                .call_method1("is_alive", (*pid,))
                .unwrap()
                .extract()
                .unwrap();
            assert!(!alive, "Failing actor {} should be dead", pid);
        }

        // Grab messages from the monitor (which is a raw Rust observer returning py objects of `EXIT`)
        let msgs: Vec<pyo3::PyObject> = rt
            .call_method1("get_messages", (*monitor_pid,))
            .unwrap()
            .extract()
            .unwrap();

        // We expect ONE EXIT message per failing worker
        assert_eq!(
            msgs.len(),
            3,
            "Monitor should have received exactly 3 exit messages"
        );

        let mut exit_pids = Vec::new();
        for msg in &msgs {
            // The PySystemMessage returned by the monitor should look like an object with type_name == 'EXIT'
            let type_name: String = msg.getattr(py, "type_name").unwrap().extract(py).unwrap();
            assert_eq!(type_name, "EXIT");
            let target_pid: u64 = msg.getattr(py, "target_pid").unwrap().extract(py).unwrap();
            exit_pids.push(target_pid);

            // Check that the reason string indicates a panic
            let reason: String = msg.getattr(py, "reason").unwrap().extract(py).unwrap();
            assert_eq!(
                reason, "normal",
                "Exit reason to supervisor should be 'panic'"
            );
        }
        assert_eq!(exit_pids.len(), 3);
    });
}
