#![cfg(feature = "pyo3")]

use pyo3::prelude::*;

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
        let make_cb = locals.get_item("make_cb").unwrap();
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
        let factory: pyo3::PyObject = locals.get_item("factory").unwrap().into();

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
        locals.get_item("cb").unwrap().to_object(py)
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
    assert!(dropped > 0, "expected drop-new to reject some messages under pressure");

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
    assert_eq!(final_len, accepted, "processed count should match accepted sends");
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
        let cb = locals
            .get_item("make_cb")
            .unwrap()
            .call1((lst,))
            .unwrap()
            .into_py(py);
        (lst.into_py(py), cb)
    };

    // normal-exit scenario
    let (_parent_list, parent_cb): (pyo3::PyObject, pyo3::PyObject) = Python::with_gil(|py| make_handler(py));
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
            .call_method1("spawn_child", (parent_pid, parent_cb.clone(), 1usize, false))
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
        locals.get_item("cb").unwrap().into_py(py)
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

    let (parent_pid, _worker_pids, lst_obj): (u64, Vec<u64>, pyo3::PyObject) = Python::with_gil(|py| {
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
        let cb: pyo3::PyObject = locals.get_item("cb").unwrap().into();

        let worker_pids: Vec<u64> = rt
            .call_method1("spawn_child_pool", (parent_pid, cb, 4usize, 64usize, false))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(worker_pids.len(), 4);

        for (i, pid) in worker_pids.iter().enumerate() {
            let payload = format!("m{}", i);
            let ok: bool = rt
                .call_method1("send", (*pid, pyo3::types::PyBytes::new(py, payload.as_bytes())))
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

// overflow policy tests -------------------------------------------------

#[tokio::test]
async fn py_overflow_drop_old() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let lst_obj: pyo3::PyObject = Python::with_gil(|py| pyo3::types::PyList::empty(py).into_py(py));

    let (pid, accepted): (u64, usize) = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", lst_obj.as_ref(py)).unwrap();
        py.run(
            r#"import time
def cb(m, lst=lst):
    time.sleep(0.005)
    lst.append(bytes(m))
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let handler = locals.get_item("cb").unwrap();
        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (handler, 100usize, 2usize, false))
            .unwrap()
            .extract()
            .unwrap();
        rt.call_method1("set_overflow_policy", (pid, "dropold", Option::<u64>::None)).unwrap();

        let mut accepted = 0usize;
        for i in 0..40u8 {
            let ok: bool = rt
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, &[i])))
                .unwrap()
                .extract()
                .unwrap();
            if ok {
                accepted += 1;
            }
        }
        (pid, accepted)
    });

    assert!(accepted > 0, "dropold accepted no messages under load");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let mut prev_len = 0usize;
    let mut stable_ticks = 0usize;
    loop {
        let current_len: usize = Python::with_gil(|py| {
            lst_obj
                .as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len()
        });
        if current_len == prev_len {
            stable_ticks += 1;
        } else {
            stable_ticks = 0;
            prev_len = current_len;
        }
        if stable_ticks >= 5 && current_len > 0 {
            break;
        }
        if std::time::Instant::now() >= deadline {
            panic!("dropold timeout waiting for processing, observed={}", current_len);
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    Python::with_gil(|py| {
        let got: Vec<Vec<u8>> = lst_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap()
            .extract()
            .unwrap();
        assert!(!got.is_empty());
        assert!(got.len() <= 40);

        let _ = pid;
    });
}

#[tokio::test]
async fn py_overflow_redirect() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let (lst1_obj, lst2_obj): (pyo3::PyObject, pyo3::PyObject) = Python::with_gil(|py| {
        (
            pyo3::types::PyList::empty(py).into_py(py),
            pyo3::types::PyList::empty(py).into_py(py),
        )
    });

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals1 = pyo3::types::PyDict::new(py);
        locals1.set_item("lst1", lst1_obj.as_ref(py)).unwrap();
        py.run(
            r#"import time
def cb1(m, lst1=lst1):
    time.sleep(0.005)
    lst1.append(bytes(m))
"#,
            Some(locals1),
            Some(locals1),
        )
        .unwrap();
        let handler1 = locals1.get_item("cb1").unwrap();

        let locals2 = pyo3::types::PyDict::new(py);
        locals2.set_item("lst2", lst2_obj.as_ref(py)).unwrap();
        let handler2 = py
            .eval("lambda m, lst2=lst2: lst2.append(bytes(m))", None, Some(locals2))
            .unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (handler1, 100usize, 1usize, false))
            .unwrap()
            .extract()
            .unwrap();
        let fid: u64 = rt
            .call_method1("spawn_py_handler", (handler2, 100usize, false))
            .unwrap()
            .extract()
            .unwrap();
        rt.call_method1("set_overflow_policy", (pid, "redirect", fid)).unwrap();

        for i in 0..50u8 {
            let _ok: bool = rt
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, &[i])))
                .unwrap()
                .extract()
                .unwrap();
        }
        pid
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let (len1, len2): (usize, usize) = Python::with_gil(|py| {
            let l1 = lst1_obj
                .as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len();
            let l2 = lst2_obj
                .as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len();
            (l1, l2)
        });
        if len1 + len2 >= 10 {
            break;
        }
        if std::time::Instant::now() >= deadline {
            panic!("redirect timeout waiting for message processing");
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        rt.call_method1("stop", (pid,)).unwrap();

        let got1: Vec<Vec<u8>> = lst1_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap()
            .extract()
            .unwrap();
        let got2: Vec<Vec<u8>> = lst2_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap()
            .extract()
            .unwrap();
        assert!(!got1.is_empty());
        assert!(!got2.is_empty());
    });
}

#[tokio::test]
async fn py_overflow_spill() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let (primary_obj, fallback_obj): (pyo3::PyObject, pyo3::PyObject) = Python::with_gil(|py| {
        (
            pyo3::types::PyList::empty(py).into_py(py),
            pyo3::types::PyList::empty(py).into_py(py),
        )
    });

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let p_locals = pyo3::types::PyDict::new(py);
        p_locals.set_item("lst", primary_obj.as_ref(py)).unwrap();
        py.run(
            r#"import time
def primary_cb(m, lst=lst):
    time.sleep(0.01)
    lst.append(bytes(m))
"#,
            Some(p_locals),
            Some(p_locals),
        )
        .unwrap();
        let primary_handler = p_locals.get_item("primary_cb").unwrap();

        let f_locals = pyo3::types::PyDict::new(py);
        f_locals.set_item("lst", fallback_obj.as_ref(py)).unwrap();
        let fallback_handler = py
            .eval("lambda m, lst=lst: lst.append(bytes(m))", None, Some(f_locals))
            .unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (primary_handler, 100usize, 1usize, false))
            .unwrap()
            .extract()
            .unwrap();

        let fid: u64 = rt
            .call_method1("spawn_py_handler", (fallback_handler, 100usize, false))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("set_overflow_policy", (pid, "spill", fid)).unwrap();

        let _ok: bool = rt
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"x")))
            .unwrap()
            .extract()
            .unwrap();
        pid
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        let (len1, len2): (usize, usize) = Python::with_gil(|py| {
            let l1 = primary_obj
                .as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len();
            let l2 = fallback_obj
                .as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len();
            (l1, l2)
        });
        if len1 + len2 >= 1 {
            break;
        }
        if std::time::Instant::now() >= deadline {
            panic!("spill timeout waiting for processing");
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        rt.call_method1("stop", (pid,)).unwrap();

        let got_primary: Vec<Vec<u8>> = primary_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap()
            .extract()
            .unwrap();
        let got_fallback: Vec<Vec<u8>> = fallback_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap()
            .extract()
            .unwrap();
        assert!(!got_primary.is_empty() || !got_fallback.is_empty());
    });
}

#[tokio::test]
async fn py_overflow_block() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type.call0().unwrap().into_py(py)
    });

    let list_obj: pyo3::PyObject = Python::with_gil(|py| pyo3::types::PyList::empty(py).into_py(py));

    let pid: u64 = Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("lst", list_obj.as_ref(py)).unwrap();
        py.run(
            r#"import time
def cb(m, lst=lst):
    time.sleep(0.005)
    lst.append(bytes(m))
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();
        let handler = locals.get_item("cb").unwrap();

        let pid: u64 = rt
            .call_method1("spawn_py_handler_bounded", (handler, 100usize, 1usize, false))
            .unwrap()
            .extract()
            .unwrap();

        rt.call_method1("set_overflow_policy", (pid, "block", Option::<u64>::None)).unwrap();

        let _ok1: bool = rt
            .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"a")))
            .unwrap()
            .extract()
            .unwrap();
        pid
    });

    let rt_for_send = rt_py.clone();
    let send_task = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            rt_for_send
                .as_ref(py)
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"b")))
                .unwrap()
                .extract::<bool>()
                .unwrap()
        })
    });

    let _ok2 = tokio::time::timeout(std::time::Duration::from_secs(2), send_task)
        .await
        .expect("block secondary send timed out")
        .expect("block secondary send join");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let len_now: usize = Python::with_gil(|py| {
            list_obj
                .as_ref(py)
                .downcast::<pyo3::types::PyList>()
                .unwrap()
                .len()
        });
        if len_now >= 2 {
            break;
        }
        if std::time::Instant::now() >= deadline {
            panic!("block timeout waiting for processing");
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    Python::with_gil(|py| {
        let rt = rt_py.as_ref(py);
        rt.call_method1("stop", (pid,)).unwrap();
        let got: Vec<Vec<u8>> = list_obj
            .as_ref(py)
            .downcast::<pyo3::types::PyList>()
            .unwrap()
            .extract()
            .unwrap();
        assert!(!got.is_empty());
    });
}

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
            .eval("lambda m, lst=lst: lst.append(bytes(m))", None, Some(locals))
            .unwrap();

        rt.call_method1("spawn_virtual_py_handler", (handler, 16usize, Option::<u64>::None))
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

