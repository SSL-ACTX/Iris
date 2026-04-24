#![cfg(all(feature = "pyo3", feature = "vortex"))]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::time::{sleep, timeout, Duration};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_ocular_api_is_exposed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        assert!(m.getattr(py, "start_tracing").is_ok());
        assert!(m.getattr(py, "stop_tracing").is_ok());
        assert!(m.getattr(py, "instruction_callback").is_ok());
        assert!(m.getattr(py, "py_start_callback").is_ok());
        assert!(m.getattr(py, "jump_callback").is_ok());
        assert!(m.getattr(py, "py_return_callback").is_ok());
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_ocular_start_stop_tracing_smoke() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();
        let sys = py.import("sys").unwrap();

        // PEP 669 monitoring APIs are required for Ocular wiring.
        if !sys.hasattr("monitoring").unwrap_or(false) {
            return;
        }

        m.getattr(py, "start_tracing")
            .unwrap()
            .call(
                py,
                (
                    "precise",
                    64_u32,
                    Vec::<String>::new(),
                    Vec::<String>::new(),
                ),
                None,
            )
            .unwrap();

        let globals = PyDict::new(py);
        py.run(
            r#"
def ocular_smoke(n):
    acc = 0
    i = 0
    while i < n:
        acc += i
        i += 1
    return acc

_r = ocular_smoke(64)
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let result: i64 = globals.get_item("_r").unwrap().unwrap().extract().unwrap();
        assert_eq!(result, 2016);

        m.getattr(py, "stop_tracing").unwrap().call0(py).unwrap();

        let stats = m
            .getattr(py, "get_tracing_stats")
            .unwrap()
            .call0(py)
            .unwrap();
        let stats = stats.downcast::<PyDict>(py).unwrap();
        let processed_events: u64 = stats
            .get_item("processed_events")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let instruction_events: u64 = stats
            .get_item("instruction_events")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert!(processed_events > 0);
        assert!(instruction_events > 0);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_ocular_captures_non_hot_paths() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();
        let sys = py.import("sys").unwrap();

        if !sys.hasattr("monitoring").unwrap_or(false) {
            return;
        }

        m.getattr(py, "start_tracing")
            .unwrap()
            .call(
                py,
                ("precise", 0_u32, Vec::<String>::new(), Vec::<String>::new()),
                None,
            )
            .unwrap();

        let globals = PyDict::new(py);
        py.run(
            r#"
def straight_path(a, b):
    c = a + b
    d = c * 2
    return d - a

_x = straight_path(7, 5)
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let value: i64 = globals.get_item("_x").unwrap().unwrap().extract().unwrap();
        assert_eq!(value, 17);

        m.getattr(py, "stop_tracing").unwrap().call0(py).unwrap();

        let stats = m
            .getattr(py, "get_tracing_stats")
            .unwrap()
            .call0(py)
            .unwrap();
        let stats = stats.downcast::<PyDict>(py).unwrap();
        let instruction_events: u64 = stats
            .get_item("instruction_events")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let unique_instruction_sites: usize = stats
            .get_item("unique_instruction_sites")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert!(instruction_events > 0);
        assert!(unique_instruction_sites > 0);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_ocular_callbacks_smoke() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        py.run(
            r#"
def cb_target():
    x = 1
    return x
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let code_obj = globals
            .get_item("cb_target")
            .unwrap()
            .unwrap()
            .getattr("__code__")
            .unwrap();

        m.getattr(py, "py_start_callback")
            .unwrap()
            .call1(py, (code_obj, 0_i32))
            .unwrap();

        let inst_res = m
            .getattr(py, "instruction_callback")
            .unwrap()
            .call1(py, (code_obj, 0_i32))
            .unwrap();
        assert!(inst_res.as_ref(py).is_none());

        let jump_res = m
            .getattr(py, "jump_callback")
            .unwrap()
            .call1(py, (code_obj, 0_i32, 0_i32))
            .unwrap();
        assert!(jump_res.as_ref(py).is_none());

        m.getattr(py, "py_return_callback")
            .unwrap()
            .call1(py, (code_obj, 0_i32, py.None()))
            .unwrap();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_preemption_on_while_true() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();

        // Define an endless loop
        let code = r#"
def endless():
    while True:
        pass
"#;
        py.run(code, Some(globals), None).unwrap();

        let endless_func = globals.get_item("endless").unwrap().unwrap();
        let original_code = endless_func
            .getattr("__code__")
            .unwrap()
            .getattr("co_code")
            .unwrap()
            .to_object(py);

        m.getattr(py, "set_budget")
            .unwrap()
            .call1(py, (5,))
            .unwrap();

        // Shadow clone transmutation should not mutate the original function object.
        let shadow = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (endless_func,))
            .unwrap();
        assert!(!shadow.as_ref(py).is(endless_func));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert!(mode == "rewrite" || mode == "fallback");
        assert!(rewrite_attempted || mode == "fallback");

        let current_original_code = endless_func
            .getattr("__code__")
            .unwrap()
            .getattr("co_code")
            .unwrap()
            .to_object(py);
        assert!(original_code
            .as_ref(py)
            .eq(current_original_code.as_ref(py))
            .unwrap());

        // Run transmuted shadow function. It should suspend by budget.
        let res = shadow.call0(py);

        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.is_instance_of::<iris::py::vortex::VortexSuspend>(py));
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_opcode_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample():
    return 42
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample").unwrap().unwrap();

        // Ensure opcode is present in sys.modules before monkeypatching it.
        py.import("opcode").unwrap();
        let sys = py.import("sys").unwrap();
        let modules = sys
            .getattr("modules")
            .unwrap()
            .downcast::<PyDict>()
            .unwrap();
        let original_opcode = modules.get_item("opcode").unwrap().unwrap().to_object(py);

        modules
            .set_item("opcode", py.eval("object()", None, None).unwrap())
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        // Restore before any assertions or further calls that might fail.
        modules.set_item("opcode", original_opcode).unwrap();

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "opcode_metadata_unavailable");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_quickening_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample2():
    return 7
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample2").unwrap().unwrap();
        let opcode = py.import("opcode").unwrap();
        let had_inline = opcode.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(
                opcode
                    .getattr("_inline_cache_entries")
                    .unwrap()
                    .to_object(py),
            )
        } else {
            None
        };

        // Force quickening metadata extraction to fail.
        opcode
            .setattr(
                "_inline_cache_entries",
                py.eval("object()", None, None).unwrap(),
            )
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        if had_inline {
            opcode
                .setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = opcode.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "quickening_metadata_unavailable");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_original_cache_layout_invalid() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample3(x):
    return x + 1
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample3").unwrap().unwrap();
        let opcode = py.import("opcode").unwrap();
        let locals = PyDict::new(py);
        locals.set_item("opcode", opcode).unwrap();
        let cache_opcode: i32 = py
            .eval("opcode.opmap.get('CACHE', -1)", Some(locals), None)
            .unwrap()
            .extract()
            .unwrap();

        if cache_opcode < 0 {
            // Python runtime has no inline cache opcode, so this path is not applicable.
            return;
        }

        let had_inline = opcode.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(
                opcode
                    .getattr("_inline_cache_entries")
                    .unwrap()
                    .to_object(py),
            )
        } else {
            None
        };

        // Force a clearly incompatible table: every non-CACHE opcode expects a cache slot.
        let mut entries = vec![1u16; 256];
        entries[cache_opcode as usize] = 0;
        opcode
            .setattr("_inline_cache_entries", PyList::new(py, entries))
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        if had_inline {
            opcode
                .setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = opcode.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "original_cache_layout_invalid");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_stack_depth_invariant_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_small_stack():
    return 1
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_small_stack").unwrap().unwrap();
        let original_code = sample.getattr("__code__").unwrap().to_object(py);
        let locals = PyDict::new(py);
        locals.set_item("fn", sample).unwrap();
        py.run(
            r#"
fn.__code__ = fn.__code__.replace(co_stacksize=1)
"#,
            Some(locals),
            Some(locals),
        )
        .unwrap();

        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        sample.setattr("__code__", original_code).unwrap();
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "stack_depth_invariant_failed");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_exception_table_invalid() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_exc():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_exc").unwrap().unwrap();
        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_EXCEPTION_TABLE_INVALID", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_EXCEPTION_TABLE_INVALID");

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "exception_table_invalid");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_exception_table_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_exc_meta_unavailable():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_exc_meta_unavailable")
            .unwrap()
            .unwrap();
        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item(
                "IRIS_VORTEX_TEST_FORCE_EXCEPTION_TABLE_METADATA_UNAVAILABLE",
                "1",
            )
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_EXCEPTION_TABLE_METADATA_UNAVAILABLE");

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "exception_table_metadata_unavailable");
        assert!(!rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_patched_exception_table_invalid() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_patched_exc():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_patched_exc").unwrap().unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item(
                "IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_INVALID",
                "1",
            )
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_INVALID");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "patched_exception_table_invalid");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_probe_extraction_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_probe_fail():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals.get_item("sample_probe_fail").unwrap().unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_PROBE_EXTRACTION_FAILED", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PROBE_EXTRACTION_FAILED");

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "probe_extraction_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_probe_instrumentation_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_probe_instrumentation_fail():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_probe_instrumentation_fail")
            .unwrap()
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_PROBE_INSTRUMENTATION_FAILED", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PROBE_INSTRUMENTATION_FAILED");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "probe_instrumentation_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_patched_stack_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_patched_stack_metadata_unavailable():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_patched_stack_metadata_unavailable")
            .unwrap()
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item(
                "IRIS_VORTEX_TEST_FORCE_PATCHED_STACK_METADATA_UNAVAILABLE",
                "1",
            )
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_PATCHED_STACK_METADATA_UNAVAILABLE");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "patched_stack_metadata_unavailable");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_patched_exception_table_metadata_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_patched_exception_table_metadata_unavailable():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_patched_exception_table_metadata_unavailable")
            .unwrap()
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item(
                "IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_METADATA_UNAVAILABLE",
                "1",
            )
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ =
            environ.del_item("IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_METADATA_UNAVAILABLE");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "patched_exception_table_metadata_unavailable");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_code_replace_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_code_replace_fail():
    a = 1
    b = 2
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_code_replace_fail")
            .unwrap()
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_CODE_REPLACE_FAILED", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_CODE_REPLACE_FAILED");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "code_replace_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_types_module_unavailable() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_types_unavailable():
    a = 3
    b = 4
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_types_unavailable")
            .unwrap()
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_TYPES_MODULE_UNAVAILABLE", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_TYPES_MODULE_UNAVAILABLE");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "types_module_unavailable");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_vortex_fallback_reports_shadow_function_construction_failed() {
    Python::with_gil(|py| {
        let m = iris::py::make_module(py).unwrap();

        let globals = PyDict::new(py);
        globals.set_item("iris", &m).unwrap();
        py.run(
            r#"
def sample_shadow_construction_fail():
    a = 5
    b = 6
    return a + b
"#,
            Some(globals),
            None,
        )
        .unwrap();

        let sample = globals
            .get_item("sample_shadow_construction_fail")
            .unwrap()
            .unwrap();
        let dis = py.import("dis").unwrap();
        let had_inline = dis.hasattr("_inline_cache_entries").unwrap();
        let original_inline = if had_inline {
            Some(dis.getattr("_inline_cache_entries").unwrap().to_object(py))
        } else {
            None
        };
        dis.setattr("_inline_cache_entries", PyList::new(py, vec![0u16; 256]))
            .unwrap();

        let os = py.import("os").unwrap();
        let environ = os.getattr("environ").unwrap();
        environ
            .set_item("IRIS_VORTEX_TEST_FORCE_SHADOW_CONSTRUCTION_FAILED", "1")
            .unwrap();

        let shadow_res = m
            .getattr(py, "transmute_function")
            .unwrap()
            .call1(py, (sample,));

        let _ = environ.del_item("IRIS_VORTEX_TEST_FORCE_SHADOW_CONSTRUCTION_FAILED");
        if had_inline {
            dis.setattr("_inline_cache_entries", original_inline.unwrap())
                .unwrap();
        } else {
            let _ = dis.delattr("_inline_cache_entries");
        }

        let shadow = shadow_res.unwrap();
        assert!(!shadow.as_ref(py).is(sample));

        let guard = m
            .getattr(py, "get_guard_status")
            .unwrap()
            .call0(py)
            .unwrap();
        let guard_dict = guard.downcast::<PyDict>(py).unwrap();
        let mode: String = guard_dict
            .get_item("mode")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let reason: String = guard_dict
            .get_item("reason")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_attempted: bool = guard_dict
            .get_item("rewrite_attempted")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        let rewrite_applied: bool = guard_dict
            .get_item("rewrite_applied")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(mode, "fallback");
        assert_eq!(reason, "shadow_function_construction_failed");
        assert!(rewrite_attempted);
        assert!(!rewrite_applied);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_auto_policy_and_telemetry() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method0("vortex_reset_auto_telemetry")
            .unwrap();

        let counts: (u64, u64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_resolution_counts")
            .unwrap()
            .extract()
            .unwrap();
        let replay: u64 = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_replay_count")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(counts, (0, 0));
        assert_eq!(replay, 0);

        let ok: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_auto_ghost_policy", ("PreferPrimary",))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok);

        let current: String = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_ghost_policy")
            .unwrap()
            .extract::<Option<String>>()
            .unwrap()
            .unwrap();
        assert_eq!(current, "PreferPrimary");
    });

    let cb = Python::with_gil(|py| {
        py.eval("lambda _msg: None", None, None)
            .unwrap()
            .to_object(py)
    });

    let pid: u64 = Python::with_gil(|py| {
        rt_py
            .as_ref(py)
            .call_method1("spawn_py_handler", (cb, 8usize, false))
            .unwrap()
            .extract()
            .unwrap()
    });

    Python::with_gil(|py| {
        for _ in 0..1400u32 {
            let _ = rt_py
                .as_ref(py)
                .call_method1("send", (pid, pyo3::types::PyBytes::new(py, b"tick")));
        }
    });

    timeout(Duration::from_secs(3), async {
        loop {
            let replay: u64 = Python::with_gil(|py| {
                rt_py
                    .as_ref(py)
                    .call_method0("vortex_get_auto_replay_count")
                    .unwrap()
                    .extract()
                    .unwrap()
            });

            if replay > 0 {
                break;
            }

            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("auto replay telemetry should increase");

    Python::with_gil(|py| {
        let counts: (u64, u64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_resolution_counts")
            .unwrap()
            .extract()
            .unwrap();
        assert!(counts.0 > 0);
        assert_eq!(counts.1, 0);

        let replay: u64 = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_replay_count")
            .unwrap()
            .extract()
            .unwrap();
        assert!(replay > 0);

        rt_py
            .as_ref(py)
            .call_method0("vortex_reset_auto_telemetry")
            .unwrap();
        let counts_after: (u64, u64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_resolution_counts")
            .unwrap()
            .extract()
            .unwrap();
        let replay_after: u64 = rt_py
            .as_ref(py)
            .call_method0("vortex_get_auto_replay_count")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(counts_after, (0, 0));
        assert_eq!(replay_after, 0);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_auto_policy_rejects_invalid_value() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        let res = rt_py
            .as_ref(py)
            .call_method1("vortex_set_auto_ghost_policy", ("not-a-policy",));
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_genetic_budgeting_toggle() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        let initial: bool = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_budgeting")
            .unwrap()
            .extract()
            .unwrap();
        assert!(!initial);

        let ok_true: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_genetic_budgeting", (true,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok_true);

        let after_true: bool = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_budgeting")
            .unwrap()
            .extract()
            .unwrap();
        assert!(after_true);

        let ok_false: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_genetic_budgeting", (false,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok_false);

        let after_false: bool = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_budgeting")
            .unwrap()
            .extract()
            .unwrap();
        assert!(!after_false);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_genetic_threshold_roundtrip() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        let (low, high): (f64, f64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_thresholds")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!((low, high), (0.4, 0.7));

        let ok: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_genetic_thresholds", (0.2f64, 0.5f64))
            .unwrap()
            .extract()
            .unwrap();
        assert!(ok);

        let (low, high): (f64, f64) = rt_py
            .as_ref(py)
            .call_method0("vortex_get_genetic_thresholds")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!((low, high), (0.2, 0.5));

        let ok_bad: bool = rt_py
            .as_ref(py)
            .call_method1("vortex_set_genetic_thresholds", (0.7f64, 0.2f64))
            .unwrap()
            .extract()
            .unwrap();
        assert!(!ok_bad);
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_watchdog_toggle() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        let enabled: Option<bool> = rt_py
            .as_ref(py)
            .call_method0("vortex_watchdog_enabled")
            .unwrap()
            .extract()
            .unwrap();
        // Watchdog starts disabled by default.
        assert!(!enabled.unwrap_or(true));

        let ok = rt_py
            .as_ref(py)
            .call_method0("vortex_watchdog_enable")
            .unwrap()
            .extract::<bool>()
            .unwrap();
        assert!(ok);

        let enabled: Option<bool> = rt_py
            .as_ref(py)
            .call_method0("vortex_watchdog_enabled")
            .unwrap()
            .extract()
            .unwrap();
        assert!(enabled.unwrap_or(false));

        let ok = rt_py
            .as_ref(py)
            .call_method0("vortex_watchdog_disable")
            .unwrap()
            .extract::<bool>()
            .unwrap();
        assert!(ok);

        let enabled: Option<bool> = rt_py
            .as_ref(py)
            .call_method0("vortex_watchdog_enabled")
            .unwrap()
            .extract()
            .unwrap();
        assert!(!enabled.unwrap_or(true));
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_isolation_disallow_ops() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        let dis = py.import("dis").unwrap();
        let opmap = dis.getattr("opmap").unwrap();
        let store_global: u8 = opmap.get_item("STORE_GLOBAL").unwrap().extract().unwrap();
        let store_attr: u8 = opmap.get_item("STORE_ATTR").unwrap().extract().unwrap();

        rt_py
            .as_ref(py)
            .call_method1(
                "vortex_set_isolation_disallowed_ops",
                (vec![store_global, store_attr],),
            )
            .unwrap();

        let ops: Option<Vec<u8>> = rt_py
            .as_ref(py)
            .call_method0("vortex_get_isolation_disallowed_ops")
            .unwrap()
            .extract()
            .unwrap();

        assert!(ops.is_some());
        let mut ops = ops.unwrap();
        ops.sort();
        let mut expected = vec![store_global, store_attr];
        expected.sort();
        assert_eq!(ops, expected);

        rt_py
            .as_ref(py)
            .call_method1("vortex_set_isolation_disallowed_ops", (Vec::<u8>::new(),))
            .unwrap();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_isolation_mode_store_blocking() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        let _ = rt_py
            .as_ref(py)
            .call_method1("vortex_set_isolation_mode", (true,));

        // Allow STORE_GLOBAL for this test (STORE_ATTR remains blocked in transform).
        rt_py
            .as_ref(py)
            .call_method1("vortex_set_isolation_disallowed_ops", (Vec::<u8>::new(),))
            .unwrap();

        let module = iris::py::make_module(py).expect("make_module");
        let globals = PyDict::new(py);
        globals.set_item("iris", &module).unwrap();
        py.run(
            "def isolated_write_read():\n    global isolation_test\n    isolation_test = 1\n    return isolation_test\n",
            Some(globals),
            None,
        )
        .unwrap();
        let fn_obj = globals.get_item("isolated_write_read").unwrap().unwrap();

        let transmuted = module
            .as_ref(py)
            .getattr("transmute_function")
            .unwrap()
            .call1((fn_obj,))
            .unwrap();

        module
            .as_ref(py)
            .getattr("set_budget")
            .unwrap()
            .call1((100usize,))
            .unwrap();

        let result: i32 = transmuted.call0().unwrap().extract().unwrap();
        // STORE_GLOBAL writes into isolated globals; LOAD_GLOBAL resolves from same isolated table.
        assert_eq!(result, 1);

        let leaked_in_original_globals = globals.contains("isolation_test").unwrap();
        assert!(!leaked_in_original_globals);

        let _ = rt_py
            .as_ref(py)
            .call_method1("vortex_set_isolation_mode", (false,));
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pyruntime_vortex_genetic_history_pickup_and_reset() {
    let rt_py = Python::with_gil(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module
            .as_ref(py)
            .getattr("PyRuntime")
            .expect("no PyRuntime type");
        runtime_type
            .call0()
            .expect("construct PyRuntime")
            .into_py(py)
    });

    Python::with_gil(|py| {
        let id: u64 = rt_py
            .as_ref(py)
            .call_method1(
                "spawn_py_handler",
                (
                    py.eval("lambda _msg: None", None, None).unwrap(),
                    4usize,
                    false,
                ),
            )
            .unwrap()
            .extract()
            .unwrap();

        let initial: Option<(usize, usize)> = rt_py
            .as_ref(py)
            .call_method1("vortex_get_genetic_history", (id,))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(initial, Some((0, 0)));

        /* exercise rendezvous */
        for _ in 0..40 {
            let _ = rt_py
                .as_ref(py)
                .call_method1("send", (id, pyo3::types::PyBytes::new(py, b"x")));
        }

        // Wait for some processing and possible suspends
        std::thread::sleep(std::time::Duration::from_millis(150));

        let after: Option<(usize, usize)> = rt_py
            .as_ref(py)
            .call_method1("vortex_get_genetic_history", (id,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(after.is_some());
        assert!(after.unwrap().1 >= 1);

        let all: Vec<(u64, usize, usize)> = rt_py
            .as_ref(py)
            .call_method0("vortex_get_all_genetic_history")
            .unwrap()
            .extract()
            .unwrap();
        assert!(all.iter().any(|(pid, _, _)| *pid == id));

        rt_py
            .as_ref(py)
            .call_method0("vortex_reset_genetic_history")
            .unwrap();

        let reset_all: Vec<(u64, usize, usize)> = rt_py
            .as_ref(py)
            .call_method0("vortex_get_all_genetic_history")
            .unwrap()
            .extract()
            .unwrap();
        assert!(reset_all.is_empty());

        rt_py.as_ref(py).call_method1("stop", (id,)).unwrap();
    });
}
