#![allow(non_local_definitions)]

use crate::vortex::vortex_bytecode::{
    decode_wordcode, encode_wordcode, evaluate_rewrite_compatibility,
    instrument_with_probe_with_sites, opcode_meta, probe_injection_sites, probe_instructions,
    quickening_support, read_exception_entries, validate_probe_compatibility, verify_cache_layout,
    verify_exception_handler_targets, verify_exception_table_invariants, verify_stacksize_minimum,
};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict};
use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

pyo3::create_exception!(iris, VortexSuspend, pyo3::exceptions::PyException);

static BUDGET: AtomicUsize = AtomicUsize::new(0);
static ISOLATION_MODE: AtomicBool = AtomicBool::new(false);
static ISOLATION_DISALLOWED_OPS: Lazy<Mutex<HashSet<u8>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));
const MAX_PATCHED_CODE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
struct GuardTelemetry {
    mode: String,
    reason: String,
    py_minor: i32,
    rewrite_attempted: bool,
    rewrite_applied: bool,
}

impl Default for GuardTelemetry {
    fn default() -> Self {
        GuardTelemetry {
            mode: "unset".to_string(),
            reason: "none".to_string(),
            py_minor: -1,
            rewrite_attempted: false,
            rewrite_applied: false,
        }
    }
}

static GUARD_TELEMETRY: Lazy<Mutex<GuardTelemetry>> =
    Lazy::new(|| Mutex::new(GuardTelemetry::default()));

fn transmute_log(msg: &str) {
    eprintln!("{}", msg);
    let _ = std::io::stderr().flush();
}

fn set_guard_telemetry(mode: &str, reason: &str, py_minor: i32, attempted: bool, applied: bool) {
    if let Ok(mut g) = GUARD_TELEMETRY.lock() {
        g.mode = mode.to_string();
        g.reason = reason.to_string();
        g.py_minor = py_minor;
        g.rewrite_attempted = attempted;
        g.rewrite_applied = applied;
    }
}

fn test_hook_enabled(py: Python<'_>, key: &str) -> bool {
    let locals = PyDict::new(py);
    if locals.set_item("_iris_key", key).is_err() {
        return false;
    }
    py.eval(
        pyo3::ffi::c_str!("__import__('os').environ.get(_iris_key, '0') == '1'"),
        None,
        Some(&locals),
    )
    .and_then(|v| v.extract::<bool>())
    .unwrap_or(false)
}

#[pyfunction]
pub fn get_guard_status(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let g = GUARD_TELEMETRY
        .lock()
        .map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("vortex/guard-status: lock poisoned")
        })?
        .clone();

    let d = PyDict::new(py);
    d.set_item("mode", g.mode)?;
    d.set_item("reason", g.reason)?;
    d.set_item("py_minor", g.py_minor)?;
    d.set_item("rewrite_attempted", g.rewrite_attempted)?;
    d.set_item("rewrite_applied", g.rewrite_applied)?;
    Ok(d.unbind().into_any())
}

#[pyfunction]
pub fn _vortex_check() -> PyResult<()> {
    let current = BUDGET.load(Ordering::Relaxed);
    if current == 0 {
        return Err(VortexSuspend::new_err("budget exhausted"));
    }
    BUDGET.store(current - 1, Ordering::Relaxed);
    Ok(())
}

#[pyfunction]
pub fn set_budget(budget: usize) {
    BUDGET.store(budget, Ordering::Relaxed);
}

#[pyfunction]
pub fn set_isolation_mode(enabled: bool) {
    ISOLATION_MODE.store(enabled, Ordering::Relaxed);
}

#[pyfunction]
pub fn get_isolation_mode() -> bool {
    ISOLATION_MODE.load(Ordering::Relaxed)
}

#[pyfunction]
pub fn set_isolation_disallowed_ops(ops: Vec<u8>) {
    let mut guard = ISOLATION_DISALLOWED_OPS.lock().unwrap();
    guard.clear();
    for op in ops {
        guard.insert(op);
    }
}

#[pyfunction]
pub fn get_isolation_disallowed_ops() -> Vec<u8> {
    let guard = ISOLATION_DISALLOWED_OPS.lock().unwrap();
    guard.iter().copied().collect()
}

#[pyfunction]
pub fn transmute_function(py: Python<'_>, py_func: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    fn opcode_name(py: Python<'_>, op: u8) -> String {
        py.import(pyo3::ffi::c_str!("opcode"))
            .and_then(|m| m.getattr("opname"))
            .and_then(|names| names.get_item(op as usize))
            .and_then(|name| name.extract::<String>())
            .unwrap_or_else(|_| format!("op{}", op))
    }

    let py_minor: i32 = py
        .eval(
            pyo3::ffi::c_str!("__import__('sys').version_info.minor"),
            None,
            None,
        )
        .and_then(|v| v.extract())
        .unwrap_or(99);

    let fn_name = py_func
        .getattr("__name__")
        .and_then(|n| n.extract::<String>())
        .unwrap_or_else(|_| "<unknown>".to_string());
    transmute_log(&format!(
        "[Ocular][Transmute] begin fn={} py_minor={}",
        fn_name, py_minor
    ));

    let code = py_func
        .getattr("__code__")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/code: {e}")))?;
    let code_ptr = code.as_ptr() as usize;
    let raw: Bound<'_, PyBytes> = code
        .getattr("co_code")?
        .extract()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/co_code: {e}")))?;
    let raw_bytes = raw.as_bytes();
    let original_stack_size: usize = code.getattr("co_stacksize")?.extract().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/co_stacksize: {e}"))
    })?;

    let globals_any = py_func
        .getattr("__globals__")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals: {e}")))?;
    let globals = globals_any.cast::<PyDict>().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals-cast: {e}"))
    })?;
    let local_mod = match py
        .import(pyo3::ffi::c_str!("sys"))
        .and_then(|s| s.getattr("modules"))
        .and_then(|mods| mods.get_item("iris"))
    {
        Ok(m) => m,
        _ => match globals.get_item("iris")? {
            Some(m) => m,
            None => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "vortex/module-lookup: iris missing",
                ))
            }
        },
    };
    let check_fn = local_mod
        .getattr("_vortex_check")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/check-fn: {e}")))?;
    globals.set_item("_vortex_check", check_fn).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals-inject: {e}"))
    })?;

    // Primary RFC path: bytecode-level shadow clone with capability checks.
    let meta = match opcode_meta(py) {
        Ok(m) => m,
        Err(_) => {
            set_guard_telemetry(
                "fallback",
                "opcode_metadata_unavailable",
                py_minor,
                false,
                false,
            );
            return fallback_with_log(py, py_func, &fn_name, "opcode metadata unavailable");
        }
    };

    let quickening = match quickening_support(py) {
        Ok(q) => q,
        Err(_) => {
            set_guard_telemetry(
                "fallback",
                "quickening_metadata_unavailable",
                py_minor,
                false,
                false,
            );
            return fallback_with_log(py, py_func, &fn_name, "quickening metadata unavailable");
        }
    };

    if let Err(reason) = evaluate_rewrite_compatibility(raw_bytes, meta.extended_arg, &quickening) {
        set_guard_telemetry("fallback", reason, py_minor, false, false);
        return fallback_with_log(py, py_func, &fn_name, reason);
    }

    if verify_stacksize_minimum(original_stack_size).is_err() {
        let original_preview = decode_wordcode(raw_bytes, meta.extended_arg);
        if let Ok(preview_probe) = probe_instructions(py, meta.extended_arg) {
            let preview_sites = probe_injection_sites(&original_preview, &meta);
            let preview_probe_desc = preview_probe
                .iter()
                .map(|ins| format!("{}({})", opcode_name(py, ins.op), ins.arg))
                .collect::<Vec<_>>()
                .join(" -> ");
            transmute_log(&format!(
                "[Ocular][Transmute] preview-only fn={} co_stacksize={} original_uops={} probe_uops={} injection_sites={} sites={:?}",
                fn_name,
                original_stack_size,
                original_preview.len(),
                preview_probe.len(),
                preview_sites.len(),
                preview_sites
            ));
            transmute_log(&format!(
                "[Ocular][Transmute] preview_injected_probe={}",
                preview_probe_desc
            ));
        }

        set_guard_telemetry(
            "fallback",
            "stack_depth_invariant_failed",
            py_minor,
            false,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "stack depth invariant failed");
    }

    let original_entries = match read_exception_entries(py, code.as_any()) {
        Ok(entries) => entries,
        Err(_) => {
            set_guard_telemetry(
                "fallback",
                "exception_table_metadata_unavailable",
                py_minor,
                false,
                false,
            );
            return fallback_with_log(
                py,
                py_func,
                &fn_name,
                "exception table metadata unavailable",
            );
        }
    };

    if test_hook_enabled(
        py,
        "IRIS_VORTEX_TEST_FORCE_EXCEPTION_TABLE_METADATA_UNAVAILABLE",
    ) {
        set_guard_telemetry(
            "fallback",
            "exception_table_metadata_unavailable",
            py_minor,
            false,
            false,
        );
        return fallback_with_log(
            py,
            py_func,
            &fn_name,
            "exception table metadata unavailable",
        );
    }

    if verify_exception_table_invariants(
        &original_entries,
        raw_bytes.len() / 2,
        original_stack_size,
    )
    .is_err()
        || test_hook_enabled(py, "IRIS_VORTEX_TEST_FORCE_EXCEPTION_TABLE_INVALID")
    {
        set_guard_telemetry(
            "fallback",
            "exception_table_invalid",
            py_minor,
            false,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "exception table invalid");
    }

    let original = decode_wordcode(raw_bytes, meta.extended_arg);
    if verify_exception_handler_targets(&original_entries, &original, &quickening).is_err() {
        set_guard_telemetry(
            "fallback",
            "exception_table_invalid",
            py_minor,
            false,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "exception table invalid");
    }

    set_guard_telemetry("rewrite", "attempt", py_minor, true, false);
    let force_patched_exception_invalid =
        test_hook_enabled(py, "IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_INVALID");
    if force_patched_exception_invalid {
        set_guard_telemetry(
            "fallback",
            "patched_exception_table_invalid",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "patched exception table invalid");
    }
    if test_hook_enabled(py, "IRIS_VORTEX_TEST_FORCE_CODE_REPLACE_FAILED") {
        set_guard_telemetry("fallback", "code_replace_failed", py_minor, true, false);
        return fallback_with_log(py, py_func, &fn_name, "code replace failed");
    }
    if test_hook_enabled(py, "IRIS_VORTEX_TEST_FORCE_TYPES_MODULE_UNAVAILABLE") {
        set_guard_telemetry(
            "fallback",
            "types_module_unavailable",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "types module unavailable");
    }
    if test_hook_enabled(py, "IRIS_VORTEX_TEST_FORCE_SHADOW_CONSTRUCTION_FAILED") {
        set_guard_telemetry(
            "fallback",
            "shadow_function_construction_failed",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "shadow function construction failed");
    }
    if test_hook_enabled(py, "IRIS_VORTEX_TEST_FORCE_PROBE_INSTRUMENTATION_FAILED") {
        set_guard_telemetry(
            "fallback",
            "probe_instrumentation_failed",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "probe instrumentation failed");
    }
    if test_hook_enabled(
        py,
        "IRIS_VORTEX_TEST_FORCE_PATCHED_STACK_METADATA_UNAVAILABLE",
    ) {
        set_guard_telemetry(
            "fallback",
            "patched_stack_metadata_unavailable",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "patched stack metadata unavailable");
    }
    if test_hook_enabled(
        py,
        "IRIS_VORTEX_TEST_FORCE_PATCHED_EXCEPTION_TABLE_METADATA_UNAVAILABLE",
    ) {
        set_guard_telemetry(
            "fallback",
            "patched_exception_table_metadata_unavailable",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(
            py,
            py_func,
            &fn_name,
            "patched exception table metadata unavailable",
        );
    }

    let probe = match probe_instructions(py, meta.extended_arg) {
        Ok(v) => v,
        Err(_) => {
            set_guard_telemetry("fallback", "probe_extraction_failed", py_minor, true, false);
            return fallback_with_log(py, py_func, &fn_name, "probe extraction failed");
        }
    };

    if test_hook_enabled(py, "IRIS_VORTEX_TEST_FORCE_PROBE_EXTRACTION_FAILED") {
        set_guard_telemetry("fallback", "probe_extraction_failed", py_minor, true, false);
        return fallback_with_log(py, py_func, &fn_name, "probe extraction failed");
    }

    if let Err(reason) = validate_probe_compatibility(&probe, &quickening) {
        set_guard_telemetry("fallback", reason, py_minor, true, false);
        return fallback_with_log(py, py_func, &fn_name, reason);
    }

    // Ocular-style visibility into transmutation: show exactly what got injected and where.
    let base_injection_sites = probe_injection_sites(&original, &meta);

    let mut offset_to_idx: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut ext_acc: u32 = 0;
    let mut i = 0usize;
    let mut ins_idx = 0usize;
    while i + 1 < raw_bytes.len() {
        let op = raw_bytes[i];
        let arg = raw_bytes[i + 1] as u32;
        if op == meta.extended_arg {
            ext_acc = (ext_acc << 8) | arg;
            i += 2;
            continue;
        }
        let _ = ext_acc;
        offset_to_idx.insert(i as i32, ins_idx);
        ins_idx = ins_idx.saturating_add(1);
        ext_acc = 0;
        i += 2;
    }

    let observed_offsets = crate::vortex::ocular::state::get_observed_offsets_for_code(code_ptr);
    let mut runtime_sites = Vec::new();
    for offset in observed_offsets {
        if let Some(idx) = offset_to_idx.get(&offset).copied() {
            runtime_sites.push(idx);
        }
    }
    runtime_sites.sort_unstable();
    runtime_sites.dedup();

    let mut merged_sites = base_injection_sites.clone();
    merged_sites.extend(runtime_sites.iter().copied());
    merged_sites.sort_unstable();
    merged_sites.dedup();

    let probe_desc = probe
        .iter()
        .map(|ins| format!("{}({})", opcode_name(py, ins.op), ins.arg))
        .collect::<Vec<_>>()
        .join(" -> ");
    let sites_preview = if merged_sites.len() > 16 {
        format!(
            "{:?} ... (+{} more)",
            &merged_sites[..16],
            merged_sites.len() - 16
        )
    } else {
        format!("{:?}", merged_sites)
    };
    transmute_log(&format!(
        "[Ocular][Transmute] fn={} original_uops={} probe_uops={} injection_sites={} runtime_guided_sites={} sites={}",
        fn_name,
        original.len(),
        probe.len(),
        merged_sites.len(),
        runtime_sites.len(),
        sites_preview
    ));
    transmute_log(&format!(
        "[Ocular][Transmute] injected_probe={}",
        probe_desc
    ));

    let patched = match instrument_with_probe_with_sites(&original, &probe, &meta, &runtime_sites) {
        Ok(v) => v,
        Err(_) => {
            set_guard_telemetry(
                "fallback",
                "probe_instrumentation_failed",
                py_minor,
                true,
                false,
            );
            return fallback_with_log(py, py_func, &fn_name, "probe instrumentation failed");
        }
    };

    if verify_cache_layout(&patched, &quickening).is_err() {
        set_guard_telemetry(
            "fallback",
            "patched_cache_layout_invalid",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "patched cache layout invalid");
    }

    let final_patched = if ISOLATION_MODE.load(Ordering::Relaxed) {
        let disallowed_ops = ISOLATION_DISALLOWED_OPS.lock().unwrap();
        match crate::vortex::vortex_bytecode::apply_isolation_transform(
            &patched,
            py,
            Some(&*disallowed_ops),
        ) {
            Ok(isolated) => {
                if verify_cache_layout(&isolated, &quickening).is_err() {
                    set_guard_telemetry(
                        "fallback",
                        "isolation_cache_layout_invalid",
                        py_minor,
                        true,
                        false,
                    );
                    return fallback_with_log(
                        py,
                        py_func,
                        &fn_name,
                        "isolation cache layout invalid",
                    );
                }
                isolated
            }
            Err(_) => {
                set_guard_telemetry(
                    "fallback",
                    "isolation_transform_failed",
                    py_minor,
                    true,
                    false,
                );
                return fallback_with_log(py, py_func, &fn_name, "isolation transform failed");
            }
        }
    } else {
        patched
    };

    transmute_log(&format!(
        "[Ocular][Transmute] fn={} patched_uops={} delta_uops={}",
        fn_name,
        final_patched.len(),
        final_patched.len() as isize - original.len() as isize
    ));

    let final_raw = encode_wordcode(&final_patched, meta.extended_arg);
    if final_raw.len() > MAX_PATCHED_CODE_BYTES {
        set_guard_telemetry("fallback", "patched_code_too_large", py_minor, true, false);
        return fallback_with_log(py, py_func, &fn_name, "patched code too large");
    }

    let kwargs = [("co_code", PyBytes::new(py, &final_raw))].into_py_dict(py)?;
    let new_code = match code.call_method("replace", (), Some(&kwargs)) {
        Ok(v) => v,
        Err(_) => {
            set_guard_telemetry("fallback", "code_replace_failed", py_minor, true, false);
            return fallback_with_log(py, py_func, &fn_name, "code replace failed");
        }
    };

    let patched_stack_size: usize = new_code.getattr("co_stacksize")?.extract()?;
    let patched_entries = match read_exception_entries(py, new_code.as_any()) {
        Ok(v) => v,
        Err(_) => {
            set_guard_telemetry(
                "fallback",
                "patched_exception_table_metadata_unavailable",
                py_minor,
                true,
                false,
            );
            return fallback_with_log(
                py,
                py_func,
                &fn_name,
                "patched exception table metadata unavailable",
            );
        }
    };
    if verify_exception_table_invariants(&patched_entries, final_raw.len() / 2, patched_stack_size)
        .is_err()
    {
        set_guard_telemetry(
            "fallback",
            "patched_exception_table_invalid",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "patched exception table invalid");
    }
    if verify_exception_handler_targets(&patched_entries, &final_patched, &quickening).is_err() {
        set_guard_telemetry(
            "fallback",
            "patched_exception_table_invalid",
            py_minor,
            true,
            false,
        );
        return fallback_with_log(py, py_func, &fn_name, "patched exception table invalid");
    }

    let types_mod = match py.import(pyo3::ffi::c_str!("types")) {
        Ok(v) => v,
        Err(_) => {
            set_guard_telemetry(
                "fallback",
                "types_module_unavailable",
                py_minor,
                true,
                false,
            );
            return fallback_with_log(py, py_func, &fn_name, "types module unavailable");
        }
    };

    let func_globals = if ISOLATION_MODE.load(Ordering::Relaxed) {
        let locals2 = PyDict::new(py);
        locals2.set_item("base_globals", globals)?;
        // Isolation uses a detached globals dict so STORE_GLOBAL/STORE_NAME mutate only
        // this shadow environment, never the original module globals.
        py.run(
            pyo3::ffi::c_str!("isolated_globals = dict(base_globals)"),
            None,
            Some(&locals2),
        )?;
        locals2
            .get_item("isolated_globals")?
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("vortex/isolated-globals: missing result")
            })?
            .cast_into::<PyDict>()?
    } else {
        globals.clone()
    };

    let shadow = match types_mod.getattr("FunctionType").and_then(|ctor| {
        ctor.call1((
            new_code,
            func_globals,
            py_func.getattr("__name__")?,
            py_func.getattr("__defaults__")?,
            py_func.getattr("__closure__")?,
        ))
    }) {
        Ok(v) => v,
        Err(_) => {
            set_guard_telemetry(
                "fallback",
                "shadow_function_construction_failed",
                py_minor,
                true,
                false,
            );
            return fallback_with_log(py, py_func, &fn_name, "shadow function construction failed");
        }
    };

    if let Ok(kwdefaults) = py_func.getattr("__kwdefaults__") {
        let _ = shadow.setattr("__kwdefaults__", kwdefaults);
    }
    set_guard_telemetry("rewrite", "applied", py_minor, true, true);
    Ok(shadow.unbind().into_any())
}

fn fallback_with_log(
    py: Python<'_>,
    py_func: &Bound<'_, PyAny>,
    fn_name: &str,
    reason: &str,
) -> PyResult<Py<PyAny>> {
    transmute_log(&format!(
        "[Ocular][Transmute] fallback fn={} reason={}",
        fn_name, reason
    ));
    fallback_shadow(py, py_func, reason)
}

fn fallback_shadow(
    py: Python<'_>,
    py_func: &Bound<'_, PyAny>,
    _reason: &str,
) -> PyResult<Py<PyAny>> {
    let globals_any = py_func
        .getattr("__globals__")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals: {e}")))?;
    let globals = globals_any.cast::<PyDict>().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/globals-cast: {e}"))
    })?;

    let locals = PyDict::new(py);
    locals.set_item("fn", py_func)?;
    locals.set_item("isolation_mode", ISOLATION_MODE.load(Ordering::Relaxed))?;
    py.run(
        pyo3::ffi::c_str!(
            r#"
def _iris_make_shadow(fn, isolation_mode=False):
    import types
    import sys

    target_fn = fn
    if isolation_mode:
        isolated_globals = dict(fn.__globals__)
        target_fn = types.FunctionType(
            fn.__code__,
            isolated_globals,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__,
        )
        if hasattr(fn, "__kwdefaults__"):
            target_fn.__kwdefaults__ = fn.__kwdefaults__

    target_code = target_fn.__code__

    def _trace(frame, event, arg):
        if frame.f_code is not target_code:
            return _trace
        if event == "call":
            return _trace
        if event == "line":
            _vortex_check()
        return _trace

    def _wrapped(*a, **k):
        old = sys.gettrace()
        sys.settrace(_trace)
        try:
            return target_fn(*a, **k)
        finally:
            sys.settrace(old)

    return _wrapped

shadow = _iris_make_shadow(fn, isolation_mode)
"#
        ),
        Some(globals),
        Some(&locals),
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("vortex/shadow-fallback: {e}"))
    })?;
    let shadow = locals.get_item("shadow")?.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("vortex/shadow-fallback: missing shadow")
    })?;

    Ok(shadow.unbind().into_any())
}

pub fn init_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VortexSuspend", m.py().get_type::<VortexSuspend>())?;
    m.add_function(wrap_pyfunction!(_vortex_check, m)?)?;
    m.add_function(wrap_pyfunction!(set_budget, m)?)?;
    m.add_function(wrap_pyfunction!(get_guard_status, m)?)?;
    m.add_function(wrap_pyfunction!(transmute_function, m)?)?;
    m.add_function(wrap_pyfunction!(set_isolation_mode, m)?)?;
    m.add_function(wrap_pyfunction!(get_isolation_mode, m)?)?;
    m.add_function(wrap_pyfunction!(set_isolation_disallowed_ops, m)?)?;
    m.add_function(wrap_pyfunction!(get_isolation_disallowed_ops, m)?)?;
    crate::vortex::ocular::init_py(m)?;
    Ok(())
}
