use crate::vortex::ocular::model::TraceEvent;
use crate::vortex::ocular::state::{
    clear_exclude_patterns, clear_include_patterns, get_ts, init_tsc_calibration, reset_stats,
    set_exclude_patterns, set_include_patterns, DEINSTRUMENT_THRESHOLD, EVENT_QUEUE, FREE_QUEUE,
    IS_PRECISE, IS_RUNNING, WORKER_THREAD,
};
use crate::vortex::ocular::telemetry::telemetry_worker;
use crossbeam_queue::ArrayQueue;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::OnceLock;
use std::thread;

const BATCH_SIZE: usize = 1024;

static DISABLE_OBJ: OnceLock<Py<PyAny>> = OnceLock::new();

thread_local! {
    static LOCAL_BATCH: RefCell<Vec<TraceEvent>> = RefCell::new(Vec::with_capacity(BATCH_SIZE));
    static SEEN_CODE_PTRS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    static HOT_OFFSETS: RefCell<HashMap<(usize, i32), u32>> = RefCell::new(HashMap::new());
    static CODE_FILTER: RefCell<CodeFilter> = RefCell::new(CodeFilter::new());
}

struct CodeFilter {
    keys: Box<[usize]>,
    values: Box<[u8]>,
    mask: usize,
}

impl CodeFilter {
    fn new() -> Self {
        let cap = 1 << 13;
        Self {
            keys: vec![0; cap].into_boxed_slice(),
            values: vec![0; cap].into_boxed_slice(),
            mask: cap - 1,
        }
    }

    #[inline(always)]
    fn hash(code_ptr: usize) -> usize {
        let x = code_ptr.wrapping_mul(11400714819323198485u64 as usize);
        x ^ (x >> 16)
    }

    #[inline(always)]
    fn get(&self, code_ptr: usize) -> Option<bool> {
        let mut idx = Self::hash(code_ptr) & self.mask;
        loop {
            let v = self.values[idx];
            if v == 0 {
                return None;
            }
            if self.keys[idx] == code_ptr {
                return Some(v == 2);
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn insert(&mut self, code_ptr: usize, allowed: bool) {
        let mut idx = Self::hash(code_ptr) & self.mask;
        loop {
            if self.values[idx] == 0 || self.keys[idx] == code_ptr {
                self.keys[idx] = code_ptr;
                self.values[idx] = if allowed { 2 } else { 1 };
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

fn resolve_code_filter(code_ptr: usize, filename: &str, func_name: &str) -> bool {
    let (include_ok, exclude_blocked) =
        crate::vortex::ocular::state::with_pattern_set(|pattern_set| {
            let include_ok = pattern_set.include.is_empty()
                || pattern_set
                    .include
                    .iter()
                    .any(|pat| filename.contains(pat) || func_name.contains(pat));

            let exclude_blocked = !pattern_set.exclude.is_empty()
                && pattern_set
                    .exclude
                    .iter()
                    .any(|pat| filename.contains(pat) || func_name.contains(pat));

            (include_ok, exclude_blocked)
        })
        .unwrap_or((true, false));

    let allowed = include_ok && !exclude_blocked;
    CODE_FILTER.with(|filter| {
        filter.borrow_mut().insert(code_ptr, allowed);
    });

    allowed
}

fn code_ptr_allowed(code_ptr: usize) -> Option<bool> {
    CODE_FILTER.with(|filter| filter.borrow().get(code_ptr))
}

#[inline(always)]
fn enqueue_event(event: TraceEvent) {
    LOCAL_BATCH.with(|batch_ref| {
        let mut batch = batch_ref.borrow_mut();
        batch.push(event);
        if batch.len() >= BATCH_SIZE {
            if let Some(queue) = EVENT_QUEUE.get() {
                let new_batch = FREE_QUEUE
                    .get()
                    .and_then(|q| q.pop())
                    .unwrap_or_else(|| Vec::with_capacity(BATCH_SIZE));
                let full_batch = std::mem::replace(&mut *batch, new_batch);
                let _ = queue.push(full_batch);
            } else {
                batch.clear();
            }
        }
    });
}

#[pyfunction]
pub fn instruction_callback(
    py: Python<'_>,
    code: &Bound<'_, PyAny>,
    instruction_offset: i32,
) -> Py<PyAny> {
    let code_ptr = code.as_ptr() as usize;
    let trace_allowed = match code_ptr_allowed(code_ptr) {
        Some(allowed) => allowed,
        None => {
            let filename = code
                .getattr("co_filename")
                .and_then(|f| f.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());
            let func_name = code
                .getattr("co_name")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());
            resolve_code_filter(code_ptr, &filename, &func_name)
        }
    };

    if !trace_allowed {
        return py.None();
    }

    enqueue_event(TraceEvent::Instruction {
        code_ptr,
        lasti: instruction_offset,
        tsc: crate::vortex::ocular::state::read_tsc(),
    });

    let threshold = DEINSTRUMENT_THRESHOLD.load(Ordering::Relaxed);
    let disable = if threshold > 0 {
        HOT_OFFSETS.with(|counts| {
            let mut map = counts.borrow_mut();
            let count = map.entry((code_ptr, instruction_offset)).or_insert(0);
            *count += 1;
            *count > threshold
        })
    } else {
        false
    };

    if disable {
        if let Some(disable_obj) = DISABLE_OBJ.get() {
            return disable_obj.clone_ref(py);
        }
    }

    py.None()
}

#[pyfunction]
pub fn py_start_callback(code: &Bound<'_, PyAny>, instruction_offset: i32) {
    let ts = get_ts();
    let code_ptr = code.as_ptr() as usize;

    let trace_allowed = match code_ptr_allowed(code_ptr) {
        Some(allowed) => allowed,
        None => {
            let filename = code
                .getattr("co_filename")
                .and_then(|f| f.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());
            let func_name = code
                .getattr("co_name")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());

            resolve_code_filter(code_ptr, &filename, &func_name)
        }
    };

    if !trace_allowed {
        return;
    }

    let code_opt = SEEN_CODE_PTRS.with(|seen| {
        if seen.borrow_mut().insert(code_ptr) {
            Some(code.clone().unbind())
        } else {
            None
        }
    });

    enqueue_event(TraceEvent::PyStart {
        code: code_opt,
        code_ptr,
        lasti: instruction_offset,
        ts,
        tsc: crate::vortex::ocular::state::read_tsc(),
    });
}

#[pyfunction]
pub fn jump_callback(
    py: Python<'_>,
    code: &Bound<'_, PyAny>,
    instruction_offset: i32,
    destination_offset: i32,
) -> Py<PyAny> {
    let ts = get_ts();
    let code_ptr = code.as_ptr() as usize;

    let trace_allowed = match code_ptr_allowed(code_ptr) {
        Some(allowed) => allowed,
        None => {
            let filename = code
                .getattr("co_filename")
                .and_then(|f| f.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());
            let func_name = code
                .getattr("co_name")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());
            resolve_code_filter(code_ptr, &filename, &func_name)
        }
    };

    if !trace_allowed {
        return py.None();
    }

    enqueue_event(TraceEvent::Jump {
        code_ptr,
        from_lasti: instruction_offset,
        to_lasti: destination_offset,
        ts,
        tsc: crate::vortex::ocular::state::read_tsc(),
    });

    let threshold = DEINSTRUMENT_THRESHOLD.load(Ordering::Relaxed);
    let disable = if threshold > 0 {
        HOT_OFFSETS.with(|counts| {
            let mut map = counts.borrow_mut();
            let count = map.entry((code_ptr, instruction_offset)).or_insert(0);
            *count += 1;
            *count > threshold
        })
    } else {
        false
    };

    if disable {
        if let Some(disable_obj) = DISABLE_OBJ.get() {
            return disable_obj.clone_ref(py);
        }
    }

    py.None()
}

#[pyfunction]
pub fn py_return_callback(
    code: &Bound<'_, PyAny>,
    _instruction_offset: i32,
    _retval: &Bound<'_, PyAny>,
) {
    let code_ptr = code.as_ptr() as usize;

    let trace_allowed = match code_ptr_allowed(code_ptr) {
        Some(allowed) => allowed,
        None => {
            let filename = code
                .getattr("co_filename")
                .and_then(|f| f.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());
            let func_name = code
                .getattr("co_name")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string());
            resolve_code_filter(code_ptr, &filename, &func_name)
        }
    };

    if !trace_allowed {
        return;
    }

    let ts = get_ts();
    enqueue_event(TraceEvent::PyReturn {
        code_ptr,
        ts,
        tsc: crate::vortex::ocular::state::read_tsc(),
    });
}

#[pyfunction]
pub fn get_tracing_stats(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let summary = crate::vortex::ocular::state::get_summary();
    let out = pyo3::types::PyDict::new(py);
    out.set_item("processed_events", summary.processed_events)?;
    out.set_item("instruction_events", summary.instruction_events)?;
    out.set_item("unique_instruction_sites", summary.unique_instruction_sites)?;
    out.set_item("loop_trace_count", summary.loop_trace_count)?;
    Ok(out.unbind().into_any())
}

#[pyfunction]
pub fn is_tracing_active() -> bool {
    IS_RUNNING.load(Ordering::Relaxed)
}

#[pyfunction]
#[pyo3(signature = (mode="precise", deinstrument_threshold=0, exclude=None, include_only=None))]
pub fn start_tracing(
    py: Python<'_>,
    mode: &str,
    deinstrument_threshold: u32,
    exclude: Option<Vec<String>>,
    include_only: Option<Vec<String>>,
) -> PyResult<()> {
    init_tsc_calibration();
    reset_stats();
    EVENT_QUEUE.get_or_init(|| ArrayQueue::new(10_000));
    FREE_QUEUE.get_or_init(|| {
        let q = ArrayQueue::new(10_000);
        for _ in 0..100 {
            let _ = q.push(Vec::with_capacity(BATCH_SIZE));
        }
        q
    });

    let is_precise = mode == "precise";
    IS_PRECISE.store(is_precise, Ordering::Relaxed);
    DEINSTRUMENT_THRESHOLD.store(deinstrument_threshold, Ordering::Relaxed);
    set_exclude_patterns(exclude.unwrap_or_default());
    set_include_patterns(include_only.unwrap_or_default());

    if !IS_RUNNING.swap(true, Ordering::Relaxed) {
        let handle = thread::spawn(telemetry_worker);
        if let Ok(mut thread_guard) = WORKER_THREAD.lock() {
            *thread_guard = Some(handle);
        }
    }

    let sys = py.import(pyo3::ffi::c_str!("sys"))?;
    let sys_mon = sys.getattr("monitoring")?;

    DISABLE_OBJ.get_or_init(|| {
        sys_mon
            .getattr("DISABLE")
            .map(|o| o.unbind())
            .unwrap_or_else(|_| py.None())
    });

    let tool_id = sys_mon.getattr("DEBUGGER_ID")?;
    let tool_id_obj = tool_id.clone().unbind();
    sys_mon.call_method0("restart_events")?;
    sys_mon.call_method1("use_tool_id", (tool_id_obj.clone_ref(py), "ocular"))?;

    let events = sys_mon.getattr("events")?;
    let instruction_event = events.getattr("INSTRUCTION")?;
    let py_start_event = events.getattr("PY_START")?;
    let jump_event = events.getattr("JUMP")?;
    let branch_event = events.getattr("BRANCH")?;
    let py_return_event = events.getattr("PY_RETURN")?;

    // We can use a dummy module or the parent module here for wrap_pyfunction.
    // However, since we are inside a function and just need a PyFunction,
    // we can use wrap_pyfunction! with the current python handle's module?
    // Actually, in 0.28, we can just use Bound::new(py, PyFunction::...) or similar.
    // But the easiest way is to use wrap_pyfunction! and provide a Bound<PyModule>.
    // Since we don't have one here, we'll create a temporary one.
    let m = PyModule::new(py, "ocular_internal")?;

    let start_cb = pyo3::wrap_pyfunction!(py_start_callback, &m)?;
    sys_mon.call_method1(
        "register_callback",
        (
            tool_id_obj.clone_ref(py),
            py_start_event.clone().unbind(),
            start_cb,
        ),
    )?;

    let jump_cb = pyo3::wrap_pyfunction!(jump_callback, &m)?;
    let jump_cb_obj = jump_cb.clone().unbind();
    sys_mon.call_method1(
        "register_callback",
        (
            tool_id_obj.clone_ref(py),
            jump_event.clone().unbind(),
            jump_cb_obj.clone_ref(py),
        ),
    )?;
    sys_mon.call_method1(
        "register_callback",
        (
            tool_id_obj.clone_ref(py),
            branch_event.clone().unbind(),
            jump_cb,
        ),
    )?;

    let return_cb = pyo3::wrap_pyfunction!(py_return_callback, &m)?;
    sys_mon.call_method1(
        "register_callback",
        (
            tool_id_obj.clone_ref(py),
            py_return_event.clone().unbind(),
            return_cb,
        ),
    )?;

    let mut combined_events: i32 = py_start_event.extract::<i32>()?
        | jump_event.extract::<i32>()?
        | branch_event.extract::<i32>()?
        | py_return_event.extract::<i32>()?;

    if is_precise {
        let inst_cb = pyo3::wrap_pyfunction!(instruction_callback, &m)?;
        sys_mon.call_method1(
            "register_callback",
            (
                tool_id_obj.clone_ref(py),
                instruction_event.clone().unbind(),
                inst_cb,
            ),
        )?;
        combined_events |= instruction_event.extract::<i32>()?;
    } else {
        sys_mon.call_method1(
            "register_callback",
            (
                tool_id_obj.clone_ref(py),
                instruction_event.clone().unbind(),
                py.None(),
            ),
        )?;
    }

    sys_mon.call_method1("set_events", (tool_id, combined_events))?;

    Ok(())
}

#[pyfunction]
pub fn stop_tracing(py: Python<'_>) -> PyResult<()> {
    let sys = py.import(pyo3::ffi::c_str!("sys"))?;
    let sys_mon = sys.getattr("monitoring")?;
    let tool_id = sys_mon.getattr("DEBUGGER_ID")?;
    let tool_id_obj = tool_id.clone().unbind();

    let events = sys_mon.getattr("events")?;
    let instruction_event = events.getattr("INSTRUCTION")?;
    let py_start_event = events.getattr("PY_START")?;
    let jump_event = events.getattr("JUMP")?;
    let branch_event = events.getattr("BRANCH")?;
    let py_return_event = events.getattr("PY_RETURN")?;

    sys_mon.call_method1("set_events", (tool_id_obj.clone_ref(py), 0))?;

    sys_mon.call_method1(
        "register_callback",
        (tool_id_obj.clone_ref(py), instruction_event, py.None()),
    )?;
    sys_mon.call_method1(
        "register_callback",
        (tool_id_obj.clone_ref(py), py_start_event, py.None()),
    )?;
    sys_mon.call_method1(
        "register_callback",
        (tool_id_obj.clone_ref(py), jump_event, py.None()),
    )?;
    sys_mon.call_method1(
        "register_callback",
        (tool_id_obj.clone_ref(py), branch_event, py.None()),
    )?;
    sys_mon.call_method1(
        "register_callback",
        (tool_id_obj.clone_ref(py), py_return_event, py.None()),
    )?;

    sys_mon.call_method1("free_tool_id", (tool_id,))?;

    LOCAL_BATCH.with(|batch_ref| {
        let mut batch = batch_ref.borrow_mut();
        if !batch.is_empty() {
            if let Some(queue) = EVENT_QUEUE.get() {
                let new_batch = FREE_QUEUE.get().and_then(|q| q.pop()).unwrap_or_default();
                let final_batch = std::mem::replace(&mut *batch, new_batch);
                let _ = queue.push(final_batch);
            }
        }
    });

    SEEN_CODE_PTRS.with(|seen| {
        seen.borrow_mut().clear();
    });

    HOT_OFFSETS.with(|offsets| {
        offsets.borrow_mut().clear();
    });

    if IS_RUNNING.swap(false, Ordering::Relaxed) {
        if let Ok(mut thread_guard) = WORKER_THREAD.lock() {
            if let Some(handle) = thread_guard.take() {
                py.detach(|| {
                    let _ = handle.join();
                });
            }
        }
    }

    clear_exclude_patterns();
    clear_include_patterns();
    Ok(())
}
