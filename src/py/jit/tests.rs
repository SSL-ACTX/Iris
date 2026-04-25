// src/py/jit/tests.rs
//! Unit tests formerly embedded in `mod.rs`.  Now separated for clarity.

use super::*;
use crate::py::jit::codegen::execute_jit_func;
use crate::py::jit::codegen::{
    compile_jit, compile_jit_quantum_variant, lookup_named_jit, quantum_profile_snapshot,
    quantum_seed_preferred_index, register_named_jit, register_quantum_jit, resolve_symbol_alias,
    seed_quantum_profile, JitReturnType, QuantumProfileSeed, SymbolAlias,
};
use cranelift::prelude::settings;
use cranelift::prelude::Configurable;
// use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

// Don't ever regress next time :(
fn quantum_shared_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

fn lock_quantum_shared_test() -> std::sync::MutexGuard<'static, ()> {
    match quantum_shared_test_lock().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[test]
fn compile_jit_basic_math() {
    let args = ["a".to_string(), "b".to_string()];
    let entry = compile_jit("a + b", &args).expect("should compile");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let values = [1.5, 2.5];
    let result = f(values.as_ptr());
    assert_eq!(result, 4.0);
}

#[test]
#[cfg(feature = "pyo3")]
fn compile_jit_int_return() {
    use crate::py::jit::codegen::{
        compile_jit_with_return_type, execute_jit_func, register_jit, JitReturnType,
    };
    use pyo3::Python;

    let args = ["x".to_string()];
    let entry = compile_jit_with_return_type("x + 1", &args, JitReturnType::Int).unwrap();
    register_jit(999_999, entry.clone());

    Python::attach(|py| {
        let tup = pyo3::types::PyTuple::new(py, [1.0_f64]).unwrap();
        let obj = execute_jit_func(py, &entry, &tup).unwrap();
        let val: i64 = obj.bind(py).extract().unwrap();
        assert_eq!(val, 2);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn compile_jit_quantum_variants() {
    let args = ["x".to_string(), "y".to_string()];
    let entries = compile_jit_quantum(
        "x + y",
        &args,
        crate::py::jit::codegen::JitReturnType::Float,
    );
    assert!(
        !entries.is_empty(),
        "quantum compile should produce at least one variant"
    );
    for entry in entries {
        assert_eq!(entry.arg_count, 2);
    }
}

#[test]
fn quantum_profile_snapshot_and_seed_roundtrip() {
    use crate::py::jit::codegen::{
        quantum_profile_snapshot, seed_quantum_profile, QuantumProfileSeed,
    };

    let args = ["x".to_string()];
    let entries = compile_jit_quantum(
        "x + 1",
        &args,
        crate::py::jit::codegen::JitReturnType::Float,
    );
    assert!(!entries.is_empty());

    let func_key = 987_654;
    register_quantum_jit(func_key, entries);

    let before = quantum_profile_snapshot(func_key).expect("snapshot should exist");
    assert!(!before.is_empty());
    assert!(before.iter().all(|s| s.runs == 0));

    let seeded = vec![QuantumProfileSeed {
        index: 0,
        ewma_ns: 1_500_000.0,
        runs: 42,
        failures: 1,
    }];
    assert!(seed_quantum_profile(func_key, &seeded));

    let after = quantum_profile_snapshot(func_key).expect("snapshot should still exist");
    assert_eq!(after[0].runs, 42);
    assert_eq!(after[0].failures, 1);
    assert!((after[0].ewma_ns - 1_500_000.0).abs() < f64::EPSILON);
}

#[test]
fn quantum_seed_prefers_lowest_latency_variant() {
    use crate::py::jit::codegen::{seed_quantum_profile, QuantumProfileSeed};

    let func_key = 9_100_001;
    assert!(seed_quantum_profile(
        func_key,
        &[
            QuantumProfileSeed {
                index: 0,
                ewma_ns: 1500.0,
                runs: 20,
                failures: 0,
            },
            QuantumProfileSeed {
                index: 1,
                ewma_ns: 300.0,
                runs: 20,
                failures: 0,
            },
            QuantumProfileSeed {
                index: 2,
                ewma_ns: 900.0,
                runs: 20,
                failures: 0,
            },
        ],
    ));

    assert_eq!(quantum_seed_preferred_index(func_key), Some(1));
}

#[test]
fn quantum_seed_penalizes_failure_heavily() {
    use crate::py::jit::codegen::{seed_quantum_profile, QuantumProfileSeed};

    let func_key = 9_100_002;
    assert!(seed_quantum_profile(
        func_key,
        &[
            QuantumProfileSeed {
                index: 0,
                ewma_ns: 1200.0,
                runs: 20,
                failures: 0,
            },
            QuantumProfileSeed {
                index: 1,
                ewma_ns: 200.0,
                runs: 20,
                failures: 30,
            },
        ],
    ));

    assert_eq!(quantum_seed_preferred_index(func_key), Some(0));
}

#[test]
fn quantum_seed_prefers_scalarfallback_when_samples_are_thin() {
    let func_key = 9_100_004;
    assert!(seed_quantum_profile(
        func_key,
        &[
            QuantumProfileSeed {
                index: 0,
                ewma_ns: 200.0,
                runs: 1,
                failures: 0,
            },
            QuantumProfileSeed {
                index: 1,
                ewma_ns: 350.0,
                runs: 1,
                failures: 0,
            },
            QuantumProfileSeed {
                index: 2,
                ewma_ns: 180.0,
                runs: 1,
                failures: 0,
            },
        ],
    ));

    assert_eq!(
        quantum_seed_preferred_index(func_key),
        Some(1),
        "expected sparse sample warm-start to prefer ScalarFallback"
    );
}

#[test]
fn quantum_snapshot_preserves_canonical_variant_id_for_single_warm_winner() {
    let args = ["x".to_string()];
    let entry = compile_jit_quantum_variant("x + 1", &args, JitReturnType::Float, 1)
        .expect("compile scalar-fallback variant");

    let func_key = 9_100_003;
    register_quantum_jit(func_key, vec![entry]);

    assert!(seed_quantum_profile(
        func_key,
        &[QuantumProfileSeed {
            index: 0,
            ewma_ns: 750.0,
            runs: 5,
            failures: 0,
        }],
    ));

    let snapshot = quantum_profile_snapshot(func_key).expect("snapshot should exist");
    assert_eq!(snapshot.len(), 1);
    assert_eq!(
        snapshot[0].index, 1,
        "expected persisted row to keep canonical ScalarFallback index"
    );
}

#[test]
fn quantum_seed_applies_to_single_variant_by_canonical_index() {
    let args = ["x".to_string()];
    let entry = compile_jit_quantum_variant("x + 1", &args, JitReturnType::Float, 1)
        .expect("compile scalar-fallback variant");

    let func_key = 9_100_005;
    register_quantum_jit(func_key, vec![entry]);

    assert!(seed_quantum_profile(
        func_key,
        &[QuantumProfileSeed {
            index: 1,
            ewma_ns: 512.0,
            runs: 9,
            failures: 2,
        }],
    ));

    let snapshot = quantum_profile_snapshot(func_key).expect("snapshot should exist");
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].index, 1);
    assert_eq!(snapshot[0].runs, 9);
    assert_eq!(snapshot[0].failures, 2);
    assert!((snapshot[0].ewma_ns - 512.0).abs() < f64::EPSILON);
}

#[test]
fn quantum_compile_budget_blocks_when_exhausted() {
    use std::env;

    let _guard = lock_quantum_shared_test();

    reset_quantum_control_state();
    env::set_var("IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS", "1");
    env::set_var("IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS", "1000000");

    let now = 1_000_000_u64;
    assert!(quantum_compile_may_run(11, now));
    record_quantum_compile_attempt(11, now, 10, true);
    assert!(!quantum_compile_may_run(11, now + 1));

    env::remove_var("IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS");
}

#[test]
fn quantum_cooldown_backoff_blocks_and_recovers() {
    use std::env;

    let _guard = lock_quantum_shared_test();

    reset_quantum_control_state();
    env::set_var("IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS", "100");
    env::set_var("IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS", "1000");

    let now = 2_000_u64;
    assert!(quantum_compile_may_run(22, now));
    record_quantum_compile_attempt(22, now, 5, false);
    assert!(!quantum_compile_may_run(22, now + 50));
    assert!(quantum_compile_may_run(22, now + 100));

    record_quantum_compile_attempt(22, now + 100, 5, false);
    assert!(!quantum_compile_may_run(22, now + 299));
    assert!(quantum_compile_may_run(22, now + 300));

    record_quantum_compile_attempt(22, now + 300, 5, true);
    assert!(quantum_compile_may_run(22, now + 301));

    env::remove_var("IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS");
}

#[test]
fn quantum_stability_score_tracks_profile_consistency() {
    use crate::py::jit::codegen::{
        quantum_stability_for, seed_quantum_profile, QuantumProfileSeed,
    };
    use std::env;

    let _guard = lock_quantum_shared_test();
    reset_quantum_control_state();

    env::set_var("IRIS_JIT_QUANTUM_STABILITY_MIN_RUNS", "1");

    let args = ["x".to_string(), "y".to_string()];
    let entries = compile_jit_quantum(
        "x + y",
        &args,
        crate::py::jit::codegen::JitReturnType::Float,
    );
    assert!(!entries.is_empty());
    let func_key = 77_001;
    register_quantum_jit(func_key, entries);

    assert!(seed_quantum_profile(
        func_key,
        &[
            QuantumProfileSeed {
                index: 0,
                ewma_ns: 1000.0,
                runs: 20,
                failures: 0
            },
            QuantumProfileSeed {
                index: 1,
                ewma_ns: 1100.0,
                runs: 20,
                failures: 0
            },
        ],
    ));
    let stable = quantum_stability_for(func_key).unwrap();

    assert!(seed_quantum_profile(
        func_key,
        &[
            QuantumProfileSeed {
                index: 0,
                ewma_ns: 200.0,
                runs: 20,
                failures: 0
            },
            QuantumProfileSeed {
                index: 1,
                ewma_ns: 10_000.0,
                runs: 20,
                failures: 30
            },
        ],
    ));
    let unstable = quantum_stability_for(func_key).unwrap();

    assert!(
        stable > unstable,
        "expected stable profile score to be greater"
    );
    env::remove_var("IRIS_JIT_QUANTUM_STABILITY_MIN_RUNS");
    reset_quantum_control_state();
}

#[test]
fn quantum_lifecycle_reclaims_repeated_failures() {
    use crate::py::jit::codegen::{
        quantum_active_variant_count, reconcile_quantum_lifecycle, seed_quantum_profile,
        QuantumProfileSeed,
    };
    use std::env;

    let _guard = lock_quantum_shared_test();
    reset_quantum_control_state();

    env::set_var("IRIS_JIT_QUANTUM_VARIANT_FAILURE_LIMIT", "2");
    env::set_var("IRIS_JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS", "8");

    let args = ["x".to_string(), "y".to_string()];
    let entries = compile_jit_quantum(
        "x + y",
        &args,
        crate::py::jit::codegen::JitReturnType::Float,
    );
    assert!(!entries.is_empty());
    let func_key = 77_002;
    register_quantum_jit(func_key, entries);

    let initial_active = quantum_active_variant_count(func_key).unwrap();
    assert!(initial_active >= 2, "expected at least 2 active variants");

    assert!(seed_quantum_profile(
        func_key,
        &[
            QuantumProfileSeed {
                index: 0,
                ewma_ns: 1000.0,
                runs: 30,
                failures: 0
            },
            QuantumProfileSeed {
                index: 1,
                ewma_ns: 2000.0,
                runs: 1,
                failures: 3
            },
        ],
    ));
    assert!(reconcile_quantum_lifecycle(func_key));
    let reclaimed_active = quantum_active_variant_count(func_key).unwrap();
    assert!(
        reclaimed_active < initial_active,
        "expected lifecycle to reclaim failing variants"
    );
    assert!(
        reclaimed_active >= 1,
        "expected at least one active variant to remain"
    );

    env::remove_var("IRIS_JIT_QUANTUM_VARIANT_FAILURE_LIMIT");
    env::remove_var("IRIS_JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS");
    reset_quantum_control_state();
}

#[test]
fn quantum_rearms_from_single_variant_on_degradation() {
    use crate::py::jit::codegen::{
        compile_jit_with_return_type, quantum_active_variant_count, register_quantum_jit,
        JitReturnType,
    };
    use std::env;

    let _guard = lock_quantum_shared_test();

    reset_quantum_control_state();
    env::set_var("IRIS_JIT_QUANTUM", "1");
    env::set_var("IRIS_JIT_QUANTUM_SPECULATION_NS", "0");
    env::set_var("IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS", "1000000000");
    env::set_var("IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS", "1000000000");
    env::set_var("IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS", "0");
    env::set_var("IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS", "0");
    env::set_var("IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES", "1");

    let args = ["x".to_string()];
    let entry = compile_jit_with_return_type("x + 1", &args, JitReturnType::Float)
        .expect("single compile for baseline state");
    let func_key = 88_001;
    register_quantum_jit(func_key, vec![entry]);
    assert_eq!(quantum_active_variant_count(func_key).unwrap(), 1);

    register_quantum_rearm_plan_for_test(func_key, "x + 1", &args, JitReturnType::Float);
    assert!(maybe_rearm_quantum_compile(func_key, 5_000_000, 1));
    assert!(
        quantum_active_variant_count(func_key).unwrap() > 1,
        "expected drift-triggered rearm to restore multi-variant quantum state"
    );

    clear_quantum_rearm_plan_for_test(func_key);
    env::remove_var("IRIS_JIT_QUANTUM");
    env::remove_var("IRIS_JIT_QUANTUM_SPECULATION_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS");
    env::remove_var("IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES");
}

#[test]
fn quantum_rearm_requires_min_samples_for_sensitivity() {
    use crate::py::jit::codegen::{
        compile_jit_with_return_type, quantum_active_variant_count, register_quantum_jit,
        JitReturnType,
    };
    use std::env;

    let _guard = lock_quantum_shared_test();

    reset_quantum_control_state();
    env::set_var("IRIS_JIT_QUANTUM", "1");
    env::set_var("IRIS_JIT_QUANTUM_SPECULATION_NS", "0");
    env::set_var("IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS", "1000000000");
    env::set_var("IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS", "1000000000");
    env::set_var("IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS", "0");
    env::set_var("IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS", "0");
    env::set_var("IRIS_JIT_QUANTUM_REARM_INTERVAL_NS", "0");
    env::set_var("IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES", "3");

    let args = ["x".to_string()];
    let entry = compile_jit_with_return_type("x + 1", &args, JitReturnType::Float)
        .expect("single compile for baseline state");
    let func_key = 88_002;
    register_quantum_jit(func_key, vec![entry]);
    assert_eq!(quantum_active_variant_count(func_key).unwrap(), 1);

    register_quantum_rearm_plan_for_test(func_key, "x + 1", &args, JitReturnType::Float);

    assert!(!maybe_rearm_quantum_compile(func_key, 5_000_000, 1));
    assert!(!maybe_rearm_quantum_compile(func_key, 5_000_000, 1));
    assert!(maybe_rearm_quantum_compile(func_key, 5_000_000, 1));

    clear_quantum_rearm_plan_for_test(func_key);
    env::remove_var("IRIS_JIT_QUANTUM");
    env::remove_var("IRIS_JIT_QUANTUM_SPECULATION_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS");
    env::remove_var("IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS");
    env::remove_var("IRIS_JIT_QUANTUM_REARM_INTERVAL_NS");
    env::remove_var("IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES");
}

#[test]
#[cfg(feature = "pyo3")]
fn quantum_speculation_logs_choice_when_slow() {
    use crate::py::jit::{
        execute_registered_jit, jit_log_clear_hook, jit_log_hook, register_quantum_jit,
    };
    use pyo3::Python;
    use std::env;
    use std::sync::{Arc, Mutex};

    let _guard = lock_quantum_shared_test();
    reset_quantum_control_state();

    // force all logs on and make the threshold 0 so we can validate the text.
    env::set_var("IRIS_JIT_LOG", "1");
    env::set_var("IRIS_JIT_QUANTUM", "1");
    env::set_var("IRIS_JIT_QUANTUM_LOG_NS", "0");

    let logs = Arc::new(Mutex::new(Vec::new()));
    let logs_clone = logs.clone();
    jit_log_hook(move |line| {
        logs_clone.lock().unwrap().push(line);
    });

    let args = ["x".to_string()];
    let entries = compile_jit_quantum(
        "x + 1",
        &args,
        crate::py::jit::codegen::JitReturnType::Float,
    );
    assert!(!entries.is_empty());

    let func_key = 12345;
    register_quantum_jit(func_key, entries);

    Python::attach(|py| {
        let tup = pyo3::types::PyTuple::new(py, [1.0_f64]).unwrap();
        let res = execute_registered_jit(py, func_key, &tup).unwrap().unwrap();
        let out: f64 = res.bind(py).extract().unwrap();
        assert_eq!(out, 2.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();

    let logged = logs.lock().unwrap();
    assert!(logged
        .iter()
        .any(|line| line.contains("[Iris][jit][quantum]")));

    jit_log_clear_hook();
    env::remove_var("IRIS_JIT_LOG");
    env::remove_var("IRIS_JIT_QUANTUM");
    env::remove_var("IRIS_JIT_QUANTUM_LOG_NS");
}

#[test]
#[cfg(feature = "pyo3")]
fn quantum_first_run_vector_container_reduction_executes() {
    use crate::py::jit::{execute_registered_jit, register_quantum_jit};
    use pyo3::Python;
    use std::env;

    let _guard = lock_quantum_shared_test();
    reset_quantum_control_state();

    env::set_var("IRIS_JIT_QUANTUM", "1");

    let args = ["x".to_string()];
    let entries = compile_jit_quantum(
        "sum((x_i * x_i for x_i in x if x_i > 0))",
        &args,
        crate::py::jit::codegen::JitReturnType::Float,
    );
    assert!(
        !entries.is_empty(),
        "quantum compile should produce at least one variant"
    );

    let func_key = 123_456_789;
    register_quantum_jit(func_key, entries);

    Python::attach(|py| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!("from array import array\nxs = array('d', [1.0, 2.0, 3.0, 4.0])"),
            None,
            Some(&locals),
        )
        .unwrap();

        let xs = locals.get_item("xs").unwrap().unwrap();
        let tuple = pyo3::types::PyTuple::new(py, [xs]).unwrap();
        let out_obj = execute_registered_jit(py, func_key, &tuple)
            .expect("quantum dispatcher should return a result")
            .expect("quantum first-run execution should succeed");
        let out: f64 = out_obj.bind(py).extract().unwrap();
        assert_eq!(out, 30.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();

    env::remove_var("IRIS_JIT_QUANTUM");
    reset_quantum_control_state();
}

#[test]
#[cfg(feature = "pyo3")]
fn quantum_first_run_multiarg_vector_executes() {
    use crate::py::jit::{execute_registered_jit, register_quantum_jit};
    use pyo3::Python;
    use std::env;

    let _guard = lock_quantum_shared_test();
    reset_quantum_control_state();

    env::set_var("IRIS_JIT_QUANTUM", "1");

    let args = vec!["price".to_string(), "vol".to_string(), "strike".to_string()];
    let entries = compile_jit_quantum(
        "price / strike + vol",
        &args,
        crate::py::jit::codegen::JitReturnType::Float,
    );
    assert!(
        !entries.is_empty(),
        "quantum compile should produce at least one variant"
    );

    let func_key = 123_456_790;
    register_quantum_jit(func_key, entries);

    Python::attach(|py| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "from array import array\n\
prices = array('d', [100.0, 101.0, 102.0])\n\
vols = array('d', [0.2, 0.2, 0.2])\n\
strikes = array('d', [105.0, 105.0, 105.0])"
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        let prices = locals.get_item("prices").unwrap().unwrap();
        let vols = locals.get_item("vols").unwrap().unwrap();
        let strikes = locals.get_item("strikes").unwrap().unwrap();

        let tuple = pyo3::types::PyTuple::new(py, [prices, vols, strikes]).unwrap();
        let out_obj = execute_registered_jit(py, func_key, &tuple)
            .expect("quantum dispatcher should return a result")
            .expect("quantum first-run multiarg execution should succeed");

        let out: Vec<f64> = out_obj.bind(py).extract().unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - ((100.0 / 105.0) + 0.2)).abs() < 1e-12);
        assert!((out[1] - ((101.0 / 105.0) + 0.2)).abs() < 1e-12);
        assert!((out[2] - ((102.0 / 105.0) + 0.2)).abs() < 1e-12);
        Ok::<(), PyErr>(())
    })
    .unwrap();

    env::remove_var("IRIS_JIT_QUANTUM");
    reset_quantum_control_state();
}

#[test]
fn jit_builder_pic_flag_behavior() {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    if cfg!(target_arch = "aarch64") {
        flag_builder.set("is_pic", "false").unwrap();
    } else {
        flag_builder.set("is_pic", "true").unwrap();
    }
    let isa_builder = cranelift_native::builder().unwrap();
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    assert_eq!(isa.flags().is_pic(), !cfg!(target_arch = "aarch64"));
}

#[test]
fn compile_jit_nested_parens_generator() {
    let args = vec!["n".to_string()];
    let entry = compile_jit("sum((i * i for i in range(int(n))))", &args);
    assert!(entry.is_some(), "nested parens generator should compile");
}

#[test]
fn compile_jit_vector_generator_should_compile() {
    let args = ["x".to_string()];
    let entry = compile_jit("sum((x_i * x_i for x_i in x))", &args)
        .expect("vector generator should now compile");
    // entry.arg_count should equal 1 (element argument)
    assert_eq!(entry.arg_count, 1);
}

#[test]
fn compile_jit_math_functions() {
    let args = ["x".to_string()];
    // trigonometry
    let entry = compile_jit("sin(x)", &args).expect("should compile sin");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [std::f64::consts::PI / 2.0];
    assert!((f(vals.as_ptr()) - 1.0).abs() < 1e-12);

    let entry2 = compile_jit("cos(x)", &args).expect("should compile cos");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    let vals2 = [0.0];
    assert!((g(vals2.as_ptr()) - 1.0).abs() < 1e-12);

    // hyperbolics
    let entryh = compile_jit("sinh(x)", &args).expect("should compile sinh");
    let sh: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entryh.func_ptr) };
    let vsh = [1.0];
    assert!((sh(vsh.as_ptr()) - vsh[0].sinh()).abs() < 1e-12);

    let entryh2 = compile_jit("cosh(x)", &args).expect("should compile cosh");
    let ch: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entryh2.func_ptr) };
    let vch = [0.0];
    assert!((ch(vch.as_ptr()) - vch[0].cosh()).abs() < 1e-12);

    let entryh3 = compile_jit("tanh(x)", &args).expect("should compile tanh");
    let th: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entryh3.func_ptr) };
    let vth = [0.0];
    assert!((th(vth.as_ptr()) - vth[0].tanh()).abs() < 1e-12);

    // exponentials / logs
    let entry3 = compile_jit("exp(x)", &args).expect("should compile exp");
    let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
    let vals3 = [1.0];
    assert!((h(vals3.as_ptr()) - std::f64::consts::E).abs() < 1e-12);

    let entry4 = compile_jit("log(x)", &args).expect("should compile log");
    let k: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry4.func_ptr) };
    let vals4 = [std::f64::consts::E];
    assert!((k(vals4.as_ptr()) - 1.0).abs() < 1e-12);

    // square root and tangent
    let entry5 = compile_jit("sqrt(x)", &args).expect("should compile sqrt");
    let s: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry5.func_ptr) };
    let vals5 = [16.0];
    assert!((s(vals5.as_ptr()) - 4.0).abs() < 1e-12);

    let entry6 = compile_jit("tan(x)", &args).expect("should compile tan");
    let t: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry6.func_ptr) };
    let vals6 = [0.0];
    assert!((t(vals6.as_ptr()) - 0.0).abs() < 1e-12);

    let entry7 = compile_jit("int(x) + float(1)", &args).expect("should compile int/float casts");
    let u: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry7.func_ptr) };
    let vals7 = [2.9];
    assert!((u(vals7.as_ptr()) - 3.0).abs() < 1e-12);

    let entry8 = compile_jit("round(x)", &args).expect("should compile round alias");
    let v: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry8.func_ptr) };
    let vals8 = [2.6];
    assert!((v(vals8.as_ptr()) - 3.0).abs() < 1e-12);
}

#[test]
fn symbol_alias_table_maps_expected_intrinsics() {
    assert_eq!(
        resolve_symbol_alias("int", 1),
        Some(SymbolAlias::Rename("trunc"))
    );
    assert_eq!(
        resolve_symbol_alias("float", 1),
        Some(SymbolAlias::Identity)
    );
    assert_eq!(
        resolve_symbol_alias("round", 1),
        Some(SymbolAlias::Rename("round"))
    );
    assert_eq!(resolve_symbol_alias("round", 2), None);
}

#[test]
fn named_jit_registry_roundtrip() {
    let args = ["x".to_string()];
    let entry = compile_jit("x + 1", &args).expect("compile named jit entry");
    register_named_jit("inner_add1", entry.clone());
    let looked = lookup_named_jit("inner_add1").expect("lookup named jit entry");
    assert_eq!(looked.func_ptr, entry.func_ptr);
    assert_eq!(looked.arg_count, entry.arg_count);
    assert_eq!(looked.reduction, entry.reduction);
}

#[test]
fn named_jit_registry_overwrites_same_name() {
    let args = ["x".to_string()];
    let entry1 = compile_jit("x + 1", &args).expect("compile first named jit");
    let entry2 = compile_jit("x + 2", &args).expect("compile second named jit");
    register_named_jit("inner_overwrite", entry1.clone());
    register_named_jit("inner_overwrite", entry2.clone());
    let looked = lookup_named_jit("inner_overwrite").expect("lookup overwritten named jit");
    assert_eq!(looked.func_ptr, entry2.func_ptr);
}

#[test]
fn named_jit_call_from_compiled_expression() {
    let args = ["x".to_string()];
    let inner = compile_jit("x + 1", &args).expect("compile inner function");
    register_named_jit("inner_add1", inner);

    let outer = compile_jit("inner_add1(x) * 2", &args).expect("compile outer function");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(outer.func_ptr) };
    let vals = [3.0_f64];
    assert_eq!(f(vals.as_ptr()), 8.0);
}

#[test]
fn named_jit_call_from_compiled_expression_five_args() {
    let inner_args = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "e".to_string(),
    ];
    let inner =
        compile_jit("a + b + c + d + e", &inner_args).expect("compile inner five-arg function");
    register_named_jit("inner_sum5", inner);

    let outer = compile_jit("inner_sum5(a, b, c, d, e) * 2", &inner_args)
        .expect("compile outer five-arg function");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(outer.func_ptr) };
    let vals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(f(vals.as_ptr()), 30.0);
}

#[test]
fn named_jit_invoke_helper_five_args_direct() {
    let inner_args = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "e".to_string(),
    ];
    let inner =
        compile_jit("a + b + c + d + e", &inner_args).expect("compile inner five-arg function");
    let out =
        crate::py::jit::codegen::iris_jit_invoke_5(inner.func_ptr as i64, 1.0, 2.0, 3.0, 4.0, 5.0);
    assert_eq!(out, 15.0);
}

#[test]
fn named_jit_call_from_compiled_expression_twelve_args() {
    let inner_args = vec![
        "a0".to_string(),
        "a1".to_string(),
        "a2".to_string(),
        "a3".to_string(),
        "a4".to_string(),
        "a5".to_string(),
        "a6".to_string(),
        "a7".to_string(),
        "a8".to_string(),
        "a9".to_string(),
        "a10".to_string(),
        "a11".to_string(),
    ];
    let inner = compile_jit("a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11", &inner_args)
        .expect("compile inner twelve-arg function");
    register_named_jit("inner_sum12", inner);

    let outer = compile_jit(
        "inner_sum12(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11) * 2",
        &inner_args,
    )
    .expect("compile outer twelve-arg function");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(outer.func_ptr) };
    let vals = [
        1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    assert_eq!(f(vals.as_ptr()), 24.0);
}

#[test]
fn named_jit_invoke_helper_sixteen_args_direct() {
    let inner_args = vec![
        "a0".to_string(),
        "a1".to_string(),
        "a2".to_string(),
        "a3".to_string(),
        "a4".to_string(),
        "a5".to_string(),
        "a6".to_string(),
        "a7".to_string(),
        "a8".to_string(),
        "a9".to_string(),
        "a10".to_string(),
        "a11".to_string(),
        "a12".to_string(),
        "a13".to_string(),
        "a14".to_string(),
        "a15".to_string(),
    ];
    let inner = compile_jit(
        "a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15",
        &inner_args,
    )
    .expect("compile inner sixteen-arg function");

    let out = crate::py::jit::codegen::iris_jit_invoke_16(
        inner.func_ptr as i64,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    );
    assert_eq!(out, 16.0);
}

#[test]
fn compile_jit_power_op() {
    // simple power
    let entry = compile_jit("2 ** 3", &[]).expect("const power");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let empty: [f64; 0] = [];
    assert_eq!(f(empty.as_ptr()), 8.0);

    // right-associative
    let entry2 = compile_jit("2 ** 3 ** 2", &[]).expect("assoc");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 512.0); // 2^(3^2)

    // strength reduction path should handle small integer exponents
    let entry3 = compile_jit("5 ** 4", &[]).expect("strength");
    let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
    assert_eq!(h(empty.as_ptr()), 625.0);

    // negative constants still use pow (result should be correct)
    let entry4 = compile_jit("2 ** -2", &[]).expect("neg exp");
    let k: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry4.func_ptr) };
    assert!((k(empty.as_ptr()) - 0.25).abs() < 1e-12);

    // fast sqrt rewrite for exponent 0.5
    let entry5 = compile_jit("9 ** 0.5", &[]).expect("sqrt rewrite");
    let q: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry5.func_ptr) };
    assert!((q(empty.as_ptr()) - 3.0).abs() < 1e-12);

    // reciprocal rewrite for exponent -1
    let entry6 = compile_jit("8 ** -1", &[]).expect("reciprocal rewrite");
    let r: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry6.func_ptr) };
    assert!((r(empty.as_ptr()) - 0.125).abs() < 1e-12);
}

#[test]
fn compile_jit_dotted_and_abs() {
    let args = ["x".to_string()];
    let entry = compile_jit("math.sin(x)", &args).expect("dotted sin");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [std::f64::consts::PI / 2.0];
    assert!((f(vals.as_ptr()) - 1.0).abs() < 1e-12);

    let entry2 = compile_jit("abs(x)", &args).expect("abs maps to fabs");
    let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    let vals2 = [-4.0];
    assert_eq!(h(vals2.as_ptr()), 4.0);
}

#[test]
fn compile_jit_sum_range_loop() {
    let args = vec!["n".to_string()];
    let entry = compile_jit("sum(i for i in range(n))", &args).expect("sum range loop");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [5.0];
    assert_eq!(f(vals.as_ptr()), 10.0);
}

#[test]
fn compile_jit_sum_range_loop_with_body_expr() {
    let args = vec!["x".to_string(), "n".to_string()];
    let entry =
        compile_jit("sum(i * x for i in range(n))", &args).expect("sum range with body expr");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [2.0, 4.0];
    assert_eq!(f(vals.as_ptr()), 12.0);
}

#[test]
fn compile_jit_sum_range_negative_step() {
    let entry = compile_jit("sum(i for i in range(5, 0, -1))", &[]).expect("negative step");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let empty: [f64; 0] = [];
    assert_eq!(f(empty.as_ptr()), 15.0);
}

#[test]
fn compile_jit_sum_range_negative_step_dynamic() {
    let args = vec!["a".to_string(), "b".to_string(), "s".to_string()];
    let entry =
        compile_jit("sum(i for i in range(a, b, s))", &args).expect("dynamic negative step");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [5.0, 0.0, -1.0];
    assert_eq!(f(vals.as_ptr()), 15.0);
}

#[test]
fn compile_jit_sum_container_with_predicate() {
    let args = ["x".to_string()];
    let entry = compile_jit("sum(x_i for x_i in x if x_i > 0)", &args)
        .expect("container generator with predicate should compile");
    assert_eq!(entry.arg_count, 1);
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

    let positive = [3.0];
    assert_eq!(f(positive.as_ptr()), 3.0);

    let negative = [-2.0];
    assert_eq!(f(negative.as_ptr()), 0.0);
}

#[test]
fn compile_jit_any_container_with_predicate() {
    let args = ["x".to_string()];
    let entry = compile_jit("any(x_i > 0 for x_i in x if x_i != 0)", &args)
        .expect("container any with predicate should compile");
    assert_eq!(entry.arg_count, 1);
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

    let positive = [2.0];
    assert_eq!(f(positive.as_ptr()), 1.0);

    let zero = [0.0];
    assert_eq!(f(zero.as_ptr()), 0.0);

    let negative = [-3.0];
    assert_eq!(f(negative.as_ptr()), 0.0);
}

#[test]
fn compile_jit_all_container_with_predicate() {
    let args = ["x".to_string()];
    let entry = compile_jit("all(x_i > 0 for x_i in x if x_i != 0)", &args)
        .expect("container all with predicate should compile");
    assert_eq!(entry.arg_count, 1);
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

    let positive = [2.0];
    assert_eq!(f(positive.as_ptr()), 1.0);

    let zero = [0.0];
    assert_eq!(f(zero.as_ptr()), 1.0);

    let negative = [-3.0];
    assert_eq!(f(negative.as_ptr()), 0.0);
}

#[test]
fn compile_jit_any_range_generator() {
    let entry = compile_jit("any(i > 3 for i in range(5))", &[]).expect("any range");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let empty: [f64; 0] = [];
    assert_eq!(f(empty.as_ptr()), 1.0);

    let entry2 = compile_jit("any(i > 10 for i in range(5))", &[]).expect("any range false");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 0.0);

    let entry3 = compile_jit("any(i > 0 for i in range(0))", &[]).expect("any empty");
    let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
    assert_eq!(h(empty.as_ptr()), 0.0);
}

#[test]
fn compile_jit_all_range_generator() {
    let entry = compile_jit("all(i < 5 for i in range(5))", &[]).expect("all range true");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let empty: [f64; 0] = [];
    assert_eq!(f(empty.as_ptr()), 1.0);

    let entry2 = compile_jit("all(i < 3 for i in range(5))", &[]).expect("all range false");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 0.0);

    let entry3 = compile_jit("all(i > 0 for i in range(0))", &[]).expect("all empty");
    let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
    assert_eq!(h(empty.as_ptr()), 1.0);
}

#[test]
fn compile_jit_any_all_with_predicate() {
    let empty: [f64; 0] = [];

    let any_pred =
        compile_jit("any(i > 3 for i in range(6) if i % 2 == 0)", &[]).expect("any with predicate");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_pred.func_ptr) };
    assert_eq!(f(empty.as_ptr()), 1.0); // included set: {0,2,4}

    let all_pred =
        compile_jit("all(i % 2 == 0 for i in range(6) if i < 5)", &[]).expect("all with predicate");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_pred.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 0.0); // included set: {0,1,2,3,4}
}

#[test]
fn compile_jit_sum_with_break_continue_intrinsics() {
    let empty: [f64; 0] = [];

    let cont = compile_jit("sum(continue_if(i % 2 == 0, i) for i in range(6))", &[])
        .expect("sum continue_if");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(cont.func_ptr) };
    assert_eq!(f(empty.as_ptr()), 9.0); // 1+3+5

    let brk =
        compile_jit("sum(break_if(i >= 4, i) for i in range(10))", &[]).expect("sum break_if");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(brk.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 6.0); // 0+1+2+3
}

#[test]
fn compile_jit_any_all_with_break_continue_intrinsics() {
    let empty: [f64; 0] = [];

    let any_cont = compile_jit("any(continue_if(i < 3, i > 5) for i in range(8))", &[])
        .expect("any continue_if");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_cont.func_ptr) };
    assert_eq!(f(empty.as_ptr()), 1.0);

    let any_break =
        compile_jit("any(break_if(i >= 3, i > 10) for i in range(8))", &[]).expect("any break_if");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_break.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 0.0);

    let all_cont = compile_jit("all(continue_if(i < 3, i < 10) for i in range(6))", &[])
        .expect("all continue_if");
    let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_cont.func_ptr) };
    assert_eq!(h(empty.as_ptr()), 1.0);

    let all_break =
        compile_jit("all(break_if(i >= 4, i < 10) for i in range(6))", &[]).expect("all break_if");
    let q: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_break.func_ptr) };
    assert_eq!(q(empty.as_ptr()), 1.0);
}

#[test]
fn compile_jit_with_break_continue_unless_intrinsics() {
    let empty: [f64; 0] = [];

    let sum_break_unless = compile_jit("sum(break_unless(i < 4, i) for i in range(10))", &[])
        .expect("sum break_unless");
    let f: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(sum_break_unless.func_ptr) };
    assert_eq!(f(empty.as_ptr()), 6.0);

    let sum_continue_unless =
        compile_jit("sum(continue_unless(i % 2 == 1, i) for i in range(6))", &[])
            .expect("sum continue_unless");
    let g: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(sum_continue_unless.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 9.0);

    let any_break_unless = compile_jit("any(break_unless(i < 3, i > 10) for i in range(8))", &[])
        .expect("any break_unless");
    let h: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(any_break_unless.func_ptr) };
    assert_eq!(h(empty.as_ptr()), 0.0);

    let all_continue_unless = compile_jit(
        "all(continue_unless(i % 2 == 0, i < 10) for i in range(6))",
        &[],
    )
    .expect("all continue_unless");
    let q: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(all_continue_unless.func_ptr) };
    assert_eq!(q(empty.as_ptr()), 1.0);

    let sum_break_when =
        compile_jit("sum(break_when(i >= 4, i) for i in range(10))", &[]).expect("sum break_when");
    let r: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(sum_break_when.func_ptr) };
    assert_eq!(r(empty.as_ptr()), 6.0);

    let sum_continue_when = compile_jit("sum(continue_when(i % 2 == 0, i) for i in range(6))", &[])
        .expect("sum continue_when");
    let s: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(sum_continue_when.func_ptr) };
    assert_eq!(s(empty.as_ptr()), 9.0);
}

#[test]
fn compile_jit_if_else_control_flow_function() {
    let args = ["x".to_string(), "y".to_string()];
    let entry = compile_jit("if_else(x < y, x, y)", &args).expect("if_else compile");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [2.0, 5.0];
    assert_eq!(f(vals.as_ptr()), 2.0);

    let entry2 = compile_jit("sum(if_else(i % 2 == 0, i, 0) for i in range(6))", &[])
        .expect("if_else in reduction compile");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    let empty: [f64; 0] = [];
    assert_eq!(g(empty.as_ptr()), 6.0);
}

#[test]
fn compile_jit_while_reductions() {
    let args = vec!["n".to_string()];

    let sum_entry =
        compile_jit("sum_while(i, 0, i < n, i + 1, i)", &args).expect("sum_while compile");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_entry.func_ptr) };
    let vals = [5.0];
    assert_eq!(f(vals.as_ptr()), 10.0);

    let sum_entry2 = compile_jit("sum_while(i, 1, i <= n, i + 1, i)", &args)
        .expect("sum_while inclusive compile");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_entry2.func_ptr) };
    assert_eq!(g(vals.as_ptr()), 15.0);
}

#[test]
fn compile_jit_while_reductions_with_loop_control() {
    let empty: [f64; 0] = [];

    let sum_break = compile_jit("sum_while(i, 0, i < 10, i + 1, break_if(i >= 4, i))", &[])
        .expect("sum_while break_if compile");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_break.func_ptr) };
    assert_eq!(f(empty.as_ptr()), 6.0);

    let sum_continue = compile_jit(
        "sum_while(i, 0, i < 10, i + 1, continue_if(i % 2 == 0, i))",
        &[],
    )
    .expect("sum_while continue_if compile");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(sum_continue.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 25.0);

    let sum_break_nan = compile_jit(
        "sum_while(i, 0, i < 10, i + 1, break_on_nan((i - i) / (i - i)))",
        &[],
    )
    .expect("sum_while break_on_nan compile");
    let h: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(sum_break_nan.func_ptr) };
    assert_eq!(h(empty.as_ptr()), 0.0);

    let any_while =
        compile_jit("any_while(i, 0, i < 8, i + 1, i >= 6)", &[]).expect("any_while compile");
    let q: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(any_while.func_ptr) };
    assert_eq!(q(empty.as_ptr()), 1.0);

    let all_while =
        compile_jit("all_while(i, 0, i < 8, i + 1, i < 8)", &[]).expect("all_while compile");
    let r: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(all_while.func_ptr) };
    assert_eq!(r(empty.as_ptr()), 1.0);

    let any_continue_nan = compile_jit(
        "any_while(i, 0, i < 5, i + 1, continue_on_nan((i - i) / (i - i)))",
        &[],
    )
    .expect("any_while continue_on_nan compile");
    let s: extern "C" fn(*const f64) -> f64 =
        unsafe { std::mem::transmute(any_continue_nan.func_ptr) };
    assert_eq!(s(empty.as_ptr()), 0.0);
}

#[test]
fn compile_jit_function_inlining_min_max() {
    let args = ["x".to_string(), "y".to_string()];
    let entry = compile_jit("max(x, y) - min(x, y)", &args).expect("min/max compile");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [2.0, 7.0];
    assert_eq!(f(vals.as_ptr()), 5.0);

    let entry2 = compile_jit("sum(max(i, 2) - min(i, 2) for i in range(5))", &[])
        .expect("min/max in reduction compile");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    let empty: [f64; 0] = [];
    assert_eq!(g(empty.as_ptr()), 6.0);
}

#[test]
fn compile_jit_range_step_and_predicate() {
    let entry = compile_jit("sum(i for i in range(0,10,2))", &[]).expect("step");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let empty: [f64; 0] = [];
    assert_eq!(f(empty.as_ptr()), 20.0);

    let entry2 = compile_jit("sum(i for i in range(5) if i % 2 == 0)", &[]).expect("pred");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    assert_eq!(g(empty.as_ptr()), 6.0);
}

#[tokio::test]
async fn compile_jit_python_api_call_tokio() {
    // same as above but run inside tokio's async test harness
    Python::attach(|py| {
        let args = ["x".to_string(), "y".to_string()];
        let entry = compile_jit("x < y", &args).expect("compare");
        let tuple = pyo3::types::PyTuple::new(py, [1.0_f64, 2.0_f64]).unwrap();
        // sanity check tuple contents using safe API
        let a: f64 = tuple.get_item(0).unwrap().extract().unwrap();
        let b: f64 = tuple.get_item(1).unwrap().extract().unwrap();
        assert_eq!(a, 1.0);
        assert_eq!(b, 2.0);
        let res_obj = execute_jit_func(py, &entry, &tuple).expect("exec");
        let res: f64 = res_obj.bind(py).extract().unwrap();
        assert_eq!(res, 1.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn compile_jit_relation_and_conditional() {
    let args = ["x".to_string(), "y".to_string()];
    let entry = compile_jit("x < y", &args).expect("compare");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [1.0, 2.0];
    assert_eq!(f(vals.as_ptr()), 1.0);

    let entry2 = compile_jit("x >= y", &args).expect("compare2");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    assert_eq!(g(vals.as_ptr()), 0.0);

    // conditional expression (ternary)
    let entry3 = compile_jit("x if x < y else y", &args).expect("ternary");
    let h: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry3.func_ptr) };
    assert_eq!(h(vals.as_ptr()), 1.0);
}

#[test]
fn compile_jit_boolean_and_or() {
    let args = ["x".to_string(), "y".to_string(), "z".to_string()];
    let entry = compile_jit("x < y and y < z", &args).expect("and compare");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals_true = [1.0, 2.0, 3.0];
    let vals_false = [3.0, 2.0, 1.0];
    assert_eq!(f(vals_true.as_ptr()), 1.0);
    assert_eq!(f(vals_false.as_ptr()), 0.0);

    let entry2 = compile_jit("x > y or y < z", &args).expect("or compare");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    assert_eq!(g(vals_true.as_ptr()), 1.0);
    assert_eq!(g(vals_false.as_ptr()), 1.0);
}

#[test]
fn compile_jit_boolean_not() {
    let args = ["x".to_string(), "y".to_string()];
    let entry = compile_jit("not x < y", &args).expect("not compare");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals_true = [3.0, 2.0];
    let vals_false = [1.0, 2.0];
    assert_eq!(f(vals_true.as_ptr()), 1.0);
    assert_eq!(f(vals_false.as_ptr()), 0.0);
}

#[test]
fn compile_jit_boolean_literals() {
    let args = ["x".to_string(), "y".to_string()];

    let entry = compile_jit("x if True else y", &args).expect("ternary true literal");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals = [3.0, 9.0];
    assert_eq!(f(vals.as_ptr()), 3.0);

    let entry2 = compile_jit("x if False else y", &args).expect("ternary false literal");
    let g: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry2.func_ptr) };
    assert_eq!(g(vals.as_ptr()), 9.0);
}

#[test]
fn compile_jit_comparison_chain() {
    let args = ["x".to_string(), "y".to_string(), "z".to_string()];
    let entry = compile_jit("x < y < z", &args).expect("comparison chain");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };
    let vals_true = [1.0, 2.0, 3.0];
    let vals_false = [3.0, 2.0, 1.0];
    assert_eq!(f(vals_true.as_ptr()), 1.0);
    assert_eq!(f(vals_false.as_ptr()), 0.0);
}

#[test]
fn compile_jit_mixed_chain_not_stress() {
    let args = ["x".to_string(), "y".to_string(), "z".to_string()];
    let entry = compile_jit("not x <= y < z and z >= y", &args).expect("mixed chain/not");
    let f: extern "C" fn(*const f64) -> f64 = unsafe { std::mem::transmute(entry.func_ptr) };

    // x<=y<z is true here, so `not ...` is false; false and true => false
    let vals_false = [1.0, 2.0, 3.0];
    assert_eq!(f(vals_false.as_ptr()), 0.0);

    // x<=y<z is false here, so `not ...` is true; true and true => true
    let vals_true = [3.0, 2.0, 2.0];
    assert_eq!(f(vals_true.as_ptr()), 1.0);
}

#[test]
fn execute_jit_accepts_mixed_scalar_types() {
    Python::attach(|py| {
        let args = ["x".to_string(), "y".to_string(), "z".to_string()];
        let entry = compile_jit("x + y + z", &args).expect("compile mixed scalar test");
        let tuple = PyTuple::new(
            py,
            [
                1_i64.into_py_any(py).unwrap(),
                true.into_py_any(py).unwrap(),
                2_i32.into_py_any(py).unwrap(),
            ],
        )
        .unwrap();
        let result = execute_jit_func(py, &entry, &tuple).expect("execute mixed scalars");
        let out: f64 = result.bind(py).extract().unwrap();
        assert_eq!(out, 4.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn execute_jit_vectorizes_non_f64_buffers() {
    let _guard = lock_quantum_shared_test();
    Python::attach(|py| {
        let array_mod = py.import(pyo3::ffi::c_str!("array")).unwrap();

        let args = ["x".to_string()];
        let mul_entry = compile_jit("x * 2", &args).expect("compile f32 buffer test");
        let f32_in = array_mod
            .getattr("array")
            .unwrap()
            .call1(("f", vec![1.5_f32, 2.0_f32, -3.0_f32]))
            .unwrap();
        let f32_tuple = PyTuple::new(py, [f32_in]).unwrap();
        let f32_out = execute_jit_func(py, &mul_entry, &f32_tuple).expect("execute f32 buffer");
        let f32_vals: Vec<f64> = f32_out
            .bind(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(f32_vals, vec![3.0, 4.0, -6.0]);

        let add_entry = compile_jit("x + 1", &args).expect("compile i32 buffer test");
        let i32_in = array_mod
            .getattr("array")
            .unwrap()
            .call1(("i", vec![1_i32, 2_i32, 7_i32]))
            .unwrap();
        let i32_tuple = PyTuple::new(py, [i32_in]).unwrap();
        let i32_out = execute_jit_func(py, &add_entry, &i32_tuple).expect("execute i32 buffer");
        let i32_vals: Vec<f64> = i32_out
            .bind(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(i32_vals, vec![2.0, 3.0, 8.0]);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn execute_jit_vectorizes_with_trailing_count() {
    Python::attach(|py| {
        let args = ["x".to_string()];
        let entry = compile_jit("x * 2", &args).expect("compile trailing count vectorized test");
        let tuple = PyTuple::new(
            py,
            [
                3.0_f64.into_py_any(py).unwrap(),
                4_i64.into_py_any(py).unwrap(),
            ],
        )
        .unwrap();
        let out_obj =
            execute_jit_func(py, &entry, &tuple).expect("execute trailing count vectorized");
        let out: Vec<f64> = out_obj
            .bind(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(out, vec![6.0, 6.0, 6.0, 6.0]);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn execute_jit_vectorize_with_negative_count_errors() {
    Python::attach(|py| {
        let args = ["x".to_string()];
        let entry = compile_jit("x + 1", &args).expect("compile negative count test");
        let tuple = PyTuple::new(
            py,
            [
                2.0_f64.into_py_any(py).unwrap(),
                (-1_i64).into_py_any(py).unwrap(),
            ],
        )
        .unwrap();
        let err = execute_jit_func(py, &entry, &tuple).expect_err("negative count should error");
        let msg = err.to_string();
        assert!(msg.contains("count"), "unexpected error message: {msg}");
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn execute_jit_handles_unaligned_f64_buffer_vectorized() {
    Python::attach(|py| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import struct\n\
buf=bytearray(1 + 8*3)\n\
struct.pack_into('ddd', buf, 1, 1.0, 2.0, 3.0)\n\
mv=memoryview(buf)[1:].cast('d')"
            ),
            None,
            Some(&locals),
        )
        .unwrap();
        let mv = locals.get_item("mv").unwrap().unwrap();

        let args = ["x".to_string()];
        let entry = compile_jit("x * 2", &args).expect("compile unaligned vectorized test");
        let tuple = PyTuple::new(py, [mv]).unwrap();
        let out_obj = execute_jit_func(py, &entry, &tuple).expect("execute unaligned vectorized");
        let out: Vec<f64> = out_obj
            .bind(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn execute_jit_handles_unaligned_f64_buffer_packed_args() {
    Python::attach(|py| {
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "import struct\n\
buf=bytearray(1 + 8*2)\n\
struct.pack_into('dd', buf, 1, 1.0, 2.0)\n\
mv=memoryview(buf)[1:].cast('d')"
            ),
            None,
            Some(&locals),
        )
        .unwrap();
        let mv = locals.get_item("mv").unwrap().unwrap();

        let args = ["a".to_string(), "b".to_string()];
        let entry = compile_jit("a + b", &args).expect("compile unaligned packed test");
        let tuple = PyTuple::new(py, [mv]).unwrap();
        let out_obj = execute_jit_func(py, &entry, &tuple).expect("execute unaligned packed");
        let out: f64 = out_obj.bind(py).extract().unwrap();
        assert_eq!(out, 3.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn execute_jit_container_reductions_with_python_lists() {
    Python::attach(|py| {
        let locals = pyo3::types::PyDict::new(py);
        let list_obj = py
            .eval(
                pyo3::ffi::c_str!("[1.0, -2.0, 3.0, 0.0]"),
                None,
                Some(&locals),
            )
            .unwrap();

        let sum_entry = compile_jit("sum(x_i * x_i for x_i in x)", &["x".to_string()])
            .expect("sum container compile");
        let sum_tuple = PyTuple::new(py, [list_obj.clone()]).unwrap();
        let sum_obj = execute_jit_func(py, &sum_entry, &sum_tuple).expect("sum container execute");
        let sum_val: f64 = sum_obj.bind(py).extract().unwrap();
        assert_eq!(sum_val, 14.0);

        let any_entry = compile_jit("any(x_i > 2 for x_i in x if x_i != 0)", &["x".to_string()])
            .expect("any container compile");
        let any_tuple = PyTuple::new(py, [list_obj.clone()]).unwrap();
        let any_obj = execute_jit_func(py, &any_entry, &any_tuple).expect("any container execute");
        let any_val: f64 = any_obj.bind(py).extract().unwrap();
        assert_eq!(any_val, 1.0);

        let all_entry = compile_jit(
            "all(x_i >= -2 for x_i in x if x_i != 0)",
            &["x".to_string()],
        )
        .expect("all container compile");
        let all_tuple = PyTuple::new(py, [list_obj]).unwrap();
        let all_obj = execute_jit_func(py, &all_entry, &all_tuple).expect("all container execute");
        let all_val: f64 = all_obj.bind(py).extract().unwrap();
        assert_eq!(all_val, 1.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[test]
fn execute_jit_container_reductions_with_loop_control_intrinsics() {
    Python::attach(|py| {
        let locals = pyo3::types::PyDict::new(py);
        let list_obj = py
            .eval(
                pyo3::ffi::c_str!("[1.0, 2.0, 3.0, 4.0, 5.0]"),
                None,
                Some(&locals),
            )
            .unwrap();

        let sum_break = compile_jit(
            "sum(break_if(x_i >= 4, x_i) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum break container compile");
        let sum_break_obj = execute_jit_func(
            py,
            &sum_break,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum break container execute");
        let sum_break_val: f64 = sum_break_obj.bind(py).extract().unwrap();
        assert_eq!(sum_break_val, 6.0);

        let sum_continue = compile_jit(
            "sum(continue_if(x_i % 2 == 0, x_i) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum continue container compile");
        let sum_continue_obj = execute_jit_func(
            py,
            &sum_continue,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum continue container execute");
        let sum_continue_val: f64 = sum_continue_obj.bind(py).extract().unwrap();
        assert_eq!(sum_continue_val, 9.0);

        let any_break = compile_jit(
            "any(break_if(x_i > 0, x_i > 10) for x_i in x)",
            &["x".to_string()],
        )
        .expect("any break container compile");
        let any_break_obj = execute_jit_func(
            py,
            &any_break,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("any break container execute");
        let any_break_val: f64 = any_break_obj.bind(py).extract().unwrap();
        assert_eq!(any_break_val, 0.0);

        let all_continue = compile_jit(
            "all(continue_if(x_i < 4, x_i > 0) for x_i in x)",
            &["x".to_string()],
        )
        .expect("all continue container compile");
        let all_continue_obj = execute_jit_func(
            py,
            &all_continue,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("all continue container execute");
        let all_continue_val: f64 = all_continue_obj.bind(py).extract().unwrap();
        assert_eq!(all_continue_val, 1.0);

        let sum_break_unless = compile_jit(
            "sum(break_unless(x_i < 4, x_i) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum break_unless container compile");
        let sum_break_unless_obj = execute_jit_func(
            py,
            &sum_break_unless,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum break_unless container execute");
        let sum_break_unless_val: f64 = sum_break_unless_obj.bind(py).extract().unwrap();
        assert_eq!(sum_break_unless_val, 6.0);

        let sum_continue_unless = compile_jit(
            "sum(continue_unless(x_i % 2 == 1, x_i) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum continue_unless container compile");
        let sum_continue_unless_obj = execute_jit_func(
            py,
            &sum_continue_unless,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum continue_unless container execute");
        let sum_continue_unless_val: f64 = sum_continue_unless_obj.bind(py).extract().unwrap();
        assert_eq!(sum_continue_unless_val, 9.0);

        let sum_break_when = compile_jit(
            "sum(break_when(x_i >= 4, x_i) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum break_when container compile");
        let sum_break_when_obj = execute_jit_func(
            py,
            &sum_break_when,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum break_when container execute");
        let sum_break_when_val: f64 = sum_break_when_obj.bind(py).extract().unwrap();
        assert_eq!(sum_break_when_val, 6.0);

        let sum_continue_when = compile_jit(
            "sum(continue_when(x_i % 2 == 0, x_i) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum continue_when container compile");
        let sum_continue_when_obj = execute_jit_func(
            py,
            &sum_continue_when,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum continue_when container execute");
        let sum_continue_when_val: f64 = sum_continue_when_obj.bind(py).extract().unwrap();
        assert_eq!(sum_continue_when_val, 9.0);

        let sum_break_on_nan = compile_jit(
            "sum(break_on_nan((x_i - x_i) / (x_i - x_i)) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum break_on_nan container compile");
        let sum_break_on_nan_obj = execute_jit_func(
            py,
            &sum_break_on_nan,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum break_on_nan container execute");
        let sum_break_on_nan_val: f64 = sum_break_on_nan_obj.bind(py).extract().unwrap();
        assert_eq!(sum_break_on_nan_val, 0.0);

        let sum_continue_on_nan = compile_jit(
            "sum(continue_on_nan((x_i - x_i) / (x_i - x_i)) for x_i in x)",
            &["x".to_string()],
        )
        .expect("sum continue_on_nan container compile");
        let sum_continue_on_nan_obj = execute_jit_func(
            py,
            &sum_continue_on_nan,
            &PyTuple::new(py, [list_obj.clone()]).unwrap(),
        )
        .expect("sum continue_on_nan container execute");
        let sum_continue_on_nan_val: f64 = sum_continue_on_nan_obj.bind(py).extract().unwrap();
        assert_eq!(sum_continue_on_nan_val, 0.0);

        let if_else_container = compile_jit(
            "sum(if_else(x_i > 0, x_i, 0) for x_i in x)",
            &["x".to_string()],
        )
        .expect("if_else container compile");
        let if_else_container_obj = execute_jit_func(
            py,
            &if_else_container,
            &PyTuple::new(py, [list_obj]).unwrap(),
        )
        .expect("if_else container execute");
        let if_else_container_val: f64 = if_else_container_obj.bind(py).extract().unwrap();
        assert_eq!(if_else_container_val, 15.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}
