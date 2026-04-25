#![cfg(all(feature = "pyo3", feature = "jit"))]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::test]
async fn py_jit_quantum_profile_persists_in_short_run() {
    let has_msgpack = Python::attach(|py| {
        let util = py.import(pyo3::ffi::c_str!("importlib.util"))?;
        let spec = util.call_method1("find_spec", ("msgpack",))?;
        Ok::<bool, PyErr>(!spec.is_none())
    })
    .unwrap_or(false);

    if !has_msgpack {
        return;
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("iris_jit_meta_short_{}", nonce));
    fs::create_dir_all(&temp_dir).unwrap();
    let module_path: PathBuf = temp_dir.join("warm_short_mod.py");

    let module_src = r#"
from iris.jit import offload

@offload(strategy="jit", return_type="float")
def warm_add_short(x):
    return x + 1
"#;
    fs::write(&module_path, module_src).unwrap();

    Python::attach(|py| {
        let ext = iris::py::make_module(py).expect("make_module");
        let sys = py.import(pyo3::ffi::c_str!("sys")).unwrap();
        let modules = sys
            .getattr("modules")
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap();

        let pkg = PyModule::new(py, "iris").unwrap();
        let package_path = vec!["iris".to_string()];
        pkg.setattr("__path__", package_path).unwrap();
        pkg.setattr("iris", ext.clone()).unwrap();

        modules.set_item("iris", pkg).unwrap();
        modules.set_item("iris.iris", ext.clone()).unwrap();

        let importlib_util = py.import(pyo3::ffi::c_str!("importlib.util")).unwrap();
        let module_name = "warm_short_mod";
        let module_path_s = module_path.to_string_lossy().to_string();

        let spec = importlib_util
            .getattr("spec_from_file_location")
            .unwrap()
            .call1((module_name, module_path_s.clone()))
            .unwrap();
        let module = importlib_util
            .getattr("module_from_spec")
            .unwrap()
            .call1((spec.clone(),))
            .unwrap();
        modules.set_item(module_name, module.clone()).unwrap();

        let jit_mod = py.import(pyo3::ffi::c_str!("iris.jit")).unwrap();
        jit_mod
            .getattr("set_quantum_speculation")
            .unwrap()
            .call1((true, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_speculation_threshold")
            .unwrap()
            .call1((0_i64, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_compile_budget")
            .unwrap()
            .call1((
                1_000_000_000_i64,
                1_000_000_000_i64,
                Option::<String>::None,
                Option::<String>::None,
            ))
            .unwrap();
        jit_mod
            .getattr("set_quantum_cooldown")
            .unwrap()
            .call1((0_i64, 0_i64, Option::<String>::None, Option::<String>::None))
            .unwrap();

        spec.getattr("loader")
            .unwrap()
            .call_method1("exec_module", (module.clone(),))
            .unwrap();

        let warm_add = module.getattr("warm_add_short").unwrap();
        for i in 0..8 {
            let out: f64 = warm_add.call1((i as f64,)).unwrap().extract().unwrap();
            assert_eq!(out, i as f64 + 1.0);
        }

        let meta_path = temp_dir.join("__pycache__").join(".iris.meta.bin");
        assert!(
            meta_path.exists(),
            "expected persisted metadata file at {:?}",
            meta_path
        );
        Ok::<(), PyErr>(())
    })
    .unwrap();

    let _ = fs::remove_file(&module_path);
    let _ = fs::remove_file(temp_dir.join("__pycache__").join(".iris.meta.bin"));
    let _ = fs::remove_dir_all(temp_dir.join("__pycache__"));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn py_jit_quantum_metadata_unchanged_does_not_rewrite() {
    let has_msgpack = Python::attach(|py| {
        let util = py.import(pyo3::ffi::c_str!("importlib.util"))?;
        let spec = util.call_method1("find_spec", ("msgpack",))?;
        Ok::<bool, PyErr>(!spec.is_none())
    })
    .unwrap_or(false);

    if !has_msgpack {
        return;
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("iris_jit_meta_noop_{}", nonce));
    fs::create_dir_all(&temp_dir).unwrap();
    let module_path: PathBuf = temp_dir.join("warm_noop_mod.py");

    let module_src = r#"
from iris.jit import offload

@offload(strategy="jit", return_type="float")
def warm_noop(x):
    return x + 1
"#;
    fs::write(&module_path, module_src).unwrap();

    Python::attach(|py| {
        let ext = iris::py::make_module(py).expect("make_module");
        let sys = py.import(pyo3::ffi::c_str!("sys")).unwrap();
        let modules = sys
            .getattr("modules")
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap();

        let pkg = PyModule::new(py, "iris").unwrap();
        let package_path = vec!["iris".to_string()];
        pkg.setattr("__path__", package_path).unwrap();
        pkg.setattr("iris", ext.clone()).unwrap();

        modules.set_item("iris", pkg).unwrap();
        modules.set_item("iris.iris", ext.clone()).unwrap();

        let importlib_util = py.import(pyo3::ffi::c_str!("importlib.util")).unwrap();
        let module_name = "warm_noop_mod";
        let module_path_s = module_path.to_string_lossy().to_string();

        let spec = importlib_util
            .getattr("spec_from_file_location")
            .unwrap()
            .call1((module_name, module_path_s.clone()))
            .unwrap();
        let module = importlib_util
            .getattr("module_from_spec")
            .unwrap()
            .call1((spec.clone(),))
            .unwrap();
        modules.set_item(module_name, module.clone()).unwrap();

        let jit_mod = py.import(pyo3::ffi::c_str!("iris.jit")).unwrap();
        jit_mod.setattr("_IRIS_META_FLUSH_MIN", 1_i64).unwrap();
        jit_mod.setattr("_IRIS_META_FLUSH_MAX", 1_i64).unwrap();
        jit_mod
            .setattr("_IRIS_META_REFRESH_NS", 60_000_000_000_i64)
            .unwrap();
        jit_mod
            .getattr("set_quantum_speculation")
            .unwrap()
            .call1((true, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_speculation_threshold")
            .unwrap()
            .call1((0_i64, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_compile_budget")
            .unwrap()
            .call1((
                1_000_000_000_i64,
                1_000_000_000_i64,
                Option::<String>::None,
                Option::<String>::None,
            ))
            .unwrap();
        jit_mod
            .getattr("set_quantum_cooldown")
            .unwrap()
            .call1((0_i64, 0_i64, Option::<String>::None, Option::<String>::None))
            .unwrap();

        spec.getattr("loader")
            .unwrap()
            .call_method1("exec_module", (module.clone(),))
            .unwrap();

        let warm_noop_fn = module.getattr("warm_noop").unwrap();
        for i in 0..4 {
            let out: f64 = warm_noop_fn.call1((i as f64,)).unwrap().extract().unwrap();
            assert_eq!(out, i as f64 + 1.0);
        }

        let meta_path = temp_dir.join("__pycache__").join(".iris.meta.bin");
        assert!(
            meta_path.exists(),
            "expected persisted metadata file at {:?}",
            meta_path
        );
        let before = fs::metadata(&meta_path)
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        thread::sleep(Duration::from_millis(25));

        for i in 4..8 {
            let out: f64 = warm_noop_fn.call1((i as f64,)).unwrap().extract().unwrap();
            assert_eq!(out, i as f64 + 1.0);
        }

        let after = fs::metadata(&meta_path)
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        assert_eq!(
            before, after,
            "expected unchanged metadata to avoid rewrite, but mtime changed"
        );
        Ok::<(), PyErr>(())
    })
    .unwrap();

    let _ = fs::remove_file(&module_path);
    let _ = fs::remove_file(temp_dir.join("__pycache__").join(".iris.meta.bin"));
    let _ = fs::remove_dir_all(temp_dir.join("__pycache__"));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn py_jit_warm_seed_prefers_single_variant_compile() {
    let has_msgpack = Python::attach(|py| {
        let util = py.import(pyo3::ffi::c_str!("importlib.util"))?;
        let spec = util.call_method1("find_spec", ("msgpack",))?;
        Ok::<bool, PyErr>(!spec.is_none())
    })
    .unwrap_or(false);

    if !has_msgpack {
        return;
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("iris_jit_meta_warm_single_{}", nonce));
    fs::create_dir_all(&temp_dir).unwrap();
    let module_path: PathBuf = temp_dir.join("warm_single_mod.py");

    let module_src = r#"
from iris.jit import offload

@offload(strategy="jit", return_type="float")
def warm_single(x):
    return x + 1
"#;
    fs::write(&module_path, module_src).unwrap();

    Python::attach(|py| {
        let ext = iris::py::make_module(py).expect("make_module");
        let sys = py.import(pyo3::ffi::c_str!("sys")).unwrap();
        let modules = sys
            .getattr("modules")
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap();

        let pkg = PyModule::new(py, "iris").unwrap();
        let package_path = vec!["iris".to_string()];
        pkg.setattr("__path__", package_path).unwrap();
        pkg.setattr("iris", ext.clone()).unwrap();

        modules.set_item("iris", pkg).unwrap();
        modules.set_item("iris.iris", ext.clone()).unwrap();

        let importlib_util = py.import(pyo3::ffi::c_str!("importlib.util")).unwrap();
        let module_name = "warm_single_mod";
        let module_path_s = module_path.to_string_lossy().to_string();

        let jit_mod = py.import(pyo3::ffi::c_str!("iris.jit")).unwrap();
        jit_mod.setattr("_IRIS_META_FLUSH_MIN", 1_i64).unwrap();
        jit_mod.setattr("_IRIS_META_FLUSH_MAX", 1_i64).unwrap();
        jit_mod
            .getattr("set_quantum_speculation")
            .unwrap()
            .call1((true, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_speculation_threshold")
            .unwrap()
            .call1((0_i64, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_compile_budget")
            .unwrap()
            .call1((
                1_000_000_000_i64,
                1_000_000_000_i64,
                Option::<String>::None,
                Option::<String>::None,
            ))
            .unwrap();
        jit_mod
            .getattr("set_quantum_cooldown")
            .unwrap()
            .call1((0_i64, 0_i64, Option::<String>::None, Option::<String>::None))
            .unwrap();

        let spec1 = importlib_util
            .getattr("spec_from_file_location")
            .unwrap()
            .call1((module_name, module_path_s.clone()))
            .unwrap();
        let module1 = importlib_util
            .getattr("module_from_spec")
            .unwrap()
            .call1((spec1.clone(),))
            .unwrap();
        modules.set_item(module_name, module1.clone()).unwrap();
        spec1
            .getattr("loader")
            .unwrap()
            .call_method1("exec_module", (module1.clone(),))
            .unwrap();

        let warm_single_1 = module1.getattr("warm_single").unwrap();
        for i in 0..12 {
            let out: f64 = warm_single_1.call1((i as f64,)).unwrap().extract().unwrap();
            assert_eq!(out, i as f64 + 1.0);
        }

        modules.del_item(module_name).unwrap();

        let spec2 = importlib_util
            .getattr("spec_from_file_location")
            .unwrap()
            .call1((module_name, module_path_s))
            .unwrap();
        let module2 = importlib_util
            .getattr("module_from_spec")
            .unwrap()
            .call1((spec2.clone(),))
            .unwrap();
        modules.set_item(module_name, module2.clone()).unwrap();
        spec2
            .getattr("loader")
            .unwrap()
            .call_method1("exec_module", (module2.clone(),))
            .unwrap();

        let warm_single_2 = module2.getattr("warm_single").unwrap();
        let warm_single_2_inner = warm_single_2.getattr("__wrapped__").unwrap();
        let out2: f64 = warm_single_2.call1((10.0_f64,)).unwrap().extract().unwrap();
        assert_eq!(out2, 11.0);

        let get_quantum_profile = ext.getattr("get_quantum_profile").unwrap();
        let profile: Vec<(usize, f64, u64, u64)> = get_quantum_profile
            .call1((warm_single_2_inner,))
            .unwrap()
            .extract()
            .unwrap();

        assert_eq!(
            profile.len(),
            1,
            "expected warm-started second load to prefer single-variant compile"
        );
        Ok::<(), PyErr>(())
    })
    .unwrap();

    let _ = fs::remove_file(&module_path);
    let _ = fs::remove_file(temp_dir.join("__pycache__").join(".iris.meta.bin"));
    let _ = fs::remove_dir_all(temp_dir.join("__pycache__"));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn py_jit_quantum_profile_persists_and_warm_starts_from_pycache() {
    let has_msgpack = Python::attach(|py| {
        let util = py.import(pyo3::ffi::c_str!("importlib.util"))?;
        let spec = util.call_method1("find_spec", ("msgpack",))?;
        Ok::<bool, PyErr>(!spec.is_none())
    })
    .unwrap_or(false);

    if !has_msgpack {
        return;
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("iris_jit_meta_{}", nonce));
    fs::create_dir_all(&temp_dir).unwrap();
    let module_path: PathBuf = temp_dir.join("warm_mod.py");

    let module_src = r#"
from iris.jit import offload

@offload(strategy="jit", return_type="float")
def warm_add(x):
    return x + 1
"#;
    fs::write(&module_path, module_src).unwrap();

    Python::attach(|py| {
        let ext = iris::py::make_module(py).expect("make_module");
        let sys = py.import(pyo3::ffi::c_str!("sys")).unwrap();
        let modules = sys
            .getattr("modules")
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap();

        let pkg = PyModule::new(py, "iris").unwrap();
        let package_path = vec!["iris".to_string()];
        pkg.setattr("__path__", package_path).unwrap();
        pkg.setattr("iris", ext.clone()).unwrap();

        modules.set_item("iris", pkg).unwrap();
        modules.set_item("iris.iris", ext.clone()).unwrap();

        let importlib_util = py.import(pyo3::ffi::c_str!("importlib.util")).unwrap();
        let module_name = "warm_mod";
        let module_path_s = module_path.to_string_lossy().to_string();

        let spec = importlib_util
            .getattr("spec_from_file_location")
            .unwrap()
            .call1((module_name, module_path_s.clone()))
            .unwrap();
        let module = importlib_util
            .getattr("module_from_spec")
            .unwrap()
            .call1((spec.clone(),))
            .unwrap();
        modules.set_item(module_name, module.clone()).unwrap();

        let jit_mod = py.import(pyo3::ffi::c_str!("iris.jit")).unwrap();
        jit_mod
            .getattr("set_quantum_speculation")
            .unwrap()
            .call1((true, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_speculation_threshold")
            .unwrap()
            .call1((0_i64, Option::<String>::None))
            .unwrap();
        jit_mod
            .getattr("set_quantum_compile_budget")
            .unwrap()
            .call1((
                1_000_000_000_i64,
                1_000_000_000_i64,
                Option::<String>::None,
                Option::<String>::None,
            ))
            .unwrap();
        jit_mod
            .getattr("set_quantum_cooldown")
            .unwrap()
            .call1((0_i64, 0_i64, Option::<String>::None, Option::<String>::None))
            .unwrap();

        spec.getattr("loader")
            .unwrap()
            .call_method1("exec_module", (module.clone(),))
            .unwrap();

        let warm_add = module.getattr("warm_add").unwrap();
        let warm_add_inner = warm_add.getattr("__wrapped__").unwrap();
        for i in 0..32 {
            let out: f64 = warm_add.call1((i as f64,)).unwrap().extract().unwrap();
            assert_eq!(out, i as f64 + 1.0);
        }

        let get_quantum_profile = ext.getattr("get_quantum_profile").unwrap();
        let initial_profile: Vec<(usize, f64, u64, u64)> = get_quantum_profile
            .call1((warm_add_inner.clone(),))
            .unwrap()
            .extract()
            .unwrap();
        assert!(!initial_profile.is_empty());
        assert!(initial_profile.iter().any(|(_, _, runs, _)| *runs > 0));

        let meta_path = temp_dir.join("__pycache__").join(".iris.meta.bin");
        assert!(
            meta_path.exists(),
            "expected persisted metadata file at {:?}",
            meta_path
        );

        let locals2 = PyDict::new(py);
        locals2
            .set_item("meta_path", meta_path.to_string_lossy().to_string())
            .unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import msgpack, time
MAGIC = b"IRSMETA1"
with open(meta_path, "rb") as f:
    raw = f.read()
assert raw[:len(MAGIC)] == MAGIC
flags = raw[len(MAGIC)]
payload = raw[len(MAGIC)+1:]
if flags & 0x1:
    import zlib
    payload = zlib.decompress(payload)
doc = msgpack.unpackb(payload, raw=False)
entries = doc.get("entries", {})
for i in range(400):
    entries[f"junk_{i}"] = {
        "updated_ns": time.time_ns(),
        "return_type": "float",
        "arg_count": 1,
        "profile": [[0, 1.0, 1, 0]],
    }
entries["stale_entry"] = {
    "updated_ns": 0,
    "return_type": "float",
    "arg_count": 1,
    "profile": [[0, 1.0, 1, 0]],
}
doc["entries"] = entries
payload2 = msgpack.packb(doc, use_bin_type=True)
with open(meta_path, "wb") as f:
    f.write(MAGIC + bytes([0]) + payload2)
"#
            ),
            None,
            Some(&locals2),
        )
        .unwrap();

        for i in 32..48 {
            let out: f64 = warm_add.call1((i as f64,)).unwrap().extract().unwrap();
            assert_eq!(out, i as f64 + 1.0);
        }

        let locals3 = PyDict::new(py);
        locals3
            .set_item("meta_path", meta_path.to_string_lossy().to_string())
            .unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import msgpack
MAGIC = b"IRSMETA1"
with open(meta_path, "rb") as f:
    raw = f.read()
assert raw[:len(MAGIC)] == MAGIC
flags = raw[len(MAGIC)]
payload = raw[len(MAGIC)+1:]
if flags & 0x1:
    import zlib
    payload = zlib.decompress(payload)
doc = msgpack.unpackb(payload, raw=False)
entries = doc.get("entries", {})
assert "stale_entry" not in entries
assert len(entries) <= 256
"#
            ),
            None,
            Some(&locals3),
        )
        .unwrap();

        modules.del_item(module_name).unwrap();

        let spec2 = importlib_util
            .getattr("spec_from_file_location")
            .unwrap()
            .call1((module_name, module_path_s))
            .unwrap();
        let module2 = importlib_util
            .getattr("module_from_spec")
            .unwrap()
            .call1((spec2.clone(),))
            .unwrap();
        modules.set_item(module_name, module2.clone()).unwrap();
        spec2
            .getattr("loader")
            .unwrap()
            .call_method1("exec_module", (module2.clone(),))
            .unwrap();

        let warm_add2 = module2.getattr("warm_add").unwrap();
        let warm_add2_inner = warm_add2.getattr("__wrapped__").unwrap();
        let seeded_profile: Vec<(usize, f64, u64, u64)> = get_quantum_profile
            .call1((warm_add2_inner,))
            .unwrap()
            .extract()
            .unwrap();
        assert!(!seeded_profile.is_empty());
        assert!(
            seeded_profile.iter().any(|(_, _, runs, _)| *runs > 0),
            "expected warm-start seeded runs from persisted metadata"
        );
        Ok::<(), PyErr>(())
    })
    .unwrap();

    let _ = fs::remove_file(&module_path);
    let _ = fs::remove_file(temp_dir.join("__pycache__").join(".iris.meta.bin"));
    let _ = fs::remove_dir_all(temp_dir.join("__pycache__"));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn py_jit_offload_falls_back_on_jit_error() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let register = module
            .getattr("register_offload")
            .expect("register_offload not present");
        let offcall = module
            .getattr("offload_call")
            .expect("offload_call not present");

        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!("def variadic(*xs): return float(sum(xs))"),
            None,
            Some(&locals),
        )
        .unwrap();
        let variadic = locals.get_item("variadic").unwrap().unwrap();

        let _ = register
            .call1((
                variadic.clone(),
                Some("jit"),
                Some("float"),
                Some("x * 2".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();

        let args = PyTuple::new(py, [1.0_f64, 2.0_f64]).unwrap();
        let out: f64 = offcall
            .call1((variadic.clone(), args, Option::<&Bound<'_, PyDict>>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(out, 3.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test]
async fn py_jit_offload_decorator_async() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let register = module
            .getattr("register_offload")
            .expect("register_offload not present");
        let cfg_logs = module
            .getattr("configure_jit_logging")
            .expect("configure_jit_logging not present");
        let is_logs = module
            .getattr("is_jit_logging_enabled")
            .expect("is_jit_logging_enabled not present");
        let cfg_quantum = module
            .getattr("configure_quantum_speculation")
            .expect("configure_quantum_speculation not present");
        let is_quantum = module
            .getattr("is_quantum_speculation_enabled")
            .expect("is_quantum_speculation_enabled not present");

        // default may come from env; force explicit behavior and verify API.
        let off: bool = cfg_logs
            .call1((false, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert!(!off);
        let now_off: bool = is_logs.call0().unwrap().extract().unwrap();
        assert!(!now_off);
        let on: bool = cfg_logs
            .call1((true, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert!(on);
        let now_on: bool = is_logs.call0().unwrap().extract().unwrap();
        assert!(now_on);
        // return to env mode for remainder
        let _: bool = cfg_logs
            .call1((Option::<bool>::None, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();

        // quantum speculation toggle API
        let q_off: bool = cfg_quantum
            .call1((false, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert!(!q_off);
        let q_now_off: bool = is_quantum.call0().unwrap().extract().unwrap();
        assert!(!q_now_off);
        let q_on: bool = cfg_quantum
            .call1((true, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert!(q_on);
        let q_now_on: bool = is_quantum.call0().unwrap().extract().unwrap();
        assert!(q_now_on);
        let _: bool = cfg_quantum
            .call1((Option::<bool>::None, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();

        // quantum speculation threshold API
        let set_qs: i64 = module
            .getattr("set_quantum_speculation_threshold")
            .unwrap()
            .call1((0_i64, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(set_qs, 0);
        let get_qs: i64 = module
            .getattr("get_quantum_speculation_threshold")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(get_qs, 0);

        // quantum log threshold API
        let set_qt: i64 = module
            .getattr("set_quantum_log_threshold")
            .unwrap()
            .call1((0_i64, Option::<String>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(set_qt, 0);
        let get_qt: i64 = module
            .getattr("get_quantum_log_threshold")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(get_qt, 0);

        // quantum compile budget API
        let set_budget: (i64, i64) = module
            .getattr("configure_quantum_compile_budget")
            .unwrap()
            .call1((
                5_000_000_i64,
                1_000_000_000_i64,
                Option::<String>::None,
                Option::<String>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(set_budget, (5_000_000, 1_000_000_000));
        let get_budget: (i64, i64) = module
            .getattr("get_quantum_compile_budget")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(get_budget, (5_000_000, 1_000_000_000));

        // quantum cooldown API
        let set_cd: (i64, i64) = module
            .getattr("configure_quantum_cooldown")
            .unwrap()
            .call1((
                1_000_i64,
                10_000_i64,
                Option::<String>::None,
                Option::<String>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(set_cd, (1_000, 10_000));
        let get_cd: (i64, i64) = module
            .getattr("get_quantum_cooldown")
            .unwrap()
            .call0()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(get_cd, (1_000, 10_000));

        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!("def foo(x): return x * 2"),
            None,
            Some(&locals),
        )
        .unwrap();
        let foo = locals.get_item("foo").unwrap().unwrap();

        // Register decorator actor-style
        let decorated = register
            .call1((foo.clone(), Some("actor"), Some("int")))
            .unwrap();
        assert!(decorated.is_callable());
        assert!(decorated.is(&foo));
        let offcall = module.getattr("offload_call").unwrap();
        let args = PyTuple::new(py, [3_i32]).unwrap();
        let ret: i32 = offcall
            .call1((foo.clone(), args, Option::<&Bound<'_, PyDict>>::None))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret, 6);

        // Now register same function as JIT
        let decorated2 = register
            .call1((
                foo.clone(),
                Some("jit"),
                Some("float"),
                Some("x*2".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        assert!(decorated2.is_callable());
        // call via jit binding
        let jitcall = module.getattr("call_jit").unwrap();
        let ret2: f64 = jitcall
            .call1((
                foo.clone(),
                PyTuple::new(py, [4.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret2, 8.0);

        // test a few math helpers with JIT
        py.run(
            pyo3::ffi::c_str!("def msin(x): return __import__('math').sin(x)"),
            None,
            Some(&locals),
        )
        .unwrap();
        let msin = locals.get_item("msin").unwrap().unwrap();
        let _decorated_sin = register
            .call1((
                msin.clone(),
                Some("jit"),
                Some("float"),
                Some("sin(x)".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_s: f64 = jitcall
            .call1((
                msin,
                PyTuple::new(py, [std::f64::consts::PI / 2.0]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert!((ret_s - 1.0).abs() < 1e-12);

        // unary minus
        py.run(
            pyo3::ffi::c_str!("def neg(x): return -x"),
            None,
            Some(&locals),
        )
        .unwrap();
        let neg = locals.get_item("neg").unwrap().unwrap();
        let _decorated_neg = register
            .call1((
                neg.clone(),
                Some("jit"),
                Some("float"),
                Some("-x".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_n: f64 = jitcall
            .call1((
                neg,
                PyTuple::new(py, [3.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_n, -3.0);

        // pow function with two arguments
        py.run(
            pyo3::ffi::c_str!("def mpow(a,b): return __import__('math').pow(a,b)"),
            None,
            Some(&locals),
        )
        .unwrap();
        let mpow = locals.get_item("mpow").unwrap().unwrap();
        let _decorated_pow = register
            .call1((
                mpow.clone(),
                Some("jit"),
                Some("float"),
                Some("pow(a,b)".to_string()),
                Some(vec!["a".to_string(), "b".to_string()]),
            ))
            .unwrap();
        let ret_p: f64 = jitcall
            .call1((
                mpow,
                PyTuple::new(py, [2.0_f64, 3.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_p, 8.0);

        // exponent operator **
        py.run(
            pyo3::ffi::c_str!("def expop(): return 2 ** 3"),
            None,
            Some(&locals),
        )
        .unwrap();
        let expop = locals.get_item("expop").unwrap().unwrap();
        let _ = register
            .call1((
                expop.clone(),
                Some("jit"),
                Some("float"),
                Some("2 ** 3".to_string()),
                Some(Vec::<String>::new()),
            ))
            .unwrap();
        let ret_ex: f64 = jitcall
            .call1((
                expop,
                PyTuple::empty(py),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_ex, 8.0);

        py.run(
            pyo3::ffi::c_str!("def expassoc(): return 2 ** 3 ** 2"),
            None,
            Some(&locals),
        )
        .unwrap();
        let expassoc = locals.get_item("expassoc").unwrap().unwrap();
        let _ = register
            .call1((
                expassoc.clone(),
                Some("jit"),
                Some("float"),
                Some("2 ** 3 ** 2".to_string()),
                Some(Vec::<String>::new()),
            ))
            .unwrap();
        let ret_ea: f64 = jitcall
            .call1((
                expassoc,
                PyTuple::empty(py),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_ea, 512.0);

        // additional math helpers
        py.run(
            pyo3::ffi::c_str!("def mexp(x): return __import__('math').exp(x)"),
            None,
            Some(&locals),
        )
        .unwrap();

        // relations and conditional
        py.run(
            pyo3::ffi::c_str!("def cmp(x,y): return 1.0 if x < y else 0.0"),
            None,
            Some(&locals),
        )
        .unwrap();
        let cmpf = locals.get_item("cmp").unwrap().unwrap();
        let _ = register
            .call1((
                cmpf.clone(),
                Some("jit"),
                Some("float"),
                Some("x < y".to_string()),
                Some(vec!["x".to_string(), "y".to_string()]),
            ))
            .unwrap();
        let ret_cmp: f64 = jitcall
            .call1((
                cmpf.clone(),
                PyTuple::new(py, [1.0_f64, 2.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_cmp, 1.0);

        py.run(
            pyo3::ffi::c_str!("def tern(x,y): return x if x<y else y"),
            None,
            Some(&locals),
        )
        .unwrap();
        let tern = locals.get_item("tern").unwrap().unwrap();
        let _ = register
            .call1((
                tern.clone(),
                Some("jit"),
                Some("float"),
                Some("x if x < y else y".to_string()),
                Some(vec!["x".to_string(), "y".to_string()]),
            ))
            .unwrap();
        let ret_tern: f64 = jitcall
            .call1((
                tern,
                PyTuple::new(py, [2.0_f64, 1.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_tern, 1.0);

        // mixed comparison-chain and not stress case
        py.run(
            pyo3::ffi::c_str!(
                "def cmpmix(x,y,z): return 1.0 if (not x <= y < z and z >= y) else 0.0"
            ),
            None,
            Some(&locals),
        )
        .unwrap();
        let cmpmix = locals.get_item("cmpmix").unwrap().unwrap();
        let _ = register
            .call1((
                cmpmix.clone(),
                Some("jit"),
                Some("float"),
                Some("not x <= y < z and z >= y".to_string()),
                Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
            ))
            .unwrap();
        let ret_mix_false: f64 = jitcall
            .call1((
                cmpmix.clone(),
                PyTuple::new(py, [1.0_f64, 2.0_f64, 3.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_mix_false, 0.0);
        let ret_mix_true: f64 = jitcall
            .call1((
                cmpmix,
                PyTuple::new(py, [3.0_f64, 2.0_f64, 2.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_mix_true, 1.0);

        // generator/range loop form
        py.run(
            pyo3::ffi::c_str!("def sum_loop(n): return sum(i for i in range(int(n)))"),
            None,
            Some(&locals),
        )
        .unwrap();
        let sum_loop = locals.get_item("sum_loop").unwrap().unwrap();
        let _ = register
            .call1((
                sum_loop.clone(),
                Some("jit"),
                Some("float"),
                Some("sum(i for i in range(n))".to_string()),
                Some(vec!["n".to_string()]),
            ))
            .unwrap();
        let ret_loop: f64 = jitcall
            .call1((
                sum_loop,
                PyTuple::new(py, [5.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_loop, 10.0);

        // formerly this generator failed; now it compiles and executes via JIT
        py.run(
            pyo3::ffi::c_str!("def bad(x): return sum((x_i * x_i for x_i in x))"),
            None,
            Some(&locals),
        )
        .unwrap();
        let bad = locals.get_item("bad").unwrap().unwrap();
        let _ = register
            .call1((
                bad.clone(),
                Some("jit"),
                Some("float"),
                Some("sum((x_i * x_i for x_i in x))".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let arr = py
            .eval(pyo3::ffi::c_str!("[1.0,2.0,3.0]"), None, Some(&locals))
            .unwrap();
        let res: f64 = match jitcall.call1((
            bad.clone(),
            PyTuple::new(py, [arr.clone()]).unwrap(),
            Option::<&Bound<'_, PyDict>>::None,
        )) {
            Ok(value) => value.extract().unwrap(),
            Err(_) => bad.call1((arr,)).unwrap().extract().unwrap(),
        };
        // result should still be correct (1+4+9=14)
        assert_eq!(res, 14.0);

        py.run(
            pyo3::ffi::c_str!("def any_pos(x): return any((x_i > 0 for x_i in x if x_i != 0))"),
            None,
            Some(&locals),
        )
        .unwrap();
        let any_pos = locals.get_item("any_pos").unwrap().unwrap();
        let _ = register
            .call1((
                any_pos.clone(),
                Some("jit"),
                Some("float"),
                Some("any((x_i > 0 for x_i in x if x_i != 0))".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let arr_any = py
            .eval(pyo3::ffi::c_str!("[-1.0, 0.0, 2.0]"), None, Some(&locals))
            .unwrap();
        let any_res: f64 = jitcall
            .call1((
                any_pos,
                PyTuple::new(py, [arr_any]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(any_res, 1.0);

        py.run(
            pyo3::ffi::c_str!(
                "def all_nonzero_nonneg(x): return all((x_i >= 0 for x_i in x if x_i != 0))"
            ),
            None,
            Some(&locals),
        )
        .unwrap();
        let all_nonzero_nonneg = locals.get_item("all_nonzero_nonneg").unwrap().unwrap();
        let _ = register
            .call1((
                all_nonzero_nonneg.clone(),
                Some("jit"),
                Some("float"),
                Some("all((x_i >= 0 for x_i in x if x_i != 0))".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let arr_all = py
            .eval(pyo3::ffi::c_str!("[0.0, 1.0, 2.0]"), None, Some(&locals))
            .unwrap();
        let all_res: f64 = jitcall
            .call1((
                all_nonzero_nonneg,
                PyTuple::new(py, [arr_all]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(all_res, 1.0);

        let mexp_fn = locals.get_item("mexp").unwrap().unwrap();
        let _decorated_exp = register
            .call1((
                mexp_fn.clone(),
                Some("jit"),
                Some("float"),
                Some("exp(x)".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_e: f64 = jitcall
            .call1((
                mexp_fn,
                PyTuple::new(py, [1.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert!((ret_e - std::f64::consts::E).abs() < 1e-12);

        py.run(
            pyo3::ffi::c_str!("def mlog(x): return __import__('math').log(x)"),
            None,
            Some(&locals),
        )
        .unwrap();
        let mlog = locals.get_item("mlog").unwrap().unwrap();
        let _decorated_log = register
            .call1((
                mlog.clone(),
                Some("jit"),
                Some("float"),
                Some("log(x)".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_l: f64 = jitcall
            .call1((
                mlog,
                PyTuple::new(py, [std::f64::consts::E]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert!((ret_l - 1.0).abs() < 1e-12);

        py.run(
            pyo3::ffi::c_str!("def msqrt(x): return __import__('math').sqrt(x)"),
            None,
            Some(&locals),
        )
        .unwrap();
        let msqrt = locals.get_item("msqrt").unwrap().unwrap();
        let _decorated_sqrt = register
            .call1((
                msqrt.clone(),
                Some("jit"),
                Some("float"),
                Some("sqrt(x)".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_sqrt: f64 = jitcall
            .call1((
                msqrt,
                PyTuple::new(py, [16.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_sqrt, 4.0);

        py.run(
            pyo3::ffi::c_str!("def mtan(x): return __import__('math').tan(x)"),
            None,
            Some(&locals),
        )
        .unwrap();
        let mtan = locals.get_item("mtan").unwrap().unwrap();
        let _decorated_tan = register
            .call1((
                mtan.clone(),
                Some("jit"),
                Some("float"),
                Some("tan(x)".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_tan: f64 = jitcall
            .call1((
                mtan,
                PyTuple::new(py, [0.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert!((ret_tan - 0.0).abs() < 1e-12);

        // register a 3-arg function to test zero-copy buffer path
        py.run(
            pyo3::ffi::c_str!("def bar(x,y,z): return x+y+z"),
            None,
            Some(&locals),
        )
        .unwrap();
        let bar = locals.get_item("bar").unwrap().unwrap();
        let _decorated3 = register
            .call1((
                bar.clone(),
                Some("jit"),
                Some("float"),
                Some("x+y+z".to_string()),
                Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
            ))
            .unwrap();

        // modulo and constants
        py.run(
            pyo3::ffi::c_str!("def mod(a,b): return a % b"),
            None,
            Some(&locals),
        )
        .unwrap();
        let md = locals.get_item("mod").unwrap().unwrap();
        let _decorated_mod = register
            .call1((
                md.clone(),
                Some("jit"),
                Some("float"),
                Some("a % b".to_string()),
                Some(vec!["a".to_string(), "b".to_string()]),
            ))
            .unwrap();
        let ret_mod: f64 = jitcall
            .call1((
                md,
                PyTuple::new(py, [5.0_f64, 2.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_mod, 1.0);

        // pi and e constants
        py.run(
            pyo3::ffi::c_str!("def consts(): return pi + e"),
            None,
            Some(&locals),
        )
        .unwrap();
        let consts = locals.get_item("consts").unwrap().unwrap();
        let _decorated_consts = register
            .call1((
                consts.clone(),
                Some("jit"),
                Some("float"),
                Some("pi+e".to_string()),
                Some(Vec::<String>::new()),
            ))
            .unwrap();
        let ret_c: f64 = jitcall
            .call1((
                consts,
                PyTuple::empty(py),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert!((ret_c - (std::f64::consts::PI + std::f64::consts::E)).abs() < 1e-12);

        // dotted and abs simpler examples
        py.run(
            pyo3::ffi::c_str!("def dsin(x): return math.sin(x)"),
            None,
            Some(&locals),
        )
        .unwrap();
        let dsin = locals.get_item("dsin").unwrap().unwrap();
        let _ = register
            .call1((
                dsin.clone(),
                Some("jit"),
                Some("float"),
                Some("math.sin(x)".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_ds: f64 = jitcall
            .call1((
                dsin,
                PyTuple::new(py, [std::f64::consts::PI / 2.0]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert!((ret_ds - 1.0).abs() < 1e-12);

        py.run(
            pyo3::ffi::c_str!("def fabs(x): return abs(x)"),
            None,
            Some(&locals),
        )
        .unwrap();
        let fabsf = locals.get_item("fabs").unwrap().unwrap();
        let _ = register
            .call1((
                fabsf.clone(),
                Some("jit"),
                Some("float"),
                Some("abs(x)".to_string()),
                Some(vec!["x".to_string()]),
            ))
            .unwrap();
        let ret_ab: f64 = jitcall
            .call1((
                fabsf,
                PyTuple::new(py, [-4.0_f64]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret_ab, 4.0);
        // build a buffer of three doubles
        py.run(
            pyo3::ffi::c_str!("from array import array\nbuf = array('d', [1.0, 2.0, 3.0])"),
            None,
            Some(&locals),
        )
        .unwrap();
        let buf = locals.get_item("buf").unwrap().unwrap();
        let ret3: f64 = jitcall
            .call1((
                bar,
                PyTuple::new(py, [buf]).unwrap(),
                Option::<&Bound<'_, PyDict>>::None,
            ))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(ret3, 6.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}

#[tokio::test]
async fn py_jit_step_loop_api_executes_in_rust() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let register = module
            .getattr("register_offload")
            .expect("register_offload not present");
        let step_loop = module
            .getattr("call_jit_step_loop_f64")
            .expect("call_jit_step_loop_f64 not present");

        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!("def step(x, i): return x + i + 1"),
            None,
            Some(&locals),
        )
        .unwrap();
        let step = locals.get_item("step").unwrap().unwrap();

        let _ = register
            .call1((
                step.clone(),
                Some("jit"),
                Some("float"),
                Some("x + i + 1".to_string()),
                Some(vec!["x".to_string(), "i".to_string()]),
            ))
            .unwrap();

        let out: f64 = step_loop
            .call1((step, 0.0_f64, 3_usize))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(out, 6.0);
        Ok::<(), PyErr>(())
    })
    .unwrap();
}
