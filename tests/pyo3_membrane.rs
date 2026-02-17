#![cfg(feature = "pyo3")]

use pyo3::prelude::*;

// TDD: expect `myrmidon::py::make_module` and `version()` to exist — failing test
#[test]
fn make_module_exposes_version() {
    Python::with_gil(|py| {
        // `make_module` does not exist yet — this test should fail initially.
        let module = myrmidon::py::make_module(py).expect("make_module failed");
        let ver_obj = module.as_ref(py).getattr("version").expect("no version attr");
        let ver: String = ver_obj.call0().unwrap().extract().unwrap();
        assert_eq!(ver, env!("CARGO_PKG_VERSION"));
    });
}
