//! Phase 2 (PyO3) — failing test (Strict TDD)
//! This test is intentionally failing until we implement the `myrmidon::py` API.

#![cfg(feature = "pyo3")]

#[test]
fn py_membrane_tdd_failing_test() {
    // Expected future API (doesn't exist yet): `myrmidon::py::init()`
    // This should fail now (TDD) and we'll implement the API next.
    myrmidon::py::init();
}
