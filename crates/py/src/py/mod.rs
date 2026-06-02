// src/py/mod.rs
//! Perfectly categorized Python modules for Iris.
#![allow(non_local_definitions)]

pub mod actor;
pub mod ffi;
pub mod runtime;
pub mod utils;

// Re-exports for convenience
pub use ffi::{init, make_module};
