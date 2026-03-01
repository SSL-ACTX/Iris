// src/py/mod.rs
//! Consolidated Python helpers split into submodules for clarity.
#![allow(non_local_definitions)]

pub mod pool;
pub mod utils;
pub mod mailbox;
pub mod runtime;
pub mod wrappers;

// re-export a few helpers for external callers (tests, build scripts, etc.)
pub use wrappers::{make_module, init};
