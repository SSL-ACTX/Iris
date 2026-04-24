// src/vortex/mod.rs
//! Experimental Vortex subsystem (Feature gated)
//!
//! This module is intentionally isolated and opt-in. It is designed to host
//! Vortex experiments including deterministic preemption and transactional
//! actor execution strategies.

#[cfg(feature = "vortex")]
pub mod engine;

#[cfg(feature = "vortex")]
pub mod scheduler;

#[cfg(feature = "vortex")]
pub mod watcher;

#[cfg(feature = "vortex")]
pub mod transmuter;

#[cfg(feature = "vortex")]
pub mod transaction;

#[cfg(feature = "vortex")]
pub mod vortex_bytecode;

#[cfg(feature = "vortex")]
pub mod rescue_pool;

#[cfg(feature = "vortex")]
pub mod ocular;

#[cfg(feature = "vortex")]
pub use engine::VortexEngine;

#[cfg(feature = "vortex")]
pub use scheduler::VortexScheduler;

#[cfg(feature = "vortex")]
pub use watcher::VortexWatcher;

#[cfg(feature = "vortex")]
pub use transmuter::VortexTransmuter;

#[cfg(feature = "vortex")]
pub use transaction::{VortexGhostPolicy, VortexGhostResolution, VortexTransaction, VortexVioCall};

#[cfg(feature = "vortex")]
pub use rescue_pool::RescuePool;

#[cfg(feature = "vortex")]
pub use transmuter::{VortexExecutionContext, VortexInstruction, VortexSuspend};
