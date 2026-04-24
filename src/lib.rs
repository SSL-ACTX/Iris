//! Iris — core runtime
//!
//! This crate contains the core logic for PID allocation, mailboxes,
//! cooperative scheduling, distributed networking, name registration,
//! and remote service discovery.

pub mod core;

pub use crate::core::buffer;
pub use crate::core::logging;
pub use crate::core::mailbox;
pub use crate::core::network;
pub use crate::core::pid;
pub use crate::core::registry;
pub use crate::core::supervisor;

#[cfg(feature = "vortex")]
pub use crate::core::vortex;

#[cfg(feature = "pyo3")]
pub mod py;

#[cfg(feature = "node")]
pub mod node;

pub use crate::core::Runtime;
pub(crate) use crate::core::RUNTIME;
