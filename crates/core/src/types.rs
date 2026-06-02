// src/core/types.rs
use crate::mailbox;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::time::Duration;

pub type BoxFutureUnit = Pin<Box<dyn Future<Output = ()> + Send>>;
pub type ErasedMessageHandler = Arc<dyn Fn(mailbox::Message) -> BoxFutureUnit + Send + Sync>;
pub const MAX_BEHAVIOR_HISTORY: usize = 16;

pub fn next_dynamic_budget(
    current: usize,
    base: usize,
    saw_suspend: bool,
    suspend_rate: f64,
    low_thresh: f64,
    high_thresh: f64,
) -> usize {
    let base = base.max(1);
    let min_budget = (base / 4).max(1);
    let max_budget = base.saturating_mul(4).max(base);

    let adjusted = if suspend_rate > high_thresh {
        (current * 60 / 100).max(min_budget)
    } else if suspend_rate > low_thresh {
        (current * 80 / 100).max(min_budget)
    } else if saw_suspend {
        (current / 2).max(min_budget)
    } else {
        (current.saturating_add(1)).min(max_budget)
    };

    adjusted.clamp(min_budget, max_budget)
}

#[derive(Clone)]
pub struct VirtualActorSpec {
    pub handler: ErasedMessageHandler,
    pub budget: usize,
    pub idle_timeout: Option<Duration>,
}
