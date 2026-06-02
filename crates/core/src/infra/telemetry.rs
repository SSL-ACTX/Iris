use crate::pid::Pid;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    ActorSpawned { pid: Pid, path: String },
    ActorStopped { pid: Pid },
    ActorCrashed { pid: Pid, reason: String },
    MessageSent { from: Pid, to: Pid, len: usize },
    MessageReceived { pid: Pid },
    MailboxFull { pid: Pid },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Starting,
    Ready,
    Busy,
    Degraded,
}

pub struct TelemetryManager {
    event_tx: mpsc::UnboundedSender<TelemetryEvent>,
    actor_count: AtomicU64,
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    mailbox_depths: Arc<DashMap<Pid, usize>>,
    health_statuses: Arc<DashMap<Pid, HealthStatus>>,
}

impl TelemetryManager {
    pub fn new() -> (Arc<Self>, mpsc::UnboundedReceiver<TelemetryEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let manager = Arc::new(Self {
            event_tx: tx,
            actor_count: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            mailbox_depths: Arc::new(DashMap::new()),
            health_statuses: Arc::new(DashMap::new()),
        });
        (manager, rx)
    }

    pub fn set_health(&self, pid: Pid, status: HealthStatus) {
        self.health_statuses.insert(pid, status);
    }

    pub fn get_health(&self, pid: Pid) -> HealthStatus {
        self.health_statuses
            .get(&pid)
            .map(|v| *v)
            .unwrap_or(HealthStatus::Ready)
    }

    pub fn log_event(&self, event: TelemetryEvent) {
        match &event {
            TelemetryEvent::ActorSpawned { .. } => {
                self.actor_count.fetch_add(1, Ordering::Relaxed);
            }
            TelemetryEvent::ActorStopped { pid } | TelemetryEvent::ActorCrashed { pid, .. } => {
                self.actor_count.fetch_sub(1, Ordering::Relaxed);
                self.mailbox_depths.remove(pid);
                self.health_statuses.remove(pid);
            }
            TelemetryEvent::MessageSent { .. } => {
                self.messages_sent.fetch_add(1, Ordering::Relaxed);
            }
            TelemetryEvent::MessageReceived { .. } => {
                self.messages_received.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        let _ = self.event_tx.send(event);
    }

    pub fn update_mailbox_depth(&self, pid: Pid, depth: usize) {
        self.mailbox_depths.insert(pid, depth);
    }

    pub fn get_actor_count(&self) -> u64 {
        self.actor_count.load(Ordering::Relaxed)
    }

    pub fn get_messages_sent(&self) -> u64 {
        self.messages_sent.load(Ordering::Relaxed)
    }

    pub fn get_messages_received(&self) -> u64 {
        self.messages_received.load(Ordering::Relaxed)
    }

    pub fn get_mailbox_depth(&self, pid: Pid) -> usize {
        self.mailbox_depths.get(&pid).map(|v| *v).unwrap_or(0)
    }
}
