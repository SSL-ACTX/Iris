use crate::pid::Pid;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

static LINK_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A Capability defines what an actor is allowed to do via a specific Link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Permission to send user messages.
    Send,
    /// Permission to send system signals (stop, etc).
    Signal,
    /// Permission to monitor lifecycle events.
    Monitor,
    /// Permission to spawn children under this actor.
    Spawn,
    /// Permission to delegate/copy this link to others.
    Delegate,
}

/// A Link is an opaque object capability that allows communication with an actor.
/// Unlike a raw Pid, a Link carries specific permissions and contracts.
#[derive(Debug, Clone)]
pub struct Link {
    pub(crate) target: Pid,
    pub capabilities: HashSet<Capability>,
    /// Unique identifier for this link instance (for revocation).
    pub(crate) link_id: u64,
}

impl Link {
    pub fn new(target: Pid, capabilities: Vec<Capability>) -> Self {
        Self {
            target,
            capabilities: capabilities.into_iter().collect(),
            link_id: LINK_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
        }
    }

    pub fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    pub fn target(&self) -> Pid {
        self.target
    }

    pub fn id(&self) -> u64 {
        self.link_id
    }

    /// Create a restricted version of this link.
    pub fn restrict(&self, caps: Vec<Capability>) -> Option<Self> {
        let mut new_caps = HashSet::new();
        for c in caps {
            if self.capabilities.contains(&c) {
                new_caps.insert(c);
            } else {
                return None; // Cannot escalate privileges
            }
        }
        Some(Self {
            target: self.target,
            capabilities: new_caps,
            link_id: LINK_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
        })
    }

    /// Helper to create a link with full permissions.
    pub fn full(target: Pid) -> Self {
        Self::new(
            target,
            vec![
                Capability::Send,
                Capability::Signal,
                Capability::Monitor,
                Capability::Spawn,
                Capability::Delegate,
            ],
        )
    }
}
