// src/vortex/transaction.rs
//! Experimental speculative transactional fiber ghosting stub.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexVioCall {
    pub op: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexGhostPolicy {
    FirstSafePointWins,
    PreferPrimary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexGhostResolution {
    pub winner_id: u64,
    pub loser_id: u64,
    pub committed_vio: Vec<VortexVioCall>,
}

#[derive(Debug, Clone)]
pub struct VortexTransaction {
    pub id: u64,
    pub committed: bool,
    pub aborted: bool,
    pub local_checkpoint: HashMap<String, Vec<u8>>,
    staged_vio: Vec<VortexVioCall>,
    committed_vio: Vec<VortexVioCall>,
}

impl VortexTransaction {
    pub fn new(id: u64) -> Self {
        VortexTransaction {
            id,
            committed: false,
            aborted: false,
            local_checkpoint: HashMap::new(),
            staged_vio: Vec::new(),
            committed_vio: Vec::new(),
        }
    }

    pub fn checkpoint_locals(&mut self, locals: HashMap<String, Vec<u8>>) {
        if self.committed || self.aborted {
            return;
        }
        self.local_checkpoint = locals;
    }

    pub fn stage_vio(&mut self, op: String, payload: Vec<u8>) -> bool {
        if self.committed || self.aborted {
            return false;
        }
        self.staged_vio.push(VortexVioCall { op, payload });
        true
    }

    pub fn staged_vio_len(&self) -> usize {
        self.staged_vio.len()
    }

    pub fn committed_vio_len(&self) -> usize {
        self.committed_vio.len()
    }

    pub fn drain_committed_vio(&mut self) -> Vec<VortexVioCall> {
        std::mem::take(&mut self.committed_vio)
    }

    pub fn resolve_ghost_race(
        primary: &mut VortexTransaction,
        ghost: &mut VortexTransaction,
        winner_id: u64,
        policy: VortexGhostPolicy,
    ) -> Option<VortexGhostResolution> {
        let resolved_winner = match policy {
            VortexGhostPolicy::FirstSafePointWins => winner_id,
            VortexGhostPolicy::PreferPrimary => primary.id,
        };

        if resolved_winner == primary.id {
            if !primary.commit() {
                return None;
            }
            let _ = ghost.abort();
            return Some(VortexGhostResolution {
                winner_id: primary.id,
                loser_id: ghost.id,
                committed_vio: primary.drain_committed_vio(),
            });
        }

        if resolved_winner == ghost.id {
            if !ghost.commit() {
                return None;
            }
            let _ = primary.abort();
            return Some(VortexGhostResolution {
                winner_id: ghost.id,
                loser_id: primary.id,
                committed_vio: ghost.drain_committed_vio(),
            });
        }

        None
    }

    pub fn commit(&mut self) -> bool {
        if self.aborted || self.committed {
            return false;
        }
        self.committed_vio.append(&mut self.staged_vio);
        self.committed = true;
        true
    }

    pub fn abort(&mut self) -> bool {
        if self.committed || self.aborted {
            return false;
        }
        self.staged_vio.clear();
        self.aborted = true;
        true
    }
}
