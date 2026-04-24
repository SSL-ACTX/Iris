// src/vortex/engine.rs
//! Experimental Vortex runtime engine.

use std::collections::HashMap;

use crate::vortex::rescue_pool::RescuePool;
use crate::vortex::transaction::{
    VortexGhostPolicy, VortexGhostResolution, VortexTransaction, VortexVioCall,
};
use crate::vortex::transmuter::{
    VortexExecutionContext, VortexInstruction, VortexSuspend, VortexTransmuter,
};

#[derive(Debug, Clone)]
pub struct VortexEngine {
    pub enabled: bool,
    pub transmuter: VortexTransmuter,
    pub transaction: Option<VortexTransaction>,
    pub rescue_pool: RescuePool,
    pub budget: usize,
    pub current_code: Option<Vec<VortexInstruction>>,
    pub context: Option<VortexExecutionContext>,
    pub pending_code_swap: Option<Vec<VortexInstruction>>,
    ghost_transactions: HashMap<u64, VortexTransaction>,
}

impl Default for VortexEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl VortexEngine {
    pub fn new() -> Self {
        VortexEngine {
            enabled: true,
            transmuter: VortexTransmuter::new(1024),
            transaction: None,
            rescue_pool: RescuePool::new(),
            budget: 1024,
            current_code: None,
            context: None,
            pending_code_swap: None,
            ghost_transactions: HashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn check_preemption(&mut self, instruction_cost: usize) -> bool {
        if !self.enabled {
            return false;
        }

        let should_continue = self.transmuter.inject_reduction_checks(instruction_cost);
        if !should_continue {
            self.enabled = false;
        }

        should_continue
    }

    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        self.transmuter.instruction_budget = budget;
        self.enabled = true;
    }

    /// Consume one reduction tick (Check) via transmuter instrumentation. Returns Err(VortexSuspend) if budget is exhausted.
    pub fn preempt_tick(&mut self) -> Result<(), VortexSuspend> {
        if !self.enabled {
            return Err(VortexSuspend);
        }

        // Execute only the injected reduction check opcode.
        let program = vec![
            VortexInstruction::IrisReductionCheck,
            VortexInstruction::ReturnValue,
        ];
        let mut ctx = VortexExecutionContext::new();
        match self.transmuter.execute_with_context(&program, &mut ctx) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.enabled = false;
                Err(e)
            }
        }
    }

    pub fn load_code(&mut self, code: Vec<VortexInstruction>) {
        let transmuted = self.transmuter.transmute(&code);
        self.current_code = Some(transmuted);
        self.context = Some(VortexExecutionContext::new());
    }

    pub fn stage_code_swap(&mut self, code: Vec<VortexInstruction>) {
        let transmuted = self.transmuter.transmute(&code);
        self.pending_code_swap = Some(transmuted);
    }

    pub fn try_apply_staged_swap(&mut self) -> bool {
        let Some(staged) = self.pending_code_swap.clone() else {
            return false;
        };

        // Idle actor: swap immediately.
        if self.current_code.is_none() || self.context.is_none() {
            self.current_code = Some(staged);
            self.context = Some(VortexExecutionContext::new());
            self.pending_code_swap = None;
            return true;
        }

        // Mid-execution: only swap at quiescent points (stack depth zero).
        if let (Some(code), Some(ctx)) = (&self.current_code, &self.context) {
            let points = self.transmuter.quiescence_points(code);
            if ctx.done || (ctx.stack_depth == 0 && points.contains(&ctx.pc)) {
                self.current_code = Some(staged);
                self.context = Some(VortexExecutionContext::new());
                self.pending_code_swap = None;
                return true;
            }
        }

        false
    }

    pub fn run(&mut self) -> Result<(), VortexSuspend> {
        if !self.enabled {
            return Err(VortexSuspend);
        }

        if let (Some(code), Some(ctx)) = (&self.current_code, &mut self.context) {
            let result = self.transmuter.execute_with_context(code, ctx);
            if result.is_ok() {
                // When a staged swap exists, apply it on completion; otherwise clear actor state.
                if self.pending_code_swap.is_some() {
                    let _ = self.try_apply_staged_swap();
                } else {
                    self.current_code = None;
                    self.context = None;
                }
            }
            result
        } else {
            Ok(())
        }
    }

    pub fn replenish_budget(&mut self, amount: usize) {
        self.transmuter.instruction_budget += amount;
        self.budget = self.transmuter.instruction_budget;
        self.enabled = true;
    }

    pub fn start_transaction(&mut self, id: u64) {
        self.transaction = Some(VortexTransaction::new(id));
    }

    pub fn start_transaction_with_checkpoint(&mut self, id: u64, locals: HashMap<String, Vec<u8>>) {
        let mut trx = VortexTransaction::new(id);
        trx.checkpoint_locals(locals);
        self.transaction = Some(trx);
    }

    pub fn stage_transaction_vio(&mut self, op: String, payload: Vec<u8>) -> bool {
        match &mut self.transaction {
            Some(trx) => trx.stage_vio(op, payload),
            None => false,
        }
    }

    pub fn start_ghost_transaction_with_checkpoint(
        &mut self,
        id: u64,
        locals: HashMap<String, Vec<u8>>,
    ) {
        let mut trx = VortexTransaction::new(id);
        trx.checkpoint_locals(locals);
        self.ghost_transactions.insert(id, trx);
    }

    pub fn stage_ghost_transaction_vio(
        &mut self,
        ghost_id: u64,
        op: String,
        payload: Vec<u8>,
    ) -> bool {
        match self.ghost_transactions.get_mut(&ghost_id) {
            Some(trx) => trx.stage_vio(op, payload),
            None => false,
        }
    }

    pub fn resolve_primary_ghost_race(
        &mut self,
        ghost_id: u64,
        winner_id: u64,
        policy: VortexGhostPolicy,
    ) -> Option<VortexGhostResolution> {
        let primary = self.transaction.as_mut()?;
        let ghost = self.ghost_transactions.get_mut(&ghost_id)?;

        let result = VortexTransaction::resolve_ghost_race(primary, ghost, winner_id, policy)?;
        self.ghost_transactions.remove(&ghost_id);
        Some(result)
    }

    pub fn replay_committed_vio_calls<F>(&self, calls: &[VortexVioCall], mut executor: F) -> usize
    where
        F: FnMut(&VortexVioCall) -> bool,
    {
        let mut applied = 0usize;
        for call in calls {
            applied = applied.saturating_add(1);
            if !executor(call) {
                break;
            }
        }
        applied
    }

    pub fn transaction_staged_vio_len(&self) -> usize {
        match &self.transaction {
            Some(trx) => trx.staged_vio_len(),
            None => 0,
        }
    }

    pub fn transaction_committed_vio_len(&self) -> usize {
        match &self.transaction {
            Some(trx) => trx.committed_vio_len(),
            None => 0,
        }
    }

    pub fn take_committed_transaction_vio(&mut self) -> Vec<VortexVioCall> {
        match &mut self.transaction {
            Some(trx) => trx.drain_committed_vio(),
            None => Vec::new(),
        }
    }

    pub fn commit_transaction(&mut self) -> bool {
        match &mut self.transaction {
            Some(trx) => trx.commit(),
            None => false,
        }
    }

    pub fn abort_transaction(&mut self) -> bool {
        match &mut self.transaction {
            Some(trx) => trx.abort(),
            None => false,
        }
    }

    pub fn detach_stalled_thread(&mut self) {
        self.rescue_pool.detach_thread();
    }

    pub fn reclaim_thread(&mut self) {
        self.rescue_pool.reclaim_thread();
    }
}
