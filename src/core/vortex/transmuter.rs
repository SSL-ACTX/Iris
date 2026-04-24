// src/vortex/transmuter.rs
//! Experimental bytecode-level transmuter and verifier.

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VortexInstruction {
    Nop,
    LoadFast(u16),
    StoreFast(u16),
    BinaryOp(u8),
    JumpForward(usize),
    JumpBackward(usize),
    PopJumpIfFalse(usize),
    IrisReductionCheck,
    ReturnValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexSuspend;

#[derive(Debug, Clone)]
pub struct VortexExecutionContext {
    pub pc: usize,
    pub stack_depth: i32,
    pub done: bool,
}

impl Default for VortexExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl VortexExecutionContext {
    pub fn new() -> Self {
        VortexExecutionContext {
            pc: 0,
            stack_depth: 0,
            done: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VortexTransmuter {
    pub enabled: bool,
    pub instruction_budget: usize,
}

impl VortexTransmuter {
    pub fn new(budget: usize) -> Self {
        VortexTransmuter {
            enabled: true,
            instruction_budget: budget,
        }
    }

    fn is_backward_branch(from_idx: usize, target_idx: usize) -> bool {
        target_idx <= from_idx
    }

    /// Inject IRIS_REDUCTION_CHECK at function entry and backward-branch targets.
    pub fn transmute(&self, code: &[VortexInstruction]) -> Vec<VortexInstruction> {
        let mut check_sites: HashSet<usize> = HashSet::new();
        if !code.is_empty() {
            check_sites.insert(0);
        }

        for (idx, instr) in code.iter().enumerate() {
            match instr {
                VortexInstruction::JumpBackward(dest) | VortexInstruction::PopJumpIfFalse(dest)
                    if *dest < code.len() && Self::is_backward_branch(idx, *dest) =>
                {
                    check_sites.insert(*dest);
                }
                _ => {}
            }
        }

        let mut output: Vec<VortexInstruction> = Vec::new();
        let mut mapping: Vec<usize> = vec![0; code.len()];

        for (idx, instr) in code.iter().enumerate() {
            mapping[idx] = output.len();
            if check_sites.contains(&idx) {
                output.push(VortexInstruction::IrisReductionCheck);
            }
            output.push(instr.clone());
        }

        let mut patched = output.clone();
        for (idx, instr) in output.iter().enumerate() {
            match instr {
                VortexInstruction::JumpForward(dest)
                | VortexInstruction::JumpBackward(dest)
                | VortexInstruction::PopJumpIfFalse(dest)
                    if *dest < mapping.len() =>
                {
                    let mapped = mapping[*dest];
                    patched[idx] = match instr {
                        VortexInstruction::JumpForward(_) => VortexInstruction::JumpForward(mapped),
                        VortexInstruction::JumpBackward(_) => {
                            VortexInstruction::JumpBackward(mapped)
                        }
                        VortexInstruction::PopJumpIfFalse(_) => {
                            VortexInstruction::PopJumpIfFalse(mapped)
                        }
                        _ => instr.clone(),
                    };
                }
                _ => {}
            }
        }

        patched
    }

    /// Return bytecode offsets where evaluation stack depth is zero.
    pub fn quiescence_points(&self, code: &[VortexInstruction]) -> Vec<usize> {
        let mut depth = 0i32;
        let mut points = Vec::new();

        for (idx, instr) in code.iter().enumerate() {
            match instr {
                VortexInstruction::LoadFast(_) => depth += 1,
                VortexInstruction::StoreFast(_) => depth = (depth - 1).max(0),
                VortexInstruction::BinaryOp(_) => depth = (depth - 1).max(0),
                VortexInstruction::PopJumpIfFalse(_) => depth = (depth - 1).max(0),
                _ => {}
            }
            if depth == 0 {
                points.push(idx);
            }
        }

        points
    }

    pub fn execute(&mut self, code: &[VortexInstruction]) -> Result<(), VortexSuspend> {
        let mut ctx = VortexExecutionContext::new();
        self.execute_with_context(code, &mut ctx)
    }

    pub fn execute_with_context(
        &mut self,
        code: &[VortexInstruction],
        ctx: &mut VortexExecutionContext,
    ) -> Result<(), VortexSuspend> {
        while !ctx.done && ctx.pc < code.len() {
            match &code[ctx.pc] {
                VortexInstruction::IrisReductionCheck => {
                    if self.instruction_budget == 0 {
                        return Err(VortexSuspend);
                    }
                    self.instruction_budget -= 1;
                    ctx.pc += 1;
                }
                VortexInstruction::LoadFast(_) => {
                    if self.instruction_budget == 0 {
                        return Err(VortexSuspend);
                    }
                    self.instruction_budget -= 1;
                    ctx.stack_depth += 1;
                    ctx.pc += 1;
                }
                VortexInstruction::StoreFast(_) => {
                    if self.instruction_budget == 0 {
                        return Err(VortexSuspend);
                    }
                    self.instruction_budget -= 1;
                    ctx.stack_depth = (ctx.stack_depth - 1).max(0);
                    ctx.pc += 1;
                }
                VortexInstruction::BinaryOp(_) => {
                    if self.instruction_budget == 0 {
                        return Err(VortexSuspend);
                    }
                    self.instruction_budget -= 1;
                    ctx.stack_depth = (ctx.stack_depth - 1).max(0);
                    ctx.pc += 1;
                }
                VortexInstruction::PopJumpIfFalse(dest) => {
                    if self.instruction_budget == 0 {
                        return Err(VortexSuspend);
                    }
                    self.instruction_budget -= 1;
                    ctx.stack_depth = (ctx.stack_depth - 1).max(0);
                    if ctx.stack_depth == 0 {
                        ctx.pc = *dest;
                    } else {
                        ctx.pc += 1;
                    }
                }
                VortexInstruction::JumpForward(dest) | VortexInstruction::JumpBackward(dest) => {
                    ctx.pc = *dest;
                }
                VortexInstruction::Nop => {
                    ctx.pc += 1;
                }
                VortexInstruction::ReturnValue => {
                    ctx.done = true;
                    return Ok(());
                }
            }
        }

        if ctx.pc >= code.len() {
            ctx.done = true;
            Ok(())
        } else {
            Err(VortexSuspend)
        }
    }

    pub fn inject_reduction_checks(&mut self, instruction_count: usize) -> bool {
        if !self.enabled {
            return false;
        }

        if instruction_count > self.instruction_budget {
            self.enabled = false;
            false
        } else {
            self.instruction_budget -= instruction_count;
            true
        }
    }
}
