// src/py/jit/codegen/jit_types.rs
//! Shared type definitions used by the JIT codegen pipeline.

/// Symbol aliasing rules used for primitive function mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SymbolAlias {
    Identity,
    Rename(&'static str),
}

pub(crate) fn resolve_symbol_alias(symbol: &str, arg_count: usize) -> Option<SymbolAlias> {
    match (symbol, arg_count) {
        ("float", 1) => Some(SymbolAlias::Identity),
        ("int", 1) => Some(SymbolAlias::Rename("trunc")),
        ("round", 1) => Some(SymbolAlias::Rename("round")),
        _ => None,
    }
}

/// A compiled function entry returned by the JIT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionMode {
    None,
    Sum,
    Any,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JitReturnType {
    #[default]
    Float,
    Int,
    Bool,
}

#[derive(Clone)]
pub struct JitEntry {
    pub func_ptr: usize,
    pub arg_count: usize,
    pub reduction: ReductionMode,
    pub return_type: JitReturnType,
    pub(crate) lowered_kernel: Option<LoweredKernel>,
    pub(crate) variant_strategy: QuantumVariantStrategy,
}

impl JitEntry {
    /// Check if this entry has a valid, compiled function pointer.
    pub fn is_valid(&self) -> bool {
        self.func_ptr != 0
    }

    pub fn with_strategy(mut self, strategy: QuantumVariantStrategy) -> Self {
        self.variant_strategy = strategy;
        self
    }

    pub(crate) fn allows_lowered_kernel(&self) -> bool {
        self.variant_strategy != QuantumVariantStrategy::ScalarFallback
    }

    pub(crate) fn loop_unroll_for_entry(&self) -> usize {
        match self.variant_strategy {
            QuantumVariantStrategy::ScalarFallback => 1,
            _ => super::simd_unroll_factor(),
        }
    }

    pub(crate) fn math_mode_for_entry(&self) -> SimdMathMode {
        match self.variant_strategy {
            QuantumVariantStrategy::FastTrigExperiment => SimdMathMode::FastApprox,
            _ => super::simd_math_mode_from_env(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweredUnaryKernel {
    Identity,
    Neg,
    Abs,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoweredBinaryKernel {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LoweredExpr {
    Const(f64),
    Input(usize),
    Add(Box<LoweredExpr>, Box<LoweredExpr>),
    Sub(Box<LoweredExpr>, Box<LoweredExpr>),
    Mul(Box<LoweredExpr>, Box<LoweredExpr>),
    Div(Box<LoweredExpr>, Box<LoweredExpr>),
    Mod(Box<LoweredExpr>, Box<LoweredExpr>),
    Eq(Box<LoweredExpr>, Box<LoweredExpr>),
    Ne(Box<LoweredExpr>, Box<LoweredExpr>),
    Lt(Box<LoweredExpr>, Box<LoweredExpr>),
    Gt(Box<LoweredExpr>, Box<LoweredExpr>),
    Le(Box<LoweredExpr>, Box<LoweredExpr>),
    Ge(Box<LoweredExpr>, Box<LoweredExpr>),
    And(Box<LoweredExpr>, Box<LoweredExpr>),
    Or(Box<LoweredExpr>, Box<LoweredExpr>),
    Neg(Box<LoweredExpr>),
    Not(Box<LoweredExpr>),
    Abs(Box<LoweredExpr>),
    Sin(Box<LoweredExpr>),
    Cos(Box<LoweredExpr>),
    Tan(Box<LoweredExpr>),
    Exp(Box<LoweredExpr>),
    Log(Box<LoweredExpr>),
    Sqrt(Box<LoweredExpr>),
    Ternary {
        cond: Box<LoweredExpr>,
        then_expr: Box<LoweredExpr>,
        else_expr: Box<LoweredExpr>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LoweredKernel {
    Unary {
        op: LoweredUnaryKernel,
        input: usize,
    },
    Binary {
        op: LoweredBinaryKernel,
        lhs: usize,
        rhs: usize,
    },
    Expr(LoweredExpr),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SimdMathMode {
    Accurate,
    FastApprox,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantumVariantStrategy {
    Auto,
    ScalarFallback,
    FastTrigExperiment,
}

/// Point in a quantum profile snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuantumProfilePoint {
    pub index: usize,
    pub ewma_ns: f64,
    pub runs: u64,
    pub failures: u64,
}

/// Seed point used to initialize/override quantum profiling decisions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuantumProfileSeed {
    pub index: usize,
    pub ewma_ns: f64,
    pub runs: u64,
    pub failures: u64,
}
