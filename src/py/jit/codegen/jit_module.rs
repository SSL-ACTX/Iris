// src/py/jit/codegen/jit_module.rs
//! Cranelift JIT module setup and shared execution profiling state.

use std::collections::HashMap;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};

use crate::py::jit::simd;

/// A small lightweight type that describes what kind of executable function
/// profile we recorded for a particular JIT function pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum JitExecProfile {
    ScalarArgs,
    PackedBuffer {
        arg_count: usize,
        elem: BufferElemType,
    },
    VectorizedBuffers {
        arg_count: usize,
        elem_types: Vec<BufferElemType>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BufferElemType {
    F64,
    F32,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

thread_local! {
    static TLS_JIT_MODULE: std::cell::RefCell<Option<JITModule>> =
    const { std::cell::RefCell::new(None) };
}

thread_local! {
    pub(crate) static TLS_JIT_TYPE_PROFILE: std::cell::RefCell<HashMap<usize, JitExecProfile>> =
    std::cell::RefCell::new(HashMap::new());
}

/// Create/use a thread-local JIT module and invoke `f` with it.
pub(crate) fn with_jit_module<F, R>(f: F) -> R
where
    F: FnOnce(&mut JITModule) -> R,
{
    TLS_JIT_MODULE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let build_isa = |plan: simd::SimdPlan| {
                let mut flag_builder = settings::builder();
                flag_builder.set("use_colocated_libcalls", "false").unwrap();
                if cfg!(target_arch = "aarch64") {
                    flag_builder.set("is_pic", "false").unwrap();
                } else {
                    flag_builder.set("is_pic", "true").unwrap();
                }
                simd::apply_cranelift_simd_flags(&mut flag_builder, plan);
                let _ = flag_builder.set("opt_level", "speed");

                let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
                    panic!("host machine is not supported: {}", msg);
                });
                isa_builder.finish(settings::Flags::new(flag_builder))
            };

            let requested_plan = simd::auto_vectorization_plan();
            let (isa, active_plan) = match build_isa(requested_plan) {
                Ok(isa) => (isa, requested_plan),
                        Err(err) if requested_plan.auto_vectorize => {
                            let fallback_plan = simd::SimdPlan::default();
                            crate::py::jit::jit_log(|| {
                                format!(
                                    "[Iris][jit][simd] host rejected SIMD plan {:?} (err={:?}); falling back to scalar",
                                        requested_plan, err
                                )
                            });
                            match build_isa(fallback_plan) {
                                Ok(isa) => (isa, fallback_plan),
                        Err(fallback_err) => {
                            panic!(
                                "failed to create ISA with SIMD plan ({:?}) and scalar fallback ({:?})",
                                   err, fallback_err
                            )
                        }
                            }
                        }
                        Err(err) => panic!("failed to create ISA: {:?}", err),
            };

            crate::py::jit::jit_log(|| {
                format!(
                    "[Iris][jit][simd] backend={:?} lane_bytes={} auto_vectorize={}",
                    active_plan.backend,
                    active_plan.lane_bytes,
                    active_plan.auto_vectorize
                )
            });

            let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
            builder.symbol("iris_jit_invoke_0", super::iris_jit_invoke_0 as *const u8);
            builder.symbol("iris_jit_invoke_1", super::iris_jit_invoke_1 as *const u8);
            builder.symbol("iris_jit_invoke_2", super::iris_jit_invoke_2 as *const u8);
            builder.symbol("iris_jit_invoke_3", super::iris_jit_invoke_3 as *const u8);
            builder.symbol("iris_jit_invoke_4", super::iris_jit_invoke_4 as *const u8);
            builder.symbol("iris_jit_invoke_5", super::iris_jit_invoke_5 as *const u8);
            builder.symbol("iris_jit_invoke_6", super::iris_jit_invoke_6 as *const u8);
            builder.symbol("iris_jit_invoke_7", super::iris_jit_invoke_7 as *const u8);
            builder.symbol("iris_jit_invoke_8", super::iris_jit_invoke_8 as *const u8);
            builder.symbol("iris_jit_invoke_9", super::iris_jit_invoke_9 as *const u8);
            builder.symbol("iris_jit_invoke_10", super::iris_jit_invoke_10 as *const u8);
            builder.symbol("iris_jit_invoke_11", super::iris_jit_invoke_11 as *const u8);
            builder.symbol("iris_jit_invoke_12", super::iris_jit_invoke_12 as *const u8);
            builder.symbol("iris_jit_invoke_13", super::iris_jit_invoke_13 as *const u8);
            builder.symbol("iris_jit_invoke_14", super::iris_jit_invoke_14 as *const u8);
            builder.symbol("iris_jit_invoke_15", super::iris_jit_invoke_15 as *const u8);
            builder.symbol("iris_jit_invoke_16", super::iris_jit_invoke_16 as *const u8);
            *opt = Some(JITModule::new(builder));
        }
        let module = opt.as_mut().unwrap();
        f(module)
    })
}
