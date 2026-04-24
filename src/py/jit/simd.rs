// src/py/jit/simd.rs
//! SIMD capability detection and auto-vectorization planning.

use cranelift::prelude::settings;
use cranelift::prelude::Configurable;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum SimdArch {
    Aarch64,
    Arm,
    X86_64,
    X86,
    Wasm32,
    #[default]
    Other,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SimdCapabilities {
    pub arch: SimdArch,
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub sse2: bool,
    pub wasm_simd128: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SimdBackend {
    Scalar,
    ArmNeon,
    ArmSve,
    X86Sse2,
    X86Avx,
    X86Avx2,
    WasmSimd128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SimdPlan {
    pub backend: SimdBackend,
    pub lane_bytes: usize,
    pub auto_vectorize: bool,
}

impl Default for SimdPlan {
    fn default() -> Self {
        Self {
            backend: SimdBackend::Scalar,
            lane_bytes: 8,
            auto_vectorize: false,
        }
    }
}

pub(crate) fn detect_host_capabilities() -> SimdCapabilities {
    let mut caps = SimdCapabilities {
        arch: host_arch(),
        ..Default::default()
    };

    #[cfg(target_arch = "aarch64")]
    {
        caps.neon = std::arch::is_aarch64_feature_detected!("neon");
        caps.sve = std::arch::is_aarch64_feature_detected!("sve");
        caps.sve2 = std::arch::is_aarch64_feature_detected!("sve2");
    }

    #[cfg(target_arch = "arm")]
    {
        caps.neon = std::arch::is_arm_feature_detected!("neon");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        caps.avx = std::arch::is_x86_feature_detected!("avx");
        caps.avx2 = std::arch::is_x86_feature_detected!("avx2");
        caps.sse2 = std::arch::is_x86_feature_detected!("sse2");
    }

    #[cfg(target_arch = "wasm32")]
    {
        caps.wasm_simd128 = cfg!(target_feature = "simd128");
    }

    caps
}

pub(crate) fn choose_plan(caps: SimdCapabilities, simd_enabled: bool) -> SimdPlan {
    if !simd_enabled {
        return SimdPlan::default();
    }

    match caps.arch {
        SimdArch::Aarch64 | SimdArch::Arm => {
            if caps.sve2 || caps.sve {
                SimdPlan {
                    backend: SimdBackend::ArmSve,
                    lane_bytes: 32,
                    auto_vectorize: true,
                }
            } else if caps.neon {
                SimdPlan {
                    backend: SimdBackend::ArmNeon,
                    lane_bytes: 16,
                    auto_vectorize: true,
                }
            } else {
                SimdPlan::default()
            }
        }
        SimdArch::X86 | SimdArch::X86_64 => {
            if caps.avx2 {
                SimdPlan {
                    backend: SimdBackend::X86Avx2,
                    lane_bytes: 32,
                    auto_vectorize: true,
                }
            } else if caps.avx {
                SimdPlan {
                    backend: SimdBackend::X86Avx,
                    lane_bytes: 32,
                    auto_vectorize: true,
                }
            } else if caps.sse2 {
                SimdPlan {
                    backend: SimdBackend::X86Sse2,
                    lane_bytes: 16,
                    auto_vectorize: true,
                }
            } else {
                SimdPlan::default()
            }
        }
        SimdArch::Wasm32 => {
            if caps.wasm_simd128 {
                SimdPlan {
                    backend: SimdBackend::WasmSimd128,
                    lane_bytes: 16,
                    auto_vectorize: true,
                }
            } else {
                SimdPlan::default()
            }
        }
        SimdArch::Other => SimdPlan::default(),
    }
}

pub(crate) fn auto_vectorization_plan() -> SimdPlan {
    let caps = detect_host_capabilities();
    choose_plan(caps, simd_enabled_from_env())
}

pub(crate) fn apply_cranelift_simd_flags(flag_builder: &mut settings::Builder, plan: SimdPlan) {
    let _ = flag_builder.set(
        "enable_simd",
        if plan.auto_vectorize { "true" } else { "false" },
    );
}

pub(crate) fn simd_enabled_from_env() -> bool {
    match std::env::var("IRIS_JIT_SIMD") {
        Ok(value) => parse_bool_env(&value),
        Err(_) => true,
    }
}

pub(crate) fn host_arch() -> SimdArch {
    if cfg!(target_arch = "aarch64") {
        SimdArch::Aarch64
    } else if cfg!(target_arch = "arm") {
        SimdArch::Arm
    } else if cfg!(target_arch = "x86_64") {
        SimdArch::X86_64
    } else if cfg!(target_arch = "x86") {
        SimdArch::X86
    } else if cfg!(target_arch = "wasm32") {
        SimdArch::Wasm32
    } else {
        SimdArch::Other
    }
}

fn parse_bool_env(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enable" | "enabled"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_prefers_sve_over_neon() {
        let caps = SimdCapabilities {
            arch: SimdArch::Aarch64,
            neon: true,
            sve: true,
            ..Default::default()
        };

        let plan = choose_plan(caps, true);
        assert_eq!(plan.backend, SimdBackend::ArmSve);
        assert!(plan.auto_vectorize);
        assert_eq!(plan.lane_bytes, 32);
    }

    #[test]
    fn arm_falls_back_to_neon_when_sve_unavailable() {
        let caps = SimdCapabilities {
            arch: SimdArch::Arm,
            neon: true,
            ..Default::default()
        };

        let plan = choose_plan(caps, true);
        assert_eq!(plan.backend, SimdBackend::ArmNeon);
        assert!(plan.auto_vectorize);
        assert_eq!(plan.lane_bytes, 16);
    }

    #[test]
    fn x86_prefers_avx2_then_sse2() {
        let avx2_caps = SimdCapabilities {
            arch: SimdArch::X86_64,
            avx2: true,
            ..Default::default()
        };
        let avx2_plan = choose_plan(avx2_caps, true);
        assert_eq!(avx2_plan.backend, SimdBackend::X86Avx2);
        assert_eq!(avx2_plan.lane_bytes, 32);

        let sse2_caps = SimdCapabilities {
            arch: SimdArch::X86_64,
            sse2: true,
            ..Default::default()
        };
        let sse2_plan = choose_plan(sse2_caps, true);
        assert_eq!(sse2_plan.backend, SimdBackend::X86Sse2);
        assert_eq!(sse2_plan.lane_bytes, 16);
    }

    #[test]
    fn disable_flag_forces_scalar_mode() {
        let caps = SimdCapabilities {
            arch: SimdArch::Aarch64,
            neon: true,
            sve: true,
            ..Default::default()
        };

        let plan = choose_plan(caps, false);
        assert_eq!(plan.backend, SimdBackend::Scalar);
        assert!(!plan.auto_vectorize);
        assert_eq!(plan.lane_bytes, 8);
    }

    #[test]
    fn wasm_uses_simd128_when_available() {
        let caps = SimdCapabilities {
            arch: SimdArch::Wasm32,
            wasm_simd128: true,
            ..Default::default()
        };

        let plan = choose_plan(caps, true);
        assert_eq!(plan.backend, SimdBackend::WasmSimd128);
        assert!(plan.auto_vectorize);
        assert_eq!(plan.lane_bytes, 16);
    }

    #[test]
    fn cranelift_flag_application_is_safe_for_scalar_and_simd() {
        let mut flags = settings::builder();
        apply_cranelift_simd_flags(&mut flags, SimdPlan::default());
        apply_cranelift_simd_flags(
            &mut flags,
            SimdPlan {
                backend: SimdBackend::ArmNeon,
                lane_bytes: 16,
                auto_vectorize: true,
            },
        );
    }
}
