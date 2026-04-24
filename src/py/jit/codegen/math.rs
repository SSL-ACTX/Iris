// src/py/jit/codegen/math.rs

use crate::py::jit::codegen::SimdMathMode;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub(crate) fn simd_math_mode_from_env() -> SimdMathMode {
    match std::env::var("IRIS_JIT_SIMD_MATH") {
        Ok(v) => {
            let key = v.trim().to_ascii_lowercase();
            if matches!(key.as_str(), "fast" | "approx" | "poly") {
                SimdMathMode::FastApprox
            } else {
                SimdMathMode::Accurate
            }
        }
        Err(_) => SimdMathMode::Accurate,
    }
}

#[inline(always)]
pub(crate) fn fast_sin_approx(x: f64) -> f64 {
    const APPROX_DOMAIN_LIMIT: f64 = 4_096.0;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

    if !x.is_finite() || x.abs() > APPROX_DOMAIN_LIMIT {
        return x.sin();
    }

    let mut y = x.rem_euclid(TAU);
    if y > PI {
        y -= TAU;
    }

    let mut sign = 1.0;
    if y > HALF_PI {
        y = PI - y;
    } else if y < -HALF_PI {
        y += PI;
        sign = -1.0;
    }

    let z = y * y;
    let p = y
        * (1.0 + z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0 + z * (1.0 / 362880.0)))));
    sign * p
}

#[inline(always)]
pub(crate) fn fast_cos_approx(x: f64) -> f64 {
    const APPROX_DOMAIN_LIMIT: f64 = 4_096.0;
    if !x.is_finite() || x.abs() > APPROX_DOMAIN_LIMIT {
        return x.cos();
    }
    fast_sin_approx(x + std::f64::consts::FRAC_PI_2)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn fast_sin_reduce_for_poly(x: f64) -> Option<(f64, f64)> {
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

    if !x.is_finite() {
        return None;
    }

    let mut y = x.rem_euclid(TAU);
    if y > PI {
        y -= TAU;
    }

    let mut sign = 1.0;
    if y > HALF_PI {
        y = PI - y;
    } else if y < -HALF_PI {
        y += PI;
        sign = -1.0;
    }

    Some((y, sign))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fast_sin_approx_pair_neon(x0: f64, x1: f64) -> (f64, f64) {
    let (y0, s0) = match fast_sin_reduce_for_poly(x0) {
        Some(reduced) => reduced,
        None => return (x0.sin(), x1.sin()),
    };
    let (y1, s1) = match fast_sin_reduce_for_poly(x1) {
        Some(reduced) => reduced,
        None => return (x0.sin(), x1.sin()),
    };

    let y_arr = [y0, y1];
    let sign_arr = [s0, s1];
    let y = vld1q_f64(y_arr.as_ptr());
    let sign = vld1q_f64(sign_arr.as_ptr());
    let z = vmulq_f64(y, y);

    let c1 = vdupq_n_f64(-1.0 / 6.0);
    let c2 = vdupq_n_f64(1.0 / 120.0);
    let c3 = vdupq_n_f64(-1.0 / 5040.0);
    let c4 = vdupq_n_f64(1.0 / 362880.0);

    let p3 = vaddq_f64(c3, vmulq_f64(z, c4));
    let p2 = vaddq_f64(c2, vmulq_f64(z, p3));
    let p1 = vaddq_f64(c1, vmulq_f64(z, p2));
    let poly = vaddq_f64(vdupq_n_f64(1.0), vmulq_f64(z, p1));
    let out = vmulq_f64(sign, vmulq_f64(y, poly));

    let mut out_arr = [0.0_f64; 2];
    vst1q_f64(out_arr.as_mut_ptr(), out);
    (out_arr[0], out_arr[1])
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fast_cos_approx_pair_neon(x0: f64, x1: f64) -> (f64, f64) {
    const APPROX_DOMAIN_LIMIT: f64 = 4_096.0;
    if !x0.is_finite()
        || !x1.is_finite()
        || x0.abs() > APPROX_DOMAIN_LIMIT
        || x1.abs() > APPROX_DOMAIN_LIMIT
    {
        return (x0.cos(), x1.cos());
    }
    fast_sin_approx_pair_neon(
        x0 + std::f64::consts::FRAC_PI_2,
        x1 + std::f64::consts::FRAC_PI_2,
    )
}
