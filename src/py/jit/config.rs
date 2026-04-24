// src/py/jit/config.rs
//! Configuration helpers for the Iris Python JIT.
//!
//! This module centralizes environment-variable backed configuration, logging hooks,
//! and other small utilities used across the JIT runtime.

use std::any::Any;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::info;

use crate::logging;
use crate::py::jit::codegen::JitReturnType;

static JIT_LOG_OVERRIDE: AtomicI8 = AtomicI8::new(-1); // -1 env, 0 off, 1 on
static JIT_LOG_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
type LogHook = Box<dyn Fn(String) + Send + Sync>;
static JIT_LOG_HOOK: OnceLock<std::sync::Mutex<Option<LogHook>>> = OnceLock::new();

static JIT_QUANTUM_OVERRIDE: AtomicI8 = AtomicI8::new(-1); // -1 env, 0 off, 1 on
static JIT_QUANTUM_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_LOG_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_SPECULATION_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COMPILE_BUDGET_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COMPILE_WINDOW_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COOLDOWN_BASE_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_COOLDOWN_MAX_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_STABILITY_MIN_SCORE_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_STABILITY_MIN_RUNS_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_VARIANT_FAILURE_LIMIT_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_REARM_INTERVAL_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_REARM_MIN_OBSERVED_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_REARM_MIN_SAMPLES_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();
static JIT_QUANTUM_REARM_MAX_VOLATILITY_ENV_VAR: OnceLock<RwLock<String>> = OnceLock::new();

macro_rules! env_var_fn {
    ($static:ident, $fn_name:ident, $default:expr) => {
        pub(crate) fn $fn_name() -> &'static RwLock<String> {
            $static.get_or_init(|| RwLock::new($default.to_string()))
        }
    };
}

env_var_fn!(JIT_LOG_ENV_VAR, jit_log_env_var, "IRIS_JIT_LOG");
env_var_fn!(JIT_QUANTUM_ENV_VAR, jit_quantum_env_var, "IRIS_JIT_QUANTUM");
env_var_fn!(
    JIT_QUANTUM_LOG_ENV_VAR,
    jit_quantum_log_env_var,
    "IRIS_JIT_QUANTUM_LOG_NS"
);
env_var_fn!(
    JIT_QUANTUM_SPECULATION_ENV_VAR,
    jit_quantum_speculation_env_var,
    "IRIS_JIT_QUANTUM_SPECULATION_NS"
);
env_var_fn!(
    JIT_QUANTUM_COMPILE_BUDGET_ENV_VAR,
    jit_quantum_compile_budget_env_var,
    "IRIS_JIT_QUANTUM_COMPILE_BUDGET_NS"
);
env_var_fn!(
    JIT_QUANTUM_COMPILE_WINDOW_ENV_VAR,
    jit_quantum_compile_window_env_var,
    "IRIS_JIT_QUANTUM_COMPILE_WINDOW_NS"
);
env_var_fn!(
    JIT_QUANTUM_COOLDOWN_BASE_ENV_VAR,
    jit_quantum_cooldown_base_env_var,
    "IRIS_JIT_QUANTUM_COOLDOWN_BASE_NS"
);
env_var_fn!(
    JIT_QUANTUM_COOLDOWN_MAX_ENV_VAR,
    jit_quantum_cooldown_max_env_var,
    "IRIS_JIT_QUANTUM_COOLDOWN_MAX_NS"
);
env_var_fn!(
    JIT_QUANTUM_STABILITY_MIN_SCORE_ENV_VAR,
    jit_quantum_stability_min_score_env_var,
    "IRIS_JIT_QUANTUM_STABILITY_MIN_SCORE"
);
env_var_fn!(
    JIT_QUANTUM_STABILITY_MIN_RUNS_ENV_VAR,
    jit_quantum_stability_min_runs_env_var,
    "IRIS_JIT_QUANTUM_STABILITY_MIN_RUNS"
);
env_var_fn!(
    JIT_QUANTUM_VARIANT_FAILURE_LIMIT_ENV_VAR,
    jit_quantum_variant_failure_limit_env_var,
    "IRIS_JIT_QUANTUM_VARIANT_FAILURE_LIMIT"
);
env_var_fn!(
    JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS_ENV_VAR,
    jit_quantum_variant_promotion_min_runs_env_var,
    "IRIS_JIT_QUANTUM_VARIANT_PROMOTION_MIN_RUNS"
);
env_var_fn!(
    JIT_QUANTUM_REARM_INTERVAL_ENV_VAR,
    jit_quantum_rearm_interval_env_var,
    "IRIS_JIT_QUANTUM_REARM_INTERVAL_NS"
);
env_var_fn!(
    JIT_QUANTUM_REARM_MIN_OBSERVED_ENV_VAR,
    jit_quantum_rearm_min_observed_env_var,
    "IRIS_JIT_QUANTUM_REARM_MIN_OBSERVED_NS"
);
env_var_fn!(
    JIT_QUANTUM_REARM_MIN_SAMPLES_ENV_VAR,
    jit_quantum_rearm_min_samples_env_var,
    "IRIS_JIT_QUANTUM_REARM_MIN_SAMPLES"
);
env_var_fn!(
    JIT_QUANTUM_REARM_MAX_VOLATILITY_ENV_VAR,
    jit_quantum_rearm_max_volatility_env_var,
    "IRIS_JIT_QUANTUM_REARM_MAX_VOLATILITY"
);

pub(crate) fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn parse_bool_env(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "debug"
    )
}

pub(crate) fn jit_logging_enabled() -> bool {
    match JIT_LOG_OVERRIDE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let env_name = jit_log_env_var().read().unwrap().clone();
            std::env::var(env_name)
                .ok()
                .map(|v| parse_bool_env(&v))
                .unwrap_or(false)
        }
    }
}

pub(crate) fn set_jit_logging_env_var(name: String) {
    *jit_log_env_var().write().unwrap() = name;
}

pub(crate) fn set_jit_logging_override(enabled: Option<bool>) {
    match enabled {
        Some(true) => JIT_LOG_OVERRIDE.store(1, Ordering::Relaxed),
        Some(false) => JIT_LOG_OVERRIDE.store(0, Ordering::Relaxed),
        None => JIT_LOG_OVERRIDE.store(-1, Ordering::Relaxed),
    }
}

#[cfg(test)]
pub(crate) fn jit_log_hook<F>(hook: F)
where
    F: Fn(String) + Send + Sync + 'static,
{
    let lock = JIT_LOG_HOOK.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = lock.lock().unwrap();
    *guard = Some(Box::new(hook));
}

#[cfg(test)]
pub(crate) fn jit_log_clear_hook() {
    if let Some(lock) = JIT_LOG_HOOK.get() {
        let mut guard = lock.lock().unwrap();
        *guard = None;
    }
}

pub(crate) fn jit_log<F>(msg: F)
where
    F: FnOnce() -> String,
{
    if !jit_logging_enabled() {
        return;
    }

    if let Some(lock) = JIT_LOG_HOOK.get() {
        let guard = lock.lock().unwrap();
        if let Some(ref hook) = *guard {
            hook(msg());
            return;
        }
    }

    logging::init_logger();
    info!(target: "iris::jit", "{}", msg());
}

pub(crate) fn quantum_speculation_enabled() -> bool {
    match JIT_QUANTUM_OVERRIDE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let env_name = jit_quantum_env_var().read().unwrap().clone();
            std::env::var(env_name)
                .ok()
                .map(|v| parse_bool_env(&v))
                .unwrap_or(false)
        }
    }
}

pub(crate) fn set_quantum_speculation_env_var(name: String) {
    *jit_quantum_env_var().write().unwrap() = name;
}

pub(crate) fn set_quantum_speculation_override(enabled: Option<bool>) {
    match enabled {
        Some(true) => JIT_QUANTUM_OVERRIDE.store(1, Ordering::Relaxed),
        Some(false) => JIT_QUANTUM_OVERRIDE.store(0, Ordering::Relaxed),
        None => JIT_QUANTUM_OVERRIDE.store(-1, Ordering::Relaxed),
    }
}

pub(crate) fn parse_return_type(rt: Option<&str>) -> JitReturnType {
    match rt {
        Some("int") => JitReturnType::Int,
        Some("bool") => JitReturnType::Bool,
        _ => JitReturnType::Float,
    }
}

fn parse_env_u64(env: &RwLock<String>, default: u64) -> u64 {
    let env_name = env.read().unwrap().clone();
    std::env::var(env_name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_env_f64(env: &RwLock<String>, default: f64, clamp: Option<(f64, f64)>) -> f64 {
    let raw = {
        let env_name = env.read().unwrap().clone();
        std::env::var(env_name)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
    };
    match raw {
        Some(v) => {
            if let Some((min, max)) = clamp {
                v.clamp(min, max)
            } else {
                v
            }
        }
        None => default,
    }
}

pub(crate) fn quantum_log_threshold_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000; // 1ms
    parse_env_u64(jit_quantum_log_env_var(), DEFAULT)
}

pub(crate) fn quantum_speculation_threshold_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000; // 1ms
    parse_env_u64(jit_quantum_speculation_env_var(), DEFAULT)
}

pub(crate) fn quantum_compile_budget_ns() -> u64 {
    const DEFAULT: u64 = 50_000_000; // 50ms compile budget per window
    parse_env_u64(jit_quantum_compile_budget_env_var(), DEFAULT)
}

pub(crate) fn quantum_compile_window_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000_000; // 1s
    parse_env_u64(jit_quantum_compile_window_env_var(), DEFAULT)
}

pub(crate) fn quantum_cooldown_base_ns() -> u64 {
    const DEFAULT: u64 = 5_000_000; // 5ms
    parse_env_u64(jit_quantum_cooldown_base_env_var(), DEFAULT)
}

pub(crate) fn quantum_cooldown_max_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000_000; // 1s
    parse_env_u64(jit_quantum_cooldown_max_env_var(), DEFAULT)
}

pub(crate) fn quantum_stability_min_score() -> f64 {
    const DEFAULT: f64 = 0.35;
    parse_env_f64(
        jit_quantum_stability_min_score_env_var(),
        DEFAULT,
        Some((0.0, 1.0)),
    )
}

pub(crate) fn quantum_stability_min_runs() -> u64 {
    const DEFAULT: u64 = 8;
    parse_env_u64(jit_quantum_stability_min_runs_env_var(), DEFAULT)
}

pub(crate) fn quantum_variant_failure_limit() -> u64 {
    const DEFAULT: u64 = 8;
    parse_env_u64(jit_quantum_variant_failure_limit_env_var(), DEFAULT)
}

pub(crate) fn quantum_variant_promotion_min_runs() -> u64 {
    const DEFAULT: u64 = 8;
    parse_env_u64(jit_quantum_variant_promotion_min_runs_env_var(), DEFAULT)
}

pub(crate) fn quantum_rearm_interval_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000_000; // 1s between rearm attempts per function
    parse_env_u64(jit_quantum_rearm_interval_env_var(), DEFAULT)
}

pub(crate) fn quantum_rearm_min_observed_ns() -> u64 {
    const DEFAULT: u64 = 1_000_000; // 1ms minimum observed latency before rearm
    parse_env_u64(jit_quantum_rearm_min_observed_env_var(), DEFAULT)
}

pub(crate) fn quantum_rearm_min_samples() -> u64 {
    const DEFAULT: u64 = 3;
    parse_env_u64(jit_quantum_rearm_min_samples_env_var(), DEFAULT).max(1)
}

pub(crate) fn quantum_rearm_max_volatility() -> f64 {
    const DEFAULT: f64 = 0.75;
    parse_env_f64(
        jit_quantum_rearm_max_volatility_env_var(),
        DEFAULT,
        Some((0.0, 4.0)),
    )
}

pub(crate) fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}
