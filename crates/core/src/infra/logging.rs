use std::sync::Once;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::EnvFilter;

static LOGGER_INIT: Once = Once::new();

/// Initialize the Iris tracing subscriber once for the entire crate.
pub fn init_logger() {
    LOGGER_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_timer(UtcTime::rfc_3339())
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .compact()
            .finish();

        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_logger_is_idempotent() {
        init_logger();
        init_logger();
    }
}
