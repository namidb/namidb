//! Cross-module test coordination for process-global configuration.
//!
//! The Search-LSM compaction policy is sampled from environment variables,
//! which are process-global: a test that sets `NAMIDB_SEARCH_LSM_*` leaks the
//! values to every test running concurrently, not only to itself. Any test
//! that either MUTATES those variables or ASSERTS behavior that depends on
//! their defaults (e.g. "the incremental policy retains deltas", "no legacy
//! rebuild marker is minted") must hold [`SEARCH_COMPACTION_ENV`] for its
//! whole body. Mutating tests additionally construct
//! [`SearchCompactionEnvRestore`] so the previous values return on drop even
//! when the test panics.

/// Serialises every test that touches or observes the Search-LSM policy env.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
pub(crate) static SEARCH_COMPACTION_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII: force a deterministic consolidation-friendly policy, restoring the
/// caller's environment on drop.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
pub(crate) struct SearchCompactionEnvRestore {
    max_segments: Option<std::ffi::OsString>,
    compact_segments: Option<std::ffi::OsString>,
    base_bytes: Option<std::ffi::OsString>,
    base_stale_percent: Option<std::ffi::OsString>,
    force_base: Option<std::ffi::OsString>,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl SearchCompactionEnvRestore {
    pub(crate) fn configure() -> Self {
        let restore = Self {
            max_segments: std::env::var_os("NAMIDB_SEARCH_LSM_MAX_SEGMENTS"),
            compact_segments: std::env::var_os("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS"),
            base_bytes: std::env::var_os("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES"),
            base_stale_percent: std::env::var_os("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT"),
            force_base: std::env::var_os("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION"),
        };
        std::env::set_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS", "8");
        std::env::set_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS", "3");
        std::env::set_var("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES", u64::MAX.to_string());
        std::env::set_var("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT", u64::MAX.to_string());
        std::env::set_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION", "true");
        restore
    }

    pub(crate) fn select_delta_runs(&self, trigger: usize) {
        std::env::set_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS", trigger.to_string());
        std::env::set_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION", "false");
    }
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl Drop for SearchCompactionEnvRestore {
    fn drop(&mut self) {
        match self.max_segments.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS"),
        }
        match self.compact_segments.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS"),
        }
        match self.base_bytes.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES"),
        }
        match self.base_stale_percent.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT"),
        }
        match self.force_base.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION"),
        }
    }
}
