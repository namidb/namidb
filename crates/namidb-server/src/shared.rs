//! Shared process-wide state for multi-tenant mode.

use std::sync::Arc;
use std::time::Duration;

use crate::auth::AuthConfig;
use crate::metrics::Metrics;
use crate::registry::NamespaceRegistry;

/// Shared process-wide state for multi-tenant mode.
///
/// In single-tenant mode, each request carries an `AppState` with a direct
/// reference to the namespace's `WriterSession`. In multi-tenant mode, the
/// router extracts the namespace from the path or header, looks it up in the
/// registry, and passes the per-namespace state to handlers.
///
/// This struct holds the process-wide configuration that is the same for all
/// namespaces: auth, metrics, and per-query limits.
///
/// This struct is Clone and cheap to clone (all fields are Arc wrappers).
#[derive(Clone)]
pub struct SharedAppState {
    /// Namespace registry for multi-tenant routing.
    pub registry: Arc<NamespaceRegistry>,
    /// Process-wide auth configuration (shared across all namespaces).
    pub auth: Arc<AuthConfig>,
    /// Process-wide metrics (aggregated across all namespaces).
    pub metrics: Arc<Metrics>,
    /// Per-read-query wall-clock budget (same for all namespaces).
    pub query_timeout: Duration,
    /// Per-write-query wall-clock budget (same for all namespaces).
    pub write_timeout: Duration,
    /// Per-read-query operator row cap (same for all namespaces).
    pub query_row_cap: usize,
    /// Soft write-stall threshold (L0 count) for backpressure.
    pub write_stall_l0: usize,
    /// Soft write-stall delay when L0 is above the threshold.
    pub write_stall_delay: Duration,
}

impl SharedAppState {
    /// Create a new shared state for multi-tenant mode.
    pub fn new(
        registry: Arc<NamespaceRegistry>,
        auth: Arc<AuthConfig>,
        metrics: Arc<Metrics>,
        query_timeout: Duration,
        write_timeout: Duration,
        query_row_cap: usize,
        write_stall_l0: usize,
        write_stall_delay: Duration,
    ) -> Self {
        Self {
            registry,
            auth,
            metrics,
            query_timeout,
            write_timeout,
            query_row_cap,
            write_stall_l0,
            write_stall_delay,
        }
    }

    /// Deadline for a read query starting now, or `None` when disabled.
    pub fn query_deadline(&self) -> Option<std::time::Instant> {
        (self.query_timeout > Duration::ZERO)
            .then(|| std::time::Instant::now() + self.query_timeout)
    }

    /// Deadline for a write query starting now, or `None` when disabled.
    pub fn write_deadline(&self) -> Option<std::time::Instant> {
        (self.write_timeout > Duration::ZERO)
            .then(|| std::time::Instant::now() + self.write_timeout)
    }

    /// Operator row cap for a read query, or `None` when disabled.
    pub fn query_row_cap(&self) -> Option<usize> {
        (self.query_row_cap > 0).then_some(self.query_row_cap)
    }

    /// If a write should be stalled given the worst bucket's current L0
    /// count, the delay to apply; otherwise `None`.
    pub fn write_stall_for(&self, max_l0: usize) -> Option<Duration> {
        (self.write_stall_l0 > 0
            && max_l0 >= self.write_stall_l0
            && self.write_stall_delay > Duration::ZERO)
            .then_some(self.write_stall_delay)
    }
}
