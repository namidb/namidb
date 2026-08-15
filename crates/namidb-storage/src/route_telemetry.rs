//! Process-wide counters for the search serving route.
//!
//! The reachability trap, systemically: every result-parity test — and every
//! production deployment — looks identical whether queries serve from the
//! native index route or silently fall back to the O(corpus) flat scan.
//! These counters make the route observable: the storage layer records each
//! index-search outcome (served natively vs declined toward the fallback),
//! and the server exports them through `/metrics`, so a deployment where
//! native serving silently died shows a flatlined native counter next to a
//! climbing fallback counter.
//!
//! The counters are process-wide and monotonic; consumers compare deltas.

use std::sync::atomic::{AtomicU64, Ordering};

static TEXT_NATIVE: AtomicU64 = AtomicU64::new(0);
static TEXT_FALLBACK: AtomicU64 = AtomicU64::new(0);
static VECTOR_NATIVE: AtomicU64 = AtomicU64::new(0);
static VECTOR_FALLBACK: AtomicU64 = AtomicU64::new(0);

/// Record one text index search outcome: `native = true` when the index
/// served the answer, `false` when it declined (freshness gate, missing or
/// unreadable generation) and the caller will flat-scan.
pub fn record_text(native: bool) {
    if native {
        TEXT_NATIVE.fetch_add(1, Ordering::Relaxed);
    } else {
        TEXT_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

/// Record one vector index search outcome; same contract as [`record_text`].
pub fn record_vector(native: bool) {
    if native {
        VECTOR_NATIVE.fetch_add(1, Ordering::Relaxed);
    } else {
        VECTOR_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

/// A monotonic snapshot of every route counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteTelemetrySnapshot {
    pub text_native: u64,
    pub text_fallback: u64,
    pub vector_native: u64,
    pub vector_fallback: u64,
}

pub fn snapshot() -> RouteTelemetrySnapshot {
    RouteTelemetrySnapshot {
        text_native: TEXT_NATIVE.load(Ordering::Relaxed),
        text_fallback: TEXT_FALLBACK.load(Ordering::Relaxed),
        vector_native: VECTOR_NATIVE.load(Ordering::Relaxed),
        vector_fallback: VECTOR_FALLBACK.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    /// Deltas, not absolutes: the counters are process-wide and other tests
    /// in the same process legitimately move them.
    #[test]
    fn counters_move_monotonically_by_kind_and_route() {
        let before = super::snapshot();
        super::record_text(true);
        super::record_text(false);
        super::record_vector(true);
        super::record_vector(false);
        let after = super::snapshot();
        assert!(after.text_native > before.text_native);
        assert!(after.text_fallback > before.text_fallback);
        assert!(after.vector_native > before.vector_native);
        assert!(after.vector_fallback > before.vector_fallback);
    }
}
