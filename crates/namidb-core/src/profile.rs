//! Cross-crate per-stage profiler with RAII guards.
//!
//! When `NAMIDB_PROFILE_DUMP=1`, instrumented sites accumulate
//! `(count, total_ns)` per named stage. When the env var is unset the
//! macro expands to a no-op so the production hot path is zero-overhead.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use namidb_core::profile_scope;
//!
//! async fn lookup_node(&self, ...) -> Result<...> {
//! profile_scope!("Snapshot::lookup_node");
//! // ... rest of the function; the guard records on drop.
//! }
//! ```
//!
//! Aggregation happens in a process-global `Profiler` behind a `Mutex`.
//! Contention is fine: the profiler is only enabled during dedicated
//! profile runs, not in production.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ENABLED_CACHE: AtomicU8 = AtomicU8::new(0); // 0 = unread, 1 = false, 2 = true
static RECORDING: AtomicBool = AtomicBool::new(true);

/// Read `NAMIDB_PROFILE_DUMP` once and cache the answer. Subsequent
/// calls are an atomic load — cheap enough to call from any hot path.
pub fn enabled() -> bool {
    match ENABLED_CACHE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var("NAMIDB_PROFILE_DUMP")
                .map(|v| v == "1")
                .unwrap_or(false);
            ENABLED_CACHE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Temporarily disable recording (e.g. during warmup) while keeping
/// `enabled()` true so guards are still allocated. Useful when the
/// caller wants to reset midway and start clean.
pub fn pause() {
    RECORDING.store(false, Ordering::Relaxed);
}

/// Re-enable recording after a pause.
pub fn resume() {
    RECORDING.store(true, Ordering::Relaxed);
}

fn recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

#[derive(Debug, Default)]
struct StageEntry {
    count: u64,
    total_ns: u128,
}

#[derive(Debug, Default)]
struct Profiler {
    stages: HashMap<&'static str, StageEntry>,
}

fn profiler() -> &'static Mutex<Profiler> {
    static P: OnceLock<Mutex<Profiler>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(Profiler::default()))
}

/// Record one observation of `stage` lasting `ns` nanoseconds.
/// Caller normally goes through [`ProfileGuard`] / [`profile_scope!`].
pub fn record(stage: &'static str, ns: u128) {
    if !recording() {
        return;
    }
    let mut p = profiler().lock().unwrap();
    let entry = p.stages.entry(stage).or_default();
    entry.count = entry.count.saturating_add(1);
    entry.total_ns = entry.total_ns.saturating_add(ns);
}

/// Snapshot of the profile for printing. Sorted by `total_ns` descending.
pub fn dump() -> Vec<(&'static str, u64, u128)> {
    let p = profiler().lock().unwrap();
    let mut out: Vec<_> = p
        .stages
        .iter()
        .map(|(k, v)| (*k, v.count, v.total_ns))
        .collect();
    out.sort_by_key(|b| std::cmp::Reverse(b.2));
    out
}

/// Pretty-printed table. Columns: stage | count | total ms | avg µs.
pub fn dump_table() -> String {
    let rows = dump();
    if rows.is_empty() {
        return "(profile empty — set NAMIDB_PROFILE_DUMP=1 and run again)\n".to_string();
    }
    let mut s = String::new();
    s.push_str("stage count total_ms avg_us\n");
    s.push_str("----- ----- -------- ------\n");
    for (stage, count, total_ns) in &rows {
        let total_ms = (*total_ns as f64) / 1_000_000.0;
        let avg_us = if *count > 0 {
            (*total_ns as f64) / (*count as f64) / 1_000.0
        } else {
            0.0
        };
        s.push_str(&format!(
            "{:<45} {:>10} {:>12.3} {:>10.3}\n",
            stage, count, total_ms, avg_us
        ));
    }
    s
}

/// Clear all accumulated stages. Used between bench warmup and the
/// measured run.
pub fn reset() {
    let mut p = profiler().lock().unwrap();
    p.stages.clear();
}

/// RAII guard. Created by [`profile_scope!`]; records on `Drop`.
#[derive(Debug)]
pub struct ProfileGuard {
    stage: &'static str,
    start: Instant,
}

impl ProfileGuard {
    pub fn new(stage: &'static str) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        record(self.stage, self.start.elapsed().as_nanos());
    }
}

/// Conditionally instrument the enclosing scope. When `NAMIDB_PROFILE_DUMP`
/// is unset, this expands to a no-op `let _ = ();` so non-profile builds
/// pay nothing. When set, it allocates a [`ProfileGuard`] that records the
/// elapsed time on drop.
///
/// Stage name must be a `&'static str` literal so the profiler's HashMap
/// can use it as the key without allocation.
#[macro_export]
macro_rules! profile_scope {
    ($stage:literal) => {
        let _profile_guard = if $crate::profile::enabled() {
            Some($crate::profile::ProfileGuard::new($stage))
        } else {
            None
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_is_empty_initially() {
        // Note: tests run in the same process; if another test already
        // recorded something we'll see it. Reset first.
        reset();
        let _ = enabled(); // populate cache
        let rows = dump();
        assert!(rows.is_empty(), "expected empty after reset, got {rows:?}");
    }

    #[test]
    fn record_accumulates_per_stage() {
        reset();
        // Force-enable: bypass env var for this unit test.
        ENABLED_CACHE.store(2, Ordering::Relaxed);
        record("alpha", 1000);
        record("alpha", 2000);
        record("beta", 5000);
        let rows = dump();
        // Sorted by total_ns desc: beta=5000 first, alpha=3000 second.
        assert_eq!(rows[0].0, "beta");
        assert_eq!(rows[0].1, 1);
        assert_eq!(rows[0].2, 5000);
        assert_eq!(rows[1].0, "alpha");
        assert_eq!(rows[1].1, 2);
        assert_eq!(rows[1].2, 3000);
        reset();
        // Restore cache so other tests see real env-var state.
        ENABLED_CACHE.store(0, Ordering::Relaxed);
    }
}
