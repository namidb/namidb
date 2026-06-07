//! Process-level query metrics and the slow-query log.
//!
//! A hand-rolled registry of lock-free atomic counters and fixed-bucket
//! histograms, rendered on demand in the Prometheus text exposition format
//! at `GET /v0/metrics`. There is no metrics-crate dependency: the surface we
//! need (a handful of counters, a gauge, and four latency histograms) is small
//! enough that hand-rolling keeps the hot path allocation-free and the
//! dependency tree honest, matching the style of [`namidb_core::profile`].
//!
//! Both serving paths feed the same registry. HTTP queries are recorded by the
//! `cypher` handler and Bolt queries by the `ServerBackend`, each calling
//! [`Metrics::observe_query`] exactly once per query with the protocol, the
//! read/write kind, whether it succeeded, and the wall-clock it took. When a
//! query crosses the configured slow-query threshold the same call emits a
//! structured `warn!` line (the query text, never its parameters, which may
//! carry sensitive values).

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::warn;

/// Which serving path a query arrived on. Used as the `protocol` label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Http,
    Bolt,
}

impl Protocol {
    fn as_str(self) -> &'static str {
        match self {
            Protocol::Http => "http",
            Protocol::Bolt => "bolt",
        }
    }
}

/// Read vs write, decided by `plan.contains_write()`. `None` is used for a
/// query that failed before planning (a parse or plan error), where the kind
/// is genuinely unknown; such a query still counts toward the error total but
/// is not placed in a latency histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    Read,
    Write,
}

impl QueryKind {
    fn as_str(self) -> &'static str {
        match self {
            QueryKind::Read => "read",
            QueryKind::Write => "write",
        }
    }
}

/// Upper bounds (seconds) for the latency histogram buckets. A query is placed
/// in the first bucket whose bound is `>= elapsed`; anything slower lands in an
/// implicit `+Inf` overflow bucket. The range spans a sub-millisecond point
/// read up to the 30s default query timeout and beyond.
const BUCKET_BOUNDS_S: [f64; 10] = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0];

/// A cumulative-renderable latency histogram: a per-bucket count, the running
/// sum, and the observation count. Counts are stored per bucket (not yet
/// cumulative) and accumulated at render time, so a single observation is one
/// atomic increment rather than one per bucket.
#[derive(Debug)]
struct Histogram {
    /// `BUCKET_BOUNDS_S.len()` bounded buckets plus one `+Inf` overflow slot.
    counts: [AtomicU64; BUCKET_BOUNDS_S.len() + 1],
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
            sum_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    fn observe(&self, elapsed: Duration) {
        let secs = elapsed.as_secs_f64();
        let mut idx = BUCKET_BOUNDS_S.len(); // default to the +Inf overflow bucket
        for (i, &bound) in BUCKET_BOUNDS_S.iter().enumerate() {
            if secs <= bound {
                idx = i;
                break;
            }
        }
        self.counts[idx].fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Render the `_bucket`/`_sum`/`_count` lines for one label set. `labels`
    /// is the inner label list without braces, e.g. `protocol="http",kind="read"`.
    fn render_into(&self, out: &mut String, name: &str, labels: &str) {
        use std::fmt::Write as _;
        let mut cumulative: u64 = 0;
        for (i, &bound) in BUCKET_BOUNDS_S.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{bound}\"}} {cumulative}");
        }
        cumulative += self.counts[BUCKET_BOUNDS_S.len()].load(Ordering::Relaxed);
        let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {cumulative}");
        let sum_s = self.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let _ = writeln!(out, "{name}_sum{{{labels}}} {sum_s}");
        let _ = writeln!(
            out,
            "{name}_count{{{labels}}} {}",
            self.count.load(Ordering::Relaxed)
        );
    }
}

/// Per-protocol counters and latency histograms.
#[derive(Debug)]
struct ProtoMetrics {
    ok: AtomicU64,
    err: AtomicU64,
    read: Histogram,
    write: Histogram,
}

impl ProtoMetrics {
    fn new() -> Self {
        Self {
            ok: AtomicU64::new(0),
            err: AtomicU64::new(0),
            read: Histogram::new(),
            write: Histogram::new(),
        }
    }
}

/// The process-wide query metrics registry. One per server, shared across all
/// connections via the `Arc` held on `AppState`.
#[derive(Debug)]
pub struct Metrics {
    started: Instant,
    version: &'static str,
    /// Queries at or above this wall-clock are logged at `warn!`. `ZERO`
    /// disables the slow-query log (the counters and histograms stay on).
    slow_threshold: Duration,
    in_flight: AtomicI64,
    slow_queries: AtomicU64,
    http: ProtoMetrics,
    bolt: ProtoMetrics,
}

impl Metrics {
    /// Build a registry. `version` labels `namidb_build_info`; `slow_threshold`
    /// of `ZERO` turns the slow-query log off.
    pub fn new(version: &'static str, slow_threshold: Duration) -> Arc<Self> {
        Arc::new(Self {
            started: Instant::now(),
            version,
            slow_threshold,
            in_flight: AtomicI64::new(0),
            slow_queries: AtomicU64::new(0),
            http: ProtoMetrics::new(),
            bolt: ProtoMetrics::new(),
        })
    }

    fn proto(&self, protocol: Protocol) -> &ProtoMetrics {
        match protocol {
            Protocol::Http => &self.http,
            Protocol::Bolt => &self.bolt,
        }
    }

    /// Increment the in-flight gauge and return a guard that decrements it on
    /// drop, so the count is correct even if the query errors or panics.
    pub fn track_in_flight(self: &Arc<Self>) -> InFlightGuard {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        InFlightGuard(Arc::clone(self))
    }

    /// Record one completed query. Called exactly once per query from each
    /// serving path. `kind` is `None` for a query that failed before planning.
    /// Emits the slow-query `warn!` when enabled and `elapsed` crosses the
    /// threshold; `query` is the source text, logged truncated and without
    /// parameters.
    pub fn observe_query(
        &self,
        protocol: Protocol,
        kind: Option<QueryKind>,
        ok: bool,
        elapsed: Duration,
        query: &str,
    ) {
        let p = self.proto(protocol);
        if ok {
            p.ok.fetch_add(1, Ordering::Relaxed);
        } else {
            p.err.fetch_add(1, Ordering::Relaxed);
        }
        match kind {
            Some(QueryKind::Read) => p.read.observe(elapsed),
            Some(QueryKind::Write) => p.write.observe(elapsed),
            None => {}
        }

        if !self.slow_threshold.is_zero() && elapsed >= self.slow_threshold {
            self.slow_queries.fetch_add(1, Ordering::Relaxed);
            warn!(
                protocol = protocol.as_str(),
                kind = kind.map(QueryKind::as_str).unwrap_or("unknown"),
                status = if ok { "ok" } else { "error" },
                elapsed_ms = elapsed.as_millis() as u64,
                query = %sanitize_query(query),
                "slow query",
            );
        }
    }

    /// Render the whole registry in the Prometheus text exposition format
    /// (`text/plain; version=0.0.4`).
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(2048);

        let _ = writeln!(out, "# HELP namidb_build_info Build information.");
        let _ = writeln!(out, "# TYPE namidb_build_info gauge");
        let _ = writeln!(out, "namidb_build_info{{version=\"{}\"}} 1", self.version);

        let _ = writeln!(
            out,
            "# HELP namidb_uptime_seconds Seconds since the server started."
        );
        let _ = writeln!(out, "# TYPE namidb_uptime_seconds gauge");
        let _ = writeln!(
            out,
            "namidb_uptime_seconds {}",
            self.started.elapsed().as_secs_f64()
        );

        let _ = writeln!(
            out,
            "# HELP namidb_queries_in_flight Cypher queries currently executing."
        );
        let _ = writeln!(out, "# TYPE namidb_queries_in_flight gauge");
        let _ = writeln!(
            out,
            "namidb_queries_in_flight {}",
            self.in_flight.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP namidb_queries_total Cypher queries executed, by protocol and status."
        );
        let _ = writeln!(out, "# TYPE namidb_queries_total counter");
        for (proto, pm) in [("http", &self.http), ("bolt", &self.bolt)] {
            let _ = writeln!(
                out,
                "namidb_queries_total{{protocol=\"{proto}\",status=\"ok\"}} {}",
                pm.ok.load(Ordering::Relaxed)
            );
            let _ = writeln!(
                out,
                "namidb_queries_total{{protocol=\"{proto}\",status=\"error\"}} {}",
                pm.err.load(Ordering::Relaxed)
            );
        }

        let _ = writeln!(
            out,
            "# HELP namidb_slow_queries_total Queries that crossed the slow-query threshold."
        );
        let _ = writeln!(out, "# TYPE namidb_slow_queries_total counter");
        let _ = writeln!(
            out,
            "namidb_slow_queries_total {}",
            self.slow_queries.load(Ordering::Relaxed)
        );

        let _ = writeln!(
            out,
            "# HELP namidb_query_duration_seconds Query execution wall-clock, by protocol and kind."
        );
        let _ = writeln!(out, "# TYPE namidb_query_duration_seconds histogram");
        for (proto, pm) in [("http", &self.http), ("bolt", &self.bolt)] {
            pm.read.render_into(
                &mut out,
                "namidb_query_duration_seconds",
                &format!("protocol=\"{proto}\",kind=\"read\""),
            );
            pm.write.render_into(
                &mut out,
                "namidb_query_duration_seconds",
                &format!("protocol=\"{proto}\",kind=\"write\""),
            );
        }

        out
    }
}

/// RAII guard returned by [`Metrics::track_in_flight`]; decrements the
/// in-flight gauge on drop.
pub struct InFlightGuard(Arc<Metrics>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Collapse whitespace and cap a query string so the slow-query log stays one
/// readable line. Parameters are never included; only the statement text is.
fn sanitize_query(query: &str) -> String {
    const MAX: usize = 300;
    let collapsed = query.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() <= MAX {
        return collapsed;
    }
    let mut end = MAX;
    while !collapsed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &collapsed[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative_with_sum_and_count() {
        let h = Histogram::new();
        h.observe(Duration::from_micros(200)); // <= 0.0005
        h.observe(Duration::from_millis(2)); // <= 0.005
        h.observe(Duration::from_secs(20)); // overflow (+Inf only)

        let mut out = String::new();
        h.render_into(&mut out, "q", "protocol=\"http\",kind=\"read\"");

        // Smallest bucket holds the 200us observation.
        assert!(out.contains("q_bucket{protocol=\"http\",kind=\"read\",le=\"0.0005\"} 1"));
        // The 2ms observation makes le=0.005 cumulative count 2.
        assert!(out.contains("q_bucket{protocol=\"http\",kind=\"read\",le=\"0.005\"} 2"));
        // The 20s observation only shows up in +Inf: 3 total.
        assert!(out.contains("q_bucket{protocol=\"http\",kind=\"read\",le=\"10\"} 2"));
        assert!(out.contains("q_bucket{protocol=\"http\",kind=\"read\",le=\"+Inf\"} 3"));
        assert!(out.contains("q_count{protocol=\"http\",kind=\"read\"} 3"));
    }

    #[test]
    fn observe_query_splits_counters_by_status_and_kind() {
        let m = Metrics::new("0.0.0-test", Duration::ZERO);
        m.observe_query(
            Protocol::Http,
            Some(QueryKind::Read),
            true,
            Duration::from_millis(1),
            "MATCH (n) RETURN n",
        );
        m.observe_query(
            Protocol::Http,
            Some(QueryKind::Write),
            true,
            Duration::from_millis(2),
            "CREATE (n)",
        );
        m.observe_query(
            Protocol::Bolt,
            None,
            false,
            Duration::from_millis(1),
            "NOT CYPHER",
        );

        let text = m.render();
        assert!(text.contains("namidb_queries_total{protocol=\"http\",status=\"ok\"} 2"));
        assert!(text.contains("namidb_queries_total{protocol=\"bolt\",status=\"error\"} 1"));
        // The parse failure (kind None) never enters a histogram.
        assert!(
            text.contains("namidb_query_duration_seconds_count{protocol=\"http\",kind=\"read\"} 1")
        );
        assert!(text
            .contains("namidb_query_duration_seconds_count{protocol=\"http\",kind=\"write\"} 1"));
        assert!(
            text.contains("namidb_query_duration_seconds_count{protocol=\"bolt\",kind=\"read\"} 0")
        );
        // Slow log disabled (threshold ZERO): never counts a slow query.
        assert!(text.contains("namidb_slow_queries_total 0"));
        assert!(text.contains("namidb_build_info{version=\"0.0.0-test\"} 1"));
    }

    #[test]
    fn slow_threshold_counts_queries_at_or_above_it() {
        let m = Metrics::new("0.0.0-test", Duration::from_millis(10));
        m.observe_query(
            Protocol::Http,
            Some(QueryKind::Read),
            true,
            Duration::from_millis(1),
            "fast",
        );
        m.observe_query(
            Protocol::Http,
            Some(QueryKind::Read),
            true,
            Duration::from_millis(50),
            "slow",
        );
        assert!(m.render().contains("namidb_slow_queries_total 1"));
    }

    #[test]
    fn in_flight_guard_increments_then_decrements() {
        let m = Metrics::new("0.0.0-test", Duration::ZERO);
        {
            let _g1 = m.track_in_flight();
            let _g2 = m.track_in_flight();
            assert!(m.render().contains("namidb_queries_in_flight 2"));
        }
        assert!(m.render().contains("namidb_queries_in_flight 0"));
    }

    #[test]
    fn sanitize_collapses_whitespace_and_truncates() {
        assert_eq!(
            sanitize_query("MATCH (n)\n  RETURN   n"),
            "MATCH (n) RETURN n"
        );
        let long = "X".repeat(400);
        let s = sanitize_query(&long);
        assert!(s.ends_with("..."));
        assert!(s.len() <= 303);
    }
}
