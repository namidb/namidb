//! Process-level query metrics and the slow-query log.
//!
//! A hand-rolled registry of lock-free atomic counters and fixed-bucket
//! histograms, rendered on demand in the Prometheus text exposition format
//! at `GET /v0/metrics`. There is no metrics-crate dependency: the surface we
//! need (a bounded set of counters, gauges, and latency histograms) is small
//! enough that hand-rolling keeps the hot path allocation-free and the
//! dependency tree honest, matching the style of [`namidb_core::profile`].
//!
//! Both serving paths feed the same registry. HTTP queries are recorded by the
//! `cypher` handler and Bolt queries by the `ServerBackend`, each calling
//! [`Metrics::observe_query`] exactly once per query with the protocol, the
//! read/write kind, whether it succeeded, and the wall-clock it took. When a
//! query crosses the configured slow-query threshold the same call emits a
//! structured `warn!` line. Query parameters are never logged, since they
//! routinely carry sensitive values. Note the statement text itself is, so a
//! value inlined as a literal in the Cypher source (rather than passed as a
//! parameter) does land in the log, the same as every SQL slow-query log.

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

/// Why a task is waiting for the namespace's single-writer mutex. The finite
/// set keeps the Prometheus label cardinality bounded while separating
/// foreground queueing from background maintenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum WriterLockKind {
    Http = 0,
    Bolt = 1,
    BoltTransaction = 2,
    Flush = 3,
    CompactionBasis = 4,
    CompactionInstall = 5,
}

impl WriterLockKind {
    const ALL: [Self; 6] = [
        Self::Http,
        Self::Bolt,
        Self::BoltTransaction,
        Self::Flush,
        Self::CompactionBasis,
        Self::CompactionInstall,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Bolt => "bolt",
            Self::BoltTransaction => "bolt_transaction",
            Self::Flush => "flush",
            Self::CompactionBasis => "compaction_basis",
            Self::CompactionInstall => "compaction_install",
        }
    }
}

/// What scheduled a compaction attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CompactionTrigger {
    Periodic = 0,
    Reactive = 1,
}

impl CompactionTrigger {
    const ALL: [Self; 2] = [Self::Periodic, Self::Reactive];

    fn as_str(self) -> &'static str {
        match self {
            Self::Periodic => "periodic",
            Self::Reactive => "reactive",
        }
    }
}

/// Individually timed compaction phases. `Prepare` is the expensive off-lock
/// merge/upload work. `InstallWait` is queueing to reacquire the writer, and
/// `InstallHold` is the manifest-validation/CAS critical section plus any
/// failure recovery that must run while retaining the writer guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CompactionPhase {
    Prepare = 0,
    InstallWait = 1,
    InstallHold = 2,
}

impl CompactionPhase {
    const ALL: [Self; 3] = [Self::Prepare, Self::InstallWait, Self::InstallHold];

    fn as_str(self) -> &'static str {
        match self {
            Self::Prepare => "prepare",
            Self::InstallWait => "install_wait",
            Self::InstallHold => "install_hold",
        }
    }
}

/// Terminal classification of one compaction attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CompactionStatus {
    Applied = 0,
    Noop = 1,
    Stale = 2,
    Coalesced = 3,
    PrepareError = 4,
    InstallError = 5,
    Cancelled = 6,
}

impl CompactionStatus {
    const ALL: [Self; 7] = [
        Self::Applied,
        Self::Noop,
        Self::Stale,
        Self::Coalesced,
        Self::PrepareError,
        Self::InstallError,
        Self::Cancelled,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Applied => "applied",
            Self::Noop => "noop",
            Self::Stale => "stale",
            Self::Coalesced => "coalesced",
            Self::PrepareError => "prepare_error",
            Self::InstallError => "install_error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Upper bounds (seconds) for the latency histogram buckets. A query is placed
/// in the first bucket whose bound is `>= elapsed`; anything slower lands in an
/// implicit `+Inf` overflow bucket. The range spans a sub-millisecond point
/// read through the multi-minute tail seen when object-store compaction is
/// backlogged, so 30–222s incidents do not all collapse into `+Inf`.
const BUCKET_BOUNDS_S: [f64; 14] = [
    0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

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
    ///
    /// The fields are independent atomics read without a lock, so a scrape that
    /// races a live `observe` would otherwise risk `+Inf < _count`, which the
    /// Prometheus exposition format forbids. We anchor on a single `count`
    /// read: every bounded bucket's cumulative value is clamped to it and the
    /// `+Inf` bucket is set equal to it, so the output always satisfies
    /// `bucket[i] <= bucket[i+1] <= +Inf == _count`. The worst a race can do is
    /// momentarily under-report a bounded bucket, which the next scrape heals.
    fn render_into(&self, out: &mut String, name: &str, labels: &str) {
        use std::fmt::Write as _;
        let total = self.count.load(Ordering::Relaxed);
        let mut cumulative: u64 = 0;
        for (i, &bound) in BUCKET_BOUNDS_S.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            // `{bound:?}` keeps the canonical float form (`1.0`, `10.0`), which
            // is the `le` label every Prometheus client library emits; plain
            // Display would drop the decimal (`1`, `10`).
            let _ = writeln!(
                out,
                "{name}_bucket{{{labels},le=\"{bound:?}\"}} {}",
                cumulative.min(total)
            );
        }
        let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {total}");
        let sum_s = self.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let _ = writeln!(out, "{name}_sum{{{labels}}} {sum_s}");
        let _ = writeln!(out, "{name}_count{{{labels}}} {total}");
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

#[derive(Debug)]
struct WriterLockMetrics {
    wait: Histogram,
    timeouts: AtomicU64,
}

impl WriterLockMetrics {
    fn new() -> Self {
        Self {
            wait: Histogram::new(),
            timeouts: AtomicU64::new(0),
        }
    }
}

#[derive(Debug)]
struct CompactionMetrics {
    phases: [Histogram; CompactionPhase::ALL.len()],
    statuses: [AtomicU64; CompactionStatus::ALL.len()],
    /// Most recently observed maximum L0 bucket depth at attempt start/end.
    /// These are intentionally process/trigger scoped: no namespace label,
    /// and therefore no user-controlled cardinality.
    l0_before: AtomicU64,
    l0_after: AtomicU64,
    removed_ssts: AtomicU64,
    written_ssts: AtomicU64,
}

impl CompactionMetrics {
    fn new() -> Self {
        Self {
            phases: std::array::from_fn(|_| Histogram::new()),
            statuses: std::array::from_fn(|_| AtomicU64::new(0)),
            l0_before: AtomicU64::new(0),
            l0_after: AtomicU64::new(0),
            removed_ssts: AtomicU64::new(0),
            written_ssts: AtomicU64::new(0),
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
    writer_locks: [WriterLockMetrics; WriterLockKind::ALL.len()],
    compactions: [CompactionMetrics; CompactionTrigger::ALL.len()],
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
            writer_locks: std::array::from_fn(|_| WriterLockMetrics::new()),
            compactions: std::array::from_fn(|_| CompactionMetrics::new()),
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

    /// Record one attempt to acquire the namespace writer mutex. Call this
    /// immediately after acquisition or timeout, before doing work under the
    /// guard. The histogram contains both successful and timed-out waits;
    /// `namidb_writer_lock_timeouts_total` separates the latter.
    pub fn observe_writer_lock(&self, kind: WriterLockKind, wait: Duration, acquired: bool) {
        let metrics = &self.writer_locks[kind as usize];
        metrics.wait.observe(wait);
        if !acquired {
            metrics.timeouts.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record one compaction phase duration. Keeping prepare, install queueing,
    /// and install critical-section time separate makes it possible to tell
    /// object-store/merge debt from actual writer-lock contention.
    pub fn observe_compaction_phase(
        &self,
        trigger: CompactionTrigger,
        phase: CompactionPhase,
        elapsed: Duration,
    ) {
        self.compactions[trigger as usize].phases[phase as usize].observe(elapsed);
    }

    /// Record the terminal result and L0/SST deltas of one compaction attempt.
    /// `l0_before`/`l0_after` are the maximum L0 bucket depths sampled at the
    /// basis/install boundaries. In multi-tenant mode the gauge is the latest
    /// observation for each trigger, avoiding a namespace label with
    /// user-controlled cardinality.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_compaction_result(
        &self,
        trigger: CompactionTrigger,
        status: CompactionStatus,
        l0_before: usize,
        l0_after: usize,
        removed: usize,
        written: usize,
    ) {
        let metrics = &self.compactions[trigger as usize];
        metrics.statuses[status as usize].fetch_add(1, Ordering::Relaxed);
        // A coalesced trigger never captured a basis of its own. Do not let
        // its placeholder zeros overwrite the last real backlog boundary.
        if !matches!(
            status,
            CompactionStatus::Coalesced | CompactionStatus::Cancelled
        ) {
            metrics.l0_before.store(l0_before as u64, Ordering::Relaxed);
            metrics.l0_after.store(l0_after as u64, Ordering::Relaxed);
        }
        metrics
            .removed_ssts
            .fetch_add(removed as u64, Ordering::Relaxed);
        metrics
            .written_ssts
            .fetch_add(written as u64, Ordering::Relaxed);
    }

    /// Render the whole registry in the Prometheus text exposition format
    /// (`text/plain; version=0.0.4`).
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(16_384);

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
            "# HELP namidb_cache_max_bytes Configured aggregate byte ceiling for process-wide storage caches."
        );
        let _ = writeln!(out, "# TYPE namidb_cache_max_bytes gauge");
        let _ = writeln!(
            out,
            "namidb_cache_max_bytes {}",
            namidb_storage::cache_max_bytes()
        );
        let _ = writeln!(
            out,
            "# HELP namidb_cache_capacity_bytes Aggregate capacity assigned to enabled storage-cache tiers."
        );
        let _ = writeln!(out, "# TYPE namidb_cache_capacity_bytes gauge");
        let _ = writeln!(
            out,
            "namidb_cache_capacity_bytes {}",
            namidb_storage::shared_cache_capacity_bytes()
        );
        let _ = writeln!(
            out,
            "# HELP namidb_cache_resident_bytes Cache-accounted bytes currently resident across storage tiers."
        );
        let _ = writeln!(out, "# TYPE namidb_cache_resident_bytes gauge");
        let _ = writeln!(
            out,
            "namidb_cache_resident_bytes {}",
            namidb_storage::shared_cache_usage_bytes()
        );
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        {
            let _ = writeln!(
                out,
                "# HELP namidb_search_index_cache_capacity_bytes Assigned byte ceiling for the shared decoded vector/full-text index pool."
            );
            let _ = writeln!(out, "# TYPE namidb_search_index_cache_capacity_bytes gauge");
            let _ = writeln!(
                out,
                "namidb_search_index_cache_capacity_bytes {}",
                namidb_storage::shared_cache_capacities().search_index_capacity_bytes()
            );
            let _ = writeln!(
                out,
                "# HELP namidb_search_index_cache_admission_rejections_total Valid search indexes rejected because their estimated decoded footprint exceeds the configured pool."
            );
            let _ = writeln!(
                out,
                "# TYPE namidb_search_index_cache_admission_rejections_total counter"
            );
            let cache = namidb_storage::shared_sst_cache();
            #[cfg(feature = "vector-index")]
            {
                let count = cache.as_ref().map_or(
                    0,
                    namidb_storage::SstCache::vector_index_capacity_rejections,
                );
                let _ = writeln!(
                    out,
                    "namidb_search_index_cache_admission_rejections_total{{kind=\"vector\"}} {count}"
                );
            }
            #[cfg(feature = "text-index")]
            {
                let count = cache
                    .as_ref()
                    .map_or(0, namidb_storage::SstCache::text_index_capacity_rejections);
                let _ = writeln!(
                    out,
                    "namidb_search_index_cache_admission_rejections_total{{kind=\"text\"}} {count}"
                );
            }
        }
        #[cfg(feature = "vector-index")]
        {
            let _ = writeln!(
                out,
                "# HELP namidb_vector_filter_bitmap_searches_total Vector searches that applied at least one embedded .vg metadata posting."
            );
            let _ = writeln!(
                out,
                "# TYPE namidb_vector_filter_bitmap_searches_total counter"
            );
            let _ = writeln!(
                out,
                "namidb_vector_filter_bitmap_searches_total {}",
                namidb_storage::vector_filter_bitmap_searches()
            );
        }

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

        let _ = writeln!(
            out,
            "# HELP namidb_writer_lock_wait_seconds Time spent waiting for the namespace writer mutex."
        );
        let _ = writeln!(out, "# TYPE namidb_writer_lock_wait_seconds histogram");
        for kind in WriterLockKind::ALL {
            self.writer_locks[kind as usize].wait.render_into(
                &mut out,
                "namidb_writer_lock_wait_seconds",
                &format!("purpose=\"{}\"", kind.as_str()),
            );
        }

        let _ = writeln!(
            out,
            "# HELP namidb_writer_lock_timeouts_total Writer-mutex acquisitions that hit their configured bound."
        );
        let _ = writeln!(out, "# TYPE namidb_writer_lock_timeouts_total counter");
        for kind in WriterLockKind::ALL {
            let _ = writeln!(
                out,
                "namidb_writer_lock_timeouts_total{{purpose=\"{}\"}} {}",
                kind.as_str(),
                self.writer_locks[kind as usize]
                    .timeouts
                    .load(Ordering::Relaxed)
            );
        }

        let _ = writeln!(
            out,
            "# HELP namidb_compaction_phase_duration_seconds Compaction wall-clock split into off-lock prepare, install wait, and install hold."
        );
        let _ = writeln!(
            out,
            "# TYPE namidb_compaction_phase_duration_seconds histogram"
        );
        for trigger in CompactionTrigger::ALL {
            for phase in CompactionPhase::ALL {
                self.compactions[trigger as usize].phases[phase as usize].render_into(
                    &mut out,
                    "namidb_compaction_phase_duration_seconds",
                    &format!(
                        "trigger=\"{}\",phase=\"{}\"",
                        trigger.as_str(),
                        phase.as_str()
                    ),
                );
            }
        }

        let _ = writeln!(
            out,
            "# HELP namidb_compactions_total Compaction attempts by trigger and terminal status."
        );
        let _ = writeln!(out, "# TYPE namidb_compactions_total counter");
        for trigger in CompactionTrigger::ALL {
            for status in CompactionStatus::ALL {
                let _ = writeln!(
                    out,
                    "namidb_compactions_total{{trigger=\"{}\",status=\"{}\"}} {}",
                    trigger.as_str(),
                    status.as_str(),
                    self.compactions[trigger as usize].statuses[status as usize]
                        .load(Ordering::Relaxed)
                );
            }
        }

        let _ = writeln!(
            out,
            "# HELP namidb_compaction_l0_backlog_ssts Most recently observed maximum L0 bucket depth at a compaction boundary."
        );
        let _ = writeln!(out, "# TYPE namidb_compaction_l0_backlog_ssts gauge");
        for trigger in CompactionTrigger::ALL {
            let metrics = &self.compactions[trigger as usize];
            let _ = writeln!(
                out,
                "namidb_compaction_l0_backlog_ssts{{trigger=\"{}\",stage=\"before\"}} {}",
                trigger.as_str(),
                metrics.l0_before.load(Ordering::Relaxed)
            );
            let _ = writeln!(
                out,
                "namidb_compaction_l0_backlog_ssts{{trigger=\"{}\",stage=\"after\"}} {}",
                trigger.as_str(),
                metrics.l0_after.load(Ordering::Relaxed)
            );
        }

        let _ = writeln!(
            out,
            "# HELP namidb_compaction_ssts_total SST descriptors removed or written by compaction."
        );
        let _ = writeln!(out, "# TYPE namidb_compaction_ssts_total counter");
        for trigger in CompactionTrigger::ALL {
            let metrics = &self.compactions[trigger as usize];
            let _ = writeln!(
                out,
                "namidb_compaction_ssts_total{{trigger=\"{}\",action=\"removed\"}} {}",
                trigger.as_str(),
                metrics.removed_ssts.load(Ordering::Relaxed)
            );
            let _ = writeln!(
                out,
                "namidb_compaction_ssts_total{{trigger=\"{}\",action=\"written\"}} {}",
                trigger.as_str(),
                metrics.written_ssts.load(Ordering::Relaxed)
            );
        }

        out
    }

    /// Render query/storage metrics plus the process resident-memory governor.
    pub fn render_with_memory(&self, memory: &crate::memory::MemoryGovernor) -> String {
        use std::fmt::Write as _;
        let mut out = self.render();
        let _ = writeln!(
            out,
            "# HELP namidb_memory_max_bytes Configured process RSS/working-set admission ceiling; zero disables it."
        );
        let _ = writeln!(out, "# TYPE namidb_memory_max_bytes gauge");
        let _ = writeln!(out, "namidb_memory_max_bytes {}", memory.max_bytes());
        let _ = writeln!(
            out,
            "# HELP namidb_memory_resident_bytes Most recently sampled process RSS/working-set bytes."
        );
        let _ = writeln!(out, "# TYPE namidb_memory_resident_bytes gauge");
        let _ = writeln!(
            out,
            "namidb_memory_resident_bytes {}",
            memory.resident_bytes()
        );
        let _ = writeln!(
            out,
            "# HELP namidb_memory_reclaims_total Shared-cache pressure-relief passes triggered near the process memory ceiling."
        );
        let _ = writeln!(out, "# TYPE namidb_memory_reclaims_total counter");
        let _ = writeln!(
            out,
            "namidb_memory_reclaims_total {}",
            memory.reclaim_events()
        );
        let _ = writeln!(
            out,
            "# HELP namidb_memory_rejected_queries_total New Cypher queries rejected at the process memory ceiling."
        );
        let _ = writeln!(out, "# TYPE namidb_memory_rejected_queries_total counter");
        let _ = writeln!(
            out,
            "namidb_memory_rejected_queries_total {}",
            memory.rejected_queries()
        );

        // Reading metrics must not be the operation that initializes an NVMe
        // cache or locks its directory. The storage snapshot is deliberately
        // synchronous and returns `None` until a real range read has sampled
        // configuration.
        let range_cache = namidb_storage::range_cache::shared_range_cache_snapshot();
        let enabled = u8::from(range_cache.is_some());
        let range_cache = range_cache.unwrap_or_default();
        let _ = writeln!(
            out,
            "# HELP namidb_range_cache_enabled Whether the immutable object-range cache has been initialized and enabled."
        );
        let _ = writeln!(out, "# TYPE namidb_range_cache_enabled gauge");
        let _ = writeln!(out, "namidb_range_cache_enabled {enabled}");
        let _ = writeln!(
            out,
            "# HELP namidb_range_cache_bytes Range-cache byte gauges by tier and kind."
        );
        let _ = writeln!(out, "# TYPE namidb_range_cache_bytes gauge");
        let _ = writeln!(
            out,
            "namidb_range_cache_bytes{{tier=\"ram\",kind=\"budget\"}} {}",
            range_cache.ram_budget_bytes
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_bytes{{tier=\"ram\",kind=\"capacity\"}} {}",
            range_cache.memory_capacity_bytes
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_bytes{{tier=\"ram\",kind=\"usage\"}} {}",
            range_cache.memory_usage_bytes
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_bytes{{tier=\"ram\",kind=\"write_buffers\"}} {}",
            range_cache.write_buffer_reservation_bytes
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_bytes{{tier=\"ram\",kind=\"persistent_index_estimate\"}} {}",
            range_cache.disk_index_estimate_bytes
        );
        let accounted = range_cache
            .memory_capacity_bytes
            .saturating_add(range_cache.write_buffer_reservation_bytes)
            .saturating_add(range_cache.disk_index_estimate_bytes);
        let _ = writeln!(
            out,
            "namidb_range_cache_bytes{{tier=\"ram\",kind=\"accounted_total\"}} {accounted}"
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_bytes{{tier=\"nvme\",kind=\"capacity\"}} {}",
            range_cache.disk_capacity_bytes
        );
        let _ = writeln!(
            out,
            "# HELP namidb_range_cache_lookups_total Immutable range-cache lookup outcomes."
        );
        let _ = writeln!(out, "# TYPE namidb_range_cache_lookups_total counter");
        let _ = writeln!(
            out,
            "namidb_range_cache_lookups_total{{outcome=\"memory_hit\"}} {}",
            range_cache.stats.memory_hits
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_lookups_total{{outcome=\"disk_hit\"}} {}",
            range_cache.stats.disk_hits
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_lookups_total{{outcome=\"miss\"}} {}",
            range_cache.stats.misses
        );
        let _ = writeln!(
            out,
            "# HELP namidb_range_cache_remote_fetches_total Object-store fetches issued after range-cache misses."
        );
        let _ = writeln!(
            out,
            "# TYPE namidb_range_cache_remote_fetches_total counter"
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_remote_fetches_total {}",
            range_cache.stats.outer_fetches
        );
        let _ = writeln!(
            out,
            "# HELP namidb_range_cache_events_total Range-cache insert, rejection, and corruption events."
        );
        let _ = writeln!(out, "# TYPE namidb_range_cache_events_total counter");
        let _ = writeln!(
            out,
            "namidb_range_cache_events_total{{event=\"insert\"}} {}",
            range_cache.stats.inserts
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_events_total{{event=\"admission_rejection\"}} {}",
            range_cache.stats.admission_rejections
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_events_total{{event=\"corrupt_entry\"}} {}",
            range_cache.stats.corrupt_entries
        );
        let _ = writeln!(
            out,
            "# HELP namidb_range_cache_nvme_io_bytes_total Bytes read from or written to the persistent NVMe tier."
        );
        let _ = writeln!(out, "# TYPE namidb_range_cache_nvme_io_bytes_total counter");
        let _ = writeln!(
            out,
            "namidb_range_cache_nvme_io_bytes_total{{operation=\"read\"}} {}",
            range_cache.disk_read_bytes
        );
        let _ = writeln!(
            out,
            "namidb_range_cache_nvme_io_bytes_total{{operation=\"write\"}} {}",
            range_cache.disk_write_bytes
        );

        // As with the immutable range cache, a scrape must not initialise the
        // process-wide semaphore or freeze its environment-derived capacity.
        let search_workspace = namidb_storage::search_workspace::search_workspace_metrics();
        let search_workspace_enabled = u8::from(search_workspace.is_some());
        let (
            workspace_capacity,
            workspace_reserved,
            workspace_peak,
            workspace_successful,
            workspace_contended,
            workspace_rejected,
        ) = search_workspace
            .map(|metrics| {
                (
                    metrics.capacity_bytes,
                    metrics.reserved_bytes,
                    metrics.peak_reserved_bytes,
                    metrics.successful_reservations,
                    metrics.contended_reservations,
                    metrics.rejected_reservations,
                )
            })
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "# HELP namidb_search_workspace_enabled Whether the process-wide object-native search workspace has been initialized."
        );
        let _ = writeln!(out, "# TYPE namidb_search_workspace_enabled gauge");
        let _ = writeln!(
            out,
            "namidb_search_workspace_enabled {search_workspace_enabled}"
        );
        let _ = writeln!(
            out,
            "# HELP namidb_search_workspace_bytes Search-workspace byte gauges by kind."
        );
        let _ = writeln!(out, "# TYPE namidb_search_workspace_bytes gauge");
        let _ = writeln!(
            out,
            "namidb_search_workspace_bytes{{kind=\"capacity\"}} {workspace_capacity}"
        );
        let _ = writeln!(
            out,
            "namidb_search_workspace_bytes{{kind=\"reserved\"}} {workspace_reserved}"
        );
        let _ = writeln!(
            out,
            "namidb_search_workspace_bytes{{kind=\"peak_reserved\"}} {workspace_peak}"
        );
        let _ = writeln!(
            out,
            "# HELP namidb_search_workspace_reservations_total Search-workspace reservation outcomes."
        );
        let _ = writeln!(
            out,
            "# TYPE namidb_search_workspace_reservations_total counter"
        );
        let _ = writeln!(
            out,
            "namidb_search_workspace_reservations_total{{outcome=\"successful\"}} {workspace_successful}"
        );
        let _ = writeln!(
            out,
            "namidb_search_workspace_reservations_total{{outcome=\"contended\"}} {workspace_contended}"
        );
        let _ = writeln!(
            out,
            "namidb_search_workspace_reservations_total{{outcome=\"rejected\"}} {workspace_rejected}"
        );
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
        assert!(out.contains("q_bucket{protocol=\"http\",kind=\"read\",le=\"10.0\"} 2"));
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
        assert!(text.contains("namidb_cache_max_bytes "));
        assert!(text.contains("namidb_cache_capacity_bytes "));
        assert!(text.contains("namidb_cache_resident_bytes "));
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        {
            assert!(text.contains("namidb_search_index_cache_capacity_bytes "));
            assert!(text.contains("namidb_search_index_cache_admission_rejections_total"));
        }
        #[cfg(feature = "vector-index")]
        assert!(text.contains("namidb_vector_filter_bitmap_searches_total "));

        let memory = crate::memory::MemoryGovernor::new(123_456);
        let with_memory = m.render_with_memory(&memory);
        assert!(with_memory.contains("namidb_memory_max_bytes 123456"));
        assert!(with_memory.contains("namidb_memory_resident_bytes "));
        assert!(with_memory.contains("namidb_memory_reclaims_total 0"));
        assert!(with_memory.contains("namidb_memory_rejected_queries_total 0"));
        assert!(with_memory.contains("namidb_range_cache_enabled "));
        assert!(with_memory.contains("namidb_range_cache_bytes{tier=\"ram\",kind=\"usage\"} "));
        assert!(with_memory.contains("namidb_range_cache_lookups_total{outcome=\"memory_hit\"} "));
        assert!(with_memory.contains("namidb_range_cache_remote_fetches_total "));
        assert!(with_memory.contains("namidb_range_cache_nvme_io_bytes_total{operation=\"read\"} "));
        assert!(with_memory.contains("namidb_search_workspace_enabled "));
        assert!(with_memory.contains("namidb_search_workspace_bytes{kind=\"reserved\"} "));
        assert!(with_memory
            .contains("namidb_search_workspace_reservations_total{outcome=\"rejected\"} "));
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
    fn writer_lock_metrics_separate_waits_and_timeouts() {
        let m = Metrics::new("0.0.0-test", Duration::ZERO);
        m.observe_writer_lock(WriterLockKind::Http, Duration::from_millis(2), true);
        m.observe_writer_lock(WriterLockKind::Http, Duration::from_secs(45), false);
        m.observe_writer_lock(
            WriterLockKind::CompactionInstall,
            Duration::from_millis(3),
            true,
        );

        let text = m.render();
        assert!(text.contains("namidb_writer_lock_wait_seconds_count{purpose=\"http\"} 2"));
        assert!(
            text.contains("namidb_writer_lock_wait_seconds_bucket{purpose=\"http\",le=\"60.0\"} 2")
        );
        assert!(text
            .contains("namidb_writer_lock_wait_seconds_count{purpose=\"compaction_install\"} 1"));
        assert!(text.contains("namidb_writer_lock_timeouts_total{purpose=\"http\"} 1"));
        assert!(
            text.contains("namidb_writer_lock_timeouts_total{purpose=\"compaction_install\"} 0")
        );
    }

    #[test]
    fn compaction_metrics_render_phase_result_and_backlog_deltas() {
        let m = Metrics::new("0.0.0-test", Duration::ZERO);
        m.observe_compaction_phase(
            CompactionTrigger::Reactive,
            CompactionPhase::Prepare,
            Duration::from_secs(90),
        );
        m.observe_compaction_phase(
            CompactionTrigger::Reactive,
            CompactionPhase::InstallWait,
            Duration::from_millis(8),
        );
        m.observe_compaction_phase(
            CompactionTrigger::Reactive,
            CompactionPhase::InstallHold,
            Duration::from_millis(4),
        );
        m.observe_compaction_result(
            CompactionTrigger::Reactive,
            CompactionStatus::Applied,
            24,
            3,
            21,
            2,
        );
        m.observe_compaction_result(
            CompactionTrigger::Periodic,
            CompactionStatus::Stale,
            8,
            9,
            0,
            0,
        );
        m.observe_compaction_result(
            CompactionTrigger::Periodic,
            CompactionStatus::Coalesced,
            9,
            9,
            0,
            0,
        );

        let text = m.render();
        assert!(text.contains(
            "namidb_compaction_phase_duration_seconds_count{trigger=\"reactive\",phase=\"prepare\"} 1"
        ));
        assert!(text.contains(
            "namidb_compaction_phase_duration_seconds_bucket{trigger=\"reactive\",phase=\"prepare\",le=\"120.0\"} 1"
        ));
        assert!(text.contains(
            "namidb_compaction_phase_duration_seconds_count{trigger=\"reactive\",phase=\"install_wait\"} 1"
        ));
        assert!(text.contains(
            "namidb_compaction_phase_duration_seconds_count{trigger=\"reactive\",phase=\"install_hold\"} 1"
        ));
        assert!(
            text.contains("namidb_compactions_total{trigger=\"reactive\",status=\"applied\"} 1")
        );
        assert!(text.contains("namidb_compactions_total{trigger=\"periodic\",status=\"stale\"} 1"));
        assert!(
            text.contains("namidb_compactions_total{trigger=\"periodic\",status=\"coalesced\"} 1")
        );
        assert!(text.contains(
            "namidb_compaction_l0_backlog_ssts{trigger=\"reactive\",stage=\"before\"} 24"
        ));
        assert!(text
            .contains("namidb_compaction_l0_backlog_ssts{trigger=\"reactive\",stage=\"after\"} 3"));
        assert!(text
            .contains("namidb_compaction_ssts_total{trigger=\"reactive\",action=\"removed\"} 21"));
        assert!(text
            .contains("namidb_compaction_ssts_total{trigger=\"reactive\",action=\"written\"} 2"));
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
