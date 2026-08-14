//! Acceptance harness for the range-readable `NAMIVG05` and `NAMIFT03`
//! artifacts.
//!
//! The benchmark intentionally talks to the format readers through an
//! instrumented byte-range source. It therefore measures the property that a
//! memory-only object-store mock cannot show: a cold query must fetch a small
//! fraction of the immutable artifact, while a warm query should be served by
//! a bounded cache. Corpus generation and both index builds are streaming and
//! deterministic, so the 1M/10M documented runs do not first materialize the
//! corpus in RAM.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::future::try_join_all;
use namidb_core::Value;
use namidb_storage::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
use namidb_storage::sst::text::{
    ExternalTextIndexBuildOptions, TextIndexExternalBuilder, TextIndexRangeSource,
    TextIndexV3Reader,
};
use namidb_storage::sst::vector::v5::external::{
    VectorV5ExternalBuildConfig, VectorV5ExternalBuildMetrics, VectorV5ExternalCollector,
};
use namidb_storage::sst::vector::v5::{
    VectorV5BuildOptions, VectorV5RangeSource, VectorV5Reader, VectorV5SearchOptions,
};
use namidb_storage::text::{
    avg_len, bm25_idf, bm25_term_score, contains_phrase, parse_query, tokenize, TextQuery,
    PREFIX_EXPANSION_LIMIT,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use crate::vector_recall::{perturbed_unit_vector, random_unit_vector};

#[path = "object_native_architecture.rs"]
mod architecture;

use architecture::{
    run_architecture, ArchitectureBuilderWorkspaceReport, GraphPagedReport, SearchLsmReport,
};

const REPORT_FORMAT: &str = "namidb-object-native-gate-v2";

#[derive(Debug, Clone)]
pub struct ObjectNativeConfig {
    pub vectors: usize,
    pub documents: usize,
    pub dim: usize,
    pub queries: usize,
    pub k: usize,
    pub clusters: usize,
    pub spread: f32,
    pub filter_buckets: usize,
    pub page_rows: usize,
    pub branch_factor: usize,
    pub nprobe: usize,
    pub max_nprobe: usize,
    pub rerank_factor: usize,
    pub cache_bytes: usize,
    pub build_memory_bytes: usize,
    pub range_latency_ms: u64,
    pub delta_segments: usize,
    pub graph_keys: usize,
    pub graph_high_degree: usize,
    pub seed: u64,
    pub gates: GateThresholds,
}

#[derive(Debug, Clone, Serialize)]
pub struct GateThresholds {
    pub min_recall_at_k: Option<f64>,
    pub max_cold_bytes_ratio: Option<f64>,
    pub max_reader_metadata_ratio: Option<f64>,
    pub max_query_p95_ms: Option<f64>,
    pub max_rss_bytes: Option<u64>,
    pub max_fanout: usize,
    pub max_in_flight: u64,
}

impl Default for GateThresholds {
    fn default() -> Self {
        Self {
            min_recall_at_k: None,
            max_cold_bytes_ratio: None,
            max_reader_metadata_ratio: None,
            max_query_p95_ms: None,
            max_rss_bytes: None,
            max_fanout: 32,
            max_in_flight: 16,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ObjectNativeReport {
    pub format: &'static str,
    pub config: ConfigReport,
    pub vectors: Vec<VectorMetricReport>,
    pub text: TextReport,
    pub search_lsm: SearchLsmReport,
    pub graph: GraphPagedReport,
    pub builder_workspace: ArchitectureBuilderWorkspaceReport,
    pub rss: RssReport,
    pub gates: GateReport,
}

#[derive(Debug, Serialize)]
pub struct ConfigReport {
    pub vectors: usize,
    pub documents: usize,
    pub dim: usize,
    pub queries: usize,
    pub k: usize,
    pub clusters: usize,
    pub spread: f32,
    pub filter_buckets: usize,
    pub page_rows: usize,
    pub branch_factor: usize,
    pub nprobe: usize,
    pub max_nprobe: usize,
    pub rerank_factor: usize,
    pub cache_bytes: usize,
    pub build_memory_bytes: usize,
    pub range_latency_ms: u64,
    pub delta_segments: usize,
    pub graph_keys: usize,
    pub graph_high_degree: usize,
    pub seed: u64,
}

#[derive(Debug, Serialize)]
pub struct VectorMetricReport {
    pub metric: &'static str,
    pub build_secs: f64,
    pub artifact_bytes: u64,
    pub artifact_bytes_per_vector: f64,
    pub page_count: usize,
    pub reader_metadata_bytes: usize,
    pub reader_metadata_artifact_ratio: f64,
    pub reader_open_io: RangeIoStats,
    pub builder: VectorBuilderReport,
    pub unfiltered: VectorQueryTrack,
    pub selective_filter: VectorQueryTrack,
}

#[derive(Debug, Serialize)]
pub struct VectorBuilderReport {
    pub indexed_rows: u64,
    pub page_count: u64,
    pub partition_count: u64,
    pub max_partition_depth: usize,
    pub peak_sample_rows: usize,
    pub peak_logical_memory_bytes: usize,
    pub peak_working_memory_bytes: usize,
    pub resident_metadata_bytes: usize,
    pub scratch_bytes_written: u64,
    pub artifact_bytes_written: u64,
    pub effective_target_rows_per_page: usize,
}

impl From<VectorV5ExternalBuildMetrics> for VectorBuilderReport {
    fn from(value: VectorV5ExternalBuildMetrics) -> Self {
        Self {
            indexed_rows: value.indexed_rows,
            page_count: value.page_count,
            partition_count: value.partition_count,
            max_partition_depth: value.max_partition_depth,
            peak_sample_rows: value.peak_sample_rows,
            peak_logical_memory_bytes: value.peak_logical_memory_bytes,
            peak_working_memory_bytes: value.peak_working_memory_bytes,
            resident_metadata_bytes: value.resident_metadata_bytes,
            scratch_bytes_written: value.scratch_bytes_written,
            artifact_bytes_written: value.artifact_bytes_written,
            effective_target_rows_per_page: value.effective_target_rows_per_page,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct VectorQueryTrack {
    pub query_count: usize,
    pub recall_at_k: f64,
    pub native_filter_applied: bool,
    pub cold_p50_us: u64,
    pub cold_p95_us: u64,
    pub warm_p50_us: u64,
    pub warm_p95_us: u64,
    pub cold_io: RangeIoAggregate,
    pub warm_io: RangeIoAggregate,
}

#[derive(Debug, Serialize)]
pub struct TextReport {
    pub build_secs: f64,
    pub artifact_bytes: u64,
    pub artifact_bytes_per_document: f64,
    pub document_count: u64,
    pub term_count: u64,
    pub reader_metadata_bytes: usize,
    pub reader_metadata_artifact_ratio: f64,
    pub reader_open_io: RangeIoStats,
    pub builder: TextBuilderReport,
    pub cold_p50_us: u64,
    pub cold_p95_us: u64,
    pub warm_p50_us: u64,
    pub warm_p95_us: u64,
    pub cold_io: RangeIoAggregate,
    pub warm_io: RangeIoAggregate,
    pub parity: Vec<TextParityReport>,
}

#[derive(Debug, Serialize)]
pub struct TextBuilderReport {
    pub memory_budget_bytes: usize,
    pub max_buffer_bytes: usize,
    pub initial_run_count: u64,
    pub run_merge_count: u64,
    pub spool_bytes_written: u64,
}

#[derive(Debug, Serialize)]
pub struct TextParityReport {
    pub name: &'static str,
    pub syntax: &'static str,
    pub result_count: usize,
    pub node_ids_equal: bool,
    pub scores_equal: bool,
    pub max_score_abs_error: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RangeIoStats {
    /// Calls made by the reader, including cache hits.
    pub logical_requests: u64,
    pub logical_bytes: u64,
    /// Underlying file/object-store reads after the cache.
    pub fetch_requests: u64,
    pub fetched_bytes: u64,
    pub cache_hits: u64,
    pub peak_in_flight: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct RangeIoAggregate {
    pub query_count: usize,
    pub logical_requests: u64,
    pub fetch_requests: u64,
    pub cache_hits: u64,
    pub fetched_bytes: u64,
    pub mean_fetched_bytes_per_query: f64,
    pub max_fetched_bytes_per_query: u64,
    pub mean_corpus_ratio: f64,
    pub max_corpus_ratio: f64,
    pub peak_in_flight: u64,
}

impl RangeIoAggregate {
    fn observe(&mut self, stats: RangeIoStats, artifact_bytes: u64) {
        self.query_count += 1;
        self.logical_requests = self.logical_requests.saturating_add(stats.logical_requests);
        self.fetch_requests = self.fetch_requests.saturating_add(stats.fetch_requests);
        self.cache_hits = self.cache_hits.saturating_add(stats.cache_hits);
        self.fetched_bytes = self.fetched_bytes.saturating_add(stats.fetched_bytes);
        self.max_fetched_bytes_per_query =
            self.max_fetched_bytes_per_query.max(stats.fetched_bytes);
        self.peak_in_flight = self.peak_in_flight.max(stats.peak_in_flight);
        let ratio = ratio(stats.fetched_bytes, artifact_bytes);
        self.max_corpus_ratio = self.max_corpus_ratio.max(ratio);
    }

    fn finish(&mut self, artifact_bytes: u64) {
        if self.query_count == 0 {
            return;
        }
        self.mean_fetched_bytes_per_query = self.fetched_bytes as f64 / self.query_count as f64;
        self.mean_corpus_ratio = self.mean_fetched_bytes_per_query / artifact_bytes.max(1) as f64;
    }
}

#[derive(Debug, Serialize)]
pub struct RssReport {
    pub portable: bool,
    pub current_before_bytes: Option<u64>,
    pub current_after_bytes: Option<u64>,
    pub process_high_water_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct GateReport {
    pub thresholds: GateThresholds,
    pub passed: bool,
    pub failures: Vec<String>,
}

impl ObjectNativeReport {
    pub fn passed(&self) -> bool {
        self.gates.passed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RangeKey {
    start: u64,
    end: u64,
}

#[derive(Debug)]
struct CacheEntry {
    bytes: Bytes,
    touched: u64,
}

#[derive(Debug, Default)]
struct BoundedCache {
    entries: HashMap<RangeKey, CacheEntry>,
    used_bytes: usize,
    clock: u64,
}

impl BoundedCache {
    fn get(&mut self, key: RangeKey) -> Option<Bytes> {
        let entry = self.entries.get_mut(&key)?;
        self.clock = self.clock.saturating_add(1);
        entry.touched = self.clock;
        Some(entry.bytes.clone())
    }

    fn insert(&mut self, key: RangeKey, bytes: Bytes, capacity: usize) {
        if capacity == 0 || bytes.len() > capacity {
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes.len());
        }
        while self.used_bytes.saturating_add(bytes.len()) > capacity {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.touched)
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(old) = self.entries.remove(&oldest) {
                self.used_bytes = self.used_bytes.saturating_sub(old.bytes.len());
            }
        }
        self.clock = self.clock.saturating_add(1);
        self.used_bytes = self.used_bytes.saturating_add(bytes.len());
        self.entries.insert(
            key,
            CacheEntry {
                bytes,
                touched: self.clock,
            },
        );
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
    }
}

/// Local file adapter with the same immutable-range contract as S3/R2.
///
/// The local backing removes network variance from CI. `--range-latency-ms`
/// can add deterministic request latency, and the counters still expose the
/// exact request/byte amplification a remote backend would pay.
#[derive(Debug)]
struct InstrumentedRangeSource {
    file: Arc<File>,
    file_len: u64,
    cache_capacity: usize,
    latency: Duration,
    max_in_flight_requests: usize,
    cache: Mutex<BoundedCache>,
    logical_requests: AtomicU64,
    logical_bytes: AtomicU64,
    fetch_requests: AtomicU64,
    fetched_bytes: AtomicU64,
    cache_hits: AtomicU64,
    in_flight: AtomicU64,
    peak_in_flight: AtomicU64,
}

impl InstrumentedRangeSource {
    fn new(
        file: File,
        file_len: u64,
        cache_capacity: usize,
        latency_ms: u64,
        max_in_flight_requests: usize,
    ) -> Self {
        Self {
            file: Arc::new(file),
            file_len,
            cache_capacity,
            latency: Duration::from_millis(latency_ms),
            max_in_flight_requests: max_in_flight_requests.max(1),
            cache: Mutex::new(BoundedCache::default()),
            logical_requests: AtomicU64::new(0),
            logical_bytes: AtomicU64::new(0),
            fetch_requests: AtomicU64::new(0),
            fetched_bytes: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            in_flight: AtomicU64::new(0),
            peak_in_flight: AtomicU64::new(0),
        }
    }

    async fn fetch(&self, range: Range<u64>) -> namidb_storage::Result<Bytes> {
        if range.start > range.end || range.end > self.file_len {
            return Err(namidb_storage::Error::invariant(format!(
                "instrumented source range {}..{} outside object length {}",
                range.start, range.end, self.file_len
            )));
        }
        let len = usize::try_from(range.end - range.start)
            .map_err(|_| namidb_storage::Error::invariant("range does not fit usize"))?;
        self.logical_requests.fetch_add(1, AtomicOrdering::Relaxed);
        self.logical_bytes
            .fetch_add(len as u64, AtomicOrdering::Relaxed);
        let key = RangeKey {
            start: range.start,
            end: range.end,
        };
        if let Some(bytes) = self.cache.lock().expect("range cache poisoned").get(key) {
            self.cache_hits.fetch_add(1, AtomicOrdering::Relaxed);
            return Ok(bytes);
        }

        self.fetch_requests.fetch_add(1, AtomicOrdering::Relaxed);
        let current = self.in_flight.fetch_add(1, AtomicOrdering::SeqCst) + 1;
        update_max(&self.peak_in_flight, current);
        // Let all futures issued by `read_ranges` enter before doing local I/O,
        // making peak-in-flight a useful proxy for object-store concurrency.
        tokio::task::yield_now().await;
        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }
        let result = read_exact_range(&self.file, range.start, len);
        self.in_flight.fetch_sub(1, AtomicOrdering::SeqCst);
        let bytes = result?;
        self.fetched_bytes
            .fetch_add(bytes.len() as u64, AtomicOrdering::Relaxed);
        self.cache.lock().expect("range cache poisoned").insert(
            key,
            bytes.clone(),
            self.cache_capacity,
        );
        Ok(bytes)
    }

    async fn fetch_ranges(&self, ranges: &[Range<u64>]) -> namidb_storage::Result<Vec<Bytes>> {
        let mut values = Vec::with_capacity(ranges.len());
        for chunk in ranges.chunks(self.max_in_flight_requests) {
            values
                .extend(try_join_all(chunk.iter().cloned().map(|range| self.fetch(range))).await?);
        }
        Ok(values)
    }

    fn clear_cache(&self) {
        self.cache.lock().expect("range cache poisoned").clear();
    }

    fn reset_stats(&self) {
        debug_assert_eq!(self.in_flight.load(AtomicOrdering::Relaxed), 0);
        self.logical_requests.store(0, AtomicOrdering::Relaxed);
        self.logical_bytes.store(0, AtomicOrdering::Relaxed);
        self.fetch_requests.store(0, AtomicOrdering::Relaxed);
        self.fetched_bytes.store(0, AtomicOrdering::Relaxed);
        self.cache_hits.store(0, AtomicOrdering::Relaxed);
        self.peak_in_flight.store(0, AtomicOrdering::Relaxed);
    }

    fn stats(&self) -> RangeIoStats {
        RangeIoStats {
            logical_requests: self.logical_requests.load(AtomicOrdering::Relaxed),
            logical_bytes: self.logical_bytes.load(AtomicOrdering::Relaxed),
            fetch_requests: self.fetch_requests.load(AtomicOrdering::Relaxed),
            fetched_bytes: self.fetched_bytes.load(AtomicOrdering::Relaxed),
            cache_hits: self.cache_hits.load(AtomicOrdering::Relaxed),
            peak_in_flight: self.peak_in_flight.load(AtomicOrdering::Relaxed),
        }
    }
}

#[async_trait]
impl VectorV5RangeSource for InstrumentedRangeSource {
    async fn read_range(&self, range: Range<u64>) -> namidb_storage::Result<Bytes> {
        self.fetch(range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> namidb_storage::Result<Vec<Bytes>> {
        self.fetch_ranges(ranges).await
    }
}

#[async_trait]
impl TextIndexRangeSource for InstrumentedRangeSource {
    async fn read_range(&self, range: Range<u64>) -> namidb_storage::Result<Bytes> {
        self.fetch(range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> namidb_storage::Result<Vec<Bytes>> {
        self.fetch_ranges(ranges).await
    }
}

#[async_trait]
impl namidb_storage::sst::search_delta::SearchVersionRangeSource for InstrumentedRangeSource {
    async fn read_range(&self, range: Range<u64>) -> namidb_storage::Result<Bytes> {
        self.fetch(range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> namidb_storage::Result<Vec<Bytes>> {
        self.fetch_ranges(ranges).await
    }
}

#[cfg(unix)]
fn read_exact_range(file: &File, offset: u64, len: usize) -> namidb_storage::Result<Bytes> {
    use std::os::unix::fs::FileExt;

    let mut out = vec![0u8; len];
    let mut read = 0;
    while read < len {
        let n = file.read_at(&mut out[read..], offset + read as u64)?;
        if n == 0 {
            return Err(namidb_storage::Error::invariant(
                "unexpected EOF in instrumented range source",
            ));
        }
        read += n;
    }
    Ok(Bytes::from(out))
}

#[cfg(not(unix))]
fn read_exact_range(file: &File, offset: u64, len: usize) -> namidb_storage::Result<Bytes> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    let mut out = vec![0u8; len];
    file.read_exact(&mut out)?;
    Ok(Bytes::from(out))
}

fn update_max(target: &AtomicU64, value: u64) {
    let mut observed = target.load(AtomicOrdering::Relaxed);
    while value > observed {
        match target.compare_exchange_weak(
            observed,
            value,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual) => observed = actual,
        }
    }
}

pub async fn run(config: ObjectNativeConfig) -> Result<ObjectNativeReport> {
    validate_config(&config)?;
    let rss_before = process_rss();
    let centroids = make_centroids(config.seed, config.clusters, config.dim);
    let mut vector_reports = Vec::with_capacity(3);
    for metric in [
        VectorMetric::Cosine,
        VectorMetric::Dot,
        VectorMetric::Euclidean,
    ] {
        vector_reports.push(run_vector_metric(&config, &centroids, metric).await?);
    }
    let text = run_text(&config).await?;
    let v5_peak_logical_bytes = vector_reports
        .iter()
        .map(|report| report.builder.peak_logical_memory_bytes)
        .max()
        .unwrap_or(0);
    let architecture = run_architecture(
        &config,
        &centroids,
        v5_peak_logical_bytes,
        text.builder.max_buffer_bytes,
    )
    .await?;
    let rss_after = process_rss();
    let rss = RssReport {
        portable: cfg!(target_os = "linux"),
        current_before_bytes: rss_before.as_ref().and_then(|r| r.current),
        current_after_bytes: rss_after.as_ref().and_then(|r| r.current),
        process_high_water_bytes: rss_after.as_ref().and_then(|r| r.high_water),
    };
    let config_report = ConfigReport {
        vectors: config.vectors,
        documents: config.documents,
        dim: config.dim,
        queries: config.queries,
        k: config.k,
        clusters: config.clusters,
        spread: config.spread,
        filter_buckets: config.filter_buckets,
        page_rows: config.page_rows,
        branch_factor: config.branch_factor,
        nprobe: config.nprobe,
        max_nprobe: config.max_nprobe,
        rerank_factor: config.rerank_factor,
        cache_bytes: config.cache_bytes,
        build_memory_bytes: config.build_memory_bytes,
        range_latency_ms: config.range_latency_ms,
        delta_segments: config.delta_segments,
        graph_keys: config.graph_keys,
        graph_high_degree: config.graph_high_degree,
        seed: config.seed,
    };
    let failures = evaluate_gates(
        &config.gates,
        &vector_reports,
        &text,
        &architecture.search_lsm,
        &architecture.graph,
        &rss,
    );
    Ok(ObjectNativeReport {
        format: REPORT_FORMAT,
        config: config_report,
        vectors: vector_reports,
        text,
        search_lsm: architecture.search_lsm,
        graph: architecture.graph,
        builder_workspace: architecture.builder_workspace,
        rss,
        gates: GateReport {
            thresholds: config.gates,
            passed: failures.is_empty(),
            failures,
        },
    })
}

fn validate_config(config: &ObjectNativeConfig) -> Result<()> {
    if config.vectors < 2 {
        bail!("--vectors must be at least 2");
    }
    if config.documents == 0 {
        bail!("--documents must be greater than zero");
    }
    if config.dim == 0 {
        bail!("--dim must be greater than zero");
    }
    if config.queries == 0 {
        bail!("--queries must be greater than zero");
    }
    if config.k == 0 || config.k > config.vectors {
        bail!("--k must be in 1..=vectors");
    }
    if config.filter_buckets == 0 {
        bail!("--filter-buckets must be greater than zero");
    }
    let eligible = (0..config.vectors)
        .filter(|&row| filter_bucket(row, config.clusters) % config.filter_buckets == 0)
        .count();
    if eligible < config.k {
        bail!(
            "selective filter has only {eligible} rows, below k={}; reduce \
             --filter-buckets or increase --vectors",
            config.k
        );
    }
    if config.page_rows < 2 {
        bail!("--page-rows must be at least 2");
    }
    if config.branch_factor < 2 {
        bail!("--branch-factor must be at least 2");
    }
    if config.nprobe == 0 || config.max_nprobe < config.nprobe {
        bail!("require 1 <= --nprobe <= --max-nprobe");
    }
    if config.rerank_factor == 0 {
        bail!("--rerank-factor must be greater than zero");
    }
    if config.cache_bytes < 4096 {
        bail!("--cache-bytes must be at least 4096 for the explicit graph range cache");
    }
    if config.build_memory_bytes < 512 * 1024 {
        bail!("--build-memory-bytes must be at least 524288 for the VG6 external builder");
    }
    if config.delta_segments == 0
        || config.delta_segments.saturating_add(1)
            > namidb_storage::search_lsm::MAX_ACTIVE_SEARCH_SEGMENTS
    {
        bail!(
            "--delta-segments must be in 1..{}",
            namidb_storage::search_lsm::MAX_ACTIVE_SEARCH_SEGMENTS
        );
    }
    if config.vectors < config.delta_segments.saturating_mul(8)
        || config.documents < config.delta_segments.saturating_mul(8)
    {
        bail!(
            "vectors/documents must each be at least 8 * --delta-segments so every \
             update/delete/filter-only fixture is non-empty"
        );
    }
    if config.graph_keys < 2 {
        bail!("--graph-keys must be at least 2");
    }
    if config.graph_high_degree < 2 {
        bail!("--graph-high-degree must be at least 2");
    }
    if config.gates.max_fanout == 0 {
        bail!("--max-fanout must be greater than zero");
    }
    if config.gates.max_in_flight == 0 {
        bail!("--max-in-flight must be greater than zero");
    }
    Ok(())
}

async fn run_vector_metric(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    metric: VectorMetric,
) -> Result<VectorMetricReport> {
    let scratch = tempfile::tempdir().context("create vector build scratch directory")?;
    let mut collector = VectorV5ExternalCollector::new_base(VectorV5ExternalBuildConfig {
        scratch_dir: scratch.path().to_path_buf(),
        memory_budget_bytes: config.build_memory_bytes,
        quantile_sample_rows: 16_384.min(config.vectors.max(2)),
    })
    .map_err(|error| anyhow!("create NAMIVG05 collector: {error}"))?;
    let build_start = Instant::now();
    for row in 0..config.vectors {
        let vector = corpus_vector(config, centroids, metric, row);
        let filters = BTreeMap::from([
            (
                "tenant".to_owned(),
                Value::Str(format!(
                    "tenant-{}",
                    filter_bucket(row, config.clusters) % config.filter_buckets
                )),
            ),
            ("active".to_owned(), Value::Bool(row % 3 != 0)),
        ]);
        collector
            .push(node_id(row), &vector, &filters)
            .map_err(|error| anyhow!("append NAMIVG05 row {row}: {error}"))?;
    }
    let desc = vector_descriptor(config.dim, metric);
    let options = VectorV5BuildOptions {
        target_rows_per_page: config.page_rows,
        branch_factor: config.branch_factor,
        compression_level: 3,
    };
    let artifact = collector
        .finish(&desc, options)
        .map_err(|error| anyhow!("finish NAMIVG05 {}: {error}", metric_name(metric)))?
        .ok_or_else(|| anyhow!("NAMIVG05 builder produced no artifact"))?;
    let build_secs = build_start.elapsed().as_secs_f64();
    let artifact_bytes = artifact.len;
    let builder = VectorBuilderReport::from(artifact.metrics);
    let source = Arc::new(InstrumentedRangeSource::new(
        artifact.file,
        artifact_bytes,
        config.cache_bytes,
        config.range_latency_ms,
        config.gates.max_in_flight as usize,
    ));
    source.clear_cache();
    source.reset_stats();
    let reader = VectorV5Reader::open(
        source.clone() as Arc<dyn VectorV5RangeSource>,
        artifact_bytes,
    )
    .await
    .map_err(|error| anyhow!("open NAMIVG05 {}: {error}", metric_name(metric)))?;
    let reader_open_io = source.stats();
    let reader_metadata_bytes = reader.resident_metadata_bytes();
    let page_count = reader.page_count();
    let search_options = VectorV5SearchOptions {
        nprobe: config.nprobe.min(page_count).max(1),
        max_nprobe: config.max_nprobe.min(page_count).max(1),
        rerank_factor: config.rerank_factor,
    };

    let mut query_vectors = Vec::with_capacity(config.queries);
    for query_no in 0..config.queries {
        query_vectors.push(query_vector(config, centroids, metric, query_no));
    }
    // One corpus pass serves every exact query and both eligibility tracks.
    // At 10M rows this avoids regenerating each high-dimensional vector
    // `2 * queries` times while retaining only O(queries * k) truth state.
    let (exact_unfiltered, exact_filtered) =
        exact_vector_top_k_batch(config, centroids, metric, &query_vectors);

    let unfiltered = measure_vector_track(
        &reader,
        &source,
        artifact_bytes,
        &query_vectors,
        &exact_unfiltered,
        search_options,
        None,
    )
    .await?;
    let selective_filter = measure_vector_track(
        &reader,
        &source,
        artifact_bytes,
        &query_vectors,
        &exact_filtered,
        search_options,
        Some(vec![(
            "tenant".to_owned(),
            vec![Value::Str("tenant-0".to_owned())],
        )]),
    )
    .await?;

    Ok(VectorMetricReport {
        metric: metric_name(metric),
        build_secs,
        artifact_bytes,
        artifact_bytes_per_vector: artifact_bytes as f64 / config.vectors as f64,
        page_count,
        reader_metadata_bytes,
        reader_metadata_artifact_ratio: reader_metadata_bytes as f64 / artifact_bytes.max(1) as f64,
        reader_open_io,
        builder,
        unfiltered,
        selective_filter,
    })
}

#[allow(clippy::too_many_arguments)]
async fn measure_vector_track(
    reader: &VectorV5Reader,
    source: &Arc<InstrumentedRangeSource>,
    artifact_bytes: u64,
    queries: &[Vec<f32>],
    exact: &[Vec<([u8; 16], f32)>],
    options: VectorV5SearchOptions,
    groups: Option<Vec<(String, Vec<Value>)>>,
) -> Result<VectorQueryTrack> {
    let mut cold_us = Vec::with_capacity(queries.len());
    let mut warm_us = Vec::with_capacity(queries.len());
    let mut cold_io = RangeIoAggregate::default();
    let mut warm_io = RangeIoAggregate::default();
    let mut recall_hits = 0usize;
    let mut recall_total = 0usize;
    let mut native_filter_applied = true;

    for (query, expected) in queries.iter().zip(exact) {
        source.clear_cache();
        source.reset_stats();
        let started = Instant::now();
        let cold = if let Some(groups) = groups.as_deref() {
            reader
                .search_filter_groups(query, expected.len(), options, groups)
                .await
                .map_err(|error| anyhow!("cold filtered NAMIVG05 query: {error}"))?
        } else {
            let hits = reader
                .search(query, expected.len(), options)
                .await
                .map_err(|error| anyhow!("cold NAMIVG05 query: {error}"))?;
            namidb_storage::sst::vector::v5::VectorV5SearchResult {
                hits,
                applied_filter_groups: 0,
                probed_pages: 0,
                eligible_rows_seen: 0,
            }
        };
        cold_us.push(elapsed_us(started));
        cold_io.observe(source.stats(), artifact_bytes);
        if let Some(groups) = groups.as_deref() {
            native_filter_applied &= cold.applied_filter_groups == groups.len();
        }
        let expected_ids = expected.iter().map(|(id, _)| *id).collect::<BTreeSet<_>>();
        recall_hits += cold
            .hits
            .iter()
            .filter(|(id, _)| expected_ids.contains(id))
            .count();
        recall_total += expected_ids.len();

        source.reset_stats();
        let started = Instant::now();
        if let Some(groups) = groups.as_deref() {
            let warm = reader
                .search_filter_groups(query, expected.len(), options, groups)
                .await
                .map_err(|error| anyhow!("warm filtered NAMIVG05 query: {error}"))?;
            native_filter_applied &= warm.applied_filter_groups == groups.len();
        } else {
            reader
                .search(query, expected.len(), options)
                .await
                .map_err(|error| anyhow!("warm NAMIVG05 query: {error}"))?;
        }
        warm_us.push(elapsed_us(started));
        warm_io.observe(source.stats(), artifact_bytes);
    }
    cold_us.sort_unstable();
    warm_us.sort_unstable();
    cold_io.finish(artifact_bytes);
    warm_io.finish(artifact_bytes);
    Ok(VectorQueryTrack {
        query_count: queries.len(),
        recall_at_k: if recall_total == 0 {
            1.0
        } else {
            recall_hits as f64 / recall_total as f64
        },
        native_filter_applied,
        cold_p50_us: percentile(&cold_us, 50.0),
        cold_p95_us: percentile(&cold_us, 95.0),
        warm_p50_us: percentile(&warm_us, 50.0),
        warm_p95_us: percentile(&warm_us, 95.0),
        cold_io,
        warm_io,
    })
}

async fn run_text(config: &ObjectNativeConfig) -> Result<TextReport> {
    let scratch = tempfile::tempdir().context("create text build scratch directory")?;
    let mut builder = TextIndexExternalBuilder::with_options(ExternalTextIndexBuildOptions {
        memory_budget_bytes: config.build_memory_bytes,
        spool_directory: Some(scratch.path().to_path_buf()),
    })
    .map_err(|error| anyhow!("create NAMIFT03 builder: {error}"))?;
    let build_start = Instant::now();
    for row in 0..config.documents {
        builder
            .push((node_id(row), legal_document(row, config.seed)))
            .map_err(|error| anyhow!("append NAMIFT03 document {row}: {error}"))?;
    }
    let artifact = builder
        .finish()
        .map_err(|error| anyhow!("finish NAMIFT03: {error}"))?
        .ok_or_else(|| anyhow!("NAMIFT03 builder produced no artifact"))?;
    let build_secs = build_start.elapsed().as_secs_f64();
    let artifact_bytes = artifact.len();
    let stats = artifact.stats().clone();
    let build_metrics = artifact.metrics();
    let (file, _, _, _) = artifact.into_parts();
    let source = Arc::new(InstrumentedRangeSource::new(
        file,
        artifact_bytes,
        config.cache_bytes,
        config.range_latency_ms,
        config.gates.max_in_flight as usize,
    ));
    source.clear_cache();
    source.reset_stats();
    let reader = TextIndexV3Reader::open(
        source.clone() as Arc<dyn TextIndexRangeSource>,
        artifact_bytes,
    )
    .await
    .map_err(|error| anyhow!("open NAMIFT03: {error}"))?;
    let reader_open_io = source.stats();
    let reader_metadata_bytes = reader.estimated_resident_bytes();

    let specs = [
        ("bm25", "plain", "reforma articulo"),
        ("phrase", "phrase", "\"derecho laboral\""),
        ("prefix", "prefix", "constitu*"),
        (
            "combined",
            "phrase+prefix",
            "\"derecho laboral\" reforma constitu*",
        ),
    ];
    let vocabulary = text_vocabulary(config.documents, config.seed);
    let parsed_queries = specs
        .iter()
        .map(|(_, _, raw)| parse_query(raw))
        .collect::<Vec<_>>();
    // As with vector truth, score every syntax variant in shared corpus
    // passes. This keeps the 10M command practical without changing the exact
    // BM25/phrase/prefix oracle.
    let exact_results = exact_text_top_k_batch(
        config.documents,
        config.seed,
        &vocabulary,
        &parsed_queries,
        config.k,
    );
    let mut cold_us = Vec::with_capacity(specs.len());
    let mut warm_us = Vec::with_capacity(specs.len());
    let mut cold_io = RangeIoAggregate::default();
    let mut warm_io = RangeIoAggregate::default();
    let mut parity = Vec::with_capacity(specs.len());
    for (((name, syntax, _), query), expected) in
        specs.into_iter().zip(&parsed_queries).zip(&exact_results)
    {
        source.clear_cache();
        source.reset_stats();
        let started = Instant::now();
        let cold = reader
            .search_query(&query, Some(config.k))
            .await
            .map_err(|error| anyhow!("cold NAMIFT03 {name} query: {error}"))?;
        cold_us.push(elapsed_us(started));
        cold_io.observe(source.stats(), artifact_bytes);

        let (node_ids_equal, scores_equal, max_score_abs_error) =
            compare_text_results(&expected, &cold);
        parity.push(TextParityReport {
            name,
            syntax,
            result_count: cold.len(),
            node_ids_equal,
            scores_equal,
            max_score_abs_error,
        });

        source.reset_stats();
        let started = Instant::now();
        let warm = reader
            .search_query(&query, Some(config.k))
            .await
            .map_err(|error| anyhow!("warm NAMIFT03 {name} query: {error}"))?;
        warm_us.push(elapsed_us(started));
        warm_io.observe(source.stats(), artifact_bytes);
        let (warm_ids, warm_scores, _) = compare_text_results(&cold, &warm);
        if !warm_ids || !warm_scores {
            bail!("NAMIFT03 warm result diverged from its cold result for {name}");
        }
    }
    cold_us.sort_unstable();
    warm_us.sort_unstable();
    cold_io.finish(artifact_bytes);
    warm_io.finish(artifact_bytes);

    Ok(TextReport {
        build_secs,
        artifact_bytes,
        artifact_bytes_per_document: artifact_bytes as f64 / config.documents as f64,
        document_count: stats.doc_count,
        term_count: stats.term_count,
        reader_metadata_bytes,
        reader_metadata_artifact_ratio: reader_metadata_bytes as f64 / artifact_bytes.max(1) as f64,
        reader_open_io,
        builder: TextBuilderReport {
            memory_budget_bytes: build_metrics.memory_budget_bytes,
            max_buffer_bytes: build_metrics.max_buffer_bytes,
            initial_run_count: build_metrics.initial_run_count,
            run_merge_count: build_metrics.run_merge_count,
            spool_bytes_written: build_metrics.spool_bytes_written,
        },
        cold_p50_us: percentile(&cold_us, 50.0),
        cold_p95_us: percentile(&cold_us, 95.0),
        warm_p50_us: percentile(&warm_us, 50.0),
        warm_p95_us: percentile(&warm_us, 95.0),
        cold_io,
        warm_io,
        parity,
    })
}

fn vector_descriptor(dim: usize, metric: VectorMetric) -> VectorIndexDescriptor {
    VectorIndexDescriptor {
        name: format!("object-native-{}", metric_name(metric)),
        label: "Document".to_owned(),
        property: "embedding".to_owned(),
        dim: dim as u32,
        metric,
        r: 32,
        l_build: 64,
        alpha: 1.2,
        quantization: VectorQuantization::None,
    }
}

fn make_centroids(seed: u64, clusters: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x4345_4e54_524f_4944);
    (0..clusters)
        .map(|_| random_unit_vector(&mut rng, dim))
        .collect()
}

fn corpus_vector(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    row: usize,
) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(mix64(
        config.seed ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
    ));
    let mut vector = if centroids.is_empty() {
        random_unit_vector(&mut rng, config.dim)
    } else {
        perturbed_unit_vector(&mut rng, &centroids[row % centroids.len()], config.spread)
    };
    if metric == VectorMetric::Dot {
        let magnitude = 0.5 + (mix64(row as u64 ^ config.seed) % 1_501) as f32 / 1_000.0;
        for component in &mut vector {
            *component *= magnitude;
        }
    }
    vector
}

fn query_vector(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    query: usize,
) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(mix64(
        config.seed ^ 0x5155_4552_595f_5635 ^ (query as u64).wrapping_mul(0xd6e8_feb8_6659_fd93),
    ));
    let mut vector = if centroids.is_empty() {
        random_unit_vector(&mut rng, config.dim)
    } else {
        perturbed_unit_vector(
            &mut rng,
            &centroids[query % centroids.len()],
            config.spread * 0.5,
        )
    };
    if metric == VectorMetric::Dot {
        for value in &mut vector {
            *value *= 1.25;
        }
    }
    vector
}

fn exact_vector_top_k_batch(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    metric: VectorMetric,
    queries: &[Vec<f32>],
) -> (Vec<Vec<([u8; 16], f32)>>, Vec<Vec<([u8; 16], f32)>>) {
    let mut unfiltered = (0..queries.len())
        .map(|_| Vec::with_capacity(config.k))
        .collect::<Vec<_>>();
    let mut filtered = (0..queries.len())
        .map(|_| Vec::with_capacity(config.k))
        .collect::<Vec<_>>();
    let query_norms = queries
        .iter()
        .map(|query| {
            if metric == VectorMetric::Cosine {
                l2_norm(query)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    for row in 0..config.vectors {
        let vector = corpus_vector(config, centroids, metric, row);
        let vector_norm = if metric == VectorMetric::Cosine {
            l2_norm(&vector)
        } else {
            0.0
        };
        let id = node_id(row);
        let eligible = filter_bucket(row, config.clusters) % config.filter_buckets == 0;
        for (query_no, query) in queries.iter().enumerate() {
            let candidate = (
                id,
                vector_score(metric, query, &vector, query_norms[query_no], vector_norm),
            );
            retain_top_k(&mut unfiltered[query_no], candidate, config.k, metric);
            if eligible {
                retain_top_k(&mut filtered[query_no], candidate, config.k, metric);
            }
        }
    }
    for hits in unfiltered.iter_mut().chain(&mut filtered) {
        hits.sort_by(|left, right| compare_vector_hits(left, right, metric));
    }
    (unfiltered, filtered)
}

fn retain_top_k(
    best: &mut Vec<([u8; 16], f32)>,
    candidate: ([u8; 16], f32),
    k: usize,
    metric: VectorMetric,
) {
    if best.len() < k {
        best.push(candidate);
        return;
    }
    let mut worst = 0;
    for index in 1..best.len() {
        if compare_vector_hits(&best[worst], &best[index], metric) == Ordering::Less {
            // `worst` sorts before `index`, therefore `index` is worse.
            worst = index;
        }
    }
    if compare_vector_hits(&candidate, &best[worst], metric) == Ordering::Less {
        best[worst] = candidate;
    }
}

fn compare_vector_hits(
    left: &([u8; 16], f32),
    right: &([u8; 16], f32),
    metric: VectorMetric,
) -> Ordering {
    let score_order = match metric {
        VectorMetric::Cosine | VectorMetric::Dot => right.1.total_cmp(&left.1),
        VectorMetric::Euclidean => left.1.total_cmp(&right.1),
    };
    score_order.then_with(|| left.0.cmp(&right.0))
}

fn vector_score(
    metric: VectorMetric,
    left: &[f32],
    right: &[f32],
    left_norm: f32,
    right_norm: f32,
) -> f32 {
    match metric {
        VectorMetric::Dot => left.iter().zip(right).map(|(a, b)| a * b).sum(),
        VectorMetric::Euclidean => left
            .iter()
            .zip(right)
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt(),
        VectorMetric::Cosine => {
            let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
            if left_norm == 0.0 || right_norm == 0.0 {
                0.0
            } else {
                dot / (left_norm * right_norm)
            }
        }
    }
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn filter_bucket(row: usize, clusters: usize) -> usize {
    row / clusters.max(1)
}

fn metric_name(metric: VectorMetric) -> &'static str {
    match metric {
        VectorMetric::Cosine => "cosine",
        VectorMetric::Dot => "dot",
        VectorMetric::Euclidean => "l2",
    }
}

fn node_id(row: usize) -> [u8; 16] {
    (row as u128 + 1).to_be_bytes()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn legal_document(row: usize, seed: u64) -> String {
    const DOMAINS: &[&str] = &[
        "laboral",
        "mercantil",
        "penal",
        "fiscal",
        "civil",
        "administrativo",
        "constitucional",
        "europeo",
    ];
    const ACTIONS: &[&str] = &[
        "reforma",
        "regula",
        "deroga",
        "protege",
        "interpreta",
        "modifica",
        "garantiza",
        "sanciona",
    ];
    const AUTHORITIES: &[&str] = &[
        "tribunal",
        "legislador",
        "ministerio",
        "audiencia",
        "comision",
        "juzgado",
    ];
    let salt = mix64(seed ^ row as u64);
    let domain = DOMAINS[(salt as usize) % DOMAINS.len()];
    // Guarantee that the combined phrase+prefix workload has enough winners
    // even in the 256-document unit corpus.
    let action = if row % 7 == 0 {
        "reforma"
    } else {
        ACTIONS[((salt >> 11) as usize) % ACTIONS.len()]
    };
    let authority = AUTHORITIES[((salt >> 23) as usize) % AUTHORITIES.len()];
    let phrase = if row % 7 == 0 {
        "derecho laboral tutela efectiva"
    } else if row % 7 == 1 {
        "derecho procesal tutela judicial"
    } else {
        "ordenamiento juridico vigente"
    };
    let constitution = match row % 11 {
        0 => "constitucion",
        1 => "constitucional",
        2 => "constitucionalidad",
        _ => "normativa",
    };
    format!(
        "articulo codigo{:04} {domain} {action} {authority} {phrase} \
         {constitution} disposicion legal expediente{:03}",
        row % 4096,
        row % 997
    )
}

fn text_vocabulary(documents: usize, seed: u64) -> BTreeSet<String> {
    let mut vocabulary = BTreeSet::new();
    for row in 0..documents {
        vocabulary.extend(tokenize(&legal_document(row, seed)));
    }
    vocabulary
}

fn exact_text_top_k_batch(
    documents: usize,
    seed: u64,
    vocabulary: &BTreeSet<String>,
    queries: &[TextQuery],
    k: usize,
) -> Vec<Vec<([u8; 16], f64)>> {
    let terms_by_query = queries
        .iter()
        .map(|query| expanded_terms(query, vocabulary))
        .collect::<Vec<_>>();
    let all_terms = terms_by_query
        .iter()
        .flat_map(|terms| terms.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut doc_freq = all_terms
        .iter()
        .map(|term| (term.clone(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut total_len = 0u64;
    for row in 0..documents {
        let tokens = tokenize(&legal_document(row, seed));
        total_len = total_len.saturating_add(tokens.len() as u64);
        let present = tokens.iter().map(String::as_str).collect::<BTreeSet<_>>();
        for term in &all_terms {
            if present.contains(term.as_str()) {
                *doc_freq.get_mut(term).expect("term initialized") += 1;
            }
        }
    }

    let average = avg_len(total_len, documents);
    let mut best = (0..queries.len())
        .map(|_| Vec::with_capacity(k))
        .collect::<Vec<_>>();
    for row in 0..documents {
        let tokens = tokenize(&legal_document(row, seed));
        let mut frequencies = HashMap::<&str, u32>::new();
        for token in &tokens {
            *frequencies.entry(token).or_default() += 1;
        }
        for (query_no, (query, scored_terms)) in queries.iter().zip(&terms_by_query).enumerate() {
            if !query
                .phrases
                .iter()
                .all(|phrase| contains_phrase(&tokens, phrase))
            {
                continue;
            }
            let mut score = 0.0;
            let mut matched = false;
            for term in scored_terms {
                let Some(tf) = frequencies.get(term.as_str()) else {
                    continue;
                };
                matched = true;
                let idf = bm25_idf(documents, doc_freq[term]);
                score += bm25_term_score(idf, *tf, tokens.len(), average);
            }
            if matched {
                retain_text_top_k(&mut best[query_no], (node_id(row), score), k);
            }
        }
    }
    for hits in &mut best {
        hits.sort_by(compare_text_hits);
    }
    best
}

fn expanded_terms(query: &TextQuery, vocabulary: &BTreeSet<String>) -> BTreeSet<String> {
    let mut terms = query
        .base_terms()
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    for prefix in &query.prefixes {
        terms.extend(
            vocabulary
                .range(prefix.clone()..)
                .take_while(|term| term.starts_with(prefix))
                .take(PREFIX_EXPANSION_LIMIT)
                .cloned(),
        );
    }
    terms
}

fn retain_text_top_k(best: &mut Vec<([u8; 16], f64)>, candidate: ([u8; 16], f64), k: usize) {
    if best.len() < k {
        best.push(candidate);
        return;
    }
    let mut worst = 0;
    for index in 1..best.len() {
        if compare_text_hits(&best[worst], &best[index]) == Ordering::Less {
            worst = index;
        }
    }
    if compare_text_hits(&candidate, &best[worst]) == Ordering::Less {
        best[worst] = candidate;
    }
}

fn compare_text_hits(left: &([u8; 16], f64), right: &([u8; 16], f64)) -> Ordering {
    right
        .1
        .partial_cmp(&left.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.0.cmp(&right.0))
}

fn compare_text_results(
    expected: &[([u8; 16], f64)],
    actual: &[([u8; 16], f64)],
) -> (bool, bool, f64) {
    let node_ids_equal = expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(left, right)| left.0 == right.0);
    let mut max_error = 0.0f64;
    let scores_equal = expected.len() == actual.len()
        && expected.iter().zip(actual).all(|(left, right)| {
            let error = (left.1 - right.1).abs();
            max_error = max_error.max(error);
            error <= 1e-9
        });
    (node_ids_equal, scores_equal, max_error)
}

fn evaluate_gates(
    thresholds: &GateThresholds,
    vectors: &[VectorMetricReport],
    text: &TextReport,
    search_lsm: &SearchLsmReport,
    graph: &GraphPagedReport,
    rss: &RssReport,
) -> Vec<String> {
    let mut failures = Vec::new();
    for report in vectors {
        for (track_name, track) in [
            ("unfiltered", &report.unfiltered),
            ("selective_filter", &report.selective_filter),
        ] {
            if let Some(minimum) = thresholds.min_recall_at_k {
                if track.recall_at_k < minimum {
                    failures.push(format!(
                        "vector {} {track_name} recall {:.6} < {:.6}",
                        report.metric, track.recall_at_k, minimum
                    ));
                }
            }
            if let Some(maximum) = thresholds.max_cold_bytes_ratio {
                if track.cold_io.max_corpus_ratio > maximum {
                    failures.push(format!(
                        "vector {} {track_name} cold ratio {:.6} > {:.6}",
                        report.metric, track.cold_io.max_corpus_ratio, maximum
                    ));
                }
            }
            if let Some(maximum) = thresholds.max_query_p95_ms {
                if track.cold_p95_us as f64 / 1_000.0 > maximum {
                    failures.push(format!(
                        "vector {} {track_name} cold p95 {:.3}ms > {:.3}ms",
                        report.metric,
                        track.cold_p95_us as f64 / 1_000.0,
                        maximum
                    ));
                }
            }
        }
        if !report.selective_filter.native_filter_applied {
            failures.push(format!(
                "vector {} did not apply the selective filter natively",
                report.metric
            ));
        }
        if let Some(maximum) = thresholds.max_reader_metadata_ratio {
            if report.reader_metadata_artifact_ratio > maximum {
                failures.push(format!(
                    "vector {} metadata ratio {:.6} > {:.6}",
                    report.metric, report.reader_metadata_artifact_ratio, maximum
                ));
            }
        }
    }
    if let Some(maximum) = thresholds.max_cold_bytes_ratio {
        if text.cold_io.max_corpus_ratio > maximum {
            failures.push(format!(
                "text cold ratio {:.6} > {:.6}",
                text.cold_io.max_corpus_ratio, maximum
            ));
        }
    }
    if let Some(maximum) = thresholds.max_reader_metadata_ratio {
        if text.reader_metadata_artifact_ratio > maximum {
            failures.push(format!(
                "text metadata ratio {:.6} > {:.6}",
                text.reader_metadata_artifact_ratio, maximum
            ));
        }
    }
    if let Some(maximum) = thresholds.max_query_p95_ms {
        if text.cold_p95_us as f64 / 1_000.0 > maximum {
            failures.push(format!(
                "text cold p95 {:.3}ms > {:.3}ms",
                text.cold_p95_us as f64 / 1_000.0,
                maximum
            ));
        }
    }
    for parity in &text.parity {
        if !parity.node_ids_equal || !parity.scores_equal {
            failures.push(format!(
                "text {} parity failed (ids={}, scores={}, max_error={:.3e})",
                parity.name, parity.node_ids_equal, parity.scores_equal, parity.max_score_abs_error
            ));
        }
    }
    let vector_options = &search_lsm.vector_serving_options;
    if vector_options.nprobe == 0
        || vector_options.max_nprobe < vector_options.nprobe
        || vector_options.rerank_factor == 0
        || !vector_options.base_algorithm.contains("clustered V5 ANN")
        || !vector_options.delta_algorithm.contains("exact VG6")
    {
        failures.push(format!(
            "V5+VG6 serving options are invalid (nprobe={}, max_nprobe={}, rerank={})",
            vector_options.nprobe, vector_options.max_nprobe, vector_options.rerank_factor
        ));
    }
    for (track_name, track) in [
        ("unfiltered", &search_lsm.vector_serving_ann.unfiltered),
        (
            "active_filter",
            &search_lsm.vector_serving_ann.active_filter,
        ),
    ] {
        if track.result_model != "serving_ann_vs_exact_oracle"
            || !track.returned_winners_valid
            || !track.returned_scores_exact
            || !track.cache_results_identical
            || track.parity.actual_hits != track.parity.expected_hits
        {
            failures.push(format!(
                "V5+VG6 serving ANN {track_name} returned an invalid winner/score \
                 (model={}, winners={}, scores={}, cache_identical={}, expected={}, actual={}, \
                  max_error={:.3e})",
                track.result_model,
                track.returned_winners_valid,
                track.returned_scores_exact,
                track.cache_results_identical,
                track.parity.expected_hits,
                track.parity.actual_hits,
                track.max_returned_score_abs_error,
            ));
        }
        if !track.recall_at_k.is_finite() || !(0.0..=1.0).contains(&track.recall_at_k) {
            failures.push(format!(
                "V5+VG6 serving ANN {track_name} emitted invalid recall {}",
                track.recall_at_k
            ));
        }
        if let Some(minimum) = thresholds.min_recall_at_k {
            if track.recall_at_k < minimum {
                failures.push(format!(
                    "V5+VG6 serving ANN {track_name} recall {:.6} < {minimum:.6}",
                    track.recall_at_k
                ));
            }
        }
        if track_name == "active_filter" && !track.native_filter_applied {
            failures.push("V5+VG6 serving ANN did not apply the active filter natively".to_owned());
        }
        if let Some(maximum) = thresholds.max_cold_bytes_ratio {
            let observed = track
                .no_cache
                .io
                .max_corpus_ratio
                .max(track.sized_cache_cold.io.max_corpus_ratio);
            if observed > maximum {
                failures.push(format!(
                    "V5+VG6 serving ANN {track_name} cold ratio {observed:.6} > {maximum:.6}"
                ));
            }
        }
        if let Some(maximum) = thresholds.max_query_p95_ms {
            let observed_us = track.no_cache.p95_us.max(track.sized_cache_cold.p95_us);
            if observed_us as f64 / 1_000.0 > maximum {
                failures.push(format!(
                    "V5+VG6 serving ANN {track_name} cold p95 {:.3}ms > {maximum:.3}ms",
                    observed_us as f64 / 1_000.0,
                ));
            }
        }
    }
    let shadow = &search_lsm.vector_exact_shadow;
    if !shadow.unfiltered.node_ids_exact
        || !shadow.unfiltered.scores_exact
        || !shadow.active_filter.node_ids_exact
        || !shadow.active_filter.scores_exact
        || !shadow.active_filter_applied
        || shadow.query_count != search_lsm.vector_serving_ann.unfiltered.query_count
        || shadow.io_accounting
            != "correctness-only exact shadow; range/latency counters excluded from serving ANN I/O and latency SLOs"
    {
        failures.push(format!(
            "V5+VG6 exact shadow reconciliation failed \
             (unfiltered_ids={}, unfiltered_scores={}, filtered_ids={}, \
              filtered_scores={}, filter_applied={})",
            shadow.unfiltered.node_ids_exact,
            shadow.unfiltered.scores_exact,
            shadow.active_filter.node_ids_exact,
            shadow.active_filter.scores_exact,
            shadow.active_filter_applied,
        ));
    }
    for (track_name, track) in [
        ("unfiltered", &search_lsm.text.unfiltered),
        ("active_filter", &search_lsm.text.active_filter),
    ] {
        if !track.parity.node_ids_exact || !track.parity.scores_exact {
            failures.push(format!(
                "FT4 {track_name} winner parity failed \
                 (ids={}, scores={}, max_error={:.3e}, expected={}, actual={})",
                track.parity.node_ids_exact,
                track.parity.scores_exact,
                track.parity.max_score_abs_error,
                track.parity.expected_hits,
                track.parity.actual_hits,
            ));
        }
        if track_name == "active_filter" && !track.native_filter_applied {
            failures.push("FT4 did not apply the active filter natively".to_owned());
        }
        if let Some(maximum) = thresholds.max_cold_bytes_ratio {
            let observed = track
                .no_cache
                .io
                .max_corpus_ratio
                .max(track.sized_cache_cold.io.max_corpus_ratio);
            if observed > maximum {
                failures.push(format!(
                    "FT4 {track_name} cold ratio {observed:.6} > {maximum:.6}"
                ));
            }
        }
        if let Some(maximum) = thresholds.max_query_p95_ms {
            let observed_us = track.no_cache.p95_us.max(track.sized_cache_cold.p95_us);
            if observed_us as f64 / 1_000.0 > maximum {
                failures.push(format!(
                    "FT4 {track_name} cold p95 {:.3}ms > {maximum:.3}ms",
                    observed_us as f64 / 1_000.0,
                ));
            }
        }
    }
    if let Some(maximum) = thresholds.max_reader_metadata_ratio {
        for (family_name, family) in [
            ("V5+VG6", &search_lsm.vector_serving_ann),
            ("FT4", &search_lsm.text),
        ] {
            if family.resident_metadata_artifact_ratio > maximum {
                failures.push(format!(
                    "{family_name} metadata ratio {:.6} > {maximum:.6}",
                    family.resident_metadata_artifact_ratio
                ));
            }
        }
    }
    if search_lsm.max_fanout_observed > thresholds.max_fanout {
        failures.push(format!(
            "Search-LSM fanout {} > {}",
            search_lsm.max_fanout_observed, thresholds.max_fanout
        ));
    }
    if search_lsm.max_in_flight_observed > thresholds.max_in_flight {
        failures.push(format!(
            "immutable range peak in-flight {} > {}",
            search_lsm.max_in_flight_observed, thresholds.max_in_flight
        ));
    }
    let v5_base = &search_lsm.v5_base_contract;
    if !v5_base.role_is_base
        || !v5_base.format_is_v5
        || !v5_base.live_count_is_absolute
        || !v5_base.suppress_count_is_zero
        || !v5_base.native_footer_validated
        || !v5_base.legacy_manifest_binding_nonzero
        || v5_base.point_count.is_none()
        || v5_base.clustered_page_count == 0
        || v5_base.bounded_builder_peak_logical_bytes == 0
    {
        failures.push(format!(
            "V5 seed is not an authoritative clustered absolute Base \
             (role_base={}, format_v5={}, live_absolute={}, suppress_zero={}, footer_valid={}, \
              binding_nonzero={}, point_count={:?}, pages={}, builder_peak={})",
            v5_base.role_is_base,
            v5_base.format_is_v5,
            v5_base.live_count_is_absolute,
            v5_base.suppress_count_is_zero,
            v5_base.native_footer_validated,
            v5_base.legacy_manifest_binding_nonzero,
            v5_base.point_count,
            v5_base.clustered_page_count,
            v5_base.bounded_builder_peak_logical_bytes,
        ));
    }
    let ft4_base = &search_lsm.ft4_base_contract;
    if !ft4_base.role_is_base
        || !ft4_base.doc_count_is_absolute
        || !ft4_base.total_len_is_absolute
        || !ft4_base.suppress_count_is_zero
    {
        failures.push(format!(
            "FT4 seed is not an authoritative absolute Base \
             (role_base={}, docs_absolute={}, total_len_absolute={}, suppress_zero={})",
            ft4_base.role_is_base,
            ft4_base.doc_count_is_absolute,
            ft4_base.total_len_is_absolute,
            ft4_base.suppress_count_is_zero,
        ));
    }
    if search_lsm.text_segment_roles.first() != Some(&"base")
        || search_lsm.text_segment_roles.len() != search_lsm.total_segments_per_family
        || search_lsm
            .text_segment_roles
            .iter()
            .skip(1)
            .any(|role| *role != "delta")
    {
        failures.push(format!(
            "FT4 segment roles are not Base + N Delta: {:?}",
            search_lsm.text_segment_roles
        ));
    }
    if search_lsm.text_segment_stats.first() != Some(&"absolute")
        || search_lsm.text_segment_stats.len() != search_lsm.total_segments_per_family
        || search_lsm
            .text_segment_stats
            .iter()
            .skip(1)
            .any(|model| *model != "delta")
    {
        failures.push(format!(
            "FT4 statistics are not Absolute Base + signed Delta: {:?}",
            search_lsm.text_segment_stats
        ));
    }
    if search_lsm.vector_segment_roles.first() != Some(&"base")
        || search_lsm.vector_segment_roles.len() != search_lsm.total_segments_per_family
        || search_lsm
            .vector_segment_roles
            .iter()
            .skip(1)
            .any(|role| *role != "delta")
    {
        failures.push(format!(
            "vector segment roles are not clustered V5 Base + N VG6 Delta: {:?}",
            search_lsm.vector_segment_roles
        ));
    }
    if search_lsm.vector_segment_stats.first() != Some(&"absolute")
        || search_lsm.vector_segment_stats.len() != search_lsm.total_segments_per_family
        || search_lsm
            .vector_segment_stats
            .iter()
            .skip(1)
            .any(|model| *model != "delta")
    {
        failures.push(format!(
            "vector statistics are not Absolute V5 Base + signed VG6 Delta: {:?}",
            search_lsm.vector_segment_stats
        ));
    }
    if !search_lsm.api_limitations.is_empty() {
        failures.push(format!(
            "object-native Search-LSM still declares API limitations: {:?}",
            search_lsm
                .api_limitations
                .iter()
                .map(|limitation| limitation.component)
                .collect::<Vec<_>>()
        ));
    }
    if search_lsm.total_segments_per_family != search_lsm.delta_segments.saturating_add(1) {
        failures.push(format!(
            "Search-LSM segment total {} != delta count {} + seed",
            search_lsm.total_segments_per_family, search_lsm.delta_segments
        ));
    }
    if !graph.parity_exact {
        failures.push("paged forward/inverse graph parity failed".to_owned());
    }
    if graph
        .forward
        .operations
        .iter()
        .chain(&graph.inverse.operations)
        .any(|operation| {
            operation.no_cache.eager_body_reads != 0
                || operation.sized_cache_cold.eager_body_reads != 0
                || operation.sized_cache_warm.eager_body_reads != 0
        })
    {
        failures.push("paged graph query fell back to an eager full-body read".to_owned());
    }
    if !graph.cache_modes.explicit_zero_and_sized_available {
        failures.push("paged graph did not exercise explicit zero/sized cache modes".to_owned());
    }
    for direction in [&graph.forward, &graph.inverse] {
        for operation in &direction.operations {
            if operation.no_cache.fetch_requests == 0
                || operation.sized_cache_cold.fetch_requests == 0
            {
                failures.push(format!(
                    "paged graph {} {} did not observe physical GETs in both cold modes",
                    direction.direction, operation.operation,
                ));
            }
            if operation.sized_cache_warm.fetched_bytes != 0 {
                failures.push(format!(
                    "paged graph {} {} warm pass fetched {} backing-store bytes",
                    direction.direction,
                    operation.operation,
                    operation.sized_cache_warm.fetched_bytes,
                ));
            }
        }
    }
    if let Some(maximum) = thresholds.max_cold_bytes_ratio {
        if graph.max_cold_artifact_ratio > maximum {
            failures.push(format!(
                "paged graph cold physical-byte ratio {:.6} > {maximum:.6}",
                graph.max_cold_artifact_ratio
            ));
        }
    }
    if let Some(maximum) = thresholds.max_reader_metadata_ratio {
        for direction in [&graph.forward, &graph.inverse] {
            if direction.resident_metadata_artifact_ratio > maximum {
                failures.push(format!(
                    "paged graph {} metadata ratio {:.6} > {maximum:.6}",
                    direction.direction, direction.resident_metadata_artifact_ratio
                ));
            }
        }
    }
    if let Some(maximum) = thresholds.max_rss_bytes {
        match rss.process_high_water_bytes {
            Some(actual) if actual > maximum => failures.push(format!(
                "process high-water RSS {actual} bytes > {maximum} bytes"
            )),
            None => failures.push(
                "RSS gate requested, but process RSS is unavailable on this platform".to_owned(),
            ),
            _ => {}
        }
    }
    failures
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn percentile(sorted: &[u64], percentile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((percentile / 100.0) * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}

#[derive(Debug)]
struct ProcessRss {
    current: Option<u64>,
    high_water: Option<u64>,
}

#[cfg(target_os = "linux")]
fn process_rss() -> Option<ProcessRss> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let parse = |name: &str| {
        status.lines().find_map(|line| {
            let raw = line.strip_prefix(name)?;
            let kib = raw
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())?;
            kib.checked_mul(1024)
        })
    };
    Some(ProcessRss {
        current: parse("VmRSS:"),
        high_water: parse("VmHWM:"),
    })
}

#[cfg(not(target_os = "linux"))]
fn process_rss() -> Option<ProcessRss> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smoke_config() -> ObjectNativeConfig {
        ObjectNativeConfig {
            vectors: 256,
            documents: 256,
            dim: 16,
            queries: 3,
            k: 5,
            clusters: 4,
            spread: 0.12,
            filter_buckets: 4,
            page_rows: 32,
            branch_factor: 4,
            nprobe: 4,
            max_nprobe: 16,
            rerank_factor: 8,
            cache_bytes: 2 * 1024 * 1024,
            build_memory_bytes: 2 * 1024 * 1024,
            range_latency_ms: 0,
            delta_segments: 2,
            graph_keys: 64,
            graph_high_degree: 128,
            seed: 42,
            gates: GateThresholds::default(),
        }
    }

    #[tokio::test]
    async fn small_deterministic_corpus_exercises_both_range_formats() {
        let report = run(smoke_config()).await.expect("object-native smoke");
        assert_eq!(report.format, REPORT_FORMAT);
        assert_eq!(report.vectors.len(), 3);
        assert!(report
            .vectors
            .iter()
            .all(|metric| metric.selective_filter.native_filter_applied));
        assert!(report
            .vectors
            .iter()
            .all(|metric| metric.reader_open_io.fetched_bytes < metric.artifact_bytes));
        assert!(report
            .text
            .parity
            .iter()
            .all(|query| query.node_ids_equal && query.scores_equal));
        assert!(report
            .text
            .parity
            .iter()
            .all(|query| query.result_count > 0));
        assert!(report.text.reader_open_io.fetched_bytes < report.text.artifact_bytes);
        assert_eq!(report.search_lsm.total_segments_per_family, 3);
        assert_eq!(
            report.search_lsm.text_segment_roles,
            vec!["base", "delta", "delta"]
        );
        assert_eq!(
            report.search_lsm.vector_segment_roles,
            vec!["base", "delta", "delta"]
        );
        assert_eq!(
            report.search_lsm.vector_segment_stats,
            vec!["absolute", "delta", "delta"]
        );
        assert_eq!(
            report.search_lsm.text_segment_stats,
            vec!["absolute", "delta", "delta"]
        );
        assert!(report.search_lsm.v5_base_contract.role_is_base);
        assert!(report.search_lsm.v5_base_contract.format_is_v5);
        assert!(report.search_lsm.v5_base_contract.live_count_is_absolute);
        assert!(report.search_lsm.v5_base_contract.suppress_count_is_zero);
        assert!(report.search_lsm.v5_base_contract.native_footer_validated);
        assert!(
            report
                .search_lsm
                .v5_base_contract
                .legacy_manifest_binding_nonzero
        );
        assert_eq!(
            report.search_lsm.v5_base_contract.point_count,
            Some(report.config.vectors as u64)
        );
        assert!(report.search_lsm.v5_base_contract.clustered_page_count > 0);
        assert!(
            report
                .search_lsm
                .v5_base_contract
                .bounded_builder_peak_logical_bytes
                <= report.config.build_memory_bytes
        );
        assert!(report.search_lsm.ft4_base_contract.role_is_base);
        assert!(report.search_lsm.ft4_base_contract.doc_count_is_absolute);
        assert!(report.search_lsm.ft4_base_contract.total_len_is_absolute);
        assert!(report.search_lsm.ft4_base_contract.suppress_count_is_zero);
        assert_eq!(
            report.search_lsm.ft4_base_contract.document_count,
            Some(report.config.documents as u64)
        );
        assert_eq!(
            report.search_lsm.vector_serving_options.nprobe,
            report.config.nprobe
        );
        assert_eq!(
            report.search_lsm.vector_serving_ann.unfiltered.result_model,
            "serving_ann_vs_exact_oracle"
        );
        assert!(
            report
                .search_lsm
                .vector_serving_ann
                .unfiltered
                .returned_winners_valid
        );
        assert!(
            report
                .search_lsm
                .vector_serving_ann
                .unfiltered
                .returned_scores_exact
        );
        assert!(
            report
                .search_lsm
                .vector_serving_ann
                .unfiltered
                .cache_results_identical
        );
        assert!((0.0..=1.0).contains(&report.search_lsm.vector_serving_ann.unfiltered.recall_at_k));
        assert!(
            report
                .search_lsm
                .vector_serving_ann
                .active_filter
                .native_filter_applied
        );
        assert!(
            report
                .search_lsm
                .vector_exact_shadow
                .unfiltered
                .node_ids_exact
        );
        assert!(
            report
                .search_lsm
                .vector_exact_shadow
                .unfiltered
                .scores_exact
        );
        assert!(
            report
                .search_lsm
                .vector_exact_shadow
                .active_filter
                .node_ids_exact
        );
        assert!(
            report
                .search_lsm
                .vector_exact_shadow
                .active_filter
                .scores_exact
        );
        assert!(report.search_lsm.vector_exact_shadow.active_filter_applied);
        assert!(report.search_lsm.text.unfiltered.parity.node_ids_exact);
        assert!(report.search_lsm.text.unfiltered.parity.scores_exact);
        assert!(report.search_lsm.text.active_filter.native_filter_applied);
        assert!(report.search_lsm.api_limitations.is_empty());
        assert!(report.graph.parity_exact);
        assert!(report.graph.forward.range_complete);
        assert!(report.graph.inverse.range_complete);
        assert!(report.graph.api_limitations.is_empty());
        assert!(report
            .graph
            .forward
            .operations
            .iter()
            .chain(&report.graph.inverse.operations)
            .all(|operation| operation.no_cache.fetch_requests > 0
                && operation.sized_cache_cold.fetch_requests > 0
                && operation.sized_cache_warm.fetched_bytes == 0));
        assert!(report.builder_workspace.peak_logical_bytes > 0);
        assert!(report.passed(), "{:?}", report.gates.failures);
    }

    #[tokio::test]
    async fn configured_gate_failure_is_reported_without_hiding_metrics() {
        let mut config = smoke_config();
        config.gates.min_recall_at_k = Some(1.01);
        let report = run(config).await.expect("object-native gated smoke");
        assert!(!report.passed());
        assert!(report
            .gates
            .failures
            .iter()
            .any(|failure| failure.contains("recall")));
        assert_eq!(report.vectors.len(), 3);
    }
}
