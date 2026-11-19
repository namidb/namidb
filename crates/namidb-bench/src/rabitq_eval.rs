//! `rabitq-eval`: build a `NAMIVG05` base from a raw f32 corpus with the
//! external builder and compare the int8 and RaBitQ navigation paths against
//! a supplied exact top-k (overlap, re-rank fan-out, bytes read, latency).
//!
//! Inputs are little-endian files: `--vectors` holds `N × dim` f32 rows (row
//! `j` becomes NodeId `j`), `--queries` `Q × dim` f32 rows and `--truth`
//! `Q × truth_k` u32 row numbers of the exact neighbours.
//!
//! `--synthetic N` instead generates a production-shaped clustered corpus
//! (`--clusters` centers, per-row noise of norm 0.45-0.8, unit rows) on the
//! fly, so a multi-million-row run needs no input file; it writes the
//! queries and their exact top-k (computed in the same pass) to `--queries`
//! and `--truth`.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use namidb_storage::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
use namidb_storage::sst::vector::v5::external::{
    VectorV5ExternalBuildConfig, VectorV5ExternalCollector,
};
use namidb_storage::sst::vector::v5::{
    VectorV5BuildOptions, VectorV5RangeSource, VectorV5Reader, VectorV5SearchOptions,
};
use namidb_storage::Error as StorageError;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct RabitqEvalConfig {
    pub vectors: Option<PathBuf>,
    pub base: PathBuf,
    pub dim: usize,
    pub metric: String,
    pub queries: PathBuf,
    pub truth: PathBuf,
    pub truth_k: usize,
    pub k: usize,
    pub efs: Vec<usize>,
    pub max_queries: usize,
    pub spool: PathBuf,
    pub build_memory_bytes: usize,
    pub latency_ms: u64,
    pub bandwidth_mib_per_sec: f64,
    pub cold_queries: usize,
    pub epsilons: Vec<f32>,
    pub synthetic: Option<usize>,
    pub clusters: usize,
    pub seed: u64,
}

#[derive(Debug, Serialize)]
pub struct BuildReport {
    #[serde(skip)]
    pub base: PathBuf,
    pub rows: u64,
    pub seconds: f64,
    pub artifact_bytes: u64,
    pub scratch_bytes_written: u64,
    pub rabitq_section_bytes: u64,
    pub page_count: u64,
    pub peak_logical_memory_bytes: usize,
    pub vm_hwm_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct PathReport {
    pub overlap_at_k: f64,
    pub reranked_rows_mean: f64,
    pub reranked_rows_p95: usize,
    pub warm_us_p50: u64,
    pub warm_us_p95: u64,
    pub cold_ms_p50: f64,
    pub cold_ms_p95: f64,
    pub cold_bytes_mean: f64,
    pub cold_requests_mean: f64,
    /// Bytes the warm cache holds after every query ran: what this path
    /// keeps resident to stay warm (code or navigation blocks plus re-rank
    /// rows touched).
    pub warm_block_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct EfReport {
    pub ef: usize,
    pub epsilon: f32,
    pub nprobe: usize,
    pub int8: PathReport,
    pub rabitq: PathReport,
}

#[derive(Debug, Serialize)]
pub struct RabitqEvalReport {
    pub build: Option<BuildReport>,
    pub base_bytes: u64,
    pub page_count: usize,
    pub point_count: u64,
    pub rabitq_padded_dim: Option<usize>,
    pub rabitq_code_block_bytes: Option<u64>,
    pub rabitq_section_bytes: Option<u64>,
    pub reader_metadata_bytes: usize,
    pub queries: usize,
    pub results: Vec<EfReport>,
}

fn metric(name: &str) -> Result<VectorMetric> {
    Ok(match name {
        "cosine" => VectorMetric::Cosine,
        "dot" => VectorMetric::Dot,
        "euclidean" => VectorMetric::Euclidean,
        other => bail!("unknown metric {other}"),
    })
}

fn node_id(row: u32) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[12..].copy_from_slice(&row.to_be_bytes());
    id
}

fn row_of(id: &[u8; 16]) -> u32 {
    u32::from_be_bytes(id[12..].try_into().expect("4 bytes"))
}

fn vm_hwm() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

fn read_f32_rows(path: &Path, dim: usize, limit: usize) -> Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let rows = bytes.len() / (dim * 4);
    Ok(bytes
        .chunks_exact(dim * 4)
        .take(limit.min(rows))
        .map(|row| {
            row.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().expect("4")))
                .collect()
        })
        .collect())
}

/// Deterministic production-shaped generator (see the module docs).
struct Synth {
    dim: usize,
    seed: u64,
    centers: Vec<Vec<f32>>,
    /// Cumulative cluster weights (Zipf-like, exponent 0.5).
    cumulative: Vec<f64>,
}

impl Synth {
    fn new(dim: usize, clusters: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let centers = (0..clusters)
            .map(|_| {
                let mut c = gaussian(&mut rng, dim);
                normalize(&mut c);
                c
            })
            .collect();
        let mut total = 0.0;
        let cumulative = (0..clusters)
            .map(|i| {
                total += 1.0 / ((i + 1) as f64).sqrt();
                total
            })
            .collect();
        Self {
            dim,
            seed,
            centers,
            cumulative,
        }
    }

    /// Row `index` of stream `stream` (0 = corpus, 1 = queries).
    fn row(&self, stream: u64, index: u64, out: &mut Vec<f32>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
            self.seed
                ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03),
        );
        let total = *self.cumulative.last().expect("clusters");
        let pick = rng.gen_range(0.0..total);
        let cluster = self
            .cumulative
            .partition_point(|w| *w < pick)
            .min(self.centers.len() - 1);
        let spread = rng.gen_range(0.45..0.8f32) / (self.dim as f32).sqrt();
        out.clear();
        out.extend_from_slice(&self.centers[cluster]);
        for (value, noise) in out.iter_mut().zip(gaussian(&mut rng, self.dim)) {
            *value += noise * spread;
        }
        normalize(out);
    }
}

fn gaussian(rng: &mut rand_chacha::ChaCha8Rng, dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut out = Vec::with_capacity(dim + 1);
    while out.len() < dim {
        let u: f64 = rng.gen_range(1e-300..1.0);
        let v: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let r = (-2.0 * u.ln()).sqrt();
        out.push((r * v.cos()) as f32);
        out.push((r * v.sin()) as f32);
    }
    out.truncate(dim);
    out
}

fn normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|v| *v as f64 * *v as f64)
        .sum::<f64>()
        .sqrt() as f32;
    if norm > 0.0 {
        vector.iter_mut().for_each(|v| *v /= norm);
    }
}

/// Generated queries and their exact top-k row numbers.
type QueriesAndTruth = (Vec<Vec<f32>>, Vec<Vec<u32>>);

/// Generate the corpus into `collector` (in parallel chunks) and return the
/// queries with their exact top-`truth_k` (cosine on unit rows = dot).
fn synthesize(
    config: &RabitqEvalConfig,
    rows: usize,
    collector: &mut VectorV5ExternalCollector,
) -> Result<QueriesAndTruth> {
    let synth = Synth::new(config.dim, config.clusters, config.seed);
    let queries: Vec<Vec<f32>> = (0..config.max_queries as u64)
        .map(|index| {
            let mut q = Vec::new();
            synth.row(1, index, &mut q);
            q
        })
        .collect();
    let truth_k = config.truth_k;
    // Per query: (score, row) min-heap of the best `truth_k`.
    type Heap = std::collections::BinaryHeap<std::cmp::Reverse<(OrdF32, u32)>>;
    let mut heaps: Vec<Heap> = vec![Heap::new(); queries.len()];
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(6);
    let chunk = 16_384usize;
    let filters = std::collections::BTreeMap::new();
    let mut start = 0usize;
    while start < rows {
        let end = (start + chunk * threads).min(rows);
        let parts: Vec<(Vec<Vec<f32>>, Vec<Heap>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (start..end)
                .step_by(chunk)
                .map(|first| {
                    let synth = &synth;
                    let queries = &queries;
                    scope.spawn(move || {
                        let last = (first + chunk).min(end);
                        let mut block = Vec::with_capacity(last - first);
                        let mut local: Vec<Heap> = vec![Heap::new(); queries.len()];
                        for index in first..last {
                            let mut row = Vec::with_capacity(synth.dim);
                            synth.row(0, index as u64, &mut row);
                            for (heap, query) in local.iter_mut().zip(queries) {
                                let score: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
                                offer(heap, truth_k, score, index as u32);
                            }
                            block.push(row);
                        }
                        (block, local)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("generator thread"))
                .collect()
        });
        let mut index = start as u32;
        for (block, local) in parts {
            for row in block {
                collector
                    .push(node_id(index), &row, &filters)
                    .map_err(|e| anyhow!("push {index}: {e}"))?;
                index += 1;
            }
            for (heap, part) in heaps.iter_mut().zip(local) {
                for std::cmp::Reverse((score, row)) in part {
                    offer(heap, truth_k, score.0, row);
                }
            }
        }
        start = end;
        if start % (1 << 20) < chunk * threads {
            eprintln!("generated {start} rows");
        }
    }
    let truth = heaps
        .into_iter()
        .map(|heap| {
            let mut best: Vec<(OrdF32, u32)> = heap.into_iter().map(|r| r.0).collect();
            best.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            best.into_iter().map(|(_, row)| row).collect()
        })
        .collect();
    Ok((queries, truth))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn offer(
    heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(OrdF32, u32)>>,
    k: usize,
    score: f32,
    row: u32,
) {
    if heap.len() < k {
        heap.push(std::cmp::Reverse((OrdF32(score), row)));
    } else if heap.peek().is_some_and(|worst| OrdF32(score) > worst.0 .0) {
        heap.pop();
        heap.push(std::cmp::Reverse((OrdF32(score), row)));
    }
}

fn write_f32_rows(path: &Path, rows: &[Vec<f32>]) -> Result<()> {
    let mut bytes = Vec::new();
    for row in rows {
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn write_u32_rows(path: &Path, rows: &[Vec<u32>]) -> Result<()> {
    let mut bytes = Vec::new();
    for row in rows {
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn build(config: &RabitqEvalConfig, desc: &VectorIndexDescriptor) -> Result<BuildReport> {
    let started = Instant::now();
    std::fs::create_dir_all(&config.spool)?;
    let mut collector = VectorV5ExternalCollector::new_base(VectorV5ExternalBuildConfig {
        scratch_dir: config.spool.clone(),
        memory_budget_bytes: config.build_memory_bytes,
        quantile_sample_rows: 16_384,
    })
    .map_err(|e| anyhow!("collector: {e}"))?;
    let mut rows = 0u32;
    if let Some(count) = config.synthetic {
        let (queries, truth) = synthesize(config, count, &mut collector)?;
        write_f32_rows(&config.queries, &queries)?;
        write_u32_rows(&config.truth, &truth)?;
        rows = count as u32;
    }
    let vectors = config.vectors.clone().unwrap_or_default();
    let mut reader: Box<dyn Read> = if config.synthetic.is_some() {
        Box::new(std::io::empty())
    } else {
        Box::new(BufReader::with_capacity(8 << 20, File::open(&vectors)?))
    };
    let mut row = vec![0u8; config.dim * 4];
    let mut vector = vec![0f32; config.dim];
    let filters = std::collections::BTreeMap::new();
    loop {
        match reader.read_exact(&mut row) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        }
        for (dst, src) in vector.iter_mut().zip(row.chunks_exact(4)) {
            *dst = f32::from_le_bytes(src.try_into().expect("4"));
        }
        collector
            .push(node_id(rows), &vector, &filters)
            .map_err(|e| anyhow!("push {rows}: {e}"))?;
        rows += 1;
    }
    let mut artifact = collector
        .finish(desc, VectorV5BuildOptions::default())
        .map_err(|e| anyhow!("finish: {e}"))?
        .ok_or_else(|| anyhow!("no artifact"))?;
    artifact.rewind().map_err(|e| anyhow!("rewind: {e}"))?;
    let metrics = artifact.metrics;
    let base = if config.base.as_os_str() == "-" {
        // Evaluate the unlinked build artifact in place (no second copy on
        // disk); it lives until the process exits.
        use std::os::fd::AsRawFd;
        let path = PathBuf::from(format!("/proc/self/fd/{}", artifact.file.as_raw_fd()));
        std::mem::forget(artifact.file);
        path
    } else {
        let mut out = File::create(&config.base)?;
        std::io::copy(&mut artifact.file, &mut out)?;
        out.sync_all()?;
        config.base.clone()
    };
    Ok(BuildReport {
        rows: rows as u64,
        seconds: started.elapsed().as_secs_f64(),
        artifact_bytes: artifact.len,
        scratch_bytes_written: metrics.scratch_bytes_written,
        rabitq_section_bytes: metrics.rabitq_section_bytes,
        page_count: metrics.page_count,
        peak_logical_memory_bytes: metrics.peak_logical_memory_bytes,
        vm_hwm_bytes: vm_hwm(),
        base,
    })
}

/// File-backed range source with an optional exact-range cache and an
/// optional simulated object store (one round trip per batch call plus a
/// bandwidth term) for cache misses.
#[derive(Debug)]
struct EvalSource {
    file: File,
    cache: Option<Mutex<HashMap<(u64, u64), Bytes>>>,
    latency: Duration,
    bytes_per_sec: f64,
    bytes: AtomicU64,
    requests: AtomicU64,
}

impl EvalSource {
    fn new(path: &Path, cached: bool, latency_ms: u64, mib_per_sec: f64) -> Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            cache: cached.then(|| Mutex::new(HashMap::new())),
            latency: Duration::from_millis(latency_ms),
            bytes_per_sec: mib_per_sec * 1024.0 * 1024.0,
            bytes: AtomicU64::new(0),
            requests: AtomicU64::new(0),
        })
    }

    fn cached_bytes(&self) -> u64 {
        self.cache.as_ref().map_or(0, |cache| {
            cache
                .lock()
                .expect("cache")
                .values()
                .map(|v| v.len() as u64)
                .sum()
        })
    }

    async fn fetch(&self, ranges: &[Range<u64>]) -> std::result::Result<Vec<Bytes>, StorageError> {
        let mut out = vec![Bytes::new(); ranges.len()];
        let mut missing = Vec::new();
        if let Some(cache) = &self.cache {
            let cache = cache.lock().expect("cache");
            for (slot, range) in ranges.iter().enumerate() {
                match cache.get(&(range.start, range.end)) {
                    Some(bytes) => out[slot] = bytes.clone(),
                    None => missing.push(slot),
                }
            }
        } else {
            missing.extend(0..ranges.len());
        }
        if missing.is_empty() {
            return Ok(out);
        }
        let mut moved = 0u64;
        for &slot in &missing {
            let range = &ranges[slot];
            let mut buf = vec![0u8; (range.end - range.start) as usize];
            self.file
                .read_exact_at(&mut buf, range.start)
                .map_err(|e| StorageError::invariant(format!("eval read: {e}")))?;
            moved += buf.len() as u64;
            out[slot] = Bytes::from(buf);
        }
        self.bytes.fetch_add(moved, Ordering::Relaxed);
        self.requests
            .fetch_add(missing.len() as u64, Ordering::Relaxed);
        if !self.latency.is_zero() || self.bytes_per_sec > 0.0 {
            let transfer = if self.bytes_per_sec > 0.0 {
                Duration::from_secs_f64(moved as f64 / self.bytes_per_sec)
            } else {
                Duration::ZERO
            };
            tokio::time::sleep(self.latency + transfer).await;
        }
        if let Some(cache) = &self.cache {
            let mut cache = cache.lock().expect("cache");
            for &slot in &missing {
                cache.insert((ranges[slot].start, ranges[slot].end), out[slot].clone());
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl VectorV5RangeSource for EvalSource {
    async fn read_range(&self, range: Range<u64>) -> std::result::Result<Bytes, StorageError> {
        Ok(self.fetch(std::slice::from_ref(&range)).await?.remove(0))
    }

    async fn read_ranges(
        &self,
        ranges: &[Range<u64>],
    ) -> std::result::Result<Vec<Bytes>, StorageError> {
        self.fetch(ranges).await
    }
}

/// The engine's probe policy for a clustered base (`read.rs`).
fn options_for(pages: usize, ef: usize) -> VectorV5SearchOptions {
    let pages = pages.max(1);
    let nprobe = ef.div_ceil(4).max(64).min(pages);
    VectorV5SearchOptions {
        nprobe,
        max_nprobe: nprobe.saturating_mul(8).max(nprobe).min(pages),
        rerank_factor: 8,
    }
}

fn percentile<T: Copy + Ord>(values: &[T], p: f64) -> T {
    let mut sorted = values.to_vec();
    sorted.sort();
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn percentile_f(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

async fn run_path(
    config: &RabitqEvalConfig,
    queries: &[Vec<f32>],
    truth: &[BTreeSet<u32>],
    options: VectorV5SearchOptions,
    rabitq: bool,
) -> Result<PathReport> {
    let len = std::fs::metadata(&config.base)?.len();
    let search = |reader: Arc<VectorV5Reader>, query: Vec<f32>| async move {
        if rabitq {
            reader.search_filter_groups(&query, 10, options, &[]).await
        } else {
            reader
                .search_filter_groups_int8(&query, 10, options, &[])
                .await
        }
    };
    // Warm: one cached source for every query; the first pass fills it.
    let warm_source = Arc::new(EvalSource::new(&config.base, true, 0, 0.0)?);
    let warm = Arc::new(
        VectorV5Reader::open(warm_source.clone(), len)
            .await
            .map_err(|e| anyhow!("open: {e}"))?,
    );
    if rabitq && !warm.has_rabitq() {
        bail!("base has no RaBitQ section");
    }
    let mut overlap = 0usize;
    let mut reranked = Vec::with_capacity(queries.len());
    for (query, truth) in queries.iter().zip(truth) {
        let result = search(warm.clone(), query.clone())
            .await
            .map_err(|e| anyhow!("search: {e}"))?;
        overlap += result
            .hits
            .iter()
            .take(config.k)
            .filter(|(id, _)| truth.contains(&row_of(id)))
            .count();
        reranked.push(result.reranked_rows);
    }
    let mut warm_us = Vec::with_capacity(queries.len());
    for query in queries {
        let started = Instant::now();
        search(warm.clone(), query.clone())
            .await
            .map_err(|e| anyhow!("search: {e}"))?;
        warm_us.push(started.elapsed().as_micros() as u64);
    }
    // Cold: a fresh reader and empty cache per query, simulated store.
    let mut cold_ms = Vec::new();
    let mut cold_bytes = Vec::new();
    let mut cold_requests = Vec::new();
    for query in queries.iter().take(config.cold_queries) {
        let source = Arc::new(EvalSource::new(
            &config.base,
            true,
            config.latency_ms,
            config.bandwidth_mib_per_sec,
        )?);
        let started = Instant::now();
        let reader = Arc::new(
            VectorV5Reader::open(source.clone(), len)
                .await
                .map_err(|e| anyhow!("open: {e}"))?,
        );
        search(reader, query.clone())
            .await
            .map_err(|e| anyhow!("search: {e}"))?;
        cold_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        cold_bytes.push(source.bytes.load(Ordering::Relaxed) as f64);
        cold_requests.push(source.requests.load(Ordering::Relaxed) as f64);
    }
    let n = queries.len().max(1) as f64;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
    Ok(PathReport {
        overlap_at_k: overlap as f64 / (n * config.k as f64),
        reranked_rows_mean: reranked.iter().sum::<usize>() as f64 / n,
        reranked_rows_p95: percentile(&reranked, 0.95),
        warm_us_p50: percentile(&warm_us, 0.5),
        warm_us_p95: percentile(&warm_us, 0.95),
        cold_ms_p50: percentile_f(&cold_ms, 0.5),
        cold_ms_p95: percentile_f(&cold_ms, 0.95),
        cold_bytes_mean: mean(&cold_bytes),
        cold_requests_mean: mean(&cold_requests),
        warm_block_bytes: warm_source.cached_bytes(),
    })
}

pub async fn run(config: RabitqEvalConfig) -> Result<RabitqEvalReport> {
    let metric = metric(&config.metric)?;
    let desc = VectorIndexDescriptor {
        name: "rabitq_eval".into(),
        label: "Doc".into(),
        property: "embedding".into(),
        dim: config.dim as u32,
        metric,
        r: 64,
        l_build: 128,
        alpha: 1.2,
        quantization: VectorQuantization::None,
    };
    let mut config = config;
    let build = if config.vectors.is_some() || config.synthetic.is_some() {
        let report = build(&config, &desc)?;
        eprintln!("build: {}", serde_json::to_string(&report)?);
        config.base = report.base.clone();
        Some(report)
    } else {
        None
    };
    let base_bytes = std::fs::metadata(&config.base)?.len();
    let probe = VectorV5Reader::open(
        Arc::new(EvalSource::new(&config.base, false, 0, 0.0)?),
        base_bytes,
    )
    .await
    .map_err(|e| anyhow!("open: {e}"))?;
    let queries = read_f32_rows(&config.queries, config.dim, config.max_queries)?;
    let truth_bytes = std::fs::read(&config.truth)?;
    let truth: Vec<BTreeSet<u32>> = truth_bytes
        .chunks_exact(config.truth_k * 4)
        .take(queries.len())
        .map(|row| {
            row.chunks_exact(4)
                .take(config.k)
                .map(|v| u32::from_le_bytes(v.try_into().expect("4")))
                .collect()
        })
        .collect();
    let stats = probe.rabitq_stats();
    let mut results = Vec::new();
    for &ef in &config.efs {
        let options = options_for(probe.page_count(), ef);
        let int8 = run_path(&config, &queries, &truth, options, false).await?;
        for &epsilon in &config.epsilons {
            namidb_storage::sst::vector::v5::override_rabitq_epsilon_for_benchmarks(epsilon);
            let rabitq = if probe.has_rabitq() {
                run_path(&config, &queries, &truth, options, true).await?
            } else {
                PathReport::default()
            };
            eprintln!(
            "ef={ef} eps={epsilon} nprobe={} int8 overlap={:.4} rerank={:.0} warm_p50={}us | rabitq overlap={:.4} rerank={:.0} (p95 {}) warm_p50={}us",
            options.nprobe,
            int8.overlap_at_k,
            int8.reranked_rows_mean,
            int8.warm_us_p50,
            rabitq.overlap_at_k,
            rabitq.reranked_rows_mean,
            rabitq.reranked_rows_p95,
            rabitq.warm_us_p50
        );
            results.push(EfReport {
                ef,
                epsilon,
                nprobe: options.nprobe,
                int8: int8.clone(),
                rabitq,
            });
        }
    }
    Ok(RabitqEvalReport {
        build,
        base_bytes,
        page_count: probe.page_count(),
        point_count: probe.point_count(),
        rabitq_padded_dim: stats.map(|s| s.padded_dim),
        rabitq_code_block_bytes: stats.map(|s| s.code_block_bytes),
        rabitq_section_bytes: stats.map(|s| s.section_bytes),
        reader_metadata_bytes: probe.resident_metadata_bytes(),
        queries: queries.len(),
        results,
    })
}
