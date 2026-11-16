//! `serve` and `http-load`: write throughput and vector search over HTTP on a
//! simulated object store.
//!
//! `serve` boots the real `namidb-server` (single- or multi-tenant) in this
//! process, with every store it opens wrapped in a [`LatencyStore`] that adds
//! a per-request latency (default 20-40 ms, the range a regional S3 bucket
//! shows) and counts requests. Run it inside a memory-limited cgroup to
//! reproduce a container size, e.g.
//!
//! ```text
//! systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 \
//!   env MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072 \
//!   namidb-bench serve --store 'file:///tmp/nb?ns=bench' --memory-max-bytes auto
//! ```
//!
//! The two `MALLOC_*` variables are the ones the official container image
//! sets; without them glibc keeps one arena per thread and the resident size
//! the memory governor admits against is not what a container would see.
//!
//! `http-load` is the client. It runs outside the server's cgroup, so the
//! ground-truth vectors it keeps for recall do not count against the server's
//! memory. Scenarios:
//!
//! - `field-report`: the request mix of a real application: one 500-row
//!   `UNWIND ... MERGE` of ~3 KB nodes, the same request again, 400
//!   relationships with properties, a point read, `search.bm25`, 60 nodes
//!   with a 1024-dim vector, and one 150-vector request (~3 MB of JSON).
//! - `bulk`: sustained `UNWIND ... MERGE` load, reported as nodes/s.
//! - `vectors`: sustained vector load, reported as vectors/s.
//! - `vector-search`: recall@k against brute-force ground truth plus p50/p95
//!   latency, on the vectors `vectors` loaded (same `--seed`, `--clusters`,
//!   `--dim`, `--count`).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::{json, Value};

use crate::latency_store::{LatencyModel, LatencyStore, StoreCounters, StoreCountersSnapshot};

/// Options of `serve`.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub store: String,
    pub listen: SocketAddr,
    pub latency: LatencyModel,
    pub multi_tenant: bool,
    pub group_commit_window: Duration,
    pub memory_max_bytes: String,
    pub flush_interval: Duration,
    pub compaction_interval: Duration,
    pub counters_file: Option<PathBuf>,
}

/// Boot the real server against the simulated store. Never returns while the
/// server runs.
pub async fn serve(config: ServeConfig) -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();
    let counters = Arc::new(StoreCounters::default());
    {
        let counters = Arc::clone(&counters);
        let model = config.latency;
        namidb_storage::uri::set_store_layer(Some(Arc::new(move |inner| {
            Arc::new(LatencyStore::new(inner, model, Arc::clone(&counters))) as _
        })));
    }
    if let Some(path) = config.counters_file.clone() {
        let counters = Arc::clone(&counters);
        tokio::spawn(async move {
            loop {
                let snapshot = counters.snapshot();
                if let Ok(body) = serde_json::to_vec(&snapshot) {
                    let tmp = path.with_extension("tmp");
                    if std::fs::write(&tmp, body).is_ok() {
                        let _ = std::fs::rename(&tmp, &path);
                    }
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
    }
    let memory_max_bytes =
        namidb_server::memory::resolve_memory_max_bytes(&config.memory_max_bytes)
            .map_err(anyhow::Error::msg)?;
    let query_memory_max_bytes =
        namidb_server::memory::resolve_query_memory_max_bytes("auto", memory_max_bytes)
            .map_err(anyhow::Error::msg)?;
    // The server's own CLI defaults, except the knobs this harness exposes.
    let server = namidb_server::Config {
        store_uri: config.store,
        listen: config.listen,
        auth_token: None,
        auth_tokens_file: None,
        auth_tokens_reload_interval: Duration::from_secs(10),
        no_auth: true,
        group_commit_window: config.group_commit_window,
        backup_target_uri: None,
        flush_interval: config.flush_interval,
        compaction_interval: config.compaction_interval,
        sweep_min_age: Duration::from_secs(24 * 3600),
        sweep_delete: true,
        restore_point_retention: Duration::ZERO,
        bolt_listen: None,
        bolt_max_message_bytes: 16 * 1024 * 1024,
        bolt_tx_timeout: Duration::from_secs(30),
        query_timeout: Duration::from_secs(30),
        write_timeout: Duration::from_secs(60),
        query_row_cap: 1_000_000,
        compaction_l0_trigger: 8,
        write_stall_l0: 24,
        write_stall_delay: Duration::from_millis(50),
        memtable_flush_bytes: 64 * 1024 * 1024,
        memtable_stall_bytes: 256 * 1024 * 1024,
        writer_lock_timeout: Duration::from_secs(30),
        http_max_body_bytes: namidb_server::DEFAULT_HTTP_CYPHER_BODY_LIMIT_BYTES,
        http_request_timeout: None,
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::from_secs(1),
        multi_tenant: config.multi_tenant,
        default_namespace: "default".into(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
        fence_policy: namidb_server::recovery::FencePolicy::Steal,
    };
    namidb_server::run_with_memory_limits(server, memory_max_bytes, query_memory_max_bytes).await
}

/// Which `http-load` workload to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Scenario {
    FieldReport,
    Bulk,
    Vectors,
    VectorSearch,
}

/// Options of `http-load`.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub url: String,
    pub namespace: Option<String>,
    pub scenario: Scenario,
    pub count: usize,
    pub batch: usize,
    pub concurrency: usize,
    pub node_bytes: usize,
    pub dim: usize,
    pub clusters: usize,
    pub queries: usize,
    pub k: usize,
    pub ef: usize,
    pub seed: u64,
    pub counters_file: Option<PathBuf>,
    pub setup: bool,
}

#[derive(Debug, Serialize)]
pub struct Step {
    pub name: String,
    pub status: u16,
    pub secs: f64,
    pub request_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<StoreCountersSnapshot>,
}

#[derive(Debug, Serialize, Default)]
pub struct LoadReport {
    pub scenario: String,
    pub steps: Vec<Step>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
}

struct Client {
    http: reqwest::Client,
    endpoint: String,
    counters_file: Option<PathBuf>,
}

struct Reply {
    status: u16,
    body: Value,
    request_bytes: usize,
}

impl Client {
    fn new(config: &LoadConfig) -> Result<Self> {
        let base = config.url.trim_end_matches('/');
        let endpoint = match &config.namespace {
            Some(ns) => format!("{base}/{ns}/v0/cypher"),
            None => format!("{base}/v0/cypher"),
        };
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .pool_max_idle_per_host(64)
            .build()?;
        Ok(Self {
            http,
            endpoint,
            counters_file: config.counters_file.clone(),
        })
    }

    async fn run(&self, query: &str, params: Value) -> Result<Reply> {
        let body = serde_json::to_vec(&json!({ "query": query, "params": params }))?;
        let request_bytes = body.len();
        let response = self
            .http
            .post(&self.endpoint)
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await?;
        let status = response.status().as_u16();
        let text = response.text().await?;
        let body = serde_json::from_str(&text).unwrap_or(Value::String(text));
        Ok(Reply {
            status,
            body,
            request_bytes,
        })
    }

    async fn ok(&self, query: &str, params: Value) -> Result<Value> {
        let reply = self.run(query, params).await?;
        if reply.status != 200 {
            bail!("HTTP {} for `{query}`: {}", reply.status, reply.body);
        }
        Ok(reply.body)
    }

    fn counters(&self) -> Option<StoreCountersSnapshot> {
        let path = self.counters_file.as_ref()?;
        let body = std::fs::read(path).ok()?;
        serde_json::from_slice(&body).ok()
    }

    async fn settle_counters(&self) -> Option<StoreCountersSnapshot> {
        // The server rewrites the file every 100 ms.
        if self.counters_file.is_some() {
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        self.counters()
    }

    /// Time one request and record it as a step.
    async fn step(&self, report: &mut LoadReport, name: &str, query: &str, params: Value) -> Reply {
        let before = self.settle_counters().await;
        let started = Instant::now();
        let reply = match self.run(query, params).await {
            Ok(reply) => reply,
            Err(error) => Reply {
                status: 0,
                body: Value::String(error.to_string()),
                request_bytes: 0,
            },
        };
        let secs = started.elapsed().as_secs_f64();
        let after = self.settle_counters().await;
        let store = match (before, after) {
            (Some(before), Some(after)) => Some(after.since(&before)),
            _ => None,
        };
        let error = (reply.status != 200).then(|| truncate(&reply.body.to_string(), 300));
        eprintln!(
            "{name:<28} HTTP {:<3} {secs:>8.3}s  {:>9} B{}",
            reply.status,
            reply.request_bytes,
            store
                .as_ref()
                .map(|s| format!("  store requests={} {:?}", s.requests, s.by_op))
                .unwrap_or_default()
        );
        report.steps.push(Step {
            name: name.to_string(),
            status: reply.status,
            secs,
            request_bytes: reply.request_bytes,
            error,
            store,
        });
        reply
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut end = n;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

const WORDS: &[&str] = &[
    "graph",
    "storage",
    "object",
    "vector",
    "index",
    "commit",
    "manifest",
    "query",
    "planner",
    "device",
    "call",
    "network",
    "latency",
    "bucket",
    "segment",
    "compaction",
    "snapshot",
    "tenant",
    "cluster",
    "replica",
    "token",
    "embedding",
    "search",
    "ranking",
    "document",
    "record",
];

/// A ~`bytes`-long text body made of dictionary words.
fn body_text(rng: &mut ChaCha8Rng, bytes: usize) -> String {
    let mut out = String::with_capacity(bytes + 16);
    while out.len() < bytes {
        out.push_str(WORDS[rng.gen_range(0..WORDS.len())]);
        out.push(' ');
    }
    out.truncate(bytes);
    out
}

fn doc_row(seed: u64, i: usize, node_bytes: usize) -> Value {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
    // ~3 KB nodes: a long body plus a few scalar properties.
    let body = body_text(&mut rng, node_bytes.saturating_sub(120).max(8));
    json!({
        "id": format!("doc-{i}"),
        "props": {
            "title": format!("Document {i}"),
            "body": body,
            "source": WORDS[i % WORDS.len()],
            "rank": (i % 1000) as i64,
            "score": rng.gen::<f64>(),
            "active": i % 3 != 0,
        }
    })
}

/// Deterministic clustered unit vectors: vector `i` is centroid `i % clusters`
/// plus Gaussian noise, normalised. Real embeddings cluster like this.
pub struct VectorGen {
    dim: usize,
    centroids: Vec<Vec<f32>>,
    seed: u64,
    spread: f32,
}

impl VectorGen {
    pub fn new(dim: usize, clusters: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let centroids = (0..clusters.max(1))
            .map(|_| unit(&mut rng, dim, None, 1.0))
            .collect();
        Self {
            dim,
            centroids,
            seed,
            spread: 0.35,
        }
    }

    pub fn vector(&self, i: usize) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ 0xA5A5_0000 ^ (i as u64) << 1);
        let c = &self.centroids[i % self.centroids.len()];
        unit(&mut rng, self.dim, Some(c), self.spread)
    }

    pub fn query(&self, q: usize) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ 0x5A5A_0000_0000 ^ q as u64);
        let c = &self.centroids[rng.gen_range(0..self.centroids.len())];
        unit(&mut rng, self.dim, Some(c), self.spread)
    }
}

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    // Box-Muller.
    let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn unit(rng: &mut ChaCha8Rng, dim: usize, center: Option<&[f32]>, spread: f32) -> Vec<f32> {
    let noise = spread / (dim as f32).sqrt();
    let mut v: Vec<f32> = (0..dim)
        .map(|d| {
            let g = gaussian(rng);
            match center {
                Some(c) => c[d] + g * noise,
                None => g,
            }
        })
        .collect();
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    for x in &mut v {
        *x /= norm;
    }
    v
}

async fn setup_schema(client: &Client, dim: usize) -> Result<()> {
    for ddl in [
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Doc) REQUIRE d.id IS UNIQUE".to_string(),
        format!(
            "CREATE VECTOR INDEX doc_emb IF NOT EXISTS ON :Doc(embedding) METRIC cosine DIMENSION {dim}"
        ),
        "CREATE FULLTEXT INDEX doc_body IF NOT EXISTS FOR (d:Doc) ON EACH [d.body]".to_string(),
    ] {
        let reply = client.run(&ddl, json!({})).await?;
        if reply.status != 200 {
            eprintln!("setup: `{ddl}` -> HTTP {}: {}", reply.status, reply.body);
        }
    }
    Ok(())
}

const MERGE_DOCS: &str = "UNWIND $rows AS r MERGE (d:Doc {id: r.id}) SET d += r.props";
const MERGE_VECTORS: &str =
    "UNWIND $rows AS r MERGE (d:Doc {id: r.id}) SET d.embedding = r.embedding, d.title = r.title";

fn vector_rows(gen: &VectorGen, range: std::ops::Range<usize>) -> Value {
    Value::Array(
        range
            .map(|i| {
                json!({
                    "id": format!("vec-{i}"),
                    "title": format!("Vector {i}"),
                    "embedding": gen.vector(i),
                })
            })
            .collect(),
    )
}

pub async fn http_load(config: LoadConfig) -> Result<LoadReport> {
    let client = Arc::new(Client::new(&config)?);
    if config.setup {
        setup_schema(&client, config.dim).await?;
    }
    let mut report = LoadReport {
        scenario: format!("{:?}", config.scenario),
        ..LoadReport::default()
    };
    match config.scenario {
        Scenario::FieldReport => field_report(&client, &config, &mut report).await?,
        Scenario::Bulk => {
            let rate = sustained(&client, &config, false).await?;
            report.metrics.insert("nodes_per_sec".into(), rate);
        }
        Scenario::Vectors => {
            let rate = sustained(&client, &config, true).await?;
            report.metrics.insert("vectors_per_sec".into(), rate);
        }
        Scenario::VectorSearch => vector_search(&client, &config, &mut report).await?,
    }
    Ok(report)
}

async fn field_report(client: &Client, config: &LoadConfig, report: &mut LoadReport) -> Result<()> {
    let rows: Vec<Value> = (0..500)
        .map(|i| doc_row(config.seed, i, config.node_bytes))
        .collect();
    client
        .step(
            report,
            "merge_500_nodes",
            MERGE_DOCS,
            json!({ "rows": rows }),
        )
        .await;
    client
        .step(
            report,
            "merge_500_nodes_again",
            MERGE_DOCS,
            json!({ "rows": rows }),
        )
        .await;
    let rels: Vec<Value> = (0..400)
        .map(|i| {
            json!({
                "a": format!("doc-{}", i % 500),
                "b": format!("doc-{}", (i * 7 + 3) % 500),
                "w": i as f64 / 400.0,
                "kind": WORDS[i % WORDS.len()],
            })
        })
        .collect();
    client
        .step(
            report,
            "create_400_relationships",
            "UNWIND $rels AS r MATCH (a:Doc {id: r.a}) MATCH (b:Doc {id: r.b}) \
             CREATE (a)-[:LINKS {weight: r.w, kind: r.kind}]->(b)",
            json!({ "rels": rels }),
        )
        .await;
    client
        .step(
            report,
            "point_read",
            "MATCH (d:Doc {id: $id}) RETURN d.id AS id, d.title AS title",
            json!({ "id": "doc-123" }),
        )
        .await;
    client
        .step(
            report,
            "search_bm25",
            "CALL search.bm25({label: 'Doc', text_property: 'body', query: 'graph latency', k: 10}) \
             YIELD node, score RETURN node.id AS id, score",
            json!({}),
        )
        .await;
    let gen = VectorGen::new(config.dim, config.clusters, config.seed);
    client
        .step(
            report,
            "merge_60_vectors",
            MERGE_VECTORS,
            json!({ "rows": vector_rows(&gen, 0..60) }),
        )
        .await;
    client
        .step(
            report,
            "merge_150_vectors",
            MERGE_VECTORS,
            json!({ "rows": vector_rows(&gen, 60..210) }),
        )
        .await;
    client
        .step(
            report,
            "point_read_after_writes",
            "MATCH (d:Doc {id: $id}) RETURN d.id AS id, d.title AS title",
            json!({ "id": "doc-321" }),
        )
        .await;
    Ok(())
}

/// Load `count` nodes (or vectors) in `batch`-row requests, `concurrency` in
/// flight. Returns rows/s over the whole run and prints a per-10% rate.
async fn sustained(client: &Arc<Client>, config: &LoadConfig, vectors: bool) -> Result<f64> {
    let gen = Arc::new(VectorGen::new(config.dim, config.clusters, config.seed));
    let batches: Vec<std::ops::Range<usize>> = (0..config.count)
        .step_by(config.batch.max(1))
        .map(|start| start..(start + config.batch).min(config.count))
        .collect();
    let queue = Arc::new(std::sync::Mutex::new(batches.into_iter()));
    let done = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let failures = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let retries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let before = client.settle_counters().await;
    let started = Instant::now();
    let mut workers = Vec::new();
    for _ in 0..config.concurrency.max(1) {
        let client = Arc::clone(client);
        let queue = Arc::clone(&queue);
        let done = Arc::clone(&done);
        let failures = Arc::clone(&failures);
        let retries = Arc::clone(&retries);
        let gen = Arc::clone(&gen);
        let seed = config.seed;
        let node_bytes = config.node_bytes;
        workers.push(tokio::spawn(async move {
            loop {
                let next = queue.lock().unwrap().next();
                let Some(range) = next else { break };
                let rows = range.len();
                let (query, params) = if vectors {
                    (MERGE_VECTORS, json!({ "rows": vector_rows(&gen, range) }))
                } else {
                    let rows: Vec<Value> = range.map(|i| doc_row(seed, i, node_bytes)).collect();
                    (MERGE_DOCS, json!({ "rows": rows }))
                };
                // Retry transient rejections like a client library would: a
                // 503 (busy, memory admission) and a transport error (a server
                // that answers 503 before reading a large body resets the
                // upload). Anything else is a failure. Only committed rows
                // count towards the rate.
                let mut attempts = 0;
                let committed = loop {
                    attempts += 1;
                    let retry = match client.run(query, params.clone()).await {
                        Ok(reply) if reply.status == 200 => break true,
                        Ok(reply) if reply.status == 503 => {
                            format!("HTTP 503 {}", truncate(&reply.body.to_string(), 200))
                        }
                        Ok(reply) => {
                            eprintln!(
                                "batch failed: HTTP {} {}",
                                reply.status,
                                truncate(&reply.body.to_string(), 300)
                            );
                            break false;
                        }
                        Err(error) => error.to_string(),
                    };
                    retries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if attempts >= 50 {
                        eprintln!("batch failed after {attempts} attempts: {retry}");
                        break false;
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                };
                if committed {
                    done.fetch_add(rows, std::sync::atomic::Ordering::Relaxed);
                } else {
                    failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }));
    }
    // Progress ticker.
    let ticker = {
        let done = Arc::clone(&done);
        let count = config.count;
        tokio::spawn(async move {
            let mut last = (Instant::now(), 0usize);
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                let now = done.load(std::sync::atomic::Ordering::Relaxed);
                let rate = (now - last.1) as f64 / last.0.elapsed().as_secs_f64();
                eprintln!("  progress {now}/{count}  window {rate:.1} rows/s");
                last = (Instant::now(), now);
            }
        })
    };
    for worker in workers {
        worker.await?;
    }
    ticker.abort();
    let secs = started.elapsed().as_secs_f64();
    let after = client.settle_counters().await;
    let loaded = done.load(std::sync::atomic::Ordering::Relaxed);
    let rate = loaded as f64 / secs;
    eprintln!(
        "{} {loaded} rows in {secs:.1}s = {rate:.1} rows/s (batch {}, concurrency {}, \
         failures {}, retried requests {})",
        if vectors { "vectors" } else { "nodes" },
        config.batch,
        config.concurrency,
        failures.load(std::sync::atomic::Ordering::Relaxed),
        retries.load(std::sync::atomic::Ordering::Relaxed),
    );
    if let (Some(before), Some(after)) = (before, after) {
        let delta = after.since(&before);
        eprintln!(
            "  store requests={} ({:.2}/row) {:?}",
            delta.requests,
            delta.requests as f64 / loaded.max(1) as f64,
            delta.by_op
        );
    }
    if failures.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        bail!("some batches failed");
    }
    Ok(rate)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

async fn vector_search(
    client: &Client,
    config: &LoadConfig,
    report: &mut LoadReport,
) -> Result<()> {
    let gen = Arc::new(VectorGen::new(config.dim, config.clusters, config.seed));
    let queries: Vec<Vec<f32>> = (0..config.queries).map(|q| gen.query(q)).collect();
    // Brute-force ground truth over the whole corpus, in parallel, without
    // keeping the corpus: each thread regenerates its slice.
    eprintln!("computing ground truth over {} vectors...", config.count);
    let gt_started = Instant::now();
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let chunk = config.count.div_ceil(threads);
    let k = config.k;
    let partials: Vec<Vec<Vec<(f32, usize)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let gen = Arc::clone(&gen);
                let queries = &queries;
                let count = config.count;
                scope.spawn(move || {
                    let mut tops: Vec<Vec<(f32, usize)>> = vec![Vec::new(); queries.len()];
                    for i in (t * chunk)..((t + 1) * chunk).min(count) {
                        let v = gen.vector(i);
                        for (q, query) in queries.iter().enumerate() {
                            let score: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                            let top = &mut tops[q];
                            if top.len() < k || score > top[top.len() - 1].0 {
                                top.push((score, i));
                                top.sort_by(|a, b| b.0.total_cmp(&a.0));
                                top.truncate(k);
                            }
                        }
                    }
                    tops
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let truth: Vec<Vec<usize>> = (0..queries.len())
        .map(|q| {
            let mut all: Vec<(f32, usize)> = partials.iter().flat_map(|p| p[q].clone()).collect();
            all.sort_by(|a, b| b.0.total_cmp(&a.0));
            all.truncate(k);
            all.into_iter().map(|(_, i)| i).collect()
        })
        .collect();
    eprintln!("ground truth in {:.1}s", gt_started.elapsed().as_secs_f64());

    let query_text =
        "CALL search.vector({label: 'Doc', property: 'embedding', query: $q, k: $k, ef: $ef}) \
                      YIELD node, score RETURN node.id AS id, score";
    // Warm-up: one pass that is not measured (opens the index, fills caches).
    for q in queries.iter().take(3) {
        let _ = client
            .ok(query_text, json!({ "q": q, "k": k, "ef": config.ef }))
            .await;
    }
    let before = client.settle_counters().await;
    let mut latencies = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0;
    for (q, query) in queries.iter().enumerate() {
        let started = Instant::now();
        let body = client
            .ok(query_text, json!({ "q": query, "k": k, "ef": config.ef }))
            .await?;
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);
        let rows = body["rows"]
            .as_array()
            .ok_or_else(|| anyhow!("no rows in {body}"))?;
        let got: std::collections::HashSet<usize> = rows
            .iter()
            .filter_map(|row| row["id"].as_str())
            .filter_map(|id| id.strip_prefix("vec-"))
            .filter_map(|n| n.parse().ok())
            .collect();
        let hits = truth[q].iter().filter(|i| got.contains(i)).count();
        recall_sum += hits as f64 / k as f64;
    }
    let after = client.settle_counters().await;
    latencies.sort_by(|a, b| a.total_cmp(b));
    let recall = recall_sum / queries.len().max(1) as f64;
    report.metrics.insert("recall_at_k".into(), recall);
    report
        .metrics
        .insert("p50_ms".into(), percentile(&latencies, 50.0));
    report
        .metrics
        .insert("p95_ms".into(), percentile(&latencies, 95.0));
    report
        .metrics
        .insert("p99_ms".into(), percentile(&latencies, 99.0));
    report
        .metrics
        .insert("max_ms".into(), percentile(&latencies, 100.0));
    if let (Some(before), Some(after)) = (before, after) {
        let delta = after.since(&before);
        report.metrics.insert(
            "store_requests_per_query".into(),
            delta.requests as f64 / queries.len().max(1) as f64,
        );
        report.metrics.insert(
            "store_bytes_read_per_query".into(),
            delta.bytes_read as f64 / queries.len().max(1) as f64,
        );
    }
    eprintln!(
        "vector search: recall@{k}={recall:.4} p50={:.1}ms p95={:.1}ms (ef {}, {} queries)",
        percentile(&latencies, 50.0),
        percentile(&latencies, 95.0),
        config.ef,
        queries.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_deterministic_unit_vectors() {
        let gen = VectorGen::new(64, 8, 7);
        let a = gen.vector(5);
        assert_eq!(a, VectorGen::new(64, 8, 7).vector(5));
        let norm: f32 = a.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-4);
        assert_ne!(a, gen.vector(6));
    }

    #[test]
    fn doc_rows_have_the_requested_size() {
        let row = doc_row(1, 2, 3000);
        let len = serde_json::to_string(&row).unwrap().len();
        assert!((2800..3400).contains(&len), "{len}");
    }
}
