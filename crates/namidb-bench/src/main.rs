//! `namidb-bench` CLI — gate harness.
//!
//! Two subcommands:
//!
//! - `generate` — writes a synthetic LDBC-shaped dataset to a directory.
//! - `run` — loads the dataset (or regenerates it inline) and bench-runs
//! each query, printing JSON to stdout.
//!
//! See `bench/README.md` for the full workflow + paired Kuzu runner.

// Cosmetic doc-rendering nit (lazy markdown list continuation); allow it.
#![allow(clippy::doc_lazy_continuation)]

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[cfg(feature = "vector-index")]
mod ann_bench;
mod dataset;
mod graph_shape;
#[cfg(feature = "http-load")]
mod http_load;
#[cfg(feature = "http-load")]
mod latency_store;
mod loader;
#[cfg(feature = "object-native")]
mod object_native;
mod queries;
mod runner;
mod vector_recall;

use dataset::{DatasetConfig, DatasetSizes};
use queries::Query;
use runner::{run_query, BenchOutput, QueryResult, SizesReport};

#[derive(Debug, Parser)]
#[command(
    version,
    author,
    about = "LDBC-shaped synthetic bench for NamiDB (gate)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Generate the synthetic CSV dataset to a directory.
    Generate {
        #[arg(short, long, default_value = "0.1")]
        scale: f64,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Run the bench end-to-end: generate (or reuse), load, time each
    /// query, print JSON.
    Run {
        #[arg(short, long, default_value = "0.1")]
        scale: f64,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
        /// Skip generation; load from this directory.
        #[arg(long)]
        dataset_dir: Option<PathBuf>,
        /// Warm-run sample count (per query, per parameter).
        #[arg(long, default_value_t = 50)]
        warm_runs: usize,
        /// How many distinct Person ids to use as `$personId` per query.
        #[arg(long, default_value_t = 3)]
        param_count: usize,
        /// Restrict to specific queries; omit = all four.
        #[arg(long, value_enum)]
        only: Vec<Query>,
    },
    /// Load-only timing: bulk-load the dataset into an InMemory namespace
    /// and emit `{load_time_secs, dataset_sizes}` JSON. Used by the
    /// publishable bench harness's "write throughput" track to compare
    /// the engine's load path against Kuzu COPY / Neo4j UNWIND on the
    /// same CSV dataset. Includes the final `flush()` so the number
    /// reflects "memtable + SST flushed" — apples-to-apples with Kuzu's
    /// COPY (which also persists to disk).
    Load {
        #[arg(short, long, default_value = "0.1")]
        scale: f64,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        dataset_dir: Option<PathBuf>,
    },
    /// Bulk-load the dataset into a *remote* S3/R2 namespace. Used by the
    /// publishable Bench B to materialise the LDBC SF1 dataset on R2 so
    /// the gateway/worker stack in production can serve it. Reads AWS
    /// credentials from env (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    /// optional `AWS_ENDPOINT_URL` for R2).
    ///
    /// Idempotent against an existing namespace at version > 0: the loader
    /// `upsert`s, so a re-run with the same dataset produces the same
    /// committed state but at a higher manifest version.
    LoadR2 {
        #[arg(short, long, default_value = "0.1")]
        scale: f64,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        dataset_dir: Option<PathBuf>,
        /// R2 / S3 bucket name.
        #[arg(long)]
        bucket: String,
        /// Namespace id (e.g. `bench-snb-sf1`).
        #[arg(long)]
        namespace: String,
        /// Object-store root prefix; must match the worker's `storage.root_prefix`.
        #[arg(long, default_value = "tenants")]
        root_prefix: String,
    },
    /// Measure int8 quantization recall@k + latency vs exact f32 on synthetic
    /// unit-norm vectors. The gate for the int8 storage work: run at the
    /// embedder dimensions (e.g. 256 and 1536) and confirm recall@10 stays high
    /// before committing to int8 on disk.
    VectorRecall {
        /// Vector dimension (e.g. 256 local, 1536 OpenAI).
        #[arg(long, default_value_t = 256)]
        dim: usize,
        /// Number of stored vectors to search over.
        #[arg(long, default_value_t = 10_000)]
        num: usize,
        /// Number of query vectors to average recall/latency over.
        #[arg(long, default_value_t = 100)]
        queries: usize,
        /// Top-k.
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// Number of cluster centroids. `0` = uniform random (pessimistic
        /// floor); `>0` draws vectors/queries around centroids, like real
        /// embeddings with meaningful neighbours.
        #[arg(long, default_value_t = 64)]
        clusters: usize,
        /// Cluster tightness (Gaussian noise stddev added to a centroid before
        /// renormalizing). Smaller = tighter clusters.
        #[arg(long, default_value_t = 0.35)]
        spread: f32,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
    },
    /// Measure the Vamana ANN **index** recall@k vs the exact flat KNN, and
    /// index latency vs full-scan latency, over the real engine: build the `.vg`
    /// by compaction, then run the same KNN query optimized (index) and
    /// un-optimized (flat scan) on one snapshot. The gate for RFC-030 ANN: run
    /// with `--clusters 64` (realistic) and `--clusters 0` (pessimistic floor)
    /// and confirm recall@10 stays high while the index out-runs the scan.
    #[cfg(feature = "vector-index")]
    AnnBench {
        /// Vector dimension (e.g. 256 local, 1536 OpenAI).
        #[arg(long, default_value_t = 256)]
        dim: usize,
        /// Number of stored vectors (corpus size) to index and search.
        #[arg(long, default_value_t = 10_000)]
        num: usize,
        /// Number of query vectors to average recall/latency over.
        #[arg(long, default_value_t = 100)]
        queries: usize,
        /// Top-k.
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        /// Number of cluster centroids. `0` = uniform random (pessimistic
        /// floor); `>0` draws vectors/queries around centroids, like real
        /// embeddings with meaningful neighbours.
        #[arg(long, default_value_t = 64)]
        clusters: usize,
        /// Cluster tightness (Gaussian noise stddev added to a centroid before
        /// renormalizing). Smaller = tighter clusters.
        #[arg(long, default_value_t = 0.35)]
        spread: f32,
        /// Index search beam width (clamped up to at least `k`). Higher trades
        /// latency for recall.
        #[arg(long, default_value_t = 64)]
        ef: usize,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
    },
    /// Write the hub-shaped graph (`graph_shape.rs`): a few `:Device` hubs,
    /// one holding most `ON_DEVICE` edges, and a small `:Call` label. Writes
    /// into `--uri` with the current engine, or NDJSON for
    /// `bench/load_ndjson.py` with `--ndjson-dir`.
    GraphShapeGenerate {
        /// Namespace URI, e.g. `file:///data/shape?ns=shape`.
        #[arg(long)]
        uri: Option<String>,
        /// Emit NDJSON here instead of writing a namespace.
        #[arg(long)]
        ndjson_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 2_400_000)]
        nodes: u64,
        #[arg(long, default_value_t = 4)]
        hubs: u64,
        #[arg(long, default_value_t = 1_700_000)]
        hub_max_degree: u64,
        #[arg(long, default_value_t = 2_958)]
        calls: u64,
        #[arg(long, default_value_t = 64)]
        embedding_dim: usize,
        #[arg(long, default_value_t = 250_000)]
        flush_every: u64,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
    },
    /// Run one read query read-only against a namespace and print a JSON
    /// report (plan, rows, wall time per run, peak RSS). `$hubs` / `$hub`
    /// are bound to the `:Device` ids automatically.
    GraphShapeQuery {
        #[arg(long)]
        uri: String,
        #[arg(long)]
        query: String,
        /// Extra parameters as a JSON object.
        #[arg(long)]
        params: Option<String>,
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        /// SST cache size; the Python client's default is 64 MiB.
        #[arg(long, default_value_t = 64 * 1024 * 1024)]
        cache_bytes: usize,
        #[arg(long)]
        timeout_ms: Option<u64>,
    },
    /// Gate V5/FT3, incremental VG6/FT4 and paged forward/inverse graph SSTs.
    ///
    /// Corpus generation and builders are streaming. Search readers use an
    /// instrumented immutable RangeSource in cache=0/sized modes; graph readers
    /// report authenticated paged I/O and exact endpoint/property parity.
    #[cfg(feature = "object-native")]
    ObjectNative {
        #[arg(long, default_value_t = 100_000)]
        vectors: usize,
        #[arg(long, default_value_t = 100_000)]
        documents: usize,
        #[arg(long, default_value_t = 256)]
        dim: usize,
        #[arg(long, default_value_t = 20)]
        queries: usize,
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 64)]
        clusters: usize,
        #[arg(long, default_value_t = 0.20)]
        spread: f32,
        /// Native `tenant` filter cardinality; one query keeps bucket zero.
        #[arg(long, default_value_t = 16)]
        filter_buckets: usize,
        #[arg(long, default_value_t = 512)]
        page_rows: usize,
        #[arg(long, default_value_t = 8)]
        branch_factor: usize,
        #[arg(long, default_value_t = 8)]
        nprobe: usize,
        #[arg(long, default_value_t = 64)]
        max_nprobe: usize,
        #[arg(long, default_value_t = 8)]
        rerank_factor: usize,
        /// Exact-range LRU capacity used for the warm pass.
        #[arg(long, default_value_t = 64 * 1024 * 1024)]
        cache_bytes: usize,
        /// Logical ceiling for each external-memory index builder.
        #[arg(long, default_value_t = 256 * 1024 * 1024)]
        build_memory_bytes: usize,
        /// Deterministic latency added to every cache-miss range request.
        #[arg(long, default_value_t = 0)]
        range_latency_ms: u64,
        /// Number of ordered VG6/FT4 mutation deltas after the complete seed.
        #[arg(long, default_value_t = 3)]
        delta_segments: usize,
        /// Regular one-to-one graph keys in addition to both adversarial hubs.
        #[arg(long, default_value_t = 1_024)]
        graph_keys: usize,
        /// Degree of the forward and inverse adversarial hubs.
        #[arg(long, default_value_t = 4_096)]
        graph_high_degree: usize,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
        /// Fail if any filtered/unfiltered metric has lower recall.
        #[arg(long)]
        min_recall_at_k: Option<f64>,
        /// Fail if one cold query fetches more than this artifact fraction.
        #[arg(long)]
        max_cold_bytes_ratio: Option<f64>,
        /// Fail if decoded reader metadata exceeds this artifact fraction.
        #[arg(long)]
        max_reader_metadata_ratio: Option<f64>,
        /// Fail if a cold vector/text p95 exceeds this many milliseconds.
        #[arg(long)]
        max_query_p95_ms: Option<f64>,
        /// Linux: fail if the process VmHWM exceeds this exact byte count.
        #[arg(long)]
        max_rss_bytes: Option<u64>,
        /// Fail if one Search-LSM family fans out to more segments.
        #[arg(long, default_value_t = 32)]
        max_fanout: usize,
        /// Execution ceiling and gate for concurrent immutable range fetches.
        #[arg(long, default_value_t = 16)]
        max_in_flight: u64,
    },
    /// Run the real HTTP server on a store with simulated object-store
    /// latency (see `http-load`).
    #[cfg(feature = "http-load")]
    Serve {
        /// Storage URI, e.g. `file:///tmp/nb?ns=bench` (`?ns=` optional with
        /// `--multi-tenant`).
        #[arg(long)]
        store: String,
        #[arg(long, default_value = "127.0.0.1:18480")]
        listen: std::net::SocketAddr,
        /// Per-request latency lower bound, in milliseconds.
        #[arg(long, default_value_t = 20)]
        latency_min_ms: u64,
        /// Per-request latency upper bound, in milliseconds.
        #[arg(long, default_value_t = 40)]
        latency_max_ms: u64,
        /// Transfer rate charged on bodies, in MiB/s (0 = unlimited).
        #[arg(long, default_value_t = 100)]
        bandwidth_mib_per_sec: u64,
        #[arg(long)]
        multi_tenant: bool,
        #[arg(long, default_value = "0s", value_parser = humantime::parse_duration)]
        group_commit_window: std::time::Duration,
        /// `NAMIDB_MEMORY_MAX_BYTES` semantics: bytes, `auto` (90% of the
        /// cgroup limit) or `0` (off).
        #[arg(long, default_value = "auto")]
        memory_max_bytes: String,
        #[arg(long, default_value = "30s", value_parser = humantime::parse_duration)]
        flush_interval: std::time::Duration,
        #[arg(long, default_value = "300s", value_parser = humantime::parse_duration)]
        compaction_interval: std::time::Duration,
        /// Where to publish the store request counters (JSON, every 100 ms).
        #[arg(long)]
        counters_file: Option<PathBuf>,
    },
    /// Drive a server over HTTP and report write throughput / vector search.
    #[cfg(feature = "http-load")]
    HttpLoad {
        #[arg(long, default_value = "http://127.0.0.1:18480")]
        url: String,
        /// Multi-tenant namespace (`/<ns>/v0/cypher`); omit for single-tenant.
        #[arg(long)]
        namespace: Option<String>,
        #[arg(long, value_enum)]
        scenario: http_load::Scenario,
        /// Rows to load (`bulk`, `vectors`) or the corpus size (`vector-search`).
        #[arg(long, default_value_t = 10_000)]
        count: usize,
        #[arg(long, default_value_t = 500)]
        batch: usize,
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        /// Approximate JSON size of one document node.
        #[arg(long, default_value_t = 3000)]
        node_bytes: usize,
        #[arg(long, default_value_t = 1024)]
        dim: usize,
        #[arg(long, default_value_t = 256)]
        clusters: usize,
        #[arg(long, default_value_t = 100)]
        queries: usize,
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        #[arg(long, default_value_t = 64)]
        ef: usize,
        #[arg(short = 'S', long, default_value_t = 42)]
        seed: u64,
        /// The `serve --counters-file`, to report store requests per step.
        #[arg(long)]
        counters_file: Option<PathBuf>,
        /// Create the constraint and the vector/fulltext indexes first.
        #[arg(long)]
        setup: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Generate { scale, seed, out } => {
            let sizes = dataset::generate(&out, &DatasetConfig { scale, seed })?;
            eprintln!(
                "generated dataset @ scale={scale} seed={seed} into {}: {sizes:?}",
                out.display()
            );
        }
        Cmd::GraphShapeGenerate {
            uri,
            ndjson_dir,
            nodes,
            hubs,
            hub_max_degree,
            calls,
            embedding_dim,
            flush_every,
            seed,
        } => {
            let cfg = graph_shape::ShapeConfig {
                nodes,
                hubs,
                hub_max_degree,
                calls,
                embedding_dim,
                flush_every,
                seed,
            };
            match (uri, ndjson_dir) {
                (Some(uri), None) => graph_shape::write_direct(&cfg, &uri).await?,
                (None, Some(dir)) => graph_shape::write_ndjson(&cfg, &dir)?,
                _ => anyhow::bail!("pass exactly one of --uri or --ndjson-dir"),
            }
        }
        Cmd::GraphShapeQuery {
            uri,
            query,
            params,
            repeat,
            cache_bytes,
            timeout_ms,
        } => {
            let report = graph_shape::run_query(
                &uri,
                &query,
                params.as_deref(),
                repeat,
                cache_bytes,
                timeout_ms,
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::Run {
            scale,
            seed,
            dataset_dir,
            warm_runs,
            param_count,
            only,
        } => {
            let (dataset_dir, _kept) = if let Some(d) = dataset_dir {
                (d, None)
            } else {
                let tmp = std::env::temp_dir()
                    .join(format!("namidb-bench-{}", uuid::Uuid::now_v7().simple()));
                std::fs::create_dir_all(&tmp)?;
                (tmp.clone(), Some(tmp))
            };
            let sizes = if dataset_dir.join("persons.csv").exists() {
                DatasetSizes::from_scale(scale)
            } else {
                dataset::generate(&dataset_dir, &DatasetConfig { scale, seed })?
            };
            let writer = loader::load_into_in_memory(&dataset_dir, "bench").await?;

            // Pick `param_count` distinct Person ids using deterministic
            // indices so Kuzu can rerun the same params.
            let params: Vec<String> = (0..param_count)
                .map(|i| make_person_id_hex(i * (sizes.persons / param_count.max(1)).max(1)))
                .collect();

            let queries: Vec<Query> = if only.is_empty() {
                vec![Query::Ic02, Query::Ic07, Query::Ic08, Query::Ic09]
            } else {
                only
            };

            // If profiling is enabled, reset accumulators so the dataset
            // load (which can dwarf the queries) doesn't dominate the
            // dump. Then run the queries; finally dump to stderr.
            if namidb_core::profile::enabled() {
                namidb_core::profile::reset();
                eprintln!("[profile] reset before queries (NAMIDB_PROFILE_DUMP=1 active)");
            }

            let mut results: Vec<QueryResult> = Vec::new();
            for q in &queries {
                for p in &params {
                    let r = run_query(&writer, *q, p, warm_runs).await?;
                    eprintln!(
                        " {} param={} rows={} cold={}µs warm_p50={}µs",
                        r.query,
                        &r.param[..8],
                        r.rows,
                        r.cold_us,
                        r.warm_p50_us
                    );
                    results.push(r);
                }
            }

            if namidb_core::profile::enabled() {
                eprintln!(
                    "\n[profile] aggregated across {} queries × {} params × ({}+1) runs:\n",
                    queries.len(),
                    params.len(),
                    warm_runs
                );
                eprintln!("{}", namidb_core::profile::dump_table());
            }

            let out = BenchOutput {
                scale,
                seed,
                dataset_sizes: SizesReport::from(&sizes),
                results,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Cmd::Load {
            scale,
            seed,
            dataset_dir,
        } => {
            // Resolve or generate the dataset, same protocol as `Run`.
            let (dataset_dir, _kept) = if let Some(d) = dataset_dir {
                (d, None)
            } else {
                let tmp = std::env::temp_dir()
                    .join(format!("namidb-bench-{}", uuid::Uuid::now_v7().simple()));
                std::fs::create_dir_all(&tmp)?;
                (tmp.clone(), Some(tmp))
            };
            let sizes = if dataset_dir.join("persons.csv").exists() {
                DatasetSizes::from_scale(scale)
            } else {
                dataset::generate(&dataset_dir, &DatasetConfig { scale, seed })?
            };

            // Time the load: open writer + bulk_load (which flushes SSTs).
            // Drop the writer afterwards so its allocations are freed by the
            // time we print the result, in case a harness reuses the process.
            // The number we publish is `load_time_secs` (includes flush). The
            // pre-load `WriterSession::open` is ~ms; we keep it inside the
            // headline number because the equivalent Kuzu `Database::new`
            // and Neo4j `Driver.connect` are similarly tiny + always counted.
            let start = std::time::Instant::now();
            let writer = loader::load_into_in_memory(&dataset_dir, "bench").await?;
            let load_time_secs = start.elapsed().as_secs_f64();
            drop(writer);

            let report = LoadReport {
                backend: "namidb-engine-inmemory",
                scale,
                seed,
                dataset_sizes: SizesReport::from(&sizes),
                load_time_secs,
                elements: sizes.persons
                    + sizes.posts
                    + sizes.comments
                    + sizes.knows
                    + sizes.has_creator
                    + sizes.likes
                    + sizes.reply_of,
            };
            eprintln!(
                "load: {} elements in {:.3}s = {:.0} elem/s",
                report.elements,
                report.load_time_secs,
                report.elements as f64 / report.load_time_secs,
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::LoadR2 {
            scale,
            seed,
            dataset_dir,
            bucket,
            namespace,
            root_prefix,
        } => {
            // Resolve or generate the dataset.
            let (dataset_dir, _kept) = if let Some(d) = dataset_dir {
                (d, None)
            } else {
                let tmp = std::env::temp_dir()
                    .join(format!("namidb-bench-{}", uuid::Uuid::now_v7().simple()));
                std::fs::create_dir_all(&tmp)?;
                (tmp.clone(), Some(tmp))
            };
            let sizes = if dataset_dir.join("persons.csv").exists() {
                DatasetSizes::from_scale(scale)
            } else {
                dataset::generate(&dataset_dir, &DatasetConfig { scale, seed })?
            };

            // Build the S3/R2 object store from env. The worker uses the
            // exact same construction in `namidb-worker::namespaces`.
            let store: std::sync::Arc<dyn object_store::ObjectStore> = {
                let mut builder =
                    object_store::aws::AmazonS3Builder::from_env().with_bucket_name(&bucket);
                if let Ok(endpoint) = std::env::var("AWS_ENDPOINT_URL") {
                    builder = builder.with_endpoint(endpoint);
                }
                if let Ok(allow) = std::env::var("AWS_ALLOW_HTTP") {
                    if allow == "true" || allow == "1" {
                        builder = builder.with_allow_http(true);
                    }
                }
                std::sync::Arc::new(builder.build()?)
            };

            eprintln!(
                "load-r2: bucket={bucket} prefix={root_prefix} namespace={namespace} \
  elements={} (scale={scale})",
                sizes.persons
                    + sizes.posts
                    + sizes.comments
                    + sizes.knows
                    + sizes.has_creator
                    + sizes.likes
                    + sizes.reply_of,
            );

            let start = std::time::Instant::now();
            let writer =
                loader::load_into_store(store, &root_prefix, &namespace, &dataset_dir).await?;
            let load_time_secs = start.elapsed().as_secs_f64();
            let manifest_version = writer.manifest_version();
            drop(writer);

            let report = serde_json::json!({
            "backend": "namidb-engine-r2",
            "scale": scale,
            "seed": seed,
            "dataset_sizes": SizesReport::from(&sizes),
            "namespace": namespace,
            "root_prefix": root_prefix,
            "bucket": bucket,
            "manifest_version": manifest_version,
            "elements": sizes.persons + sizes.posts + sizes.comments
            + sizes.knows + sizes.has_creator + sizes.likes + sizes.reply_of,
            "load_time_secs": load_time_secs,
            });
            eprintln!(
                "load-r2 done: manifest_version={manifest_version} \
  load_time={load_time_secs:.1}s ({:.0} elem/s)",
                (sizes.persons
                    + sizes.posts
                    + sizes.comments
                    + sizes.knows
                    + sizes.has_creator
                    + sizes.likes
                    + sizes.reply_of) as f64
                    / load_time_secs,
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::VectorRecall {
            dim,
            num,
            queries,
            k,
            clusters,
            spread,
            seed,
        } => {
            let report = vector_recall::run(dim, num, queries, k, clusters, spread, seed);
            eprintln!(
                "vector-recall dim={dim} num={num} queries={queries} k={k} clusters={clusters}: \
                 recall@{k}={:.4} (fixed-scale {:.4}) compression={:.2}x  \
                 exact p50={}µs  int8 p50={}µs",
                report.recall_at_k,
                report.recall_at_k_fixed_scale,
                report.compression_ratio,
                report.exact_p50_us,
                report.int8_p50_us,
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        #[cfg(feature = "vector-index")]
        Cmd::AnnBench {
            dim,
            num,
            queries,
            k,
            clusters,
            spread,
            ef,
            seed,
        } => {
            let report = ann_bench::run(dim, num, queries, k, clusters, spread, ef, seed).await?;
            if !report.cypher_index_path_reachable {
                eprintln!(
                    "NOTE: cypher_index_path_reachable=false — the optimizer did not rewrite \
                     a plain KNN query to VectorSearch for this catalog; the figures below come \
                     from the low-level index reader."
                );
            }
            eprintln!(
                "ann-bench dim={dim} num={num} queries={queries} k={k} clusters={clusters} ef={}: \
                 recall@{k}={:.4}  index p50={}µs  flat p50={}µs  speedup={:.1}x  build={:.2}s",
                report.ef,
                report.recall_at_k,
                report.index_p50_us,
                report.flat_p50_us,
                report.speedup_p50,
                report.build_secs,
            );
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        #[cfg(feature = "object-native")]
        Cmd::ObjectNative {
            vectors,
            documents,
            dim,
            queries,
            k,
            clusters,
            spread,
            filter_buckets,
            page_rows,
            branch_factor,
            nprobe,
            max_nprobe,
            rerank_factor,
            cache_bytes,
            build_memory_bytes,
            range_latency_ms,
            delta_segments,
            graph_keys,
            graph_high_degree,
            seed,
            min_recall_at_k,
            max_cold_bytes_ratio,
            max_reader_metadata_ratio,
            max_query_p95_ms,
            max_rss_bytes,
            max_fanout,
            max_in_flight,
        } => {
            let report = object_native::run(object_native::ObjectNativeConfig {
                vectors,
                documents,
                dim,
                queries,
                k,
                clusters,
                spread,
                filter_buckets,
                page_rows,
                branch_factor,
                nprobe,
                max_nprobe,
                rerank_factor,
                cache_bytes,
                build_memory_bytes,
                range_latency_ms,
                delta_segments,
                graph_keys,
                graph_high_degree,
                seed,
                gates: object_native::GateThresholds {
                    min_recall_at_k,
                    max_cold_bytes_ratio,
                    max_reader_metadata_ratio,
                    max_query_p95_ms,
                    max_rss_bytes,
                    max_fanout,
                    max_in_flight,
                },
            })
            .await?;
            let passed = report.passed();
            let failures = report.gates.failures.clone();
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !passed {
                anyhow::bail!("object-native gates failed: {}", failures.join("; "));
            }
        }
        #[cfg(feature = "http-load")]
        Cmd::Serve {
            store,
            listen,
            latency_min_ms,
            latency_max_ms,
            bandwidth_mib_per_sec,
            multi_tenant,
            group_commit_window,
            memory_max_bytes,
            flush_interval,
            compaction_interval,
            counters_file,
        } => {
            http_load::serve(http_load::ServeConfig {
                store,
                listen,
                latency: latency_store::LatencyModel::new(
                    latency_min_ms,
                    latency_max_ms,
                    bandwidth_mib_per_sec,
                ),
                multi_tenant,
                group_commit_window,
                memory_max_bytes,
                flush_interval,
                compaction_interval,
                counters_file,
            })
            .await?;
        }
        #[cfg(feature = "http-load")]
        Cmd::HttpLoad {
            url,
            namespace,
            scenario,
            count,
            batch,
            concurrency,
            node_bytes,
            dim,
            clusters,
            queries,
            k,
            ef,
            seed,
            counters_file,
            setup,
        } => {
            let report = http_load::http_load(http_load::LoadConfig {
                url,
                namespace,
                scenario,
                count,
                batch,
                concurrency,
                node_bytes,
                dim,
                clusters,
                queries,
                k,
                ef,
                seed,
                counters_file,
                setup,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

/// One-shot load-only timing report emitted by `Cmd::Load`. The
/// publishable bench harness consumes this JSON via
/// `scripts/bench_publish/bench_d_write_throughput.py`.
#[derive(Debug, serde::Serialize)]
struct LoadReport {
    backend: &'static str,
    scale: f64,
    seed: u64,
    dataset_sizes: SizesReport,
    elements: usize,
    load_time_secs: f64,
}

fn make_person_id_hex(i: usize) -> String {
    let mut bytes = [0u8; 16];
    bytes[0] = b'P';
    let i_bytes = (i as u128).to_be_bytes();
    bytes[1..].copy_from_slice(&i_bytes[1..]);
    let mut s = String::with_capacity(32);
    for b in bytes {
        let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{:02x}", b));
    }
    s
}
