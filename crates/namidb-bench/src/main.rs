//! `namidb-bench` CLI — gate harness.
//!
//! Two subcommands:
//!
//! - `generate` — writes a synthetic LDBC-shaped dataset to a directory.
//! - `run` — loads the dataset (or regenerates it inline) and bench-runs
//! each query, printing JSON to stdout.
//!
//! See `bench/README.md` for the full workflow + paired Kuzu runner.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod dataset;
mod loader;
mod queries;
mod runner;

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
