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
 }
 Ok(())
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
