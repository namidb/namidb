//! Per-query timing runner. Produces a `QueryResult` per (query,
//! parameter) tuple with cold / warm timings and the row count NamiDB
//! returned. Output JSON is shape-compatible with the Python Kuzu
//! adapter (`bench/kuzu_runner.py`) so the two can be diffed.

use std::time::Instant;

use anyhow::Result;
use namidb_query::cost::StatsCatalog;
use namidb_query::{execute, parse, plan, Params};
use namidb_storage::WriterSession;
use serde::Serialize;

use crate::queries::Query;

/// One bench iteration produces this record. Times are in microseconds.
#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub backend: String,
    pub query: &'static str,
    pub param: String,
    pub rows: usize,
    pub cold_us: u64,
    pub warm_p50_us: u64,
    pub warm_p95_us: u64,
    pub warm_p99_us: u64,
    pub warm_runs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchOutput {
    pub scale: f64,
    pub seed: u64,
    pub dataset_sizes: SizesReport,
    pub results: Vec<QueryResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizesReport {
    pub persons: usize,
    pub posts: usize,
    pub comments: usize,
    pub knows: usize,
    pub has_creator: usize,
    pub likes: usize,
    pub reply_of: usize,
}

impl From<&crate::dataset::DatasetSizes> for SizesReport {
    fn from(s: &crate::dataset::DatasetSizes) -> Self {
        Self {
            persons: s.persons,
            posts: s.posts,
            comments: s.comments,
            knows: s.knows,
            has_creator: s.has_creator,
            likes: s.likes,
            reply_of: s.reply_of,
        }
    }
}

/// Run `query` `warm_runs` times against a writer-pinned snapshot, plus
/// a single "cold" run that builds a fresh snapshot first. Returns
/// timings + the row count of the first run (used as the comparison
/// baseline).
pub async fn run_query(
    writer: &WriterSession,
    query: Query,
    param: &str,
    warm_runs: usize,
) -> Result<QueryResult> {
    // Cold timing: build snapshot + catalog fresh each iteration to
    // exclude the cost of caching open Parquet metadata.
    let cold_start = Instant::now();
    let (cold_rows, _) = exec_once(writer, query, param).await?;
    let cold_us = cold_start.elapsed().as_micros() as u64;

    // Warm timings: keep the snapshot live. The first run still pays
    // some cache costs; we record `warm_runs` total samples and skip
    // the first when reporting p50/p95.
    let mut times: Vec<u64> = Vec::with_capacity(warm_runs);
    for _ in 0..warm_runs {
        let start = Instant::now();
        let _ = exec_once(writer, query, param).await?;
        times.push(start.elapsed().as_micros() as u64);
    }
    times.sort_unstable();
    let p50 = pct(&times, 0.50);
    let p95 = pct(&times, 0.95);
    let p99 = pct(&times, 0.99);

    Ok(QueryResult {
        backend: "namidb".into(),
        query: query.name(),
        param: param.into(),
        rows: cold_rows,
        cold_us,
        warm_p50_us: p50,
        warm_p95_us: p95,
        warm_p99_us: p99,
        warm_runs,
    })
}

async fn exec_once(writer: &WriterSession, query: Query, param: &str) -> Result<(usize, ())> {
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let q = parse(&query.cypher(param))
        .map_err(|errs| anyhow::anyhow!("parse {} param={}: {:?}", query.name(), param, errs))?;
    let plan = plan(&q, &catalog)
        .map_err(|e| anyhow::anyhow!("plan {} param={}: {e}", query.name(), param))?;
    let rows = execute(&plan, &snap, &Params::default())
        .await
        .map_err(|e| anyhow::anyhow!("execute {} param={}: {e}", query.name(), param))?;
    Ok((rows.len(), ()))
}

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
