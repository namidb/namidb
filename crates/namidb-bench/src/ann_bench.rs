//! `ann-bench` — measure the Vamana ANN **index** recall@k against the exact
//! flat KNN, plus index latency vs full-scan latency, over the *real* storage
//! engine.
//!
//! Unlike `vector-recall` (which measures only int8-quantization arithmetic in
//! memory), this builds a namespace on an `InMemory` object store, registers a
//! cosine `VectorIndexDescriptor`, writes the corpus across two L0 SSTs, and
//! `compact_l0`s so the compactor materialises the `.vg` graph. It then:
//!
//! - **index path** — queries the Vamana graph directly via
//!   [`Snapshot::vector_search`] (the RFC-030 `.vg` reader);
//! - **flat path** — brute-force cosine over the same corpus. This is exact, so
//!   it is the ground truth for recall and the compute floor for the scan.
//!
//! `recall@k = |index_topk ∩ flat_topk| / k`, averaged over the queries, keyed
//! on node id (both paths search the identical corpus). The latency comparison
//! is the core search only (both exclude row materialisation, which each path
//! would pay alike), so the reported speedup is a conservative lower bound on
//! what a full scan costs end-to-end.
//!
//! It also reports `cypher_index_path_reachable`: whether the optimizer's
//! `VectorSearch` rewrite fires for a plain KNN Cypher query against this
//! catalog — i.e. whether real queries (not just the low-level reader) reach the
//! index.
//!
//! Deterministic from `--seed`. Two workloads: `--clusters 0` is uniform on the
//! sphere (a pessimistic floor — no meaningful neighbours); `--clusters N` draws
//! the corpus and queries around `N` centroids, like real embeddings.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::{lower, optimize, parse, StatsCatalog};
use namidb_storage::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use crate::vector_recall::{percentile, perturbed_unit_vector, random_unit_vector};

const INDEX_NAME: &str = "doc_emb";

/// JSON report emitted to stdout.
#[derive(Debug, Serialize)]
pub struct AnnBenchReport {
    pub dim: usize,
    pub num_vectors: usize,
    pub num_queries: usize,
    pub k: usize,
    pub clusters: usize,
    /// Beam width passed to the index search (higher ⇒ better recall, more work).
    pub ef: usize,
    /// Whether the optimizer rewrites a plain KNN Cypher query to `VectorSearch`
    /// against this catalog — i.e. whether real queries reach the `.vg`, not just
    /// the low-level reader this bench calls.
    pub cypher_index_path_reachable: bool,
    /// Mean recall@k of the indexed answer vs the exact flat top-k.
    pub recall_at_k: f64,
    /// Seconds spent building the corpus + compacting the `.vg`.
    pub build_secs: f64,
    pub index_p50_us: u128,
    pub index_p99_us: u128,
    pub flat_p50_us: u128,
    pub flat_p99_us: u128,
    /// `flat_p50 / index_p50` — how much faster the index serves the median
    /// query than the brute-force scan (>1 ⇒ the index wins). A lower bound:
    /// the flat number is compute-only, the real scan also materialises rows.
    pub speedup_p50: f64,
}

fn schema(dim: u32) -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim }, false).unwrap(),
                PropertyDef::new("title", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn rec(title: &str, emb: Vec<f32>) -> NodeWriteRecord {
    let mut p = BTreeMap::new();
    p.insert("title".into(), CoreValue::Str(title.into()));
    p.insert("embedding".into(), CoreValue::Vec(emb));
    NodeWriteRecord {
        properties: p,
        schema_version: 1,
        ..Default::default()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Brute-force exact top-k indices into `stored` by cosine to `q`.
fn exact_top_k(stored: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..stored.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        cosine(&stored[b], q)
            .partial_cmp(&cosine(&stored[a], q))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

/// Run the harness. `clusters == 0` is uniform random; `clusters > 0` draws the
/// corpus and queries around that many centroids (`spread` is cluster tightness).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    dim: usize,
    num_vectors: usize,
    num_queries: usize,
    k: usize,
    clusters: usize,
    spread: f32,
    ef: usize,
    seed: u64,
) -> Result<AnnBenchReport> {
    if num_vectors < k {
        return Err(anyhow!(
            "num_vectors ({num_vectors}) must be >= k ({k}) for recall@k"
        ));
    }
    let ef = ef.max(k);

    // ── 1. Generate the corpus + queries deterministically. ───────────────
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centroids: Vec<Vec<f32>> = (0..clusters)
        .map(|_| random_unit_vector(&mut rng, dim))
        .collect();
    let stored: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            if clusters == 0 {
                random_unit_vector(&mut rng, dim)
            } else {
                perturbed_unit_vector(&mut rng, &centroids[i % clusters], spread)
            }
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|i| {
            if clusters == 0 {
                random_unit_vector(&mut rng, dim)
            } else {
                // Near a centroid: the true neighbours are that cluster's
                // members, so the top-k are well separated.
                perturbed_unit_vector(&mut rng, &centroids[i % clusters], spread * 0.5)
            }
        })
        .collect();

    // ── 2. Build the namespace + Vamana index on an InMemory store. ───────
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("ann-bench").unwrap());
    let mut w = WriterSession::open(store, paths)
        .await
        .map_err(|e| anyhow!("open writer: {e}"))?;
    let sch = schema(dim as u32);
    w.register_vector_index(VectorIndexDescriptor {
        name: INDEX_NAME.into(),
        label: "Doc".into(),
        property: "embedding".into(),
        dim: dim as u32,
        metric: VectorMetric::Cosine,
        r: 32,
        l_build: 64,
        alpha: 1.2,
        quantization: VectorQuantization::None,
    })
    .await
    .map_err(|e| anyhow!("register index: {e}"))?;

    let build_start = Instant::now();
    // `stored[i]` ↔ `node_ids[i]`, so an index hit's NodeId maps back to its
    // generated vector for the recall comparison. Split across two L0 SSTs so
    // compaction has a real multi-SST merge to fold into the graph.
    let mut node_ids: Vec<NodeId> = Vec::with_capacity(num_vectors);
    let half = stored.len().div_ceil(2);
    for (i, emb) in stored.iter().enumerate() {
        let id = NodeId::new();
        node_ids.push(id);
        w.upsert_node("Doc", id, &rec(&format!("d{i}"), emb.clone()))
            .map_err(|e| anyhow!("upsert d{i}: {e}"))?;
        if i + 1 == half {
            w.flush(sch.clone())
                .await
                .map_err(|e| anyhow!("flush L0 #1: {e}"))?;
        }
    }
    w.flush(sch.clone())
        .await
        .map_err(|e| anyhow!("flush L0 #2: {e}"))?;
    w.compact_l0(&sch)
        .await
        .map_err(|e| anyhow!("compact_l0: {e}"))?;
    let build_secs = build_start.elapsed().as_secs_f64();

    let snap = w.snapshot();
    let id_to_idx: BTreeMap<NodeId, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Sanity: the `.vg` must answer, else we'd be benchmarking nothing.
    let probe = snap
        .vector_search(INDEX_NAME, &queries[0], k, ef)
        .await
        .map_err(|e| anyhow!("vector_search probe: {e}"))?;
    if probe.is_empty() {
        return Err(anyhow!(
            "vector_search returned no hits — the .vg index was not built \
             (compaction did not materialise it)"
        ));
    }

    // Does a plain KNN Cypher query reach the index through the optimizer?
    let cypher_index_path_reachable = cypher_reaches_index(&snap, k)?;

    // ── 3. Run, timing each path and scoring recall against the flat top-k. ─
    let mut index_us: Vec<u128> = Vec::with_capacity(num_queries);
    let mut flat_us: Vec<u128> = Vec::with_capacity(num_queries);
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in &queries {
        let t = Instant::now();
        let hits_idx = snap
            .vector_search(INDEX_NAME, q, k, ef)
            .await
            .map_err(|e| anyhow!("vector_search: {e}"))?;
        index_us.push(t.elapsed().as_micros());

        let t = Instant::now();
        let exact = exact_top_k(&stored, q, k);
        flat_us.push(t.elapsed().as_micros());

        // Compare by corpus index (NodeId → index for the ANN hits).
        let exact_set: HashSet<usize> = exact.iter().copied().collect();
        let index_idxs: HashSet<usize> = hits_idx
            .iter()
            .filter_map(|(id, _)| id_to_idx.get(id).copied())
            .collect();
        hits += index_idxs.iter().filter(|i| exact_set.contains(i)).count();
        total += exact_set.len();
    }

    index_us.sort_unstable();
    flat_us.sort_unstable();
    let recall_at_k = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };
    let index_p50 = percentile(&index_us, 50.0);
    let flat_p50 = percentile(&flat_us, 50.0);
    let speedup_p50 = if index_p50 > 0 {
        flat_p50 as f64 / index_p50 as f64
    } else {
        0.0
    };

    Ok(AnnBenchReport {
        dim,
        num_vectors,
        num_queries,
        k,
        clusters,
        ef,
        cypher_index_path_reachable,
        recall_at_k,
        build_secs,
        index_p50_us: index_p50,
        index_p99_us: percentile(&index_us, 99.0),
        flat_p50_us: flat_p50,
        flat_p99_us: percentile(&flat_us, 99.0),
        speedup_p50,
    })
}

/// `true` if the optimizer rewrites a plain KNN Cypher query into a
/// `VectorSearch` against this snapshot's catalog (the index is reachable from
/// real queries, not only the low-level reader). Detected by checking the
/// optimized plan tree contains the operator.
fn cypher_reaches_index(snap: &namidb_storage::Snapshot<'_>, k: usize) -> Result<bool> {
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let cypher = format!(
        "MATCH (d:Doc) WHERE d.embedding IS NOT NULL \
         RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
         ORDER BY score DESC LIMIT {k}"
    );
    let flat_plan = lower(&parse(&cypher).map_err(|e| anyhow!("parse: {e:?}"))?)
        .map_err(|e| anyhow!("lower: {e}"))?;
    let index_plan = optimize(flat_plan, &catalog);
    Ok(serde_json::to_string(&index_plan)?.contains("VectorSearch"))
}
