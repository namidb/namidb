//! DiskANN/Vamana `VectorGraph` SST body (RFC-030, `vector-index` feature).
//!
//! A `.vg` body is self-contained: it carries the indexed embeddings (f32, the
//! recall-golden representation) plus the Vamana search graph, so a read query
//! needs no extra object GETs to answer a top-k. The format is an 8-byte magic
//! + a bincode-serialised [`VectorGraphBody`]. Built during compaction from the
//! merged node rows ([`build_body`]); searched by decoding into a
//! [`VectorGraphIndex`] and calling [`VectorGraphIndex::search`].
//!
//! All three metrics are served from the index. The body stores the **original
//! (un-normalised) f32 vectors** plus a navigation graph; [`VectorGraphIndex::
//! search`] navigates with a metric-appropriate space and then **reranks the
//! candidates with the real metric**, so the returned score equals the flat
//! scan's `vector_score` exactly (to f32 tolerance): cosine similarity and raw
//! dot product (higher = closer), L2 distance (lower = closer). `cosine`/`dot`
//! navigate with cosine (scale-invariant — a fine, rank-correlated navigator for
//! dot); `euclidean` navigates with an L2 space (cosine would mis-rank whenever
//! magnitudes vary).

use bytes::Bytes;
use namidb_ann::{
    build_with_seed, search, BuildParams, F32CosineSpace, InitStrategy, L2Space, VamanaGraph,
};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

use crate::error::Error;
use crate::manifest::{VectorIndexDescriptor, VectorMetric};

/// On-disk magic + format major version (`NAMI` `VG` `\0` major=2). Major bumped
/// to 2: `dot` vectors are no longer normalised at build (the body now keeps the
/// original vectors and reranks with the true metric), and `euclidean` is now
/// indexable. A v1 reader would mis-score a v2 dot index and vice-versa, so the
/// magic forces a rebuild; the read path skips an undecodable `.vg` and falls
/// back to the flat scan rather than erroring.
const MAGIC: &[u8; 8] = b"NAMIVG02";

/// Canonical short metric name stored in the body / stats.
fn metric_name(m: VectorMetric) -> &'static str {
    match m {
        VectorMetric::Cosine => "cosine",
        VectorMetric::Dot => "dot",
        VectorMetric::Euclidean => "euclidean",
    }
}

/// The body of a `SstKind::VectorGraph` SST, bincode-serialised after the
/// 8-byte [`MAGIC`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorGraphBody {
    /// Embedding dimensionality.
    pub dim: u32,
    /// Canonical metric name (`"cosine"` / `"dot"` / `"euclidean"`).
    pub metric: String,
    /// `NodeId` per graph node `i`, parallel to `vectors` and the graph
    /// adjacency (`graph.adjacency[i]`).
    pub ids: Vec<[u8; 16]>,
    /// f32 embedding per graph node `i`.
    pub vectors: Vec<Vec<f32>>,
    /// The Vamana search graph.
    pub graph: VamanaGraph,
}

/// Stats harvested at build time, mirrored into
/// [`crate::manifest::KindSpecificStats::VectorGraph`].
#[derive(Debug, Clone)]
pub struct VectorGraphBuildStats {
    pub dim: u32,
    pub metric: String,
    pub point_count: u64,
    pub r: usize,
    pub l_build: usize,
    pub alpha: f32,
    pub entry_medoid: u32,
}

/// Metric-faithful score of stored vector `a` against `query`, computed in f64
/// to match the query engine's `vector_score`: returns `(value, higher_is_
/// better)`. Cosine similarity and raw dot product are higher-is-closer; L2
/// distance is lower-is-closer. This is the rerank applied to the navigation
/// candidates so the index's returned score equals the flat scan's.
fn metric_score(metric: VectorMetric, a: &[f32], query: &[f32]) -> (f64, bool) {
    match metric {
        VectorMetric::Cosine => {
            let dot: f64 = a
                .iter()
                .zip(query)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
            let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
            let nq: f64 = query
                .iter()
                .map(|x| *x as f64 * *x as f64)
                .sum::<f64>()
                .sqrt();
            if na == 0.0 || nq == 0.0 {
                (0.0, true)
            } else {
                (dot / (na * nq), true)
            }
        }
        VectorMetric::Dot => {
            let dot: f64 = a
                .iter()
                .zip(query)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
            (dot, true)
        }
        VectorMetric::Euclidean => {
            let s: f64 = a
                .iter()
                .zip(query)
                .map(|(x, y)| {
                    let d = *x as f64 - *y as f64;
                    d * d
                })
                .sum();
            (s.sqrt(), false)
        }
    }
}

/// Parse the canonical metric name stored in a `.vg` body back into the enum.
fn metric_from_name(name: &str) -> Option<VectorMetric> {
    match name {
        "cosine" => Some(VectorMetric::Cosine),
        "dot" => Some(VectorMetric::Dot),
        "euclidean" => Some(VectorMetric::Euclidean),
        _ => None,
    }
}

/// Build a `.vg` body from `(node_id, embedding)` pairs for one index.
///
/// Returns `Ok(None)` only when the set has fewer than 2 members — the caller
/// then skips emitting a VectorGraph SST and the query falls through to the flat
/// scan. All three metrics are indexable: `cosine`/`dot` navigate with cosine,
/// `euclidean` with an L2 space, and the original (un-normalised) vectors are
/// stored so search can rerank with the true metric.
pub fn build_body(
    desc: &VectorIndexDescriptor,
    mut members: Vec<([u8; 16], Vec<f32>)>,
) -> Result<Option<(Bytes, VectorGraphBuildStats)>, Error> {
    if members.len() < 2 {
        return Ok(None);
    }
    let dim = desc.dim as usize;
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(members.len());
    let mut ids: Vec<[u8; 16]> = Vec::with_capacity(members.len());
    for (id, v) in members.drain(..) {
        if v.len() != dim {
            return Err(Error::invariant(format!(
                "vector index `{}`: embedding dim {} != declared {}",
                desc.name,
                v.len(),
                dim
            )));
        }
        ids.push(id);
        vectors.push(v);
    }

    let params = BuildParams {
        r: desc.r,
        l_build: desc.l_build,
        alpha: desc.alpha,
        init: InitStrategy::Auto,
    };
    // Deterministic build: seed from the index name so two builds of the same
    // (data, descriptor) yield the same graph, while different indexes diverge.
    let seed = xxh3::xxh3_64(desc.name.as_bytes());
    // Navigate with a metric-appropriate space over the original vectors. Cosine
    // is scale-invariant, so it correctly navigates both cosine and dot; L2 is
    // required for euclidean (cosine ignores magnitude and would mis-rank).
    let graph = match desc.metric {
        VectorMetric::Euclidean => build_with_seed(&L2Space::new(vectors.clone()), params, seed),
        _ => build_with_seed(&F32CosineSpace::new(vectors.clone()), params, seed),
    };

    let stats = VectorGraphBuildStats {
        dim: desc.dim,
        metric: metric_name(desc.metric).to_string(),
        point_count: ids.len() as u64,
        r: desc.r,
        l_build: desc.l_build,
        alpha: desc.alpha,
        entry_medoid: graph.entry,
    };

    let body = VectorGraphBody {
        dim: desc.dim,
        metric: metric_name(desc.metric).to_string(),
        ids,
        vectors,
        graph,
    };
    let payload = bincode::serialize(&body)
        .map_err(|e| Error::invariant(format!("vector graph encode failed: {e}")))?;
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&payload);
    Ok(Some((Bytes::from(bytes), stats)))
}

/// The navigation space a decoded index uses to walk its Vamana graph. Cosine
/// for cosine/dot indexes, L2 for euclidean — matching the build.
#[derive(Debug)]
enum NavSpace {
    Cosine(F32CosineSpace),
    L2(L2Space),
}

/// A decoded, searchable VectorGraph index.
#[derive(Debug)]
pub struct VectorGraphIndex {
    body: VectorGraphBody,
    metric: VectorMetric,
    nav: NavSpace,
}

impl VectorGraphIndex {
    /// Decode a `.vg` body (magic + bincode). Errors on a truncated/foreign
    /// file, a magic mismatch (incl. a legacy v1 body), an unknown metric, or a
    /// graph whose entry point is out of range (a corrupt body — the body has no
    /// checksum). The read path treats any decode error as "index absent" and
    /// falls back to the flat scan, so this never panics a query.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < MAGIC.len() {
            return Err(Error::invariant("vector graph body too short for magic"));
        }
        let (magic, rest) = bytes.split_at(MAGIC.len());
        if magic != MAGIC {
            return Err(Error::invariant(format!(
                "vector graph magic mismatch: {:?}",
                magic
            )));
        }
        let body: VectorGraphBody = bincode::deserialize(rest)
            .map_err(|e| Error::invariant(format!("vector graph decode failed: {e}")))?;
        let metric = metric_from_name(&body.metric).ok_or_else(|| {
            Error::invariant(format!("vector graph unknown metric: {}", body.metric))
        })?;
        // Validate the graph's internal consistency before trusting it: the entry
        // point and every id must be in range, since the body carries no checksum
        // and search indexes adjacency/vectors by id directly.
        let n = body.vectors.len();
        if n != body.ids.len() || n != body.graph.adjacency.len() {
            return Err(Error::invariant("vector graph body length mismatch"));
        }
        if n > 0 && body.graph.entry as usize >= n {
            return Err(Error::invariant("vector graph entry out of range"));
        }
        let nav = match metric {
            VectorMetric::Euclidean => NavSpace::L2(L2Space::new(body.vectors.clone())),
            _ => NavSpace::Cosine(F32CosineSpace::new(body.vectors.clone())),
        };
        Ok(Self { body, metric, nav })
    }

    /// Number of vectors indexed.
    pub fn point_count(&self) -> u64 {
        self.body.ids.len() as u64
    }

    /// Dimensionality.
    pub fn dim(&self) -> u32 {
        self.body.dim
    }

    /// Metric name (`"cosine"` / `"dot"` / `"euclidean"`).
    pub fn metric(&self) -> &str {
        &self.body.metric
    }

    /// `true` when a higher score means a closer match (cosine / dot); `false`
    /// for euclidean, where lower (distance) is closer. The caller uses this to
    /// orient a multi-SST union / delta merge.
    pub fn higher_is_better(&self) -> bool {
        !matches!(self.metric, VectorMetric::Euclidean)
    }

    /// Approximate top-`k` nearest to `query`, returning `(NodeId, score)` pairs
    /// sorted best-first. `score` is **metric-faithful**: cosine similarity or
    /// raw dot product (higher = closer), or L2 distance (lower = closer) — equal
    /// to the flat scan's `vector_score` to f32 tolerance. `ef` is the beam width
    /// (≥ `k`; larger → better recall, more work). The graph is navigated with
    /// the metric's navigation space to gather up to `ef` candidates, which are
    /// then reranked by the true metric from the original f32 vectors, so the
    /// approximation is only in *which* nodes the graph visits, not the score.
    ///
    /// Returns an empty vec when `query`'s dimensionality does not match the
    /// index's (the caller falls back to the flat scan, which raises the
    /// canonical dimension-mismatch error) — never a prefix-scored wrong answer.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<([u8; 16], f32)> {
        if k == 0 || query.len() != self.body.dim as usize {
            return Vec::new();
        }
        let ef = ef.max(k);
        // Navigate for up to `ef` candidates (k = ef), then rerank by the true
        // metric. For dot the navigation metric (cosine) differs from the score
        // metric, so the wider candidate pool is what lets the rerank surface the
        // true dot-nearest; for cosine/euclidean navigation already is the metric.
        let cands = match &self.nav {
            NavSpace::Cosine(s) => search(s, &self.body.graph, query, ef, ef),
            NavSpace::L2(s) => search(s, &self.body.graph, query, ef, ef),
        };
        let mut scored: Vec<([u8; 16], f32)> = cands
            .into_iter()
            .map(|nb| {
                let v = &self.body.vectors[nb.id as usize];
                let (score, _hib) = metric_score(self.metric, v, query);
                (self.body.ids[nb.id as usize], score as f32)
            })
            .collect();
        if self.higher_is_better() {
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        } else {
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// L2-normalise in place (test fixtures build unit vectors).
    fn normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v {
                *x /= n;
            }
        }
    }

    fn desc(name: &str, metric: VectorMetric, dim: u32) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: name.into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim,
            metric,
            r: 16,
            l_build: 32,
            alpha: 1.2,
        }
    }

    fn clustered_members(n: usize, dim: usize, seed: u64) -> Vec<([u8; 16], Vec<f32>)> {
        // 4 well-separated centroids; members perturbed around them.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        use rand::Rng;
        let mut centroids: Vec<Vec<f32>> = Vec::new();
        for _ in 0..4 {
            let mut c = vec![0.0f32; dim];
            for x in &mut c {
                *x = rng.gen();
            }
            centroids.push(c);
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = &centroids[i % 4];
            let mut v: Vec<f32> = base.iter().map(|b| b + 0.02 * rng.gen::<f32>()).collect();
            normalize(&mut v);
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&(i as u64).to_be_bytes());
            out.push((id, v));
        }
        out
    }

    #[test]
    fn tiny_sets_are_not_indexed() {
        // Fewer than 2 members → no graph (caller keeps the flat scan).
        let d = desc("c", VectorMetric::Cosine, 4);
        assert!(build_body(&d, clustered_members(1, 4, 2))
            .unwrap()
            .is_none());
    }

    #[test]
    fn all_three_metrics_are_indexable_and_score_faithfully() {
        // Every metric now produces a `.vg`, and the returned score equals the
        // engine's metric (cosine sim / raw dot / L2 distance) to f32 tolerance.
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let d = desc("m", metric, 8);
            let members = clustered_members(60, 8, 11);
            let (body, _) = build_body(&d, members.clone()).unwrap().unwrap();
            let idx = VectorGraphIndex::decode(&body).unwrap();
            assert_eq!(idx.higher_is_better(), metric != VectorMetric::Euclidean);

            let query = members[3].1.clone();
            let hits = idx.search(&query, 5, 32);
            assert!(!hits.is_empty(), "{metric:?} produced no hits");
            // Best-first ordering matches the metric orientation.
            for w in hits.windows(2) {
                if metric == VectorMetric::Euclidean {
                    assert!(w[0].1 <= w[1].1 + 1e-5, "{metric:?} not asc: {hits:?}");
                } else {
                    assert!(w[0].1 >= w[1].1 - 1e-5, "{metric:?} not desc: {hits:?}");
                }
            }
            // The top score equals a direct metric computation on the same id.
            let (top_id, top_score) = hits[0];
            let top_vec = members
                .iter()
                .find(|(id, _)| *id == top_id)
                .map(|(_, v)| v.clone())
                .unwrap();
            let (want, _) = metric_score(metric, &top_vec, &query);
            assert!(
                (want as f32 - top_score).abs() < 1e-4,
                "{metric:?}: index score {top_score} != metric {want}"
            );
        }
    }

    #[test]
    fn decode_rejects_out_of_range_entry() {
        // A corrupt body with an entry past the graph size is rejected, not
        // trusted into a panic on search.
        let d = desc("x", VectorMetric::Cosine, 4);
        let (body, _) = build_body(&d, clustered_members(10, 4, 3))
            .unwrap()
            .unwrap();
        // Decode, corrupt the entry, re-encode, and assert decode rejects it.
        let (_, rest) = body.split_at(MAGIC.len());
        let mut decoded: VectorGraphBody = bincode::deserialize(rest).unwrap();
        decoded.graph.entry = 9999;
        let mut bad = MAGIC.to_vec();
        bad.extend_from_slice(&bincode::serialize(&decoded).unwrap());
        assert!(VectorGraphIndex::decode(&bad).is_err());
    }

    #[test]
    fn build_decode_search_round_trip() {
        let d = desc("docs", VectorMetric::Cosine, 16);
        let members = clustered_members(200, 16, 7);
        let (body, stats) = build_body(&d, members.clone()).unwrap().unwrap();
        assert_eq!(stats.point_count, 200);
        assert_eq!(stats.metric, "cosine");

        let idx = VectorGraphIndex::decode(&body).unwrap();
        assert_eq!(idx.point_count(), 200);
        assert_eq!(idx.dim(), 16);

        // Query near cluster 0's centroid → top hits should be cluster-0 ids.
        let q = {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
            let mut v: Vec<f32> = (0..16).map(|_| 0.02 * rng.gen::<f32>()).collect();
            normalize(&mut v);
            v
        };
        let hits = idx.search(&q, 10, 32);
        assert_eq!(hits.len(), 10);
        // Best-first: similarities non-increasing.
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1 - 1e-5, "not sorted: {:?}", hits);
        }
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let err = VectorGraphIndex::decode(b"XXXXXXXxyy");
        assert!(err.is_err());
        let err = VectorGraphIndex::decode(b"");
        assert!(err.is_err());
    }

    #[test]
    fn recall_on_indexed_clustered_set_is_high() {
        // Same fixture as the namidb-ann recall test, end-to-end through the
        // SST body encode/decode: indexed recall@10 should track brute force.
        let n = 400;
        let dim = 32;
        let members = clustered_members(n, dim, 31);
        let mut d = desc("recall", VectorMetric::Cosine, dim as u32);
        // DiskANN-ish defaults for a few-hundred-point set: enough degree and
        // beam to clear a high recall floor.
        d.r = 32;
        d.l_build = 64;
        let (body, _) = build_body(&d, members.clone()).unwrap().unwrap();
        let idx = VectorGraphIndex::decode(&body).unwrap();

        let k = 10;
        let mut total = 0.0;
        for q in 0..30 {
            let query = members[q % 50].1.clone();
            // Brute-force truth (cosine similarity, top-k ids by id bytes).
            let mut scored: Vec<(f64, [u8; 16])> = members
                .iter()
                .map(|(id, v)| (cosine(&query, v), *id))
                .collect();
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let truth: std::collections::HashSet<[u8; 16]> =
                scored.iter().take(k).map(|(_, id)| *id).collect();
            let approx: std::collections::HashSet<[u8; 16]> = idx
                .search(&query, k, 64)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let hits = approx.intersection(&truth).count();
            total += hits as f64 / k as f64;
        }
        let avg = total / 30.0;
        assert!(
            avg >= 0.85,
            "indexed recall@{k} = {avg:.3}, expected >= 0.85"
        );
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}
