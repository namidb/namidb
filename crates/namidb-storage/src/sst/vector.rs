//! DiskANN/Vamana `VectorGraph` SST body (RFC-030, `vector-index` feature).
//!
//! A `.vg` body is self-contained: it carries the indexed embeddings (f32, the
//! recall-golden representation) plus the Vamana search graph, so a read query
//! needs no extra object GETs to answer a top-k. The format is an 8-byte magic
//! + a bincode-serialised [`VectorGraphBody`]. Built during compaction from the
//! merged node rows ([`build_body`]); searched by decoding into a
//! [`VectorGraphIndex`] and calling [`VectorGraphIndex::search`].
//!
//! `cosine` and `dot` are both served by an f32 cosine space (for `dot` the
//! vectors are L2-normalised at build time, so dot ranking coincides with
//! cosine ranking). `euclidean` is **not** indexable here — [`build_body`]
//! returns `Ok(None)` for it so the caller keeps the flat-scan fallback.

use bytes::Bytes;
use namidb_ann::{build_with_seed, search, BuildParams, F32CosineSpace, InitStrategy, VamanaGraph};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

use crate::error::Error;
use crate::manifest::{VectorIndexDescriptor, VectorMetric};

/// On-disk magic + format major version (`NAMI` `VG` `\0` major=1). Bumped on
/// any incompatible layout change to the body below so a reader never silently
/// misparses a future/legacy file.
const MAGIC: &[u8; 8] = b"NAMIVG01";

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

/// L2-normalise `v` in place (no-op for a zero vector). Dot-product ranking on
/// unit vectors coincides with cosine ranking, so normalising lets one cosine
/// space serve both metrics.
fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v {
            *x /= n;
        }
    }
}

/// Build a `.vg` body from `(node_id, embedding)` pairs for one index.
///
/// Returns `Ok(None)` when the metric is not indexable here (`euclidean`) or
/// the set has fewer than 2 members — the caller then skips emitting a
/// VectorGraph SST and the query falls through to the flat scan.
pub fn build_body(
    desc: &VectorIndexDescriptor,
    mut members: Vec<([u8; 16], Vec<f32>)>,
) -> Result<Option<(Bytes, VectorGraphBuildStats)>, Error> {
    if desc.metric == VectorMetric::Euclidean {
        return Ok(None);
    }
    if members.len() < 2 {
        return Ok(None);
    }
    let dim = desc.dim as usize;
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(members.len());
    let mut ids: Vec<[u8; 16]> = Vec::with_capacity(members.len());
    for (id, mut v) in members.drain(..) {
        if v.len() != dim {
            return Err(Error::invariant(format!(
                "vector index `{}`: embedding dim {} != declared {}",
                desc.name,
                v.len(),
                dim
            )));
        }
        if desc.metric == VectorMetric::Dot {
            normalize(&mut v);
        }
        ids.push(id);
        vectors.push(v);
    }

    let space = F32CosineSpace::new(vectors.clone());
    let params = BuildParams {
        r: desc.r,
        l_build: desc.l_build,
        alpha: desc.alpha,
        init: InitStrategy::Auto,
    };
    // Deterministic build: seed from the index name so two builds of the same
    // (data, descriptor) yield the same graph, while different indexes diverge.
    let seed = xxh3::xxh3_64(desc.name.as_bytes());
    let graph = build_with_seed(&space, params, seed);

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

/// A decoded, searchable VectorGraph index.
#[derive(Debug)]
pub struct VectorGraphIndex {
    body: VectorGraphBody,
    space: F32CosineSpace,
}

impl VectorGraphIndex {
    /// Decode a `.vg` body (magic + bincode). Errors on a truncated/foreign
    /// file or a magic mismatch.
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
        let space = F32CosineSpace::new(body.vectors.clone());
        Ok(Self { body, space })
    }

    /// Number of vectors indexed.
    pub fn point_count(&self) -> u64 {
        self.body.ids.len() as u64
    }

    /// Dimensionality.
    pub fn dim(&self) -> u32 {
        self.body.dim
    }

    /// Metric name (`"cosine"` / `"dot"`).
    pub fn metric(&self) -> &str {
        &self.body.metric
    }

    /// Approximate top-`k` nearest to `query`, returning `(NodeId, score)`
    /// pairs sorted best-first. `score` is the similarity (higher = closer) for
    /// cosine/dot; `ef` is the search beam width (≥ `k`; larger → better
    /// recall, more work). Candidates are full-precision reranked from the
    /// stored f32 vectors, so the returned scores are exact for the stored
    /// representation (the approximation is only in *which* nodes the graph
    /// visits, not in the score math).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<([u8; 16], f32)> {
        let ef = ef.max(k);
        let hits = search(&self.space, &self.body.graph, query, k, ef);
        hits.into_iter()
            .map(|nb| {
                // The graph distance is cosine distance (1 - similarity); flip
                // back to similarity so higher = closer (matches the builtins).
                let sim = 1.0 - nb.dist;
                (self.body.ids[nb.id as usize], sim)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

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
    fn euclidean_and_tiny_sets_are_not_indexed() {
        let d = desc("e", VectorMetric::Euclidean, 4);
        let m = clustered_members(50, 4, 1);
        assert!(build_body(&d, m).unwrap().is_none());

        let d = desc("c", VectorMetric::Cosine, 4);
        assert!(build_body(&d, clustered_members(1, 4, 2))
            .unwrap()
            .is_none());
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
