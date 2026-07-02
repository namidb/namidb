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
//! dot product (higher = closer), L2 distance (lower = closer). `cosine`
//! navigates with cosine; `dot` navigates with cosine over **MIPS-augmented**
//! vectors (see [`mips_augment`] — plain cosine is magnitude-blind and misses
//! the large-norm vectors that dominate a true inner-product top-k);
//! `euclidean` navigates with an L2 space (cosine would mis-rank whenever
//! magnitudes vary).

use bytes::Bytes;
use namidb_ann::{
    build_with_seed, search, BuildParams, F32CosineSpace, InitStrategy, Int8Space, L2Space,
    VamanaGraph,
};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

use namidb_core::quantize::quantize_i8;

use crate::error::Error;
use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};

/// On-disk magic + format major version (`NAMI` `VG` `\0` major=3). v2 stored the
/// original f32 vectors and reranked with the true metric (all three metrics
/// indexable); v3 generalises the vector store to f32 OR per-vector int8 codes
/// (`quantization: int8`, ~4× smaller, cosine-only). A reader of an older body
/// mismatches the magic; the read path skips an undecodable `.vg` and falls back
/// to the flat scan rather than erroring.
const MAGIC: &[u8; 8] = b"NAMIVG03";

/// Canonical short metric name stored in the body / stats.
fn metric_name(m: VectorMetric) -> &'static str {
    match m {
        VectorMetric::Cosine => "cosine",
        VectorMetric::Dot => "dot",
        VectorMetric::Euclidean => "euclidean",
    }
}

/// The stored vectors inside a `.vg` body — full f32, or per-vector int8 codes
/// plus a scale (`x_i ≈ codes_i · scale`), one entry per graph node `i`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorStorage {
    F32(Vec<Vec<f32>>),
    Int8 {
        codes: Vec<Vec<i8>>,
        scales: Vec<f32>,
    },
}

impl VectorStorage {
    /// Number of stored vectors.
    fn len(&self) -> usize {
        match self {
            VectorStorage::F32(v) => v.len(),
            VectorStorage::Int8 { codes, .. } => codes.len(),
        }
    }
    /// The vector for node `i` materialised as f32 (dequantising int8).
    fn f32_at(&self, i: usize) -> Vec<f32> {
        match self {
            VectorStorage::F32(v) => v[i].clone(),
            VectorStorage::Int8 { codes, scales } => {
                codes[i].iter().map(|&c| c as f32 * scales[i]).collect()
            }
        }
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
    /// `NodeId` per graph node `i`, parallel to `storage` and the graph
    /// adjacency (`graph.adjacency[i]`).
    pub ids: Vec<[u8; 16]>,
    /// f32 or int8-quantised embedding per graph node `i`.
    pub storage: VectorStorage,
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

/// Parse the metric name stored in a `.vg` body back into the enum plus
/// whether the navigation graph was built over MIPS-augmented vectors.
/// `"dot"` is a legacy body whose graph was built with plain cosine over the
/// raw vectors (magnitude-blind — poor recall when norms vary); `"dot-mips"`
/// marks the current reduction, so old bodies keep working until the next
/// authoritative compaction rebuilds them.
fn metric_from_name(name: &str) -> Option<(VectorMetric, bool)> {
    match name {
        "cosine" => Some((VectorMetric::Cosine, false)),
        "dot" => Some((VectorMetric::Dot, false)),
        "dot-mips" => Some((VectorMetric::Dot, true)),
        "euclidean" => Some((VectorMetric::Euclidean, false)),
        _ => None,
    }
}

/// Bachrach et al. (2014) MIPS→cosine reduction: append `sqrt(M² − ‖x‖²)`
/// to every vector (`M` = max corpus norm), making them all norm `M`. Against
/// a zero-augmented query, cosine over the augmented set orders EXACTLY by
/// inner product — so a Vamana graph built/navigated with cosine on the
/// augmented vectors surfaces the true dot-nearest candidates, magnitudes
/// included. Plain cosine navigation is magnitude-blind, and dot's top-k is
/// dominated by large-norm vectors — exactly the case users pick `dot` for.
fn mips_augment(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let max_sq = vectors
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>())
        .fold(0.0f32, f32::max);
    vectors
        .iter()
        .map(|v| {
            let sq: f32 = v.iter().map(|x| x * x).sum();
            let mut a = Vec::with_capacity(v.len() + 1);
            a.extend_from_slice(v);
            a.push((max_sq - sq).max(0.0).sqrt());
            a
        })
        .collect()
}

/// The navigation query for a MIPS-augmented graph: the raw query with a 0
/// appended (its dot with the augmentation coordinate vanishes, leaving the
/// pure inner product in the cosine numerator).
fn mips_query(query: &[f32]) -> Vec<f32> {
    let mut q = Vec::with_capacity(query.len() + 1);
    q.extend_from_slice(query);
    q.push(0.0);
    q
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
        // A zero-norm (all-zero) vector is not cosine-rankable — the flat scan's
        // `vector_score(Cosine, …)` returns None and drops it — so exclude it from
        // a cosine index too, keeping the indexed corpus equal to the flat scan's.
        // (Dot and L2 are well-defined on the zero vector, so keep it there.)
        if desc.metric == VectorMetric::Cosine && v.iter().all(|x| *x == 0.0) {
            continue;
        }
        ids.push(id);
        vectors.push(v);
    }
    // Fewer than 2 indexable members after filtering → no graph (flat scan).
    if vectors.len() < 2 {
        return Ok(None);
    }

    // int8 quantization is cosine-only (the scale-invariant Int8Space). Reject a
    // misconfigured index loudly rather than silently building a wrong one.
    if desc.quantization == VectorQuantization::Int8 && desc.metric != VectorMetric::Cosine {
        return Err(Error::invariant(format!(
            "vector index `{}`: int8 quantization requires metric cosine (got {})",
            desc.name,
            metric_name(desc.metric)
        )));
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

    // Navigate, and choose the on-disk store, per quantization + metric. int8
    // quantizes per-vector and navigates/scores with the scale-invariant cosine
    // Int8Space (~4× smaller body). f32 keeps the original vectors and navigates
    // with cosine (cosine/dot) or L2 (euclidean), reranking with the true metric.
    let (graph, storage) = match desc.quantization {
        VectorQuantization::Int8 => {
            let members8: Vec<(Vec<i8>, f32)> = vectors.iter().map(|v| quantize_i8(v)).collect();
            let graph = build_with_seed(&Int8Space::new(members8.clone()), params, seed);
            let (codes, scales) = members8.into_iter().unzip();
            (graph, VectorStorage::Int8 { codes, scales })
        }
        VectorQuantization::None => {
            let graph = match desc.metric {
                VectorMetric::Euclidean => {
                    build_with_seed(&L2Space::new(vectors.clone()), params, seed)
                }
                // MIPS: build the graph over the augmented vectors so cosine
                // navigation orders by true inner product (see mips_augment);
                // the body stores the ORIGINALS for the exact rerank.
                VectorMetric::Dot => {
                    build_with_seed(&F32CosineSpace::new(mips_augment(&vectors)), params, seed)
                }
                VectorMetric::Cosine => {
                    build_with_seed(&F32CosineSpace::new(vectors.clone()), params, seed)
                }
            };
            (graph, VectorStorage::F32(vectors))
        }
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

    // The body's metric string doubles as the navigation-geometry marker:
    // a dot graph built over MIPS-augmented vectors is tagged "dot-mips" so
    // decode() knows to augment (legacy "dot" bodies keep plain-cosine
    // navigation until an authoritative compaction rebuilds them). The
    // descriptor-facing stats keep the canonical "dot".
    let body_metric = if desc.metric == VectorMetric::Dot {
        "dot-mips".to_string()
    } else {
        metric_name(desc.metric).to_string()
    };
    let body = VectorGraphBody {
        dim: desc.dim,
        metric: body_metric,
        ids,
        storage,
        graph,
    };
    let payload = bincode::serialize(&body)
        .map_err(|e| Error::invariant(format!("vector graph encode failed: {e}")))?;
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&payload);
    Ok(Some((Bytes::from(bytes), stats)))
}

/// The navigation space a decoded index uses to walk its Vamana graph. Cosine
/// for f32 cosine/dot indexes, L2 for f32 euclidean, Int8 for a quantized
/// (cosine-only) index — matching the build.
#[derive(Debug)]
enum NavSpace {
    Cosine(F32CosineSpace),
    L2(L2Space),
    Int8(Int8Space),
}

/// A decoded, searchable VectorGraph index.
#[derive(Debug)]
pub struct VectorGraphIndex {
    body: VectorGraphBody,
    metric: VectorMetric,
    nav: NavSpace,
    /// The graph was built over MIPS-augmented vectors ("dot-mips"): navigate
    /// with the zero-augmented query, not the raw one.
    mips: bool,
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
        let (metric, mips) = metric_from_name(&body.metric).ok_or_else(|| {
            Error::invariant(format!("vector graph unknown metric: {}", body.metric))
        })?;
        // Validate the graph's internal consistency before trusting it: the entry
        // point and every id must be in range, since the body carries no checksum
        // and search indexes adjacency/storage by id directly.
        let n = body.storage.len();
        if n != body.ids.len() || n != body.graph.adjacency.len() {
            return Err(Error::invariant("vector graph body length mismatch"));
        }
        if n > 0 && body.graph.entry as usize >= n {
            return Err(Error::invariant("vector graph entry out of range"));
        }
        let nav = match &body.storage {
            VectorStorage::Int8 { codes, scales } => {
                if codes.len() != scales.len() {
                    return Err(Error::invariant("vector graph int8 codes/scales mismatch"));
                }
                let members: Vec<(Vec<i8>, f32)> =
                    codes.iter().cloned().zip(scales.iter().copied()).collect();
                NavSpace::Int8(Int8Space::new(members))
            }
            VectorStorage::F32(v) if metric == VectorMetric::Euclidean => {
                NavSpace::L2(L2Space::new(v.clone()))
            }
            // dot-mips: rebuild the augmentation the graph was constructed
            // over (deterministic from the stored originals, so no format
            // field is needed).
            VectorStorage::F32(v) if mips => NavSpace::Cosine(F32CosineSpace::new(mips_augment(v))),
            VectorStorage::F32(v) => NavSpace::Cosine(F32CosineSpace::new(v.clone())),
        };
        Ok(Self {
            body,
            metric,
            nav,
            mips,
        })
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
        // Navigate for up to `ef` candidates (k = ef), then score them. For f32
        // we rerank by the TRUE metric from the original vectors (for dot the
        // navigation metric — cosine — differs, so the wider pool surfaces the
        // true dot-nearest); for int8 the navigation distance already IS the
        // (cosine-only) score, so we just flip distance → similarity.
        // dot-mips navigates in the augmented space (query gains a zero
        // coordinate); every other geometry navigates with the raw query.
        let nav_query: Vec<f32>;
        let nq: &[f32] = if self.mips {
            nav_query = mips_query(query);
            &nav_query
        } else {
            query
        };
        let cands = match &self.nav {
            NavSpace::Cosine(s) => search(s, &self.body.graph, nq, ef, ef),
            NavSpace::L2(s) => search(s, &self.body.graph, nq, ef, ef),
            NavSpace::Int8(s) => search(s, &self.body.graph, nq, ef, ef),
        };
        let is_int8 = matches!(self.nav, NavSpace::Int8(_));
        let mut scored: Vec<([u8; 16], f32)> = cands
            .into_iter()
            .map(|nb| {
                let score = if is_int8 {
                    // int8 cosine similarity (the stored, quantized score).
                    1.0 - nb.dist
                } else {
                    let v = self.body.storage.f32_at(nb.id as usize);
                    metric_score(self.metric, &v, query).0 as f32
                };
                (self.body.ids[nb.id as usize], score)
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
        desc_q(name, metric, dim, VectorQuantization::None)
    }

    fn desc_q(
        name: &str,
        metric: VectorMetric,
        dim: u32,
        quantization: VectorQuantization,
    ) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: name.into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim,
            metric,
            r: 16,
            l_build: 32,
            alpha: 1.2,
            quantization,
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
    fn dot_index_surfaces_large_norm_vectors_beyond_the_cosine_beam() {
        // Adversarial MIPS fixture: 200 small-norm vectors (0.5–1.5) biased
        // toward the query direction (cosine ≈ 0.3–0.95, so their dot tops out
        // ~1.4) plus ONE norm-10 vector at ~80° (dot ≈ 1.74 — the true top-1,
        // but cosine rank near dead last). A cosine-navigated graph fills its
        // ef=64 beam with the small cluster and never even reranks the
        // big-norm vector; MIPS-augmented navigation must put it first.
        let dim = 8usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut members: Vec<([u8; 16], Vec<f32>)> = Vec::new();
        for i in 0..200u64 {
            let mut v = vec![0.0f32; dim];
            v[0] = 0.5;
            for x in v.iter_mut().skip(1) {
                *x = rng.gen_range(-0.5..0.5);
            }
            normalize(&mut v);
            let norm = rng.gen_range(0.5..1.5);
            for x in v.iter_mut() {
                *x *= norm;
            }
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&i.to_be_bytes());
            members.push((id, v));
        }
        let mut big = vec![0.0f32; dim];
        big[0] = (80.0f32).to_radians().cos() * 10.0;
        big[1] = (80.0f32).to_radians().sin() * 10.0;
        let big_id = [0xBB; 16];
        members.push((big_id, big.clone()));

        let d = desc("mips", VectorMetric::Dot, dim as u32);
        let (body, _) = build_body(&d, members).unwrap().unwrap();
        let idx = VectorGraphIndex::decode(&body).unwrap();
        assert_eq!(idx.metric(), "dot-mips");

        let query = {
            let mut q = vec![0.0f32; dim];
            q[0] = 1.0;
            q
        };
        let hits = idx.search(&query, 5, 64);
        assert_eq!(
            hits[0].0, big_id,
            "true dot top-1 (norm 10, dot ≈ 1.74) must surface: {hits:?}"
        );
        let want = big[0] as f64; // dot(query, big) = big[0]
        assert!(
            (hits[0].1 as f64 - want).abs() < 1e-4,
            "score is the exact dot: {} vs {want}",
            hits[0].1
        );
    }

    #[test]
    fn legacy_plain_dot_bodies_still_decode() {
        // A pre-MIPS body carries metric "dot": it must decode and search
        // (plain-cosine navigation) rather than being rejected, so existing
        // indexes keep serving until compaction rebuilds them.
        let d = desc("legacy", VectorMetric::Dot, 8);
        let (bytes, _) = build_body(&d, clustered_members(40, 8, 3)).unwrap().unwrap();
        // Rewrite the body's metric tag to the legacy name.
        let mut body: VectorGraphBody = bincode::deserialize(&bytes[MAGIC.len()..]).unwrap();
        body.metric = "dot".to_string();
        let mut legacy = MAGIC.to_vec();
        legacy.extend_from_slice(&bincode::serialize(&body).unwrap());
        let idx = VectorGraphIndex::decode(&legacy).unwrap();
        assert_eq!(idx.metric(), "dot");
        assert!(!idx.search(&clustered_members(1, 8, 3)[0].1, 3, 16).is_empty());
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

    /// Clustered unit vectors with enough spread that the top-k is well-defined
    /// (the tight `clustered_members` fixture makes near-duplicates whose top-k
    /// is noise — useless for a recall measurement). Mirrors the `namidb-ann`
    /// int8 recall fixture (spread 0.15).
    fn spread_members(
        n: usize,
        dim: usize,
        clusters: usize,
        spread: f32,
        seed: u64,
    ) -> Vec<([u8; 16], Vec<f32>)> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let cents: Vec<Vec<f32>> = (0..clusters)
            .map(|_| {
                let mut c: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                normalize(&mut c);
                c
            })
            .collect();
        (0..n)
            .map(|i| {
                let base = &cents[i % clusters];
                let mut v: Vec<f32> = base
                    .iter()
                    .map(|b| b + spread * (rng.gen::<f32>() * 2.0 - 1.0))
                    .collect();
                normalize(&mut v);
                let mut id = [0u8; 16];
                id[0..8].copy_from_slice(&(i as u64).to_be_bytes());
                (id, v)
            })
            .collect()
    }

    #[test]
    fn int8_index_is_smaller_and_recalls_well() {
        // int8 quantization makes the body materially smaller while keeping
        // recall above the documented floor on well-separated data.
        let n = 400;
        let dim = 64;
        let members = spread_members(n, dim, 16, 0.15, 17);
        let f32_d = desc("f32", VectorMetric::Cosine, dim as u32);
        let int8_d = desc_q(
            "int8",
            VectorMetric::Cosine,
            dim as u32,
            VectorQuantization::Int8,
        );
        let (f32_body, _) = build_body(&f32_d, members.clone()).unwrap().unwrap();
        let (int8_body, stats) = build_body(&int8_d, members.clone()).unwrap().unwrap();
        assert_eq!(stats.point_count, n as u64);
        assert!(
            int8_body.len() < f32_body.len(),
            "int8 body {} should be smaller than f32 body {}",
            int8_body.len(),
            f32_body.len()
        );

        let idx = VectorGraphIndex::decode(&int8_body).unwrap();
        assert_eq!(idx.point_count(), n as u64);
        let k = 10;
        let mut total = 0.0;
        for q in 0..30 {
            let query = members[q % 50].1.clone();
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
            total += approx.intersection(&truth).count() as f64 / k as f64;
        }
        let avg = total / 30.0;
        assert!(avg >= 0.80, "int8 recall@{k} = {avg:.3}, expected >= 0.80");
    }

    #[test]
    fn int8_requires_cosine_metric() {
        // int8 + a non-cosine metric is rejected at build (not silently wrong).
        let d = desc_q("bad", VectorMetric::Dot, 8, VectorQuantization::Int8);
        assert!(build_body(&d, clustered_members(10, 8, 1)).is_err());
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
