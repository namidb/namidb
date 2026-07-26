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

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use bincode::Options;
use bytes::Bytes;
use namidb_ann::{
    build_with_seed, search, BuildParams, F32CosineSpace, InitStrategy, Int8Space, L2Space,
    VamanaGraph,
};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

use namidb_core::quantize::quantize_i8;
use namidb_core::Value;

use crate::error::Error;
use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};

/// Current on-disk magic. v4 adds bounded typed metadata postings, mapping
/// `(property, ScalarV1 value)` to sorted vector ordinals. The postings power
/// native pre-filtering without hydrating the matching node corpus.
const MAGIC: &[u8; 8] = b"NAMIVG04";
/// v3 body (same vector/graph representation, no metadata postings). It remains
/// readable: filtered queries simply report "native filter unsupported" and use
/// the existing adaptive widening/exact fallback until compaction emits v4.
const LEGACY_MAGIC_V3: &[u8; 8] = b"NAMIVG03";

/// Default maximum number of distinct values retained for one vector-filter
/// property. High-cardinality keys (document ids, URLs) are deliberately left
/// to the equality sidecar/residual path rather than bloating every `.vg`.
pub const DEFAULT_VECTOR_FILTER_MAX_DISTINCT: usize = 4_096;
/// Default aggregate estimated bytes retained for all metadata postings in one
/// vector index body.
pub const DEFAULT_VECTOR_FILTER_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Monotonic process counter for `.vg` queries that applied at least one
/// persisted metadata posting. Exposed for production telemetry and regression
/// tests: a selective filtered query should increment this without first
/// hydrating a corpus-sized NodeId set.
static VECTOR_FILTER_BITMAP_SEARCHES: AtomicU64 = AtomicU64::new(0);

/// Number of searches that applied at least one embedded vector-filter group.
pub fn vector_filter_bitmap_searches() -> u64 {
    VECTOR_FILTER_BITMAP_SEARCHES.load(Ordering::Relaxed)
}

/// Build-time bounds for native vector metadata postings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VectorFilterLimits {
    pub max_distinct_per_property: usize,
    pub max_bytes: usize,
}

impl Default for VectorFilterLimits {
    fn default() -> Self {
        Self {
            max_distinct_per_property: DEFAULT_VECTOR_FILTER_MAX_DISTINCT,
            max_bytes: DEFAULT_VECTOR_FILTER_MAX_BYTES,
        }
    }
}

impl VectorFilterLimits {
    pub(crate) fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            max_distinct_per_property: std::env::var("NAMIDB_VECTOR_FILTER_MAX_DISTINCT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(defaults.max_distinct_per_property),
            max_bytes: std::env::var("NAMIDB_VECTOR_FILTER_MAX_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(defaults.max_bytes),
        }
    }
}

/// Adaptive posting encoded directly in a v4 `.vg`.
///
/// Sparse values keep sorted vector ordinals; dense values keep a bit per
/// vector ordinal. The builder switches representations while it streams the
/// authoritative corpus, so a Boolean posting over 750k vectors occupies about
/// 92 KiB instead of a 1.5 MiB `Vec<u32>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum VectorFilterPosting {
    Sparse(Vec<u32>),
    Dense(Vec<u64>),
}

impl VectorFilterPosting {
    fn from_sorted_ordinals(ordinals: Vec<u32>) -> Self {
        let Some(&last) = ordinals.last() else {
            return Self::Sparse(ordinals);
        };
        let words = (last as usize + 1).div_ceil(64);
        // Use the intentionally conservative N/64 crossover requested for the
        // filter path: once there is more than one hit per bitmap word, dense
        // query-time OR/intersection is preferable to replaying every ordinal.
        if ordinals.len() > words {
            let mut dense = vec![0u64; words];
            for ordinal in ordinals {
                let ordinal = ordinal as usize;
                dense[ordinal / 64] |= 1u64 << (ordinal % 64);
            }
            Self::Dense(dense)
        } else {
            Self::Sparse(ordinals)
        }
    }

    fn into_sorted_ordinals(self) -> Vec<u32> {
        match self {
            Self::Sparse(ordinals) => ordinals,
            Self::Dense(words) => {
                let mut out = Vec::new();
                for (word_index, mut word) in words.into_iter().enumerate() {
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        out.push((word_index * 64 + bit) as u32);
                        word &= word - 1;
                    }
                }
                out
            }
        }
    }

    fn payload_bytes(&self) -> usize {
        match self {
            Self::Sparse(ordinals) => ordinals.len().saturating_mul(std::mem::size_of::<u32>()),
            Self::Dense(words) => words.len().saturating_mul(std::mem::size_of::<u64>()),
        }
    }
}

/// `(property -> ScalarV1 key -> complete adaptive ordinal posting)`.
pub(crate) type VectorFilterPostings = BTreeMap<String, BTreeMap<String, VectorFilterPosting>>;

/// Vector-v4-local typed key encoding.
///
/// Equality sidecars keep raw String keys for rolling compatibility with old
/// readers. Old readers reject `NAMIVG04` and flat-fall-back, so this new body
/// can retain a String tag and avoid `Str("b:1")` colliding with `Bool(true)`.
fn encode_vector_filter_value(value: &Value) -> Option<Cow<'_, str>> {
    match value {
        Value::Bool(true) => Some(Cow::Borrowed("b:1")),
        Value::Bool(false) => Some(Cow::Borrowed("b:0")),
        Value::Str(value) => Some(Cow::Owned(format!("s:{value}"))),
        _ => None,
    }
}

#[derive(Debug)]
struct BuildingPosting {
    posting: VectorFilterPosting,
    cardinality: usize,
}

impl BuildingPosting {
    fn new(ordinal: u32) -> Self {
        Self {
            posting: VectorFilterPosting::Sparse(vec![ordinal]),
            cardinality: 1,
        }
    }

    /// Insert an ordinal from the monotonically increasing compaction stream
    /// and retain the representation selected by the N/64 crossover.
    fn insert(&mut self, ordinal: u32) {
        let next_cardinality = self.cardinality.saturating_add(1);
        let words = (ordinal as usize + 1).div_ceil(64);
        match &mut self.posting {
            VectorFilterPosting::Sparse(ordinals) => {
                debug_assert!(ordinals.last().is_none_or(|last| *last < ordinal));
                ordinals.push(ordinal);
                if next_cardinality > words {
                    let sparse = std::mem::take(ordinals);
                    self.posting = VectorFilterPosting::from_sorted_ordinals(sparse);
                }
            }
            VectorFilterPosting::Dense(dense) if next_cardinality > words => {
                dense.resize(words, 0);
                let ordinal = ordinal as usize;
                dense[ordinal / 64] |= 1u64 << (ordinal % 64);
            }
            VectorFilterPosting::Dense(_) => {
                // A once-common value can become sparse when the corpus grows
                // and only a far-later ordinal carries it. Convert back instead
                // of extending a mostly-empty bitmap to that far ordinal.
                let old =
                    std::mem::replace(&mut self.posting, VectorFilterPosting::Sparse(Vec::new()));
                let mut sparse = old.into_sorted_ordinals();
                sparse.push(ordinal);
                self.posting = VectorFilterPosting::Sparse(sparse);
            }
        }
        self.cardinality = next_cardinality;
    }

    fn payload_bytes(&self) -> usize {
        self.posting.payload_bytes()
    }
}

#[derive(Debug)]
enum FilterPropertyState {
    Active {
        postings: BTreeMap<String, BuildingPosting>,
        estimated_bytes: usize,
    },
    Disabled,
}

/// Streaming, bounded collector used by authoritative node compaction.
///
/// It never retains NodeIds or duplicate property payloads: every posting is a
/// `u32` ordinal into the vector body's existing id table. A property is dropped
/// atomically (never truncated) when either cap is crossed, preserving the
/// invariant that presence in the body means complete coverage.
#[derive(Debug)]
pub(crate) struct VectorFilterPostingsBuilder {
    properties: BTreeMap<String, FilterPropertyState>,
    limits: VectorFilterLimits,
    estimated_bytes: usize,
}

impl VectorFilterPostingsBuilder {
    const ENTRY_OVERHEAD: usize = 64;

    pub(crate) fn new(
        properties: impl IntoIterator<Item = String>,
        limits: VectorFilterLimits,
    ) -> Self {
        let mut out = Self {
            properties: BTreeMap::new(),
            limits,
            estimated_bytes: 0,
        };
        if limits.max_bytes == 0 || limits.max_distinct_per_property == 0 {
            return out;
        }
        for property in properties {
            let base = property.len().saturating_add(Self::ENTRY_OVERHEAD);
            if out.estimated_bytes.saturating_add(base) > limits.max_bytes {
                out.properties
                    .insert(property, FilterPropertyState::Disabled);
                continue;
            }
            out.estimated_bytes = out.estimated_bytes.saturating_add(base);
            out.properties.insert(
                property,
                FilterPropertyState::Active {
                    postings: BTreeMap::new(),
                    estimated_bytes: base,
                },
            );
        }
        out
    }

    pub(crate) fn observe(&mut self, ordinal: u32, values: &BTreeMap<String, Value>) {
        let mut disable = Vec::new();
        for (property, state) in &mut self.properties {
            let FilterPropertyState::Active {
                postings,
                estimated_bytes,
            } = state
            else {
                continue;
            };
            let Some(value) = values.get(property) else {
                continue;
            };
            let Some(key) = encode_vector_filter_value(value) else {
                continue;
            };
            let is_new = !postings.contains_key(key.as_ref());
            if is_new && postings.len() >= self.limits.max_distinct_per_property {
                self.estimated_bytes = self.estimated_bytes.saturating_sub(*estimated_bytes);
                disable.push(property.clone());
                continue;
            }
            let key_len = key.len();
            let new_entry_overhead = if is_new {
                key_len.saturating_add(Self::ENTRY_OVERHEAD)
            } else {
                0
            };
            let (old_payload, new_payload) = if let Some(posting) = postings.get_mut(key.as_ref()) {
                let old_payload = posting.payload_bytes();
                posting.insert(ordinal);
                (old_payload, posting.payload_bytes())
            } else {
                let posting = BuildingPosting::new(ordinal);
                let new_payload = posting.payload_bytes();
                postings.insert(key.into_owned(), posting);
                (0, new_payload)
            };
            let growth = new_entry_overhead.saturating_add(new_payload);
            let shrink = old_payload;
            let projected = self
                .estimated_bytes
                .saturating_sub(shrink)
                .saturating_add(growth);
            if projected > self.limits.max_bytes {
                // Presence means complete coverage, so crossing the real
                // adaptive representation's cap drops the whole property.
                // Never retain a truncated posting.
                self.estimated_bytes = self.estimated_bytes.saturating_sub(*estimated_bytes);
                disable.push(property.clone());
                continue;
            }
            *estimated_bytes = estimated_bytes
                .saturating_sub(shrink)
                .saturating_add(growth);
            self.estimated_bytes = projected;
        }
        for property in disable {
            let Some(state) = self.properties.get_mut(&property) else {
                continue;
            };
            *state = FilterPropertyState::Disabled;
        }
    }

    pub(crate) fn finish(self) -> VectorFilterPostings {
        self.properties
            .into_iter()
            .filter_map(|(property, state)| match state {
                FilterPropertyState::Active { postings, .. } => Some((property, postings)),
                FilterPropertyState::Disabled => None,
            })
            .map(|(property, postings)| {
                (
                    property,
                    postings
                        .into_iter()
                        .map(|(key, posting)| (key, posting.posting))
                        .collect(),
                )
            })
            .collect()
    }

    #[cfg(test)]
    fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }
}

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
    /// Complete low-cardinality String/Bool metadata postings. An absent
    /// property means "not materialised" (never "no matches"), so readers must
    /// retain their residual fallback.
    pub(crate) filter_postings: VectorFilterPostings,
}

/// Exact v3 wire shape, retained solely for backward-compatible decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyVectorGraphBodyV3 {
    dim: u32,
    metric: String,
    ids: Vec<[u8; 16]>,
    storage: VectorStorage,
    graph: VamanaGraph,
}

/// Stats harvested at build time, mirrored into
/// [`crate::manifest::KindSpecificStats::VectorGraph`].
#[derive(Debug, Clone)]
pub struct VectorGraphBuildStats {
    pub dim: u32,
    pub metric: String,
    pub point_count: u64,
    /// Exact NodeId bounds of the vectors retained in this graph. Persisted in
    /// the generic SST key range so freshness checks can prove that a newer
    /// node SST for another label cannot contain a relabel/delete of a member.
    pub min_node_id: [u8; 16],
    pub max_node_id: [u8; 16],
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
    members: Vec<([u8; 16], Vec<f32>)>,
) -> Result<Option<(Bytes, VectorGraphBuildStats)>, Error> {
    build_body_with_filter_postings(desc, members, VectorFilterPostings::new())
}

/// Build a vector body plus complete, bounded metadata postings harvested by
/// the authoritative node merge.
pub(crate) fn build_body_with_filter_postings(
    desc: &VectorIndexDescriptor,
    mut members: Vec<([u8; 16], Vec<f32>)>,
    mut filter_postings: VectorFilterPostings,
) -> Result<Option<(Bytes, VectorGraphBuildStats)>, Error> {
    if members.len() < 2 {
        return Ok(None);
    }
    let dim = desc.dim as usize;
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(members.len());
    let mut ids: Vec<[u8; 16]> = Vec::with_capacity(members.len());
    let mut ordinal_remap: Vec<Option<u32>> = Vec::with_capacity(members.len());
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
            ordinal_remap.push(None);
            continue;
        }
        let ordinal = u32::try_from(ids.len()).map_err(|_| {
            Error::invariant(format!(
                "vector index `{}` exceeds the u32 ordinal space",
                desc.name
            ))
        })?;
        ordinal_remap.push(Some(ordinal));
        ids.push(id);
        vectors.push(v);
    }
    // Fewer than 2 indexable members after filtering → no graph (flat scan).
    if vectors.len() < 2 {
        return Ok(None);
    }

    // Collector ordinals refer to the pre-validation member stream. Remap them
    // after cosine zero-vector removal so every posting stays parallel to
    // `ids`/`storage`/`graph.adjacency`. A malformed out-of-range ordinal is
    // dropped defensively; compaction's collector cannot produce one.
    for values in filter_postings.values_mut() {
        for posting in values.values_mut() {
            let old = std::mem::replace(posting, VectorFilterPosting::Sparse(Vec::new()));
            let remapped = old
                .into_sorted_ordinals()
                .into_iter()
                .filter_map(|ordinal| ordinal_remap.get(ordinal as usize).copied().flatten())
                .collect();
            *posting = VectorFilterPosting::from_sorted_ordinals(remapped);
        }
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

    let (Some(&min_node_id), Some(&max_node_id)) = (ids.iter().min(), ids.iter().max()) else {
        return Ok(None);
    };
    let stats = VectorGraphBuildStats {
        dim: desc.dim,
        metric: metric_name(desc.metric).to_string(),
        point_count: ids.len() as u64,
        min_node_id,
        max_node_id,
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
        filter_postings,
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

/// Validate every ordinal-bearing and numeric part of a decoded body before it
/// reaches `namidb-ann`. The ANN spaces deliberately trust their constructor:
/// unequal vector dimensions can trip debug assertions, non-finite distances
/// violate the search heap's ordering contract, and an invalid graph ordinal
/// can index outside the stored corpus. A `.vg` is an optional accelerator, so
/// rejecting the whole body is safer than partially repairing it.
fn validate_decoded_body(body: &VectorGraphBody, metric: VectorMetric) -> Result<(), Error> {
    let n = body.storage.len();
    if n != body.ids.len() || n != body.graph.adjacency.len() {
        return Err(Error::invariant("vector graph body length mismatch"));
    }
    let n_u32 = u32::try_from(n)
        .map_err(|_| Error::invariant("vector graph exceeds the u32 ordinal space"))?;
    if n == 0 {
        if body.graph.entry != 0 {
            return Err(Error::invariant(
                "empty vector graph has a non-zero entry point",
            ));
        }
    } else if body.graph.entry as usize >= n {
        return Err(Error::invariant("vector graph entry out of range"));
    }

    let dim = body.dim as usize;
    match &body.storage {
        VectorStorage::F32(vectors) => {
            for (ordinal, vector) in vectors.iter().enumerate() {
                if vector.len() != dim {
                    return Err(Error::invariant(format!(
                        "vector graph f32 vector {ordinal} has dimension {} (expected {dim})",
                        vector.len()
                    )));
                }
                let mut norm_sq = 0.0f32;
                for component in vector {
                    if !component.is_finite() {
                        return Err(Error::invariant(format!(
                            "vector graph f32 vector {ordinal} has a non-finite component"
                        )));
                    }
                    norm_sq += component * component;
                    if !norm_sq.is_finite() {
                        return Err(Error::invariant(format!(
                            "vector graph f32 vector {ordinal} has a non-finite norm"
                        )));
                    }
                }
            }
        }
        VectorStorage::Int8 { codes, scales } => {
            if metric != VectorMetric::Cosine {
                return Err(Error::invariant(
                    "vector graph int8 storage requires the cosine metric",
                ));
            }
            if codes.len() != scales.len() {
                return Err(Error::invariant("vector graph int8 codes/scales mismatch"));
            }
            for (ordinal, (code, scale)) in codes.iter().zip(scales).enumerate() {
                if code.len() != dim {
                    return Err(Error::invariant(format!(
                        "vector graph int8 code {ordinal} has dimension {} (expected {dim})",
                        code.len()
                    )));
                }
                if !scale.is_finite() || *scale < 0.0 {
                    return Err(Error::invariant(format!(
                        "vector graph int8 code {ordinal} has an invalid scale"
                    )));
                }
                if *scale == 0.0 && code.iter().any(|component| *component != 0) {
                    return Err(Error::invariant(format!(
                        "vector graph int8 code {ordinal} has non-zero components at zero scale"
                    )));
                }
                let code_norm = code
                    .iter()
                    .map(|component| (*component as f32) * (*component as f32))
                    .sum::<f32>()
                    .sqrt();
                if !(code_norm * scale).is_finite() {
                    return Err(Error::invariant(format!(
                        "vector graph int8 code {ordinal} has a non-finite norm"
                    )));
                }
            }
        }
    }

    // Reuse one u32-per-member scratch for both duplicate NodeId detection and
    // duplicate-neighbour detection. This keeps validation memory linear and
    // avoids a HashSet allocation per adjacency list.
    let mut ordinal_scratch: Vec<u32> = (0..n_u32).collect();
    ordinal_scratch
        .sort_unstable_by(|left, right| body.ids[*left as usize].cmp(&body.ids[*right as usize]));
    if ordinal_scratch
        .windows(2)
        .any(|pair| body.ids[pair[0] as usize] == body.ids[pair[1] as usize])
    {
        return Err(Error::invariant("vector graph contains duplicate node ids"));
    }
    ordinal_scratch.fill(n_u32);
    for (source, neighbors) in body.graph.adjacency.iter().enumerate() {
        if neighbors.len() > n.saturating_sub(1) {
            return Err(Error::invariant(format!(
                "vector graph adjacency {source} exceeds the corpus"
            )));
        }
        let source_u32 = source as u32;
        for &neighbor in neighbors {
            let neighbor = neighbor as usize;
            if neighbor >= n {
                return Err(Error::invariant(format!(
                    "vector graph adjacency {source} has an out-of-range neighbor"
                )));
            }
            if neighbor == source {
                return Err(Error::invariant(format!(
                    "vector graph adjacency {source} contains a self-loop"
                )));
            }
            if ordinal_scratch[neighbor] == source_u32 {
                return Err(Error::invariant(format!(
                    "vector graph adjacency {source} contains a duplicate neighbor"
                )));
            }
            ordinal_scratch[neighbor] = source_u32;
        }
    }

    let posting_words = n.div_ceil(64);
    for values in body.filter_postings.values() {
        for posting in values.values() {
            match posting {
                VectorFilterPosting::Sparse(ordinals) => {
                    if ordinals.windows(2).any(|pair| pair[0] >= pair[1])
                        || ordinals
                            .last()
                            .is_some_and(|ordinal| *ordinal as usize >= n)
                    {
                        return Err(Error::invariant(
                            "vector graph sparse filter posting is invalid",
                        ));
                    }
                }
                VectorFilterPosting::Dense(words) => {
                    if words.len() > posting_words {
                        return Err(Error::invariant(
                            "vector graph dense filter posting exceeds corpus",
                        ));
                    }
                    if words.len() == posting_words && n % 64 != 0 {
                        let valid_mask = (1u64 << (n % 64)) - 1;
                        if words.last().is_some_and(|word| word & !valid_mask != 0) {
                            return Err(Error::invariant(
                                "vector graph dense filter posting has out-of-range bits",
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
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
        let body: VectorGraphBody = if magic == MAGIC {
            bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .reject_trailing_bytes()
                .deserialize(rest)
                .map_err(|e| Error::invariant(format!("vector graph decode failed: {e}")))?
        } else if magic == LEGACY_MAGIC_V3 {
            let old: LegacyVectorGraphBodyV3 = bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .reject_trailing_bytes()
                .deserialize(rest)
                .map_err(|e| Error::invariant(format!("legacy vector graph decode failed: {e}")))?;
            VectorGraphBody {
                dim: old.dim,
                metric: old.metric,
                ids: old.ids,
                storage: old.storage,
                graph: old.graph,
                filter_postings: VectorFilterPostings::new(),
            }
        } else {
            return Err(Error::invariant(format!(
                "vector graph magic mismatch: {:?}",
                magic
            )));
        };
        let (metric, mips) = metric_from_name(&body.metric).ok_or_else(|| {
            Error::invariant(format!("vector graph unknown metric: {}", body.metric))
        })?;
        validate_decoded_body(&body, metric)?;
        let nav = match &body.storage {
            VectorStorage::Int8 { codes, scales } => {
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
        self.search_filtered(query, k, ef, |_| true)
    }

    /// Approximate top-`k` restricted to members accepted by `is_allowed`.
    ///
    /// The predicate is evaluated against the index's NodeId table while the
    /// ANN candidate pool is still inside the decoded `.vg`, before metric
    /// reranking and, critically, before the `k` truncation. Rejected members
    /// remain usable as graph-navigation waypoints (filtering them out of the
    /// Vamana walk can disconnect a selective slice), but they never consume a
    /// result slot and their full node/property payload is not hydrated.
    ///
    /// `ef` therefore controls the unfiltered navigation pool while `k` is the
    /// number of *eligible* neighbours requested. A selective filter can return
    /// fewer than `k`; the query layer widens `ef` geometrically and retains its
    /// exact flat fallback when the ANN pool cannot fill the requested slice.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        mut is_allowed: impl FnMut(&[u8; 16]) -> bool,
    ) -> Vec<([u8; 16], f32)> {
        self.search_filtered_ordinals(query, k, ef, |_, id| is_allowed(id))
    }

    /// Native metadata-filtered search. Each group is `(property, OR-values)`;
    /// groups AND-combine. Unsupported properties are left to the executor's
    /// residual predicate, while every materialised group narrows an ordinal
    /// bitmap. `None` means this body could apply none of the groups (v3 body,
    /// high-cardinality property dropped by the cap, or unindexed metadata).
    pub(crate) fn search_filter_groups(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        groups: &[(String, Vec<Value>)],
    ) -> Option<(Vec<([u8; 16], f32)>, usize)> {
        let word_count = self.body.ids.len().div_ceil(64);
        let mut eligible: Option<Vec<u64>> = None;
        for (property, alternatives) in groups {
            let Some(postings) = self.body.filter_postings.get(property) else {
                continue;
            };
            let mut group = vec![0u64; word_count];
            let mut group_supported = true;
            for value in alternatives {
                let Some(key) = encode_vector_filter_value(value) else {
                    group_supported = false;
                    break;
                };
                if let Some(posting) = postings.get(key.as_ref()) {
                    match posting {
                        VectorFilterPosting::Sparse(ordinals) => {
                            for &ordinal in ordinals {
                                let ordinal = ordinal as usize;
                                if ordinal < self.body.ids.len() {
                                    group[ordinal / 64] |= 1u64 << (ordinal % 64);
                                }
                            }
                        }
                        VectorFilterPosting::Dense(words) => {
                            for (dst, src) in group.iter_mut().zip(words) {
                                *dst |= src;
                            }
                        }
                    }
                }
            }
            if !group_supported {
                continue;
            }
            eligible = Some(match eligible {
                None => group,
                Some(mut current) => {
                    for (word, rhs) in current.iter_mut().zip(group) {
                        *word &= rhs;
                    }
                    current
                }
            });
            if eligible
                .as_ref()
                .is_some_and(|bitmap| bitmap.iter().all(|word| *word == 0))
            {
                break;
            }
        }
        let eligible = eligible?;
        VECTOR_FILTER_BITMAP_SEARCHES.fetch_add(1, Ordering::Relaxed);
        let eligible_count = eligible.iter().map(|word| word.count_ones() as usize).sum();
        if eligible_count == 0 {
            return Some((Vec::new(), 0));
        }
        let hits = self.search_filtered_ordinals(query, k, ef, |ordinal, _| {
            let ordinal = ordinal as usize;
            eligible
                .get(ordinal / 64)
                .is_some_and(|word| word & (1u64 << (ordinal % 64)) != 0)
        });
        Some((hits, eligible_count))
    }

    fn search_filtered_ordinals(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        mut is_allowed: impl FnMut(u32, &[u8; 16]) -> bool,
    ) -> Vec<([u8; 16], f32)> {
        let point_count = self.body.ids.len();
        let k = k.min(point_count);
        if k == 0 || query.len() != self.body.dim as usize {
            return Vec::new();
        }
        // Both knobs are query-controlled. The ANN crate also clamps its
        // scratch buffers internally, but enforce the corpus bound at this API
        // boundary so future implementations cannot reserve `usize::MAX`
        // before observing that this body contains only `point_count` rows.
        let ef = ef.max(k).min(point_count);
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
            .filter_map(|nb| {
                let id = &self.body.ids[nb.id as usize];
                if !is_allowed(nb.id, id) {
                    return None;
                }
                let score = if is_int8 {
                    // int8 cosine similarity (the stored, quantized score).
                    1.0 - nb.dist
                } else {
                    let v = self.body.storage.f32_at(nb.id as usize);
                    metric_score(self.metric, &v, query).0 as f32
                };
                Some((*id, score))
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

    fn wire_body(bytes: &[u8]) -> VectorGraphBody {
        bincode::deserialize(&bytes[MAGIC.len()..]).unwrap()
    }

    fn encode_wire_body(body: &VectorGraphBody) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&bincode::serialize(body).unwrap());
        bytes
    }

    fn assert_body_rejected(body: &VectorGraphBody, expected: &str) {
        let err = VectorGraphIndex::decode(&encode_wire_body(body))
            .expect_err("corrupt vector body must be rejected");
        assert!(
            err.to_string().contains(expected),
            "expected `{expected}` in decode error, got `{err}`"
        );
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
    fn filtered_search_applies_k_after_eligibility() {
        // The first eight members are closest to +x but belong to the excluded
        // slice. Only the final four ids are eligible and deliberately rank
        // farther away. Filtering an already-truncated k=3 list would return
        // zero; filtering inside the index candidate pool must return three.
        let mut members = Vec::new();
        for i in 0..8u64 {
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&i.to_be_bytes());
            members.push((id, vec![1.0, i as f32 * 0.001, 0.0, 0.0]));
        }
        for i in 8..12u64 {
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&i.to_be_bytes());
            members.push((id, vec![0.2, 1.0 + i as f32 * 0.001, 0.0, 0.0]));
        }
        let d = desc("filtered-k", VectorMetric::Cosine, 4);
        let (body, _) = build_body(&d, members).unwrap().unwrap();
        let idx = VectorGraphIndex::decode(&body).unwrap();
        let hits = idx.search_filtered(&[1.0, 0.0, 0.0, 0.0], 3, 12, |id| {
            u64::from_be_bytes(id[0..8].try_into().unwrap()) >= 8
        });
        assert_eq!(hits.len(), 3, "k is counted among eligible members");
        assert!(
            hits.iter()
                .all(|(id, _)| u64::from_be_bytes(id[0..8].try_into().unwrap()) >= 8),
            "excluded near neighbours must not consume result slots: {hits:?}"
        );
    }

    #[test]
    fn search_clamps_extreme_k_and_ef_to_the_corpus() {
        let members = clustered_members(12, 4, 27);
        let query = members[0].1.clone();
        let d = desc("bounded-query-knobs", VectorMetric::Cosine, 4);
        let (body, _) = build_body(&d, members).unwrap().unwrap();
        let idx = VectorGraphIndex::decode(&body).unwrap();

        let hits = idx.search(&query, usize::MAX, usize::MAX);
        assert_eq!(hits.len(), 12);
    }

    #[test]
    fn native_posting_filter_applies_k_and_distinguishes_absence() {
        let mut members = Vec::new();
        let mut postings = VectorFilterPostingsBuilder::new(
            ["vigente".to_string(), "ambito".to_string()],
            VectorFilterLimits::default(),
        );
        for i in 0..8u64 {
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&i.to_be_bytes());
            members.push((id, vec![1.0, i as f32 * 0.001, 0.0, 0.0]));
            postings.observe(
                i as u32,
                &BTreeMap::from([
                    ("vigente".into(), Value::Bool(false)),
                    ("ambito".into(), Value::Str("civil".into())),
                ]),
            );
        }
        for i in 8..12u64 {
            let mut id = [0u8; 16];
            id[0..8].copy_from_slice(&i.to_be_bytes());
            members.push((id, vec![0.2, 1.0 + i as f32 * 0.001, 0.0, 0.0]));
            postings.observe(
                i as u32,
                &BTreeMap::from([
                    ("vigente".into(), Value::Bool(true)),
                    ("ambito".into(), Value::Str("laboral".into())),
                ]),
            );
        }
        let d = desc("posting-filtered-k", VectorMetric::Cosine, 4);
        let (body, _) = build_body_with_filter_postings(&d, members, postings.finish())
            .unwrap()
            .unwrap();
        let idx = VectorGraphIndex::decode(&body).unwrap();
        let before = vector_filter_bitmap_searches();
        let (hits, eligible) = idx
            .search_filter_groups(
                &[1.0, 0.0, 0.0, 0.0],
                3,
                12,
                &[("vigente".into(), vec![Value::Bool(true)])],
            )
            .expect("materialised property applies natively");
        assert_eq!(eligible, 4);
        assert_eq!(hits.len(), 3, "k is counted after the native filter");
        assert!(hits
            .iter()
            .all(|(id, _)| { u64::from_be_bytes(id[0..8].try_into().unwrap()) >= 8 }));
        assert_eq!(vector_filter_bitmap_searches(), before + 1);

        // An absent property means the body cannot narrow it: residual path.
        assert!(idx
            .search_filter_groups(
                &[1.0, 0.0, 0.0, 0.0],
                3,
                12,
                &[("jurisdiccion".into(), vec![Value::Str("laboral".into())],)],
            )
            .is_none());
        // An absent value under a complete, present property means exact empty.
        let (hits, eligible) = idx
            .search_filter_groups(
                &[1.0, 0.0, 0.0, 0.0],
                3,
                12,
                &[("ambito".into(), vec![Value::Str("penal".into())])],
            )
            .expect("present property with absent value is supported");
        assert!(hits.is_empty());
        assert_eq!(eligible, 0);
        assert!(
            idx.search_filter_groups(
                &[1.0, 0.0, 0.0, 0.0],
                3,
                12,
                &[("vigente".into(), vec![Value::I64(1)])],
            )
            .is_none(),
            "an unencodable group must remain residual, not become exact empty"
        );

        let (_hits, eligible) = idx
            .search_filter_groups(
                &[1.0, 0.0, 0.0, 0.0],
                3,
                12,
                &[
                    ("vigente".into(), vec![Value::Bool(true)]),
                    ("ambito".into(), vec![Value::Str("laboral".into())]),
                ],
            )
            .expect("String and Bool groups AND without NodeId hydration");
        assert_eq!(eligible, 4);

        let (_hits, eligible) = idx
            .search_filter_groups(
                &[1.0, 0.0, 0.0, 0.0],
                3,
                12,
                &[(
                    "vigente".into(),
                    vec![Value::Bool(true), Value::Bool(false)],
                )],
            )
            .expect("alternatives OR");
        assert_eq!(eligible, 12);
        let (hits, eligible) = idx
            .search_filter_groups(
                &[1.0, 0.0, 0.0, 0.0],
                3,
                12,
                &[
                    ("vigente".into(), vec![Value::Bool(true)]),
                    ("vigente".into(), vec![Value::Bool(false)]),
                ],
            )
            .expect("groups AND");
        assert!(hits.is_empty());
        assert_eq!(eligible, 0);
    }

    #[test]
    fn boolean_postings_stay_bounded_at_seven_hundred_fifty_thousand_vectors() {
        const N: u32 = 750_000;
        const CAP: usize = 1024 * 1024;
        let mut builder = VectorFilterPostingsBuilder::new(
            ["vigente".to_string()],
            VectorFilterLimits {
                max_distinct_per_property: 8,
                max_bytes: CAP,
            },
        );
        let mut values = BTreeMap::from([("vigente".into(), Value::Bool(false))]);
        for ordinal in 0..N {
            *values.get_mut("vigente").unwrap() = Value::Bool(ordinal % 2 == 0);
            builder.observe(ordinal, &values);
        }
        assert!(
            builder.estimated_bytes() <= CAP,
            "collector must enforce the configured real adaptive-byte cap"
        );
        let postings = builder.finish();
        let values = postings
            .get("vigente")
            .expect("low-cardinality Boolean property must be retained");
        assert_eq!(values.len(), 2);
        let payload_bytes: usize = values
            .values()
            .map(VectorFilterPosting::payload_bytes)
            .sum();
        assert!(
            payload_bytes < 200_000,
            "two 750k-bit postings should be ~184 KiB, got {payload_bytes}"
        );
        assert!(
            values
                .values()
                .all(|posting| matches!(posting, VectorFilterPosting::Dense(_))),
            "50% selective Boolean values should serialize as dense bitmaps"
        );
        let cardinality: usize = values
            .values()
            .cloned()
            .map(VectorFilterPosting::into_sorted_ordinals)
            .map(|ordinals| ordinals.len())
            .sum();
        assert_eq!(cardinality, N as usize);
    }

    #[test]
    fn posting_caps_drop_properties_atomically_never_truncate() {
        let mut builder = VectorFilterPostingsBuilder::new(
            ["kind".to_string()],
            VectorFilterLimits {
                max_distinct_per_property: 2,
                max_bytes: 1024 * 1024,
            },
        );
        for (ordinal, value) in ["a", "b", "c"].into_iter().enumerate() {
            builder.observe(
                ordinal as u32,
                &BTreeMap::from([("kind".into(), Value::Str(value.into()))]),
            );
        }
        assert!(
            !builder.finish().contains_key("kind"),
            "crossing distinct cap drops the complete property, not one value"
        );

        let mut builder = VectorFilterPostingsBuilder::new(
            ["vigente".to_string()],
            VectorFilterLimits {
                max_distinct_per_property: 2,
                max_bytes: 80,
            },
        );
        builder.observe(0, &BTreeMap::from([("vigente".into(), Value::Bool(true))]));
        assert!(
            !builder.finish().contains_key("vigente"),
            "crossing byte cap drops the complete property"
        );
    }

    #[test]
    fn vector_v4_keys_separate_string_from_boolean_tags() {
        assert_eq!(
            encode_vector_filter_value(&Value::Bool(true)).as_deref(),
            Some("b:1")
        );
        assert_eq!(
            encode_vector_filter_value(&Value::Str("b:1".into())).as_deref(),
            Some("s:b:1")
        );
        let mut builder =
            VectorFilterPostingsBuilder::new(["mixed".to_string()], VectorFilterLimits::default());
        builder.observe(0, &BTreeMap::from([("mixed".into(), Value::Bool(true))]));
        builder.observe(
            1,
            &BTreeMap::from([("mixed".into(), Value::Str("b:1".into()))]),
        );
        let postings = builder.finish();
        let values = &postings["mixed"];
        assert!(values.contains_key("b:1"));
        assert!(values.contains_key("s:b:1"));
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn real_v3_wire_body_decodes_and_leaves_filter_residual() {
        let d = desc("legacy-v3", VectorMetric::Cosine, 8);
        let members = clustered_members(40, 8, 29);
        let query = members[0].1.clone();
        let (v4, _) = build_body(&d, members).unwrap().unwrap();
        let body: VectorGraphBody = bincode::deserialize(&v4[MAGIC.len()..]).unwrap();
        let legacy = LegacyVectorGraphBodyV3 {
            dim: body.dim,
            metric: body.metric,
            ids: body.ids,
            storage: body.storage,
            graph: body.graph,
        };
        let mut bytes = LEGACY_MAGIC_V3.to_vec();
        bytes.extend_from_slice(&bincode::serialize(&legacy).unwrap());

        let idx = VectorGraphIndex::decode(&bytes).expect("real NAMIVG03 body remains readable");
        assert!(!idx.search(&query, 3, 16).is_empty());
        assert!(
            idx.search_filter_groups(
                &query,
                3,
                16,
                &[("vigente".into(), vec![Value::Bool(true)])],
            )
            .is_none(),
            "v3 has no postings and must preserve residual fallback"
        );
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
        let (bytes, _) = build_body(&d, clustered_members(40, 8, 3))
            .unwrap()
            .unwrap();
        // Rewrite the body's metric tag to the legacy name.
        let mut body: VectorGraphBody = bincode::deserialize(&bytes[MAGIC.len()..]).unwrap();
        body.metric = "dot".to_string();
        let mut legacy = MAGIC.to_vec();
        legacy.extend_from_slice(&bincode::serialize(&body).unwrap());
        let idx = VectorGraphIndex::decode(&legacy).unwrap();
        assert_eq!(idx.metric(), "dot");
        assert!(!idx
            .search(&clustered_members(1, 8, 3)[0].1, 3, 16)
            .is_empty());
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
        let mut decoded = wire_body(&body);
        decoded.graph.entry = 9999;
        assert_body_rejected(&decoded, "entry out of range");
    }

    #[test]
    fn decode_rejects_truncation_and_trailing_tamper() {
        let d = desc("wire-integrity", VectorMetric::Cosine, 4);
        let (body, _) = build_body(&d, clustered_members(10, 4, 3))
            .unwrap()
            .unwrap();

        let mut truncated = body.to_vec();
        truncated.pop();
        let err = VectorGraphIndex::decode(&truncated).unwrap_err();
        assert!(err.to_string().contains("decode failed"), "{err}");

        let mut trailing = body.to_vec();
        trailing.push(0xA5);
        let err = VectorGraphIndex::decode(&trailing).unwrap_err();
        assert!(
            err.to_string().contains("Slice had bytes remaining"),
            "{err}"
        );
    }

    #[test]
    fn decode_rejects_wrong_f32_dimension_and_non_finite_values() {
        let d = desc("bad-f32", VectorMetric::Cosine, 4);
        let (bytes, _) = build_body(&d, clustered_members(10, 4, 3))
            .unwrap()
            .unwrap();
        let original = wire_body(&bytes);

        let mut body = original.clone();
        let VectorStorage::F32(vectors) = &mut body.storage else {
            panic!("fixture is f32");
        };
        vectors[0].pop();
        assert_body_rejected(&body, "has dimension 3 (expected 4)");

        let mut body = original.clone();
        let VectorStorage::F32(vectors) = &mut body.storage else {
            panic!("fixture is f32");
        };
        vectors[0][0] = f32::NAN;
        assert_body_rejected(&body, "non-finite component");

        let mut body = original;
        let VectorStorage::F32(vectors) = &mut body.storage else {
            panic!("fixture is f32");
        };
        vectors[0][0] = f32::MAX;
        assert_body_rejected(&body, "non-finite norm");
    }

    #[test]
    fn decode_rejects_wrong_int8_dimension_and_non_finite_scale_or_norm() {
        let d = desc_q(
            "bad-int8",
            VectorMetric::Cosine,
            4,
            VectorQuantization::Int8,
        );
        let (bytes, _) = build_body(&d, clustered_members(10, 4, 3))
            .unwrap()
            .unwrap();
        let original = wire_body(&bytes);

        let mut body = original.clone();
        let VectorStorage::Int8 { codes, .. } = &mut body.storage else {
            panic!("fixture is int8");
        };
        codes[0].pop();
        assert_body_rejected(&body, "has dimension 3 (expected 4)");

        let mut body = original.clone();
        let VectorStorage::Int8 { scales, .. } = &mut body.storage else {
            panic!("fixture is int8");
        };
        scales[0] = f32::NAN;
        assert_body_rejected(&body, "invalid scale");

        let mut body = original;
        let VectorStorage::Int8 { codes, scales } = &mut body.storage else {
            panic!("fixture is int8");
        };
        codes[0].fill(i8::MAX);
        scales[0] = f32::MAX;
        assert_body_rejected(&body, "non-finite norm");
    }

    #[test]
    fn decode_rejects_invalid_ids_and_adjacency() {
        let d = desc("bad-ordinals", VectorMetric::Cosine, 4);
        let (bytes, _) = build_body(&d, clustered_members(10, 4, 3))
            .unwrap()
            .unwrap();
        let original = wire_body(&bytes);

        let mut body = original.clone();
        body.ids[1] = body.ids[0];
        assert_body_rejected(&body, "duplicate node ids");

        let mut body = original.clone();
        body.graph.adjacency[0] = vec![body.ids.len() as u32];
        assert_body_rejected(&body, "out-of-range neighbor");

        let mut body = original.clone();
        body.graph.adjacency[0] = vec![0];
        assert_body_rejected(&body, "self-loop");

        let mut body = original;
        body.graph.adjacency[0] = vec![1, 1];
        assert_body_rejected(&body, "duplicate neighbor");
    }

    #[test]
    fn decode_rejects_out_of_range_filter_postings() {
        let d = desc("bad-filter", VectorMetric::Cosine, 4);
        let (body, _) = build_body(&d, clustered_members(10, 4, 3))
            .unwrap()
            .unwrap();
        let mut decoded = wire_body(&body);
        decoded.filter_postings.insert(
            "vigente".into(),
            BTreeMap::from([(
                "b:1".into(),
                VectorFilterPosting::Sparse(vec![decoded.ids.len() as u32]),
            )]),
        );
        assert_body_rejected(&decoded, "sparse filter posting is invalid");

        decoded.filter_postings.insert(
            "vigente".into(),
            BTreeMap::from([("b:1".into(), VectorFilterPosting::Dense(vec![u64::MAX]))]),
        );
        assert_body_rejected(&decoded, "dense filter posting has out-of-range bits");
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
