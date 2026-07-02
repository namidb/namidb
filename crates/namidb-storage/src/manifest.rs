//! Versioned manifest with compare-and-swap commit protocol.
//!
//! See [`docs/rfc/001-storage-engine.md`](../../../docs/rfc/001-storage-engine.md)
//! §"Manifest protocol" for the design.
//!
//! ## Invariants enforced here
//!
//! 1. **Write-once manifest versions.** `manifest/v<N>.json` is created with
//! `PutMode::Create` (HTTP `If-None-Match: *`). Two writers that pick the
//! same `N` cannot both succeed.
//! 2. **Create-only versioned pointer (RFC-029).** The current pointer is the
//! highest `N` in the Create-only family `manifest/pointer/p<N>.json`; each
//! file is written once with `PutMode::Create`. Two writers that pick the
//! same `N` cannot both succeed, so the pointer CAS uses the *same*
//! conditional primitive as the body — `If-None-Match: *` — and never the
//! spottily-supported `If-Match` overwrite. Losing the create yields
//! [`Error::ManifestCommitCas`] (or [`Error::Fenced`] under a higher epoch).
//! 3. **Monotonic version + epoch.** Commit refuses to write a manifest with
//! `version <= current.version`. Epoch may only increase.
//!
//! Recovery: any version `v` we created locally that did not become
//! `current` is garbage below the pointer; at `pointer + 1` it is a stalled
//! commit that [`ManifestStore::claim_writer`] repairs (publish or delete —
//! see `repair_stalled_commit`). A future janitor will delete orphan
//! manifests below the pointer.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use namidb_core::{LabelDictionary, Schema};

use crate::error::{Error, Result};
use crate::fence::{Epoch, WriterFence};
use crate::paths::NamespacePaths;
use crate::sst::bloom::BloomDescriptor;
use crate::sst::stats::{DegreeHistogram, HllSketchBytes, PropertyColumnStats, StatScalar};
use crate::wal::WalSegment;

/// Top-level versioned manifest. Self-contained snapshot of every artefact
/// that belongs to the namespace at this version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Monotonically increasing per namespace.
    pub version: u64,
    /// Writer epoch — bumped whenever a new writer claims the namespace.
    pub epoch: Epoch,
    /// UUID of the writer process that produced this manifest. Audit only.
    pub writer_id: Uuid,
    /// UTC creation time.
    pub created_at: DateTime<Utc>,
    /// Schema snapshot.
    pub schema: Schema,
    /// All SSTs visible to readers of this version.
    #[serde(default)]
    pub ssts: Vec<SstDescriptor>,
    /// WAL segments that are still required for recovery (i.e. not fully
    /// flushed into SSTs yet).
    #[serde(default)]
    pub wal_segments: Vec<WalSegmentDescriptor>,
    /// Namespace-wide dictionary mapping every label name to a stable,
    /// compact [`LabelId`](namidb_core::LabelId). A multi-label node carries
    /// its labels on-row as packed `LabelId`s; this is the source of truth
    /// that resolves them back to names. Append-only, cloned forward on every
    /// commit. Empty for older manifests (`serde(default)` round-trips them
    /// unchanged).
    #[serde(default)]
    pub label_dict: LabelDictionary,
    /// Registered DiskANN/Vamana vector indexes (RFC-030, `vector-index`
    /// feature). `CREATE VECTOR INDEX` appends here; the compaction build hook
    /// and the query optimizer both discover indexes by matching
    /// `(label, property)`. `serde(default)` keeps pre-feature manifests
    /// loading unchanged.
    #[serde(default)]
    pub vector_indexes: Vec<VectorIndexDescriptor>,

    /// Registered full-text (BM25) indexes (`text-index` feature). The
    /// compaction build hook materializes a `TextIndex` SST per descriptor and
    /// `CALL search.bm25` discovers one by matching `(label, properties)`.
    /// `serde(default)` keeps pre-feature manifests loading unchanged.
    #[serde(default)]
    pub text_indexes: Vec<TextIndexDescriptor>,
}

impl Manifest {
    /// Empty manifest at version 0 with the supplied epoch.
    pub fn empty(epoch: Epoch, writer_id: Uuid) -> Self {
        Self {
            version: 0,
            epoch,
            writer_id,
            created_at: Utc::now(),
            schema: Schema::empty(),
            ssts: Vec::new(),
            wal_segments: Vec::new(),
            label_dict: LabelDictionary::new(),
            vector_indexes: Vec::new(),
            text_indexes: Vec::new(),
        }
    }

    /// Returns a copy of `self` with `version` incremented, `created_at`
    /// refreshed, and `writer_id` set to the caller-supplied id. Convenience
    /// helper for higher layers that mutate manifests.
    pub fn next_version(&self, writer_id: Uuid) -> Self {
        Self {
            version: self.version + 1,
            epoch: self.epoch,
            writer_id,
            created_at: Utc::now(),
            schema: self.schema.clone(),
            ssts: self.ssts.clone(),
            wal_segments: self.wal_segments.clone(),
            label_dict: self.label_dict.clone(),
            vector_indexes: self.vector_indexes.clone(),
            text_indexes: self.text_indexes.clone(),
        }
    }
}

/// What sits in `manifest/current.json`. Tiny on purpose: every read path
/// fetches it before doing anything else, so it must be cheap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestPointer {
    pub version: u64,
    pub epoch: Epoch,
    pub manifest_path: String,
}

/// What kind of artefact an SST contains.
///
/// RFC-002 §4.1: `Edges` was split into `EdgesFwd` and `EdgesInv` (forward
/// and inverse partner CSRs). `VectorGraph` is a DiskANN/Vamana ANN search
/// graph body (see `namidb-ann` + the `vector-index` Cargo feature).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SstKind {
    /// Property-column SST for a node label (Parquet).
    Nodes,
    /// CSR adjacency SST for an edge type, sorted by `src_id`.
    EdgesFwd,
    /// CSR adjacency SST for an edge type, sorted by `dst_id` (inverse partner).
    EdgesInv,
    /// DiskANN/Vamana ANN search-graph body for one vector index. Self-contained
    /// (vectors + graph serialized inline), built during compaction. Has no
    /// meaningful lexicographic key range, so its descriptors carry full-range
    /// `min_key/max_key` and are looked up by `(kind, scope=index_name)`.
    VectorGraph,
    /// Full-text (BM25) inverted-index body for one text index (`text-index`
    /// feature). Self-contained (postings + corpus stats serialized inline),
    /// built during compaction. Like `VectorGraph` it has no lexicographic key
    /// range and is looked up by `(kind, scope=index_name)`.
    TextIndex,
}

impl SstKind {
    /// Path tag used in the SST filename (RFC-002 §1).
    pub fn path_tag(self) -> &'static str {
        match self {
            SstKind::Nodes => "nodes",
            SstKind::EdgesFwd => "edges-fwd",
            SstKind::EdgesInv => "edges-inv",
            SstKind::VectorGraph => "vector-graph",
            SstKind::TextIndex => "text-index",
        }
    }
}

/// Level in the LSM tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SstLevel(pub u32);

impl SstLevel {
    pub const L0: SstLevel = SstLevel(0);
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Per-kind statistics carried alongside every `SstDescriptor`. The variant
/// must match `SstDescriptor::kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum KindSpecificStats {
    Nodes {
        tombstone_count: u64,
    },
    /// `degree_histogram` is boxed to keep the enum compact — Edges'
    /// histogram (~272 B) is much larger than the Nodes variant.
    Edges {
        /// Distinct keys (src for `EdgesFwd`, dst for `EdgesInv`).
        key_count: u64,
        tombstone_count: u64,
        degree_histogram: Box<DegreeHistogram>,
    },
    /// Stats for a `VectorGraph` (DiskANN/Vamana) SST. Records the build
    /// parameters and the graph shape so the read path and observability can
    /// reason about it without decoding the body.
    VectorGraph {
        /// Embedding dimensionality.
        dim: u32,
        /// Distance metric the index was built for (`"cosine"`/`"dot"`/`"euclidean"`).
        metric: String,
        /// Number of vectors (graph nodes) indexed in this SST.
        point_count: u64,
        /// Vamana max out-degree (`R`).
        r: usize,
        /// Vamana build beam (`L_build`).
        l_build: usize,
        /// Vamana α diversification.
        alpha: f32,
        /// Entry-point medoid id.
        entry_medoid: u32,
    },
    /// Stats for a `TextIndex` (BM25 inverted-index) SST. Records the corpus
    /// shape so observability can reason about it without decoding the body.
    TextIndex {
        /// Number of documents indexed in this SST.
        doc_count: u64,
        /// Distinct terms in the inverted index.
        term_count: u64,
        /// Sum of all document lengths in tokens (→ average document length).
        total_len: u64,
    },
}

/// Compact description of a single SST file (RFC-002 §4.1).
///
/// Statistics that are small and useful for query gating live here so the
/// read path can prune candidate SSTs with **zero extra GETs** beyond the
/// manifest fetch itself. The bloom filter is the one exception: it lives
/// in a side-car file (see [`BloomDescriptor`]) because its size scales
/// with `key_count` and would blow the manifest budget if inlined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SstDescriptor {
    // ── identity ──
    pub id: Uuid,
    pub kind: SstKind,
    /// Label or edge-type name this SST belongs to.
    pub scope: String,
    pub level: SstLevel,
    /// Object-store path relative to the namespace prefix.
    pub path: String,

    // ── physical ──
    pub size_bytes: u64,
    /// Node rows or edge rows (rows in the body).
    pub row_count: u64,
    pub created_at: DateTime<Utc>,

    // ── key range (raw 16-byte bounds; JSON-encoded as base64) ──
    #[serde(with = "serde_key16")]
    pub min_key: [u8; 16],
    #[serde(with = "serde_key16")]
    pub max_key: [u8; 16],
    pub min_lsn: u64,
    pub max_lsn: u64,
    pub schema_version_min: u64,
    pub schema_version_max: u64,

    // ── stats embedded ──
    #[serde(default)]
    pub property_stats: Vec<PropertyColumnStats>,
    pub kind_specific: KindSpecificStats,

    // ── bloom side-car pointer (None when omitted per RFC-002 §4.2) ──
    #[serde(default)]
    pub bloom: Option<BloomDescriptor>,

    // ── unique-property side-car pointers (RFC-pending) ──
    //
    // For every `PropertyDef::unique == true` in the SST's label schema
    // at flush time, the writer emits a sidecar mapping
    // `value_string → NodeId`. The reader's `lookup_node_by_property`
    // loads these on demand instead of full-scanning the label.
    //
    // Empty for edge SSTs and for older manifests that pre-date the
    // sidecar emission (`serde(default)` covers backward compatibility).
    #[serde(default)]
    pub unique_property_indices: Vec<UniquePropertyIndexDescriptor>,
    // Secondary equality-index sidecars for `indexed` (non-unique)
    // properties. Same idea as `unique_property_indices`, but each value
    // maps to MANY node ids (a posting list), so the reader unions the
    // posting lists across SSTs and confirms each candidate. Empty for edge
    // SSTs and older manifests (`serde(default)`).
    #[serde(default)]
    pub equality_property_indices: Vec<EqualityIndexDescriptor>,
    // Label-index side-car pointer (multi-label nodes). Once node SSTs stop
    // being partitioned by label, `scan_label(L)` can no longer just read the
    // SSTs whose scope is `L`; instead each node SST ships one sidecar mapping
    // `LabelId → posting list of NodeIds` so the reader can union across SSTs
    // and confirm by id. `None` for edge SSTs and for legacy single-label node
    // SSTs (whose `scope` still names their one label); `serde(default)` keeps
    // older manifests loading unchanged.
    #[serde(default)]
    pub label_index: Option<LabelIndexDescriptor>,

    // Per-(label, property) statistics for id-primary node SSTs (RFC 025).
    // Because one node SST spans many labels and every property rides in a
    // single `__overflow_json` column, the per-column Parquet footer stats that
    // `property_stats` used to carry no longer exist. These are computed at
    // flush/compaction by grouping the SST's rows by their label set, and the
    // cost model folds them into `(label, property)` PropStats. Empty for edge
    // SSTs, for legacy typed-column SSTs (which still use `property_stats`), and
    // for manifests written before this field existed (`serde(default)`).
    #[serde(default)]
    pub per_label_property_stats: Vec<PerLabelPropertyStat>,
}

/// One `(label, property)` statistics entry for an id-primary node SST
/// (RFC 025). `label_id` resolves to a name via the manifest's `label_dict`,
/// the same dictionary the label-index posting counts use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerLabelPropertyStat {
    pub label_id: u32,
    /// Logical property name (no `prop_` prefix).
    pub property: String,
    /// Rows carrying this label for which the property is absent / null /
    /// non-scalar. The cost model derives `non_null_count` as
    /// `node_count - null_count` after the merge.
    pub null_count: u64,
    pub min: Option<StatScalar>,
    pub max: Option<StatScalar>,
    /// Serialised HLL sketch of the non-null scalar values; merged across node
    /// SSTs at read time into `PropStats::ndv`. `None` when no sketchable value
    /// was observed.
    #[serde(default)]
    pub ndv_estimate: Option<HllSketchBytes>,
}

/// Side-car pointer for a single `(SST, unique property)` pair. The
/// sidecar body is a bincode-serialised `BTreeMap<String, NodeId>` —
/// sorted by value string so a future binary-search reader can probe
/// in O(log N) without deserialising the whole map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniquePropertyIndexDescriptor {
    /// Name of the unique property this sidecar indexes (e.g. `id`
    /// for the LDBC SNB anchor pattern).
    pub property: String,
    /// Object-store path relative to the namespace prefix.
    pub path: String,
    /// On-disk size of the sidecar body. Used for budget accounting
    /// when foyer caches the body.
    pub size_bytes: u64,
    /// Number of `(value, NodeId)` entries. Mirrors the SST's
    /// non-tombstone row count modulo nulls; surfaced for diagnostics
    /// and the cache prewarm decision.
    pub entry_count: u64,
}

/// Side-car pointer for a single `(SST, indexed non-unique property)` pair.
/// The sidecar body is a bincode-serialised `BTreeMap<String, Vec<NodeId>>`
/// (a posting list per value, sorted by value string). Unlike the unique
/// sidecar a value may map to several ids, so the reader unions the lists
/// across all in-scope SSTs and confirms each candidate against the node's
/// current value (which also discards tombstoned or value-changed ids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EqualityIndexDescriptor {
    /// Name of the indexed property this sidecar covers.
    pub property: String,
    /// Object-store path relative to the namespace prefix.
    pub path: String,
    /// On-disk size of the sidecar body.
    pub size_bytes: u64,
    /// Number of distinct values in the sidecar (posting-list keys).
    pub distinct_values: u64,
}

/// Side-car pointer for a node SST's label index. The sidecar body is a
/// bincode-serialised `BTreeMap<u32, Vec<[u8; 16]>>` — a posting list of
/// NodeIds per [`LabelId`](namidb_core::LabelId), with both the keys and each
/// posting list sorted. It replaces the old "the SST partition IS the label
/// index" arrangement once a single node SST spans every label: the reader
/// resolves `scan_label(L)` by unioning the posting lists for `L`'s id across
/// all node SSTs (plus the memtable) and confirming each candidate by id.
/// Tombstones contribute nothing; last-LSN-wins at confirm time handles
/// removal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelIndexDescriptor {
    /// Object-store path relative to the namespace prefix.
    pub path: String,
    /// On-disk size of the sidecar body.
    pub size_bytes: u64,
    /// Number of distinct labels (posting-list keys) in the sidecar.
    pub label_count: u64,
    /// Total number of `(label, NodeId)` postings across every key.
    pub posting_count: u64,
    /// Live posting count per `LabelId` — the length of each label's posting
    /// list in the sidecar (tombstones already excluded by the builder). Sorted
    /// by label id. Now that node SSTs are no longer partitioned by label
    /// (`scope` is empty), the cost model recovers each label's `node_count` by
    /// summing these across node SSTs and resolving the id via the manifest's
    /// `label_dict` — a manifest-only, `O(|ssts|)` derivation that needs no
    /// sidecar body read. Empty for manifests written before this field existed
    /// (`serde(default)` round-trips them unchanged).
    #[serde(default)]
    pub per_label_counts: Vec<(u32, u64)>,
}

/// Distance metric a vector index is built for. The build and search must use
/// the same metric; the optimizer matches it against the query's distance
/// function when deciding whether the index can serve the lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorMetric {
    Cosine,
    Dot,
    Euclidean,
}

impl VectorMetric {
    /// The builtin Cypher function name that computes this metric
    /// (`cosine_similarity` / `dot_product` / `euclidean_distance`), so the
    /// optimizer rewrite can match a query against the index.
    pub fn builtin_name(self) -> &'static str {
        match self {
            VectorMetric::Cosine => "cosine_similarity",
            VectorMetric::Dot => "dot_product",
            VectorMetric::Euclidean => "euclidean_distance",
        }
    }
}

/// On-disk representation of the indexed vectors inside a `.vg` body.
///
/// `None` stores full f32 vectors (recall-golden, ~`4·dim` bytes/vector).
/// `Int8` stores per-vector int8 codes + an f32 scale (~`dim+4` bytes/vector,
/// ~4× smaller) — the DiskANN-style memory/storage win for object-storage-first
/// indexes, where the whole `.vg` is fetched per search. int8 navigation and
/// scoring are cosine-only (scale-invariant, exact-in-f32 arithmetic but lossy
/// vs the original embedding), so `Int8` requires `metric: cosine`; recall is
/// slightly lower (the `namidb-ann` floor is ~0.80 vs ~0.85 at f32) and a
/// `WHERE score >= t` threshold compares against the quantized score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorQuantization {
    /// Full-precision f32 vectors (default).
    #[default]
    None,
    /// Per-vector int8 codes + scale (~4× smaller, cosine-only, lossy).
    Int8,
}

/// A registered DiskANN/Vamana vector index over one `(label, property)`.
///
/// `CREATE VECTOR INDEX` appends one of these to [`Manifest::vector_indexes`];
/// the compaction build hook then materializes a `SstKind::VectorGraph` body for
/// it, and the query optimizer rewrites a matching KNN pattern into a
/// `VectorSearch` when (and only when) a descriptor for `(label, property)`
/// with the right metric exists. The index's dimensionality is the source of
/// truth on the schema's `DataType::FloatVector`/`Int8Vector` for the property;
/// `dim` is recorded here for convenience and validated against the schema at
/// build time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorIndexDescriptor {
    /// Index name (unique within the namespace).
    pub name: String,
    /// Node label whose embeddings are indexed.
    pub label: String,
    /// Property holding the embedding vector.
    pub property: String,
    /// Embedding dimensionality.
    pub dim: u32,
    /// Distance metric.
    pub metric: VectorMetric,
    /// Vamana max out-degree (`R`).
    pub r: usize,
    /// Vamana build beam (`L_build`).
    pub l_build: usize,
    /// Vamana α diversification.
    pub alpha: f32,
    /// On-disk vector quantization (`#[serde(default)]` → existing manifests
    /// without the field decode as `None`).
    #[serde(default)]
    pub quantization: VectorQuantization,
}

impl VectorIndexDescriptor {
    /// `true` iff this index covers `(label, property)` with `metric`.
    pub fn matches(&self, label: &str, property: &str, metric: VectorMetric) -> bool {
        self.label == label && self.property == property && self.metric == metric
    }
}

/// A registered full-text (BM25) index over a `(label, properties)` set.
///
/// `CREATE FULLTEXT INDEX` appends one of these to [`Manifest::text_indexes`];
/// the compaction build hook materializes a `SstKind::TextIndex` body for it
/// (concatenating the listed properties per document), and `CALL search.bm25`
/// answers from the index when its `(label, properties)` match the request,
/// falling back to a flat scan otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextIndexDescriptor {
    /// Index name (unique within the namespace).
    pub name: String,
    /// Node label whose text is indexed.
    pub label: String,
    /// Text properties concatenated (in this order) to form each document.
    /// Stored sorted so lookups are order-independent.
    pub properties: Vec<String>,
}

impl TextIndexDescriptor {
    /// Build a descriptor, normalizing `properties` to sorted order so two
    /// descriptors (or a descriptor and a query request) over the same property
    /// set compare equal regardless of the order they were given.
    pub fn new(name: String, label: String, mut properties: Vec<String>) -> Self {
        properties.sort();
        properties.dedup();
        Self {
            name,
            label,
            properties,
        }
    }

    /// `true` iff this index covers `label` over exactly the `properties` set
    /// (order-independent).
    pub fn matches(&self, label: &str, properties: &[String]) -> bool {
        if self.label != label {
            return false;
        }
        let mut req = properties.to_vec();
        req.sort();
        req.dedup();
        self.properties == req
    }
}

/// JSON serde helper: `[u8; 16]` ↔ base64-standard string.
mod serde_key16 {
    use base64::Engine as _;
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &[u8; 16], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(v))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 16], D::Error> {
        let raw = String::deserialize(d)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&raw)
            .map_err(|e| D::Error::custom(format!("base64 decode: {e}")))?;
        if bytes.len() != 16 {
            return Err(D::Error::custom(format!(
                "expected 16 bytes, got {}",
                bytes.len()
            )));
        }
        let mut out = [0u8; 16];
        out.copy_from_slice(&bytes);
        Ok(out)
    }
}

/// WAL segment that still has un-flushed records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalSegmentDescriptor {
    pub seq: u64,
    pub path: String,
    /// Inclusive max LSN durably written in this segment.
    pub last_lsn: u64,
}

/// Wraps an [`ObjectStore`] with the manifest CAS protocol bound to a single
/// namespace.
#[derive(Clone)]
pub struct ManifestStore {
    store: Arc<dyn ObjectStore>,
    paths: NamespacePaths,
}

impl std::fmt::Debug for ManifestStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManifestStore")
            .field("paths", &self.paths)
            .finish()
    }
}

/// Result of [`ManifestStore::load_current`].
#[derive(Debug, Clone)]
pub struct LoadedManifest {
    pub pointer: ManifestPointer,
    /// E-tag of the **pointer** object as observed during load. CAS commits
    /// must thread this back so the storage engine can detect that the
    /// canonical pointer moved underneath us.
    pub pointer_etag: Option<String>,
    /// E-tag-style version (some backends use this instead of e-tag).
    pub pointer_version: Option<String>,
    pub manifest: Manifest,
    /// Pre-computed sorted-by-min-key index over `manifest.ssts`, bucketed
    /// by `(kind, scope)`. The read path uses this to skip the linear scan
    /// every lookup used to do. Wrapped in `Arc` so cloning a
    /// `LoadedManifest` (which happens once per `Snapshot`) is cheap.
    pub index: Arc<DescriptorIndex>,
}

impl LoadedManifest {
    /// Build a `LoadedManifest` and the descriptor index over its SSTs.
    /// All three constructors in this module go through here so the index
    /// is always present and consistent with `manifest.ssts`.
    pub fn new(
        pointer: ManifestPointer,
        pointer_etag: Option<String>,
        pointer_version: Option<String>,
        manifest: Manifest,
    ) -> Self {
        let index = Arc::new(DescriptorIndex::build(&manifest.ssts));
        Self {
            pointer,
            pointer_etag,
            pointer_version,
            manifest,
            index,
        }
    }
}

impl ManifestStore {
    pub fn new(store: Arc<dyn ObjectStore>, paths: NamespacePaths) -> Self {
        Self { store, paths }
    }

    pub fn paths(&self) -> &NamespacePaths {
        &self.paths
    }

    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// Initialise an empty namespace.
    ///
    /// Writes `manifest/v0.json` and `manifest/pointer/p0.json` for the first
    /// time, both with `PutMode::Create`. Fails with [`Error::Precondition`] if
    /// either object already exists — pointing at an already-initialised
    /// namespace.
    #[instrument(skip(self), fields(namespace = %self.paths.namespace()))]
    pub async fn bootstrap(&self, writer_id: Uuid) -> Result<LoadedManifest> {
        let manifest = Manifest::empty(Epoch::ZERO, writer_id);
        let manifest_path = self.paths.manifest_version(manifest.version);
        // Body PUT: tolerate `AlreadyExists`. A prior bootstrap may have
        // written `v0.json` and then crashed BEFORE the pointer landed — the
        // half-write this method must complete rather than wedge on (without
        // this, that namespace could neither bootstrap nor load: `v0.json`
        // exists so bootstrap errored, but no pointer exists so load_current
        // returned NotFound). The pointer PUT below still guards a genuinely
        // bootstrapped namespace.
        match self
            .put_create(&manifest_path, serde_json::to_vec(&manifest)?.into())
            .await
        {
            Ok(_) => {}
            Err(Error::ObjectStore(object_store::Error::AlreadyExists { .. })) => {}
            Err(other) => return Err(other),
        }

        let pointer = ManifestPointer {
            version: manifest.version,
            epoch: manifest.epoch,
            manifest_path: manifest_path.as_ref().to_string(),
        };
        let pointer_path = self.paths.pointer_version(manifest.version);
        let pointer_bytes: Bytes = serde_json::to_vec(&pointer)?.into();
        let put_res = self
            .put_create(&pointer_path, pointer_bytes.clone())
            .await
            .map_err(|e| match e {
                // The pointer already exists: this is a genuinely bootstrapped
                // (in-use) namespace, NOT the half-write we just recovered.
                // Refuse to hand back a v0 LoadedManifest that would shadow
                // any higher versions.
                Error::ObjectStore(object_store::Error::AlreadyExists { .. }) => {
                    Error::precondition(format!(
                        "namespace '{}' already bootstrapped: pointer {} exists",
                        self.paths.namespace(),
                        pointer_path
                    ))
                }
                other => other,
            })?;

        // Publish the advisory `current.json` for the freshly-bootstrapped
        // namespace (see `write_advisory_current`).
        self.write_advisory_current(pointer_bytes).await?;

        Ok(LoadedManifest::new(
            pointer,
            put_res.e_tag,
            put_res.version,
            manifest,
        ))
    }

    /// Load a specific historical manifest version's immutable body
    /// (`manifest/v{version}.json`). Manifest bodies are written once per
    /// commit / flush / compaction and never mutated, so every version from
    /// genesis to current is readable. The horizon-aware sweep uses this to
    /// union the live object set across every retained version.
    pub async fn load_manifest_at(&self, version: u64) -> Result<Manifest> {
        let path = self.paths.manifest_version(version);
        let res = self.store.get(&path).await?;
        let body = res.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&body)?;
        Ok(manifest)
    }

    /// Resolve the current pointer, then read the manifest it points at.
    #[instrument(skip(self), fields(namespace = %self.paths.namespace()))]
    pub async fn load_current(&self) -> Result<LoadedManifest> {
        let (pointer, pointer_etag, pointer_version) = self.load_pointer().await?;

        let manifest_path = Path::from(pointer.manifest_path.clone());
        let manifest_res = self.store.get(&manifest_path).await?;
        let manifest_body = manifest_res.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&manifest_body)?;

        if manifest.version != pointer.version {
            return Err(Error::Corrupted {
                path: manifest_path.as_ref().to_string(),
                detail: format!(
                    "manifest version {} does not match pointer version {}",
                    manifest.version, pointer.version
                ),
            });
        }

        Ok(LoadedManifest::new(
            pointer,
            pointer_etag,
            pointer_version,
            manifest,
        ))
    }

    /// Resolve the current [`ManifestPointer`] and the e-tag/version of the
    /// pointer object it was read from (RFC-029).
    ///
    /// The authoritative source is the Create-only family
    /// `manifest/pointer/p<N>.json`: the current pointer is the highest `N`
    /// present. A namespace bootstrapped before the family existed (or a
    /// snapshot produced by the pre-RFC backup path) has no family and is read
    /// through the legacy `manifest/current.json`. A namespace with neither is
    /// uninitialised and surfaces the object store's `NotFound`, which
    /// `WriterSession::open` turns into a bootstrap.
    async fn load_pointer(&self) -> Result<(ManifestPointer, Option<String>, Option<String>)> {
        let path = match self.max_pointer_version().await? {
            Some(max_n) => {
                let n = self.probe_pointer_forward(max_n).await?;
                self.paths.pointer_version(n)
            }
            None => self.paths.current_pointer(),
        };
        let res = self.store.get(&path).await?;
        let etag = res.meta.e_tag.clone();
        let version = res.meta.version.clone();
        let body = res.bytes().await?;
        let pointer: ManifestPointer = serde_json::from_slice(&body)?;
        Ok((pointer, etag, version))
    }

    /// Highest `N` in the Create-only pointer family, or `None` if the family
    /// is empty (a legacy or uninitialised namespace).
    ///
    /// On eventually-consistent-LIST stores, an empty/stale LIST does **not**
    /// prove non-existence. We recover via two non-LIST probes, both using
    /// GET/HEAD of a specific key (read-after-write consistent everywhere):
    ///
    /// 1. HEAD `p0.json` — closes the window for a *fresh* namespace whose
    ///    LIST has simply not caught up to the just-created pointer.
    /// 2. GET `current.json` (the advisory) — closes the window for an *aged*
    ///    namespace where the janitor has reclaimed `p0` (and every pointer
    ///    below the retention horizon). Without this, a stale empty LIST made
    ///    the family look empty, `load_current` returned `NotFound`, and
    ///    `WriterSession::open` re-bootstrapped a live namespace (data loss).
    ///    The advisory is written on every commit/bootstrap; its `.version`
    ///    is a lower bound the forward probe then advances to true current.
    async fn max_pointer_version(&self) -> Result<Option<u64>> {
        let dir = self.paths.pointer_dir();
        let mut stream = self.store.list(Some(&dir));
        let mut max: Option<u64> = None;
        while let Some(meta) = stream.try_next().await? {
            if let Some(v) = parse_pointer_version(&meta.location) {
                max = Some(max.map_or(v, |m| m.max(v)));
            }
        }
        if max.is_none() {
            // Fresh namespace: LIST lagged behind a just-created p0.
            if self
                .store
                .head(&self.paths.pointer_version(0))
                .await
                .is_ok()
            {
                max = Some(0);
            } else if let Ok(res) = self.store.get(&self.paths.current_pointer()).await {
                // Aged namespace: p0 was reclaimed below the horizon, but the
                // advisory current.json still names a valid version. Only trust
                // it when the pointer at that version actually exists — a legacy
                // namespace has current.json but NO pointer family, and must fall
                // through to the None branch (which reads current.json directly).
                if let Ok(body) = res.bytes().await {
                    if let Ok(p) = serde_json::from_slice::<ManifestPointer>(&body) {
                        if self
                            .store
                            .head(&self.paths.pointer_version(p.version))
                            .await
                            .is_ok()
                        {
                            max = Some(p.version);
                        }
                    }
                }
            }
        }
        Ok(max)
    }

    /// Bounded forward HEAD probe from `n`. A LIST that has not yet caught up
    /// to a just-created `p<n+1>.json` (some S3-compatible stores are only
    /// eventually consistent for LIST, though GET/HEAD of a specific key is
    /// read-after-write consistent everywhere we target) would otherwise hand
    /// a writer a stale base. Galloping a few HEADs closes that window.
    ///
    /// **Note:** The janitor GC may delete pointers below the retention horizon,
    /// creating gaps at the low end of the sequence. A stale LIST that returns
    /// `[p5, p6, p7]` when `p3` exists (but was GC'd) can cause the forward
    /// probe to skip from `None` → `p5`, missing the true current. This is
    /// tolerated because the caller will fail the epoch check and retry; the
    /// bounded probe is a best-effort mitigation, not a guarantee. Bounded as
    /// a defensive backstop.
    async fn probe_pointer_forward(&self, mut n: u64) -> Result<u64> {
        // Higher bound to tolerate aggressive LIST staleness on high-throughput
        // namespaces. At 100 commits/sec, a 5-sec stale LIST would lag ~500
        // versions; 8192 gives significant headroom while still bounded as a
        // defensive backstop (the probe is cheap HEAD-only, no body reads).
        const MAX_PROBE: u32 = 8192;
        let start = n;
        let mut probed = 0u32;
        // Track whether we found the end of the family (a NotFound gap) vs.
        // ran out of probe budget. Distinguishing these is load-bearing: a
        // namespace whose true current sits *exactly* MAX_PROBE steps ahead
        // exits the loop on the budget guard on the same iteration it would
        // have found the gap, so `probed >= MAX_PROBE` alone would wrongly
        // fail it. We only fail when the gap was genuinely not reached.
        let mut gap_found = false;
        while probed < MAX_PROBE {
            let next = n.saturating_add(1);
            match self.store.head(&self.paths.pointer_version(next)).await {
                Ok(_) => n = next,
                Err(object_store::Error::NotFound { .. }) => {
                    gap_found = true;
                    break;
                }
                Err(e) => return Err(Error::ObjectStore(e)),
            }
            probed += 1;
        }
        if !gap_found {
            // The probe ran the full window WITHOUT finding a gap, which means
            // the true current pointer is >8192 versions ahead of the lower
            // bound (`start`) we began from. Handing back `n` would serve a
            // stale pointer (and a stale manifest). Fail closed with a
            // RETRYABLE error: by the next attempt the LIST / advisory
            // `current.json` (read-after-write consistent) has advanced the
            // lower bound close enough to current that the probe terminates.
            tracing::warn!(
                from_version = start,
                max_probe = MAX_PROBE,
                "manifest pointer forward-probe exhausted its window; returning retryable stale error"
            );
            return Err(Error::PointerResolveStale);
        }
        Ok(n)
    }

    /// Commit a new manifest version using the two-step CAS protocol:
    ///
    /// 1. `PutMode::Create` the new immutable manifest body.
    /// 2. `PutMode::Create` the pointer `manifest/pointer/p<v+1>.json`
    ///    (RFC-029) — the same conditional primitive as step 1.
    ///
    /// On a lost CAS race we return [`Error::ManifestCommitCas`]; the caller
    /// must reload, fence-check, and retry from a fresh base.
    ///
    /// Callers that need to overlap the body PUT with another
    /// independent object-store write (e.g. the WAL segment that the
    /// manifest will reference) can instead drive the two phases
    /// directly through [`Self::put_body`] + [`Self::cas_pointer`].
    #[instrument(
 skip(self, fence, new_manifest, base),
 fields(
 namespace = %self.paths.namespace(),
 base_version = base.pointer.version,
 new_version = new_manifest.version,
 )
 )]
    pub async fn commit(
        &self,
        fence: &WriterFence,
        base: &LoadedManifest,
        new_manifest: Manifest,
    ) -> Result<LoadedManifest> {
        let pointer = self.put_body(fence, base, &new_manifest).await?;
        self.cas_pointer(fence, base, new_manifest, pointer).await
    }

    /// Phase 1 of [`Self::commit`]: PUT the immutable manifest body.
    /// Returns the [`ManifestPointer`] the caller will later CAS into
    /// place via [`Self::cas_pointer`].
    ///
    /// Splitting the commit lets `WriterSession::commit_batch`
    /// pipeline the body PUT against the independent WAL segment PUT;
    /// in the common case that turns two serial round-trips into one.
    /// If the WAL append fails after this method succeeded, the body
    /// stays orphaned but harmless — the pointer never moved, and the
    /// next manifest version will overwrite the reference.
    pub async fn put_body(
        &self,
        fence: &WriterFence,
        base: &LoadedManifest,
        new_manifest: &Manifest,
    ) -> Result<ManifestPointer> {
        fence.assert_alive(base.manifest.epoch)?;
        if new_manifest.version != base.manifest.version + 1 {
            return Err(Error::invariant(format!(
                "new manifest version {} must be {} (base + 1)",
                new_manifest.version,
                base.manifest.version + 1
            )));
        }
        if new_manifest.epoch < base.manifest.epoch {
            return Err(Error::invariant(format!(
                "new epoch {} cannot regress below base epoch {}",
                new_manifest.epoch, base.manifest.epoch
            )));
        }

        let manifest_path = self.paths.manifest_version(new_manifest.version);
        let body: Bytes = serde_json::to_vec(new_manifest)?.into();
        debug!(path = %manifest_path, "writing immutable manifest body");
        match self.put_create(&manifest_path, body.clone()).await {
            Ok(_) => {}
            Err(Error::ObjectStore(object_store::Error::AlreadyExists { .. })) => {
                // `AlreadyExists` does not always mean a competitor: the
                // existing body can be OUR OWN from a prior attempt of this
                // same commit — a retry after the create landed but its
                // response was lost, or a retry after the pipelined WAL PUT
                // failed while this body PUT succeeded (see
                // `WriterSession::commit_batch`). Failing such a retry as a
                // CAS loss strands an orphan body at `base + 1` that no
                // writer can supersede (versions are Create-only), wedging
                // the namespace until `claim_writer`'s stall repair runs.
                // Adopt the body instead and proceed to the pointer CAS.
                if !self
                    .existing_body_is_ours(&manifest_path, &body, new_manifest)
                    .await?
                {
                    // A genuine competitor chose the same version. Before
                    // raising a plain CAS loss, reload to discover whether
                    // the namespace has actually advanced past our epoch —
                    // in that case we are fenced and the caller must drop
                    // this writer state, not retry.
                    let reloaded = self.load_current().await?;
                    if reloaded.manifest.epoch > fence.epoch {
                        return Err(Error::Fenced {
                            mine: fence.epoch.as_u64(),
                            current: reloaded.manifest.epoch.as_u64(),
                        });
                    }
                    return Err(Error::ManifestCommitCas {
                        expected: base.pointer.version,
                        found: new_manifest.version,
                    });
                }
                debug!(
                    path = %manifest_path,
                    "manifest body already durable from a prior attempt; adopting it"
                );
            }
            Err(other) => return Err(other),
        }

        Ok(ManifestPointer {
            version: new_manifest.version,
            epoch: new_manifest.epoch,
            manifest_path: manifest_path.as_ref().to_string(),
        })
    }

    /// `true` iff the durable body at `manifest_path` is the one this writer
    /// attempted to PUT: byte-identical to `attempted`, or equal to
    /// `new_manifest` in every field except `created_at` (each retry
    /// re-stamps the timestamp; it is audit-only). Manifests embed the
    /// writer's `writer_id`, so equality modulo `created_at` proves the body
    /// is ours — no other writer can produce it.
    async fn existing_body_is_ours(
        &self,
        manifest_path: &Path,
        attempted: &Bytes,
        new_manifest: &Manifest,
    ) -> Result<bool> {
        let existing = match self.store.get(manifest_path).await {
            Ok(res) => res.bytes().await?,
            // Deleted between the failed create and this read (a concurrent
            // claim repair); report the CAS loss and let the caller retry
            // from a fresh base.
            Err(object_store::Error::NotFound { .. }) => return Ok(false),
            Err(e) => return Err(Error::ObjectStore(e)),
        };
        if existing == *attempted {
            return Ok(true);
        }
        let Ok(mut parsed) = serde_json::from_slice::<Manifest>(&existing) else {
            return Ok(false);
        };
        parsed.created_at = new_manifest.created_at;
        Ok(parsed == *new_manifest)
    }

    /// Phase 2 of [`Self::commit`]: publish the pointer for the body
    /// previously written by [`Self::put_body`] by creating
    /// `manifest/pointer/p<version>.json` with `PutMode::Create` (RFC-029).
    /// The pointer file for a version is the linearization point: whoever
    /// creates it first owns that version. Returns the freshly-loaded manifest
    /// on success.
    pub async fn cas_pointer(
        &self,
        fence: &WriterFence,
        base: &LoadedManifest,
        new_manifest: Manifest,
        pointer: ManifestPointer,
    ) -> Result<LoadedManifest> {
        let pointer_path = self.paths.pointer_version(pointer.version);
        let pointer_bytes: Bytes = serde_json::to_vec(&pointer)?.into();
        let put_res = match self.put_create(&pointer_path, pointer_bytes.clone()).await {
            Ok(r) => r,
            Err(Error::ObjectStore(object_store::Error::AlreadyExists { .. })) => {
                // Another writer already published this version's pointer.
                // Reload to surface the actual state. Same fence/CAS split as
                // the body branch in `put_body`: an advanced epoch means we
                // are fenced and must drop this writer, not retry.
                let reloaded = self.load_current().await?;
                if reloaded.manifest.epoch > fence.epoch {
                    return Err(Error::Fenced {
                        mine: fence.epoch.as_u64(),
                        current: reloaded.manifest.epoch.as_u64(),
                    });
                }
                warn!(
                    expected = base.pointer.version,
                    found = reloaded.pointer.version,
                    "manifest pointer create lost (version already published)"
                );
                return Err(Error::ManifestCommitCas {
                    expected: base.pointer.version,
                    found: reloaded.pointer.version,
                });
            }
            Err(other) => return Err(other),
        };

        // Publish the advisory `current.json` so the version is findable via
        // a non-LIST read on EC stores even after the janitor reclaims `p0`.
        // See `write_advisory_current`. A failure here is a commit failure:
        // the pointer Create already succeeded (the version is durable), but
        // until the advisory lands it is not reliably discoverable, so we
        // surface the error rather than leave a half-published version.
        self.write_advisory_current(pointer_bytes.clone()).await?;

        Ok(LoadedManifest::new(
            pointer,
            put_res.e_tag,
            put_res.version,
            new_manifest,
        ))
    }

    /// Atomically bump epoch under CAS. Used by a new writer when it claims
    /// the namespace, fencing whoever was there before.
    ///
    /// Returns the loaded manifest at the new epoch alongside a fresh fence
    /// that the caller should hold for the lifetime of its writer session.
    #[instrument(skip(self), fields(namespace = %self.paths.namespace()))]
    pub async fn claim_writer(&self) -> Result<(LoadedManifest, WriterFence)> {
        // A genuine concurrent claim resolves quickly: the winner advances
        // the pointer, so a reloaded base sees a higher version within a
        // couple of rounds and we make progress. A CAS loss where the
        // pointer NEVER advances is the signature of an orphan manifest
        // body at `base.version + 1` — a writer wrote the body via
        // `PutMode::Create` but crashed before the pointer CAS (e.g. a
        // transient error in `cas_pointer`, or mid-`commit_batch` between
        // the body PUT and the pointer CAS). Nobody can supersede that
        // version under `Create`, so an unbounded loop would spin forever.
        // Bound the *stall* (consecutive CAS losses at the same pointer
        // version), then run the orphan repair: publish the orphan's pointer
        // when it is a complete, durable commit, or delete it otherwise (see
        // `repair_stalled_commit` for the decision rule and its safety
        // argument). The stall rounds double as a grace period for a live
        // writer mid-commit to finish on its own.
        const MAX_STALLED_ROUNDS: usize = 8;
        // A repair can legitimately need more than one pass (fence the WAL
        // slot + delete the body, then commit fresh). If the stall persists
        // beyond a few passes, something keeps re-creating unpublishable
        // bodies — surface the terminal error instead of looping forever.
        const MAX_REPAIR_PASSES: usize = 4;
        let mut stalled_rounds = 0usize;
        let mut repair_passes = 0usize;
        let mut last_version: Option<u64> = None;
        loop {
            let base = self.load_current().await?;
            let mut new_manifest = base.manifest.next_version(Uuid::now_v7());
            new_manifest.epoch = base.manifest.epoch.next();
            let fence = WriterFence::new(new_manifest.epoch);
            // We claim the epoch with `fence.epoch == new_manifest.epoch`, so
            // the alive check inside `commit` happens against the *base*
            // epoch which is one less — that always passes.
            let pretend = WriterFence {
                epoch: base.manifest.epoch,
                writer_id: fence.writer_id,
            };
            match self.commit(&pretend, &base, new_manifest).await {
                Ok(loaded) => return Ok((loaded, fence)),
                Err(Error::ManifestCommitCas { .. }) => {
                    // Reload and retry while we keep making progress (the
                    // pointer version advances). If it stalls at the same
                    // version, we are colliding with an orphan body: repair
                    // it, then keep claiming.
                    if last_version == Some(base.pointer.version) {
                        stalled_rounds += 1;
                        if stalled_rounds >= MAX_STALLED_ROUNDS {
                            if repair_passes >= MAX_REPAIR_PASSES {
                                return Err(Error::OrphanManifestBody {
                                    version: base.pointer.version.saturating_add(1),
                                });
                            }
                            repair_passes += 1;
                            self.repair_stalled_commit(&base).await?;
                            stalled_rounds = 0;
                        }
                    } else {
                        last_version = Some(base.pointer.version);
                        stalled_rounds = 0;
                    }
                    continue;
                }
                Err(other) => return Err(other),
            }
        }
    }

    /// Repair a stalled commit at `base.pointer.version + 1`: a manifest
    /// body exists there but its pointer was never created, so every
    /// writer's Create at that version loses and the pointer can never
    /// advance — without intervention the namespace is permanently
    /// unwritable. Decision rule:
    ///
    /// - **Adopt** (publish `p<N+1>`) when the body is a well-formed
    ///   manifest extending the current lineage AND every object it
    ///   references beyond the current manifest's set is durable — for WAL
    ///   segments, durable with exactly the declared `last_lsn` (recovery
    ///   refuses a mismatch). Publishing writes the exact pointer bytes the
    ///   interrupted writer's own `cas_pointer` would have written (the
    ///   content is deterministic), so it is safe even if that writer is
    ///   still alive mid-commit: its pointer Create merely observes
    ///   `AlreadyExists` and reports a CAS loss / fence, which its contract
    ///   already maps to "drop the session and reopen". Acked data is
    ///   unaffected; the interrupted commit's unacked records become
    ///   durable, which at-least-once semantics permit.
    /// - **Delete** the body otherwise. Deletion is safe against a live
    ///   writer because it is only reached when (a) the body is not a
    ///   protocol-written manifest (no writer holds it mid-commit), or
    ///   (b) a referenced WAL slot is empty — in which case we first FENCE
    ///   the slot by Create-ing an empty segment there, so the interrupted
    ///   commit's WAL PUT can never succeed and its `cas_pointer` (which
    ///   runs only after a WAL success) can never fire — or (c) the WAL
    ///   slot's content mismatches the descriptor, which only sequential
    ///   failed attempts of an already-abandoned commit can produce (slots
    ///   are Create-once), or (d) a referenced SST is missing, which — since
    ///   flush/compaction PUT every SST strictly before the manifest body —
    ///   means the janitor swept the unpublished orphan's outputs long after
    ///   any in-flight commit window.
    async fn repair_stalled_commit(&self, base: &LoadedManifest) -> Result<()> {
        let version = base.pointer.version.saturating_add(1);
        let body_path = self.paths.manifest_version(version);

        // The stall signature is "body exists, pointer does not". Re-check
        // the pointer: if it landed while we were losing CAS rounds, the
        // namespace is advancing and there is nothing to repair.
        match self.store.head(&self.paths.pointer_version(version)).await {
            Ok(_) => return Ok(()),
            Err(object_store::Error::NotFound { .. }) => {}
            Err(e) => return Err(Error::ObjectStore(e)),
        }

        let body = match self.store.get(&body_path).await {
            Ok(res) => res.bytes().await?,
            // The orphan vanished (a concurrent repairer deleted it); the
            // claim loop's next commit attempt finds the version free.
            Err(object_store::Error::NotFound { .. }) => return Ok(()),
            Err(e) => return Err(Error::ObjectStore(e)),
        };

        match self.orphan_ready_to_publish(base, version, &body).await? {
            Some(orphan) => {
                let pointer = ManifestPointer {
                    version,
                    epoch: orphan.epoch,
                    manifest_path: body_path.as_ref().to_string(),
                };
                let pointer_bytes: Bytes = serde_json::to_vec(&pointer)?.into();
                match self
                    .put_create(&self.paths.pointer_version(version), pointer_bytes.clone())
                    .await
                {
                    Ok(_) => {
                        warn!(
                            version,
                            "adopted orphan manifest body: published the pointer of an \
                             interrupted commit"
                        );
                        self.write_advisory_current(pointer_bytes).await?;
                    }
                    // Someone (possibly the interrupted writer itself)
                    // published first; the pointer advances either way.
                    Err(Error::ObjectStore(object_store::Error::AlreadyExists { .. })) => {}
                    Err(e) => return Err(e),
                }
            }
            None => {
                warn!(
                    version,
                    "deleting unpublishable orphan manifest body (incomplete interrupted commit)"
                );
                match self.store.delete(&body_path).await {
                    Ok(()) => {}
                    Err(object_store::Error::NotFound { .. }) => {}
                    Err(e) => return Err(Error::ObjectStore(e)),
                }
            }
        }
        Ok(())
    }

    /// Decide whether the orphan body at `version` is a complete commit that
    /// may be published. Returns the parsed manifest when it is; `None` when
    /// the body must be deleted instead. When a referenced WAL slot is
    /// empty, this FENCES it (Create of an empty segment) before answering,
    /// so a `None` answer guarantees the interrupted commit can never
    /// complete concurrently with the caller's deletion.
    async fn orphan_ready_to_publish(
        &self,
        base: &LoadedManifest,
        version: u64,
        body: &[u8],
    ) -> Result<Option<Manifest>> {
        // Not a manifest, or one that does not extend the current lineage:
        // no protocol writer produced it for this slot (put_body validates
        // version and epoch against the same base before writing).
        let Ok(orphan) = serde_json::from_slice::<Manifest>(body) else {
            return Ok(None);
        };
        if orphan.version != version || orphan.epoch < base.manifest.epoch {
            return Ok(None);
        }

        // Every SST the orphan adds over the base must still be durable.
        // Flush/compaction PUT SSTs strictly before the manifest body, so
        // they existed when the body was created; a missing one means the
        // janitor has since swept the unpublished orphan's outputs.
        let base_ssts: HashSet<&str> = base.manifest.ssts.iter().map(|s| s.path.as_str()).collect();
        for sst in orphan
            .ssts
            .iter()
            .filter(|s| !base_ssts.contains(s.path.as_str()))
        {
            // SST descriptor paths are relative to the namespace prefix
            // (same resolution the read path uses).
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), sst.path);
            match self.store.head(&Path::from(absolute)).await {
                Ok(_) => {}
                Err(object_store::Error::NotFound { .. }) => return Ok(None),
                Err(e) => return Err(Error::ObjectStore(e)),
            }
        }

        // Every WAL segment the orphan adds over the base must be durable
        // with exactly the declared `last_lsn`. `commit_batch` pipelines the
        // WAL PUT with the body PUT (and `commit_body_first` PUTs the body
        // first), so a body can exist whose WAL segment never landed —
        // publishing it would wedge recovery on a missing/mismatched
        // segment instead.
        let base_wal: HashSet<u64> = base.manifest.wal_segments.iter().map(|s| s.seq).collect();
        for seg in orphan
            .wal_segments
            .iter()
            .filter(|s| !base_wal.contains(&s.seq))
        {
            let seg_path = self.paths.wal_segment(seg.seq);
            if seg.path != seg_path.as_ref() {
                // Descriptor/path mismatch: not written by this namespace's
                // protocol. Recovery would read `seg_path` and miss.
                return Ok(None);
            }
            if !self
                .wal_slot_matches(seg.seq, seg.last_lsn, &seg_path)
                .await?
            {
                return Ok(None);
            }
        }
        Ok(Some(orphan))
    }

    /// `true` iff the WAL object at `seg_path` is durable and carries
    /// exactly the records the manifest descriptor declares (recovery
    /// matches on `last_lsn`). When the slot is empty, first fence it by
    /// Create-ing an *empty* segment there: WAL slots are Create-once, so
    /// winning that create atomically guarantees the interrupted commit's
    /// own WAL PUT — possibly still in flight — fails `AlreadyExists`, and
    /// without a WAL success it never reaches `cas_pointer`. Losing the
    /// create means the real segment just landed; re-read and re-judge. The
    /// fence sentinel is a valid empty segment (`last_lsn = 0`, which no
    /// real commit declares), so a peer repairer that reads it reaches the
    /// same "mismatch → delete" verdict, and the janitor eventually sweeps
    /// it as an unreferenced segment.
    async fn wal_slot_matches(
        &self,
        seq: u64,
        declared_last_lsn: u64,
        seg_path: &Path,
    ) -> Result<bool> {
        loop {
            let bytes = match self.store.get(seg_path).await {
                Ok(res) => res.bytes().await?,
                Err(object_store::Error::NotFound { .. }) => {
                    let sentinel = WalSegment::new(seq).encode();
                    let opts = PutOptions::from(PutMode::Create);
                    match self
                        .store
                        .put_opts(seg_path, PutPayload::from(sentinel), opts)
                        .await
                    {
                        Ok(_) => {
                            warn!(seq, path = %seg_path, "fenced missing WAL slot of an interrupted commit");
                            return Ok(false);
                        }
                        // Raced the in-flight WAL PUT; the slot settled —
                        // re-read it and judge the real content.
                        Err(object_store::Error::AlreadyExists { .. }) => continue,
                        Err(e) => return Err(Error::ObjectStore(e)),
                    }
                }
                Err(e) => return Err(Error::ObjectStore(e)),
            };
            return Ok(match WalSegment::decode(seq, bytes) {
                Ok(segment) => !segment.is_empty() && segment.last_lsn() == declared_last_lsn,
                Err(_) => false,
            });
        }
    }

    /// Helper: `put_opts(path, payload, PutMode::Create)`.
    async fn put_create(&self, path: &Path, body: Bytes) -> Result<object_store::PutResult> {
        let opts = PutOptions::from(PutMode::Create);
        Ok(self
            .store
            .put_opts(path, PutPayload::from(body), opts)
            .await?)
    }

    /// Write the advisory `current.json` carrying the same body as the
    /// just-published pointer `p<N>.json`.
    ///
    /// This is **not** part of the CAS contract — the Create-only pointer
    /// family remains authoritative. It exists to close the data-loss window
    /// in [`Self::max_pointer_version`] on eventually-consistent-LIST stores:
    /// once the janitor reclaims `p0` (and every pointer below the retention
    /// horizon), a stale empty LIST makes the family look empty. Without an
    /// authoritative non-LIST signal, `load_current` fell through to a
    /// `current.json` that post-RFC commits never wrote, returned `NotFound`,
    /// and `WriterSession::open` re-bootstrapped a **live** namespace —
    /// silent data loss on exactly the EC stores RFC-029 targets.
    ///
    /// The advisory is a plain PUT (`Overwrite`), the one universally
    /// supported write primitive (no conditional needed), so it adds no
    /// portability dependency. It is last-writer-wins and at worst slightly
    /// stale; even a stale advisory points at a valid immutable manifest
    /// body, and the forward probe in `load_pointer` then advances to the
    /// true current. Its failure is treated as a commit failure: until it is
    /// durable the version is not reliably findable on an EC store.
    async fn write_advisory_current(&self, pointer_bytes: Bytes) -> Result<()> {
        let path = self.paths.current_pointer();
        self.store
            .put(&path, PutPayload::from(pointer_bytes))
            .await
            .map_err(Error::ObjectStore)?;
        Ok(())
    }
}

/// Parse `N` out of a `p<16-hex>.json` pointer filename (RFC-029). Returns
/// `None` for `current.json` (the legacy pointer) and any other key, so it is
/// safe to run over a `manifest/pointer/` LIST.
fn parse_pointer_version(location: &Path) -> Option<u64> {
    location
        .filename()
        .and_then(|f| f.strip_prefix('p'))
        .and_then(|f| f.strip_suffix(".json"))
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
}

/// Sorted-by-min-key index over [`Manifest::ssts`], bucketed by
/// `(SstKind, scope)`.
///
/// Built once when a `LoadedManifest` is constructed; reused across
/// every `Snapshot` lookup on that manifest. Lets the read path skip
/// the O(N) linear scan over `Manifest::ssts` that
/// `Snapshot::lookup_node` used to do — at 100 M nodes / 10 SSTs that
/// scan cost ~1 ms per warm lookup and pushed the warm p50 over the
/// plan gate by 0.58 ms. Extrapolated to ~100 SSTs the cost is
/// ~10 ms, i.e. the entire gate consumed by descriptor iteration.
///
/// ## Layout
///
/// One bucket per `(kind, scope)` pair (e.g. `(Nodes, "Person")`).
/// Each bucket stores indices into the parent `Manifest::ssts` vec,
/// sorted ascending by `min_key`. Two consequences:
///
/// 1. **Disjoint ranges** (post-compaction L1+, where the writer
/// guarantees ranges don't overlap) → binary-search to exactly one
/// candidate.
/// 2. **Overlapping ranges** (L0, where writers may flush SSTs whose
/// `(min_key, max_key)` straddle each other) → binary-search to the
/// first descriptor whose `min_key > target`, then walk backwards
/// collecting every earlier descriptor whose `max_key >= target`.
/// In practice L0 has a small bounded count, so the walk is short.
#[derive(Debug, Default)]
pub struct DescriptorIndex {
    buckets: HashMap<(SstKind, String), Vec<usize>>,
}

impl DescriptorIndex {
    /// Bucket `ssts` by `(kind, scope)` and sort each bucket by `min_key`.
    pub fn build(ssts: &[SstDescriptor]) -> Self {
        let mut buckets: HashMap<(SstKind, String), Vec<usize>> = HashMap::new();
        for (i, d) in ssts.iter().enumerate() {
            buckets
                .entry((d.kind, d.scope.clone()))
                .or_default()
                .push(i);
        }
        for v in buckets.values_mut() {
            v.sort_by_key(|&i| ssts[i].min_key);
        }
        Self { buckets }
    }

    /// Return descriptor indices (into `ssts`) whose `(min_key, max_key)`
    /// range straddles `target` for the given `(kind, scope)`. The
    /// caller still has to bloom-probe + body-fetch to confirm — this
    /// only prunes obvious non-candidates.
    pub fn lookup_candidates(
        &self,
        ssts: &[SstDescriptor],
        kind: SstKind,
        scope: &str,
        target: &[u8; 16],
    ) -> Vec<usize> {
        // Borrowed key for the HashMap lookup — cheap because the
        // String inside the key is owned.
        let bucket = match self.buckets.get(&(kind, scope.to_string())) {
            Some(b) => b,
            None => return Vec::new(),
        };
        // `partition_point` returns the first i where the predicate is
        // false, i.e. the first descriptor with `min_key > target`.
        // Everything to the left has `min_key <= target` and is a
        // potential candidate (still has to clear the `max_key >= target`
        // check, because L0 ranges can be wider on one side).
        let after = bucket.partition_point(|&idx| ssts[idx].min_key <= *target);
        let mut out = Vec::new();
        for j in (0..after).rev() {
            let idx = bucket[j];
            if ssts[idx].max_key >= *target {
                out.push(idx);
            }
            // We deliberately do NOT break: ranges sorted by `min_key`
            // are not necessarily sorted by `max_key` under L0 overlap.
            // In the disjoint case (L1+) `after - 1` is the only hit
            // anyway, so the loop body runs once.
        }
        out
    }

    /// All descriptor indices for `(kind, scope)` in ascending `min_key`
    /// order. Used by full-label scans like `Snapshot::scan_label`.
    pub fn scope_descriptors(&self, kind: SstKind, scope: &str) -> &[usize] {
        self.buckets
            .get(&(kind, scope.to_string()))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// All node-SST descriptor indices across every scope: the id-primary
    /// partition (`scope = ""`) plus any legacy per-label scopes. Now that
    /// nodes are not partitioned by label, the read path scans the union and
    /// filters by each row's decoded label set.
    pub fn node_descriptors(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .buckets
            .iter()
            .filter(|((kind, _), _)| *kind == SstKind::Nodes)
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        out.sort_unstable();
        out
    }

    /// Node-SST candidate indices whose `(min_key, max_key)` straddles
    /// `target`, across every node scope. The scope-agnostic analogue of
    /// [`lookup_candidates`] for id-primary point lookups.
    pub fn node_candidates(&self, ssts: &[SstDescriptor], target: &[u8; 16]) -> Vec<usize> {
        let mut out = Vec::new();
        for (kind, scope) in self.buckets.keys() {
            if *kind != SstKind::Nodes {
                continue;
            }
            out.extend(self.lookup_candidates(ssts, SstKind::Nodes, scope, target));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use namidb_core::NamespaceId;
    use object_store::memory::InMemory;

    use super::*;

    fn store() -> (Arc<dyn ObjectStore>, NamespacePaths) {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = NamespacePaths::new("", NamespaceId::new("acme").unwrap());
        (store, paths)
    }

    #[tokio::test]
    async fn bootstrap_then_load() {
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let writer = Uuid::now_v7();
        let initial = ms.bootstrap(writer).await.unwrap();
        assert_eq!(initial.manifest.version, 0);
        assert_eq!(initial.manifest.epoch, Epoch::ZERO);

        let reloaded = ms.load_current().await.unwrap();
        assert_eq!(reloaded.manifest, initial.manifest);
        assert_eq!(reloaded.pointer, initial.pointer);
    }

    #[tokio::test]
    async fn bootstrap_is_idempotent_safe() {
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let w = Uuid::now_v7();
        ms.bootstrap(w).await.unwrap();
        let err = ms.bootstrap(w).await.unwrap_err();
        match err {
            Error::Precondition(_) => {}
            other => panic!("expected Precondition, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bootstrap_recovers_from_half_written_state() {
        // Simulate a bootstrap that wrote v0.json then crashed BEFORE the
        // pointer landed (the wedge case): a later bootstrap must complete the
        // pointer rather than error "v0 exists". Before the fix this namespace
        // could neither bootstrap (v0 exists) nor load (no pointer) — wedged.
        let (store, paths) = store();
        let w = Uuid::now_v7();
        let manifest = Manifest::empty(Epoch::ZERO, w);
        let v0 = paths.manifest_version(0);
        let ms = ManifestStore::new(store, paths);
        ms.put_create(&v0, serde_json::to_vec(&manifest).unwrap().into())
            .await
            .unwrap();
        // p0.json is deliberately absent (the crash).

        let loaded = ms
            .bootstrap(w)
            .await
            .expect("bootstrap must recover a half-written state");
        assert_eq!(loaded.manifest.version, 0);

        // Pointer + advisory are now in place; load_current resolves.
        let reloaded = ms.load_current().await.unwrap();
        assert_eq!(reloaded.manifest.version, 0);
    }

    #[tokio::test]
    async fn happy_path_commit_chain() {
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let w = Uuid::now_v7();
        let mut current = ms.bootstrap(w).await.unwrap();
        let fence = WriterFence::new(current.manifest.epoch);

        for expected_version in 1u64..=5 {
            let next = current.manifest.next_version(fence.writer_id);
            current = ms.commit(&fence, &current, next).await.unwrap();
            assert_eq!(current.manifest.version, expected_version);
        }

        let reloaded = ms.load_current().await.unwrap();
        assert_eq!(reloaded.manifest.version, 5);
    }

    #[tokio::test]
    async fn cas_loss_is_reported() {
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let w = Uuid::now_v7();
        let base = ms.bootstrap(w).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        // Writer A advances to v1.
        let a_next = base.manifest.next_version(fence.writer_id);
        let _committed = ms.commit(&fence, &base, a_next).await.unwrap();

        // Writer B still holds the stale base; its commit must lose CAS.
        let b_next = base.manifest.next_version(fence.writer_id);
        let err = ms.commit(&fence, &base, b_next).await.unwrap_err();
        match err {
            Error::ManifestCommitCas { expected, found } => {
                assert_eq!(expected, 0);
                assert_eq!(found, 1);
            }
            other => panic!("expected ManifestCommitCas, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn claim_writer_increments_epoch_and_fences_old_writer() {
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let w = Uuid::now_v7();
        let base = ms.bootstrap(w).await.unwrap();
        let stale_fence = WriterFence::new(base.manifest.epoch); // e0

        let (loaded, fresh_fence) = ms.claim_writer().await.unwrap();
        assert_eq!(loaded.manifest.epoch, base.manifest.epoch.next());
        assert_eq!(fresh_fence.epoch, loaded.manifest.epoch);

        // Stale writer trying to assert against the new epoch is fenced.
        let err = stale_fence.assert_alive(loaded.manifest.epoch).unwrap_err();
        match err {
            Error::Fenced { mine, current } => {
                assert_eq!(mine, base.manifest.epoch.as_u64());
                assert_eq!(current, loaded.manifest.epoch.as_u64());
            }
            other => panic!("expected Fenced, got {other:?}"),
        }
    }

    /// Plant an orphan manifest body at `version` (Create), pointer left
    /// behind — the crash window between the body PUT and the pointer CAS.
    async fn plant_orphan_body(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        orphan: &Manifest,
    ) {
        store
            .put_opts(
                &paths.manifest_version(orphan.version),
                PutPayload::from(serde_json::to_vec(orphan).unwrap()),
                PutOptions::from(PutMode::Create),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn claim_writer_adopts_orphan_body_and_unwedges_namespace() {
        // Reproduce the partial-commit window: a writer wrote the manifest
        // body at version 1 via PutMode::Create but never advanced the
        // pointer (crash between the body PUT and the pointer CAS). The
        // orphan references no new WAL segment or SST, so it is a complete
        // commit — claim_writer must publish its pointer (completing the
        // interrupted commit) and then claim on top of it, instead of
        // wedging the namespace forever.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let w = Uuid::now_v7();
        let base = ms.bootstrap(w).await.unwrap();
        assert_eq!(base.manifest.version, 0);

        let orphan = base.manifest.next_version(Uuid::now_v7());
        assert_eq!(orphan.version, 1);
        plant_orphan_body(&store, &paths, &orphan).await;

        let (loaded, fence) =
            tokio::time::timeout(std::time::Duration::from_secs(5), ms.claim_writer())
                .await
                .expect("claim_writer must not hang on an orphan manifest body")
                .expect("claim_writer must repair the stalled commit");
        // The adopted orphan became v1 and the claim landed on top at v2.
        assert_eq!(loaded.manifest.version, 2);
        assert_eq!(loaded.manifest.epoch, Epoch(1));
        assert_eq!(fence.epoch, Epoch(1));
        assert!(
            store.head(&paths.pointer_version(1)).await.is_ok(),
            "adoption must publish the orphan's pointer p1"
        );
        let adopted = ms.load_manifest_at(1).await.unwrap();
        assert_eq!(adopted, orphan, "v1 must still be the orphan body");

        // The namespace is writable again.
        let next = loaded.manifest.next_version(fence.writer_id);
        let committed = ms.commit(&fence, &loaded, next).await.unwrap();
        assert_eq!(committed.manifest.version, 3);
    }

    #[tokio::test]
    async fn claim_writer_adopts_orphan_whose_wal_segment_is_durable() {
        // The interrupted commit's WAL PUT landed (pipelined with the body
        // PUT in commit_batch) but the pointer CAS never ran. The orphan is
        // a complete commit: adopt it, keeping the WAL segment referenced so
        // no durable records are dropped.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        let mut segment = crate::wal::WalSegment::new(1);
        segment.push(crate::wal::WalRecord {
            lsn: 7,
            payload: Bytes::from_static(b"payload"),
        });
        let wal_store = crate::wal::WalStore::new(store.clone(), paths.clone());
        wal_store.append_segment(&segment).await.unwrap();

        let mut orphan = base.manifest.next_version(Uuid::now_v7());
        orphan.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 7,
        });
        plant_orphan_body(&store, &paths, &orphan).await;

        let (loaded, _fence) = ms.claim_writer().await.unwrap();
        assert_eq!(loaded.manifest.version, 2);
        assert_eq!(loaded.manifest.epoch, Epoch(1));
        assert_eq!(
            loaded.manifest.wal_segments.len(),
            1,
            "the adopted commit's WAL segment must stay referenced"
        );
        assert_eq!(loaded.manifest.wal_segments[0].seq, 1);
        assert_eq!(loaded.manifest.wal_segments[0].last_lsn, 7);
    }

    #[tokio::test]
    async fn claim_writer_deletes_orphan_with_missing_wal_segment() {
        // The interrupted commit's pipelined WAL PUT never landed: the body
        // references a segment that does not exist. Publishing it would
        // wedge recovery on the missing segment, so the repair must fence
        // the WAL slot, delete the orphan, and let the claim proceed.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        let mut orphan = base.manifest.next_version(Uuid::now_v7());
        orphan.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 42,
        });
        plant_orphan_body(&store, &paths, &orphan).await;

        let (loaded, _fence) =
            tokio::time::timeout(std::time::Duration::from_secs(5), ms.claim_writer())
                .await
                .expect("claim_writer must not hang")
                .expect("claim_writer must delete the incomplete orphan and claim");
        // The claim itself became v1 (the orphan was deleted, freeing the
        // version), and the phantom segment is NOT referenced.
        assert_eq!(loaded.manifest.version, 1);
        assert_eq!(loaded.manifest.epoch, Epoch(1));
        assert!(
            loaded.manifest.wal_segments.is_empty(),
            "the phantom WAL segment must not be published"
        );
        let body = ms.load_manifest_at(1).await.unwrap();
        assert_ne!(body, orphan, "the orphan body must have been replaced");

        // The WAL slot was fenced with a valid empty segment so the
        // interrupted commit can never complete it.
        let sentinel = store.get(&paths.wal_segment(1)).await.unwrap();
        let bytes = sentinel.bytes().await.unwrap();
        let decoded = crate::wal::WalSegment::decode(1, bytes).unwrap();
        assert!(decoded.is_empty(), "the fence sentinel must be empty");
    }

    #[tokio::test]
    async fn claim_writer_deletes_orphan_with_mismatched_wal_segment() {
        // The WAL slot holds a segment whose content does not match the
        // orphan's descriptor (a later attempt of the same seq, or a fence
        // sentinel from a crashed repairer). Recovery refuses last_lsn
        // mismatches, so the orphan must be deleted, not published.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        let mut segment = crate::wal::WalSegment::new(1);
        segment.push(crate::wal::WalRecord {
            lsn: 9,
            payload: Bytes::from_static(b"payload"),
        });
        let wal_store = crate::wal::WalStore::new(store.clone(), paths.clone());
        wal_store.append_segment(&segment).await.unwrap();

        let mut orphan = base.manifest.next_version(Uuid::now_v7());
        orphan.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 7, // declared 7, body carries 9
        });
        plant_orphan_body(&store, &paths, &orphan).await;

        let (loaded, _fence) = ms.claim_writer().await.unwrap();
        assert_eq!(loaded.manifest.version, 1);
        assert!(loaded.manifest.wal_segments.is_empty());
        // The mismatched segment itself is untouched (the janitor owns it).
        assert!(store.head(&paths.wal_segment(1)).await.is_ok());
    }

    #[tokio::test]
    async fn claim_writer_deletes_corrupt_orphan_body() {
        // A body that does not parse as a manifest was not produced by any
        // protocol writer; it can never be completed, only removed.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let _ = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        store
            .put_opts(
                &paths.manifest_version(1),
                PutPayload::from(&b"not json"[..]),
                PutOptions::from(PutMode::Create),
            )
            .await
            .unwrap();

        let (loaded, _fence) = ms.claim_writer().await.unwrap();
        assert_eq!(loaded.manifest.version, 1);
        assert_eq!(loaded.manifest.epoch, Epoch(1));
    }

    #[tokio::test]
    async fn put_body_retry_with_identical_body_adopts_existing() {
        // Retry-after-lost-response: the first put_body landed but the
        // caller never saw the success (e.g. a CAS-pointer timeout forced a
        // whole-commit retry). The second attempt writes byte-identical
        // content and must adopt the durable body instead of failing with
        // ManifestCommitCas — which would strand an orphan nobody can
        // supersede.
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let next = base.manifest.next_version(fence.writer_id);
        let first = ms.put_body(&fence, &base, &next).await.unwrap();
        // Same Manifest value → byte-identical body.
        let second = ms.put_body(&fence, &base, &next).await.unwrap();
        assert_eq!(first, second);

        let committed = ms.cas_pointer(&fence, &base, next, second).await.unwrap();
        assert_eq!(committed.manifest.version, 1);
        assert_eq!(ms.load_current().await.unwrap().manifest.version, 1);
    }

    #[tokio::test]
    async fn put_body_retry_adopts_body_differing_only_in_created_at() {
        // A rebuilt retry (commit_batch re-derives the next manifest after a
        // transient WAL PUT failure) re-stamps created_at, so the bytes are
        // not identical — but the manifest is semantically the same commit
        // from the same writer. It must still be adopted.
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let mut next = base.manifest.next_version(fence.writer_id);
        ms.put_body(&fence, &base, &next).await.unwrap();

        next.created_at += chrono::Duration::seconds(3);
        let pointer = ms.put_body(&fence, &base, &next).await.unwrap();
        let committed = ms.cas_pointer(&fence, &base, next, pointer).await.unwrap();
        assert_eq!(committed.manifest.version, 1);
    }

    #[tokio::test]
    async fn put_body_still_loses_cas_to_a_competitor_body() {
        // A different writer's body at the same version is NOT ours: the
        // adoption path must not fire and the CAS loss must surface.
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        // Competitor (different writer_id) creates v1 first.
        let competitor = base.manifest.next_version(Uuid::now_v7());
        ms.put_body(&fence, &base, &competitor).await.unwrap();

        let ours = base.manifest.next_version(fence.writer_id);
        let err = ms.put_body(&fence, &base, &ours).await.unwrap_err();
        match err {
            Error::ManifestCommitCas { expected, found } => {
                assert_eq!(expected, 0);
                assert_eq!(found, 1);
            }
            other => panic!("expected ManifestCommitCas, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_body_does_not_adopt_own_body_with_different_content() {
        // Same writer, same version, but the durable body references a
        // different WAL set (records were appended between attempts).
        // Adopting it would publish a manifest that disagrees with what the
        // caller is about to apply — must stay a CAS loss.
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let mut first = base.manifest.next_version(fence.writer_id);
        first.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 100,
        });
        ms.put_body(&fence, &base, &first).await.unwrap();

        let mut second = base.manifest.next_version(fence.writer_id);
        second.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 150,
        });
        let err = ms.put_body(&fence, &base, &second).await.unwrap_err();
        assert!(
            matches!(err, Error::ManifestCommitCas { .. }),
            "expected ManifestCommitCas, got {err:?}"
        );
    }

    #[tokio::test]
    async fn version_must_be_monotonic() {
        let (store, paths) = store();
        let ms = ManifestStore::new(store, paths);
        let w = Uuid::now_v7();
        let base = ms.bootstrap(w).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        // Try to write the same version we already have — must error out
        // *before* hitting object storage.
        let mut bad = base.manifest.clone();
        bad.version = base.manifest.version; // not + 1
        let err = ms.commit(&fence, &base, bad).await.unwrap_err();
        match err {
            Error::Invariant(msg) => assert!(msg.contains("must be")),
            other => panic!("expected Invariant, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn commit_creates_versioned_pointer_family() {
        // RFC-029: each commit publishes `pointer/p<N>.json` via Create. The
        // advisory `current.json` (same body) is ALSO published so the version
        // is findable via a non-LIST read on EC stores after the janitor
        // reclaims p0 (see `max_pointer_version` / `write_advisory_current`).
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let w = Uuid::now_v7();
        let mut current = ms.bootstrap(w).await.unwrap();
        let fence = WriterFence::new(current.manifest.epoch);
        for _ in 1u64..=3 {
            let next = current.manifest.next_version(fence.writer_id);
            current = ms.commit(&fence, &current, next).await.unwrap();
        }
        for v in 0u64..=3 {
            assert!(
                store.head(&paths.pointer_version(v)).await.is_ok(),
                "pointer p{v} must exist"
            );
        }
        assert!(
            store.head(&paths.current_pointer()).await.is_ok(),
            "the advisory current.json must be published on commit"
        );
        assert_eq!(ms.load_current().await.unwrap().manifest.version, 3);
    }

    #[tokio::test]
    async fn load_current_resolves_after_p0_reclaimed_and_stale_list() {
        // Regression for the data-loss bug: once the janitor reclaims p0
        // (below the horizon) AND a LIST returns stale-empty, the family must
        // STILL resolve via the advisory current.json — not fall through to
        // NotFound and re-bootstrap a live namespace.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let w = Uuid::now_v7();
        let mut current = ms.bootstrap(w).await.unwrap();
        let fence = WriterFence::new(current.manifest.epoch);
        for _ in 1u64..=3 {
            let next = current.manifest.next_version(fence.writer_id);
            current = ms.commit(&fence, &current, next).await.unwrap();
        }
        // Simulate the janitor reclaiming p0 (and below-horizon bodies/pointers).
        store.delete(&paths.pointer_version(0)).await.unwrap();
        // current.json advisory (written on commit) still points at version 3.
        assert!(store.head(&paths.current_pointer()).await.is_ok());
        // load_current must resolve to 3 (via advisory), never NotFound.
        assert_eq!(ms.load_current().await.unwrap().manifest.version, 3);

        // Even if we also deleted the advisory, the pointer family LIST (p1..p3)
        // still resolves it the normal way; only the joint absence (p0 gone +
        // stale empty LIST + no advisory) would fall through, and that is a
        // truly uninitialised namespace, correctly bootstrapped by open.
    }

    #[tokio::test]
    async fn load_current_falls_back_to_legacy_current_json() {
        // A namespace bootstrapped before RFC-029 has manifest bodies and a
        // single `current.json` but no pointer family; load_current must still
        // resolve it through the legacy fallback.
        let (store, paths) = store();
        let manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        store
            .put(
                &paths.manifest_version(0),
                PutPayload::from(serde_json::to_vec(&manifest).unwrap()),
            )
            .await
            .unwrap();
        let pointer = ManifestPointer {
            version: 0,
            epoch: Epoch::ZERO,
            manifest_path: paths.manifest_version(0).as_ref().to_string(),
        };
        store
            .put(
                &paths.current_pointer(),
                PutPayload::from(serde_json::to_vec(&pointer).unwrap()),
            )
            .await
            .unwrap();

        let ms = ManifestStore::new(store, paths);
        let loaded = ms.load_current().await.unwrap();
        assert_eq!(loaded.manifest.version, 0);
    }

    #[tokio::test]
    async fn cas_pointer_create_conflict_is_cas_loss() {
        // Exercises cas_pointer's AlreadyExists branch directly: PUT the body,
        // let a competitor publish the pointer first, then our create loses.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let w = Uuid::now_v7();
        let base = ms.bootstrap(w).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let next = base.manifest.next_version(fence.writer_id);
        let pointer = ms.put_body(&fence, &base, &next).await.unwrap();

        // Competitor publishes p1 first.
        store
            .put_opts(
                &paths.pointer_version(1),
                PutPayload::from(serde_json::to_vec(&pointer).unwrap()),
                PutOptions::from(PutMode::Create),
            )
            .await
            .unwrap();

        let err = ms
            .cas_pointer(&fence, &base, next, pointer)
            .await
            .unwrap_err();
        match err {
            Error::ManifestCommitCas { expected, found } => {
                assert_eq!(expected, 0);
                assert_eq!(found, 1);
            }
            other => panic!("expected ManifestCommitCas, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn forward_probe_advances_past_stale_list_max() -> Result<()> {
        // Simulates a stale LIST on an eventually-consistent store: create
        // pointers p0-p5, then add p6-p10. A stale LIST that only saw p0-p5 would
        // return max=5, but the forward probe should discover p6-p10 via HEAD.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let w = Uuid::now_v7();
        let mut current = ms.bootstrap(w).await?;
        let fence = WriterFence::new(current.manifest.epoch);

        // Create p0-p5 (bootstrap gives us p0, advance to p5)
        for _ in 0..5 {
            let next = current.manifest.next_version(fence.writer_id);
            current = ms.commit(&fence, &current, next).await?;
        }
        assert_eq!(current.manifest.version, 5);

        // Simulate a stale LIST by manually constructing a max that only sees p5:
        // the forward probe should discover p6-p10 via HEAD and land on p10.
        let max_from_stale_list = 5u64;
        let probed = ms.probe_pointer_forward(max_from_stale_list).await?;
        assert_eq!(
            probed, 5,
            "forward probe from stale max should stay at 5 until more commits"
        );

        // Now create p6-p10; forward probe from 5 should discover them.
        for _ in 0..5 {
            let next = current.manifest.next_version(fence.writer_id);
            current = ms.commit(&fence, &current, next).await?;
        }
        assert_eq!(current.manifest.version, 10);

        // Forward probe from stale max=5 should now discover up to p10.
        let probed = ms.probe_pointer_forward(5).await?;
        assert_eq!(
            probed, 10,
            "forward probe should advance from stale max to actual current"
        );

        Ok(())
    }

    fn sample_node_descriptor() -> SstDescriptor {
        SstDescriptor {
            id: Uuid::now_v7(),
            kind: SstKind::Nodes,
            scope: "Person".into(),
            level: SstLevel::L0,
            path: "sst/level0/0195-nodes-Person.parquet".into(),
            size_bytes: 4 * 1024 * 1024,
            row_count: 12_345,
            created_at: Utc::now(),
            min_key: [0x01u8; 16],
            max_key: [0xFEu8; 16],
            min_lsn: 100,
            max_lsn: 150,
            schema_version_min: 3,
            schema_version_max: 3,
            property_stats: vec![PropertyColumnStats {
                name: "prop_age".into(),
                null_count: 2,
                min: Some(crate::sst::stats::StatScalar::Int32(18)),
                max: Some(crate::sst::stats::StatScalar::Int32(90)),
                ndv_estimate: None,
            }],
            kind_specific: KindSpecificStats::Nodes { tombstone_count: 4 },
            bloom: Some(BloomDescriptor {
                path: "sst/level0/0195-nodes-Person.bloom".into(),
                size_bytes: 250_036,
                key_count: 12_345,
                bits_per_key: 10,
                block_count: 482,
                xxhash3_64: 0xDEADBEEFCAFEBABE,
            }),
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            per_label_property_stats: Vec::new(),
        }
    }

    fn sample_edge_descriptor() -> SstDescriptor {
        let mut h = DegreeHistogram::empty();
        for d in [1u64, 2, 4, 1024] {
            h.observe(d);
        }
        SstDescriptor {
            id: Uuid::now_v7(),
            kind: SstKind::EdgesFwd,
            scope: "KNOWS".into(),
            level: SstLevel::L0,
            path: "sst/level0/0195-edges-fwd-KNOWS.csr".into(),
            size_bytes: 2 * 1024 * 1024,
            row_count: 50_000,
            created_at: Utc::now(),
            min_key: [0; 16],
            max_key: [0xFF; 16],
            min_lsn: 1,
            max_lsn: 999,
            schema_version_min: 1,
            schema_version_max: 2,
            property_stats: vec![],
            kind_specific: KindSpecificStats::Edges {
                key_count: 4,
                tombstone_count: 0,
                degree_histogram: Box::new(h),
            },
            bloom: None, // small SST → no side-car
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            per_label_property_stats: Vec::new(),
        }
    }

    #[test]
    fn sst_descriptor_round_trips_through_json_nodes() {
        let d = sample_node_descriptor();
        let s = serde_json::to_string(&d).unwrap();
        let back: SstDescriptor = serde_json::from_str(&s).unwrap();
        assert_eq!(d, back);
        // JSON encodes [u8; 16] as base64 string of length 24.
        assert!(s.contains("\"min_key\":\""));
        assert!(s.contains("\"max_key\":\""));
    }

    #[test]
    fn sst_descriptor_round_trips_through_json_edges_with_no_bloom() {
        let d = sample_edge_descriptor();
        let s = serde_json::to_string(&d).unwrap();
        let back: SstDescriptor = serde_json::from_str(&s).unwrap();
        assert_eq!(d, back);
        // bloom = None serialises as JSON null.
        assert!(s.contains("\"bloom\":null"));
        // kind_specific is internally-tagged via `tag = "kind"`.
        assert!(s.contains("\"kind_specific\":{\"kind\":\"Edges\""));
    }

    #[test]
    fn sst_descriptor_rejects_wrong_key_length_in_json() {
        let mut s = serde_json::to_string(&sample_node_descriptor()).unwrap();
        // Tamper with min_key: replace the 24-char base64 with a too-short one.
        let needle = "\"min_key\":\"";
        let pos = s.find(needle).unwrap() + needle.len();
        let end = s[pos..].find('"').unwrap() + pos;
        s.replace_range(pos..end, "AAAA"); // 4 chars → decodes to 3 bytes
        let err = serde_json::from_str::<SstDescriptor>(&s).unwrap_err();
        assert!(err.to_string().contains("expected 16 bytes"));
    }

    #[test]
    fn manifest_with_sst_round_trips() {
        // A full Manifest carrying one node SST descriptor must round-trip
        // through serde_json (the on-disk format we PUT to object storage).
        let mut m = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        m.ssts.push(sample_node_descriptor());
        let bytes = serde_json::to_vec(&m).unwrap();
        let back: Manifest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn manifest_without_multilabel_fields_loads_with_defaults() {
        // Back-compat contract of the inert prep step: a manifest written
        // before multi-label has no top-level `label_dict`, and its SST
        // descriptors have no `label_index`. Both must default cleanly (empty
        // dict / None) via `serde(default)` so existing namespaces keep
        // loading unchanged.
        let mut m = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        m.ssts.push(sample_node_descriptor());
        let mut value = serde_json::to_value(&m).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.remove("label_dict");
        for sst in obj["ssts"].as_array_mut().unwrap() {
            sst.as_object_mut().unwrap().remove("label_index");
        }
        let back: Manifest = serde_json::from_value(value).unwrap();
        assert!(
            back.label_dict.is_empty(),
            "missing label_dict must default empty"
        );
        assert!(
            back.ssts[0].label_index.is_none(),
            "missing label_index must default to None"
        );
    }

    #[test]
    fn sst_kind_path_tag_matches_rfc() {
        assert_eq!(SstKind::Nodes.path_tag(), "nodes");
        assert_eq!(SstKind::EdgesFwd.path_tag(), "edges-fwd");
        assert_eq!(SstKind::EdgesInv.path_tag(), "edges-inv");
    }
}
