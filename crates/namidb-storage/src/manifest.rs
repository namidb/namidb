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
//! `current` is garbage. A future janitor will delete orphan manifests.

use std::collections::HashMap;
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
/// and inverse partner CSRs). `Vector` lands with RFC-007.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SstKind {
    /// Property-column SST for a node label (Parquet).
    Nodes,
    /// CSR adjacency SST for an edge type, sorted by `src_id`.
    EdgesFwd,
    /// CSR adjacency SST for an edge type, sorted by `dst_id` (inverse partner).
    EdgesInv,
}

impl SstKind {
    /// Path tag used in the SST filename (RFC-002 §1).
    pub fn path_tag(self) -> &'static str {
        match self {
            SstKind::Nodes => "nodes",
            SstKind::EdgesFwd => "edges-fwd",
            SstKind::EdgesInv => "edges-inv",
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
        self.put_create(&manifest_path, serde_json::to_vec(&manifest)?.into())
            .await
            .map_err(|e| match e {
                Error::ObjectStore(object_store::Error::AlreadyExists { .. }) => {
                    Error::precondition(format!(
                        "namespace '{}' already bootstrapped: {} exists",
                        self.paths.namespace(),
                        manifest_path
                    ))
                }
                other => other,
            })?;

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
            if self.store.head(&self.paths.pointer_version(0)).await.is_ok() {
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
        let mut probed = 0u32;
        while probed < MAX_PROBE {
            let next = n.saturating_add(1);
            match self.store.head(&self.paths.pointer_version(next)).await {
                Ok(_) => n = next,
                Err(object_store::Error::NotFound { .. }) => break,
                Err(e) => return Err(Error::ObjectStore(e)),
            }
            probed += 1;
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
        debug!(path = %manifest_path, "writing immutable manifest body");
        match self
            .put_create(&manifest_path, serde_json::to_vec(new_manifest)?.into())
            .await
        {
            Ok(_) => {}
            Err(Error::ObjectStore(object_store::Error::AlreadyExists { .. })) => {
                // Another writer chose the same version. Before raising
                // a plain CAS loss, reload to discover whether the
                // namespace has actually advanced past our epoch — in
                // that case we are fenced and the caller must drop
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
            Err(other) => return Err(other),
        }

        Ok(ManifestPointer {
            version: new_manifest.version,
            epoch: new_manifest.epoch,
            manifest_path: manifest_path.as_ref().to_string(),
        })
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
        // transient error in `cas_pointer`). Nobody can supersede that
        // version under `Create`, so an unbounded loop would spin forever.
        // Bound the *stall* (consecutive CAS losses at the same pointer
        // version) and surface a distinct terminal error instead of hanging.
        const MAX_STALLED_ROUNDS: usize = 8;
        let mut stalled_rounds = 0usize;
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
                    // Reload and retry only while we keep making progress
                    // (the pointer version advances). If it stalls at the
                    // same version, we are colliding with an orphan body
                    // and must stop rather than loop forever.
                    if last_version == Some(base.pointer.version) {
                        stalled_rounds += 1;
                        if stalled_rounds >= MAX_STALLED_ROUNDS {
                            return Err(Error::OrphanManifestBody {
                                version: base.pointer.version.saturating_add(1),
                            });
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

    #[tokio::test]
    async fn claim_writer_surfaces_orphan_manifest_body_instead_of_hanging() {
        // Reproduce the partial-commit window: a writer wrote the manifest
        // body at version 1 via PutMode::Create but never advanced the
        // pointer (e.g. a transient, non-Precondition error in cas_pointer).
        // The body at v1 is now a durable orphan with the pointer stuck at
        // v0. Without a stall bound, claim_writer would spin forever (Create
        // at v1 -> AlreadyExists -> ManifestCommitCas -> reload still v0 ->
        // repeat). It must instead terminate with a distinct error.
        let (store, paths) = store();
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let w = Uuid::now_v7();
        let base = ms.bootstrap(w).await.unwrap();
        assert_eq!(base.manifest.version, 0);

        // Plant the orphan body at v1, pointer left at v0.
        let orphan = base.manifest.next_version(Uuid::now_v7());
        assert_eq!(orphan.version, 1);
        store
            .put_opts(
                &paths.manifest_version(1),
                PutPayload::from(serde_json::to_vec(&orphan).unwrap()),
                PutOptions::from(PutMode::Create),
            )
            .await
            .unwrap();

        // Must terminate (not hang) and surface OrphanManifestBody.
        let err = tokio::time::timeout(std::time::Duration::from_secs(5), ms.claim_writer())
            .await
            .expect("claim_writer must not hang on an orphan manifest body")
            .unwrap_err();
        match err {
            Error::OrphanManifestBody { version } => assert_eq!(version, 1),
            other => panic!("expected OrphanManifestBody, got {other:?}"),
        }
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
        let max_from_stale_list = Some(5u64);
        let probed = ms.probe_pointer_forward(max_from_stale_list.unwrap()).await?;
        assert_eq!(probed, 5, "forward probe from stale max should stay at 5 until more commits");

        // Now create p6-p10; forward probe from 5 should discover them.
        for _ in 0..5 {
            let next = current.manifest.next_version(fence.writer_id);
            current = ms.commit(&fence, &current, next).await?;
        }
        assert_eq!(current.manifest.version, 10);

        // Forward probe from stale max=5 should now discover up to p10.
        let probed = ms.probe_pointer_forward(5).await?;
        assert_eq!(probed, 10, "forward probe should advance from stale max to actual current");

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
