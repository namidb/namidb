//! Leveled-lite compaction.
//!
//! Each [`crate::flush::flush`] call appends a new L0 SST per `(kind,
//! scope)` bucket that had memtable rows. Without compaction, a
//! namespace's L0 footprint grows monotonically with every batch and
//! every point lookup pays an `O(L0 count)` candidate-SST scan.
//!
//! The compactor keeps one SST per `(kind, scope, level)` across L1..Lk with
//! a per-level byte budget (`budget(Li) = base * ratio^(i-1)`). L0s drain
//! into L1; a merge cascades into the next deeper level only when the
//! accumulated bytes exceed a level's budget, so the large base levels are
//! rewritten rarely (bounded write amplification) while space and read
//! amplification stay bounded. Each sweep commits a manifest version that
//! removes the merged source descriptors and adds the new one in a single
//! CAS. The source SST bodies become orphans in object storage (no current
//! manifest version references them after the commit); the horizon-aware
//! [`crate::janitor::sweep_orphans`] reclaims them once no pinned reader
//! needs them.
//!
//! ## Merge semantics
//!
//! Per `(node_id)` (nodes) or `(key_id, partner_id)` (edges): the row with
//! the highest LSN wins; lower-LSN versions are dropped. A winning
//! **tombstone** is dropped entirely (RFC-027 P3) only when the merge output
//! is the bucket's deepest occupied level: the LSM invariant (a shallower
//! level holds the newer LSN for a key) means no un-merged deeper level can
//! hold a live row the tombstone was shadowing, so dropping it can never
//! resurrect a row. A reader pinned at an older manifest version still
//! observes the delete through the retained source bodies, never through the
//! new SST.
//!
//! ## What's deliberately not here
//!
//! - Range-partitioned leveled compaction (multiple non-overlapping SSTs per
//! level, rewriting only the overlapping key ranges). leveled-lite keeps one
//! SST per `(bucket, level)`, so a cascade rewrites the whole next level, not
//! just the overlapping range. That refinement is the remaining RFC-027 P4
//! step.
//! - Background scheduling beyond the periodic maintenance tick and the
//! reactive L0-count trigger / write stall (RFC-027 P5).
//!
//! Declared edge property streams (RFC-002 §3.2.7) are preserved
//! end-to-end: the compactor reads each declared stream from every
//! source SST, joins it with the per-edge enumeration, and re-emits
//! the merged stream into the new SST body alongside `__overflow_json`.
//!
//! ## Prepare / commit split
//!
//! A sweep is two phases so a host can keep its writer lock out of the
//! expensive part: [`prepare_compaction`] does the planning, every input
//! GET, the CPU merges and index rebuilds, and every output PUT (all at
//! immutable UUID paths no manifest references yet); [`install_prepared`]
//! folds the result into the manifest **current at commit time** and runs
//! the fence-checked CAS. [`crate::ingest::WriterSession::compaction_basis`]
//! snapshots the inputs under the lock so the prepare can run off it.
//!
//! ## Streaming merge
//!
//! The prepare phase merges each bucket with a k-way streaming merge
//! instead of materialising every decoded source row. Per-source cursors
//! decode one bounded record batch (nodes) / one partner block +
//! property-stream mini-batch (edges) at a time; a binary heap keyed by
//! `(key asc, lsn desc, source order)` picks the winner per key, shadowed
//! duplicates are skipped without ever being converted, and only winners
//! pay the row materialisation (for nodes, the JSON property-map
//! re-encode — typically 3-10x the Parquet size) on their way into a
//! bounded chunk buffer feeding the incremental SST writer. The
//! sidecar/stat harvesters and the vector/text index member collectors
//! observe the same winner stream, so nothing retains the merged bucket.
//!
//! Residual memory per bucket, by design: file-backed mappings of the
//! compressed source bodies, one small decoded batch per activated node
//! source, one chunk of winner rows, the sidecar maps, and bounded
//! vector/text index-build buffers. Search corpora and their final immutable
//! objects are spooled to local disk and uploaded as fixed multipart windows;
//! neither embeddings, documents, postings, nor the finished `.vg`/`.ft`
//! body are retained in full.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, FixedSizeBinaryArray, ListArray, RecordBatch, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_ipc::reader::StreamReader;
use bytes::Bytes;
use chrono::Utc;
use futures::StreamExt;
use memmap2::MmapOptions;
use object_store::path::Path;
use object_store::{GetResultPayload, ObjectStore, ObjectStoreExt};
use parquet::arrow::arrow_reader::{
    ArrowReaderOptions, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder,
};
use tokio::io::AsyncWriteExt;
use tracing::{debug, instrument};
use uuid::Uuid;
use xxhash_rust::xxh3::Xxh3;

use namidb_core::{DataType, EdgeTypeDef, LabelDef, LabelDictionary, Schema, Value};

use crate::error::{Error, Result};
use crate::fence::WriterFence;
use crate::flush::{
    encode_exact_node_record, EqualitySidecarCollector, IncrementalNodeSstWriter,
    LabelIndexCollector, NodeRow, NodeWriteRecord, PerLabelStatsCollector, SidecarPayload,
    UniqueSidecarCollector, NODE_SST_BATCH_ROWS,
};
#[cfg(feature = "vector-index")]
use crate::manifest::VectorIndexDescriptor;
use crate::manifest::{
    KindSpecificStats, LoadedManifest, ManifestStore, PerLabelPropertyStat, SstDescriptor, SstKind,
    SstLevel,
};
use crate::memtable::MemOp;
use crate::paths::NamespacePaths;
use crate::read::arrow_value_to_value;
use crate::search_lsm::{
    encode_search_barrier, search_barrier_descriptor, validate_search_lsm, CoverageDisposition,
    SearchCoverage, SearchEventRange, SearchLsmState, SearchLsmStatus,
};
use crate::sst::bloom::{BloomDescriptor, BloomFilter};
use crate::sst::edges::encoding::{read_offset, read_partner_block, OffsetWidth};
use crate::sst::edges::format::{
    CODEC_NONE, CODEC_PROPERTY_PAGED_NONE, CODEC_PROPERTY_PAGED_ZSTD, CODEC_ZSTD,
    OVERFLOW_JSON_NAME, SECTION_KEY_IDS, SECTION_OFFSETS, SECTION_PARTNERS, SECTION_PER_EDGE_LSN,
    SECTION_PER_EDGE_TOMBSTONES, SECTION_PROPERTY_STREAM,
};
use crate::sst::edges::property_pages::{
    decode_property_page, PropertyPageEntry, PropertyPageIndex,
};
use crate::sst::edges::reader::EdgeSstReader;
use crate::sst::edges::writer::{EdgeRecord, EdgeSstBuild, EdgeSstWriter, EdgeSstWriterOptions};
use crate::sst::edges::EdgeDirection;
use crate::sst::nodes::{
    node_arrow_schema, prop_column_name, NodeSstFinish, NodeSstWriterOptions, COL_LABELS, COL_LSN,
    COL_NODE_ID, COL_TOMBSTONE, OVERFLOW_JSON, SCHEMA_VERSION,
};
#[cfg(test)]
use crate::sst::nodes::{parse_node_sst_metadata, NodeSstReader};

#[path = "search_lsm_compact.rs"]
mod search_lsm_compact;
use search_lsm_compact::PreparedSearchCompaction;

/// Outcome of [`compact_l0_to_l1`].
#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    pub committed: LoadedManifest,
    pub source_ssts_removed: usize,
    pub new_ssts_written: usize,
    pub bloom_sidecars_written: usize,
}

/// The expensive half of a compaction sweep, produced by
/// [`prepare_compaction`] (off the writer lock via
/// [`CompactionBasis::prepare`]). Planning, every input GET, the row
/// merges, the vector/text index rebuilds, and every output PUT have
/// already happened: the merged bodies, blooms, and sidecars sit at
/// immutable UUID-derived paths that no manifest version references yet.
/// Only the manifest CAS ([`install_prepared`]) remains. An abandoned
/// prepare therefore leaks nothing durable — its objects are unreferenced
/// garbage the janitor's [`crate::janitor::sweep_orphans`] reclaims once
/// past `min_age`.
#[derive(Debug, Clone)]
pub struct PreparedCompaction {
    /// Descriptors of the merged SSTs whose bodies are already durable.
    new_descs: Vec<SstDescriptor>,
    /// Ids of the merged source descriptors to drop from the manifest.
    removed_ids: Vec<Uuid>,
    /// Bloom sidecars written alongside the new bodies.
    bloom_count: usize,
    /// Manifest version the plan was computed against. The commit CAS runs
    /// against the manifest current at install time, which may be newer.
    base_version: u64,
    /// Schema from the manifest the prepare was planned against. Data commits
    /// and flushes may advance the manifest while prepare runs, but a DDL
    /// change can alter the columns/sidecars the prepared node SST must carry.
    base_schema: Schema,
    /// Search-index catalogs used to decide which immutable index bodies to
    /// rebuild or retain. Installing across CREATE/DROP INDEX would otherwise
    /// publish outputs computed for a different catalog.
    base_vector_indexes: Vec<crate::manifest::VectorIndexDescriptor>,
    base_text_indexes: Vec<crate::manifest::TextIndexDescriptor>,
    /// Captured Search-LSM state is a proof prefix, not an install image.
    /// Ordinary flushes may append to it while this prepare runs.
    base_search_lsm: Vec<crate::search_lsm::SearchLsmState>,
    /// Exact Nodes descriptor replacements produced off-lock. Search-LSM
    /// coverage is rebased from these inputs onto the current manifest during
    /// install; no search corpus object is rewritten here.
    node_rewrites: Vec<PreparedNodeRewrite>,
    /// Full Search-LSM bases built and uploaded off-lock. Install replaces
    /// only their captured physical prefix and preserves append-only flushes.
    search_compactions: Vec<PreparedSearchCompaction>,
    search_build_states: Vec<crate::manifest::SearchIndexBuildState>,
    /// 2.0.6-interop markers certifying each full base a BasePrefix
    /// Search-LSM consolidation installs. A downgraded writer drops the
    /// unknown `search_lsm` state but keeps these; on upgrade they are what
    /// lets adoption re-bind the surviving base metadata-only instead of
    /// rebuilding the corpus. Kept separate from `search_build_states`, which
    /// also drives `replaced_search_lsm` and the rewrite/replace install
    /// guard — reusing it would wrongly clear the active generation.
    consolidated_base_markers: Vec<crate::manifest::SearchIndexBuildState>,
    /// Indexes whose interop marker certified an adoptable base but whose
    /// physical body deterministically disproved it (unsupported magic or an
    /// unsafe wrap). Install drops these markers so the next pass plans the
    /// full rebuild instead of stalling on an adoption that can never land.
    unadoptable_search_markers: Vec<(SstKind, String)>,
    search_lsm_activations: Vec<PreparedSearchLsmActivation>,
    replaced_search_lsm: Vec<(crate::search_lsm::SearchLsmKind, String)>,
}

#[derive(Debug, Clone, PartialEq)]
struct PreparedNodeRewrite {
    inputs: Vec<SstDescriptor>,
    /// `None` is legal only for an authoritative merge whose winner stream is
    /// empty (normally tombstone GC).
    output: Option<SstDescriptor>,
}

#[derive(Debug, Clone)]
struct PreparedSearchLsmActivation {
    state: crate::search_lsm::SearchLsmState,
    barrier: SstDescriptor,
    barrier_already_present: bool,
}

impl PreparedCompaction {
    /// `true` when the sweep found nothing to merge; installing is a no-op.
    pub fn is_noop(&self) -> bool {
        self.removed_ids.is_empty()
            && self.search_lsm_activations.is_empty()
            && self.unadoptable_search_markers.is_empty()
    }

    /// Manifest version the prepare ran against.
    pub fn base_version(&self) -> u64 {
        self.base_version
    }
}

/// Snapshot of everything the expensive compaction prepare phase needs,
/// cloned out of a [`crate::ingest::WriterSession`] under the writer lock
/// (see [`crate::ingest::WriterSession::compaction_basis`]) so
/// [`Self::prepare`] can then run WITHOUT the lock while writes proceed.
#[derive(Debug, Clone)]
pub struct CompactionBasis {
    pub(crate) manifest_store: ManifestStore,
    pub(crate) fence: WriterFence,
    pub(crate) base: LoadedManifest,
}

impl CompactionBasis {
    /// Manifest version this basis was captured at.
    pub fn manifest_version(&self) -> u64 {
        self.base.manifest.version
    }

    /// Schema committed in the basis manifest — what the maintenance loops
    /// hand to [`Self::prepare`].
    pub fn schema(&self) -> &Schema {
        &self.base.manifest.schema
    }

    /// Worst per-bucket L0 backlog captured by this basis.
    ///
    /// This is the read-amplification number the server publishes around a
    /// background pass. It is sampled from the immutable basis rather than a
    /// live writer so the "before" value describes exactly the inputs the
    /// prepare phase planned against.
    pub fn max_l0_bucket_len(&self) -> usize {
        let mut counts: std::collections::HashMap<(SstKind, &str), usize> =
            std::collections::HashMap::new();
        for sst in &self.base.manifest.ssts {
            if sst.level == SstLevel::L0 {
                *counts.entry((sst.kind, sst.scope.as_str())).or_insert(0) += 1;
            }
        }
        counts.values().copied().max().unwrap_or(0)
    }

    /// Cheap, metadata-only "would a sweep merge anything?" predicate, so a
    /// maintenance tick can skip [`Self::prepare`] entirely on an idle
    /// namespace. NOTE: this is not `max_l0_bucket_len() >= 2` — a single
    /// L0 above an existing L1 (or a leveled-only over-budget cascade)
    /// still plans a merge.
    pub fn needs_compaction(&self) -> bool {
        any_bucket_plans(
            &self.base.manifest,
            compaction_base_bytes(),
            compaction_level_ratio(),
        )
    }

    /// Run the expensive prepare phase (input GETs, merges, index rebuilds,
    /// output PUTs) against this basis. Holds no lock; see
    /// [`prepare_compaction`].
    pub async fn prepare(&self, schema: &Schema) -> Result<PreparedCompaction> {
        prepare_compaction(&self.manifest_store, &self.fence, &self.base, schema).await
    }
}

/// `true` when any `(kind, scope)` bucket of `ssts` would plan a merge under
/// the given budgets — a metadata-only mirror of the per-bucket
/// [`plan_bucket_merge`] calls the prepare phase makes, with no object-store
/// I/O. Vector/text index SSTs are rebuilt from node buckets rather than
/// planned directly, so only node and edge buckets participate.
fn any_bucket_plans(manifest: &crate::manifest::Manifest, base_bytes: u64, ratio: u64) -> bool {
    if search_lsm_adoption_needed(manifest) {
        return true;
    }
    if search_lsm_compact::search_compaction_needed(
        manifest,
        search_lsm_compact::SearchCompactionPolicy::for_scheduling(),
    ) {
        return true;
    }
    let mut buckets: std::collections::HashMap<(SstKind, &str), Vec<&SstDescriptor>> =
        std::collections::HashMap::new();
    for d in &manifest.ssts {
        if matches!(
            d.kind,
            SstKind::Nodes | SstKind::EdgesFwd | SstKind::EdgesInv
        ) {
            buckets
                .entry((d.kind, d.scope.as_str()))
                .or_default()
                .push(d);
        }
    }
    let single_node_scope = buckets
        .keys()
        .filter(|(kind, _)| *kind == SstKind::Nodes)
        .count()
        <= 1;
    let rebuild_search = single_node_scope && search_indexes_need_rebuild(manifest);
    buckets.iter().any(|((kind, scope), sources)| {
        if *kind == SstKind::Nodes {
            let required = if scope.is_empty() {
                crate::flush::union_indexed_props(&manifest.schema)
            } else {
                manifest
                    .schema
                    .label(scope)
                    .cloned()
                    .unwrap_or_else(|| LabelDef {
                        name: (*scope).to_string(),
                        properties: Vec::new(),
                    })
            };
            plan_node_bucket(sources, base_bytes, ratio, &required, rebuild_search).is_some()
        } else {
            plan_bucket_merge(sources, base_bytes, ratio).is_some()
        }
    })
}

/// Run a pure-CPU compaction section (row merges, index construction) on the
/// blocking pool so it does not stall the async runtime — under the off-lock
/// prepare the surrounding task shares its runtime with live queries and
/// writes. The closure owns its inputs; a panic surfaces as an invariant
/// error instead of unwinding the caller.
async fn run_cpu<T, F>(f: F) -> Result<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Error::invariant(format!("compaction CPU task panicked: {e}")))
}

// ── Leveled-lite level budgets ──────────────────────────────────────────
//
// One SST per `(kind, scope, level)`. L0s drain into L1; a merge cascades
// into a deeper level only when the accumulated bytes exceed that level's
// budget, so the large base levels are rewritten rarely. Read from the
// environment so an operator can tune them without a rebuild.

/// `L1` byte budget when `NAMIDB_COMPACTION_BASE_BYTES` is unset.
const DEFAULT_COMPACTION_BASE_BYTES: u64 = 8 * 1024 * 1024;
/// Per-level size ratio when `NAMIDB_COMPACTION_LEVEL_RATIO` is unset.
const DEFAULT_COMPACTION_LEVEL_RATIO: u64 = 10;

/// `L1` byte budget. Deeper levels are `base * ratio^(level-1)`.
fn compaction_base_bytes() -> u64 {
    std::env::var("NAMIDB_COMPACTION_BASE_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|b| *b > 0)
        .unwrap_or(DEFAULT_COMPACTION_BASE_BYTES)
}

/// Per-level size ratio. A higher ratio means fewer, larger levels.
fn compaction_level_ratio() -> u64 {
    std::env::var("NAMIDB_COMPACTION_LEVEL_RATIO")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|r| *r >= 2)
        .unwrap_or(DEFAULT_COMPACTION_LEVEL_RATIO)
}

/// Byte budget for `level` (>= 1): `base * ratio^(level-1)`, saturating.
fn level_budget_bytes(level: u32, base: u64, ratio: u64) -> u64 {
    let mut budget = base;
    for _ in 1..level {
        budget = budget.saturating_mul(ratio);
    }
    budget
}

/// The leveled-lite merge chosen for one `(kind, scope)` bucket.
struct BucketPlan<'a> {
    /// SSTs to read and merge: the L0s plus the levels the cascade reaches.
    inputs: Vec<&'a SstDescriptor>,
    /// Level the merged SST is written at.
    target_level: u32,
    /// Whether `target_level` is the bucket's deepest occupied level, so no
    /// un-merged deeper level can hold a row a tombstone is shadowing and
    /// tombstone / superseded-version GC is safe.
    is_deepest: bool,
}

/// Decide the leveled-lite merge for one bucket, or `None` when fewer than
/// two SSTs would be merged (nothing worth rewriting).
///
/// L0s always drain into L1. When the accumulated bytes would exceed a
/// level's budget the merge cascades into the next deeper occupied level, so
/// the shallow levels stay small and the large base levels are rewritten only
/// when a shallower level overflows into them. A brand-new bucket lands its
/// first SST in L1 and cascades later, once a shallow level actually
/// overflows.
fn plan_bucket_merge<'a>(
    sources: &[&'a SstDescriptor],
    base: u64,
    ratio: u64,
) -> Option<BucketPlan<'a>> {
    let mut l0: Vec<&SstDescriptor> = Vec::new();
    let mut leveled: BTreeMap<u32, Vec<&SstDescriptor>> = BTreeMap::new();
    for d in sources {
        let lvl = d.level.as_u32();
        if lvl == 0 {
            l0.push(*d);
        } else {
            leveled.entry(lvl).or_default().push(*d);
        }
    }
    let deepest_present = leveled.keys().copied().max().unwrap_or(0);

    let mut inputs: Vec<&SstDescriptor> = l0.clone();
    let mut cum: u64 = l0.iter().map(|d| d.size_bytes).sum();
    let mut target: u32 = 1;
    loop {
        if let Some(ds) = leveled.get(&target) {
            for d in ds {
                inputs.push(*d);
                cum += d.size_bytes;
            }
        }
        if cum <= level_budget_bytes(target, base, ratio) {
            break;
        }
        if target < deepest_present {
            // Cascade into the next deeper occupied level.
            target += 1;
            continue;
        }
        // At (or past) the deepest occupied level and still over budget. Spill
        // into one fresh deeper level, but only when there is leveled data to
        // push past; a new bucket's first SST lands in L1 even if it exceeds
        // the budget and cascades on a later sweep.
        if deepest_present >= 1 {
            target += 1;
        }
        break;
    }

    if inputs.len() < 2 {
        return None;
    }
    let is_deepest = target >= deepest_present;
    Some(BucketPlan {
        inputs,
        target_level: target,
        is_deepest,
    })
}

/// Node-specific planner wrapper that performs a one-time online format
/// migration. A namespace created by 2.0.4 can consist of one fully compacted
/// L1 SST, so ordinary leveled policy would never rewrite it. If any source
/// lacks the exact row locator (or one of its property indexes lacks the
/// range-readable mirror), merge the complete bucket at its deepest level.
/// The output carries every current sidecar and the predicate becomes false,
/// preventing rewrite churn on later maintenance ticks.
fn plan_node_bucket<'a>(
    sources: &[&'a SstDescriptor],
    base: u64,
    ratio: u64,
    required: &LabelDef,
    force_search_rebuild: bool,
) -> Option<BucketPlan<'a>> {
    if let Some(plan) = plan_bucket_merge(sources, base, ratio) {
        return Some(plan);
    }
    let needs_migration = force_search_rebuild
        || sources
            .iter()
            .any(|desc| node_descriptor_needs_migration(desc, required));
    if !needs_migration || sources.is_empty() {
        return None;
    }
    let target_level = sources
        .iter()
        .map(|desc| desc.level.as_u32())
        .max()
        .unwrap_or(1)
        .max(1);
    Some(BucketPlan {
        inputs: sources.to_vec(),
        target_level,
        is_deepest: true,
    })
}

fn node_descriptor_needs_migration(desc: &SstDescriptor, required: &LabelDef) -> bool {
    !crate::manifest::node_locator_has_exact_records(desc)
        || !node_descriptor_has_property_pages(desc)
        || node_descriptor_needs_non_record_migration(desc, required)
}

fn node_descriptor_has_property_pages(desc: &SstDescriptor) -> bool {
    crate::manifest::node_property_pages_sidecar(desc).is_some_and(|properties| {
        properties.format_version
            == crate::sst::nodes::property_pages::NODE_PROPERTY_PAGES_FORMAT_VERSION
            && properties.is_bound_to(desc)
    })
}

fn node_descriptor_needs_non_record_migration(desc: &SstDescriptor, required: &LabelDef) -> bool {
    let required_equality: Vec<&str> = required
        .properties
        .iter()
        .filter(|property| {
            property.indexed
                && matches!(
                    property.data_type,
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Bool
                )
        })
        .map(|property| property.name.as_str())
        .collect();
    desc.unique_property_indices.iter().any(|index| {
        index.format != crate::manifest::PropertyIndexFormat::PagedV1
            && index.paged.is_none()
            && !index.paged_build_unsupported
    }) || desc.equality_property_indices.iter().any(|index| {
        index.format != crate::manifest::PropertyIndexFormat::PagedV1
            && index.paged.is_none()
            && !index.paged_build_unsupported
    }) || required_equality.iter().any(|property| {
        !desc.equality_property_indices.iter().any(|index| {
            index.property == *property
                && index.mixed_type_complete
                && (index.format == crate::manifest::PropertyIndexFormat::PagedV1
                    || index.paged.is_some()
                    || index.paged_build_unsupported)
        })
    })
}

fn search_indexes_need_rebuild(manifest: &crate::manifest::Manifest) -> bool {
    let max_node_lsn = manifest
        .ssts
        .iter()
        .filter(|sst| sst.kind == SstKind::Nodes)
        .map(|sst| sst.max_lsn)
        .max()
        .unwrap_or(0);
    let valid_lsm = crate::search_lsm::validate_search_lsm(manifest).is_ok();
    #[cfg(feature = "vector-index")]
    if manifest.vector_indexes.iter().any(|index| {
        let matching_states = manifest
            .search_lsm
            .iter()
            .filter(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Vector
                    && state.index_name == index.name
            })
            .collect::<Vec<_>>();
        if !matching_states.is_empty() {
            return !(valid_lsm
                && matching_states.len() == 1
                && matching_states[0].status == crate::search_lsm::SearchLsmStatus::Active);
        }
        if legacy_search_base_needs_adoption(
            manifest,
            crate::search_lsm::SearchLsmKind::Vector,
            &index.name,
            max_node_lsn,
        ) {
            return false;
        }
        let signature = vector_catalog_signature(manifest, index);
        !manifest.search_index_builds.iter().any(|state| {
            state.kind == SstKind::VectorGraph
                && state.name == index.name
                && state.catalog_signature == signature
                && state.max_node_lsn >= max_node_lsn
        })
    }) {
        return true;
    }
    #[cfg(feature = "text-index")]
    if manifest.text_indexes.iter().any(|index| {
        let matching_states = manifest
            .search_lsm
            .iter()
            .filter(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Text
                    && state.index_name == index.name
            })
            .collect::<Vec<_>>();
        if !matching_states.is_empty() {
            return !(valid_lsm
                && matching_states.len() == 1
                && matching_states[0].status == crate::search_lsm::SearchLsmStatus::Active);
        }
        if legacy_search_base_needs_adoption(
            manifest,
            crate::search_lsm::SearchLsmKind::Text,
            &index.name,
            max_node_lsn,
        ) {
            return false;
        }
        let signature = text_catalog_signature(index);
        !manifest.search_index_builds.iter().any(|state| {
            state.kind == SstKind::TextIndex
                && state.name == index.name
                && state.catalog_signature == signature
                && state.max_node_lsn >= max_node_lsn
        })
    }) {
        return true;
    }
    let _ = (max_node_lsn, valid_lsm);
    false
}

fn legacy_search_base_needs_adoption(
    manifest: &crate::manifest::Manifest,
    kind: crate::search_lsm::SearchLsmKind,
    index_name: &str,
    max_node_lsn: u64,
) -> bool {
    if manifest
        .search_lsm
        .iter()
        .any(|state| state.kind == kind && state.index_name == index_name)
    {
        return false;
    }
    let mut bodies = manifest.ssts.iter().filter(|descriptor| {
        descriptor.kind == kind.sst_kind()
            && descriptor.scope == index_name
            && !crate::search_lsm::is_canonical_search_barrier_descriptor(descriptor)
    });
    let Some(base) = bodies.next() else {
        return false;
    };
    if bodies.next().is_some() || base.max_lsn < max_node_lsn {
        return false;
    }
    let Some((canonical_signature, legacy_signature)) =
        adoption_catalog_signatures(manifest, kind, index_name)
    else {
        return false;
    };
    manifest.search_index_builds.iter().any(|state| {
        state.kind == kind.sst_kind()
            && state.name == index_name
            && state.max_node_lsn >= max_node_lsn
            && (state.catalog_signature == canonical_signature
                || state.catalog_signature == legacy_signature)
    })
}

/// Resolve one unambiguous current catalog entry and both signatures that a
/// pre-Search-LSM build marker may legitimately carry. Accepting only these
/// values prevents a DDL-changed label/property/analyzer from wrapping a
/// physically valid but semantically stale V5/V3 body into a new generation.
fn adoption_catalog_signatures(
    manifest: &crate::manifest::Manifest,
    kind: crate::search_lsm::SearchLsmKind,
    index_name: &str,
) -> Option<(String, String)> {
    match kind {
        crate::search_lsm::SearchLsmKind::Vector => {
            let mut matches = manifest
                .vector_indexes
                .iter()
                .filter(|descriptor| descriptor.name == index_name);
            let descriptor = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            Some((
                crate::search_lsm::vector_catalog_signature(manifest, descriptor),
                crate::search_lsm::legacy_vector_catalog_signature(manifest, descriptor),
            ))
        }
        crate::search_lsm::SearchLsmKind::Text => {
            let mut matches = manifest
                .text_indexes
                .iter()
                .filter(|descriptor| descriptor.name == index_name);
            let descriptor = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            Some((
                crate::search_lsm::text_catalog_signature(descriptor),
                crate::search_lsm::legacy_text_catalog_signature(descriptor),
            ))
        }
    }
}

fn search_lsm_adoption_needed(manifest: &crate::manifest::Manifest) -> bool {
    let max_node_lsn = manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .map(|descriptor| descriptor.max_lsn)
        .max()
        .unwrap_or(0);
    #[cfg(feature = "vector-index")]
    if manifest.vector_indexes.iter().any(|index| {
        legacy_search_base_needs_adoption(
            manifest,
            crate::search_lsm::SearchLsmKind::Vector,
            &index.name,
            max_node_lsn,
        )
    }) {
        return true;
    }
    #[cfg(feature = "text-index")]
    if manifest.text_indexes.iter().any(|index| {
        legacy_search_base_needs_adoption(
            manifest,
            crate::search_lsm::SearchLsmKind::Text,
            &index.name,
            max_node_lsn,
        )
    }) {
        return true;
    }
    let _ = (manifest, max_node_lsn);
    false
}

#[cfg(feature = "vector-index")]
fn vector_catalog_signature(
    manifest: &crate::manifest::Manifest,
    index: &crate::manifest::VectorIndexDescriptor,
) -> String {
    crate::search_lsm::vector_catalog_signature(manifest, index)
}

#[cfg(feature = "text-index")]
fn text_catalog_signature(index: &crate::manifest::TextIndexDescriptor) -> String {
    crate::search_lsm::text_catalog_signature(index)
}

fn catalog_build_states(
    manifest: &crate::manifest::Manifest,
    max_node_lsn: u64,
    attempted: &HashSet<(SstKind, String)>,
) -> Vec<crate::manifest::SearchIndexBuildState> {
    #[allow(unused_mut)]
    let mut states = Vec::new();
    #[cfg(feature = "vector-index")]
    states.extend(
        manifest
            .vector_indexes
            .iter()
            .filter(|index| attempted.contains(&(SstKind::VectorGraph, index.name.clone())))
            .map(|index| crate::manifest::SearchIndexBuildState {
                kind: SstKind::VectorGraph,
                name: index.name.clone(),
                catalog_signature: vector_catalog_signature(manifest, index),
                max_node_lsn,
            }),
    );
    #[cfg(feature = "text-index")]
    states.extend(
        manifest
            .text_indexes
            .iter()
            .filter(|index| attempted.contains(&(SstKind::TextIndex, index.name.clone())))
            .map(|index| crate::manifest::SearchIndexBuildState {
                kind: SstKind::TextIndex,
                name: index.name.clone(),
                catalog_signature: text_catalog_signature(index),
                max_node_lsn,
            }),
    );
    let _ = (manifest, max_node_lsn, attempted);
    states
}

async fn recover_preserved_search_barrier(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    projected: &crate::manifest::Manifest,
    kind: crate::search_lsm::SearchLsmKind,
    index_name: &str,
    scoped: &[SstDescriptor],
) -> Option<PreparedSearchLsmActivation> {
    let barriers = scoped
        .iter()
        .filter(|descriptor| crate::search_lsm::is_canonical_search_barrier_descriptor(descriptor))
        .collect::<Vec<_>>();
    if barriers.is_empty() {
        return None;
    }

    let mut valid = Vec::new();
    for barrier in barriers {
        let body = match get_sst_body(store.as_ref(), paths, barrier).await {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(
                    index = index_name,
                    path = %barrier.path,
                    error = %error,
                    "preserved Search-LSM barrier is unavailable"
                );
                continue;
            }
        };
        if body.len() as u64 != barrier.size_bytes {
            tracing::warn!(
                index = index_name,
                path = %barrier.path,
                manifest_size = barrier.size_bytes,
                object_size = body.len(),
                "preserved Search-LSM barrier size disagrees with its descriptor"
            );
            continue;
        }
        let state = match crate::search_lsm::decode_search_barrier(&body) {
            Ok(state) => state,
            Err(error) => {
                tracing::warn!(
                    index = index_name,
                    path = %barrier.path,
                    error = %error,
                    "preserved Search-LSM barrier is corrupt"
                );
                continue;
            }
        };
        if state.kind != kind
            || state.index_name != index_name
            || state.compat_barrier_sst_id != Some(barrier.id)
            || crate::search_lsm::validate_search_barrier(&state, &body).is_err()
        {
            tracing::warn!(
                index = index_name,
                path = %barrier.path,
                "preserved Search-LSM barrier identity does not match its descriptor"
            );
            continue;
        }
        let mut validation = projected.clone();
        validation
            .search_lsm
            .retain(|existing| existing.kind != kind || existing.index_name != index_name);
        validation.search_lsm.push(state.clone());
        if let Err(error) = crate::search_lsm::validate_search_lsm(&validation) {
            tracing::warn!(
                index = index_name,
                path = %barrier.path,
                error = %error,
                "preserved Search-LSM state is no longer valid for the visible catalog/corpus"
            );
            continue;
        }
        valid.push(PreparedSearchLsmActivation {
            state,
            barrier: barrier.clone(),
            barrier_already_present: true,
        });
    }

    if valid.len() == 1 {
        return valid.pop();
    }
    if valid.len() > 1 {
        tracing::warn!(
            index = index_name,
            candidates = valid.len(),
            "multiple preserved Search-LSM barriers validate; retiring ambiguity and rebuilding"
        );
    }
    None
}

async fn prepare_search_lsm_activations(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    basis: &crate::manifest::Manifest,
    new_descs: &[SstDescriptor],
    removed_ids: &[Uuid],
    new_build_states: &[crate::manifest::SearchIndexBuildState],
    replaced: &[(crate::search_lsm::SearchLsmKind, String)],
) -> Result<(
    Vec<PreparedSearchLsmActivation>,
    Vec<Uuid>,
    Vec<(SstKind, String)>,
)> {
    use crate::search_lsm::{
        encode_search_barrier, search_barrier_descriptor, wrap_legacy_search_base, SearchLsmKind,
    };
    let mut unadoptable: Vec<(SstKind, String)> = Vec::new();

    let removed = removed_ids.iter().copied().collect::<HashSet<_>>();
    let mut projected = basis.clone();
    projected
        .ssts
        .retain(|descriptor| !removed.contains(&descriptor.id));
    projected.ssts.extend(new_descs.iter().cloned());
    for state in new_build_states {
        projected
            .search_index_builds
            .retain(|existing| existing.kind != state.kind || existing.name != state.name);
        projected.search_index_builds.push(state.clone());
    }
    projected.search_lsm.retain(|state| {
        !replaced
            .iter()
            .any(|(kind, name)| state.kind == *kind && state.index_name == *name)
    });

    #[allow(unused_mut)]
    let mut candidates: Vec<(SearchLsmKind, String)> = Vec::new();
    #[cfg(feature = "vector-index")]
    candidates.extend(
        projected
            .vector_indexes
            .iter()
            .map(|index| (SearchLsmKind::Vector, index.name.clone())),
    );
    #[cfg(feature = "text-index")]
    candidates.extend(
        projected
            .text_indexes
            .iter()
            .map(|index| (SearchLsmKind::Text, index.name.clone())),
    );

    let mut activations = Vec::new();
    let mut retired_barriers = Vec::new();
    let max_node_lsn = projected
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .map(|descriptor| descriptor.max_lsn)
        .max()
        .unwrap_or(0);
    for (kind, index_name) in candidates {
        if projected
            .search_lsm
            .iter()
            .any(|state| state.kind == kind && state.index_name == index_name)
        {
            continue;
        }
        let scoped_snapshot = projected
            .ssts
            .iter()
            .filter(|descriptor| {
                descriptor.kind == kind.sst_kind() && descriptor.scope == index_name
            })
            .cloned()
            .collect::<Vec<_>>();
        if let Some(recovered) = recover_preserved_search_barrier(
            store.clone(),
            paths,
            &projected,
            kind,
            &index_name,
            &scoped_snapshot,
        )
        .await
        {
            activations.push(recovered);
            continue;
        }

        // An old writer can preserve the ordinary `.slb` descriptor while
        // dropping the unknown top-level state. If its footer is corrupt,
        // ambiguous, or stale, retire only that tiny compatibility artifact
        // and either re-adopt the still-fresh base or let the normal rebuild
        // path replace it. Keeping it would make two physical bodies look
        // permanently ambiguous.
        let stale_barrier_ids = scoped_snapshot
            .iter()
            .filter(|descriptor| {
                crate::search_lsm::is_canonical_search_barrier_descriptor(descriptor)
            })
            .map(|descriptor| descriptor.id)
            .collect::<HashSet<_>>();
        if !stale_barrier_ids.is_empty() {
            retired_barriers.extend(stale_barrier_ids.iter().copied());
            projected
                .ssts
                .retain(|descriptor| !stale_barrier_ids.contains(&descriptor.id));
        }
        if !legacy_search_base_needs_adoption(&projected, kind, &index_name, max_node_lsn) {
            continue;
        }
        let scoped = projected
            .ssts
            .iter()
            .filter(|descriptor| {
                descriptor.kind == kind.sst_kind() && descriptor.scope == index_name
            })
            .collect::<Vec<_>>();
        let [base] = scoped.as_slice() else {
            continue;
        };
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), base.path);
        let object_path = Path::from(absolute);
        // Only the feature-gated magic comparisons below read this probe.
        #[cfg_attr(
            not(any(feature = "vector-index", feature = "text-index")),
            allow(unused_variables)
        )]
        let magic = match store.get_range(&object_path, 0..8).await {
            Ok(magic) => magic,
            Err(error) => {
                tracing::warn!(
                    index = %index_name,
                    path = %base.path,
                    error = %error,
                    "cannot probe legacy search base for Search-LSM adoption"
                );
                continue;
            }
        };
        let supported_magic = match kind {
            #[cfg(feature = "vector-index")]
            SearchLsmKind::Vector => magic.as_ref() == crate::sst::vector::v5::MAGIC_V5,
            #[cfg(not(feature = "vector-index"))]
            SearchLsmKind::Vector => false,
            #[cfg(feature = "text-index")]
            SearchLsmKind::Text => magic.as_ref() == crate::sst::text::RANGE_READABLE_MAGIC,
            #[cfg(not(feature = "text-index"))]
            SearchLsmKind::Text => false,
        };
        if !supported_magic {
            // Deterministic: this body can never satisfy the adoption the
            // marker certifies. Dropping the marker un-suppresses the full
            // rebuild; a transient read error above keeps the marker and
            // retries instead.
            tracing::warn!(
                index = %index_name,
                path = %base.path,
                "legacy search base magic is not adoptable; scheduling a rebuild"
            );
            unadoptable.push((kind.sst_kind(), index_name.clone()));
            continue;
        }

        let generation_id = Uuid::now_v7();
        let barrier_id = Uuid::now_v7();
        let state = match wrap_legacy_search_base(
            &projected,
            kind,
            &index_name,
            base.id,
            generation_id,
            barrier_id,
        ) {
            Ok(state) => state,
            Err(error) => {
                tracing::warn!(
                    index = %index_name,
                    error = %error,
                    "legacy search base is not safe to adopt; scheduling a rebuild"
                );
                unadoptable.push((kind.sst_kind(), index_name.clone()));
                continue;
            }
        };
        let body = encode_search_barrier(&state)
            .map_err(|error| Error::invariant(format!("barrier encode failed: {error}")))?;
        let file_name = format!(
            "{}-{}-{}.slb",
            uuid_path_id(&barrier_id),
            kind.sst_kind().path_tag(),
            index_name
        );
        let object_path = paths.sst_object(base.level.as_u32(), &file_name);
        let relative_path = relative_sst_path(base.level.as_u32(), &file_name);
        crate::flush::put_object(store.clone(), &object_path, body.clone()).await?;
        let barrier = search_barrier_descriptor(
            &state,
            barrier_id,
            base.level,
            relative_path,
            body.len() as u64,
        );
        activations.push(PreparedSearchLsmActivation {
            state,
            barrier,
            barrier_already_present: false,
        });
    }
    Ok((activations, retired_barriers, unadoptable))
}

/// Run one leveled-lite compaction sweep across every `(kind, scope)`
/// bucket, reading the level budgets from the environment.
pub async fn compact_l0_to_l1(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    base: &LoadedManifest,
    schema: &Schema,
) -> Result<CompactionOutcome> {
    compact_leveled(
        manifest_store,
        fence,
        base,
        schema,
        compaction_base_bytes(),
        compaction_level_ratio(),
    )
    .await
}

/// Prepare one leveled-lite compaction sweep with the environment-configured
/// budgets: plan every bucket, GET the inputs, merge, rebuild the SST-backed
/// indexes, and PUT every output — WITHOUT committing a manifest. Pair with
/// [`install_prepared`]. Callers that hold no writer lock reach this through
/// [`CompactionBasis::prepare`].
pub async fn prepare_compaction(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    base: &LoadedManifest,
    schema: &Schema,
) -> Result<PreparedCompaction> {
    prepare_leveled(
        manifest_store,
        fence,
        base,
        schema,
        compaction_base_bytes(),
        compaction_level_ratio(),
    )
    .await
}

/// Run one leveled-lite compaction sweep with explicit level budgets —
/// [`prepare_leveled`] and [`install_prepared`] back-to-back against the
/// same base. The public [`compact_l0_to_l1`] wraps this with the
/// environment-configured budgets; tests call it directly with small
/// budgets to exercise the cascade deterministically without touching
/// process-wide env.
async fn compact_leveled(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    base: &LoadedManifest,
    schema: &Schema,
    base_bytes: u64,
    ratio: u64,
) -> Result<CompactionOutcome> {
    let prepared = prepare_leveled(manifest_store, fence, base, schema, base_bytes, ratio).await?;
    install_prepared(manifest_store, fence, base, prepared).await
}

/// Prepare phase of [`compact_leveled`]: everything expensive — planning,
/// input GETs, the CPU merges and index rebuilds, and every output PUT.
/// The new bodies land at immutable UUID paths no manifest references, so
/// a prepare that is never installed strands only unreferenced garbage the
/// janitor's orphan sweep reclaims.
#[instrument(
 skip(manifest_store, fence, base, schema),
 fields(
 namespace = %manifest_store.paths().namespace(),
 base_version = base.manifest.version,
 )
)]
async fn prepare_leveled(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    base: &LoadedManifest,
    schema: &Schema,
    base_bytes: u64,
    ratio: u64,
) -> Result<PreparedCompaction> {
    fence.assert_alive(base.manifest.epoch)?;

    // Group every SST by (kind, scope), every level together, so
    // `plan_bucket_merge` sees the whole bucket shape and can decide which
    // levels to merge and the output level. Tombstone and superseded-version
    // GC (RFC-027 P3) is gated below on the merge being authoritative: the
    // output is the deepest level (no older un-merged level below it can hold
    // a shadowed row), and for nodes the bucket is single-scope. A reader
    // pinned at an older version reads the retained source bodies, not the new
    // SST (the horizon-aware sweep keeps them alive).
    let mut node_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    let mut fwd_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    let mut inv_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    let mut vector_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    let mut text_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    for desc in &base.manifest.ssts {
        match desc.kind {
            SstKind::Nodes => node_buckets
                .entry(desc.scope.clone())
                .or_default()
                .push(desc),
            SstKind::EdgesFwd => fwd_buckets
                .entry(desc.scope.clone())
                .or_default()
                .push(desc),
            SstKind::EdgesInv => inv_buckets
                .entry(desc.scope.clone())
                .or_default()
                .push(desc),
            // VectorGraph SSTs (RFC-030 / `vector-index`). Bucketed per index
            // name (the descriptor scope). With the feature off none are ever
            // written, so this stays empty.
            SstKind::VectorGraph => vector_buckets
                .entry(desc.scope.clone())
                .or_default()
                .push(desc),
            // TextIndex SSTs (`text-index`). Bucketed per index name; empty when
            // the feature is off (none are ever written).
            SstKind::TextIndex => text_buckets
                .entry(desc.scope.clone())
                .or_default()
                .push(desc),
        }
    }

    let store = manifest_store.store().clone();
    let paths = manifest_store.paths();
    let mut new_descs: Vec<SstDescriptor> = Vec::new();
    let mut removed_ids: Vec<Uuid> = Vec::new();
    let mut bloom_count: usize = 0;
    let mut search_build_states = Vec::new();
    let mut node_rewrites = Vec::new();
    let search_policy = search_lsm_compact::SearchCompactionPolicy::from_env()?;
    let mut search_selections =
        search_lsm_compact::select_search_compactions(&base.manifest, search_policy)?;

    // Node tombstone GC needs the merge authoritative for every node key it
    // touches. Nodes are id-primary, so a key can live in any node SST
    // regardless of scope; if more than one node scope is present (a legacy
    // per-label SST alongside the id-primary `""` one), a single-scope merge
    // is NOT authoritative and dropping a tombstone could resurrect a live
    // row from the other scope. Restrict node GC to the single-scope case
    // (the id-primary norm); the per-bucket deepest-level check below adds the
    // second condition. Edges are keyed within `(edge_type, direction)`, so an
    // edge bucket is authoritative on its own and only the deepest-level check
    // applies.
    let node_scopes: HashSet<&str> = base
        .manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes)
        .map(|d| d.scope.as_str())
        .collect();
    let node_gc_safe = node_scopes.len() <= 1;
    let rebuild_search = node_gc_safe && search_indexes_need_rebuild(&base.manifest);

    // Nodes.
    for (label, sources) in node_buckets {
        let label_def = schema.label(&label).cloned().unwrap_or_else(|| LabelDef {
            name: label.clone(),
            properties: vec![],
        });
        // Sidecar-harvesting def: for the id-primary "" bucket the label_def is
        // empty (no declared columns), so unique/equality sidecars would be
        // harvested from zero properties and silently dropped on every
        // compaction — degrading indexed lookups to full label scans. Mirror
        // flush: harvest from the schema's union of indexed properties. Legacy
        // per-label buckets keep their own def.
        let sidecar_def = if label.is_empty() {
            crate::flush::union_indexed_props(schema)
        } else {
            label_def.clone()
        };
        let Some(plan) =
            plan_node_bucket(&sources, base_bytes, ratio, &sidecar_def, rebuild_search)
        else {
            continue;
        };
        // A lone, otherwise-current legacy SST needs only its access bundle
        // (`.nloc2` + `.npp`). Rewriting its complete Parquet body would retain
        // another multi-gigabyte vector corpus and double disk/network I/O for
        // no logical data change. Build both sidecars in one bounded source
        // pass, preserve every existing descriptor/search body, and replace
        // only the locator bundle in the next manifest.
        if plan.inputs.len() == 1
            && (!crate::manifest::node_locator_has_exact_records(plan.inputs[0])
                || !node_descriptor_has_property_pages(plan.inputs[0]))
            && !node_descriptor_needs_non_record_migration(plan.inputs[0], &sidecar_def)
            && !rebuild_search
        {
            let source = (*plan.inputs[0]).clone();
            let body = get_sst_body(store.as_ref(), paths, &source).await?;
            let locator_label = label_def.clone();
            let parent_sst_id = source.id;
            let (locator_upload, property_upload) = run_cpu(move || {
                build_node_access_sidecars_from_source(body, &locator_label, parent_sst_id)
            })
            .await??;
            if locator_upload.entry_count() != source.row_count
                || property_upload.stats().node_count != source.row_count
            {
                return Err(Error::invariant(
                    "rebuilt node access bundle row count differs from parent Nodes SST",
                ));
            }
            // Use a fresh sidecar UUID even though the authoritative Parquet
            // descriptor keeps its id. A failed pre-manifest upload can then be
            // retried without colliding with a complete orphan from the prior
            // attempt.
            let sidecar_id = Uuid::now_v7();
            let (mut node_locator, (sidecar_path, sidecar_body)) =
                crate::flush::prepare_node_locator_upload_sidecar(
                    paths,
                    source.level.as_u32(),
                    &sidecar_id,
                    locator_upload,
                )?;
            let (property_pages, (property_path, property_body)) =
                crate::flush::prepare_node_property_pages_upload_sidecar(
                    paths,
                    source.level.as_u32(),
                    &source.id,
                    &sidecar_id,
                    property_upload,
                )?;
            node_locator.property_pages = Some(property_pages);
            crate::flush::put_sidecar_payload(store.clone(), &sidecar_path, sidecar_body).await?;
            crate::flush::put_sidecar_payload(store.clone(), &property_path, property_body).await?;
            let mut migrated = source.clone();
            migrated.node_locator = Some(node_locator);
            node_rewrites.push(PreparedNodeRewrite {
                inputs: vec![source.clone()],
                output: Some(migrated.clone()),
            });
            removed_ids.push(source.id);
            new_descs.push(migrated);
            continue;
        }
        // GC tombstones only when this merge is authoritative: a single node
        // scope (no other scope can hold the key) AND the output is the
        // bucket's deepest level (no older un-merged level below it).
        let gc = node_gc_safe && plan.is_deepest;
        // GET every input body up front as a file-backed mapping. Even small
        // L0 files avoid heap residency because their aggregate fan-in can be
        // large; decoded rows remain bounded by the merge cursor batch size.
        let mut bodies: Vec<NodeMergeInput> = Vec::with_capacity(plan.inputs.len());
        for desc in &plan.inputs {
            bodies.push(NodeMergeInput {
                body: get_sst_body(store.as_ref(), paths, desc).await?,
                min_key: desc.min_key,
            });
        }
        // Vector/text member collection happens during the winner stream, and
        // is gated on both a stale search generation and an authoritative
        // merge of the FULL corpus: deepest level AND a single node scope
        // (`gc`). `plan.is_deepest` alone treated a per-bucket deepest merge in
        // a mixed-scope namespace (legacy per-label + id-primary "" scopes)
        // as corpus-complete, rebuilding the index from one bucket and
        // permanently truncating it — the same rule node-tombstone GC uses.
        // On a partial merge, or a physical-only migration whose durable build
        // markers are already fresh, the spec list stays empty and the
        // existing `.vg`/`.ft` remains untouched.
        // A sidecar-only migration rewrites the physical node SST without
        // changing the logical search corpus. If the durable build markers
        // already cover this node generation, retain the existing `.vg` /
        // `.ft` descriptors instead of cloning every embedding/document and
        // rebuilding corpus-sized indexes just to attach a new node sidecar.
        // `gc` is still required: a stale search generation may only rebuild
        // from an authoritative, full-corpus merge.
        let rebuild_search_for_bucket = gc && rebuild_search;
        let index_specs = NodeMergeIndexSpecs {
            #[cfg(feature = "vector-index")]
            vector: if rebuild_search_for_bucket {
                base.manifest.vector_indexes.clone()
            } else {
                Vec::new()
            },
            #[cfg(feature = "text-index")]
            text: if rebuild_search_for_bucket {
                base.manifest.text_indexes.clone()
            } else {
                Vec::new()
            },
        };
        // The whole k-way merge (per-row-group decode, heap, winner
        // re-encode, incremental Parquet write, sidecar/stat/index-member
        // harvesting) is pure CPU over the owned bodies; run it on the
        // blocking pool so a large bucket does not stall the async runtime
        // for its duration.
        let merge_def = label_def.clone();
        let merge_sidecar_def = sidecar_def;
        let merge_schema = schema.clone();
        let merge_dict = base.manifest.label_dict.clone();
        let merge_scope = label.clone();
        let output_sst_id = Uuid::now_v7();
        let out = run_cpu(move || {
            merge_node_sources(
                bodies,
                &merge_def,
                &merge_sidecar_def,
                gc,
                &merge_schema,
                &merge_dict,
                &merge_scope,
                output_sst_id,
                index_specs,
            )
        })
        .await??;
        // The highest LSN covered by this authoritative generation. Use the
        // source high-water mark rather than only the surviving winner rows:
        // an all-tombstone merge has no output rows, but it still advances the
        // search corpus to an authoritatively empty generation.
        let finish_max_lsn = plan
            .inputs
            .iter()
            .map(|source| source.max_lsn)
            .max()
            .unwrap_or(out.finish.stats.max_lsn);

        // Rebuild vector/text bodies before the empty-node-body fast path.
        // An authoritative merge can legitimately reduce an index corpus to
        // zero (or one vector): that successful empty build must remove the
        // prior body and persist a generation marker, otherwise compaction
        // either loops forever or keeps serving a stale pre-delete index.
        #[allow(unused_mut)]
        let mut attempted_search_builds: HashSet<(SstKind, String)> = HashSet::new();

        #[cfg(feature = "vector-index")]
        if rebuild_search_for_bucket {
            let (new_vg, old_vg_ids, attempted) = build_vector_indexes_from_members(
                store.clone(),
                paths,
                plan.target_level,
                finish_max_lsn,
                out.vector_members,
                &vector_buckets,
            )
            .await?;
            new_descs.extend(new_vg);
            removed_ids.extend(old_vg_ids);
            attempted_search_builds.extend(
                attempted
                    .into_iter()
                    .map(|name| (SstKind::VectorGraph, name)),
            );
        }

        #[cfg(feature = "text-index")]
        if rebuild_search_for_bucket {
            let (new_ft, old_ft_ids, attempted) = build_text_indexes_from_members(
                store.clone(),
                paths,
                plan.target_level,
                finish_max_lsn,
                out.text_members,
                &text_buckets,
            )
            .await?;
            new_descs.extend(new_ft);
            removed_ids.extend(old_ft_ids);
            attempted_search_builds
                .extend(attempted.into_iter().map(|name| (SstKind::TextIndex, name)));
        }

        if rebuild_search_for_bucket {
            search_build_states =
                catalog_build_states(&base.manifest, finish_max_lsn, &attempted_search_builds);
        }
        if out.finish.stats.row_count == 0 {
            if !gc {
                return Err(Error::invariant(
                    "non-authoritative node compaction produced an empty output",
                ));
            }
            // Nothing to write; still mark the merged sources for removal so
            // the bucket truly shrinks.
            node_rewrites.push(PreparedNodeRewrite {
                inputs: plan.inputs.iter().map(|source| (*source).clone()).collect(),
                output: None,
            });
            for src in &plan.inputs {
                removed_ids.push(src.id);
            }
            continue;
        }
        let (mut descriptor, wrote_bloom) = put_node_sst_leveled(
            store.clone(),
            paths,
            plan.target_level,
            &label,
            output_sst_id,
            out.sidecars,
            out.finish,
        )
        .await?;
        // The descriptor is also the corpus freshness fence used by the
        // vector/text readers. Preserve the source high-water mark even when
        // the highest-LSN winner was a tombstone removed by authoritative GC;
        // otherwise the freshly rebuilt search indexes appear to outrun the
        // node corpus and every read falls back to a flat scan.
        descriptor.max_lsn = finish_max_lsn;
        if wrote_bloom {
            bloom_count += 1;
        }
        for src in &plan.inputs {
            removed_ids.push(src.id);
        }
        node_rewrites.push(PreparedNodeRewrite {
            inputs: plan.inputs.iter().map(|source| (*source).clone()).collect(),
            output: Some(descriptor.clone()),
        });
        new_descs.push(descriptor);
    }

    // Edges (forward).
    for (edge_type, sources) in fwd_buckets {
        let Some(plan) = plan_bucket_merge(&sources, base_bytes, ratio) else {
            continue;
        };
        let (desc, wrote_bloom, removed) = compact_and_write_edges(
            store.clone(),
            paths,
            schema,
            &edge_type,
            &plan.inputs,
            EdgeDirection::Forward,
            plan.target_level,
            plan.is_deepest,
        )
        .await?;
        if wrote_bloom {
            bloom_count += 1;
        }
        for id in removed {
            removed_ids.push(id);
        }
        if let Some(d) = desc {
            new_descs.push(d);
        }
    }

    // Edges (inverse).
    for (edge_type, sources) in inv_buckets {
        let Some(plan) = plan_bucket_merge(&sources, base_bytes, ratio) else {
            continue;
        };
        let (desc, wrote_bloom, removed) = compact_and_write_edges(
            store.clone(),
            paths,
            schema,
            &edge_type,
            &plan.inputs,
            EdgeDirection::Inverse,
            plan.target_level,
            plan.is_deepest,
        )
        .await?;
        if wrote_bloom {
            bloom_count += 1;
        }
        for id in removed {
            removed_ids.push(id);
        }
        if let Some(d) = desc {
            new_descs.push(d);
        }
    }

    // VectorGraph SSTs (RFC-030 / `vector-index`). A Vamana graph is not
    // row-mergeable: on a compaction that picks up an existing VectorGraph
    // bucket, the index is *rebuilt* from the current merged node rows by
    // `build_vector_indexes_for_nodes` (feature-gated, invoked in the
    // node-bucket loop), and the prior VectorGraph SSTs are marked for removal
    // there. Surviving (not-rebuilt) VectorGraph SSTs carry forward via
    // `next.ssts.retain`. Off-feature no VectorGraph SST is ever written, so the
    // bucket is empty — keep a use so the binding isn't flagged unused.
    #[cfg(not(feature = "vector-index"))]
    let _ = &vector_buckets;

    let replaced_search_lsm = search_build_states
        .iter()
        .filter_map(|state| match state.kind {
            SstKind::VectorGraph => {
                Some((crate::search_lsm::SearchLsmKind::Vector, state.name.clone()))
            }
            SstKind::TextIndex => {
                Some((crate::search_lsm::SearchLsmKind::Text, state.name.clone()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    // A stale/building catalog causes the authoritative Nodes merge above to
    // rebuild every registered legacy base. Do not also publish a physical
    // Search-LSM base for the same key in this pass: the two builders select
    // different replacement contracts and would retire each other's inputs.
    // Prefer the logical rebuild, while independent active generations may
    // still compact physically in the same off-lock prepare.
    search_selections.retain(|selection| {
        let overlaps = replaced_search_lsm.iter().any(|(kind, name)| {
            *kind == selection.captured_state.kind && *name == selection.captured_state.index_name
        });
        if overlaps {
            tracing::warn!(
                index = %selection.captured_state.index_name,
                kind = ?selection.captured_state.kind,
                abort_reason = "simultaneous_logical_search_rebuild",
                "skipping redundant physical Search-LSM compaction"
            );
        }
        !overlaps
    });
    let search_compactions = search_lsm_compact::prepare_search_compactions(
        store.clone(),
        paths,
        base,
        search_selections,
    )
    .await?;
    for search in &search_compactions {
        removed_ids.extend(search.selection.selected_ids());
        if let Some(output) = search.output_descriptor() {
            new_descs.push(output.clone());
        }
    }
    let (search_lsm_activations, retired_search_barriers, unadoptable_search_markers) =
        prepare_search_lsm_activations(
            store,
            paths,
            &base.manifest,
            &new_descs,
            &removed_ids,
            &search_build_states,
            &replaced_search_lsm,
        )
        .await?;
    removed_ids.extend(retired_search_barriers);

    if !node_rewrites.is_empty() {
        validate_search_lsm(&base.manifest).map_err(|error| {
            Error::invariant(format!(
                "cannot prepare a Nodes rewrite from invalid Search-LSM state: {error}"
            ))
        })?;
    }

    // Every BasePrefix consolidation certifies a complete physical base for
    // its index, which is exactly what a 2.0.6-interop marker means: without
    // one, a downgraded writer that drops the unknown `search_lsm` state
    // leaves a manifest the upgrade path can only repair by rebuilding the
    // corpus instead of re-adopting the surviving base metadata-only. The
    // basis high-water is correct because the BasePrefix build scans every
    // Nodes SST at this basis; a flush racing prepare→install leaves the
    // marker conservatively stale, which forces a rebuild rather than a
    // wrong adoption. Empty-output consolidations (authoritatively empty
    // corpus) mint a marker too, mirroring the 2.0.6 rule that keeps
    // compaction from replanning an empty build forever.
    #[cfg_attr(
        not(any(feature = "vector-index", feature = "text-index")),
        allow(unused_variables)
    )]
    let basis_max_node_lsn = base
        .manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .map(|descriptor| descriptor.max_lsn)
        .max()
        .unwrap_or(0);
    // The signature must be the one the 2.0.6 downgrade paths compare
    // against (`catalog_build_states` / `adoption_catalog_signatures`), NOT
    // the native `captured_state` signature: for text those forks (the LSM
    // form hashes the manifest-scoped filter set), and a marker carrying a
    // signature nobody accepts certifies nothing.
    #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
    let consolidated_base_markers = Vec::new();
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    let consolidated_base_markers = search_compactions
        .iter()
        .filter(|search| {
            search.selection.mode == search_lsm_compact::SearchCompactionMode::BasePrefix
        })
        .filter_map(|search| {
            let name = search.selection.captured_state.index_name.as_str();
            let catalog_signature = match search.selection.captured_state.kind {
                #[cfg(feature = "vector-index")]
                crate::search_lsm::SearchLsmKind::Vector => base
                    .manifest
                    .vector_indexes
                    .iter()
                    .find(|index| index.name == name)
                    .map(|index| vector_catalog_signature(&base.manifest, index))?,
                #[cfg(feature = "text-index")]
                crate::search_lsm::SearchLsmKind::Text => base
                    .manifest
                    .text_indexes
                    .iter()
                    .find(|index| index.name == name)
                    .map(text_catalog_signature)?,
                // With the feature off no search compaction is ever selected.
                #[cfg(not(feature = "vector-index"))]
                crate::search_lsm::SearchLsmKind::Vector => return None,
                #[cfg(not(feature = "text-index"))]
                crate::search_lsm::SearchLsmKind::Text => return None,
            };
            Some(crate::manifest::SearchIndexBuildState {
                kind: search.selection.captured_state.kind.sst_kind(),
                name: name.to_owned(),
                catalog_signature,
                max_node_lsn: basis_max_node_lsn,
            })
        })
        .collect();

    Ok(PreparedCompaction {
        new_descs,
        removed_ids,
        bloom_count,
        base_version: base.manifest.version,
        base_schema: base.manifest.schema.clone(),
        base_vector_indexes: base.manifest.vector_indexes.clone(),
        base_text_indexes: base.manifest.text_indexes.clone(),
        base_search_lsm: base.manifest.search_lsm.clone(),
        node_rewrites,
        search_compactions,
        search_build_states,
        consolidated_base_markers,
        unadoptable_search_markers,
        search_lsm_activations,
        replaced_search_lsm,
    })
}

#[derive(Debug)]
struct RebasedBarrier {
    old_id: Uuid,
    state: SearchLsmState,
    descriptor: SstDescriptor,
    object_path: Path,
    body: Bytes,
}

fn verify_node_rewrite_inputs(
    manifest: &crate::manifest::Manifest,
    rewrites: &[PreparedNodeRewrite],
) -> Result<()> {
    let mut claimed = HashSet::new();
    for rewrite in rewrites {
        if rewrite.inputs.is_empty() {
            return Err(Error::invariant(
                "prepared Nodes rewrite has no input descriptors",
            ));
        }
        for input in &rewrite.inputs {
            if input.kind != SstKind::Nodes || !claimed.insert(input.id) {
                return Err(Error::invariant(
                    "prepared Nodes rewrite inputs are not unique Nodes descriptors",
                ));
            }
            let mut matches = manifest
                .ssts
                .iter()
                .filter(|descriptor| descriptor.id == input.id);
            match (matches.next(), matches.next()) {
                (Some(current), None) if current == input => {}
                (Some(_), None) => {
                    return Err(Error::precondition(format!(
                        "abandoning prepared compaction: Nodes input {} changed after prepare",
                        input.id
                    )));
                }
                _ => {
                    return Err(Error::precondition(format!(
                        "abandoning prepared compaction: Nodes input {} is missing or ambiguous",
                        input.id
                    )));
                }
            }
        }
        if let Some(output) = &rewrite.output {
            if output.kind != SstKind::Nodes {
                return Err(Error::invariant(
                    "prepared Nodes rewrite output is not a Nodes descriptor",
                ));
            }
        }
    }
    Ok(())
}

fn verify_search_state_append_only(
    captured: &SearchLsmState,
    current: &SearchLsmState,
) -> Result<()> {
    let identity_matches = current.kind == captured.kind
        && current.index_name == captured.index_name
        && current.catalog_signature == captured.catalog_signature
        && current.generation_id == captured.generation_id
        && current.status == captured.status
        && current.base_frontier == captured.base_frontier
        && current.equal_lsn_conflict_count == captured.equal_lsn_conflict_count;
    let prefix_matches = current.next_event_seq >= captured.next_event_seq
        && current.segments.starts_with(&captured.segments)
        && current
            .proven_empty_event_ranges
            .starts_with(&captured.proven_empty_event_ranges)
        && current.coverage.starts_with(&captured.coverage);
    if !identity_matches || !prefix_matches {
        return Err(Error::precondition(format!(
            "abandoning prepared compaction: Search-LSM generation '{}' did not advance \
             append-only from the captured prefix",
            captured.index_name
        )));
    }
    Ok(())
}

fn compressed_coverage_union(coverages: &[SearchCoverage]) -> Result<Vec<SearchEventRange>> {
    let mut ranges = coverages
        .iter()
        .flat_map(|coverage| coverage.event_ranges.iter().copied())
        .collect::<Vec<_>>();
    if ranges.is_empty() || ranges.iter().any(|range| range.start >= range.end) {
        return Err(Error::invariant(
            "Nodes rewrite input coverage has no valid event range",
        ));
    }
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut compressed: Vec<SearchEventRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = compressed.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        compressed.push(range);
    }
    Ok(compressed)
}

fn input_coverage_digest(coverages: &[SearchCoverage]) -> Result<u64> {
    let mut ordered = coverages.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|coverage| *coverage.node_sst_id.as_bytes());
    let mut hasher = Xxh3::new();
    hasher.update(b"NamiDB/NodesCoverageLogicalRewrite/v1");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for coverage in ordered {
        hasher.update(coverage.node_sst_id.as_bytes());
        hasher.update(&coverage.node_sst_max_lsn.to_le_bytes());
        hasher.update(&(coverage.event_ranges.len() as u64).to_le_bytes());
        for range in &coverage.event_ranges {
            if range.start >= range.end {
                return Err(Error::invariant(
                    "Nodes rewrite input coverage contains an invalid event range",
                ));
            }
            hasher.update(&range.start.to_le_bytes());
            hasher.update(&range.end.to_le_bytes());
        }
        match coverage.disposition {
            CoverageDisposition::Segment => hasher.update(&[1]),
            CoverageDisposition::ProvenEmpty {
                classifier_version,
                before_after_digest,
            } => {
                if classifier_version == 0 || before_after_digest == 0 {
                    return Err(Error::invariant(
                        "Nodes rewrite input has invalid ProvenEmpty coverage",
                    ));
                }
                hasher.update(&[2]);
                hasher.update(&classifier_version.to_le_bytes());
                hasher.update(&before_after_digest.to_le_bytes());
            }
            CoverageDisposition::LogicalRewrite {
                input_coverage_digest,
            } => {
                if input_coverage_digest == 0 {
                    return Err(Error::invariant(
                        "Nodes rewrite input has a zero coverage digest",
                    ));
                }
                hasher.update(&[3]);
                hasher.update(&input_coverage_digest.to_le_bytes());
            }
            CoverageDisposition::Unknown => {
                return Err(Error::invariant(
                    "Nodes rewrite input has unknown coverage disposition",
                ));
            }
        }
    }
    Ok(hasher.digest().max(1))
}

fn rewrite_one_state_coverage(
    captured: &SearchLsmState,
    current: &mut SearchLsmState,
    rewrite: &PreparedNodeRewrite,
) -> Result<bool> {
    let input_ids = rewrite
        .inputs
        .iter()
        .map(|descriptor| descriptor.id)
        .collect::<HashSet<_>>();
    let mut captured_inputs = Vec::with_capacity(input_ids.len());
    let mut missing_inputs = 0usize;
    for input in &rewrite.inputs {
        let matches = captured
            .coverage
            .iter()
            .filter(|coverage| coverage.node_sst_id == input.id)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [coverage] => captured_inputs.push((*coverage).clone()),
            [] if captured.status == SearchLsmStatus::Building => {
                missing_inputs += 1;
            }
            [] => {
                return Err(Error::precondition(format!(
                    "abandoning prepared compaction: active Search-LSM '{}' did not cover Nodes \
                     input {} at prepare time",
                    captured.index_name, input.id
                )));
            }
            _ => {
                return Err(Error::invariant(
                    "captured Search-LSM has duplicate Nodes coverage",
                ));
            }
        }
    }
    if missing_inputs != 0 {
        if captured_inputs.is_empty() {
            return Ok(false);
        }
        return Err(Error::precondition(format!(
            "abandoning prepared compaction: Building Search-LSM '{}' has partial coverage for \
             the selected Nodes inputs",
            captured.index_name
        )));
    }
    for coverage in &captured_inputs {
        let matches = current
            .coverage
            .iter()
            .filter(|candidate| candidate.node_sst_id == coverage.node_sst_id)
            .collect::<Vec<_>>();
        if matches.as_slice() != [coverage] {
            return Err(Error::precondition(format!(
                "abandoning prepared compaction: Nodes input coverage {} changed after prepare",
                coverage.node_sst_id
            )));
        }
    }

    if let Some(output) = &rewrite.output {
        if current.coverage.iter().any(|coverage| {
            coverage.node_sst_id == output.id && !input_ids.contains(&coverage.node_sst_id)
        }) {
            return Err(Error::precondition(format!(
                "abandoning prepared compaction: Nodes output UUID {} already has unrelated coverage",
                output.id
            )));
        }
    }

    let inherited_event_ranges = compressed_coverage_union(&captured_inputs)?;
    let digest = input_coverage_digest(&captured_inputs)?;
    let replacement = rewrite.output.as_ref().map(|output| SearchCoverage {
        node_sst_id: output.id,
        node_sst_max_lsn: output.max_lsn,
        event_ranges: inherited_event_ranges,
        disposition: CoverageDisposition::LogicalRewrite {
            input_coverage_digest: digest,
        },
    });
    let first_input = current
        .coverage
        .iter()
        .position(|coverage| input_ids.contains(&coverage.node_sst_id))
        .ok_or_else(|| {
            Error::precondition(
                "abandoning prepared compaction: captured Nodes inputs disappeared from coverage",
            )
        })?;
    let old = std::mem::take(&mut current.coverage);
    let mut next = Vec::with_capacity(
        old.len()
            .saturating_sub(input_ids.len())
            .saturating_add(usize::from(replacement.is_some())),
    );
    let mut replacement = replacement;
    for (position, coverage) in old.into_iter().enumerate() {
        if position == first_input {
            if let Some(replacement) = replacement.take() {
                next.push(replacement);
            }
        }
        if !input_ids.contains(&coverage.node_sst_id) {
            next.push(coverage);
        }
    }
    current.coverage = next;
    Ok(true)
}

fn rebase_search_lsm_for_node_rewrites(
    captured_states: &[SearchLsmState],
    current_states: &[SearchLsmState],
    rewrites: &[PreparedNodeRewrite],
) -> Result<Vec<SearchLsmState>> {
    if rewrites.is_empty() {
        return Ok(Vec::new());
    }
    let captured_keys = captured_states
        .iter()
        .map(|state| (state.kind, state.index_name.as_str()))
        .collect::<HashSet<_>>();
    let current_keys = current_states
        .iter()
        .map(|state| (state.kind, state.index_name.as_str()))
        .collect::<HashSet<_>>();
    if captured_keys.len() != captured_states.len()
        || current_keys.len() != current_states.len()
        || captured_keys != current_keys
    {
        return Err(Error::precondition(
            "abandoning prepared compaction: Search-LSM generation set changed after prepare",
        ));
    }

    let mut rebased = Vec::new();
    for captured in captured_states {
        let current = current_states
            .iter()
            .find(|state| state.kind == captured.kind && state.index_name == captured.index_name)
            .expect("state key sets checked above");
        verify_search_state_append_only(captured, current)?;
        let mut state = current.clone();
        let mut changed = false;
        for rewrite in rewrites {
            changed |= rewrite_one_state_coverage(captured, &mut state, rewrite)?;
        }
        if changed {
            rebased.push(state);
        }
    }
    Ok(rebased)
}

fn verify_search_replacement_inputs(
    captured_states: &[SearchLsmState],
    current_states: &[SearchLsmState],
    replaced: &[(crate::search_lsm::SearchLsmKind, String)],
    activations: &[PreparedSearchLsmActivation],
) -> Result<()> {
    let mut affected = replaced.iter().cloned().collect::<HashSet<_>>();
    affected.extend(
        activations
            .iter()
            .map(|activation| (activation.state.kind, activation.state.index_name.clone())),
    );
    for (kind, index_name) in affected {
        let captured = captured_states
            .iter()
            .filter(|state| state.kind == kind && state.index_name == index_name)
            .collect::<Vec<_>>();
        let current = current_states
            .iter()
            .filter(|state| state.kind == kind && state.index_name == index_name)
            .collect::<Vec<_>>();
        if captured != current {
            return Err(Error::precondition(format!(
                "abandoning prepared compaction: physically replaced Search-LSM generation \
                 '{index_name}' changed after prepare"
            )));
        }
    }
    Ok(())
}

fn prepare_rebased_barrier(
    paths: &NamespacePaths,
    state: &SearchLsmState,
) -> Result<Option<RebasedBarrier>> {
    let mut state = state.clone();
    let Some(old_id) = state.compat_barrier_sst_id else {
        if state.status == SearchLsmStatus::Active {
            return Err(Error::invariant(
                "active Search-LSM rewrite has no compatibility barrier",
            ));
        }
        return Ok(None);
    };
    let barrier_id = Uuid::now_v7();
    state.compat_barrier_sst_id = Some(barrier_id);
    let body = encode_search_barrier(&state)
        .map_err(|error| Error::invariant(format!("rebased barrier encode failed: {error}")))?;
    let file_name = format!(
        "{}-{}-{}.slb",
        uuid_path_id(&barrier_id),
        state.kind.sst_kind().path_tag(),
        state.index_name
    );
    let level = SstLevel::L0;
    let object_path = paths.sst_object(level.as_u32(), &file_name);
    let descriptor = search_barrier_descriptor(
        &state,
        barrier_id,
        level,
        relative_sst_path(level.as_u32(), &file_name),
        body.len() as u64,
    );
    crate::search_lsm::validate_search_barrier(&state, &body)
        .map_err(|error| Error::invariant(format!("rebased barrier self-check failed: {error}")))?;
    Ok(Some(RebasedBarrier {
        old_id,
        state,
        descriptor,
        object_path,
        body,
    }))
}

/// Commit phase of the prepare/commit split: fold a [`PreparedCompaction`]
/// into `current` — the manifest at commit time, which may have advanced
/// past the prepare's basis via writes and flushes — and run the
/// fence-checked manifest CAS.
///
/// A flush that landed during the prepare simply contributed new L0 SSTs:
/// they survive into `next` untouched and merge on a later sweep, and an
/// SST-backed index (`.vg` / `.ft`) rebuilt by this prepare is older than
/// such an L0, so the LSN freshness gate
/// ([`crate::read::Snapshot::index_outrun_by_nodes`]) already routes those
/// reads to the exact flat scan.
///
/// Every merged input must still be referenced by `current`: writes and
/// flushes only ADD SSTs, so only another compaction (or a DROP INDEX)
/// removes them, and folding this plan in anyway would resurrect
/// merged-away descriptors. A missing input aborts the install with
/// [`Error::Precondition`], leaving the manifest untouched; the prepared
/// bodies stay unreferenced for the janitor's orphan sweep.
///
/// The schema and vector/text index catalogs must also still match the
/// prepare basis. Normal data commits and flushes preserve those catalogs and
/// remain safe to interleave; DDL does not, because it changes which columns,
/// property sidecars, or search-index bodies the compacted outputs must carry.
#[instrument(
 skip(manifest_store, fence, current, prepared),
 fields(
 namespace = %manifest_store.paths().namespace(),
 base_version = prepared.base_version,
 current_version = current.manifest.version,
 )
)]
pub async fn install_prepared(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    current: &LoadedManifest,
    prepared: PreparedCompaction,
) -> Result<CompactionOutcome> {
    if prepared.removed_ids.is_empty()
        && prepared.search_lsm_activations.is_empty()
        && prepared.unadoptable_search_markers.is_empty()
    {
        debug!("compactor found no bucket worth merging; nothing to install");
        return Ok(CompactionOutcome {
            committed: current.clone(),
            source_ssts_removed: 0,
            new_ssts_written: 0,
            bloom_sidecars_written: 0,
        });
    }
    fence.assert_alive(current.manifest.epoch)?;

    let mut catalog_drift = Vec::new();
    if current.manifest.schema != prepared.base_schema {
        catalog_drift.push("schema");
    }
    if current.manifest.vector_indexes != prepared.base_vector_indexes {
        catalog_drift.push("vector indexes");
    }
    if current.manifest.text_indexes != prepared.base_text_indexes {
        catalog_drift.push("text indexes");
    }
    if !catalog_drift.is_empty() {
        return Err(Error::precondition(format!(
            "abandoning prepared compaction (basis v{}): {} changed in manifest v{}; \
             the prepared bodies are left for the orphan sweep",
            prepared.base_version,
            catalog_drift.join(", "),
            current.manifest.version
        )));
    }

    verify_search_replacement_inputs(
        &prepared.base_search_lsm,
        &current.manifest.search_lsm,
        &prepared.replaced_search_lsm,
        &prepared.search_lsm_activations,
    )?;
    let mut rebased_search_states = if prepared.node_rewrites.is_empty() {
        Vec::new()
    } else {
        validate_search_lsm(&current.manifest).map_err(|error| {
            Error::precondition(format!(
                "abandoning prepared compaction (basis v{}): current Search-LSM state is \
                 invalid: {error}",
                prepared.base_version
            ))
        })?;
        verify_node_rewrite_inputs(&current.manifest, &prepared.node_rewrites)?;
        // States this very prepare REPLACES (the authoritative rebuild path —
        // e.g. a freshly recreated index still Building with partial
        // coverage) are retired below, not rebased: validating their partial
        // coverage here would abort exactly the compaction that repairs
        // them, wedging maintenance after every DROP+CREATE INDEX cycle.
        let replaced_keys: HashSet<(crate::search_lsm::SearchLsmKind, &str)> = prepared
            .replaced_search_lsm
            .iter()
            .map(|(kind, name)| (*kind, name.as_str()))
            .collect();
        let captured_kept: Vec<crate::search_lsm::SearchLsmState> = prepared
            .base_search_lsm
            .iter()
            .filter(|state| !replaced_keys.contains(&(state.kind, state.index_name.as_str())))
            .cloned()
            .collect();
        let current_kept: Vec<crate::search_lsm::SearchLsmState> = current
            .manifest
            .search_lsm
            .iter()
            .filter(|state| !replaced_keys.contains(&(state.kind, state.index_name.as_str())))
            .cloned()
            .collect();
        rebase_search_lsm_for_node_rewrites(&captured_kept, &current_kept, &prepared.node_rewrites)?
    };
    if !prepared.search_compactions.is_empty() {
        if prepared.node_rewrites.is_empty() {
            validate_search_lsm(&current.manifest).map_err(|error| {
                Error::precondition(format!(
                    "abandoning prepared compaction (basis v{}): current Search-LSM state is \
                     invalid: {error}",
                    prepared.base_version
                ))
            })?;
        }
        let mut physical_keys = HashSet::new();
        for search in &prepared.search_compactions {
            let key = (
                search.selection.captured_state.kind,
                search.selection.captured_state.index_name.clone(),
            );
            if !physical_keys.insert(key.clone()) {
                return Err(Error::invariant(
                    "prepared compaction contains duplicate physical Search-LSM plans",
                ));
            }
            let output_missing = search.output_descriptor().is_some_and(|output| {
                !prepared
                    .new_descs
                    .iter()
                    .any(|descriptor| descriptor == output)
            });
            if output_missing
                || search
                    .selection
                    .selected_ids()
                    .any(|id| !prepared.removed_ids.contains(&id))
            {
                return Err(Error::invariant(
                    "prepared physical Search-LSM plan is not reflected in descriptor changes",
                ));
            }
            let logical_position = rebased_search_states
                .iter()
                .position(|state| state.kind == key.0 && state.index_name == key.1);
            let working = match logical_position {
                Some(position) => rebased_search_states.remove(position),
                None => current
                    .manifest
                    .search_lsm
                    .iter()
                    .find(|state| state.kind == key.0 && state.index_name == key.1)
                    .cloned()
                    .ok_or_else(|| {
                        Error::precondition(format!(
                            "abandoning physical Search-LSM compaction: generation '{}' \
                             disappeared",
                            key.1
                        ))
                    })?,
            };
            rebased_search_states.push(search_lsm_compact::rebase_prepared_search_compaction(
                &current.manifest,
                search,
                &working,
            )?);
        }
    }
    if rebased_search_states.iter().any(|state| {
        prepared
            .replaced_search_lsm
            .iter()
            .any(|(kind, name)| state.kind == *kind && state.index_name == *name)
    }) {
        return Err(Error::precondition(
            "abandoning prepared compaction: one Search-LSM generation was both logically \
             rewritten and physically replaced",
        ));
    }

    let live: HashSet<Uuid> = current.manifest.ssts.iter().map(|d| d.id).collect();
    if let Some(missing) = prepared.removed_ids.iter().find(|id| !live.contains(id)) {
        return Err(Error::precondition(format!(
            "abandoning prepared compaction (basis v{}): input SST {missing} is no longer \
 referenced by manifest v{}; the prepared bodies are left for the orphan sweep",
            prepared.base_version, current.manifest.version
        )));
    }

    let mut source_count = prepared.removed_ids.len();
    let mut new_count = prepared.new_descs.len();
    let mut installed_activation_count = 0usize;
    let mut next = current.manifest.next_version(fence.writer_id);
    let removed_set: HashSet<Uuid> = prepared.removed_ids.into_iter().collect();
    next.ssts.retain(|d| !removed_set.contains(&d.id));
    next.ssts.extend(prepared.new_descs);
    next.search_lsm.retain(|state| {
        !prepared
            .replaced_search_lsm
            .iter()
            .any(|(kind, name)| state.kind == *kind && state.index_name == *name)
    });
    for state in prepared.search_build_states {
        next.search_index_builds
            .retain(|existing| existing.kind != state.kind || existing.name != state.name);
        next.search_index_builds.push(state);
    }
    let mut dropped_unadoptable_markers = 0usize;
    for (kind, name) in &prepared.unadoptable_search_markers {
        let before = next.search_index_builds.len();
        next.search_index_builds
            .retain(|marker| !(marker.kind == *kind && marker.name == *name));
        dropped_unadoptable_markers += before - next.search_index_builds.len();
    }
    // Position-preserving upsert: migration tests pin unrelated-marker order.
    // Install is all-or-nothing, so a marker can never commit without the
    // base it certifies.
    for marker in prepared.consolidated_base_markers {
        match next
            .search_index_builds
            .iter_mut()
            .find(|existing| existing.kind == marker.kind && existing.name == marker.name)
        {
            Some(existing) => *existing = marker,
            None => next.search_index_builds.push(marker),
        }
    }
    for activation in prepared.search_lsm_activations {
        let mut expected_nodes = activation
            .state
            .coverage
            .iter()
            .map(|coverage| (coverage.node_sst_id, coverage.node_sst_max_lsn))
            .collect::<Vec<_>>();
        let mut visible_nodes = next
            .ssts
            .iter()
            .filter(|descriptor| descriptor.kind == SstKind::Nodes)
            .map(|descriptor| (descriptor.id, descriptor.max_lsn))
            .collect::<Vec<_>>();
        expected_nodes.sort_unstable();
        visible_nodes.sort_unstable();
        if expected_nodes != visible_nodes {
            tracing::info!(
                index = %activation.state.index_name,
                "prepared Search-LSM activation was outrun by a concurrent node commit; \
                 installing the legacy base without its barrier and retaining exact fallback"
            );
            continue;
        }

        let mut validation = next.clone();
        if !activation.barrier_already_present {
            validation.ssts.push(activation.barrier.clone());
        }
        validation.search_lsm.retain(|state| {
            state.kind != activation.state.kind || state.index_name != activation.state.index_name
        });
        validation.search_lsm.push(activation.state.clone());
        if let Err(error) = crate::search_lsm::validate_search_lsm(&validation) {
            tracing::warn!(
                index = %activation.state.index_name,
                error = %error,
                "prepared Search-LSM activation failed final validation; retaining exact fallback"
            );
            continue;
        }
        next.search_lsm.retain(|state| {
            state.kind != activation.state.kind || state.index_name != activation.state.index_name
        });
        if !activation.barrier_already_present {
            next.ssts.push(activation.barrier);
            new_count += 1;
        }
        next.search_lsm.push(activation.state);
        installed_activation_count += 1;
    }

    let mut rebased_barriers = Vec::new();
    for state in rebased_search_states {
        let final_state = match prepare_rebased_barrier(manifest_store.paths(), &state)? {
            Some(rotation) => {
                next.ssts
                    .retain(|descriptor| descriptor.id != rotation.old_id);
                next.ssts.push(rotation.descriptor.clone());
                source_count = source_count.saturating_add(1);
                new_count = new_count.saturating_add(1);
                let final_state = rotation.state.clone();
                rebased_barriers.push(rotation);
                final_state
            }
            None => state,
        };
        let position = next
            .search_lsm
            .iter()
            .position(|existing| {
                existing.kind == final_state.kind && existing.index_name == final_state.index_name
            })
            .ok_or_else(|| {
                Error::precondition(
                    "abandoning prepared compaction: rebased Search-LSM generation disappeared",
                )
            })?;
        next.search_lsm[position] = final_state;
    }

    validate_search_lsm(&next).map_err(|error| {
        Error::precondition(format!(
            "abandoning prepared compaction (basis v{}): rebased Search-LSM manifest is invalid: \
             {error}",
            prepared.base_version
        ))
    })?;
    let mut barrier_puts = futures::stream::FuturesUnordered::new();
    for rotation in rebased_barriers {
        let store = manifest_store.store().clone();
        barrier_puts.push(async move {
            crate::flush::put_sidecar_payload(
                store,
                &rotation.object_path,
                SidecarPayload::InMemory(rotation.body.into()),
            )
            .await
        });
    }
    while let Some(result) = barrier_puts.next().await {
        result?;
    }

    if source_count == 0
        && new_count == 0
        && installed_activation_count == 0
        && dropped_unadoptable_markers == 0
    {
        debug!("prepared Search-LSM activation was outrun; nothing to install");
        return Ok(CompactionOutcome {
            committed: current.clone(),
            source_ssts_removed: 0,
            new_ssts_written: 0,
            bloom_sidecars_written: 0,
        });
    }
    let committed = manifest_store.commit(fence, current, next).await?;

    Ok(CompactionOutcome {
        committed,
        source_ssts_removed: source_count,
        new_ssts_written: new_count,
        bloom_sidecars_written: prepared.bloom_count,
    })
}

#[allow(clippy::too_many_arguments)]
async fn compact_and_write_edges(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    schema: &Schema,
    edge_type: &str,
    sources: &[&SstDescriptor],
    direction: EdgeDirection,
    level: u32,
    gc_tombstones: bool,
) -> Result<(Option<SstDescriptor>, bool, Vec<Uuid>)> {
    let edge_def = schema.edge_type(edge_type).cloned();
    let declared_property_names: Vec<String> = edge_def
        .as_ref()
        .map(|def| def.properties.iter().map(|p| p.name.clone()).collect())
        .unwrap_or_default();
    // GET every source body up front as a file-backed mapping, then k-way
    // stream the merge on the blocking pool — decoded partner blocks and
    // property strings never accumulate beyond the per-source cursor positions.
    let mut bodies: Vec<Bytes> = Vec::with_capacity(sources.len());
    for desc in sources {
        bodies.push(get_sst_body(store.as_ref(), paths, desc).await?);
    }
    let merge_type = edge_type.to_string();
    let merge_def = edge_def.clone();
    let finish = run_cpu(move || {
        merge_edge_sources(
            bodies,
            &merge_type,
            merge_def.as_ref(),
            &declared_property_names,
            direction,
            gc_tombstones,
        )
    })
    .await??;
    let removed: Vec<Uuid> = sources.iter().map(|d| d.id).collect();
    if finish.stats.edge_count == 0 {
        return Ok((None, false, removed));
    }
    let (descriptor, wrote_bloom) =
        put_edge_sst_leveled(store, paths, level, edge_type, direction, finish).await?;
    Ok((Some(descriptor), wrote_bloom, removed))
}

// ── Streaming k-way node merge ──────────────────────────────────────────

/// Typed column accessors for one decoded batch of a node source, downcast
/// once per batch so per-row peeks during the merge stay cheap.
struct NodeBatchView {
    batch: RecordBatch,
    ids: FixedSizeBinaryArray,
    tombstones: BooleanArray,
    lsns: UInt64Array,
    overflow: StringArray,
    schema_versions: UInt64Array,
}

impl NodeBatchView {
    fn new(batch: RecordBatch) -> Result<Self> {
        fn col<T: Clone + 'static>(batch: &RecordBatch, name: &str) -> Result<T> {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<T>())
                .cloned()
                .ok_or_else(|| Error::invariant(format!("{name} column missing")))
        }
        Ok(Self {
            ids: col(&batch, COL_NODE_ID)?,
            tombstones: col(&batch, COL_TOMBSTONE)?,
            lsns: col(&batch, COL_LSN)?,
            overflow: col(&batch, OVERFLOW_JSON)?,
            schema_versions: col(&batch, SCHEMA_VERSION)?,
            batch,
        })
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn key(&self, row: usize) -> Result<([u8; 16], u64)> {
        let id: [u8; 16] = self
            .ids
            .value(row)
            .try_into()
            .map_err(|_| Error::invariant("node_id row length != 16"))?;
        Ok((id, self.lsns.value(row)))
    }

    /// Convert one row to a [`NodeRow`] — the JSON property-map re-encode
    /// the merge pays for winners only. Returns the decoded record alongside
    /// so the sidecar/stat/index-member collectors don't re-decode it.
    fn materialize(
        &self,
        row: usize,
        label_def: &LabelDef,
    ) -> Result<(NodeRow, Option<NodeWriteRecord>)> {
        let (id, lsn) = self.key(row)?;
        if self.tombstones.value(row) {
            return Ok((
                NodeRow {
                    id,
                    lsn,
                    op: MemOp::Tombstone,
                },
                None,
            ));
        }
        // Rebuild properties: declared columns + overflow_json.
        let mut properties: BTreeMap<String, Value> = BTreeMap::new();
        for p in &label_def.properties {
            let col_name = prop_column_name(p);
            let col = self
                .batch
                .column_by_name(&col_name)
                .ok_or_else(|| Error::invariant(format!("missing column {col_name}")))?;
            if let Some(v) = arrow_value_to_value(col.as_ref(), row, &p.data_type)? {
                properties.insert(p.name.clone(), v);
            }
        }
        if !self.overflow.is_null(row) {
            let extra: BTreeMap<String, Value> = serde_json::from_str(self.overflow.value(row))?;
            properties.extend(extra);
        }
        let rec = NodeWriteRecord {
            properties,
            schema_version: self.schema_versions.value(row),
            // Preserve the on-row label set (raw LabelIds) so the merged
            // SST keeps it. Legacy SSTs have no __labels column and yield an
            // empty set; their output stays scope-typed and reads via
            // fallback.
            labels: raw_labels_from_batch(&self.batch, row),
        };
        let payload = rec.encode()?;
        Ok((
            NodeRow {
                id,
                lsn,
                op: MemOp::Upsert(payload),
            },
            Some(rec),
        ))
    }
}

/// One immutable node source and its manifest lower bound.
///
/// `min_key` lets the k-way merge defer decoding this source until its first
/// row can actually compete with the active heap minimum. The descriptor is
/// checked against the first decoded row before any result can commit.
struct NodeMergeInput {
    body: Bytes,
    min_key: [u8; 16],
}

/// Keep an activated source to at most one small decoded Arrow batch.
///
/// A vector-bearing row can be tens of KiB once `__overflow_json` is decoded.
/// The Parquet default of 1,024 rows multiplied by a large L0 fan-in is still
/// enough to exhaust a small host, even though the compressed inputs are
/// mmap-backed. Sixty-four rows bounds that fan-in term while preserving
/// sequential page decode.
const NODE_MERGE_INPUT_BATCH_ROWS: usize = 64;

/// Sorted row cursor over one node source SST.
///
/// Opening the cursor parses metadata and constructs a lazy Parquet reader but
/// does not decode its first batch. Activation happens only when the source's
/// manifest `min_key` reaches the active heap frontier. Once activated, at most
/// one `NODE_MERGE_INPUT_BATCH_ROWS` batch is retained; a complete row group is
/// never collected into memory.
struct NodeSourceCursor {
    batches: ParquetRecordBatchReader,
    /// Decoded CURRENT batch only.
    view: Option<NodeBatchView>,
    /// Row index into `view`.
    row: usize,
    /// `(id, lsn)` of the current row; `None` once exhausted.
    current: Option<([u8; 16], u64)>,
    /// Whether the lazy reader has been activated.
    started: bool,
    /// Batches decoded so far (test probe).
    batches_decoded: usize,
    /// Total row count per the Parquet footer (bloom sizing upper bound).
    total_rows: u64,
}

impl NodeSourceCursor {
    fn open(label_def: &LabelDef, body: Bytes) -> Result<Self> {
        Self::open_with_batch_rows(label_def, body, NODE_MERGE_INPUT_BATCH_ROWS)
    }

    fn open_with_batch_rows(label_def: &LabelDef, body: Bytes, batch_rows: usize) -> Result<Self> {
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            body,
            ArrowReaderOptions::new().with_page_index(true),
        )
        .map_err(|e| Error::invariant(format!("parquet open: {e}")))?;
        let expected = node_arrow_schema(label_def);
        let got = builder.schema();
        if got.fields().len() != expected.fields().len()
            || got
                .fields()
                .iter()
                .zip(expected.fields())
                .any(|(got, want)| got.name() != want.name() || got.data_type() != want.data_type())
        {
            return Err(Error::Corrupted {
                path: "<compaction-input>".into(),
                detail: "node SST schema does not match the declared node schema".into(),
            });
        }
        let total_rows = builder.metadata().file_metadata().num_rows().max(0) as u64;
        let batches = builder
            .with_batch_size(batch_rows.max(1))
            .build()
            .map_err(|e| Error::invariant(format!("parquet build: {e}")))?;
        Ok(Self {
            batches,
            view: None,
            row: 0,
            current: None,
            started: false,
            batches_decoded: 0,
            total_rows,
        })
    }

    /// Activate this source and decode only its first bounded batch.
    fn ensure_positioned(&mut self) -> Result<()> {
        if !self.started {
            self.started = true;
            self.position()?;
        }
        Ok(())
    }

    /// Advance `view`/`row` to the next available row (decoding one further
    /// bounded batch as needed) and cache its key in `current`.
    fn position(&mut self) -> Result<()> {
        loop {
            if let Some(view) = &self.view {
                if self.row < view.len() {
                    self.current = Some(view.key(self.row)?);
                    return Ok(());
                }
                self.view = None;
                self.row = 0;
            }
            match self.batches.next() {
                Some(Ok(batch)) => {
                    self.batches_decoded += 1;
                    if batch.num_rows() > 0 {
                        self.view = Some(NodeBatchView::new(batch)?);
                    }
                }
                Some(Err(error)) => {
                    return Err(Error::invariant(format!("parquet read: {error}")));
                }
                None => {
                    self.current = None;
                    return Ok(());
                }
            }
        }
    }

    /// `(id, lsn)` of the current row without materialising it.
    fn peek(&self) -> Option<([u8; 16], u64)> {
        self.current
    }

    /// Materialise the current row (winners only — losers skip straight to
    /// [`Self::advance`]).
    fn materialize_current(
        &self,
        label_def: &LabelDef,
    ) -> Result<(NodeRow, Option<NodeWriteRecord>)> {
        let view = self
            .view
            .as_ref()
            .ok_or_else(|| Error::invariant("node merge cursor materialised past its end"))?;
        view.materialize(self.row, label_def)
    }

    fn advance(&mut self) -> Result<()> {
        if !self.started {
            return Err(Error::invariant(
                "node merge cursor advanced before activation",
            ));
        }
        self.row += 1;
        self.position()
    }
}

fn build_node_access_sidecars_from_source(
    body: Bytes,
    label_def: &LabelDef,
    parent_sst_id: Uuid,
) -> Result<(
    crate::sst::paged_index::NodeLocatorRecordUpload,
    crate::sst::nodes::property_pages::NodePropertyPageUpload,
)> {
    let mut cursor = NodeSourceCursor::open(label_def, body)?;
    cursor.ensure_positioned()?;
    let mut locator = crate::sst::paged_index::NodeLocatorRecordBuilder::new();
    let mut properties = crate::sst::nodes::property_pages::NodePropertyPageBuilder::new_bound(
        crate::sst::nodes::property_pages::NodePropertyPageConfig::from_env()?,
        parent_sst_id,
    )?;
    let mut ordinal = 0_u64;
    while cursor.peek().is_some() {
        let (row, record) = cursor.materialize_current(label_def)?;
        let exact_record = encode_exact_node_record(&row)?;
        locator.push(&row.id, &exact_record)?;
        match record {
            Some(record) => properties.push_sorted(row.id, ordinal, &record.properties)?,
            None => properties.push_sorted(row.id, ordinal, &BTreeMap::new())?,
        }
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| Error::invariant("node property ordinal exceeds u64"))?;
        cursor.advance()?;
    }
    Ok((locator.finish_upload()?, properties.finish()?))
}

/// Heap key for the node k-way merge: id ascending, then LSN **descending**
/// (the first entry popped for an id is its winner), then source order —
/// the same total order the materialised merge's stable
/// `sort_by(id, lsn desc)` over plan-input-concatenated rows produced, so
/// exact `(id, lsn)` ties still resolve to the earlier source.
#[derive(PartialEq, Eq)]
struct NodeHeapEntry {
    id: [u8; 16],
    lsn: u64,
    src: usize,
}

impl Ord for NodeHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id
            .cmp(&other.id)
            .then(other.lsn.cmp(&self.lsn))
            .then(self.src.cmp(&other.src))
    }
}

impl PartialOrd for NodeHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Index descriptors whose members the streaming node merge collects while
/// the winner stream advances. Populated only for authoritative merges;
/// empty otherwise (and empty with the features off).
#[derive(Default)]
struct NodeMergeIndexSpecs {
    #[cfg(feature = "vector-index")]
    vector: Vec<VectorIndexDescriptor>,
    #[cfg(feature = "text-index")]
    text: Vec<crate::manifest::TextIndexDescriptor>,
}

/// Aggregate (not per-index) memory contract for all search-index collectors
/// retained by one authoritative node merge.
///
/// Text collectors buffer occurrences concurrently while the winner stream
/// advances, and those buffers remain alive while vector artifacts are built.
/// Giving every collector the full environment value therefore multiplies the
/// operator's intended ceiling by the catalog size. We partition the single
/// budget deterministically across every live collector. A builder may use
/// less than its share, but can never borrow memory that another still owns.
const DEFAULT_INDEX_BUILD_MEMORY_BYTES: usize = 256 * 1024 * 1024;
const MIN_INDEX_BUILD_MEMORY_PER_COLLECTOR: usize = 64 * 1024;

fn aggregate_index_build_memory_bytes() -> Result<usize> {
    const ENV: &str = "NAMIDB_INDEX_BUILD_MEMORY_BYTES";
    let bytes = match std::env::var(ENV) {
        Ok(value) => value.trim().parse::<usize>().map_err(|error| {
            Error::precondition(format!(
                "{ENV} must be an exact positive byte count: {error}"
            ))
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_INDEX_BUILD_MEMORY_BYTES,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::precondition(format!("{ENV} is not valid UTF-8")));
        }
    };
    if bytes == 0 {
        return Err(Error::precondition(format!(
            "{ENV} must be greater than zero"
        )));
    }
    Ok(bytes)
}

fn per_collector_index_build_memory_bytes(collector_count: usize) -> Result<Option<usize>> {
    if collector_count == 0 {
        return Ok(None);
    }
    let aggregate = aggregate_index_build_memory_bytes()?;
    partition_index_build_memory(aggregate, collector_count).map(Some)
}

fn partition_index_build_memory(aggregate: usize, collector_count: usize) -> Result<usize> {
    if collector_count == 0 {
        return Err(Error::invariant(
            "cannot partition search-index memory across zero collectors",
        ));
    }
    let per_collector = aggregate / collector_count;
    if per_collector < MIN_INDEX_BUILD_MEMORY_PER_COLLECTOR {
        return Err(Error::precondition(format!(
            "NAMIDB_INDEX_BUILD_MEMORY_BYTES={aggregate} cannot fund {collector_count} \
             concurrent search-index collectors: each requires at least \
             {MIN_INDEX_BUILD_MEMORY_PER_COLLECTOR} bytes"
        )));
    }
    Ok(per_collector)
}

/// Everything the streaming node merge harvests from the winner stream for
/// [`put_node_sst_leveled`], in place of the old `&merged_rows` re-walks.
struct NodeSidecarHarvest {
    unique: UniqueSidecarCollector,
    equality: EqualitySidecarCollector,
    label_index: LabelIndexCollector,
    node_locator_upload: crate::sst::paged_index::NodeLocatorRecordUpload,
    property_pages_upload: crate::sst::nodes::property_pages::NodePropertyPageUpload,
    per_label_property_stats: Vec<PerLabelPropertyStat>,
}

/// Per-index external vector collectors produced by the winner stream.
#[cfg(feature = "vector-index")]
type VectorIndexMembers = Vec<VectorMemberCollector>;

/// Per-index external text collectors produced by the winner stream.
#[cfg(feature = "text-index")]
type TextIndexMembers = Vec<TextMemberCollector>;

/// Output of [`merge_node_sources`].
struct NodeMergeOutput {
    finish: NodeSstFinish,
    sidecars: NodeSidecarHarvest,
    #[cfg(feature = "vector-index")]
    vector_members: VectorIndexMembers,
    #[cfg(feature = "text-index")]
    text_members: TextIndexMembers,
}

/// Rows buffered per output chunk during the streaming merge. Reads
/// `NAMIDB_COMPACTION_MERGE_CHUNK_ROWS` so chunk-boundary tests can force
/// tiny chunks; falls back to the flush path's 16 Ki.
fn merge_chunk_rows() -> usize {
    std::env::var("NAMIDB_COMPACTION_MERGE_CHUNK_ROWS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(NODE_SST_BATCH_ROWS)
}

/// K-way streaming merge of one node bucket. Preserves the materialised
/// merge's semantics exactly: per id the highest-LSN row wins (source order
/// breaks exact ties), lower-LSN versions are dropped, and a winning
/// tombstone is dropped entirely when `gc_tombstones` (RFC-027 P3 — see the
/// caller for the authority rule; readers pinned at older versions still
/// see the delete through the retained source bodies). Winners stream into
/// the incremental SST writer in bounded chunks and are observed by the
/// sidecar/stat harvesters and the vector/text member collectors as they
/// pass; shadowed duplicates are skipped without ever being materialised.
#[allow(clippy::too_many_arguments)]
fn merge_node_sources(
    inputs: Vec<NodeMergeInput>,
    label_def: &LabelDef,
    sidecar_def: &LabelDef,
    gc_tombstones: bool,
    schema: &Schema,
    label_dict: &LabelDictionary,
    bucket_scope: &str,
    output_sst_id: Uuid,
    index_specs: NodeMergeIndexSpecs,
) -> Result<NodeMergeOutput> {
    let mut cursors: Vec<NodeSourceCursor> = Vec::with_capacity(inputs.len());
    let mut unopened: BinaryHeap<Reverse<([u8; 16], usize)>> =
        BinaryHeap::with_capacity(inputs.len());
    let mut total_rows: u64 = 0;
    for (src, input) in inputs.into_iter().enumerate() {
        let cursor = NodeSourceCursor::open(label_def, input.body)?;
        total_rows = total_rows.saturating_add(cursor.total_rows);
        cursors.push(cursor);
        unopened.push(Reverse((input.min_key, src)));
    }

    // `expected_keys` sizes the bloom from the pre-dedup input total — an
    // upper bound (the merged count is unknowable without a second pass),
    // so the filter errs slightly larger / lower-FP than the materialised
    // merge's exact sizing. Everything else about the output is identical.
    let options = NodeSstWriterOptions {
        expected_keys: total_rows,
        ..Default::default()
    };
    let mut writer = IncrementalNodeSstWriter::new(label_def, options, merge_chunk_rows())?;
    let mut unique = UniqueSidecarCollector::new(sidecar_def)?;
    let mut equality = EqualitySidecarCollector::new(sidecar_def)?;
    let mut label_index = LabelIndexCollector::new()?;
    let mut node_locator_records = crate::sst::paged_index::NodeLocatorRecordBuilder::new();
    let mut property_pages = crate::sst::nodes::property_pages::NodePropertyPageBuilder::new_bound(
        crate::sst::nodes::property_pages::NodePropertyPageConfig::from_env()?,
        output_sst_id,
    )?;
    let mut output_ordinal = 0_u64;
    let mut stats = PerLabelStatsCollector::new();
    #[allow(unused_mut)]
    let mut search_collector_count = 0usize;
    #[cfg(feature = "vector-index")]
    {
        search_collector_count = search_collector_count
            .checked_add(index_specs.vector.len())
            .ok_or_else(|| Error::invariant("search-index collector count overflows usize"))?;
    }
    #[cfg(feature = "text-index")]
    {
        search_collector_count = search_collector_count
            .checked_add(index_specs.text.len())
            .ok_or_else(|| Error::invariant("search-index collector count overflows usize"))?;
    }
    // Only the feature-gated collectors below consume this budget, but the
    // check itself must still run so a misconfigured rail fails the same way in
    // every build.
    #[cfg_attr(
        not(any(feature = "vector-index", feature = "text-index")),
        allow(unused_variables)
    )]
    let per_collector_memory =
        per_collector_index_build_memory_bytes(search_collector_count)?.unwrap_or(0);
    #[cfg(feature = "vector-index")]
    let mut vector_collectors: Vec<VectorMemberCollector> = index_specs
        .vector
        .into_iter()
        .map(|desc| VectorMemberCollector::new(desc, label_dict, schema, per_collector_memory))
        .collect::<Result<Vec<_>>>()?;
    #[cfg(feature = "text-index")]
    let mut text_collectors: Vec<TextMemberCollector> = index_specs
        .text
        .into_iter()
        .map(|desc| TextMemberCollector::new(desc, label_dict, per_collector_memory))
        .collect::<Result<Vec<_>>>()?;
    #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
    let _ = &index_specs;

    let mut heap: BinaryHeap<Reverse<NodeHeapEntry>> = BinaryHeap::with_capacity(cursors.len());

    let mut last_id: Option<[u8; 16]> = None;
    loop {
        // A source whose manifest minimum is above the active heap minimum
        // cannot affect the next winner, so leave its Parquet batches entirely
        // undecoded. Activate every source at or below the frontier (including
        // equal minima, whose LSNs must participate in tie-breaking).
        while let Some(Reverse((hinted_min, src))) = unopened.peek().copied() {
            let active_min = heap.peek().map(|entry| entry.0.id);
            if active_min.is_some_and(|active| hinted_min > active) {
                break;
            }
            unopened.pop();
            let cursor = &mut cursors[src];
            cursor.ensure_positioned()?;
            let Some((id, lsn)) = cursor.peek() else {
                return Err(Error::invariant(
                    "manifest references an empty node compaction input",
                ));
            };
            if id != hinted_min {
                return Err(Error::Corrupted {
                    path: "<compaction-input>".into(),
                    detail: "node SST first key disagrees with manifest min_key".into(),
                });
            }
            heap.push(Reverse(NodeHeapEntry { id, lsn, src }));
        }

        let Some(Reverse(entry)) = heap.pop() else {
            break;
        };
        let cursor = &mut cursors[entry.src];
        if last_id != Some(entry.id) {
            // First (highest-LSN) observation of this id: the winner.
            last_id = Some(entry.id);
            let (row, rec) = cursor.materialize_current(label_def)?;
            if !(gc_tombstones && matches!(row.op, MemOp::Tombstone)) {
                let exact_record = encode_exact_node_record(&row)?;
                node_locator_records.push(&row.id, &exact_record)?;
                if let Some(rec) = &rec {
                    unique.observe(row.id, rec)?;
                    equality.observe(row.id, rec)?;
                    label_index.observe(row.id, rec)?;
                    stats.observe(rec);
                    #[cfg(feature = "vector-index")]
                    for collector in &mut vector_collectors {
                        collector.observe(row.id, rec, bucket_scope)?;
                    }
                    #[cfg(feature = "text-index")]
                    for collector in &mut text_collectors {
                        collector.observe(row.id, rec, bucket_scope)?;
                    }
                }
                match &rec {
                    Some(record) => {
                        property_pages.push_sorted(row.id, output_ordinal, &record.properties)?;
                    }
                    None => {
                        property_pages.push_sorted(row.id, output_ordinal, &BTreeMap::new())?;
                    }
                }
                output_ordinal = output_ordinal
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("node property ordinal exceeds u64"))?;
                writer.push(row)?;
            }
        }
        // Shadowed duplicate or consumed winner: step past it and re-arm.
        cursor.advance()?;
        if let Some((id, lsn)) = cursor.peek() {
            heap.push(Reverse(NodeHeapEntry {
                id,
                lsn,
                src: entry.src,
            }));
        }
    }

    #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
    let _ = bucket_scope;

    let finish = writer.finish()?;
    let node_locator_upload = node_locator_records.finish_upload()?;
    let property_pages_upload = property_pages.finish()?;
    if property_pages_upload.stats().node_count != finish.stats.row_count {
        return Err(Error::invariant(
            "compacted node property pages row count differs from Nodes SST",
        ));
    }
    let per_label_property_stats = stats.finish(schema, label_dict)?;
    Ok(NodeMergeOutput {
        finish,
        sidecars: NodeSidecarHarvest {
            unique,
            equality,
            label_index,
            node_locator_upload,
            property_pages_upload,
            per_label_property_stats,
        },
        #[cfg(feature = "vector-index")]
        vector_members: vector_collectors,
        #[cfg(feature = "text-index")]
        text_members: text_collectors,
    })
}

/// Read a node row's `__labels` column as raw `LabelId` values. Empty when the
/// SST predates the column (legacy single-label).
fn raw_labels_from_batch(batch: &arrow_array::RecordBatch, row: usize) -> Vec<u32> {
    let Some(list) = batch
        .column_by_name(COL_LABELS)
        .and_then(|c| c.as_any().downcast_ref::<ListArray>())
    else {
        return Vec::new();
    };
    if list.is_null(row) {
        return Vec::new();
    }
    match list.value(row).as_any().downcast_ref::<UInt32Array>() {
        Some(a) => (0..a.len())
            .filter(|&i| !a.is_null(i))
            .map(|i| a.value(i))
            .collect(),
        None => Vec::new(),
    }
}

// ── Streaming k-way edge merge ──────────────────────────────────────────

/// Incremental reader over one edge property stream. Legacy Arrow IPC is
/// streamed one record batch at a time; current codec-2/3 bodies decode one
/// independently compressed property page at a time.
struct PropertyStreamCursor {
    name: String,
    inner: PropertyStreamCursorInner,
    rows_read: u64,
    edge_count: u64,
}

enum PropertyStreamCursorInner {
    Arrow {
        reader: StreamReader<Box<dyn std::io::Read + Send>>,
        current: Option<StringArray>,
        row: usize,
    },
    Paged {
        bytes: Bytes,
        entries: std::vec::IntoIter<PropertyPageEntry>,
        compressed: bool,
        current: Vec<Option<String>>,
        row: usize,
    },
}

impl PropertyStreamCursor {
    fn open(name: &str, bytes: Bytes, codec: u8, edge_count: u64) -> Result<Self> {
        let inner = match codec {
            CODEC_NONE | CODEC_ZSTD => {
                let read: Box<dyn std::io::Read + Send> = if codec == CODEC_NONE {
                    Box::new(std::io::Cursor::new(bytes))
                } else {
                    Box::new(
                        zstd::stream::read::Decoder::new(std::io::Cursor::new(bytes)).map_err(
                            |e| {
                                Error::invariant(format!(
                                    "zstd decode (property stream {name}): {e}"
                                ))
                            },
                        )?,
                    )
                };
                let reader = StreamReader::try_new(read, None)
                    .map_err(|e| Error::invariant(format!("property IPC reader ({name}): {e}")))?;
                PropertyStreamCursorInner::Arrow {
                    reader,
                    current: None,
                    row: 0,
                }
            }
            CODEC_PROPERTY_PAGED_NONE | CODEC_PROPERTY_PAGED_ZSTD => {
                let index = PropertyPageIndex::parse_prefix(&bytes, bytes.len() as u64)?;
                if index.row_count != edge_count {
                    return Err(Error::Corrupted {
                        path: "<edges>".into(),
                        detail: format!(
                            "property stream {name} row count {} != edge_count {edge_count}",
                            index.row_count
                        ),
                    });
                }
                PropertyStreamCursorInner::Paged {
                    bytes,
                    entries: index.entries.into_iter(),
                    compressed: codec == CODEC_PROPERTY_PAGED_ZSTD,
                    current: Vec::new(),
                    row: 0,
                }
            }
            other => {
                return Err(Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!("unknown codec {other} for property stream {name}"),
                })
            }
        };
        Ok(Self {
            name: name.to_string(),
            inner,
            rows_read: 0,
            edge_count,
        })
    }

    /// Value for the next edge in enumeration order. `want == false` (a
    /// shadowed loser) still advances the stream but skips materialising
    /// the string.
    fn next(&mut self, want: bool) -> Result<Option<String>> {
        loop {
            match &mut self.inner {
                PropertyStreamCursorInner::Arrow {
                    reader,
                    current,
                    row,
                } => {
                    if let Some(batch) = current {
                        if *row < batch.len() {
                            let out = if want && !batch.is_null(*row) {
                                Some(batch.value(*row).to_string())
                            } else {
                                None
                            };
                            *row += 1;
                            self.rows_read += 1;
                            return Ok(out);
                        }
                        *current = None;
                    }
                    let Some(batch) = reader.next() else {
                        return property_short_stream_error(
                            &self.name,
                            self.rows_read,
                            self.edge_count,
                        );
                    };
                    let batch = batch.map_err(|e| {
                        Error::invariant(format!("property IPC batch ({}): {e}", self.name))
                    })?;
                    *current = Some(
                        batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                Error::invariant(format!(
                                    "property IPC column ({}) is not Utf8",
                                    self.name
                                ))
                            })?
                            .clone(),
                    );
                    *row = 0;
                }
                PropertyStreamCursorInner::Paged {
                    bytes,
                    entries,
                    compressed,
                    current,
                    row,
                } => {
                    if *row < current.len() {
                        let out = if want { current[*row].clone() } else { None };
                        *row += 1;
                        self.rows_read += 1;
                        return Ok(out);
                    }
                    let Some(entry) = entries.next() else {
                        return property_short_stream_error(
                            &self.name,
                            self.rows_read,
                            self.edge_count,
                        );
                    };
                    let start = usize::try_from(entry.offset)
                        .map_err(|_| Error::invariant("property page offset does not fit usize"))?;
                    let end = usize::try_from(
                        entry
                            .offset
                            .checked_add(entry.encoded_len)
                            .ok_or_else(|| Error::invariant("property page end exceeds u64"))?,
                    )
                    .map_err(|_| Error::invariant("property page end does not fit usize"))?;
                    let encoded = bytes.get(start..end).ok_or_else(|| Error::Corrupted {
                        path: "<edges>".into(),
                        detail: format!("property page {} is truncated", self.name),
                    })?;
                    *current = decode_property_page(encoded, entry, *compressed)?;
                    *row = 0;
                }
            }
        }
    }

    /// After the cursor consumed exactly `edge_count` values, the stream
    /// must be empty too — the streaming equivalent of the whole-stream
    /// row-count check the materialised decode performed.
    fn assert_exhausted(&mut self) -> Result<()> {
        let leftover = match &mut self.inner {
            PropertyStreamCursorInner::Arrow {
                reader,
                current,
                row,
            } => match current {
                Some(current) if *row < current.len() => true,
                _ => match reader.next() {
                    Some(batch) => {
                        batch
                            .map_err(|e| {
                                Error::invariant(format!("property IPC batch ({}): {e}", self.name))
                            })?
                            .num_rows()
                            > 0
                    }
                    None => false,
                },
            },
            PropertyStreamCursorInner::Paged {
                entries,
                current,
                row,
                ..
            } => *row < current.len() || !entries.as_slice().is_empty(),
        };
        if leftover {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "property stream {} carries more than edge_count {} rows",
                    self.name, self.edge_count
                ),
            });
        }
        Ok(())
    }
}

fn property_short_stream_error<T>(name: &str, rows_read: u64, edge_count: u64) -> Result<T> {
    Err(Error::Corrupted {
        path: "<edges>".into(),
        detail: format!("property stream {name} row count {rows_read} != edge_count {edge_count}"),
    })
}

/// Verified owned slice of one edge-SST section: `reader.section` checks the
/// xxhash, then the shared `body` handle is re-sliced so the cursor owns the
/// bytes without borrowing from the reader. `Ok(None)` when the section is
/// absent.
fn edge_section_slice(
    body: &Bytes,
    reader: &EdgeSstReader,
    kind: u16,
    name: &str,
) -> Result<Option<(Bytes, u8)>> {
    let entry = if name.is_empty() {
        reader.footer().find_kind(kind)
    } else {
        reader.footer().find(kind, name)
    };
    let Some(entry) = entry else {
        return Ok(None);
    };
    let codec = entry.codec;
    let (start, end) = (
        entry.offset as usize,
        (entry.offset + entry.length) as usize,
    );
    reader.section(kind, name)?;
    Ok(Some((body.slice(start..end), codec)))
}

/// Sorted row cursor over one edge source SST. Walks keys in `key_ids`
/// order, one decoded partner block at a time, with each property stream
/// read incrementally alongside — the per-source working set is one partner
/// block plus one IPC mini-batch per stream. Each source SST is already in
/// the caller's orientation (grouped by SstKind::EdgesFwd vs EdgesInv), so
/// `(key_id, partner_id)` pass to the writer unchanged.
///
/// Property streams (RFC-002 §3.2.7): `__overflow_json` (ad-hoc /
/// undeclared properties) plus one named stream per declared property.
/// `None` cursors mean the SST has no such stream (legacy pre-RFC-005 body,
/// or an all-null column the writer elided); every edge then yields `None`.
struct EdgeSourceCursor {
    key_ids: Bytes,
    offsets: Bytes,
    partners: Bytes,
    lsns: Bytes,
    tombstones: Option<Bytes>,
    offset_width: OffsetWidth,
    key_count: usize,
    key_idx: usize,
    current_key: [u8; 16],
    current_partners: Vec<[u8; 16]>,
    partner_idx: usize,
    /// Global edge-enumeration index of the current edge.
    edge_idx: usize,
    overflow: Option<PropertyStreamCursor>,
    declared: Vec<Option<PropertyStreamCursor>>,
}

impl EdgeSourceCursor {
    fn open(body: Bytes, declared_property_names: &[String]) -> Result<Self> {
        // `EdgeSstReader::open` validates the header/footer and cross-checks
        // the offsets/partners sections against `edge_count`; the cursor
        // then re-slices the verified sections out of the shared body.
        let reader = EdgeSstReader::open(body.clone())?;
        let key_count = reader.key_count() as usize;
        let edge_count = reader.edge_count();
        let offset_width = OffsetWidth::from_bits(reader.footer().offsets_bits)?;
        let required = |kind: u16, what: &str| -> Result<Bytes> {
            edge_section_slice(&body, &reader, kind, "")?
                .map(|(bytes, _)| bytes)
                .ok_or_else(|| Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!("edge SST missing mandatory section {what}"),
                })
        };
        let key_ids = required(SECTION_KEY_IDS, "key_ids")?;
        let offsets = required(SECTION_OFFSETS, "offsets")?;
        let partners = required(SECTION_PARTNERS, "partners")?;
        let lsns = required(SECTION_PER_EDGE_LSN, "per_edge_lsn")?;
        let tombstones = edge_section_slice(&body, &reader, SECTION_PER_EDGE_TOMBSTONES, "")?
            .map(|(bytes, _)| bytes);
        // Validate section geometry once so per-row access can index
        // directly.
        if key_ids.len() != key_count * 16 {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "key_ids section is {} bytes for {} keys",
                    key_ids.len(),
                    key_count
                ),
            });
        }
        if lsns.len() != edge_count as usize * 8 {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "per_edge_lsn section is {} bytes for {} edges",
                    lsns.len(),
                    edge_count
                ),
            });
        }
        if let Some(tombstones) = &tombstones {
            if tombstones.len() < edge_count.div_ceil(8) as usize {
                return Err(Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!(
                        "per_edge_tombstones section is {} bytes for {} edges",
                        tombstones.len(),
                        edge_count
                    ),
                });
            }
        }
        let overflow = match edge_section_slice(
            &body,
            &reader,
            SECTION_PROPERTY_STREAM,
            OVERFLOW_JSON_NAME,
        )? {
            Some((bytes, codec)) => Some(PropertyStreamCursor::open(
                OVERFLOW_JSON_NAME,
                bytes,
                codec,
                edge_count,
            )?),
            None => None,
        };
        let mut declared: Vec<Option<PropertyStreamCursor>> =
            Vec::with_capacity(declared_property_names.len());
        for name in declared_property_names {
            declared.push(
                match edge_section_slice(&body, &reader, SECTION_PROPERTY_STREAM, name)? {
                    Some((bytes, codec)) => {
                        Some(PropertyStreamCursor::open(name, bytes, codec, edge_count)?)
                    }
                    None => None,
                },
            );
        }
        let mut cursor = Self {
            key_ids,
            offsets,
            partners,
            lsns,
            tombstones,
            offset_width,
            key_count,
            key_idx: 0,
            current_key: [0u8; 16],
            current_partners: Vec::new(),
            partner_idx: 0,
            edge_idx: 0,
            overflow,
            declared,
        };
        cursor.load_key()?;
        Ok(cursor)
    }

    /// Decode the key + partner block at `key_idx` (no-op past the end).
    fn load_key(&mut self) -> Result<()> {
        if self.key_idx >= self.key_count {
            self.current_partners.clear();
            self.partner_idx = 0;
            return Ok(());
        }
        self.current_key = self.key_ids[self.key_idx * 16..(self.key_idx + 1) * 16]
            .try_into()
            .map_err(|_| Error::invariant("key_ids row length != 16"))?;
        let start = read_offset(
            &self.offsets,
            self.key_idx * self.offset_width.bytes(),
            self.offset_width,
        )? as usize;
        let (partners, _consumed) = read_partner_block(&self.partners, start)?;
        self.current_partners = partners;
        self.partner_idx = 0;
        Ok(())
    }

    /// `(key, partner, lsn)` of the current edge; `None` once exhausted.
    fn peek(&self) -> Option<([u8; 16], [u8; 16], u64)> {
        if self.partner_idx >= self.current_partners.len() {
            return None;
        }
        let off = self.edge_idx * 8;
        let lsn = u64::from_le_bytes(self.lsns[off..off + 8].try_into().unwrap());
        Some((
            self.current_key,
            self.current_partners[self.partner_idx],
            lsn,
        ))
    }

    /// Consume the current edge, advancing every property stream in
    /// lockstep. Returns the full record when `want_props` (the winner);
    /// a shadowed loser passes `false` and skips the string materialisation.
    fn pop(&mut self, want_props: bool) -> Result<Option<EdgeRecord>> {
        let Some((key_id, partner_id, lsn)) = self.peek() else {
            return Err(Error::invariant("edge merge cursor popped past its end"));
        };
        let tombstone = match &self.tombstones {
            Some(bits) => (bits[self.edge_idx / 8] >> (self.edge_idx % 8)) & 1 == 1,
            None => false,
        };
        let overflow_json = match &mut self.overflow {
            Some(cursor) => cursor.next(want_props)?,
            None => None,
        };
        let mut declared_properties: Vec<Option<String>> = Vec::with_capacity(self.declared.len());
        for stream in &mut self.declared {
            declared_properties.push(match stream {
                Some(cursor) => cursor.next(want_props)?,
                None => None,
            });
        }
        self.edge_idx += 1;
        self.partner_idx += 1;
        if self.partner_idx >= self.current_partners.len() {
            self.key_idx += 1;
            self.load_key()?;
            if self.key_idx >= self.key_count {
                // Fully drained: every property stream must be too.
                for stream in self
                    .overflow
                    .iter_mut()
                    .chain(self.declared.iter_mut().flatten())
                {
                    stream.assert_exhausted()?;
                }
            }
        }
        Ok(want_props.then_some(EdgeRecord {
            key_id,
            partner_id,
            lsn,
            tombstone,
            declared_properties,
            overflow_json,
        }))
    }
}

/// Heap key for the edge k-way merge: `(key, partner)` ascending, then LSN
/// **descending** (first popped per pair is its winner), then source order —
/// mirroring the materialised merge's stable sort tie-breaks.
#[derive(PartialEq, Eq)]
struct EdgeHeapEntry {
    key_id: [u8; 16],
    partner_id: [u8; 16],
    lsn: u64,
    src: usize,
}

impl Ord for EdgeHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key_id
            .cmp(&other.key_id)
            .then(self.partner_id.cmp(&other.partner_id))
            .then(other.lsn.cmp(&self.lsn))
            .then(self.src.cmp(&other.src))
    }
}

impl PartialOrd for EdgeHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// K-way streaming merge of one edge bucket. Per `(key, partner)` the
/// highest-LSN observation wins (source order breaks exact ties); a winning
/// tombstone is dropped when `gc_tombstones` — same reasoning as the node
/// path: when this merge is authoritative for the `(edge_type, direction)`
/// bucket (its output is the bucket's deepest level) the tombstone shadows
/// nothing. Otherwise a deeper un-merged level may still hold a row the
/// tombstone masks, so it is kept. Old readers see the delete through the
/// retained source bodies. Winners feed the row-at-a-time [`EdgeSstWriter`]
/// directly.
fn merge_edge_sources(
    bodies: Vec<Bytes>,
    edge_type: &str,
    edge_def: Option<&EdgeTypeDef>,
    declared_property_names: &[String],
    direction: EdgeDirection,
    gc_tombstones: bool,
) -> Result<EdgeSstBuild> {
    let mut cursors: Vec<EdgeSourceCursor> = Vec::with_capacity(bodies.len());
    let mut total_keys: u64 = 0;
    for body in bodies {
        let cursor = EdgeSourceCursor::open(body, declared_property_names)?;
        total_keys = total_keys.saturating_add(cursor.key_count as u64);
        cursors.push(cursor);
    }

    let (src_label, dst_label) = match edge_def {
        Some(def) => (def.src_label.clone(), def.dst_label.clone()),
        None => ("_".to_string(), "_".to_string()),
    };
    let mut options = EdgeSstWriterOptions::new(direction, edge_type, src_label, dst_label);
    // Pre-dedup upper bound (see the node merge for why); sizes the bloom
    // slightly conservatively. The partner skew threshold is fixed so exact
    // probes retain a corpus-size-independent bound.
    options.expected_keys = total_keys.max(1);
    if let Some(def) = edge_def {
        options.declared_properties = def.properties.iter().map(|p| p.name.clone()).collect();
    }
    let mut writer = EdgeSstWriter::new(options);

    let mut heap: BinaryHeap<Reverse<EdgeHeapEntry>> = BinaryHeap::with_capacity(cursors.len());
    for (src, cursor) in cursors.iter().enumerate() {
        if let Some((key_id, partner_id, lsn)) = cursor.peek() {
            heap.push(Reverse(EdgeHeapEntry {
                key_id,
                partner_id,
                lsn,
                src,
            }));
        }
    }

    let mut last: Option<([u8; 16], [u8; 16])> = None;
    while let Some(Reverse(entry)) = heap.pop() {
        let cursor = &mut cursors[entry.src];
        let pair = (entry.key_id, entry.partner_id);
        if last != Some(pair) {
            last = Some(pair);
            let record = cursor
                .pop(true)?
                .ok_or_else(|| Error::invariant("edge merge winner yielded no record"))?;
            if !(gc_tombstones && record.tombstone) {
                writer.append(record)?;
            }
        } else {
            // Shadowed duplicate: advance without materialising strings.
            cursor.pop(false)?;
        }
        if let Some((key_id, partner_id, lsn)) = cursor.peek() {
            heap.push(Reverse(EdgeHeapEntry {
                key_id,
                partner_id,
                lsn,
                src: entry.src,
            }));
        }
    }

    writer.finish_with_point_index()
}

// ── PUT helpers (L1 variants) ───────────────────────────────────────────

async fn put_node_sst_leveled(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    label: &str,
    id: Uuid,
    // Sidecar/stat harvest the streaming merge collected off the winner
    // stream (unique + equality maps keyed by the sidecar def — for the
    // id-primary "" bucket that's `union_indexed_props(schema)`, mirroring
    // flush; for legacy per-label buckets the label's own def — plus the
    // label-index postings and the RFC-025 per-(label, property) stats).
    sidecars: NodeSidecarHarvest,
    finish: NodeSstFinish,
) -> Result<(SstDescriptor, bool)> {
    let level = SstLevel(out_level);
    let file_name = format!(
        "{}-{}-{}.parquet",
        uuid_path_id(&id),
        SstKind::Nodes.path_tag(),
        label
    );
    let object_path = paths.sst_object(level.as_u32(), &file_name);
    let relative_path = relative_sst_path(level.as_u32(), &file_name);

    let body = finish.body;
    let body_len = body.len() as u64;
    crate::flush::put_object(store.clone(), &object_path, body).await?;

    let (bloom_descriptor, wrote_bloom) = put_bloom_sidecar(
        store.clone(),
        paths,
        level.as_u32(),
        &id,
        SstKind::Nodes.path_tag(),
        label,
        finish.bloom,
    )
    .await?;

    // Re-emit unique-property side-cars for the merged L1 SST so the
    // reader's `lookup_node_by_property` keeps the O(log N) probe
    // path after compaction. Without this, every compaction silently
    // demotes affected queries back to the legacy full label scan
    // (P4.19 only emitted sidecars on flush).
    let (unique_property_indices, index_sidecars) =
        sidecars.unique.finish(paths, level.as_u32(), &id, label)?;
    let mut index_sidecars: Vec<(Path, crate::flush::SidecarPayload)> = index_sidecars;
    // Re-emit equality-index posting-list sidecars too, harvested from the
    // already-reconciled winner stream (tombstones dropped, highest-lsn per
    // id), so the L1 sidecar supersedes all the L0 partials.
    let (equality_property_indices, equality_sidecars) =
        sidecars
            .equality
            .finish(paths, level.as_u32(), &id, label)?;
    index_sidecars.extend(equality_sidecars);
    // Rebuild the label-index sidecar from the reconciled rows. id-primary
    // buckets (scope == "") carry per-row label sets, so this re-emits the
    // `LabelId -> [NodeId]` postings (with per-label counts) the cost model
    // needs; without it, every compaction would silently reset per-label
    // `node_count` to 0 and the optimizer would prune non-empty labels
    // again. Legacy per-label buckets have empty label sets and yield
    // `None` here, falling back to `scope`-based counting downstream.
    let (label_index, label_sidecar) = sidecars.label_index.finish(paths, level.as_u32(), &id)?;
    if let Some((path, body)) = label_sidecar {
        index_sidecars.push((path, body));
    }
    let (mut node_locator, locator_sidecar) = crate::flush::prepare_node_locator_upload_sidecar(
        paths,
        level.as_u32(),
        &id,
        sidecars.node_locator_upload,
    )?;
    index_sidecars.push(locator_sidecar);
    let (property_pages, property_sidecar) =
        crate::flush::prepare_node_property_pages_upload_sidecar(
            paths,
            level.as_u32(),
            &id,
            &id,
            sidecars.property_pages_upload,
        )?;
    node_locator.property_pages = Some(property_pages);
    index_sidecars.push(property_sidecar);
    for (path, body) in index_sidecars {
        crate::flush::put_sidecar_payload(store.clone(), &path, body).await?;
    }

    let stats = finish.stats;
    let descriptor = SstDescriptor {
        id,
        kind: SstKind::Nodes,
        scope: label.to_string(),
        level,
        path: relative_path,
        size_bytes: body_len,
        row_count: stats.row_count,
        created_at: Utc::now(),
        min_key: stats.min_node_id,
        max_key: stats.max_node_id,
        min_lsn: stats.min_lsn,
        max_lsn: stats.max_lsn,
        schema_version_min: stats.schema_version_min,
        schema_version_max: stats.schema_version_max,
        property_stats: stats.property_stats,
        kind_specific: KindSpecificStats::Nodes {
            tombstone_count: stats.tombstone_count,
        },
        bloom: bloom_descriptor,
        unique_property_indices,
        equality_property_indices,
        composite_equality_indices: Vec::new(),
        label_index,
        node_locator: Some(node_locator),
        // Per-(label, property) stats recomputed off the winner stream so
        // they survive L0->L1 the same way the label index does (RFC 025).
        per_label_property_stats: sidecars.per_label_property_stats,
    };
    Ok((descriptor, wrote_bloom))
}

/// Streaming member collector for one registered vector index.
///
/// Winner rows arrive in strict NodeId order and are written immediately to
/// the external V5 spool. Only one decoded vector and its filter values are
/// live here; clustering/partitioning later uses bounded multi-pass scratch
/// I/O. The collector is instantiated only for authoritative merges: a
/// partial winner stream is a strict subset of the corpus and must never
/// replace the current full generation.
#[cfg(feature = "vector-index")]
struct VectorMemberCollector {
    desc: VectorIndexDescriptor,
    /// The index label resolved to its raw dictionary id (if interned).
    label_id: Option<u32>,
    /// Complete String/Bool properties embedded as page-local typed bitmaps.
    filter_properties: Vec<String>,
    builder: crate::sst::vector::v5::external::VectorV5ExternalCollector,
}

#[cfg(feature = "vector-index")]
impl VectorMemberCollector {
    fn new(
        desc: VectorIndexDescriptor,
        label_dict: &LabelDictionary,
        schema: &Schema,
        memory_budget_bytes: usize,
    ) -> Result<Self> {
        let mut filter_properties: Vec<String> = schema
            .label(&desc.label)
            .into_iter()
            .flat_map(|label| &label.properties)
            .filter(|property| {
                (property.indexed || property.unique)
                    && matches!(
                        property.data_type,
                        DataType::Bool | DataType::Utf8 | DataType::LargeUtf8
                    )
            })
            .map(|property| property.name.clone())
            .collect();
        filter_properties.sort();
        filter_properties.dedup();
        let mut config = crate::sst::vector::v5::external::VectorV5ExternalBuildConfig::from_env()?;
        config.memory_budget_bytes = memory_budget_bytes;
        Ok(Self {
            label_id: label_dict.id(&desc.label).map(|lid| lid.0),
            desc,
            filter_properties,
            builder: crate::sst::vector::v5::external::VectorV5ExternalCollector::new(config)?,
        })
    }

    fn observe(&mut self, id: [u8; 16], rec: &NodeWriteRecord, bucket_scope: &str) -> Result<()> {
        // id-primary rows carry an authoritative label set; legacy rows
        // (empty set) fall back to the bucket scope as their label.
        let carries_label = match self.label_id {
            Some(lid) => {
                rec.labels.contains(&lid)
                    || (rec.labels.is_empty() && bucket_scope == self.desc.label)
            }
            None => rec.labels.is_empty() && bucket_scope == self.desc.label,
        };
        if !carries_label {
            return Ok(());
        }
        let Some(val) = rec.properties.get(&self.desc.property) else {
            return Ok(());
        };

        let decoded;
        let vector: &[f32] = match val {
            Value::Vec(vector) => vector,
            Value::VecI8 { codes, scale } => {
                decoded = codes
                    .iter()
                    .map(|&code| code as f32 * *scale)
                    .collect::<Vec<_>>();
                &decoded
            }
            _ => return Ok(()),
        };

        // Including every configured property with Null for absence is what
        // lets V5 distinguish "supported but no row matches" from an
        // unsupported residual predicate without a global postings map.
        let filters = self
            .filter_properties
            .iter()
            .map(|property| {
                (
                    property.clone(),
                    rec.properties.get(property).cloned().unwrap_or(Value::Null),
                )
            })
            .collect::<BTreeMap<_, _>>();
        self.builder.push(id, vector, &filters)?;
        Ok(())
    }

    #[cfg(test)]
    fn from_test_members(
        desc: VectorIndexDescriptor,
        members: Vec<([u8; 16], Vec<f32>)>,
    ) -> Result<Self> {
        let mut collector = Self {
            desc,
            label_id: None,
            filter_properties: Vec::new(),
            builder: crate::sst::vector::v5::external::VectorV5ExternalCollector::new(
                crate::sst::vector::v5::external::VectorV5ExternalBuildConfig::default(),
            )?,
        };
        for (id, vector) in members {
            collector.builder.push(id, &vector, &BTreeMap::new())?;
        }
        Ok(collector)
    }
}

/// Streaming member collector for one registered full-text index.
///
/// Documents feed a bounded occurrence buffer. Sorted runs, postings,
/// dictionary blocks and the finished object live on scratch disk rather than
/// in a corpus-sized `BTreeMap`.
#[cfg(feature = "text-index")]
struct TextMemberCollector {
    desc: crate::manifest::TextIndexDescriptor,
    label_id: Option<u32>,
    builder: crate::sst::text::TextIndexExternalBuilder,
}

#[cfg(feature = "text-index")]
impl TextMemberCollector {
    fn new(
        desc: crate::manifest::TextIndexDescriptor,
        label_dict: &LabelDictionary,
        memory_budget_bytes: usize,
    ) -> Result<Self> {
        let mut options = crate::sst::text::ExternalTextIndexBuildOptions::from_env()?;
        options.memory_budget_bytes = memory_budget_bytes;
        Ok(Self {
            label_id: label_dict.id(&desc.label).map(|lid| lid.0),
            desc,
            builder: crate::sst::text::TextIndexExternalBuilder::with_options(options)?,
        })
    }

    fn observe(&mut self, id: [u8; 16], rec: &NodeWriteRecord, bucket_scope: &str) -> Result<()> {
        // Same legacy-row fallback as the vector collector above.
        let carries_label = match self.label_id {
            Some(lid) => {
                rec.labels.contains(&lid)
                    || (rec.labels.is_empty() && bucket_scope == self.desc.label)
            }
            None => rec.labels.is_empty() && bucket_scope == self.desc.label,
        };
        if !carries_label {
            return Ok(());
        }
        // Concatenate the indexed properties' string values into one document.
        let mut parts: Vec<&str> = Vec::new();
        for prop in &self.desc.properties {
            if let Some(Value::Str(s)) = rec.properties.get(prop) {
                parts.push(s.as_str());
            }
        }
        if parts.is_empty() {
            return Ok(()); // not a member of this index's corpus
        }
        self.builder.push((id, parts.join(" ")))?;
        Ok(())
    }
}

/// RFC-030 (`vector-index`): for every registered index, build a fresh
/// Vamana `VectorGraph` SST from the members the winner stream collected
/// and return the new descriptors **plus** the ids of any prior VectorGraph
/// SSTs for the same index (which must be removed — a graph is not
/// row-mergeable, so compaction rebuilds rather than merges). Indexes whose
/// authoritative corpus has fewer than two live embeddings, it yields no new
/// body and removes any prior SST; a durable generation marker prevents
/// repeatedly rewriting the node level for that legitimate empty result.
///
/// The authority gate lives at the collection site (`prepare_leveled`): on
/// a partial (non-authoritative) merge no members are collected, this
/// receives an empty list, and the existing `.vg` is left untouched — the
/// freshness gate (`index_outrun_by_nodes`) detects the now-newer Nodes SST
/// and falls back to the exact flat scan until an authoritative merge
/// rebuilds the index.
#[cfg(feature = "vector-index")]
fn vector_v5_build_options() -> crate::sst::vector::v5::VectorV5BuildOptions {
    let defaults = crate::sst::vector::v5::VectorV5BuildOptions::default();
    crate::sst::vector::v5::VectorV5BuildOptions {
        target_rows_per_page: std::env::var("NAMIDB_VECTOR_PAGE_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(defaults.target_rows_per_page),
        branch_factor: std::env::var("NAMIDB_VECTOR_BRANCH_FACTOR")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(defaults.branch_factor),
        compression_level: std::env::var("NAMIDB_VECTOR_PAGE_ZSTD_LEVEL")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(defaults.compression_level),
    }
}

#[cfg(feature = "vector-index")]
async fn build_vector_indexes_from_members(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    corpus_max_lsn: u64,
    collected: VectorIndexMembers,
    old_vector_by_scope: &BTreeMap<String, Vec<&SstDescriptor>>,
) -> Result<(Vec<SstDescriptor>, Vec<Uuid>, Vec<String>)> {
    let mut new_descs = Vec::new();
    let mut removed = Vec::new();
    let mut attempted = Vec::new();

    for collector in collected {
        let VectorMemberCollector {
            desc,
            label_id: _,
            filter_properties: _,
            builder,
        } = collector;
        // Reaching the deterministic body builder is an authoritative attempt
        // for this exact catalog signature + node generation. Persist that
        // fact even when validation rejects the body: otherwise an idle lone
        // L1 is rewritten on every maintenance tick without any input change.
        // Physical availability is still represented solely by a VectorGraph
        // SST descriptor; this marker is never consulted by reads.
        attempted.push(desc.name.clone());
        // Skip-and-warn on a per-index deterministic build error rather than
        // wedging compaction permanently. External partitioning is CPU +
        // scratch I/O and remains on the blocking pool.
        let build_desc = desc.clone();
        let input_rows = builder.input_rows();
        let options = vector_v5_build_options();
        let built = match run_cpu(move || builder.finish(&build_desc, options)).await? {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(index = %desc.name, error = %e, "skipping vector index build");
                // A body from an older catalog/generation must not remain
                // physically discoverable after this authoritative generation
                // was rejected. Removing only its manifest descriptor makes
                // reads choose the exact flat fallback; the attempt marker
                // suppresses unchanged maintenance churn.
                if let Some(old) = old_vector_by_scope.get(&desc.name) {
                    removed.extend(old.iter().map(|d| d.id));
                }
                continue;
            }
        };
        // `Ok(None)` is a successful authoritative build of a corpus too
        // small for a graph. Remove any older graph; the durable attempt marker
        // prevents a rewrite loop while reads use the exact flat fallback.
        let Some(artifact) = built else {
            if let Some(old) = old_vector_by_scope.get(&desc.name) {
                removed.extend(old.iter().map(|d| d.id));
            }
            continue;
        };
        let crate::sst::vector::v5::external::VectorV5ExternalArtifact {
            file,
            len: body_len,
            stats,
            metrics,
        } = artifact;
        tracing::info!(
            index = %desc.name,
            input_rows,
            point_count = stats.point_count,
            pages = metrics.page_count,
            partitions = metrics.partition_count,
            peak_build_bytes = metrics.peak_logical_memory_bytes,
            resident_metadata_bytes = metrics.resident_metadata_bytes,
            scratch_bytes = metrics.scratch_bytes_written,
            artifact_bytes = body_len,
            format = "NAMIVG05",
            "built bounded-memory vector search index"
        );

        let id = Uuid::now_v7();
        let level = SstLevel(out_level);
        let file_name = format!(
            "{}-{}-{}.vg",
            uuid_path_id(&id),
            SstKind::VectorGraph.path_tag(),
            desc.name
        );
        let object_path = paths.sst_object(level.as_u32(), &file_name);
        let relative_path = relative_sst_path(level.as_u32(), &file_name);
        crate::spooled_object::put_spooled_object(
            store.clone(),
            &object_path,
            crate::spooled_object::SpooledObject::from_file(file, body_len),
        )
        .await?;

        let descriptor = SstDescriptor {
            id,
            kind: SstKind::VectorGraph,
            scope: desc.name.clone(),
            level,
            path: relative_path,
            size_bytes: body_len,
            row_count: stats.point_count,
            created_at: Utc::now(),
            // For index SSTs the generic key range is the exact NodeId member
            // range (rather than the old 00..FF sentinel). The freshness gate
            // uses it to prove that a newer, unrelated-label Nodes SST cannot
            // contain a relabel/delete of an indexed member.
            min_key: stats.min_node_id,
            max_key: stats.max_node_id,
            min_lsn: 0,
            // Stamp the indexed corpus's high-water LSN so a later read can tell
            // whether a newer Nodes SST has outrun this `.vg` (freshness gate).
            max_lsn: corpus_max_lsn,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: vec![],
            kind_specific: KindSpecificStats::VectorGraph {
                dim: stats.dim,
                metric: stats.metric,
                point_count: stats.point_count,
                r: stats.r,
                l_build: stats.l_build,
                alpha: stats.alpha,
                entry_medoid: stats.entry_medoid,
            },
            bloom: None,
            unique_property_indices: vec![],
            equality_property_indices: vec![],
            composite_equality_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: vec![],
        };
        new_descs.push(descriptor);

        // Rebuild-not-merge: drop prior VectorGraph SSTs for this index.
        if let Some(old) = old_vector_by_scope.get(&desc.name) {
            removed.extend(old.iter().map(|d| d.id));
        }
    }

    Ok((new_descs, removed, attempted))
}

/// (`text-index`): for every registered full-text index, build a fresh
/// `TextIndex` SST from the documents the winner stream collected and return
/// the new descriptors **plus** the ids of any prior TextIndex SSTs for the
/// same index (rebuild-not-merge, like the vector hook). A document is the
/// space-joined string value of the index's properties; nodes carrying none
/// of them are not part of the corpus.
///
/// Like [`build_vector_indexes_from_members`], the authority gate lives at
/// the collection site: on a partial merge no members are collected, the
/// prior `.ft` is kept, and the freshness gate falls back to the flat scan
/// rather than truncating the index to the shallow subset. An authoritative
/// empty corpus does remove the prior body and records a completed generation.
#[cfg(feature = "text-index")]
async fn build_text_indexes_from_members(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    corpus_max_lsn: u64,
    collected: TextIndexMembers,
    old_text_by_scope: &BTreeMap<String, Vec<&SstDescriptor>>,
) -> Result<(Vec<SstDescriptor>, Vec<Uuid>, Vec<String>)> {
    let mut new_descs = Vec::new();
    let mut removed = Vec::new();
    let mut attempted = Vec::new();

    for collector in collected {
        let TextMemberCollector {
            desc,
            label_id: _,
            builder,
        } = collector;
        // Same generation-attempt contract as vector builds. A deterministic
        // encoder/build invariant must not make an unchanged L1 rewrite
        // forever; no TextIndex descriptor means reads remain on the exact
        // fallback. Runtime join failures and object-store errors still abort
        // compaction and therefore publish no marker.
        attempted.push(desc.name.clone());
        // External occurrence sorting and posting assembly remain on the
        // blocking pool. Their owned buffers are bounded by
        // NAMIDB_INDEX_BUILD_MEMORY_BYTES.
        let built = match run_cpu(move || builder.finish()).await? {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(
                    index = %desc.name,
                    error = %error,
                    "skipping text index build"
                );
                if let Some(old) = old_text_by_scope.get(&desc.name) {
                    removed.extend(old.iter().map(|d| d.id));
                }
                continue;
            }
        };
        let Some(artifact) = built else {
            if let Some(old) = old_text_by_scope.get(&desc.name) {
                removed.extend(old.iter().map(|d| d.id));
            }
            continue;
        };
        let (file, body_len, stats, metrics) = artifact.into_parts();
        tracing::info!(
            index = %desc.name,
            documents = stats.doc_count,
            terms = stats.term_count,
            max_buffer_bytes = metrics.max_buffer_bytes,
            initial_runs = metrics.initial_run_count,
            run_merges = metrics.run_merge_count,
            scratch_bytes = metrics.spool_bytes_written,
            artifact_bytes = body_len,
            format = "NAMIFT03",
            "built bounded-memory full-text index"
        );

        let id = Uuid::now_v7();
        let level = SstLevel(out_level);
        let file_name = format!(
            "{}-{}-{}.ft",
            uuid_path_id(&id),
            SstKind::TextIndex.path_tag(),
            desc.name
        );
        let object_path = paths.sst_object(level.as_u32(), &file_name);
        let relative_path = relative_sst_path(level.as_u32(), &file_name);
        crate::spooled_object::put_spooled_object(
            store.clone(),
            &object_path,
            crate::spooled_object::SpooledObject::from_file(file, body_len),
        )
        .await?;

        let descriptor = SstDescriptor {
            id,
            kind: SstKind::TextIndex,
            scope: desc.name.clone(),
            level,
            path: relative_path,
            size_bytes: body_len,
            row_count: stats.doc_count,
            created_at: Utc::now(),
            // TextIndex uses the same NodeId-member range contract as
            // VectorGraph; legacy descriptors retain 00..FF and therefore
            // conservatively overlap every newer Nodes SST until rebuilt.
            min_key: stats.min_node_id,
            max_key: stats.max_node_id,
            min_lsn: 0,
            // High-water LSN of the indexed corpus — lets a later read detect a
            // newer Nodes SST and fall back to the flat scan (freshness gate).
            max_lsn: corpus_max_lsn,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: vec![],
            kind_specific: KindSpecificStats::TextIndex {
                doc_count: stats.doc_count,
                term_count: stats.term_count,
                total_len: stats.total_len,
            },
            bloom: None,
            unique_property_indices: vec![],
            equality_property_indices: vec![],
            composite_equality_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: vec![],
        };
        new_descs.push(descriptor);

        // Rebuild-not-merge: drop prior TextIndex SSTs for this index.
        if let Some(old) = old_text_by_scope.get(&desc.name) {
            removed.extend(old.iter().map(|d| d.id));
        }
    }

    Ok((new_descs, removed, attempted))
}

async fn put_edge_sst_leveled(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    edge_type: &str,
    direction: EdgeDirection,
    build: EdgeSstBuild,
) -> Result<(SstDescriptor, bool)> {
    let EdgeSstBuild {
        id,
        body,
        stats,
        bloom,
        point_index,
    } = build;
    let level = SstLevel(out_level);
    let kind = match direction {
        EdgeDirection::Forward => SstKind::EdgesFwd,
        EdgeDirection::Inverse => SstKind::EdgesInv,
    };
    let point_index_present = point_index.is_some();
    let file_name = format!(
        "{}-{}-{}.{}csr",
        uuid_path_id(&id),
        direction.path_tag(),
        edge_type,
        if point_index_present { "ep." } else { "" },
    );
    let object_path = paths.sst_object(level.as_u32(), &file_name);
    let relative_path = relative_sst_path(level.as_u32(), &file_name);

    let body_len = body.size_bytes();
    crate::flush::put_sidecar_payload(
        store.clone(),
        &object_path,
        crate::flush::SidecarPayload::Spooled(body.into_spooled_object()),
    )
    .await?;
    let point_file_name = format!(
        "{}-{}-{}.epidx",
        uuid_path_id(&id),
        direction.path_tag(),
        edge_type
    );
    let point_path = paths.sst_object(level.as_u32(), &point_file_name);
    if let Some(point_body) = point_index {
        crate::flush::put_sidecar_payload(
            store.clone(),
            &point_path,
            crate::flush::SidecarPayload::EdgePoint(point_body),
        )
        .await?;
    }

    let (bloom_descriptor, wrote_bloom) = put_bloom_sidecar(
        store,
        paths,
        level.as_u32(),
        &id,
        direction.path_tag(),
        edge_type,
        bloom,
    )
    .await?;

    let descriptor = SstDescriptor {
        id,
        kind,
        scope: edge_type.to_string(),
        level,
        path: relative_path,
        size_bytes: body_len,
        row_count: stats.edge_count,
        created_at: Utc::now(),
        min_key: stats.min_key_id,
        max_key: stats.max_key_id,
        min_lsn: stats.min_lsn,
        max_lsn: stats.max_lsn,
        schema_version_min: stats.schema_version_min,
        schema_version_max: stats.schema_version_max,
        property_stats: stats.property_stats,
        kind_specific: KindSpecificStats::Edges {
            key_count: stats.key_count,
            tombstone_count: stats.tombstone_count,
            degree_histogram: Box::new(stats.degree_histogram),
        },
        bloom: bloom_descriptor,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        composite_equality_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    };
    Ok((descriptor, wrote_bloom))
}

async fn put_bloom_sidecar(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    level: u32,
    sst_id: &Uuid,
    tag: &str,
    scope: &str,
    bloom: Option<BloomFilter>,
) -> Result<(Option<BloomDescriptor>, bool)> {
    let Some(bloom) = bloom else {
        return Ok((None, false));
    };
    let file_name = format!("{}-{}-{}.bloom", uuid_path_id(sst_id), tag, scope);
    let object_path = paths.sst_object(level, &file_name);
    let relative = relative_sst_path(level, &file_name);

    let body = bloom.to_bytes();
    let descriptor = BloomDescriptor::from_body(relative, &body)?;
    crate::flush::put_object(store, &object_path, body).await?;
    Ok((Some(descriptor), true))
}

async fn get_sst_body(
    store: &dyn ObjectStore,
    paths: &NamespacePaths,
    desc: &SstDescriptor,
) -> Result<Bytes> {
    let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), desc.path);
    let path = Path::from(absolute);
    let res = store.get(&path).await?;
    let body_len = res
        .range
        .end
        .checked_sub(res.range.start)
        .ok_or_else(|| Error::invariant("compaction GET returned an inverted range"))?;
    if res.range.start != 0 || res.range.end != res.meta.size {
        return Err(Error::invariant(
            "full compaction GET unexpectedly returned a partial range",
        ));
    }
    if res.meta.size != desc.size_bytes {
        return Err(Error::Corrupted {
            path: desc.path.clone(),
            detail: format!(
                "SST object size {} disagrees with manifest size {}",
                res.meta.size, desc.size_bytes
            ),
        });
    }
    if body_len == 0 {
        return Err(Error::Corrupted {
            path: desc.path.clone(),
            detail: "SST body is empty".into(),
        });
    }
    let body_len = usize::try_from(body_len)
        .map_err(|_| Error::invariant("compaction SST body exceeds addressable memory"))?;

    let file = match res.payload {
        GetResultPayload::File(file, _) => file,
        GetResultPayload::Stream(mut stream) => {
            // Every remote SST, including individually small L0 files, is
            // streamed to the disk-backed spool. Compaction deliberately drains
            // the complete L0 backlog, so a per-object heap threshold turns
            // hundreds of "small" bodies into a multi-GiB aggregate.
            let file = crate::sst::paged_index::create_spool_file()?;
            let mut file = tokio::fs::File::from_std(file);
            while let Some(chunk) = stream.next().await {
                file.write_all(&chunk?).await?;
            }
            file.flush().await?;
            // Force delayed allocation/writeback before retaining the mmap
            // for the whole compaction pass. Otherwise a full remote L0
            // backlog can leave `sum(inputs)` as dirty page cache and surface
            // ENOSPC only after substantial merge work has already run.
            file.sync_data().await?;
            file.into_std().await
        }
    };
    if file.metadata()?.len() != body_len as u64 {
        return Err(Error::invariant(
            "compaction SST spool length disagrees with object metadata",
        ));
    }
    // SAFETY: the immutable file owns at least `body_len` bytes (checked
    // above), remains unchanged for the mapping's lifetime, and `Mmap` owns
    // the mapping after the file handle is dropped.
    let mapped = unsafe { MmapOptions::new().len(body_len).map(&file)? };
    Ok(Bytes::from_owner(mapped))
}

fn uuid_path_id(u: &Uuid) -> String {
    u.simple().to_string()
}

fn relative_sst_path(level: u32, file_name: &str) -> String {
    format!("sst/level{level}/{file_name}")
}

#[cfg(test)]
// The std-mutex env guard is deliberately held across awaits: it serializes
// whole test bodies against process-global policy env.
#[allow(clippy::await_holding_lock)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use namidb_core::{
        DataType, EdgeTypeDef, LabelDef, NamespaceId, NodeId, PropertyDef, SchemaBuilder, Value,
    };
    use object_store::memory::InMemory;
    #[cfg(feature = "text-index")]
    use object_store::PutPayload;

    use super::*;
    use crate::flush::{flush, EdgeWriteRecord, NodeWriteRecord};
    use crate::manifest::ManifestStore;
    use crate::memtable::{MemKey, Memtable};
    use crate::read::EdgeView;
    use crate::read::Snapshot;

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    async fn open_compacted_property_pages(
        store: Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        descriptor: &SstDescriptor,
    ) -> crate::sst::nodes::property_pages::NodePropertyPageReader {
        let properties = crate::manifest::node_property_pages_sidecar(descriptor)
            .expect("compacted Nodes SST must carry property pages");
        assert!(properties.is_bound_to(descriptor));
        let absolute = Path::from(format!(
            "{}/{}",
            paths.namespace_prefix().as_ref(),
            properties.path
        ));
        let meta = store.head(&absolute).await.unwrap();
        assert_eq!(meta.size, properties.size_bytes);
        let source = Arc::new(
            crate::range_cache::PinnedObjectRangeSource::from_create_only_meta(store, meta)
                .await
                .unwrap(),
        );
        let reader = crate::sst::nodes::property_pages::NodePropertyPageReader::open(
            source,
            descriptor.id,
            crate::sst::nodes::property_pages::NodePropertyPageConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(reader.node_count(), descriptor.row_count);
        assert_eq!(reader.content_xxh3(), properties.content_xxh3);
        reader
    }

    #[test]
    fn search_index_build_memory_is_one_aggregate_budget() {
        let aggregate = 256 * 1024 * 1024;
        let share = partition_index_build_memory(aggregate, 5).unwrap();
        assert_eq!(share, aggregate / 5);
        assert!(share.checked_mul(5).unwrap() <= aggregate);
    }

    #[test]
    fn search_index_build_memory_rejects_an_unfundable_catalog() {
        let error = partition_index_build_memory(2 * MIN_INDEX_BUILD_MEMORY_PER_COLLECTOR - 1, 2)
            .unwrap_err();
        assert!(error.to_string().contains("each requires at least"));
    }

    fn person_label() -> LabelDef {
        LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("age", DataType::Int32, true).unwrap(),
            ],
        }
    }

    fn knows_edge() -> EdgeTypeDef {
        EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![],
        }
    }

    fn schema() -> Schema {
        SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build()
    }

    fn sorted_node_id(b: u8) -> NodeId {
        let mut bytes = [0u8; 16];
        bytes[15] = b;
        NodeId::from_uuid(Uuid::from_bytes(bytes))
    }

    #[cfg(feature = "vector-index")]
    struct TestVectorV5Source(Bytes);

    #[cfg(feature = "vector-index")]
    #[async_trait::async_trait]
    impl crate::sst::vector::v5::VectorV5RangeSource for TestVectorV5Source {
        async fn read_range(&self, range: std::ops::Range<u64>) -> Result<Bytes> {
            let start = usize::try_from(range.start)
                .map_err(|_| Error::invariant("test V5 range start exceeds usize"))?;
            let end = usize::try_from(range.end)
                .map_err(|_| Error::invariant("test V5 range end exceeds usize"))?;
            if start > end || end > self.0.len() {
                return Err(Error::invariant("test V5 range is out of bounds"));
            }
            Ok(self.0.slice(start..end))
        }
    }

    #[cfg(feature = "text-index")]
    struct TestTextV4Source(Bytes);

    #[cfg(feature = "text-index")]
    #[async_trait::async_trait]
    impl crate::sst::search_delta::SearchVersionRangeSource for TestTextV4Source {
        async fn read_range(&self, range: std::ops::Range<u64>) -> Result<Bytes> {
            let start = usize::try_from(range.start)
                .map_err(|_| Error::invariant("test FT4 range start exceeds usize"))?;
            let end = usize::try_from(range.end)
                .map_err(|_| Error::invariant("test FT4 range end exceeds usize"))?;
            if start > end || end > self.0.len() {
                return Err(Error::invariant("test FT4 range is out of bounds"));
            }
            Ok(self.0.slice(start..end))
        }
    }

    // Shared with the ingest tests: any test that mutates or observes the
    // Search-LSM policy environment must hold the same lock, or force_base
    // leaks across concurrently running tests.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    use crate::test_support::{SearchCompactionEnvRestore, SEARCH_COMPACTION_ENV};

    #[cfg(feature = "vector-index")]
    fn physical_search_schema() -> Schema {
        SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![PropertyDef::new(
                    "embedding",
                    DataType::FloatVector { dim: 2 },
                    false,
                )
                .unwrap()],
            })
            .unwrap()
            .build()
    }

    #[cfg(feature = "vector-index")]
    fn physical_search_payload(label_id: u32, vector: Vec<f32>) -> Bytes {
        NodeWriteRecord {
            properties: BTreeMap::from([("embedding".into(), Value::Vec(vector))]),
            schema_version: 1,
            labels: vec![label_id],
        }
        .encode()
        .unwrap()
    }

    #[cfg(feature = "text-index")]
    fn physical_text_schema() -> Schema {
        SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("body", DataType::Utf8, false).unwrap(),
                    PropertyDef::new("note", DataType::Utf8, false).unwrap(),
                ],
            })
            .unwrap()
            .build()
    }

    #[cfg(feature = "text-index")]
    fn physical_text_payload(label_id: u32, body: &str, note: &str) -> Bytes {
        NodeWriteRecord {
            properties: BTreeMap::from([
                ("body".into(), Value::Str(body.into())),
                ("note".into(), Value::Str(note.into())),
            ]),
            schema_version: 1,
            labels: vec![label_id],
        }
        .encode()
        .unwrap()
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn search_generation_node_sst(max_lsn: u64) -> SstDescriptor {
        SstDescriptor {
            id: Uuid::now_v7(),
            kind: SstKind::Nodes,
            scope: String::new(),
            level: SstLevel(1),
            path: "sst/level1/search-generation.parquet".into(),
            size_bytes: 1,
            row_count: 1,
            created_at: Utc::now(),
            min_key: [0; 16],
            max_key: [0xff; 16],
            min_lsn: max_lsn,
            max_lsn,
            schema_version_min: 1,
            schema_version_max: 1,
            property_stats: Vec::new(),
            kind_specific: KindSpecificStats::Nodes { tombstone_count: 0 },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            composite_equality_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        }
    }

    fn rewrite_test_descriptor(
        id: Uuid,
        kind: SstKind,
        scope: &str,
        min_lsn: u64,
        max_lsn: u64,
    ) -> SstDescriptor {
        let kind_specific = match kind {
            SstKind::Nodes => KindSpecificStats::Nodes { tombstone_count: 0 },
            SstKind::VectorGraph => KindSpecificStats::VectorGraph {
                dim: 2,
                metric: "cosine".into(),
                point_count: 1,
                r: 8,
                l_build: 16,
                alpha: 1.2,
                entry_medoid: 0,
            },
            other => panic!("unexpected rewrite fixture kind {other:?}"),
        };
        SstDescriptor {
            id,
            kind,
            scope: scope.into(),
            level: SstLevel::L0,
            path: format!("sst/level0/{id}-rewrite-fixture"),
            size_bytes: 128,
            row_count: 1,
            created_at: Utc::now(),
            min_key: [0; 16],
            max_key: [0xff; 16],
            min_lsn,
            max_lsn,
            schema_version_min: 1,
            schema_version_max: 1,
            property_stats: Vec::new(),
            kind_specific,
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            composite_equality_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        }
    }

    fn rewrite_test_segment(
        id: Uuid,
        start: u64,
        end: u64,
        min_lsn: u64,
        max_lsn: u64,
    ) -> crate::search_lsm::SearchSegmentRef {
        crate::search_lsm::SearchSegmentRef {
            sst_id: id,
            role: crate::search_lsm::SearchSegmentRole::Delta,
            format: crate::search_lsm::SearchSegmentFormat::VectorV6,
            payload: crate::search_lsm::SearchSegmentPayload::Complete,
            event_ranges: vec![SearchEventRange::new(start, end)],
            min_lsn,
            max_lsn,
            mutation_count: 1,
            live_payload_count: 1,
            suppress_count: 0,
            content_xxh3: end.saturating_add(100),
            complete_filter_properties: Vec::new(),
            stats: crate::search_lsm::SearchSegmentStats::Vector {
                live_count: crate::search_lsm::SearchStatValue::Delta(1),
            },
            equal_lsn_conflict_count: 0,
        }
    }

    fn active_rewrite_manifest() -> crate::manifest::Manifest {
        use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};

        let mut manifest =
            crate::manifest::Manifest::empty(crate::fence::Epoch::ZERO, Uuid::now_v7());
        manifest.schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![PropertyDef::new(
                    "embedding",
                    DataType::FloatVector { dim: 2 },
                    false,
                )
                .unwrap()],
            })
            .unwrap()
            .build();
        manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_vec".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });
        let node_a = Uuid::from_u128(0x101);
        let node_b = Uuid::from_u128(0x102);
        let segment_a = Uuid::from_u128(0x201);
        let segment_b = Uuid::from_u128(0x202);
        let barrier = Uuid::from_u128(0x301);
        manifest.ssts.extend([
            rewrite_test_descriptor(node_a, SstKind::Nodes, "", 1, 10),
            rewrite_test_descriptor(node_b, SstKind::Nodes, "", 11, 20),
            rewrite_test_descriptor(segment_a, SstKind::VectorGraph, "doc_vec", 1, 10),
            rewrite_test_descriptor(segment_b, SstKind::VectorGraph, "doc_vec", 11, 20),
        ]);
        let state = SearchLsmState {
            index_name: "doc_vec".into(),
            kind: crate::search_lsm::SearchLsmKind::Vector,
            catalog_signature: crate::search_lsm::vector_catalog_signature(
                &manifest,
                &manifest.vector_indexes[0],
            ),
            generation_id: Uuid::from_u128(0x401),
            status: SearchLsmStatus::Active,
            next_event_seq: 2,
            base_frontier: None,
            segments: vec![
                rewrite_test_segment(segment_a, 0, 1, 1, 10),
                rewrite_test_segment(segment_b, 1, 2, 11, 20),
            ],
            proven_empty_event_ranges: Vec::new(),
            coverage: vec![
                SearchCoverage {
                    node_sst_id: node_a,
                    node_sst_max_lsn: 10,
                    event_ranges: vec![SearchEventRange::new(0, 1)],
                    disposition: CoverageDisposition::Segment,
                },
                SearchCoverage {
                    node_sst_id: node_b,
                    node_sst_max_lsn: 20,
                    event_ranges: vec![SearchEventRange::new(1, 2)],
                    disposition: CoverageDisposition::Segment,
                },
            ],
            compat_barrier_sst_id: Some(barrier),
            equal_lsn_conflict_count: 0,
        };
        let barrier_body = encode_search_barrier(&state).unwrap();
        manifest.ssts.push(search_barrier_descriptor(
            &state,
            barrier,
            SstLevel::L0,
            format!("sst/level0/{barrier}-doc_vec.slb"),
            barrier_body.len() as u64,
        ));
        manifest.search_lsm.push(state);
        validate_search_lsm(&manifest).unwrap();
        manifest
    }

    fn rewrite_plan(
        manifest: &crate::manifest::Manifest,
        output: Option<SstDescriptor>,
    ) -> PreparedNodeRewrite {
        PreparedNodeRewrite {
            inputs: manifest
                .ssts
                .iter()
                .filter(|descriptor| descriptor.kind == SstKind::Nodes)
                .cloned()
                .collect(),
            output,
        }
    }

    fn append_rewrite_fixture_event(manifest: &mut crate::manifest::Manifest) -> (Uuid, Uuid) {
        let node = Uuid::from_u128(0x104);
        let segment = Uuid::from_u128(0x203);
        let barrier = Uuid::from_u128(0x302);
        let old_barrier = manifest.search_lsm[0].compat_barrier_sst_id.unwrap();
        manifest
            .ssts
            .retain(|descriptor| descriptor.id != old_barrier);
        manifest
            .ssts
            .push(rewrite_test_descriptor(node, SstKind::Nodes, "", 21, 30));
        manifest.ssts.push(rewrite_test_descriptor(
            segment,
            SstKind::VectorGraph,
            "doc_vec",
            21,
            30,
        ));
        let state = &mut manifest.search_lsm[0];
        state
            .segments
            .push(rewrite_test_segment(segment, 2, 3, 21, 30));
        state.coverage.push(SearchCoverage {
            node_sst_id: node,
            node_sst_max_lsn: 30,
            event_ranges: vec![SearchEventRange::new(2, 3)],
            disposition: CoverageDisposition::Segment,
        });
        state.next_event_seq = 3;
        state.compat_barrier_sst_id = Some(barrier);
        let body = encode_search_barrier(state).unwrap();
        manifest.ssts.push(search_barrier_descriptor(
            state,
            barrier,
            SstLevel::L0,
            format!("sst/level0/{barrier}-doc_vec.slb"),
            body.len() as u64,
        ));
        validate_search_lsm(manifest).unwrap();
        (node, segment)
    }

    #[test]
    fn node_coverage_rewrite_is_deterministic_and_empty_output_removes_coverage() {
        let manifest = active_rewrite_manifest();
        let output = rewrite_test_descriptor(Uuid::from_u128(0x103), SstKind::Nodes, "", 1, 20);
        let plan = rewrite_plan(&manifest, Some(output));
        let first = rebase_search_lsm_for_node_rewrites(
            &manifest.search_lsm,
            &manifest.search_lsm,
            std::slice::from_ref(&plan),
        )
        .unwrap();
        let second = rebase_search_lsm_for_node_rewrites(
            &manifest.search_lsm,
            &manifest.search_lsm,
            std::slice::from_ref(&plan),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first[0].next_event_seq, 2, "rewrite allocates no event");
        assert_eq!(
            first[0].segments, manifest.search_lsm[0].segments,
            "node rewrite never mutates search payload segments"
        );
        assert_eq!(
            first[0].coverage[0].event_ranges,
            vec![SearchEventRange::new(0, 2)]
        );
        assert!(matches!(
            first[0].coverage[0].disposition,
            CoverageDisposition::LogicalRewrite {
                input_coverage_digest
            } if input_coverage_digest != 0
        ));

        let empty_plan = rewrite_plan(&manifest, None);
        let empty = rebase_search_lsm_for_node_rewrites(
            &manifest.search_lsm,
            &manifest.search_lsm,
            &[empty_plan],
        )
        .unwrap();
        assert!(empty[0].coverage.is_empty());
        assert_eq!(empty[0].segments, manifest.search_lsm[0].segments);

        let mut projected = manifest.clone();
        projected
            .ssts
            .retain(|descriptor| descriptor.kind != SstKind::Nodes);
        projected.search_lsm = empty;
        validate_search_lsm(&projected).unwrap();
    }

    #[test]
    fn node_coverage_rebase_preserves_concurrent_append_and_rejects_drift() {
        let manifest = active_rewrite_manifest();
        let output = rewrite_test_descriptor(Uuid::from_u128(0x103), SstKind::Nodes, "", 1, 20);
        let plan = rewrite_plan(&manifest, Some(output));
        let mut current = manifest.search_lsm[0].clone();
        let concurrent_node = Uuid::from_u128(0x104);
        let concurrent_segment = Uuid::from_u128(0x203);
        current
            .segments
            .push(rewrite_test_segment(concurrent_segment, 2, 3, 21, 30));
        current.coverage.push(SearchCoverage {
            node_sst_id: concurrent_node,
            node_sst_max_lsn: 30,
            event_ranges: vec![SearchEventRange::new(2, 3)],
            disposition: CoverageDisposition::Segment,
        });
        current.next_event_seq = 3;
        current.compat_barrier_sst_id = Some(Uuid::from_u128(0x302));
        let rebased = rebase_search_lsm_for_node_rewrites(
            &manifest.search_lsm,
            std::slice::from_ref(&current),
            std::slice::from_ref(&plan),
        )
        .unwrap();
        assert_eq!(rebased[0].segments, current.segments);
        assert_eq!(rebased[0].next_event_seq, 3);
        assert_eq!(rebased[0].coverage.len(), 2);
        assert_eq!(rebased[0].coverage[1].node_sst_id, concurrent_node);

        let mut generation_drift = current.clone();
        generation_drift.generation_id = Uuid::now_v7();
        assert!(rebase_search_lsm_for_node_rewrites(
            &manifest.search_lsm,
            &[generation_drift],
            std::slice::from_ref(&plan),
        )
        .is_err());

        let mut ddl_drift = current.clone();
        ddl_drift.catalog_signature.push_str("-ddl");
        assert!(rebase_search_lsm_for_node_rewrites(
            &manifest.search_lsm,
            &[ddl_drift],
            std::slice::from_ref(&plan),
        )
        .is_err());

        let mut conflicting_manifest = manifest.clone();
        conflicting_manifest
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.id == plan.inputs[0].id)
            .unwrap()
            .max_lsn += 1;
        assert!(
            verify_node_rewrite_inputs(&conflicting_manifest, std::slice::from_ref(&plan)).is_err()
        );
    }

    #[tokio::test]
    async fn install_node_rewrite_rotates_bound_barrier_and_keeps_active_segments() {
        let s = store();
        let p = paths("compact-search-logical-rewrite");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let boot = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(boot.manifest.epoch);
        let fixture = active_rewrite_manifest();
        let mut seeded = boot.manifest.next_version(fence.writer_id);
        seeded.schema = fixture.schema;
        seeded.ssts = fixture.ssts;
        seeded.vector_indexes = fixture.vector_indexes;
        seeded.search_lsm = fixture.search_lsm;
        let current = ms.commit(&fence, &boot, seeded).await.unwrap();
        let old_barrier = current.manifest.search_lsm[0]
            .compat_barrier_sst_id
            .unwrap();
        let output = rewrite_test_descriptor(Uuid::from_u128(0x105), SstKind::Nodes, "", 1, 20);
        let rewrite = rewrite_plan(&current.manifest, Some(output.clone()));
        let removed_ids = rewrite
            .inputs
            .iter()
            .map(|descriptor| descriptor.id)
            .collect::<Vec<_>>();
        let prepared = PreparedCompaction {
            new_descs: vec![output.clone()],
            removed_ids,
            bloom_count: 0,
            base_version: current.manifest.version,
            base_schema: current.manifest.schema.clone(),
            base_vector_indexes: current.manifest.vector_indexes.clone(),
            base_text_indexes: current.manifest.text_indexes.clone(),
            base_search_lsm: current.manifest.search_lsm.clone(),
            node_rewrites: vec![rewrite],
            search_compactions: Vec::new(),
            search_build_states: Vec::new(),
            consolidated_base_markers: Vec::new(),
            unadoptable_search_markers: Vec::new(),
            search_lsm_activations: Vec::new(),
            replaced_search_lsm: Vec::new(),
        };
        let installed = install_prepared(&ms, &fence, &current, prepared)
            .await
            .unwrap();
        validate_search_lsm(&installed.committed.manifest).unwrap();
        let state = &installed.committed.manifest.search_lsm[0];
        assert_eq!(state.coverage.len(), 1);
        assert_eq!(state.coverage[0].node_sst_id, output.id);
        assert_eq!(state.next_event_seq, 2);
        assert_eq!(state.segments, current.manifest.search_lsm[0].segments);
        let new_barrier = state.compat_barrier_sst_id.unwrap();
        assert_ne!(new_barrier, old_barrier);
        assert!(!installed
            .committed
            .manifest
            .ssts
            .iter()
            .any(|descriptor| descriptor.id == old_barrier));
        let barrier = installed
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == new_barrier)
            .unwrap();
        let barrier_body = get_sst_body(s.as_ref(), &p, barrier).await.unwrap();
        crate::search_lsm::validate_search_barrier(state, &barrier_body).unwrap();
        assert!(matches!(
            crate::search_lsm::select_search_read_plan(
                &installed.committed.manifest,
                crate::search_lsm::SearchLsmKind::Vector,
                "doc_vec",
            ),
            crate::search_lsm::SearchReadPlan::ActiveSegments { .. }
        ));
    }

    #[tokio::test]
    async fn install_node_rewrite_rebases_over_concurrent_flush_append() {
        let s = store();
        let p = paths("compact-search-rebase-flush");
        let ms = ManifestStore::new(s, p);
        let boot = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(boot.manifest.epoch);
        let fixture = active_rewrite_manifest();
        let mut seeded = boot.manifest.next_version(fence.writer_id);
        seeded.schema = fixture.schema;
        seeded.ssts = fixture.ssts;
        seeded.vector_indexes = fixture.vector_indexes;
        seeded.search_lsm = fixture.search_lsm;
        let basis = ms.commit(&fence, &boot, seeded).await.unwrap();

        let output = rewrite_test_descriptor(Uuid::from_u128(0x105), SstKind::Nodes, "", 1, 20);
        let rewrite = rewrite_plan(&basis.manifest, Some(output.clone()));
        let prepared = PreparedCompaction {
            new_descs: vec![output.clone()],
            removed_ids: rewrite
                .inputs
                .iter()
                .map(|descriptor| descriptor.id)
                .collect(),
            bloom_count: 0,
            base_version: basis.manifest.version,
            base_schema: basis.manifest.schema.clone(),
            base_vector_indexes: basis.manifest.vector_indexes.clone(),
            base_text_indexes: basis.manifest.text_indexes.clone(),
            base_search_lsm: basis.manifest.search_lsm.clone(),
            node_rewrites: vec![rewrite],
            search_compactions: Vec::new(),
            search_build_states: Vec::new(),
            consolidated_base_markers: Vec::new(),
            unadoptable_search_markers: Vec::new(),
            search_lsm_activations: Vec::new(),
            replaced_search_lsm: Vec::new(),
        };

        let mut concurrent_manifest = basis.manifest.next_version(fence.writer_id);
        let (concurrent_node, concurrent_segment) =
            append_rewrite_fixture_event(&mut concurrent_manifest);
        let concurrent = ms
            .commit(&fence, &basis, concurrent_manifest)
            .await
            .unwrap();
        let installed = install_prepared(&ms, &fence, &concurrent, prepared)
            .await
            .unwrap();
        validate_search_lsm(&installed.committed.manifest).unwrap();
        let state = &installed.committed.manifest.search_lsm[0];
        assert_eq!(state.next_event_seq, 3);
        assert_eq!(state.segments.len(), 3);
        assert_eq!(state.segments[2].sst_id, concurrent_segment);
        assert_eq!(state.coverage.len(), 2);
        assert_eq!(state.coverage[0].node_sst_id, output.id);
        assert_eq!(state.coverage[1].node_sst_id, concurrent_node);
        assert!(installed
            .committed
            .manifest
            .ssts
            .iter()
            .any(|descriptor| descriptor.id == concurrent_segment));
        assert!(installed
            .committed
            .manifest
            .ssts
            .iter()
            .any(|descriptor| descriptor.id == concurrent_node));
    }

    #[tokio::test]
    async fn install_node_rewrite_rejects_concurrent_schema_ddl() {
        let s = store();
        let p = paths("compact-search-rebase-ddl");
        let ms = ManifestStore::new(s, p);
        let boot = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(boot.manifest.epoch);
        let fixture = active_rewrite_manifest();
        let mut seeded = boot.manifest.next_version(fence.writer_id);
        seeded.schema = fixture.schema;
        seeded.ssts = fixture.ssts;
        seeded.vector_indexes = fixture.vector_indexes;
        seeded.search_lsm = fixture.search_lsm;
        let basis = ms.commit(&fence, &boot, seeded).await.unwrap();

        let output = rewrite_test_descriptor(Uuid::from_u128(0x105), SstKind::Nodes, "", 1, 20);
        let rewrite = rewrite_plan(&basis.manifest, Some(output.clone()));
        let prepared = PreparedCompaction {
            new_descs: vec![output],
            removed_ids: rewrite
                .inputs
                .iter()
                .map(|descriptor| descriptor.id)
                .collect(),
            bloom_count: 0,
            base_version: basis.manifest.version,
            base_schema: basis.manifest.schema.clone(),
            base_vector_indexes: basis.manifest.vector_indexes.clone(),
            base_text_indexes: basis.manifest.text_indexes.clone(),
            base_search_lsm: basis.manifest.search_lsm.clone(),
            node_rewrites: vec![rewrite],
            search_compactions: Vec::new(),
            search_build_states: Vec::new(),
            consolidated_base_markers: Vec::new(),
            unadoptable_search_markers: Vec::new(),
            search_lsm_activations: Vec::new(),
            replaced_search_lsm: Vec::new(),
        };

        let mut ddl_manifest = basis.manifest.next_version(fence.writer_id);
        ddl_manifest
            .schema
            .labels
            .get_mut("Doc")
            .unwrap()
            .properties
            .push(PropertyDef::new("title", DataType::Utf8, false).unwrap());
        let ddl = ms.commit(&fence, &basis, ddl_manifest).await.unwrap();
        let error = install_prepared(&ms, &fence, &ddl, prepared)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("schema"));
        let still_current = ms.load_current().await.unwrap();
        assert_eq!(still_current.manifest.version, ddl.manifest.version);
    }

    /// Exercises the production prepare/install seam, including immutable PUT
    /// before CAS and append-only rebase while the off-lock build is pending.
    #[cfg(feature = "vector-index")]
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn physical_search_prepare_install_builds_one_base_and_preserves_append() {
        use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};

        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = SearchCompactionEnvRestore::configure();
        let store = store();
        let paths = paths("physical-search-compact-e2e");
        let manifest_store = ManifestStore::new(store.clone(), paths.clone());
        let mut current = manifest_store.bootstrap(Uuid::now_v7()).await.unwrap();
        let label_id = current.manifest.label_dict.intern("Doc").0;
        current.manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_embedding".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });
        let schema = physical_search_schema();
        let fence = WriterFence::new(current.manifest.epoch);

        for (event, vector) in [vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]]
            .into_iter()
            .enumerate()
        {
            let mut memtable = Memtable::new();
            memtable.apply(
                MemKey::Node {
                    id: sorted_node_id(event as u8 + 1),
                },
                event as u64 + 1,
                MemOp::Upsert(physical_search_payload(label_id, vector)),
            );
            current = flush(
                &manifest_store,
                &fence,
                &current,
                &memtable.freeze(),
                schema.clone(),
            )
            .await
            .unwrap()
            .committed;
        }
        let captured = current.manifest.search_lsm[0].clone();
        assert_eq!(captured.segments.len(), 3);

        let prepared = prepare_compaction(&manifest_store, &fence, &current, &schema)
            .await
            .unwrap();
        assert_eq!(prepared.search_compactions.len(), 1);
        let physical = &prepared.search_compactions[0];
        assert!(physical.metrics.peak_resident_input_bytes > 0);
        assert!(
            physical.metrics.peak_resident_input_bytes < physical.metrics.input_bytes,
            "disjoint-range Nodes SSTs must stream through the base build, \
             not co-reside: peak {} vs total {}",
            physical.metrics.peak_resident_input_bytes,
            physical.metrics.input_bytes
        );
        let selected_ids = physical.selection.selected_ids().collect::<Vec<_>>();
        let output = physical.output.as_ref().expect("non-empty V5 base");
        assert_eq!(
            output.segment.role,
            crate::search_lsm::SearchSegmentRole::Base
        );
        assert_eq!(
            output.segment.event_ranges,
            vec![SearchEventRange::new(0, 3)]
        );
        store
            .head(&search_lsm_compact::descriptor_path(
                &paths,
                &output.descriptor,
            ))
            .await
            .expect("base object must exist before manifest CAS");

        // Foreground flush lands while the base is already uploaded but still
        // invisible. Its delta and Nodes coverage must survive installation.
        let mut memtable = Memtable::new();
        memtable.apply(
            MemKey::Node {
                id: sorted_node_id(4),
            },
            4,
            MemOp::Upsert(physical_search_payload(label_id, vec![-1.0, 0.0])),
        );
        let concurrent = flush(
            &manifest_store,
            &fence,
            &current,
            &memtable.freeze(),
            schema.clone(),
        )
        .await
        .unwrap()
        .committed;
        let appended_segment = concurrent.manifest.search_lsm[0]
            .segments
            .last()
            .unwrap()
            .sst_id;

        let installed = install_prepared(&manifest_store, &fence, &concurrent, prepared)
            .await
            .unwrap();
        validate_search_lsm(&installed.committed.manifest).unwrap();
        let state = installed
            .committed
            .manifest
            .search_lsm
            .iter()
            .find(|state| state.index_name == "doc_embedding")
            .unwrap();
        assert_eq!(state.next_event_seq, 4);
        assert_eq!(state.base_frontier, Some(3));
        assert_eq!(
            state
                .segments
                .iter()
                .filter(|segment| segment.role == crate::search_lsm::SearchSegmentRole::Base)
                .count(),
            1
        );
        assert_eq!(state.segments.len(), 2);
        assert_eq!(state.segments[1].sst_id, appended_segment);
        assert!(selected_ids.iter().all(|selected| {
            !installed
                .committed
                .manifest
                .ssts
                .iter()
                .any(|descriptor| descriptor.id == *selected)
        }));

        let base_descriptor = installed
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == state.segments[0].sst_id)
            .unwrap();
        let base_body = get_sst_body(store.as_ref(), &paths, base_descriptor)
            .await
            .unwrap();
        let reader = crate::sst::vector::v5::VectorV5Reader::open(
            Arc::new(TestVectorV5Source(base_body.clone())),
            base_body.len() as u64,
        )
        .await
        .expect("V5 footer and page directory must verify");
        assert_eq!(reader.point_count(), 3);
        assert_eq!(reader.dim(), 2);

        let appended_descriptor = installed
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == appended_segment)
            .unwrap();
        store
            .head(&search_lsm_compact::descriptor_path(
                &paths,
                appended_descriptor,
            ))
            .await
            .expect("concurrent delta object remains described");
        let barrier = installed
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == state.compat_barrier_sst_id.unwrap())
            .unwrap();
        let barrier_body = get_sst_body(store.as_ref(), &paths, barrier).await.unwrap();
        crate::search_lsm::validate_search_barrier(state, &barrier_body).unwrap();
    }

    #[cfg(feature = "vector-index")]
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn physical_search_prepare_install_publishes_proven_empty_without_payload() {
        use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};

        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = SearchCompactionEnvRestore::configure();
        let store = store();
        let paths = paths("physical-search-empty-e2e");
        let manifest_store = ManifestStore::new(store.clone(), paths.clone());
        let mut current = manifest_store.bootstrap(Uuid::now_v7()).await.unwrap();
        let label_id = current.manifest.label_dict.intern("Doc").0;
        current.manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_embedding".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });
        let schema = physical_search_schema();
        let fence = WriterFence::new(current.manifest.epoch);
        let first = sorted_node_id(1);
        let second = sorted_node_id(2);

        let mut create = Memtable::new();
        create.apply(
            MemKey::Node { id: first },
            1,
            MemOp::Upsert(physical_search_payload(label_id, vec![1.0, 0.0])),
        );
        create.apply(
            MemKey::Node { id: second },
            1,
            MemOp::Upsert(physical_search_payload(label_id, vec![0.0, 1.0])),
        );
        current = flush(
            &manifest_store,
            &fence,
            &current,
            &create.freeze(),
            schema.clone(),
        )
        .await
        .unwrap()
        .committed;
        for (lsn, id) in [(2, first), (3, second)] {
            let mut remove = Memtable::new();
            remove.apply(MemKey::Node { id }, lsn, MemOp::Tombstone);
            current = flush(
                &manifest_store,
                &fence,
                &current,
                &remove.freeze(),
                schema.clone(),
            )
            .await
            .unwrap()
            .committed;
        }
        assert_eq!(current.manifest.search_lsm[0].segments.len(), 3);

        let prepared = prepare_compaction(&manifest_store, &fence, &current, &schema)
            .await
            .unwrap();
        assert_eq!(prepared.search_compactions.len(), 1);
        assert!(prepared.search_compactions[0].output.is_none());
        assert!(prepared.search_compactions[0]
            .empty_proof_digest
            .is_some_and(|digest| digest != 0));

        let installed = install_prepared(&manifest_store, &fence, &current, prepared)
            .await
            .unwrap();
        validate_search_lsm(&installed.committed.manifest).unwrap();
        let state = installed
            .committed
            .manifest
            .search_lsm
            .iter()
            .find(|state| state.index_name == "doc_embedding")
            .unwrap();
        assert!(state.segments.is_empty());
        assert_eq!(state.base_frontier, None);
        assert_eq!(
            state.proven_empty_event_ranges,
            vec![SearchEventRange::new(0, 3)]
        );
        assert!(installed
            .committed
            .manifest
            .ssts
            .iter()
            .all(|descriptor| descriptor.kind != SstKind::Nodes));
        let barrier = installed
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == state.compat_barrier_sst_id.unwrap())
            .unwrap();
        let barrier_body = get_sst_body(store.as_ref(), &paths, barrier).await.unwrap();
        crate::search_lsm::validate_search_barrier(state, &barrier_body).unwrap();
    }

    #[cfg(feature = "text-index")]
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn physical_text_delta_run_preserves_term_sums_and_source_winner_lsn() {
        use crate::manifest::TextIndexDescriptor;
        use crate::search_lsm::{SearchSegmentRole, SearchSegmentStats, SearchStatValue};
        use crate::sst::text::v4::{TextV4GlobalStats, TextV4Payload, TextV4Reader};

        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let env_restore = SearchCompactionEnvRestore::configure();
        let store = store();
        let paths = paths("physical-text-delta-run-e2e");
        let manifest_store = ManifestStore::new(store.clone(), paths.clone());
        let mut current = manifest_store.bootstrap(Uuid::now_v7()).await.unwrap();
        let label_id = current.manifest.label_dict.intern("Doc").0;
        current.manifest.text_indexes.push(TextIndexDescriptor::new(
            "doc_body".into(),
            "Doc".into(),
            vec!["body".into()],
        ));
        let schema = physical_text_schema();
        let fence = WriterFence::new(current.manifest.epoch);
        let first = sorted_node_id(1);
        let second = sorted_node_id(2);

        // Seed one document and force exactly one authoritative FT4 base.
        let mut create = Memtable::new();
        create.apply(
            MemKey::Node { id: first },
            1,
            MemOp::Upsert(physical_text_payload(
                label_id,
                "alpha desaparecido",
                "seed",
            )),
        );
        current = flush(
            &manifest_store,
            &fence,
            &current,
            &create.freeze(),
            schema.clone(),
        )
        .await
        .unwrap()
        .committed;
        let prepared_base = prepare_compaction(&manifest_store, &fence, &current, &schema)
            .await
            .unwrap();
        assert_eq!(prepared_base.search_compactions.len(), 1);
        assert_eq!(
            prepared_base.search_compactions[0]
                .output
                .as_ref()
                .unwrap()
                .segment
                .role,
            SearchSegmentRole::Base
        );
        current = install_prepared(&manifest_store, &fence, &current, prepared_base)
            .await
            .unwrap()
            .committed;

        // Routine mode compacts the next two deltas, never the retained base.
        env_restore.select_delta_runs(2);
        let mut update = Memtable::new();
        update.apply(
            MemKey::Node { id: first },
            2,
            MemOp::Upsert(physical_text_payload(label_id, "alpha gamma", "relevant")),
        );
        current = flush(
            &manifest_store,
            &fence,
            &current,
            &update.freeze(),
            schema.clone(),
        )
        .await
        .unwrap()
        .committed;
        let mut second_create = Memtable::new();
        second_create.apply(
            MemKey::Node { id: second },
            3,
            MemOp::Upsert(physical_text_payload(label_id, "beta", "new")),
        );
        current = flush(
            &manifest_store,
            &fence,
            &current,
            &second_create.freeze(),
            schema.clone(),
        )
        .await
        .unwrap()
        .committed;

        // A later Nodes event changes only an unindexed property. It is
        // ProvenEmpty for FT4, so Nodes resolves LSN 4 while the physical
        // search winner must remain the selected NAMISV01 record at LSN 2.
        let mut irrelevant = Memtable::new();
        irrelevant.apply(
            MemKey::Node { id: first },
            4,
            MemOp::Upsert(physical_text_payload(
                label_id,
                "alpha gamma",
                "irrelevant-newer-lsn",
            )),
        );
        current = flush(
            &manifest_store,
            &fence,
            &current,
            &irrelevant.freeze(),
            schema.clone(),
        )
        .await
        .unwrap()
        .committed;
        let captured = current
            .manifest
            .search_lsm
            .iter()
            .find(|state| state.index_name == "doc_body")
            .unwrap();
        assert_eq!(captured.segments.len(), 3);
        assert_eq!(
            captured.proven_empty_event_ranges,
            vec![SearchEventRange::new(3, 4)]
        );

        let prepared = prepare_compaction(&manifest_store, &fence, &current, &schema)
            .await
            .unwrap();
        assert_eq!(prepared.search_compactions.len(), 1);
        let output = prepared.search_compactions[0]
            .output
            .as_ref()
            .expect("DeltaRun must emit final winner records");
        assert_eq!(output.segment.role, SearchSegmentRole::Delta);
        assert_eq!(
            output.segment.event_ranges,
            vec![SearchEventRange::new(1, 3)]
        );
        assert_eq!(
            output.segment.stats,
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Delta(1),
                total_len: SearchStatValue::Delta(1),
                term_df_violation_count: 0,
            }
        );
        let output_id = output.segment.sst_id;
        assert_eq!(prepared.search_compactions[0].metrics.input_rows, 2);
        assert_eq!(prepared.search_compactions[0].metrics.touched_rows, 2);
        store
            .head(&search_lsm_compact::descriptor_path(
                &paths,
                &output.descriptor,
            ))
            .await
            .expect("FT4 DeltaRun object must exist before manifest CAS");

        let installed = install_prepared(&manifest_store, &fence, &current, prepared)
            .await
            .unwrap();
        validate_search_lsm(&installed.committed.manifest).unwrap();
        let state = installed
            .committed
            .manifest
            .search_lsm
            .iter()
            .find(|state| state.index_name == "doc_body")
            .unwrap();
        assert_eq!(state.segments.len(), 2);
        assert_eq!(state.segments[0].role, SearchSegmentRole::Base);
        assert_eq!(state.segments[1].sst_id, output_id);
        assert_eq!(
            state.proven_empty_event_ranges,
            vec![SearchEventRange::new(3, 4)]
        );
        let segment = &state.segments[1];
        let descriptor = installed
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == output_id)
            .unwrap();
        let body = get_sst_body(store.as_ref(), &paths, descriptor)
            .await
            .unwrap();
        let reader = TextV4Reader::open(
            Arc::new(TestTextV4Source(body.clone())),
            body.len() as u64,
            state,
            segment,
        )
        .await
        .unwrap();
        reader.verify_all().await.unwrap();
        assert_eq!(reader.term_delta_df("alpha").await.unwrap(), 0);
        assert_eq!(reader.term_delta_df("beta").await.unwrap(), 1);
        assert_eq!(reader.term_delta_df("desaparecido").await.unwrap(), -1);
        assert_eq!(reader.term_delta_df("gamma").await.unwrap(), 1);

        let winner = reader
            .version_reader()
            .point_probe(*first.as_bytes())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(winner.lsn, 2);
        let expected_payload = TextV4Payload {
            text: "alpha gamma".into(),
            filters: BTreeMap::new(),
        };
        assert_eq!(
            winner.payload_fingerprint,
            crate::sst::text::v4::text_v4_payload_fingerprint(&expected_payload).unwrap()
        );

        let empty_memtable = crate::memtable::MemtableSnapshot::empty();
        let snapshot =
            crate::read::Snapshot::new(installed.committed.clone(), &empty_memtable, store, paths);
        assert_eq!(
            snapshot.lookup_node("", first).await.unwrap().unwrap().lsn,
            4
        );

        let query = crate::text::TextQuery::from_terms(&["alpha".into()]);
        let global = TextV4GlobalStats {
            document_count: 2,
            total_document_len: 3,
            document_frequency: BTreeMap::from([("alpha".into(), 1)]),
        };
        let hits = reader
            .search_query_exact(&query, &global, 10, &[])
            .await
            .unwrap();
        assert_eq!(hits.hits.len(), 1);
        assert_eq!(hits.hits[0].node_id, *first.as_bytes());
        assert_eq!(hits.hits[0].lsn, 2);
    }

    fn node_payload(name: &str, age: Option<i32>) -> Bytes {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        if let Some(a) = age {
            props.insert("age".into(), Value::I64(a as i64));
        }
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            // Single label "Person" -> LabelId(0) on a fresh dict, carried
            // on-row so the id-primary read path resolves the node to "Person".
            labels: vec![0],
        }
        .encode()
        .unwrap()
    }

    fn edge_payload() -> Bytes {
        EdgeWriteRecord {
            properties: BTreeMap::new(),
            schema_version: 1,
        }
        .encode()
        .unwrap()
    }

    /// Bootstrap + flush twice → 2 L0 node SSTs in the same scope.
    async fn build_two_l0_node_ssts() -> (
        Arc<dyn ObjectStore>,
        NamespacePaths,
        ManifestStore,
        WriterFence,
        LoadedManifest,
    ) {
        let s = store();
        let p = paths("compact-nodes");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person" through
        // both flushes and the L0->L1 compaction (the dict is cloned forward).
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);
        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Node { id: alice },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen1 = mt1.freeze();
        let after1 = flush(&ms, &fence, &base, &frozen1, schema()).await.unwrap();

        let bob = sorted_node_id(2);
        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Node { id: bob },
            20,
            MemOp::Upsert(node_payload("Bob", None)),
        );
        let frozen2 = mt2.freeze();
        let after2 = flush(&ms, &fence, &after1.committed, &frozen2, schema())
            .await
            .unwrap();

        assert_eq!(after2.committed.manifest.ssts.len(), 2);
        (s, p, ms, fence, after2.committed)
    }

    #[tokio::test]
    async fn empty_l0_is_noop() {
        let s = store();
        let p = paths("compact-empty");
        let ms = ManifestStore::new(s, p);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let out = compact_l0_to_l1(&ms, &fence, &base, &schema())
            .await
            .unwrap();
        assert_eq!(out.source_ssts_removed, 0);
        assert_eq!(out.new_ssts_written, 0);
    }

    #[tokio::test]
    async fn single_sst_per_scope_is_noop() {
        let s = store();
        let p = paths("compact-single");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: alice },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt.freeze();
        let after = flush(&ms, &fence, &base, &frozen, schema()).await.unwrap();
        assert_eq!(after.committed.manifest.ssts.len(), 1);

        let out = compact_l0_to_l1(&ms, &fence, &after.committed, &schema())
            .await
            .unwrap();
        assert_eq!(out.source_ssts_removed, 0);
        assert_eq!(out.new_ssts_written, 0);
    }

    #[tokio::test]
    async fn merges_two_disjoint_node_ssts_into_one_l1() {
        let (s, p, ms, fence, base) = build_two_l0_node_ssts().await;

        let out = compact_l0_to_l1(&ms, &fence, &base, &schema())
            .await
            .unwrap();
        assert_eq!(out.source_ssts_removed, 2);
        assert_eq!(out.new_ssts_written, 1);

        // The new manifest has exactly the L1 SST, no L0 left for that scope.
        let manifest = &out.committed.manifest;
        assert_eq!(manifest.ssts.len(), 1);
        let only = &manifest.ssts[0];
        assert_eq!(only.level, SstLevel(1));
        assert_eq!(only.kind, SstKind::Nodes);
        assert_eq!(only.row_count, 2);
        let property_reader = open_compacted_property_pages(s.clone(), &p, only).await;
        let (projected, _) = property_reader
            .project_node_ids(
                &["name".into(), "age".into()],
                &[*sorted_node_id(1).as_bytes(), *sorted_node_id(2).as_bytes()],
            )
            .await
            .unwrap();
        assert_eq!(
            projected[0].properties["name"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::Str("Alice".into()))
        );
        assert_eq!(
            projected[0].properties["age"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::I64(30))
        );
        assert_eq!(
            projected[1].properties["name"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::Str("Bob".into()))
        );
        assert_eq!(
            projected[1].properties["age"],
            crate::sst::nodes::property_pages::PropertyCell::Absent
        );

        // Snapshot through the new manifest must still see both nodes.
        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let v_alice = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(
            v_alice.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
        let v_bob = snap.lookup_node("Person", bob).await.unwrap().unwrap();
        assert_eq!(
            v_bob.properties.get("name"),
            Some(&Value::Str("Bob".into()))
        );
    }

    #[tokio::test]
    async fn merges_overlapping_node_keys_keeping_highest_lsn() {
        let s = store();
        let p = paths("compact-overlap");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);

        // L0 SST #1: alice@v1, name=Alice
        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Node { id: alice },
            5,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen1 = mt1.freeze();
        let after1 = flush(&ms, &fence, &base, &frozen1, schema()).await.unwrap();

        // L0 SST #2: alice@v2, name=Alicia, with a later LSN — it wins.
        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Node { id: alice },
            12,
            MemOp::Upsert(node_payload("Alicia", Some(31))),
        );
        let frozen2 = mt2.freeze();
        let after2 = flush(&ms, &fence, &after1.committed, &frozen2, schema())
            .await
            .unwrap();
        assert_eq!(after2.committed.manifest.ssts.len(), 2);

        let out = compact_l0_to_l1(&ms, &fence, &after2.committed, &schema())
            .await
            .unwrap();
        assert_eq!(out.source_ssts_removed, 2);
        assert_eq!(out.new_ssts_written, 1);

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(view.lsn, 12);
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alicia".into()))
        );
    }

    #[tokio::test]
    async fn tombstone_at_higher_lsn_wins_in_compaction() {
        let s = store();
        let p = paths("compact-tomb");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);

        // L0 SST #1: alice upsert at LSN 5.
        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Node { id: alice },
            5,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen1 = mt1.freeze();
        let after1 = flush(&ms, &fence, &base, &frozen1, schema()).await.unwrap();

        // L0 SST #2: alice tombstone at LSN 9 — wins.
        let mut mt2 = Memtable::new();
        mt2.apply(MemKey::Node { id: alice }, 9, MemOp::Tombstone);
        let frozen2 = mt2.freeze();
        let after2 = flush(&ms, &fence, &after1.committed, &frozen2, schema())
            .await
            .unwrap();

        let out = compact_l0_to_l1(&ms, &fence, &after2.committed, &schema())
            .await
            .unwrap();
        assert_eq!(out.source_ssts_removed, 2);
        // RFC-027 P3: the winning op for `alice` is a tombstone, so the
        // full-bucket merge drops it entirely — the node bucket is now empty
        // and no L1 SST is written at all (the delete is fully reclaimed).
        assert_eq!(out.new_ssts_written, 0);

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        let v = snap.lookup_node("Person", alice).await.unwrap();
        assert!(v.is_none(), "the deleted node stays absent after GC");
    }

    #[tokio::test]
    async fn full_bucket_compaction_gcs_tombstone_arriving_after_l1() {
        // Steady-state GC: alice+bob land and compact to an L1, then alice is
        // deleted in a later flush. The next compaction is full-bucket (it
        // pulls the prior L1 in as a source), so alice's tombstone is GC'd
        // without resurrecting her, bob survives, and the bucket stays at one
        // L1. This is the case a pure L0->L1 merge could never reclaim.
        let s = store();
        let p = paths("compact-gc-steady");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);

        // Flush alice (L0 #1) and bob (L0 #2), then compact to one L1.
        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Node { id: alice },
            1,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let after1 = flush(&ms, &fence, &base, &mt1.freeze(), schema())
            .await
            .unwrap();
        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Node { id: bob },
            2,
            MemOp::Upsert(node_payload("Bob", Some(40))),
        );
        let after2 = flush(&ms, &fence, &after1.committed, &mt2.freeze(), schema())
            .await
            .unwrap();
        let comp1 = compact_l0_to_l1(&ms, &fence, &after2.committed, &schema())
            .await
            .unwrap();
        assert_eq!(comp1.new_ssts_written, 1, "alice+bob collapse to one L1");
        let l1_count = comp1
            .committed
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::Nodes)
            .count();
        assert_eq!(l1_count, 1);

        // Delete alice in a later flush (L0 #3).
        let mut mt3 = Memtable::new();
        mt3.apply(MemKey::Node { id: alice }, 9, MemOp::Tombstone);
        let after3 = flush(&ms, &fence, &comp1.committed, &mt3.freeze(), schema())
            .await
            .unwrap();

        // Full-bucket compaction: prior L1 + the tombstone L0 are both
        // sources, so alice's tombstone is dropped and bob is kept.
        let comp2 = compact_l0_to_l1(&ms, &fence, &after3.committed, &schema())
            .await
            .unwrap();
        assert_eq!(
            comp2.source_ssts_removed, 2,
            "the prior L1 and the tombstone L0 are both merged"
        );
        assert_eq!(comp2.new_ssts_written, 1, "bob remains in one L1");
        let node_ssts = comp2
            .committed
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::Nodes)
            .collect::<Vec<_>>();
        assert_eq!(node_ssts.len(), 1, "bucket stays at one L1");
        let tombstone_count = match &node_ssts[0].kind_specific {
            KindSpecificStats::Nodes { tombstone_count } => *tombstone_count,
            other => panic!("expected node stats, got {other:?}"),
        };
        assert_eq!(
            tombstone_count, 0,
            "the GC'd tombstone is not carried into the new L1"
        );

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(comp2.committed.clone(), &mt_view, s, p);
        assert!(
            snap.lookup_node("Person", alice).await.unwrap().is_none(),
            "alice stays deleted after GC, not resurrected"
        );
        assert!(
            snap.lookup_node("Person", bob).await.unwrap().is_some(),
            "bob survives the GC compaction"
        );
    }

    #[tokio::test]
    async fn compacts_forward_and_inverse_edges_independently() {
        let s = store();
        let p = paths("compact-edges");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        // Flush #1: alice→bob.
        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(edge_payload()),
        );
        let frozen1 = mt1.freeze();
        let after1 = flush(&ms, &fence, &base, &frozen1, schema()).await.unwrap();

        // Flush #2: alice→carol.
        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            11,
            MemOp::Upsert(edge_payload()),
        );
        let frozen2 = mt2.freeze();
        let after2 = flush(&ms, &fence, &after1.committed, &frozen2, schema())
            .await
            .unwrap();
        // 2 flushes × (fwd + inv) = 4 L0 edge SSTs.
        assert_eq!(after2.committed.manifest.ssts.len(), 4);

        let out = compact_l0_to_l1(&ms, &fence, &after2.committed, &schema())
            .await
            .unwrap();
        assert_eq!(out.source_ssts_removed, 4);
        assert_eq!(out.new_ssts_written, 2);
        let kinds: Vec<SstKind> = out.committed.manifest.ssts.iter().map(|d| d.kind).collect();
        assert!(kinds.contains(&SstKind::EdgesFwd));
        assert!(kinds.contains(&SstKind::EdgesInv));
        for d in &out.committed.manifest.ssts {
            assert_eq!(d.level, SstLevel(1));
        }

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        let out_edges = snap.out_edges("KNOWS", alice).await.unwrap();
        assert_eq!(out_edges.edges.len(), 2);
        let dsts: Vec<NodeId> = out_edges.edges.iter().map(|e| e.dst).collect();
        assert!(dsts.contains(&bob));
        assert!(dsts.contains(&carol));

        let in_bob = snap.in_edges("KNOWS", bob).await.unwrap();
        assert_eq!(in_bob.edges.len(), 1);
        assert_eq!(in_bob.edges[0].src, alice);
    }

    /// Edge-bucket tombstone GC at the deepest merge: an authoritative
    /// compaction must physically drop edge tombstones from BOTH directions,
    /// keep forward/inverse row sets mirror-consistent, and leave manifest
    /// stats that agree (count_edge_type reads them directly).
    #[tokio::test]
    async fn edge_tombstone_gc_at_the_deepest_merge_drops_rows_in_both_directions() {
        let s = store();
        let p = paths("compact-edge-tombstone-gc");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);
        let dave = sorted_node_id(4);

        let mut mt1 = Memtable::new();
        for (src, dst, lsn) in [(alice, bob, 10u64), (alice, carol, 11), (bob, carol, 12)] {
            mt1.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src,
                    dst,
                },
                lsn,
                MemOp::Upsert(edge_payload()),
            );
        }
        let after1 = flush(&ms, &fence, &base, &mt1.freeze(), schema())
            .await
            .unwrap();

        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            20,
            MemOp::Tombstone,
        );
        mt2.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: dave,
            },
            21,
            MemOp::Upsert(edge_payload()),
        );
        let after2 = flush(&ms, &fence, &after1.committed, &mt2.freeze(), schema())
            .await
            .unwrap();

        let out = compact_l0_to_l1(&ms, &fence, &after2.committed, &schema())
            .await
            .unwrap();

        let mut fwd_pairs = Vec::new();
        let mut inv_pairs = Vec::new();
        for desc in &out.committed.manifest.ssts {
            if !matches!(desc.kind, SstKind::EdgesFwd | SstKind::EdgesInv) {
                continue;
            }
            let body = get_sst_body(s.as_ref(), &p, desc).await.unwrap();
            let reader = crate::sst::edges::EdgeSstReader::open(body).unwrap();
            let rows = reader.scan_all_edges().unwrap();
            assert!(
                rows.iter().all(|row| !row.tombstone),
                "an authoritative merge must physically drop edge tombstones \
                 ({:?})",
                desc.kind
            );
            if let crate::manifest::KindSpecificStats::Edges {
                tombstone_count, ..
            } = &desc.kind_specific
            {
                assert_eq!(*tombstone_count, 0, "stats must reflect the drop");
            } else {
                panic!("edge SST must carry Edges stats");
            }
            assert_eq!(desc.row_count, 3);
            for row in rows {
                let pair = (row.key_id, row.partner_id);
                match desc.kind {
                    SstKind::EdgesFwd => fwd_pairs.push(pair),
                    _ => inv_pairs.push((pair.1, pair.0)),
                }
            }
        }
        fwd_pairs.sort();
        inv_pairs.sort();
        assert_eq!(
            fwd_pairs, inv_pairs,
            "forward and inverse row sets must mirror after the drop"
        );
        assert_eq!(fwd_pairs.len(), 3);

        let mt = Memtable::new();
        let view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &view, s.clone(), p.clone());
        let outgoing = snap.out_edges("KNOWS", alice).await.unwrap();
        let mut dsts: Vec<NodeId> = outgoing.edges.iter().map(|edge| edge.dst).collect();
        dsts.sort();
        assert_eq!(dsts, vec![carol, dave], "the tombstoned edge must be gone");
        let incoming = snap.in_edges("KNOWS", bob).await.unwrap();
        assert!(
            incoming.edges.is_empty(),
            "the inverse partner of the dropped edge must be gone too"
        );
        assert_eq!(snap.count_edge_type("KNOWS").await.unwrap(), 3);
    }

    /// The `gc_tombstones` flag is what separates the authoritative drop
    /// from the shadow-preserving merge: a non-authoritative merge must KEEP
    /// the tombstone so a deeper un-merged level cannot resurrect the edge.
    #[tokio::test]
    async fn non_authoritative_edge_merge_preserves_tombstones() {
        let s = store();
        let p = paths("compact-edge-tombstone-keep");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: sorted_node_id(1),
                dst: sorted_node_id(2),
            },
            20,
            MemOp::Tombstone,
        );
        let after = flush(&ms, &fence, &base, &mt.freeze(), schema())
            .await
            .unwrap();
        let fwd = after
            .committed
            .manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::EdgesFwd)
            .expect("tombstone-only flush still writes the shadow SST");
        let body = get_sst_body(s.as_ref(), &p, fwd).await.unwrap();

        for (gc, expect_edges, expect_tombs) in [(false, 1u64, 1u64), (true, 0, 0)] {
            let merged = merge_edge_sources(
                vec![body.clone()],
                "KNOWS",
                None,
                &[],
                crate::sst::edges::EdgeDirection::Forward,
                gc,
            )
            .unwrap();
            assert_eq!(
                (merged.stats.edge_count, merged.stats.tombstone_count),
                (expect_edges, expect_tombs),
                "gc_tombstones={gc} must {} the tombstone",
                if gc { "drop" } else { "keep" }
            );
        }
    }

    fn knows_edge_with_declared_props() -> EdgeTypeDef {
        EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![
                namidb_core::PropertyDef::new("since", DataType::Int64, true).unwrap(),
                namidb_core::PropertyDef::new("weight", DataType::Float64, true).unwrap(),
            ],
        }
    }

    fn schema_with_declared_edge() -> Schema {
        SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge_with_declared_props())
            .unwrap()
            .build()
    }

    fn edge_payload_with_props(props: BTreeMap<String, Value>) -> Bytes {
        EdgeWriteRecord {
            properties: props,
            schema_version: 1,
        }
        .encode()
        .unwrap()
    }

    #[tokio::test]
    async fn compaction_preserves_declared_edge_property_streams() {
        // RFC-002 §3.2.7: after a multi-flush compact, the merged L1
        // SST must still expose both declared property streams
        // (`since`, `weight`) AND any ad-hoc props in __overflow_json.
        let s = store();
        let p = paths("compact-edges-declared");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let sc = schema_with_declared_edge();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        let mut props_ab: BTreeMap<String, Value> = BTreeMap::new();
        props_ab.insert("since".into(), Value::I64(2020));
        props_ab.insert("weight".into(), Value::F64(0.5));
        props_ab.insert("note".into(), Value::Str("first".into()));

        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(edge_payload_with_props(props_ab.clone())),
        );
        let frozen1 = mt1.freeze();
        let after1 = flush(&ms, &fence, &base, &frozen1, sc.clone())
            .await
            .unwrap();

        let mut props_ac: BTreeMap<String, Value> = BTreeMap::new();
        props_ac.insert("since".into(), Value::Null);
        props_ac.insert("note".into(), Value::Null);

        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            11,
            MemOp::Upsert(edge_payload_with_props(props_ac.clone())),
        );
        let frozen2 = mt2.freeze();
        let after2 = flush(&ms, &fence, &after1.committed, &frozen2, sc.clone())
            .await
            .unwrap();

        let out = compact_l0_to_l1(&ms, &fence, &after2.committed, &sc)
            .await
            .unwrap();
        assert!(out.source_ssts_removed >= 2);

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        let outs = snap.out_edges("KNOWS", alice).await.unwrap();
        assert_eq!(outs.edges.len(), 2);
        let by_dst: BTreeMap<NodeId, &EdgeView> = outs.edges.iter().map(|e| (e.dst, e)).collect();
        assert_eq!(by_dst[&bob].properties, props_ab);
        assert_eq!(by_dst[&carol].properties, props_ac);
        let exact = snap
            .lookup_edge_via_sst("KNOWS", alice, carol)
            .await
            .unwrap()
            .expect("compacted exact edge");
        assert_eq!(exact.properties.get("since"), Some(&Value::Null));
        assert_eq!(exact.properties.get("note"), Some(&Value::Null));
        assert!(
            !exact.properties.contains_key("weight"),
            "absent declared value must not be reconstructed as explicit null"
        );
        assert!(out
            .committed
            .manifest
            .ssts
            .iter()
            .any(|descriptor| descriptor.kind == SstKind::EdgesFwd
                && descriptor.path.ends_with(".ep.csr")));
    }

    // ── Leveled-lite ────────────────────────────────────────────────────

    /// Flush one node op (upsert or tombstone) as its own L0 SST.
    async fn flush_node_op(
        ms: &ManifestStore,
        fence: &WriterFence,
        base: &LoadedManifest,
        id: NodeId,
        lsn: u64,
        op: MemOp,
    ) -> LoadedManifest {
        let mut mt = Memtable::new();
        mt.apply(MemKey::Node { id }, lsn, op);
        flush(ms, fence, base, &mt.freeze(), schema())
            .await
            .unwrap()
            .committed
    }

    fn node_levels(m: &LoadedManifest) -> Vec<u32> {
        let mut levels: Vec<u32> = m
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::Nodes)
            .map(|d| d.level.as_u32())
            .collect();
        levels.sort_unstable();
        levels
    }

    /// `plan_bucket_merge` reduced to `(target_level, is_deepest, n_inputs)`,
    /// so the borrowed `BucketPlan` does not escape the call.
    fn plan_levels(owned: &[SstDescriptor], base: u64, ratio: u64) -> Option<(u32, bool, usize)> {
        let refs: Vec<&SstDescriptor> = owned.iter().collect();
        plan_bucket_merge(&refs, base, ratio)
            .map(|p| (p.target_level, p.is_deepest, p.inputs.len()))
    }

    #[tokio::test]
    async fn plan_bucket_merge_cascades_and_gates_gc_on_the_deepest_level() {
        // Build one real descriptor to clone; `plan_bucket_merge` only reads
        // `level` and `size_bytes`, so cloning + mutating those is enough to
        // drive every branch deterministically without env or flush.
        let (.., base) = build_two_l0_node_ssts().await;
        let template = base.manifest.ssts[0].clone();
        let mk = |level: u32, size: u64| {
            let mut d = template.clone();
            d.id = Uuid::now_v7();
            d.level = SstLevel(level);
            d.size_bytes = size;
            d
        };
        // budgets: L1 = 100, L2 = 1000, L3 = 10_000.
        let (bb, r) = (100u64, 10u64);

        // 1. L0 + L1 within L1's budget, no deeper level → land in L1, GC.
        assert_eq!(
            plan_levels(&[mk(0, 10), mk(1, 50)], bb, r),
            Some((1, true, 2))
        );

        // 2. L0 + L1 over L1's budget, deepest present = 1 → spill to L2, GC.
        assert_eq!(
            plan_levels(&[mk(0, 80), mk(1, 80)], bb, r).map(|(l, d, _)| (l, d)),
            Some((2, true))
        );

        // 3. L0 + L1 within budget but a deeper L3 exists → land in L1, NO GC
        //    (the tombstone could still be shadowing a row down in L3), and L3
        //    is left untouched (only L0 + L1 are merged).
        assert_eq!(
            plan_levels(&[mk(0, 10), mk(1, 20), mk(3, 5000)], bb, r),
            Some((1, false, 2))
        );

        // 4. L0 + L1 over L1's budget, cascades through L2 which fits → output
        //    L2 (the deepest), GC; all three levels merged.
        assert_eq!(
            plan_levels(&[mk(0, 60), mk(1, 60), mk(2, 200)], bb, r),
            Some((2, true, 3))
        );

        // 5. A single L0 with nothing else → nothing worth merging.
        assert_eq!(plan_levels(&[mk(0, 10)], bb, r), None);

        // 6. Fresh bucket, two big L0s over L1's budget, no leveled data → land
        //    in L1 (no spilling past a non-existent deeper level); cascades on
        //    a later sweep.
        assert_eq!(
            plan_levels(&[mk(0, 80), mk(0, 80)], bb, r).map(|(l, d, _)| (l, d)),
            Some((1, true))
        );

        // 7. A lone over-budget L1 with no L0 → not worth a pure rewrite.
        assert_eq!(plan_levels(&[mk(1, 5000)], bb, r), None);
    }

    #[tokio::test]
    async fn lone_legacy_l1_is_rewritten_once_for_sidecar_migration() {
        let s = store();
        let p = paths("compact-migrate-node-locator");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let mut indexed_person = person_label();
        indexed_person.properties[0].indexed = true;
        let sc = SchemaBuilder::new()
            .label(indexed_person)
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                id: sorted_node_id(7),
            },
            1,
            MemOp::Upsert(node_payload("legacy", None)),
        );
        let current = flush(&ms, &fence, &base, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;

        // The coverage marker alone is sufficient to schedule a rebuild.
        // Pre-fix ScalarV1 sidecars could advertise a property while silently
        // omitting a value whose label declared the same name with another
        // type; readers must not trust their negative answers.
        let mut incomplete_coverage = current.clone();
        let descriptor = incomplete_coverage
            .manifest
            .ssts
            .iter_mut()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        descriptor.level = SstLevel(1);
        for index in &mut descriptor.equality_property_indices {
            index.mixed_type_complete = false;
        }
        let incomplete_refs: Vec<_> = incomplete_coverage
            .manifest
            .ssts
            .iter()
            .filter(|sst| sst.kind == SstKind::Nodes)
            .collect();
        let required = crate::flush::union_indexed_props(&sc);
        assert!(
            plan_node_bucket(&incomplete_refs, u64::MAX, 10, &required, false).is_some(),
            "an incomplete-coverage marker must force a one-time rewrite"
        );

        // Simulate a fully-compacted 2.0.4 namespace: one L1 body whose
        // manifest predates both the locator and paged mirrors.
        let mut legacy = current.clone();
        let descriptor = legacy
            .manifest
            .ssts
            .iter_mut()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        descriptor.level = SstLevel(1);
        descriptor.node_locator = None;
        let legacy_sst_id = descriptor.id;
        for index in &mut descriptor.unique_property_indices {
            index.format = crate::manifest::PropertyIndexFormat::BincodeV0;
            index.paged = None;
        }
        for index in &mut descriptor.equality_property_indices {
            index.format = crate::manifest::PropertyIndexFormat::BincodeV0;
            index.paged = None;
        }
        let refs: Vec<_> = legacy
            .manifest
            .ssts
            .iter()
            .filter(|sst| sst.kind == SstKind::Nodes)
            .collect();
        let required = crate::flush::union_indexed_props(&sc);
        let migration =
            plan_node_bucket(&refs, u64::MAX, 10, &required, false).expect("migration plan");
        assert_eq!(migration.inputs.len(), 1);
        assert_eq!(migration.target_level, 1);

        let out = compact_leveled(&ms, &fence, &legacy, &sc, u64::MAX, 10)
            .await
            .unwrap();
        assert_eq!(out.source_ssts_removed, 1);
        let upgraded = out
            .committed
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        assert_ne!(
            upgraded.id, legacy_sst_id,
            "a true legacy property-index generation requires one full Nodes rewrite"
        );
        assert!(upgraded.node_locator.is_some());
        assert!(upgraded
            .unique_property_indices
            .iter()
            .all(
                |index| index.format == crate::manifest::PropertyIndexFormat::PagedV1
                    || index.paged.is_some()
            ));
        assert!(upgraded
            .equality_property_indices
            .iter()
            .all(
                |index| index.format == crate::manifest::PropertyIndexFormat::PagedV1
                    || index.paged.is_some()
            ));
        assert!(upgraded
            .equality_property_indices
            .iter()
            .all(|index| index.mixed_type_complete));
        let property_reader = open_compacted_property_pages(s.clone(), &p, upgraded).await;
        let (projected, _) = property_reader
            .project_node_ids(&["name".into()], &[*sorted_node_id(7).as_bytes()])
            .await
            .unwrap();
        assert_eq!(
            projected[0].properties["name"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::Str("legacy".into()))
        );

        let upgraded_refs = vec![upgraded];
        assert!(
            plan_node_bucket(&upgraded_refs, u64::MAX, 10, &required, false).is_none(),
            "migration must not rewrite the same L1 again"
        );
    }

    #[tokio::test]
    async fn lone_l1_missing_property_pages_rebuilds_access_bundle_without_parquet_rewrite() {
        let s = store();
        let p = paths("compact-migrate-missing-property-pages");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let sc = schema();
        let alice = sorted_node_id(7);

        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: alice },
            1,
            MemOp::Upsert(node_payload("sidecar-only", Some(42))),
        );
        let current = flush(&ms, &fence, &base, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;

        // Model the exact backup/restore gap: Parquet and the exact-record
        // `.nloc2` survived, while only the nested `.npp` descriptor was
        // absent. A lone settled L1 must rebuild the complete access bundle
        // without copying its authoritative Parquet object.
        let mut missing_pages = current.clone();
        let source = missing_pages
            .manifest
            .ssts
            .iter_mut()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        source.level = SstLevel(1);
        let source_id = source.id;
        let source_path = source.path.clone();
        let prior_locator_path = source.node_locator.as_ref().unwrap().path.clone();
        source.node_locator.as_mut().unwrap().property_pages = None;
        assert!(crate::manifest::node_locator_has_exact_records(source));
        assert!(!node_descriptor_has_property_pages(source));

        let prepared = prepare_leveled(&ms, &fence, &missing_pages, &sc, u64::MAX, 10)
            .await
            .unwrap();
        let prepared_node = prepared
            .new_descs
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .expect("sidecar-only replacement");
        assert_eq!(
            (prepared_node.id, prepared_node.path.as_str()),
            (source_id, source_path.as_str()),
            "sidecar-only prepare must retain the parent Parquet object"
        );
        let prepared_locator = prepared_node.node_locator.as_ref().unwrap();
        assert_ne!(
            prepared_locator.path, prior_locator_path,
            "migration must publish a fresh retry-safe locator object"
        );
        let prepared_properties = prepared_locator
            .property_pages
            .as_ref()
            .expect("prepared property pages");
        assert_eq!(prepared_properties.parent_sst_id, source_id);
        assert!(prepared_properties.is_bound_to(prepared_node));

        // Both access objects are already durable while the manifest still
        // points at the old generation: prepare always uploads before CAS.
        for relative in [&prepared_locator.path, &prepared_properties.path] {
            let absolute = Path::from(format!("{}/{}", p.namespace_prefix().as_ref(), relative));
            s.head(&absolute).await.unwrap();
        }

        let out = install_prepared(&ms, &fence, &missing_pages, prepared)
            .await
            .unwrap();
        let upgraded = out
            .committed
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        assert_eq!(
            (upgraded.id, upgraded.path.as_str()),
            (source_id, source_path.as_str())
        );
        let reader = open_compacted_property_pages(s, &p, upgraded).await;
        let (projected, _) = reader
            .project_node_ids(&["name".into(), "age".into()], &[*alice.as_bytes()])
            .await
            .unwrap();
        assert_eq!(
            projected[0].properties["name"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::Str(
                "sidecar-only".into()
            ))
        );
        assert_eq!(
            projected[0].properties["age"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::I64(42))
        );

        let required = crate::flush::union_indexed_props(&sc);
        assert!(
            plan_node_bucket(&[upgraded], u64::MAX, 10, &required, false).is_none(),
            "the completed access bundle must not schedule another migration"
        );
    }

    #[tokio::test]
    async fn create_property_index_rewrites_a_lone_l1_once() {
        let s = store();
        let p = paths("compact-ddl-index-migration");
        let ms = ManifestStore::new(s, p);
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let old_schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
            })
            .unwrap()
            .build();

        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                id: sorted_node_id(1),
            },
            1,
            MemOp::Upsert(node_payload("Alice", None)),
        );
        let old = flush(&ms, &fence, &base, &mt.freeze(), old_schema)
            .await
            .unwrap()
            .committed;
        assert!(old.manifest.ssts[0].equality_property_indices.is_empty());

        // Model metadata-only CREATE INDEX after the namespace has settled at
        // one L1. No data LSN changes, so the schema coverage test itself must
        // schedule the rewrite.
        let indexed_schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, false)
                    .unwrap()
                    .with_indexed(true)],
            })
            .unwrap()
            .build();
        let mut ddl = old.manifest.next_version(fence.writer_id);
        ddl.schema = indexed_schema.clone();
        ddl.ssts[0].level = SstLevel(1);
        let ddl = ms.commit(&fence, &old, ddl).await.unwrap();
        let refs: Vec<_> = ddl
            .manifest
            .ssts
            .iter()
            .filter(|sst| sst.kind == SstKind::Nodes)
            .collect();
        let required = crate::flush::union_indexed_props(&indexed_schema);
        assert!(plan_node_bucket(&refs, u64::MAX, 10, &required, false).is_some());

        let out = compact_leveled(&ms, &fence, &ddl, &indexed_schema, u64::MAX, 10)
            .await
            .unwrap();
        let node = out
            .committed
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        assert!(node
            .equality_property_indices
            .iter()
            .any(|index| index.property == "name" && index.paged.is_some()));
        let refs = vec![node];
        assert!(
            plan_node_bucket(&refs, u64::MAX, 10, &required, false).is_none(),
            "the DDL migration must be a one-time rewrite"
        );
    }

    #[tokio::test]
    async fn oversized_string_keys_keep_legacy_indexes_without_rewrite_churn() {
        let s = store();
        let p = paths("compact-oversized-paged-key");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let indexed_schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, false)
                    .unwrap()
                    .with_unique(true)
                    .with_indexed(true)],
            })
            .unwrap()
            .build();
        let long_key = "x".repeat(4 * 1024 + 512);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                id: sorted_node_id(1),
            },
            1,
            MemOp::Upsert(node_payload(&long_key, None)),
        );
        let current = flush(&ms, &fence, &base, &mt.freeze(), indexed_schema.clone())
            .await
            .unwrap()
            .committed;
        let node = current
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        let equality = node
            .equality_property_indices
            .iter()
            .find(|index| index.property == "name")
            .unwrap();
        assert!(equality.paged.is_none() && equality.paged_build_unsupported);

        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(current.clone(), &view, s, p);
        let found = snap
            .lookup_node_by_property("Person", "name", &long_key)
            .await
            .unwrap()
            .expect("legacy equality sidecar remains authoritative");
        assert_eq!(found.id, sorted_node_id(1));

        let refs = vec![node];
        let required = crate::flush::union_indexed_props(&indexed_schema);
        assert!(
            plan_node_bucket(&refs, u64::MAX, 10, &required, false).is_none(),
            "unsupported PagedV1 key must not trigger endless maintenance"
        );
    }

    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn vector_build_attempt_markers_prevent_error_churn_and_retry_on_change() {
        use crate::manifest::{Manifest, VectorIndexDescriptor, VectorMetric, VectorQuantization};

        let descriptor = VectorIndexDescriptor {
            name: "doc_emb".into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 16,
            l_build: 32,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        };
        let schema_without_filter = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("emb", DataType::FloatVector { dim: 2 }, false).unwrap(),
                    PropertyDef::new("vigente", DataType::Bool, false).unwrap(),
                ],
            })
            .unwrap()
            .build();
        let mut manifest = Manifest::empty(crate::Epoch::ZERO, Uuid::now_v7());
        manifest.schema = schema_without_filter;
        manifest.vector_indexes.push(descriptor.clone());
        manifest.ssts.push(search_generation_node_sst(7));

        // A malformed two-member corpus reaches vector validation and is
        // skipped per-index. The authoritative attempt is still durable for
        // this exact generation, so unchanged maintenance does not rewrite L1
        // forever. No VectorGraph descriptor is emitted, hence reads continue
        // through the exact fallback.
        let invalid_members = vec![VectorMemberCollector::from_test_members(
            descriptor.clone(),
            vec![
                ((*sorted_node_id(1).as_bytes()), vec![1.0]),
                ((*sorted_node_id(2).as_bytes()), vec![0.0]),
            ],
        )
        .unwrap()];
        let mut old_vector = search_generation_node_sst(7);
        old_vector.kind = SstKind::VectorGraph;
        old_vector.scope = "doc_emb".into();
        old_vector.kind_specific = KindSpecificStats::VectorGraph {
            dim: 2,
            metric: "cosine".into(),
            point_count: 2,
            r: 16,
            l_build: 32,
            alpha: 1.2,
            entry_medoid: 0,
        };
        let old_vector_id = old_vector.id;
        let old_vectors = BTreeMap::from([("doc_emb".to_string(), vec![&old_vector])]);
        let (new_descriptors, removed, attempted) = build_vector_indexes_from_members(
            store(),
            &paths("vector-marker-error"),
            1,
            7,
            invalid_members,
            &old_vectors,
        )
        .await
        .unwrap();
        assert!(new_descriptors.is_empty());
        assert_eq!(
            removed,
            vec![old_vector_id],
            "a rejected generation must retire any stale physical graph"
        );
        assert_eq!(attempted, vec!["doc_emb"]);
        let attempted_set = HashSet::from([(SstKind::VectorGraph, "doc_emb".to_string())]);
        manifest.search_index_builds = catalog_build_states(&manifest, 7, &attempted_set);
        assert!(
            !search_indexes_need_rebuild(&manifest),
            "the rejected generation was already attempted"
        );

        manifest.ssts[0].max_lsn = 8;
        assert!(
            search_indexes_need_rebuild(&manifest),
            "a newer node generation must retry the rejected build"
        );
        manifest.ssts[0].max_lsn = 7;

        // One valid vector is a legitimate `Ok(None)` corpus: no graph body,
        // but the attempted generation is durable and must not rewrite L1 on
        // every maintenance tick.
        let small_members = vec![VectorMemberCollector::from_test_members(
            descriptor.clone(),
            vec![((*sorted_node_id(1).as_bytes()), vec![1.0, 0.0])],
        )
        .unwrap()];
        let (_, _, attempted) = build_vector_indexes_from_members(
            store(),
            &paths("vector-marker-empty"),
            1,
            7,
            small_members,
            &BTreeMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(attempted, vec!["doc_emb"]);
        let attempted_set = HashSet::from([(SstKind::VectorGraph, "doc_emb".to_string())]);
        manifest.search_index_builds = catalog_build_states(&manifest, 7, &attempted_set);
        assert!(!search_indexes_need_rebuild(&manifest));

        // Native filter postings are part of `.vg`. CREATE INDEX on a filter
        // property changes the signature without any node write/LSN advance.
        manifest.schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("emb", DataType::FloatVector { dim: 2 }, false).unwrap(),
                    PropertyDef::new("vigente", DataType::Bool, false)
                        .unwrap()
                        .with_indexed(true),
                ],
            })
            .unwrap()
            .build();
        assert!(
            search_indexes_need_rebuild(&manifest),
            "filter DDL must invalidate the vector build generation"
        );
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn text_build_attempt_marker_retries_only_on_generation_or_catalog_change() {
        use crate::manifest::{Manifest, TextIndexDescriptor};

        let mut manifest = Manifest::empty(crate::Epoch::ZERO, Uuid::now_v7());
        manifest.text_indexes.push(TextIndexDescriptor::new(
            "doc_text".into(),
            "Doc".into(),
            vec!["title".into()],
        ));
        manifest.ssts.push(search_generation_node_sst(11));

        // `attempted` is populated before the deterministic FTS body builder.
        // Thus the same marker covers success, an empty corpus, and a rejected
        // body without claiming that a physical TextIndex SST exists.
        let attempted = HashSet::from([(SstKind::TextIndex, "doc_text".to_string())]);
        manifest.search_index_builds = catalog_build_states(&manifest, 11, &attempted);
        assert!(!search_indexes_need_rebuild(&manifest));
        assert!(
            manifest
                .ssts
                .iter()
                .all(|sst| sst.kind != SstKind::TextIndex),
            "an attempt marker alone must not manufacture index availability"
        );

        manifest.ssts[0].max_lsn = 12;
        assert!(
            search_indexes_need_rebuild(&manifest),
            "a newer node generation must retry FTS"
        );
        manifest.ssts[0].max_lsn = 11;

        manifest.text_indexes[0] = TextIndexDescriptor::new(
            "doc_text".into(),
            "Doc".into(),
            vec!["title".into(), "body".into()],
        );
        assert!(
            search_indexes_need_rebuild(&manifest),
            "a catalog signature change must retry FTS"
        );
    }

    #[tokio::test]
    async fn plan_bucket_merge_includes_the_entire_l0_backlog() {
        let (.., base) = build_two_l0_node_ssts().await;
        let mut template = base.manifest.ssts[0].clone();
        template.level = SstLevel::L0;
        template.size_bytes = 1;
        let backlog: Vec<SstDescriptor> = (0..11)
            .map(|_| SstDescriptor {
                id: Uuid::now_v7(),
                ..template.clone()
            })
            .collect();
        let refs: Vec<&SstDescriptor> = backlog.iter().collect();
        let plan = plan_bucket_merge(&refs, 1024, 10).expect("eleven L0s must compact");

        assert_eq!(
            plan.inputs.len(),
            backlog.len(),
            "one pass must scale with the captured backlog, not stop after three inputs"
        );
        assert!(
            plan.inputs.iter().all(|sst| sst.level == SstLevel::L0),
            "every captured L0 descriptor belongs to the plan"
        );
    }

    #[tokio::test]
    async fn one_compaction_pass_drains_more_than_three_l0_files() {
        let s = store();
        let p = paths("compact-full-l0-backlog");
        let ms = ManifestStore::new(s, p);
        let mut current = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        current.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(current.manifest.epoch);

        const BACKLOG: usize = 7;
        for i in 0..BACKLOG {
            current = flush_node_op(
                &ms,
                &fence,
                &current,
                sorted_node_id(i as u8 + 1),
                i as u64 + 1,
                MemOp::Upsert(node_payload(&format!("person-{i}"), None)),
            )
            .await;
        }
        let l0_before = current
            .manifest
            .ssts
            .iter()
            .filter(|sst| sst.kind == SstKind::Nodes && sst.level == SstLevel::L0)
            .count();
        assert_eq!(l0_before, BACKLOG);

        let outcome = compact_l0_to_l1(&ms, &fence, &current, &schema())
            .await
            .unwrap();
        assert_eq!(
            outcome.source_ssts_removed, BACKLOG,
            "the pass must consume every eligible L0 input captured in its basis"
        );
        assert_eq!(
            outcome
                .committed
                .manifest
                .ssts
                .iter()
                .filter(|sst| sst.kind == SstKind::Nodes && sst.level == SstLevel::L0)
                .count(),
            0,
            "no captured L0 may be left for artificial three-file follow-up passes"
        );
    }

    #[tokio::test]
    async fn compact_leveled_cascades_into_a_deeper_level() {
        let s = store();
        let p = paths("compact-cascade");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        let a = sorted_node_id(1);
        let b = sorted_node_id(2);
        let c = sorted_node_id(3);

        // Two L0s → compact (tiny budget, but a brand-new bucket still lands in
        // L1, not deeper).
        let m = flush_node_op(
            &ms,
            &fence,
            &base,
            a,
            10,
            MemOp::Upsert(node_payload("a", None)),
        )
        .await;
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            b,
            11,
            MemOp::Upsert(node_payload("b", None)),
        )
        .await;
        let after_l1 = compact_leveled(&ms, &fence, &m, &schema(), 1, 2)
            .await
            .unwrap();
        assert_eq!(
            node_levels(&after_l1.committed),
            vec![1],
            "two fresh L0s land in L1"
        );

        // A third L0 alongside the L1 overflows the tiny budget → cascade to L2.
        let m = flush_node_op(
            &ms,
            &fence,
            &after_l1.committed,
            c,
            12,
            MemOp::Upsert(node_payload("c", None)),
        )
        .await;
        let after_l2 = compact_leveled(&ms, &fence, &m, &schema(), 1, 2)
            .await
            .unwrap();
        assert_eq!(
            node_levels(&after_l2.committed),
            vec![2],
            "L0 + L1 over budget cascades into L2"
        );

        // All three nodes remain readable through the deeper level.
        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(after_l2.committed.clone(), &mt_view, s, p);
        for id in [a, b, c] {
            assert!(
                snap.lookup_node("Person", id).await.unwrap().is_some(),
                "node {id} must read through L2"
            );
        }
    }

    #[tokio::test]
    async fn tombstone_above_a_deeper_level_is_preserved_then_gcd_at_the_deepest() {
        // Resurrection safety: a tombstone merged at a shallow level while a
        // deeper level still holds the row's value must NOT be dropped, or the
        // delete would be undone. It is dropped only at the deepest merge.
        let s = store();
        let p = paths("compact-tomb-levels");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(1);
        let x = sorted_node_id(2);
        let y = sorted_node_id(3);
        let bob = sorted_node_id(4);
        let z = sorted_node_id(5);
        let big = 16 * 1024 * 1024u64; // larger than any SST these flushes make

        // Push alice's VALUE down into L2 (tiny budget forces the cascade).
        let m = flush_node_op(
            &ms,
            &fence,
            &base,
            alice,
            10,
            MemOp::Upsert(node_payload("alice", None)),
        )
        .await;
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            x,
            11,
            MemOp::Upsert(node_payload("x", None)),
        )
        .await;
        let m = compact_leveled(&ms, &fence, &m, &schema(), 1, 2)
            .await
            .unwrap()
            .committed; // L1
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            y,
            12,
            MemOp::Upsert(node_payload("y", None)),
        )
        .await;
        let m = compact_leveled(&ms, &fence, &m, &schema(), 1, 2)
            .await
            .unwrap()
            .committed; // L2
        assert_eq!(node_levels(&m), vec![2], "alice's value now lives in L2");

        // Build a fresh L1 from two new L0s (big budget keeps them at L1).
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            bob,
            13,
            MemOp::Upsert(node_payload("bob", None)),
        )
        .await;
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            z,
            14,
            MemOp::Upsert(node_payload("z", None)),
        )
        .await;
        let m = compact_leveled(&ms, &fence, &m, &schema(), big, 10)
            .await
            .unwrap()
            .committed;
        assert_eq!(
            node_levels(&m),
            vec![1, 2],
            "L1 (bob, z) sits above L2 (alice, x, y)"
        );

        // Tombstone alice in a new L0, then compact L0 + L1 → L1. The output is
        // NOT the deepest level (L2 is below), so the tombstone must survive.
        let m = flush_node_op(&ms, &fence, &m, alice, 20, MemOp::Tombstone).await;
        let m = compact_leveled(&ms, &fence, &m, &schema(), big, 10)
            .await
            .unwrap()
            .committed;
        assert_eq!(
            node_levels(&m),
            vec![1, 2],
            "the shallow merge stays at L1; L2 untouched"
        );
        let shallow = m
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes && sst.level == SstLevel(1))
            .expect("shallow L1 Nodes output");
        assert_eq!(
            match &shallow.kind_specific {
                KindSpecificStats::Nodes { tombstone_count } => *tombstone_count,
                _ => unreachable!("Nodes descriptor carries node stats"),
            },
            1,
            "the non-authoritative output must retain alice's tombstone"
        );
        let shallow_properties = open_compacted_property_pages(s.clone(), &p, shallow).await;
        let (projected, _) = shallow_properties
            .project_node_ids(&["name".into()], &[*alice.as_bytes()])
            .await
            .unwrap();
        assert_eq!(
            projected[0].properties["name"],
            crate::sst::nodes::property_pages::PropertyCell::Absent,
            "a retained tombstone has an exact empty property map"
        );
        assert_eq!(
            projected[0].ordinal, None,
            "empty tombstone rows must not invent a property cell"
        );

        // alice reads as deleted: the L1 tombstone (LSN 20) shadows the L2
        // value (LSN 10). If the shallow merge had GC'd the tombstone, alice
        // would resurrect from L2.
        {
            let mt = Memtable::new();
            let mt_view = mt.snapshot_view();
            let snap = Snapshot::new(m.clone(), &mt_view, s.clone(), p.clone());
            assert!(
                snap.lookup_node("Person", alice).await.unwrap().is_none(),
                "the preserved tombstone must keep alice deleted"
            );
            assert!(snap.lookup_node("Person", bob).await.unwrap().is_some());
        }

        // Now compact down to the deepest level (tiny budget merges L1 + L2):
        // the tombstone is authoritative and is dropped together with the
        // shadowed value, so alice is physically gone and the survivors remain.
        let m = compact_leveled(&ms, &fence, &m, &schema(), 1, 2)
            .await
            .unwrap()
            .committed;
        let deepest = m
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .expect("deepest Nodes output");
        assert_eq!(
            match &deepest.kind_specific {
                KindSpecificStats::Nodes { tombstone_count } => *tombstone_count,
                _ => unreachable!("Nodes descriptor carries node stats"),
            },
            0,
            "the authoritative merge must omit the GC'd tombstone"
        );
        let deepest_properties = open_compacted_property_pages(s.clone(), &p, deepest).await;
        assert_eq!(
            deepest_properties.node_count(),
            deepest.row_count,
            "property pages contain exactly the surviving winner stream"
        );
        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(m.clone(), &mt_view, s, p);
        assert!(
            snap.lookup_node("Person", alice).await.unwrap().is_none(),
            "alice stays deleted after the deepest merge GCs the tombstone"
        );
        for id in [x, y, bob, z] {
            assert!(
                snap.lookup_node("Person", id).await.unwrap().is_some(),
                "survivor {id} must remain after GC"
            );
        }
    }

    /// A prepare that is never installed strands only unreferenced objects:
    /// the min_age guard keeps a fresh prepare (possibly about to be
    /// installed) alive, and once past it the janitor's orphan sweep
    /// reclaims the prepared bodies while the still-referenced inputs keep
    /// serving reads.
    #[tokio::test]
    async fn abandoned_prepare_is_reclaimed_by_the_orphan_sweep() {
        use std::time::Duration;

        use crate::janitor::sweep_orphans;

        let (s, p, ms, fence, base) = build_two_l0_node_ssts().await;
        let prepared = prepare_compaction(&ms, &fence, &base, &schema())
            .await
            .unwrap();
        assert!(!prepared.is_noop());

        // The prepared bodies are durable at their UUID paths but referenced
        // by no manifest version.
        let object_path =
            |rel: &str| Path::from(format!("{}/{}", p.namespace_prefix().as_ref(), rel));
        let prepared_paths: Vec<String> =
            prepared.new_descs.iter().map(|d| d.path.clone()).collect();
        assert!(!prepared_paths.is_empty());
        for rel in &prepared_paths {
            assert!(
                s.head(&object_path(rel)).await.is_ok(),
                "prepared body must be durable before the sweep: {rel}"
            );
        }

        // Never installed. A young prepare survives the min_age guard…
        let young = sweep_orphans(&ms, u64::MAX, Duration::from_secs(86_400), 4, true)
            .await
            .unwrap();
        assert_eq!(young.orphans_deleted, 0, "a young prepare must survive");

        // …but once past min_age the unreferenced bodies are reclaimed.
        let swept = sweep_orphans(&ms, u64::MAX, Duration::ZERO, 4, true)
            .await
            .unwrap();
        assert!(
            swept.orphans_deleted >= prepared_paths.len(),
            "expected >= {} orphans deleted, got {}",
            prepared_paths.len(),
            swept.orphans_deleted
        );
        for rel in &prepared_paths {
            assert!(
                s.head(&object_path(rel)).await.is_err(),
                "abandoned prepared body must be reclaimed: {rel}"
            );
        }

        // The manifest-referenced L0 inputs are untouched: reads still serve.
        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(base.clone(), &mt_view, s, p);
        for id in [sorted_node_id(1), sorted_node_id(2)] {
            assert!(
                snap.lookup_node("Person", id).await.unwrap().is_some(),
                "input SSTs must keep serving after the sweep"
            );
        }
    }

    /// RFC-030 (`vector-index`): end-to-end through real compaction — write
    /// clustered `Doc` embeddings across two L0 SSTs, compact to L1, and the
    /// build hook materialises a searchable `VectorGraph` SST whose recall
    /// tracks brute force.
    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn compaction_builds_a_searchable_text_index() {
        use crate::manifest::TextIndexDescriptor;

        // Same forcing rationale as the vector twin: the assertions require
        // one consolidated base, which the default incremental policy would
        // (correctly) not produce from two fresh deltas.
        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = SearchCompactionEnvRestore::configure();

        fn idx_id(i: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..16].copy_from_slice(&i.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }
        fn doc_payload(body: &str, label_id: u32) -> Bytes {
            let mut props: BTreeMap<String, Value> = BTreeMap::new();
            props.insert("body".into(), Value::Str(body.into()));
            NodeWriteRecord {
                properties: props,
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }

        let s = store();
        let p = paths("compact-text");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let note_id = base.manifest.label_dict.intern("Note");
        base.manifest.text_indexes.push(TextIndexDescriptor::new(
            "note_ft".into(),
            "Note".into(),
            vec!["body".into()],
        ));
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Note".into(),
                properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        // "fox" appears in exactly one document (rare → high IDF); "common" in
        // the rest. Six docs across two L0 SSTs.
        let bodies = [
            "fox the cat",
            "common the cat",
            "common the dog",
            "common the bird",
            "common the lizard",
            "common the fish",
        ];
        let mut cur = base;
        let mut i: u64 = 0;
        for chunk in bodies.chunks(3) {
            let mut mt = Memtable::new();
            for b in chunk {
                let id = idx_id(i + 1);
                mt.apply(
                    MemKey::Node { id },
                    i + 1,
                    MemOp::Upsert(doc_payload(b, note_id.0)),
                );
                i += 1;
            }
            let frozen = mt.freeze();
            let after = flush(&ms, &fence, &cur, &frozen, schema.clone())
                .await
                .unwrap();
            cur = after.committed;
        }

        // Compact L0 → L1. The build hook emits one TextIndex SST.
        let out = compact_l0_to_l1(&ms, &fence, &cur, &schema).await.unwrap();
        let text_state = out
            .committed
            .manifest
            .search_lsm
            .iter()
            .find(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Text
                    && state.index_name == "note_ft"
            })
            .expect("text base registered as an active Search-LSM generation");
        let barrier_id = text_state.compat_barrier_sst_id.unwrap();
        let fts: Vec<&SstDescriptor> = out
            .committed
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::TextIndex && d.id != barrier_id)
            .collect();
        assert_eq!(fts.len(), 1, "exactly one TextIndex SST after compaction");
        assert_eq!(
            out.committed
                .manifest
                .ssts
                .iter()
                .filter(|d| d.kind == SstKind::TextIndex)
                .count(),
            2,
            "data base and downgrade barrier are both ordinary SST descriptors"
        );
        let barrier = out
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == barrier_id)
            .unwrap();
        let barrier_body = get_sst_body(s.as_ref(), &p, barrier).await.unwrap();
        crate::search_lsm::validate_search_barrier(text_state, &barrier_body).unwrap();
        assert_eq!(fts[0].scope, "note_ft");
        let doc_count = match &fts[0].kind_specific {
            KindSpecificStats::TextIndex { doc_count, .. } => *doc_count,
            _ => 0,
        };
        assert_eq!(doc_count, bodies.len() as u64, "all docs indexed");

        // Search through the production dispatch: the consolidated base is a
        // range-readable FT4 artifact, not the 2.0.6 monolithic body, so the
        // legacy `TextIndex::decode` cannot read it. `text_search` returning
        // `Some` also proves the index actually served (a flat-scan fallback
        // yields `None`), which the old in-memory decode never could.
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &empty_view, s.clone(), p.clone());
        let hits = snap
            .text_search(
                "note_ft",
                "Note",
                &crate::text::parse_query("fox common"),
                None,
            )
            .await
            .unwrap()
            .expect("the consolidated FT4 base must serve, not fall back");
        assert_eq!(hits.len(), bodies.len(), "every doc matches a query term");
        assert_eq!(
            hits[0].0,
            idx_id(1),
            "the rare-term doc ranks first via real IDF"
        );
        assert!(
            snap.text_search("note_ft", "Note", &crate::text::parse_query("fox"), Some(5),)
                .await
                .unwrap()
                .is_some(),
            "a valid base + barrier generation must serve"
        );
        let mut corrupt_barrier = barrier_body.to_vec();
        corrupt_barrier[0] ^= 1;
        let barrier_absolute = format!("{}/{}", p.namespace_prefix().as_ref(), barrier.path);
        s.put(
            &Path::from(barrier_absolute),
            PutPayload::from(Bytes::from(corrupt_barrier)),
        )
        .await
        .unwrap();
        let snap = Snapshot::new(out.committed.clone(), &empty_view, s.clone(), p.clone());
        assert!(
            snap.text_search("note_ft", "Note", &crate::text::parse_query("fox"), Some(5),)
                .await
                .unwrap()
                .is_none(),
            "a corrupt barrier must select the exact fallback"
        );
    }

    /// The text twin of the vector downgrade-adoption cycle (plan item 30):
    /// an old writer drops the unknown Search-LSM state but PRESERVES the
    /// checksummed `.slb` barrier. The next maintenance pass must re-adopt
    /// the generation metadata-only — zero SSTs rewritten, the barrier
    /// reused byte-for-byte — and a DDL-stale marker must refuse adoption.
    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn text_preserved_barrier_readopts_after_state_wipe() {
        use crate::manifest::TextIndexDescriptor;
        use crate::text::parse_query;

        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = SearchCompactionEnvRestore::configure();

        fn idx_id(i: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..16].copy_from_slice(&i.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }
        fn doc_payload(body: &str, label_id: u32) -> Bytes {
            NodeWriteRecord {
                properties: BTreeMap::from([("body".into(), Value::Str(body.into()))]),
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }

        let s = store();
        let p = paths("compact-text-preserved-barrier");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let note_id = base.manifest.label_dict.intern("Note");
        base.manifest.text_indexes.push(TextIndexDescriptor::new(
            "note_ft".into(),
            "Note".into(),
            vec!["body".into()],
        ));
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Note".into(),
                properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        let mut cur = base;
        for (index, body) in ["fox the cat", "common the dog", "common the bird"]
            .iter()
            .enumerate()
        {
            let mut mt = Memtable::new();
            let lsn = (index + 1) as u64;
            mt.apply(
                MemKey::Node { id: idx_id(lsn) },
                lsn,
                MemOp::Upsert(doc_payload(body, note_id.0)),
            );
            cur = flush(&ms, &fence, &cur, &mt.freeze(), schema.clone())
                .await
                .unwrap()
                .committed;
        }
        let settled = compact_l0_to_l1(&ms, &fence, &cur, &schema)
            .await
            .unwrap()
            .committed;
        let text_state = settled
            .manifest
            .search_lsm
            .iter()
            .find(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Text
                    && state.index_name == "note_ft"
            })
            .expect("active text generation");
        let barrier_id = text_state.compat_barrier_sst_id.unwrap();
        let base_body_id = text_state.segments[0].sst_id;

        // The downgrade: state dropped, both descriptors (body + barrier)
        // preserved. A DDL-stale marker variant must refuse adoption.
        let mut wiped = settled.manifest.next_version(fence.writer_id);
        wiped.search_lsm.clear();
        let mut ddl_stale = wiped.clone();
        ddl_stale
            .search_index_builds
            .iter_mut()
            .find(|state| state.kind == SstKind::TextIndex && state.name == "note_ft")
            .expect("text build marker")
            .catalog_signature
            .push_str("-pre-ddl");
        assert!(
            !search_lsm_adoption_needed(&ddl_stale),
            "a DDL-stale text marker must rebuild, never adopt"
        );
        let legacy = ms.commit(&fence, &settled, wiped).await.unwrap();
        assert!(search_lsm_adoption_needed(&legacy.manifest));

        let adopted = compact_leveled(&ms, &fence, &legacy, &schema, u64::MAX, 10)
            .await
            .unwrap();
        assert_eq!(
            (adopted.source_ssts_removed, adopted.new_ssts_written),
            (0, 0),
            "preserved-barrier text adoption must be metadata-only"
        );
        let adopted_state = adopted
            .committed
            .manifest
            .search_lsm
            .iter()
            .find(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Text
                    && state.index_name == "note_ft"
            })
            .expect("the wiped text generation must come back");
        assert_eq!(adopted_state.segments[0].sst_id, base_body_id);
        assert_eq!(
            adopted_state.compat_barrier_sst_id,
            Some(barrier_id),
            "the preserved barrier must be reused byte-for-byte"
        );
        crate::search_lsm::validate_search_lsm(&adopted.committed.manifest).unwrap();

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(adopted.committed.clone(), &empty_view, s.clone(), p.clone());
        let hits = snap
            .text_search("note_ft", "Note", &parse_query("fox"), Some(5))
            .await
            .unwrap()
            .expect("the re-adopted generation must serve natively");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, idx_id(1));
    }

    /// The residual left open by the interop-marker work: a 2.0.6 downgrade
    /// drops the unknown Search-LSM state, and in this variant the `.slb`
    /// barrier object is lost with it. The FT4 base survives as an ordinary
    /// `TextIndex` descriptor next to the minted build marker, so the marker
    /// keeps suppressing the rebuild while the adoption probe (NAMIFT03-only)
    /// keeps rejecting the FT4 body — every query flat-scans forever. The
    /// contract pinned here: the first pass drops the deterministically
    /// disproven marker, the second plans the full rebuild and serves.
    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn text_base_with_lost_barrier_falls_back_to_rebuild() {
        use crate::manifest::TextIndexDescriptor;
        use crate::text::parse_query;

        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = SearchCompactionEnvRestore::configure();

        fn idx_id(i: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..16].copy_from_slice(&i.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }
        fn doc_payload(body: &str, label_id: u32) -> Bytes {
            NodeWriteRecord {
                properties: BTreeMap::from([("body".into(), Value::Str(body.into()))]),
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }

        let s = store();
        let p = paths("compact-text-lost-barrier");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let note_id = base.manifest.label_dict.intern("Note");
        base.manifest.text_indexes.push(TextIndexDescriptor::new(
            "note_ft".into(),
            "Note".into(),
            vec!["body".into()],
        ));
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Note".into(),
                properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        let bodies = ["fox the cat", "common the dog", "common the bird"];
        let mut cur = base;
        for (index, body) in bodies.iter().enumerate() {
            let mut mt = Memtable::new();
            let lsn = (index + 1) as u64;
            mt.apply(
                MemKey::Node { id: idx_id(lsn) },
                lsn,
                MemOp::Upsert(doc_payload(body, note_id.0)),
            );
            cur = flush(&ms, &fence, &cur, &mt.freeze(), schema.clone())
                .await
                .unwrap()
                .committed;
        }
        let settled = compact_l0_to_l1(&ms, &fence, &cur, &schema)
            .await
            .unwrap()
            .committed;
        assert!(!search_indexes_need_rebuild(&settled.manifest));
        assert!(
            settled
                .manifest
                .search_index_builds
                .iter()
                .any(|marker| marker.kind == SstKind::TextIndex && marker.name == "note_ft"),
            "the consolidation must mint the 2.0.6-interop marker this scenario turns on"
        );

        // Downgrade: the old writer preserves ordinary descriptors but drops
        // the unknown top-level state, and the barrier is lost outright.
        let mut downgraded = settled.clone();
        downgraded.manifest.search_lsm.clear();
        let barrier_ids: HashSet<Uuid> = downgraded
            .manifest
            .ssts
            .iter()
            .filter(|descriptor| {
                crate::search_lsm::is_canonical_search_barrier_descriptor(descriptor)
            })
            .map(|descriptor| descriptor.id)
            .collect();
        assert!(
            !barrier_ids.is_empty(),
            "the settled generation has a barrier to lose"
        );
        for descriptor in downgraded
            .manifest
            .ssts
            .iter()
            .filter(|descriptor| barrier_ids.contains(&descriptor.id))
        {
            let absolute = format!("{}/{}", p.namespace_prefix().as_ref(), descriptor.path);
            s.delete(&Path::from(absolute)).await.unwrap();
        }
        downgraded
            .manifest
            .ssts
            .retain(|descriptor| !barrier_ids.contains(&descriptor.id));
        // The stall precondition: metadata alone still promises an adoption,
        // so the rebuild stays suppressed.
        assert!(!search_indexes_need_rebuild(&downgraded.manifest));

        // Pass 1: the magic probe deterministically disproves the promise;
        // the marker must fall with it instead of certifying forever.
        let pass1 = compact_leveled(&ms, &fence, &downgraded, &schema, u64::MAX, 10)
            .await
            .unwrap()
            .committed;
        assert!(
            !pass1
                .manifest
                .search_index_builds
                .iter()
                .any(|marker| marker.kind == SstKind::TextIndex && marker.name == "note_ft"),
            "an unadoptable base must drop its interop marker"
        );
        assert!(
            search_indexes_need_rebuild(&pass1.manifest),
            "with the marker gone the full rebuild is unsuppressed"
        );

        // Pass 2: the rebuild plans from the settled tree and serves.
        let pass2 = compact_leveled(&ms, &fence, &pass1, &schema, u64::MAX, 10)
            .await
            .unwrap()
            .committed;
        assert!(!search_indexes_need_rebuild(&pass2.manifest));
        assert!(
            pass2.manifest.search_lsm.iter().any(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Text
                    && state.index_name == "note_ft"
            }),
            "a fresh Active generation must replace the stranded base"
        );
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(pass2.clone(), &empty_view, s.clone(), p.clone());
        let hits = snap
            .text_search("note_ft", "Note", &parse_query("fox"), Some(5))
            .await
            .unwrap()
            .expect("the rebuilt generation must serve natively, not flat-scan");
        assert_eq!(hits.len(), 1, "exactly the one fox document matches");
        assert_eq!(hits[0].0, idx_id(1));
    }

    /// A `.ft` body with a legacy magic (a NAMIFT01 file left behind by a
    /// format bump) must not error queries: `text_search` treats the index as
    /// absent (`Ok(None)` → flat-scan fallback, the `.vg` convention) until
    /// the next authoritative compaction rebuilds it.
    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn legacy_text_index_body_falls_back_to_flat_scan() {
        use crate::manifest::TextIndexDescriptor;
        use crate::text::parse_query;

        fn idx_id(i: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..16].copy_from_slice(&i.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }

        let s = store();
        let p = paths("compact-text-legacy");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let note_id = base.manifest.label_dict.intern("Note");
        base.manifest.text_indexes.push(TextIndexDescriptor::new(
            "note_ft".into(),
            "Note".into(),
            vec!["body".into()],
        ));
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Note".into(),
                properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        let mut cur = base;
        for (i, body) in ["fox the cat", "common the dog"].iter().enumerate() {
            let mut props: BTreeMap<String, Value> = BTreeMap::new();
            props.insert("body".into(), Value::Str((*body).into()));
            let rec = NodeWriteRecord {
                properties: props,
                schema_version: 1,
                labels: vec![note_id.0],
            };
            let mut mt = Memtable::new();
            mt.apply(
                MemKey::Node {
                    id: idx_id(i as u64 + 1),
                },
                i as u64 + 1,
                MemOp::Upsert(rec.encode().unwrap()),
            );
            cur = flush(&ms, &fence, &cur, &mt.freeze(), schema.clone())
                .await
                .unwrap()
                .committed;
        }
        let out = compact_l0_to_l1(&ms, &fence, &cur, &schema)
            .await
            .unwrap()
            .committed;
        let ft = out
            .manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::TextIndex)
            .expect("compaction builds the .ft SST")
            .clone();

        let empty = Memtable::new();
        let mt_view = empty.snapshot_view();
        let snap = Snapshot::new(out.clone(), &mt_view, s.clone(), p.clone());
        let q = parse_query("fox");

        // Sanity: the freshly-built v2 body serves.
        let hits = snap
            .text_search("note_ft", "Note", &q, Some(5))
            .await
            .unwrap()
            .expect("the compacted index must serve");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, idx_id(1));

        // Overwrite the object with an old-magic body: decode fails, and the
        // search reports "index absent" instead of erroring the query.
        let absolute = format!("{}/{}", p.namespace_prefix().as_ref(), ft.path);
        s.put(
            &Path::from(absolute),
            PutPayload::from_static(b"NAMIFT01legacy-postings"),
        )
        .await
        .unwrap();
        let got = snap
            .text_search("note_ft", "Note", &q, Some(5))
            .await
            .unwrap();
        assert!(got.is_none(), "a legacy body must fall back, not error");
    }

    // ── Streaming k-way merge ───────────────────────────────────────────

    fn indexed_person_label() -> LabelDef {
        LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false)
                    .unwrap()
                    .with_indexed(true),
                PropertyDef::new("age", DataType::Int32, true).unwrap(),
            ],
        }
    }

    fn parity_schema() -> Schema {
        SchemaBuilder::new()
            .label(indexed_person_label())
            .unwrap()
            .edge_type(knows_edge_with_declared_props())
            .unwrap()
            .build()
    }

    fn props_payload(pairs: &[(&str, Value)]) -> Bytes {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        for (k, v) in pairs {
            props.insert((*k).to_string(), v.clone());
        }
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            labels: vec![0],
        }
        .encode()
        .unwrap()
    }

    /// Multi-level fixture the streaming-merge tests share: overlapping node
    /// keys with declared + overflow properties and a node tombstone across
    /// L0s AND an L1 (from an intermediate compaction), plus fwd+inv edges
    /// with declared + overflow properties and an edge tombstone. Returned
    /// at the point where a three-bucket merge (nodes, fwd, inv) is pending.
    async fn multi_level_fixture(
        ns: &str,
    ) -> (
        Arc<dyn ObjectStore>,
        NamespacePaths,
        ManifestStore,
        WriterFence,
        LoadedManifest,
    ) {
        let s = store();
        let p = paths(ns);
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let sc = parity_schema();

        let a = sorted_node_id(1);
        let b = sorted_node_id(2);
        let c = sorted_node_id(3);
        let d = sorted_node_id(4);
        let knows = |src, dst| MemKey::Edge {
            edge_type: "KNOWS".into(),
            src,
            dst,
        };

        // Flush 1: first versions of a and b; edges a->b and a->c.
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: a },
            10,
            MemOp::Upsert(props_payload(&[
                ("name", Value::Str("a0".into())),
                ("age", Value::I64(30)),
            ])),
        );
        mt.apply(
            MemKey::Node { id: b },
            11,
            MemOp::Upsert(props_payload(&[("name", Value::Str("b0".into()))])),
        );
        let mut ab: BTreeMap<String, Value> = BTreeMap::new();
        ab.insert("since".into(), Value::I64(2020));
        ab.insert("weight".into(), Value::F64(0.5));
        ab.insert("note".into(), Value::Str("first".into()));
        mt.apply(knows(a, b), 12, MemOp::Upsert(edge_payload_with_props(ab)));
        let mut ac: BTreeMap<String, Value> = BTreeMap::new();
        ac.insert("since".into(), Value::I64(2021));
        mt.apply(knows(a, c), 13, MemOp::Upsert(edge_payload_with_props(ac)));
        let m = flush(&ms, &fence, &base, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;

        // Flush 2: overlapping updates — a gets a newer version, c appears
        // with an overflow (undeclared) property, edge a->b is rewritten.
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: c },
            20,
            MemOp::Upsert(props_payload(&[
                ("name", Value::Str("c0".into())),
                ("nickname", Value::Str("ce".into())),
            ])),
        );
        mt.apply(
            MemKey::Node { id: a },
            21,
            MemOp::Upsert(props_payload(&[
                ("name", Value::Str("a1".into())),
                ("age", Value::I64(31)),
            ])),
        );
        let mut ab2: BTreeMap<String, Value> = BTreeMap::new();
        ab2.insert("since".into(), Value::I64(2024));
        ab2.insert("weight".into(), Value::F64(0.9));
        ab2.insert("note".into(), Value::Str("second".into()));
        mt.apply(knows(a, b), 22, MemOp::Upsert(edge_payload_with_props(ab2)));
        let m = flush(&ms, &fence, &m, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;

        // Intermediate compaction: every bucket lands in L1.
        let m = compact_leveled(&ms, &fence, &m, &sc, 1, 2)
            .await
            .unwrap()
            .committed;

        // Flush 3: d re-uses b's name (equality posting with two ids across
        // time), b is deleted, edge a->c is deleted.
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: d },
            30,
            MemOp::Upsert(props_payload(&[("name", Value::Str("b0".into()))])),
        );
        mt.apply(MemKey::Node { id: b }, 31, MemOp::Tombstone);
        mt.apply(knows(a, c), 32, MemOp::Tombstone);
        let m = flush(&ms, &fence, &m, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;

        (s, p, ms, fence, m)
    }

    #[tokio::test]
    async fn streaming_merge_multi_level_parity_nodes_and_edges() {
        let (s, p, ms, fence, m) = multi_level_fixture("compact-stream-parity").await;
        let sc = parity_schema();

        // Merge the pending L0s + L1s (tiny budget → cascade to the deepest
        // level, so tombstone GC applies), then add one more flush and merge
        // again so the final node merge spans L0 + a deep level.
        let m = compact_leveled(&ms, &fence, &m, &sc, 1, 2)
            .await
            .unwrap()
            .committed;
        let e = sorted_node_id(5);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: e },
            40,
            MemOp::Upsert(props_payload(&[("name", Value::Str("e0".into()))])),
        );
        let m = flush(&ms, &fence, &m, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;
        let m = compact_leveled(&ms, &fence, &m, &sc, 1, 2)
            .await
            .unwrap()
            .committed;

        let a = sorted_node_id(1);
        let b = sorted_node_id(2);
        let c = sorted_node_id(3);
        let d = sorted_node_id(4);

        // Structural checks on the merged node SST: counts, key/LSN ranges,
        // GC'd tombstones, sidecar descriptors.
        let node_desc = m
            .manifest
            .ssts
            .iter()
            .find(|dsc| dsc.kind == SstKind::Nodes)
            .expect("one merged node SST");
        assert_eq!(node_desc.row_count, 4, "a, c, d, e survive; b is GC'd");
        assert_eq!(
            node_desc.kind_specific,
            KindSpecificStats::Nodes { tombstone_count: 0 }
        );
        assert_eq!(node_desc.min_key, *a.as_bytes());
        assert_eq!(node_desc.max_key, *e.as_bytes());
        assert_eq!(node_desc.min_lsn, 20);
        assert_eq!(node_desc.max_lsn, 40);
        assert!(
            node_desc
                .equality_property_indices
                .iter()
                .any(|d| d.property == "name"),
            "the equality sidecar for the indexed property must survive the merge"
        );
        let label_index = node_desc
            .label_index
            .as_ref()
            .expect("label-index sidecar re-emitted");
        assert_eq!(label_index.per_label_counts, vec![(0, 4)]);

        // Read parity: winners, overflow properties, tombstones, edges.
        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(m.clone(), &mt_view, s, p);

        let va = snap.lookup_node("Person", a).await.unwrap().unwrap();
        assert_eq!(va.lsn, 21);
        assert_eq!(va.properties.get("name"), Some(&Value::Str("a1".into())));
        assert_eq!(va.properties.get("age"), Some(&Value::I64(31)));
        assert!(snap.lookup_node("Person", b).await.unwrap().is_none());
        let vc = snap.lookup_node("Person", c).await.unwrap().unwrap();
        assert_eq!(vc.properties.get("name"), Some(&Value::Str("c0".into())));
        assert_eq!(
            vc.properties.get("nickname"),
            Some(&Value::Str("ce".into())),
            "overflow (undeclared) properties must survive the merge"
        );
        assert!(snap.lookup_node("Person", d).await.unwrap().is_some());
        assert!(snap.lookup_node("Person", e).await.unwrap().is_some());

        // Equality sidecar still resolves: "b0" now maps to d only (b is
        // deleted), "a1" to a.
        let hits = snap
            .lookup_nodes_by_property("Person", "name", "b0")
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, d);
        let hits = snap
            .lookup_nodes_by_property("Person", "name", "a1")
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, a);

        // Edges: a->b carries the rewritten declared + overflow properties;
        // the tombstoned a->c is gone in both orientations.
        let outs = snap.out_edges("KNOWS", a).await.unwrap();
        assert_eq!(outs.edges.len(), 1);
        assert_eq!(outs.edges[0].dst, b);
        assert_eq!(outs.edges[0].lsn, 22);
        assert_eq!(
            outs.edges[0].properties.get("since"),
            Some(&Value::I64(2024))
        );
        assert_eq!(
            outs.edges[0].properties.get("weight"),
            Some(&Value::F64(0.9))
        );
        assert_eq!(
            outs.edges[0].properties.get("note"),
            Some(&Value::Str("second".into()))
        );
        let ins = snap.in_edges("KNOWS", b).await.unwrap();
        assert_eq!(ins.edges.len(), 1);
        assert_eq!(ins.edges[0].src, a);
        assert!(snap.in_edges("KNOWS", c).await.unwrap().edges.is_empty());

        for desc in m
            .manifest
            .ssts
            .iter()
            .filter(|dsc| matches!(dsc.kind, SstKind::EdgesFwd | SstKind::EdgesInv))
        {
            assert_eq!(desc.row_count, 1, "only a->b survives ({:?})", desc.kind);
            match &desc.kind_specific {
                KindSpecificStats::Edges {
                    key_count,
                    tombstone_count,
                    ..
                } => {
                    assert_eq!(*key_count, 1);
                    assert_eq!(*tombstone_count, 0, "the GC'd edge tombstone is gone");
                }
                other => panic!("expected edge stats, got {other:?}"),
            }
        }
    }

    /// `SstDescriptor` reduced to its deterministic parts: everything except
    /// the freshly-minted UUID, the UUID-derived paths, and `created_at`.
    fn normalized_desc(d: &SstDescriptor) -> String {
        format!(
            "{:?}|{}|{}|{}|{}|{:?}|{:?}|{}|{}|{}|{}|{:?}|{}|{:?}|{:?}|{:?}|{:?}",
            d.kind,
            d.scope,
            d.level.as_u32(),
            d.size_bytes,
            d.row_count,
            d.min_key,
            d.max_key,
            d.min_lsn,
            d.max_lsn,
            d.schema_version_min,
            d.schema_version_max,
            d.kind_specific,
            d.bloom.is_some(),
            d.unique_property_indices
                .iter()
                .map(|u| (u.property.clone(), u.size_bytes, u.entry_count))
                .collect::<Vec<_>>(),
            d.equality_property_indices
                .iter()
                .map(|u| (u.property.clone(), u.size_bytes, u.distinct_values))
                .collect::<Vec<_>>(),
            d.label_index.as_ref().map(|l| (
                l.size_bytes,
                l.label_count,
                l.posting_count,
                l.per_label_counts.clone()
            )),
            d.per_label_property_stats,
        )
    }

    // Holds the chunk-env lock READ-side: a concurrent
    // `NAMIDB_COMPACTION_MERGE_CHUNK_ROWS` mutation between the two prepare
    // runs would legitimately change Parquet page boundaries and fail the
    // byte comparison.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn streaming_prepare_is_deterministic_modulo_uuids() {
        let _guard = MERGE_CHUNK_ENV.lock().unwrap_or_else(|e| e.into_inner());
        let (s, p, ms, fence, m) = multi_level_fixture("compact-stream-determinism").await;
        let sc = parity_schema();

        let first = prepare_compaction(&ms, &fence, &m, &sc).await.unwrap();
        let second = prepare_compaction(&ms, &fence, &m, &sc).await.unwrap();
        assert!(!first.is_noop());
        assert_eq!(
            first.new_descs.len(),
            3,
            "nodes + fwd + inv buckets all plan a merge"
        );

        let normalize = |descs: &[SstDescriptor]| {
            let mut v: Vec<String> = descs.iter().map(normalized_desc).collect();
            v.sort();
            v
        };
        assert_eq!(
            normalize(&first.new_descs),
            normalize(&second.new_descs),
            "two prepares over the same basis must agree modulo UUIDs"
        );

        // Body bytes are identical too: match descriptors by (kind, scope)
        // and GET both runs' durable bodies.
        for d1 in &first.new_descs {
            let d2 = second
                .new_descs
                .iter()
                .find(|d| d.kind == d1.kind && d.scope == d1.scope)
                .expect("matching descriptor in the second run");
            let get = |rel: &str| {
                let path = Path::from(format!("{}/{}", p.namespace_prefix().as_ref(), rel));
                let store = s.clone();
                async move { store.get(&path).await.unwrap().bytes().await.unwrap() }
            };
            let b1 = get(&d1.path).await;
            let b2 = get(&d2.path).await;
            // An edge SST binds its body to its own UUID so a misrouted object
            // cannot be served under another descriptor. Two prepares therefore
            // differ in the binding section and in the page checksums covering
            // it — but in nothing the merge itself produced.
            match d1.kind {
                SstKind::EdgesFwd | SstKind::EdgesInv => {
                    assert_edge_bodies_agree_modulo_identity(&b1, &b2, d1)
                }
                _ => assert_eq!(
                    b1, b2,
                    "{:?}/{} bodies must be byte-identical",
                    d1.kind, d1.scope
                ),
            }
        }
    }

    /// Assert two edge SST bodies carry identical merge output, allowing only
    /// the per-object identity binding to differ.
    fn assert_edge_bodies_agree_modulo_identity(b1: &Bytes, b2: &Bytes, desc: &SstDescriptor) {
        use crate::sst::edges::{
            EdgeFileFooter, EdgeSstBinding, SECTION_PAGE_CHECKSUMS, SECTION_SST_BINDING,
        };

        let what = format!("{:?}/{}", desc.kind, desc.scope);
        let (f1, _) = EdgeFileFooter::decode(b1).expect("first run edge footer");
        let (f2, _) = EdgeFileFooter::decode(b2).expect("second run edge footer");

        assert_eq!(
            (
                f1.key_count,
                f1.edge_count,
                f1.offsets_bits,
                f1.min_key_id,
                f1.max_key_id,
                f1.min_lsn,
                f1.max_lsn,
                f1.schema_version_min,
                f1.schema_version_max,
            ),
            (
                f2.key_count,
                f2.edge_count,
                f2.offsets_bits,
                f2.min_key_id,
                f2.max_key_id,
                f2.min_lsn,
                f2.max_lsn,
                f2.schema_version_min,
                f2.schema_version_max,
            ),
            "{what} footer scalars"
        );
        assert_eq!(f1.sections.len(), f2.sections.len(), "{what} section count");

        for (e1, e2) in f1.sections.iter().zip(&f2.sections) {
            assert_eq!(
                (e1.kind, &e1.name, e1.offset, e1.length, e1.codec),
                (e2.kind, &e2.name, e2.offset, e2.length, e2.codec),
                "{what} section layout"
            );
            if matches!(e1.kind, SECTION_SST_BINDING | SECTION_PAGE_CHECKSUMS) {
                continue;
            }
            let range = e1.offset as usize..(e1.offset + e1.length) as usize;
            assert_eq!(
                e1.xxhash3_64, e2.xxhash3_64,
                "{what} section {} content hash",
                e1.kind
            );
            assert_eq!(
                b1[range.clone()],
                b2[range],
                "{what} section {} bytes",
                e1.kind
            );
        }

        let entry = f1
            .find_kind(SECTION_SST_BINDING)
            .expect("edge SST carries an identity binding");
        let range = entry.offset as usize..(entry.offset + entry.length) as usize;
        let k1 = EdgeSstBinding::decode(&b1[range.clone()]).expect("first run binding");
        let k2 = EdgeSstBinding::decode(&b2[range]).expect("second run binding");
        assert_ne!(
            k1.sst_id, k2.sst_id,
            "{what} two prepares must mint distinct object ids"
        );
        assert_eq!(
            (k1.header_xxhash3_64, k1.sections_xxhash3_64),
            (k2.header_xxhash3_64, k2.sections_xxhash3_64),
            "{what} binding covers identical header and section root"
        );
    }

    #[tokio::test]
    async fn duplicate_key_across_three_sources_keeps_only_the_highest_lsn() {
        let s = store();
        let p = paths("compact-stream-shadow");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        let x = sorted_node_id(1);
        let anchor = sorted_node_id(2);
        let m = flush_node_op(
            &ms,
            &fence,
            &base,
            x,
            5,
            MemOp::Upsert(node_payload("v1", None)),
        )
        .await;
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            x,
            9,
            MemOp::Upsert(node_payload("v2", None)),
        )
        .await;
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            x,
            12,
            MemOp::Upsert(node_payload("v3", None)),
        )
        .await;
        let m = flush_node_op(
            &ms,
            &fence,
            &m,
            anchor,
            13,
            MemOp::Upsert(node_payload("anchor", None)),
        )
        .await;

        let out = compact_l0_to_l1(&ms, &fence, &m, &schema()).await.unwrap();
        assert_eq!(out.source_ssts_removed, 4);
        let node_desc = out
            .committed
            .manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::Nodes)
            .unwrap();
        assert_eq!(node_desc.row_count, 2, "x deduped to one version + anchor");

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        let vx = snap.lookup_node("Person", x).await.unwrap().unwrap();
        assert_eq!(vx.lsn, 12, "exactly the highest-LSN version survives");
        assert_eq!(vx.properties.get("name"), Some(&Value::Str("v3".into())));
    }

    #[tokio::test]
    async fn tombstone_winner_across_three_sources_gcs_only_when_authoritative() {
        // Authoritative (deepest) merge: the tombstone winner disappears.
        {
            let s = store();
            let p = paths("compact-stream-tomb-gc");
            let ms = ManifestStore::new(s.clone(), p.clone());
            let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
            base.manifest.label_dict.intern("Person");
            let fence = WriterFence::new(base.manifest.epoch);

            let x = sorted_node_id(1);
            let anchor = sorted_node_id(2);
            let m = flush_node_op(
                &ms,
                &fence,
                &base,
                x,
                5,
                MemOp::Upsert(node_payload("v1", None)),
            )
            .await;
            let m = flush_node_op(
                &ms,
                &fence,
                &m,
                x,
                9,
                MemOp::Upsert(node_payload("v2", None)),
            )
            .await;
            let m = flush_node_op(&ms, &fence, &m, x, 12, MemOp::Tombstone).await;
            let m = flush_node_op(
                &ms,
                &fence,
                &m,
                anchor,
                13,
                MemOp::Upsert(node_payload("anchor", None)),
            )
            .await;
            let out = compact_l0_to_l1(&ms, &fence, &m, &schema()).await.unwrap();
            let node_desc = out
                .committed
                .manifest
                .ssts
                .iter()
                .find(|d| d.kind == SstKind::Nodes)
                .unwrap();
            assert_eq!(node_desc.row_count, 1, "only the anchor survives GC");
            assert_eq!(
                node_desc.kind_specific,
                KindSpecificStats::Nodes { tombstone_count: 0 }
            );
            let mt = Memtable::new();
            let mt_view = mt.snapshot_view();
            let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
            assert!(snap.lookup_node("Person", x).await.unwrap().is_none());
        }

        // Non-authoritative merge (a deeper level exists): the tombstone
        // winner is preserved so it keeps shadowing.
        {
            let s = store();
            let p = paths("compact-stream-tomb-keep");
            let ms = ManifestStore::new(s.clone(), p.clone());
            let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
            base.manifest.label_dict.intern("Person");
            let fence = WriterFence::new(base.manifest.epoch);

            let x = sorted_node_id(1);
            let y = sorted_node_id(2);
            let z = sorted_node_id(3);
            let big = 16 * 1024 * 1024u64;

            // Push y and z down to L2 with tiny budgets.
            let m = flush_node_op(
                &ms,
                &fence,
                &base,
                y,
                1,
                MemOp::Upsert(node_payload("y", None)),
            )
            .await;
            let m = flush_node_op(
                &ms,
                &fence,
                &m,
                z,
                2,
                MemOp::Upsert(node_payload("z", None)),
            )
            .await;
            let m = compact_leveled(&ms, &fence, &m, &schema(), 1, 2)
                .await
                .unwrap()
                .committed;
            let m = flush_node_op(
                &ms,
                &fence,
                &m,
                y,
                3,
                MemOp::Upsert(node_payload("y2", None)),
            )
            .await;
            let m = compact_leveled(&ms, &fence, &m, &schema(), 1, 2)
                .await
                .unwrap()
                .committed;
            assert_eq!(node_levels(&m), vec![2]);

            // Three L0 sources for x, tombstone at the highest LSN; a big
            // budget keeps the merge at L1 above the untouched L2.
            let m = flush_node_op(
                &ms,
                &fence,
                &m,
                x,
                5,
                MemOp::Upsert(node_payload("v1", None)),
            )
            .await;
            let m = flush_node_op(
                &ms,
                &fence,
                &m,
                x,
                9,
                MemOp::Upsert(node_payload("v2", None)),
            )
            .await;
            let m = flush_node_op(&ms, &fence, &m, x, 12, MemOp::Tombstone).await;
            let m = compact_leveled(&ms, &fence, &m, &schema(), big, 10)
                .await
                .unwrap()
                .committed;
            assert_eq!(node_levels(&m), vec![1, 2]);
            let l1 = m
                .manifest
                .ssts
                .iter()
                .find(|d| d.kind == SstKind::Nodes && d.level == SstLevel(1))
                .unwrap();
            assert_eq!(l1.row_count, 1, "the winning tombstone is the only row");
            assert_eq!(
                l1.kind_specific,
                KindSpecificStats::Nodes { tombstone_count: 1 },
                "a non-authoritative merge must keep the tombstone"
            );

            let mt = Memtable::new();
            let mt_view = mt.snapshot_view();
            let snap = Snapshot::new(m.clone(), &mt_view, s, p);
            assert!(snap.lookup_node("Person", x).await.unwrap().is_none());
            assert!(snap.lookup_node("Person", y).await.unwrap().is_some());
            assert!(snap.lookup_node("Person", z).await.unwrap().is_some());
        }
    }

    /// Serialises the tests that mutate `NAMIDB_COMPACTION_MERGE_CHUNK_ROWS`
    /// (process-global), restoring the previous value afterwards.
    static MERGE_CHUNK_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Intentional: the guard serialises the env mutation across the whole
    // compaction; the test drives its own single-threaded runtime.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn tiny_merge_chunks_round_trip_the_whole_bucket() {
        let _guard = MERGE_CHUNK_ENV.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("NAMIDB_COMPACTION_MERGE_CHUNK_ROWS").ok();
        std::env::set_var("NAMIDB_COMPACTION_MERGE_CHUNK_ROWS", "5");

        let s = store();
        let p = paths("compact-stream-chunks");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        // Two 40-row flushes overlapping on ids 21..=40; the merged bucket
        // (60 rows) spans many 5-row chunks and the overlap crosses several
        // chunk boundaries.
        let mut mt = Memtable::new();
        for i in 1..=40u8 {
            mt.apply(
                MemKey::Node {
                    id: sorted_node_id(i),
                },
                100 + i as u64,
                MemOp::Upsert(node_payload(&format!("first{i}"), None)),
            );
        }
        let m = flush(&ms, &fence, &base, &mt.freeze(), schema())
            .await
            .unwrap()
            .committed;
        let mut mt = Memtable::new();
        for i in 21..=60u8 {
            mt.apply(
                MemKey::Node {
                    id: sorted_node_id(i),
                },
                200 + i as u64,
                MemOp::Upsert(node_payload(&format!("second{i}"), None)),
            );
        }
        let m = flush(&ms, &fence, &m, &mt.freeze(), schema())
            .await
            .unwrap()
            .committed;

        let out = compact_l0_to_l1(&ms, &fence, &m, &schema()).await;
        match prev {
            Some(v) => std::env::set_var("NAMIDB_COMPACTION_MERGE_CHUNK_ROWS", v),
            None => std::env::remove_var("NAMIDB_COMPACTION_MERGE_CHUNK_ROWS"),
        }
        drop(_guard);

        let out = out.unwrap();
        let node_desc = out
            .committed
            .manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::Nodes)
            .unwrap();
        assert_eq!(node_desc.row_count, 60);
        assert_eq!(node_desc.min_key, *sorted_node_id(1).as_bytes());
        assert_eq!(node_desc.max_key, *sorted_node_id(60).as_bytes());

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        for i in 1..=60u8 {
            let v = snap
                .lookup_node("Person", sorted_node_id(i))
                .await
                .unwrap()
                .unwrap_or_else(|| panic!("node {i} lost across a chunk boundary"));
            let expected = if i >= 21 {
                format!("second{i}")
            } else {
                format!("first{i}")
            };
            assert_eq!(
                v.properties.get("name"),
                Some(&Value::Str(expected)),
                "wrong winner for overlapping id {i}"
            );
        }
    }

    #[test]
    fn node_cursor_defers_decode_and_bounds_resident_batch() {
        let label = LabelDef {
            name: String::new(),
            properties: Vec::new(),
        };
        // 16 rows at 4 rows per row group → 4 row groups.
        let options = NodeSstWriterOptions {
            row_group_target_rows: 4,
            expected_keys: 16,
            ..Default::default()
        };
        let mut writer = IncrementalNodeSstWriter::new(&label, options, 4).unwrap();
        for i in 1..=16u8 {
            writer
                .push(NodeRow {
                    id: *sorted_node_id(i).as_bytes(),
                    lsn: i as u64,
                    op: MemOp::Upsert(node_payload(&format!("n{i}"), None)),
                })
                .unwrap();
        }
        let finish = writer.finish().unwrap();
        assert_eq!(
            parse_node_sst_metadata(&finish.body)
                .unwrap()
                .num_row_groups(),
            4,
            "fixture must be multi-row-group"
        );

        let mut cursor = NodeSourceCursor::open_with_batch_rows(&label, finish.body, 3).unwrap();
        assert_eq!(
            cursor.batches_decoded, 0,
            "opening every fan-in source must not decode its first row group"
        );
        assert!(
            cursor.peek().is_none(),
            "unactivated cursor has no resident row"
        );
        cursor.ensure_positioned().unwrap();
        assert_eq!(cursor.batches_decoded, 1);
        let mut seen = 0u8;
        while let Some((id, lsn)) = cursor.peek() {
            seen += 1;
            assert_eq!(id, *sorted_node_id(seen).as_bytes());
            assert_eq!(lsn, seen as u64);
            assert!(
                cursor.view.as_ref().unwrap().len() <= 3,
                "one cursor may retain only its configured bounded batch"
            );
            cursor.advance().unwrap();
        }
        assert_eq!(seen, 16);
        assert!(
            (6..=8).contains(&cursor.batches_decoded),
            "16 rows require at least six bounded batches; row-group boundaries may split more"
        );
    }

    #[test]
    fn merge_node_sources_streams_across_row_group_boundaries() {
        let label = LabelDef {
            name: String::new(),
            properties: Vec::new(),
        };
        // Two multi-row-group sources overlapping on ids 11..=20; source B's
        // higher LSNs win the overlap.
        let build = |ids: std::ops::RangeInclusive<u8>, lsn_base: u64, tag: &str| {
            let options = NodeSstWriterOptions {
                row_group_target_rows: 3,
                expected_keys: 20,
                ..Default::default()
            };
            let mut writer = IncrementalNodeSstWriter::new(&label, options, 3).unwrap();
            for i in ids {
                writer
                    .push(NodeRow {
                        id: *sorted_node_id(i).as_bytes(),
                        lsn: lsn_base + i as u64,
                        op: MemOp::Upsert(node_payload(&format!("{tag}{i}"), None)),
                    })
                    .unwrap();
            }
            writer.finish().unwrap().body
        };
        let body_a = build(1..=20, 100, "a");
        let body_b = build(11..=30, 200, "b");
        let output_sst_id = Uuid::now_v7();

        let out = merge_node_sources(
            vec![
                NodeMergeInput {
                    body: body_a,
                    min_key: *sorted_node_id(1).as_bytes(),
                },
                NodeMergeInput {
                    body: body_b,
                    min_key: *sorted_node_id(11).as_bytes(),
                },
            ],
            &label,
            &label,
            true,
            &schema(),
            &LabelDictionary::new(),
            "",
            output_sst_id,
            NodeMergeIndexSpecs::default(),
        )
        .unwrap();
        assert_eq!(out.finish.stats.row_count, 30);
        assert_eq!(out.finish.stats.min_lsn, 101);
        assert_eq!(out.finish.stats.max_lsn, 230);
        assert_eq!(out.sidecars.property_pages_upload.sst_id(), output_sst_id);
        assert_eq!(
            out.sidecars.property_pages_upload.stats().node_count,
            out.finish.stats.row_count
        );

        // Decode the merged body and verify order + winners row by row.
        let reader = NodeSstReader::open(label.clone(), out.finish.body).unwrap();
        let mut rows: Vec<([u8; 16], u64)> = Vec::new();
        for batch in reader.scan().unwrap() {
            let ids = batch
                .column_by_name(COL_NODE_ID)
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
                .unwrap()
                .clone();
            let lsns = batch
                .column_by_name(COL_LSN)
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .unwrap()
                .clone();
            for row in 0..batch.num_rows() {
                rows.push((ids.value(row).try_into().unwrap(), lsns.value(row)));
            }
        }
        assert_eq!(rows.len(), 30);
        for (idx, (id, lsn)) in rows.iter().enumerate() {
            let i = idx as u8 + 1;
            assert_eq!(
                *id,
                *sorted_node_id(i).as_bytes(),
                "output must stay sorted"
            );
            let expected_lsn = if i >= 11 {
                200 + i as u64
            } else {
                100 + i as u64
            };
            assert_eq!(*lsn, expected_lsn, "id {i} must keep the highest LSN");
        }
    }

    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn compaction_builds_a_searchable_vector_graph() {
        use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
        use rand::Rng;
        use rand::SeedableRng;

        // This test's downgrade/adoption scenario needs one complete base, so
        // it forces consolidation. Under the default incremental policy the
        // two flushed deltas would (correctly) survive compaction instead.
        // The force is one-shot: once the singleton base exists, the later
        // adoption compaction has no consolidatable debt and stays metadata-only.
        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = SearchCompactionEnvRestore::configure();

        fn normalize_inplace(v: &mut [f32]) {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in v {
                    *x /= n;
                }
            }
        }
        fn idx_id(i: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..16].copy_from_slice(&i.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }
        fn doc_payload(emb: Vec<f32>, label_id: u32) -> Bytes {
            let mut props: BTreeMap<String, Value> = BTreeMap::new();
            props.insert("emb".into(), Value::Vec(emb));
            NodeWriteRecord {
                properties: props,
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }

        let s = store();
        let p = paths("compact-vector");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let doc_id = base.manifest.label_dict.intern("Doc");
        base.manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_emb".into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim: 16,
            metric: VectorMetric::Cosine,
            r: 32,
            l_build: 64,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("emb", DataType::FloatVector { dim: 16 }, false).unwrap(),
                ],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        // 4 well-separated centroids; 160 docs (40/cluster) across 2 L0 SSTs.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2024);
        let centroids: Vec<Vec<f32>> = (0..4)
            .map(|_| {
                let mut c: Vec<f32> = (0..16).map(|_| rng.gen::<f32>()).collect();
                normalize_inplace(&mut c);
                c
            })
            .collect();
        let mut cluster_of: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();

        let mut cur = base;
        let mut i: u64 = 0;
        for _sst in 0..2 {
            let mut mt = Memtable::new();
            for _ in 0..80 {
                let cluster = (i % 4) as usize;
                let mut emb: Vec<f32> = centroids[cluster]
                    .iter()
                    .map(|b| b + 0.02 * rng.gen::<f32>())
                    .collect();
                normalize_inplace(&mut emb);
                let id = idx_id(i + 1);
                cluster_of.insert(id, cluster);
                mt.apply(
                    MemKey::Node { id },
                    i + 1,
                    MemOp::Upsert(doc_payload(emb, doc_id.0)),
                );
                i += 1;
            }
            let frozen = mt.freeze();
            let after = flush(&ms, &fence, &cur, &frozen, schema.clone())
                .await
                .unwrap();
            cur = after.committed;
        }
        assert!(cur.manifest.ssts.iter().all(|d| d.level == SstLevel::L0));

        // Compact L0 → L1. The build hook emits one VectorGraph SST alongside
        // the merged node SST.
        let out = compact_l0_to_l1(&ms, &fence, &cur, &schema).await.unwrap();
        let vector_state = out
            .committed
            .manifest
            .search_lsm
            .iter()
            .find(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Vector
                    && state.index_name == "doc_emb"
            })
            .expect("vector base registered as an active Search-LSM generation");
        let barrier_id = vector_state.compat_barrier_sst_id.unwrap();
        let vgs: Vec<&SstDescriptor> = out
            .committed
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::VectorGraph && d.id != barrier_id)
            .collect();
        assert_eq!(vgs.len(), 1, "exactly one VectorGraph SST after compaction");
        assert_eq!(
            out.committed
                .manifest
                .ssts
                .iter()
                .filter(|d| d.kind == SstKind::VectorGraph)
                .count(),
            2,
            "data base and downgrade barrier are both ordinary SST descriptors"
        );
        assert_eq!(vgs[0].scope, "doc_emb");
        let stats = match &vgs[0].kind_specific {
            KindSpecificStats::VectorGraph { point_count, .. } => *point_count,
            _ => 0,
        };
        assert_eq!(stats, 160, "all 160 docs indexed");

        // Query through the production magic-dispatch + range reader; a query
        // near centroid 0 must surface cluster-0 docs without downloading the
        // V5 artifact in full.
        let mut q = centroids[0].clone();
        normalize_inplace(&mut q);
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snapshot = Snapshot::new(out.committed.clone(), &empty_view, s.clone(), p.clone());
        let (hits, point_count) = snapshot
            .try_vector_search_with_point_count("doc_emb", &q, 10, 48)
            .await
            .unwrap()
            .expect("fresh V5 index");
        assert_eq!(point_count, 160);
        assert_eq!(hits.len(), 10);
        // The query sits on centroid 0; the true top-10 are cluster-0 docs.
        // Count how many returned hits belong to cluster 0 (recall proxy).
        let cluster0_hits = hits
            .iter()
            .filter(|(id, _)| cluster_of.get(id) == Some(&0))
            .count();
        assert!(
            cluster0_hits >= 8,
            "expected >= 8/10 hits from cluster 0, got {cluster0_hits}"
        );

        // Model a rolling downgrade/upgrade: an old writer reserializes the
        // manifest, drops the unknown Search-LSM state, but preserves both
        // ordinary SST descriptors. The next maintenance pass must recover
        // the checksummed `.slb` state instead of remaining in ambiguous
        // flat-scan mode or rebuilding the 160-vector corpus.
        let vector_base_id = vgs[0].id;
        let mut legacy_manifest = out.committed.manifest.next_version(fence.writer_id);
        legacy_manifest.search_lsm.clear();
        let mut ddl_stale = legacy_manifest.clone();
        ddl_stale
            .search_index_builds
            .iter_mut()
            .find(|state| state.kind == SstKind::VectorGraph && state.name == "doc_emb")
            .expect("vector build marker")
            .catalog_signature
            .push_str("-pre-ddl");
        assert!(
            !search_lsm_adoption_needed(&ddl_stale),
            "a physically valid body with a stale catalog marker must rebuild, never be adopted"
        );
        let legacy = ms
            .commit(&fence, &out.committed, legacy_manifest)
            .await
            .unwrap();
        assert!(search_lsm_adoption_needed(&legacy.manifest));

        let adopted = compact_leveled(&ms, &fence, &legacy, &schema, u64::MAX, 10)
            .await
            .unwrap();
        assert_eq!(
            adopted.source_ssts_removed, 0,
            "metadata adoption must not rewrite node or vector data"
        );
        assert_eq!(
            adopted.new_ssts_written, 0,
            "recovering a preserved checksummed barrier writes no new SST"
        );
        let adopted_state = adopted
            .committed
            .manifest
            .search_lsm
            .iter()
            .find(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Vector
                    && state.index_name == "doc_emb"
            })
            .expect("the legacy V5 base must become an active generation");
        assert_eq!(adopted_state.segments[0].sst_id, vector_base_id);
        crate::search_lsm::validate_search_lsm(&adopted.committed.manifest).unwrap();
        let adopted_barrier = adopted
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| Some(descriptor.id) == adopted_state.compat_barrier_sst_id)
            .expect("adopted generation barrier descriptor");
        assert_eq!(
            adopted_barrier.id, barrier_id,
            "the preserved barrier should be reused byte-for-byte"
        );
        let adopted_barrier_body = get_sst_body(s.as_ref(), &p, adopted_barrier).await.unwrap();
        crate::search_lsm::validate_search_barrier(adopted_state, &adopted_barrier_body).unwrap();

        // A still older writer/backup may drop both the state and its barrier.
        // That case remains metadata-only too, but creates one fresh barrier.
        let mut missing_barrier_manifest = adopted.committed.manifest.next_version(fence.writer_id);
        missing_barrier_manifest.search_lsm.clear();
        missing_barrier_manifest
            .ssts
            .retain(|descriptor| descriptor.id != adopted_barrier.id);
        let missing_barrier = ms
            .commit(&fence, &adopted.committed, missing_barrier_manifest)
            .await
            .unwrap();
        let recreated = compact_leveled(&ms, &fence, &missing_barrier, &schema, u64::MAX, 10)
            .await
            .unwrap();
        assert_eq!(recreated.source_ssts_removed, 0);
        assert_eq!(recreated.new_ssts_written, 1);
        crate::search_lsm::validate_search_lsm(&recreated.committed.manifest).unwrap();
    }

    /// End-to-end index parity through the streaming merge: overlapping doc
    /// updates + a tombstone across two L0s, an authoritative compaction
    /// rebuilds the `.vg` and `.ft` from the winner stream, and both serve
    /// results that match a brute-force flat scan of the reconciled corpus
    /// (including the freshness stamps that keep them servable at all).
    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    #[tokio::test]
    async fn streaming_compaction_index_results_match_flat_scan() {
        use rand::{Rng, SeedableRng};

        use crate::manifest::{TextIndexDescriptor, VectorIndexDescriptor, VectorMetric};
        use crate::text::parse_query;

        // 2.1.0: flush writes native VG6/FT4 deltas under an Active Search-LSM
        // generation, so the legacy authoritative rebuild is correctly skipped
        // and the default incremental policy would (also correctly) retain the
        // deltas. This parity test needs exactly one consolidated base per
        // index, so it forces consolidation; the force works because the
        // flushed deltas are consolidatable debt (it is a one-shot no-op on an
        // already-consolidated singleton base).
        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = SearchCompactionEnvRestore::configure();

        fn idx_id(i: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..16].copy_from_slice(&i.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }
        fn doc_payload(emb: Vec<f32>, body: &str, label_id: u32) -> Bytes {
            let mut props: BTreeMap<String, Value> = BTreeMap::new();
            props.insert("emb".into(), Value::Vec(emb));
            props.insert("body".into(), Value::Str(body.into()));
            NodeWriteRecord {
                properties: props,
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }
        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        }

        let s = store();
        let p = paths("compact-stream-index-parity");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let doc_label = base.manifest.label_dict.intern("Doc");
        base.manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_emb".into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim: 8,
            metric: VectorMetric::Cosine,
            r: 32,
            l_build: 64,
            alpha: 1.2,
            quantization: crate::manifest::VectorQuantization::None,
        });
        base.manifest.text_indexes.push(TextIndexDescriptor::new(
            "note_ft".into(),
            "Doc".into(),
            vec!["body".into()],
        ));
        let sc = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("emb", DataType::FloatVector { dim: 8 }, false).unwrap(),
                    PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                ],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut corpus: BTreeMap<u64, Vec<f32>> = BTreeMap::new();

        // Flush 1: docs 1..=12.
        let mut mt = Memtable::new();
        for i in 1..=12u64 {
            let emb: Vec<f32> = (0..8).map(|_| rng.gen::<f32>()).collect();
            corpus.insert(i, emb.clone());
            mt.apply(
                MemKey::Node { id: idx_id(i) },
                i,
                MemOp::Upsert(doc_payload(emb, "alpha common", doc_label.0)),
            );
        }
        let m = flush(&ms, &fence, &base, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;

        // Flush 2: doc 3 rewritten (new embedding, new body), doc 5 deleted.
        let mut mt = Memtable::new();
        let new_emb: Vec<f32> = (0..8).map(|_| rng.gen::<f32>()).collect();
        corpus.insert(3, new_emb.clone());
        corpus.remove(&5);
        mt.apply(
            MemKey::Node { id: idx_id(3) },
            20,
            MemOp::Upsert(doc_payload(new_emb, "bravo target", doc_label.0)),
        );
        mt.apply(MemKey::Node { id: idx_id(5) }, 21, MemOp::Tombstone);
        let m = flush(&ms, &fence, &m, &mt.freeze(), sc.clone())
            .await
            .unwrap()
            .committed;

        let out = compact_l0_to_l1(&ms, &fence, &m, &sc).await.unwrap();
        let manifest = &out.committed.manifest;
        let node_desc = manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::Nodes)
            .unwrap();
        // Barrier-aware singleton lookups: the consolidation publishes a data
        // base plus a downgrade barrier per index, both ordinary descriptors
        // of the same kind. A first-match `find` could name the barrier.
        let vector_barrier = manifest
            .search_lsm
            .iter()
            .find(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Vector
                    && state.index_name == "doc_emb"
            })
            .expect("vector base registered as an active Search-LSM generation")
            .compat_barrier_sst_id
            .unwrap();
        let text_barrier = manifest
            .search_lsm
            .iter()
            .find(|state| {
                state.kind == crate::search_lsm::SearchLsmKind::Text
                    && state.index_name == "note_ft"
            })
            .expect("text base registered as an active Search-LSM generation")
            .compat_barrier_sst_id
            .unwrap();
        let vgs: Vec<&SstDescriptor> = manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::VectorGraph && d.id != vector_barrier)
            .collect();
        assert_eq!(vgs.len(), 1, "exactly one consolidated V5 base");
        let vg = vgs[0];
        let fts: Vec<&SstDescriptor> = manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::TextIndex && d.id != text_barrier)
            .collect();
        assert_eq!(fts.len(), 1, "exactly one consolidated FT4 base");
        let ft = fts[0];
        assert_eq!(vg.row_count, 11, "12 docs - 1 tombstone, update deduped");
        assert_eq!(ft.row_count, 11);
        // Freshness stamps. The V5 base carries the corpus high-water LSN
        // from its coverage. The FT4 base stamps its members' max mutation
        // LSN: the high-water event here is the id-5 tombstone at LSN 21,
        // which has no live member, so the base stamps the id-3 update at 20;
        // the Search-LSM coverage (not the descriptor stamp) is what proves
        // freshness through the tombstone.
        assert_eq!(vg.max_lsn, node_desc.max_lsn);
        assert_eq!(ft.max_lsn, 20);
        // Index descriptor key bounds are member NodeId bounds, not the legacy
        // 00..FF sentinel. Both corpora contain reconciled ids 1..=12 (id 5 is
        // deleted, but the extrema remain 1 and 12).
        assert_eq!(vg.min_key, *idx_id(1).as_bytes());
        assert_eq!(vg.max_key, *idx_id(12).as_bytes());
        assert_eq!(ft.min_key, *idx_id(1).as_bytes());
        assert_eq!(ft.max_key, *idx_id(12).as_bytes());

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);

        // KNN parity: with ef >= corpus size the Vamana search is exhaustive,
        // so the ids must equal the brute-force cosine top-k over the
        // reconciled corpus (updated doc 3 in, deleted doc 5 out).
        let query: Vec<f32> = corpus[&3].iter().map(|x| x + 0.01).collect();
        let hits = snap.vector_search("doc_emb", &query, 5, 64).await.unwrap();
        assert_eq!(hits.len(), 5);
        let got: Vec<NodeId> = hits.iter().map(|(id, _)| *id).collect();
        let mut flat: Vec<(u64, f32)> = corpus
            .iter()
            .map(|(i, emb)| (*i, cosine(&query, emb)))
            .collect();
        flat.sort_by(|a, b| b.1.total_cmp(&a.1));
        let expected: Vec<NodeId> = flat.iter().take(5).map(|(i, _)| idx_id(*i)).collect();
        assert_eq!(
            got, expected,
            "KNN through the rebuilt index must match the flat scan"
        );
        assert_eq!(
            got[0],
            idx_id(3),
            "the updated embedding wins, not the stale one"
        );

        // BM25 parity: "bravo" exists only in doc 3's REWRITTEN body; "alpha"
        // matches the other 10 live docs (not the deleted 5, not the stale 3).
        let hits = snap
            .text_search("note_ft", "Doc", &parse_query("bravo"), Some(5))
            .await
            .unwrap()
            .expect("the rebuilt .ft must serve (freshness gate passes)");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, idx_id(3));
        let hits = snap
            .text_search("note_ft", "Doc", &parse_query("alpha"), None)
            .await
            .unwrap()
            .expect("the rebuilt .ft must serve");
        let ids: std::collections::BTreeSet<NodeId> = hits.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 10);
        assert!(!ids.contains(&idx_id(3)), "doc 3's old body must be gone");
        assert!(!ids.contains(&idx_id(5)), "the deleted doc must be gone");
    }

    /// Upgrading a settled 2.0.5 node SST to the exact-record locator is a
    /// physical-only rewrite. Fresh vector/FTS generations already cover the
    /// same logical node high-water mark, so rebuilding either index would
    /// merely clone the complete embedding/document corpus and can exhaust
    /// memory on a legal-scale store.
    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn sidecar_only_migration_preserves_fresh_vector_and_text_generations() {
        // Observer of the DEFAULT policy: "fresh generations preserved" holds
        // only while no concurrent test leaks force_base=true.
        let _env_lock = SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        use crate::manifest::{
            NodeLocatorDescriptor, TextIndexDescriptor, VectorIndexDescriptor, VectorMetric,
            VectorQuantization,
        };
        use crate::text::parse_query;

        fn idx_id(i: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..].copy_from_slice(&i.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }

        fn doc_payload(embedding: Vec<f32>, body: &str, label_id: u32) -> Bytes {
            NodeWriteRecord {
                properties: BTreeMap::from([
                    ("emb".into(), Value::Vec(embedding)),
                    ("body".into(), Value::Str(body.into())),
                ]),
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }

        let s = store();
        let p = paths("compact-sidecar-only-keeps-search");
        let ms = ManifestStore::new(s.clone(), p.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let doc_label = base.manifest.label_dict.intern("Doc");
        base.manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_emb".into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 16,
            l_build: 32,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });
        base.manifest.text_indexes.push(TextIndexDescriptor::new(
            "doc_text".into(),
            "Doc".into(),
            vec!["body".into()],
        ));
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("emb", DataType::FloatVector { dim: 2 }, false).unwrap(),
                    PropertyDef::new("body", DataType::Utf8, false)
                        .unwrap()
                        .with_indexed(true),
                ],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        let docs = [
            (vec![1.0, 0.0], "alpha one"),
            (vec![0.9, 0.1], "alpha two"),
            (vec![0.0, 1.0], "beta three"),
            (vec![0.1, 0.9], "gamma four"),
        ];
        let mut current = base;
        for (chunk_index, chunk) in docs.chunks(2).enumerate() {
            let mut mt = Memtable::new();
            for (row_index, (embedding, body)) in chunk.iter().enumerate() {
                let ordinal = (chunk_index * 2 + row_index + 1) as u64;
                mt.apply(
                    MemKey::Node {
                        id: idx_id(ordinal),
                    },
                    ordinal,
                    MemOp::Upsert(doc_payload(embedding.clone(), body, doc_label.0)),
                );
            }
            current = flush(&ms, &fence, &current, &mt.freeze(), schema.clone())
                .await
                .unwrap()
                .committed;
        }

        // Build real, serving `.vg` and `.ft` generations first.
        let settled = compact_l0_to_l1(&ms, &fence, &current, &schema)
            .await
            .unwrap()
            .committed;
        assert!(
            !search_indexes_need_rebuild(&settled.manifest),
            "the first authoritative compaction must stamp both generations fresh"
        );
        let vector_id = settled
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::VectorGraph)
            .expect("real vector graph")
            .id;
        let text_id = settled
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::TextIndex)
            .expect("real text index")
            .id;
        let build_states = settled.manifest.search_index_builds.clone();
        let settled_node = settled
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        let settled_node_id = settled_node.id;
        let settled_node_path = settled_node.path.clone();

        // Model a valid 2.0.5 locator-only sidecar on the settled L1. The
        // source Parquet and search bodies are the real objects emitted above.
        let legacy_locator =
            crate::sst::paged_index::build_node_locator((1..=4).map(|i| *idx_id(i).as_bytes()))
                .unwrap();
        let legacy_relative = {
            let node = settled
                .manifest
                .ssts
                .iter()
                .find(|sst| sst.kind == SstKind::Nodes)
                .unwrap();
            let current = node.node_locator.as_ref().unwrap();
            format!(
                "{}.nloc",
                current
                    .path
                    .strip_suffix(".nloc2")
                    .expect("current locator suffix")
            )
        };
        let legacy_absolute = format!("{}/{}", p.namespace_prefix().as_ref(), legacy_relative);
        s.put(
            &Path::from(legacy_absolute),
            PutPayload::from(legacy_locator.clone()),
        )
        .await
        .unwrap();

        let mut legacy = settled.clone();
        let node = legacy
            .manifest
            .ssts
            .iter_mut()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        node.node_locator = Some(NodeLocatorDescriptor {
            path: legacy_relative,
            size_bytes: legacy_locator.len() as u64,
            entry_count: node.row_count,
            property_pages: None,
        });
        assert!(
            !search_indexes_need_rebuild(&legacy.manifest),
            "changing only locator format must not stale search generations"
        );

        let migrated = compact_leveled(&ms, &fence, &legacy, &schema, u64::MAX, 10)
            .await
            .unwrap()
            .committed;
        let migrated_node = migrated
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        assert!(
            migrated_node
                .node_locator
                .as_ref()
                .is_some_and(|locator| locator.path.ends_with(".nloc2")),
            "the legacy node locator must still migrate"
        );
        assert_eq!(
            (migrated_node.id, migrated_node.path.as_str()),
            (settled_node_id, settled_node_path.as_str()),
            "sidecar-only migration must not rewrite authoritative Parquet"
        );
        let property_reader = open_compacted_property_pages(s.clone(), &p, migrated_node).await;
        let (projected, _) = property_reader
            .project_node_ids(&["body".into()], &[*idx_id(1).as_bytes()])
            .await
            .unwrap();
        assert_eq!(
            projected[0].properties["body"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::Str("alpha one".into()))
        );
        assert_eq!(
            migrated
                .manifest
                .ssts
                .iter()
                .find(|sst| sst.kind == SstKind::VectorGraph)
                .unwrap()
                .id,
            vector_id,
            "sidecar-only migration must retain the physical vector generation"
        );
        assert_eq!(
            migrated
                .manifest
                .ssts
                .iter()
                .find(|sst| sst.kind == SstKind::TextIndex)
                .unwrap()
                .id,
            text_id,
            "sidecar-only migration must retain the physical text generation"
        );
        assert_eq!(
            migrated.manifest.search_index_builds, build_states,
            "fresh durable generation markers must remain byte-for-byte stable"
        );
        assert!(!search_indexes_need_rebuild(&migrated.manifest));

        // Exercise the general merge path too (the exact-locator-only case
        // above has its own descriptor-preserving fast path). A legacy paged
        // equality mirror requires rebuilding the node SST's property
        // sidecars, but still does not change the logical search corpus.
        let mut legacy_paged = migrated.clone();
        let legacy_equality: BTreeMap<String, Vec<[u8; 16]>> = docs
            .iter()
            .enumerate()
            .map(|(index, (_, body))| {
                (
                    (*body).to_owned(),
                    vec![*idx_id((index + 1) as u64).as_bytes()],
                )
            })
            .collect();
        let legacy_equality_body = bincode::serialize(&legacy_equality).unwrap();
        let current_equality_path = legacy_paged
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .and_then(|node| {
                node.equality_property_indices
                    .iter()
                    .find(|index| index.property == "body")
            })
            .expect("indexed body equality sidecar")
            .path
            .clone();
        let legacy_equality_path = format!("{current_equality_path}.legacy.bin");
        s.put(
            &Path::from(format!(
                "{}/{}",
                p.namespace_prefix().as_ref(),
                legacy_equality_path
            )),
            PutPayload::from(legacy_equality_body.clone()),
        )
        .await
        .unwrap();
        let node = legacy_paged
            .manifest
            .ssts
            .iter_mut()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        let equality = node
            .equality_property_indices
            .iter_mut()
            .find(|index| index.property == "body")
            .expect("indexed body equality sidecar");
        equality.path = legacy_equality_path;
        equality.size_bytes = legacy_equality_body.len() as u64;
        equality.format = crate::manifest::PropertyIndexFormat::BincodeV0;
        equality.paged = None;
        equality.paged_build_unsupported = false;
        let required = crate::flush::union_indexed_props(&schema);
        assert!(node_descriptor_needs_non_record_migration(node, &required));

        let migrated = compact_leveled(&ms, &fence, &legacy_paged, &schema, u64::MAX, 10)
            .await
            .unwrap()
            .committed;
        assert_eq!(
            migrated
                .manifest
                .ssts
                .iter()
                .find(|sst| sst.kind == SstKind::VectorGraph)
                .unwrap()
                .id,
            vector_id,
            "a non-search node migration must also retain the vector generation"
        );
        assert_eq!(
            migrated
                .manifest
                .ssts
                .iter()
                .find(|sst| sst.kind == SstKind::TextIndex)
                .unwrap()
                .id,
            text_id,
            "a non-search node migration must also retain the text generation"
        );
        assert_eq!(migrated.manifest.search_index_builds, build_states);
        assert!(!search_indexes_need_rebuild(&migrated.manifest));

        // The preserved bodies remain discoverable and serving through the
        // rewritten node generation, proving the freshness fence is intact.
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(migrated, &view, s, p);
        assert_eq!(
            snap.vector_search("doc_emb", &[1.0, 0.0], 2, 16)
                .await
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            snap.text_search("doc_text", "Doc", &parse_query("alpha"), Some(5))
                .await
                .unwrap()
                .expect("preserved text generation must serve")
                .len(),
            2
        );
    }
}
