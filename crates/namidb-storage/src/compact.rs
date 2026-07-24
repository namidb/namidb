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
//! decode one row group (nodes) / one partner block + property-stream
//! mini-batch (edges) at a time; a binary heap keyed by
//! `(key asc, lsn desc, source order)` picks the winner per key, shadowed
//! duplicates are skipped without ever being converted, and only winners
//! pay the row materialisation (for nodes, the JSON property-map
//! re-encode — typically 3-10x the Parquet size) on their way into a
//! bounded chunk buffer feeding the incremental SST writer. The
//! sidecar/stat harvesters and the vector/text index member collectors
//! observe the same winner stream, so nothing retains the merged bucket.
//!
//! Residual memory per bucket, by design: the **compressed** source bodies
//! (all sources must be open simultaneously; their sum is bounded by the
//! level budget), one decoded row group per node source, one chunk of
//! winner rows, the sidecar maps, and — the true lower bound — the
//! embeddings / documents collected for a vector/text index rebuild, which
//! the Vamana/BM25 builders inherently need in full.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashSet, VecDeque};
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, FixedSizeBinaryArray, ListArray, RecordBatch, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_ipc::reader::StreamReader;
use bytes::Bytes;
use chrono::Utc;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::file::metadata::ParquetMetaData;
use tracing::{debug, instrument};
use uuid::Uuid;

use namidb_core::{EdgeTypeDef, LabelDef, LabelDictionary, Schema, Value};

use crate::error::{Error, Result};
use crate::fence::WriterFence;
use crate::flush::{
    EqualitySidecarCollector, IncrementalNodeSstWriter, LabelIndexCollector, NodeRow,
    NodeWriteRecord, PerLabelStatsCollector, UniqueSidecarCollector, NODE_SST_BATCH_ROWS,
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
use crate::sst::bloom::{BloomDescriptor, BloomFilter};
use crate::sst::edges::encoding::{read_offset, read_partner_block, OffsetWidth};
use crate::sst::edges::format::{
    CODEC_NONE, CODEC_ZSTD, OVERFLOW_JSON_NAME, SECTION_KEY_IDS, SECTION_OFFSETS, SECTION_PARTNERS,
    SECTION_PER_EDGE_LSN, SECTION_PER_EDGE_TOMBSTONES, SECTION_PROPERTY_STREAM,
};
use crate::sst::edges::reader::EdgeSstReader;
use crate::sst::edges::writer::{EdgeRecord, EdgeSstFinish, EdgeSstWriter, EdgeSstWriterOptions};
use crate::sst::edges::EdgeDirection;
use crate::sst::nodes::{
    parse_node_sst_metadata, prop_column_name, NodeSstFinish, NodeSstReader, NodeSstWriterOptions,
    COL_LABELS, COL_LSN, COL_NODE_ID, COL_TOMBSTONE, OVERFLOW_JSON, SCHEMA_VERSION,
};

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
}

impl PreparedCompaction {
    /// `true` when the sweep found nothing to merge; installing is a no-op.
    pub fn is_noop(&self) -> bool {
        self.removed_ids.is_empty()
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

    /// Cheap, metadata-only "would a sweep merge anything?" predicate, so a
    /// maintenance tick can skip [`Self::prepare`] entirely on an idle
    /// namespace. NOTE: this is not `max_l0_bucket_len() >= 2` — a single
    /// L0 above an existing L1 (or a leveled-only over-budget cascade)
    /// still plans a merge.
    pub fn needs_compaction(&self) -> bool {
        any_bucket_plans(
            &self.base.manifest.ssts,
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
fn any_bucket_plans(ssts: &[SstDescriptor], base_bytes: u64, ratio: u64) -> bool {
    let mut buckets: std::collections::HashMap<(SstKind, &str), Vec<&SstDescriptor>> =
        std::collections::HashMap::new();
    for d in ssts {
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
    buckets
        .values()
        .any(|sources| plan_bucket_merge(sources, base_bytes, ratio).is_some())
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

    // Nodes.
    for (label, sources) in node_buckets {
        let Some(plan) = plan_bucket_merge(&sources, base_bytes, ratio) else {
            continue;
        };
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
        // GC tombstones only when this merge is authoritative: a single node
        // scope (no other scope can hold the key) AND the output is the
        // bucket's deepest level (no older un-merged level below it).
        let gc = node_gc_safe && plan.is_deepest;
        // GET every input body up front: the k-way merge needs each source
        // open simultaneously, but only as COMPRESSED bytes — the level
        // budget bounds their sum. Decoded rows never accumulate; the merge
        // streams them row-group by row-group.
        let mut bodies: Vec<Bytes> = Vec::with_capacity(plan.inputs.len());
        for desc in &plan.inputs {
            bodies.push(get_sst_body(store.as_ref(), paths, desc).await?);
        }
        // Vector/text member collection happens during the winner stream, and
        // is gated on the merge being authoritative for the FULL corpus:
        // deepest level AND a single node scope (`gc`). `plan.is_deepest`
        // alone treated a per-bucket deepest merge in a mixed-scope namespace
        // (legacy per-label + id-primary "" scopes) as corpus-complete,
        // rebuilding the index from one bucket and permanently truncating it
        // — the same rule node-tombstone GC uses. On a partial merge the
        // spec list stays empty: the existing `.vg`/`.ft` is left untouched
        // and the freshness gate (`index_outrun_by_nodes`) routes reads to
        // the exact flat scan until an authoritative merge rebuilds it.
        let index_specs = NodeMergeIndexSpecs {
            #[cfg(feature = "vector-index")]
            vector: if gc {
                base.manifest.vector_indexes.clone()
            } else {
                Vec::new()
            },
            #[cfg(feature = "text-index")]
            text: if gc {
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
        let out = run_cpu(move || {
            merge_node_sources(
                bodies,
                &merge_def,
                &merge_sidecar_def,
                gc,
                &merge_schema,
                &merge_dict,
                &merge_scope,
                index_specs,
            )
        })
        .await??;
        // The highest LSN in the merged corpus — stamped onto any SST-backed
        // index (vector / text) rebuilt from these rows so a later freshness
        // check can tell whether a newer Nodes SST has outrun the index.
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        let finish_max_lsn = out.finish.stats.max_lsn;
        if out.finish.stats.row_count == 0 {
            // Nothing to write; still mark the merged sources for removal so
            // the bucket truly shrinks.
            for src in &plan.inputs {
                removed_ids.push(src.id);
            }
            continue;
        }
        let (descriptor, wrote_bloom) = put_node_sst_leveled(
            store.clone(),
            paths,
            plan.target_level,
            &label,
            out.sidecars,
            out.finish,
        )
        .await?;
        if wrote_bloom {
            bloom_count += 1;
        }
        for src in &plan.inputs {
            removed_ids.push(src.id);
        }
        new_descs.push(descriptor);

        // RFC-030 (`vector-index`): rebuild Vamana indexes whose label has
        // nodes in this bucket. A graph is not row-mergeable, so any prior
        // VectorGraph SST for a rebuilt index is dropped (replaced) and the
        // fresh one built from the members the winner stream collected.
        #[cfg(feature = "vector-index")]
        {
            let (new_vg, old_vg_ids) = build_vector_indexes_from_members(
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
        }

        // (`text-index`): rebuild full-text indexes whose label has nodes in this
        // bucket, from the same winner stream (rebuild-not-merge).
        #[cfg(feature = "text-index")]
        {
            let (new_ft, old_ft_ids) = build_text_indexes_from_members(
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
        }
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

    Ok(PreparedCompaction {
        new_descs,
        removed_ids,
        bloom_count,
        base_version: base.manifest.version,
    })
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
    if prepared.removed_ids.is_empty() {
        debug!("compactor found no bucket worth merging; nothing to install");
        return Ok(CompactionOutcome {
            committed: current.clone(),
            source_ssts_removed: 0,
            new_ssts_written: 0,
            bloom_sidecars_written: 0,
        });
    }
    fence.assert_alive(current.manifest.epoch)?;

    let live: HashSet<Uuid> = current.manifest.ssts.iter().map(|d| d.id).collect();
    if let Some(missing) = prepared.removed_ids.iter().find(|id| !live.contains(id)) {
        return Err(Error::precondition(format!(
            "abandoning prepared compaction (basis v{}): input SST {missing} is no longer \
 referenced by manifest v{}; the prepared bodies are left for the orphan sweep",
            prepared.base_version, current.manifest.version
        )));
    }

    let source_count = prepared.removed_ids.len();
    let new_count = prepared.new_descs.len();
    let mut next = current.manifest.next_version(fence.writer_id);
    let removed_set: HashSet<Uuid> = prepared.removed_ids.into_iter().collect();
    next.ssts.retain(|d| !removed_set.contains(&d.id));
    next.ssts.extend(prepared.new_descs);
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
    // GET every source body up front (compressed bytes only; the level
    // budget bounds their sum), then k-way stream the merge on the blocking
    // pool — decoded partner blocks and property strings never accumulate
    // beyond the per-source cursor positions.
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

/// Sorted row cursor over one node source SST. Decodes ON DEMAND, one row
/// group at a time (the writer keeps `node_id` strictly ascending across
/// the SST, so cursor order is key order); at any moment only the current
/// row group's batches are resident. `row_groups_decoded` is the probe the
/// laziness tests assert on.
struct NodeSourceCursor {
    reader: NodeSstReader,
    md: Arc<ParquetMetaData>,
    next_row_group: usize,
    row_group_count: usize,
    /// Decoded batches of the CURRENT row group, front-first.
    views: VecDeque<NodeBatchView>,
    /// Row index into `views.front()`.
    row: usize,
    /// `(id, lsn)` of the current row; `None` once exhausted.
    current: Option<([u8; 16], u64)>,
    /// Row groups decoded so far (test probe).
    row_groups_decoded: usize,
    /// Total row count per the Parquet footer (bloom sizing upper bound).
    total_rows: u64,
}

impl NodeSourceCursor {
    fn open(label_def: &LabelDef, body: Bytes) -> Result<Self> {
        let md = parse_node_sst_metadata(&body)?;
        let reader = NodeSstReader::open(label_def.clone(), body)?;
        let row_group_count = md.num_row_groups();
        let total_rows = md.file_metadata().num_rows().max(0) as u64;
        let mut cursor = Self {
            reader,
            md,
            next_row_group: 0,
            row_group_count,
            views: VecDeque::new(),
            row: 0,
            current: None,
            row_groups_decoded: 0,
            total_rows,
        };
        cursor.position()?;
        Ok(cursor)
    }

    /// Advance `views`/`row` to the next available row (decoding further row
    /// groups as needed) and cache its key in `current`.
    fn position(&mut self) -> Result<()> {
        loop {
            if let Some(front) = self.views.front() {
                if self.row < front.len() {
                    self.current = Some(front.key(self.row)?);
                    return Ok(());
                }
                self.views.pop_front();
                self.row = 0;
                continue;
            }
            if self.next_row_group >= self.row_group_count {
                self.current = None;
                return Ok(());
            }
            let rg = self.next_row_group;
            self.next_row_group += 1;
            self.row_groups_decoded += 1;
            for (_, batches) in self.reader.scan_row_groups_each(&self.md, &[rg])? {
                for batch in batches {
                    if batch.num_rows() > 0 {
                        self.views.push_back(NodeBatchView::new(batch)?);
                    }
                }
            }
            self.row = 0;
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
            .views
            .front()
            .ok_or_else(|| Error::invariant("node merge cursor materialised past its end"))?;
        view.materialize(self.row, label_def)
    }

    fn advance(&mut self) -> Result<()> {
        self.row += 1;
        self.position()
    }
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

/// Everything the streaming node merge harvests from the winner stream for
/// [`put_node_sst_leveled`], in place of the old `&merged_rows` re-walks.
struct NodeSidecarHarvest {
    unique: UniqueSidecarCollector,
    equality: EqualitySidecarCollector,
    label_index: LabelIndexCollector,
    per_label_property_stats: Vec<PerLabelPropertyStat>,
}

/// Per-index `(descriptor, collected (id, embedding) members)` pairs the
/// winner stream produced. Aliased for clippy's type-complexity lint.
#[cfg(feature = "vector-index")]
type VectorIndexMembers = Vec<(VectorIndexDescriptor, Vec<([u8; 16], Vec<f32>)>)>;

/// Per-index `(descriptor, collected (id, document) members)` pairs the
/// winner stream produced. Aliased for clippy's type-complexity lint.
#[cfg(feature = "text-index")]
type TextIndexMembers = Vec<(
    crate::manifest::TextIndexDescriptor,
    Vec<([u8; 16], String)>,
)>;

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
    bodies: Vec<Bytes>,
    label_def: &LabelDef,
    sidecar_def: &LabelDef,
    gc_tombstones: bool,
    schema: &Schema,
    label_dict: &LabelDictionary,
    bucket_scope: &str,
    index_specs: NodeMergeIndexSpecs,
) -> Result<NodeMergeOutput> {
    let mut cursors: Vec<NodeSourceCursor> = Vec::with_capacity(bodies.len());
    let mut total_rows: u64 = 0;
    for body in bodies {
        let cursor = NodeSourceCursor::open(label_def, body)?;
        total_rows = total_rows.saturating_add(cursor.total_rows);
        cursors.push(cursor);
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
    let mut unique = UniqueSidecarCollector::new(sidecar_def);
    let mut equality = EqualitySidecarCollector::new(sidecar_def);
    let mut label_index = LabelIndexCollector::new();
    let mut stats = PerLabelStatsCollector::new();
    #[cfg(feature = "vector-index")]
    let mut vector_collectors: Vec<VectorMemberCollector> = index_specs
        .vector
        .into_iter()
        .map(|desc| VectorMemberCollector::new(desc, label_dict))
        .collect();
    #[cfg(feature = "text-index")]
    let mut text_collectors: Vec<TextMemberCollector> = index_specs
        .text
        .into_iter()
        .map(|desc| TextMemberCollector::new(desc, label_dict))
        .collect();
    #[cfg(not(any(feature = "vector-index", feature = "text-index")))]
    let _ = &index_specs;

    let mut heap: BinaryHeap<Reverse<NodeHeapEntry>> = BinaryHeap::with_capacity(cursors.len());
    for (src, cursor) in cursors.iter().enumerate() {
        if let Some((id, lsn)) = cursor.peek() {
            heap.push(Reverse(NodeHeapEntry { id, lsn, src }));
        }
    }

    let mut last_id: Option<[u8; 16]> = None;
    while let Some(Reverse(entry)) = heap.pop() {
        let cursor = &mut cursors[entry.src];
        if last_id != Some(entry.id) {
            // First (highest-LSN) observation of this id: the winner.
            last_id = Some(entry.id);
            let (row, rec) = cursor.materialize_current(label_def)?;
            if !(gc_tombstones && matches!(row.op, MemOp::Tombstone)) {
                if let Some(rec) = &rec {
                    unique.observe(row.id, rec);
                    equality.observe(row.id, rec);
                    label_index.observe(row.id, rec);
                    stats.observe(rec);
                    #[cfg(feature = "vector-index")]
                    for collector in &mut vector_collectors {
                        collector.observe(row.id, rec, bucket_scope);
                    }
                    #[cfg(feature = "text-index")]
                    for collector in &mut text_collectors {
                        collector.observe(row.id, rec, bucket_scope);
                    }
                }
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

    Ok(NodeMergeOutput {
        finish: writer.finish()?,
        sidecars: NodeSidecarHarvest {
            unique,
            equality,
            label_index,
            per_label_property_stats: stats.finish(schema, label_dict)?,
        },
        #[cfg(feature = "vector-index")]
        vector_members: vector_collectors
            .into_iter()
            .map(|c| (c.desc, c.members))
            .collect(),
        #[cfg(feature = "text-index")]
        text_members: text_collectors
            .into_iter()
            .map(|c| (c.desc, c.members))
            .collect(),
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

/// Incremental reader over one edge property stream (Arrow IPC of
/// JSON-encoded `Value` strings, optionally zstd-compressed). Yields values
/// in edge-enumeration order one mini-batch at a time — the zstd frame is
/// decoded through a streaming `Read`, so at no point does the whole
/// stream's `String` set exist in memory.
struct PropertyStreamCursor {
    name: String,
    reader: StreamReader<Box<dyn std::io::Read + Send>>,
    current: Option<StringArray>,
    row: usize,
    rows_read: u64,
    edge_count: u64,
}

impl PropertyStreamCursor {
    fn open(name: &str, bytes: Bytes, codec: u8, edge_count: u64) -> Result<Self> {
        let read: Box<dyn std::io::Read + Send> = match codec {
            CODEC_NONE => Box::new(std::io::Cursor::new(bytes)),
            CODEC_ZSTD => Box::new(
                zstd::stream::read::Decoder::new(std::io::Cursor::new(bytes)).map_err(|e| {
                    Error::invariant(format!("zstd decode (property stream {name}): {e}"))
                })?,
            ),
            other => {
                return Err(Error::Corrupted {
                    path: "<edges>".into(),
                    detail: format!("unknown codec {other} for property stream {name}"),
                });
            }
        };
        let reader = StreamReader::try_new(read, None)
            .map_err(|e| Error::invariant(format!("property IPC reader ({name}): {e}")))?;
        Ok(Self {
            name: name.to_string(),
            reader,
            current: None,
            row: 0,
            rows_read: 0,
            edge_count,
        })
    }

    /// Value for the next edge in enumeration order. `want == false` (a
    /// shadowed loser) still advances the stream but skips materialising
    /// the string.
    fn next(&mut self, want: bool) -> Result<Option<String>> {
        loop {
            if let Some(current) = &self.current {
                if self.row < current.len() {
                    let out = if want && !current.is_null(self.row) {
                        Some(current.value(self.row).to_string())
                    } else {
                        None
                    };
                    self.row += 1;
                    self.rows_read += 1;
                    return Ok(out);
                }
                self.current = None;
            }
            match self.reader.next() {
                Some(batch) => {
                    let batch = batch.map_err(|e| {
                        Error::invariant(format!("property IPC batch ({}): {e}", self.name))
                    })?;
                    let column = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            Error::invariant(format!(
                                "property IPC column ({}) is not Utf8",
                                self.name
                            ))
                        })?
                        .clone();
                    self.current = Some(column);
                    self.row = 0;
                }
                None => {
                    return Err(Error::Corrupted {
                        path: "<edges>".into(),
                        detail: format!(
                            "property stream {} row count {} != edge_count {}",
                            self.name, self.rows_read, self.edge_count
                        ),
                    });
                }
            }
        }
    }

    /// After the cursor consumed exactly `edge_count` values, the stream
    /// must be empty too — the streaming equivalent of the whole-stream
    /// row-count check the materialised decode performed.
    fn assert_exhausted(&mut self) -> Result<()> {
        let leftover = match &self.current {
            Some(current) if self.row < current.len() => true,
            _ => match self.reader.next() {
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
) -> Result<EdgeSstFinish> {
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
    // and the skew threshold slightly conservatively.
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

    writer.finish()
}

// ── PUT helpers (L1 variants) ───────────────────────────────────────────

async fn put_node_sst_leveled(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    label: &str,
    // Sidecar/stat harvest the streaming merge collected off the winner
    // stream (unique + equality maps keyed by the sidecar def — for the
    // id-primary "" bucket that's `union_indexed_props(schema)`, mirroring
    // flush; for legacy per-label buckets the label's own def — plus the
    // label-index postings and the RFC-025 per-(label, property) stats).
    sidecars: NodeSidecarHarvest,
    finish: NodeSstFinish,
) -> Result<(SstDescriptor, bool)> {
    let id = Uuid::now_v7();
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
    let (unique_property_indices, mut index_sidecars) =
        sidecars.unique.finish(paths, level.as_u32(), &id, label)?;
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
    if let Some(sidecar) = label_sidecar {
        index_sidecars.push(sidecar);
    }
    for (path, body) in &index_sidecars {
        crate::flush::put_object(store.clone(), path, body.clone()).await?;
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
        label_index,
        // Per-(label, property) stats recomputed off the winner stream so
        // they survive L0->L1 the same way the label index does (RFC 025).
        per_label_property_stats: sidecars.per_label_property_stats,
    };
    Ok((descriptor, wrote_bloom))
}

/// Streaming member collector for one registered vector index: observes
/// each reconciled winner row as the merge advances and keeps the
/// `(id, embedding)` pairs the Vamana build needs — the inherent lower
/// bound, since graph construction requires every member vector at once.
/// Only instantiated for authoritative merges (see the caller): on a
/// partial merge the winner stream is a strict subset of the corpus and
/// rebuilding from it would silently truncate the index.
#[cfg(feature = "vector-index")]
struct VectorMemberCollector {
    desc: VectorIndexDescriptor,
    /// The index label resolved to its raw dictionary id (if interned).
    label_id: Option<u32>,
    members: Vec<([u8; 16], Vec<f32>)>,
}

#[cfg(feature = "vector-index")]
impl VectorMemberCollector {
    fn new(desc: VectorIndexDescriptor, label_dict: &LabelDictionary) -> Self {
        Self {
            label_id: label_dict.id(&desc.label).map(|lid| lid.0),
            desc,
            members: Vec::new(),
        }
    }

    fn observe(&mut self, id: [u8; 16], rec: &NodeWriteRecord, bucket_scope: &str) {
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
            return;
        }
        let Some(val) = rec.properties.get(&self.desc.property) else {
            return;
        };
        let v: Vec<f32> = match val {
            Value::Vec(v) => v.clone(),
            Value::VecI8 { codes, scale } => codes.iter().map(|&c| c as f32 * *scale).collect(),
            _ => return,
        };
        self.members.push((id, v));
    }
}

/// Streaming member collector for one registered full-text index: keeps
/// `(id, concatenated document)` pairs for the BM25 build (which needs the
/// whole corpus for its N / avgdl / df statistics — the text analogue of
/// the vector lower bound). Same label-filter and authority rules as
/// [`VectorMemberCollector`].
#[cfg(feature = "text-index")]
struct TextMemberCollector {
    desc: crate::manifest::TextIndexDescriptor,
    label_id: Option<u32>,
    members: Vec<([u8; 16], String)>,
}

#[cfg(feature = "text-index")]
impl TextMemberCollector {
    fn new(desc: crate::manifest::TextIndexDescriptor, label_dict: &LabelDictionary) -> Self {
        Self {
            label_id: label_dict.id(&desc.label).map(|lid| lid.0),
            desc,
            members: Vec::new(),
        }
    }

    fn observe(&mut self, id: [u8; 16], rec: &NodeWriteRecord, bucket_scope: &str) {
        // Same legacy-row fallback as the vector collector above.
        let carries_label = match self.label_id {
            Some(lid) => {
                rec.labels.contains(&lid)
                    || (rec.labels.is_empty() && bucket_scope == self.desc.label)
            }
            None => rec.labels.is_empty() && bucket_scope == self.desc.label,
        };
        if !carries_label {
            return;
        }
        // Concatenate the indexed properties' string values into one document.
        let mut parts: Vec<&str> = Vec::new();
        for prop in &self.desc.properties {
            if let Some(Value::Str(s)) = rec.properties.get(prop) {
                parts.push(s.as_str());
            }
        }
        if parts.is_empty() {
            return; // not a member of this index's corpus
        }
        self.members.push((id, parts.join(" ")));
    }
}

/// RFC-030 (`vector-index`): for every registered index, build a fresh
/// Vamana `VectorGraph` SST from the members the winner stream collected
/// and return the new descriptors **plus** the ids of any prior VectorGraph
/// SSTs for the same index (which must be removed — a graph is not
/// row-mergeable, so compaction rebuilds rather than merges). Indexes whose
/// label had no nodes in the bucket, or with fewer than two live
/// embeddings, yield nothing (and keep their prior SST).
///
/// The authority gate lives at the collection site (`prepare_leveled`): on
/// a partial (non-authoritative) merge no members are collected, this
/// receives an empty list, and the existing `.vg` is left untouched — the
/// freshness gate (`index_outrun_by_nodes`) detects the now-newer Nodes SST
/// and falls back to the exact flat scan until an authoritative merge
/// rebuilds the index.
#[cfg(feature = "vector-index")]
async fn build_vector_indexes_from_members(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    corpus_max_lsn: u64,
    collected: VectorIndexMembers,
    old_vector_by_scope: &BTreeMap<String, Vec<&SstDescriptor>>,
) -> Result<(Vec<SstDescriptor>, Vec<Uuid>)> {
    use crate::sst::vector::build_body;

    let mut new_descs = Vec::new();
    let mut removed = Vec::new();

    for (desc, members) in collected {
        // Skip-and-warn on a per-index build error (e.g. a malformed descriptor)
        // rather than `?`-aborting the whole compaction — one bad index must
        // never wedge the namespace's compaction permanently. The Vamana
        // construction is O(n·R·L·dim) pure CPU, so it runs on the blocking
        // pool instead of stalling the async runtime for its duration.
        let build_desc = desc.clone();
        let built = match run_cpu(move || build_body(&build_desc, members)).await? {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(index = %desc.name, error = %e, "skipping vector index build");
                continue;
            }
        };
        let Some((body, stats)) = built else {
            continue;
        };

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
        let body_len = body.len() as u64;
        crate::flush::put_object(store.clone(), &object_path, body).await?;

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
            label_index: None,
            per_label_property_stats: vec![],
        };
        new_descs.push(descriptor);

        // Rebuild-not-merge: drop prior VectorGraph SSTs for this index.
        if let Some(old) = old_vector_by_scope.get(&desc.name) {
            removed.extend(old.iter().map(|d| d.id));
        }
    }

    Ok((new_descs, removed))
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
/// rather than truncating the index to the shallow subset.
#[cfg(feature = "text-index")]
async fn build_text_indexes_from_members(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    corpus_max_lsn: u64,
    collected: TextIndexMembers,
    old_text_by_scope: &BTreeMap<String, Vec<&SstDescriptor>>,
) -> Result<(Vec<SstDescriptor>, Vec<Uuid>)> {
    use crate::sst::text::build_body;

    let mut new_descs = Vec::new();
    let mut removed = Vec::new();

    for (desc, members) in collected {
        // BM25 postings construction is pure CPU over the collected corpus;
        // run it on the blocking pool like the Vamana build above.
        let Some((body, stats)) = run_cpu(move || build_body(members)).await?? else {
            continue;
        };

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
        let body_len = body.len() as u64;
        crate::flush::put_object(store.clone(), &object_path, body).await?;

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
            label_index: None,
            per_label_property_stats: vec![],
        };
        new_descs.push(descriptor);

        // Rebuild-not-merge: drop prior TextIndex SSTs for this index.
        if let Some(old) = old_text_by_scope.get(&desc.name) {
            removed.extend(old.iter().map(|d| d.id));
        }
    }

    Ok((new_descs, removed))
}

async fn put_edge_sst_leveled(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    out_level: u32,
    edge_type: &str,
    direction: EdgeDirection,
    finish: EdgeSstFinish,
) -> Result<(SstDescriptor, bool)> {
    let id = Uuid::now_v7();
    let level = SstLevel(out_level);
    let kind = match direction {
        EdgeDirection::Forward => SstKind::EdgesFwd,
        EdgeDirection::Inverse => SstKind::EdgesInv,
    };
    let file_name = format!(
        "{}-{}-{}.csr",
        uuid_path_id(&id),
        direction.path_tag(),
        edge_type
    );
    let object_path = paths.sst_object(level.as_u32(), &file_name);
    let relative_path = relative_sst_path(level.as_u32(), &file_name);

    let body = finish.body;
    let body_len = body.len() as u64;
    crate::flush::put_object(store.clone(), &object_path, body).await?;

    let (bloom_descriptor, wrote_bloom) = put_bloom_sidecar(
        store,
        paths,
        level.as_u32(),
        &id,
        direction.path_tag(),
        edge_type,
        finish.bloom,
    )
    .await?;

    let stats = finish.stats;
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
        label_index: None,
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
    let body = res.bytes().await?;
    Ok(body)
}

fn uuid_path_id(u: &Uuid) -> String {
    u.simple().to_string()
}

fn relative_sst_path(level: u32, file_name: &str) -> String {
    format!("sst/level{level}/{file_name}")
}

#[cfg(test)]
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
        props_ac.insert("since".into(), Value::I64(2024));
        props_ac.insert("weight".into(), Value::F64(0.9));

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
        use crate::sst::text::TextIndex;

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
        let fts: Vec<&SstDescriptor> = out
            .committed
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::TextIndex)
            .collect();
        assert_eq!(fts.len(), 1, "exactly one TextIndex SST after compaction");
        assert_eq!(fts[0].scope, "note_ft");
        let doc_count = match &fts[0].kind_specific {
            KindSpecificStats::TextIndex { doc_count, .. } => *doc_count,
            _ => 0,
        };
        assert_eq!(doc_count, bodies.len() as u64, "all docs indexed");

        // Decode + search: the rare-term doc must rank first via real IDF.
        let body = get_sst_body(s.as_ref(), &p, fts[0]).await.unwrap();
        let idx = TextIndex::decode(&body).unwrap();
        let hits = idx.search(&crate::text::tokenize("fox common"), None);
        assert_eq!(hits.len(), bodies.len(), "every doc matches a query term");
        assert_eq!(
            hits[0].0,
            *idx_id(1).as_bytes(),
            "the rare-term doc ranks first"
        );
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
            assert_eq!(
                b1, b2,
                "{:?}/{} bodies must be byte-identical",
                d1.kind, d1.scope
            );
        }
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
    fn node_cursor_decodes_row_groups_lazily() {
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

        let mut cursor = NodeSourceCursor::open(&label, finish.body).unwrap();
        assert_eq!(cursor.row_group_count, 4, "fixture must be multi-row-group");
        assert_eq!(
            cursor.row_groups_decoded, 1,
            "open decodes only the first row group, not the whole body"
        );
        let mut seen = 0u8;
        while let Some((id, lsn)) = cursor.peek() {
            seen += 1;
            assert_eq!(id, *sorted_node_id(seen).as_bytes());
            assert_eq!(lsn, seen as u64);
            if seen <= 4 {
                assert_eq!(
                    cursor.row_groups_decoded, 1,
                    "rows of the first group must not trigger further decodes"
                );
            }
            cursor.advance().unwrap();
        }
        assert_eq!(seen, 16);
        assert_eq!(
            cursor.row_groups_decoded, 4,
            "all groups decoded exactly on demand"
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

        let out = merge_node_sources(
            vec![body_a, body_b],
            &label,
            &label,
            true,
            &schema(),
            &LabelDictionary::new(),
            "",
            NodeMergeIndexSpecs::default(),
        )
        .unwrap();
        assert_eq!(out.finish.stats.row_count, 30);
        assert_eq!(out.finish.stats.min_lsn, 101);
        assert_eq!(out.finish.stats.max_lsn, 230);

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
        use crate::sst::vector::VectorGraphIndex;
        use rand::Rng;
        use rand::SeedableRng;

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
        let vgs: Vec<&SstDescriptor> = out
            .committed
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == SstKind::VectorGraph)
            .collect();
        assert_eq!(vgs.len(), 1, "exactly one VectorGraph SST after compaction");
        assert_eq!(vgs[0].scope, "doc_emb");
        let stats = match &vgs[0].kind_specific {
            KindSpecificStats::VectorGraph { point_count, .. } => *point_count,
            _ => 0,
        };
        assert_eq!(stats, 160, "all 160 docs indexed");

        // Decode + search; a query near centroid 0 must surface cluster-0 docs.
        let body = get_sst_body(s.as_ref(), &p, vgs[0]).await.unwrap();
        let idx = VectorGraphIndex::decode(&body).unwrap();
        assert_eq!(idx.point_count(), 160);

        let mut q = centroids[0].clone();
        normalize_inplace(&mut q);
        let hits = idx.search(&q, 10, 48);
        assert_eq!(hits.len(), 10);
        // The query sits on centroid 0; the true top-10 are cluster-0 docs.
        // Count how many returned hits belong to cluster 0 (recall proxy).
        let cluster0_hits = hits
            .iter()
            .filter(|(id, _)| cluster_of.get(&NodeId::from_uuid(Uuid::from_bytes(*id))) == Some(&0))
            .count();
        assert!(
            cluster0_hits >= 8,
            "expected >= 8/10 hits from cluster 0, got {cluster0_hits}"
        );
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
        let vg = manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::VectorGraph)
            .expect("authoritative merge rebuilds the .vg");
        let ft = manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::TextIndex)
            .expect("authoritative merge rebuilds the .ft");
        assert_eq!(vg.row_count, 11, "12 docs - 1 tombstone, update deduped");
        assert_eq!(ft.row_count, 11);
        // Freshness stamps: both indexes carry the merged corpus's
        // high-water LSN, so the gate lets them serve.
        assert_eq!(vg.max_lsn, node_desc.max_lsn);
        assert_eq!(ft.max_lsn, node_desc.max_lsn);
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
}
