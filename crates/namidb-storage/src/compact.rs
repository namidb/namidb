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

use std::collections::{BTreeMap, HashSet};

use arrow_array::{
    Array, BooleanArray, FixedSizeBinaryArray, ListArray, StringArray, UInt32Array, UInt64Array,
};
use bytes::Bytes;
use chrono::Utc;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use tracing::{debug, instrument};
use uuid::Uuid;

use namidb_core::{LabelDef, LabelDictionary, Schema, Value};

use crate::error::{Error, Result};
use crate::fence::WriterFence;
use crate::flush::{build_edge_sst, build_node_sst, NodeRow, NodeWriteRecord};
use crate::manifest::{
    KindSpecificStats, LoadedManifest, ManifestStore, SstDescriptor, SstKind, SstLevel,
};
use crate::memtable::MemOp;
use crate::paths::NamespacePaths;
use crate::read::arrow_value_to_value;
use crate::sst::bloom::{BloomDescriptor, BloomFilter};
use crate::sst::edges::reader::EdgeSstReader;
use crate::sst::edges::writer::{EdgeRecord, EdgeSstFinish};
use crate::sst::edges::EdgeDirection;
use crate::sst::nodes::{
    prop_column_name, NodeSstFinish, NodeSstReader, COL_LABELS, COL_LSN, COL_NODE_ID,
    COL_TOMBSTONE, OVERFLOW_JSON, SCHEMA_VERSION,
};

/// Outcome of [`compact_l0_to_l1`].
#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    pub committed: LoadedManifest,
    pub source_ssts_removed: usize,
    pub new_ssts_written: usize,
    pub bloom_sidecars_written: usize,
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

/// Run one leveled-lite compaction sweep with explicit level budgets. The
/// public [`compact_l0_to_l1`] wraps this with the environment-configured
/// budgets; tests call it directly with small budgets to exercise the
/// cascade deterministically without touching process-wide env.
#[instrument(
 skip(manifest_store, fence, base, schema),
 fields(
 namespace = %manifest_store.paths().namespace(),
 base_version = base.manifest.version,
 )
)]
async fn compact_leveled(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    base: &LoadedManifest,
    schema: &Schema,
    base_bytes: u64,
    ratio: u64,
) -> Result<CompactionOutcome> {
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
        let mut readers: Vec<NodeSstReader> = Vec::with_capacity(plan.inputs.len());
        for desc in &plan.inputs {
            let body = get_sst_body(store.as_ref(), paths, desc).await?;
            readers.push(NodeSstReader::open(label_def.clone(), body)?);
        }
        // GC tombstones only when this merge is authoritative: a single node
        // scope (no other scope can hold the key) AND the output is the
        // bucket's deepest level (no older un-merged level below it).
        let gc = node_gc_safe && plan.is_deepest;
        let (finish, merged_rows) = compact_node_ssts(&label_def, &readers, gc)?;
        if finish.stats.row_count == 0 {
            // Nothing to write; still mark the merged sources for removal so
            // the bucket truly shrinks.
            for src in &plan.inputs {
                removed_ids.push(src.id);
            }
            continue;
        }
        let (descriptor, wrote_bloom) = put_node_sst_leveled(
            store.as_ref(),
            paths,
            plan.target_level,
            &label,
            &label_def,
            &merged_rows,
            finish,
            schema,
            &base.manifest.label_dict,
        )
        .await?;
        if wrote_bloom {
            bloom_count += 1;
        }
        for src in &plan.inputs {
            removed_ids.push(src.id);
        }
        new_descs.push(descriptor);
    }

    // Edges (forward).
    for (edge_type, sources) in fwd_buckets {
        let Some(plan) = plan_bucket_merge(&sources, base_bytes, ratio) else {
            continue;
        };
        let (desc, wrote_bloom, removed) = compact_and_write_edges(
            store.as_ref(),
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
            store.as_ref(),
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
    // bucket, the index must be *rebuilt* from the current merged node rows,
    // not merged graph-to-graph. The rebuild hook (`vector_index::rebuild`)
    // is feature-gated; off-feature no VectorGraph SST is ever written, so this
    // bucket is empty and the pass-through below is a no-op. Either way,
    // surviving VectorGraph SSTs (un-touched here) carry forward via
    // `next.ssts.retain`.
    #[cfg(feature = "vector-index")]
    {
        // On-feature rebuild lands in Step 5; until then these are rebuilt by
        // the dedicated hook invoked from the node-bucket loop. Any leftover
        // descriptors here pass through unchanged.
        for (_index_name, _sources) in &vector_buckets {}
    }
    #[cfg(not(feature = "vector-index"))]
    {
        let _ = &vector_buckets;
    }

    if removed_ids.is_empty() {
        debug!("compactor found no L0 bucket with >1 SSTs; nothing to do");
        return Ok(CompactionOutcome {
            committed: base.clone(),
            source_ssts_removed: 0,
            new_ssts_written: 0,
            bloom_sidecars_written: 0,
        });
    }

    let source_count = removed_ids.len();
    let new_count = new_descs.len();
    let mut next = base.manifest.next_version(fence.writer_id);
    let removed_set: HashSet<Uuid> = removed_ids.into_iter().collect();
    next.ssts.retain(|d| !removed_set.contains(&d.id));
    next.ssts.extend(new_descs);
    let committed = manifest_store.commit(fence, base, next).await?;

    Ok(CompactionOutcome {
        committed,
        source_ssts_removed: source_count,
        new_ssts_written: new_count,
        bloom_sidecars_written: bloom_count,
    })
}

#[allow(clippy::too_many_arguments)]
async fn compact_and_write_edges(
    store: &dyn ObjectStore,
    paths: &NamespacePaths,
    schema: &Schema,
    edge_type: &str,
    sources: &[&SstDescriptor],
    direction: EdgeDirection,
    level: u32,
    gc_tombstones: bool,
) -> Result<(Option<SstDescriptor>, bool, Vec<Uuid>)> {
    let edge_def = schema.edge_type(edge_type).cloned();
    let mut readers: Vec<EdgeSstReader> = Vec::with_capacity(sources.len());
    for desc in sources {
        let body = get_sst_body(store, paths, desc).await?;
        readers.push(EdgeSstReader::open(body)?);
    }
    let finish = compact_edge_ssts(
        edge_type,
        edge_def.as_ref(),
        &readers,
        direction,
        gc_tombstones,
    )?;
    let removed: Vec<Uuid> = sources.iter().map(|d| d.id).collect();
    if finish.stats.edge_count == 0 {
        return Ok((None, false, removed));
    }
    let (descriptor, wrote_bloom) =
        put_edge_sst_leveled(store, paths, level, edge_type, direction, finish).await?;
    Ok((Some(descriptor), wrote_bloom, removed))
}

fn compact_node_ssts(
    label_def: &LabelDef,
    sources: &[NodeSstReader],
    gc_tombstones: bool,
) -> Result<(NodeSstFinish, Vec<NodeRow>)> {
    let mut rows: Vec<NodeRow> = Vec::new();
    for reader in sources {
        rows.extend(extract_node_rows_from_reader(reader, label_def)?);
    }
    // Sort by node_id ascending; within ties, highest LSN first so the
    // dedup_by_key below preserves the winner.
    rows.sort_by(|a, b| a.id.cmp(&b.id).then(b.lsn.cmp(&a.lsn)));
    rows.dedup_by_key(|r| r.id);
    // Tombstone GC (RFC-027 P3): a key whose winning op is a tombstone has
    // no live value, and when this merge is authoritative for the node
    // keyspace (`gc_tombstones`, see the caller) it is the only SST that
    // could hold the key at the new manifest version, so the tombstone
    // shadows nothing and is dropped entirely. Readers pinned at older
    // versions still see the delete through the retained source bodies (the
    // horizon-aware sweep keeps them). This is what stops a delete-heavy
    // workload from carrying its tombstones forever.
    if gc_tombstones {
        rows.retain(|r| !matches!(r.op, MemOp::Tombstone));
    }
    let finish = build_node_sst(label_def, &rows)?;
    // Caller (`put_node_sst_l1`) consumes the merged rows to re-emit
    // the unique-property side-cars; without this, post-compaction
    // lookups on `(label, unique_prop)` fall back to the legacy full
    // scan (P4.19 sidecar emission only happened in flush).
    Ok((finish, rows))
}

fn extract_node_rows_from_reader(
    reader: &NodeSstReader,
    label_def: &LabelDef,
) -> Result<Vec<NodeRow>> {
    let batches = reader.scan()?;
    let mut out: Vec<NodeRow> = Vec::new();
    for batch in batches {
        let id_col = batch
            .column_by_name(COL_NODE_ID)
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
            .ok_or_else(|| Error::invariant("node_id column missing"))?;
        let tomb_col = batch
            .column_by_name(COL_TOMBSTONE)
            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
            .ok_or_else(|| Error::invariant("tombstone column missing"))?;
        let lsn_col = batch
            .column_by_name(COL_LSN)
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| Error::invariant("lsn column missing"))?;
        let ovf_col = batch
            .column_by_name(OVERFLOW_JSON)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Error::invariant("__overflow_json column missing"))?;
        let sv_col = batch
            .column_by_name(SCHEMA_VERSION)
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| Error::invariant("__schema_version column missing"))?;

        for row in 0..batch.num_rows() {
            let id: [u8; 16] = id_col
                .value(row)
                .try_into()
                .map_err(|_| Error::invariant("node_id row length != 16"))?;
            let lsn = lsn_col.value(row);
            let tomb = tomb_col.value(row);
            if tomb {
                out.push(NodeRow {
                    id,
                    lsn,
                    op: MemOp::Tombstone,
                });
                continue;
            }

            // Rebuild properties: declared columns + overflow_json.
            let mut properties: BTreeMap<String, Value> = BTreeMap::new();
            for p in &label_def.properties {
                let col_name = prop_column_name(p);
                let col = batch
                    .column_by_name(&col_name)
                    .ok_or_else(|| Error::invariant(format!("missing column {col_name}")))?;
                if let Some(v) = arrow_value_to_value(col.as_ref(), row, &p.data_type)? {
                    properties.insert(p.name.clone(), v);
                }
            }
            if !ovf_col.is_null(row) {
                let extra: BTreeMap<String, Value> = serde_json::from_str(ovf_col.value(row))?;
                properties.extend(extra);
            }
            let schema_version = sv_col.value(row);
            let payload = NodeWriteRecord {
                properties,
                schema_version,
                // Preserve the on-row label set (raw LabelIds) so the merged L1
                // SST keeps it. Legacy SSTs have no __labels column and yield an
                // empty set; their L1 stays scope-typed and reads via fallback.
                labels: raw_labels_from_batch(&batch, row),
            }
            .encode()?;
            out.push(NodeRow {
                id,
                lsn,
                op: MemOp::Upsert(payload),
            });
        }
    }
    Ok(out)
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

fn compact_edge_ssts(
    edge_type: &str,
    edge_def: Option<&namidb_core::EdgeTypeDef>,
    sources: &[EdgeSstReader],
    direction: EdgeDirection,
    gc_tombstones: bool,
) -> Result<EdgeSstFinish> {
    // Each source SST is already in `direction` orientation (the
    // caller groups by SstKind::EdgesFwd vs EdgesInv before invoking
    // us), so we can read `(key_id, partner_id)` straight from the
    // scan and pass them to the writer unchanged.
    //
    // Property streams: each source SST exposes
    // - `__overflow_json` (a single Utf8 column of ad-hoc / undeclared
    // properties) via `read_overflow_strings`, and
    // - one declared property stream per `EdgeTypeDef.properties` name
    // (RFC-002 §3.2.7) via `read_declared_property_strings(name)`.
    // The compactor reads both kinds and threads them through to the
    // L1 SST so reads at the merged version see the same property maps.
    let declared_property_names: Vec<String> = edge_def
        .map(|def| def.properties.iter().map(|p| p.name.clone()).collect())
        .unwrap_or_default();
    let mut rows: Vec<EdgeRecord> = Vec::new();
    for reader in sources {
        let edges = reader.scan_all_edges()?;
        let overflows = reader.read_overflow_strings()?;
        // For each declared property name, fetch the stream once per
        // reader; `None` if this SST has no such stream (legacy
        // pre-RFC-005 body, or all-null column).
        let mut declared_streams: Vec<Option<Vec<Option<String>>>> =
            Vec::with_capacity(declared_property_names.len());
        for name in &declared_property_names {
            declared_streams.push(reader.read_declared_property_strings(name)?);
        }
        for (idx, e) in edges.into_iter().enumerate() {
            let overflow_json = overflows
                .as_ref()
                .and_then(|v| v.get(idx).cloned())
                .flatten();
            let declared_properties: Vec<Option<String>> = declared_streams
                .iter()
                .map(|s| s.as_ref().and_then(|v| v.get(idx).cloned()).flatten())
                .collect();
            rows.push(EdgeRecord {
                key_id: e.key_id,
                partner_id: e.partner_id,
                lsn: e.lsn,
                tombstone: e.tombstone,
                declared_properties,
                overflow_json,
            });
        }
    }

    // Sort by (key, partner) asc, then by lsn desc so the dedup keeps
    // the highest-LSN observation per (key, partner).
    rows.sort_by(|a, b| {
        a.key_id
            .cmp(&b.key_id)
            .then(a.partner_id.cmp(&b.partner_id))
            .then(b.lsn.cmp(&a.lsn))
    });
    rows.dedup_by_key(|r| (r.key_id, r.partner_id));
    // Tombstone GC (RFC-027 P3), same reasoning as the node path: when this
    // merge is authoritative for the (edge_type, direction) bucket — its
    // output is the bucket's deepest level — a winning tombstone shadows
    // nothing and is dropped. Otherwise a deeper un-merged level may still
    // hold a row the tombstone masks, so it is kept. Old readers see the
    // delete through the retained source bodies.
    if gc_tombstones {
        rows.retain(|r| !r.tombstone);
    }

    build_edge_sst(edge_type, edge_def, &rows, direction)
}

// ── PUT helpers (L1 variants) ───────────────────────────────────────────

// `schema` + `label_dict` join the existing five to seed per-label property
// stats for declared-but-absent properties during the L0->L1 rebuild; the
// params are all distinct and bundling them would not aid readability.
#[allow(clippy::too_many_arguments)]
async fn put_node_sst_leveled(
    store: &dyn ObjectStore,
    paths: &NamespacePaths,
    out_level: u32,
    label: &str,
    label_def: &LabelDef,
    merged_rows: &[NodeRow],
    finish: NodeSstFinish,
    schema: &Schema,
    label_dict: &LabelDictionary,
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
    put_create(store, &object_path, body).await?;

    let (bloom_descriptor, wrote_bloom) = put_bloom_sidecar(
        store,
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
        crate::flush::prepare_unique_property_sidecars(
            paths,
            level.as_u32(),
            &id,
            label,
            label_def,
            merged_rows,
        )?;
    // Re-emit equality-index posting-list sidecars too, rebuilt from the
    // already-reconciled `merged_rows` (tombstones dropped, highest-lsn per
    // id), so the L1 sidecar supersedes all the L0 partials.
    let (equality_property_indices, equality_sidecars) =
        crate::flush::prepare_equality_property_sidecars(
            paths,
            level.as_u32(),
            &id,
            label,
            label_def,
            merged_rows,
        )?;
    index_sidecars.extend(equality_sidecars);
    // Rebuild the label-index sidecar from the reconciled rows. id-primary
    // buckets (scope == "") carry per-row label sets in `merged_rows`, so this
    // re-emits the `LabelId -> [NodeId]` postings (with per-label counts) the
    // cost model needs; without it, every compaction would silently reset
    // per-label `node_count` to 0 and the optimizer would prune non-empty
    // labels again. Legacy per-label buckets have empty label sets and yield
    // `None` here, falling back to `scope`-based counting downstream.
    let (label_index, label_sidecar) =
        crate::flush::prepare_label_index_sidecar(paths, level.as_u32(), &id, merged_rows)?;
    if let Some(sidecar) = label_sidecar {
        index_sidecars.push(sidecar);
    }
    for (path, body) in &index_sidecars {
        put_create(store, path, body.clone()).await?;
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
        // Recompute per-(label, property) stats from the merged rows so they
        // survive L0->L1 the same way the label index does (RFC 025).
        per_label_property_stats: crate::flush::compute_per_label_property_stats(
            merged_rows,
            schema,
            label_dict,
        )?,
    };
    Ok((descriptor, wrote_bloom))
}

async fn put_edge_sst_leveled(
    store: &dyn ObjectStore,
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
    put_create(store, &object_path, body).await?;

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
    store: &dyn ObjectStore,
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
    put_create(store, &object_path, body).await?;
    Ok((Some(descriptor), true))
}

async fn put_create(store: &dyn ObjectStore, path: &Path, body: Bytes) -> Result<()> {
    let opts = PutOptions::from(PutMode::Create);
    store
        .put_opts(path, PutPayload::from(body), opts)
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
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
}
