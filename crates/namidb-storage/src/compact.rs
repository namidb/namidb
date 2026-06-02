//! Stateless L0 → L1 compaction.
//!
//! Each [`crate::flush::flush`] call appends a new L0 SST per `(kind,
//! scope)` bucket that had memtable rows. Without compaction, a
//! namespace's L0 footprint grows monotonically with every batch and
//! every point lookup pays an `O(L0 count)` candidate-SST scan.
//!
//! The compactor groups L0 SSTs by `(kind, scope)`, merges them into a
//! single L1 SST per bucket, and commits a manifest version that
//! removes the source descriptors and adds the new ones in a single
//! CAS. The source SST bodies become orphans in object storage (no
//! manifest version references them after the commit); a future
//! janitor sweeps them.
//!
//! ## Merge semantics
//!
//! Per `(node_id)` (nodes) or `(key_id, partner_id)` (edges): the row
//! with the highest LSN wins. Tombstones at the winning LSN are
//! preserved in the L1 SST — has no snapshot-retention policy
//! so a tombstone might still be load-bearing for a reader pinned at
//! a manifest version that pre-dates the compaction.
//!
//! ## What's deliberately not here
//!
//! - Multi-level compaction (L1 → L2, etc.). Only L0 → L1 in v1.
//! - Background scheduling. The caller invokes [`compact_l0_to_l1`]
//! manually; auto-trigger lands when the bench shape demands it.
//! - Range-partitioned compaction (split a bucket into multiple L1
//! files by key range). Each bucket emits exactly one L1 SST today.
//!
//! Declared edge property streams (RFC-002 §3.2.7) are preserved
//! end-to-end: the compactor reads each declared stream from every
//! source SST, joins it with the per-edge enumeration, and re-emits
//! the merged stream into the L1 body alongside `__overflow_json`.

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

use namidb_core::{LabelDef, Schema, Value};

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

/// Run one L0 → L1 compaction sweep across every `(kind, scope)`
/// bucket that holds more than one L0 SST.
#[instrument(
 skip(manifest_store, fence, base, schema),
 fields(
 namespace = %manifest_store.paths().namespace(),
 base_version = base.manifest.version,
 )
)]
pub async fn compact_l0_to_l1(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    base: &LoadedManifest,
    schema: &Schema,
) -> Result<CompactionOutcome> {
    fence.assert_alive(base.manifest.epoch)?;

    // Group L0 SSTs by (kind, scope).
    let mut node_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    let mut fwd_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    let mut inv_buckets: BTreeMap<String, Vec<&SstDescriptor>> = BTreeMap::new();
    for desc in &base.manifest.ssts {
        if desc.level != SstLevel::L0 {
            continue;
        }
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
        }
    }

    let store = manifest_store.store().clone();
    let paths = manifest_store.paths();
    let mut new_descs: Vec<SstDescriptor> = Vec::new();
    let mut removed_ids: Vec<Uuid> = Vec::new();
    let mut bloom_count: usize = 0;

    // Nodes.
    for (label, sources) in node_buckets {
        if sources.len() < 2 {
            continue;
        }
        let label_def = schema.label(&label).cloned().unwrap_or_else(|| LabelDef {
            name: label.clone(),
            properties: vec![],
        });
        let mut readers: Vec<NodeSstReader> = Vec::with_capacity(sources.len());
        for desc in &sources {
            let body = get_sst_body(store.as_ref(), paths, desc).await?;
            readers.push(NodeSstReader::open(label_def.clone(), body)?);
        }
        let (finish, merged_rows) = compact_node_ssts(&label_def, &readers)?;
        if finish.stats.row_count == 0 {
            // Nothing to write; still mark sources for removal so the
            // bucket truly shrinks.
            for src in &sources {
                removed_ids.push(src.id);
            }
            continue;
        }
        let (descriptor, wrote_bloom) = put_node_sst_l1(
            store.as_ref(),
            paths,
            &label,
            &label_def,
            &merged_rows,
            finish,
        )
        .await?;
        if wrote_bloom {
            bloom_count += 1;
        }
        for src in &sources {
            removed_ids.push(src.id);
        }
        new_descs.push(descriptor);
    }

    // Edges (forward).
    for (edge_type, sources) in fwd_buckets {
        if sources.len() < 2 {
            continue;
        }
        let (desc, wrote_bloom, removed) = compact_and_write_edges(
            store.as_ref(),
            paths,
            schema,
            &edge_type,
            &sources,
            EdgeDirection::Forward,
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
        if sources.len() < 2 {
            continue;
        }
        let (desc, wrote_bloom, removed) = compact_and_write_edges(
            store.as_ref(),
            paths,
            schema,
            &edge_type,
            &sources,
            EdgeDirection::Inverse,
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

async fn compact_and_write_edges(
    store: &dyn ObjectStore,
    paths: &NamespacePaths,
    schema: &Schema,
    edge_type: &str,
    sources: &[&SstDescriptor],
    direction: EdgeDirection,
) -> Result<(Option<SstDescriptor>, bool, Vec<Uuid>)> {
    let edge_def = schema.edge_type(edge_type).cloned();
    let mut readers: Vec<EdgeSstReader> = Vec::with_capacity(sources.len());
    for desc in sources {
        let body = get_sst_body(store, paths, desc).await?;
        readers.push(EdgeSstReader::open(body)?);
    }
    let finish = compact_edge_ssts(edge_type, edge_def.as_ref(), &readers, direction)?;
    let removed: Vec<Uuid> = sources.iter().map(|d| d.id).collect();
    if finish.stats.edge_count == 0 {
        return Ok((None, false, removed));
    }
    let (descriptor, wrote_bloom) =
        put_edge_sst_l1(store, paths, edge_type, direction, finish).await?;
    Ok((Some(descriptor), wrote_bloom, removed))
}

fn compact_node_ssts(
    label_def: &LabelDef,
    sources: &[NodeSstReader],
) -> Result<(NodeSstFinish, Vec<NodeRow>)> {
    let mut rows: Vec<NodeRow> = Vec::new();
    for reader in sources {
        rows.extend(extract_node_rows_from_reader(reader, label_def)?);
    }
    // Sort by node_id ascending; within ties, highest LSN first so the
    // dedup_by_key below preserves the winner.
    rows.sort_by(|a, b| a.id.cmp(&b.id).then(b.lsn.cmp(&a.lsn)));
    rows.dedup_by_key(|r| r.id);
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

    build_edge_sst(edge_type, edge_def, &rows, direction)
}

// ── PUT helpers (L1 variants) ───────────────────────────────────────────

async fn put_node_sst_l1(
    store: &dyn ObjectStore,
    paths: &NamespacePaths,
    label: &str,
    label_def: &LabelDef,
    merged_rows: &[NodeRow],
    finish: NodeSstFinish,
) -> Result<(SstDescriptor, bool)> {
    let id = Uuid::now_v7();
    let level = SstLevel(1);
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
        label_index: None,
    };
    Ok((descriptor, wrote_bloom))
}

async fn put_edge_sst_l1(
    store: &dyn ObjectStore,
    paths: &NamespacePaths,
    edge_type: &str,
    direction: EdgeDirection,
    finish: EdgeSstFinish,
) -> Result<(SstDescriptor, bool)> {
    let id = Uuid::now_v7();
    let level = SstLevel(1);
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
        assert_eq!(out.new_ssts_written, 1);

        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(out.committed.clone(), &mt_view, s, p);
        let v = snap.lookup_node("Person", alice).await.unwrap();
        assert!(v.is_none(), "tombstone in L1 must hide the upsert");
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
}
