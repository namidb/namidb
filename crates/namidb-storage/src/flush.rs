//! Flush path: `FrozenMemtable` → SSTs + bloom side-cars → manifest CAS.
//!
//! See [RFC-002](../../../docs/rfc/002-sst-format.md) for the SST format and
//! [RFC-001](../../../docs/rfc/001-storage-engine.md) §"Manifest protocol"
//! for the CAS commit.
//!
//! ## Memtable payload encoding
//!
//! The memtable stores opaque [`bytes::Bytes`] against each
//! [`MemKey`](crate::memtable::MemKey). The flush layer defines the wire
//! format of those bytes as JSON-serialised typed records:
//!
//! - [`NodeWriteRecord`] for node upserts (`MemKey::Node`),
//! - [`EdgeWriteRecord`] for edge upserts (`MemKey::Edge`).
//!
//! JSON keeps `Value`'s `#[serde(untagged)]` shape interpretable across
//! tools and through the WAL replay path. The bytes never leave RAM long
//! enough to make the JSON overhead matter; SSTs are columnar Parquet/CSR.
//!
//! Tombstones are represented by [`MemOp::Tombstone`](crate::memtable::MemOp)
//! and need no payload.
//!
//! ## Flow
//!
//! 1. Bucket the frozen memtable by `label` / `edge_type` (BTreeMap order
//! inside the memtable guarantees `node_id`-sorted node buckets and
//! `(src, dst)`-sorted edge buckets).
//! 2. For every node bucket, build the canonical [`RecordBatch`] and feed
//! it to a [`NodeSstWriter`].
//! 3. For every edge bucket, build [`EdgeStreamRow`]s for the forward
//! partner SST and transpose them for the inverse partner SST. Each
//! feeds its own [`EdgeSstWriter`].
//! 4. PUT every SST body + every non-omitted bloom side-car with
//! `PutMode::Create` so an in-flight conflicting writer cannot stomp.
//! 5. Build a fresh manifest carrying every new [`SstDescriptor`] and clear
//! `wal_segments` (every record they reference is now durable inside an
//! SST).
//! 6. Commit through [`ManifestStore::commit`] (CAS).
//!
//! On any error before the manifest commit we return immediately. Orphan
//! SST/bloom objects survive in object storage; they cost space but cannot
//! affect correctness because no manifest version references them. A future
//! janitor will sweep them.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Date32Builder, FixedSizeBinaryBuilder, FixedSizeListBuilder,
    Float32Builder, Float64Builder, Int32Builder, Int64Builder, LargeStringBuilder, StringBuilder,
    TimestampMicrosecondBuilder, UInt64Builder,
};
use arrow_array::{ArrayRef, RecordBatch};
use bytes::Bytes;
use chrono::Utc;
use object_store::path::Path;
use object_store::{ObjectStore, PutMode, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tracing::{debug, instrument};
use uuid::Uuid;

use namidb_core::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, Value};

use crate::error::{Error, Result};
use crate::fence::WriterFence;
use crate::manifest::{
    KindSpecificStats, LoadedManifest, ManifestStore, SstDescriptor, SstKind, SstLevel,
};
use crate::memtable::{FrozenMemtable, MemKey, MemOp};
use crate::paths::NamespacePaths;
use crate::sst::bloom::{BloomDescriptor, BloomFilter};
use crate::sst::edges::inverse::transpose_forward_to_inverse;
use crate::sst::edges::writer::{
    EdgeRecord as EdgeStreamRow, EdgeSstFinish, EdgeSstWriter, EdgeSstWriterOptions,
};
use crate::sst::edges::EdgeDirection;
use crate::sst::nodes::{node_arrow_schema, NodeSstFinish, NodeSstWriter, NodeSstWriterOptions};

// ── Wire-level records ─────────────────────────────────────────────────

/// Decoded payload of a [`MemOp::Upsert`](crate::memtable::MemOp::Upsert) for
/// a node. Stored bytes-on-wire encoding is JSON (see module docs for why).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NodeWriteRecord {
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    #[serde(default)]
    pub schema_version: u64,
}

impl NodeWriteRecord {
    pub fn encode(&self) -> Result<Bytes> {
        let bytes = serde_json::to_vec(self)?;
        Ok(Bytes::from(bytes))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let v = serde_json::from_slice(bytes)?;
        Ok(v)
    }
}

/// Decoded payload of a [`MemOp::Upsert`](crate::memtable::MemOp::Upsert) for
/// an edge. Stored bytes-on-wire encoding is JSON (see module docs for why).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EdgeWriteRecord {
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    #[serde(default)]
    pub schema_version: u64,
}

impl EdgeWriteRecord {
    pub fn encode(&self) -> Result<Bytes> {
        let bytes = serde_json::to_vec(self)?;
        Ok(Bytes::from(bytes))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let v = serde_json::from_slice(bytes)?;
        Ok(v)
    }
}

/// Outcome of a successful [`flush`].
#[derive(Debug)]
pub struct FlushOutcome {
    pub committed: LoadedManifest,
    pub ssts_written: usize,
    pub bloom_sidecars_written: usize,
}

// ── Entry point ────────────────────────────────────────────────────────

/// Orchestrate the full flush path. See module docs for the algorithm.
#[instrument(
 skip(manifest_store, fence, base, frozen, schema),
 fields(
 namespace = %manifest_store.paths().namespace(),
 base_version = base.manifest.version,
 memtable_entries = frozen.len(),
 )
)]
pub async fn flush(
    manifest_store: &ManifestStore,
    fence: &WriterFence,
    base: &LoadedManifest,
    frozen: &FrozenMemtable,
    schema: Schema,
) -> Result<FlushOutcome> {
    fence.assert_alive(base.manifest.epoch)?;

    if frozen.is_empty() {
        debug!("flush invoked on empty memtable; returning base manifest");
        return Ok(FlushOutcome {
            committed: base.clone(),
            ssts_written: 0,
            bloom_sidecars_written: 0,
        });
    }

    let (node_buckets, edge_buckets) = bucket_by_scope(frozen);

    let store = manifest_store.store().clone();
    let paths = manifest_store.paths();

    // 1. CPU phase — build every SST + bloom body in RAM. Sequential
    // because Arrow builders/encoders aren't cheap to spin up across
    // threads here; this is the same work as before, just decoupled
    // from the I/O.
    let mut pendings: Vec<PendingSst> = Vec::new();
    for (label, rows) in node_buckets {
        let label_def = schema_label_or_synthetic(&schema, &label);
        let finish = build_node_sst(&label_def, &rows)?;
        pendings.push(prepare_node_pending(
            paths, &label, &label_def, &rows, finish,
        )?);
    }
    for (edge_type, rows) in edge_buckets {
        let edge_def = schema.edge_type(&edge_type).cloned();
        let declared_property_names: Vec<String> = edge_def
            .as_ref()
            .map(|d| d.properties.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();
        let forward_rows = build_edge_stream_rows(&rows, &declared_property_names)?;
        let inverse_rows = transpose_forward_to_inverse(&forward_rows);
        let fwd = build_edge_sst(
            &edge_type,
            edge_def.as_ref(),
            &forward_rows,
            EdgeDirection::Forward,
        )?;
        let inv = build_edge_sst(
            &edge_type,
            edge_def.as_ref(),
            &inverse_rows,
            EdgeDirection::Inverse,
        )?;
        pendings.push(prepare_edge_pending(
            paths,
            &edge_type,
            EdgeDirection::Forward,
            fwd,
        ));
        pendings.push(prepare_edge_pending(
            paths,
            &edge_type,
            EdgeDirection::Inverse,
            inv,
        ));
    }

    // 2. I/O phase — issue every body + bloom PUT concurrently. The PUTs
    // are independent (each targets a fresh UUIDv7 path created above),
    // so the only ordering constraint is that they all complete before
    // the manifest CAS. `try_join_all` keeps the failure semantics:
    // the first error short-circuits the rest, mirroring the old
    // sequential behaviour. Orphan objects from in-flight PUTs that
    // succeeded before the failure are reclaimed by the janitor.
    let mut put_futures: Vec<_> = Vec::with_capacity(pendings.len() * 2);
    for p in &pendings {
        let body = p.body.clone();
        let path = p.body_path.clone();
        let store_ref = store.clone();
        put_futures.push(
            Box::pin(async move { put_object(store_ref, &path, body).await })
                as std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>,
        );
        if let (Some(bloom_path), Some(bloom_body)) = (&p.bloom_path, &p.bloom_body) {
            let body = bloom_body.clone();
            let path = bloom_path.clone();
            let store_ref = store.clone();
            put_futures.push(Box::pin(
                async move { put_object(store_ref, &path, body).await },
            ));
        }
        // Per-unique-property side-cars (RFC-pending). PUT'd alongside
        // the body / bloom so the entire SST + its lookup acceleration
        // structures land atomically from the writer's perspective; the
        // manifest CAS below makes the new descriptors visible only
        // when every sidecar has been durably persisted.
        for (path, body) in &p.index_sidecars {
            let body = body.clone();
            let path = path.clone();
            let store_ref = store.clone();
            put_futures.push(Box::pin(
                async move { put_object(store_ref, &path, body).await },
            ));
        }
    }
    futures::future::try_join_all(put_futures).await?;

    let bloom_count = pendings.iter().filter(|p| p.bloom_body.is_some()).count();
    let new_ssts: Vec<SstDescriptor> = pendings.into_iter().map(|p| p.descriptor).collect();
    let ssts_written = new_ssts.len();

    let mut next = base.manifest.next_version(fence.writer_id);
    next.schema = schema;
    next.ssts.extend(new_ssts);
    next.wal_segments.clear();

    let committed = manifest_store.commit(fence, base, next).await?;

    Ok(FlushOutcome {
        committed,
        ssts_written,
        bloom_sidecars_written: bloom_count,
    })
}

/// Per-SST work product: descriptor + body bytes + their object-store paths,
/// kept together so the parallel-PUT phase can issue them without re-touching
/// the schema/Arrow builders. `index_sidecars` is non-empty when the SST is
/// a Node SST and the label declares one or more `unique` properties — each
/// entry is a `(value_string → NodeId)` map serialised to bincode that the
/// reader can probe in O(log N) without rescanning the SST body.
struct PendingSst {
    descriptor: SstDescriptor,
    body_path: Path,
    body: Bytes,
    bloom_path: Option<Path>,
    bloom_body: Option<Bytes>,
    /// `(path, body)` for each unique-property side-car emitted alongside
    /// this SST. Empty for edge SSTs and for node SSTs whose label has no
    /// `PropertyDef::unique == true`.
    index_sidecars: Vec<(Path, Bytes)>,
}

// ── Bucketing ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct NodeRow {
    pub(crate) id: [u8; 16],
    pub(crate) lsn: u64,
    pub(crate) op: MemOp,
}

#[derive(Debug, Clone)]
pub(crate) struct EdgeRow {
    pub(crate) src: [u8; 16],
    pub(crate) dst: [u8; 16],
    pub(crate) lsn: u64,
    pub(crate) op: MemOp,
}

/// Convert the frozen memtable into ordered, per-scope buckets. Memtable
/// iteration order (BTreeMap) guarantees the rows in each bucket are
/// already sorted by `node_id` (nodes) or `(src, dst)` (edges).
fn bucket_by_scope(
    frozen: &FrozenMemtable,
) -> (
    BTreeMap<String, Vec<NodeRow>>,
    BTreeMap<String, Vec<EdgeRow>>,
) {
    let mut nodes: BTreeMap<String, Vec<NodeRow>> = BTreeMap::new();
    let mut edges: BTreeMap<String, Vec<EdgeRow>> = BTreeMap::new();
    for (k, e) in frozen.iter() {
        match k {
            MemKey::Node { label, id } => {
                nodes.entry(label.clone()).or_default().push(NodeRow {
                    id: *id.as_bytes(),
                    lsn: e.lsn,
                    op: e.op.clone(),
                });
            }
            MemKey::Edge {
                edge_type,
                src,
                dst,
            } => {
                edges.entry(edge_type.clone()).or_default().push(EdgeRow {
                    src: *src.as_bytes(),
                    dst: *dst.as_bytes(),
                    lsn: e.lsn,
                    op: e.op.clone(),
                });
            }
        }
    }
    (nodes, edges)
}

/// Lookup the [`LabelDef`] in the schema, or fall back to a synthetic
/// empty-property label so untyped writes still flow through the path.
/// Declared properties always win at the SST level; anything else lands in
/// `__overflow_json` (see RFC-002 §2.1).
fn schema_label_or_synthetic(schema: &Schema, label: &str) -> LabelDef {
    schema.label(label).cloned().unwrap_or_else(|| LabelDef {
        name: label.to_string(),
        properties: Vec::new(),
    })
}

// ── Node SST building ──────────────────────────────────────────────────

pub(crate) fn build_node_sst(label: &LabelDef, rows: &[NodeRow]) -> Result<NodeSstFinish> {
    let options = NodeSstWriterOptions {
        expected_keys: rows.len() as u64,
        ..Default::default()
    };
    let mut writer = NodeSstWriter::new(label.clone(), options)?;
    let arrow_schema = node_arrow_schema(label);
    let batch = build_node_record_batch(&arrow_schema, label, rows)?;
    if batch.num_rows() > 0 {
        writer.write_batch(&batch)?;
    }
    writer.finish()
}

fn build_node_record_batch(
    arrow_schema: &arrow_schema::SchemaRef,
    label: &LabelDef,
    rows: &[NodeRow],
) -> Result<RecordBatch> {
    let n = rows.len();

    let mut node_id_b = FixedSizeBinaryBuilder::with_capacity(n, 16);
    let mut tomb_b = BooleanBuilder::with_capacity(n);
    let mut lsn_b = UInt64Builder::with_capacity(n);
    let mut prop_builders: Vec<PropertyBuilder> = label
        .properties
        .iter()
        .map(|p| PropertyBuilder::new(&p.data_type, n))
        .collect::<Result<Vec<_>>>()?;
    let mut overflow_b = StringBuilder::with_capacity(n, 32 * n.max(1));
    let mut schema_version_b = UInt64Builder::with_capacity(n);

    let declared_names: Vec<&str> = label.properties.iter().map(|p| p.name.as_str()).collect();

    for row in rows {
        node_id_b
            .append_value(row.id)
            .map_err(|e| Error::invariant(format!("node_id append: {e}")))?;
        match &row.op {
            MemOp::Upsert(bytes) => {
                let rec = NodeWriteRecord::decode(bytes)?;
                tomb_b.append_value(false);
                lsn_b.append_value(row.lsn);

                for (idx, p) in label.properties.iter().enumerate() {
                    let value = rec.properties.get(&p.name);
                    prop_builders[idx].append(value, p)?;
                }
                let overflow: BTreeMap<&String, &Value> = rec
                    .properties
                    .iter()
                    .filter(|(k, _)| !declared_names.contains(&k.as_str()))
                    .collect();
                if overflow.is_empty() {
                    overflow_b.append_null();
                } else {
                    let json = serde_json::to_string(&overflow)
                        .map_err(|e| Error::invariant(format!("overflow encode: {e}")))?;
                    overflow_b.append_value(&json);
                }
                schema_version_b.append_value(rec.schema_version);
            }
            MemOp::Tombstone => {
                tomb_b.append_value(true);
                lsn_b.append_value(row.lsn);
                for b in &mut prop_builders {
                    b.append_null();
                }
                overflow_b.append_null();
                schema_version_b.append_value(0);
            }
        }
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    columns.push(Arc::new(node_id_b.finish()));
    columns.push(Arc::new(tomb_b.finish()));
    columns.push(Arc::new(lsn_b.finish()));
    for b in &mut prop_builders {
        columns.push(b.finish());
    }
    columns.push(Arc::new(overflow_b.finish()));
    columns.push(Arc::new(schema_version_b.finish()));

    RecordBatch::try_new(arrow_schema.clone(), columns)
        .map_err(|e| Error::invariant(format!("node batch build: {e}")))
}

// ── Edge SST building ──────────────────────────────────────────────────

fn build_edge_stream_rows(
    rows: &[EdgeRow],
    declared_property_names: &[String],
) -> Result<Vec<EdgeStreamRow>> {
    let mut out: Vec<EdgeStreamRow> = Vec::with_capacity(rows.len());
    let empty_declared: Vec<Option<String>> = vec![None; declared_property_names.len()];
    for row in rows {
        match &row.op {
            MemOp::Upsert(bytes) => {
                let rec = EdgeWriteRecord::decode(bytes)?;
                // RFC-002 §3.2.7: each property whose key matches a
                // declared edge-type property routes to that property's
                // named stream (JSON-encoded `Value`). The remainder
                // (a.k.a. ad-hoc properties on a declared edge type, or
                // every property on an undeclared edge type) collapses
                // into the legacy `__overflow_json` stream.
                let mut declared: Vec<Option<String>> = vec![None; declared_property_names.len()];
                let mut overflow_map: BTreeMap<String, Value> = BTreeMap::new();
                for (name, value) in &rec.properties {
                    if let Some(idx) = declared_property_names.iter().position(|n| n == name) {
                        let encoded = serde_json::to_string(value).map_err(|e| {
                            Error::invariant(format!("edge property '{name}' encode: {e}"))
                        })?;
                        declared[idx] = Some(encoded);
                    } else {
                        overflow_map.insert(name.clone(), value.clone());
                    }
                }
                let overflow_json = if overflow_map.is_empty() {
                    None
                } else {
                    let json = serde_json::to_string(&overflow_map)
                        .map_err(|e| Error::invariant(format!("edge overflow encode: {e}")))?;
                    Some(json)
                };
                out.push(EdgeStreamRow {
                    key_id: row.src,
                    partner_id: row.dst,
                    lsn: row.lsn,
                    tombstone: false,
                    declared_properties: declared,
                    overflow_json,
                });
            }
            MemOp::Tombstone => {
                out.push(EdgeStreamRow {
                    key_id: row.src,
                    partner_id: row.dst,
                    lsn: row.lsn,
                    tombstone: true,
                    declared_properties: empty_declared.clone(),
                    overflow_json: None,
                });
            }
        }
    }
    Ok(out)
}

pub(crate) fn build_edge_sst(
    edge_type: &str,
    edge_def: Option<&EdgeTypeDef>,
    rows: &[EdgeStreamRow],
    direction: EdgeDirection,
) -> Result<EdgeSstFinish> {
    let (src_label, dst_label) = match edge_def {
        Some(def) => (def.src_label.clone(), def.dst_label.clone()),
        None => ("_".to_string(), "_".to_string()),
    };

    // Count distinct keys (rows are pre-sorted by key_id ascending) so the
    // bloom can be sized correctly.
    let mut last_key: Option<[u8; 16]> = None;
    let mut distinct_keys: u64 = 0;
    for r in rows {
        if Some(r.key_id) != last_key {
            distinct_keys += 1;
            last_key = Some(r.key_id);
        }
    }

    let mut options = EdgeSstWriterOptions::new(direction, edge_type, src_label, dst_label);
    options.expected_keys = distinct_keys.max(1);
    if let Some(def) = edge_def {
        options.declared_properties = def.properties.iter().map(|p| p.name.clone()).collect();
    }

    let mut writer = EdgeSstWriter::new(options);
    for row in rows {
        writer.append(row.clone())?;
    }
    writer.finish()
}

// ── PUT helpers ────────────────────────────────────────────────────────

fn prepare_node_pending(
    paths: &NamespacePaths,
    label: &str,
    label_def: &LabelDef,
    rows: &[NodeRow],
    finish: NodeSstFinish,
) -> Result<PendingSst> {
    let id = Uuid::now_v7();
    let level = SstLevel::L0;
    let file_name = format!(
        "{}-{}-{}.parquet",
        uuid_path_id(&id),
        SstKind::Nodes.path_tag(),
        label
    );
    let body_path = paths.sst_object(level.as_u32(), &file_name);
    let relative_path = relative_sst_path(level.as_u32(), &file_name);
    let body_len = finish.body.len() as u64;

    let (bloom_descriptor, bloom_path, bloom_body) = prepare_bloom_sidecar(
        paths,
        level.as_u32(),
        &id,
        SstKind::Nodes.path_tag(),
        label,
        finish.bloom,
    );

    // Per-property unique sidecars (RFC-pending): for each declared
    // `unique` property emit a `value_string → NodeId` map serialised to
    // bincode. The reader probes these instead of full-scanning the SST
    // to resolve `MATCH (a:Label {prop: 'X'})`.
    let (unique_property_indices, index_sidecars) =
        prepare_unique_property_sidecars(paths, level.as_u32(), &id, label, label_def, rows)?;

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
    };

    Ok(PendingSst {
        descriptor,
        body_path,
        body: finish.body,
        bloom_path,
        bloom_body,
        index_sidecars,
    })
}

/// Parallel outputs of [`prepare_unique_property_sidecars`]: the
/// per-property index descriptors (destined for the manifest) and the
/// matching `(path, body)` objects to PUT next to the SST body. Aliased
/// to keep the return type under clippy's type-complexity threshold.
type UniquePropertySidecars = (
    Vec<crate::manifest::UniquePropertyIndexDescriptor>,
    Vec<(Path, Bytes)>,
);

/// For every `PropertyDef::unique == true` in `label_def.properties`,
/// walk `rows`, harvest `(value_string, NodeId)` pairs, sort them into a
/// `BTreeMap`, serialise to bincode, and produce one
/// `(UniquePropertyIndexDescriptor, (path, body))` pair per property.
///
/// Returns the parallel collections so the descriptor can land in the
/// manifest and the body can be PUT alongside the SST body.
///
/// Tombstoned rows contribute nothing — they're encoded in the SST body
/// and the reader's last-LSN-wins logic surfaces them correctly. Rows
/// without the property (nullable column, schema-evolved out, ...)
/// contribute nothing either. Non-string property values are skipped —
/// v0 covers LDBC's `id` only; a future bump can promote typed keys.
///
/// `pub(crate)` so `compact.rs` can re-emit sidecars when merging
/// L0 SSTs into L1 (without this, post-compaction `lookup_node_by_property`
/// falls back to the legacy full label scan because none of the L1
/// SSTs carry the sidecar).
pub(crate) fn prepare_unique_property_sidecars(
    paths: &NamespacePaths,
    level: u32,
    sst_id: &Uuid,
    label: &str,
    label_def: &LabelDef,
    rows: &[NodeRow],
) -> Result<UniquePropertySidecars> {
    let mut descriptors = Vec::new();
    let mut bodies = Vec::new();
    for prop in &label_def.properties {
        if !prop.unique {
            continue;
        }
        let mut index: BTreeMap<String, [u8; 16]> = BTreeMap::new();
        for row in rows {
            if let MemOp::Upsert(payload) = &row.op {
                let rec = NodeWriteRecord::decode(payload)?;
                if let Some(Value::Str(s)) = rec.properties.get(&prop.name) {
                    // Last-write-wins within one SST; the BTreeMap
                    // overwrites cleanly when the same value re-occurs
                    // (writer guarantees row order matches lsn order).
                    index.insert(s.clone(), row.id);
                }
            }
        }
        if index.is_empty() {
            continue;
        }
        let body_bytes = bincode::serialize(&index)
            .map_err(|e| Error::invariant(format!("unique-index bincode: {e}")))?;
        let entry_count = index.len() as u64;
        let body = Bytes::from(body_bytes);

        let file_name = format!(
            "{}-{}-{}.idx_{}.bin",
            uuid_path_id(sst_id),
            SstKind::Nodes.path_tag(),
            label,
            prop.name,
        );
        let object_path = paths.sst_object(level, &file_name);
        let relative = relative_sst_path(level, &file_name);
        descriptors.push(crate::manifest::UniquePropertyIndexDescriptor {
            property: prop.name.clone(),
            path: relative,
            size_bytes: body.len() as u64,
            entry_count,
        });
        bodies.push((object_path, body));
    }
    Ok((descriptors, bodies))
}

fn prepare_edge_pending(
    paths: &NamespacePaths,
    edge_type: &str,
    direction: EdgeDirection,
    finish: EdgeSstFinish,
) -> PendingSst {
    let id = Uuid::now_v7();
    let level = SstLevel::L0;
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
    let body_path = paths.sst_object(level.as_u32(), &file_name);
    let relative_path = relative_sst_path(level.as_u32(), &file_name);
    let body_len = finish.body.len() as u64;

    let (bloom_descriptor, bloom_path, bloom_body) = prepare_bloom_sidecar(
        paths,
        level.as_u32(),
        &id,
        direction.path_tag(),
        edge_type,
        finish.bloom,
    );

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
    };

    PendingSst {
        descriptor,
        body_path,
        body: finish.body,
        bloom_path,
        bloom_body,
        index_sidecars: Vec::new(),
    }
}

fn prepare_bloom_sidecar(
    paths: &NamespacePaths,
    level: u32,
    sst_id: &Uuid,
    tag: &str,
    scope: &str,
    bloom: Option<BloomFilter>,
) -> (Option<BloomDescriptor>, Option<Path>, Option<Bytes>) {
    let Some(bloom) = bloom else {
        return (None, None, None);
    };
    let file_name = format!("{}-{}-{}.bloom", uuid_path_id(sst_id), tag, scope);
    let object_path = paths.sst_object(level, &file_name);
    let relative = relative_sst_path(level, &file_name);
    let body = bloom.to_bytes();
    let descriptor =
        BloomDescriptor::from_body(relative, &body).expect("bloom side-car body is well-formed");
    (Some(descriptor), Some(object_path), Some(body))
}

async fn put_create(store: &dyn ObjectStore, path: &Path, body: Bytes) -> Result<()> {
    let opts = PutOptions::from(PutMode::Create);
    store
        .put_opts(path, PutPayload::from(body), opts)
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

/// Threshold above which an SST body is uploaded via multipart instead of a
/// single PUT. Sits just below the S3 5 MiB per-part minimum so any body
/// that produces at least one full part (~SF1 edge SSTs at 2–5 MiB and
/// every L1 compacted SST) crosses the multipart path; bodies smaller than
/// this still go single-PUT (no multipart overhead and `PutMode::Create`
/// stays available for collision protection).
const MULTIPART_THRESHOLD: usize = 4 * 1024 * 1024;

/// Per-part chunk size for multipart uploads. S3's hard floor is 5 MiB on
/// every part except the trailing one; we match that floor so each part
/// is valid in isolation. R2 inherits the same floor.
const MULTIPART_PART_SIZE: usize = 5 * 1024 * 1024;

/// Default in-flight concurrency for multipart upload parts. The
/// `object_store::buffered::BufWriter` default is also 8; we mirror it
/// explicitly so the call site documents the chosen rate.
const MULTIPART_MAX_CONCURRENCY: usize = 8;

/// Upload `body` to `path`. For small bodies, falls back to the single-PUT
/// `PutMode::Create` path so the CAS-style "no overwrite" semantics still
/// protect against a competing writer stomping on a UUIDv7 path. For
/// bodies past [`MULTIPART_THRESHOLD`] (SST bodies in the LDBC SNB SF1
/// range — 10–50 MiB), uses `BufWriter` with `MULTIPART_PART_SIZE` chunks
/// and `MULTIPART_MAX_CONCURRENCY` in-flight uploads.
///
/// Why the split: S3 / R2 multipart uploads do NOT honour the `If-None-Match`
/// header that backs `PutMode::Create`. SST paths embed a UUIDv7 per writer
/// (see [`crate::flush`] §"PUT helpers") so collisions are impossible in
/// practice; the small-PUT branch is kept for bloom side-cars and any
/// future small body, where the CAS protection is cheap to keep.
async fn put_object(
    store: std::sync::Arc<dyn ObjectStore>,
    path: &Path,
    body: Bytes,
) -> Result<()> {
    if body.len() < MULTIPART_THRESHOLD {
        return put_create(store.as_ref(), path, body).await;
    }
    let mut writer =
        object_store::buffered::BufWriter::with_capacity(store, path.clone(), MULTIPART_PART_SIZE)
            .with_max_concurrency(MULTIPART_MAX_CONCURRENCY);
    writer.put(body).await.map_err(Error::ObjectStore)?;
    writer.shutdown().await.map_err(|e| {
        Error::ObjectStore(object_store::Error::Generic {
            store: "BufWriter",
            source: Box::new(e),
        })
    })?;
    Ok(())
}

/// Render a UUID in its lowercase simple (32-hex-char) form. RFC-002 §1
/// pins the full UUIDv7 to the SST filename so writers that flush more
/// than once per millisecond cannot collide.
fn uuid_path_id(u: &Uuid) -> String {
    u.simple().to_string()
}

fn relative_sst_path(level: u32, file_name: &str) -> String {
    format!("sst/level{level}/{file_name}")
}

// ── Per-property Arrow column builder ──────────────────────────────────

enum PropertyBuilder {
    Bool(BooleanBuilder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    Utf8(StringBuilder),
    LargeUtf8(LargeStringBuilder),
    Binary(BinaryBuilder),
    Date32(Date32Builder),
    Timestamp(TimestampMicrosecondBuilder),
    FloatVector {
        dim: u32,
        builder: FixedSizeListBuilder<Float32Builder>,
    },
    Json(StringBuilder),
}

impl PropertyBuilder {
    fn new(dt: &DataType, capacity: usize) -> Result<Self> {
        Ok(match dt {
            DataType::Bool => PropertyBuilder::Bool(BooleanBuilder::with_capacity(capacity)),
            DataType::Int32 => PropertyBuilder::Int32(Int32Builder::with_capacity(capacity)),
            DataType::Int64 => PropertyBuilder::Int64(Int64Builder::with_capacity(capacity)),
            DataType::Float32 => PropertyBuilder::Float32(Float32Builder::with_capacity(capacity)),
            DataType::Float64 => PropertyBuilder::Float64(Float64Builder::with_capacity(capacity)),
            DataType::Utf8 => {
                PropertyBuilder::Utf8(StringBuilder::with_capacity(capacity, 32 * capacity.max(1)))
            }
            DataType::LargeUtf8 => PropertyBuilder::LargeUtf8(LargeStringBuilder::with_capacity(
                capacity,
                32 * capacity.max(1),
            )),
            DataType::Binary => PropertyBuilder::Binary(BinaryBuilder::with_capacity(
                capacity,
                32 * capacity.max(1),
            )),
            DataType::Date32 => PropertyBuilder::Date32(Date32Builder::with_capacity(capacity)),
            DataType::TimestampMicrosUtc => PropertyBuilder::Timestamp(
                TimestampMicrosecondBuilder::with_capacity(capacity).with_timezone("UTC"),
            ),
            DataType::FloatVector { dim } => {
                let inner = Float32Builder::with_capacity(capacity * *dim as usize);
                let builder = FixedSizeListBuilder::new(inner, *dim as i32);
                PropertyBuilder::FloatVector { dim: *dim, builder }
            }
            DataType::Json => {
                PropertyBuilder::Json(StringBuilder::with_capacity(capacity, 64 * capacity.max(1)))
            }
        })
    }

    fn append(&mut self, value: Option<&Value>, def: &PropertyDef) -> Result<()> {
        let Some(value) = value else {
            self.append_null();
            return Ok(());
        };
        if value.is_null() {
            self.append_null();
            return Ok(());
        }
        match (self, value) {
            (PropertyBuilder::Bool(b), Value::Bool(v)) => {
                b.append_value(*v);
                Ok(())
            }
            (PropertyBuilder::Int32(b), Value::I64(v)) => {
                let v32: i32 = (*v).try_into().map_err(|_| {
                    Error::invariant(format!(
                        "property '{}' i64={} does not fit Int32",
                        def.name, v
                    ))
                })?;
                b.append_value(v32);
                Ok(())
            }
            (PropertyBuilder::Int64(b), Value::I64(v)) => {
                b.append_value(*v);
                Ok(())
            }
            (PropertyBuilder::Float32(b), Value::F64(v)) => {
                let downcast = *v as f32;
                // Reject silent overflow: a finite f64 outside the f32
                // range becomes ±inf via `as f32`, which is data loss the
                // caller cannot recover. Precision loss for in-range
                // values is documented and tolerated.
                if v.is_finite() && !downcast.is_finite() {
                    return Err(Error::invariant(format!(
                        "property '{}' f64={v} overflows Float32",
                        def.name
                    )));
                }
                b.append_value(downcast);
                Ok(())
            }
            (PropertyBuilder::Float64(b), Value::F64(v)) => {
                b.append_value(*v);
                Ok(())
            }
            (PropertyBuilder::Utf8(b), Value::Str(s)) => {
                b.append_value(s);
                Ok(())
            }
            (PropertyBuilder::LargeUtf8(b), Value::Str(s)) => {
                b.append_value(s);
                Ok(())
            }
            (PropertyBuilder::Binary(b), Value::Bytes(v)) => {
                b.append_value(v);
                Ok(())
            }
            (PropertyBuilder::Date32(b), Value::I64(v)) => {
                let v32: i32 = (*v).try_into().map_err(|_| {
                    Error::invariant(format!(
                        "property '{}' date i64={} does not fit i32",
                        def.name, v
                    ))
                })?;
                b.append_value(v32);
                Ok(())
            }
            (PropertyBuilder::Date32(b), Value::Date(v)) => {
                b.append_value(*v);
                Ok(())
            }
            (PropertyBuilder::Timestamp(b), Value::I64(v)) => {
                b.append_value(*v);
                Ok(())
            }
            (PropertyBuilder::Timestamp(b), Value::DateTime(v)) => {
                b.append_value(*v);
                Ok(())
            }
            (PropertyBuilder::FloatVector { dim, builder }, Value::Vec(v)) => {
                if v.len() != *dim as usize {
                    return Err(Error::invariant(format!(
                        "property '{}' float vector dim={} != declared {}",
                        def.name,
                        v.len(),
                        dim
                    )));
                }
                for x in v {
                    builder.values().append_value(*x);
                }
                builder.append(true);
                Ok(())
            }
            (PropertyBuilder::Json(b), v) => {
                let s = serde_json::to_string(v).map_err(|e| {
                    Error::invariant(format!("json encode for '{}': {e}", def.name))
                })?;
                b.append_value(&s);
                Ok(())
            }
            (slot, v) => Err(Error::invariant(format!(
                "property '{}' value {:?} does not match declared type {}",
                def.name,
                v,
                slot.kind_str()
            ))),
        }
    }

    fn append_null(&mut self) {
        match self {
            PropertyBuilder::Bool(b) => b.append_null(),
            PropertyBuilder::Int32(b) => b.append_null(),
            PropertyBuilder::Int64(b) => b.append_null(),
            PropertyBuilder::Float32(b) => b.append_null(),
            PropertyBuilder::Float64(b) => b.append_null(),
            PropertyBuilder::Utf8(b) => b.append_null(),
            PropertyBuilder::LargeUtf8(b) => b.append_null(),
            PropertyBuilder::Binary(b) => b.append_null(),
            PropertyBuilder::Date32(b) => b.append_null(),
            PropertyBuilder::Timestamp(b) => b.append_null(),
            PropertyBuilder::FloatVector { dim, builder } => {
                // FixedSizeList null requires advancing the inner builder by
                // `dim` entries so parallel arrays stay aligned.
                for _ in 0..*dim {
                    builder.values().append_value(0.0);
                }
                builder.append(false);
            }
            PropertyBuilder::Json(b) => b.append_null(),
        }
    }

    fn kind_str(&self) -> &'static str {
        match self {
            PropertyBuilder::Bool(_) => "Bool",
            PropertyBuilder::Int32(_) => "Int32",
            PropertyBuilder::Int64(_) => "Int64",
            PropertyBuilder::Float32(_) => "Float32",
            PropertyBuilder::Float64(_) => "Float64",
            PropertyBuilder::Utf8(_) => "Utf8",
            PropertyBuilder::LargeUtf8(_) => "LargeUtf8",
            PropertyBuilder::Binary(_) => "Binary",
            PropertyBuilder::Date32(_) => "Date32",
            PropertyBuilder::Timestamp(_) => "TimestampMicrosUtc",
            PropertyBuilder::FloatVector { .. } => "FloatVector",
            PropertyBuilder::Json(_) => "Json",
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            PropertyBuilder::Bool(b) => Arc::new(b.finish()),
            PropertyBuilder::Int32(b) => Arc::new(b.finish()),
            PropertyBuilder::Int64(b) => Arc::new(b.finish()),
            PropertyBuilder::Float32(b) => Arc::new(b.finish()),
            PropertyBuilder::Float64(b) => Arc::new(b.finish()),
            PropertyBuilder::Utf8(b) => Arc::new(b.finish()),
            PropertyBuilder::LargeUtf8(b) => Arc::new(b.finish()),
            PropertyBuilder::Binary(b) => Arc::new(b.finish()),
            PropertyBuilder::Date32(b) => Arc::new(b.finish()),
            PropertyBuilder::Timestamp(b) => Arc::new(b.finish()),
            PropertyBuilder::FloatVector { builder, .. } => Arc::new(builder.finish()),
            PropertyBuilder::Json(b) => Arc::new(b.finish()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use namidb_core::{EdgeTypeDef, LabelDef, NamespaceId, NodeId, PropertyDef, SchemaBuilder};
    use object_store::memory::InMemory;
    use object_store::ObjectStoreExt;

    use super::*;
    use crate::manifest::SstKind;
    use crate::memtable::{MemKey, Memtable};
    use crate::paths::NamespacePaths;
    use crate::sst::edges::reader::EdgeSstReader;
    use crate::sst::nodes::NodeSstReader;
    use bytes::Bytes;
    use uuid::Uuid;

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

    fn sorted_node_id(ix: u8) -> NodeId {
        // Build a UUIDv7-shaped 16-byte id whose ordering follows `ix`.
        let mut bytes = [0u8; 16];
        bytes[15] = ix;
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
        }
        .encode()
        .unwrap()
    }

    fn edge_payload(since: Option<i64>) -> Bytes {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        if let Some(s) = since {
            props.insert("since".into(), Value::I64(s));
        }
        EdgeWriteRecord {
            properties: props,
            schema_version: 1,
        }
        .encode()
        .unwrap()
    }

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn make_paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    #[test]
    fn node_write_record_round_trips() {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("name".into(), Value::Str("Alice".into()));
        props.insert("age".into(), Value::I64(30));
        let r = NodeWriteRecord {
            properties: props,
            schema_version: 7,
        };
        let bytes = r.encode().unwrap();
        let back = NodeWriteRecord::decode(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn edge_write_record_round_trips() {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("since".into(), Value::I64(2020));
        let r = EdgeWriteRecord {
            properties: props,
            schema_version: 3,
        };
        let bytes = r.encode().unwrap();
        let back = EdgeWriteRecord::decode(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[tokio::test]
    async fn flush_empty_memtable_is_noop() {
        let store = make_store();
        let paths = make_paths("e2e-empty");
        let ms = ManifestStore::new(store, paths);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let frozen = Memtable::new().freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, Schema::empty())
            .await
            .unwrap();
        assert_eq!(outcome.ssts_written, 0);
        assert_eq!(outcome.bloom_sidecars_written, 0);
        assert_eq!(outcome.committed.manifest.version, base.manifest.version);
    }

    #[tokio::test]
    async fn flush_writes_node_and_edge_ssts_then_commits_manifest() {
        let store = make_store();
        let paths = make_paths("e2e-flush");
        let ms = ManifestStore::new(store.clone(), paths);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: bob,
            },
            11,
            MemOp::Upsert(node_payload("Bob", None)),
        );
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: carol,
            },
            12,
            MemOp::Tombstone,
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            13,
            MemOp::Upsert(edge_payload(Some(2020))),
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: bob,
                dst: alice,
            },
            14,
            MemOp::Upsert(edge_payload(None)),
        );

        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Three SST descriptors: nodes-Person + edges-fwd-KNOWS + edges-inv-KNOWS.
        assert_eq!(outcome.ssts_written, 3);
        assert_eq!(
            outcome.committed.manifest.version,
            base.manifest.version + 1
        );
        assert_eq!(outcome.committed.manifest.ssts.len(), 3);
        assert!(outcome.committed.manifest.wal_segments.is_empty());

        let kinds: Vec<SstKind> = outcome
            .committed
            .manifest
            .ssts
            .iter()
            .map(|d| d.kind)
            .collect();
        assert!(kinds.contains(&SstKind::Nodes));
        assert!(kinds.contains(&SstKind::EdgesFwd));
        assert!(kinds.contains(&SstKind::EdgesInv));

        // Read the node SST back from the store and verify rows.
        let node_d = outcome
            .committed
            .manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::Nodes)
            .unwrap();
        let abs = ms
            .paths()
            .sst_object(node_d.level.as_u32(), file_basename(&node_d.path));
        let body = store.get(&abs).await.unwrap().bytes().await.unwrap();
        let reader = NodeSstReader::open(person_label(), body).unwrap();
        let batches = reader.scan().unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);

        // Read forward edge SST back and confirm a partner lookup succeeds.
        let fwd_d = outcome
            .committed
            .manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::EdgesFwd)
            .unwrap();
        let abs_fwd = ms
            .paths()
            .sst_object(fwd_d.level.as_u32(), file_basename(&fwd_d.path));
        let body_fwd = store.get(&abs_fwd).await.unwrap().bytes().await.unwrap();
        let reader_fwd = EdgeSstReader::open(body_fwd).unwrap();
        let look = reader_fwd.lookup(alice.as_bytes()).unwrap().unwrap();
        assert_eq!(look.partners, vec![*bob.as_bytes()]);

        // The inverse SST must answer the in-edge lookup from `bob`'s side.
        let inv_d = outcome
            .committed
            .manifest
            .ssts
            .iter()
            .find(|d| d.kind == SstKind::EdgesInv)
            .unwrap();
        let abs_inv = ms
            .paths()
            .sst_object(inv_d.level.as_u32(), file_basename(&inv_d.path));
        let body_inv = store.get(&abs_inv).await.unwrap().bytes().await.unwrap();
        let reader_inv = EdgeSstReader::open(body_inv).unwrap();
        let look_in = reader_inv.lookup(bob.as_bytes()).unwrap().unwrap();
        assert_eq!(look_in.partners, vec![*alice.as_bytes()]);

        // Schema snapshot was carried forward.
        assert_eq!(outcome.committed.manifest.schema, schema);
    }

    #[tokio::test]
    async fn flush_clears_wal_segments_from_base() {
        use crate::manifest::WalSegmentDescriptor;

        let store = make_store();
        let paths = make_paths("e2e-clearwal");
        let ms = ManifestStore::new(store, paths);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        // Pretend a previous step recorded a WAL segment.
        let mut step1 = base.manifest.next_version(fence.writer_id);
        step1.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: "wal/0000000000000001.wal".into(),
            last_lsn: 9,
        });
        let with_wal = ms.commit(&fence, &base, step1).await.unwrap();
        assert_eq!(with_wal.manifest.wal_segments.len(), 1);

        // Now flush: even with an empty memtable we'd skip the work; build
        // something tiny so flush goes through and confirms the WAL list is
        // cleared.
        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &with_wal, &frozen, Schema::empty())
            .await
            .unwrap();
        assert!(outcome.committed.manifest.wal_segments.is_empty());
        // C5: `frozen` is borrowed, so the caller still owns it after flush.
        assert_eq!(frozen.len(), 1);
    }

    #[tokio::test]
    async fn flush_returns_cas_loss_without_consuming_frozen() {
        // C5 (bug audit): a flush that loses the CAS race must NOT
        // consume the frozen memtable, so the caller can reload the
        // manifest and retry against fresh base without rebuilding from
        // the WAL.
        let store = make_store();
        let paths = make_paths("e2e-flush-cas");
        let ms = ManifestStore::new(store, paths);
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        // A competitor advances the manifest to v1, so our `base` (at v0)
        // is stale and any flush against it must lose the pointer CAS.
        let competitor = base.manifest.next_version(fence.writer_id);
        let _ = ms.commit(&fence, &base, competitor).await.unwrap();

        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt.freeze();

        let err = flush(&ms, &fence, &base, &frozen, Schema::empty())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::ManifestCommitCas { .. }));
        // The caller still owns `frozen` and can retry.
        assert_eq!(frozen.len(), 1);
    }

    /// Helper: extract the trailing filename from a relative SST path.
    fn file_basename(path: &str) -> &str {
        path.rsplit('/').next().unwrap_or(path)
    }
}
