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
//! 4. PUT every SST body + every non-omitted bloom side-car to its immutable
//! UUIDv7 path. Small objects retain `PutMode::Create`; large bodies use
//! multipart upload, whose APIs do not support create-only semantics.
//! 5. Build a fresh manifest carrying every new [`SstDescriptor`] and clear
//! `wal_segments` (every record they reference is now durable inside an
//! SST).
//! 6. Commit through [`ManifestStore::commit`] (CAS).
//!
//! On any error before the manifest commit, every started upload is allowed to
//! finish its cleanup before the error is returned. Complete orphan SST/bloom
//! objects can survive in object storage; they cost space but cannot affect
//! correctness because no manifest version references them, and the janitor
//! sweeps them. Multipart errors are explicitly aborted; operators should also
//! configure an incomplete-multipart lifecycle rule for process crashes.

use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Date32Builder, FixedSizeBinaryBuilder, FixedSizeListBuilder,
    Float32Builder, Float64Builder, Int32Builder, Int64Builder, LargeStringBuilder, ListBuilder,
    StringBuilder, TimestampMicrosecondBuilder, UInt32Builder, UInt64Builder,
};
use arrow_array::{ArrayRef, RecordBatch};
use bytes::Bytes;
use chrono::Utc;
use futures::stream::{FuturesUnordered, StreamExt};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tracing::{debug, instrument};
use uuid::Uuid;

use namidb_core::{
    DataType, EdgeTypeDef, LabelDef, LabelDictionary, LabelId, PropertyDef, Schema, Value,
};

use crate::error::{Error, Result};
use crate::fence::WriterFence;
use crate::manifest::{
    KindSpecificStats, LoadedManifest, ManifestStore, PerLabelPropertyStat, SstDescriptor, SstKind,
    SstLevel,
};
use crate::memtable::{FrozenMemtable, MemKey, MemOp};
use crate::paths::NamespacePaths;
use crate::spooled_object::{
    put_spooled_object, MultipartUploadGuard, SpooledObject, MULTIPART_MAX_CONCURRENCY,
    MULTIPART_PART_SIZE, MULTIPART_THRESHOLD,
};
use crate::sst::bloom::{BloomDescriptor, BloomFilter};
use crate::sst::edges::inverse::transpose_forward_to_inverse;
use crate::sst::edges::writer::{
    EdgeRecord as EdgeStreamRow, EdgeSstBuild, EdgeSstWriter, EdgeSstWriterOptions,
};
use crate::sst::edges::EdgeDirection;
use crate::sst::hll::{Hll, DEFAULT_PRECISION};
use crate::sst::nodes::{
    max_scalar, min_scalar, node_arrow_schema, NodeSstFinish, NodeSstWriter, NodeSstWriterOptions,
};
use crate::sst::stats::StatScalar;

/// Process-wide bound for CPU/local-disk flush builds.
///
/// `spawn_blocking` tasks cannot be stopped after they begin. The permit is
/// therefore moved into the blocking closure, not held by the async waiter:
/// cancelling a request detaches at most one build, and a retry cannot start a
/// second corpus-sized encoder/spool until the first task has actually exited.
static FLUSH_BUILD_GATE: Semaphore = Semaphore::const_new(1);

// ── Wire-level records ─────────────────────────────────────────────────

/// Decoded payload of a [`MemOp::Upsert`](crate::memtable::MemOp::Upsert) for
/// a node. Stored bytes-on-wire encoding is JSON (see module docs for why).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NodeWriteRecord {
    #[serde(default)]
    pub properties: BTreeMap<String, Value>,
    #[serde(default)]
    pub schema_version: u64,
    /// The node's label set as interned [`LabelId`](namidb_core::LabelId)
    /// values, sorted and deduped. Carried in the value (not the key) so the
    /// id-primary memtable/SST keep one row per id regardless of label count.
    /// Empty for older payloads (`serde(default)`); the read path then falls
    /// back to the SST scope for legacy single-label data.
    #[serde(default)]
    pub labels: Vec<u32>,
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

// ── Exact-node sidecar records ────────────────────────────────────────

// A `.nloc2` object carries the ordinary NodeId -> row-ordinal tree first
// (for 2.0.4/2.0.5 readers) and an exact NodeId -> record tree after it. The
// latter deliberately stores the existing JSON wire payload: it is already
// the WAL/memtable compatibility format, so the accelerator cannot acquire a
// second property codec that drifts from recovery. Large JSON payloads (most
// notably float vectors) are compressed independently so one point read can
// fetch/decompress one node without inflating the sidecar to JSON size.
const NODE_RECORD_TOMBSTONE: u8 = 0;
const NODE_RECORD_RAW: u8 = 1;
const NODE_RECORD_ZSTD: u8 = 2;
const NODE_RECORD_HEADER_BYTES: usize = 1 + 8 + 4;

pub(crate) fn encode_exact_node_record(row: &NodeRow) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    match &row.op {
        MemOp::Tombstone => {
            out.reserve_exact(NODE_RECORD_HEADER_BYTES);
            out.push(NODE_RECORD_TOMBSTONE);
            out.extend_from_slice(&row.lsn.to_le_bytes());
            out.extend_from_slice(&0_u32.to_le_bytes());
        }
        MemOp::Upsert(payload) => {
            let raw_len = u32::try_from(payload.len())
                .map_err(|_| Error::invariant("exact node record exceeds 4 GiB"))?;
            let compressed = zstd::bulk::compress(payload, 1)
                .map_err(|e| Error::invariant(format!("exact node record zstd encode: {e}")))?;
            let (tag, body) = if compressed.len() < payload.len() {
                (NODE_RECORD_ZSTD, compressed.as_slice())
            } else {
                (NODE_RECORD_RAW, payload.as_ref())
            };
            out.reserve_exact(NODE_RECORD_HEADER_BYTES.saturating_add(body.len()));
            out.push(tag);
            out.extend_from_slice(&row.lsn.to_le_bytes());
            out.extend_from_slice(&raw_len.to_le_bytes());
            out.extend_from_slice(body);
        }
    }
    Ok(out)
}

pub(crate) fn decode_exact_node_record(bytes: &[u8]) -> Result<(u64, MemOp)> {
    if bytes.len() < NODE_RECORD_HEADER_BYTES {
        return Err(Error::invariant("truncated exact node record"));
    }
    let tag = bytes[0];
    let lsn = u64::from_le_bytes(
        bytes[1..9]
            .try_into()
            .map_err(|_| Error::invariant("invalid exact node record LSN"))?,
    );
    let raw_len = u32::from_le_bytes(
        bytes[9..13]
            .try_into()
            .map_err(|_| Error::invariant("invalid exact node record length"))?,
    ) as usize;
    let body = &bytes[NODE_RECORD_HEADER_BYTES..];
    let op = match tag {
        NODE_RECORD_TOMBSTONE if raw_len == 0 && body.is_empty() => MemOp::Tombstone,
        NODE_RECORD_TOMBSTONE => {
            return Err(Error::invariant(
                "tombstone exact node record carries a payload",
            ));
        }
        NODE_RECORD_RAW if body.len() == raw_len => MemOp::Upsert(Bytes::copy_from_slice(body)),
        NODE_RECORD_RAW => {
            return Err(Error::invariant(
                "raw exact node record length does not match its header",
            ));
        }
        NODE_RECORD_ZSTD => {
            let decoded = zstd::bulk::decompress(body, raw_len)
                .map_err(|e| Error::invariant(format!("exact node record zstd decode: {e}")))?;
            if decoded.len() != raw_len {
                return Err(Error::invariant(
                    "decoded exact node record length does not match its header",
                ));
            }
            MemOp::Upsert(Bytes::from(decoded))
        }
        _ => {
            return Err(Error::invariant(format!(
                "unknown exact node record encoding {tag}"
            )));
        }
    };
    Ok((lsn, op))
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

    let (node_rows, edge_buckets) = bucket_nodes_and_edges(frozen);

    let store = manifest_store.store().clone();
    let paths = manifest_store.paths();
    // 1. CPU/local-disk phase. Arrow/CSR encoding, Parquet compression and
    // the node spool's `sync_data` are deliberately kept off Tokio workers.
    // This also makes delayed-allocation errors surface before any object PUT.
    let build_paths: NamespacePaths = paths.clone();
    let build_schema = schema.clone();
    let build_label_dict = base.manifest.label_dict.clone();
    let (pendings, node_rows) = run_flush_build(move || {
        let pendings = build_pending_ssts(
            &build_paths,
            &node_rows,
            edge_buckets,
            &build_schema,
            &build_label_dict,
        )?;
        Ok((pendings, node_rows))
    })
    .await?;

    // Classify and build search deltas before the first object PUT. A segment
    // cap/build failure therefore cannot expose a Nodes SST without exact
    // search coverage.
    let search_plan = match pendings
        .iter()
        .find(|pending| pending.descriptor.kind == SstKind::Nodes)
    {
        Some(nodes) => {
            crate::search_lsm_flush::prepare_search_flush(
                manifest_store,
                base,
                &schema,
                &nodes.descriptor,
                &node_rows,
            )
            .await?
        }
        None => crate::search_lsm_flush::SearchFlushPlan::default(),
    };
    let (search_uploads, search_manifest_update) = search_plan.into_parts();
    let search_descriptor_count = search_manifest_update.descriptor_count();

    // 2. I/O phase — issue body + bloom PUTs with bounded object-level
    // concurrency. The PUTs are independent (each targets a fresh UUIDv7
    // path created above), so the only ordering constraint is that they all
    // complete before the manifest CAS. We deliberately collect every result
    // instead of short-circuiting: dropping a multipart future on the first
    // sibling failure would bypass its explicit abort cleanup. Complete
    // orphan objects are harmless and reclaimed by the janitor.
    let mut put_futures: Vec<ObjectUploadFuture> = Vec::with_capacity(pendings.len() * 2);
    let mut new_ssts = Vec::with_capacity(pendings.len());
    let mut bloom_count = 0usize;
    for p in pendings {
        let PendingSst {
            descriptor,
            body_path,
            body,
            bloom_path,
            bloom_body,
            index_sidecars,
        } = p;
        let path = body_path;
        let store_ref = store.clone();
        put_futures.push(Box::pin(async move {
            put_sidecar_payload(store_ref, &path, body).await
        }));
        if let (Some(path), Some(body)) = (bloom_path, bloom_body) {
            bloom_count += 1;
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
        for (path, body) in index_sidecars {
            let store_ref = store.clone();
            put_futures.push(Box::pin(async move {
                put_sidecar_payload(store_ref, &path, body).await
            }));
        }
        new_ssts.push(descriptor);
    }
    for upload in search_uploads {
        let store_ref = store.clone();
        put_futures.push(Box::pin(async move {
            put_sidecar_payload(store_ref, &upload.path, upload.body).await
        }));
    }
    await_all_object_uploads(put_futures).await?;

    let ssts_written = new_ssts.len().saturating_add(search_descriptor_count);

    let mut next = base.manifest.next_version(fence.writer_id);
    next.schema = schema;
    next.ssts.extend(new_ssts);
    search_manifest_update.apply(&mut next);
    next.wal_segments.clear();

    let committed = manifest_store.commit(fence, base, next).await?;

    Ok(FlushOutcome {
        committed,
        ssts_written,
        bloom_sidecars_written: bloom_count,
    })
}

pub(crate) async fn run_flush_build<T, F>(build: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let permit = FLUSH_BUILD_GATE
        .acquire()
        .await
        .map_err(|_| Error::invariant("flush SST build gate closed"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        build()
    })
    .await
    .map_err(|error| Error::invariant(format!("flush SST build task failed: {error}")))?
}

/// Per-SST work product: descriptor + body bytes + their object-store paths,
/// kept together so the parallel-PUT phase can issue them without re-touching
/// the schema/Arrow builders. `index_sidecars` contains optional Node lookup
/// accelerators and, for current forward Edge SSTs, the complete exact-edge
/// point sidecar.
#[derive(Debug)]
struct PendingSst {
    descriptor: SstDescriptor,
    body_path: Path,
    body: SidecarPayload,
    bloom_path: Option<Path>,
    bloom_body: Option<Bytes>,
    /// `(path, body)` for each optional accelerator emitted alongside this
    /// SST. Every object lands before the manifest can expose its marker.
    index_sidecars: Vec<(Path, SidecarPayload)>,
}

/// Upload body for an immutable lookup accelerator.
///
/// Most sidecars are compact enough to remain scatter/gather memory payloads.
/// Exact node and edge-point records can be corpus-sized. Node values are
/// spooled behind their existing in-memory locator prefix; edge-point pages
/// and values are independent spools consumed incrementally during multipart
/// PUT.
#[derive(Debug)]
pub(crate) enum SidecarPayload {
    InMemory(PutPayload),
    Spooled(SpooledObject),
    NodeLocator(crate::sst::paged_index::NodeLocatorRecordUpload),
    NodePropertyPages(crate::sst::nodes::property_pages::NodePropertyPageUpload),
    EdgePoint(crate::sst::paged_index::EdgePointIndexUpload),
    Paged(crate::sst::paged_index::SpooledPagedIndexUpload),
}

impl From<Bytes> for SidecarPayload {
    fn from(value: Bytes) -> Self {
        Self::InMemory(value.into())
    }
}

impl From<PutPayload> for SidecarPayload {
    fn from(value: PutPayload) -> Self {
        Self::InMemory(value)
    }
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

/// Convert the frozen memtable into ordered buckets. Nodes are id-primary, so
/// they collapse into ONE bucket spanning every label (a node's label set
/// rides in its value); memtable order (BTreeMap, nodes sort before edges and
/// by id within nodes) keeps that bucket node_id-ascending for free. Edges
/// still bucket by type, already sorted by `(src, dst)`.
fn bucket_nodes_and_edges(
    frozen: &FrozenMemtable,
) -> (Vec<NodeRow>, BTreeMap<String, Vec<EdgeRow>>) {
    let mut nodes: Vec<NodeRow> = Vec::new();
    let mut edges: BTreeMap<String, Vec<EdgeRow>> = BTreeMap::new();
    for (k, e) in frozen.iter() {
        match k {
            MemKey::Node { id } => {
                nodes.push(NodeRow {
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

fn build_pending_ssts(
    paths: &NamespacePaths,
    node_rows: &[NodeRow],
    edge_buckets: BTreeMap<String, Vec<EdgeRow>>,
    schema: &Schema,
    label_dict: &LabelDictionary,
) -> Result<Vec<PendingSst>> {
    let mut pendings = Vec::new();

    // Nodes: one identity-partitioned SST spanning every label, built with an
    // empty LabelDef (fixed layout — no prop_* columns; every property rides in
    // __overflow_json), plus a label->node-ids sidecar so `scan_label` resolves
    // without per-label partitions.
    if !node_rows.is_empty() {
        let column_label = LabelDef {
            name: String::new(),
            properties: Vec::new(),
        };
        // Equality-index sidecars are harvested from the record values keyed by
        // the schema's `indexed` properties, so the secondary index survives
        // the id-primary move.
        let index_props = union_indexed_props(schema);
        let finish = build_node_sst(&column_label, node_rows)?;
        pendings.push(prepare_node_pending(
            paths,
            &index_props,
            node_rows,
            finish,
            schema,
            label_dict,
        )?);
    }

    for (edge_type, rows) in edge_buckets {
        let edge_def = schema.edge_type(&edge_type).cloned();
        let declared_property_names: Vec<String> = edge_def
            .as_ref()
            .map(|def| {
                def.properties
                    .iter()
                    .map(|property| property.name.clone())
                    .collect()
            })
            .unwrap_or_default();
        let forward_rows = build_edge_stream_rows(&rows, &declared_property_names)?;
        let inverse_rows = transpose_forward_to_inverse(&forward_rows);
        let forward = build_edge_sst(
            &edge_type,
            edge_def.as_ref(),
            &forward_rows,
            EdgeDirection::Forward,
        )?;
        let inverse = build_edge_sst(
            &edge_type,
            edge_def.as_ref(),
            &inverse_rows,
            EdgeDirection::Inverse,
        )?;
        pendings.push(prepare_edge_pending(
            paths,
            &edge_type,
            EdgeDirection::Forward,
            forward,
        ));
        pendings.push(prepare_edge_pending(
            paths,
            &edge_type,
            EdgeDirection::Inverse,
            inverse,
        ));
    }

    Ok(pendings)
}

/// Union of every `indexed` or `unique` property across all schema labels, as
/// a synthetic `LabelDef` used ONLY to harvest a node SST's equality posting
/// sidecars (the columns themselves are built from an empty `LabelDef`).
///
/// A single-value unique sidecar cannot represent per-label uniqueness across
/// a multi-label SST, so `unique` is cleared and `indexed` is set instead. The
/// posting list safely represents duplicate values from different labels and
/// also gives label-agnostic `MATCH (n {key})` a global lookup route; callers
/// confirm each candidate's current value and labels.
pub(crate) fn union_indexed_props(schema: &Schema) -> LabelDef {
    let mut by_name: BTreeMap<String, PropertyDef> = BTreeMap::new();
    for label in schema.labels.values() {
        for p in &label.properties {
            if p.indexed || p.unique {
                let mut candidate = p.clone();
                candidate.unique = false;
                candidate.indexed = true;
                by_name
                    .entry(p.name.clone())
                    .and_modify(|current| {
                        // The synthetic definition is global: declarations
                        // with the same property name may legitimately use
                        // different types on different labels.  Its type is
                        // only a capability marker for the equality
                        // harvester/planner, so never let an arbitrary
                        // lexically-first unsupported declaration (for
                        // example Int64) hide a later String/Bool index.
                        let current_supported = matches!(
                            current.data_type,
                            DataType::Utf8 | DataType::LargeUtf8 | DataType::Bool
                        );
                        let candidate_supported = matches!(
                            p.data_type,
                            DataType::Utf8 | DataType::LargeUtf8 | DataType::Bool
                        );
                        if !current_supported && candidate_supported {
                            *current = candidate.clone();
                        }
                    })
                    .or_insert(candidate);
            }
        }
    }
    LabelDef {
        name: String::new(),
        properties: by_name.into_values().collect(),
    }
}

// ── Node SST building ──────────────────────────────────────────────────

/// Rows buffered per `RecordBatch` handed to the underlying
/// [`NodeSstWriter`]. Bounds the second materialisation of the row data
/// (Arrow builders) regardless of how many rows the caller streams in.
pub(crate) const NODE_SST_BATCH_ROWS: usize = 16 * 1024;

/// Incremental node-SST writer: accepts reconciled [`NodeRow`]s one at a
/// time (`node_id` ascending), buffers them into bounded chunks, and feeds
/// each chunk to the underlying [`NodeSstWriter`] as its own
/// `RecordBatch`. [`build_node_sst`] and the compaction streaming merge
/// (`compact.rs`) share it so the two write paths cannot drift: peak
/// memory per output SST is one chunk of rows plus the Parquet encoder's
/// own state, independent of the total row count.
pub(crate) struct IncrementalNodeSstWriter {
    writer: NodeSstWriter,
    arrow_schema: arrow_schema::SchemaRef,
    label: LabelDef,
    chunk: Vec<NodeRow>,
    chunk_rows: usize,
}

impl IncrementalNodeSstWriter {
    pub(crate) fn new(
        label: &LabelDef,
        options: NodeSstWriterOptions,
        chunk_rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            writer: NodeSstWriter::new(label.clone(), options)?,
            arrow_schema: node_arrow_schema(label),
            label: label.clone(),
            chunk: Vec::new(),
            chunk_rows: chunk_rows.max(1),
        })
    }

    /// Buffer one row; flushes a full chunk into the Parquet writer.
    pub(crate) fn push(&mut self, row: NodeRow) -> Result<()> {
        self.chunk.push(row);
        if self.chunk.len() >= self.chunk_rows {
            self.flush_chunk()?;
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> Result<()> {
        if self.chunk.is_empty() {
            return Ok(());
        }
        let batch = build_node_record_batch(&self.arrow_schema, &self.label, &self.chunk)?;
        if batch.num_rows() > 0 {
            self.writer.write_batch(&batch)?;
        }
        self.chunk.clear();
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<NodeSstFinish> {
        self.flush_chunk()?;
        self.writer.finish()
    }
}

pub(crate) fn build_node_sst(label: &LabelDef, rows: &[NodeRow]) -> Result<NodeSstFinish> {
    let options = NodeSstWriterOptions {
        expected_keys: rows.len() as u64,
        ..Default::default()
    };
    // Feed the writer in bounded chunks rather than one monolithic
    // RecordBatch: a deep-level compaction merges the whole bucket, and the
    // single all-rows batch (every column fully materialised a second time)
    // was the largest single allocation of the merge. The writer accepts any
    // number of ascending-ordered batches; `rows` are already sorted.
    let mut writer = IncrementalNodeSstWriter::new(label, options, NODE_SST_BATCH_ROWS)?;
    for row in rows {
        writer.push(row.clone())?;
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
    let mut labels_b = ListBuilder::new(UInt32Builder::new());
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
                for &lid in &rec.labels {
                    labels_b.values().append_value(lid);
                }
                labels_b.append(true);

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
                labels_b.append(true); // empty label list on a tombstone row
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
    columns.push(Arc::new(labels_b.finish()));
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
) -> Result<EdgeSstBuild> {
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
    writer.finish_with_point_index()
}

// ── PUT helpers ────────────────────────────────────────────────────────

fn prepare_node_pending(
    paths: &NamespacePaths,
    label_def: &LabelDef,
    rows: &[NodeRow],
    finish: NodeSstFinish,
    schema: &Schema,
    label_dict: &LabelDictionary,
) -> Result<PendingSst> {
    let id = Uuid::now_v7();
    let level = SstLevel::L0;
    let file_name = format!(
        "{}-{}.parquet",
        uuid_path_id(&id),
        SstKind::Nodes.path_tag()
    );
    let body_path = paths.sst_object(level.as_u32(), &file_name);
    let relative_path = relative_sst_path(level.as_u32(), &file_name);
    let body_len = finish.body.len() as u64;

    let (bloom_descriptor, bloom_path, bloom_body) = prepare_bloom_sidecar(
        paths,
        level.as_u32(),
        &id,
        SstKind::Nodes.path_tag(),
        "",
        finish.bloom,
    );

    // Node SSTs are no longer partitioned by label and are built with an empty
    // LabelDef, so the declared-property sidecars harvest nothing; the calls
    // stay wired for symmetry and a future typed-column layout.
    let (unique_property_indices, index_sidecars) =
        prepare_unique_property_sidecars(paths, level.as_u32(), &id, "", label_def, rows)?;
    let mut index_sidecars: Vec<(Path, SidecarPayload)> = index_sidecars;
    let (equality_property_indices, equality_sidecars) =
        prepare_equality_property_sidecars(paths, level.as_u32(), &id, "", label_def, rows)?;
    index_sidecars.extend(equality_sidecars);

    // The label index (`LabelId -> [NodeId, ...]`) replaces "the SST partition
    // IS the label index" now that one node SST spans every label.
    let (label_index, label_sidecar) =
        prepare_label_index_sidecar(paths, level.as_u32(), &id, rows)?;
    if let Some((path, body)) = label_sidecar {
        index_sidecars.push((path, body));
    }
    let mut locator_records = crate::sst::paged_index::NodeLocatorRecordBuilder::new();
    for row in rows {
        let record = encode_exact_node_record(row)?;
        locator_records.push(&row.id, &record)?;
    }
    let (mut node_locator, locator_sidecar) =
        prepare_node_locator_sidecar(paths, level.as_u32(), &id, locator_records)?;
    index_sidecars.push(locator_sidecar);

    let stats = finish.stats;
    // Transitional dual-write: Parquet remains the authoritative complete
    // node representation (and exact fallback), while `.npp` supplies
    // independently ranged schemaless projection. Do not remove overflow
    // properties until the node-wire compatibility barrier moves in a
    // dedicated format migration.
    let mut property_builder =
        crate::sst::nodes::property_pages::NodePropertyPageBuilder::new_bound(
            crate::sst::nodes::property_pages::NodePropertyPageConfig::from_env()?,
            id,
        )?;
    for (ordinal, row) in rows.iter().enumerate() {
        let ordinal = u64::try_from(ordinal)
            .map_err(|_| Error::invariant("node property ordinal exceeds u64"))?;
        match &row.op {
            MemOp::Upsert(payload) => {
                let record = NodeWriteRecord::decode(payload)?;
                property_builder.push_sorted(row.id, ordinal, &record.properties)?;
            }
            MemOp::Tombstone => {
                property_builder.push_sorted(row.id, ordinal, &BTreeMap::new())?;
            }
        }
    }
    let property_upload = property_builder.finish()?;
    let property_stats = property_upload.stats();
    if property_upload.sst_id() != id || property_stats.node_count != stats.row_count {
        return Err(Error::invariant(
            "node property pages are not bound to their Nodes SST",
        ));
    }
    let (property_pages, property_sidecar) = prepare_node_property_pages_upload_sidecar(
        paths,
        level.as_u32(),
        &id,
        &id,
        property_upload,
    )?;
    node_locator.property_pages = Some(property_pages);
    index_sidecars.push(property_sidecar);

    let descriptor = SstDescriptor {
        id,
        kind: SstKind::Nodes,
        scope: String::new(),
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
        node_locator: Some(node_locator),
        per_label_property_stats: compute_per_label_property_stats(rows, schema, label_dict)?,
    };

    Ok(PendingSst {
        descriptor,
        body_path,
        body: finish.body.into(),
        bloom_path,
        bloom_body,
        index_sidecars,
    })
}

/// Build the exact `NodeId -> row ordinal` locator paired with a node SST.
///
/// The input order is the physical Parquet row order. Current flush and
/// compaction writers both guarantee strict NodeId ordering, which also makes
/// the locator's B+tree input sorted without an additional allocation/sort.
pub(crate) fn prepare_node_locator_sidecar(
    paths: &NamespacePaths,
    level: u32,
    sst_id: &Uuid,
    builder: crate::sst::paged_index::NodeLocatorRecordBuilder,
) -> Result<(
    crate::manifest::NodeLocatorDescriptor,
    (Path, SidecarPayload),
)> {
    let upload = builder.finish_upload()?;
    prepare_node_locator_upload_sidecar(paths, level, sst_id, upload)
}

/// Attach paths and manifest metadata to an already-finalised node locator.
///
/// Compaction finalises the corpus-sized exact-record spool inside its
/// blocking merge task, before returning to Tokio or rebuilding search
/// indexes. Flush uses [`prepare_node_locator_sidecar`] because its entire
/// build phase already runs on the blocking pool.
pub(crate) fn prepare_node_locator_upload_sidecar(
    paths: &NamespacePaths,
    level: u32,
    sst_id: &Uuid,
    upload: crate::sst::paged_index::NodeLocatorRecordUpload,
) -> Result<(
    crate::manifest::NodeLocatorDescriptor,
    (Path, SidecarPayload),
)> {
    let file_name = format!(
        "{}-{}.nloc2",
        uuid_path_id(sst_id),
        SstKind::Nodes.path_tag()
    );
    let object_path = paths.sst_object(level, &file_name);
    let relative = relative_sst_path(level, &file_name);
    let descriptor = crate::manifest::NodeLocatorDescriptor {
        path: relative,
        size_bytes: upload.size_bytes(),
        entry_count: upload.entry_count(),
        property_pages: None,
    };
    Ok((
        descriptor,
        (object_path, SidecarPayload::NodeLocator(upload)),
    ))
}

/// Attach canonical path and manifest binding metadata to finished schemaless
/// node-property pages. `object_path_id` may be a fresh retry UUID for a
/// sidecar-only migration while `parent_sst_id` remains the immutable Parquet
/// UUID authenticated by the header/footer.
pub(crate) fn prepare_node_property_pages_upload_sidecar(
    paths: &NamespacePaths,
    level: u32,
    parent_sst_id: &Uuid,
    object_path_id: &Uuid,
    upload: crate::sst::nodes::property_pages::NodePropertyPageUpload,
) -> Result<(
    crate::manifest::NodePropertyPagesDescriptor,
    (Path, SidecarPayload),
)> {
    if upload.sst_id() != *parent_sst_id {
        return Err(Error::invariant(
            "node property pages UUID differs from parent Nodes SST",
        ));
    }
    let stats = upload.stats();
    let file_name = crate::paths::node_property_pages_file_name(object_path_id);
    let object_path = paths.node_property_pages(level, object_path_id);
    let descriptor = crate::manifest::NodePropertyPagesDescriptor {
        id: *parent_sst_id,
        parent_sst_id: *parent_sst_id,
        path: relative_sst_path(level, &file_name),
        size_bytes: upload.size_bytes(),
        format_version: crate::sst::nodes::property_pages::NODE_PROPERTY_PAGES_FORMAT_VERSION,
        node_count: stats.node_count,
        cell_count: stats.cell_count,
        property_count: stats.property_count,
        page_count: stats.page_count,
        content_xxh3: upload.content_xxh3(),
    };
    Ok((
        descriptor,
        (object_path, SidecarPayload::NodePropertyPages(upload)),
    ))
}

/// Build the per-SST label membership sidecar. Observations are externally
/// sorted into composite `(LabelId big-endian, NodeId) -> ()` PagedV2 keys, so
/// one label can be point-probed or cursor-paged without a corpus-sized
/// posting allocation. Tombstones contribute nothing (last-LSN-wins at read
/// time handles removal). Returns `(None, None)` when no live labelled node is
/// present.
/// Output of [`prepare_label_index_sidecar`]: the manifest descriptor plus the
/// `(path, body)` to PUT next to the SST. Aliased to keep clippy's
/// type-complexity lint happy, mirroring the unique/equality sidecar aliases.
type LabelIndexSidecar = (
    Option<crate::manifest::LabelIndexDescriptor>,
    Option<(Path, SidecarPayload)>,
);

pub(crate) fn prepare_label_index_sidecar(
    paths: &NamespacePaths,
    level: u32,
    sst_id: &Uuid,
    rows: &[NodeRow],
) -> Result<LabelIndexSidecar> {
    let mut collector = LabelIndexCollector::new()?;
    for row in rows {
        if let MemOp::Upsert(payload) = &row.op {
            let rec = NodeWriteRecord::decode(payload)?;
            collector.observe(row.id, &rec)?;
        }
    }
    collector.finish(paths, level, sst_id)
}

/// Streaming harvester behind [`prepare_label_index_sidecar`]: the
/// compaction merge feeds it one reconciled winner row at a time (id
/// ascending, tombstones excluded) so the label postings never require the
/// whole merged bucket in memory — only the posting lists themselves.
#[derive(Debug)]
pub(crate) struct LabelIndexCollector {
    sorter: crate::sst::external_pairs::ExternalPairSorter,
}

impl LabelIndexCollector {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            sorter: crate::sst::external_pairs::ExternalPairSorter::from_env()?,
        })
    }

    pub(crate) fn observe(&mut self, id: [u8; 16], rec: &NodeWriteRecord) -> Result<()> {
        for &lid in &rec.labels {
            // The external sort key is `(property,key,id)`. Reuse `property`
            // for LabelId and an empty value key, yielding exact
            // `(LabelId,NodeId)` order without retaining posting vectors.
            self.sorter.push(lid, &[], id)?;
        }
        Ok(())
    }

    /// Emit a composite `(LabelId BE,NodeId)->()` tree directly. A fixed
    /// footer binds the tree to exact per-label manifest counts.
    pub(crate) fn finish(
        self,
        paths: &NamespacePaths,
        level: u32,
        sst_id: &Uuid,
    ) -> Result<LabelIndexSidecar> {
        let mut sorted = self.sorter.finish()?;
        let mut builder =
            crate::sst::paged_index::SortedSpooledPagedIndexBuilder::label_membership();
        let mut per_label_counts = Vec::new();
        let mut active_label = None;
        let mut active_count = 0_u64;
        let mut posting_count = 0_u64;
        while let Some(pair) = sorted.next_pair()? {
            if !pair.key.is_empty() {
                return Err(Error::invariant(
                    "label-index scratch record unexpectedly carries a property key",
                ));
            }
            if active_label != Some(pair.property) {
                if let Some(label) = active_label {
                    per_label_counts.push((label, active_count));
                }
                active_label = Some(pair.property);
                active_count = 0;
            }
            let mut key = [0_u8; 20];
            key[..4].copy_from_slice(&pair.property.to_be_bytes());
            key[4..].copy_from_slice(&pair.id);
            builder.push_inline(&key, &[])?;
            active_count = active_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("per-label posting count exceeds u64"))?;
            posting_count = posting_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("label posting count exceeds u64"))?;
        }
        if let Some(label) = active_label {
            per_label_counts.push((label, active_count));
        }
        if posting_count == 0 {
            return Ok((None, None));
        }
        let upload = builder
            .finish()?
            .bind_label_counts(&per_label_counts, *sst_id.as_bytes())?;
        let label_count = per_label_counts.len() as u64;
        let file_name = format!(
            "{}-{}.labelidx.pidx",
            uuid_path_id(sst_id),
            SstKind::Nodes.path_tag()
        );
        let object_path = paths.sst_object(level, &file_name);
        let relative = relative_sst_path(level, &file_name);
        let descriptor = crate::manifest::LabelIndexDescriptor {
            path: relative,
            size_bytes: upload.size_bytes(),
            label_count,
            posting_count,
            format: crate::manifest::PropertyIndexFormat::PagedV1,
            per_label_counts,
        };
        Ok((
            Some(descriptor),
            Some((object_path, SidecarPayload::Paged(upload))),
        ))
    }
}

/// Compute per-(label, property) statistics for an id-primary node SST
/// (RFC 025). Walks the reconciled rows, and for every label a row carries,
/// folds each scalar property value into a `(LabelId, property)` accumulator:
/// min/max, an HLL for ndv, and a non-null count. `null_count` is then the
/// label's live-row count minus the non-null count (a property absent / null /
/// non-scalar on a row carrying the label counts as null). Tombstones
/// contribute nothing (only `MemOp::Upsert` rows are folded).
pub(crate) fn compute_per_label_property_stats(
    rows: &[NodeRow],
    schema: &Schema,
    label_dict: &LabelDictionary,
) -> Result<Vec<PerLabelPropertyStat>> {
    let mut collector = PerLabelStatsCollector::new();
    for row in rows {
        let MemOp::Upsert(payload) = &row.op else {
            continue;
        };
        let rec = NodeWriteRecord::decode(payload)?;
        collector.observe(&rec);
    }
    collector.finish(schema, label_dict)
}

/// One `(LabelId, property)` accumulator of [`PerLabelStatsCollector`].
struct PerLabelStatAcc {
    min: Option<StatScalar>,
    max: Option<StatScalar>,
    hll: Hll,
    non_null: u64,
}

/// Streaming accumulator behind [`compute_per_label_property_stats`]: the
/// compaction merge feeds it one reconciled winner record at a time, so the
/// stats only ever hold the per-`(label, property)` accumulators — never
/// the merged rows themselves.
pub(crate) struct PerLabelStatsCollector {
    accs: BTreeMap<(u32, String), PerLabelStatAcc>,
    live_per_label: BTreeMap<u32, u64>,
}

impl PerLabelStatsCollector {
    pub(crate) fn new() -> Self {
        Self {
            accs: BTreeMap::new(),
            live_per_label: BTreeMap::new(),
        }
    }

    pub(crate) fn observe(&mut self, rec: &NodeWriteRecord) {
        for &lid in &rec.labels {
            *self.live_per_label.entry(lid).or_default() += 1;
            for (name, value) in &rec.properties {
                let Some(scalar) = value_to_stat_scalar(value) else {
                    continue;
                };
                let acc = self
                    .accs
                    .entry((lid, name.clone()))
                    .or_insert_with(|| PerLabelStatAcc {
                        min: None,
                        max: None,
                        hll: Hll::new(DEFAULT_PRECISION),
                        non_null: 0,
                    });
                acc.non_null += 1;
                acc.hll.add_scalar(&scalar);
                acc.min = Some(match acc.min.take() {
                    Some(prev) => min_scalar(prev, scalar.clone()),
                    None => scalar.clone(),
                });
                acc.max = Some(match acc.max.take() {
                    Some(prev) => max_scalar(prev, scalar),
                    None => scalar,
                });
            }
        }
    }

    /// Fold the accumulators into the manifest-level stat rows.
    pub(crate) fn finish(
        self,
        schema: &Schema,
        label_dict: &LabelDictionary,
    ) -> Result<Vec<PerLabelPropertyStat>> {
        let Self {
            mut accs,
            live_per_label,
        } = self;

        // Seed declared-but-absent properties. A property declared on a label
        // but null on every row of this SST yields no accumulator above, so it
        // would emit no entry; the cost model's backfill (`non_null =
        // node_count - null_count`) then leaves `null_count = 0` and the
        // property reads as fully non-null. Seeding an empty accumulator for
        // every declared property of a label present here makes its
        // `null_count` resolve to `live` (all rows null), which is correct and
        // additive across SSTs. Only labels present in this SST are seeded;
        // their declared set comes from the schema, resolved through the label
        // dictionary (an un-resolvable or undeclared label simply falls back
        // to the observed-only behavior).
        for &label_id in live_per_label.keys() {
            let Some(name) = label_dict.name(LabelId(label_id)) else {
                continue;
            };
            let Some(def) = schema.label(name) else {
                continue;
            };
            for prop in &def.properties {
                accs.entry((label_id, prop.name.clone()))
                    .or_insert_with(|| PerLabelStatAcc {
                        min: None,
                        max: None,
                        hll: Hll::new(DEFAULT_PRECISION),
                        non_null: 0,
                    });
            }
        }

        let mut out = Vec::with_capacity(accs.len());
        for ((label_id, property), acc) in accs {
            let live = live_per_label
                .get(&label_id)
                .copied()
                .unwrap_or(acc.non_null);
            out.push(PerLabelPropertyStat {
                label_id,
                property,
                null_count: live.saturating_sub(acc.non_null),
                min: acc.min,
                max: acc.max,
                ndv_estimate: if acc.hll.is_empty() {
                    None
                } else {
                    Some(acc.hll.to_sketch_bytes())
                },
            });
        }
        Ok(out)
    }
}

/// Map a core [`Value`] to the [`StatScalar`] the optimizer's min/max/ndv
/// machinery understands. Returns `None` for non-scalar values (null, vector,
/// list, map) which carry no useful per-column statistic.
fn value_to_stat_scalar(v: &Value) -> Option<StatScalar> {
    match v {
        Value::Bool(b) => Some(StatScalar::Bool(*b)),
        Value::I64(i) => Some(StatScalar::Int64(*i)),
        Value::F64(f) => Some(StatScalar::Float64(*f)),
        Value::Str(s) => Some(StatScalar::Utf8(s.clone())),
        Value::Bytes(b) => Some(StatScalar::Binary(b.clone())),
        Value::Date(d) => Some(StatScalar::Date32(*d)),
        Value::DateTime(m) => Some(StatScalar::TimestampMicrosUtc(*m)),
        Value::Null | Value::Vec(_) | Value::VecI8 { .. } | Value::List(_) | Value::Map(_) => None,
    }
}

/// Parallel outputs of [`prepare_unique_property_sidecars`]: the
/// per-property index descriptors (destined for the manifest) and the
/// matching `(path, body)` objects to PUT next to the SST body. Aliased
/// to keep the return type under clippy's type-complexity threshold.
type UniquePropertySidecars = (
    Vec<crate::manifest::UniquePropertyIndexDescriptor>,
    Vec<(Path, SidecarPayload)>,
);

/// Explicit rolling-upgrade mode for old readers that require the monolithic
/// bincode property map. Zero/default emits only PagedV2. A positive cap is a
/// hard per-sidecar bound: crossing it fails the flush/compaction rather than
/// silently dropping either authoritative representation.
/// Size cap for the legacy bincode property sidecar.
///
/// The legacy body stays authoritative by default: it is the only shape a key
/// too large to page can fall back to, and dropping it would leave such a
/// property with no index at all. An explicit `0` opts out entirely; any other
/// value caps the spool, and exceeding that cap degrades to paged-only rather
/// than failing the flush.
fn legacy_property_index_max_bytes() -> Result<Option<u64>> {
    match std::env::var("NAMIDB_LEGACY_PROPERTY_INDEX_MAX_BYTES") {
        Ok(raw) => {
            let bytes = raw.parse::<u64>().map_err(|error| {
                Error::precondition(format!(
                    "invalid NAMIDB_LEGACY_PROPERTY_INDEX_MAX_BYTES={raw:?}: {error}"
                ))
            })?;
            Ok((bytes > 0).then_some(bytes))
        }
        Err(std::env::VarError::NotPresent) => Ok(Some(u64::MAX)),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::precondition(
            "NAMIDB_LEGACY_PROPERTY_INDEX_MAX_BYTES is not valid UTF-8",
        )),
    }
}

struct LegacyMapSpool {
    file: std::fs::File,
    len: u64,
    entries: u64,
    limit: u64,
}

impl LegacyMapSpool {
    fn new(limit: u64) -> Result<Self> {
        let mut this = Self {
            file: crate::sst::paged_index::create_spool_file()?,
            len: 0,
            entries: 0,
            limit,
        };
        this.append(&0_u64.to_le_bytes())?;
        Ok(this)
    }

    fn write_unique(&mut self, key: &[u8], id: &[u8; 16]) -> Result<()> {
        self.write_string(key)?;
        self.append(id)?;
        self.bump_entries()
    }

    fn write_posting(
        &mut self,
        key: &[u8],
        posting: &mut std::fs::File,
        posting_len: u64,
    ) -> Result<()> {
        if posting_len % 16 != 0 {
            return Err(Error::invariant(
                "legacy equality posting is not NodeId-aligned",
            ));
        }
        self.write_string(key)?;
        self.append(&(posting_len / 16).to_le_bytes())?;
        self.ensure_capacity(posting_len)?;
        posting.rewind()?;
        let copied = std::io::copy(posting, &mut self.file)?;
        if copied != posting_len {
            return Err(Error::invariant(
                "legacy property posting scratch length changed",
            ));
        }
        self.len += copied;
        self.bump_entries()
    }

    fn finish(mut self) -> Result<(std::fs::File, u64)> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.entries.to_le_bytes())?;
        if self.file.metadata()?.len() != self.len {
            return Err(Error::invariant("legacy property map spool length changed"));
        }
        self.file.sync_data()?;
        self.file.rewind()?;
        Ok((self.file, self.len))
    }

    fn write_string(&mut self, key: &[u8]) -> Result<()> {
        let key_len = u64::try_from(key.len())
            .map_err(|_| Error::invariant("legacy property key exceeds u64"))?;
        self.append(&key_len.to_le_bytes())?;
        self.append(key)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        self.ensure_capacity(bytes.len() as u64)?;
        self.file.write_all(bytes)?;
        self.len += bytes.len() as u64;
        Ok(())
    }

    fn ensure_capacity(&self, additional: u64) -> Result<()> {
        let next = self
            .len
            .checked_add(additional)
            .ok_or_else(|| Error::invariant("legacy property map exceeds u64"))?;
        if next > self.limit {
            return Err(Error::precondition(format!(
                "legacy property sidecar requires {next} bytes, above \
                 NAMIDB_LEGACY_PROPERTY_INDEX_MAX_BYTES={}",
                self.limit
            )));
        }
        Ok(())
    }

    fn bump_entries(&mut self) -> Result<()> {
        self.entries = self
            .entries
            .checked_add(1)
            .ok_or_else(|| Error::invariant("legacy property map entry count exceeds u64"))?;
        Ok(())
    }
}

/// For every `PropertyDef::unique == true` in `label_def.properties`,
/// walk `rows`, harvest `(value_string, NodeId)` pairs through a bounded
/// external sort, and produce one direct range-readable PagedV2
/// `(UniquePropertyIndexDescriptor, (path, body))` pair per property.
///
/// Returns the parallel collections so the descriptor can land in the
/// manifest and the body can be PUT alongside the SST body.
///
/// Tombstoned rows contribute nothing — they're encoded in the SST body
/// and the reader's last-LSN-wins logic surfaces them correctly. Rows
/// without the property (nullable column, schema-evolved out, ...) contribute
/// nothing either. Non-string property values are skipped. An explicitly
/// configured bounded legacy-migration mode also streams the old bincode map.
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
    let mut collector = UniqueSidecarCollector::new(label_def)?;
    for row in rows {
        if let MemOp::Upsert(payload) = &row.op {
            let rec = NodeWriteRecord::decode(payload)?;
            collector.observe(row.id, &rec)?;
        }
    }
    collector.finish(paths, level, sst_id, label)
}

/// Streaming harvester behind [`prepare_unique_property_sidecars`]: fed one
/// reconciled winner row at a time by the compaction merge, so only the
/// sidecar-relevant `(value → id)` maps stay in memory. Entries keep the
/// def's property order so the emitted descriptors match the row-slice path
/// exactly.
#[derive(Debug)]
pub(crate) struct UniqueSidecarCollector {
    properties: Vec<String>,
    sorter: crate::sst::external_pairs::ExternalPairSorter,
}

impl UniqueSidecarCollector {
    pub(crate) fn new(label_def: &LabelDef) -> Result<Self> {
        let properties: Vec<String> = label_def
            .properties
            .iter()
            .filter(|p| p.unique)
            .map(|p| p.name.clone())
            .collect();
        u32::try_from(properties.len())
            .map_err(|_| Error::invariant("unique property count exceeds u32"))?;
        Ok(Self {
            properties,
            sorter: crate::sst::external_pairs::ExternalPairSorter::from_env()?,
        })
    }

    pub(crate) fn observe(&mut self, id: [u8; 16], rec: &NodeWriteRecord) -> Result<()> {
        for (ordinal, name) in self.properties.iter().enumerate() {
            if let Some(Value::Str(s)) = rec.properties.get(name) {
                self.sorter.push(ordinal as u32, s.as_bytes(), id)?;
            }
        }
        Ok(())
    }

    /// Externally sort and emit one authoritative PagedV2 sidecar per
    /// non-empty property. Duplicate values retain the greatest NodeId,
    /// matching the previous id-ordered `BTreeMap::insert` behavior.
    pub(crate) fn finish(
        self,
        paths: &NamespacePaths,
        level: u32,
        sst_id: &Uuid,
        label: &str,
    ) -> Result<UniquePropertySidecars> {
        let mut sorted = self.sorter.finish()?;
        let mut builders: Vec<_> = self
            .properties
            .iter()
            .map(|_| crate::sst::paged_index::SortedSpooledPagedIndexBuilder::unique())
            .collect();
        let legacy_limit = legacy_property_index_max_bytes()?;
        let mut legacy: Vec<Option<LegacyMapSpool>> = self
            .properties
            .iter()
            .map(|_| legacy_limit.map(LegacyMapSpool::new).transpose())
            .collect::<Result<_>>()?;
        let mut counts = vec![0_u64; self.properties.len()];
        let mut pending: Option<crate::sst::external_pairs::ExternalPair> = None;
        while let Some(pair) = sorted.next_pair()? {
            if pair.property as usize >= builders.len() {
                return Err(Error::invariant(
                    "unique-index scratch property ordinal is out of range",
                ));
            }
            match &mut pending {
                Some(previous)
                    if previous.property == pair.property && previous.key == pair.key =>
                {
                    // Runs are sorted by NodeId after `(property,key)`, so the
                    // last duplicate exactly reproduces prior last-insert.
                    previous.id = pair.id;
                }
                Some(_) => {
                    let previous = pending.take().expect("matched Some above");
                    let ordinal = previous.property as usize;
                    builders[ordinal].push_inline(&previous.key, &previous.id)?;
                    if let Some(legacy) = &mut legacy[ordinal] {
                        legacy.write_unique(&previous.key, &previous.id)?;
                    }
                    counts[ordinal] = counts[ordinal]
                        .checked_add(1)
                        .ok_or_else(|| Error::invariant("unique-index count exceeds u64"))?;
                    pending = Some(pair);
                }
                None => pending = Some(pair),
            }
        }
        if let Some(previous) = pending {
            let ordinal = previous.property as usize;
            builders[ordinal].push_inline(&previous.key, &previous.id)?;
            if let Some(legacy) = &mut legacy[ordinal] {
                legacy.write_unique(&previous.key, &previous.id)?;
            }
            counts[ordinal] = counts[ordinal]
                .checked_add(1)
                .ok_or_else(|| Error::invariant("unique-index count exceeds u64"))?;
        }

        let mut descriptors = Vec::new();
        let mut bodies = Vec::new();
        for (((name, builder), legacy), entry_count) in self
            .properties
            .into_iter()
            .zip(builders)
            .zip(legacy)
            .zip(counts)
        {
            if entry_count == 0 {
                continue;
            }
            // See the equality path: an unpageable key keeps the legacy body
            // instead of failing the flush.
            let declined = builder.declined();
            let upload = if declined {
                None
            } else {
                Some(builder.finish()?)
            };
            let paged_name = format!(
                "{}-{}-{}.idx_{}.pidx",
                uuid_path_id(sst_id),
                SstKind::Nodes.path_tag(),
                label,
                name,
            );
            let paged_path = paths.sst_object(level, &paged_name);
            let paged_relative = relative_sst_path(level, &paged_name);
            let paged_size = upload.as_ref().map_or(0, |upload| upload.size_bytes());
            if let Some(legacy) = legacy {
                let legacy_name = format!(
                    "{}-{}-{}.idx_{}.bin",
                    uuid_path_id(sst_id),
                    SstKind::Nodes.path_tag(),
                    label,
                    name,
                );
                let legacy_path = paths.sst_object(level, &legacy_name);
                let legacy_relative = relative_sst_path(level, &legacy_name);
                let (legacy_file, legacy_size) = legacy.finish()?;
                descriptors.push(crate::manifest::UniquePropertyIndexDescriptor {
                    property: name,
                    path: legacy_relative,
                    size_bytes: legacy_size,
                    entry_count,
                    format: crate::manifest::PropertyIndexFormat::BincodeV0,
                    paged: upload
                        .as_ref()
                        .map(|_| crate::manifest::PagedPropertyIndexDescriptor {
                            path: paged_relative,
                            size_bytes: paged_size,
                        }),
                    paged_build_unsupported: declined,
                });
                bodies.push((
                    legacy_path,
                    SidecarPayload::Spooled(SpooledObject::from_file(legacy_file, legacy_size)),
                ));
                if let Some(upload) = upload {
                    bodies.push((paged_path, SidecarPayload::Paged(upload)));
                }
            } else {
                let Some(upload) = upload else {
                    continue;
                };
                descriptors.push(crate::manifest::UniquePropertyIndexDescriptor {
                    property: name,
                    path: paged_relative,
                    size_bytes: paged_size,
                    entry_count,
                    format: crate::manifest::PropertyIndexFormat::PagedV1,
                    paged: None,
                    paged_build_unsupported: false,
                });
                bodies.push((paged_path, SidecarPayload::Paged(upload)));
            }
        }
        Ok((descriptors, bodies))
    }
}

/// Parallel outputs of [`prepare_equality_property_sidecars`].
type EqualityPropertySidecars = (
    Vec<crate::manifest::EqualityIndexDescriptor>,
    Vec<(Path, SidecarPayload)>,
);

/// For every `PropertyDef::indexed == true`, harvest `value_string ->
/// [NodeId, ...]` posting lists from `rows` and emit one sidecar per
/// property. Unlike [`prepare_unique_property_sidecars`] a value maps to
/// MANY ids, so the reader unions postings across SSTs and confirms each
/// candidate against the node's current value (which discards tombstoned
/// or value-changed ids). Scalar-v1 currently materialises String and Boolean
/// declarations. Numeric Cypher equality crosses I64/F64 domains, while
/// temporal values require schema coercion; both retain the exact scan
/// fallback until their canonical encodings are proven end-to-end.
///
/// `pub(crate)` so `compact.rs` can re-emit the sidecars on L0->L1 merge.
pub(crate) fn prepare_equality_property_sidecars(
    paths: &NamespacePaths,
    level: u32,
    sst_id: &Uuid,
    label: &str,
    label_def: &LabelDef,
    rows: &[NodeRow],
) -> Result<EqualityPropertySidecars> {
    let mut collector = EqualitySidecarCollector::new(label_def)?;
    for row in rows {
        if let MemOp::Upsert(payload) = &row.op {
            let rec = NodeWriteRecord::decode(payload)?;
            collector.observe(row.id, &rec)?;
        }
    }
    collector.finish(paths, level, sst_id, label)
}

/// One property's harvested `value → [id, ...]` postings, in def order.
///
/// Once a property is declared with a ScalarV1-compatible type, the collector
/// harvests every actually encodable String/Bool runtime value. This is
/// deliberate even for a legacy label-scoped SST: the raw storage API can
/// contain rows that predate or disagree with the later schema declaration,
/// and an authoritative negative-answer index must cover those rows too.
/// Streaming harvester behind [`prepare_equality_property_sidecars`]; the
/// posting-list analogue of [`UniqueSidecarCollector`].
#[derive(Debug)]
pub(crate) struct EqualitySidecarCollector {
    properties: Vec<String>,
    sorter: crate::sst::external_pairs::ExternalPairSorter,
}

impl EqualitySidecarCollector {
    pub(crate) fn new(label_def: &LabelDef) -> Result<Self> {
        let properties: Vec<String> = label_def
            .properties
            .iter()
            .filter(|p| {
                p.indexed
                    && matches!(
                        p.data_type,
                        DataType::Utf8 | DataType::LargeUtf8 | DataType::Bool
                    )
            })
            .map(|p| p.name.clone())
            .collect();
        u32::try_from(properties.len())
            .map_err(|_| Error::invariant("equality property count exceeds u32"))?;
        Ok(Self {
            properties,
            sorter: crate::sst::external_pairs::ExternalPairSorter::from_env()?,
        })
    }

    pub(crate) fn observe(&mut self, id: [u8; 16], rec: &NodeWriteRecord) -> Result<()> {
        for (ordinal, name) in self.properties.iter().enumerate() {
            let value = rec.properties.get(name);
            // The declaration only decides whether this property has a
            // ScalarV1 sidecar. Coverage follows the stored value: both
            // supported runtime types must be harvested so a schema change,
            // heterogeneous label, or legacy mismatched row cannot become an
            // authoritative false miss.
            let compatible = matches!(
                value,
                Some(namidb_core::Value::Str(_) | namidb_core::Value::Bool(_))
            );
            if let Some(key) = compatible
                .then_some(value)
                .flatten()
                .and_then(crate::cache::encode_equality_property_value)
            {
                self.sorter.push(ordinal as u32, key.as_bytes(), id)?;
            }
        }
        Ok(())
    }

    /// Serialise one sidecar per indexed property, including an empty map.
    ///
    /// The descriptor is also a coverage marker: readers can only prove that
    /// a label-agnostic posting lookup is complete when every node SST
    /// advertises the property. Omitting an empty sidecar (for example on a
    /// tombstone-only SST) would force every later lookup back to a full graph
    /// scan even though that SST contributes no claimant.
    pub(crate) fn finish(
        self,
        paths: &NamespacePaths,
        level: u32,
        sst_id: &Uuid,
        label: &str,
    ) -> Result<EqualityPropertySidecars> {
        let mut sorted = self.sorter.finish()?;
        let mut builders: Vec<_> = self
            .properties
            .iter()
            .map(|_| crate::sst::paged_index::SortedSpooledPagedIndexBuilder::equality())
            .collect();
        let legacy_limit = legacy_property_index_max_bytes()?;
        let mut legacy: Vec<Option<LegacyMapSpool>> = self
            .properties
            .iter()
            .map(|_| legacy_limit.map(LegacyMapSpool::new).transpose())
            .collect::<Result<_>>()?;
        let mut distinct_counts = vec![0_u64; self.properties.len()];
        let mut posting: Option<PendingExternalPosting> = None;
        while let Some(pair) = sorted.next_pair()? {
            if pair.property as usize >= builders.len() {
                return Err(Error::invariant(
                    "equality-index scratch property ordinal is out of range",
                ));
            }
            if posting
                .as_ref()
                .is_some_and(|active| active.property != pair.property || active.key != pair.key)
            {
                finish_external_posting(
                    posting.take().expect("active posting checked above"),
                    &mut builders,
                    &mut distinct_counts,
                    &mut legacy,
                )?;
            }
            if posting.is_none() {
                posting = Some(PendingExternalPosting::new(pair.property, pair.key)?);
            }
            posting
                .as_mut()
                .expect("posting initialized above")
                .push(pair.id)?;
        }
        if let Some(posting) = posting {
            finish_external_posting(posting, &mut builders, &mut distinct_counts, &mut legacy)?;
        }

        let mut descriptors = Vec::new();
        let mut bodies = Vec::new();
        for (((name, builder), legacy), distinct_values) in self
            .properties
            .into_iter()
            .zip(builders)
            .zip(legacy)
            .zip(distinct_counts)
        {
            // One key too wide to page must not cost the property its index:
            // keep the legacy body and mark the accelerator unsupported, which
            // is the same contract the pre-paged writer offered.
            let declined = builder.declined();
            let upload = if declined {
                None
            } else {
                Some(builder.finish()?)
            };
            let paged_name = format!(
                "{}-{}-{}.eqidx_{}.pidx",
                uuid_path_id(sst_id),
                SstKind::Nodes.path_tag(),
                label,
                name,
            );
            let paged_path = paths.sst_object(level, &paged_name);
            let paged_relative = relative_sst_path(level, &paged_name);
            let paged_size = upload.as_ref().map_or(0, |upload| upload.size_bytes());
            if let Some(legacy) = legacy {
                let legacy_name = format!(
                    "{}-{}-{}.eqidx_{}.bin",
                    uuid_path_id(sst_id),
                    SstKind::Nodes.path_tag(),
                    label,
                    name,
                );
                let legacy_path = paths.sst_object(level, &legacy_name);
                let legacy_relative = relative_sst_path(level, &legacy_name);
                let (legacy_file, legacy_size) = legacy.finish()?;
                descriptors.push(crate::manifest::EqualityIndexDescriptor {
                    property: name,
                    path: legacy_relative,
                    size_bytes: legacy_size,
                    distinct_values,
                    key_encoding: crate::manifest::EqualityKeyEncoding::ScalarV1,
                    mixed_type_complete: true,
                    format: crate::manifest::PropertyIndexFormat::BincodeV0,
                    paged: upload
                        .as_ref()
                        .map(|_| crate::manifest::PagedPropertyIndexDescriptor {
                            path: paged_relative,
                            size_bytes: paged_size,
                        }),
                    paged_build_unsupported: declined,
                });
                bodies.push((
                    legacy_path,
                    SidecarPayload::Spooled(SpooledObject::from_file(legacy_file, legacy_size)),
                ));
                if let Some(upload) = upload {
                    bodies.push((paged_path, SidecarPayload::Paged(upload)));
                }
            } else {
                // The operator opted out of the legacy body, so a decline here
                // leaves no accelerator at all; readers fall back to the scan.
                let Some(upload) = upload else {
                    continue;
                };
                descriptors.push(crate::manifest::EqualityIndexDescriptor {
                    property: name,
                    path: paged_relative,
                    size_bytes: paged_size,
                    distinct_values,
                    key_encoding: crate::manifest::EqualityKeyEncoding::ScalarV1,
                    mixed_type_complete: true,
                    format: crate::manifest::PropertyIndexFormat::PagedV1,
                    paged: None,
                    paged_build_unsupported: false,
                });
                bodies.push((paged_path, SidecarPayload::Paged(upload)));
            }
        }
        Ok((descriptors, bodies))
    }
}

struct PendingExternalPosting {
    property: u32,
    key: Vec<u8>,
    file: std::fs::File,
    len: u64,
    checksum: crc32fast::Hasher,
}

impl PendingExternalPosting {
    fn new(property: u32, key: Vec<u8>) -> Result<Self> {
        Ok(Self {
            property,
            key,
            file: crate::sst::paged_index::create_spool_file()?,
            len: 0,
            checksum: crc32fast::Hasher::new(),
        })
    }

    fn push(&mut self, id: [u8; 16]) -> Result<()> {
        let next_len = self
            .len
            .checked_add(16)
            .ok_or_else(|| Error::invariant("equality posting length exceeds u64"))?;
        if next_len > u32::MAX as u64 {
            return Err(Error::precondition(
                "one equality posting exceeds the 4 GiB PagedV2 wire limit",
            ));
        }
        self.file.write_all(&id)?;
        self.checksum.update(&id);
        self.len = next_len;
        Ok(())
    }
}

fn finish_external_posting(
    mut posting: PendingExternalPosting,
    builders: &mut [crate::sst::paged_index::SortedSpooledPagedIndexBuilder],
    distinct_counts: &mut [u64],
    legacy: &mut [Option<LegacyMapSpool>],
) -> Result<()> {
    posting.file.sync_data()?;
    let ordinal = posting.property as usize;
    let checksum = posting.checksum.finalize();
    builders[ordinal].push_external_file(&posting.key, &mut posting.file, posting.len, checksum)?;
    if let Some(legacy) = &mut legacy[ordinal] {
        legacy.write_posting(&posting.key, &mut posting.file, posting.len)?;
    }
    distinct_counts[ordinal] = distinct_counts[ordinal]
        .checked_add(1)
        .ok_or_else(|| Error::invariant("equality distinct-value count exceeds u64"))?;
    Ok(())
}

fn prepare_edge_pending(
    paths: &NamespacePaths,
    edge_type: &str,
    direction: EdgeDirection,
    build: EdgeSstBuild,
) -> PendingSst {
    let EdgeSstBuild {
        id,
        body,
        stats,
        bloom,
        point_index,
    } = build;
    let level = SstLevel::L0;
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
    let body_path = paths.sst_object(level.as_u32(), &file_name);
    let relative_path = relative_sst_path(level.as_u32(), &file_name);
    let body_len = body.size_bytes();
    let point_file_name = format!(
        "{}-{}-{}.epidx",
        uuid_path_id(&id),
        direction.path_tag(),
        edge_type
    );
    let point_path = paths.sst_object(level.as_u32(), &point_file_name);
    let index_sidecars = point_index
        .map(|point_body| vec![(point_path, SidecarPayload::EdgePoint(point_body))])
        .unwrap_or_default();

    let (bloom_descriptor, bloom_path, bloom_body) = prepare_bloom_sidecar(
        paths,
        level.as_u32(),
        &id,
        direction.path_tag(),
        edge_type,
        bloom,
    );

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
        node_locator: None,
        per_label_property_stats: Vec::new(),
    };

    PendingSst {
        descriptor,
        body_path,
        body: SidecarPayload::Spooled(body.into_spooled_object()),
        bloom_path,
        bloom_body,
        index_sidecars,
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

async fn put_create_payload(store: &dyn ObjectStore, path: &Path, body: PutPayload) -> Result<()> {
    let opts = PutOptions::from(PutMode::Create);
    store
        .put_opts(path, body, opts)
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

/// Maximum number of independent SST/sidecar objects uploaded concurrently by
/// one flush. Combined with [`MULTIPART_MAX_CONCURRENCY`], this caps a flush at
/// 32 in-flight part requests instead of multiplying by every label/sidecar.
const OBJECT_UPLOAD_MAX_CONCURRENCY: usize = 4;

/// Type-erased immutable-object PUT used by flush and offline SST attach.
pub(crate) type ObjectUploadFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>;

/// Await immutable-object PUTs with bounded concurrency, without
/// short-circuiting on the first error.
///
/// Collecting every result is intentional: each active multipart future gets
/// to finish its own error cleanup, while queued futures are still driven to a
/// terminal result. The first error is returned only after the full set has
/// drained.
pub(crate) async fn await_all_object_uploads(uploads: Vec<ObjectUploadFuture>) -> Result<()> {
    let results: Vec<Result<()>> = futures::stream::iter(uploads)
        .buffer_unordered(OBJECT_UPLOAD_MAX_CONCURRENCY)
        .collect()
        .await;
    for result in results {
        result?;
    }
    Ok(())
}

/// Split a scatter/gather object body into valid S3 multipart parts without
/// coalescing its chunks. Every part except the last is exactly `part_size`;
/// `Bytes::slice` keeps the underlying allocations shared.
fn multipart_payloads(body: &PutPayload, part_size: usize) -> Vec<PutPayload> {
    debug_assert!(part_size > 0);
    let mut parts = Vec::new();
    let mut current = Vec::new();
    let mut current_len = 0usize;
    for chunk in body {
        let mut offset = 0usize;
        while offset < chunk.len() {
            let take = (part_size - current_len).min(chunk.len() - offset);
            current.push(chunk.slice(offset..offset + take));
            current_len += take;
            offset += take;
            if current_len == part_size {
                parts.push(std::mem::take(&mut current).into_iter().collect());
                current_len = 0;
            }
        }
    }
    if current_len > 0 {
        parts.push(current.into_iter().collect());
    }
    parts
}

/// Upload a scatter/gather `body` to `path`. For small bodies, falls back to
/// the single-PUT
/// `PutMode::Create` path so the CAS-style "no overwrite" semantics still
/// protect against a competing writer stomping on a UUIDv7 path. For
/// bodies at or past [`MULTIPART_THRESHOLD`] (SST bodies in the LDBC SNB SF1
/// range — 10–50 MiB), uploads fixed-size `MULTIPART_PART_SIZE` chunks with
/// at most `MULTIPART_MAX_CONCURRENCY` requests in flight. Any part or
/// completion error explicitly aborts the upload so S3/R2 do not retain
/// orphan parts after a recoverable failure. Dropping/cancelling this future
/// also schedules an abort through [`MultipartUploadGuard`].
///
/// Why the split: S3 / R2 multipart uploads do NOT honour the `If-None-Match`
/// header that backs `PutMode::Create`. SST paths embed a UUIDv7 per writer
/// (see [`crate::flush`] §"PUT helpers") so collisions are impossible in
/// practice; the small-PUT branch is kept for bloom side-cars and any
/// future small body, where the CAS protection is cheap to keep.
pub(crate) async fn put_payload(
    store: std::sync::Arc<dyn ObjectStore>,
    path: &Path,
    body: PutPayload,
) -> Result<()> {
    let body_len = body.content_length();
    if body_len < MULTIPART_THRESHOLD {
        return put_create_payload(store.as_ref(), path, body).await;
    }

    let upload = store
        .put_multipart(path)
        .await
        .map_err(Error::ObjectStore)?;
    let mut upload = MultipartUploadGuard::new(upload, path);
    let mut pending = FuturesUnordered::new();
    let mut part_error = None;

    for part in multipart_payloads(&body, MULTIPART_PART_SIZE) {
        pending.push(upload.put_part(part));
        if pending.len() < MULTIPART_MAX_CONCURRENCY {
            continue;
        }
        if let Some(Err(source)) = pending.next().await {
            part_error = Some(source);
            break;
        }
    }

    while let Some(result) = pending.next().await {
        if part_error.is_none() {
            if let Err(source) = result {
                part_error = Some(source);
            }
        }
    }
    if let Some(source) = part_error {
        upload.abort_after_error("part failure").await;
        return Err(Error::ObjectStore(source));
    }

    if let Err(source) = upload.complete().await {
        upload.abort_after_error("completion failure").await;
        return Err(Error::ObjectStore(source));
    }
    Ok(())
}

/// Upload one optional accelerator, preserving create-only PUTs for ordinary
/// in-memory bodies and streaming corpus-sized exact node/edge records from
/// their anonymous spool files.
pub(crate) async fn put_sidecar_payload(
    store: Arc<dyn ObjectStore>,
    path: &Path,
    body: SidecarPayload,
) -> Result<()> {
    match body {
        SidecarPayload::InMemory(body) => put_payload(store, path, body).await,
        SidecarPayload::Spooled(body) => put_spooled_object(store, path, body).await,
        SidecarPayload::NodeLocator(body) => put_node_locator_upload(store, path, body).await,
        SidecarPayload::NodePropertyPages(body) => {
            put_node_property_pages_upload(store, path, body).await
        }
        SidecarPayload::EdgePoint(body) => put_edge_point_upload(store, path, body).await,
        SidecarPayload::Paged(body) => put_spooled_paged_upload(store, path, body).await,
    }
}

async fn put_node_property_pages_upload(
    store: Arc<dyn ObjectStore>,
    path: &Path,
    body: crate::sst::nodes::property_pages::NodePropertyPageUpload,
) -> Result<()> {
    let body_len = body.size_bytes();
    let (files, exact_len) = body.into_files();
    debug_assert_eq!(body_len, exact_len);
    put_spooled_object(
        store,
        path,
        SpooledObject::from_files(Vec::new(), files, body_len),
    )
    .await
}

async fn put_edge_point_upload(
    store: Arc<dyn ObjectStore>,
    path: &Path,
    body: crate::sst::paged_index::EdgePointIndexUpload,
) -> Result<()> {
    let body_len = body.size_bytes();
    let (files, exact_len) = body.into_files();
    debug_assert_eq!(body_len, exact_len);
    put_spooled_object(
        store,
        path,
        SpooledObject::from_files(Vec::new(), files, body_len),
    )
    .await
}

async fn put_spooled_paged_upload(
    store: Arc<dyn ObjectStore>,
    path: &Path,
    body: crate::sst::paged_index::SpooledPagedIndexUpload,
) -> Result<()> {
    let body_len = body.size_bytes();
    let (files, exact_len) = body.into_files();
    debug_assert_eq!(body_len, exact_len);
    put_spooled_object(
        store,
        path,
        SpooledObject::from_files(Vec::new(), files, body_len),
    )
    .await
}

async fn put_node_locator_upload(
    store: Arc<dyn ObjectStore>,
    path: &Path,
    body: crate::sst::paged_index::NodeLocatorRecordUpload,
) -> Result<()> {
    let body_len = body.size_bytes();
    let (prefix, file, _file_bytes) = body.into_parts();
    put_spooled_object(
        store,
        path,
        SpooledObject::from_parts(prefix, file, body_len),
    )
    .await
}

/// Contiguous-body convenience wrapper used by ordinary SSTs and existing
/// sidecars. Large exact node/edge sidecars use [`put_sidecar_payload`] so
/// their corpus-sized value regions are never materialised in RAM.
pub(crate) async fn put_object(
    store: std::sync::Arc<dyn ObjectStore>,
    path: &Path,
    body: Bytes,
) -> Result<()> {
    put_payload(store, path, body.into()).await
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
    /// int8-quantized vector packed into one `FixedSizeBinary(4 + dim)`:
    /// 4-byte little-endian f32 scale, then `dim` int8 code bytes.
    Int8Vector {
        dim: u32,
        builder: FixedSizeBinaryBuilder,
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
            DataType::Int8Vector { dim } => {
                let builder = FixedSizeBinaryBuilder::with_capacity(capacity, 4 + *dim as i32);
                PropertyBuilder::Int8Vector { dim: *dim, builder }
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
            // Already-quantized int8 vector: pack scale + codes and store.
            (PropertyBuilder::Int8Vector { dim, builder }, Value::VecI8 { codes, scale }) => {
                if codes.len() != *dim as usize {
                    return Err(Error::invariant(format!(
                        "property '{}' int8 vector dim={} != declared {}",
                        def.name,
                        codes.len(),
                        dim
                    )));
                }
                builder
                    .append_value(pack_i8vec(*scale, codes))
                    .map_err(|e| {
                        Error::invariant(format!("int8 vector append for '{}': {e}", def.name))
                    })?;
                Ok(())
            }
            // Convenience: an f32 vector written to an int8 column is quantized
            // on the fly, so callers can hand the writer raw embeddings.
            (PropertyBuilder::Int8Vector { dim, builder }, Value::Vec(v)) => {
                if v.len() != *dim as usize {
                    return Err(Error::invariant(format!(
                        "property '{}' vector dim={} != declared int8 vector {}",
                        def.name,
                        v.len(),
                        dim
                    )));
                }
                let (codes, scale) = namidb_core::quantize::quantize_i8(v);
                builder
                    .append_value(pack_i8vec(scale, &codes))
                    .map_err(|e| {
                        Error::invariant(format!("int8 vector append for '{}': {e}", def.name))
                    })?;
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
            PropertyBuilder::Int8Vector { builder, .. } => builder.append_null(),
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
            PropertyBuilder::Int8Vector { .. } => "Int8Vector",
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
            PropertyBuilder::Int8Vector { builder, .. } => Arc::new(builder.finish()),
            PropertyBuilder::Json(b) => Arc::new(b.finish()),
        }
    }
}

/// Pack an int8-quantized vector into the `FixedSizeBinary(4 + dim)` layout:
/// 4-byte little-endian f32 scale, then the int8 codes as raw bytes. The read
/// path in `read.rs` reverses this exactly.
fn pack_i8vec(scale: f32, codes: &[i8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + codes.len());
    buf.extend_from_slice(&scale.to_le_bytes());
    buf.extend(codes.iter().map(|&c| c as u8));
    buf
}

// ───────────────────────── Offline SST builder facade (RFC-023) ─────────

/// Public, attach-ready SST builder for offline / out-of-band ingestion
/// (RFC-023). An import-box binary links `namidb-storage`, builds finished
/// SSTs from already-minted node ids, and hands them to
/// [`WriterSession::attach_ssts`](crate::WriterSession::attach_ssts).
///
/// This is a CHILD module of `flush`, so it reaches `flush`'s private
/// builders (`prepare_node_pending`, `build_edge_stream_rows`, …) and the
/// private `PendingSst` via `super::` with no visibility changes. The public
/// surface is deliberately narrow — `NodeInput` / `EdgeInput` / `BuiltSst`
/// plus two build fns — over the engine's already-public `Schema` / `Value`
/// vocabulary; no internal type (`MemOp`, `NodeRow`, `NodeSstFinish`, …)
/// crosses the boundary.
pub mod builder {
    use std::collections::BTreeMap;

    use bytes::Bytes;
    use object_store::path::Path;

    use namidb_core::{LabelDef, LabelDictionary, Schema, Value};

    use crate::error::{Error, Result};
    use crate::manifest::SstDescriptor;
    use crate::memtable::MemOp;
    use crate::paths::NamespacePaths;
    use crate::sst::edges::inverse::transpose_forward_to_inverse;
    use crate::sst::edges::EdgeDirection;

    use super::{
        build_edge_stream_rows, prepare_edge_pending, prepare_node_pending, union_indexed_props,
        EdgeRow, EdgeWriteRecord, NodeRow, NodeWriteRecord, PendingSst,
    };

    /// One node to ingest. `id` is the ALREADY-MINTED 16-byte NodeId — the
    /// deterministic UUIDv5 mint is the cloud seam's job
    /// (`namidb_cloud_shared::mint`); the builder never sees the natural key.
    /// `properties` is the decoded property map; the natural-key value lives
    /// as an ordinary entry (e.g. `{"id": Value::Str("…")}`) and is what the
    /// unique-property sidecar harvests when the label declares it `unique`.
    /// Set `tombstone` for a delete marker (`properties` then ignored).
    #[derive(Debug, Clone)]
    pub struct NodeInput {
        /// Already-minted 16-byte NodeId.
        pub id: [u8; 16],
        /// Decoded property map (the natural key is a normal entry).
        pub properties: BTreeMap<String, Value>,
        /// When true, emit a delete marker instead of an upsert.
        pub tombstone: bool,
    }

    /// One edge to ingest. `src` / `dst` are already-minted 16-byte NodeIds.
    /// `properties` is decoded; the facade routes declared edge-type
    /// properties into their positional streams and the rest into the
    /// overflow stream internally.
    #[derive(Debug, Clone)]
    pub struct EdgeInput {
        /// Already-minted source NodeId.
        pub src: [u8; 16],
        /// Already-minted destination NodeId.
        pub dst: [u8; 16],
        /// Decoded edge property map.
        pub properties: BTreeMap<String, Value>,
        /// When true, emit a delete marker instead of an upsert.
        pub tombstone: bool,
    }

    /// A finished, attach-ready SST: body + optional bloom + unique-property
    /// sidecars plus a fully-assembled manifest [`SstDescriptor`]. Opaque —
    /// the caller gets read-only views;
    /// [`WriterSession::attach_ssts`](crate::WriterSession::attach_ssts)
    /// consumes it via the in-crate `into_parts`.
    #[derive(Debug)]
    pub struct BuiltSst {
        inner: PendingSst,
        /// Label names this SST's nodes carry (empty for edge SSTs). The
        /// builder interned them into on-row `LabelId`s; `attach_ssts` re-interns
        /// the names into the namespace dictionary so those ids resolve.
        label_dict: LabelDictionary,
    }

    impl BuiltSst {
        /// Exact `(raw LabelId, name)` mapping baked into this SST's node
        /// rows. Empty for edge SSTs. `attach_ssts` validates every mapping
        /// before issuing object-store writes so independently-built node
        /// SSTs cannot silently disagree about an id.
        pub(crate) fn label_entries(&self) -> Vec<(u32, String)> {
            self.label_dict
                .iter()
                .map(|(id, name)| (id.get(), name.to_string()))
                .collect()
        }

        /// Highest LSN carried by this SST (the `next_lsn` floor after attach).
        #[must_use]
        pub fn max_lsn(&self) -> u64 {
            self.inner.descriptor.max_lsn
        }
        /// Body size in bytes.
        #[must_use]
        pub fn size_bytes(&self) -> u64 {
            self.inner.descriptor.size_bytes
        }
        /// Manifest-relative path this SST will occupy.
        #[must_use]
        pub fn relative_path(&self) -> &str {
            &self.inner.descriptor.path
        }
        /// Row count (nodes: rows; edges: edge count).
        #[must_use]
        pub fn row_count(&self) -> u64 {
            self.inner.descriptor.row_count
        }

        /// Hand the PUT bodies + descriptor to `attach_ssts`. Lives here
        /// because only a descendant of `flush` may read `PendingSst`'s
        /// private fields.
        #[allow(clippy::type_complexity)]
        pub(crate) fn into_parts(
            self,
        ) -> (
            Path,
            super::SidecarPayload,
            Option<(Path, Bytes)>,
            Vec<(Path, super::SidecarPayload)>,
            SstDescriptor,
        ) {
            let PendingSst {
                descriptor,
                body_path,
                body,
                bloom_path,
                bloom_body,
                index_sidecars,
            } = self.inner;
            let bloom = match (bloom_path, bloom_body) {
                (Some(p), Some(b)) => Some((p, b)),
                _ => None,
            };
            (body_path, body, bloom, index_sidecars, descriptor)
        }
    }

    /// Build ONE node SST (+ a unique-property sidecar per `unique` string
    /// key) for `label`. The facade OWNS sort + dedup: it sorts `rows` by id
    /// ascending and, on a duplicate id, keeps the LAST occurrence
    /// (silent-upsert), then assigns `lsn = sorted position` so the unique
    /// sidecar's last-write-wins (row order == lsn order) holds. This is the
    /// contract `NodeSstWriter` (which rejects `nid <= prev`) and the sidecar
    /// depend on — done here so the import binary cannot get it wrong.
    ///
    /// `paths` MUST be the SAME [`NamespacePaths`] the attaching session
    /// targets. Returns `None` when `rows` is empty.
    pub fn build_node_sst(
        paths: &NamespacePaths,
        schema: &Schema,
        label: &str,
        rows: Vec<NodeInput>,
    ) -> Result<Option<BuiltSst>> {
        if rows.is_empty() {
            return Ok(None);
        }
        let rows = dedup_keep_last_node(rows);
        // Every independent builder call must mint the same raw LabelId for
        // the same schema label. A singleton dictionary assigned id 0 to
        // every label, so attaching separately-built A and B SSTs made B's
        // rows resolve as A. Schema labels live in a BTreeMap: intern the
        // complete catalog in that stable order before selecting this row's
        // id, and carry the mapping with the opaque BuiltSst for attach-time
        // preflight.
        let mut node_dict = LabelDictionary::new();
        for name in schema.labels.keys() {
            node_dict.intern(name);
        }
        let lid = node_dict.id(label).ok_or_else(|| {
            Error::precondition(format!(
                "offline node SST label '{label}' is not declared in the supplied schema"
            ))
        })?;
        let lid = lid.get();
        let node_rows: Vec<NodeRow> = rows
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                let op = if n.tombstone {
                    MemOp::Tombstone
                } else {
                    MemOp::Upsert(
                        NodeWriteRecord {
                            properties: n.properties,
                            schema_version: 0,
                            labels: vec![lid],
                        }
                        .encode()?,
                    )
                };
                Ok(NodeRow {
                    id: n.id,
                    lsn: (i as u64) + 1,
                    op,
                })
            })
            .collect::<Result<_>>()?;

        // Build with the fixed empty-LabelDef layout (matching `flush()`), and
        // harvest equality sidecars from the schema's indexed properties.
        let column_label = LabelDef {
            name: String::new(),
            properties: Vec::new(),
        };
        let index_props = union_indexed_props(schema);
        let finish = super::build_node_sst(&column_label, &node_rows)?;
        let pending =
            prepare_node_pending(paths, &index_props, &node_rows, finish, schema, &node_dict)?;
        Ok(Some(BuiltSst {
            inner: pending,
            label_dict: node_dict,
        }))
    }

    /// Build BOTH the forward and inverse CSR SSTs for `edge_type`, owning the
    /// sort, keep-last dedup (by `(src, dst)`), and monotonic lsn assignment,
    /// mirroring `flush()`'s edge path. Returns `[]` when `rows` is empty,
    /// else `[forward, inverse]`.
    pub fn build_edge_ssts(
        paths: &NamespacePaths,
        schema: &Schema,
        edge_type: &str,
        rows: Vec<EdgeInput>,
    ) -> Result<Vec<BuiltSst>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let rows = dedup_keep_last_edge(rows);
        let edge_def = schema.edge_type(edge_type).cloned();
        let declared_property_names: Vec<String> = edge_def
            .as_ref()
            .map(|d| d.properties.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();

        let edge_rows: Vec<EdgeRow> = rows
            .into_iter()
            .enumerate()
            .map(|(i, e)| {
                let op = if e.tombstone {
                    MemOp::Tombstone
                } else {
                    MemOp::Upsert(
                        EdgeWriteRecord {
                            properties: e.properties,
                            schema_version: 0,
                        }
                        .encode()?,
                    )
                };
                Ok(EdgeRow {
                    src: e.src,
                    dst: e.dst,
                    lsn: (i as u64) + 1,
                    op,
                })
            })
            .collect::<Result<_>>()?;

        let forward_rows = build_edge_stream_rows(&edge_rows, &declared_property_names)?;
        let inverse_rows = transpose_forward_to_inverse(&forward_rows);
        let fwd = super::build_edge_sst(
            edge_type,
            edge_def.as_ref(),
            &forward_rows,
            EdgeDirection::Forward,
        )?;
        let inv = super::build_edge_sst(
            edge_type,
            edge_def.as_ref(),
            &inverse_rows,
            EdgeDirection::Inverse,
        )?;
        Ok(vec![
            BuiltSst {
                inner: prepare_edge_pending(paths, edge_type, EdgeDirection::Forward, fwd),
                label_dict: LabelDictionary::new(),
            },
            BuiltSst {
                inner: prepare_edge_pending(paths, edge_type, EdgeDirection::Inverse, inv),
                label_dict: LabelDictionary::new(),
            },
        ])
    }

    /// Sort by id ascending and keep the LAST input per id (silent-upsert).
    /// `Vec::dedup_by` keeps the first, so fold explicitly.
    fn dedup_keep_last_node(mut rows: Vec<NodeInput>) -> Vec<NodeInput> {
        rows.sort_by_key(|n| n.id);
        let mut out: Vec<NodeInput> = Vec::with_capacity(rows.len());
        for n in rows {
            match out.last_mut() {
                Some(prev) if prev.id == n.id => *prev = n,
                _ => out.push(n),
            }
        }
        out
    }

    /// Sort by `(src, dst)` ascending and keep the LAST input per pair.
    fn dedup_keep_last_edge(mut rows: Vec<EdgeInput>) -> Vec<EdgeInput> {
        rows.sort_by_key(|e| (e.src, e.dst));
        let mut out: Vec<EdgeInput> = Vec::with_capacity(rows.len());
        for e in rows {
            match out.last_mut() {
                Some(prev) if prev.src == e.src && prev.dst == e.dst => *prev = e,
                _ => out.push(e),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

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

    #[derive(Debug)]
    struct AbortTrackingMultipart {
        part_started: tokio::sync::mpsc::UnboundedSender<()>,
        aborted: tokio::sync::mpsc::UnboundedSender<()>,
    }

    #[async_trait::async_trait]
    impl object_store::MultipartUpload for AbortTrackingMultipart {
        fn put_part(&mut self, _data: PutPayload) -> object_store::UploadPart {
            let _ = self.part_started.send(());
            Box::pin(std::future::pending::<object_store::Result<()>>())
        }

        async fn complete(&mut self) -> object_store::Result<object_store::PutResult> {
            Ok(object_store::PutResult {
                e_tag: None,
                version: None,
            })
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            let _ = self.aborted.send(());
            Ok(())
        }
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_waiter_keeps_build_gate_until_blocking_task_exits() {
        let (first_started_tx, first_started_rx) = tokio::sync::oneshot::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();
        let first = tokio::spawn(run_flush_build(move || {
            let _ = first_started_tx.send(());
            release_first_rx
                .recv()
                .map_err(|error| Error::invariant(format!("test release channel: {error}")))?;
            Ok(())
        }));
        tokio::time::timeout(Duration::from_secs(5), first_started_rx)
            .await
            .expect("first blocking build must start")
            .expect("first start signal must survive");

        first.abort();
        assert!(
            first
                .await
                .expect_err("aborted waiter must be cancelled")
                .is_cancelled(),
            "the async waiter, not the blocking task, should be cancelled"
        );

        let (second_started_tx, mut second_started_rx) = tokio::sync::oneshot::channel();
        let second = tokio::spawn(run_flush_build(move || {
            let _ = second_started_tx.send(());
            Ok(())
        }));
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut second_started_rx)
                .await
                .is_err(),
            "a retry must not start while the detached blocking build still owns the gate"
        );

        release_first_tx
            .send(())
            .expect("detached blocking build must still be waiting");
        tokio::time::timeout(Duration::from_secs(5), second_started_rx)
            .await
            .expect("retry must start after the first build exits")
            .expect("second start signal must survive");
        tokio::time::timeout(Duration::from_secs(5), second)
            .await
            .expect("second build must finish")
            .expect("second waiter must join")
            .expect("second build must succeed");
    }

    #[tokio::test]
    async fn put_object_round_trips_single_and_multipart_bodies() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        for (name, body) in [
            ("single.bin", Bytes::from_static(b"small immutable body")),
            (
                "multipart.bin",
                Bytes::from(vec![0x5a; MULTIPART_THRESHOLD + 17]),
            ),
        ] {
            let path = Path::from(name);
            put_object(store.clone(), &path, body.clone())
                .await
                .unwrap();
            let stored = store.get(&path).await.unwrap().bytes().await.unwrap();
            assert_eq!(stored, body, "{name} must survive its upload path");
        }
    }

    #[tokio::test]
    async fn multipart_guard_aborts_when_the_owner_task_is_cancelled() {
        let (part_started_tx, mut part_started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (aborted_tx, mut aborted_rx) = tokio::sync::mpsc::unbounded_channel();
        let path = Path::from("cancelled-multipart.bin");
        let task = tokio::spawn(async move {
            let mut upload = MultipartUploadGuard::new(
                Box::new(AbortTrackingMultipart {
                    part_started: part_started_tx,
                    aborted: aborted_tx,
                }),
                &path,
            );
            upload
                .put_part(PutPayload::from(Bytes::from_static(b"pending part")))
                .await
                .unwrap();
        });

        tokio::time::timeout(std::time::Duration::from_secs(1), part_started_rx.recv())
            .await
            .expect("multipart part did not start")
            .expect("multipart test channel closed");
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        tokio::time::timeout(std::time::Duration::from_secs(1), aborted_rx.recv())
            .await
            .expect("cancellation did not schedule multipart abort")
            .expect("multipart abort channel closed");
    }

    #[tokio::test]
    async fn object_upload_concurrency_is_bounded_and_errors_drain_all_siblings() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let total = OBJECT_UPLOAD_MAX_CONCURRENCY * 3;
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();

        let uploads: Vec<ObjectUploadFuture> = (0..total)
            .map(|index| {
                let active = active.clone();
                let maximum = maximum.clone();
                let completed = completed.clone();
                let gate = gate.clone();
                let started_tx = started_tx.clone();
                Box::pin(async move {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(now, Ordering::SeqCst);
                    let _ = started_tx.send(());
                    let _permit = gate.acquire().await.expect("test semaphore closed");
                    active.fetch_sub(1, Ordering::SeqCst);
                    completed.fetch_add(1, Ordering::SeqCst);
                    if index == 0 {
                        Err(Error::invariant("injected object upload failure"))
                    } else {
                        Ok(())
                    }
                }) as ObjectUploadFuture
            })
            .collect();
        drop(started_tx);

        let uploads_task = tokio::spawn(await_all_object_uploads(uploads));
        for _ in 0..OBJECT_UPLOAD_MAX_CONCURRENCY {
            tokio::time::timeout(std::time::Duration::from_secs(1), started_rx.recv())
                .await
                .expect("bounded object upload did not start")
                .expect("object upload start channel closed");
        }
        tokio::task::yield_now().await;
        assert_eq!(active.load(Ordering::SeqCst), OBJECT_UPLOAD_MAX_CONCURRENCY);
        assert_eq!(
            maximum.load(Ordering::SeqCst),
            OBJECT_UPLOAD_MAX_CONCURRENCY
        );
        assert!(
            started_rx.try_recv().is_err(),
            "queued object uploads started above the concurrency cap"
        );

        gate.add_permits(total);
        assert!(
            uploads_task.await.unwrap().is_err(),
            "the injected object failure must be returned"
        );
        assert_eq!(
            completed.load(Ordering::SeqCst),
            total,
            "all siblings must finish before the first error is returned"
        );
    }

    #[tokio::test]
    async fn spooled_node_locator_upload_round_trips_across_multipart_boundaries() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("nodes-spooled.nloc2");
        let mut builder = crate::sst::paged_index::NodeLocatorRecordBuilder::new();
        let mut expected = None;
        for n in 0..7_u128 {
            let node_id = n.to_be_bytes();
            let mut record = vec![(n as u8).wrapping_mul(31); 900 * 1024 + n as usize];
            record[..16].copy_from_slice(&node_id);
            if n == 6 {
                expected = Some(record.clone());
            }
            builder.push(&node_id, &record).unwrap();
        }
        let upload = builder.finish_upload().unwrap();
        assert!(upload.size_bytes() > MULTIPART_THRESHOLD as u64);
        put_sidecar_payload(store.clone(), &path, SidecarPayload::NodeLocator(upload))
            .await
            .unwrap();

        let (records, _) =
            crate::sst::paged_index::probe_node_records(store, path, &[6_u128.to_be_bytes()])
                .await
                .unwrap();
        assert_eq!(records.get(&6_u128.to_be_bytes()), expected.as_ref());
    }

    #[tokio::test]
    async fn spooled_edge_point_upload_round_trips_across_multipart_boundaries() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edges-spooled.epidx");
        let mut builder = crate::sst::paged_index::EdgePointIndexBuilder::new();
        let resident_bound = builder.resident_bound_bytes();
        let mut expected = None;
        for n in 0..7_u128 {
            let src = (n * 2).to_be_bytes();
            let dst = (n * 2 + 1).to_be_bytes();
            let value: Vec<u8> = (0..(900 * 1024 + n as usize))
                .map(|offset| (offset as u8).wrapping_mul(31).wrapping_add(n as u8))
                .collect();
            if n == 6 {
                expected = Some(value.clone());
            }
            builder.push(&src, &dst, &value).unwrap();
        }
        let upload = builder.finish_upload().unwrap();
        assert!(upload.size_bytes() > MULTIPART_THRESHOLD as u64);
        assert!(resident_bound < 32 * 1024);
        assert_eq!(upload.spooled_page_bytes(), (64 + 4096) as u64);
        assert!(upload.spooled_value_bytes() > MULTIPART_THRESHOLD as u64);
        put_sidecar_payload(store.clone(), &path, SidecarPayload::EdgePoint(upload))
            .await
            .unwrap();

        let probes = [((12_u128).to_be_bytes(), (13_u128).to_be_bytes())];
        let (found, _) = crate::sst::paged_index::probe_edge_points(store, path, &probes)
            .await
            .unwrap();
        assert_eq!(found.get(&probes[0]), expected.as_ref());
    }

    #[test]
    fn build_node_sst_chunked_batches_round_trip() {
        // Enough rows to force several 16k-row write_batch chunks; the body
        // must read back complete, ordered, and value-faithful — the writer
        // enforces ascending ids ACROSS batches, so this also guards the
        // chunk-boundary ordering contract.
        let label = person_label();
        let n: usize = 40_000;
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut id = [0u8; 16];
            id[8..16].copy_from_slice(&(i as u64).to_be_bytes());
            let mut props = std::collections::BTreeMap::new();
            props.insert("name".to_string(), Value::Str(format!("p{i}")));
            rows.push(NodeRow {
                id,
                lsn: i as u64 + 1,
                op: MemOp::Upsert(
                    NodeWriteRecord {
                        properties: props,
                        schema_version: 1,
                        labels: vec![],
                    }
                    .encode()
                    .unwrap(),
                ),
            });
        }
        let finish = build_node_sst(&label, &rows).unwrap();
        assert_eq!(finish.stats.row_count, n as u64);
        assert_eq!(finish.stats.max_lsn, n as u64);

        let reader = NodeSstReader::open(label, finish.body.clone()).unwrap();
        let batches = reader.scan().unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, n, "all chunked rows must be readable");
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
            ..Default::default()
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
            ..Default::default()
        };
        let bytes = r.encode().unwrap();
        let back = NodeWriteRecord::decode(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn exact_node_record_round_trips_vector_and_tombstone() {
        let record = NodeWriteRecord {
            properties: BTreeMap::from([
                ("key".into(), Value::Str("articulo-1".into())),
                (
                    "embedding".into(),
                    Value::Vec((0..1024).map(|i| (i % 17) as f32 / 17.0).collect()),
                ),
            ]),
            schema_version: 7,
            labels: vec![3],
        };
        let payload = record.encode().unwrap();
        let row = NodeRow {
            id: *sorted_node_id(1).as_bytes(),
            lsn: 99,
            op: MemOp::Upsert(payload.clone()),
        };
        let encoded = encode_exact_node_record(&row).unwrap();
        assert!(
            encoded.len() < payload.len(),
            "a 1024d JSON vector should use the per-record zstd representation"
        );
        let (lsn, op) = decode_exact_node_record(&encoded).unwrap();
        assert_eq!(lsn, 99);
        assert_eq!(op, MemOp::Upsert(payload));

        let tombstone = NodeRow {
            id: *sorted_node_id(2).as_bytes(),
            lsn: 100,
            op: MemOp::Tombstone,
        };
        let encoded = encode_exact_node_record(&tombstone).unwrap();
        assert_eq!(
            decode_exact_node_record(&encoded).unwrap(),
            (100, MemOp::Tombstone)
        );
    }

    #[test]
    fn exact_node_record_rejects_corrupt_lengths_and_encoding() {
        assert!(decode_exact_node_record(b"short").is_err());

        let mut invalid = vec![NODE_RECORD_RAW];
        invalid.extend_from_slice(&1_u64.to_le_bytes());
        invalid.extend_from_slice(&100_u32.to_le_bytes());
        invalid.extend_from_slice(b"tiny");
        assert!(decode_exact_node_record(&invalid).is_err());

        invalid[0] = u8::MAX;
        assert!(decode_exact_node_record(&invalid).is_err());
    }

    #[test]
    fn multipart_payload_split_preserves_order_without_coalescing() {
        let body: PutPayload = [
            Bytes::from_static(b"abc"),
            Bytes::from_static(b"defgh"),
            Bytes::from_static(b"ijkl"),
        ]
        .into_iter()
        .collect();
        let parts = multipart_payloads(&body, 5);
        assert_eq!(parts.len(), 3);
        assert_eq!(
            parts
                .iter()
                .map(PutPayload::content_length)
                .collect::<Vec<_>>(),
            vec![5, 5, 2]
        );
        let reconstructed: Vec<u8> = parts
            .iter()
            .flat_map(|part| part.iter().flat_map(|chunk| chunk.iter().copied()))
            .collect();
        assert_eq!(reconstructed, b"abcdefghijkl");
    }

    #[test]
    fn label_scoped_equality_sidecar_covers_supported_runtime_type_mismatches() {
        // Legacy per-label SSTs can contain rows written before a later schema
        // declaration (and the raw storage API is intentionally schemaless).
        // Once advertised as complete, their posting sidecar must therefore
        // cover the stored String/Bool value, not merely the declared type.
        for (suffix, declared, runtime, encoded) in [
            (
                "string-decl-bool-row",
                DataType::Utf8,
                Value::Bool(true),
                "b:1",
            ),
            (
                "bool-decl-string-row",
                DataType::Bool,
                Value::Str("runtime-string".into()),
                "runtime-string",
            ),
        ] {
            let label = LabelDef {
                name: "Legacy".into(),
                properties: vec![PropertyDef::new("key", declared.clone(), false)
                    .unwrap()
                    .with_indexed(true)],
            };
            let id = *sorted_node_id(1).as_bytes();
            let record = NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), runtime)]),
                schema_version: 1,
                ..Default::default()
            };
            let mut collector = EqualitySidecarCollector::new(&label).unwrap();
            collector.observe(id, &record).unwrap();
            let paths = make_paths(suffix);
            let (descriptors, bodies) = collector
                .finish(&paths, 0, &Uuid::now_v7(), "Legacy")
                .unwrap();

            assert_eq!(descriptors.len(), 1);
            assert!(
                descriptors[0].mixed_type_complete,
                "only a sidecar with full runtime-type coverage may advertise completeness"
            );
            // The legacy body stays authoritative so a key too wide to page can
            // still fall back to it; the paged sidecar rides along as the
            // range-readable mirror that point probes actually use.
            assert_eq!(
                descriptors[0].format,
                crate::manifest::PropertyIndexFormat::BincodeV0
            );
            assert!(descriptors[0].paged.is_some());
            assert!(!descriptors[0].paged_build_unsupported);
            let body = bodies
                .into_iter()
                .find_map(|(_, payload)| match payload {
                    SidecarPayload::Paged(upload) => Some(upload.into_bytes().unwrap()),
                    _ => None,
                })
                .expect("the paged equality mirror must be published");
            let postings = crate::sst::paged_index::decode_all_equality(&body).unwrap();
            assert_eq!(
                postings.get(encoded),
                Some(&vec![id]),
                "{declared:?} declaration must not omit its {encoded:?} runtime claimant"
            );
            assert!(
                !postings.contains_key("absent"),
                "an absent key remains an authoritative miss"
            );
        }
    }

    #[test]
    fn bounded_legacy_spool_is_bincode_identical_and_fails_above_cap() {
        use std::io::Read as _;

        let unique = BTreeMap::from([
            ("a".to_string(), 1_u128.to_be_bytes()),
            ("z".to_string(), 2_u128.to_be_bytes()),
        ]);
        let mut writer = LegacyMapSpool::new(1024).unwrap();
        for (key, id) in &unique {
            writer.write_unique(key.as_bytes(), id).unwrap();
        }
        let (mut file, _) = writer.finish().unwrap();
        let mut actual = Vec::new();
        file.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, bincode::serialize(&unique).unwrap());

        let equality = BTreeMap::from([(
            "b:1".to_string(),
            vec![1_u128.to_be_bytes(), 2_u128.to_be_bytes()],
        )]);
        let mut posting = crate::sst::paged_index::create_spool_file().unwrap();
        for id in &equality["b:1"] {
            posting.write_all(id).unwrap();
        }
        let mut writer = LegacyMapSpool::new(1024).unwrap();
        writer.write_posting(b"b:1", &mut posting, 32).unwrap();
        let (mut file, _) = writer.finish().unwrap();
        let mut actual = Vec::new();
        file.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, bincode::serialize(&equality).unwrap());

        let mut capped = LegacyMapSpool::new(8).unwrap();
        assert!(matches!(
            capped.write_unique(b"x", &1_u128.to_be_bytes()),
            Err(Error::Precondition(_))
        ));
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

    #[test]
    fn int8_vector_property_round_trips_through_arrow() {
        use namidb_core::value::Value;
        let dt = DataType::Int8Vector { dim: 4 };
        let def = PropertyDef::new("emb", dt.clone(), true).unwrap();
        let mut b = PropertyBuilder::new(&dt, 3).unwrap();

        // Row 0: an already-quantized vector (exact round-trip).
        let v0 = Value::VecI8 {
            codes: vec![127, -127, 0, 64],
            scale: 0.01,
        };
        b.append(Some(&v0), &def).unwrap();
        // Row 1: null.
        b.append(None, &def).unwrap();
        // Row 2: an f32 vector, quantized on the fly by the writer.
        let f = vec![0.5f32, -0.25, 0.0, 0.1];
        b.append(Some(&Value::Vec(f.clone())), &def).unwrap();

        let arr = b.finish();

        assert_eq!(
            crate::read::arrow_value_to_value(arr.as_ref(), 0, &dt).unwrap(),
            Some(v0)
        );
        assert_eq!(
            crate::read::arrow_value_to_value(arr.as_ref(), 1, &dt).unwrap(),
            None
        );
        match crate::read::arrow_value_to_value(arr.as_ref(), 2, &dt).unwrap() {
            Some(Value::VecI8 { codes, scale }) => {
                let back: Vec<f32> = codes.iter().map(|&c| c as f32 * scale).collect();
                for (x, y) in f.iter().zip(&back) {
                    assert!((x - y).abs() <= 0.5 * scale + 1e-6, "{x} vs {y}");
                }
            }
            other => panic!("expected VecI8, got {other:?}"),
        }
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
            MemKey::Node { id: alice },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        mt.apply(
            MemKey::Node { id: bob },
            11,
            MemOp::Upsert(node_payload("Bob", None)),
        );
        mt.apply(MemKey::Node { id: carol }, 12, MemOp::Tombstone);
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
        let property_descriptor = node_d
            .node_locator
            .as_ref()
            .and_then(|locator| locator.property_pages.as_ref())
            .expect("current node flush emits ranged property pages");
        assert!(property_descriptor.is_bound_to(node_d));
        let property_path = object_store::path::Path::from(format!(
            "{}/{}",
            ms.paths().namespace_prefix().as_ref(),
            property_descriptor.path
        ));
        let property_meta = store.head(&property_path).await.unwrap();
        assert_eq!(property_meta.size, property_descriptor.size_bytes);
        let property_source = Arc::new(
            crate::range_cache::PinnedObjectRangeSource::from_create_only_meta(
                store.clone(),
                property_meta,
            )
            .await
            .unwrap(),
        );
        let property_reader = crate::sst::nodes::property_pages::NodePropertyPageReader::open(
            property_source,
            node_d.id,
            crate::sst::nodes::property_pages::NodePropertyPageConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            property_reader.content_xxh3(),
            property_descriptor.content_xxh3
        );
        let (projected, _) = property_reader
            .project_node_ids(&["name".into()], &[*alice.as_bytes(), *bob.as_bytes()])
            .await
            .unwrap();
        assert_eq!(
            projected[0].properties["name"],
            crate::sst::nodes::property_pages::PropertyCell::Value(Value::Str("Alice".into()))
        );
        let abs = ms
            .paths()
            .sst_object(node_d.level.as_u32(), file_basename(&node_d.path));
        let body = store.get(&abs).await.unwrap().bytes().await.unwrap();
        // Id-primary node SSTs use the fixed layout (an empty `LabelDef`, no
        // `prop_*` columns; every property rides in `__overflow_json`), so the
        // reader must be opened with that same empty layout — mirroring the
        // production read path's `label_def_for_node_sst` for an empty scope.
        let reader = NodeSstReader::open(
            LabelDef {
                name: String::new(),
                properties: Vec::new(),
            },
            body,
        )
        .unwrap();
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
            xxh3: None,
        });
        let with_wal = ms.commit(&fence, &base, step1).await.unwrap();
        assert_eq!(with_wal.manifest.wal_segments.len(), 1);

        // Now flush: even with an empty memtable we'd skip the work; build
        // something tiny so flush goes through and confirms the WAL list is
        // cleared.
        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: alice },
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
            MemKey::Node { id: alice },
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
