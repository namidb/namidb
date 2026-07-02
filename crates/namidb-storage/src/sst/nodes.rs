//! Node SST writer and reader (Parquet body).
//!
//! Defined by [RFC-002](../../../../docs/rfc/002-sst-format.md) §2.
//!
//! The writer takes a stream of `RecordBatch`es matching the canonical
//! node-SST schema and emits a Parquet body in memory; the caller is
//! responsible for PUTting the body to object_store and for committing the
//! resulting `NodeSstStats` to a new manifest version.
//!
//! ## Schema
//!
//! For a `LabelDef` with declared properties `p_1: T_1, …, p_k: T_k`:
//!
//! | Column | Arrow type | Nullable |
//! |---------------------|------------------------|----------|
//! | `node_id` | `FixedSizeBinary(16)` | no |
//! | `tombstone` | `Boolean` | no |
//! | `lsn` | `UInt64` | no |
//! | `prop_<p_i>` | `<T_i>` | per def |
//! | `__overflow_json` | `Utf8` | yes |
//! | `__schema_version` | `UInt64` | no |
//!
//! The writer enforces:
//!
//! - Row order: `node_id` strictly ascending.
//! - All non-nullable columns have non-null values.
//! - The batch's schema is exactly the canonical schema built from `LabelDef`.

use std::collections::BTreeMap;
use std::sync::Arc;

use std::ops::Range;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampMicrosecondArray, UInt64Array,
};
use arrow_schema::{DataType as ArrowDataType, Field, Schema as ArrowSchema, SchemaRef};
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::{FutureExt, TryStreamExt};
use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, GetRange, ObjectStore};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::{ArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::errors::ParquetError;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::file::statistics::Statistics as ParquetStatistics;

use namidb_core::{DataType, LabelDef, PropertyDef};

use crate::error::{Error, Result};
use crate::sst::bloom::{BloomFilter, DEFAULT_BITS_PER_KEY};
use crate::sst::hll::{hash_bytes, Hll, DEFAULT_PRECISION as HLL_PRECISION};
use crate::sst::predicates::{eval_row_group, RowGroupVerdict, ScanPredicate};
use crate::sst::stats::{HllSketchBytes, PropertyColumnStats, StatScalar};

/// Canonical column names.
pub const COL_NODE_ID: &str = "node_id";
pub const COL_TOMBSTONE: &str = "tombstone";
pub const COL_LSN: &str = "lsn";
/// Always-present column for properties the schema did not declare.
pub const OVERFLOW_JSON: &str = "__overflow_json";
/// Always-present column carrying the schema version this row was written under.
pub const SCHEMA_VERSION: &str = "__schema_version";
/// Always-present column carrying the node's label set as a `List<UInt32>` of
/// [`LabelId`](namidb_core::LabelId) values (multi-label nodes). Legacy
/// single-label SSTs predate this column; [`NodeSstReader::open`] tolerates its
/// absence and the read path then derives the label set from the SST scope.
pub const COL_LABELS: &str = "__labels";

/// Tuning knobs for `NodeSstWriter`.
#[derive(Debug, Clone)]
pub struct NodeSstWriterOptions {
    pub compression: Compression,
    pub row_group_target_rows: usize,
    pub data_page_size: usize,
    pub write_batch_size: usize,
    pub bits_per_key: u8,
    pub expected_keys: u64,
    pub schema_version: u64,
}

/// Row-group row target for node SST writers. Reads
/// `NAMIDB_NODE_SST_ROW_GROUP_ROWS`; falls back to 128 Ki rows. Small
/// values force multi-row-group SSTs — used by row-group-pruning tests
/// so they don't have to write 128 k+ rows per fixture.
pub fn node_sst_row_group_target_rows() -> usize {
    std::env::var("NAMIDB_NODE_SST_ROW_GROUP_ROWS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(128 * 1024)
}

impl Default for NodeSstWriterOptions {
    fn default() -> Self {
        Self {
            compression: Compression::ZSTD(ZstdLevel::try_new(6).unwrap()),
            row_group_target_rows: node_sst_row_group_target_rows(),
            data_page_size: 1024 * 1024,
            write_batch_size: 8192,
            bits_per_key: DEFAULT_BITS_PER_KEY,
            expected_keys: 0,
            schema_version: 0,
        }
    }
}

/// Build the canonical Arrow schema for a node label.
///
/// Column order: `node_id, tombstone, lsn, __labels, prop_*, __overflow_json,
/// __schema_version`. Production node SSTs are built with an empty `LabelDef`
/// (no `prop_*` columns; every property rides in `__overflow_json`), but the
/// schema stays parameterised so the writer/reader remain general.
pub fn node_arrow_schema(label: &LabelDef) -> SchemaRef {
    let mut fields = Vec::with_capacity(label.properties.len() + 6);
    fields.push(Field::new(
        COL_NODE_ID,
        ArrowDataType::FixedSizeBinary(16),
        false,
    ));
    fields.push(Field::new(COL_TOMBSTONE, ArrowDataType::Boolean, false));
    fields.push(Field::new(COL_LSN, ArrowDataType::UInt64, false));
    fields.push(Field::new(
        COL_LABELS,
        // `item` nullable matches what `ListBuilder<UInt32Builder>` emits, so
        // the built RecordBatch's column type equals this field exactly.
        ArrowDataType::List(Arc::new(Field::new("item", ArrowDataType::UInt32, true))),
        false,
    ));
    for p in &label.properties {
        fields.push(prop_field(p));
    }
    fields.push(Field::new(OVERFLOW_JSON, ArrowDataType::Utf8, true));
    fields.push(Field::new(SCHEMA_VERSION, ArrowDataType::UInt64, false));
    Arc::new(ArrowSchema::new(fields))
}

/// Arrow field name for a declared property (with the `prop_` prefix).
pub fn prop_column_name(p: &PropertyDef) -> String {
    format!("prop_{}", p.name)
}

/// The SST-level field for a property is **always nullable**, even if the
/// `PropertyDef` is declared `nullable = false`. Tombstone rows carry `null`
/// in every property column by definition (RFC-002 §2.4), so non-null
/// declarations are enforced at the ingest layer, not at the SST layer.
fn prop_field(p: &PropertyDef) -> Field {
    Field::new(prop_column_name(p), p.data_type.to_arrow(), true)
}

/// Result of finalising a [`NodeSstWriter`].
#[derive(Debug)]
pub struct NodeSstFinish {
    pub body: Bytes,
    pub stats: NodeSstStats,
    pub bloom: Option<BloomFilter>,
}

/// Per-SST stats embedded in the manifest's `SstDescriptor` (RFC-002 §2.5).
#[derive(Debug, Clone, PartialEq)]
pub struct NodeSstStats {
    pub row_count: u64,
    pub tombstone_count: u64,
    pub min_node_id: [u8; 16],
    pub max_node_id: [u8; 16],
    pub min_lsn: u64,
    pub max_lsn: u64,
    pub property_stats: Vec<PropertyColumnStats>,
    pub schema_version_min: u64,
    pub schema_version_max: u64,
}

/// In-memory streaming writer for a node SST.
pub struct NodeSstWriter {
    label: LabelDef,
    schema: SchemaRef,
    inner: ArrowWriter<Vec<u8>>,
    bloom: BloomFilter,
    // running stats
    row_count: u64,
    tombstone_count: u64,
    min_node_id: Option<[u8; 16]>,
    max_node_id: Option<[u8; 16]>,
    min_lsn: u64,
    max_lsn: u64,
    last_node_id: Option<[u8; 16]>,
    schema_version_min: u64,
    schema_version_max: u64,
    /// One HLL sketch per declared property column (`prop_<name>` → Hll).
    /// Populated during [`write_batch`] by hashing each non-null value;
    /// serialised into [`PropertyColumnStats::ndv_estimate`] at `finish()`.
    /// Vector / JSON columns are skipped (HLL on vector embeddings has
    /// no meaningful interpretation).
    property_hlls: BTreeMap<String, Hll>,
}

impl std::fmt::Debug for NodeSstWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeSstWriter")
            .field("label", &self.label.name)
            .field("row_count", &self.row_count)
            .finish()
    }
}

impl NodeSstWriter {
    pub fn new(label: LabelDef, options: NodeSstWriterOptions) -> Result<Self> {
        // Re-validate the property names against the v1 reserved namespace.
        for p in &label.properties {
            PropertyDef::new(&p.name, p.data_type.clone(), p.nullable).map_err(|e| {
                Error::Invariant(format!(
                    "label '{}' contains invalid property '{}': {e}",
                    label.name, p.name
                ))
            })?;
        }

        let schema = node_arrow_schema(&label);
        let writer_props = build_writer_properties(&options);
        let inner = ArrowWriter::try_new(Vec::new(), schema.clone(), Some(writer_props))
            .map_err(|e| Error::invariant(format!("parquet writer init: {e}")))?;

        let bloom =
            BloomFilter::with_capacity(options.expected_keys.max(1), options.bits_per_key.max(1));

        let mut property_hlls = BTreeMap::new();
        for p in &label.properties {
            if hll_supported_for_datatype(&p.data_type) {
                property_hlls.insert(prop_column_name(p), Hll::new(HLL_PRECISION));
            }
        }

        Ok(Self {
            label,
            schema,
            inner,
            bloom,
            row_count: 0,
            tombstone_count: 0,
            min_node_id: None,
            max_node_id: None,
            min_lsn: u64::MAX,
            max_lsn: 0,
            last_node_id: None,
            schema_version_min: u64::MAX,
            schema_version_max: 0,
            property_hlls,
        })
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Write one batch of rows. The batch's schema must equal `self.schema()`.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.schema() != self.schema {
            return Err(Error::invariant(format!(
                "node SST batch schema mismatch: expected {:?}, got {:?}",
                self.schema,
                batch.schema()
            )));
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let node_id_col = batch
            .column_by_name(COL_NODE_ID)
            .ok_or_else(|| Error::invariant("missing node_id column"))?
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .ok_or_else(|| Error::invariant("node_id column has wrong array type"))?;

        let tombstone_col = batch
            .column_by_name(COL_TOMBSTONE)
            .ok_or_else(|| Error::invariant("missing tombstone column"))?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| Error::invariant("tombstone column has wrong array type"))?;

        let lsn_col = batch
            .column_by_name(COL_LSN)
            .ok_or_else(|| Error::invariant("missing lsn column"))?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::invariant("lsn column has wrong array type"))?;

        let schema_version_col = batch
            .column_by_name(SCHEMA_VERSION)
            .ok_or_else(|| Error::invariant("missing __schema_version column"))?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::invariant("__schema_version column has wrong array type"))?;

        for i in 0..batch.num_rows() {
            if node_id_col.is_null(i) {
                return Err(Error::invariant("node_id may not be null"));
            }
            let nid: [u8; 16] = node_id_col
                .value(i)
                .try_into()
                .map_err(|_| Error::invariant("node_id row length != 16"))?;

            if let Some(prev) = self.last_node_id {
                if nid <= prev {
                    return Err(Error::invariant(format!(
                        "node SST rows must be sorted by node_id ascending; \
 observed {} after {} at row {}",
                        hex_short(&nid),
                        hex_short(&prev),
                        i
                    )));
                }
            }
            self.last_node_id = Some(nid);
            if self.min_node_id.is_none() {
                self.min_node_id = Some(nid);
            }
            self.max_node_id = Some(nid);
            self.bloom.insert(&nid);

            let is_tomb = tombstone_col.value(i);
            if is_tomb {
                self.tombstone_count += 1;
            }

            let lsn = lsn_col.value(i);
            self.min_lsn = self.min_lsn.min(lsn);
            self.max_lsn = self.max_lsn.max(lsn);

            let sv = schema_version_col.value(i);
            self.schema_version_min = self.schema_version_min.min(sv);
            self.schema_version_max = self.schema_version_max.max(sv);

            self.row_count += 1;
        }

        update_property_hlls(&self.label, &mut self.property_hlls, batch)?;

        self.inner
            .write(batch)
            .map_err(|e| Error::invariant(format!("parquet write: {e}")))?;
        Ok(())
    }

    /// Close the Parquet body and return everything the flush path needs
    /// to build a `SstDescriptor` and a bloom side-car.
    pub fn finish(mut self) -> Result<NodeSstFinish> {
        if self.row_count == 0 {
            // Empty-SST case: the flush path may decide to skip such files,
            // but the writer should not enforce that policy. Coerce
            // sentinel-MAX into 0 so stats are interpretable.
            self.min_lsn = 0;
            self.schema_version_min = 0;
        }

        let body = self
            .inner
            .into_inner()
            .map_err(|e| Error::invariant(format!("parquet close: {e}")))?;
        let body = Bytes::from(body);

        let property_stats = compute_property_stats(&self.label, &body, &self.property_hlls)?;

        let stats = NodeSstStats {
            row_count: self.row_count,
            tombstone_count: self.tombstone_count,
            min_node_id: self.min_node_id.unwrap_or([0u8; 16]),
            max_node_id: self.max_node_id.unwrap_or([0u8; 16]),
            min_lsn: self.min_lsn,
            max_lsn: self.max_lsn,
            property_stats,
            schema_version_min: self.schema_version_min,
            schema_version_max: self.schema_version_max,
        };

        let bloom = if body.len() as u64 >= crate::sst::bloom::BLOOM_OMIT_THRESHOLD_BYTES {
            Some(self.bloom)
        } else {
            None
        };

        Ok(NodeSstFinish { body, stats, bloom })
    }
}

/// Reader for a node SST. We accept the full body in memory because that is
/// the simplest path; ranged-GET / row-group skipping lands in
/// the read-path RFC.
#[derive(Debug)]
pub struct NodeSstReader {
    body: Bytes,
    schema: SchemaRef,
    label: LabelDef,
}

impl NodeSstReader {
    /// Open a reader from an in-memory Parquet body.
    pub fn open(label: LabelDef, body: Bytes) -> Result<Self> {
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            body.clone(),
            ArrowReaderOptions::new().with_page_index(true),
        )
        .map_err(|e| Error::invariant(format!("parquet open: {e}")))?;
        let schema = builder.schema().clone();

        // Sanity-check: every expected column is present.
        let expected = node_arrow_schema(&label);
        if schema.fields().len() != expected.fields().len() {
            return Err(Error::Corrupted {
                path: "<in-memory>".into(),
                detail: format!(
                    "node SST has {} columns, expected {}",
                    schema.fields().len(),
                    expected.fields().len()
                ),
            });
        }
        for (got, want) in schema.fields().iter().zip(expected.fields().iter()) {
            if got.name() != want.name() || got.data_type() != want.data_type() {
                return Err(Error::Corrupted {
                    path: "<in-memory>".into(),
                    detail: format!("node SST column mismatch: got {:?}, want {:?}", got, want),
                });
            }
        }
        Ok(Self {
            body,
            schema,
            label,
        })
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn label(&self) -> &LabelDef {
        &self.label
    }

    /// Read every row group sequentially. Cheap.
    /// Equivalent to `scan_with_predicates_and_projection(&[], None)`.
    pub fn scan(&self) -> Result<Vec<RecordBatch>> {
        self.scan_with_predicates_and_projection(&[], None)
    }

    /// Row-group-pruned scan (RFC-013). Equivalent to
    /// `scan_with_predicates_and_projection(predicates, None)`.
    pub fn scan_with_predicates(&self, predicates: &[ScanPredicate]) -> Result<Vec<RecordBatch>> {
        self.scan_with_predicates_and_projection(predicates, None)
    }

    /// Row-group-pruned + column-projected scan (RFC-015).
    /// Composes the row-group skipping with a Parquet
    /// `ProjectionMask` so the decoded `RecordBatch`es only contain
    /// the requested property columns (plus the engine columns that
    /// every reader needs: `node_id`, `tombstone`, `lsn`,
    /// `__schema_version`, `__overflow_json`).
    ///
    /// When `projection.is_none()` the reader behaves exactly like
    /// `scan_with_predicates`. When `projection.is_some(&[])` it
    /// reads the engine columns ONLY — useful for COUNT(*) / EXISTS
    /// shapes where no property is needed.
    pub fn scan_with_predicates_and_projection(
        &self,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
    ) -> Result<Vec<RecordBatch>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.body.clone())
            .map_err(|e| Error::invariant(format!("parquet open: {e}")))?;
        let md = builder.metadata().clone();
        let schema_descr = md.file_metadata().schema_descr();

        // 1) Row-group pruning (RFC-013).
        let keep = if predicates.is_empty() {
            (0..md.row_groups().len()).collect::<Vec<_>>()
        } else {
            // Resolve (column → leaf index + declared PropertyDef) for
            // every column any predicate references.
            let mut col_lookup: std::collections::BTreeMap<&str, Option<(usize, &PropertyDef)>> =
                std::collections::BTreeMap::new();
            for pred in predicates {
                let col = pred.column();
                if col_lookup.contains_key(col) {
                    continue;
                }
                let prop = self.label.properties.iter().find(|p| p.name == col);
                let resolved = prop.and_then(|p| {
                    let parquet_name = prop_column_name(p);
                    schema_descr
                        .columns()
                        .iter()
                        .position(|c| c.name() == parquet_name)
                        .map(|idx| (idx, p))
                });
                col_lookup.insert(col, resolved);
            }

            let mut keep: Vec<usize> = Vec::new();
            for (rg_idx, rg) in md.row_groups().iter().enumerate() {
                let mut absent = false;
                for pred in predicates {
                    let col = pred.column();
                    let resolved = col_lookup.get(col).and_then(|r| r.as_ref());
                    let synthetic = match resolved {
                        Some((leaf_idx, prop)) => {
                            let cc = rg.column(*leaf_idx);
                            synthesize_property_stats(col, &prop.data_type, cc.statistics())
                        }
                        None => PropertyColumnStats::empty(col),
                    };
                    if eval_row_group(pred, &synthetic) == RowGroupVerdict::Absent {
                        absent = true;
                        break;
                    }
                }
                if !absent {
                    keep.push(rg_idx);
                }
            }
            keep
        };

        if keep.is_empty() {
            return Ok(Vec::new());
        }

        // 2) Column projection (RFC-015). Build a leaf-index mask that
        // always includes the engine columns plus the property leafs the
        // caller requested. When `projection.is_none()` we skip the mask
        // and Parquet reads every leaf.
        let projection_mask = projection.map(|cols| {
            let mut leaves: Vec<usize> = Vec::with_capacity(cols.len() + 6);
            // Match on the leaf column's top-level path component, not its leaf
            // name: `__labels` is a `List<UInt32>` whose Parquet leaf is named
            // `element` (path `__labels.list.element`), so a `c.name()` match
            // would silently miss it — and eliding `__labels` makes
            // `decode_node_labels` fall back to the (now empty) scope and drop
            // every row at the label filter under the id-primary layout.
            let root_of = |c: &parquet::schema::types::ColumnDescriptor| -> Option<String> {
                c.path().parts().first().cloned()
            };
            for engine in [
                COL_NODE_ID,
                COL_TOMBSTONE,
                COL_LSN,
                SCHEMA_VERSION,
                OVERFLOW_JSON,
                COL_LABELS,
            ] {
                if let Some(idx) = schema_descr
                    .columns()
                    .iter()
                    .position(|c| root_of(c).as_deref() == Some(engine))
                {
                    leaves.push(idx);
                }
            }
            for col in cols {
                let prop = self.label.properties.iter().find(|p| p.name == *col);
                if let Some(p) = prop {
                    let parquet_name = prop_column_name(p);
                    if let Some(idx) = schema_descr
                        .columns()
                        .iter()
                        .position(|c| c.name() == parquet_name)
                    {
                        leaves.push(idx);
                    }
                }
            }
            ProjectionMask::leaves(schema_descr, leaves)
        });

        // Optimisation: full row-group set + no projection ⇒ avoid the
        // overhead of `with_row_groups` / `with_projection`.
        let all_row_groups = keep.len() == md.row_groups().len();
        let mut builder = builder;
        if !all_row_groups {
            builder = builder.with_row_groups(keep);
        }
        if let Some(mask) = projection_mask {
            builder = builder.with_projection(mask);
        }
        let reader = builder
            .build()
            .map_err(|e| Error::invariant(format!("parquet build: {e}")))?;
        let mut batches: Vec<RecordBatch> = Vec::new();
        for b in reader {
            batches.push(b.map_err(|e| Error::invariant(format!("parquet read: {e}")))?);
        }
        Ok(batches)
    }

    /// Decode only the row groups whose `node_id` column min/max stats
    /// straddle `target`. The writer enforces strict ascending `node_id`
    /// across the whole SST, so at most one row group matches — bringing
    /// the per-lookup decode cost from `O(rows_in_sst)` down to
    /// `O(rows_per_row_group)`.
    ///
    /// Falls back to a full scan for row groups that ship no stats
    /// (defensive — every writer path we control emits `EnabledStatistics::Chunk`).
    pub fn targeted_scan(&self, target: &[u8; 16]) -> Result<Vec<RecordBatch>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.body.clone())
            .map_err(|e| Error::invariant(format!("parquet open: {e}")))?;
        let md = builder.metadata().clone();

        let leaf_idx = md
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .position(|c| c.name() == COL_NODE_ID)
            .ok_or_else(|| Error::invariant("node_id column not in parquet schema"))?;

        let mut keep: Vec<usize> = Vec::new();
        for (rg_idx, rg) in md.row_groups().iter().enumerate() {
            let cc = rg.column(leaf_idx);
            let in_range = match cc.statistics() {
                Some(stats) => match (stats.min_bytes_opt(), stats.max_bytes_opt()) {
                    (Some(min), Some(max)) => target.as_slice() >= min && target.as_slice() <= max,
                    _ => true,
                },
                None => true,
            };
            if in_range {
                keep.push(rg_idx);
            }
        }

        if keep.is_empty() {
            return Ok(Vec::new());
        }

        let reader = builder
            .with_row_groups(keep)
            .build()
            .map_err(|e| Error::invariant(format!("parquet build: {e}")))?;
        let mut batches: Vec<RecordBatch> = Vec::new();
        for b in reader {
            batches.push(b.map_err(|e| Error::invariant(format!("parquet read: {e}")))?);
        }
        Ok(batches)
    }

    /// Decode exactly `row_groups` (ascending footer indices), in order.
    /// The caller has already done the pruning — typically via
    /// [`row_groups_for_keys`] against cached footer metadata.
    pub fn scan_row_groups(&self, row_groups: Vec<usize>) -> Result<Vec<RecordBatch>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.body.clone())
            .map_err(|e| Error::invariant(format!("parquet open: {e}")))?;
        let reader = builder
            .with_row_groups(row_groups)
            .build()
            .map_err(|e| Error::invariant(format!("parquet build: {e}")))?;
        let mut batches: Vec<RecordBatch> = Vec::new();
        for b in reader {
            batches.push(b.map_err(|e| Error::invariant(format!("parquet read: {e}")))?);
        }
        Ok(batches)
    }

    /// Decode each of `row_groups` into its own batch vector, reusing the
    /// caller's parsed footer `md` (no per-group footer re-parse).
    ///
    /// Unlike [`Self::scan_row_groups`] — where the sync arrow reader
    /// streams the selected groups contiguously and a batch can span two
    /// of them, sharing decode buffers across the boundary — this decodes
    /// one group per reader, so every returned vector owns right-sized
    /// buffers. That matters for the decoded row-group cache: entries
    /// must not pin each other's memory, and their byte weights must
    /// reflect what eviction actually frees.
    pub fn scan_row_groups_each(
        &self,
        md: &Arc<ParquetMetaData>,
        row_groups: &[usize],
    ) -> Result<Vec<(usize, Vec<RecordBatch>)>> {
        let meta = ArrowReaderMetadata::try_new(md.clone(), ArrowReaderOptions::new())
            .map_err(|e| Error::invariant(format!("parquet metadata reuse: {e}")))?;
        let mut out: Vec<(usize, Vec<RecordBatch>)> = Vec::with_capacity(row_groups.len());
        for &rg in row_groups {
            let reader =
                ParquetRecordBatchReaderBuilder::new_with_metadata(self.body.clone(), meta.clone())
                    .with_row_groups(vec![rg])
                    .build()
                    .map_err(|e| Error::invariant(format!("parquet build: {e}")))?;
            let mut batches: Vec<RecordBatch> = Vec::new();
            for b in reader {
                batches.push(b.map_err(|e| Error::invariant(format!("parquet read: {e}")))?);
            }
            out.push((rg, batches));
        }
        Ok(out)
    }
}

/// Row groups whose `node_id` min/max range can contain at least one of
/// `sorted_keys` (ascending). The writer keeps `node_id` strictly
/// ascending across the SST, so per-row-group stats partition the key
/// space and a sorted probe set resolves each row group with one
/// binary search. Row groups without stats are admitted (defensive —
/// every writer path we control emits `EnabledStatistics::Chunk`),
/// matching [`NodeSstReader::targeted_scan`].
pub fn row_groups_for_keys(md: &ParquetMetaData, sorted_keys: &[[u8; 16]]) -> Result<Vec<usize>> {
    let leaf_idx = md
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|c| c.name() == COL_NODE_ID)
        .ok_or_else(|| Error::invariant("node_id column not in parquet schema"))?;
    let mut keep: Vec<usize> = Vec::new();
    for (rg_idx, rg) in md.row_groups().iter().enumerate() {
        let cc = rg.column(leaf_idx);
        let in_range = match cc.statistics() {
            Some(stats) => match (stats.min_bytes_opt(), stats.max_bytes_opt()) {
                (Some(min), Some(max)) => {
                    let start = sorted_keys.partition_point(|k| k.as_slice() < min);
                    start < sorted_keys.len() && sorted_keys[start].as_slice() <= max
                }
                _ => true,
            },
            None => true,
        };
        if in_range {
            keep.push(rg_idx);
        }
    }
    Ok(keep)
}

/// Partition `batches` (decoded from `row_groups`, in order) back into
/// per-row-group vectors using the footer's per-row-group row counts.
/// The arrow readers never emit a batch spanning two row groups, but the
/// split slices defensively (zero-copy) rather than relying on that, and
/// errors out on any row-count mismatch so a partial decode can never be
/// cached as a complete row group.
pub fn split_batches_by_row_group(
    md: &ParquetMetaData,
    row_groups: &[usize],
    batches: Vec<RecordBatch>,
) -> Result<Vec<(usize, Vec<RecordBatch>)>> {
    let mut out: Vec<(usize, Vec<RecordBatch>)> =
        row_groups.iter().map(|&rg| (rg, Vec::new())).collect();
    let mut slot = 0usize;
    let mut remaining: usize = match out.first() {
        Some((rg, _)) => md.row_group(*rg).num_rows() as usize,
        None => 0,
    };
    for batch in batches {
        let mut rest = batch;
        while rest.num_rows() > 0 {
            while remaining == 0 {
                slot += 1;
                if slot >= out.len() {
                    return Err(Error::invariant(
                        "row-group split: decoded more rows than the footer declares",
                    ));
                }
                remaining = md.row_group(out[slot].0).num_rows() as usize;
            }
            let take = rest.num_rows().min(remaining);
            out[slot].1.push(rest.slice(0, take));
            remaining -= take;
            rest = rest.slice(take, rest.num_rows() - take);
        }
    }
    if !out.is_empty() && (slot != out.len() - 1 || remaining != 0) {
        return Err(Error::invariant(
            "row-group split: decoded fewer rows than the footer declares",
        ));
    }
    Ok(out)
}

/// Parse the footer + page index of an in-memory node SST body into the
/// same `Arc<ParquetMetaData>` shape the ranged path produces, so both
/// can share one entry in [`crate::cache::SstCache`]'s metadata map.
pub fn parse_node_sst_metadata(body: &Bytes) -> Result<Arc<ParquetMetaData>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
        body.clone(),
        ArrowReaderOptions::new().with_page_index(true),
    )
    .map_err(|e| Error::invariant(format!("parquet open: {e}")))?;
    Ok(builder.metadata().clone())
}

/// Fetch the footer + page index of a node SST over ranged GETs, without
/// pulling the body. Used by `Snapshot::batch_lookup_nodes` when ranged
/// reads are in effect and the body is not cached.
pub async fn load_node_sst_metadata_async(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    file_size: u64,
) -> Result<Arc<ParquetMetaData>> {
    let mut reader = ObjectStoreRangedReader {
        store,
        path: path.clone(),
        file_size,
        cached_metadata: None,
    };
    AsyncFileReader::get_metadata(&mut reader, None)
        .await
        .map_err(|e| Error::invariant(format!("parquet metadata async {path}: {e}")))
}

/// Thin `parquet::AsyncFileReader` wrapper around our
/// `Arc<dyn ObjectStore>`. Exists because `parquet 55`'s bundled
/// `ParquetObjectReader` is wired against `object_store 0.12`, while
/// our workspace pins `object_store 0.13`; the two `ObjectStore` trait
/// objects are not interchangeable. Implementing `AsyncFileReader`
/// directly against our store crate side-steps the version split
/// without adding a parallel object_store dependency.
///
/// Every `get_bytes` issues a single `GET` with an `If-Match` byte
/// range; that maps to a `Range` header on the wire — same path the
/// sync `Bytes`-backed reader would take if we cached entire bodies.
///
/// `cached_metadata` short-circuits the footer + page-index round-trip
/// when the caller has it from a previous warm lookup on the same SST
/// (RFC-003 §"Cache integration"). Bypasses `get_metadata` entirely.
struct ObjectStoreRangedReader {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    file_size: u64,
    cached_metadata: Option<Arc<ParquetMetaData>>,
}

impl AsyncFileReader for ObjectStoreRangedReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, std::result::Result<Bytes, ParquetError>> {
        let store = self.store.clone();
        let path = self.path.clone();
        async move {
            let opts = GetOptions {
                range: Some(GetRange::Bounded(range)),
                ..Default::default()
            };
            let resp = store
                .get_opts(&path, opts)
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))?;
            resp.bytes()
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))
        }
        .boxed()
    }

    /// Override the default sequential `get_byte_ranges` to use
    /// `object_store::get_ranges`, which coalesces nearby ranges into
    /// a single HTTP `Range:` request and issues the remaining ones
    /// in parallel. Without this override the parquet reader would
    /// fetch each column page back-to-back, paying one full RTT per
    /// GET — measured against R2 that turned a single cold lookup
    /// into 8–10 serial round-trips and inflated the wall time from
    /// ~500 ms (full body) to ~2 s (naive ranged). With coalescing
    /// the count drops to 1–3 round-trips of larger ranges.
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, std::result::Result<Vec<Bytes>, ParquetError>> {
        let store = self.store.clone();
        let path = self.path.clone();
        async move {
            store
                .get_ranges(&path, &ranges)
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, std::result::Result<Arc<ParquetMetaData>, ParquetError>> {
        // Cache hit: serve the metadata without round-tripping. RFC-003
        // §"Cache integration" — the warm ranged-read path skips the
        // footer + page-index fetch entirely.
        if let Some(meta) = self.cached_metadata.clone() {
            return async move { Ok(meta) }.boxed();
        }
        let file_size = self.file_size;
        async move {
            // Prefetch hint: ask the metadata reader to pull 256 KiB
            // off the tail of the file in one GET instead of doing a
            // narrow footer probe (~8 KiB) followed by another GET for
            // the page index. On R2 from a laptop, each round-trip is
            // ~150–250 ms; saving even one of them is worth more than
            // the extra ~250 KiB on the wire.
            let metadata = ParquetMetaDataReader::new()
                .with_column_indexes(true)
                .with_offset_indexes(true)
                .with_prefetch_hint(Some(256 * 1024))
                .load_and_finish(self, file_size)
                .await?;
            Ok(Arc::new(metadata))
        }
        .boxed()
    }
}

/// Ranged-read variant of [`NodeSstReader::targeted_scan`] (RFC-003).
///
/// Instead of consuming a `Bytes` body that the caller has already
/// fetched in full, this driver builds an `AsyncFileReader` over
/// `(store, path)` and lets the parquet stream builder issue
/// **byte-ranged** GETs against the object store:
///
/// 1. **Footer fetch.** The metadata reader reads the trailing
/// ~8 KB to parse the footer + column-chunk metadata. One round
/// trip, ~8 KB transferred.
/// 2. **Row-group pruning.** Same `(min_key, max_key)` per-row-group
/// filter the sync `targeted_scan` does — pick the single row
/// group that straddles `target`.
/// 3. **Page index + column reads.** With `with_page_index(true)`
/// the builder also fetches the offset + column index for the
/// chosen row group; subsequent `next()` calls on the stream
/// issue one ranged GET per column page that the row group
/// decoder needs. For a `Person`-shaped SST that's ~8 narrow
/// GETs of 1–8 KB each (typically coalesced).
///
/// Total wire footprint for a cold lookup: ~50–100 KB (vs the
/// full-body GET which is 7 MiB for 1 M nodes / ~70 MiB for 10 M).
/// On real-WAN S3 from a laptop this drops cold p50 by ~50× per the
/// R2 doc.
///
/// `file_size` is the SST body length, available from
/// `SstDescriptor.size_bytes` — passing it avoids a HEAD round trip.
/// `label` is the declared schema used to instantiate the right
/// column projection (same as the sync path).
pub async fn targeted_scan_async(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    file_size: u64,
    label: &LabelDef,
    target: &[u8; 16],
    cached_metadata: Option<Arc<ParquetMetaData>>,
) -> Result<(Vec<RecordBatch>, Arc<ParquetMetaData>)> {
    // Row-group prune by min/max stats on node_id. Same logic as
    // `targeted_scan`.
    let target = *target;
    ranged_scan_selected_row_groups(store, path, file_size, label, cached_metadata, move |md| {
        row_groups_for_keys(md, std::slice::from_ref(&target))
    })
    .await
}

/// Ranged-read decode of caller-selected row groups (no pruning of its
/// own — pair with [`row_groups_for_keys`] against cached metadata).
/// Used by `Snapshot::batch_lookup_nodes` so a multi-id probe over a
/// large SST fetches only the column pages of the row groups that can
/// contain a probe id.
pub async fn scan_row_groups_async(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    file_size: u64,
    label: &LabelDef,
    row_groups: Vec<usize>,
    cached_metadata: Option<Arc<ParquetMetaData>>,
) -> Result<Vec<RecordBatch>> {
    let (batches, _md) =
        ranged_scan_selected_row_groups(store, path, file_size, label, cached_metadata, move |_| {
            Ok(row_groups)
        })
        .await?;
    Ok(batches)
}

/// Shared driver for the ranged-read scans: open the stream builder over
/// byte-ranged GETs, sanity-check the schema, let `select` pick the row
/// groups from the footer metadata, and collect the decoded batches.
async fn ranged_scan_selected_row_groups(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    file_size: u64,
    label: &LabelDef,
    cached_metadata: Option<Arc<ParquetMetaData>>,
    select: impl FnOnce(&ParquetMetaData) -> Result<Vec<usize>>,
) -> Result<(Vec<RecordBatch>, Arc<ParquetMetaData>)> {
    let reader = ObjectStoreRangedReader {
        store,
        path: path.clone(),
        file_size,
        cached_metadata,
    };
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        reader,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await
    .map_err(|e| Error::invariant(format!("parquet open async {path}: {e}")))?;
    let md = builder.metadata().clone();

    // Same schema sanity check the sync `open()` does. The async reader
    // wouldn't necessarily catch a column rename until decode time.
    let expected = node_arrow_schema(label);
    let got = builder.schema();
    if got.fields().len() != expected.fields().len() {
        return Err(Error::Corrupted {
            path: path.to_string(),
            detail: format!(
                "node SST has {} columns, expected {}",
                got.fields().len(),
                expected.fields().len()
            ),
        });
    }

    let keep = select(&md)?;
    if keep.is_empty() {
        return Ok((Vec::new(), md));
    }

    let stream = builder
        .with_row_groups(keep)
        .build()
        .map_err(|e| Error::invariant(format!("parquet build async {path}: {e}")))?;

    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .map_err(|e| Error::invariant(format!("parquet stream collect async {path}: {e}")))?;
    Ok((batches, md))
}

fn build_writer_properties(opts: &NodeSstWriterOptions) -> WriterProperties {
    WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(opts.compression)
        .set_max_row_group_size(opts.row_group_target_rows)
        .set_data_page_size_limit(opts.data_page_size)
        .set_write_batch_size(opts.write_batch_size)
        .set_dictionary_enabled(true)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_encoding(Encoding::PLAIN)
        .build()
}

/// Aggregate Parquet column statistics from every row group into the
/// per-property [`PropertyColumnStats`] that the manifest carries.
///
/// `EnabledStatistics::Chunk` is configured in [`build_writer_properties`],
/// so the writer emits per-row-group `min`/`max`/`null_count` for free.
/// This function reads the footer and merges them into one value per
/// declared property.
///
/// Defensive: a property with no matching Parquet column, no statistics,
/// or a type mismatch between the declared `DataType` and the Parquet
/// physical type falls back to [`PropertyColumnStats::empty`]. The
/// optimizer treats `min: None` / `max: None` as "no stats" and uses
/// its folklore fallbacks (RFC-010 §2).
fn compute_property_stats(
    label: &LabelDef,
    body: &Bytes,
    hlls: &BTreeMap<String, Hll>,
) -> Result<Vec<PropertyColumnStats>> {
    if label.properties.is_empty() {
        return Ok(Vec::new());
    }
    let builder = ParquetRecordBatchReaderBuilder::try_new(body.clone())
        .map_err(|e| Error::invariant(format!("parquet open for stats: {e}")))?;
    let md = builder.metadata().clone();
    let schema_descr = md.file_metadata().schema_descr();

    let mut out = Vec::with_capacity(label.properties.len());
    for prop in &label.properties {
        let col_name = prop_column_name(prop);
        let leaf_idx = schema_descr
            .columns()
            .iter()
            .position(|c| c.name() == col_name);
        let Some(leaf_idx) = leaf_idx else {
            out.push(PropertyColumnStats::empty(col_name));
            continue;
        };

        let mut null_count: u64 = 0;
        let mut acc_min: Option<StatScalar> = None;
        let mut acc_max: Option<StatScalar> = None;
        for rg in md.row_groups() {
            let cc = rg.column(leaf_idx);
            let Some(stats) = cc.statistics() else {
                continue;
            };
            null_count = null_count.saturating_add(stats.null_count_opt().unwrap_or(0));
            if let Some(s) = parquet_stat_to_min(stats, &prop.data_type) {
                acc_min = match acc_min {
                    Some(prev) => Some(min_scalar(prev, s)),
                    None => Some(s),
                };
            }
            if let Some(s) = parquet_stat_to_max(stats, &prop.data_type) {
                acc_max = match acc_max {
                    Some(prev) => Some(max_scalar(prev, s)),
                    None => Some(s),
                };
            }
        }
        // The HLL sketch is populated incrementally during `write_batch`
        // (only for supported scalar types — vectors / JSON columns are
        // skipped because cardinality estimates over them are not
        // meaningful for the optimizer). Empty sketches still serialise
        // — `Hll::estimate()` returns 0 — but a `None` is what the
        // optimizer expects when there's no useful signal.
        let ndv_estimate = hlls.get(&col_name).and_then(|h| {
            if h.is_empty() {
                None
            } else {
                Some(HllSketchBytes(h.to_bytes()))
            }
        });
        out.push(PropertyColumnStats {
            name: col_name,
            null_count,
            min: acc_min,
            max: acc_max,
            ndv_estimate,
        });
    }
    Ok(out)
}

/// Should the writer accumulate an HLL sketch for a column of the
/// given declared `DataType`? Vector embeddings and JSON columns don't
/// expose a meaningful ordering / equality for cardinality purposes;
/// the optimizer would not benefit from `ndv` estimates on them.
fn hll_supported_for_datatype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Bool
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::Date32
            | DataType::TimestampMicrosUtc
    )
}

/// Walk each declared property column of `batch` and feed non-null
/// values into the corresponding HLL sketch. Defensive: an empty
/// column, a missing array, or a downcast failure simply skips that
/// column without erroring — the optimizer falls back to the eq
/// fallback if `ndv_estimate` is `None`.
fn update_property_hlls(
    label: &LabelDef,
    hlls: &mut BTreeMap<String, Hll>,
    batch: &RecordBatch,
) -> Result<()> {
    for prop in &label.properties {
        let col_name = prop_column_name(prop);
        let Some(hll) = hlls.get_mut(&col_name) else {
            continue;
        };
        let Some(column) = batch.column_by_name(&col_name) else {
            continue;
        };
        feed_column_into_hll(hll, column.as_ref(), &prop.data_type);
    }
    Ok(())
}

fn feed_column_into_hll(hll: &mut Hll, array: &dyn Array, dt: &DataType) {
    match dt {
        DataType::Bool => {
            if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    let byte = if a.value(i) { 1u8 } else { 0u8 };
                    hll.add_hash(hash_bytes(&[byte]));
                }
            }
        }
        DataType::Int32 => {
            if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    hll.add_hash(hash_bytes(&a.value(i).to_le_bytes()));
                }
            }
        }
        DataType::Int64 => {
            if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    hll.add_hash(hash_bytes(&a.value(i).to_le_bytes()));
                }
            }
        }
        DataType::Float32 => {
            if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    let v = a.value(i);
                    let normalized = if v.is_nan() { f32::NAN } else { v };
                    hll.add_hash(hash_bytes(&normalized.to_le_bytes()));
                }
            }
        }
        DataType::Float64 => {
            if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    let v = a.value(i);
                    let normalized = if v.is_nan() { f64::NAN } else { v };
                    hll.add_hash(hash_bytes(&normalized.to_le_bytes()));
                }
            }
        }
        DataType::Utf8 => {
            if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    hll.add_hash(hash_bytes(a.value(i).as_bytes()));
                }
            }
        }
        DataType::LargeUtf8 => {
            if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    hll.add_hash(hash_bytes(a.value(i).as_bytes()));
                }
            }
        }
        DataType::Binary => {
            if let Some(a) = array.as_any().downcast_ref::<BinaryArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    hll.add_hash(hash_bytes(a.value(i)));
                }
            }
        }
        DataType::Date32 => {
            if let Some(a) = array.as_any().downcast_ref::<Date32Array>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    hll.add_hash(hash_bytes(&a.value(i).to_le_bytes()));
                }
            }
        }
        DataType::TimestampMicrosUtc => {
            if let Some(a) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
                for i in 0..a.len() {
                    if a.is_null(i) {
                        continue;
                    }
                    hll.add_hash(hash_bytes(&a.value(i).to_le_bytes()));
                }
            }
        }
        // FixedSizeBinary (vector embeddings) and JSON are not fed into
        // HLL — they are filtered out at construction time by
        // `hll_supported_for_datatype`. This match arm is defensive in
        // case the schema declares an unsupported type but the writer
        // still tries to feed it.
        _ => {}
    }
}

/// Build a per-row-group `PropertyColumnStats` view from chunk-level
/// Parquet statistics. Used by `scan_with_predicates` (RFC-013).
/// Returns an empty `PropertyColumnStats` when stats are absent or the
/// scalar types don't match the declared `DataType` — the predicate
/// evaluator interprets those defensively as `MaybePresent`.
fn synthesize_property_stats(
    column_name: &str,
    dt: &DataType,
    parquet_stats: Option<&ParquetStatistics>,
) -> PropertyColumnStats {
    let Some(stats) = parquet_stats else {
        return PropertyColumnStats::empty(column_name);
    };
    let null_count = stats.null_count_opt().unwrap_or(0);
    let min = parquet_stat_to_min(stats, dt);
    let max = parquet_stat_to_max(stats, dt);
    PropertyColumnStats {
        name: column_name.to_string(),
        null_count,
        min,
        max,
        ndv_estimate: None,
    }
}

fn parquet_stat_to_min(stats: &ParquetStatistics, dt: &DataType) -> Option<StatScalar> {
    parquet_stat_to_scalar(stats, dt, ParquetStatBound::Min)
}

fn parquet_stat_to_max(stats: &ParquetStatistics, dt: &DataType) -> Option<StatScalar> {
    parquet_stat_to_scalar(stats, dt, ParquetStatBound::Max)
}

#[derive(Copy, Clone)]
enum ParquetStatBound {
    Min,
    Max,
}

fn parquet_stat_to_scalar(
    stats: &ParquetStatistics,
    dt: &DataType,
    bound: ParquetStatBound,
) -> Option<StatScalar> {
    match (stats, dt) {
        (ParquetStatistics::Boolean(s), DataType::Bool) => {
            extract_bool(s, bound).map(StatScalar::Bool)
        }
        (ParquetStatistics::Int32(s), DataType::Int32) => {
            extract_i32(s, bound).map(StatScalar::Int32)
        }
        (ParquetStatistics::Int32(s), DataType::Date32) => {
            extract_i32(s, bound).map(StatScalar::Date32)
        }
        (ParquetStatistics::Int64(s), DataType::Int64) => {
            extract_i64(s, bound).map(StatScalar::Int64)
        }
        (ParquetStatistics::Int64(s), DataType::TimestampMicrosUtc) => {
            extract_i64(s, bound).map(StatScalar::TimestampMicrosUtc)
        }
        (ParquetStatistics::Float(s), DataType::Float32) => {
            extract_f32(s, bound).map(StatScalar::Float32)
        }
        (ParquetStatistics::Double(s), DataType::Float64) => {
            extract_f64(s, bound).map(StatScalar::Float64)
        }
        (ParquetStatistics::ByteArray(s), DataType::Utf8) => {
            extract_byte_array(s, bound).and_then(|bytes| {
                std::str::from_utf8(&bytes)
                    .ok()
                    .map(|s| StatScalar::Utf8(s.to_string()))
            })
        }
        (ParquetStatistics::ByteArray(s), DataType::LargeUtf8) => extract_byte_array(s, bound)
            .and_then(|bytes| {
                std::str::from_utf8(&bytes)
                    .ok()
                    .map(|s| StatScalar::LargeUtf8(s.to_string()))
            }),
        (ParquetStatistics::ByteArray(s), DataType::Binary)
        | (ParquetStatistics::ByteArray(s), DataType::Json) => {
            extract_byte_array(s, bound).map(StatScalar::Binary)
        }
        // FixedLenByteArray (vector embeddings) does not surface stats
        // we know how to compare lex/numerically — skip.
        _ => None,
    }
}

fn extract_bool(
    s: &parquet::file::statistics::ValueStatistics<bool>,
    bound: ParquetStatBound,
) -> Option<bool> {
    match bound {
        ParquetStatBound::Min => s.min_opt().copied(),
        ParquetStatBound::Max => s.max_opt().copied(),
    }
}

fn extract_i32(
    s: &parquet::file::statistics::ValueStatistics<i32>,
    bound: ParquetStatBound,
) -> Option<i32> {
    match bound {
        ParquetStatBound::Min => s.min_opt().copied(),
        ParquetStatBound::Max => s.max_opt().copied(),
    }
}

fn extract_i64(
    s: &parquet::file::statistics::ValueStatistics<i64>,
    bound: ParquetStatBound,
) -> Option<i64> {
    match bound {
        ParquetStatBound::Min => s.min_opt().copied(),
        ParquetStatBound::Max => s.max_opt().copied(),
    }
}

fn extract_f32(
    s: &parquet::file::statistics::ValueStatistics<f32>,
    bound: ParquetStatBound,
) -> Option<f32> {
    match bound {
        ParquetStatBound::Min => s.min_opt().copied(),
        ParquetStatBound::Max => s.max_opt().copied(),
    }
}

fn extract_f64(
    s: &parquet::file::statistics::ValueStatistics<f64>,
    bound: ParquetStatBound,
) -> Option<f64> {
    match bound {
        ParquetStatBound::Min => s.min_opt().copied(),
        ParquetStatBound::Max => s.max_opt().copied(),
    }
}

fn extract_byte_array(
    s: &parquet::file::statistics::ValueStatistics<parquet::data_type::ByteArray>,
    bound: ParquetStatBound,
) -> Option<Vec<u8>> {
    match bound {
        ParquetStatBound::Min => s.min_opt().map(|b| b.data().to_vec()),
        ParquetStatBound::Max => s.max_opt().map(|b| b.data().to_vec()),
    }
}

pub(crate) fn min_scalar(a: StatScalar, b: StatScalar) -> StatScalar {
    if scalar_lt(&a, &b) {
        a
    } else {
        b
    }
}

pub(crate) fn max_scalar(a: StatScalar, b: StatScalar) -> StatScalar {
    if scalar_lt(&a, &b) {
        b
    } else {
        a
    }
}

/// Total ordering on [`StatScalar`] within the same type variant. Cross-
/// type comparisons return `false` (treated as equal) — those indicate
/// schema drift and the caller falls through to a single value.
fn scalar_lt(a: &StatScalar, b: &StatScalar) -> bool {
    match (a, b) {
        (StatScalar::Bool(x), StatScalar::Bool(y)) => x < y,
        (StatScalar::Int32(x), StatScalar::Int32(y)) => x < y,
        (StatScalar::Int64(x), StatScalar::Int64(y)) => x < y,
        (StatScalar::Float32(x), StatScalar::Float32(y)) => x < y,
        (StatScalar::Float64(x), StatScalar::Float64(y)) => x < y,
        (StatScalar::Utf8(x), StatScalar::Utf8(y)) => x < y,
        (StatScalar::LargeUtf8(x), StatScalar::LargeUtf8(y)) => x < y,
        (StatScalar::Binary(x), StatScalar::Binary(y)) => x < y,
        (StatScalar::Date32(x), StatScalar::Date32(y)) => x < y,
        (StatScalar::TimestampMicrosUtc(x), StatScalar::TimestampMicrosUtc(y)) => x < y,
        _ => false,
    }
}

fn hex_short(b: &[u8; 16]) -> String {
    let mut s = String::with_capacity(8);
    for byte in &b[..4] {
        use std::fmt::Write as _;
        let _ = write!(s, "{byte:02x}");
    }
    s.push('…');
    s
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::builder::{
        BooleanBuilder, FixedSizeBinaryBuilder, StringBuilder, UInt64Builder,
    };
    use arrow_array::cast::AsArray;
    use arrow_array::ArrayRef;

    use namidb_core::DataType;

    use super::*;

    fn person_label() -> LabelDef {
        LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("age", DataType::Int32, true).unwrap(),
            ],
        }
    }

    /// (seed_byte, tombstone, lsn, name_or_none, age_or_none, schema_version, overflow_or_none)
    type Row<'a> = (
        u8,
        bool,
        u64,
        Option<&'a str>,
        Option<i32>,
        u64,
        Option<&'a str>,
    );

    fn build_batch(rows: &[Row<'_>]) -> RecordBatch {
        let label = person_label();
        let schema = node_arrow_schema(&label);

        let mut nid = FixedSizeBinaryBuilder::with_capacity(rows.len(), 16);
        let mut tomb = BooleanBuilder::with_capacity(rows.len());
        let mut lsn = UInt64Builder::with_capacity(rows.len());
        let mut labels =
            arrow_array::builder::ListBuilder::new(arrow_array::builder::UInt32Builder::new());
        let mut prop_name = StringBuilder::with_capacity(rows.len(), 32);
        let mut prop_age = arrow_array::builder::Int32Builder::with_capacity(rows.len());
        let mut overflow = StringBuilder::with_capacity(rows.len(), 32);
        let mut sv = UInt64Builder::with_capacity(rows.len());

        for (seed, t, l, name_opt, age_opt, schema_v, ovf) in rows {
            let mut id = [0u8; 16];
            id[15] = *seed;
            nid.append_value(id).unwrap();
            tomb.append_value(*t);
            lsn.append_value(*l);
            labels.append(true); // empty __labels list for these property tests
            match name_opt {
                Some(n) => prop_name.append_value(n),
                None => prop_name.append_null(),
            }
            match age_opt {
                Some(a) => prop_age.append_value(*a),
                None => prop_age.append_null(),
            }
            match ovf {
                Some(s) => overflow.append_value(s),
                None => overflow.append_null(),
            }
            sv.append_value(*schema_v);
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(nid.finish()),
            Arc::new(tomb.finish()),
            Arc::new(lsn.finish()),
            Arc::new(labels.finish()),
            Arc::new(prop_name.finish()),
            Arc::new(prop_age.finish()),
            Arc::new(overflow.finish()),
            Arc::new(sv.finish()),
        ];
        RecordBatch::try_new(schema, columns).unwrap()
    }

    #[test]
    fn round_trip_one_batch() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label.clone(), NodeSstWriterOptions::default()).unwrap();
        let batch = build_batch(&[
            (1, false, 10, Some("Alice"), Some(30), 1, None),
            (
                2,
                false,
                11,
                Some("Bob"),
                None,
                1,
                Some(r#"{"city":"Quito"}"#),
            ),
            (3, true, 12, None, None, 1, None),
        ]);
        w.write_batch(&batch).unwrap();
        let finish = w.finish().unwrap();
        assert_eq!(finish.stats.row_count, 3);
        assert_eq!(finish.stats.tombstone_count, 1);
        assert_eq!(finish.stats.min_node_id[15], 1);
        assert_eq!(finish.stats.max_node_id[15], 3);
        assert_eq!(finish.stats.min_lsn, 10);
        assert_eq!(finish.stats.max_lsn, 12);
        assert_eq!(finish.stats.schema_version_min, 1);
        assert_eq!(finish.stats.schema_version_max, 1);

        let reader = NodeSstReader::open(label, finish.body).unwrap();
        let scanned = reader.scan().unwrap();
        let total_rows: usize = scanned.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);
    }

    #[test]
    fn rejects_out_of_order_node_ids() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        let batch = build_batch(&[
            (2, false, 10, Some("Bob"), None, 1, None),
            (1, false, 11, Some("Alice"), None, 1, None),
        ]);
        let err = w.write_batch(&batch).unwrap_err();
        match err {
            Error::Invariant(msg) => assert!(msg.contains("sorted")),
            other => panic!("expected Invariant, got {other:?}"),
        }
    }

    #[test]
    fn overflow_round_trips() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label.clone(), NodeSstWriterOptions::default()).unwrap();
        let payload = r#"{"city":"Cuenca","tags":["leg","gov"]}"#;
        let batch = build_batch(&[(7, false, 99, Some("X"), Some(7), 1, Some(payload))]);
        w.write_batch(&batch).unwrap();
        let finish = w.finish().unwrap();
        let reader = NodeSstReader::open(label, finish.body).unwrap();
        let batches = reader.scan().unwrap();
        let b = &batches[0];
        let overflow_col = b.column_by_name(OVERFLOW_JSON).unwrap().as_string::<i32>();
        assert_eq!(overflow_col.value(0), payload);
    }

    #[test]
    fn schema_version_tracks_min_max() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label.clone(), NodeSstWriterOptions::default()).unwrap();
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("a"), Some(1), 5, None),
            (2, false, 2, Some("b"), Some(2), 7, None),
            (3, false, 3, Some("c"), Some(3), 6, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        assert_eq!(finish.stats.schema_version_min, 5);
        assert_eq!(finish.stats.schema_version_max, 7);
    }

    #[test]
    fn small_sst_omits_bloom() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        // 1 row → body is tiny, well below the 256 KiB threshold.
        w.write_batch(&build_batch(&[(1, false, 1, Some("a"), Some(1), 0, None)]))
            .unwrap();
        let finish = w.finish().unwrap();
        assert!(finish.bloom.is_none());
        assert!(finish.body.len() < 256 * 1024);
    }

    #[test]
    fn property_stats_capture_int32_min_max_and_nulls() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        // ages: 30, NULL, 18, 99, NULL. min=18, max=99, null_count=2.
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("a"), Some(30), 0, None),
            (2, false, 2, Some("b"), None, 0, None),
            (3, false, 3, Some("c"), Some(18), 0, None),
            (4, false, 4, Some("d"), Some(99), 0, None),
            (5, false, 5, Some("e"), None, 0, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        let age = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_age")
            .expect("age stats present");
        assert_eq!(age.null_count, 2);
        assert_eq!(age.min, Some(StatScalar::Int32(18)));
        assert_eq!(age.max, Some(StatScalar::Int32(99)));
    }

    #[test]
    fn property_stats_capture_utf8_min_max() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("Charlie"), Some(1), 0, None),
            (2, false, 2, Some("Alice"), Some(2), 0, None),
            (3, false, 3, Some("Bob"), Some(3), 0, None),
            (4, false, 4, Some("Zoe"), Some(4), 0, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        let name = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_name")
            .expect("name stats present");
        assert_eq!(name.null_count, 0);
        assert_eq!(name.min, Some(StatScalar::Utf8("Alice".into())));
        assert_eq!(name.max, Some(StatScalar::Utf8("Zoe".into())));
    }

    #[test]
    fn property_stats_full_null_column_keeps_min_max_none() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        // All `age` values NULL.
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("a"), None, 0, None),
            (2, false, 2, Some("b"), None, 0, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        let age = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_age")
            .expect("age stats present");
        assert_eq!(age.null_count, 2);
        assert_eq!(age.min, None);
        assert_eq!(age.max, None);
    }

    #[test]
    fn property_stats_merge_across_row_groups() {
        let label = person_label();
        // Force two row groups so stats have to be merged across them.
        let opts = NodeSstWriterOptions {
            row_group_target_rows: 2,
            ..NodeSstWriterOptions::default()
        };
        let mut w = NodeSstWriter::new(label, opts).unwrap();
        // RG1: ages 50, 60. RG2: ages 10, 90. Merged: min=10, max=90.
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("a"), Some(50), 0, None),
            (2, false, 2, Some("b"), Some(60), 0, None),
            (3, false, 3, Some("c"), Some(10), 0, None),
            (4, false, 4, Some("d"), Some(90), 0, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        let age = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_age")
            .expect("age stats present");
        assert_eq!(age.min, Some(StatScalar::Int32(10)));
        assert_eq!(age.max, Some(StatScalar::Int32(90)));
    }

    #[test]
    fn property_stats_capture_ndv_estimate() {
        // 6 unique names → HLL estimate should fall in [5, 8].
        let label = person_label();
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("Alice"), Some(30), 0, None),
            (2, false, 2, Some("Bob"), Some(25), 0, None),
            (3, false, 3, Some("Carol"), Some(35), 0, None),
            (4, false, 4, Some("Dave"), Some(28), 0, None),
            (5, false, 5, Some("Eve"), Some(40), 0, None),
            (6, false, 6, Some("Frank"), Some(33), 0, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        let name = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_name")
            .expect("name stats present");
        let sketch = name.ndv_estimate.as_ref().expect("HLL sketch attached");
        let hll = crate::sst::hll::Hll::from_bytes(sketch.as_bytes()).unwrap();
        let est = hll.estimate();
        assert!(
            (5..=8).contains(&est),
            "expected ndv ~6, got {} (sketch {} bytes)",
            est,
            sketch.as_bytes().len()
        );

        // The age column also gets a sketch — 6 distinct ints.
        let age = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_age")
            .expect("age stats present");
        let age_sketch = age.ndv_estimate.as_ref().expect("age sketch present");
        let age_hll = crate::sst::hll::Hll::from_bytes(age_sketch.as_bytes()).unwrap();
        let age_est = age_hll.estimate();
        assert!(
            (5..=8).contains(&age_est),
            "expected age ndv ~6, got {}",
            age_est
        );
    }

    #[test]
    fn property_stats_ndv_estimate_merges_across_row_groups() {
        // Two row groups, distinct names across both. Total 4 unique names.
        let label = person_label();
        let opts = NodeSstWriterOptions {
            row_group_target_rows: 2,
            ..NodeSstWriterOptions::default()
        };
        let mut w = NodeSstWriter::new(label, opts).unwrap();
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("Alice"), Some(30), 0, None),
            (2, false, 2, Some("Bob"), Some(25), 0, None),
            (3, false, 3, Some("Carol"), Some(35), 0, None),
            (4, false, 4, Some("Dave"), Some(28), 0, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        let name = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_name")
            .expect("name stats present");
        let hll = crate::sst::hll::Hll::from_bytes(name.ndv_estimate.as_ref().unwrap().as_bytes())
            .unwrap();
        // 4 unique names across two row groups — HLL is per-writer (not
        // per-row-group), so the merge happens during `write_batch`, not
        // post-hoc. Either way: 4 distinct → estimate close to 4.
        assert!(
            (3..=6).contains(&hll.estimate()),
            "expected ndv ~4, got {}",
            hll.estimate()
        );
    }

    #[test]
    fn property_stats_all_null_column_skips_ndv() {
        let label = person_label();
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        w.write_batch(&build_batch(&[
            (1, false, 1, Some("a"), None, 0, None),
            (2, false, 2, Some("b"), None, 0, None),
        ]))
        .unwrap();
        let finish = w.finish().unwrap();
        let age = finish
            .stats
            .property_stats
            .iter()
            .find(|p| p.name == "prop_age")
            .expect("age stats present");
        // No non-null ages → HLL never touched → sketch absent.
        assert!(
            age.ndv_estimate.is_none(),
            "all-null column should not ship an HLL sketch"
        );
    }

    #[test]
    fn property_stats_empty_label_returns_empty_vec() {
        let label = LabelDef {
            name: "Empty".into(),
            properties: vec![],
        };
        let mut w = NodeSstWriter::new(label, NodeSstWriterOptions::default()).unwrap();
        // Build a batch with just engine columns — no declared props.
        let schema = node_arrow_schema(&LabelDef {
            name: "Empty".into(),
            properties: vec![],
        });
        let mut nid = FixedSizeBinaryBuilder::with_capacity(1, 16);
        nid.append_value([0u8; 16]).unwrap();
        let mut tomb = BooleanBuilder::with_capacity(1);
        tomb.append_value(false);
        let mut lsn = UInt64Builder::with_capacity(1);
        lsn.append_value(1);
        // `node_arrow_schema` now carries a `__labels` `List<UInt32>` column at
        // slot index 3 (between `lsn` and the overflow JSON); emit an empty
        // label list so the batch matches the schema's field count.
        let mut labels =
            arrow_array::builder::ListBuilder::new(arrow_array::builder::UInt32Builder::new());
        labels.append(true);
        let mut overflow = StringBuilder::with_capacity(1, 0);
        overflow.append_null();
        let mut sv = UInt64Builder::with_capacity(1);
        sv.append_value(0);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(nid.finish()),
            Arc::new(tomb.finish()),
            Arc::new(lsn.finish()),
            Arc::new(labels.finish()),
            Arc::new(overflow.finish()),
            Arc::new(sv.finish()),
        ];
        let batch = RecordBatch::try_new(schema, columns).unwrap();
        w.write_batch(&batch).unwrap();
        let finish = w.finish().unwrap();
        assert!(finish.stats.property_stats.is_empty());
    }

    // ─── scan_with_predicates ────────────────────────────────────

    fn build_age_split_sst(ages_per_rg: &[Vec<Option<i32>>]) -> (LabelDef, Bytes) {
        // All chunks must share a length so we can pick a single
        // `row_group_target_rows` that splits the body at chunk boundaries.
        let chunk_size = ages_per_rg[0].len();
        assert!(
            ages_per_rg.iter().all(|r| r.len() == chunk_size),
            "all row-group chunks must share length for uniform RG split",
        );
        let label = person_label();
        let opts = NodeSstWriterOptions {
            row_group_target_rows: chunk_size,
            ..NodeSstWriterOptions::default()
        };
        let mut w = NodeSstWriter::new(label.clone(), opts).unwrap();
        let mut id_seq: u8 = 1;
        let mut rows: Vec<Row<'_>> = Vec::new();
        for rg in ages_per_rg {
            for age in rg {
                rows.push((id_seq, false, id_seq as u64, Some("x"), *age, 0u64, None));
                id_seq += 1;
            }
        }
        let batch = build_batch(&rows);
        w.write_batch(&batch).unwrap();
        let finish = w.finish().unwrap();
        (label, finish.body)
    }

    #[test]
    fn scan_with_empty_predicates_returns_all_rows() {
        let (label, body) = build_age_split_sst(&[vec![Some(30), Some(40), Some(50)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let batches = reader.scan_with_predicates(&[]).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn scan_with_predicates_skips_row_group_above_range() {
        // RG1 ages 10,20 ; RG2 ages 80,90. WHERE age < 30 → only RG1.
        let (label, body) =
            build_age_split_sst(&[vec![Some(10), Some(20)], vec![Some(80), Some(90)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicate = ScanPredicate::Lt {
            column: "age".into(),
            value: StatScalar::Int32(30),
        };
        let batches = reader.scan_with_predicates(&[predicate]).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // Row-group pruning leaves 2 rows from RG1; row-level filter
        // happens upstream.
        assert_eq!(total, 2);
    }

    #[test]
    fn scan_with_predicates_skips_row_group_below_range() {
        let (label, body) =
            build_age_split_sst(&[vec![Some(10), Some(20)], vec![Some(80), Some(90)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicate = ScanPredicate::Gt {
            column: "age".into(),
            value: StatScalar::Int32(50),
        };
        let batches = reader.scan_with_predicates(&[predicate]).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn scan_with_predicates_skips_all_row_groups_when_out_of_range() {
        let (label, body) =
            build_age_split_sst(&[vec![Some(10), Some(20)], vec![Some(80), Some(90)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicate = ScanPredicate::Gt {
            column: "age".into(),
            value: StatScalar::Int32(999),
        };
        let batches = reader.scan_with_predicates(&[predicate]).unwrap();
        assert!(batches.is_empty(), "every row-group should be skipped");
    }

    #[test]
    fn scan_with_predicates_keeps_all_row_groups_in_range() {
        // RG1 ages 10..30, RG2 ages 20..40 → WHERE age > 5 keeps both.
        let (label, body) = build_age_split_sst(&[
            vec![Some(10), Some(20), Some(30)],
            vec![Some(20), Some(30), Some(40)],
        ]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicate = ScanPredicate::Gt {
            column: "age".into(),
            value: StatScalar::Int32(5),
        };
        let batches = reader.scan_with_predicates(&[predicate]).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn scan_with_predicates_uses_eq_pruning() {
        // RG1 ages 10..30, RG2 ages 80..100. WHERE age = 55 → both skip.
        let (label, body) = build_age_split_sst(&[
            vec![Some(10), Some(20), Some(30)],
            vec![Some(80), Some(90), Some(100)],
        ]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicate = ScanPredicate::Eq {
            column: "age".into(),
            value: StatScalar::Int32(55),
        };
        let batches = reader.scan_with_predicates(&[predicate]).unwrap();
        assert!(batches.is_empty());
    }

    #[test]
    fn scan_with_predicates_handles_multi_predicate_and() {
        // RG1 ages 10..30; RG2 ages 50..70. Two predicates:
        // age > 40 → RG1 absent, RG2 keep
        // age < 60 → RG1 keep, RG2 maybepresent (max=70)
        // AND of verdicts = keep iff both MaybePresent. → only RG2 survives.
        let (label, body) = build_age_split_sst(&[
            vec![Some(10), Some(20), Some(30)],
            vec![Some(50), Some(60), Some(70)],
        ]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicates = vec![
            ScanPredicate::Gt {
                column: "age".into(),
                value: StatScalar::Int32(40),
            },
            ScanPredicate::Lt {
                column: "age".into(),
                value: StatScalar::Int32(60),
            },
        ];
        let batches = reader.scan_with_predicates(&predicates).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // RG2 (3 rows) decodes; row-level filter elsewhere drops to 1.
        assert_eq!(total, 3);
    }

    #[test]
    fn scan_with_predicates_passes_through_when_column_undeclared() {
        // Predicate column not declared in label → defensive
        // MaybePresent → full scan (no pruning).
        let (label, body) = build_age_split_sst(&[vec![Some(30), Some(40)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicate = ScanPredicate::Eq {
            column: "doesnotexist".into(),
            value: StatScalar::Int32(99),
        };
        let batches = reader.scan_with_predicates(&[predicate]).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn scan_with_predicates_handles_is_null() {
        // RG1 mixes nulls; RG2 has no nulls. WHERE age IS NULL → RG1 keep,
        // RG2 absent.
        let (label, body) = build_age_split_sst(&[
            vec![Some(10), None, Some(30)],
            vec![Some(40), Some(50), Some(60)],
        ]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let predicate = ScanPredicate::IsNull {
            column: "age".into(),
        };
        let batches = reader.scan_with_predicates(&[predicate]).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3); // RG1 only
    }

    // ─── projection ──────────────────────────────────────────────

    #[test]
    fn scan_with_projection_some_includes_only_requested_property_columns() {
        let (label, body) = build_age_split_sst(&[vec![Some(30), Some(40)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        // Project only `age`; `name` is NOT requested.
        let proj = vec!["age".to_string()];
        let batches = reader
            .scan_with_predicates_and_projection(&[], Some(&proj))
            .unwrap();
        let batch = &batches[0];
        // Engine columns + age must be present; name must NOT be.
        assert!(batch.column_by_name(COL_NODE_ID).is_some());
        assert!(batch.column_by_name(COL_TOMBSTONE).is_some());
        assert!(batch.column_by_name(COL_LSN).is_some());
        assert!(batch.column_by_name(SCHEMA_VERSION).is_some());
        assert!(batch.column_by_name(OVERFLOW_JSON).is_some());
        // `__labels` must survive projection: it is the id-primary source of
        // truth for a row's label set, and dropping it makes label scans return
        // nothing (its Parquet leaf is nested, so the mask matches on path root).
        assert!(
            batch.column_by_name(COL_LABELS).is_some(),
            "__labels must always be kept in a projected scan"
        );
        assert!(batch.column_by_name("prop_age").is_some());
        assert!(
            batch.column_by_name("prop_name").is_none(),
            "name should have been projected out, columns={:?}",
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn scan_with_projection_none_reads_every_declared_column() {
        let (label, body) = build_age_split_sst(&[vec![Some(30)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let batches = reader
            .scan_with_predicates_and_projection(&[], None)
            .unwrap();
        let batch = &batches[0];
        assert!(batch.column_by_name("prop_age").is_some());
        assert!(batch.column_by_name("prop_name").is_some());
    }

    #[test]
    fn scan_with_projection_empty_keeps_only_engine_columns() {
        let (label, body) = build_age_split_sst(&[vec![Some(30)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let proj: Vec<String> = Vec::new();
        let batches = reader
            .scan_with_predicates_and_projection(&[], Some(&proj))
            .unwrap();
        let batch = &batches[0];
        assert!(batch.column_by_name(COL_NODE_ID).is_some());
        assert!(batch.column_by_name("prop_name").is_none());
        assert!(batch.column_by_name("prop_age").is_none());
    }

    #[test]
    fn scan_with_projection_unknown_property_is_silently_skipped() {
        let (label, body) = build_age_split_sst(&[vec![Some(30)]]);
        let reader = NodeSstReader::open(label, body).unwrap();
        let proj = vec!["does_not_exist".to_string(), "age".to_string()];
        let batches = reader
            .scan_with_predicates_and_projection(&[], Some(&proj))
            .unwrap();
        let batch = &batches[0];
        // Engine + age present; missing property simply not included.
        assert!(batch.column_by_name("prop_age").is_some());
    }

    /// 12 rows at 4 per row group → 3 row groups over seeds 1..=12.
    fn build_multi_row_group_sst() -> (LabelDef, Bytes) {
        let label = person_label();
        let opts = NodeSstWriterOptions {
            row_group_target_rows: 4,
            ..Default::default()
        };
        let mut w = NodeSstWriter::new(label.clone(), opts).unwrap();
        let rows: Vec<Row<'_>> = (1..=12u8)
            .map(|seed| (seed, false, 10 + seed as u64, Some("n"), Some(1), 1, None))
            .collect();
        w.write_batch(&build_batch(&rows)).unwrap();
        (label, w.finish().unwrap().body)
    }

    fn seed_key(seed: u8) -> [u8; 16] {
        let mut k = [0u8; 16];
        k[15] = seed;
        k
    }

    #[test]
    fn row_groups_for_keys_prunes_by_node_id_stats() {
        let (_label, body) = build_multi_row_group_sst();
        let md = parse_node_sst_metadata(&body).unwrap();
        assert_eq!(md.num_row_groups(), 3);

        // Keys in row group 0 (seeds 1..=4) and 2 (seeds 9..=12) only.
        let keys = [seed_key(2), seed_key(3), seed_key(10)];
        assert_eq!(row_groups_for_keys(&md, &keys).unwrap(), vec![0, 2]);
        // A key between row groups' rows still maps to exactly its group.
        assert_eq!(row_groups_for_keys(&md, &[seed_key(6)]).unwrap(), vec![1]);
        // Out-of-range keys prune everything.
        assert_eq!(
            row_groups_for_keys(&md, &[seed_key(0), seed_key(200)]).unwrap(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn scan_row_groups_split_reassembles_footer_row_counts() {
        let (label, body) = build_multi_row_group_sst();
        let md = parse_node_sst_metadata(&body).unwrap();
        let reader = NodeSstReader::open(label, body).unwrap();

        let selected = vec![0usize, 2];
        let batches = reader.scan_row_groups(selected.clone()).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 8, "two row groups of 4 rows each");

        let split = split_batches_by_row_group(&md, &selected, batches).unwrap();
        assert_eq!(split.len(), 2);
        for (rg, group) in &split {
            let rows: usize = group.iter().map(|b| b.num_rows()).sum();
            assert_eq!(rows, md.row_group(*rg).num_rows() as usize);
        }
        // Row group 2 holds seeds 9..=12; its first row must be seed 9.
        let (rg, group) = &split[1];
        assert_eq!(*rg, 2);
        let id_col = group[0]
            .column_by_name(COL_NODE_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        assert_eq!(id_col.value(0), seed_key(9));
    }

    #[test]
    fn split_batches_rejects_row_count_mismatch() {
        let (label, body) = build_multi_row_group_sst();
        let md = parse_node_sst_metadata(&body).unwrap();
        let reader = NodeSstReader::open(label, body).unwrap();
        let batches = reader.scan_row_groups(vec![0]).unwrap();
        // Claiming two row groups for one row group's rows must fail, so a
        // partial decode can never be cached as a complete row group.
        assert!(split_batches_by_row_group(&md, &[0, 1], batches).is_err());
    }
}
