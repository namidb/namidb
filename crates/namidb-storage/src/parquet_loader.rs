//! Parquet → `WriterSession` ingest path.
//!
//! Reads node rows from a Parquet file and feeds them through the public
//! `WriterSession::upsert_node` / `commit_batch` surface. This is the
//! literal "ingest from Parquet" path the plan calls out — bulk
//! load from an Arrow/Parquet file, no per-record JSON, no client-side
//! `BTreeMap::insert` loop.
//!
//! ## Wire convention (v0)
//!
//! - Required column `node_id`: `FixedSizeBinary(16)`. The 16 bytes are
//! interpreted as the raw UUID for [`NodeId`].
//! - Every other column is treated as a property whose name is the
//! Parquet column name. Reserved names (`tombstone`, `lsn`) and any
//! name reaching engine-managed columns are rejected.
//! - Schema validation between the Parquet schema and the namespace
//! schema lives at flush time — the loader's job is only to stream
//! the rows into the memtable. A row whose Arrow type can't be
//! converted to [`Value`] fails the load (no silent skipping).
//!
//! ## Type mapping (Arrow → [`Value`])
//!
//! | Arrow type | Value |
//! |-------------------------------------|----------------|
//! | `Boolean` | `Value::Bool` |
//! | `Int8`/`Int16`/`Int32`/`Int64` | `Value::I64` |
//! | `UInt8`/`UInt16`/`UInt32` | `Value::I64` |
//! | `Float32`/`Float64` | `Value::F64` |
//! | `Utf8`/`LargeUtf8` | `Value::Str` |
//! | `Binary`/`LargeBinary` | `Value::Bytes` |
//! | `Date32` (days since epoch) | `Value::I64` |
//! | `Timestamp(Microsecond, _)` | `Value::I64` |
//! | `FixedSizeList<Float32, dim>` | `Value::Vec` |
//! | null | `Value::Null` |
//!
//! Anything else returns [`Error::invariant`] with the column name and
//! the offending type. We don't widen the table speculatively — every
//! Arrow type we accept needs a tested round-trip through flush.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use arrow_array::{
    cast::AsArray, types::*, Array, BinaryArray, BooleanArray, FixedSizeBinaryArray,
    FixedSizeListArray, LargeBinaryArray, LargeStringArray, RecordBatch, StringArray,
};
use arrow_schema::DataType as ArrowDataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use uuid::Uuid;

use namidb_core::{NodeId, Value};

use crate::error::{Error, Result};
use crate::flush::{EdgeWriteRecord, NodeWriteRecord};
use crate::ingest::WriterSession;

/// Default Parquet batch size. Picked to match the row-group target the
/// rest of the engine uses; the reader doesn't need to load a whole row
/// group at once but keeping the batch size aligned keeps the iteration
/// cost-stable.
const DEFAULT_BATCH_SIZE: usize = 8192;

/// Outcome of a successful [`load_nodes`] run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LoadOutcome {
    /// Rows pushed to the writer via `upsert_node`.
    pub rows_loaded: usize,
    /// Number of `commit_batch` calls fired during the load (not
    /// counting the implicit one inside `flush`).
    pub commit_batches: usize,
}

/// Load every row in a Parquet file as a node upsert for `label`.
///
/// The caller owns `writer` and is responsible for invoking `flush`
/// after the load (the loader leaves the final batch in the pending WAL
/// segment to give the caller control over flush cadence).
///
/// `commit_every` is the number of rows that may accumulate between
/// `commit_batch` calls. `0` means "never commit during the load, leave
/// everything pending for the caller's final flush" — useful when the
/// caller wants exactly one durability boundary at the end of the load.
pub async fn load_nodes(
    path: &Path,
    writer: &mut WriterSession,
    label: &str,
    commit_every: usize,
) -> Result<LoadOutcome> {
    let label = label.to_string();
    load_with(path, writer, commit_every, move |batch, writer, outcome| {
        ingest_batch(batch, writer, &label, outcome)
    })
    .await
}

/// Load every row in a Parquet file as an edge upsert for `edge_type`.
///
/// Wire convention: required `src` and `dst` columns of `FixedSizeBinary(16)`
/// (the raw endpoint UUIDs); every other column is a property. Endpoint nodes
/// are NOT auto-created, so load the node files first. Flush/commit cadence
/// matches [`load_nodes`]: the caller owns `writer` and flushes after the load.
pub async fn load_edges(
    path: &Path,
    writer: &mut WriterSession,
    edge_type: &str,
    commit_every: usize,
) -> Result<LoadOutcome> {
    let edge_type = edge_type.to_string();
    load_with(path, writer, commit_every, move |batch, writer, outcome| {
        ingest_edge_batch(batch, writer, &edge_type, outcome)
    })
    .await
}

/// Shared Parquet reader driver for [`load_nodes`] / [`load_edges`]: opens the
/// file, sizes the reader batch to the commit cadence, and streams each
/// `RecordBatch` through `ingest`, committing every `commit_every` rows.
async fn load_with<F>(
    path: &Path,
    writer: &mut WriterSession,
    commit_every: usize,
    mut ingest: F,
) -> Result<LoadOutcome>
where
    F: FnMut(&RecordBatch, &mut WriterSession, &mut LoadOutcome) -> Result<()>,
{
    let file = File::open(path)
        .map_err(|e| Error::invariant(format!("open parquet file {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::invariant(format!("open parquet reader: {e}")))?;
    // Cap the reader batch to the commit cadence: otherwise a small
    // `commit_every` would only fire once per (much larger) Arrow batch,
    // since we only check the commit threshold at batch boundaries.
    let batch_size = if commit_every > 0 {
        DEFAULT_BATCH_SIZE.min(commit_every)
    } else {
        DEFAULT_BATCH_SIZE
    };
    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .map_err(|e| Error::invariant(format!("build parquet reader: {e}")))?;

    let mut outcome = LoadOutcome::default();
    let mut rows_since_commit: usize = 0;

    for batch in reader {
        let batch: RecordBatch =
            batch.map_err(|e| Error::invariant(format!("read parquet batch: {e}")))?;
        ingest(&batch, writer, &mut outcome)?;

        rows_since_commit += batch.num_rows();
        if commit_every > 0 && rows_since_commit >= commit_every {
            writer.commit_batch().await?;
            outcome.commit_batches += 1;
            rows_since_commit = 0;
        }
    }
    Ok(outcome)
}

/// Process one node `RecordBatch`. Pulled out so the per-row loop is
/// testable without spinning up Parquet I/O.
fn ingest_batch(
    batch: &RecordBatch,
    writer: &mut WriterSession,
    label: &str,
    outcome: &mut LoadOutcome,
) -> Result<()> {
    let schema = batch.schema();
    let (id_idx, id_array) = fixed16_column(batch, "node_id")?;
    let prop_indices = collect_property_columns(&schema, &[id_idx])?;

    for row in 0..batch.num_rows() {
        if id_array.is_null(row) {
            return Err(Error::invariant(format!(
                "row {row}: 'node_id' is null; null ids are not allowed"
            )));
        }
        let id = NodeId::from_uuid(Uuid::from_bytes(fixed16_at(id_array, row)));
        let properties = build_row_properties(batch, &prop_indices, row)?;
        let record = NodeWriteRecord {
            properties,
            schema_version: 1,
            ..Default::default()
        };
        writer.upsert_node(label, id, &record)?;
        outcome.rows_loaded += 1;
    }
    Ok(())
}

/// Process one edge `RecordBatch`: `src`/`dst` endpoint columns plus
/// properties, one `upsert_edge` per row. Endpoint nodes must already exist
/// (a dangling endpoint is the caller's responsibility, surfaced at read time).
fn ingest_edge_batch(
    batch: &RecordBatch,
    writer: &mut WriterSession,
    edge_type: &str,
    outcome: &mut LoadOutcome,
) -> Result<()> {
    let schema = batch.schema();
    let (src_idx, src_array) = fixed16_column(batch, "src")?;
    let (dst_idx, dst_array) = fixed16_column(batch, "dst")?;
    let prop_indices = collect_property_columns(&schema, &[src_idx, dst_idx])?;

    for row in 0..batch.num_rows() {
        if src_array.is_null(row) || dst_array.is_null(row) {
            return Err(Error::invariant(format!(
                "row {row}: edge 'src'/'dst' endpoints must be non-null"
            )));
        }
        let src = NodeId::from_uuid(Uuid::from_bytes(fixed16_at(src_array, row)));
        let dst = NodeId::from_uuid(Uuid::from_bytes(fixed16_at(dst_array, row)));
        let properties = build_row_properties(batch, &prop_indices, row)?;
        let record = EdgeWriteRecord {
            properties,
            schema_version: 1,
        };
        writer.upsert_edge(edge_type, src, dst, &record)?;
        outcome.rows_loaded += 1;
    }
    Ok(())
}

/// Locate a required `FixedSizeBinary(16)` id/endpoint column by name,
/// returning its index and the typed array.
fn fixed16_column<'a>(
    batch: &'a RecordBatch,
    col: &str,
) -> Result<(usize, &'a FixedSizeBinaryArray)> {
    let idx = batch.schema().index_of(col).map_err(|_| {
        Error::invariant(format!("parquet schema is missing required '{col}' column"))
    })?;
    let column = batch.column(idx);
    let array = column
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            Error::invariant(format!(
                "'{col}' column must be FixedSizeBinary(16), got {:?}",
                column.data_type()
            ))
        })?;
    if array.value_length() != 16 {
        return Err(Error::invariant(format!(
            "'{col}' FixedSizeBinary width must be 16, got {}",
            array.value_length()
        )));
    }
    Ok((idx, array))
}

/// Copy a 16-byte UUID out of a `FixedSizeBinary(16)` array cell.
fn fixed16_at(array: &FixedSizeBinaryArray, row: usize) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(array.value(row));
    bytes
}

/// Collect (column index, name) for every property column, skipping the
/// reserved id/endpoint columns and rejecting engine-managed names.
fn collect_property_columns(
    schema: &arrow_schema::Schema,
    reserved_idx: &[usize],
) -> Result<Vec<(usize, String)>> {
    let mut prop_indices: Vec<(usize, String)> = Vec::with_capacity(schema.fields().len());
    for (idx, field) in schema.fields().iter().enumerate() {
        if reserved_idx.contains(&idx) {
            continue;
        }
        let name = field.name();
        if matches!(name.as_str(), "tombstone" | "lsn") {
            return Err(Error::invariant(format!(
                "parquet column '{name}' collides with engine-managed column"
            )));
        }
        prop_indices.push((idx, name.clone()));
    }
    Ok(prop_indices)
}

/// Build the property map for one row, skipping null cells (absent and
/// explicit-null are equivalent at flush time, and absent saves a field).
fn build_row_properties(
    batch: &RecordBatch,
    prop_indices: &[(usize, String)],
    row: usize,
) -> Result<BTreeMap<String, Value>> {
    let mut properties: BTreeMap<String, Value> = BTreeMap::new();
    for (col_idx, name) in prop_indices {
        let array = batch.column(*col_idx);
        let value = arrow_value_at(array.as_ref(), row, name)?;
        if !value.is_null() {
            properties.insert(name.clone(), value);
        }
    }
    Ok(properties)
}

/// Convert a single Arrow array cell to a [`Value`]. Returns
/// [`Value::Null`] for null cells. Errors on unsupported Arrow types.
fn arrow_value_at(array: &dyn Array, row: usize, col: &str) -> Result<Value> {
    if array.is_null(row) {
        return Ok(Value::Null);
    }
    match array.data_type() {
        ArrowDataType::Boolean => {
            let a = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("downcast Boolean");
            Ok(Value::Bool(a.value(row)))
        }
        ArrowDataType::Int8 => Ok(Value::I64(
            array.as_primitive::<Int8Type>().value(row) as i64
        )),
        ArrowDataType::Int16 => Ok(Value::I64(
            array.as_primitive::<Int16Type>().value(row) as i64
        )),
        ArrowDataType::Int32 => Ok(Value::I64(
            array.as_primitive::<Int32Type>().value(row) as i64
        )),
        ArrowDataType::Int64 => Ok(Value::I64(array.as_primitive::<Int64Type>().value(row))),
        ArrowDataType::UInt8 => Ok(Value::I64(
            array.as_primitive::<UInt8Type>().value(row) as i64
        )),
        ArrowDataType::UInt16 => Ok(Value::I64(
            array.as_primitive::<UInt16Type>().value(row) as i64
        )),
        ArrowDataType::UInt32 => Ok(Value::I64(
            array.as_primitive::<UInt32Type>().value(row) as i64
        )),
        ArrowDataType::Float32 => Ok(Value::F64(
            array.as_primitive::<Float32Type>().value(row) as f64
        )),
        ArrowDataType::Float64 => Ok(Value::F64(array.as_primitive::<Float64Type>().value(row))),
        ArrowDataType::Utf8 => {
            let a = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("downcast Utf8");
            Ok(Value::Str(a.value(row).to_owned()))
        }
        ArrowDataType::LargeUtf8 => {
            let a = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("downcast LargeUtf8");
            Ok(Value::Str(a.value(row).to_owned()))
        }
        ArrowDataType::Binary => {
            let a = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("downcast Binary");
            Ok(Value::Bytes(a.value(row).to_vec()))
        }
        ArrowDataType::LargeBinary => {
            let a = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("downcast LargeBinary");
            Ok(Value::Bytes(a.value(row).to_vec()))
        }
        ArrowDataType::Date32 => Ok(Value::I64(
            array.as_primitive::<Date32Type>().value(row) as i64
        )),
        ArrowDataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => Ok(Value::I64(
            array.as_primitive::<TimestampMicrosecondType>().value(row),
        )),
        ArrowDataType::FixedSizeList(field, dim) => {
            if !matches!(field.data_type(), ArrowDataType::Float32) {
                return Err(Error::invariant(format!(
                    "column '{col}' FixedSizeList element must be Float32, got {:?}",
                    field.data_type()
                )));
            }
            let a = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("downcast FixedSizeList");
            let cell = a.value(row);
            let floats = cell
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| {
                    Error::invariant(format!(
                        "column '{col}' FixedSizeList values must be Float32"
                    ))
                })?;
            let mut v = Vec::with_capacity(*dim as usize);
            for i in 0..floats.len() {
                v.push(floats.value(i));
            }
            Ok(Value::Vec(v))
        }
        other => Err(Error::invariant(format!(
            "column '{col}': unsupported Arrow type {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use namidb_core::NamespaceId;
    use parquet::arrow::ArrowWriter;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::paths::NamespacePaths;

    fn synth_id(i: u64) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[8..].copy_from_slice(&i.to_be_bytes());
        bytes
    }

    fn write_synth_parquet(path: &Path, n_rows: usize) {
        let schema = Schema::new(vec![
            Field::new("node_id", DataType::FixedSizeBinary(16), false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, true),
        ]);

        let ids: Vec<[u8; 16]> = (0..n_rows as u64).map(synth_id).collect();
        let id_array =
            FixedSizeBinaryArray::try_from_iter(ids.iter().map(|b| b.as_slice())).unwrap();
        let names: Vec<String> = (0..n_rows).map(|i| format!("user-{i}")).collect();
        let name_array = StringArray::from(names);
        let ages: Vec<i32> = (0..n_rows as i32).map(|i| i % 100).collect();
        let age_array = Int32Array::from(ages);

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(id_array),
                Arc::new(name_array),
                Arc::new(age_array),
            ],
        )
        .unwrap();

        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[tokio::test]
    async fn loads_small_parquet_through_writer_session() {
        let parquet = NamedTempFile::new().unwrap();
        write_synth_parquet(parquet.path(), 1000);

        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let paths = NamespacePaths::new("test", NamespaceId::new("pq-load").unwrap());
        let mut writer = WriterSession::open(store, paths).await.unwrap();

        let outcome = load_nodes(parquet.path(), &mut writer, "Person", 100)
            .await
            .unwrap();
        // Commit any tail before checking the count.
        writer.commit_batch().await.unwrap();

        assert_eq!(outcome.rows_loaded, 1000);
        assert_eq!(outcome.commit_batches, 10);
    }

    fn write_synth_edges_parquet(path: &Path, pairs: &[(u64, u64)]) {
        let schema = Schema::new(vec![
            Field::new("src", DataType::FixedSizeBinary(16), false),
            Field::new("dst", DataType::FixedSizeBinary(16), false),
            Field::new("weight", DataType::Int32, true),
        ]);
        let srcs: Vec<[u8; 16]> = pairs.iter().map(|(s, _)| synth_id(*s)).collect();
        let dsts: Vec<[u8; 16]> = pairs.iter().map(|(_, d)| synth_id(*d)).collect();
        let src_array =
            FixedSizeBinaryArray::try_from_iter(srcs.iter().map(|b| b.as_slice())).unwrap();
        let dst_array =
            FixedSizeBinaryArray::try_from_iter(dsts.iter().map(|b| b.as_slice())).unwrap();
        let weights = Int32Array::from(pairs.iter().map(|(s, _)| *s as i32).collect::<Vec<_>>());

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(src_array), Arc::new(dst_array), Arc::new(weights)],
        )
        .unwrap();

        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[tokio::test]
    async fn loads_edges_parquet_through_writer_session() {
        let parquet = NamedTempFile::new().unwrap();
        let pairs = [(0u64, 1u64), (1, 2), (2, 0)];
        write_synth_edges_parquet(parquet.path(), &pairs);

        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let paths = NamespacePaths::new("test", NamespaceId::new("pq-load-edges").unwrap());
        let mut writer = WriterSession::open(store, paths).await.unwrap();

        let outcome = load_edges(parquet.path(), &mut writer, "KNOWS", 0)
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();
        assert_eq!(outcome.rows_loaded, 3);

        // The first pair (id 0 -> id 1) is traversable as a KNOWS edge.
        let snap = writer.snapshot();
        let src0 = NodeId::from_uuid(Uuid::from_bytes(synth_id(0)));
        let view = snap.out_edges("KNOWS", src0).await.unwrap();
        assert_eq!(view.edges.len(), 1);
    }
}
