//! Bulk-load CSVs emitted by `dataset.rs` into a NamiDB namespace.
//!
//! The loader opens a `WriterSession`, ingests rows in batches via
//! `upsert_node` / `upsert_edge`, and calls `flush` at the end to
//! materialise SSTs. Schema is fixed and matches the LDBC-shaped
//! subset described in `dataset.rs`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, SchemaBuilder, Value};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

/// Construct the fixed LDBC-shaped schema this benchmark works against.
pub fn schema() -> Schema {
 SchemaBuilder::new()
 .label(LabelDef {
 name: "Person".into(),
 properties: vec![
 PropertyDef::new("firstName", DataType::Utf8, true).unwrap(),
 PropertyDef::new("lastName", DataType::Utf8, true).unwrap(),
 PropertyDef::new("age", DataType::Int64, true).unwrap(),
 PropertyDef::new("creationDate", DataType::Int64, true).unwrap(),
 ],
 })
 .unwrap()
 .label(LabelDef {
 name: "Post".into(),
 properties: vec![
 PropertyDef::new("content", DataType::Utf8, true).unwrap(),
 PropertyDef::new("creationDate", DataType::Int64, true).unwrap(),
 PropertyDef::new("length", DataType::Int64, true).unwrap(),
 ],
 })
 .unwrap()
 .label(LabelDef {
 name: "Comment".into(),
 properties: vec![
 PropertyDef::new("content", DataType::Utf8, true).unwrap(),
 PropertyDef::new("creationDate", DataType::Int64, true).unwrap(),
 PropertyDef::new("length", DataType::Int64, true).unwrap(),
 ],
 })
 .unwrap()
 .edge_type(EdgeTypeDef {
 name: "KNOWS".into(),
 src_label: "Person".into(),
 dst_label: "Person".into(),
 properties: vec![PropertyDef::new("since", DataType::Int64, true).unwrap()],
 })
 .unwrap()
 .edge_type(EdgeTypeDef {
 name: "HAS_CREATOR".into(),
 src_label: "Post".into(),
 dst_label: "Person".into(),
 properties: vec![],
 })
 .unwrap()
 .edge_type(EdgeTypeDef {
 name: "LIKES".into(),
 src_label: "Person".into(),
 dst_label: "Post".into(),
 properties: vec![PropertyDef::new("creationDate", DataType::Int64, true).unwrap()],
 })
 .unwrap()
 .edge_type(EdgeTypeDef {
 name: "REPLY_OF".into(),
 src_label: "Comment".into(),
 dst_label: "Post".into(),
 properties: vec![],
 })
 .unwrap()
 .build()
}

/// Open an in-memory object store and bulk-load the dataset at
/// `csv_dir`. Returns the live `WriterSession` (already flushed) so the
/// runner can pull a `Snapshot` from it.
pub async fn load_into_in_memory(csv_dir: &Path, namespace: &str) -> Result<WriterSession> {
 let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
 let paths = NamespacePaths::new("tenants", NamespaceId::new(namespace).unwrap());
 let mut writer = WriterSession::open(store, paths)
 .await
 .context("open writer session")?;
 bulk_load(&mut writer, csv_dir).await?;
 Ok(writer)
}

/// Load every file from `csv_dir` into `writer` and flush.
pub async fn bulk_load(writer: &mut WriterSession, csv_dir: &Path) -> Result<()> {
 load_persons(writer, &csv_dir.join("persons.csv"))?;
 load_posts(writer, &csv_dir.join("posts.csv"))?;
 load_comments(writer, &csv_dir.join("comments.csv"))?;
 load_knows(writer, &csv_dir.join("knows.csv"))?;
 load_has_creator(writer, &csv_dir.join("has_creator.csv"))?;
 load_likes(writer, &csv_dir.join("likes.csv"))?;
 load_reply_of(writer, &csv_dir.join("reply_of.csv"))?;
 writer.flush(schema()).await.context("flush")?;
 Ok(())
}

fn load_persons(writer: &mut WriterSession, path: &Path) -> Result<()> {
 let reader = BufReader::new(File::open(path)?);
 for (i, line) in reader.lines().enumerate() {
 if i == 0 {
 continue;
 }
 let line = line?;
 let parts: Vec<&str> = line.split('|').collect();
 if parts.len() < 5 {
 continue;
 }
 let id = parse_node_id(parts[0])?;
 let mut props: BTreeMap<String, Value> = BTreeMap::new();
 props.insert("firstName".into(), Value::Str(parts[1].into()));
 props.insert("lastName".into(), Value::Str(parts[2].into()));
 props.insert("age".into(), Value::I64(parts[3].parse::<i64>()?));
 props.insert("creationDate".into(), Value::I64(parts[4].parse::<i64>()?));
 writer.upsert_node(
 "Person",
 id,
 &NodeWriteRecord {
 properties: props,
 schema_version: 1,
 },
 )?;
 }
 Ok(())
}

fn load_posts(writer: &mut WriterSession, path: &Path) -> Result<()> {
 let reader = BufReader::new(File::open(path)?);
 for (i, line) in reader.lines().enumerate() {
 if i == 0 {
 continue;
 }
 let line = line?;
 let parts: Vec<&str> = line.split('|').collect();
 if parts.len() < 4 {
 continue;
 }
 let id = parse_node_id(parts[0])?;
 let mut props: BTreeMap<String, Value> = BTreeMap::new();
 props.insert("content".into(), Value::Str(parts[1].into()));
 props.insert("creationDate".into(), Value::I64(parts[2].parse::<i64>()?));
 props.insert("length".into(), Value::I64(parts[3].parse::<i64>()?));
 writer.upsert_node(
 "Post",
 id,
 &NodeWriteRecord {
 properties: props,
 schema_version: 1,
 },
 )?;
 }
 Ok(())
}

fn load_comments(writer: &mut WriterSession, path: &Path) -> Result<()> {
 let reader = BufReader::new(File::open(path)?);
 for (i, line) in reader.lines().enumerate() {
 if i == 0 {
 continue;
 }
 let line = line?;
 let parts: Vec<&str> = line.split('|').collect();
 if parts.len() < 4 {
 continue;
 }
 let id = parse_node_id(parts[0])?;
 let mut props: BTreeMap<String, Value> = BTreeMap::new();
 props.insert("content".into(), Value::Str(parts[1].into()));
 props.insert("creationDate".into(), Value::I64(parts[2].parse::<i64>()?));
 props.insert("length".into(), Value::I64(parts[3].parse::<i64>()?));
 writer.upsert_node(
 "Comment",
 id,
 &NodeWriteRecord {
 properties: props,
 schema_version: 1,
 },
 )?;
 }
 Ok(())
}

fn load_knows(writer: &mut WriterSession, path: &Path) -> Result<()> {
 let reader = BufReader::new(File::open(path)?);
 for (i, line) in reader.lines().enumerate() {
 if i == 0 {
 continue;
 }
 let line = line?;
 let parts: Vec<&str> = line.split('|').collect();
 if parts.len() < 3 {
 continue;
 }
 let src = parse_node_id(parts[0])?;
 let dst = parse_node_id(parts[1])?;
 let mut props: BTreeMap<String, Value> = BTreeMap::new();
 props.insert("since".into(), Value::I64(parts[2].parse::<i64>()?));
 writer.upsert_edge(
 "KNOWS",
 src,
 dst,
 &EdgeWriteRecord {
 properties: props,
 schema_version: 1,
 },
 )?;
 }
 Ok(())
}

fn load_has_creator(writer: &mut WriterSession, path: &Path) -> Result<()> {
 let reader = BufReader::new(File::open(path)?);
 for (i, line) in reader.lines().enumerate() {
 if i == 0 {
 continue;
 }
 let line = line?;
 let parts: Vec<&str> = line.split('|').collect();
 if parts.len() < 2 {
 continue;
 }
 let src = parse_node_id(parts[0])?;
 let dst = parse_node_id(parts[1])?;
 writer.upsert_edge(
 "HAS_CREATOR",
 src,
 dst,
 &EdgeWriteRecord {
 properties: BTreeMap::new(),
 schema_version: 1,
 },
 )?;
 }
 Ok(())
}

fn load_likes(writer: &mut WriterSession, path: &Path) -> Result<()> {
 let reader = BufReader::new(File::open(path)?);
 for (i, line) in reader.lines().enumerate() {
 if i == 0 {
 continue;
 }
 let line = line?;
 let parts: Vec<&str> = line.split('|').collect();
 if parts.len() < 3 {
 continue;
 }
 let src = parse_node_id(parts[0])?;
 let dst = parse_node_id(parts[1])?;
 let mut props: BTreeMap<String, Value> = BTreeMap::new();
 props.insert("creationDate".into(), Value::I64(parts[2].parse::<i64>()?));
 writer.upsert_edge(
 "LIKES",
 src,
 dst,
 &EdgeWriteRecord {
 properties: props,
 schema_version: 1,
 },
 )?;
 }
 Ok(())
}

fn load_reply_of(writer: &mut WriterSession, path: &Path) -> Result<()> {
 let reader = BufReader::new(File::open(path)?);
 for (i, line) in reader.lines().enumerate() {
 if i == 0 {
 continue;
 }
 let line = line?;
 let parts: Vec<&str> = line.split('|').collect();
 if parts.len() < 2 {
 continue;
 }
 let src = parse_node_id(parts[0])?;
 let dst = parse_node_id(parts[1])?;
 writer.upsert_edge(
 "REPLY_OF",
 src,
 dst,
 &EdgeWriteRecord {
 properties: BTreeMap::new(),
 schema_version: 1,
 },
 )?;
 }
 Ok(())
}

/// Parse the 32-hex-char id format `dataset::encode_id` produces back
/// into a `NodeId`.
fn parse_node_id(s: &str) -> Result<NodeId> {
 let s = s.trim();
 if s.len() != 32 {
 anyhow::bail!("expected 32-char id, got {} chars: {s}", s.len());
 }
 let mut bytes = [0u8; 16];
 for i in 0..16 {
 bytes[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
 }
 Ok(NodeId::from_uuid(uuid::Uuid::from_bytes(bytes)))
}
