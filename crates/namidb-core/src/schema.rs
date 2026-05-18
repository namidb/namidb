//! Property graph schema primitives.
//!
//! NamiDB is schemaless-by-default but tracks every shape it observes and
//! exposes that shape as a [`Schema`]. Users can also pre-declare labels and
//! edge types when they want stricter validation; this also lets the
//! optimizer plan more aggressively.

use std::collections::BTreeMap;

use arrow_schema::{DataType as ArrowDataType, Field};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A subset of Arrow's logical type system that maps cleanly to Cypher and to
/// JSON ingest.
///
/// We deliberately do not expose every Arrow type; we add things as the engine
/// needs them. Anything more exotic is stored as `Bytes` or `Json` and decoded
/// at the query layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
 Bool,
 Int32,
 Int64,
 Float32,
 Float64,
 Utf8,
 LargeUtf8,
 Binary,
 Date32,
 /// Timestamp in microseconds since the Unix epoch, UTC.
 TimestampMicrosUtc,
 /// Fixed-size float vector (used for embeddings).
 FloatVector {
 dim: u32,
 },
 /// Catch-all for JSON-shaped values that the engine still needs to
 /// represent as one Arrow column.
 Json,
}

impl DataType {
 /// Convert into an Arrow [`ArrowDataType`].
 pub fn to_arrow(&self) -> ArrowDataType {
 match self {
 DataType::Bool => ArrowDataType::Boolean,
 DataType::Int32 => ArrowDataType::Int32,
 DataType::Int64 => ArrowDataType::Int64,
 DataType::Float32 => ArrowDataType::Float32,
 DataType::Float64 => ArrowDataType::Float64,
 DataType::Utf8 => ArrowDataType::Utf8,
 DataType::LargeUtf8 => ArrowDataType::LargeUtf8,
 DataType::Binary => ArrowDataType::Binary,
 DataType::Date32 => ArrowDataType::Date32,
 DataType::TimestampMicrosUtc => {
 ArrowDataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into()))
 }
 DataType::FloatVector { dim } => ArrowDataType::FixedSizeList(
 std::sync::Arc::new(Field::new("item", ArrowDataType::Float32, false)),
 *dim as i32,
 ),
 DataType::Json => ArrowDataType::Utf8,
 }
 }
}

/// Definition of a property on a node label or edge type.
///
/// Property names share a namespace with engine-managed columns in the
/// on-disk SST representation (see [RFC-002](../../../docs/rfc/002-sst-format.md)
/// §2.1). The constructor rejects names that would collide:
///
/// - names starting with `prop_` (would double-prefix on disk),
/// - names starting with `__` (engine-private namespace),
/// - the bare names `node_id`, `tombstone`, `lsn`.
///
/// Construct with [`PropertyDef::new`] to get validation; the serde path
/// also revalidates on deserialise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "PropertyDefRepr", into = "PropertyDefRepr")]
pub struct PropertyDef {
 pub name: String,
 pub data_type: DataType,
 pub nullable: bool,
}

/// Wire-level representation used only by serde. The public type validates
/// on construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PropertyDefRepr {
 name: String,
 data_type: DataType,
 #[serde(default)]
 nullable: bool,
}

impl TryFrom<PropertyDefRepr> for PropertyDef {
 type Error = Error;
 fn try_from(r: PropertyDefRepr) -> Result<Self> {
 PropertyDef::new(r.name, r.data_type, r.nullable)
 }
}

impl From<PropertyDef> for PropertyDefRepr {
 fn from(p: PropertyDef) -> Self {
 PropertyDefRepr {
 name: p.name,
 data_type: p.data_type,
 nullable: p.nullable,
 }
 }
}

/// Reserved column names that cannot be used for declared properties.
/// Kept exported so other crates can share the canonical list.
pub const RESERVED_PROPERTY_NAMES: &[&str] = &["node_id", "tombstone", "lsn"];

impl PropertyDef {
 /// Construct a validated [`PropertyDef`].
 ///
 /// Returns [`Error::Schema`] if `name` collides with any reserved column
 /// or with the `prop_`/`__` prefixes the SST layer manages.
 pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Result<Self> {
 let name = name.into();
 validate_property_name(&name)?;
 Ok(Self {
 name,
 data_type,
 nullable,
 })
 }

 pub fn to_arrow_field(&self) -> Field {
 Field::new(&self.name, self.data_type.to_arrow(), self.nullable)
 }
}

fn validate_property_name(name: &str) -> Result<()> {
 if name.is_empty() {
 return Err(Error::schema("property name must be non-empty"));
 }
 if name.starts_with("__") {
 return Err(Error::schema(format!(
 "property name '{name}' uses the reserved '__' prefix"
 )));
 }
 if name.starts_with("prop_") {
 return Err(Error::schema(format!(
 "property name '{name}' uses the reserved 'prop_' prefix"
 )));
 }
 if RESERVED_PROPERTY_NAMES.contains(&name) {
 return Err(Error::schema(format!(
 "property name '{name}' collides with an engine-managed column"
 )));
 }
 Ok(())
}

/// Definition of a node label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelDef {
 pub name: String,
 pub properties: Vec<PropertyDef>,
}

/// Definition of a directed edge type between two labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeTypeDef {
 pub name: String,
 pub src_label: String,
 pub dst_label: String,
 #[serde(default)]
 pub properties: Vec<PropertyDef>,
}

/// Logical schema for a graph.
///
/// `version` is monotonic per-namespace. Each schema-altering manifest commit
/// bumps it; queries pin their schema view to the manifest they read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Schema {
 pub version: u64,
 pub labels: BTreeMap<String, LabelDef>,
 pub edge_types: BTreeMap<String, EdgeTypeDef>,
}

impl Schema {
 pub fn empty() -> Self {
 Self::default()
 }

 pub fn label(&self, name: &str) -> Option<&LabelDef> {
 self.labels.get(name)
 }
 pub fn edge_type(&self, name: &str) -> Option<&EdgeTypeDef> {
 self.edge_types.get(name)
 }

 pub fn builder() -> SchemaBuilder {
 SchemaBuilder::new()
 }
}

/// Fluent builder for [`Schema`].
#[derive(Debug, Default)]
pub struct SchemaBuilder {
 schema: Schema,
}

impl SchemaBuilder {
 pub fn new() -> Self {
 Self::default()
 }

 pub fn version(mut self, v: u64) -> Self {
 self.schema.version = v;
 self
 }

 pub fn label(mut self, label: LabelDef) -> Result<Self> {
 if self.schema.labels.contains_key(&label.name) {
 return Err(Error::schema(format!(
 "label '{}' declared twice",
 label.name
 )));
 }
 self.schema.labels.insert(label.name.clone(), label);
 Ok(self)
 }

 pub fn edge_type(mut self, edge: EdgeTypeDef) -> Result<Self> {
 if self.schema.edge_types.contains_key(&edge.name) {
 return Err(Error::schema(format!(
 "edge type '{}' declared twice",
 edge.name
 )));
 }
 if !self.schema.labels.contains_key(&edge.src_label) {
 return Err(Error::schema(format!(
 "edge '{}': src_label '{}' is not declared",
 edge.name, edge.src_label
 )));
 }
 if !self.schema.labels.contains_key(&edge.dst_label) {
 return Err(Error::schema(format!(
 "edge '{}': dst_label '{}' is not declared",
 edge.name, edge.dst_label
 )));
 }
 self.schema.edge_types.insert(edge.name.clone(), edge);
 Ok(self)
 }

 pub fn build(self) -> Schema {
 self.schema
 }
}

#[cfg(test)]
mod tests {
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

 #[test]
 fn schema_builder_happy_path() {
 let schema = Schema::builder()
 .label(person_label())
 .unwrap()
 .edge_type(EdgeTypeDef {
 name: "KNOWS".into(),
 src_label: "Person".into(),
 dst_label: "Person".into(),
 properties: vec![],
 })
 .unwrap()
 .build();
 assert!(schema.label("Person").is_some());
 assert_eq!(schema.edge_type("KNOWS").unwrap().src_label, "Person");
 }

 #[test]
 fn schema_rejects_dangling_edge() {
 let err = Schema::builder()
 .edge_type(EdgeTypeDef {
 name: "KNOWS".into(),
 src_label: "Ghost".into(),
 dst_label: "Ghost".into(),
 properties: vec![],
 })
 .unwrap_err();
 match err {
 Error::Schema(msg) => assert!(msg.contains("not declared")),
 other => panic!("unexpected error: {other:?}"),
 }
 }

 #[test]
 fn schema_round_trips_through_json() {
 let schema = Schema::builder().label(person_label()).unwrap().build();
 let serialized = serde_json::to_string(&schema).unwrap();
 let round: Schema = serde_json::from_str(&serialized).unwrap();
 assert_eq!(schema, round);
 }

 #[test]
 fn property_def_maps_to_arrow_field() {
 let p = PropertyDef::new("embedding", DataType::FloatVector { dim: 1536 }, true).unwrap();
 let f = p.to_arrow_field();
 assert!(matches!(
 f.data_type(),
 ArrowDataType::FixedSizeList(_, 1536)
 ));
 assert!(f.is_nullable());
 }

 #[test]
 fn property_def_rejects_reserved_names() {
 for bad in ["node_id", "tombstone", "lsn"] {
 let err = PropertyDef::new(bad, DataType::Utf8, false).unwrap_err();
 assert!(matches!(err, Error::Schema(_)), "{bad}: {err:?}");
 }
 }

 #[test]
 fn property_def_rejects_reserved_prefixes() {
 for bad in [
 "__overflow_json",
 "__schema_version",
 "__anything",
 "prop_age",
 ] {
 let err = PropertyDef::new(bad, DataType::Utf8, false).unwrap_err();
 assert!(matches!(err, Error::Schema(_)), "{bad}: {err:?}");
 }
 }

 #[test]
 fn property_def_rejects_empty_name() {
 let err = PropertyDef::new("", DataType::Utf8, false).unwrap_err();
 assert!(matches!(err, Error::Schema(_)));
 }

 #[test]
 fn property_def_json_revalidates() {
 let bad_json = r#"{"name":"node_id","data_type":"Utf8","nullable":false}"#;
 let res: std::result::Result<PropertyDef, _> = serde_json::from_str(bad_json);
 assert!(res.is_err(), "serde must reject reserved property names");
 }
}
