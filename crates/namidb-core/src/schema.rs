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
use crate::id::LabelId;

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
    /// Fixed-size int8-quantized vector (used for embeddings, 4x smaller than
    /// `FloatVector`). Stored as one `FixedSizeBinary(4 + dim)` column: the
    /// first 4 bytes are the per-vector f32 scale (little-endian), the next
    /// `dim` bytes are the int8 codes, with `x_i ≈ code_i * scale` (see
    /// [`crate::quantize`]). One property = one column, like `FloatVector`.
    Int8Vector {
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
            // 4-byte f32 scale prefix + `dim` int8 code bytes, in one column.
            DataType::Int8Vector { dim } => ArrowDataType::FixedSizeBinary(4 + *dim as i32),
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
    /// When `true`, the planner is allowed to assume *at most one* node
    /// of the parent label has any given value for this property, which
    /// lets `MATCH (a:Label {prop: literal})` lower to a point-lookup
    /// instead of a full label scan + filter. The engine does NOT
    /// enforce uniqueness on write — it's a planner hint, equivalent
    /// to Kuzu's `PRIMARY KEY` or Neo4j's `UNIQUENESS CONSTRAINT`
    /// without the enforcement. Caller takes responsibility.
    ///
    /// Defaults to `false` so existing schemas / manifests load
    /// unchanged via the `#[serde(default)]` on the wire repr.
    pub unique: bool,
    /// When `true`, the flush layer emits a secondary equality-index
    /// sidecar for this property (value -> the node ids carrying it), so
    /// `MATCH (a:Label {prop: literal})` can resolve through the index
    /// instead of a full label scan + filter, even when the value is NOT
    /// unique. Orthogonal to `unique`: a property may be `indexed` without
    /// being unique. Like `unique`, it is a planner/storage hint.
    ///
    /// Defaults to `false`; the wire repr carries `#[serde(default)]` so
    /// existing manifests load unchanged.
    pub indexed: bool,
}

/// Wire-level representation used only by serde. The public type validates
/// on construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PropertyDefRepr {
    name: String,
    data_type: DataType,
    #[serde(default)]
    nullable: bool,
    #[serde(default)]
    unique: bool,
    #[serde(default)]
    indexed: bool,
}

impl TryFrom<PropertyDefRepr> for PropertyDef {
    type Error = Error;
    fn try_from(r: PropertyDefRepr) -> Result<Self> {
        let mut p = PropertyDef::new(r.name, r.data_type, r.nullable)?;
        p.unique = r.unique;
        p.indexed = r.indexed;
        Ok(p)
    }
}

impl From<PropertyDef> for PropertyDefRepr {
    fn from(p: PropertyDef) -> Self {
        PropertyDefRepr {
            name: p.name,
            data_type: p.data_type,
            nullable: p.nullable,
            unique: p.unique,
            indexed: p.indexed,
        }
    }
}

/// Reserved column names that cannot be used for declared properties.
/// Kept exported so other crates can share the canonical list.
pub const RESERVED_PROPERTY_NAMES: &[&str] = &["node_id", "tombstone", "lsn"];

/// Whether `name` is reserved by the engine and so cannot be stored as a
/// property: the `__`/`prop_` prefixes the SST layer manages, or one of the
/// bare names in [`RESERVED_PROPERTY_NAMES`]. Exported so loaders that build
/// property maps without going through [`PropertyDef::new`] (e.g. the markdown
/// vault loader) can drop these without re-deriving the list and drifting from
/// the engine's own `validate_property_name`.
pub fn is_reserved_property_name(name: &str) -> bool {
    name.starts_with("__") || name.starts_with("prop_") || RESERVED_PROPERTY_NAMES.contains(&name)
}

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
            unique: false,
            indexed: false,
        })
    }

    /// Builder method: declare this property as unique (planner hint —
    /// the engine does NOT enforce uniqueness on write).
    ///
    /// ```ignore
    /// PropertyDef::new("id", DataType::Utf8, true)?.with_unique(true)
    /// ```
    pub fn with_unique(mut self, unique: bool) -> Self {
        self.unique = unique;
        self
    }

    /// Builder method: declare this property as indexed, so the flush layer
    /// emits a secondary equality-index sidecar for it (non-unique).
    ///
    /// ```ignore
    /// PropertyDef::new("city", DataType::Utf8, true)?.with_indexed(true)
    /// ```
    pub fn with_indexed(mut self, indexed: bool) -> Self {
        self.indexed = indexed;
        self
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

/// Bidirectional, append-only mapping between label names and compact
/// [`LabelId`]s.
///
/// A node carries its labels on-row as a set of [`LabelId`]s; this dictionary
/// is the per-namespace source of truth that resolves those ids back to names
/// and back again. Ids are handed out in first-seen order and never change or
/// get reused, so a `LabelId` minted in one manifest commit means the same
/// label in every later one — that stability is what lets the storage layer
/// store labels as a packed `List<UInt32>` rather than repeating strings.
///
/// On the wire the dictionary is just the ordered list of names: a label's id
/// is its index in that list, and the reverse lookup is rebuilt on load.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Vec<String>", into = "Vec<String>")]
pub struct LabelDictionary {
    /// `names[i]` is the label carrying `LabelId(i as u32)`. Append-only.
    names: Vec<String>,
    /// Reverse index (name -> id). Derived from `names`, never serialised.
    ids: BTreeMap<String, LabelId>,
}

impl LabelDictionary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a name to its id, minting a fresh one on first sight.
    ///
    /// Idempotent: interning the same name twice yields the same id.
    pub fn intern(&mut self, name: &str) -> LabelId {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        let id = LabelId(self.names.len() as u32);
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    /// Resolve a name to its id without minting one.
    pub fn id(&self, name: &str) -> Option<LabelId> {
        self.ids.get(name).copied()
    }

    /// Resolve an id back to its label name.
    pub fn name(&self, id: LabelId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Iterate `(id, name)` pairs in id order.
    pub fn iter(&self) -> impl Iterator<Item = (LabelId, &str)> {
        self.names
            .iter()
            .enumerate()
            .map(|(i, n)| (LabelId(i as u32), n.as_str()))
    }
}

impl From<Vec<String>> for LabelDictionary {
    fn from(names: Vec<String>) -> Self {
        // Names persist in id order, so the index is the id. Build the reverse
        // map keeping the first occurrence should a stale list carry a dup.
        let mut ids = BTreeMap::new();
        for (i, name) in names.iter().enumerate() {
            ids.entry(name.clone()).or_insert(LabelId(i as u32));
        }
        LabelDictionary { names, ids }
    }
}

impl From<LabelDictionary> for Vec<String> {
    fn from(dict: LabelDictionary) -> Self {
        dict.names
    }
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

    #[test]
    fn is_reserved_property_name_matches_the_validator() {
        // The exported predicate must agree with `validate_property_name`'s
        // reject set so loaders that use it cannot drift from the engine.
        for reserved in ["node_id", "tombstone", "lsn", "__internal", "prop_x"] {
            assert!(is_reserved_property_name(reserved));
            assert!(validate_property_name(reserved).is_err());
        }
        for ok in ["name", "title", "key", "body", "role"] {
            assert!(!is_reserved_property_name(ok));
            assert!(validate_property_name(ok).is_ok());
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
    fn int8_vector_maps_to_fixed_size_binary() {
        // 4-byte f32 scale prefix + `dim` int8 code bytes, one column.
        let p = PropertyDef::new("emb", DataType::Int8Vector { dim: 256 }, true).unwrap();
        let f = p.to_arrow_field();
        assert!(matches!(f.data_type(), ArrowDataType::FixedSizeBinary(260)));
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

    #[test]
    fn label_dictionary_interns_in_first_seen_order() {
        let mut dict = LabelDictionary::new();
        assert!(dict.is_empty());
        let person = dict.intern("Person");
        let employee = dict.intern("Employee");
        assert_eq!(person, LabelId::new(0));
        assert_eq!(employee, LabelId::new(1));
        // Idempotent: re-interning returns the same id, no new slot.
        assert_eq!(dict.intern("Person"), person);
        assert_eq!(dict.len(), 2);
    }

    #[test]
    fn label_dictionary_resolves_both_directions() {
        let mut dict = LabelDictionary::new();
        let id = dict.intern("Person");
        assert_eq!(dict.id("Person"), Some(id));
        assert_eq!(dict.name(id), Some("Person"));
        assert_eq!(dict.id("Missing"), None);
        assert_eq!(dict.name(LabelId::new(99)), None);
    }

    #[test]
    fn label_dictionary_iterates_in_id_order() {
        let mut dict = LabelDictionary::new();
        dict.intern("A");
        dict.intern("B");
        dict.intern("C");
        let pairs: Vec<_> = dict.iter().collect();
        assert_eq!(
            pairs,
            vec![
                (LabelId::new(0), "A"),
                (LabelId::new(1), "B"),
                (LabelId::new(2), "C"),
            ]
        );
    }

    #[test]
    fn label_dictionary_round_trips_and_keeps_ids_stable() {
        let mut dict = LabelDictionary::new();
        dict.intern("Person");
        dict.intern("Employee");
        dict.intern("Manager");

        // On the wire it's just the ordered name list.
        let json = serde_json::to_string(&dict).unwrap();
        assert_eq!(json, r#"["Person","Employee","Manager"]"#);

        let round: LabelDictionary = serde_json::from_str(&json).unwrap();
        assert_eq!(round, dict);
        // Ids survive the round-trip, and a re-intern of an existing name
        // reuses its id rather than appending.
        assert_eq!(round.id("Employee"), Some(LabelId::new(1)));
        let mut round = round;
        assert_eq!(round.intern("Person"), LabelId::new(0));
        assert_eq!(round.intern("New"), LabelId::new(3));
    }
}
