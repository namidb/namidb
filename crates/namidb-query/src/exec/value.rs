//! Runtime values handled by the executor.
//!
//! `RuntimeValue` is richer than `namidb-core::Value` because Cypher
//! needs lists, maps, nodes and relationships as first-class. See
//! RFC-008 §"Tipo runtime: RuntimeValue".

use std::collections::BTreeMap;

use namidb_core::id::NodeId;
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeView, NodeView};

/// A value produced or consumed by the executor.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    List(Vec<RuntimeValue>),
    Map(BTreeMap<String, RuntimeValue>),
    Node(Box<NodeValue>),
    Rel(Box<RelValue>),
    /// Days since 1970-01-01. Cypher `date()`.
    Date(i32),
    /// Microseconds since 1970-01-01T00:00:00Z. Cypher `datetime()` UTC.
    DateTime(i64),
    /// Raw bytes — carried through but no arithmetic.
    Bytes(Vec<u8>),
    /// Float-vector — carried through but no arithmetic in v0.
    Vector(Vec<f32>),
    /// Alternating sequence `Node, Rel, Node, Rel, ..., Node` produced
    /// by binding a pattern part `p = (a)-[r]->(b)`. RFC-009 §"Path
    /// binding (caso simple)". Variable-length paths land.
    Path(Vec<RuntimeValue>),
}

impl RuntimeValue {
    pub fn is_null(&self) -> bool {
        matches!(self, RuntimeValue::Null)
    }

    /// `true` for `true`-ish boolean. `Null`/anything else returns `None`
    /// (three-valued logic: caller decides whether to treat as `false`
    /// or propagate NULL).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            RuntimeValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Type tag used in error messages (`"INTEGER"`, `"FLOAT"`, `"NODE"`).
    pub fn type_name(&self) -> &'static str {
        match self {
            RuntimeValue::Null => "NULL",
            RuntimeValue::Bool(_) => "BOOLEAN",
            RuntimeValue::Integer(_) => "INTEGER",
            RuntimeValue::Float(_) => "FLOAT",
            RuntimeValue::String(_) => "STRING",
            RuntimeValue::List(_) => "LIST",
            RuntimeValue::Map(_) => "MAP",
            RuntimeValue::Node(_) => "NODE",
            RuntimeValue::Rel(_) => "RELATIONSHIP",
            RuntimeValue::Date(_) => "DATE",
            RuntimeValue::DateTime(_) => "DATETIME",
            RuntimeValue::Bytes(_) => "BYTES",
            RuntimeValue::Vector(_) => "VECTOR",
            RuntimeValue::Path(_) => "PATH",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeValue {
    pub id: NodeId,
    pub label: String,
    pub properties: BTreeMap<String, RuntimeValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelValue {
    pub edge_type: String,
    pub src: NodeId,
    pub dst: NodeId,
    pub properties: BTreeMap<String, RuntimeValue>,
}

impl From<NodeView> for NodeValue {
    fn from(v: NodeView) -> Self {
        // NodeValue is still single-label here; collapse the set to its
        // representative (lowest) label. The NodeValue multi-label flip is a
        // later step in the series.
        let label = v.labels.into_iter().next().unwrap_or_default();
        NodeValue {
            id: v.id,
            label,
            properties: v
                .properties
                .into_iter()
                .map(|(k, val)| (k, RuntimeValue::from(val)))
                .collect(),
        }
    }
}

impl From<EdgeView> for RelValue {
    fn from(v: EdgeView) -> Self {
        RelValue {
            edge_type: v.edge_type,
            src: v.src,
            dst: v.dst,
            properties: v
                .properties
                .into_iter()
                .map(|(k, val)| (k, RuntimeValue::from(val)))
                .collect(),
        }
    }
}

impl From<CoreValue> for RuntimeValue {
    fn from(v: CoreValue) -> Self {
        match v {
            CoreValue::Null => RuntimeValue::Null,
            CoreValue::Bool(b) => RuntimeValue::Bool(b),
            CoreValue::I64(n) => RuntimeValue::Integer(n),
            CoreValue::F64(f) => RuntimeValue::Float(f),
            CoreValue::Str(s) => RuntimeValue::String(s),
            CoreValue::Bytes(b) => RuntimeValue::Bytes(b),
            CoreValue::Vec(v) => RuntimeValue::Vector(v),
            CoreValue::Date(d) => RuntimeValue::Date(d),
            CoreValue::DateTime(m) => RuntimeValue::DateTime(m),
            CoreValue::List(items) => {
                RuntimeValue::List(items.into_iter().map(RuntimeValue::from).collect())
            }
            CoreValue::Map(entries) => RuntimeValue::Map(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, RuntimeValue::from(v)))
                    .collect(),
            ),
        }
    }
}

impl From<bool> for RuntimeValue {
    fn from(b: bool) -> Self {
        RuntimeValue::Bool(b)
    }
}

impl From<i64> for RuntimeValue {
    fn from(n: i64) -> Self {
        RuntimeValue::Integer(n)
    }
}

impl From<i32> for RuntimeValue {
    fn from(n: i32) -> Self {
        RuntimeValue::Integer(n as i64)
    }
}

impl From<f64> for RuntimeValue {
    fn from(f: f64) -> Self {
        RuntimeValue::Float(f)
    }
}

impl From<String> for RuntimeValue {
    fn from(s: String) -> Self {
        RuntimeValue::String(s)
    }
}

impl From<&str> for RuntimeValue {
    fn from(s: &str) -> Self {
        RuntimeValue::String(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_names_are_uppercase() {
        assert_eq!(RuntimeValue::Null.type_name(), "NULL");
        assert_eq!(RuntimeValue::Integer(0).type_name(), "INTEGER");
        assert_eq!(RuntimeValue::String("a".into()).type_name(), "STRING");
    }

    #[test]
    fn from_core_value_passes_through_scalars() {
        assert_eq!(
            RuntimeValue::from(CoreValue::I64(42)),
            RuntimeValue::Integer(42)
        );
        assert_eq!(
            RuntimeValue::from(CoreValue::Str("ok".into())),
            RuntimeValue::String("ok".into())
        );
        assert_eq!(RuntimeValue::from(CoreValue::Null), RuntimeValue::Null);
    }

    #[test]
    fn is_null_only_for_null() {
        assert!(RuntimeValue::Null.is_null());
        assert!(!RuntimeValue::Bool(false).is_null());
        assert!(!RuntimeValue::Integer(0).is_null());
    }

    #[test]
    fn as_bool_distinguishes_null_from_false() {
        assert_eq!(RuntimeValue::Bool(true).as_bool(), Some(true));
        assert_eq!(RuntimeValue::Bool(false).as_bool(), Some(false));
        assert_eq!(RuntimeValue::Null.as_bool(), None);
        assert_eq!(RuntimeValue::Integer(0).as_bool(), None);
    }
}
