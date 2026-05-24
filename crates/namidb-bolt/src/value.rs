//! PackStream value model.
//!
//! Every value the wire carries fits in [`Value`]. The codec lives in
//! [`crate::codec`] and turns this enum into bytes (and back).
//!
//! Structs (Node, Relationship, Path, Date, …) are represented as
//! `Value::Struct { tag, fields }` rather than dedicated variants so
//! the codec stays simple. Higher-level code that wants to operate on
//! a `Node` reaches for [`Node::from_struct`] / [`Node::to_struct`].

use std::collections::BTreeMap;

use crate::error::{BoltError, Result};

/// One PackStream value. Maps 1:1 to the spec's "value" production.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
    Struct { tag: u8, fields: Vec<Value> },
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "Null",
            Value::Bool(_) => "Bool",
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::String(_) => "String",
            Value::Bytes(_) => "Bytes",
            Value::List(_) => "List",
            Value::Map(_) => "Map",
            Value::Struct { .. } => "Struct",
        }
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Value::String(s) => Ok(s),
            other => Err(BoltError::Protocol(format!(
                "expected String, got {}",
                other.type_name()
            ))),
        }
    }

    pub fn as_map(&self) -> Result<&BTreeMap<String, Value>> {
        match self {
            Value::Map(m) => Ok(m),
            other => Err(BoltError::Protocol(format!(
                "expected Map, got {}",
                other.type_name()
            ))),
        }
    }

    pub fn into_map(self) -> Result<BTreeMap<String, Value>> {
        match self {
            Value::Map(m) => Ok(m),
            other => Err(BoltError::Protocol(format!(
                "expected Map, got {}",
                other.type_name()
            ))),
        }
    }

    pub fn as_int(&self) -> Result<i64> {
        match self {
            Value::Int(n) => Ok(*n),
            other => Err(BoltError::Protocol(format!(
                "expected Int, got {}",
                other.type_name()
            ))),
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}
impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Int(n)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

/// Struct tags the spec assigns. Used by both encode and decode.
pub mod struct_tag {
    // Response messages
    pub const SUCCESS: u8 = 0x70;
    pub const RECORD: u8 = 0x71;
    pub const IGNORED: u8 = 0x7E;
    pub const FAILURE: u8 = 0x7F;

    // Request messages (v4.4 + v5)
    pub const HELLO: u8 = 0x01;
    pub const GOODBYE: u8 = 0x02;
    pub const RESET: u8 = 0x0F;
    pub const RUN: u8 = 0x10;
    pub const BEGIN: u8 = 0x11;
    pub const COMMIT: u8 = 0x12;
    pub const ROLLBACK: u8 = 0x13;
    pub const DISCARD: u8 = 0x2F;
    pub const PULL: u8 = 0x3F;
    pub const ROUTE: u8 = 0x66;
    pub const LOGON: u8 = 0x6A;
    pub const LOGOFF: u8 = 0x6B;
    pub const TELEMETRY: u8 = 0x54;

    // Value structs (per the Bolt 5.x spec)
    pub const NODE: u8 = 0x4E;
    pub const RELATIONSHIP: u8 = 0x52;
    pub const UNBOUND_RELATIONSHIP: u8 = 0x72;
    pub const PATH: u8 = 0x50;
    pub const DATE: u8 = 0x44;
    /// LocalDateTime: 2 fields (seconds, nanoseconds since epoch, no zone).
    pub const LOCAL_DATETIME: u8 = 0x64;
    /// DateTime (Bolt 5+, UTC seconds + nanos + tz offset seconds).
    pub const DATETIME: u8 = 0x49;
    /// DateTimeZoneId (Bolt 5+, UTC seconds + nanos + zone id string).
    pub const DATETIME_ZONE_ID: u8 = 0x69;
    /// DateTime (Bolt 4 legacy, seconds + nanos + tz offset seconds).
    pub const DATETIME_LEGACY: u8 = 0x46;
    /// Time: 2 fields (nanoseconds of day, tz offset seconds).
    pub const TIME: u8 = 0x54;
    /// LocalTime: 1 field (nanoseconds of day).
    pub const LOCAL_TIME: u8 = 0x74;
    pub const DURATION: u8 = 0x45;
}

/// A decoded Bolt Node struct (tag 0x4E). v5 layout: `{ id, labels,
/// properties, element_id }`. v4 has the same first three fields and
/// no `element_id`; the codec writes whichever shape the negotiated
/// version asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: i64,
    pub labels: Vec<String>,
    pub properties: BTreeMap<String, Value>,
    pub element_id: Option<String>,
}

impl Node {
    pub fn to_struct(self, with_element_id: bool) -> Value {
        let mut fields = vec![
            Value::Int(self.id),
            Value::List(self.labels.into_iter().map(Value::String).collect()),
            Value::Map(self.properties),
        ];
        if with_element_id {
            fields.push(Value::String(self.element_id.unwrap_or_default()));
        }
        Value::Struct {
            tag: struct_tag::NODE,
            fields,
        }
    }
}

/// A decoded Bolt Relationship struct (tag 0x52). v5 layout:
/// `{ id, start_id, end_id, type, properties, element_id,
///   start_element_id, end_element_id }`.
#[derive(Debug, Clone, PartialEq)]
pub struct Relationship {
    pub id: i64,
    pub start_id: i64,
    pub end_id: i64,
    pub rel_type: String,
    pub properties: BTreeMap<String, Value>,
    pub element_id: Option<String>,
    pub start_element_id: Option<String>,
    pub end_element_id: Option<String>,
}

impl Relationship {
    pub fn to_struct(self, with_element_id: bool) -> Value {
        let mut fields = vec![
            Value::Int(self.id),
            Value::Int(self.start_id),
            Value::Int(self.end_id),
            Value::String(self.rel_type),
            Value::Map(self.properties),
        ];
        if with_element_id {
            fields.push(Value::String(self.element_id.unwrap_or_default()));
            fields.push(Value::String(self.start_element_id.unwrap_or_default()));
            fields.push(Value::String(self.end_element_id.unwrap_or_default()));
        }
        Value::Struct {
            tag: struct_tag::RELATIONSHIP,
            fields,
        }
    }
}
