//! Owned property values used by ingest APIs.
//!
//! On the hot path we always go through Arrow batches; [`Value`] only exists
//! to give a friendly Rust-native shape to ad-hoc insertions and to feed the
//! ingest path before columnarisation.

use serde::{Deserialize, Serialize};

/// A single property value. Loose, JSON-ish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
 Null,
 Bool(bool),
 I64(i64),
 F64(f64),
 Str(String),
 Bytes(#[serde(with = "serde_bytes")] Vec<u8>),
 Vec(Vec<f32>),
}

impl Value {
 pub fn is_null(&self) -> bool {
 matches!(self, Value::Null)
 }
}

impl From<bool> for Value {
 fn from(v: bool) -> Self {
 Value::Bool(v)
 }
}
impl From<i32> for Value {
 fn from(v: i32) -> Self {
 Value::I64(v as i64)
 }
}
impl From<i64> for Value {
 fn from(v: i64) -> Self {
 Value::I64(v)
 }
}
impl From<f32> for Value {
 fn from(v: f32) -> Self {
 Value::F64(v as f64)
 }
}
impl From<f64> for Value {
 fn from(v: f64) -> Self {
 Value::F64(v)
 }
}
impl From<&str> for Value {
 fn from(v: &str) -> Self {
 Value::Str(v.to_owned())
 }
}
impl From<String> for Value {
 fn from(v: String) -> Self {
 Value::Str(v)
 }
}
impl From<Vec<f32>> for Value {
 fn from(v: Vec<f32>) -> Self {
 Value::Vec(v)
 }
}

// Lightweight bytes shim so we don't need to pull serde_bytes as a real dep.
mod serde_bytes {
 use serde::{Deserialize, Deserializer, Serialize, Serializer};

 pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
 v.serialize(s)
 }
 pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
 Vec::<u8>::deserialize(d)
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn value_from_primitives() {
 assert_eq!(Value::from(true), Value::Bool(true));
 assert_eq!(Value::from(42i32), Value::I64(42));
 assert_eq!(Value::from("hi"), Value::Str("hi".into()));
 assert_eq!(Value::from(vec![1.0f32, 2.0]), Value::Vec(vec![1.0, 2.0]));
 }

 #[test]
 fn value_serde_roundtrip() {
 let v = Value::Str("hello".into());
 let s = serde_json::to_string(&v).unwrap();
 let back: Value = serde_json::from_str(&s).unwrap();
 assert_eq!(v, back);
 }
}
