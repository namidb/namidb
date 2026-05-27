//! Owned property values used by ingest APIs.
//!
//! On the hot path we always go through Arrow batches; [`Value`] only exists
//! to give a friendly Rust-native shape to ad-hoc insertions and to feed the
//! ingest path before columnarisation.

use std::collections::BTreeMap;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// JSON object key used to tag a [`Value::Date`] when it round-trips
/// through `__overflow_json`. Picked to be reserved-by-convention so a
/// user property called `"$date"` doesn't collide with the wire form.
const TAG_DATE: &str = "$date";
/// Same idea for [`Value::DateTime`].
const TAG_DATETIME: &str = "$datetime";
/// Tag for `Value::List(...)` — keeps the array shape from being
/// silently re-decoded as `Vec<f32>` and lets the deserializer recover
/// the heterogeneous element types.
const TAG_LIST: &str = "$list";
/// Tag for `Value::Map(...)` — disambiguates from the `{}` shape that
/// Date / DateTime already use.
const TAG_MAP: &str = "$map";
/// Tag for `Value::Bytes(...)`. Without this, `serde_json` encodes
/// bytes as an untagged number array and the deserialiser's
/// `visit_seq` cannot tell them apart from a `Vec<f32>` vector,
/// silently turning `b"\x00\x01\x02"` into `Vec([0.0, 1.0, 2.0])`
/// on the way back. The tagged form `{"$bytes": [0, 1, 2]}` keeps
/// the type stable through `__overflow_json`.
const TAG_BYTES: &str = "$bytes";

/// A single property value. Loose, JSON-ish.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
    Bytes(Vec<u8>),
    Vec(Vec<f32>),
    /// Calendar date stored as days since 1970-01-01. Round-trips
    /// through Arrow `Date32`. Tagged as `{"$date": <days>}` in JSON
    /// so the typing survives the `__overflow_json` path.
    Date(i32),
    /// Wall-clock instant stored as microseconds since
    /// 1970-01-01T00:00:00Z. Round-trips through Arrow
    /// `Timestamp(Microsecond, UTC)`. Tagged as `{"$datetime": <us>}`
    /// in JSON for the same reason as [`Value::Date`].
    DateTime(i64),
    /// Heterogeneous list of values. Stored only through
    /// `__overflow_json` (declared properties stay in their typed
    /// column); the JSON shape is `{"$list": [v0, v1, ...]}` so a
    /// plain array can keep round-tripping as `Vec<f32>`.
    List(Vec<Value>),
    /// String-keyed map of values. JSON shape is
    /// `{"$map": {"k": v, ...}}` to keep `{}` reserved for the
    /// existing date/datetime tags.
    Map(BTreeMap<String, Value>),
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

// ── Serde wire format ────────────────────────────────────────────────
//
// Most variants serialise to their natural JSON shape (a number, a
// string, an array, ...). Date and DateTime are tagged as one-key
// objects so the typing survives a JSON round-trip; without the tag
// `serde(untagged)` would see a plain integer and parse it back as
// `Value::I64`, losing the type.

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Null => s.serialize_unit(),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::I64(n) => s.serialize_i64(*n),
            Value::F64(f) => s.serialize_f64(*f),
            Value::Str(v) => s.serialize_str(v),
            Value::Bytes(b) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry(TAG_BYTES, b)?;
                m.end()
            }
            Value::Vec(v) => v.serialize(s),
            Value::Date(d) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry(TAG_DATE, d)?;
                m.end()
            }
            Value::DateTime(us) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry(TAG_DATETIME, us)?;
                m.end()
            }
            Value::List(items) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry(TAG_LIST, items)?;
                m.end()
            }
            Value::Map(entries) => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry(TAG_MAP, entries)?;
                m.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = Value;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON scalar, array, or {$date|$datetime: N} map")
            }

            fn visit_unit<E>(self) -> Result<Value, E> {
                Ok(Value::Null)
            }
            fn visit_none<E>(self) -> Result<Value, E> {
                Ok(Value::Null)
            }
            fn visit_some<D2: Deserializer<'de>>(self, d: D2) -> Result<Value, D2::Error> {
                Value::deserialize(d)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
                Ok(Value::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
                Ok(Value::I64(v))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
                Ok(Value::I64(v as i64))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
                Ok(Value::F64(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Value, E>
            where
                E: de::Error,
            {
                Ok(Value::Str(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<Value, E> {
                Ok(Value::Str(v))
            }
            fn visit_bytes<E>(self, v: &[u8]) -> Result<Value, E> {
                Ok(Value::Bytes(v.to_vec()))
            }
            fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Value, E> {
                Ok(Value::Bytes(v))
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
                // Float vectors round-trip through `Value::Vec`. JSON
                // arrays decode to `Vec<f32>` when every element is
                // numeric, falling back to an empty vec otherwise.
                let mut out = Vec::new();
                while let Some(item) = seq.next_element::<f64>()? {
                    out.push(item as f32);
                }
                Ok(Value::Vec(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
                let Some(key) = map.next_key::<String>()? else {
                    return Ok(Value::Null);
                };
                match key.as_str() {
                    TAG_DATE => {
                        let days: i32 = map.next_value()?;
                        // Reject extra keys to stay strict.
                        if map.next_key::<String>()?.is_some() {
                            return Err(de::Error::custom("$date map must have exactly one key"));
                        }
                        Ok(Value::Date(days))
                    }
                    TAG_DATETIME => {
                        let us: i64 = map.next_value()?;
                        if map.next_key::<String>()?.is_some() {
                            return Err(de::Error::custom(
                                "$datetime map must have exactly one key",
                            ));
                        }
                        Ok(Value::DateTime(us))
                    }
                    TAG_LIST => {
                        let items: Vec<Value> = map.next_value()?;
                        if map.next_key::<String>()?.is_some() {
                            return Err(de::Error::custom("$list map must have exactly one key"));
                        }
                        Ok(Value::List(items))
                    }
                    TAG_MAP => {
                        let entries: BTreeMap<String, Value> = map.next_value()?;
                        if map.next_key::<String>()?.is_some() {
                            return Err(de::Error::custom("$map map must have exactly one key"));
                        }
                        Ok(Value::Map(entries))
                    }
                    TAG_BYTES => {
                        let bytes: Vec<u8> = map.next_value()?;
                        if map.next_key::<String>()?.is_some() {
                            return Err(de::Error::custom("$bytes map must have exactly one key"));
                        }
                        Ok(Value::Bytes(bytes))
                    }
                    other => Err(de::Error::unknown_field(
                        other,
                        &[TAG_DATE, TAG_DATETIME, TAG_LIST, TAG_MAP, TAG_BYTES],
                    )),
                }
            }
        }
        d.deserialize_any(ValueVisitor)
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

    fn round(v: &Value) -> Value {
        let s = serde_json::to_string(v).unwrap();
        serde_json::from_str(&s).unwrap()
    }

    #[test]
    fn value_serde_roundtrip_scalars() {
        for v in [
            Value::Null,
            Value::Bool(false),
            Value::Bool(true),
            Value::I64(-1_234_567_890_123),
            Value::F64(1.5),
            Value::Str("hello".into()),
            Value::Vec(vec![0.1, 0.2, 0.3]),
        ] {
            assert_eq!(round(&v), v);
        }
    }

    #[test]
    fn value_bytes_roundtrip_through_json() {
        let v = Value::Bytes(vec![0u8, 1, 2]);
        let json = serde_json::to_string(&v).unwrap();
        eprintln!("bytes serialized as: {json}");
        let back: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v, "bytes must round-trip; got {back:?}");
    }

    #[test]
    fn value_serde_roundtrip_date_and_datetime() {
        let d = Value::Date(20_597);
        assert_eq!(round(&d), d);
        let dt = Value::DateTime(1_779_625_845_000_000);
        assert_eq!(round(&dt), dt);
    }

    #[test]
    fn value_date_is_tagged_in_json() {
        let s = serde_json::to_string(&Value::Date(5)).unwrap();
        assert_eq!(s, r#"{"$date":5}"#);
        let s = serde_json::to_string(&Value::DateTime(42)).unwrap();
        assert_eq!(s, r#"{"$datetime":42}"#);
    }

    #[test]
    fn value_list_roundtrips_through_json() {
        let v = Value::List(vec![
            Value::I64(1),
            Value::Str("two".into()),
            Value::Bool(true),
            Value::List(vec![Value::I64(3), Value::I64(4)]),
        ]);
        assert_eq!(round(&v), v);
        // Ensure the tag is present so a plain array does not
        // accidentally re-decode this list as `Vec<f32>`.
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.starts_with(r#"{"$list":"#), "got {s}");
    }

    #[test]
    fn value_map_roundtrips_through_json() {
        let mut m = BTreeMap::new();
        m.insert("name".to_string(), Value::Str("Ada".into()));
        m.insert("age".to_string(), Value::I64(36));
        m.insert(
            "tags".to_string(),
            Value::List(vec![Value::Str("rust".into())]),
        );
        let v = Value::Map(m);
        assert_eq!(round(&v), v);
        let s = serde_json::to_string(&v).unwrap();
        assert!(s.starts_with(r#"{"$map":"#), "got {s}");
    }

    #[test]
    fn value_plain_array_still_decodes_as_vec() {
        // Vec<f32> stays untagged so legacy JSON keeps round-tripping
        // without the `$list` marker. The deserializer's visit_seq
        // path is exercised by writing a bare array and reading it
        // back.
        let json = "[0.5, 1.5, 2.5]";
        let v: Value = serde_json::from_str(json).unwrap();
        assert_eq!(v, Value::Vec(vec![0.5, 1.5, 2.5]));
    }
}
