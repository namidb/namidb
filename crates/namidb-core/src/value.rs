//! Owned property values used by ingest APIs.
//!
//! On the hot path we always go through Arrow batches; [`Value`] only exists
//! to give a friendly Rust-native shape to ad-hoc insertions and to feed the
//! ingest path before columnarisation.

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// JSON object key used to tag a [`Value::Date`] when it round-trips
/// through `__overflow_json`. Picked to be reserved-by-convention so a
/// user property called `"$date"` doesn't collide with the wire form.
const TAG_DATE: &str = "$date";
/// Same idea for [`Value::DateTime`].
const TAG_DATETIME: &str = "$datetime";

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
            Value::Bytes(b) => s.serialize_bytes(b),
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
                    other => Err(de::Error::unknown_field(other, &[TAG_DATE, TAG_DATETIME])),
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
}
