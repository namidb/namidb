//! PackStream encoder and decoder.
//!
//! PackStream is the binary serialisation Bolt uses for every value on
//! the wire. The spec lives at
//! <https://neo4j.com/docs/bolt/current/packstream/>; this module is a
//! direct, marker-by-marker translation.
//!
//! ## Marker table (informative)
//!
//! ```text
//! 0xC0                Null
//! 0xC2 / 0xC3         Bool false / true
//! 0xF0..0xFF, 0x00..0x7F  TinyInt (i8 inlined in the marker)
//! 0xC8 i8             Int8
//! 0xC9 i16            Int16  (big-endian)
//! 0xCA i32            Int32
//! 0xCB i64            Int64
//! 0xC1 f64            Float (big-endian)
//! 0x80..0x8F payload  TinyString  (4-bit length)
//! 0xD0 u8 payload     String8
//! 0xD1 u16 payload    String16
//! 0xD2 u32 payload    String32
//! 0xCC u8 payload     Bytes8
//! 0xCD u16 payload    Bytes16
//! 0xCE u32 payload    Bytes32
//! 0x90..0x9F items    TinyList
//! 0xD4 u8 items       List8
//! 0xD5 u16 items      List16
//! 0xD6 u32 items      List32
//! 0xA0..0xAF kv-pairs TinyMap
//! 0xD8 u8 kv-pairs    Map8
//! 0xD9 u16 kv-pairs   Map16
//! 0xDA u32 kv-pairs   Map32
//! 0xB0..0xBF tag      TinyStruct  (4-bit field count, 1-byte tag follows)
//! 0xDC u8 tag         Struct8     (8-bit field count)
//! 0xDD u16 tag        Struct16    (16-bit field count)
//! ```

use std::collections::BTreeMap;

use bytes::{Buf, BufMut, BytesMut};

use crate::error::{BoltError, Result};
use crate::value::Value;

/// Max byte length of any single PackStream container we accept on the
/// wire pre-auth. After LOGON the message decoder may raise this; the
/// pre-auth cap keeps an unauthenticated client from forcing huge
/// allocations.
pub const DEFAULT_MAX_LEN: usize = 1 << 24; // 16 MiB

/// Max container-nesting depth we decode. PackStream values are `List`s,
/// `Map`s, and `Struct`s that can nest; a malformed message can declare
/// nesting far deeper than any real value to drive `decode_inner` into
/// unbounded recursion and overflow the (non-unwindable) worker-thread
/// stack, aborting the whole process — reachable pre-auth. Legitimate Bolt
/// values nest only a handful of levels (a Node inside a Relationship inside
/// a Path inside a result List), so 128 is generous while still bounding the
/// stack. Exceeding it yields a clean `NestingTooDeep` FAILURE.
pub const MAX_NESTING_DEPTH: usize = 128;

/// Encode `value` into `out`.
pub fn encode(out: &mut BytesMut, value: &Value) -> Result<()> {
    match value {
        Value::Null => out.put_u8(0xC0),
        Value::Bool(false) => out.put_u8(0xC2),
        Value::Bool(true) => out.put_u8(0xC3),
        Value::Int(n) => encode_int(out, *n),
        Value::Float(f) => {
            out.put_u8(0xC1);
            out.put_f64(*f);
        }
        Value::String(s) => encode_string(out, s),
        Value::Bytes(b) => encode_bytes(out, b),
        Value::List(items) => {
            encode_list_header(out, items.len())?;
            for v in items {
                encode(out, v)?;
            }
        }
        Value::Map(m) => {
            encode_map_header(out, m.len())?;
            for (k, v) in m {
                encode_string(out, k);
                encode(out, v)?;
            }
        }
        Value::Struct { tag, fields } => {
            encode_struct_header(out, fields.len(), *tag)?;
            for v in fields {
                encode(out, v)?;
            }
        }
    }
    Ok(())
}

fn encode_int(out: &mut BytesMut, n: i64) {
    if (-16..=127).contains(&n) {
        out.put_u8(n as u8);
    } else if (i8::MIN as i64..=i8::MAX as i64).contains(&n) {
        out.put_u8(0xC8);
        out.put_i8(n as i8);
    } else if (i16::MIN as i64..=i16::MAX as i64).contains(&n) {
        out.put_u8(0xC9);
        out.put_i16(n as i16);
    } else if (i32::MIN as i64..=i32::MAX as i64).contains(&n) {
        out.put_u8(0xCA);
        out.put_i32(n as i32);
    } else {
        out.put_u8(0xCB);
        out.put_i64(n);
    }
}

fn encode_string(out: &mut BytesMut, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len <= 15 {
        out.put_u8(0x80 | (len as u8));
    } else if len <= u8::MAX as usize {
        out.put_u8(0xD0);
        out.put_u8(len as u8);
    } else if len <= u16::MAX as usize {
        out.put_u8(0xD1);
        out.put_u16(len as u16);
    } else {
        out.put_u8(0xD2);
        out.put_u32(len as u32);
    }
    out.put_slice(bytes);
}

fn encode_bytes(out: &mut BytesMut, b: &[u8]) {
    let len = b.len();
    if len <= u8::MAX as usize {
        out.put_u8(0xCC);
        out.put_u8(len as u8);
    } else if len <= u16::MAX as usize {
        out.put_u8(0xCD);
        out.put_u16(len as u16);
    } else {
        out.put_u8(0xCE);
        out.put_u32(len as u32);
    }
    out.put_slice(b);
}

fn encode_list_header(out: &mut BytesMut, len: usize) -> Result<()> {
    if len <= 15 {
        out.put_u8(0x90 | (len as u8));
    } else if len <= u8::MAX as usize {
        out.put_u8(0xD4);
        out.put_u8(len as u8);
    } else if len <= u16::MAX as usize {
        out.put_u8(0xD5);
        out.put_u16(len as u16);
    } else if len <= u32::MAX as usize {
        out.put_u8(0xD6);
        out.put_u32(len as u32);
    } else {
        return Err(BoltError::TooLarge {
            what: "List",
            len,
            max: u32::MAX as usize,
        });
    }
    Ok(())
}

fn encode_map_header(out: &mut BytesMut, len: usize) -> Result<()> {
    if len <= 15 {
        out.put_u8(0xA0 | (len as u8));
    } else if len <= u8::MAX as usize {
        out.put_u8(0xD8);
        out.put_u8(len as u8);
    } else if len <= u16::MAX as usize {
        out.put_u8(0xD9);
        out.put_u16(len as u16);
    } else if len <= u32::MAX as usize {
        out.put_u8(0xDA);
        out.put_u32(len as u32);
    } else {
        return Err(BoltError::TooLarge {
            what: "Map",
            len,
            max: u32::MAX as usize,
        });
    }
    Ok(())
}

fn encode_struct_header(out: &mut BytesMut, fields: usize, tag: u8) -> Result<()> {
    if fields <= 15 {
        out.put_u8(0xB0 | (fields as u8));
        out.put_u8(tag);
    } else if fields <= u8::MAX as usize {
        out.put_u8(0xDC);
        out.put_u8(fields as u8);
        out.put_u8(tag);
    } else if fields <= u16::MAX as usize {
        out.put_u8(0xDD);
        out.put_u16(fields as u16);
        out.put_u8(tag);
    } else {
        return Err(BoltError::TooLarge {
            what: "Struct",
            len: fields,
            max: u16::MAX as usize,
        });
    }
    Ok(())
}

/// Decode one value from `buf`, advancing it past the consumed bytes.
pub fn decode(buf: &mut &[u8]) -> Result<Value> {
    decode_inner(buf, DEFAULT_MAX_LEN, 0)
}

/// Decode one value with a custom max-container-length bound. The
/// session uses a lower bound pre-auth.
pub fn decode_with_limit(buf: &mut &[u8], max_len: usize) -> Result<Value> {
    decode_inner(buf, max_len, 0)
}

fn decode_inner(buf: &mut &[u8], max_len: usize, depth: usize) -> Result<Value> {
    if depth > MAX_NESTING_DEPTH {
        return Err(BoltError::NestingTooDeep {
            max: MAX_NESTING_DEPTH,
        });
    }
    let marker = read_u8(buf, "marker")?;
    // Hot path: tiny structures use the marker's high nibble + low
    // nibble for length. We check ranges in order of expected
    // frequency.
    match marker {
        0xC0 => Ok(Value::Null),
        0xC2 => Ok(Value::Bool(false)),
        0xC3 => Ok(Value::Bool(true)),
        0xC1 => {
            let f = read_f64(buf)?;
            Ok(Value::Float(f))
        }
        0xC8 => Ok(Value::Int(read_i8(buf)? as i64)),
        0xC9 => Ok(Value::Int(read_i16(buf)? as i64)),
        0xCA => Ok(Value::Int(read_i32(buf)? as i64)),
        0xCB => Ok(Value::Int(read_i64(buf)?)),

        // TinyInt: 0xF0..0xFF (negative) or 0x00..0x7F (non-negative).
        0xF0..=0xFF => Ok(Value::Int(marker as i8 as i64)),
        0x00..=0x7F => Ok(Value::Int(marker as i64)),

        // TinyString
        0x80..=0x8F => {
            let len = (marker & 0x0F) as usize;
            decode_string_body(buf, len, max_len)
        }
        0xD0 => {
            let len = read_u8(buf, "String8 length")? as usize;
            decode_string_body(buf, len, max_len)
        }
        0xD1 => {
            let len = read_u16(buf, "String16 length")? as usize;
            decode_string_body(buf, len, max_len)
        }
        0xD2 => {
            let len = read_u32(buf, "String32 length")? as usize;
            decode_string_body(buf, len, max_len)
        }

        0xCC => {
            let len = read_u8(buf, "Bytes8 length")? as usize;
            decode_bytes_body(buf, len, max_len)
        }
        0xCD => {
            let len = read_u16(buf, "Bytes16 length")? as usize;
            decode_bytes_body(buf, len, max_len)
        }
        0xCE => {
            let len = read_u32(buf, "Bytes32 length")? as usize;
            decode_bytes_body(buf, len, max_len)
        }

        // TinyList
        0x90..=0x9F => {
            let len = (marker & 0x0F) as usize;
            decode_list_body(buf, len, max_len, depth)
        }
        0xD4 => {
            let len = read_u8(buf, "List8 length")? as usize;
            decode_list_body(buf, len, max_len, depth)
        }
        0xD5 => {
            let len = read_u16(buf, "List16 length")? as usize;
            decode_list_body(buf, len, max_len, depth)
        }
        0xD6 => {
            let len = read_u32(buf, "List32 length")? as usize;
            decode_list_body(buf, len, max_len, depth)
        }

        // TinyMap
        0xA0..=0xAF => {
            let len = (marker & 0x0F) as usize;
            decode_map_body(buf, len, max_len, depth)
        }
        0xD8 => {
            let len = read_u8(buf, "Map8 length")? as usize;
            decode_map_body(buf, len, max_len, depth)
        }
        0xD9 => {
            let len = read_u16(buf, "Map16 length")? as usize;
            decode_map_body(buf, len, max_len, depth)
        }
        0xDA => {
            let len = read_u32(buf, "Map32 length")? as usize;
            decode_map_body(buf, len, max_len, depth)
        }

        // TinyStruct
        0xB0..=0xBF => {
            let fields = (marker & 0x0F) as usize;
            decode_struct_body(buf, fields, max_len, depth)
        }
        0xDC => {
            let fields = read_u8(buf, "Struct8 size")? as usize;
            decode_struct_body(buf, fields, max_len, depth)
        }
        0xDD => {
            let fields = read_u16(buf, "Struct16 size")? as usize;
            decode_struct_body(buf, fields, max_len, depth)
        }

        other => Err(BoltError::InvalidMarker {
            byte: other,
            expected: "any PackStream value",
        }),
    }
}

fn decode_string_body(buf: &mut &[u8], len: usize, max_len: usize) -> Result<Value> {
    bound_check("String", len, max_len)?;
    if buf.len() < len {
        return Err(BoltError::UnexpectedEof { what: "String" });
    }
    let s = std::str::from_utf8(&buf[..len])?.to_string();
    buf.advance(len);
    Ok(Value::String(s))
}

fn decode_bytes_body(buf: &mut &[u8], len: usize, max_len: usize) -> Result<Value> {
    bound_check("Bytes", len, max_len)?;
    if buf.len() < len {
        return Err(BoltError::UnexpectedEof { what: "Bytes" });
    }
    let out = buf[..len].to_vec();
    buf.advance(len);
    Ok(Value::Bytes(out))
}

fn decode_list_body(buf: &mut &[u8], len: usize, max_len: usize, depth: usize) -> Result<Value> {
    bound_check("List", len, max_len)?;
    let mut items = Vec::with_capacity(len);
    for _ in 0..len {
        items.push(decode_inner(buf, max_len, depth + 1)?);
    }
    Ok(Value::List(items))
}

fn decode_map_body(buf: &mut &[u8], len: usize, max_len: usize, depth: usize) -> Result<Value> {
    bound_check("Map", len, max_len)?;
    let mut out = BTreeMap::new();
    for _ in 0..len {
        let k = decode_inner(buf, max_len, depth + 1)?;
        let key = match k {
            Value::String(s) => s,
            other => {
                return Err(BoltError::MalformedStruct {
                    struct_name: "Map",
                    detail: format!("key must be String, got {}", other.type_name()),
                })
            }
        };
        let value = decode_inner(buf, max_len, depth + 1)?;
        out.insert(key, value);
    }
    Ok(Value::Map(out))
}

fn decode_struct_body(
    buf: &mut &[u8],
    fields: usize,
    max_len: usize,
    depth: usize,
) -> Result<Value> {
    let tag = read_u8(buf, "Struct tag")?;
    bound_check("Struct", fields, max_len)?;
    let mut out = Vec::with_capacity(fields);
    for _ in 0..fields {
        out.push(decode_inner(buf, max_len, depth + 1)?);
    }
    Ok(Value::Struct { tag, fields: out })
}

fn bound_check(what: &'static str, len: usize, max: usize) -> Result<()> {
    if len > max {
        Err(BoltError::TooLarge { what, len, max })
    } else {
        Ok(())
    }
}

fn read_u8(buf: &mut &[u8], what: &'static str) -> Result<u8> {
    if buf.is_empty() {
        return Err(BoltError::UnexpectedEof { what });
    }
    Ok(buf.get_u8())
}
fn read_u16(buf: &mut &[u8], what: &'static str) -> Result<u16> {
    if buf.len() < 2 {
        return Err(BoltError::UnexpectedEof { what });
    }
    Ok(buf.get_u16())
}
fn read_u32(buf: &mut &[u8], what: &'static str) -> Result<u32> {
    if buf.len() < 4 {
        return Err(BoltError::UnexpectedEof { what });
    }
    Ok(buf.get_u32())
}
fn read_i8(buf: &mut &[u8]) -> Result<i8> {
    if buf.is_empty() {
        return Err(BoltError::UnexpectedEof { what: "Int8" });
    }
    Ok(buf.get_i8())
}
fn read_i16(buf: &mut &[u8]) -> Result<i16> {
    if buf.len() < 2 {
        return Err(BoltError::UnexpectedEof { what: "Int16" });
    }
    Ok(buf.get_i16())
}
fn read_i32(buf: &mut &[u8]) -> Result<i32> {
    if buf.len() < 4 {
        return Err(BoltError::UnexpectedEof { what: "Int32" });
    }
    Ok(buf.get_i32())
}
fn read_i64(buf: &mut &[u8]) -> Result<i64> {
    if buf.len() < 8 {
        return Err(BoltError::UnexpectedEof { what: "Int64" });
    }
    Ok(buf.get_i64())
}
fn read_f64(buf: &mut &[u8]) -> Result<f64> {
    if buf.len() < 8 {
        return Err(BoltError::UnexpectedEof { what: "Float" });
    }
    Ok(buf.get_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn roundtrip(v: &Value) -> Value {
        let mut bytes = BytesMut::new();
        encode(&mut bytes, v).expect("encode");
        let frozen = bytes.freeze();
        let mut slice: &[u8] = &frozen;
        let decoded = decode(&mut slice).expect("decode");
        assert!(slice.is_empty(), "trailing bytes: {:?}", slice);
        decoded
    }

    #[test]
    fn null_roundtrip() {
        assert_eq!(roundtrip(&Value::Null), Value::Null);
    }

    #[test]
    fn deeply_nested_lists_error_instead_of_overflowing_the_stack() {
        // A malformed message can declare nesting far deeper than any real
        // value to drive the recursive decoder into a stack overflow (a
        // non-unwindable process abort), reachable pre-auth. `0x91` is a
        // TinyList-of-1: a long run of them is a legal-looking but pathological
        // nesting. The decoder must reject it with a clean `NestingTooDeep`
        // error, not recurse ~N frames deep. Terminate with a Null leaf.
        let mut body = vec![0x91u8; MAX_NESTING_DEPTH + 50];
        body.push(0xC0);
        let mut slice: &[u8] = &body;
        let err = decode(&mut slice).expect_err("must reject over-deep nesting");
        assert!(
            matches!(err, BoltError::NestingTooDeep { .. }),
            "expected NestingTooDeep, got {err:?}"
        );
    }

    #[test]
    fn nesting_at_the_limit_still_decodes() {
        // A value nested exactly to the limit must still decode — the guard
        // rejects only what exceeds it, so legitimate (shallow) values are
        // unaffected. Build MAX_NESTING_DEPTH TinyLists wrapping a Null.
        let mut body = vec![0x91u8; MAX_NESTING_DEPTH];
        body.push(0xC0);
        let mut slice: &[u8] = &body;
        let v = decode(&mut slice).expect("nesting at the limit must decode");
        // Unwrap the nested single-element lists down to the Null leaf.
        let mut cur = &v;
        let mut depth = 0;
        while let Value::List(items) = cur {
            assert_eq!(items.len(), 1);
            cur = &items[0];
            depth += 1;
        }
        assert_eq!(cur, &Value::Null);
        assert_eq!(depth, MAX_NESTING_DEPTH);
    }

    #[test]
    fn bool_roundtrip() {
        assert_eq!(roundtrip(&Value::Bool(false)), Value::Bool(false));
        assert_eq!(roundtrip(&Value::Bool(true)), Value::Bool(true));
    }

    #[test]
    fn tiny_ints_are_one_byte() {
        // -16..=127 are TinyInt; check both ends + a known v5 example
        // from the spec (0x7F → 127, 0xF0 → -16).
        for n in [-16i64, -1, 0, 1, 42, 127] {
            let mut bytes = BytesMut::new();
            encode(&mut bytes, &Value::Int(n)).unwrap();
            assert_eq!(bytes.len(), 1, "{} should be 1 byte", n);
            let mut slice: &[u8] = &bytes;
            assert_eq!(decode(&mut slice).unwrap(), Value::Int(n));
        }
    }

    #[test]
    fn int_widths() {
        // -17 escapes TinyInt downward → Int8 (0xC8).
        let mut b = BytesMut::new();
        encode(&mut b, &Value::Int(-17)).unwrap();
        assert_eq!(b[0], 0xC8);
        assert_eq!(roundtrip(&Value::Int(-17)), Value::Int(-17));

        // 128 escapes TinyInt upward → Int16 (0xC9). Int8 max is 127.
        let mut b = BytesMut::new();
        encode(&mut b, &Value::Int(128)).unwrap();
        assert_eq!(b[0], 0xC9);
        assert_eq!(
            roundtrip(&Value::Int(i32::MIN as i64)),
            Value::Int(i32::MIN as i64)
        );
        assert_eq!(roundtrip(&Value::Int(i64::MAX)), Value::Int(i64::MAX));
        assert_eq!(roundtrip(&Value::Int(i64::MIN)), Value::Int(i64::MIN));
    }

    #[test]
    fn float_roundtrip() {
        assert_eq!(roundtrip(&Value::Float(0.0)), Value::Float(0.0));
        assert_eq!(roundtrip(&Value::Float(-1.5)), Value::Float(-1.5));
        assert_eq!(
            roundtrip(&Value::Float(std::f64::consts::PI)),
            Value::Float(std::f64::consts::PI)
        );
    }

    #[test]
    fn string_widths() {
        // Tiny (<= 15)
        let s = "hello";
        let mut b = BytesMut::new();
        encode(&mut b, &Value::String(s.into())).unwrap();
        assert_eq!(b[0], 0x85);
        assert_eq!(&b[1..], b"hello");

        // String8 (16..=255)
        let s = "x".repeat(20);
        let mut b = BytesMut::new();
        encode(&mut b, &Value::String(s.clone())).unwrap();
        assert_eq!(b[0], 0xD0);
        assert_eq!(b[1], 20);
        assert_eq!(roundtrip(&Value::String(s.clone())), Value::String(s));

        // String16 (256..=65535)
        let s = "y".repeat(300);
        assert_eq!(roundtrip(&Value::String(s.clone())), Value::String(s));
    }

    #[test]
    fn list_widths() {
        let xs = (0..3).map(Value::Int).collect::<Vec<_>>();
        let v = Value::List(xs.clone());
        let mut b = BytesMut::new();
        encode(&mut b, &v).unwrap();
        assert_eq!(b[0], 0x93); // tiny-list, 3 items
        assert_eq!(roundtrip(&v), v);

        // List8
        let xs: Vec<_> = (0..20).map(|i| Value::Int(i as i64)).collect();
        assert_eq!(roundtrip(&Value::List(xs.clone())), Value::List(xs));
    }

    #[test]
    fn map_roundtrip() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), Value::Int(1));
        m.insert("b".to_string(), Value::String("two".into()));
        m.insert("c".to_string(), Value::Bool(true));
        assert_eq!(roundtrip(&Value::Map(m.clone())), Value::Map(m));
    }

    #[test]
    fn struct_roundtrip() {
        // HELLO {user_agent: "test/0"} → TinyStruct(1) tag=0x01 with a
        // single Map field.
        let mut m = BTreeMap::new();
        m.insert("user_agent".to_string(), Value::String("test/0".into()));
        let v = Value::Struct {
            tag: 0x01,
            fields: vec![Value::Map(m)],
        };
        let rt = roundtrip(&v);
        assert_eq!(rt, v);
    }

    #[test]
    fn truncated_input_errors() {
        // String8 marker says length=10 but only 3 bytes follow.
        let bytes = [0xD0u8, 0x0A, b'a', b'b', b'c'];
        let mut slice: &[u8] = &bytes;
        let err = decode(&mut slice).unwrap_err();
        assert!(matches!(err, BoltError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_utf8_errors() {
        let bytes = [0x82u8, 0xFF, 0xFE]; // tiny-string len=2, non-utf8
        let mut slice: &[u8] = &bytes;
        let err = decode(&mut slice).unwrap_err();
        assert!(matches!(err, BoltError::InvalidUtf8(_)));
    }

    proptest! {
        #[test]
        fn prop_int_roundtrip(n in any::<i64>()) {
            prop_assert_eq!(roundtrip(&Value::Int(n)), Value::Int(n));
        }

        #[test]
        fn prop_string_roundtrip(s in ".{0,500}") {
            prop_assert_eq!(roundtrip(&Value::String(s.clone())), Value::String(s));
        }

        #[test]
        fn prop_bytes_roundtrip(b in proptest::collection::vec(any::<u8>(), 0..500)) {
            prop_assert_eq!(roundtrip(&Value::Bytes(b.clone())), Value::Bytes(b));
        }

        #[test]
        fn prop_list_of_ints(xs in proptest::collection::vec(any::<i64>(), 0..30)) {
            let v = Value::List(xs.iter().copied().map(Value::Int).collect());
            prop_assert_eq!(roundtrip(&v.clone()), v);
        }
    }
}
