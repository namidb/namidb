//! Bolt request and response messages.
//!
//! On the wire every message is a PackStream `Struct` with a 1-byte
//! tag plus N positional fields. This module decodes inbound bytes
//! into [`Request`] and encodes [`Response`] back out, leaning on the
//! generic codec in [`crate::codec`].
//!
//! See RFC-022 §"Message vocabulary" for the v5.4 surface.

use std::collections::BTreeMap;

use bytes::BytesMut;

use crate::codec::{decode_with_limit, encode};
use crate::error::{BoltError, Result};
use crate::value::{struct_tag, Value};

/// Pre-auth message body cap. After LOGON the session may raise it.
pub const PRE_AUTH_MESSAGE_BYTES: usize = 64 * 1024;
/// Post-auth message body cap. 16 MiB is the same ceiling Memgraph
/// applies; large `RUN` parameter maps stay under that comfortably.
pub const POST_AUTH_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Request messages a client can send.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    Hello(BTreeMap<String, Value>),
    Logon(BTreeMap<String, Value>),
    Logoff,
    Goodbye,
    Reset,
    Run {
        cypher: String,
        params: BTreeMap<String, Value>,
        extra: BTreeMap<String, Value>,
    },
    Pull {
        extra: BTreeMap<String, Value>,
    },
    Discard {
        extra: BTreeMap<String, Value>,
    },
    Begin(BTreeMap<String, Value>),
    Commit,
    Rollback,
    Route {
        routing: BTreeMap<String, Value>,
        bookmarks: Vec<Value>,
        extra: Value, // db / impersonated_user — driver-shaped, opaque to us
    },
    Telemetry(BTreeMap<String, Value>),
    /// Marker for an unrecognised struct tag. The session emits a
    /// `FAILURE { code: "Neo.ClientError.Request.Invalid" }` and
    /// transitions to `FAILED` (until RESET).
    Unknown {
        tag: u8,
        fields: Vec<Value>,
    },
}

impl Request {
    /// Decode a Bolt request message from a body that came off the
    /// chunked framing layer. The body must be a single PackStream
    /// `Struct` value.
    pub fn decode(body: &[u8], max_len: usize) -> Result<Self> {
        let mut slice: &[u8] = body;
        let value = decode_with_limit(&mut slice, max_len)?;
        if !slice.is_empty() {
            return Err(BoltError::Protocol(format!(
                "trailing bytes after message struct: {} bytes",
                slice.len()
            )));
        }
        Self::from_value(value)
    }

    fn from_value(value: Value) -> Result<Self> {
        let (tag, fields) = match value {
            Value::Struct { tag, fields } => (tag, fields),
            other => {
                return Err(BoltError::Protocol(format!(
                    "expected Struct message, got {}",
                    other.type_name()
                )));
            }
        };
        match tag {
            struct_tag::HELLO => {
                let extra = take_map_field("HELLO", &fields, 0)?;
                Ok(Request::Hello(extra))
            }
            struct_tag::LOGON => {
                let extra = take_map_field("LOGON", &fields, 0)?;
                Ok(Request::Logon(extra))
            }
            struct_tag::LOGOFF => Ok(Request::Logoff),
            struct_tag::GOODBYE => Ok(Request::Goodbye),
            struct_tag::RESET => Ok(Request::Reset),
            struct_tag::RUN => {
                let cypher = take_string_field("RUN", &fields, 0)?;
                let params = take_map_field("RUN", &fields, 1)?;
                let extra = fields
                    .get(2)
                    .cloned()
                    .map(|v| match v {
                        Value::Map(m) => Ok(m),
                        other => Err(BoltError::MalformedStruct {
                            struct_name: "RUN",
                            detail: format!("field 2 must be Map, got {}", other.type_name()),
                        }),
                    })
                    .transpose()?
                    .unwrap_or_default();
                Ok(Request::Run {
                    cypher,
                    params,
                    extra,
                })
            }
            struct_tag::PULL => {
                let extra = take_map_field("PULL", &fields, 0)?;
                Ok(Request::Pull { extra })
            }
            struct_tag::DISCARD => {
                let extra = take_map_field("DISCARD", &fields, 0)?;
                Ok(Request::Discard { extra })
            }
            struct_tag::BEGIN => {
                let extra = take_map_field("BEGIN", &fields, 0)?;
                Ok(Request::Begin(extra))
            }
            struct_tag::COMMIT => Ok(Request::Commit),
            struct_tag::ROLLBACK => Ok(Request::Rollback),
            struct_tag::ROUTE => {
                let routing = take_map_field("ROUTE", &fields, 0)?;
                let bookmarks = match fields.get(1) {
                    Some(Value::List(items)) => items.clone(),
                    Some(Value::Null) | None => Vec::new(),
                    Some(other) => {
                        return Err(BoltError::MalformedStruct {
                            struct_name: "ROUTE",
                            detail: format!("field 1 must be List, got {}", other.type_name()),
                        });
                    }
                };
                let extra = fields.get(2).cloned().unwrap_or(Value::Null);
                Ok(Request::Route {
                    routing,
                    bookmarks,
                    extra,
                })
            }
            struct_tag::TELEMETRY => {
                let extra = take_map_field("TELEMETRY", &fields, 0)?;
                Ok(Request::Telemetry(extra))
            }
            other => Ok(Request::Unknown { tag: other, fields }),
        }
    }

    /// The static label this message carries on the wire. Used by the
    /// session to log + by error reporting to name the offender.
    pub fn name(&self) -> &'static str {
        match self {
            Request::Hello(_) => "HELLO",
            Request::Logon(_) => "LOGON",
            Request::Logoff => "LOGOFF",
            Request::Goodbye => "GOODBYE",
            Request::Reset => "RESET",
            Request::Run { .. } => "RUN",
            Request::Pull { .. } => "PULL",
            Request::Discard { .. } => "DISCARD",
            Request::Begin(_) => "BEGIN",
            Request::Commit => "COMMIT",
            Request::Rollback => "ROLLBACK",
            Request::Route { .. } => "ROUTE",
            Request::Telemetry(_) => "TELEMETRY",
            Request::Unknown { .. } => "UNKNOWN",
        }
    }
}

fn take_map_field(
    struct_name: &'static str,
    fields: &[Value],
    idx: usize,
) -> Result<BTreeMap<String, Value>> {
    match fields.get(idx) {
        Some(Value::Map(m)) => Ok(m.clone()),
        Some(other) => Err(BoltError::MalformedStruct {
            struct_name,
            detail: format!("field {idx} must be Map, got {}", other.type_name()),
        }),
        None => Err(BoltError::MalformedStruct {
            struct_name,
            detail: format!("missing field at index {idx}"),
        }),
    }
}

fn take_string_field(struct_name: &'static str, fields: &[Value], idx: usize) -> Result<String> {
    match fields.get(idx) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(BoltError::MalformedStruct {
            struct_name,
            detail: format!("field {idx} must be String, got {}", other.type_name()),
        }),
        None => Err(BoltError::MalformedStruct {
            struct_name,
            detail: format!("missing field at index {idx}"),
        }),
    }
}

/// Response messages a server sends.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    Success(BTreeMap<String, Value>),
    Record(Vec<Value>),
    Ignored,
    Failure(BTreeMap<String, Value>),
}

impl Response {
    pub fn name(&self) -> &'static str {
        match self {
            Response::Success(_) => "SUCCESS",
            Response::Record(_) => "RECORD",
            Response::Ignored => "IGNORED",
            Response::Failure(_) => "FAILURE",
        }
    }

    /// Encode the response as a single PackStream-encoded message
    /// body (no chunked framing, that lives in [`crate::chunk`]).
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut buf = BytesMut::new();
        let value = self.to_value();
        encode(&mut buf, &value)?;
        Ok(buf.to_vec())
    }

    fn to_value(&self) -> Value {
        match self {
            Response::Success(meta) => Value::Struct {
                tag: struct_tag::SUCCESS,
                fields: vec![Value::Map(meta.clone())],
            },
            Response::Record(values) => Value::Struct {
                tag: struct_tag::RECORD,
                fields: vec![Value::List(values.clone())],
            },
            Response::Ignored => Value::Struct {
                tag: struct_tag::IGNORED,
                fields: vec![],
            },
            Response::Failure(meta) => Value::Struct {
                tag: struct_tag::FAILURE,
                fields: vec![Value::Map(meta.clone())],
            },
        }
    }

    /// Convenience: build a `FAILURE { code, message }` map.
    pub fn failure(code: &str, message: impl Into<String>) -> Self {
        let mut m = BTreeMap::new();
        m.insert("code".into(), Value::String(code.into()));
        m.insert("message".into(), Value::String(message.into()));
        Response::Failure(m)
    }

    /// Convenience: build an empty `SUCCESS {}`.
    pub fn success_empty() -> Self {
        Response::Success(BTreeMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode;

    fn pack_struct(tag: u8, fields: Vec<Value>) -> Vec<u8> {
        let mut buf = BytesMut::new();
        encode(&mut buf, &Value::Struct { tag, fields }).unwrap();
        buf.to_vec()
    }

    #[test]
    fn decode_hello_minimal() {
        let mut m = BTreeMap::new();
        m.insert(
            "user_agent".into(),
            Value::String("neo4j-python/5.27.0".into()),
        );
        let body = pack_struct(struct_tag::HELLO, vec![Value::Map(m.clone())]);
        let req = Request::decode(&body, POST_AUTH_MESSAGE_BYTES).unwrap();
        assert_eq!(req, Request::Hello(m));
    }

    #[test]
    fn decode_run_with_extras() {
        let mut params = BTreeMap::new();
        params.insert("name".into(), Value::String("Alice".into()));
        let mut extra = BTreeMap::new();
        extra.insert("mode".into(), Value::String("r".into()));
        let body = pack_struct(
            struct_tag::RUN,
            vec![
                Value::String("MATCH (n) RETURN n LIMIT $k".into()),
                Value::Map(params.clone()),
                Value::Map(extra.clone()),
            ],
        );
        let req = Request::decode(&body, POST_AUTH_MESSAGE_BYTES).unwrap();
        match req {
            Request::Run {
                cypher,
                params: p,
                extra: e,
            } => {
                assert_eq!(cypher, "MATCH (n) RETURN n LIMIT $k");
                assert_eq!(p, params);
                assert_eq!(e, extra);
            }
            other => panic!("expected Run, got {:?}", other),
        }
    }

    #[test]
    fn decode_zero_arg_messages() {
        for (tag, expected) in &[
            (struct_tag::GOODBYE, Request::Goodbye),
            (struct_tag::RESET, Request::Reset),
            (struct_tag::COMMIT, Request::Commit),
            (struct_tag::ROLLBACK, Request::Rollback),
            (struct_tag::LOGOFF, Request::Logoff),
        ] {
            let body = pack_struct(*tag, vec![]);
            let req = Request::decode(&body, POST_AUTH_MESSAGE_BYTES).unwrap();
            assert_eq!(&req, expected);
        }
    }

    #[test]
    fn decode_unknown_tag() {
        let body = pack_struct(0x77, vec![]);
        let req = Request::decode(&body, POST_AUTH_MESSAGE_BYTES).unwrap();
        match req {
            Request::Unknown { tag, fields } => {
                assert_eq!(tag, 0x77);
                assert!(fields.is_empty());
            }
            other => panic!("expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn encode_record_roundtrip() {
        let resp = Response::Record(vec![Value::Int(1), Value::String("ok".into())]);
        let bytes = resp.encode().unwrap();
        let mut slice: &[u8] = &bytes;
        let decoded = crate::codec::decode(&mut slice).unwrap();
        match decoded {
            Value::Struct { tag, fields } => {
                assert_eq!(tag, struct_tag::RECORD);
                assert_eq!(fields.len(), 1);
                match &fields[0] {
                    Value::List(items) => assert_eq!(items.len(), 2),
                    other => panic!("expected List, got {:?}", other),
                }
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn encode_failure_carries_code_and_message() {
        let resp = Response::failure("Neo.ClientError.Statement.SyntaxError", "bad parse");
        let bytes = resp.encode().unwrap();
        let mut slice: &[u8] = &bytes;
        let decoded = crate::codec::decode(&mut slice).unwrap();
        let (tag, fields) = match decoded {
            Value::Struct { tag, fields } => (tag, fields),
            other => panic!("expected Struct, got {:?}", other),
        };
        assert_eq!(tag, struct_tag::FAILURE);
        let map = fields[0].as_map().unwrap();
        assert_eq!(
            map.get("code").unwrap(),
            &Value::String("Neo.ClientError.Statement.SyntaxError".into())
        );
        assert_eq!(
            map.get("message").unwrap(),
            &Value::String("bad parse".into())
        );
    }

    #[test]
    fn trailing_bytes_rejected() {
        let body = pack_struct(struct_tag::RESET, vec![]);
        let mut padded = body.clone();
        padded.push(0xFF);
        let err = Request::decode(&padded, POST_AUTH_MESSAGE_BYTES).unwrap_err();
        assert!(matches!(err, BoltError::Protocol(_)));
    }
}
