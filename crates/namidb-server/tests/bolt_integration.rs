//! End-to-end smoke test for the Bolt listener.
//!
//! Binds `namidb-server` on an OS-assigned port, drives a Bolt
//! session through `namidb-bolt`'s wire primitives (handshake,
//! chunked framing, message codec), and verifies that a CREATE
//! followed by a MATCH round-trips through the protocol.
//!
//! The test exercises the full pipeline:
//!
//! 1. Handshake (`0x6060B017` magic + Bolt 5.4).
//! 2. HELLO + LOGON with a bearer token.
//! 3. RUN `CREATE (a:Person {name: 'Alice', age: 30}) RETURN a.name AS name`
//!    + PULL — verifies the writer commits and the RECORD is shaped right.
//! 4. RUN `MATCH (p:Person) RETURN p.name, p.age` + PULL — verifies
//!    the read path sees the just-written node.
//! 5. GOODBYE.

use std::collections::BTreeMap;
use std::time::Duration;

use bytes::BytesMut;
use namidb_bolt::chunk::{read_message, write_message};
use namidb_bolt::codec::{decode, encode};
use namidb_bolt::message::POST_AUTH_MESSAGE_BYTES;
use namidb_bolt::value::{struct_tag, Value};
use namidb_bolt::Response;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

fn pack(v: &Value) -> Vec<u8> {
    let mut buf = BytesMut::new();
    encode(&mut buf, v).expect("encode");
    buf.to_vec()
}

async fn send_msg(stream: &mut TcpStream, body: &[u8]) {
    write_message(stream, body).await.expect("write_message");
}

async fn recv_msg(stream: &mut TcpStream) -> Response {
    let body = read_message(stream, POST_AUTH_MESSAGE_BYTES)
        .await
        .expect("read_message");
    decode_response(&body)
}

fn decode_response(body: &[u8]) -> Response {
    let mut slice: &[u8] = body;
    let v = decode(&mut slice).expect("decode");
    let (tag, mut fields) = match v {
        Value::Struct { tag, fields } => (tag, fields),
        other => panic!("expected struct, got {:?}", other),
    };
    match tag {
        struct_tag::SUCCESS => Response::Success(
            fields
                .pop()
                .and_then(|v| match v {
                    Value::Map(m) => Some(m),
                    _ => None,
                })
                .unwrap_or_default(),
        ),
        struct_tag::RECORD => Response::Record(
            fields
                .pop()
                .and_then(|v| match v {
                    Value::List(l) => Some(l),
                    _ => None,
                })
                .unwrap_or_default(),
        ),
        struct_tag::IGNORED => Response::Ignored,
        struct_tag::FAILURE => Response::Failure(
            fields
                .pop()
                .and_then(|v| match v {
                    Value::Map(m) => Some(m),
                    _ => None,
                })
                .unwrap_or_default(),
        ),
        other => panic!("unexpected response tag 0x{:02X}", other),
    }
}

async fn handshake(stream: &mut TcpStream) {
    // 0x6060B017 magic + Bolt 5.4 only.
    let bytes = [
        0x60, 0x60, 0xB0, 0x17, // magic
        0x00, 0x00, 0x04, 0x05, // 5.4
        0x00, 0x00, 0x00, 0x00, //
        0x00, 0x00, 0x00, 0x00, //
        0x00, 0x00, 0x00, 0x00, //
    ];
    stream.write_all(&bytes).await.expect("handshake send");
    let mut reply = [0u8; 4];
    stream.read_exact(&mut reply).await.expect("handshake read");
    assert_eq!(reply, [0, 0, 4, 5], "expected Bolt 5.4 negotiated");
}

async fn hello_and_logon(stream: &mut TcpStream, token: &str) {
    // HELLO {} (v5 carries no auth here)
    let hello = Value::Struct {
        tag: struct_tag::HELLO,
        fields: vec![Value::Map({
            let mut m = BTreeMap::new();
            m.insert("user_agent".into(), Value::String("test-driver/0".into()));
            m
        })],
    };
    send_msg(stream, &pack(&hello)).await;
    let r = recv_msg(stream).await;
    assert!(
        matches!(r, Response::Success(_)),
        "HELLO not SUCCESS: {r:?}"
    );

    // LOGON {scheme: "basic", credentials: $token}
    let logon = Value::Struct {
        tag: struct_tag::LOGON,
        fields: vec![Value::Map({
            let mut m = BTreeMap::new();
            m.insert("scheme".into(), Value::String("basic".into()));
            m.insert("principal".into(), Value::String("ignored".into()));
            m.insert("credentials".into(), Value::String(token.into()));
            m
        })],
    };
    send_msg(stream, &pack(&logon)).await;
    let r = recv_msg(stream).await;
    assert!(
        matches!(r, Response::Success(_)),
        "LOGON not SUCCESS: {r:?}"
    );
}

/// One row keyed by field name, decoded off the wire.
type RowMap = BTreeMap<String, Value>;

/// Send `RUN` + `PULL`, return `(fields_in_emission_order, rows)`.
/// Each row is decoded back into a `BTreeMap` keyed by field name so
/// callers don't need to depend on the alphabetical column ordering
/// the executor uses today.
async fn run_pull(stream: &mut TcpStream, cypher: &str) -> (Vec<String>, Vec<RowMap>) {
    let run = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String(cypher.into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    send_msg(stream, &pack(&run)).await;

    // Head SUCCESS { fields: [...] }.
    let fields = match recv_msg(stream).await {
        Response::Success(meta) => match meta.get("fields") {
            Some(Value::List(items)) => items
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => panic!("head SUCCESS missing fields list"),
        },
        other => panic!("expected head SUCCESS, got {other:?}"),
    };

    // RECORDs streamed by the server, terminated by the closing
    // SUCCESS after we send PULL.
    let mut rows: Vec<RowMap> = Vec::new();
    loop {
        let msg = recv_msg(stream).await;
        match msg {
            Response::Record(values) => {
                let mut row = BTreeMap::new();
                for (k, v) in fields.iter().cloned().zip(values.into_iter()) {
                    row.insert(k, v);
                }
                rows.push(row);
            }
            other => panic!("unexpected message during RUN stream: {other:?}"),
        }
        if rows.len() > 10_000 {
            panic!("runaway result set");
        }
        // No clean signal in the buffered model that the last RECORD
        // arrived, so request the close after each RECORD; the
        // server answers PULL with a closing SUCCESS that drops us
        // out of the loop.
        let pull = Value::Struct {
            tag: struct_tag::PULL,
            fields: vec![Value::Map({
                let mut m = BTreeMap::new();
                m.insert("n".into(), Value::Int(-1));
                m
            })],
        };
        send_msg(stream, &pack(&pull)).await;
        let closer = recv_msg(stream).await;
        match closer {
            Response::Success(meta) => {
                assert!(
                    meta.contains_key("type"),
                    "missing type meta in closing SUCCESS"
                );
                return (fields, rows);
            }
            Response::Record(values) => {
                let mut row = BTreeMap::new();
                for (k, v) in fields.iter().cloned().zip(values.into_iter()) {
                    row.insert(k, v);
                }
                rows.push(row);
            }
            other => panic!("unexpected closer: {other:?}"),
        }
    }
}

async fn goodbye(stream: &mut TcpStream) {
    let bye = Value::Struct {
        tag: struct_tag::GOODBYE,
        fields: vec![],
    };
    send_msg(stream, &pack(&bye)).await;
}

#[tokio::test]
async fn bolt_create_then_match_roundtrip() {
    // Bind on an ephemeral port.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .with_test_writer()
        .try_init();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bolt_addr = listener.local_addr().unwrap();
    drop(listener); // free the port; server will rebind it

    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);

    let config = namidb_server::Config {
        store_uri: "memory://bolt-test".into(),
        listen: http_addr,
        auth_token: Some("test-token".into()),
        flush_interval: Duration::ZERO,
        bolt_listen: Some(bolt_addr),
    };

    let server_task = tokio::spawn(async move {
        if let Err(e) = namidb_server::run(config).await {
            eprintln!("server exited: {e}");
        }
    });

    // Give the server a beat to bind both listeners. We could poll
    // /v0/health but the bolt path is what we want to drive.
    for _ in 0..50 {
        if TcpStream::connect(bolt_addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    // 1) CREATE: returns one row with name = "Alice".
    let (_fields, rows) = run_pull(
        &mut stream,
        "CREATE (a:Person {name: 'Alice', age: 30}) RETURN a.name AS name",
    )
    .await;
    assert_eq!(rows.len(), 1, "CREATE returned {} rows", rows.len());
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));

    // 2) MATCH the just-written node.
    let (fields, rows) = run_pull(
        &mut stream,
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age",
    )
    .await;
    assert_eq!(rows.len(), 1, "MATCH returned {} rows", rows.len());
    assert!(fields.iter().any(|f| f == "name"));
    assert!(fields.iter().any(|f| f == "age"));
    assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));
    assert_eq!(rows[0].get("age"), Some(&Value::Int(30)));

    goodbye(&mut stream).await;
    stream.shutdown().await.ok();

    server_task.abort();
}

#[tokio::test]
async fn bolt_bad_token_yields_failure() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bolt_addr = listener.local_addr().unwrap();
    drop(listener);

    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);

    let config = namidb_server::Config {
        store_uri: "memory://bolt-bad-auth".into(),
        listen: http_addr,
        auth_token: Some("correct-token".into()),
        flush_interval: Duration::ZERO,
        bolt_listen: Some(bolt_addr),
    };

    let server_task = tokio::spawn(async move {
        let _ = namidb_server::run(config).await;
    });

    for _ in 0..50 {
        if TcpStream::connect(bolt_addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect");
    handshake(&mut stream).await;

    // HELLO {} succeeds (v5 splits auth).
    let hello = Value::Struct {
        tag: struct_tag::HELLO,
        fields: vec![Value::Map(BTreeMap::new())],
    };
    send_msg(&mut stream, &pack(&hello)).await;
    let _ = recv_msg(&mut stream).await;

    // LOGON with wrong token: FAILURE.
    let logon = Value::Struct {
        tag: struct_tag::LOGON,
        fields: vec![Value::Map({
            let mut m = BTreeMap::new();
            m.insert("scheme".into(), Value::String("bearer".into()));
            m.insert("credentials".into(), Value::String("wrong".into()));
            m
        })],
    };
    send_msg(&mut stream, &pack(&logon)).await;
    let r = recv_msg(&mut stream).await;
    match r {
        Response::Failure(meta) => {
            let code = meta.get("code").cloned();
            assert_eq!(
                code,
                Some(Value::String(
                    "Neo.ClientError.Security.Unauthorized".into()
                )),
                "wrong error code: {code:?}"
            );
        }
        other => panic!("expected FAILURE, got {other:?}"),
    }

    server_task.abort();
}
