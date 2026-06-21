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
                for (k, v) in fields.iter().cloned().zip(values) {
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
                for (k, v) in fields.iter().cloned().zip(values) {
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

async fn begin(stream: &mut TcpStream) {
    let msg = Value::Struct {
        tag: struct_tag::BEGIN,
        fields: vec![Value::Map(BTreeMap::new())],
    };
    send_msg(stream, &pack(&msg)).await;
    match recv_msg(stream).await {
        Response::Success(_) => {}
        other => panic!("BEGIN expected SUCCESS, got {other:?}"),
    }
}

async fn commit(stream: &mut TcpStream) {
    let msg = Value::Struct {
        tag: struct_tag::COMMIT,
        fields: vec![],
    };
    send_msg(stream, &pack(&msg)).await;
    match recv_msg(stream).await {
        Response::Success(_) => {}
        other => panic!("COMMIT expected SUCCESS, got {other:?}"),
    }
}

async fn rollback(stream: &mut TcpStream) {
    let msg = Value::Struct {
        tag: struct_tag::ROLLBACK,
        fields: vec![],
    };
    send_msg(stream, &pack(&msg)).await;
    match recv_msg(stream).await {
        Response::Success(_) => {}
        other => panic!("ROLLBACK expected SUCCESS, got {other:?}"),
    }
}

/// Boot a server on ephemeral ports and return the bound Bolt address plus
/// the server task handle. Mirrors the boilerplate in the older tests.
async fn boot_bolt(
    ns: &str,
    tx_timeout: Duration,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    boot_bolt_full(ns, tx_timeout, Duration::ZERO).await
}

/// Like [`boot_bolt`] but also sets the per-read-query timeout.
async fn boot_bolt_full(
    ns: &str,
    tx_timeout: Duration,
    query_timeout: Duration,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bolt_addr = listener.local_addr().unwrap();
    drop(listener);
    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);

    let config = namidb_server::Config {
        store_uri: format!("memory://{ns}"),
        listen: http_addr,
        auth_token: Some("test-token".into()),
        auth_tokens_file: None,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: Some(bolt_addr),
        bolt_tx_timeout: tx_timeout,
        query_timeout,
        write_timeout: query_timeout,
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: ns.to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
    };
    let task = tokio::spawn(async move {
        if let Err(e) = namidb_server::run(config).await {
            eprintln!("server exited: {e}");
        }
    });
    for _ in 0..50 {
        if TcpStream::connect(bolt_addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (bolt_addr, task)
}

/// Boot a Bolt server whose auth comes from a JSON tokens file (per-token
/// roles). Writes `tokens_json` to a temp file the server reads at boot.
async fn boot_bolt_tokens(
    ns: &str,
    tokens_json: &str,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let tokens_path = std::env::temp_dir().join(format!("namidb-bolt-tokens-{ns}.json"));
    std::fs::write(&tokens_path, tokens_json).unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bolt_addr = listener.local_addr().unwrap();
    drop(listener);
    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);

    let config = namidb_server::Config {
        store_uri: format!("memory://{ns}"),
        listen: http_addr,
        auth_token: None,
        auth_tokens_file: Some(tokens_path),
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: Some(bolt_addr),
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::ZERO,
        write_timeout: Duration::ZERO,
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: ns.to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
    };
    let task = tokio::spawn(async move {
        if let Err(e) = namidb_server::run(config).await {
            eprintln!("server exited: {e}");
        }
    });
    for _ in 0..50 {
        if TcpStream::connect(bolt_addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (bolt_addr, task)
}

#[tokio::test]
async fn bolt_read_only_token_cannot_write() {
    let tokens = r#"{ "tokens": [
        { "name": "reader", "token": "rkey", "role": "read-only" },
        { "name": "writer", "token": "wkey", "role": "read-write" }
    ] }"#;
    let (bolt_addr, task) = boot_bolt_tokens("bolt-ro", tokens).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    // LOGON with the read-only token succeeds (the helper asserts SUCCESS), so
    // the token authenticates — reads would be served. A write must not be.
    hello_and_logon(&mut stream, "rkey").await;

    let run = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String("CREATE (:Person {name: 'x'})".into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    send_msg(&mut stream, &pack(&run)).await;
    match recv_msg(&mut stream).await {
        Response::Failure(meta) => assert_eq!(
            meta.get("code").cloned(),
            Some(Value::String("Neo.ClientError.Security.Forbidden".into())),
            "a read-only write must fail with the forbidden code, meta: {meta:?}"
        ),
        other => panic!("a read-only write must fail, got {other:?}"),
    }

    stream.shutdown().await.ok();
    task.abort();
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
        auth_tokens_file: None,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: Some(bolt_addr),
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::ZERO,
        write_timeout: Duration::ZERO,
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: "bolt-test".to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
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
        auth_tokens_file: None,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: Some(bolt_addr),
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::ZERO,
        write_timeout: Duration::ZERO,
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: "bolt-bad-auth".to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
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

/// Like [`run_pull`], but issues a single `PULL` and drains every
/// buffered `RECORD` up to the closing `SUCCESS`. Safe for result sets
/// of any size (including zero rows), where the per-record `PULL` loop
/// in `run_pull` would over-send.
async fn pull_all(stream: &mut TcpStream, cypher: &str) -> (Vec<String>, Vec<RowMap>) {
    let run = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String(cypher.into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    send_msg(stream, &pack(&run)).await;

    let fields = match recv_msg(stream).await {
        Response::Success(meta) => match meta.get("fields") {
            Some(Value::List(items)) => items
                .iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        },
        other => panic!("expected head SUCCESS, got {other:?}"),
    };

    let pull = Value::Struct {
        tag: struct_tag::PULL,
        fields: vec![Value::Map({
            let mut m = BTreeMap::new();
            m.insert("n".into(), Value::Int(-1));
            m
        })],
    };
    send_msg(stream, &pack(&pull)).await;

    let mut rows: Vec<RowMap> = Vec::new();
    loop {
        match recv_msg(stream).await {
            Response::Record(values) => {
                let mut row = BTreeMap::new();
                for (k, v) in fields.iter().cloned().zip(values) {
                    row.insert(k, v);
                }
                rows.push(row);
            }
            Response::Success(_) => return (fields, rows),
            other => panic!("unexpected message draining PULL: {other:?}"),
        }
    }
}

#[tokio::test]
async fn bolt_memgraph_introspection_populates_schema() {
    // A Memgraph-flavoured GUI (e.g. G.V()/gdotv) fires schema
    // procedures on connect. The Cypher parser has no CALL clause, so
    // the `introspect` shim must answer them from the live snapshot
    // before the parser would reject them as a syntax error.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bolt_addr = listener.local_addr().unwrap();
    drop(listener);

    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);

    let config = namidb_server::Config {
        store_uri: "memory://bolt-introspect".into(),
        listen: http_addr,
        auth_token: Some("test-token".into()),
        auth_tokens_file: None,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: Some(bolt_addr),
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::ZERO,
        write_timeout: Duration::ZERO,
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: "bolt-introspect".to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
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

    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    // Seed a small graph with ad-hoc (schemaless) properties and one
    // edge connecting two distinct labels.
    pull_all(&mut stream, "CREATE (a:Person {name: 'Alice', age: 30})").await;
    pull_all(&mut stream, "CREATE (c:Company {name: 'NamiDB'})").await;
    pull_all(
        &mut stream,
        "MATCH (a:Person {name:'Alice'}),(c:Company {name:'NamiDB'}) CREATE (a)-[:WORKS_AT]->(c)",
    )
    .await;

    // schema.node_type_properties(): one row per (label, property),
    // surfacing the sampled schemaless types.
    let (fields, rows) = pull_all(&mut stream, "CALL schema.node_type_properties() YIELD *").await;
    assert!(fields.iter().any(|f| f == "nodeLabels"));
    assert!(fields.iter().any(|f| f == "propertyName"));
    assert!(fields.iter().any(|f| f == "propertyTypes"));
    let person_name = rows
        .iter()
        .find(|r| {
            r.get("propertyName") == Some(&Value::String("name".into()))
                && matches!(
                    r.get("nodeLabels"),
                    Some(Value::List(l)) if l.contains(&Value::String("Person".into()))
                )
        })
        .expect("Person.name row present");
    assert_eq!(
        person_name.get("propertyTypes"),
        Some(&Value::String("String".into())),
    );

    // meta_util.schema(): a single row whose `schema` map carries the
    // node and relationship lists G.V() renders.
    let (_f, rows) = pull_all(&mut stream, "CALL meta_util.schema() YIELD *;").await;
    assert_eq!(rows.len(), 1, "meta_util.schema must return one row");
    let schema = match rows[0].get("schema") {
        Some(Value::Map(m)) => m,
        other => panic!("schema column not a map: {other:?}"),
    };
    let nodes = match schema.get("nodes") {
        Some(Value::List(l)) => l,
        other => panic!("nodes not a list: {other:?}"),
    };
    assert_eq!(nodes.len(), 2, "expected Person + Company node types");
    let rels = match schema.get("relationships") {
        Some(Value::List(l)) => l,
        other => panic!("relationships not a list: {other:?}"),
    };
    assert_eq!(rels.len(), 1, "expected one WORKS_AT edge type");

    // The edge's start/end must reference ids that exist in `nodes` so
    // the client can resolve endpoint labels.
    let node_ids: std::collections::BTreeSet<i64> = nodes
        .iter()
        .filter_map(|n| match n {
            Value::Map(m) => match m.get("id") {
                Some(Value::Int(i)) => Some(*i),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let rel = match &rels[0] {
        Value::Map(m) => m,
        other => panic!("rel not a map: {other:?}"),
    };
    for key in ["start", "end"] {
        match rel.get(key) {
            Some(Value::Int(i)) => assert!(node_ids.contains(i), "{key} {i} dangling"),
            other => panic!("{key} not an int: {other:?}"),
        }
    }
    assert_eq!(rel.get("label"), Some(&Value::String("WORKS_AT".into())));

    // rel_type_properties() lists the edge type.
    let (_f, rows) = pull_all(&mut stream, "CALL schema.rel_type_properties() YIELD *").await;
    assert!(
        rows.iter()
            .any(|r| r.get("relType") == Some(&Value::String(":`WORKS_AT`".into()))),
        "WORKS_AT not listed by rel_type_properties: {rows:?}"
    );

    goodbye(&mut stream).await;
    stream.shutdown().await.ok();
    server_task.abort();
}

#[tokio::test]
async fn bolt_rollback_discards_write() {
    let (bolt_addr, task) = boot_bolt("bolt-tx-rollback", Duration::ZERO).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    begin(&mut stream).await;
    // The write executes and returns its row inside the transaction.
    let (_f, rows) = pull_all(
        &mut stream,
        "CREATE (a:Person {name: 'Zoe'}) RETURN a.name AS name",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("name"), Some(&Value::String("Zoe".into())));

    rollback(&mut stream).await;

    // After ROLLBACK the write must be gone (this is what was broken: the
    // per-RUN commit made it durable before ROLLBACK was ever seen).
    let (_f, rows) = pull_all(
        &mut stream,
        "MATCH (p:Person {name: 'Zoe'}) RETURN p.name AS name",
    )
    .await;
    assert!(
        rows.is_empty(),
        "ROLLBACK must discard the write, got {rows:?}"
    );

    goodbye(&mut stream).await;
    stream.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_commit_persists_write() {
    let (bolt_addr, task) = boot_bolt("bolt-tx-commit", Duration::ZERO).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    begin(&mut stream).await;
    let (_f, rows) = pull_all(
        &mut stream,
        "CREATE (a:Person {name: 'Yan'}) RETURN a.name AS name",
    )
    .await;
    assert_eq!(rows.len(), 1);
    commit(&mut stream).await;

    let (_f, rows) = pull_all(
        &mut stream,
        "MATCH (p:Person {name: 'Yan'}) RETURN p.name AS name",
    )
    .await;
    assert_eq!(rows.len(), 1, "COMMIT must persist the write");
    assert_eq!(rows[0].get("name"), Some(&Value::String("Yan".into())));

    goodbye(&mut stream).await;
    stream.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_multi_statement_commit_is_atomic() {
    let (bolt_addr, task) = boot_bolt("bolt-tx-multi", Duration::ZERO).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    // Two statements in one transaction, then commit: both must land.
    begin(&mut stream).await;
    pull_all(&mut stream, "CREATE (a:Person {name: 'Ann'})").await;
    pull_all(&mut stream, "CREATE (a:Person {name: 'Ben'})").await;
    commit(&mut stream).await;

    let (_f, rows) = pull_all(&mut stream, "MATCH (p:Person) RETURN p.name AS name").await;
    assert_eq!(rows.len(), 2, "both committed creates must be visible");

    // A third create rolled back must NOT appear, and must not disturb the
    // two committed rows.
    begin(&mut stream).await;
    pull_all(&mut stream, "CREATE (a:Person {name: 'Cara'})").await;
    rollback(&mut stream).await;

    let (_f, rows) = pull_all(&mut stream, "MATCH (p:Person) RETURN p.name AS name").await;
    assert_eq!(rows.len(), 2, "rolled-back create must not appear");

    goodbye(&mut stream).await;
    stream.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_in_tx_read_sees_own_staged_write() {
    // RFC-026 read-your-own-writes: a read in statement N of a transaction
    // must see what statements 1..N-1 staged, while other sessions (on the
    // committed snapshot) must NOT see the uncommitted work.
    let (bolt_addr, task) = boot_bolt("bolt-tx-ryow", Duration::ZERO).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    begin(&mut stream).await;
    // Statement 1: stage a node, no commit.
    pull_all(&mut stream, "CREATE (a:Person {name: 'Uma'})").await;
    // Statement 2: a read in the SAME transaction sees the staged node.
    // Before read-your-own-writes this returned zero rows.
    let (_f, rows) = pull_all(
        &mut stream,
        "MATCH (p:Person {name: 'Uma'}) RETURN p.name AS name",
    )
    .await;
    assert_eq!(
        rows.len(),
        1,
        "an in-tx read must see the tx's own staged write"
    );
    assert_eq!(rows[0].get("name"), Some(&Value::String("Uma".into())));

    // Isolation: a second connection reads the committed snapshot and must
    // not observe the still-uncommitted node. (A read never needs the writer
    // lock that the open transaction holds.)
    let mut other = TcpStream::connect(bolt_addr).await.expect("connect other");
    handshake(&mut other).await;
    hello_and_logon(&mut other, "test-token").await;
    let (_f, rows2) = pull_all(&mut other, "MATCH (p:Person) RETURN p.name AS name").await;
    assert!(
        rows2.is_empty(),
        "an uncommitted staged write must not be visible to other sessions, got {rows2:?}"
    );
    goodbye(&mut other).await;
    other.shutdown().await.ok();

    commit(&mut stream).await;
    goodbye(&mut stream).await;
    stream.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_in_tx_read_sees_own_staged_edge() {
    // RFC-026 edge overlay: a traversal in statement N of a transaction must
    // see an edge that statements 1..N-1 staged, while another session on the
    // committed snapshot must not. This is the edge counterpart of
    // `bolt_in_tx_read_sees_own_staged_write`.
    let (bolt_addr, task) = boot_bolt("bolt-tx-ryow-edge", Duration::ZERO).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    begin(&mut stream).await;
    // Statement 1: stage two nodes and an edge between them, no commit.
    pull_all(
        &mut stream,
        "CREATE (a:Person {name: 'Uma'})-[:KNOWS]->(b:Person {name: 'Ivo'})",
    )
    .await;
    // Statement 2: a traversal in the SAME transaction follows the staged
    // edge. Before the edge overlay this returned zero rows.
    let (_f, rows) = pull_all(
        &mut stream,
        "MATCH (:Person {name: 'Uma'})-[:KNOWS]->(x) RETURN x.name AS name",
    )
    .await;
    assert_eq!(
        rows.len(),
        1,
        "an in-tx traversal must see the tx's own staged edge"
    );
    assert_eq!(rows[0].get("name"), Some(&Value::String("Ivo".into())));

    // Isolation: a second connection reads the committed snapshot and must
    // not observe the still-uncommitted edge (nor its endpoints).
    let mut other = TcpStream::connect(bolt_addr).await.expect("connect other");
    handshake(&mut other).await;
    hello_and_logon(&mut other, "test-token").await;
    let (_f, rows2) = pull_all(
        &mut other,
        "MATCH (:Person)-[:KNOWS]->(x) RETURN x.name AS name",
    )
    .await;
    assert!(
        rows2.is_empty(),
        "an uncommitted staged edge must not be visible to other sessions, got {rows2:?}"
    );
    goodbye(&mut other).await;
    other.shutdown().await.ok();

    commit(&mut stream).await;
    goodbye(&mut stream).await;
    stream.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_read_query_times_out() {
    // A 1ns read budget: the deadline (now + 1ns) is already past by the
    // time the executor reaches its first operator guard, so a read RUN
    // fails with a timeout instead of returning rows. Planning alone takes
    // far longer than a nanosecond, so this is deterministic.
    let (bolt_addr, task) =
        boot_bolt_full("bolt-qtimeout", Duration::ZERO, Duration::from_nanos(1)).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    let run = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String("MATCH (p:Person) RETURN p".into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    send_msg(&mut stream, &pack(&run)).await;
    let resp = recv_msg(&mut stream).await;
    assert!(
        matches!(resp, Response::Failure(_)),
        "a read past its 1ns budget must fail, got {resp:?}"
    );

    // Session is FAILED after the error; just drop the connection.
    stream.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_write_query_times_out() {
    // A 1ns write budget: the deadline (now + 1ns) is already past by the
    // time the write executor reaches its first per-row guard, so an
    // auto-commit write RUN fails with a timeout and commits nothing.
    // `boot_bolt_full` sets the write budget equal to the read budget.
    let (bolt_addr, task) =
        boot_bolt_full("bolt-wtimeout", Duration::ZERO, Duration::from_nanos(1)).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    let run = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String("CREATE (p:Person {name: 'Ada'})".into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    send_msg(&mut stream, &pack(&run)).await;
    let resp = recv_msg(&mut stream).await;
    assert!(
        matches!(resp, Response::Failure(_)),
        "a write past its 1ns budget must fail, got {resp:?}"
    );

    // Session is FAILED after the error; just drop the connection.
    stream.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_idle_transaction_times_out_and_releases_writer() {
    // A short idle timeout keeps the test fast.
    let (bolt_addr, task) = boot_bolt("bolt-tx-timeout", Duration::from_millis(300)).await;

    // Connection A: open a transaction, stage a write, then go idle. The
    // server must roll it back (releasing the writer) and fail it.
    let mut a = TcpStream::connect(bolt_addr).await.expect("connect a");
    handshake(&mut a).await;
    hello_and_logon(&mut a, "test-token").await;
    begin(&mut a).await;
    pull_all(&mut a, "CREATE (p:Person {name: 'Idle'})").await;
    let timed_out = recv_msg(&mut a).await;
    assert!(
        matches!(timed_out, Response::Failure(_)),
        "an idle open transaction should be failed by the server, got {timed_out:?}"
    );

    // Connection B: a WRITE must succeed — if A still held the writer lock
    // this would block forever — and only B's node exists (A's was rolled
    // back).
    let mut b = TcpStream::connect(bolt_addr).await.expect("connect b");
    handshake(&mut b).await;
    hello_and_logon(&mut b, "test-token").await;
    pull_all(&mut b, "CREATE (p:Person {name: 'After'})").await;
    let (_f, rows) = pull_all(&mut b, "MATCH (p:Person) RETURN p.name AS name").await;
    assert_eq!(
        rows.len(),
        1,
        "only the post-timeout write should exist, got {rows:?}"
    );
    assert_eq!(rows[0].get("name"), Some(&Value::String("After".into())));

    goodbye(&mut b).await;
    a.shutdown().await.ok();
    b.shutdown().await.ok();
    task.abort();
}

#[tokio::test]
async fn bolt_failed_in_tx_statement_releases_writer() {
    // A statement that fails inside an explicit transaction (here a parse
    // error; a mid-tx query/write timeout takes the same path) must not leave
    // the transaction pinning the global writer. No tx idle timeout is set, so
    // the writer has to be freed by the failure and the client's RESET — not by
    // the idle-timeout fallback that bolt_idle_transaction_* relies on.
    let (bolt_addr, task) = boot_bolt("bolt-tx-fail-release", Duration::ZERO).await;

    // Connection A: BEGIN, then run a statement that cannot parse. The in-tx
    // RUN fails and the session goes FAILED with the transaction still holding
    // the writer it took at BEGIN.
    let mut a = TcpStream::connect(bolt_addr).await.expect("connect a");
    handshake(&mut a).await;
    hello_and_logon(&mut a, "test-token").await;
    begin(&mut a).await;
    let bad = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String("this is not valid cypher !!!".into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    send_msg(&mut a, &pack(&bad)).await;
    let resp = recv_msg(&mut a).await;
    assert!(
        matches!(resp, Response::Failure(_)),
        "an unparseable in-tx statement must fail, got {resp:?}"
    );
    // RESET recovers the FAILED session to READY. The writer must already be
    // free; before the fix it stayed pinned for the life of the connection.
    let reset = Value::Struct {
        tag: struct_tag::RESET,
        fields: vec![],
    };
    send_msg(&mut a, &pack(&reset)).await;
    match recv_msg(&mut a).await {
        Response::Success(_) => {}
        other => panic!("RESET should recover to READY, got {other:?}"),
    }

    // Connection B: a WRITE must complete without blocking — if A still held
    // the writer this hangs until the harness kills the test. The aborted tx
    // staged nothing, so only B's node exists.
    let mut b = TcpStream::connect(bolt_addr).await.expect("connect b");
    handshake(&mut b).await;
    hello_and_logon(&mut b, "test-token").await;
    pull_all(&mut b, "CREATE (p:Person {name: 'After'})").await;
    let (_f, rows) = pull_all(&mut b, "MATCH (p:Person) RETURN p.name AS name").await;
    assert_eq!(
        rows.len(),
        1,
        "the aborted tx must have staged nothing, got {rows:?}"
    );
    assert_eq!(rows[0].get("name"), Some(&Value::String("After".into())));

    goodbye(&mut b).await;
    a.shutdown().await.ok();
    b.shutdown().await.ok();
    task.abort();
}

/// Run a statement and return the metadata of its closing `SUCCESS`
/// (after a single PULL) — e.g. to inspect the write `stats` map.
async fn run_capture_close(stream: &mut TcpStream, cypher: &str) -> BTreeMap<String, Value> {
    let run = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String(cypher.into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    send_msg(stream, &pack(&run)).await;
    match recv_msg(stream).await {
        Response::Success(_) => {}
        other => panic!("expected head SUCCESS, got {other:?}"),
    }
    let pull = Value::Struct {
        tag: struct_tag::PULL,
        fields: vec![Value::Map({
            let mut m = BTreeMap::new();
            m.insert("n".into(), Value::Int(-1));
            m
        })],
    };
    send_msg(stream, &pack(&pull)).await;
    loop {
        match recv_msg(stream).await {
            Response::Record(_) => {}
            Response::Success(meta) => return meta,
            other => panic!("unexpected message capturing close: {other:?}"),
        }
    }
}

#[tokio::test]
async fn bolt_neo4j_type_introspection_and_counters() {
    // The Neo4j connection type fires db.*, apoc.meta.* and SHOW, uses
    // elementId(), and expects write counters in the closing SUCCESS.
    let (bolt_addr, task) = boot_bolt("bolt-neo4j", Duration::ZERO).await;
    let mut stream = TcpStream::connect(bolt_addr).await.expect("connect bolt");
    handshake(&mut stream).await;
    hello_and_logon(&mut stream, "test-token").await;

    // A write reports counters in the closing SUCCESS `stats` map.
    let close = run_capture_close(
        &mut stream,
        "CREATE (a:Person {name: 'Alice', age: 30})-[:KNOWS]->(b:Person {name: 'Bob'})",
    )
    .await;
    let stats = match close.get("stats") {
        Some(Value::Map(m)) => m,
        other => panic!("write summary has no stats map: {other:?}"),
    };
    assert_eq!(stats.get("nodes-created"), Some(&Value::Int(2)));
    assert_eq!(stats.get("relationships-created"), Some(&Value::Int(1)));

    pull_all(&mut stream, "CREATE (c:Company {name: 'NamiDB'})").await;
    pull_all(
        &mut stream,
        "MATCH (a:Person {name:'Alice'}),(c:Company {name:'NamiDB'}) \
         CREATE (a)-[:WORKS_AT {role: 'founder'}]->(c)",
    )
    .await;

    // db.labels()
    let (_f, rows) = pull_all(&mut stream, "CALL db.labels()").await;
    assert!(rows
        .iter()
        .any(|r| r.get("label") == Some(&Value::String("Person".into()))));

    // apoc.meta.nodeTypeProperties(): APOC shape, propertyTypes is a list.
    let (fields, rows) = pull_all(&mut stream, "CALL apoc.meta.nodeTypeProperties()").await;
    assert!(fields.iter().any(|f| f == "nodeLabels"));
    let person_name = rows
        .iter()
        .find(|r| {
            r.get("propertyName") == Some(&Value::String("name".into()))
                && matches!(
                    r.get("nodeLabels"),
                    Some(Value::List(l)) if l.contains(&Value::String("Person".into()))
                )
        })
        .expect("Person.name row present");
    assert!(
        matches!(person_name.get("propertyTypes"), Some(Value::List(_))),
        "apoc propertyTypes must be a list, got {:?}",
        person_name.get("propertyTypes")
    );

    // apoc.meta.relTypeProperties(): endpoint labels populated.
    let (_f, rows) = pull_all(&mut stream, "CALL apoc.meta.relTypeProperties()").await;
    let works = rows
        .iter()
        .find(|r| r.get("relType") == Some(&Value::String(":`WORKS_AT`".into())))
        .expect("WORKS_AT row present");
    assert!(matches!(
        works.get("sourceNodeLabels"),
        Some(Value::List(l)) if l.contains(&Value::String("Person".into()))
    ));
    assert!(matches!(
        works.get("targetNodeLabels"),
        Some(Value::List(l)) if l.contains(&Value::String("Company".into()))
    ));

    // SHOW DATABASES resolves the default database name.
    let (_f, rows) = pull_all(&mut stream, "SHOW DATABASES").await;
    assert!(rows
        .iter()
        .any(|r| r.get("name") == Some(&Value::String("neo4j".into()))));

    // elementId() returns a non-empty id string (G.V() uses it to fetch
    // and expand nodes/edges).
    let (_f, rows) = pull_all(
        &mut stream,
        "MATCH (n:Person {name:'Alice'}) RETURN elementId(n) AS eid",
    )
    .await;
    assert!(
        matches!(rows.first().and_then(|r| r.get("eid")), Some(Value::String(s)) if !s.is_empty()),
        "elementId(n) should be a non-empty string: {rows:?}"
    );

    goodbye(&mut stream).await;
    stream.shutdown().await.ok();
    task.abort();
}
