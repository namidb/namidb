//! A read on a slow store stops when its request goes away.
//!
//! A variable-length pattern into a bound endpoint that binds its path reads
//! the node reached at every hop with one point lookup, on the ranged route
//! once the node SST is large. The read runs on a thread of its own, and a
//! drop guard on the request raises its cancel flag when the client
//! disconnects or the HTTP request timeout fires; the walk only stops if the
//! lookups read that flag. Each case starts that read, ends the request
//! mid-query and checks that the read thread is gone within two seconds:
//!
//! 1. the HTTP client disconnects;
//! 2. the server's own HTTP request timeout fires;
//! 3. the Bolt client disconnects mid-RUN.
//!
//! Single-test binary: [`namidb_server::running_isolated_reads`] is
//! process-wide, and the HTTP request timeout is read from the environment
//! when a router is built.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::stream::BoxStream;
use futures::StreamExt;
use namidb_bolt::chunk::{read_message, write_message};
use namidb_bolt::codec::{decode, encode};
use namidb_bolt::message::{DEFAULT_POST_AUTH_MESSAGE_BYTES, POST_AUTH_MESSAGE_BYTES};
use namidb_bolt::value::{struct_tag, Value};
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// What one read costs once the store is slowed down.
const READ_LATENCY: Duration = Duration::from_millis(4);
/// The layered graph: `LAYERS` layers of `WIDTH` nodes, every node joined to
/// every node of the next layer.
const LAYERS: usize = 6;
const WIDTH: usize = 5;
/// Bytes of text per node, enough for the node SST to take the ranged route.
const BODY: usize = 1024 * 1024;
/// How long an abandoned read may keep its thread.
const STOP_WITHIN: Duration = Duration::from_secs(2);

/// Every path of up to ten hops between the first node and the far layer:
/// far more lookups than any of these cases waits for.
const TRAIL: &str = "MATCH (a:Doc {k: 0}), (b:Doc {k: 25}) \
                     MATCH q = (a)-[:LINKS*1..10]-(b) RETURN count(q) AS c";

/// Delays every read while `slow` is set, like a cold remote store.
#[derive(Debug)]
struct SlowReads {
    inner: Arc<dyn ObjectStore>,
    slow: Arc<AtomicBool>,
}

impl fmt::Display for SlowReads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SlowReads({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for SlowReads {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if self.slow.load(Ordering::Relaxed) {
            tokio::time::sleep(READ_LATENCY).await;
        }
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// Text that barely compresses, distinct per node.
fn body(k: usize) -> String {
    const ALPHABET: &[u8; 64] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";
    let mut x = (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..BODY)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            char::from(ALPHABET[(x % 64) as usize])
        })
        .collect()
}

/// The layered graph, flushed to one node SST above the ranged-read
/// threshold, on a store that is slow from then on.
async fn slow_graph() -> WriterSession {
    let slow = Arc::new(AtomicBool::new(false));
    let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store: Arc<dyn ObjectStore> = Arc::new(SlowReads {
        inner: Arc::clone(&inner),
        slow: Arc::clone(&slow),
    });
    let paths = NamespacePaths::new("tenants", NamespaceId::new("abandoned").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let mut ids = Vec::with_capacity(LAYERS * WIDTH);
    for k in 0..LAYERS * WIDTH {
        let id = NodeId::new();
        let mut properties = BTreeMap::new();
        properties.insert("k".to_string(), CoreValue::I64(k as i64));
        properties.insert("body".to_string(), CoreValue::Str(body(k)));
        writer
            .upsert_node_with_labels(
                ["Doc".to_string()],
                id,
                &NodeWriteRecord {
                    properties,
                    schema_version: 0,
                    labels: vec![],
                },
            )
            .unwrap();
        ids.push(id);
    }
    for layer in 0..LAYERS - 1 {
        for from in 0..WIDTH {
            for to in 0..WIDTH {
                writer
                    .upsert_edge(
                        "LINKS",
                        ids[layer * WIDTH + from],
                        ids[(layer + 1) * WIDTH + to],
                        &EdgeWriteRecord::default(),
                    )
                    .unwrap();
            }
        }
    }
    writer.commit_batch().await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("k", DataType::Int64, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let largest = inner
        .list(None)
        .map(|meta| meta.unwrap().size)
        .fold(0, |a, b| async move { a.max(b) })
        .await;
    assert!(
        largest >= namidb_storage::read::DEFAULT_RANGED_THRESHOLD_BYTES,
        "the node SST ({largest} bytes) must take the ranged route"
    );
    slow.store(true, Ordering::Relaxed);
    writer
}

async fn wait_for(what: &str, limit: Duration, mut done: impl FnMut() -> bool) -> Duration {
    let started = Instant::now();
    while !done() {
        assert!(
            started.elapsed() < limit,
            "{what} did not happen within {limit:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    started.elapsed()
}

/// Wait for the read to start, check that it really is long, and return.
async fn read_is_running(case: &str) {
    wait_for(
        &format!("{case}: the read to start"),
        Duration::from_secs(30),
        || namidb_server::running_isolated_reads() == 1,
    )
    .await;
    // Past the anchor lookups: the walk itself is running.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(
        namidb_server::running_isolated_reads(),
        1,
        "{case}: the read finished on its own; it cannot show a cancel"
    );
}

async fn read_stops(case: &str) {
    let stopped_in = wait_for(
        &format!("{case}: the abandoned read to stop"),
        STOP_WITHIN,
        || namidb_server::running_isolated_reads() == 0,
    )
    .await;
    eprintln!("{case}: the read stopped {stopped_in:?} after its request ended");
}

/// Serve the router on a listener of its own and return its address.
async fn serve_http(state: namidb_server::AppState) -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = namidb_server::build_router(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

async fn post(addr: std::net::SocketAddr, query: &str) -> TcpStream {
    let body = serde_json::to_vec(&serde_json::json!({ "query": query })).unwrap();
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let head = format!(
        "POST /v0/cypher HTTP/1.1\r\nhost: localhost\r\n\
         content-type: application/json\r\ncontent-length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(&body).await.unwrap();
    stream
}

/// A loopback address for the Bolt listener, which binds an address itself.
/// Reserved below the ephemeral range so no other socket takes it between
/// the check and the bind.
fn bolt_addr() -> std::net::SocketAddr {
    let offset = std::process::id() % 12_000;
    for n in 0..12_000 {
        let port = (32_000 - (offset + n) % 12_000) as u16;
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        if std::net::TcpListener::bind(addr).is_ok() {
            return addr;
        }
    }
    panic!("no free loopback port for Bolt");
}

fn pack(v: &Value) -> Vec<u8> {
    let mut buf = bytes::BytesMut::new();
    encode(&mut buf, v).expect("encode");
    buf.to_vec()
}

fn map(entries: &[(&str, &str)]) -> Value {
    Value::Map(
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect(),
    )
}

async fn expect_success(stream: &mut TcpStream, what: &str) {
    let body = read_message(stream, POST_AUTH_MESSAGE_BYTES)
        .await
        .expect("read_message");
    let mut slice: &[u8] = &body;
    match decode(&mut slice).expect("decode") {
        Value::Struct { tag, .. } if tag == struct_tag::SUCCESS => {}
        other => panic!("{what}: expected SUCCESS, got {other:?}"),
    }
}

/// Connect, negotiate Bolt 5.4, HELLO and LOGON.
async fn bolt_session(addr: std::net::SocketAddr, token: &str) -> TcpStream {
    let started = Instant::now();
    let mut stream = loop {
        match TcpStream::connect(addr).await {
            Ok(stream) => break stream,
            Err(error) => {
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "the Bolt listener did not come up: {error}"
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    };
    let handshake = [
        0x60, 0x60, 0xB0, 0x17, // magic
        0x00, 0x00, 0x04, 0x05, // 5.4
        0x00, 0x00, 0x00, 0x00, //
        0x00, 0x00, 0x00, 0x00, //
        0x00, 0x00, 0x00, 0x00, //
    ];
    stream.write_all(&handshake).await.unwrap();
    let mut reply = [0u8; 4];
    stream.read_exact(&mut reply).await.unwrap();
    assert_eq!(reply, [0, 0, 4, 5], "expected Bolt 5.4 negotiated");
    let hello = Value::Struct {
        tag: struct_tag::HELLO,
        fields: vec![map(&[("user_agent", "test-driver/0")])],
    };
    write_message(&mut stream, &pack(&hello)).await.unwrap();
    expect_success(&mut stream, "HELLO").await;
    let logon = Value::Struct {
        tag: struct_tag::LOGON,
        fields: vec![map(&[
            ("scheme", "basic"),
            ("principal", "ignored"),
            ("credentials", token),
        ])],
    };
    write_message(&mut stream, &pack(&logon)).await.unwrap();
    expect_success(&mut stream, "LOGON").await;
    stream
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_abandoned_read_on_a_slow_store_stops() {
    let writer = slow_graph().await;
    let state = namidb_server::AppState::new(writer, None, "abandoned".into())
        .with_query_timeout(Duration::ZERO)
        .with_query_row_cap(0);

    // 1. The HTTP client disconnects mid-query.
    let addr = serve_http(state.clone()).await;
    let client = post(addr, TRAIL).await;
    read_is_running("HTTP disconnect").await;
    drop(client);
    read_stops("HTTP disconnect").await;

    // 2. The server's own HTTP request timeout ends the request.
    std::env::set_var("NAMIDB_HTTP_REQUEST_TIMEOUT", "2500ms");
    let addr = serve_http(state.clone()).await;
    std::env::remove_var("NAMIDB_HTTP_REQUEST_TIMEOUT");
    let mut client = post(addr, TRAIL).await;
    read_is_running("HTTP request timeout").await;
    let mut head = [0u8; 12];
    tokio::time::timeout(Duration::from_secs(10), client.read_exact(&mut head))
        .await
        .expect("the request timeout must answer")
        .unwrap();
    assert_eq!(&head, b"HTTP/1.1 408", "{}", String::from_utf8_lossy(&head));
    read_stops("HTTP request timeout").await;
    drop(client);

    // 3. The Bolt client disconnects mid-RUN.
    let addr = bolt_addr();
    let (_shutdown, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(namidb_server::bolt::serve(
        state.clone(),
        addr,
        Arc::new(namidb_server::auth::AuthConfig::single_read_write("t")),
        Duration::ZERO,
        DEFAULT_POST_AUTH_MESSAGE_BYTES,
        shutdown_rx,
        None,
    ));
    let mut client = bolt_session(addr, "t").await;
    let run = Value::Struct {
        tag: struct_tag::RUN,
        fields: vec![
            Value::String(TRAIL.into()),
            Value::Map(BTreeMap::new()),
            Value::Map(BTreeMap::new()),
        ],
    };
    let pull = Value::Struct {
        tag: struct_tag::PULL,
        fields: vec![Value::Map(BTreeMap::from([(
            "n".to_string(),
            Value::Int(-1),
        )]))],
    };
    write_message(&mut client, &pack(&run)).await.unwrap();
    write_message(&mut client, &pack(&pull)).await.unwrap();
    read_is_running("Bolt disconnect").await;
    drop(client);
    read_stops("Bolt disconnect").await;
}
