//! A variable-length pattern into a bound endpoint that binds its path, and
//! an `allShortestPaths` that binds its path, read the node reached at every
//! hop with one point lookup. The walk only probed the query's guards every
//! few thousand edges, so on a store where each lookup costs milliseconds it
//! ran minutes past its deadline, and kept running after the client had gone.
//! The point lookup now probes the guards itself: the walk stops within one
//! lookup of the deadline or of a cancel.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::stream::BoxStream;
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    execute_with_limits, lower, optimize, parse, ExecError, LogicalPlan, Params, StatsCatalog,
};
use namidb_storage::{
    EdgeWriteRecord, NamespacePaths, NodeWriteRecord, SessionCaches, WriterSession,
};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

/// How late the timeout may arrive.
const GRACE: Duration = Duration::from_secs(1);
/// What one read costs once the store is slowed down.
const READ_LATENCY: Duration = Duration::from_millis(4);
/// The layered graph: `LAYERS` layers of `WIDTH` nodes, every node joined to
/// every node of the next layer.
const LAYERS: usize = 6;
const WIDTH: usize = 5;
/// Bytes of text per node: what makes each lookup's page decode cost
/// milliseconds.
const BODY: usize = 64 * 1024;

/// Incompressible-ish text, distinct per node.
fn body(k: usize) -> String {
    let mut x = (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..BODY)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            char::from(b'a' + (x % 26) as u8)
        })
        .collect()
}

/// Delays every read while `slow` is set, like a cold remote store.
#[derive(Debug)]
struct SlowReads {
    inner: Arc<dyn ObjectStore>,
    slow: Arc<AtomicBool>,
    reads: Arc<AtomicUsize>,
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
            self.reads.fetch_add(1, Ordering::Relaxed);
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

struct Graph {
    writer: WriterSession,
    slow: Arc<AtomicBool>,
    reads: Arc<AtomicUsize>,
}

/// The layered graph, flushed and uncached. Its snapshots read node rows
/// with ranged requests, the route a large SST takes, so every point lookup
/// pays the store's latency.
async fn graph(name: &str) -> Graph {
    let slow = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicUsize::new(0));
    let store: Arc<dyn ObjectStore> = Arc::new(SlowReads {
        inner: Arc::new(InMemory::new()),
        slow: Arc::clone(&slow),
        reads: Arc::clone(&reads),
    });
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    // No caches: every lookup reads the store, as on a cold one.
    let mut writer = WriterSession::open_with_caches(store, paths, SessionCaches::none())
        .await
        .unwrap();
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
    Graph {
        writer,
        slow,
        reads,
    }
}

fn plan(query: &str) -> LogicalPlan {
    optimize(
        lower(&parse(query).unwrap()).unwrap(),
        &StatsCatalog::empty(),
    )
}

/// The far endpoint: the first node of the last layer.
fn far() -> i64 {
    ((LAYERS - 1) * WIDTH) as i64
}

/// A path-binding var-length count into a bound endpoint, its CALL-subquery
/// form, a bounded path listing, and an allShortestPaths with a path.
fn queries() -> Vec<String> {
    let t = far();
    vec![
        format!(
            "MATCH (a:Doc {{k: 0}}), (b:Doc {{k: {t}}}) \
             MATCH q = (a)-[:LINKS*1..10]-(b) RETURN count(q) AS c"
        ),
        format!(
            "MATCH (a:Doc {{k: 0}}), (b:Doc {{k: {t}}}) \
             CALL {{ WITH a, b MATCH q = (a)-[:LINKS*1..10]-(b) RETURN count(q) AS c }} \
             RETURN c"
        ),
        format!(
            "MATCH (a:Doc {{k: 0}}), (b:Doc {{k: {t}}}) \
             MATCH q = (a)-[:LINKS*1..10]-(b) RETURN length(q) AS l LIMIT 100000"
        ),
        format!(
            "MATCH (a:Doc {{k: 0}}), (b:Doc {{k: {t}}}) \
             MATCH p = allShortestPaths((a)-[:LINKS*..10]-(b)) RETURN count(*) AS n"
        ),
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_slow_path_walk_stops_at_its_deadline() {
    let g = graph("trail-deadline").await;
    let snap = g.writer.snapshot().with_ranged_reads(true);
    g.slow.store(true, Ordering::Relaxed);
    for query in queries() {
        let plan = plan(&query);
        let timeout = Duration::from_secs(1);
        let before = g.reads.load(Ordering::Relaxed);
        let start = Instant::now();
        let err = execute_with_limits(&plan, &snap, &Params::new(), Some(start + timeout), None)
            .await
            .expect_err(&query);
        let elapsed = start.elapsed();
        assert!(matches!(err, ExecError::Timeout), "{query}: {err:?}");
        assert!(
            elapsed < timeout + GRACE,
            "{query}: timed out after {elapsed:?}"
        );
        // The walk really was reading the store when the deadline hit.
        assert!(
            g.reads.load(Ordering::Relaxed) - before > 50,
            "{query}: too few reads to exercise the path"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_slow_path_walk_stops_when_cancelled() {
    let g = graph("trail-cancel").await;
    let snap = g.writer.snapshot().with_ranged_reads(true);
    g.slow.store(true, Ordering::Relaxed);
    for query in queries() {
        let plan = plan(&query);
        let flag = Arc::new(AtomicBool::new(false));
        let delay = Duration::from_millis(300);
        let flipper = {
            let flag = Arc::clone(&flag);
            std::thread::spawn(move || {
                std::thread::sleep(delay);
                flag.store(true, Ordering::Relaxed);
                Instant::now()
            })
        };
        let result = namidb_storage::cancel::with_cancel_flag(
            flag,
            execute_with_limits(&plan, &snap, &Params::new(), None, None),
        )
        .await;
        let stopped = Instant::now();
        let cancelled_at = flipper.join().unwrap();
        let err = result.expect_err(&query);
        assert!(
            matches!(
                err,
                ExecError::Cancelled | ExecError::Storage(namidb_storage::Error::Cancelled)
            ),
            "{query}: {err:?}"
        );
        assert!(
            stopped.duration_since(cancelled_at) < GRACE,
            "{query}: kept running {:?} after the cancel",
            stopped.duration_since(cancelled_at)
        );
    }
}
