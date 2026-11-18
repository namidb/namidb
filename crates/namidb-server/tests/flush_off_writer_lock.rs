//! A periodic flush must not hold the writer while it uploads SSTs.
//!
//! Before the off-lock flush, the flush task held the writer lock for the
//! whole SST build and upload. On a remote store with a large memtable that
//! is tens of seconds during which every write queued and then failed with
//! 503 at the writer-lock timeout, which turned a steady bulk load into
//! bursts and stalls. This test stalls every SST upload and checks that
//! writes keep committing meanwhile, and that the flush still lands with no
//! row lost once the uploads resume.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use tokio::net::TcpStream;

/// Holds every SST upload while `hold` is set.
#[derive(Debug)]
struct StallSstUploads {
    inner: Arc<dyn ObjectStore>,
    gate: Arc<Gate>,
}

#[derive(Debug, Default)]
struct Gate {
    hold: AtomicBool,
    stalled: AtomicUsize,
    released: tokio::sync::Notify,
}

impl Gate {
    async fn pass(&self, location: &Path) {
        if !location.as_ref().contains("/sst/") && !location.as_ref().starts_with("sst/") {
            return;
        }
        let mut counted = false;
        while self.hold.load(Ordering::SeqCst) {
            let released = self.released.notified();
            if !counted {
                self.stalled.fetch_add(1, Ordering::SeqCst);
                counted = true;
            }
            if !self.hold.load(Ordering::SeqCst) {
                break;
            }
            released.await;
        }
    }
}

impl fmt::Display for StallSstUploads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StallSstUploads({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for StallSstUploads {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.gate.pass(location).await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.gate.pass(location).await;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
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

/// The store layer is process-wide: tests that install one run one at a time.
static LAYER_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn boot(store_uri: String) -> String {
    boot_with_memtable(store_uri, 0, 0).await
}

async fn boot_with_memtable(store_uri: String, flush_bytes: usize, stall_bytes: usize) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let config = namidb_server::Config {
        store_uri,
        listen: addr,
        auth_token: None,
        auth_tokens_file: None,
        auth_tokens_reload_interval: Duration::ZERO,
        no_auth: true,
        backup_target_uri: None,
        group_commit_window: Duration::ZERO,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::from_millis(200),
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::from_secs(3600),
        sweep_superseded_min_age: Duration::from_secs(3600),
        sweep_delete: false,
        restore_point_retention: Duration::ZERO,
        bolt_listen: None,
        bolt_max_message_bytes: 64 << 20,
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::from_secs(30),
        write_timeout: Duration::from_secs(30),
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        memtable_flush_bytes: flush_bytes,
        memtable_stall_bytes: stall_bytes,
        // A write that has to wait for the writer this long fails with 503.
        writer_lock_timeout: Duration::from_secs(2),
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: "flush-off-lock".to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
        fence_policy: Default::default(),
        http_max_body_bytes: namidb_server::DEFAULT_HTTP_CYPHER_BODY_LIMIT_BYTES,
        http_request_timeout: None,
    };
    tokio::spawn(async move {
        if let Err(e) = namidb_server::run_with_memory_limits(config, 0, 0).await {
            eprintln!("server exited: {e}");
        }
    });
    for _ in 0..200 {
        if TcpStream::connect(addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    format!("http://{addr}")
}

async fn cypher(base: &str, query: &str) -> (u16, serde_json::Value) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/cypher"))
        .timeout(Duration::from_secs(20))
        .json(&serde_json::json!({ "query": query }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_commit_while_a_flush_upload_is_stalled() {
    let _serial = LAYER_LOCK.lock().await;
    let gate = Arc::new(Gate::default());
    {
        let gate = Arc::clone(&gate);
        namidb_storage::uri::set_store_layer(Some(Arc::new(move |inner| {
            Arc::new(StallSstUploads {
                inner,
                gate: Arc::clone(&gate),
            }) as Arc<dyn ObjectStore>
        })));
    }
    let base = boot("memory://flush-off-lock".to_string()).await;
    namidb_storage::uri::set_store_layer(None);

    gate.hold.store(true, Ordering::SeqCst);
    let (status, body) = cypher(&base, "UNWIND range(1, 50) AS i CREATE (:Row {i: i})").await;
    assert_eq!(status, 200, "{body}");
    // Wait for the periodic flush to reach its (stalled) SST upload.
    for _ in 0..200 {
        if gate.stalled.load(Ordering::SeqCst) > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(
        gate.stalled.load(Ordering::SeqCst) > 0,
        "the flush never started uploading"
    );

    // The flush is stuck uploading. Writes and reads must not wait for it.
    for round in 0..3 {
        let (status, body) = cypher(
            &base,
            &format!("UNWIND range(1, 10) AS i CREATE (:Row {{i: 1000 + i, round: {round}}})"),
        )
        .await;
        assert_eq!(status, 200, "write during a stalled flush: {body}");
    }
    let (status, body) = cypher(&base, "MATCH (r:Row) RETURN count(r) AS n").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rows"][0]["n"], 80);

    // Let the uploads through: the flush installs and nothing is lost.
    gate.hold.store(false, Ordering::SeqCst);
    gate.released.notify_waiters();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (status, body) = cypher(&base, "MATCH (r:Row) RETURN count(r) AS n").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rows"][0]["n"], 80);
    let (status, body) = cypher(&base, "CREATE (:Row {i: -1})").await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cypher(&base, "MATCH (r:Row) RETURN count(r) AS n").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rows"][0]["n"], 81);
}

/// Writes that keep committing during an off-lock flush must not grow the
/// memtable without bound: once it reaches the stall threshold, the flush
/// takes the writer back until it installs, so further writes wait (and fail
/// at the writer-lock timeout) as they did behind the synchronous flush. No
/// row is lost either way.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stalled_flush_bounds_the_memtable_at_the_stall_threshold() {
    let _serial = LAYER_LOCK.lock().await;
    let gate = Arc::new(Gate::default());
    {
        let gate = Arc::clone(&gate);
        namidb_storage::uri::set_store_layer(Some(Arc::new(move |inner| {
            Arc::new(StallSstUploads {
                inner,
                gate: Arc::clone(&gate),
            }) as Arc<dyn ObjectStore>
        })));
    }
    // Every commit is past the flush threshold; the stall threshold is a few
    // of the batches below.
    let base = boot_with_memtable("memory://flush-off-lock-bound".to_string(), 1, 32 * 1024).await;
    namidb_storage::uri::set_store_layer(None);

    gate.hold.store(true, Ordering::SeqCst);
    let (status, body) = cypher(&base, "UNWIND range(1, 50) AS i CREATE (:Row {i: i})").await;
    assert_eq!(status, 200, "{body}");
    for _ in 0..200 {
        if gate.stalled.load(Ordering::SeqCst) > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(
        gate.stalled.load(Ordering::SeqCst) > 0,
        "the flush never started uploading"
    );

    // Write while the upload is stuck until a write is refused.
    let mut committed = 50u64;
    let mut refused = false;
    for round in 0..200 {
        let (status, body) = cypher(
            &base,
            &format!(
                "UNWIND range(1, 50) AS i \
                 CREATE (:Row {{i: 1000 * {round} + i, pad: 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'}})"
            ),
        )
        .await;
        if status == 200 {
            committed += 50;
        } else {
            assert_eq!(status, 503, "{body}");
            refused = true;
            break;
        }
    }
    assert!(
        refused,
        "200 writes committed while the flush was stuck: the memtable grew without bound"
    );

    gate.hold.store(false, Ordering::SeqCst);
    gate.released.notify_waiters();
    let mut n = serde_json::Value::Null;
    for _ in 0..100 {
        let (status, body) = cypher(&base, "MATCH (r:Row) RETURN count(r) AS n").await;
        if status == 200 {
            n = body["rows"][0]["n"].clone();
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(n, committed, "every committed row survives the flush");
    let (status, body) = cypher(&base, "CREATE (:Row {i: -1})").await;
    assert_eq!(status, 200, "writes resume once the flush installs: {body}");
}
