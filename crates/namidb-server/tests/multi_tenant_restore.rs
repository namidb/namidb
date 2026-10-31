//! `POST /:namespace/v0/admin/restore` end to end over a real multi-tenant
//! server: back up a namespace, restore it into a NEW namespace through the
//! HTTP API, and query the restored data. Also pins the refusals: an existing
//! or open target (409), a source outside `--backup-target-uri` (403), a
//! read-only token (403), a missing source (404), and single-tenant mode (no
//! route), plus the retry paths (an interrupted copy, a failed `verify`, a
//! backup missing an object) and, over a store that can hold a copy
//! mid-flight, the registry guard and a dropped request.
//!
//! The S3 variant is `#[ignore]` because it needs an S3-compatible endpoint
//! with conditional writes; it uses the same env as the storage crate's
//! `s3_integration` test:
//!
//! ```bash
//! AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… \
//! AWS_ENDPOINT_URL=http://127.0.0.1:4566 AWS_ALLOW_HTTP=true \
//! AWS_REGION=us-east-1 NAMIDB_TEST_BUCKET=namidb-tests \
//! cargo test -p namidb-server --test multi_tenant_restore -- --ignored
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use namidb_server::auth::AuthConfig;
use namidb_server::build_multi_tenant_router;
use namidb_server::memory::MemoryGovernor;
use namidb_server::metrics::Metrics;
use namidb_server::registry::{MaintenanceConfig, NamespaceRegistry};
use namidb_server::shared::SharedAppState;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use tokio::net::TcpStream;
use tower::ServiceExt;

const RW: &str = "restore-rw-token";
const RO: &str = "restore-ro-token";

fn config(
    store_uri: String,
    listen: std::net::SocketAddr,
    tokens_path: std::path::PathBuf,
    backup_target_uri: Option<String>,
    multi_tenant: bool,
) -> namidb_server::Config {
    namidb_server::Config {
        store_uri,
        listen,
        auth_token: None,
        auth_tokens_file: Some(tokens_path),
        auth_tokens_reload_interval: Duration::ZERO,
        no_auth: false,
        backup_target_uri,
        group_commit_window: Duration::ZERO,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval: Duration::ZERO,
        compaction_interval: Duration::ZERO,
        sweep_min_age: Duration::ZERO,
        sweep_delete: false,
        bolt_listen: None,
        bolt_max_message_bytes: 64 << 20,
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::from_secs(30),
        write_timeout: Duration::from_secs(30),
        query_row_cap: 0,
        compaction_l0_trigger: 0,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        memtable_flush_bytes: 0,
        memtable_stall_bytes: 0,
        writer_lock_timeout: Duration::from_secs(5),
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant,
        default_namespace: "default".to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
        fence_policy: Default::default(),
        http_request_timeout: None,
    }
}

/// Boot a server with a read-write and a read-only token and return its base
/// URL. The tokens file lives in `scratch`, which the caller owns.
async fn boot(
    scratch: &std::path::Path,
    store_uri: String,
    backup_target_uri: Option<String>,
    multi_tenant: bool,
) -> String {
    let tokens_path = scratch.join("tokens.json");
    std::fs::write(
        &tokens_path,
        serde_json::json!({ "tokens": [
            { "name": "rw", "token": RW, "role": "read-write" },
            { "name": "ro", "token": RO, "role": "read-only" },
        ] })
        .to_string(),
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let config = config(
        store_uri,
        addr,
        tokens_path,
        backup_target_uri,
        multi_tenant,
    );
    tokio::spawn(async move {
        if let Err(e) = namidb_server::run(config).await {
            eprintln!("server exited: {e}");
        }
    });
    for _ in 0..100 {
        if TcpStream::connect(addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    format!("http://{addr}")
}

async fn post(
    url: String,
    token: &str,
    namespace_header: Option<&str>,
    body: serde_json::Value,
) -> (u16, serde_json::Value) {
    let mut request = reqwest::Client::new()
        .post(url)
        .bearer_auth(token)
        .json(&body);
    if let Some(ns) = namespace_header {
        request = request.header("x-namidb-namespace", ns);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    let text = response.text().await.unwrap();
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
    (status, json)
}

async fn cypher(base: &str, ns: &str, query: &str) -> (u16, serde_json::Value) {
    post(
        format!("{base}/{ns}/v0/cypher"),
        RW,
        None,
        serde_json::json!({ "query": query }),
    )
    .await
}

async fn restore(base: &str, ns: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    post(format!("{base}/{ns}/v0/admin/restore"), RW, None, body).await
}

/// Names of the `:Item` nodes in `ns`, sorted.
async fn item_names(base: &str, ns: &str) -> Vec<String> {
    let (status, body) = cypher(
        base,
        ns,
        "MATCH (i:Item) RETURN i.name AS name ORDER BY name",
    )
    .await;
    assert_eq!(status, 200, "query {ns}: {body}");
    body["rows"]
        .as_array()
        .unwrap_or_else(|| panic!("no rows in {body}"))
        .iter()
        .map(|row| row["name"].as_str().unwrap().to_string())
        .collect()
}

/// The full scenario over one store/backup-prefix pair. `tag` keeps the
/// namespace names unique on a shared bucket.
async fn backup_restore_round_trip(base: &str, prefix: &str, tag: &str) {
    let src = format!("{tag}-src");
    // Flushed data (SSTs) plus a write that only lives in the WAL: the
    // restore must carry both.
    let (status, body) = cypher(
        base,
        &src,
        "CREATE (:Item {name: 'a'}), (:Item {name: 'b'})",
    )
    .await;
    assert_eq!(status, 200, "seed: {body}");
    let (status, body) = post(
        format!("{base}/{src}/v0/admin/flush"),
        RW,
        None,
        serde_json::json!({}),
    )
    .await;
    assert_eq!(status, 200, "flush: {body}");
    let (status, body) = cypher(base, &src, "CREATE (:Item {name: 'c'})").await;
    assert_eq!(status, 200, "wal-only write: {body}");

    let backup_uri = format!("{prefix}/{tag}-b1?ns={tag}-b1");
    let (status, body) = post(
        format!("{base}/{src}/v0/admin/backup"),
        RW,
        None,
        serde_json::json!({ "to": backup_uri }),
    )
    .await;
    assert_eq!(status, 200, "backup: {body}");

    // Restore into a namespace this server has never seen.
    let target = format!("{tag}-restored");
    let (status, body) = restore(
        base,
        &target,
        serde_json::json!({ "from": backup_uri, "verify": true }),
    )
    .await;
    assert_eq!(status, 200, "restore: {body}");
    assert!(body["objects_copied"].as_u64().unwrap() > 0, "{body}");
    assert!(body["bytes_copied"].as_u64().unwrap() > 0, "{body}");
    // The backup was renumbered to version 0; that is what restore copies.
    assert_eq!(body["source_version"].as_u64(), Some(0), "{body}");
    assert_eq!(item_names(base, &target).await, ["a", "b", "c"]);
    // The restored namespace is a live writer of its own, and the source is
    // untouched by writes to it.
    let (status, body) = cypher(base, &target, "CREATE (:Item {name: 'd'})").await;
    assert_eq!(status, 200, "write to restored: {body}");
    assert_eq!(item_names(base, &target).await, ["a", "b", "c", "d"]);
    assert_eq!(item_names(base, &src).await, ["a", "b", "c"]);

    // Open target: the registry refuses before any copy (its message, not
    // the pointer probe's).
    for open in [&target, &src] {
        let (status, body) = restore(base, open, serde_json::json!({ "from": backup_uri })).await;
        assert_eq!(status, 409, "restore over open {open}: {body}");
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("is open on this server"),
            "{body}"
        );
    }

    // Existing but never opened here: the copy refuses its manifest pointer.
    let cold = format!("{tag}-cold");
    let (status, body) = restore(
        base,
        &cold,
        serde_json::json!({ "from": backup_uri, "version": 0 }),
    )
    .await;
    assert_eq!(status, 200, "restore pinned version: {body}");
    let (status, body) = restore(base, &cold, serde_json::json!({ "from": backup_uri })).await;
    assert_eq!(status, 409, "restore over an existing pointer: {body}");
    assert!(
        body["error"].as_str().unwrap().contains("already exists"),
        "{body}"
    );
    assert_eq!(item_names(base, &cold).await, ["a", "b", "c"]);

    // The unprefixed form resolves the namespace from the header.
    let via_header = format!("{tag}-header");
    let (status, body) = post(
        format!("{base}/v0/admin/restore"),
        RW,
        Some(&via_header),
        serde_json::json!({ "from": backup_uri }),
    )
    .await;
    assert_eq!(status, 200, "unprefixed restore: {body}");
    assert_eq!(item_names(base, &via_header).await, ["a", "b", "c"]);

    // A source inside the allowlist that holds no backup (or not that
    // version) is a 404, and leaves the target openable.
    let missing = format!("{tag}-missing");
    let (status, body) = restore(
        base,
        &missing,
        serde_json::json!({ "from": format!("{prefix}/{tag}-nope?ns={tag}-nope") }),
    )
    .await;
    assert_eq!(status, 404, "missing source: {body}");
    let (status, body) = restore(
        base,
        &missing,
        serde_json::json!({ "from": backup_uri, "version": 7 }),
    )
    .await;
    assert_eq!(status, 404, "missing version: {body}");
    let (status, body) = restore(base, &missing, serde_json::json!({ "from": backup_uri })).await;
    assert_eq!(status, 200, "retry after a failed restore: {body}");
    assert_eq!(item_names(base, &missing).await, ["a", "b", "c"]);
}

/// The refusals that never reach the copy.
async fn gate_refusals(base: &str, prefix: &str, tag: &str) {
    let target = format!("{tag}-gated");
    let inside = format!("{prefix}/{tag}-b1?ns={tag}-b1");
    let src = format!("{tag}-src");

    // Outside the allowlist: a sibling that only shares a prefix, another
    // scheme, traversal out of the prefix (raw and percent-encoded), and
    // store options the prefix does not set (another endpoint, plain HTTP,
    // a second namespace).
    for outside in [
        format!("{prefix}-evil/{tag}-b1?ns={tag}-b1"),
        "file:///etc?ns=x".to_string(),
        format!("{prefix}/../data?ns={src}"),
        format!("{prefix}/{tag}-b1/../../data?ns={src}"),
        format!("{prefix}/%2e%2e/data?ns={src}"),
        format!("{prefix}/.%2E/data?ns={src}"),
        format!("{prefix}/./{tag}-b1?ns={tag}-b1"),
        format!("{prefix}//{tag}-b1?ns={tag}-b1"),
        format!("{prefix}/{tag}-b1?ns={tag}-b1&endpoint=http://127.0.0.1:9"),
        format!("{prefix}/{tag}-b1?ns={tag}-b1&allow_http=true"),
        format!("{prefix}/{tag}-b1?ns={tag}-b1&region=us-east-1"),
        format!("{prefix}/{tag}-b1?ns={tag}-b1&ns={src}"),
    ] {
        let (status, body) = restore(base, &target, serde_json::json!({ "from": outside })).await;
        assert_eq!(status, 403, "outside allowlist {outside}: {body}");
        assert!(
            body["error"].as_str().unwrap().contains("allowlist"),
            "{body}"
        );
        // Backup destinations share the boundary (`force` would otherwise
        // overwrite whatever the traversal reaches).
        let (status, body) = post(
            format!("{base}/{src}/v0/admin/backup"),
            RW,
            None,
            serde_json::json!({ "to": outside, "force": true }),
        )
        .await;
        assert_eq!(status, 403, "backup outside allowlist {outside}: {body}");
    }
    // Read-only token.
    let (status, body) = post(
        format!("{base}/{target}/v0/admin/restore"),
        RO,
        None,
        serde_json::json!({ "from": inside }),
    )
    .await;
    assert_eq!(status, 403, "read-only token: {body}");
    // Inside the allowlist but not a namespace URI.
    let (status, body) = restore(
        base,
        &target,
        serde_json::json!({ "from": format!("{prefix}/{tag}-b1") }),
    )
    .await;
    assert_eq!(status, 400, "source without ?ns=: {body}");
    // Invalid target namespace name.
    let (status, body) = restore(base, "bad%2Fname", serde_json::json!({ "from": inside })).await;
    assert_eq!(status, 400, "invalid namespace: {body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_into_new_namespace_on_file_store() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let backups = dir.path().join("backups");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&backups).unwrap();
    let prefix = format!("file://{}", backups.display());
    let base = boot(
        dir.path(),
        format!("file://{}", data.display()),
        Some(prefix.clone()),
        true,
    )
    .await;

    backup_restore_round_trip(&base, &prefix, "f").await;
    gate_refusals(&base, &prefix, "f").await;

    // The restored namespaces sit at the multi-tenant layout (`<root>/<ns>`),
    // where a fresh server over the same store finds them.
    for ns in ["f-restored", "f-cold", "f-header"] {
        assert!(data.join(ns).is_dir(), "{ns} missing under {data:?}");
    }

    let backup_uri = format!("{prefix}/f-b1?ns=f-b1");
    interrupted_restore_is_retried(&base, &data, &backup_uri).await;
    failed_restores_leave_a_retryable_target(&base, &data, &backups, &prefix).await;
}

/// Every file under `dir`, recursively (empty directories are not objects).
fn files_under(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files
}

/// A copy that stopped between its `v0.json` and its pointer (a crash, a
/// failed pointer PUT) leaves the objects without a pointer. A retry clears
/// them and succeeds, instead of colliding with the stray `v0.json`.
async fn interrupted_restore_is_retried(base: &str, data: &std::path::Path, backup_uri: &str) {
    let ns = "f-interrupted";
    let (status, body) = restore(base, ns, serde_json::json!({ "from": backup_uri })).await;
    assert_eq!(status, 200, "first restore: {body}");
    // Simulate the interruption on disk, before anything opens it.
    let manifest = data.join(ns).join("manifest");
    std::fs::remove_dir_all(manifest.join("pointer")).unwrap();
    std::fs::remove_file(manifest.join("current.json")).unwrap();
    assert!(
        files_under(&manifest).iter().any(|f| f
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with('v')),
        "the copied v0 body stays behind"
    );
    let stray = data.join(ns).join("sst").join("stray.bin");
    std::fs::create_dir_all(stray.parent().unwrap()).unwrap();
    std::fs::write(&stray, b"left by an older attempt").unwrap();

    let (status, body) = restore(base, ns, serde_json::json!({ "from": backup_uri })).await;
    assert_eq!(status, 200, "retry after an interrupted restore: {body}");
    assert!(!stray.exists(), "the leftovers are cleared before copying");
    assert_eq!(item_names(base, ns).await, ["a", "b", "c"]);
}

/// `verify` failing after the pointer landed, and a backup missing an object
/// (a NotFound mid-copy, which is a 500, not "source not found"), both leave
/// the target without any object, so the restore can be retried once the
/// backup is repaired.
async fn failed_restores_leave_a_retryable_target(
    base: &str,
    data: &std::path::Path,
    backups: &std::path::Path,
    prefix: &str,
) {
    let backup_uri = format!("{prefix}/f-b2?ns=f-b2");
    let (status, body) = post(
        format!("{base}/f-src/v0/admin/backup"),
        RW,
        None,
        serde_json::json!({ "to": backup_uri }),
    )
    .await;
    assert_eq!(status, 200, "backup: {body}");
    let wal = files_under(&backups.join("f-b2").join("f-b2").join("wal"));
    let segment = wal
        .first()
        .expect("the backup carries a WAL segment")
        .clone();
    let original = std::fs::read(&segment).unwrap();

    let ns = "f-verify";
    std::fs::write(&segment, b"").unwrap();
    let (status, body) = restore(
        base,
        ns,
        serde_json::json!({ "from": backup_uri, "verify": true }),
    )
    .await;
    assert_eq!(status, 409, "verify mismatch (as for backup): {body}");
    let error = body["error"].as_str().unwrap();
    assert!(error.contains("verify"), "{body}");
    assert!(error.contains("can be retried"), "{body}");
    assert_eq!(
        files_under(&data.join(ns)),
        Vec::<std::path::PathBuf>::new()
    );

    std::fs::remove_file(&segment).unwrap();
    let (status, body) = restore(base, ns, serde_json::json!({ "from": backup_uri })).await;
    assert_eq!(status, 500, "an object missing mid-copy: {body}");
    assert!(
        !body["error"]
            .as_str()
            .unwrap()
            .starts_with("restore source not found"),
        "not reported as a missing source: {body}"
    );
    assert_eq!(
        files_under(&data.join(ns)),
        Vec::<std::path::PathBuf>::new()
    );

    std::fs::write(&segment, original).unwrap();
    let (status, body) = restore(
        base,
        ns,
        serde_json::json!({ "from": backup_uri, "verify": true }),
    )
    .await;
    assert_eq!(status, 200, "retry after the backup is repaired: {body}");
    assert_eq!(item_names(base, ns).await, ["a", "b", "c"]);
}

/// Delegating store whose writes under `gate` block until [`Gate::open`]:
/// it holds a restore mid-copy so the test can probe the registry guard.
#[derive(Debug)]
struct GatedStore {
    inner: Arc<dyn ObjectStore>,
    gate: Arc<Gate>,
}

#[derive(Debug)]
struct Gate {
    prefix: String,
    open: tokio::sync::watch::Sender<bool>,
    blocked: AtomicUsize,
}

impl Gate {
    fn new(prefix: &str) -> Arc<Self> {
        Arc::new(Self {
            prefix: format!("{prefix}/"),
            open: tokio::sync::watch::channel(false).0,
            blocked: AtomicUsize::new(0),
        })
    }

    async fn pass(&self, location: &ObjectPath) {
        if !location.as_ref().starts_with(&self.prefix) {
            return;
        }
        let mut open = self.open.subscribe();
        if !*open.borrow() {
            self.blocked.fetch_add(1, Ordering::SeqCst);
        }
        open.wait_for(|open| *open).await.unwrap();
    }

    fn open(&self) {
        self.open.send_replace(true);
    }
}

impl std::fmt::Display for GatedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GatedStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for GatedStore {
    async fn put_opts(
        &self,
        location: &ObjectPath,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.gate.pass(location).await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectPath,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.gate.pass(location).await;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.gate.pass(to).await;
        self.inner.copy_opts(from, to, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
        self.inner.delete_stream(locations)
    }
}

/// Send one JSON POST through the router (open auth: no token needed).
async fn call(app: &axum::Router, path: &str, body: serde_json::Value) -> (u16, serde_json::Value) {
    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into()));
    (status, json)
}

/// While a restore copies, the registry guard (not the pointer probe: the
/// target has no pointer yet) refuses every open of the target with 503 and
/// a second restore with 409. The copy runs on its own task, so dropping the
/// request mid-copy does not abandon it: it finishes, releases its source
/// lease, and only then lets the namespace open, with the data.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restore_holds_the_target_and_survives_a_dropped_request() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let backups = dir.path().join("backups");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&backups).unwrap();
    let prefix = format!("file://{}", backups.display());

    let gate = Gate::new("held");
    let store: Arc<dyn ObjectStore> = Arc::new(GatedStore {
        inner: namidb_storage::parse_store(&format!("file://{}", data.display())).unwrap(),
        gate: gate.clone(),
    });
    let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
    let registry = Arc::new(NamespaceRegistry::new(
        store,
        String::new(),
        0,
        Duration::from_secs(3600),
        metrics.clone(),
        MaintenanceConfig::default(),
    ));
    let shared = SharedAppState::new_with_memory(
        registry,
        Arc::new(AuthConfig::open()),
        metrics,
        Duration::from_secs(30),
        Duration::from_secs(30),
        0,
        0,
        Duration::ZERO,
        0,
        0,
        Arc::new(MemoryGovernor::new(0)),
        Duration::from_secs(5),
        "default".to_string(),
    )
    .with_backup_target(Some(prefix.clone()));
    let app = build_multi_tenant_router(shared);

    let (status, body) = call(
        &app,
        "/src/v0/cypher",
        serde_json::json!({ "query": "CREATE (:Item {name: 'a'}), (:Item {name: 'b'})" }),
    )
    .await;
    assert_eq!(status, 200, "seed: {body}");
    let backup_uri = format!("{prefix}/b?ns=b");
    let (status, body) = call(
        &app,
        "/src/v0/admin/backup",
        serde_json::json!({ "to": backup_uri }),
    )
    .await;
    assert_eq!(status, 200, "backup: {body}");

    let restoring = tokio::spawn({
        let app = app.clone();
        let from = backup_uri.clone();
        async move {
            call(
                &app,
                "/held/v0/admin/restore",
                serde_json::json!({ "from": from }),
            )
            .await
        }
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        while gate.blocked.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the restore reaches its first target write");

    let (status, body) = call(
        &app,
        "/held/v0/cypher",
        serde_json::json!({ "query": "MATCH (i:Item) RETURN i.name AS name" }),
    )
    .await;
    assert_eq!(status, 503, "open during the restore: {body}");
    assert!(
        body["error"].as_str().unwrap().contains("being restored"),
        "{body}"
    );
    let (status, body) = call(
        &app,
        "/held/v0/admin/restore",
        serde_json::json!({ "from": backup_uri }),
    )
    .await;
    assert_eq!(status, 409, "second restore: {body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("already being restored"),
        "{body}"
    );

    // The client goes away mid-copy; the copy carries on once unblocked.
    restoring.abort();
    assert!(restoring.await.unwrap_err().is_cancelled());
    gate.open();

    let names = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let (status, body) = call(
                &app,
                "/held/v0/cypher",
                serde_json::json!({ "query": "MATCH (i:Item) RETURN i.name AS name ORDER BY name" }),
            )
            .await;
            if status == 200 {
                break body["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| row["name"].as_str().unwrap().to_string())
                    .collect::<Vec<_>>();
            }
            assert_eq!(status, 503, "only the guard may refuse: {body}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the abandoned restore completes and releases the target");
    assert_eq!(names, ["a", "b"]);
    // The source lease the copy held is released, not leaked.
    let pins = files_under(&backups.join("b").join("b").join("manifest").join("pins"));
    assert!(pins.is_empty(), "leaked retention leases: {pins:?}");
}

#[tokio::test]
async fn restore_is_disabled_without_backup_target() {
    let dir = tempfile::tempdir().unwrap();
    let base = boot(
        dir.path(),
        format!("file://{}", dir.path().join("data").display()),
        None,
        true,
    )
    .await;
    let (status, body) = restore(
        &base,
        "fresh",
        serde_json::json!({ "from": "file:///tmp/x?ns=x" }),
    )
    .await;
    assert_eq!(status, 403, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("--backup-target-uri"),
        "{body}"
    );
}

/// Single-tenant servers have no restore route: the namespace they serve is
/// always open, and restore never fences a live writer.
#[tokio::test]
async fn single_tenant_has_no_restore_route() {
    let dir = tempfile::tempdir().unwrap();
    let backups = dir.path().join("backups");
    std::fs::create_dir_all(&backups).unwrap();
    let base = boot(
        dir.path(),
        format!("file://{}?ns=solo", dir.path().join("data").display()),
        Some(format!("file://{}", backups.display())),
        false,
    )
    .await;
    let from = format!("file://{}/b?ns=b", backups.display());
    let (status, _) = post(
        format!("{base}/v0/admin/restore"),
        RW,
        None,
        serde_json::json!({ "from": from }),
    )
    .await;
    assert_eq!(status, 404);
    let (status, _) = restore(&base, "solo", serde_json::json!({ "from": from })).await;
    assert_eq!(status, 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn restore_into_new_namespace_on_s3() {
    let bucket = std::env::var("NAMIDB_TEST_BUCKET").expect("NAMIDB_TEST_BUCKET must be set");
    let dir = tempfile::tempdir().unwrap();
    // Unique per run: the bucket may be shared, and this test leaves its
    // namespaces and backups behind for inspection.
    let tag = format!(
        "it{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let prefix = format!("s3://{bucket}/restore-it/{tag}");
    let base = boot(
        dir.path(),
        format!("s3://{bucket}"),
        Some(prefix.clone()),
        true,
    )
    .await;
    backup_restore_round_trip(&base, &prefix, &tag).await;
    gate_refusals(&base, &prefix, &tag).await;
}
