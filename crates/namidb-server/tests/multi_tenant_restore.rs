//! `POST /:namespace/v0/admin/restore` end to end over a real multi-tenant
//! server: back up a namespace, restore it into a NEW namespace through the
//! HTTP API, and query the restored data. Also pins the refusals: an existing
//! or open target (409), a source outside `--backup-target-uri` (403), a
//! read-only token (403), a missing source (404), and single-tenant mode (no
//! route).
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

use std::time::Duration;

use tokio::net::TcpStream;

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

    // Open target: refused before any copy.
    let (status, body) = restore(base, &target, serde_json::json!({ "from": backup_uri })).await;
    assert_eq!(status, 409, "restore over an open namespace: {body}");
    let (status, body) = restore(base, &src, serde_json::json!({ "from": backup_uri })).await;
    assert_eq!(status, 409, "restore over the open source: {body}");

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

    // Outside the allowlist, including a sibling that only shares a prefix.
    for outside in [
        format!("{prefix}-evil/{tag}-b1?ns={tag}-b1"),
        "file:///etc?ns=x".to_string(),
    ] {
        let (status, body) = restore(base, &target, serde_json::json!({ "from": outside })).await;
        assert_eq!(status, 403, "outside allowlist {outside}: {body}");
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
