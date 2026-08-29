//! Item 40 (25 TB readiness), server half: a disk-full flush must degrade
//! the namespace into read-only with a typed error instead of wedging it.
//!
//! Contract pinned here, over the real HTTP surface:
//! 1. A flush that fails on local persistence answers 507 (not 500) and
//!    marks the namespace persistence-degraded.
//! 2. While degraded: writes are rejected 507 with the reason BEFORE
//!    queueing on the writer mutex; reads keep serving every acknowledged
//!    row; `/v0/health` reports the writer degraded with the reason.
//! 3. After the disk condition clears, a flush retry succeeds, clears the
//!    degraded state, and writes resume — no restart, nothing lost.
//!
//! This file must stay a single-test integration binary: it mutates the
//! process-global `NAMIDB_SPOOL_DIR`.

use std::time::Duration;

use tokio::net::TcpStream;

const NS: &str = "disk-full-degraded";
const TOKEN: &str = "test-token";

async fn boot() -> String {
    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);
    let config = namidb_server::Config {
        store_uri: format!("memory://{NS}"),
        listen: http_addr,
        auth_token: Some(TOKEN.into()),
        auth_tokens_file: None,
        auth_tokens_reload_interval: std::time::Duration::ZERO,
        no_auth: false,
        backup_target_uri: None,
        group_commit_window: Duration::ZERO,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        // No background flush: the test drives flushes via /v0/admin/flush
        // so every transition is deterministic.
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
        multi_tenant: false,
        default_namespace: NS.to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
    };
    tokio::spawn(async move {
        if let Err(e) = namidb_server::run(config).await {
            eprintln!("server exited: {e}");
        }
    });
    for _ in 0..100 {
        if TcpStream::connect(http_addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    format!("http://{http_addr}")
}

async fn cypher(base: &str, query: &str) -> (u16, String) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/cypher"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({ "query": query }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.text().await.unwrap())
}

async fn admin_flush(base: &str) -> (u16, String) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/admin/flush"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.text().await.unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disk_full_degrades_to_read_only_and_self_heals() {
    let base = boot().await;

    // Healthy intake.
    let (status, _) = cypher(&base, "CREATE (:Doc {k: 'antes'}) RETURN 1 AS ok").await;
    assert_eq!(status, 200);

    // Disk goes bad: the flush must fail typed, not wedge.
    std::env::set_var("NAMIDB_SPOOL_DIR", "/nonexistent/namidb-disk-full-test");
    let (status, body) = admin_flush(&base).await;
    assert_eq!(
        status, 507,
        "flush on a broken spool must answer 507: {body}"
    );
    assert!(body.contains("namespace degraded"), "typed body: {body}");

    // Writes: typed 507 with the reason, fast (no writer-mutex queueing).
    let (status, body) = cypher(&base, "CREATE (:Doc {k: 'durante'}) RETURN 1 AS ok").await;
    assert_eq!(status, 507, "writes while degraded must answer 507: {body}");
    assert!(body.contains("namespace degraded"), "typed body: {body}");

    // Reads: keep serving the acknowledged state.
    let (status, body) = cypher(&base, "MATCH (d:Doc) RETURN count(d) AS c").await;
    assert_eq!(
        status, 200,
        "reads must keep serving while degraded: {body}"
    );
    assert!(body.contains("\"c\":1"), "read must see the row: {body}");

    // Health surfaces the reason.
    let health = reqwest::Client::new()
        .get(format!("{base}/v0/health"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    let health_status = health.status().as_u16();
    let health_body = health.text().await.unwrap();
    assert_eq!(health_status, 503, "health must degrade: {health_body}");
    assert!(
        health_body.contains("flush failed on local persistence"),
        "health must carry the reason: {health_body}"
    );

    // Disk recovers: the retry clears the state without a restart.
    let spool = tempfile::tempdir().unwrap();
    std::env::set_var("NAMIDB_SPOOL_DIR", spool.path());
    let (status, body) = admin_flush(&base).await;
    assert_eq!(status, 200, "flush retry must succeed: {body}");

    let (status, body) = cypher(&base, "CREATE (:Doc {k: 'despues'}) RETURN 1 AS ok").await;
    assert_eq!(status, 200, "writes must resume after recovery: {body}");
    let (status, body) = cypher(&base, "MATCH (d:Doc) RETURN count(d) AS c").await;
    assert_eq!(status, 200);
    assert!(body.contains("\"c\":2"), "both rows must exist: {body}");

    let health = reqwest::Client::new()
        .get(format!("{base}/v0/health"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(health.status().as_u16(), 200, "health must clear");
}
