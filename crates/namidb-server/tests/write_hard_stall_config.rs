//! The L0 hard write stall is armed only when compaction can drain L0 below
//! the bound by itself, judged from the resolved server [`Config`] (CLI flag
//! or env alike), not from raw environment variables.
//!
//! Each server below is configured so that no reactive pass can fire below
//! the bound. Flushes are driven through `/v0/admin/flush`, so L0 reaches the
//! bound deterministically; a held write would then wait the writer-lock
//! bound and answer 503 for as long as nothing compacts. With the stall off,
//! as in 3.5.4, the write is accepted.
//!
//! This file must stay a single-test integration binary: it sets the
//! process-global `NAMIDB_WRITE_HARD_STALL_L0` before the first server boots.

use std::time::Duration;

use tokio::net::TcpStream;

const TOKEN: &str = "test-token";
const HARD_L0: usize = 3;

async fn boot(
    ns: &str,
    flush_interval: Duration,
    compaction_interval: Duration,
    l0_trigger: usize,
) -> String {
    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);
    let config = namidb_server::Config {
        store_uri: format!("memory://{ns}"),
        listen: http_addr,
        auth_token: Some(TOKEN.into()),
        auth_tokens_file: None,
        auth_tokens_reload_interval: Duration::ZERO,
        no_auth: false,
        backup_target_uri: None,
        group_commit_window: Duration::ZERO,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        flush_interval,
        compaction_interval,
        sweep_min_age: Duration::ZERO,
        sweep_superseded_min_age: Duration::ZERO,
        sweep_delete: false,
        restore_point_retention: Duration::ZERO,
        bolt_listen: None,
        bolt_max_message_bytes: 64 << 20,
        bolt_tx_timeout: Duration::ZERO,
        query_timeout: Duration::from_secs(30),
        write_timeout: Duration::from_secs(30),
        query_row_cap: 0,
        compaction_l0_trigger: l0_trigger,
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        memtable_flush_bytes: 0,
        memtable_stall_bytes: 0,
        writer_lock_timeout: Duration::from_millis(600),
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: ns.to_string(),
        max_namespaces: 100,
        namespace_idle_timeout: Duration::from_secs(3600),
        fence_policy: Default::default(),
        http_max_body_bytes: namidb_server::DEFAULT_HTTP_CYPHER_BODY_LIMIT_BYTES,
        http_request_timeout: None,
    };
    tokio::spawn(async move {
        if let Err(e) = namidb_server::run(config).await {
            eprintln!("server exited: {e}");
        }
    });
    for _ in 0..200 {
        if TcpStream::connect(http_addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    format!("http://{http_addr}")
}

async fn post(base: &str, path: &str, body: Option<serde_json::Value>) -> (u16, String) {
    let mut request = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .bearer_auth(TOKEN);
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.text().await.unwrap())
}

async fn write(base: &str, k: usize) -> (u16, String) {
    post(
        base,
        "/v0/cypher",
        Some(serde_json::json!({ "query": format!("CREATE (:Doc {{k: {k}}}) RETURN 1 AS ok") })),
    )
    .await
}

/// Writes `HARD_L0` flushed L0 SSTs, then one more write: the one a hard
/// stall would hold.
async fn write_past_the_bound(base: &str) -> (u16, String) {
    for k in 0..HARD_L0 {
        let (status, body) = write(base, k).await;
        assert_eq!(status, 200, "write {k}: {body}");
        let (status, body) = post(base, "/v0/admin/flush", None).await;
        assert_eq!(status, 200, "flush {k}: {body}");
    }
    write(base, HARD_L0).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_hard_stall_is_off_when_nothing_would_drain_l0() {
    std::env::set_var("NAMIDB_WRITE_HARD_STALL_L0", HARD_L0.to_string());
    // Neither raw env var is set: the settings come from the Config, as from
    // `--compaction-l0-trigger` / `--compaction-interval` flags.
    std::env::remove_var("NAMIDB_COMPACTION_L0_TRIGGER");
    std::env::remove_var("NAMIDB_COMPACTION_INTERVAL");
    let hour = Duration::from_secs(3600);

    // Reactive and periodic compaction both off.
    let base = boot("stall-cfg-both-off", hour, Duration::ZERO, 0).await;
    let (status, body) = write_past_the_bound(&base).await;
    assert_eq!(status, 200, "both off: the write is accepted: {body}");

    // A reactive trigger above the hard bound never fires below it.
    let base = boot("stall-cfg-trigger-above", hour, Duration::ZERO, 100).await;
    let (status, body) = write_past_the_bound(&base).await;
    assert_eq!(status, 200, "trigger above the bound: accepted: {body}");

    // Only the periodic pass drains L0 (here an hour away): writes are not
    // refused until it runs.
    let base = boot("stall-cfg-periodic-only", hour, hour, 0).await;
    let (status, body) = write_past_the_bound(&base).await;
    assert_eq!(status, 200, "periodic only: accepted: {body}");

    // No flush loop: nothing requests a reactive pass either.
    let base = boot("stall-cfg-no-flush-loop", Duration::ZERO, Duration::ZERO, 2).await;
    let (status, body) = write_past_the_bound(&base).await;
    assert_eq!(status, 200, "no flush loop: accepted: {body}");
}
