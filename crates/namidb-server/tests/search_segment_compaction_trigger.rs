//! A sustained load into a vector-indexed label must not wedge its flushes.
//!
//! Every periodic flush that touches an indexed label adds one search delta
//! segment. Up to 3.4.0, at `NAMIDB_SEARCH_LSM_MAX_SEGMENTS` (then 8) every
//! further flush failed until a compaction consolidated the deltas; an
//! off-lock flush now goes past it up to the protocol ceiling, so only the
//! compaction keeps the count bounded. A loader flushes
//! every few seconds while the periodic compaction runs every few minutes,
//! so the server must request a compaction when the deltas are due, not only
//! when L0 is deep. This test turns the L0 trigger and the periodic
//! compaction off: only the search-delta trigger of the (off-lock) periodic
//! flush can keep the flushes going.
#![cfg(feature = "vector-index")]

use std::time::Duration;

use tokio::net::TcpStream;

async fn boot(store_uri: String) -> String {
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
        flush_interval: Duration::from_millis(100),
        // No periodic compaction and no L0 trigger.
        compaction_interval: Duration::ZERO,
        compaction_l0_trigger: 0,
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
        write_stall_l0: 0,
        write_stall_delay: Duration::ZERO,
        memtable_flush_bytes: 0,
        memtable_stall_bytes: 0,
        writer_lock_timeout: Duration::from_secs(30),
        tls_cert: None,
        tls_key: None,
        slow_query_threshold: Duration::ZERO,
        multi_tenant: false,
        default_namespace: "search-trigger".to_string(),
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
        .timeout(Duration::from_secs(30))
        .json(&serde_json::json!({ "query": query }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

async fn memtable_bytes(base: &str) -> u64 {
    let health: serde_json::Value = reqwest::get(format!("{base}/v0/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    health["memtable_bytes"].as_u64().unwrap()
}

/// Wait until the periodic flush has drained the memtable.
async fn wait_flushed(base: &str, round: usize) {
    for _ in 0..100 {
        if memtable_bytes(base).await == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!(
        "round {round}: the periodic flush did not drain the memtable \
         (search delta segments at their cap and no compaction requested?)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flushes_keep_landing_past_the_search_segment_cap() {
    let base = boot("memory://search-trigger".to_string()).await;
    let (status, body) = cypher(
        &base,
        "CREATE VECTOR INDEX emb IF NOT EXISTS ON :Item(embedding) METRIC cosine DIMENSION 4",
    )
    .await;
    assert_eq!(status, 200, "{body}");

    // Three times the default cap of flushes, each adding one delta.
    for round in 0..24 {
        let (status, body) = cypher(
            &base,
            &format!(
                "UNWIND range(1, 5) AS i CREATE (:Item {{round: {round}, i: i, \
                 embedding: [toFloat(i), toFloat({round}), 1.0, 0.5]}})"
            ),
        )
        .await;
        assert_eq!(status, 200, "round {round}: {body}");
        wait_flushed(&base, round).await;
    }

    let (status, body) = cypher(&base, "MATCH (n:Item) RETURN count(n) AS n").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rows"][0]["n"], 120);
    let (status, body) = cypher(
        &base,
        "CALL search.vector({label: 'Item', property: 'embedding', \
         query: [5.0, 23.0, 1.0, 0.5], k: 3}) YIELD node, score \
         RETURN node.round AS round, node.i AS i ORDER BY score DESC",
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["rows"][0]["round"], 23, "{body}");
    assert_eq!(body["rows"][0]["i"], 5, "{body}");
}
