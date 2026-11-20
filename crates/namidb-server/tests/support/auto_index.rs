//! Shared fixture of the automatic-property-index switch tests: a server
//! with background maintenance off, a label above the size threshold, and
//! statements whose `WHERE` equality has to scan because the property has no
//! index (the pattern that offers an automatic index).

#![allow(dead_code)]

use std::time::Duration;

use serde_json::{json, Value};
use tokio::net::TcpStream;

pub const NS: &str = "auto-index-switch";
/// Nodes of the fixture label; the tests lower the size threshold below it.
pub const NODES: i64 = 300;
/// The automatic index the fixture's statements ask for.
pub const AUTO_INDEX: &str = "auto_index_Item_level";

pub async fn boot(store_uri: String) -> String {
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
        // Flushes and compactions only when the test asks.
        flush_interval: Duration::ZERO,
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
        default_namespace: NS.to_string(),
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

pub async fn cypher(base: &str, query: &str, params: Value) -> Value {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/cypher"))
        .json(&json!({ "query": query, "params": params }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    let body: Value = response.json().await.unwrap();
    assert_eq!(status, 200, "{query}: {body}");
    body
}

pub async fn admin(base: &str, action: &str) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/admin/{action}"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200, "admin {action}");
}

/// Create the fixture label (full-text indexed), persist it, and run the
/// statements that report `(Item, level)` as an unindexed filter: a
/// `search.bm25` filter on a numeric property no index covers (numeric, so
/// an index on it would not change the search index's native filters).
pub async fn offer_hints(base: &str) {
    cypher(
        base,
        "CREATE FULLTEXT INDEX item_ft ON :Item(title)",
        json!({}),
    )
    .await;
    cypher(
        base,
        "UNWIND range(0, $n - 1) AS i CREATE (:Item {id: i, level: i % 5, \
         title: 'alpha item ' + toString(i)})",
        json!({ "n": NODES }),
    )
    .await;
    admin(base, "flush").await;
    for level in 0..5 {
        let body = cypher(
            base,
            "CALL search.bm25({label: 'Item', text_properties: ['title'], query: 'alpha', \
             k: 5, filter: {level: $level}}) YIELD node RETURN node.level AS level",
            json!({ "level": level }),
        )
        .await;
        let rows = body["rows"].as_array().unwrap();
        assert!(!rows.is_empty(), "{body}");
        assert!(
            rows.iter().all(|row| row["level"] == json!(level)),
            "{body}"
        );
    }
}

/// Names listed by `SHOW INDEXES`.
pub async fn index_names(base: &str) -> Vec<String> {
    let body = cypher(base, "SHOW INDEXES", json!({})).await;
    body["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|row| row["name"].as_str().map(str::to_string))
        .collect()
}

/// Value of `namidb_auto_index_total{outcome=...}` on `/v0/metrics`.
pub async fn auto_index_counter(base: &str, outcome: &str) -> u64 {
    let text = reqwest::get(format!("{base}/v0/metrics"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let prefix = format!("namidb_auto_index_total{{outcome=\"{outcome}\"}} ");
    text.lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("/v0/metrics lacks {prefix}"))
        .trim()
        .parse()
        .unwrap()
}

/// Poll until `done` holds or `within` elapses; returns whether it held.
pub async fn eventually<F, Fut>(within: Duration, mut done: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let started = std::time::Instant::now();
    loop {
        if done().await {
            return true;
        }
        if started.elapsed() >= within {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
