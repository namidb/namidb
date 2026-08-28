//! Item 41 (25 TB readiness): the scan admission gate. With
//! `NAMIDB_MAX_CONCURRENT_SCANS=1`, many concurrent full-label scans must
//! all complete correctly (serialized through the gate, no deadlock, no
//! starvation), and non-scan traffic keeps flowing while scans queue.
//!
//! Single-test integration binary: it sets the process-global env knob
//! before the gate's lazy initialization.

use std::time::Duration;

use tokio::net::TcpStream;

const NS: &str = "scan-gate";
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
        no_auth: false,
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
    (response.status().as_u16(), response.text().await.unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_scans_serialize_through_the_gate_without_starvation() {
    // Before the gate's lazy init (first read query).
    std::env::set_var("NAMIDB_MAX_CONCURRENT_SCANS", "1");
    let base = boot().await;

    for chunk in 0..4 {
        let parts: Vec<String> = (0..250)
            .map(|i| format!("(:Person {{seq: {}}})", chunk * 250 + i))
            .collect();
        let (status, body) = cypher(&base, &format!("CREATE {}", parts.join(", "))).await;
        assert_eq!(status, 200, "{body}");
    }

    // Eight concurrent full scans through one permit: all must complete
    // with the exact count — serialized, not deadlocked, none starved.
    let mut handles = Vec::new();
    for _ in 0..8 {
        let base = base.clone();
        handles.push(tokio::spawn(async move {
            cypher(&base, "MATCH (p:Person) RETURN count(p) AS c").await
        }));
    }
    // Non-scan traffic keeps flowing while the scans queue.
    let (status, body) = cypher(&base, "RETURN 1 AS ok").await;
    assert_eq!(status, 200, "{body}");
    for handle in handles {
        let (status, body) = handle.await.unwrap();
        assert_eq!(status, 200, "{body}");
        assert!(
            body.contains("\"c\":1000"),
            "exact count under gating: {body}"
        );
    }
}
