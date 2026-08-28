//! Item 39 (25 TB readiness): the server serves `EXPLAIN` instead of
//! silently executing, renders the OPTIMIZED plan against the real manifest
//! catalog (so `NodeByPropertyValue` is finally visible), and a `# route:`
//! footer states the physical access path — index, memtable, numeric-scan —
//! that was previously observable only through `elapsed_ms`.

use std::time::Duration;

use tokio::net::TcpStream;

const NS: &str = "explain-surface";
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
        backup_target_uri: None,
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

async fn person_count(base: &str) -> i64 {
    let (status, body) = cypher(base, "MATCH (p:Person) RETURN count(p) AS c").await;
    assert_eq!(status, 200, "{body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    parsed["rows"][0]["c"].as_i64().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explain_renders_the_plan_and_its_physical_route_without_executing() {
    let base = boot().await;

    let (status, body) = cypher(
        &base,
        "CREATE CONSTRAINT persona_cedula IF NOT EXISTS \
         FOR (p:Person) REQUIRE p.cedula IS UNIQUE",
    )
    .await;
    assert_eq!(status, 200, "{body}");

    // Seed into the memtable only (no flush yet).
    for chunk in 0..2 {
        let parts: Vec<String> = (0..50)
            .map(|i| {
                format!(
                    "(:Person {{cedula: 'ced-{:03}', seq: {}}})",
                    chunk * 50 + i,
                    chunk * 50 + i
                )
            })
            .collect();
        let (status, body) = cypher(&base, &format!("CREATE {}", parts.join(", "))).await;
        assert_eq!(status, 200, "{body}");
    }

    // EXPLAIN of a write must NOT execute.
    let before = person_count(&base).await;
    let (status, body) = cypher(&base, "EXPLAIN CREATE (:Person {cedula: 'fantasma'})").await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("\"plan\""), "plan rows expected: {body}");
    assert!(
        body.contains("Create"),
        "plan must show the operator: {body}"
    );
    assert_eq!(
        person_count(&base).await,
        before,
        "EXPLAIN of a write must create nothing"
    );

    // Memtable-only: the optimizer picks the index operator (schema says
    // unique) and the route footer says the store is memtable-backed.
    let (status, body) = cypher(
        &base,
        "EXPLAIN MATCH (p:Person {cedula: 'ced-007'}) RETURN p.seq AS seq",
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(
        body.contains("NodeByPropertyValue"),
        "the optimized plan must show the index operator: {body}"
    );
    assert!(
        body.contains("route: Person.cedula") && body.contains("memtable"),
        "memtable route note expected: {body}"
    );

    // Flushed with the constraint predating the flush: full sidecar
    // coverage, so the footer states the index route with the coverage.
    let flush = reqwest::Client::new()
        .post(format!("{base}/v0/admin/flush"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(flush.status().as_u16(), 200);
    let (status, body) = cypher(
        &base,
        "EXPLAIN MATCH (p:Person {cedula: 'ced-007'}) RETURN p.seq AS seq",
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(
        body.contains("→ index") && body.contains("unique lookup"),
        "index route note expected after flush: {body}"
    );

    // A numeric equality is not posting-indexed — the footer says so
    // instead of letting the operator name imply index speed.
    let (status, body) = cypher(
        &base,
        "CREATE INDEX persona_seq IF NOT EXISTS FOR (p:Person) ON (p.seq)",
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = cypher(&base, "EXPLAIN MATCH (p:Person {seq: 7}) RETURN p").await;
    assert_eq!(status, 200, "{body}");
    if body.contains("NodeByPropertyValue") {
        assert!(
            body.contains("numeric equality is not posting-indexed"),
            "numeric route caveat expected: {body}"
        );
    }

    // VERBOSE adds the estimate header.
    let (status, body) = cypher(
        &base,
        "EXPLAIN VERBOSE MATCH (p:Person {cedula: 'ced-007'}) RETURN p",
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("Estimated rows"), "verbose estimates: {body}");

    // The route counters exist on /v0/metrics, and running the real lookup
    // moves the native counter.
    let (status, body) = cypher(
        &base,
        "MATCH (p:Person {cedula: 'ced-007'}) RETURN p.seq AS seq",
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let metrics = reqwest::Client::new()
        .get(format!("{base}/v0/metrics"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let native = metrics
        .lines()
        .find(|l| l.starts_with("namidb_property_lookup_route_total{route=\"native\"}"))
        .expect("native property route counter must render");
    assert!(
        !native.trim_end().ends_with(" 0"),
        "an indexed lookup must count as native: {native}"
    );
    assert!(
        metrics.contains("namidb_property_lookup_route_total{route=\"fallback\"}"),
        "fallback series must render"
    );
}
