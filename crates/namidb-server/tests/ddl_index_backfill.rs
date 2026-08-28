//! Item 38 (25 TB readiness): `CREATE CONSTRAINT` / `CREATE INDEX` on
//! ALREADY-LOADED data must materialize the posting sidecars without
//! waiting for the periodic compaction tick (or forever, when periodic
//! compaction is disabled).
//!
//! The compaction planner has recognized "this SST predates the current
//! indexed set" since 2.0.5 (`node_descriptor_needs_migration`); what was
//! missing is the trigger at DDL time. This test disables every periodic
//! maintenance loop so the ONLY thing that can rewrite the SSTs is the
//! DDL-requested pass, then watches the manifest (via a second store handle
//! on the same file:// directory) until the sidecars appear.

use std::time::Duration;

use namidb_storage::{ManifestStore, SstKind};
use tokio::net::TcpStream;

const NS: &str = "ddl-backfill";
const TOKEN: &str = "test-token";

async fn boot(store_uri: String) -> String {
    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);
    let config = namidb_server::Config {
        store_uri,
        listen: http_addr,
        auth_token: Some(TOKEN.into()),
        auth_tokens_file: None,
        no_auth: false,
        backup_target_uri: None,
        #[cfg(feature = "jwt")]
        jwt: None,
        #[cfg(feature = "pdp")]
        pdp_url: None,
        // Every periodic loop off: only the DDL trigger may compact.
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

/// For every Nodes SST in the current manifest: does each carry a complete
/// equality posting index for `property`? (None when there are no node SSTs.)
async fn all_node_ssts_cover(manifest_store: &ManifestStore, property: &str) -> Option<bool> {
    let loaded = manifest_store.load_current().await.unwrap();
    let node_descs: Vec<_> = loaded
        .manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes)
        .collect();
    if node_descs.is_empty() {
        return None;
    }
    Some(node_descs.iter().all(|d| {
        d.equality_property_indices
            .iter()
            .any(|idx| idx.property == property && idx.mixed_type_complete)
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_index_on_loaded_data_materializes_without_periodic_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let store_uri = format!("file://{}?ns={NS}", dir.path().display());
    let base = boot(store_uri.clone()).await;

    // Load 300 rows across two string properties, then flush: the SSTs are
    // born WITHOUT any posting sidecars (nothing is indexed yet).
    for chunk in 0..3 {
        let mut parts = Vec::new();
        for i in 0..100 {
            let n = chunk * 100 + i;
            parts.push(format!(
                "(:Person {{cedula: 'ced-{n:04}', ciudad: 'ciudad-{:02}'}})",
                n % 20
            ));
        }
        let (status, body) = cypher(&base, &format!("CREATE {}", parts.join(", "))).await;
        assert_eq!(status, 200, "seed chunk: {body}");
    }
    let flush = reqwest::Client::new()
        .post(format!("{base}/v0/admin/flush"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(flush.status().as_u16(), 200);

    let (store, paths) = namidb_storage::parse_uri(&store_uri).unwrap();
    let manifest_store = ManifestStore::new(store, paths);
    assert_eq!(
        all_node_ssts_cover(&manifest_store, "cedula").await,
        Some(false),
        "pre-DDL SSTs must not carry the sidecar (or the test proves nothing)"
    );

    // The DDL under test: a unique constraint and a plain index, on data
    // that is already flushed. Each must answer immediately (metadata-only)
    // and schedule the materializing pass itself.
    let (status, body) = cypher(
        &base,
        "CREATE CONSTRAINT persona_cedula IF NOT EXISTS \
         FOR (p:Person) REQUIRE p.cedula IS UNIQUE",
    )
    .await;
    assert_eq!(status, 200, "constraint DDL: {body}");
    let (status, body) = cypher(
        &base,
        "CREATE INDEX persona_ciudad IF NOT EXISTS FOR (p:Person) ON (p.ciudad)",
    )
    .await;
    assert_eq!(status, 200, "index DDL: {body}");

    // No periodic loops are running: only the DDL-triggered pass can do
    // this. Poll the manifest from the outside until both sidecars exist.
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        let cedula = all_node_ssts_cover(&manifest_store, "cedula").await;
        let ciudad = all_node_ssts_cover(&manifest_store, "ciudad").await;
        if cedula == Some(true) && ciudad == Some(true) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the DDL-triggered compaction must materialize both sidecars; \
             last state: cedula={cedula:?} ciudad={ciudad:?}"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // The materialized index answers correctly through the query surface.
    let (status, body) = cypher(
        &base,
        "MATCH (p:Person {cedula: 'ced-0123'}) RETURN p.ciudad AS ciudad",
    )
    .await;
    assert_eq!(status, 200);
    assert!(
        body.contains("ciudad-03"),
        "unique lookup must return the row: {body}"
    );

    // And the trigger is visible in metrics as its own kind.
    let metrics = reqwest::Client::new()
        .get(format!("{base}/v0/metrics"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let applied = metrics.lines().find(|l| {
        l.contains("namidb_compactions_total")
            && l.contains("trigger=\"ddl\"")
            && l.contains("status=\"applied\"")
    });
    assert!(
        applied.is_some_and(|l| !l.trim_end().ends_with(" 0")),
        "an applied ddl-triggered compaction must be counted; got: {applied:?}"
    );
}
