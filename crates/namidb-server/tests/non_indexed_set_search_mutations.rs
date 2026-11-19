//! A bulk `SET` of scalars no search index covers must leave the search
//! indexes of the label alone: the flush that persists it is accounted as an
//! event proven empty for every index, with no full-text or vector mutation
//! and no new segment. A delta per such flush would carry every touched node
//! again, grow the indexes' delta debt without changing an answer, and on a
//! full-text index with an older base was what moved it off its native
//! reader.
#![cfg(all(feature = "text-index", feature = "vector-index"))]

use std::time::Duration;

use namidb_storage::search_lsm::SearchLsmKind;
use namidb_storage::ManifestStore;
use serde_json::{json, Value};
use tokio::net::TcpStream;

const NS: &str = "non-indexed-set";
const NODES: i64 = 400;

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

async fn cypher(base: &str, query: &str, params: Value) -> Value {
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

async fn admin(base: &str, action: &str) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/admin/{action}"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200, "admin {action}");
}

/// Per index: (generation, segment count, mutations, live payloads,
/// next event, proven-empty event count).
type Accounting = (uuid::Uuid, usize, u64, u64, u64, u64);

async fn accounting(manifests: &ManifestStore, kind: SearchLsmKind, index: &str) -> Accounting {
    let loaded = manifests.load_current().await.unwrap();
    let state = loaded
        .manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == kind && state.index_name == index)
        .unwrap_or_else(|| panic!("{index} has no Search-LSM generation"));
    (
        state.generation_id,
        state.segments.len(),
        state.segments.iter().map(|s| s.mutation_count).sum(),
        state.segments.iter().map(|s| s.live_payload_count).sum(),
        state.next_event_seq,
        state
            .proven_empty_event_ranges
            .iter()
            .map(|range| range.end - range.start)
            .sum(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bulk_set_of_non_indexed_scalars_creates_no_search_mutations() {
    let dir = tempfile::tempdir().unwrap();
    let store_uri = format!("file://{}?ns={NS}", dir.path().display());
    let base = boot(store_uri.clone()).await;
    cypher(
        &base,
        "CREATE FULLTEXT INDEX art_ft ON :Article(title, body)",
        json!({}),
    )
    .await;
    cypher(
        &base,
        "CREATE VECTOR INDEX art_vec ON :Article(embedding) METRIC cosine DIMENSION 4",
        json!({}),
    )
    .await;
    cypher(
        &base,
        "UNWIND range(0, $n - 1) AS i CREATE (:Article {id: i, \
         title: 'title ' + toString(i), body: 'graph storage body ' + toString(i % 17), \
         embedding: [toFloat(i % 7), toFloat(i % 5), 1.0, 0.5], flag: false, rank: 0})",
        json!({ "n": NODES }),
    )
    .await;
    admin(&base, "flush").await;
    admin(&base, "compact").await;

    let (store, paths) = namidb_storage::parse_uri(&store_uri).unwrap();
    let manifests = ManifestStore::new(store, paths);
    let text_before = accounting(&manifests, SearchLsmKind::Text, "art_ft").await;
    let vector_before = accounting(&manifests, SearchLsmKind::Vector, "art_vec").await;
    let search = "CALL search.bm25({label: 'Article', text_properties: ['title', 'body'], \
                  query: 'graph storage body', k: 5}) YIELD node, score \
                  RETURN node.id AS id, score";
    let vector = "CALL search.vector({label: 'Article', property: 'embedding', \
                  query: [3.0, 2.0, 1.0, 0.5], k: 5}) YIELD node, score \
                  RETURN node.id AS id, score";
    let text_answer = cypher(&base, search, json!({})).await["rows"].clone();
    let vector_answer = cypher(&base, vector, json!({})).await["rows"].clone();

    // The statement shape of a bulk scalar update.
    let rows = (0..NODES)
        .map(|id| json!({ "id": id, "v": id % 2 == 0, "j": id * 3 }))
        .collect::<Vec<_>>();
    cypher(
        &base,
        "UNWIND $rows AS r MATCH (a:Article {id: r.id}) SET a.flag = r.v, a.rank = r.j",
        json!({ "rows": rows }),
    )
    .await;
    admin(&base, "flush").await;

    for (kind, index, before) in [
        (SearchLsmKind::Text, "art_ft", text_before),
        (SearchLsmKind::Vector, "art_vec", vector_before),
    ] {
        let after = accounting(&manifests, kind, index).await;
        assert_eq!(after.0, before.0, "{index}: generation replaced");
        assert_eq!(
            (after.1, after.2, after.3),
            (before.1, before.2, before.3),
            "{index}: the non-indexed SET created search segments or mutations"
        );
        assert!(after.4 > before.4, "{index}: the flush was not accounted");
        assert_eq!(
            after.5 - before.5,
            after.4 - before.4,
            "{index}: every new event is proven empty"
        );
    }
    let updated = cypher(
        &base,
        "MATCH (a:Article) WHERE a.flag = true RETURN count(a) AS n",
        json!({}),
    )
    .await;
    assert_eq!(updated["rows"][0]["n"], NODES / 2);
    assert_eq!(cypher(&base, search, json!({})).await["rows"], text_answer);
    assert_eq!(
        cypher(&base, vector, json!({})).await["rows"],
        vector_answer
    );
}
