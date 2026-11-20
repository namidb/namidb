//! Writes and DDL never stop search reads.
//!
//! `CREATE INDEX` on a Bool/String property of a label with a vector and a
//! full-text index changes their native-filter set. Up to 3.5.1 that retired
//! both generations: every search on the label fell back to a scan (on a
//! large label, the 503 `index_building` budget) until both were rebuilt
//! from scratch. Now the generations keep serving through the DDL, writes
//! keep reaching them, a background base rebuild adds the new filter and
//! swaps in, and `SHOW INDEXES` stays ONLINE throughout.
//!
//! A search loop with interleaved writes runs across the DDL until the swap
//! has landed, then a restart while a rebuild is pending; every request must
//! succeed with exact answers.
#![cfg(all(feature = "text-index", feature = "vector-index"))]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_storage::search_lsm::SearchLsmKind;
use namidb_storage::ManifestStore;
use serde_json::{json, Value};
use tokio::net::TcpStream;

const NS: &str = "filter-ddl";
const ARTICLES: usize = 1_500;
const DIM: usize = 16;
const CATS: [&str; 4] = ["c0", "c1", "c2", "c3"];

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
        flush_interval: Duration::from_millis(300),
        compaction_interval: Duration::from_secs(2),
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
        // A restart is a second server on the same store: the first one
        // steps aside instead of fencing it back.
        fence_policy: namidb_server::recovery::FencePolicy::Yield,
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

async fn cypher(base: &str, query: &str, params: Value) -> Result<Value, String> {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/cypher"))
        .json(&json!({ "query": query, "params": params }))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    let status = response.status().as_u16();
    let body = response.text().await.map_err(|error| error.to_string())?;
    if status != 200 {
        return Err(format!("{status}: {body}"));
    }
    serde_json::from_str(&body).map_err(|error| error.to_string())
}

async fn admin(base: &str, action: &str) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/admin/{action}"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200, "admin {action}");
}

fn embedding(seed: usize) -> Vec<f64> {
    let mut state = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

fn body(seed: usize) -> String {
    let words = ["graph", "storage", "engine", "index", "vector", "lexical"];
    format!(
        "{} {} w{} w{} doc{seed}",
        words[seed % words.len()],
        words[(seed / 7) % words.len()],
        seed % 31,
        seed % 17
    )
}

fn article(id: usize) -> Value {
    json!({
        "id": id as i64,
        "title": format!("title {}", id % 23),
        "body": body(id),
        "cat": CATS[id % CATS.len()],
        "embedding": embedding(id),
    })
}

async fn insert(base: &str, rows: Vec<Value>) -> Result<Value, String> {
    cypher(
        base,
        "UNWIND $rows AS r CREATE (:Article {id: r.id, title: r.title, body: r.body, \
         cat: r.cat, embedding: r.embedding})",
        json!({ "rows": rows }),
    )
    .await
}

/// `(state, migrationPercent)` per search index on `Article`.
async fn show(base: &str) -> Result<BTreeMap<String, (String, Option<f64>)>, String> {
    let body = cypher(base, "SHOW INDEXES", json!({})).await?;
    Ok(body["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["type"] == "FULLTEXT" || row["type"] == "VECTOR")
        .map(|row| {
            (
                row["name"].as_str().unwrap().to_string(),
                (
                    row["state"].as_str().unwrap().to_string(),
                    row["migrationPercent"].as_f64(),
                ),
            )
        })
        .collect())
}

fn ids_scores(body: &Value) -> Vec<(i64, u64)> {
    body["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            (
                row["id"].as_i64().unwrap(),
                row["score"].as_f64().unwrap().to_bits(),
            )
        })
        .collect()
}

/// One round of the loop: a write, then every search kind checked exactly.
/// Returns the first discrepancy.
async fn round(base: &str, ordinal: usize) -> Result<(), String> {
    // A write on the indexed label, visible to the very next search.
    let id = 1_000_000 + ordinal;
    let token = format!("fresh{ordinal}");
    let mut row = article(id);
    row["body"] = json!(format!("{token} {}", body(id)));
    insert(base, vec![row]).await?;

    let shown = show(base).await?;
    for index in ["art_ft", "art_vec"] {
        let (state, _) = shown
            .get(index)
            .ok_or_else(|| format!("{index} missing from SHOW INDEXES"))?;
        if state != "ONLINE" {
            return Err(format!("{index} is {state}"));
        }
    }

    let fresh = cypher(
        base,
        "CALL search.bm25({label: 'Article', text_properties: ['title', 'body'], query: $q, \
         k: 5}) YIELD node, score RETURN node.id AS id, score",
        json!({ "q": token }),
    )
    .await?;
    if ids_scores(&fresh).first().map(|(id, _)| *id) != Some(id as i64) {
        return Err(format!("bm25 misses the write {id}: {fresh}"));
    }

    // A filtered bm25 must be exactly the filtered prefix of the unfiltered
    // ranking: N, average length and df are corpus-wide either way.
    let cat = CATS[ordinal % CATS.len()];
    let query = ["graph storage", "engine w3", "index lexical w5"][ordinal % 3];
    let filtered = cypher(
        base,
        "CALL search.bm25({label: 'Article', text_properties: ['title', 'body'], query: $q, \
         k: 10, filter: {cat: $cat}}) YIELD node, score RETURN node.id AS id, score",
        json!({ "q": query, "cat": cat }),
    )
    .await?;
    let everything = cypher(
        base,
        "CALL search.bm25({label: 'Article', text_properties: ['title', 'body'], query: $q, \
         k: 100000}) YIELD node, score RETURN node.id AS id, node.cat AS cat, score",
        json!({ "q": query }),
    )
    .await?;
    let expected = everything["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["cat"] == cat)
        .take(10)
        .map(|row| {
            (
                row["id"].as_i64().unwrap(),
                row["score"].as_f64().unwrap().to_bits(),
            )
        })
        .collect::<Vec<_>>();
    let got = ids_scores(&filtered);
    // Equal scores may order differently between the two routes.
    let sorted = |mut rows: Vec<(i64, u64)>| {
        rows.sort_by_key(|(id, score)| (std::cmp::Reverse(*score), *id));
        rows
    };
    if got.len() != expected.len()
        || sorted(got.clone())
            .iter()
            .map(|(_, s)| s)
            .collect::<Vec<_>>()
            != sorted(expected.clone())
                .iter()
                .map(|(_, s)| s)
                .collect::<Vec<_>>()
    {
        return Err(format!(
            "filtered bm25 {query:?} cat={cat}: {got:?} != {expected:?}"
        ));
    }

    // A filtered vector search returns k nodes, every one matching.
    let vector = cypher(
        base,
        "CALL search.vector({label: 'Article', property: 'embedding', query: $q, k: 5, \
         filter: {cat: $cat}}) YIELD node, score RETURN node.id AS id, node.cat AS cat, score",
        json!({ "q": embedding(ordinal * 13), "cat": cat }),
    )
    .await?;
    let rows = vector["rows"].as_array().unwrap();
    if rows.len() != 5 || rows.iter().any(|row| row["cat"] != cat) {
        return Err(format!("filtered vector cat={cat}: {vector}"));
    }
    // The write reached the vector index too.
    let own = cypher(
        base,
        "CALL search.vector({label: 'Article', property: 'embedding', query: $q, k: 10}) \
         YIELD node, score RETURN node.id AS id, score",
        json!({ "q": embedding(id) }),
    )
    .await?;
    if !ids_scores(&own)
        .iter()
        .any(|(found, _)| *found == id as i64)
    {
        return Err(format!("vector misses the write {id}: {own}"));
    }

    // Hybrid with the filter: both legs, every result matching.
    let hybrid = cypher(
        base,
        "CALL search.hybrid({label: 'Article', query_text: $q, text_properties: ['title', \
         'body'], query_vector: $v, vector_property: 'embedding', k: 10, filter: {cat: $cat}}) \
         YIELD node, score RETURN node.cat AS cat",
        json!({ "q": query, "v": embedding(ordinal), "cat": cat }),
    )
    .await?;
    let rows = hybrid["rows"].as_array().unwrap();
    if rows.is_empty() || rows.iter().any(|row| row["cat"] != cat) {
        return Err(format!("filtered hybrid cat={cat}: {hybrid}"));
    }
    Ok(())
}

/// Per search index: generation id and whether every segment carries `cat`.
async fn generations(manifests: &ManifestStore) -> BTreeMap<String, (uuid::Uuid, bool)> {
    let loaded = manifests.load_current().await.unwrap();
    loaded
        .manifest
        .search_lsm
        .iter()
        .filter(|state| matches!(state.kind, SearchLsmKind::Text | SearchLsmKind::Vector))
        .map(|state| {
            (
                state.index_name.clone(),
                (
                    state.generation_id,
                    state.segments.iter().all(|segment| {
                        segment
                            .complete_filter_properties
                            .iter()
                            .any(|p| p == "cat")
                    }),
                ),
            )
        })
        .collect()
}

/// Run [`round`] until `stop`, counting rounds and failures.
fn search_loop(
    base: String,
    stop: Arc<AtomicBool>,
    first: usize,
) -> tokio::task::JoinHandle<(usize, Vec<String>)> {
    tokio::spawn(async move {
        let mut rounds = 0usize;
        let mut failures = Vec::new();
        while !stop.load(Ordering::Relaxed) {
            if let Err(failure) = round(&base, first + rounds).await {
                failures.push(failure);
            }
            rounds += 1;
        }
        (rounds, failures)
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_keeps_serving_exactly_through_a_filter_index_ddl() {
    let dir = tempfile::tempdir().unwrap();
    let store_uri = format!("file://{}?ns={NS}", dir.path().display());
    let base = boot(store_uri.clone()).await;
    cypher(
        &base,
        "CREATE VECTOR INDEX art_vec ON :Article(embedding) METRIC cosine DIMENSION 16",
        json!({}),
    )
    .await
    .unwrap();
    cypher(
        &base,
        "CREATE FULLTEXT INDEX art_ft ON :Article(title, body)",
        json!({}),
    )
    .await
    .unwrap();
    for chunk in (0..ARTICLES).collect::<Vec<_>>().chunks(500) {
        insert(&base, chunk.iter().map(|id| article(*id)).collect())
            .await
            .unwrap();
    }
    admin(&base, "flush").await;
    admin(&base, "compact").await;
    let (store, paths) = namidb_storage::parse_uri(&store_uri).unwrap();
    let manifests = ManifestStore::new(store, paths);
    let before = generations(&manifests).await;
    assert_eq!(before.len(), 2, "{before:?}");
    assert!(before.values().all(|(_, cat)| !cat));
    round(&base, 0).await.unwrap();

    // The DDL lands while the loop runs; the loop keeps going until the
    // rebuilt bases carrying the new filter have swapped in.
    let stop = Arc::new(AtomicBool::new(false));
    let looping = search_loop(base.clone(), stop.clone(), 1);
    tokio::time::sleep(Duration::from_millis(500)).await;
    cypher(
        &base,
        "CREATE INDEX art_cat FOR (a:Article) ON (a.cat)",
        json!({}),
    )
    .await
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(150);
    let mut saw_pending = false;
    loop {
        let shown = show(&base).await.unwrap();
        saw_pending |= shown.values().any(|(_, migration)| migration.is_some());
        assert!(
            shown.values().all(|(state, _)| state == "ONLINE"),
            "{shown:?}"
        );
        let now = generations(&manifests).await;
        if now.values().all(|(_, cat)| *cat)
            && shown.values().all(|(_, migration)| migration.is_none())
        {
            break;
        }
        assert!(Instant::now() < deadline, "rebuild not swapped in: {now:?}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    // A few more rounds on the swapped generations, then stop.
    tokio::time::sleep(Duration::from_secs(2)).await;
    stop.store(true, Ordering::Relaxed);
    let (rounds, failures) = looping.await.unwrap();
    assert!(
        failures.is_empty(),
        "{} of {rounds} rounds failed: {failures:#?}",
        failures.len()
    );
    assert!(rounds >= 3, "the loop ran only {rounds} rounds");
    let after = generations(&manifests).await;
    for (index, (generation, _)) in &before {
        assert_eq!(
            after[index].0, *generation,
            "{index}: the generation was replaced instead of rebuilt in place"
        );
    }
    let _ = saw_pending;

    // A uniqueness constraint that changes nothing for these filters, then a
    // second filter index followed at once by a restart: the restarted
    // server serves the re-signed generations and finishes the rebuild.
    cypher(
        &base,
        "CREATE CONSTRAINT art_id FOR (a:Article) REQUIRE a.id IS UNIQUE",
        json!({}),
    )
    .await
    .unwrap();
    round(&base, 10_000).await.unwrap();
    cypher(
        &base,
        "CREATE INDEX art_title FOR (a:Article) ON (a.title)",
        json!({}),
    )
    .await
    .unwrap();
    let base = boot(store_uri.clone()).await;
    let stop = Arc::new(AtomicBool::new(false));
    let looping = search_loop(base.clone(), stop.clone(), 20_000);
    let deadline = Instant::now() + Duration::from_secs(150);
    loop {
        let shown = show(&base).await.unwrap();
        assert!(
            shown.values().all(|(state, _)| state == "ONLINE"),
            "{shown:?}"
        );
        let loaded = manifests.load_current().await.unwrap();
        let swapped = loaded.manifest.search_lsm.iter().all(|state| {
            state.segments.iter().all(|segment| {
                segment
                    .complete_filter_properties
                    .iter()
                    .any(|p| p == "title")
            })
        });
        if swapped && shown.values().all(|(_, migration)| migration.is_none()) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "rebuild after restart not swapped in"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    stop.store(true, Ordering::Relaxed);
    let (rounds, failures) = looping.await.unwrap();
    assert!(
        failures.is_empty(),
        "{} of {rounds} rounds failed: {failures:#?}",
        failures.len()
    );
    let after_restart = generations(&manifests).await;
    for (index, (generation, _)) in &before {
        assert_eq!(after_restart[index].0, *generation, "{index}");
    }
}
