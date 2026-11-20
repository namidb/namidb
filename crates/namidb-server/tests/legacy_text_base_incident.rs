//! The 3.5.1 full-text regression, end to end through the HTTP server.
//!
//! A full-text index built by 3.4.x has a V3 base (the format before FT4).
//! The first 3.5.x flush appends an FT4 delta to it. Up to 3.5.1 no reader
//! accepted that mix: every `search.bm25` / `search.hybrid` scanned the corpus
//! (and hit the query timeout on a large one), `SHOW INDEXES` reported the
//! index BUILDING 0%, and nothing scheduled the rebuild that repairs it.
//!
//! This replays the sequence that left an index stuck — a V3 base, an FT4
//! delta, a bulk `SET` of non-indexed scalars, a `CREATE INDEX` on a string
//! property of the same label (which changes the text index's native-filter
//! set; since 3.5.2 it re-signs the generation in place, which keeps
//! serving), and a restart — and checks at every step that searches are
//! served natively unless the generation is really gone (then they fail
//! fast, naming the index), that `SHOW INDEXES` says so truthfully, and that
//! after the
//! restart every index is ONLINE on an FT4 base within bounded time, with
//! the same answers as before.
#![cfg(feature = "text-index")]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use namidb_core::id::NodeId;
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::search_lsm::{SearchLsmKind, SearchSegmentFormat};
use namidb_storage::{ManifestStore, NodeWriteRecord, WriterSession};
use serde_json::{json, Value};
use tokio::net::TcpStream;

const NS: &str = "legacy-text";
const TOKEN: &str = "test-token";
const ARTICLES: i64 = 300;
const NOTES: i64 = 120;
/// Below the Article count, so a scan standing in for a BUILDING index must
/// fail fast instead of answering.
const BUILDING_SCAN_BUDGET: usize = 100;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Article".into(),
            properties: vec![
                PropertyDef::new("id", DataType::Int64, true).unwrap(),
                PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("cat", DataType::Utf8, true).unwrap(),
                PropertyDef::new("flag", DataType::Bool, true).unwrap(),
                PropertyDef::new("rank", DataType::Int64, true).unwrap(),
            ],
        })
        .unwrap()
        .label(LabelDef {
            name: "Note".into(),
            properties: vec![
                PropertyDef::new("id", DataType::Int64, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

struct Words(u64);

impl Words {
    fn next(&mut self) -> String {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        const TOPICAL: [&str; 6] = ["graph", "storage", "object", "engine", "search", "index"];
        let pick = (self.0 >> 33) % 120;
        if pick < 6 {
            TOPICAL[pick as usize].to_string()
        } else {
            format!("w{}", pick - 6)
        }
    }

    fn text(&mut self, words: usize) -> String {
        (0..words)
            .map(|_| self.next())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn record(properties: Vec<(&str, CoreValue)>) -> NodeWriteRecord {
    NodeWriteRecord {
        properties: properties
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect::<BTreeMap<_, _>>(),
        schema_version: 1,
        ..Default::default()
    }
}

fn article(id: i64, words: &mut Words) -> NodeWriteRecord {
    record(vec![
        ("id", CoreValue::I64(id)),
        ("title", CoreValue::Str(words.text(4))),
        ("body", CoreValue::Str(words.text(30))),
        ("cat", CoreValue::Str(format!("c{}", id % 4))),
        ("flag", CoreValue::Bool(false)),
        ("rank", CoreValue::I64(0)),
    ])
}

fn note(id: i64, words: &mut Words) -> NodeWriteRecord {
    record(vec![
        ("id", CoreValue::I64(id)),
        ("body", CoreValue::Str(words.text(20))),
    ])
}

/// The store a 3.4.x engine left and a 3.5.x flush then wrote to: both
/// full-text indexes have a V3 base (built by a logical rebuild in a Nodes
/// merge right after `CREATE FULLTEXT INDEX`, as 3.4.x did) and one FT4
/// delta.
async fn seed_legacy_store(store_uri: &str) {
    let (store, paths) = namidb_storage::parse_uri(store_uri).unwrap();
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let mut words = Words(3);
    let mut article_ids = Vec::new();
    for id in 0..ARTICLES {
        let node = NodeId::new();
        writer
            .upsert_node("Article", node, &article(id, &mut words))
            .unwrap();
        article_ids.push(node);
    }
    let mut note_ids = Vec::new();
    for id in 0..NOTES {
        let node = NodeId::new();
        writer
            .upsert_node("Note", node, &note(id, &mut words))
            .unwrap();
        note_ids.push(node);
    }
    writer.flush(schema()).await.unwrap();
    for (name, label, properties) in [
        ("art_ft", "Article", vec!["title", "body"]),
        ("note_ft", "Note", vec!["body"]),
    ] {
        writer
            .register_text_index(
                TextIndexDescriptor::new(
                    name.into(),
                    label.into(),
                    properties.into_iter().map(String::from).collect(),
                ),
                false,
            )
            .await
            .unwrap();
    }
    writer.compact_l0(&schema()).await.unwrap();
    for name in ["art_ft", "note_ft"] {
        assert_eq!(
            text_formats(writer.current_manifest(), name),
            [SearchSegmentFormat::TextV3Base]
        );
    }

    // The first flush of the newer engine: a few updates, inserts and a
    // delete become one FT4 delta on each V3 base.
    for (offset, node) in article_ids.iter().take(3).enumerate() {
        let mut updated = article(offset as i64, &mut words);
        updated.properties.insert(
            "body".into(),
            CoreValue::Str(format!("graph storage engine {}", words.text(8))),
        );
        writer.upsert_node("Article", *node, &updated).unwrap();
    }
    for id in ARTICLES..ARTICLES + 2 {
        writer
            .upsert_node("Article", NodeId::new(), &article(id, &mut words))
            .unwrap();
    }
    writer.tombstone_node("Article", article_ids[10]).unwrap();
    writer
        .upsert_node("Note", note_ids[0], &note(0, &mut words))
        .unwrap();
    writer.flush(schema()).await.unwrap();
    for name in ["art_ft", "note_ft"] {
        assert_eq!(
            text_formats(writer.current_manifest(), name),
            [SearchSegmentFormat::TextV3Base, SearchSegmentFormat::TextV4]
        );
    }
}

fn text_formats(manifest: &namidb_storage::Manifest, index: &str) -> Vec<SearchSegmentFormat> {
    manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Text && state.index_name == index)
        .map(|state| {
            state
                .segments
                .iter()
                .map(|segment| segment.format)
                .collect()
        })
        .unwrap_or_default()
}

async fn boot(store_uri: String, maintenance: Option<Duration>) -> String {
    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_addr = http_listener.local_addr().unwrap();
    drop(http_listener);
    let interval = maintenance.unwrap_or(Duration::ZERO);
    let config = namidb_server::Config {
        store_uri,
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
        flush_interval: interval,
        compaction_interval: interval,
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

async fn cypher(base: &str, query: &str, params: Value) -> (u16, Value) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/cypher"))
        .bearer_auth(TOKEN)
        .json(&json!({ "query": query, "params": params }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    let body = response.text().await.unwrap();
    (
        status,
        serde_json::from_str(&body).unwrap_or(Value::String(body)),
    )
}

async fn admin(base: &str, action: &str) {
    let response = reqwest::Client::new()
        .post(format!("{base}/v0/admin/{action}"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200, "admin {action}");
}

/// `(state, populationPercent, migrationPercent)` per full-text index.
async fn show_fulltext(base: &str) -> BTreeMap<String, (String, f64, Option<f64>)> {
    let (status, body) = cypher(base, "SHOW INDEXES", json!({})).await;
    assert_eq!(status, 200, "{body}");
    let columns = body["columns"].as_array().unwrap();
    assert_eq!(
        columns.iter().rev().take(3).collect::<Vec<_>>(),
        ["migrationPercent", "populationPercent", "state"],
        "existing columns keep their order; the indicator trails them"
    );
    body["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["type"] == "FULLTEXT")
        .map(|row| {
            (
                row["name"].as_str().unwrap().to_string(),
                (
                    row["state"].as_str().unwrap().to_string(),
                    row["populationPercent"].as_f64().unwrap(),
                    row["migrationPercent"].as_f64(),
                ),
            )
        })
        .collect()
}

struct Search {
    label: &'static str,
    props: &'static [&'static str],
    query: &'static str,
    filter: Option<Value>,
}

fn searches() -> Vec<Search> {
    let article = |query, filter| Search {
        label: "Article",
        props: &["title", "body"],
        query,
        filter,
    };
    let note = |query| Search {
        label: "Note",
        props: &["body"],
        query,
        filter: None,
    };
    vec![
        article("graph storage", None),
        article("w1 w2 w3", None),
        article("\"storage engine\"", None),
        article("w1*", None),
        article("graph storage", Some(json!({ "cat": "c1" }))),
        article("engine w7", Some(json!({ "cat": ["c2", "c3"] }))),
        note("graph index"),
        note("w5 w9*"),
    ]
}

/// Ids and score bits, or the failure.
type Answer = Result<Vec<(i64, u64)>, (u16, String)>;

async fn run_search(base: &str, search: &Search) -> (Answer, Duration) {
    let statement = if search.filter.is_some() {
        "CALL search.bm25({label: $label, text_properties: $props, query: $q, k: 10, \
         filter: $filter}) YIELD node, score RETURN node.id AS id, score"
    } else {
        "CALL search.bm25({label: $label, text_properties: $props, query: $q, k: 10}) \
         YIELD node, score RETURN node.id AS id, score"
    };
    let params = json!({
        "label": search.label,
        "props": search.props,
        "q": search.query,
        "filter": search.filter.clone().unwrap_or(Value::Null),
    });
    let started = Instant::now();
    let (status, body) = cypher(base, statement, params).await;
    let elapsed = started.elapsed();
    if status != 200 {
        return (Err((status, body.to_string())), elapsed);
    }
    let rows = body["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            (
                row["id"].as_i64().unwrap(),
                row["score"].as_f64().unwrap().to_bits(),
            )
        })
        .collect();
    (Ok(rows), elapsed)
}

fn index_of(search: &Search) -> &'static str {
    match search.label {
        "Article" => "art_ft",
        _ => "note_ft",
    }
}

/// Every search, checked against what `SHOW INDEXES` says of its index: an
/// ONLINE index answers natively (the text route's fallback counter does not
/// move); a BUILDING one answers within the scan budget or fails at once
/// with an error naming the index and its state, never by running into the
/// query timeout. Returns the answers.
async fn searches_match_show(base: &str, step: &str) -> Vec<Answer> {
    let shown = show_fulltext(base).await;
    let mut answers = Vec::new();
    for search in searches() {
        let index = index_of(&search);
        let (state, percent, migration) = shown
            .get(index)
            .unwrap_or_else(|| panic!("{step}: {index} missing from SHOW INDEXES"))
            .clone();
        let fallbacks = namidb_storage::route_telemetry::snapshot().text_fallback;
        let (answer, elapsed) = run_search(base, &search).await;
        match state.as_str() {
            "ONLINE" => {
                assert_eq!(percent, 100.0, "{step}: {index}");
                assert!(answer.is_ok(), "{step}: {index} ONLINE but {answer:?}");
                assert_eq!(
                    namidb_storage::route_telemetry::snapshot().text_fallback,
                    fallbacks,
                    "{step}: {index} is ONLINE ({migration:?}) but {:?} used the scan",
                    search.query
                );
            }
            "BUILDING" => {
                assert!(migration.is_none(), "{step}: {index}");
                if let Err((status, message)) = &answer {
                    assert_eq!(*status, 503, "{step}: {message}");
                    assert!(
                        message.contains(index) && message.contains("BUILDING"),
                        "{step}: {message}"
                    );
                }
            }
            other => panic!("{step}: unexpected state {other}"),
        }
        assert!(
            elapsed < Duration::from_secs(10),
            "{step}: {:?} took {elapsed:?}",
            search.query
        );
        answers.push(answer);
    }
    answers
}

fn text_mutations(manifest: &namidb_storage::Manifest, index: &str) -> (usize, u64, u64) {
    let state = manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Text && state.index_name == index)
        .unwrap();
    (
        state.segments.len(),
        state
            .segments
            .iter()
            .map(|segment| segment.mutation_count)
            .sum(),
        state.next_event_seq,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_legacy_full_text_base_keeps_serving_through_the_incident_sequence() {
    std::env::set_var(
        "NAMIDB_SEARCH_BUILDING_SCAN_MAX_DOCS",
        BUILDING_SCAN_BUDGET.to_string(),
    );
    let dir = tempfile::tempdir().unwrap();
    let store_uri = format!("file://{}?ns={NS}", dir.path().display());
    seed_legacy_store(&store_uri).await;
    let (store, paths) = namidb_storage::parse_uri(&store_uri).unwrap();
    let manifests = ManifestStore::new(store, paths);

    // Step 1: the V3 base with an FT4 delta serves natively and is ONLINE,
    // with its migration pending. No maintenance runs in this server, so the
    // state stays put while it is checked.
    let base = boot(store_uri.clone(), None).await;
    let shown = show_fulltext(&base).await;
    for index in ["art_ft", "note_ft"] {
        assert_eq!(
            shown[index],
            ("ONLINE".to_string(), 100.0, Some(0.0)),
            "{index}"
        );
    }
    let before = searches_match_show(&base, "v3 base + ft4 delta").await;
    assert!(before.iter().all(Result::is_ok), "{before:?}");
    assert!(before[0].as_ref().unwrap().len() >= 3);

    // Step 2: a bulk SET of non-indexed scalars on the indexed label. It must
    // not touch the text index: no FT4 mutation, only an event proven empty.
    let loaded = manifests.load_current().await.unwrap();
    let (segments, mutations, events) = text_mutations(&loaded.manifest, "art_ft");
    let rows = (0..ARTICLES)
        .map(|id| json!({ "id": id, "v": id % 2 == 0, "j": id * 3 }))
        .collect::<Vec<_>>();
    let (status, body) = cypher(
        &base,
        "UNWIND $rows AS r MATCH (a:Article {id: r.id}) SET a.flag = r.v, a.rank = r.j",
        json!({ "rows": rows }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    // Unflushed, the rows carry the label but not a text change: still native.
    assert_eq!(
        searches_match_show(&base, "bulk SET, unflushed").await,
        before
    );
    admin(&base, "flush").await;
    let loaded = manifests.load_current().await.unwrap();
    let (after_segments, after_mutations, after_events) =
        text_mutations(&loaded.manifest, "art_ft");
    assert_eq!(
        (after_segments, after_mutations),
        (segments, mutations),
        "a non-indexed SET created full-text mutations"
    );
    assert!(
        after_events > events,
        "the flush is accounted as an empty event"
    );
    assert_eq!(
        searches_match_show(&base, "bulk SET, flushed").await,
        before
    );

    // Step 3: an equality index on a string property of the label changes the
    // text index's native-filter catalog. The generation is re-signed in
    // place and keeps serving every search with the same answers (a filter
    // on the new property is applied as a residual until the background
    // rebuild adds it); SHOW INDEXES keeps it ONLINE.
    let (status, body) = cypher(
        &base,
        "CREATE INDEX art_cat FOR (a:Article) ON (a.cat)",
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    for round in 0..8 {
        let answers = searches_match_show(&base, &format!("after CREATE INDEX #{round}")).await;
        assert_eq!(answers, before, "every index keeps serving");
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // Step 4: restart with maintenance on. Every index must come back ONLINE
    // on an FT4 base, migrations done, within bounded time, serving the same
    // answers, and stay consistent with SHOW INDEXES while it gets there. The
    // text never changed since step 1: the FT4 bases answer exactly what the
    // V3 bases and their delta answered, filters and scores included.
    let base = boot(store_uri.clone(), Some(Duration::from_secs(1))).await;
    assert_eq!(
        until_online_on_ft4(&base, &manifests, "after restart").await,
        before
    );

    // Step 5: the same sequence, restarted while the filter rebuild has not
    // run yet (the schema change committed, no compaction ran since). The
    // restarted server must rebuild it on its own, serving meanwhile, and end
    // ONLINE on FT4 bases with the same answers (up to the order of equal
    // scores: node ids differ in this second store).
    let dir = tempfile::tempdir().unwrap();
    let store_uri = format!("file://{}?ns={NS}", dir.path().display());
    seed_legacy_store(&store_uri).await;
    let (store, paths) = namidb_storage::parse_uri(&store_uri).unwrap();
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    writer
        .create_property_index_named(Some("art_cat"), "Article", "cat", false)
        .await
        .unwrap();
    let shown = namidb_storage::search_progress::index_status(
        writer.current_manifest(),
        SearchLsmKind::Text,
        "art_ft",
    );
    assert_eq!(
        shown.state,
        namidb_storage::search_progress::IndexState::Online,
        "the schema change keeps the full-text generation serving"
    );
    assert!(
        shown.migration_percent.is_some(),
        "and schedules its rebuild"
    );
    drop(writer);
    let manifests = ManifestStore::new(store, paths);
    let base = boot(store_uri, Some(Duration::from_secs(1))).await;
    let after = until_online_on_ft4(&base, &manifests, "restart before the rebuild").await;
    assert_eq!(after.len(), before.len());
    for (search, (left, right)) in searches().iter().zip(before.iter().zip(&after)) {
        let scores = |answer: &Answer| {
            let mut scores = answer
                .as_ref()
                .unwrap()
                .iter()
                .map(|(_, score)| *score)
                .collect::<Vec<_>>();
            scores.sort_unstable();
            scores
        };
        assert_eq!(scores(left), scores(right), "{:?}", search.query);
    }
}

/// Poll until both full-text indexes are ONLINE on FT4 bases with no
/// migration pending, checking every search against `SHOW INDEXES` on the
/// way and that `/v0/health` reports base builds. Returns the final answers.
async fn until_online_on_ft4(base: &str, manifests: &ManifestStore, step: &str) -> Vec<Answer> {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let health = reqwest::Client::new()
            .get(format!("{base}/v0/health"))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert!(
            health["search_builds_in_flight"].is_array(),
            "{step}: /v0/health lacks search_builds_in_flight: {health}"
        );
        // Every poll checks each search against what SHOW INDEXES says.
        searches_match_show(base, step).await;
        let shown = show_fulltext(base).await;
        let loaded = manifests.load_current().await.unwrap();
        let migrated = ["art_ft", "note_ft"].iter().all(|index| {
            shown[*index] == ("ONLINE".to_string(), 100.0, None)
                && text_formats(&loaded.manifest, index)
                    .first()
                    .is_some_and(|format| *format == SearchSegmentFormat::TextV4)
        });
        if migrated {
            // The answers above may predate the swap (a search that ran
            // while the index was still BUILDING); ask again now that both
            // are ONLINE, which they stay.
            return searches_match_show(base, step).await;
        }
        assert!(
            Instant::now() < deadline,
            "{step}: indexes not ONLINE on FT4 bases in time: {shown:?}"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
