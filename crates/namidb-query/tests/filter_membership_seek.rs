//! A search filter whose equality/IN conjunct is served by a property index
//! resolves its membership set through that index instead of reading the
//! filter's columns over the whole label. The two routes must name exactly
//! the same nodes, whatever the order of creates, value moves, relabels,
//! deletes, flushes, merges and restarts, unflushed or flushed, and inside a
//! write transaction that sees its own staged rows. The index route must
//! decline (and the column read answer) for a filter with no indexed
//! equality conjunct and for a value that admits more nodes than its cap.
//! (Its own test binary: the route counters are process-wide.)

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    filter_membership_ids_for_tests, filter_membership_seek_budget,
    filter_membership_seek_budget_declines, filter_membership_seek_cache_hits,
    filter_membership_seek_ids_for_tests, filter_membership_seek_speculation,
    filter_membership_seeks, set_filter_membership_seek_cap_for_tests,
    set_filter_membership_seek_population_divisor, with_query_memory, QueryMemoryBudget,
    QueryMemoryPool, RuntimeValue,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, Snapshot, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const LABEL: &str = "Section";
const OTHER: &str = "Doc";
const PARENTS: usize = 9;

/// The route counters are process-wide: tests that read them run one at a
/// time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn schema() -> Schema {
    let props = || {
        vec![
            PropertyDef::new("parentId", DataType::Utf8, true)
                .unwrap()
                .with_indexed(true),
            PropertyDef::new("rank", DataType::Int64, true)
                .unwrap()
                .with_indexed(true),
            PropertyDef::new("level", DataType::Int64, true).unwrap(),
            PropertyDef::new("active", DataType::Bool, true).unwrap(),
        ]
    };
    SchemaBuilder::new()
        .label(LabelDef {
            name: LABEL.into(),
            properties: props(),
        })
        .unwrap()
        .label(LabelDef {
            name: OTHER.into(),
            properties: props(),
        })
        .unwrap()
        .build()
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

#[derive(Clone)]
struct Section {
    label: &'static str,
    parent: usize,
    rank: i64,
    level: i64,
    active: bool,
}

impl Section {
    fn record(&self) -> NodeWriteRecord {
        let mut properties = BTreeMap::new();
        properties.insert("parentId".into(), CoreValue::Str(parent(self.parent)));
        properties.insert("rank".into(), CoreValue::I64(self.rank));
        properties.insert("level".into(), CoreValue::I64(self.level));
        properties.insert("active".into(), CoreValue::Bool(self.active));
        NodeWriteRecord {
            properties,
            schema_version: 1,
            ..Default::default()
        }
    }
}

fn parent(n: usize) -> String {
    format!("p{n}")
}

fn map(entries: &[(&str, RuntimeValue)]) -> RuntimeValue {
    RuntimeValue::Map(
        entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect(),
    )
}

fn string(text: &str) -> RuntimeValue {
    RuntimeValue::String(text.into())
}

/// Whether the index route serves a filter.
#[derive(Clone, Copy, PartialEq)]
enum Route {
    /// Always, outside a write transaction (which keys numbers apart, so
    /// storage declines a numeric probe there).
    Index,
    /// When the property's postings can answer it.
    Either,
    /// Never: no indexed equality conjunct.
    Columns,
}

/// `(filter, how it is served)`.
fn filters() -> Vec<(RuntimeValue, Route)> {
    let lte = |n: i64| map(&[("lte", RuntimeValue::Integer(n))]);
    vec![
        (map(&[("parentId", string("p1"))]), Route::Index),
        (map(&[("parentId", string("p4"))]), Route::Index),
        // A value no node has.
        (map(&[("parentId", string("p-none"))]), Route::Index),
        // IN, a bare list and an explicit `in`.
        (
            map(&[(
                "parentId",
                RuntimeValue::List(vec![string("p2"), string("p3"), string("p-none")]),
            )]),
            Route::Index,
        ),
        (
            map(&[(
                "parentId",
                map(&[("in", RuntimeValue::List(vec![string("p0"), string("p5")]))]),
            )]),
            Route::Index,
        ),
        // A residual conjunct the index cannot decide.
        (
            map(&[
                ("parentId", string("p2")),
                ("active", RuntimeValue::Bool(true)),
            ]),
            Route::Index,
        ),
        (
            map(&[("parentId", string("p6")), ("level", lte(2))]),
            Route::Index,
        ),
        // Two indexed conjuncts.
        (
            map(&[
                ("parentId", string("p3")),
                ("rank", RuntimeValue::Integer(3)),
            ]),
            Route::Index,
        ),
        // Numbers: Cypher `3 = 3.0` holds, and so must the index route.
        (map(&[("rank", RuntimeValue::Integer(3))]), Route::Index),
        (map(&[("rank", RuntimeValue::Float(3.0))]), Route::Index),
        (map(&[("rank", RuntimeValue::Float(3.5))]), Route::Index),
        (
            map(&[(
                "rank",
                RuntimeValue::List(vec![RuntimeValue::Integer(1), RuntimeValue::Float(2.0)]),
            )]),
            Route::Index,
        ),
        // A number never equals the string that spells it.
        (
            map(&[("parentId", RuntimeValue::Integer(1))]),
            Route::Either,
        ),
        // Filters with no indexed equality conjunct: the column read.
        (map(&[("active", RuntimeValue::Bool(true))]), Route::Columns),
        (map(&[("level", lte(4))]), Route::Columns),
        (
            map(&[("parentId", map(&[("ne", string("p1"))]))]),
            Route::Columns,
        ),
    ]
}

/// Ground truth: the filter evaluated on the model.
fn model_ids(docs: &BTreeMap<NodeId, Section>, filter: &RuntimeValue) -> Vec<NodeId> {
    fn matches(doc: &Section, key: &str, condition: &RuntimeValue) -> bool {
        let num = |value: &RuntimeValue| match value {
            RuntimeValue::Integer(n) => Some(*n as f64),
            RuntimeValue::Float(n) => Some(*n),
            _ => None,
        };
        let eq = |value: &RuntimeValue| match (key, value) {
            ("parentId", RuntimeValue::String(text)) => parent(doc.parent) == *text,
            ("parentId", _) => false,
            ("rank", value) => num(value) == Some(doc.rank as f64),
            ("active", RuntimeValue::Bool(flag)) => doc.active == *flag,
            other => unreachable!("{other:?}"),
        };
        match condition {
            RuntimeValue::List(values) => values.iter().any(eq),
            RuntimeValue::Map(ops) => ops.iter().all(|(op, value)| match op.as_str() {
                "in" => match value {
                    RuntimeValue::List(values) => values.iter().any(eq),
                    _ => unreachable!(),
                },
                "ne" => !eq(value),
                "lte" => match key {
                    "level" => num(value).is_some_and(|n| (doc.level as f64) <= n),
                    _ => unreachable!(),
                },
                other => unreachable!("{other}"),
            }),
            scalar => eq(scalar),
        }
    }
    let RuntimeValue::Map(conditions) = filter else {
        unreachable!()
    };
    docs.iter()
        .filter(|(_, doc)| doc.label == LABEL)
        .filter(|(_, doc)| {
            conditions
                .iter()
                .all(|(key, condition)| matches(doc, key, condition))
        })
        .map(|(id, _)| *id)
        .collect()
}

/// Every filter: the index route, when it serves, names exactly the nodes
/// the column read does, the set kept across statements (built either way,
/// then carried or patched over writes, flushes, compactions and reopens)
/// names them too, and all agree with the model.
async fn assert_exact(
    snapshot: &Snapshot<'_>,
    docs: &BTreeMap<NodeId, Section>,
    in_transaction: bool,
    context: &str,
) {
    for (filter, route) in filters() {
        let seek = filter_membership_seek_ids_for_tests(snapshot, LABEL, &filter)
            .await
            .unwrap();
        let exact = filter_membership_ids_for_tests(snapshot, LABEL, &filter, true)
            .await
            .unwrap();
        let model = model_ids(docs, &filter);
        assert_eq!(
            exact.as_deref(),
            Some(model.as_slice()),
            "{context}: column read {filter:?}"
        );
        let kept = filter_membership_ids_for_tests(snapshot, LABEL, &filter, false)
            .await
            .unwrap();
        assert_eq!(
            kept.as_deref(),
            Some(model.as_slice()),
            "{context}: kept set {filter:?}"
        );
        // A write transaction keys numbers apart, so storage declines a
        // numeric probe there; everything else is served.
        let numeric = matches!(&filter, RuntimeValue::Map(m) if m.contains_key("rank"));
        match route {
            Route::Index if !(in_transaction && numeric) => {
                assert_eq!(
                    seek.as_deref(),
                    Some(model.as_slice()),
                    "{context}: index {filter:?}"
                )
            }
            Route::Index | Route::Either => {
                if let Some(seek) = seek {
                    assert_eq!(seek, model, "{context}: index {filter:?}");
                }
            }
            Route::Columns => {
                assert_eq!(seek, None, "{context}: {filter:?} has no indexed equality")
            }
        }
    }
}

async fn randomized(seed: u64) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let name = format!("membership-seek-{seed}");
    let mut writer = WriterSession::open(store.clone(), paths(&name))
        .await
        .unwrap();
    let mut rng = Rng(seed);
    let mut docs: BTreeMap<NodeId, Section> = BTreeMap::new();
    let new_doc = |rng: &mut Rng| Section {
        label: LABEL,
        parent: rng.below(PARENTS),
        rank: rng.below(6) as i64,
        level: rng.below(7) as i64,
        active: rng.below(2) == 0,
    };
    for _ in 0..80 {
        let id = NodeId::new();
        let doc = new_doc(&mut rng);
        writer.upsert_node(doc.label, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    let mut pending = false;
    for step in 0..140 {
        let context = format!("seed {seed} step {step}");
        match rng.below(13) {
            0 | 1 => {
                for _ in 0..1 + rng.below(4) {
                    let id = NodeId::new();
                    let doc = new_doc(&mut rng);
                    writer.upsert_node(doc.label, id, &doc.record()).unwrap();
                    docs.insert(id, doc);
                }
                pending = true;
            }
            2 | 3 => {
                // Move a node to another parent (or rank), flushed or not.
                let ids = docs.keys().copied().collect::<Vec<_>>();
                let id = ids[rng.below(ids.len())];
                let doc = docs.get_mut(&id).unwrap();
                if rng.below(3) == 0 {
                    doc.rank = (doc.rank + 1) % 6;
                } else {
                    doc.parent = (doc.parent + 1 + rng.below(PARENTS - 1)) % PARENTS;
                }
                doc.active = !doc.active;
                writer.upsert_node(doc.label, id, &doc.record()).unwrap();
                pending = true;
            }
            4 => {
                // Relabel, either way: the other label shares the property.
                let ids = docs.keys().copied().collect::<Vec<_>>();
                let id = ids[rng.below(ids.len())];
                let doc = docs.get_mut(&id).unwrap();
                doc.label = if doc.label == LABEL { OTHER } else { LABEL };
                writer.upsert_node(doc.label, id, &doc.record()).unwrap();
                pending = true;
            }
            5 => {
                let ids = docs.keys().copied().collect::<Vec<_>>();
                if ids.len() > 30 {
                    let id = ids[rng.below(ids.len())];
                    let doc = docs.remove(&id).unwrap();
                    writer.tombstone_node(doc.label, id).unwrap();
                    pending = true;
                }
            }
            6 => {
                writer.commit_batch().await.unwrap();
                pending = false;
                writer.flush(schema()).await.unwrap();
            }
            7 => {
                if pending {
                    writer.commit_batch().await.unwrap();
                    pending = false;
                }
                writer.compact_l0(&schema()).await.unwrap();
            }
            8 => {
                writer.commit_batch().await.unwrap();
                pending = false;
                writer.flush(schema()).await.unwrap();
                writer.compact_l0(&schema()).await.unwrap();
            }
            9 => {
                writer.commit_batch().await.unwrap();
                pending = false;
                writer.flush(schema()).await.unwrap();
                drop(writer);
                writer = WriterSession::open(store.clone(), paths(&name))
                    .await
                    .unwrap();
            }
            10 if pending => {
                // A write transaction reads its own staged rows.
                assert_exact(
                    &writer.transactional_overlay_snapshot(),
                    &docs,
                    true,
                    &context,
                )
                .await;
            }
            _ => {
                if pending {
                    writer.commit_batch().await.unwrap();
                    pending = false;
                }
                assert_exact(&writer.snapshot(), &docs, false, &context).await;
            }
        }
    }
    writer.commit_batch().await.unwrap();
    assert_exact(
        &writer.snapshot(),
        &docs,
        false,
        &format!("seed {seed} end"),
    )
    .await;
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    assert_exact(
        &writer.snapshot(),
        &docs,
        false,
        &format!("seed {seed} merged"),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_index_route_names_the_same_nodes_as_the_column_read() {
    let _serial = SERIAL.lock().await;
    let (served_before, _) = filter_membership_seeks();
    let cached_before = filter_membership_seek_cache_hits();
    for seed in [3, 11, 42, 2024, 77777] {
        randomized(seed).await;
    }
    let (served, _) = filter_membership_seeks();
    assert!(served > served_before);
    assert!(
        filter_membership_seek_cache_hits() > cached_before,
        "index-built sets were carried or patched instead of seeked again"
    );
}

/// A value whose posting set exceeds the cap declines before any node is
/// looked up, and the column read answers it; smaller values of the same
/// property keep the index route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_over_the_cap_falls_back_to_the_column_read() {
    let _serial = SERIAL.lock().await;
    // The fixed cap alone: no label-relative budget.
    set_filter_membership_seek_population_divisor(Some(0));
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut writer = WriterSession::open(store, paths("membership-seek-cap"))
        .await
        .unwrap();
    // The default cap is 4096 nodes.
    let section = |parent: usize| Section {
        label: LABEL,
        parent,
        rank: 0,
        level: 1,
        active: true,
    };
    let mut docs = BTreeMap::new();
    // The small parents first, flushed: the flush also installs the schema
    // that declares the index.
    for n in 0..100 {
        let id = NodeId::new();
        let doc = section(1 + n % 3);
        writer.upsert_node(LABEL, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    for _ in 0..4100 {
        let id = NodeId::new();
        let doc = section(0);
        writer.upsert_node(LABEL, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    // The broad parent unflushed, then flushed.
    check_cap(&writer.snapshot(), &docs).await;
    writer.flush(schema()).await.unwrap();
    check_cap(&writer.snapshot(), &docs).await;
    writer.compact_l0(&schema()).await.unwrap();
    check_cap(&writer.snapshot(), &docs).await;
    set_filter_membership_seek_population_divisor(None);
}

/// The budget is the fixed cap lowered to a share of a known label
/// population, never below a floor, and the cap alone when the population is
/// unknown or the share is turned off.
#[test]
fn the_seek_budget_is_a_share_of_a_known_population() {
    assert_eq!(filter_membership_seek_budget(4096, None, 64), 4096);
    assert_eq!(filter_membership_seek_budget(4096, Some(30_086), 0), 4096);
    // A 30 086-node label: 470 candidates, not 4096.
    assert_eq!(filter_membership_seek_budget(4096, Some(30_086), 64), 470);
    // A 411 082-node label: the cap binds.
    assert_eq!(filter_membership_seek_budget(4096, Some(411_082), 64), 4096);
    // Small labels keep the floor.
    assert_eq!(filter_membership_seek_budget(4096, Some(1_000), 64), 256);
    assert_eq!(filter_membership_seek_budget(100, Some(1_000), 64), 100);
}

/// A value admitting more than the label-relative budget reads the columns
/// (counted as a budget decline) and answers the same nodes; a selective
/// value of the same property keeps the index route, flushed and compacted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_over_the_population_budget_reads_the_columns() {
    let _serial = SERIAL.lock().await;
    // A quarter of the label: 1000 nodes allow max(250, 256) = 256.
    set_filter_membership_seek_population_divisor(Some(4));
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut writer = WriterSession::open(store, paths("membership-seek-budget"))
        .await
        .unwrap();
    let section = |parent: usize| Section {
        label: LABEL,
        parent,
        rank: 0,
        level: 1,
        active: true,
    };
    let mut docs = BTreeMap::new();
    for n in 0..1000 {
        let id = NodeId::new();
        // p0: 400 nodes; p1..p3: 200 each.
        let doc = section(if n < 400 { 0 } else { 1 + n % 3 });
        writer.upsert_node(LABEL, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    let broad = map(&[("parentId", string("p0"))]);
    let narrow = map(&[("parentId", string("p1"))]);
    // Nothing flushed yet: the manifest knows no population (the budget then
    // falls back to the fixed cap, see the budget test).
    assert!(writer.snapshot().manifest_label_population(LABEL).is_none());
    // The flush also installs the schema that declares the index.
    writer.flush(schema()).await.unwrap();
    for stage in ["flushed", "compacted"] {
        if stage == "compacted" {
            writer.compact_l0(&schema()).await.unwrap();
        }
        let snapshot = writer.snapshot();
        assert_eq!(
            snapshot.manifest_label_population(LABEL),
            Some(1000),
            "{stage}"
        );
        let budget_before = filter_membership_seek_budget_declines();
        assert_eq!(
            filter_membership_seek_ids_for_tests(&snapshot, LABEL, &broad)
                .await
                .unwrap(),
            None,
            "{stage}: 400 of 1000 nodes exceed the budget"
        );
        assert_eq!(filter_membership_seek_budget_declines(), budget_before + 1);
        assert_eq!(
            filter_membership_ids_for_tests(&snapshot, LABEL, &broad, true)
                .await
                .unwrap(),
            Some(model_ids(&docs, &broad)),
            "{stage}: the column read answers"
        );
        assert_eq!(
            filter_membership_seek_ids_for_tests(&snapshot, LABEL, &narrow)
                .await
                .unwrap(),
            Some(model_ids(&docs, &narrow)),
            "{stage}: 200 nodes stay on the index route"
        );
    }
    set_filter_membership_seek_population_divisor(None);
}

async fn check_cap(snapshot: &Snapshot<'_>, docs: &BTreeMap<NodeId, Section>) {
    let broad = map(&[("parentId", string("p0"))]);
    let (_, declined_before) = filter_membership_seeks();
    assert_eq!(
        filter_membership_seek_ids_for_tests(snapshot, LABEL, &broad)
            .await
            .unwrap(),
        None,
        "4100 nodes exceed the cap"
    );
    let (_, declined) = filter_membership_seeks();
    assert_eq!(declined, declined_before + 1);
    assert_eq!(
        filter_membership_ids_for_tests(snapshot, LABEL, &broad, true)
            .await
            .unwrap()
            .map(|ids| ids.len()),
        Some(4100)
    );
    // Two alternatives that pass the cap together decline too.
    let both = map(&[(
        "parentId",
        RuntimeValue::List(vec![string("p1"), string("p0")]),
    )]);
    assert_eq!(
        filter_membership_seek_ids_for_tests(snapshot, LABEL, &both)
            .await
            .unwrap(),
        None
    );
    // A broad conjunct next to a selective one: the selective one serves,
    // in either order.
    for filter in [
        map(&[
            ("parentId", string("p2")),
            ("rank", RuntimeValue::Integer(0)),
        ]),
        map(&[
            ("rank", RuntimeValue::Integer(0)),
            ("parentId", string("p2")),
        ]),
    ] {
        assert_eq!(
            filter_membership_seek_ids_for_tests(snapshot, LABEL, &filter)
                .await
                .unwrap(),
            Some(model_ids(docs, &filter)),
            "{filter:?}"
        );
    }
}

/// The search legs end to end (see the module comment).
#[cfg(all(feature = "text-index", feature = "vector-index"))]
mod search_legs {
    // `search.bm25`, `search.vector` and `search.hybrid` with a selective
    // equality filter on an indexed, high-cardinality property resolve the
    // filter's membership through the property index instead of reading the
    // filter's column over the whole label. Every result (ids and score bits)
    // must equal the column route's on the same snapshot: flushed, with
    // unflushed creates, moves between values, deletes and relabels, inside a
    // write transaction, and after the flush. Filters with no indexed equality
    // conjunct keep the column route, and a fresh value per statement no longer
    // recomputes a membership set from the columns.

    use std::collections::BTreeMap;
    use std::sync::Arc;

    use namidb_core::id::{NamespaceId, NodeId};
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
    use namidb_core::value::Value as CoreValue;
    use namidb_query::{
        execute, filter_membership_ids_for_tests, filter_membership_paths,
        filter_membership_seek_cache_hits, filter_membership_seeks, lower, optimize, parse,
        set_filter_membership_seek_enabled, Params, RuntimeValue, StatsCatalog,
    };
    use namidb_storage::manifest::{
        TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
    };
    use namidb_storage::{NamespacePaths, NodeWriteRecord, Snapshot, WriterSession};
    use object_store::memory::InMemory;

    const LABEL: &str = "Section";
    const DIM: usize = 4;
    const PARENTS: usize = 30;

    use super::SERIAL;

    fn schema() -> Schema {
        SchemaBuilder::new()
            .label(LabelDef {
                name: LABEL.into(),
                properties: vec![
                    PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                    PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                    PropertyDef::new("parentId", DataType::Utf8, true)
                        .unwrap()
                        .with_indexed(true),
                    PropertyDef::new("level", DataType::Int64, true).unwrap(),
                    PropertyDef::new("active", DataType::Bool, true).unwrap(),
                ],
            })
            .unwrap()
            .build()
    }

    struct Words(u64);

    impl Words {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn word(&mut self) -> String {
            const TOPICAL: [&str; 6] = ["graph", "storage", "engine", "search", "index", "vector"];
            let pick = self.next() % 40;
            if pick < 6 {
                TOPICAL[pick as usize].to_string()
            } else {
                format!("w{}", pick - 6)
            }
        }

        fn text(&mut self, words: usize) -> String {
            (0..words)
                .map(|_| self.word())
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn embedding(&mut self) -> Vec<f32> {
            (0..DIM)
                .map(|_| (self.next() % 1000) as f32 / 1000.0 + 0.01)
                .collect()
        }
    }

    #[derive(Clone)]
    struct Section {
        label: &'static str,
        title: String,
        body: String,
        parent: usize,
        level: i64,
        active: bool,
        embedding: Vec<f32>,
    }

    impl Section {
        fn record(&self) -> NodeWriteRecord {
            let mut properties = BTreeMap::new();
            properties.insert("title".into(), CoreValue::Str(self.title.clone()));
            properties.insert("body".into(), CoreValue::Str(self.body.clone()));
            properties.insert(
                "parentId".into(),
                CoreValue::Str(format!("p{}", self.parent)),
            );
            properties.insert("level".into(), CoreValue::I64(self.level));
            properties.insert("active".into(), CoreValue::Bool(self.active));
            properties.insert("embedding".into(), CoreValue::Vec(self.embedding.clone()));
            NodeWriteRecord {
                properties,
                schema_version: 1,
                ..Default::default()
            }
        }
    }

    struct Model {
        docs: BTreeMap<NodeId, Section>,
        words: Words,
    }

    impl Model {
        fn new_doc(&mut self) -> Section {
            Section {
                label: LABEL,
                title: self.words.text(3),
                body: self.words.text(16),
                parent: self.words.below(PARENTS),
                level: self.words.below(8) as i64,
                active: self.words.below(2) == 0,
                embedding: self.words.embedding(),
            }
        }

        fn write(&mut self, writer: &mut WriterSession, id: NodeId, doc: Section) {
            writer.upsert_node(doc.label, id, &doc.record()).unwrap();
            self.docs.insert(id, doc);
        }

        fn ids(&self) -> Vec<NodeId> {
            self.docs.keys().copied().collect()
        }
    }

    fn paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    async fn run(snapshot: &Snapshot<'_>, cypher: &str, params: &Params) -> Vec<(NodeId, u64)> {
        let query = parse(cypher).unwrap_or_else(|error| panic!("parse {cypher}: {error:?}"));
        let plan = optimize(lower(&query).unwrap(), &StatsCatalog::empty());
        execute(&plan, snapshot, params)
            .await
            .unwrap_or_else(|error| panic!("execute {cypher}: {error}"))
            .iter()
            .map(|row| {
                let id = match row.get("node") {
                    Some(RuntimeValue::Node(node)) => node.id,
                    other => panic!("node: {other:?}"),
                };
                let score = match row.get("score") {
                    Some(RuntimeValue::Float(score)) => score.to_bits(),
                    other => panic!("score: {other:?}"),
                };
                (id, score)
            })
            .collect()
    }

    /// The three legs with `filter` (a Cypher map literal), `k` results.
    fn legs(filter: &str, k: usize, text: &str, vector: &[f32]) -> Vec<(String, Params)> {
        let mut params = Params::new();
        params.insert("q".into(), RuntimeValue::String(text.into()));
        params.insert("v".into(), RuntimeValue::Vector(vector.to_vec()));
        [
            format!(
                "CALL search.bm25({{label: '{LABEL}', text_properties: ['title', 'body'], \
                 query: $q, k: {k}, filter: {filter}}}) YIELD node, score RETURN node, score"
            ),
            format!(
                "CALL search.vector({{label: '{LABEL}', property: 'embedding', query: $v, \
                 k: {k}, filter: {filter}}}) YIELD node, score RETURN node, score"
            ),
            format!(
                "CALL search.hybrid({{label: '{LABEL}', query_text: $q, \
                 text_properties: ['title', 'body'], query_vector: $v, \
                 vector_property: 'embedding', k: {k}, filter: {filter}}}) \
                 YIELD node, score RETURN node, score"
            ),
        ]
        .into_iter()
        .map(|cypher| (cypher, params.clone()))
        .collect()
    }

    /// `(filter, the index route serves it)`.
    fn filters(parents: &[usize]) -> Vec<(String, bool)> {
        let mut out = Vec::new();
        for p in parents {
            out.push((format!("{{parentId: 'p{p}'}}"), true));
            out.push((format!("{{parentId: 'p{p}', level: {{lte: 3}}}}"), true));
            out.push((format!("{{parentId: 'p{p}', active: true}}"), true));
        }
        out.push((
            format!("{{parentId: ['p{}', 'p{}']}}", parents[0], parents[1]),
            true,
        ));
        out.push(("{parentId: 'p-none'}".into(), true));
        out.push(("{active: true}".into(), false));
        out.push(("{level: {lte: 4}}".into(), false));
        out
    }

    /// Every leg, filter and `k`: the index route answers exactly like the column
    /// route on `snapshot`.
    async fn assert_parity(snapshot: &Snapshot<'_>, model: &mut Model, context: &str) {
        let parents = [0, 7, 13, 21, 29];
        let mut cases = Vec::new();
        for (filter, indexed) in filters(&parents) {
            for k in [3, 10] {
                for _ in 0..2 {
                    let text = format!("{} {}", model.words.word(), model.words.word());
                    let vector = model.words.embedding();
                    for (cypher, params) in legs(&filter, k, &text, &vector) {
                        cases.push((cypher, params, indexed));
                    }
                }
            }
        }
        // The index route first: the column route caches its sets across
        // statements, and a cached set would answer before the index is asked.
        set_filter_membership_seek_enabled(true);
        let mut served = Vec::with_capacity(cases.len());
        for (cypher, params, indexed) in &cases {
            let (seeks_before, _) = filter_membership_seeks();
            served.push(run(snapshot, cypher, params).await);
            let (seeks, _) = filter_membership_seeks();
            if !*indexed {
                assert_eq!(
                    seeks, seeks_before,
                    "{context}: {cypher} took the index route"
                );
            }
        }
        set_filter_membership_seek_enabled(false);
        for ((cypher, params, _), served) in cases.iter().zip(served) {
            let columns = run(snapshot, cypher, params).await;
            assert_eq!(served, columns, "{context}: {cypher} {params:?}");
        }
        set_filter_membership_seek_enabled(true);
    }

    async fn indexed_corpus(name: &str) -> (WriterSession, Model) {
        let mut writer = WriterSession::open(Arc::new(InMemory::new()), paths(name))
            .await
            .unwrap();
        writer
            .register_text_index(
                TextIndexDescriptor::new(
                    "section_ft".into(),
                    LABEL.into(),
                    vec!["title".into(), "body".into()],
                ),
                false,
            )
            .await
            .unwrap();
        writer
            .register_vector_index(
                VectorIndexDescriptor {
                    name: "section_emb".into(),
                    label: LABEL.into(),
                    property: "embedding".into(),
                    dim: DIM as u32,
                    metric: VectorMetric::Cosine,
                    r: 32,
                    l_build: 64,
                    alpha: 1.2,
                    quantization: VectorQuantization::None,
                },
                false,
            )
            .await
            .unwrap();
        let mut model = Model {
            docs: BTreeMap::new(),
            words: Words(19),
        };
        for _ in 0..360 {
            let doc = model.new_doc();
            model.write(&mut writer, NodeId::new(), doc);
        }
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
        writer.compact_l0(&schema()).await.unwrap();
        (writer, model)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn every_leg_answers_like_the_column_route() {
        let _serial = SERIAL.lock().await;
        let (mut writer, mut model) = indexed_corpus("seek-parity").await;
        assert_parity(&writer.snapshot(), &mut model, "flushed").await;

        // Unflushed: creates, moves between parents, deletes, a relabel.
        let ids = model.ids();
        for _ in 0..12 {
            let mut doc = model.new_doc();
            doc.parent = [0, 7, 13][model.words.below(3)];
            model.write(&mut writer, NodeId::new(), doc);
        }
        for id in ids.iter().take(20) {
            let mut doc = model.docs[id].clone();
            doc.parent = [0, 7, 13, 21, 29][model.words.below(5)];
            doc.active = !doc.active;
            model.write(&mut writer, *id, doc);
        }
        for id in ids.iter().skip(20).take(6) {
            let doc = model.docs.remove(id).unwrap();
            writer.tombstone_node(doc.label, *id).unwrap();
        }
        {
            let mut doc = model.docs[&ids[30]].clone();
            doc.label = "Other";
            model.write(&mut writer, ids[30], doc);
        }
        writer.commit_batch().await.unwrap();
        assert_parity(&writer.snapshot(), &mut model, "unflushed").await;

        // A write transaction sees its own staged rows.
        for id in ids.iter().skip(40).take(5) {
            let mut doc = model.docs[id].clone();
            doc.parent = 21;
            model.write(&mut writer, *id, doc);
        }
        let doc = model.docs.remove(&ids[50]).unwrap();
        writer.tombstone_node(doc.label, ids[50]).unwrap();
        assert_parity(
            &writer.transactional_overlay_snapshot(),
            &mut model,
            "transaction",
        )
        .await;
        writer.commit_batch().await.unwrap();

        writer.flush(schema()).await.unwrap();
        assert_parity(&writer.snapshot(), &mut model, "flushed again").await;
        writer.compact_l0(&schema()).await.unwrap();
        assert_parity(&writer.snapshot(), &mut model, "merged").await;
    }

    /// A fresh parent value resolves through the index (once: the other legs
    /// with the same value reuse the cached set) and never recomputes a
    /// membership set from the columns; a filter with no indexed equality
    /// still does.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fresh_values_never_recompute_a_column_set() {
        let _serial = SERIAL.lock().await;
        set_filter_membership_seek_enabled(true);
        let (writer, mut model) = indexed_corpus("seek-no-recompute").await;
        let snapshot = writer.snapshot();
        let (_, _, _, recomputed_before) = filter_membership_paths();
        let (seeks_before, _) = filter_membership_seeks();
        let mut statements = 0;
        for p in 0..PARENTS {
            let text = model.words.text(2);
            let vector = model.words.embedding();
            for (cypher, params) in legs(&format!("{{parentId: 'p{p}'}}"), 10, &text, &vector) {
                run(&snapshot, &cypher, &params).await;
                statements += 1;
            }
        }
        let (_, _, _, recomputed) = filter_membership_paths();
        let (seeks, _) = filter_membership_seeks();
        assert_eq!(
            recomputed, recomputed_before,
            "no column set was recomputed"
        );
        assert!(
            seeks >= seeks_before + PARENTS as u64,
            "every value resolved through the index ({} seeks, {PARENTS} values)",
            seeks - seeks_before
        );
        assert!(statements > PARENTS);

        let text = model.words.text(2);
        let vector = model.words.embedding();
        for (cypher, params) in legs("{level: {lte: 4}}", 10, &text, &vector) {
            run(&snapshot, &cypher, &params).await;
        }
        let (_, _, _, recomputed) = filter_membership_paths();
        assert!(
            recomputed > recomputed_before,
            "a filter with no indexed equality reads its column"
        );
    }
    /// The legs' answers with the index route on, and how many index seeks
    /// they took.
    async fn served_and_seeks(
        snapshot: &Snapshot<'_>,
        legs: &[(String, Params)],
    ) -> (Vec<Vec<(NodeId, u64)>>, u64, u64) {
        let (before, _) = filter_membership_seeks();
        let hits_before = filter_membership_seek_cache_hits();
        let mut out = Vec::new();
        for (cypher, params) in legs {
            out.push(run(snapshot, cypher, params).await);
        }
        let (after, _) = filter_membership_seeks();
        (
            out,
            after - before,
            filter_membership_seek_cache_hits() - hits_before,
        )
    }

    /// The legs' answers with the index route off (no seek may happen).
    async fn column_answers(
        snapshot: &Snapshot<'_>,
        legs: &[(String, Params)],
    ) -> Vec<Vec<(NodeId, u64)>> {
        set_filter_membership_seek_enabled(false);
        let (before, _) = filter_membership_seeks();
        let mut out = Vec::new();
        for (cypher, params) in legs {
            out.push(run(snapshot, cypher, params).await);
        }
        let (after, _) = filter_membership_seeks();
        set_filter_membership_seek_enabled(true);
        assert_eq!(after, before, "the index route is off");
        out
    }

    /// A set the index route built is kept across statements like a column
    /// set: the same filter again does not seek (a cache hit), a committed
    /// write patches it and a flush or a compaction carries it, still without
    /// a seek, and every answer equals the column route's. With the route off
    /// the kept index set is not used.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_index_set_is_kept_and_carried_like_a_column_set() {
        let _serial = SERIAL.lock().await;
        set_filter_membership_seek_enabled(true);
        let (mut writer, mut model) = indexed_corpus("seek-kept").await;
        let filter = "{parentId: 'p3', level: {lte: 5}}";

        let snapshot = writer.snapshot();
        let text = model.words.text(2);
        let vector = model.words.embedding();
        let first = legs(filter, 10, &text, &vector);
        let (served, seeks, _) = served_and_seeks(&snapshot, &first).await;
        assert!(
            seeks >= 1,
            "the first statement resolves the set through the index"
        );
        let (again, seeks, hits) = served_and_seeks(&snapshot, &first).await;
        assert_eq!(seeks, 0, "the same filter on the same data does not seek");
        assert!(hits >= 1, "it is a cache hit");
        assert_eq!(again, served);
        let text = model.words.text(2);
        let vector = model.words.embedding();
        let other_query = legs(filter, 3, &text, &vector);
        let (other_served, seeks, _) = served_and_seeks(&snapshot, &other_query).await;
        assert_eq!(
            seeks, 0,
            "another query with the same filter reuses the set"
        );
        // The route off: the kept index set is replaced by a column read, and
        // the answers are the same.
        assert_eq!(
            column_answers(&snapshot, &first).await,
            served,
            "kept index set vs columns"
        );
        assert_eq!(column_answers(&snapshot, &other_query).await, other_served);
        drop(snapshot);

        // Build the set through the index again for a value the column
        // passes above did not touch, then change the data under it.
        let filter = "{parentId: 'p5', level: {lte: 5}}";
        let snapshot = writer.snapshot();
        let text = model.words.text(2);
        let vector = model.words.embedding();
        let (_, seeks, _) = served_and_seeks(&snapshot, &legs(filter, 10, &text, &vector)).await;
        assert!(seeks >= 1);
        drop(snapshot);
        let ids = model.ids();
        // A new member, a member moved out, a non-member moved in, a delete.
        let mut doc = model.new_doc();
        doc.parent = 5;
        doc.level = 1;
        model.write(&mut writer, NodeId::new(), doc);
        let member = ids
            .iter()
            .copied()
            .find(|id| model.docs[id].parent == 5 && model.docs[id].level <= 5);
        if let Some(id) = member {
            let mut doc = model.docs[&id].clone();
            doc.parent = 6;
            model.write(&mut writer, id, doc);
        }
        let outsider = ids.iter().copied().find(|id| model.docs[id].parent == 2);
        if let Some(id) = outsider {
            let mut doc = model.docs[&id].clone();
            doc.parent = 5;
            doc.level = 2;
            model.write(&mut writer, id, doc);
        }
        let gone = ids
            .iter()
            .copied()
            .find(|id| model.docs[id].parent == 5 && Some(*id) != member);
        if let Some(id) = gone {
            let doc = model.docs.remove(&id).unwrap();
            writer.tombstone_node(doc.label, id).unwrap();
        }
        writer.commit_batch().await.unwrap();
        // The same filter as a procedure map, for the membership hooks (which
        // neither read nor fill the statement caches when `exact`).
        let filter_value = super::map(&[
            ("parentId", super::string("p5")),
            ("level", super::map(&[("lte", RuntimeValue::Integer(5))])),
        ]);
        let mut last = None;
        for context in ["after writes", "after a flush", "after a compaction"] {
            match context {
                "after a flush" => {
                    writer.flush(schema()).await.unwrap();
                }
                "after a compaction" => {
                    writer.compact_l0(&schema()).await.unwrap();
                }
                _ => {}
            }
            let snapshot = writer.snapshot();
            let text = model.words.text(2);
            let vector = model.words.embedding();
            let legs = legs(filter, 10, &text, &vector);
            let (served, seeks, hits) = served_and_seeks(&snapshot, &legs).await;
            assert_eq!(
                seeks, 0,
                "{context}: the kept set is patched or carried, not seeked again"
            );
            assert!(hits >= 1, "{context}: a cache hit");
            let exact = filter_membership_ids_for_tests(&snapshot, LABEL, &filter_value, true)
                .await
                .unwrap();
            let mut expected: Vec<NodeId> = model
                .docs
                .iter()
                .filter(|(_, d)| d.label == LABEL && d.parent == 5 && d.level <= 5)
                .map(|(id, _)| *id)
                .collect();
            expected.sort_unstable();
            assert_eq!(
                exact.as_deref(),
                Some(expected.as_slice()),
                "{context}: model"
            );
            last = Some((legs, served));
        }
        // Once at the end: a column read with the route off replaces the kept set.
        let (legs, served) = last.unwrap();
        assert_eq!(
            column_answers(&writer.snapshot(), &legs).await,
            served,
            "compacted, vs columns"
        );
    }
}

/// A filtered vector search whose filter has no index and whose members sit
/// far from the query (see the module comment).
#[cfg(feature = "vector-index")]
mod member_scan {
    // `search.vector` with an unindexed range filter admitting ~8% of the
    // label, all of it in a cluster far from the queries: the ANN probe the
    // first round sizes from the filter's selectivity finds none of them, and
    // widening used to run until the exact scan of every node. After one
    // under-filled round the search now ranks exactly the members from the
    // index's codes. Its answers must equal the widening route's (ids, scores
    // to f32 precision), flushed and with unflushed moves into and out of the
    // member set, re-embeddings and deletes; filters the first round fills
    // never take it.

    use std::collections::BTreeMap;
    use std::sync::Arc;

    use namidb_core::id::{NamespaceId, NodeId};
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
    use namidb_core::value::Value as CoreValue;
    use namidb_query::{
        execute, lower, optimize, parse, set_vector_member_scan_enabled, vector_member_scans,
        Params, RuntimeValue, StatsCatalog,
    };
    use namidb_storage::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
    use namidb_storage::{NamespacePaths, NodeWriteRecord, Snapshot, WriterSession};
    use object_store::memory::InMemory;

    use super::SERIAL;

    const LABEL: &str = "Doc";
    const DIM: usize = 16;
    const DOCS: usize = 54_000;

    fn schema() -> Schema {
        SchemaBuilder::new()
            .label(LabelDef {
                name: LABEL.into(),
                properties: vec![
                    PropertyDef::new("level", DataType::Int64, true).unwrap(),
                    PropertyDef::new("parentId", DataType::Utf8, true)
                        .unwrap()
                        .with_indexed(true),
                ],
            })
            .unwrap()
            .build()
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn unit(&mut self) -> f32 {
            (self.next() % 10_000) as f32 / 10_000.0 - 0.5
        }

        /// Around `+center` on the first axis (near) or `-center` (far).
        fn embedding(&mut self, far: bool) -> Vec<f32> {
            (0..DIM)
                .map(|axis| {
                    let center = if axis == 0 {
                        if far {
                            -3.0
                        } else {
                            3.0
                        }
                    } else {
                        0.0
                    };
                    center + self.unit()
                })
                .collect()
        }
    }

    fn record(level: i64, parent: usize, embedding: &[f32]) -> NodeWriteRecord {
        let mut properties = BTreeMap::new();
        properties.insert("level".into(), CoreValue::I64(level));
        properties.insert("parentId".into(), CoreValue::Str(format!("p{parent}")));
        properties.insert("embedding".into(), CoreValue::Vec(embedding.to_vec()));
        NodeWriteRecord {
            properties,
            schema_version: 1,
            ..Default::default()
        }
    }

    async fn run(snapshot: &Snapshot<'_>, cypher: &str, params: &Params) -> Vec<(NodeId, f64)> {
        let query = parse(cypher).unwrap();
        let plan = optimize(lower(&query).unwrap(), &StatsCatalog::empty());
        execute(&plan, snapshot, params)
            .await
            .unwrap_or_else(|error| panic!("execute {cypher}: {error}"))
            .iter()
            .map(|row| {
                let id = match row.get("node") {
                    Some(RuntimeValue::Node(node)) => node.id,
                    other => panic!("node: {other:?}"),
                };
                let score = match row.get("score") {
                    Some(RuntimeValue::Float(score)) => *score,
                    other => panic!("score: {other:?}"),
                };
                (id, score)
            })
            .collect()
    }

    fn search(filter: &str, k: usize) -> String {
        format!(
            "CALL search.vector({{label: '{LABEL}', property: 'embedding', query: $v, k: {k}, \
             filter: {filter}}}) YIELD node, score RETURN node, score"
        )
    }

    /// `(filter, the member scan answers it)`.
    /// More members than the exact flat route takes (4096), fewer than the
    /// first round's over-fetch reaches; a selective one and a broad one.
    const FILTERS: [(&str, bool); 4] = [
        ("{level: {lte: 4}}", true),
        ("{level: {lt: 5}}", true),
        ("{level: {lte: 4}, parentId: 'p3'}", false),
        ("{level: {gte: 5}}", false),
    ];

    async fn assert_parity(snapshot: &Snapshot<'_>, rng: &mut Rng, context: &str) {
        let mut cases = Vec::new();
        for (filter, scanned) in FILTERS {
            for k in [10, 50] {
                for _ in 0..2 {
                    let mut params = Params::new();
                    params.insert("v".into(), RuntimeValue::Vector(rng.embedding(false)));
                    cases.push((search(filter, k), params, scanned, k));
                }
            }
        }
        set_vector_member_scan_enabled(true);
        let mut scanned_results = Vec::new();
        for (cypher, params, scanned, _) in &cases {
            let before = vector_member_scans();
            scanned_results.push(run(snapshot, cypher, params).await);
            let took = vector_member_scans() > before;
            if *scanned {
                assert!(took, "{context}: {cypher} did not take the member scan");
            } else {
                assert!(!took, "{context}: {cypher} took the member scan");
            }
        }
        set_vector_member_scan_enabled(false);
        for ((cypher, params, _, k), scanned) in cases.iter().zip(scanned_results) {
            let widened = run(snapshot, cypher, params).await;
            assert_eq!(scanned.len(), widened.len(), "{context}: {cypher} k={k}");
            assert!(scanned.len() == *k || scanned.len() < *k, "{context}");
            for (position, ((left_id, left), (right_id, right))) in
                scanned.iter().zip(&widened).enumerate()
            {
                assert!(
                    (left - right).abs() <= 1e-5 * right.abs().max(1.0),
                    "{context}: {cypher} k={k} rank {position}: score {left} vs {right}"
                );
                // Ids agree except between scores tied to f32 precision.
                if left_id != right_id {
                    let tied = widened
                        .iter()
                        .chain(&scanned)
                        .filter(|(_, score)| (score - left).abs() <= 1e-5 * left.abs().max(1.0))
                        .count();
                    assert!(
                        tied > 2,
                        "{context}: {cypher} k={k} rank {position}: ids differ"
                    );
                }
            }
        }
        set_vector_member_scan_enabled(true);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn far_members_are_ranked_from_the_codes_like_the_widening_route() {
        let _serial = SERIAL.lock().await;
        let paths = NamespacePaths::new("tenants", NamespaceId::new("member-scan").unwrap());
        let mut writer = WriterSession::open(Arc::new(InMemory::new()), paths)
            .await
            .unwrap();
        writer
            .register_vector_index(
                VectorIndexDescriptor {
                    name: "doc_emb".into(),
                    label: LABEL.into(),
                    property: "embedding".into(),
                    dim: DIM as u32,
                    metric: VectorMetric::Cosine,
                    r: 32,
                    l_build: 64,
                    alpha: 1.2,
                    quantization: VectorQuantization::None,
                },
                false,
            )
            .await
            .unwrap();
        let mut rng = Rng(353);
        let mut ids = Vec::with_capacity(DOCS);
        for index in 0..DOCS {
            // One in twelve is a member (level <= 4, 4500 of them), far from
            // the queries.
            let member = index % 12 == 0;
            let level = if member {
                (index % 5) as i64
            } else {
                5 + (index % 7) as i64
            };
            let id = NodeId::new();
            writer
                .upsert_node(
                    LABEL,
                    id,
                    &record(level, index % 40, &rng.embedding(member)),
                )
                .unwrap();
            ids.push((id, member));
        }
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
        writer.compact_l0(&schema()).await.unwrap();
        assert_parity(&writer.snapshot(), &mut rng, "flushed").await;

        // Unflushed: near nodes moved into the member set (and re-embedded far),
        // members moved out, members re-embedded, members deleted.
        // Fewer fresh members than `k`: a delta that fills a round on its own
        // ends the search on the first round, member scan or not.
        let mut moved_in = 0;
        for (position, (id, member)) in ids.iter().enumerate().take(2_400) {
            match (member, position % 4) {
                (false, 1) if moved_in < 8 => {
                    moved_in += 1;
                    writer
                        .upsert_node(LABEL, *id, &record(3, 3, &rng.embedding(true)))
                        .unwrap();
                }
                (true, 0) => {
                    writer
                        .upsert_node(LABEL, *id, &record(9, 3, &rng.embedding(false)))
                        .unwrap();
                }
                (true, _) if position % 5 == 0 => {
                    writer.tombstone_node(LABEL, *id).unwrap();
                }
                (true, _) if moved_in < 8 => {
                    moved_in += 1;
                    writer
                        .upsert_node(LABEL, *id, &record(1, 3, &rng.embedding(true)))
                        .unwrap();
                }
                _ => {}
            }
        }
        writer.commit_batch().await.unwrap();
        assert_parity(&writer.snapshot(), &mut rng, "unflushed").await;

        writer.flush(schema()).await.unwrap();
        assert_parity(&writer.snapshot(), &mut rng, "flushed again").await;
        writer.compact_l0(&schema()).await.unwrap();
        assert_parity(&writer.snapshot(), &mut rng, "merged").await;
    }
}

/// 1000 flushed `LABEL` nodes: `p0` admits 400, `p1`..`p3` 200 each.
async fn speculation_fixture(name: &str) -> (WriterSession, BTreeMap<NodeId, Section>) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut writer = WriterSession::open(store, paths(name)).await.unwrap();
    let mut docs = BTreeMap::new();
    for n in 0..1000 {
        let id = NodeId::new();
        let doc = Section {
            label: LABEL,
            parent: if n < 400 { 0 } else { 1 + n % 3 },
            rank: (n % 7) as i64,
            level: 1 + (n % 2) as i64,
            active: true,
        };
        writer.upsert_node(LABEL, id, &doc.record()).unwrap();
        docs.insert(id, doc);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    (writer, docs)
}

fn speculation_pool() -> &'static Arc<QueryMemoryPool> {
    static POOL: std::sync::OnceLock<Arc<QueryMemoryPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Arc::new(QueryMemoryPool::new(1 << 30)))
}

/// One first sighting of `filter` (the route a search statement takes) under
/// a pooled memory account; the pool must be back at its baseline after.
async fn first_sighting(writer: &WriterSession, filter: &RuntimeValue) -> Option<Vec<NodeId>> {
    let snapshot = writer.snapshot();
    let budget = Arc::new(QueryMemoryBudget::new(256 << 20).with_pool(speculation_pool().clone()));
    let ids = with_query_memory(
        Some(budget.clone()),
        filter_membership_ids_for_tests(&snapshot, LABEL, filter, false),
    )
    .await
    .unwrap();
    drop(budget);
    assert_eq!(
        speculation_pool().reserved_bytes(),
        0,
        "the statement's account (the speculative read's included) is released"
    );
    ids
}

/// On a label small enough for the label-relative budget to bind, the column
/// read starts next to the seek. A seek that declines (one value over the
/// budget, or a later IN alternative that overruns what the earlier ones
/// left) takes the column read's answer; a seek that serves drops the column
/// read. Either way the answer is exact and the memory comes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_small_label_reads_its_columns_while_the_seek_decides() {
    let _serial = SERIAL.lock().await;
    // 1000 nodes / 4: a budget of max(250, 256) = 256, below the 4096 cap.
    set_filter_membership_seek_population_divisor(Some(4));
    let (writer, docs) = speculation_fixture("membership-speculation").await;
    let cases = [
        (
            "one value over the budget",
            map(&[("parentId", string("p0"))]),
            false,
        ),
        (
            "a later alternative overruns the budget",
            map(&[(
                "parentId",
                RuntimeValue::List(vec![string("p1"), string("p0")]),
            )]),
            false,
        ),
        (
            "a value the seek serves",
            map(&[("parentId", string("p2"))]),
            true,
        ),
    ];
    for (context, filter, served) in cases {
        let (used0, dropped0) = filter_membership_seek_speculation();
        let (served0, _) = filter_membership_seeks();
        let budget0 = filter_membership_seek_budget_declines();
        let ids = first_sighting(&writer, &filter).await;
        assert_eq!(ids, Some(model_ids(&docs, &filter)), "{context}: exact");
        let (used, dropped) = filter_membership_seek_speculation();
        let (served_now, _) = filter_membership_seeks();
        if served {
            assert_eq!(
                (used, dropped),
                (used0, dropped0 + 1),
                "{context}: read dropped"
            );
            assert_eq!(served_now, served0 + 1, "{context}: the seek served");
        } else {
            assert_eq!(
                (used, dropped),
                (used0 + 1, dropped0),
                "{context}: read used"
            );
            assert_eq!(
                filter_membership_seek_budget_declines(),
                budget0 + 1,
                "{context}: a budget decline"
            );
        }
    }
    set_filter_membership_seek_population_divisor(None);
}

/// Where the fixed cap binds (a large label, here a cap below the label's
/// share), the routes run one after the other: no column read is started
/// next to the seek, served or declined.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_label_the_cap_bounds_never_speculates() {
    let _serial = SERIAL.lock().await;
    // 1000 nodes / 1 = 1000 > the 300 cap: the cap binds.
    set_filter_membership_seek_population_divisor(Some(1));
    set_filter_membership_seek_cap_for_tests(Some(300));
    let (writer, docs) = speculation_fixture("membership-no-speculation").await;
    let before = filter_membership_seek_speculation();
    for filter in [
        map(&[("parentId", string("p0"))]),
        map(&[("parentId", string("p1"))]),
    ] {
        let ids = first_sighting(&writer, &filter).await;
        assert_eq!(ids, Some(model_ids(&docs, &filter)));
    }
    assert_eq!(
        filter_membership_seek_speculation(),
        before,
        "no speculative read"
    );
    set_filter_membership_seek_cap_for_tests(None);
    set_filter_membership_seek_population_divisor(None);
}

/// A posting length a probe advertised is remembered for the process: a
/// later statement with a different filter sharing the over-budget value
/// declines that value without opening the sidecars again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_known_posting_length_declines_without_probing() {
    let _serial = SERIAL.lock().await;
    set_filter_membership_seek_population_divisor(Some(4));
    let (writer, docs) = speculation_fixture("membership-posting-memo").await;
    let first = map(&[("parentId", string("p0"))]);
    let probes0 = namidb_storage::equality_sidecar_probe_stats();
    assert_eq!(
        first_sighting(&writer, &first).await,
        Some(model_ids(&docs, &first))
    );
    let probes1 = namidb_storage::equality_sidecar_probe_stats();
    assert!(probes1.probes > probes0.probes, "the first sighting probes");
    // A different filter (its own kept set) that shares parentId = 'p0'.
    let second = map(&[
        ("parentId", string("p0")),
        ("active", RuntimeValue::Bool(true)),
    ]);
    assert_eq!(
        first_sighting(&writer, &second).await,
        Some(model_ids(&docs, &second))
    );
    let probes2 = namidb_storage::equality_sidecar_probe_stats();
    assert_eq!(probes2.probes, probes1.probes, "no sidecar is probed again");
    assert_eq!(
        probes2.memo_declines,
        probes1.memo_declines + 1,
        "declined from the remembered length"
    );
    set_filter_membership_seek_population_divisor(None);
}
