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
    filter_membership_ids_for_tests, filter_membership_seek_ids_for_tests, filter_membership_seeks,
    RuntimeValue,
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
/// the column read does, and both agree with the model.
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
    for seed in [3, 11, 42, 2024, 77777] {
        randomized(seed).await;
    }
    let (served, _) = filter_membership_seeks();
    assert!(served > served_before);
}

/// A value whose posting set exceeds the cap declines before any node is
/// looked up, and the column read answers it; smaller values of the same
/// property keep the index route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_over_the_cap_falls_back_to_the_column_read() {
    let _serial = SERIAL.lock().await;
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
        execute, filter_membership_paths, filter_membership_seeks, lower, optimize, parse,
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

    /// A fresh parent value per statement resolves through the index on every
    /// leg and never recomputes a membership set from the columns; a filter with
    /// no indexed equality still does.
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
            seeks >= seeks_before + statements,
            "every statement resolved through the index ({} of {statements})",
            seeks - seeks_before
        );

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
}
