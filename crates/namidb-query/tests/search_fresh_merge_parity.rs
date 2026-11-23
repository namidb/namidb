//! Unflushed writes no longer turn a full-text read into an O(corpus) scan:
//! the index serves, with the memtable's documents of the label merged in.
//! The merged answer must be exactly the flat scan's, ids and score bits,
//! for every kind of unflushed write, on every index layout, unfiltered and
//! filtered, and inside a transaction that sees its own writes.
//!
//! The oracle is a twin namespace with the same logical corpus and no index,
//! where `search.bm25` can only run the exact flat scan.
#![cfg(all(feature = "text-index", feature = "vector-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::{execute, lower, optimize, parse, Params, RuntimeValue, StatsCatalog};
use namidb_storage::manifest::{
    TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::search_lsm::{SearchLsmKind, SearchSegmentFormat};
use namidb_storage::{NamespacePaths, NodeWriteRecord, Snapshot, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const LABEL: &str = "Doc";
const INDEX: &str = "doc_ft";
const DIM: usize = 4;

fn schema() -> Schema {
    let mut vigente = PropertyDef::new("vigente", DataType::Bool, true).unwrap();
    vigente.indexed = true;
    SchemaBuilder::new()
        .label(LabelDef {
            name: LABEL.into(),
            properties: vec![
                PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("n", DataType::Int64, true).unwrap(),
                vigente,
            ],
        })
        .unwrap()
        .build()
}

/// Deterministic words from a small vocabulary, so terms repeat.
struct Words(u64);

impl Words {
    fn next(&mut self) -> String {
        const TOPICAL: [&str; 6] = ["graph", "storage", "engine", "search", "index", "vector"];
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (self.0 >> 33) % 60;
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

    fn embedding(&mut self) -> Vec<f32> {
        (0..DIM)
            .map(|_| {
                self.next();
                ((self.0 >> 40) % 1000) as f32 / 1000.0 + 0.01
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct Doc {
    label: &'static str,
    title: String,
    body: Option<String>,
    n: i64,
    vigente: bool,
    embedding: Vec<f32>,
}

impl Doc {
    fn record(&self) -> NodeWriteRecord {
        let mut properties = BTreeMap::new();
        properties.insert("title".into(), CoreValue::Str(self.title.clone()));
        if let Some(body) = &self.body {
            properties.insert("body".into(), CoreValue::Str(body.clone()));
        }
        properties.insert("n".into(), CoreValue::I64(self.n));
        properties.insert("vigente".into(), CoreValue::Bool(self.vigente));
        properties.insert("embedding".into(), CoreValue::Vec(self.embedding.clone()));
        NodeWriteRecord {
            properties,
            schema_version: 1,
            ..Default::default()
        }
    }
}

/// The logical corpus, written to the indexed namespace and replayed into
/// the flat oracle.
struct Model {
    docs: BTreeMap<NodeId, Doc>,
    words: Words,
}

impl Model {
    fn write(&mut self, writer: &mut WriterSession, id: NodeId, doc: Doc) {
        writer.upsert_node(doc.label, id, &doc.record()).unwrap();
        self.docs.insert(id, doc);
    }

    fn delete(&mut self, writer: &mut WriterSession, id: NodeId) {
        let doc = self.docs.remove(&id).expect("deleting a known doc");
        writer.tombstone_node(doc.label, id).unwrap();
    }

    fn new_doc(&mut self, ordinal: usize) -> Doc {
        Doc {
            label: LABEL,
            title: self.words.text(3),
            body: Some(self.words.text(18)),
            n: ordinal as i64,
            vigente: !ordinal.is_multiple_of(3),
            embedding: self.words.embedding(),
        }
    }

    fn ids(&self) -> Vec<NodeId> {
        self.docs.keys().copied().collect()
    }

    async fn oracle(&self, name: &str) -> WriterSession {
        let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
        for (id, doc) in &self.docs {
            writer.upsert_node(doc.label, *id, &doc.record()).unwrap();
        }
        writer.commit_batch().await.unwrap();
        writer
    }
}

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn text_formats(writer: &WriterSession) -> Vec<SearchSegmentFormat> {
    writer
        .snapshot()
        .manifest()
        .manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Text && state.index_name == INDEX)
        .map(|state| {
            state
                .segments
                .iter()
                .map(|segment| segment.format)
                .collect()
        })
        .unwrap_or_default()
}

fn text_index() -> TextIndexDescriptor {
    TextIndexDescriptor::new(
        INDEX.into(),
        LABEL.into(),
        vec!["title".into(), "body".into()],
    )
}

fn vector_index() -> VectorIndexDescriptor {
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
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Layout {
    /// One FT4 base.
    Ft4Base,
    /// An FT4 base plus FT4 deltas.
    Ft4Segments,
    /// A V3 base as a 3.4.x engine built it, alone.
    LoneV3Base,
    /// That V3 base with FT4 deltas on top.
    V3BaseWithDeltas,
}

async fn indexed(name: &str, layout: Layout) -> (WriterSession, Model) {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let mut model = Model {
        docs: BTreeMap::new(),
        words: Words(11),
    };
    let legacy = matches!(layout, Layout::LoneV3Base | Layout::V3BaseWithDeltas);
    if !legacy {
        writer
            .register_text_index(text_index(), false)
            .await
            .unwrap();
        writer
            .register_vector_index(vector_index(), false)
            .await
            .unwrap();
    }
    for ordinal in 0..120 {
        let doc = model.new_doc(ordinal);
        model.write(&mut writer, NodeId::new(), doc);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    if legacy {
        // The way 3.4.x built it: nodes flushed first, the index registered,
        // then rebuilt logically inside the next Nodes merge.
        writer
            .register_text_index(text_index(), false)
            .await
            .unwrap();
    }
    writer.compact_l0(&schema()).await.unwrap();
    if matches!(layout, Layout::Ft4Segments | Layout::V3BaseWithDeltas) {
        for round in 0..2 {
            let ids = model.ids();
            for id in ids.iter().skip(round * 11).take(4) {
                let mut doc = model.docs[id].clone();
                doc.body = Some(format!("graph storage {}", model.words.text(9)));
                model.write(&mut writer, *id, doc);
            }
            model.delete(&mut writer, ids[60 + round]);
            for ordinal in 0..3 {
                let doc = model.new_doc(1000 + round * 10 + ordinal);
                model.write(&mut writer, NodeId::new(), doc);
            }
            writer.commit_batch().await.unwrap();
            writer.flush(schema()).await.unwrap();
        }
    }
    let formats = text_formats(&writer);
    match layout {
        Layout::Ft4Base => assert_eq!(formats, [SearchSegmentFormat::TextV4], "{layout:?}"),
        Layout::Ft4Segments => {
            assert!(formats.len() >= 2, "{layout:?}: {formats:?}");
            assert!(formats.iter().all(|f| *f == SearchSegmentFormat::TextV4));
        }
        Layout::LoneV3Base => assert_eq!(formats, [SearchSegmentFormat::TextV3Base]),
        Layout::V3BaseWithDeltas => {
            assert_eq!(formats[0], SearchSegmentFormat::TextV3Base, "{formats:?}");
            assert!(formats.len() >= 2, "{formats:?}");
        }
    }
    (writer, model)
}

/// Every kind of unflushed write, committed to the memtable.
async fn mutate(writer: &mut WriterSession, model: &mut Model) {
    let ids = model.ids();
    // CREATE.
    for ordinal in 0..5 {
        let mut doc = model.new_doc(2000 + ordinal);
        doc.body = Some(format!("fresh graph engine {}", model.words.text(8)));
        model.write(writer, NodeId::new(), doc);
    }
    // SET of indexed text on documents the index holds.
    for id in ids.iter().take(6) {
        let mut doc = model.docs[id].clone();
        doc.body = Some(format!("search index {}", model.words.text(6)));
        model.write(writer, *id, doc);
    }
    // SET of non-indexed properties only.
    for id in ids.iter().skip(6).take(4) {
        let mut doc = model.docs[id].clone();
        doc.n += 500;
        doc.embedding = model.words.embedding();
        model.write(writer, *id, doc);
    }
    // The native filter value flips, the text stays.
    for id in ids.iter().skip(10).take(6) {
        let mut doc = model.docs[id].clone();
        doc.vigente = !doc.vigente;
        model.write(writer, *id, doc);
    }
    // An indexed property removed.
    {
        let mut doc = model.docs[&ids[16]].clone();
        doc.body = None;
        model.write(writer, ids[16], doc);
    }
    // DELETE of documents the index holds.
    for id in ids.iter().skip(20).take(4) {
        model.delete(writer, *id);
    }
    // Label removed (relabel to another label).
    for id in ids.iter().skip(24).take(2) {
        let mut doc = model.docs[id].clone();
        doc.label = "Other";
        model.write(writer, *id, doc);
    }
    writer.commit_batch().await.unwrap();
    // Label added: a persisted non-Doc node becomes a Doc... written as Other
    // first (committed), then relabeled in a later commit.
    let joined = NodeId::new();
    let mut doc = model.new_doc(3000);
    doc.label = "Other";
    model.write(writer, joined, doc.clone());
    writer.commit_batch().await.unwrap();
    doc.label = LABEL;
    doc.body = Some("graph engine joins the corpus".into());
    model.write(writer, joined, doc);
    // A node rewritten twice, in two commits.
    let twice = ids[30];
    let mut doc = model.docs[&twice].clone();
    doc.body = Some("vector storage first".into());
    model.write(writer, twice, doc.clone());
    writer.commit_batch().await.unwrap();
    doc.body = Some("vector search second graph".into());
    model.write(writer, twice, doc);
    writer.commit_batch().await.unwrap();
}

const QUERIES: [&str; 10] = [
    "graph",
    "graph storage engine",
    "search index",
    "w1 w2 w3",
    "\"graph engine\"",
    "vector search",
    // Terms only unflushed documents carry.
    "fresh",
    "joins corpus",
    "w4*",
    // A prefix the index does not expand at all, matched by unflushed
    // documents only.
    "fre*",
];

async fn run(snapshot: &Snapshot<'_>, cypher: &str, params: &Params) -> Vec<namidb_query::Row> {
    let query = parse(cypher).unwrap_or_else(|error| panic!("parse {cypher}: {error:?}"));
    let plan = optimize(lower(&query).unwrap(), &StatsCatalog::empty());
    execute(&plan, snapshot, params)
        .await
        .unwrap_or_else(|error| panic!("execute {cypher}: {error}"))
}

fn hits(rows: &[namidb_query::Row]) -> Vec<(NodeId, u64)> {
    rows.iter()
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

fn bm25(query: &str, k: Option<usize>, filter: Option<&str>) -> (String, Params) {
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::String(query.into()));
    let k = k.map_or(String::new(), |k| format!(", k: {k}"));
    let filter = filter.map_or(String::new(), |filter| format!(", filter: {filter}"));
    (
        format!(
            "CALL search.bm25({{label: '{LABEL}', text_properties: ['title', 'body'], \
             query: $q{k}{filter}}}) YIELD node, score RETURN node, score"
        ),
        params,
    )
}

/// Every query, k and filter: the indexed snapshot answers exactly like the
/// flat oracle, and the plain-term queries were served by the index.
async fn assert_parity(indexed: &Snapshot<'_>, oracle: &Snapshot<'_>, context: &str) {
    for query in QUERIES {
        for k in [Some(1), Some(5), Some(20), None] {
            for filter in [None, Some("{vigente: true}"), Some("{vigente: false}")] {
                let (cypher, params) = bm25(query, k, filter);
                let served = hits(&run(indexed, &cypher, &params).await);
                let flat = hits(&run(oracle, &cypher, &params).await);
                assert_eq!(
                    served, flat,
                    "{context}: {query:?} k={k:?} filter={filter:?}"
                );
            }
        }
        if !query.contains('*') {
            let parsed = namidb_storage::text::parse_query(query);
            assert!(
                indexed
                    .text_search(INDEX, LABEL, &parsed, Some(10))
                    .await
                    .unwrap()
                    .is_some(),
                "{context}: {query:?} fell back to the exact scan"
            );
        }
    }
}

async fn layout_parity(name: &str, layout: Layout) {
    let (mut writer, mut model) = indexed(name, layout).await;
    let clean = model.oracle(&format!("{name}-oracle-clean")).await;
    assert_parity(&writer.snapshot(), &clean.snapshot(), "clean").await;
    drop(clean);

    mutate(&mut writer, &mut model).await;
    let oracle = model.oracle(&format!("{name}-oracle")).await;
    assert_parity(&writer.snapshot(), &oracle.snapshot(), "merged").await;

    // A transaction sees its own writes: the overlay holds a delete, a
    // rewrite and an insert on top of the committed memtable.
    let ids = model.ids();
    model.delete(&mut writer, ids[40]);
    let mut doc = model.docs[&ids[41]].clone();
    doc.body = Some("graph graph staged rewrite".into());
    model.write(&mut writer, ids[41], doc);
    let mut doc = model.new_doc(4000);
    doc.body = Some("staged graph insert storage".into());
    model.write(&mut writer, NodeId::new(), doc);
    let staged_oracle = model.oracle(&format!("{name}-oracle-staged")).await;
    assert_parity(
        &writer.overlay_snapshot(),
        &staged_oracle.snapshot(),
        "overlay",
    )
    .await;
    writer.commit_batch().await.unwrap();

    // Flushed, the index holds everything again.
    writer.flush(schema()).await.unwrap();
    assert_parity(&writer.snapshot(), &staged_oracle.snapshot(), "flushed").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lone_ft4_base_merges_unflushed_documents_exactly() {
    layout_parity("fresh-ft4-base", Layout::Ft4Base).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ft4_segments_merge_unflushed_documents_exactly() {
    layout_parity("fresh-ft4-segments", Layout::Ft4Segments).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lone_v3_base_merges_unflushed_documents_exactly() {
    layout_parity("fresh-v3-base", Layout::LoneV3Base).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_v3_base_with_deltas_merges_unflushed_documents_exactly() {
    layout_parity("fresh-v3-deltas", Layout::V3BaseWithDeltas).await;
}

/// A native filter value that flips in the memtable takes effect at once on
/// every leg: the document enters or leaves `search.bm25`, `search.vector`
/// and `search.hybrid` results before any flush.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_filter_value_flipped_in_the_memtable_applies_immediately_on_every_leg() {
    let (mut writer, mut model) = indexed("fresh-filter-flip", Layout::Ft4Base).await;
    let ids = model.ids();
    let target = ids[3];
    let mut doc = model.docs[&target].clone();
    doc.vigente = false;
    doc.body = Some("zyzzyva graph".into());
    model.write(&mut writer, target, doc.clone());
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    let query_vector = doc.embedding.clone();
    let contains = |rows: &[namidb_query::Row]| {
        rows.iter().any(|row| match row.get("node") {
            Some(RuntimeValue::Node(node)) => node.id == target,
            _ => false,
        })
    };
    let legs = |vigente: bool| {
        let mut params = Params::new();
        params.insert("q".into(), RuntimeValue::String("zyzzyva".into()));
        params.insert("v".into(), RuntimeValue::Vector(query_vector.clone()));
        let filter = format!("{{vigente: {vigente}}}");
        [
            format!(
                "CALL search.bm25({{label: '{LABEL}', text_properties: ['title', 'body'], \
                 query: $q, k: 5, filter: {filter}}}) YIELD node, score RETURN node, score"
            ),
            format!(
                "CALL search.vector({{label: '{LABEL}', property: 'embedding', query: $v, \
                 k: 5, filter: {filter}}}) YIELD node, score RETURN node, score"
            ),
            format!(
                "CALL search.hybrid({{label: '{LABEL}', query_text: $q, \
                 text_properties: ['title', 'body'], query_vector: $v, \
                 vector_property: 'embedding', k: 5, filter: {filter}}}) \
                 YIELD node, score RETURN node, score"
            ),
        ]
        .map(|cypher| (cypher, params.clone()))
    };
    for (cypher, params) in legs(true) {
        assert!(
            !contains(&run(&writer.snapshot(), &cypher, &params).await),
            "{cypher}"
        );
    }
    for (cypher, params) in legs(false) {
        assert!(
            contains(&run(&writer.snapshot(), &cypher, &params).await),
            "{cypher}"
        );
    }

    // The flip, unflushed.
    doc.vigente = true;
    model.write(&mut writer, target, doc.clone());
    writer.commit_batch().await.unwrap();
    for (cypher, params) in legs(true) {
        assert!(
            contains(&run(&writer.snapshot(), &cypher, &params).await),
            "enters: {cypher}"
        );
    }
    for (cypher, params) in legs(false) {
        assert!(
            !contains(&run(&writer.snapshot(), &cypher, &params).await),
            "leaves: {cypher}"
        );
    }
    let oracle = model.oracle("fresh-filter-flip-oracle").await;
    for (cypher, params) in legs(true).into_iter().chain(legs(false)).take(1) {
        assert_eq!(
            hits(&run(&writer.snapshot(), &cypher, &params).await),
            hits(&run(&oracle.snapshot(), &cypher, &params).await),
            "{cypher}"
        );
    }
    let parsed = namidb_storage::text::parse_query("zyzzyva");
    assert!(
        writer
            .snapshot()
            .text_search_filter_groups(
                INDEX,
                LABEL,
                &parsed,
                Some(5),
                &[("vigente".into(), vec![CoreValue::Bool(true)])],
            )
            .await
            .unwrap()
            .is_some_and(|hits| hits.iter().any(|(id, _)| *id == target)),
        "the native filter group serves and sees the memtable value"
    );
}

/// Past the fresh-document bound the read takes the exact scan (still
/// exact), counts it, and asks for an early flush.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn past_the_bound_the_read_scans_and_requests_a_flush() {
    let (mut writer, mut model) = indexed("fresh-overflow", Layout::Ft4Base).await;
    let limit = namidb_storage::search_freshness::fresh_text_max_docs();
    let before = namidb_storage::route_telemetry::snapshot().text_fresh_overflow;
    let _ = namidb_storage::search_freshness::take_flush_request("fresh-overflow");
    for ordinal in 0..=limit {
        let doc = model.new_doc(10_000 + ordinal);
        model.write(&mut writer, NodeId::new(), doc);
    }
    writer.commit_batch().await.unwrap();
    let parsed = namidb_storage::text::parse_query("graph");
    assert!(writer
        .snapshot()
        .text_search(INDEX, LABEL, &parsed, Some(10))
        .await
        .unwrap()
        .is_none());
    assert!(namidb_storage::route_telemetry::snapshot().text_fresh_overflow > before);
    assert!(namidb_storage::search_freshness::take_flush_request(
        "fresh-overflow"
    ));
    let oracle = model.oracle("fresh-overflow-oracle").await;
    let (cypher, params) = bm25("graph engine", Some(10), None);
    assert_eq!(
        hits(&run(&writer.snapshot(), &cypher, &params).await),
        hits(&run(&oracle.snapshot(), &cypher, &params).await)
    );
}
