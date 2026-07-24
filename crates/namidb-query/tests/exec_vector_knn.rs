//! End-to-end vector KNN: a `cosine_similarity(...)` ORDER BY over stored
//! embeddings ranks nodes by semantic closeness, with a WHERE pre-filter on the
//! candidate set. This is the Phase 1 "no dedicated operator" path: semantic
//! search expressed through the existing scan + sort + limit operators.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::parser::{Expression, ExpressionKind, SourceSpan};
use namidb_query::plan::logical::VectorDistance;
use namidb_query::plan::RowCount;
use namidb_query::{execute, lower, parse, LogicalPlan, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// Seed three `:Doc` nodes whose 3-D embeddings point in clearly different
/// directions, plus one with no embedding to exercise the pre-filter.
async fn seed_docs(name: &str) -> WriterSession {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let docs: [(&str, Option<Vec<f32>>); 4] = [
        ("x-ish", Some(vec![1.0, 0.0, 0.0])),
        ("y-ish", Some(vec![0.0, 1.0, 0.0])),
        ("xy-ish", Some(vec![0.9, 0.1, 0.0])),
        ("no-embedding", None),
    ];
    for (title, emb) in docs {
        let mut p = BTreeMap::new();
        p.insert("title".into(), CoreValue::Str(title.into()));
        if let Some(emb) = emb {
            p.insert("embedding".into(), CoreValue::Vec(emb));
        }
        writer
            .upsert_node(
                "Doc",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
}

fn titles(rows: &[namidb_query::Row]) -> Vec<String> {
    rows.iter()
        .filter_map(|r| match r.get("title") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn cosine_knn_ranks_by_closeness() {
    let writer = seed_docs("knn-rank").await;
    let snap = writer.snapshot();

    // Query along the x axis: x-ish closest, then xy-ish, then y-ish.
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));

    let plan = lower(
        &parse(
            "MATCH (d:Doc) WHERE d.embedding IS NOT NULL \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 2",
        )
        .unwrap(),
    )
    .unwrap();

    let rows = execute(&plan, &snap, &params).await.unwrap();
    assert_eq!(
        titles(&rows),
        vec!["x-ish".to_string(), "xy-ish".to_string()]
    );
}

#[tokio::test]
async fn cosine_knn_honors_parameterized_limit() {
    // A parameterized `LIMIT $k` must use the bound value. The vector-search
    // lowering resolves the KNN into an operator that carries the limit; a
    // hardcoded `unwrap_or(10)` default silently ignored $k, returning up to 10
    // rows regardless. Assert $k=1 yields exactly one row (the nearest).
    let writer = seed_docs("knn-param-limit").await;
    let snap = writer.snapshot();

    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    params.insert("k".into(), RuntimeValue::Integer(1));

    let plan = lower(
        &parse(
            "MATCH (d:Doc) WHERE d.embedding IS NOT NULL \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT $k",
        )
        .unwrap(),
    )
    .unwrap();

    let rows = execute(&plan, &snap, &params).await.unwrap();
    assert_eq!(
        titles(&rows),
        vec!["x-ish".to_string()],
        "LIMIT $k=1 → 1 row"
    );
}

#[tokio::test]
async fn knn_prefilter_excludes_nodes_without_embedding() {
    let writer = seed_docs("knn-prefilter").await;
    let snap = writer.snapshot();

    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![0.0, 1.0, 0.0]));

    // No LIMIT: the WHERE pre-filter must drop the embedding-less node, so only
    // the three real embeddings come back (closest to the y axis first).
    let plan = lower(
        &parse(
            "MATCH (d:Doc) WHERE d.embedding IS NOT NULL \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC",
        )
        .unwrap(),
    )
    .unwrap();

    let rows = execute(&plan, &snap, &params).await.unwrap();
    let got = titles(&rows);
    assert_eq!(got.len(), 3, "the no-embedding doc must be filtered out");
    assert_eq!(got.first().map(String::as_str), Some("y-ish"));
    assert!(!got.iter().any(|t| t == "no-embedding"));
}

#[tokio::test]
async fn typeless_knn_returns_multilabel_once_and_includes_unlabelled() {
    let mut writer = WriterSession::open(store(), paths("knn-typeless"))
        .await
        .unwrap();
    let rec = |title: &str, embedding: Vec<f32>| NodeWriteRecord {
        properties: BTreeMap::from([
            ("title".into(), CoreValue::Str(title.into())),
            ("embedding".into(), CoreValue::Vec(embedding)),
        ]),
        schema_version: 1,
        ..Default::default()
    };
    writer
        .upsert_node_with_labels(
            ["A".to_string(), "B".to_string()],
            NodeId::new(),
            &rec("multi", vec![1.0, 0.0, 0.0]),
        )
        .unwrap();
    writer
        .upsert_node_with_labels(
            Vec::<String>::new(),
            NodeId::new(),
            &rec("bare", vec![0.8, 0.2, 0.0]),
        )
        .unwrap();
    writer
        .upsert_node("C", NodeId::new(), &rec("other", vec![0.0, 1.0, 0.0]))
        .unwrap();
    writer.commit_batch().await.unwrap();

    // Exercise VectorSearch's exact label-less fallback directly. No index can
    // serve a cross-label KNN, so it must scan physical nodes once.
    let plan = LogicalPlan::VectorSearch {
        label: None,
        alias: "d".into(),
        property: "embedding".into(),
        query: Expression {
            kind: ExpressionKind::Parameter("q".into()),
            span: SourceSpan::point(0),
        },
        k: RowCount::Const(10),
        distance: VectorDistance::Cosine,
        score_alias: "score".into(),
        post_filter: None,
    };
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    let rows = execute(&plan, &writer.snapshot(), &params).await.unwrap();
    let got: Vec<String> = rows
        .iter()
        .map(|row| match row.get("d") {
            Some(RuntimeValue::Node(node)) => match node.properties.get("title") {
                Some(RuntimeValue::String(title)) => title.clone(),
                other => panic!("expected title string, got {other:?}"),
            },
            other => panic!("expected node, got {other:?}"),
        })
        .collect();

    assert_eq!(got, vec!["multi", "bare", "other"]);
}

// ── RFC-030 indexed path: freshness (delta-union) + filtered ANN ──────────
// These exercise the Vamana index + the optimizer's VectorSearch rewrite, so
// they run the full `optimize` pass with a catalog and require the feature.
#[cfg(feature = "vector-index")]
mod indexed {
    use super::*;
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
    use namidb_query::{optimize, StatsCatalog};
    use namidb_storage::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
    use object_store::ObjectStoreExt;

    const DIM: u32 = 4;
    const KNN3: &str = "MATCH (d:Doc) WHERE d.embedding IS NOT NULL \
         RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
         ORDER BY score DESC LIMIT 3";

    fn schema() -> Schema {
        SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("embedding", DataType::FloatVector { dim: DIM }, false)
                        .unwrap(),
                    PropertyDef::new("kind", DataType::Utf8, true).unwrap(),
                    PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                ],
            })
            .unwrap()
            .build()
    }

    fn rec(title: &str, kind: &str, emb: Vec<f32>) -> NodeWriteRecord {
        let mut p = BTreeMap::new();
        p.insert("title".into(), CoreValue::Str(title.into()));
        p.insert("kind".into(), CoreValue::Str(kind.into()));
        p.insert("embedding".into(), CoreValue::Vec(emb));
        NodeWriteRecord {
            properties: p,
            schema_version: 1,
            ..Default::default()
        }
    }

    /// Register a Doc cosine index, write the docs across two L0 SSTs, and
    /// compact so the `.vg` is materialised. Returns the writer + title→id map.
    async fn build_index(
        name: &str,
        docs: &[(&str, &str, Vec<f32>)],
    ) -> (WriterSession, BTreeMap<String, NodeId>) {
        let mut w = WriterSession::open(store(), paths(name)).await.unwrap();
        w.register_vector_index(
            VectorIndexDescriptor {
                name: "doc_emb".into(),
                label: "Doc".into(),
                property: "embedding".into(),
                dim: DIM,
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
        let mut ids = BTreeMap::new();
        let half = docs.len().div_ceil(2);
        for (i, (title, kind, emb)) in docs.iter().enumerate() {
            let id = NodeId::new();
            ids.insert(title.to_string(), id);
            w.upsert_node("Doc", id, &rec(title, kind, emb.clone()))
                .unwrap();
            if i + 1 == half {
                w.flush(schema()).await.unwrap(); // L0 #1
            }
        }
        w.flush(schema()).await.unwrap(); // L0 #2
        w.compact_l0(&schema()).await.unwrap(); // build the .vg
        (w, ids)
    }

    async fn run(w: &WriterSession, cypher: &str, q: Vec<f32>) -> Vec<namidb_query::Row> {
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
        let mut params = Params::new();
        params.insert("q".into(), RuntimeValue::Vector(q));
        execute(&plan, &snap, &params).await.unwrap()
    }

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    /// Exact brute-force top-k titles over a live (title, embedding) set.
    fn exact_topk(live: &[(String, Vec<f32>)], q: &[f32], k: usize) -> Vec<String> {
        let mut s: Vec<(f64, String)> = live.iter().map(|(t, e)| (cos(e, q), t.clone())).collect();
        s.sort_by(|a, b| b.0.total_cmp(&a.0));
        s.truncate(k);
        s.into_iter().map(|(_, t)| t).collect()
    }

    #[tokio::test]
    async fn index_knn_equals_flat_and_sees_fresh_memtable_writes() {
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.0, 0.0, 1.0, 0.0]),
            ("d", "Y", vec![0.0, 0.0, 0.0, 1.0]),
            ("e", "X", vec![0.9, 0.1, 0.0, 0.0]),
            ("f", "Y", vec![0.1, 0.0, 0.9, 0.0]),
        ];
        let (mut w, ids) = build_index("idx-fresh", &docs).await;
        let q = vec![1.0, 0.0, 0.0, 0.0];

        // Baseline: index path (no delta) equals the exact flat KNN.
        let live0: Vec<(String, Vec<f32>)> = docs
            .iter()
            .map(|(t, _, e)| (t.to_string(), e.clone()))
            .collect();
        assert_eq!(
            titles(&run(&w, KNN3, q.clone()).await),
            exact_topk(&live0, &q, 3)
        );

        // Fresh memtable writes the index has NOT absorbed: a brand-new top hit,
        // an update that moves `b` onto the query, and a delete of the old #1.
        w.upsert_node("Doc", NodeId::new(), &rec("new", "X", q.clone()))
            .unwrap();
        w.upsert_node("Doc", ids["b"], &rec("b", "X", vec![0.99, 0.01, 0.0, 0.0]))
            .unwrap();
        w.tombstone_node("Doc", ids["a"]).unwrap();
        w.commit_batch().await.unwrap();

        let got = titles(&run(&w, KNN3, q.clone()).await);

        // Exact flat KNN over the live set: `a` removed, `b` moved, `new` added.
        let mut live1: Vec<(String, Vec<f32>)> = docs
            .iter()
            .filter(|(t, _, _)| *t != "a")
            .map(|(t, _, e)| {
                let emb = if *t == "b" {
                    vec![0.99, 0.01, 0.0, 0.0]
                } else {
                    e.clone()
                };
                (t.to_string(), emb)
            })
            .collect();
        live1.push(("new".to_string(), q.clone()));

        assert_eq!(
            got,
            exact_topk(&live1, &q, 3),
            "ANN+delta must equal flat KNN"
        );
        assert!(
            got.contains(&"new".to_string()),
            "fresh memtable node visible"
        );
        assert!(!got.contains(&"a".to_string()), "deleted node excluded");
    }

    #[tokio::test]
    async fn undecodable_vector_index_with_full_delta_falls_back_to_flat() {
        // Regression: storage used to collapse "the .vg failed to decode" into
        // an empty hit list. If the fresh memtable delta alone could fill k, the
        // executor then returned that delta without flat-scanning the persisted
        // corpus, potentially omitting a much better neighbour.
        let backing = store();
        let p = paths("idx-corrupt-with-delta");
        let mut w = WriterSession::open(backing.clone(), p.clone())
            .await
            .unwrap();
        w.register_vector_index(
            VectorIndexDescriptor {
                name: "doc_emb".into(),
                label: "Doc".into(),
                property: "embedding".into(),
                dim: DIM,
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
        w.upsert_node(
            "Doc",
            NodeId::new(),
            &rec("persisted-best", "X", vec![1.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
        w.flush(schema()).await.unwrap();
        w.upsert_node(
            "Doc",
            NodeId::new(),
            &rec("persisted-far", "X", vec![0.0, 0.0, 1.0, 0.0]),
        )
        .unwrap();
        w.flush(schema()).await.unwrap();
        w.compact_l0(&schema()).await.unwrap();

        let vg_path = {
            let snap = w.snapshot();
            snap.manifest()
                .manifest
                .ssts
                .iter()
                .find(|d| d.kind == namidb_storage::manifest::SstKind::VectorGraph)
                .expect("compaction builds .vg")
                .path
                .clone()
        };
        let absolute = format!("{}/{}", p.namespace_prefix().as_ref(), vg_path);
        backing
            .put(
                &object_store::path::Path::from(absolute),
                object_store::PutPayload::from_static(b"NAMIVG00corrupt"),
            )
            .await
            .unwrap();

        // One inferior fresh candidate is enough to fill LIMIT 1. It must not
        // mask the persisted exact best merely because the graph is unreadable.
        w.upsert_node(
            "Doc",
            NodeId::new(),
            &rec("fresh-inferior", "X", vec![0.0, 1.0, 0.0, 0.0]),
        )
        .unwrap();
        w.commit_batch().await.unwrap();

        let cypher = "MATCH (d:Doc) RETURN d.title AS title, \
             cosine_similarity(d.embedding, $q) AS score ORDER BY score DESC LIMIT 1";
        let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0, 0.0]).await);
        assert_eq!(got, vec!["persisted-best".to_string()]);
    }

    #[tokio::test]
    async fn index_knn_sees_flushed_but_uncompacted_l0_writes() {
        // A node flushed to an L0 SST but NOT yet compacted into the `.vg` lives
        // in neither the index (uncompacted) nor the memtable (cleared on flush).
        // The L0 freshness gate must fall back to the exact flat scan so the
        // indexed query still sees it. (Regression: the gate matched `scope ==
        // label`, but id-primary node SSTs flush with an empty scope, so it was
        // dead — this window silently returned stale top-k.)
        let docs = vec![
            ("a", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 0.0, 1.0, 0.0]),
            ("c", "Y", vec![0.0, 0.0, 0.0, 1.0]),
            ("d", "Y", vec![0.1, 0.9, 0.0, 0.0]),
        ];
        let (mut w, _) = build_index("idx-l0-window", &docs).await;
        let q = vec![1.0, 0.0, 0.0, 0.0];

        // New top hit, flushed to L0 but deliberately NOT compacted.
        w.upsert_node("Doc", NodeId::new(), &rec("fresh-top", "X", q.clone()))
            .unwrap();
        w.flush(schema()).await.unwrap();

        let got = titles(&run(&w, KNN3, q.clone()).await);
        assert!(
            got.contains(&"fresh-top".to_string()),
            "flushed-but-uncompacted L0 node must be visible (got {got:?})"
        );
        assert_eq!(got.first().map(String::as_str), Some("fresh-top"));
    }

    #[tokio::test]
    async fn index_knn_label_filter_is_applied_not_dropped() {
        // The top hit by score is the WRONG kind: if the WHERE were dropped it
        // would surface; with filtered ANN it must be excluded.
        let docs = vec![
            ("p1", "Y", vec![1.0, 0.0, 0.0, 0.0]), // closest, but kind Y
            ("p2", "X", vec![0.95, 0.05, 0.0, 0.0]),
            ("p3", "X", vec![0.9, 0.1, 0.0, 0.0]),
            ("p4", "Y", vec![0.0, 1.0, 0.0, 0.0]),
            ("p5", "X", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let (w, _) = build_index("idx-label", &docs).await;
        let q = vec![1.0, 0.0, 0.0, 0.0];

        let cypher = "MATCH (d:Doc) WHERE d.kind = 'X' \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 2";
        let got = titles(&run(&w, cypher, q).await);
        assert_eq!(got, vec!["p2".to_string(), "p3".to_string()]);
        assert!(
            !got.contains(&"p1".to_string()),
            "wrong-kind top hit excluded"
        );
    }

    #[tokio::test]
    async fn index_knn_threshold_returns_only_passing() {
        let docs = vec![
            ("near1", "X", vec![1.0, 0.0, 0.0, 0.0]),   // 1.000
            ("near2", "X", vec![0.95, 0.05, 0.0, 0.0]), // ~0.999
            ("mid", "Y", vec![0.7, 0.3, 0.0, 0.0]),     // ~0.919
            ("far1", "Y", vec![0.0, 1.0, 0.0, 0.0]),    // 0.0
            ("far2", "X", vec![0.0, 0.0, 1.0, 0.0]),    // 0.0
        ];
        let (w, _) = build_index("idx-thresh", &docs).await;
        let q = vec![1.0, 0.0, 0.0, 0.0];

        // Threshold 0.95: only near1 + near2 clear it.
        let cypher = "MATCH (d:Doc) WHERE cosine_similarity(d.embedding, $q) >= 0.95 \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 10";
        let got = titles(&run(&w, cypher, q).await);
        assert_eq!(got, vec!["near1".to_string(), "near2".to_string()]);
        assert!(!got.iter().any(|t| t == "mid" || t == "far1" || t == "far2"));
    }

    /// Regression: a real terminal-`RETURN` KNN must *reach* the index — its
    /// optimized plan has to contain a `VectorSearch`. The result-equivalence
    /// tests above would pass even if the rewrite never fired (the flat fallback
    /// is exact), so this asserts the indexed path itself is taken.
    #[tokio::test]
    async fn cypher_knn_actually_reaches_the_index() {
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.0, 0.0, 1.0, 0.0]),
            ("e", "Y", vec![0.0, 0.0, 0.0, 1.0]),
        ];
        let (w, _) = build_index("idx-reach", &docs).await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

        // Plain KNN, terminal RETURN, no WHERE (Project sits outside the TopN).
        let plain = optimize(
            lower(
                &parse(
                    "MATCH (d:Doc) RETURN d.title AS title, \
                     cosine_similarity(d.embedding, $q) AS score \
                     ORDER BY score DESC LIMIT 3",
                )
                .unwrap(),
            )
            .unwrap(),
            &catalog,
        );
        assert!(
            serde_json::to_string(&plain)
                .unwrap()
                .contains("VectorSearch"),
            "terminal-RETURN KNN must rewrite to the indexed VectorSearch path"
        );

        // The entity-resolution pattern: label filter + similarity threshold.
        // References only `d`, so it folds into `post_filter` and still rewrites.
        let filtered = optimize(
            lower(
                &parse(
                    "MATCH (d:Doc) WHERE d.kind = 'X' \
                     AND cosine_similarity(d.embedding, $q) >= 0.5 \
                     RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
                     ORDER BY score DESC LIMIT 3",
                )
                .unwrap(),
            )
            .unwrap(),
            &catalog,
        );
        assert!(
            serde_json::to_string(&filtered)
                .unwrap()
                .contains("VectorSearch"),
            "filtered KNN (label + threshold) must also reach the index"
        );
    }

    /// WITH-based KNN shapes must ALSO reach the index. Real lowering emits a
    /// Project directly above the TopN per stage, so a WITH stage produces
    /// Project{Project{TopN}} — which the rewrite used to miss, silently
    /// flat-scanning despite the index.
    #[tokio::test]
    async fn with_based_knn_reaches_the_index() {
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let (w, _) = build_index("idx-with-reach", &docs).await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

        for cypher in [
            // WITH carrying the score forward, then RETURN.
            "MATCH (d:Doc) WITH d, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 3 RETURN d.title AS title, score",
            // WITH ordering by the distance expression, projecting only d.
            "MATCH (d:Doc) WITH d ORDER BY cosine_similarity(d.embedding, $q) DESC \
             LIMIT 3 RETURN d.title AS title",
        ] {
            let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
            assert!(
                serde_json::to_string(&plan)
                    .unwrap()
                    .contains("VectorSearch"),
                "WITH-based KNN must reach the index, plan: {cypher}"
            );
        }
    }

    /// Build a `.vg` for an arbitrary metric (mirrors `build_index`, which is
    /// cosine-only).
    async fn build_index_metric(
        name: &str,
        metric: VectorMetric,
        docs: &[(&str, &str, Vec<f32>)],
    ) -> WriterSession {
        build_index_q(name, metric, VectorQuantization::None, docs).await
    }

    async fn build_index_q(
        name: &str,
        metric: VectorMetric,
        quantization: VectorQuantization,
        docs: &[(&str, &str, Vec<f32>)],
    ) -> WriterSession {
        let mut w = WriterSession::open(store(), paths(name)).await.unwrap();
        w.register_vector_index(
            VectorIndexDescriptor {
                name: "doc_emb".into(),
                label: "Doc".into(),
                property: "embedding".into(),
                dim: DIM,
                metric,
                r: 32,
                l_build: 64,
                alpha: 1.2,
                quantization,
            },
            false,
        )
        .await
        .unwrap();
        let half = docs.len().div_ceil(2);
        for (i, (title, kind, emb)) in docs.iter().enumerate() {
            w.upsert_node("Doc", NodeId::new(), &rec(title, kind, emb.clone()))
                .unwrap();
            if i + 1 == half {
                w.flush(schema()).await.unwrap();
            }
        }
        w.flush(schema()).await.unwrap();
        w.compact_l0(&schema()).await.unwrap();
        w
    }

    fn exact_topk_dot(live: &[(String, Vec<f32>)], q: &[f32], k: usize) -> Vec<String> {
        let mut s: Vec<(f64, String)> = live
            .iter()
            .map(|(t, e)| {
                let dot: f64 = e.iter().zip(q).map(|(x, y)| *x as f64 * *y as f64).sum();
                (dot, t.clone())
            })
            .collect();
        s.sort_by(|a, b| b.0.total_cmp(&a.0));
        s.truncate(k);
        s.into_iter().map(|(_, t)| t).collect()
    }

    fn exact_topk_l2(live: &[(String, Vec<f32>)], q: &[f32], k: usize) -> Vec<String> {
        let mut s: Vec<(f64, String)> = live
            .iter()
            .map(|(t, e)| {
                let d: f64 = e
                    .iter()
                    .zip(q)
                    .map(|(x, y)| {
                        let d = *x as f64 - *y as f64;
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt();
                (d, t.clone())
            })
            .collect();
        s.sort_by(|a, b| a.0.total_cmp(&b.0));
        s.truncate(k);
        s.into_iter().map(|(_, t)| t).collect()
    }

    #[tokio::test]
    async fn dot_knn_reaches_index_and_matches_bruteforce() {
        // Dot is magnitude-sensitive: `a` (largest magnitude along x) beats the
        // unit `b`. The rewrite must fire (DESC) and the index score = raw dot.
        let docs = vec![
            ("a", "X", vec![2.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("c", "Y", vec![0.0, 1.0, 0.0, 0.0]),
            ("e", "Y", vec![0.5, 0.5, 0.0, 0.0]),
        ];
        let w = build_index_metric("idx-dot", VectorMetric::Dot, &docs).await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let cypher = "MATCH (d:Doc) RETURN d.title AS title, \
             dot_product(d.embedding, $q) AS score ORDER BY score DESC LIMIT 3";
        let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
        assert!(
            serde_json::to_string(&plan)
                .unwrap()
                .contains("VectorSearch"),
            "dot KNN must reach the index"
        );
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let got = titles(&run(&w, cypher, q.clone()).await);
        let live: Vec<(String, Vec<f32>)> = docs
            .iter()
            .map(|(t, _, e)| (t.to_string(), e.clone()))
            .collect();
        assert_eq!(got, exact_topk_dot(&live, &q, 3));
    }

    #[tokio::test]
    async fn euclidean_knn_reaches_index_and_matches_bruteforce() {
        // Euclidean is nearest-first ASC; the rewrite must fire on ASC and the
        // index score = L2 distance, ranked ascending.
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.9, 0.1, 0.0, 0.0]),
            ("e", "Y", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let w = build_index_metric("idx-l2", VectorMetric::Euclidean, &docs).await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let cypher = "MATCH (d:Doc) RETURN d.title AS title, \
             euclidean_distance(d.embedding, $q) AS score ORDER BY score ASC LIMIT 3";
        let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
        assert!(
            serde_json::to_string(&plan)
                .unwrap()
                .contains("VectorSearch"),
            "euclidean ASC KNN must reach the index"
        );
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let got = titles(&run(&w, cypher, q.clone()).await);
        let live: Vec<(String, Vec<f32>)> = docs
            .iter()
            .map(|(t, _, e)| (t.to_string(), e.clone()))
            .collect();
        assert_eq!(got, exact_topk_l2(&live, &q, 3));
    }

    #[tokio::test]
    async fn query_nodes_procedure_serves_from_the_index() {
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.0, 0.0, 1.0, 0.0]),
            ("e", "Y", vec![0.5, 0.5, 0.0, 0.0]),
        ];
        let w = build_index_metric("idx-qn", VectorMetric::Cosine, &docs).await;
        // Neo4j-style positional call; resolves `doc_emb` by name.
        let cypher = "CALL db.index.vector.queryNodes('doc_emb', 2, $q) \
             YIELD node, score RETURN node.title AS title";
        let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0, 0.0]).await);
        // query ∥ x → a (cos 1.0) then e (cos ~0.707).
        assert_eq!(got, vec!["a".to_string(), "e".to_string()]);
    }

    #[tokio::test]
    async fn query_nodes_procedure_filter_serves_from_the_index() {
        // The same corpus, but a `filter: { kind: 'Y' }` in the optional 4th map.
        // Unfiltered, the top-2 for q∥x is [a, e] (both not all Y). Constrained to
        // kind Y, the index over-fetch must surface e (cos ~.707) then c (cos 0),
        // dropping the higher-ranked a (kind X) — i.e. the filter is applied
        // index-side, NOT as a post-truncation of an already-cut top-k.
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.0, 0.0, 1.0, 0.0]),
            ("e", "Y", vec![0.5, 0.5, 0.0, 0.0]),
        ];
        let w = build_index_metric("idx-qn-filter", VectorMetric::Cosine, &docs).await;
        let cypher =
            "CALL db.index.vector.queryNodes('doc_emb', 2, $q, { filter: { kind: 'Y' } }) \
             YIELD node, score RETURN node.title AS title";
        let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0, 0.0]).await);
        assert_eq!(got, vec!["e".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn int8_quantized_index_reaches_index_and_ranks_correctly() {
        // An int8-quantized cosine index serves the same KNN (lossy but the
        // well-separated fixtures rank identically to f32/brute force).
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.9, 0.1, 0.0, 0.0]),
            ("e", "Y", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let w = build_index_q(
            "idx-i8",
            VectorMetric::Cosine,
            VectorQuantization::Int8,
            &docs,
        )
        .await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let cypher = "MATCH (d:Doc) RETURN d.title AS title, \
             cosine_similarity(d.embedding, $q) AS score ORDER BY score DESC LIMIT 2";
        let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
        assert!(
            serde_json::to_string(&plan)
                .unwrap()
                .contains("VectorSearch"),
            "int8 cosine KNN must reach the index"
        );
        let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0, 0.0]).await);
        // query ∥ x → a closest, then c (≈x).
        assert_eq!(got, vec!["a".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn int8_index_serves_exact_scores_not_quantized_ones() {
        // `a`'s tiny second component quantizes with a ~60% relative error
        // (0.005 → code 1 → 0.00787), so the quantized cosine against a
        // y-axis query is ~0.00787 while the exact score is ~0.005. The index
        // must rescore candidates with the true f32 metric: the served score
        // equals the flat scan's to fp tolerance, and stays identical whether
        // the node lives in the index or the fresh delta.
        let docs = vec![
            ("a", "X", vec![1.0, 0.005, 0.0, 0.0]),
            ("b", "X", vec![0.0, 0.0, 1.0, 0.0]),
            ("c", "Y", vec![0.5, 0.5, 0.0, 0.0]),
        ];
        let w = build_index_q(
            "idx-i8-rescore",
            VectorMetric::Cosine,
            VectorQuantization::Int8,
            &docs,
        )
        .await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let cypher = "MATCH (d:Doc) RETURN d.title AS title, \
             cosine_similarity(d.embedding, $q) AS score ORDER BY score DESC LIMIT 3";
        let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
        assert!(
            serde_json::to_string(&plan)
                .unwrap()
                .contains("VectorSearch"),
            "the scores must come from the index path, not a flat scan"
        );
        let rows = run(&w, cypher, vec![0.0, 1.0, 0.0, 0.0]).await;
        let a_score = rows
            .iter()
            .find(|r| matches!(r.get("title"), Some(RuntimeValue::String(t)) if t == "a"))
            .and_then(|r| match r.get("score") {
                Some(RuntimeValue::Float(s)) => Some(*s),
                _ => None,
            })
            .expect("`a` must be in the top-3");
        // Exact cosine of a=[1, 0.005, 0, 0] against q=[0, 1, 0, 0].
        let exact = 0.005f64 / (1.0f64 + 0.005 * 0.005).sqrt();
        assert!(
            (a_score - exact).abs() < 1e-6,
            "served score {a_score} must be the exact cosine {exact}, \
             not the quantized ~0.00787"
        );
    }

    // ── Zero-magnitude cosine semantics (issue h): index ≡ flat ≡ builtin ──

    #[tokio::test]
    async fn index_stored_zero_vector_absent_matches_flat() {
        // A stored all-zero vector makes cosine undefined → it must be ABSENT from
        // a KNN result on the index path, exactly as the flat scan drops it.
        let docs = vec![
            ("zero", "X", vec![0.0, 0.0, 0.0, 0.0]),
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.9, 0.1, 0.0, 0.0]),
        ];
        let w = build_index_metric("idx-zero-stored", VectorMetric::Cosine, &docs).await;
        let cypher = "MATCH (d:Doc) RETURN d.title AS title, \
             cosine_similarity(d.embedding, $q) AS score ORDER BY score DESC LIMIT 3";
        let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0, 0.0]).await);
        assert!(
            !got.contains(&"zero".to_string()),
            "all-zero stored vector must be dropped: {got:?}"
        );
        // The three nonzero docs in similarity order (b has cosine 0 but is a valid
        // nonzero vector, so it stays).
        assert_eq!(got, vec!["a".to_string(), "c".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn index_zero_query_returns_empty_matches_flat() {
        // A zero-magnitude QUERY makes cosine undefined for every candidate. The
        // flat scan returns []; the index rerank would otherwise return k rows
        // scored 0.0 — the zero-query guard makes the index agree (empty).
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.9, 0.1, 0.0, 0.0]),
        ];
        let w = build_index_metric("idx-zero-query", VectorMetric::Cosine, &docs).await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        // ORDER BY form …
        let order = "MATCH (d:Doc) RETURN d.title AS title, \
             cosine_similarity(d.embedding, $q) AS score ORDER BY score DESC LIMIT 3";
        let plan = optimize(lower(&parse(order).unwrap()).unwrap(), &catalog);
        assert!(
            serde_json::to_string(&plan)
                .unwrap()
                .contains("VectorSearch"),
            "the empty result must come from the index path, not a flat scan"
        );
        assert!(
            titles(&run(&w, order, vec![0.0, 0.0, 0.0, 0.0]).await).is_empty(),
            "zero query → empty (ORDER BY form)"
        );
        // … and the explicit-threshold form the limitation calls out.
        let threshold = "MATCH (d:Doc) WHERE cosine_similarity(d.embedding, $q) >= 0.0 \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 10";
        assert!(
            titles(&run(&w, threshold, vec![0.0, 0.0, 0.0, 0.0]).await).is_empty(),
            "zero query → empty (>= threshold form)"
        );
    }

    // ── Adaptive filtered-ANN widening (issue d) ─────────────────────────────

    #[tokio::test]
    async fn widening_serves_selective_filter_correctly() {
        // 95 "common" docs cluster tightly around the query (cosine ≈ 1) and 5
        // "rare" docs sit far away (cosine ≈ 0.3). For q∥x the index's first
        // over-fetched window (8·k) is ALL common docs — zero `rare` survivors —
        // so the widening loop must grow the fetch (round 2) to reach the rare
        // docs before the flat fallback. Either way the result must be the exact
        // top-k of the rare subset.
        let mut owned: Vec<(String, &'static str, Vec<f32>)> = Vec::new();
        for i in 0..95 {
            // Tiny per-doc perturbation off the x-axis keeps them distinct but all
            // near-parallel to q (high cosine).
            let p = (i as f32) * 1e-4;
            owned.push((format!("c{i}"), "common", vec![1.0, p, 0.0, 0.0]));
        }
        for i in 0..5 {
            let p = (i as f32) * 1e-3;
            owned.push((format!("r{i}"), "rare", vec![0.3, 0.95 + p, 0.0, 0.0]));
        }
        let docs: Vec<(&str, &str, Vec<f32>)> = owned
            .iter()
            .map(|(t, k, e)| (t.as_str(), *k, e.clone()))
            .collect();
        let w = build_index_metric("idx-widen", VectorMetric::Cosine, &docs).await;
        let cypher = "MATCH (d:Doc) WHERE d.kind = 'rare' \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 3";
        let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0, 0.0]).await);
        let live: Vec<(String, Vec<f32>)> = owned
            .iter()
            .filter(|(_, k, _)| *k == "rare")
            .map(|(t, _, e)| (t.clone(), e.clone()))
            .collect();
        assert_eq!(got, exact_topk(&live, &[1.0, 0.0, 0.0, 0.0], 3));
        assert!(
            got.iter().all(|t| t.starts_with('r')),
            "only rare docs: {got:?}"
        );
    }

    #[tokio::test]
    async fn widening_too_selective_falls_back_to_flat() {
        // A filter that matches exactly ONE doc with LIMIT 5: the index is
        // exhausted before k survivors accumulate, so the loop breaks and the flat
        // fallback returns the single correct match (never a short, wrong result).
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "Y", vec![0.9, 0.1, 0.0, 0.0]),
            ("c", "Y", vec![0.0, 1.0, 0.0, 0.0]),
            ("d", "Y", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let w = build_index_metric("idx-widen-fallback", VectorMetric::Cosine, &docs).await;
        let cypher = "MATCH (n:Doc) WHERE n.kind = 'X' \
             RETURN n.title AS title, cosine_similarity(n.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 5";
        let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0, 0.0]).await);
        assert_eq!(got, vec!["a".to_string()]);
    }

    // ── Natural-form beam width via `$__vector_ef` (issue e) ──────────────────

    #[tokio::test]
    async fn natural_form_filter_and_vector_ef_coexist() {
        // The natural/operator form is the only one with real filtered ANN, and it
        // now also honours a beam-width override via the reserved `$__vector_ef`
        // param — so filtered-ANN + tunable ef finally compose in one query.
        let docs = vec![
            ("a", "X", vec![1.0, 0.0, 0.0, 0.0]),
            ("b", "X", vec![0.0, 1.0, 0.0, 0.0]),
            ("c", "Y", vec![0.9, 0.1, 0.0, 0.0]),
            ("e", "Y", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let w = build_index_metric("idx-ef", VectorMetric::Cosine, &docs).await;
        let snap = w.snapshot();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let cypher = "MATCH (d:Doc) WHERE d.kind = 'Y' \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 2";
        let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
        assert!(
            serde_json::to_string(&plan)
                .unwrap()
                .contains("VectorSearch"),
            "filtered KNN must reach the index"
        );
        let mut params = Params::new();
        params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0, 0.0]));
        params.insert("__vector_ef".into(), RuntimeValue::Integer(256));
        let rows = execute(&plan, &snap, &params).await.unwrap();
        // kind Y only → c (cos ≈ .994) then e (cos 0); the wide beam doesn't change
        // the (correct) ranking, it only raises recall.
        assert_eq!(titles(&rows), vec!["c".to_string(), "e".to_string()]);
    }
}
