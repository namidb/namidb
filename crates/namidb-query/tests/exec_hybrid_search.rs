//! End-to-end hybrid search: `CALL search.hybrid({...})` fuses a dense (vector
//! KNN) and a sparse (BM25) retrieval with Reciprocal Rank Fusion (default) or a
//! weighted-linear blend. Also covers the `search.vector` and Neo4j-compatible
//! `db.index.vector.queryNodes` procedures. These run the flat-scan path (no
//! index required), which is freshness-equivalent to the indexed path.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// Seed three `:Doc` nodes, each with a `title`, a `body` (BM25 corpus) and a
/// 3-D `embedding` (vector corpus).
async fn seed(name: &str) -> WriterSession {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let docs: [(&str, &str, Vec<f32>); 3] = [
        ("alpha", "quantum physics lecture", vec![1.0, 0.0, 0.0]),
        ("beta", "italian pasta recipe", vec![0.0, 1.0, 0.0]),
        ("gamma", "quantum pasta experiment", vec![0.8, 0.2, 0.0]),
    ];
    for (title, body, emb) in docs {
        let mut p = BTreeMap::new();
        p.insert("title".into(), CoreValue::Str(title.into()));
        p.insert("body".into(), CoreValue::Str(body.into()));
        p.insert("embedding".into(), CoreValue::Vec(emb));
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

/// Titles in result order (rows project `node.title AS title`).
fn titles(rows: &[namidb_query::Row]) -> Vec<String> {
    rows.iter()
        .filter_map(|r| match r.get("title") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

async fn run(w: &WriterSession, cypher: &str, q: Vec<f32>) -> Vec<namidb_query::Row> {
    let snap = w.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(q));
    execute(&plan, &snap, &params).await.unwrap()
}

#[tokio::test]
async fn hybrid_rrf_rewards_documents_strong_in_both_legs() {
    let w = seed("hybrid-rrf").await;
    // Query: text "quantum", vector along x. `alpha` is the top of both legs;
    // `gamma` is second in both; `beta` is only weakly in the vector leg and
    // absent from the text leg. RRF should rank alpha > gamma > beta.
    let cypher = "CALL search.hybrid({ \
         label: 'Doc', \
         query_text: 'quantum', text_property: 'body', \
         query_vector: $q, vector_property: 'embedding', \
         k: 3 \
       }) YIELD node, score RETURN node.title AS title, score";
    let rows = run(&w, cypher, vec![1.0, 0.0, 0.0]).await;
    assert_eq!(
        titles(&rows),
        vec!["alpha".to_string(), "gamma".to_string(), "beta".to_string()],
        "RRF fused order"
    );
    // Scores are non-increasing.
    let scores: Vec<f64> = rows
        .iter()
        .filter_map(|r| match r.get("score") {
            Some(RuntimeValue::Float(f)) => Some(*f),
            _ => None,
        })
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1] - 1e-9), "{scores:?}");
}

#[tokio::test]
async fn hybrid_dense_only_equals_vector_ranking() {
    let w = seed("hybrid-dense").await;
    // Dense leg only (no query_text) → identical to a pure vector KNN.
    let hybrid = "CALL search.hybrid({ label: 'Doc', \
         query_vector: $q, vector_property: 'embedding', k: 3 }) \
         YIELD node, score RETURN node.title AS title, score";
    let vector = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3 }) YIELD node, score RETURN node.title AS title, score";
    let h = titles(&run(&w, hybrid, vec![1.0, 0.0, 0.0]).await);
    let v = titles(&run(&w, vector, vec![1.0, 0.0, 0.0]).await);
    assert_eq!(h, v, "dense-only hybrid must equal search.vector");
    assert_eq!(
        h,
        vec!["alpha".to_string(), "gamma".to_string(), "beta".to_string()]
    );
}

#[tokio::test]
async fn hybrid_sparse_only_returns_only_text_matches() {
    let w = seed("hybrid-sparse").await;
    // Sparse leg only ("quantum") → alpha and gamma match; beta does not appear.
    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'quantum', text_property: 'body', k: 3 }) \
         YIELD node, score RETURN node.title AS title, score";
    let got = titles(&run(&w, cypher, vec![]).await);
    assert!(got.contains(&"alpha".to_string()) && got.contains(&"gamma".to_string()));
    assert!(
        !got.contains(&"beta".to_string()),
        "beta has no query term: {got:?}"
    );
}

#[tokio::test]
async fn hybrid_linear_fusion_runs_and_orders() {
    let w = seed("hybrid-linear").await;
    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'quantum', text_property: 'body', \
         query_vector: $q, vector_property: 'embedding', \
         fusion: 'linear', alpha: 0.5, k: 3 }) \
         YIELD node, score RETURN node.title AS title, score";
    let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0]).await);
    // alpha is best in both legs → it wins under any sensible weighting.
    assert_eq!(got.first(), Some(&"alpha".to_string()), "{got:?}");
}

#[tokio::test]
async fn hybrid_requires_at_least_one_leg() {
    let w = seed("hybrid-empty").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse("CALL search.hybrid({ label: 'Doc', k: 3 }) YIELD node, score RETURN node").unwrap(),
    )
    .unwrap();
    let err = execute(&plan, &snap, &Params::new()).await;
    assert!(err.is_err(), "hybrid with no legs configured must error");
}

// ── Procedure `filter` (issue c): index-side filtering, not post-truncation ──

#[tokio::test]
async fn vector_procedure_filter_overfetches_not_post_truncates() {
    let w = seed("vec-filter-overfetch").await;
    // q = x-axis ⇒ vector rank is alpha (1,0,0) > gamma (.8,.2,0) > beta (0,1,0).
    // Ask for k=1 but constrain to `beta`, the FARTHEST doc. A naive post-filter
    // over the k=1 top-list would take alpha, then drop it → 0 rows. The
    // over-fetch path must instead surface beta. This is the whole point of (c).
    let cypher = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 1, filter: { title: 'beta' } }) \
         YIELD node, score RETURN node.title AS title";
    let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0]).await);
    assert_eq!(
        got,
        vec!["beta".to_string()],
        "k=1 + filter must over-fetch, not post-truncate to zero"
    );
}

#[tokio::test]
async fn vector_procedure_filter_equality_and_in() {
    let w = seed("vec-filter-shapes").await;
    // Equality: only gamma, regardless of its vector rank.
    let eq = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3, filter: { title: 'gamma' } }) \
         YIELD node, score RETURN node.title AS title";
    assert_eq!(
        titles(&run(&w, eq, vec![1.0, 0.0, 0.0]).await),
        vec!["gamma".to_string()]
    );
    // List value ⇒ IN: alpha + beta, in vector-rank order — gamma is excluded
    // even though it out-ranks beta by similarity.
    let isin = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3, filter: { title: ['alpha', 'beta'] } }) \
         YIELD node, score RETURN node.title AS title";
    assert_eq!(
        titles(&run(&w, isin, vec![1.0, 0.0, 0.0]).await),
        vec!["alpha".to_string(), "beta".to_string()],
        "IN filter keeps the lower-ranked beta and drops the higher-ranked gamma"
    );
}

#[tokio::test]
async fn hybrid_filter_applies_to_both_legs() {
    let w = seed("hybrid-filter").await;
    // Both legs active, but constrained to `gamma`. The dense leg filters via
    // over-fetch; the sparse ("quantum") leg also surfaces alpha, which the
    // fused-output filter must drop.
    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'quantum', text_property: 'body', \
         query_vector: $q, vector_property: 'embedding', \
         k: 3, filter: { title: 'gamma' } }) \
         YIELD node, score RETURN node.title AS title";
    assert_eq!(
        titles(&run(&w, cypher, vec![1.0, 0.0, 0.0]).await),
        vec!["gamma".to_string()],
        "filter drops non-matching nodes from both legs"
    );
}

#[tokio::test]
async fn vector_procedure_filter_in_via_param_map() {
    // The explicit `in` operator (a reserved keyword, unusable as a bare inline
    // map key) is reachable when the filter is supplied as a $param — its runtime
    // map keys are plain strings, never parsed.
    let w = seed("vec-filter-param-in").await;
    let cypher = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3, filter: $f }) \
         YIELD node, score RETURN node.title AS title";
    let snap = w.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    // filter: { title: { in: ['alpha', 'gamma'] } }
    let mut inmap = BTreeMap::new();
    inmap.insert(
        "in".to_string(),
        RuntimeValue::List(vec![
            RuntimeValue::String("alpha".into()),
            RuntimeValue::String("gamma".into()),
        ]),
    );
    let mut filter = BTreeMap::new();
    filter.insert("title".to_string(), RuntimeValue::Map(inmap));
    params.insert("f".into(), RuntimeValue::Map(filter));
    let rows = execute(&plan, &snap, &params).await.unwrap();
    assert_eq!(
        titles(&rows),
        vec!["alpha".to_string(), "gamma".to_string()]
    );
}

#[tokio::test]
async fn vector_procedure_filter_null_value_rejected() {
    // `filter: { k: null }` would 3VL-match nothing silently; it must be rejected.
    let w = seed("vec-filter-null").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse(
            "CALL search.vector({ label: 'Doc', property: 'embedding', query: $q, k: 3, \
             filter: { title: null } }) YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    let err = execute(&plan, &snap, &params).await;
    assert!(err.is_err(), "a null filter value must be rejected");
}

#[tokio::test]
async fn hybrid_sparse_filter_does_not_starve_past_k_sparse() {
    // 520 `other`-tenant docs with a high-tf body rank above one `target` doc
    // whose tf=1 is diluted by length. This deliberately places the only valid
    // hit past the old fixed `k * 512` post-filter window. Adaptive refill must
    // either reach it or take the exact filtered fallback; returning zero is
    // never acceptable while a matching document exists.
    let mut writer = WriterSession::open(store(), paths("hybrid-starve"))
        .await
        .unwrap();
    for i in 0..520 {
        let mut p = BTreeMap::new();
        p.insert("title".into(), CoreValue::Str(format!("other{i}")));
        p.insert("tenant".into(), CoreValue::Str("other".into()));
        p.insert(
            "body".into(),
            CoreValue::Str("alpha alpha alpha alpha".into()),
        );
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
    let mut tp = BTreeMap::new();
    tp.insert("title".into(), CoreValue::Str("target".into()));
    tp.insert("tenant".into(), CoreValue::Str("acme".into()));
    // tf=1 for "alpha", padded with non-matching terms so length normalization
    // pushes it below every high-tf `other` doc.
    tp.insert(
        "body".into(),
        CoreValue::Str("alpha w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11".into()),
    );
    writer
        .upsert_node(
            "Doc",
            NodeId::new(),
            &NodeWriteRecord {
                properties: tp,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'alpha', text_property: 'body', \
         k: 1, filter: { tenant: 'acme' } }) \
         YIELD node, score RETURN node.title AS title";
    let got = titles(&run(&writer, cypher, vec![]).await);
    assert_eq!(
        got,
        vec!["target".to_string()],
        "the filter-matching doc beyond k_sparse must not be starved"
    );
}

#[tokio::test]
async fn vector_procedure_filter_unknown_operator_errors() {
    let w = seed("vec-filter-bad").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse(
            "CALL search.vector({ label: 'Doc', property: 'embedding', query: $q, k: 3, \
             filter: { title: { wat: 'x' } } }) YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    let err = execute(&plan, &snap, &params).await;
    assert!(err.is_err(), "an unknown filter operator must be rejected");
}

#[tokio::test]
async fn hybrid_rejects_alpha_out_of_range() {
    let w = seed("hybrid-alpha").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse(
            "CALL search.hybrid({ label: 'Doc', query_text: 'quantum', text_property: 'body', \
             fusion: 'linear', alpha: 1.5, k: 3 }) YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        execute(&plan, &snap, &Params::new()).await.is_err(),
        "alpha outside [0,1] must error"
    );
}

#[tokio::test]
async fn hybrid_rejects_partial_dense_leg() {
    let w = seed("hybrid-partial").await;
    let snap = w.snapshot();
    // query_vector without vector_property → error, not a silently-disabled leg.
    let plan = lower(
        &parse(
            "CALL search.hybrid({ label: 'Doc', query_vector: [1.0, 0.0, 0.0], k: 3 }) \
             YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(execute(&plan, &snap, &Params::new()).await.is_err());
}

#[tokio::test]
async fn search_vector_procedure_ranks_by_closeness() {
    let w = seed("vec-proc").await;
    let cypher = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 2 }) YIELD node, score RETURN node.title AS title";
    let got = titles(&run(&w, cypher, vec![0.0, 1.0, 0.0]).await);
    // Query along y → beta closest, then gamma.
    assert_eq!(got, vec!["beta".to_string(), "gamma".to_string()]);
}

#[cfg(feature = "text-index")]
mod indexed_sparse_filter {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, MutexGuard};

    use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
    use namidb_storage::manifest::{ManifestStore, TextIndexDescriptor};
    use namidb_storage::memtable::{MemKey, MemOp, Memtable};
    use namidb_storage::text::parse_query;
    use namidb_storage::{compact_l0_to_l1, flush, SessionCaches, WriterFence, WriterSession};
    use object_store::path::Path;
    use object_store::ObjectStoreExt;

    use super::*;

    const INDEX_NAME: &str = "doc_ft";
    const CAP_ENV: &str = "NAMIDB_HYBRID_TEXT_FILTER_CANDIDATE_CAP";

    /// These tests change the process-global candidate cap. Keep their complete
    /// async execution under one lock so each query sees the intended value.
    static CAP_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct CapEnvGuard {
        previous: Option<String>,
        _lock: MutexGuard<'static, ()>,
    }

    impl CapEnvGuard {
        fn set(value: usize) -> Self {
            let lock = CAP_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let previous = std::env::var(CAP_ENV).ok();
            std::env::set_var(CAP_ENV, value.to_string());
            Self {
                previous,
                _lock: lock,
            }
        }

        fn update(&self, value: usize) {
            std::env::set_var(CAP_ENV, value.to_string());
        }
    }

    impl Drop for CapEnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var(CAP_ENV, value),
                None => std::env::remove_var(CAP_ENV),
            }
        }
    }

    /// Cache-free probes must GET the immutable `.ft` once per widening round.
    /// Counting those GETs makes the indexed route observable without relying on
    /// execution time or on a corpus scan returning a different logical answer.
    #[derive(Debug)]
    struct TextGetProbe {
        inner: Arc<dyn ObjectStore>,
        text_gets: AtomicUsize,
        /// Generation-barrier pins. Every `search_lsm_read` coordinator
        /// invocation pins its `.slb` barrier exactly once via `store.head()`,
        /// so this counts query-side probes on the NATIVE route only — a flat
        /// fallback pins no barrier, and HEADs bypass the range cache, making
        /// the count deterministic.
        barrier_pins: AtomicUsize,
    }

    impl TextGetProbe {
        fn new() -> Self {
            Self {
                inner: Arc::new(InMemory::new()),
                text_gets: AtomicUsize::new(0),
                barrier_pins: AtomicUsize::new(0),
            }
        }

        fn text_gets(&self) -> usize {
            self.text_gets.load(Ordering::SeqCst)
        }

        fn barrier_pins(&self) -> usize {
            self.barrier_pins.load(Ordering::SeqCst)
        }
    }

    impl std::fmt::Display for TextGetProbe {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TextGetProbe({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for TextGetProbe {
        async fn put_opts(
            &self,
            location: &Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            if options.head && location.as_ref().ends_with(".slb") {
                self.barrier_pins.fetch_add(1, Ordering::SeqCst);
            }
            // Legacy monolithic bodies and native FT4 segments are both text
            // index traffic; kept for the stale/corrupt tests, whose gates
            // fire before any index fetch on either route.
            if location.as_ref().ends_with(".ft") || location.as_ref().ends_with(".ft4") {
                self.text_gets.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.get_opts(location, options).await
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }
    }

    /// Build two node L0s, compact the complete corpus into one authoritative
    /// `.ft`, then reopen without caches so each text-index probe is observable.
    async fn indexed_corpus(
        name: &str,
        docs: &[(&str, &str, &str)],
    ) -> (WriterSession, BTreeMap<String, NodeId>, Arc<TextGetProbe>) {
        indexed_corpus_with(name, docs, false).await
    }

    /// `tenant_indexed` marks `tenant` as an indexed property, which makes
    /// `text_native_filter_properties` include it and the FT4 segments
    /// advertise it — the precondition for native filter-group serving.
    async fn indexed_corpus_with(
        name: &str,
        docs: &[(&str, &str, &str)],
        tenant_indexed: bool,
    ) -> (WriterSession, BTreeMap<String, NodeId>, Arc<TextGetProbe>) {
        let probe = Arc::new(TextGetProbe::new());
        let store: Arc<dyn ObjectStore> = probe.clone();
        let namespace_paths = paths(name);
        let manifest_store = ManifestStore::new(store.clone(), namespace_paths.clone());
        let mut current = manifest_store
            .bootstrap(uuid::Uuid::now_v7())
            .await
            .unwrap();
        let label_id = current.manifest.label_dict.intern("Doc");
        current.manifest.text_indexes.push(TextIndexDescriptor::new(
            INDEX_NAME.into(),
            "Doc".into(),
            vec!["body".into()],
        ));
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![
                    PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                    PropertyDef::new("tenant", DataType::Utf8, true)
                        .unwrap()
                        .with_indexed(tenant_indexed),
                    PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                    // Numeric filter target: numeric equality is never
                    // extracted into native filter groups, so `filter:
                    // { rank: N }` routes through the plain text_search the
                    // fixture serves natively — exercising the walker's
                    // residual-widening discipline on the native route.
                    PropertyDef::new("rank", DataType::Int64, true).unwrap(),
                ],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(current.manifest.epoch);
        let mut ids = BTreeMap::new();
        let mut lsn = 0u64;
        let chunk_size = docs.len().div_ceil(2).max(1);
        for chunk in docs.chunks(chunk_size) {
            let mut memtable = Memtable::new();
            for &(title, tenant, body) in chunk {
                let id = NodeId::new();
                ids.insert(title.to_string(), id);
                let mut properties = BTreeMap::new();
                properties.insert("title".into(), CoreValue::Str(title.into()));
                properties.insert("tenant".into(), CoreValue::Str(tenant.into()));
                properties.insert(
                    "rank".into(),
                    CoreValue::I64(if tenant == "acme" { 7 } else { 0 }),
                );
                properties.insert("body".into(), CoreValue::Str(body.into()));
                let record = NodeWriteRecord {
                    properties,
                    schema_version: 1,
                    labels: vec![label_id.0],
                };
                lsn += 1;
                memtable.apply(
                    MemKey::Node { id },
                    lsn,
                    MemOp::Upsert(record.encode().unwrap()),
                );
            }
            current = flush(
                &manifest_store,
                &fence,
                &current,
                &memtable.freeze(),
                schema.clone(),
            )
            .await
            .unwrap()
            .committed;
        }
        compact_l0_to_l1(&manifest_store, &fence, &current, &schema)
            .await
            .unwrap();

        let writer = WriterSession::open_with_caches(store, namespace_paths, SessionCaches::none())
            .await
            .unwrap();
        (writer, ids, probe)
    }

    async fn authoritative_hits(
        writer: &WriterSession,
        query: &str,
        k: usize,
    ) -> Vec<(NodeId, f64)> {
        writer
            .snapshot()
            .text_search(INDEX_NAME, "Doc", &parse_query(query), Some(k))
            .await
            .unwrap()
            .expect("the compacted .ft must be authoritative")
    }

    fn other_docs(count: usize) -> Vec<(String, String, String)> {
        (0..count)
            .map(|index| {
                (
                    format!("other-{index:02}"),
                    "other".to_string(),
                    "alpha alpha alpha alpha".to_string(),
                )
            })
            .collect()
    }

    fn borrowed_docs(docs: &[(String, String, String)]) -> Vec<(&str, &str, &str)> {
        docs.iter()
            .map(|(title, tenant, body)| (title.as_str(), tenant.as_str(), body.as_str()))
            .collect()
    }

    #[tokio::test]
    async fn hybrid_filtered_bm25_authoritative_ft_widens_without_starvation() {
        let _cap = CapEnvGuard::set(64);
        let mut docs = other_docs(16);
        docs.push((
            "target".into(),
            "acme".into(),
            "alpha w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11".into(),
        ));
        let borrowed = borrowed_docs(&docs);
        let (writer, ids, probe) = indexed_corpus("hybrid-ft-widen", &borrowed).await;

        let pins_before = probe.barrier_pins();
        let initial = authoritative_hits(&writer, "alpha", 8).await;
        assert_eq!(
            probe.barrier_pins() - pins_before,
            1,
            "one authoritative probe pins the generation barrier exactly once"
        );
        assert_eq!(initial.len(), 8);
        assert!(
            initial.iter().all(|(id, _)| *id != ids["target"]),
            "the matching tenant must sit beyond the initial indexed prefix"
        );
        let widened = authoritative_hits(&writer, "alpha", 32).await;
        assert!(
            widened.iter().any(|(id, _)| *id == ids["target"]),
            "the second authoritative prefix must expose the target"
        );

        let before = probe.barrier_pins();
        let rows = run(
            &writer,
            "CALL search.hybrid({ label: 'Doc', query_text: 'alpha', \
             text_property: 'body', k: 1, k_sparse: 1, \
             filter: { rank: 7 } }) \
             YIELD node, score RETURN node.title AS title, score",
            vec![],
        )
        .await;
        assert_eq!(titles(&rows), vec!["target".to_string()]);
        // Barrier pins count coordinator invocations exactly and only on the
        // native route (a flat fallback pins no barrier), so the original
        // probe discipline is asserted verbatim: one probe at fetch=8, one
        // more at the widened fetch=32, nothing else.
        assert_eq!(
            probe.barrier_pins() - before,
            2,
            "cache-free execution must probe the generation at fetch=8 and \
             once more at the widened fetch=32"
        );
    }

    #[tokio::test]
    async fn hybrid_filtered_bm25_authoritative_ft_returns_exact_short_page() {
        let _cap = CapEnvGuard::set(64);
        let mut docs = other_docs(10);
        docs.push((
            "eligible-a".into(),
            "acme".into(),
            "alpha a1 a2 a3 a4 a5 a6".into(),
        ));
        docs.push((
            "eligible-b".into(),
            "acme".into(),
            "alpha b1 b2 b3 b4 b5 b6 b7 b8 b9 b10".into(),
        ));
        let borrowed = borrowed_docs(&docs);
        let (writer, _, probe) = indexed_corpus("hybrid-ft-short", &borrowed).await;
        let pins_before = probe.barrier_pins();
        assert_eq!(
            authoritative_hits(&writer, "alpha", 40).await.len(),
            docs.len(),
            "fetch=40 proves the authoritative matching corpus is exhausted"
        );
        assert_eq!(
            probe.barrier_pins() - pins_before,
            1,
            "one authoritative probe pins the generation barrier exactly once"
        );

        let before = probe.barrier_pins();
        let rows = run(
            &writer,
            "CALL search.hybrid({ label: 'Doc', query_text: 'alpha', \
             text_property: 'body', k: 5, k_sparse: 5, \
             filter: { rank: 7 } }) \
             YIELD node, score RETURN node.title AS title, score",
            vec![],
        )
        .await;
        assert_eq!(
            titles(&rows),
            vec!["eligible-a".to_string(), "eligible-b".to_string()]
        );
        assert_eq!(
            probe.barrier_pins() - before,
            1,
            "an exhausted generation returns the exact <k page without \
             another probe"
        );
    }

    #[tokio::test]
    async fn hybrid_filtered_bm25_candidate_cap_falls_back_with_result_parity() {
        let cap = CapEnvGuard::set(64);
        let mut docs = other_docs(16);
        docs.extend([
            (
                "eligible-a".into(),
                "acme".into(),
                "alpha a1 a2 a3 a4 a5".into(),
            ),
            (
                "eligible-b".into(),
                "acme".into(),
                "alpha b1 b2 b3 b4 b5 b6 b7".into(),
            ),
            (
                "eligible-c".into(),
                "acme".into(),
                "alpha c1 c2 c3 c4 c5 c6 c7 c8 c9".into(),
            ),
        ]);
        let borrowed = borrowed_docs(&docs);
        let (writer, ids, probe) = indexed_corpus("hybrid-ft-cap", &borrowed).await;
        let pins_before = probe.barrier_pins();
        let top_eight = authoritative_hits(&writer, "alpha", 8).await;
        assert_eq!(
            probe.barrier_pins() - pins_before,
            1,
            "one authoritative probe pins the generation barrier exactly once"
        );
        assert!(
            ["eligible-a", "eligible-b", "eligible-c"]
                .iter()
                .all(|title| top_eight.iter().all(|(id, _)| *id != ids[*title])),
            "cap=8 must end before every eligible document"
        );
        let cypher = "CALL search.hybrid({ label: 'Doc', query_text: 'alpha', \
             text_property: 'body', k: 3, k_sparse: 3, \
             filter: { rank: 7 } }) \
             YIELD node, score RETURN node.title AS title, score";

        let indexed_before = probe.barrier_pins();
        let indexed = run(&writer, cypher, vec![]).await;
        assert_eq!(
            probe.barrier_pins() - indexed_before,
            1,
            "the uncapped authoritative route exhausts this corpus in one probe"
        );

        cap.update(8);
        let capped_before = probe.barrier_pins();
        let capped = run(&writer, cypher, vec![]).await;
        assert_eq!(
            probe.barrier_pins() - capped_before,
            1,
            "the capped route consults the initial authoritative prefix \
             exactly once before the exact fallback"
        );
        assert_eq!(
            titles(&capped),
            titles(&indexed),
            "reaching the candidate cap must fall back without changing recall or rank"
        );
        assert_eq!(
            titles(&capped),
            vec![
                "eligible-a".to_string(),
                "eligible-b".to_string(),
                "eligible-c".to_string()
            ]
        );
    }

    /// Native filter-group serving: with `tenant` schema-indexed, the FT4
    /// segments advertise the property and the coordinator applies the
    /// equality at the postings level — one probe finds what the residual
    /// route needs a widening round for. Pins the equality-group fast path
    /// distinctly from the residual-widening discipline above.
    #[tokio::test]
    async fn hybrid_filtered_bm25_equality_group_served_natively() {
        let _cap = CapEnvGuard::set(64);
        let mut docs = other_docs(16);
        docs.push((
            "target".into(),
            "acme".into(),
            "alpha w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11".into(),
        ));
        let borrowed = borrowed_docs(&docs);
        let (writer, ids, probe) =
            indexed_corpus_with("hybrid-ft-native-group", &borrowed, true).await;

        let initial = authoritative_hits(&writer, "alpha", 8).await;
        assert!(
            initial.iter().all(|(id, _)| *id != ids["target"]),
            "the matching tenant must sit beyond the unfiltered top-8"
        );

        let before = probe.barrier_pins();
        let rows = run(
            &writer,
            "CALL search.hybrid({ label: 'Doc', query_text: 'alpha', \
             text_property: 'body', k: 1, k_sparse: 1, \
             filter: { tenant: 'acme' } }) \
             YIELD node, score RETURN node.title AS title, score",
            vec![],
        )
        .await;
        assert_eq!(titles(&rows), vec!["target".to_string()]);
        assert_eq!(
            probe.barrier_pins() - before,
            1,
            "postings-level native filtering finds the target in one probe, \
             where the residual route needs a widening round"
        );
    }

    /// The group-refusal twin: with `tenant` NOT schema-indexed the
    /// coordinator refuses the postings-level filter route. Refusal must
    /// downgrade to the plain native route with the equality applied
    /// residually — not surrender the whole query to the flat scorer, which
    /// pins no barrier and pays O(corpus) on every call.
    #[tokio::test]
    async fn hybrid_filtered_bm25_unindexed_group_retries_plain_native_route() {
        let _cap = CapEnvGuard::set(64);
        let mut docs = other_docs(16);
        docs.push((
            "target".into(),
            "acme".into(),
            "alpha w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11".into(),
        ));
        let borrowed = borrowed_docs(&docs);
        let (writer, ids, probe) =
            indexed_corpus_with("hybrid-ft-refused-group", &borrowed, false).await;

        let initial = authoritative_hits(&writer, "alpha", 8).await;
        assert!(
            initial.iter().all(|(id, _)| *id != ids["target"]),
            "the matching tenant must sit beyond the unfiltered top-8"
        );

        let before = probe.barrier_pins();
        let rows = run(
            &writer,
            "CALL search.hybrid({ label: 'Doc', query_text: 'alpha', \
             text_property: 'body', k: 1, k_sparse: 1, \
             filter: { tenant: 'acme' } }) \
             YIELD node, score RETURN node.title AS title, score",
            vec![],
        )
        .await;
        assert_eq!(titles(&rows), vec!["target".to_string()]);
        assert!(
            probe.barrier_pins() - before >= 2,
            "a refused filter group must retry the plain native route (one \
             pin per widening round); a flat-scan surrender pins nothing"
        );
    }

    #[tokio::test]
    async fn hybrid_filtered_bm25_stale_ft_falls_back_to_fresh_corpus() {
        let _cap = CapEnvGuard::set(64);
        let docs = vec![
            ("old-a", "other", "alpha alpha"),
            ("old-b", "other", "alpha beta"),
        ];
        let (mut writer, _, probe) = indexed_corpus("hybrid-ft-stale", &docs).await;
        assert_eq!(authoritative_hits(&writer, "alpha", 8).await.len(), 2);

        let mut properties = BTreeMap::new();
        properties.insert("title".into(), CoreValue::Str("fresh".into()));
        properties.insert("tenant".into(), CoreValue::Str("acme".into()));
        properties.insert("body".into(), CoreValue::Str("zebra".into()));
        writer
            .upsert_node(
                "Doc",
                NodeId::new(),
                &NodeWriteRecord {
                    properties,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        writer.commit_batch().await.unwrap();
        assert!(
            writer
                .snapshot()
                .text_search(INDEX_NAME, "Doc", &parse_query("zebra"), Some(8))
                .await
                .unwrap()
                .is_none(),
            "a same-label delta must make the persisted .ft non-authoritative"
        );

        let before = probe.text_gets();
        let rows = run(
            &writer,
            "CALL search.hybrid({ label: 'Doc', query_text: 'zebra', \
             text_property: 'body', k: 1, k_sparse: 1, \
             filter: { tenant: 'acme' } }) \
             YIELD node, score RETURN node.title AS title",
            vec![],
        )
        .await;
        assert_eq!(titles(&rows), vec!["fresh".to_string()]);
        assert_eq!(
            probe.text_gets() - before,
            0,
            "the freshness gate must reject stale .ft before fetching its body"
        );
    }

    #[tokio::test]
    async fn hybrid_filtered_bm25_corrupt_or_missing_ft_uses_exact_fallback() {
        let _cap = CapEnvGuard::set(64);
        let mut docs = other_docs(12);
        docs.push((
            "target".into(),
            "acme".into(),
            "alpha w1 w2 w3 w4 w5 w6 w7 w8".into(),
        ));
        let borrowed = borrowed_docs(&docs);
        let (writer, _, probe) = indexed_corpus("hybrid-ft-missing", &borrowed).await;
        assert_eq!(
            authoritative_hits(&writer, "alpha", 32).await.len(),
            docs.len(),
            "sanity: the authoritative .ft initially serves the whole corpus"
        );

        let relative = writer
            .snapshot()
            .manifest()
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == namidb_storage::manifest::SstKind::TextIndex)
            .expect("compaction builds .ft")
            .path
            .clone();
        let absolute = Path::from(format!(
            "{}/{}",
            paths("hybrid-ft-missing").namespace_prefix().as_ref(),
            relative
        ));
        let cypher = "CALL search.hybrid({ label: 'Doc', query_text: 'alpha', \
             text_property: 'body', k: 1, k_sparse: 1, \
             filter: { tenant: 'acme' } }) \
             YIELD node, score RETURN node.title AS title";

        probe
            .inner
            .put(
                &absolute,
                object_store::PutPayload::from_static(b"NAMIFT02corrupt"),
            )
            .await
            .unwrap();
        assert_eq!(
            titles(&run(&writer, cypher, vec![]).await),
            vec!["target".to_string()],
            "an undecodable .ft must flat-score the filtered corpus"
        );

        probe.inner.delete(&absolute).await.unwrap();
        assert_eq!(
            titles(&run(&writer, cypher, vec![]).await),
            vec!["target".to_string()],
            "a swept .ft must use the same exact filtered fallback"
        );
    }
}
