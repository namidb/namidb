//! A loader's `UNWIND ... MERGE (a:L {id: r.id}) SET a += r WITH a, r
//! MATCH (n:P {id: r.pid}) MERGE (n)-[:R]->(a)` over a label whose key has a
//! RANGE index but no UNIQUE constraint.
//!
//! Before 3.3.3 the non-unique equality path filled the writer's
//! transactional postings by scanning the whole label (every column, so all
//! of a large text corpus) before it could answer the first row, and then
//! hydrated every matched node with its own targeted SST walk. On a remote
//! object store a 300-row batch of existing ids took longer than the 30 s
//! write timeout, and the abandoned map was rebuilt by every retry.
//!
//! These tests pin the batched shape: no label scan populates the map, the
//! statement reads a small fraction of the stored corpus, and the answers
//! stay exact across duplicates in the batch, staged changes and rollback.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures::stream::BoxStream;
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::cost::StatsCatalog;
use namidb_query::{
    execute_write, execute_write_staged, lower, optimize, parse, LogicalPlan, Params, RuntimeValue,
};
use namidb_storage::{
    EdgeWriteRecord, NamespacePaths, NodeWriteRecord, SessionCaches, SstCache, WriterSession,
};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

const LOADER: &str = "UNWIND $rows AS r \
     MERGE (a:Article {id: r.id}) SET a += r \
     WITH a, r \
     MATCH (n:Act {id: r.act_id}) \
     MERGE (n)-[c:HAS_ARTICLE]->(a) SET c.position = r.position";

const ARTICLES: usize = 4_000;
const ACTS: usize = 200;
const TEXT_BYTES: usize = 2_000;

/// Counts GET requests and the bytes they return; every GET also waits a
/// little, like a remote object store does.
#[derive(Debug)]
struct MeteredStore {
    inner: Arc<dyn ObjectStore>,
    gets: AtomicU64,
    bytes: AtomicU64,
    latency_ms: AtomicU64,
}

impl fmt::Display for MeteredStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeteredStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for MeteredStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let latency = self.latency_ms.load(Ordering::Relaxed);
        if latency > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(latency)).await;
        }
        let head = options.head;
        let result = self.inner.get_opts(location, options).await?;
        if !head {
            self.gets.fetch_add(1, Ordering::Relaxed);
            self.bytes
                .fetch_add(result.range.end - result.range.start, Ordering::Relaxed);
        }
        Ok(result)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn schema() -> namidb_core::Schema {
    let indexed = |name: &str| {
        PropertyDef::new(name, DataType::Utf8, true)
            .unwrap()
            .with_indexed(true)
    };
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Article".into(),
            properties: vec![indexed("id"), indexed("act_id")],
        })
        .unwrap()
        .label(LabelDef {
            name: "Act".into(),
            properties: vec![indexed("id")],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "HAS_ARTICLE".into(),
            src_label: "Act".into(),
            dst_label: "Article".into(),
            properties: vec![PropertyDef::new("position", DataType::Int64, true).unwrap()],
        })
        .unwrap()
        .build()
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Incompressible article text, so the stored corpus is as large as it looks.
fn text(n: u64) -> String {
    let mut out = String::with_capacity(TEXT_BYTES);
    let mut w = 0;
    while out.len() < TEXT_BYTES {
        out.push_str(&format!("{:016x} ", splitmix(n << 20 | w)));
        w += 1;
    }
    out.truncate(TEXT_BYTES);
    out
}

fn article_id(i: usize) -> String {
    format!("art-{i:07}")
}

fn act_id(i: usize) -> String {
    format!("act-{i:06}")
}

fn act_of(i: usize) -> usize {
    i % ACTS
}

/// Store an legal-corpus-shaped corpus (articles with their HAS_ARTICLE edge) in two
/// flushed SSTs, then return a cold writer over it.
async fn seeded(name: &str) -> (WriterSession, Arc<MeteredStore>, SstCache) {
    let store = Arc::new(MeteredStore {
        inner: Arc::new(InMemory::new()),
        gets: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        latency_ms: AtomicU64::new(0),
    });
    let mut writer = WriterSession::open(store.clone(), paths(name))
        .await
        .unwrap();
    let mut act_ids = Vec::with_capacity(ACTS);
    for i in 0..ACTS {
        let id = NodeId::new();
        act_ids.push(id);
        writer
            .upsert_node(
                "Act",
                id,
                &NodeWriteRecord {
                    properties: BTreeMap::from([("id".into(), CoreValue::Str(act_id(i)))]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for i in 0..ARTICLES {
        let id = NodeId::new();
        let text = text(i as u64);
        writer
            .upsert_node(
                "Article",
                id,
                &NodeWriteRecord {
                    properties: BTreeMap::from([
                        ("id".into(), CoreValue::Str(article_id(i))),
                        ("act_id".into(), CoreValue::Str(act_id(act_of(i)))),
                        ("position".into(), CoreValue::I64(i as i64)),
                        ("body_fold".into(), CoreValue::Str(text.to_lowercase())),
                        ("body".into(), CoreValue::Str(text)),
                    ]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        writer
            .upsert_edge(
                "HAS_ARTICLE",
                act_ids[act_of(i)],
                id,
                &EdgeWriteRecord {
                    properties: BTreeMap::from([("position".into(), CoreValue::I64(i as i64))]),
                    schema_version: 1,
                },
            )
            .unwrap();
        if i == ARTICLES / 2 {
            writer.commit_batch().await.unwrap();
            writer.flush(schema()).await.unwrap();
        }
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    drop(writer);

    let cache = SstCache::new(256 * 1024 * 1024);
    let writer = WriterSession::open_with_caches(
        store.clone(),
        paths(name),
        SessionCaches {
            sst_cache: Some(cache.clone()),
            node_cache: None,
            adjacency_cache: None,
        },
    )
    .await
    .unwrap();
    (writer, store, cache)
}

fn row(id: &str, act: &str, position: i64, body: &str) -> RuntimeValue {
    RuntimeValue::Map(BTreeMap::from([
        ("id".into(), RuntimeValue::String(id.into())),
        ("act_id".into(), RuntimeValue::String(act.into())),
        ("position".into(), RuntimeValue::Integer(position)),
        ("estado".into(), RuntimeValue::String("vigente".into())),
        ("body".into(), RuntimeValue::String(body.into())),
        (
            "body_fold".into(),
            RuntimeValue::String(body.to_lowercase()),
        ),
    ]))
}

fn rows_param(rows: Vec<RuntimeValue>) -> Params {
    Params::from([("rows".into(), RuntimeValue::List(rows))])
}

fn loader_plan(writer: &WriterSession) -> LogicalPlan {
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    optimize(lower(&parse(LOADER).unwrap()).unwrap(), &catalog)
}

async fn count(writer: &WriterSession, query: &str, params: &Params) -> i64 {
    let plan = lower(&parse(query).unwrap()).unwrap();
    let snapshot = writer.snapshot();
    let rows = namidb_query::execute(&plan, &snapshot, params)
        .await
        .unwrap();
    match rows[0].get("c") {
        Some(RuntimeValue::Integer(c)) => *c,
        other => panic!("expected a count, got {other:?}"),
    }
}

/// Ids spread over the whole key space, like a loader resuming over rows it
/// has already sent.
fn existing_indices(n: usize) -> Vec<usize> {
    (0..n).map(|k| (k * 7919 + 13) % ARTICLES).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loader_merge_on_non_unique_key_seeds_postings_and_reads_a_fraction_of_the_corpus() {
    let (mut writer, store, _cache) = seeded("merge-nonunique-batch").await;
    let stored_bytes: u64 = {
        use futures::TryStreamExt;
        let objects: Vec<ObjectMeta> = store.inner.list(None).try_collect().await.unwrap();
        objects.iter().map(|o| o.size).sum()
    };
    let plan = loader_plan(&writer);

    // 150 existing ids (with a changed text) and 150 new ones, as one
    // 300-row batch. Two new rows repeat an id: the second must match the
    // node the first one created.
    let mut rows = Vec::new();
    for (k, i) in existing_indices(150).into_iter().enumerate() {
        rows.push(row(
            &article_id(i),
            &act_id(act_of(i)),
            i as i64,
            &format!("{} v2-{k}", text(i as u64)),
        ));
    }
    for k in 0..148 {
        let i = ARTICLES + k;
        rows.push(row(
            &article_id(i),
            &act_id(k % 3),
            i as i64,
            &text(i as u64),
        ));
    }
    rows.push(row(&article_id(ARTICLES), &act_id(0), 1, "dup a"));
    rows.push(row(&article_id(ARTICLES + 1), &act_id(1), 2, "dup b"));
    let params = rows_param(rows);

    store.latency_ms.store(2, Ordering::Relaxed);
    let scans = writer.unique_index().populate_scans();
    let (gets, bytes) = (
        store.gets.load(Ordering::Relaxed),
        store.bytes.load(Ordering::Relaxed),
    );
    let started = Instant::now();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    let elapsed = started.elapsed();
    let gets = store.gets.load(Ordering::Relaxed) - gets;
    let bytes = store.bytes.load(Ordering::Relaxed) - bytes;
    store.latency_ms.store(0, Ordering::Relaxed);
    eprintln!(
        "loader batch: {elapsed:?}, {gets} GETs, {bytes} bytes read of {stored_bytes} stored, \
         populate scans {}",
        writer.unique_index().populate_scans() - scans
    );

    assert_eq!(
        outcome.nodes_created, 148,
        "only the distinct new ids create"
    );
    assert_eq!(
        outcome.edges_created, 148,
        "existing articles keep their HAS_ARTICLE edge"
    );
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans,
        "MERGE and MATCH on a non-unique RANGE key must seed point postings, not scan the label"
    );
    assert!(
        bytes * 4 < stored_bytes,
        "a 300-row batch must not read the stored corpus: read {bytes} of {stored_bytes} bytes"
    );
    assert!(
        gets < 150,
        "matched rows must hydrate in batches, not one SST walk per row: {gets} GETs"
    );

    // Exact results: no duplicate articles, the texts were replaced, every
    // article hangs from exactly one Act.
    let total = count(
        &writer,
        "MATCH (a:Article) RETURN count(a) AS c",
        &Params::new(),
    )
    .await;
    assert_eq!(total, (ARTICLES + 148) as i64);
    let i = existing_indices(1)[0];
    let updated = count(
        &writer,
        "MATCH (a:Article {id: $id}) WHERE a.body ENDS WITH ' v2-0' RETURN count(a) AS c",
        &Params::from([("id".into(), RuntimeValue::String(article_id(i)))]),
    )
    .await;
    assert_eq!(updated, 1);
    let dup = count(
        &writer,
        "MATCH (:Act)-[:HAS_ARTICLE]->(a:Article {id: $id}) RETURN count(a) AS c",
        &Params::from([("id".into(), RuntimeValue::String(article_id(ARTICLES)))]),
    )
    .await;
    assert_eq!(dup, 1, "a repeated new id creates one node and one edge");

    // Replaying the batch (a loader retry) creates nothing and still does
    // not scan.
    let replay = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(replay.nodes_created, 0);
    assert_eq!(replay.edges_created, 0);
    assert_eq!(writer.unique_index().populate_scans(), scans);
}

#[tokio::test]
async fn seeded_postings_follow_staged_changes_and_rollback() {
    let (mut writer, _store, _cache) = seeded("merge-nonunique-staged").await;
    // Every MERGE carries 20 existing ids besides the one under test, so the
    // batch seeds point postings (a lone value would populate the whole map).
    let fillers: Vec<String> = existing_indices(21)
        .into_iter()
        .filter(|i| *i != 7)
        .take(20)
        .map(article_id)
        .collect();
    let merge = |id: &str| {
        let mut ids: Vec<RuntimeValue> = fillers
            .iter()
            .map(|f| RuntimeValue::String(f.clone()))
            .collect();
        ids.push(RuntimeValue::String(id.into()));
        (
            lower(
                &parse("UNWIND $ids AS id MERGE (a:Article {id: id}) RETURN a.id AS id").unwrap(),
            )
            .unwrap(),
            Params::from([("ids".into(), RuntimeValue::List(ids))]),
        )
    };

    // Staged rename of an existing article: its old id is now free and its
    // new id taken, inside the same transaction.
    let rename = lower(&parse("MATCH (a:Article {id: $old}) SET a.id = $new").unwrap()).unwrap();
    execute_write_staged(
        &rename,
        &mut writer,
        &Params::from([
            ("old".into(), RuntimeValue::String(article_id(7))),
            ("new".into(), RuntimeValue::String("renamed".into())),
        ]),
    )
    .await
    .unwrap();
    // The one-key MATCH above may populate the map once; the batches below
    // must not scan again.
    let scans = writer.unique_index().populate_scans();
    let (plan, params) = merge("renamed");
    let out = execute_write_staged(&plan, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(out.nodes_created, 0, "the staged new id must match");
    let (plan, params) = merge(&article_id(7));
    let out = execute_write_staged(&plan, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(out.nodes_created, 1, "the staged old id is free");
    let (plan, params) = merge("brand-new");
    let out = execute_write_staged(&plan, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(out.nodes_created, 1);
    let out = execute_write_staged(&plan, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(
        out.nodes_created, 0,
        "a staged create is visible to the next MERGE"
    );
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans,
        "batched MERGEs inside the transaction seed point postings"
    );

    // Roll everything back: none of the staged answers may survive.
    writer.discard_batch();
    let (plan, params) = merge("renamed");
    let out = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(
        out.nodes_created, 1,
        "the rolled-back rename no longer holds the id"
    );
    let (plan, params) = merge(&article_id(7));
    let out = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(out.nodes_created, 0, "the rolled-back rename frees nothing");
    let (plan, params) = merge("brand-new");
    let out = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(
        out.nodes_created, 1,
        "the rolled-back create left no phantom"
    );

    let total = count(
        &writer,
        "MATCH (a:Article) RETURN count(a) AS c",
        &Params::new(),
    )
    .await;
    assert_eq!(total, (ARTICLES + 2) as i64);
}

#[tokio::test]
async fn merge_on_non_unique_key_matches_every_existing_holder() {
    let (mut writer, _store, _cache) = seeded("merge-nonunique-dups").await;
    // A second committed node with an id that already exists: without a
    // UNIQUE constraint both are legal and MERGE matches both.
    let create = lower(&parse("CREATE (:Article {id: $id, copy: true})").unwrap()).unwrap();
    execute_write(
        &create,
        &mut writer,
        &Params::from([("id".into(), RuntimeValue::String(article_id(42)))]),
    )
    .await
    .unwrap();
    let plan =
        lower(&parse("UNWIND $ids AS id MERGE (a:Article {id: id}) RETURN a.id AS id").unwrap())
            .unwrap();
    let params = Params::from([(
        "ids".into(),
        RuntimeValue::List(vec![
            RuntimeValue::String(article_id(42)),
            RuntimeValue::String(article_id(43)),
        ]),
    )]);
    let out = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(out.nodes_created, 0);
    assert_eq!(out.rows.len(), 3, "two holders of art-42 plus art-43");
}
