//! Endpoint hydration of a low-degree hop over many seeds is batched across
//! the seeds, so the dependent store round trips do not grow with the seed
//! count.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::stream::BoxStream;
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    execute_factor_path, execute_flat_path, lower, optimize, parse, Params, RuntimeValue,
    StatsCatalog,
};
use namidb_storage::memtable::Memtable;
use namidb_storage::read::Snapshot;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, SstCache, WriterSession};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

/// Delays every GET by `delay_ms`, like an object store's first-byte
/// latency, so dependent reads add up in wall time.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn ObjectStore>,
    delay_ms: AtomicU64,
}

impl fmt::Display for CountingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for CountingStore {
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
        let delay = self.delay_ms.load(Ordering::Relaxed);
        if delay > 0 {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        self.inner.get_opts(location, options).await
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

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Random ids, like the uuid5 ids of the report, so every node SST spans
/// the whole key range and the two flushes overlap.
fn random_id(n: u64) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&splitmix(n).to_be_bytes());
    bytes[8..].copy_from_slice(&splitmix(n ^ 0xABCD).to_be_bytes());
    NodeId::from_uuid(uuid::Uuid::from_bytes(bytes))
}

/// Incompressible filler standing in for the embeddings.
fn blob(n: u64, words: u64) -> String {
    (0..words)
        .map(|w| format!("{:016x}", splitmix(n << 16 | w)))
        .collect()
}

fn node(name: String, n: i64, filler: String) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("name".into(), CoreValue::Str(name));
    properties.insert("n".into(), CoreValue::I64(n));
    properties.insert("emb".into(), CoreValue::Str(filler));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

const PARENTS: u64 = 4000;

/// A degree-1 hop from many seeds, `MATCH (c:Child) ... MATCH
/// (p:Parent)-[:OF]->(c) RETURN p.name`: every seed has its own parent to
/// hydrate. The endpoints of a whole seed window are read in one batch, so
/// 200 seeds cost about what 2 do (they cost a few dependent store round
/// trips per seed), and the rows are the same.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn low_degree_hop_hydrates_the_endpoints_of_all_seeds_together() {
    // Raw reads, not the 256 KiB pages the shared range cache rounds to.
    std::env::set_var("NAMIDB_RAM_PAGE_CACHE_MAX_BYTES", "0");
    let counting = Arc::new(CountingStore {
        inner: Arc::new(InMemory::new()),
        delay_ms: AtomicU64::new(0),
    });
    let store: Arc<dyn ObjectStore> = counting.clone();
    let paths = NamespacePaths::new("tenants", NamespaceId::new("seed-prewarm").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    for p in 0..PARENTS {
        let parent = random_id(p);
        writer
            .upsert_node(
                "Parent",
                parent,
                &node(format!("p{p}"), p as i64, blob(p, 16)),
            )
            .unwrap();
        let child = random_id(1_000_000 + p);
        writer
            .upsert_node(
                "Child",
                child,
                &node(format!("c{p}"), p as i64, String::new()),
            )
            .unwrap();
        writer
            .upsert_edge(
                "OF",
                parent,
                child,
                &EdgeWriteRecord {
                    properties: BTreeMap::new(),
                    schema_version: 1,
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema).await.unwrap();
    drop(writer);

    let manifest = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone())
        .load_current()
        .await
        .unwrap();
    let catalog = StatsCatalog::from_manifest(&manifest.manifest);
    let memtable = Memtable::new();
    let view = memtable.snapshot_view();
    const DELAY_MS: u64 = 10;
    counting.delay_ms.store(DELAY_MS, Ordering::Relaxed);
    let mut reads = Vec::new();
    for seeds in [2u64, 200] {
        for factor in [false, true] {
            let query = format!(
                "MATCH (c:Child) WHERE c.n < {seeds} \
                 MATCH (p:Parent)-[:OF]->(c) RETURN c.name AS c, p.name AS p"
            );
            let plan = optimize(lower(&parse(&query).unwrap()).unwrap(), &catalog);
            // A cold reader each time, so nothing an earlier query cached
            // helps this one.
            let cache = SstCache::new(64 * 1024 * 1024);
            let snapshot = Snapshot::new(manifest.clone(), &view, store.clone(), paths.clone())
                .with_cache(cache);
            let started = Instant::now();
            let rows = if factor {
                execute_factor_path(&plan, &snapshot, &Params::new()).await
            } else {
                execute_flat_path(&plan, &snapshot, &Params::new()).await
            }
            .unwrap();
            let mut got: Vec<(String, String)> = rows
                .iter()
                .map(|row| match (row.get("c"), row.get("p")) {
                    (Some(RuntimeValue::String(c)), Some(RuntimeValue::String(p))) => {
                        (c.clone(), p.clone())
                    }
                    other => panic!("unexpected row {other:?}"),
                })
                .collect();
            got.sort();
            let mut expected: Vec<(String, String)> = (0..seeds)
                .map(|n| (format!("c{n}"), format!("p{n}")))
                .collect();
            expected.sort();
            assert_eq!(got, expected, "{query} (factorized: {factor})");
            reads.push((seeds, factor, started.elapsed()));
        }
    }
    let of = |seeds: u64, factor: bool| {
        reads
            .iter()
            .find(|(s, f, _)| *s == seeds && *f == factor)
            .unwrap()
            .2
    };
    // One dependent read per seed alone would add 198 * DELAY_MS. The
    // factorized executor (opt-in) is held to the same rows only.
    assert!(
        of(200, false) < of(2, false) + Duration::from_millis(100 * DELAY_MS),
        "the hop's wall time grew with the seeds: {reads:?}"
    );
}
