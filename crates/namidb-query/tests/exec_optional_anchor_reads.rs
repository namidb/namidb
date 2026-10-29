//! Field report: `MATCH (i:Identifier) WHERE id(i) = $id
//! OPTIONAL MATCH (i)-[r:ON_DEVICE]-(d:Device) RETURN ...` grew to 13 GB on
//! a 2.4M node graph whose node SSTs overlap. The query only needs the
//! anchor, its ON_DEVICE adjacency in both directions and the labels of
//! the few neighbours it finds. This pins that envelope with a byte count
//! per stored object and the scan counters of the SST cache, on a cold
//! reader over two overlapping node SSTs, in both executors.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};

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

/// Records the byte range every GET returns, per object.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn ObjectStore>,
    read: Mutex<BTreeMap<String, Vec<Range<u64>>>>,
}

/// Bytes of an object covered by `ranges`, each byte counted once.
fn covered(ranges: &[Range<u64>]) -> u64 {
    let mut sorted = ranges.to_vec();
    sorted.sort_by_key(|r| r.start);
    let (mut total, mut end) = (0, 0);
    for r in sorted {
        let start = r.start.max(end);
        if r.end > start {
            total += r.end - start;
            end = r.end;
        }
    }
    total
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
        let head = options.head;
        let result = self.inner.get_opts(location, options).await?;
        if !head {
            self.read
                .lock()
                .unwrap()
                .entry(location.to_string())
                .or_default()
                .push(result.range.clone());
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

fn node(name: String, filler: String) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("name".into(), CoreValue::Str(name));
    properties.insert("emb".into(), CoreValue::Str(filler));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn edge(note: String) -> EdgeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("note".into(), CoreValue::Str(note));
    EdgeWriteRecord {
        properties,
        schema_version: 1,
    }
}

const ARTIFACTS: u64 = 16000;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anchored_undirected_optional_hop_reads_only_the_anchor_neighbourhood() {
    // Raw reads, not the 256 KiB pages the shared range cache rounds to.
    std::env::set_var("NAMIDB_RAM_PAGE_CACHE_MAX_BYTES", "0");
    let counting = Arc::new(CountingStore {
        inner: Arc::new(InMemory::new()),
        read: Mutex::new(BTreeMap::new()),
    });
    let store: Arc<dyn ObjectStore> = counting.clone();
    let paths = NamespacePaths::new("tenants", NamespaceId::new("optional-anchor").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    let devices: Vec<NodeId> = (0..3).map(|d| random_id(1_000_000 + d)).collect();
    let linked = random_id(2_000_000);
    let lonely = random_id(2_000_001);
    // Two flushes over random ids: two node SSTs whose key ranges overlap.
    for half in 0..2u64 {
        if half == 0 {
            for (d, id) in devices.iter().enumerate() {
                writer
                    .upsert_node("Device", *id, &node(format!("dev{d}"), blob(d as u64, 4)))
                    .unwrap();
            }
            writer
                .upsert_node("Identifier", linked, &node("linked".into(), blob(7, 4)))
                .unwrap();
        } else {
            writer
                .upsert_node("Identifier", lonely, &node("lonely".into(), blob(8, 4)))
                .unwrap();
            // The anchor's own ON_DEVICE edges, one each way.
            writer
                .upsert_edge("ON_DEVICE", linked, devices[0], &edge("out".into()))
                .unwrap();
            writer
                .upsert_edge("ON_DEVICE", devices[1], linked, &edge("in".into()))
                .unwrap();
        }
        for a in (half..ARTIFACTS).step_by(2) {
            let id = random_id(a);
            writer
                .upsert_node("Artifact", id, &node(format!("a{a}"), blob(a, 32)))
                .unwrap();
            writer
                .upsert_edge(
                    "ON_DEVICE",
                    id,
                    devices[(a % 3) as usize],
                    &edge(blob(a + 99, 12)),
                )
                .unwrap();
        }
        writer.commit_batch().await.unwrap();
        let schema = writer.snapshot().manifest().manifest.schema.clone();
        writer.flush(schema).await.unwrap();
    }
    drop(writer);

    let manifest = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone())
        .load_current()
        .await
        .unwrap();
    let node_ssts = manifest
        .manifest
        .ssts
        .iter()
        .filter(|desc| format!("{:?}", desc.kind) == "Nodes")
        .count();
    assert_eq!(node_ssts, 2, "the fixture needs two node SSTs");
    // Every SST body and sidecar but the bloom filters, which are read
    // whole by design and are a few KiB.
    let sizes: BTreeMap<String, u64> = {
        use futures::TryStreamExt;
        let prefix = Path::from(format!("{}/sst", paths.namespace_prefix().as_ref()));
        store
            .list(Some(&prefix))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .into_iter()
            .filter(|meta| !meta.location.as_ref().ends_with(".bloom"))
            .map(|meta| (meta.location.to_string(), meta.size))
            .collect()
    };
    let catalog = StatsCatalog::from_manifest(&manifest.manifest);
    let memtable = Memtable::new();
    let view = memtable.snapshot_view();

    // The first two read `r` as a value, so the hop loads edge properties;
    // the others only need the topology.
    let shapes = [
        "(i)-[r:ON_DEVICE]-(d:Device) RETURN r.note AS note, id(d) AS d, d.name AS name",
        "(i)-[r:ON_DEVICE]-(d:Device) RETURN i, r, d",
        "(i)-[:ON_DEVICE]-(d:Device) RETURN id(d) AS d",
        "(i)-[r:ON_DEVICE]-(d:Device) RETURN id(d) AS d, d.name AS name",
    ];
    for (anchor, expected) in [
        (
            linked,
            vec![format!("in|{}", devices[1]), format!("out|{}", devices[0])],
        ),
        (lonely, vec!["NULL".to_string()]),
    ] {
        for shape in shapes {
            for factor in [false, true] {
                let query =
                    format!("MATCH (i:Identifier) WHERE id(i) = '{anchor}' OPTIONAL MATCH {shape}");
                let plan = optimize(lower(&parse(&query).unwrap()).unwrap(), &catalog);
                // A cold reader each time, so nothing an earlier query cached
                // helps this one.
                let cache = SstCache::new(64 * 1024 * 1024);
                let snapshot = Snapshot::new(manifest.clone(), &view, store.clone(), paths.clone())
                    .with_cache(cache.clone());
                counting.read.lock().unwrap().clear();
                let rows = if factor {
                    execute_factor_path(&plan, &snapshot, &Params::new()).await
                } else {
                    execute_flat_path(&plan, &snapshot, &Params::new()).await
                }
                .unwrap();
                let mut got: Vec<String> = rows
                    .iter()
                    .map(|row| match (row.get("r"), row.get("note"), row.get("d")) {
                        (Some(RuntimeValue::Rel(r)), _, Some(RuntimeValue::Node(d))) => {
                            match &r.properties["note"] {
                                RuntimeValue::String(note) => format!("{note}|{}", d.id),
                                other => panic!("note was {other:?}"),
                            }
                        }
                        (_, Some(RuntimeValue::String(note)), Some(RuntimeValue::String(d))) => {
                            format!("{note}|{d}")
                        }
                        (None, None, Some(RuntimeValue::String(d))) => d.clone(),
                        (Some(RuntimeValue::Null), _, Some(RuntimeValue::Null))
                        | (_, Some(RuntimeValue::Null), Some(RuntimeValue::Null))
                        | (None, None, Some(RuntimeValue::Null)) => "NULL".to_string(),
                        other => panic!("unexpected row {other:?}"),
                    })
                    .collect();
                got.sort();
                let what = format!("{query} (factorized: {factor})");
                let expected: Vec<String> = if shape.contains("note") || shape.contains(" r,") {
                    expected.clone()
                } else {
                    let mut ids: Vec<String> = expected
                        .iter()
                        .map(|e| e.rsplit('|').next().unwrap().to_string())
                        .collect();
                    ids.sort();
                    ids
                };
                assert_eq!(got, expected, "{what}");

                // No label or node scan ran, and no edge property stream was
                // decoded whole.
                assert_eq!(cache.label_selected_node_rows(), 0, "{what}");
                assert_eq!(cache.sparse_node_filter_scans(), 0, "{what}");
                assert_eq!(cache.decoded_node_row_group_misses(), 0, "{what}");
                assert_eq!(cache.limited_node_scan_examined_rows(), 0, "{what}");
                assert_eq!(cache.edge_streams_misses(), 0, "{what}");
                // And no SST file was read in full: the anchor, its adjacency
                // and its neighbours' labels are a slice of every file. Node
                // files get a fixed budget of a few pages. Edge files are read
                // at a coarser grain (an edge's properties come in pages of up
                // to 1024 rows), so at this size they keep a relative bound.
                // With NAMIDB_ADJACENCY set, a topology-only hop builds the
                // CSR of the whole type instead; that mode is not covered here.
                let read = counting.read.lock().unwrap().clone();
                for (path, ranges) in &read {
                    if let Some(size) = sizes.get(path) {
                        let bytes = covered(ranges);
                        let budget = if path.contains("-edges-") {
                            size / 2
                        } else {
                            64 * 1024
                        };
                        assert!(
                            bytes < budget,
                            "{what}: read {bytes} of the {size} bytes of {path}"
                        );
                    }
                }
            }
        }
    }
}

/// An OPTIONAL MATCH with a WHERE runs its pattern once per outer row. The
/// WHERE used to stay above the whole subplan, so an `id()` pin on the
/// pattern's new start scanned that label for every outer row, and a plain
/// property test on it never reached the scan.
#[tokio::test]
async fn optional_match_where_reaches_the_scan_of_its_new_start() {
    let cache = SstCache::new(64 * 1024 * 1024);
    let mut caches = namidb_storage::SessionCaches::none();
    caches.sst_cache = Some(cache.clone());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("optional-where").unwrap());
    let mut writer = WriterSession::open_with_caches(Arc::new(InMemory::new()), paths, caches)
        .await
        .unwrap();
    let devices: Vec<NodeId> = (0..3).map(|d| random_id(1_000 + d)).collect();
    for (d, id) in devices.iter().enumerate() {
        writer
            .upsert_node("Device", *id, &node(format!("dev{d}"), String::new()))
            .unwrap();
    }
    let idents: Vec<NodeId> = (0..300).map(random_id).collect();
    for (n, id) in idents.iter().enumerate() {
        let kind = if n % 100 == 7 { "phone" } else { "mail" };
        writer
            .upsert_node("Identifier", *id, &node(kind.into(), String::new()))
            .unwrap();
        writer
            .upsert_edge("CONTACTED", *id, devices[n % 2], &edge(String::new()))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema).await.unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);

    let pinned = idents[4];
    // The pinned identifier contacted one device and the phones (7, 107,
    // 207) all contacted device 1; every other device is padded once.
    for (filter, total) in [
        (format!("id(i) = '{pinned}'"), 3usize),
        ("i.name = 'phone'".to_string(), 5),
    ] {
        let query = format!(
            "MATCH (d:Device) OPTIONAL MATCH (i:Identifier)-[:CONTACTED]->(d) \
             WHERE {filter} RETURN id(d) AS d, id(i) AS i"
        );
        let lowered = lower(&parse(&query).unwrap()).unwrap();
        let plan = optimize(lowered.clone(), &catalog);
        let explain = namidb_query::explain(&plan);
        let ids = |rows: Vec<namidb_query::Row>| {
            let mut out: Vec<String> = rows
                .iter()
                .map(|r| format!("{:?}|{:?}", r.get("d"), r.get("i")))
                .collect();
            out.sort();
            out
        };
        let expected = ids(execute_flat_path(&lowered, &snapshot, &Params::new())
            .await
            .unwrap());
        assert_eq!(expected.len(), total, "{query}");
        for factor in [false, true] {
            let before = cache.label_selected_node_rows();
            let rows = if factor {
                execute_factor_path(&plan, &snapshot, &Params::new()).await
            } else {
                execute_flat_path(&plan, &snapshot, &Params::new()).await
            }
            .unwrap();
            assert_eq!(ids(rows), expected, "{query} (factorized: {factor})");
            if filter.starts_with("id(") {
                // Only the three Device rows are scanned; the pin is a seek.
                assert_eq!(cache.label_selected_node_rows() - before, 3, "{explain}");
            }
        }
        let head_scan = if filter.starts_with("id(") {
            "NodeById label=Identifier".to_string()
        } else {
            "NodeScan label=Identifier alias=i projection=[id, name] predicates=[i.name = \"phone\"]"
                .to_string()
        };
        assert!(explain.contains(&head_scan), "{explain}");
    }
}
