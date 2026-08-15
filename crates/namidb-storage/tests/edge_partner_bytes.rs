//! Plan item 14 (docs/testing/25tb-readiness.md): the partner-list route
//! (`edge_lookup_via_sst`, the production default) must hydrate one key's
//! edges — partners, LSNs, tombstones AND properties — through ranged reads,
//! never a whole-body GET. At 25 TB an O(edge-SST) read per hop is
//! disqualifying; this pins the byte envelope of a cold lookup against a
//! multi-megabyte edge SST.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::stream::BoxStream;
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::memtable::Memtable;
use namidb_storage::read::Snapshot;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

/// Counts every byte a GET-family call returns. Puts/lists are delegated
/// untouched.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn ObjectStore>,
    read_bytes: AtomicU64,
}

impl CountingStore {
    fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self {
            inner,
            read_bytes: AtomicU64::new(0),
        }
    }

    fn read_bytes(&self) -> u64 {
        self.read_bytes.load(Ordering::SeqCst)
    }
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
        let span = if head {
            0
        } else {
            result.range.end.saturating_sub(result.range.start)
        };
        self.read_bytes.fetch_add(span, Ordering::SeqCst);
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

const KEYS: u64 = 2048;
const DEGREE: u64 = 64;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![PropertyDef::new("title", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "CITES".into(),
            src_label: "Doc".into(),
            dst_label: "Doc".into(),
            properties: vec![PropertyDef::new("note", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .build()
}

fn doc_id(ordinal: u64) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[0] = 0x3E;
    bytes[8..].copy_from_slice(&ordinal.to_be_bytes());
    NodeId::from_uuid(uuid::Uuid::from_bytes(bytes))
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Incompressible-ish payload so zstd cannot shrink the fixture back into a
/// toy: 12 pseudo-random u64s rendered as hex per edge (~200 bytes).
fn note(src: u64, slot: u64) -> String {
    let mut text = format!("note-{src}-{slot}-");
    for word in 0..12u64 {
        text.push_str(&format!("{:016x}", splitmix(src << 32 | slot << 8 | word)));
    }
    text
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_partner_lookup_reads_a_sliver_of_a_multi_mb_edge_sst() {
    let counting = Arc::new(CountingStore::new(Arc::new(InMemory::new())));
    let store: Arc<dyn ObjectStore> = counting.clone();
    let paths = NamespacePaths::new("tenants", NamespaceId::new("edge-partner-bytes").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();

    for ordinal in 0..KEYS {
        let mut properties = BTreeMap::new();
        properties.insert("title".into(), CoreValue::Str(format!("doc {ordinal}")));
        writer
            .upsert_node(
                "Doc",
                doc_id(ordinal),
                &NodeWriteRecord {
                    properties,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for src in 0..KEYS {
        for slot in 0..DEGREE {
            let dst = (src + 1 + slot) % KEYS;
            let mut properties = BTreeMap::new();
            properties.insert("note".into(), CoreValue::Str(note(src, slot)));
            writer
                .upsert_edge(
                    "CITES",
                    doc_id(src),
                    doc_id(dst),
                    &EdgeWriteRecord {
                        properties,
                        schema_version: 1,
                    },
                )
                .unwrap();
        }
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    let manifest_store = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone());
    let loaded = manifest_store.load_current().await.unwrap();
    let fwd_bytes: u64 = loaded
        .manifest
        .ssts
        .iter()
        .filter(|descriptor| {
            format!("{:?}", descriptor.kind) == "EdgesFwd" && descriptor.scope == "CITES"
        })
        .map(|descriptor| descriptor.size_bytes)
        .sum();
    assert!(
        fwd_bytes > 10_000_000,
        "the fixture must build a multi-MB forward edge SST, got {fwd_bytes} bytes"
    );

    // Reader-node snapshot: committed manifest + empty memtable, cold caches.
    let memtable = Memtable::new();
    let view = memtable.snapshot_view();
    let snapshot = Snapshot::new(loaded, &view, store.clone(), paths.clone());

    let probe = 137u64;
    let before = counting.read_bytes();
    let edges = snapshot.out_edges("CITES", doc_id(probe)).await.unwrap();
    let cold_delta = counting.read_bytes() - before;

    assert_eq!(edges.edges.len(), DEGREE as usize, "full partner list");
    let mut expected: Vec<String> = (0..DEGREE).map(|slot| note(probe, slot)).collect();
    expected.sort();
    let mut got: Vec<String> = edges
        .edges
        .iter()
        .map(|edge| match edge.properties.get("note") {
            Some(CoreValue::Str(text)) => text.clone(),
            other => panic!("edge property must hydrate, got {other:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got, expected,
        "ranged hydration must return exact properties"
    );

    // The envelope: a cold single-key lookup (reader open + key lookup +
    // property row range) must read a sliver of the body, not the body.
    let ceiling = fwd_bytes / 20;
    assert!(
        cold_delta < ceiling,
        "cold partner lookup read {cold_delta} bytes of a {fwd_bytes}-byte edge SST \
         (ceiling {ceiling}); the whole-body hydration regressed"
    );

    // Warm repeat: the snapshot caches the paged reader, so a second lookup
    // for a different key must cost even less than the cold one.
    let before = counting.read_bytes();
    let edges = snapshot
        .out_edges("CITES", doc_id(probe + 1))
        .await
        .unwrap();
    let warm_delta = counting.read_bytes() - before;
    assert_eq!(edges.edges.len(), DEGREE as usize);
    assert!(
        warm_delta < cold_delta,
        "warm lookup ({warm_delta} bytes) must not exceed the cold one ({cold_delta})"
    );

    // Inverse direction shares the contract.
    let before = counting.read_bytes();
    let edges = snapshot.in_edges("CITES", doc_id(probe)).await.unwrap();
    let inverse_delta = counting.read_bytes() - before;
    assert_eq!(edges.edges.len(), DEGREE as usize, "inverse partner list");
    assert!(
        inverse_delta < ceiling,
        "cold inverse lookup read {inverse_delta} bytes (ceiling {ceiling})"
    );
}
