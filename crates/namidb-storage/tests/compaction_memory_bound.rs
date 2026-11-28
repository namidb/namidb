//! Peak heap of a node-bucket compaction over a wide L0 backlog.
//!
//! The merge must not keep state per input it has already consumed, and one
//! pass must not take an unbounded number of L0 inputs, so its heap stays a
//! small multiple of one input's decoded window however long the backlog is.
//! The test counts allocator bytes (its own global allocator, so only this
//! binary pays for it) while a backlog of wide rows is compacted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{SstKind, SstLevel};
use namidb_storage::{LocalFileObjectStore, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::ObjectStore;

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to `System` with the caller's layout; the
// counters are plain atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            if new_size >= layout.size() {
                let grow = new_size - layout.size();
                let now = CURRENT.fetch_add(grow, Ordering::Relaxed) + grow;
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const DIM: usize = 256;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Chunk".into(),
            properties: vec![
                PropertyDef::new("chunk_id", DataType::Utf8, true)
                    .unwrap()
                    .with_indexed(true),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn chunk(ordinal: u64) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert(
        "chunk_id".into(),
        CoreValue::Str(format!("chunk-{ordinal:08}")),
    );
    // Wide, poorly compressible rows: a text of distinct words and an
    // embedding sent as a plain list (the shape a JSON client produces).
    let body: String = (0..300)
        .map(|w| format!("w{}x{} ", ordinal.wrapping_mul(7919) % 100_003, w))
        .collect();
    props.insert("body".into(), CoreValue::Str(body));
    props.insert(
        "embedding".into(),
        CoreValue::List(
            (0..DIM)
                .map(|i| CoreValue::F64(((ordinal as f64) * 0.37 + i as f64).sin()))
                .collect(),
        ),
    );
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

fn l0_nodes(writer: &WriterSession) -> usize {
    writer
        .snapshot()
        .manifest()
        .manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes && d.level == SstLevel::L0)
        .count()
}

/// Build `files` L0 Nodes SSTs of `rows` rows each, then compact once and
/// return `(peak heap above the pre-compaction level, L0 count after)`.
async fn compact_backlog(name: &str, files: u64, rows: u64) -> (usize, usize) {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileObjectStore::new(dir.path()).unwrap());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let mut ordinal = 0u64;
    for _ in 0..files {
        for _ in 0..rows {
            writer
                .upsert_node("Chunk", NodeId::new(), &chunk(ordinal))
                .unwrap();
            ordinal += 1;
        }
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
    }
    assert_eq!(l0_nodes(&writer) as u64, files);
    let base = CURRENT.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    writer.compact_l0(&schema()).await.unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    (peak, l0_nodes(&writer))
}

/// Serialises the tests of this binary: they read process-wide counters
/// and knobs.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_merge_heap_does_not_grow_with_the_number_of_consumed_inputs() {
    let _serial = SERIAL.lock().await;
    // The same rows, spread over four times as many inputs. A merge that
    // keeps per-input state alive until it ends needs far more heap for the
    // wider fan-in; a bounded one needs about the same.
    let (narrow, _) = compact_backlog("merge-heap-narrow", 8, 600).await;
    let (wide, after) = compact_backlog("merge-heap-wide", 32, 150).await;
    // Five times the rows: the buffered chunk and the open row group are
    // bounded by bytes, so the peak must not follow the row count.
    let (deep, _) = compact_backlog("merge-heap-deep", 24, 1000).await;
    eprintln!(
        "peak heap: 4800 rows in 8 inputs {narrow} B, in 32 inputs {wide} B; \
         24000 rows in 24 inputs {deep} B; L0 after {after}"
    );
    assert_eq!(after, 0);
    if std::env::var_os("NAMIDB_COMPACTION_RELEASE_EXHAUSTED_INPUTS").is_none()
        && std::env::var_os("NAMIDB_NODE_SST_ROW_GROUP_MAX_BYTES").is_none()
    {
        assert!(
            wide < narrow.saturating_mul(3) / 2,
            "peak heap grew with the fan-in: 8 inputs {narrow} B, 32 inputs {wide} B"
        );
        assert!(
            deep < narrow.saturating_mul(2),
            "peak heap grew with the rows: 4800 rows {narrow} B, 24000 rows {deep} B"
        );
    }
}

fn versioned(key: u64, version: u64) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert("chunk_id".into(), CoreValue::Str(format!("chunk-{key:04}")));
    props.insert("version".into(), CoreValue::I64(version as i64));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

/// With an L0 fan-in cap, a backlog deeper than the cap drains through
/// several passes, each merging a run of at most the cap back into L0 and
/// then into L1, and every read stays exactly last-writer-wins on the way.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capped_l0_merges_converge_and_keep_last_writer_wins() {
    let _serial = SERIAL.lock().await;
    std::env::set_var("NAMIDB_COMPACTION_MAX_L0_INPUTS", "4");
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileObjectStore::new(dir.path()).unwrap());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("capped-l0").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let ids: Vec<NodeId> = (0..40).map(|_| NodeId::new()).collect();
    // key -> Some(version) live, None deleted.
    let mut model: BTreeMap<usize, Option<u64>> = BTreeMap::new();
    const FILES: u64 = 19;
    for file in 0..FILES {
        for (key, id) in ids.iter().enumerate() {
            // Each file rewrites a different subset; some files delete.
            if (key as u64 + file) % 3 == 0 {
                writer
                    .upsert_node("Chunk", *id, &versioned(key as u64, file))
                    .unwrap();
                model.insert(key, Some(file));
            } else if (key as u64 * 7 + file) % 11 == 0 && model.contains_key(&key) {
                writer.tombstone_node("Chunk", *id).unwrap();
                model.insert(key, None);
            }
        }
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
    }
    async fn check(
        writer: &WriterSession,
        ids: &[NodeId],
        model: &BTreeMap<usize, Option<u64>>,
        phase: &str,
    ) {
        let snapshot = writer.snapshot();
        for (key, id) in ids.iter().enumerate() {
            let got = snapshot
                .lookup_node("Chunk", *id)
                .await
                .unwrap()
                .and_then(|view| match view.properties.get("version") {
                    Some(CoreValue::I64(v)) => Some(*v as u64),
                    _ => None,
                });
            assert_eq!(
                got,
                model.get(&key).copied().flatten(),
                "{phase}: key {key}"
            );
        }
        let live = model.values().filter(|v| v.is_some()).count() as u64;
        assert_eq!(
            snapshot.count_nodes(Some("Chunk")).await.unwrap(),
            live,
            "{phase}: count"
        );
    }
    check(&writer, &ids, &model, "before").await;
    let mut l0 = l0_nodes(&writer);
    assert_eq!(l0 as u64, FILES);
    let mut passes = 0;
    while l0 > 1 && passes < 20 {
        writer.compact_l0(&schema()).await.unwrap();
        passes += 1;
        let after = l0_nodes(&writer);
        assert!(
            after < l0,
            "pass {passes}: L0 {l0} -> {after} did not shrink"
        );
        if l0 > 4 {
            assert!(
                after > 0,
                "pass {passes}: a capped pass drains only one run"
            );
        }
        check(&writer, &ids, &model, &format!("after pass {passes}")).await;
        l0 = after;
    }
    std::env::remove_var("NAMIDB_COMPACTION_MAX_L0_INPUTS");
    assert!(
        passes >= 3,
        "19 inputs under a cap of 4 need several passes, ran {passes}"
    );
    assert!(l0 <= 1, "the backlog drained, {l0} L0 left");
    let (tiered, _) = namidb_storage::compact::compaction_bound_stats();
    assert!(tiered >= 2, "tiered merges counted: {tiered}");
}

/// Write `files` L0 flushes over the same 40 ids (rewrites and deletes),
/// returning the ids and the model of what each must read as.
async fn overlapping_backlog(
    writer: &mut WriterSession,
    files: u64,
) -> (Vec<NodeId>, BTreeMap<usize, Option<u64>>) {
    let ids: Vec<NodeId> = (0..40).map(|_| NodeId::new()).collect();
    let mut model: BTreeMap<usize, Option<u64>> = BTreeMap::new();
    for file in 0..files {
        for (key, id) in ids.iter().enumerate() {
            if (key as u64 + file) % 3 == 0 {
                writer
                    .upsert_node("Chunk", *id, &versioned(key as u64, file))
                    .unwrap();
                model.insert(key, Some(file));
            } else if (key as u64 * 7 + file) % 11 == 0 && model.contains_key(&key) {
                writer.tombstone_node("Chunk", *id).unwrap();
                model.insert(key, None);
            }
        }
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
    }
    (ids, model)
}

fn version_of(view: &Option<namidb_storage::NodeView>) -> Option<u64> {
    view.as_ref()
        .and_then(|view| match view.properties.get("version") {
            Some(CoreValue::I64(v)) => Some(*v as u64),
            _ => None,
        })
}

/// A batch lookup over more overlapping SSTs than one probe wave folds them
/// wave by wave and still returns exactly the newest version of each id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_lookup_over_many_overlapping_ssts_folds_in_waves() {
    let _serial = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileObjectStore::new(dir.path()).unwrap());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("lookup-waves").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let (ids, model) = overlapping_backlog(&mut writer, 30).await;
    assert_eq!(l0_nodes(&writer), 30);
    let waves_before = namidb_storage::read::batch_lookup_probe_waves();
    let snapshot = writer.snapshot();
    let views = snapshot.batch_lookup_nodes("Chunk", &ids).await.unwrap();
    for (key, view) in views.iter().enumerate() {
        assert_eq!(
            version_of(view),
            model.get(&key).copied().flatten(),
            "key {key}"
        );
    }
    let waves = namidb_storage::read::batch_lookup_probe_waves() - waves_before;
    if !matches!(
        std::env::var("NAMIDB_BATCH_LOOKUP_WAVES").as_deref(),
        Ok("0")
    ) {
        // 30 overlapping SSTs, 8 per wave.
        assert!(waves >= 4, "waves {waves}");
    }
}

/// Right after an open, with overlapping SSTs and unflushed writes (no
/// manifest count can answer), a label count is exact: by the one
/// reconciliation by default, and by the label's seed with
/// `NAMIDB_COUNT_WAIT_FOR_SEED=1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn label_count_after_open_is_exact() {
    let _serial = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileObjectStore::new(dir.path()).unwrap());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("count-seed").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    let (_ids, model) = overlapping_backlog(&mut writer, 12).await;
    // Another label in the same SSTs must not leak into the count.
    for _ in 0..7 {
        writer
            .upsert_node("Doc", NodeId::new(), &versioned(0, 0))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    drop(writer);

    let mut writer = WriterSession::open(store, paths).await.unwrap();
    // An unflushed write: the manifest alone cannot count.
    writer
        .upsert_node("Chunk", NodeId::new(), &versioned(99, 99))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let live = model.values().filter(|v| v.is_some()).count() as u64 + 1;
    let (seeded_before, scanned_before) = namidb_storage::read::count_seed_wait_stats();
    let count = writer.snapshot().count_nodes(Some("Chunk")).await.unwrap();
    assert_eq!(count, live);
    assert_eq!(writer.snapshot().count_nodes(Some("Doc")).await.unwrap(), 7);
    let (seeded, scanned) = namidb_storage::read::count_seed_wait_stats();
    if matches!(
        std::env::var("NAMIDB_COUNT_WAIT_FOR_SEED").as_deref(),
        Ok("1")
    ) {
        assert!(
            seeded + scanned > seeded_before + scanned_before,
            "the label count took the seed route ({seeded}/{scanned})"
        );
    }
    // A later count is answered from the installed counts.
    assert_eq!(
        writer.snapshot().count_nodes(Some("Chunk")).await.unwrap(),
        live
    );
}

/// Manual probe: `PROBE_FILES` x `PROBE_ROWS` (default 24 x 1000), prints the
/// peak heap of one compaction. Run with `--ignored --nocapture`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn merge_heap_probe() {
    let _serial = SERIAL.lock().await;
    let var = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (files, rows) = (var("PROBE_FILES", 24), var("PROBE_ROWS", 1000));
    let (peak, after) = compact_backlog("merge-heap-probe", files, rows).await;
    eprintln!("probe: {files} inputs x {rows} rows: peak heap {peak} B, L0 after {after}");
}

/// At the memory ceiling a merge stops at its next slice boundary, and when
/// the ceiling still holds after the pause limit it gives up with
/// `MemoryLimit`, installing nothing; once the pressure is gone the same
/// pass completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_at_the_memory_ceiling_is_deferred_and_completes_after() {
    use namidb_storage::background::{memory_gate_stats, set_memory_pressure, MemoryPressure};
    let _serial = SERIAL.lock().await;
    if !namidb_storage::background::memory_gate_enabled() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalFileObjectStore::new(dir.path()).unwrap());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("merge-gate").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let mut ordinal = 0u64;
    for _ in 0..6 {
        for _ in 0..600 {
            writer
                .upsert_node("Chunk", NodeId::new(), &chunk(ordinal))
                .unwrap();
            ordinal += 1;
        }
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
    }
    let version = writer.snapshot().manifest().manifest.version;
    let (_, _, deferrals_before) = memory_gate_stats();
    set_memory_pressure(MemoryPressure::Ceiling);
    let deferred = writer.compact_l0(&schema()).await;
    set_memory_pressure(MemoryPressure::Normal);
    assert!(
        matches!(deferred, Err(namidb_storage::Error::MemoryLimit)),
        "a merge held at the ceiling is deferred: {deferred:?}"
    );
    assert_eq!(writer.snapshot().manifest().manifest.version, version);
    assert_eq!(l0_nodes(&writer), 6, "nothing installed");
    assert!(memory_gate_stats().2 > deferrals_before);

    writer.compact_l0(&schema()).await.unwrap();
    assert_eq!(l0_nodes(&writer), 0);
    assert_eq!(
        writer.snapshot().count_nodes(Some("Chunk")).await.unwrap(),
        ordinal
    );
}
