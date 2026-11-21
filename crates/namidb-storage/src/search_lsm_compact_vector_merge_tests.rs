//! A vector base rebuilt from the generation's own segments: parity with the
//! node-scan rebuild, and scheduling past the Nodes input cap.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::Value;
use object_store::memory::InMemory;
use object_store::ObjectStore;

use super::*;
use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
use crate::read::{Snapshot, VectorFilterSearch};
use crate::test_support::SEARCH_COMPACTION_ENV;
use crate::{NodeWriteRecord, WriterSession};

const INDEX: &str = "item_vec";
const LABEL: &str = "Item";
const OTHER: &str = "Other";
const TAGS: [&str; 3] = ["red", "green", "blue"];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Seed rows and dimension; scale locally with these variables to measure
/// the two rebuilds on a larger corpus.
fn seed_rows() -> u64 {
    env_usize("NAMIDB_VECTOR_MERGE_TEST_ROWS", 1_200) as u64
}

fn dim() -> u32 {
    env_usize("NAMIDB_VECTOR_MERGE_TEST_DIM", 8) as u32
}

fn schema() -> Schema {
    let dim = dim();
    SchemaBuilder::new()
        .label(LabelDef {
            name: LABEL.into(),
            properties: vec![
                PropertyDef::new("emb", DataType::FloatVector { dim }, true).unwrap(),
                PropertyDef::new("tag", DataType::Utf8, true)
                    .unwrap()
                    .with_indexed(true),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .label(LabelDef {
            name: OTHER.into(),
            properties: vec![
                PropertyDef::new("emb", DataType::FloatVector { dim }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn item_id(ordinal: u64) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[0] = 0x5a;
    bytes[8..].copy_from_slice(&ordinal.to_be_bytes());
    NodeId::from_uuid(uuid::Uuid::from_bytes(bytes))
}

fn embedding(ordinal: u64, version: u64) -> Vec<f32> {
    (0..u64::from(dim()))
        .map(|axis| {
            let bits = splitmix(ordinal.wrapping_mul(131) ^ version.rotate_left(29) ^ (axis << 40));
            (bits % 20_001) as f32 / 10_000.0 - 1.0
        })
        .collect()
}

fn tag(ordinal: u64) -> Option<&'static str> {
    (ordinal % 7 != 0).then(|| TAGS[(ordinal % 3) as usize])
}

/// About 3 KB of poorly compressible text per node, so the node tables are
/// much larger than the vector segments.
fn body(ordinal: u64) -> String {
    (0..190)
        .map(|chunk| format!("{:016x}", splitmix(ordinal * 1_000 + chunk)))
        .collect()
}

fn record(vector: Option<Vec<f32>>, tag: Option<&str>, ordinal: u64) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    if let Some(vector) = vector {
        properties.insert("emb".into(), Value::Vec(vector));
    }
    if let Some(tag) = tag {
        properties.insert("tag".into(), Value::Str(tag.into()));
    }
    properties.insert("body".into(), Value::Str(body(ordinal)));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

type Model = BTreeMap<NodeId, (Vec<f32>, Option<String>)>;

struct Corpus {
    writer: WriterSession,
    store: Arc<dyn ObjectStore>,
    paths: NamespacePaths,
    metric: VectorMetric,
    /// Live index members: vector and tag.
    model: Model,
}

impl Corpus {
    fn put_vector(&mut self, ordinal: u64, vector: Vec<f32>, tag: Option<&str>) {
        let id = item_id(ordinal);
        self.writer
            .upsert_node(LABEL, id, &record(Some(vector.clone()), tag, ordinal))
            .unwrap();
        if self.metric == VectorMetric::Cosine && vector.iter().all(|value| *value == 0.0) {
            self.model.remove(&id);
        } else {
            self.model.insert(id, (vector, tag.map(str::to_owned)));
        }
    }

    fn put(&mut self, ordinal: u64, version: u64, tag: Option<&str>) {
        self.put_vector(ordinal, embedding(ordinal, version), tag);
    }

    fn put_without_vector(&mut self, ordinal: u64) {
        let id = item_id(ordinal);
        self.writer
            .upsert_node(LABEL, id, &record(None, tag(ordinal), ordinal))
            .unwrap();
        self.model.remove(&id);
    }

    fn relabel(&mut self, ordinal: u64) {
        let id = item_id(ordinal);
        self.writer
            .upsert_node(
                OTHER,
                id,
                &record(Some(embedding(ordinal, 9)), None, ordinal),
            )
            .unwrap();
        self.model.remove(&id);
    }

    fn delete(&mut self, ordinal: u64) {
        let id = item_id(ordinal);
        self.writer.tombstone_node(LABEL, id).unwrap();
        self.model.remove(&id);
    }

    async fn flush(&mut self) {
        self.writer.flush(schema()).await.unwrap();
    }

    fn loaded(&self) -> LoadedManifest {
        self.writer.snapshot().manifest().clone()
    }
}

fn vector_state(manifest: &Manifest) -> &SearchLsmState {
    manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Vector && state.index_name == INDEX)
        .expect("vector generation")
}

async fn register(writer: &mut WriterSession, metric: VectorMetric) {
    writer
        .register_vector_index(
            VectorIndexDescriptor {
                name: INDEX.into(),
                label: LABEL.into(),
                property: "emb".into(),
                dim: dim(),
                metric,
                r: 16,
                l_build: 32,
                alpha: 1.2,
                quantization: VectorQuantization::None,
            },
            false,
        )
        .await
        .unwrap();
}

/// A seeded generation, then three flushes of churn: updates (the same nodes
/// twice), deletes, re-inserts, filter-only changes, a vector removed, a zero
/// vector, a relabel, and new nodes.
///
/// `index_first: false` writes the seed before the index exists, so the
/// generation is activated by a base built over the node tables (base + three
/// deltas). `index_first: true` creates the index on the empty namespace: the
/// generation is Active from its first flush and has no base (four deltas).
async fn corpus(name: &str, metric: VectorMetric, index_first: bool) -> Corpus {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    let mut corpus = Corpus {
        writer,
        store,
        paths,
        metric,
        model: Model::new(),
    };
    if index_first {
        register(&mut corpus.writer, metric).await;
    }
    let seed = seed_rows();
    for ordinal in 0..seed {
        corpus.put(ordinal, 0, tag(ordinal));
    }
    for ordinal in 50_000..50_020 {
        corpus
            .writer
            .upsert_node(
                OTHER,
                item_id(ordinal),
                &record(Some(embedding(ordinal, 0)), None, ordinal),
            )
            .unwrap();
    }
    corpus.put_without_vector(60_000);
    corpus.flush().await;

    if index_first {
        let loaded = corpus.loaded();
        let state = vector_state(&loaded.manifest);
        assert_eq!(state.status, SearchLsmStatus::Active);
        assert_eq!(state.segments.len(), 1);
        assert_eq!(state.segments[0].role, SearchSegmentRole::Delta);
    } else {
        register(&mut corpus.writer, metric).await;
        for ordinal in seed..seed + 10 {
            corpus.put(ordinal, 0, tag(ordinal));
        }
        corpus.flush().await;
        assert_eq!(
            vector_state(&corpus.loaded().manifest).status,
            SearchLsmStatus::Building
        );
        corpus.writer.compact_search_segments().await.unwrap();
        let loaded = corpus.loaded();
        let state = vector_state(&loaded.manifest);
        assert_eq!(state.status, SearchLsmStatus::Active);
        assert_eq!(state.segments.len(), 1);
        assert_eq!(state.segments[0].format, SearchSegmentFormat::VectorV5Base);
    }

    let fresh = seed + 1_000;
    for ordinal in 0..60 {
        corpus.put(ordinal, 1, tag(ordinal));
    }
    for ordinal in 100..130 {
        corpus.delete(ordinal);
    }
    for ordinal in fresh..fresh + 60 {
        corpus.put(ordinal, 0, tag(ordinal));
    }
    for ordinal in 200..215 {
        corpus.put(ordinal, 0, Some("violet"));
    }
    corpus.put_vector(300, vec![0.0; dim() as usize], tag(300));
    corpus.flush().await;

    for ordinal in 0..30 {
        corpus.put(ordinal, 2, tag(ordinal));
    }
    for ordinal in 100..115 {
        corpus.put(ordinal, 3, tag(ordinal));
    }
    for ordinal in fresh..fresh + 15 {
        corpus.delete(ordinal);
    }
    corpus.delete(400);
    corpus.put_without_vector(401);
    corpus.flush().await;

    for ordinal in 500..530 {
        corpus.put(ordinal, 4, None);
    }
    corpus.relabel(600);
    corpus.put(fresh + 1, 5, tag(fresh + 1));
    corpus.put(300, 6, tag(300));
    corpus.flush().await;

    let loaded = corpus.loaded();
    let state = vector_state(&loaded.manifest);
    assert_eq!(state.status, SearchLsmStatus::Active);
    let first = if index_first {
        (SearchSegmentRole::Delta, SearchSegmentFormat::VectorV6)
    } else {
        (SearchSegmentRole::Base, SearchSegmentFormat::VectorV5Base)
    };
    assert_eq!(
        state
            .segments
            .iter()
            .map(|segment| (segment.role, segment.format))
            .collect::<Vec<_>>(),
        vec![
            first,
            (SearchSegmentRole::Delta, SearchSegmentFormat::VectorV6),
            (SearchSegmentRole::Delta, SearchSegmentFormat::VectorV6),
            (SearchSegmentRole::Delta, SearchSegmentFormat::VectorV6),
        ],
        "a base (or the seed delta) followed by three flushed deltas"
    );
    corpus
}

fn policy(force_base: bool) -> SearchCompactionPolicy {
    SearchCompactionPolicy {
        delta_trigger_segments: 4,
        hard_segment_cap: 24,
        base_compact_bytes: u64::MAX,
        base_stale_percent: u64::MAX,
        force_base,
        vector_base_delta_rows: u64::MAX,
        vector_base_max_input_bytes: u64::MAX,
    }
}

fn vector_base_selections(
    manifest: &Manifest,
    policy: SearchCompactionPolicy,
) -> Vec<SearchCompactionSelection> {
    select_search_compactions(manifest, policy)
        .unwrap()
        .into_iter()
        .filter(|selection| {
            selection.captured_state.kind == SearchLsmKind::Vector
                && selection.mode == SearchCompactionMode::BasePrefix
        })
        .collect()
}

fn nodes_bytes(manifest: &Manifest) -> u64 {
    manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .map(|descriptor| descriptor.size_bytes)
        .sum()
}

fn segment_bytes(manifest: &Manifest) -> u64 {
    let state = vector_state(manifest);
    state
        .segments
        .iter()
        .map(|segment| {
            manifest
                .ssts
                .iter()
                .find(|descriptor| descriptor.id == segment.sst_id)
                .expect("segment descriptor")
                .size_bytes
        })
        .sum()
}

/// The manifest the install commits: the prepared base replaces its inputs
/// and, as the install does, a rebased compatibility barrier is published for
/// the new generation state.
async fn installed(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    manifest: &Manifest,
    prepared: &PreparedSearchCompaction,
) -> Manifest {
    let rebased =
        rebase_prepared_search_compaction(manifest, prepared, &prepared.selection.captured_state)
            .unwrap();
    let barrier = super::super::prepare_rebased_barrier(paths, &rebased)
        .unwrap()
        .expect("an active generation has a barrier");
    store
        .put(&barrier.object_path, barrier.body.clone().into())
        .await
        .unwrap();
    let selected = prepared.selection.selected_ids().collect::<HashSet<_>>();
    let mut projected = manifest.clone();
    projected
        .ssts
        .retain(|descriptor| !selected.contains(&descriptor.id) && descriptor.id != barrier.old_id);
    projected
        .ssts
        .push(prepared.output.as_ref().unwrap().descriptor.clone());
    projected.ssts.push(barrier.descriptor.clone());
    let position = projected
        .search_lsm
        .iter()
        .position(|state| state.kind == rebased.kind && state.index_name == rebased.index_name)
        .unwrap();
    projected.search_lsm[position] = barrier.state;
    crate::search_lsm::validate_search_lsm(&projected).unwrap();
    projected
}

fn score(metric: VectorMetric, vector: &[f32], query: &[f32]) -> f64 {
    let dot = vector
        .iter()
        .zip(query)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum::<f64>();
    match metric {
        VectorMetric::Dot => dot,
        VectorMetric::Cosine => {
            let norm = |values: &[f32]| {
                values
                    .iter()
                    .map(|value| f64::from(*value).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            dot / (norm(vector) * norm(query))
        }
        VectorMetric::Euclidean => -vector
            .iter()
            .zip(query)
            .map(|(left, right)| (f64::from(*left) - f64::from(*right)).powi(2))
            .sum::<f64>(),
    }
}

fn exact_top_k(
    model: &Model,
    metric: VectorMetric,
    query: &[f32],
    k: usize,
    tag: Option<&str>,
) -> Vec<NodeId> {
    let mut scored = model
        .iter()
        .filter(|(_, (_, row_tag))| tag.is_none_or(|tag| row_tag.as_deref() == Some(tag)))
        .map(|(id, (vector, _))| (score(metric, vector, query), *id))
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

async fn search(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    manifest: Manifest,
    query: &[f32],
    k: usize,
    tag: Option<&str>,
) -> Vec<NodeId> {
    let loaded = LoadedManifest::new(base.pointer.clone(), None, None, manifest);
    let memtable = MemtableSnapshot::empty();
    let snapshot = Snapshot::new(loaded, &memtable, store.clone(), paths.clone());
    let hits = match tag {
        None => snapshot
            .try_vector_search(INDEX, query, k, 256)
            .await
            .unwrap()
            .expect("native vector route"),
        Some(tag) => match snapshot
            .try_vector_search_filter_groups(
                INDEX,
                query,
                k,
                256,
                &[("tag".into(), vec![Value::Str(tag.into())])],
            )
            .await
            .unwrap()
            .expect("native filtered vector route")
        {
            VectorFilterSearch::Applied { hits, .. } => hits,
            VectorFilterSearch::Unsupported => panic!("the tag filter is native"),
        },
    };
    hits.into_iter().map(|(id, _)| id).collect()
}

/// Every row of a V5 body, NodeId-sorted.
async fn base_rows(body: Bytes) -> Vec<crate::sst::vector::v5::VectorV5StoredRow> {
    let len = body.len() as u64;
    let source: Arc<dyn crate::sst::vector::v5::VectorV5RangeSource> =
        Arc::new(LocalSegmentSource { body });
    let reader = crate::sst::vector::v5::VectorV5Reader::open(source, len)
        .await
        .unwrap();
    let mut rows = Vec::new();
    for page in 0..reader.page_count() {
        rows.extend(reader.read_page_rows(page).await.unwrap());
    }
    rows.sort_by_key(|row| row.id);
    rows
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn segment_merged_base_equals_the_node_scan_base() {
    let _env = SEARCH_COMPACTION_ENV
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (metric, index_first) in [
        (VectorMetric::Cosine, false),
        (VectorMetric::Euclidean, false),
        (VectorMetric::Cosine, true),
    ] {
        let name = format!("vector-merge-parity-{metric:?}-{index_first}").to_lowercase();
        let corpus = corpus(&name, metric, index_first).await;
        let loaded = corpus.loaded();
        let manifest = &loaded.manifest;
        let selections = vector_base_selections(manifest, policy(true));
        assert_eq!(selections.len(), 1);
        assert!(vector_segment_merge_supported(manifest, &selections[0]));

        let started = Instant::now();
        let merged = prepare_search_compactions_with(
            corpus.store.clone(),
            &corpus.paths,
            &loaded,
            selections.clone(),
            true,
            true,
            false,
        )
        .await
        .unwrap()
        .0;
        let merge_elapsed = started.elapsed();
        let started = Instant::now();
        let scanned = prepare_search_compactions_with(
            corpus.store.clone(),
            &corpus.paths,
            &loaded,
            selections,
            false,
            true,
            false,
        )
        .await
        .unwrap()
        .0;
        let scan_elapsed = started.elapsed();
        let [merged] = merged.as_slice() else {
            panic!("one merged base")
        };
        let [scanned] = scanned.as_slice() else {
            panic!("one scanned base")
        };
        eprintln!(
            "{metric:?} (index first: {index_first}): segment merge {merge_elapsed:?} (input {} bytes, nodes resident {}, \
             workspace {}), node scan {scan_elapsed:?} (input {} bytes, nodes resident {}, \
             workspace {})",
            merged.metrics.input_bytes,
            merged.metrics.peak_resident_input_bytes,
            merged.metrics.peak_workspace_bytes,
            scanned.metrics.input_bytes,
            scanned.metrics.peak_resident_input_bytes,
            scanned.metrics.peak_workspace_bytes,
        );

        // Each build took its own path.
        assert_eq!(merged.metrics.input_bytes, segment_bytes(manifest));
        assert_eq!(merged.metrics.peak_resident_input_bytes, 0);
        assert_eq!(scanned.metrics.input_bytes, nodes_bytes(manifest));
        assert!(scanned.metrics.peak_resident_input_bytes > 0);

        // Byte-identical bodies and identical base metadata.
        let merged_output = merged.output.as_ref().expect("merged base");
        let scanned_output = scanned.output.as_ref().expect("scanned base");
        let merged_body = super::super::get_sst_body(
            corpus.store.as_ref(),
            &corpus.paths,
            &merged_output.descriptor,
        )
        .await
        .unwrap();
        let scanned_body = super::super::get_sst_body(
            corpus.store.as_ref(),
            &corpus.paths,
            &scanned_output.descriptor,
        )
        .await
        .unwrap();
        assert!(
            merged_body == scanned_body,
            "the segment merge must write the node-scan base byte for byte"
        );
        assert_eq!(
            merged_output.descriptor.kind_specific,
            scanned_output.descriptor.kind_specific
        );
        assert_eq!(
            (
                merged_output.descriptor.min_key,
                merged_output.descriptor.max_key,
                merged_output.descriptor.row_count,
                merged_output.descriptor.max_lsn,
            ),
            (
                scanned_output.descriptor.min_key,
                scanned_output.descriptor.max_key,
                scanned_output.descriptor.row_count,
                scanned_output.descriptor.max_lsn,
            )
        );
        assert_eq!(
            merged_output.segment.complete_filter_properties,
            vec!["tag".to_owned()]
        );

        // Winners and vectors equal the reconciled corpus exactly.
        let rows = base_rows(merged_body).await;
        assert_eq!(rows.len(), corpus.model.len());
        for (row, (id, (vector, tag))) in rows.iter().zip(&corpus.model) {
            assert_eq!(&row.id, id.as_bytes());
            assert_eq!(&row.vector, vector);
            let expected = tag
                .iter()
                .map(|tag| ("tag".to_owned(), Value::Str(tag.clone())))
                .collect::<Vec<_>>();
            assert_eq!(row.filters, expected);
        }

        // Queries: before (base + deltas), after, and exact all agree.
        let after = installed(&corpus.store, &corpus.paths, manifest, merged).await;
        for probe in [7u64, 21, 205, 300, 512, seed_rows() + 1_001] {
            let query = embedding(probe, 11);
            for tag in [None, Some("red"), Some("violet")] {
                let exact = exact_top_k(&corpus.model, metric, &query, 10, tag);
                let (store, paths) = (&corpus.store, &corpus.paths);
                let before = search(store, paths, &loaded, manifest.clone(), &query, 10, tag).await;
                let rebuilt = search(store, paths, &loaded, after.clone(), &query, 10, tag).await;
                assert_eq!(before, exact, "{metric:?} base + deltas, tag {tag:?}");
                assert_eq!(rebuilt, exact, "{metric:?} merged base, tag {tag:?}");
            }
        }
    }
}

/// Restores the vector base policy variables on drop.
struct VectorBaseEnv {
    delta_rows: Option<std::ffi::OsString>,
    max_input_bytes: Option<std::ffi::OsString>,
}

impl VectorBaseEnv {
    const DELTA_ROWS: &'static str = "NAMIDB_SEARCH_LSM_VECTOR_BASE_DELTA_ROWS";
    const MAX_INPUT_BYTES: &'static str = "NAMIDB_SEARCH_LSM_VECTOR_BASE_MAX_INPUT_BYTES";

    fn set(delta_rows: u64, max_input_bytes: u64) -> Self {
        let restore = Self {
            delta_rows: std::env::var_os(Self::DELTA_ROWS),
            max_input_bytes: std::env::var_os(Self::MAX_INPUT_BYTES),
        };
        std::env::set_var(Self::DELTA_ROWS, delta_rows.to_string());
        std::env::set_var(Self::MAX_INPUT_BYTES, max_input_bytes.to_string());
        restore
    }
}

impl Drop for VectorBaseEnv {
    fn drop(&mut self) {
        for (name, value) in [
            (Self::DELTA_ROWS, self.delta_rows.take()),
            (Self::MAX_INPUT_BYTES, self.max_input_bytes.take()),
        ] {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vector_base_is_rebuilt_from_segments_past_the_nodes_input_cap() {
    let _env = SEARCH_COMPACTION_ENV
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut corpus = corpus("vector-merge-cap", VectorMetric::Cosine, false).await;
    let loaded = corpus.loaded();
    let manifest = &loaded.manifest;
    let segments = segment_bytes(manifest);
    let nodes = nodes_bytes(manifest);
    assert!(
        nodes > 4 * segments,
        "fixture: node tables ({nodes} bytes) dwarf the vector segments ({segments} bytes)"
    );
    // The flat rows are due for a base and the segments fit the cap, but the
    // node tables do not: only a segment merge may rebuild this base.
    let cap = segments * 2;
    let _policy_env = VectorBaseEnv::set(50, cap);
    let policy = SearchCompactionPolicy::from_env().unwrap();
    assert_eq!(policy.vector_base_max_input_bytes, cap);
    assert!(nodes > policy.vector_base_max_input_bytes);

    let selections = vector_base_selections(manifest, policy);
    assert_eq!(
        selections.len(),
        1,
        "the vector base rebuild must be scheduled although the node tables exceed the cap"
    );
    assert!(vector_segment_merge_supported(manifest, &selections[0]));
    let generation = selections[0].captured_state.generation_id;
    let (prepared, hold) =
        prepare_search_compactions_held(corpus.store.clone(), &corpus.paths, &loaded, selections)
            .await
            .unwrap();
    // A base folded from segments (no node scan) is registered too, and stays
    // in flight while the caller holds it for the install.
    assert!(hold.is_some());
    assert!(crate::search_progress::build_percent(generation).is_some());
    drop(hold);
    assert_eq!(crate::search_progress::build_percent(generation), None);
    let [prepared] = prepared.as_slice() else {
        panic!("one prepared base")
    };
    assert_eq!(
        prepared.metrics.input_bytes, segments,
        "the base was folded from the segments, not scanned from the node tables"
    );
    assert_eq!(prepared.metrics.peak_resident_input_bytes, 0);

    // End to end through the scheduled pass and the install.
    assert!(corpus.writer.search_compaction_due());
    corpus.writer.compact_search_segments().await.unwrap();
    let loaded = corpus.loaded();
    let state = vector_state(&loaded.manifest);
    assert_eq!(state.segments.len(), 1, "flat deltas folded into the base");
    assert_eq!(state.segments[0].format, SearchSegmentFormat::VectorV5Base);
    assert_eq!(
        state.segments[0].live_payload_count,
        corpus.model.len() as u64
    );
    let query = embedding(42, 3);
    assert_eq!(
        search(
            &corpus.store,
            &corpus.paths,
            &loaded,
            loaded.manifest.clone(),
            &query,
            10,
            None
        )
        .await,
        exact_top_k(&corpus.model, VectorMetric::Cosine, &query, 10, None)
    );

    // A generation whose segments cannot describe its corpus keeps the node
    // scan and its Nodes-sized input cap.
    let filters = vector_generation_filters(manifest, vector_state(manifest)).unwrap();
    let mut building = vector_state(manifest).clone();
    building.status = SearchLsmStatus::Building;
    assert!(!vector_segments_mergeable(&building, &filters));
    let mut shadow = vector_state(manifest).clone();
    shadow.segments[1].payload = SearchSegmentPayload::ShadowOnly;
    assert!(!vector_segments_mergeable(&shadow, &filters));
    let mut unfiltered = vector_state(manifest).clone();
    unfiltered.segments[2].complete_filter_properties.clear();
    assert!(!vector_segments_mergeable(&unfiltered, &filters));
    assert!(vector_segments_mergeable(vector_state(manifest), &filters));
}
