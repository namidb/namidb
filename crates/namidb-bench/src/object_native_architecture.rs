//! Production-architecture acceptance tracks for `object-native`.
//!
//! This module deliberately uses only public storage APIs. A clustered V5
//! Base, VG6 deltas and FT4 Base/deltas are opened through an instrumented
//! range source and coordinated here so the benchmark can distinguish format
//! regressions from manifest/query-executor plumbing.
//! The graph track injects an explicit immutable range cache and instruments
//! the object store below it, so logical reads, cache hits and physical GETs
//! remain separate measurements.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use futures::stream::BoxStream;
use namidb_core::Value;
use namidb_storage::manifest::{
    KindSpecificStats, SstDescriptor, SstKind, SstLevel, VectorMetric, VectorQuantization,
};
use namidb_storage::search_lsm::{
    legacy_base_content_fingerprint, SearchEventRange, SearchLsmKind, SearchLsmState,
    SearchLsmStatus, SearchSegmentFormat, SearchSegmentPayload, SearchSegmentRef,
    SearchSegmentRole, SearchSegmentStats, SearchStatValue,
};
use namidb_storage::sst::edges::{
    EdgeDirection, EdgeRecord, EdgeSstWriter, EdgeSstWriterOptions, PagedEdgeIoStats,
    PagedEdgeReader,
};
use namidb_storage::sst::search_delta::{
    reconcile_node_versions, SearchFilterValue, SearchVersionOperation, SearchVersionRangeSource,
};
use namidb_storage::sst::text::v4::{
    TextV4BuildContext, TextV4ExternalBuildConfig, TextV4ExternalBuildMetrics,
    TextV4ExternalBuilder, TextV4GlobalStats, TextV4Hit, TextV4Mutation, TextV4Payload,
    TextV4Reader,
};
use namidb_storage::sst::vector::v5::external::{
    VectorV5ExternalBuildConfig, VectorV5ExternalCollector,
};
use namidb_storage::sst::vector::v5::{
    VectorV5BuildOptions, VectorV5RangeSource, VectorV5Reader, VectorV5SearchOptions,
};
use namidb_storage::sst::vector::v6::{
    VectorV6BuildContext, VectorV6BuildOptions, VectorV6ExternalBuildConfig,
    VectorV6ExternalBuilder, VectorV6Mutation, VectorV6Payload, VectorV6Reader,
};
use namidb_storage::text::{avg_len, bm25_idf, bm25_term_score, parse_query, tokenize, TextQuery};
use namidb_storage::{ImmutableRangeCache, RangeCacheConfig, RangeCacheStats};
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use serde::Serialize;
use uuid::Uuid;

use super::{
    compare_text_results, corpus_vector, elapsed_us, l2_norm, legal_document, node_id, percentile,
    query_vector, ratio, InstrumentedRangeSource, ObjectNativeConfig, RangeIoAggregate,
    RangeIoStats,
};

const ACTIVE_FILTER: &str = "active";
const TENANT_FILTER: &str = "tenant";
const LSM_VECTOR_INDEX: &str = "object-native-vg6-lsm";
const LSM_TEXT_INDEX: &str = "object-native-ft4-lsm";

#[derive(Debug)]
pub(super) struct ArchitectureRun {
    pub search_lsm: SearchLsmReport,
    pub graph: GraphPagedReport,
    pub builder_workspace: ArchitectureBuilderWorkspaceReport,
}

#[derive(Debug, Serialize)]
pub struct SearchLsmReport {
    pub seed_segment_model: &'static str,
    pub delta_segments: usize,
    pub total_segments_per_family: usize,
    pub vector_segment_roles: Vec<&'static str>,
    pub text_segment_roles: Vec<&'static str>,
    pub vector_segment_stats: Vec<&'static str>,
    pub text_segment_stats: Vec<&'static str>,
    pub v5_base_contract: V5BaseContractReport,
    pub ft4_base_contract: Ft4BaseContractReport,
    pub mutation_mix: Vec<DeltaMutationReport>,
    pub vector_serving_options: VectorServingOptionsReport,
    pub vector_serving_ann: VersionedSearchFamilyReport,
    pub vector_exact_shadow: VectorExactShadowReport,
    pub text: VersionedSearchFamilyReport,
    pub max_fanout_observed: usize,
    pub max_in_flight_observed: u64,
    pub winner_adapter: &'static str,
    pub api_limitations: Vec<ApiLimitation>,
}

#[derive(Debug, Serialize)]
pub struct VectorServingOptionsReport {
    pub base_algorithm: &'static str,
    pub delta_algorithm: &'static str,
    pub nprobe: usize,
    pub max_nprobe: usize,
    pub rerank_factor: usize,
}

#[derive(Debug, Serialize)]
pub struct VectorExactShadowReport {
    pub query_count: usize,
    pub unfiltered: WinnerParityReport,
    pub active_filter: WinnerParityReport,
    pub active_filter_applied: bool,
    pub io_accounting: &'static str,
}

#[derive(Debug, Serialize)]
pub struct V5BaseContractReport {
    pub role_is_base: bool,
    pub format_is_v5: bool,
    pub live_count_is_absolute: bool,
    pub suppress_count_is_zero: bool,
    pub native_footer_validated: bool,
    pub legacy_manifest_binding_nonzero: bool,
    pub point_count: Option<u64>,
    pub clustered_page_count: u64,
    pub bounded_builder_peak_logical_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct Ft4BaseContractReport {
    pub role_is_base: bool,
    pub doc_count_is_absolute: bool,
    pub total_len_is_absolute: bool,
    pub suppress_count_is_zero: bool,
    pub document_count: Option<u64>,
    pub total_document_len: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct DeltaMutationReport {
    pub delta: usize,
    pub vector: MutationCounts,
    pub text: MutationCounts,
}

#[derive(Debug, Default, Serialize)]
pub struct MutationCounts {
    pub updates: usize,
    pub deletes: usize,
    pub filter_only_updates: usize,
}

#[derive(Debug, Serialize)]
pub struct VersionedSearchFamilyReport {
    pub format: &'static str,
    pub artifact_bytes: u64,
    pub resident_metadata_bytes: usize,
    pub resident_metadata_artifact_ratio: f64,
    pub reader_open_io_no_cache: RangeIoStats,
    pub reader_open_io_sized_cache: RangeIoStats,
    pub builder_peak_logical_bytes: usize,
    pub unfiltered: VersionedQueryTrack,
    pub active_filter: VersionedQueryTrack,
}

#[derive(Debug, Serialize)]
pub struct VersionedQueryTrack {
    pub result_model: &'static str,
    pub query_count: usize,
    pub native_filter_applied: bool,
    pub recall_at_k: f64,
    pub returned_winners_valid: bool,
    pub returned_scores_exact: bool,
    pub max_returned_score_abs_error: f64,
    pub cache_results_identical: bool,
    pub parity: WinnerParityReport,
    pub no_cache: QueryPassReport,
    pub sized_cache_cold: QueryPassReport,
    pub sized_cache_warm: QueryPassReport,
    pub max_fanout: usize,
}

#[derive(Debug, Default, Serialize)]
pub struct WinnerParityReport {
    pub node_ids_exact: bool,
    pub scores_exact: bool,
    pub max_score_abs_error: f64,
    pub expected_hits: usize,
    pub actual_hits: usize,
}

#[derive(Debug, Serialize)]
pub struct QueryPassReport {
    pub cache: &'static str,
    pub p50_us: u64,
    pub p95_us: u64,
    pub io: RangeIoAggregate,
}

#[derive(Debug, Serialize)]
pub struct ArchitectureBuilderWorkspaceReport {
    pub v5_peak_logical_bytes: usize,
    pub ft3_peak_logical_bytes: usize,
    pub vg6_peak_logical_bytes: usize,
    pub ft4_peak_logical_bytes: usize,
    pub peak_logical_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct GraphPagedReport {
    pub artifact_bytes: u64,
    pub edge_count: u64,
    pub high_degree: usize,
    pub forward: GraphDirectionReport,
    pub inverse: GraphDirectionReport,
    pub parity_exact: bool,
    pub max_cold_artifact_ratio: f64,
    pub cache_modes: GraphCacheModeReport,
    pub api_limitations: Vec<ApiLimitation>,
}

#[derive(Debug, Serialize)]
pub struct GraphDirectionReport {
    pub direction: &'static str,
    pub range_complete: bool,
    pub resident_metadata_bytes: usize,
    pub resident_metadata_artifact_ratio: f64,
    pub reader_open_no_cache: GraphIoSample,
    pub reader_open_sized_cache: GraphIoSample,
    pub operations: Vec<GraphOperationReport>,
}

#[derive(Debug, Serialize)]
pub struct GraphOperationReport {
    pub operation: &'static str,
    pub parity_exact: bool,
    pub no_cache: GraphIoSample,
    pub sized_cache_cold: GraphIoSample,
    pub sized_cache_warm: GraphIoSample,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct GraphIoSample {
    pub logical_requests: u64,
    pub logical_bytes: u64,
    pub fetch_requests: u64,
    pub fetched_bytes: u64,
    pub cache_hits: u64,
    pub peak_in_flight: u64,
    pub eager_body_reads: u64,
    pub artifact_ratio: f64,
}

#[derive(Debug, Serialize)]
pub struct GraphCacheModeReport {
    pub explicit_zero_and_sized_available: bool,
    pub measured_counters: &'static str,
    pub no_cache: &'static str,
    pub sized_cache_cold: &'static str,
    pub sized_cache_warm: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ApiLimitation {
    pub component: &'static str,
    pub minimum_public_api_required: &'static str,
    pub consequence: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MutationKind {
    Update,
    Delete,
    FilterOnly,
}

struct VectorArtifact {
    file: File,
    len: u64,
    segment: SearchSegmentRef,
    kind: VectorArtifactKind,
    page_count: u64,
    builder_peak_logical_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorArtifactKind {
    V5Base,
    V6Delta,
}

struct TextArtifact {
    file: File,
    len: u64,
    segment: SearchSegmentRef,
    metrics: TextV4ExternalBuildMetrics,
}

struct VectorReaderSet {
    readers: Vec<VectorSegmentReader>,
    sources: Vec<Arc<InstrumentedRangeSource>>,
    open_io: RangeIoStats,
    metadata_bytes: usize,
}

enum VectorSegmentReader {
    V5Base(VectorV5Reader),
    V6Delta(VectorV6Reader),
}

impl VectorSegmentReader {
    fn resident_metadata_bytes(&self) -> usize {
        match self {
            Self::V5Base(reader) => reader.resident_metadata_bytes(),
            Self::V6Delta(reader) => reader.resident_metadata_bytes(),
        }
    }

    fn version_reader(
        &self,
    ) -> Option<&namidb_storage::sst::search_delta::SearchVersionTableReader> {
        match self {
            Self::V5Base(_) => None,
            Self::V6Delta(reader) => Some(reader.version_reader()),
        }
    }
}

struct TextReaderSet {
    readers: Vec<TextV4Reader>,
    sources: Vec<Arc<InstrumentedRangeSource>>,
    open_io: RangeIoStats,
    metadata_bytes: usize,
}

pub(super) async fn run_architecture(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    v5_peak_logical_bytes: usize,
    ft3_peak_logical_bytes: usize,
) -> Result<ArchitectureRun> {
    let mut vector_state = search_state(SearchLsmKind::Vector, LSM_VECTOR_INDEX, 0x56);
    vector_state.base_frontier = Some(1);
    vector_state.next_event_seq = config.delta_segments as u64 + 1;
    let mut text_state = search_state(SearchLsmKind::Text, LSM_TEXT_INDEX, 0x54);
    text_state.base_frontier = Some(1);
    text_state.next_event_seq = config.delta_segments as u64 + 1;
    let descriptor = namidb_storage::manifest::VectorIndexDescriptor {
        name: LSM_VECTOR_INDEX.to_owned(),
        label: "Document".to_owned(),
        property: "embedding".to_owned(),
        dim: config.dim as u32,
        metric: VectorMetric::Cosine,
        r: 32,
        l_build: 64,
        alpha: 1.2,
        quantization: VectorQuantization::None,
    };
    let vector_scratch =
        tempfile::tempdir().context("create V5 base benchmark scratch directory")?;
    let vector_artifacts = build_vector_segments(
        config,
        centroids,
        &vector_state,
        &descriptor,
        vector_scratch.path(),
    )?;
    let text_scratch = tempfile::tempdir().context("create FT4 benchmark scratch directory")?;
    let text_artifacts = build_text_segments(config, &text_state, text_scratch.path())?;

    let search_v5_peak = vector_artifacts
        .iter()
        .filter(|artifact| artifact.kind == VectorArtifactKind::V5Base)
        .map(|artifact| artifact.builder_peak_logical_bytes)
        .max()
        .unwrap_or(0);
    let vg6_peak = vector_artifacts
        .iter()
        .filter(|artifact| artifact.kind == VectorArtifactKind::V6Delta)
        .map(|artifact| artifact.builder_peak_logical_bytes)
        .max()
        .unwrap_or(0);
    let text_peak = text_artifacts
        .iter()
        .map(|artifact| artifact.metrics.peak_logical_memory_bytes)
        .max()
        .unwrap_or(0);
    let (vector_serving_ann, vector_exact_shadow) = run_vector_lsm(
        config,
        centroids,
        &vector_state,
        &descriptor,
        &vector_artifacts,
    )
    .await?;
    let text = run_text_lsm(config, &text_state, &text_artifacts).await?;
    let max_in_flight_observed =
        max_family_in_flight(&vector_serving_ann).max(max_family_in_flight(&text));
    let max_fanout_observed = vector_serving_ann
        .unfiltered
        .max_fanout
        .max(vector_serving_ann.active_filter.max_fanout)
        .max(text.unfiltered.max_fanout)
        .max(text.active_filter.max_fanout);
    let search_lsm = SearchLsmReport {
        seed_segment_model:
            "clustered absolute V5 Base + N VG6 deltas; absolute FT4 Base + N FT4 deltas",
        delta_segments: config.delta_segments,
        total_segments_per_family: config.delta_segments + 1,
        vector_segment_roles: vector_artifacts
            .iter()
            .map(|artifact| segment_role_name(artifact.segment.role))
            .collect(),
        text_segment_roles: text_artifacts
            .iter()
            .map(|artifact| segment_role_name(artifact.segment.role))
            .collect(),
        vector_segment_stats: vector_artifacts
            .iter()
            .map(|artifact| vector_stat_model(&artifact.segment))
            .collect(),
        text_segment_stats: text_artifacts
            .iter()
            .map(|artifact| text_stat_model(&artifact.segment))
            .collect(),
        v5_base_contract: v5_base_contract(&vector_artifacts),
        ft4_base_contract: ft4_base_contract(&text_artifacts),
        mutation_mix: delta_mutation_reports(config),
        vector_serving_options: VectorServingOptionsReport {
            base_algorithm: "clustered V5 ANN with native-filter adaptive page widening",
            delta_algorithm: "exhaustive exact VG6 delta pages with highest-LSN reconciliation",
            nprobe: config.nprobe,
            max_nprobe: config.max_nprobe,
            rerank_factor: config.rerank_factor,
        },
        vector_serving_ann,
        vector_exact_shadow,
        text,
        max_fanout_observed,
        max_in_flight_observed,
        winner_adapter: "bench-only public-reader coordinator: serving V5 ANN + VG6 with highest-LSN fingerprint validation; page-bounded exact shadow kept outside serving metrics",
        api_limitations: Vec::new(),
    };
    let graph = run_graph(config).await?;
    let effective_v5_peak = v5_peak_logical_bytes.max(search_v5_peak);
    let peak_logical_bytes = effective_v5_peak
        .max(ft3_peak_logical_bytes)
        .max(vg6_peak)
        .max(text_peak);
    Ok(ArchitectureRun {
        search_lsm,
        graph,
        builder_workspace: ArchitectureBuilderWorkspaceReport {
            v5_peak_logical_bytes: effective_v5_peak,
            ft3_peak_logical_bytes,
            vg6_peak_logical_bytes: vg6_peak,
            ft4_peak_logical_bytes: text_peak,
            peak_logical_bytes,
        },
    })
}

fn v5_base_contract(artifacts: &[VectorArtifact]) -> V5BaseContractReport {
    let Some(base) = artifacts.first() else {
        return V5BaseContractReport {
            role_is_base: false,
            format_is_v5: false,
            live_count_is_absolute: false,
            suppress_count_is_zero: false,
            native_footer_validated: false,
            legacy_manifest_binding_nonzero: false,
            point_count: None,
            clustered_page_count: 0,
            bounded_builder_peak_logical_bytes: 0,
        };
    };
    let point_count = match base.segment.stats {
        SearchSegmentStats::Vector {
            live_count: SearchStatValue::Absolute(point_count),
        } => Some(point_count),
        _ => None,
    };
    V5BaseContractReport {
        role_is_base: base.segment.role == SearchSegmentRole::Base,
        format_is_v5: base.segment.format == SearchSegmentFormat::VectorV5Base
            && base.kind == VectorArtifactKind::V5Base,
        live_count_is_absolute: point_count.is_some(),
        suppress_count_is_zero: base.segment.suppress_count == 0,
        // run_vector_lsm opens this artifact in both zero-cache and sized-cache
        // reader sets before the report is constructed. Open validates magic,
        // trailer, footer bounds, footer checksum and the complete directory.
        native_footer_validated: true,
        legacy_manifest_binding_nonzero: base.segment.content_xxh3 != 0,
        point_count,
        clustered_page_count: base.page_count,
        bounded_builder_peak_logical_bytes: base.builder_peak_logical_bytes,
    }
}

const fn segment_role_name(role: SearchSegmentRole) -> &'static str {
    match role {
        SearchSegmentRole::Base => "base",
        SearchSegmentRole::Delta => "delta",
    }
}

fn vector_stat_model(segment: &SearchSegmentRef) -> &'static str {
    match (segment.role, segment.stats) {
        (
            SearchSegmentRole::Base,
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Absolute(_),
            },
        ) => "absolute",
        (
            SearchSegmentRole::Delta,
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(_),
            },
        ) => "delta",
        _ => "invalid",
    }
}

fn text_stat_model(segment: &SearchSegmentRef) -> &'static str {
    match (segment.role, segment.stats) {
        (
            SearchSegmentRole::Base,
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Absolute(_),
                total_len: SearchStatValue::Absolute(_),
                ..
            },
        ) => "absolute",
        (
            SearchSegmentRole::Delta,
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Delta(_),
                total_len: SearchStatValue::Delta(_),
                ..
            },
        ) => "delta",
        _ => "invalid",
    }
}

fn ft4_base_contract(artifacts: &[TextArtifact]) -> Ft4BaseContractReport {
    let Some(base) = artifacts.first() else {
        return Ft4BaseContractReport {
            role_is_base: false,
            doc_count_is_absolute: false,
            total_len_is_absolute: false,
            suppress_count_is_zero: false,
            document_count: None,
            total_document_len: None,
        };
    };
    let (document_count, total_document_len) = match base.segment.stats {
        SearchSegmentStats::Text {
            doc_count: SearchStatValue::Absolute(documents),
            total_len: SearchStatValue::Absolute(total_len),
            term_df_violation_count: 0,
        } => (Some(documents), Some(total_len)),
        _ => (None, None),
    };
    Ft4BaseContractReport {
        role_is_base: base.segment.role == SearchSegmentRole::Base,
        doc_count_is_absolute: document_count.is_some(),
        total_len_is_absolute: total_document_len.is_some(),
        suppress_count_is_zero: base.segment.suppress_count == 0,
        document_count,
        total_document_len,
    }
}

fn search_state(kind: SearchLsmKind, index_name: &str, discriminator: u8) -> SearchLsmState {
    let mut generation = [0u8; 16];
    generation[0] = discriminator;
    generation[15] = 2;
    SearchLsmState {
        index_name: index_name.to_owned(),
        kind,
        catalog_signature: format!("object-native-v2-{index_name}"),
        generation_id: Uuid::from_bytes(generation),
        status: SearchLsmStatus::Active,
        ..SearchLsmState::default()
    }
}

fn mutation_kind(row: usize, delta: usize, delta_count: usize) -> Option<MutationKind> {
    let bucket = row % delta_count.saturating_mul(8);
    if bucket == delta - 1 {
        Some(MutationKind::Delete)
    } else if bucket == delta_count + delta - 1 {
        Some(MutationKind::Update)
    } else if bucket == delta_count.saturating_mul(2) + delta - 1 {
        Some(MutationKind::FilterOnly)
    } else {
        None
    }
}

fn row_mutation(row: usize, delta_count: usize) -> Option<(usize, MutationKind)> {
    (1..=delta_count)
        .find_map(|delta| mutation_kind(row, delta, delta_count).map(|kind| (delta, kind)))
}

fn delta_mutation_reports(config: &ObjectNativeConfig) -> Vec<DeltaMutationReport> {
    (1..=config.delta_segments)
        .map(|delta| {
            let mut report = DeltaMutationReport {
                delta,
                vector: MutationCounts::default(),
                text: MutationCounts::default(),
            };
            for row in 0..config.vectors {
                match mutation_kind(row, delta, config.delta_segments) {
                    Some(MutationKind::Update) => report.vector.updates += 1,
                    Some(MutationKind::Delete) => report.vector.deletes += 1,
                    Some(MutationKind::FilterOnly) => {
                        report.vector.filter_only_updates += 1;
                    }
                    None => {}
                }
            }
            for row in 0..config.documents {
                match mutation_kind(row, delta, config.delta_segments) {
                    Some(MutationKind::Update) => report.text.updates += 1,
                    Some(MutationKind::Delete) => report.text.deletes += 1,
                    Some(MutationKind::FilterOnly) => {
                        report.text.filter_only_updates += 1;
                    }
                    None => {}
                }
            }
            report
        })
        .collect()
}

fn filter_values(row: usize, active: bool, buckets: usize) -> BTreeMap<String, SearchFilterValue> {
    BTreeMap::from([
        (ACTIVE_FILTER.to_owned(), SearchFilterValue::Bool(active)),
        (
            TENANT_FILTER.to_owned(),
            SearchFilterValue::String(format!("tenant-{}", row % buckets)),
        ),
    ])
}

fn base_active(row: usize) -> bool {
    row % 3 != 0
}

fn updated_vector(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    row: usize,
    delta: usize,
) -> Vec<f32> {
    let mut vector = corpus_vector(config, centroids, VectorMetric::Cosine, row);
    if !vector.is_empty() {
        let vector_len = vector.len();
        let first = delta % vector_len;
        vector[first] += 0.35;
        if vector_len > 1 {
            let second = (first + 1) % vector_len;
            vector[second] -= 0.17;
        }
        let norm = l2_norm(&vector);
        if norm != 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
    }
    vector
}

fn vector_payload(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    row: usize,
    mutation: Option<(usize, MutationKind)>,
) -> Option<VectorV6Payload> {
    if mutation.is_some_and(|(_, kind)| kind == MutationKind::Delete) {
        return None;
    }
    let vector = match mutation {
        Some((delta, MutationKind::Update)) => updated_vector(config, centroids, row, delta),
        _ => corpus_vector(config, centroids, VectorMetric::Cosine, row),
    };
    let active = match mutation {
        Some((_, MutationKind::FilterOnly)) => !base_active(row),
        _ => base_active(row),
    };
    Some(VectorV6Payload {
        vector,
        filters: filter_values(row, active, config.filter_buckets),
    })
}

fn v5_base_filters(row: usize, active: bool, buckets: usize) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (ACTIVE_FILTER.to_owned(), Value::Bool(active)),
        (
            TENANT_FILTER.to_owned(),
            Value::Str(format!("tenant-{}", row % buckets)),
        ),
    ])
}

fn text_payload(
    config: &ObjectNativeConfig,
    row: usize,
    mutation: Option<(usize, MutationKind)>,
) -> Option<TextV4Payload> {
    if mutation.is_some_and(|(_, kind)| kind == MutationKind::Delete) {
        return None;
    }
    let text = match mutation {
        Some((delta, MutationKind::Update)) => format!(
            "{} actualización consolidada reforma delta{delta}",
            legal_document(row, config.seed)
        ),
        _ => legal_document(row, config.seed),
    };
    let active = match mutation {
        Some((_, MutationKind::FilterOnly)) => !base_active(row),
        _ => base_active(row),
    };
    Some(TextV4Payload {
        text,
        filters: filter_values(row, active, config.filter_buckets),
    })
}

fn build_vector_segments(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    state: &SearchLsmState,
    descriptor: &namidb_storage::manifest::VectorIndexDescriptor,
    spool_directory: &std::path::Path,
) -> Result<Vec<VectorArtifact>> {
    let mut artifacts = Vec::with_capacity(config.delta_segments + 1);
    let mut base_builder = VectorV5ExternalCollector::new_base(VectorV5ExternalBuildConfig {
        scratch_dir: spool_directory.to_path_buf(),
        memory_budget_bytes: config.build_memory_bytes,
        quantile_sample_rows: 16_384.min(config.vectors.max(2)),
    })
    .map_err(|error| anyhow!("create clustered V5 Search-LSM base: {error}"))?;
    for row in 0..config.vectors {
        let vector = corpus_vector(config, centroids, VectorMetric::Cosine, row);
        base_builder
            .push(
                node_id(row),
                &vector,
                &v5_base_filters(row, base_active(row), config.filter_buckets),
            )
            .map_err(|error| anyhow!("append V5 Search-LSM base row {row}: {error}"))?;
    }
    let base = base_builder
        .finish(
            descriptor,
            VectorV5BuildOptions {
                target_rows_per_page: config.page_rows,
                branch_factor: config.branch_factor,
                compression_level: 3,
            },
        )
        .map_err(|error| anyhow!("finish clustered V5 Search-LSM base: {error}"))?
        .ok_or_else(|| anyhow!("clustered V5 Search-LSM base unexpectedly empty"))?;
    let base_id = Uuid::from_bytes({
        let mut bytes = [0u8; 16];
        bytes[0] = 0x75;
        bytes[15] = 1;
        bytes
    });
    let complete_filter_properties = vec![ACTIVE_FILTER.to_owned(), TENANT_FILTER.to_owned()];
    let base_manifest_descriptor = SstDescriptor {
        id: base_id,
        kind: SstKind::VectorGraph,
        scope: descriptor.name.clone(),
        level: SstLevel(1),
        path: format!("bench/{}.vg", base_id.simple()),
        size_bytes: base.len,
        row_count: base.stats.point_count,
        created_at: Utc::now(),
        min_key: base.stats.min_node_id,
        max_key: base.stats.max_node_id,
        min_lsn: 1,
        max_lsn: 1,
        schema_version_min: 0,
        schema_version_max: 0,
        property_stats: Vec::new(),
        kind_specific: KindSpecificStats::VectorGraph {
            dim: base.stats.dim,
            metric: base.stats.metric.clone(),
            point_count: base.stats.point_count,
            r: base.stats.r,
            l_build: base.stats.l_build,
            alpha: base.stats.alpha,
            entry_medoid: base.stats.entry_medoid,
        },
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    };
    let base_segment = SearchSegmentRef {
        sst_id: base_id,
        role: SearchSegmentRole::Base,
        format: SearchSegmentFormat::VectorV5Base,
        payload: SearchSegmentPayload::Complete,
        event_ranges: vec![SearchEventRange::new(0, 1)],
        min_lsn: 1,
        max_lsn: 1,
        mutation_count: base.stats.point_count,
        live_payload_count: base.stats.point_count,
        suppress_count: 0,
        content_xxh3: legacy_base_content_fingerprint(
            &base_manifest_descriptor,
            SearchSegmentFormat::VectorV5Base,
            &complete_filter_properties,
        ),
        complete_filter_properties,
        stats: SearchSegmentStats::Vector {
            live_count: SearchStatValue::Absolute(base.stats.point_count),
        },
        equal_lsn_conflict_count: 0,
    };
    artifacts.push(VectorArtifact {
        file: base.file,
        len: base.len,
        segment: base_segment,
        kind: VectorArtifactKind::V5Base,
        page_count: base.metrics.page_count,
        builder_peak_logical_bytes: base.metrics.peak_logical_memory_bytes,
    });

    for segment_no in 1..=config.delta_segments {
        let mut bytes = [0u8; 16];
        bytes[0] = 0x76;
        bytes[8..].copy_from_slice(&(segment_no as u64 + 1).to_be_bytes());
        let context = VectorV6BuildContext {
            sst_id: Uuid::from_bytes(bytes),
            event_ranges: vec![SearchEventRange::new(
                segment_no as u64,
                segment_no as u64 + 1,
            )],
            complete_filter_properties: vec![ACTIVE_FILTER.to_owned(), TENANT_FILTER.to_owned()],
        };
        let mut builder = VectorV6ExternalBuilder::with_config(
            state,
            descriptor,
            context,
            VectorV6ExternalBuildConfig {
                memory_budget_bytes: config.build_memory_bytes,
                wire: VectorV6BuildOptions {
                    rows_per_page: config.page_rows,
                    compression_level: 3,
                },
                ..VectorV6ExternalBuildConfig::default()
            },
        )
        .map_err(|error| anyhow!("create VG6 segment {segment_no}: {error}"))?;
        for row in 0..config.vectors {
            if mutation_kind(row, segment_no, config.delta_segments).is_none() {
                continue;
            }
            let before = vector_payload(config, centroids, row, None);
            let after = vector_payload(
                config,
                centroids,
                row,
                mutation_kind(row, segment_no, config.delta_segments)
                    .map(|kind| (segment_no, kind)),
            );
            builder
                .push(VectorV6Mutation {
                    node_id: node_id(row),
                    lsn: segment_no as u64 + 1,
                    before,
                    after,
                })
                .map_err(|error| anyhow!("append VG6 segment {segment_no} row {row}: {error}"))?;
        }
        let artifact = builder
            .finish()
            .map_err(|error| anyhow!("finish VG6 segment {segment_no}: {error}"))?
            .ok_or_else(|| anyhow!("VG6 segment {segment_no} unexpectedly empty"))?;
        let page_count = u64::from(artifact.output.page_count);
        let builder_peak_logical_bytes = artifact.metrics.peak_logical_memory_bytes;
        artifacts.push(VectorArtifact {
            file: artifact.file,
            len: artifact.len,
            segment: artifact.output.segment,
            kind: VectorArtifactKind::V6Delta,
            page_count,
            builder_peak_logical_bytes,
        });
    }
    Ok(artifacts)
}

fn build_text_segments(
    config: &ObjectNativeConfig,
    state: &SearchLsmState,
    spool_directory: &std::path::Path,
) -> Result<Vec<TextArtifact>> {
    let mut artifacts = Vec::with_capacity(config.delta_segments + 1);
    for segment_no in 0..=config.delta_segments {
        let mut bytes = [0u8; 16];
        bytes[0] = 0x74;
        bytes[8..].copy_from_slice(&(segment_no as u64 + 1).to_be_bytes());
        let context = TextV4BuildContext {
            sst_id: Uuid::from_bytes(bytes),
            event_ranges: vec![SearchEventRange::new(
                segment_no as u64,
                segment_no as u64 + 1,
            )],
            complete_filter_properties: vec![ACTIVE_FILTER.to_owned(), TENANT_FILTER.to_owned()],
        };
        let build_config = TextV4ExternalBuildConfig {
            memory_budget_bytes: config.build_memory_bytes,
            spool_directory: Some(spool_directory.to_path_buf()),
            ..TextV4ExternalBuildConfig::default()
        };
        let mut builder = if segment_no == 0 {
            TextV4ExternalBuilder::with_config_base(state, context, build_config)
        } else {
            TextV4ExternalBuilder::with_config(state, context, build_config)
        }
        .map_err(|error| anyhow!("create FT4 segment {segment_no}: {error}"))?;
        for row in 0..config.documents {
            let selected = if segment_no == 0 {
                true
            } else {
                mutation_kind(row, segment_no, config.delta_segments).is_some()
            };
            if !selected {
                continue;
            }
            let before = if segment_no == 0 {
                None
            } else {
                text_payload(config, row, None)
            };
            let after = if segment_no == 0 {
                text_payload(config, row, None)
            } else {
                text_payload(
                    config,
                    row,
                    mutation_kind(row, segment_no, config.delta_segments)
                        .map(|kind| (segment_no, kind)),
                )
            };
            builder
                .push(TextV4Mutation {
                    node_id: node_id(row),
                    lsn: segment_no as u64 + 1,
                    before,
                    after,
                })
                .map_err(|error| anyhow!("append FT4 segment {segment_no} row {row}: {error}"))?;
        }
        let artifact = builder
            .finish()
            .map_err(|error| anyhow!("finish FT4 segment {segment_no}: {error}"))?
            .ok_or_else(|| anyhow!("FT4 segment {segment_no} unexpectedly empty"))?;
        artifacts.push(TextArtifact {
            file: artifact.file,
            len: artifact.len,
            segment: artifact.output.segment,
            metrics: artifact.metrics,
        });
    }
    Ok(artifacts)
}

async fn open_vector_readers(
    config: &ObjectNativeConfig,
    state: &SearchLsmState,
    descriptor: &namidb_storage::manifest::VectorIndexDescriptor,
    artifacts: &[VectorArtifact],
    cache_bytes: usize,
) -> Result<VectorReaderSet> {
    let per_source_cache = if cache_bytes == 0 {
        0
    } else {
        (cache_bytes / artifacts.len().max(1)).max(1)
    };
    let mut readers = Vec::with_capacity(artifacts.len());
    let mut sources = Vec::with_capacity(artifacts.len());
    for (segment_no, artifact) in artifacts.iter().enumerate() {
        let source = Arc::new(InstrumentedRangeSource::new(
            artifact
                .file
                .try_clone()
                .with_context(|| format!("clone VG6 segment {segment_no}"))?,
            artifact.len,
            per_source_cache,
            config.range_latency_ms,
            config.gates.max_in_flight as usize,
        ));
        let reader = match artifact.kind {
            VectorArtifactKind::V5Base => {
                let range_source: Arc<dyn VectorV5RangeSource> = source.clone();
                let reader = VectorV5Reader::open(range_source, artifact.len)
                    .await
                    .map_err(|error| anyhow!("open V5 base segment {segment_no}: {error}"))?;
                if reader.point_count() != artifact.segment.live_payload_count
                    || reader.dim() != descriptor.dim
                    || reader.metric() != descriptor.metric
                    || reader.page_count() as u64 != artifact.page_count
                    || artifact.segment.role != SearchSegmentRole::Base
                    || artifact.segment.format != SearchSegmentFormat::VectorV5Base
                    || artifact.segment.suppress_count != 0
                    || !matches!(
                        artifact.segment.stats,
                        SearchSegmentStats::Vector {
                            live_count: SearchStatValue::Absolute(count)
                        } if count == reader.point_count()
                    )
                    || artifact
                        .segment
                        .complete_filter_properties
                        .iter()
                        .any(|property| !reader.supports_filter_property(property))
                {
                    bail!("V5 base native footer disagrees with its benchmark segment contract");
                }
                VectorSegmentReader::V5Base(reader)
            }
            VectorArtifactKind::V6Delta => {
                let range_source: Arc<dyn SearchVersionRangeSource> = source.clone();
                let reader = VectorV6Reader::open(
                    range_source,
                    artifact.len,
                    state,
                    &artifact.segment,
                    descriptor,
                )
                .await
                .map_err(|error| anyhow!("open VG6 delta segment {segment_no}: {error}"))?;
                if reader.page_count() as u64 != artifact.page_count
                    || reader.segment().role != SearchSegmentRole::Delta
                    || reader.segment().format != SearchSegmentFormat::VectorV6
                    || !matches!(
                        reader.segment().stats,
                        SearchSegmentStats::Vector {
                            live_count: SearchStatValue::Delta(_)
                        }
                    )
                {
                    bail!("VG6 native footer disagrees with its benchmark delta contract");
                }
                VectorSegmentReader::V6Delta(reader)
            }
        };
        readers.push(reader);
        sources.push(source);
    }
    let open_io = aggregate_source_stats(&sources);
    let metadata_bytes = readers
        .iter()
        .map(VectorSegmentReader::resident_metadata_bytes)
        .sum();
    Ok(VectorReaderSet {
        readers,
        sources,
        open_io,
        metadata_bytes,
    })
}

async fn open_text_readers(
    config: &ObjectNativeConfig,
    state: &SearchLsmState,
    artifacts: &[TextArtifact],
    cache_bytes: usize,
) -> Result<TextReaderSet> {
    let per_source_cache = if cache_bytes == 0 {
        0
    } else {
        (cache_bytes / artifacts.len().max(1)).max(1)
    };
    let mut readers = Vec::with_capacity(artifacts.len());
    let mut sources = Vec::with_capacity(artifacts.len());
    for (segment_no, artifact) in artifacts.iter().enumerate() {
        let source = Arc::new(InstrumentedRangeSource::new(
            artifact
                .file
                .try_clone()
                .with_context(|| format!("clone FT4 segment {segment_no}"))?,
            artifact.len,
            per_source_cache,
            config.range_latency_ms,
            config.gates.max_in_flight as usize,
        ));
        let range_source: Arc<dyn SearchVersionRangeSource> = source.clone();
        let reader = TextV4Reader::open(range_source, artifact.len, state, &artifact.segment)
            .await
            .map_err(|error| anyhow!("open FT4 segment {segment_no}: {error}"))?;
        readers.push(reader);
        sources.push(source);
    }
    let open_io = aggregate_source_stats(&sources);
    let metadata_bytes = readers
        .iter()
        .map(TextV4Reader::resident_metadata_bytes)
        .sum();
    Ok(TextReaderSet {
        readers,
        sources,
        open_io,
        metadata_bytes,
    })
}

fn aggregate_source_stats(sources: &[Arc<InstrumentedRangeSource>]) -> RangeIoStats {
    sources
        .iter()
        .map(|source| source.stats())
        .fold(RangeIoStats::default(), |mut total, stats| {
            total.logical_requests = total
                .logical_requests
                .saturating_add(stats.logical_requests);
            total.logical_bytes = total.logical_bytes.saturating_add(stats.logical_bytes);
            total.fetch_requests = total.fetch_requests.saturating_add(stats.fetch_requests);
            total.fetched_bytes = total.fetched_bytes.saturating_add(stats.fetched_bytes);
            total.cache_hits = total.cache_hits.saturating_add(stats.cache_hits);
            total.peak_in_flight = total.peak_in_flight.max(stats.peak_in_flight);
            total
        })
}

fn prepare_sources(sources: &[Arc<InstrumentedRangeSource>], clear_cache: bool) {
    for source in sources {
        if clear_cache {
            source.clear_cache();
        }
        source.reset_stats();
    }
}

async fn run_vector_lsm(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    state: &SearchLsmState,
    descriptor: &namidb_storage::manifest::VectorIndexDescriptor,
    artifacts: &[VectorArtifact],
) -> Result<(VersionedSearchFamilyReport, VectorExactShadowReport)> {
    let artifact_bytes = artifacts.iter().map(|artifact| artifact.len).sum::<u64>();
    let no_cache = open_vector_readers(config, state, descriptor, artifacts, 0).await?;
    let sized =
        open_vector_readers(config, state, descriptor, artifacts, config.cache_bytes).await?;
    let queries = (0..config.queries)
        .map(|query| query_vector(config, centroids, VectorMetric::Cosine, query))
        .collect::<Vec<_>>();
    let exact_unfiltered = exact_final_vectors(config, centroids, &queries, false);
    let exact_filtered = exact_final_vectors(config, centroids, &queries, true);
    let exact_shadow =
        run_vector_exact_shadow(&no_cache, &queries, &exact_unfiltered, &exact_filtered).await?;
    let unfiltered = measure_vector_track(
        config,
        centroids,
        &no_cache,
        &sized,
        artifact_bytes,
        &queries,
        &exact_unfiltered,
        &[],
    )
    .await?;
    let active_filter = measure_vector_track(
        config,
        centroids,
        &no_cache,
        &sized,
        artifact_bytes,
        &queries,
        &exact_filtered,
        &[(
            ACTIVE_FILTER.to_owned(),
            vec![SearchFilterValue::Bool(true)],
        )],
    )
    .await?;
    let serving = VersionedSearchFamilyReport {
        format: "NAMIVG05 Base + NAMIVG06 Delta",
        artifact_bytes,
        resident_metadata_bytes: sized.metadata_bytes,
        resident_metadata_artifact_ratio: sized.metadata_bytes as f64
            / artifact_bytes.max(1) as f64,
        reader_open_io_no_cache: no_cache.open_io,
        reader_open_io_sized_cache: sized.open_io,
        builder_peak_logical_bytes: artifacts
            .iter()
            .map(|artifact| artifact.builder_peak_logical_bytes)
            .max()
            .unwrap_or(0),
        unfiltered,
        active_filter,
    };
    Ok((serving, exact_shadow))
}

async fn run_vector_exact_shadow(
    readers: &VectorReaderSet,
    queries: &[Vec<f32>],
    exact_unfiltered: &[Vec<([u8; 16], f32)>],
    exact_filtered: &[Vec<([u8; 16], f32)>],
) -> Result<VectorExactShadowReport> {
    let mut unfiltered = WinnerParityReport {
        node_ids_exact: true,
        scores_exact: true,
        ..WinnerParityReport::default()
    };
    let mut active_filter = WinnerParityReport {
        node_ids_exact: true,
        scores_exact: true,
        ..WinnerParityReport::default()
    };
    let mut active_filter_applied = true;
    for ((query, expected_unfiltered), expected_filtered) in
        queries.iter().zip(exact_unfiltered).zip(exact_filtered)
    {
        prepare_sources(&readers.sources, false);
        let actual = coordinated_vector_query(
            &readers.readers,
            query,
            expected_unfiltered.len(),
            &[],
            V5BaseQueryMode::ExactShadow,
        )
        .await?;
        observe_vector_parity(&mut unfiltered, expected_unfiltered, &actual.hits);

        prepare_sources(&readers.sources, false);
        let actual = coordinated_vector_query(
            &readers.readers,
            query,
            expected_filtered.len(),
            &[(
                ACTIVE_FILTER.to_owned(),
                vec![SearchFilterValue::Bool(true)],
            )],
            V5BaseQueryMode::ExactShadow,
        )
        .await?;
        active_filter_applied &= actual.native_filter_applied;
        observe_vector_parity(&mut active_filter, expected_filtered, &actual.hits);
    }
    Ok(VectorExactShadowReport {
        query_count: queries.len(),
        unfiltered,
        active_filter,
        active_filter_applied,
        io_accounting:
            "correctness-only exact shadow; range/latency counters excluded from serving ANN I/O and latency SLOs",
    })
}

fn exact_final_vectors(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    queries: &[Vec<f32>],
    active_only: bool,
) -> Vec<Vec<([u8; 16], f32)>> {
    let mut best = (0..queries.len())
        .map(|_| Vec::with_capacity(config.k))
        .collect::<Vec<_>>();
    for row in 0..config.vectors {
        let mutation = row_mutation(row, config.delta_segments);
        let Some(payload) = vector_payload(config, centroids, row, mutation) else {
            continue;
        };
        if active_only && payload.filters.get(ACTIVE_FILTER) != Some(&SearchFilterValue::Bool(true))
        {
            continue;
        }
        for (query_no, query) in queries.iter().enumerate() {
            let score = vector_cosine_v6(&payload.vector, query);
            retain_vector_top_k(&mut best[query_no], (node_id(row), score), config.k);
        }
    }
    for values in &mut best {
        values.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
    }
    best
}

fn vector_cosine_v6(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum::<f64>();
    let left_norm = left
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm * right_norm)) as f32
    }
}

fn retain_vector_top_k(best: &mut Vec<([u8; 16], f32)>, candidate: ([u8; 16], f32), k: usize) {
    best.push(candidate);
    best.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    best.truncate(k);
}

struct VectorCoordinatorResult {
    hits: Vec<([u8; 16], f32)>,
    native_filter_applied: bool,
    fanout: usize,
}

#[derive(Clone)]
struct VersionedVectorCandidate {
    segment: usize,
    node_id: [u8; 16],
    score: f32,
    version: VectorCandidateVersion,
}

#[derive(Clone, Copy)]
enum VectorCandidateVersion {
    V5Base,
    V6Delta { lsn: u64, payload_fingerprint: u64 },
}

#[derive(Clone, Copy)]
enum V5BaseQueryMode {
    ServingAnn(VectorV5SearchOptions),
    ExactShadow,
}

fn v5_query_filter_groups(
    groups: &[(String, Vec<SearchFilterValue>)],
) -> Result<Vec<(String, Vec<Value>)>> {
    groups
        .iter()
        .map(|(property, alternatives)| {
            let values = alternatives
                .iter()
                .map(|value| match value {
                    SearchFilterValue::Bool(value) => Ok(Value::Bool(*value)),
                    SearchFilterValue::String(value) => Ok(Value::Str(value.clone())),
                    _ => bail!(
                        "V5 benchmark base advertises only Bool/String native filter properties"
                    ),
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((property.clone(), values))
        })
        .collect()
}

async fn coordinated_vector_query(
    readers: &[VectorSegmentReader],
    query: &[f32],
    k: usize,
    groups: &[(String, Vec<SearchFilterValue>)],
    base_mode: V5BaseQueryMode,
) -> Result<VectorCoordinatorResult> {
    if k == 0 {
        return Ok(VectorCoordinatorResult {
            hits: Vec::new(),
            native_filter_applied: groups.is_empty(),
            fanout: readers.len(),
        });
    }
    let local_live = readers
        .iter()
        .map(|reader| match reader {
            VectorSegmentReader::V5Base(reader) => {
                usize::try_from(reader.point_count()).unwrap_or(usize::MAX)
            }
            VectorSegmentReader::V6Delta(reader) => reader.segment().live_payload_count as usize,
        })
        .collect::<Vec<_>>();
    let mut limits = local_live
        .iter()
        .map(|live| (*live).min(k).max(usize::from(*live > 0)))
        .collect::<Vec<_>>();
    loop {
        let mut candidates = Vec::new();
        let mut statuses = Vec::with_capacity(readers.len());
        let mut native_filter_applied = true;
        for (segment, reader) in readers.iter().enumerate() {
            if limits[segment] == 0 {
                statuses.push((0usize, true));
                continue;
            }
            match reader {
                VectorSegmentReader::V5Base(reader) => {
                    let value_groups = v5_query_filter_groups(groups)?;
                    let result = match base_mode {
                        V5BaseQueryMode::ServingAnn(options) => {
                            reader
                                .search_filter_groups(
                                    query,
                                    limits[segment],
                                    options,
                                    &value_groups,
                                )
                                .await
                        }
                        V5BaseQueryMode::ExactShadow => {
                            reader
                                .search_exact_filter_groups(query, limits[segment], &value_groups)
                                .await
                        }
                    }
                    .map_err(|error| {
                        anyhow!("V5 coordinated base query segment {segment}: {error}")
                    })?;
                    native_filter_applied &= result.applied_filter_groups == groups.len();
                    let exhausted = result.eligible_rows_seen <= limits[segment];
                    candidates.extend(result.hits.into_iter().map(|(node_id, score)| {
                        VersionedVectorCandidate {
                            segment,
                            node_id,
                            score,
                            version: VectorCandidateVersion::V5Base,
                        }
                    }));
                    statuses.push((0usize, exhausted));
                }
                VectorSegmentReader::V6Delta(reader) => {
                    let result = reader
                        .search_exact(query, limits[segment], groups)
                        .await
                        .map_err(|error| {
                            anyhow!("VG6 coordinated delta query segment {segment}: {error}")
                        })?;
                    native_filter_applied &= result.applied_filter_groups == groups.len();
                    let exhausted = result.eligible_rows_seen <= limits[segment];
                    candidates.extend(result.hits.into_iter().map(|hit| {
                        VersionedVectorCandidate {
                            segment,
                            node_id: hit.node_id,
                            score: hit.score,
                            version: VectorCandidateVersion::V6Delta {
                                lsn: hit.lsn,
                                payload_fingerprint: hit.payload_fingerprint,
                            },
                        }
                    }));
                    statuses.push((0usize, exhausted));
                }
            }
        }
        let ids = candidates
            .iter()
            .map(|candidate| candidate.node_id)
            .collect::<Vec<_>>();
        let mut versions_by_segment = Vec::with_capacity(readers.len());
        for reader in readers {
            if let Some(version_reader) = reader.version_reader() {
                versions_by_segment.push(
                    version_reader
                        .point_probe_many(&ids)
                        .await
                        .map_err(|error| anyhow!("VG6 winner-table batch probe: {error}"))?,
                );
            }
        }
        let mut winners = BTreeMap::<[u8; 16], ([u8; 16], f32)>::new();
        for (position, candidate) in candidates.into_iter().enumerate() {
            let record = reconcile_node_versions(
                versions_by_segment
                    .iter()
                    .filter_map(|versions| versions[position]),
            )
            .map_err(|error| anyhow!("VG6 winner reconciliation: {error}"))?;
            let winning = match candidate.version {
                VectorCandidateVersion::V5Base => record.is_none(),
                VectorCandidateVersion::V6Delta {
                    lsn,
                    payload_fingerprint,
                } => record.is_some_and(|record| {
                    record.node_id == candidate.node_id
                        && record.lsn == lsn
                        && record.payload_fingerprint == payload_fingerprint
                        && matches!(record.operation, SearchVersionOperation::Live { .. })
                }),
            };
            if winning {
                statuses[candidate.segment].0 += 1;
                winners.insert(candidate.node_id, (candidate.node_id, candidate.score));
            }
        }
        let mut hits = winners.into_values().collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        hits.truncate(k);
        let safe_top_k = statuses
            .iter()
            .all(|(accepted, exhausted)| *accepted >= k || *exhausted);
        if statuses.iter().all(|(_, exhausted)| *exhausted) || (hits.len() >= k && safe_top_k) {
            return Ok(VectorCoordinatorResult {
                hits,
                native_filter_applied,
                fanout: readers.len(),
            });
        }
        let globally_short = hits.len() < k;
        let mut changed = false;
        for (segment, (accepted, exhausted)) in statuses.into_iter().enumerate() {
            if exhausted || (!globally_short && accepted >= k) {
                continue;
            }
            let next = limits[segment]
                .saturating_mul(2)
                .max(limits[segment].saturating_add(k.saturating_sub(accepted).max(1)))
                .min(local_live[segment]);
            if next > limits[segment] {
                limits[segment] = next;
                changed = true;
            }
        }
        if !changed {
            bail!("VG6 coordinator could not prove an exact top-k after widening");
        }
    }
}

async fn measure_vector_track(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    no_cache: &VectorReaderSet,
    sized: &VectorReaderSet,
    artifact_bytes: u64,
    queries: &[Vec<f32>],
    exact: &[Vec<([u8; 16], f32)>],
    groups: &[(String, Vec<SearchFilterValue>)],
) -> Result<VersionedQueryTrack> {
    let mut no_cache_us = Vec::with_capacity(queries.len());
    let mut cold_us = Vec::with_capacity(queries.len());
    let mut warm_us = Vec::with_capacity(queries.len());
    let mut no_cache_io = RangeIoAggregate::default();
    let mut cold_io = RangeIoAggregate::default();
    let mut warm_io = RangeIoAggregate::default();
    let mut parity = WinnerParityReport {
        node_ids_exact: true,
        scores_exact: true,
        ..WinnerParityReport::default()
    };
    let mut native_filter_applied = true;
    let mut max_fanout = 0usize;
    let mut recall_sum = 0.0f64;
    let mut returned_winners_valid = true;
    let mut returned_scores_exact = true;
    let mut max_returned_score_abs_error = 0.0f64;
    let mut cache_results_identical = true;
    let serving = V5BaseQueryMode::ServingAnn(VectorV5SearchOptions {
        nprobe: config.nprobe,
        max_nprobe: config.max_nprobe,
        rerank_factor: config.rerank_factor,
    });

    for (query, expected) in queries.iter().zip(exact) {
        prepare_sources(&no_cache.sources, false);
        let started = Instant::now();
        let uncached =
            coordinated_vector_query(&no_cache.readers, query, expected.len(), groups, serving)
                .await?;
        no_cache_us.push(elapsed_us(started));
        no_cache_io.observe(aggregate_source_stats(&no_cache.sources), artifact_bytes);

        prepare_sources(&sized.sources, true);
        let started = Instant::now();
        let cold = coordinated_vector_query(&sized.readers, query, expected.len(), groups, serving)
            .await?;
        cold_us.push(elapsed_us(started));
        cold_io.observe(aggregate_source_stats(&sized.sources), artifact_bytes);

        prepare_sources(&sized.sources, false);
        let started = Instant::now();
        let warm = coordinated_vector_query(&sized.readers, query, expected.len(), groups, serving)
            .await?;
        warm_us.push(elapsed_us(started));
        warm_io.observe(aggregate_source_stats(&sized.sources), artifact_bytes);

        observe_vector_parity(&mut parity, expected, &uncached.hits);
        recall_sum += vector_recall_at_k(expected, &uncached.hits);
        let returned =
            validate_serving_vector_hits(config, centroids, query, groups, &uncached.hits);
        returned_winners_valid &= returned.0;
        returned_scores_exact &= returned.1;
        max_returned_score_abs_error = max_returned_score_abs_error.max(returned.2);
        let cold_equal = same_vector_hits(&uncached.hits, &cold.hits);
        let warm_equal = same_vector_hits(&uncached.hits, &warm.hits);
        cache_results_identical &= cold_equal.0 && cold_equal.1 && warm_equal.0 && warm_equal.1;
        parity.node_ids_exact &= cold_equal.0 && warm_equal.0;
        parity.scores_exact &= cold_equal.1 && warm_equal.1;
        parity.max_score_abs_error = parity
            .max_score_abs_error
            .max(cold_equal.2)
            .max(warm_equal.2);
        native_filter_applied &= uncached.native_filter_applied
            && cold.native_filter_applied
            && warm.native_filter_applied;
        max_fanout = max_fanout
            .max(uncached.fanout)
            .max(cold.fanout)
            .max(warm.fanout);
    }
    no_cache_us.sort_unstable();
    cold_us.sort_unstable();
    warm_us.sort_unstable();
    no_cache_io.finish(artifact_bytes);
    cold_io.finish(artifact_bytes);
    warm_io.finish(artifact_bytes);
    Ok(VersionedQueryTrack {
        result_model: "serving_ann_vs_exact_oracle",
        query_count: queries.len(),
        native_filter_applied,
        recall_at_k: recall_sum / queries.len().max(1) as f64,
        returned_winners_valid,
        returned_scores_exact,
        max_returned_score_abs_error,
        cache_results_identical,
        parity,
        no_cache: QueryPassReport {
            cache: "zero",
            p50_us: percentile(&no_cache_us, 50.0),
            p95_us: percentile(&no_cache_us, 95.0),
            io: no_cache_io,
        },
        sized_cache_cold: QueryPassReport {
            cache: "sized_cold",
            p50_us: percentile(&cold_us, 50.0),
            p95_us: percentile(&cold_us, 95.0),
            io: cold_io,
        },
        sized_cache_warm: QueryPassReport {
            cache: "sized_warm",
            p50_us: percentile(&warm_us, 50.0),
            p95_us: percentile(&warm_us, 95.0),
            io: warm_io,
        },
        max_fanout,
    })
}

fn validate_serving_vector_hits(
    config: &ObjectNativeConfig,
    centroids: &[Vec<f32>],
    query: &[f32],
    groups: &[(String, Vec<SearchFilterValue>)],
    hits: &[([u8; 16], f32)],
) -> (bool, bool, f64) {
    let active_required = groups.iter().any(|(property, alternatives)| {
        property == ACTIVE_FILTER && alternatives.contains(&SearchFilterValue::Bool(true))
    });
    let mut winners_valid = true;
    let mut scores_exact = true;
    let mut max_error = 0.0f64;
    for (node_id, actual_score) in hits {
        let ordinal = u128::from_be_bytes(*node_id)
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok());
        let Some(row) = ordinal.filter(|row| *row < config.vectors) else {
            winners_valid = false;
            scores_exact = false;
            continue;
        };
        let Some(payload) = vector_payload(
            config,
            centroids,
            row,
            row_mutation(row, config.delta_segments),
        ) else {
            winners_valid = false;
            continue;
        };
        if active_required
            && payload.filters.get(ACTIVE_FILTER) != Some(&SearchFilterValue::Bool(true))
        {
            winners_valid = false;
        }
        let expected_score = vector_cosine_v6(&payload.vector, query);
        let error = f64::from((expected_score - *actual_score).abs());
        max_error = max_error.max(error);
        scores_exact &= expected_score.to_bits() == actual_score.to_bits();
    }
    (winners_valid, scores_exact, max_error)
}

fn vector_recall_at_k(expected: &[([u8; 16], f32)], actual: &[([u8; 16], f32)]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let expected_ids = expected
        .iter()
        .map(|(node_id, _)| *node_id)
        .collect::<BTreeSet<_>>();
    actual
        .iter()
        .filter(|(node_id, _)| expected_ids.contains(node_id))
        .count() as f64
        / expected.len() as f64
}

fn observe_vector_parity(
    parity: &mut WinnerParityReport,
    expected: &[([u8; 16], f32)],
    actual: &[([u8; 16], f32)],
) {
    parity.expected_hits += expected.len();
    parity.actual_hits += actual.len();
    let comparison = same_vector_hits(expected, actual);
    parity.node_ids_exact &= comparison.0;
    parity.scores_exact &= comparison.1;
    parity.max_score_abs_error = parity.max_score_abs_error.max(comparison.2);
}

fn same_vector_hits(expected: &[([u8; 16], f32)], actual: &[([u8; 16], f32)]) -> (bool, bool, f64) {
    let ids = expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(left, right)| left.0 == right.0);
    let mut max_error = 0.0f64;
    let scores = expected.len() == actual.len()
        && expected.iter().zip(actual).all(|(left, right)| {
            max_error = max_error.max(f64::from((left.1 - right.1).abs()));
            left.1.to_bits() == right.1.to_bits()
        });
    (ids, scores, max_error)
}

async fn run_text_lsm(
    config: &ObjectNativeConfig,
    state: &SearchLsmState,
    artifacts: &[TextArtifact],
) -> Result<VersionedSearchFamilyReport> {
    let artifact_bytes = artifacts.iter().map(|artifact| artifact.len).sum::<u64>();
    let no_cache = open_text_readers(config, state, artifacts, 0).await?;
    let sized = open_text_readers(config, state, artifacts, config.cache_bytes).await?;
    let query = parse_query("reforma articulo");
    let (global, exact_unfiltered) = exact_final_text(config, &query, false);
    let (_, exact_filtered) = exact_final_text(config, &query, true);
    let queries = vec![query; config.queries];
    let exact_unfiltered = vec![exact_unfiltered; config.queries];
    let exact_filtered = vec![exact_filtered; config.queries];
    let unfiltered = measure_text_track(
        &no_cache,
        &sized,
        artifact_bytes,
        &queries,
        &global,
        &exact_unfiltered,
        &[],
    )
    .await?;
    let active_filter = measure_text_track(
        &no_cache,
        &sized,
        artifact_bytes,
        &queries,
        &global,
        &exact_filtered,
        &[(
            ACTIVE_FILTER.to_owned(),
            vec![SearchFilterValue::Bool(true)],
        )],
    )
    .await?;
    Ok(VersionedSearchFamilyReport {
        format: "NAMIFT04",
        artifact_bytes,
        resident_metadata_bytes: sized.metadata_bytes,
        resident_metadata_artifact_ratio: sized.metadata_bytes as f64
            / artifact_bytes.max(1) as f64,
        reader_open_io_no_cache: no_cache.open_io,
        reader_open_io_sized_cache: sized.open_io,
        builder_peak_logical_bytes: artifacts
            .iter()
            .map(|artifact| artifact.metrics.peak_logical_memory_bytes)
            .max()
            .unwrap_or(0),
        unfiltered,
        active_filter,
    })
}

fn exact_final_text(
    config: &ObjectNativeConfig,
    query: &TextQuery,
    active_only: bool,
) -> (TextV4GlobalStats, Vec<([u8; 16], f64)>) {
    let terms = query
        .base_terms()
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let mut document_frequency = terms
        .iter()
        .map(|term| (term.clone(), 0u64))
        .collect::<BTreeMap<_, _>>();
    let mut document_count = 0u64;
    let mut total_document_len = 0u64;
    for row in 0..config.documents {
        let Some(payload) = text_payload(config, row, row_mutation(row, config.delta_segments))
        else {
            continue;
        };
        let tokens = tokenize(&payload.text);
        document_count += 1;
        total_document_len = total_document_len.saturating_add(tokens.len() as u64);
        let present = tokens.iter().map(String::as_str).collect::<BTreeSet<_>>();
        for term in &terms {
            if present.contains(term.as_str()) {
                *document_frequency
                    .get_mut(term)
                    .expect("query term initialized") += 1;
            }
        }
    }
    let global = TextV4GlobalStats {
        document_count,
        total_document_len,
        document_frequency,
    };
    let average = avg_len(total_document_len, document_count as usize);
    let mut best = Vec::with_capacity(config.k);
    for row in 0..config.documents {
        let Some(payload) = text_payload(config, row, row_mutation(row, config.delta_segments))
        else {
            continue;
        };
        if active_only && payload.filters.get(ACTIVE_FILTER) != Some(&SearchFilterValue::Bool(true))
        {
            continue;
        }
        let tokens = tokenize(&payload.text);
        let mut frequencies = BTreeMap::<&str, u32>::new();
        for token in &tokens {
            *frequencies.entry(token).or_default() += 1;
        }
        let mut matched = false;
        let mut score = 0.0f64;
        for term in &terms {
            let Some(tf) = frequencies.get(term.as_str()) else {
                continue;
            };
            matched = true;
            let idf = bm25_idf(
                document_count as usize,
                global.document_frequency[term] as usize,
            );
            score += bm25_term_score(idf, *tf, tokens.len(), average);
        }
        if matched {
            retain_text_result(&mut best, (node_id(row), score), config.k);
        }
    }
    best.sort_by(compare_versioned_text_hits);
    (global, best)
}

fn retain_text_result(best: &mut Vec<([u8; 16], f64)>, candidate: ([u8; 16], f64), k: usize) {
    best.push(candidate);
    best.sort_by(compare_versioned_text_hits);
    best.truncate(k);
}

fn compare_versioned_text_hits(left: &([u8; 16], f64), right: &([u8; 16], f64)) -> Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}

struct TextCoordinatorResult {
    hits: Vec<([u8; 16], f64)>,
    native_filter_applied: bool,
    fanout: usize,
}

#[derive(Clone)]
struct VersionedTextCandidate {
    segment: usize,
    hit: TextV4Hit,
}

async fn coordinated_text_query(
    readers: &[TextV4Reader],
    query: &TextQuery,
    global: &TextV4GlobalStats,
    k: usize,
    groups: &[(String, Vec<SearchFilterValue>)],
) -> Result<TextCoordinatorResult> {
    if k == 0 {
        return Ok(TextCoordinatorResult {
            hits: Vec::new(),
            native_filter_applied: groups.is_empty(),
            fanout: readers.len(),
        });
    }
    let local_live = readers
        .iter()
        .map(|reader| reader.live_document_count() as usize)
        .collect::<Vec<_>>();
    let mut limits = local_live
        .iter()
        .map(|live| (*live).min(k).max(usize::from(*live > 0)))
        .collect::<Vec<_>>();
    loop {
        let mut candidates = Vec::new();
        let mut statuses = Vec::with_capacity(readers.len());
        let mut native_filter_applied = true;
        for (segment, reader) in readers.iter().enumerate() {
            if limits[segment] == 0 {
                statuses.push((0usize, true));
                continue;
            }
            let result = reader
                .search_query_exact(query, global, limits[segment], groups)
                .await
                .map_err(|error| anyhow!("FT4 coordinated query segment {segment}: {error}"))?;
            native_filter_applied &= result.applied_filter_groups == groups.len()
                || (result.hits.is_empty()
                    && groups
                        .iter()
                        .all(|(property, _)| reader.supports_filter_property(property)));
            let exhausted =
                local_live[segment] <= limits[segment] || result.hits.len() < limits[segment];
            candidates.extend(
                result
                    .hits
                    .into_iter()
                    .map(|hit| VersionedTextCandidate { segment, hit }),
            );
            statuses.push((0usize, exhausted));
        }
        let ids = candidates
            .iter()
            .map(|candidate| candidate.hit.node_id)
            .collect::<Vec<_>>();
        let mut versions_by_segment = Vec::with_capacity(readers.len());
        for reader in readers {
            versions_by_segment.push(
                reader
                    .version_reader()
                    .point_probe_many(&ids)
                    .await
                    .map_err(|error| anyhow!("FT4 winner-table batch probe: {error}"))?,
            );
        }
        let mut winners = BTreeMap::<[u8; 16], ([u8; 16], f64)>::new();
        for (position, candidate) in candidates.into_iter().enumerate() {
            let record = reconcile_node_versions(
                versions_by_segment
                    .iter()
                    .filter_map(|versions| versions[position]),
            )
            .map_err(|error| anyhow!("FT4 winner reconciliation: {error}"))?;
            let winning = record.is_some_and(|record| {
                record.node_id == candidate.hit.node_id
                    && record.lsn == candidate.hit.lsn
                    && record.payload_fingerprint == candidate.hit.payload_fingerprint
                    && matches!(record.operation, SearchVersionOperation::Live { .. })
            });
            if winning {
                statuses[candidate.segment].0 += 1;
                winners.insert(
                    candidate.hit.node_id,
                    (candidate.hit.node_id, candidate.hit.score),
                );
            }
        }
        let mut hits = winners.into_values().collect::<Vec<_>>();
        hits.sort_by(compare_versioned_text_hits);
        hits.truncate(k);
        let safe_top_k = statuses
            .iter()
            .all(|(accepted, exhausted)| *accepted >= k || *exhausted);
        if statuses.iter().all(|(_, exhausted)| *exhausted) || (hits.len() >= k && safe_top_k) {
            return Ok(TextCoordinatorResult {
                hits,
                native_filter_applied,
                fanout: readers.len(),
            });
        }
        let globally_short = hits.len() < k;
        let mut changed = false;
        for (segment, (accepted, exhausted)) in statuses.into_iter().enumerate() {
            if exhausted || (!globally_short && accepted >= k) {
                continue;
            }
            let next = limits[segment]
                .saturating_mul(2)
                .max(limits[segment].saturating_add(k.saturating_sub(accepted).max(1)))
                .min(local_live[segment]);
            if next > limits[segment] {
                limits[segment] = next;
                changed = true;
            }
        }
        if !changed {
            bail!("FT4 coordinator could not prove an exact top-k after widening");
        }
    }
}

async fn measure_text_track(
    no_cache: &TextReaderSet,
    sized: &TextReaderSet,
    artifact_bytes: u64,
    queries: &[TextQuery],
    global: &TextV4GlobalStats,
    exact: &[Vec<([u8; 16], f64)>],
    groups: &[(String, Vec<SearchFilterValue>)],
) -> Result<VersionedQueryTrack> {
    let mut no_cache_us = Vec::with_capacity(queries.len());
    let mut cold_us = Vec::with_capacity(queries.len());
    let mut warm_us = Vec::with_capacity(queries.len());
    let mut no_cache_io = RangeIoAggregate::default();
    let mut cold_io = RangeIoAggregate::default();
    let mut warm_io = RangeIoAggregate::default();
    let mut parity = WinnerParityReport {
        node_ids_exact: true,
        scores_exact: true,
        ..WinnerParityReport::default()
    };
    let mut native_filter_applied = true;
    let mut max_fanout = 0usize;
    let mut cache_results_identical = true;

    for (query, expected) in queries.iter().zip(exact) {
        prepare_sources(&no_cache.sources, false);
        let started = Instant::now();
        let uncached =
            coordinated_text_query(&no_cache.readers, query, global, expected.len(), groups)
                .await?;
        no_cache_us.push(elapsed_us(started));
        no_cache_io.observe(aggregate_source_stats(&no_cache.sources), artifact_bytes);

        prepare_sources(&sized.sources, true);
        let started = Instant::now();
        let cold =
            coordinated_text_query(&sized.readers, query, global, expected.len(), groups).await?;
        cold_us.push(elapsed_us(started));
        cold_io.observe(aggregate_source_stats(&sized.sources), artifact_bytes);

        prepare_sources(&sized.sources, false);
        let started = Instant::now();
        let warm =
            coordinated_text_query(&sized.readers, query, global, expected.len(), groups).await?;
        warm_us.push(elapsed_us(started));
        warm_io.observe(aggregate_source_stats(&sized.sources), artifact_bytes);

        observe_text_parity(&mut parity, expected, &uncached.hits);
        let cold_equal = compare_text_results(expected, &cold.hits);
        let warm_equal = compare_text_results(expected, &warm.hits);
        cache_results_identical &= cold_equal.0 && cold_equal.1 && warm_equal.0 && warm_equal.1;
        parity.node_ids_exact &= cold_equal.0 && warm_equal.0;
        parity.scores_exact &= cold_equal.1 && warm_equal.1;
        parity.max_score_abs_error = parity
            .max_score_abs_error
            .max(cold_equal.2)
            .max(warm_equal.2);
        native_filter_applied &= uncached.native_filter_applied
            && cold.native_filter_applied
            && warm.native_filter_applied;
        max_fanout = max_fanout
            .max(uncached.fanout)
            .max(cold.fanout)
            .max(warm.fanout);
    }
    no_cache_us.sort_unstable();
    cold_us.sort_unstable();
    warm_us.sort_unstable();
    no_cache_io.finish(artifact_bytes);
    cold_io.finish(artifact_bytes);
    warm_io.finish(artifact_bytes);
    let text_ids_exact = parity.node_ids_exact;
    let text_scores_exact = parity.scores_exact;
    let text_max_error = parity.max_score_abs_error;
    Ok(VersionedQueryTrack {
        result_model: "exact_bm25_parity",
        query_count: queries.len(),
        native_filter_applied,
        recall_at_k: if text_ids_exact { 1.0 } else { 0.0 },
        returned_winners_valid: text_ids_exact,
        returned_scores_exact: text_scores_exact,
        max_returned_score_abs_error: text_max_error,
        cache_results_identical,
        parity,
        no_cache: QueryPassReport {
            cache: "zero",
            p50_us: percentile(&no_cache_us, 50.0),
            p95_us: percentile(&no_cache_us, 95.0),
            io: no_cache_io,
        },
        sized_cache_cold: QueryPassReport {
            cache: "sized_cold",
            p50_us: percentile(&cold_us, 50.0),
            p95_us: percentile(&cold_us, 95.0),
            io: cold_io,
        },
        sized_cache_warm: QueryPassReport {
            cache: "sized_warm",
            p50_us: percentile(&warm_us, 50.0),
            p95_us: percentile(&warm_us, 95.0),
            io: warm_io,
        },
        max_fanout,
    })
}

fn observe_text_parity(
    parity: &mut WinnerParityReport,
    expected: &[([u8; 16], f64)],
    actual: &[([u8; 16], f64)],
) {
    parity.expected_hits += expected.len();
    parity.actual_hits += actual.len();
    let comparison = compare_text_results(expected, actual);
    parity.node_ids_exact &= comparison.0;
    parity.scores_exact &= comparison.1;
    parity.max_score_abs_error = parity.max_score_abs_error.max(comparison.2);
}

fn max_family_in_flight(report: &VersionedSearchFamilyReport) -> u64 {
    [
        &report.unfiltered.no_cache,
        &report.unfiltered.sized_cache_cold,
        &report.unfiltered.sized_cache_warm,
        &report.active_filter.no_cache,
        &report.active_filter.sized_cache_cold,
        &report.active_filter.sized_cache_warm,
    ]
    .into_iter()
    .map(|pass| pass.io.peak_in_flight)
    .max()
    .unwrap_or(0)
}

struct GraphArtifact {
    body: bytes::Bytes,
    canonical_edge_count: u64,
}

#[derive(Debug, Default)]
struct PhysicalIoCounters {
    fetch_requests: AtomicU64,
    fetched_bytes: AtomicU64,
    in_flight: AtomicU64,
    peak_in_flight: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default)]
struct PhysicalIoSnapshot {
    fetch_requests: u64,
    fetched_bytes: u64,
    peak_in_flight: u64,
}

#[derive(Debug)]
struct CountingObjectStore {
    inner: Arc<InMemory>,
    counters: Arc<PhysicalIoCounters>,
    latency: Duration,
}

impl CountingObjectStore {
    fn new(inner: Arc<InMemory>, latency: Duration) -> Self {
        Self {
            inner,
            counters: Arc::new(PhysicalIoCounters::default()),
            latency,
        }
    }

    fn reset(&self) {
        self.counters
            .fetch_requests
            .store(0, AtomicOrdering::SeqCst);
        self.counters.fetched_bytes.store(0, AtomicOrdering::SeqCst);
        self.counters.in_flight.store(0, AtomicOrdering::SeqCst);
        self.counters
            .peak_in_flight
            .store(0, AtomicOrdering::SeqCst);
    }

    fn snapshot(&self) -> PhysicalIoSnapshot {
        PhysicalIoSnapshot {
            fetch_requests: self.counters.fetch_requests.load(AtomicOrdering::SeqCst),
            fetched_bytes: self.counters.fetched_bytes.load(AtomicOrdering::SeqCst),
            peak_in_flight: self.counters.peak_in_flight.load(AtomicOrdering::SeqCst),
        }
    }
}

impl fmt::Display for CountingObjectStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ObjectNativeCountingStore")
    }
}

#[async_trait]
impl ObjectStore for CountingObjectStore {
    async fn put_opts(
        &self,
        location: &ObjectPath,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjectPath,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, options).await
    }

    async fn get_opts(
        &self,
        location: &ObjectPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if options.range.is_none() || options.head {
            return self.inner.get_opts(location, options).await;
        }
        self.counters
            .fetch_requests
            .fetch_add(1, AtomicOrdering::SeqCst);
        let in_flight = self
            .counters
            .in_flight
            .fetch_add(1, AtomicOrdering::SeqCst)
            .saturating_add(1);
        self.counters
            .peak_in_flight
            .fetch_max(in_flight, AtomicOrdering::SeqCst);
        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }
        let result = self.inner.get_opts(location, options).await;
        self.counters.in_flight.fetch_sub(1, AtomicOrdering::SeqCst);
        if let Ok(result) = &result {
            self.counters.fetched_bytes.fetch_add(
                result.range.end.saturating_sub(result.range.start) as u64,
                AtomicOrdering::SeqCst,
            );
        }
        result
    }

    fn list(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjectPath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjectPath,
        to: &ObjectPath,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<ObjectPath>>,
    ) -> BoxStream<'static, object_store::Result<ObjectPath>> {
        self.inner.delete_stream(locations)
    }
}

struct GraphReaderHarness {
    reader: PagedEdgeReader,
    store: Arc<CountingObjectStore>,
    cache: Option<ImmutableRangeCache>,
    open_io: GraphIoSample,
}

async fn run_graph(config: &ObjectNativeConfig) -> Result<GraphPagedReport> {
    let forward_artifact = build_graph_artifact(config, EdgeDirection::Forward)?;
    let inverse_artifact = build_graph_artifact(config, EdgeDirection::Inverse)?;
    if forward_artifact.canonical_edge_count != inverse_artifact.canonical_edge_count {
        bail!("forward/inverse graph fixtures disagree on canonical edge count");
    }
    let forward_bytes = forward_artifact.body.len() as u64;
    let inverse_bytes = inverse_artifact.body.len() as u64;
    let backing = Arc::new(InMemory::new());
    let forward_path = ObjectPath::from("object-native-v2/forward.edge");
    let inverse_path = ObjectPath::from("object-native-v2/inverse.edge");
    backing
        .put(
            &forward_path,
            PutPayload::from(forward_artifact.body.clone()),
        )
        .await
        .context("put forward paged graph fixture")?;
    backing
        .put(
            &inverse_path,
            PutPayload::from(inverse_artifact.body.clone()),
        )
        .await
        .context("put inverse paged graph fixture")?;
    let forward_meta = backing
        .head(&forward_path)
        .await
        .context("HEAD forward paged graph fixture")?;
    let inverse_meta = backing
        .head(&inverse_path)
        .await
        .context("HEAD inverse paged graph fixture")?;
    let graph_cache = open_graph_cache(config.cache_bytes, "object-native-v2-graph")
        .await
        .context("open shared graph range cache")?;
    let forward_no_cache = open_graph_reader(
        backing.clone(),
        forward_meta.clone(),
        None,
        forward_bytes,
        config.range_latency_ms,
    )
    .await
    .context("open forward paged graph fixture without cache")?;
    let forward_sized_cache = open_graph_reader(
        backing.clone(),
        forward_meta,
        Some(graph_cache.clone()),
        forward_bytes,
        config.range_latency_ms,
    )
    .await
    .context("open forward paged graph fixture with sized cache")?;
    let inverse_no_cache = open_graph_reader(
        backing.clone(),
        inverse_meta.clone(),
        None,
        inverse_bytes,
        config.range_latency_ms,
    )
    .await
    .context("open inverse paged graph fixture without cache")?;
    let inverse_sized_cache = open_graph_reader(
        backing,
        inverse_meta,
        Some(graph_cache),
        inverse_bytes,
        config.range_latency_ms,
    )
    .await
    .context("open inverse paged graph fixture with sized cache")?;
    let forward = measure_graph_direction(
        &forward_no_cache,
        &forward_sized_cache,
        EdgeDirection::Forward,
        forward_bytes,
        config,
    )
    .await?;
    let inverse = measure_graph_direction(
        &inverse_no_cache,
        &inverse_sized_cache,
        EdgeDirection::Inverse,
        inverse_bytes,
        config,
    )
    .await?;
    let parity_exact = forward
        .operations
        .iter()
        .all(|operation| operation.parity_exact)
        && inverse
            .operations
            .iter()
            .all(|operation| operation.parity_exact)
        && forward.range_complete
        && inverse.range_complete;
    let max_cold_artifact_ratio = forward
        .operations
        .iter()
        .chain(&inverse.operations)
        .flat_map(|operation| {
            [
                operation.no_cache.artifact_ratio,
                operation.sized_cache_cold.artifact_ratio,
            ]
        })
        .fold(0.0f64, f64::max);
    Ok(GraphPagedReport {
        artifact_bytes: forward_bytes.saturating_add(inverse_bytes),
        edge_count: forward_artifact.canonical_edge_count,
        high_degree: config.graph_high_degree,
        forward,
        inverse,
        parity_exact,
        max_cold_artifact_ratio,
        cache_modes: GraphCacheModeReport {
            explicit_zero_and_sized_available: true,
            measured_counters:
                "reader logical authenticated ranges plus physical ObjectStore range GETs below cache",
            no_cache: "PagedEdgeReader opened with cache=None",
            sized_cache_cold: "isolated bounded ImmutableRangeCache cleared before the operation",
            sized_cache_warm:
                "identical immediate repeat against the populated bounded ImmutableRangeCache",
        },
        api_limitations: Vec::new(),
    })
}

async fn open_graph_cache(bytes: usize, namespace: &str) -> Result<ImmutableRangeCache> {
    let mut cache_config = RangeCacheConfig::memory_only(bytes);
    cache_config.key_namespace = namespace.to_owned();
    ImmutableRangeCache::open(cache_config)
        .await
        .map_err(|error| anyhow!("open {namespace} immutable range cache: {error}"))
}

async fn open_graph_reader(
    backing: Arc<InMemory>,
    meta: ObjectMeta,
    cache: Option<ImmutableRangeCache>,
    artifact_bytes: u64,
    latency_ms: u64,
) -> Result<GraphReaderHarness> {
    let store = Arc::new(CountingObjectStore::new(
        backing,
        Duration::from_millis(latency_ms),
    ));
    let object_store: Arc<dyn ObjectStore> = store.clone();
    let cache_before = cache
        .as_ref()
        .map(ImmutableRangeCache::stats)
        .unwrap_or_default();
    let reader = PagedEdgeReader::open_with_meta_and_cache(object_store, meta, cache.clone())
        .await
        .context("open PagedEdgeReader with explicit cache policy")?;
    let cache_after = cache
        .as_ref()
        .map(ImmutableRangeCache::stats)
        .unwrap_or_default();
    let open_io = graph_io_sample(
        reader.io_stats(),
        store.snapshot(),
        subtract_cache_stats(cache_after, cache_before),
        artifact_bytes,
    );
    Ok(GraphReaderHarness {
        reader,
        store,
        cache,
        open_io,
    })
}

fn graph_id(group: u64, ordinal: usize) -> [u8; 16] {
    ((u128::from(group) << 64) | ordinal as u128).to_be_bytes()
}

fn graph_lsn(kind: GraphEdgeKind, ordinal: usize) -> u64 {
    let base = match kind {
        GraphEdgeKind::OutboundHub => 1_000u64,
        GraphEdgeKind::Regular => 1_000_000u64,
        GraphEdgeKind::InboundHub => 2_000_000u64,
    };
    base.saturating_add(ordinal as u64)
}

#[derive(Debug, Clone, Copy)]
enum GraphEdgeKind {
    OutboundHub,
    Regular,
    InboundHub,
}

fn graph_code(kind: GraphEdgeKind, ordinal: usize) -> String {
    let prefix = match kind {
        GraphEdgeKind::OutboundHub => "OUT",
        GraphEdgeKind::Regular => "REG",
        GraphEdgeKind::InboundHub => "IN",
    };
    format!("{prefix}-{ordinal}")
}

fn graph_record(
    key_id: [u8; 16],
    partner_id: [u8; 16],
    kind: GraphEdgeKind,
    ordinal: usize,
) -> EdgeRecord {
    EdgeRecord {
        key_id,
        partner_id,
        lsn: graph_lsn(kind, ordinal),
        tombstone: false,
        declared_properties: vec![Some(format!("\"{}\"", graph_code(kind, ordinal)))],
        overflow_json: None,
    }
}

fn build_graph_artifact(
    config: &ObjectNativeConfig,
    direction: EdgeDirection,
) -> Result<GraphArtifact> {
    let expected_keys = match direction {
        EdgeDirection::Forward => 1usize
            .saturating_add(config.graph_keys)
            .saturating_add(config.graph_high_degree),
        EdgeDirection::Inverse => config
            .graph_high_degree
            .saturating_add(config.graph_keys)
            .saturating_add(1),
    };
    let mut options = EdgeSstWriterOptions::new(direction, "REFERENCIA", "Articulo", "Articulo");
    options.schema_version = 2;
    options.skew_threshold = Some(32);
    options.expected_keys = expected_keys as u64;
    options.compress_property_streams = true;
    options.declared_properties = vec!["codigo".to_owned()];
    let mut writer = EdgeSstWriter::new(options);
    match direction {
        EdgeDirection::Forward => {
            for ordinal in 0..config.graph_high_degree {
                writer
                    .append(graph_record(
                        graph_id(1, 0),
                        graph_id(10, ordinal),
                        GraphEdgeKind::OutboundHub,
                        ordinal,
                    ))
                    .map_err(|error| anyhow!("append forward outbound hub {ordinal}: {error}"))?;
            }
            for ordinal in 0..config.graph_keys {
                writer
                    .append(graph_record(
                        graph_id(2, ordinal),
                        graph_id(20, ordinal),
                        GraphEdgeKind::Regular,
                        ordinal,
                    ))
                    .map_err(|error| anyhow!("append forward regular edge {ordinal}: {error}"))?;
            }
            for ordinal in 0..config.graph_high_degree {
                writer
                    .append(graph_record(
                        graph_id(3, ordinal),
                        graph_id(30, 0),
                        GraphEdgeKind::InboundHub,
                        ordinal,
                    ))
                    .map_err(|error| anyhow!("append forward inbound hub {ordinal}: {error}"))?;
            }
        }
        EdgeDirection::Inverse => {
            for ordinal in 0..config.graph_high_degree {
                writer
                    .append(graph_record(
                        graph_id(10, ordinal),
                        graph_id(1, 0),
                        GraphEdgeKind::OutboundHub,
                        ordinal,
                    ))
                    .map_err(|error| anyhow!("append inverse outbound hub {ordinal}: {error}"))?;
            }
            for ordinal in 0..config.graph_keys {
                writer
                    .append(graph_record(
                        graph_id(20, ordinal),
                        graph_id(2, ordinal),
                        GraphEdgeKind::Regular,
                        ordinal,
                    ))
                    .map_err(|error| anyhow!("append inverse regular edge {ordinal}: {error}"))?;
            }
            for ordinal in 0..config.graph_high_degree {
                writer
                    .append(graph_record(
                        graph_id(30, 0),
                        graph_id(3, ordinal),
                        GraphEdgeKind::InboundHub,
                        ordinal,
                    ))
                    .map_err(|error| anyhow!("append inverse inbound hub {ordinal}: {error}"))?;
            }
        }
    }
    let canonical_edge_count = config
        .graph_high_degree
        .saturating_mul(2)
        .saturating_add(config.graph_keys) as u64;
    let finish = writer
        .finish()
        .map_err(|error| anyhow!("finish {direction:?} graph fixture: {error}"))?;
    if finish.stats.edge_count != canonical_edge_count {
        bail!(
            "{direction:?} graph writer emitted {} edges, expected {canonical_edge_count}",
            finish.stats.edge_count
        );
    }
    Ok(GraphArtifact {
        body: finish.body,
        canonical_edge_count,
    })
}

#[derive(Debug, Clone, Copy)]
enum GraphOperation {
    HighDegreeAdjacency,
    ExactEndpointAndProperty,
    RegularAdjacency,
}

impl GraphOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::HighDegreeAdjacency => "high_degree_adjacency",
            Self::ExactEndpointAndProperty => "exact_endpoint_and_property",
            Self::RegularAdjacency => "regular_adjacency",
        }
    }
}

async fn measure_graph_direction(
    no_cache: &GraphReaderHarness,
    sized_cache: &GraphReaderHarness,
    direction: EdgeDirection,
    artifact_bytes: u64,
    config: &ObjectNativeConfig,
) -> Result<GraphDirectionReport> {
    let mut operations = Vec::new();
    for operation in [
        GraphOperation::HighDegreeAdjacency,
        GraphOperation::ExactEndpointAndProperty,
        GraphOperation::RegularAdjacency,
    ] {
        let (no_cache_parity, no_cache_io) =
            measure_graph_operation(no_cache, direction, operation, artifact_bytes, config).await?;
        let cache = sized_cache
            .cache
            .as_ref()
            .ok_or_else(|| anyhow!("sized graph harness is missing its explicit cache"))?;
        cache
            .clear()
            .await
            .map_err(|error| anyhow!("clear sized graph range cache: {error}"))?;
        let (cold_parity, cold_io) =
            measure_graph_operation(sized_cache, direction, operation, artifact_bytes, config)
                .await?;
        let (warm_parity, warm_io) =
            measure_graph_operation(sized_cache, direction, operation, artifact_bytes, config)
                .await?;
        operations.push(GraphOperationReport {
            operation: operation.name(),
            parity_exact: no_cache_parity && cold_parity && warm_parity,
            no_cache: no_cache_io,
            sized_cache_cold: cold_io,
            sized_cache_warm: warm_io,
        });
    }
    let resident_metadata_bytes = sized_cache.reader.resident_metadata_bytes();
    Ok(GraphDirectionReport {
        direction: match direction {
            EdgeDirection::Forward => "forward",
            EdgeDirection::Inverse => "inverse",
        },
        range_complete: no_cache.reader.is_range_complete()
            && sized_cache.reader.is_range_complete(),
        resident_metadata_bytes,
        resident_metadata_artifact_ratio: resident_metadata_bytes as f64
            / artifact_bytes.max(1) as f64,
        reader_open_no_cache: no_cache.open_io,
        reader_open_sized_cache: sized_cache.open_io,
        operations,
    })
}

async fn measure_graph_operation(
    harness: &GraphReaderHarness,
    direction: EdgeDirection,
    operation: GraphOperation,
    artifact_bytes: u64,
    config: &ObjectNativeConfig,
) -> Result<(bool, GraphIoSample)> {
    harness.store.reset();
    let logical_before = harness.reader.io_stats();
    let cache_before = harness
        .cache
        .as_ref()
        .map(ImmutableRangeCache::stats)
        .unwrap_or_default();
    let parity = execute_graph_operation(&harness.reader, direction, operation, config).await?;
    let logical_after = harness.reader.io_stats();
    let cache_after = harness
        .cache
        .as_ref()
        .map(ImmutableRangeCache::stats)
        .unwrap_or_default();
    Ok((
        parity,
        graph_io_sample(
            subtract_graph_io(logical_after, logical_before),
            harness.store.snapshot(),
            subtract_cache_stats(cache_after, cache_before),
            artifact_bytes,
        ),
    ))
}

async fn execute_graph_operation(
    reader: &PagedEdgeReader,
    direction: EdgeDirection,
    operation: GraphOperation,
    config: &ObjectNativeConfig,
) -> Result<bool> {
    match operation {
        GraphOperation::HighDegreeAdjacency => {
            let (key, partner_group, kind) = match direction {
                EdgeDirection::Forward => (graph_id(1, 0), 10, GraphEdgeKind::OutboundHub),
                EdgeDirection::Inverse => (graph_id(30, 0), 3, GraphEdgeKind::InboundHub),
            };
            let Some(found) = reader
                .lookup(&key)
                .await
                .map_err(|error| anyhow!("{direction:?} high-degree lookup: {error}"))?
            else {
                return Ok(false);
            };
            if found.partners.len() != config.graph_high_degree
                || found.lsns.len() != config.graph_high_degree
                || found.tombstones.len() != config.graph_high_degree
            {
                return Ok(false);
            }
            Ok((0..config.graph_high_degree).all(|ordinal| {
                found.partners[ordinal] == graph_id(partner_group, ordinal)
                    && found.lsns[ordinal] == graph_lsn(kind, ordinal)
                    && !found.tombstones[ordinal]
            }))
        }
        GraphOperation::ExactEndpointAndProperty => {
            let ordinal = config.graph_high_degree / 2;
            let (key, partner, kind) = match direction {
                EdgeDirection::Forward => (
                    graph_id(1, 0),
                    graph_id(10, ordinal),
                    GraphEdgeKind::OutboundHub,
                ),
                EdgeDirection::Inverse => (
                    graph_id(30, 0),
                    graph_id(3, ordinal),
                    GraphEdgeKind::InboundHub,
                ),
            };
            let Some(found) = reader
                .lookup_partner(&key, &partner)
                .await
                .map_err(|error| anyhow!("{direction:?} exact endpoint lookup: {error}"))?
            else {
                return Ok(false);
            };
            let start = found.edge_offset as u64;
            let values = reader
                .read_property_rows("codigo", start..start + 1)
                .await
                .map_err(|error| anyhow!("{direction:?} exact property row: {error}"))?;
            Ok(found.lsn == graph_lsn(kind, ordinal)
                && !found.tombstone
                && values == Some(vec![Some(format!("\"{}\"", graph_code(kind, ordinal)))]))
        }
        GraphOperation::RegularAdjacency => {
            let (key, partner) = match direction {
                EdgeDirection::Forward => (graph_id(2, 0), graph_id(20, 0)),
                EdgeDirection::Inverse => (graph_id(20, 0), graph_id(2, 0)),
            };
            let Some(found) = reader
                .lookup(&key)
                .await
                .map_err(|error| anyhow!("{direction:?} regular adjacency lookup: {error}"))?
            else {
                return Ok(false);
            };
            Ok(found.partners == vec![partner]
                && found.lsns == vec![graph_lsn(GraphEdgeKind::Regular, 0)]
                && found.tombstones == vec![false])
        }
    }
}

fn subtract_graph_io(after: PagedEdgeIoStats, before: PagedEdgeIoStats) -> PagedEdgeIoStats {
    PagedEdgeIoStats {
        range_requests: after.range_requests.saturating_sub(before.range_requests),
        bytes_read: after.bytes_read.saturating_sub(before.bytes_read),
        eager_body_reads: after
            .eager_body_reads
            .saturating_sub(before.eager_body_reads),
    }
}

fn subtract_cache_stats(after: RangeCacheStats, before: RangeCacheStats) -> RangeCacheStats {
    RangeCacheStats {
        memory_hits: after.memory_hits.saturating_sub(before.memory_hits),
        disk_hits: after.disk_hits.saturating_sub(before.disk_hits),
        misses: after.misses.saturating_sub(before.misses),
        outer_fetches: after.outer_fetches.saturating_sub(before.outer_fetches),
        inserts: after.inserts.saturating_sub(before.inserts),
        admission_rejections: after
            .admission_rejections
            .saturating_sub(before.admission_rejections),
        corrupt_entries: after.corrupt_entries.saturating_sub(before.corrupt_entries),
    }
}

fn graph_io_sample(
    logical: PagedEdgeIoStats,
    physical: PhysicalIoSnapshot,
    cache: RangeCacheStats,
    artifact_bytes: u64,
) -> GraphIoSample {
    GraphIoSample {
        logical_requests: logical.range_requests,
        logical_bytes: logical.bytes_read,
        fetch_requests: physical.fetch_requests,
        fetched_bytes: physical.fetched_bytes,
        cache_hits: cache.memory_hits.saturating_add(cache.disk_hits),
        peak_in_flight: physical.peak_in_flight,
        eager_body_reads: logical.eager_body_reads,
        artifact_ratio: ratio(physical.fetched_bytes, artifact_bytes),
    }
}
