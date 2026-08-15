//! Query-scoped Search-LSM reader coordination.
//!
//! Manifest selection is pure and lives in `search_lsm`. This module owns the
//! physical all-or-nothing boundary: it pins the barrier and every referenced
//! object before a query, opens all native readers, and reconciles their
//! version tables without retaining a corpus-sized winner map.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use object_store::path::Path;
use object_store::ObjectStoreExt;
use uuid::Uuid;

use namidb_core::{NodeId, Value};

use super::{optional_accelerator_fallback, SearchObjectRangeSource, Snapshot};
use crate::error::{Error, Result};
use crate::manifest::{SstDescriptor, VectorMetric};
use crate::search_lsm::{
    select_search_read_plan, validate_search_barrier, SearchLsmKind, SearchLsmState,
    SearchReadPlan, SearchSegmentFormat, SearchSegmentRef, SearchSegmentRole, SearchSegmentStats,
    SearchStatValue,
};
use crate::sst::search_delta::{
    SearchFilterValue, SearchVersionOperation, SearchVersionRangeSource, SearchVersionRecord,
    SearchVersionTableReader,
};

const WINNER_PROBE_BATCH: usize = 512;
/// Segment lookups inside one query are independent object-store reads. Issuing
/// them serially turns a generation of N segments into N dependent round trips
/// before anything can be scored, which dominates cold latency far more than
/// the bytes involved. Ordered buffering keeps every fold deterministic.
const SEARCH_SEGMENT_FANOUT: usize = 16;
#[cfg(feature = "vector-index")]
const COORDINATOR_VECTOR_CANDIDATE_BYTES: usize = 512;
#[cfg(feature = "text-index")]
const COORDINATOR_TEXT_TERM_BYTES: usize = 192;
/// Candidates fetched per segment per requested hit. Reconciliation only ever
/// *removes* rows a newer segment supersedes, so a small multiple almost always
/// survives in full; the refill loop covers the case where it does not.
#[cfg(feature = "text-index")]
const TEXT_SEGMENT_OVERFETCH: usize = 4;

#[cfg(feature = "vector-index")]
#[derive(Debug)]
enum OpenedVectorSegment {
    V5Base {
        segment: SearchSegmentRef,
        reader: Arc<crate::sst::vector::v5::VectorV5Reader>,
    },
    V6 {
        segment: SearchSegmentRef,
        reader: Arc<crate::sst::vector::v6::VectorV6Reader>,
    },
}

#[cfg(feature = "vector-index")]
impl OpenedVectorSegment {
    fn segment(&self) -> &SearchSegmentRef {
        match self {
            Self::V5Base { segment, .. } | Self::V6 { segment, .. } => segment,
        }
    }

    fn local_live_count(&self) -> u64 {
        match self {
            Self::V5Base { reader, .. } => reader.point_count(),
            Self::V6 { segment, .. } => segment.live_payload_count,
        }
    }

    fn resident_metadata_bytes(&self) -> usize {
        match self {
            Self::V5Base { reader, .. } => reader.resident_metadata_bytes(),
            Self::V6 { reader, .. } => reader.resident_metadata_bytes(),
        }
    }

    fn supports_filter_property(&self, property: &str) -> bool {
        self.segment()
            .complete_filter_properties
            .binary_search_by(|candidate| candidate.as_str().cmp(property))
            .is_ok()
            && match self {
                Self::V5Base { reader, .. } => reader.supports_filter_property(property),
                Self::V6 { reader, .. } => reader.supports_filter_property(property),
            }
    }

    fn version_reader(&self) -> Option<&SearchVersionTableReader> {
        match self {
            Self::V5Base { .. } => None,
            Self::V6 { reader, .. } => Some(reader.version_reader()),
        }
    }
}

#[cfg(feature = "vector-index")]
#[derive(Debug)]
struct OpenedVectorGeneration {
    state: SearchLsmState,
    segments: Vec<OpenedVectorSegment>,
    point_count: u64,
    metric: VectorMetric,
}

#[cfg(feature = "text-index")]
#[derive(Debug)]
struct OpenedTextSegment {
    segment: SearchSegmentRef,
    reader: Arc<crate::sst::text::v4::TextV4Reader>,
}

#[cfg(feature = "text-index")]
impl OpenedTextSegment {
    fn supports_filter_property(&self, property: &str) -> bool {
        self.segment
            .complete_filter_properties
            .binary_search_by(|candidate| candidate.as_str().cmp(property))
            .is_ok()
            && self.reader.supports_filter_property(property)
    }
}

#[cfg(feature = "text-index")]
#[derive(Debug)]
struct OpenedTextGeneration {
    state: SearchLsmState,
    segments: Vec<OpenedTextSegment>,
    document_count: u64,
    total_document_len: u64,
}

#[cfg(feature = "text-index")]
impl OpenedTextGeneration {
    /// The compatibility barrier and every segment/footer binding have
    /// already been pinned and validated by `open_text_generation`. These
    /// additional checks prove that no winner reconciliation is necessary.
    fn authoritative_single_base(&self, dirty_ids: &[[u8; 16]]) -> Option<&OpenedTextSegment> {
        if !dirty_ids.is_empty()
            || !self.state.proven_empty_event_ranges.is_empty()
            || self.state.segments.len() != 1
            || self.segments.len() != 1
            || self.state.equal_lsn_conflict_count != 0
        {
            return None;
        }
        let segment = &self.segments[0];
        let frontier = self.state.base_frontier?;
        let SearchSegmentStats::Text {
            doc_count: SearchStatValue::Absolute(documents),
            total_len: SearchStatValue::Absolute(total_len),
            term_df_violation_count: 0,
        } = segment.segment.stats
        else {
            return None;
        };
        let event_range = segment.segment.event_ranges.as_slice();
        if frontier == 0
            || self.state.next_event_seq != frontier
            || event_range.len() != 1
            || event_range[0].start != 0
            || event_range[0].end != frontier
            || segment.segment.role != SearchSegmentRole::Base
            || segment.segment.format != SearchSegmentFormat::TextV4
            || segment.segment.suppress_count != 0
            || segment.segment.equal_lsn_conflict_count != 0
            || segment.segment.mutation_count != segment.segment.live_payload_count
            || segment.segment.live_payload_count != documents
            || documents != self.document_count
            || total_len != self.total_document_len
            || segment.reader.live_document_count() != documents
            || u64::try_from(segment.reader.delta_docs()).ok() != Some(documents)
            || u64::try_from(segment.reader.delta_total_len()).ok() != Some(total_len)
            || self.state.segments.first() != Some(&segment.segment)
            || segment.reader.segment() != &segment.segment
        {
            return None;
        }
        Some(segment)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinnerOperation {
    Live,
    Suppress,
}

impl From<SearchVersionOperation> for WinnerOperation {
    fn from(operation: SearchVersionOperation) -> Self {
        match operation {
            SearchVersionOperation::Live { .. } => Self::Live,
            SearchVersionOperation::Suppress => Self::Suppress,
        }
    }
}

#[derive(Debug)]
struct WinnerBatch {
    records: Vec<Option<SearchVersionRecord>>,
    conflict: bool,
}

/// Search-LSM routing result. `NotActive` lets the caller retain the legacy
/// base path; `Unavailable` means the selected active generation must fall
/// back atomically rather than serving a partial segment set.
#[derive(Debug)]
pub(super) enum ActiveSearch<T> {
    NotActive,
    Unavailable,
    Ready(T),
}

#[cfg(feature = "vector-index")]
#[derive(Debug)]
pub(super) struct CoordinatedVectorResult {
    pub hits: Vec<(NodeId, f32)>,
    pub point_count: u64,
    /// Exact only when every eligible candidate was exhausted; otherwise the
    /// sentinel prevents callers from claiming completeness from a partial
    /// native-filter enumeration.
    pub eligible_count: usize,
}

#[cfg(feature = "vector-index")]
#[derive(Debug)]
pub(super) enum CoordinatedVectorFilterResult {
    Applied(CoordinatedVectorResult),
    Unsupported,
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn active_state(
    snapshot: &Snapshot<'_>,
    kind: SearchLsmKind,
    index_name: &str,
) -> ActiveSearch<(SearchLsmState, Uuid)> {
    match select_search_read_plan(&snapshot.manifest.manifest, kind, index_name) {
        SearchReadPlan::ActiveSegments {
            state,
            barrier_sst_id,
        } => ActiveSearch::Ready((state, barrier_sst_id)),
        SearchReadPlan::Legacy { .. } | SearchReadPlan::ActiveLegacyBase { .. } => {
            ActiveSearch::NotActive
        }
        SearchReadPlan::FlatFallback(reason) => {
            tracing::debug!(
                index = index_name,
                ?kind,
                ?reason,
                "Search-LSM coordinator selected exact fallback"
            );
            ActiveSearch::Unavailable
        }
    }
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn unique_descriptor<'a>(
    snapshot: &'a Snapshot<'_>,
    id: Uuid,
    state: &SearchLsmState,
) -> Option<&'a SstDescriptor> {
    let mut matches = snapshot
        .manifest
        .manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.id == id);
    let descriptor = matches.next()?;
    if matches.next().is_some()
        || descriptor.kind != state.kind.sst_kind()
        || descriptor.scope != state.index_name
    {
        return None;
    }
    Some(descriptor)
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
async fn pin_descriptor(
    snapshot: &Snapshot<'_>,
    descriptor: &SstDescriptor,
) -> Result<Option<Arc<SearchObjectRangeSource>>> {
    crate::cancel::check()?;
    let absolute = format!(
        "{}/{}",
        snapshot.paths.namespace_prefix().as_ref(),
        descriptor.path
    );
    let path = Path::from(absolute.as_str());
    let meta = match snapshot.store.head(&path).await {
        Ok(meta) => meta,
        Err(error) => {
            let error = Error::ObjectStore(error);
            if optional_accelerator_fallback(&error) {
                tracing::warn!(
                    path = %descriptor.path,
                    error = %error,
                    "Search-LSM object disappeared; falling back atomically"
                );
                return Ok(None);
            }
            return Err(error);
        }
    };
    if meta.location != path || meta.size != descriptor.size_bytes {
        tracing::warn!(
            path = %descriptor.path,
            manifest_size = descriptor.size_bytes,
            object_size = meta.size,
            "Search-LSM object identity/size disagrees with descriptor"
        );
        return Ok(None);
    }
    match SearchObjectRangeSource::new(snapshot.store.clone(), meta).await {
        Ok(source) => Ok(Some(Arc::new(source))),
        Err(error) if optional_accelerator_fallback(&error) => {
            tracing::warn!(
                path = %descriptor.path,
                error = %error,
                "Search-LSM object could not be generation-pinned"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
async fn validate_and_pin_generation_barrier(
    snapshot: &Snapshot<'_>,
    state: &SearchLsmState,
    barrier_sst_id: Uuid,
) -> Result<bool> {
    let Some(descriptor) = unique_descriptor(snapshot, barrier_sst_id, state) else {
        return Ok(false);
    };
    let Some(source) = pin_descriptor(snapshot, descriptor).await? else {
        return Ok(false);
    };
    let body = match source.read(0..source.file_len()).await {
        Ok(body) => body,
        Err(error) if optional_accelerator_fallback(&error) => return Ok(false),
        Err(error) => return Err(error),
    };
    if body.len() as u64 != descriptor.size_bytes {
        return Ok(false);
    }
    if let Err(error) = validate_search_barrier(state, &body) {
        tracing::warn!(
            index = %state.index_name,
            generation = %state.generation_id,
            error = %error,
            "Search-LSM barrier validation failed"
        );
        return Ok(false);
    }
    Ok(true)
}

fn apply_stat(role: SearchSegmentRole, current: u64, value: SearchStatValue) -> Option<u64> {
    match (role, value) {
        (SearchSegmentRole::Base, SearchStatValue::Absolute(value)) => Some(value),
        (SearchSegmentRole::Delta, SearchStatValue::Delta(delta)) if delta >= 0 => {
            current.checked_add(delta as u64)
        }
        (SearchSegmentRole::Delta, SearchStatValue::Delta(delta)) => {
            current.checked_sub(delta.unsigned_abs())
        }
        _ => None,
    }
}

fn vector_live_count(state: &SearchLsmState) -> Option<u64> {
    let mut live = 0u64;
    for segment in &state.segments {
        let SearchSegmentStats::Vector { live_count } = segment.stats else {
            return None;
        };
        live = apply_stat(segment.role, live, live_count)?;
    }
    Some(live)
}

fn text_live_stats(state: &SearchLsmState) -> Option<(u64, u64)> {
    let (mut documents, mut total_len) = (0u64, 0u64);
    for segment in &state.segments {
        let SearchSegmentStats::Text {
            doc_count,
            total_len: segment_len,
            term_df_violation_count: 0,
        } = segment.stats
        else {
            return None;
        };
        documents = apply_stat(segment.role, documents, doc_count)?;
        total_len = apply_stat(segment.role, total_len, segment_len)?;
        if documents == 0 && total_len != 0 {
            return None;
        }
    }
    Some((documents, total_len))
}

fn metadata_fits_workspace(bytes: usize) -> bool {
    bytes <= crate::search_workspace::shared_search_workspace().capacity_bytes()
}

#[cfg(feature = "vector-index")]
async fn open_vector_generation(
    snapshot: &Snapshot<'_>,
    index_name: &str,
) -> Result<ActiveSearch<OpenedVectorGeneration>> {
    let (state, barrier_sst_id) = match active_state(snapshot, SearchLsmKind::Vector, index_name) {
        ActiveSearch::NotActive => return Ok(ActiveSearch::NotActive),
        ActiveSearch::Unavailable => return Ok(ActiveSearch::Unavailable),
        ActiveSearch::Ready(value) => value,
    };
    if !validate_and_pin_generation_barrier(snapshot, &state, barrier_sst_id).await? {
        return Ok(ActiveSearch::Unavailable);
    }
    let mut catalog_matches = snapshot
        .manifest
        .manifest
        .vector_indexes
        .iter()
        .filter(|descriptor| descriptor.name == index_name);
    let Some(catalog) = catalog_matches.next() else {
        return Ok(ActiveSearch::Unavailable);
    };
    if catalog_matches.next().is_some() {
        return Ok(ActiveSearch::Unavailable);
    }
    let catalog = catalog.clone();
    let point_count = match vector_live_count(&state) {
        Some(value) => value,
        None => return Ok(ActiveSearch::Unavailable),
    };
    let mut segments = Vec::with_capacity(state.segments.len());
    let mut resident_bytes = segments
        .capacity()
        .saturating_mul(std::mem::size_of::<OpenedVectorSegment>());
    for segment in &state.segments {
        let Some(descriptor) = unique_descriptor(snapshot, segment.sst_id, &state) else {
            return Ok(ActiveSearch::Unavailable);
        };
        let Some(source) = pin_descriptor(snapshot, descriptor).await? else {
            return Ok(ActiveSearch::Unavailable);
        };
        let opened = match segment.format {
            SearchSegmentFormat::VectorV5Base => {
                let source: Arc<dyn crate::sst::vector::v5::VectorV5RangeSource> = source;
                let reader = match crate::sst::vector::v5::VectorV5Reader::open(
                    source,
                    descriptor.size_bytes,
                )
                .await
                {
                    Ok(reader) => Arc::new(reader),
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            path = %descriptor.path,
                            error = %error,
                            "Search-LSM V5 base validation failed"
                        );
                        return Ok(ActiveSearch::Unavailable);
                    }
                    Err(error) => return Err(error),
                };
                let wrapped = super::VectorSearchIndex::Ranged(reader.clone());
                if !super::active_vector_base_matches(&wrapped, descriptor, segment) {
                    return Ok(ActiveSearch::Unavailable);
                }
                OpenedVectorSegment::V5Base {
                    segment: segment.clone(),
                    reader,
                }
            }
            SearchSegmentFormat::VectorV6 => {
                let source: Arc<dyn SearchVersionRangeSource> = source;
                let reader = match crate::sst::vector::v6::VectorV6Reader::open(
                    source,
                    descriptor.size_bytes,
                    &state,
                    segment,
                    &catalog,
                )
                .await
                {
                    Ok(reader) => Arc::new(reader),
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            path = %descriptor.path,
                            error = %error,
                            "Search-LSM VG6 footer/lineage validation failed"
                        );
                        return Ok(ActiveSearch::Unavailable);
                    }
                    Err(error) => return Err(error),
                };
                OpenedVectorSegment::V6 {
                    segment: segment.clone(),
                    reader,
                }
            }
            _ => return Ok(ActiveSearch::Unavailable),
        };
        resident_bytes = match resident_bytes.checked_add(opened.resident_metadata_bytes()) {
            Some(value) => value,
            None => return Ok(ActiveSearch::Unavailable),
        };
        if !metadata_fits_workspace(resident_bytes) {
            tracing::warn!(
                index = index_name,
                resident_bytes,
                "Search-LSM vector metadata exceeds query workspace"
            );
            return Ok(ActiveSearch::Unavailable);
        }
        segments.push(opened);
    }
    if segments.len() != state.segments.len()
        || segments
            .iter()
            .any(|segment| segment.segment().sst_id.is_nil())
    {
        return Ok(ActiveSearch::Unavailable);
    }
    Ok(ActiveSearch::Ready(OpenedVectorGeneration {
        state,
        segments,
        point_count,
        metric: catalog.metric,
    }))
}

#[cfg(feature = "text-index")]
async fn open_text_generation(
    snapshot: &Snapshot<'_>,
    index_name: &str,
) -> Result<ActiveSearch<OpenedTextGeneration>> {
    let (state, barrier_sst_id) = match active_state(snapshot, SearchLsmKind::Text, index_name) {
        ActiveSearch::NotActive => return Ok(ActiveSearch::NotActive),
        ActiveSearch::Unavailable => return Ok(ActiveSearch::Unavailable),
        ActiveSearch::Ready(value) => value,
    };
    if !validate_and_pin_generation_barrier(snapshot, &state, barrier_sst_id).await? {
        return Ok(ActiveSearch::Unavailable);
    }
    let (document_count, total_document_len) = match text_live_stats(&state) {
        Some(value) => value,
        None => return Ok(ActiveSearch::Unavailable),
    };
    let mut segments = Vec::with_capacity(state.segments.len());
    let mut resident_bytes = segments
        .capacity()
        .saturating_mul(std::mem::size_of::<OpenedTextSegment>());
    for segment in &state.segments {
        if segment.format != SearchSegmentFormat::TextV4 {
            return Ok(ActiveSearch::Unavailable);
        }
        let Some(descriptor) = unique_descriptor(snapshot, segment.sst_id, &state) else {
            return Ok(ActiveSearch::Unavailable);
        };
        let Some(source) = pin_descriptor(snapshot, descriptor).await? else {
            return Ok(ActiveSearch::Unavailable);
        };
        let source: Arc<dyn SearchVersionRangeSource> = source;
        let reader = match crate::sst::text::v4::TextV4Reader::open(
            source,
            descriptor.size_bytes,
            &state,
            segment,
        )
        .await
        {
            Ok(reader) => Arc::new(reader),
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    path = %descriptor.path,
                    error = %error,
                    "Search-LSM FT4 footer/lineage validation failed"
                );
                return Ok(ActiveSearch::Unavailable);
            }
            Err(error) => return Err(error),
        };
        resident_bytes = match resident_bytes.checked_add(reader.resident_metadata_bytes()) {
            Some(value) => value,
            None => return Ok(ActiveSearch::Unavailable),
        };
        if !metadata_fits_workspace(resident_bytes) {
            tracing::warn!(
                index = index_name,
                resident_bytes,
                "Search-LSM text metadata exceeds query workspace"
            );
            return Ok(ActiveSearch::Unavailable);
        }
        segments.push(OpenedTextSegment {
            segment: segment.clone(),
            reader,
        });
    }
    if segments.len() != state.segments.len() {
        return Ok(ActiveSearch::Unavailable);
    }
    Ok(ActiveSearch::Ready(OpenedTextGeneration {
        state,
        segments,
        document_count,
        total_document_len,
    }))
}

async fn probe_winners<'a>(
    readers_newest_first: impl IntoIterator<Item = &'a SearchVersionTableReader>,
    node_ids: &[[u8; 16]],
) -> Result<WinnerBatch> {
    use futures::{StreamExt, TryStreamExt};

    // Collected eagerly: a borrowing closure inside the opaque stream type
    // would carry a higher-ranked `FnOnce` obligation that rustc rejects once
    // this future crosses the executor's `Send` boundary (rust-lang/rust
    // #102211). A `Vec` of already-constructed futures has no closure in its
    // type at all.
    let probes = futures::stream::iter(
        readers_newest_first
            .into_iter()
            .map(|reader| reader.point_probe_many(node_ids))
            .collect::<Vec<_>>(),
    )
    .buffered(SEARCH_SEGMENT_FANOUT)
    .try_collect::<Vec<_>>()
    .await?;

    let mut records = vec![None; node_ids.len()];
    let mut conflict = false;
    // Ordered buffering preserved newest-first, so the merge is unchanged.
    for probed in probes {
        if probed.len() != node_ids.len() {
            return Err(Error::invariant(
                "Search-LSM winner probe returned the wrong record count",
            ));
        }
        for ((winner, candidate), expected_node_id) in records.iter_mut().zip(probed).zip(node_ids)
        {
            let Some(candidate) = candidate else {
                continue;
            };
            if candidate.node_id != *expected_node_id {
                return Err(Error::invariant(
                    "Search-LSM winner probe returned a record for the wrong NodeId",
                ));
            }
            conflict |= merge_winner_record(winner, candidate);
        }
    }
    Ok(WinnerBatch { records, conflict })
}

fn merge_winner_record(
    winner: &mut Option<SearchVersionRecord>,
    candidate: SearchVersionRecord,
) -> bool {
    match winner {
        None => {
            *winner = Some(candidate);
            false
        }
        Some(current) if candidate.lsn > current.lsn => {
            *winner = Some(candidate);
            false
        }
        Some(current) if candidate.lsn < current.lsn => false,
        Some(current) => {
            candidate.payload_fingerprint != current.payload_fingerprint
                || WinnerOperation::from(candidate.operation)
                    != WinnerOperation::from(current.operation)
        }
    }
}

fn versioned_candidate_wins(
    candidate_id: [u8; 16],
    candidate_lsn: u64,
    candidate_fingerprint: u64,
    winner: Option<SearchVersionRecord>,
) -> bool {
    winner.is_some_and(|winner| {
        winner.node_id == candidate_id
            && winner.lsn == candidate_lsn
            && winner.payload_fingerprint == candidate_fingerprint
            && matches!(winner.operation, SearchVersionOperation::Live { .. })
    })
}

#[cfg(feature = "vector-index")]
#[derive(Debug, Clone, Copy)]
enum VectorCandidateVersion {
    V5Base,
    Versioned { lsn: u64, fingerprint: u64 },
}

#[cfg(feature = "vector-index")]
#[derive(Debug, Clone, Copy)]
struct VectorCandidate {
    node_id: [u8; 16],
    score: f32,
    version: VectorCandidateVersion,
}

#[cfg(feature = "vector-index")]
#[derive(Debug)]
struct SegmentVectorCandidates {
    hits: Vec<VectorCandidate>,
    eligible_rows_seen: usize,
    /// `eligible_rows_seen` counts only the rows inside the pages actually
    /// probed. It is a segment-wide total — and therefore usable as an
    /// exhaustion proof — only when every page was probed. An approximate
    /// probe that happens to see few eligible rows and return all of them
    /// proves nothing about the rows it never looked at.
    probed_all_pages: bool,
}

#[cfg(feature = "vector-index")]
#[derive(Debug, Clone, Copy)]
struct RankedVectorHit {
    node_id: NodeId,
    score: f32,
    higher_is_better: bool,
}

#[cfg(feature = "vector-index")]
impl PartialEq for RankedVectorHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

#[cfg(feature = "vector-index")]
impl Eq for RankedVectorHit {}

#[cfg(feature = "vector-index")]
impl PartialOrd for RankedVectorHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "vector-index")]
impl Ord for RankedVectorHit {
    fn cmp(&self, other: &Self) -> Ordering {
        let score = if self.higher_is_better {
            // Lower similarity is the worse (heap-root) value.
            other.score.total_cmp(&self.score)
        } else {
            // Higher distance is the worse value.
            self.score.total_cmp(&other.score)
        };
        score.then_with(|| self.node_id.cmp(&other.node_id))
    }
}

#[cfg(feature = "vector-index")]
#[derive(Debug)]
struct BoundedVectorResults {
    limit: usize,
    higher_is_better: bool,
    heap: BinaryHeap<RankedVectorHit>,
    retained_ids: BTreeSet<NodeId>,
    evicted: bool,
}

#[cfg(feature = "vector-index")]
impl BoundedVectorResults {
    fn new(limit: usize, higher_is_better: bool) -> Self {
        Self {
            limit,
            higher_is_better,
            heap: BinaryHeap::with_capacity(limit.saturating_add(1)),
            retained_ids: BTreeSet::new(),
            evicted: false,
        }
    }

    fn push(&mut self, candidate: VectorCandidate) {
        if self.limit == 0 {
            return;
        }
        let node_id = NodeId(Uuid::from_bytes(candidate.node_id));
        if !self.retained_ids.insert(node_id) {
            return;
        }
        self.heap.push(RankedVectorHit {
            node_id,
            score: candidate.score,
            higher_is_better: self.higher_is_better,
        });
        if self.heap.len() > self.limit {
            let removed = self.heap.pop().expect("overfull vector result heap");
            self.retained_ids.remove(&removed.node_id);
            self.evicted = true;
        }
    }

    fn finish(self) -> Vec<(NodeId, f32)> {
        let mut hits = self
            .heap
            .into_iter()
            .map(|hit| (hit.node_id, hit.score))
            .collect::<Vec<_>>();
        if self.higher_is_better {
            hits.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
        } else {
            hits.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
        }
        hits
    }
}

#[cfg(feature = "vector-index")]
fn vector_candidate_cap() -> usize {
    crate::search_workspace::shared_search_workspace()
        .capacity_bytes()
        .checked_div(COORDINATOR_VECTOR_CANDIDATE_BYTES)
        .unwrap_or(0)
        .max(1)
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn canonical_search_filter_groups(
    groups: &[(String, Vec<Value>)],
) -> Option<Vec<(String, Vec<SearchFilterValue>)>> {
    let mut canonical = Vec::with_capacity(groups.len());
    for (property, alternatives) in groups {
        let mut values = alternatives
            .iter()
            .map(SearchFilterValue::from_value)
            .collect::<Option<Vec<_>>>()?;
        values.sort();
        values.dedup();
        canonical.push((property.clone(), values));
    }
    Some(canonical)
}

#[cfg(feature = "vector-index")]
async fn search_vector_segment(
    segment: &OpenedVectorSegment,
    query: &[f32],
    limit: usize,
    ef: usize,
    value_groups: &[(String, Vec<Value>)],
    canonical_groups: &[(String, Vec<SearchFilterValue>)],
) -> Result<SegmentVectorCandidates> {
    match segment {
        OpenedVectorSegment::V5Base { reader, .. } => {
            // A clustered V5 base is an IVF index: probing every page turns it
            // back into a full scan of the corpus it exists to avoid. `ef` is
            // the public accuracy dial on both routes, so both derive their
            // probe budget from the same policy and cannot drift apart.
            // Not `max(1)`: a base with no pages was trivially probed in full,
            // and treating it as unprobed would keep the widening loop asking
            // for pages that do not exist.
            let pages = reader.page_count();
            let result = reader
                .search_filter_groups(
                    query,
                    limit,
                    super::vector_v5_search_options(reader, ef),
                    value_groups,
                )
                .await?;
            if result.applied_filter_groups != value_groups.len() {
                return Err(Error::invariant(
                    "Search-LSM V5 base did not apply every native filter",
                ));
            }
            Ok(SegmentVectorCandidates {
                hits: result
                    .hits
                    .into_iter()
                    .map(|(node_id, score)| VectorCandidate {
                        node_id,
                        score,
                        version: VectorCandidateVersion::V5Base,
                    })
                    .collect(),
                eligible_rows_seen: result.eligible_rows_seen,
                probed_all_pages: result.probed_pages >= pages,
            })
        }
        OpenedVectorSegment::V6 { reader, .. } => {
            let result = reader.search_exact(query, limit, canonical_groups).await?;
            if result.applied_filter_groups != canonical_groups.len()
                || result.scanned_pages != reader.page_count()
            {
                return Err(Error::invariant(
                    "Search-LSM VG6 did not exhaust pages/apply every native filter",
                ));
            }
            Ok(SegmentVectorCandidates {
                hits: result
                    .hits
                    .into_iter()
                    .map(|hit| VectorCandidate {
                        node_id: hit.node_id,
                        score: hit.score,
                        version: VectorCandidateVersion::Versioned {
                            lsn: hit.lsn,
                            fingerprint: hit.payload_fingerprint,
                        },
                    })
                    .collect(),
                eligible_rows_seen: result.eligible_rows_seen,
                // The invariant above already refused anything less.
                probed_all_pages: true,
            })
        }
    }
}

#[cfg(feature = "vector-index")]
async fn reconcile_vector_candidates(
    generation: &OpenedVectorGeneration,
    candidates: &[VectorCandidate],
) -> Result<Option<Vec<VectorCandidate>>> {
    let mut accepted = Vec::with_capacity(candidates.len());
    for chunk in candidates.chunks(WINNER_PROBE_BATCH) {
        crate::cancel::check()?;
        let node_ids = chunk
            .iter()
            .map(|candidate| candidate.node_id)
            .collect::<Vec<_>>();
        let winners = probe_winners(
            generation
                .segments
                .iter()
                .rev()
                .filter_map(OpenedVectorSegment::version_reader),
            &node_ids,
        )
        .await?;
        if winners.conflict {
            tracing::warn!(
                index = %generation.state.index_name,
                generation = %generation.state.generation_id,
                "Search-LSM equal-LSN vector winner conflict; falling back"
            );
            return Ok(None);
        }
        for (candidate, winner) in chunk.iter().copied().zip(winners.records) {
            let keep = match candidate.version {
                VectorCandidateVersion::V5Base => winner.is_none(),
                VectorCandidateVersion::Versioned { lsn, fingerprint } => {
                    versioned_candidate_wins(candidate.node_id, lsn, fingerprint, winner)
                }
            };
            if keep {
                accepted.push(candidate);
            }
        }
    }
    Ok(Some(accepted))
}

#[cfg(feature = "vector-index")]
pub(super) async fn vector_search(
    snapshot: &Snapshot<'_>,
    index_name: &str,
    query: &[f32],
    k: usize,
    ef: usize,
    groups: &[(String, Vec<Value>)],
) -> Result<ActiveSearch<CoordinatedVectorFilterResult>> {
    let generation = match open_vector_generation(snapshot, index_name).await? {
        ActiveSearch::NotActive => return Ok(ActiveSearch::NotActive),
        ActiveSearch::Unavailable => return Ok(ActiveSearch::Unavailable),
        ActiveSearch::Ready(generation) => generation,
    };
    let Some(canonical_groups) = canonical_search_filter_groups(groups) else {
        return Ok(ActiveSearch::Ready(
            CoordinatedVectorFilterResult::Unsupported,
        ));
    };
    if groups.iter().any(|(property, _)| {
        generation
            .segments
            .iter()
            .any(|segment| !segment.supports_filter_property(property))
    }) {
        return Ok(ActiveSearch::Ready(
            CoordinatedVectorFilterResult::Unsupported,
        ));
    }
    let higher_is_better = generation.metric != VectorMetric::Euclidean;
    let target = k.min(usize::try_from(generation.point_count).unwrap_or(usize::MAX));
    if target == 0 {
        return Ok(ActiveSearch::Ready(CoordinatedVectorFilterResult::Applied(
            CoordinatedVectorResult {
                hits: Vec::new(),
                point_count: generation.point_count,
                eligible_count: 0,
            },
        )));
    }
    let candidate_cap = vector_candidate_cap();
    if target > candidate_cap {
        tracing::debug!(
            index = index_name,
            target,
            candidate_cap,
            "Search-LSM vector target exceeds bounded coordinator capacity"
        );
        return Ok(ActiveSearch::Unavailable);
    }

    let local_live = generation
        .segments
        .iter()
        .map(|segment| usize::try_from(segment.local_live_count()).unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    let mut limits = local_live
        .iter()
        .map(|live| target.min(*live))
        .collect::<Vec<_>>();
    // Per-segment accuracy budget. A short segment may be short because the
    // approximate probe has not looked at enough pages yet, so widening only
    // the candidate limit would re-run the identical scan and never converge.
    let mut probe_ef = vec![ef.max(1); generation.segments.len()];
    let (ranked, every_segment_exhausted) = loop {
        let mut ranked = BoundedVectorResults::new(target, higher_is_better);
        let mut statuses = Vec::with_capacity(generation.segments.len());
        for (segment_index, segment) in generation.segments.iter().enumerate() {
            let limit = limits[segment_index];
            if limit == 0 {
                statuses.push((0usize, true, true));
                continue;
            }
            let searched = match search_vector_segment(
                segment,
                query,
                limit,
                probe_ef[segment_index],
                groups,
                &canonical_groups,
            )
            .await
            {
                Ok(result) => result,
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        index = index_name,
                        segment = %segment.segment().sst_id,
                        error = %error,
                        "Search-LSM vector segment query failed; falling back atomically"
                    );
                    return Ok(ActiveSearch::Unavailable);
                }
                Err(error) => return Err(error),
            };
            let accepted = match reconcile_vector_candidates(&generation, &searched.hits).await {
                Ok(Some(accepted)) => accepted,
                Ok(None) => return Ok(ActiveSearch::Unavailable),
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        index = index_name,
                        segment = %segment.segment().sst_id,
                        error = %error,
                        "Search-LSM vector winner page is corrupt"
                    );
                    return Ok(ActiveSearch::Unavailable);
                }
                Err(error) => return Err(error),
            };
            // Only a probe that covered every page can prove the segment holds
            // nothing further; otherwise a short approximate result would be
            // reported as an exact eligible count.
            let segment_exhausted = searched.probed_all_pages
                && searched.eligible_rows_seen <= limit
                && searched.hits.len() == searched.eligible_rows_seen;
            let accepted_len = accepted.len();
            for candidate in accepted {
                ranked.push(candidate);
            }
            statuses.push((accepted_len, segment_exhausted, searched.probed_all_pages));
        }

        let every_segment_exhausted = statuses.iter().all(|(_, exhausted, _)| *exhausted);
        let each_segment_covered = statuses
            .iter()
            .all(|(accepted, exhausted, _)| *accepted >= target || *exhausted);
        if every_segment_exhausted || (ranked.heap.len() >= target && each_segment_covered) {
            break (ranked, every_segment_exhausted);
        }

        let globally_short = ranked.heap.len() < target;
        let mut changed = false;
        for (segment_index, (accepted, exhausted, probe_saturated)) in
            statuses.into_iter().enumerate()
        {
            if exhausted || (!globally_short && accepted >= target) {
                continue;
            }
            let limit = limits[segment_index];
            let limit_saturated = limit >= candidate_cap || limit >= local_live[segment_index];
            if limit_saturated && probe_saturated {
                tracing::debug!(
                    index = index_name,
                    segment = %generation.segments[segment_index].segment().sst_id,
                    limit,
                    candidate_cap,
                    "Search-LSM vector stale/duplicate candidates exceeded widening budget"
                );
                return Ok(ActiveSearch::Unavailable);
            }
            if !limit_saturated {
                let missing = target.saturating_sub(accepted).max(1);
                let next = limit
                    .saturating_mul(2)
                    .max(limit.saturating_add(missing))
                    .min(local_live[segment_index])
                    .min(candidate_cap);
                if next > limit {
                    limits[segment_index] = next;
                    changed = true;
                }
            }
            if !probe_saturated {
                // Widening `ef` widens the derived page budget; once it reaches
                // every page the segment reports exhaustion and the loop ends.
                let probe = probe_ef[segment_index];
                let next = probe.saturating_mul(4).max(probe.saturating_add(1));
                if next > probe {
                    probe_ef[segment_index] = next;
                    changed = true;
                }
            }
        }
        if !changed {
            return Ok(ActiveSearch::Unavailable);
        }
    };
    let exact_eligible_count = every_segment_exhausted && !ranked.evicted;
    let hits = ranked.finish();
    let eligible_count = if exact_eligible_count {
        hits.len()
    } else {
        usize::MAX
    };
    tracing::debug!(
        index = index_name,
        generation = %generation.state.generation_id,
        segments = generation.segments.len(),
        point_count = generation.point_count,
        returned = hits.len(),
        exact_eligible_count,
        "Search-LSM vector query reconciled"
    );
    Ok(ActiveSearch::Ready(CoordinatedVectorFilterResult::Applied(
        CoordinatedVectorResult {
            hits,
            point_count: generation.point_count,
            eligible_count,
        },
    )))
}

#[cfg(feature = "text-index")]
#[derive(Debug, Clone, Copy)]
struct TextCandidate {
    node_id: [u8; 16],
    lsn: u64,
    fingerprint: u64,
    score: f64,
}

#[cfg(feature = "text-index")]
#[derive(Debug, Clone, Copy)]
struct RankedTextHit {
    node_id: NodeId,
    score: f64,
}

#[cfg(feature = "text-index")]
impl PartialEq for RankedTextHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

#[cfg(feature = "text-index")]
impl Eq for RankedTextHit {}

#[cfg(feature = "text-index")]
impl PartialOrd for RankedTextHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "text-index")]
impl Ord for RankedTextHit {
    fn cmp(&self, other: &Self) -> Ordering {
        // Worst retained score, then largest NodeId, is the heap root.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

#[cfg(feature = "text-index")]
#[derive(Debug)]
struct BoundedTextResults {
    limit: usize,
    heap: BinaryHeap<RankedTextHit>,
    retained_ids: BTreeSet<NodeId>,
    evicted: bool,
}

#[cfg(feature = "text-index")]
impl BoundedTextResults {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit.saturating_add(1)),
            retained_ids: BTreeSet::new(),
            evicted: false,
        }
    }

    fn push(&mut self, candidate: TextCandidate) {
        if self.limit == 0 {
            return;
        }
        let node_id = NodeId(Uuid::from_bytes(candidate.node_id));
        if !self.retained_ids.insert(node_id) {
            return;
        }
        self.heap.push(RankedTextHit {
            node_id,
            score: candidate.score,
        });
        if self.heap.len() > self.limit {
            let removed = self.heap.pop().expect("overfull text result heap");
            self.retained_ids.remove(&removed.node_id);
            self.evicted = true;
        }
    }

    fn finish(self) -> Vec<(NodeId, f64)> {
        let mut hits = self
            .heap
            .into_iter()
            .map(|hit| (hit.node_id, hit.score))
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        hits
    }
}

#[cfg(feature = "text-index")]
fn apply_term_df(role: SearchSegmentRole, current: u64, contribution: i64) -> Option<u64> {
    match role {
        SearchSegmentRole::Base => u64::try_from(contribution).ok(),
        SearchSegmentRole::Delta if contribution >= 0 => current.checked_add(contribution as u64),
        SearchSegmentRole::Delta => current.checked_sub(contribution.unsigned_abs()),
    }
}

#[cfg(feature = "text-index")]
async fn globally_expanded_text_query(
    generation: &OpenedTextGeneration,
    query: &crate::text::TextQuery,
) -> Result<Option<crate::text::TextQuery>> {
    let term_cap = crate::search_workspace::shared_search_workspace()
        .capacity_bytes()
        .checked_div(COORDINATOR_TEXT_TERM_BYTES)
        .unwrap_or(0);
    let mut expanded = query.clone();
    expanded.prefixes.clear();
    let mut terms = query
        .base_terms()
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if terms.len() > term_cap {
        return Ok(None);
    }
    for prefix in &query.prefixes {
        use futures::{StreamExt, TryStreamExt};

        let mut prefix_terms = BTreeSet::new();
        let per_segment = futures::stream::iter(
            generation
                .segments
                .iter()
                .map(|segment| {
                    segment
                        .reader
                        .expand_prefix_terms(prefix, crate::text::PREFIX_EXPANSION_LIMIT)
                })
                .collect::<Vec<_>>(),
        )
        .buffered(SEARCH_SEGMENT_FANOUT)
        .try_collect::<Vec<_>>()
        .await?;

        let mut every_segment_exhausted = true;
        for values in per_segment {
            every_segment_exhausted &= values.len() < crate::text::PREFIX_EXPANSION_LIMIT;
            for value in values {
                prefix_terms.insert(value);
                if prefix_terms.len() > term_cap {
                    return Ok(None);
                }
            }
        }
        if prefix_terms.len() < crate::text::PREFIX_EXPANSION_LIMIT && !every_segment_exhausted {
            // Per-segment expansion is intentionally capped at 64. When lists
            // overlap heavily, fewer than 64 distinct terms do not prove that
            // later terms are absent; exact flat expansion must finish it.
            return Ok(None);
        }
        let selected = prefix_terms
            .into_iter()
            .take(crate::text::PREFIX_EXPANSION_LIMIT)
            .collect::<Vec<_>>();
        // Up to 64 expanded terms, each itself a per-segment fan-out. Serially
        // that was the single longest dependent chain in the whole query.
        let expanded_df = futures::stream::iter(
            selected
                .iter()
                .map(|term| reconciled_text_df(generation, term))
                .collect::<Vec<_>>(),
        )
        .buffered(SEARCH_SEGMENT_FANOUT)
        .try_collect::<Vec<_>>()
        .await?;

        for (term, df) in selected.into_iter().zip(expanded_df) {
            let Some(df) = df else {
                return Ok(None);
            };
            if df == 0 {
                // A lexically early term present only in stale/suppressed
                // versions must not consume one of the current-vocabulary
                // prefix slots. FT4 has no continuation token yet, so fall
                // back rather than silently under-fill the expansion.
                return Ok(None);
            }
            terms.insert(term);
            if terms.len() > term_cap {
                return Ok(None);
            }
        }
    }
    expanded.terms = terms.into_iter().collect();
    Ok(Some(expanded))
}

#[cfg(feature = "text-index")]
async fn reconciled_text_df(generation: &OpenedTextGeneration, term: &str) -> Result<Option<u64>> {
    use futures::{StreamExt, TryStreamExt};

    // Eager Vec for the same #102211 reason as `probe_winners`.
    let contributions = futures::stream::iter(
        generation
            .segments
            .iter()
            .map(|segment| segment.reader.term_delta_df(term))
            .collect::<Vec<_>>(),
    )
    .buffered(SEARCH_SEGMENT_FANOUT)
    .try_collect::<Vec<_>>()
    .await?;

    let mut frequency = 0u64;
    // Signed contributions must still be applied base-then-delta, in order.
    for (segment, contribution) in generation.segments.iter().zip(contributions) {
        let Some(next) = apply_term_df(segment.segment.role, frequency, contribution) else {
            return Ok(None);
        };
        frequency = next;
    }
    if frequency > generation.document_count {
        return Ok(None);
    }
    Ok(Some(frequency))
}

#[cfg(feature = "text-index")]
async fn global_text_stats(
    generation: &OpenedTextGeneration,
    query: &crate::text::TextQuery,
) -> Result<Option<crate::sst::text::v4::TextV4GlobalStats>> {
    let terms = query
        .base_terms()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    use futures::{StreamExt, TryStreamExt};

    // Term x segment is a fully independent grid; only the per-term fold is
    // ordered. Resolving it serially is what made a five-term cold query pay
    // dozens of dependent round trips before scoring a single document.
    let frequencies = futures::stream::iter(
        terms
            .iter()
            .map(|term| reconciled_text_df(generation, term))
            .collect::<Vec<_>>(),
    )
    .buffered(SEARCH_SEGMENT_FANOUT)
    .try_collect::<Vec<_>>()
    .await?;

    let mut document_frequency = BTreeMap::new();
    for (term, frequency) in terms.into_iter().zip(frequencies) {
        let Some(frequency) = frequency else {
            return Ok(None);
        };
        document_frequency.insert(term, frequency);
    }
    Ok(Some(crate::sst::text::v4::TextV4GlobalStats {
        document_count: generation.document_count,
        total_document_len: generation.total_document_len,
        document_frequency,
    }))
}

#[cfg(feature = "text-index")]
async fn text_generation_contains_live(
    generation: &OpenedTextGeneration,
    node_ids: &[[u8; 16]],
) -> Result<Option<bool>> {
    for chunk in node_ids.chunks(WINNER_PROBE_BATCH) {
        let winners = probe_winners(
            generation
                .segments
                .iter()
                .rev()
                .map(|segment| segment.reader.version_reader()),
            chunk,
        )
        .await?;
        if winners.conflict {
            return Ok(None);
        }
        if winners
            .records
            .into_iter()
            .flatten()
            .any(|record| matches!(record.operation, SearchVersionOperation::Live { .. }))
        {
            return Ok(Some(true));
        }
    }
    Ok(Some(false))
}

#[cfg(feature = "text-index")]
async fn reconcile_text_candidates(
    generation: &OpenedTextGeneration,
    candidates: &[TextCandidate],
) -> Result<Option<Vec<TextCandidate>>> {
    let mut accepted = Vec::with_capacity(candidates.len());
    for chunk in candidates.chunks(WINNER_PROBE_BATCH) {
        crate::cancel::check()?;
        let node_ids = chunk
            .iter()
            .map(|candidate| candidate.node_id)
            .collect::<Vec<_>>();
        let winners = probe_winners(
            generation
                .segments
                .iter()
                .rev()
                .map(|segment| segment.reader.version_reader()),
            &node_ids,
        )
        .await?;
        if winners.conflict {
            tracing::warn!(
                index = %generation.state.index_name,
                generation = %generation.state.generation_id,
                "Search-LSM equal-LSN text winner conflict; falling back"
            );
            return Ok(None);
        }
        for (candidate, winner) in chunk.iter().copied().zip(winners.records) {
            if versioned_candidate_wins(
                candidate.node_id,
                candidate.lsn,
                candidate.fingerprint,
                winner,
            ) {
                accepted.push(candidate);
            }
        }
    }
    Ok(Some(accepted))
}

#[cfg(feature = "text-index")]
pub(super) async fn text_search(
    snapshot: &Snapshot<'_>,
    index_name: &str,
    query: &crate::text::TextQuery,
    k: Option<usize>,
    groups: &[(String, Vec<Value>)],
    dirty_ids: &[[u8; 16]],
) -> Result<ActiveSearch<Vec<(NodeId, f64)>>> {
    let generation = match open_text_generation(snapshot, index_name).await? {
        ActiveSearch::NotActive => return Ok(ActiveSearch::NotActive),
        ActiveSearch::Unavailable => return Ok(ActiveSearch::Unavailable),
        ActiveSearch::Ready(generation) => generation,
    };
    if !dirty_ids.is_empty() {
        match text_generation_contains_live(&generation, dirty_ids).await {
            Ok(Some(false)) => {}
            Ok(Some(true) | None) => return Ok(ActiveSearch::Unavailable),
            Err(error) if optional_accelerator_fallback(&error) => {
                return Ok(ActiveSearch::Unavailable);
            }
            Err(error) => return Err(error),
        }
    }
    let Some(canonical_groups) = canonical_search_filter_groups(groups) else {
        return Ok(ActiveSearch::Unavailable);
    };
    if groups.iter().any(|(property, _)| {
        generation
            .segments
            .iter()
            .any(|segment| !segment.supports_filter_property(property))
    }) {
        return Ok(ActiveSearch::Unavailable);
    }
    if query.is_empty() || k == Some(0) || generation.document_count == 0 {
        return Ok(ActiveSearch::Ready(Vec::new()));
    }
    let expanded = match globally_expanded_text_query(&generation, query).await {
        Ok(Some(query)) => query,
        Ok(None) => return Ok(ActiveSearch::Unavailable),
        Err(error) if optional_accelerator_fallback(&error) => {
            return Ok(ActiveSearch::Unavailable);
        }
        Err(error) => return Err(error),
    };
    if expanded.is_empty() {
        return Ok(ActiveSearch::Ready(Vec::new()));
    }
    let global = match global_text_stats(&generation, &expanded).await {
        Ok(Some(stats)) => stats,
        Ok(None) => return Ok(ActiveSearch::Unavailable),
        Err(error) if optional_accelerator_fallback(&error) => {
            return Ok(ActiveSearch::Unavailable);
        }
        Err(error) => return Err(error),
    };
    let result_cap = crate::search_workspace::search_max_text_result_hits();
    let segment_limit_cap = result_cap.saturating_add(1);
    let retained = match k {
        Some(limit) => limit.min(usize::try_from(generation.document_count).unwrap_or(usize::MAX)),
        None => segment_limit_cap,
    };
    if retained == 0 {
        return Ok(ActiveSearch::Ready(Vec::new()));
    }
    if retained > segment_limit_cap {
        tracing::debug!(
            index = index_name,
            retained,
            segment_limit_cap,
            "Search-LSM FT4 coordinator result exceeds its bounded materialisation cap"
        );
        return Ok(ActiveSearch::Unavailable);
    }
    if let Some(base) = generation.authoritative_single_base(dirty_ids) {
        let result = match base
            .reader
            .search_query_base_block_max_exact(&expanded, &global, retained, &canonical_groups)
            .await
        {
            Ok(result) => result,
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    index = index_name,
                    segment = %base.segment.sst_id,
                    error = %error,
                    "authoritative FT4 base fast path failed; falling back atomically"
                );
                return Ok(ActiveSearch::Unavailable);
            }
            Err(error) => return Err(error),
        };
        if result.applied_filter_groups != canonical_groups.len()
            && !(result.hits.is_empty() && result.applied_filter_groups == 0)
        {
            return Ok(ActiveSearch::Unavailable);
        }
        if k.is_none() && result.hits.len() > result_cap {
            return Err(Error::SearchResultLimitExceeded {
                index_kind: "full-text",
                estimated_bytes: result.hits.len().saturating_mul(
                    crate::search_workspace::MATERIALISED_TEXT_RESULT_BYTES_PER_HIT,
                ),
                limit_bytes: crate::search_workspace::search_max_result_bytes(),
            });
        }
        let postings_decoded = result.postings_decoded;
        let posting_blocks_fetched = result.posting_blocks_fetched;
        let posting_blocks_skipped = result.posting_blocks_skipped;
        let block_max_pruning = result.block_max_pruning;
        let block_max_fallback = result.block_max_fallback.unwrap_or("none");
        let hits = result
            .hits
            .into_iter()
            .map(|hit| (NodeId(Uuid::from_bytes(hit.node_id)), hit.score))
            .collect::<Vec<_>>();
        tracing::debug!(
            index = index_name,
            generation = %generation.state.generation_id,
            documents = generation.document_count,
            returned = hits.len(),
            postings_decoded,
            posting_blocks_fetched,
            posting_blocks_skipped,
            block_max_pruning,
            block_max_fallback,
            "Search-LSM text query used authoritative single-base fast path"
        );
        return Ok(ActiveSearch::Ready(hits));
    }
    let mut ranked = BoundedTextResults::new(retained);
    for segment in &generation.segments {
        let local_documents =
            usize::try_from(segment.reader.live_document_count()).unwrap_or(usize::MAX);
        if local_documents == 0 {
            continue;
        }
        // Asking every segment for its complete match set made a `LIMIT 10`
        // query materialise hundreds of thousands of hits per segment. Scores
        // come from reconciled global statistics, so they are comparable across
        // segments: once `retained` of a segment's highest-scoring rows survive
        // reconciliation, everything it did not return scores no higher and
        // cannot enter the global top-k. Ties at the boundary resolve
        // arbitrarily, as they already do inside the heap.
        let segment_ceiling = local_documents.min(segment_limit_cap).max(1);
        let mut segment_limit = retained
            .saturating_mul(TEXT_SEGMENT_OVERFETCH)
            .min(segment_ceiling)
            .max(1);
        loop {
            let result = match segment
                .reader
                .search_query_exact(&expanded, &global, segment_limit, &canonical_groups)
                .await
            {
                Ok(result) => result,
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        index = index_name,
                        segment = %segment.segment.sst_id,
                        error = %error,
                        "Search-LSM FT4 query failed; falling back atomically"
                    );
                    return Ok(ActiveSearch::Unavailable);
                }
                Err(error) => return Err(error),
            };
            if result.applied_filter_groups != canonical_groups.len()
                && !(result.hits.is_empty() && result.applied_filter_groups == 0)
            {
                return Ok(ActiveSearch::Unavailable);
            }
            let matching_exhausted =
                local_documents <= segment_limit || result.hits.len() < segment_limit;
            let candidates = result
                .hits
                .into_iter()
                .map(|hit| TextCandidate {
                    node_id: hit.node_id,
                    lsn: hit.lsn,
                    fingerprint: hit.payload_fingerprint,
                    score: hit.score,
                })
                .collect::<Vec<_>>();
            let accepted = match reconcile_text_candidates(&generation, &candidates).await {
                Ok(Some(accepted)) => accepted,
                Ok(None) => return Ok(ActiveSearch::Unavailable),
                Err(error) if optional_accelerator_fallback(&error) => {
                    return Ok(ActiveSearch::Unavailable);
                }
                Err(error) => return Err(error),
            };
            let accepted_len = accepted.len();
            // Re-pushing a superset is safe: the heap deduplicates by node id.
            for candidate in accepted {
                ranked.push(candidate);
            }
            if accepted_len >= retained || matching_exhausted {
                break;
            }
            if segment_limit >= segment_ceiling {
                tracing::debug!(
                    index = index_name,
                    segment = %segment.segment.sst_id,
                    segment_limit,
                    "Search-LSM FT4 reconciliation exceeds its bounded result cap"
                );
                return Ok(ActiveSearch::Unavailable);
            }
            segment_limit = segment_limit.saturating_mul(4).min(segment_ceiling);
        }
    }
    if k.is_none() && (ranked.evicted || ranked.heap.len() > result_cap) {
        return Err(Error::SearchResultLimitExceeded {
            index_kind: "full-text",
            estimated_bytes: ranked
                .heap
                .len()
                .max(result_cap.saturating_add(1))
                .saturating_mul(crate::search_workspace::MATERIALISED_TEXT_RESULT_BYTES_PER_HIT),
            limit_bytes: crate::search_workspace::search_max_result_bytes(),
        });
    }
    let hits = ranked.finish();
    tracing::debug!(
        index = index_name,
        generation = %generation.state.generation_id,
        segments = generation.segments.len(),
        documents = generation.document_count,
        returned = hits.len(),
        "Search-LSM text query reconciled"
    );
    Ok(ActiveSearch::Ready(hits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    use object_store::memory::InMemory;
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    use object_store::{ObjectStore, ObjectStoreExt};

    #[cfg(feature = "vector-index")]
    struct VectorFixture {
        loaded: crate::manifest::LoadedManifest,
        store: Arc<dyn ObjectStore>,
        paths: crate::paths::NamespacePaths,
        barrier_path: String,
        first_segment_path: String,
        first_segment_len: usize,
    }

    #[cfg(feature = "vector-index")]
    fn vector_payload(vector: [f32; 2], vigente: bool) -> crate::sst::vector::v6::VectorV6Payload {
        crate::sst::vector::v6::VectorV6Payload {
            vector: vector.into(),
            filters: BTreeMap::from([("vigente".into(), SearchFilterValue::Bool(vigente))]),
        }
    }

    #[cfg(feature = "vector-index")]
    fn search_descriptor(
        segment: &SearchSegmentRef,
        path: String,
        size_bytes: u64,
        index: &crate::manifest::VectorIndexDescriptor,
    ) -> SstDescriptor {
        SstDescriptor {
            id: segment.sst_id,
            kind: crate::manifest::SstKind::VectorGraph,
            scope: index.name.clone(),
            level: crate::manifest::SstLevel(0),
            path,
            size_bytes,
            row_count: segment.live_payload_count,
            created_at: chrono::Utc::now(),
            min_key: [0; 16],
            max_key: [0xFF; 16],
            min_lsn: segment.min_lsn,
            max_lsn: segment.max_lsn,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: Vec::new(),
            kind_specific: crate::manifest::KindSpecificStats::VectorGraph {
                dim: index.dim,
                metric: "cosine".into(),
                point_count: segment.live_payload_count,
                r: index.r,
                l_build: index.l_build,
                alpha: index.alpha,
                entry_medoid: 0,
            },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        }
    }

    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    async fn put_relative(
        store: &Arc<dyn ObjectStore>,
        paths: &crate::paths::NamespacePaths,
        relative: &str,
        body: bytes::Bytes,
    ) {
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), relative);
        store.put(&Path::from(absolute), body.into()).await.unwrap();
    }

    #[cfg(feature = "vector-index")]
    async fn vector_fixture() -> VectorFixture {
        vector_fixture_with_conflict(false).await
    }

    #[cfg(feature = "vector-index")]
    async fn vector_fixture_with_conflict(equal_lsn_conflict: bool) -> VectorFixture {
        use crate::manifest::{
            LoadedManifest, Manifest, ManifestPointer, VectorIndexDescriptor, VectorQuantization,
        };
        use crate::search_lsm::{
            encode_search_barrier, search_barrier_descriptor, vector_catalog_signature,
            SearchEventRange, SearchLsmStatus,
        };
        use crate::sst::vector::v6::{
            build_delta_v6, VectorV6BuildContext, VectorV6BuildOptions, VectorV6Mutation,
        };

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = crate::paths::NamespacePaths::new(
            "tests",
            namidb_core::NamespaceId::new("search-lsm-vector-read").unwrap(),
        );
        let index = VectorIndexDescriptor {
            name: "law_vec".into(),
            label: "Articulo".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        };
        let writer_id = Uuid::from_u128(700);
        let mut manifest = Manifest::empty(crate::fence::Epoch::ZERO, writer_id);
        manifest.vector_indexes.push(index.clone());
        let mut state = SearchLsmState {
            index_name: index.name.clone(),
            kind: SearchLsmKind::Vector,
            catalog_signature: vector_catalog_signature(&manifest, &index),
            generation_id: Uuid::from_u128(701),
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        };
        let id = |value| *Uuid::from_u128(value).as_bytes();
        let a0 = vector_payload([1.0, 0.0], true);
        let b0 = vector_payload([0.95, 0.05], false);
        let c0 = vector_payload([0.0, 1.0], true);
        let a1 = vector_payload([-1.0, 0.0], true);
        let d0 = vector_payload([0.8, 0.2], true);
        let c1 = vector_payload([0.0, 1.0], false);
        let e0 = vector_payload([0.7, 0.3], true);
        let mutation_sets = vec![
            vec![
                VectorV6Mutation {
                    node_id: id(1),
                    lsn: 1,
                    before: None,
                    after: Some(a0.clone()),
                },
                VectorV6Mutation {
                    node_id: id(2),
                    lsn: 2,
                    before: None,
                    after: Some(b0.clone()),
                },
                VectorV6Mutation {
                    node_id: id(3),
                    lsn: 3,
                    before: None,
                    after: Some(c0.clone()),
                },
            ],
            vec![
                VectorV6Mutation {
                    node_id: id(1),
                    lsn: if equal_lsn_conflict { 1 } else { 4 },
                    before: Some(a0),
                    after: Some(a1),
                },
                VectorV6Mutation {
                    node_id: id(2),
                    lsn: 5,
                    before: Some(b0),
                    after: None,
                },
                VectorV6Mutation {
                    node_id: id(4),
                    lsn: 6,
                    before: None,
                    after: Some(d0),
                },
            ],
            vec![
                VectorV6Mutation {
                    node_id: id(3),
                    lsn: 7,
                    before: Some(c0),
                    after: Some(c1),
                },
                VectorV6Mutation {
                    node_id: id(5),
                    lsn: 8,
                    before: None,
                    after: Some(e0),
                },
            ],
        ];

        let mut segments = Vec::new();
        let mut descriptors = Vec::new();
        let mut first_segment_path = String::new();
        let mut first_segment_len = 0usize;
        for (ordinal, mutations) in mutation_sets.into_iter().enumerate() {
            let sst_id = Uuid::from_u128(710 + ordinal as u128);
            let artifact = build_delta_v6(
                &state,
                &index,
                VectorV6BuildContext {
                    sst_id,
                    event_ranges: vec![SearchEventRange::new(ordinal as u64, ordinal as u64 + 1)],
                    complete_filter_properties: vec!["vigente".into()],
                },
                mutations,
                VectorV6BuildOptions {
                    rows_per_page: 2,
                    compression_level: 1,
                },
            )
            .unwrap()
            .unwrap();
            let relative = format!("sst/L0/{sst_id}.vg");
            if ordinal == 0 {
                first_segment_path = relative.clone();
                first_segment_len = artifact.body.len();
            }
            put_relative(&store, &paths, &relative, artifact.body.clone()).await;
            descriptors.push(search_descriptor(
                &artifact.output.segment,
                relative,
                artifact.body.len() as u64,
                &index,
            ));
            segments.push(artifact.output.segment);
        }
        let barrier_id = Uuid::from_u128(799);
        state.status = SearchLsmStatus::Active;
        state.next_event_seq = segments.len() as u64;
        state.segments = segments;
        state.compat_barrier_sst_id = Some(barrier_id);
        let barrier = encode_search_barrier(&state).unwrap();
        let barrier_path = format!("sst/L0/{barrier_id}.slb");
        put_relative(&store, &paths, &barrier_path, barrier.clone()).await;
        descriptors.push(search_barrier_descriptor(
            &state,
            barrier_id,
            crate::manifest::SstLevel(0),
            barrier_path.clone(),
            barrier.len() as u64,
        ));
        manifest.ssts = descriptors;
        manifest.search_lsm.push(state);
        let pointer = ManifestPointer {
            version: manifest.version,
            epoch: manifest.epoch,
            manifest_path: "manifest/v0.json".into(),
        };
        VectorFixture {
            loaded: LoadedManifest::new(pointer, None, None, manifest),
            store,
            paths,
            barrier_path,
            first_segment_path,
            first_segment_len,
        }
    }

    #[cfg(feature = "text-index")]
    struct TextFixture {
        loaded: crate::manifest::LoadedManifest,
        store: Arc<dyn ObjectStore>,
        paths: crate::paths::NamespacePaths,
    }

    #[cfg(feature = "text-index")]
    fn text_payload(text: &str, vigente: bool) -> crate::sst::text::v4::TextV4Payload {
        crate::sst::text::v4::TextV4Payload {
            text: text.into(),
            filters: BTreeMap::from([("vigente".into(), SearchFilterValue::Bool(vigente))]),
        }
    }

    #[cfg(feature = "text-index")]
    fn text_search_descriptor(
        segment: &SearchSegmentRef,
        path: String,
        size_bytes: u64,
        index_name: &str,
    ) -> SstDescriptor {
        SstDescriptor {
            id: segment.sst_id,
            kind: crate::manifest::SstKind::TextIndex,
            scope: index_name.into(),
            level: crate::manifest::SstLevel(0),
            path,
            size_bytes,
            row_count: segment.live_payload_count,
            created_at: chrono::Utc::now(),
            min_key: [0; 16],
            max_key: [0xFF; 16],
            min_lsn: segment.min_lsn,
            max_lsn: segment.max_lsn,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: Vec::new(),
            kind_specific: crate::manifest::KindSpecificStats::TextIndex {
                doc_count: segment.live_payload_count,
                term_count: 0,
                total_len: 0,
            },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        }
    }

    #[cfg(feature = "text-index")]
    async fn text_fixture() -> TextFixture {
        use crate::manifest::{LoadedManifest, Manifest, ManifestPointer, TextIndexDescriptor};
        use crate::search_lsm::{
            encode_search_barrier, search_barrier_descriptor, text_lsm_catalog_signature,
            SearchEventRange, SearchLsmStatus,
        };
        use crate::sst::text::v4::{
            build_delta_v4, TextV4BuildContext, TextV4BuildOptions, TextV4Mutation,
        };

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = crate::paths::NamespacePaths::new(
            "tests",
            namidb_core::NamespaceId::new("search-lsm-text-read").unwrap(),
        );
        let index =
            TextIndexDescriptor::new("law_text".into(), "Articulo".into(), vec!["texto".into()]);
        let writer_id = Uuid::from_u128(800);
        let mut manifest = Manifest::empty(crate::fence::Epoch::ZERO, writer_id);
        manifest.text_indexes.push(index.clone());
        let mut state = SearchLsmState {
            index_name: index.name.clone(),
            kind: SearchLsmKind::Text,
            catalog_signature: text_lsm_catalog_signature(&manifest, &index),
            generation_id: Uuid::from_u128(801),
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        };
        let id = |value| *Uuid::from_u128(value).as_bytes();
        let a0 = text_payload("aardvark alpha obsolete", true);
        let b0 = text_payload("alpha beta", true);
        let c0 = text_payload("gamma", false);
        let a1 = text_payload("omega", true);
        let d0 = text_payload("alpha delta", true);
        let c1 = text_payload("gamma", true);
        let e0 = text_payload("alpha epsilon", false);
        let mutation_sets = vec![
            vec![
                TextV4Mutation {
                    node_id: id(1),
                    lsn: 1,
                    before: None,
                    after: Some(a0.clone()),
                },
                TextV4Mutation {
                    node_id: id(2),
                    lsn: 2,
                    before: None,
                    after: Some(b0.clone()),
                },
                TextV4Mutation {
                    node_id: id(3),
                    lsn: 3,
                    before: None,
                    after: Some(c0.clone()),
                },
            ],
            vec![
                TextV4Mutation {
                    node_id: id(1),
                    lsn: 4,
                    before: Some(a0),
                    after: Some(a1),
                },
                TextV4Mutation {
                    node_id: id(2),
                    lsn: 5,
                    before: Some(b0),
                    after: None,
                },
                TextV4Mutation {
                    node_id: id(4),
                    lsn: 6,
                    before: None,
                    after: Some(d0),
                },
            ],
            vec![
                TextV4Mutation {
                    node_id: id(3),
                    lsn: 7,
                    before: Some(c0),
                    after: Some(c1),
                },
                TextV4Mutation {
                    node_id: id(5),
                    lsn: 8,
                    before: None,
                    after: Some(e0),
                },
            ],
        ];
        let mut segments = Vec::new();
        let mut descriptors = Vec::new();
        for (ordinal, mutations) in mutation_sets.into_iter().enumerate() {
            let sst_id = Uuid::from_u128(810 + ordinal as u128);
            let artifact = build_delta_v4(
                &state,
                TextV4BuildContext {
                    sst_id,
                    event_ranges: vec![SearchEventRange::new(ordinal as u64, ordinal as u64 + 1)],
                    complete_filter_properties: vec!["vigente".into()],
                },
                mutations,
                TextV4BuildOptions {
                    postings_per_block: 2,
                    terms_per_dictionary_block: 2,
                    compression_level: 1,
                },
            )
            .unwrap()
            .unwrap();
            let relative = format!("sst/L0/{sst_id}.ft");
            put_relative(&store, &paths, &relative, artifact.body.clone()).await;
            descriptors.push(text_search_descriptor(
                &artifact.output.segment,
                relative,
                artifact.body.len() as u64,
                &index.name,
            ));
            segments.push(artifact.output.segment);
        }
        let barrier_id = Uuid::from_u128(899);
        state.status = SearchLsmStatus::Active;
        state.next_event_seq = segments.len() as u64;
        state.segments = segments;
        state.compat_barrier_sst_id = Some(barrier_id);
        let barrier = encode_search_barrier(&state).unwrap();
        let barrier_path = format!("sst/L0/{barrier_id}.slb");
        put_relative(&store, &paths, &barrier_path, barrier.clone()).await;
        descriptors.push(search_barrier_descriptor(
            &state,
            barrier_id,
            crate::manifest::SstLevel(0),
            barrier_path,
            barrier.len() as u64,
        ));
        manifest.ssts = descriptors;
        manifest.search_lsm.push(state);
        let pointer = ManifestPointer {
            version: manifest.version,
            epoch: manifest.epoch,
            manifest_path: "manifest/v0.json".into(),
        };
        TextFixture {
            loaded: LoadedManifest::new(pointer, None, None, manifest),
            store,
            paths,
        }
    }

    #[cfg(feature = "text-index")]
    async fn authoritative_text_base_fixture() -> TextFixture {
        use crate::manifest::{LoadedManifest, Manifest, ManifestPointer, TextIndexDescriptor};
        use crate::search_lsm::{
            encode_search_barrier, search_barrier_descriptor, text_lsm_catalog_signature,
            SearchEventRange, SearchLsmStatus,
        };
        use crate::sst::text::v4::{
            build_base_v4, TextV4BuildContext, TextV4BuildOptions, TextV4Mutation,
        };

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = crate::paths::NamespacePaths::new(
            "tests",
            namidb_core::NamespaceId::new("search-lsm-text-base-read").unwrap(),
        );
        let index =
            TextIndexDescriptor::new("law_text".into(), "Articulo".into(), vec!["texto".into()]);
        let writer_id = Uuid::from_u128(900);
        let mut manifest = Manifest::empty(crate::fence::Epoch::ZERO, writer_id);
        manifest.text_indexes.push(index.clone());
        let mut state = SearchLsmState {
            index_name: index.name.clone(),
            kind: SearchLsmKind::Text,
            catalog_signature: text_lsm_catalog_signature(&manifest, &index),
            generation_id: Uuid::from_u128(901),
            status: SearchLsmStatus::Building,
            base_frontier: Some(1),
            next_event_seq: 1,
            ..SearchLsmState::default()
        };
        let documents = (1..=32u128)
            .map(|value| TextV4Mutation {
                node_id: *Uuid::from_u128(value).as_bytes(),
                lsn: value as u64,
                before: None,
                after: Some(text_payload(
                    if value == 1 {
                        "rare rare rare common"
                    } else {
                        "common tail"
                    },
                    value % 2 == 1,
                )),
            })
            .collect::<Vec<_>>();
        let sst_id = Uuid::from_u128(910);
        let artifact = build_base_v4(
            &state,
            TextV4BuildContext {
                sst_id,
                event_ranges: vec![SearchEventRange::new(0, 1)],
                complete_filter_properties: vec!["vigente".into()],
            },
            documents,
            TextV4BuildOptions {
                postings_per_block: 2,
                terms_per_dictionary_block: 2,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        let relative = format!("sst/L0/{sst_id}.ft");
        put_relative(&store, &paths, &relative, artifact.body.clone()).await;
        let mut descriptors = vec![text_search_descriptor(
            &artifact.output.segment,
            relative,
            artifact.body.len() as u64,
            &index.name,
        )];
        state.status = SearchLsmStatus::Active;
        state.segments = vec![artifact.output.segment];
        let barrier_id = Uuid::from_u128(999);
        state.compat_barrier_sst_id = Some(barrier_id);
        let barrier = encode_search_barrier(&state).unwrap();
        let barrier_path = format!("sst/L0/{barrier_id}.slb");
        put_relative(&store, &paths, &barrier_path, barrier.clone()).await;
        descriptors.push(search_barrier_descriptor(
            &state,
            barrier_id,
            crate::manifest::SstLevel(0),
            barrier_path,
            barrier.len() as u64,
        ));
        manifest.ssts = descriptors;
        manifest.search_lsm.push(state);
        let pointer = ManifestPointer {
            version: manifest.version,
            epoch: manifest.epoch,
            manifest_path: "manifest/v0.json".into(),
        };
        TextFixture {
            loaded: LoadedManifest::new(pointer, None, None, manifest),
            store,
            paths,
        }
    }

    #[test]
    fn checked_statistics_reject_wrong_modes_and_underflow() {
        assert_eq!(
            apply_stat(SearchSegmentRole::Base, 99, SearchStatValue::Absolute(7)),
            Some(7)
        );
        assert_eq!(
            apply_stat(SearchSegmentRole::Delta, 7, SearchStatValue::Delta(-3)),
            Some(4)
        );
        assert_eq!(
            apply_stat(SearchSegmentRole::Delta, 2, SearchStatValue::Delta(-3)),
            None
        );
        assert_eq!(
            apply_stat(SearchSegmentRole::Delta, 2, SearchStatValue::Absolute(3)),
            None
        );
    }

    #[test]
    fn exact_candidate_match_requires_live_winner() {
        let id = *Uuid::from_u128(1).as_bytes();
        assert!(versioned_candidate_wins(
            id,
            7,
            9,
            Some(SearchVersionRecord::live(id, 7, 9, 0))
        ));
        assert!(!versioned_candidate_wins(
            id,
            7,
            9,
            Some(SearchVersionRecord::suppress(id, 7, 9))
        ));
        assert!(!versioned_candidate_wins(
            id,
            7,
            9,
            Some(SearchVersionRecord::live(id, 8, 9, 0))
        ));
    }

    #[test]
    fn winner_oracle_prefers_lsn_and_rejects_semantic_ties() {
        let id = *Uuid::from_u128(1).as_bytes();
        let mut winner = None;
        assert!(!merge_winner_record(
            &mut winner,
            SearchVersionRecord::live(id, 4, 11, 0)
        ));
        assert!(!merge_winner_record(
            &mut winner,
            SearchVersionRecord::suppress(id, 9, 17)
        ));
        assert!(matches!(
            winner,
            Some(SearchVersionRecord {
                lsn: 9,
                operation: SearchVersionOperation::Suppress,
                ..
            })
        ));
        // The segment-local payload ordinal is deliberately irrelevant when
        // the canonical payload fingerprint and operation agree.
        let mut same = Some(SearchVersionRecord::live(id, 12, 21, 3));
        assert!(!merge_winner_record(
            &mut same,
            SearchVersionRecord::live(id, 12, 21, 99)
        ));
        assert!(merge_winner_record(
            &mut same,
            SearchVersionRecord::suppress(id, 12, 21)
        ));
        assert!(merge_winner_record(
            &mut same,
            SearchVersionRecord::live(id, 12, 22, 3)
        ));
    }

    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn three_vg6_segments_reconcile_update_delete_filter_and_k_above_live() {
        let fixture = vector_fixture().await;
        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(fixture.loaded, &view, fixture.store, fixture.paths);
        let (hits, point_count) = snapshot
            .try_vector_search_with_point_count("law_vec", &[1.0, 0.0], 10, 64)
            .await
            .unwrap()
            .expect("valid active VG6 generation");
        assert_eq!(point_count, 4);
        assert_eq!(
            hits.iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
            [4u128, 5, 3, 1]
                .into_iter()
                .map(Uuid::from_u128)
                .collect::<Vec<_>>(),
            "stale high-score A and deleted B must not survive the winner oracle"
        );
        assert!(hits.windows(2).all(|pair| pair[0].1 >= pair[1].1));

        let filtered = snapshot
            .try_vector_search_filter_groups(
                "law_vec",
                &[1.0, 0.0],
                10,
                64,
                &[("vigente".into(), vec![Value::Bool(true)])],
            )
            .await
            .unwrap()
            .expect("valid filtered generation");
        let super::super::VectorFilterSearch::Applied {
            hits,
            point_count,
            eligible_count,
        } = filtered
        else {
            panic!("complete native filter must be applied");
        };
        assert_eq!(point_count, 4);
        assert_eq!(eligible_count, 3);
        assert_eq!(
            hits.iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
            [4u128, 5, 1]
                .into_iter()
                .map(Uuid::from_u128)
                .collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn equal_lsn_payload_conflict_falls_back_atomically() {
        let fixture = vector_fixture_with_conflict(true).await;
        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(fixture.loaded, &view, fixture.store, fixture.paths);

        assert!(snapshot
            .try_vector_search("law_vec", &[1.0, 0.0], 10, 64)
            .await
            .unwrap()
            .is_none());
    }

    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn missing_barrier_missing_segment_and_corruption_fall_back_atomically() {
        let missing = vector_fixture().await;
        let barrier_absolute = format!(
            "{}/{}",
            missing.paths.namespace_prefix().as_ref(),
            missing.barrier_path
        );
        missing
            .store
            .delete(&Path::from(barrier_absolute))
            .await
            .unwrap();
        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(missing.loaded, &view, missing.store, missing.paths);
        assert!(snapshot
            .try_vector_search("law_vec", &[1.0, 0.0], 2, 64)
            .await
            .unwrap()
            .is_none());

        let missing_segment = vector_fixture().await;
        let segment_absolute = format!(
            "{}/{}",
            missing_segment.paths.namespace_prefix().as_ref(),
            missing_segment.first_segment_path
        );
        missing_segment
            .store
            .delete(&Path::from(segment_absolute))
            .await
            .unwrap();
        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(
            missing_segment.loaded,
            &view,
            missing_segment.store,
            missing_segment.paths,
        );
        assert!(snapshot
            .try_vector_search("law_vec", &[1.0, 0.0], 2, 64)
            .await
            .unwrap()
            .is_none());

        let corrupt = vector_fixture().await;
        let segment_absolute = format!(
            "{}/{}",
            corrupt.paths.namespace_prefix().as_ref(),
            corrupt.first_segment_path
        );
        corrupt
            .store
            .put(
                &Path::from(segment_absolute),
                bytes::Bytes::from(vec![0u8; corrupt.first_segment_len]).into(),
            )
            .await
            .unwrap();
        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(corrupt.loaded, &view, corrupt.store, corrupt.paths);
        assert!(snapshot
            .try_vector_search("law_vec", &[1.0, 0.0], 2, 64)
            .await
            .unwrap()
            .is_none());
    }

    /// Plan item 4 (25 TB readiness): the bounded over-fetch loop must widen
    /// when reconciliation supersedes an entire first batch. An update-heavy
    /// corpus routinely produces segments whose top-scoring candidates are
    /// all stale; returning a silent short page instead of refilling is the
    /// regression this pins.
    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn refill_loop_widens_past_a_fully_superseded_first_batch() {
        use crate::manifest::{LoadedManifest, Manifest, ManifestPointer, TextIndexDescriptor};
        use crate::search_lsm::{
            encode_search_barrier, search_barrier_descriptor, text_lsm_catalog_signature,
            SearchEventRange, SearchLsmStatus,
        };
        use crate::sst::text::v4::{
            build_delta_v4, TextV4BuildContext, TextV4BuildOptions, TextV4Mutation,
        };

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = crate::paths::NamespacePaths::new(
            "tests",
            namidb_core::NamespaceId::new("search-lsm-text-refill").unwrap(),
        );
        let index =
            TextIndexDescriptor::new("law_text".into(), "Articulo".into(), vec!["texto".into()]);
        let writer_id = Uuid::from_u128(950);
        let mut manifest = Manifest::empty(crate::fence::Epoch::ZERO, writer_id);
        manifest.text_indexes.push(index.clone());
        let mut state = SearchLsmState {
            index_name: index.name.clone(),
            kind: SearchLsmKind::Text,
            catalog_signature: text_lsm_catalog_signature(&manifest, &index),
            generation_id: Uuid::from_u128(951),
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        };
        let id = |value| *Uuid::from_u128(value).as_bytes();

        // Twelve `alpha` docs. The four shortest score highest under BM25
        // length normalisation — exactly the over-fetch batch for k=1 — and
        // every one of them is deleted by the second segment.
        let mut first = Vec::new();
        let mut doomed = Vec::new();
        for ordinal in 0..12u128 {
            let text = if ordinal < 4 {
                "alpha".to_string()
            } else if ordinal == 4 {
                "alpha uno".to_string()
            } else {
                format!("alpha filler{ordinal} more words")
            };
            let payload = text_payload(&text, true);
            first.push(TextV4Mutation {
                node_id: id(ordinal + 1),
                lsn: ordinal as u64 + 1,
                before: None,
                after: Some(payload.clone()),
            });
            if ordinal < 4 {
                doomed.push((ordinal, payload));
            }
        }
        let second = doomed
            .into_iter()
            .map(|(ordinal, payload)| TextV4Mutation {
                node_id: id(ordinal + 1),
                lsn: 20 + ordinal as u64,
                before: Some(payload),
                after: None,
            })
            .collect::<Vec<_>>();

        let mut segments = Vec::new();
        let mut descriptors = Vec::new();
        for (ordinal, mutations) in [first, second].into_iter().enumerate() {
            let sst_id = Uuid::from_u128(960 + ordinal as u128);
            let artifact = build_delta_v4(
                &state,
                TextV4BuildContext {
                    sst_id,
                    event_ranges: vec![SearchEventRange::new(ordinal as u64, ordinal as u64 + 1)],
                    complete_filter_properties: vec!["vigente".into()],
                },
                mutations,
                TextV4BuildOptions {
                    postings_per_block: 2,
                    terms_per_dictionary_block: 2,
                    compression_level: 1,
                },
            )
            .unwrap()
            .unwrap();
            let relative = format!("sst/L0/{sst_id}.ft");
            put_relative(&store, &paths, &relative, artifact.body.clone()).await;
            descriptors.push(text_search_descriptor(
                &artifact.output.segment,
                relative,
                artifact.body.len() as u64,
                &index.name,
            ));
            segments.push(artifact.output.segment);
        }
        let barrier_id = Uuid::from_u128(999);
        state.status = SearchLsmStatus::Active;
        state.next_event_seq = segments.len() as u64;
        state.segments = segments;
        state.compat_barrier_sst_id = Some(barrier_id);
        let barrier = encode_search_barrier(&state).unwrap();
        let barrier_path = format!("sst/L0/{barrier_id}.slb");
        put_relative(&store, &paths, &barrier_path, barrier.clone()).await;
        descriptors.push(search_barrier_descriptor(
            &state,
            barrier_id,
            crate::manifest::SstLevel(0),
            barrier_path,
            barrier.len() as u64,
        ));
        manifest.ssts = descriptors;
        manifest.search_lsm.push(state);
        let pointer = ManifestPointer {
            version: manifest.version,
            epoch: manifest.epoch,
            manifest_path: "manifest/v0.json".into(),
        };
        let loaded = LoadedManifest::new(pointer, None, None, manifest);

        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(loaded, &view, store, paths);
        let hits = snapshot
            .text_search(
                "law_text",
                "Articulo",
                &crate::text::parse_query("alpha"),
                Some(1),
            )
            .await
            .unwrap()
            .expect("the generation must serve, not fall back");
        assert_eq!(
            hits.len(),
            1,
            "a fully superseded first batch must refill, not return short"
        );
        assert_eq!(
            hits[0].0 .0,
            Uuid::from_u128(5),
            "the top surviving document must win after the refill"
        );
    }

    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn three_ft4_segments_use_global_stats_winners_prefix_and_native_filter() {
        let fixture = text_fixture().await;
        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(fixture.loaded, &view, fixture.store, fixture.paths);
        let query = crate::text::parse_query("alpha");
        let hits = snapshot
            .text_search("law_text", "Articulo", &query, Some(10))
            .await
            .unwrap()
            .expect("valid active FT4 generation");
        assert_eq!(
            hits.iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
            [4u128, 5]
                .into_iter()
                .map(Uuid::from_u128)
                .collect::<Vec<_>>(),
            "stale alpha versions and the deleted document must be suppressed"
        );
        let expected = crate::text::bm25_term_score(crate::text::bm25_idf(4, 2), 1, 2, 1.5);
        assert!(
            hits.iter()
                .all(|(_, score)| score.to_bits() == expected.to_bits()),
            "all FT4 segments must score with reconciled global statistics"
        );

        let prefix = snapshot
            .text_search(
                "law_text",
                "Articulo",
                &crate::text::parse_query("alp*"),
                Some(10),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(prefix, hits);

        assert!(
            snapshot
                .text_search(
                    "law_text",
                    "Articulo",
                    &crate::text::parse_query("aard*"),
                    Some(10),
                )
                .await
                .unwrap()
                .is_none(),
            "a stale-only lexicographic prefix term must select the exact fallback"
        );

        let filtered = snapshot
            .text_search_filter_groups(
                "law_text",
                "Articulo",
                &query,
                Some(10),
                &[("vigente".into(), vec![Value::Bool(true)])],
            )
            .await
            .unwrap()
            .expect("complete FT4 filter");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0, NodeId(Uuid::from_u128(4)));
        assert_eq!(filtered[0].1.to_bits(), expected.to_bits());
    }

    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn authoritative_single_ft4_base_uses_exact_fast_path_only_when_clean() {
        let fixture = authoritative_text_base_fixture().await;
        let memtable = crate::memtable::Memtable::new();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(fixture.loaded, &view, fixture.store, fixture.paths);
        let mut generation = match open_text_generation(&snapshot, "law_text").await.unwrap() {
            ActiveSearch::Ready(generation) => generation,
            other => panic!("expected active authoritative base, found {other:?}"),
        };
        assert!(generation.authoritative_single_base(&[]).is_some());
        assert!(
            generation
                .authoritative_single_base(&[*Uuid::from_u128(500).as_bytes()])
                .is_none(),
            "any dirty NodeId disables the fast path even when absent from the base"
        );
        generation.state.next_event_seq += 1;
        assert!(
            generation.authoritative_single_base(&[]).is_none(),
            "frontier drift must disable the fast path"
        );
        drop(generation);

        let hits = snapshot
            .text_search(
                "law_text",
                "Articulo",
                &crate::text::parse_query("rare common"),
                Some(1),
            )
            .await
            .unwrap()
            .expect("authoritative base query must remain on the native path");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, NodeId(Uuid::from_u128(1)));

        let filtered = snapshot
            .text_search_filter_groups(
                "law_text",
                "Articulo",
                &crate::text::parse_query("common"),
                Some(3),
                &[("vigente".into(), vec![Value::Bool(true)])],
            )
            .await
            .unwrap()
            .expect("authoritative base filter must remain native");
        assert_eq!(filtered.len(), 3);
        assert!(filtered.iter().all(|(node, _)| node.0.as_u128() % 2 == 1));

        let all = snapshot
            .text_search(
                "law_text",
                "Articulo",
                &crate::text::parse_query("common"),
                None,
            )
            .await
            .unwrap()
            .expect("unbounded authoritative-base query must remain native");
        assert_eq!(all.len(), 32, "k=None must use the cap+1 exact path");
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn bounded_vector_heap_dedupes_and_respects_metric_orientation() {
        let candidate = |id, score| VectorCandidate {
            node_id: *Uuid::from_u128(id).as_bytes(),
            score,
            version: VectorCandidateVersion::V5Base,
        };
        let mut cosine = BoundedVectorResults::new(2, true);
        cosine.push(candidate(3, 0.5));
        cosine.push(candidate(1, 0.9));
        cosine.push(candidate(2, 0.7));
        cosine.push(candidate(1, 0.9));
        let cosine = cosine.finish();
        assert_eq!(
            cosine,
            vec![
                (NodeId(Uuid::from_u128(1)), 0.9),
                (NodeId(Uuid::from_u128(2)), 0.7)
            ]
        );

        let mut euclidean = BoundedVectorResults::new(2, false);
        euclidean.push(candidate(3, 3.0));
        euclidean.push(candidate(1, 1.0));
        euclidean.push(candidate(2, 2.0));
        assert_eq!(
            euclidean.finish(),
            vec![
                (NodeId(Uuid::from_u128(1)), 1.0),
                (NodeId(Uuid::from_u128(2)), 2.0)
            ]
        );
    }
}
