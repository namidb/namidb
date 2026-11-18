//! Physical Search-LSM base-compaction planning and CAS rebase.
//!
//! The expensive builders are deliberately separate from this pure contract:
//! a prepare captures one complete active prefix and uploads one immutable
//! full base, then install may replace that prefix only when the generation,
//! catalog, physical descriptors, coverage proof, and every captured list
//! prefix still match. Ordinary flushes append after those prefixes and are
//! preserved.
//!
//! Without `vector-index` or `text-index` no search body can be rebuilt, so the
//! builder half of this module is inert while the pure planning/CAS contract
//! still compiles and is still tested. That build alone waives the resulting
//! dead-code diagnostics.
#![cfg_attr(
    not(any(feature = "vector-index", feature = "text-index")),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)]

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use chrono::Utc;
use object_store::path::Path;
use object_store::ObjectStore;
// `head` lives on the 0.13 extension trait and is only reached by the search
// builders, so the import follows the same gate as its single call site.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
use object_store::ObjectStoreExt;
use uuid::Uuid;
use xxhash_rust::xxh3::Xxh3;

use namidb_core::{LabelDef, NodeId, Value};

use crate::error::{Error, Result};
use crate::manifest::{
    KindSpecificStats, LoadedManifest, Manifest, SstDescriptor, SstKind, SstLevel,
};
use crate::memtable::MemOp;
use crate::memtable::MemtableSnapshot;
use crate::paths::NamespacePaths;
use crate::search_lsm::{
    legacy_base_content_fingerprint, validate_search_lsm, CoverageDisposition, SearchEventRange,
    SearchLsmKind, SearchLsmState, SearchLsmStatus, SearchSegmentFormat, SearchSegmentPayload,
    SearchSegmentRef, SearchSegmentRole, SearchSegmentStats, SearchStatValue,
    MAX_ACTIVE_SEARCH_SEGMENTS,
};
use crate::sst::search_delta::{
    reconcile_node_versions, SearchFilterValue, SearchVersionOperation, SearchVersionRangeSource,
    SearchVersionRecord, SearchVersionTableReader,
};

const SEARCH_LSM_MAX_SEGMENTS_ENV: &str = "NAMIDB_SEARCH_LSM_MAX_SEGMENTS";
const SEARCH_LSM_COMPACT_SEGMENTS_ENV: &str = "NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS";
const SEARCH_LSM_BASE_COMPACT_BYTES_ENV: &str = "NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES";
const SEARCH_LSM_BASE_STALE_PERCENT_ENV: &str = "NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT";
const SEARCH_LSM_FORCE_BASE_COMPACTION_ENV: &str = "NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION";
const SEARCH_LSM_VECTOR_BASE_DELTA_ROWS_ENV: &str = "NAMIDB_SEARCH_LSM_VECTOR_BASE_DELTA_ROWS";
const SEARCH_LSM_VECTOR_BASE_MAX_INPUT_BYTES_ENV: &str =
    "NAMIDB_SEARCH_LSM_VECTOR_BASE_MAX_INPUT_BYTES";
/// Flush ceiling. Compaction is due long before it (see
/// [`default_delta_trigger`]); the room between the two absorbs the flushes
/// that land while a compaction runs, so a flush is not refused merely because
/// a merge is in progress.
pub(crate) const DEFAULT_SEARCH_LSM_MAX_SEGMENTS: usize = 24;
/// Similar-sized delta segments that make one routine DeltaRun.
const DEFAULT_SEARCH_LSM_TIER_SEGMENTS: usize = 4;
/// Exact (flat) vector rows a generation may hold in delta segments before
/// they are folded into its clustered base. Every query scans the deltas in
/// full, so this bounds the per-query cost of the unclustered tail.
const DEFAULT_SEARCH_LSM_VECTOR_BASE_DELTA_ROWS: u64 = 32 * 1024;
/// A generation whose newest delta is at least this old has stopped taking
/// writes: its remaining flat tail is folded into the base even below
/// [`DEFAULT_SEARCH_LSM_VECTOR_BASE_DELTA_ROWS`].
const VECTOR_BASE_IDLE_SECS: i64 = 60;
/// ... but only once the tail is worth a rebuild: at least this many rows and
/// at least 1/[`VECTOR_BASE_IDLE_BASE_FRACTION`] of the base, so a trickle of
/// writes cannot rebuild a large base for every few rows.
const VECTOR_BASE_IDLE_MIN_ROWS: u64 = 1024;
const VECTOR_BASE_IDLE_BASE_FRACTION: u64 = 64;
const DEFAULT_SEARCH_LSM_BASE_COMPACT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_SEARCH_LSM_BASE_STALE_PERCENT: u64 = 800;
const MIN_SEARCH_LSM_SEGMENTS: usize = 2;
/// Fetched-but-not-yet-activated Nodes bodies buffered ahead of the base
/// merge. Matches the DeltaRun pipeline's channel depth: enough to overlap
/// fetch latency with build CPU, small enough that lookahead residency stays
/// a constant, not a corpus fraction.
const SEARCH_BASE_FETCH_LOOKAHEAD: usize = 2;
const EMPTY_BASE_CLASSIFIER_VERSION: u16 = 2;
const EMPTY_DELTA_CLASSIFIER_VERSION: u16 = 3;

/// Physical rewrite shape. Milestone one builds an authoritative base prefix;
/// the explicit mode keeps the proof/install API ready for a bounded
/// delta-only run without conflating its suppression and statistics rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SearchCompactionMode {
    BasePrefix,
    DeltaRun,
}

/// Dual physical policy: bounded tail consolidation is the routine path,
/// while the corpus-wide base rebuild requires an explicit, bytes, or
/// accumulated-staleness trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SearchCompactionPolicy {
    pub(super) delta_trigger_segments: usize,
    pub(super) hard_segment_cap: usize,
    pub(super) base_compact_bytes: u64,
    pub(super) base_stale_percent: u64,
    pub(super) force_base: bool,
    /// Flat vector delta rows that schedule a clustered base rebuild.
    pub(super) vector_base_delta_rows: u64,
    /// Largest input (descriptor bytes) the opportunistic vector base rebuild
    /// may read: the generation's segments when it folds them, else the Nodes
    /// corpus, a large share of which a node scan holds in memory.
    pub(super) vector_base_max_input_bytes: u64,
}

impl SearchCompactionPolicy {
    pub(super) fn from_env() -> Result<Self> {
        let hard_segment_cap =
            parse_segment_env(SEARCH_LSM_MAX_SEGMENTS_ENV, DEFAULT_SEARCH_LSM_MAX_SEGMENTS)?;
        let default_trigger = default_delta_trigger(hard_segment_cap);
        let delta_trigger_segments =
            parse_segment_env(SEARCH_LSM_COMPACT_SEGMENTS_ENV, default_trigger)?
                .min(hard_segment_cap);
        let base_compact_bytes = parse_positive_u64_env(
            SEARCH_LSM_BASE_COMPACT_BYTES_ENV,
            DEFAULT_SEARCH_LSM_BASE_COMPACT_BYTES,
        )?;
        let base_stale_percent = parse_positive_u64_env(
            SEARCH_LSM_BASE_STALE_PERCENT_ENV,
            DEFAULT_SEARCH_LSM_BASE_STALE_PERCENT,
        )?;
        let force_base = parse_bool_env(SEARCH_LSM_FORCE_BASE_COMPACTION_ENV, false)?;
        let vector_base_delta_rows = parse_positive_u64_env(
            SEARCH_LSM_VECTOR_BASE_DELTA_ROWS_ENV,
            DEFAULT_SEARCH_LSM_VECTOR_BASE_DELTA_ROWS,
        )?;
        let vector_base_max_input_bytes =
            match std::env::var_os(SEARCH_LSM_VECTOR_BASE_MAX_INPUT_BYTES_ENV) {
                Some(_) => parse_positive_u64_env(SEARCH_LSM_VECTOR_BASE_MAX_INPUT_BYTES_ENV, 1)?,
                None => default_vector_base_max_input_bytes(),
            };
        Ok(Self {
            delta_trigger_segments,
            hard_segment_cap,
            base_compact_bytes,
            base_stale_percent,
            force_base,
            vector_base_delta_rows,
            vector_base_max_input_bytes,
        })
    }

    /// Scheduling cannot surface configuration errors through its boolean
    /// API. Keep it conservative and visible: malformed settings still wake
    /// the prepare phase, whose validated parser returns the actionable error.
    pub(super) fn for_scheduling() -> Self {
        match Self::from_env() {
            Ok(policy) => policy,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "invalid Search-LSM compaction policy; scheduling with conservative defaults"
                );
                Self {
                    delta_trigger_segments: default_delta_trigger(DEFAULT_SEARCH_LSM_MAX_SEGMENTS),
                    hard_segment_cap: DEFAULT_SEARCH_LSM_MAX_SEGMENTS,
                    base_compact_bytes: DEFAULT_SEARCH_LSM_BASE_COMPACT_BYTES,
                    base_stale_percent: DEFAULT_SEARCH_LSM_BASE_STALE_PERCENT,
                    force_base: false,
                    vector_base_delta_rows: DEFAULT_SEARCH_LSM_VECTOR_BASE_DELTA_ROWS,
                    vector_base_max_input_bytes: default_vector_base_max_input_bytes(),
                }
            }
        }
    }
}

fn default_delta_trigger(hard_segment_cap: usize) -> usize {
    hard_segment_cap
        .saturating_sub(2)
        .clamp(MIN_SEARCH_LSM_SEGMENTS, DEFAULT_SEARCH_LSM_TIER_SEGMENTS)
        .min(hard_segment_cap)
}

/// Number of segments at which a generation is under segment pressure: the
/// routine size-tiered choice is abandoned for the newest
/// `delta_trigger_segments` deltas whatever their sizes, so the count always
/// comes down well before the flush ceiling.
fn pressure_segments(policy: SearchCompactionPolicy) -> usize {
    policy
        .delta_trigger_segments
        .saturating_mul(2)
        .min(policy.hard_segment_cap.saturating_sub(2))
        .max(policy.delta_trigger_segments)
}

/// Most deltas one pressure-relief DeltaRun merges.
const MAX_PRESSURE_DELTA_RUN: usize = 16;

/// Deltas up to this size always join a tier.
const TIER_SMALL_DELTA_BYTES: u64 = 1024 * 1024;

/// Length of the newest run of similar-sized deltas (size-tiered selection).
///
/// Walking back from the newest delta, an older delta joins the run while it
/// is at most twice everything newer already in the run (or small). Equal
/// flushes therefore merge four at a time, their outputs merge with each other
/// once there are enough of the same size, and a large consolidated delta is
/// rewritten only when newer data has caught up with it: every row is
/// rewritten O(log(rows)) times instead of once per merge (the newest-suffix
/// rule rewrote every consolidated delta on every routine pass).
fn tiered_delta_run_len(state: &SearchLsmState, descriptors: &[SstDescriptor]) -> usize {
    let mut run = 0usize;
    let mut newer_bytes = 0u64;
    for (segment, descriptor) in state.segments.iter().zip(descriptors).rev() {
        if segment.role != SearchSegmentRole::Delta
            || segment.payload != SearchSegmentPayload::Complete
        {
            break;
        }
        let size = descriptor.size_bytes.max(1);
        // Similar-sized: no more than twice the newer ones combined. Small
        // deltas are cheap to rewrite and always tier together.
        if run > 0 && size > TIER_SMALL_DELTA_BYTES && size > newer_bytes.saturating_mul(2) {
            break;
        }
        newer_bytes = newer_bytes.saturating_add(size);
        run += 1;
    }
    run
}

/// Newest complete deltas, contiguous from the physical tail.
fn trailing_delta_len(state: &SearchLsmState) -> usize {
    state
        .segments
        .iter()
        .rev()
        .take_while(|segment| {
            segment.role == SearchSegmentRole::Delta
                && segment.payload == SearchSegmentPayload::Complete
        })
        .count()
}

/// Half of the process's cgroup memory limit, or unbounded without one. A
/// node-scan base rebuild streams the namespace's Nodes bodies through a k-way
/// merge whose resident inputs were measured at ~45% of their total bytes;
/// within half the limit that stays under a quarter of the process. A segment
/// merge streams its inputs, so the same cap on its (much smaller) segment
/// bytes is conservative.
fn default_vector_base_max_input_bytes() -> u64 {
    cgroup_memory_limit_bytes().map_or(u64::MAX, |limit| limit / 2)
}

/// Finite cgroup v2 (then v1) memory limit of this process, if any.
fn cgroup_memory_limit_bytes() -> Option<u64> {
    let own = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|groups| {
            groups
                .lines()
                .find_map(|line| line.strip_prefix("0::").map(str::to_owned))
        });
    let mut candidates = Vec::new();
    if let Some(own) = own {
        candidates.push(format!(
            "/sys/fs/cgroup{}/memory.max",
            own.trim_end_matches('/')
        ));
    }
    candidates.push("/sys/fs/cgroup/memory.max".to_owned());
    candidates.push("/sys/fs/cgroup/memory/memory.limit_in_bytes".to_owned());
    candidates.into_iter().find_map(|path| {
        let raw = std::fs::read_to_string(path).ok()?;
        let bytes = raw.trim().parse::<u64>().ok()?;
        // cgroup v1 reports "unlimited" as a huge page-aligned value.
        (bytes > 0 && bytes < (1u64 << 60)).then_some(bytes)
    })
}

/// Flat vector rows held in deltas: every query scans them exactly.
///
/// `input_bytes` is what the rebuild would read: the generation's own
/// segments when they can be merged ([`vector_segments_mergeable`]), the
/// namespace's Nodes corpus otherwise.
/// Whether the generation's vector base predates k-means splits: a cosine
/// or Euclidean base whose descriptor lacks the layout tag.
#[cfg(feature = "vector-index")]
fn vector_base_layout_stale(state: &SearchLsmState, descriptors: &[SstDescriptor]) -> bool {
    state
        .segments
        .iter()
        .zip(descriptors)
        .any(|(segment, descriptor)| {
            segment.role == SearchSegmentRole::Base
                && segment.payload == SearchSegmentPayload::Complete
                && matches!(
                    &descriptor.kind_specific,
                    KindSpecificStats::VectorGraph { metric, entry_medoid, .. }
                        if metric != "dot"
                            && *entry_medoid != crate::sst::vector::v5::VECTOR_V5_KMEANS_LAYOUT
                )
        })
}

#[cfg(not(feature = "vector-index"))]
fn vector_base_layout_stale(_: &SearchLsmState, _: &[SstDescriptor]) -> bool {
    false
}

fn vector_base_rebuild_due(
    state: &SearchLsmState,
    descriptors: &[SstDescriptor],
    policy: SearchCompactionPolicy,
    now: chrono::DateTime<Utc>,
    input_bytes: u64,
) -> Result<bool> {
    if state.kind != SearchLsmKind::Vector || input_bytes > policy.vector_base_max_input_bytes {
        return Ok(false);
    }
    let mut delta_rows = 0u64;
    let mut base_rows = 0u64;
    let mut newest_delta = None;
    for (segment, descriptor) in state.segments.iter().zip(descriptors) {
        match segment.role {
            SearchSegmentRole::Delta => {
                delta_rows = delta_rows
                    .checked_add(segment.mutation_count)
                    .ok_or_else(|| Error::invariant("Search-LSM delta row count overflows"))?;
                newest_delta = newest_delta.max(Some(descriptor.created_at));
            }
            SearchSegmentRole::Base => base_rows = segment.live_payload_count,
        }
    }
    // A base written before k-means splits is rebuilt once, from its own
    // segments when they fold, within the same input bound as any rebuild.
    if vector_base_layout_stale(state, descriptors) {
        return Ok(true);
    }
    let Some(newest_delta) = newest_delta else {
        return Ok(false);
    };
    if delta_rows >= policy.vector_base_delta_rows {
        return Ok(true);
    }
    let idle = now.signed_duration_since(newest_delta).num_seconds() >= VECTOR_BASE_IDLE_SECS;
    Ok(idle
        && delta_rows >= VECTOR_BASE_IDLE_MIN_ROWS
        && delta_rows >= base_rows / VECTOR_BASE_IDLE_BASE_FRACTION)
}

fn parse_positive_u64_env(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(value) => {
            let parsed = value.trim().parse::<u64>().map_err(|error| {
                Error::precondition(format!("{name} must be an exact positive integer: {error}"))
            })?;
            if parsed == 0 {
                return Err(Error::precondition(format!(
                    "{name} must be greater than zero"
                )));
            }
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::precondition(format!("{name} is not valid UTF-8")))
        }
    }
}

fn parse_bool_env(name: &str, default: bool) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(Error::precondition(format!(
                "{name} must be one of 1/0, true/false, yes/no, or on/off"
            ))),
        },
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::precondition(format!("{name} is not valid UTF-8")))
        }
    }
}

fn parse_segment_env(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => parse_segment_value(name, &value),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::precondition(format!("{name} is not valid UTF-8")))
        }
    }
}

fn parse_segment_value(name: &str, value: &str) -> Result<usize> {
    let parsed = value.trim().parse::<usize>().map_err(|error| {
        Error::precondition(format!("{name} must be an exact integer: {error}"))
    })?;
    if parsed < MIN_SEARCH_LSM_SEGMENTS {
        return Err(Error::precondition(format!(
            "{name} must be at least {MIN_SEARCH_LSM_SEGMENTS}"
        )));
    }
    Ok(parsed.min(MAX_ACTIVE_SEARCH_SEGMENTS))
}

/// Complete active prefix selected for one authoritative full-base rebuild.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SearchCompactionSelection {
    pub(super) mode: SearchCompactionMode,
    pub(super) captured_state: SearchLsmState,
    /// First captured segment replaced. DeltaRun always selects the physical
    /// suffix; BasePrefix begins at zero.
    pub(super) selected_segment_start: usize,
    pub(super) selected_descriptors: Vec<SstDescriptor>,
    /// Exact compressed event union replaced by the output.
    pub(super) selected_event_ranges: Vec<SearchEventRange>,
    pub(super) captured_coverage_digest: u64,
    pub(super) hard_segment_cap: usize,
}

impl SearchCompactionSelection {
    pub(super) fn selected_ids(&self) -> impl Iterator<Item = uuid::Uuid> + '_ {
        self.selected_descriptors
            .iter()
            .map(|descriptor| descriptor.id)
    }
}

/// One immutable full-base payload already uploaded by the prepare phase.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedSearchPayload {
    pub(super) descriptor: SstDescriptor,
    pub(super) segment: SearchSegmentRef,
}

/// Logical and physical counters retained for structured diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SearchCompactionBuildMetrics {
    pub(super) input_bytes: u64,
    pub(super) output_bytes: u64,
    /// Physical source version records examined.
    pub(super) input_rows: u64,
    /// Distinct NodeIds reconciled and point-read from the captured snapshot.
    pub(super) touched_rows: u64,
    pub(super) live_rows: u64,
    /// Peak builder-owned logical RAM, excluding shared object/range caches and
    /// the point-read batch owned by this coordinator.
    pub(super) peak_workspace_bytes: u64,
    pub(super) scratch_bytes: u64,
    /// Accounted bytes written to bounded temporary artifacts/sort runs.
    pub(super) spill_bytes: u64,
    pub(super) prepare_millis: u64,
    /// Peak bytes of simultaneously resident (activated, unexhausted) Nodes
    /// SST input bodies during a BasePrefix merge; 0 for DeltaRun. This is the
    /// bound that keeps a full-corpus rebuild from co-residing the corpus.
    pub(super) peak_resident_input_bytes: u64,
}

/// An immutable full base already uploaded by the prepare phase, or an
/// authoritative empty-corpus proof with no payload object.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedSearchCompaction {
    pub(super) selection: SearchCompactionSelection,
    pub(super) output: Option<PreparedSearchPayload>,
    /// Required iff `output` is absent. It binds the complete winner scan that
    /// proved the indexed corpus empty.
    pub(super) empty_proof_digest: Option<u64>,
    pub(super) metrics: SearchCompactionBuildMetrics,
}

impl PreparedSearchCompaction {
    pub(super) fn output_descriptor(&self) -> Option<&SstDescriptor> {
        self.output.as_ref().map(|output| &output.descriptor)
    }
}

fn kind_sort_key(kind: SearchLsmKind) -> u8 {
    match kind {
        SearchLsmKind::Vector => 0,
        SearchLsmKind::Text => 1,
    }
}

fn compressed_event_union(
    ranges: impl IntoIterator<Item = SearchEventRange>,
) -> Result<Vec<SearchEventRange>> {
    let mut ranges = ranges.into_iter().collect::<Vec<_>>();
    if ranges.is_empty() || ranges.iter().any(|range| !range.is_valid()) {
        return Err(Error::invariant(
            "Search-LSM compaction prefix has no valid event ownership",
        ));
    }
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut compressed: Vec<SearchEventRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = compressed.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        compressed.push(range);
    }
    Ok(compressed)
}

fn apply_signed_stat(current: u64, value: SearchStatValue) -> Result<u64> {
    match value {
        SearchStatValue::Absolute(value) => Ok(value),
        SearchStatValue::Delta(value) if value >= 0 => current
            .checked_add(value as u64)
            .ok_or_else(|| Error::invariant("Search-LSM live statistics overflow")),
        SearchStatValue::Delta(value) => current
            .checked_sub(value.unsigned_abs())
            .ok_or_else(|| Error::invariant("Search-LSM live statistics become negative")),
    }
}

fn current_live_count(state: &SearchLsmState) -> Result<u64> {
    let mut current = 0u64;
    for segment in &state.segments {
        let value = match segment.stats {
            SearchSegmentStats::Vector { live_count } => live_count,
            SearchSegmentStats::Text { doc_count, .. } => doc_count,
            SearchSegmentStats::Unknown => {
                return Err(Error::invariant(
                    "active Search-LSM segment has unknown statistics",
                ));
            }
        };
        current = apply_signed_stat(current, value)?;
    }
    Ok(current)
}

fn base_rebuild_needed(
    state: &SearchLsmState,
    descriptors: &[SstDescriptor],
    policy: SearchCompactionPolicy,
) -> Result<bool> {
    // A shadow has only the authenticated version/suppress table. A DeltaRun
    // cannot reconstruct its missing postings, so repair must scan the
    // authoritative Nodes snapshot and publish a complete base.
    if state
        .segments
        .iter()
        .any(|segment| segment.payload == SearchSegmentPayload::ShadowOnly)
    {
        return Ok(true);
    }
    let has_delta = state
        .segments
        .iter()
        .any(|segment| segment.role == SearchSegmentRole::Delta);
    let has_consolidatable_state = has_delta || !state.proven_empty_event_ranges.is_empty();
    // An operational force is one-shot debt repayment, not a perpetual
    // rewrite request for an already complete singleton base.
    if policy.force_base && has_consolidatable_state {
        return Ok(true);
    }
    if !has_delta {
        return Ok(false);
    }
    // A large healthy base must not retrigger its own rewrite forever. The byte
    // threshold measures accumulated delta debt only.
    let delta_bytes = state
        .segments
        .iter()
        .zip(descriptors)
        .filter(|(segment, _)| segment.role == SearchSegmentRole::Delta)
        .try_fold(0u64, |total, (_, descriptor)| {
            total
                .checked_add(descriptor.size_bytes)
                .ok_or_else(|| Error::invariant("Search-LSM delta byte count overflows"))
        })?;
    if delta_bytes >= policy.base_compact_bytes {
        return Ok(true);
    }
    let delta_mutations = state
        .segments
        .iter()
        .filter(|segment| segment.role == SearchSegmentRole::Delta)
        .try_fold(0u64, |total, segment| {
            total
                .checked_add(segment.mutation_count)
                .ok_or_else(|| Error::invariant("Search-LSM delta mutation count overflows"))
        })?;
    if delta_mutations == 0 {
        return Ok(false);
    }
    let live = current_live_count(state)?;
    Ok(u128::from(delta_mutations).saturating_mul(100)
        >= u128::from(live.max(1)).saturating_mul(u128::from(policy.base_stale_percent)))
}

/// Select routine bounded delta tails, or a rare authoritative full base when
/// its independent bytes/staleness/explicit trigger fires.
pub(super) fn select_search_compactions(
    manifest: &Manifest,
    policy: SearchCompactionPolicy,
) -> Result<Vec<SearchCompactionSelection>> {
    if policy.delta_trigger_segments < 2
        || policy.hard_segment_cap < 2
        || policy.delta_trigger_segments > policy.hard_segment_cap
        || policy.hard_segment_cap > MAX_ACTIVE_SEARCH_SEGMENTS
        || policy.base_compact_bytes == 0
        || policy.base_stale_percent == 0
        || policy.vector_base_delta_rows == 0
    {
        return Err(Error::precondition(
            "Search-LSM compaction policy has invalid segment bounds",
        ));
    }
    validate_search_lsm(manifest).map_err(|error| {
        Error::precondition(format!(
            "cannot plan physical Search-LSM compaction from invalid state: {error}"
        ))
    })?;

    let now = Utc::now();
    let nodes_bytes = manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .fold(0u64, |total, descriptor| {
            total.saturating_add(descriptor.size_bytes)
        });
    let mut selections = Vec::new();
    for state in manifest
        .search_lsm
        .iter()
        .filter(|state| building_activation_eligible(state))
    {
        if let Some(selection) = select_building_activation(manifest, state, policy)? {
            selections.push(selection);
        }
    }
    for state in manifest
        .search_lsm
        .iter()
        .filter(|state| state.status == SearchLsmStatus::Active && !state.segments.is_empty())
    {
        if state.next_event_seq == 0 {
            return Err(Error::invariant(
                "active Search-LSM with physical segments has a zero event frontier",
            ));
        }
        let mut descriptors = Vec::with_capacity(state.segments.len());
        let mut descriptor_ids = HashSet::with_capacity(state.segments.len());
        for segment in &state.segments {
            let matches = manifest
                .ssts
                .iter()
                .filter(|descriptor| descriptor.id == segment.sst_id)
                .collect::<Vec<_>>();
            let [descriptor] = matches.as_slice() else {
                return Err(Error::invariant(format!(
                    "Search-LSM segment {} is missing or ambiguous during compaction planning",
                    segment.sst_id
                )));
            };
            if descriptor.kind != state.kind.sst_kind()
                || descriptor.scope != state.index_name
                || !descriptor_ids.insert(descriptor.id)
            {
                return Err(Error::invariant(
                    "Search-LSM compaction selected an invalid physical descriptor set",
                ));
            }
            descriptors.push((*descriptor).clone());
        }
        let trailing_deltas = trailing_delta_len(state);
        let tiered_run = tiered_delta_run_len(state, &descriptors);
        let under_pressure = state.segments.len() >= pressure_segments(policy);
        // Size-tiered: merge the newest `delta_trigger_segments` deltas once
        // none of them is larger than the newer ones combined, so a merged
        // output waits for peers of its size instead of being rewritten by
        // every later run (which made each row's rewrites grow with the
        // number of flushes). Under pressure, merge them whatever their sizes.
        let mut delta_run_len = ((tiered_run >= policy.delta_trigger_segments)
            || (under_pressure && trailing_deltas >= policy.delta_trigger_segments))
            .then_some(policy.delta_trigger_segments);
        // At the ceiling, a routine run too short to get back below it used
        // to force a corpus-wide base rebuild. Merging the whole trailing
        // delta tail (bounded) frees the flush slot for the cost of the tail
        // alone; the base remains the route only when the tail cannot.
        if state.segments.len() >= policy.hard_segment_cap {
            let routine_output = delta_run_len
                .map(|len| state.segments.len() - len + 1)
                .unwrap_or(usize::MAX);
            let tail = trailing_deltas.min(MAX_PRESSURE_DELTA_RUN);
            if routine_output >= policy.hard_segment_cap
                && tail >= MIN_SEARCH_LSM_SEGMENTS
                && state.segments.len() - tail + 1 < policy.hard_segment_cap
            {
                delta_run_len = Some(tail);
            }
        }
        let delta_run_output_len = delta_run_len.and_then(|len| {
            state
                .segments
                .len()
                .checked_sub(len)
                .and_then(|retained| retained.checked_add(1))
        });
        // Lowering the configured cap below an existing generation, or
        // reaching cap with Base + too short a delta tail, must still have a
        // route that frees a foreground flush slot.
        let cap_requires_base = state.segments.len() >= policy.hard_segment_cap
            && delta_run_output_len
                .map(|len| len >= policy.hard_segment_cap)
                .unwrap_or(true);
        // A vector base folded from the generation's own segments reads only
        // them; otherwise the rebuild scans the whole Nodes corpus.
        let vector_rebuild_input_bytes = if state.kind == SearchLsmKind::Vector
            && vector_generation_filters(manifest, state)
                .is_some_and(|filters| vector_segments_mergeable(state, &filters))
        {
            descriptors.iter().fold(0u64, |total, descriptor| {
                total.saturating_add(descriptor.size_bytes)
            })
        } else {
            nodes_bytes
        };
        let rebuild_base = base_rebuild_needed(state, &descriptors, policy)?
            || vector_base_rebuild_due(
                state,
                &descriptors,
                policy,
                now,
                vector_rebuild_input_bytes,
            )?
            || cap_requires_base;
        let (mode, selected_segment_start) = if rebuild_base {
            (SearchCompactionMode::BasePrefix, 0)
        } else if let Some(run_len) = delta_run_len {
            let start = state
                .segments
                .len()
                .checked_sub(run_len)
                .ok_or_else(|| Error::invariant("Search-LSM delta suffix underflows"))?;
            if state.segments[start..].iter().any(|segment| {
                segment.role != SearchSegmentRole::Delta
                    || segment.payload != SearchSegmentPayload::Complete
            }) {
                return Err(Error::invariant(
                    "Search-LSM DeltaRun selection would include a base or shadow segment",
                ));
            }
            (SearchCompactionMode::DeltaRun, start)
        } else {
            continue;
        };
        let selected_descriptors = descriptors[selected_segment_start..].to_vec();
        let selected_event_ranges = match mode {
            SearchCompactionMode::BasePrefix => {
                let ranges = compressed_event_union(
                    state
                        .segments
                        .iter()
                        .flat_map(|segment| segment.event_ranges.iter().copied())
                        .chain(state.proven_empty_event_ranges.iter().copied()),
                )?;
                if ranges.as_slice() != [SearchEventRange::new(0, state.next_event_seq)] {
                    return Err(Error::invariant(
                        "full-base Search-LSM selection is not one exact contiguous prefix",
                    ));
                }
                ranges
            }
            SearchCompactionMode::DeltaRun => compressed_event_union(
                state.segments[selected_segment_start..]
                    .iter()
                    .flat_map(|segment| segment.event_ranges.iter().copied()),
            )?,
        };
        selections.push(SearchCompactionSelection {
            mode,
            captured_state: state.clone(),
            selected_segment_start,
            selected_descriptors,
            selected_event_ranges,
            captured_coverage_digest: super::input_coverage_digest(&state.coverage)?,
            hard_segment_cap: policy.hard_segment_cap,
        });
    }
    selections.sort_by(|left, right| {
        kind_sort_key(left.captured_state.kind)
            .cmp(&kind_sort_key(right.captured_state.kind))
            .then_with(|| {
                left.captured_state
                    .index_name
                    .cmp(&right.captured_state.index_name)
            })
    });
    Ok(selections)
}

/// The delta runs a search-delta pass may install while bases are built
/// elsewhere: routine runs that start past the protected prefix of their
/// index (shortened to start there when they do not), plus a run over the
/// deltas past that prefix of a generation whose base or activation is due.
/// A Building generation gains one delta per flush until its activation
/// installs; merging them meanwhile keeps it below the flush ceiling.
pub(super) fn delta_runs_past(
    manifest: &Manifest,
    selections: Vec<SearchCompactionSelection>,
    protected: &std::collections::HashMap<(SearchLsmKind, String), usize>,
    policy: SearchCompactionPolicy,
) -> Result<Vec<SearchCompactionSelection>> {
    let mut runs = Vec::new();
    for selection in selections {
        let state = &selection.captured_state;
        let floor = protected
            .get(&(state.kind, state.index_name.clone()))
            .copied();
        match selection.mode {
            SearchCompactionMode::DeltaRun => match floor {
                Some(floor) if selection.selected_segment_start < floor => {
                    if let Some(run) = delta_run_from(manifest, state, floor, policy)? {
                        runs.push(run);
                    }
                }
                _ => runs.push(selection),
            },
            SearchCompactionMode::BasePrefix => {
                let first_delta = state
                    .segments
                    .iter()
                    .position(|segment| segment.role == SearchSegmentRole::Delta)
                    .unwrap_or(state.segments.len());
                let start = floor.unwrap_or(0).max(first_delta);
                if state.segments.len().saturating_sub(start) >= policy.delta_trigger_segments {
                    if let Some(run) = delta_run_from(manifest, state, start, policy)? {
                        runs.push(run);
                    }
                }
            }
        }
    }
    Ok(runs)
}

/// One DeltaRun over `state`'s segments from `start` (the newest
/// [`MAX_PRESSURE_DELTA_RUN`] at most), if at least two complete deltas are
/// there.
fn delta_run_from(
    manifest: &Manifest,
    state: &SearchLsmState,
    start: usize,
    policy: SearchCompactionPolicy,
) -> Result<Option<SearchCompactionSelection>> {
    let start = start.max(state.segments.len().saturating_sub(MAX_PRESSURE_DELTA_RUN));
    let selected = state.segments.get(start..).unwrap_or_default();
    if selected.len() < MIN_SEARCH_LSM_SEGMENTS
        || selected.iter().any(|segment| {
            segment.role != SearchSegmentRole::Delta
                || segment.payload != SearchSegmentPayload::Complete
        })
    {
        return Ok(None);
    }
    let mut descriptors = Vec::with_capacity(selected.len());
    for segment in selected {
        let matches = manifest
            .ssts
            .iter()
            .filter(|descriptor| descriptor.id == segment.sst_id)
            .collect::<Vec<_>>();
        let [descriptor] = matches.as_slice() else {
            return Ok(None);
        };
        descriptors.push((*descriptor).clone());
    }
    Ok(Some(SearchCompactionSelection {
        mode: SearchCompactionMode::DeltaRun,
        captured_state: state.clone(),
        selected_segment_start: start,
        selected_descriptors: descriptors,
        selected_event_ranges: compressed_event_union(
            selected
                .iter()
                .flat_map(|segment| segment.event_ranges.iter().copied()),
        )?,
        captured_coverage_digest: super::input_coverage_digest(&state.coverage)?,
        hard_segment_cap: policy.hard_segment_cap,
    }))
}

/// A Building generation that a physical base build can activate: it has
/// seen at least one flush (events and a barrier).
pub(super) fn building_activation_eligible(state: &SearchLsmState) -> bool {
    state.status == SearchLsmStatus::Building
        && state.next_event_seq > 0
        && state.compat_barrier_sst_id.is_some()
}

/// Plan the activation of a Building generation: one authoritative base over
/// every visible Nodes SST that owns the generation's whole event prefix.
///
/// A generation starts Building when its index is created (or its catalog
/// changes) over existing nodes: its flush deltas describe only later
/// changes. The base scans the corpus once and, on install, replaces every
/// captured segment, covers the Nodes SSTs that predate the generation and
/// flips it Active. Like any BasePrefix it rebases over the flushes that land
/// while it builds, so sustained writes cannot outrun it.
fn select_building_activation(
    manifest: &Manifest,
    state: &SearchLsmState,
    policy: SearchCompactionPolicy,
) -> Result<Option<SearchCompactionSelection>> {
    let mut descriptors = Vec::with_capacity(state.segments.len());
    for segment in &state.segments {
        let matches = manifest
            .ssts
            .iter()
            .filter(|descriptor| descriptor.id == segment.sst_id)
            .collect::<Vec<_>>();
        let [descriptor] = matches.as_slice() else {
            return Ok(None);
        };
        if descriptor.kind != state.kind.sst_kind() || descriptor.scope != state.index_name {
            return Ok(None);
        }
        descriptors.push((*descriptor).clone());
    }
    let Ok(ranges) = compressed_event_union(
        state
            .segments
            .iter()
            .flat_map(|segment| segment.event_ranges.iter().copied())
            .chain(state.proven_empty_event_ranges.iter().copied()),
    ) else {
        return Ok(None);
    };
    if ranges.as_slice() != [SearchEventRange::new(0, state.next_event_seq)]
        || state
            .segments
            .iter()
            .any(|segment| segment.role != SearchSegmentRole::Delta)
    {
        tracing::warn!(
            index = %state.index_name,
            "Building Search-LSM generation does not own one contiguous event prefix; \
             leaving it to the logical rebuild"
        );
        return Ok(None);
    }
    if !manifest
        .ssts
        .iter()
        .any(|descriptor| descriptor.kind == SstKind::Nodes)
    {
        return Ok(None);
    }
    Ok(Some(SearchCompactionSelection {
        mode: SearchCompactionMode::BasePrefix,
        captured_state: state.clone(),
        selected_segment_start: 0,
        selected_descriptors: descriptors,
        selected_event_ranges: ranges,
        captured_coverage_digest: super::input_coverage_digest(&state.coverage)?,
        hard_segment_cap: policy.hard_segment_cap,
    }))
}

/// Complete an activation once the projected Nodes set is known: every Nodes
/// SST the generation does not cover predates it (flushes cover what they
/// write; a Nodes merge refuses to mix covered and uncovered inputs) and its
/// content is therefore in the activation base, which owns `0..frontier`.
/// Same-scope bodies that are neither segments nor the barrier (legacy bases
/// a logical rebuild left behind) are returned for removal.
pub(super) fn complete_building_activation(
    state: &mut SearchLsmState,
    frontier: u64,
    empty_proof_digest: Option<u64>,
    ssts: &[SstDescriptor],
) -> Result<Vec<Uuid>> {
    if state.status != SearchLsmStatus::Active || frontier == 0 {
        return Err(Error::invariant(
            "Search-LSM activation completion requires an activated generation",
        ));
    }
    let disposition = match empty_proof_digest {
        None => CoverageDisposition::Segment,
        Some(digest) if digest != 0 => CoverageDisposition::ProvenEmpty {
            classifier_version: EMPTY_BASE_CLASSIFIER_VERSION,
            before_after_digest: digest,
        },
        Some(_) => {
            return Err(Error::invariant(
                "empty Search-LSM activation lacks a non-zero winner proof",
            ))
        }
    };
    let visible = ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .map(|descriptor| descriptor.id)
        .collect::<HashSet<_>>();
    state
        .coverage
        .retain(|coverage| visible.contains(&coverage.node_sst_id));
    let covered = state
        .coverage
        .iter()
        .map(|coverage| coverage.node_sst_id)
        .collect::<HashSet<_>>();
    for descriptor in ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes && !covered.contains(&descriptor.id))
    {
        state.coverage.push(crate::search_lsm::SearchCoverage {
            node_sst_id: descriptor.id,
            node_sst_max_lsn: descriptor.max_lsn,
            event_ranges: vec![SearchEventRange::new(0, frontier)],
            disposition,
        });
    }
    let data = state
        .segments
        .iter()
        .map(|segment| segment.sst_id)
        .chain(state.compat_barrier_sst_id)
        .collect::<HashSet<_>>();
    Ok(ssts
        .iter()
        .filter(|descriptor| {
            descriptor.kind == state.kind.sst_kind()
                && descriptor.scope == state.index_name
                && !data.contains(&descriptor.id)
        })
        .map(|descriptor| descriptor.id)
        .collect())
}

pub(super) fn search_compaction_needed(
    manifest: &Manifest,
    policy: SearchCompactionPolicy,
) -> bool {
    match select_search_compactions(manifest, policy) {
        Ok(selections) => !selections.is_empty(),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "invalid Search-LSM state/policy during metadata scheduling; waking prepare"
            );
            true
        }
    }
}

fn current_state_for<'a>(
    manifest: &'a Manifest,
    selection: &SearchCompactionSelection,
) -> Result<&'a SearchLsmState> {
    let matching = manifest
        .search_lsm
        .iter()
        .filter(|state| {
            state.kind == selection.captured_state.kind
                && state.index_name == selection.captured_state.index_name
        })
        .collect::<Vec<_>>();
    match matching.as_slice() {
        [state] => Ok(*state),
        _ => Err(Error::precondition(format!(
            "abandoning physical Search-LSM compaction: generation '{}' disappeared or became \
             ambiguous",
            selection.captured_state.index_name
        ))),
    }
}

fn verify_selected_descriptors(
    manifest: &Manifest,
    prepared: &PreparedSearchCompaction,
) -> Result<()> {
    let selection = &prepared.selection;
    let selected_end = selection
        .selected_segment_start
        .checked_add(selection.selected_descriptors.len())
        .ok_or_else(|| Error::invariant("Search-LSM selected segment range overflows"))?;
    let activation = selection.captured_state.status == SearchLsmStatus::Building
        && selection.mode == SearchCompactionMode::BasePrefix;
    if (selection.selected_descriptors.is_empty() && !activation)
        || selected_end != selection.captured_state.segments.len()
        || selection
            .selected_descriptors
            .iter()
            .zip(&selection.captured_state.segments[selection.selected_segment_start..])
            .any(|(descriptor, segment)| descriptor.id != segment.sst_id)
    {
        return Err(Error::invariant(
            "prepared Search-LSM descriptor selection does not match its captured segments",
        ));
    }
    for selected in &selection.selected_descriptors {
        let matches = manifest
            .ssts
            .iter()
            .filter(|descriptor| descriptor.id == selected.id)
            .collect::<Vec<_>>();
        if matches.as_slice() != [selected] {
            return Err(Error::precondition(format!(
                "abandoning physical Search-LSM compaction: selected descriptor {} changed, \
                 disappeared, or became ambiguous",
                selected.id
            )));
        }
    }
    if let Some(output) = prepared.output_descriptor() {
        if manifest
            .ssts
            .iter()
            .any(|descriptor| descriptor.id == output.id)
        {
            return Err(Error::precondition(format!(
                "abandoning physical Search-LSM compaction: output UUID {} is already visible",
                output.id
            )));
        }
    }
    Ok(())
}

fn summed_delta_stats(selection: &SearchCompactionSelection) -> Result<SearchSegmentStats> {
    let selected = &selection.captured_state.segments[selection.selected_segment_start..];
    match selection.captured_state.kind {
        SearchLsmKind::Vector => {
            let mut total = 0i64;
            for segment in selected {
                let SearchSegmentStats::Vector {
                    live_count: SearchStatValue::Delta(value),
                } = segment.stats
                else {
                    return Err(Error::invariant(
                        "DeltaRun selected non-delta vector statistics",
                    ));
                };
                total = total
                    .checked_add(value)
                    .ok_or_else(|| Error::invariant("DeltaRun vector statistics overflow"))?;
            }
            Ok(SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(total),
            })
        }
        SearchLsmKind::Text => {
            let mut docs = 0i64;
            let mut total_len = 0i64;
            let mut violations = 0u64;
            for segment in selected {
                let SearchSegmentStats::Text {
                    doc_count: SearchStatValue::Delta(segment_docs),
                    total_len: SearchStatValue::Delta(segment_len),
                    term_df_violation_count,
                } = segment.stats
                else {
                    return Err(Error::invariant(
                        "DeltaRun selected non-delta text statistics",
                    ));
                };
                docs = docs
                    .checked_add(segment_docs)
                    .ok_or_else(|| Error::invariant("DeltaRun text doc statistics overflow"))?;
                total_len = total_len
                    .checked_add(segment_len)
                    .ok_or_else(|| Error::invariant("DeltaRun text length statistics overflow"))?;
                violations = violations
                    .checked_add(term_df_violation_count)
                    .ok_or_else(|| Error::invariant("DeltaRun text violations overflow"))?;
            }
            Ok(SearchSegmentStats::Text {
                doc_count: SearchStatValue::Delta(docs),
                total_len: SearchStatValue::Delta(total_len),
                term_df_violation_count: violations,
            })
        }
    }
}

fn delta_stats_are_zero(stats: SearchSegmentStats) -> bool {
    matches!(
        stats,
        SearchSegmentStats::Vector {
            live_count: SearchStatValue::Delta(0)
        } | SearchSegmentStats::Text {
            doc_count: SearchStatValue::Delta(0),
            total_len: SearchStatValue::Delta(0),
            term_df_violation_count: 0,
        }
    )
}

fn verify_output(prepared: &PreparedSearchCompaction) -> Result<()> {
    let state = &prepared.selection.captured_state;
    let Some(output) = &prepared.output else {
        if prepared.empty_proof_digest.is_none_or(|digest| digest == 0) {
            return Err(Error::invariant(
                "empty physical Search-LSM output lacks a non-zero winner proof",
            ));
        }
        if prepared.selection.mode == SearchCompactionMode::DeltaRun
            && !delta_stats_are_zero(summed_delta_stats(&prepared.selection)?)
        {
            return Err(Error::invariant(
                "empty DeltaRun output would discard non-zero authenticated statistics",
            ));
        }
        return Ok(());
    };
    if prepared.empty_proof_digest.is_some() {
        return Err(Error::invariant(
            "physical Search-LSM payload unexpectedly also carries an empty proof",
        ));
    }
    let segment = &output.segment;
    let descriptor = &output.descriptor;
    let expected_format = match (prepared.selection.mode, state.kind) {
        (SearchCompactionMode::BasePrefix, SearchLsmKind::Vector) => {
            SearchSegmentFormat::VectorV5Base
        }
        (SearchCompactionMode::BasePrefix, SearchLsmKind::Text) => SearchSegmentFormat::TextV4,
        (SearchCompactionMode::DeltaRun, SearchLsmKind::Vector) => SearchSegmentFormat::VectorV6,
        (SearchCompactionMode::DeltaRun, SearchLsmKind::Text) => SearchSegmentFormat::TextV4,
    };
    let expected_role = match prepared.selection.mode {
        SearchCompactionMode::BasePrefix => SearchSegmentRole::Base,
        SearchCompactionMode::DeltaRun => SearchSegmentRole::Delta,
    };
    if segment.sst_id != descriptor.id
        || descriptor.kind != state.kind.sst_kind()
        || descriptor.scope != state.index_name
        || segment.role != expected_role
        || segment.format != expected_format
        || segment.event_ranges != prepared.selection.selected_event_ranges
        || segment.min_lsn != descriptor.min_lsn
        || segment.max_lsn != descriptor.max_lsn
        || descriptor.row_count != segment.mutation_count
    {
        return Err(Error::invariant(
            "prepared physical Search-LSM output descriptor/segment binding is inconsistent",
        ));
    }
    match prepared.selection.mode {
        SearchCompactionMode::BasePrefix
            if segment.mutation_count != segment.live_payload_count
                || segment.suppress_count != 0 =>
        {
            return Err(Error::invariant(
                "prepared full base contains non-live operations",
            ));
        }
        SearchCompactionMode::DeltaRun
            if segment.stats != summed_delta_stats(&prepared.selection)? =>
        {
            return Err(Error::invariant(
                "prepared DeltaRun statistics do not equal the authenticated source sum",
            ));
        }
        _ => {}
    }
    Ok(())
}

/// Rebase one uploaded full base over append-only foreground flushes.
///
/// `working_state` normally starts as `current_state`, but may already contain
/// the logical coverage rewrite from a simultaneous Nodes compaction. Its
/// segment/proven-empty prefixes must still be byte-identical to current.
pub(super) fn rebase_prepared_search_compaction(
    manifest: &Manifest,
    prepared: &PreparedSearchCompaction,
    working_state: &SearchLsmState,
) -> Result<SearchLsmState> {
    let selection = &prepared.selection;
    let captured = &selection.captured_state;
    let current = current_state_for(manifest, selection)?;
    // Coverage may have been rewritten by a Nodes merge meanwhile: events
    // keep their identity across such rewrites, so only the segment and
    // proven-empty prefixes this compaction replaces must be unchanged.
    let delta_at = match selection.mode {
        SearchCompactionMode::BasePrefix => {
            super::verify_search_segments_append_only(captured, current)?;
            selection.selected_segment_start
        }
        SearchCompactionMode::DeltaRun => delta_run_position(captured, current, selection)?,
    };
    if super::input_coverage_digest(&captured.coverage)? != selection.captured_coverage_digest {
        return Err(Error::invariant(
            "prepared Search-LSM coverage digest disagrees with its captured state",
        ));
    }
    verify_selected_descriptors(manifest, prepared)?;
    verify_output(prepared)?;
    if working_state.kind != current.kind
        || working_state.index_name != current.index_name
        || working_state.catalog_signature != current.catalog_signature
        || working_state.generation_id != current.generation_id
        || working_state.status != current.status
        || working_state.next_event_seq != current.next_event_seq
        || working_state.base_frontier != current.base_frontier
        || working_state.segments != current.segments
        || working_state.proven_empty_event_ranges != current.proven_empty_event_ranges
        || working_state.compat_barrier_sst_id != current.compat_barrier_sst_id
        || working_state.equal_lsn_conflict_count != current.equal_lsn_conflict_count
    {
        return Err(Error::precondition(
            "abandoning physical Search-LSM compaction: working state drifted outside a logical \
             Nodes coverage rewrite",
        ));
    }

    let mut rebased = working_state.clone();
    if captured.status == SearchLsmStatus::Building
        && selection.mode == SearchCompactionMode::BasePrefix
    {
        rebased.status = SearchLsmStatus::Active;
    }
    match selection.mode {
        SearchCompactionMode::BasePrefix => {
            if selection.selected_segment_start != 0 {
                return Err(Error::invariant(
                    "BasePrefix Search-LSM compaction does not begin at segment zero",
                ));
            }
            let selected_len = captured.segments.len();
            let empty_prefix_len = captured.proven_empty_event_ranges.len();
            let mut segments = Vec::with_capacity(
                usize::from(prepared.output.is_some())
                    .saturating_add(current.segments.len().saturating_sub(selected_len)),
            );
            if let Some(output) = &prepared.output {
                segments.push(output.segment.clone());
            }
            segments.extend_from_slice(&current.segments[selected_len..]);
            rebased.segments = segments;
            rebased.base_frontier = prepared.output.as_ref().map(|_| captured.next_event_seq);
            let mut proven_empty = Vec::new();
            if prepared.output.is_none() {
                proven_empty.extend(selection.selected_event_ranges.iter().copied());
            }
            proven_empty.extend_from_slice(&current.proven_empty_event_ranges[empty_prefix_len..]);
            rebased.proven_empty_event_ranges = proven_empty;

            // Events exactly empty when first classified are now covered by
            // the authoritative base. An empty corpus instead binds every
            // captured coverage entry to the full winner proof. Append-era
            // coverage ends after the captured frontier and stays untouched.
            for coverage in &mut rebased.coverage {
                if coverage
                    .event_ranges
                    .iter()
                    .all(|range| range.end <= captured.next_event_seq)
                {
                    if prepared.output.is_some() {
                        if matches!(
                            coverage.disposition,
                            CoverageDisposition::ProvenEmpty { .. }
                        ) {
                            coverage.disposition = CoverageDisposition::Segment;
                        }
                    } else {
                        coverage.disposition = CoverageDisposition::ProvenEmpty {
                            classifier_version: EMPTY_BASE_CLASSIFIER_VERSION,
                            before_after_digest: prepared
                                .empty_proof_digest
                                .expect("verified empty proof"),
                        };
                    }
                }
            }
        }
        SearchCompactionMode::DeltaRun => {
            // Where the selected run sits now: its captured position, or
            // after a base that replaced the prefix before it meanwhile.
            let selected_start = delta_at;
            let captured_end = delta_at
                + captured
                    .segments
                    .len()
                    .saturating_sub(selection.selected_segment_start);
            let mut segments = Vec::with_capacity(
                selected_start
                    .saturating_add(usize::from(prepared.output.is_some()))
                    .saturating_add(current.segments.len().saturating_sub(captured_end)),
            );
            segments.extend_from_slice(&current.segments[..selected_start]);
            if let Some(output) = &prepared.output {
                segments.push(output.segment.clone());
            }
            segments.extend_from_slice(&current.segments[captured_end..]);
            rebased.segments = segments;
            rebased.base_frontier = current.base_frontier;

            if prepared.output.is_none() {
                rebased.proven_empty_event_ranges = compressed_event_union(
                    current
                        .proven_empty_event_ranges
                        .iter()
                        .copied()
                        .chain(selection.selected_event_ranges.iter().copied()),
                )?;
                let proof = prepared.empty_proof_digest.expect("verified empty proof");
                for coverage in &mut rebased.coverage {
                    let selected_count = coverage
                        .event_ranges
                        .iter()
                        .filter(|range| {
                            selection
                                .selected_event_ranges
                                .iter()
                                .any(|owner| owner.start <= range.start && range.end <= owner.end)
                        })
                        .count();
                    let overlaps_selection = coverage.event_ranges.iter().any(|range| {
                        selection
                            .selected_event_ranges
                            .iter()
                            .any(|owner| range.start < owner.end && owner.start < range.end)
                    });
                    if selected_count == coverage.event_ranges.len() && selected_count != 0 {
                        coverage.disposition = CoverageDisposition::ProvenEmpty {
                            classifier_version: EMPTY_DELTA_CLASSIFIER_VERSION,
                            before_after_digest: proof,
                        };
                    } else if overlaps_selection
                        && matches!(coverage.disposition, CoverageDisposition::Segment)
                    {
                        return Err(Error::precondition(
                            "empty DeltaRun cannot split one Segment coverage entry across \
                             retained and proven-empty events",
                        ));
                    }
                }
            }
        }
    }
    // Off-lock flushes may take a generation past its configured ceiling (up
    // to the protocol one) while this compaction ran; it still lowers the
    // count, so only the protocol ceiling refuses it.
    if rebased.segments.len() > MAX_ACTIVE_SEARCH_SEGMENTS {
        return Err(Error::precondition(format!(
            "abandoning physical Search-LSM compaction: rebased generation '{}' would retain {} \
             segments above hard cap {}",
            rebased.index_name,
            rebased.segments.len(),
            prepared.selection.hard_segment_cap
        )));
    }
    Ok(rebased)
}

/// Position of a DeltaRun's selected segments in the current generation.
///
/// Normally the generation only advanced append-only from the captured state
/// and the run sits where it was captured. A base build (or the activation of
/// a Building generation) may instead have installed meanwhile over the
/// segments before the run: the delta loop never selects what a base in
/// flight replaces, so the run is still present as one contiguous slice, now
/// after the new base. The run still merges into the same place, so it
/// rebases there instead of abandoning work that is still valid.
fn delta_run_position(
    captured: &SearchLsmState,
    current: &SearchLsmState,
    selection: &SearchCompactionSelection,
) -> Result<usize> {
    let append_only = super::verify_search_segments_append_only(captured, current);
    if append_only.is_ok() {
        return Ok(selection.selected_segment_start);
    }
    let selected = captured
        .segments
        .get(selection.selected_segment_start..)
        .unwrap_or_default();
    let same_generation = current.kind == captured.kind
        && current.index_name == captured.index_name
        && current.catalog_signature == captured.catalog_signature
        && current.generation_id == captured.generation_id
        && current.next_event_seq >= captured.next_event_seq;
    if !same_generation || selected.is_empty() {
        return append_only.map(|()| selection.selected_segment_start);
    }
    let mut positions = (0..=current.segments.len().saturating_sub(selected.len()))
        .filter(|&at| current.segments[at..].starts_with(selected));
    let (Some(at), None) = (positions.next(), positions.next()) else {
        return append_only.map(|()| selection.selected_segment_start);
    };
    let events = |segments: &[crate::search_lsm::SearchSegmentRef]| {
        segments
            .iter()
            .flat_map(|segment| segment.event_ranges.iter().copied())
            .collect::<Vec<_>>()
    };
    let run = events(selected);
    let run_start = run.iter().map(|range| range.start).min();
    let run_end = run.iter().map(|range| range.end).max();
    let (Some(run_start), Some(run_end)) = (run_start, run_end) else {
        return append_only.map(|()| selection.selected_segment_start);
    };
    // The segments before the run must hold only older events and the ones
    // after it only newer events, exactly as when the run was captured.
    let ordered = events(&current.segments[..at])
        .iter()
        .all(|range| range.end <= run_start)
        && events(&current.segments[at + selected.len()..])
            .iter()
            .all(|range| range.start >= run_end);
    if !ordered {
        return append_only.map(|()| selection.selected_segment_start);
    }
    Ok(at)
}

/// Metadata for one planned Nodes SST input. Bodies do not travel here: the
/// producer fetches them in activation order and streams each over a bounded
/// channel, so at most one unactivated body is ever in flight per lookahead
/// slot instead of the corpus co-residing.
struct SearchNodeSourcePlan {
    min_key: [u8; 16],
    size_bytes: u64,
    row_count: u64,
    label_def: LabelDef,
    scope: String,
}

struct SearchNodeCursor {
    cursor: super::NodeSourceCursor,
    label_def: LabelDef,
    scope: String,
}

struct BuiltSearchCompaction {
    prepared: PreparedSearchCompaction,
    file: Option<std::fs::File>,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
#[derive(Debug)]
struct CompactionSearchRangeSource {
    pinned: crate::range_cache::PinnedObjectRangeSource,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
#[async_trait::async_trait]
impl SearchVersionRangeSource for CompactionSearchRangeSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.pinned.read_range(range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.pinned.read_ranges(ranges).await
    }
}

/// A whole segment body fetched to the local spool (file-backed), read by
/// range without further object-store requests.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
#[derive(Debug)]
struct LocalSegmentSource {
    body: Bytes,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl LocalSegmentSource {
    fn slice(&self, range: Range<u64>) -> Result<Bytes> {
        let start = usize::try_from(range.start)
            .map_err(|_| Error::invariant("local segment range start exceeds usize"))?;
        let end = usize::try_from(range.end)
            .map_err(|_| Error::invariant("local segment range end exceeds usize"))?;
        if start > end || end > self.body.len() {
            return Err(Error::invariant(format!(
                "local segment range {start}..{end} leaves its {}-byte body",
                self.body.len()
            )));
        }
        Ok(self.body.slice(start..end))
    }
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
#[async_trait::async_trait]
impl SearchVersionRangeSource for LocalSegmentSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.slice(range)
    }
}

#[cfg(feature = "vector-index")]
#[async_trait::async_trait]
impl crate::sst::vector::v5::VectorV5RangeSource for LocalSegmentSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.slice(range)
    }
}

/// An Active vector generation whose segments alone hold its whole indexed
/// corpus at full precision: an optional complete clustered base followed by
/// complete VG6 deltas, each advertising every native filter of the
/// generation. A generation is Active only once its segments (plus proven-empty
/// events) describe every node — it starts Active only over an empty
/// namespace, and otherwise is activated by a base over the node corpus — so
/// such a generation needs no node scan to rebuild its base, with or without
/// an existing base.
fn vector_segments_mergeable(state: &SearchLsmState, filters: &[String]) -> bool {
    state.kind == SearchLsmKind::Vector
        && state.status == SearchLsmStatus::Active
        && !state.segments.is_empty()
        && state
            .segments
            .iter()
            .enumerate()
            .all(|(position, segment)| {
                segment.payload == SearchSegmentPayload::Complete
                    && match (segment.role, segment.format) {
                        (SearchSegmentRole::Base, SearchSegmentFormat::VectorV5Base) => {
                            position == 0
                        }
                        (SearchSegmentRole::Delta, SearchSegmentFormat::VectorV6) => true,
                        _ => false,
                    }
                    && filters
                        .iter()
                        .all(|property| segment.complete_filter_properties.contains(property))
            })
}

/// The generation's native-filter properties, when its catalog entry exists.
fn vector_generation_filters(manifest: &Manifest, state: &SearchLsmState) -> Option<Vec<String>> {
    let matches = manifest
        .vector_indexes
        .iter()
        .filter(|descriptor| descriptor.name == state.index_name)
        .collect::<Vec<_>>();
    let [descriptor] = matches.as_slice() else {
        return None;
    };
    Some(crate::search_lsm::vector_native_filter_properties(
        manifest, descriptor,
    ))
}

/// A vector BasePrefix that [`build_vector_segment_merge`] can fold from the
/// generation's segments instead of scanning the node corpus.
fn vector_segment_merge_supported(
    manifest: &Manifest,
    selection: &SearchCompactionSelection,
) -> bool {
    selection.mode == SearchCompactionMode::BasePrefix
        && selection.selected_segment_start == 0
        && selection.selected_descriptors.len() == selection.captured_state.segments.len()
        && vector_generation_filters(manifest, &selection.captured_state)
            .is_some_and(|filters| vector_segments_mergeable(&selection.captured_state, &filters))
}

#[cfg(feature = "text-index")]
fn segment_merge_supported(selection: &SearchCompactionSelection) -> bool {
    let selected = &selection.captured_state.segments[selection.selected_segment_start..];
    selection.captured_state.kind == SearchLsmKind::Text
        && (selection.captured_state.status == SearchLsmStatus::Active
            || selection.mode == SearchCompactionMode::DeltaRun)
        && !selected.is_empty()
        && selected.iter().all(|segment| {
            segment.format == SearchSegmentFormat::TextV4
                && segment.payload == SearchSegmentPayload::Complete
        })
        && match selection.mode {
            SearchCompactionMode::DeltaRun => true,
            // A base fold must cover the generation's whole event prefix; a
            // generation whose events do not yet account for every Nodes SST
            // (no base) needs the corpus scan.
            SearchCompactionMode::BasePrefix => {
                selection.selected_segment_start == 0 && selected[0].role == SearchSegmentRole::Base
            }
        }
}

/// Fold the selected FT4 segments into one by merging their postings (see
/// `sst/text/v4_merge.rs`): no node reads and no re-tokenization, so the work
/// is linear in the selected segments' bytes. A DeltaRun yields a signed
/// delta; a BasePrefix over a FT4 base yields the new absolute base.
#[cfg(feature = "text-index")]
async fn build_text_segment_merge(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    selection: SearchCompactionSelection,
) -> Result<BuiltSearchCompaction> {
    let started = Instant::now();
    let manifest = &base.manifest;
    let matches = manifest
        .text_indexes
        .iter()
        .filter(|descriptor| descriptor.name == selection.captured_state.index_name)
        .collect::<Vec<_>>();
    let [descriptor] = matches.as_slice() else {
        return Err(Error::precondition(
            "text segment merge catalog entry is missing or ambiguous",
        ));
    };
    let descriptor = (*descriptor).clone();
    let selected_segments =
        selection.captured_state.segments[selection.selected_segment_start..].to_vec();
    if selected_segments.len() != selection.selected_descriptors.len() {
        return Err(Error::invariant(
            "text segment merge segment/descriptor cardinality differs",
        ));
    }
    // Every merged source must carry a filter for it to stay complete.
    let catalog_filters = crate::search_lsm::text_native_filter_properties(manifest, &descriptor);
    let filters = catalog_filters
        .into_iter()
        .filter(|property| {
            selected_segments.iter().all(|segment| {
                segment
                    .complete_filter_properties
                    .binary_search(property)
                    .is_ok()
            })
        })
        .collect::<Vec<_>>();

    // One sequential GET per source to the file-backed spool, fetched
    // concurrently: the merge then reads ranges locally.
    let fetches = selection
        .selected_descriptors
        .iter()
        .map(|source| {
            let store = store.clone();
            async move { super::get_sst_body(store.as_ref(), paths, source).await }
        })
        .collect::<Vec<_>>();
    let bodies = futures::future::try_join_all(fetches).await?;
    let mut readers = Vec::with_capacity(bodies.len());
    let mut input_bytes = 0u64;
    for ((body, segment), source_descriptor) in bodies
        .into_iter()
        .zip(&selected_segments)
        .zip(&selection.selected_descriptors)
    {
        input_bytes = input_bytes.saturating_add(source_descriptor.size_bytes);
        let len = body.len() as u64;
        let source: Arc<dyn SearchVersionRangeSource> = Arc::new(LocalSegmentSource { body });
        let reader = crate::sst::text::v4::TextV4Reader::open(
            source,
            len,
            &selection.captured_state,
            segment,
        )
        .await
        .map_err(|error| Error::Corrupted {
            path: source_descriptor.path.clone(),
            detail: error.to_string(),
        })?;
        readers.push(reader);
    }

    let stats = match selection.mode {
        SearchCompactionMode::DeltaRun => {
            let SearchSegmentStats::Text {
                doc_count: SearchStatValue::Delta(doc_count_delta),
                total_len: SearchStatValue::Delta(total_len_delta),
                term_df_violation_count: 0,
            } = summed_delta_stats(&selection)?
            else {
                return Err(Error::invariant(
                    "text DeltaRun selected invalid signed statistics",
                ));
            };
            Some(crate::sst::text::v4::ReconciledTextDeltaStats {
                doc_count_delta,
                total_len_delta,
            })
        }
        SearchCompactionMode::BasePrefix => None,
    };
    let mut config = crate::sst::text::v4::TextV4ExternalBuildConfig::from_env(
        crate::sst::text::v4::TextV4BuildOptions::default(),
    )?;
    config.memory_budget_bytes = super::per_collector_index_build_memory_bytes(1)?
        .ok_or_else(|| Error::invariant("text segment merge has no build memory share"))?;
    let context = crate::sst::text::v4::TextV4BuildContext {
        sst_id: Uuid::now_v7(),
        event_ranges: selection.selected_event_ranges.clone(),
        complete_filter_properties: filters,
    };
    let state = selection.captured_state.clone();
    let merged = tokio::task::spawn_blocking(move || {
        crate::sst::text::v4::merge_text_v4_segments(&state, context, config, stats, &readers)
    })
    .await
    .map_err(|error| Error::invariant(format!("text segment merge panicked: {error}")))??;

    let Some(artifact) = merged.artifact else {
        if selection.mode == SearchCompactionMode::DeltaRun {
            return Err(Error::invariant(
                "non-empty text DeltaRun segment merge produced no output",
            ));
        }
        // Every winner is a suppression: the generation is empty.
        return Ok(BuiltSearchCompaction {
            prepared: PreparedSearchCompaction {
                selection,
                output: None,
                empty_proof_digest: Some(merged.winner_digest),
                metrics: SearchCompactionBuildMetrics {
                    input_bytes,
                    input_rows: merged.input_rows,
                    touched_rows: merged.touched_rows,
                    prepare_millis: elapsed_millis(started),
                    ..Default::default()
                },
            },
            file: None,
        });
    };
    let segment = artifact.output.segment.clone();
    if segment.mutation_count != merged.written_rows
        || artifact.output.version_table.record_count != segment.mutation_count
        || artifact.output.object_len != artifact.len
        || (selection.mode == SearchCompactionMode::DeltaRun
            && merged.touched_rows != segment.mutation_count)
    {
        return Err(Error::invariant(
            "text segment merge output counts disagree with its winner stream",
        ));
    }
    let file_name = format!(
        "{}-{}-{}.ft4",
        segment.sst_id.simple(),
        SstKind::TextIndex.path_tag(),
        descriptor.name
    );
    let output_descriptor = SstDescriptor {
        id: segment.sst_id,
        kind: SstKind::TextIndex,
        scope: descriptor.name,
        level: SstLevel(1),
        path: super::relative_sst_path(1, &file_name),
        size_bytes: artifact.len,
        row_count: segment.mutation_count,
        created_at: Utc::now(),
        min_key: artifact.output.version_table.min_node_id,
        max_key: artifact.output.version_table.max_node_id,
        min_lsn: segment.min_lsn,
        max_lsn: segment.max_lsn,
        schema_version_min: 0,
        schema_version_max: 0,
        property_stats: Vec::new(),
        kind_specific: KindSpecificStats::TextIndex {
            doc_count: segment.live_payload_count,
            term_count: 0,
            total_len: merged.live_total_len,
        },
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        composite_equality_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    };
    let scratch_bytes = artifact
        .metrics
        .spool_bytes_written
        .saturating_add(artifact.metrics.filter_spool_bytes);
    Ok(BuiltSearchCompaction {
        prepared: PreparedSearchCompaction {
            selection,
            output: Some(PreparedSearchPayload {
                descriptor: output_descriptor,
                segment: segment.clone(),
            }),
            empty_proof_digest: None,
            metrics: SearchCompactionBuildMetrics {
                input_bytes,
                output_bytes: artifact.len,
                input_rows: merged.input_rows,
                touched_rows: merged.touched_rows,
                live_rows: segment.live_payload_count,
                peak_workspace_bytes: artifact.metrics.peak_logical_memory_bytes as u64,
                scratch_bytes,
                spill_bytes: scratch_bytes,
                prepare_millis: elapsed_millis(started),
                peak_resident_input_bytes: 0,
            },
        },
        file: Some(artifact.file),
    })
}

/// Rebuild a vector generation's clustered base from its own segments (see
/// `sst/vector/base_merge.rs`): the base rows and the deltas' winners, at full
/// precision, stream into the same V5 collector a node scan feeds, so the
/// result is that scan's base without reading the node corpus. The work is
/// linear in the selected segments' bytes. `Ok(None)` abandons the rebuild
/// exactly where a node scan would (a single live row).
#[cfg(feature = "vector-index")]
async fn build_vector_segment_merge(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    selection: SearchCompactionSelection,
) -> Result<Option<BuiltSearchCompaction>> {
    let started = Instant::now();
    let manifest = &base.manifest;
    if !vector_segment_merge_supported(manifest, &selection) {
        return Err(Error::precondition(
            "vector segment merge selected a generation its segments cannot rebuild",
        ));
    }
    let matches = manifest
        .vector_indexes
        .iter()
        .filter(|descriptor| descriptor.name == selection.captured_state.index_name)
        .collect::<Vec<_>>();
    let [descriptor] = matches.as_slice() else {
        return Err(Error::precondition(
            "vector segment merge catalog entry is missing or ambiguous",
        ));
    };
    let descriptor = (*descriptor).clone();
    let filters = crate::search_lsm::vector_native_filter_properties(manifest, &descriptor);
    let segments = selection.captured_state.segments.clone();

    // One sequential GET per source to the file-backed spool, fetched
    // concurrently: the merge then reads ranges locally.
    let fetches = selection
        .selected_descriptors
        .iter()
        .map(|source| {
            let store = store.clone();
            async move { super::get_sst_body(store.as_ref(), paths, source).await }
        })
        .collect::<Vec<_>>();
    let bodies = futures::future::try_join_all(fetches).await?;
    let mut base_reader = None;
    let mut deltas = Vec::with_capacity(bodies.len());
    let mut input_bytes = 0u64;
    for ((body, segment), source_descriptor) in bodies
        .into_iter()
        .zip(&segments)
        .zip(&selection.selected_descriptors)
    {
        if source_descriptor.id != segment.sst_id {
            return Err(Error::invariant(
                "vector segment merge descriptor/segment order differs",
            ));
        }
        input_bytes = input_bytes.saturating_add(source_descriptor.size_bytes);
        let len = body.len() as u64;
        let local = Arc::new(LocalSegmentSource { body });
        let corrupted = |error: Error| Error::Corrupted {
            path: source_descriptor.path.clone(),
            detail: error.to_string(),
        };
        match segment.role {
            SearchSegmentRole::Base => {
                let source: Arc<dyn crate::sst::vector::v5::VectorV5RangeSource> = local;
                let reader = crate::sst::vector::v5::VectorV5Reader::open(source, len)
                    .await
                    .map_err(corrupted)?;
                let (min_key, max_key) = reader.node_id_bounds();
                if reader.point_count() != segment.live_payload_count
                    || reader.point_count() != source_descriptor.row_count
                    || min_key != source_descriptor.min_key
                    || max_key != source_descriptor.max_key
                    || !matches!(
                        segment.stats,
                        SearchSegmentStats::Vector {
                            live_count: SearchStatValue::Absolute(count)
                        } if count == reader.point_count()
                    )
                {
                    return Err(corrupted(Error::invariant(
                        "vector base body disagrees with its descriptor and segment",
                    )));
                }
                base_reader = Some(reader);
            }
            SearchSegmentRole::Delta => {
                let source: Arc<dyn SearchVersionRangeSource> = local;
                let reader = crate::sst::vector::v6::VectorV6Reader::open(
                    source,
                    len,
                    &selection.captured_state,
                    segment,
                    &descriptor,
                )
                .await
                .map_err(corrupted)?;
                if reader.version_reader().reference().record_count != source_descriptor.row_count {
                    return Err(corrupted(Error::invariant(
                        "vector delta version count disagrees with its descriptor",
                    )));
                }
                deltas.push(reader);
            }
        }
    }

    let mut collector = crate::sst::vector::v5::external::VectorV5ExternalBuildConfig::from_env()?;
    collector.memory_budget_bytes = super::per_collector_index_build_memory_bytes(1)?
        .ok_or_else(|| Error::invariant("vector segment merge has no build memory share"))?;
    let options = super::vector_v5_build_options();
    let merge_descriptor = descriptor.clone();
    let merge_filters = filters.clone();
    let merged = tokio::task::spawn_blocking(move || {
        crate::sst::vector::base_merge::merge_vector_base(
            crate::sst::vector::base_merge::VectorBaseMergeInput {
                descriptor: &merge_descriptor,
                filter_properties: &merge_filters,
                base: base_reader.as_ref(),
                deltas: &deltas,
                collector,
                options,
            },
        )
    })
    .await
    .map_err(|error| Error::invariant(format!("vector segment merge panicked: {error}")))??;

    let mut digest = winner_hasher(&selection);
    digest.update(&merged.winner_digest.to_le_bytes());
    finish_vector_base(
        selection,
        descriptor,
        filters,
        merged.artifact,
        VectorBaseCounts {
            input_bytes,
            input_rows: merged.input_rows,
            touched_rows: merged.touched_rows,
            live_rows: merged.live_rows,
            empty_proof_digest: digest.digest().max(1),
            merge_workspace_bytes: merged.peak_merge_workspace_bytes as u64,
        },
        started,
    )
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
async fn open_compaction_search_source(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    descriptor: &SstDescriptor,
) -> Result<Arc<CompactionSearchRangeSource>> {
    let absolute = descriptor_path(paths, descriptor);
    let meta = store.head(&absolute).await?;
    if meta.location != absolute || meta.size != descriptor.size_bytes {
        return Err(Error::Corrupted {
            path: absolute.to_string(),
            detail: format!(
                "Search-LSM compaction source HEAD disagrees with descriptor (location={}, \
                 size={}, expected_size={})",
                meta.location, meta.size, descriptor.size_bytes
            ),
        });
    }
    let pinned =
        crate::range_cache::PinnedObjectRangeSource::from_create_only_meta(store, meta).await?;
    Ok(Arc::new(CompactionSearchRangeSource { pinned }))
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
#[derive(Debug, Default)]
struct VersionRunCursor {
    page: usize,
    offset: usize,
    records: Vec<SearchVersionRecord>,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl VersionRunCursor {
    fn current(&self) -> Option<SearchVersionRecord> {
        self.records.get(self.offset).copied()
    }
}

/// Bounded k-way merge over already-sorted NAMISV01 runs. It retains one
/// checksummed page (at most 512 records) per selected source.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
struct VersionRunMerge<'a> {
    readers: Vec<&'a SearchVersionTableReader>,
    cursors: Vec<VersionRunCursor>,
    heap: BinaryHeap<Reverse<([u8; 16], usize)>>,
    input_rows: u64,
    touched_rows: u64,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl<'a> VersionRunMerge<'a> {
    async fn open(readers: Vec<&'a SearchVersionTableReader>) -> Result<Self> {
        if readers.len() < 2 {
            return Err(Error::invariant(
                "DeltaRun physical merge requires at least two source runs",
            ));
        }
        let input_rows = readers.iter().try_fold(0u64, |total, reader| {
            total
                .checked_add(reader.reference().record_count)
                .ok_or_else(|| Error::invariant("DeltaRun source record count overflows"))
        })?;
        let mut cursors = Vec::with_capacity(readers.len());
        let mut heap = BinaryHeap::with_capacity(readers.len());
        for (source, reader) in readers.iter().enumerate() {
            if reader.page_count() == 0 {
                return Err(Error::Corrupted {
                    path: "<search-delta-version-table>".into(),
                    detail: "selected DeltaRun source has no version pages".into(),
                });
            }
            let records = reader.read_page(0).await?;
            let first = records.first().ok_or_else(|| Error::Corrupted {
                path: "<search-delta-version-table>".into(),
                detail: "selected DeltaRun source begins with an empty version page".into(),
            })?;
            heap.push(Reverse((first.node_id, source)));
            cursors.push(VersionRunCursor {
                page: 0,
                offset: 0,
                records,
            });
        }
        Ok(Self {
            readers,
            cursors,
            heap,
            input_rows,
            touched_rows: 0,
        })
    }

    fn input_rows(&self) -> u64 {
        self.input_rows
    }

    fn touched_rows(&self) -> u64 {
        self.touched_rows
    }

    async fn advance_source(&mut self, source: usize) -> Result<()> {
        let cursor = self
            .cursors
            .get_mut(source)
            .ok_or_else(|| Error::invariant("DeltaRun heap source is out of bounds"))?;
        cursor.offset = cursor
            .offset
            .checked_add(1)
            .ok_or_else(|| Error::invariant("DeltaRun page offset overflows"))?;
        if cursor.offset == cursor.records.len() {
            cursor.page = cursor
                .page
                .checked_add(1)
                .ok_or_else(|| Error::invariant("DeltaRun page ordinal overflows"))?;
            if cursor.page == self.readers[source].page_count() {
                cursor.records.clear();
                cursor.offset = 0;
                return Ok(());
            }
            cursor.records = self.readers[source].read_page(cursor.page).await?;
            cursor.offset = 0;
            if cursor.records.is_empty() {
                return Err(Error::Corrupted {
                    path: "<search-delta-version-table>".into(),
                    detail: "selected DeltaRun source contains an empty version page".into(),
                });
            }
        }
        let next = cursor
            .current()
            .ok_or_else(|| Error::invariant("DeltaRun cursor lost its current record"))?;
        self.heap.push(Reverse((next.node_id, source)));
        Ok(())
    }

    /// Return the next distinct-NodeId winner batch.
    async fn next_batch(&mut self, limit: usize) -> Result<Vec<SearchVersionRecord>> {
        if limit == 0 {
            return Err(Error::precondition(
                "DeltaRun merge batch limit must be positive",
            ));
        }
        let mut output = Vec::with_capacity(limit);
        while output.len() < limit {
            let Some(Reverse((node_id, _))) = self.heap.peek().copied() else {
                break;
            };
            let mut versions = Vec::new();
            while let Some(Reverse((candidate, source))) = self.heap.peek().copied() {
                if candidate != node_id {
                    break;
                }
                self.heap.pop();
                let record = self.cursors[source].current().ok_or_else(|| {
                    Error::invariant("DeltaRun heap points at an exhausted cursor")
                })?;
                if record.node_id != node_id {
                    return Err(Error::invariant(
                        "DeltaRun heap key disagrees with its version record",
                    ));
                }
                versions.push(record);
                self.advance_source(source).await?;
            }
            let winner = reconcile_node_versions(versions).map_err(|error| Error::Corrupted {
                path: "<search-delta-version-tables>".into(),
                detail: error.to_string(),
            })?;
            output.push(winner.ok_or_else(|| {
                Error::invariant("non-empty DeltaRun version group produced no winner")
            })?);
            self.touched_rows = self
                .touched_rows
                .checked_add(1)
                .ok_or_else(|| Error::invariant("DeltaRun touched row count overflows"))?;
        }
        Ok(output)
    }
}

#[cfg(feature = "text-index")]
#[derive(Debug, Default)]
struct TermDeltaRunCursor {
    block: usize,
    offset: usize,
    terms: Vec<crate::sst::text::v4::TextV4TermDelta>,
}

#[cfg(feature = "text-index")]
impl TermDeltaRunCursor {
    fn current(&self) -> Option<&crate::sst::text::v4::TextV4TermDelta> {
        self.terms.get(self.offset)
    }
}

/// Bounded lexical k-way merge of authenticated FT4 per-term signed
/// statistics. Like [`VersionRunMerge`], it retains one checksummed block per
/// source rather than a generation-wide vocabulary.
#[cfg(feature = "text-index")]
struct TermDeltaRunMerge<'a> {
    readers: Vec<&'a crate::sst::text::v4::TextV4Reader>,
    cursors: Vec<TermDeltaRunCursor>,
    heap: BinaryHeap<Reverse<(String, usize)>>,
}

#[cfg(feature = "text-index")]
impl<'a> TermDeltaRunMerge<'a> {
    async fn open(readers: Vec<&'a crate::sst::text::v4::TextV4Reader>) -> Result<Self> {
        let mut cursors = Vec::with_capacity(readers.len());
        let mut heap = BinaryHeap::with_capacity(readers.len());
        for (source, reader) in readers.iter().enumerate() {
            let terms = if reader.term_delta_block_count() == 0 {
                Vec::new()
            } else {
                reader.read_term_delta_block(0).await?
            };
            if reader.term_delta_block_count() != 0 && terms.is_empty() {
                return Err(Error::Corrupted {
                    path: "<text-delta-dictionary>".into(),
                    detail: "FT4 dictionary contains an empty term-delta block".into(),
                });
            }
            if let Some(first) = terms.first() {
                heap.push(Reverse((first.term.clone(), source)));
            }
            cursors.push(TermDeltaRunCursor {
                block: 0,
                offset: 0,
                terms,
            });
        }
        Ok(Self {
            readers,
            cursors,
            heap,
        })
    }

    async fn advance_source(&mut self, source: usize) -> Result<()> {
        let cursor = self
            .cursors
            .get_mut(source)
            .ok_or_else(|| Error::invariant("FT4 term heap source is out of bounds"))?;
        cursor.offset = cursor
            .offset
            .checked_add(1)
            .ok_or_else(|| Error::invariant("FT4 term block offset overflows"))?;
        if cursor.offset == cursor.terms.len() {
            cursor.block = cursor
                .block
                .checked_add(1)
                .ok_or_else(|| Error::invariant("FT4 term block ordinal overflows"))?;
            if cursor.block == self.readers[source].term_delta_block_count() {
                cursor.terms.clear();
                cursor.offset = 0;
                return Ok(());
            }
            cursor.terms = self.readers[source]
                .read_term_delta_block(cursor.block)
                .await?;
            cursor.offset = 0;
            if cursor.terms.is_empty() {
                return Err(Error::Corrupted {
                    path: "<text-delta-dictionary>".into(),
                    detail: "FT4 dictionary contains an empty term-delta block".into(),
                });
            }
        }
        let next = cursor
            .current()
            .ok_or_else(|| Error::invariant("FT4 term cursor lost its current record"))?;
        self.heap.push(Reverse((next.term.clone(), source)));
        Ok(())
    }

    async fn next_batch(
        &mut self,
        limit: usize,
    ) -> Result<Vec<crate::sst::text::v4::TextV4TermDelta>> {
        if limit == 0 {
            return Err(Error::precondition(
                "FT4 term merge batch limit must be positive",
            ));
        }
        let mut output = Vec::with_capacity(limit);
        while output.len() < limit {
            let Some(Reverse((term, _))) = self.heap.peek().cloned() else {
                break;
            };
            let mut delta_df = 0i64;
            while let Some(Reverse((candidate, source))) = self.heap.peek().cloned() {
                if candidate != term {
                    break;
                }
                self.heap.pop();
                let current = self.cursors[source].current().ok_or_else(|| {
                    Error::invariant("FT4 term heap points at an exhausted cursor")
                })?;
                if current.term != term {
                    return Err(Error::invariant(
                        "FT4 term heap key disagrees with its dictionary record",
                    ));
                }
                delta_df =
                    delta_df
                        .checked_add(current.delta_df)
                        .ok_or_else(|| Error::Corrupted {
                            path: "<text-delta-dictionaries>".into(),
                            detail: format!(
                                "signed document frequency for term '{term}' overflows"
                            ),
                        })?;
                self.advance_source(source).await?;
            }
            output.push(crate::sst::text::v4::TextV4TermDelta { term, delta_df });
        }
        Ok(output)
    }
}

/// Build and upload selected physical Search-LSM rewrites without holding the
/// writer lock.
///
/// Rare BasePrefix work shares one global Node winner scan. Routine DeltaRun
/// work instead scrubs and k-way merges only its selected version tables, then
/// point-reads the touched IDs from this exact [`LoadedManifest`]. Immutable
/// data objects are uploaded before the returned plan can reach manifest CAS.
pub(super) async fn prepare_search_compactions(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    selections: Vec<SearchCompactionSelection>,
) -> Result<Vec<PreparedSearchCompaction>> {
    prepare_search_compactions_with(store, paths, base, selections, true).await
}

/// Serializes Search-LSM base rebuilds across every namespace of a process.
static BASE_REBUILD_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// [`prepare_search_compactions`]; `vector_segment_merge: false` forces every
/// vector base through the node scan (tests compare the two builds).
async fn prepare_search_compactions_with(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    selections: Vec<SearchCompactionSelection>,
    vector_segment_merge: bool,
) -> Result<Vec<PreparedSearchCompaction>> {
    if selections.is_empty() {
        return Ok(Vec::new());
    }
    let manifest = &base.manifest;
    let (base_selections, delta_selections): (Vec<_>, Vec<_>) = selections
        .into_iter()
        .partition(|selection| selection.mode == SearchCompactionMode::BasePrefix);
    // Base rebuilds run one at a time per process: each is bounded by the
    // index-build memory and spool budgets, so several namespaces rebuilding
    // at once (every pre-3.5.1 vector base after an upgrade, say) would
    // multiply both. Delta runs stay concurrent.
    let base_gate = if base_selections.is_empty() {
        None
    } else {
        Some(BASE_REBUILD_GATE.lock().await)
    };
    // A full-text base over FT4 segments folds by merging postings and a
    // vector base over VG5/VG6 segments by merging their rows; only the rest
    // (legacy or shadow bases, Building generations) scan the node corpus.
    let mut built = Vec::with_capacity(base_selections.len() + delta_selections.len());
    // A merge that fails falls back to the node scan, so one unexpected
    // segment never holds back every base of the pass.
    #[cfg(feature = "vector-index")]
    let base_selections = if vector_segment_merge {
        let (merged, mut scanned): (Vec<_>, Vec<_>) = base_selections
            .into_iter()
            .partition(|selection| vector_segment_merge_supported(manifest, selection));
        for selection in merged {
            match build_vector_segment_merge(store.clone(), paths, base, selection.clone()).await {
                Ok(Some(merged)) => built.push(merged),
                // Abandoned like a node scan would be (a single live row).
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        index = %selection.captured_state.index_name,
                        %error,
                        "vector segment merge failed; rebuilding the base from node documents"
                    );
                    scanned.push(selection);
                }
            }
        }
        scanned
    } else {
        base_selections
    };
    #[cfg(not(feature = "vector-index"))]
    let _ = vector_segment_merge;
    #[cfg(feature = "text-index")]
    let base_selections = {
        let (merged, mut scanned): (Vec<_>, Vec<_>) = base_selections
            .into_iter()
            .partition(segment_merge_supported);
        for selection in merged {
            match build_text_segment_merge(store.clone(), paths, base, selection.clone()).await {
                Ok(merged) => built.push(merged),
                Err(error) => {
                    tracing::warn!(
                        index = %selection.captured_state.index_name,
                        %error,
                        "text segment merge failed; rebuilding the base from node documents"
                    );
                    scanned.push(selection);
                }
            }
        }
        scanned
    };

    if !base_selections.is_empty() {
        let mut plan = Vec::new();
        let mut descriptors = Vec::new();
        let mut input_bytes = 0u64;
        for descriptor in manifest
            .ssts
            .iter()
            .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        {
            input_bytes = input_bytes
                .checked_add(descriptor.size_bytes)
                .ok_or_else(|| Error::invariant("Search-LSM compaction input bytes overflow"))?;
            let label_def = if descriptor.scope.is_empty() {
                LabelDef {
                    name: String::new(),
                    properties: Vec::new(),
                }
            } else {
                manifest
                    .schema
                    .label(&descriptor.scope)
                    .cloned()
                    .unwrap_or_else(|| LabelDef {
                        name: descriptor.scope.clone(),
                        properties: Vec::new(),
                    })
            };
            plan.push(SearchNodeSourcePlan {
                min_key: descriptor.min_key,
                size_bytes: descriptor.size_bytes,
                row_count: descriptor.row_count,
                label_def,
                scope: descriptor.scope.clone(),
            });
            descriptors.push(descriptor.clone());
        }
        if plan.is_empty() {
            return Err(Error::invariant(
                "Search-LSM BasePrefix compaction has no Nodes SST inputs",
            ));
        }

        // Producer/consumer pipeline. The merge activates sources in
        // (min_key, ordinal) order, so fetching in that exact order lets the
        // builder receive each body precisely when its heap entry activates.
        // Residency is bounded by activated-unexhausted bodies plus the
        // channel's lookahead — never the corpus.
        let mut fetch_order: Vec<usize> = (0..plan.len()).collect();
        fetch_order.sort_by_key(|&ordinal| (plan[ordinal].min_key, ordinal));
        let (sender, receiver) =
            tokio::sync::mpsc::channel::<(usize, Bytes)>(SEARCH_BASE_FETCH_LOOKAHEAD);
        let build_manifest = manifest.clone();
        let started = Instant::now();
        tracing::info!(
            input_bytes,
            sources = plan.len(),
            "starting Search-LSM base build; spool scratch scales with input \
             plus the external builders' bounded working sets"
        );
        let builder_handle = tokio::task::spawn_blocking(move || {
            build_search_compactions_blocking(
                &build_manifest,
                base_selections,
                plan,
                receiver,
                input_bytes,
                started,
            )
        });
        let producer_result: Result<()> = async {
            for ordinal in fetch_order {
                let body =
                    super::get_sst_body(store.as_ref(), paths, &descriptors[ordinal]).await?;
                if sender.send((ordinal, body)).await.is_err() {
                    // The builder hung up early; its own error surfaces below.
                    return Ok(());
                }
            }
            Ok(())
        }
        .await;
        drop(sender);
        let build_result = builder_handle.await.map_err(|error| {
            Error::invariant(format!(
                "Search-LSM base compaction build panicked: {error}"
            ))
        })?;
        // A genuine fetch error outranks the builder's derived channel-closed
        // error; a builder error must not be masked by the send failure it
        // caused.
        producer_result?;
        built.extend(build_result?);
    }

    drop(base_gate);
    for selection in delta_selections {
        built.push(match selection.mode {
            SearchCompactionMode::DeltaRun => {
                build_delta_run_compaction(store.clone(), paths, base, selection).await?
            }
            SearchCompactionMode::BasePrefix => {
                return Err(Error::invariant("BasePrefix routed to the delta runs"))
            }
        });
    }

    let mut prepared = Vec::with_capacity(built.len());
    for built in built {
        validate_built_compaction(manifest, &built.prepared)?;
        if let Some(file) = built.file {
            let output = built.prepared.output.as_ref().ok_or_else(|| {
                Error::invariant("Search-LSM compaction file has no output descriptor")
            })?;
            let file_len = file.metadata()?.len();
            if file_len != output.descriptor.size_bytes {
                return Err(Error::invariant(
                    "Search-LSM compaction artifact length disagrees with its descriptor",
                ));
            }
            let absolute = descriptor_path(paths, &output.descriptor);
            crate::spooled_object::put_spooled_object(
                store.clone(),
                &absolute,
                crate::spooled_object::SpooledObject::from_file(file, file_len),
            )
            .await?;
        } else if built.prepared.output.is_some() {
            return Err(Error::invariant(
                "Search-LSM compaction output is missing its immutable artifact",
            ));
        }
        tracing::info!(
            index = %built.prepared.selection.captured_state.index_name,
            kind = ?built.prepared.selection.captured_state.kind,
            mode = ?built.prepared.selection.mode,
            input_bytes = built.prepared.metrics.input_bytes,
            output_bytes = built.prepared.metrics.output_bytes,
            input_rows = built.prepared.metrics.input_rows,
            touched_rows = built.prepared.metrics.touched_rows,
            live_rows = built.prepared.metrics.live_rows,
            peak_workspace_bytes = built.prepared.metrics.peak_workspace_bytes,
            peak_resident_input_bytes = built.prepared.metrics.peak_resident_input_bytes,
            scratch_bytes = built.prepared.metrics.scratch_bytes,
            spill_bytes = built.prepared.metrics.spill_bytes,
            prepare_millis = built.prepared.metrics.prepare_millis,
            empty = built.prepared.output.is_none(),
            "prepared and uploaded bounded Search-LSM compaction"
        );
        prepared.push(built.prepared);
    }
    Ok(prepared)
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
async fn build_delta_run_compaction(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    selection: SearchCompactionSelection,
) -> Result<BuiltSearchCompaction> {
    if selection.mode != SearchCompactionMode::DeltaRun {
        return Err(Error::invariant(
            "DeltaRun builder received a BasePrefix selection",
        ));
    }
    match selection.captured_state.kind {
        SearchLsmKind::Vector => {
            #[cfg(feature = "vector-index")]
            {
                build_vector_delta_run(store, paths, base, selection).await
            }
            #[cfg(not(feature = "vector-index"))]
            {
                let _ = (store, paths, base, selection);
                Err(Error::precondition(
                    "vector DeltaRun compaction requires the vector-index feature",
                ))
            }
        }
        SearchLsmKind::Text => {
            #[cfg(feature = "text-index")]
            {
                if segment_merge_supported(&selection) {
                    match build_text_segment_merge(store.clone(), paths, base, selection.clone())
                        .await
                    {
                        Ok(built) => return Ok(built),
                        Err(error) => tracing::warn!(
                            index = %selection.captured_state.index_name,
                            %error,
                            "text segment merge failed; rebuilding the run from node documents"
                        ),
                    }
                }
                build_text_delta_run(store, paths, base, selection).await
            }
            #[cfg(not(feature = "text-index"))]
            {
                let _ = (store, paths, base, selection);
                Err(Error::precondition(
                    "text DeltaRun compaction requires the text-index feature",
                ))
            }
        }
    }
}

#[cfg(not(any(feature = "vector-index", feature = "text-index")))]
async fn build_delta_run_compaction(
    _store: Arc<dyn ObjectStore>,
    _paths: &NamespacePaths,
    _base: &LoadedManifest,
    _selection: SearchCompactionSelection,
) -> Result<BuiltSearchCompaction> {
    Err(Error::precondition(
        "physical Search-LSM DeltaRun compaction requires a search-index build feature",
    ))
}

#[cfg(feature = "vector-index")]
fn vector_payload_from_node(
    descriptor: &crate::manifest::VectorIndexDescriptor,
    filters: &[String],
    node: Option<&crate::read::NodeView>,
) -> Option<crate::sst::vector::v6::VectorV6Payload> {
    let node = node?;
    if !node.labels.contains(&descriptor.label) {
        return None;
    }
    let vector = node
        .properties
        .get(&descriptor.property)?
        .embedding_f32(descriptor.dim as usize)?
        .into_owned();
    if vector.len() != descriptor.dim as usize
        || vector.iter().any(|component| !component.is_finite())
    {
        return None;
    }
    Some(crate::sst::vector::v6::VectorV6Payload {
        vector,
        filters: native_filter_payload(&node.properties, filters),
    })
}

#[cfg(feature = "vector-index")]
fn reconcile_vector_payload(
    descriptor: &crate::manifest::VectorIndexDescriptor,
    filters: &[String],
    record: SearchVersionRecord,
    node: Option<&crate::read::NodeView>,
) -> Result<Option<crate::sst::vector::v6::VectorV6Payload>> {
    let payload = vector_payload_from_node(descriptor, filters, node);
    match (record.operation, payload) {
        (SearchVersionOperation::Live { .. }, Some(payload)) => {
            let fingerprint = crate::sst::vector::v6::vector_v6_payload_fingerprint(&payload)?;
            if fingerprint != record.payload_fingerprint {
                return Err(Error::Corrupted {
                    path: "<captured-nodes-snapshot>".into(),
                    detail: format!(
                        "vector payload for NodeId {:02x?} fingerprints to {fingerprint:#x}, \
                         source winner requires {:#x}",
                        record.node_id, record.payload_fingerprint
                    ),
                });
            }
            Ok(Some(payload))
        }
        (SearchVersionOperation::Suppress, None)
            if record.payload_fingerprint
                == crate::sst::search_delta::search_suppress_fingerprint() =>
        {
            Ok(None)
        }
        // Up to 3.4.1 a flush recorded a bare numeric list as no embedding
        // (a suppression); it reads as one now. The delta keeps what it
        // recorded, and the next base build indexes the list.
        (SearchVersionOperation::Suppress, Some(_))
            if record.payload_fingerprint
                == crate::sst::search_delta::search_suppress_fingerprint()
                && node.is_some_and(|node| {
                    matches!(
                        node.properties.get(&descriptor.property),
                        Some(Value::List(_))
                    )
                }) =>
        {
            Ok(None)
        }
        (SearchVersionOperation::Suppress, None) => Err(Error::Corrupted {
            path: "<search-delta-version-tables>".into(),
            detail: format!(
                "suppression for NodeId {:02x?} has a non-canonical fingerprint",
                record.node_id
            ),
        }),
        (SearchVersionOperation::Live { .. }, None)
        | (SearchVersionOperation::Suppress, Some(_)) => Err(Error::Corrupted {
            path: "<captured-nodes-snapshot>".into(),
            detail: format!(
                "vector membership for NodeId {:02x?} disagrees with its source winner",
                record.node_id
            ),
        }),
    }
}

#[cfg(feature = "vector-index")]
/// Scrubs and merges only the selected delta objects. Its sequential work is
/// O(selected-delta bytes); payload hydration is a bounded set of Node point
/// reads and never scans the Nodes corpus.
async fn build_vector_delta_run(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    selection: SearchCompactionSelection,
) -> Result<BuiltSearchCompaction> {
    const MAX_RECONCILE_BATCH_ROWS: usize = 512;
    let started = Instant::now();

    let manifest = &base.manifest;
    let matches = manifest
        .vector_indexes
        .iter()
        .filter(|descriptor| descriptor.name == selection.captured_state.index_name)
        .collect::<Vec<_>>();
    let [descriptor] = matches.as_slice() else {
        return Err(Error::precondition(
            "vector DeltaRun catalog entry is missing or ambiguous",
        ));
    };
    let descriptor = (*descriptor).clone();
    let filters = crate::search_lsm::vector_native_filter_properties(manifest, &descriptor);
    let selected_segments = &selection.captured_state.segments[selection.selected_segment_start..];
    if selected_segments.len() != selection.selected_descriptors.len() {
        return Err(Error::invariant(
            "vector DeltaRun segment/descriptor cardinality differs",
        ));
    }

    let mut readers = Vec::with_capacity(selected_segments.len());
    let mut input_bytes = 0u64;
    for (segment, source_descriptor) in selected_segments
        .iter()
        .zip(&selection.selected_descriptors)
    {
        if segment.role != SearchSegmentRole::Delta
            || segment.format != SearchSegmentFormat::VectorV6
        {
            return Err(Error::invariant(
                "vector DeltaRun selected a non-VG6 delta source",
            ));
        }
        input_bytes = input_bytes
            .checked_add(source_descriptor.size_bytes)
            .ok_or_else(|| Error::invariant("vector DeltaRun input bytes overflow"))?;
        let source = open_compaction_search_source(store.clone(), paths, source_descriptor).await?;
        let source: Arc<dyn SearchVersionRangeSource> = source;
        let reader = crate::sst::vector::v6::VectorV6Reader::open(
            source,
            source_descriptor.size_bytes,
            &selection.captured_state,
            segment,
            &descriptor,
        )
        .await
        .map_err(|error| Error::Corrupted {
            path: source_descriptor.path.clone(),
            detail: error.to_string(),
        })?;
        reader
            .verify_all()
            .await
            .map_err(|error| Error::Corrupted {
                path: source_descriptor.path.clone(),
                detail: error.to_string(),
            })?;
        readers.push(reader);
    }
    let version_readers = readers
        .iter()
        .map(|reader| reader.version_reader())
        .collect::<Vec<_>>();
    let mut merge = VersionRunMerge::open(version_readers).await?;
    let input_rows = merge.input_rows();
    if input_rows
        != selection
            .selected_descriptors
            .iter()
            .try_fold(0u64, |total, descriptor| {
                total
                    .checked_add(descriptor.row_count)
                    .ok_or_else(|| Error::invariant("vector DeltaRun descriptor rows overflow"))
            })?
    {
        return Err(Error::Corrupted {
            path: "<search-delta-version-tables>".into(),
            detail: "vector source version counts disagree with descriptors".into(),
        });
    }

    let SearchSegmentStats::Vector {
        live_count: SearchStatValue::Delta(authenticated_live_count_delta),
    } = summed_delta_stats(&selection)?
    else {
        return Err(Error::invariant(
            "vector DeltaRun selected invalid signed statistics",
        ));
    };
    let mut config = crate::sst::vector::v6::VectorV6ExternalBuildConfig::from_env(
        crate::sst::vector::v6::VectorV6BuildOptions::default(),
    )?;
    config.memory_budget_bytes = super::per_collector_index_build_memory_bytes(1)?
        .ok_or_else(|| Error::invariant("vector DeltaRun has no build memory share"))?;
    let payload_row_bytes = (descriptor.dim as usize)
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_add(512)
        .max(1);
    let batch_rows =
        MAX_RECONCILE_BATCH_ROWS.min((config.memory_budget_bytes / 8 / payload_row_bytes).max(1));

    type VectorReconciledBatch = Vec<(
        SearchVersionRecord,
        Option<crate::sst::vector::v6::VectorV6Payload>,
    )>;
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<VectorReconciledBatch>(2);
    let build_state = selection.captured_state.clone();
    let build_descriptor = descriptor.clone();
    let build_context = crate::sst::vector::v6::VectorV6BuildContext {
        sst_id: Uuid::now_v7(),
        event_ranges: selection.selected_event_ranges.clone(),
        complete_filter_properties: filters.clone(),
    };
    let builder_handle = tokio::task::spawn_blocking(move || {
        let mut builder = crate::sst::vector::v6::VectorV6ExternalBuilder::with_config_reconciled(
            &build_state,
            &build_descriptor,
            build_context,
            authenticated_live_count_delta,
            config,
        )?;
        while let Some(batch) = receiver.blocking_recv() {
            for (record, payload) in batch {
                builder.push_reconciled(record, payload)?;
            }
        }
        builder.finish()
    });

    let empty_memtable = MemtableSnapshot::empty();
    let snapshot = crate::read::Snapshot::new(base.clone(), &empty_memtable, store, paths.clone());
    let producer_result: Result<()> = async {
        loop {
            let winners = merge.next_batch(batch_rows).await?;
            if winners.is_empty() {
                break;
            }
            let ids = winners
                .iter()
                .map(|record| NodeId::from_uuid(Uuid::from_bytes(record.node_id)))
                .collect::<Vec<_>>();
            let nodes = snapshot.batch_lookup_nodes("", &ids).await?;
            if nodes.len() != winners.len() {
                return Err(Error::invariant(
                    "vector DeltaRun point-read returned the wrong cardinality",
                ));
            }
            let batch = winners
                .into_iter()
                .zip(nodes.iter())
                .map(|(record, node)| {
                    reconcile_vector_payload(&descriptor, &filters, record, node.as_ref())
                        .map(|payload| (record, payload))
                })
                .collect::<Result<Vec<_>>>()?;
            sender.send(batch).await.map_err(|_| {
                Error::precondition("vector DeltaRun builder stopped before consuming all winners")
            })?;
        }
        Ok(())
    }
    .await;
    drop(sender);
    let artifact_result = builder_handle
        .await
        .map_err(|error| Error::invariant(format!("vector DeltaRun builder panicked: {error}")))?;
    producer_result?;
    let artifact = artifact_result?.ok_or_else(|| {
        Error::invariant("non-empty vector DeltaRun source merge produced no output artifact")
    })?;
    let touched_rows = merge.touched_rows();
    if touched_rows == 0 || touched_rows != artifact.output.segment.mutation_count {
        return Err(Error::invariant(
            "vector DeltaRun output count disagrees with reconciled winners",
        ));
    }

    let segment = artifact.output.segment.clone();
    let file_name = format!(
        "{}-{}-{}.vg6",
        segment.sst_id.simple(),
        SstKind::VectorGraph.path_tag(),
        descriptor.name
    );
    let output_descriptor = SstDescriptor {
        id: segment.sst_id,
        kind: SstKind::VectorGraph,
        scope: descriptor.name,
        level: SstLevel(1),
        path: super::relative_sst_path(1, &file_name),
        size_bytes: artifact.len,
        row_count: segment.mutation_count,
        created_at: Utc::now(),
        min_key: artifact.output.version_table.min_node_id,
        max_key: artifact.output.version_table.max_node_id,
        min_lsn: segment.min_lsn,
        max_lsn: segment.max_lsn,
        schema_version_min: 0,
        schema_version_max: 0,
        property_stats: Vec::new(),
        kind_specific: KindSpecificStats::VectorGraph {
            dim: descriptor.dim,
            metric: match descriptor.metric {
                crate::manifest::VectorMetric::Cosine => "cosine",
                crate::manifest::VectorMetric::Dot => "dot",
                crate::manifest::VectorMetric::Euclidean => "euclidean",
            }
            .into(),
            point_count: segment.live_payload_count,
            r: descriptor.r,
            l_build: descriptor.l_build,
            alpha: descriptor.alpha,
            entry_medoid: 0,
        },
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        composite_equality_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    };
    let elapsed = elapsed_millis(started);
    let output_bytes = artifact.len;
    let live_rows = segment.live_payload_count;
    let peak_workspace_bytes = artifact.metrics.peak_logical_memory_bytes as u64;
    Ok(BuiltSearchCompaction {
        prepared: PreparedSearchCompaction {
            selection,
            output: Some(PreparedSearchPayload {
                descriptor: output_descriptor,
                segment,
            }),
            empty_proof_digest: None,
            metrics: SearchCompactionBuildMetrics {
                input_bytes,
                output_bytes,
                input_rows,
                touched_rows,
                live_rows,
                peak_workspace_bytes,
                scratch_bytes: output_bytes,
                spill_bytes: output_bytes,
                prepare_millis: elapsed,
                peak_resident_input_bytes: 0,
            },
        },
        file: Some(artifact.file),
    })
}

#[cfg(feature = "text-index")]
fn text_payload_from_node(
    descriptor: &crate::manifest::TextIndexDescriptor,
    filters: &[String],
    node: Option<&crate::read::NodeView>,
) -> Option<crate::sst::text::v4::TextV4Payload> {
    let node = node?;
    if !node.labels.contains(&descriptor.label) {
        return None;
    }
    let parts = descriptor
        .properties
        .iter()
        .filter_map(|property| match node.properties.get(property) {
            Some(Value::Str(value)) => Some(value.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return None;
    }
    Some(crate::sst::text::v4::TextV4Payload {
        text: parts.join(" "),
        filters: native_filter_payload(&node.properties, filters),
    })
}

#[cfg(feature = "text-index")]
fn reconcile_text_payload(
    descriptor: &crate::manifest::TextIndexDescriptor,
    filters: &[String],
    record: SearchVersionRecord,
    node: Option<&crate::read::NodeView>,
) -> Result<Option<crate::sst::text::v4::TextV4Payload>> {
    let payload = text_payload_from_node(descriptor, filters, node);
    match (record.operation, payload) {
        (SearchVersionOperation::Live { .. }, Some(payload)) => {
            let fingerprint = crate::sst::text::v4::text_v4_payload_fingerprint(&payload)?;
            if fingerprint != record.payload_fingerprint {
                return Err(Error::Corrupted {
                    path: "<captured-nodes-snapshot>".into(),
                    detail: format!(
                        "text payload for NodeId {:02x?} fingerprints to {fingerprint:#x}, \
                         source winner requires {:#x}",
                        record.node_id, record.payload_fingerprint
                    ),
                });
            }
            Ok(Some(payload))
        }
        (SearchVersionOperation::Suppress, None)
            if record.payload_fingerprint
                == crate::sst::search_delta::search_suppress_fingerprint() =>
        {
            Ok(None)
        }
        (SearchVersionOperation::Suppress, None) => Err(Error::Corrupted {
            path: "<search-delta-version-tables>".into(),
            detail: format!(
                "text suppression for NodeId {:02x?} has a non-canonical fingerprint",
                record.node_id
            ),
        }),
        (SearchVersionOperation::Live { .. }, None)
        | (SearchVersionOperation::Suppress, Some(_)) => Err(Error::Corrupted {
            path: "<captured-nodes-snapshot>".into(),
            detail: format!(
                "text membership for NodeId {:02x?} disagrees with its source winner",
                record.node_id
            ),
        }),
    }
}

#[cfg(feature = "text-index")]
enum TextDeltaBuilderMessage {
    Winners(
        Vec<(
            SearchVersionRecord,
            Option<crate::sst::text::v4::TextV4Payload>,
        )>,
    ),
    Terms(Vec<crate::sst::text::v4::TextV4TermDelta>),
}

#[cfg(feature = "text-index")]
/// Scrubs and merges only selected FT4 delta bytes, point-reading just the
/// touched Nodes. Both version and vocabulary merges retain one block per run.
async fn build_text_delta_run(
    store: Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    base: &LoadedManifest,
    selection: SearchCompactionSelection,
) -> Result<BuiltSearchCompaction> {
    const MAX_RECONCILE_BATCH_ROWS: usize = 512;
    const MAX_TERM_BATCH: usize = 512;
    const ESTIMATED_TEXT_ROW_BYTES: usize = 64 * 1024;
    let started = Instant::now();

    let manifest = &base.manifest;
    let matches = manifest
        .text_indexes
        .iter()
        .filter(|descriptor| descriptor.name == selection.captured_state.index_name)
        .collect::<Vec<_>>();
    let [descriptor] = matches.as_slice() else {
        return Err(Error::precondition(
            "text DeltaRun catalog entry is missing or ambiguous",
        ));
    };
    let descriptor = (*descriptor).clone();
    let filters = crate::search_lsm::text_native_filter_properties(manifest, &descriptor);
    let selected_segments = &selection.captured_state.segments[selection.selected_segment_start..];
    if selected_segments.len() != selection.selected_descriptors.len() {
        return Err(Error::invariant(
            "text DeltaRun segment/descriptor cardinality differs",
        ));
    }

    let mut readers = Vec::with_capacity(selected_segments.len());
    let mut input_bytes = 0u64;
    for (segment, source_descriptor) in selected_segments
        .iter()
        .zip(&selection.selected_descriptors)
    {
        if segment.role != SearchSegmentRole::Delta || segment.format != SearchSegmentFormat::TextV4
        {
            return Err(Error::invariant(
                "text DeltaRun selected a non-FT4 delta source",
            ));
        }
        input_bytes = input_bytes
            .checked_add(source_descriptor.size_bytes)
            .ok_or_else(|| Error::invariant("text DeltaRun input bytes overflow"))?;
        let source = open_compaction_search_source(store.clone(), paths, source_descriptor).await?;
        let source: Arc<dyn SearchVersionRangeSource> = source;
        let reader = crate::sst::text::v4::TextV4Reader::open(
            source,
            source_descriptor.size_bytes,
            &selection.captured_state,
            segment,
        )
        .await
        .map_err(|error| Error::Corrupted {
            path: source_descriptor.path.clone(),
            detail: error.to_string(),
        })?;
        reader
            .verify_all()
            .await
            .map_err(|error| Error::Corrupted {
                path: source_descriptor.path.clone(),
                detail: error.to_string(),
            })?;
        readers.push(reader);
    }
    let version_readers = readers
        .iter()
        .map(|reader| reader.version_reader())
        .collect::<Vec<_>>();
    let mut version_merge = VersionRunMerge::open(version_readers).await?;
    let mut term_merge = TermDeltaRunMerge::open(readers.iter().collect::<Vec<_>>()).await?;
    let input_rows = version_merge.input_rows();
    if input_rows
        != selection
            .selected_descriptors
            .iter()
            .try_fold(0u64, |total, descriptor| {
                total
                    .checked_add(descriptor.row_count)
                    .ok_or_else(|| Error::invariant("text DeltaRun descriptor rows overflow"))
            })?
    {
        return Err(Error::Corrupted {
            path: "<search-delta-version-tables>".into(),
            detail: "text source version counts disagree with descriptors".into(),
        });
    }

    let SearchSegmentStats::Text {
        doc_count: SearchStatValue::Delta(doc_count_delta),
        total_len: SearchStatValue::Delta(total_len_delta),
        term_df_violation_count: 0,
    } = summed_delta_stats(&selection)?
    else {
        return Err(Error::invariant(
            "text DeltaRun selected invalid signed statistics",
        ));
    };
    let stats = crate::sst::text::v4::ReconciledTextDeltaStats {
        doc_count_delta,
        total_len_delta,
    };
    let mut config = crate::sst::text::v4::TextV4ExternalBuildConfig::from_env(
        crate::sst::text::v4::TextV4BuildOptions::default(),
    )?;
    config.memory_budget_bytes = super::per_collector_index_build_memory_bytes(1)?
        .ok_or_else(|| Error::invariant("text DeltaRun has no build memory share"))?;
    let batch_rows = MAX_RECONCILE_BATCH_ROWS
        .min((config.memory_budget_bytes / 8 / ESTIMATED_TEXT_ROW_BYTES).max(1));
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<TextDeltaBuilderMessage>(2);
    let build_state = selection.captured_state.clone();
    let build_context = crate::sst::text::v4::TextV4BuildContext {
        sst_id: Uuid::now_v7(),
        event_ranges: selection.selected_event_ranges.clone(),
        complete_filter_properties: filters.clone(),
    };
    let builder_handle = tokio::task::spawn_blocking(move || {
        let mut builder = crate::sst::text::v4::TextV4ExternalBuilder::with_config_reconciled(
            &build_state,
            build_context,
            config,
            stats,
        )?;
        while let Some(message) = receiver.blocking_recv() {
            match message {
                TextDeltaBuilderMessage::Winners(batch) => {
                    for (record, payload) in batch {
                        builder.push_reconciled(record, payload)?;
                    }
                }
                TextDeltaBuilderMessage::Terms(batch) => {
                    for term in batch {
                        builder.push_term_delta(term.term, term.delta_df)?;
                    }
                }
            }
        }
        builder.finish()
    });

    let empty_memtable = MemtableSnapshot::empty();
    let snapshot = crate::read::Snapshot::new(base.clone(), &empty_memtable, store, paths.clone());
    let mut live_total_len = 0u64;
    let producer_result: Result<()> = async {
        loop {
            let winners = version_merge.next_batch(batch_rows).await?;
            if winners.is_empty() {
                break;
            }
            let ids = winners
                .iter()
                .map(|record| NodeId::from_uuid(Uuid::from_bytes(record.node_id)))
                .collect::<Vec<_>>();
            let nodes = snapshot.batch_lookup_nodes("", &ids).await?;
            if nodes.len() != winners.len() {
                return Err(Error::invariant(
                    "text DeltaRun point-read returned the wrong cardinality",
                ));
            }
            let batch = winners
                .into_iter()
                .zip(nodes.iter())
                .map(|(record, node)| {
                    let payload =
                        reconcile_text_payload(&descriptor, &filters, record, node.as_ref())?;
                    if let Some(payload) = &payload {
                        live_total_len = live_total_len
                            .checked_add(crate::text::tokenize(&payload.text).len() as u64)
                            .ok_or_else(|| {
                                Error::invariant("text DeltaRun live token count overflows")
                            })?;
                    }
                    Ok((record, payload))
                })
                .collect::<Result<Vec<_>>>()?;
            sender
                .send(TextDeltaBuilderMessage::Winners(batch))
                .await
                .map_err(|_| {
                    Error::precondition(
                        "text DeltaRun builder stopped before consuming all winners",
                    )
                })?;
        }
        loop {
            let terms = term_merge.next_batch(MAX_TERM_BATCH).await?;
            if terms.is_empty() {
                break;
            }
            sender
                .send(TextDeltaBuilderMessage::Terms(terms))
                .await
                .map_err(|_| {
                    Error::precondition(
                        "text DeltaRun builder stopped before consuming all term statistics",
                    )
                })?;
        }
        Ok(())
    }
    .await;
    drop(sender);
    let artifact_result = builder_handle
        .await
        .map_err(|error| Error::invariant(format!("text DeltaRun builder panicked: {error}")))?;
    producer_result?;
    let artifact = artifact_result?.ok_or_else(|| {
        Error::invariant("non-empty text DeltaRun source merge produced no output artifact")
    })?;
    let touched_rows = version_merge.touched_rows();
    if touched_rows == 0 || touched_rows != artifact.output.segment.mutation_count {
        return Err(Error::invariant(
            "text DeltaRun output count disagrees with reconciled winners",
        ));
    }

    let segment = artifact.output.segment.clone();
    let file_name = format!(
        "{}-{}-{}.ft4",
        segment.sst_id.simple(),
        SstKind::TextIndex.path_tag(),
        descriptor.name
    );
    let output_descriptor = SstDescriptor {
        id: segment.sst_id,
        kind: SstKind::TextIndex,
        scope: descriptor.name,
        level: SstLevel(1),
        path: super::relative_sst_path(1, &file_name),
        size_bytes: artifact.len,
        row_count: segment.mutation_count,
        created_at: Utc::now(),
        min_key: artifact.output.version_table.min_node_id,
        max_key: artifact.output.version_table.max_node_id,
        min_lsn: segment.min_lsn,
        max_lsn: segment.max_lsn,
        schema_version_min: 0,
        schema_version_max: 0,
        property_stats: Vec::new(),
        kind_specific: KindSpecificStats::TextIndex {
            doc_count: segment.live_payload_count,
            term_count: 0,
            total_len: live_total_len,
        },
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        composite_equality_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    };
    let output_bytes = artifact.len;
    let scratch_bytes = artifact
        .metrics
        .spool_bytes_written
        .saturating_add(artifact.metrics.filter_spool_bytes);
    let live_rows = segment.live_payload_count;
    Ok(BuiltSearchCompaction {
        prepared: PreparedSearchCompaction {
            selection,
            output: Some(PreparedSearchPayload {
                descriptor: output_descriptor,
                segment,
            }),
            empty_proof_digest: None,
            metrics: SearchCompactionBuildMetrics {
                input_bytes,
                output_bytes,
                input_rows,
                touched_rows,
                live_rows,
                peak_workspace_bytes: artifact.metrics.peak_logical_memory_bytes as u64,
                scratch_bytes,
                spill_bytes: scratch_bytes,
                prepare_millis: elapsed_millis(started),
                peak_resident_input_bytes: 0,
            },
        },
        file: Some(artifact.file),
    })
}

pub(super) fn descriptor_path(paths: &NamespacePaths, descriptor: &SstDescriptor) -> Path {
    Path::from(format!(
        "{}/{}",
        paths.namespace_prefix().as_ref(),
        descriptor.path
    ))
}

fn validate_built_compaction(
    manifest: &Manifest,
    prepared: &PreparedSearchCompaction,
) -> Result<()> {
    verify_output(prepared)?;
    let rebased =
        rebase_prepared_search_compaction(manifest, prepared, &prepared.selection.captured_state)?;
    let selected = prepared.selection.selected_ids().collect::<HashSet<_>>();
    let mut projected = manifest.clone();
    projected
        .ssts
        .retain(|descriptor| !selected.contains(&descriptor.id));
    if let Some(output) = &prepared.output {
        projected.ssts.push(output.descriptor.clone());
    }
    let mut rebased = rebased;
    if prepared.selection.captured_state.status == SearchLsmStatus::Building
        && prepared.selection.mode == SearchCompactionMode::BasePrefix
    {
        let retired = complete_building_activation(
            &mut rebased,
            prepared.selection.captured_state.next_event_seq,
            prepared.empty_proof_digest,
            &projected.ssts,
        )?;
        projected
            .ssts
            .retain(|descriptor| !retired.contains(&descriptor.id));
    }
    let position = projected
        .search_lsm
        .iter()
        .position(|state| state.kind == rebased.kind && state.index_name == rebased.index_name)
        .ok_or_else(|| Error::invariant("prepared Search-LSM generation disappeared"))?;
    projected.search_lsm[position] = rebased;
    validate_search_lsm(&projected).map_err(|error| {
        Error::invariant(format!(
            "prepared physical Search-LSM output failed projected validation: {error}"
        ))
    })
}

fn winner_hasher(selection: &SearchCompactionSelection) -> Xxh3 {
    let mut digest = Xxh3::new();
    digest.update(b"NamiDB/SearchLsmBaseWinner/v1");
    digest.update(&[match selection.captured_state.kind {
        SearchLsmKind::Vector => 0,
        SearchLsmKind::Text => 1,
    }]);
    digest.update(selection.captured_state.generation_id.as_bytes());
    digest.update(selection.captured_state.index_name.as_bytes());
    digest.update(selection.captured_state.catalog_signature.as_bytes());
    digest.update(&selection.captured_coverage_digest.to_le_bytes());
    for range in &selection.selected_event_ranges {
        digest.update(&range.start.to_le_bytes());
        digest.update(&range.end.to_le_bytes());
    }
    digest
}

fn update_winner_digest(
    digest: &mut Xxh3,
    node_id: [u8; 16],
    lsn: u64,
    live: bool,
    payload_fingerprint: Option<u64>,
) {
    digest.update(&node_id);
    digest.update(&lsn.to_le_bytes());
    digest.update(&[u8::from(live)]);
    match payload_fingerprint {
        Some(fingerprint) => {
            digest.update(&[1]);
            digest.update(&fingerprint.to_le_bytes());
        }
        None => digest.update(&[0]),
    }
}

fn carries_label(
    label_id: Option<u32>,
    label: &str,
    record: &crate::flush::NodeWriteRecord,
    bucket_scope: &str,
) -> bool {
    match label_id {
        Some(label_id) => {
            record.labels.contains(&label_id) || (record.labels.is_empty() && bucket_scope == label)
        }
        None => record.labels.is_empty() && bucket_scope == label,
    }
}

fn native_filter_payload(
    properties: &BTreeMap<String, Value>,
    filters: &[String],
) -> BTreeMap<String, SearchFilterValue> {
    filters
        .iter()
        .filter_map(|property| {
            properties
                .get(property)
                .and_then(SearchFilterValue::from_value)
                .map(|value| (property.clone(), value))
        })
        .collect()
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
#[allow(clippy::large_enum_variant)] // one instance per index, never collected
enum SearchBuildWork {
    #[cfg(feature = "vector-index")]
    Vector(VectorBaseWork),
    #[cfg(feature = "text-index")]
    Text(TextBaseWork),
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl SearchBuildWork {
    fn observe(
        &mut self,
        node_id: [u8; 16],
        lsn: u64,
        record: Option<&crate::flush::NodeWriteRecord>,
        bucket_scope: &str,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "vector-index")]
            Self::Vector(work) => work.observe(node_id, lsn, record, bucket_scope),
            #[cfg(feature = "text-index")]
            Self::Text(work) => work.observe(node_id, lsn, record, bucket_scope),
        }
    }

    fn finish(self, input_bytes: u64, started: Instant) -> Result<Option<BuiltSearchCompaction>> {
        match self {
            #[cfg(feature = "vector-index")]
            Self::Vector(work) => work.finish(input_bytes, started),
            #[cfg(feature = "text-index")]
            Self::Text(work) => work.finish(input_bytes, started),
        }
    }
}

#[cfg(feature = "vector-index")]
struct VectorBaseWork {
    selection: SearchCompactionSelection,
    descriptor: crate::manifest::VectorIndexDescriptor,
    label_id: Option<u32>,
    filters: Vec<String>,
    builder: crate::sst::vector::v5::external::VectorV5ExternalCollector,
    digest: Xxh3,
}

#[cfg(feature = "vector-index")]
impl VectorBaseWork {
    fn observe(
        &mut self,
        node_id: [u8; 16],
        lsn: u64,
        record: Option<&crate::flush::NodeWriteRecord>,
        bucket_scope: &str,
    ) -> Result<()> {
        let Some(record) = record else {
            update_winner_digest(&mut self.digest, node_id, lsn, false, None);
            return Ok(());
        };
        let member = carries_label(self.label_id, &self.descriptor.label, record, bucket_scope);
        let Some(value) = member
            .then(|| record.properties.get(&self.descriptor.property))
            .flatten()
        else {
            update_winner_digest(&mut self.digest, node_id, lsn, true, None);
            return Ok(());
        };
        let Some(vector) = value.embedding_f32(self.descriptor.dim as usize) else {
            update_winner_digest(&mut self.digest, node_id, lsn, true, None);
            return Ok(());
        };
        let vector: &[f32] = &vector;
        if vector.len() != self.descriptor.dim as usize
            || vector.iter().any(|component| !component.is_finite())
        {
            update_winner_digest(&mut self.digest, node_id, lsn, true, None);
            return Ok(());
        }
        // The V5 builder skips zero-norm rows under Cosine because they are not
        // cosine-rankable, so pushing one here would make `input_rows` exceed
        // the indexed `point_count` and trip the count check in `finish`. The
        // node carries the label but contributes no indexable payload, which is
        // exactly the `(member, None)` winner shape used above.
        if self.descriptor.metric == crate::manifest::VectorMetric::Cosine
            && vector.iter().all(|component| *component == 0.0)
        {
            update_winner_digest(&mut self.digest, node_id, lsn, true, None);
            return Ok(());
        }
        let fingerprint_filters = native_filter_payload(&record.properties, &self.filters);
        let fingerprint = crate::sst::vector::v6::vector_v6_payload_fingerprint(
            &crate::sst::vector::v6::VectorV6Payload {
                vector: vector.to_vec(),
                filters: fingerprint_filters,
            },
        )?;
        update_winner_digest(&mut self.digest, node_id, lsn, true, Some(fingerprint));
        // Include every canonical key with Null for absence. This makes the
        // V5 footer advertise the complete filter schema even when a property
        // is absent from every row in one physical page.
        let filters = self
            .filters
            .iter()
            .map(|property| {
                (
                    property.clone(),
                    record
                        .properties
                        .get(property)
                        .cloned()
                        .unwrap_or(Value::Null),
                )
            })
            .collect::<BTreeMap<_, _>>();
        self.builder.push(node_id, vector, &filters)
    }

    fn finish(self, input_bytes: u64, started: Instant) -> Result<Option<BuiltSearchCompaction>> {
        let Self {
            selection,
            descriptor,
            filters,
            builder,
            digest,
            ..
        } = self;
        let input_rows = builder.input_rows();
        let artifact = builder.finish(&descriptor, super::vector_v5_build_options())?;
        finish_vector_base(
            selection,
            descriptor,
            filters,
            artifact,
            VectorBaseCounts {
                input_bytes,
                input_rows,
                touched_rows: input_rows,
                live_rows: input_rows,
                empty_proof_digest: digest.digest().max(1),
                merge_workspace_bytes: 0,
            },
            started,
        )
    }
}

/// Winner-stream counters of one vector base build, node scan or segment
/// merge.
#[cfg(feature = "vector-index")]
struct VectorBaseCounts {
    input_bytes: u64,
    input_rows: u64,
    touched_rows: u64,
    /// Live winners pushed into the V5 collector.
    live_rows: u64,
    /// Non-zero digest binding the winners, used when none is live.
    empty_proof_digest: u64,
    /// Merge-side workspace held before the collector's own build.
    merge_workspace_bytes: u64,
}

/// Turn a finished V5 collector into the prepared base (or empty proof).
#[cfg(feature = "vector-index")]
fn finish_vector_base(
    selection: SearchCompactionSelection,
    descriptor: crate::manifest::VectorIndexDescriptor,
    filters: Vec<String>,
    artifact: Option<crate::sst::vector::v5::external::VectorV5ExternalArtifact>,
    counts: VectorBaseCounts,
    started: Instant,
) -> Result<Option<BuiltSearchCompaction>> {
    let elapsed = elapsed_millis(started);
    let Some(artifact) = artifact else {
        if counts.live_rows != 0 {
            tracing::warn!(
                index = %descriptor.name,
                kind = "vector",
                live_rows = counts.live_rows,
                prepare_millis = elapsed,
                abort_reason = "vector_v5_requires_at_least_two_points",
                "abandoning physical Search-LSM compaction without changing its generation"
            );
            return Ok(None);
        }
        return Ok(Some(BuiltSearchCompaction {
            prepared: PreparedSearchCompaction {
                selection,
                output: None,
                empty_proof_digest: Some(counts.empty_proof_digest.max(1)),
                metrics: SearchCompactionBuildMetrics {
                    input_bytes: counts.input_bytes,
                    input_rows: counts.input_rows,
                    touched_rows: counts.touched_rows,
                    prepare_millis: elapsed,
                    ..Default::default()
                },
            },
            file: None,
        }));
    };
    let input_rows = counts.live_rows;
    let crate::sst::vector::v5::external::VectorV5ExternalArtifact {
        file,
        len,
        stats,
        metrics,
    } = artifact;
    if stats.point_count != input_rows || metrics.input_rows != input_rows {
        return Err(Error::invariant(
            "V5 Search-LSM base counts disagree with its winner stream",
        ));
    }
    let id = Uuid::now_v7();
    let file_name = format!(
        "{}-{}-{}.vg",
        id.simple(),
        SstKind::VectorGraph.path_tag(),
        descriptor.name
    );
    let corpus_max_lsn = selection
        .captured_state
        .coverage
        .iter()
        .map(|coverage| coverage.node_sst_max_lsn)
        .max()
        .unwrap_or(0);
    let output_descriptor = SstDescriptor {
        id,
        kind: SstKind::VectorGraph,
        scope: descriptor.name.clone(),
        level: SstLevel(1),
        path: super::relative_sst_path(1, &file_name),
        size_bytes: len,
        row_count: stats.point_count,
        created_at: Utc::now(),
        min_key: stats.min_node_id,
        max_key: stats.max_node_id,
        min_lsn: 0,
        max_lsn: corpus_max_lsn,
        schema_version_min: 0,
        schema_version_max: 0,
        property_stats: Vec::new(),
        kind_specific: KindSpecificStats::VectorGraph {
            dim: stats.dim,
            metric: stats.metric,
            point_count: stats.point_count,
            r: stats.r,
            l_build: stats.l_build,
            alpha: stats.alpha,
            entry_medoid: stats.entry_medoid,
        },
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        composite_equality_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    };
    let segment = SearchSegmentRef {
        sst_id: id,
        role: SearchSegmentRole::Base,
        format: SearchSegmentFormat::VectorV5Base,
        payload: SearchSegmentPayload::Complete,
        event_ranges: selection.selected_event_ranges.clone(),
        min_lsn: output_descriptor.min_lsn,
        max_lsn: output_descriptor.max_lsn,
        mutation_count: stats.point_count,
        live_payload_count: stats.point_count,
        suppress_count: 0,
        content_xxh3: legacy_base_content_fingerprint(
            &output_descriptor,
            SearchSegmentFormat::VectorV5Base,
            &filters,
        ),
        complete_filter_properties: filters,
        stats: SearchSegmentStats::Vector {
            live_count: SearchStatValue::Absolute(stats.point_count),
        },
        equal_lsn_conflict_count: 0,
    };
    Ok(Some(BuiltSearchCompaction {
        prepared: PreparedSearchCompaction {
            selection,
            output: Some(PreparedSearchPayload {
                descriptor: output_descriptor,
                segment,
            }),
            empty_proof_digest: None,
            metrics: SearchCompactionBuildMetrics {
                input_bytes: counts.input_bytes,
                output_bytes: len,
                input_rows: counts.input_rows,
                touched_rows: counts.touched_rows,
                live_rows: input_rows,
                peak_workspace_bytes: (metrics.peak_logical_memory_bytes as u64)
                    .max(counts.merge_workspace_bytes),
                scratch_bytes: metrics.scratch_bytes_written,
                spill_bytes: metrics.scratch_bytes_written,
                prepare_millis: elapsed,
                // A node scan stamps its Nodes residency centrally once the
                // whole merge's peak is known; a segment merge reads none.
                peak_resident_input_bytes: 0,
            },
        },
        file: Some(file),
    }))
}

#[cfg(feature = "text-index")]
struct TextBaseWork {
    selection: SearchCompactionSelection,
    descriptor: crate::manifest::TextIndexDescriptor,
    label_id: Option<u32>,
    filters: Vec<String>,
    builder: crate::sst::text::v4::TextV4ExternalBuilder,
    digest: Xxh3,
    live_rows: u64,
    live_total_len: u64,
}

#[cfg(feature = "text-index")]
impl TextBaseWork {
    fn observe(
        &mut self,
        node_id: [u8; 16],
        lsn: u64,
        record: Option<&crate::flush::NodeWriteRecord>,
        bucket_scope: &str,
    ) -> Result<()> {
        let Some(record) = record else {
            update_winner_digest(&mut self.digest, node_id, lsn, false, None);
            return Ok(());
        };
        if !carries_label(self.label_id, &self.descriptor.label, record, bucket_scope) {
            update_winner_digest(&mut self.digest, node_id, lsn, true, None);
            return Ok(());
        }
        let parts = self
            .descriptor
            .properties
            .iter()
            .filter_map(|property| match record.properties.get(property) {
                Some(Value::Str(value)) => Some(value.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if parts.is_empty() {
            update_winner_digest(&mut self.digest, node_id, lsn, true, None);
            return Ok(());
        }
        let payload = crate::sst::text::v4::TextV4Payload {
            text: parts.join(" "),
            filters: native_filter_payload(&record.properties, &self.filters),
        };
        let fingerprint = crate::sst::text::v4::text_v4_payload_fingerprint(&payload)?;
        update_winner_digest(&mut self.digest, node_id, lsn, true, Some(fingerprint));
        self.live_rows = self
            .live_rows
            .checked_add(1)
            .ok_or_else(|| Error::invariant("FT4 Search-LSM base row count overflows"))?;
        self.live_total_len = self
            .live_total_len
            .checked_add(crate::text::tokenize(&payload.text).len() as u64)
            .ok_or_else(|| Error::invariant("FT4 Search-LSM base token count overflows"))?;
        self.builder.push(crate::sst::text::v4::TextV4Mutation {
            node_id,
            lsn,
            before: None,
            after: Some(payload),
        })
    }

    fn finish(self, input_bytes: u64, started: Instant) -> Result<Option<BuiltSearchCompaction>> {
        let Self {
            selection,
            descriptor,
            builder,
            digest,
            live_rows,
            live_total_len,
            ..
        } = self;
        let artifact = builder.finish()?;
        let elapsed = elapsed_millis(started);
        let Some(artifact) = artifact else {
            if live_rows != 0 {
                return Err(Error::invariant(
                    "FT4 Search-LSM builder lost non-empty winner input",
                ));
            }
            return Ok(Some(BuiltSearchCompaction {
                prepared: PreparedSearchCompaction {
                    selection,
                    output: None,
                    empty_proof_digest: Some(digest.digest().max(1)),
                    metrics: SearchCompactionBuildMetrics {
                        input_bytes,
                        prepare_millis: elapsed,
                        ..Default::default()
                    },
                },
                file: None,
            }));
        };
        let crate::sst::text::v4::TextV4ExternalArtifact {
            file,
            len,
            output,
            metrics,
        } = artifact;
        let segment = output.segment;
        if segment.live_payload_count != live_rows
            || segment.mutation_count != live_rows
            || segment.suppress_count != 0
            || output.version_table.record_count != segment.mutation_count
            || output.version_table.live_count != segment.live_payload_count
            || output.version_table.suppress_count != segment.suppress_count
            || output.version_table.min_lsn != segment.min_lsn
            || output.version_table.max_lsn != segment.max_lsn
            || output.object_len != len
        {
            return Err(Error::invariant(
                "FT4 Search-LSM base footer/count binding disagrees with its winner stream",
            ));
        }
        let file_name = format!(
            "{}-{}-{}.ft4",
            segment.sst_id.simple(),
            SstKind::TextIndex.path_tag(),
            descriptor.name
        );
        let output_descriptor = SstDescriptor {
            id: segment.sst_id,
            kind: SstKind::TextIndex,
            scope: descriptor.name,
            level: SstLevel(1),
            path: super::relative_sst_path(1, &file_name),
            size_bytes: len,
            row_count: live_rows,
            created_at: Utc::now(),
            min_key: output.version_table.min_node_id,
            max_key: output.version_table.max_node_id,
            min_lsn: segment.min_lsn,
            max_lsn: segment.max_lsn,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: Vec::new(),
            kind_specific: KindSpecificStats::TextIndex {
                doc_count: live_rows,
                term_count: 0,
                total_len: live_total_len,
            },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            composite_equality_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        };
        Ok(Some(BuiltSearchCompaction {
            prepared: PreparedSearchCompaction {
                selection,
                output: Some(PreparedSearchPayload {
                    descriptor: output_descriptor,
                    segment,
                }),
                empty_proof_digest: None,
                metrics: SearchCompactionBuildMetrics {
                    input_bytes,
                    output_bytes: len,
                    input_rows: live_rows,
                    touched_rows: live_rows,
                    live_rows,
                    peak_workspace_bytes: metrics.peak_logical_memory_bytes as u64,
                    scratch_bytes: metrics
                        .spool_bytes_written
                        .saturating_add(metrics.filter_spool_bytes),
                    spill_bytes: metrics
                        .spool_bytes_written
                        .saturating_add(metrics.filter_spool_bytes),
                    // Stamped centrally once the whole merge's peak is known.
                    peak_resident_input_bytes: 0,
                    prepare_millis: elapsed,
                },
            },
            file: Some(file),
        }))
    }
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn build_search_compactions_blocking(
    manifest: &Manifest,
    selections: Vec<SearchCompactionSelection>,
    plan: Vec<SearchNodeSourcePlan>,
    mut bodies: tokio::sync::mpsc::Receiver<(usize, Bytes)>,
    input_bytes: u64,
    started: Instant,
) -> Result<Vec<BuiltSearchCompaction>> {
    let per_builder_memory = super::per_collector_index_build_memory_bytes(selections.len())?
        .ok_or_else(|| Error::invariant("Search-LSM compaction has no build memory share"))?;
    // `SHOW INDEXES` reports a Building generation's progress from this.
    let progress = crate::search_progress::begin_build(
        selections
            .iter()
            .map(|selection| selection.captured_state.generation_id),
        plan.iter()
            .fold(0u64, |total, entry| total.saturating_add(entry.row_count)),
    );
    let mut works = Vec::with_capacity(selections.len());
    for selection in selections {
        match selection.captured_state.kind {
            SearchLsmKind::Vector => {
                #[cfg(feature = "vector-index")]
                {
                    let matches = manifest
                        .vector_indexes
                        .iter()
                        .filter(|descriptor| descriptor.name == selection.captured_state.index_name)
                        .collect::<Vec<_>>();
                    let [descriptor] = matches.as_slice() else {
                        return Err(Error::precondition(
                            "vector Search-LSM compaction catalog entry is missing or ambiguous",
                        ));
                    };
                    let filters =
                        crate::search_lsm::vector_native_filter_properties(manifest, descriptor);
                    let mut config =
                        crate::sst::vector::v5::external::VectorV5ExternalBuildConfig::from_env()?;
                    config.memory_budget_bytes = per_builder_memory;
                    let digest = winner_hasher(&selection);
                    works.push(SearchBuildWork::Vector(VectorBaseWork {
                        label_id: manifest
                            .label_dict
                            .id(&descriptor.label)
                            .map(|label| label.0),
                        descriptor: (*descriptor).clone(),
                        filters,
                        builder: crate::sst::vector::v5::external::VectorV5ExternalCollector::new(
                            config,
                        )?,
                        selection,
                        digest,
                    }));
                }
                #[cfg(not(feature = "vector-index"))]
                {
                    return Err(Error::precondition(
                        "vector Search-LSM compaction requires the vector-index feature",
                    ));
                }
            }
            SearchLsmKind::Text => {
                #[cfg(feature = "text-index")]
                {
                    let matches = manifest
                        .text_indexes
                        .iter()
                        .filter(|descriptor| descriptor.name == selection.captured_state.index_name)
                        .collect::<Vec<_>>();
                    let [descriptor] = matches.as_slice() else {
                        return Err(Error::precondition(
                            "text Search-LSM compaction catalog entry is missing or ambiguous",
                        ));
                    };
                    let filters =
                        crate::search_lsm::text_native_filter_properties(manifest, descriptor);
                    let sst_id = Uuid::now_v7();
                    let mut config = crate::sst::text::v4::TextV4ExternalBuildConfig::from_env(
                        crate::sst::text::v4::TextV4BuildOptions::default(),
                    )?;
                    config.memory_budget_bytes = per_builder_memory;
                    let builder = crate::sst::text::v4::TextV4ExternalBuilder::with_config_base(
                        &selection.captured_state,
                        crate::sst::text::v4::TextV4BuildContext {
                            sst_id,
                            event_ranges: selection.selected_event_ranges.clone(),
                            complete_filter_properties: filters.clone(),
                        },
                        config,
                    )?;
                    let digest = winner_hasher(&selection);
                    works.push(SearchBuildWork::Text(TextBaseWork {
                        label_id: manifest
                            .label_dict
                            .id(&descriptor.label)
                            .map(|label| label.0),
                        descriptor: (*descriptor).clone(),
                        filters,
                        builder,
                        selection,
                        digest,
                        live_rows: 0,
                        live_total_len: 0,
                    }));
                }
                #[cfg(not(feature = "text-index"))]
                {
                    return Err(Error::precondition(
                        "text Search-LSM compaction requires the text-index feature",
                    ));
                }
            }
        }
    }

    // Lazy activation, eager release. A cursor exists only between its heap
    // entry activating and its last row being consumed; dropping it unmaps the
    // body and frees its unlinked spool immediately. The comparator matches
    // the producer's send order exactly, so each `blocking_recv` yields the
    // body the activation loop is asking for.
    let mut cursors: Vec<Option<SearchNodeCursor>> = (0..plan.len()).map(|_| None).collect();
    let mut unopened = BinaryHeap::with_capacity(plan.len());
    for (source, entry) in plan.iter().enumerate() {
        unopened.push(Reverse((entry.min_key, source)));
    }
    let mut resident_input_bytes = 0u64;
    let mut peak_resident_input_bytes = 0u64;
    let mut heap: BinaryHeap<Reverse<super::NodeHeapEntry>> = BinaryHeap::with_capacity(plan.len());
    let mut last_id = None;
    loop {
        while let Some(Reverse((hinted_min, source))) = unopened.peek().copied() {
            let active_min = heap.peek().map(|entry| entry.0.id);
            if active_min.is_some_and(|active| hinted_min > active) {
                break;
            }
            unopened.pop();
            let (ordinal, body) = bodies.blocking_recv().ok_or_else(|| {
                Error::invariant(
                    "Search-LSM base fetch stream ended before all planned Nodes SSTs arrived",
                )
            })?;
            if ordinal != source {
                return Err(Error::invariant(
                    "Search-LSM base fetch delivery disagrees with activation order",
                ));
            }
            let entry = &plan[source];
            let cursor = super::NodeSourceCursor::open(&entry.label_def, body)?;
            let mut opened = SearchNodeCursor {
                cursor,
                label_def: entry.label_def.clone(),
                scope: entry.scope.clone(),
            };
            resident_input_bytes = resident_input_bytes.saturating_add(entry.size_bytes);
            peak_resident_input_bytes = peak_resident_input_bytes.max(resident_input_bytes);
            opened.cursor.ensure_positioned()?;
            let Some((id, lsn)) = opened.cursor.peek() else {
                return Err(Error::invariant(
                    "manifest references an empty Nodes SST during Search-LSM compaction",
                ));
            };
            if id != hinted_min {
                return Err(Error::Corrupted {
                    path: "<search-lsm-compaction-input>".into(),
                    detail: "Nodes SST first key disagrees with manifest min_key".into(),
                });
            }
            heap.push(Reverse(super::NodeHeapEntry {
                id,
                lsn,
                src: source,
            }));
            cursors[source] = Some(opened);
        }
        let Some(Reverse(entry)) = heap.pop() else {
            break;
        };
        progress.scanned(1);
        let exhausted = {
            let source = cursors[entry.src]
                .as_mut()
                .ok_or_else(|| Error::invariant("Search-LSM merge popped an inactive source"))?;
            if last_id != Some(entry.id) {
                last_id = Some(entry.id);
                let (row, record) = source.cursor.materialize_current(&source.label_def)?;
                let record = if matches!(row.op, MemOp::Tombstone) {
                    None
                } else {
                    record.as_ref()
                };
                for work in &mut works {
                    work.observe(row.id, row.lsn, record, &source.scope)?;
                }
            }
            source.cursor.advance()?;
            match source.cursor.peek() {
                Some((id, lsn)) => {
                    heap.push(Reverse(super::NodeHeapEntry {
                        id,
                        lsn,
                        src: entry.src,
                    }));
                    false
                }
                None => true,
            }
        };
        if exhausted {
            cursors[entry.src] = None;
            resident_input_bytes = resident_input_bytes.saturating_sub(plan[entry.src].size_bytes);
        }
    }

    progress.scan_finished();
    let mut finished = works
        .into_iter()
        .filter_map(|work| work.finish(input_bytes, started).transpose())
        .collect::<Result<Vec<_>>>()?;
    for built in &mut finished {
        built.prepared.metrics.peak_resident_input_bytes = peak_resident_input_bytes;
    }
    Ok(finished)
}

#[cfg(not(any(feature = "vector-index", feature = "text-index")))]
fn build_search_compactions_blocking(
    _manifest: &Manifest,
    selections: Vec<SearchCompactionSelection>,
    _plan: Vec<SearchNodeSourcePlan>,
    _bodies: tokio::sync::mpsc::Receiver<(usize, Bytes)>,
    _input_bytes: u64,
    _started: Instant,
) -> Result<Vec<BuiltSearchCompaction>> {
    if selections.is_empty() {
        Ok(Vec::new())
    } else {
        Err(Error::precondition(
            "physical Search-LSM compaction requires a search-index build feature",
        ))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    use std::ops::Range;
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    use std::sync::Arc;

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    use bytes::Bytes;
    use chrono::Utc;
    use namidb_core::{DataType, LabelDef, PropertyDef, SchemaBuilder};
    use uuid::Uuid;

    use super::*;
    use crate::fence::Epoch;
    use crate::manifest::{
        KindSpecificStats, SstLevel, VectorIndexDescriptor, VectorMetric, VectorQuantization,
    };
    use crate::search_lsm::{
        legacy_base_content_fingerprint, search_barrier_descriptor, CoverageDisposition,
        SearchCoverage, SearchSegmentFormat, SearchSegmentPayload, SearchSegmentStats,
        SearchStatValue,
    };

    fn descriptor(
        id: Uuid,
        kind: SstKind,
        scope: &str,
        min_lsn: u64,
        max_lsn: u64,
        rows: u64,
    ) -> SstDescriptor {
        SstDescriptor {
            id,
            kind,
            scope: scope.into(),
            level: SstLevel::L0,
            path: format!("sst/level0/{id}.body"),
            size_bytes: 100,
            row_count: rows,
            created_at: Utc::now(),
            min_key: [id.as_bytes()[15]; 16],
            max_key: [id.as_bytes()[15]; 16],
            min_lsn,
            max_lsn,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: Vec::new(),
            kind_specific: match kind {
                SstKind::Nodes => KindSpecificStats::Nodes { tombstone_count: 0 },
                SstKind::VectorGraph => KindSpecificStats::VectorGraph {
                    dim: 2,
                    metric: "cosine".into(),
                    point_count: rows,
                    r: 8,
                    l_build: 16,
                    alpha: 1.2,
                    #[cfg(feature = "vector-index")]
                    entry_medoid: crate::sst::vector::v5::VECTOR_V5_KMEANS_LAYOUT,
                    #[cfg(not(feature = "vector-index"))]
                    entry_medoid: 0,
                },
                _ => unreachable!(),
            },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            composite_equality_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        }
    }

    fn delta(id: Uuid, event: u64, lsn: u64) -> SearchSegmentRef {
        SearchSegmentRef {
            sst_id: id,
            role: SearchSegmentRole::Delta,
            format: SearchSegmentFormat::VectorV6,
            payload: SearchSegmentPayload::Complete,
            event_ranges: vec![SearchEventRange::new(event, event + 1)],
            min_lsn: lsn,
            max_lsn: lsn,
            mutation_count: 1,
            live_payload_count: 1,
            suppress_count: 0,
            content_xxh3: lsn + 100,
            complete_filter_properties: Vec::new(),
            stats: SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(1),
            },
            equal_lsn_conflict_count: 0,
        }
    }

    fn fixture() -> Manifest {
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![PropertyDef::new(
                    "embedding",
                    DataType::FloatVector { dim: 2 },
                    false,
                )
                .unwrap()],
            })
            .unwrap()
            .build();
        manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_vec".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });

        let mut segments = Vec::new();
        let mut coverage = Vec::new();
        for event in 0..7u64 {
            let node = Uuid::from_u128(0x100 + u128::from(event));
            manifest.ssts.push(descriptor(
                node,
                SstKind::Nodes,
                "",
                event + 1,
                event + 1,
                1,
            ));
            if event == 3 {
                coverage.push(SearchCoverage {
                    node_sst_id: node,
                    node_sst_max_lsn: event + 1,
                    event_ranges: vec![SearchEventRange::new(event, event + 1)],
                    disposition: CoverageDisposition::ProvenEmpty {
                        classifier_version: 1,
                        before_after_digest: 33,
                    },
                });
                continue;
            }
            let segment_id = Uuid::from_u128(0x200 + u128::from(event));
            manifest.ssts.push(descriptor(
                segment_id,
                SstKind::VectorGraph,
                "doc_vec",
                event + 1,
                event + 1,
                1,
            ));
            segments.push(delta(segment_id, event, event + 1));
            coverage.push(SearchCoverage {
                node_sst_id: node,
                node_sst_max_lsn: event + 1,
                event_ranges: vec![SearchEventRange::new(event, event + 1)],
                disposition: CoverageDisposition::Segment,
            });
        }
        let barrier = Uuid::from_u128(0x300);
        let state = SearchLsmState {
            index_name: "doc_vec".into(),
            kind: SearchLsmKind::Vector,
            catalog_signature: crate::search_lsm::vector_catalog_signature(
                &manifest,
                &manifest.vector_indexes[0],
            ),
            generation_id: Uuid::from_u128(0x400),
            status: SearchLsmStatus::Active,
            next_event_seq: 7,
            base_frontier: None,
            segments,
            proven_empty_event_ranges: vec![SearchEventRange::new(3, 4)],
            coverage,
            compat_barrier_sst_id: Some(barrier),
            equal_lsn_conflict_count: 0,
        };
        let barrier_body = crate::search_lsm::encode_search_barrier(&state).unwrap();
        manifest.ssts.push(search_barrier_descriptor(
            &state,
            barrier,
            SstLevel::L0,
            format!("sst/level0/{barrier}.slb"),
            barrier_body.len() as u64,
        ));
        manifest.search_lsm.push(state);
        validate_search_lsm(&manifest).unwrap();
        manifest
    }

    fn prepared(selection: SearchCompactionSelection) -> PreparedSearchCompaction {
        let id = Uuid::from_u128(0x500);
        let mut output_descriptor = descriptor(id, SstKind::VectorGraph, "doc_vec", 0, 7, 6);
        output_descriptor.level = SstLevel(1);
        let filters = Vec::new();
        let output_segment = SearchSegmentRef {
            sst_id: id,
            role: SearchSegmentRole::Base,
            format: SearchSegmentFormat::VectorV5Base,
            payload: SearchSegmentPayload::Complete,
            event_ranges: vec![SearchEventRange::new(0, 7)],
            min_lsn: 0,
            max_lsn: 7,
            mutation_count: 6,
            live_payload_count: 6,
            suppress_count: 0,
            content_xxh3: legacy_base_content_fingerprint(
                &output_descriptor,
                SearchSegmentFormat::VectorV5Base,
                &filters,
            ),
            complete_filter_properties: filters,
            stats: SearchSegmentStats::Vector {
                live_count: SearchStatValue::Absolute(6),
            },
            equal_lsn_conflict_count: 0,
        };
        PreparedSearchCompaction {
            selection,
            output: Some(PreparedSearchPayload {
                descriptor: output_descriptor,
                segment: output_segment,
            }),
            empty_proof_digest: None,
            metrics: SearchCompactionBuildMetrics::default(),
        }
    }

    fn prepared_delta(selection: SearchCompactionSelection) -> PreparedSearchCompaction {
        assert_eq!(selection.mode, SearchCompactionMode::DeltaRun);
        let id = Uuid::from_u128(0x501);
        let selected = &selection.captured_state.segments[selection.selected_segment_start..];
        let mutation_count = selected.iter().map(|segment| segment.mutation_count).sum();
        let live_payload_count = selected
            .iter()
            .map(|segment| segment.live_payload_count)
            .sum();
        let suppress_count = selected.iter().map(|segment| segment.suppress_count).sum();
        let min_lsn = selected
            .iter()
            .map(|segment| segment.min_lsn)
            .min()
            .unwrap();
        let max_lsn = selected
            .iter()
            .map(|segment| segment.max_lsn)
            .max()
            .unwrap();
        let mut output_descriptor = descriptor(
            id,
            SstKind::VectorGraph,
            "doc_vec",
            min_lsn,
            max_lsn,
            mutation_count,
        );
        output_descriptor.level = SstLevel(1);
        let output_segment = SearchSegmentRef {
            sst_id: id,
            role: SearchSegmentRole::Delta,
            format: SearchSegmentFormat::VectorV6,
            payload: SearchSegmentPayload::Complete,
            event_ranges: selection.selected_event_ranges.clone(),
            min_lsn,
            max_lsn,
            mutation_count,
            live_payload_count,
            suppress_count,
            content_xxh3: 0x501,
            complete_filter_properties: Vec::new(),
            stats: summed_delta_stats(&selection).unwrap(),
            equal_lsn_conflict_count: 0,
        };
        PreparedSearchCompaction {
            selection,
            output: Some(PreparedSearchPayload {
                descriptor: output_descriptor,
                segment: output_segment,
            }),
            empty_proof_digest: None,
            metrics: SearchCompactionBuildMetrics::default(),
        }
    }

    /// Active vector generation of plain deltas: `(size_bytes, rows)` each,
    /// oldest first, every delta created `age_secs` ago.
    fn sized_delta_fixture(deltas: &[(u64, u64)], age_secs: i64) -> Manifest {
        let mut manifest = fixture();
        manifest
            .ssts
            .retain(|descriptor| descriptor.kind != SstKind::VectorGraph);
        manifest
            .ssts
            .retain(|descriptor| descriptor.kind != SstKind::Nodes);
        manifest.search_lsm.clear();
        let created_at = Utc::now() - chrono::Duration::seconds(age_secs);
        let mut segments = Vec::new();
        let mut coverage = Vec::new();
        for (event, (size_bytes, rows)) in deltas.iter().enumerate() {
            let event = event as u64;
            let node = Uuid::from_u128(0x1000 + u128::from(event));
            manifest.ssts.push(descriptor(
                node,
                SstKind::Nodes,
                "",
                event + 1,
                event + 1,
                *rows,
            ));
            let segment_id = Uuid::from_u128(0x2000 + u128::from(event));
            let mut segment_descriptor = descriptor(
                segment_id,
                SstKind::VectorGraph,
                "doc_vec",
                event + 1,
                event + 1,
                *rows,
            );
            // Sizes are given in MiB, above the small-delta floor.
            segment_descriptor.size_bytes = *size_bytes * 1024 * 1024;
            segment_descriptor.created_at = created_at;
            manifest.ssts.push(segment_descriptor);
            let mut segment = delta(segment_id, event, event + 1);
            segment.mutation_count = *rows;
            segment.live_payload_count = *rows;
            segment.stats = SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(*rows as i64),
            };
            segments.push(segment);
            coverage.push(SearchCoverage {
                node_sst_id: node,
                node_sst_max_lsn: event + 1,
                event_ranges: vec![SearchEventRange::new(event, event + 1)],
                disposition: CoverageDisposition::Segment,
            });
        }
        let barrier = Uuid::from_u128(0x3000);
        let state = SearchLsmState {
            index_name: "doc_vec".into(),
            kind: SearchLsmKind::Vector,
            catalog_signature: crate::search_lsm::vector_catalog_signature(
                &manifest,
                &manifest.vector_indexes[0],
            ),
            generation_id: Uuid::from_u128(0x4000),
            status: SearchLsmStatus::Active,
            next_event_seq: deltas.len() as u64,
            base_frontier: None,
            segments,
            proven_empty_event_ranges: Vec::new(),
            coverage,
            compat_barrier_sst_id: Some(barrier),
            equal_lsn_conflict_count: 0,
        };
        let barrier_body = crate::search_lsm::encode_search_barrier(&state).unwrap();
        manifest.ssts.push(search_barrier_descriptor(
            &state,
            barrier,
            SstLevel::L0,
            format!("sst/level0/{barrier}.slb"),
            barrier_body.len() as u64,
        ));
        manifest.search_lsm.push(state);
        validate_search_lsm(&manifest).unwrap();
        manifest
    }

    #[test]
    fn delta_runs_are_size_tiered_and_leave_large_consolidated_deltas_alone() {
        let policy = delta_policy(4, 24);
        // Three fresh flushes behind a large consolidated delta: not due.
        let manifest = sized_delta_fixture(&[(4_000, 40), (100, 1), (100, 1), (100, 1)], 0);
        assert!(select_search_compactions(&manifest, policy)
            .unwrap()
            .is_empty());

        // A fourth equal flush makes the run due; the large delta stays.
        let manifest =
            sized_delta_fixture(&[(4_000, 40), (100, 1), (100, 1), (100, 1), (100, 1)], 0);
        let selection = select_search_compactions(&manifest, policy)
            .unwrap()
            .remove(0);
        assert_eq!(selection.mode, SearchCompactionMode::DeltaRun);
        assert_eq!(selection.selected_segment_start, 1);
        assert_eq!(selection.selected_descriptors.len(), 4);

        // The merged output then waits for peers of its own size: the next
        // run of small flushes does not rewrite it...
        let manifest = sized_delta_fixture(
            &[(400, 4), (100, 1), (100, 1), (100, 1), (100, 1), (100, 1)],
            0,
        );
        let selection = select_search_compactions(&manifest, policy)
            .unwrap()
            .remove(0);
        assert_eq!(selection.selected_segment_start, 2);
        assert_eq!(selection.selected_descriptors.len(), 4);
        // ...and four outputs of one size merge into the next tier.
        let manifest = sized_delta_fixture(&[(400, 4), (400, 4), (400, 4), (400, 4)], 0);
        let selection = select_search_compactions(&manifest, policy)
            .unwrap()
            .remove(0);
        assert_eq!(selection.selected_segment_start, 0);
        assert_eq!(selection.selected_descriptors.len(), 4);
    }

    #[test]
    fn segment_pressure_merges_the_newest_deltas_whatever_their_sizes() {
        // Strictly growing sizes never form a tier...
        let sizes = [(9_000, 9), (5_000, 5), (3_000, 3), (1_000, 1)];
        let policy = delta_policy(4, 24);
        assert!(
            select_search_compactions(&sized_delta_fixture(&sizes, 0), policy)
                .unwrap()
                .is_empty()
        );
        // ...but at the pressure mark (2 x trigger) the newest run is merged.
        let many = [
            (90_000, 90),
            (50_000, 50),
            (30_000, 30),
            (17_000, 17),
            (9_000, 9),
            (5_000, 5),
            (3_000, 3),
            (1_000, 1),
        ];
        let selection = select_search_compactions(&sized_delta_fixture(&many, 0), policy)
            .unwrap()
            .remove(0);
        assert_eq!(selection.mode, SearchCompactionMode::DeltaRun);
        assert_eq!(selection.selected_segment_start, 4);
    }

    #[test]
    fn flat_vector_rows_fold_into_a_clustered_base() {
        let policy = SearchCompactionPolicy {
            vector_base_delta_rows: 1_000,
            vector_base_max_input_bytes: u64::MAX,
            ..delta_policy(4, 24)
        };
        // Below the row bound, a busy generation keeps its flat deltas.
        let manifest = sized_delta_fixture(&[(100, 300), (100, 300)], 0);
        assert!(select_search_compactions(&manifest, policy)
            .unwrap()
            .is_empty());
        // At the bound the deltas are rebuilt into the clustered base: every
        // query scans flat deltas in full.
        let manifest = sized_delta_fixture(&[(100, 600), (100, 600)], 0);
        let selection = select_search_compactions(&manifest, policy)
            .unwrap()
            .remove(0);
        assert_eq!(selection.mode, SearchCompactionMode::BasePrefix);
        // Not when the rebuild would read more Nodes bytes than memory allows.
        let tight = SearchCompactionPolicy {
            vector_base_max_input_bytes: 1,
            ..policy
        };
        assert!(select_search_compactions(&manifest, tight)
            .unwrap()
            .is_empty());
        // An idle generation folds a smaller tail too...
        let manifest = sized_delta_fixture(&[(100, 300), (100, 300)], 0);
        assert!(select_search_compactions(&manifest, policy)
            .unwrap()
            .is_empty());
        let idle = sized_delta_fixture(
            &[
                (100, VECTOR_BASE_IDLE_MIN_ROWS / 2),
                (100, VECTOR_BASE_IDLE_MIN_ROWS / 2),
            ],
            VECTOR_BASE_IDLE_SECS + 5,
        );
        let selection = select_search_compactions(
            &idle,
            SearchCompactionPolicy {
                vector_base_delta_rows: u64::MAX,
                vector_base_max_input_bytes: u64::MAX,
                ..policy
            },
        )
        .unwrap()
        .remove(0);
        assert_eq!(selection.mode, SearchCompactionMode::BasePrefix);
        // ...but not a few stray rows.
        let trickle = sized_delta_fixture(&[(100, 3), (100, 3)], VECTOR_BASE_IDLE_SECS + 5);
        assert!(select_search_compactions(&trickle, policy)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_base_built_before_kmeans_splits_is_rebuilt_once() {
        let policy = delta_policy(6, 8);
        let mut manifest = compacted_single_base_fixture();
        assert!(select_search_compactions(&manifest, policy)
            .unwrap()
            .is_empty());
        let base = manifest.search_lsm[0].segments[0].sst_id;
        let untag = |manifest: &mut Manifest, metric: &str| {
            let descriptor = manifest
                .ssts
                .iter_mut()
                .find(|descriptor| descriptor.id == base)
                .unwrap();
            descriptor.kind_specific = KindSpecificStats::VectorGraph {
                dim: 2,
                metric: metric.into(),
                point_count: descriptor.row_count,
                r: 8,
                l_build: 16,
                alpha: 1.2,
                entry_medoid: 0,
            };
            let descriptor = descriptor.clone();
            let segment = &mut manifest.search_lsm[0].segments[0];
            if matches!(segment.format, SearchSegmentFormat::VectorV5Base) {
                segment.content_xxh3 = crate::search_lsm::legacy_base_content_fingerprint(
                    &descriptor,
                    segment.format,
                    &segment.complete_filter_properties,
                );
            }
        };
        untag(&mut manifest, "cosine");
        let selection = select_search_compactions(&manifest, policy)
            .unwrap()
            .remove(0);
        assert_eq!(selection.mode, SearchCompactionMode::BasePrefix);
        // Within the same input bound as any vector rebuild.
        let tight = SearchCompactionPolicy {
            vector_base_max_input_bytes: 1,
            ..policy
        };
        assert!(select_search_compactions(&manifest, tight)
            .unwrap()
            .is_empty());
        // Dot-product bases keep direction splits and are left alone.
        untag(&mut manifest, "dot");
        let state = &manifest.search_lsm[0];
        let descriptors: Vec<SstDescriptor> = state
            .segments
            .iter()
            .map(|segment| {
                manifest
                    .ssts
                    .iter()
                    .find(|descriptor| descriptor.id == segment.sst_id)
                    .unwrap()
                    .clone()
            })
            .collect();
        assert!(!vector_base_layout_stale(state, &descriptors));
        untag(&mut manifest, "euclidean");
        let state = &manifest.search_lsm[0];
        let descriptors: Vec<SstDescriptor> = state
            .segments
            .iter()
            .map(|segment| {
                manifest
                    .ssts
                    .iter()
                    .find(|descriptor| descriptor.id == segment.sst_id)
                    .unwrap()
                    .clone()
            })
            .collect();
        assert!(vector_base_layout_stale(state, &descriptors));
    }

    fn delta_policy(trigger: usize, hard_cap: usize) -> SearchCompactionPolicy {
        SearchCompactionPolicy {
            delta_trigger_segments: trigger,
            hard_segment_cap: hard_cap,
            base_compact_bytes: u64::MAX,
            base_stale_percent: u64::MAX,
            force_base: false,
            vector_base_delta_rows: u64::MAX,
            vector_base_max_input_bytes: u64::MAX,
        }
    }

    fn base_policy(trigger: usize, hard_cap: usize) -> SearchCompactionPolicy {
        SearchCompactionPolicy {
            force_base: true,
            ..delta_policy(trigger, hard_cap)
        }
    }

    fn compacted_single_base_fixture() -> Manifest {
        let basis = fixture();
        let selection = select_search_compactions(&basis, base_policy(6, 8))
            .unwrap()
            .remove(0);
        let prepared = prepared(selection);
        let rebased =
            rebase_prepared_search_compaction(&basis, &prepared, &basis.search_lsm[0]).unwrap();
        let selected = prepared.selection.selected_ids().collect::<HashSet<_>>();
        let output_descriptor = prepared.output.as_ref().unwrap().descriptor.clone();
        let mut projected = basis;
        projected
            .ssts
            .retain(|descriptor| !selected.contains(&descriptor.id));
        projected.ssts.push(output_descriptor);
        projected.search_lsm[0] = rebased;
        validate_search_lsm(&projected).unwrap();
        projected
    }

    fn two_delta_fixture() -> Manifest {
        let mut manifest = fixture();
        let state = &manifest.search_lsm[0];
        let keep = state.segments[..2]
            .iter()
            .map(|segment| segment.sst_id)
            .chain(
                state.coverage[..2]
                    .iter()
                    .map(|coverage| coverage.node_sst_id),
            )
            .chain(state.compat_barrier_sst_id)
            .collect::<HashSet<_>>();
        manifest
            .ssts
            .retain(|descriptor| keep.contains(&descriptor.id));
        let state = &mut manifest.search_lsm[0];
        state.segments.truncate(2);
        state.coverage.truncate(2);
        state.proven_empty_event_ranges.clear();
        state.next_event_seq = 2;
        validate_search_lsm(&manifest).unwrap();
        manifest
    }

    #[test]
    fn env_segment_policy_rejects_one_and_every_accepted_default_can_progress() {
        for name in [SEARCH_LSM_MAX_SEGMENTS_ENV, SEARCH_LSM_COMPACT_SEGMENTS_ENV] {
            assert!(parse_segment_value(name, "0").is_err());
            assert!(parse_segment_value(name, "1").is_err());
            assert_eq!(parse_segment_value(name, " 2 ").unwrap(), 2);
            assert_eq!(
                parse_segment_value(name, "999").unwrap(),
                MAX_ACTIVE_SEARCH_SEGMENTS
            );
            assert!(parse_segment_value(name, "many").is_err());
        }
        for hard_cap in MIN_SEARCH_LSM_SEGMENTS..=MAX_ACTIVE_SEARCH_SEGMENTS {
            let trigger = default_delta_trigger(hard_cap);
            assert!(
                (MIN_SEARCH_LSM_SEGMENTS..=hard_cap).contains(&trigger),
                "cap {hard_cap} produced trigger {trigger}"
            );
        }
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    #[derive(Debug)]
    struct MemoryVersionSource {
        body: Bytes,
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    #[async_trait::async_trait]
    impl SearchVersionRangeSource for MemoryVersionSource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
            let start = usize::try_from(range.start).unwrap();
            let end = usize::try_from(range.end).unwrap();
            self.body
                .get(start..end)
                .map(Bytes::copy_from_slice)
                .ok_or_else(|| Error::invariant("test version range is out of bounds"))
        }
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    async fn version_reader(records: Vec<SearchVersionRecord>) -> SearchVersionTableReader {
        let (body, reference) =
            crate::sst::search_delta::encode_search_version_table(records).unwrap();
        SearchVersionTableReader::open(Arc::new(MemoryVersionSource { body }), reference)
            .await
            .unwrap()
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    #[tokio::test]
    async fn version_run_merge_keeps_highest_lsn_and_rejects_equal_lsn_conflict() {
        let first = version_reader(vec![
            SearchVersionRecord::live(1_u128.to_be_bytes(), 10, 101, 0),
            SearchVersionRecord::suppress(
                2_u128.to_be_bytes(),
                30,
                crate::sst::search_delta::search_suppress_fingerprint(),
            ),
        ])
        .await;
        let second = version_reader(vec![
            SearchVersionRecord::live(1_u128.to_be_bytes(), 20, 202, 7),
            SearchVersionRecord::live(3_u128.to_be_bytes(), 40, 303, 8),
        ])
        .await;
        let mut merge = VersionRunMerge::open(vec![&first, &second]).await.unwrap();
        let winners = merge.next_batch(8).await.unwrap();
        assert_eq!(winners.len(), 3);
        assert_eq!(winners[0].node_id, 1_u128.to_be_bytes());
        assert_eq!(winners[0].lsn, 20);
        assert_eq!(winners[0].payload_fingerprint, 202);
        assert_eq!(winners[1].node_id, 2_u128.to_be_bytes());
        assert_eq!(winners[2].node_id, 3_u128.to_be_bytes());
        assert_eq!(merge.input_rows(), 4);
        assert_eq!(merge.touched_rows(), 3);
        assert!(merge.next_batch(8).await.unwrap().is_empty());

        let conflict_left = version_reader(vec![SearchVersionRecord::live(
            9_u128.to_be_bytes(),
            55,
            1,
            0,
        )])
        .await;
        let conflict_right = version_reader(vec![SearchVersionRecord::live(
            9_u128.to_be_bytes(),
            55,
            2,
            0,
        )])
        .await;
        let mut conflict = VersionRunMerge::open(vec![&conflict_left, &conflict_right])
            .await
            .unwrap();
        let error = conflict.next_batch(1).await.unwrap_err();
        assert!(matches!(error, Error::Corrupted { .. }));
    }

    #[test]
    fn selection_triggers_before_cap_and_is_deterministic() {
        let manifest = fixture();
        let policy = delta_policy(6, 8);
        let first = select_search_compactions(&manifest, policy).unwrap();
        let second = select_search_compactions(&manifest, policy).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].mode, SearchCompactionMode::DeltaRun);
        assert_eq!(first[0].selected_segment_start, 0);
        assert_eq!(first[0].selected_descriptors.len(), 6);
        assert_eq!(
            first[0].selected_event_ranges,
            vec![SearchEventRange::new(0, 3), SearchEventRange::new(4, 7)]
        );
        assert_ne!(first[0].captured_coverage_digest, 0);
        assert!(search_compaction_needed(&manifest, policy));

        let later = delta_policy(7, 8);
        assert!(select_search_compactions(&manifest, later)
            .unwrap()
            .is_empty());
        assert!(!search_compaction_needed(&manifest, later));

        let forced = select_search_compactions(&manifest, base_policy(6, 8))
            .unwrap()
            .remove(0);
        assert_eq!(forced.mode, SearchCompactionMode::BasePrefix);
        assert_eq!(
            forced.selected_event_ranges,
            vec![SearchEventRange::new(0, 7)]
        );
    }

    #[test]
    fn cap_two_has_progress_for_base_tail_and_delta_only_states() {
        let mut base_tail = compacted_single_base_fixture();
        let node = Uuid::from_u128(0x107);
        let segment = Uuid::from_u128(0x207);
        base_tail
            .ssts
            .push(descriptor(node, SstKind::Nodes, "", 8, 8, 1));
        base_tail.ssts.push(descriptor(
            segment,
            SstKind::VectorGraph,
            "doc_vec",
            8,
            8,
            1,
        ));
        base_tail.search_lsm[0].segments.push(delta(segment, 7, 8));
        base_tail.search_lsm[0].coverage.push(SearchCoverage {
            node_sst_id: node,
            node_sst_max_lsn: 8,
            event_ranges: vec![SearchEventRange::new(7, 8)],
            disposition: CoverageDisposition::Segment,
        });
        base_tail.search_lsm[0].next_event_seq = 8;
        validate_search_lsm(&base_tail).unwrap();

        let policy = delta_policy(2, 2);
        let base_selection = select_search_compactions(&base_tail, policy)
            .unwrap()
            .remove(0);
        assert_eq!(base_selection.mode, SearchCompactionMode::BasePrefix);

        let deltas = two_delta_fixture();
        let delta_selection = select_search_compactions(&deltas, policy)
            .unwrap()
            .remove(0);
        assert_eq!(delta_selection.mode, SearchCompactionMode::DeltaRun);
        assert_eq!(delta_selection.selected_segment_start, 0);
        assert_eq!(delta_selection.selected_descriptors.len(), 2);
    }

    #[test]
    fn shadow_segment_forces_base_repair_and_complete_singleton_stays_idle() {
        let mut manifest = compacted_single_base_fixture();
        manifest.search_lsm[0].segments[0].payload = SearchSegmentPayload::ShadowOnly;
        validate_search_lsm(&manifest).unwrap();

        let policy = delta_policy(2, 8);
        let repair = select_search_compactions(&manifest, policy)
            .unwrap()
            .remove(0);
        assert_eq!(repair.mode, SearchCompactionMode::BasePrefix);
        assert_eq!(repair.selected_segment_start, 0);

        manifest.search_lsm[0].segments[0].payload = SearchSegmentPayload::Complete;
        validate_search_lsm(&manifest).unwrap();
        assert!(select_search_compactions(&manifest, policy)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn delta_bytes_rebuild_once_even_when_base_is_large_or_forced() {
        let mut manifest = fixture();
        let state = &mut manifest.search_lsm[0];
        state.segments[0].role = SearchSegmentRole::Base;
        state.segments[0].format = SearchSegmentFormat::VectorV5Base;
        state.segments[0].stats = SearchSegmentStats::Vector {
            live_count: SearchStatValue::Absolute(1),
        };
        state.base_frontier = Some(1);
        for segment in &mut state.segments[1..] {
            // Repeated updates of the same singleton retain membership.
            segment.stats = SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(0),
            };
        }
        let base_id = state.segments[0].sst_id;
        let base_descriptor = manifest
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.id == base_id)
            .unwrap();
        base_descriptor.level = SstLevel(1);
        base_descriptor.row_count = 1;
        if let KindSpecificStats::VectorGraph { point_count, .. } =
            &mut base_descriptor.kind_specific
        {
            *point_count = 1;
        }
        state.segments[0].content_xxh3 = legacy_base_content_fingerprint(
            base_descriptor,
            SearchSegmentFormat::VectorV5Base,
            &[],
        );
        validate_search_lsm(&manifest).unwrap();

        let policy = SearchCompactionPolicy {
            delta_trigger_segments: 3,
            hard_segment_cap: 6,
            base_compact_bytes: 50,
            base_stale_percent: u64::MAX,
            force_base: false,
            vector_base_delta_rows: u64::MAX,
            vector_base_max_input_bytes: u64::MAX,
        };
        let selection = select_search_compactions(&manifest, policy)
            .unwrap()
            .remove(0);
        assert_eq!(selection.mode, SearchCompactionMode::BasePrefix);

        let id = Uuid::from_u128(0x502);
        let mut output_descriptor = descriptor(id, SstKind::VectorGraph, "doc_vec", 0, 7, 1);
        output_descriptor.level = SstLevel(1);
        // Deliberately larger than the byte threshold: only delta debt, never a
        // healthy base's own size, may schedule another BasePrefix.
        output_descriptor.size_bytes = policy.base_compact_bytes + 1;
        let output_segment = SearchSegmentRef {
            sst_id: id,
            role: SearchSegmentRole::Base,
            format: SearchSegmentFormat::VectorV5Base,
            payload: SearchSegmentPayload::Complete,
            event_ranges: vec![SearchEventRange::new(0, 7)],
            min_lsn: 0,
            max_lsn: 7,
            mutation_count: 1,
            live_payload_count: 1,
            suppress_count: 0,
            content_xxh3: legacy_base_content_fingerprint(
                &output_descriptor,
                SearchSegmentFormat::VectorV5Base,
                &[],
            ),
            complete_filter_properties: Vec::new(),
            stats: SearchSegmentStats::Vector {
                live_count: SearchStatValue::Absolute(1),
            },
            equal_lsn_conflict_count: 0,
        };
        let prepared = PreparedSearchCompaction {
            selection,
            output: Some(PreparedSearchPayload {
                descriptor: output_descriptor.clone(),
                segment: output_segment,
            }),
            empty_proof_digest: None,
            metrics: SearchCompactionBuildMetrics::default(),
        };
        let rebased =
            rebase_prepared_search_compaction(&manifest, &prepared, &manifest.search_lsm[0])
                .unwrap();
        assert_eq!(rebased.segments.len(), 1);
        assert_eq!(rebased.segments[0].live_payload_count, 1);

        let selected = prepared.selection.selected_ids().collect::<HashSet<_>>();
        let mut projected = manifest;
        projected
            .ssts
            .retain(|descriptor| !selected.contains(&descriptor.id));
        projected.ssts.push(output_descriptor);
        projected.search_lsm[0] = rebased;
        validate_search_lsm(&projected).unwrap();
        assert!(select_search_compactions(&projected, policy)
            .unwrap()
            .is_empty());
        assert!(select_search_compactions(
            &projected,
            SearchCompactionPolicy {
                force_base: true,
                ..policy
            }
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn cas_rebase_preserves_append_and_rejects_generation_or_input_drift() {
        let basis = fixture();
        let selection = select_search_compactions(&basis, base_policy(6, 8))
            .unwrap()
            .remove(0);
        let prepared = prepared(selection);

        let mut current = basis.clone();
        let node = Uuid::from_u128(0x107);
        let segment = Uuid::from_u128(0x207);
        current
            .ssts
            .push(descriptor(node, SstKind::Nodes, "", 8, 8, 1));
        current.ssts.push(descriptor(
            segment,
            SstKind::VectorGraph,
            "doc_vec",
            8,
            8,
            1,
        ));
        current.search_lsm[0].segments.push(delta(segment, 7, 8));
        current.search_lsm[0].coverage.push(SearchCoverage {
            node_sst_id: node,
            node_sst_max_lsn: 8,
            event_ranges: vec![SearchEventRange::new(7, 8)],
            disposition: CoverageDisposition::Segment,
        });
        current.search_lsm[0].next_event_seq = 8;

        let rebased =
            rebase_prepared_search_compaction(&current, &prepared, &current.search_lsm[0]).unwrap();
        assert_eq!(rebased.next_event_seq, 8);
        assert_eq!(rebased.base_frontier, Some(7));
        assert_eq!(rebased.segments.len(), 2);
        assert_eq!(
            rebased.segments[0],
            prepared.output.as_ref().unwrap().segment
        );
        assert_eq!(rebased.segments[1].sst_id, segment);
        assert!(rebased.proven_empty_event_ranges.is_empty());
        assert!(matches!(
            rebased.coverage[3].disposition,
            CoverageDisposition::Segment
        ));
        assert_eq!(rebased.coverage.last().unwrap().node_sst_id, node);

        let mut generation_drift = current.clone();
        generation_drift.search_lsm[0].generation_id = Uuid::now_v7();
        assert!(rebase_prepared_search_compaction(
            &generation_drift,
            &prepared,
            &generation_drift.search_lsm[0],
        )
        .is_err());

        let mut input_drift = current;
        let selected = prepared.selection.selected_descriptors[0].id;
        input_drift
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.id == selected)
            .unwrap()
            .size_bytes += 1;
        assert!(rebase_prepared_search_compaction(
            &input_drift,
            &prepared,
            &input_drift.search_lsm[0],
        )
        .is_err());
    }

    #[test]
    fn delta_rebase_replaces_only_tail_and_preserves_concurrent_append() {
        let basis = fixture();
        let selection = select_search_compactions(&basis, delta_policy(3, 8))
            .unwrap()
            .remove(0);
        assert_eq!(selection.mode, SearchCompactionMode::DeltaRun);
        assert_eq!(selection.selected_segment_start, 3);
        assert_eq!(
            selection.selected_event_ranges,
            vec![SearchEventRange::new(4, 7)]
        );
        let prepared = prepared_delta(selection);

        let mut current = basis;
        let node = Uuid::from_u128(0x107);
        let segment = Uuid::from_u128(0x207);
        current
            .ssts
            .push(descriptor(node, SstKind::Nodes, "", 8, 8, 1));
        current.ssts.push(descriptor(
            segment,
            SstKind::VectorGraph,
            "doc_vec",
            8,
            8,
            1,
        ));
        current.search_lsm[0].segments.push(delta(segment, 7, 8));
        current.search_lsm[0].coverage.push(SearchCoverage {
            node_sst_id: node,
            node_sst_max_lsn: 8,
            event_ranges: vec![SearchEventRange::new(7, 8)],
            disposition: CoverageDisposition::Segment,
        });
        current.search_lsm[0].next_event_seq = 8;

        let rebased =
            rebase_prepared_search_compaction(&current, &prepared, &current.search_lsm[0]).unwrap();
        assert_eq!(
            &rebased.segments[..3],
            &prepared.selection.captured_state.segments[..3]
        );
        assert_eq!(
            rebased.segments[3],
            prepared.output.as_ref().unwrap().segment
        );
        assert_eq!(rebased.segments[4].sst_id, segment);
        assert_eq!(rebased.next_event_seq, 8);
        assert_eq!(rebased.base_frontier, None);
        assert_eq!(
            rebased.proven_empty_event_ranges,
            vec![SearchEventRange::new(3, 4)]
        );
    }

    #[test]
    fn empty_delta_requires_zero_stats_and_rewrites_only_owned_coverage() {
        let mut basis = fixture();
        // The two selected physical deltas have an exact net-zero signed
        // contribution. This exercises the metadata proof path independently
        // of the physical builder's ability to elide every reconciled winner.
        basis.search_lsm[0].segments[4].stats = SearchSegmentStats::Vector {
            live_count: SearchStatValue::Delta(-1),
        };
        validate_search_lsm(&basis).unwrap();
        let selection = select_search_compactions(&basis, delta_policy(2, 8))
            .unwrap()
            .remove(0);
        assert_eq!(
            summed_delta_stats(&selection).unwrap(),
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(0)
            }
        );
        let prepared = PreparedSearchCompaction {
            selection,
            output: None,
            empty_proof_digest: Some(0x000d_311a),
            metrics: SearchCompactionBuildMetrics::default(),
        };
        let rebased =
            rebase_prepared_search_compaction(&basis, &prepared, &basis.search_lsm[0]).unwrap();
        assert_eq!(rebased.segments.len(), 4);
        assert_eq!(
            rebased.proven_empty_event_ranges,
            vec![SearchEventRange::new(3, 4), SearchEventRange::new(5, 7)]
        );
        for coverage in &rebased.coverage[..5] {
            if coverage.event_ranges == [SearchEventRange::new(3, 4)] {
                assert!(matches!(
                    coverage.disposition,
                    CoverageDisposition::ProvenEmpty {
                        classifier_version: 1,
                        ..
                    }
                ));
            } else {
                assert_eq!(coverage.disposition, CoverageDisposition::Segment);
            }
        }
        for coverage in &rebased.coverage[5..] {
            assert_eq!(
                coverage.disposition,
                CoverageDisposition::ProvenEmpty {
                    classifier_version: EMPTY_DELTA_CLASSIFIER_VERSION,
                    before_after_digest: 0x000d_311a,
                }
            );
        }

        let mut non_zero = prepared;
        non_zero.selection.captured_state.segments[4].stats = SearchSegmentStats::Vector {
            live_count: SearchStatValue::Delta(1),
        };
        assert!(verify_output(&non_zero).is_err());
    }

    #[test]
    fn empty_base_rebase_keeps_append_tail_and_installs_a_proof() {
        let basis = fixture();
        let selection = select_search_compactions(&basis, base_policy(6, 8))
            .unwrap()
            .remove(0);
        let empty_digest = 0xfeed_beef;
        let prepared = PreparedSearchCompaction {
            selection,
            output: None,
            empty_proof_digest: Some(empty_digest),
            metrics: SearchCompactionBuildMetrics::default(),
        };

        let mut current = basis;
        let node = Uuid::from_u128(0x107);
        current
            .ssts
            .push(descriptor(node, SstKind::Nodes, "", 8, 8, 1));
        current.search_lsm[0]
            .proven_empty_event_ranges
            .push(SearchEventRange::new(7, 8));
        current.search_lsm[0].coverage.push(SearchCoverage {
            node_sst_id: node,
            node_sst_max_lsn: 8,
            event_ranges: vec![SearchEventRange::new(7, 8)],
            disposition: CoverageDisposition::ProvenEmpty {
                classifier_version: 1,
                before_after_digest: 77,
            },
        });
        current.search_lsm[0].next_event_seq = 8;

        let rebased =
            rebase_prepared_search_compaction(&current, &prepared, &current.search_lsm[0]).unwrap();
        assert!(rebased.segments.is_empty());
        assert_eq!(rebased.base_frontier, None);
        assert_eq!(
            rebased.proven_empty_event_ranges,
            vec![SearchEventRange::new(0, 7), SearchEventRange::new(7, 8)]
        );
        for coverage in &rebased.coverage[..7] {
            assert_eq!(
                coverage.disposition,
                CoverageDisposition::ProvenEmpty {
                    classifier_version: EMPTY_BASE_CLASSIFIER_VERSION,
                    before_after_digest: empty_digest,
                }
            );
        }
        assert_eq!(
            rebased.coverage[7].disposition,
            CoverageDisposition::ProvenEmpty {
                classifier_version: 1,
                before_after_digest: 77,
            }
        );
    }

    #[test]
    fn cas_rebase_accepts_concurrent_append_above_configured_cap() {
        let basis = fixture();
        let selection = select_search_compactions(&basis, base_policy(6, 8))
            .unwrap()
            .remove(0);
        let prepared = prepared(selection);
        let mut current = basis;

        for event in 7..15u64 {
            let node = Uuid::from_u128(0x100 + u128::from(event));
            let segment = Uuid::from_u128(0x200 + u128::from(event));
            current.ssts.push(descriptor(
                node,
                SstKind::Nodes,
                "",
                event + 1,
                event + 1,
                1,
            ));
            current.ssts.push(descriptor(
                segment,
                SstKind::VectorGraph,
                "doc_vec",
                event + 1,
                event + 1,
                1,
            ));
            current.search_lsm[0]
                .segments
                .push(delta(segment, event, event + 1));
            current.search_lsm[0].coverage.push(SearchCoverage {
                node_sst_id: node,
                node_sst_max_lsn: event + 1,
                event_ranges: vec![SearchEventRange::new(event, event + 1)],
                disposition: CoverageDisposition::Segment,
            });
        }
        current.search_lsm[0].next_event_seq = 15;
        validate_search_lsm(&current).unwrap();

        // Off-lock flushes may take a generation past its configured cap
        // while the compaction ran; the compaction still lowers the count, so
        // it installs. Only the protocol ceiling refuses it.
        let rebased =
            rebase_prepared_search_compaction(&current, &prepared, &current.search_lsm[0]).unwrap();
        assert!(rebased.segments.len() > 8);
        assert!(rebased.segments.len() < current.search_lsm[0].segments.len());
        assert!(rebased.segments.len() <= MAX_ACTIVE_SEARCH_SEGMENTS);
    }

    #[test]
    fn delta_rebase_accepts_concurrent_append_above_configured_cap() {
        // Cap 5 keeps the DeltaRun route; two concurrent appends push the
        // rebased generation past it.
        let basis = fixture();
        let selection = select_search_compactions(&basis, delta_policy(3, 5))
            .unwrap()
            .remove(0);
        let prepared = prepared_delta(selection);
        let mut current = basis;

        for event in 7..9u64 {
            let node = Uuid::from_u128(0x100 + u128::from(event));
            let segment = Uuid::from_u128(0x200 + u128::from(event));
            current.ssts.push(descriptor(
                node,
                SstKind::Nodes,
                "",
                event + 1,
                event + 1,
                1,
            ));
            current.ssts.push(descriptor(
                segment,
                SstKind::VectorGraph,
                "doc_vec",
                event + 1,
                event + 1,
                1,
            ));
            current.search_lsm[0]
                .segments
                .push(delta(segment, event, event + 1));
            current.search_lsm[0].coverage.push(SearchCoverage {
                node_sst_id: node,
                node_sst_max_lsn: event + 1,
                event_ranges: vec![SearchEventRange::new(event, event + 1)],
                disposition: CoverageDisposition::Segment,
            });
        }
        current.search_lsm[0].next_event_seq = 9;
        validate_search_lsm(&current).unwrap();

        // See `cas_rebase_accepts_concurrent_append_above_configured_cap`.
        let rebased =
            rebase_prepared_search_compaction(&current, &prepared, &current.search_lsm[0]).unwrap();
        assert!(rebased.segments.len() > 5);
        assert!(rebased.segments.len() < current.search_lsm[0].segments.len());
    }
}

#[cfg(all(test, feature = "vector-index"))]
#[path = "search_lsm_compact_vector_merge_tests.rs"]
mod vector_merge_tests;
