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
const DEFAULT_SEARCH_LSM_MAX_SEGMENTS: usize = 8;
const DEFAULT_SEARCH_LSM_BASE_COMPACT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_SEARCH_LSM_BASE_STALE_PERCENT: u64 = 800;
const MIN_SEARCH_LSM_SEGMENTS: usize = 2;
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
        Ok(Self {
            delta_trigger_segments,
            hard_segment_cap,
            base_compact_bytes,
            base_stale_percent,
            force_base,
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
                }
            }
        }
    }
}

fn default_delta_trigger(hard_segment_cap: usize) -> usize {
    hard_segment_cap
        .saturating_sub(2)
        .max(MIN_SEARCH_LSM_SEGMENTS)
        .min(hard_segment_cap)
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
    pub(super) fn key(&self) -> (SearchLsmKind, &str) {
        (
            self.captured_state.kind,
            self.captured_state.index_name.as_str(),
        )
    }

    pub(super) fn frontier(&self) -> u64 {
        self.captured_state.next_event_seq
    }

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

    let mut selections = Vec::new();
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
        let delta_count = state
            .segments
            .iter()
            .filter(|segment| segment.role == SearchSegmentRole::Delta)
            .count();
        let delta_run_output_len = state
            .segments
            .len()
            .checked_sub(policy.delta_trigger_segments)
            .and_then(|retained| retained.checked_add(1));
        // Lowering the configured cap below an existing generation, or
        // reaching cap with Base + too short a delta tail, must still have a
        // route that frees a foreground flush slot.
        let cap_requires_base = state.segments.len() >= policy.hard_segment_cap
            && (delta_count < policy.delta_trigger_segments
                || delta_run_output_len
                    .map(|len| len >= policy.hard_segment_cap)
                    .unwrap_or(true));
        let rebuild_base = base_rebuild_needed(state, &descriptors, policy)? || cap_requires_base;
        let (mode, selected_segment_start) = if rebuild_base {
            (SearchCompactionMode::BasePrefix, 0)
        } else if delta_count >= policy.delta_trigger_segments {
            let start = state
                .segments
                .len()
                .checked_sub(policy.delta_trigger_segments)
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
    if selection.selected_descriptors.is_empty()
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
    super::verify_search_state_append_only(captured, current)?;
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
            let selected_start = selection.selected_segment_start;
            let captured_end = captured.segments.len();
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
    if rebased.segments.len() > prepared.selection.hard_segment_cap
        || rebased.segments.len() > MAX_ACTIVE_SEARCH_SEGMENTS
    {
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

struct SearchNodeInput {
    body: Bytes,
    min_key: [u8; 16],
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
    if selections.is_empty() {
        return Ok(Vec::new());
    }
    let manifest = &base.manifest;
    let (base_selections, delta_selections): (Vec<_>, Vec<_>) = selections
        .into_iter()
        .partition(|selection| selection.mode == SearchCompactionMode::BasePrefix);
    let mut built = Vec::with_capacity(base_selections.len() + delta_selections.len());

    if !base_selections.is_empty() {
        let mut inputs = Vec::new();
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
            inputs.push(SearchNodeInput {
                body: super::get_sst_body(store.as_ref(), paths, descriptor).await?,
                min_key: descriptor.min_key,
                label_def,
                scope: descriptor.scope.clone(),
            });
        }
        if inputs.is_empty() {
            return Err(Error::invariant(
                "active Search-LSM BasePrefix compaction has no Nodes SST inputs",
            ));
        }

        let build_manifest = manifest.clone();
        let started = Instant::now();
        built.extend(
            super::run_cpu(move || {
                build_search_compactions_blocking(
                    &build_manifest,
                    base_selections,
                    inputs,
                    input_bytes,
                    started,
                )
            })
            .await??,
        );
    }

    for selection in delta_selections {
        built.push(build_delta_run_compaction(store.clone(), paths, base, selection).await?);
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
    let vector = match node.properties.get(&descriptor.property)? {
        Value::Vec(vector) => vector.clone(),
        Value::VecI8 { codes, scale } => {
            codes.iter().map(|code| f32::from(*code) * *scale).collect()
        }
        _ => return None,
    };
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
        let decoded;
        let vector: &[f32] = match value {
            Value::Vec(vector) => vector,
            Value::VecI8 { codes, scale } => {
                decoded = codes
                    .iter()
                    .map(|code| f32::from(*code) * *scale)
                    .collect::<Vec<_>>();
                &decoded
            }
            _ => {
                update_winner_digest(&mut self.digest, node_id, lsn, true, None);
                return Ok(());
            }
        };
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
        let elapsed = elapsed_millis(started);
        let Some(artifact) = artifact else {
            if input_rows != 0 {
                tracing::warn!(
                    index = %descriptor.name,
                    kind = "vector",
                    live_rows = input_rows,
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
                    input_bytes,
                    output_bytes: len,
                    input_rows,
                    touched_rows: input_rows,
                    live_rows: input_rows,
                    peak_workspace_bytes: metrics.peak_logical_memory_bytes as u64,
                    scratch_bytes: metrics.scratch_bytes_written,
                    spill_bytes: metrics.scratch_bytes_written,
                    prepare_millis: elapsed,
                },
            },
            file: Some(file),
        }))
    }
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
    inputs: Vec<SearchNodeInput>,
    input_bytes: u64,
    started: Instant,
) -> Result<Vec<BuiltSearchCompaction>> {
    let per_builder_memory = super::per_collector_index_build_memory_bytes(selections.len())?
        .ok_or_else(|| Error::invariant("Search-LSM compaction has no build memory share"))?;
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

    let mut cursors = Vec::with_capacity(inputs.len());
    let mut unopened = BinaryHeap::with_capacity(inputs.len());
    for (source, input) in inputs.into_iter().enumerate() {
        let cursor = super::NodeSourceCursor::open(&input.label_def, input.body)?;
        cursors.push(SearchNodeCursor {
            cursor,
            label_def: input.label_def,
            scope: input.scope,
        });
        unopened.push(Reverse((input.min_key, source)));
    }
    let mut heap: BinaryHeap<Reverse<super::NodeHeapEntry>> =
        BinaryHeap::with_capacity(cursors.len());
    let mut last_id = None;
    loop {
        while let Some(Reverse((hinted_min, source))) = unopened.peek().copied() {
            let active_min = heap.peek().map(|entry| entry.0.id);
            if active_min.is_some_and(|active| hinted_min > active) {
                break;
            }
            unopened.pop();
            let source_cursor = &mut cursors[source].cursor;
            source_cursor.ensure_positioned()?;
            let Some((id, lsn)) = source_cursor.peek() else {
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
        }
        let Some(Reverse(entry)) = heap.pop() else {
            break;
        };
        let source = &mut cursors[entry.src];
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
        if let Some((id, lsn)) = source.cursor.peek() {
            heap.push(Reverse(super::NodeHeapEntry {
                id,
                lsn,
                src: entry.src,
            }));
        }
    }

    works
        .into_iter()
        .filter_map(|work| work.finish(input_bytes, started).transpose())
        .collect()
}

#[cfg(not(any(feature = "vector-index", feature = "text-index")))]
fn build_search_compactions_blocking(
    _manifest: &Manifest,
    selections: Vec<SearchCompactionSelection>,
    _inputs: Vec<SearchNodeInput>,
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
                    entry_medoid: 0,
                },
                _ => unreachable!(),
            },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
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

    fn delta_policy(trigger: usize, hard_cap: usize) -> SearchCompactionPolicy {
        SearchCompactionPolicy {
            delta_trigger_segments: trigger,
            hard_segment_cap: hard_cap,
            base_compact_bytes: u64::MAX,
            base_stale_percent: u64::MAX,
            force_base: false,
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
    fn cas_rebase_rejects_concurrent_append_above_hard_cap() {
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

        let error = rebase_prepared_search_compaction(&current, &prepared, &current.search_lsm[0])
            .unwrap_err();
        assert!(
            error.to_string().contains("above hard cap"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn delta_rebase_rejects_concurrent_append_above_hard_cap() {
        // A cap of 4 never reaches this path: planning escalates the fixture's
        // six segments to BasePrefix because a DeltaRun would still retain four
        // and free no foreground flush slot. Cap 5 keeps the DeltaRun route —
        // it rebases to three retained plus one output — so two concurrent
        // appends are what push the rebased generation to six.
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

        let error = rebase_prepared_search_compaction(&current, &prepared, &current.search_lsm[0])
            .unwrap_err();
        assert!(
            error.to_string().contains("above hard cap"),
            "unexpected error: {error}"
        );
    }
}
