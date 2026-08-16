//! Manifest model and pure validation for incremental search indexes.
//!
//! Search payloads remain ordinary [`SstDescriptor`](crate::manifest::SstDescriptor)
//! objects. This module adds only the manifest-side lineage needed to prove
//! that a base+delta set covers every visible `Nodes` SST. It performs no
//! object-store reads and deliberately does not activate the read path: a
//! caller may use a state only after [`validate_search_lsm`] succeeds and the
//! segment footers pass their format-specific checks.
//!
//! The numbered validation invariants match
//! `docs/architecture/search-lsm.md`. Invariant 9 is a wire/footer invariant
//! and is intentionally outside this pure manifest validator: the read path
//! validates the migration barrier plus V5/V3 native footer today, while
//! future VG6/FT4 readers will validate lineage in every data footer.

use std::collections::{HashMap, HashSet};
use std::fmt;

use bincode::Options;
use bytes::Bytes;
use chrono::Utc;
use namidb_core::DataType;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::manifest::{
    KindSpecificStats, Manifest, SstDescriptor, SstKind, SstLevel, TextIndexDescriptor,
    VectorIndexDescriptor, VectorMetric, VectorQuantization,
};

const CATALOG_SIGNATURE_VERSION: u16 = 1;
const TEXT_ANALYZER_VERSION: u16 = 1;
const VECTOR_FILTER_ENCODING_VERSION: u16 = 1;
const BARRIER_FORMAT_VERSION: u16 = 2;
const LEGACY_BARRIER_FORMAT_VERSION: u16 = 1;
const LEGACY_BASE_BINDING_VERSION: u16 = 1;
const BARRIER_TRAILER_LEN: usize = 8 + 8 + 4;
const MAX_BARRIER_FOOTER_BYTES: usize = 1024 * 1024;
const MAX_BARRIER_BODY_BYTES: usize =
    SEARCH_LSM_BARRIER_MAGIC.len() + MAX_BARRIER_FOOTER_BYTES + BARRIER_TRAILER_LEN;

/// Deliberately unsupported by every pre-Search-LSM reader.
pub const SEARCH_LSM_BARRIER_MAGIC: &[u8; 8] = b"NAMISLB1";
const SEARCH_LSM_BARRIER_TRAILER_MAGIC: &[u8; 8] = b"SLBEND01";

/// Search-index family governed by one LSM state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchLsmKind {
    #[default]
    Vector,
    Text,
}

impl SearchLsmKind {
    /// Physical SST kind used by this search family.
    pub const fn sst_kind(self) -> SstKind {
        match self {
            Self::Vector => SstKind::VectorGraph,
            Self::Text => SstKind::TextIndex,
        }
    }
}

/// Lifecycle state. `Building` is never authoritative for queries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchLsmStatus {
    #[default]
    Building,
    Active,
}

/// Logical role of one physical search segment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchSegmentRole {
    Base,
    #[default]
    Delta,
}

/// Whether the searchable payload was built completely.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchSegmentPayload {
    #[default]
    Complete,
    /// The version/suppress table is valid, but queries must use their exact
    /// fallback until maintenance replaces this segment.
    ShadowOnly,
}

/// On-object format. `Unknown` is safe for serde evolution but invalid in an
/// active generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchSegmentFormat {
    #[default]
    Unknown,
    VectorV5Base,
    VectorV6,
    TextV3Base,
    TextV4,
}

/// Half-open logical event range `[start, end)`.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(default)]
pub struct SearchEventRange {
    pub start: u64,
    pub end: u64,
}

impl SearchEventRange {
    pub const fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    pub const fn is_valid(self) -> bool {
        self.start < self.end
    }

    fn contains(self, other: Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }
}

/// A scalar statistic is absolute in a base and signed in a delta.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStatValue {
    Absolute(u64),
    Delta(i64),
}

impl Default for SearchStatValue {
    fn default() -> Self {
        Self::Absolute(0)
    }
}

/// Manifest-resident statistics needed to validate base/delta arithmetic.
///
/// Full-text per-term signed document frequencies live in the range-readable
/// object. `term_df_violation_count` is the builder's manifest summary; the
/// FT4 footer validator must independently confirm it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SearchSegmentStats {
    #[default]
    Unknown,
    Vector {
        #[serde(default)]
        live_count: SearchStatValue,
    },
    Text {
        #[serde(default)]
        doc_count: SearchStatValue,
        #[serde(default)]
        total_len: SearchStatValue,
        #[serde(default)]
        term_df_violation_count: u64,
    },
}

/// One physical base/delta object referenced by a search generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchSegmentRef {
    pub sst_id: Uuid,
    pub role: SearchSegmentRole,
    pub format: SearchSegmentFormat,
    pub payload: SearchSegmentPayload,
    pub event_ranges: Vec<SearchEventRange>,
    pub min_lsn: u64,
    pub max_lsn: u64,
    pub mutation_count: u64,
    pub live_payload_count: u64,
    pub suppress_count: u64,
    /// V6/FT4 content digest, or for a wrapped V5/V3 base the deterministic
    /// descriptor/native-footer binding returned by
    /// [`legacy_base_content_fingerprint`].
    pub content_xxh3: u64,
    /// Sorted complete native-filter properties. A partially built property
    /// must be omitted atomically by the future segment builder.
    pub complete_filter_properties: Vec<String>,
    pub stats: SearchSegmentStats,
    /// Builder/compactor summary for invariant 10. Footer validation will
    /// independently bind it to the version table.
    pub equal_lsn_conflict_count: u64,
}

/// Why a current `Nodes` SST introduced no standalone search payload.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CoverageDisposition {
    #[default]
    Unknown,
    /// The coverage ranges are owned by one or more physical segments.
    Segment,
    /// Exact before/after classification proved no logical search change.
    ProvenEmpty {
        classifier_version: u16,
        before_after_digest: u64,
    },
    /// Node compaction rewrote already-covered rows without a new mutation.
    LogicalRewrite { input_coverage_digest: u64 },
}

/// Search coverage attached to one currently visible `Nodes` SST.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchCoverage {
    pub node_sst_id: Uuid,
    pub node_sst_max_lsn: u64,
    pub event_ranges: Vec<SearchEventRange>,
    pub disposition: CoverageDisposition,
}

/// Manifest state for one catalog index and one immutable LSM generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchLsmState {
    pub index_name: String,
    pub kind: SearchLsmKind,
    pub catalog_signature: String,
    pub generation_id: Uuid,
    pub status: SearchLsmStatus,
    /// First unallocated logical event. Events begin at zero.
    pub next_event_seq: u64,
    /// Exclusive event frontier represented by the absolute base.
    pub base_frontier: Option<u64>,
    pub segments: Vec<SearchSegmentRef>,
    /// Events proven to have no logical search effect. Together with segment
    /// ranges these must partition `0..next_event_seq`.
    pub proven_empty_event_ranges: Vec<SearchEventRange>,
    /// Exactly one entry per currently visible `Nodes` SST in an active state.
    pub coverage: Vec<SearchCoverage>,
    /// Same-kind/scope object that makes 2.0.6 readers choose exact fallback.
    pub compat_barrier_sst_id: Option<Uuid>,
    /// Cross-segment equal-LSN conflicts observed by the builder/compactor.
    pub equal_lsn_conflict_count: u64,
}

/// Checksummed payload of the downgrade barrier.
///
/// Legacy V5/V3 bases cannot repeat Search-LSM lineage in their native footer
/// without an in-place format change. During migration the barrier is the
/// generation footer: it binds the complete manifest state while the native
/// base reader independently validates its magic, footer checksum and
/// descriptor statistics. V6/FT4 will additionally repeat this lineage in
/// every data footer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LegacySearchBarrierFooter {
    format_version: u16,
    state: SearchLsmState,
}

/// V2 keeps the manifest model behind its JSON evolution boundary. Bincode is
/// used only for this fixed primitive envelope; internally tagged manifest
/// enums require `deserialize_any`, which bincode deliberately cannot decode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SearchBarrierFooterV2 {
    format_version: u16,
    state_json: Vec<u8>,
}

/// Why generation selection conservatively chose the exact node scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchReadFallback {
    NoPhysicalBody,
    AmbiguousPhysicalBodies,
    BuildingGeneration,
    InvalidGeneration,
    UnsupportedSegmentSet,
    SegmentLimitExceeded,
}

/// Manifest-only search generation selection.
///
/// Object validation is deliberately a separate read-path step. An
/// `ActiveLegacyBase` is usable only after its barrier and native V5/V3 footer
/// validate; a missing or corrupt optional object remains an exact fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchReadPlan {
    Legacy {
        sst_id: Uuid,
    },
    ActiveLegacyBase {
        state: SearchLsmState,
        base_sst_id: Uuid,
        barrier_sst_id: Uuid,
    },
    /// Native base/delta generation. Object readers must still validate the
    /// barrier and every VG6/FT4 footer before serving any partial result.
    ActiveSegments {
        state: SearchLsmState,
        barrier_sst_id: Uuid,
    },
    FlatFallback(SearchReadFallback),
}

/// Absolute protocol ceiling. The operational writer cap may be lower, but a
/// manifest above this fan-out is never queried: resident sparse directories,
/// winner probes and object-store request amplification must remain bounded
/// even when a malformed/imported manifest bypasses normal maintenance.
pub const MAX_ACTIVE_SEARCH_SEGMENTS: usize = 32;

/// Barrier wire validation error. It never makes node data unavailable; read
/// callers map it to the exact search fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchBarrierError {
    pub detail: String,
}

impl SearchBarrierError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for SearchBarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "search LSM compatibility barrier: {}", self.detail)
    }
}

impl std::error::Error for SearchBarrierError {}

/// Numbered, manifest-verifiable Search-LSM invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SearchLsmInvariant {
    Catalog = 1,
    UniqueState = 2,
    CompatibilityBarrier = 3,
    PhysicalSegments = 4,
    SegmentOrdering = 5,
    NodeCoverage = 6,
    EventCoverage = 7,
    Statistics = 8,
    VersionTie = 10,
}

/// Deterministic validation failure. The detail is diagnostic only; callers
/// should branch on [`Self::invariant`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchLsmValidationError {
    pub invariant: SearchLsmInvariant,
    pub index_name: String,
    pub detail: String,
}

impl SearchLsmValidationError {
    fn new(
        invariant: SearchLsmInvariant,
        index_name: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            invariant,
            index_name: index_name.into(),
            detail: detail.into(),
        }
    }
}

impl fmt::Display for SearchLsmValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "search LSM invariant {} failed for '{}': {}",
            self.invariant as u8, self.index_name, self.detail
        )
    }
}

impl std::error::Error for SearchLsmValidationError {}

fn frame_search_barrier_footer(encoded: &[u8]) -> Result<Bytes, SearchBarrierError> {
    if encoded.is_empty() || encoded.len() > MAX_BARRIER_FOOTER_BYTES {
        return Err(SearchBarrierError::new(format!(
            "footer length {} is outside 1..={MAX_BARRIER_FOOTER_BYTES}",
            encoded.len()
        )));
    }
    let mut body =
        Vec::with_capacity(SEARCH_LSM_BARRIER_MAGIC.len() + encoded.len() + BARRIER_TRAILER_LEN);
    body.extend_from_slice(SEARCH_LSM_BARRIER_MAGIC);
    body.extend_from_slice(encoded);
    body.extend_from_slice(SEARCH_LSM_BARRIER_TRAILER_MAGIC);
    body.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    body.extend_from_slice(&crc32fast::hash(encoded).to_le_bytes());
    Ok(Bytes::from(body))
}

/// Encode the tiny downgrade barrier.
///
/// V2 keeps the manifest-shaped state in JSON and uses bincode only for a
/// fixed primitive envelope. This is intentional: internally tagged serde
/// enums cannot be decoded by bincode's non-self-describing data model.
pub fn encode_search_barrier(state: &SearchLsmState) -> Result<Bytes, SearchBarrierError> {
    let state_json = serde_json::to_vec(state)
        .map_err(|error| SearchBarrierError::new(format!("state JSON encode failed: {error}")))?;
    let footer = SearchBarrierFooterV2 {
        format_version: BARRIER_FORMAT_VERSION,
        state_json,
    };
    let encoded = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(&footer)
        .map_err(|error| SearchBarrierError::new(format!("footer encode failed: {error}")))?;
    frame_search_barrier_footer(&encoded)
}

/// Decode and checksum a downgrade barrier without trusting allocation sizes.
pub fn decode_search_barrier(body: &[u8]) -> Result<SearchLsmState, SearchBarrierError> {
    let minimum = SEARCH_LSM_BARRIER_MAGIC.len() + 1 + BARRIER_TRAILER_LEN;
    if body.len() < minimum {
        return Err(SearchBarrierError::new("body is too short"));
    }
    if &body[..SEARCH_LSM_BARRIER_MAGIC.len()] != SEARCH_LSM_BARRIER_MAGIC {
        return Err(SearchBarrierError::new("magic mismatch"));
    }
    let trailer_start = body.len() - BARRIER_TRAILER_LEN;
    let trailer = &body[trailer_start..];
    if &trailer[..SEARCH_LSM_BARRIER_TRAILER_MAGIC.len()] != SEARCH_LSM_BARRIER_TRAILER_MAGIC {
        return Err(SearchBarrierError::new("trailer magic mismatch"));
    }
    let length_offset = SEARCH_LSM_BARRIER_TRAILER_MAGIC.len();
    let footer_len = u64::from_le_bytes(
        trailer[length_offset..length_offset + 8]
            .try_into()
            .expect("fixed barrier trailer"),
    );
    let footer_len = usize::try_from(footer_len)
        .map_err(|_| SearchBarrierError::new("footer length does not fit usize"))?;
    if footer_len == 0 || footer_len > MAX_BARRIER_FOOTER_BYTES {
        return Err(SearchBarrierError::new(format!(
            "footer length {footer_len} is invalid"
        )));
    }
    let footer_start = trailer_start
        .checked_sub(footer_len)
        .ok_or_else(|| SearchBarrierError::new("footer starts before the body"))?;
    if footer_start != SEARCH_LSM_BARRIER_MAGIC.len() {
        return Err(SearchBarrierError::new(
            "body contains unaccounted bytes outside its footer",
        ));
    }
    let footer_bytes = &body[footer_start..trailer_start];
    let crc_offset = length_offset + 8;
    let expected_crc = u32::from_le_bytes(
        trailer[crc_offset..crc_offset + 4]
            .try_into()
            .expect("fixed barrier trailer"),
    );
    if crc32fast::hash(footer_bytes) != expected_crc {
        return Err(SearchBarrierError::new("footer checksum mismatch"));
    }
    if footer_bytes.len() < std::mem::size_of::<u16>() {
        return Err(SearchBarrierError::new(
            "footer is too short to contain a version",
        ));
    }
    let wire_version = u16::from_le_bytes(
        footer_bytes[..2]
            .try_into()
            .expect("checked barrier version bytes"),
    );
    match wire_version {
        BARRIER_FORMAT_VERSION => {
            let footer: SearchBarrierFooterV2 = bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .with_limit(footer_bytes.len() as u64)
                .reject_trailing_bytes()
                .deserialize(footer_bytes)
                .map_err(|error| {
                    SearchBarrierError::new(format!("V2 footer decode failed: {error}"))
                })?;
            if footer.format_version != BARRIER_FORMAT_VERSION {
                return Err(SearchBarrierError::new(format!(
                    "footer version {} is unsupported",
                    footer.format_version
                )));
            }
            serde_json::from_slice(&footer.state_json).map_err(|error| {
                SearchBarrierError::new(format!("V2 state JSON decode failed: {error}"))
            })
        }
        LEGACY_BARRIER_FORMAT_VERSION => {
            let footer: LegacySearchBarrierFooter = bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .with_limit(footer_bytes.len() as u64)
                .reject_trailing_bytes()
                .deserialize(footer_bytes)
                .map_err(|error| {
                    SearchBarrierError::new(format!("legacy footer decode failed: {error}"))
                })?;
            Ok(footer.state)
        }
        other => Err(SearchBarrierError::new(format!(
            "footer version {other} is unsupported"
        ))),
    }
}

/// Validate that the physical barrier repeats the selected generation.
pub fn validate_search_barrier(
    expected: &SearchLsmState,
    body: &[u8],
) -> Result<(), SearchBarrierError> {
    let actual = decode_search_barrier(body)?;
    if &actual != expected {
        return Err(SearchBarrierError::new(
            "footer state does not match the manifest generation",
        ));
    }
    Ok(())
}

/// Construct the ordinary SST descriptor that keeps the barrier in every
/// existing manifest/backup/pin/orphan live-set traversal.
pub fn search_barrier_descriptor(
    state: &SearchLsmState,
    id: Uuid,
    level: SstLevel,
    path: String,
    size_bytes: u64,
) -> SstDescriptor {
    let kind_specific = match state.kind {
        SearchLsmKind::Vector => KindSpecificStats::VectorGraph {
            dim: 0,
            metric: "compat-barrier".into(),
            point_count: 0,
            r: 0,
            l_build: 0,
            alpha: 0.0,
            entry_medoid: 0,
        },
        SearchLsmKind::Text => KindSpecificStats::TextIndex {
            doc_count: 0,
            term_count: 0,
            total_len: 0,
        },
    };
    SstDescriptor {
        id,
        kind: state.kind.sst_kind(),
        scope: state.index_name.clone(),
        level,
        path,
        size_bytes,
        row_count: 0,
        created_at: Utc::now(),
        min_key: [0; 16],
        max_key: [0; 16],
        min_lsn: 0,
        max_lsn: 0,
        schema_version_min: 0,
        schema_version_max: 0,
        property_stats: Vec::new(),
        kind_specific,
        bloom: None,
        unique_property_indices: Vec::new(),
        equality_property_indices: Vec::new(),
        label_index: None,
        node_locator: None,
        per_label_property_stats: Vec::new(),
    }
}

/// Whether `descriptor` is the zero-valued, same-kind SST envelope reserved
/// for a Search-LSM compatibility barrier.
///
/// This predicate intentionally does not require a `SearchLsmState`: a newer
/// compactor can use it to recognize and retire the barrier left behind when
/// an older writer reserialized the manifest and dropped the unknown
/// `search_lsm` field. The checks are strict enough that an empty legitimate
/// V5/V3 data body is never classified as a barrier.
pub(crate) fn is_canonical_search_barrier_descriptor(descriptor: &SstDescriptor) -> bool {
    let size_is_bounded = descriptor.size_bytes
        >= (SEARCH_LSM_BARRIER_MAGIC.len() + BARRIER_TRAILER_LEN + 1) as u64
        && descriptor.size_bytes <= MAX_BARRIER_BODY_BYTES as u64;
    let generic_zero = !descriptor.id.is_nil()
        && descriptor.path.ends_with(".slb")
        && descriptor.row_count == 0
        && descriptor.min_key == [0; 16]
        && descriptor.max_key == [0; 16]
        && descriptor.min_lsn == 0
        && descriptor.max_lsn == 0
        && descriptor.schema_version_min == 0
        && descriptor.schema_version_max == 0
        && descriptor.property_stats.is_empty()
        && descriptor.bloom.is_none()
        && descriptor.unique_property_indices.is_empty()
        && descriptor.equality_property_indices.is_empty()
        && descriptor.label_index.is_none()
        && descriptor.node_locator.is_none()
        && descriptor.per_label_property_stats.is_empty();
    let kind_zero = match (descriptor.kind, &descriptor.kind_specific) {
        (
            SstKind::VectorGraph,
            KindSpecificStats::VectorGraph {
                dim,
                metric,
                point_count,
                r,
                l_build,
                alpha,
                entry_medoid,
            },
        ) => {
            *dim == 0
                && metric == "compat-barrier"
                && *point_count == 0
                && *r == 0
                && *l_build == 0
                && alpha.to_bits() == 0.0f32.to_bits()
                && *entry_medoid == 0
        }
        (
            SstKind::TextIndex,
            KindSpecificStats::TextIndex {
                doc_count,
                term_count,
                total_len,
            },
        ) => *doc_count == 0 && *term_count == 0 && *total_len == 0,
        _ => false,
    };
    size_is_bounded && generic_zero && kind_zero
}

/// Canonical complete native-filter property list for a vector descriptor.
///
/// Unique identity keys are deliberately excluded even when their backing
/// constraint has index semantics: advertising every key as a complete native
/// filter would turn the catalog signature and every vector segment into a
/// high-cardinality filter obligation. Operators opt properties in by marking
/// a supported non-unique Bool/String property as indexed.
pub fn vector_native_filter_properties(
    manifest: &Manifest,
    descriptor: &VectorIndexDescriptor,
) -> Vec<String> {
    let mut filters = manifest
        .schema
        .label(&descriptor.label)
        .into_iter()
        .flat_map(|label| label.properties.iter())
        .filter(|property| {
            property.indexed
                && !property.unique
                && matches!(
                    property.data_type,
                    DataType::Bool | DataType::Utf8 | DataType::LargeUtf8
                )
        })
        .map(|property| property.name.clone())
        .collect::<Vec<_>>();
    filters.sort();
    filters.dedup();
    filters
}

/// Stable vector catalog signature.
///
/// It covers the registered descriptor plus the sorted, explicitly indexed,
/// non-unique Bool/String native-filter schema. It intentionally does not
/// depend on schema declaration order or platform `usize` width.
pub fn vector_catalog_signature(manifest: &Manifest, descriptor: &VectorIndexDescriptor) -> String {
    let mut hasher = signature_hasher(SearchLsmKind::Vector);
    hash_str(&mut hasher, &descriptor.name);
    hash_str(&mut hasher, &descriptor.label);
    hash_str(&mut hasher, &descriptor.property);
    hash_u64(&mut hasher, descriptor.dim as u64);
    hasher.update(&[match descriptor.metric {
        VectorMetric::Cosine => 0,
        VectorMetric::Dot => 1,
        VectorMetric::Euclidean => 2,
    }]);
    hash_u64(&mut hasher, descriptor.r as u64);
    hash_u64(&mut hasher, descriptor.l_build as u64);
    hasher.update(&descriptor.alpha.to_bits().to_le_bytes());
    hasher.update(&[match descriptor.quantization {
        VectorQuantization::None => 0,
        VectorQuantization::Int8 => 1,
    }]);
    hasher.update(&VECTOR_FILTER_ENCODING_VERSION.to_le_bytes());

    let mut filters = manifest
        .schema
        .label(&descriptor.label)
        .into_iter()
        .flat_map(|label| label.properties.iter())
        .filter(|property| {
            property.indexed
                && !property.unique
                && matches!(
                    property.data_type,
                    DataType::Bool | DataType::Utf8 | DataType::LargeUtf8
                )
        })
        .collect::<Vec<_>>();
    filters.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| data_type_tag(&left.data_type).cmp(&data_type_tag(&right.data_type)))
            .then_with(|| left.indexed.cmp(&right.indexed))
            .then_with(|| left.unique.cmp(&right.unique))
    });
    hash_u64(&mut hasher, filters.len() as u64);
    for property in filters {
        hash_str(&mut hasher, &property.name);
        hash_data_type(&mut hasher, &property.data_type);
        hasher.update(&[property.indexed as u8, property.unique as u8]);
    }
    finish_signature(hasher)
}

/// Catalog signature emitted by the pre-Search-LSM compactor.
///
/// A metadata-only V5 migration may accept this value in
/// `SearchIndexBuildState` in addition to [`vector_catalog_signature`]. It
/// must never ignore the build marker altogether: doing so could adopt a body
/// built for an older label/property/filter catalog after DDL.
pub(crate) fn legacy_vector_catalog_signature(
    manifest: &Manifest,
    descriptor: &VectorIndexDescriptor,
) -> String {
    let mut filter_properties: Vec<(&String, &DataType, bool, bool)> = manifest
        .schema
        .label(&descriptor.label)
        .into_iter()
        .flat_map(|label| &label.properties)
        .filter(|property| {
            (property.indexed || property.unique)
                && matches!(
                    property.data_type,
                    DataType::Bool | DataType::Utf8 | DataType::LargeUtf8
                )
        })
        .map(|property| {
            (
                &property.name,
                &property.data_type,
                property.indexed,
                property.unique,
            )
        })
        .collect();
    filter_properties.sort_by(|left, right| left.0.cmp(right.0));
    serde_json::to_string(&(descriptor, filter_properties))
        .expect("vector descriptor and filter schema serialize")
}

/// Stable text catalog signature.
///
/// Text property order is canonicalized because `TextIndexDescriptor::new`
/// gives set semantics. The analyzer version is explicit so a tokenizer change
/// cannot inherit an old generation accidentally.
pub fn text_catalog_signature(descriptor: &TextIndexDescriptor) -> String {
    let mut hasher = signature_hasher(SearchLsmKind::Text);
    hash_str(&mut hasher, &descriptor.name);
    hash_str(&mut hasher, &descriptor.label);
    hasher.update(&TEXT_ANALYZER_VERSION.to_le_bytes());
    let mut properties = descriptor.properties.clone();
    properties.sort();
    properties.dedup();
    hash_u64(&mut hasher, properties.len() as u64);
    for property in properties {
        hash_str(&mut hasher, &property);
    }
    finish_signature(hasher)
}

/// Canonical complete native-filter property list for a text descriptor.
///
/// As with vector indexes, high-cardinality identity constraints are not an
/// implicit search-filter obligation. Only explicitly indexed, non-unique
/// Bool/String properties on the indexed label participate.
pub fn text_native_filter_properties(
    manifest: &Manifest,
    descriptor: &TextIndexDescriptor,
) -> Vec<String> {
    let mut filters = manifest
        .schema
        .label(&descriptor.label)
        .into_iter()
        .flat_map(|label| label.properties.iter())
        .filter(|property| {
            property.indexed
                && !property.unique
                && matches!(
                    property.data_type,
                    DataType::Bool | DataType::Utf8 | DataType::LargeUtf8
                )
        })
        .map(|property| property.name.clone())
        .collect::<Vec<_>>();
    filters.sort();
    filters.dedup();
    filters
}

/// Search-LSM text signature, including every native-filter obligation.
///
/// [`text_catalog_signature`] intentionally remains the legacy/no-filter
/// signature used by pre-FT4 build markers. New generations use this
/// manifest-aware signature so a schema change cannot inherit segments whose
/// complete filter set was built for an older catalog.
pub fn text_lsm_catalog_signature(manifest: &Manifest, descriptor: &TextIndexDescriptor) -> String {
    let mut hasher = signature_hasher(SearchLsmKind::Text);
    hash_str(&mut hasher, &descriptor.name);
    hash_str(&mut hasher, &descriptor.label);
    hasher.update(&TEXT_ANALYZER_VERSION.to_le_bytes());
    let mut properties = descriptor.properties.clone();
    properties.sort();
    properties.dedup();
    hash_u64(&mut hasher, properties.len() as u64);
    for property in properties {
        hash_str(&mut hasher, &property);
    }
    hasher.update(&VECTOR_FILTER_ENCODING_VERSION.to_le_bytes());
    let mut filters = manifest
        .schema
        .label(&descriptor.label)
        .into_iter()
        .flat_map(|label| label.properties.iter())
        .filter(|property| {
            property.indexed
                && !property.unique
                && matches!(
                    property.data_type,
                    DataType::Bool | DataType::Utf8 | DataType::LargeUtf8
                )
        })
        .collect::<Vec<_>>();
    filters.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| data_type_tag(&left.data_type).cmp(&data_type_tag(&right.data_type)))
            .then_with(|| left.indexed.cmp(&right.indexed))
            .then_with(|| left.unique.cmp(&right.unique))
    });
    hash_u64(&mut hasher, filters.len() as u64);
    for property in filters {
        hash_str(&mut hasher, &property.name);
        hash_data_type(&mut hasher, &property.data_type);
        hasher.update(&[property.indexed as u8, property.unique as u8]);
    }
    finish_signature(hasher)
}

/// Catalog signature emitted by the pre-Search-LSM text compactor.
pub(crate) fn legacy_text_catalog_signature(descriptor: &TextIndexDescriptor) -> String {
    serde_json::to_string(descriptor).expect("text descriptor serializes")
}

/// Resolve exactly one catalog descriptor and compute its signature.
///
/// `None` means missing or ambiguous catalog identity, which invariant 1
/// rejects for an LSM state.
pub fn catalog_signature(
    manifest: &Manifest,
    kind: SearchLsmKind,
    index_name: &str,
) -> Option<String> {
    match kind {
        SearchLsmKind::Vector => {
            let mut matches = manifest
                .vector_indexes
                .iter()
                .filter(|descriptor| descriptor.name == index_name);
            let descriptor = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            Some(vector_catalog_signature(manifest, descriptor))
        }
        SearchLsmKind::Text => {
            let mut matches = manifest
                .text_indexes
                .iter()
                .filter(|descriptor| descriptor.name == index_name);
            let descriptor = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            Some(text_lsm_catalog_signature(manifest, descriptor))
        }
    }
}

/// Digest of the manifest fields that a legacy V5/V3 native footer must
/// reproduce when it is opened.
///
/// The native reader first validates magic + footer CRC. The read path then
/// compares its decoded counts, metric/dimension, bounds and complete filters
/// with the descriptor and this digest, giving legacy bases the same
/// generation binding without changing their wire format.
pub fn legacy_base_content_fingerprint(
    descriptor: &SstDescriptor,
    format: SearchSegmentFormat,
    complete_filter_properties: &[String],
) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"namidb-search-lsm-legacy-base");
    hasher.update(&LEGACY_BASE_BINDING_VERSION.to_le_bytes());
    hasher.update(&[search_segment_format_tag(format)]);
    hasher.update(descriptor.id.as_bytes());
    hasher.update(&[sst_kind_tag(descriptor.kind)]);
    hash_str(&mut hasher, &descriptor.scope);
    hash_u64(&mut hasher, descriptor.size_bytes);
    hash_u64(&mut hasher, descriptor.row_count);
    hasher.update(&descriptor.min_key);
    hasher.update(&descriptor.max_key);
    hash_u64(&mut hasher, descriptor.min_lsn);
    hash_u64(&mut hasher, descriptor.max_lsn);
    match &descriptor.kind_specific {
        KindSpecificStats::VectorGraph {
            dim,
            metric,
            point_count,
            r,
            l_build,
            alpha,
            entry_medoid,
        } => {
            hasher.update(&[0]);
            hasher.update(&dim.to_le_bytes());
            hash_str(&mut hasher, metric);
            hash_u64(&mut hasher, *point_count);
            hash_u64(&mut hasher, *r as u64);
            hash_u64(&mut hasher, *l_build as u64);
            hasher.update(&alpha.to_bits().to_le_bytes());
            hasher.update(&entry_medoid.to_le_bytes());
        }
        KindSpecificStats::TextIndex {
            doc_count,
            term_count,
            total_len,
        } => {
            hasher.update(&[1]);
            hash_u64(&mut hasher, *doc_count);
            hash_u64(&mut hasher, *term_count);
            hash_u64(&mut hasher, *total_len);
        }
        _ => {
            hasher.update(&[u8::MAX]);
        }
    }
    hash_u64(&mut hasher, complete_filter_properties.len() as u64);
    for property in complete_filter_properties {
        hash_str(&mut hasher, property);
    }
    let digest = hasher.finalize();
    let fingerprint = u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 digest has at least eight bytes"),
    );
    // Zero is reserved as "builder did not bind this segment".
    fingerprint.max(1)
}

/// Wrap one already-authoritative NAMIVG05/NAMIFT03 body as an active
/// generation. The caller must write the barrier body/descriptor and publish
/// both together in one manifest CAS.
pub fn wrap_legacy_search_base(
    manifest: &Manifest,
    kind: SearchLsmKind,
    index_name: &str,
    base_sst_id: Uuid,
    generation_id: Uuid,
    barrier_sst_id: Uuid,
) -> Result<SearchLsmState, SearchLsmValidationError> {
    let signature = catalog_signature(manifest, kind, index_name).ok_or_else(|| {
        SearchLsmValidationError::new(
            SearchLsmInvariant::Catalog,
            index_name,
            "catalog descriptor is missing or ambiguous",
        )
    })?;
    if generation_id.is_nil() || barrier_sst_id.is_nil() || generation_id == barrier_sst_id {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::Catalog,
            index_name,
            "generation and barrier UUIDs must be distinct and non-nil",
        ));
    }
    let mut base_matches = manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.id == base_sst_id);
    let base = base_matches.next().ok_or_else(|| {
        SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            index_name,
            format!("legacy base descriptor {base_sst_id} is missing"),
        )
    })?;
    if base_matches.next().is_some() || base.kind != kind.sst_kind() || base.scope != index_name {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            index_name,
            "legacy base UUID is ambiguous or has the wrong kind/scope",
        ));
    }
    let mut nodes = manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
        .collect::<Vec<_>>();
    nodes.sort_by_key(|descriptor| *descriptor.id.as_bytes());
    if nodes.is_empty() {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::NodeCoverage,
            index_name,
            "cannot register a physical legacy base without a visible Nodes SST",
        ));
    }
    if let Some(outrunning) = nodes
        .iter()
        .find(|node| node.max_lsn > base.max_lsn)
        .copied()
    {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::NodeCoverage,
            index_name,
            format!(
                "Nodes SST {} at LSN {} outruns legacy base LSN {}",
                outrunning.id, outrunning.max_lsn, base.max_lsn
            ),
        ));
    }
    let event_end = nodes.len() as u64;
    let format = match kind {
        SearchLsmKind::Vector => SearchSegmentFormat::VectorV5Base,
        SearchLsmKind::Text => SearchSegmentFormat::TextV3Base,
    };
    let complete_filter_properties = match kind {
        SearchLsmKind::Vector => {
            let descriptor = manifest
                .vector_indexes
                .iter()
                .find(|descriptor| descriptor.name == index_name)
                .expect("catalog identity checked above");
            vector_native_filter_properties(manifest, descriptor)
        }
        SearchLsmKind::Text => Vec::new(),
    };
    let (live_payload_count, stats) = match (&kind, &base.kind_specific) {
        (SearchLsmKind::Vector, KindSpecificStats::VectorGraph { point_count, .. }) => (
            *point_count,
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Absolute(*point_count),
            },
        ),
        (
            SearchLsmKind::Text,
            KindSpecificStats::TextIndex {
                doc_count,
                total_len,
                ..
            },
        ) => (
            *doc_count,
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Absolute(*doc_count),
                total_len: SearchStatValue::Absolute(*total_len),
                term_df_violation_count: 0,
            },
        ),
        _ => {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::PhysicalSegments,
                index_name,
                "legacy base descriptor has mismatched kind statistics",
            ));
        }
    };
    if base.row_count != live_payload_count {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            index_name,
            "legacy base row count disagrees with kind statistics",
        ));
    }
    let event_ranges = vec![SearchEventRange::new(0, event_end)];
    let coverage = nodes
        .into_iter()
        .enumerate()
        .map(|(event, node)| SearchCoverage {
            node_sst_id: node.id,
            node_sst_max_lsn: node.max_lsn,
            event_ranges: vec![SearchEventRange::new(event as u64, event as u64 + 1)],
            disposition: CoverageDisposition::Segment,
        })
        .collect();
    Ok(SearchLsmState {
        index_name: index_name.into(),
        kind,
        catalog_signature: signature,
        generation_id,
        status: SearchLsmStatus::Active,
        next_event_seq: event_end,
        base_frontier: Some(event_end),
        segments: vec![SearchSegmentRef {
            sst_id: base.id,
            role: SearchSegmentRole::Base,
            format,
            payload: SearchSegmentPayload::Complete,
            event_ranges,
            min_lsn: base.min_lsn,
            max_lsn: base.max_lsn,
            mutation_count: live_payload_count,
            live_payload_count,
            suppress_count: 0,
            content_xxh3: legacy_base_content_fingerprint(
                base,
                format,
                &complete_filter_properties,
            ),
            complete_filter_properties,
            stats,
            equal_lsn_conflict_count: 0,
        }],
        proven_empty_event_ranges: Vec::new(),
        coverage,
        compat_barrier_sst_id: Some(barrier_sst_id),
        equal_lsn_conflict_count: 0,
    })
}

/// Select the only search path that is safe from manifest metadata.
pub fn select_search_read_plan(
    manifest: &Manifest,
    kind: SearchLsmKind,
    index_name: &str,
) -> SearchReadPlan {
    let matching_states = manifest
        .search_lsm
        .iter()
        .filter(|state| state.kind == kind && state.index_name == index_name)
        .collect::<Vec<_>>();
    if matching_states.is_empty() {
        let bodies = manifest
            .ssts
            .iter()
            .filter(|descriptor| {
                descriptor.kind == kind.sst_kind() && descriptor.scope == index_name
            })
            .collect::<Vec<_>>();
        return match bodies.as_slice() {
            [] => SearchReadPlan::FlatFallback(SearchReadFallback::NoPhysicalBody),
            [body] => SearchReadPlan::Legacy { sst_id: body.id },
            _ => SearchReadPlan::FlatFallback(SearchReadFallback::AmbiguousPhysicalBodies),
        };
    }
    if matching_states.len() != 1 {
        return SearchReadPlan::FlatFallback(SearchReadFallback::InvalidGeneration);
    }
    let state = matching_states[0];
    if state.status != SearchLsmStatus::Active {
        return SearchReadPlan::FlatFallback(SearchReadFallback::BuildingGeneration);
    }
    if validate_search_lsm(manifest).is_err() {
        return SearchReadPlan::FlatFallback(SearchReadFallback::InvalidGeneration);
    }
    let Some(barrier_sst_id) = state.compat_barrier_sst_id else {
        return SearchReadPlan::FlatFallback(SearchReadFallback::InvalidGeneration);
    };
    if state.segments.len() > MAX_ACTIVE_SEARCH_SEGMENTS {
        return SearchReadPlan::FlatFallback(SearchReadFallback::SegmentLimitExceeded);
    }
    if state
        .segments
        .iter()
        .any(|segment| segment.payload != SearchSegmentPayload::Complete)
    {
        return SearchReadPlan::FlatFallback(SearchReadFallback::UnsupportedSegmentSet);
    }
    if let [base] = state.segments.as_slice() {
        let expected_format = match kind {
            SearchLsmKind::Vector => SearchSegmentFormat::VectorV5Base,
            SearchLsmKind::Text => SearchSegmentFormat::TextV3Base,
        };
        if base.role == SearchSegmentRole::Base && base.format == expected_format {
            return SearchReadPlan::ActiveLegacyBase {
                state: state.clone(),
                base_sst_id: base.sst_id,
                barrier_sst_id,
            };
        }
    }

    let native_set_supported = match kind {
        SearchLsmKind::Vector => state
            .segments
            .iter()
            .enumerate()
            .all(|(position, segment)| {
                segment.format == SearchSegmentFormat::VectorV6
                    || (position == 0
                        && segment.role == SearchSegmentRole::Base
                        && segment.format == SearchSegmentFormat::VectorV5Base)
            }),
        // V3 cannot yet consume caller-supplied global delta statistics.
        // Rebuilding that base to FT4 is required before segmented BM25 may
        // serve; falling back here preserves exact ranking during migration.
        SearchLsmKind::Text => state
            .segments
            .iter()
            .all(|segment| segment.format == SearchSegmentFormat::TextV4),
    };
    if native_set_supported {
        SearchReadPlan::ActiveSegments {
            state: state.clone(),
            barrier_sst_id,
        }
    } else {
        SearchReadPlan::FlatFallback(SearchReadFallback::UnsupportedSegmentSet)
    }
}

/// Validate every Search-LSM state using manifest metadata only.
///
/// `Building` states validate catalog identity and any references they already
/// publish, but may be incomplete. `Active` states enforce invariants 1-8 and
/// 10. Invariant 9 (footer/object binding) belongs to object readers: the
/// migration read path enforces it for barrier + V5/V3, and VG6/FT4 will bind
/// it directly in each segment footer.
/// Retire every Search-LSM generation whose catalog signature no longer
/// matches `manifest`'s current catalog/schema — the aftermath of a property
/// index / uniqueness DDL that changed the native-filter set. Removes the
/// generation state, its physical segment and barrier descriptors, and any
/// interop build marker for the same index, leaving a manifest that
/// validates cleanly and lets the next flush start a fresh Building
/// generation with the new signature. Returns the retired index names.
pub fn retire_signature_stale_generations(manifest: &mut Manifest) -> Vec<String> {
    let mut retired_names = Vec::new();
    let mut retired_ids: HashSet<Uuid> = HashSet::new();
    let mut kept = Vec::with_capacity(manifest.search_lsm.len());
    for state in std::mem::take(&mut manifest.search_lsm) {
        let expected = catalog_signature(manifest, state.kind, &state.index_name);
        if expected.as_deref() == Some(state.catalog_signature.as_str()) {
            kept.push(state);
            continue;
        }
        for segment in &state.segments {
            retired_ids.insert(segment.sst_id);
        }
        if let Some(barrier) = state.compat_barrier_sst_id {
            retired_ids.insert(barrier);
        }
        let sst_kind = state.kind.sst_kind();
        manifest
            .search_index_builds
            .retain(|marker| !(marker.kind == sst_kind && marker.name == state.index_name));
        retired_names.push(state.index_name);
    }
    manifest.search_lsm = kept;
    if !retired_ids.is_empty() {
        manifest
            .ssts
            .retain(|descriptor| !retired_ids.contains(&descriptor.id));
    }
    retired_names
}

pub fn validate_search_lsm(manifest: &Manifest) -> Result<(), SearchLsmValidationError> {
    let mut keys = HashSet::new();
    let mut claimed_descriptors: HashMap<Uuid, String> = HashMap::new();
    for state in &manifest.search_lsm {
        let key = (state.kind, state.index_name.as_str());
        if !keys.insert(key) {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::UniqueState,
                &state.index_name,
                "more than one state has the same kind and index name",
            ));
        }
        validate_catalog(manifest, state)?;
        validate_state(manifest, state, &mut claimed_descriptors)?;
    }
    Ok(())
}

fn validate_catalog(
    manifest: &Manifest,
    state: &SearchLsmState,
) -> Result<(), SearchLsmValidationError> {
    let Some(expected) = catalog_signature(manifest, state.kind, &state.index_name) else {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::Catalog,
            &state.index_name,
            "catalog descriptor is missing or ambiguous",
        ));
    };
    if state.catalog_signature != expected {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::Catalog,
            &state.index_name,
            "catalog signature does not match the registered descriptor/schema",
        ));
    }
    if state.generation_id.is_nil() {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::Catalog,
            &state.index_name,
            "generation UUID is nil",
        ));
    }
    Ok(())
}

fn validate_state(
    manifest: &Manifest,
    state: &SearchLsmState,
    claimed_descriptors: &mut HashMap<Uuid, String>,
) -> Result<(), SearchLsmValidationError> {
    let descriptors = descriptors_by_id(&manifest.ssts);
    let expected_kind = state.kind.sst_kind();
    let mut data_ids = HashSet::new();

    for segment in &state.segments {
        if !data_ids.insert(segment.sst_id) {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::PhysicalSegments,
                &state.index_name,
                format!("segment {} is listed more than once", segment.sst_id),
            ));
        }
        claim_descriptor(claimed_descriptors, state, segment.sst_id)?;
        let descriptor = unique_descriptor(&descriptors, state, segment.sst_id)?;
        validate_physical_descriptor(state, descriptor, expected_kind, false)?;
        validate_segment_catalog_binding(manifest, state, descriptor)?;
        validate_segment_metadata(state, segment, descriptor)?;
    }

    if let Some(barrier_id) = state.compat_barrier_sst_id {
        if data_ids.contains(&barrier_id) {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::CompatibilityBarrier,
                &state.index_name,
                "compatibility barrier is also listed as a data segment",
            ));
        }
        claim_descriptor(claimed_descriptors, state, barrier_id)?;
        let barrier = unique_descriptor(&descriptors, state, barrier_id).map_err(|error| {
            SearchLsmValidationError::new(
                SearchLsmInvariant::CompatibilityBarrier,
                &state.index_name,
                error.detail,
            )
        })?;
        validate_physical_descriptor(state, barrier, expected_kind, true).map_err(|error| {
            SearchLsmValidationError::new(
                SearchLsmInvariant::CompatibilityBarrier,
                &state.index_name,
                error.detail,
            )
        })?;
        validate_barrier_descriptor(state, barrier)?;
    } else if state.status == SearchLsmStatus::Active {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::CompatibilityBarrier,
            &state.index_name,
            "active generation has no compatibility barrier",
        ));
    }

    if state.status == SearchLsmStatus::Building {
        return Ok(());
    }

    // In an active generation, every same-kind/scope physical object is
    // classified exactly once as data or barrier. Legacy/unlisted bodies make
    // the generation ambiguous and therefore unavailable.
    let barrier_id = state
        .compat_barrier_sst_id
        .expect("active barrier checked above");
    for descriptor in manifest.ssts.iter().filter(|descriptor| {
        descriptor.kind == expected_kind && descriptor.scope == state.index_name
    }) {
        if descriptor.id != barrier_id && !data_ids.contains(&descriptor.id) {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::PhysicalSegments,
                &state.index_name,
                format!("physical descriptor {} is not classified", descriptor.id),
            ));
        }
    }

    validate_segment_ordering(state)?;
    validate_node_coverage(manifest, state)?;
    validate_event_partition(state)?;
    validate_statistics(state)?;
    if state.equal_lsn_conflict_count != 0
        || state
            .segments
            .iter()
            .any(|segment| segment.equal_lsn_conflict_count != 0)
    {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::VersionTie,
            &state.index_name,
            "equal-LSN records with different payload fingerprints were reported",
        ));
    }
    Ok(())
}

fn descriptors_by_id(ssts: &[SstDescriptor]) -> HashMap<Uuid, Vec<&SstDescriptor>> {
    let mut by_id: HashMap<Uuid, Vec<&SstDescriptor>> = HashMap::new();
    for descriptor in ssts {
        by_id.entry(descriptor.id).or_default().push(descriptor);
    }
    by_id
}

fn claim_descriptor(
    claimed: &mut HashMap<Uuid, String>,
    state: &SearchLsmState,
    id: Uuid,
) -> Result<(), SearchLsmValidationError> {
    if let Some(owner) = claimed.insert(id, state.index_name.clone()) {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!("descriptor {id} is already claimed by '{owner}'"),
        ));
    }
    Ok(())
}

fn unique_descriptor<'a>(
    descriptors: &'a HashMap<Uuid, Vec<&'a SstDescriptor>>,
    state: &SearchLsmState,
    id: Uuid,
) -> Result<&'a SstDescriptor, SearchLsmValidationError> {
    match descriptors.get(&id).map(Vec::as_slice) {
        Some([descriptor]) => Ok(*descriptor),
        None => Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!("descriptor {id} is missing"),
        )),
        Some(_) => Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!("descriptor UUID {id} is not unique"),
        )),
    }
}

fn validate_physical_descriptor(
    state: &SearchLsmState,
    descriptor: &SstDescriptor,
    expected_kind: SstKind,
    barrier: bool,
) -> Result<(), SearchLsmValidationError> {
    if descriptor.kind != expected_kind || descriptor.scope != state.index_name {
        return Err(SearchLsmValidationError::new(
            if barrier {
                SearchLsmInvariant::CompatibilityBarrier
            } else {
                SearchLsmInvariant::PhysicalSegments
            },
            &state.index_name,
            format!(
                "descriptor {} has kind/scope {:?}/'{}', expected {:?}/'{}'",
                descriptor.id, descriptor.kind, descriptor.scope, expected_kind, state.index_name
            ),
        ));
    }
    let stats_match = matches!(
        (expected_kind, &descriptor.kind_specific),
        (SstKind::VectorGraph, KindSpecificStats::VectorGraph { .. })
            | (SstKind::TextIndex, KindSpecificStats::TextIndex { .. })
    );
    if !stats_match {
        return Err(SearchLsmValidationError::new(
            if barrier {
                SearchLsmInvariant::CompatibilityBarrier
            } else {
                SearchLsmInvariant::PhysicalSegments
            },
            &state.index_name,
            format!(
                "descriptor {} has mismatched kind statistics",
                descriptor.id
            ),
        ));
    }
    Ok(())
}

fn validate_segment_catalog_binding(
    manifest: &Manifest,
    state: &SearchLsmState,
    descriptor: &SstDescriptor,
) -> Result<(), SearchLsmValidationError> {
    let matches_catalog = match state.kind {
        SearchLsmKind::Vector => {
            let Some(index) = manifest
                .vector_indexes
                .iter()
                .find(|index| index.name == state.index_name)
            else {
                return Err(SearchLsmValidationError::new(
                    SearchLsmInvariant::Catalog,
                    &state.index_name,
                    "vector catalog descriptor disappeared during validation",
                ));
            };
            let KindSpecificStats::VectorGraph {
                dim,
                metric,
                r,
                l_build,
                alpha,
                ..
            } = &descriptor.kind_specific
            else {
                return Ok(());
            };
            *dim == index.dim
                && metric
                    == match index.metric {
                        VectorMetric::Cosine => "cosine",
                        VectorMetric::Dot => "dot",
                        VectorMetric::Euclidean => "euclidean",
                    }
                && *r == index.r
                && *l_build == index.l_build
                && alpha.to_bits() == index.alpha.to_bits()
        }
        // Text V3 has no index-configuration fields beyond the catalog-bound
        // analyzer/properties; its decoded corpus statistics are checked by
        // the native footer binding.
        SearchLsmKind::Text => true,
    };
    if !matches_catalog {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!(
                "segment descriptor {} disagrees with its catalog configuration",
                descriptor.id
            ),
        ));
    }
    Ok(())
}

fn validate_segment_metadata(
    state: &SearchLsmState,
    segment: &SearchSegmentRef,
    descriptor: &SstDescriptor,
) -> Result<(), SearchLsmValidationError> {
    let format_matches = matches!(
        (state.kind, segment.role, segment.format),
        (
            SearchLsmKind::Vector,
            SearchSegmentRole::Base,
            SearchSegmentFormat::VectorV5Base | SearchSegmentFormat::VectorV6
        ) | (
            SearchLsmKind::Vector,
            SearchSegmentRole::Delta,
            SearchSegmentFormat::VectorV6
        ) | (
            SearchLsmKind::Text,
            SearchSegmentRole::Base,
            SearchSegmentFormat::TextV3Base | SearchSegmentFormat::TextV4
        ) | (
            SearchLsmKind::Text,
            SearchSegmentRole::Delta,
            SearchSegmentFormat::TextV4
        )
    );
    if !format_matches {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!(
                "segment {} format {:?} is invalid for {:?} {:?}",
                segment.sst_id, segment.format, state.kind, segment.role
            ),
        ));
    }
    if segment.mutation_count > 0 && segment.min_lsn > segment.max_lsn {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!("segment {} has reversed LSN bounds", segment.sst_id),
        ));
    }
    if segment
        .live_payload_count
        .checked_add(segment.suppress_count)
        != Some(segment.mutation_count)
    {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!("segment {} mutation counts do not add up", segment.sst_id),
        ));
    }
    if !is_strictly_sorted(&segment.complete_filter_properties) {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!(
                "segment {} filter properties are not strictly sorted",
                segment.sst_id
            ),
        ));
    }
    if segment.min_lsn != descriptor.min_lsn || segment.max_lsn != descriptor.max_lsn {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::PhysicalSegments,
            &state.index_name,
            format!(
                "segment {} LSN bounds disagree with its descriptor",
                segment.sst_id
            ),
        ));
    }
    if matches!(
        segment.format,
        SearchSegmentFormat::VectorV5Base | SearchSegmentFormat::TextV3Base
    ) {
        if segment.role != SearchSegmentRole::Base
            || segment.suppress_count != 0
            || descriptor.row_count != segment.live_payload_count
            || segment.mutation_count != segment.live_payload_count
        {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::PhysicalSegments,
                &state.index_name,
                format!(
                    "legacy base segment {} has inconsistent physical counts",
                    segment.sst_id
                ),
            ));
        }
        let expected_fingerprint = legacy_base_content_fingerprint(
            descriptor,
            segment.format,
            &segment.complete_filter_properties,
        );
        if segment.content_xxh3 == 0 || segment.content_xxh3 != expected_fingerprint {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::PhysicalSegments,
                &state.index_name,
                format!(
                    "legacy base segment {} descriptor/footer fingerprint disagrees",
                    segment.sst_id
                ),
            ));
        }
    }
    Ok(())
}

fn validate_barrier_descriptor(
    state: &SearchLsmState,
    descriptor: &SstDescriptor,
) -> Result<(), SearchLsmValidationError> {
    if !is_canonical_search_barrier_descriptor(descriptor) {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::CompatibilityBarrier,
            &state.index_name,
            "barrier descriptor is not the canonical zero-valued artifact",
        ));
    }
    Ok(())
}

fn validate_segment_ordering(state: &SearchLsmState) -> Result<(), SearchLsmValidationError> {
    let mut base_count = 0usize;
    let mut previous_first = None;
    let mut all_ranges = Vec::new();
    for (position, segment) in state.segments.iter().enumerate() {
        validate_ranges(
            state,
            &segment.event_ranges,
            SearchLsmInvariant::SegmentOrdering,
            "segment",
        )?;
        let first = segment
            .event_ranges
            .first()
            .expect("validate_ranges rejects empty range lists")
            .start;
        if previous_first.is_some_and(|previous| first <= previous) {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::SegmentOrdering,
                &state.index_name,
                "segments are not strictly ordered by first event",
            ));
        }
        previous_first = Some(first);
        match segment.role {
            SearchSegmentRole::Base => {
                base_count += 1;
                if position != 0 {
                    return Err(SearchLsmValidationError::new(
                        SearchLsmInvariant::SegmentOrdering,
                        &state.index_name,
                        "base is not the first segment",
                    ));
                }
            }
            SearchSegmentRole::Delta => {}
        }
        all_ranges.extend(segment.event_ranges.iter().copied());
    }
    if base_count > 1 {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::SegmentOrdering,
            &state.index_name,
            "more than one base segment exists",
        ));
    }
    match (state.base_frontier, state.segments.first()) {
        (Some(frontier), Some(base)) if base.role == SearchSegmentRole::Base => {
            if frontier == 0 || base.event_ranges.as_slice() != [SearchEventRange::new(0, frontier)]
            {
                return Err(SearchLsmValidationError::new(
                    SearchLsmInvariant::SegmentOrdering,
                    &state.index_name,
                    "base must own exactly the contiguous prefix 0..base_frontier",
                ));
            }
            if state.segments.iter().skip(1).any(|segment| {
                segment
                    .event_ranges
                    .iter()
                    .any(|range| range.start < frontier)
            }) {
                return Err(SearchLsmValidationError::new(
                    SearchLsmInvariant::SegmentOrdering,
                    &state.index_name,
                    "delta begins before the base frontier",
                ));
            }
        }
        (None, Some(first)) if first.role == SearchSegmentRole::Delta => {}
        (None, None) => {}
        _ => {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::SegmentOrdering,
                &state.index_name,
                "base_frontier and base segment disagree",
            ));
        }
    }
    ensure_non_overlapping(
        state,
        &mut all_ranges,
        SearchLsmInvariant::SegmentOrdering,
        "segment event ranges overlap",
    )
}

fn validate_node_coverage(
    manifest: &Manifest,
    state: &SearchLsmState,
) -> Result<(), SearchLsmValidationError> {
    let mut node_descriptors: HashMap<Uuid, Vec<&SstDescriptor>> = HashMap::new();
    for descriptor in manifest
        .ssts
        .iter()
        .filter(|descriptor| descriptor.kind == SstKind::Nodes)
    {
        node_descriptors
            .entry(descriptor.id)
            .or_default()
            .push(descriptor);
    }
    let mut covered = HashSet::new();
    for coverage in &state.coverage {
        if !covered.insert(coverage.node_sst_id) {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::NodeCoverage,
                &state.index_name,
                format!(
                    "Nodes SST {} has more than one coverage entry",
                    coverage.node_sst_id
                ),
            ));
        }
        let descriptor = match node_descriptors
            .get(&coverage.node_sst_id)
            .map(Vec::as_slice)
        {
            Some([descriptor]) => *descriptor,
            None => {
                return Err(SearchLsmValidationError::new(
                    SearchLsmInvariant::NodeCoverage,
                    &state.index_name,
                    format!(
                        "coverage references missing Nodes SST {}",
                        coverage.node_sst_id
                    ),
                ));
            }
            Some(_) => {
                return Err(SearchLsmValidationError::new(
                    SearchLsmInvariant::NodeCoverage,
                    &state.index_name,
                    format!("Nodes SST UUID {} is not unique", coverage.node_sst_id),
                ));
            }
        };
        if descriptor.max_lsn != coverage.node_sst_max_lsn {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::NodeCoverage,
                &state.index_name,
                format!(
                    "coverage max LSN for {} is {}, descriptor has {}",
                    coverage.node_sst_id, coverage.node_sst_max_lsn, descriptor.max_lsn
                ),
            ));
        }
        validate_ranges(
            state,
            &coverage.event_ranges,
            SearchLsmInvariant::NodeCoverage,
            "coverage",
        )?;
    }
    if covered.len() != node_descriptors.len() {
        let missing = node_descriptors
            .keys()
            .find(|id| !covered.contains(id))
            .copied()
            .unwrap_or_default();
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::NodeCoverage,
            &state.index_name,
            format!("Nodes SST {missing} has no coverage entry"),
        ));
    }
    Ok(())
}

fn validate_event_partition(state: &SearchLsmState) -> Result<(), SearchLsmValidationError> {
    if !state.proven_empty_event_ranges.is_empty() {
        validate_ranges(
            state,
            &state.proven_empty_event_ranges,
            SearchLsmInvariant::EventCoverage,
            "proven-empty",
        )?;
    }
    let mut owners = state
        .segments
        .iter()
        .flat_map(|segment| segment.event_ranges.iter().copied())
        .chain(state.proven_empty_event_ranges.iter().copied())
        .collect::<Vec<_>>();
    owners.sort_by_key(|range| (range.start, range.end));
    let mut cursor = 0u64;
    for range in owners {
        if range.start != cursor {
            let detail = if range.start < cursor {
                "search events have multiple owners"
            } else {
                "search event coverage has a gap"
            };
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::EventCoverage,
                &state.index_name,
                detail,
            ));
        }
        cursor = range.end;
    }
    if cursor != state.next_event_seq {
        return Err(SearchLsmValidationError::new(
            SearchLsmInvariant::EventCoverage,
            &state.index_name,
            format!(
                "event ownership ends at {cursor}, next_event_seq is {}",
                state.next_event_seq
            ),
        ));
    }

    let segment_ranges = state
        .segments
        .iter()
        .flat_map(|segment| segment.event_ranges.iter().copied())
        .collect::<Vec<_>>();
    for coverage in &state.coverage {
        let expected_owners = match coverage.disposition {
            CoverageDisposition::Segment => &segment_ranges,
            CoverageDisposition::ProvenEmpty {
                classifier_version,
                before_after_digest,
            } => {
                if classifier_version == 0 || before_after_digest == 0 {
                    return Err(SearchLsmValidationError::new(
                        SearchLsmInvariant::EventCoverage,
                        &state.index_name,
                        "ProvenEmpty coverage has a zero classifier version or digest",
                    ));
                }
                &state.proven_empty_event_ranges
            }
            CoverageDisposition::LogicalRewrite {
                input_coverage_digest,
            } => {
                if input_coverage_digest == 0 {
                    return Err(SearchLsmValidationError::new(
                        SearchLsmInvariant::EventCoverage,
                        &state.index_name,
                        "LogicalRewrite coverage has input digest zero",
                    ));
                }
                continue;
            }
            CoverageDisposition::Unknown => {
                return Err(SearchLsmValidationError::new(
                    SearchLsmInvariant::EventCoverage,
                    &state.index_name,
                    "active coverage has unknown disposition",
                ));
            }
        };
        if coverage.event_ranges.iter().any(|required| {
            !expected_owners
                .iter()
                .any(|owner| owner.contains(*required))
        }) {
            return Err(SearchLsmValidationError::new(
                SearchLsmInvariant::EventCoverage,
                &state.index_name,
                format!(
                    "coverage disposition for Nodes SST {} disagrees with event ownership",
                    coverage.node_sst_id
                ),
            ));
        }
    }
    Ok(())
}

fn validate_statistics(state: &SearchLsmState) -> Result<(), SearchLsmValidationError> {
    match state.kind {
        SearchLsmKind::Vector => {
            let mut live = 0u64;
            for segment in &state.segments {
                let SearchSegmentStats::Vector { live_count } = segment.stats else {
                    return statistics_error(state, "vector segment has non-vector statistics");
                };
                live = apply_stat(state, segment.role, live, live_count, "live_count")?;
            }
        }
        SearchLsmKind::Text => {
            let mut docs = 0u64;
            let mut total_len = 0u64;
            for segment in &state.segments {
                let SearchSegmentStats::Text {
                    doc_count,
                    total_len: segment_len,
                    term_df_violation_count,
                } = segment.stats
                else {
                    return statistics_error(state, "text segment has non-text statistics");
                };
                if term_df_violation_count != 0 {
                    return statistics_error(
                        state,
                        "text segment reports invalid/overflowing term document frequencies",
                    );
                }
                docs = apply_stat(state, segment.role, docs, doc_count, "doc_count")?;
                total_len = apply_stat(state, segment.role, total_len, segment_len, "total_len")?;
                if docs == 0 && total_len != 0 {
                    return statistics_error(
                        state,
                        "zero-document corpus has non-zero total length",
                    );
                }
            }
        }
    }
    Ok(())
}

fn apply_stat(
    state: &SearchLsmState,
    role: SearchSegmentRole,
    current: u64,
    value: SearchStatValue,
    name: &str,
) -> Result<u64, SearchLsmValidationError> {
    match (role, value) {
        (SearchSegmentRole::Base, SearchStatValue::Absolute(value)) => Ok(value),
        (SearchSegmentRole::Delta, SearchStatValue::Delta(delta)) => {
            let next = if delta >= 0 {
                current.checked_add(delta as u64)
            } else {
                current.checked_sub(delta.unsigned_abs())
            };
            next.ok_or_else(|| {
                SearchLsmValidationError::new(
                    SearchLsmInvariant::Statistics,
                    &state.index_name,
                    format!("{name} overflows or becomes negative"),
                )
            })
        }
        (SearchSegmentRole::Base, SearchStatValue::Delta(_)) => Err(SearchLsmValidationError::new(
            SearchLsmInvariant::Statistics,
            &state.index_name,
            format!("base {name} is signed instead of absolute"),
        )),
        (SearchSegmentRole::Delta, SearchStatValue::Absolute(_)) => {
            Err(SearchLsmValidationError::new(
                SearchLsmInvariant::Statistics,
                &state.index_name,
                format!("delta {name} is absolute instead of signed"),
            ))
        }
    }
}

fn statistics_error<T>(
    state: &SearchLsmState,
    detail: impl Into<String>,
) -> Result<T, SearchLsmValidationError> {
    Err(SearchLsmValidationError::new(
        SearchLsmInvariant::Statistics,
        &state.index_name,
        detail,
    ))
}

fn validate_ranges(
    state: &SearchLsmState,
    ranges: &[SearchEventRange],
    invariant: SearchLsmInvariant,
    owner: &str,
) -> Result<(), SearchLsmValidationError> {
    if ranges.is_empty() {
        return Err(SearchLsmValidationError::new(
            invariant,
            &state.index_name,
            format!("{owner} has no event ranges"),
        ));
    }
    let mut previous_end = None;
    for range in ranges {
        if !range.is_valid() || range.end > state.next_event_seq {
            return Err(SearchLsmValidationError::new(
                invariant,
                &state.index_name,
                format!(
                    "{owner} has invalid event range {}..{} for frontier {}",
                    range.start, range.end, state.next_event_seq
                ),
            ));
        }
        if previous_end.is_some_and(|end| range.start < end) {
            return Err(SearchLsmValidationError::new(
                invariant,
                &state.index_name,
                format!("{owner} ranges overlap or are out of order"),
            ));
        }
        previous_end = Some(range.end);
    }
    Ok(())
}

fn ensure_non_overlapping(
    state: &SearchLsmState,
    ranges: &mut [SearchEventRange],
    invariant: SearchLsmInvariant,
    detail: &str,
) -> Result<(), SearchLsmValidationError> {
    ranges.sort_by_key(|range| (range.start, range.end));
    if ranges.windows(2).any(|pair| pair[1].start < pair[0].end) {
        return Err(SearchLsmValidationError::new(
            invariant,
            &state.index_name,
            detail,
        ));
    }
    Ok(())
}

fn is_strictly_sorted(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn signature_hasher(kind: SearchLsmKind) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"namidb-search-lsm-catalog");
    hasher.update(&CATALOG_SIGNATURE_VERSION.to_le_bytes());
    hasher.update(&[match kind {
        SearchLsmKind::Vector => 0,
        SearchLsmKind::Text => 1,
    }]);
    hasher
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn finish_signature(hasher: blake3::Hasher) -> String {
    format!(
        "search-lsm-v{CATALOG_SIGNATURE_VERSION}:{}",
        hasher.finalize().to_hex()
    )
}

fn data_type_tag(data_type: &DataType) -> u8 {
    match data_type {
        DataType::Bool => 0,
        DataType::Int32 => 1,
        DataType::Int64 => 2,
        DataType::Float32 => 3,
        DataType::Float64 => 4,
        DataType::Utf8 => 5,
        DataType::LargeUtf8 => 6,
        DataType::Binary => 7,
        DataType::Date32 => 8,
        DataType::TimestampMicrosUtc => 9,
        DataType::FloatVector { .. } => 10,
        DataType::Int8Vector { .. } => 11,
        DataType::Json => 12,
    }
}

fn search_segment_format_tag(format: SearchSegmentFormat) -> u8 {
    match format {
        SearchSegmentFormat::Unknown => 0,
        SearchSegmentFormat::VectorV5Base => 1,
        SearchSegmentFormat::VectorV6 => 2,
        SearchSegmentFormat::TextV3Base => 3,
        SearchSegmentFormat::TextV4 => 4,
    }
}

fn sst_kind_tag(kind: SstKind) -> u8 {
    match kind {
        SstKind::Nodes => 0,
        SstKind::EdgesFwd => 1,
        SstKind::EdgesInv => 2,
        SstKind::VectorGraph => 3,
        SstKind::TextIndex => 4,
    }
}

fn hash_data_type(hasher: &mut blake3::Hasher, data_type: &DataType) {
    hasher.update(&[data_type_tag(data_type)]);
    match data_type {
        DataType::FloatVector { dim } | DataType::Int8Vector { dim } => {
            hasher.update(&dim.to_le_bytes());
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use namidb_core::{LabelDef, PropertyDef, Schema};

    use super::*;
    use crate::fence::Epoch;
    use crate::manifest::SstLevel;

    const LEGACY_MANIFEST_JSON: &str = r#"{
        "version": 7,
        "epoch": 2,
        "writer_id": "018f0000-0000-7000-8000-000000000001",
        "created_at": "2026-07-01T00:00:00Z",
        "schema": {
            "version": 0,
            "labels": {},
            "edge_types": {},
            "constraints": []
        },
        "ssts": [],
        "wal_segments": [],
        "label_dict": [],
        "vector_indexes": [],
        "text_indexes": [],
        "search_index_builds": []
    }"#;

    /// A delta body starts where the previous segment ended. Invariant 4
    /// requires the segment's declared LSN window to equal its descriptor's, so
    /// a fixture delta cannot reuse the base's `min_lsn`.
    fn delta_descriptor(
        id: Uuid,
        kind: SstKind,
        scope: &str,
        min_lsn: u64,
        max_lsn: u64,
    ) -> SstDescriptor {
        let mut descriptor = descriptor(id, kind, scope, max_lsn);
        descriptor.min_lsn = min_lsn;
        descriptor
    }

    fn descriptor(id: Uuid, kind: SstKind, scope: &str, max_lsn: u64) -> SstDescriptor {
        let kind_specific = match kind {
            SstKind::Nodes => KindSpecificStats::Nodes { tombstone_count: 0 },
            SstKind::VectorGraph => KindSpecificStats::VectorGraph {
                dim: 4,
                metric: "cosine".into(),
                point_count: 10,
                r: 16,
                l_build: 32,
                alpha: 1.2,
                entry_medoid: 0,
            },
            SstKind::TextIndex => KindSpecificStats::TextIndex {
                doc_count: 10,
                term_count: 20,
                total_len: 100,
            },
            other => panic!("unexpected fixture kind {other:?}"),
        };
        SstDescriptor {
            id,
            kind,
            scope: scope.into(),
            level: SstLevel::L0,
            path: format!("sst/level0/{id}-{kind:?}"),
            size_bytes: 1024,
            row_count: 10,
            created_at: Utc::now(),
            min_key: [0; 16],
            max_key: [0xff; 16],
            min_lsn: 1,
            max_lsn,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: Vec::new(),
            kind_specific,
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        }
    }

    fn vector_schema(filter_order_reversed: bool) -> Schema {
        let embedding =
            PropertyDef::new("embedding", DataType::FloatVector { dim: 4 }, false).unwrap();
        let tenant = PropertyDef::new("tenant", DataType::Utf8, false)
            .unwrap()
            .with_indexed(true);
        let active = PropertyDef::new("active", DataType::Bool, false)
            .unwrap()
            .with_unique(true);
        let properties = if filter_order_reversed {
            vec![tenant, embedding, active]
        } else {
            vec![active, embedding, tenant]
        };
        let mut schema = Schema::empty();
        schema.labels.insert(
            "Doc".into(),
            LabelDef {
                name: "Doc".into(),
                properties,
            },
        );
        schema
    }

    fn vector_descriptor() -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: "doc_vec".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 4,
            metric: VectorMetric::Cosine,
            r: 16,
            l_build: 32,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        }
    }

    fn vector_segment(
        sst_id: Uuid,
        role: SearchSegmentRole,
        range: SearchEventRange,
        stat: SearchStatValue,
        mutations: u64,
    ) -> SearchSegmentRef {
        SearchSegmentRef {
            sst_id,
            role,
            format: match role {
                SearchSegmentRole::Base => SearchSegmentFormat::VectorV5Base,
                SearchSegmentRole::Delta => SearchSegmentFormat::VectorV6,
            },
            payload: SearchSegmentPayload::Complete,
            event_ranges: vec![range],
            min_lsn: 1,
            max_lsn: range.end.saturating_mul(10),
            mutation_count: mutations,
            live_payload_count: mutations,
            suppress_count: 0,
            content_xxh3: 7,
            complete_filter_properties: vec!["active".into(), "tenant".into()],
            stats: SearchSegmentStats::Vector { live_count: stat },
            equal_lsn_conflict_count: 0,
        }
    }

    fn active_vector_manifest() -> Manifest {
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.schema = vector_schema(false);
        manifest.vector_indexes.push(vector_descriptor());

        let node0 = Uuid::now_v7();
        let node1 = Uuid::now_v7();
        let base = Uuid::now_v7();
        let delta = Uuid::now_v7();
        let barrier = Uuid::now_v7();
        manifest
            .ssts
            .push(descriptor(node0, SstKind::Nodes, "", 10));
        manifest
            .ssts
            .push(descriptor(node1, SstKind::Nodes, "", 20));
        manifest
            .ssts
            .push(descriptor(base, SstKind::VectorGraph, "doc_vec", 10));
        manifest
            .ssts
            .push(descriptor(delta, SstKind::VectorGraph, "doc_vec", 20));
        manifest.ssts.push(search_barrier_descriptor(
            &SearchLsmState {
                index_name: "doc_vec".into(),
                kind: SearchLsmKind::Vector,
                ..SearchLsmState::default()
            },
            barrier,
            SstLevel::L0,
            format!("sst/level0/{barrier}-barrier.slb"),
            256,
        ));

        let mut state = SearchLsmState {
            index_name: "doc_vec".into(),
            kind: SearchLsmKind::Vector,
            catalog_signature: vector_catalog_signature(&manifest, &manifest.vector_indexes[0]),
            generation_id: Uuid::now_v7(),
            status: SearchLsmStatus::Active,
            next_event_seq: 2,
            base_frontier: Some(1),
            segments: vec![
                vector_segment(
                    base,
                    SearchSegmentRole::Base,
                    SearchEventRange::new(0, 1),
                    SearchStatValue::Absolute(10),
                    10,
                ),
                vector_segment(
                    delta,
                    SearchSegmentRole::Delta,
                    SearchEventRange::new(1, 2),
                    SearchStatValue::Delta(1),
                    1,
                ),
            ],
            proven_empty_event_ranges: Vec::new(),
            coverage: vec![
                SearchCoverage {
                    node_sst_id: node0,
                    node_sst_max_lsn: 10,
                    event_ranges: vec![SearchEventRange::new(0, 1)],
                    disposition: CoverageDisposition::Segment,
                },
                SearchCoverage {
                    node_sst_id: node1,
                    node_sst_max_lsn: 20,
                    event_ranges: vec![SearchEventRange::new(1, 2)],
                    disposition: CoverageDisposition::Segment,
                },
            ],
            compat_barrier_sst_id: Some(barrier),
            equal_lsn_conflict_count: 0,
        };
        let base_descriptor = manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == base)
            .unwrap();
        state.segments[0].content_xxh3 = legacy_base_content_fingerprint(
            base_descriptor,
            state.segments[0].format,
            &state.segments[0].complete_filter_properties,
        );
        manifest.search_lsm.push(state);
        manifest
    }

    fn active_text_manifest() -> Manifest {
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.schema.labels.insert(
            "Doc".into(),
            LabelDef {
                name: "Doc".into(),
                properties: vec![PropertyDef::new("body", DataType::Utf8, false).unwrap()],
            },
        );
        manifest.text_indexes.push(TextIndexDescriptor::new(
            "doc_ft".into(),
            "Doc".into(),
            vec!["body".into()],
        ));
        let node0 = Uuid::now_v7();
        let node1 = Uuid::now_v7();
        let base = Uuid::now_v7();
        let delta = Uuid::now_v7();
        let barrier = Uuid::now_v7();
        manifest
            .ssts
            .push(descriptor(node0, SstKind::Nodes, "", 10));
        manifest
            .ssts
            .push(descriptor(node1, SstKind::Nodes, "", 20));
        manifest
            .ssts
            .push(descriptor(base, SstKind::TextIndex, "doc_ft", 10));
        manifest.ssts.push(delta_descriptor(
            delta,
            SstKind::TextIndex,
            "doc_ft",
            11,
            20,
        ));
        manifest.ssts.push(search_barrier_descriptor(
            &SearchLsmState {
                index_name: "doc_ft".into(),
                kind: SearchLsmKind::Text,
                ..SearchLsmState::default()
            },
            barrier,
            SstLevel::L0,
            format!("sst/level0/{barrier}-barrier.slb"),
            256,
        ));
        let mut state = SearchLsmState {
            index_name: "doc_ft".into(),
            kind: SearchLsmKind::Text,
            catalog_signature: text_lsm_catalog_signature(&manifest, &manifest.text_indexes[0]),
            generation_id: Uuid::now_v7(),
            status: SearchLsmStatus::Active,
            next_event_seq: 2,
            base_frontier: Some(1),
            segments: vec![
                SearchSegmentRef {
                    sst_id: base,
                    role: SearchSegmentRole::Base,
                    format: SearchSegmentFormat::TextV3Base,
                    payload: SearchSegmentPayload::Complete,
                    event_ranges: vec![SearchEventRange::new(0, 1)],
                    min_lsn: 1,
                    max_lsn: 10,
                    mutation_count: 10,
                    live_payload_count: 10,
                    stats: SearchSegmentStats::Text {
                        doc_count: SearchStatValue::Absolute(10),
                        total_len: SearchStatValue::Absolute(100),
                        term_df_violation_count: 0,
                    },
                    ..SearchSegmentRef::default()
                },
                SearchSegmentRef {
                    sst_id: delta,
                    role: SearchSegmentRole::Delta,
                    format: SearchSegmentFormat::TextV4,
                    payload: SearchSegmentPayload::Complete,
                    event_ranges: vec![SearchEventRange::new(1, 2)],
                    min_lsn: 11,
                    max_lsn: 20,
                    mutation_count: 1,
                    live_payload_count: 0,
                    suppress_count: 1,
                    stats: SearchSegmentStats::Text {
                        doc_count: SearchStatValue::Delta(-1),
                        total_len: SearchStatValue::Delta(-10),
                        term_df_violation_count: 0,
                    },
                    ..SearchSegmentRef::default()
                },
            ],
            coverage: vec![
                SearchCoverage {
                    node_sst_id: node0,
                    node_sst_max_lsn: 10,
                    event_ranges: vec![SearchEventRange::new(0, 1)],
                    disposition: CoverageDisposition::Segment,
                },
                SearchCoverage {
                    node_sst_id: node1,
                    node_sst_max_lsn: 20,
                    event_ranges: vec![SearchEventRange::new(1, 2)],
                    disposition: CoverageDisposition::Segment,
                },
            ],
            compat_barrier_sst_id: Some(barrier),
            ..SearchLsmState::default()
        };
        let base_descriptor = manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.id == base)
            .unwrap();
        state.segments[0].content_xxh3 = legacy_base_content_fingerprint(
            base_descriptor,
            state.segments[0].format,
            &state.segments[0].complete_filter_properties,
        );
        manifest.search_lsm.push(state);
        manifest
    }

    fn assert_invariant(manifest: &Manifest, expected: SearchLsmInvariant) {
        let error = validate_search_lsm(manifest).expect_err("fixture must be invalid");
        assert_eq!(error.invariant, expected, "{error}");
    }

    #[test]
    fn legacy_manifest_without_search_lsm_defaults_empty() {
        let manifest: Manifest = serde_json::from_str(LEGACY_MANIFEST_JSON).unwrap();
        assert!(manifest.search_lsm.is_empty());
        assert_eq!(manifest.version, 7);
    }

    #[test]
    fn state_round_trip_and_next_version_preserve_metadata() {
        let manifest = active_vector_manifest();
        let round_trip: Manifest =
            serde_json::from_slice(&serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert_eq!(round_trip.search_lsm, manifest.search_lsm);
        let next = manifest.next_version(Uuid::now_v7());
        assert_eq!(next.search_lsm, manifest.search_lsm);
    }

    #[test]
    fn barrier_round_trip_binds_state_and_rejects_corruption() {
        let manifest = active_vector_manifest();
        let state = &manifest.search_lsm[0];
        let body = encode_search_barrier(state).unwrap();
        assert_eq!(&body[..8], SEARCH_LSM_BARRIER_MAGIC);
        validate_search_barrier(state, &body).unwrap();

        let mut changed = state.clone();
        changed.next_event_seq += 1;
        assert!(validate_search_barrier(&changed, &body).is_err());

        let mut bad_magic = body.to_vec();
        bad_magic[0] ^= 1;
        assert!(decode_search_barrier(&bad_magic)
            .unwrap_err()
            .detail
            .contains("magic"));

        let mut bad_crc = body.to_vec();
        bad_crc[SEARCH_LSM_BARRIER_MAGIC.len()] ^= 1;
        assert!(decode_search_barrier(&bad_crc)
            .unwrap_err()
            .detail
            .contains("checksum"));

        // A valid outer checksum must not make malformed V2 state acceptable.
        let trailer_start = body.len() - BARRIER_TRAILER_LEN;
        let footer_bytes = &body[SEARCH_LSM_BARRIER_MAGIC.len()..trailer_start];
        let mut footer: SearchBarrierFooterV2 = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(footer_bytes)
            .unwrap();
        footer.state_json = b"{not valid JSON".to_vec();
        let malformed_footer = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize(&footer)
            .unwrap();
        let malformed_body = frame_search_barrier_footer(&malformed_footer).unwrap();
        assert!(decode_search_barrier(&malformed_body)
            .unwrap_err()
            .detail
            .contains("state JSON"));
    }

    #[test]
    fn legacy_v1_barrier_with_decodable_state_remains_compatible() {
        // V1 embedded the manifest-shaped state directly in bincode. Tagged
        // enum values were never decodable, but simple historical payloads
        // that bincode can represent must remain readable after the V2 fix.
        let state = SearchLsmState::default();
        let footer = LegacySearchBarrierFooter {
            format_version: LEGACY_BARRIER_FORMAT_VERSION,
            state: state.clone(),
        };
        let encoded = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize(&footer)
            .unwrap();
        let body = frame_search_barrier_footer(&encoded).unwrap();
        assert_eq!(decode_search_barrier(&body).unwrap(), state);
    }

    #[test]
    fn canonical_barrier_descriptor_predicate_is_strict_for_both_kinds() {
        fn barrier(manifest: &Manifest) -> SstDescriptor {
            let barrier_id = manifest.search_lsm[0].compat_barrier_sst_id.unwrap();
            manifest
                .ssts
                .iter()
                .find(|descriptor| descriptor.id == barrier_id)
                .unwrap()
                .clone()
        }

        let vector = barrier(&active_vector_manifest());
        let text = barrier(&active_text_manifest());
        assert!(is_canonical_search_barrier_descriptor(&vector));
        assert!(is_canonical_search_barrier_descriptor(&text));

        let mut wrong_suffix = vector.clone();
        wrong_suffix.path = "sst/level0/not-a-barrier.vg".into();
        assert!(!is_canonical_search_barrier_descriptor(&wrong_suffix));

        let mut nil_id = vector.clone();
        nil_id.id = Uuid::nil();
        assert!(!is_canonical_search_barrier_descriptor(&nil_id));

        let mut too_small = vector.clone();
        too_small.size_bytes = (SEARCH_LSM_BARRIER_MAGIC.len() + BARRIER_TRAILER_LEN) as u64;
        assert!(!is_canonical_search_barrier_descriptor(&too_small));

        let mut too_large = vector.clone();
        too_large.size_bytes = MAX_BARRIER_BODY_BYTES as u64 + 1;
        assert!(!is_canonical_search_barrier_descriptor(&too_large));

        let mut non_zero = vector.clone();
        non_zero.max_lsn = 1;
        assert!(!is_canonical_search_barrier_descriptor(&non_zero));

        let mut wrong_stats = text;
        wrong_stats.kind_specific = KindSpecificStats::TextIndex {
            doc_count: 0,
            term_count: 1,
            total_len: 0,
        };
        assert!(!is_canonical_search_barrier_descriptor(&wrong_stats));
    }

    #[test]
    fn wraps_a_fresh_legacy_base_and_selection_is_downgrade_safe() {
        let mut manifest = active_vector_manifest();
        let base_id = manifest.search_lsm[0].segments[0].sst_id;
        let barrier_id = manifest.search_lsm[0].compat_barrier_sst_id.unwrap();
        let delta_id = manifest.search_lsm[0].segments[1].sst_id;
        let second_node = manifest.search_lsm[0].coverage[1].node_sst_id;
        manifest.search_lsm.clear();
        manifest.ssts.retain(|descriptor| {
            descriptor.id != barrier_id && descriptor.id != delta_id && descriptor.id != second_node
        });

        let generation = Uuid::now_v7();
        let barrier = Uuid::now_v7();
        let mut outrun = manifest.clone();
        outrun
            .ssts
            .push(descriptor(Uuid::now_v7(), SstKind::Nodes, "", 11));
        let error = wrap_legacy_search_base(
            &outrun,
            SearchLsmKind::Vector,
            "doc_vec",
            base_id,
            generation,
            barrier,
        )
        .unwrap_err();
        assert_eq!(
            error.invariant,
            SearchLsmInvariant::NodeCoverage,
            "a concurrently newer Nodes SST must cancel activation"
        );

        let state = wrap_legacy_search_base(
            &manifest,
            SearchLsmKind::Vector,
            "doc_vec",
            base_id,
            generation,
            barrier,
        )
        .unwrap();
        let body = encode_search_barrier(&state).unwrap();
        manifest.ssts.push(search_barrier_descriptor(
            &state,
            barrier,
            SstLevel::L0,
            "sst/level0/doc-vec.slb".into(),
            body.len() as u64,
        ));
        manifest.search_lsm.push(state.clone());
        validate_search_lsm(&manifest).unwrap();
        assert!(matches!(
            select_search_read_plan(&manifest, SearchLsmKind::Vector, "doc_vec"),
            SearchReadPlan::ActiveLegacyBase {
                base_sst_id,
                barrier_sst_id,
                ..
            } if base_sst_id == base_id && barrier_sst_id == barrier
        ));

        // Simulate an old writer dropping the unknown top-level field while
        // preserving both physical descriptors: two bodies cannot be mistaken
        // for one full generation by either old or new selection.
        manifest.search_lsm.clear();
        assert_eq!(
            select_search_read_plan(&manifest, SearchLsmKind::Vector, "doc_vec"),
            SearchReadPlan::FlatFallback(SearchReadFallback::AmbiguousPhysicalBodies)
        );

        // Without the barrier there is exactly one ordinary legacy body and
        // the unchanged legacy freshness path may serve it.
        manifest.ssts.retain(|descriptor| descriptor.id != barrier);
        assert_eq!(
            select_search_read_plan(&manifest, SearchLsmKind::Vector, "doc_vec"),
            SearchReadPlan::Legacy { sst_id: base_id }
        );
    }

    #[test]
    fn partial_barrier_publish_and_building_state_always_fall_back() {
        let mut missing_body = active_vector_manifest();
        let barrier = missing_body.search_lsm[0].compat_barrier_sst_id.unwrap();
        missing_body
            .ssts
            .retain(|descriptor| descriptor.id != barrier);
        assert_eq!(
            select_search_read_plan(&missing_body, SearchLsmKind::Vector, "doc_vec"),
            SearchReadPlan::FlatFallback(SearchReadFallback::InvalidGeneration)
        );

        let mut building = active_vector_manifest();
        building.search_lsm[0].status = SearchLsmStatus::Building;
        assert_eq!(
            select_search_read_plan(&building, SearchLsmKind::Vector, "doc_vec"),
            SearchReadPlan::FlatFallback(SearchReadFallback::BuildingGeneration)
        );
    }

    #[test]
    fn native_segment_selection_is_explicit_and_rejects_unsafe_migrations() {
        let vector = active_vector_manifest();
        assert!(matches!(
            select_search_read_plan(&vector, SearchLsmKind::Vector, "doc_vec"),
            SearchReadPlan::ActiveSegments {
                state,
                barrier_sst_id,
            } if state.segments.len() == 2
                && Some(barrier_sst_id) == state.compat_barrier_sst_id
        ));

        // A V3 text base cannot be merged with FT4 delta scores until it can
        // consume the same reconstructed global statistics as FT4.
        let text = active_text_manifest();
        assert_eq!(
            select_search_read_plan(&text, SearchLsmKind::Text, "doc_ft"),
            SearchReadPlan::FlatFallback(SearchReadFallback::UnsupportedSegmentSet)
        );

        let mut shadow = active_vector_manifest();
        shadow.search_lsm[0].segments[1].payload = SearchSegmentPayload::ShadowOnly;
        assert_eq!(
            select_search_read_plan(&shadow, SearchLsmKind::Vector, "doc_vec"),
            SearchReadPlan::FlatFallback(SearchReadFallback::UnsupportedSegmentSet)
        );
    }

    #[test]
    fn signatures_are_canonical_and_schema_sensitive() {
        let left = active_vector_manifest();
        assert_eq!(
            vector_native_filter_properties(&left, &left.vector_indexes[0]),
            vec!["tenant".to_owned()],
            "a unique identity key is not an implicit vector-filter obligation"
        );
        let mut right = left.clone();
        right.schema = vector_schema(true);
        assert_eq!(
            vector_catalog_signature(&left, &left.vector_indexes[0]),
            vector_catalog_signature(&right, &right.vector_indexes[0]),
            "schema declaration order must not affect the signature"
        );
        assert_eq!(
            legacy_vector_catalog_signature(&left, &left.vector_indexes[0]),
            legacy_vector_catalog_signature(&right, &right.vector_indexes[0]),
            "the migration signature must reproduce the old sorted schema encoding"
        );
        right
            .schema
            .labels
            .get_mut("Doc")
            .unwrap()
            .properties
            .iter_mut()
            .find(|property| property.name == "tenant")
            .unwrap()
            .indexed = false;
        assert_ne!(
            vector_catalog_signature(&left, &left.vector_indexes[0]),
            vector_catalog_signature(&right, &right.vector_indexes[0]),
        );
        assert_ne!(
            legacy_vector_catalog_signature(&left, &left.vector_indexes[0]),
            legacy_vector_catalog_signature(&right, &right.vector_indexes[0]),
        );

        let ordered =
            TextIndexDescriptor::new("ft".into(), "Doc".into(), vec!["a".into(), "b".into()]);
        let malformed_order = TextIndexDescriptor {
            name: "ft".into(),
            label: "Doc".into(),
            properties: vec!["b".into(), "a".into(), "a".into()],
        };
        assert_eq!(
            text_catalog_signature(&ordered),
            text_catalog_signature(&malformed_order)
        );
        assert_eq!(
            legacy_text_catalog_signature(&ordered),
            serde_json::to_string(&ordered).unwrap()
        );
        assert_ne!(
            legacy_text_catalog_signature(&ordered),
            legacy_text_catalog_signature(&malformed_order),
            "legacy marker matching intentionally reproduces its order-sensitive encoding"
        );

        let mut filtered_text = active_text_manifest();
        filtered_text
            .schema
            .labels
            .get_mut("Doc")
            .unwrap()
            .properties
            .push(
                PropertyDef::new("active", DataType::Bool, true)
                    .unwrap()
                    .with_indexed(true),
            );
        assert_eq!(
            text_native_filter_properties(&filtered_text, &filtered_text.text_indexes[0]),
            vec!["active".to_owned()]
        );
        let with_filter =
            text_lsm_catalog_signature(&filtered_text, &filtered_text.text_indexes[0]);
        filtered_text
            .schema
            .labels
            .get_mut("Doc")
            .unwrap()
            .properties
            .last_mut()
            .unwrap()
            .indexed = false;
        assert_ne!(
            with_filter,
            text_lsm_catalog_signature(&filtered_text, &filtered_text.text_indexes[0]),
            "FT4 catalog signatures must bind native-filter obligations"
        );
        assert_eq!(
            text_catalog_signature(&filtered_text.text_indexes[0]),
            text_catalog_signature(&active_text_manifest().text_indexes[0]),
            "the legacy/no-filter signature stays manifest-independent"
        );
    }

    #[test]
    fn valid_vector_text_and_proven_empty_states_pass() {
        let vector = active_vector_manifest();
        validate_search_lsm(&vector).unwrap();
        validate_search_lsm(&active_text_manifest()).unwrap();

        let mut empty = vector;
        let state = &mut empty.search_lsm[0];
        let removed = state.segments.pop().unwrap().sst_id;
        empty.ssts.retain(|descriptor| descriptor.id != removed);
        state.proven_empty_event_ranges = vec![SearchEventRange::new(1, 2)];
        state.coverage[1].disposition = CoverageDisposition::ProvenEmpty {
            classifier_version: 1,
            before_after_digest: 42,
        };
        validate_search_lsm(&empty).unwrap();
    }

    #[test]
    fn building_state_may_be_incomplete_but_catalog_bound() {
        let mut manifest = active_vector_manifest();
        let signature = manifest.search_lsm[0].catalog_signature.clone();
        let generation = manifest.search_lsm[0].generation_id;
        manifest.search_lsm[0] = SearchLsmState {
            index_name: "doc_vec".into(),
            kind: SearchLsmKind::Vector,
            catalog_signature: signature,
            generation_id: generation,
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        };
        validate_search_lsm(&manifest).unwrap();
    }

    #[test]
    fn invariant_1_rejects_missing_ambiguous_or_changed_catalog() {
        let mut changed = active_vector_manifest();
        changed.search_lsm[0].catalog_signature.push('x');
        assert_invariant(&changed, SearchLsmInvariant::Catalog);

        let mut missing = active_vector_manifest();
        missing.vector_indexes.clear();
        assert_invariant(&missing, SearchLsmInvariant::Catalog);

        let mut ambiguous = active_vector_manifest();
        ambiguous
            .vector_indexes
            .push(ambiguous.vector_indexes[0].clone());
        assert_invariant(&ambiguous, SearchLsmInvariant::Catalog);
    }

    #[test]
    fn invariant_2_rejects_duplicate_state_identity() {
        let mut manifest = active_vector_manifest();
        manifest.search_lsm.push(manifest.search_lsm[0].clone());
        assert_invariant(&manifest, SearchLsmInvariant::UniqueState);
    }

    #[test]
    fn invariant_3_rejects_missing_or_reused_barrier() {
        let mut missing = active_vector_manifest();
        missing.search_lsm[0].compat_barrier_sst_id = None;
        assert_invariant(&missing, SearchLsmInvariant::CompatibilityBarrier);

        let mut reused = active_vector_manifest();
        let data_id = reused.search_lsm[0].segments[0].sst_id;
        reused.search_lsm[0].compat_barrier_sst_id = Some(data_id);
        assert_invariant(&reused, SearchLsmInvariant::CompatibilityBarrier);

        let mut non_canonical = active_vector_manifest();
        let barrier_id = non_canonical.search_lsm[0].compat_barrier_sst_id.unwrap();
        non_canonical
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.id == barrier_id)
            .unwrap()
            .row_count = 1;
        assert_invariant(&non_canonical, SearchLsmInvariant::CompatibilityBarrier);

        let mut oversized = active_vector_manifest();
        let barrier_id = oversized.search_lsm[0].compat_barrier_sst_id.unwrap();
        oversized
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.id == barrier_id)
            .unwrap()
            .size_bytes = MAX_BARRIER_BODY_BYTES as u64 + 1;
        assert_invariant(&oversized, SearchLsmInvariant::CompatibilityBarrier);
    }

    #[test]
    fn invariant_4_rejects_missing_unlisted_wrong_format_and_bad_counts() {
        let mut missing = active_vector_manifest();
        let id = missing.search_lsm[0].segments[1].sst_id;
        missing.ssts.retain(|descriptor| descriptor.id != id);
        assert_invariant(&missing, SearchLsmInvariant::PhysicalSegments);

        let mut unlisted = active_vector_manifest();
        unlisted.ssts.push(descriptor(
            Uuid::now_v7(),
            SstKind::VectorGraph,
            "doc_vec",
            30,
        ));
        assert_invariant(&unlisted, SearchLsmInvariant::PhysicalSegments);

        let mut wrong_format = active_vector_manifest();
        wrong_format.search_lsm[0].segments[1].format = SearchSegmentFormat::TextV4;
        assert_invariant(&wrong_format, SearchLsmInvariant::PhysicalSegments);

        let mut bad_counts = active_vector_manifest();
        bad_counts.search_lsm[0].segments[1].suppress_count = 1;
        assert_invariant(&bad_counts, SearchLsmInvariant::PhysicalSegments);

        let mut filters = active_vector_manifest();
        filters.search_lsm[0].segments[0].complete_filter_properties =
            vec!["tenant".into(), "active".into()];
        assert_invariant(&filters, SearchLsmInvariant::PhysicalSegments);
    }

    #[test]
    fn invariant_5_rejects_overlap_order_and_base_frontier_drift() {
        let mut overlap = active_vector_manifest();
        overlap.search_lsm[0].segments[1].event_ranges = vec![SearchEventRange::new(0, 2)];
        assert_invariant(&overlap, SearchLsmInvariant::SegmentOrdering);

        let mut order = active_vector_manifest();
        order.search_lsm[0].segments.swap(0, 1);
        assert_invariant(&order, SearchLsmInvariant::SegmentOrdering);

        let mut frontier = active_vector_manifest();
        frontier.search_lsm[0].base_frontier = Some(2);
        assert_invariant(&frontier, SearchLsmInvariant::SegmentOrdering);
    }

    #[test]
    fn invariant_6_rejects_missing_duplicate_stale_or_foreign_coverage() {
        let mut missing = active_vector_manifest();
        missing.search_lsm[0].coverage.pop();
        assert_invariant(&missing, SearchLsmInvariant::NodeCoverage);

        let mut duplicate = active_vector_manifest();
        let duplicate_coverage = duplicate.search_lsm[0].coverage[0].clone();
        duplicate.search_lsm[0].coverage.push(duplicate_coverage);
        assert_invariant(&duplicate, SearchLsmInvariant::NodeCoverage);

        let mut stale_lsn = active_vector_manifest();
        stale_lsn.search_lsm[0].coverage[0].node_sst_max_lsn += 1;
        assert_invariant(&stale_lsn, SearchLsmInvariant::NodeCoverage);

        let mut foreign = active_vector_manifest();
        foreign.search_lsm[0].coverage[0].node_sst_id = Uuid::now_v7();
        assert_invariant(&foreign, SearchLsmInvariant::NodeCoverage);
    }

    #[test]
    fn invariant_7_rejects_event_gaps_overlap_and_disposition_mismatch() {
        let mut gap = active_vector_manifest();
        gap.search_lsm[0].segments.pop();
        let delta_id = gap
            .ssts
            .iter()
            .find(|descriptor| {
                descriptor.kind == SstKind::VectorGraph
                    && Some(descriptor.id) != gap.search_lsm[0].compat_barrier_sst_id
                    && descriptor.id != gap.search_lsm[0].segments[0].sst_id
            })
            .unwrap()
            .id;
        gap.ssts.retain(|descriptor| descriptor.id != delta_id);
        assert_invariant(&gap, SearchLsmInvariant::EventCoverage);

        let mut overlap = active_vector_manifest();
        overlap.search_lsm[0].proven_empty_event_ranges = vec![SearchEventRange::new(1, 2)];
        assert_invariant(&overlap, SearchLsmInvariant::EventCoverage);

        let mut disposition = active_vector_manifest();
        disposition.search_lsm[0].coverage[1].disposition = CoverageDisposition::ProvenEmpty {
            classifier_version: 1,
            before_after_digest: 1,
        };
        assert_invariant(&disposition, SearchLsmInvariant::EventCoverage);

        let mut classifier = active_vector_manifest();
        let state = &mut classifier.search_lsm[0];
        let removed = state.segments.pop().unwrap().sst_id;
        classifier
            .ssts
            .retain(|descriptor| descriptor.id != removed);
        state.proven_empty_event_ranges = vec![SearchEventRange::new(1, 2)];
        state.coverage[1].disposition = CoverageDisposition::ProvenEmpty {
            classifier_version: 0,
            before_after_digest: 1,
        };
        assert_invariant(&classifier, SearchLsmInvariant::EventCoverage);
    }

    #[test]
    fn invariant_8_rejects_wrong_stat_mode_negative_totals_and_bad_term_df() {
        let mut mode = active_vector_manifest();
        mode.search_lsm[0].segments[1].stats = SearchSegmentStats::Vector {
            live_count: SearchStatValue::Absolute(1),
        };
        assert_invariant(&mode, SearchLsmInvariant::Statistics);

        let mut negative = active_vector_manifest();
        negative.search_lsm[0].segments[1].stats = SearchSegmentStats::Vector {
            live_count: SearchStatValue::Delta(-11),
        };
        assert_invariant(&negative, SearchLsmInvariant::Statistics);

        let mut text = active_text_manifest();
        match &mut text.search_lsm[0].segments[1].stats {
            SearchSegmentStats::Text {
                term_df_violation_count,
                ..
            } => *term_df_violation_count = 1,
            _ => unreachable!(),
        }
        assert_invariant(&text, SearchLsmInvariant::Statistics);

        let mut impossible_len = active_text_manifest();
        impossible_len.search_lsm[0].segments[1].stats = SearchSegmentStats::Text {
            doc_count: SearchStatValue::Delta(-10),
            total_len: SearchStatValue::Delta(-10),
            term_df_violation_count: 0,
        };
        assert_invariant(&impossible_len, SearchLsmInvariant::Statistics);
    }

    #[test]
    fn invariant_10_rejects_local_and_cross_segment_equal_lsn_conflicts() {
        let mut local = active_vector_manifest();
        local.search_lsm[0].segments[1].equal_lsn_conflict_count = 1;
        assert_invariant(&local, SearchLsmInvariant::VersionTie);

        let mut cross = active_vector_manifest();
        cross.search_lsm[0].equal_lsn_conflict_count = 1;
        assert_invariant(&cross, SearchLsmInvariant::VersionTie);
    }
}
