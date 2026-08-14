//! Incremental object-native vector delta (`NAMIVG06`).
//!
//! V6 delta objects keep the common [`NAMISV01`](crate::sst::search_delta)
//! winner table next to independently compressed exact-vector pages and
//! complete adaptive native-filter postings. The first production mode is an
//! exact flat delta: flush-sized segments avoid ANN build amplification,
//! remain range-readable, and can later be compacted into clustered V6 bases.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::io::{Cursor, Read, Seek, Write};
use std::ops::Range;
use std::sync::Arc;

use bincode::Options;
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};
use crate::manifest::{VectorIndexDescriptor, VectorMetric};
use crate::search_lsm::{
    SearchEventRange, SearchLsmKind, SearchLsmState, SearchSegmentFormat, SearchSegmentPayload,
    SearchSegmentRef, SearchSegmentRole, SearchSegmentStats, SearchStatValue,
};
use crate::search_workspace::shared_search_workspace;
use crate::sst::search_delta::{
    search_suppress_fingerprint, SearchFilterValue, SearchSegmentWireBinding,
    SearchVersionOperation, SearchVersionRangeSource, SearchVersionRecord,
    SearchVersionTableReader, SearchVersionTableRef, SearchVersionTableWriter,
};

#[path = "v6_external.rs"]
mod external;

pub use external::{
    VectorV6ExternalArtifact, VectorV6ExternalBuildConfig, VectorV6ExternalBuildMetrics,
    VectorV6ExternalBuilder,
};

pub const MAGIC_V6: &[u8; 8] = b"NAMIVG06";
const TRAILER_MAGIC: &[u8; 8] = b"NVG6END!";
const TRAILER_LEN: usize = 8 + 8 + 4;
const FORMAT_VERSION: u16 = 6;
const FOOTER_VERSION: u16 = 1;
const VECTOR_PAGE_HEADER_LEN: usize = 8;
const VECTOR_ROW_PREFIX_LEN: usize = 16 + 8 + 8;
const MAX_FOOTER_BYTES: u64 = 128 * 1024 * 1024;
const MAX_COMPRESSED_BLOCK_BYTES: u64 = 512 * 1024 * 1024;
const MAX_RAW_BLOCK_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DIM: u32 = 1_048_576;
const CONTENT_DOMAIN: &[u8] = b"NamiDB/VectorV6Content/v1";
const PAYLOAD_FINGERPRINT_DOMAIN: &[u8] = b"NamiDB/VectorV6Payload/v1";
const INDEX_BUILD_MEMORY_ENV: &str = "NAMIDB_INDEX_BUILD_MEMORY_BYTES";
const DEFAULT_INDEX_BUILD_MEMORY_BYTES: usize = 64 * 1024 * 1024;

/// One complete live vector after-image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorV6Payload {
    pub vector: Vec<f32>,
    /// Values for properties advertised as complete by the segment. A missing
    /// property is an authoritative empty posting for this row.
    #[serde(default)]
    pub filters: BTreeMap<String, SearchFilterValue>,
}

/// One exactly classified before/after vector mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorV6Mutation {
    pub node_id: [u8; 16],
    pub lsn: u64,
    pub before: Option<VectorV6Payload>,
    pub after: Option<VectorV6Payload>,
}

/// Manifest-independent identity/coverage supplied by the flush transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorV6BuildContext {
    pub sst_id: Uuid,
    pub event_ranges: Vec<SearchEventRange>,
    pub complete_filter_properties: Vec<String>,
}

/// Deterministic exact-delta build controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorV6BuildOptions {
    pub rows_per_page: usize,
    pub compression_level: i32,
}

impl Default for VectorV6BuildOptions {
    fn default() -> Self {
        Self {
            rows_per_page: 256,
            compression_level: 3,
        }
    }
}

/// Metadata returned next to the finished object/spool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorV6BuildOutput {
    pub segment: SearchSegmentRef,
    pub object_len: u64,
    pub page_count: u32,
    pub version_table: SearchVersionTableRef,
}

/// In-memory convenience artifact. Production flushes should call
/// [`write_delta_v6`] with a file-backed spool.
#[derive(Debug, Clone)]
pub struct VectorV6Artifact {
    pub body: Bytes,
    pub output: VectorV6BuildOutput,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorV6Hit {
    pub node_id: [u8; 16],
    pub lsn: u64,
    pub payload_fingerprint: u64,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorV6SearchResult {
    pub hits: Vec<VectorV6Hit>,
    pub applied_filter_groups: usize,
    pub scanned_pages: usize,
    pub eligible_rows_seen: usize,
    pub peak_live_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct BlockRef {
    offset: u64,
    len: u32,
    raw_len: u32,
    compressed_crc32: u32,
    raw_xxh3: u64,
}

impl BlockRef {
    fn range(&self) -> Result<Range<u64>> {
        let end = self
            .offset
            .checked_add(u64::from(self.len))
            .ok_or_else(|| Error::invariant("vector v6 block range overflows"))?;
        Ok(self.offset..end)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct VectorPageRef {
    first_ordinal: u64,
    row_count: u32,
    first_node_id: [u8; 16],
    last_node_id: [u8; 16],
    min_lsn: u64,
    max_lsn: u64,
    wire: BlockRef,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FilterBlockRef {
    property: String,
    row_count: u64,
    values: Vec<FilterValueRef>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum FilterPostingEncoding {
    SparseDeltaVarint,
    DenseBitmap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FilterValueRef {
    value: SearchFilterValue,
    cardinality: u64,
    encoding: FilterPostingEncoding,
    wire: BlockRef,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum VectorV6Mode {
    FlatExact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Footer {
    footer_version: u16,
    mode: VectorV6Mode,
    binding: SearchSegmentWireBinding,
    dim: u32,
    metric: VectorMetric,
    live_count_delta: i64,
    pages: Vec<VectorPageRef>,
    filters: Vec<FilterBlockRef>,
}

#[derive(Debug, Serialize)]
struct ContentDigestMaterial<'a> {
    domain: &'a [u8],
    format_version: u16,
    mode: VectorV6Mode,
    version_table: &'a SearchVersionTableRef,
    dim: u32,
    metric: VectorMetric,
    live_count_delta: i64,
    pages: &'a [VectorPageRef],
    filters: &'a [FilterBlockRef],
}

#[derive(Debug)]
struct VectorPageInput<'a> {
    ordinal: u64,
    node_id: [u8; 16],
    lsn: u64,
    payload_fingerprint: u64,
    vector: &'a [f32],
}

#[derive(Debug)]
struct VectorPageRow {
    ordinal: u64,
    node_id: [u8; 16],
    lsn: u64,
    payload_fingerprint: u64,
    vector: Vec<f32>,
}

#[derive(Debug)]
enum QueryFilterMask {
    Sparse(Vec<u64>),
    Dense(Vec<u8>),
}

impl QueryFilterMask {
    fn empty() -> Self {
        Self::Sparse(Vec::new())
    }

    fn contains(&self, ordinal: u64) -> bool {
        match self {
            Self::Sparse(ordinals) => ordinals.binary_search(&ordinal).is_ok(),
            Self::Dense(bitmap) => dense_filter_contains(bitmap, ordinal),
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            Self::Sparse(ordinals) => ordinals
                .capacity()
                .saturating_mul(std::mem::size_of::<u64>()),
            Self::Dense(bitmap) => bitmap.capacity(),
        }
    }

    fn union(self, other: Self, row_count: u64) -> Result<Self> {
        match (self, other) {
            (Self::Sparse(left), Self::Sparse(right)) => {
                let mut merged = Vec::with_capacity(left.len().saturating_add(right.len()));
                let (mut left_index, mut right_index) = (0usize, 0usize);
                while left_index < left.len() || right_index < right.len() {
                    let next = match (left.get(left_index), right.get(right_index)) {
                        (Some(left), Some(right)) if left < right => {
                            left_index += 1;
                            *left
                        }
                        (Some(left), Some(right)) if right < left => {
                            right_index += 1;
                            *right
                        }
                        (Some(value), Some(_)) => {
                            left_index += 1;
                            right_index += 1;
                            *value
                        }
                        (Some(value), None) => {
                            left_index += 1;
                            *value
                        }
                        (None, Some(value)) => {
                            right_index += 1;
                            *value
                        }
                        (None, None) => break,
                    };
                    merged.push(next);
                }
                sparse_or_dense(merged, row_count)
            }
            (Self::Dense(mut dense), Self::Sparse(sparse))
            | (Self::Sparse(sparse), Self::Dense(mut dense)) => {
                for ordinal in sparse {
                    set_dense_filter_bit(&mut dense, ordinal, row_count)?;
                }
                Ok(Self::Dense(dense))
            }
            (Self::Dense(mut left), Self::Dense(right)) => {
                if left.len() != right.len() {
                    return Err(Error::invariant(
                        "vector v6 dense query filter lengths diverged",
                    ));
                }
                for (target, source) in left.iter_mut().zip(right) {
                    *target |= source;
                }
                Ok(Self::Dense(left))
            }
        }
    }

    fn intersect(self, other: Self, row_count: u64) -> Result<Self> {
        match (self, other) {
            (Self::Sparse(left), Self::Sparse(right)) => {
                let mut intersection = Vec::with_capacity(left.len().min(right.len()));
                let (mut left_index, mut right_index) = (0usize, 0usize);
                while let (Some(left), Some(right)) = (left.get(left_index), right.get(right_index))
                {
                    match left.cmp(right) {
                        Ordering::Less => left_index += 1,
                        Ordering::Greater => right_index += 1,
                        Ordering::Equal => {
                            intersection.push(*left);
                            left_index += 1;
                            right_index += 1;
                        }
                    }
                }
                Ok(Self::Sparse(intersection))
            }
            (Self::Dense(dense), Self::Sparse(mut sparse))
            | (Self::Sparse(mut sparse), Self::Dense(dense)) => {
                sparse.retain(|ordinal| dense_filter_contains(&dense, *ordinal));
                Ok(Self::Sparse(sparse))
            }
            (Self::Dense(mut left), Self::Dense(right)) => {
                if left.len() != right.len() {
                    return Err(Error::invariant(
                        "vector v6 dense query filter lengths diverged",
                    ));
                }
                for (target, source) in left.iter_mut().zip(right) {
                    *target &= source;
                }
                maybe_sparsify_dense(left, row_count)
            }
        }
    }
}

/// Metadata-only/range-readable V6 reader.
pub struct VectorV6Reader {
    source: Arc<dyn SearchVersionRangeSource>,
    file_len: u64,
    footer_offset: u64,
    footer: Footer,
    version_reader: SearchVersionTableReader,
}

impl std::fmt::Debug for VectorV6Reader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorV6Reader")
            .field("file_len", &self.file_len)
            .field("footer_offset", &self.footer_offset)
            .field("dim", &self.footer.dim)
            .field("metric", &self.footer.metric)
            .field("pages", &self.footer.pages.len())
            .field("filters", &self.footer.filters.len())
            .field("segment", &self.footer.binding.segment.sst_id)
            .finish()
    }
}

/// Canonical live-payload fingerprint shared by builders and the future
/// authoritative winner oracle.
pub fn vector_v6_payload_fingerprint(payload: &VectorV6Payload) -> Result<u64> {
    validate_filter_map_keys(&payload.filters)?;
    let encoded = serialize_bounded(payload, MAX_RAW_BLOCK_BYTES, "vector payload fingerprint")?;
    let mut material = Vec::with_capacity(PAYLOAD_FINGERPRINT_DOMAIN.len() + encoded.len());
    material.extend_from_slice(PAYLOAD_FINGERPRINT_DOMAIN);
    material.extend_from_slice(&encoded);
    Ok(non_zero_xxh3(&material))
}

/// Build a V6 delta in memory. Returns `None` when every supplied mutation is
/// an exact search no-op and the caller should commit `ProvenEmpty` coverage.
pub fn build_delta_v6(
    state: &SearchLsmState,
    descriptor: &VectorIndexDescriptor,
    context: VectorV6BuildContext,
    mutations: Vec<VectorV6Mutation>,
    options: VectorV6BuildOptions,
) -> Result<Option<VectorV6Artifact>> {
    let cursor = Cursor::new(Vec::new());
    let Some((cursor, output)) =
        write_delta_v6(cursor, state, descriptor, context, mutations, options)?
    else {
        return Ok(None);
    };
    Ok(Some(VectorV6Artifact {
        body: Bytes::from(cursor.into_inner()),
        output,
    }))
}

/// Stream a deterministic V6 exact-delta object to any seekable spool.
///
/// Only a single vector page and sparse filter ordinals for the explicitly
/// bounded flush batch are materialized. The input mutation batch is sorted
/// in place, and vectors are borrowed rather than cloned into page rows.
pub fn write_delta_v6<W: Write + Seek>(
    mut writer: W,
    state: &SearchLsmState,
    descriptor: &VectorIndexDescriptor,
    mut context: VectorV6BuildContext,
    mut mutations: Vec<VectorV6Mutation>,
    options: VectorV6BuildOptions,
) -> Result<Option<(W, VectorV6BuildOutput)>> {
    validate_build_configuration(&mut writer, state, descriptor, &mut context, options)?;
    mutations.sort_by_key(|mutation| mutation.node_id);
    if mutations
        .windows(2)
        .any(|pair| pair[0].node_id == pair[1].node_id)
    {
        return Err(Error::invariant(
            "vector v6 delta contains duplicate NodeIds; reconcile before building",
        ));
    }
    for mutation in &mutations {
        if mutation.lsn == 0 {
            return Err(Error::invariant(
                "vector v6 mutation uses reserved LSN zero",
            ));
        }
        if let Some(before) = &mutation.before {
            validate_payload(before, descriptor, &context.complete_filter_properties)?;
        }
        if let Some(after) = &mutation.after {
            validate_payload(after, descriptor, &context.complete_filter_properties)?;
        }
    }
    mutations.retain(|mutation| mutation.before != mutation.after);
    if mutations.is_empty() {
        return Ok(None);
    }
    let build_memory_budget = enforce_build_memory_budget(&mutations, descriptor, options)?;
    let payload_fingerprints = mutations
        .iter()
        .map(|mutation| match &mutation.after {
            Some(after) => vector_v6_payload_fingerprint(after),
            None => Ok(search_suppress_fingerprint()),
        })
        .collect::<Result<Vec<_>>>()?;

    writer.write_all(MAGIC_V6)?;
    let mut version_writer = SearchVersionTableWriter::new(writer)?;
    let mut live_ordinal = 0u64;
    for (mutation, payload_fingerprint) in mutations.iter().zip(&payload_fingerprints) {
        let record = if mutation.after.is_some() {
            let record = SearchVersionRecord::live(
                mutation.node_id,
                mutation.lsn,
                *payload_fingerprint,
                live_ordinal,
            );
            live_ordinal = live_ordinal
                .checked_add(1)
                .ok_or_else(|| Error::invariant("vector v6 live ordinal overflows"))?;
            record
        } else {
            SearchVersionRecord::suppress(mutation.node_id, mutation.lsn, *payload_fingerprint)
        };
        version_writer.push(record)?;
    }
    let (mut writer, version_table) = version_writer.finish()?;

    let mut pages = Vec::new();
    let mut page_rows = Vec::with_capacity(options.rows_per_page);
    let mut ordinal = 0u64;
    for (mutation, payload_fingerprint) in mutations.iter().zip(&payload_fingerprints) {
        let Some(after) = &mutation.after else {
            continue;
        };
        page_rows.push(VectorPageInput {
            ordinal,
            node_id: mutation.node_id,
            lsn: mutation.lsn,
            payload_fingerprint: *payload_fingerprint,
            vector: &after.vector,
        });
        ordinal += 1;
        if page_rows.len() == options.rows_per_page {
            pages.push(write_vector_page(
                &mut writer,
                &page_rows,
                descriptor.dim,
                options.compression_level,
            )?);
            page_rows.clear();
        }
    }
    if !page_rows.is_empty() {
        pages.push(write_vector_page(
            &mut writer,
            &page_rows,
            descriptor.dim,
            options.compression_level,
        )?);
    }
    if ordinal != live_ordinal {
        return Err(Error::invariant(
            "vector v6 live payload/version ordinals diverged",
        ));
    }

    let mut filters = Vec::with_capacity(context.complete_filter_properties.len());
    for property in &context.complete_filter_properties {
        let postings = build_filter_postings(property, live_ordinal, &mutations)?;
        let mut values = Vec::with_capacity(postings.len());
        for (value, ordinals) in postings {
            let (encoding, raw) = encode_filter_posting(&ordinals, live_ordinal)?;
            values.push(FilterValueRef {
                value,
                cardinality: ordinals.len() as u64,
                encoding,
                wire: write_compressed_block(
                    &mut writer,
                    &raw,
                    options.compression_level,
                    "vector filter posting block",
                )?,
            });
        }
        filters.push(FilterBlockRef {
            property: property.clone(),
            row_count: live_ordinal,
            values,
        });
    }

    let live_count_delta = mutations.iter().try_fold(0i64, |total, mutation| {
        let after = if mutation.after.is_some() { 1i64 } else { 0 };
        let before = if mutation.before.is_some() { 1i64 } else { 0 };
        let delta = after - before;
        total
            .checked_add(delta)
            .ok_or_else(|| Error::invariant("vector v6 live-count delta overflows"))
    })?;
    let content_xxh3 = content_digest(
        &version_table,
        descriptor.dim,
        descriptor.metric,
        live_count_delta,
        &pages,
        &filters,
    )?;
    let min_lsn = mutations
        .iter()
        .map(|mutation| mutation.lsn)
        .min()
        .ok_or_else(|| Error::invariant("vector v6 effective delta unexpectedly empty"))?;
    let max_lsn = mutations
        .iter()
        .map(|mutation| mutation.lsn)
        .max()
        .ok_or_else(|| Error::invariant("vector v6 effective delta unexpectedly empty"))?;
    let segment = SearchSegmentRef {
        sst_id: context.sst_id,
        role: SearchSegmentRole::Delta,
        format: SearchSegmentFormat::VectorV6,
        payload: SearchSegmentPayload::Complete,
        event_ranges: context.event_ranges,
        min_lsn,
        max_lsn,
        mutation_count: mutations.len() as u64,
        live_payload_count: live_ordinal,
        suppress_count: mutations.len() as u64 - live_ordinal,
        content_xxh3,
        complete_filter_properties: context.complete_filter_properties,
        stats: SearchSegmentStats::Vector {
            live_count: SearchStatValue::Delta(live_count_delta),
        },
        equal_lsn_conflict_count: 0,
    };
    let binding = SearchSegmentWireBinding::new(state, &segment, version_table.clone())?;
    let footer = Footer {
        footer_version: FOOTER_VERSION,
        mode: VectorV6Mode::FlatExact,
        binding,
        dim: descriptor.dim,
        metric: descriptor.metric,
        live_count_delta,
        pages,
        filters,
    };
    let footer_wire_len = bincode_options(MAX_FOOTER_BYTES)
        .serialized_size(&footer)
        .map_err(|error| Error::precondition(format!("vector v6 footer size failed: {error}")))?;
    if footer_wire_len > (build_memory_budget / 4).max(4 * 1024) as u64 {
        return Err(Error::precondition(format!(
            "vector v6 footer requires {footer_wire_len} bytes, above the configured footer workspace"
        )));
    }
    let footer_bytes = serialize_bounded(&footer, MAX_FOOTER_BYTES, "vector v6 footer")?;
    let footer_offset = writer.stream_position()?;
    writer.write_all(&footer_bytes)?;
    writer.write_all(TRAILER_MAGIC)?;
    writer.write_all(&(footer_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&crc32fast::hash(&footer_bytes).to_le_bytes())?;
    let object_len = writer.stream_position()?;
    if object_len
        != footer_offset
            .checked_add(footer_bytes.len() as u64)
            .and_then(|offset| offset.checked_add(TRAILER_LEN as u64))
            .ok_or_else(|| Error::invariant("vector v6 object length overflows"))?
    {
        return Err(Error::invariant(
            "vector v6 final writer position is inconsistent",
        ));
    }
    let page_count = u32::try_from(footer.pages.len())
        .map_err(|_| Error::invariant("vector v6 page count exceeds u32"))?;
    Ok(Some((
        writer,
        VectorV6BuildOutput {
            segment,
            object_len,
            page_count,
            version_table,
        },
    )))
}

impl VectorV6Reader {
    /// Open with three bounded reads (magic/trailer, footer, NAMISV01
    /// header+directory). Vector and filter payloads remain remote.
    pub async fn open(
        source: Arc<dyn SearchVersionRangeSource>,
        file_len: u64,
        state: &SearchLsmState,
        segment: &SearchSegmentRef,
        descriptor: &VectorIndexDescriptor,
    ) -> Result<Self> {
        let minimum = (MAGIC_V6.len() + TRAILER_LEN) as u64;
        if file_len < minimum {
            return Err(Error::invariant("vector v6 body is too short"));
        }
        let trailer_start = file_len - TRAILER_LEN as u64;
        let probes = source
            .read_ranges(&[0..MAGIC_V6.len() as u64, trailer_start..file_len])
            .await?;
        if probes.len() != 2 || probes[0].as_ref() != MAGIC_V6 || probes[1].len() != TRAILER_LEN {
            return Err(Error::invariant(
                "vector v6 magic/trailer range probes are malformed",
            ));
        }
        let (footer_len, footer_crc) = decode_trailer(&probes[1])?;
        if footer_len == 0 || footer_len > MAX_FOOTER_BYTES {
            return Err(Error::invariant("vector v6 footer length is invalid"));
        }
        let footer_offset = trailer_start
            .checked_sub(footer_len)
            .ok_or_else(|| Error::invariant("vector v6 footer starts before object"))?;
        let footer_bytes = source.read_range(footer_offset..trailer_start).await?;
        require_len(
            &footer_bytes,
            usize::try_from(footer_len)
                .map_err(|_| Error::invariant("vector v6 footer does not fit usize"))?,
            "footer",
        )?;
        if crc32fast::hash(&footer_bytes) != footer_crc {
            return Err(Error::invariant("vector v6 footer checksum mismatch"));
        }
        let footer: Footer =
            deserialize_bounded(&footer_bytes, MAX_FOOTER_BYTES, "vector v6 footer")?;
        validate_footer(&footer, footer_offset, state, segment, descriptor)?;
        let version_reader =
            SearchVersionTableReader::open(source.clone(), footer.binding.version_table.clone())
                .await?;
        Ok(Self {
            source,
            file_len,
            footer_offset,
            footer,
            version_reader,
        })
    }

    pub fn segment(&self) -> &SearchSegmentRef {
        &self.footer.binding.segment
    }

    pub fn version_reader(&self) -> &SearchVersionTableReader {
        &self.version_reader
    }

    pub fn dim(&self) -> u32 {
        self.footer.dim
    }

    pub fn metric(&self) -> VectorMetric {
        self.footer.metric
    }

    pub fn page_count(&self) -> usize {
        self.footer.pages.len()
    }

    pub fn supports_filter_property(&self, property: &str) -> bool {
        self.footer
            .filters
            .binary_search_by(|candidate| candidate.property.as_str().cmp(property))
            .is_ok()
    }

    pub fn resident_metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(
                self.footer
                    .pages
                    .capacity()
                    .saturating_mul(std::mem::size_of::<VectorPageRef>()),
            )
            .saturating_add(
                self.footer
                    .filters
                    .iter()
                    .map(|filter| {
                        filter
                            .property
                            .capacity()
                            .saturating_add(std::mem::size_of::<FilterBlockRef>())
                            .saturating_add(
                                filter
                                    .values
                                    .iter()
                                    .map(|value| {
                                        std::mem::size_of::<FilterValueRef>()
                                            + filter_value_resident_bytes(&value.value)
                                    })
                                    .sum::<usize>(),
                            )
                    })
                    .sum::<usize>(),
            )
            .saturating_add(self.version_reader.resident_metadata_bytes())
    }

    /// Exhaustive exact search of this delta's live payload pages.
    pub async fn search_exact(
        &self,
        query: &[f32],
        k: usize,
        groups: &[(String, Vec<SearchFilterValue>)],
    ) -> Result<VectorV6SearchResult> {
        validate_query(query, self.footer.dim)?;
        if k == 0 || self.footer.binding.segment.live_payload_count == 0 {
            return Ok(VectorV6SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                scanned_pages: 0,
                eligible_rows_seen: 0,
                peak_live_bytes: 0,
            });
        }
        let workspace_bytes =
            estimate_vector_search_workspace(&self.footer.pages, &self.footer.filters, groups, k)?;
        let _workspace = shared_search_workspace()
            .reserve("vector v6 exact search", workspace_bytes)
            .await?;
        let (allowed, applied_filter_groups) = self.load_filter_mask(groups).await?;
        let higher_is_better = self.footer.metric != VectorMetric::Euclidean;
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        let mut eligible_rows_seen = 0usize;
        let mask_bytes = allowed
            .as_ref()
            .map(QueryFilterMask::resident_bytes)
            .unwrap_or(0);
        let mut peak_live_bytes = mask_bytes.saturating_add(
            heap.capacity()
                .saturating_mul(std::mem::size_of::<RankedHit>()),
        );

        for page in &self.footer.pages {
            let compressed = self.source.read_range(page.wire.range()?).await?;
            let raw = decode_block(&compressed, &page.wire, "vector page")?;
            let rows = decode_vector_page(&raw, page, self.footer.dim)?;
            peak_live_bytes = peak_live_bytes.max(
                mask_bytes
                    .saturating_add(vector_rows_resident_bytes(&rows))
                    .saturating_add(
                        heap.capacity()
                            .saturating_mul(std::mem::size_of::<RankedHit>()),
                    ),
            );
            for row in rows {
                if allowed
                    .as_ref()
                    .is_some_and(|mask| !mask.contains(row.ordinal))
                {
                    continue;
                }
                eligible_rows_seen = eligible_rows_seen.saturating_add(1);
                let score = metric_score(self.footer.metric, &row.vector, query);
                heap.push(RankedHit {
                    hit: VectorV6Hit {
                        node_id: row.node_id,
                        lsn: row.lsn,
                        payload_fingerprint: row.payload_fingerprint,
                        score,
                    },
                    higher_is_better,
                });
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        let mut ranked = heap
            .into_iter()
            .map(|candidate| candidate.hit)
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| compare_hits(left, right, higher_is_better));
        Ok(VectorV6SearchResult {
            hits: ranked,
            applied_filter_groups,
            scanned_pages: self.footer.pages.len(),
            eligible_rows_seen,
            peak_live_bytes,
        })
    }

    /// Bounded full scrub, including page/version identity parity.
    pub async fn verify_all(&self) -> Result<()> {
        self.version_reader.verify_all().await?;
        for page in &self.footer.pages {
            let compressed = self.source.read_range(page.wire.range()?).await?;
            let raw = decode_block(&compressed, &page.wire, "vector page")?;
            let rows = decode_vector_page(&raw, page, self.footer.dim)?;
            let ids = rows.iter().map(|row| row.node_id).collect::<Vec<_>>();
            let versions = self.version_reader.point_probe_many(&ids).await?;
            for (row, version) in rows.iter().zip(versions) {
                let Some(version) = version else {
                    return Err(Error::invariant(
                        "vector v6 live payload has no version-table record",
                    ));
                };
                if version.node_id != row.node_id
                    || version.lsn != row.lsn
                    || version.payload_fingerprint != row.payload_fingerprint
                    || !matches!(
                        version.operation,
                        SearchVersionOperation::Live { payload_ordinal }
                            if payload_ordinal == row.ordinal
                    )
                {
                    return Err(Error::invariant(
                        "vector v6 live payload disagrees with version table",
                    ));
                }
            }
        }
        for filter in &self.footer.filters {
            validate_filter_directory(filter, self.footer.binding.segment.live_payload_count)?;
            for value in &filter.values {
                let compressed = self.source.read_range(value.wire.range()?).await?;
                let raw = decode_block(&compressed, &value.wire, "filter posting block")?;
                decode_filter_query_mask(raw, value, filter.row_count)?;
            }
        }
        Ok(())
    }

    async fn load_filter_mask(
        &self,
        groups: &[(String, Vec<SearchFilterValue>)],
    ) -> Result<(Option<QueryFilterMask>, usize)> {
        let row_count = self.footer.binding.segment.live_payload_count;
        let mut combined: Option<QueryFilterMask> = None;
        let mut applied = 0usize;
        for (property, alternatives) in groups {
            let Ok(index) = self
                .footer
                .filters
                .binary_search_by(|candidate| candidate.property.as_str().cmp(property))
            else {
                continue;
            };
            let reference = &self.footer.filters[index];
            validate_filter_directory(reference, row_count)?;
            let mut group = QueryFilterMask::empty();
            for value in alternatives
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
            {
                let Ok(index) = reference
                    .values
                    .binary_search_by(|candidate| candidate.value.cmp(value))
                else {
                    continue;
                };
                let value = &reference.values[index];
                let compressed = self.source.read_range(value.wire.range()?).await?;
                let raw = decode_block(&compressed, &value.wire, "filter posting block")?;
                group = group.union(decode_filter_query_mask(raw, value, row_count)?, row_count)?;
            }
            combined = Some(match combined {
                Some(mask) => mask.intersect(group, row_count)?,
                None => group,
            });
            applied += 1;
        }
        Ok((combined, applied))
    }
}

#[derive(Debug)]
struct RankedHit {
    hit: VectorV6Hit,
    higher_is_better: bool,
}

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedHit {}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The heap root is the worst retained candidate.
impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        debug_assert_eq!(self.higher_is_better, other.higher_is_better);
        let score_order = self.hit.score.total_cmp(&other.hit.score);
        let worse_score = if self.higher_is_better {
            score_order.reverse()
        } else {
            score_order
        };
        worse_score.then_with(|| self.hit.node_id.cmp(&other.hit.node_id))
    }
}

fn compare_hits(left: &VectorV6Hit, right: &VectorV6Hit, higher_is_better: bool) -> Ordering {
    let score = if higher_is_better {
        right.score.total_cmp(&left.score)
    } else {
        left.score.total_cmp(&right.score)
    };
    score.then_with(|| left.node_id.cmp(&right.node_id))
}

fn validate_build_configuration<W: Seek>(
    writer: &mut W,
    state: &SearchLsmState,
    descriptor: &VectorIndexDescriptor,
    context: &mut VectorV6BuildContext,
    options: VectorV6BuildOptions,
) -> Result<()> {
    if writer.stream_position()? != 0 {
        return Err(Error::invariant(
            "vector v6 object writer must start at offset zero",
        ));
    }
    if state.kind != SearchLsmKind::Vector
        || state.index_name != descriptor.name
        || state.generation_id.is_nil()
        || context.sst_id.is_nil()
    {
        return Err(Error::invariant(
            "vector v6 build context disagrees with vector generation",
        ));
    }
    if descriptor.dim == 0
        || descriptor.dim > MAX_DIM
        || !descriptor.alpha.is_finite()
        || descriptor.r == 0
        || descriptor.l_build == 0
    {
        return Err(Error::invariant(
            "vector v6 descriptor configuration is invalid",
        ));
    }
    if options.rows_per_page == 0 {
        return Err(Error::invariant("vector v6 rows_per_page must be positive"));
    }
    let row_len = vector_row_len(descriptor.dim)?;
    let page_len = VECTOR_PAGE_HEADER_LEN
        .checked_add(
            row_len
                .checked_mul(options.rows_per_page)
                .ok_or_else(|| Error::invariant("vector v6 page size overflows"))?,
        )
        .ok_or_else(|| Error::invariant("vector v6 page size overflows"))?;
    if page_len as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(
            "vector v6 configured page exceeds the raw block limit",
        ));
    }
    validate_event_ranges(&context.event_ranges)?;
    context.complete_filter_properties.sort();
    context.complete_filter_properties.dedup();
    if context
        .complete_filter_properties
        .iter()
        .any(|property| property.is_empty())
    {
        return Err(Error::invariant(
            "vector v6 complete filter property is empty",
        ));
    }
    Ok(())
}

fn validate_event_ranges(ranges: &[SearchEventRange]) -> Result<()> {
    if ranges.is_empty()
        || ranges.iter().any(|range| !range.is_valid())
        || ranges.windows(2).any(|pair| pair[0].end > pair[1].start)
    {
        return Err(Error::invariant(
            "search delta event ranges are empty, invalid, or overlapping",
        ));
    }
    Ok(())
}

fn enforce_build_memory_budget(
    mutations: &[VectorV6Mutation],
    descriptor: &VectorIndexDescriptor,
    options: VectorV6BuildOptions,
) -> Result<usize> {
    let budget = match std::env::var(INDEX_BUILD_MEMORY_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            Error::precondition(format!(
                "{INDEX_BUILD_MEMORY_ENV} must be an exact byte count: {error}"
            ))
        })?,
        Err(std::env::VarError::NotPresent) => DEFAULT_INDEX_BUILD_MEMORY_BYTES,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::precondition(format!(
                "{INDEX_BUILD_MEMORY_ENV} is not valid UTF-8"
            )));
        }
    };
    if budget == 0 {
        return Err(Error::precondition(format!(
            "{INDEX_BUILD_MEMORY_ENV} must be positive"
        )));
    }
    let input_bytes = mutations.iter().fold(
        mutations
            .len()
            .saturating_mul(std::mem::size_of::<VectorV6Mutation>()),
        |total, mutation| {
            [mutation.before.as_ref(), mutation.after.as_ref()]
                .into_iter()
                .flatten()
                .fold(total, |total, payload| {
                    total
                        .saturating_add(
                            payload
                                .vector
                                .capacity()
                                .saturating_mul(std::mem::size_of::<f32>()),
                        )
                        .saturating_add(payload.filters.iter().fold(
                            0usize,
                            |bytes, (property, value)| {
                                bytes
                                    .saturating_add(property.capacity())
                                    .saturating_add(std::mem::size_of::<SearchFilterValue>())
                                    .saturating_add(filter_value_resident_bytes(value))
                            },
                        ))
                })
        },
    );
    let live_filter_occurrences = mutations
        .iter()
        .filter_map(|mutation| mutation.after.as_ref())
        .map(|payload| payload.filters.len())
        .sum::<usize>();
    let filter_workspace = live_filter_occurrences.saturating_mul(
        std::mem::size_of::<u64>()
            + std::mem::size_of::<SearchFilterValue>()
            + std::mem::size_of::<Vec<u64>>()
            + 64,
    );
    let page_raw = vector_row_len(descriptor.dim)?
        .saturating_mul(options.rows_per_page)
        .saturating_add(VECTOR_PAGE_HEADER_LEN);
    let required = input_bytes
        .saturating_add(mutations.len().saturating_mul(std::mem::size_of::<u64>()))
        .saturating_add(filter_workspace)
        .saturating_add(page_raw.saturating_mul(3));
    if required > budget {
        return Err(Error::precondition(format!(
            "vector v6 delta build requires an estimated {required} bytes, above the \
             {budget}-byte {INDEX_BUILD_MEMORY_ENV} cap; flush a smaller delta batch"
        )));
    }
    Ok(budget)
}

fn validate_payload(
    payload: &VectorV6Payload,
    descriptor: &VectorIndexDescriptor,
    complete_filter_properties: &[String],
) -> Result<()> {
    if payload.vector.len() != descriptor.dim as usize {
        return Err(Error::invariant(format!(
            "vector v6 payload dimension {} != declared {}",
            payload.vector.len(),
            descriptor.dim
        )));
    }
    if payload.vector.iter().any(|value| !value.is_finite()) {
        return Err(Error::invariant(
            "vector v6 payload contains a non-finite component",
        ));
    }
    validate_filter_map_keys(&payload.filters)?;
    if payload
        .filters
        .keys()
        .any(|property| complete_filter_properties.binary_search(property).is_err())
    {
        return Err(Error::invariant(
            "vector v6 payload contains an unadvertised native-filter property",
        ));
    }
    Ok(())
}

fn filter_value_resident_bytes(value: &SearchFilterValue) -> usize {
    match value {
        SearchFilterValue::String(value) => value.capacity(),
        SearchFilterValue::Bytes(value) => value.capacity(),
        SearchFilterValue::Bool(_)
        | SearchFilterValue::I64(_)
        | SearchFilterValue::F64Bits(_)
        | SearchFilterValue::Date(_)
        | SearchFilterValue::DateTime(_) => 0,
    }
}

fn validate_filter_map_keys(filters: &BTreeMap<String, SearchFilterValue>) -> Result<()> {
    if filters.keys().any(|property| property.is_empty()) {
        return Err(Error::invariant("native-filter property name is empty"));
    }
    Ok(())
}

fn build_filter_postings(
    property: &str,
    row_count: u64,
    mutations: &[VectorV6Mutation],
) -> Result<BTreeMap<SearchFilterValue, Vec<u64>>> {
    let mut postings = BTreeMap::<SearchFilterValue, Vec<u64>>::new();
    let mut ordinal = 0u64;
    for mutation in mutations {
        let Some(after) = &mutation.after else {
            continue;
        };
        if let Some(value) = after.filters.get(property) {
            postings.entry(value.clone()).or_default().push(ordinal);
        }
        ordinal += 1;
    }
    if ordinal != row_count {
        return Err(Error::invariant(
            "vector v6 filter ordinal accounting diverged",
        ));
    }
    Ok(postings)
}

fn encode_filter_posting(
    ordinals: &[u64],
    row_count: u64,
) -> Result<(FilterPostingEncoding, Vec<u8>)> {
    if ordinals.is_empty()
        || ordinals.windows(2).any(|pair| pair[0] >= pair[1])
        || ordinals.last().is_some_and(|ordinal| *ordinal >= row_count)
    {
        return Err(Error::invariant(
            "vector v6 filter posting input is inconsistent",
        ));
    }
    let mut sparse = Vec::new();
    let mut previous = 0u64;
    for (index, ordinal) in ordinals.iter().copied().enumerate() {
        let delta = if index == 0 {
            ordinal
        } else {
            ordinal - previous
        };
        encode_u64_varint(delta, &mut sparse);
        previous = ordinal;
    }
    let dense_len = bitmap_words(row_count)?
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::invariant("vector v6 dense filter length overflows"))?;
    if sparse.len() < dense_len {
        return Ok((FilterPostingEncoding::SparseDeltaVarint, sparse));
    }
    let mut dense = vec![0u8; dense_len];
    for ordinal in ordinals {
        set_dense_filter_bit(&mut dense, *ordinal, row_count)?;
    }
    Ok((FilterPostingEncoding::DenseBitmap, dense))
}

fn validate_filter_directory(reference: &FilterBlockRef, row_count: u64) -> Result<()> {
    if reference.property.is_empty()
        || reference.row_count != row_count
        || reference
            .values
            .windows(2)
            .any(|pair| pair[0].value >= pair[1].value)
    {
        return Err(Error::invariant(
            "vector v6 filter directory is inconsistent",
        ));
    }
    let dense_len = bitmap_words(row_count)?
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::invariant("vector v6 dense filter length overflows"))?;
    for value in &reference.values {
        if value.cardinality == 0 || value.cardinality > row_count {
            return Err(Error::invariant(
                "vector v6 filter cardinality is inconsistent",
            ));
        }
        match value.encoding {
            FilterPostingEncoding::SparseDeltaVarint => {
                if value.wire.raw_len == 0
                    || u64::from(value.wire.raw_len) > value.cardinality.saturating_mul(10)
                {
                    return Err(Error::invariant(
                        "vector v6 sparse filter length is inconsistent",
                    ));
                }
            }
            FilterPostingEncoding::DenseBitmap => {
                if usize::try_from(value.wire.raw_len).ok() != Some(dense_len) {
                    return Err(Error::invariant(
                        "vector v6 dense filter length is inconsistent",
                    ));
                }
            }
        }
        validate_block_limits(&value.wire)?;
    }
    Ok(())
}

fn decode_filter_query_mask(
    raw: Vec<u8>,
    reference: &FilterValueRef,
    row_count: u64,
) -> Result<QueryFilterMask> {
    match reference.encoding {
        FilterPostingEncoding::SparseDeltaVarint => {
            let capacity = usize::try_from(reference.cardinality)
                .map_err(|_| Error::invariant("vector v6 filter cardinality exceeds usize"))?;
            let mut ordinals = Vec::with_capacity(capacity);
            let mut cursor = 0usize;
            let mut previous = 0u64;
            for index in 0..reference.cardinality {
                let delta = decode_u64_varint(&raw, &mut cursor)?;
                if index > 0 && delta == 0 {
                    return Err(Error::invariant(
                        "vector v6 sparse filter ordinals are duplicated",
                    ));
                }
                let ordinal = previous
                    .checked_add(delta)
                    .ok_or_else(|| Error::invariant("vector v6 filter ordinal overflows"))?;
                if ordinal >= row_count {
                    return Err(Error::invariant(
                        "vector v6 sparse filter ordinal leaves vector rows",
                    ));
                }
                ordinals.push(ordinal);
                previous = ordinal;
            }
            if cursor != raw.len() {
                return Err(Error::invariant(
                    "vector v6 sparse filter posting has trailing bytes",
                ));
            }
            Ok(QueryFilterMask::Sparse(ordinals))
        }
        FilterPostingEncoding::DenseBitmap => {
            let expected = bitmap_words(row_count)?
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or_else(|| Error::invariant("vector v6 dense filter length overflows"))?;
            if raw.len() != expected {
                return Err(Error::invariant(
                    "vector v6 dense filter bitmap length is inconsistent",
                ));
            }
            let cardinality = raw
                .iter()
                .map(|byte| u64::from(byte.count_ones()))
                .sum::<u64>();
            let remainder = row_count % 64;
            let last_word = raw
                .chunks_exact(8)
                .last()
                .map(|bytes| {
                    u64::from_le_bytes(bytes.try_into().expect("fixed filter bitmap word"))
                })
                .unwrap_or(0);
            if (remainder != 0 && last_word & (!0u64 << remainder) != 0)
                || cardinality != reference.cardinality
            {
                return Err(Error::invariant(
                    "vector v6 dense filter bitmap/cardinality is inconsistent",
                ));
            }
            Ok(QueryFilterMask::Dense(raw))
        }
    }
}

fn sparse_or_dense(ordinals: Vec<u64>, row_count: u64) -> Result<QueryFilterMask> {
    let dense_len = bitmap_words(row_count)?
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::invariant("vector v6 query bitmap length overflows"))?;
    if ordinals.len().saturating_mul(std::mem::size_of::<u64>()) < dense_len {
        return Ok(QueryFilterMask::Sparse(ordinals));
    }
    let mut dense = vec![0u8; dense_len];
    for ordinal in ordinals {
        set_dense_filter_bit(&mut dense, ordinal, row_count)?;
    }
    Ok(QueryFilterMask::Dense(dense))
}

fn maybe_sparsify_dense(dense: Vec<u8>, row_count: u64) -> Result<QueryFilterMask> {
    let cardinality = dense
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum::<usize>();
    if cardinality.saturating_mul(std::mem::size_of::<u64>()) >= dense.len() {
        return Ok(QueryFilterMask::Dense(dense));
    }
    let mut ordinals = Vec::with_capacity(cardinality);
    for ordinal in 0..row_count {
        if dense_filter_contains(&dense, ordinal) {
            ordinals.push(ordinal);
        }
    }
    Ok(QueryFilterMask::Sparse(ordinals))
}

fn set_dense_filter_bit(bitmap: &mut [u8], ordinal: u64, row_count: u64) -> Result<()> {
    if ordinal >= row_count {
        return Err(Error::invariant(
            "vector v6 filter ordinal leaves vector rows",
        ));
    }
    let byte = usize::try_from(ordinal / 8)
        .map_err(|_| Error::invariant("vector v6 filter ordinal exceeds usize"))?;
    let target = bitmap
        .get_mut(byte)
        .ok_or_else(|| Error::invariant("vector v6 filter ordinal leaves bitmap"))?;
    *target |= 1u8 << (ordinal % 8);
    Ok(())
}

fn dense_filter_contains(bitmap: &[u8], ordinal: u64) -> bool {
    usize::try_from(ordinal / 8)
        .ok()
        .and_then(|byte| bitmap.get(byte))
        .is_some_and(|byte| byte & (1u8 << (ordinal % 8)) != 0)
}

fn encode_u64_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn decode_u64_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..=63).step_by(7) {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| Error::invariant("vector v6 sparse varint is truncated"))?;
        *cursor += 1;
        let payload = u64::from(byte & 0x7f);
        if shift == 63 && payload > 1 {
            return Err(Error::invariant("vector v6 sparse varint overflows"));
        }
        value |= payload << shift;
        if byte & 0x80 == 0 {
            if shift > 0 && payload == 0 {
                return Err(Error::invariant("vector v6 sparse varint is not canonical"));
            }
            return Ok(value);
        }
    }
    Err(Error::invariant("vector v6 sparse varint is too long"))
}

fn bitmap_words(row_count: u64) -> Result<usize> {
    usize::try_from(row_count.div_ceil(64))
        .map_err(|_| Error::invariant("native-filter bitmap does not fit usize"))
}

fn write_vector_page<W: Write + Seek>(
    writer: &mut W,
    rows: &[VectorPageInput<'_>],
    dim: u32,
    compression_level: i32,
) -> Result<VectorPageRef> {
    let first = rows
        .first()
        .ok_or_else(|| Error::invariant("cannot write an empty vector v6 page"))?;
    let last = rows
        .last()
        .ok_or_else(|| Error::invariant("cannot write an empty vector v6 page"))?;
    if rows
        .windows(2)
        .any(|pair| pair[0].ordinal + 1 != pair[1].ordinal || pair[0].node_id >= pair[1].node_id)
    {
        return Err(Error::invariant(
            "vector v6 page rows are not contiguous and sorted",
        ));
    }
    let row_len = vector_row_len(dim)?;
    let capacity = VECTOR_PAGE_HEADER_LEN
        .checked_add(
            row_len
                .checked_mul(rows.len())
                .ok_or_else(|| Error::invariant("vector v6 page capacity overflows"))?,
        )
        .ok_or_else(|| Error::invariant("vector v6 page capacity overflows"))?;
    let mut raw = Vec::with_capacity(capacity);
    raw.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    raw.extend_from_slice(&dim.to_le_bytes());
    for row in rows {
        raw.extend_from_slice(&row.node_id);
        raw.extend_from_slice(&row.lsn.to_le_bytes());
        raw.extend_from_slice(&row.payload_fingerprint.to_le_bytes());
        for value in row.vector {
            raw.extend_from_slice(&value.to_le_bytes());
        }
    }
    let wire = write_compressed_block(writer, &raw, compression_level, "vector page")?;
    Ok(VectorPageRef {
        first_ordinal: first.ordinal,
        row_count: u32::try_from(rows.len())
            .map_err(|_| Error::invariant("vector v6 page row count exceeds u32"))?,
        first_node_id: first.node_id,
        last_node_id: last.node_id,
        min_lsn: rows.iter().map(|row| row.lsn).min().unwrap_or(0),
        max_lsn: rows.iter().map(|row| row.lsn).max().unwrap_or(0),
        wire,
    })
}

fn decode_vector_page(
    raw: &[u8],
    reference: &VectorPageRef,
    dim: u32,
) -> Result<Vec<VectorPageRow>> {
    if raw.len() < VECTOR_PAGE_HEADER_LEN
        || read_u32(raw, 0)? != reference.row_count
        || read_u32(raw, 4)? != dim
    {
        return Err(Error::invariant("vector v6 page header is inconsistent"));
    }
    let row_len = vector_row_len(dim)?;
    let expected = VECTOR_PAGE_HEADER_LEN
        .checked_add(
            row_len
                .checked_mul(reference.row_count as usize)
                .ok_or_else(|| Error::invariant("vector v6 page length overflows"))?,
        )
        .ok_or_else(|| Error::invariant("vector v6 page length overflows"))?;
    if raw.len() != expected {
        return Err(Error::invariant("vector v6 page length is inconsistent"));
    }
    let mut rows = Vec::with_capacity(reference.row_count as usize);
    for row_index in 0..reference.row_count as usize {
        let start = VECTOR_PAGE_HEADER_LEN + row_index * row_len;
        let node_id = raw[start..start + 16]
            .try_into()
            .expect("fixed vector v6 NodeId");
        let lsn = read_u64(raw, start + 16)?;
        let payload_fingerprint = read_u64(raw, start + 24)?;
        if lsn == 0 {
            return Err(Error::invariant("vector v6 page contains LSN zero"));
        }
        let mut vector = Vec::with_capacity(dim as usize);
        let mut cursor = start + VECTOR_ROW_PREFIX_LEN;
        for _ in 0..dim {
            let bits = read_u32(raw, cursor)?;
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                return Err(Error::invariant(
                    "vector v6 page contains a non-finite component",
                ));
            }
            vector.push(value);
            cursor += 4;
        }
        rows.push(VectorPageRow {
            ordinal: reference.first_ordinal + row_index as u64,
            node_id,
            lsn,
            payload_fingerprint,
            vector,
        });
    }
    if rows.first().map(|row| row.node_id) != Some(reference.first_node_id)
        || rows.last().map(|row| row.node_id) != Some(reference.last_node_id)
        || rows
            .windows(2)
            .any(|pair| pair[0].node_id >= pair[1].node_id)
        || rows.iter().map(|row| row.lsn).min() != Some(reference.min_lsn)
        || rows.iter().map(|row| row.lsn).max() != Some(reference.max_lsn)
    {
        return Err(Error::invariant(
            "vector v6 page disagrees with its directory entry",
        ));
    }
    Ok(rows)
}

fn vector_row_len(dim: u32) -> Result<usize> {
    VECTOR_ROW_PREFIX_LEN
        .checked_add(
            (dim as usize)
                .checked_mul(4)
                .ok_or_else(|| Error::invariant("vector v6 row size overflows"))?,
        )
        .ok_or_else(|| Error::invariant("vector v6 row size overflows"))
}

fn write_compressed_block<W: Write + Seek>(
    writer: &mut W,
    raw: &[u8],
    compression_level: i32,
    what: &str,
) -> Result<BlockRef> {
    if raw.is_empty() || raw.len() as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v6 {what} raw length is invalid"
        )));
    }
    let compressed =
        zstd::stream::encode_all(Cursor::new(raw), compression_level).map_err(|error| {
            Error::invariant(format!("vector v6 {what} compression failed: {error}"))
        })?;
    if compressed.is_empty() || compressed.len() as u64 > MAX_COMPRESSED_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v6 {what} compressed length is invalid"
        )));
    }
    let offset = writer.stream_position()?;
    writer.write_all(&compressed)?;
    Ok(BlockRef {
        offset,
        len: u32::try_from(compressed.len())
            .map_err(|_| Error::invariant("vector v6 compressed block exceeds u32"))?,
        raw_len: u32::try_from(raw.len())
            .map_err(|_| Error::invariant("vector v6 raw block exceeds u32"))?,
        compressed_crc32: crc32fast::hash(&compressed),
        raw_xxh3: non_zero_xxh3(raw),
    })
}

fn decode_block(compressed: &[u8], reference: &BlockRef, what: &str) -> Result<Vec<u8>> {
    require_len(compressed, reference.len as usize, what)?;
    if crc32fast::hash(compressed) != reference.compressed_crc32 {
        return Err(Error::invariant(format!(
            "vector v6 {what} compressed checksum mismatch"
        )));
    }
    if reference.raw_len == 0 || u64::from(reference.raw_len) > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v6 {what} raw length is invalid"
        )));
    }
    let decoder = zstd::stream::read::Decoder::new(compressed)
        .map_err(|error| Error::invariant(format!("vector v6 {what} decoder failed: {error}")))?;
    let mut raw = Vec::with_capacity(reference.raw_len as usize);
    decoder
        .take(u64::from(reference.raw_len) + 1)
        .read_to_end(&mut raw)
        .map_err(|error| Error::invariant(format!("vector v6 {what} decode failed: {error}")))?;
    require_len(&raw, reference.raw_len as usize, what)?;
    if non_zero_xxh3(&raw) != reference.raw_xxh3 {
        return Err(Error::invariant(format!(
            "vector v6 {what} raw checksum mismatch"
        )));
    }
    Ok(raw)
}

fn validate_footer(
    footer: &Footer,
    footer_offset: u64,
    state: &SearchLsmState,
    segment: &SearchSegmentRef,
    descriptor: &VectorIndexDescriptor,
) -> Result<()> {
    if footer.footer_version != FOOTER_VERSION
        || footer.mode != VectorV6Mode::FlatExact
        || footer.dim != descriptor.dim
        || footer.metric != descriptor.metric
        || descriptor.name != state.index_name
    {
        return Err(Error::invariant(
            "vector v6 footer configuration is unsupported",
        ));
    }
    footer.binding.validate(state, segment)?;
    if footer.binding.version_table.offset != MAGIC_V6.len() as u64 {
        return Err(Error::invariant(
            "vector v6 version table does not immediately follow magic",
        ));
    }
    let expected_digest = content_digest(
        &footer.binding.version_table,
        footer.dim,
        footer.metric,
        footer.live_count_delta,
        &footer.pages,
        &footer.filters,
    )?;
    if expected_digest != segment.content_xxh3
        || segment.complete_filter_properties
            != footer
                .filters
                .iter()
                .map(|filter| filter.property.clone())
                .collect::<Vec<_>>()
        || segment.stats
            != (SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(footer.live_count_delta),
            })
    {
        return Err(Error::invariant(
            "vector v6 footer disagrees with segment statistics/content",
        ));
    }
    validate_block_layout(footer, footer_offset)?;
    Ok(())
}

fn validate_block_layout(footer: &Footer, footer_offset: u64) -> Result<()> {
    let version_end = footer
        .binding
        .version_table
        .offset
        .checked_add(footer.binding.version_table.len)
        .ok_or_else(|| Error::invariant("vector v6 version range overflows"))?;
    let mut expected_ordinal = 0u64;
    let mut previous_node: Option<[u8; 16]> = None;
    let mut previous_end = version_end;
    for page in &footer.pages {
        if page.row_count == 0
            || page.first_ordinal != expected_ordinal
            || page.first_node_id > page.last_node_id
            || page.min_lsn == 0
            || page.min_lsn > page.max_lsn
            || previous_node.is_some_and(|node| node >= page.first_node_id)
        {
            return Err(Error::invariant("vector v6 page directory is inconsistent"));
        }
        validate_block_position(&page.wire, previous_end, footer_offset)?;
        previous_end = page.wire.range()?.end;
        previous_node = Some(page.last_node_id);
        expected_ordinal = expected_ordinal
            .checked_add(u64::from(page.row_count))
            .ok_or_else(|| Error::invariant("vector v6 page ordinals overflow"))?;
    }
    if expected_ordinal != footer.binding.segment.live_payload_count {
        return Err(Error::invariant(
            "vector v6 page directory live count is inconsistent",
        ));
    }
    let mut previous_property: Option<&str> = None;
    for filter in &footer.filters {
        if filter.property.is_empty()
            || previous_property.is_some_and(|previous| previous >= filter.property.as_str())
        {
            return Err(Error::invariant(
                "vector v6 filter directory is not strictly sorted",
            ));
        }
        validate_filter_directory(filter, footer.binding.segment.live_payload_count)?;
        for value in &filter.values {
            validate_block_position(&value.wire, previous_end, footer_offset)?;
            previous_end = value.wire.range()?.end;
        }
        previous_property = Some(&filter.property);
    }
    if previous_end != footer_offset {
        return Err(Error::invariant(
            "vector v6 payload blocks do not end at footer",
        ));
    }
    Ok(())
}

fn validate_block_position(reference: &BlockRef, minimum: u64, footer_offset: u64) -> Result<()> {
    let range = reference.range()?;
    validate_block_limits(reference)?;
    if range.start != minimum || range.end > footer_offset {
        return Err(Error::invariant(
            "vector v6 block directory contains an invalid range",
        ));
    }
    Ok(())
}

fn validate_block_limits(reference: &BlockRef) -> Result<()> {
    if reference.len == 0
        || u64::from(reference.len) > MAX_COMPRESSED_BLOCK_BYTES
        || reference.raw_len == 0
        || u64::from(reference.raw_len) > MAX_RAW_BLOCK_BYTES
        || reference.raw_xxh3 == 0
    {
        return Err(Error::invariant(
            "vector v6 block directory contains invalid limits",
        ));
    }
    Ok(())
}

fn content_digest(
    version_table: &SearchVersionTableRef,
    dim: u32,
    metric: VectorMetric,
    live_count_delta: i64,
    pages: &[VectorPageRef],
    filters: &[FilterBlockRef],
) -> Result<u64> {
    let material = ContentDigestMaterial {
        domain: CONTENT_DOMAIN,
        format_version: FORMAT_VERSION,
        mode: VectorV6Mode::FlatExact,
        version_table,
        dim,
        metric,
        live_count_delta,
        pages,
        filters,
    };
    let encoded = serialize_bounded(&material, MAX_FOOTER_BYTES, "vector v6 content digest")?;
    Ok(non_zero_xxh3(&encoded))
}

fn validate_query(query: &[f32], dim: u32) -> Result<()> {
    if query.len() != dim as usize {
        return Err(Error::invariant(format!(
            "vector v6 query dimension {} != index dimension {dim}",
            query.len()
        )));
    }
    if query.iter().any(|value| !value.is_finite()) {
        return Err(Error::invariant(
            "vector v6 query contains a non-finite component",
        ));
    }
    Ok(())
}

fn vector_rows_resident_bytes(rows: &[VectorPageRow]) -> usize {
    rows.len()
        .saturating_mul(std::mem::size_of::<VectorPageRow>())
        .saturating_add(
            rows.iter()
                .map(|row| {
                    row.vector
                        .capacity()
                        .saturating_mul(std::mem::size_of::<f32>())
                })
                .sum::<usize>(),
        )
}

fn estimate_vector_search_workspace(
    pages: &[VectorPageRef],
    filters: &[FilterBlockRef],
    groups: &[(String, Vec<SearchFilterValue>)],
    k: usize,
) -> Result<usize> {
    let page_bytes = pages
        .iter()
        .map(|page| {
            (page.wire.raw_len as usize)
                .saturating_mul(3)
                .saturating_add(
                    (page.row_count as usize).saturating_mul(std::mem::size_of::<VectorPageRow>()),
                )
        })
        .max()
        .unwrap_or(0);
    let row_count = filters.first().map(|filter| filter.row_count).unwrap_or(0);
    let dense_bytes = bitmap_words(row_count)?
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::invariant("vector v6 query filter estimate overflows"))?;
    let mut combined_bytes = 0usize;
    let mut filter_peak = 0usize;
    for (property, alternatives) in groups {
        let Ok(index) =
            filters.binary_search_by(|candidate| candidate.property.as_str().cmp(property))
        else {
            continue;
        };
        let reference = &filters[index];
        let mut selected_raw_max = 0usize;
        let mut sparse_bytes = 0usize;
        let mut dense = false;
        for value in alternatives
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
        {
            let Ok(index) = reference
                .values
                .binary_search_by(|candidate| candidate.value.cmp(value))
            else {
                continue;
            };
            let selected = &reference.values[index];
            selected_raw_max = selected_raw_max.max(selected.wire.raw_len as usize);
            match selected.encoding {
                FilterPostingEncoding::SparseDeltaVarint => {
                    sparse_bytes = sparse_bytes.saturating_add(
                        usize::try_from(selected.cardinality)
                            .unwrap_or(usize::MAX)
                            .saturating_mul(std::mem::size_of::<u64>()),
                    );
                }
                FilterPostingEncoding::DenseBitmap => dense = true,
            }
        }
        let group_bytes = if dense || sparse_bytes >= dense_bytes {
            dense_bytes
        } else {
            sparse_bytes
        };
        filter_peak = filter_peak.max(
            combined_bytes
                .saturating_add(group_bytes)
                .saturating_add(selected_raw_max.saturating_mul(2)),
        );
        combined_bytes = if combined_bytes == 0 {
            group_bytes
        } else {
            combined_bytes.min(group_bytes)
        };
    }
    let heap_bytes = k
        .saturating_add(1)
        .saturating_mul(std::mem::size_of::<RankedHit>());
    Ok(page_bytes
        .saturating_add(filter_peak)
        .saturating_add(heap_bytes)
        .saturating_add(64 * 1024))
}

fn metric_score(metric: VectorMetric, vector: &[f32], query: &[f32]) -> f32 {
    match metric {
        VectorMetric::Cosine => {
            let dot = vector
                .iter()
                .zip(query)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>();
            let left_norm = vector
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            let right_norm = query
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
        VectorMetric::Dot => vector
            .iter()
            .zip(query)
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum::<f64>() as f32,
        VectorMetric::Euclidean => vector
            .iter()
            .zip(query)
            .map(|(left, right)| {
                let difference = f64::from(*left) - f64::from(*right);
                difference * difference
            })
            .sum::<f64>()
            .sqrt() as f32,
    }
}

fn decode_trailer(bytes: &[u8]) -> Result<(u64, u32)> {
    if bytes.len() != TRAILER_LEN || &bytes[..8] != TRAILER_MAGIC {
        return Err(Error::invariant("vector v6 trailer magic/length mismatch"));
    }
    Ok((read_u64(bytes, 8)?, read_u32(bytes, 16)?))
}

fn serialize_bounded<T: Serialize>(value: &T, limit: u64, what: &str) -> Result<Vec<u8>> {
    let encoded = bincode_options(limit)
        .serialize(value)
        .map_err(|error| Error::invariant(format!("{what} encode failed: {error}")))?;
    if encoded.is_empty() || encoded.len() as u64 > limit {
        return Err(Error::invariant(format!("{what} exceeds its wire limit")));
    }
    Ok(encoded)
}

fn deserialize_bounded<T: DeserializeOwned>(bytes: &[u8], limit: u64, what: &str) -> Result<T> {
    if bytes.is_empty() || bytes.len() as u64 > limit {
        return Err(Error::invariant(format!("{what} length is invalid")));
    }
    bincode_options(limit)
        .deserialize(bytes)
        .map_err(|error| Error::invariant(format!("{what} decode failed: {error}")))
}

fn bincode_options(limit: u64) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(limit)
}

fn require_len(bytes: &[u8], expected: usize, what: &str) -> Result<()> {
    if bytes.len() != expected {
        return Err(Error::invariant(format!(
            "vector v6 {what} range returned {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error::invariant("vector v6 u32 offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invariant("vector v6 u32 is out of bounds"))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("checked vector v6 u32"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::invariant("vector v6 u64 offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invariant("vector v6 u64 is out of bounds"))?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("checked vector v6 u64"),
    ))
}

fn non_zero_xxh3(bytes: &[u8]) -> u64 {
    xxh3_64(bytes).max(1)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;

    use super::*;
    use crate::manifest::VectorQuantization;
    use crate::search_lsm::{SearchLsmStatus, SearchSegmentPayload};

    #[derive(Debug)]
    struct MemorySource {
        body: Bytes,
        ranges: Mutex<Vec<Range<u64>>>,
    }

    #[async_trait]
    impl SearchVersionRangeSource for MemorySource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
            self.ranges.lock().unwrap().push(range.clone());
            let start = usize::try_from(range.start)
                .map_err(|_| Error::invariant("test range start does not fit usize"))?;
            let end = usize::try_from(range.end)
                .map_err(|_| Error::invariant("test range end does not fit usize"))?;
            self.body
                .get(start..end)
                .map(Bytes::copy_from_slice)
                .ok_or_else(|| Error::invariant("test range leaves vector v6 body"))
        }
    }

    fn node(value: u64) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[8..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn descriptor(metric: VectorMetric) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: "law_vec".into(),
            label: "Articulo".into(),
            property: "embedding".into(),
            dim: 3,
            metric,
            r: 16,
            l_build: 32,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        }
    }

    fn state() -> SearchLsmState {
        SearchLsmState {
            index_name: "law_vec".into(),
            kind: SearchLsmKind::Vector,
            catalog_signature: "vector-catalog-v1".into(),
            generation_id: Uuid::from_u128(1),
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        }
    }

    fn context() -> VectorV6BuildContext {
        VectorV6BuildContext {
            sst_id: Uuid::from_u128(2),
            event_ranges: vec![SearchEventRange::new(10, 14)],
            complete_filter_properties: vec!["ambito".into(), "vigente".into()],
        }
    }

    fn payload(vector: [f32; 3], vigente: bool, ambito: &str) -> VectorV6Payload {
        VectorV6Payload {
            vector: vector.to_vec(),
            filters: BTreeMap::from([
                ("vigente".into(), SearchFilterValue::Bool(vigente)),
                ("ambito".into(), SearchFilterValue::String(ambito.into())),
            ]),
        }
    }

    fn mutations() -> Vec<VectorV6Mutation> {
        vec![
            VectorV6Mutation {
                node_id: node(3),
                lsn: 13,
                before: Some(payload([0.0, 1.0, 0.0], true, "civil")),
                after: None,
            },
            VectorV6Mutation {
                node_id: node(1),
                lsn: 11,
                before: None,
                after: Some(payload([1.0, 0.0, 0.0], true, "laboral")),
            },
            VectorV6Mutation {
                node_id: node(2),
                lsn: 12,
                before: Some(payload([0.0, 0.5, 0.0], false, "laboral")),
                after: Some(payload([0.9, 0.1, 0.0], true, "laboral")),
            },
            VectorV6Mutation {
                node_id: node(4),
                lsn: 14,
                before: Some(payload([0.0, 0.0, 1.0], true, "civil")),
                after: Some(payload([0.0, 0.0, 1.0], true, "civil")),
            },
        ]
    }

    #[tokio::test]
    async fn deterministic_delta_round_trip_filters_tombstones_and_exact_search() {
        let first = build_delta_v6(
            &state(),
            &descriptor(VectorMetric::Cosine),
            context(),
            mutations(),
            VectorV6BuildOptions {
                rows_per_page: 1,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        let second = build_delta_v6(
            &state(),
            &descriptor(VectorMetric::Cosine),
            context(),
            mutations(),
            VectorV6BuildOptions {
                rows_per_page: 1,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.body, second.body);
        assert_eq!(first.output, second.output);
        assert_eq!(first.output.segment.mutation_count, 3);
        assert_eq!(first.output.segment.live_payload_count, 2);
        assert_eq!(first.output.segment.suppress_count, 1);
        assert_eq!(
            first.output.segment.stats,
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(0)
            }
        );

        let source = Arc::new(MemorySource {
            body: first.body.clone(),
            ranges: Mutex::new(Vec::new()),
        });
        let reader = VectorV6Reader::open(
            source,
            first.body.len() as u64,
            &state(),
            &first.output.segment,
            &descriptor(VectorMetric::Cosine),
        )
        .await
        .unwrap();
        assert_eq!(reader.page_count(), 2);
        let result = reader
            .search_exact(
                &[1.0, 0.0, 0.0],
                10,
                &[
                    ("vigente".into(), vec![SearchFilterValue::Bool(true)]),
                    (
                        "ambito".into(),
                        vec![SearchFilterValue::String("laboral".into())],
                    ),
                ],
            )
            .await
            .unwrap();
        assert_eq!(result.applied_filter_groups, 2);
        assert_eq!(
            result
                .hits
                .iter()
                .map(|hit| hit.node_id)
                .collect::<Vec<_>>(),
            vec![node(1), node(2)]
        );
        assert!(result.hits[0].score > result.hits[1].score);
        let tombstone = reader
            .version_reader()
            .point_probe(node(3))
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            tombstone.operation,
            SearchVersionOperation::Suppress
        ));
        reader.verify_all().await.unwrap();
    }

    #[tokio::test]
    async fn ten_thousand_unique_filters_stay_sparse_linear_and_query_selectively() {
        let row_count = 10_000u64;
        let high_cardinality_context = VectorV6BuildContext {
            sst_id: Uuid::from_u128(3),
            event_ranges: vec![SearchEventRange::new(1, row_count + 1)],
            complete_filter_properties: vec!["codigo".into()],
        };
        let mutations = (1..=row_count)
            .map(|value| VectorV6Mutation {
                node_id: node(value),
                lsn: value,
                before: None,
                after: Some(VectorV6Payload {
                    vector: vec![value as f32, 1.0, 0.0],
                    filters: BTreeMap::from([(
                        "codigo".into(),
                        SearchFilterValue::String(format!("codigo-{value:05}")),
                    )]),
                }),
            })
            .collect();
        let artifact = build_delta_v6(
            &state(),
            &descriptor(VectorMetric::Dot),
            high_cardinality_context,
            mutations,
            VectorV6BuildOptions {
                rows_per_page: 256,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            artifact.output.segment.complete_filter_properties,
            vec!["codigo".to_owned()]
        );
        assert!(
            artifact.body.len() < 16 * 1024 * 1024,
            "high-cardinality vector wire grew non-linearly: {} bytes",
            artifact.body.len()
        );
        let reader = VectorV6Reader::open(
            Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &state(),
            &artifact.output.segment,
            &descriptor(VectorMetric::Dot),
        )
        .await
        .unwrap();
        let result = reader
            .search_exact(
                &[1.0, 0.0, 0.0],
                5,
                &[(
                    "codigo".into(),
                    vec![SearchFilterValue::String("codigo-05000".into())],
                )],
            )
            .await
            .unwrap();
        assert_eq!(result.applied_filter_groups, 1);
        assert_eq!(result.eligible_rows_seen, 1);
        assert_eq!(result.hits[0].node_id, node(5_000));
        assert!(result.peak_live_bytes < 512 * 1024);
    }

    #[test]
    fn noops_use_proven_empty_and_invalid_payloads_fail_closed() {
        let same = payload([1.0, 0.0, 0.0], true, "laboral");
        let no_op = VectorV6Mutation {
            node_id: node(1),
            lsn: 1,
            before: Some(same.clone()),
            after: Some(same),
        };
        assert!(build_delta_v6(
            &state(),
            &descriptor(VectorMetric::Cosine),
            context(),
            vec![no_op],
            VectorV6BuildOptions::default(),
        )
        .unwrap()
        .is_none());

        let mut bad = mutations();
        bad[0].after = Some(payload([f32::NAN, 0.0, 0.0], true, "civil"));
        assert!(build_delta_v6(
            &state(),
            &descriptor(VectorMetric::Cosine),
            context(),
            bad,
            VectorV6BuildOptions::default(),
        )
        .is_err());

        let mut duplicate = mutations();
        duplicate.push(duplicate[0].clone());
        assert!(build_delta_v6(
            &state(),
            &descriptor(VectorMetric::Cosine),
            context(),
            duplicate,
            VectorV6BuildOptions::default(),
        )
        .is_err());
    }

    #[tokio::test]
    async fn corruption_and_manifest_drift_are_rejected() {
        let artifact = build_delta_v6(
            &state(),
            &descriptor(VectorMetric::Euclidean),
            context(),
            mutations(),
            VectorV6BuildOptions::default(),
        )
        .unwrap()
        .unwrap();
        let mut drift = artifact.output.segment.clone();
        drift.payload = SearchSegmentPayload::ShadowOnly;
        assert!(VectorV6Reader::open(
            Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &state(),
            &drift,
            &descriptor(VectorMetric::Euclidean),
        )
        .await
        .is_err());

        let page_offset = {
            let source = Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            });
            let reader = VectorV6Reader::open(
                source,
                artifact.body.len() as u64,
                &state(),
                &artifact.output.segment,
                &descriptor(VectorMetric::Euclidean),
            )
            .await
            .unwrap();
            reader.footer.pages[0].wire.offset as usize
        };
        let mut corrupt = artifact.body.to_vec();
        corrupt[page_offset] ^= 0x80;
        let reader = VectorV6Reader::open(
            Arc::new(MemorySource {
                body: Bytes::from(corrupt),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &state(),
            &artifact.output.segment,
            &descriptor(VectorMetric::Euclidean),
        )
        .await
        .unwrap();
        assert!(reader.search_exact(&[0.0, 0.0, 0.0], 2, &[]).await.is_err());
    }
}
