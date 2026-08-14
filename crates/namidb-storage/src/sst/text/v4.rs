//! Incremental object-native full-text segment (`NAMIFT04`).
//!
//! FT4 retains V3's sparse range-readable dictionary, compressed postings and
//! positions, then adds the common `NAMISV01` winner table, exact signed corpus
//! statistics, complete native-filter bitmaps, and lineage bound to one
//! Search-LSM generation. The same wire can represent either an authoritative
//! base (absolute statistics, no suppressions) or a signed delta. Deltas retain
//! the exhaustive global-stat scorer needed for winner reconciliation; a
//! single authoritative base can additionally use exact Block-Max pruning over
//! the already-authenticated posting metadata.

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
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use crate::error::{Error, Result};
#[cfg(test)]
use crate::search_lsm::SearchSegmentFormat;
use crate::search_lsm::{
    SearchEventRange, SearchLsmKind, SearchLsmState, SearchSegmentPayload, SearchSegmentRef,
    SearchSegmentRole, SearchSegmentStats, SearchStatValue,
};
use crate::search_workspace::{estimated_text_result_bytes, shared_search_workspace};
use crate::sst::search_delta::{
    search_suppress_fingerprint, SearchFilterValue, SearchSegmentWireBinding,
    SearchVersionOperation, SearchVersionRangeSource, SearchVersionRecord,
    SearchVersionTableReader, SearchVersionTableRef, SearchVersionTableWriter,
};
use crate::text::{
    avg_len, bm25_idf, bm25_term_score, tokenize, TextQuery, PREFIX_EXPANSION_LIMIT,
};

#[path = "v4_external.rs"]
mod external;

pub use external::{
    ReconciledTextDeltaStats, TextV4ExternalArtifact, TextV4ExternalBuildConfig,
    TextV4ExternalBuildMetrics, TextV4ExternalBuilder,
};

pub const MAGIC_V4: &[u8; 8] = b"NAMIFT04";
const TRAILER_MAGIC: &[u8; 8] = b"NFT4END!";
const TRAILER_LEN: usize = 8 + 8 + 4;
const FORMAT_VERSION: u16 = 4;
const FOOTER_VERSION: u16 = 1;
const DOC_RECORD_LEN: u64 = 16 + 8 + 8 + 4 + 4;
const POSTING_HEADER_LEN: usize = 4;
const POSTING_PREFIX_LEN: usize = 8 + 4 + 4;
const MAX_FOOTER_BYTES: u64 = 128 * 1024 * 1024;
const MAX_COMPRESSED_BLOCK_BYTES: u64 = 256 * 1024 * 1024;
const MAX_RAW_BLOCK_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DOC_BATCH: usize = 512;
const MAX_FILTER_RANGE_BATCH: usize = 64;
const CONTENT_DOMAIN: &[u8] = b"NamiDB/TextV4Content/v1";
const PAYLOAD_FINGERPRINT_DOMAIN: &[u8] = b"NamiDB/TextV4Payload/v1";
const POSTINGS_REGION_DOMAIN: &[u8] = b"NamiDB/TextV4PostingsRegion/v1";

/// One complete live document after-image.
#[derive(Debug, Clone, PartialEq)]
pub struct TextV4Payload {
    pub text: String,
    /// Values for properties advertised as complete by the segment. Missing
    /// means an authoritative empty posting for that row.
    pub filters: BTreeMap<String, SearchFilterValue>,
}

/// One exactly classified before/after text-index mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct TextV4Mutation {
    pub node_id: [u8; 16],
    pub lsn: u64,
    pub before: Option<TextV4Payload>,
    pub after: Option<TextV4Payload>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextV4BuildContext {
    pub sst_id: Uuid,
    pub event_ranges: Vec<SearchEventRange>,
    pub complete_filter_properties: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextV4BuildOptions {
    pub postings_per_block: usize,
    pub terms_per_dictionary_block: usize,
    pub compression_level: i32,
}

impl Default for TextV4BuildOptions {
    fn default() -> Self {
        Self {
            postings_per_block: 256,
            terms_per_dictionary_block: 128,
            compression_level: 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextV4BuildOutput {
    pub segment: SearchSegmentRef,
    pub object_len: u64,
    pub dictionary_block_count: u32,
    pub version_table: SearchVersionTableRef,
}

#[derive(Debug, Clone)]
pub struct TextV4Artifact {
    pub body: Bytes,
    pub output: TextV4BuildOutput,
}

/// Snapshot-wide BM25 statistics supplied by the multi-segment coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextV4GlobalStats {
    pub document_count: u64,
    pub total_document_len: u64,
    pub document_frequency: BTreeMap<String, u64>,
}

/// One authenticated signed document-frequency contribution from an FT4
/// dictionary block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextV4TermDelta {
    pub term: String,
    pub delta_df: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextV4Hit {
    pub node_id: [u8; 16],
    pub lsn: u64,
    pub payload_fingerprint: u64,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextV4SearchResult {
    pub hits: Vec<TextV4Hit>,
    pub applied_filter_groups: usize,
    pub postings_decoded: usize,
    /// Posting bodies fetched and decoded by this query. Dictionary, filter,
    /// and document-table ranges are intentionally accounted separately by
    /// the caller's range source.
    pub posting_blocks_fetched: usize,
    /// Authenticated posting blocks rejected from metadata without fetching
    /// their bodies.
    pub posting_blocks_skipped: usize,
    /// True only when conservative Block-Max pruning was active.
    pub block_max_pruning: bool,
    /// Stable diagnostic for an exact exhaustive fallback selected before any
    /// Block-Max pruning. Corruption remains an error rather than a fallback.
    pub block_max_fallback: Option<&'static str>,
    /// Conservative high-water mark of decoded postings, native-filter state,
    /// and the top-k heap retained simultaneously by this reader.
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
            .ok_or_else(|| Error::invariant("text v4 block range overflows"))?;
        Ok(self.offset..end)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DocTableRef {
    offset: u64,
    len: u64,
    row_count: u64,
    content_xxh3: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RegionRef {
    offset: u64,
    len: u64,
    block_count: u64,
    metadata_xxh3: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PostingBlockRef {
    first_doc: u64,
    last_doc: u64,
    posting_count: u32,
    max_tf: u32,
    min_doc_len: u32,
    wire: BlockRef,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TermEntry {
    term: String,
    delta_df: i64,
    live_doc_freq: u64,
    blocks: Vec<PostingBlockRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DictionaryBlockRef {
    first_term: String,
    last_term: String,
    term_count: u32,
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
    /// Strictly increasing document ordinals encoded as unsigned delta
    /// varints. The first delta is relative to zero.
    SparseDeltaVarint,
    /// One little-endian bit per live document ordinal.
    DenseBitmap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FilterValueRef {
    value: SearchFilterValue,
    cardinality: u64,
    encoding: FilterPostingEncoding,
    wire: BlockRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Footer {
    footer_version: u16,
    binding: SearchSegmentWireBinding,
    delta_docs: i64,
    delta_total_len: i64,
    doc_table: DocTableRef,
    postings_region: RegionRef,
    dictionary: Vec<DictionaryBlockRef>,
    filters: Vec<FilterBlockRef>,
}

#[derive(Debug, Serialize)]
struct ContentDigestMaterial<'a> {
    domain: &'a [u8],
    format_version: u16,
    version_table: &'a SearchVersionTableRef,
    delta_docs: i64,
    delta_total_len: i64,
    doc_table: &'a DocTableRef,
    postings_region: &'a RegionRef,
    dictionary: &'a [DictionaryBlockRef],
    filters: &'a [FilterBlockRef],
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct PreparedPayload {
    tokens: Vec<String>,
    filters: BTreeMap<String, SearchFilterValue>,
}

#[cfg(test)]
#[derive(Debug)]
struct EffectiveMutation {
    node_id: [u8; 16],
    lsn: u64,
    before: Option<PreparedPayload>,
    after: Option<PreparedPayload>,
    payload_fingerprint: u64,
}

#[derive(Debug, Clone)]
struct Posting {
    doc: u64,
    doc_len: u32,
    positions: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct DocRecord {
    node_id: [u8; 16],
    lsn: u64,
    payload_fingerprint: u64,
    doc_len: u32,
}

#[derive(Debug)]
enum QueryMask {
    Sparse(Vec<u64>),
    Dense(Vec<u8>),
}

impl QueryMask {
    fn empty() -> Self {
        Self::Sparse(Vec::new())
    }

    fn contains(&self, ordinal: u64) -> bool {
        match self {
            Self::Sparse(ordinals) => ordinals.binary_search(&ordinal).is_ok(),
            Self::Dense(bitmap) => usize::try_from(ordinal / 8)
                .ok()
                .and_then(|byte| bitmap.get(byte))
                .is_some_and(|byte| byte & (1u8 << (ordinal % 8)) != 0),
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
                        "text v4 dense query filter lengths diverged",
                    ));
                }
                for (target, source) in left.iter_mut().zip(right) {
                    *target &= source;
                }
                QueryMask::maybe_sparsify_dense(left, row_count)
            }
        }
    }

    /// Pick the smaller exact representation without turning a genuinely
    /// sparse result into a corpus-sized bitmap. `ordinals` must already be
    /// strictly increasing.
    fn from_sorted_sparse(ordinals: Vec<u64>, row_count: u64) -> Result<Self> {
        let dense_len = dense_mask_bytes(row_count)?;
        if ordinals.len().saturating_mul(std::mem::size_of::<u64>()) < dense_len {
            return Ok(Self::Sparse(ordinals));
        }
        let mut dense = vec![0u8; dense_len];
        for ordinal in ordinals {
            set_dense_filter_bit_bytes(&mut dense, ordinal, row_count)?;
        }
        Ok(Self::Dense(dense))
    }

    fn maybe_sparsify_dense(dense: Vec<u8>, row_count: u64) -> Result<Self> {
        let cardinality = dense
            .iter()
            .map(|byte| byte.count_ones() as usize)
            .sum::<usize>();
        if cardinality.saturating_mul(std::mem::size_of::<u64>()) >= dense.len() {
            return Ok(Self::Dense(dense));
        }
        let mut sparse = Vec::with_capacity(cardinality);
        for (byte_index, byte) in dense.iter().copied().enumerate() {
            let mut remaining = byte;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let ordinal = byte_index
                    .checked_mul(8)
                    .and_then(|base| base.checked_add(bit))
                    .and_then(|ordinal| u64::try_from(ordinal).ok())
                    .ok_or_else(|| Error::invariant("text v4 sparse mask ordinal overflows"))?;
                if ordinal >= row_count {
                    return Err(Error::invariant(
                        "text v4 dense query filter has bits past row count",
                    ));
                }
                sparse.push(ordinal);
                remaining &= remaining - 1;
            }
        }
        Ok(Self::Sparse(sparse))
    }
}

/// OR accumulator for one native-filter value group.
///
/// Repeatedly merging sorted vectors makes a 10k-value `IN` predicate
/// quadratic. Appending sparse ordinals and sorting once is linearithmic and
/// retains at most the final sparse representation. As soon as that
/// representation would be no smaller than a bitmap, the accumulator switches
/// exactly once to dense and all later postings are folded in place.
#[derive(Debug)]
struct QueryMaskUnionBuilder {
    row_count: u64,
    mask: QueryMask,
}

impl QueryMaskUnionBuilder {
    fn new(row_count: u64) -> Self {
        Self {
            row_count,
            mask: QueryMask::empty(),
        }
    }

    fn resident_bytes(&self) -> usize {
        self.mask.resident_bytes()
    }

    /// Absorb one exact value posting and return a conservative peak for the
    /// two input representations plus any newly allocated output.
    fn absorb(&mut self, incoming: QueryMask) -> Result<usize> {
        let left_bytes = self.mask.resident_bytes();
        let right_bytes = incoming.resident_bytes();
        let dense_len = dense_mask_bytes(self.row_count)?;
        let previous = std::mem::replace(&mut self.mask, QueryMask::empty());
        let (next, output_bytes) = match (previous, incoming) {
            (QueryMask::Sparse(mut left), QueryMask::Sparse(mut right)) => {
                let projected = left.len().saturating_add(right.len());
                if projected.saturating_mul(std::mem::size_of::<u64>()) >= dense_len {
                    let mut dense = vec![0u8; dense_len];
                    for ordinal in left.into_iter().chain(right) {
                        set_dense_filter_bit_bytes(&mut dense, ordinal, self.row_count)?;
                    }
                    (QueryMask::Dense(dense), dense_len)
                } else {
                    left.append(&mut right);
                    let bytes = left.capacity().saturating_mul(std::mem::size_of::<u64>());
                    (QueryMask::Sparse(left), bytes)
                }
            }
            (QueryMask::Dense(mut dense), QueryMask::Sparse(sparse)) => {
                for ordinal in sparse {
                    set_dense_filter_bit_bytes(&mut dense, ordinal, self.row_count)?;
                }
                let bytes = dense.capacity();
                (QueryMask::Dense(dense), bytes)
            }
            (QueryMask::Sparse(sparse), QueryMask::Dense(mut dense)) => {
                for ordinal in sparse {
                    set_dense_filter_bit_bytes(&mut dense, ordinal, self.row_count)?;
                }
                let bytes = dense.capacity();
                (QueryMask::Dense(dense), bytes)
            }
            (QueryMask::Dense(mut left), QueryMask::Dense(right)) => {
                if left.len() != right.len() {
                    return Err(Error::invariant(
                        "text v4 dense query filter lengths diverged",
                    ));
                }
                for (target, source) in left.iter_mut().zip(right) {
                    *target |= source;
                }
                let bytes = left.capacity();
                (QueryMask::Dense(left), bytes)
            }
        };
        self.mask = next;
        Ok(left_bytes
            .saturating_add(right_bytes)
            .saturating_add(output_bytes))
    }

    fn finish(self) -> Result<(QueryMask, usize)> {
        match self.mask {
            QueryMask::Sparse(mut ordinals) => {
                ordinals.sort_unstable();
                ordinals.dedup();
                let sparse_bytes = ordinals
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u64>());
                if ordinals.len().saturating_mul(std::mem::size_of::<u64>())
                    < dense_mask_bytes(self.row_count)?
                {
                    return Ok((QueryMask::Sparse(ordinals), sparse_bytes));
                }
                let dense = QueryMask::from_sorted_sparse(ordinals, self.row_count)?;
                let peak = sparse_bytes.saturating_add(dense.resident_bytes());
                Ok((dense, peak))
            }
            QueryMask::Dense(dense) => {
                let bytes = dense.capacity();
                Ok((QueryMask::Dense(dense), bytes))
            }
        }
    }
}

#[derive(Debug)]
struct PostingCursor {
    term: String,
    idf: f64,
    blocks: Vec<PostingBlockRef>,
    block_upper_bounds: Vec<f64>,
    next_block: usize,
    postings: Vec<Posting>,
    next_posting: usize,
    /// When metadata skips a window ending inside the next block, discard
    /// decoded postings through this ordinal after the block is authenticated.
    skip_through: Option<u64>,
}

impl PostingCursor {
    fn current(&self) -> Option<&Posting> {
        self.postings.get(self.next_posting)
    }

    fn current_block_index(&self) -> Option<usize> {
        if self.current().is_some() {
            self.next_block.checked_sub(1)
        } else {
            self.blocks.get(self.next_block).map(|_| self.next_block)
        }
    }

    fn current_block(&self) -> Option<&PostingBlockRef> {
        self.current_block_index()
            .and_then(|index| self.blocks.get(index))
    }

    fn current_block_upper_bound(&self) -> Option<f64> {
        self.current_block_index()
            .and_then(|index| self.block_upper_bounds.get(index))
            .copied()
    }

    /// Exact current posting when decoded; otherwise a conservative lower
    /// bound for the first posting after a metadata-only skip.
    fn potential_doc(&self) -> Option<u64> {
        if let Some(posting) = self.current() {
            return Some(posting.doc);
        }
        let block = self.blocks.get(self.next_block)?;
        match self.skip_through {
            Some(target) if target >= block.first_doc => target.checked_add(1),
            _ => Some(block.first_doc),
        }
    }

    fn advance_loaded_through(&mut self, target: u64) {
        if self.current().is_none() {
            return;
        }
        let remaining = &self.postings[self.next_posting..];
        self.next_posting = self
            .next_posting
            .saturating_add(remaining.partition_point(|posting| posting.doc <= target));
        if self.next_posting == self.postings.len() {
            self.postings = Vec::new();
            self.next_posting = 0;
        }
    }

    fn skip_unloaded_through(
        &mut self,
        target: u64,
        posting_blocks_skipped: &mut usize,
    ) -> Result<()> {
        if self.current().is_some() {
            return Ok(());
        }
        while self
            .blocks
            .get(self.next_block)
            .is_some_and(|block| block.last_doc <= target)
        {
            self.next_block += 1;
            *posting_blocks_skipped = posting_blocks_skipped
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 skipped block count overflows"))?;
        }
        self.skip_through = self
            .blocks
            .get(self.next_block)
            .filter(|block| block.first_doc <= target)
            .map(|_| target);
        Ok(())
    }

    fn resident_bytes(&self) -> usize {
        self.term
            .capacity()
            .saturating_add(
                self.blocks
                    .capacity()
                    .saturating_mul(std::mem::size_of::<PostingBlockRef>()),
            )
            .saturating_add(
                self.block_upper_bounds
                    .capacity()
                    .saturating_mul(std::mem::size_of::<f64>()),
            )
            .saturating_add(
                self.postings
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Posting>())
                    .saturating_add(
                        self.postings
                            .iter()
                            .map(|posting| {
                                posting
                                    .positions
                                    .capacity()
                                    .saturating_mul(std::mem::size_of::<u32>())
                            })
                            .sum::<usize>(),
                    ),
            )
    }
}

fn next_up_nonnegative(value: f64) -> Option<f64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    if value == 0.0 {
        return Some(f64::from_bits(1));
    }
    let next = f64::from_bits(value.to_bits().checked_add(1)?);
    next.is_finite().then_some(next)
}

fn conservative_block_upper_bound(
    idf: f64,
    block: &PostingBlockRef,
    corpus_avg_len: f64,
) -> Option<f64> {
    if !idf.is_finite()
        || idf < 0.0
        || !corpus_avg_len.is_finite()
        || corpus_avg_len <= 0.0
        || block.max_tf == 0
        || block.min_doc_len == 0
    {
        return None;
    }
    next_up_nonnegative(bm25_term_score(
        idf,
        block.max_tf,
        block.min_doc_len as usize,
        corpus_avg_len,
    ))
}

fn conservative_upper_bound_sum<'a>(values: impl IntoIterator<Item = &'a f64>) -> Option<f64> {
    let mut total = 0.0;
    for value in values {
        if !value.is_finite() || *value < 0.0 {
            return None;
        }
        total = next_up_nonnegative(total + *value)?;
    }
    Some(total)
}

/// One head in the exact document-at-a-time postings merge. Ordering is
/// reversed so Rust's max-heap exposes the smallest `(doc, cursor)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CursorHead {
    doc: u64,
    cursor: usize,
}

impl PartialOrd for CursorHead {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CursorHead {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .doc
            .cmp(&self.doc)
            .then_with(|| other.cursor.cmp(&self.cursor))
    }
}

/// Metadata-only/range-readable FT4 reader.
pub struct TextV4Reader {
    source: Arc<dyn SearchVersionRangeSource>,
    file_len: u64,
    footer_offset: u64,
    footer: Footer,
    version_reader: SearchVersionTableReader,
}

impl std::fmt::Debug for TextV4Reader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextV4Reader")
            .field("file_len", &self.file_len)
            .field("footer_offset", &self.footer_offset)
            .field("documents", &self.footer.doc_table.row_count)
            .field("dictionary_blocks", &self.footer.dictionary.len())
            .field("filters", &self.footer.filters.len())
            .field("segment", &self.footer.binding.segment.sst_id)
            .finish()
    }
}

/// Canonical fingerprint over analyzed tokens and complete filter values.
pub fn text_v4_payload_fingerprint(payload: &TextV4Payload) -> Result<u64> {
    let prepared = prepare_payload(payload)?;
    prepared_payload_fingerprint(&prepared)
}

/// In-memory convenience builder. Production flushes should use
/// [`write_delta_v4`] with a file-backed spool.
pub fn build_delta_v4(
    state: &SearchLsmState,
    context: TextV4BuildContext,
    mutations: Vec<TextV4Mutation>,
    options: TextV4BuildOptions,
) -> Result<Option<TextV4Artifact>> {
    let cursor = Cursor::new(Vec::new());
    let Some((cursor, output)) = write_delta_v4(cursor, state, context, mutations, options)? else {
        return Ok(None);
    };
    Ok(Some(TextV4Artifact {
        body: Bytes::from(cursor.into_inner()),
        output,
    }))
}

/// In-memory convenience builder for one authoritative FT4 base.
///
/// Every mutation must be a live after-image (`before = None`,
/// `after = Some(_)`). Production compaction should prefer
/// [`TextV4ExternalBuilder::new_base`] and stream NodeId-sorted winners.
pub fn build_base_v4(
    state: &SearchLsmState,
    context: TextV4BuildContext,
    documents: Vec<TextV4Mutation>,
    options: TextV4BuildOptions,
) -> Result<Option<TextV4Artifact>> {
    let cursor = Cursor::new(Vec::new());
    let Some((cursor, output)) = write_base_v4(cursor, state, context, documents, options)? else {
        return Ok(None);
    };
    Ok(Some(TextV4Artifact {
        body: Bytes::from(cursor.into_inner()),
        output,
    }))
}

/// Bounded convenience wrapper over [`TextV4ExternalBuilder`].
///
/// The caller-owned `Vec` is capped before building. Production flush paths
/// should push one sorted mutation at a time into the external builder and
/// avoid materializing that input vector entirely.
pub fn write_delta_v4<W: Write + Seek>(
    mut writer: W,
    state: &SearchLsmState,
    context: TextV4BuildContext,
    mut mutations: Vec<TextV4Mutation>,
    options: TextV4BuildOptions,
) -> Result<Option<(W, TextV4BuildOutput)>> {
    if writer.stream_position()? != 0 {
        return Err(Error::invariant(
            "text v4 object writer must start at offset zero",
        ));
    }
    let config = TextV4ExternalBuildConfig {
        wire: options,
        ..TextV4ExternalBuildConfig::default()
    };
    let input_bytes = estimated_mutation_input_bytes(&mutations)?;
    if input_bytes > config.memory_budget_bytes {
        return Err(Error::precondition(format!(
            "text v4 convenience input requires {input_bytes} bytes, above its {}-byte cap; \
             use TextV4ExternalBuilder::push",
            config.memory_budget_bytes
        )));
    }
    mutations.sort_by_key(|mutation| mutation.node_id);
    let mut builder = TextV4ExternalBuilder::with_config(state, context, config)?;
    for mutation in mutations {
        builder.push(mutation)?;
    }
    copy_external_artifact(writer, builder)
}

/// Bounded convenience wrapper for an authoritative FT4 base.
///
/// The input contract is the same as [`build_base_v4`]. Production compaction
/// should stream directly into [`TextV4ExternalBuilder::with_config_base`].
pub fn write_base_v4<W: Write + Seek>(
    writer: W,
    state: &SearchLsmState,
    context: TextV4BuildContext,
    mut documents: Vec<TextV4Mutation>,
    options: TextV4BuildOptions,
) -> Result<Option<(W, TextV4BuildOutput)>> {
    let config = TextV4ExternalBuildConfig {
        wire: options,
        ..TextV4ExternalBuildConfig::default()
    };
    let input_bytes = estimated_mutation_input_bytes(&documents)?;
    if input_bytes > config.memory_budget_bytes {
        return Err(Error::precondition(format!(
            "text v4 convenience input requires {input_bytes} bytes, above its {}-byte cap; \
             use TextV4ExternalBuilder::push",
            config.memory_budget_bytes
        )));
    }
    documents.sort_by_key(|mutation| mutation.node_id);
    let mut builder = TextV4ExternalBuilder::with_config_base(state, context, config)?;
    for document in documents {
        builder.push(document)?;
    }
    copy_external_artifact(writer, builder)
}

fn copy_external_artifact<W: Write + Seek>(
    mut writer: W,
    builder: TextV4ExternalBuilder,
) -> Result<Option<(W, TextV4BuildOutput)>> {
    if writer.stream_position()? != 0 {
        return Err(Error::invariant(
            "text v4 object writer must start at offset zero",
        ));
    }
    let Some(mut artifact) = builder.finish()? else {
        return Ok(None);
    };
    artifact.file.rewind()?;
    let copied = std::io::copy(&mut artifact.file, &mut writer)?;
    if copied != artifact.len || writer.stream_position()? != artifact.len {
        return Err(Error::invariant(
            "text v4 external artifact copy length changed",
        ));
    }
    Ok(Some((writer, artifact.output)))
}

fn estimated_mutation_input_bytes(mutations: &[TextV4Mutation]) -> Result<usize> {
    mutations.iter().try_fold(
        mutations
            .len()
            .saturating_mul(std::mem::size_of::<TextV4Mutation>()),
        |total, mutation| {
            [mutation.before.as_ref(), mutation.after.as_ref()]
                .into_iter()
                .flatten()
                .try_fold(total, |total, payload| {
                    let filters =
                        payload
                            .filters
                            .iter()
                            .try_fold(0usize, |bytes, (property, value)| {
                                bytes
                                    .checked_add(property.capacity())
                                    .and_then(|bytes| {
                                        bytes.checked_add(std::mem::size_of::<SearchFilterValue>())
                                    })
                                    .and_then(|bytes| {
                                        bytes.checked_add(match value {
                                            SearchFilterValue::String(value) => value.capacity(),
                                            SearchFilterValue::Bytes(value) => value.capacity(),
                                            _ => 0,
                                        })
                                    })
                                    .ok_or_else(|| {
                                        Error::precondition(
                                            "text v4 convenience input accounting overflows",
                                        )
                                    })
                            })?;
                    total
                        .checked_add(payload.text.capacity())
                        .and_then(|bytes| bytes.checked_add(filters))
                        .ok_or_else(|| {
                            Error::precondition("text v4 convenience input accounting overflows")
                        })
                })
        },
    )
}

#[cfg(test)]
fn write_delta_v4_in_memory<W: Write + Seek>(
    mut writer: W,
    state: &SearchLsmState,
    mut context: TextV4BuildContext,
    mut mutations: Vec<TextV4Mutation>,
    options: TextV4BuildOptions,
) -> Result<Option<(W, TextV4BuildOutput)>> {
    validate_build_configuration(&mut writer, state, &mut context, options)?;
    mutations.sort_by_key(|mutation| mutation.node_id);
    if mutations
        .windows(2)
        .any(|pair| pair[0].node_id == pair[1].node_id)
    {
        return Err(Error::invariant(
            "text v4 delta contains duplicate NodeIds; reconcile before building",
        ));
    }

    let mut effective = Vec::with_capacity(mutations.len());
    for mutation in mutations {
        if mutation.lsn == 0 {
            return Err(Error::invariant("text v4 mutation uses reserved LSN zero"));
        }
        let before = mutation.before.as_ref().map(prepare_payload).transpose()?;
        let after = mutation.after.as_ref().map(prepare_payload).transpose()?;
        if before.as_ref().is_some_and(|payload| {
            payload.filters.keys().any(|property| {
                context
                    .complete_filter_properties
                    .binary_search(property)
                    .is_err()
            })
        }) || after.as_ref().is_some_and(|payload| {
            payload.filters.keys().any(|property| {
                context
                    .complete_filter_properties
                    .binary_search(property)
                    .is_err()
            })
        }) {
            return Err(Error::invariant(
                "text v4 payload contains an unadvertised native-filter property",
            ));
        }
        if before == after {
            continue;
        }
        let payload_fingerprint = match &after {
            Some(after) => prepared_payload_fingerprint(after)?,
            None => search_suppress_fingerprint(),
        };
        effective.push(EffectiveMutation {
            node_id: mutation.node_id,
            lsn: mutation.lsn,
            before,
            after,
            payload_fingerprint,
        });
    }
    if effective.is_empty() {
        return Ok(None);
    }

    writer.write_all(MAGIC_V4)?;
    let mut version_writer = SearchVersionTableWriter::new(writer)?;
    let mut live_ordinal = 0u64;
    for mutation in &effective {
        let record = if mutation.after.is_some() {
            let record = SearchVersionRecord::live(
                mutation.node_id,
                mutation.lsn,
                mutation.payload_fingerprint,
                live_ordinal,
            );
            live_ordinal = live_ordinal
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 live ordinal overflows"))?;
            record
        } else {
            SearchVersionRecord::suppress(
                mutation.node_id,
                mutation.lsn,
                mutation.payload_fingerprint,
            )
        };
        version_writer.push(record)?;
    }
    let (mut writer, version_table) = version_writer.finish()?;

    let (doc_table, prepared_postings) =
        write_doc_table_and_collect_postings(&mut writer, &effective, live_ordinal)?;
    let postings_start = writer.stream_position()?;
    let mut term_entries = Vec::with_capacity(prepared_postings.len());
    let mut postings_metadata_hasher = Xxh3::new();
    postings_metadata_hasher.update(POSTINGS_REGION_DOMAIN);
    let mut posting_block_count = 0u64;
    for (term, term_data) in prepared_postings {
        let mut blocks = Vec::new();
        for chunk in term_data.postings.chunks(options.postings_per_block) {
            let raw = encode_posting_block(chunk)?;
            let wire = write_compressed_block(
                &mut writer,
                &raw,
                options.compression_level,
                "posting block",
            )?;
            let reference = PostingBlockRef {
                first_doc: chunk.first().map(|posting| posting.doc).unwrap_or(0),
                last_doc: chunk.last().map(|posting| posting.doc).unwrap_or(0),
                posting_count: u32::try_from(chunk.len())
                    .map_err(|_| Error::invariant("text v4 posting block exceeds u32"))?,
                max_tf: chunk
                    .iter()
                    .map(|posting| posting.positions.len() as u32)
                    .max()
                    .unwrap_or(0),
                min_doc_len: chunk
                    .iter()
                    .map(|posting| posting.doc_len)
                    .min()
                    .unwrap_or(0),
                wire,
            };
            postings_metadata_hasher.update(&serialize_bounded(
                &reference,
                MAX_RAW_BLOCK_BYTES,
                "posting metadata digest",
            )?);
            posting_block_count = posting_block_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 posting block count overflows"))?;
            blocks.push(reference);
        }
        term_entries.push(TermEntry {
            term,
            delta_df: term_data.delta_df,
            live_doc_freq: term_data.postings.len() as u64,
            blocks,
        });
    }
    let postings_end = writer.stream_position()?;
    let postings_region = RegionRef {
        offset: postings_start,
        len: postings_end
            .checked_sub(postings_start)
            .ok_or_else(|| Error::invariant("text v4 postings region underflows"))?,
        block_count: posting_block_count,
        metadata_xxh3: non_zero_digest(postings_metadata_hasher),
    };

    let mut dictionary = Vec::new();
    for terms in term_entries.chunks(options.terms_per_dictionary_block) {
        let raw = serialize_bounded(terms, MAX_RAW_BLOCK_BYTES, "dictionary block")?;
        dictionary.push(DictionaryBlockRef {
            first_term: terms
                .first()
                .map(|entry| entry.term.clone())
                .ok_or_else(|| Error::invariant("text v4 dictionary chunk is empty"))?,
            last_term: terms
                .last()
                .map(|entry| entry.term.clone())
                .ok_or_else(|| Error::invariant("text v4 dictionary chunk is empty"))?,
            term_count: u32::try_from(terms.len())
                .map_err(|_| Error::invariant("text v4 dictionary block exceeds u32"))?,
            wire: write_compressed_block(
                &mut writer,
                &raw,
                options.compression_level,
                "dictionary block",
            )?,
        });
    }

    let mut filters = Vec::with_capacity(context.complete_filter_properties.len());
    for property in &context.complete_filter_properties {
        let postings = build_filter_postings(property, live_ordinal, &effective)?;
        let mut values = Vec::with_capacity(postings.len());
        for (value, bitmap) in postings {
            let (encoding, raw, cardinality) = filter_posting_wire(&bitmap, live_ordinal)?;
            values.push(FilterValueRef {
                value,
                cardinality,
                encoding,
                wire: write_compressed_block(
                    &mut writer,
                    &raw,
                    options.compression_level,
                    "filter posting block",
                )?,
            });
        }
        filters.push(FilterBlockRef {
            property: property.clone(),
            row_count: live_ordinal,
            values,
        });
    }

    let delta_docs = effective.iter().try_fold(0i64, |total, mutation| {
        let after = if mutation.after.is_some() { 1i64 } else { 0 };
        let before = if mutation.before.is_some() { 1i64 } else { 0 };
        let delta = after - before;
        total
            .checked_add(delta)
            .ok_or_else(|| Error::invariant("text v4 document delta overflows"))
    })?;
    let delta_total_len = effective.iter().try_fold(0i64, |total, mutation| {
        let before = mutation
            .before
            .as_ref()
            .map(|payload| payload.tokens.len())
            .unwrap_or(0);
        let after = mutation
            .after
            .as_ref()
            .map(|payload| payload.tokens.len())
            .unwrap_or(0);
        let before = i64::try_from(before)
            .map_err(|_| Error::invariant("text v4 before length exceeds i64"))?;
        let after = i64::try_from(after)
            .map_err(|_| Error::invariant("text v4 after length exceeds i64"))?;
        total
            .checked_add(after - before)
            .ok_or_else(|| Error::invariant("text v4 total-length delta overflows"))
    })?;
    let content_xxh3 = content_digest(
        &version_table,
        delta_docs,
        delta_total_len,
        &doc_table,
        &postings_region,
        &dictionary,
        &filters,
    )?;
    let min_lsn = effective
        .iter()
        .map(|mutation| mutation.lsn)
        .min()
        .ok_or_else(|| Error::invariant("text v4 effective delta unexpectedly empty"))?;
    let max_lsn = effective
        .iter()
        .map(|mutation| mutation.lsn)
        .max()
        .ok_or_else(|| Error::invariant("text v4 effective delta unexpectedly empty"))?;
    let segment = SearchSegmentRef {
        sst_id: context.sst_id,
        role: SearchSegmentRole::Delta,
        format: SearchSegmentFormat::TextV4,
        payload: SearchSegmentPayload::Complete,
        event_ranges: context.event_ranges,
        min_lsn,
        max_lsn,
        mutation_count: effective.len() as u64,
        live_payload_count: live_ordinal,
        suppress_count: effective.len() as u64 - live_ordinal,
        content_xxh3,
        complete_filter_properties: context.complete_filter_properties,
        stats: SearchSegmentStats::Text {
            doc_count: SearchStatValue::Delta(delta_docs),
            total_len: SearchStatValue::Delta(delta_total_len),
            term_df_violation_count: 0,
        },
        equal_lsn_conflict_count: 0,
    };
    let binding = SearchSegmentWireBinding::new(state, &segment, version_table.clone())?;
    let footer = Footer {
        footer_version: FOOTER_VERSION,
        binding,
        delta_docs,
        delta_total_len,
        doc_table,
        postings_region,
        dictionary,
        filters,
    };
    let footer_bytes = serialize_bounded(&footer, MAX_FOOTER_BYTES, "text v4 footer")?;
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
            .ok_or_else(|| Error::invariant("text v4 object length overflows"))?
    {
        return Err(Error::invariant(
            "text v4 final writer position is inconsistent",
        ));
    }
    Ok(Some((
        writer,
        TextV4BuildOutput {
            segment,
            object_len,
            dictionary_block_count: u32::try_from(footer.dictionary.len())
                .map_err(|_| Error::invariant("text v4 dictionary block count exceeds u32"))?,
            version_table,
        },
    )))
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TermBuildData {
    delta_df: i64,
    postings: Vec<Posting>,
}

#[cfg(test)]
fn write_doc_table_and_collect_postings<W: Write + Seek>(
    writer: &mut W,
    effective: &[EffectiveMutation],
    live_count: u64,
) -> Result<(DocTableRef, BTreeMap<String, TermBuildData>)> {
    let offset = writer.stream_position()?;
    let mut hasher = Xxh3::new();
    let mut terms = BTreeMap::<String, TermBuildData>::new();
    let mut ordinal = 0u64;
    for mutation in effective {
        if let Some(before) = &mutation.before {
            let mut unique_terms = before.tokens.iter().collect::<Vec<_>>();
            unique_terms.sort_unstable();
            unique_terms.dedup();
            for term in unique_terms {
                let entry = terms.entry(term.clone()).or_default();
                entry.delta_df = entry
                    .delta_df
                    .checked_sub(1)
                    .ok_or_else(|| Error::invariant("text v4 term delta_df underflows"))?;
            }
        }
        let Some(after) = &mutation.after else {
            continue;
        };
        let doc_len = u32::try_from(after.tokens.len())
            .map_err(|_| Error::invariant("text v4 document token count exceeds u32"))?;
        let record = DocRecord {
            node_id: mutation.node_id,
            lsn: mutation.lsn,
            payload_fingerprint: mutation.payload_fingerprint,
            doc_len,
        };
        let encoded = encode_doc_record(record);
        writer.write_all(&encoded)?;
        hasher.update(&encoded);

        let mut positions = BTreeMap::<String, Vec<u32>>::new();
        for (position, term) in after.tokens.iter().enumerate() {
            positions.entry(term.clone()).or_default().push(
                u32::try_from(position)
                    .map_err(|_| Error::invariant("text v4 token position exceeds u32"))?,
            );
        }
        for (term, positions) in positions {
            let entry = terms.entry(term).or_default();
            entry.delta_df = entry
                .delta_df
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 term delta_df overflows"))?;
            entry.postings.push(Posting {
                doc: ordinal,
                doc_len,
                positions,
            });
        }
        ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text v4 document ordinal overflows"))?;
    }
    if ordinal != live_count {
        return Err(Error::invariant(
            "text v4 document table/version ordinals diverged",
        ));
    }
    terms.retain(|_, entry| entry.delta_df != 0 || !entry.postings.is_empty());
    let len = live_count
        .checked_mul(DOC_RECORD_LEN)
        .ok_or_else(|| Error::invariant("text v4 document table length overflows"))?;
    if writer.stream_position()? != offset + len {
        return Err(Error::invariant(
            "text v4 document table writer position is inconsistent",
        ));
    }
    Ok((
        DocTableRef {
            offset,
            len,
            row_count: live_count,
            content_xxh3: non_zero_digest(hasher),
        },
        terms,
    ))
}

impl TextV4Reader {
    pub async fn open(
        source: Arc<dyn SearchVersionRangeSource>,
        file_len: u64,
        state: &SearchLsmState,
        segment: &SearchSegmentRef,
    ) -> Result<Self> {
        let minimum = (MAGIC_V4.len() + TRAILER_LEN) as u64;
        if file_len < minimum {
            return Err(Error::invariant("text v4 body is too short"));
        }
        let trailer_start = file_len - TRAILER_LEN as u64;
        let probes = source
            .read_ranges(&[0..MAGIC_V4.len() as u64, trailer_start..file_len])
            .await?;
        if probes.len() != 2 || probes[0].as_ref() != MAGIC_V4 || probes[1].len() != TRAILER_LEN {
            return Err(Error::invariant(
                "text v4 magic/trailer range probes are malformed",
            ));
        }
        let (footer_len, footer_crc) = decode_trailer(&probes[1])?;
        if footer_len == 0 || footer_len > MAX_FOOTER_BYTES {
            return Err(Error::invariant("text v4 footer length is invalid"));
        }
        let footer_offset = trailer_start
            .checked_sub(footer_len)
            .ok_or_else(|| Error::invariant("text v4 footer starts before object"))?;
        let footer_bytes = source.read_range(footer_offset..trailer_start).await?;
        require_len(
            &footer_bytes,
            usize::try_from(footer_len)
                .map_err(|_| Error::invariant("text v4 footer does not fit usize"))?,
            "footer",
        )?;
        if crc32fast::hash(&footer_bytes) != footer_crc {
            return Err(Error::invariant("text v4 footer checksum mismatch"));
        }
        let footer: Footer =
            deserialize_bounded(&footer_bytes, MAX_FOOTER_BYTES, "text v4 footer")?;
        validate_footer(&footer, footer_offset, state, segment)?;
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

    pub fn delta_docs(&self) -> i64 {
        self.footer.delta_docs
    }

    pub fn delta_total_len(&self) -> i64 {
        self.footer.delta_total_len
    }

    pub fn live_document_count(&self) -> u64 {
        self.footer.doc_table.row_count
    }

    /// Number of independently authenticated sparse dictionary blocks.
    ///
    /// Callers can k-way merge block-at-a-time term statistics across FT4
    /// runs without retaining a generation-wide vocabulary.
    pub fn term_delta_block_count(&self) -> usize {
        self.footer.dictionary.len()
    }

    /// Decode one independently checksummed dictionary block as signed term
    /// statistics.
    ///
    /// Opening the reader has already authenticated and bound the footer to
    /// the manifest segment. This call additionally authenticates,
    /// decompresses and structurally validates only the selected dictionary
    /// block. The returned allocation is therefore bounded by one FT4 block,
    /// never by the complete vocabulary.
    pub async fn read_term_delta_block(&self, block: usize) -> Result<Vec<TextV4TermDelta>> {
        let reference = self.footer.dictionary.get(block).ok_or_else(|| {
            Error::precondition(format!(
                "text v4 dictionary block {block} leaves the {}-block directory",
                self.footer.dictionary.len()
            ))
        })?;
        self.read_dictionary_block(reference).await.map(|entries| {
            entries
                .into_iter()
                .map(|entry| TextV4TermDelta {
                    term: entry.term,
                    delta_df: entry.delta_df,
                })
                .collect()
        })
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
                    .dictionary
                    .iter()
                    .map(|block| {
                        std::mem::size_of::<DictionaryBlockRef>()
                            + block.first_term.capacity()
                            + block.last_term.capacity()
                    })
                    .sum::<usize>(),
            )
            .saturating_add(
                self.footer
                    .filters
                    .iter()
                    .map(|filter| {
                        std::mem::size_of::<FilterBlockRef>()
                            + filter.property.capacity()
                            + filter
                                .values
                                .iter()
                                .map(|value| {
                                    std::mem::size_of::<FilterValueRef>()
                                        + filter_value_resident_bytes(&value.value)
                                })
                                .sum::<usize>()
                    })
                    .sum::<usize>(),
            )
            .saturating_add(self.version_reader.resident_metadata_bytes())
    }

    /// Exact role-relative document-frequency contribution for one term.
    ///
    /// The value is an absolute, non-negative frequency for a base segment and
    /// a signed contribution for a delta segment. The historical method name
    /// remains for source compatibility.
    pub async fn term_delta_df(&self, term: &str) -> Result<i64> {
        Ok(self
            .lookup_term(term)
            .await?
            .map(|entry| entry.delta_df)
            .unwrap_or(0))
    }

    /// Expand one prefix from this segment's sparse dictionary, retaining at
    /// most `limit` lexicographically smallest terms.
    ///
    /// Multi-segment coordinators call this on every FT4 segment, merge those
    /// already-bounded sorted lists, and apply the query-wide prefix cap. The
    /// explicit limit prevents a caller from turning dictionary planning into
    /// a vocabulary scan.
    pub async fn expand_prefix_terms(&self, prefix: &str, limit: usize) -> Result<Vec<String>> {
        if limit > PREFIX_EXPANSION_LIMIT {
            return Err(Error::precondition(format!(
                "text v4 prefix expansion limit {limit} exceeds {PREFIX_EXPANSION_LIMIT}"
            )));
        }
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut terms = self.expand_prefix(prefix).await?;
        terms.truncate(limit);
        Ok(terms)
    }

    /// Exhaustive FT4 scoring with caller-supplied snapshot-wide BM25 stats.
    ///
    /// All live postings in this delta are consumed, so there is no unsafe
    /// segment-local IDF or unproven early termination.
    pub async fn search_query_exact(
        &self,
        query: &TextQuery,
        global: &TextV4GlobalStats,
        k: usize,
        groups: &[(String, Vec<SearchFilterValue>)],
    ) -> Result<TextV4SearchResult> {
        if k == 0 || query.is_empty() || self.footer.doc_table.row_count == 0 {
            return Ok(TextV4SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                postings_decoded: 0,
                posting_blocks_fetched: 0,
                posting_blocks_skipped: 0,
                block_max_pruning: false,
                block_max_fallback: None,
                peak_live_bytes: 0,
            });
        }
        validate_global_stats(global)?;
        let mut scored_terms = query
            .base_terms()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for prefix in &query.prefixes {
            scored_terms.extend(self.expand_prefix(prefix).await?);
        }
        scored_terms.sort();
        scored_terms.dedup();
        let mut entries = BTreeMap::<String, TermEntry>::new();
        for term in &scored_terms {
            if let Some(entry) = self.lookup_term(term).await? {
                entries.insert(term.clone(), entry);
            }
        }
        if query
            .phrases
            .iter()
            .flatten()
            .any(|term| !entries.contains_key(term))
            || entries.is_empty()
        {
            return Ok(TextV4SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                postings_decoded: 0,
                posting_blocks_fetched: 0,
                posting_blocks_skipped: 0,
                block_max_pruning: false,
                block_max_fallback: None,
                peak_live_bytes: 0,
            });
        }

        let retained_k = k.min(
            usize::try_from(self.footer.doc_table.row_count)
                .map_err(|_| Error::invariant("text v4 row count does not fit usize"))?,
        );
        let workspace_bytes =
            estimate_exact_search_workspace(&entries, groups, &self.footer.filters, retained_k)?;
        let _workspace = shared_search_workspace()
            .reserve("text v4 exact search", workspace_bytes)
            .await?;
        let (allowed, applied_filter_groups, filter_peak_bytes) =
            self.load_filter_mask(groups).await?;
        let n_docs = usize::try_from(global.document_count)
            .map_err(|_| Error::invariant("global BM25 document count does not fit usize"))?;
        let corpus_avg_len = avg_len(global.total_document_len, n_docs);
        let mut idfs = BTreeMap::new();
        for term in &scored_terms {
            let df = *global.document_frequency.get(term).ok_or_else(|| {
                Error::invariant(format!("global BM25 statistics omit query term {term:?}"))
            })?;
            if df > global.document_count {
                return Err(Error::invariant(format!(
                    "global BM25 df for {term:?} exceeds document count"
                )));
            }
            idfs.insert(
                term.clone(),
                bm25_idf(
                    n_docs,
                    usize::try_from(df)
                        .map_err(|_| Error::invariant("global BM25 df does not fit usize"))?,
                ),
            );
        }

        let mut cursors = entries
            .into_iter()
            .map(|(term, entry)| {
                let idf = *idfs
                    .get(&term)
                    .ok_or_else(|| Error::invariant("text v4 IDF cache lost a term"))?;
                Ok(PostingCursor {
                    term,
                    idf,
                    blocks: entry.blocks,
                    block_upper_bounds: Vec::new(),
                    next_block: 0,
                    postings: Vec::new(),
                    next_posting: 0,
                    skip_through: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        drop(idfs);
        drop(scored_terms);
        let mut postings_decoded = 0usize;
        let mut posting_blocks_fetched = 0usize;
        let mask_bytes = allowed.as_ref().map(QueryMask::resident_bytes).unwrap_or(0);
        let mut peak_live_bytes = filter_peak_bytes.max(mask_bytes);
        let initial_cursors = (0..cursors.len()).collect::<Vec<_>>();
        let transient = self
            .load_next_cursor_blocks(
                &mut cursors,
                &initial_cursors,
                &mut postings_decoded,
                &mut posting_blocks_fetched,
            )
            .await?;
        peak_live_bytes = peak_live_bytes.max(
            cursor_live_bytes(&cursors)
                .saturating_add(mask_bytes)
                .saturating_add(
                    initial_cursors
                        .capacity()
                        .saturating_mul(std::mem::size_of::<usize>()),
                )
                .saturating_add(transient),
        );
        drop(initial_cursors);

        // Exact OR across scored terms. One heap head per cursor replaces the
        // O(terms) minimum scan at every candidate; phrase terms impose their
        // exact positional AND below. We deliberately do not apply block-max
        // skipping because FT4 has no cross-term upper bound proof yet.
        let mut merge_heap = BinaryHeap::with_capacity(cursors.len());
        for (cursor, state) in cursors.iter().enumerate() {
            if let Some(posting) = state.current() {
                merge_heap.push(CursorHead {
                    doc: posting.doc,
                    cursor,
                });
            }
        }
        let mut matched_cursors = Vec::with_capacity(cursors.len());
        let mut top_k = BinaryHeap::with_capacity(retained_k.saturating_add(1));
        let mut documents_evaluated = 0usize;
        peak_live_bytes = peak_live_bytes.max(exact_query_live_bytes(
            &cursors,
            allowed.as_ref(),
            &merge_heap,
            &matched_cursors,
            &top_k,
        ));

        while let Some(head) = merge_heap.pop() {
            documents_evaluated = documents_evaluated.saturating_add(1);
            if documents_evaluated & (crate::cancel::CHECK_STRIDE - 1) == 0 {
                crate::cancel::check()?;
            }
            let doc = head.doc;
            matched_cursors.clear();
            matched_cursors.push(head.cursor);
            while merge_heap.peek().is_some_and(|head| head.doc == doc) {
                matched_cursors.push(
                    merge_heap
                        .pop()
                        .expect("peeked exact text merge head")
                        .cursor,
                );
            }
            debug_assert!(
                matched_cursors.windows(2).all(|pair| pair[0] < pair[1]),
                "equal-doc cursor heads must retain lexical term order"
            );
            if allowed.as_ref().is_some_and(|mask| !mask.contains(doc))
                || !phrases_match_cursors(doc, &query.phrases, &cursors)
            {
                // Every cursor positioned at this candidate must advance even
                // when a native filter or phrase rejects it.
            } else {
                let mut score = 0.0f64;
                for &cursor_index in &matched_cursors {
                    let cursor = &cursors[cursor_index];
                    let posting = cursor
                        .current()
                        .expect("merge heap head must name a current posting");
                    debug_assert_eq!(posting.doc, doc);
                    score += bm25_term_score(
                        cursor.idf,
                        posting.positions.len() as u32,
                        posting.doc_len as usize,
                        corpus_avg_len,
                    );
                }
                if score > 0.0 && score.is_finite() {
                    top_k.push(RankedOrdinal { doc, score });
                    if top_k.len() > retained_k {
                        top_k.pop();
                    }
                };
            }

            let mut next_blocks = Vec::new();
            for &cursor_index in &matched_cursors {
                let next_doc = {
                    let cursor = &mut cursors[cursor_index];
                    cursor.next_posting += 1;
                    if cursor.next_posting == cursor.postings.len() {
                        next_blocks.push(cursor_index);
                        None
                    } else {
                        cursor.current().map(|posting| posting.doc)
                    }
                };
                if let Some(doc) = next_doc {
                    merge_heap.push(CursorHead {
                        doc,
                        cursor: cursor_index,
                    });
                }
            }
            let transient = self
                .load_next_cursor_blocks(
                    &mut cursors,
                    &next_blocks,
                    &mut postings_decoded,
                    &mut posting_blocks_fetched,
                )
                .await?;
            for &cursor_index in &next_blocks {
                if let Some(posting) = cursors[cursor_index].current() {
                    merge_heap.push(CursorHead {
                        doc: posting.doc,
                        cursor: cursor_index,
                    });
                }
            }
            peak_live_bytes = peak_live_bytes.max(
                exact_query_live_bytes(
                    &cursors,
                    allowed.as_ref(),
                    &merge_heap,
                    &matched_cursors,
                    &top_k,
                )
                .saturating_add(
                    next_blocks
                        .capacity()
                        .saturating_mul(std::mem::size_of::<usize>()),
                )
                .saturating_add(transient),
            );
        }
        drop(cursors);
        drop(allowed);
        drop(merge_heap);
        drop(matched_cursors);

        let mut ranked = top_k.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc.cmp(&right.doc))
        });
        peak_live_bytes = peak_live_bytes.max(estimated_text_result_bytes(ranked.len()));
        let records = self
            .read_doc_records(&ranked.iter().map(|value| value.doc).collect::<Vec<_>>())
            .await?;
        let hits = ranked
            .into_iter()
            .zip(records)
            .map(|(ranked, record)| TextV4Hit {
                node_id: record.node_id,
                lsn: record.lsn,
                payload_fingerprint: record.payload_fingerprint,
                score: ranked.score,
            })
            .collect();
        if peak_live_bytes > workspace_bytes {
            return Err(Error::invariant(format!(
                "text v4 exact search used {peak_live_bytes} modeled bytes above its \
                 {workspace_bytes}-byte reservation"
            )));
        }
        Ok(TextV4SearchResult {
            hits,
            applied_filter_groups,
            postings_decoded,
            posting_blocks_fetched,
            posting_blocks_skipped: 0,
            block_max_pruning: false,
            block_max_fallback: None,
            peak_live_bytes,
        })
    }

    /// Exact top-k scorer for one authoritative FT4 base.
    ///
    /// Posting-block metadata provides conservative BM25 upper bounds. A
    /// block is skipped only when the upward-rounded sum is strictly below
    /// the current worst retained score; equality is never pruned because a
    /// lower document ordinal can win the deterministic tie-break. Any
    /// numerically unsafe bound selects the exhaustive scorer before pruning.
    pub async fn search_query_base_block_max_exact(
        &self,
        query: &TextQuery,
        global: &TextV4GlobalStats,
        k: usize,
        groups: &[(String, Vec<SearchFilterValue>)],
    ) -> Result<TextV4SearchResult> {
        if !self.is_authoritative_base_for(global) {
            return self
                .search_query_exact_with_block_max_fallback(
                    query,
                    global,
                    k,
                    groups,
                    "non_authoritative_base",
                )
                .await;
        }
        if k == 0 || query.is_empty() || self.footer.doc_table.row_count == 0 {
            return Ok(TextV4SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                postings_decoded: 0,
                posting_blocks_fetched: 0,
                posting_blocks_skipped: 0,
                block_max_pruning: true,
                block_max_fallback: None,
                peak_live_bytes: 0,
            });
        }
        validate_global_stats(global)?;
        let mut scored_terms = query
            .base_terms()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for prefix in &query.prefixes {
            scored_terms.extend(self.expand_prefix(prefix).await?);
        }
        scored_terms.sort();
        scored_terms.dedup();
        let mut entries = BTreeMap::<String, TermEntry>::new();
        for term in &scored_terms {
            if let Some(entry) = self.lookup_term(term).await? {
                entries.insert(term.clone(), entry);
            }
        }
        if query
            .phrases
            .iter()
            .flatten()
            .any(|term| !entries.contains_key(term))
            || entries.is_empty()
        {
            return Ok(TextV4SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                postings_decoded: 0,
                posting_blocks_fetched: 0,
                posting_blocks_skipped: 0,
                block_max_pruning: true,
                block_max_fallback: None,
                peak_live_bytes: 0,
            });
        }

        let retained_k = k.min(
            usize::try_from(self.footer.doc_table.row_count)
                .map_err(|_| Error::invariant("text v4 row count does not fit usize"))?,
        );
        let workspace_bytes =
            estimate_exact_search_workspace(&entries, groups, &self.footer.filters, retained_k)?;
        let n_docs = usize::try_from(global.document_count)
            .map_err(|_| Error::invariant("global BM25 document count does not fit usize"))?;
        let corpus_avg_len = avg_len(global.total_document_len, n_docs);
        let mut idfs = BTreeMap::new();
        for term in &scored_terms {
            let df = *global.document_frequency.get(term).ok_or_else(|| {
                Error::invariant(format!("global BM25 statistics omit query term {term:?}"))
            })?;
            if df > global.document_count {
                return Err(Error::invariant(format!(
                    "global BM25 df for {term:?} exceeds document count"
                )));
            }
            idfs.insert(
                term.clone(),
                bm25_idf(
                    n_docs,
                    usize::try_from(df)
                        .map_err(|_| Error::invariant("global BM25 df does not fit usize"))?,
                ),
            );
        }

        let mut unsafe_bounds = false;
        let mut cursors = entries
            .into_iter()
            .map(|(term, entry)| {
                let idf = *idfs
                    .get(&term)
                    .ok_or_else(|| Error::invariant("text v4 IDF cache lost a term"))?;
                let block_upper_bounds = entry
                    .blocks
                    .iter()
                    .map(|block| {
                        let bound = conservative_block_upper_bound(idf, block, corpus_avg_len);
                        unsafe_bounds |= bound.is_none();
                        bound.unwrap_or(0.0)
                    })
                    .collect::<Vec<_>>();
                unsafe_bounds |= block_upper_bounds.is_empty();
                Ok(PostingCursor {
                    term,
                    idf,
                    blocks: entry.blocks,
                    block_upper_bounds,
                    next_block: 0,
                    postings: Vec::new(),
                    next_posting: 0,
                    skip_through: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let maximum_query_bound =
            conservative_upper_bound_sum(cursors.iter().filter_map(|cursor| {
                cursor
                    .block_upper_bounds
                    .iter()
                    .max_by(|left, right| left.total_cmp(right))
            }));
        unsafe_bounds |= maximum_query_bound.is_none();
        drop(idfs);
        drop(scored_terms);
        if unsafe_bounds {
            return self
                .search_query_exact_with_block_max_fallback(
                    query,
                    global,
                    k,
                    groups,
                    "unsafe_block_bound",
                )
                .await;
        }

        let _workspace = shared_search_workspace()
            .reserve("text v4 base block-max search", workspace_bytes)
            .await?;
        let (allowed, applied_filter_groups, filter_peak_bytes) =
            self.load_filter_mask(groups).await?;
        let mask_bytes = allowed.as_ref().map(QueryMask::resident_bytes).unwrap_or(0);
        let mut peak_live_bytes = filter_peak_bytes.max(mask_bytes);
        let mut postings_decoded = 0usize;
        let mut posting_blocks_fetched = 0usize;
        let mut posting_blocks_skipped = 0usize;
        let mut merge_heap = BinaryHeap::with_capacity(cursors.len());
        let mut matched_cursors = Vec::with_capacity(cursors.len());
        let mut top_k = BinaryHeap::with_capacity(retained_k.saturating_add(1));
        let mut documents_evaluated = 0usize;
        let mut pruning_dirty = true;

        loop {
            if top_k.len() == retained_k && pruning_dirty {
                if try_prune_block_window(&mut cursors, &top_k, &mut posting_blocks_skipped)? {
                    rebuild_cursor_heap(&cursors, &mut merge_heap);
                    pruning_dirty = true;
                    continue;
                }
                pruning_dirty = false;
            }

            let (transient, loaded_blocks) = self
                .load_block_max_heads(
                    &mut cursors,
                    &mut merge_heap,
                    &mut postings_decoded,
                    &mut posting_blocks_fetched,
                )
                .await?;
            if loaded_blocks {
                pruning_dirty = true;
                peak_live_bytes = peak_live_bytes.max(
                    exact_query_live_bytes(
                        &cursors,
                        allowed.as_ref(),
                        &merge_heap,
                        &matched_cursors,
                        &top_k,
                    )
                    .saturating_add(transient),
                );
                if top_k.len() == retained_k {
                    continue;
                }
            }
            let Some(head) = merge_heap.pop() else {
                break;
            };
            documents_evaluated = documents_evaluated.saturating_add(1);
            if documents_evaluated & (crate::cancel::CHECK_STRIDE - 1) == 0 {
                crate::cancel::check()?;
            }
            let doc = head.doc;
            matched_cursors.clear();
            matched_cursors.push(head.cursor);
            while merge_heap.peek().is_some_and(|head| head.doc == doc) {
                matched_cursors.push(
                    merge_heap
                        .pop()
                        .expect("peeked block-max text merge head")
                        .cursor,
                );
            }
            debug_assert!(
                matched_cursors.windows(2).all(|pair| pair[0] < pair[1]),
                "equal-doc block-max cursor heads must retain lexical term order"
            );
            let previous_threshold = top_k.peek().map(|ranked| ranked.score.to_bits());
            if allowed.as_ref().is_some_and(|mask| !mask.contains(doc))
                || !phrases_match_cursors(doc, &query.phrases, &cursors)
            {
                // Native filters and phrases can only remove candidates, so
                // the unfiltered block upper bound remains conservative.
            } else {
                let mut score = 0.0f64;
                for &cursor_index in &matched_cursors {
                    let cursor = &cursors[cursor_index];
                    let posting = cursor
                        .current()
                        .expect("block-max heap head must name a current posting");
                    debug_assert_eq!(posting.doc, doc);
                    score += bm25_term_score(
                        cursor.idf,
                        posting.positions.len() as u32,
                        posting.doc_len as usize,
                        corpus_avg_len,
                    );
                }
                if score > 0.0 && score.is_finite() {
                    top_k.push(RankedOrdinal { doc, score });
                    if top_k.len() > retained_k {
                        top_k.pop();
                    }
                }
            }

            let mut crossed_block_boundary = false;
            for &cursor_index in &matched_cursors {
                let cursor = &mut cursors[cursor_index];
                cursor.next_posting += 1;
                if cursor.next_posting == cursor.postings.len() {
                    cursor.postings = Vec::new();
                    cursor.next_posting = 0;
                    crossed_block_boundary = true;
                } else if let Some(posting) = cursor.current() {
                    merge_heap.push(CursorHead {
                        doc: posting.doc,
                        cursor: cursor_index,
                    });
                }
            }
            let threshold_changed =
                previous_threshold != top_k.peek().map(|ranked| ranked.score.to_bits());
            pruning_dirty |= crossed_block_boundary || threshold_changed;
            peak_live_bytes = peak_live_bytes.max(exact_query_live_bytes(
                &cursors,
                allowed.as_ref(),
                &merge_heap,
                &matched_cursors,
                &top_k,
            ));
        }
        drop(cursors);
        drop(allowed);
        drop(merge_heap);
        drop(matched_cursors);

        let mut ranked = top_k.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc.cmp(&right.doc))
        });
        peak_live_bytes = peak_live_bytes.max(estimated_text_result_bytes(ranked.len()));
        let records = self
            .read_doc_records(&ranked.iter().map(|value| value.doc).collect::<Vec<_>>())
            .await?;
        let hits = ranked
            .into_iter()
            .zip(records)
            .map(|(ranked, record)| TextV4Hit {
                node_id: record.node_id,
                lsn: record.lsn,
                payload_fingerprint: record.payload_fingerprint,
                score: ranked.score,
            })
            .collect();
        if peak_live_bytes > workspace_bytes {
            return Err(Error::invariant(format!(
                "text v4 block-max search used {peak_live_bytes} modeled bytes above its \
                 {workspace_bytes}-byte reservation"
            )));
        }
        Ok(TextV4SearchResult {
            hits,
            applied_filter_groups,
            postings_decoded,
            posting_blocks_fetched,
            posting_blocks_skipped,
            block_max_pruning: true,
            block_max_fallback: None,
            peak_live_bytes,
        })
    }

    fn is_authoritative_base_for(&self, global: &TextV4GlobalStats) -> bool {
        matches!(
            self.segment().stats,
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Absolute(documents),
                total_len: SearchStatValue::Absolute(total_len),
                term_df_violation_count: 0,
            } if self.segment().role == SearchSegmentRole::Base
                && self.segment().suppress_count == 0
                && self.segment().mutation_count == self.segment().live_payload_count
                && self.segment().live_payload_count == documents
                && documents == global.document_count
                && documents == self.footer.doc_table.row_count
                && total_len == global.total_document_len
                && u64::try_from(self.footer.delta_docs).ok() == Some(documents)
                && u64::try_from(self.footer.delta_total_len).ok() == Some(total_len)
        )
    }

    async fn search_query_exact_with_block_max_fallback(
        &self,
        query: &TextQuery,
        global: &TextV4GlobalStats,
        k: usize,
        groups: &[(String, Vec<SearchFilterValue>)],
        reason: &'static str,
    ) -> Result<TextV4SearchResult> {
        let mut result = self.search_query_exact(query, global, k, groups).await?;
        result.block_max_fallback = Some(reason);
        Ok(result)
    }

    async fn load_block_max_heads(
        &self,
        cursors: &mut [PostingCursor],
        merge_heap: &mut BinaryHeap<CursorHead>,
        postings_decoded: &mut usize,
        posting_blocks_fetched: &mut usize,
    ) -> Result<(usize, bool)> {
        let mut transient_peak = 0usize;
        let mut loaded_any = false;
        loop {
            let loaded_min = merge_heap.peek().map(|head| head.doc);
            let unloaded_min = cursors
                .iter()
                .filter(|cursor| cursor.current().is_none())
                .filter_map(PostingCursor::potential_doc)
                .min();
            let Some(unloaded_min) = unloaded_min else {
                break;
            };
            if loaded_min.is_some_and(|loaded| loaded < unloaded_min) {
                break;
            }
            let due = cursors
                .iter()
                .enumerate()
                .filter(|(_, cursor)| cursor.current().is_none())
                .filter(|(_, cursor)| cursor.potential_doc() == Some(unloaded_min))
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if due.is_empty() {
                return Err(Error::invariant(
                    "text v4 block-max planner lost its next posting block",
                ));
            }
            let due_bytes = due.capacity().saturating_mul(std::mem::size_of::<usize>());
            transient_peak = transient_peak.max(
                self.load_next_cursor_blocks(
                    cursors,
                    &due,
                    postings_decoded,
                    posting_blocks_fetched,
                )
                .await?
                .saturating_add(due_bytes),
            );
            for cursor_index in due {
                let posting = cursors[cursor_index].current().ok_or_else(|| {
                    Error::invariant("text v4 block-max loaded an empty posting block")
                })?;
                merge_heap.push(CursorHead {
                    doc: posting.doc,
                    cursor: cursor_index,
                });
            }
            loaded_any = true;
        }
        Ok((transient_peak, loaded_any))
    }

    /// Bounded integrity scrub of every version, doc, dictionary, posting and
    /// filter block, including live-ordinal identity parity.
    pub async fn verify_all(&self) -> Result<()> {
        self.version_reader.verify_all().await?;
        let mut doc_hasher = Xxh3::new();
        let mut previous_node: Option<[u8; 16]> = None;
        let mut live_total_len = 0u64;
        for first in (0..self.footer.doc_table.row_count).step_by(MAX_DOC_BATCH) {
            let end = (first + MAX_DOC_BATCH as u64).min(self.footer.doc_table.row_count);
            let ordinals = (first..end).collect::<Vec<_>>();
            let ranges = ordinals
                .iter()
                .map(|ordinal| self.doc_record_range(*ordinal))
                .collect::<Result<Vec<_>>>()?;
            let bodies = self.source.read_ranges(&ranges).await?;
            if bodies.len() != ordinals.len() {
                return Err(Error::invariant(
                    "text v4 source returned wrong document scrub count",
                ));
            }
            let mut ids = Vec::with_capacity(bodies.len());
            let mut records = Vec::with_capacity(bodies.len());
            for body in bodies {
                require_len(&body, DOC_RECORD_LEN as usize, "document record")?;
                doc_hasher.update(&body);
                let record = decode_doc_record(&body)?;
                if previous_node.is_some_and(|node| node >= record.node_id) {
                    return Err(Error::invariant(
                        "text v4 document table is not NodeId-sorted",
                    ));
                }
                previous_node = Some(record.node_id);
                live_total_len = live_total_len
                    .checked_add(u64::from(record.doc_len))
                    .ok_or_else(|| Error::invariant("text v4 live total length overflows"))?;
                ids.push(record.node_id);
                records.push(record);
            }
            let versions = self.version_reader.point_probe_many(&ids).await?;
            for ((ordinal, record), version) in ordinals.iter().zip(records).zip(versions) {
                let Some(version) = version else {
                    return Err(Error::invariant(
                        "text v4 live document has no version-table record",
                    ));
                };
                if version.node_id != record.node_id
                    || version.lsn != record.lsn
                    || version.payload_fingerprint != record.payload_fingerprint
                    || !matches!(
                        version.operation,
                        SearchVersionOperation::Live { payload_ordinal }
                            if payload_ordinal == *ordinal
                    )
                {
                    return Err(Error::invariant(
                        "text v4 live document disagrees with version table",
                    ));
                }
            }
        }
        if non_zero_digest(doc_hasher) != self.footer.doc_table.content_xxh3 {
            return Err(Error::invariant("text v4 document table checksum mismatch"));
        }
        if self.footer.binding.segment.role == SearchSegmentRole::Base
            && u64::try_from(self.footer.delta_total_len).ok() != Some(live_total_len)
        {
            return Err(Error::invariant(
                "text v4 base total length disagrees with document table",
            ));
        }

        let mut previous_term: Option<String> = None;
        let mut metadata_hasher = Xxh3::new();
        metadata_hasher.update(POSTINGS_REGION_DOMAIN);
        let mut block_count = 0u64;
        for dictionary in &self.footer.dictionary {
            let entries = self.read_dictionary_block(dictionary).await?;
            for entry in entries {
                if previous_term
                    .as_ref()
                    .is_some_and(|term| term >= &entry.term)
                {
                    return Err(Error::invariant("text v4 terms are not globally sorted"));
                }
                let decoded = self.verify_term_postings(&entry).await?;
                if decoded != entry.live_doc_freq {
                    return Err(Error::invariant(
                        "text v4 term live df disagrees with postings",
                    ));
                }
                for block in &entry.blocks {
                    metadata_hasher.update(&serialize_bounded(
                        block,
                        MAX_RAW_BLOCK_BYTES,
                        "posting metadata digest",
                    )?);
                    block_count += 1;
                }
                previous_term = Some(entry.term);
            }
        }
        if block_count != self.footer.postings_region.block_count
            || non_zero_digest(metadata_hasher) != self.footer.postings_region.metadata_xxh3
        {
            return Err(Error::invariant(
                "text v4 postings region metadata checksum mismatch",
            ));
        }
        for filter in &self.footer.filters {
            validate_filter_directory(filter, self.footer.doc_table.row_count)?;
            for value in &filter.values {
                let compressed = self.source.read_range(value.wire.range()?).await?;
                let raw = decode_block(&compressed, &value.wire, "filter posting block")?;
                decode_filter_query_mask(raw, value, filter.row_count)?;
            }
        }
        Ok(())
    }

    async fn lookup_term(&self, term: &str) -> Result<Option<TermEntry>> {
        let mut low = 0usize;
        let mut high = self.footer.dictionary.len();
        while low < high {
            let middle = low + (high - low) / 2;
            if self.footer.dictionary[middle].last_term.as_str() < term {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        let Some(reference) = self.footer.dictionary.get(low) else {
            return Ok(None);
        };
        if term < reference.first_term.as_str() || term > reference.last_term.as_str() {
            return Ok(None);
        }
        let entries = self.read_dictionary_block(reference).await?;
        let Ok(index) = entries.binary_search_by(|entry| entry.term.as_str().cmp(term)) else {
            return Ok(None);
        };
        Ok(entries.into_iter().nth(index))
    }

    async fn expand_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        if prefix.is_empty() {
            return Ok(Vec::new());
        }
        let mut expanded = Vec::new();
        for reference in &self.footer.dictionary {
            if reference.last_term.as_str() < prefix {
                continue;
            }
            if !reference.first_term.starts_with(prefix)
                && reference.first_term.as_str() > prefix
                && expanded.is_empty()
            {
                // It can still overlap when the prefix sorts before
                // first_term (e.g. "ab" and "aba"). Decode this block once.
            } else if !expanded.is_empty() && !reference.first_term.starts_with(prefix) {
                break;
            }
            for entry in self.read_dictionary_block(reference).await? {
                if entry.term.starts_with(prefix) {
                    expanded.push(entry.term);
                    if expanded.len() == PREFIX_EXPANSION_LIMIT {
                        return Ok(expanded);
                    }
                } else if entry.term.as_str() > prefix {
                    // Terms with one prefix form one contiguous lexical
                    // interval. Once the first term above that interval is
                    // observed, no later dictionary block can match. The old
                    // `expanded.is_empty()` guard scanned the entire
                    // vocabulary for an absent low-sorting prefix.
                    return Ok(expanded);
                }
            }
        }
        Ok(expanded)
    }

    async fn read_dictionary_block(
        &self,
        reference: &DictionaryBlockRef,
    ) -> Result<Vec<TermEntry>> {
        let compressed = self.source.read_range(reference.wire.range()?).await?;
        let raw = decode_block(&compressed, &reference.wire, "dictionary block")?;
        let entries: Vec<TermEntry> =
            deserialize_bounded(&raw, MAX_RAW_BLOCK_BYTES, "text v4 dictionary block")?;
        validate_dictionary_block(
            &entries,
            reference,
            &self.footer.postings_region,
            self.footer.binding.segment.role,
        )?;
        Ok(entries)
    }

    async fn verify_term_postings(&self, entry: &TermEntry) -> Result<u64> {
        let mut decoded = 0u64;
        let mut previous_doc = None;
        for reference in &entry.blocks {
            let compressed = self.source.read_range(reference.wire.range()?).await?;
            let raw = decode_block(&compressed, &reference.wire, "posting block")?;
            let postings = decode_posting_block(&raw, reference)?;
            if previous_doc.is_some_and(|previous| previous >= postings[0].doc) {
                return Err(Error::invariant(
                    "text v4 postings are not sorted across blocks",
                ));
            }
            previous_doc = postings.last().map(|posting| posting.doc);
            decoded = decoded
                .checked_add(postings.len() as u64)
                .ok_or_else(|| Error::invariant("text v4 posting count overflows"))?;
        }
        if decoded != entry.live_doc_freq {
            return Err(Error::invariant(
                "text v4 term posting count is inconsistent",
            ));
        }
        Ok(decoded)
    }

    async fn load_next_cursor_blocks(
        &self,
        cursors: &mut [PostingCursor],
        cursor_indices: &[usize],
        postings_decoded: &mut usize,
        posting_blocks_fetched: &mut usize,
    ) -> Result<usize> {
        if cursor_indices.is_empty() {
            return Ok(0);
        }

        let mut jobs = Vec::<(usize, PostingBlockRef)>::with_capacity(cursor_indices.len());
        let mut previous = None;
        for &cursor_index in cursor_indices {
            if previous.is_some_and(|previous| previous >= cursor_index) {
                return Err(Error::invariant(
                    "text v4 posting cursor batch is not strictly increasing",
                ));
            }
            previous = Some(cursor_index);
            let cursor = cursors.get_mut(cursor_index).ok_or_else(|| {
                Error::invariant("text v4 posting cursor index leaves query plan")
            })?;
            // Release the previous block before fetching/decompressing the
            // next; `Vec::clear` would retain it beside compressed + raw +
            // decoded successors at every boundary.
            cursor.postings = Vec::new();
            cursor.next_posting = 0;
            let Some(reference) = cursor.blocks.get(cursor.next_block).cloned() else {
                continue;
            };
            cursor.next_block += 1;
            jobs.push((cursor_index, reference));
        }
        if jobs.is_empty() {
            return Ok(0);
        }
        *posting_blocks_fetched = posting_blocks_fetched
            .checked_add(jobs.len())
            .ok_or_else(|| Error::invariant("text v4 fetched block count overflows"))?;

        let ranges = jobs
            .iter()
            .map(|(_, reference)| reference.wire.range())
            .collect::<Result<Vec<_>>>()?;
        let compressed_blocks = self.source.read_ranges(&ranges).await?;
        if compressed_blocks.len() != jobs.len() {
            return Err(Error::invariant(
                "text v4 source returned wrong posting block count",
            ));
        }
        let retained_batch_bytes = jobs
            .capacity()
            .saturating_mul(std::mem::size_of::<(usize, PostingBlockRef)>())
            .saturating_add(
                ranges
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Range<u64>>()),
            )
            .saturating_add(
                compressed_blocks
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Bytes>()),
            )
            .saturating_add(compressed_blocks.iter().map(Bytes::len).sum::<usize>());
        let mut largest_raw_bytes = 0usize;
        for ((cursor_index, reference), compressed) in jobs.into_iter().zip(compressed_blocks) {
            let raw = decode_block(&compressed, &reference.wire, "posting block")?;
            largest_raw_bytes = largest_raw_bytes.max(raw.capacity());
            let decoded = decode_posting_block(&raw, &reference)?;
            *postings_decoded = postings_decoded
                .checked_add(decoded.len())
                .ok_or_else(|| Error::invariant("text v4 decoded posting count overflows"))?;
            cursors[cursor_index].postings = decoded;
            if let Some(target) = cursors[cursor_index].skip_through.take() {
                cursors[cursor_index].next_posting = cursors[cursor_index]
                    .postings
                    .partition_point(|posting| posting.doc <= target);
                if cursors[cursor_index].next_posting == cursors[cursor_index].postings.len() {
                    return Err(Error::invariant(
                        "text v4 block-max seek target disagrees with posting metadata",
                    ));
                }
            }
        }
        Ok(retained_batch_bytes.saturating_add(largest_raw_bytes))
    }

    async fn load_filter_mask(
        &self,
        groups: &[(String, Vec<SearchFilterValue>)],
    ) -> Result<(Option<QueryMask>, usize, usize)> {
        let row_count = self.footer.doc_table.row_count;
        let mut combined: Option<QueryMask> = None;
        let mut applied = 0usize;
        let mut peak_live_bytes = 0usize;
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
            let selected = selected_filter_value_indices(reference, alternatives);
            let selected_bytes = selected
                .capacity()
                .saturating_mul(std::mem::size_of::<usize>());
            let mut group = QueryMaskUnionBuilder::new(row_count);
            for batch in selected.chunks(MAX_FILTER_RANGE_BATCH) {
                crate::cancel::check()?;
                let ranges = batch
                    .iter()
                    .map(|value_index| reference.values[*value_index].wire.range())
                    .collect::<Result<Vec<_>>>()?;
                let compressed_blocks = self.source.read_ranges(&ranges).await?;
                if compressed_blocks.len() != batch.len() {
                    return Err(Error::invariant(
                        "text v4 source returned wrong filter posting block count",
                    ));
                }
                let retained_batch_bytes = ranges
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Range<u64>>())
                    .saturating_add(
                        compressed_blocks
                            .capacity()
                            .saturating_mul(std::mem::size_of::<Bytes>()),
                    )
                    .saturating_add(compressed_blocks.iter().map(Bytes::len).sum::<usize>());
                for (&value_index, compressed) in batch.iter().zip(compressed_blocks) {
                    let value = &reference.values[value_index];
                    let raw = decode_block(&compressed, &value.wire, "filter posting block")?;
                    let raw_bytes = raw.capacity();
                    let posting = decode_filter_query_mask(raw, value, row_count)?;
                    let posting_bytes = posting.resident_bytes();
                    let combined_bytes = combined
                        .as_ref()
                        .map(QueryMask::resident_bytes)
                        .unwrap_or(0);
                    peak_live_bytes = peak_live_bytes.max(
                        combined_bytes
                            .saturating_add(selected_bytes)
                            .saturating_add(group.resident_bytes())
                            .saturating_add(retained_batch_bytes)
                            .saturating_add(raw_bytes)
                            .saturating_add(posting_bytes),
                    );
                    let union_peak = group.absorb(posting)?;
                    peak_live_bytes = peak_live_bytes.max(
                        combined_bytes
                            .saturating_add(selected_bytes)
                            .saturating_add(retained_batch_bytes)
                            .saturating_add(union_peak),
                    );
                }
            }
            let (group, finish_peak) = group.finish()?;
            let combined_bytes = combined
                .as_ref()
                .map(QueryMask::resident_bytes)
                .unwrap_or(0);
            peak_live_bytes = peak_live_bytes.max(
                combined_bytes
                    .saturating_add(selected_bytes)
                    .saturating_add(finish_peak),
            );
            combined = Some(match combined {
                Some(mask) => {
                    peak_live_bytes = peak_live_bytes.max(
                        selected_bytes.saturating_add(query_mask_intersection_peak(&mask, &group)),
                    );
                    mask.intersect(group, row_count)?
                }
                None => group,
            });
            peak_live_bytes = peak_live_bytes.max(
                combined
                    .as_ref()
                    .map(QueryMask::resident_bytes)
                    .unwrap_or(0),
            );
            applied += 1;
        }
        Ok((combined, applied, peak_live_bytes))
    }

    async fn read_doc_records(&self, ordinals: &[u64]) -> Result<Vec<DocRecord>> {
        let mut records = Vec::with_capacity(ordinals.len());
        for chunk in ordinals.chunks(MAX_DOC_BATCH) {
            let ranges = chunk
                .iter()
                .map(|ordinal| self.doc_record_range(*ordinal))
                .collect::<Result<Vec<_>>>()?;
            let bodies = self.source.read_ranges(&ranges).await?;
            if bodies.len() != chunk.len() {
                return Err(Error::invariant(
                    "text v4 source returned wrong document record count",
                ));
            }
            for body in bodies {
                require_len(&body, DOC_RECORD_LEN as usize, "document record")?;
                records.push(decode_doc_record(&body)?);
            }
        }
        Ok(records)
    }

    fn doc_record_range(&self, ordinal: u64) -> Result<Range<u64>> {
        if ordinal >= self.footer.doc_table.row_count {
            return Err(Error::invariant("text v4 document ordinal leaves table"));
        }
        let start = self
            .footer
            .doc_table
            .offset
            .checked_add(
                ordinal
                    .checked_mul(DOC_RECORD_LEN)
                    .ok_or_else(|| Error::invariant("text v4 document offset overflows"))?,
            )
            .ok_or_else(|| Error::invariant("text v4 document offset overflows"))?;
        Ok(start..start + DOC_RECORD_LEN)
    }
}

#[derive(Debug)]
struct RankedOrdinal {
    doc: u64,
    score: f64,
}

impl PartialEq for RankedOrdinal {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedOrdinal {}

impl PartialOrd for RankedOrdinal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Heap root is the worst retained score, then the largest NodeId ordinal.
impl Ord for RankedOrdinal {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.doc.cmp(&other.doc))
    }
}

fn rebuild_cursor_heap(cursors: &[PostingCursor], heap: &mut BinaryHeap<CursorHead>) {
    heap.clear();
    for (cursor, state) in cursors.iter().enumerate() {
        if let Some(posting) = state.current() {
            heap.push(CursorHead {
                doc: posting.doc,
                cursor,
            });
        }
    }
}

fn try_prune_block_window(
    cursors: &mut [PostingCursor],
    top_k: &BinaryHeap<RankedOrdinal>,
    posting_blocks_skipped: &mut usize,
) -> Result<bool> {
    let Some(threshold) = top_k.peek().map(|ranked| ranked.score) else {
        return Ok(false);
    };
    if !threshold.is_finite() || threshold <= 0.0 {
        return Ok(false);
    }
    let Some(window_start) = cursors
        .iter()
        .filter_map(PostingCursor::potential_doc)
        .min()
    else {
        return Ok(false);
    };
    let window_end = cursors
        .iter()
        .filter_map(PostingCursor::current_block)
        .map(|block| block.last_doc)
        .min()
        .ok_or_else(|| Error::invariant("text v4 block-max window has no posting block"))?;
    if window_start > window_end {
        return Err(Error::invariant(
            "text v4 block-max window boundaries are inconsistent",
        ));
    }
    let mut upper_bound = 0.0f64;
    for cursor in cursors.iter() {
        if cursor.potential_doc().is_some_and(|doc| doc <= window_end) {
            let bound = cursor.current_block_upper_bound().ok_or_else(|| {
                Error::invariant("text v4 block-max cursor lost an authenticated upper bound")
            })?;
            upper_bound = next_up_nonnegative(upper_bound + bound).ok_or_else(|| {
                Error::invariant("text v4 block-max upper-bound sum became non-finite")
            })?;
        }
    }
    // Strictness is part of correctness: at equality, an unseen lower
    // document ordinal can displace the current worst retained hit.
    if upper_bound >= threshold {
        return Ok(false);
    }
    for cursor in cursors {
        cursor.advance_loaded_through(window_end);
        cursor.skip_unloaded_through(window_end, posting_blocks_skipped)?;
    }
    Ok(true)
}

fn validate_build_configuration<W: Seek>(
    writer: &mut W,
    state: &SearchLsmState,
    context: &mut TextV4BuildContext,
    options: TextV4BuildOptions,
) -> Result<()> {
    if writer.stream_position()? != 0 {
        return Err(Error::invariant(
            "text v4 object writer must start at offset zero",
        ));
    }
    if state.kind != SearchLsmKind::Text
        || state.generation_id.is_nil()
        || state.index_name.is_empty()
        || context.sst_id.is_nil()
    {
        return Err(Error::invariant(
            "text v4 build context disagrees with text generation",
        ));
    }
    if options.postings_per_block == 0 || options.terms_per_dictionary_block == 0 {
        return Err(Error::invariant(
            "text v4 block cardinalities must be positive",
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
            "text v4 complete filter property is empty",
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

fn prepare_payload(payload: &TextV4Payload) -> Result<PreparedPayload> {
    if payload.filters.keys().any(|property| property.is_empty()) {
        return Err(Error::invariant("native-filter property name is empty"));
    }
    Ok(PreparedPayload {
        tokens: tokenize(&payload.text),
        filters: payload.filters.clone(),
    })
}

fn prepared_payload_fingerprint(payload: &PreparedPayload) -> Result<u64> {
    let encoded = serialize_bounded(payload, MAX_RAW_BLOCK_BYTES, "text payload fingerprint")?;
    let mut material = Vec::with_capacity(PAYLOAD_FINGERPRINT_DOMAIN.len() + encoded.len());
    material.extend_from_slice(PAYLOAD_FINGERPRINT_DOMAIN);
    material.extend_from_slice(&encoded);
    Ok(non_zero_xxh3(&material))
}

#[cfg(test)]
fn build_filter_postings(
    property: &str,
    row_count: u64,
    effective: &[EffectiveMutation],
) -> Result<BTreeMap<SearchFilterValue, Vec<u64>>> {
    let words = bitmap_words(row_count)?;
    let mut postings = BTreeMap::<SearchFilterValue, Vec<u64>>::new();
    let mut ordinal = 0u64;
    for mutation in effective {
        let Some(after) = &mutation.after else {
            continue;
        };
        if let Some(value) = after.filters.get(property) {
            let bitmap = postings
                .entry(value.clone())
                .or_insert_with(|| vec![0; words]);
            set_bitmap(bitmap, ordinal)?;
        }
        ordinal += 1;
    }
    if ordinal != row_count {
        return Err(Error::invariant(
            "text v4 filter ordinal accounting diverged",
        ));
    }
    Ok(postings)
}

fn validate_filter_directory(reference: &FilterBlockRef, row_count: u64) -> Result<()> {
    let words = bitmap_words(row_count)?;
    if reference.property.is_empty()
        || reference.row_count != row_count
        || reference
            .values
            .windows(2)
            .any(|pair| pair[0].value >= pair[1].value)
    {
        return Err(Error::invariant("text v4 filter directory is inconsistent"));
    }
    let dense_len = words
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::invariant("text v4 filter bitmap length overflows"))?;
    for value in &reference.values {
        if value.cardinality == 0 || value.cardinality > row_count {
            return Err(Error::invariant(
                "text v4 filter cardinality is inconsistent",
            ));
        }
        match value.encoding {
            FilterPostingEncoding::SparseDeltaVarint => {
                if value.wire.raw_len == 0
                    || u64::from(value.wire.raw_len) > value.cardinality.saturating_mul(10)
                {
                    return Err(Error::invariant(
                        "text v4 sparse filter posting length is inconsistent",
                    ));
                }
            }
            FilterPostingEncoding::DenseBitmap => {
                if usize::try_from(value.wire.raw_len).ok() != Some(dense_len) {
                    return Err(Error::invariant(
                        "text v4 dense filter posting length is inconsistent",
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
) -> Result<QueryMask> {
    match reference.encoding {
        FilterPostingEncoding::SparseDeltaVarint => {
            let capacity = usize::try_from(reference.cardinality)
                .map_err(|_| Error::invariant("text v4 filter cardinality exceeds usize"))?;
            let mut ordinals = Vec::with_capacity(capacity);
            let mut cursor = 0usize;
            let mut previous = 0u64;
            for index in 0..reference.cardinality {
                let delta = decode_u64_varint(&raw, &mut cursor)?;
                if index > 0 && delta == 0 {
                    return Err(Error::invariant(
                        "text v4 sparse filter ordinals are not increasing",
                    ));
                }
                let ordinal = previous
                    .checked_add(delta)
                    .ok_or_else(|| Error::invariant("text v4 filter ordinal overflows"))?;
                if ordinal >= row_count {
                    return Err(Error::invariant(
                        "text v4 sparse filter ordinal leaves document table",
                    ));
                }
                ordinals.push(ordinal);
                previous = ordinal;
            }
            if cursor != raw.len() {
                return Err(Error::invariant(
                    "text v4 sparse filter posting has trailing bytes",
                ));
            }
            Ok(QueryMask::Sparse(ordinals))
        }
        FilterPostingEncoding::DenseBitmap => {
            let expected = bitmap_words(row_count)?
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or_else(|| Error::invariant("text v4 filter bitmap length overflows"))?;
            if raw.len() != expected {
                return Err(Error::invariant(
                    "text v4 dense filter bitmap length is inconsistent",
                ));
            }
            let count = raw
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
            if remainder != 0 && last_word & (!0u64 << remainder) != 0
                || count != reference.cardinality
            {
                return Err(Error::invariant(
                    "text v4 dense filter bitmap/cardinality is inconsistent",
                ));
            }
            Ok(QueryMask::Dense(raw))
        }
    }
}

fn dense_mask_bytes(row_count: u64) -> Result<usize> {
    bitmap_words(row_count)?
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::invariant("text v4 query bitmap length overflows"))
}

/// Resolve caller alternatives to sorted, distinct footer indices without
/// cloning filter values or building a tree node per alternative. Memory is
/// proportional to the already-present request, never to the filter corpus.
fn selected_filter_value_indices(
    reference: &FilterBlockRef,
    alternatives: &[SearchFilterValue],
) -> Vec<usize> {
    let mut selected = Vec::with_capacity(alternatives.len().min(reference.values.len()));
    for value in alternatives {
        if let Ok(index) = reference
            .values
            .binary_search_by(|candidate| candidate.value.cmp(value))
        {
            selected.push(index);
        }
    }
    selected.sort_unstable();
    selected.dedup();
    selected
}

/// Conservative allocation high-water for exact AND of two query masks.
fn query_mask_intersection_peak(left: &QueryMask, right: &QueryMask) -> usize {
    let left_bytes = left.resident_bytes();
    let right_bytes = right.resident_bytes();
    let output_bytes = match (left, right) {
        (QueryMask::Sparse(left), QueryMask::Sparse(right)) => left
            .len()
            .min(right.len())
            .saturating_mul(std::mem::size_of::<u64>()),
        (QueryMask::Dense(_), QueryMask::Sparse(_))
        | (QueryMask::Sparse(_), QueryMask::Dense(_)) => 0,
        // Dense∩dense is in-place, but adaptive sparsification may allocate a
        // second representation before the dense input is released.
        (QueryMask::Dense(left), QueryMask::Dense(_)) => left.capacity(),
    };
    left_bytes
        .saturating_add(right_bytes)
        .saturating_add(output_bytes)
}

fn set_dense_filter_bit_bytes(bitmap: &mut [u8], ordinal: u64, row_count: u64) -> Result<()> {
    if ordinal >= row_count {
        return Err(Error::invariant(
            "text v4 query filter ordinal leaves document table",
        ));
    }
    let byte = usize::try_from(ordinal / 8)
        .map_err(|_| Error::invariant("text v4 query filter ordinal exceeds usize"))?;
    let target = bitmap
        .get_mut(byte)
        .ok_or_else(|| Error::invariant("text v4 query filter ordinal leaves bitmap"))?;
    *target |= 1u8 << (ordinal % 8);
    Ok(())
}

fn dense_filter_contains(bitmap: &[u8], ordinal: u64) -> bool {
    usize::try_from(ordinal / 8)
        .ok()
        .and_then(|byte| bitmap.get(byte))
        .is_some_and(|byte| byte & (1u8 << (ordinal % 8)) != 0)
}

#[cfg(test)]
fn filter_posting_wire(
    bitmap: &[u64],
    row_count: u64,
) -> Result<(FilterPostingEncoding, Vec<u8>, u64)> {
    if bitmap.len() != bitmap_words(row_count)? || has_bits_past(bitmap, row_count) {
        return Err(Error::invariant(
            "text v4 filter bitmap input is inconsistent",
        ));
    }
    let cardinality = bitmap
        .iter()
        .map(|word| u64::from(word.count_ones()))
        .sum::<u64>();
    if cardinality == 0 {
        return Err(Error::invariant("text v4 empty filter posting"));
    }
    let mut sparse = Vec::new();
    let mut previous = 0u64;
    let mut seen = 0u64;
    for ordinal in 0..row_count {
        if bitmap_contains(bitmap, ordinal) {
            let delta = if seen == 0 {
                ordinal
            } else {
                ordinal - previous
            };
            encode_u64_varint(delta, &mut sparse);
            previous = ordinal;
            seen += 1;
        }
    }
    let dense_len = bitmap
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::invariant("text v4 dense bitmap length overflows"))?;
    if sparse.len() < dense_len {
        Ok((
            FilterPostingEncoding::SparseDeltaVarint,
            sparse,
            cardinality,
        ))
    } else {
        let mut dense = Vec::with_capacity(dense_len);
        for word in bitmap {
            dense.extend_from_slice(&word.to_le_bytes());
        }
        Ok((FilterPostingEncoding::DenseBitmap, dense, cardinality))
    }
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
            .ok_or_else(|| Error::invariant("text v4 sparse varint is truncated"))?;
        *cursor += 1;
        let payload = u64::from(byte & 0x7f);
        if shift == 63 && payload > 1 {
            return Err(Error::invariant("text v4 sparse varint overflows"));
        }
        value |= payload << shift;
        if byte & 0x80 == 0 {
            if shift > 0 && payload == 0 {
                return Err(Error::invariant("text v4 sparse varint is not canonical"));
            }
            return Ok(value);
        }
    }
    Err(Error::invariant("text v4 sparse varint is too long"))
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

fn bitmap_words(row_count: u64) -> Result<usize> {
    usize::try_from(row_count.div_ceil(64))
        .map_err(|_| Error::invariant("native-filter bitmap does not fit usize"))
}

#[cfg(test)]
fn set_bitmap(bitmap: &mut [u64], ordinal: u64) -> Result<()> {
    let word = usize::try_from(ordinal / 64)
        .map_err(|_| Error::invariant("native-filter ordinal does not fit usize"))?;
    let target = bitmap
        .get_mut(word)
        .ok_or_else(|| Error::invariant("native-filter ordinal leaves bitmap"))?;
    *target |= 1u64 << (ordinal % 64);
    Ok(())
}

#[cfg(test)]
fn bitmap_contains(bitmap: &[u64], ordinal: u64) -> bool {
    usize::try_from(ordinal / 64)
        .ok()
        .and_then(|word| bitmap.get(word))
        .is_some_and(|word| word & (1u64 << (ordinal % 64)) != 0)
}

#[cfg(test)]
fn has_bits_past(bitmap: &[u64], row_count: u64) -> bool {
    let remainder = row_count % 64;
    remainder != 0
        && bitmap
            .last()
            .is_some_and(|word| word & (!0u64 << remainder) != 0)
}

fn encode_doc_record(record: DocRecord) -> [u8; DOC_RECORD_LEN as usize] {
    let mut out = [0u8; DOC_RECORD_LEN as usize];
    out[..16].copy_from_slice(&record.node_id);
    out[16..24].copy_from_slice(&record.lsn.to_le_bytes());
    out[24..32].copy_from_slice(&record.payload_fingerprint.to_le_bytes());
    out[32..36].copy_from_slice(&record.doc_len.to_le_bytes());
    out
}

fn decode_doc_record(bytes: &[u8]) -> Result<DocRecord> {
    if bytes.len() != DOC_RECORD_LEN as usize || bytes[36..40].iter().any(|byte| *byte != 0) {
        return Err(Error::invariant("text v4 document record is malformed"));
    }
    let record = DocRecord {
        node_id: bytes[..16].try_into().expect("fixed text v4 NodeId"),
        lsn: read_u64(bytes, 16)?,
        payload_fingerprint: read_u64(bytes, 24)?,
        doc_len: read_u32(bytes, 32)?,
    };
    if record.lsn == 0 {
        return Err(Error::invariant("text v4 document record uses LSN zero"));
    }
    Ok(record)
}

fn encode_posting_block(postings: &[Posting]) -> Result<Vec<u8>> {
    if postings.is_empty()
        || postings.windows(2).any(|pair| pair[0].doc >= pair[1].doc)
        || postings.iter().any(|posting| {
            posting.positions.is_empty()
                || posting.positions.len() > posting.doc_len as usize
                || posting.positions.windows(2).any(|pair| pair[0] >= pair[1])
                || posting
                    .positions
                    .last()
                    .is_some_and(|position| *position >= posting.doc_len)
        })
    {
        return Err(Error::invariant(
            "text v4 posting block input is inconsistent",
        ));
    }
    let positions = postings.iter().try_fold(0usize, |total, posting| {
        total
            .checked_add(posting.positions.len())
            .ok_or_else(|| Error::invariant("text v4 posting positions overflow"))
    })?;
    let capacity = POSTING_HEADER_LEN
        .checked_add(
            postings
                .len()
                .checked_mul(POSTING_PREFIX_LEN)
                .ok_or_else(|| Error::invariant("text v4 posting block size overflows"))?,
        )
        .and_then(|value| value.checked_add(positions.checked_mul(4)?))
        .ok_or_else(|| Error::invariant("text v4 posting block size overflows"))?;
    if capacity as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant("text v4 posting block exceeds raw limit"));
    }
    let mut raw = Vec::with_capacity(capacity);
    raw.extend_from_slice(
        &u32::try_from(postings.len())
            .map_err(|_| Error::invariant("text v4 posting count exceeds u32"))?
            .to_le_bytes(),
    );
    for posting in postings {
        raw.extend_from_slice(&posting.doc.to_le_bytes());
        raw.extend_from_slice(&posting.doc_len.to_le_bytes());
        raw.extend_from_slice(
            &u32::try_from(posting.positions.len())
                .map_err(|_| Error::invariant("text v4 term frequency exceeds u32"))?
                .to_le_bytes(),
        );
        for position in &posting.positions {
            raw.extend_from_slice(&position.to_le_bytes());
        }
    }
    Ok(raw)
}

fn decode_posting_block(raw: &[u8], reference: &PostingBlockRef) -> Result<Vec<Posting>> {
    if raw.len() < POSTING_HEADER_LEN || read_u32(raw, 0)? != reference.posting_count {
        return Err(Error::invariant(
            "text v4 posting block header is inconsistent",
        ));
    }
    let mut cursor = POSTING_HEADER_LEN;
    let mut postings = Vec::with_capacity(reference.posting_count as usize);
    for _ in 0..reference.posting_count {
        let prefix_end = cursor
            .checked_add(POSTING_PREFIX_LEN)
            .ok_or_else(|| Error::invariant("text v4 posting cursor overflows"))?;
        if prefix_end > raw.len() {
            return Err(Error::invariant("text v4 posting prefix is truncated"));
        }
        let doc = read_u64(raw, cursor)?;
        let doc_len = read_u32(raw, cursor + 8)?;
        let tf = read_u32(raw, cursor + 12)?;
        cursor = prefix_end;
        if tf == 0 || tf > doc_len {
            return Err(Error::invariant(
                "text v4 posting tf/document length is invalid",
            ));
        }
        let positions_end = cursor
            .checked_add(tf as usize * 4)
            .ok_or_else(|| Error::invariant("text v4 posting positions overflow"))?;
        if positions_end > raw.len() {
            return Err(Error::invariant("text v4 posting positions are truncated"));
        }
        let mut positions = Vec::with_capacity(tf as usize);
        while cursor < positions_end {
            positions.push(read_u32(raw, cursor)?);
            cursor += 4;
        }
        if positions.windows(2).any(|pair| pair[0] >= pair[1])
            || positions
                .last()
                .is_some_and(|position| *position >= doc_len)
        {
            return Err(Error::invariant(
                "text v4 posting positions are inconsistent",
            ));
        }
        postings.push(Posting {
            doc,
            doc_len,
            positions,
        });
    }
    if cursor != raw.len()
        || postings.first().map(|posting| posting.doc) != Some(reference.first_doc)
        || postings.last().map(|posting| posting.doc) != Some(reference.last_doc)
        || postings.windows(2).any(|pair| pair[0].doc >= pair[1].doc)
        || postings
            .iter()
            .map(|posting| posting.positions.len() as u32)
            .max()
            != Some(reference.max_tf)
        || postings.iter().map(|posting| posting.doc_len).min() != Some(reference.min_doc_len)
    {
        return Err(Error::invariant(
            "text v4 posting block disagrees with directory",
        ));
    }
    Ok(postings)
}

fn validate_dictionary_block(
    entries: &[TermEntry],
    reference: &DictionaryBlockRef,
    postings_region: &RegionRef,
    role: SearchSegmentRole,
) -> Result<()> {
    if entries.len() != reference.term_count as usize
        || entries.is_empty()
        || entries.first().map(|entry| entry.term.as_str()) != Some(reference.first_term.as_str())
        || entries.last().map(|entry| entry.term.as_str()) != Some(reference.last_term.as_str())
        || entries.windows(2).any(|pair| pair[0].term >= pair[1].term)
    {
        return Err(Error::invariant(
            "text v4 dictionary block disagrees with sparse directory",
        ));
    }
    let postings_end = postings_region
        .offset
        .checked_add(postings_region.len)
        .ok_or_else(|| Error::invariant("text v4 postings region overflows"))?;
    for entry in entries {
        if entry.term.is_empty()
            || (entry.delta_df == 0 && entry.live_doc_freq == 0)
            || (role == SearchSegmentRole::Base
                && u64::try_from(entry.delta_df).ok() != Some(entry.live_doc_freq))
            || entry
                .blocks
                .iter()
                .map(|block| u64::from(block.posting_count))
                .sum::<u64>()
                != entry.live_doc_freq
        {
            return Err(Error::invariant(
                "text v4 term entry statistics are inconsistent",
            ));
        }
        let mut previous_doc: Option<u64> = None;
        let mut previous_end = None;
        for block in &entry.blocks {
            let range = block.wire.range()?;
            if block.posting_count == 0
                || block.max_tf == 0
                || block.min_doc_len == 0
                || block.first_doc > block.last_doc
                || previous_doc.is_some_and(|doc| doc >= block.first_doc)
                || range.start < postings_region.offset
                || range.end > postings_end
                || previous_end.is_some_and(|end| end != range.start)
            {
                return Err(Error::invariant(
                    "text v4 posting directory is inconsistent",
                ));
            }
            validate_block_limits(&block.wire)?;
            previous_doc = Some(block.last_doc);
            previous_end = Some(range.end);
        }
        if entry.live_doc_freq == 0 && !entry.blocks.is_empty()
            || entry.live_doc_freq > 0 && entry.blocks.is_empty()
        {
            return Err(Error::invariant(
                "text v4 term live df/block presence is inconsistent",
            ));
        }
    }
    Ok(())
}

fn cursor_live_bytes(cursors: &[PostingCursor]) -> usize {
    cursors
        .len()
        .saturating_mul(std::mem::size_of::<PostingCursor>())
        .saturating_add(
            cursors
                .iter()
                .map(PostingCursor::resident_bytes)
                .sum::<usize>(),
        )
}

fn exact_query_live_bytes(
    cursors: &[PostingCursor],
    allowed: Option<&QueryMask>,
    merge_heap: &BinaryHeap<CursorHead>,
    matched_cursors: &[usize],
    top_k: &BinaryHeap<RankedOrdinal>,
) -> usize {
    cursor_live_bytes(cursors)
        .saturating_add(allowed.map(QueryMask::resident_bytes).unwrap_or(0))
        .saturating_add(
            merge_heap
                .capacity()
                .saturating_mul(std::mem::size_of::<CursorHead>()),
        )
        .saturating_add(
            matched_cursors
                .len()
                .saturating_mul(std::mem::size_of::<usize>()),
        )
        .saturating_add(
            top_k
                .capacity()
                .saturating_mul(std::mem::size_of::<RankedOrdinal>()),
        )
}

fn current_cursor_posting<'a>(
    cursors: &'a [PostingCursor],
    term: &str,
    doc: u64,
) -> Option<&'a Posting> {
    let index = cursors
        .binary_search_by(|cursor| cursor.term.as_str().cmp(term))
        .ok()?;
    cursors[index]
        .current()
        .filter(|posting| posting.doc == doc)
}

fn phrases_match_cursors(doc: u64, phrases: &[Vec<String>], cursors: &[PostingCursor]) -> bool {
    phrases.iter().all(|phrase| {
        let Some((first, rest)) = phrase.split_first() else {
            // The parser never emits these, but preserve the ranged reader's
            // fail-closed behaviour for a manually constructed query.
            return false;
        };
        let Some(first_posting) = current_cursor_posting(cursors, first, doc) else {
            return false;
        };
        first_posting.positions.iter().any(|start| {
            rest.iter().enumerate().all(|(offset, term)| {
                let expected = u32::try_from(offset)
                    .ok()
                    .and_then(|offset| start.checked_add(offset))
                    .and_then(|position| position.checked_add(1));
                expected.is_some_and(|expected| {
                    current_cursor_posting(cursors, term, doc)
                        .is_some_and(|posting| posting.positions.binary_search(&expected).is_ok())
                })
            })
        })
    })
}

fn estimate_exact_search_workspace(
    entries: &BTreeMap<String, TermEntry>,
    groups: &[(String, Vec<SearchFilterValue>)],
    filters: &[FilterBlockRef],
    k: usize,
) -> Result<usize> {
    let posting_bytes = entries.values().fold(0usize, |total, entry| {
        let largest = entry
            .blocks
            .iter()
            .map(|block| {
                (block.wire.len as usize)
                    // The raw block and its decoded position vectors can
                    // coexist while the cursor installs the successor.
                    .saturating_add((block.wire.raw_len as usize).saturating_mul(2))
                    .saturating_add(
                        (block.posting_count as usize)
                            .saturating_mul(std::mem::size_of::<Posting>()),
                    )
            })
            .max()
            .unwrap_or(0);
        total
            .saturating_add(largest)
            .saturating_add(std::mem::size_of::<PostingCursor>())
            .saturating_add(std::mem::size_of::<usize>())
            .saturating_add(std::mem::size_of::<(usize, PostingBlockRef)>())
            .saturating_add(std::mem::size_of::<Range<u64>>())
            .saturating_add(std::mem::size_of::<Bytes>())
            .saturating_add(
                entry
                    .blocks
                    .capacity()
                    .saturating_mul(std::mem::size_of::<PostingBlockRef>()),
            )
            .saturating_add(
                entry
                    .blocks
                    .capacity()
                    .saturating_mul(std::mem::size_of::<f64>()),
            )
            .saturating_add(entry.term.capacity())
    });
    let mut combined_bytes = 0usize;
    let mut filter_peak = 0usize;
    for (property, alternatives) in groups {
        let Ok(index) =
            filters.binary_search_by(|candidate| candidate.property.as_str().cmp(property))
        else {
            continue;
        };
        let reference = &filters[index];
        let dense_bytes = dense_mask_bytes(reference.row_count)?;
        let selected_indices = selected_filter_value_indices(reference, alternatives);
        let selected_index_bytes = selected_indices
            .capacity()
            .saturating_mul(std::mem::size_of::<usize>());
        let compressed_batch_peak = selected_indices
            .chunks(MAX_FILTER_RANGE_BATCH)
            .map(|batch| {
                batch.iter().fold(
                    batch.len().saturating_mul(
                        std::mem::size_of::<Range<u64>>()
                            .saturating_add(std::mem::size_of::<Bytes>()),
                    ),
                    |total, index| total.saturating_add(reference.values[*index].wire.len as usize),
                )
            })
            .max()
            .unwrap_or(0);
        let mut selected_value_peak = 0usize;
        let mut sparse_bytes = 0usize;
        let mut dense = false;
        for &index in &selected_indices {
            let selected = &reference.values[index];
            let decoded_bytes = match selected.encoding {
                FilterPostingEncoding::SparseDeltaVarint => usize::try_from(selected.cardinality)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(std::mem::size_of::<u64>()),
                FilterPostingEncoding::DenseBitmap => dense_bytes,
            };
            selected_value_peak = selected_value_peak
                .max((selected.wire.raw_len as usize).saturating_add(decoded_bytes));
            match selected.encoding {
                FilterPostingEncoding::SparseDeltaVarint => {
                    sparse_bytes = sparse_bytes.saturating_add(decoded_bytes);
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
                .saturating_add(selected_index_bytes)
                // Sparse append/dense conversion and group intersection can
                // briefly coexist with both inputs and one output.
                .saturating_add(group_bytes.saturating_mul(3))
                .saturating_add(compressed_batch_peak)
                .saturating_add(selected_value_peak),
        );
        combined_bytes = if combined_bytes == 0 {
            group_bytes
        } else {
            combined_bytes.min(group_bytes)
        };
    }
    let heap_bytes = k
        .saturating_add(1)
        .saturating_mul(std::mem::size_of::<RankedOrdinal>());
    let merge_bytes = entries.len().saturating_mul(
        std::mem::size_of::<CursorHead>()
            // Matched-cursor and successor-block index batches coexist at
            // posting block boundaries.
            .saturating_add(std::mem::size_of::<usize>().saturating_mul(2)),
    );
    Ok(posting_bytes
        .saturating_add(filter_peak)
        .saturating_add(heap_bytes)
        .saturating_add(merge_bytes)
        .saturating_add(estimated_text_result_bytes(k))
        .saturating_add(64 * 1024))
}

fn validate_global_stats(global: &TextV4GlobalStats) -> Result<()> {
    if global.document_count == 0
        || global
            .document_frequency
            .values()
            .any(|df| *df > global.document_count)
    {
        return Err(Error::invariant("global BM25 statistics are inconsistent"));
    }
    Ok(())
}

fn write_compressed_block<W: Write + Seek>(
    writer: &mut W,
    raw: &[u8],
    compression_level: i32,
    what: &str,
) -> Result<BlockRef> {
    if raw.is_empty() || raw.len() as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "text v4 {what} raw length is invalid"
        )));
    }
    let compressed = zstd::stream::encode_all(Cursor::new(raw), compression_level)
        .map_err(|error| Error::invariant(format!("text v4 {what} compression failed: {error}")))?;
    if compressed.is_empty() || compressed.len() as u64 > MAX_COMPRESSED_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "text v4 {what} compressed length is invalid"
        )));
    }
    let offset = writer.stream_position()?;
    writer.write_all(&compressed)?;
    Ok(BlockRef {
        offset,
        len: u32::try_from(compressed.len())
            .map_err(|_| Error::invariant("text v4 compressed block exceeds u32"))?,
        raw_len: u32::try_from(raw.len())
            .map_err(|_| Error::invariant("text v4 raw block exceeds u32"))?,
        compressed_crc32: crc32fast::hash(&compressed),
        raw_xxh3: non_zero_xxh3(raw),
    })
}

fn decode_block(compressed: &[u8], reference: &BlockRef, what: &str) -> Result<Vec<u8>> {
    require_len(compressed, reference.len as usize, what)?;
    if crc32fast::hash(compressed) != reference.compressed_crc32 {
        return Err(Error::invariant(format!(
            "text v4 {what} compressed checksum mismatch"
        )));
    }
    validate_block_limits(reference)?;
    let decoder = zstd::stream::read::Decoder::new(compressed)
        .map_err(|error| Error::invariant(format!("text v4 {what} decoder failed: {error}")))?;
    let mut raw = Vec::with_capacity(reference.raw_len as usize);
    decoder
        .take(u64::from(reference.raw_len) + 1)
        .read_to_end(&mut raw)
        .map_err(|error| Error::invariant(format!("text v4 {what} decode failed: {error}")))?;
    require_len(&raw, reference.raw_len as usize, what)?;
    if non_zero_xxh3(&raw) != reference.raw_xxh3 {
        return Err(Error::invariant(format!(
            "text v4 {what} raw checksum mismatch"
        )));
    }
    Ok(raw)
}

fn validate_block_limits(reference: &BlockRef) -> Result<()> {
    if reference.len == 0
        || u64::from(reference.len) > MAX_COMPRESSED_BLOCK_BYTES
        || reference.raw_len == 0
        || u64::from(reference.raw_len) > MAX_RAW_BLOCK_BYTES
        || reference.raw_xxh3 == 0
    {
        return Err(Error::invariant(
            "text v4 block reference exceeds wire limits",
        ));
    }
    Ok(())
}

fn validate_footer(
    footer: &Footer,
    footer_offset: u64,
    state: &SearchLsmState,
    segment: &SearchSegmentRef,
) -> Result<()> {
    if footer.footer_version != FOOTER_VERSION || state.kind != SearchLsmKind::Text {
        return Err(Error::invariant(
            "text v4 footer version/generation kind is unsupported",
        ));
    }
    footer.binding.validate(state, segment)?;
    if footer.binding.version_table.offset != MAGIC_V4.len() as u64 {
        return Err(Error::invariant(
            "text v4 version table does not immediately follow magic",
        ));
    }
    let expected_digest = content_digest(
        &footer.binding.version_table,
        footer.delta_docs,
        footer.delta_total_len,
        &footer.doc_table,
        &footer.postings_region,
        &footer.dictionary,
        &footer.filters,
    )?;
    let expected_stats =
        text_segment_stats(segment.role, footer.delta_docs, footer.delta_total_len)?;
    if expected_digest != segment.content_xxh3
        || footer.doc_table.row_count != segment.live_payload_count
        || segment.complete_filter_properties
            != footer
                .filters
                .iter()
                .map(|filter| filter.property.clone())
                .collect::<Vec<_>>()
        || segment.stats != expected_stats
        || (segment.role == SearchSegmentRole::Base
            && (segment.suppress_count != 0
                || segment.mutation_count != segment.live_payload_count
                || u64::try_from(footer.delta_docs).ok() != Some(footer.doc_table.row_count)))
    {
        return Err(Error::invariant(
            "text v4 footer disagrees with segment statistics/content",
        ));
    }
    validate_layout(footer, footer_offset)
}

pub(super) fn text_segment_stats(
    role: SearchSegmentRole,
    documents: i64,
    total_len: i64,
) -> Result<SearchSegmentStats> {
    let (doc_count, total_len) = match role {
        SearchSegmentRole::Base => (
            SearchStatValue::Absolute(u64::try_from(documents).map_err(|_| {
                Error::invariant("text v4 base document count is negative or exceeds u64")
            })?),
            SearchStatValue::Absolute(u64::try_from(total_len).map_err(|_| {
                Error::invariant("text v4 base total length is negative or exceeds u64")
            })?),
        ),
        SearchSegmentRole::Delta => (
            SearchStatValue::Delta(documents),
            SearchStatValue::Delta(total_len),
        ),
    };
    Ok(SearchSegmentStats::Text {
        doc_count,
        total_len,
        term_df_violation_count: 0,
    })
}

fn validate_layout(footer: &Footer, footer_offset: u64) -> Result<()> {
    let version_end = footer
        .binding
        .version_table
        .offset
        .checked_add(footer.binding.version_table.len)
        .ok_or_else(|| Error::invariant("text v4 version range overflows"))?;
    let expected_doc_len = footer
        .doc_table
        .row_count
        .checked_mul(DOC_RECORD_LEN)
        .ok_or_else(|| Error::invariant("text v4 document table length overflows"))?;
    if footer.doc_table.offset != version_end
        || footer.doc_table.len != expected_doc_len
        || footer.doc_table.content_xxh3 == 0
    {
        return Err(Error::invariant(
            "text v4 document table reference is inconsistent",
        ));
    }
    let doc_end = footer
        .doc_table
        .offset
        .checked_add(footer.doc_table.len)
        .ok_or_else(|| Error::invariant("text v4 document table range overflows"))?;
    if footer.postings_region.offset != doc_end || footer.postings_region.metadata_xxh3 == 0 {
        return Err(Error::invariant(
            "text v4 postings region reference is inconsistent",
        ));
    }
    let mut previous_end = footer
        .postings_region
        .offset
        .checked_add(footer.postings_region.len)
        .ok_or_else(|| Error::invariant("text v4 postings region overflows"))?;
    if previous_end > footer_offset {
        return Err(Error::invariant("text v4 postings region overlaps footer"));
    }
    let mut previous_term: Option<&str> = None;
    for dictionary in &footer.dictionary {
        if dictionary.term_count == 0
            || dictionary.first_term.is_empty()
            || dictionary.first_term > dictionary.last_term
            || previous_term.is_some_and(|term| term >= dictionary.first_term.as_str())
        {
            return Err(Error::invariant(
                "text v4 sparse dictionary is not strictly ordered",
            ));
        }
        let range = dictionary.wire.range()?;
        validate_block_limits(&dictionary.wire)?;
        if range.start != previous_end || range.end > footer_offset {
            return Err(Error::invariant(
                "text v4 dictionary block ranges are not contiguous",
            ));
        }
        previous_end = range.end;
        previous_term = Some(&dictionary.last_term);
    }
    let mut previous_property: Option<&str> = None;
    for filter in &footer.filters {
        if filter.property.is_empty()
            || previous_property.is_some_and(|property| property >= filter.property.as_str())
        {
            return Err(Error::invariant(
                "text v4 filter directory is not strictly ordered",
            ));
        }
        validate_filter_directory(filter, footer.doc_table.row_count)?;
        for value in &filter.values {
            let range = value.wire.range()?;
            if range.start != previous_end || range.end > footer_offset {
                return Err(Error::invariant(
                    "text v4 filter posting block ranges are not contiguous",
                ));
            }
            previous_end = range.end;
        }
        previous_property = Some(&filter.property);
    }
    if previous_end != footer_offset {
        return Err(Error::invariant(
            "text v4 payload blocks do not end at footer",
        ));
    }
    Ok(())
}

fn content_digest(
    version_table: &SearchVersionTableRef,
    delta_docs: i64,
    delta_total_len: i64,
    doc_table: &DocTableRef,
    postings_region: &RegionRef,
    dictionary: &[DictionaryBlockRef],
    filters: &[FilterBlockRef],
) -> Result<u64> {
    let material = ContentDigestMaterial {
        domain: CONTENT_DOMAIN,
        format_version: FORMAT_VERSION,
        version_table,
        delta_docs,
        delta_total_len,
        doc_table,
        postings_region,
        dictionary,
        filters,
    };
    let encoded = serialize_bounded(&material, MAX_FOOTER_BYTES, "text v4 content digest")?;
    Ok(non_zero_xxh3(&encoded))
}

fn decode_trailer(bytes: &[u8]) -> Result<(u64, u32)> {
    if bytes.len() != TRAILER_LEN || &bytes[..8] != TRAILER_MAGIC {
        return Err(Error::invariant("text v4 trailer magic/length mismatch"));
    }
    Ok((read_u64(bytes, 8)?, read_u32(bytes, 16)?))
}

fn serialize_bounded<T: Serialize + ?Sized>(value: &T, limit: u64, what: &str) -> Result<Vec<u8>> {
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
            "text v4 {what} range returned {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error::invariant("text v4 u32 offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invariant("text v4 u32 is out of bounds"))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("checked text v4 u32"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::invariant("text v4 u64 offset overflows"))?;
    let value = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invariant("text v4 u64 is out of bounds"))?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("checked text v4 u64"),
    ))
}

fn non_zero_xxh3(bytes: &[u8]) -> u64 {
    xxh3_64(bytes).max(1)
}

fn non_zero_digest(hasher: Xxh3) -> u64 {
    hasher.digest().max(1)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use async_trait::async_trait;

    use super::*;
    use crate::search_lsm::{SearchLsmStatus, SearchSegmentPayload};
    use crate::text::parse_query;

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
                .ok_or_else(|| Error::invariant("test range leaves text v4 body"))
        }
    }

    fn node(value: u64) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[8..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn state() -> SearchLsmState {
        SearchLsmState {
            index_name: "law_fts".into(),
            kind: SearchLsmKind::Text,
            catalog_signature: "text-catalog-v1".into(),
            generation_id: Uuid::from_u128(11),
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        }
    }

    fn context() -> TextV4BuildContext {
        TextV4BuildContext {
            sst_id: Uuid::from_u128(12),
            event_ranges: vec![SearchEventRange::new(20, 24)],
            complete_filter_properties: vec!["ambito".into(), "vigente".into()],
        }
    }

    fn payload(text: &str, vigente: bool, ambito: &str) -> TextV4Payload {
        TextV4Payload {
            text: text.into(),
            filters: BTreeMap::from([
                ("vigente".into(), SearchFilterValue::Bool(vigente)),
                ("ambito".into(), SearchFilterValue::String(ambito.into())),
            ]),
        }
    }

    fn base_state() -> SearchLsmState {
        SearchLsmState {
            base_frontier: Some(1),
            next_event_seq: 1,
            ..state()
        }
    }

    fn base_context(sst_id: u128) -> TextV4BuildContext {
        TextV4BuildContext {
            sst_id: Uuid::from_u128(sst_id),
            event_ranges: vec![SearchEventRange::new(0, 1)],
            complete_filter_properties: vec!["ambito".into(), "vigente".into()],
        }
    }

    async fn base_global_stats(reader: &TextV4Reader, query: &TextQuery) -> TextV4GlobalStats {
        let mut terms = query
            .base_terms()
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        for prefix in &query.prefixes {
            terms.extend(reader.expand_prefix(prefix).await.unwrap());
        }
        let document_frequency = {
            let mut values = BTreeMap::new();
            for term in terms {
                values.insert(
                    term.clone(),
                    u64::try_from(reader.term_delta_df(&term).await.unwrap()).unwrap(),
                );
            }
            values
        };
        let SearchSegmentStats::Text {
            doc_count: SearchStatValue::Absolute(document_count),
            total_len: SearchStatValue::Absolute(total_document_len),
            term_df_violation_count: 0,
        } = reader.segment().stats
        else {
            panic!("test reader must be an authoritative base");
        };
        TextV4GlobalStats {
            document_count,
            total_document_len,
            document_frequency,
        }
    }

    fn assert_search_parity(exhaustive: &TextV4SearchResult, block_max: &TextV4SearchResult) {
        assert_eq!(
            exhaustive.applied_filter_groups,
            block_max.applied_filter_groups
        );
        assert_eq!(exhaustive.hits.len(), block_max.hits.len());
        for (expected, actual) in exhaustive.hits.iter().zip(&block_max.hits) {
            assert_eq!(expected.node_id, actual.node_id);
            assert_eq!(expected.lsn, actual.lsn);
            assert_eq!(expected.payload_fingerprint, actual.payload_fingerprint);
            assert_eq!(expected.score.to_bits(), actual.score.to_bits());
        }
    }

    fn mutations() -> Vec<TextV4Mutation> {
        vec![
            TextV4Mutation {
                node_id: node(3),
                lsn: 23,
                before: Some(payload("contrato temporal derogado", false, "laboral")),
                after: None,
            },
            TextV4Mutation {
                node_id: node(1),
                lsn: 21,
                before: None,
                after: Some(payload("contrato laboral vigente salario", true, "laboral")),
            },
            TextV4Mutation {
                node_id: node(2),
                lsn: 22,
                before: Some(payload("norma civil antigua", false, "civil")),
                after: Some(payload("norma laboral vigente contrato", true, "laboral")),
            },
            TextV4Mutation {
                node_id: node(4),
                lsn: 24,
                before: Some(payload("sin cambio", true, "civil")),
                after: Some(payload("SIN   CAMBIO", true, "civil")),
            },
        ]
    }

    #[tokio::test]
    async fn deterministic_round_trip_signed_stats_filters_tombstone_and_phrase_search() {
        let options = TextV4BuildOptions {
            postings_per_block: 1,
            terms_per_dictionary_block: 2,
            compression_level: 1,
        };
        let first = build_delta_v4(&state(), context(), mutations(), options)
            .unwrap()
            .unwrap();
        let second = build_delta_v4(&state(), context(), mutations(), options)
            .unwrap()
            .unwrap();
        assert_eq!(first.body, second.body);
        assert_eq!(first.output, second.output);
        let legacy_cursor = Cursor::new(Vec::new());
        let (legacy_cursor, legacy_output) =
            write_delta_v4_in_memory(legacy_cursor, &state(), context(), mutations(), options)
                .unwrap()
                .unwrap();
        assert_eq!(first.body.as_ref(), legacy_cursor.into_inner());
        assert_eq!(first.output, legacy_output);
        assert_eq!(first.output.segment.mutation_count, 3);
        assert_eq!(first.output.segment.live_payload_count, 2);
        assert_eq!(first.output.segment.suppress_count, 1);
        assert_eq!(
            first.output.segment.stats,
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Delta(0),
                total_len: SearchStatValue::Delta(2),
                term_df_violation_count: 0,
            }
        );

        let reader = TextV4Reader::open(
            Arc::new(MemorySource {
                body: first.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            first.body.len() as u64,
            &state(),
            &first.output.segment,
        )
        .await
        .unwrap();
        assert_eq!(reader.term_delta_df("contrato").await.unwrap(), 1);
        assert_eq!(reader.term_delta_df("vigente").await.unwrap(), 2);
        assert_eq!(reader.term_delta_df("derogado").await.unwrap(), -1);
        let mut term_deltas = Vec::new();
        for block in 0..reader.term_delta_block_count() {
            let page = reader.read_term_delta_block(block).await.unwrap();
            assert!(
                page.len() <= options.terms_per_dictionary_block,
                "one read must remain bounded by one dictionary block"
            );
            term_deltas.extend(page);
        }
        assert!(term_deltas
            .windows(2)
            .all(|pair| pair[0].term < pair[1].term));
        assert!(term_deltas
            .iter()
            .any(|entry| entry.term == "derogado" && entry.delta_df == -1));
        assert!(reader
            .read_term_delta_block(reader.term_delta_block_count())
            .await
            .is_err());

        let query = parse_query("\"contrato laboral\" vigente");
        let result = reader
            .search_query_exact(
                &query,
                &TextV4GlobalStats {
                    document_count: 100,
                    total_document_len: 1_000,
                    document_frequency: BTreeMap::from([
                        ("contrato".into(), 10),
                        ("laboral".into(), 20),
                        ("vigente".into(), 5),
                    ]),
                },
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
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].node_id, node(1));
        let guarded = reader
            .search_query_base_block_max_exact(
                &query,
                &TextV4GlobalStats {
                    document_count: 100,
                    total_document_len: 1_000,
                    document_frequency: BTreeMap::from([
                        ("contrato".into(), 10),
                        ("laboral".into(), 20),
                        ("vigente".into(), 5),
                    ]),
                },
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
        assert_search_parity(&result, &guarded);
        assert_eq!(guarded.block_max_fallback, Some("non_authoritative_base"));
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
    async fn authoritative_base_round_trip_uses_absolute_stats_and_exact_search() {
        let documents = vec![
            TextV4Mutation {
                node_id: node(2),
                lsn: 42,
                before: None,
                after: Some(payload("norma civil vigente", true, "civil")),
            },
            TextV4Mutation {
                node_id: node(1),
                lsn: 41,
                before: None,
                after: Some(payload("contrato laboral vigente salario", true, "laboral")),
            },
        ];
        let artifact = build_base_v4(
            &state(),
            context(),
            documents,
            TextV4BuildOptions {
                postings_per_block: 1,
                terms_per_dictionary_block: 2,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(artifact.output.segment.role, SearchSegmentRole::Base);
        assert_eq!(artifact.output.segment.mutation_count, 2);
        assert_eq!(artifact.output.segment.live_payload_count, 2);
        assert_eq!(artifact.output.segment.suppress_count, 0);
        assert_eq!(
            artifact.output.segment.stats,
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Absolute(2),
                total_len: SearchStatValue::Absolute(7),
                term_df_violation_count: 0,
            }
        );

        let reader = TextV4Reader::open(
            Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &state(),
            &artifact.output.segment,
        )
        .await
        .unwrap();
        assert_eq!(reader.delta_docs(), 2);
        assert_eq!(reader.delta_total_len(), 7);
        assert_eq!(reader.term_delta_df("vigente").await.unwrap(), 2);
        assert_eq!(reader.term_delta_df("contrato").await.unwrap(), 1);
        let result = reader
            .search_query_exact(
                &parse_query("contrato vigente"),
                &TextV4GlobalStats {
                    document_count: 2,
                    total_document_len: 7,
                    document_frequency: BTreeMap::from([
                        ("contrato".into(), 1),
                        ("vigente".into(), 2),
                    ]),
                },
                10,
                &[(
                    "ambito".into(),
                    vec![SearchFilterValue::String("laboral".into())],
                )],
            )
            .await
            .unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].node_id, node(1));
        reader.verify_all().await.unwrap();

        let before_image = TextV4Mutation {
            node_id: node(1),
            lsn: 1,
            before: Some(payload("viejo", false, "civil")),
            after: Some(payload("nuevo", true, "laboral")),
        };
        assert!(build_base_v4(
            &state(),
            context(),
            vec![before_image],
            TextV4BuildOptions::default(),
        )
        .is_err());
        let suppression = TextV4Mutation {
            node_id: node(1),
            lsn: 1,
            before: None,
            after: None,
        };
        assert!(build_base_v4(
            &state(),
            context(),
            vec![suppression],
            TextV4BuildOptions::default(),
        )
        .is_err());
    }

    #[tokio::test]
    async fn authoritative_base_block_max_matches_exhaustive_for_all_query_shapes() {
        let documents = (1..=64)
            .map(|value| {
                let text = match value {
                    // max_tf=4 belongs to this long document.
                    1 => "rare rare rare rare common phrase alpha alphanumeric extra words",
                    // min_doc_len=1 belongs to a different document in the
                    // same rare posting block.
                    2 => "rare",
                    3 => "phrase alpha common tie",
                    64 => "alpine common tail tie",
                    _ => "common tail tie",
                };
                TextV4Mutation {
                    node_id: node(value),
                    lsn: value,
                    before: None,
                    after: Some(payload(
                        text,
                        value % 2 == 0,
                        if value % 3 == 0 { "civil" } else { "laboral" },
                    )),
                }
            })
            .collect::<Vec<_>>();
        let artifact = build_base_v4(
            &base_state(),
            base_context(0xB10C),
            documents,
            TextV4BuildOptions {
                postings_per_block: 4,
                terms_per_dictionary_block: 3,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        let reader = TextV4Reader::open(
            Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &base_state(),
            &artifact.output.segment,
        )
        .await
        .unwrap();

        let rare = reader.lookup_term("rare").await.unwrap().unwrap();
        assert_eq!(rare.blocks[0].max_tf, 4);
        assert_eq!(rare.blocks[0].min_doc_len, 1);
        let filters = vec![("vigente".into(), vec![SearchFilterValue::Bool(true)])];
        let cases = vec![
            ("ordinary", parse_query("rare common"), 1, Vec::new()),
            ("phrase", parse_query("\"phrase alpha\""), 5, Vec::new()),
            ("prefix", parse_query("alp*"), 5, Vec::new()),
            ("filter", parse_query("common"), 5, filters),
            ("ties", parse_query("tie"), 3, Vec::new()),
            ("k_above_matches", parse_query("rare"), 10, Vec::new()),
        ];
        let mut skipped = 0usize;
        for (name, query, k, groups) in cases {
            let global = base_global_stats(&reader, &query).await;
            let exhaustive = reader
                .search_query_exact(&query, &global, k, &groups)
                .await
                .unwrap();
            let block_max = reader
                .search_query_base_block_max_exact(&query, &global, k, &groups)
                .await
                .unwrap();
            assert_search_parity(&exhaustive, &block_max);
            assert!(block_max.block_max_pruning);
            assert_eq!(block_max.block_max_fallback, None);
            assert!(block_max.posting_blocks_fetched <= exhaustive.posting_blocks_fetched);
            if name == "ordinary" {
                assert!(block_max.posting_blocks_skipped > 0);
                assert!(block_max.posting_blocks_fetched < exhaustive.posting_blocks_fetched);
            }
            if name == "ties" {
                assert_eq!(
                    block_max.posting_blocks_skipped, 0,
                    "an upper bound equal to the kth score must not be pruned"
                );
            }
            skipped = skipped.saturating_add(block_max.posting_blocks_skipped);
        }
        assert!(
            skipped > 0,
            "the high-score prefix must prune low-score common tail blocks"
        );
    }

    #[tokio::test]
    async fn block_max_fallback_is_exact_and_corrupt_bound_metadata_fails_closed() {
        let documents = vec![
            TextV4Mutation {
                node_id: node(1),
                lsn: 1,
                before: None,
                after: Some(payload("rare rare common", true, "laboral")),
            },
            TextV4Mutation {
                node_id: node(2),
                lsn: 2,
                before: None,
                after: Some(payload("common", false, "civil")),
            },
        ];
        let artifact = build_base_v4(
            &base_state(),
            base_context(0xFA11),
            documents,
            TextV4BuildOptions {
                postings_per_block: 1,
                terms_per_dictionary_block: 2,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        let reader = TextV4Reader::open(
            Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &base_state(),
            &artifact.output.segment,
        )
        .await
        .unwrap();
        let query = parse_query("rare common");
        let global = base_global_stats(&reader, &query).await;
        let mut mismatched = global.clone();
        mismatched.document_count += 1;
        let exhaustive = reader
            .search_query_exact(&query, &mismatched, 2, &[])
            .await
            .unwrap();
        let fallback = reader
            .search_query_base_block_max_exact(&query, &mismatched, 2, &[])
            .await
            .unwrap();
        assert_search_parity(&exhaustive, &fallback);
        assert!(!fallback.block_max_pruning);
        assert_eq!(fallback.block_max_fallback, Some("non_authoritative_base"));

        let reference = reader.footer.dictionary[0].wire.clone();
        let mut corrupt = artifact.body.to_vec();
        corrupt[reference.offset as usize] ^= 0x40;
        let corrupt_reader = TextV4Reader::open(
            Arc::new(MemorySource {
                body: Bytes::from(corrupt),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &base_state(),
            &artifact.output.segment,
        )
        .await
        .unwrap();
        assert!(
            corrupt_reader.read_term_delta_block(0).await.is_err(),
            "dictionary pagination must fail closed on authenticated block corruption"
        );
        assert!(
            corrupt_reader
                .search_query_base_block_max_exact(&query, &global, 2, &[])
                .await
                .is_err(),
            "authenticated dictionary/bound corruption must fail closed"
        );

        let invalid = PostingBlockRef {
            first_doc: 0,
            last_doc: 0,
            posting_count: 1,
            max_tf: 1,
            min_doc_len: 1,
            wire: BlockRef {
                offset: 0,
                len: 1,
                raw_len: 1,
                compressed_crc32: 1,
                raw_xxh3: 1,
            },
        };
        assert!(conservative_block_upper_bound(f64::NAN, &invalid, 1.0).is_none());
        assert!(next_up_nonnegative(f64::MAX).is_none());
    }

    #[test]
    fn canonical_analysis_elides_noops_and_adversarial_inputs_fail_closed() {
        let no_op = TextV4Mutation {
            node_id: node(1),
            lsn: 1,
            before: Some(payload("NORMA   LABORAL", true, "laboral")),
            after: Some(payload("norma laboral", true, "laboral")),
        };
        assert!(build_delta_v4(
            &state(),
            context(),
            vec![no_op],
            TextV4BuildOptions::default(),
        )
        .unwrap()
        .is_none());

        let mut duplicate = mutations();
        duplicate.push(duplicate[0].clone());
        assert!(build_delta_v4(
            &state(),
            context(),
            duplicate,
            TextV4BuildOptions::default(),
        )
        .is_err());
        assert!(build_delta_v4(
            &state(),
            context(),
            mutations(),
            TextV4BuildOptions {
                postings_per_block: 0,
                ..TextV4BuildOptions::default()
            },
        )
        .is_err());
    }

    #[tokio::test]
    async fn corruption_and_manifest_drift_are_rejected() {
        let artifact = build_delta_v4(
            &state(),
            context(),
            mutations(),
            TextV4BuildOptions::default(),
        )
        .unwrap()
        .unwrap();
        let mut drift = artifact.output.segment.clone();
        drift.payload = SearchSegmentPayload::ShadowOnly;
        assert!(TextV4Reader::open(
            Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &state(),
            &drift,
        )
        .await
        .is_err());

        let posting_offset = {
            let reader = TextV4Reader::open(
                Arc::new(MemorySource {
                    body: artifact.body.clone(),
                    ranges: Mutex::new(Vec::new()),
                }),
                artifact.body.len() as u64,
                &state(),
                &artifact.output.segment,
            )
            .await
            .unwrap();
            reader.footer.postings_region.offset as usize
        };
        let mut corrupt = artifact.body.to_vec();
        corrupt[posting_offset] ^= 0x40;
        let reader = TextV4Reader::open(
            Arc::new(MemorySource {
                body: Bytes::from(corrupt),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &state(),
            &artifact.output.segment,
        )
        .await
        .unwrap();
        assert!(reader.verify_all().await.is_err());
    }

    #[tokio::test]
    async fn hundred_thousand_candidate_phrase_search_keeps_one_block_per_term() {
        let document_count = 100_000u64;
        let stream_context = TextV4BuildContext {
            sst_id: Uuid::from_u128(13),
            event_ranges: vec![SearchEventRange::new(1, document_count + 1)],
            complete_filter_properties: Vec::new(),
        };
        let mutations = (1..=document_count)
            .map(|value| TextV4Mutation {
                node_id: node(value),
                lsn: value,
                before: None,
                after: Some(TextV4Payload {
                    text: if value % 2 == 0 {
                        "comun legal".into()
                    } else {
                        "legal comun".into()
                    },
                    filters: BTreeMap::new(),
                }),
            })
            .collect();
        let artifact = build_delta_v4(
            &state(),
            stream_context,
            mutations,
            TextV4BuildOptions {
                postings_per_block: 64,
                terms_per_dictionary_block: 8,
                compression_level: 1,
            },
        )
        .unwrap()
        .unwrap();
        let reader = TextV4Reader::open(
            Arc::new(MemorySource {
                body: artifact.body.clone(),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.body.len() as u64,
            &state(),
            &artifact.output.segment,
        )
        .await
        .unwrap();
        let result = reader
            .search_query_exact(
                &parse_query("\"comun legal\""),
                &TextV4GlobalStats {
                    document_count,
                    total_document_len: document_count * 2,
                    document_frequency: BTreeMap::from([
                        ("comun".into(), document_count),
                        ("legal".into(), document_count),
                    ]),
                },
                10,
                &[],
            )
            .await
            .unwrap();
        assert_eq!(result.postings_decoded, document_count as usize * 2);
        assert_eq!(result.hits.len(), 10);
        let idf = bm25_idf(document_count as usize, document_count as usize);
        let expected_score = bm25_term_score(idf, 1, 2, 2.0) + bm25_term_score(idf, 1, 2, 2.0);
        assert!(
            result
                .hits
                .iter()
                .all(|hit| hit.score.to_bits() == expected_score.to_bits()),
            "streaming phrase scorer must remain bit-exact"
        );
        assert_eq!(
            result
                .hits
                .iter()
                .map(|hit| hit.node_id)
                .collect::<Vec<_>>(),
            (1..=10)
                .map(|ordinal| node(ordinal * 2))
                .collect::<Vec<_>>(),
            "equal-score top-k must retain the lowest document ordinals"
        );
        assert!(
            result.peak_live_bytes < 1024 * 1024,
            "common-phrase query retained {} bytes",
            result.peak_live_bytes
        );
    }

    #[test]
    fn adaptive_query_mask_unions_ten_thousand_values_without_corpus_bitmap() {
        const VALUES: u64 = 10_000;
        const DOCS_PER_VALUE: u64 = 10;
        const CANDIDATES: usize = (VALUES * DOCS_PER_VALUE) as usize;
        // A sparse 100k-candidate result is ~0.8 MiB; a corpus bitmap would be
        // ~12 MiB. The accumulator must stay proportional to candidates and
        // sort only once after ingesting all 10k value postings.
        const ROW_COUNT: u64 = 100_000_000;

        let mut builder = QueryMaskUnionBuilder::new(ROW_COUNT);
        let mut peak = 0usize;
        for value in 0..VALUES {
            let ordinals = (0..DOCS_PER_VALUE)
                .map(|slot| slot * VALUES + value)
                .collect::<Vec<_>>();
            peak = peak.max(builder.absorb(QueryMask::Sparse(ordinals)).unwrap());
        }
        let (mask, finish_peak) = builder.finish().unwrap();
        peak = peak.max(finish_peak);
        let QueryMask::Sparse(ordinals) = mask else {
            panic!("100k sparse candidates must not expand to a corpus bitmap");
        };
        assert_eq!(ordinals.len(), CANDIDATES);
        assert!(ordinals.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(ordinals.first(), Some(&0));
        assert_eq!(ordinals.last(), Some(&((VALUES * DOCS_PER_VALUE) - 1)));
        assert!(
            peak < 4 * 1024 * 1024,
            "10k-value sparse union used {peak} modeled bytes"
        );

        // The same candidate set over a genuinely small corpus should choose
        // the smaller dense form, proving the representation is adaptive
        // rather than hard-wired to sparse.
        let mut dense_builder = QueryMaskUnionBuilder::new(CANDIDATES as u64);
        for start in (0..CANDIDATES as u64).step_by(10) {
            dense_builder
                .absorb(QueryMask::Sparse(
                    (start..(start + 10).min(CANDIDATES as u64)).collect(),
                ))
                .unwrap();
        }
        let (dense, _) = dense_builder.finish().unwrap();
        assert!(
            matches!(dense, QueryMask::Dense(_)),
            "dense mask must win only when it is the smaller exact form"
        );
    }
}
