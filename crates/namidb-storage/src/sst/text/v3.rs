//! Range-readable full-text SST format (`NAMIFT03`).
//!
//! The v2 format is one bincode object, so answering one term requires
//! downloading and decoding the complete corpus index. V3 deliberately keeps
//! the immutable object self-contained while making its hot pieces
//! independently addressable:
//!
//! ```text
//! +----------+----------------+-------------------+--------+---------+
//! | magic    | fixed docs     | posting blocks    | dict   | footer  |
//! +----------+----------------+-------------------+--------+---------+
//!                                                        + trailer
//! ```
//!
//! The trailer locates a compact footer. The footer carries corpus statistics,
//! the fixed-width document table, and a sparse lexicographic directory. A
//! query reads the footer, only the dictionary blocks that can contain its
//! terms, only those terms' posting blocks, and finally the NodeIds of the
//! winning ordinals. Posting and dictionary blocks are independently zstd
//! compressed and checksummed.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bincode::Options;
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::{TextIndexBody, TextIndexBuildStats};
use crate::error::Error;
use crate::search_workspace::{
    search_max_result_bytes, shared_search_workspace,
    MATERIALISED_TEXT_RESULT_BYTES_PER_HIT as MATERIALISED_RESULT_BYTES_PER_HIT,
};
use crate::text::{
    avg_len, bm25_idf, bm25_term_score, tokenize, TextQuery, PREFIX_EXPANSION_LIMIT,
};

#[path = "v3_external.rs"]
mod external;

pub use external::{
    ExternalTextIndexBuildMetrics, ExternalTextIndexBuildOptions, TextIndexExternalBuilder,
    TextIndexFileArtifact, COMPACTION_SPOOL_DIR_ENV, INDEX_BUILD_MEMORY_ENV,
};

pub(super) const MAGIC_V3: &[u8; 8] = b"NAMIFT03";
const TRAILER_MAGIC: &[u8; 8] = b"NFT3END!";
const TRAILER_LEN: usize = 8 + 8 + 4;
const DOC_RECORD_LEN: u64 = 16 + 4;
const FORMAT_VERSION: u16 = 3;
const TERMS_PER_DICTIONARY_BLOCK: usize = 128;
const POSTINGS_PER_BLOCK: usize = 256;
const ZSTD_LEVEL: i32 = 3;
const MAX_FOOTER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_BLOCK_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RAW_BLOCK_BYTES: u64 = 256 * 1024 * 1024;
// One encoded position can occupy one byte but becomes a u32. Include the raw
// decompression buffer and posting/vector headers as well as decoded positions.
const POSTING_DECODE_WORKSPACE_MULTIPLIER: usize = 5;
// A dictionary block is present compressed, decompressed, deserialised, and
// selected entries may briefly be cloned into the query plan.
const DICTIONARY_DECODE_WORKSPACE_MULTIPLIER: usize = 4;
// zstd's decoder keeps tables/window state outside the returned raw Vec. Only
// one block is decompressed at a time.
const ZSTD_DECODER_WORKSPACE_BYTES: usize = 1024 * 1024;
const DOC_ID_RANGE_BATCH_SIZE: usize = 16;
const SCORED_TERM_WORKSPACE_OVERHEAD_BYTES: usize = 64;

/// A source capable of fetching immutable byte ranges from one `.ft` object.
///
/// Integrations normally wrap `object_store::ObjectStore::get_range` and
/// override [`Self::read_ranges`] with `get_ranges`, so all posting blocks for
/// a query are fetched concurrently/coalesced. The trait intentionally knows
/// nothing about manifests or caches: a page-cache implementation can wrap it
/// without coupling the format reader to the engine read path.
#[async_trait]
pub trait TextIndexRangeSource: Send + Sync {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error>;

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
        let mut out = Vec::with_capacity(ranges.len());
        for range in ranges {
            out.push(self.read_range(range.clone()).await?);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlockRef {
    offset: u64,
    len: u32,
    raw_len: u32,
    crc32: u32,
}

impl BlockRef {
    fn range(&self) -> Result<Range<u64>, Error> {
        let end = self
            .offset
            .checked_add(self.len as u64)
            .ok_or_else(|| Error::invariant("text v3 block range overflows u64"))?;
        Ok(self.offset..end)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostingBlockRef {
    wire: BlockRef,
    first_doc: u32,
    last_doc: u32,
    max_tf: u32,
    min_doc_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TermEntry {
    term: String,
    doc_freq: u32,
    blocks: Vec<PostingBlockRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DictionaryBlockRef {
    first_term: String,
    last_term: String,
    wire: BlockRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Footer {
    format_version: u16,
    n_docs: u32,
    total_len: u64,
    doc_table_offset: u64,
    doc_table_len: u64,
    term_count: u64,
    min_node_id: [u8; 16],
    max_node_id: [u8; 16],
    dictionary: Vec<DictionaryBlockRef>,
}

#[derive(Debug, Clone)]
struct Posting {
    doc: u32,
    tf: u32,
    doc_len: u32,
    positions: Vec<u32>,
}

#[derive(Debug)]
struct PostingCursor {
    term: String,
    blocks: Vec<PostingBlockRef>,
    next_block: usize,
    current: Vec<Posting>,
    position: usize,
    decoded_count: usize,
    previous_doc: Option<u32>,
    declared_doc_freq: usize,
    scored_idf: Option<f64>,
}

impl PostingCursor {
    fn current(&self) -> Option<&Posting> {
        self.current.get(self.position)
    }

    fn advance_current(&mut self) -> Result<bool, Error> {
        if self.current().is_none() {
            return Err(Error::invariant(format!(
                "text v3 cursor for {:?} advanced without a posting",
                self.term
            )));
        }
        self.position += 1;
        if self.position < self.current.len() {
            return Ok(false);
        }
        self.current = Vec::new();
        self.position = 0;
        Ok(self.next_block < self.blocks.len())
    }

    fn install_block(
        &mut self,
        block_index: usize,
        meta: &PostingBlockRef,
        postings: Vec<Posting>,
    ) -> Result<(), Error> {
        if block_index != self.next_block || !self.current.is_empty() {
            return Err(Error::invariant(format!(
                "text v3 cursor for {:?} received an out-of-order block",
                self.term
            )));
        }
        validate_posting_block(&postings, meta)?;
        if self
            .previous_doc
            .is_some_and(|previous| postings[0].doc <= previous)
        {
            return Err(Error::invariant(format!(
                "text v3 term {:?} postings are not strictly sorted across blocks",
                self.term
            )));
        }
        let decoded_count = self
            .decoded_count
            .checked_add(postings.len())
            .ok_or_else(|| Error::invariant("text v3 decoded posting count overflows usize"))?;
        if decoded_count > self.declared_doc_freq {
            return Err(Error::invariant(format!(
                "text v3 term {:?} declares df {}, decoded at least {decoded_count}",
                self.term, self.declared_doc_freq
            )));
        }
        if block_index + 1 == self.blocks.len() && decoded_count != self.declared_doc_freq {
            return Err(Error::invariant(format!(
                "text v3 term {:?} declares df {}, decoded {decoded_count}",
                self.term, self.declared_doc_freq
            )));
        }

        self.previous_doc = postings.last().map(|posting| posting.doc);
        self.decoded_count = decoded_count;
        self.next_block += 1;
        self.current = postings;
        self.position = 0;
        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TextV3QueryMetrics {
    posting_blocks_read: usize,
    max_live_posting_blocks: usize,
    max_live_postings: usize,
    documents_evaluated: usize,
    retained_results: usize,
}

#[derive(Debug)]
enum RankCollector {
    TopK {
        k: usize,
        heap: std::collections::BinaryHeap<Reverse<RankedOrdinal>>,
    },
    All {
        values: Vec<RankedOrdinal>,
        limit_bytes: usize,
    },
}

impl RankCollector {
    fn new(k: Option<usize>, result_limit_bytes: usize) -> Self {
        match k {
            Some(k) => Self::TopK {
                k,
                heap: std::collections::BinaryHeap::new(),
            },
            None => Self::All {
                values: Vec::new(),
                limit_bytes: result_limit_bytes,
            },
        }
    }

    fn push(&mut self, value: RankedOrdinal) -> Result<(), Error> {
        match self {
            Self::TopK { k, heap } => {
                heap.push(Reverse(value));
                if heap.len() > *k {
                    heap.pop();
                }
            }
            Self::All {
                values,
                limit_bytes,
            } => {
                let next_len = values.len().saturating_add(1);
                let estimated_bytes = next_len.saturating_mul(MATERIALISED_RESULT_BYTES_PER_HIT);
                if estimated_bytes > *limit_bytes {
                    return Err(Error::SearchResultLimitExceeded {
                        index_kind: "full-text",
                        estimated_bytes,
                        limit_bytes: *limit_bytes,
                    });
                }
                values.push(value);
            }
        }
        Ok(())
    }

    fn len(&self) -> usize {
        match self {
            Self::TopK { heap, .. } => heap.len(),
            Self::All { values, .. } => values.len(),
        }
    }

    fn finish(self) -> Vec<(u32, f64)> {
        let mut ranked = match self {
            Self::TopK { heap, .. } => heap
                .into_iter()
                .map(|Reverse(value)| value)
                .collect::<Vec<_>>(),
            Self::All { values, .. } => values,
        };
        ranked.sort_unstable_by(|left, right| right.cmp(left));
        ranked
            .into_iter()
            .map(|ranked| (ranked.doc, ranked.score))
            .collect()
    }
}

/// Metadata retained by the range reader after opening an index.
///
/// This is proportional to the number of dictionary blocks (one entry per 128
/// terms by default), not to documents, postings, positions, or corpus bytes.
pub struct TextIndexV3Reader {
    source: Arc<dyn TextIndexRangeSource>,
    file_len: u64,
    footer_offset: u64,
    footer: Footer,
}

impl std::fmt::Debug for TextIndexV3Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextIndexV3Reader")
            .field("file_len", &self.file_len)
            .field("footer_offset", &self.footer_offset)
            .field("n_docs", &self.footer.n_docs)
            .field("term_count", &self.footer.term_count)
            .field("dictionary_blocks", &self.footer.dictionary.len())
            .finish()
    }
}

impl TextIndexV3Reader {
    /// Open a v3 object by fetching its 8-byte header, fixed trailer, and
    /// compact footer. No document, term, posting, or position payload is
    /// decoded.
    pub async fn open(source: Arc<dyn TextIndexRangeSource>, file_len: u64) -> Result<Self, Error> {
        let minimum = (MAGIC_V3.len() + TRAILER_LEN) as u64;
        if file_len < minimum {
            return Err(Error::invariant("text v3 body too short"));
        }
        let trailer_start = file_len - TRAILER_LEN as u64;
        let probes = source
            .read_ranges(&[0..MAGIC_V3.len() as u64, trailer_start..file_len])
            .await?;
        if probes.len() != 2 {
            return Err(Error::invariant(
                "text v3 range source returned the wrong probe count",
            ));
        }
        require_exact_len(&probes[0], MAGIC_V3.len(), "header")?;
        if probes[0].as_ref() != MAGIC_V3 {
            return Err(Error::invariant("text v3 magic mismatch"));
        }
        require_exact_len(&probes[1], TRAILER_LEN, "trailer")?;
        let (footer_len, footer_crc) = decode_trailer(&probes[1])?;
        if footer_len == 0 || footer_len > MAX_FOOTER_BYTES {
            return Err(Error::invariant(format!(
                "text v3 footer length {footer_len} is invalid"
            )));
        }
        let footer_offset = trailer_start
            .checked_sub(footer_len)
            .ok_or_else(|| Error::invariant("text v3 footer starts before the object"))?;
        if footer_offset < MAGIC_V3.len() as u64 {
            return Err(Error::invariant("text v3 footer overlaps the header"));
        }
        let footer_bytes = source.read_range(footer_offset..trailer_start).await?;
        require_exact_len(
            &footer_bytes,
            usize::try_from(footer_len)
                .map_err(|_| Error::invariant("text v3 footer does not fit usize"))?,
            "footer",
        )?;
        if crc32fast::hash(&footer_bytes) != footer_crc {
            return Err(Error::invariant("text v3 footer checksum mismatch"));
        }
        let footer: Footer = deserialize_bounded(&footer_bytes, "footer")?;
        validate_footer(&footer, footer_offset)?;
        Ok(Self {
            source,
            file_len,
            footer_offset,
            footer,
        })
    }

    pub fn doc_count(&self) -> u64 {
        self.footer.n_docs as u64
    }

    pub fn term_count(&self) -> u64 {
        self.footer.term_count
    }

    pub fn total_len(&self) -> u64 {
        self.footer.total_len
    }

    pub fn node_id_bounds(&self) -> ([u8; 16], [u8; 16]) {
        (self.footer.min_node_id, self.footer.max_node_id)
    }

    /// Approximate heap bytes retained after opening. Posting, dictionary and
    /// document payload blocks are query-local (or live in the shared page
    /// cache), so this deliberately counts only the sparse footer directory.
    pub fn estimated_resident_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(
                self.footer
                    .dictionary
                    .capacity()
                    .saturating_mul(std::mem::size_of::<DictionaryBlockRef>()),
            )
            .saturating_add(
                self.footer
                    .dictionary
                    .iter()
                    .map(|block| {
                        block
                            .first_term
                            .capacity()
                            .saturating_add(block.last_term.capacity())
                    })
                    .fold(0usize, usize::saturating_add),
            )
    }

    /// Test whether any target NodeId belongs to this corpus without loading
    /// the full fixed-width document table.
    ///
    /// All targets advance one binary-search round together, so a remote
    /// source can issue/coalesce the distinct midpoint ranges concurrently.
    /// The table is NodeId-sorted by the v3 builder.
    pub async fn contains_any_doc(&self, targets: &[[u8; 16]]) -> Result<bool, Error> {
        if targets.is_empty() || self.footer.n_docs == 0 {
            return Ok(false);
        }
        let mut targets = targets
            .iter()
            .copied()
            .filter(|target| {
                *target >= self.footer.min_node_id && *target <= self.footer.max_node_id
            })
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        if targets.is_empty() {
            return Ok(false);
        }

        let mut probes = targets
            .into_iter()
            .map(|target| (target, 0u32, self.footer.n_docs))
            .collect::<Vec<_>>();
        loop {
            let mut mids = probes
                .iter()
                .filter(|(_, low, high)| low < high)
                .map(|(_, low, high)| low + (high - low) / 2)
                .collect::<Vec<_>>();
            if mids.is_empty() {
                return Ok(false);
            }
            mids.sort_unstable();
            mids.dedup();
            let ranges = mids
                .iter()
                .map(|mid| {
                    let start = self.footer.doc_table_offset + *mid as u64 * DOC_RECORD_LEN;
                    start..start + 16
                })
                .collect::<Vec<_>>();
            let values = self.source.read_ranges(&ranges).await?;
            if values.len() != mids.len() {
                return Err(Error::invariant(
                    "text v3 range source returned the wrong membership probe count",
                ));
            }
            let mut midpoint_ids = HashMap::with_capacity(mids.len());
            for (mid, value) in mids.iter().copied().zip(values) {
                require_exact_len(&value, 16, "document membership probe")?;
                let mut id = [0u8; 16];
                id.copy_from_slice(&value);
                midpoint_ids.insert(mid, id);
            }

            for (target, low, high) in &mut probes {
                if *low >= *high {
                    continue;
                }
                let mid = *low + (*high - *low) / 2;
                let id = midpoint_ids.get(&mid).ok_or_else(|| {
                    Error::invariant("text v3 membership midpoint response is absent")
                })?;
                match id.cmp(target) {
                    Ordering::Equal => return Ok(true),
                    Ordering::Less => *low = mid.saturating_add(1),
                    Ordering::Greater => *high = mid,
                }
            }
        }
    }

    /// Search the v3 index without decoding unrelated dictionary entries,
    /// postings, positions, documents, or the complete object.
    pub async fn search(
        &self,
        query_terms: &[String],
        k: Option<usize>,
    ) -> Result<Vec<([u8; 16], f64)>, Error> {
        self.search_query(&TextQuery::from_terms(query_terms), k)
            .await
    }

    /// Full v2-compatible BM25/query-syntax semantics over range reads.
    ///
    /// Relevant terms advance through one posting block at a time in document
    /// order. A finite top-k therefore retains O(terms × posting-block + k)
    /// query memory regardless of corpus size. Only final winners' NodeIds are
    /// fetched from the fixed document table.
    pub async fn search_query(
        &self,
        query: &TextQuery,
        k: Option<usize>,
    ) -> Result<Vec<([u8; 16], f64)>, Error> {
        let mut metrics = TextV3QueryMetrics::default();
        let result = self
            .search_query_inner(query, k, search_max_result_bytes(), &mut metrics)
            .await;
        let workspace = shared_search_workspace().metrics();
        tracing::debug!(
            posting_blocks_read = metrics.posting_blocks_read,
            max_live_posting_blocks = metrics.max_live_posting_blocks,
            max_live_postings = metrics.max_live_postings,
            documents_evaluated = metrics.documents_evaluated,
            retained_results = metrics.retained_results,
            workspace_capacity_bytes = workspace.capacity_bytes,
            workspace_reserved_bytes = workspace.reserved_bytes,
            workspace_peak_reserved_bytes = workspace.peak_reserved_bytes,
            "completed object-native full-text query"
        );
        result
    }

    async fn search_query_inner(
        &self,
        query: &TextQuery,
        k: Option<usize>,
        result_limit_bytes: usize,
        metrics: &mut TextV3QueryMetrics,
    ) -> Result<Vec<([u8; 16], f64)>, Error> {
        if matches!(k, Some(0)) {
            return Ok(Vec::new());
        }

        // Dictionary blocks are independently bounded before they are
        // decompressed/deserialised. Prefix expansion can touch at most two
        // 128-term blocks because it stops after 64 matches.
        let dictionary_workspace = self.dictionary_workspace_bytes(query);
        let dictionary_reservation = shared_search_workspace()
            .reserve("full-text dictionary planning", dictionary_workspace)
            .await?;
        let mut dictionary_cache: HashMap<usize, Vec<TermEntry>> = HashMap::new();
        let mut scored_terms = query
            .base_terms()
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        for prefix in &query.prefixes {
            scored_terms.extend(self.expand_prefix(prefix, &mut dictionary_cache).await?);
        }

        // Phrase tokens are already base terms, but looking up all terms once
        // here also gives phrase evaluation the same decoded posting vectors.
        let mut entries = BTreeMap::<String, TermEntry>::new();
        for term in &scored_terms {
            if let Some(entry) = self.lookup_term(term, &mut dictionary_cache).await? {
                entries.insert(term.clone(), entry);
            }
        }
        for phrase in &query.phrases {
            for term in phrase {
                if entries.contains_key(term) {
                    continue;
                }
                if let Some(entry) = self.lookup_term(term, &mut dictionary_cache).await? {
                    entries.insert(term.clone(), entry);
                }
            }
        }

        // A missing phrase token makes its hard adjacency constraint
        // impossible, exactly like the legacy `phrase_docs` intersection.
        if query
            .phrases
            .iter()
            .flatten()
            .any(|term| !entries.contains_key(term))
        {
            return Ok(Vec::new());
        }
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.footer.n_docs as usize;
        let avgdl = avg_len(self.footer.total_len, n);
        let posting_workspace = posting_query_workspace_bytes(
            &entries,
            self.footer.n_docs as usize,
            k,
            result_limit_bytes,
        );
        // Selected entries are part of the posting-phase estimate. Release all
        // decoded dictionary blocks before waiting for that phase's permit.
        drop(dictionary_cache);
        drop(dictionary_reservation);
        let _posting_reservation = shared_search_workspace()
            .reserve("full-text posting search", posting_workspace)
            .await?;

        let mut cursors = Vec::with_capacity(entries.len());
        for (term, entry) in entries {
            let scored_idf = scored_terms
                .contains(term.as_str())
                .then(|| bm25_idf(n, entry.doc_freq as usize));
            cursors.push(PostingCursor {
                term,
                blocks: entry.blocks,
                next_block: 0,
                current: Vec::new(),
                position: 0,
                decoded_count: 0,
                previous_doc: None,
                declared_doc_freq: entry.doc_freq as usize,
                scored_idf,
            });
        }
        let term_to_cursor = cursors
            .iter()
            .enumerate()
            .map(|(index, cursor)| (cursor.term.clone(), index))
            .collect::<HashMap<_, _>>();

        let first_blocks = (0..cursors.len()).collect::<Vec<_>>();
        self.load_next_posting_blocks(&mut cursors, &first_blocks, metrics)
            .await?;

        let mut ranked = RankCollector::new(k, result_limit_bytes);
        loop {
            let Some(doc) = cursors
                .iter()
                .filter_map(|cursor| cursor.current().map(|posting| posting.doc))
                .min()
            else {
                break;
            };
            metrics.documents_evaluated = metrics.documents_evaluated.saturating_add(1);
            if metrics.documents_evaluated & (crate::cancel::CHECK_STRIDE - 1) == 0 {
                crate::cancel::check()?;
            }

            let phrases_match = query
                .phrases
                .iter()
                .all(|phrase| phrase_matches_cursor_doc(phrase, doc, &term_to_cursor, &cursors));
            if phrases_match {
                let mut score = 0.0;
                let mut matched_scored_term = false;
                // Cursors are in BTreeMap/lexicographic order, matching the
                // legacy per-term floating-point summation order exactly.
                for cursor in &cursors {
                    let Some(idf) = cursor.scored_idf else {
                        continue;
                    };
                    let Some(posting) = cursor.current().filter(|posting| posting.doc == doc)
                    else {
                        continue;
                    };
                    score += bm25_term_score(idf, posting.tf, posting.doc_len as usize, avgdl);
                    matched_scored_term = true;
                }
                if matched_scored_term {
                    ranked.push(RankedOrdinal { doc, score })?;
                    metrics.retained_results = metrics.retained_results.max(ranked.len());
                }
            }

            let mut next_blocks = Vec::new();
            for (index, cursor) in cursors.iter_mut().enumerate() {
                if cursor.current().is_some_and(|posting| posting.doc == doc)
                    && cursor.advance_current()?
                {
                    next_blocks.push(index);
                }
            }
            self.load_next_posting_blocks(&mut cursors, &next_blocks, metrics)
                .await?;
        }

        let ranked = ranked.finish();
        let ids = self
            .read_doc_ids(&ranked.iter().map(|(doc, _)| *doc).collect::<Vec<_>>())
            .await?;
        let mut out = Vec::with_capacity(ranked.len());
        for (doc, score) in ranked {
            let id = ids.get(&doc).copied().ok_or_else(|| {
                Error::invariant(format!("text v3 missing NodeId for document {doc}"))
            })?;
            out.push((id, score));
        }
        Ok(out)
    }

    fn dictionary_workspace_bytes(&self, query: &TextQuery) -> usize {
        let mut indexes = HashSet::new();
        let mut query_term_bytes = 0usize;
        for term in query.base_terms() {
            query_term_bytes = query_term_bytes
                .saturating_add(term.len())
                .saturating_add(SCORED_TERM_WORKSPACE_OVERHEAD_BYTES);
            if let Some(index) = dictionary_block_for_term(&self.footer.dictionary, term) {
                indexes.insert(index);
            }
        }
        for prefix in &query.prefixes {
            if let Some(index) =
                first_dictionary_block_whose_last_is_at_least(&self.footer.dictionary, prefix)
            {
                indexes.insert(index);
                if index + 1 < self.footer.dictionary.len() {
                    indexes.insert(index + 1);
                }
            }
        }
        indexes.into_iter().fold(
            ZSTD_DECODER_WORKSPACE_BYTES.saturating_add(query_term_bytes),
            |total, index| {
                let Some(block) = self.footer.dictionary.get(index) else {
                    return usize::MAX;
                };
                total
                    .saturating_add(block.wire.len as usize)
                    .saturating_add(
                        (block.wire.raw_len as usize)
                            .saturating_mul(DICTIONARY_DECODE_WORKSPACE_MULTIPLIER),
                    )
            },
        )
    }

    async fn lookup_term(
        &self,
        term: &str,
        cache: &mut HashMap<usize, Vec<TermEntry>>,
    ) -> Result<Option<TermEntry>, Error> {
        let Some(index) = dictionary_block_for_term(&self.footer.dictionary, term) else {
            return Ok(None);
        };
        let entries = self.dictionary_block(index, cache).await?;
        Ok(entries
            .binary_search_by(|entry| entry.term.as_str().cmp(term))
            .ok()
            .map(|position| entries[position].clone()))
    }

    async fn expand_prefix(
        &self,
        prefix: &str,
        cache: &mut HashMap<usize, Vec<TermEntry>>,
    ) -> Result<Vec<String>, Error> {
        let Some(mut block_index) =
            first_dictionary_block_whose_last_is_at_least(&self.footer.dictionary, prefix)
        else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        let mut blocks_read = 0usize;
        while block_index < self.footer.dictionary.len() && out.len() < PREFIX_EXPANSION_LIMIT {
            // Canonical V3 writers put 128 entries in every non-final
            // dictionary block. A 64-term prefix can therefore cross at most
            // one boundary. Reject a non-canonical/corrupt sparse layout
            // instead of decoding an attacker-controlled number of blocks
            // beyond the workspace reservation.
            if blocks_read == 2 {
                return Err(Error::invariant(
                    "text v3 prefix expansion crosses more than two dictionary blocks",
                ));
            }
            let entries = self.dictionary_block(block_index, cache).await?;
            blocks_read += 1;
            for entry in entries {
                if entry.term.as_str() < prefix {
                    continue;
                }
                if entry.term.starts_with(prefix) {
                    out.push(entry.term.clone());
                    if out.len() == PREFIX_EXPANSION_LIMIT {
                        break;
                    }
                } else {
                    return Ok(out);
                }
            }
            block_index += 1;
        }
        Ok(out)
    }

    async fn dictionary_block<'a>(
        &self,
        index: usize,
        cache: &'a mut HashMap<usize, Vec<TermEntry>>,
    ) -> Result<&'a [TermEntry], Error> {
        if !cache.contains_key(&index) {
            let block = self.footer.dictionary.get(index).ok_or_else(|| {
                Error::invariant(format!("text v3 dictionary block {index} is absent"))
            })?;
            let bytes = self.read_block(&block.wire, "dictionary").await?;
            let entries: Vec<TermEntry> = deserialize_bounded(&bytes, "dictionary block")?;
            validate_dictionary_entries(&entries, block, self.footer.n_docs, self.footer_offset)?;
            cache.insert(index, entries);
        }
        Ok(cache
            .get(&index)
            .expect("dictionary cache entry inserted above"))
    }

    async fn load_next_posting_blocks(
        &self,
        cursors: &mut [PostingCursor],
        cursor_indexes: &[usize],
        metrics: &mut TextV3QueryMetrics,
    ) -> Result<(), Error> {
        if cursor_indexes.is_empty() {
            return Ok(());
        }

        let mut jobs = Vec::<(usize, usize, PostingBlockRef)>::with_capacity(cursor_indexes.len());
        for &cursor_index in cursor_indexes {
            let cursor = cursors.get(cursor_index).ok_or_else(|| {
                Error::invariant("text v3 posting cursor index is outside the query plan")
            })?;
            let block_index = cursor.next_block;
            let block = cursor.blocks.get(block_index).ok_or_else(|| {
                Error::invariant(format!(
                    "text v3 cursor for {:?} has no next posting block",
                    cursor.term
                ))
            })?;
            let range = block.wire.range()?;
            if range.end > self.footer_offset {
                return Err(Error::invariant(
                    "text v3 posting block overlaps the footer",
                ));
            }
            jobs.push((cursor_index, block_index, block.clone()));
        }
        let ranges = jobs
            .iter()
            .map(|(_, _, block)| block.wire.range())
            .collect::<Result<Vec<_>, _>>()?;
        let blocks = self.source.read_ranges(&ranges).await?;
        if blocks.len() != jobs.len() {
            return Err(Error::invariant(
                "text v3 range source returned the wrong posting block count",
            ));
        }
        for ((cursor_index, block_index, meta), compressed) in jobs.into_iter().zip(blocks) {
            let raw = decode_block_bytes(&compressed, &meta.wire, "posting")?;
            let decoded = decode_posting_block(&raw, self.footer.n_docs)?;
            cursors[cursor_index].install_block(block_index, &meta, decoded)?;
            metrics.posting_blocks_read = metrics.posting_blocks_read.saturating_add(1);
        }
        let live_blocks = cursors
            .iter()
            .filter(|cursor| !cursor.current.is_empty())
            .count();
        let live_postings = cursors.iter().fold(0usize, |total, cursor| {
            total.saturating_add(cursor.current.len())
        });
        metrics.max_live_posting_blocks = metrics.max_live_posting_blocks.max(live_blocks);
        metrics.max_live_postings = metrics.max_live_postings.max(live_postings);
        Ok(())
    }

    async fn read_block(&self, block: &BlockRef, kind: &str) -> Result<Bytes, Error> {
        let range = block.range()?;
        if range.end > self.footer_offset {
            return Err(Error::invariant(format!(
                "text v3 {kind} block overlaps the footer"
            )));
        }
        let bytes = self.source.read_range(range).await?;
        decode_block_bytes(&bytes, block, kind).map(Bytes::from)
    }

    async fn read_doc_ids(&self, ordinals: &[u32]) -> Result<HashMap<u32, [u8; 16]>, Error> {
        if ordinals.is_empty() {
            return Ok(HashMap::new());
        }
        let mut sorted = ordinals.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted
            .last()
            .is_some_and(|last| *last >= self.footer.n_docs)
        {
            return Err(Error::invariant(
                "text v3 result ordinal exceeds the document table",
            ));
        }

        // Fetch adjacent winners as one span. A gap of at most four records is
        // cheaper than another remote request and remains bounded by O(k).
        let mut spans = Vec::<(u32, u32)>::new();
        let mut start = sorted[0];
        let mut end = start;
        for &doc in &sorted[1..] {
            if doc <= end.saturating_add(5) {
                end = doc;
            } else {
                spans.push((start, end));
                start = doc;
                end = doc;
            }
        }
        spans.push((start, end));

        let wanted = sorted.into_iter().collect::<HashSet<_>>();
        let mut out = HashMap::with_capacity(wanted.len());
        for batch in spans.chunks(DOC_ID_RANGE_BATCH_SIZE) {
            let ranges = batch
                .iter()
                .map(|(start, end)| {
                    let begin_delta =
                        u64::from(*start)
                            .checked_mul(DOC_RECORD_LEN)
                            .ok_or_else(|| {
                                Error::invariant("text v3 document span start overflows u64")
                            })?;
                    let finish_ordinal = u64::from(*end).checked_add(1).ok_or_else(|| {
                        Error::invariant("text v3 document span end overflows u64")
                    })?;
                    let finish_delta =
                        finish_ordinal.checked_mul(DOC_RECORD_LEN).ok_or_else(|| {
                            Error::invariant("text v3 document span finish overflows u64")
                        })?;
                    let begin = self
                        .footer
                        .doc_table_offset
                        .checked_add(begin_delta)
                        .ok_or_else(|| {
                            Error::invariant("text v3 document span offset overflows u64")
                        })?;
                    let finish = self
                        .footer
                        .doc_table_offset
                        .checked_add(finish_delta)
                        .ok_or_else(|| {
                            Error::invariant("text v3 document span offset overflows u64")
                        })?;
                    Ok(begin..finish)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let chunks = self.source.read_ranges(&ranges).await?;
            if chunks.len() != batch.len() {
                return Err(Error::invariant(
                    "text v3 range source returned the wrong document span count",
                ));
            }
            for ((span_start, span_end), chunk) in batch.iter().copied().zip(chunks) {
                let records_u64 = u64::from(span_end)
                    .checked_sub(u64::from(span_start))
                    .and_then(|delta| delta.checked_add(1))
                    .ok_or_else(|| Error::invariant("text v3 document span is reversed"))?;
                let records = usize::try_from(records_u64)
                    .map_err(|_| Error::invariant("text v3 span count does not fit usize"))?;
                let expected_bytes =
                    records
                        .checked_mul(DOC_RECORD_LEN as usize)
                        .ok_or_else(|| {
                            Error::invariant("text v3 document span bytes overflow usize")
                        })?;
                require_exact_len(&chunk, expected_bytes, "document span")?;
                for local in 0..records {
                    let local_u32 = u32::try_from(local).map_err(|_| {
                        Error::invariant("text v3 local document ordinal does not fit u32")
                    })?;
                    let doc = span_start.checked_add(local_u32).ok_or_else(|| {
                        Error::invariant("text v3 document ordinal overflows u32")
                    })?;
                    if !wanted.contains(&doc) {
                        continue;
                    }
                    let at = local.checked_mul(DOC_RECORD_LEN as usize).ok_or_else(|| {
                        Error::invariant("text v3 document record offset overflows usize")
                    })?;
                    let id_end = at.checked_add(16).ok_or_else(|| {
                        Error::invariant("text v3 document id end overflows usize")
                    })?;
                    let id_bytes = chunk.get(at..id_end).ok_or_else(|| {
                        Error::invariant("text v3 document id lies outside its span")
                    })?;
                    let mut id = [0u8; 16];
                    id.copy_from_slice(id_bytes);
                    out.insert(doc, id);
                }
            }
        }
        Ok(out)
    }
}

/// Build a v3 object. Input is sorted by NodeId so document ordinals are also
/// the deterministic score tie-break and the fixed document table can support
/// future range-based membership probes.
pub(super) fn build(
    mut members: Vec<([u8; 16], String)>,
) -> Result<Option<(Bytes, TextIndexBuildStats)>, Error> {
    if members.is_empty() {
        return Ok(None);
    }
    if members.len() > u32::MAX as usize {
        return Err(Error::invariant("text v3 document count exceeds u32"));
    }
    members.sort_unstable_by_key(|(id, _)| *id);
    if members.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(Error::invariant(
            "text v3 cannot index the same NodeId twice",
        ));
    }

    let min_node_id = members.first().expect("non-empty").0;
    let max_node_id = members.last().expect("non-empty").0;
    let mut docs = Vec::<([u8; 16], u32)>::with_capacity(members.len());
    let mut postings = BTreeMap::<String, Vec<Posting>>::new();
    let mut total_len = 0u64;
    for (doc, (id, text)) in members.into_iter().enumerate() {
        let doc = doc as u32;
        let tokens = tokenize(&text);
        if tokens.len() > u32::MAX as usize {
            return Err(Error::invariant("text v3 document token count exceeds u32"));
        }
        let doc_len = tokens.len() as u32;
        total_len = total_len
            .checked_add(doc_len as u64)
            .ok_or_else(|| Error::invariant("text v3 total token count overflows u64"))?;
        docs.push((id, doc_len));
        let mut positions = HashMap::<String, Vec<u32>>::new();
        for (position, term) in tokens.into_iter().enumerate() {
            positions.entry(term).or_default().push(position as u32);
        }
        for (term, positions) in positions {
            postings.entry(term).or_default().push(Posting {
                doc,
                tf: positions.len() as u32,
                doc_len,
                positions,
            });
        }
    }

    let mut file = MAGIC_V3.to_vec();
    let doc_table_offset = file.len() as u64;
    for (id, len) in &docs {
        file.extend_from_slice(id);
        file.extend_from_slice(&len.to_le_bytes());
    }
    let doc_table_len = file.len() as u64 - doc_table_offset;

    let mut terms = Vec::<TermEntry>::with_capacity(postings.len());
    for (term, list) in postings {
        let mut blocks = Vec::new();
        for chunk in list.chunks(POSTINGS_PER_BLOCK) {
            let raw = encode_posting_block(chunk)?;
            let wire = append_compressed_block(&mut file, &raw, "posting")?;
            let first = chunk.first().expect("non-empty posting block");
            let last = chunk.last().expect("non-empty posting block");
            blocks.push(PostingBlockRef {
                wire,
                first_doc: first.doc,
                last_doc: last.doc,
                max_tf: chunk.iter().map(|posting| posting.tf).max().unwrap_or(0),
                min_doc_len: chunk
                    .iter()
                    .map(|posting| posting.doc_len)
                    .min()
                    .unwrap_or(0),
            });
        }
        terms.push(TermEntry {
            term,
            doc_freq: list.len() as u32,
            blocks,
        });
    }

    let term_count = terms.len() as u64;
    let mut dictionary = Vec::new();
    for chunk in terms.chunks(TERMS_PER_DICTIONARY_BLOCK) {
        let raw = bincode::serialize(chunk)
            .map_err(|e| Error::invariant(format!("text v3 dictionary encode failed: {e}")))?;
        let wire = append_compressed_block(&mut file, &raw, "dictionary")?;
        dictionary.push(DictionaryBlockRef {
            first_term: chunk
                .first()
                .expect("non-empty dictionary block")
                .term
                .clone(),
            last_term: chunk
                .last()
                .expect("non-empty dictionary block")
                .term
                .clone(),
            wire,
        });
    }

    let footer = Footer {
        format_version: FORMAT_VERSION,
        n_docs: docs.len() as u32,
        total_len,
        doc_table_offset,
        doc_table_len,
        term_count,
        min_node_id,
        max_node_id,
        dictionary,
    };
    let footer_bytes = bincode::serialize(&footer)
        .map_err(|e| Error::invariant(format!("text v3 footer encode failed: {e}")))?;
    if footer_bytes.len() as u64 > MAX_FOOTER_BYTES {
        return Err(Error::invariant("text v3 footer exceeds the format limit"));
    }
    file.extend_from_slice(&footer_bytes);
    file.extend_from_slice(&(footer_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(TRAILER_MAGIC);
    file.extend_from_slice(&crc32fast::hash(&footer_bytes).to_le_bytes());

    let stats = TextIndexBuildStats {
        doc_count: docs.len() as u64,
        term_count,
        total_len,
        min_node_id,
        max_node_id,
    };
    Ok(Some((Bytes::from(file), stats)))
}

/// Decode a complete v3 object into the legacy in-memory representation.
///
/// This keeps the current monolithic engine read path backward-compatible
/// while callers migrate to [`TextIndexV3Reader`]. It is deliberately not used
/// by the range reader.
pub(super) fn decode_whole(bytes: &[u8]) -> Result<TextIndexBody, Error> {
    let (footer, footer_offset) = decode_footer_from_whole(bytes)?;
    let mut doc_ids = Vec::with_capacity(footer.n_docs as usize);
    let mut doc_lens = Vec::with_capacity(footer.n_docs as usize);
    let docs_start = usize::try_from(footer.doc_table_offset)
        .map_err(|_| Error::invariant("text v3 document offset does not fit usize"))?;
    for doc in 0..footer.n_docs as usize {
        let at = docs_start + doc * DOC_RECORD_LEN as usize;
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes[at..at + 16]);
        let len = u32::from_le_bytes(
            bytes[at + 16..at + 20]
                .try_into()
                .expect("fixed document record"),
        );
        doc_ids.push(id);
        doc_lens.push(len);
    }
    if doc_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::invariant(
            "text v3 document table is not strictly sorted by NodeId",
        ));
    }

    let mut postings = BTreeMap::<String, Vec<(u32, u32, Vec<u32>)>>::new();
    let mut previous_term: Option<String> = None;
    let mut seen_terms = 0u64;
    for directory_entry in &footer.dictionary {
        let raw =
            decode_block_from_whole(bytes, &directory_entry.wire, footer_offset, "dictionary")?;
        let entries: Vec<TermEntry> = deserialize_bounded(&raw, "dictionary block")?;
        validate_dictionary_entries(&entries, directory_entry, footer.n_docs, footer_offset)?;
        for entry in entries {
            if previous_term
                .as_ref()
                .is_some_and(|term| term >= &entry.term)
            {
                return Err(Error::invariant(
                    "text v3 dictionary terms are not strictly sorted",
                ));
            }
            previous_term = Some(entry.term.clone());
            seen_terms += 1;
            let mut decoded = Vec::with_capacity(entry.doc_freq as usize);
            for block in &entry.blocks {
                let raw = decode_block_from_whole(bytes, &block.wire, footer_offset, "posting")?;
                let values = decode_posting_block(&raw, footer.n_docs)?;
                validate_posting_block(&values, block)?;
                for posting in values {
                    if doc_lens[posting.doc as usize] != posting.doc_len {
                        return Err(Error::invariant(format!(
                            "text v3 posting document length mismatch for ordinal {}",
                            posting.doc
                        )));
                    }
                    decoded.push((posting.doc, posting.tf, posting.positions));
                }
            }
            if decoded.len() != entry.doc_freq as usize
                || decoded.windows(2).any(|pair| pair[0].0 >= pair[1].0)
            {
                return Err(Error::invariant(format!(
                    "text v3 term {:?} has invalid document frequency/order",
                    entry.term
                )));
            }
            postings.insert(entry.term, decoded);
        }
    }
    if seen_terms != footer.term_count {
        return Err(Error::invariant(format!(
            "text v3 footer declares {} terms, decoded {seen_terms}",
            footer.term_count
        )));
    }
    Ok(TextIndexBody {
        n_docs: footer.n_docs,
        total_len: footer.total_len,
        doc_ids,
        doc_lens,
        postings,
    })
}

fn append_compressed_block(file: &mut Vec<u8>, raw: &[u8], kind: &str) -> Result<BlockRef, Error> {
    if raw.len() as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "text v3 {kind} block exceeds the raw size limit"
        )));
    }
    let compressed = zstd::stream::encode_all(raw, ZSTD_LEVEL)
        .map_err(|e| Error::invariant(format!("text v3 {kind} compression failed: {e}")))?;
    if compressed.len() as u64 > MAX_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "text v3 {kind} block exceeds the wire size limit"
        )));
    }
    let offset = file.len() as u64;
    file.extend_from_slice(&compressed);
    Ok(BlockRef {
        offset,
        len: compressed.len() as u32,
        raw_len: raw.len() as u32,
        crc32: crc32fast::hash(&compressed),
    })
}

fn decode_block_from_whole(
    bytes: &[u8],
    block: &BlockRef,
    footer_offset: u64,
    kind: &str,
) -> Result<Vec<u8>, Error> {
    validate_block_ref(block, footer_offset, kind)?;
    let range = block.range()?;
    let start = usize::try_from(range.start)
        .map_err(|_| Error::invariant("text v3 block start does not fit usize"))?;
    let end = usize::try_from(range.end)
        .map_err(|_| Error::invariant("text v3 block end does not fit usize"))?;
    decode_block_bytes(&bytes[start..end], block, kind)
}

fn decode_block_bytes(bytes: &[u8], block: &BlockRef, kind: &str) -> Result<Vec<u8>, Error> {
    require_exact_len(bytes, block.len as usize, kind)?;
    if crc32fast::hash(bytes) != block.crc32 {
        return Err(Error::invariant(format!(
            "text v3 {kind} block checksum mismatch"
        )));
    }
    if block.raw_len as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "text v3 {kind} raw block length exceeds the format limit"
        )));
    }
    let decoder = zstd::stream::read::Decoder::new(bytes)
        .map_err(|e| Error::invariant(format!("text v3 {kind} decompression failed: {e}")))?;
    // Never let a corrupt frame expand beyond its checksummed declared size
    // before checking it. `decode_all` performed this check only after an
    // attacker-controlled allocation had already happened.
    let mut limited = decoder.take(block.raw_len as u64 + 1);
    let mut raw = Vec::with_capacity(block.raw_len as usize);
    limited
        .read_to_end(&mut raw)
        .map_err(|e| Error::invariant(format!("text v3 {kind} decompression failed: {e}")))?;
    require_exact_len(&raw, block.raw_len as usize, "decompressed block")?;
    Ok(raw)
}

fn encode_posting_block(postings: &[Posting]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    put_varint(postings.len() as u64, &mut out);
    let mut previous_doc = 0u32;
    for (index, posting) in postings.iter().enumerate() {
        if index > 0 && posting.doc <= previous_doc {
            return Err(Error::invariant(
                "text v3 builder received unsorted postings",
            ));
        }
        let delta = if index == 0 {
            posting.doc
        } else {
            posting.doc - previous_doc
        };
        put_varint(delta as u64, &mut out);
        put_varint(posting.tf as u64, &mut out);
        put_varint(posting.doc_len as u64, &mut out);
        put_varint(posting.positions.len() as u64, &mut out);
        let mut previous_position = 0u32;
        for (position_index, position) in posting.positions.iter().copied().enumerate() {
            if position >= posting.doc_len || (position_index > 0 && position <= previous_position)
            {
                return Err(Error::invariant(
                    "text v3 builder received invalid token positions",
                ));
            }
            let delta = if position_index == 0 {
                position
            } else {
                position - previous_position
            };
            put_varint(delta as u64, &mut out);
            previous_position = position;
        }
        previous_doc = posting.doc;
    }
    Ok(out)
}

fn decode_posting_block(bytes: &[u8], n_docs: u32) -> Result<Vec<Posting>, Error> {
    let mut cursor = 0usize;
    let count = usize::try_from(take_varint(bytes, &mut cursor)?)
        .map_err(|_| Error::invariant("text v3 posting count does not fit usize"))?;
    if count > POSTINGS_PER_BLOCK {
        return Err(Error::invariant(format!(
            "text v3 posting block has {count} records, limit is {POSTINGS_PER_BLOCK}"
        )));
    }
    let mut out = Vec::with_capacity(count);
    let mut previous_doc = 0u32;
    for index in 0..count {
        let delta = take_u32_varint(bytes, &mut cursor, "document delta")?;
        let doc = if index == 0 {
            delta
        } else {
            previous_doc
                .checked_add(delta)
                .ok_or_else(|| Error::invariant("text v3 document delta overflows"))?
        };
        if doc >= n_docs || (index > 0 && doc <= previous_doc) {
            return Err(Error::invariant(
                "text v3 posting document ordinal is invalid",
            ));
        }
        let tf = take_u32_varint(bytes, &mut cursor, "term frequency")?;
        let doc_len = take_u32_varint(bytes, &mut cursor, "document length")?;
        let position_count = usize::try_from(take_varint(bytes, &mut cursor)?)
            .map_err(|_| Error::invariant("text v3 position count does not fit usize"))?;
        if tf == 0 || position_count != tf as usize || tf > doc_len {
            return Err(Error::invariant(
                "text v3 term frequency/position count is invalid",
            ));
        }
        // Every position needs at least one encoded varint byte. Prove the
        // payload can contain the declaration before using it as a capacity;
        // otherwise a tiny corrupt block could request an enormous vector.
        if position_count > bytes.len().saturating_sub(cursor) {
            return Err(Error::invariant(
                "text v3 position count exceeds the remaining posting payload",
            ));
        }
        let mut positions = Vec::new();
        positions
            .try_reserve_exact(position_count)
            .map_err(|error| {
                Error::invariant(format!(
                    "text v3 position allocation for {position_count} entries failed: {error}"
                ))
            })?;
        let mut previous_position = 0u32;
        for position_index in 0..position_count {
            let delta = take_u32_varint(bytes, &mut cursor, "position delta")?;
            let position = if position_index == 0 {
                delta
            } else {
                previous_position
                    .checked_add(delta)
                    .ok_or_else(|| Error::invariant("text v3 position delta overflows"))?
            };
            if position >= doc_len || (position_index > 0 && position <= previous_position) {
                return Err(Error::invariant("text v3 posting positions are invalid"));
            }
            positions.push(position);
            previous_position = position;
        }
        out.push(Posting {
            doc,
            tf,
            doc_len,
            positions,
        });
        previous_doc = doc;
    }
    if cursor != bytes.len() {
        return Err(Error::invariant("text v3 posting block has trailing bytes"));
    }
    Ok(out)
}

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn take_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64, Error> {
    let mut value = 0u64;
    for shift in (0..=63).step_by(7) {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| Error::invariant("text v3 varint is truncated"))?;
        *cursor += 1;
        if shift == 63 && byte > 1 {
            return Err(Error::invariant("text v3 varint overflows u64"));
        }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(Error::invariant("text v3 varint is too long"))
}

fn take_u32_varint(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u32, Error> {
    u32::try_from(take_varint(bytes, cursor)?)
        .map_err(|_| Error::invariant(format!("text v3 {what} exceeds u32")))
}

fn decode_trailer(bytes: &[u8]) -> Result<(u64, u32), Error> {
    require_exact_len(bytes, TRAILER_LEN, "trailer")?;
    let footer_len = u64::from_le_bytes(bytes[0..8].try_into().expect("fixed trailer"));
    if &bytes[8..16] != TRAILER_MAGIC {
        return Err(Error::invariant("text v3 trailer magic mismatch"));
    }
    let footer_crc = u32::from_le_bytes(bytes[16..20].try_into().expect("fixed trailer"));
    Ok((footer_len, footer_crc))
}

fn decode_footer_from_whole(bytes: &[u8]) -> Result<(Footer, u64), Error> {
    if bytes.len() < MAGIC_V3.len() + TRAILER_LEN {
        return Err(Error::invariant("text v3 body too short"));
    }
    if &bytes[..MAGIC_V3.len()] != MAGIC_V3 {
        return Err(Error::invariant("text v3 magic mismatch"));
    }
    let trailer_start = bytes.len() - TRAILER_LEN;
    let (footer_len, footer_crc) = decode_trailer(&bytes[trailer_start..])?;
    if footer_len == 0 || footer_len > MAX_FOOTER_BYTES {
        return Err(Error::invariant("text v3 footer length is invalid"));
    }
    let footer_len_usize = usize::try_from(footer_len)
        .map_err(|_| Error::invariant("text v3 footer does not fit usize"))?;
    let footer_start = trailer_start
        .checked_sub(footer_len_usize)
        .ok_or_else(|| Error::invariant("text v3 footer starts before the object"))?;
    if footer_start < MAGIC_V3.len() {
        return Err(Error::invariant("text v3 footer overlaps the header"));
    }
    let footer_bytes = &bytes[footer_start..trailer_start];
    if crc32fast::hash(footer_bytes) != footer_crc {
        return Err(Error::invariant("text v3 footer checksum mismatch"));
    }
    let footer: Footer = deserialize_bounded(footer_bytes, "footer")?;
    validate_footer(&footer, footer_start as u64)?;
    Ok((footer, footer_start as u64))
}

fn validate_footer(footer: &Footer, footer_offset: u64) -> Result<(), Error> {
    if footer.format_version != FORMAT_VERSION {
        return Err(Error::invariant(format!(
            "text v3 footer version {} is unsupported",
            footer.format_version
        )));
    }
    if footer.n_docs == 0 {
        return Err(Error::invariant(
            "text v3 non-empty object declares zero documents",
        ));
    }
    if footer.min_node_id > footer.max_node_id {
        return Err(Error::invariant("text v3 NodeId bounds are reversed"));
    }
    if footer.doc_table_offset != MAGIC_V3.len() as u64 {
        return Err(Error::invariant(
            "text v3 document table must immediately follow the magic",
        ));
    }
    let expected_doc_len = footer.n_docs as u64 * DOC_RECORD_LEN;
    if footer.doc_table_len != expected_doc_len {
        return Err(Error::invariant(format!(
            "text v3 document table length {}, expected {expected_doc_len}",
            footer.doc_table_len
        )));
    }
    let docs_end = footer
        .doc_table_offset
        .checked_add(footer.doc_table_len)
        .ok_or_else(|| Error::invariant("text v3 document table range overflows"))?;
    if docs_end > footer_offset {
        return Err(Error::invariant(
            "text v3 document table overlaps the footer",
        ));
    }
    if footer.term_count == 0 && !footer.dictionary.is_empty() {
        return Err(Error::invariant(
            "text v3 empty vocabulary has dictionary blocks",
        ));
    }
    if footer.term_count > 0 && footer.dictionary.is_empty() {
        return Err(Error::invariant(
            "text v3 non-empty vocabulary has no dictionary",
        ));
    }
    let mut previous_last: Option<&str> = None;
    let mut previous_end = docs_end;
    for block in &footer.dictionary {
        if block.first_term.is_empty()
            || block.first_term > block.last_term
            || previous_last.is_some_and(|last| last >= block.first_term.as_str())
        {
            return Err(Error::invariant(
                "text v3 dictionary directory is not strictly sorted",
            ));
        }
        validate_block_ref(&block.wire, footer_offset, "dictionary")?;
        if block.wire.offset < previous_end {
            return Err(Error::invariant(
                "text v3 dictionary blocks overlap or are out of order",
            ));
        }
        previous_end = block.wire.range()?.end;
        previous_last = Some(&block.last_term);
    }
    Ok(())
}

fn validate_block_ref(block: &BlockRef, footer_offset: u64, kind: &str) -> Result<(), Error> {
    if block.len == 0
        || block.len as u64 > MAX_BLOCK_BYTES
        || block.raw_len == 0
        || block.raw_len as u64 > MAX_RAW_BLOCK_BYTES
    {
        return Err(Error::invariant(format!(
            "text v3 {kind} block lengths are invalid"
        )));
    }
    let range = block.range()?;
    if range.start < MAGIC_V3.len() as u64 || range.end > footer_offset {
        return Err(Error::invariant(format!(
            "text v3 {kind} block is outside the data section"
        )));
    }
    Ok(())
}

fn validate_dictionary_entries(
    entries: &[TermEntry],
    directory: &DictionaryBlockRef,
    n_docs: u32,
    footer_offset: u64,
) -> Result<(), Error> {
    if entries.is_empty() || entries.len() > TERMS_PER_DICTIONARY_BLOCK {
        return Err(Error::invariant(
            "text v3 dictionary block entry count is invalid",
        ));
    }
    if entries.first().map(|entry| &entry.term) != Some(&directory.first_term)
        || entries.last().map(|entry| &entry.term) != Some(&directory.last_term)
        || entries.windows(2).any(|pair| pair[0].term >= pair[1].term)
    {
        return Err(Error::invariant(
            "text v3 dictionary block bounds/order mismatch",
        ));
    }
    for entry in entries {
        if entry.term.is_empty() || entry.doc_freq == 0 || entry.doc_freq > n_docs {
            return Err(Error::invariant(
                "text v3 dictionary term metadata is invalid",
            ));
        }
        let mut count = 0u64;
        let mut previous_last = None;
        for block in &entry.blocks {
            validate_block_ref(&block.wire, footer_offset, "posting")?;
            if block.first_doc > block.last_doc
                || block.last_doc >= n_docs
                || block.max_tf == 0
                || block.min_doc_len == 0
                || previous_last.is_some_and(|last| last >= block.first_doc)
            {
                return Err(Error::invariant(
                    "text v3 posting block metadata is invalid",
                ));
            }
            previous_last = Some(block.last_doc);
            count += (block.last_doc as u64)
                .saturating_sub(block.first_doc as u64)
                .saturating_add(1)
                .min(POSTINGS_PER_BLOCK as u64);
        }
        if entry.blocks.is_empty()
            || entry.doc_freq as usize > entry.blocks.len() * POSTINGS_PER_BLOCK
        {
            return Err(Error::invariant(
                "text v3 term block count cannot contain its document frequency",
            ));
        }
        // `count` is only a loose range sanity bound because document ordinals
        // inside a block may have gaps.
        if count < entry.blocks.len() as u64 {
            return Err(Error::invariant(
                "text v3 posting block document ranges are invalid",
            ));
        }
    }
    Ok(())
}

fn validate_posting_block(postings: &[Posting], meta: &PostingBlockRef) -> Result<(), Error> {
    let Some(first) = postings.first() else {
        return Err(Error::invariant("text v3 posting block is empty"));
    };
    let last = postings.last().expect("non-empty");
    if first.doc != meta.first_doc
        || last.doc != meta.last_doc
        || postings.iter().map(|posting| posting.tf).max() != Some(meta.max_tf)
        || postings.iter().map(|posting| posting.doc_len).min() != Some(meta.min_doc_len)
    {
        return Err(Error::invariant(
            "text v3 posting block metadata does not match its payload",
        ));
    }
    Ok(())
}

fn dictionary_block_for_term(directory: &[DictionaryBlockRef], term: &str) -> Option<usize> {
    let index = directory.partition_point(|block| block.last_term.as_str() < term);
    directory
        .get(index)
        .filter(|block| block.first_term.as_str() <= term)
        .map(|_| index)
}

fn first_dictionary_block_whose_last_is_at_least(
    directory: &[DictionaryBlockRef],
    term: &str,
) -> Option<usize> {
    let index = directory.partition_point(|block| block.last_term.as_str() < term);
    (index < directory.len()).then_some(index)
}

fn posting_query_workspace_bytes(
    entries: &BTreeMap<String, TermEntry>,
    n_docs: usize,
    k: Option<usize>,
    result_limit_bytes: usize,
) -> usize {
    let mut total = entries
        .len()
        .saturating_mul(std::mem::size_of::<PostingCursor>())
        .saturating_add(ZSTD_DECODER_WORKSPACE_BYTES);
    for (term, entry) in entries {
        let max_raw = entry
            .blocks
            .iter()
            .map(|block| block.wire.raw_len as usize)
            .max()
            .unwrap_or(0);
        let max_wire = entry
            .blocks
            .iter()
            .map(|block| block.wire.len as usize)
            .max()
            .unwrap_or(0);
        total = total
            // Cursor term plus the `term_to_cursor` key clone/hash entry.
            .saturating_add(term.capacity().saturating_mul(2))
            .saturating_add(std::mem::size_of::<(String, usize)>())
            .saturating_add(
                entry
                    .blocks
                    .capacity()
                    .saturating_mul(std::mem::size_of::<PostingBlockRef>()),
            )
            .saturating_add(max_wire)
            .saturating_add(max_raw.saturating_mul(POSTING_DECODE_WORKSPACE_MULTIPLIER))
            .saturating_add(POSTINGS_PER_BLOCK.saturating_mul(std::mem::size_of::<Posting>()));
    }

    let maximum_result_bytes = n_docs.saturating_mul(MATERIALISED_RESULT_BYTES_PER_HIT);
    let result_bytes = match k {
        Some(k) => k
            .min(n_docs)
            .saturating_mul(MATERIALISED_RESULT_BYTES_PER_HIT),
        None => maximum_result_bytes.min(result_limit_bytes),
    };
    total.saturating_add(result_bytes)
}

fn phrase_matches_cursor_doc(
    phrase: &[String],
    doc: u32,
    term_to_cursor: &HashMap<String, usize>,
    cursors: &[PostingCursor],
) -> bool {
    let Some(first_term) = phrase.first() else {
        // Empty phrases are not emitted by the parser. Preserve the previous
        // ranged-reader behaviour if one is constructed manually.
        return false;
    };
    let Some(first) = term_to_cursor
        .get(first_term)
        .and_then(|index| cursors.get(*index))
        .and_then(PostingCursor::current)
        .filter(|posting| posting.doc == doc)
    else {
        return false;
    };

    'start_positions: for &start in &first.positions {
        for (offset, term) in phrase.iter().enumerate().skip(1) {
            let Some(target) = u32::try_from(offset)
                .ok()
                .and_then(|offset| start.checked_add(offset))
            else {
                continue 'start_positions;
            };
            let Some(posting) = term_to_cursor
                .get(term)
                .and_then(|index| cursors.get(*index))
                .and_then(PostingCursor::current)
                .filter(|posting| posting.doc == doc)
            else {
                continue 'start_positions;
            };
            if posting.positions.binary_search(&target).is_err() {
                continue 'start_positions;
            }
        }
        return true;
    }
    false
}

#[derive(Debug, Copy, Clone)]
struct RankedOrdinal {
    doc: u32,
    score: f64,
}

impl PartialEq for RankedOrdinal {
    fn eq(&self, other: &Self) -> bool {
        self.doc == other.doc && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for RankedOrdinal {}

impl PartialOrd for RankedOrdinal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedOrdinal {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            // Smaller NodeId/document ordinal wins deterministic ties.
            .then_with(|| other.doc.cmp(&self.doc))
    }
}

fn require_exact_len(bytes: &[u8], expected: usize, what: &str) -> Result<(), Error> {
    if bytes.len() != expected {
        return Err(Error::invariant(format!(
            "text v3 {what} range returned {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok(())
}

fn deserialize_bounded<T: DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T, Error> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(bytes.len() as u64)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| Error::invariant(format!("text v3 {what} decode failed: {error}")))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::sst::text::TextIndex;
    use crate::text::parse_query;

    #[derive(Debug)]
    struct MemoryRangeSource {
        body: Bytes,
    }

    #[async_trait]
    impl TextIndexRangeSource for MemoryRangeSource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error> {
            if range.start > range.end || range.end > self.body.len() as u64 {
                return Err(Error::invariant("test range is outside the text body"));
            }
            Ok(self.body.slice(range.start as usize..range.end as usize))
        }

        async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
            let mut out = Vec::with_capacity(ranges.len());
            for range in ranges {
                out.push(self.read_range(range.clone()).await?);
            }
            Ok(out)
        }
    }

    #[derive(Debug)]
    struct BatchTrackingRangeSource {
        body: Bytes,
        batch_sizes: Mutex<Vec<usize>>,
    }

    #[async_trait]
    impl TextIndexRangeSource for BatchTrackingRangeSource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error> {
            if range.start > range.end || range.end > self.body.len() as u64 {
                return Err(Error::invariant("test range is outside the text body"));
            }
            Ok(self.body.slice(range.start as usize..range.end as usize))
        }

        async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
            self.batch_sizes.lock().unwrap().push(ranges.len());
            let mut out = Vec::with_capacity(ranges.len());
            for range in ranges {
                out.push(self.read_range(range.clone()).await?);
            }
            Ok(out)
        }
    }

    fn id(value: u8) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[15] = value;
        id
    }

    fn wide_id(value: u32) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[12..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn fixture() -> Vec<u8> {
        build(vec![
            (id(1), "alpha beta beta".into()),
            (id(2), "beta gamma".into()),
        ])
        .unwrap()
        .unwrap()
        .0
        .to_vec()
    }

    fn rewrite_footer(body: &mut [u8], mutate: impl FnOnce(&mut Footer)) {
        let (mut footer, footer_offset) = decode_footer_from_whole(body).unwrap();
        mutate(&mut footer);
        let encoded = bincode::serialize(&footer).unwrap();
        let trailer_start = body.len() - TRAILER_LEN;
        assert_eq!(
            encoded.len(),
            trailer_start - footer_offset as usize,
            "test mutation must preserve the footer's encoded length"
        );
        body[footer_offset as usize..trailer_start].copy_from_slice(&encoded);
        body[trailer_start..trailer_start + 8]
            .copy_from_slice(&(encoded.len() as u64).to_le_bytes());
        body[trailer_start + 16..].copy_from_slice(&crc32fast::hash(&encoded).to_le_bytes());
    }

    #[test]
    fn rejects_oversized_footer_before_deserialization() {
        let mut body = fixture();
        let trailer = body.len() - TRAILER_LEN;
        body[trailer..trailer + 8].copy_from_slice(&(MAX_FOOTER_BYTES + 1).to_le_bytes());
        let error = decode_whole(&body).unwrap_err().to_string();
        assert!(error.contains("footer length"), "{error}");
    }

    #[test]
    fn rejects_footer_checksum_corruption() {
        let mut body = fixture();
        let last = body.len() - 1;
        body[last] ^= 0x80;
        let error = decode_whole(&body).unwrap_err().to_string();
        assert!(error.contains("footer checksum"), "{error}");
    }

    #[test]
    fn rejects_oversized_or_overlapping_dictionary_ranges() {
        let mut oversized = fixture();
        rewrite_footer(&mut oversized, |footer| {
            footer.dictionary[0].wire.len = (MAX_BLOCK_BYTES + 1) as u32;
        });
        let error = decode_whole(&oversized).unwrap_err().to_string();
        assert!(error.contains("block lengths"), "{error}");

        let mut overlapping = fixture();
        rewrite_footer(&mut overlapping, |footer| {
            footer.dictionary[0].wire.offset = footer.doc_table_offset;
        });
        let error = decode_whole(&overlapping).unwrap_err().to_string();
        assert!(
            error.contains("overlap") || error.contains("out of order"),
            "{error}"
        );
    }

    #[test]
    fn rejects_corrupt_dictionary_bounds_and_posting_crc() {
        let mut bad_bounds = fixture();
        rewrite_footer(&mut bad_bounds, |footer| {
            // Same byte length keeps the test mutation localized to the
            // checksummed footer while making its sparse term range invalid.
            footer.dictionary[0].first_term = "zeta!".into();
        });
        let error = decode_whole(&bad_bounds).unwrap_err().to_string();
        assert!(error.contains("directory"), "{error}");

        let mut bad_posting = fixture();
        let (footer, footer_offset) = decode_footer_from_whole(&bad_posting).unwrap();
        let directory = &footer.dictionary[0];
        let raw =
            decode_block_from_whole(&bad_posting, &directory.wire, footer_offset, "dictionary")
                .unwrap();
        let entries: Vec<TermEntry> = deserialize_bounded(&raw, "dictionary block").unwrap();
        let offset = entries[0].blocks[0].wire.offset as usize;
        bad_posting[offset] ^= 0x01;
        let error = decode_whole(&bad_posting).unwrap_err().to_string();
        assert!(error.contains("posting block checksum"), "{error}");
    }

    #[test]
    fn bounded_decoder_stops_after_declared_raw_length() {
        let expanded = vec![b'x'; 32 * 1024];
        let compressed = zstd::stream::encode_all(expanded.as_slice(), 1).unwrap();
        let block = BlockRef {
            offset: 8,
            len: compressed.len() as u32,
            raw_len: 8,
            crc32: crc32fast::hash(&compressed),
        };
        let error = decode_block_bytes(&compressed, &block, "adversarial")
            .unwrap_err()
            .to_string();
        assert!(error.contains("returned 9 bytes, expected 8"), "{error}");
    }

    #[test]
    fn corrupt_position_count_is_rejected_before_capacity_reservation() {
        let mut raw = Vec::new();
        put_varint(1, &mut raw); // postings
        put_varint(0, &mut raw); // first document ordinal
        put_varint(u32::MAX as u64, &mut raw); // tf
        put_varint(u32::MAX as u64, &mut raw); // document length
        put_varint(u32::MAX as u64, &mut raw); // positions, but no payload

        let result = std::panic::catch_unwind(|| decode_posting_block(raw.as_slice(), 1)).unwrap();
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("exceeds the remaining posting payload"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn streaming_postings_are_bit_exact_and_one_block_per_term() {
        let docs = (0..1_300u32)
            .map(|ordinal| {
                let text = match ordinal % 4 {
                    0 => "graph database dataflow common common legal",
                    1 => "database graph dataset common legal",
                    2 => "graph database datum common legal",
                    _ => "unrelated common legal text",
                };
                (wide_id(ordinal), text.to_owned())
            })
            .collect::<Vec<_>>();
        let (body, _) = build(docs).unwrap().unwrap();
        let expected = TextIndex::decode(&body).unwrap();
        let source = Arc::new(MemoryRangeSource { body: body.clone() });
        let reader = TextIndexV3Reader::open(source, body.len() as u64)
            .await
            .unwrap();

        for (raw_query, k) in [
            ("\"graph database\" data* common", Some(17)),
            ("\"common common\" graph", Some(31)),
            ("legal data*", None),
        ] {
            let query = parse_query(raw_query);
            let expected_hits = expected.search_query(&query, k);
            let mut metrics = TextV3QueryMetrics::default();
            let actual = reader
                .search_query_inner(&query, k, 4 * 1024 * 1024, &mut metrics)
                .await
                .unwrap();
            assert_eq!(actual.len(), expected_hits.len(), "{raw_query:?}");
            for (actual, expected) in actual.iter().zip(&expected_hits) {
                assert_eq!(actual.0, expected.0, "{raw_query:?}");
                assert_eq!(
                    actual.1.to_bits(),
                    expected.1.to_bits(),
                    "score mismatch for {raw_query:?}: {} != {}",
                    actual.1,
                    expected.1
                );
            }

            let relevant_terms =
                query.base_terms().len() + query.prefixes.len() * PREFIX_EXPANSION_LIMIT;
            assert!(
                metrics.max_live_posting_blocks <= relevant_terms,
                "{metrics:?}"
            );
            assert!(
                metrics.max_live_postings <= relevant_terms.saturating_mul(POSTINGS_PER_BLOCK),
                "{metrics:?}"
            );
            assert!(
                metrics.posting_blocks_read > metrics.max_live_posting_blocks,
                "fixture must cross posting-block boundaries: {metrics:?}"
            );
            if let Some(k) = k {
                assert!(metrics.retained_results <= k, "{metrics:?}");
            }
        }
    }

    #[tokio::test]
    async fn streaming_prefix_preserves_lexicographic_expansion_limit() {
        let mut docs = (0..120u32)
            .map(|ordinal| (wide_id(ordinal), format!("aaa{ordinal:03}")))
            .collect::<Vec<_>>();
        docs.extend(
            (0..200u32)
                .map(|ordinal| (wide_id(1_000 + ordinal), format!("prefix{ordinal:03}")))
                .collect::<Vec<_>>(),
        );
        let (body, _) = build(docs).unwrap().unwrap();
        let expected = TextIndex::decode(&body).unwrap();
        let source = Arc::new(MemoryRangeSource { body: body.clone() });
        let reader = TextIndexV3Reader::open(source, body.len() as u64)
            .await
            .unwrap();
        let query = parse_query("prefix*");
        let expected_hits = expected.search_query(&query, None);
        assert_eq!(expected_hits.len(), PREFIX_EXPANSION_LIMIT);

        let mut metrics = TextV3QueryMetrics::default();
        let actual = reader
            .search_query_inner(&query, None, 1024 * 1024, &mut metrics)
            .await
            .unwrap();
        assert_eq!(actual.len(), expected_hits.len());
        for (actual, expected) in actual.iter().zip(expected_hits) {
            assert_eq!(actual.0, expected.0);
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
        assert_eq!(metrics.max_live_posting_blocks, PREFIX_EXPANSION_LIMIT);
        assert!(
            metrics.max_live_postings <= PREFIX_EXPANSION_LIMIT.saturating_mul(POSTINGS_PER_BLOCK),
            "{metrics:?}"
        );
    }

    #[tokio::test]
    async fn unbounded_result_limit_errors_instead_of_truncating() {
        let docs = (0..10u32)
            .map(|ordinal| (wide_id(ordinal), "common".to_owned()))
            .collect::<Vec<_>>();
        let (body, _) = build(docs).unwrap().unwrap();
        let source = Arc::new(MemoryRangeSource { body: body.clone() });
        let reader = TextIndexV3Reader::open(source, body.len() as u64)
            .await
            .unwrap();
        let query = parse_query("common");

        let mut metrics = TextV3QueryMetrics::default();
        let error = reader
            .search_query_inner(
                &query,
                None,
                2 * MATERIALISED_RESULT_BYTES_PER_HIT,
                &mut metrics,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            Error::SearchResultLimitExceeded {
                index_kind: "full-text",
                estimated_bytes,
                limit_bytes,
            } if estimated_bytes == 3 * MATERIALISED_RESULT_BYTES_PER_HIT
                && limit_bytes == 2 * MATERIALISED_RESULT_BYTES_PER_HIT
        ));

        // The unbounded-result guard is not an implicit cap on a caller that
        // explicitly requests top-k.
        let mut metrics = TextV3QueryMetrics::default();
        let hits = reader
            .search_query_inner(&query, Some(2), 0, &mut metrics)
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[tokio::test]
    async fn winning_document_ranges_are_checked_and_batched() {
        let docs = (0..400u32)
            .map(|ordinal| (wide_id(ordinal), format!("term{ordinal}")))
            .collect::<Vec<_>>();
        let (body, _) = build(docs).unwrap().unwrap();
        let source = Arc::new(BatchTrackingRangeSource {
            body: body.clone(),
            batch_sizes: Mutex::new(Vec::new()),
        });
        let reader = TextIndexV3Reader::open(source.clone(), body.len() as u64)
            .await
            .unwrap();
        source.batch_sizes.lock().unwrap().clear();

        // Gaps larger than the coalescing threshold force forty independent
        // spans. The reader must never retain all forty range bodies at once.
        let ordinals = (0..400u32).step_by(10).collect::<Vec<_>>();
        let ids = reader.read_doc_ids(&ordinals).await.unwrap();
        assert_eq!(ids.len(), ordinals.len());
        assert_eq!(
            source.batch_sizes.lock().unwrap().as_slice(),
            &[DOC_ID_RANGE_BATCH_SIZE, DOC_ID_RANGE_BATCH_SIZE, 8]
        );
    }
}
