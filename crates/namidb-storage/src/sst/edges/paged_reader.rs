//! Range-readable edge SST lookups.
//!
//! [`EdgeSstReader`](super::reader::EdgeSstReader) remains the eager scan and
//! compaction reader. `PagedEdgeReader` is the point/adjacency counterpart: it
//! pins one immutable object generation, reads the header and footer, keeps
//! only the sparse fence (or the bounded unfenced key array), and fetches the
//! selected key's offsets, partner block, LSNs, and tombstone bytes.
//!
//! v1.0 bodies lack cumulative edge ordinals and v1.1 bodies cannot
//! authenticate partial section reads. Both therefore use an exact, lazy
//! full-read fallback. v1.2 roots fixed-size page checksums in the footer and
//! is the first wire version served entirely through verified ranges.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};
use tokio::sync::OnceCell;
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};
use crate::range_cache::{ImmutableRangeCache, PinnedObjectRangeSource};
use crate::sst::edges::encoding::{
    binary_search_fixed_16, find_partner_in_block, read_offset, read_partner_block_expected,
    read_varint, validate_partner_block_read_bounds, OffsetWidth, TAG_DENSE, TAG_SPLIT,
};
use crate::sst::edges::fence_index::FenceIndex;
use crate::sst::edges::format::{
    EdgeFileFooter, EdgeFileHeader, EdgePageChecksumDirectory, EdgeSstBinding, SectionEntry,
    CODEC_NONE, CODEC_PROPERTY_PAGED_NONE, CODEC_PROPERTY_PAGED_ZSTD, FLAG_HAS_TOMBSTONES,
    FOOTER_TRAILER_LEN, HEADER_LEN, MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES, OVERFLOW_JSON_NAME,
    SECTION_EDGE_ORDINALS, SECTION_FENCE_INDEX, SECTION_KEY_IDS, SECTION_OFFSETS,
    SECTION_PAGE_CHECKSUMS, SECTION_PARTNERS, SECTION_PER_EDGE_LSN, SECTION_PER_EDGE_TOMBSTONES,
    SECTION_PROPERTY_STREAM, SECTION_SST_BINDING,
};
use crate::sst::edges::property_pages::{
    decode_property_page, PropertyPageIndex, PROPERTY_PAGES_HEADER_LEN,
};
use crate::sst::edges::reader::{EdgeLookup, EdgePointLookup, EdgeSstReader};
use crate::sst::edges::EdgeDirection;

/// Initial suffix request. Normal edge footers are a few hundred bytes; a
/// larger property schema transparently triggers one exact-footer request.
pub const EDGE_FOOTER_PREFETCH_BYTES: u64 = 4 * 1024;
/// Defensive bound before allocating an exact footer suffix. A legitimate
/// footer is O(property columns) and normally below a few KiB.
pub const MAX_EDGE_FOOTER_BYTES: u64 = 16 * 1024 * 1024;
/// Retained sparse fence ceiling. At the standard stride of 256 this covers
/// roughly 178 million keys; larger/custom indexes use bounded remote probes.
pub const MAX_CACHED_EDGE_FENCE_BYTES: u64 = 16 * 1024 * 1024;
/// Standard v1 writers emit a fence before `key_ids` grows beyond 1 MiB.
/// A custom/older writer may not; keep the reader's retained fallback bounded
/// and use conditional 16-byte probes for those anomalous bodies.
pub const MAX_UNFENCED_KEY_IDS_BYTES: u64 = 1024 * 1024;
/// Transient verified pages retained by one dense endpoint binary search.
const MAX_DENSE_PROBE_PAGES: usize = 4;

#[derive(Debug, Default)]
struct IoCounters {
    range_requests: AtomicU64,
    bytes_read: AtomicU64,
    eager_body_reads: AtomicU64,
}

/// Point-in-time logical I/O counters for one reader.
///
/// `range_requests` and `bytes_read` count ranges delivered to this reader,
/// whether they came from RAM, NVMe, or the authoritative store. Physical
/// cache hits/misses and remote fetches live in [`crate::range_cache::RangeCacheStats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedEdgeIoStats {
    pub range_requests: u64,
    pub bytes_read: u64,
    pub eager_body_reads: u64,
}

/// Immutable object generation used by every request issued by one reader.
///
/// The common range source owns the generation precondition, shared
/// RAM/NVMe cache, and remote semaphore. This wrapper adds edge page
/// verification plus reader-local logical I/O counters.
#[derive(Debug)]
struct PinnedEdgeObject {
    source: PinnedObjectRangeSource,
    counters: Arc<IoCounters>,
    page_checksums: OnceLock<Arc<EdgePageChecksumDirectory>>,
}

impl PinnedEdgeObject {
    fn new(source: PinnedObjectRangeSource) -> Self {
        Self {
            source,
            counters: Arc::new(IoCounters::default()),
            page_checksums: OnceLock::new(),
        }
    }

    fn path(&self) -> &Path {
        self.source.location()
    }

    fn size(&self) -> u64 {
        self.source.object_size()
    }

    fn stats(&self) -> PagedEdgeIoStats {
        PagedEdgeIoStats {
            range_requests: self.counters.range_requests.load(Ordering::Relaxed),
            bytes_read: self.counters.bytes_read.load(Ordering::Relaxed),
            eager_body_reads: self.counters.eager_body_reads.load(Ordering::Relaxed),
        }
    }

    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        if range.start > range.end || range.end > self.size() {
            return Err(self.corrupted(format!(
                "requested range [{}, {}) outside object size {}",
                range.start,
                range.end,
                self.size()
            )));
        }
        if range.start == range.end {
            return Ok(Bytes::new());
        }

        self.counters.range_requests.fetch_add(1, Ordering::Relaxed);
        let bytes = self.source.read_range(range.clone()).await?;
        let expected_len = usize::try_from(range.end - range.start)
            .map_err(|_| Error::invariant("edge range length does not fit usize"))?;
        if bytes.len() != expected_len {
            return Err(self.corrupted(format!(
                "range layer returned {} bytes for a {expected_len}-byte range",
                bytes.len()
            )));
        }
        self.counters
            .bytes_read
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    fn install_page_checksums(&self, directory: EdgePageChecksumDirectory) -> Result<()> {
        self.page_checksums
            .set(Arc::new(directory))
            .map_err(|_| Error::invariant("edge page checksum directory installed twice"))
    }

    async fn read_full_body(&self) -> Result<Bytes> {
        self.counters
            .eager_body_reads
            .fetch_add(1, Ordering::Relaxed);
        self.read_range(0..self.size()).await
    }

    async fn read_section(&self, entry: &SectionEntry) -> Result<Bytes> {
        let end = entry
            .offset
            .checked_add(entry.length)
            .ok_or_else(|| self.corrupted("section offset+length overflows u64"))?;
        let bytes = self.read_range(entry.offset..end).await?;
        let actual = xxh3_64(&bytes);
        if actual != entry.xxhash3_64 {
            return Err(self.corrupted(format!(
                "section '{}' (kind 0x{:04x}) xxhash mismatch",
                entry.name, entry.kind
            )));
        }
        Ok(bytes)
    }

    async fn read_section_range(
        &self,
        entry: &SectionEntry,
        relative: Range<u64>,
    ) -> Result<Bytes> {
        if relative.start > relative.end || relative.end > entry.length {
            return Err(self.corrupted(format!(
                "relative section range [{}, {}) outside '{}' length {}",
                relative.start, relative.end, entry.name, entry.length
            )));
        }
        if relative.is_empty() {
            return Ok(Bytes::new());
        }
        let directory = self.page_checksums.get().ok_or_else(|| {
            self.corrupted(
                "partial edge section read refused because this wire version has no page checksums",
            )
        })?;
        let section = directory.for_section(entry).ok_or_else(|| {
            self.corrupted(format!(
                "page checksum directory does not cover section '{}' kind 0x{:04x}",
                entry.name, entry.kind
            ))
        })?;
        let page_size = u64::from(directory.page_size);
        let first_page = relative.start / page_size;
        let last_page = relative
            .end
            .checked_sub(1)
            .ok_or_else(|| self.corrupted("non-empty section range underflowed"))?
            / page_size;
        let expanded_start = first_page
            .checked_mul(page_size)
            .ok_or_else(|| self.corrupted("page-aligned section start overflows u64"))?;
        let expanded_end = last_page
            .checked_add(1)
            .and_then(|page| page.checked_mul(page_size))
            .map(|end| end.min(entry.length))
            .ok_or_else(|| self.corrupted("page-aligned section end overflows u64"))?;
        let absolute_start = entry
            .offset
            .checked_add(expanded_start)
            .ok_or_else(|| self.corrupted("section page start overflows u64"))?;
        let absolute_end = entry
            .offset
            .checked_add(expanded_end)
            .ok_or_else(|| self.corrupted("section page end overflows u64"))?;
        let bytes = self.read_range(absolute_start..absolute_end).await?;

        for page_index in first_page..=last_page {
            let checksum_index = usize::try_from(page_index)
                .map_err(|_| self.corrupted("page checksum index does not fit usize"))?;
            let expected = *section.checksums.get(checksum_index).ok_or_else(|| {
                self.corrupted(format!(
                    "page checksum {page_index} missing for section kind 0x{:04x}",
                    entry.kind
                ))
            })?;
            let page_start = page_index
                .checked_mul(page_size)
                .ok_or_else(|| self.corrupted("section page offset overflows u64"))?;
            let page_end = page_start
                .checked_add(page_size)
                .map(|end| end.min(entry.length))
                .ok_or_else(|| self.corrupted("section page end overflows u64"))?;
            let local_start = usize::try_from(page_start - expanded_start)
                .map_err(|_| self.corrupted("local page start does not fit usize"))?;
            let local_end = usize::try_from(page_end - expanded_start)
                .map_err(|_| self.corrupted("local page end does not fit usize"))?;
            let page = bytes
                .get(local_start..local_end)
                .ok_or_else(|| self.corrupted("verified section page is truncated"))?;
            if xxh3_64(page) != expected {
                return Err(self.corrupted(format!(
                    "section '{}' kind 0x{:04x} page {page_index} xxhash mismatch",
                    entry.name, entry.kind
                )));
            }
        }

        let slice_start = usize::try_from(relative.start - expanded_start)
            .map_err(|_| self.corrupted("section slice start does not fit usize"))?;
        let slice_end = usize::try_from(relative.end - expanded_start)
            .map_err(|_| self.corrupted("section slice end does not fit usize"))?;
        Ok(bytes.slice(slice_start..slice_end))
    }

    fn corrupted(&self, detail: impl Into<String>) -> Error {
        Error::Corrupted {
            path: self.path().to_string(),
            detail: detail.into(),
        }
    }
}

/// Async point reader over one immutable edge SST.
#[derive(Debug)]
pub struct PagedEdgeReader {
    object: PinnedEdgeObject,
    header: EdgeFileHeader,
    footer: EdgeFileFooter,
    offset_width: OffsetWidth,
    fence_index: Option<FenceIndex>,
    /// Entire key section only for standard unfenced SSTs (normally <=1 MiB).
    /// Fetching it once avoids a remote binary-search round trip per comparison.
    unfenced_key_ids: Option<Bytes>,
    /// True only for authenticated v1.2 bodies. False selects the exact
    /// v1.0/v1.1 full-read fallback.
    range_complete: bool,
    eager_reader: OnceCell<Arc<EdgeSstReader>>,
}

impl PagedEdgeReader {
    /// HEAD the object and open a generation-pinned reader.
    pub async fn open(store: Arc<dyn ObjectStore>, path: Path) -> Result<Self> {
        let meta = store.head(&path).await?;
        if meta.location != path {
            return Err(Error::Corrupted {
                path: path.to_string(),
                detail: format!(
                    "edge HEAD returned metadata for unexpected location {}",
                    meta.location
                ),
            });
        }
        Self::open_with_meta(store, meta).await
    }

    /// Open from metadata already obtained by the caller. NamiDB edge SST
    /// paths are create-only UUID objects: a native version/ETag is preferred,
    /// while adapters that expose neither use the common source's explicit
    /// immutable-path generation.
    pub async fn open_with_meta(store: Arc<dyn ObjectStore>, meta: ObjectMeta) -> Result<Self> {
        let source = PinnedObjectRangeSource::from_create_only_meta(store, meta).await?;
        Self::open_with_source(source).await
    }

    /// Explicit cache injection keeps the shared-environment constructor small
    /// and makes cache reuse/eviction testable without mutating process env.
    pub async fn open_with_meta_and_cache(
        store: Arc<dyn ObjectStore>,
        meta: ObjectMeta,
        range_cache: Option<ImmutableRangeCache>,
    ) -> Result<Self> {
        let source = PinnedObjectRangeSource::from_meta_with_cache(store, meta, range_cache)?;
        Self::open_with_source(source).await
    }

    async fn open_with_source(source: PinnedObjectRangeSource) -> Result<Self> {
        let object = PinnedEdgeObject::new(source);
        if object.size() < (HEADER_LEN + FOOTER_TRAILER_LEN) as u64 {
            return Err(object.corrupted(format!("edge SST too small: {} bytes", object.size())));
        }

        let suffix_start = object.size().saturating_sub(EDGE_FOOTER_PREFETCH_BYTES);
        let (header_bytes, suffix) = tokio::try_join!(
            object.read_range(0..HEADER_LEN as u64),
            object.read_range(suffix_start..object.size())
        )?;
        let header = EdgeFileHeader::decode(&header_bytes)
            .map_err(|error| with_object_path(error, object.path()))?;

        let footer_len = EdgeFileFooter::encoded_len_from_trailer(&suffix)
            .map_err(|error| with_object_path(error, object.path()))?
            as u64;
        if footer_len > object.size() {
            return Err(object.corrupted(format!(
                "footer_len {footer_len} exceeds object size {}",
                object.size()
            )));
        }
        if footer_len > MAX_EDGE_FOOTER_BYTES {
            return Err(object.corrupted(format!(
                "footer_len {footer_len} exceeds defensive limit {MAX_EDGE_FOOTER_BYTES}"
            )));
        }
        let footer_bytes = if footer_len <= suffix.len() as u64 {
            suffix.slice(suffix.len() - footer_len as usize..)
        } else {
            object
                .read_range(object.size() - footer_len..object.size())
                .await?
        };
        let (footer, footer_start) = EdgeFileFooter::decode_from_tail(object.size(), &footer_bytes)
            .map_err(|error| with_object_path(error, object.path()))?;
        if footer_start < HEADER_LEN as u64 {
            return Err(object.corrupted("edge footer overlaps fixed header"));
        }

        let offset_width = OffsetWidth::from_bits(footer.offsets_bits)
            .map_err(|error| with_object_path(error, object.path()))?;
        let range_complete = validate_layout(&object, &header, &footer, offset_width)?;
        if range_complete {
            let entry = required_section(&object, &footer, SECTION_PAGE_CHECKSUMS)?;
            if entry.length > MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES {
                return Err(object.corrupted(format!(
                    "page checksum directory length {} exceeds limit \
                     {MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES}",
                    entry.length
                )));
            }
            let encoded = object.read_section(entry).await?;
            let directory = EdgePageChecksumDirectory::decode(&encoded, &footer)
                .map_err(|error| with_object_path(error, object.path()))?;
            object.install_page_checksums(directory)?;
        }
        let bindings: Vec<_> = footer
            .sections
            .iter()
            .filter(|entry| entry.kind == SECTION_SST_BINDING)
            .collect();
        if bindings.len() > 1 {
            return Err(object.corrupted("edge SST contains duplicate identity bindings"));
        }
        if let Some(entry) = bindings.first() {
            if entry.codec != CODEC_NONE || !entry.name.is_empty() {
                return Err(
                    object.corrupted("edge SST identity binding must be unnamed and uncompressed")
                );
            }
            let encoded = object.read_section(entry).await?;
            EdgeSstBinding::decode(&encoded)
                .and_then(|binding| {
                    binding.validate(
                        &header_bytes,
                        &footer.sections,
                        expected_sst_id_from_path(object.path()),
                    )
                })
                .map_err(|error| with_object_path(error, object.path()))?;
        }

        // v1.0/v1.1 cannot authenticate arbitrary section ranges. Keep their
        // full-read fallback lazy so merely opening metadata never downloads
        // the SST.
        let (fence_index, unfenced_key_ids) = if range_complete {
            load_key_index(&object, &footer).await?
        } else {
            (None, None)
        };

        if range_complete {
            validate_index_sentinels(&object, &footer, offset_width).await?;
        }

        Ok(Self {
            object,
            header,
            footer,
            offset_width,
            fence_index,
            unfenced_key_ids,
            range_complete,
            eager_reader: OnceCell::new(),
        })
    }

    pub fn header(&self) -> &EdgeFileHeader {
        &self.header
    }

    pub fn footer(&self) -> &EdgeFileFooter {
        &self.footer
    }

    pub fn direction(&self) -> EdgeDirection {
        EdgeDirection::from_flags(self.header.flags)
    }

    pub fn key_count(&self) -> u64 {
        self.footer.key_count
    }

    pub fn edge_count(&self) -> u64 {
        self.footer.edge_count
    }

    /// Whether this v1.2 body has both edge ordinals and the rooted page
    /// directory needed for an independently verifiable ranged result.
    pub fn is_range_complete(&self) -> bool {
        self.range_complete
    }

    pub fn io_stats(&self) -> PagedEdgeIoStats {
        self.object.stats()
    }

    /// Conservative bytes retained by this query-scoped reader.
    ///
    /// Immutable data pages live in the shared bounded RAM/NVMe range cache;
    /// this reports only routing/footer/checksum metadata plus the small
    /// unfenced key window. A legacy eager fallback is charged as the complete
    /// object because its `EdgeSstReader` owns that full body.
    pub fn resident_metadata_bytes(&self) -> usize {
        let section_bytes = self.footer.sections.iter().fold(0usize, |total, section| {
            total
                .saturating_add(std::mem::size_of::<SectionEntry>())
                .saturating_add(section.name.capacity())
        });
        let fence_bytes = self.fence_index.as_ref().map_or(0, |fence| {
            std::mem::size_of::<FenceIndex>().saturating_add(
                fence.entries.capacity().saturating_mul(std::mem::size_of::<
                    crate::sst::edges::fence_index::FenceEntry,
                >()),
            )
        });
        let checksum_bytes = self.object.page_checksums.get().map_or(0, |directory| {
            std::mem::size_of::<EdgePageChecksumDirectory>().saturating_add(
                directory.sections.iter().fold(0usize, |total, section| {
                    total
                        .saturating_add(std::mem::size_of::<
                            crate::sst::edges::format::SectionPageChecksums,
                        >())
                        .saturating_add(
                            section
                                .checksums
                                .capacity()
                                .saturating_mul(std::mem::size_of::<u64>()),
                        )
                }),
            )
        });
        let eager_body_bytes = self.eager_reader.get().map_or(0, |_| {
            usize::try_from(self.object.size()).unwrap_or(usize::MAX)
        });
        std::mem::size_of::<Self>()
            .saturating_add(self.object.path().as_ref().len())
            .saturating_add(section_bytes)
            .saturating_add(fence_bytes)
            .saturating_add(self.unfenced_key_ids.as_ref().map_or(0, bytes::Bytes::len))
            .saturating_add(checksum_bytes)
            .saturating_add(eager_body_bytes)
    }

    /// Lazily materialise the legacy reader. This remains the escape hatch for
    /// scans and legacy Arrow property streams; current property pages use
    /// [`Self::read_property_rows`] instead.
    pub async fn eager_reader(&self) -> Result<Arc<EdgeSstReader>> {
        let reader = self
            .eager_reader
            .get_or_try_init(|| async {
                let body = self.object.read_full_body().await?;
                let reader = EdgeSstReader::open(body)
                    .map_err(|error| with_object_path(error, self.object.path()))?;
                Ok::<Arc<EdgeSstReader>, Error>(Arc::new(reader))
            })
            .await?;
        Ok(reader.clone())
    }

    /// Hydrate exactly one property row range. Current codec-2/3 sections read
    /// only their compact directory and the independently compressed pages
    /// overlapping `rows`; legacy Arrow streams retain the exact eager
    /// fallback for compatibility.
    pub async fn read_property_rows(
        &self,
        name: &str,
        rows: Range<u64>,
    ) -> Result<Option<Vec<Option<String>>>> {
        if rows.start > rows.end || rows.end > self.footer.edge_count {
            return Err(self.object.corrupted(format!(
                "property row range [{}, {}) exceeds edge_count {}",
                rows.start, rows.end, self.footer.edge_count
            )));
        }
        let Some(entry) = self.footer.find(SECTION_PROPERTY_STREAM, name) else {
            return Ok(None);
        };
        if !matches!(
            entry.codec,
            CODEC_PROPERTY_PAGED_NONE | CODEC_PROPERTY_PAGED_ZSTD
        ) {
            let eager = self.eager_reader().await?;
            let values = if name == OVERFLOW_JSON_NAME {
                eager.read_overflow_strings()?
            } else {
                eager.read_declared_property_strings(name)?
            };
            let Some(values) = values else {
                return Ok(None);
            };
            let start = usize::try_from(rows.start)
                .map_err(|_| Error::invariant("property row start does not fit usize"))?;
            let end = usize::try_from(rows.end)
                .map_err(|_| Error::invariant("property row end does not fit usize"))?;
            return Ok(Some(values[start..end].to_vec()));
        }
        if rows.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let header = self
            .object
            .read_section_range(entry, 0..PROPERTY_PAGES_HEADER_LEN as u64)
            .await?;
        let prefix_len = PropertyPageIndex::required_prefix_len(&header, entry.length)?;
        let prefix = self
            .object
            .read_section_range(entry, 0..prefix_len as u64)
            .await?;
        let index = PropertyPageIndex::parse_prefix(&prefix, entry.length)?;
        if index.row_count != self.footer.edge_count {
            return Err(self.object.corrupted(format!(
                "property stream '{name}' rows {} != edge_count {}",
                index.row_count, self.footer.edge_count
            )));
        }
        let mut out = Vec::with_capacity(
            usize::try_from(rows.end - rows.start)
                .map_err(|_| Error::invariant("property row range does not fit usize"))?,
        );
        for page in index.entries {
            let page_end = page
                .first_row
                .checked_add(u64::from(page.row_count))
                .ok_or_else(|| self.object.corrupted("property page row end exceeds u64"))?;
            if page_end <= rows.start || page.first_row >= rows.end {
                continue;
            }
            let encoded_end = page
                .offset
                .checked_add(page.encoded_len)
                .ok_or_else(|| self.object.corrupted("property page byte end exceeds u64"))?;
            let encoded = self
                .object
                .read_section_range(entry, page.offset..encoded_end)
                .await?;
            let decoded =
                decode_property_page(&encoded, page, entry.codec == CODEC_PROPERTY_PAGED_ZSTD)
                    .map_err(|error| with_object_path(error, self.object.path()))?;
            let take_start = rows.start.saturating_sub(page.first_row) as usize;
            let take_end = (rows.end.min(page_end) - page.first_row) as usize;
            out.extend_from_slice(&decoded[take_start..take_end]);
        }
        if out.len() as u64 != rows.end - rows.start {
            return Err(self
                .object
                .corrupted("property pages did not cover the requested row range"));
        }
        Ok(Some(out))
    }

    /// Resolve a key to its sorted key-array position.
    pub async fn position_of(&self, key: &[u8; 16]) -> Result<Option<u64>> {
        if !self.range_complete {
            // Old bodies cannot prove the partial topology and metadata rows
            // together, so use the one verified full reader.
            let eager = self.eager_reader().await?;
            return eager.position_of(key);
        }
        if self.footer.key_count == 0
            || key < &self.footer.min_key_id
            || key > &self.footer.max_key_id
        {
            return Ok(None);
        }

        if let Some(key_ids) = self.unfenced_key_ids.as_ref() {
            return Ok(binary_search_fixed_16(key_ids, key)?.map(|idx| idx as u64));
        }

        let Some(fence) = self.fence_index.as_ref() else {
            return self.remote_position_of(key).await;
        };
        let (lo, hi) = fence.bracket_for(key, self.footer.key_count);
        if lo >= hi {
            return Ok(None);
        }
        let key_entry = required_section(&self.object, &self.footer, SECTION_KEY_IDS)?;
        let start = lo
            .checked_mul(16)
            .ok_or_else(|| Error::invariant("key window start overflows u64"))?;
        let end = hi
            .checked_mul(16)
            .ok_or_else(|| Error::invariant("key window end overflows u64"))?;
        let window = self
            .object
            .read_section_range(key_entry, start..end)
            .await?;
        validate_sorted_key_window(&self.object, &window, lo, hi, fence)?;
        Ok(binary_search_fixed_16(&window, key)?.map(|offset| lo + offset as u64))
    }

    async fn remote_position_of(&self, key: &[u8; 16]) -> Result<Option<u64>> {
        let key_entry = required_section(&self.object, &self.footer, SECTION_KEY_IDS)?;
        let mut lo = 0u64;
        let mut hi = self.footer.key_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let start = mid
                .checked_mul(16)
                .ok_or_else(|| Error::invariant("remote key probe offset overflows u64"))?;
            let bytes = self
                .object
                .read_section_range(key_entry, start..start + 16)
                .await?;
            match bytes.as_ref().cmp(key.as_slice()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    /// Fetch one complete adjacency plus its LSNs/tombstone bits.
    pub async fn lookup(&self, key: &[u8; 16]) -> Result<Option<EdgeLookup>> {
        if !self.range_complete {
            return self.eager_reader().await?.lookup(key);
        }
        let Some(block) = self.locate_block(key).await? else {
            return Ok(None);
        };
        let (partners, consumed) = read_partner_block_expected(&block.bytes, 0, block.degree)
            .map_err(|error| with_object_path(error, self.object.path()))?;
        if consumed != block.bytes.len() {
            return Err(self.object.corrupted(format!(
                "partner block consumed {consumed} bytes, expected {}",
                block.bytes.len()
            )));
        }
        debug_assert_eq!(partners.len(), block.degree);
        let (lsns, tombstones) = self
            .read_edge_metadata(block.edge_start, block.degree)
            .await?;
        let edge_offset = usize::try_from(block.edge_start)
            .map_err(|_| Error::invariant("edge ordinal does not fit usize"))?;
        Ok(Some(EdgeLookup {
            partners,
            lsns,
            tombstones,
            edge_offset,
        }))
    }

    /// Exact endpoint probe. It reads the selected partner block but only the
    /// winning 8-byte LSN and (when present) one tombstone byte.
    pub async fn lookup_partner(
        &self,
        key: &[u8; 16],
        partner: &[u8; 16],
    ) -> Result<Option<EdgePointLookup>> {
        if !self.range_complete {
            return self.eager_reader().await?.lookup_partner(key, partner);
        }
        let Some(block) = self.locate_block_descriptor(key).await? else {
            return Ok(None);
        };
        let partners = required_section(&self.object, &self.footer, SECTION_PARTNERS)?;
        let block_len = block.partner_range.end - block.partner_range.start;
        let prefix_len = block_len.min(11);
        let prefix = self
            .object
            .read_section_range(
                partners,
                block.partner_range.start..block.partner_range.start + prefix_len,
            )
            .await?;
        let (wire_degree, varint_len) =
            read_varint(&prefix, 0).map_err(|error| with_object_path(error, self.object.path()))?;
        let wire_degree = usize::try_from(wire_degree)
            .map_err(|_| Error::invariant("partner block degree does not fit usize"))?;
        if wire_degree != block.degree {
            return Err(self.object.corrupted(format!(
                "partner block degree {wire_degree} disagrees with ordinal delta {}",
                block.degree
            )));
        }
        let tag = *prefix.get(varint_len).ok_or_else(|| {
            self.object
                .corrupted("partner block is missing its encoding tag")
        })?;
        let relative = match tag {
            TAG_DENSE => {
                self.find_dense_partner_remote(partners, &block, varint_len + 1, partner)
                    .await?
            }
            TAG_SPLIT => {
                validate_partner_block_read_bounds(block.degree, block_len)
                    .map_err(|error| with_object_path(error, self.object.path()))?;
                let bytes = self
                    .object
                    .read_section_range(partners, block.partner_range.clone())
                    .await?;
                find_partner_in_block(&bytes, 0, partner)
                    .map_err(|error| with_object_path(error, self.object.path()))?
            }
            other => {
                return Err(self
                    .object
                    .corrupted(format!("unknown partner block tag 0x{other:02x}")));
            }
        };
        let Some(relative) = relative else {
            return Ok(None);
        };
        if relative >= block.degree {
            return Err(self
                .object
                .corrupted("partner search returned an index outside block degree"));
        }
        let edge_ordinal = block
            .edge_start
            .checked_add(relative as u64)
            .ok_or_else(|| Error::invariant("edge ordinal overflows u64"))?;
        let lsn = self.read_one_lsn(edge_ordinal).await?;
        let tombstone = self.read_one_tombstone(edge_ordinal).await?;
        let edge_offset = usize::try_from(edge_ordinal)
            .map_err(|_| Error::invariant("edge ordinal does not fit usize"))?;
        Ok(Some(EdgePointLookup {
            lsn,
            tombstone,
            edge_offset,
        }))
    }

    async fn find_dense_partner_remote(
        &self,
        partners: &SectionEntry,
        block: &LocatedBlockDescriptor,
        header_len: usize,
        target: &[u8; 16],
    ) -> Result<Option<usize>> {
        let payload_start = block
            .partner_range
            .start
            .checked_add(header_len as u64)
            .ok_or_else(|| Error::invariant("dense partner payload start overflows u64"))?;
        let payload_len = (block.degree as u64)
            .checked_mul(16)
            .ok_or_else(|| Error::invariant("dense partner payload length overflows u64"))?;
        let payload_end = payload_start
            .checked_add(payload_len)
            .ok_or_else(|| Error::invariant("dense partner payload end overflows u64"))?;
        if payload_end != block.partner_range.end {
            return Err(self.object.corrupted(format!(
                "dense partner payload ends at {payload_end}, expected {}",
                block.partner_range.end
            )));
        }

        let mut lo = 0usize;
        let mut hi = block.degree;
        // Keep a tiny bounded set of pages visited by this binary search. This
        // avoids repeated authenticated GETs near the winning row when the
        // process-wide cache is disabled without retaining a hub-sized block.
        let page_size = u64::from(
            self.object
                .page_checksums
                .get()
                .ok_or_else(|| {
                    self.object
                        .corrupted("page checksum directory is not installed")
                })?
                .page_size,
        );
        let mut pages: HashMap<u64, Bytes> = HashMap::new();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let start = payload_start
                .checked_add(
                    (mid as u64)
                        .checked_mul(16)
                        .ok_or_else(|| Error::invariant("dense partner probe overflows u64"))?,
                )
                .ok_or_else(|| Error::invariant("dense partner probe start overflows u64"))?;
            let page_index = start / page_size;
            let page_start = page_index
                .checked_mul(page_size)
                .ok_or_else(|| Error::invariant("dense partner page start overflows u64"))?;
            let page_end = page_start
                .checked_add(page_size)
                .map(|end| end.min(partners.length))
                .ok_or_else(|| Error::invariant("dense partner page end overflows u64"))?;
            let page = match pages.get(&page_index) {
                Some(page) => page.clone(),
                None => {
                    let page = self
                        .object
                        .read_section_range(partners, page_start..page_end)
                        .await?;
                    if pages.len() >= MAX_DENSE_PROBE_PAGES {
                        pages.clear();
                    }
                    pages.insert(page_index, page.clone());
                    page
                }
            };
            let local_start = usize::try_from(start - page_start)
                .map_err(|_| Error::invariant("dense partner local offset does not fit usize"))?;
            let local_end = local_start
                .checked_add(16)
                .ok_or_else(|| Error::invariant("dense partner local end overflows usize"))?;
            let mut crossing_probe = [0u8; 16];
            let probe = if let Some(probe) = page.get(local_start..local_end) {
                probe
            } else {
                let first = page.get(local_start..).ok_or_else(|| {
                    self.object
                        .corrupted("dense partner probe starts outside verified page")
                })?;
                if first.len() >= 16 {
                    return Err(self
                        .object
                        .corrupted("dense partner page slice has inconsistent length"));
                }
                crossing_probe[..first.len()].copy_from_slice(first);
                let next_index = page_index
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("dense partner page index overflows u64"))?;
                let next_start = next_index
                    .checked_mul(page_size)
                    .ok_or_else(|| Error::invariant("dense partner next page overflows u64"))?;
                let next_end = next_start
                    .checked_add(page_size)
                    .map(|end| end.min(partners.length))
                    .ok_or_else(|| Error::invariant("dense partner next page end overflows u64"))?;
                let next = match pages.get(&next_index) {
                    Some(page) => page.clone(),
                    None => {
                        let page = self
                            .object
                            .read_section_range(partners, next_start..next_end)
                            .await?;
                        if pages.len() >= MAX_DENSE_PROBE_PAGES {
                            pages.clear();
                        }
                        pages.insert(next_index, page.clone());
                        page
                    }
                };
                let remaining = 16 - first.len();
                let tail = next.get(..remaining).ok_or_else(|| {
                    self.object
                        .corrupted("dense partner probe is truncated across pages")
                })?;
                crossing_probe[first.len()..].copy_from_slice(tail);
                crossing_probe.as_slice()
            };
            match probe.cmp(target.as_slice()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid)),
            }
        }
        Ok(None)
    }

    async fn locate_block(&self, key: &[u8; 16]) -> Result<Option<LocatedBlock>> {
        let Some(descriptor) = self.locate_block_descriptor(key).await? else {
            return Ok(None);
        };
        let block_len = descriptor.partner_range.end - descriptor.partner_range.start;
        validate_partner_block_read_bounds(descriptor.degree, block_len)
            .map_err(|error| with_object_path(error, self.object.path()))?;
        let partners = required_section(&self.object, &self.footer, SECTION_PARTNERS)?;
        let bytes = self
            .object
            .read_section_range(partners, descriptor.partner_range)
            .await?;
        Ok(Some(LocatedBlock {
            bytes,
            edge_start: descriptor.edge_start,
            degree: descriptor.degree,
        }))
    }

    async fn locate_block_descriptor(
        &self,
        key: &[u8; 16],
    ) -> Result<Option<LocatedBlockDescriptor>> {
        let Some(index) = self.position_of(key).await? else {
            return Ok(None);
        };
        let offsets = required_section(&self.object, &self.footer, SECTION_OFFSETS)?;
        let ordinals = required_section(&self.object, &self.footer, SECTION_EDGE_ORDINALS)?;
        let stride = self.offset_width.bytes() as u64;
        let offsets_start = index
            .checked_mul(stride)
            .ok_or_else(|| Error::invariant("offset row start overflows u64"))?;
        let offsets_end = offsets_start
            .checked_add(stride * 2)
            .ok_or_else(|| Error::invariant("offset row end overflows u64"))?;
        let ordinal_start = index
            .checked_mul(8)
            .ok_or_else(|| Error::invariant("ordinal row start overflows u64"))?;
        let ordinal_end = ordinal_start
            .checked_add(16)
            .ok_or_else(|| Error::invariant("ordinal row end overflows u64"))?;

        let (offset_bytes, ordinal_bytes) = tokio::try_join!(
            self.object
                .read_section_range(offsets, offsets_start..offsets_end),
            self.object
                .read_section_range(ordinals, ordinal_start..ordinal_end)
        )?;
        let partner_start = read_offset(&offset_bytes, 0, self.offset_width)?;
        let partner_end = read_offset(&offset_bytes, self.offset_width.bytes(), self.offset_width)?;
        if partner_start >= partner_end {
            return Err(self.object.corrupted(format!(
                "key index {index} has invalid partner range [{partner_start}, {partner_end})"
            )));
        }
        let partners = required_section(&self.object, &self.footer, SECTION_PARTNERS)?;
        if partner_end > partners.length {
            return Err(self.object.corrupted(format!(
                "partner range end {partner_end} exceeds section length {}",
                partners.length
            )));
        }

        let edge_start = read_u64_pair_start(&ordinal_bytes)?;
        let edge_end = read_u64_pair_end(&ordinal_bytes)?;
        if edge_start >= edge_end || edge_end > self.footer.edge_count {
            return Err(self.object.corrupted(format!(
                "key index {index} has invalid edge ordinal range [{edge_start}, {edge_end})"
            )));
        }
        let degree = usize::try_from(edge_end - edge_start)
            .map_err(|_| Error::invariant("edge degree does not fit usize"))?;
        Ok(Some(LocatedBlockDescriptor {
            partner_range: partner_start..partner_end,
            edge_start,
            degree,
        }))
    }

    async fn read_edge_metadata(
        &self,
        edge_start: u64,
        degree: usize,
    ) -> Result<(Vec<u64>, Vec<bool>)> {
        let degree_u64 =
            u64::try_from(degree).map_err(|_| Error::invariant("edge degree exceeds u64"))?;
        let edge_end = edge_start
            .checked_add(degree_u64)
            .ok_or_else(|| Error::invariant("edge metadata range overflows u64"))?;
        let lsn_entry = required_section(&self.object, &self.footer, SECTION_PER_EDGE_LSN)?;
        let lsn_start = edge_start
            .checked_mul(8)
            .ok_or_else(|| Error::invariant("LSN byte start overflows u64"))?;
        let lsn_end = edge_end
            .checked_mul(8)
            .ok_or_else(|| Error::invariant("LSN byte end overflows u64"))?;

        let tomb_entry = self.footer.find_kind(SECTION_PER_EDGE_TOMBSTONES);
        let (lsn_bytes, tomb_bytes, tomb_bit_base) = if let Some(tomb_entry) = tomb_entry {
            let tomb_start = edge_start / 8;
            let tomb_end = edge_end
                .checked_add(7)
                .ok_or_else(|| Error::invariant("tombstone end rounds past u64"))?
                / 8;
            let (lsns, tombs) = tokio::try_join!(
                self.object
                    .read_section_range(lsn_entry, lsn_start..lsn_end),
                self.object
                    .read_section_range(tomb_entry, tomb_start..tomb_end)
            )?;
            (lsns, Some(tombs), tomb_start * 8)
        } else {
            (
                self.object
                    .read_section_range(lsn_entry, lsn_start..lsn_end)
                    .await?,
                None,
                0,
            )
        };

        let mut lsns = Vec::with_capacity(degree);
        for row in lsn_bytes.chunks_exact(8) {
            lsns.push(u64::from_le_bytes(row.try_into().unwrap()));
        }
        if lsns.len() != degree {
            return Err(self
                .object
                .corrupted(format!("decoded {} LSNs for degree {degree}", lsns.len())));
        }
        let tombstones = if let Some(bytes) = tomb_bytes {
            (edge_start..edge_end)
                .map(|ordinal| {
                    let local = ordinal - tomb_bit_base;
                    (bytes[local as usize / 8] >> (local % 8)) & 1 == 1
                })
                .collect()
        } else {
            vec![false; degree]
        };
        Ok((lsns, tombstones))
    }

    async fn read_one_lsn(&self, edge_ordinal: u64) -> Result<u64> {
        if edge_ordinal >= self.footer.edge_count {
            return Err(self
                .object
                .corrupted("edge ordinal exceeds footer edge_count"));
        }
        let entry = required_section(&self.object, &self.footer, SECTION_PER_EDGE_LSN)?;
        let start = edge_ordinal
            .checked_mul(8)
            .ok_or_else(|| Error::invariant("LSN row start overflows u64"))?;
        let end = start
            .checked_add(8)
            .ok_or_else(|| Error::invariant("LSN row end overflows u64"))?;
        let bytes = self.object.read_section_range(entry, start..end).await?;
        Ok(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
    }

    async fn read_one_tombstone(&self, edge_ordinal: u64) -> Result<bool> {
        let Some(entry) = self.footer.find_kind(SECTION_PER_EDGE_TOMBSTONES) else {
            return Ok(false);
        };
        let byte_index = edge_ordinal / 8;
        let bytes = self
            .object
            .read_section_range(entry, byte_index..byte_index + 1)
            .await?;
        Ok((bytes[0] >> (edge_ordinal % 8)) & 1 == 1)
    }
}

#[derive(Debug)]
struct LocatedBlock {
    bytes: Bytes,
    edge_start: u64,
    degree: usize,
}

#[derive(Debug)]
struct LocatedBlockDescriptor {
    partner_range: Range<u64>,
    edge_start: u64,
    degree: usize,
}

fn validate_layout(
    object: &PinnedEdgeObject,
    header: &EdgeFileHeader,
    footer: &EdgeFileFooter,
    offset_width: OffsetWidth,
) -> Result<bool> {
    for kind in [
        SECTION_KEY_IDS,
        SECTION_OFFSETS,
        SECTION_PARTNERS,
        SECTION_PER_EDGE_LSN,
    ] {
        let entry = required_section(object, footer, kind)?;
        if entry.codec != CODEC_NONE {
            return Err(object.corrupted(format!(
                "topology section kind 0x{kind:04x} uses unsupported codec {}",
                entry.codec
            )));
        }
    }

    let key_ids = required_section(object, footer, SECTION_KEY_IDS)?;
    let expected_keys = footer
        .key_count
        .checked_mul(16)
        .ok_or_else(|| Error::invariant("key_ids length overflows u64"))?;
    if key_ids.length != expected_keys {
        return Err(object.corrupted(format!(
            "key_ids length {} != key_count*16 ({expected_keys})",
            key_ids.length
        )));
    }

    let offsets = required_section(object, footer, SECTION_OFFSETS)?;
    let expected_offsets = footer
        .key_count
        .checked_add(1)
        .and_then(|rows| rows.checked_mul(offset_width.bytes() as u64))
        .ok_or_else(|| Error::invariant("offsets length overflows u64"))?;
    if offsets.length != expected_offsets {
        return Err(object.corrupted(format!(
            "offsets length {} != (key_count+1)*{} ({expected_offsets})",
            offsets.length,
            offset_width.bytes()
        )));
    }

    let lsn = required_section(object, footer, SECTION_PER_EDGE_LSN)?;
    let expected_lsn = footer
        .edge_count
        .checked_mul(8)
        .ok_or_else(|| Error::invariant("per-edge LSN length overflows u64"))?;
    if lsn.length != expected_lsn {
        return Err(object.corrupted(format!(
            "per_edge_lsn length {} != edge_count*8 ({expected_lsn})",
            lsn.length
        )));
    }
    if footer.key_count > footer.edge_count {
        return Err(object.corrupted(format!(
            "key_count {} exceeds edge_count {}",
            footer.key_count, footer.edge_count
        )));
    }

    let tomb = unique_optional_section(object, footer, SECTION_PER_EDGE_TOMBSTONES)?;
    let flag_has_tombstones = header.flags & FLAG_HAS_TOMBSTONES != 0;
    if flag_has_tombstones != tomb.is_some() {
        return Err(object.corrupted("HAS_TOMBSTONES flag and tombstone section presence disagree"));
    }
    if let Some(tomb) = tomb {
        if tomb.codec != CODEC_NONE {
            return Err(object.corrupted("tombstone section must be uncompressed"));
        }
        let expected = footer
            .edge_count
            .checked_add(7)
            .ok_or_else(|| Error::invariant("tombstone length rounds past u64"))?
            / 8;
        if tomb.length != expected {
            return Err(object.corrupted(format!(
                "tombstone length {} != ceil(edge_count/8) ({expected})",
                tomb.length
            )));
        }
    }

    if let Some(fence) = unique_optional_section(object, footer, SECTION_FENCE_INDEX)? {
        if fence.codec != CODEC_NONE {
            return Err(object.corrupted("fence index must be uncompressed"));
        }
    }

    let ordinals = unique_optional_section(object, footer, SECTION_EDGE_ORDINALS)?;
    if let Some(ordinals) = ordinals {
        if ordinals.codec != CODEC_NONE {
            return Err(object.corrupted("edge ordinal section must be uncompressed"));
        }
        let expected = footer
            .key_count
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(8))
            .ok_or_else(|| Error::invariant("edge ordinal length overflows u64"))?;
        if ordinals.length != expected {
            return Err(object.corrupted(format!(
                "edge ordinal length {} != (key_count+1)*8 ({expected})",
                ordinals.length
            )));
        }
    }
    let page_checksums = unique_optional_section(object, footer, SECTION_PAGE_CHECKSUMS)?;
    if let Some(page_checksums) = page_checksums {
        if page_checksums.codec != CODEC_NONE || !page_checksums.name.is_empty() {
            return Err(
                object.corrupted("page checksum directory must be unnamed and uncompressed")
            );
        }
        if page_checksums.length > MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES {
            return Err(object.corrupted(format!(
                "page checksum directory length {} exceeds limit \
                 {MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES}",
                page_checksums.length
            )));
        }
    }

    if header.format_minor < 2 {
        // v1.0 has no ordinals; v1.1 has ordinals but no authenticated pages.
        // Both take the exact full-body path rather than trusting raw ranges.
        return Ok(false);
    }
    if ordinals.is_none() {
        return Err(object.corrupted("edge SST v1.2 is missing its edge ordinal section"));
    }
    if page_checksums.is_none() {
        return Err(object.corrupted("edge SST v1.2 is missing its page checksum directory"));
    }
    Ok(true)
}

async fn load_key_index(
    object: &PinnedEdgeObject,
    footer: &EdgeFileFooter,
) -> Result<(Option<FenceIndex>, Option<Bytes>)> {
    if let Some(entry) = footer.find_kind(SECTION_FENCE_INDEX) {
        if entry.length > MAX_CACHED_EDGE_FENCE_BYTES {
            return Ok((None, None));
        }
        let bytes = object.read_section(entry).await?;
        let fence =
            FenceIndex::decode(&bytes).map_err(|error| with_object_path(error, object.path()))?;
        validate_fence(object, footer, &fence)?;
        Ok((Some(fence), None))
    } else {
        let entry = required_section(object, footer, SECTION_KEY_IDS)?;
        if entry.length > MAX_UNFENCED_KEY_IDS_BYTES {
            return Ok((None, None));
        }
        let bytes = object.read_section(entry).await?;
        validate_full_key_array(object, footer, &bytes)?;
        Ok((None, Some(bytes)))
    }
}

fn validate_fence(
    object: &PinnedEdgeObject,
    footer: &EdgeFileFooter,
    fence: &FenceIndex,
) -> Result<()> {
    if footer.key_count == 0 {
        if !fence.entries.is_empty() {
            return Err(object.corrupted("empty SST has a non-empty fence index"));
        }
        return Ok(());
    }
    if fence.stride == 0 || fence.entries.is_empty() {
        return Err(object.corrupted("non-empty fence index has zero stride or no entries"));
    }
    let expected_entries = footer.key_count.div_ceil(fence.stride as u64);
    if fence.entries.len() as u64 != expected_entries {
        return Err(object.corrupted(format!(
            "fence entry count {} != ceil(key_count/stride) ({expected_entries})",
            fence.entries.len()
        )));
    }
    for (index, entry) in fence.entries.iter().enumerate() {
        let expected_offset = (index as u64)
            .checked_mul(fence.stride as u64)
            .and_then(|row| row.checked_mul(16))
            .ok_or_else(|| Error::invariant("fence offset overflows u64"))?;
        if entry.key_ids_offset != expected_offset {
            return Err(object.corrupted(format!(
                "fence entry {index} offset {} != expected {expected_offset}",
                entry.key_ids_offset
            )));
        }
        if index > 0 && fence.entries[index - 1].key >= entry.key {
            return Err(object.corrupted("fence keys are not strictly increasing"));
        }
    }
    if fence.entries[0].key != footer.min_key_id {
        return Err(object.corrupted("first fence key disagrees with footer min_key_id"));
    }
    Ok(())
}

fn validate_full_key_array(
    object: &PinnedEdgeObject,
    footer: &EdgeFileFooter,
    bytes: &[u8],
) -> Result<()> {
    let expected = footer
        .key_count
        .checked_mul(16)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| Error::invariant("unfenced key array length does not fit usize"))?;
    if bytes.len() != expected {
        return Err(object.corrupted("unfenced key array has unexpected length"));
    }
    if footer.key_count == 0 {
        return Ok(());
    }
    let row_count = bytes.len() / 16;
    for index in 1..row_count {
        let previous = &bytes[(index - 1) * 16..index * 16];
        let current = &bytes[index * 16..(index + 1) * 16];
        if previous >= current {
            return Err(object.corrupted("key_ids are not strictly increasing"));
        }
    }
    if bytes[..16] != footer.min_key_id || bytes[bytes.len() - 16..] != footer.max_key_id {
        return Err(object.corrupted("key_ids endpoints disagree with footer min/max"));
    }
    Ok(())
}

fn validate_sorted_key_window(
    object: &PinnedEdgeObject,
    bytes: &[u8],
    lo: u64,
    hi: u64,
    fence: &FenceIndex,
) -> Result<()> {
    let expected = usize::try_from((hi - lo) * 16)
        .map_err(|_| Error::invariant("key window length does not fit usize"))?;
    if bytes.len() != expected || bytes.is_empty() {
        return Err(object.corrupted(format!(
            "key window length {} != expected {expected}",
            bytes.len()
        )));
    }
    let row_count = bytes.len() / 16;
    for index in 1..row_count {
        let previous = &bytes[(index - 1) * 16..index * 16];
        let current = &bytes[index * 16..(index + 1) * 16];
        if previous >= current {
            return Err(object.corrupted("key window is not strictly increasing"));
        }
    }
    let fence_index = usize::try_from(lo / fence.stride as u64)
        .map_err(|_| Error::invariant("fence index does not fit usize"))?;
    if fence
        .entries
        .get(fence_index)
        .map(|entry| entry.key.as_slice())
        != Some(&bytes[..16])
    {
        return Err(object.corrupted("key window first row disagrees with fence"));
    }
    if let Some(next) = fence.entries.get(fence_index + 1) {
        if &bytes[bytes.len() - 16..] >= next.key.as_slice() {
            return Err(object.corrupted("key window crosses its next fence"));
        }
    }
    Ok(())
}

async fn validate_index_sentinels(
    object: &PinnedEdgeObject,
    footer: &EdgeFileFooter,
    width: OffsetWidth,
) -> Result<()> {
    let offsets = required_section(object, footer, SECTION_OFFSETS)?;
    let ordinals = required_section(object, footer, SECTION_EDGE_ORDINALS)?;
    let offset_stride = width.bytes() as u64;
    let last_offset_start = footer
        .key_count
        .checked_mul(offset_stride)
        .ok_or_else(|| Error::invariant("last offset position overflows u64"))?;
    let last_ordinal_start = footer
        .key_count
        .checked_mul(8)
        .ok_or_else(|| Error::invariant("last ordinal position overflows u64"))?;
    let (first_offset, last_offset, first_ordinal, last_ordinal) = tokio::try_join!(
        object.read_section_range(offsets, 0..offset_stride),
        object.read_section_range(
            offsets,
            last_offset_start..last_offset_start + offset_stride
        ),
        object.read_section_range(ordinals, 0..8),
        object.read_section_range(ordinals, last_ordinal_start..last_ordinal_start + 8)
    )?;
    let partners = required_section(object, footer, SECTION_PARTNERS)?;
    if read_offset(&first_offset, 0, width)? != 0
        || read_offset(&last_offset, 0, width)? != partners.length
    {
        return Err(object.corrupted("offset sentinels do not span the complete partners section"));
    }
    if read_u64(&first_ordinal)? != 0 || read_u64(&last_ordinal)? != footer.edge_count {
        return Err(object.corrupted("edge ordinal sentinels do not span footer.edge_count"));
    }
    if footer.key_count > 0 {
        let key_ids = required_section(object, footer, SECTION_KEY_IDS)?;
        let last_key_start = footer
            .key_count
            .checked_sub(1)
            .and_then(|row| row.checked_mul(16))
            .ok_or_else(|| Error::invariant("last key position overflows u64"))?;
        let (first_key, last_key) = tokio::try_join!(
            object.read_section_range(key_ids, 0..16),
            object.read_section_range(key_ids, last_key_start..last_key_start + 16)
        )?;
        if first_key.as_ref() != footer.min_key_id.as_slice()
            || last_key.as_ref() != footer.max_key_id.as_slice()
        {
            return Err(object.corrupted("key_id sentinels disagree with footer min/max"));
        }
    }
    Ok(())
}

fn required_section<'a>(
    object: &PinnedEdgeObject,
    footer: &'a EdgeFileFooter,
    kind: u16,
) -> Result<&'a SectionEntry> {
    let mut matches = footer.sections.iter().filter(|entry| entry.kind == kind);
    let first = matches.next().ok_or_else(|| {
        object.corrupted(format!(
            "edge SST missing mandatory section kind 0x{kind:04x}"
        ))
    })?;
    if matches.next().is_some() {
        return Err(object.corrupted(format!(
            "edge SST contains duplicate section kind 0x{kind:04x}"
        )));
    }
    Ok(first)
}

fn unique_optional_section<'a>(
    object: &PinnedEdgeObject,
    footer: &'a EdgeFileFooter,
    kind: u16,
) -> Result<Option<&'a SectionEntry>> {
    let mut matches = footer.sections.iter().filter(|entry| entry.kind == kind);
    let first = matches.next();
    if matches.next().is_some() {
        return Err(object.corrupted(format!(
            "edge SST contains duplicate section kind 0x{kind:04x}"
        )));
    }
    Ok(first)
}

fn read_u64(bytes: &[u8]) -> Result<u64> {
    let row: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| Error::invariant("u64 row truncated"))?
        .try_into()
        .unwrap();
    Ok(u64::from_le_bytes(row))
}

fn read_u64_pair_start(bytes: &[u8]) -> Result<u64> {
    read_u64(bytes)
}

fn read_u64_pair_end(bytes: &[u8]) -> Result<u64> {
    read_u64(
        bytes
            .get(8..)
            .ok_or_else(|| Error::invariant("u64 pair truncated"))?,
    )
}

fn with_object_path(error: Error, path: &Path) -> Error {
    match error {
        Error::Corrupted { detail, .. } => Error::Corrupted {
            path: path.to_string(),
            detail,
        },
        other => other,
    }
}

fn expected_sst_id_from_path(path: &Path) -> Option<[u8; 16]> {
    let filename = path.filename()?;
    let prefix = filename.split('-').next()?;
    if prefix.len() != 32 {
        return None;
    }
    Uuid::parse_str(prefix).ok().map(|id| *id.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::{ObjectStoreExt, PutPayload};

    use crate::range_cache::RangeCacheConfig;
    use crate::sst::edges::format::{
        EDGE_CHECKSUM_PAGE_BYTES, FORMAT_MINOR, SECTION_EDGE_ORDINALS, SECTION_FENCE_INDEX,
        SECTION_KEY_IDS, SECTION_OFFSETS, SECTION_PAGE_CHECKSUMS, SECTION_PARTNERS,
        SECTION_PER_EDGE_LSN, SECTION_PER_EDGE_TOMBSTONES, SECTION_PROPERTY_STREAM,
        SST_BINDING_LEN,
    };
    use crate::sst::edges::writer::{EdgeRecord, EdgeSstWriter, EdgeSstWriterOptions};

    fn id(namespace: u64, value: u64) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[..8].copy_from_slice(&namespace.to_be_bytes());
        id[8..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn write_body(key_count: u64, degree: u64, fence_stride: u32) -> Bytes {
        let mut options =
            EdgeSstWriterOptions::new(EdgeDirection::Forward, "CITES", "Articulo", "Articulo");
        options.fence_threshold = 4;
        options.fence_stride = fence_stride;
        let mut writer = EdgeSstWriter::new(options);
        for key_index in 0..key_count {
            for partner_index in 0..degree {
                let ordinal = key_index * degree + partner_index;
                writer
                    .append(EdgeRecord {
                        key_id: id(1, key_index),
                        partner_id: id(2, ordinal),
                        lsn: 10_000 + ordinal,
                        tombstone: ordinal % 997 == 17,
                        declared_properties: vec![],
                        overflow_json: None,
                    })
                    .unwrap();
            }
        }
        writer.finish().unwrap().body
    }

    fn write_body_with_property() -> Bytes {
        let mut options =
            EdgeSstWriterOptions::new(EdgeDirection::Forward, "CITES", "Articulo", "Articulo");
        options.fence_threshold = 4;
        options.fence_stride = 2;
        options.compress_property_streams = false;
        options.declared_properties = vec!["codigo".into()];
        let mut writer = EdgeSstWriter::new(options);
        for key_index in 0..8 {
            writer
                .append(EdgeRecord {
                    key_id: id(1, key_index),
                    partner_id: id(2, key_index),
                    lsn: 10_000 + key_index,
                    tombstone: key_index == 3,
                    declared_properties: vec![Some(format!("\"C-{key_index}\""))],
                    overflow_json: None,
                })
                .unwrap();
        }
        writer.finish().unwrap().body
    }

    async fn put_body(store: &Arc<dyn ObjectStore>, path: &Path, body: Bytes) -> ObjectMeta {
        store.put(path, PutPayload::from(body)).await.unwrap();
        store.head(path).await.unwrap()
    }

    #[tokio::test]
    async fn ranged_lookup_matches_eager_without_fetching_the_sst() {
        let body = write_body(4_096, 3, 64);
        let eager = EdgeSstReader::open(body.clone()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/ranged.sst");
        put_body(&store, &path, body.clone()).await;

        let reader = PagedEdgeReader::open(store, path).await.unwrap();
        assert!(reader.is_range_complete());
        assert_eq!(reader.header().format_minor, FORMAT_MINOR);
        assert_eq!(reader.io_stats().eager_body_reads, 0);

        for key_index in [0, 17, 2_048, 4_095] {
            let target = id(1, key_index);
            assert_eq!(
                reader.lookup(&target).await.unwrap(),
                eager.lookup(&target).unwrap()
            );
        }
        assert!(reader.lookup(&id(9, 9)).await.unwrap().is_none());

        let stats = reader.io_stats();
        assert_eq!(stats.eager_body_reads, 0);
        assert!(
            stats.range_requests > 0 && stats.bytes_read > 0,
            "v1.2 lookup must be served by authenticated ranges"
        );
    }

    #[tokio::test]
    async fn property_pages_hydrate_exact_row_range_without_eager_body() {
        let body = write_body_with_property();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/property-pages.sst");
        put_body(&store, &path, body).await;
        let reader = PagedEdgeReader::open(store, path).await.unwrap();
        let values = reader
            .read_property_rows("codigo", 2..6)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            values,
            vec![
                Some(r#""C-2""#.into()),
                Some(r#""C-3""#.into()),
                Some(r#""C-4""#.into()),
                Some(r#""C-5""#.into()),
            ]
        );
        assert_eq!(reader.io_stats().eager_body_reads, 0);
    }

    #[tokio::test]
    async fn binding_rejects_a_different_uuid_object_path() {
        let body = write_body(8, 1, 4);
        let (footer, _) = EdgeFileFooter::decode(&body).unwrap();
        let binding_entry = footer.find_kind(SECTION_SST_BINDING).unwrap();
        let binding_start = binding_entry.offset as usize;
        let binding_end = (binding_entry.offset + binding_entry.length) as usize;
        let binding = EdgeSstBinding::decode(&body[binding_start..binding_end]).unwrap();
        let mut wrong = Uuid::from_u128(1);
        if wrong.as_bytes() == &binding.sst_id {
            wrong = Uuid::from_u128(2);
        }
        let path = Path::from(format!("{}-edges-fwd-CITES.ep.csr", wrong.simple()));
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        put_body(&store, &path, body).await;
        assert!(matches!(
            PagedEdgeReader::open(store, path).await.unwrap_err(),
            Error::Corrupted { .. }
        ));
    }

    #[tokio::test]
    async fn exact_partner_reads_only_the_winning_metadata_row() {
        // One dense high-degree bucket makes the partner block large enough to
        // distinguish "one LSN" from accidentally fetching all LSN rows.
        let body = write_body(1, 4_096, 64);
        let eager = EdgeSstReader::open(body.clone()).unwrap();
        let (footer, _) = EdgeFileFooter::decode(&body).unwrap();
        let partner_bytes = footer.find_kind(SECTION_PARTNERS).unwrap().length;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/point.sst");
        put_body(&store, &path, body).await;
        let reader = PagedEdgeReader::open(store, path).await.unwrap();

        let before = reader.io_stats();
        let source = id(1, 0);
        let partner = id(2, 3_500);
        let actual = reader
            .lookup_partner(&source, &partner)
            .await
            .unwrap()
            .unwrap();
        let expected = eager.lookup_partner(&source, &partner).unwrap().unwrap();
        assert_eq!(actual, expected);
        let delta = reader.io_stats().bytes_read - before.bytes_read;
        assert!(
            delta <= u64::from(EDGE_CHECKSUM_PAGE_BYTES) * 16,
            "point lookup read {delta} bytes for a {partner_bytes}-byte dense partner block"
        );
    }

    #[tokio::test]
    async fn readers_share_aligned_pages_and_ram_eviction_preserves_exactness() {
        let body = write_body(4_096, 3, 64);
        let eager = EdgeSstReader::open(body.clone()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/shared-page-cache.sst");
        let meta = put_body(&store, &path, body).await;

        let mut config = RangeCacheConfig::memory_only(2 * 1024 * 1024);
        config.max_entry_bytes = 4 * 1024;
        config.page_size_bytes = 4 * 1024;
        config.key_namespace = "paged-edge-reader-test".into();
        let cache = ImmutableRangeCache::open(config).await.unwrap();
        let target = id(1, 2_048);
        let expected = eager.lookup(&target).unwrap();

        let first = PagedEdgeReader::open_with_meta_and_cache(
            store.clone(),
            meta.clone(),
            Some(cache.clone()),
        )
        .await
        .unwrap();
        assert_eq!(first.lookup(&target).await.unwrap(), expected);
        let warmed_fetches = cache.stats().outer_fetches;
        assert!(
            warmed_fetches > 0,
            "the first reader must populate at least one aligned page"
        );

        let second = PagedEdgeReader::open_with_meta_and_cache(
            store.clone(),
            meta.clone(),
            Some(cache.clone()),
        )
        .await
        .unwrap();
        assert_eq!(second.lookup(&target).await.unwrap(), expected);
        assert_eq!(
            cache.stats().outer_fetches,
            warmed_fetches,
            "same path + ETag + aligned pages must be reused across readers"
        );

        cache.clear_memory();
        assert_eq!(cache.memory_usage_bytes(), 0);
        let third = PagedEdgeReader::open_with_meta_and_cache(store, meta, Some(cache.clone()))
            .await
            .unwrap();
        assert_eq!(third.lookup(&target).await.unwrap(), expected);
        assert!(
            cache.stats().outer_fetches > warmed_fetches,
            "evicting RAM must cause an exact authoritative refetch"
        );
    }

    #[tokio::test]
    async fn v1_0_body_uses_one_lazy_exact_fallback() {
        let modern = write_body(32, 3, 8);
        let legacy = remove_section(modern, SECTION_EDGE_ORDINALS, 0);
        let eager = EdgeSstReader::open(legacy.clone()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/legacy-v1.sst");
        put_body(&store, &path, legacy).await;
        let reader = PagedEdgeReader::open(store, path).await.unwrap();

        assert!(!reader.is_range_complete());
        assert_eq!(reader.io_stats().eager_body_reads, 0);
        let key_a = id(1, 3);
        let key_b = id(1, 29);
        assert_eq!(
            reader.lookup(&key_a).await.unwrap(),
            eager.lookup(&key_a).unwrap()
        );
        assert_eq!(reader.io_stats().eager_body_reads, 1);
        assert_eq!(
            reader.lookup(&key_b).await.unwrap(),
            eager.lookup(&key_b).unwrap()
        );
        assert_eq!(
            reader.io_stats().eager_body_reads,
            1,
            "OnceCell must retain the exact legacy reader"
        );
    }

    #[tokio::test]
    async fn v1_1_body_with_ordinals_uses_verified_full_read_fallback() {
        let modern = write_body(32, 3, 8);
        let legacy = remove_section(modern, SECTION_PAGE_CHECKSUMS, 1);
        let eager = EdgeSstReader::open(legacy.clone()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/legacy-v1-1.sst");
        put_body(&store, &path, legacy).await;
        let reader = PagedEdgeReader::open(store, path).await.unwrap();

        assert!(!reader.is_range_complete());
        assert_eq!(reader.io_stats().eager_body_reads, 0);
        let key = id(1, 9);
        assert_eq!(
            reader.lookup(&key).await.unwrap(),
            eager.lookup(&key).unwrap()
        );
        assert_eq!(reader.io_stats().eager_body_reads, 1);
    }

    #[tokio::test]
    async fn stale_meta_between_head_and_first_range_is_rejected() {
        let body_a = write_body(16, 2, 4);
        let body_b = write_body(17, 2, 4);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/replaced-before-open.sst");
        let stale_meta = put_body(&store, &path, body_a).await;
        put_body(&store, &path, body_b).await;

        let error = PagedEdgeReader::open_with_meta(store, stale_meta)
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::ObjectStore(_)),
            "conditional stale read should fail in object store, got {error:?}"
        );
    }

    /// Uncached mechanism: with no cache in front, every read hits the store
    /// and a replaced object fails its ETag pin outright.
    #[tokio::test]
    async fn replacement_after_open_fails_closed_without_cache() {
        let body_a = write_body(64, 2, 8);
        let body_b = write_body(65, 2, 8);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/replaced-after-open-uncached.sst");
        let meta = put_body(&store, &path, body_a).await;
        let reader = PagedEdgeReader::open_with_meta_and_cache(store.clone(), meta, None)
            .await
            .unwrap();
        put_body(&store, &path, body_b).await;

        let error = reader.lookup(&id(1, 10)).await.unwrap_err();
        assert!(
            matches!(error, Error::ObjectStore(_)),
            "a replaced object must fail its ETag pin, got {error:?}"
        );
    }

    /// Cached semantics: the invariant is that a pinned reader never observes
    /// bytes of another generation — not that every read hits the store. Warm
    /// ranges keep serving the pinned generation; cold ranges fail the pin
    /// instead of tearing in replacement bytes; a fresh reader pinned to the
    /// replacement sees only the replacement.
    #[tokio::test]
    async fn replacement_after_open_never_mixes_generations() {
        let body_a = write_body(64, 2, 8);
        let body_b = write_body(65, 2, 8);
        let oracle_a = EdgeSstReader::open(body_a.clone()).unwrap();
        let oracle_b = EdgeSstReader::open(body_b.clone()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/replaced-after-open.sst");
        let meta_a = put_body(&store, &path, body_a).await;

        let mut config = RangeCacheConfig::memory_only(2 * 1024 * 1024);
        config.max_entry_bytes = 4 * 1024;
        config.page_size_bytes = 4 * 1024;
        config.key_namespace = "paged-edge-reader-replacement".into();
        let cache = ImmutableRangeCache::open(config).await.unwrap();

        let reader =
            PagedEdgeReader::open_with_meta_and_cache(store.clone(), meta_a, Some(cache.clone()))
                .await
                .unwrap();
        let target = id(1, 10);
        let pinned = reader.lookup(&target).await.unwrap();
        assert_eq!(pinned, oracle_a.lookup(&target).unwrap());

        let meta_b = put_body(&store, &path, body_b).await;
        assert_eq!(
            reader.lookup(&target).await.unwrap(),
            pinned,
            "warm ranges keep serving the pinned generation, never generation B"
        );

        cache.clear_memory();
        let error = reader.lookup(&target).await.unwrap_err();
        assert!(
            matches!(error, Error::ObjectStore(_)),
            "cold ranges must fail the ETag pin instead of tearing, got {error:?}"
        );

        let fresh = PagedEdgeReader::open_with_meta_and_cache(store, meta_b, Some(cache))
            .await
            .unwrap();
        assert_eq!(
            fresh.lookup(&target).await.unwrap(),
            oracle_b.lookup(&target).unwrap(),
            "a generation-B reader sharing the cache sees only generation B"
        );
    }

    #[tokio::test]
    async fn create_only_object_without_generation_token_uses_immutable_path_pin() {
        let body = write_body(8, 1, 4);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/no-generation.sst");
        let mut meta = put_body(&store, &path, body).await;
        meta.e_tag = None;
        meta.version = None;
        let reader = PagedEdgeReader::open_with_meta(store, meta).await.unwrap();
        assert_eq!(
            reader.lookup(&id(1, 3)).await.unwrap().unwrap().lsns,
            vec![10_003]
        );
    }

    #[tokio::test]
    async fn strict_legacy_constructor_refuses_object_without_generation_token() {
        let body = write_body(8, 1, 4);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/no-generation-strict.sst");
        let mut meta = put_body(&store, &path, body).await;
        meta.e_tag = None;
        meta.version = None;
        let error = PagedEdgeReader::open_with_meta_and_cache(store, meta, None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Precondition(_)));
    }

    #[tokio::test]
    async fn malformed_selected_partner_block_is_rejected() {
        let body = write_body(8, 2, 4);
        let (footer, footer_start) = EdgeFileFooter::decode(&body).unwrap();
        let partners = footer.find_kind(SECTION_PARTNERS).unwrap();
        let mut corrupt = body.to_vec();
        // Every block starts with a one-byte degree for degree=2, followed by
        // its tag. Make the first selected tag unknown. The footer itself and
        // all absolute section ranges remain well-formed.
        corrupt[partners.offset as usize + 1] = 0xff;
        assert!(footer_start > partners.offset as usize);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/corrupt-block.sst");
        put_body(&store, &path, Bytes::from(corrupt)).await;
        let reader = PagedEdgeReader::open(store, path).await.unwrap();
        let error = reader.lookup(&id(1, 0)).await.unwrap_err();
        assert!(matches!(error, Error::Corrupted { .. }));
    }

    #[tokio::test]
    async fn corruption_in_every_ranged_topology_section_is_rejected() {
        let body = write_body(16, 2, 2);
        for kind in [
            SECTION_KEY_IDS,
            SECTION_OFFSETS,
            SECTION_EDGE_ORDINALS,
            SECTION_PARTNERS,
            SECTION_PER_EDGE_LSN,
            SECTION_PER_EDGE_TOMBSTONES,
            SECTION_FENCE_INDEX,
            SECTION_PAGE_CHECKSUMS,
        ] {
            let (footer, _) = EdgeFileFooter::decode(&body).unwrap();
            let entry = footer.find_kind(kind).unwrap();
            assert!(entry.length > 0, "test section 0x{kind:04x} is empty");
            let mut corrupt = body.to_vec();
            corrupt[entry.offset as usize] ^= 0x5a;
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let path = Path::from(format!("edge/corrupt-{kind:04x}.sst"));
            put_body(&store, &path, Bytes::from(corrupt)).await;
            let error = match PagedEdgeReader::open(store, path).await {
                Err(error) => error,
                Ok(reader) => reader.lookup(&id(1, 0)).await.unwrap_err(),
            };
            assert!(
                matches!(error, Error::Corrupted { .. }),
                "section 0x{kind:04x} returned {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn corrupt_property_page_is_rejected_before_hydration_decode() {
        let body = write_body_with_property();
        let (footer, _) = EdgeFileFooter::decode(&body).unwrap();
        let property = footer.find(SECTION_PROPERTY_STREAM, "codigo").unwrap();
        let mut corrupt = body.to_vec();
        corrupt[property.offset as usize] ^= 0x5a;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/corrupt-property.sst");
        put_body(&store, &path, Bytes::from(corrupt)).await;
        let reader = PagedEdgeReader::open(store, path).await.unwrap();
        let entry = reader
            .footer
            .find(SECTION_PROPERTY_STREAM, "codigo")
            .unwrap();
        let error = reader
            .object
            .read_section_range(entry, 0..entry.length.min(8))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Corrupted { .. }));
    }

    #[tokio::test]
    async fn footer_corruption_and_object_truncation_are_rejected() {
        let body = write_body(8, 2, 2);
        let (footer, footer_start) = EdgeFileFooter::decode(&body).unwrap();
        assert!(!footer.sections.is_empty());

        let mut corrupt_footer = body.to_vec();
        corrupt_footer[footer_start] ^= 0x80;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let footer_path = Path::from("edge/corrupt-footer.sst");
        put_body(&store, &footer_path, Bytes::from(corrupt_footer)).await;
        assert!(matches!(
            PagedEdgeReader::open(store, footer_path).await.unwrap_err(),
            Error::Corrupted { .. }
        ));

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let truncated_path = Path::from("edge/truncated.sst");
        put_body(
            &store,
            &truncated_path,
            body.slice(..body.len() - FOOTER_TRAILER_LEN),
        )
        .await;
        assert!(matches!(
            PagedEdgeReader::open(store, truncated_path)
                .await
                .unwrap_err(),
            Error::Corrupted { .. }
        ));
    }

    #[tokio::test]
    async fn malicious_wire_degree_is_checked_against_authenticated_ordinals() {
        let body = write_body(8, 2, 2);
        let (footer, _) = EdgeFileFooter::decode(&body).unwrap();
        let partners = footer.find_kind(SECTION_PARTNERS).unwrap();
        let mut corrupt = body.to_vec();
        // Keep the one-byte varint shape, but claim the maximum one-byte
        // degree. Re-root all hashes so this reaches semantic validation
        // instead of being stopped by the page checksum.
        corrupt[partners.offset as usize] = 127;
        let rerooted = reroot_after_data_mutation(corrupt);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge/malicious-degree.sst");
        put_body(&store, &path, rerooted).await;
        let reader = PagedEdgeReader::open(store, path).await.unwrap();
        let error = reader.lookup(&id(1, 0)).await.unwrap_err();
        assert!(matches!(
            error,
            Error::Corrupted { .. } | Error::Invariant(_)
        ));
    }

    /// Re-seal a hand-mutated edge body so the reader reaches the behaviour a
    /// test is actually about instead of rejecting the fixture as corrupt.
    ///
    /// The order is forced and it terminates: section digests first, then the
    /// identity binding — whose section root deliberately skips both itself and
    /// the checksum directory — and finally the directory, which covers the
    /// binding but which nothing else covers.
    fn reseal_edge_body(file: &mut [u8], sections: &mut [SectionEntry], footer_start: usize) {
        fn rehash(file: &[u8], entry: &mut SectionEntry) {
            let start = entry.offset as usize;
            entry.xxhash3_64 = xxh3_64(&file[start..start + entry.length as usize]);
        }

        for entry in sections.iter_mut() {
            if !matches!(entry.kind, SECTION_SST_BINDING | SECTION_PAGE_CHECKSUMS) {
                rehash(file, entry);
            }
        }

        if let Some(index) = sections
            .iter()
            .position(|entry| entry.kind == SECTION_SST_BINDING)
        {
            let start = sections[index].offset as usize;
            let sst_id = EdgeSstBinding::decode(&file[start..start + SST_BINDING_LEN])
                .unwrap()
                .sst_id;
            let header = file[..HEADER_LEN].to_vec();
            let rebuilt = EdgeSstBinding::for_sections(sst_id, &header, sections)
                .unwrap()
                .encode();
            file[start..start + SST_BINDING_LEN].copy_from_slice(&rebuilt);
            rehash(file, &mut sections[index]);
        }

        if let Some(index) = sections
            .iter()
            .position(|entry| entry.kind == SECTION_PAGE_CHECKSUMS)
        {
            let covered: Vec<SectionEntry> = sections
                .iter()
                .filter(|entry| entry.kind != SECTION_PAGE_CHECKSUMS)
                .cloned()
                .collect();
            let encoded = EdgePageChecksumDirectory::build(
                &file[..footer_start],
                &covered,
                EDGE_CHECKSUM_PAGE_BYTES,
            )
            .unwrap()
            .encode()
            .unwrap();
            assert_eq!(encoded.len() as u64, sections[index].length);
            let start = sections[index].offset as usize;
            file[start..start + encoded.len()].copy_from_slice(&encoded);
            rehash(file, &mut sections[index]);
        }
    }

    fn reroot_after_data_mutation(mut file: Vec<u8>) -> Bytes {
        let original = Bytes::copy_from_slice(&file);
        let (footer, footer_start) = EdgeFileFooter::decode(&original).unwrap();
        let mut sections = footer.sections.clone();
        reseal_edge_body(&mut file, &mut sections, footer_start);
        file.truncate(footer_start);
        EdgeFileFooter {
            sections,
            key_count: footer.key_count,
            edge_count: footer.edge_count,
            offsets_bits: footer.offsets_bits,
            min_key_id: footer.min_key_id,
            max_key_id: footer.max_key_id,
            min_lsn: footer.min_lsn,
            max_lsn: footer.max_lsn,
            schema_version_min: footer.schema_version_min,
            schema_version_max: footer.schema_version_max,
        }
        .encode(&mut file)
        .unwrap();
        Bytes::from(file)
    }

    fn remove_section(body: Bytes, kind: u16, format_minor: u8) -> Bytes {
        let (footer, _) = EdgeFileFooter::decode(&body).unwrap();
        let mut file = body[..HEADER_LEN].to_vec();
        file[9] = format_minor;
        let mut sections = Vec::new();
        for entry in footer
            .sections
            .iter()
            .filter(|entry| entry.kind != kind && entry.kind != SECTION_PAGE_CHECKSUMS)
        {
            let start = entry.offset as usize;
            let end = start + entry.length as usize;
            let mut rewritten = entry.clone();
            rewritten.offset = file.len() as u64;
            file.extend_from_slice(&body[start..end]);
            sections.push(rewritten);
        }
        // Synthesising a legacy body rewrites the header minor version and
        // every section offset, so neither the copied binding nor the copied
        // digests still describe what they cover.
        let footer_start = file.len();
        reseal_edge_body(&mut file, &mut sections, footer_start);
        EdgeFileFooter {
            sections,
            key_count: footer.key_count,
            edge_count: footer.edge_count,
            offsets_bits: footer.offsets_bits,
            min_key_id: footer.min_key_id,
            max_key_id: footer.max_key_id,
            min_lsn: footer.min_lsn,
            max_lsn: footer.max_lsn,
            schema_version_min: footer.schema_version_min,
            schema_version_max: footer.schema_version_max,
        }
        .encode(&mut file)
        .unwrap();
        Bytes::from(file)
    }
}
