//! Range-readable immutable B+tree sidecars.
//!
//! The original property sidecars were bincode-encoded `BTreeMap`s. A point
//! probe consequently fetched and decoded every key in the map. This format
//! stores a shallow B+tree in fixed-size pages so a cold probe reads the
//! 64-byte header plus only the internal/leaf pages on its search path.
//! Unique and node-locator values are inline; equality postings and exact
//! node/edge records live in a value region and are fetched only for matching
//! keys. Corpus-sized exact-record value regions are spooled to disk while
//! their fixed-page search trees are built.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, Write};
use std::ops::Range;
#[cfg(test)]
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
#[cfg(test)]
use object_store::path::Path;
#[cfg(test)]
use object_store::{GetOptions, GetRange, ObjectStore};

use crate::error::{Error, Result};
use crate::range_cache::PinnedObjectRangeSource;

const MAGIC_V1: &[u8; 8] = b"NAMIPG01";
const MAGIC_V2: &[u8; 8] = b"NAMIPG02";
const HEADER_SIZE: usize = 64;
const HEADER_CHECKSUM_OFFSET: usize = 44;
const PAGE_SIZE: usize = 4096;
const PAGE_HEADER_SIZE: usize = 16;
const NO_PAGE: u32 = u32::MAX;

const PAGE_LEAF: u8 = 0;
const PAGE_INTERNAL: u8 = 1;
const PAGE_PAYLOAD_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;
const EXTERNAL_LEAF_OVERHEAD_V1: usize = 2 + 8 + 4;
const EXTERNAL_LEAF_OVERHEAD_V2: usize = EXTERNAL_LEAF_OVERHEAD_V1 + 4;
const EQUALITY_LEAF_OVERHEAD: usize = EXTERNAL_LEAF_OVERHEAD_V2;
const SEPARATOR_SPOOL_HEADER_BYTES: usize = 12;
const LABEL_FOOTER_MAGIC: &[u8; 8] = b"NAMILF02";
const LABEL_FOOTER_BYTES: usize = 72;
const LABEL_FOOTER_CHECKSUM_OFFSET: usize = 68;
type ProbeMatch = (Vec<u8>, Option<Vec<u8>>, Option<(u64, u32, Option<u32>)>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PagedFormat {
    V1,
    V2,
}

/// Logical payload carried by a paged sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PagedIndexKind {
    Unique = 1,
    Equality = 2,
    NodeLocator = 3,
    EdgePoint = 4,
    /// Exact binary node records appended after a compatible
    /// [`Self::NodeLocator`] body.
    NodeRecord = 5,
    /// Composite `(LabelId big-endian, NodeId) -> ()` membership entries.
    ///
    /// Keeping one inline key per posting makes both prefix pagination and
    /// exact membership range-readable; a giant label never becomes one
    /// giant external posting that must be hydrated as a unit.
    LabelMembership = 6,
}

impl PagedIndexKind {
    fn from_byte(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Unique),
            2 => Ok(Self::Equality),
            3 => Ok(Self::NodeLocator),
            4 => Ok(Self::EdgePoint),
            5 => Ok(Self::NodeRecord),
            6 => Ok(Self::LabelMembership),
            _ => Err(Error::invariant(format!(
                "paged index has unknown kind {value}"
            ))),
        }
    }

    fn external_values(self) -> bool {
        matches!(self, Self::Equality | Self::EdgePoint | Self::NodeRecord)
    }
}

/// Observable amount of index work performed by a probe.
///
/// Tests and diagnostics use this to assert that a point lookup did not
/// silently turn into a full sidecar scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedProbeStats {
    pub index_entries: u64,
    pub pages_read: usize,
    pub leaf_entries_examined: usize,
    pub bytes_read: usize,
    /// Logical bytes of every matched external value before an optional read
    /// cap (posting cardinality can be derived without fetching the posting).
    pub matched_value_bytes: u64,
    pub values_truncated: bool,
}

#[derive(Debug, Clone)]
struct Header {
    format: PagedFormat,
    kind: PagedIndexKind,
    root_page: u32,
    page_count: u32,
    leaf_count: u32,
    entry_count: u64,
    values_offset: u64,
}

impl Header {
    fn encode(&self) -> Bytes {
        let mut out = BytesMut::with_capacity(HEADER_SIZE);
        out.extend_from_slice(match self.format {
            PagedFormat::V1 => MAGIC_V1,
            PagedFormat::V2 => MAGIC_V2,
        });
        out.put_u8(self.kind as u8);
        out.extend_from_slice(&[0; 3]);
        out.put_u32_le(PAGE_SIZE as u32);
        out.put_u32_le(self.root_page);
        out.put_u32_le(self.page_count);
        out.put_u32_le(self.leaf_count);
        out.put_u64_le(self.entry_count);
        out.put_u64_le(self.values_offset);
        out.resize(HEADER_SIZE, 0);
        if self.format == PagedFormat::V2 {
            let checksum = checksum_around_slot(&out, HEADER_CHECKSUM_OFFSET);
            out[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 4]
                .copy_from_slice(&checksum.to_le_bytes());
        }
        out.freeze()
    }

    fn decode(bytes: &[u8], expected: PagedIndexKind) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(Error::invariant("invalid paged-index header"));
        }
        // Callers that already hold a complete sidecar may pass the whole
        // body. The checksum envelope covers exactly the fixed header, not
        // pages or external values (those carry their own CRCs).
        let bytes = &bytes[..HEADER_SIZE];
        let format = match &bytes[..8] {
            magic if magic == MAGIC_V1 => PagedFormat::V1,
            magic if magic == MAGIC_V2 => PagedFormat::V2,
            _ => return Err(Error::invariant("invalid paged-index header")),
        };
        if format == PagedFormat::V2 {
            let expected_checksum = read_u32(bytes, HEADER_CHECKSUM_OFFSET)?;
            let actual_checksum = checksum_around_slot(bytes, HEADER_CHECKSUM_OFFSET);
            if expected_checksum != actual_checksum {
                return Err(Error::invariant("paged-index header checksum mismatch"));
            }
        }
        let kind = PagedIndexKind::from_byte(bytes[8])?;
        if kind != expected {
            return Err(Error::invariant(format!(
                "paged-index kind mismatch: expected {expected:?}, got {kind:?}"
            )));
        }
        let page_size = read_u32(bytes, 12)? as usize;
        if page_size != PAGE_SIZE {
            return Err(Error::invariant(format!(
                "unsupported paged-index page size {page_size}"
            )));
        }
        let header = Self {
            format,
            kind,
            root_page: read_u32(bytes, 16)?,
            page_count: read_u32(bytes, 20)?,
            leaf_count: read_u32(bytes, 24)?,
            entry_count: read_u64(bytes, 28)?,
            values_offset: read_u64(bytes, 36)?,
        };
        if header.page_count == 0
            || header.root_page >= header.page_count
            || header.leaf_count == 0
            || header.leaf_count > header.page_count
            || header.values_offset
                != (HEADER_SIZE as u64).saturating_add(header.page_count as u64 * PAGE_SIZE as u64)
        {
            return Err(Error::invariant("invalid paged-index header bounds"));
        }
        Ok(header)
    }

    fn require_authoritative_integrity(&self) -> Result<()> {
        if self.format != PagedFormat::V2 {
            return Err(Error::invariant(
                "paged-index V1 has no complete integrity envelope; use the authoritative fallback",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LabelFooter {
    page_region_bytes: u64,
    entry_count: u64,
    label_count: u64,
    header_checksum: u32,
    counts_checksum: u32,
    page_region_checksum: u32,
    sst_id: [u8; 16],
}

fn encode_label_footer(
    page_region_bytes: u64,
    entry_count: u64,
    label_count: u64,
    header_checksum: u32,
    counts_checksum: u32,
    page_region_checksum: u32,
    sst_id: [u8; 16],
) -> [u8; LABEL_FOOTER_BYTES] {
    let mut out = [0_u8; LABEL_FOOTER_BYTES];
    out[..8].copy_from_slice(LABEL_FOOTER_MAGIC);
    out[8..10].copy_from_slice(&2_u16.to_le_bytes());
    out[10] = PagedIndexKind::LabelMembership as u8;
    out[12..20].copy_from_slice(&page_region_bytes.to_le_bytes());
    out[20..28].copy_from_slice(&entry_count.to_le_bytes());
    out[28..36].copy_from_slice(&label_count.to_le_bytes());
    out[36..40].copy_from_slice(&header_checksum.to_le_bytes());
    out[40..44].copy_from_slice(&counts_checksum.to_le_bytes());
    out[44..48].copy_from_slice(&page_region_checksum.to_le_bytes());
    out[48..64].copy_from_slice(&sst_id);
    let checksum = checksum_around_slot(&out, LABEL_FOOTER_CHECKSUM_OFFSET);
    out[LABEL_FOOTER_CHECKSUM_OFFSET..LABEL_FOOTER_CHECKSUM_OFFSET + 4]
        .copy_from_slice(&checksum.to_le_bytes());
    out
}

fn decode_label_footer(bytes: &[u8]) -> Result<LabelFooter> {
    if bytes.len() != LABEL_FOOTER_BYTES
        || &bytes[..8] != LABEL_FOOTER_MAGIC
        || read_u16(bytes, 8)? != 2
        || bytes[10] != PagedIndexKind::LabelMembership as u8
        || bytes[11] != 0
        || bytes[64..68].iter().any(|byte| *byte != 0)
    {
        return Err(Error::invariant("invalid label-index V2 footer"));
    }
    let expected = read_u32(bytes, LABEL_FOOTER_CHECKSUM_OFFSET)?;
    if checksum_around_slot(bytes, LABEL_FOOTER_CHECKSUM_OFFSET) != expected {
        return Err(Error::invariant("label-index V2 footer checksum mismatch"));
    }
    Ok(LabelFooter {
        page_region_bytes: read_u64(bytes, 12)?,
        entry_count: read_u64(bytes, 20)?,
        label_count: read_u64(bytes, 28)?,
        header_checksum: read_u32(bytes, 36)?,
        counts_checksum: read_u32(bytes, 40)?,
        page_region_checksum: read_u32(bytes, 44)?,
        sst_id: bytes[48..64]
            .try_into()
            .map_err(|_| Error::invariant("invalid label-index SST binding"))?,
    })
}

fn label_counts_checksum(counts: &[(u32, u64)]) -> u32 {
    let mut checksum = crc32fast::Hasher::new();
    for (label, count) in counts {
        checksum.update(&label.to_be_bytes());
        checksum.update(&count.to_le_bytes());
    }
    checksum.finalize()
}

/// Bind every label page checksum to the exact SST UUID and to a digest of
/// the complete pre-binding page region. The latter is persisted in the
/// footer; an object-native reader can validate any fetched page locally
/// without downloading all sibling pages.
fn bind_label_page_region(
    pages: &mut std::fs::File,
    page_region_bytes: u64,
    sst_id: [u8; 16],
) -> Result<u32> {
    let page_bytes = page_region_bytes
        .checked_sub(HEADER_SIZE as u64)
        .ok_or_else(|| Error::invariant("label page region is shorter than its header"))?;
    if page_bytes == 0 || page_bytes % PAGE_SIZE as u64 != 0 {
        return Err(Error::invariant("label page region is not page aligned"));
    }
    let page_count = page_bytes / PAGE_SIZE as u64;
    let mut region_checksum = crc32fast::Hasher::new();
    pages.seek(std::io::SeekFrom::Start(HEADER_SIZE as u64))?;
    let mut page = vec![0_u8; PAGE_SIZE];
    for _ in 0..page_count {
        pages.read_exact(&mut page)?;
        region_checksum.update(&page);
    }
    let region_checksum = region_checksum.finalize();

    for page_id in 0..page_count {
        let offset = (HEADER_SIZE as u64)
            .checked_add(
                page_id
                    .checked_mul(PAGE_SIZE as u64)
                    .ok_or_else(|| Error::invariant("label page offset exceeds u64"))?,
            )
            .ok_or_else(|| Error::invariant("label page offset exceeds u64"))?;
        pages.seek(std::io::SeekFrom::Start(offset))?;
        pages.read_exact(&mut page)?;
        let checksum = label_page_checksum(&page, sst_id, region_checksum, page_id);
        page[12..16].copy_from_slice(&checksum.to_le_bytes());
        pages.seek(std::io::SeekFrom::Start(offset))?;
        pages.write_all(&page)?;
    }
    Ok(region_checksum)
}

fn label_page_checksum(
    page: &[u8],
    sst_id: [u8; 16],
    page_region_checksum: u32,
    page_id: u64,
) -> u32 {
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&page[..12]);
    checksum.update(&page[16..]);
    checksum.update(&sst_id);
    checksum.update(&page_region_checksum.to_le_bytes());
    checksum.update(&page_id.to_le_bytes());
    checksum.finalize()
}

/// Build a range-readable unique `String -> NodeId` sidecar.
#[cfg(test)]
pub fn build_unique(index: &BTreeMap<String, [u8; 16]>) -> Result<Bytes> {
    let mut builder = PagedIndexBuilder::new(PagedIndexKind::Unique);
    for (key, id) in index {
        builder.push_inline(key.as_bytes(), id)?;
    }
    builder.finish()
}

/// Build a range-readable equality `encoded scalar -> [NodeId]` sidecar.
#[cfg(test)]
pub fn build_equality(index: &BTreeMap<String, Vec<[u8; 16]>>) -> Result<Bytes> {
    let mut builder = PagedIndexBuilder::new(PagedIndexKind::Equality);
    for (key, ids) in index {
        builder.push_posting(key.as_bytes(), ids)?;
    }
    builder.finish()
}

/// Build a range-readable `NodeId -> physical row ordinal` locator.
///
/// Kept as the compatibility builder for standalone `.nloc` tooling and
/// wire-format tests. Production node SSTs use
/// [`NodeLocatorRecordBuilder::finish_upload`].
#[allow(dead_code)]
pub fn build_node_locator(ids: impl IntoIterator<Item = [u8; 16]>) -> Result<Bytes> {
    let mut builder = PagedIndexBuilder::new(PagedIndexKind::NodeLocator);
    for (row, id) in ids.into_iter().enumerate() {
        let ordinal = u64::try_from(row)
            .map_err(|_| Error::invariant("node locator exceeds u64 row ordinals"))?
            .to_le_bytes();
        builder.push_inline(&id, &ordinal)?;
    }
    builder.finish()
}

/// Streaming builder for a combined node-locator + exact-record sidecar.
///
/// The resulting body is deliberately concatenated in this order:
///
/// ```text
/// [ordinary NodeLocator V2 body][NodeRecord V2 body]
/// ```
///
/// A node locator has only inline values, so its `values_offset` is also its
/// exact body length and therefore the start of the appended record index.
/// Existing readers can keep probing the prefix with [`probe_node_locator`]
/// and ignore the appended bytes. New readers use [`probe_node_records`] to
/// range-read only the requested binary records, avoiding wide Parquet page
/// hydration for read-modify-write updates.
#[derive(Debug)]
pub struct NodeLocatorRecordBuilder {
    locator: PagedIndexBuilder,
    records: PagedIndexBuilder,
    next_ordinal: u64,
}

/// Bounded-memory upload product for one combined `.nloc2` sidecar.
///
/// The two B+tree page regions remain in memory (tens of bytes per node), but
/// the potentially multi-gigabyte exact-record value region is written to an
/// anonymous temporary file while rows stream through flush/compaction. The
/// upload path consumes that file in fixed multipart chunks and dropping this
/// value removes it automatically.
#[derive(Debug)]
pub(crate) struct NodeLocatorRecordUpload {
    prefix: Vec<Bytes>,
    values: SpooledValueRegion,
    size_bytes: u64,
    entry_count: u64,
}

impl NodeLocatorRecordUpload {
    pub(crate) fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub(crate) fn entry_count(&self) -> u64 {
        self.entry_count
    }

    pub(crate) fn into_parts(self) -> (Vec<Bytes>, Option<std::fs::File>, u64) {
        (self.prefix, self.values.file, self.values.len)
    }
}

#[derive(Debug)]
struct SpooledValueRegion {
    file: Option<std::fs::File>,
    len: u64,
}

impl NodeLocatorRecordBuilder {
    pub fn new() -> Self {
        Self {
            locator: PagedIndexBuilder::new(PagedIndexKind::NodeLocator),
            records: PagedIndexBuilder::new(PagedIndexKind::NodeRecord),
            next_ordinal: 0,
        }
    }

    /// Append one node in physical SST row order.
    ///
    /// `record` is opaque to the sidecar. Callers may choose the storage
    /// engine's compact binary record encoding without coupling this index
    /// module to node schema or WAL types.
    pub fn push(&mut self, id: &[u8; 16], record: &[u8]) -> Result<()> {
        let ordinal = self.next_ordinal.to_le_bytes();
        // Validate and append the variable-width value first. Once that
        // succeeds, the fixed eight-byte locator append for the same key
        // cannot fail for a record-specific size reason and both trees stay
        // aligned.
        self.records.push_external(id, record)?;
        self.locator.push_inline(id, &ordinal)?;
        self.next_ordinal = self
            .next_ordinal
            .checked_add(1)
            .ok_or_else(|| Error::invariant("node locator exceeds u64 row ordinals"))?;
        Ok(())
    }

    /// Flattened compatibility/test form. Production upload uses
    /// [`Self::finish_upload`] so the external value region stays on disk.
    #[allow(dead_code)]
    pub fn finish(self) -> Result<Bytes> {
        let chunks = self.finish_chunks()?;
        let len = chunks
            .iter()
            .fold(0usize, |total, chunk| total.saturating_add(chunk.len()));
        let mut out = BytesMut::with_capacity(len);
        for chunk in chunks {
            out.extend_from_slice(&chunk);
        }
        Ok(out.freeze())
    }

    /// Finalise as bounded-memory upload parts in exact wire order.
    ///
    /// Production flush/compaction streams the returned anonymous spool file
    /// into multipart PUTs. This avoids retaining (or duplicating) the complete
    /// exact-record corpus while Parquet and search-index builders are live.
    pub(crate) fn finish_upload(self) -> Result<NodeLocatorRecordUpload> {
        if self.locator.entry_count != self.records.entry_count
            || self.locator.entry_count != self.next_ordinal
        {
            return Err(Error::invariant(
                "node locator and exact-record builders are misaligned",
            ));
        }
        let (locator_pages, locator_values) = self.locator.finish_parts()?;
        let (record_pages, record_values) = self.records.finish_spooled_parts()?;
        debug_assert!(locator_values.is_empty());
        let prefix: Vec<Bytes> = [locator_pages, locator_values, record_pages]
            .into_iter()
            .filter(|chunk| !chunk.is_empty())
            .collect();
        let prefix_bytes = prefix
            .iter()
            .try_fold(0_u64, |total, chunk| total.checked_add(chunk.len() as u64));
        let size_bytes = prefix_bytes
            .and_then(|total| total.checked_add(record_values.len))
            .ok_or_else(|| Error::invariant("node locator sidecar size exceeds u64"))?;
        Ok(NodeLocatorRecordUpload {
            prefix,
            values: record_values,
            size_bytes,
            entry_count: self.next_ordinal,
        })
    }

    /// In-memory compatibility/test form. Production must use
    /// [`Self::finish_upload`] so the exact-record corpus remains spooled.
    pub fn finish_chunks(self) -> Result<Vec<Bytes>> {
        let upload = self.finish_upload()?;
        let (mut chunks, file, _) = upload.into_parts();
        if let Some(mut file) = file {
            let mut values = Vec::new();
            file.read_to_end(&mut values)?;
            if !values.is_empty() {
                chunks.push(Bytes::from(values));
            }
        }
        Ok(chunks)
    }
}

impl Default for NodeLocatorRecordBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming range-readable `(key_id, partner_id) -> edge point value`
/// sidecar builder.
///
/// Edge writers already receive rows in this exact composite-key order, so
/// the B+tree can be emitted alongside the CSR without retaining another
/// graph-sized map. Values are external/variable-width because they carry the
/// row LSN, tombstone bit and encoded property map used by exact MERGE.
#[derive(Debug)]
pub struct EdgePointIndexBuilder {
    inner: SpooledPagedIndexBuilder,
}

impl EdgePointIndexBuilder {
    pub fn new() -> Self {
        Self {
            inner: SpooledPagedIndexBuilder::new(PagedIndexKind::EdgePoint),
        }
    }

    pub fn push(&mut self, key_id: &[u8; 16], partner_id: &[u8; 16], value: &[u8]) -> Result<()> {
        let mut key = [0u8; 32];
        key[..16].copy_from_slice(key_id);
        key[16..].copy_from_slice(partner_id);
        self.inner.push_external(&key, value)
    }

    /// Flattened compatibility/test form. Production upload uses
    /// [`Self::finish_upload`].
    #[cfg(test)]
    pub fn finish(self) -> Result<Bytes> {
        self.finish_upload()?.into_bytes()
    }

    /// Finalise the byte-identical V2 body as independent page/value spools.
    pub(crate) fn finish_upload(self) -> Result<EdgePointIndexUpload> {
        self.inner.finish()
    }

    #[cfg(test)]
    pub(crate) fn resident_bound_bytes(&self) -> usize {
        self.inner.resident_bound_bytes()
    }
}

impl Default for EdgePointIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Bounded-memory upload product for one `.epidx` sidecar.
///
/// Both the B+tree page region and exact-value region stay in independent
/// anonymous spool files through multipart completion/cancellation. Keeping
/// them separate avoids an extra corpus-sized local copy merely to concatenate
/// the final wire body.
#[derive(Debug)]
pub(crate) struct SpooledPagedIndexUpload {
    pages: std::fs::File,
    page_region_bytes: u64,
    values: SpooledValueRegion,
    size_bytes: u64,
    entry_count: u64,
}

pub(crate) type EdgePointIndexUpload = SpooledPagedIndexUpload;

impl SpooledPagedIndexUpload {
    pub(crate) fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub(crate) fn entry_count(&self) -> u64 {
        self.entry_count
    }

    #[cfg(test)]
    pub(crate) fn spooled_page_bytes(&self) -> u64 {
        self.pages
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn spooled_value_bytes(&self) -> u64 {
        self.values.len
    }

    pub(crate) fn into_files(self) -> (Vec<std::fs::File>, u64) {
        debug_assert_eq!(
            self.pages.metadata().map(|meta| meta.len()).ok(),
            Some(self.size_bytes.saturating_sub(self.values.len))
        );
        let mut files = Vec::with_capacity(2);
        files.push(self.pages);
        if let Some(values) = self.values.file {
            files.push(values);
        }
        (files, self.size_bytes)
    }

    #[cfg(test)]
    pub(crate) fn into_bytes(self) -> Result<Bytes> {
        let expected_len = usize::try_from(self.size_bytes)
            .map_err(|_| Error::invariant("paged sidecar exceeds addressable memory"))?;
        let (files, _) = self.into_files();
        let mut body = BytesMut::with_capacity(expected_len);
        for mut file in files {
            file.rewind()?;
            let mut chunk = [0_u8; 64 * 1024];
            loop {
                let read = file.read(&mut chunk)?;
                if read == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..read]);
            }
        }
        if body.len() != expected_len {
            return Err(Error::invariant(
                "paged upload parts disagree with descriptor size",
            ));
        }
        Ok(body.freeze())
    }

    /// Append the fixed LabelV2 footer that binds the tree header to exact
    /// per-label manifest counts. Pages remain individually checksummed and
    /// range-readable; opening a label index needs only header + footer.
    pub(crate) fn bind_label_counts(
        mut self,
        counts: &[(u32, u64)],
        sst_id: [u8; 16],
    ) -> Result<Self> {
        if self.values.len != 0 {
            return Err(Error::invariant(
                "label membership index unexpectedly has external values",
            ));
        }
        let label_count = counts.len() as u64;
        let posting_count = counts.iter().try_fold(0_u64, |total, (_, count)| {
            total
                .checked_add(*count)
                .ok_or_else(|| Error::invariant("label posting count exceeds u64"))
        })?;
        if posting_count != self.entry_count {
            return Err(Error::invariant(
                "label membership tree disagrees with per-label posting counts",
            ));
        }
        let mut header = [0_u8; HEADER_SIZE];
        self.pages.rewind()?;
        self.pages.read_exact(&mut header)?;
        Header::decode(&header, PagedIndexKind::LabelMembership)?
            .require_authoritative_integrity()?;
        let page_region_checksum =
            bind_label_page_region(&mut self.pages, self.page_region_bytes, sst_id)?;
        let footer = encode_label_footer(
            self.page_region_bytes,
            self.entry_count,
            label_count,
            read_u32(&header, HEADER_CHECKSUM_OFFSET)?,
            label_counts_checksum(counts),
            page_region_checksum,
            sst_id,
        );
        self.pages.seek(std::io::SeekFrom::End(0))?;
        self.pages.write_all(&footer)?;
        self.pages.sync_data()?;
        self.pages.rewind()?;
        self.size_bytes = self
            .size_bytes
            .checked_add(LABEL_FOOTER_BYTES as u64)
            .ok_or_else(|| Error::invariant("label sidecar size exceeds u64"))?;
        Ok(self)
    }
}

/// Public-to-storage sorted-input builder for direct PagedV2 sidecars.
///
/// Callers perform their external sort and feed strictly ascending keys. Both
/// leaves and internal levels remain disk-backed.
#[derive(Debug)]
pub(crate) struct SortedSpooledPagedIndexBuilder {
    inner: SpooledPagedIndexBuilder,
}

impl SortedSpooledPagedIndexBuilder {
    pub(crate) fn unique() -> Self {
        Self {
            inner: SpooledPagedIndexBuilder::new(PagedIndexKind::Unique),
        }
    }

    pub(crate) fn equality() -> Self {
        Self {
            inner: SpooledPagedIndexBuilder::new(PagedIndexKind::Equality),
        }
    }

    pub(crate) fn label_membership() -> Self {
        Self {
            inner: SpooledPagedIndexBuilder::new(PagedIndexKind::LabelMembership),
        }
    }

    pub(crate) fn push_inline(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.push_inline(key, value)
    }

    #[cfg(test)]
    pub(crate) fn push_external(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.push_external(key, value)
    }

    pub(crate) fn push_external_file(
        &mut self,
        key: &[u8],
        file: &mut std::fs::File,
        len: u64,
        checksum: u32,
    ) -> Result<()> {
        self.inner.push_external_file(key, file, len, checksum)
    }

    /// True when an input could not be represented as a paged sidecar, so the
    /// caller must publish the legacy body alone and mark the paged build
    /// unsupported rather than uploading a partial accelerator.
    pub(crate) fn declined(&self) -> bool {
        self.inner.declined
    }

    pub(crate) fn finish(self) -> Result<SpooledPagedIndexUpload> {
        self.inner.finish()
    }

    #[cfg(test)]
    pub(crate) fn resident_bound_bytes(&self) -> usize {
        self.inner.resident_bound_bytes()
    }
}

/// Disk-backed fixed-page builder for inline or external-value indexes.
///
/// Leaves are appended directly to `pages`; their `(max_key,page_id)`
/// separators go to a checksummed scratch run. Finalisation repeatedly reads
/// one separator run and writes the next internal level plus a new run. At no
/// point does memory grow with entries, leaves, page bytes, or value bytes.
#[derive(Debug)]
struct SpooledPagedIndexBuilder {
    format: PagedFormat,
    kind: PagedIndexKind,
    pages: Option<std::fs::File>,
    page_bytes: u64,
    leaf_separators: Option<SeparatorRun>,
    values: SpooledValueRegion,
    payload: BytesMut,
    current_count: u32,
    entry_count: u64,
    leaf_count: u32,
    last_key: Option<Vec<u8>>,
    /// Set when a record cannot be represented in this format at all, which
    /// today means a single leaf entry wider than one page. The paged sidecar
    /// is an optional accelerator, so an input it cannot express must leave the
    /// caller free to publish the legacy body alone. Failing the build instead
    /// would abort the whole flush over one oversized key.
    declined: bool,
}

#[derive(Debug)]
struct SeparatorRun {
    file: std::fs::File,
    count: u64,
    max_key_len: usize,
}

impl SpooledPagedIndexBuilder {
    fn new(kind: PagedIndexKind) -> Self {
        Self {
            format: PagedFormat::V2,
            kind,
            pages: None,
            page_bytes: 0,
            leaf_separators: None,
            values: SpooledValueRegion { file: None, len: 0 },
            payload: BytesMut::with_capacity(PAGE_PAYLOAD_SIZE),
            current_count: 0,
            entry_count: 0,
            leaf_count: 0,
            last_key: None,
            declined: false,
        }
    }

    fn push_inline(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if self.kind.external_values() {
            return Err(Error::invariant(
                "external paged index received an inline value",
            ));
        }
        let value_len = u16::try_from(value.len())
            .map_err(|_| Error::invariant("inline paged-index value too large"))?;
        let record_len = 2usize
            .checked_add(key.len())
            .and_then(|bytes| bytes.checked_add(2))
            .and_then(|bytes| bytes.checked_add(value.len()))
            .ok_or_else(|| Error::invariant("paged-index leaf entry size exceeds usize"))?;
        let Some(key_len) = self.prepare_entry(key, record_len)? else {
            return Ok(());
        };
        self.payload.put_u16_le(key_len);
        self.payload.extend_from_slice(key);
        self.payload.put_u16_le(value_len);
        self.payload.extend_from_slice(value);
        Ok(())
    }

    fn push_external(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if !self.kind.external_values() {
            return Err(Error::invariant(
                "inline paged index received an external value",
            ));
        }
        let value_len = u32::try_from(value.len())
            .map_err(|_| Error::invariant("paged-index value exceeds 4 GiB"))?;
        let record_len = EQUALITY_LEAF_OVERHEAD
            .checked_add(key.len())
            .ok_or_else(|| Error::invariant("paged-index leaf entry size exceeds usize"))?;
        let Some(key_len) = self.prepare_entry(key, record_len)? else {
            return Ok(());
        };

        self.payload.put_u16_le(key_len);
        self.payload.extend_from_slice(key);
        self.payload.put_u64_le(self.values.len);
        self.payload.put_u32_le(value_len);
        self.payload.put_u32_le(crc32fast::hash(value));
        append_spooled_value(&mut self.values, value)?;

        Ok(())
    }

    fn push_external_file(
        &mut self,
        key: &[u8],
        file: &mut std::fs::File,
        len: u64,
        checksum: u32,
    ) -> Result<()> {
        if !self.kind.external_values() {
            return Err(Error::invariant(
                "inline paged index received an external value",
            ));
        }
        let value_len =
            u32::try_from(len).map_err(|_| Error::invariant("paged-index value exceeds 4 GiB"))?;
        let record_len = EQUALITY_LEAF_OVERHEAD
            .checked_add(key.len())
            .ok_or_else(|| Error::invariant("paged-index leaf entry size exceeds usize"))?;
        let Some(key_len) = self.prepare_entry(key, record_len)? else {
            return Ok(());
        };
        self.payload.put_u16_le(key_len);
        self.payload.extend_from_slice(key);
        self.payload.put_u64_le(self.values.len);
        self.payload.put_u32_le(value_len);
        self.payload.put_u32_le(checksum);

        file.rewind()?;
        let values = match &mut self.values.file {
            Some(values) => values,
            None => self.values.file.insert(create_spool_file()?),
        };
        let copied = std::io::copy(file, values)?;
        if copied != len {
            return Err(Error::invariant(
                "paged-index external scratch length changed",
            ));
        }
        self.values.len = self
            .values
            .len
            .checked_add(len)
            .ok_or_else(|| Error::invariant("paged-index value region exceeds u64"))?;
        Ok(())
    }

    /// `None` means the record is not representable, which marks the whole
    /// build declined; the caller must then skip writing that record.
    fn prepare_entry(&mut self, key: &[u8], record_len: usize) -> Result<Option<u16>> {
        let key_len = u16::try_from(key.len())
            .map_err(|_| Error::invariant("paged-index key exceeds 65535 bytes"))?;
        if self
            .last_key
            .as_deref()
            .is_some_and(|previous| previous >= key)
        {
            return Err(Error::invariant(
                "paged-index input keys must be strictly ascending",
            ));
        }
        if record_len > PAGE_PAYLOAD_SIZE {
            // Not expressible in this format. Record the decline and keep
            // accepting input so the caller finishes its single streaming
            // pass; the half-built body simply must never be published.
            self.declined = true;
            return Ok(None);
        }
        if self.current_count > 0
            && self.payload.len().saturating_add(record_len) > PAGE_PAYLOAD_SIZE
        {
            self.flush_leaf()?;
        }
        self.current_count = self
            .current_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("paged-index leaf entry count overflow"))?;
        self.entry_count = self
            .entry_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("paged-index entry count overflow"))?;
        match &mut self.last_key {
            Some(last_key) => {
                last_key.clear();
                last_key.extend_from_slice(key);
            }
            None => self.last_key = Some(key.to_vec()),
        }
        Ok(Some(key_len))
    }

    fn flush_leaf(&mut self) -> Result<()> {
        if self.current_count == 0 {
            return Ok(());
        }
        let next = self
            .leaf_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("paged-index leaf count exceeds u32"))?;
        let payload = std::mem::replace(
            &mut self.payload,
            BytesMut::with_capacity(PAGE_PAYLOAD_SIZE),
        )
        .freeze();
        let page = encode_page(self.format, PAGE_LEAF, self.current_count, next, payload)?;
        let page_id = self.append_page(&page)?;
        if page_id != self.leaf_count {
            return Err(Error::invariant(
                "spooled paged-index leaf page ids are not contiguous",
            ));
        }
        let max_key = self
            .last_key
            .as_deref()
            .ok_or_else(|| Error::invariant("non-empty leaf has no maximum key"))?
            .to_vec();
        write_separator(self.leaf_separator_run()?, &max_key, page_id)?;
        self.leaf_count = next;
        self.current_count = 0;
        Ok(())
    }

    fn finish(mut self) -> Result<SpooledPagedIndexUpload> {
        self.flush_leaf()?;
        if self.leaf_count == 0 {
            let page = encode_page(self.format, PAGE_LEAF, 0, NO_PAGE, Bytes::new())?;
            let page_id = self.append_page(&page)?;
            write_separator(self.leaf_separator_run()?, &[], page_id)?;
            self.leaf_count = 1;
        } else {
            self.patch_last_leaf()?;
        }

        let mut level = self
            .leaf_separators
            .take()
            .ok_or_else(|| Error::invariant("spooled paged-index has no leaf separators"))?;
        finish_separator_run(&mut level)?;
        while level.count > 1 {
            level = self.build_internal_level(level)?;
        }
        let root_page = read_single_separator(&mut level)?.1;

        let page_region_bytes = self
            .page_bytes
            .checked_sub(HEADER_SIZE as u64)
            .ok_or_else(|| Error::invariant("spooled paged-index page region is truncated"))?;
        if page_region_bytes == 0 || page_region_bytes % PAGE_SIZE as u64 != 0 {
            return Err(Error::invariant(
                "spooled paged-index page region is not page aligned",
            ));
        }
        let page_count_u64 = page_region_bytes / PAGE_SIZE as u64;
        let page_count = u32::try_from(page_count_u64)
            .map_err(|_| Error::invariant("paged-index page count exceeds u32"))?;
        let header = Header {
            format: self.format,
            kind: self.kind,
            root_page,
            page_count,
            leaf_count: self.leaf_count,
            entry_count: self.entry_count,
            values_offset: self.page_bytes,
        };
        let mut pages = self
            .pages
            .take()
            .ok_or_else(|| Error::invariant("spooled paged-index has no page file"))?;
        pages.rewind()?;
        pages.write_all(&header.encode())?;
        if pages.metadata()?.len() != self.page_bytes {
            return Err(Error::invariant(
                "paged-index page spool length changed during finalisation",
            ));
        }
        pages.sync_data()?;
        pages.rewind()?;
        finish_spooled_value_region(&mut self.values)?;

        let size_bytes = self
            .page_bytes
            .checked_add(self.values.len)
            .ok_or_else(|| Error::invariant("paged sidecar size exceeds u64"))?;
        Ok(SpooledPagedIndexUpload {
            pages,
            page_region_bytes: self.page_bytes,
            values: self.values,
            size_bytes,
            entry_count: self.entry_count,
        })
    }

    fn append_page(&mut self, page: &[u8]) -> Result<u32> {
        if page.len() != PAGE_SIZE {
            return Err(Error::invariant(
                "spooled paged-index received a non-page-sized block",
            ));
        }
        self.page_file()?;
        let page_id_u64 = self
            .page_bytes
            .checked_sub(HEADER_SIZE as u64)
            .ok_or_else(|| Error::invariant("spooled paged-index header is missing"))?
            / PAGE_SIZE as u64;
        let page_id = u32::try_from(page_id_u64)
            .map_err(|_| Error::invariant("paged-index page count exceeds u32"))?;
        let pages = self
            .pages
            .as_mut()
            .expect("page_file initialized the page spool");
        pages.seek(std::io::SeekFrom::End(0))?;
        pages.write_all(page)?;
        self.page_bytes = self
            .page_bytes
            .checked_add(PAGE_SIZE as u64)
            .ok_or_else(|| Error::invariant("paged-index page bytes exceed u64"))?;
        Ok(page_id)
    }

    fn page_file(&mut self) -> Result<&mut std::fs::File> {
        if self.pages.is_none() {
            let mut pages = create_spool_file()?;
            pages.write_all(&[0_u8; HEADER_SIZE])?;
            self.page_bytes = HEADER_SIZE as u64;
            self.pages = Some(pages);
        }
        Ok(self.pages.as_mut().expect("page file initialized above"))
    }

    fn leaf_separator_run(&mut self) -> Result<&mut SeparatorRun> {
        if self.leaf_separators.is_none() {
            self.leaf_separators = Some(SeparatorRun {
                file: create_spool_file()?,
                count: 0,
                max_key_len: 0,
            });
        }
        Ok(self
            .leaf_separators
            .as_mut()
            .expect("separator run initialized above"))
    }

    fn patch_last_leaf(&mut self) -> Result<()> {
        let page_start = (HEADER_SIZE as u64)
            .checked_add(
                u64::from(self.leaf_count - 1)
                    .checked_mul(PAGE_SIZE as u64)
                    .ok_or_else(|| Error::invariant("last leaf offset exceeds u64"))?,
            )
            .ok_or_else(|| Error::invariant("last leaf offset exceeds u64"))?;
        let pages = self
            .pages
            .as_mut()
            .ok_or_else(|| Error::invariant("spooled paged-index has no page file"))?;
        pages.seek(std::io::SeekFrom::Start(page_start))?;
        let mut page = vec![0_u8; PAGE_SIZE];
        pages.read_exact(&mut page)?;
        page[8..12].copy_from_slice(&NO_PAGE.to_le_bytes());
        let checksum = page_checksum(&page, self.format);
        page[12..16].copy_from_slice(&checksum.to_le_bytes());
        pages.seek(std::io::SeekFrom::Start(page_start))?;
        pages.write_all(&page)?;
        Ok(())
    }

    fn build_internal_level(&mut self, mut input: SeparatorRun) -> Result<SeparatorRun> {
        let largest_record = 2usize
            .checked_add(input.max_key_len)
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or_else(|| Error::invariant("paged-index separator size exceeds usize"))?;
        if input.count > 1
            && largest_record
                .checked_mul(2)
                .is_none_or(|two_records| two_records > PAGE_PAYLOAD_SIZE)
        {
            return Err(Error::invariant(
                "paged-index separator fanout is below two",
            ));
        }
        input.file.rewind()?;
        let mut output = SeparatorRun {
            file: create_spool_file()?,
            count: 0,
            max_key_len: 0,
        };
        let mut payload = BytesMut::with_capacity(PAGE_PAYLOAD_SIZE);
        let mut count = 0_u32;
        let mut last_key: Option<Vec<u8>> = None;
        let mut previous_key: Option<Vec<u8>> = None;

        for _ in 0..input.count {
            let (key, child) = read_separator(&mut input.file)?;
            if previous_key
                .as_deref()
                .is_some_and(|previous| previous >= key.as_slice())
            {
                return Err(Error::invariant(
                    "paged-index separator run is not strictly ascending",
                ));
            }
            let key_len = u16::try_from(key.len())
                .map_err(|_| Error::invariant("paged-index separator too long"))?;
            let record_len = 2usize
                .checked_add(key.len())
                .and_then(|bytes| bytes.checked_add(4))
                .ok_or_else(|| Error::invariant("paged-index separator size exceeds usize"))?;
            if record_len > PAGE_PAYLOAD_SIZE {
                return Err(Error::invariant(
                    "paged-index separator does not fit in one page",
                ));
            }
            if count > 0 && payload.len().saturating_add(record_len) > PAGE_PAYLOAD_SIZE {
                self.emit_internal_page(&mut payload, count, last_key.take(), &mut output)?;
                count = 0;
            }
            payload.put_u16_le(key_len);
            payload.extend_from_slice(&key);
            payload.put_u32_le(child);
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("paged-index internal entry count overflow"))?;
            previous_key = Some(key.clone());
            last_key = Some(key);
        }
        if count > 0 {
            self.emit_internal_page(&mut payload, count, last_key.take(), &mut output)?;
        }
        assert_separator_eof(&mut input.file)?;
        if input.count > 1 && output.count >= input.count {
            return Err(Error::invariant(
                "paged-index separator fanout is below two",
            ));
        }
        finish_separator_run(&mut output)?;
        Ok(output)
    }

    fn emit_internal_page(
        &mut self,
        payload: &mut BytesMut,
        count: u32,
        max_key: Option<Vec<u8>>,
        output: &mut SeparatorRun,
    ) -> Result<()> {
        let body = std::mem::replace(payload, BytesMut::with_capacity(PAGE_PAYLOAD_SIZE)).freeze();
        let page = encode_page(self.format, PAGE_INTERNAL, count, NO_PAGE, body)?;
        let page_id = self.append_page(&page)?;
        let max_key =
            max_key.ok_or_else(|| Error::invariant("internal page has no maximum key"))?;
        write_separator(output, &max_key, page_id)
    }

    #[cfg(test)]
    fn resident_bound_bytes(&self) -> usize {
        // Current leaf/internal payload, encoded page, separator decode and
        // current/previous/max fixed 32-byte edge keys. Independent of
        // entry/page count.
        4 * PAGE_SIZE + 4 * (32 + SEPARATOR_SPOOL_HEADER_BYTES)
    }
}

fn append_spooled_value(values: &mut SpooledValueRegion, value: &[u8]) -> Result<()> {
    let file = match &mut values.file {
        Some(file) => file,
        None => values.file.insert(create_spool_file()?),
    };
    file.write_all(value)?;
    values.len = values
        .len
        .checked_add(value.len() as u64)
        .ok_or_else(|| Error::invariant("paged-index value region exceeds u64"))?;
    Ok(())
}

fn finish_spooled_value_region(values: &mut SpooledValueRegion) -> Result<()> {
    if let Some(file) = &mut values.file {
        if file.metadata()?.len() != values.len {
            return Err(Error::invariant(
                "paged-index value spool length changed while building sidecar",
            ));
        }
        file.sync_data()?;
        file.rewind()?;
    }
    Ok(())
}

fn write_separator(run: &mut SeparatorRun, key: &[u8], child: u32) -> Result<()> {
    let key_len = u32::try_from(key.len())
        .map_err(|_| Error::invariant("paged-index scratch separator exceeds u32"))?;
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&child.to_le_bytes());
    checksum.update(key);
    run.file.write_all(&key_len.to_le_bytes())?;
    run.file.write_all(&child.to_le_bytes())?;
    run.file.write_all(&checksum.finalize().to_le_bytes())?;
    run.file.write_all(key)?;
    run.count = run
        .count
        .checked_add(1)
        .ok_or_else(|| Error::invariant("paged-index separator count exceeds u64"))?;
    run.max_key_len = run.max_key_len.max(key.len());
    Ok(())
}

fn read_separator(file: &mut std::fs::File) -> Result<(Vec<u8>, u32)> {
    let mut header = [0_u8; SEPARATOR_SPOOL_HEADER_BYTES];
    file.read_exact(&mut header)?;
    let key_len = u32::from_le_bytes(
        header[..4]
            .try_into()
            .map_err(|_| Error::invariant("invalid separator scratch key length"))?,
    ) as usize;
    if key_len > u16::MAX as usize {
        return Err(Error::invariant(
            "paged-index scratch separator exceeds wire key limit",
        ));
    }
    let child = u32::from_le_bytes(
        header[4..8]
            .try_into()
            .map_err(|_| Error::invariant("invalid separator scratch child"))?,
    );
    let expected_checksum = u32::from_le_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| Error::invariant("invalid separator scratch checksum"))?,
    );
    let mut key = vec![0_u8; key_len];
    file.read_exact(&mut key)?;
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&child.to_le_bytes());
    checksum.update(&key);
    if checksum.finalize() != expected_checksum {
        return Err(Error::invariant(
            "paged-index separator scratch checksum mismatch",
        ));
    }
    Ok((key, child))
}

fn finish_separator_run(run: &mut SeparatorRun) -> Result<()> {
    run.file.sync_data()?;
    run.file.rewind()?;
    Ok(())
}

fn assert_separator_eof(file: &mut std::fs::File) -> Result<()> {
    let mut extra = [0_u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(Error::invariant(
            "paged-index separator scratch has trailing bytes",
        ));
    }
    Ok(())
}

fn read_single_separator(run: &mut SeparatorRun) -> Result<(Vec<u8>, u32)> {
    if run.count != 1 {
        return Err(Error::invariant(
            "paged-index root separator run must contain one record",
        ));
    }
    run.file.rewind()?;
    let root = read_separator(&mut run.file)?;
    assert_separator_eof(&mut run.file)?;
    Ok(root)
}

/// Streaming builder shared by all four physical index kinds.
///
/// It borrows keys and values from the authoritative maps one at a time.
/// Equality postings are copied directly into the final value region rather
/// than first flattening every posting into a second `Vec`. At legal-corpus
/// scale this avoids retaining another complete key/posting corpus while the
/// legacy body and Parquet writer are already live.
#[derive(Debug)]
struct PagedIndexBuilder {
    format: PagedFormat,
    kind: PagedIndexKind,
    /// Header placeholder followed by every leaf/internal page. Keeping pages
    /// in their final contiguous allocation avoids a second full-page-region
    /// copy when the body is finalized.
    pages: BytesMut,
    leaf_max_keys: Vec<Vec<u8>>,
    value_region: BytesMut,
    /// Exact node records are large (notably 1024d embeddings) and coexist
    /// with the Parquet output during flush/compaction. Spool only that value
    /// region to anonymous local storage; other, historically small paged
    /// sidecars retain their existing in-memory representation.
    spooled_value_region: Option<SpooledValueRegion>,
    payload: BytesMut,
    current_count: u32,
    entry_count: u64,
    last_key: Option<Vec<u8>>,
}

impl PagedIndexBuilder {
    fn new(kind: PagedIndexKind) -> Self {
        let mut pages = BytesMut::with_capacity(HEADER_SIZE + PAGE_SIZE);
        pages.resize(HEADER_SIZE, 0);
        Self {
            format: PagedFormat::V2,
            kind,
            pages,
            leaf_max_keys: Vec::new(),
            value_region: BytesMut::new(),
            spooled_value_region: (kind == PagedIndexKind::NodeRecord)
                .then_some(SpooledValueRegion { file: None, len: 0 }),
            payload: BytesMut::with_capacity(PAGE_PAYLOAD_SIZE),
            current_count: 0,
            entry_count: 0,
            last_key: None,
        }
    }

    fn push_inline(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if self.kind.external_values() {
            return Err(Error::invariant(
                "external paged index received an inline value",
            ));
        }
        let value_len = u16::try_from(value.len())
            .map_err(|_| Error::invariant("inline paged-index value too large"))?;
        let record_len = 2 + key.len() + 2 + value.len();
        let key_len = self.prepare_entry(key, record_len)?;
        self.payload.put_u16_le(key_len);
        self.payload.extend_from_slice(key);
        self.payload.put_u16_le(value_len);
        self.payload.extend_from_slice(value);
        Ok(())
    }

    #[cfg(test)]
    fn push_posting(&mut self, key: &[u8], ids: &[[u8; 16]]) -> Result<()> {
        if !self.kind.external_values() {
            return Err(Error::invariant(
                "inline paged index received an external posting",
            ));
        }
        let posting_len = ids
            .len()
            .checked_mul(16)
            .ok_or_else(|| Error::invariant("equality posting byte length overflow"))?;
        let posting_len = u32::try_from(posting_len)
            .map_err(|_| Error::invariant("equality posting exceeds 4 GiB"))?;
        let key_len = self.prepare_entry(key, EQUALITY_LEAF_OVERHEAD + key.len())?;
        let mut value_checksum = crc32fast::Hasher::new();
        for id in ids {
            value_checksum.update(id);
        }
        self.payload.put_u16_le(key_len);
        self.payload.extend_from_slice(key);
        self.payload.put_u64_le(self.external_value_len());
        self.payload.put_u32_le(posting_len);
        self.payload.put_u32_le(value_checksum.finalize());
        for id in ids {
            self.value_region.extend_from_slice(id);
        }
        Ok(())
    }

    fn push_external(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if !self.kind.external_values() {
            return Err(Error::invariant(
                "inline paged index received an external value",
            ));
        }
        let value_len = u32::try_from(value.len())
            .map_err(|_| Error::invariant("paged-index value exceeds 4 GiB"))?;
        let key_len = self.prepare_entry(key, EQUALITY_LEAF_OVERHEAD + key.len())?;
        let value_offset = self.external_value_len();
        self.payload.put_u16_le(key_len);
        self.payload.extend_from_slice(key);
        self.payload.put_u64_le(value_offset);
        self.payload.put_u32_le(value_len);
        self.payload.put_u32_le(crc32fast::hash(value));
        self.append_external_value(value)?;
        Ok(())
    }

    fn external_value_len(&self) -> u64 {
        self.spooled_value_region
            .as_ref()
            .map_or(self.value_region.len() as u64, |values| values.len)
    }

    fn append_external_value(&mut self, value: &[u8]) -> Result<()> {
        let Some(values) = &mut self.spooled_value_region else {
            self.value_region.extend_from_slice(value);
            return Ok(());
        };
        let file = match &mut values.file {
            Some(file) => file,
            None => values.file.insert(create_spool_file()?),
        };
        file.write_all(value)?;
        values.len = values
            .len
            .checked_add(value.len() as u64)
            .ok_or_else(|| Error::invariant("paged-index value region exceeds u64"))?;
        Ok(())
    }

    fn prepare_entry(&mut self, key: &[u8], record_len: usize) -> Result<u16> {
        let key_len = u16::try_from(key.len())
            .map_err(|_| Error::invariant("paged-index key exceeds 65535 bytes"))?;
        if self
            .last_key
            .as_deref()
            .is_some_and(|previous| previous >= key)
        {
            return Err(Error::invariant(
                "paged-index input keys must be strictly ascending",
            ));
        }
        if record_len > PAGE_PAYLOAD_SIZE {
            return Err(Error::invariant(
                "paged-index leaf entry does not fit in one page",
            ));
        }
        if self.current_count > 0
            && self.payload.len().saturating_add(record_len) > PAGE_PAYLOAD_SIZE
        {
            self.flush_leaf()?;
        }
        self.current_count = self
            .current_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("paged-index leaf entry count overflow"))?;
        self.entry_count = self
            .entry_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("paged-index entry count overflow"))?;
        if let Some(last_key) = &mut self.last_key {
            last_key.clear();
            last_key.extend_from_slice(key);
        } else {
            self.last_key = Some(key.to_vec());
        }
        Ok(key_len)
    }

    fn flush_leaf(&mut self) -> Result<()> {
        if self.current_count == 0 {
            return Ok(());
        }
        let next = u32::try_from(self.leaf_max_keys.len() + 1)
            .map_err(|_| Error::invariant("paged-index page count exceeds u32"))?;
        let payload = std::mem::replace(
            &mut self.payload,
            BytesMut::with_capacity(PAGE_PAYLOAD_SIZE),
        )
        .freeze();
        self.leaf_max_keys.push(
            self.last_key
                .clone()
                .expect("non-empty leaf has a maximum key"),
        );
        self.pages.extend_from_slice(&encode_page(
            self.format,
            PAGE_LEAF,
            self.current_count,
            next,
            payload,
        )?);
        self.current_count = 0;
        Ok(())
    }

    fn finish(self) -> Result<Bytes> {
        if self.spooled_value_region.is_some() {
            let (pages, values) = self.finish_spooled_parts()?;
            let total_len = usize::try_from(values.len)
                .ok()
                .and_then(|values_len| pages.len().checked_add(values_len))
                .ok_or_else(|| Error::invariant("paged-index body exceeds addressable memory"))?;
            let mut out = BytesMut::with_capacity(total_len);
            out.extend_from_slice(&pages);
            if let Some(mut file) = values.file {
                let mut chunk = [0_u8; 64 * 1024];
                loop {
                    let read = file.read(&mut chunk)?;
                    if read == 0 {
                        break;
                    }
                    out.extend_from_slice(&chunk[..read]);
                }
            }
            return Ok(out.freeze());
        }

        let (pages, values) = self.finish_parts()?;
        if values.is_empty() {
            return Ok(pages);
        }
        let mut out = BytesMut::with_capacity(pages.len().saturating_add(values.len()));
        out.extend_from_slice(&pages);
        out.extend_from_slice(&values);
        Ok(out.freeze())
    }

    fn finish_parts(mut self) -> Result<(Bytes, Bytes)> {
        if self.spooled_value_region.is_some() {
            return Err(Error::invariant(
                "spooled paged-index values require finish_spooled_parts",
            ));
        }
        self.finish_pages()?;
        Ok((self.pages.freeze(), self.value_region.freeze()))
    }

    fn finish_spooled_parts(mut self) -> Result<(Bytes, SpooledValueRegion)> {
        if self.spooled_value_region.is_none() {
            return Err(Error::invariant(
                "in-memory paged-index values require finish_parts",
            ));
        }
        self.finish_pages()?;
        let mut values = self
            .spooled_value_region
            .take()
            .expect("spooled value region checked above");
        if let Some(file) = &mut values.file {
            if file.metadata()?.len() != values.len {
                return Err(Error::invariant(
                    "paged-index value spool length changed while building sidecar",
                ));
            }
            // Exact node/edge records can be several GiB. Surface delayed
            // allocation/writeback failures before multipart upload and keep
            // those pages reclaimable while the authoritative SST and B+tree
            // pages are still live.
            file.sync_data()?;
            file.rewind()?;
        }
        Ok((self.pages.freeze(), values))
    }

    fn finish_pages(&mut self) -> Result<()> {
        self.flush_leaf()?;
        if self.leaf_max_keys.is_empty() {
            self.leaf_max_keys.push(Vec::new());
            self.pages.extend_from_slice(&encode_page(
                self.format,
                PAGE_LEAF,
                0,
                NO_PAGE,
                Bytes::new(),
            )?);
        } else {
            // Every leaf is initially linked to its predicted successor. The
            // final leaf has no successor; patch its next pointer in place and
            // refresh the page checksum without copying the 4 KiB page.
            let page_start = HEADER_SIZE + (self.leaf_max_keys.len() - 1).saturating_mul(PAGE_SIZE);
            self.pages[page_start + 8..page_start + 12].copy_from_slice(&NO_PAGE.to_le_bytes());
            let checksum =
                page_checksum(&self.pages[page_start..page_start + PAGE_SIZE], self.format);
            self.pages[page_start + 12..page_start + 16].copy_from_slice(&checksum.to_le_bytes());
        }

        let leaf_count = self.leaf_max_keys.len() as u32;
        let mut level: Vec<(Vec<u8>, u32)> = std::mem::take(&mut self.leaf_max_keys)
            .into_iter()
            .enumerate()
            .map(|(id, max_key)| (max_key, id as u32))
            .collect();

        while level.len() > 1 {
            let largest_record = level
                .iter()
                .map(|(key, _)| 2usize.saturating_add(key.len()).saturating_add(4))
                .max()
                .unwrap_or(0);
            if largest_record
                .checked_mul(2)
                .is_none_or(|two_records| two_records > PAGE_PAYLOAD_SIZE)
            {
                return Err(Error::invariant(
                    "paged-index separator fanout is below two",
                ));
            }
            let mut next_level = Vec::new();
            let mut pos = 0usize;
            while pos < level.len() {
                let start = pos;
                let mut payload = BytesMut::with_capacity(PAGE_PAYLOAD_SIZE);
                while pos < level.len() {
                    let (max_key, child) = &level[pos];
                    let key_len = u16::try_from(max_key.len())
                        .map_err(|_| Error::invariant("paged-index separator too long"))?;
                    let record_len = 2 + max_key.len() + 4;
                    if payload.len().saturating_add(record_len) > PAGE_PAYLOAD_SIZE {
                        if pos == start {
                            return Err(Error::invariant(
                                "paged-index separator does not fit in one page",
                            ));
                        }
                        break;
                    }
                    payload.put_u16_le(key_len);
                    payload.extend_from_slice(max_key);
                    payload.put_u32_le(*child);
                    pos += 1;
                }
                let page_id = ((self.pages.len() - HEADER_SIZE) / PAGE_SIZE) as u32;
                self.pages.extend_from_slice(&encode_page(
                    self.format,
                    PAGE_INTERNAL,
                    (pos - start) as u32,
                    NO_PAGE,
                    payload.freeze(),
                )?);
                next_level.push((level[pos - 1].0.clone(), page_id));
            }
            level = next_level;
        }

        let page_count = ((self.pages.len() - HEADER_SIZE) / PAGE_SIZE) as u32;
        let header = Header {
            format: self.format,
            kind: self.kind,
            root_page: level[0].1,
            page_count,
            leaf_count,
            entry_count: self.entry_count,
            values_offset: self.pages.len() as u64,
        };
        self.pages[..HEADER_SIZE].copy_from_slice(&header.encode());
        Ok(())
    }
}

pub(crate) fn create_spool_file() -> std::io::Result<std::fs::File> {
    match std::env::var_os("NAMIDB_SPOOL_DIR").filter(|path| !path.is_empty()) {
        Some(directory) => tempfile::tempfile_in(directory),
        None => {
            // `/tmp` is commonly a RAM-backed tmpfs on Linux hosts. Falling
            // back there for a multi-gigabyte vector sidecar would merely move
            // the original OOM out of the allocator. `/var/tmp` is the
            // conventional disk-backed temporary location on Unix; other
            // platforms keep their native temporary-directory behavior.
            #[cfg(unix)]
            {
                tempfile::tempfile_in("/var/tmp")
            }
            #[cfg(not(unix))]
            {
                tempfile::tempfile()
            }
        }
    }
}

fn encode_page(
    format: PagedFormat,
    kind: u8,
    count: u32,
    next: u32,
    payload: Bytes,
) -> Result<Bytes> {
    if PAGE_HEADER_SIZE + payload.len() > PAGE_SIZE {
        return Err(Error::invariant("paged-index page overflow"));
    }
    let mut out = BytesMut::with_capacity(PAGE_SIZE);
    out.put_u8(kind);
    out.extend_from_slice(&[0; 3]);
    out.put_u32_le(count);
    out.put_u32_le(next);
    out.put_u32_le(0);
    out.extend_from_slice(&payload);
    out.resize(PAGE_SIZE, 0);
    let checksum = page_checksum(&out, format);
    out[12..16].copy_from_slice(&checksum.to_le_bytes());
    Ok(out.freeze())
}

fn checksum_around_slot(bytes: &[u8], slot: usize) -> u32 {
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&bytes[..slot]);
    checksum.update(&bytes[slot + 4..]);
    checksum.finalize()
}

fn page_checksum(page: &[u8], format: PagedFormat) -> u32 {
    match format {
        PagedFormat::V1 => crc32fast::hash(&page[PAGE_HEADER_SIZE..]),
        PagedFormat::V2 => checksum_around_slot(page, 12),
    }
}

fn page_range(page: u32) -> Range<u64> {
    page_range_at(0, page).expect("u32 page offsets fit in u64")
}

fn page_range_at(base_offset: u64, page: u32) -> Result<Range<u64>> {
    let start = base_offset
        .checked_add(HEADER_SIZE as u64)
        .and_then(|offset| offset.checked_add(page as u64 * PAGE_SIZE as u64))
        .ok_or_else(|| Error::invariant("paged-index page offset overflow"))?;
    let end = start
        .checked_add(PAGE_SIZE as u64)
        .ok_or_else(|| Error::invariant("paged-index page end overflow"))?;
    Ok(start..end)
}

fn header_range_at(base_offset: u64) -> Result<Range<u64>> {
    let end = base_offset
        .checked_add(HEADER_SIZE as u64)
        .ok_or_else(|| Error::invariant("paged-index header offset overflow"))?;
    Ok(base_offset..end)
}

fn external_value_range(values_offset: u64, offset: u64, len: usize) -> Result<Range<u64>> {
    let len = u64::try_from(len)
        .map_err(|_| Error::invariant("paged-index external value length exceeds u64"))?;
    let start = values_offset
        .checked_add(offset)
        .ok_or_else(|| Error::invariant("paged-index external value offset overflow"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::invariant("paged-index external value end overflow"))?;
    Ok(start..end)
}

#[cfg(test)]
async fn read_range(store: &Arc<dyn ObjectStore>, path: &Path, range: Range<u64>) -> Result<Bytes> {
    let response = store
        .get_opts(
            path,
            GetOptions {
                range: Some(GetRange::Bounded(range)),
                ..Default::default()
            },
        )
        .await?;
    Ok(response.bytes().await?)
}

#[cfg(test)]
async fn read_ranges(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
    ranges: &[Range<u64>],
) -> Result<Vec<Bytes>> {
    if ranges.is_empty() {
        return Ok(Vec::new());
    }
    store.get_ranges(path, ranges).await.map_err(Error::from)
}

#[async_trait::async_trait]
trait PagedRangeSource: Send + Sync {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes>;
    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>>;
}

#[cfg(test)]
struct LegacyObjectRangeSource {
    store: Arc<dyn ObjectStore>,
    path: Path,
}

#[cfg(test)]
#[async_trait::async_trait]
impl PagedRangeSource for LegacyObjectRangeSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        read_range(&self.store, &self.path, range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        read_ranges(&self.store, &self.path, ranges).await
    }
}

#[async_trait::async_trait]
impl PagedRangeSource for PinnedObjectRangeSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        PinnedObjectRangeSource::read_range(self, range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        PinnedObjectRangeSource::read_ranges(self, ranges).await
    }
}

/// Probe unique keys without fetching or decoding the complete sidecar.
#[cfg(test)]
pub async fn probe_unique(
    store: Arc<dyn ObjectStore>,
    path: Path,
    values: &[String],
) -> Result<(BTreeMap<String, [u8; 16]>, PagedProbeStats)> {
    let source = LegacyObjectRangeSource { store, path };
    probe_unique_with_source(&source, values).await
}

/// Generation-pinned unique probe backed by the shared RAM/NVMe range cache.
pub async fn probe_unique_from_source(
    source: &PinnedObjectRangeSource,
    values: &[String],
) -> Result<(BTreeMap<String, [u8; 16]>, PagedProbeStats)> {
    probe_unique_with_source(source, values).await
}

async fn probe_unique_with_source(
    source: &dyn PagedRangeSource,
    values: &[String],
) -> Result<(BTreeMap<String, [u8; 16]>, PagedProbeStats)> {
    let keys: Vec<Vec<u8>> = values.iter().map(|v| v.as_bytes().to_vec()).collect();
    let (found, stats) = probe_source(source, PagedIndexKind::Unique, &keys, None).await?;
    let mut out = BTreeMap::new();
    for (key, value) in found {
        if value.len() != 16 {
            return Err(Error::invariant("unique paged-index value is not 16 bytes"));
        }
        let mut id = [0; 16];
        id.copy_from_slice(&value);
        out.insert(
            String::from_utf8(key)
                .map_err(|e| Error::invariant(format!("unique index key utf8: {e}")))?,
            id,
        );
    }
    Ok((out, stats))
}

/// Probe equality keys without fetching unrelated posting lists.
#[cfg(test)]
pub async fn probe_equality(
    store: Arc<dyn ObjectStore>,
    path: Path,
    values: &[String],
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    probe_equality_limited(store, path, values, None).await
}

/// Generation-pinned equality probe backed by the shared range cache.
pub async fn probe_equality_from_source(
    source: &PinnedObjectRangeSource,
    values: &[String],
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    probe_equality_limited_from_source(source, values, None).await
}

/// Equality probe that reads at most `max_ids_per_value` ids from each
/// posting. `stats.matched_value_bytes` still reports the complete logical
/// posting size, and `values_truncated` tells the caller to continue/fallback
/// if confirmation exhausts the prefix.
#[cfg(test)]
pub async fn probe_equality_limited(
    store: Arc<dyn ObjectStore>,
    path: Path,
    values: &[String],
    max_ids_per_value: Option<usize>,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    let source = LegacyObjectRangeSource { store, path };
    probe_equality_limited_with_source(&source, values, max_ids_per_value).await
}

/// Generation-pinned limited equality probe.
pub async fn probe_equality_limited_from_source(
    source: &PinnedObjectRangeSource,
    values: &[String],
    max_ids_per_value: Option<usize>,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    probe_equality_limited_with_source(source, values, max_ids_per_value).await
}

async fn probe_equality_limited_with_source(
    source: &dyn PagedRangeSource,
    values: &[String],
    max_ids_per_value: Option<usize>,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    let keys: Vec<Vec<u8>> = values.iter().map(|v| v.as_bytes().to_vec()).collect();
    let (found, stats) =
        probe_equality_bytes_limited_with_source(source, &keys, max_ids_per_value).await?;
    let mut out = BTreeMap::new();
    for (key, ids) in found {
        out.insert(
            String::from_utf8(key)
                .map_err(|e| Error::invariant(format!("equality index key utf8: {e}")))?,
            ids,
        );
    }
    Ok((out, stats))
}

/// Generation-pinned equality probe over raw byte keys.
///
/// Composite tuple sidecars key postings by [`TupleV1`] encodings, which
/// are not UTF-8; this is the byte-transparent surface the String probes
/// above wrap.
///
/// [`TupleV1`]: crate::manifest::CompositeKeyEncoding::TupleV1
pub async fn probe_equality_bytes_limited_from_source(
    source: &PinnedObjectRangeSource,
    keys: &[Vec<u8>],
    max_ids_per_value: Option<usize>,
) -> Result<(BTreeMap<Vec<u8>, Vec<[u8; 16]>>, PagedProbeStats)> {
    probe_equality_bytes_limited_with_source(source, keys, max_ids_per_value).await
}

async fn probe_equality_bytes_limited_with_source(
    source: &dyn PagedRangeSource,
    keys: &[Vec<u8>],
    max_ids_per_value: Option<usize>,
) -> Result<(BTreeMap<Vec<u8>, Vec<[u8; 16]>>, PagedProbeStats)> {
    let byte_limit = max_ids_per_value.map(|ids| ids.saturating_mul(16));
    let (found, stats) = probe_source(source, PagedIndexKind::Equality, keys, byte_limit).await?;
    let mut out = BTreeMap::new();
    for (key, value) in found {
        if value.len() % 16 != 0 {
            return Err(Error::invariant(
                "equality paged-index posting is not NodeId-aligned",
            ));
        }
        let ids = value
            .chunks_exact(16)
            .map(|chunk| {
                let mut id = [0; 16];
                id.copy_from_slice(chunk);
                id
            })
            .collect();
        out.insert(key, ids);
    }
    Ok((out, stats))
}

/// One range-readable page of a label's NodeIds.
///
/// `next_after` is the exclusive cursor for the next call and is present only
/// when another matching id was observed. The cursor is a NodeId rather than a
/// byte offset, so immutable generations remain independently pageable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelIdPage {
    pub ids: Vec<[u8; 16]>,
    pub next_after: Option<[u8; 16]>,
}

struct LabelBinding {
    header: Header,
    sst_id: [u8; 16],
    page_region_checksum: u32,
}

/// Exact membership probe against a composite
/// `(LabelId big-endian, NodeId)` PagedV2 sidecar.
///
/// The fixed footer is checked against the manifest's exact per-label counts
/// before any tree page is trusted. Only header, footer and one B+tree path
/// are read.
#[cfg(test)]
pub async fn label_contains_from_source(
    source: &PinnedObjectRangeSource,
    label_id: u32,
    id: [u8; 16],
    expected_sst_id: [u8; 16],
    per_label_counts: &[(u32, u64)],
) -> Result<(bool, PagedProbeStats)> {
    let (binding, mut stats) = validate_label_binding(
        source,
        source.object_size(),
        expected_sst_id,
        per_label_counts,
    )
    .await?;
    let key = label_membership_key(label_id, id);
    let found = find_inline_key(source, &binding, &key, &mut stats).await?;
    Ok((found, stats))
}

/// Batch label-membership probe with one header/footer validation.
///
/// Probes are grouped by B+tree page and issued in bounded range batches;
/// duplicate ids inside a chunk share the same walk. The returned booleans
/// retain the caller's exact order.
pub async fn batch_label_contains_from_source(
    source: &PinnedObjectRangeSource,
    label_id: u32,
    ids: &[[u8; 16]],
    expected_sst_id: [u8; 16],
    per_label_counts: &[(u32, u64)],
) -> Result<(Vec<bool>, PagedProbeStats)> {
    const PROBE_CHUNK_IDS: usize = 512;

    let (binding, mut stats) = validate_label_binding(
        source,
        source.object_size(),
        expected_sst_id,
        per_label_counts,
    )
    .await?;
    let mut result = Vec::with_capacity(ids.len());
    for chunk in ids.chunks(PROBE_CHUNK_IDS) {
        let keys: Vec<Vec<u8>> = chunk
            .iter()
            .map(|id| label_membership_key(label_id, *id).to_vec())
            .collect();
        let found = batch_find_inline_keys(source, &binding, &keys, &mut stats).await?;
        result.extend(keys.iter().map(|key| found.contains(key)));
    }
    Ok((result, stats))
}

/// Prefix/range page for one label without hydrating its complete posting
/// list. Each call reads one search path and as many linked leaf pages as the
/// requested `limit` needs.
pub async fn page_label_ids_from_source(
    source: &PinnedObjectRangeSource,
    label_id: u32,
    after: Option<[u8; 16]>,
    limit: usize,
    expected_sst_id: [u8; 16],
    per_label_counts: &[(u32, u64)],
) -> Result<(LabelIdPage, PagedProbeStats)> {
    let (binding, mut stats) = validate_label_binding(
        source,
        source.object_size(),
        expected_sst_id,
        per_label_counts,
    )
    .await?;
    if limit == 0 {
        return Ok((
            LabelIdPage {
                ids: Vec::new(),
                next_after: None,
            },
            stats,
        ));
    }

    let start = label_membership_key(label_id, after.unwrap_or([0_u8; 16]));
    let (mut page_id, first_page) = descend_to_leaf(source, &binding, &start, &mut stats).await?;
    let mut prefetched_page = Some(first_page);
    let mut ids = Vec::with_capacity(limit.saturating_add(1).min(4096));
    let mut first_leaf = true;
    let mut done = false;
    while !done {
        crate::cancel::check()?;
        if page_id >= binding.header.leaf_count {
            return Err(Error::invariant(
                "label-index leaf link points outside leaf region",
            ));
        }
        let page = match prefetched_page.take() {
            Some(page) => page,
            None => {
                let page = source.read_range(page_range(page_id)).await?;
                stats.pages_read += 1;
                stats.bytes_read = stats.bytes_read.saturating_add(page.len());
                page
            }
        };
        let entries = parse_label_leaf(&page, page_id, &binding)?;
        stats.leaf_entries_examined = stats.leaf_entries_examined.saturating_add(entries.len());
        let start_at = if first_leaf {
            entries.partition_point(|entry| {
                if after.is_some() {
                    entry.key.as_slice() <= start.as_slice()
                } else {
                    entry.key.as_slice() < start.as_slice()
                }
            })
        } else {
            0
        };
        first_leaf = false;
        for entry in entries.into_iter().skip(start_at) {
            let Some((candidate_label, candidate_id)) = decode_label_membership_key(&entry.key)?
            else {
                return Err(Error::invariant(
                    "label-index membership key has invalid width",
                ));
            };
            if candidate_label < label_id {
                continue;
            }
            if candidate_label > label_id {
                done = true;
                break;
            }
            match entry.value {
                LeafValue::Inline(value) if value.is_empty() => {}
                _ => {
                    return Err(Error::invariant(
                        "label-index membership entry has a non-empty value",
                    ));
                }
            }
            ids.push(candidate_id);
            if ids.len() > limit {
                done = true;
                break;
            }
        }
        if done {
            break;
        }
        let next = read_u32(&page, 8)?;
        if next == NO_PAGE {
            if page_id + 1 != binding.header.leaf_count {
                return Err(Error::invariant(
                    "label-index leaf chain ends before the final leaf",
                ));
            }
            break;
        }
        if next != page_id + 1 {
            return Err(Error::invariant("label-index leaf chain is not contiguous"));
        }
        page_id = next;
    }
    let has_more = ids.len() > limit;
    if has_more {
        ids.pop();
    }
    let next_after = has_more.then(|| {
        *ids.last()
            .expect("limit is non-zero and one extra id proves a returned id")
    });
    Ok((LabelIdPage { ids, next_after }, stats))
}

async fn validate_label_binding(
    source: &dyn PagedRangeSource,
    object_size: u64,
    expected_sst_id: [u8; 16],
    per_label_counts: &[(u32, u64)],
) -> Result<(LabelBinding, PagedProbeStats)> {
    let footer_start = object_size
        .checked_sub(LABEL_FOOTER_BYTES as u64)
        .ok_or_else(|| Error::invariant("label-index object is shorter than its footer"))?;
    let ranges = [header_range_at(0)?, footer_start..object_size];
    let parts = source.read_ranges(&ranges).await?;
    if parts.len() != 2 || parts[0].len() != HEADER_SIZE || parts[1].len() != LABEL_FOOTER_BYTES {
        return Err(Error::invariant("label-index binding ranges are truncated"));
    }
    let header = Header::decode(&parts[0], PagedIndexKind::LabelMembership)?;
    header.require_authoritative_integrity()?;
    let footer = decode_label_footer(&parts[1])?;
    let mut previous = None;
    let mut posting_count = 0_u64;
    for &(label, count) in per_label_counts {
        if count == 0 || previous.is_some_and(|prior| prior >= label) {
            return Err(Error::invariant(
                "label-index per-label counts are not strictly ordered/non-zero",
            ));
        }
        previous = Some(label);
        posting_count = posting_count
            .checked_add(count)
            .ok_or_else(|| Error::invariant("label posting count exceeds u64"))?;
    }
    if footer.page_region_bytes != header.values_offset
        || footer
            .page_region_bytes
            .checked_add(LABEL_FOOTER_BYTES as u64)
            != Some(object_size)
        || footer.entry_count != header.entry_count
        || footer.entry_count != posting_count
        || footer.label_count != per_label_counts.len() as u64
        || footer.header_checksum != read_u32(&parts[0], HEADER_CHECKSUM_OFFSET)?
        || footer.counts_checksum != label_counts_checksum(per_label_counts)
        || footer.sst_id != expected_sst_id
    {
        return Err(Error::invariant(
            "label-index footer disagrees with header/manifest binding",
        ));
    }
    Ok((
        LabelBinding {
            header,
            sst_id: footer.sst_id,
            page_region_checksum: footer.page_region_checksum,
        },
        PagedProbeStats {
            index_entries: footer.entry_count,
            bytes_read: parts.iter().map(Bytes::len).sum(),
            ..Default::default()
        },
    ))
}

#[cfg(test)]
async fn find_inline_key(
    source: &dyn PagedRangeSource,
    binding: &LabelBinding,
    key: &[u8],
    stats: &mut PagedProbeStats,
) -> Result<bool> {
    let (leaf, page) = descend_to_leaf(source, binding, key, stats).await?;
    let entries = parse_label_leaf(&page, leaf, binding)?;
    stats.leaf_entries_examined = stats.leaf_entries_examined.saturating_add(entries.len());
    let Ok(position) = entries.binary_search_by(|entry| entry.key.as_slice().cmp(key)) else {
        return Ok(false);
    };
    match &entries[position].value {
        LeafValue::Inline(value) if value.is_empty() => Ok(true),
        _ => Err(Error::invariant(
            "label-index membership entry has a non-empty value",
        )),
    }
}

async fn batch_find_inline_keys(
    source: &dyn PagedRangeSource,
    binding: &LabelBinding,
    keys: &[Vec<u8>],
    stats: &mut PagedProbeStats,
) -> Result<BTreeSet<Vec<u8>>> {
    let unique: Vec<Vec<u8>> = keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if unique.is_empty() {
        return Ok(BTreeSet::new());
    }
    let mut assignments = BTreeMap::from([(binding.header.root_page, unique)]);
    let mut found = BTreeSet::new();
    loop {
        crate::cancel::check()?;
        let page_assignments: Vec<(u32, Vec<Vec<u8>>)> = assignments.into_iter().collect();
        let mut next: BTreeMap<u32, Vec<Vec<u8>>> = BTreeMap::new();
        let mut level_kind = None;
        for batch in page_assignments.chunks(crate::range_cache::MAX_PINNED_OBJECT_RANGE_READS) {
            let ranges = batch
                .iter()
                .map(|(page_id, _)| page_range(*page_id))
                .collect::<Vec<_>>();
            let pages = source.read_ranges(&ranges).await?;
            if pages.len() != batch.len() {
                return Err(Error::invariant(
                    "label-index range source returned the wrong page count",
                ));
            }
            stats.pages_read = stats.pages_read.saturating_add(pages.len());
            stats.bytes_read = stats
                .bytes_read
                .saturating_add(pages.iter().map(Bytes::len).sum::<usize>());
            for ((page_id, page_keys), page) in batch.iter().zip(pages) {
                if *page_id >= binding.header.page_count {
                    return Err(Error::invariant(
                        "label-index child points outside page region",
                    ));
                }
                match page.first().copied() {
                    Some(PAGE_INTERNAL) => {
                        if level_kind.replace(PAGE_INTERNAL) == Some(PAGE_LEAF) {
                            return Err(Error::invariant(
                                "label-index level mixes internal and leaf pages",
                            ));
                        }
                        let children = parse_label_internal(&page, *page_id, binding)?;
                        for key in page_keys {
                            let child = children
                                .iter()
                                .find(|(max, _)| key.as_slice() <= max.as_slice())
                                .or_else(|| children.last())
                                .map(|(_, child)| *child)
                                .ok_or_else(|| {
                                    Error::invariant("empty label-index internal page")
                                })?;
                            next.entry(child).or_default().push(key.clone());
                        }
                    }
                    Some(PAGE_LEAF) => {
                        if level_kind.replace(PAGE_LEAF) == Some(PAGE_INTERNAL) {
                            return Err(Error::invariant(
                                "label-index level mixes internal and leaf pages",
                            ));
                        }
                        let entries = parse_label_leaf(&page, *page_id, binding)?;
                        stats.leaf_entries_examined =
                            stats.leaf_entries_examined.saturating_add(entries.len());
                        for key in page_keys {
                            if let Ok(position) =
                                entries.binary_search_by(|entry| entry.key.as_slice().cmp(key))
                            {
                                match &entries[position].value {
                                    LeafValue::Inline(value) if value.is_empty() => {
                                        found.insert(key.clone());
                                    }
                                    _ => {
                                        return Err(Error::invariant(
                                            "label-index membership entry has a non-empty value",
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    Some(other) => {
                        return Err(Error::invariant(format!(
                            "invalid label-index page kind {other}"
                        )));
                    }
                    None => return Err(Error::invariant("short label-index page")),
                }
            }
        }
        if level_kind == Some(PAGE_INTERNAL) {
            assignments = next;
        } else {
            break;
        }
    }
    Ok(found)
}

async fn descend_to_leaf(
    source: &dyn PagedRangeSource,
    binding: &LabelBinding,
    key: &[u8],
    stats: &mut PagedProbeStats,
) -> Result<(u32, Bytes)> {
    let mut page_id = binding.header.root_page;
    loop {
        crate::cancel::check()?;
        if page_id >= binding.header.page_count {
            return Err(Error::invariant(
                "paged-index child points outside page region",
            ));
        }
        let page = source.read_range(page_range(page_id)).await?;
        stats.pages_read += 1;
        stats.bytes_read = stats.bytes_read.saturating_add(page.len());
        match page.first().copied() {
            Some(PAGE_LEAF) => return Ok((page_id, page)),
            Some(PAGE_INTERNAL) => {
                let children = parse_label_internal(&page, page_id, binding)?;
                page_id = children
                    .iter()
                    .find(|(max, _)| key <= max.as_slice())
                    .or_else(|| children.last())
                    .map(|(_, child)| *child)
                    .ok_or_else(|| Error::invariant("empty paged-index internal page"))?;
            }
            Some(other) => {
                return Err(Error::invariant(format!(
                    "invalid paged-index page kind {other}"
                )));
            }
            None => return Err(Error::invariant("short paged-index page")),
        }
    }
}

fn label_membership_key(label_id: u32, id: [u8; 16]) -> [u8; 20] {
    let mut key = [0_u8; 20];
    key[..4].copy_from_slice(&label_id.to_be_bytes());
    key[4..].copy_from_slice(&id);
    key
}

fn decode_label_membership_key(key: &[u8]) -> Result<Option<(u32, [u8; 16])>> {
    if key.len() != 20 {
        return Ok(None);
    }
    let label = u32::from_be_bytes(
        key[..4]
            .try_into()
            .map_err(|_| Error::invariant("invalid label membership prefix"))?,
    );
    let id = key[4..]
        .try_into()
        .map_err(|_| Error::invariant("invalid label membership NodeId"))?;
    Ok(Some((label, id)))
}

/// Resolve NodeIds to exact physical row ordinals.
#[cfg(test)]
pub async fn probe_node_locator(
    store: Arc<dyn ObjectStore>,
    path: Path,
    ids: &[[u8; 16]],
) -> Result<(BTreeMap<[u8; 16], u64>, PagedProbeStats)> {
    let source = LegacyObjectRangeSource { store, path };
    probe_node_locator_with_source(&source, ids).await
}

/// Generation-pinned node-locator probe backed by shared range pages.
pub async fn probe_node_locator_from_source(
    source: &PinnedObjectRangeSource,
    ids: &[[u8; 16]],
) -> Result<(BTreeMap<[u8; 16], u64>, PagedProbeStats)> {
    probe_node_locator_with_source(source, ids).await
}

async fn probe_node_locator_with_source(
    source: &dyn PagedRangeSource,
    ids: &[[u8; 16]],
) -> Result<(BTreeMap<[u8; 16], u64>, PagedProbeStats)> {
    let keys: Vec<Vec<u8>> = ids.iter().map(|id| id.to_vec()).collect();
    let (found, stats) = probe_source(source, PagedIndexKind::NodeLocator, &keys, None).await?;
    let mut out = BTreeMap::new();
    for (key, value) in found {
        if key.len() != 16 || value.len() != 8 {
            return Err(Error::invariant("invalid node-locator entry"));
        }
        let mut id = [0; 16];
        id.copy_from_slice(&key);
        let mut ordinal = [0; 8];
        ordinal.copy_from_slice(&value);
        out.insert(id, u64::from_le_bytes(ordinal));
    }
    Ok((out, stats))
}

/// Fetch exact binary node records from a combined
/// [`NodeLocatorRecordBuilder`] sidecar.
///
/// The ordinary locator header at byte zero remains the compatibility and
/// integrity envelope for the prefix. Because node-locator values are inline,
/// its `values_offset` points at the appended `NodeRecord` B+tree. Both trees
/// must advertise the same entry count before the record index is accepted as
/// authoritative.
#[cfg(test)]
pub async fn probe_node_records(
    store: Arc<dyn ObjectStore>,
    path: Path,
    ids: &[[u8; 16]],
) -> Result<(BTreeMap<[u8; 16], Vec<u8>>, PagedProbeStats)> {
    let source = LegacyObjectRangeSource { store, path };
    probe_node_records_with_source(&source, ids).await
}

/// Generation-pinned exact-record probe over a combined `.nloc2` sidecar.
pub async fn probe_node_records_from_source(
    source: &PinnedObjectRangeSource,
    ids: &[[u8; 16]],
) -> Result<(BTreeMap<[u8; 16], Vec<u8>>, PagedProbeStats)> {
    probe_node_records_with_source(source, ids).await
}

async fn probe_node_records_with_source(
    source: &dyn PagedRangeSource,
    ids: &[[u8; 16]],
) -> Result<(BTreeMap<[u8; 16], Vec<u8>>, PagedProbeStats)> {
    crate::cancel::check()?;
    let locator_header_bytes = source.read_range(header_range_at(0)?).await?;
    let locator_header = Header::decode(&locator_header_bytes, PagedIndexKind::NodeLocator)?;
    locator_header.require_authoritative_integrity()?;

    let keys: Vec<Vec<u8>> = ids.iter().map(|id| id.to_vec()).collect();
    let (found, mut stats) = probe_at_source(
        source,
        PagedIndexKind::NodeRecord,
        &keys,
        None,
        locator_header.values_offset,
    )
    .await?;
    stats.bytes_read = stats.bytes_read.saturating_add(locator_header_bytes.len());
    if stats.index_entries != locator_header.entry_count {
        return Err(Error::invariant(
            "node-record index entry count differs from locator prefix",
        ));
    }

    let mut out = BTreeMap::new();
    for (key, value) in found {
        if key.len() != 16 {
            return Err(Error::invariant(
                "node-record paged-index key is not 16 bytes",
            ));
        }
        let mut id = [0; 16];
        id.copy_from_slice(&key);
        out.insert(id, value);
    }
    Ok((out, stats))
}

/// Probe exact edge composite keys without fetching the CSR body or unrelated
/// point values. Duplicate probes are coalesced by the shared B+tree walker.
#[cfg(test)]
pub async fn probe_edge_points(
    store: Arc<dyn ObjectStore>,
    path: Path,
    pairs: &[([u8; 16], [u8; 16])],
) -> Result<(BTreeMap<([u8; 16], [u8; 16]), Vec<u8>>, PagedProbeStats)> {
    let source = LegacyObjectRangeSource { store, path };
    probe_edge_points_with_source(&source, pairs).await
}

/// Generation-pinned edge-point sidecar probe. Edge SST body readers remain
/// separate; this only covers the paged point-index sidecar.
pub async fn probe_edge_points_from_source(
    source: &PinnedObjectRangeSource,
    pairs: &[([u8; 16], [u8; 16])],
) -> Result<(BTreeMap<([u8; 16], [u8; 16]), Vec<u8>>, PagedProbeStats)> {
    probe_edge_points_with_source(source, pairs).await
}

async fn probe_edge_points_with_source(
    source: &dyn PagedRangeSource,
    pairs: &[([u8; 16], [u8; 16])],
) -> Result<(BTreeMap<([u8; 16], [u8; 16]), Vec<u8>>, PagedProbeStats)> {
    let keys: Vec<Vec<u8>> = pairs
        .iter()
        .map(|(key_id, partner_id)| {
            let mut key = Vec::with_capacity(32);
            key.extend_from_slice(key_id);
            key.extend_from_slice(partner_id);
            key
        })
        .collect();
    let (found, stats) = probe_source(source, PagedIndexKind::EdgePoint, &keys, None).await?;
    let mut out = BTreeMap::new();
    for (key, value) in found {
        if key.len() != 32 {
            return Err(Error::invariant(
                "edge-point paged-index key is not 32 bytes",
            ));
        }
        let mut key_id = [0u8; 16];
        let mut partner_id = [0u8; 16];
        key_id.copy_from_slice(&key[..16]);
        partner_id.copy_from_slice(&key[16..]);
        out.insert((key_id, partner_id), value);
    }
    Ok((out, stats))
}

/// Decode every equality posting from an already-fetched paged body.
///
/// Kept for ordered scans that intentionally walk the whole keyspace; point
/// predicates must use [`probe_equality`] so they stay range-readable.
pub fn decode_all_equality(body: &Bytes) -> Result<BTreeMap<String, Vec<[u8; 16]>>> {
    crate::cancel::check()?;
    let header = Header::decode(body, PagedIndexKind::Equality)?;
    header.require_authoritative_integrity()?;
    let values_offset = usize::try_from(header.values_offset)
        .map_err(|_| Error::invariant("paged equality values offset exceeds usize"))?;
    if body.len() < values_offset {
        return Err(Error::invariant("truncated paged equality index"));
    }
    let mut out = BTreeMap::new();
    for page_id in 0..header.leaf_count {
        if page_id as usize % crate::cancel::CHECK_STRIDE == 0 {
            crate::cancel::check()?;
        }
        let range = page_range(page_id);
        let page_start = usize::try_from(range.start)
            .map_err(|_| Error::invariant("paged equality page offset exceeds usize"))?;
        let page_end = usize::try_from(range.end)
            .map_err(|_| Error::invariant("paged equality page end exceeds usize"))?;
        let page = body
            .get(page_start..page_end)
            .ok_or_else(|| Error::invariant("truncated paged equality leaf"))?;
        for entry in parse_leaf(page, PagedIndexKind::Equality, header.format)? {
            let LeafValue::External(offset, len, checksum) = entry.value else {
                return Err(Error::invariant("equality leaf contains inline value"));
            };
            let offset = usize::try_from(offset)
                .map_err(|_| Error::invariant("equality posting offset exceeds usize"))?;
            let len = usize::try_from(len)
                .map_err(|_| Error::invariant("equality posting length exceeds usize"))?;
            let start = values_offset
                .checked_add(offset)
                .ok_or_else(|| Error::invariant("equality posting offset overflow"))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| Error::invariant("equality posting end overflow"))?;
            let value = body
                .get(start..end)
                .ok_or_else(|| Error::invariant("truncated equality posting"))?;
            if checksum.is_some_and(|expected| crc32fast::hash(value) != expected) {
                return Err(Error::invariant(
                    "equality external-value checksum mismatch",
                ));
            }
            if value.len() % 16 != 0 {
                return Err(Error::invariant("unaligned equality posting"));
            }
            let mut ids = Vec::with_capacity(value.len() / 16);
            for chunk in value.chunks_exact(16) {
                let mut id = [0; 16];
                id.copy_from_slice(chunk);
                ids.push(id);
            }
            out.insert(
                String::from_utf8(entry.key)
                    .map_err(|e| Error::invariant(format!("equality key utf8: {e}")))?,
                ids,
            );
        }
    }
    Ok(out)
}

/// Read the smallest equality keys until at least `min_postings` NodeIds have
/// been returned. Leaf `next` links make `ORDER BY ... SKIP/LIMIT` proportional
/// to its requested prefix instead of total distinct keys.
#[cfg(test)]
pub async fn equality_prefix(
    store: Arc<dyn ObjectStore>,
    path: Path,
    min_postings: usize,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    let source = LegacyObjectRangeSource { store, path };
    equality_prefix_with_source(&source, min_postings).await
}

/// Generation-pinned ordered equality prefix.
pub async fn equality_prefix_from_source(
    source: &PinnedObjectRangeSource,
    min_postings: usize,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    equality_prefix_with_source(source, min_postings).await
}

async fn equality_prefix_with_source(
    source: &dyn PagedRangeSource,
    min_postings: usize,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    type SelectedPosting = (Vec<u8>, u64, usize, usize, Option<u32>);

    crate::cancel::check()?;
    let header_bytes = source.read_range(0..HEADER_SIZE as u64).await?;
    let header = Header::decode(&header_bytes, PagedIndexKind::Equality)?;
    header.require_authoritative_integrity()?;
    let mut stats = PagedProbeStats {
        index_entries: header.entry_count,
        bytes_read: header_bytes.len(),
        ..Default::default()
    };
    if min_postings == 0 {
        return Ok((BTreeMap::new(), stats));
    }
    let mut page_id = 0u32;
    let mut selected: Vec<SelectedPosting> = Vec::new();
    let mut postings = 0usize;
    while page_id != NO_PAGE && postings < min_postings {
        crate::cancel::check()?;
        let page = source.read_range(page_range(page_id)).await?;
        stats.pages_read += 1;
        stats.bytes_read += page.len();
        let next = read_u32(&page, 8)?;
        let entries = parse_leaf(&page, PagedIndexKind::Equality, header.format)?;
        stats.leaf_entries_examined += entries.len();
        for entry in entries {
            let LeafValue::External(offset, len, checksum) = entry.value else {
                return Err(Error::invariant("equality prefix found inline value"));
            };
            if len as usize % 16 != 0 {
                return Err(Error::invariant("unaligned equality prefix posting"));
            }
            let full_ids = len as usize / 16;
            let take_ids = full_ids.min(min_postings.saturating_sub(postings));
            stats.matched_value_bytes = stats.matched_value_bytes.saturating_add(len as u64);
            stats.values_truncated |= take_ids < full_ids;
            selected.push((entry.key, offset, len as usize, take_ids * 16, checksum));
            postings += take_ids;
            if postings >= min_postings {
                break;
            }
        }
        page_id = next;
    }
    stats.values_truncated |= selected.len() < header.entry_count as usize;
    let external: Vec<(usize, Range<u64>)> = selected
        .iter()
        .enumerate()
        .filter_map(|(index, (_, offset, _, read_len, _))| {
            if *read_len == 0 {
                return None;
            }
            Some(
                external_value_range(header.values_offset, *offset, *read_len)
                    .map(|range| (index, range)),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ranges: Vec<Range<u64>> = external.iter().map(|(_, range)| range.clone()).collect();
    let values = source.read_ranges(&ranges).await?;
    if values.len() != ranges.len() {
        return Err(Error::invariant(format!(
            "paged equality range source returned {} values for {} ranges",
            values.len(),
            ranges.len()
        )));
    }
    stats.bytes_read += values.iter().map(Bytes::len).sum::<usize>();
    let mut fetched: BTreeMap<usize, Bytes> = external
        .into_iter()
        .zip(values)
        .map(|((index, _), value)| (index, value))
        .collect();
    let mut out = BTreeMap::new();
    for (index, (key, _, full_len, _, checksum)) in selected.into_iter().enumerate() {
        let value = fetched.remove(&index).unwrap_or_default();
        // A truncated prefix cannot validate the full-value CRC. Its
        // `values_truncated` bit is therefore part of the contract: callers
        // hydrate/confirm the returned NodeIds and widen or fall back if that
        // confirmed prefix under-fills the requested page. Complete values
        // are always verified here before they can be treated as exhaustive.
        if value.len() == full_len
            && checksum.is_some_and(|expected| crc32fast::hash(&value) != expected)
        {
            return Err(Error::invariant(
                "equality prefix external-value checksum mismatch",
            ));
        }
        let mut ids = Vec::with_capacity(value.len() / 16);
        for chunk in value.chunks_exact(16) {
            let mut id = [0; 16];
            id.copy_from_slice(chunk);
            ids.push(id);
        }
        out.insert(
            String::from_utf8(key)
                .map_err(|e| Error::invariant(format!("equality prefix key utf8: {e}")))?,
            ids,
        );
    }
    Ok((out, stats))
}

async fn probe_source(
    source: &dyn PagedRangeSource,
    expected: PagedIndexKind,
    keys: &[Vec<u8>],
    external_value_limit: Option<usize>,
) -> Result<(BTreeMap<Vec<u8>, Vec<u8>>, PagedProbeStats)> {
    probe_at_source(source, expected, keys, external_value_limit, 0).await
}

async fn probe_at_source(
    source: &dyn PagedRangeSource,
    expected: PagedIndexKind,
    keys: &[Vec<u8>],
    external_value_limit: Option<usize>,
    base_offset: u64,
) -> Result<(BTreeMap<Vec<u8>, Vec<u8>>, PagedProbeStats)> {
    crate::cancel::check()?;
    let header_bytes = source.read_range(header_range_at(base_offset)?).await?;
    let header = Header::decode(&header_bytes, expected)?;
    header.require_authoritative_integrity()?;
    let mut stats = PagedProbeStats {
        index_entries: header.entry_count,
        bytes_read: header_bytes.len(),
        ..Default::default()
    };
    let mut unique_keys: Vec<Vec<u8>> = keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if unique_keys.is_empty() {
        return Ok((BTreeMap::new(), stats));
    }

    let mut assignments: BTreeMap<u32, Vec<Vec<u8>>> =
        BTreeMap::from([(header.root_page, std::mem::take(&mut unique_keys))]);
    let mut found_meta: Vec<ProbeMatch> = Vec::new();
    loop {
        crate::cancel::check()?;
        let page_ids: Vec<u32> = assignments.keys().copied().collect();
        let ranges: Vec<_> = page_ids
            .iter()
            .map(|id| page_range_at(base_offset, *id))
            .collect::<Result<Vec<_>>>()?;
        let pages = source.read_ranges(&ranges).await?;
        if pages.len() != ranges.len() {
            return Err(Error::invariant(format!(
                "paged range source returned {} pages for {} ranges",
                pages.len(),
                ranges.len()
            )));
        }
        stats.pages_read += pages.len();
        stats.bytes_read += pages.iter().map(Bytes::len).sum::<usize>();
        let mut next: BTreeMap<u32, Vec<Vec<u8>>> = BTreeMap::new();
        let mut saw_internal = false;
        for ((page_id, page_keys), page) in assignments.into_iter().zip(pages) {
            if page.len() != PAGE_SIZE {
                return Err(Error::invariant(format!(
                    "short paged-index page {page_id}"
                )));
            }
            match page[0] {
                PAGE_INTERNAL => {
                    saw_internal = true;
                    let children = parse_internal(&page, header.format)?;
                    for key in page_keys {
                        let child = children
                            .iter()
                            .find(|(max, _)| key.as_slice() <= max.as_slice())
                            .or_else(|| children.last())
                            .map(|(_, child)| *child)
                            .ok_or_else(|| Error::invariant("empty paged-index internal page"))?;
                        next.entry(child).or_default().push(key);
                    }
                }
                PAGE_LEAF => {
                    let entries = parse_leaf(&page, expected, header.format)?;
                    stats.leaf_entries_examined += entries.len();
                    for key in page_keys {
                        if let Ok(pos) =
                            entries.binary_search_by(|entry| entry.key.as_slice().cmp(&key))
                        {
                            match &entries[pos].value {
                                LeafValue::Inline(value) => {
                                    found_meta.push((key, Some(value.clone()), None));
                                }
                                LeafValue::External(offset, len, checksum) => {
                                    found_meta.push((key, None, Some((*offset, *len, *checksum))));
                                }
                            }
                        }
                    }
                }
                other => {
                    return Err(Error::invariant(format!(
                        "invalid paged-index page kind {other}"
                    )));
                }
            }
        }
        if saw_internal {
            assignments = next;
        } else {
            break;
        }
    }

    let values_offset = base_offset
        .checked_add(header.values_offset)
        .ok_or_else(|| Error::invariant("paged-index values offset overflow"))?;
    let mut external: Vec<(usize, Range<u64>, Option<u32>)> = Vec::new();
    for (idx, (_, value, external_meta)) in found_meta.iter_mut().enumerate() {
        let Some((offset, len, checksum)) = *external_meta else {
            continue;
        };
        stats.matched_value_bytes = stats.matched_value_bytes.saturating_add(len as u64);
        let read_len = external_value_limit
            .map(|limit| (len as usize).min(limit))
            .unwrap_or(len as usize);
        stats.values_truncated |= read_len < len as usize;
        if read_len == 0 {
            // S3/R2 reject an empty HTTP byte range. Preserve the matched key
            // with an empty prefix and the truncation bit without issuing a
            // zero-length `get_ranges` request.
            *value = Some(Vec::new());
            continue;
        }
        external.push((
            idx,
            external_value_range(values_offset, offset, read_len)?,
            (read_len == len as usize).then_some(checksum).flatten(),
        ));
    }
    let value_ranges: Vec<_> = external.iter().map(|(_, range, _)| range.clone()).collect();
    crate::cancel::check()?;
    let values = source.read_ranges(&value_ranges).await?;
    if values.len() != value_ranges.len() {
        return Err(Error::invariant(format!(
            "paged range source returned {} values for {} ranges",
            values.len(),
            value_ranges.len()
        )));
    }
    stats.bytes_read += values.iter().map(Bytes::len).sum::<usize>();
    for ((idx, _, checksum), value) in external.into_iter().zip(values) {
        if checksum.is_some_and(|expected| crc32fast::hash(&value) != expected) {
            return Err(Error::invariant(
                "paged-index external-value checksum mismatch",
            ));
        }
        found_meta[idx].1 = Some(value.to_vec());
    }

    let found = found_meta
        .into_iter()
        .filter_map(|(key, value, _)| value.map(|value| (key, value)))
        .collect();
    Ok((found, stats))
}

#[derive(Debug)]
enum LeafValue {
    Inline(Vec<u8>),
    External(u64, u32, Option<u32>),
}

#[derive(Debug)]
struct LeafEntry {
    key: Vec<u8>,
    value: LeafValue,
}

fn parse_internal(page: &[u8], format: PagedFormat) -> Result<Vec<(Vec<u8>, u32)>> {
    validate_page(page, format)?;
    parse_internal_body(page)
}

fn parse_label_internal(
    page: &[u8],
    page_id: u32,
    binding: &LabelBinding,
) -> Result<Vec<(Vec<u8>, u32)>> {
    validate_label_page(page, page_id, binding)?;
    parse_internal_body(page)
}

fn parse_internal_body(page: &[u8]) -> Result<Vec<(Vec<u8>, u32)>> {
    let count = read_u32(page, 4)? as usize;
    let mut offset = PAGE_HEADER_SIZE;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let key_len = read_u16(page, offset)? as usize;
        offset += 2;
        let key = take(page, &mut offset, key_len)?.to_vec();
        let child = read_u32(page, offset)?;
        offset += 4;
        out.push((key, child));
    }
    Ok(out)
}

fn parse_leaf(page: &[u8], kind: PagedIndexKind, format: PagedFormat) -> Result<Vec<LeafEntry>> {
    validate_page(page, format)?;
    parse_leaf_body(page, kind, format)
}

fn parse_label_leaf(page: &[u8], page_id: u32, binding: &LabelBinding) -> Result<Vec<LeafEntry>> {
    validate_label_page(page, page_id, binding)?;
    parse_leaf_body(page, PagedIndexKind::LabelMembership, PagedFormat::V2)
}

fn parse_leaf_body(
    page: &[u8],
    kind: PagedIndexKind,
    format: PagedFormat,
) -> Result<Vec<LeafEntry>> {
    let count = read_u32(page, 4)? as usize;
    let mut offset = PAGE_HEADER_SIZE;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let key_len = read_u16(page, offset)? as usize;
        offset += 2;
        let key = take(page, &mut offset, key_len)?.to_vec();
        let value = if kind.external_values() {
            let value_offset = read_u64(page, offset)?;
            offset += 8;
            let value_len = read_u32(page, offset)?;
            offset += 4;
            let checksum = if format == PagedFormat::V2 {
                let checksum = read_u32(page, offset)?;
                offset += 4;
                Some(checksum)
            } else {
                None
            };
            LeafValue::External(value_offset, value_len, checksum)
        } else {
            let value_len = read_u16(page, offset)? as usize;
            offset += 2;
            LeafValue::Inline(take(page, &mut offset, value_len)?.to_vec())
        };
        out.push(LeafEntry { key, value });
    }
    Ok(out)
}

fn validate_label_page(page: &[u8], page_id: u32, binding: &LabelBinding) -> Result<()> {
    if page.len() != PAGE_SIZE {
        return Err(Error::invariant("short label-index page"));
    }
    let expected = read_u32(page, 12)?;
    let actual = label_page_checksum(
        page,
        binding.sst_id,
        binding.page_region_checksum,
        u64::from(page_id),
    );
    if expected != actual {
        return Err(Error::invariant(
            "label-index page/footer binding checksum mismatch",
        ));
    }
    Ok(())
}

fn validate_page(page: &[u8], format: PagedFormat) -> Result<()> {
    if page.len() != PAGE_SIZE {
        return Err(Error::invariant("short paged-index page"));
    }
    let expected = read_u32(page, 12)?;
    let actual = page_checksum(page, format);
    if expected != actual {
        return Err(Error::invariant("paged-index page checksum mismatch"));
    }
    Ok(())
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::invariant("paged-index offset overflow"))?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| Error::invariant("truncated paged-index page"))?;
    *offset = end;
    Ok(value)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let raw: [u8; 2] = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| Error::invariant("truncated paged-index u16"))?
        .try_into()
        .expect("slice length checked");
    Ok(u16::from_le_bytes(raw))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::invariant("truncated paged-index u32"))?
        .try_into()
        .expect("slice length checked");
    Ok(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw: [u8; 8] = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::invariant("truncated paged-index u64"))?
        .try_into()
        .expect("slice length checked");
    Ok(u64::from_le_bytes(raw))
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };

    use crate::range_cache::{ImmutableRangeCache, RangeCacheConfig};

    fn id(n: u128) -> [u8; 16] {
        n.to_be_bytes()
    }

    #[derive(Debug, Default)]
    struct RangeProbeState {
        range_gets: AtomicUsize,
        active: AtomicUsize,
        max_active: AtomicUsize,
        missing_generation_pin: AtomicUsize,
        corrupt_range: AtomicBool,
    }

    #[derive(Debug)]
    struct RangeProbeStore {
        inner: Arc<InMemory>,
        state: Arc<RangeProbeState>,
        delay: Duration,
    }

    impl RangeProbeStore {
        fn new(delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                inner: Arc::new(InMemory::new()),
                state: Arc::new(RangeProbeState::default()),
                delay,
            })
        }
    }

    impl fmt::Display for RangeProbeStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("PagedIndexRangeProbeStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for RangeProbeStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            if options.range.is_none() || options.head {
                return self.inner.get_opts(location, options).await;
            }

            self.state.range_gets.fetch_add(1, Ordering::SeqCst);
            if options.version.is_none() && options.if_match.is_none() {
                self.state
                    .missing_generation_pin
                    .fetch_add(1, Ordering::SeqCst);
            }
            let active = self.state.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.state.max_active.fetch_max(active, Ordering::SeqCst);
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            let result = self.inner.get_opts(location, options).await;
            self.state.active.fetch_sub(1, Ordering::SeqCst);
            let mut result = result?;
            if self.state.corrupt_range.load(Ordering::SeqCst) {
                result.range.start = result.range.start.saturating_add(1);
            }
            Ok(result)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }
    }

    async fn pinned_test_source(
        store: Arc<RangeProbeStore>,
        path: &Path,
        cache: Option<ImmutableRangeCache>,
    ) -> PinnedObjectRangeSource {
        let object_store: Arc<dyn ObjectStore> = store;
        let meta = object_store.head(path).await.unwrap();
        PinnedObjectRangeSource::from_meta_with_cache(object_store, meta, cache).unwrap()
    }

    fn page_cache_config() -> RangeCacheConfig {
        let mut config = RangeCacheConfig::memory_only(2 * 1024 * 1024);
        config.max_entry_bytes = 64 * 1024;
        config.page_size_bytes = PAGE_SIZE;
        config.key_namespace = "paged-index-tests".into();
        config
    }

    /// Pre-streaming implementation retained only as a wire-compatibility
    /// oracle. Production builders must not use it: it clones the full input.
    fn reference_build(kind: PagedIndexKind, entries: Vec<(Vec<u8>, Vec<u8>)>) -> Result<Bytes> {
        reference_build_format(kind, entries, PagedFormat::V2)
    }

    fn reference_build_format(
        kind: PagedIndexKind,
        entries: Vec<(Vec<u8>, Vec<u8>)>,
        format: PagedFormat,
    ) -> Result<Bytes> {
        let mut value_region = BytesMut::new();
        let mut leaves: Vec<(Vec<u8>, Bytes)> = Vec::new();
        if entries.is_empty() {
            leaves.push((
                Vec::new(),
                encode_page(format, PAGE_LEAF, 0, NO_PAGE, Bytes::new())?,
            ));
        }
        let mut cursor = 0usize;
        while cursor < entries.len() {
            let start = cursor;
            let mut payload = BytesMut::with_capacity(PAGE_PAYLOAD_SIZE);
            while cursor < entries.len() {
                let (key, value) = &entries[cursor];
                let key_len = u16::try_from(key.len()).unwrap();
                let record_len = if kind.external_values() {
                    (match format {
                        PagedFormat::V1 => EXTERNAL_LEAF_OVERHEAD_V1,
                        PagedFormat::V2 => EXTERNAL_LEAF_OVERHEAD_V2,
                    }) + key.len()
                } else {
                    2 + key.len() + 2 + value.len()
                };
                if payload.len() + record_len > PAGE_PAYLOAD_SIZE {
                    break;
                }
                payload.put_u16_le(key_len);
                payload.extend_from_slice(key);
                if kind.external_values() {
                    payload.put_u64_le(value_region.len() as u64);
                    payload.put_u32_le(value.len() as u32);
                    if format == PagedFormat::V2 {
                        payload.put_u32_le(crc32fast::hash(value));
                    }
                    value_region.extend_from_slice(value);
                } else {
                    payload.put_u16_le(value.len() as u16);
                    payload.extend_from_slice(value);
                }
                cursor += 1;
            }
            leaves.push((
                entries[cursor - 1].0.clone(),
                encode_page(
                    format,
                    PAGE_LEAF,
                    (cursor - start) as u32,
                    leaves.len() as u32 + 1,
                    payload.freeze(),
                )?,
            ));
        }
        if let Some((_, last)) = leaves.last_mut() {
            let count = read_u32(last, 4)?;
            *last = encode_page(
                format,
                PAGE_LEAF,
                count,
                NO_PAGE,
                last.slice(PAGE_HEADER_SIZE..PAGE_SIZE),
            )?;
        }

        let leaf_count = leaves.len() as u32;
        let mut pages: Vec<Bytes> = leaves.iter().map(|(_, page)| page.clone()).collect();
        let mut level: Vec<(Vec<u8>, u32)> = leaves
            .into_iter()
            .enumerate()
            .map(|(page, (max, _))| (max, page as u32))
            .collect();
        while level.len() > 1 {
            let mut next = Vec::new();
            let mut pos = 0usize;
            while pos < level.len() {
                let start = pos;
                let mut payload = BytesMut::with_capacity(PAGE_PAYLOAD_SIZE);
                while pos < level.len() {
                    let (max_key, child) = &level[pos];
                    let record_len = 2 + max_key.len() + 4;
                    if payload.len() + record_len > PAGE_PAYLOAD_SIZE {
                        break;
                    }
                    payload.put_u16_le(max_key.len() as u16);
                    payload.extend_from_slice(max_key);
                    payload.put_u32_le(*child);
                    pos += 1;
                }
                let page_id = pages.len() as u32;
                pages.push(encode_page(
                    format,
                    PAGE_INTERNAL,
                    (pos - start) as u32,
                    NO_PAGE,
                    payload.freeze(),
                )?);
                next.push((level[pos - 1].0.clone(), page_id));
            }
            level = next;
        }
        let page_count = pages.len() as u32;
        let header = Header {
            format,
            kind,
            root_page: level[0].1,
            page_count,
            leaf_count,
            entry_count: entries.len() as u64,
            values_offset: HEADER_SIZE as u64 + page_count as u64 * PAGE_SIZE as u64,
        };
        let mut out = BytesMut::with_capacity(header.values_offset as usize + value_region.len());
        out.extend_from_slice(&header.encode());
        for page in pages {
            out.extend_from_slice(&page);
        }
        out.extend_from_slice(&value_region);
        Ok(out.freeze())
    }

    #[test]
    fn streaming_builders_are_wire_identical_and_cover_empty_roots() {
        let unique: BTreeMap<String, [u8; 16]> = (0..2_000u128)
            .map(|n| (format!("unique-{n:08}"), id(n)))
            .collect();
        let unique_reference = reference_build(
            PagedIndexKind::Unique,
            unique
                .iter()
                .map(|(key, value)| (key.as_bytes().to_vec(), value.to_vec()))
                .collect(),
        )
        .unwrap();
        assert_eq!(build_unique(&unique).unwrap(), unique_reference);

        let equality: BTreeMap<String, Vec<[u8; 16]>> = (0..2_000u128)
            .map(|n| {
                (
                    format!("eq-{n:08}"),
                    (0..3).map(|offset| id(n * 3 + offset)).collect(),
                )
            })
            .collect();
        let equality_reference = reference_build(
            PagedIndexKind::Equality,
            equality
                .iter()
                .map(|(key, posting)| {
                    (
                        key.as_bytes().to_vec(),
                        posting.iter().flat_map(|id| id.iter().copied()).collect(),
                    )
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(build_equality(&equality).unwrap(), equality_reference);
        assert_eq!(
            decode_all_equality(&build_equality(&equality).unwrap()).unwrap(),
            equality,
            "whole-body header decode must checksum only the fixed header"
        );

        let locator_ids: Vec<_> = (0..2_000u128).map(id).collect();
        let locator_reference = reference_build(
            PagedIndexKind::NodeLocator,
            locator_ids
                .iter()
                .enumerate()
                .map(|(row, id)| (id.to_vec(), (row as u64).to_le_bytes().to_vec()))
                .collect(),
        )
        .unwrap();
        assert_eq!(
            build_node_locator(locator_ids.iter().copied()).unwrap(),
            locator_reference
        );

        let empty_unique = BTreeMap::new();
        let empty_equality = BTreeMap::new();
        assert_eq!(
            build_unique(&empty_unique).unwrap(),
            reference_build(PagedIndexKind::Unique, Vec::new()).unwrap()
        );
        assert_eq!(
            build_equality(&empty_equality).unwrap(),
            reference_build(PagedIndexKind::Equality, Vec::new()).unwrap()
        );
        assert_eq!(
            build_node_locator(Vec::new()).unwrap(),
            reference_build(PagedIndexKind::NodeLocator, Vec::new()).unwrap()
        );
    }

    #[test]
    fn node_locator_rejects_unsorted_ids() {
        let error = build_node_locator([id(2), id(1)]).unwrap_err();
        assert!(matches!(error, Error::Invariant(_)));
    }

    #[test]
    fn spooled_node_record_builder_is_wire_identical_and_keeps_values_off_heap() {
        let entries: Vec<([u8; 16], Vec<u8>)> = (0..512_u128)
            .map(|n| (id(n), vec![(n % 251) as u8; 16 * 1024 + n as usize % 97]))
            .collect();

        let mut builder = NodeLocatorRecordBuilder::new();
        for (node_id, record) in &entries {
            builder.push(node_id, record).unwrap();
        }
        let actual = builder.finish().unwrap();

        let mut expected = BytesMut::new();
        expected.extend_from_slice(
            &build_node_locator(entries.iter().map(|(node_id, _)| *node_id)).unwrap(),
        );
        expected.extend_from_slice(
            &reference_build(
                PagedIndexKind::NodeRecord,
                entries
                    .iter()
                    .map(|(node_id, record)| (node_id.to_vec(), record.clone()))
                    .collect(),
            )
            .unwrap(),
        );
        assert_eq!(actual, expected.freeze());

        let mut builder = NodeLocatorRecordBuilder::new();
        for (node_id, record) in &entries {
            builder.push(node_id, record).unwrap();
        }
        let upload = builder.finish_upload().unwrap();
        assert_eq!(upload.entry_count(), entries.len() as u64);
        assert!(
            upload.prefix.iter().map(Bytes::len).sum::<usize>() < 128 * 1024,
            "only locator/index pages should remain resident"
        );
        let file = upload
            .values
            .file
            .as_ref()
            .expect("non-empty exact records use an anonymous spool file");
        assert_eq!(file.metadata().unwrap().len(), upload.values.len);
        assert_eq!(upload.size_bytes(), actual.len() as u64);
    }

    #[tokio::test]
    async fn combined_node_records_keep_locator_prefix_compatible_and_probe_exact_values() {
        let mut builder = NodeLocatorRecordBuilder::new();
        let mut expected = BTreeMap::new();
        for n in 0..10_000u128 {
            let mut record = vec![(n % 251) as u8; 1_024 + (n as usize % 31)];
            record[..16].copy_from_slice(&id(n));
            builder.push(&id(n), &record).unwrap();
            if matches!(n, 1 | 5_001 | 9_999) {
                expected.insert(id(n), record);
            }
        }
        let body = builder.finish().unwrap();

        // The first body is an ordinary locator. Its values_offset marks its
        // exact end because locator values are inline; the appended body starts
        // with its own V2 header.
        let locator_header =
            Header::decode(&body[..HEADER_SIZE], PagedIndexKind::NodeLocator).unwrap();
        let record_start = locator_header.values_offset as usize;
        assert_eq!(&body[record_start..record_start + 8], MAGIC_V2);
        Header::decode(
            &body[record_start..record_start + HEADER_SIZE],
            PagedIndexKind::NodeRecord,
        )
        .unwrap();

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("nodes.nloc2");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();

        // An old reader sees only the compatible locator prefix.
        let probes = vec![id(1), id(5_001), id(9_999), id(20_000)];
        let (located, _) = probe_node_locator(store.clone(), path.clone(), &probes)
            .await
            .unwrap();
        assert_eq!(located.get(&id(1)), Some(&1));
        assert_eq!(located.get(&id(5_001)), Some(&5_001));
        assert_eq!(located.get(&id(9_999)), Some(&9_999));
        assert!(!located.contains_key(&id(20_000)));

        // A new reader range-fetches only requested opaque records. Duplicate
        // probes and misses do not duplicate output or trigger a full body read.
        let record_probes = vec![id(1), id(5_001), id(9_999), id(20_000), id(1)];
        let (records, stats) = probe_node_records(store, path, &record_probes)
            .await
            .unwrap();
        assert_eq!(records, expected);
        assert_eq!(stats.index_entries, 10_000);
        assert!(stats.leaf_entries_examined < 1_000);
        assert!(stats.bytes_read < body.len() / 20);
    }

    #[tokio::test]
    async fn combined_node_record_chunks_upload_without_flattening() {
        let mut builder = NodeLocatorRecordBuilder::new();
        builder.push(&id(1), b"one").unwrap();
        builder.push(&id(2), b"two").unwrap();
        let chunks = builder.finish_chunks().unwrap();

        // Locator pages, record-index pages, record-value region. The locator
        // has no external value chunk.
        assert_eq!(chunks.len(), 3);
        let locator_header = Header::decode(&chunks[0], PagedIndexKind::NodeLocator).unwrap();
        assert_eq!(locator_header.values_offset as usize, chunks[0].len());
        let record_header = Header::decode(&chunks[1], PagedIndexKind::NodeRecord).unwrap();
        assert_eq!(record_header.values_offset as usize, chunks[1].len());

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("chunked.nloc2");
        let payload: PutPayload = chunks.into_iter().collect();
        store.put(&path, payload).await.unwrap();
        let (records, _) = probe_node_records(store, path, &[id(2)]).await.unwrap();
        assert_eq!(
            records.get(&id(2)).map(Vec::as_slice),
            Some(b"two".as_slice())
        );
    }

    #[tokio::test]
    async fn node_record_probe_rejects_missing_misaligned_or_corrupt_extension() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        let locator_only = build_node_locator([id(1)]).unwrap();
        let locator_only_path = Path::from("locator-only.nloc");
        store
            .put(&locator_only_path, PutPayload::from(locator_only.clone()))
            .await
            .unwrap();
        assert!(
            probe_node_records(store.clone(), locator_only_path, &[id(1)])
                .await
                .is_err()
        );

        let mut one_record = PagedIndexBuilder::new(PagedIndexKind::NodeRecord);
        one_record.push_external(&id(1), b"one").unwrap();
        let one_record = one_record.finish().unwrap();
        let mut misaligned = BytesMut::new();
        misaligned.extend_from_slice(&build_node_locator([id(1), id(2)]).unwrap());
        misaligned.extend_from_slice(&one_record);
        let misaligned_path = Path::from("misaligned.nloc2");
        store
            .put(&misaligned_path, PutPayload::from(misaligned.freeze()))
            .await
            .unwrap();
        assert!(matches!(
            probe_node_records(store.clone(), misaligned_path, &[id(1)]).await,
            Err(Error::Invariant(_))
        ));

        let mut builder = NodeLocatorRecordBuilder::new();
        builder.push(&id(1), b"one").unwrap();
        builder.push(&id(2), b"two").unwrap();
        let mut corrupt = builder.finish().unwrap().to_vec();
        *corrupt.last_mut().unwrap() ^= 0x01;
        let corrupt_path = Path::from("corrupt.nloc2");
        store
            .put(&corrupt_path, PutPayload::from(corrupt))
            .await
            .unwrap();
        assert!(matches!(
            probe_node_records(store.clone(), corrupt_path.clone(), &[id(2)]).await,
            Err(Error::Invariant(_))
        ));
        // Corruption in the appended record value cannot poison the compatible
        // locator prefix used by old readers.
        let (located, _) = probe_node_locator(store, corrupt_path, &[id(2)])
            .await
            .unwrap();
        assert_eq!(located.get(&id(2)), Some(&1));
    }

    #[test]
    fn combined_node_record_builder_rejects_unsorted_ids() {
        let mut builder = NodeLocatorRecordBuilder::new();
        builder.push(&id(2), b"two").unwrap();
        let error = builder.push(&id(1), b"one").unwrap_err();
        assert!(matches!(error, Error::Invariant(_)));
    }

    #[tokio::test]
    async fn unique_probe_reads_search_paths_not_the_whole_index() {
        let index: BTreeMap<String, [u8; 16]> = (0..100_000u128)
            .map(|n| (format!("key-{n:012}"), id(n)))
            .collect();
        let body = build_unique(&index).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("index.pidx");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();

        let probes = vec![
            "key-000000000001".to_string(),
            "key-000000050001".to_string(),
            "key-000000099999".to_string(),
            "missing".to_string(),
        ];
        let (found, stats) = probe_unique(store, path, &probes).await.unwrap();
        assert_eq!(found.get(&probes[0]), Some(&id(1)));
        assert_eq!(found.get(&probes[1]), Some(&id(50_001)));
        assert_eq!(found.get(&probes[2]), Some(&id(99_999)));
        assert!(!found.contains_key("missing"));
        assert!(stats.leaf_entries_examined < 1_000);
        assert!(stats.bytes_read < body.len() / 20);
    }

    #[tokio::test]
    async fn locator_interleaved_probe_is_sublinear() {
        // Mirrors the reported legal corpus: ~783k total ids and 2k existing
        // MERGE candidates uniformly interleaved across the SST.
        let ids: Vec<_> = (0..783_000u128).map(|n| id(n * 17)).collect();
        let body = build_node_locator(ids.iter().copied()).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("nodes.nloc");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let probes: Vec<_> = (0..2_000usize)
            .map(|i| ids[i * (ids.len() / 2_000)])
            .collect();
        let (found, stats) = probe_node_locator(store, path, &probes).await.unwrap();
        assert_eq!(found.len(), probes.len());
        for probe in probes {
            assert_eq!(ids[found[&probe] as usize], probe);
        }
        assert!(stats.leaf_entries_examined < ids.len() / 2);
        assert!(stats.bytes_read < body.len() / 2);
    }

    #[tokio::test]
    async fn equality_fetches_only_requested_posting() {
        let index: BTreeMap<String, Vec<[u8; 16]>> = (0..10_000u128)
            .map(|n| {
                (
                    format!("v-{n:08}"),
                    (0..5).map(|m| id(n * 10 + m)).collect(),
                )
            })
            .collect();
        let body = build_equality(&index).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("eq.pidx");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let probes = vec!["v-00000042".to_string()];
        let (found, stats) = probe_equality(store, path, &probes).await.unwrap();
        assert_eq!(found[&probes[0]], index[&probes[0]]);
        assert!(stats.bytes_read < body.len() / 20);
    }

    #[tokio::test]
    async fn edge_point_batch_probe_is_range_readable_and_coalesces_duplicates() {
        let mut builder = EdgePointIndexBuilder::new();
        for n in 0..50_000u128 {
            builder
                .push(&id(n * 2), &id(n * 2 + 1), &n.to_le_bytes())
                .unwrap();
        }
        let body = builder.finish().unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edges.epidx");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let probes = vec![
            (id(2), id(3)),
            (id(50_000), id(50_001)),
            (id(99_998), id(99_999)),
            (id(2), id(4)),
            (id(2), id(3)),
        ];
        let (found, stats) = probe_edge_points(store, path, &probes).await.unwrap();
        assert_eq!(found.len(), 3, "duplicate pairs are coalesced");
        assert_eq!(found[&(id(2), id(3))], 1u128.to_le_bytes());
        assert_eq!(found[&(id(50_000), id(50_001))], 25_000u128.to_le_bytes());
        assert_eq!(found[&(id(99_998), id(99_999))], 49_999u128.to_le_bytes());
        assert_eq!(stats.index_entries, 50_000);
        assert!(stats.leaf_entries_examined < 1_000);
        assert!(stats.bytes_read < body.len() / 20);
    }

    #[test]
    fn edge_point_large_values_are_spooled_with_a_bounded_wire_identical_tree() {
        const VALUE_BYTES: usize = 257 * 1024;
        const ENTRIES: u128 = 32;

        let mut builder = EdgePointIndexBuilder::new();
        let resident_bound = builder.resident_bound_bytes();
        let mut reference = Vec::new();
        let mut expected_value_bytes = 0_u64;
        for n in 0..ENTRIES {
            let key_id = id(n * 2);
            let partner_id = id(n * 2 + 1);
            let value: Vec<u8> = (0..VALUE_BYTES)
                .map(|offset| (offset as u8).wrapping_add(n as u8))
                .collect();
            builder.push(&key_id, &partner_id, &value).unwrap();
            let mut key = Vec::with_capacity(32);
            key.extend_from_slice(&key_id);
            key.extend_from_slice(&partner_id);
            expected_value_bytes += value.len() as u64;
            reference.push((key, value));
        }

        let upload = builder.finish_upload().unwrap();
        assert_eq!(upload.entry_count(), ENTRIES as u64);
        assert!(
            resident_bound < 32 * 1024,
            "the builder keeps only page/key-sized working buffers"
        );
        assert_eq!(
            upload.spooled_page_bytes(),
            (HEADER_SIZE + PAGE_SIZE) as u64
        );
        assert_eq!(upload.spooled_value_bytes(), expected_value_bytes);
        assert!(
            upload.values.file.is_some(),
            "large exact edge values must remain in an anonymous disk spool"
        );
        assert_eq!(
            upload
                .values
                .file
                .as_ref()
                .unwrap()
                .metadata()
                .unwrap()
                .len(),
            expected_value_bytes
        );

        let actual = upload.into_bytes().unwrap();
        let expected = reference_build(PagedIndexKind::EdgePoint, reference).unwrap();
        assert_eq!(
            actual, expected,
            "spooling must not change the V2 wire body"
        );
    }

    #[test]
    fn edge_point_pages_and_internal_levels_grow_only_on_disk() {
        let mut builder = EdgePointIndexBuilder::new();
        let resident_bound = builder.resident_bound_bytes();
        for n in 0..100_000_u128 {
            builder
                .push(&id(n * 2), &id(n * 2 + 1), &[n as u8])
                .unwrap();
        }
        assert!(
            builder.resident_bound_bytes() <= resident_bound,
            "resident memory must be independent of edge count"
        );

        let upload = builder.finish_upload().unwrap();
        assert_eq!(upload.entry_count(), 100_000);
        assert!(
            upload.spooled_page_bytes() > 4 * 1024 * 1024,
            "leaf and internal page corpus belongs in the page spool"
        );
        assert_eq!(upload.spooled_value_bytes(), 100_000);
        assert!(resident_bound < 32 * 1024);
    }

    #[test]
    fn sorted_spooled_property_builders_preserve_the_existing_v2_wire() {
        let unique: BTreeMap<String, [u8; 16]> = (0..20_000_u128)
            .map(|n| (format!("key-{n:08}"), id(n)))
            .collect();
        let mut unique_builder = SortedSpooledPagedIndexBuilder::unique();
        let resident_bound = unique_builder.resident_bound_bytes();
        for (key, node) in &unique {
            unique_builder.push_inline(key.as_bytes(), node).unwrap();
        }
        assert!(
            unique_builder.resident_bound_bytes() <= resident_bound,
            "sorted sidecar memory must remain independent of entry count"
        );
        assert_eq!(
            unique_builder.finish().unwrap().into_bytes().unwrap(),
            build_unique(&unique).unwrap()
        );

        let equality: BTreeMap<String, Vec<[u8; 16]>> = (0..2_000_u128)
            .map(|n| (format!("value-{n:08}"), vec![id(n * 2), id(n * 2 + 1)]))
            .collect();
        let mut equality_builder = SortedSpooledPagedIndexBuilder::equality();
        for (key, ids) in &equality {
            let mut posting = Vec::with_capacity(ids.len() * 16);
            for node in ids {
                posting.extend_from_slice(node);
            }
            equality_builder
                .push_external(key.as_bytes(), &posting)
                .unwrap();
        }
        assert_eq!(
            equality_builder.finish().unwrap().into_bytes().unwrap(),
            build_equality(&equality).unwrap()
        );
    }

    #[test]
    fn oversized_separators_fail_deterministically_instead_of_looping() {
        let mut builder = SpooledPagedIndexBuilder::new(PagedIndexKind::EdgePoint);
        let first = vec![b'a'; 3_000];
        let mut second = vec![b'a'; 3_000];
        *second.last_mut().unwrap() = b'b';
        builder.push_external(&first, &[0]).unwrap();
        builder.push_external(&second, &[1]).unwrap();
        let error = builder.finish().unwrap_err();
        assert!(
            matches!(error, Error::Invariant(message) if message.contains("fanout")),
            "multi-page trees with fanout one must be rejected deterministically"
        );
    }

    #[tokio::test]
    async fn v2_integrity_covers_header_page_metadata_and_external_values() {
        let index: BTreeMap<String, Vec<[u8; 16]>> = (0..1_000u128)
            .map(|n| (format!("k-{n:08}"), vec![id(n)]))
            .collect();
        let original = build_equality(&index).unwrap();

        let mut bad_header = original.to_vec();
        bad_header[16] ^= 0x01; // root_page, covered by the global header CRC.
        assert!(matches!(
            decode_all_equality(&Bytes::from(bad_header)),
            Err(Error::Invariant(_))
        ));

        let mut bad_page_header = original.to_vec();
        bad_page_header[HEADER_SIZE + 8] ^= 0x01; // leaf `next`.
        assert!(matches!(
            decode_all_equality(&Bytes::from(bad_page_header)),
            Err(Error::Invariant(_))
        ));

        let mut bad_value = original.to_vec();
        *bad_value.last_mut().expect("one external posting") ^= 0x01;
        let bad_value = Bytes::from(bad_value);
        assert!(matches!(
            decode_all_equality(&bad_value),
            Err(Error::Invariant(_))
        ));
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("corrupt-external.eqidx");
        store.put(&path, PutPayload::from(bad_value)).await.unwrap();
        assert!(matches!(
            probe_equality(store.clone(), path.clone(), &["k-00000999".into()]).await,
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            equality_prefix(store, path, 1_000).await,
            Err(Error::Invariant(_))
        ));
    }

    #[tokio::test]
    async fn v1_is_recognized_but_never_authoritative_for_a_probe() {
        let body = reference_build_format(
            PagedIndexKind::Equality,
            vec![(b"k".to_vec(), id(1).to_vec())],
            PagedFormat::V1,
        )
        .unwrap();
        let header = Header::decode(&body, PagedIndexKind::Equality).unwrap();
        assert_eq!(header.format, PagedFormat::V1);
        assert!(header.require_authoritative_integrity().is_err());
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("legacy-v1.eqidx");
        store.put(&path, PutPayload::from(body)).await.unwrap();
        assert!(matches!(
            probe_equality(store, path, &["k".into()]).await,
            Err(Error::Invariant(_))
        ));
    }

    #[tokio::test]
    async fn equality_limit_does_not_download_a_low_cardinality_posting() {
        let mut index = BTreeMap::new();
        index.insert(
            "b:1".to_string(),
            (0..100_000u128).map(id).collect::<Vec<_>>(),
        );
        let body = build_equality(&index).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("bool.eqidx");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let (found, stats) = probe_equality_limited(store, path, &["b:1".into()], Some(5))
            .await
            .unwrap();
        assert_eq!(found["b:1"].len(), 5);
        assert_eq!(stats.matched_value_bytes, 100_000 * 16);
        assert!(stats.values_truncated);
        assert!(stats.bytes_read < body.len() / 100);
    }

    #[tokio::test]
    async fn zero_equality_limit_avoids_an_empty_object_store_range() {
        let mut index = BTreeMap::new();
        index.insert("b:1".to_string(), vec![id(1), id(2)]);
        let body = build_equality(&index).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("zero-limit.eqidx");
        store.put(&path, PutPayload::from(body)).await.unwrap();

        let (found, stats) = probe_equality_limited(store, path, &["b:1".into()], Some(0))
            .await
            .unwrap();
        assert_eq!(found.get("b:1"), Some(&Vec::new()));
        assert_eq!(stats.matched_value_bytes, 32);
        assert!(stats.values_truncated);
    }

    #[tokio::test]
    async fn ordered_prefix_skips_empty_posting_ranges() {
        let index = BTreeMap::from([("empty".to_string(), Vec::new())]);
        let body = build_equality(&index).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("empty-posting.eqidx");
        store.put(&path, PutPayload::from(body)).await.unwrap();
        let (found, _) = equality_prefix(store, path, 1).await.unwrap();
        assert_eq!(found.get("empty"), Some(&Vec::new()));
    }

    #[tokio::test]
    async fn corrupt_external_offsets_return_invariant_instead_of_overflowing() {
        let index = BTreeMap::from([("k".to_string(), vec![id(1)])]);
        let mut body = BytesMut::from(build_equality(&index).unwrap().as_ref());
        let page_start = HEADER_SIZE;
        let key_len = read_u16(&body, page_start + PAGE_HEADER_SIZE).unwrap() as usize;
        let external_offset = page_start + PAGE_HEADER_SIZE + 2 + key_len;
        body[external_offset..external_offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let checksum = page_checksum(&body[page_start..page_start + PAGE_SIZE], PagedFormat::V2);
        body[page_start + 12..page_start + 16].copy_from_slice(&checksum.to_le_bytes());
        let body = body.freeze();

        assert!(matches!(
            decode_all_equality(&body),
            Err(Error::Invariant(_))
        ));

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("corrupt-offset.eqidx");
        store.put(&path, PutPayload::from(body)).await.unwrap();
        assert!(matches!(
            probe_equality(store.clone(), path.clone(), &["k".into()]).await,
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            equality_prefix(store, path, 1).await,
            Err(Error::Invariant(_))
        ));
    }

    #[tokio::test]
    async fn ordered_prefix_walks_only_the_first_leaves() {
        let index: BTreeMap<String, Vec<[u8; 16]>> = (0..50_000u128)
            .map(|n| (format!("k-{n:08}"), vec![id(n)]))
            .collect();
        let body = build_equality(&index).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("ordered.eqidx");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let (prefix, stats) = equality_prefix(store, path, 1_010).await.unwrap();
        assert_eq!(prefix.len(), 1_010);
        assert_eq!(prefix.keys().next().unwrap(), "k-00000000");
        assert_eq!(prefix.keys().next_back().unwrap(), "k-00001009");
        assert!(stats.values_truncated);
        assert!(stats.bytes_read < body.len() / 10);
    }

    #[tokio::test]
    async fn label_membership_supports_point_and_cursor_pages_with_footer_binding() {
        let mut builder = SortedSpooledPagedIndexBuilder::label_membership();
        for label in [7_u32, 8] {
            let count = if label == 7 { 10_000_u128 } else { 3 };
            for node in 0..count {
                builder
                    .push_inline(&label_membership_key(label, id(node)), &[])
                    .unwrap();
            }
        }
        let counts = vec![(7, 10_000), (8, 3)];
        let sst_id = id(42);
        let body = builder
            .finish()
            .unwrap()
            .bind_label_counts(&counts, sst_id)
            .unwrap()
            .into_bytes()
            .unwrap();
        let store = RangeProbeStore::new(Duration::ZERO);
        let path = Path::from("labels/membership.pidx");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let source = pinned_test_source(store.clone(), &path, None).await;

        let (present, point_stats) =
            label_contains_from_source(&source, 7, id(9_999), sst_id, &counts)
                .await
                .unwrap();
        assert!(present);
        assert!(point_stats.pages_read < 8);
        let (absent, _) = label_contains_from_source(&source, 8, id(99), sst_id, &counts)
            .await
            .unwrap();
        assert!(!absent);
        let batch_ids = [id(1), id(9_999), id(10_000), id(1)];
        let (membership, batch_stats) =
            batch_label_contains_from_source(&source, 7, &batch_ids, sst_id, &counts)
                .await
                .unwrap();
        assert_eq!(membership, vec![true, true, false, true]);
        assert!(
            batch_stats.pages_read < batch_ids.len() * point_stats.pages_read,
            "batch probes should coalesce shared internal pages"
        );

        let (first, first_stats) =
            page_label_ids_from_source(&source, 7, None, 127, sst_id, &counts)
                .await
                .unwrap();
        assert_eq!(first.ids, (0..127).map(id).collect::<Vec<_>>());
        assert_eq!(first.next_after, Some(id(126)));
        assert!(
            first_stats.bytes_read < body.len() / 10,
            "one prefix page must not hydrate a 10k-id label"
        );
        let (second, _) =
            page_label_ids_from_source(&source, 7, first.next_after, 127, sst_id, &counts)
                .await
                .unwrap();
        assert_eq!(second.ids.first(), Some(&id(127)));
        assert_eq!(second.ids.last(), Some(&id(253)));

        let wrong_counts = vec![(7, 9_999), (8, 3)];
        assert!(matches!(
            label_contains_from_source(&source, 7, id(1), sst_id, &wrong_counts).await,
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            label_contains_from_source(&source, 7, id(1), id(43), &counts).await,
            Err(Error::Invariant(_))
        ));

        let mut swapped = body.to_vec();
        let first = swapped[HEADER_SIZE..HEADER_SIZE + PAGE_SIZE].to_vec();
        let second = swapped[HEADER_SIZE + PAGE_SIZE..HEADER_SIZE + 2 * PAGE_SIZE].to_vec();
        swapped[HEADER_SIZE..HEADER_SIZE + PAGE_SIZE].copy_from_slice(&second);
        swapped[HEADER_SIZE + PAGE_SIZE..HEADER_SIZE + 2 * PAGE_SIZE].copy_from_slice(&first);
        let swapped_path = Path::from("labels/swapped-pages.pidx");
        store
            .put(&swapped_path, PutPayload::from(swapped))
            .await
            .unwrap();
        let swapped_source = pinned_test_source(store, &swapped_path, None).await;
        assert!(matches!(
            page_label_ids_from_source(&swapped_source, 7, None, 1, sst_id, &counts).await,
            Err(Error::Invariant(_))
        ));
    }

    #[tokio::test]
    async fn pinned_probe_rejects_replaced_object_without_direct_retry() {
        let old = BTreeMap::from([("key".to_string(), id(1))]);
        let new = BTreeMap::from([("key".to_string(), id(2))]);
        let old_body = build_unique(&old).unwrap();
        let new_body = build_unique(&new).unwrap();
        assert_eq!(old_body.len(), new_body.len());

        let store = RangeProbeStore::new(Duration::ZERO);
        let path = Path::from("pinned/replaced.pidx");
        store.put(&path, PutPayload::from(old_body)).await.unwrap();
        let cache = ImmutableRangeCache::open(page_cache_config())
            .await
            .unwrap();
        let source = pinned_test_source(store.clone(), &path, Some(cache)).await;

        store.put(&path, PutPayload::from(new_body)).await.unwrap();
        let error = probe_unique_from_source(&source, &["key".into()])
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::ObjectStore(_)),
            "generation precondition must surface, got {error:?}"
        );
        assert_eq!(
            store.state.range_gets.load(Ordering::SeqCst),
            1,
            "a remote/precondition failure hidden by cache read-through must not be retried"
        );
    }

    #[tokio::test]
    async fn pinned_probe_rejects_defective_object_store_range_shape() {
        let index = BTreeMap::from([("key".to_string(), id(1))]);
        let body = build_unique(&index).unwrap();
        let store = RangeProbeStore::new(Duration::ZERO);
        let path = Path::from("pinned/defective-range.pidx");
        store.put(&path, PutPayload::from(body)).await.unwrap();
        let source = pinned_test_source(store.clone(), &path, None).await;
        store.state.corrupt_range.store(true, Ordering::SeqCst);

        let error = probe_unique_from_source(&source, &["key".into()])
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::Corrupted { .. }),
            "wrong response range must be rejected before decode, got {error:?}"
        );
        assert_eq!(store.state.range_gets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn pinned_probe_reuses_warm_shared_range_pages() {
        let index: BTreeMap<String, [u8; 16]> = (0..20_000u128)
            .map(|n| (format!("key-{n:08}"), id(n)))
            .collect();
        let body = build_unique(&index).unwrap();
        let store = RangeProbeStore::new(Duration::ZERO);
        let path = Path::from("pinned/warm.pidx");
        store.put(&path, PutPayload::from(body)).await.unwrap();
        let cache = ImmutableRangeCache::open(page_cache_config())
            .await
            .unwrap();
        let source = pinned_test_source(store.clone(), &path, Some(cache.clone())).await;
        let probes = vec![
            "key-00000001".to_string(),
            "key-00010000".to_string(),
            "key-00019999".to_string(),
        ];

        let (first, _) = probe_unique_from_source(&source, &probes).await.unwrap();
        let cold_gets = store.state.range_gets.load(Ordering::SeqCst);
        assert_eq!(first.len(), probes.len());
        assert!(cold_gets > 0);

        let (second, _) = probe_unique_from_source(&source, &probes).await.unwrap();
        assert_eq!(second, first);
        assert_eq!(
            store.state.range_gets.load(Ordering::SeqCst),
            cold_gets,
            "the identical second probe must be served entirely from warm pages"
        );
        assert!(cache.stats().memory_hits > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pinned_batches_preserve_order_and_cap_remote_concurrency() {
        let index: BTreeMap<String, [u8; 16]> = (0..100_000u128)
            .map(|n| (format!("key-{n:08}"), id(n)))
            .collect();
        let body = build_unique(&index).unwrap();
        let store = RangeProbeStore::new(Duration::from_millis(2));
        let path = Path::from("pinned/concurrency.pidx");
        store
            .put(&path, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let source = pinned_test_source(store.clone(), &path, None).await;

        let ranges = vec![8_192..8_256, 0..64, 4_096..4_160, 12_288..12_352];
        let ordered = source.read_ranges(&ranges).await.unwrap();
        assert_eq!(ordered.len(), ranges.len());
        for (range, bytes) in ranges.iter().zip(&ordered) {
            assert_eq!(
                bytes.as_ref(),
                &body[range.start as usize..range.end as usize]
            );
        }

        let probes: Vec<String> = (0..512usize)
            .map(|slot| format!("key-{:08}", slot * (100_000 / 512)))
            .collect();
        let (found, _) = probe_unique_from_source(&source, &probes).await.unwrap();
        assert_eq!(found.len(), probes.len());
        let max_active = store.state.max_active.load(Ordering::SeqCst);
        assert!(
            (2..=crate::range_cache::MAX_PINNED_OBJECT_RANGE_READS).contains(&max_active),
            "expected parallel reads bounded at 16, observed {max_active}"
        );
        assert_eq!(
            store.state.missing_generation_pin.load(Ordering::SeqCst),
            0,
            "every range GET must carry version or If-Match"
        );
    }
}
