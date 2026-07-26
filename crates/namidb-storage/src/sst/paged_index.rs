//! Range-readable immutable B+tree sidecars.
//!
//! The original property sidecars were bincode-encoded `BTreeMap`s. A point
//! probe consequently fetched and decoded every key in the map. This format
//! stores a shallow B+tree in fixed-size pages so a cold probe reads the
//! 64-byte header plus only the internal/leaf pages on its search path.
//! Unique and node-locator values are inline; potentially large equality
//! posting lists live in a value region and are fetched only for matching
//! keys.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore};

use crate::error::{Error, Result};

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
const UNIQUE_LEAF_OVERHEAD: usize = 2 + 2 + 16;
const EXTERNAL_LEAF_OVERHEAD_V1: usize = 2 + 8 + 4;
const EXTERNAL_LEAF_OVERHEAD_V2: usize = EXTERNAL_LEAF_OVERHEAD_V1 + 4;
const EQUALITY_LEAF_OVERHEAD: usize = EXTERNAL_LEAF_OVERHEAD_V2;
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
}

impl PagedIndexKind {
    fn from_byte(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Unique),
            2 => Ok(Self::Equality),
            3 => Ok(Self::NodeLocator),
            4 => Ok(Self::EdgePoint),
            _ => Err(Error::invariant(format!(
                "paged index has unknown kind {value}"
            ))),
        }
    }

    fn external_values(self) -> bool {
        matches!(self, Self::Equality | Self::EdgePoint)
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

/// Build a range-readable unique `String -> NodeId` sidecar.
pub fn build_unique(index: &BTreeMap<String, [u8; 16]>) -> Result<Bytes> {
    let mut builder = PagedIndexBuilder::new(PagedIndexKind::Unique);
    for (key, id) in index {
        builder.push_inline(key.as_bytes(), id)?;
    }
    builder.finish()
}

/// Whether every unique key can be represented in a PagedV1 leaf.
///
/// Legacy bincode remains authoritative for keys above this physical format
/// limit; callers persist an omission marker instead of failing the write.
pub fn unique_keys_fit(index: &BTreeMap<String, [u8; 16]>) -> bool {
    index
        .keys()
        .all(|key| key.len().saturating_add(UNIQUE_LEAF_OVERHEAD) <= PAGE_PAYLOAD_SIZE)
}

/// Build a range-readable equality `encoded scalar -> [NodeId]` sidecar.
pub fn build_equality(index: &BTreeMap<String, Vec<[u8; 16]>>) -> Result<Bytes> {
    let mut builder = PagedIndexBuilder::new(PagedIndexKind::Equality);
    for (key, ids) in index {
        builder.push_posting(key.as_bytes(), ids)?;
    }
    builder.finish()
}

/// Whether every equality key can be represented in a PagedV1 leaf.
pub fn equality_keys_fit(index: &BTreeMap<String, Vec<[u8; 16]>>) -> bool {
    index
        .keys()
        .all(|key| key.len().saturating_add(EQUALITY_LEAF_OVERHEAD) <= PAGE_PAYLOAD_SIZE)
}

/// Build a range-readable `NodeId -> physical row ordinal` locator.
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

/// Streaming range-readable `(key_id, partner_id) -> edge point value`
/// sidecar builder.
///
/// Edge writers already receive rows in this exact composite-key order, so
/// the B+tree can be emitted alongside the CSR without retaining another
/// graph-sized map. Values are external/variable-width because they carry the
/// row LSN, tombstone bit and encoded property map used by exact MERGE.
#[derive(Debug)]
pub struct EdgePointIndexBuilder {
    inner: PagedIndexBuilder,
}

impl EdgePointIndexBuilder {
    pub fn new() -> Self {
        Self {
            inner: PagedIndexBuilder::new(PagedIndexKind::EdgePoint),
        }
    }

    pub fn push(&mut self, key_id: &[u8; 16], partner_id: &[u8; 16], value: &[u8]) -> Result<()> {
        let mut key = [0u8; 32];
        key[..16].copy_from_slice(key_id);
        key[16..].copy_from_slice(partner_id);
        self.inner.push_external(&key, value)
    }

    pub fn finish(self) -> Result<Bytes> {
        self.inner.finish()
    }
}

impl Default for EdgePointIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
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
        self.payload.put_u64_le(self.value_region.len() as u64);
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
        self.payload.put_u16_le(key_len);
        self.payload.extend_from_slice(key);
        self.payload.put_u64_le(self.value_region.len() as u64);
        self.payload.put_u32_le(value_len);
        self.payload.put_u32_le(crc32fast::hash(value));
        self.value_region.extend_from_slice(value);
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

    fn finish(mut self) -> Result<Bytes> {
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
        let mut level: Vec<(Vec<u8>, u32)> = self
            .leaf_max_keys
            .into_iter()
            .enumerate()
            .map(|(id, max_key)| (max_key, id as u32))
            .collect();

        while level.len() > 1 {
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
        self.pages.extend_from_slice(&self.value_region);
        Ok(self.pages.freeze())
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
    let start = HEADER_SIZE as u64 + page as u64 * PAGE_SIZE as u64;
    start..start + PAGE_SIZE as u64
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

/// Probe unique keys without fetching or decoding the complete sidecar.
pub async fn probe_unique(
    store: Arc<dyn ObjectStore>,
    path: Path,
    values: &[String],
) -> Result<(BTreeMap<String, [u8; 16]>, PagedProbeStats)> {
    let keys: Vec<Vec<u8>> = values.iter().map(|v| v.as_bytes().to_vec()).collect();
    let (found, stats) = probe(store, path, PagedIndexKind::Unique, &keys, None).await?;
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
pub async fn probe_equality(
    store: Arc<dyn ObjectStore>,
    path: Path,
    values: &[String],
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    probe_equality_limited(store, path, values, None).await
}

/// Equality probe that reads at most `max_ids_per_value` ids from each
/// posting. `stats.matched_value_bytes` still reports the complete logical
/// posting size, and `values_truncated` tells the caller to continue/fallback
/// if confirmation exhausts the prefix.
pub async fn probe_equality_limited(
    store: Arc<dyn ObjectStore>,
    path: Path,
    values: &[String],
    max_ids_per_value: Option<usize>,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    let keys: Vec<Vec<u8>> = values.iter().map(|v| v.as_bytes().to_vec()).collect();
    let byte_limit = max_ids_per_value.map(|ids| ids.saturating_mul(16));
    let (found, stats) = probe(store, path, PagedIndexKind::Equality, &keys, byte_limit).await?;
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
        out.insert(
            String::from_utf8(key)
                .map_err(|e| Error::invariant(format!("equality index key utf8: {e}")))?,
            ids,
        );
    }
    Ok((out, stats))
}

/// Resolve NodeIds to exact physical row ordinals.
pub async fn probe_node_locator(
    store: Arc<dyn ObjectStore>,
    path: Path,
    ids: &[[u8; 16]],
) -> Result<(BTreeMap<[u8; 16], u64>, PagedProbeStats)> {
    let keys: Vec<Vec<u8>> = ids.iter().map(|id| id.to_vec()).collect();
    let (found, stats) = probe(store, path, PagedIndexKind::NodeLocator, &keys, None).await?;
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

/// Probe exact edge composite keys without fetching the CSR body or unrelated
/// point values. Duplicate probes are coalesced by the shared B+tree walker.
pub async fn probe_edge_points(
    store: Arc<dyn ObjectStore>,
    path: Path,
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
    let (found, stats) = probe(store, path, PagedIndexKind::EdgePoint, &keys, None).await?;
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
pub async fn equality_prefix(
    store: Arc<dyn ObjectStore>,
    path: Path,
    min_postings: usize,
) -> Result<(BTreeMap<String, Vec<[u8; 16]>>, PagedProbeStats)> {
    type SelectedPosting = (Vec<u8>, u64, usize, usize, Option<u32>);

    crate::cancel::check()?;
    let header_bytes = read_range(&store, &path, 0..HEADER_SIZE as u64).await?;
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
        let page = read_range(&store, &path, page_range(page_id)).await?;
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
    let values = read_ranges(&store, &path, &ranges).await?;
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

async fn probe(
    store: Arc<dyn ObjectStore>,
    path: Path,
    expected: PagedIndexKind,
    keys: &[Vec<u8>],
    external_value_limit: Option<usize>,
) -> Result<(BTreeMap<Vec<u8>, Vec<u8>>, PagedProbeStats)> {
    crate::cancel::check()?;
    let header_bytes = read_range(&store, &path, 0..HEADER_SIZE as u64).await?;
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
        let ranges: Vec<_> = page_ids.iter().map(|id| page_range(*id)).collect();
        let pages = read_ranges(&store, &path, &ranges).await?;
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
            external_value_range(header.values_offset, offset, read_len)?,
            (read_len == len as usize).then_some(checksum).flatten(),
        ));
    }
    let value_ranges: Vec<_> = external.iter().map(|(_, range, _)| range.clone()).collect();
    crate::cancel::check()?;
    let values = read_ranges(&store, &path, &value_ranges).await?;
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
    use super::*;
    use object_store::memory::InMemory;
    use object_store::{ObjectStoreExt, PutPayload};

    fn id(n: u128) -> [u8; 16] {
        n.to_be_bytes()
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
}
