//! Common range-readable version table for incremental vector/text segments.
//!
//! `NAMIVG06` and `NAMIFT04` carry different searchable payloads, but stale
//! candidate reconciliation is identical: for every node touched by a
//! segment, retain exactly one winning `(NodeId, LSN, operation,
//! payload_fingerprint)` record. This module provides that shared immutable
//! wire component.
//!
//! The table is designed for object storage:
//!
//! ```text
//! +------------------+-----------------------+-------------------+
//! | fixed header     | fixed version records | sparse page dir   |
//! +------------------+-----------------------+-------------------+
//! ```
//!
//! Records are grouped into independently checksummed ~24 KiB pages. Opening
//! reads only the 160-byte header and a compact directory (about 1.5 MiB for
//! ten million records); an exact point probe fetches one page. Builders write
//! pages sequentially to any `Write + Seek` target and retain only one page
//! plus the sparse directory, so the node corpus is never buffered in RAM.

use std::collections::BTreeMap;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use namidb_core::Value;

use crate::error::{Error, Result};
use crate::search_lsm::{
    SearchLsmKind, SearchLsmState, SearchSegmentFormat, SearchSegmentRef, SearchSegmentStats,
    SearchStatValue,
};

pub const SEARCH_VERSION_TABLE_MAGIC: &[u8; 8] = b"NAMISV01";
const FORMAT_VERSION: u16 = 1;
const HEADER_LEN: usize = 160;
const HEADER_CRC_OFFSET: usize = 144;
const RECORD_LEN: usize = 48;
const DIRECTORY_ENTRY_LEN: usize = 80;
const RECORDS_PER_PAGE: usize = 512;
const MAX_DIRECTORY_BYTES: usize = 64 * 1024 * 1024;
const MAX_PAGE_COUNT: usize = MAX_DIRECTORY_BYTES / DIRECTORY_ENTRY_LEN;
const MAX_POINT_PROBE_PAGES_PER_BATCH: usize = 64;
const MAX_VERIFY_PAGES_PER_BATCH: usize = 32;
const SUPPRESS_ORDINAL: u64 = u64::MAX;
const LIVE_TAG: u8 = 1;
const SUPPRESS_TAG: u8 = 2;
const SEARCH_SEGMENT_BINDING_VERSION: u16 = 1;
const SUPPRESS_FINGERPRINT_DOMAIN: &[u8] = b"NamiDB/SearchSuppress/v1";

/// Canonical scalar key used by complete native-filter bitmaps in VG6/FT4.
///
/// Float keys retain canonical IEEE bits rather than depending on a
/// locale/text representation. `-0.0` is normalized to `0.0`; NaN and
/// infinities are rejected because Cypher equality over them cannot provide a
/// stable cross-language index key. Complex/list/vector values intentionally
/// remain residual predicates.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SearchFilterValue {
    Bool(bool),
    I64(i64),
    F64Bits(u64),
    String(String),
    Bytes(Vec<u8>),
    Date(i32),
    DateTime(i64),
}

impl SearchFilterValue {
    /// Convert one storage value into the canonical native-filter domain.
    /// `None` means the value must remain a residual predicate.
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Bool(value) => Some(Self::Bool(*value)),
            Value::I64(value) => Some(Self::I64(*value)),
            Value::F64(value) if value.is_finite() => {
                let canonical = if *value == 0.0 { 0.0 } else { *value };
                Some(Self::F64Bits(canonical.to_bits()))
            }
            Value::Str(value) => Some(Self::String(value.clone())),
            Value::Bytes(value) => Some(Self::Bytes(value.clone())),
            Value::Date(value) => Some(Self::Date(*value)),
            Value::DateTime(value) => Some(Self::DateTime(*value)),
            Value::Null
            | Value::F64(_)
            | Value::Vec(_)
            | Value::VecI8 { .. }
            | Value::List(_)
            | Value::Map(_) => None,
        }
    }
}

/// Stable logical fingerprint for a suppress/tombstone operation.
///
/// The operation tag remains part of equal-LSN reconciliation, so every
/// suppression may share this domain-separated payload fingerprint without
/// confusing it with a live payload.
pub fn search_suppress_fingerprint() -> u64 {
    non_zero_xxh3(SUPPRESS_FINGERPRINT_DOMAIN)
}

/// Logical effect of one node mutation on the search index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchVersionOperation {
    /// The node remains a member. `payload_ordinal` resolves into the
    /// format-specific VG6/FT4 live-payload table.
    Live { payload_ordinal: u64 },
    /// Delete, relabel, indexed-property removal, or another transition out of
    /// membership. A suppress record shadows every older payload.
    Suppress,
}

/// One exact winner record in a search segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchVersionRecord {
    pub node_id: [u8; 16],
    pub lsn: u64,
    pub operation: SearchVersionOperation,
    /// Stable hash of the canonical after-image payload. It participates in
    /// equal-LSN conflict detection; suppressions may use their canonical
    /// tombstone/relabel fingerprint.
    pub payload_fingerprint: u64,
}

impl SearchVersionRecord {
    pub fn live(
        node_id: [u8; 16],
        lsn: u64,
        payload_fingerprint: u64,
        payload_ordinal: u64,
    ) -> Self {
        Self {
            node_id,
            lsn,
            operation: SearchVersionOperation::Live { payload_ordinal },
            payload_fingerprint,
        }
    }

    pub fn suppress(node_id: [u8; 16], lsn: u64, payload_fingerprint: u64) -> Self {
        Self {
            node_id,
            lsn,
            operation: SearchVersionOperation::Suppress,
            payload_fingerprint,
        }
    }

    fn validate(self) -> Result<()> {
        if self.lsn == 0 {
            return Err(Error::invariant(
                "search version record uses reserved LSN zero",
            ));
        }
        if matches!(
            self.operation,
            SearchVersionOperation::Live {
                payload_ordinal: SUPPRESS_ORDINAL
            }
        ) {
            return Err(Error::invariant(
                "search version live ordinal uses the suppress sentinel",
            ));
        }
        Ok(())
    }

    fn encode(self) -> [u8; RECORD_LEN] {
        let mut out = [0u8; RECORD_LEN];
        out[..16].copy_from_slice(&self.node_id);
        out[16..24].copy_from_slice(&self.lsn.to_le_bytes());
        out[24..32].copy_from_slice(&self.payload_fingerprint.to_le_bytes());
        match self.operation {
            SearchVersionOperation::Live { payload_ordinal } => {
                out[32..40].copy_from_slice(&payload_ordinal.to_le_bytes());
                out[40] = LIVE_TAG;
            }
            SearchVersionOperation::Suppress => {
                out[32..40].copy_from_slice(&SUPPRESS_ORDINAL.to_le_bytes());
                out[40] = SUPPRESS_TAG;
            }
        }
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != RECORD_LEN {
            return Err(Error::invariant(
                "search version record has the wrong fixed width",
            ));
        }
        if bytes[41..].iter().any(|byte| *byte != 0) {
            return Err(Error::invariant(
                "search version record reserved bytes are non-zero",
            ));
        }
        let node_id = bytes[..16].try_into().expect("fixed search version NodeId");
        let lsn = read_u64(bytes, 16)?;
        let payload_fingerprint = read_u64(bytes, 24)?;
        let ordinal = read_u64(bytes, 32)?;
        let operation = match bytes[40] {
            LIVE_TAG if ordinal != SUPPRESS_ORDINAL => SearchVersionOperation::Live {
                payload_ordinal: ordinal,
            },
            LIVE_TAG => {
                return Err(Error::invariant(
                    "search version live record uses suppress sentinel",
                ));
            }
            SUPPRESS_TAG if ordinal == SUPPRESS_ORDINAL => SearchVersionOperation::Suppress,
            SUPPRESS_TAG => {
                return Err(Error::invariant(
                    "search version suppress record has a payload ordinal",
                ));
            }
            tag => {
                return Err(Error::invariant(format!(
                    "search version record has unknown operation tag {tag}"
                )));
            }
        };
        let record = Self {
            node_id,
            lsn,
            operation,
            payload_fingerprint,
        };
        record.validate()?;
        Ok(record)
    }
}

/// Resolve a group of versions for one NodeId.
///
/// The highest LSN wins. Repeats with the same operation class and payload
/// fingerprint are harmless; live ordinals are segment-local and therefore
/// deliberately excluded from the tie identity. Divergent logical values are
/// rejected instead of inventing a tie-break that could disagree with the
/// authoritative node reader.
pub fn reconcile_node_versions(
    records: impl IntoIterator<Item = SearchVersionRecord>,
) -> Result<Option<SearchVersionRecord>> {
    let mut winner: Option<SearchVersionRecord> = None;
    for record in records {
        record.validate()?;
        match winner {
            None => winner = Some(record),
            Some(current) if current.node_id != record.node_id => {
                return Err(Error::invariant(
                    "cannot reconcile search versions for different NodeIds",
                ));
            }
            Some(current) if record.lsn > current.lsn => winner = Some(record),
            Some(current) if record.lsn < current.lsn => {}
            Some(current)
                if current.payload_fingerprint == record.payload_fingerprint
                    && matches!(
                        (current.operation, record.operation),
                        (
                            SearchVersionOperation::Live { .. },
                            SearchVersionOperation::Live { .. }
                        ) | (
                            SearchVersionOperation::Suppress,
                            SearchVersionOperation::Suppress
                        )
                    ) => {}
            Some(current) => {
                return Err(Error::invariant(format!(
                    "conflicting search versions for NodeId {:02x?} at LSN {}",
                    current.node_id, current.lsn
                )));
            }
        }
    }
    Ok(winner)
}

/// Footer-facing summary/reference for one embedded version table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchVersionTableRef {
    /// Absolute byte offset in the containing VG6/FT4 object.
    pub offset: u64,
    pub len: u64,
    pub record_count: u64,
    pub live_count: u64,
    pub suppress_count: u64,
    pub page_count: u32,
    pub min_node_id: [u8; 16],
    pub max_node_id: [u8; 16],
    pub min_lsn: u64,
    pub max_lsn: u64,
    /// XXH3 over the concatenated fixed-width record bytes.
    pub content_xxh3: u64,
}

impl SearchVersionTableRef {
    fn validate_basic(&self) -> Result<()> {
        if self.record_count == 0
            || self.page_count == 0
            || self.min_node_id > self.max_node_id
            || self.min_lsn == 0
            || self.min_lsn > self.max_lsn
            || self.content_xxh3 == 0
            || self.live_count.checked_add(self.suppress_count) != Some(self.record_count)
        {
            return Err(Error::invariant(
                "search version table reference has inconsistent statistics",
            ));
        }
        let expected_pages = page_count_for(self.record_count)?;
        if u64::from(self.page_count) != expected_pages {
            return Err(Error::invariant(
                "search version table reference has inconsistent page count",
            ));
        }
        let expected_len = table_len_for(self.record_count, expected_pages)?;
        if self.len != expected_len {
            return Err(Error::invariant(
                "search version table reference has inconsistent byte length",
            ));
        }
        self.offset
            .checked_add(self.len)
            .ok_or_else(|| Error::invariant("search version table object range overflows u64"))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct DirectoryEntry {
    first_node_id: [u8; 16],
    last_node_id: [u8; 16],
    first_record: u64,
    record_count: u32,
    live_count: u32,
    suppress_count: u32,
    page_crc32: u32,
    min_lsn: u64,
    max_lsn: u64,
}

impl DirectoryEntry {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.first_node_id);
        out.extend_from_slice(&self.last_node_id);
        out.extend_from_slice(&self.first_record.to_le_bytes());
        out.extend_from_slice(&self.record_count.to_le_bytes());
        out.extend_from_slice(&self.live_count.to_le_bytes());
        out.extend_from_slice(&self.suppress_count.to_le_bytes());
        out.extend_from_slice(&self.page_crc32.to_le_bytes());
        out.extend_from_slice(&self.min_lsn.to_le_bytes());
        out.extend_from_slice(&self.max_lsn.to_le_bytes());
        out.extend_from_slice(&[0; 8]);
        debug_assert_eq!(out.len() % DIRECTORY_ENTRY_LEN, 0);
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != DIRECTORY_ENTRY_LEN || bytes[72..].iter().any(|byte| *byte != 0) {
            return Err(Error::invariant(
                "search version directory entry is malformed",
            ));
        }
        Ok(Self {
            first_node_id: bytes[..16]
                .try_into()
                .expect("fixed directory first NodeId"),
            last_node_id: bytes[16..32]
                .try_into()
                .expect("fixed directory last NodeId"),
            first_record: read_u64(bytes, 32)?,
            record_count: read_u32(bytes, 40)?,
            live_count: read_u32(bytes, 44)?,
            suppress_count: read_u32(bytes, 48)?,
            page_crc32: read_u32(bytes, 52)?,
            min_lsn: read_u64(bytes, 56)?,
            max_lsn: read_u64(bytes, 64)?,
        })
    }
}

#[derive(Debug, Clone)]
struct Header {
    record_count: u64,
    live_count: u64,
    suppress_count: u64,
    page_count: u32,
    directory_offset: u64,
    directory_len: u64,
    min_lsn: u64,
    max_lsn: u64,
    min_node_id: [u8; 16],
    max_node_id: [u8; 16],
    content_xxh3: u64,
    directory_xxh3: u64,
}

impl Header {
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[..8].copy_from_slice(SEARCH_VERSION_TABLE_MAGIC);
        out[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        out[10..12].copy_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        out[12..14].copy_from_slice(&(RECORD_LEN as u16).to_le_bytes());
        out[14..16].copy_from_slice(&(DIRECTORY_ENTRY_LEN as u16).to_le_bytes());
        out[16..20].copy_from_slice(&(RECORDS_PER_PAGE as u32).to_le_bytes());
        // bytes 20..24 are flags, currently zero.
        out[24..32].copy_from_slice(&self.record_count.to_le_bytes());
        out[32..40].copy_from_slice(&self.live_count.to_le_bytes());
        out[40..48].copy_from_slice(&self.suppress_count.to_le_bytes());
        out[48..52].copy_from_slice(&self.page_count.to_le_bytes());
        // bytes 52..56 are reserved.
        out[56..64].copy_from_slice(&(HEADER_LEN as u64).to_le_bytes());
        out[64..72].copy_from_slice(&self.directory_offset.to_le_bytes());
        out[72..80].copy_from_slice(&self.directory_len.to_le_bytes());
        out[80..88].copy_from_slice(&self.min_lsn.to_le_bytes());
        out[88..96].copy_from_slice(&self.max_lsn.to_le_bytes());
        out[96..112].copy_from_slice(&self.min_node_id);
        out[112..128].copy_from_slice(&self.max_node_id);
        out[128..136].copy_from_slice(&self.content_xxh3.to_le_bytes());
        out[136..144].copy_from_slice(&self.directory_xxh3.to_le_bytes());
        let checksum = checksum_around_slot(&out, HEADER_CRC_OFFSET);
        out[HEADER_CRC_OFFSET..HEADER_CRC_OFFSET + 4].copy_from_slice(&checksum.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != HEADER_LEN || &bytes[..8] != SEARCH_VERSION_TABLE_MAGIC {
            return Err(Error::invariant(
                "search version table header magic/length mismatch",
            ));
        }
        let expected_crc = read_u32(bytes, HEADER_CRC_OFFSET)?;
        if checksum_around_slot(bytes, HEADER_CRC_OFFSET) != expected_crc {
            return Err(Error::invariant(
                "search version table header checksum mismatch",
            ));
        }
        if read_u16(bytes, 8)? != FORMAT_VERSION
            || usize::from(read_u16(bytes, 10)?) != HEADER_LEN
            || usize::from(read_u16(bytes, 12)?) != RECORD_LEN
            || usize::from(read_u16(bytes, 14)?) != DIRECTORY_ENTRY_LEN
            || read_u32(bytes, 16)? as usize != RECORDS_PER_PAGE
            || bytes[20..24].iter().any(|byte| *byte != 0)
            || bytes[52..56].iter().any(|byte| *byte != 0)
            || bytes[148..].iter().any(|byte| *byte != 0)
            || read_u64(bytes, 56)? != HEADER_LEN as u64
        {
            return Err(Error::invariant(
                "search version table header format is unsupported",
            ));
        }
        let header = Self {
            record_count: read_u64(bytes, 24)?,
            live_count: read_u64(bytes, 32)?,
            suppress_count: read_u64(bytes, 40)?,
            page_count: read_u32(bytes, 48)?,
            directory_offset: read_u64(bytes, 64)?,
            directory_len: read_u64(bytes, 72)?,
            min_lsn: read_u64(bytes, 80)?,
            max_lsn: read_u64(bytes, 88)?,
            min_node_id: bytes[96..112]
                .try_into()
                .expect("fixed header minimum NodeId"),
            max_node_id: bytes[112..128]
                .try_into()
                .expect("fixed header maximum NodeId"),
            content_xxh3: read_u64(bytes, 128)?,
            directory_xxh3: read_u64(bytes, 136)?,
        };
        let reference = header.to_reference(0)?;
        reference.validate_basic()?;
        if header.directory_xxh3 == 0
            || header.directory_len > MAX_DIRECTORY_BYTES as u64
            || header.directory_offset
                != (HEADER_LEN as u64)
                    .checked_add(
                        header
                            .record_count
                            .checked_mul(RECORD_LEN as u64)
                            .ok_or_else(|| {
                                Error::invariant(
                                    "search version table records overflow object offsets",
                                )
                            })?,
                    )
                    .ok_or_else(|| {
                        Error::invariant("search version table directory offset overflows")
                    })?
        {
            return Err(Error::invariant(
                "search version table header bounds are inconsistent",
            ));
        }
        Ok(header)
    }

    fn to_reference(&self, offset: u64) -> Result<SearchVersionTableRef> {
        let len = self
            .directory_offset
            .checked_add(self.directory_len)
            .ok_or_else(|| Error::invariant("search version table length overflows"))?;
        Ok(SearchVersionTableRef {
            offset,
            len,
            record_count: self.record_count,
            live_count: self.live_count,
            suppress_count: self.suppress_count,
            page_count: self.page_count,
            min_node_id: self.min_node_id,
            max_node_id: self.max_node_id,
            min_lsn: self.min_lsn,
            max_lsn: self.max_lsn,
            content_xxh3: self.content_xxh3,
        })
    }
}

/// Streaming writer for an embedded search version table.
pub struct SearchVersionTableWriter<W: Write + Seek> {
    writer: W,
    start_offset: u64,
    page: Vec<u8>,
    page_first_node_id: Option<[u8; 16]>,
    page_last_node_id: Option<[u8; 16]>,
    page_min_lsn: u64,
    page_max_lsn: u64,
    page_live_count: u32,
    page_suppress_count: u32,
    directory: Vec<DirectoryEntry>,
    last_node_id: Option<[u8; 16]>,
    min_node_id: Option<[u8; 16]>,
    max_node_id: Option<[u8; 16]>,
    min_lsn: u64,
    max_lsn: u64,
    record_count: u64,
    live_count: u64,
    suppress_count: u64,
    content_hasher: Xxh3,
}

impl<W: Write + Seek> std::fmt::Debug for SearchVersionTableWriter<W> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SearchVersionTableWriter")
            .field("start_offset", &self.start_offset)
            .field("record_count", &self.record_count)
            .field("live_count", &self.live_count)
            .field("suppress_count", &self.suppress_count)
            .field("completed_pages", &self.directory.len())
            .field("buffered_page_bytes", &self.page.len())
            .finish_non_exhaustive()
    }
}

impl<W: Write + Seek> SearchVersionTableWriter<W> {
    pub fn new(mut writer: W) -> Result<Self> {
        let start_offset = writer.stream_position()?;
        writer.write_all(&[0; HEADER_LEN])?;
        Ok(Self {
            writer,
            start_offset,
            page: Vec::with_capacity(RECORDS_PER_PAGE * RECORD_LEN),
            page_first_node_id: None,
            page_last_node_id: None,
            page_min_lsn: u64::MAX,
            page_max_lsn: 0,
            page_live_count: 0,
            page_suppress_count: 0,
            directory: Vec::new(),
            last_node_id: None,
            min_node_id: None,
            max_node_id: None,
            min_lsn: u64::MAX,
            max_lsn: 0,
            record_count: 0,
            live_count: 0,
            suppress_count: 0,
            content_hasher: Xxh3::new(),
        })
    }

    /// Append one already-reconciled record. Upstream external sorting must
    /// present strictly increasing NodeIds.
    pub fn push(&mut self, record: SearchVersionRecord) -> Result<()> {
        record.validate()?;
        if self
            .last_node_id
            .is_some_and(|previous| record.node_id <= previous)
        {
            return Err(Error::invariant(
                "search version records are not strictly sorted by NodeId",
            ));
        }
        if self.page.len() / RECORD_LEN == RECORDS_PER_PAGE {
            self.flush_page()?;
        }
        let encoded = record.encode();
        self.content_hasher.update(&encoded);
        self.page.extend_from_slice(&encoded);
        self.page_first_node_id.get_or_insert(record.node_id);
        self.page_last_node_id = Some(record.node_id);
        self.page_min_lsn = self.page_min_lsn.min(record.lsn);
        self.page_max_lsn = self.page_max_lsn.max(record.lsn);
        match record.operation {
            SearchVersionOperation::Live { .. } => {
                self.page_live_count = self
                    .page_live_count
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("search version page live count overflows"))?;
                self.live_count = self
                    .live_count
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("search version live count overflows"))?;
            }
            SearchVersionOperation::Suppress => {
                self.page_suppress_count =
                    self.page_suppress_count.checked_add(1).ok_or_else(|| {
                        Error::invariant("search version page suppress count overflows")
                    })?;
                self.suppress_count = self
                    .suppress_count
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("search version suppress count overflows"))?;
            }
        }
        self.record_count = self
            .record_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("search version record count overflows"))?;
        self.min_node_id.get_or_insert(record.node_id);
        self.max_node_id = Some(record.node_id);
        self.min_lsn = self.min_lsn.min(record.lsn);
        self.max_lsn = self.max_lsn.max(record.lsn);
        self.last_node_id = Some(record.node_id);
        Ok(())
    }

    pub fn finish(mut self) -> Result<(W, SearchVersionTableRef)> {
        if self.record_count == 0 {
            return Err(Error::invariant(
                "empty search delta must use ProvenEmpty coverage, not a version table",
            ));
        }
        self.flush_page()?;

        let expected_directory_offset = self
            .start_offset
            .checked_add(HEADER_LEN as u64)
            .and_then(|offset| {
                self.record_count
                    .checked_mul(RECORD_LEN as u64)
                    .and_then(|records| offset.checked_add(records))
            })
            .ok_or_else(|| Error::invariant("search version table offset overflows"))?;
        if self.writer.stream_position()? != expected_directory_offset {
            return Err(Error::invariant(
                "search version writer position disagrees with record count",
            ));
        }

        let directory_capacity = self
            .directory
            .len()
            .checked_mul(DIRECTORY_ENTRY_LEN)
            .ok_or_else(|| Error::invariant("search version directory size overflows"))?;
        if directory_capacity > MAX_DIRECTORY_BYTES {
            return Err(Error::invariant(
                "search version directory exceeds the bounded resident limit",
            ));
        }
        let mut directory_bytes = Vec::with_capacity(directory_capacity);
        for entry in &self.directory {
            entry.encode(&mut directory_bytes);
        }
        let directory_xxh3 = non_zero_xxh3(&directory_bytes);
        self.writer.write_all(&directory_bytes)?;
        let end_offset = self.writer.stream_position()?;
        let table_len = end_offset
            .checked_sub(self.start_offset)
            .ok_or_else(|| Error::invariant("search version table end precedes its start"))?;
        let content_xxh3 = self.content_hasher.digest().max(1);
        let header = Header {
            record_count: self.record_count,
            live_count: self.live_count,
            suppress_count: self.suppress_count,
            page_count: u32::try_from(self.directory.len())
                .map_err(|_| Error::invariant("search version page count exceeds u32"))?,
            directory_offset: expected_directory_offset - self.start_offset,
            directory_len: directory_bytes.len() as u64,
            min_lsn: self.min_lsn,
            max_lsn: self.max_lsn,
            min_node_id: self.min_node_id.expect("non-empty table"),
            max_node_id: self.max_node_id.expect("non-empty table"),
            content_xxh3,
            directory_xxh3,
        };
        let reference = SearchVersionTableRef {
            offset: self.start_offset,
            len: table_len,
            ..header.to_reference(self.start_offset)?
        };
        reference.validate_basic()?;
        self.writer.seek(SeekFrom::Start(self.start_offset))?;
        self.writer.write_all(&header.encode())?;
        self.writer.seek(SeekFrom::Start(end_offset))?;
        Ok((self.writer, reference))
    }

    fn flush_page(&mut self) -> Result<()> {
        if self.page.is_empty() {
            return Ok(());
        }
        if self.directory.len() >= MAX_PAGE_COUNT {
            return Err(Error::invariant(
                "search version table exceeds the bounded sparse directory",
            ));
        }
        let page_records = self.page.len() / RECORD_LEN;
        let page_records = u32::try_from(page_records)
            .map_err(|_| Error::invariant("search version page count exceeds u32"))?;
        let first_record = self
            .record_count
            .checked_sub(u64::from(page_records))
            .ok_or_else(|| Error::invariant("search version page accounting underflows"))?;
        self.writer.write_all(&self.page)?;
        self.directory.push(DirectoryEntry {
            first_node_id: self.page_first_node_id.expect("non-empty page"),
            last_node_id: self.page_last_node_id.expect("non-empty page"),
            first_record,
            record_count: page_records,
            live_count: self.page_live_count,
            suppress_count: self.page_suppress_count,
            page_crc32: crc32fast::hash(&self.page),
            min_lsn: self.page_min_lsn,
            max_lsn: self.page_max_lsn,
        });
        self.page.clear();
        self.page_first_node_id = None;
        self.page_last_node_id = None;
        self.page_min_lsn = u64::MAX;
        self.page_max_lsn = 0;
        self.page_live_count = 0;
        self.page_suppress_count = 0;
        Ok(())
    }
}

/// Convenience builder for tests and small deltas. Production builders should
/// use [`SearchVersionTableWriter`] with a file-backed spool.
pub fn encode_search_version_table(
    records: impl IntoIterator<Item = SearchVersionRecord>,
) -> Result<(Bytes, SearchVersionTableRef)> {
    let mut builder = SearchVersionTableWriter::new(Cursor::new(Vec::new()))?;
    for record in records {
        builder.push(record)?;
    }
    let (cursor, reference) = builder.finish()?;
    Ok((Bytes::from(cursor.into_inner()), reference))
}

/// Immutable range source for a containing VG6/FT4 object.
#[async_trait]
pub trait SearchVersionRangeSource: Send + Sync {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes>;

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let mut values = Vec::with_capacity(ranges.len());
        for range in ranges {
            values.push(self.read_range(range.clone()).await?);
        }
        Ok(values)
    }
}

/// Open, sparse-directory-resident version table.
pub struct SearchVersionTableReader {
    source: Arc<dyn SearchVersionRangeSource>,
    reference: SearchVersionTableRef,
    directory: Vec<DirectoryEntry>,
}

impl std::fmt::Debug for SearchVersionTableReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SearchVersionTableReader")
            .field("reference", &self.reference)
            .field("resident_pages", &self.directory.len())
            .finish_non_exhaustive()
    }
}

impl SearchVersionTableReader {
    pub async fn open(
        source: Arc<dyn SearchVersionRangeSource>,
        reference: SearchVersionTableRef,
    ) -> Result<Self> {
        reference.validate_basic()?;
        let header_end = reference
            .offset
            .checked_add(HEADER_LEN as u64)
            .ok_or_else(|| Error::invariant("search version header range overflows"))?;
        let header_bytes = source.read_range(reference.offset..header_end).await?;
        require_exact_len(&header_bytes, HEADER_LEN, "header")?;
        let header = Header::decode(&header_bytes)?;
        if header.to_reference(reference.offset)? != reference {
            return Err(Error::invariant(
                "search version header disagrees with its outer footer reference",
            ));
        }

        let directory_start = reference
            .offset
            .checked_add(header.directory_offset)
            .ok_or_else(|| Error::invariant("search version directory start overflows"))?;
        let directory_end = directory_start
            .checked_add(header.directory_len)
            .ok_or_else(|| Error::invariant("search version directory end overflows"))?;
        if directory_end
            != reference
                .offset
                .checked_add(reference.len)
                .ok_or_else(|| Error::invariant("search version table end overflows"))?
        {
            return Err(Error::invariant(
                "search version directory does not end at table boundary",
            ));
        }
        let directory_bytes = source.read_range(directory_start..directory_end).await?;
        require_exact_len(
            &directory_bytes,
            usize::try_from(header.directory_len)
                .map_err(|_| Error::invariant("search version directory does not fit usize"))?,
            "directory",
        )?;
        if non_zero_xxh3(&directory_bytes) != header.directory_xxh3 {
            return Err(Error::invariant(
                "search version directory checksum mismatch",
            ));
        }
        let directory = decode_directory(&directory_bytes, &header)?;
        Ok(Self {
            source,
            reference,
            directory,
        })
    }

    pub fn reference(&self) -> &SearchVersionTableRef {
        &self.reference
    }

    /// Number of independently checksummed record pages.
    ///
    /// Compaction cursors use this together with [`Self::read_page`] to retain
    /// at most one bounded page per input run while performing a k-way merge.
    pub fn page_count(&self) -> usize {
        self.directory.len()
    }

    /// Read and validate one complete record page by its stable directory
    /// ordinal.
    ///
    /// The returned records are strictly NodeId ordered and contain at most
    /// [`RECORDS_PER_PAGE`] entries. This is deliberately a page API rather
    /// than an unbounded iterator: callers can account the exact resident
    /// workspace and can retry an object-store range read deterministically.
    pub async fn read_page(&self, page: usize) -> Result<Vec<SearchVersionRecord>> {
        let entry = self.directory.get(page).ok_or_else(|| {
            Error::precondition(format!(
                "search version page {page} is outside page count {}",
                self.directory.len()
            ))
        })?;
        let range = self.page_range(entry)?;
        let body = self.source.read_range(range.clone()).await?;
        require_exact_len(
            &body,
            usize::try_from(range.end - range.start)
                .map_err(|_| Error::invariant("search version page length overflows"))?,
            "page",
        )?;
        decode_page(&body, entry)
    }

    /// Resident bytes attributable to the sparse directory, excluding the
    /// shared object/range cache.
    pub fn resident_metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.directory.capacity() * std::mem::size_of::<DirectoryEntry>())
    }

    pub async fn point_probe(&self, node_id: [u8; 16]) -> Result<Option<SearchVersionRecord>> {
        let mut values = self.point_probe_many(&[node_id]).await?;
        Ok(values.pop().flatten())
    }

    /// Batch exact probes. NodeIds targeting the same page share one range
    /// read, and page fetches are bounded in groups to cap transient memory.
    pub async fn point_probe_many(
        &self,
        node_ids: &[[u8; 16]],
    ) -> Result<Vec<Option<SearchVersionRecord>>> {
        let mut output = vec![None; node_ids.len()];
        let mut by_page: BTreeMap<usize, Vec<(usize, [u8; 16])>> = BTreeMap::new();
        for (position, node_id) in node_ids.iter().copied().enumerate() {
            if let Some(page) = self.find_page(node_id) {
                by_page.entry(page).or_default().push((position, node_id));
            }
        }
        let groups = by_page.into_iter().collect::<Vec<_>>();
        for chunk in groups.chunks(MAX_POINT_PROBE_PAGES_PER_BATCH) {
            let ranges = chunk
                .iter()
                .map(|(page, _)| self.page_range(&self.directory[*page]))
                .collect::<Result<Vec<_>>>()?;
            let bodies = self.source.read_ranges(&ranges).await?;
            if bodies.len() != chunk.len() {
                return Err(Error::invariant(
                    "search version range source returned the wrong page count",
                ));
            }
            for (((page, probes), body), range) in chunk.iter().zip(bodies).zip(ranges.iter()) {
                require_exact_len(
                    &body,
                    usize::try_from(range.end - range.start)
                        .map_err(|_| Error::invariant("search version page length overflows"))?,
                    "page",
                )?;
                let records = decode_page(&body, &self.directory[*page])?;
                for (position, node_id) in probes {
                    if let Ok(found) =
                        records.binary_search_by_key(node_id, |record| record.node_id)
                    {
                        output[*position] = Some(records[found]);
                    }
                }
            }
        }
        Ok(output)
    }

    /// Scrub every page and verify the footer-facing semantic content hash
    /// without ever materializing the complete table.
    pub async fn verify_all(&self) -> Result<()> {
        let mut hasher = Xxh3::new();
        for first in (0..self.directory.len()).step_by(MAX_VERIFY_PAGES_PER_BATCH) {
            let end = (first + MAX_VERIFY_PAGES_PER_BATCH).min(self.directory.len());
            let entries = &self.directory[first..end];
            let ranges = entries
                .iter()
                .map(|entry| self.page_range(entry))
                .collect::<Result<Vec<_>>>()?;
            let bodies = self.source.read_ranges(&ranges).await?;
            if bodies.len() != entries.len() {
                return Err(Error::invariant(
                    "search version range source returned the wrong scrub page count",
                ));
            }
            for ((entry, body), range) in entries.iter().zip(bodies).zip(ranges.iter()) {
                require_exact_len(
                    &body,
                    usize::try_from(range.end - range.start)
                        .map_err(|_| Error::invariant("search version page length overflows"))?,
                    "page",
                )?;
                decode_page(&body, entry)?;
                hasher.update(&body);
            }
        }
        if hasher.digest().max(1) != self.reference.content_xxh3 {
            return Err(Error::invariant(
                "search version semantic content checksum mismatch",
            ));
        }
        Ok(())
    }

    fn find_page(&self, node_id: [u8; 16]) -> Option<usize> {
        let mut low = 0usize;
        let mut high = self.directory.len();
        while low < high {
            let middle = low + (high - low) / 2;
            if self.directory[middle].last_node_id < node_id {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        self.directory.get(low).and_then(|entry| {
            (entry.first_node_id <= node_id && node_id <= entry.last_node_id).then_some(low)
        })
    }

    fn page_range(&self, entry: &DirectoryEntry) -> Result<Range<u64>> {
        let relative_start = (HEADER_LEN as u64)
            .checked_add(
                entry
                    .first_record
                    .checked_mul(RECORD_LEN as u64)
                    .ok_or_else(|| Error::invariant("search version page offset overflows"))?,
            )
            .ok_or_else(|| Error::invariant("search version page start overflows"))?;
        let relative_end = relative_start
            .checked_add(u64::from(entry.record_count) * RECORD_LEN as u64)
            .ok_or_else(|| Error::invariant("search version page end overflows"))?;
        let table_end = self
            .reference
            .offset
            .checked_add(self.reference.len)
            .ok_or_else(|| Error::invariant("search version table end overflows"))?;
        let start = self
            .reference
            .offset
            .checked_add(relative_start)
            .ok_or_else(|| Error::invariant("search version absolute page start overflows"))?;
        let end = self
            .reference
            .offset
            .checked_add(relative_end)
            .ok_or_else(|| Error::invariant("search version absolute page end overflows"))?;
        if start < self.reference.offset || end > table_end || start >= end {
            return Err(Error::invariant(
                "search version page range leaves the containing table",
            ));
        }
        Ok(start..end)
    }
}

/// Common lineage block that VG6/FT4 repeat in their native footer.
///
/// The format-specific footer adds vector/text directories and statistics;
/// this block binds those payloads to the manifest generation and exact
/// version table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchSegmentWireBinding {
    pub format_version: u16,
    pub generation_id: Uuid,
    pub index_name: String,
    pub catalog_signature: String,
    pub segment: SearchSegmentRef,
    pub version_table: SearchVersionTableRef,
}

// Manifest-facing search enums intentionally use internally tagged JSON for
// compatibility/readability. Bincode cannot decode such enums because it does
// not implement `deserialize_any`, so the native footer uses this explicit
// external-tagged mirror instead of serializing `SearchSegmentRef` directly.
#[derive(Serialize, Deserialize)]
struct SearchSegmentWireBindingSerde {
    format_version: u16,
    generation_id: Uuid,
    index_name: String,
    catalog_signature: String,
    segment: SearchSegmentRefSerde,
    version_table: SearchVersionTableRef,
}

#[derive(Serialize, Deserialize)]
struct SearchSegmentRefSerde {
    sst_id: Uuid,
    role: crate::search_lsm::SearchSegmentRole,
    format: SearchSegmentFormat,
    payload: crate::search_lsm::SearchSegmentPayload,
    event_ranges: Vec<crate::search_lsm::SearchEventRange>,
    min_lsn: u64,
    max_lsn: u64,
    mutation_count: u64,
    live_payload_count: u64,
    suppress_count: u64,
    content_xxh3: u64,
    complete_filter_properties: Vec<String>,
    stats: SearchSegmentStatsSerde,
    equal_lsn_conflict_count: u64,
}

#[derive(Serialize, Deserialize)]
enum SearchStatValueSerde {
    Absolute(u64),
    Delta(i64),
}

#[derive(Serialize, Deserialize)]
enum SearchSegmentStatsSerde {
    Unknown,
    Vector {
        live_count: SearchStatValueSerde,
    },
    Text {
        doc_count: SearchStatValueSerde,
        total_len: SearchStatValueSerde,
        term_df_violation_count: u64,
    },
}

impl Serialize for SearchSegmentWireBinding {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        SearchSegmentWireBindingSerde::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SearchSegmentWireBinding {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(SearchSegmentWireBindingSerde::deserialize(deserializer)?.into())
    }
}

impl From<SearchSegmentWireBinding> for SearchSegmentWireBindingSerde {
    fn from(binding: SearchSegmentWireBinding) -> Self {
        Self {
            format_version: binding.format_version,
            generation_id: binding.generation_id,
            index_name: binding.index_name,
            catalog_signature: binding.catalog_signature,
            segment: binding.segment.into(),
            version_table: binding.version_table,
        }
    }
}

impl From<SearchSegmentWireBindingSerde> for SearchSegmentWireBinding {
    fn from(binding: SearchSegmentWireBindingSerde) -> Self {
        Self {
            format_version: binding.format_version,
            generation_id: binding.generation_id,
            index_name: binding.index_name,
            catalog_signature: binding.catalog_signature,
            segment: binding.segment.into(),
            version_table: binding.version_table,
        }
    }
}

impl From<SearchSegmentRef> for SearchSegmentRefSerde {
    fn from(segment: SearchSegmentRef) -> Self {
        Self {
            sst_id: segment.sst_id,
            role: segment.role,
            format: segment.format,
            payload: segment.payload,
            event_ranges: segment.event_ranges,
            min_lsn: segment.min_lsn,
            max_lsn: segment.max_lsn,
            mutation_count: segment.mutation_count,
            live_payload_count: segment.live_payload_count,
            suppress_count: segment.suppress_count,
            content_xxh3: segment.content_xxh3,
            complete_filter_properties: segment.complete_filter_properties,
            stats: segment.stats.into(),
            equal_lsn_conflict_count: segment.equal_lsn_conflict_count,
        }
    }
}

impl From<SearchSegmentRefSerde> for SearchSegmentRef {
    fn from(segment: SearchSegmentRefSerde) -> Self {
        Self {
            sst_id: segment.sst_id,
            role: segment.role,
            format: segment.format,
            payload: segment.payload,
            event_ranges: segment.event_ranges,
            min_lsn: segment.min_lsn,
            max_lsn: segment.max_lsn,
            mutation_count: segment.mutation_count,
            live_payload_count: segment.live_payload_count,
            suppress_count: segment.suppress_count,
            content_xxh3: segment.content_xxh3,
            complete_filter_properties: segment.complete_filter_properties,
            stats: segment.stats.into(),
            equal_lsn_conflict_count: segment.equal_lsn_conflict_count,
        }
    }
}

impl From<SearchStatValue> for SearchStatValueSerde {
    fn from(value: SearchStatValue) -> Self {
        match value {
            SearchStatValue::Absolute(value) => Self::Absolute(value),
            SearchStatValue::Delta(value) => Self::Delta(value),
        }
    }
}

impl From<SearchStatValueSerde> for SearchStatValue {
    fn from(value: SearchStatValueSerde) -> Self {
        match value {
            SearchStatValueSerde::Absolute(value) => Self::Absolute(value),
            SearchStatValueSerde::Delta(value) => Self::Delta(value),
        }
    }
}

impl From<SearchSegmentStats> for SearchSegmentStatsSerde {
    fn from(stats: SearchSegmentStats) -> Self {
        match stats {
            SearchSegmentStats::Unknown => Self::Unknown,
            SearchSegmentStats::Vector { live_count } => Self::Vector {
                live_count: live_count.into(),
            },
            SearchSegmentStats::Text {
                doc_count,
                total_len,
                term_df_violation_count,
            } => Self::Text {
                doc_count: doc_count.into(),
                total_len: total_len.into(),
                term_df_violation_count,
            },
        }
    }
}

impl From<SearchSegmentStatsSerde> for SearchSegmentStats {
    fn from(stats: SearchSegmentStatsSerde) -> Self {
        match stats {
            SearchSegmentStatsSerde::Unknown => Self::Unknown,
            SearchSegmentStatsSerde::Vector { live_count } => Self::Vector {
                live_count: live_count.into(),
            },
            SearchSegmentStatsSerde::Text {
                doc_count,
                total_len,
                term_df_violation_count,
            } => Self::Text {
                doc_count: doc_count.into(),
                total_len: total_len.into(),
                term_df_violation_count,
            },
        }
    }
}

impl SearchSegmentWireBinding {
    pub fn new(
        state: &SearchLsmState,
        segment: &SearchSegmentRef,
        version_table: SearchVersionTableRef,
    ) -> Result<Self> {
        let binding = Self {
            format_version: SEARCH_SEGMENT_BINDING_VERSION,
            generation_id: state.generation_id,
            index_name: state.index_name.clone(),
            catalog_signature: state.catalog_signature.clone(),
            segment: segment.clone(),
            version_table,
        };
        binding.validate(state, segment)?;
        Ok(binding)
    }

    pub fn validate(&self, state: &SearchLsmState, segment: &SearchSegmentRef) -> Result<()> {
        let expected_kind = match segment.format {
            SearchSegmentFormat::VectorV6 => SearchLsmKind::Vector,
            SearchSegmentFormat::TextV4 => SearchLsmKind::Text,
            _ => {
                return Err(Error::invariant(
                    "common delta footer can bind only VG6/FT4 segments",
                ));
            }
        };
        if self.format_version != SEARCH_SEGMENT_BINDING_VERSION
            || state.kind != expected_kind
            || self.generation_id != state.generation_id
            || self.index_name != state.index_name
            || self.catalog_signature != state.catalog_signature
            || &self.segment != segment
        {
            return Err(Error::invariant(
                "search segment wire footer disagrees with manifest lineage",
            ));
        }
        self.version_table.validate_basic()?;
        if self.version_table.record_count != segment.mutation_count
            || self.version_table.live_count != segment.live_payload_count
            || self.version_table.suppress_count != segment.suppress_count
            || self.version_table.min_lsn != segment.min_lsn
            || self.version_table.max_lsn != segment.max_lsn
        {
            return Err(Error::invariant(
                "search segment wire footer disagrees with version-table statistics",
            ));
        }
        Ok(())
    }
}

fn decode_directory(bytes: &[u8], header: &Header) -> Result<Vec<DirectoryEntry>> {
    if bytes.len() % DIRECTORY_ENTRY_LEN != 0
        || bytes.len() / DIRECTORY_ENTRY_LEN != header.page_count as usize
    {
        return Err(Error::invariant(
            "search version directory entry count is inconsistent",
        ));
    }
    let mut entries = Vec::with_capacity(header.page_count as usize);
    let mut next_record = 0u64;
    let mut live_count = 0u64;
    let mut suppress_count = 0u64;
    let mut min_lsn = u64::MAX;
    let mut max_lsn = 0u64;
    let mut previous_last: Option<[u8; 16]> = None;
    for chunk in bytes.chunks_exact(DIRECTORY_ENTRY_LEN) {
        let entry = DirectoryEntry::decode(chunk)?;
        if entry.record_count == 0
            || entry.record_count as usize > RECORDS_PER_PAGE
            || entry.first_record != next_record
            || entry.first_node_id > entry.last_node_id
            || entry.min_lsn == 0
            || entry.min_lsn > entry.max_lsn
            || entry.live_count.checked_add(entry.suppress_count) != Some(entry.record_count)
            || previous_last.is_some_and(|last| last >= entry.first_node_id)
        {
            return Err(Error::invariant(
                "search version directory is not a valid ordered partition",
            ));
        }
        next_record = next_record
            .checked_add(u64::from(entry.record_count))
            .ok_or_else(|| Error::invariant("search version directory count overflows"))?;
        live_count = live_count
            .checked_add(u64::from(entry.live_count))
            .ok_or_else(|| Error::invariant("search version directory live count overflows"))?;
        suppress_count = suppress_count
            .checked_add(u64::from(entry.suppress_count))
            .ok_or_else(|| Error::invariant("search version directory suppress count overflows"))?;
        min_lsn = min_lsn.min(entry.min_lsn);
        max_lsn = max_lsn.max(entry.max_lsn);
        previous_last = Some(entry.last_node_id);
        entries.push(entry);
    }
    if next_record != header.record_count
        || live_count != header.live_count
        || suppress_count != header.suppress_count
        || min_lsn != header.min_lsn
        || max_lsn != header.max_lsn
        || entries
            .first()
            .is_none_or(|entry| entry.first_node_id != header.min_node_id)
        || entries
            .last()
            .is_none_or(|entry| entry.last_node_id != header.max_node_id)
    {
        return Err(Error::invariant(
            "search version directory disagrees with header statistics",
        ));
    }
    Ok(entries)
}

fn decode_page(bytes: &[u8], entry: &DirectoryEntry) -> Result<Vec<SearchVersionRecord>> {
    let expected_len = entry.record_count as usize * RECORD_LEN;
    if bytes.len() != expected_len || crc32fast::hash(bytes) != entry.page_crc32 {
        return Err(Error::invariant(
            "search version page length/checksum mismatch",
        ));
    }
    let mut records = Vec::with_capacity(entry.record_count as usize);
    let mut previous: Option<[u8; 16]> = None;
    let mut live_count = 0u32;
    let mut suppress_count = 0u32;
    let mut min_lsn = u64::MAX;
    let mut max_lsn = 0u64;
    for encoded in bytes.chunks_exact(RECORD_LEN) {
        let record = SearchVersionRecord::decode(encoded)?;
        if previous.is_some_and(|node_id| node_id >= record.node_id) {
            return Err(Error::invariant(
                "search version page records are not strictly sorted",
            ));
        }
        match record.operation {
            SearchVersionOperation::Live { .. } => live_count += 1,
            SearchVersionOperation::Suppress => suppress_count += 1,
        }
        min_lsn = min_lsn.min(record.lsn);
        max_lsn = max_lsn.max(record.lsn);
        previous = Some(record.node_id);
        records.push(record);
    }
    if records.first().map(|record| record.node_id) != Some(entry.first_node_id)
        || records.last().map(|record| record.node_id) != Some(entry.last_node_id)
        || live_count != entry.live_count
        || suppress_count != entry.suppress_count
        || min_lsn != entry.min_lsn
        || max_lsn != entry.max_lsn
    {
        return Err(Error::invariant(
            "search version page disagrees with its sparse directory entry",
        ));
    }
    Ok(records)
}

fn page_count_for(record_count: u64) -> Result<u64> {
    if record_count == 0 {
        return Ok(0);
    }
    let pages = record_count
        .checked_add(RECORDS_PER_PAGE as u64 - 1)
        .ok_or_else(|| Error::invariant("search version page count overflows"))?
        / RECORDS_PER_PAGE as u64;
    if pages > MAX_PAGE_COUNT as u64 || pages > u32::MAX as u64 {
        return Err(Error::invariant(
            "search version table exceeds the bounded sparse directory",
        ));
    }
    Ok(pages)
}

fn table_len_for(record_count: u64, page_count: u64) -> Result<u64> {
    (HEADER_LEN as u64)
        .checked_add(
            record_count
                .checked_mul(RECORD_LEN as u64)
                .ok_or_else(|| Error::invariant("search version record bytes overflow"))?,
        )
        .and_then(|value| {
            page_count
                .checked_mul(DIRECTORY_ENTRY_LEN as u64)
                .and_then(|directory| value.checked_add(directory))
        })
        .ok_or_else(|| Error::invariant("search version table length overflows"))
}

fn checksum_around_slot(bytes: &[u8], slot: usize) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..slot]);
    hasher.update(&[0; 4]);
    hasher.update(&bytes[slot + 4..]);
    hasher.finalize()
}

fn non_zero_xxh3(bytes: &[u8]) -> u64 {
    xxh3_64(bytes).max(1)
}

fn require_exact_len(bytes: &[u8], expected: usize, what: &str) -> Result<()> {
    if bytes.len() != expected {
        return Err(Error::invariant(format!(
            "search version {what} range returned {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| Error::invariant("search version u16 offset overflows"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invariant("search version u16 is out of bounds"))?;
    Ok(u16::from_le_bytes(
        slice.try_into().expect("checked u16 slice"),
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error::invariant("search version u32 offset overflows"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invariant("search version u32 is out of bounds"))?;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("checked u32 slice"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::invariant("search version u64 offset overflows"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| Error::invariant("search version u64 is out of bounds"))?;
    Ok(u64::from_le_bytes(
        slice.try_into().expect("checked u64 slice"),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use bincode::Options as _;

    use super::*;
    use crate::search_lsm::{
        SearchEventRange, SearchLsmStatus, SearchSegmentPayload, SearchSegmentRole,
        SearchSegmentStats, SearchStatValue,
    };

    #[derive(Debug)]
    struct MemorySource {
        body: Bytes,
        ranges: Mutex<Vec<Range<u64>>>,
    }

    impl MemorySource {
        fn new(body: Bytes) -> Self {
            Self {
                body,
                ranges: Mutex::new(Vec::new()),
            }
        }

        fn bytes_read(&self) -> u64 {
            self.ranges
                .lock()
                .unwrap()
                .iter()
                .map(|range| range.end - range.start)
                .sum()
        }
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
                .ok_or_else(|| Error::invariant("test range leaves body"))
        }

        async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
            let mut values = Vec::with_capacity(ranges.len());
            for range in ranges {
                values.push(self.read_range(range.clone()).await?);
            }
            Ok(values)
        }
    }

    fn node(value: u64) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[8..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn records(count: u64) -> Vec<SearchVersionRecord> {
        (1..=count)
            .map(|value| {
                if value % 5 == 0 {
                    SearchVersionRecord::suppress(node(value), value + 10, value * 17)
                } else {
                    SearchVersionRecord::live(node(value), value + 10, value * 17, value - 1)
                }
            })
            .collect()
    }

    #[test]
    fn reconciliation_uses_highest_lsn_and_rejects_conflicting_ties() {
        let id = node(7);
        let old = SearchVersionRecord::live(id, 10, 1, 0);
        let new = SearchVersionRecord::suppress(id, 20, 2);
        assert_eq!(reconcile_node_versions([new, old, new]).unwrap(), Some(new));
        assert_eq!(
            reconcile_node_versions([
                SearchVersionRecord::live(id, 30, 3, 7),
                SearchVersionRecord::live(id, 30, 3, 99),
            ])
            .unwrap(),
            Some(SearchVersionRecord::live(id, 30, 3, 7)),
            "payload ordinals are local to their source segment"
        );

        let conflict = SearchVersionRecord::live(id, 20, 2, 0);
        assert!(reconcile_node_versions([new, conflict])
            .unwrap_err()
            .to_string()
            .contains("conflicting"));
        assert!(
            reconcile_node_versions([old, SearchVersionRecord::live(node(8), 30, 3, 1)]).is_err()
        );
    }

    #[test]
    fn writer_is_streaming_embeddable_and_rejects_invalid_order() {
        let prefix = b"vg6-prefix".to_vec();
        let mut cursor = Cursor::new(prefix.clone());
        cursor.set_position(prefix.len() as u64);
        let mut writer = SearchVersionTableWriter::new(cursor).unwrap();
        writer
            .push(SearchVersionRecord::live(node(1), 1, 11, 0))
            .unwrap();
        assert!(writer
            .push(SearchVersionRecord::suppress(node(1), 2, 22))
            .is_err());

        let mut cursor = Cursor::new(prefix.clone());
        cursor.set_position(prefix.len() as u64);
        let mut writer = SearchVersionTableWriter::new(cursor).unwrap();
        writer
            .push(SearchVersionRecord::live(node(1), 1, 11, 0))
            .unwrap();
        let (cursor, reference) = writer.finish().unwrap();
        assert_eq!(reference.offset, prefix.len() as u64);
        assert_eq!(
            &cursor.into_inner()[prefix.len()..prefix.len() + SEARCH_VERSION_TABLE_MAGIC.len()],
            SEARCH_VERSION_TABLE_MAGIC
        );

        let empty = SearchVersionTableWriter::new(Cursor::new(Vec::new()))
            .unwrap()
            .finish()
            .unwrap_err();
        assert!(empty.to_string().contains("ProvenEmpty"));
    }

    #[tokio::test]
    async fn ranged_open_probe_batch_and_scrub_are_exact() {
        let input = records(2_000);
        let (body, reference) = encode_search_version_table(input.clone()).unwrap();
        let source = Arc::new(MemorySource::new(body.clone()));
        let reader = SearchVersionTableReader::open(source.clone(), reference.clone())
            .await
            .unwrap();
        assert_eq!(reader.reference(), &reference);
        assert_eq!(reference.page_count, 4);
        assert!(
            source.bytes_read() < body.len() as u64 / 10,
            "open must retain only header + sparse directory"
        );

        assert_eq!(
            reader.point_probe(node(1_337)).await.unwrap(),
            Some(input[1_336])
        );
        assert_eq!(reader.point_probe(node(9_999)).await.unwrap(), None);
        let batch = reader
            .point_probe_many(&[node(1), node(512), node(513), node(1), node(2_000)])
            .await
            .unwrap();
        assert_eq!(
            batch,
            vec![
                Some(input[0]),
                Some(input[511]),
                Some(input[512]),
                Some(input[0]),
                Some(input[1_999]),
            ]
        );
        reader.verify_all().await.unwrap();
    }

    #[tokio::test]
    async fn corruption_is_detected_at_header_directory_and_page_boundaries() {
        let input = records(700);
        let (body, reference) = encode_search_version_table(input).unwrap();

        let mut bad_header = body.to_vec();
        bad_header[24] ^= 1;
        assert!(SearchVersionTableReader::open(
            Arc::new(MemorySource::new(Bytes::from(bad_header))),
            reference.clone(),
        )
        .await
        .is_err());

        let directory_start = HEADER_LEN + reference.record_count as usize * RECORD_LEN;
        let mut bad_directory = body.to_vec();
        bad_directory[directory_start] ^= 1;
        assert!(SearchVersionTableReader::open(
            Arc::new(MemorySource::new(Bytes::from(bad_directory))),
            reference.clone(),
        )
        .await
        .is_err());

        let mut bad_page = body.to_vec();
        bad_page[HEADER_LEN + 1] ^= 1;
        let reader = SearchVersionTableReader::open(
            Arc::new(MemorySource::new(Bytes::from(bad_page))),
            reference,
        )
        .await
        .unwrap();
        assert!(reader.point_probe(node(1)).await.is_err());
    }

    #[test]
    fn common_wire_binding_repeats_manifest_lineage_and_table_stats() {
        let (body, version_table) = encode_search_version_table(records(3)).unwrap();
        assert!(!body.is_empty());
        let state = SearchLsmState {
            index_name: "doc_vec".into(),
            kind: SearchLsmKind::Vector,
            catalog_signature: "catalog-v1".into(),
            generation_id: Uuid::now_v7(),
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        };
        let segment = SearchSegmentRef {
            sst_id: Uuid::now_v7(),
            role: SearchSegmentRole::Delta,
            format: SearchSegmentFormat::VectorV6,
            payload: SearchSegmentPayload::Complete,
            event_ranges: vec![SearchEventRange::new(1, 2)],
            min_lsn: version_table.min_lsn,
            max_lsn: version_table.max_lsn,
            mutation_count: version_table.record_count,
            live_payload_count: version_table.live_count,
            suppress_count: version_table.suppress_count,
            content_xxh3: version_table.content_xxh3,
            complete_filter_properties: Vec::new(),
            stats: SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(2),
            },
            equal_lsn_conflict_count: 0,
        };
        let binding = SearchSegmentWireBinding::new(&state, &segment, version_table).unwrap();
        binding.validate(&state, &segment).unwrap();

        let mut drift = segment.clone();
        drift.max_lsn += 1;
        assert!(binding.validate(&state, &drift).is_err());
    }

    #[test]
    fn binding_dto_round_trips_every_stats_shape_and_rejects_unknown_or_trailing_wire() {
        fn options() -> impl bincode::Options {
            bincode::DefaultOptions::new()
                .with_fixint_encoding()
                .reject_trailing_bytes()
                .with_limit(1024 * 1024)
        }

        let (_, version_table) = encode_search_version_table(records(3)).unwrap();
        let stats = [
            SearchSegmentStats::Unknown,
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Absolute(3),
            },
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(-2),
            },
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Absolute(8),
                total_len: SearchStatValue::Absolute(80),
                term_df_violation_count: 0,
            },
            SearchSegmentStats::Text {
                doc_count: SearchStatValue::Delta(-3),
                total_len: SearchStatValue::Delta(17),
                term_df_violation_count: 4,
            },
        ];
        for (index, stats) in stats.into_iter().enumerate() {
            let format = if matches!(stats, SearchSegmentStats::Text { .. }) {
                SearchSegmentFormat::TextV4
            } else {
                SearchSegmentFormat::VectorV6
            };
            let binding = SearchSegmentWireBinding {
                format_version: SEARCH_SEGMENT_BINDING_VERSION,
                generation_id: Uuid::from_u128(100 + index as u128),
                index_name: format!("index-{index}"),
                catalog_signature: format!("catalog-{index}"),
                segment: SearchSegmentRef {
                    sst_id: Uuid::from_u128(200 + index as u128),
                    role: SearchSegmentRole::Delta,
                    format,
                    payload: SearchSegmentPayload::Complete,
                    event_ranges: vec![SearchEventRange::new(1, 2)],
                    min_lsn: version_table.min_lsn,
                    max_lsn: version_table.max_lsn,
                    mutation_count: version_table.record_count,
                    live_payload_count: version_table.live_count,
                    suppress_count: version_table.suppress_count,
                    content_xxh3: 99,
                    complete_filter_properties: vec!["active".into()],
                    stats,
                    equal_lsn_conflict_count: 0,
                },
                version_table: version_table.clone(),
            };
            let encoded = options().serialize(&binding).unwrap();
            let decoded: SearchSegmentWireBinding = options().deserialize(&encoded).unwrap();
            assert_eq!(decoded, binding);

            let mut trailing = encoded;
            trailing.push(0);
            assert!(options()
                .deserialize::<SearchSegmentWireBinding>(&trailing)
                .is_err());
        }

        let mut unknown_tag = options()
            .serialize(&SearchSegmentStatsSerde::Unknown)
            .unwrap();
        unknown_tag[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(options()
            .deserialize::<SearchSegmentStatsSerde>(&unknown_tag)
            .is_err());
    }
}
