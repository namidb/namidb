//! `EdgeSstWriter`: orchestrates the encoding of a single edge SST.
//!
//! The writer accepts pre-sorted `EdgeRecord`s (sorted ascending by
//! `key_id`, then by `partner_id`) and emits the on-disk byte body in
//! one shot via [`EdgeSstWriter::finish`].
//!
//! Declared edge property streams (RFC-002 §3.2.7) are wired through:
//! each declared property name becomes its own `SECTION_PROPERTY_STREAM`
//! with independently decodable pages of JSON-encoded `Value` payloads.
//! Properties NOT in the declared schema (or carried by overflow-only
//! edge types) fall back to the legacy single `__overflow_json` stream.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(test)]
use std::sync::Arc;

use bytes::Bytes;
use namidb_core::Value;
use uuid::Uuid;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use crate::error::{Error, Result};
use crate::spooled_object::SpooledObject;
use crate::sst::bloom::{BloomFilter, BLOOM_OMIT_THRESHOLD_BYTES, DEFAULT_BITS_PER_KEY};
use crate::sst::edges::encoding::{
    varint_len, write_offset, write_varint, OffsetWidth, TAG_DENSE, TAG_SPLIT,
};
use crate::sst::edges::fence_index::{DEFAULT_FENCE_STRIDE, FENCE_INDEX_THRESHOLD};
use crate::sst::edges::format::{
    EdgeFileFooter, EdgeFileHeader, EdgeSstBinding, SectionEntry, CODEC_NONE,
    CODEC_PROPERTY_PAGED_NONE, CODEC_PROPERTY_PAGED_ZSTD, EDGE_CHECKSUM_PAGE_BYTES,
    FLAG_HAS_PROPERTIES, FLAG_HAS_TOMBSTONES, FLAG_INVERSE_PARTNER, FLAG_SKEW_BUCKETS, HEADER_LEN,
    MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES, OVERFLOW_JSON_NAME, SECTION_EDGE_ORDINALS,
    SECTION_FENCE_INDEX, SECTION_KEY_IDS, SECTION_OFFSETS, SECTION_PAGE_CHECKSUMS,
    SECTION_PARTNERS, SECTION_PER_EDGE_LSN, SECTION_PER_EDGE_TOMBSTONES, SECTION_PROPERTY_STREAM,
    SECTION_SST_BINDING,
};
use crate::sst::edges::property_pages::{
    PropertyPageEntry, MAX_PROPERTY_PAGE_DECODE_BYTES, PROPERTY_PAGES_HEADER_LEN,
    PROPERTY_PAGES_MAGIC, PROPERTY_PAGES_VERSION, PROPERTY_PAGE_ENTRY_LEN,
};
use crate::sst::edges::EdgeDirection;
use crate::sst::paged_index::{EdgePointIndexBuilder, EdgePointIndexUpload};
use crate::sst::stats::{DegreeHistogram, PropertyColumnStats};

/// Maximum degree kept in the sequentially-decodable split representation by
/// default. Above this bound the dense block supports allocation-free binary
/// search for exact endpoint probes.
pub const DEFAULT_SKEW_THRESHOLD: usize = 1024;
/// Optional per-relationship exact-record ceiling. Zero (the default) is
/// unlimited now that values are disk-spooled. A non-zero operator ceiling is
/// enforced as an explicit write error, never as silent accelerator omission.
pub const DEFAULT_EDGE_POINT_MAX_ENTRY_BYTES: usize = 0;
/// Optional hard `.epidx` size ceiling. Zero (the default) is unlimited:
/// production must not silently remove the exact accelerator merely because
/// a compaction produced a large corpus. A non-zero operator ceiling fails
/// the authoritative write explicitly when crossed.
pub const DEFAULT_EDGE_POINT_MAX_SST_BYTES: usize = 0;
/// Public [`EdgeSstWriter::finish`] is a compatibility helper for small
/// fixtures and embedded callers that explicitly need one contiguous body.
/// Production flush/compaction never use it. The cap prevents an accidental
/// multi-gigabyte flatten from undoing the disk-spooled writer.
pub const DEFAULT_EDGE_IN_MEMORY_FINISH_MAX_BYTES: usize = 64 * 1024 * 1024;
pub const EDGE_IN_MEMORY_FINISH_MAX_BYTES_ENV: &str = "NAMIDB_EDGE_IN_MEMORY_FINISH_MAX_BYTES";
const EDGE_POINT_ESTIMATED_OVERHEAD_PER_ENTRY: usize = 32 + 18 + 64;
const EDGE_POINT_MIN_SERIALIZED_BYTES: usize = 64 + 4096;

/// One row in the edge SST input. `key_id` and `partner_id` carry the
/// **direction-specific** mapping: for a forward partner SST `key_id` is
/// `src_id` and `partner_id` is `dst_id`; for an inverse partner SST the
/// mapping is swapped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeRecord {
    pub key_id: [u8; 16],
    pub partner_id: [u8; 16],
    pub lsn: u64,
    pub tombstone: bool,
    /// JSON-encoded `Value` per declared property column, in the exact
    /// order of [`EdgeSstWriterOptions::declared_properties`]. `None` =
    /// the property is missing (renders as JSON `null` if the column
    /// is decoded, or as absent in the materialised `EdgeView.properties`
    /// map). Empty when the edge type has no declared properties.
    pub declared_properties: Vec<Option<String>>,
    /// JSON object holding any properties **not** in the declared schema
    /// (RFC-002 §3.2.7 fallback). `None` when there are no extras.
    pub overflow_json: Option<String>,
}

/// Tuning knobs for [`EdgeSstWriter`].
#[derive(Debug, Clone)]
pub struct EdgeSstWriterOptions {
    pub direction: EdgeDirection,
    pub edge_type: String,
    pub src_label: String,
    pub dst_label: String,
    pub schema_version: u64,
    /// Force a specific skew threshold. `None` → [`DEFAULT_SKEW_THRESHOLD`].
    pub skew_threshold: Option<usize>,
    pub fence_stride: u32,
    pub fence_threshold: u64,
    /// Bloom density (only emitted for SSTs larger than the omit threshold).
    pub bits_per_key: u8,
    pub expected_keys: u64,
    /// Compress each independently decodable overflow/declared-property page
    /// with its own Zstd frame.
    /// Default: true.
    pub compress_property_streams: bool,
    /// Declared property column names for this edge type (RFC-002 §3.2.7).
    /// Each becomes its own `SECTION_PROPERTY_STREAM` with a JSON-encoded
    /// `Value` payload per edge. Empty when the schema declares none.
    pub declared_properties: Vec<String>,
}

impl EdgeSstWriterOptions {
    pub fn new(
        direction: EdgeDirection,
        edge_type: impl Into<String>,
        src_label: impl Into<String>,
        dst_label: impl Into<String>,
    ) -> Self {
        Self {
            direction,
            edge_type: edge_type.into(),
            src_label: src_label.into(),
            dst_label: dst_label.into(),
            schema_version: 0,
            skew_threshold: None,
            fence_stride: DEFAULT_FENCE_STRIDE,
            fence_threshold: FENCE_INDEX_THRESHOLD,
            bits_per_key: DEFAULT_BITS_PER_KEY,
            expected_keys: 0,
            compress_property_streams: true,
            declared_properties: Vec::new(),
        }
    }
}

/// Result of finalising an [`EdgeSstWriter`].
#[derive(Debug)]
pub struct EdgeSstFinish {
    pub body: Bytes,
    pub stats: EdgeSstStats,
    pub bloom: Option<BloomFilter>,
}

/// Disk-backed authoritative edge body. The files are already ordered exactly
/// as they appear on the wire (data sections, page directory, footer); the
/// fixed 64-byte header is retained as the only in-memory prefix.
#[derive(Debug)]
pub(crate) struct EdgeSstUpload {
    header: Bytes,
    files: Vec<File>,
    size_bytes: u64,
}

impl EdgeSstUpload {
    pub(crate) fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub(crate) fn into_spooled_object(self) -> SpooledObject {
        SpooledObject::from_files(vec![self.header], self.files, self.size_bytes)
    }

    fn into_bytes(mut self, cap: usize) -> Result<Bytes> {
        if self.size_bytes > cap as u64 {
            return Err(Error::invariant(format!(
                "edge SST body requires {} bytes, above \
                 {EDGE_IN_MEMORY_FINISH_MAX_BYTES_ENV}={cap}; use the spooled \
                 flush/compaction path",
                self.size_bytes
            )));
        }
        let capacity = usize::try_from(self.size_bytes)
            .map_err(|_| Error::invariant("edge SST body length does not fit usize"))?;
        let mut body = Vec::with_capacity(capacity);
        body.extend_from_slice(&self.header);
        for file in &mut self.files {
            file.seek(SeekFrom::Start(0))?;
            file.read_to_end(&mut body)?;
        }
        if body.len() != capacity {
            return Err(Error::invariant(format!(
                "edge SST spools produced {} bytes, descriptor requires {capacity}",
                body.len()
            )));
        }
        Ok(Bytes::from(body))
    }
}

/// Internal flush/compaction product. The authoritative body never exists as
/// one resident [`Bytes`].
#[derive(Debug)]
pub(crate) struct EdgeSstBuild {
    pub(crate) id: Uuid,
    pub(crate) body: EdgeSstUpload,
    pub(crate) stats: EdgeSstStats,
    pub(crate) bloom: Option<BloomFilter>,
    pub(crate) point_index: Option<EdgePointIndexUpload>,
}

/// Statistics for the manifest's `SstDescriptor` (RFC-002 §3.3).
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSstStats {
    pub direction: EdgeDirection,
    pub key_count: u64,
    pub edge_count: u64,
    pub tombstone_count: u64,
    pub min_key_id: [u8; 16],
    pub max_key_id: [u8; 16],
    pub min_lsn: u64,
    pub max_lsn: u64,
    pub degree_histogram: DegreeHistogram,
    /// Reserved for typed edge-column statistics; current JSON property pages
    /// do not populate it.
    pub property_stats: Vec<PropertyColumnStats>,
    pub schema_version_min: u64,
    pub schema_version_max: u64,
}

/// Streaming writer with corpus-sized state externalised to anonymous local
/// files. Resident memory is bounded by one property mini-batch plus fixed
/// I/O buffers; even a single high-degree hub is staged on disk rather than
/// retained as `Vec<[u8; 16]>`.
#[derive(Debug)]
pub struct EdgeSstWriter {
    options: EdgeSstWriterOptions,
    /// Generated before the body is built so the immutable object path,
    /// descriptor, and future wire binding all share one identity.
    sst_id: Uuid,
    /// Pre-computed at `new()` so streaming partner-block encoding can
    /// produce stable byte output without re-tuning per call.
    skew_threshold: usize,

    // ── current key bucket — file-backed even for pathological hubs ──
    current_key: Option<[u8; 16]>,
    current_partners: DiskSpool,
    current_degree: u64,
    current_split_cost: u64,
    current_prev_top: Option<u64>,
    last_partner_in_key: Option<[u8; 16]>,

    // ── monotonic disk spools ──────────────────────────────────────────
    partners: DiskSpool,
    /// Native `u64` offsets are transformed to the selected wire width in one
    /// bounded second pass at finalisation.
    raw_offsets: DiskSpool,
    /// Cumulative per-edge row ordinal for each key. Starts with zero and
    /// receives one sentinel after every closed key bucket, so its final shape
    /// is always `key_count + 1`. This v1.1 accelerator lets ranged readers
    /// locate LSN/tombstone rows without scanning earlier partner blocks.
    edge_ordinals: DiskSpool,
    ordinals_started: bool,
    key_ids: DiskSpool,
    /// Candidate fixed-width fence entries (without the 8-byte header). They
    /// are emitted as a section only after the final key count crosses the
    /// configured threshold.
    fence_entries: DiskSpool,
    fence_entry_count: u32,
    /// Per-edge LSN bytes in the order partners arrive.
    lsns: DiskSpool,
    /// Per-edge tombstones are packed one live byte at a time.
    tombstones: DiskSpool,
    tombstone_live_byte: u8,
    tombstone_live_bits: u8,

    // ── running counters / stats ──
    key_count: u64,
    edge_count: u64,
    tombstone_count: u64,
    min_lsn: u64,
    max_lsn: u64,
    any_skew_block: bool,
    degree_histogram: DegreeHistogram,
    min_key_id: Option<[u8; 16]>,
    max_key_id: Option<[u8; 16]>,
    // ── range-readable property-page streams ───────────────────────────
    overflow: PropertyStream,
    /// One stream per declared property (RFC-002 §3.2.7), in the order
    /// of `options.declared_properties`. Each holds JSON-encoded
    /// JSON-encoded `Value` payloads.
    declared_streams: Vec<PropertyStream>,

    bloom: BloomFilter,
    point_index: Option<EdgePointIndexBuilder>,
    point_index_estimated_bytes: usize,
    point_index_max_entry_bytes: usize,
    point_index_max_sst_bytes: usize,
}

/// Lazily-created anonymous spool. Delaying creation keeps `new()` infallible
/// for API compatibility while every first write still reports ENOSPC and
/// directory errors through the surrounding `Result`.
#[derive(Debug, Default)]
struct DiskSpool {
    file: Option<File>,
    len: u64,
}

impl DiskSpool {
    fn ensure_file(&mut self) -> std::io::Result<&mut File> {
        if self.file.is_none() {
            self.file = Some(crate::sst::paged_index::create_spool_file()?);
        }
        Ok(self.file.as_mut().expect("file installed above"))
    }

    fn rewind(&mut self) -> std::io::Result<()> {
        self.ensure_file()?.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    fn clear(&mut self) -> std::io::Result<()> {
        let file = self.ensure_file()?;
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        self.len = 0;
        Ok(())
    }

    fn into_file(mut self) -> Result<(File, u64)> {
        self.ensure_file()?;
        let mut file = self.file.take().expect("file installed above");
        file.flush()?;
        file.sync_data()?;
        let actual = file.metadata()?.len();
        if actual != self.len {
            return Err(Error::invariant(format!(
                "edge spool length {} disagrees with file length {actual}",
                self.len
            )));
        }
        file.seek(SeekFrom::Start(0))?;
        Ok((file, self.len))
    }
}

impl Write for DiskSpool {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.ensure_file()?.write(buf)?;
        self.len = self
            .len
            .checked_add(written as u64)
            .ok_or_else(|| std::io::Error::other("edge spool length exceeds u64"))?;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.ensure_file()?.flush()
    }
}

/// One independently decodable property-page stream. Both its page directory
/// and payload are disk-spooled; resident strings are bounded by row count and
/// encoded bytes, whichever threshold is reached first.
struct PropertyStream {
    /// Stream name — `__overflow_json` for the catch-all bucket or the
    /// declared property's logical name (no `prop_` prefix).
    name: String,
    compress: bool,
    payload: DiskSpool,
    directory: DiskSpool,
    pending: Vec<Option<String>>,
    pending_bytes: usize,
    any_value: bool,
    row_count: u64,
    page_count: u32,
}

impl std::fmt::Debug for PropertyStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropertyStream")
            .field("name", &self.name)
            .field("pending_len", &self.pending.len())
            .field("pending_bytes", &self.pending_bytes)
            .field("any_value", &self.any_value)
            .field("row_count", &self.row_count)
            .field("page_count", &self.page_count)
            .finish()
    }
}

const PROPERTY_STREAM_MINI_BATCH: usize = 1024;
const PROPERTY_STREAM_MAX_PENDING_BYTES: usize = 1024 * 1024;

impl PropertyStream {
    fn new(name: impl Into<String>, compress: bool) -> Self {
        Self {
            name: name.into(),
            compress,
            payload: DiskSpool::default(),
            directory: DiskSpool::default(),
            pending: Vec::with_capacity(PROPERTY_STREAM_MINI_BATCH),
            pending_bytes: 0,
            any_value: false,
            row_count: 0,
            page_count: 0,
        }
    }

    fn append(&mut self, value: Option<String>) -> Result<()> {
        if value.is_some() {
            self.any_value = true;
        }
        let value_bytes = value.as_ref().map_or(0, String::len);
        if !self.pending.is_empty()
            && self.pending_bytes.saturating_add(value_bytes) > PROPERTY_STREAM_MAX_PENDING_BYTES
        {
            self.flush_batch()?;
        }
        self.pending_bytes = self.pending_bytes.saturating_add(value_bytes);
        self.pending.push(value);
        self.row_count = self
            .row_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("edge property row count exceeds u64"))?;
        if self.pending.len() >= PROPERTY_STREAM_MINI_BATCH
            || self.pending_bytes >= PROPERTY_STREAM_MAX_PENDING_BYTES
        {
            self.flush_batch()?;
        }
        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let page_rows = u32::try_from(self.pending.len())
            .map_err(|_| Error::invariant("edge property page row count exceeds u32"))?;
        let first_row = self
            .row_count
            .checked_sub(u64::from(page_rows))
            .ok_or_else(|| Error::invariant("edge property first-row underflow"))?;
        let mut raw = DiskSpool::default();
        raw.write_all(&page_rows.to_le_bytes())?;
        for value in &self.pending {
            match value {
                None => raw.write_all(&[0])?,
                Some(value) => {
                    let len = u32::try_from(value.len()).map_err(|_| {
                        Error::invariant(format!(
                            "edge property '{}' value exceeds u32 bytes",
                            self.name
                        ))
                    })?;
                    raw.write_all(&[1])?;
                    raw.write_all(&len.to_le_bytes())?;
                    raw.write_all(value.as_bytes())?;
                }
            }
        }
        if raw.len > MAX_PROPERTY_PAGE_DECODE_BYTES as u64 {
            return Err(Error::invariant(format!(
                "edge property '{}' page requires {} decoded bytes, above {}",
                self.name, raw.len, MAX_PROPERTY_PAGE_DECODE_BYTES
            )));
        }
        let decoded_len = raw.len;
        let mut encoded = if self.compress {
            raw.rewind()?;
            let mut encoder =
                zstd::stream::write::Encoder::new(DiskSpool::default(), 3).map_err(|error| {
                    Error::invariant(format!(
                        "edge property '{}' zstd encoder: {error}",
                        self.name
                    ))
                })?;
            std::io::copy(
                raw.file
                    .as_mut()
                    .ok_or_else(|| Error::invariant("edge property raw spool disappeared"))?,
                &mut encoder,
            )?;
            encoder.finish().map_err(|error| {
                Error::invariant(format!(
                    "edge property '{}' zstd finish: {error}",
                    self.name
                ))
            })?
        } else {
            raw
        };
        let payload_offset = self.payload.len;
        encoded.rewind()?;
        let mut hasher = Xxh3::new();
        let mut buffer = vec![0u8; EDGE_CHECKSUM_PAGE_BYTES as usize];
        let mut remaining = encoded.len;
        let source = encoded
            .file
            .as_mut()
            .ok_or_else(|| Error::invariant("edge property encoded spool disappeared"))?;
        while remaining > 0 {
            let take = usize::try_from(remaining.min(buffer.len() as u64))
                .expect("property copy buffer is bounded by usize");
            source.read_exact(&mut buffer[..take])?;
            hasher.update(&buffer[..take]);
            self.payload.write_all(&buffer[..take])?;
            remaining -= take as u64;
        }
        let entry = PropertyPageEntry {
            first_row,
            row_count: page_rows,
            offset: payload_offset,
            encoded_len: encoded.len,
            decoded_len,
            checksum: hasher.digest(),
        };
        let mut raw_entry = Vec::with_capacity(PROPERTY_PAGE_ENTRY_LEN);
        entry.encode(&mut raw_entry);
        self.directory.write_all(&raw_entry)?;
        self.page_count = self
            .page_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("edge property page count exceeds u32"))?;
        self.pending.clear();
        self.pending_bytes = 0;
        Ok(())
    }

    /// `Ok(Some((spool, codec)))` when the stream observed at least one
    /// non-null value; `Ok(None)` when every appended row was `None` —
    /// the writer skips the section entirely in that case.
    fn finish(mut self) -> Result<Option<(DiskSpool, u8)>> {
        if !self.any_value {
            return Ok(None);
        }
        self.flush_batch()?;
        let directory_len = u64::from(self.page_count)
            .checked_mul(PROPERTY_PAGE_ENTRY_LEN as u64)
            .ok_or_else(|| Error::invariant("edge property directory length exceeds u64"))?;
        if self.directory.len != directory_len {
            return Err(Error::invariant(
                "edge property directory spool length mismatch",
            ));
        }
        let payload_start = (PROPERTY_PAGES_HEADER_LEN as u64)
            .checked_add(directory_len)
            .ok_or_else(|| Error::invariant("edge property payload start exceeds u64"))?;
        let mut out = DiskSpool::default();
        out.write_all(&PROPERTY_PAGES_MAGIC)?;
        out.write_all(&PROPERTY_PAGES_VERSION.to_le_bytes())?;
        out.write_all(&0u16.to_le_bytes())?;
        out.write_all(&self.page_count.to_le_bytes())?;
        out.write_all(&self.row_count.to_le_bytes())?;
        out.write_all(&(PROPERTY_PAGE_ENTRY_LEN as u32).to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;

        self.directory.rewind()?;
        let directory = self
            .directory
            .file
            .as_mut()
            .ok_or_else(|| Error::invariant("edge property directory spool disappeared"))?;
        for _ in 0..self.page_count {
            let mut raw = [0u8; PROPERTY_PAGE_ENTRY_LEN];
            directory.read_exact(&mut raw)?;
            let payload_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap());
            let section_offset = payload_start
                .checked_add(payload_offset)
                .ok_or_else(|| Error::invariant("edge property page offset exceeds u64"))?;
            raw[12..20].copy_from_slice(&section_offset.to_le_bytes());
            out.write_all(&raw)?;
        }
        self.payload.rewind()?;
        std::io::copy(
            self.payload
                .file
                .as_mut()
                .ok_or_else(|| Error::invariant("edge property payload spool disappeared"))?,
            &mut out,
        )?;
        let codec = if self.compress {
            CODEC_PROPERTY_PAGED_ZSTD
        } else {
            CODEC_PROPERTY_PAGED_NONE
        };
        Ok(Some((out, codec)))
    }
}

impl EdgeSstWriter {
    pub fn new(options: EdgeSstWriterOptions) -> Self {
        let bloom =
            BloomFilter::with_capacity(options.expected_keys.max(1), options.bits_per_key.max(1));
        // Keep split-block point scans bounded independently of corpus size.
        // The former `4 * sqrt(key_count)` default reached 4k at one million
        // keys, making exact relationship MERGE linear in a hub's degree after
        // compaction. Callers can still pin a different threshold explicitly.
        let skew_threshold = options.skew_threshold.unwrap_or(DEFAULT_SKEW_THRESHOLD);
        let compress_property_streams = options.compress_property_streams;
        let declared_streams: Vec<PropertyStream> = options
            .declared_properties
            .iter()
            .map(|name| PropertyStream::new(name, compress_property_streams))
            .collect();
        let point_index_max_entry_bytes = env_usize(
            "NAMIDB_EDGE_POINT_MAX_ENTRY_BYTES",
            DEFAULT_EDGE_POINT_MAX_ENTRY_BYTES,
        );
        let point_index_max_sst_bytes = env_usize(
            "NAMIDB_EDGE_POINT_MAX_SST_BYTES",
            DEFAULT_EDGE_POINT_MAX_SST_BYTES,
        );
        let point_index_enabled = matches!(options.direction, EdgeDirection::Forward);
        Self {
            options,
            sst_id: Uuid::now_v7(),
            skew_threshold,
            current_key: None,
            current_partners: DiskSpool::default(),
            current_degree: 0,
            current_split_cost: 0,
            current_prev_top: None,
            last_partner_in_key: None,
            partners: DiskSpool::default(),
            raw_offsets: DiskSpool::default(),
            edge_ordinals: DiskSpool::default(),
            ordinals_started: false,
            key_ids: DiskSpool::default(),
            fence_entries: DiskSpool::default(),
            fence_entry_count: 0,
            lsns: DiskSpool::default(),
            tombstones: DiskSpool::default(),
            tombstone_live_byte: 0,
            tombstone_live_bits: 0,
            key_count: 0,
            edge_count: 0,
            tombstone_count: 0,
            min_lsn: u64::MAX,
            max_lsn: 0,
            any_skew_block: false,
            degree_histogram: DegreeHistogram::empty(),
            min_key_id: None,
            max_key_id: None,
            overflow: PropertyStream::new(OVERFLOW_JSON_NAME, compress_property_streams),
            declared_streams,
            bloom,
            point_index: point_index_enabled.then(EdgePointIndexBuilder::new),
            point_index_estimated_bytes: EDGE_POINT_MIN_SERIALIZED_BYTES,
            point_index_max_entry_bytes,
            point_index_max_sst_bytes,
        }
    }

    /// Append one edge. Records must arrive sorted ascending by `key_id`,
    /// then by `partner_id`. The writer validates the ordering and drains
    /// the closed bucket's partners into the output section as soon as
    /// `key_id` advances.
    ///
    /// `record.declared_properties` must have exactly the same length as
    /// `EdgeSstWriterOptions::declared_properties` — one entry per
    /// declared column, in the same order. The values are JSON-encoded
    /// `Value` strings (or `None` when the property is absent on this
    /// edge).
    pub fn append(&mut self, record: EdgeRecord) -> Result<()> {
        if record.declared_properties.len() != self.declared_streams.len() {
            return Err(Error::invariant(format!(
                "edge SST record carries {} declared properties; writer expects {} (edge_type {})",
                record.declared_properties.len(),
                self.declared_streams.len(),
                self.options.edge_type,
            )));
        }
        if self.point_index.is_some() {
            // Refuse obviously oversized source material before decoding and
            // re-encoding it into a second property map. False refusals are
            // safe (the CSR remains authoritative); allocating beyond the
            // configured per-entry ceiling is not.
            let source_bytes = self
                .options
                .declared_properties
                .iter()
                .zip(&record.declared_properties)
                .fold(
                    record.overflow_json.as_ref().map_or(0, String::len),
                    |size, (name, value)| {
                        size.saturating_add(name.len())
                            .saturating_add(value.as_ref().map_or(0, String::len))
                    },
                );
            if self.point_index_max_entry_bytes > 0
                && !record.tombstone
                && source_bytes.saturating_add(EDGE_POINT_ESTIMATED_OVERHEAD_PER_ENTRY)
                    > self.point_index_max_entry_bytes
            {
                return Err(Error::invariant(format!(
                    "edge point record exceeds NAMIDB_EDGE_POINT_MAX_ENTRY_BYTES={} for {}",
                    self.point_index_max_entry_bytes, self.options.edge_type
                )));
            }
        }
        if let Some(point_index) = self.point_index.as_mut() {
            let point_properties = if record.tombstone {
                Bytes::new()
            } else {
                encode_point_properties(
                    &self.options.declared_properties,
                    &record.declared_properties,
                    record.overflow_json.as_deref(),
                )?
            };
            let point_value = crate::sst::edges::point_index::encode(
                record.lsn,
                record.tombstone,
                &point_properties,
            )?;
            let next_estimate = self
                .point_index_estimated_bytes
                .saturating_add(EDGE_POINT_ESTIMATED_OVERHEAD_PER_ENTRY)
                .saturating_add(point_value.len());
            let entry_estimate = point_value
                .len()
                .saturating_add(EDGE_POINT_ESTIMATED_OVERHEAD_PER_ENTRY);
            if self.point_index_max_entry_bytes > 0
                && entry_estimate > self.point_index_max_entry_bytes
            {
                return Err(Error::invariant(format!(
                    "encoded edge point record exceeds NAMIDB_EDGE_POINT_MAX_ENTRY_BYTES={} for {}",
                    self.point_index_max_entry_bytes, self.options.edge_type
                )));
            }
            if self.point_index_max_sst_bytes > 0 && next_estimate > self.point_index_max_sst_bytes
            {
                return Err(Error::invariant(format!(
                    "edge point sidecar exceeds NAMIDB_EDGE_POINT_MAX_SST_BYTES={} for {}",
                    self.point_index_max_sst_bytes, self.options.edge_type
                )));
            }
            point_index.push(&record.key_id, &record.partner_id, &point_value)?;
            self.point_index_estimated_bytes = next_estimate;
        }
        if let Some(prev_key) = self.current_key {
            if record.key_id < prev_key {
                return Err(Error::invariant(
                    "edge SST records must be sorted by key_id ascending",
                ));
            }
            if record.key_id == prev_key {
                if let Some(prev_p) = self.last_partner_in_key {
                    if record.partner_id <= prev_p {
                        return Err(Error::invariant(
                            "edge SST partners within a key must be sorted ascending and unique",
                        ));
                    }
                }
            } else {
                // Boundary: flush the closed bucket, then start a new one.
                self.flush_current_bucket()?;
            }
        }

        if self.current_key != Some(record.key_id) {
            self.current_key = Some(record.key_id);
            self.bloom.insert(&record.key_id);
        }
        self.current_partners.write_all(&record.partner_id)?;
        let top = u64::from_le_bytes(record.partner_id[..8].try_into().unwrap());
        let top_wire = match self.current_prev_top {
            Some(previous) => top.wrapping_sub(previous),
            None => top,
        };
        self.current_split_cost = self
            .current_split_cost
            .checked_add((varint_len(top_wire) + 8) as u64)
            .ok_or_else(|| Error::invariant("edge partner split cost exceeds u64"))?;
        self.current_prev_top = Some(top);
        self.current_degree = self
            .current_degree
            .checked_add(1)
            .ok_or_else(|| Error::invariant("edge degree exceeds u64"))?;
        self.last_partner_in_key = Some(record.partner_id);

        // Per-edge spools happen right here (one pass).
        self.lsns.write_all(&record.lsn.to_le_bytes())?;
        if record.tombstone {
            self.tombstone_live_byte |= 1 << self.tombstone_live_bits;
        }
        self.tombstone_live_bits += 1;
        if self.tombstone_live_bits == 8 {
            self.tombstones.write_all(&[self.tombstone_live_byte])?;
            self.tombstone_live_byte = 0;
            self.tombstone_live_bits = 0;
        }
        if record.tombstone {
            self.tombstone_count += 1;
        }
        self.min_lsn = self.min_lsn.min(record.lsn);
        self.max_lsn = self.max_lsn.max(record.lsn);
        self.edge_count += 1;
        self.overflow.append(record.overflow_json)?;
        // Append the declared property values in the exact order set up
        // by `options.declared_properties`. The length-check above
        // already enforced cardinality.
        for (stream, value) in self
            .declared_streams
            .iter_mut()
            .zip(record.declared_properties)
        {
            stream.append(value)?;
        }
        Ok(())
    }

    /// Convenience: extend from any iterator yielding pre-sorted records.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = EdgeRecord>) -> Result<()> {
        for r in iter {
            self.append(r)?;
        }
        Ok(())
    }

    pub fn record_count(&self) -> usize {
        self.edge_count as usize
    }

    /// Drain the file-backed current bucket into the permanent partner spool
    /// and reset it for reuse. The resulting bytes are identical to
    /// `write_partner_block`, but no high-degree partner vector is created.
    fn flush_current_bucket(&mut self) -> Result<()> {
        let Some(key) = self.current_key else {
            return Ok(());
        };
        if self.current_degree == 0 {
            return Ok(());
        }
        let degree = self.current_degree;
        let dense_cost = degree
            .checked_mul(16)
            .ok_or_else(|| Error::invariant("dense edge bucket length exceeds u64"))?;
        let is_skew = degree > self.skew_threshold as u64;
        let tag = if is_skew || self.current_split_cost >= dense_cost {
            TAG_DENSE
        } else {
            TAG_SPLIT
        };

        self.raw_offsets
            .write_all(&self.partners.len.to_le_bytes())?;
        let mut block_header = Vec::with_capacity(11);
        write_varint(degree, &mut block_header);
        block_header.push(tag);
        self.partners.write_all(&block_header)?;
        self.current_partners.rewind()?;
        let source = self
            .current_partners
            .file
            .as_mut()
            .ok_or_else(|| Error::invariant("edge partner bucket spool disappeared"))?;
        match tag {
            TAG_DENSE => {
                std::io::copy(source, &mut self.partners)?;
            }
            TAG_SPLIT => {
                let mut previous_top = None;
                let mut raw = [0u8; 16];
                for _ in 0..degree {
                    source.read_exact(&mut raw)?;
                    let top = u64::from_le_bytes(raw[..8].try_into().unwrap());
                    let bottom = &raw[8..];
                    let encoded_top = match previous_top {
                        Some(previous) => top.wrapping_sub(previous),
                        None => top,
                    };
                    let mut varint = Vec::with_capacity(10);
                    write_varint(encoded_top, &mut varint);
                    self.partners.write_all(&varint)?;
                    self.partners.write_all(bottom)?;
                    previous_top = Some(top);
                }
            }
            _ => unreachable!(),
        }
        if is_skew {
            self.any_skew_block = true;
        }
        self.degree_histogram.observe(degree);
        self.key_ids.write_all(&key)?;
        let stride = u64::from(self.options.fence_stride.max(1));
        if self.key_count % stride == 0 {
            self.fence_entries.write_all(&key)?;
            self.fence_entries
                .write_all(&self.key_count.saturating_mul(16).to_le_bytes())?;
            self.fence_entry_count = self
                .fence_entry_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("edge fence entry count exceeds u32"))?;
        }
        self.key_count += 1;
        // `edge_count` already includes every record in the bucket being
        // closed and no record from the next bucket (the boundary flush runs
        // before the next append mutates the per-edge accumulators).
        if !self.ordinals_started {
            self.edge_ordinals.write_all(&0u64.to_le_bytes())?;
            self.ordinals_started = true;
        }
        self.edge_ordinals
            .write_all(&self.edge_count.to_le_bytes())?;
        if self.min_key_id.is_none() {
            self.min_key_id = Some(key);
        }
        self.max_key_id = Some(key);
        self.current_partners.clear()?;
        self.current_degree = 0;
        self.current_split_cost = 0;
        self.current_prev_top = None;
        self.last_partner_in_key = None;
        Ok(())
    }

    /// Serialise the SST body.
    pub fn finish(mut self) -> Result<EdgeSstFinish> {
        // Public callers asked only for the authoritative CSR product. Avoid
        // finalising and syncing a `.epidx` spool that this API cannot return;
        // flush/compaction use `finish_with_point_index`.
        self.point_index = None;
        let build = self.finish_with_point_index()?;
        let cap = env_usize(
            EDGE_IN_MEMORY_FINISH_MAX_BYTES_ENV,
            DEFAULT_EDGE_IN_MEMORY_FINISH_MAX_BYTES,
        );
        let EdgeSstBuild {
            body, stats, bloom, ..
        } = build;
        Ok(EdgeSstFinish {
            body: body.into_bytes(cap)?,
            stats,
            bloom,
        })
    }

    pub(crate) fn finish_with_point_index(mut self) -> Result<EdgeSstBuild> {
        // Drain any open bucket.
        self.flush_current_bucket()?;

        let key_count = self.key_count;
        let edge_count = self.edge_count;
        let direction = self.options.direction;
        let schema_version = self.options.schema_version;
        let fence_threshold = self.options.fence_threshold;
        let fence_stride = self.options.fence_stride.max(1);

        // Sentinel offset.
        self.raw_offsets
            .write_all(&self.partners.len.to_le_bytes())?;
        if !self.ordinals_started {
            self.edge_ordinals.write_all(&0u64.to_le_bytes())?;
            self.ordinals_started = true;
        }
        if self.tombstone_live_bits > 0 {
            self.tombstones.write_all(&[self.tombstone_live_byte])?;
            self.tombstone_live_byte = 0;
            self.tombstone_live_bits = 0;
        }

        // ── Bitpack offsets ────────────────────────────────────────────
        let max_offset = self.partners.len;
        if max_offset >= (1u64 << 48) {
            return Err(Error::invariant(format!(
                "edge partner section requires {max_offset} bytes; format v1 supports <2^48"
            )));
        }
        let offset_width = OffsetWidth::for_max(max_offset);
        let mut offsets = DiskSpool::default();
        self.raw_offsets.rewind()?;
        let raw_offsets = self
            .raw_offsets
            .file
            .as_mut()
            .ok_or_else(|| Error::invariant("raw edge offsets spool disappeared"))?;
        let offset_rows = key_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("edge offset row count exceeds u64"))?;
        for _ in 0..offset_rows {
            let mut raw = [0u8; 8];
            raw_offsets.read_exact(&mut raw)?;
            let mut encoded = Vec::with_capacity(offset_width.bytes());
            write_offset(u64::from_le_bytes(raw), offset_width, &mut encoded);
            offsets.write_all(&encoded)?;
        }

        // Tombstone bytes: drop the section entirely if the SST has no
        // tombstones — that's the wire-format invariant the reader keys
        // off of via the `HAS_TOMBSTONES` flag.
        let has_tombstones = self.tombstone_count > 0;

        // Prefix the already-streamed candidate entries only when this SST is
        // large enough to need a fence.
        let fence = if key_count > fence_threshold {
            let mut fence = DiskSpool::default();
            fence.write_all(&fence_stride.to_le_bytes())?;
            fence.write_all(&self.fence_entry_count.to_le_bytes())?;
            self.fence_entries.rewind()?;
            let entries = self
                .fence_entries
                .file
                .as_mut()
                .ok_or_else(|| Error::invariant("edge fence spool disappeared"))?;
            std::io::copy(entries, &mut fence)?;
            Some(fence)
        } else {
            None
        };

        // Overflow section.
        let overflow_section = self.overflow.finish()?;
        // Declared property sections (RFC-002 §3.2.7). One independently
        // paged stream per declared property — the order matches
        // `options.declared_properties`. Streams whose every appended
        // value was `None` are skipped (Ok(None) from `PropertyStream::finish`).
        let declared_property_names = self.options.declared_properties.clone();
        let mut declared_sections: Vec<(String, DiskSpool, u8)> =
            Vec::with_capacity(self.declared_streams.len());
        for (name, stream) in declared_property_names
            .iter()
            .zip(std::mem::take(&mut self.declared_streams))
        {
            if let Some((body, codec)) = stream.finish()? {
                declared_sections.push((name.clone(), body, codec));
            }
        }

        // Min/max LSN sanity for empty SST.
        let min_lsn = if edge_count == 0 { 0 } else { self.min_lsn };
        let max_lsn = self.max_lsn;
        let min_key_id = self.min_key_id.unwrap_or([0u8; 16]);
        let max_key_id = self.max_key_id.unwrap_or([0u8; 16]);

        let mut flags = 0u32;
        if matches!(direction, EdgeDirection::Inverse) {
            flags |= FLAG_INVERSE_PARTNER;
        }
        if has_tombstones {
            flags |= FLAG_HAS_TOMBSTONES;
        }
        if self.any_skew_block {
            flags |= FLAG_SKEW_BUCKETS;
        }
        if overflow_section.is_some() || !declared_sections.is_empty() {
            flags |= FLAG_HAS_PROPERTIES;
        }
        let mut header = Vec::with_capacity(HEADER_LEN);
        EdgeFileHeader::new(
            &self.options.edge_type,
            &self.options.src_label,
            &self.options.dst_label,
            flags,
        )
        .encode(&mut header);
        debug_assert_eq!(header.len(), HEADER_LEN);

        let mut pending_sections = vec![
            PendingSection::new(SECTION_KEY_IDS, "", CODEC_NONE, self.key_ids),
            PendingSection::new(SECTION_OFFSETS, "", CODEC_NONE, offsets),
            PendingSection::new(SECTION_EDGE_ORDINALS, "", CODEC_NONE, self.edge_ordinals),
            PendingSection::new(SECTION_PARTNERS, "", CODEC_NONE, self.partners),
            PendingSection::new(SECTION_PER_EDGE_LSN, "", CODEC_NONE, self.lsns),
        ];
        if has_tombstones {
            pending_sections.push(PendingSection::new(
                SECTION_PER_EDGE_TOMBSTONES,
                "",
                CODEC_NONE,
                self.tombstones,
            ));
        }
        if let Some(fence) = fence {
            pending_sections.push(PendingSection::new(
                SECTION_FENCE_INDEX,
                "",
                CODEC_NONE,
                fence,
            ));
        }
        if let Some((body, codec)) = overflow_section {
            pending_sections.push(PendingSection::new(
                SECTION_PROPERTY_STREAM,
                OVERFLOW_JSON_NAME,
                codec,
                body,
            ));
        }
        for (name, body, codec) in declared_sections {
            pending_sections.push(PendingSection::new(
                SECTION_PROPERTY_STREAM,
                name,
                codec,
                body,
            ));
        }

        // Finalise each independent file, calculating its complete checksum
        // and 64-KiB page hashes in one bounded scan.
        let mut next_offset = HEADER_LEN as u64;
        let mut final_sections = Vec::with_capacity(pending_sections.len());
        for section in pending_sections {
            let section = finalise_section(section, next_offset)?;
            next_offset = next_offset
                .checked_add(section.entry.length)
                .ok_or_else(|| Error::invariant("edge SST section offsets exceed u64"))?;
            final_sections.push(section);
        }
        let data_entries: Vec<SectionEntry> = final_sections
            .iter()
            .map(|section| section.entry.clone())
            .collect();
        let binding =
            EdgeSstBinding::for_sections(*self.sst_id.as_bytes(), &header, &data_entries)?;
        let mut binding_spool = DiskSpool::default();
        binding_spool.write_all(&binding.encode())?;
        let binding_section = finalise_section(
            PendingSection::new(SECTION_SST_BINDING, "", CODEC_NONE, binding_spool),
            next_offset,
        )?;
        next_offset = next_offset
            .checked_add(binding_section.entry.length)
            .ok_or_else(|| Error::invariant("edge SST binding end exceeds u64"))?;
        final_sections.push(binding_section);
        let page_directory = build_page_checksum_spool(&mut final_sections)?;
        let page_directory = finalise_section(
            PendingSection::new(SECTION_PAGE_CHECKSUMS, "", CODEC_NONE, page_directory),
            next_offset,
        )?;
        next_offset = next_offset
            .checked_add(page_directory.entry.length)
            .ok_or_else(|| Error::invariant("edge SST page-directory end exceeds u64"))?;

        let mut sections: Vec<SectionEntry> = final_sections
            .iter()
            .map(|section| section.entry.clone())
            .collect();
        sections.push(page_directory.entry.clone());

        let footer = EdgeFileFooter {
            sections,
            key_count,
            edge_count,
            offsets_bits: offset_width.as_bits(),
            min_key_id,
            max_key_id,
            min_lsn,
            max_lsn,
            schema_version_min: schema_version,
            schema_version_max: schema_version,
        };
        let mut footer_bytes = Vec::new();
        footer.encode(&mut footer_bytes)?;
        let mut footer_spool = DiskSpool::default();
        footer_spool.write_all(&footer_bytes)?;
        let (footer_file, footer_len) = footer_spool.into_file()?;
        let body_len = next_offset
            .checked_add(footer_len)
            .ok_or_else(|| Error::invariant("edge SST body length exceeds u64"))?;
        let bloom = if body_len >= BLOOM_OMIT_THRESHOLD_BYTES {
            Some(self.bloom)
        } else {
            None
        };

        let stats = EdgeSstStats {
            direction,
            key_count,
            edge_count,
            tombstone_count: self.tombstone_count,
            min_key_id,
            max_key_id,
            min_lsn,
            max_lsn,
            degree_histogram: self.degree_histogram,
            property_stats: Vec::new(),
            schema_version_min: schema_version,
            schema_version_max: schema_version,
        };
        let point_index_max_sst_bytes = self.point_index_max_sst_bytes;
        let point_index = self
            .point_index
            .map(EdgePointIndexBuilder::finish_upload)
            .transpose()?;
        if point_index_max_sst_bytes > 0
            && point_index
                .as_ref()
                .is_some_and(|body| body.size_bytes() > point_index_max_sst_bytes as u64)
        {
            return Err(Error::invariant(format!(
                "final edge point sidecar exceeds NAMIDB_EDGE_POINT_MAX_SST_BYTES={point_index_max_sst_bytes}"
            )));
        }
        if point_index
            .as_ref()
            .is_some_and(|body| body.entry_count() != edge_count)
        {
            return Err(Error::invariant(
                "edge point sidecar entry count disagrees with edge SST",
            ));
        }

        let mut files = Vec::with_capacity(final_sections.len() + 2);
        files.extend(final_sections.into_iter().map(|section| section.file));
        files.push(page_directory.file);
        files.push(footer_file);

        Ok(EdgeSstBuild {
            id: self.sst_id,
            body: EdgeSstUpload {
                header: Bytes::from(header),
                files,
                size_bytes: body_len,
            },
            stats,
            bloom,
            point_index,
        })
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(default)
}

fn encode_point_properties(
    declared_names: &[String],
    declared_values: &[Option<String>],
    overflow_json: Option<&str>,
) -> Result<Bytes> {
    let mut properties: BTreeMap<String, Value> = match overflow_json {
        Some(json) => serde_json::from_str(json)
            .map_err(|error| Error::invariant(format!("edge overflow decode: {error}")))?,
        None => BTreeMap::new(),
    };
    for (name, encoded) in declared_names.iter().zip(declared_values) {
        let Some(encoded) = encoded else {
            continue;
        };
        let value: Value = serde_json::from_str(encoded).map_err(|error| {
            Error::invariant(format!("edge declared property '{name}' decode: {error}"))
        })?;
        properties.insert(name.clone(), value);
    }
    serde_json::to_vec(&properties)
        .map(Bytes::from)
        .map_err(|error| Error::invariant(format!("edge point properties encode: {error}")))
}

#[derive(Debug)]
struct PendingSection {
    kind: u16,
    name: String,
    codec: u8,
    spool: DiskSpool,
}

impl PendingSection {
    fn new(kind: u16, name: impl Into<String>, codec: u8, spool: DiskSpool) -> Self {
        Self {
            kind,
            name: name.into(),
            codec,
            spool,
        }
    }
}

#[derive(Debug)]
struct FinalSection {
    entry: SectionEntry,
    file: File,
    page_checksums: File,
    page_count: u32,
}

fn finalise_section(section: PendingSection, offset: u64) -> Result<FinalSection> {
    let (mut file, length) = section.spool.into_file()?;
    let mut whole = Xxh3::new();
    let mut checksum_spool = DiskSpool::default();
    let mut page = vec![0u8; EDGE_CHECKSUM_PAGE_BYTES as usize];
    let mut remaining = length;
    let mut page_count = 0u32;
    while remaining > 0 {
        let take = usize::try_from(remaining.min(page.len() as u64))
            .expect("edge checksum page is bounded by usize");
        file.read_exact(&mut page[..take])?;
        whole.update(&page[..take]);
        checksum_spool.write_all(&xxh3_64(&page[..take]).to_le_bytes())?;
        page_count = page_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("edge section page count exceeds u32"))?;
        remaining -= take as u64;
    }
    file.seek(SeekFrom::Start(0))?;
    let (page_checksums, checksum_bytes) = checksum_spool.into_file()?;
    if checksum_bytes != u64::from(page_count) * 8 {
        return Err(Error::invariant(
            "edge section page-checksum spool length mismatch",
        ));
    }
    Ok(FinalSection {
        entry: SectionEntry {
            kind: section.kind,
            offset,
            length,
            codec: section.codec,
            xxhash3_64: whole.digest(),
            name: section.name,
        },
        file,
        page_checksums,
        page_count,
    })
}

/// Build the v1.2 checksum directory directly into a spool. Per-section
/// checksum sequences are copied from their own files, so writer memory stays
/// one fixed 64-KiB page regardless of the edge-body size.
fn build_page_checksum_spool(sections: &mut [FinalSection]) -> Result<DiskSpool> {
    let section_count = u32::try_from(sections.len())
        .map_err(|_| Error::invariant("edge checksum section count exceeds u32"))?;
    let encoded_len = 16u64
        .checked_add(
            sections
                .iter()
                .try_fold(0u64, |total, section| {
                    total
                        .checked_add(20)
                        .and_then(|value| {
                            value.checked_add(u64::from(section.page_count).saturating_mul(8))
                        })
                        .ok_or(())
                })
                .map_err(|()| Error::invariant("edge checksum directory length exceeds u64"))?,
        )
        .ok_or_else(|| Error::invariant("edge checksum directory length exceeds u64"))?;
    if encoded_len > MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES {
        return Err(Error::invariant(format!(
            "page checksum directory requires {encoded_len} bytes, above limit \
             {MAX_EDGE_PAGE_CHECKSUM_DIRECTORY_BYTES}"
        )));
    }

    let mut out = DiskSpool::default();
    out.write_all(b"TGEPGC02")?;
    out.write_all(&EDGE_CHECKSUM_PAGE_BYTES.to_le_bytes())?;
    out.write_all(&section_count.to_le_bytes())?;
    for section in sections {
        out.write_all(&section.entry.offset.to_le_bytes())?;
        out.write_all(&section.entry.length.to_le_bytes())?;
        out.write_all(&section.page_count.to_le_bytes())?;
        section.page_checksums.seek(SeekFrom::Start(0))?;
        std::io::copy(&mut section.page_checksums, &mut out)?;
    }
    if out.len != encoded_len {
        return Err(Error::invariant(format!(
            "edge checksum directory wrote {} bytes, expected {encoded_len}",
            out.len
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sst::edges::format::{
        EdgeFileHeader, EdgePageChecksumDirectory, FLAG_INVERSE_PARTNER,
    };
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

    fn key(top: u64, bot: u64) -> [u8; 16] {
        let mut k = [0u8; 16];
        k[..8].copy_from_slice(&top.to_le_bytes());
        k[8..].copy_from_slice(&bot.to_le_bytes());
        k
    }

    fn record(k: [u8; 16], p: [u8; 16], lsn: u64) -> EdgeRecord {
        EdgeRecord {
            key_id: k,
            partner_id: p,
            lsn,
            tombstone: false,
            declared_properties: vec![],
            overflow_json: None,
        }
    }

    #[test]
    fn writer_round_trip_minimal() {
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "Person", "Person");
        let mut w = EdgeSstWriter::new(opts);
        let k1 = key(1, 1);
        let k2 = key(1, 2);
        let k3 = key(2, 0);
        let p1 = key(10, 1);
        let p2 = key(10, 2);
        let p3 = key(10, 3);
        w.append(record(k1, p1, 100)).unwrap();
        w.append(record(k1, p2, 101)).unwrap();
        w.append(record(k2, p3, 102)).unwrap();
        w.append(record(k3, p1, 103)).unwrap();
        let finish = w.finish().unwrap();
        assert_eq!(finish.stats.key_count, 3);
        assert_eq!(finish.stats.edge_count, 4);
        assert_eq!(finish.stats.min_lsn, 100);
        assert_eq!(finish.stats.max_lsn, 103);
        // The body must round-trip through the footer decoder.
        let (footer, _) = EdgeFileFooter::decode(&finish.body).unwrap();
        assert_eq!(footer.key_count, 3);
        assert_eq!(footer.edge_count, 4);
        let ordinals = footer.find_kind(SECTION_EDGE_ORDINALS).unwrap();
        let ordinal_bytes =
            &finish.body[ordinals.offset as usize..(ordinals.offset + ordinals.length) as usize];
        let decoded_ordinals: Vec<u64> = ordinal_bytes
            .chunks_exact(8)
            .map(|row| u64::from_le_bytes(row.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_ordinals, vec![0, 2, 3, 4]);
        let page_directory = footer.find_kind(SECTION_PAGE_CHECKSUMS).unwrap();
        let page_bytes = &finish.body[page_directory.offset as usize
            ..(page_directory.offset + page_directory.length) as usize];
        let decoded = EdgePageChecksumDirectory::decode(page_bytes, &footer).unwrap();
        assert_eq!(decoded.page_size, EDGE_CHECKSUM_PAGE_BYTES);
        assert_eq!(decoded.sections.len() + 1, footer.sections.len());
        // Header decodes too.
        let header = EdgeFileHeader::decode(&finish.body).unwrap();
        assert_eq!(header.flags & FLAG_INVERSE_PARTNER, 0);
    }

    #[test]
    fn writer_rejects_unsorted_keys() {
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        w.append(record(key(2, 0), key(1, 0), 1)).unwrap();
        let err = w.append(record(key(1, 0), key(2, 0), 1)).unwrap_err();
        assert!(matches!(err, Error::Invariant(_)));
    }

    #[test]
    fn writer_rejects_duplicate_partner_in_key() {
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        let k = key(1, 0);
        let p = key(2, 0);
        w.append(record(k, p, 1)).unwrap();
        let err = w.append(record(k, p, 2)).unwrap_err();
        assert!(matches!(err, Error::Invariant(_)));
    }

    #[test]
    fn writer_sets_inverse_flag() {
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Inverse, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        w.append(record(key(1, 0), key(2, 0), 1)).unwrap();
        let finish = w.finish().unwrap();
        let header = EdgeFileHeader::decode(&finish.body).unwrap();
        assert!(header.flags & FLAG_INVERSE_PARTNER != 0);
    }

    #[tokio::test]
    async fn point_sidecar_preserves_absent_null_and_declared_precedence() {
        let mut opts =
            EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "Person", "Person");
        opts.declared_properties = vec!["absent".into(), "explicit_null".into(), "shadow".into()];
        let mut writer = EdgeSstWriter::new(opts);
        writer
            .append(EdgeRecord {
                key_id: key(1, 0),
                partner_id: key(2, 0),
                lsn: 42,
                tombstone: false,
                declared_properties: vec![None, Some("null".into()), Some(r#""declared""#.into())],
                overflow_json: Some(r#"{"extra":7,"shadow":"overflow"}"#.into()),
            })
            .unwrap();
        let build = writer.finish_with_point_index().unwrap();
        let point = build
            .point_index
            .expect("small forward SST receives a complete sidecar")
            .into_bytes()
            .unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("edge.epidx");
        store.put(&path, PutPayload::from(point)).await.unwrap();
        let (found, stats) =
            crate::sst::paged_index::probe_edge_points(store, path, &[(key(1, 0), key(2, 0))])
                .await
                .unwrap();
        assert_eq!(stats.index_entries, 1);
        let decoded = crate::sst::edges::point_index::decode(
            found.get(&(key(1, 0), key(2, 0))).unwrap(),
            true,
        )
        .unwrap();
        assert!(!decoded.properties.contains_key("absent"));
        assert_eq!(decoded.properties.get("explicit_null"), Some(&Value::Null));
        assert_eq!(decoded.properties.get("extra"), Some(&Value::I64(7)));
        assert_eq!(
            decoded.properties.get("shadow"),
            Some(&Value::Str("declared".into()))
        );
    }

    #[test]
    fn point_sidecar_caps_fail_explicitly_and_inverse_never_emits_one() {
        let mut by_entry = EdgeSstWriter::new(EdgeSstWriterOptions::new(
            EdgeDirection::Forward,
            "KNOWS",
            "P",
            "P",
        ));
        by_entry.point_index_max_entry_bytes = 256;
        by_entry.append(record(key(1, 0), key(2, 0), 1)).unwrap();
        let mut oversized = record(key(1, 1), key(2, 1), 2);
        oversized.overflow_json = Some(format!(r#"{{"blob":"{}"}}"#, "x".repeat(512)));
        assert!(
            by_entry.append(oversized).is_err(),
            "an operator cap must fail the write, never remove the accelerator"
        );

        let mut by_total = EdgeSstWriter::new(EdgeSstWriterOptions::new(
            EdgeDirection::Forward,
            "KNOWS",
            "P",
            "P",
        ));
        by_total.point_index_max_sst_bytes = EDGE_POINT_MIN_SERIALIZED_BYTES + 256;
        let mut cap_error = None;
        for n in 0..8 {
            if let Err(error) = by_total.append(record(key(1, n), key(2, n), n + 1)) {
                cap_error = Some(error);
                break;
            }
        }
        assert!(cap_error.is_some(), "the explicit total cap must fail");

        let mut inverse = EdgeSstWriter::new(EdgeSstWriterOptions::new(
            EdgeDirection::Inverse,
            "KNOWS",
            "P",
            "P",
        ));
        inverse.append(record(key(1, 0), key(2, 0), 1)).unwrap();
        assert!(inverse
            .finish_with_point_index()
            .unwrap()
            .point_index
            .is_none());
    }

    #[test]
    fn writer_emits_fence_index_above_threshold() {
        let mut opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        opts.fence_threshold = 4; // force fence index for tiny test
        opts.fence_stride = 2;
        let mut w = EdgeSstWriter::new(opts);
        for i in 0..8u64 {
            w.append(record(key(1, i), key(2, i), 100 + i)).unwrap();
        }
        let finish = w.finish().unwrap();
        let (footer, _) = EdgeFileFooter::decode(&finish.body).unwrap();
        assert!(footer.find_kind(SECTION_FENCE_INDEX).is_some());
    }

    #[test]
    fn writer_omits_fence_index_below_threshold() {
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        for i in 0..3u64 {
            w.append(record(key(1, i), key(2, i), 100)).unwrap();
        }
        let finish = w.finish().unwrap();
        let (footer, _) = EdgeFileFooter::decode(&finish.body).unwrap();
        assert!(footer.find_kind(SECTION_FENCE_INDEX).is_none());
    }

    #[test]
    fn writer_emits_tombstones_only_when_present() {
        // No tombstones path.
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        w.append(record(key(1, 0), key(2, 0), 1)).unwrap();
        let finish = w.finish().unwrap();
        let header = EdgeFileHeader::decode(&finish.body).unwrap();
        assert_eq!(header.flags & FLAG_HAS_TOMBSTONES, 0);
        let (footer, _) = EdgeFileFooter::decode(&finish.body).unwrap();
        assert!(footer.find_kind(SECTION_PER_EDGE_TOMBSTONES).is_none());

        // With tombstones path.
        let opts2 = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w2 = EdgeSstWriter::new(opts2);
        let mut r = record(key(1, 0), key(2, 0), 1);
        r.tombstone = true;
        w2.append(r).unwrap();
        let finish2 = w2.finish().unwrap();
        let header2 = EdgeFileHeader::decode(&finish2.body).unwrap();
        assert!(header2.flags & FLAG_HAS_TOMBSTONES != 0);
        let (footer2, _) = EdgeFileFooter::decode(&finish2.body).unwrap();
        assert!(footer2.find_kind(SECTION_PER_EDGE_TOMBSTONES).is_some());
        assert_eq!(finish2.stats.tombstone_count, 1);
    }

    #[test]
    fn skew_buckets_flag_only_set_on_true_super_nodes() {
        use crate::sst::edges::format::FLAG_SKEW_BUCKETS;

        // (a) Force the writer's encoding fallback to dense by giving a
        // single group with partner deltas wide enough that split loses.
        // This should NOT set the SKEW_BUCKETS flag.
        let mut opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        opts.skew_threshold = Some(10_000); // very high → never a true skew bucket here
        let mut w = EdgeSstWriter::new(opts);
        let src = key(1, 0);
        for i in 0..10u64 {
            // Partners spread across the full u64 top64 range so split loses
            // (~17 B/partner) vs dense (16 B/partner).
            let partner_top = i * (1u64 << 60);
            let mut p = [0u8; 16];
            p[..8].copy_from_slice(&partner_top.to_le_bytes());
            p[8..].copy_from_slice(&i.to_le_bytes());
            w.append(record(src, p, 100 + i)).unwrap();
        }
        let finish = w.finish().unwrap();
        let header = crate::sst::edges::format::EdgeFileHeader::decode(&finish.body).unwrap();
        assert_eq!(
            header.flags & FLAG_SKEW_BUCKETS,
            0,
            "encoding-fallback dense block must NOT set SKEW_BUCKETS"
        );

        // (b) Real super-node: degree exceeds the skew threshold.
        let mut opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        opts.skew_threshold = Some(2); // tiny threshold for the test
        let mut w = EdgeSstWriter::new(opts);
        let src = key(1, 0);
        for i in 0..5u64 {
            w.append(record(src, key(99, i), 100 + i)).unwrap();
        }
        let finish = w.finish().unwrap();
        let header = crate::sst::edges::format::EdgeFileHeader::decode(&finish.body).unwrap();
        assert!(
            header.flags & FLAG_SKEW_BUCKETS != 0,
            "true super-node (deg > threshold) must set SKEW_BUCKETS"
        );
    }

    #[test]
    fn writer_streams_partner_blocks_on_key_change() {
        // Regression for I3: after `append` returns, the writer must
        // already have flushed the previous key's partner block into the
        // monotonic partner spool. We can't measure RAM directly
        // in a unit test, but `record_count` + the bucket invariants tell
        // us the streaming pipeline is on the happy path.
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        assert_eq!(w.record_count(), 0);

        // First key with two partners.
        w.append(record(key(1, 0), key(2, 1), 100)).unwrap();
        w.append(record(key(1, 0), key(2, 2), 101)).unwrap();
        assert_eq!(w.record_count(), 2);
        assert_eq!(w.current_degree, 2);
        assert_eq!(w.partners.len, 0, "first key still open");

        // Crossing a key boundary: previous bucket should be drained.
        w.append(record(key(2, 0), key(2, 3), 102)).unwrap();
        assert_eq!(w.record_count(), 3);
        assert_eq!(w.current_degree, 1);
        assert!(w.partners.len > 0, "first bucket drained");
        assert_eq!(w.key_count, 1, "only the closed bucket counts so far");

        let finish = w.finish().unwrap();
        assert_eq!(finish.stats.key_count, 2);
        assert_eq!(finish.stats.edge_count, 3);
    }

    #[test]
    fn writer_handles_10k_records_with_overflow_strings() {
        // Smoke: streaming pipeline survives a workload large enough that
        // the old "Vec<EdgeRecord>" approach would have held ~10k strings.
        // We confirm the SST decodes back to the same edge count.
        //
        // Note: the `key` helper above uses little-endian, which loses
        // lexicographic monotonicity past 256 keys. Use a big-endian
        // counter here so the writer's sort-order check accepts the input.
        fn key_be(top: u64, bot: u64) -> [u8; 16] {
            let mut k = [0u8; 16];
            k[..8].copy_from_slice(&top.to_be_bytes());
            k[8..].copy_from_slice(&bot.to_be_bytes());
            k
        }
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "EDGE", "L", "R");
        let mut w = EdgeSstWriter::new(opts);
        for i in 0..10_000u64 {
            w.append(EdgeRecord {
                key_id: key_be(1, i / 4), // 4 partners per key
                partner_id: key_be(2, i),
                lsn: 100 + i,
                tombstone: i % 1000 == 0,
                declared_properties: vec![],
                overflow_json: Some(format!("{{\"i\":{i}}}")),
            })
            .unwrap();
        }
        let finish = w.finish().unwrap();
        assert_eq!(finish.stats.edge_count, 10_000);
        assert_eq!(finish.stats.key_count, 2_500);
        assert_eq!(finish.stats.tombstone_count, 10);
        let (footer, _) = EdgeFileFooter::decode(&finish.body).unwrap();
        assert!(footer
            .find(SECTION_PROPERTY_STREAM, OVERFLOW_JSON_NAME)
            .is_some());
    }

    #[test]
    fn overflow_section_emitted_when_any_record_has_overflow() {
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        w.append(record(key(1, 0), key(2, 0), 1)).unwrap();
        let mut r2 = record(key(1, 1), key(2, 0), 2);
        r2.overflow_json = Some(r#"{"city":"Quito"}"#.into());
        w.append(r2).unwrap();
        let finish = w.finish().unwrap();
        let (footer, _) = EdgeFileFooter::decode(&finish.body).unwrap();
        let header = EdgeFileHeader::decode(&finish.body).unwrap();
        assert!(header.flags & FLAG_HAS_PROPERTIES != 0);
        let s = footer
            .find(SECTION_PROPERTY_STREAM, OVERFLOW_JSON_NAME)
            .expect("overflow section missing");
        assert_eq!(s.codec, CODEC_PROPERTY_PAGED_ZSTD);
    }

    #[test]
    fn spooled_high_degree_100k_round_trips() {
        let mut opts =
            EdgeSstWriterOptions::new(EdgeDirection::Inverse, "CITES", "Articulo", "Articulo");
        opts.skew_threshold = Some(1024);
        let mut writer = EdgeSstWriter::new(opts);
        let hub = [0x44; 16];
        for ordinal in 0..100_001u128 {
            writer
                .append(EdgeRecord {
                    key_id: hub,
                    partner_id: ordinal.to_be_bytes(),
                    lsn: ordinal as u64 + 1,
                    tombstone: false,
                    declared_properties: Vec::new(),
                    overflow_json: None,
                })
                .unwrap();
        }
        assert_eq!(writer.current_degree, 100_001);
        assert_eq!(writer.current_partners.len, 100_001 * 16);
        let finish = writer.finish().unwrap();
        let reader = crate::sst::edges::reader::EdgeSstReader::open(finish.body).unwrap();
        let adjacency = reader.lookup(&hub).unwrap().unwrap();
        assert_eq!(adjacency.partners.len(), 100_001);
        assert_eq!(adjacency.partners[100_000], 100_000u128.to_be_bytes());
    }

    #[test]
    fn large_vector_like_properties_are_paged_and_round_trip() {
        let mut opts =
            EdgeSstWriterOptions::new(EdgeDirection::Inverse, "SIMILAR", "Articulo", "Articulo");
        opts.declared_properties = vec!["embedding_json".into()];
        let mut writer = EdgeSstWriter::new(opts);
        let large = format!(r#""{}""#, "0.125,".repeat(1_500));
        for ordinal in 0..2_048u128 {
            writer
                .append(EdgeRecord {
                    key_id: ordinal.to_be_bytes(),
                    partner_id: (ordinal + 10_000).to_be_bytes(),
                    lsn: ordinal as u64 + 1,
                    tombstone: false,
                    declared_properties: vec![Some(large.clone())],
                    overflow_json: None,
                })
                .unwrap();
        }
        let finish = writer.finish().unwrap();
        let reader = crate::sst::edges::reader::EdgeSstReader::open(finish.body).unwrap();
        let values = reader
            .read_declared_property_strings("embedding_json")
            .unwrap()
            .unwrap();
        assert_eq!(values.len(), 2_048);
        assert_eq!(values[0].as_deref(), Some(large.as_str()));
        assert_eq!(values[2_047].as_deref(), Some(large.as_str()));
    }

    #[test]
    fn contiguous_fixture_helper_fails_closed_above_its_cap() {
        let mut writer = EdgeSstWriter::new(EdgeSstWriterOptions::new(
            EdgeDirection::Inverse,
            "EDGE",
            "L",
            "R",
        ));
        writer.append(record([0x01; 16], [0x02; 16], 1)).unwrap();
        let build = writer.finish_with_point_index().unwrap();
        let cap = usize::try_from(build.body.size_bytes() - 1).unwrap();
        assert!(build.body.into_bytes(cap).is_err());
    }
}
