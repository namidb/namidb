//! `EdgeSstWriter`: orchestrates the encoding of a single edge SST.
//!
//! The writer accepts pre-sorted `EdgeRecord`s (sorted ascending by
//! `key_id`, then by `partner_id`) and emits the on-disk byte body in
//! one shot via [`EdgeSstWriter::finish`].
//!
//! Declared edge property streams (RFC-002 §3.2.7) are wired through:
//! each declared property name becomes its own `SECTION_PROPERTY_STREAM`
//! with a JSON-encoded `Value` payload per edge (one Utf8 column).
//! Properties NOT in the declared schema (or carried by overflow-only
//! edge types) fall back to the legacy single `__overflow_json` stream.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};
use crate::sst::bloom::{BloomFilter, BLOOM_OMIT_THRESHOLD_BYTES, DEFAULT_BITS_PER_KEY};
use crate::sst::edges::encoding::{write_offset, write_partner_block, OffsetWidth};
use crate::sst::edges::fence_index::{FenceIndex, DEFAULT_FENCE_STRIDE, FENCE_INDEX_THRESHOLD};
use crate::sst::edges::format::{
    EdgeFileFooter, EdgeFileHeader, SectionEntry, CODEC_NONE, CODEC_ZSTD, FLAG_HAS_PROPERTIES,
    FLAG_HAS_TOMBSTONES, FLAG_INVERSE_PARTNER, FLAG_SKEW_BUCKETS, HEADER_LEN, OVERFLOW_JSON_NAME,
    SECTION_FENCE_INDEX, SECTION_KEY_IDS, SECTION_OFFSETS, SECTION_PARTNERS, SECTION_PER_EDGE_LSN,
    SECTION_PER_EDGE_TOMBSTONES, SECTION_PROPERTY_STREAM,
};
use crate::sst::edges::EdgeDirection;
use crate::sst::stats::{DegreeHistogram, PropertyColumnStats};

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
    /// Force a specific skew threshold. `None` → bench-driven default:
    /// `max(1024, 4 * sqrt(key_count))`.
    pub skew_threshold: Option<usize>,
    pub fence_stride: u32,
    pub fence_threshold: u64,
    /// Bloom density (only emitted for SSTs larger than the omit threshold).
    pub bits_per_key: u8,
    pub expected_keys: u64,
    /// Compress the overflow + declared property streams with Zstd.
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
    /// Always empty in v1 of the writer — declared property streams are not
    /// yet wired through (see module docs).
    pub property_stats: Vec<PropertyColumnStats>,
    pub schema_version_min: u64,
    pub schema_version_max: u64,
}

/// Streaming writer. RAM usage is `O(max_degree_of_any_key)` for the live
/// partner bucket plus `O(output_bytes_seen_so_far)` for the monotonically-
/// growing section accumulators — independent of `edge_count`. This is the
/// fix for I3 of the bug audit.
///
/// The earlier implementation held every [`EdgeRecord`] (including its
/// owned `overflow_json: Option<String>`) in a `Vec` until `finish()`. For
/// SSTs with millions of edges and even sparse overflow strings that
/// staging cost dominated flush RAM. Now the writer drains records into
/// the output sections on every `append`, and the IPC stream of overflow
/// values is fed via small mini-batches so the per-record `String` is
/// dropped right after it lands in the IPC buffer.
#[derive(Debug)]
pub struct EdgeSstWriter {
    options: EdgeSstWriterOptions,
    /// Pre-computed at `new()` so streaming partner-block encoding can
    /// produce stable byte output without re-tuning per call.
    skew_threshold: usize,

    // ── current key bucket — emitted to the accumulators on key change ──
    current_key: Option<[u8; 16]>,
    current_partners: Vec<[u8; 16]>,
    last_partner_in_key: Option<[u8; 16]>,

    // ── monotonic accumulators (grow only with output, never with input) ──
    partners_bytes: Vec<u8>,
    /// One offset per key plus a final sentinel appended in `finish()`.
    offsets_values: Vec<u64>,
    key_ids_bytes: Vec<u8>,
    /// Per-edge LSN bytes in the order partners arrive.
    lsn_bytes: Vec<u8>,
    /// Per-edge tombstone bits, packed lsb-first; capacity matches
    /// `ceil(edge_count / 8)` after each append.
    tombstone_bits: Vec<u8>,

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
    /// Distinct key ids, retained to feed `FenceIndex::build` in finish().
    /// 16 B per key — same growth as `key_ids_bytes`, kept in struct form
    /// to skip a parse step at finalisation time.
    key_ids: Vec<[u8; 16]>,

    // ── overflow IPC stream (lazy + bufferised) ──
    overflow: PropertyStream,
    /// One stream per declared property (RFC-002 §3.2.7), in the order
    /// of `options.declared_properties`. Each holds JSON-encoded
    /// `Value` payloads as Utf8 columns.
    declared_streams: Vec<PropertyStream>,

    bloom: BloomFilter,
}

/// Per-property mini-batched Arrow IPC stream of JSON-encoded `Value`
/// payloads. Used for both `__overflow_json` (the legacy single stream)
/// and each declared property's named stream (RFC-002 §3.2.7).
///
/// Buffers up to `MINI_BATCH` rows in RAM before handing them to the
/// IPC writer so per-call overhead is amortised but no full edge_count's
/// worth of `String`s survives between `append`s.
struct PropertyStream {
    /// Stream name — `__overflow_json` for the catch-all bucket or the
    /// declared property's logical name (no `prop_` prefix).
    name: String,
    schema: Arc<ArrowSchema>,
    /// `None` until the first `append` actually arrives — keeps zero-edge
    /// SSTs from spending bytes on an empty IPC header.
    writer: Option<StreamWriter<Vec<u8>>>,
    pending: Vec<Option<String>>,
    any_value: bool,
}

impl std::fmt::Debug for PropertyStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropertyStream")
            .field("name", &self.name)
            .field("pending_len", &self.pending.len())
            .field("any_value", &self.any_value)
            .field("writer_active", &self.writer.is_some())
            .finish()
    }
}

const PROPERTY_STREAM_MINI_BATCH: usize = 1024;

impl PropertyStream {
    fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            name.clone(),
            ArrowDataType::Utf8,
            true,
        )]));
        Self {
            name,
            schema,
            writer: None,
            pending: Vec::with_capacity(PROPERTY_STREAM_MINI_BATCH),
            any_value: false,
        }
    }

    fn append(&mut self, value: Option<String>) -> Result<()> {
        if value.is_some() {
            self.any_value = true;
        }
        self.pending.push(value);
        if self.pending.len() >= PROPERTY_STREAM_MINI_BATCH {
            self.flush_batch()?;
        }
        Ok(())
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let strings: Vec<Option<&str>> = self.pending.iter().map(|s| s.as_deref()).collect();
        let column: ArrayRef = Arc::new(StringArray::from(strings));
        let batch = RecordBatch::try_new(self.schema.clone(), vec![column])
            .map_err(|e| Error::invariant(format!("property batch ({}): {e}", self.name)))?;
        if self.writer.is_none() {
            let buf: Vec<u8> = Vec::new();
            let writer = StreamWriter::try_new(buf, &self.schema)
                .map_err(|e| Error::invariant(format!("ipc writer ({}): {e}", self.name)))?;
            self.writer = Some(writer);
        }
        let writer = self.writer.as_mut().unwrap();
        writer
            .write(&batch)
            .map_err(|e| Error::invariant(format!("ipc write ({}): {e}", self.name)))?;
        self.pending.clear();
        Ok(())
    }

    /// `Ok(Some((bytes, codec)))` when the stream observed at least one
    /// non-null value; `Ok(None)` when every appended row was `None` —
    /// the writer skips the section entirely in that case.
    fn finish(mut self, compress: bool) -> Result<Option<(Vec<u8>, u8)>> {
        if !self.any_value {
            return Ok(None);
        }
        self.flush_batch()?;
        let mut writer = self.writer.ok_or_else(|| {
            Error::invariant(format!(
                "property stream {} had records but no writer",
                self.name
            ))
        })?;
        writer
            .finish()
            .map_err(|e| Error::invariant(format!("ipc finish ({}): {e}", self.name)))?;
        let raw = writer
            .into_inner()
            .map_err(|e| Error::invariant(format!("ipc into_inner ({}): {e}", self.name)))?;
        if compress {
            let compressed = zstd::stream::encode_all(&raw[..], 3)
                .map_err(|e| Error::invariant(format!("zstd encode ({}): {e}", self.name)))?;
            Ok(Some((compressed, CODEC_ZSTD)))
        } else {
            Ok(Some((raw, CODEC_NONE)))
        }
    }
}

impl EdgeSstWriter {
    pub fn new(options: EdgeSstWriterOptions) -> Self {
        let bloom =
            BloomFilter::with_capacity(options.expected_keys.max(1), options.bits_per_key.max(1));
        // Pick the skew threshold up-front from `expected_keys`. Used to
        // mean it was deferred to finish() and recomputed from the actual
        // key_count, but streaming needs deterministic partner-block
        // encoding at `append` time. Callers that know the exact
        // distribution can still pin `skew_threshold` via the options.
        let skew_threshold = options.skew_threshold.unwrap_or_else(|| {
            let sqrt = (options.expected_keys as f64).sqrt() as usize;
            (4 * sqrt).max(1024)
        });
        let declared_streams: Vec<PropertyStream> = options
            .declared_properties
            .iter()
            .map(PropertyStream::new)
            .collect();
        Self {
            options,
            skew_threshold,
            current_key: None,
            current_partners: Vec::new(),
            last_partner_in_key: None,
            partners_bytes: Vec::new(),
            offsets_values: Vec::new(),
            key_ids_bytes: Vec::new(),
            lsn_bytes: Vec::new(),
            tombstone_bits: Vec::new(),
            key_count: 0,
            edge_count: 0,
            tombstone_count: 0,
            min_lsn: u64::MAX,
            max_lsn: 0,
            any_skew_block: false,
            degree_histogram: DegreeHistogram::empty(),
            min_key_id: None,
            max_key_id: None,
            key_ids: Vec::new(),
            overflow: PropertyStream::new(OVERFLOW_JSON_NAME),
            declared_streams,
            bloom,
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
                self.flush_current_bucket();
            }
        }

        if self.current_key != Some(record.key_id) {
            self.current_key = Some(record.key_id);
            self.bloom.insert(&record.key_id);
        }
        self.current_partners.push(record.partner_id);
        self.last_partner_in_key = Some(record.partner_id);

        // Per-edge accumulators happen right here (one-pass).
        self.lsn_bytes.extend_from_slice(&record.lsn.to_le_bytes());
        let edge_index = self.edge_count as usize;
        push_bit(&mut self.tombstone_bits, edge_index, record.tombstone);
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

    /// Drain `current_partners` into the partner_block + offsets + key_ids
    /// accumulators and reset bucket state. Called on every key change and
    /// once at finalisation.
    fn flush_current_bucket(&mut self) {
        let Some(key) = self.current_key else {
            return;
        };
        if self.current_partners.is_empty() {
            return;
        }
        let deg = self.current_partners.len();
        let is_skew = deg > self.skew_threshold;
        self.offsets_values.push(self.partners_bytes.len() as u64);
        write_partner_block(
            &self.current_partners,
            self.skew_threshold,
            &mut self.partners_bytes,
        );
        if is_skew {
            self.any_skew_block = true;
        }
        self.degree_histogram.observe(deg as u64);
        self.key_ids_bytes.extend_from_slice(&key);
        self.key_ids.push(key);
        self.key_count += 1;
        if self.min_key_id.is_none() {
            self.min_key_id = Some(key);
        }
        self.max_key_id = Some(key);
        self.current_partners.clear();
        self.last_partner_in_key = None;
    }

    /// Serialise the SST body.
    pub fn finish(mut self) -> Result<EdgeSstFinish> {
        // Drain any open bucket.
        self.flush_current_bucket();

        let opts = &self.options;
        let key_count = self.key_count;
        let edge_count = self.edge_count;

        // Sentinel offset.
        self.offsets_values.push(self.partners_bytes.len() as u64);

        // ── Bitpack offsets ────────────────────────────────────────────
        let max_offset = *self.offsets_values.iter().max().unwrap_or(&0);
        let offset_width = OffsetWidth::for_max(max_offset);
        let mut offsets_bytes =
            Vec::with_capacity(self.offsets_values.len() * offset_width.bytes());
        for v in &self.offsets_values {
            write_offset(*v, offset_width, &mut offsets_bytes);
        }

        // Tombstone bytes: drop the section entirely if the SST has no
        // tombstones — that's the wire-format invariant the reader keys
        // off of via the `HAS_TOMBSTONES` flag.
        let has_tombstones = self.tombstone_count > 0;
        let tombstone_bytes = if has_tombstones {
            Some(self.tombstone_bits.clone())
        } else {
            None
        };

        // Fence index built from accumulated key_ids.
        let fence_bytes = if key_count > opts.fence_threshold {
            Some(FenceIndex::build(&self.key_ids, opts.fence_stride).encode())
        } else {
            None
        };

        // Overflow section.
        let overflow_section = self.overflow.finish(opts.compress_property_streams)?;
        // Declared property sections (RFC-002 §3.2.7). One emitted Arrow
        // IPC stream per declared property — the order matches
        // `options.declared_properties`. Streams whose every appended
        // value was `None` are skipped (Ok(None) from `PropertyStream::finish`).
        let declared_property_names: Vec<String> = self.options.declared_properties.clone();
        let mut declared_sections: Vec<(String, Vec<u8>, u8)> =
            Vec::with_capacity(self.declared_streams.len());
        for (name, stream) in declared_property_names
            .iter()
            .zip(std::mem::take(&mut self.declared_streams))
        {
            if let Some((body, codec)) = stream.finish(opts.compress_property_streams)? {
                declared_sections.push((name.clone(), body, codec));
            }
        }

        // Min/max LSN sanity for empty SST.
        let min_lsn = if edge_count == 0 { 0 } else { self.min_lsn };
        let max_lsn = self.max_lsn;
        let min_key_id = self.min_key_id.unwrap_or([0u8; 16]);
        let max_key_id = self.max_key_id.unwrap_or([0u8; 16]);

        // ── Compose the file body ─────────────────────────────────────
        let mut file = Vec::new();

        let mut flags = 0u32;
        if matches!(opts.direction, EdgeDirection::Inverse) {
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
        EdgeFileHeader::new(&opts.edge_type, &opts.src_label, &opts.dst_label, flags)
            .encode(&mut file);
        debug_assert_eq!(file.len(), HEADER_LEN);

        let mut sections = vec![
            emit_section(
                SECTION_KEY_IDS,
                "",
                CODEC_NONE,
                &self.key_ids_bytes,
                &mut file,
            ),
            emit_section(SECTION_OFFSETS, "", CODEC_NONE, &offsets_bytes, &mut file),
            emit_section(
                SECTION_PARTNERS,
                "",
                CODEC_NONE,
                &self.partners_bytes,
                &mut file,
            ),
            emit_section(
                SECTION_PER_EDGE_LSN,
                "",
                CODEC_NONE,
                &self.lsn_bytes,
                &mut file,
            ),
        ];
        if let Some(tb) = tombstone_bytes.as_ref() {
            sections.push(emit_section(
                SECTION_PER_EDGE_TOMBSTONES,
                "",
                CODEC_NONE,
                tb,
                &mut file,
            ));
        }
        if let Some(fb) = fence_bytes.as_ref() {
            sections.push(emit_section(
                SECTION_FENCE_INDEX,
                "",
                CODEC_NONE,
                fb,
                &mut file,
            ));
        }
        if let Some((body, codec)) = overflow_section.as_ref() {
            sections.push(emit_section(
                SECTION_PROPERTY_STREAM,
                OVERFLOW_JSON_NAME,
                *codec,
                body,
                &mut file,
            ));
        }
        for (name, body, codec) in &declared_sections {
            sections.push(emit_section(
                SECTION_PROPERTY_STREAM,
                name,
                *codec,
                body,
                &mut file,
            ));
        }

        let footer = EdgeFileFooter {
            sections,
            key_count,
            edge_count,
            offsets_bits: offset_width.as_bits(),
            min_key_id,
            max_key_id,
            min_lsn,
            max_lsn,
            schema_version_min: opts.schema_version,
            schema_version_max: opts.schema_version,
        };
        footer.encode(&mut file)?;

        let body = Bytes::from(file);
        let body_len = body.len();
        let bloom = if body_len as u64 >= BLOOM_OMIT_THRESHOLD_BYTES {
            Some(self.bloom)
        } else {
            None
        };

        let stats = EdgeSstStats {
            direction: opts.direction,
            key_count,
            edge_count,
            tombstone_count: self.tombstone_count,
            min_key_id,
            max_key_id,
            min_lsn,
            max_lsn,
            degree_histogram: self.degree_histogram,
            property_stats: Vec::new(),
            schema_version_min: opts.schema_version,
            schema_version_max: opts.schema_version,
        };

        Ok(EdgeSstFinish { body, stats, bloom })
    }
}

/// Append `bit` at position `count` in `bits`, growing the buffer with a
/// zero byte whenever a fresh byte is needed.
fn push_bit(bits: &mut Vec<u8>, count: usize, bit: bool) {
    if count % 8 == 0 {
        bits.push(0);
    }
    if bit {
        bits[count / 8] |= 1u8 << (count % 8);
    }
}

/// Append `body` to `file` at its current end and return a fully-populated
/// `SectionEntry` describing the resulting byte range.
fn emit_section(kind: u16, name: &str, codec: u8, body: &[u8], file: &mut Vec<u8>) -> SectionEntry {
    let offset = file.len() as u64;
    let length = body.len() as u64;
    let xxhash3_64 = xxh3_64(body);
    file.extend_from_slice(body);
    SectionEntry {
        kind,
        offset,
        length,
        codec,
        xxhash3_64,
        name: name.to_string(),
    }
}

// The earlier `encode_overflow_stream` helper has been folded into
// `OverflowStream` so the writer can emit mini-batches incrementally
// instead of buffering an entire edge_count's worth of `String`s.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sst::edges::format::{EdgeFileHeader, FLAG_INVERSE_PARTNER};

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
        // monotonic `partners_bytes` buffer. We can't measure RAM directly
        // in a unit test, but `record_count` + the bucket invariants tell
        // us the streaming pipeline is on the happy path.
        let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
        let mut w = EdgeSstWriter::new(opts);
        assert_eq!(w.record_count(), 0);

        // First key with two partners.
        w.append(record(key(1, 0), key(2, 1), 100)).unwrap();
        w.append(record(key(1, 0), key(2, 2), 101)).unwrap();
        assert_eq!(w.record_count(), 2);
        assert_eq!(w.current_partners.len(), 2);
        assert!(w.partners_bytes.is_empty(), "first key still open");

        // Crossing a key boundary: previous bucket should be drained.
        w.append(record(key(2, 0), key(2, 3), 102)).unwrap();
        assert_eq!(w.record_count(), 3);
        assert_eq!(w.current_partners.len(), 1);
        assert!(!w.partners_bytes.is_empty(), "first bucket drained");
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
        assert_eq!(s.codec, CODEC_ZSTD);
    }
}
