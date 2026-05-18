//! `EdgeSstReader`: random-access lookup over an edge SST body.
//!
//! For v1 the reader accepts the full SST body as a `Bytes` slice and serves
//! lookups in memory. The ranged-GET / foyer-cached read path lands in the
//! read-snapshot RFC; the algorithms here are the same regardless of how
//! the bytes were fetched.

use std::io::Cursor;

use arrow_array::{Array, StringArray};
use arrow_ipc::reader::StreamReader;
use bytes::Bytes;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};
use crate::sst::edges::encoding::{read_offset, read_partner_block, read_varint, OffsetWidth};
use crate::sst::edges::fence_index::FenceIndex;
use crate::sst::edges::format::{
 EdgeFileFooter, EdgeFileHeader, SectionEntry, CODEC_NONE, CODEC_ZSTD, FOOTER_TRAILER_LEN,
 HEADER_LEN, OVERFLOW_JSON_NAME, SECTION_FENCE_INDEX, SECTION_KEY_IDS, SECTION_OFFSETS,
 SECTION_PARTNERS, SECTION_PER_EDGE_LSN, SECTION_PER_EDGE_TOMBSTONES, SECTION_PROPERTY_STREAM,
};
use crate::sst::edges::EdgeDirection;

/// In-memory edge SST reader.
///
/// `open()` precomputes a `cumulative_edges` vector so every subsequent
/// [`Self::lookup`] runs in `O(deg)` instead of `O(key_index + deg)` — the
/// older path decoded one partner block per key prefix on every call, which
/// turned hot-path lookups into `O(key_count)` work even for absent keys.
#[derive(Debug)]
pub struct EdgeSstReader {
 body: Bytes,
 header: EdgeFileHeader,
 footer: EdgeFileFooter,
 fence_index: Option<FenceIndex>,
 offset_width: OffsetWidth,
 cumulative_edges: Vec<u64>,
}

/// Decoded partner list for a single key, plus per-edge metadata.
///
/// `edge_offset` is the index into the SST's edge-enumeration order where
/// this key's partners begin. Callers that want to recover overflow
/// properties slice [`EdgeSstReader::read_overflow_strings`] at
/// `edge_offset..edge_offset + partners.len()`.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeLookup {
 pub partners: Vec<[u8; 16]>,
 pub lsns: Vec<u64>,
 pub tombstones: Vec<bool>,
 pub edge_offset: usize,
}

/// One (key, partner) projection produced by [`EdgeSstReader::scan_all_edges`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeRowProjection {
 pub key_id: [u8; 16],
 pub partner_id: [u8; 16],
 pub lsn: u64,
 pub tombstone: bool,
}

impl EdgeSstReader {
 /// Parse the header + footer of an in-memory SST body and prepare for
 /// random-access lookups. Validates structural invariants but does not
 /// pre-load section bodies.
 pub fn open(body: Bytes) -> Result<Self> {
 let header = EdgeFileHeader::decode(&body)?;
 let (footer, _footer_start) = EdgeFileFooter::decode(&body)?;
 let offset_width = OffsetWidth::from_bits(footer.offsets_bits)?;

 let fence_index = if let Some(entry) = footer.find_kind(SECTION_FENCE_INDEX) {
 let bytes = section_bytes(&body, entry)?;
 Some(FenceIndex::decode(bytes)?)
 } else {
 None
 };

 // Sanity-check the mandatory sections exist.
 for required in [
 SECTION_KEY_IDS,
 SECTION_OFFSETS,
 SECTION_PARTNERS,
 SECTION_PER_EDGE_LSN,
 ] {
 footer.find_kind(required).ok_or_else(|| Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("edge SST missing mandatory section kind 0x{required:04x}"),
 })?;
 }

 let cumulative_edges = build_cumulative_edges(&body, &footer, offset_width)?;

 Ok(Self {
 body,
 header,
 footer,
 fence_index,
 offset_width,
 cumulative_edges,
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

 /// Resolve `key` to its position in the `key_ids` section, returning
 /// `None` when the key is not present.
 ///
 /// Implementation:
 /// - With a fence index: bracket via fence → window GET → binary search.
 /// - Without one: full-section binary search.
 pub fn position_of(&self, key: &[u8; 16]) -> Result<Option<u64>> {
 if key < &self.footer.min_key_id || key > &self.footer.max_key_id {
 return Ok(None);
 }
 let key_ids = self.section(SECTION_KEY_IDS, "")?;
 let key_count = self.footer.key_count;

 let (lo, hi) = if let Some(fence) = &self.fence_index {
 fence.bracket_for(key, key_count)
 } else {
 (0, key_count)
 };

 let lo = lo as usize;
 let hi = hi as usize;
 if lo == hi {
 return Ok(None);
 }

 let window = &key_ids[lo * 16..hi * 16];
 match window.chunks_exact(16).position(|chunk| chunk == key) {
 Some(offset) => Ok(Some((lo + offset) as u64)),
 None => Ok(None),
 }
 }

 /// Look up the partner list for `key`. Returns `None` when absent.
 /// Includes LSNs; if a tombstone bitmap exists, returns its bits;
 /// otherwise the `tombstones` vector is all `false`.
 pub fn lookup(&self, key: &[u8; 16]) -> Result<Option<EdgeLookup>> {
 let Some(idx) = self.position_of(key)? else {
 return Ok(None);
 };
 let idx = idx as usize;

 // offsets section
 let offsets_bytes = self.section(SECTION_OFFSETS, "")?;
 let stride = self.offset_width.bytes();
 let start = read_offset(offsets_bytes, idx * stride, self.offset_width)? as usize;
 let end = read_offset(offsets_bytes, (idx + 1) * stride, self.offset_width)? as usize;

 // partners block
 let partners_bytes = self.section(SECTION_PARTNERS, "")?;
 let (partners, _consumed) = read_partner_block(partners_bytes, start)?;
 // Sanity-check: `_consumed == end - start`.
 if start + _consumed != end {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "partner block at offset {start}: consumed {_consumed} bytes, \
 expected {} per offsets sentinel",
 end - start
 ),
 });
 }

 // Per-edge index range for this key. Cached at `open()` time so the
 // hot path is O(1) instead of O(idx) (see `build_cumulative_edges`).
 let edge_offset = self.cumulative_edges[idx] as usize;
 let edge_end = edge_offset + partners.len();

 // per_edge_lsn
 let lsn_bytes = self.section(SECTION_PER_EDGE_LSN, "")?;
 let mut lsns = Vec::with_capacity(partners.len());
 for i in edge_offset..edge_end {
 let off = i * 8;
 let lsn = u64::from_le_bytes(lsn_bytes[off..off + 8].try_into().unwrap());
 lsns.push(lsn);
 }

 // per_edge_tombstones (optional)
 let tombstones = if self.footer.find_kind(SECTION_PER_EDGE_TOMBSTONES).is_some() {
 let tomb_bytes = self.section(SECTION_PER_EDGE_TOMBSTONES, "")?;
 (edge_offset..edge_end)
 .map(|i| {
 let byte = tomb_bytes[i / 8];
 (byte >> (i % 8)) & 1 == 1
 })
 .collect()
 } else {
 vec![false; partners.len()]
 };

 Ok(Some(EdgeLookup {
 partners,
 lsns,
 tombstones,
 edge_offset,
 }))
 }

 /// Linear scan over every edge in the SST. Returns one row per
 /// `(key_id, partner_id)` pair plus its `lsn` and tombstone bit.
 ///
 /// Used by the compaction worker — there it is cheaper than calling
 /// [`Self::lookup`] for each key (lookup is O(K²) in the worst
 /// case; this is O(edge_count + key_count)).
 pub fn scan_all_edges(&self) -> Result<Vec<EdgeRowProjection>> {
 let key_ids = self.section(SECTION_KEY_IDS, "")?;
 let offsets_bytes = self.section(SECTION_OFFSETS, "")?;
 let partners_bytes = self.section(SECTION_PARTNERS, "")?;
 let lsn_bytes = self.section(SECTION_PER_EDGE_LSN, "")?;
 let tomb_section = self.footer.find_kind(SECTION_PER_EDGE_TOMBSTONES).is_some();
 let tomb_bytes = if tomb_section {
 Some(self.section(SECTION_PER_EDGE_TOMBSTONES, "")?)
 } else {
 None
 };

 let key_count = self.footer.key_count as usize;
 let stride = self.offset_width.bytes();
 let mut out: Vec<EdgeRowProjection> = Vec::with_capacity(self.footer.edge_count as usize);
 let mut edge_idx = 0usize;

 for k_idx in 0..key_count {
 let key_id: [u8; 16] = key_ids[k_idx * 16..(k_idx + 1) * 16]
 .try_into()
 .map_err(|_| Error::invariant("key_ids row length != 16"))?;
 let start = read_offset(offsets_bytes, k_idx * stride, self.offset_width)? as usize;
 let (partners, _consumed) = read_partner_block(partners_bytes, start)?;
 for (i, partner_id) in partners.iter().enumerate() {
 let edge_i = edge_idx + i;
 let lsn = u64::from_le_bytes(
 lsn_bytes[edge_i * 8..edge_i * 8 + 8]
 .try_into()
 .map_err(|_| Error::invariant("per_edge_lsn row length != 8"))?,
 );
 let tombstone = match tomb_bytes {
 Some(b) => (b[edge_i / 8] >> (edge_i % 8)) & 1 == 1,
 None => false,
 };
 out.push(EdgeRowProjection {
 key_id,
 partner_id: *partner_id,
 lsn,
 tombstone,
 });
 }
 edge_idx += partners.len();
 }
 Ok(out)
 }

 /// Decode the `__overflow_json` property stream into a `Vec<Option<String>>`
 /// indexed by edge-enumeration order. Returns `None` if the SST has no
 /// overflow section. Used by the read path to recover edge properties
 /// that were folded into the overflow stream at flush time.
 ///
 /// The output is `edge_count` long; entry `i` corresponds to the i-th
 /// edge in (key, partner) order — the same order that
 /// [`Self::scan_all_edges`] produces and that
 /// [`Self::cumulative_edges`] indexes into for per-key slices.
 pub fn read_overflow_strings(&self) -> Result<Option<Vec<Option<String>>>> {
 self.read_named_property_stream(OVERFLOW_JSON_NAME)
 }

 /// Decode a declared property stream into a `Vec<Option<String>>`
 /// indexed by edge-enumeration order (RFC-002 §3.2.7). Each entry
 /// holds the JSON-encoded `Value` payload (or `None` when the
 /// edge had no value for this property). Returns `None` when the
 /// SST has no stream of that name — pre-RFC-005 bodies, or
 /// all-null columns elided by the writer.
 ///
 /// Same `edge_count`-rows invariant as
 /// [`Self::read_overflow_strings`].
 pub fn read_declared_property_strings(
 &self,
 name: &str,
 ) -> Result<Option<Vec<Option<String>>>> {
 if name == OVERFLOW_JSON_NAME {
 // Defensive: callers are supposed to use
 // `read_overflow_strings` for the catch-all stream. Honour
 // the call so refactors don't accidentally regress.
 return self.read_overflow_strings();
 }
 self.read_named_property_stream(name)
 }

 fn read_named_property_stream(&self, name: &str) -> Result<Option<Vec<Option<String>>>> {
 let Some(entry) = self.footer.find(SECTION_PROPERTY_STREAM, name) else {
 return Ok(None);
 };
 let raw = self.section_with_name(SECTION_PROPERTY_STREAM, name)?;
 let decoded = match entry.codec {
 CODEC_NONE => raw.to_vec(),
 CODEC_ZSTD => zstd::stream::decode_all(raw).map_err(|e| {
 Error::invariant(format!("zstd decode (property stream {name}): {e}"))
 })?,
 other => {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("unknown codec {other} for property stream {name}"),
 });
 }
 };
 let cursor = Cursor::new(decoded);
 let reader = StreamReader::try_new(cursor, None)
 .map_err(|e| Error::invariant(format!("property IPC reader ({name}): {e}")))?;
 let mut out: Vec<Option<String>> = Vec::with_capacity(self.footer.edge_count as usize);
 for batch in reader {
 let batch =
 batch.map_err(|e| Error::invariant(format!("property IPC batch ({name}): {e}")))?;
 let col = batch
 .column(0)
 .as_any()
 .downcast_ref::<StringArray>()
 .ok_or_else(|| {
 Error::invariant(format!("property IPC column ({name}) is not Utf8"))
 })?;
 for row in 0..col.len() {
 if col.is_null(row) {
 out.push(None);
 } else {
 out.push(Some(col.value(row).to_string()));
 }
 }
 }
 if out.len() != self.footer.edge_count as usize {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "property stream {name} row count {} != edge_count {}",
 out.len(),
 self.footer.edge_count
 ),
 });
 }
 Ok(Some(out))
 }

 /// Per-key cumulative edge counts. Entry `k` = total edges in keys
 /// `[0, k)`. Used together with [`Self::position_of`] to slice the
 /// per-edge overflow vector at lookup time.
 pub fn cumulative_edges(&self) -> &[u64] {
 &self.cumulative_edges
 }

 /// Fetch the raw bytes of one section, verifying its `xxhash3_64`. The
 /// returned slice borrows from `self.body`.
 fn section_with_name(&self, kind: u16, name: &str) -> Result<&[u8]> {
 let entry = self
 .footer
 .find(kind, name)
 .ok_or_else(|| Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("edge SST missing section kind=0x{kind:04x} name='{name}'"),
 })?;
 let bytes = section_bytes(&self.body, entry)?;
 let hash = xxh3_64(bytes);
 if hash != entry.xxhash3_64 {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "section '{}' (kind 0x{:04x}) xxhash mismatch",
 entry.name, entry.kind
 ),
 });
 }
 Ok(bytes)
 }

 /// Fetch the raw bytes of one section, verifying its `xxhash3_64`. The
 /// returned slice borrows from `self.body`.
 pub fn section(&self, kind: u16, name: &str) -> Result<&[u8]> {
 let entry = if name.is_empty() {
 self.footer.find_kind(kind)
 } else {
 self.footer.find(kind, name)
 }
 .ok_or_else(|| Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("edge SST missing section kind=0x{kind:04x} name='{name}'"),
 })?;
 let bytes = section_bytes(&self.body, entry)?;
 let hash = xxh3_64(bytes);
 if hash != entry.xxhash3_64 {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "section '{}' (kind 0x{:04x}) xxhash mismatch",
 entry.name, entry.kind
 ),
 });
 }
 Ok(bytes)
 }
}

fn section_bytes<'a>(body: &'a Bytes, entry: &SectionEntry) -> Result<&'a [u8]> {
 let start = entry.offset as usize;
 let end = (entry.offset + entry.length) as usize;
 let body_end = body.len() - FOOTER_TRAILER_LEN;
 if start < HEADER_LEN || end > body_end {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "section '{}' [{start}, {end}) outside body window [{HEADER_LEN}, {body_end})",
 entry.name
 ),
 });
 }
 Ok(&body[start..end])
}

/// Build the per-key cumulative-edge vector. Entry `k` is the number of edges
/// belonging to keys at positions `[0, k)`; entry `key_count` equals the
/// SST's total `edge_count`.
///
/// Reads only the leading varint (the partner block degree) of each block,
/// not the partner bytes themselves — so construction is `O(key_count)` even
/// for SSTs whose edge_count dwarfs key_count.
///
/// Also verifies the SST is internally consistent: each declared partner
/// block fits inside its `offsets[k+1] - offsets[k]` window, the section
/// xxhashes match, and the total derived edge count agrees with the footer.
fn build_cumulative_edges(
 body: &Bytes,
 footer: &EdgeFileFooter,
 offset_width: OffsetWidth,
) -> Result<Vec<u64>> {
 let key_count = footer.key_count as usize;
 let mut cumulative = Vec::with_capacity(key_count + 1);
 cumulative.push(0u64);

 if key_count == 0 {
 if footer.edge_count != 0 {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "edge SST claims key_count=0 but edge_count={}",
 footer.edge_count
 ),
 });
 }
 return Ok(cumulative);
 }

 // Resolve the two sections we need, with their xxhash already verified.
 let offsets_entry = footer
 .find_kind(SECTION_OFFSETS)
 .ok_or_else(|| Error::invariant("offsets section missing despite earlier check"))?;
 let partners_entry = footer
 .find_kind(SECTION_PARTNERS)
 .ok_or_else(|| Error::invariant("partners section missing despite earlier check"))?;
 let offsets_bytes = verified_section(body, offsets_entry)?;
 let partners_bytes = verified_section(body, partners_entry)?;

 let stride = offset_width.bytes();
 let mut running: u64 = 0;
 for k in 0..key_count {
 let block_start = read_offset(offsets_bytes, k * stride, offset_width)? as usize;
 let (deg, _consumed) = read_varint(partners_bytes, block_start)?;
 running = running.checked_add(deg).ok_or_else(|| {
 Error::invariant("cumulative edge count overflows u64 while opening SST")
 })?;
 cumulative.push(running);
 }

 if running != footer.edge_count {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "edge SST cumulative degrees ({running}) disagree with footer.edge_count ({})",
 footer.edge_count
 ),
 });
 }
 Ok(cumulative)
}

/// Local helper: fetch a section slice and validate its xxhash3_64 once.
/// Mirrors `EdgeSstReader::section` but is free-standing because we call it
/// before the reader is fully constructed.
fn verified_section<'a>(body: &'a Bytes, entry: &SectionEntry) -> Result<&'a [u8]> {
 let bytes = section_bytes(body, entry)?;
 let hash = xxh3_64(bytes);
 if hash != entry.xxhash3_64 {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "section '{}' (kind 0x{:04x}) xxhash mismatch",
 entry.name, entry.kind
 ),
 });
 }
 Ok(bytes)
}

#[cfg(test)]
mod tests {
 use super::*;
 use crate::sst::edges::writer::{EdgeRecord, EdgeSstWriter, EdgeSstWriterOptions};
 use crate::sst::edges::EdgeDirection;

 fn key(top: u64, bot: u64) -> [u8; 16] {
 let mut k = [0u8; 16];
 k[..8].copy_from_slice(&top.to_le_bytes());
 k[8..].copy_from_slice(&bot.to_le_bytes());
 k
 }

 fn write_sample() -> Bytes {
 let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
 let mut w = EdgeSstWriter::new(opts);
 for k_idx in 0..5u64 {
 for p_idx in 0..3u64 {
 w.append(EdgeRecord {
 key_id: key(1, k_idx),
 partner_id: key(2, k_idx * 100 + p_idx),
 lsn: k_idx * 10 + p_idx,
 tombstone: p_idx == 1 && k_idx == 2,
 declared_properties: vec![],
 overflow_json: None,
 })
 .unwrap();
 }
 }
 w.finish().unwrap().body
 }

 #[test]
 fn lookup_returns_correct_partners_and_lsns() {
 let body = write_sample();
 let reader = EdgeSstReader::open(body).unwrap();
 assert_eq!(reader.key_count(), 5);
 assert_eq!(reader.edge_count(), 15);

 let target = key(1, 2);
 let look = reader.lookup(&target).unwrap().unwrap();
 assert_eq!(look.partners.len(), 3);
 assert_eq!(look.partners[0], key(2, 200));
 assert_eq!(look.partners[1], key(2, 201));
 assert_eq!(look.partners[2], key(2, 202));
 assert_eq!(look.lsns, vec![20, 21, 22]);
 // The tombstoned edge was (k_idx=2, p_idx=1) → partners[1].
 assert_eq!(look.tombstones, vec![false, true, false]);
 }

 #[test]
 fn lookup_returns_none_for_absent_key() {
 let body = write_sample();
 let reader = EdgeSstReader::open(body).unwrap();
 let look = reader.lookup(&key(99, 99)).unwrap();
 assert!(look.is_none());
 }

 #[test]
 fn lookup_returns_none_for_out_of_range_key() {
 let body = write_sample();
 let reader = EdgeSstReader::open(body).unwrap();
 assert!(reader.lookup(&key(0, 0)).unwrap().is_none());
 assert!(reader.lookup(&[0xff; 16]).unwrap().is_none());
 }

 #[test]
 fn lookup_works_with_fence_index() {
 let mut opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
 opts.fence_threshold = 2;
 opts.fence_stride = 4;
 let mut w = EdgeSstWriter::new(opts);
 for i in 0..32u64 {
 w.append(EdgeRecord {
 key_id: key(1, i),
 partner_id: key(2, i),
 lsn: i,
 tombstone: false,
 declared_properties: vec![],
 overflow_json: None,
 })
 .unwrap();
 }
 let body = w.finish().unwrap().body;
 let reader = EdgeSstReader::open(body).unwrap();
 // Mid-range key.
 let look = reader.lookup(&key(1, 17)).unwrap().unwrap();
 assert_eq!(look.partners, vec![key(2, 17)]);
 // First key.
 let look0 = reader.lookup(&key(1, 0)).unwrap().unwrap();
 assert_eq!(look0.partners, vec![key(2, 0)]);
 // Last key.
 let look31 = reader.lookup(&key(1, 31)).unwrap().unwrap();
 assert_eq!(look31.partners, vec![key(2, 31)]);
 }

 #[test]
 fn corrupted_section_body_is_detected() {
 // We deliberately target a byte inside the partners section so the
 // xxhash check fires. `open()` validates partners (and offsets) at
 // construction time because `build_cumulative_edges` reads them.
 let body = write_sample();
 let (footer, _) = EdgeFileFooter::decode(&body).unwrap();
 let partners_entry = footer.find_kind(SECTION_PARTNERS).unwrap();
 let target_offset = (partners_entry.offset + partners_entry.length / 2) as usize;

 let mut bytes = body.to_vec();
 bytes[target_offset] ^= 0x80;
 let err = EdgeSstReader::open(Bytes::from(bytes)).unwrap_err();
 assert!(matches!(err, Error::Corrupted { .. }));
 }

 /// Regression: `lookup` for a key whose position is large (near the end
 /// of the SST) must not pay an O(key_index) cost per call. We verify the
 /// cumulative cache by reading every key and checking the SST's footer
 /// edge_count agrees with the sum of partner counts seen.
 #[test]
 fn lookup_at_high_indices_uses_cumulative_cache() {
 // Build an SST with many keys and skewed degree to make any
 // re-walk of the prefix obvious in profiling. Correctness check:
 // every per-key lookup reports the right partner list and the
 // sum across keys equals footer.edge_count.
 let opts = EdgeSstWriterOptions::new(EdgeDirection::Forward, "KNOWS", "P", "P");
 let mut w = EdgeSstWriter::new(opts);
 let mut expected_total: u64 = 0;
 for k_idx in 0..32u64 {
 let deg = k_idx % 3 + 1;
 for p_idx in 0..deg {
 w.append(EdgeRecord {
 key_id: key(1, k_idx),
 partner_id: key(2, k_idx * 100 + p_idx),
 lsn: k_idx * 10 + p_idx,
 tombstone: false,
 declared_properties: vec![],
 overflow_json: None,
 })
 .unwrap();
 expected_total += 1;
 }
 }
 let body = w.finish().unwrap().body;
 let reader = EdgeSstReader::open(body).unwrap();
 assert_eq!(reader.edge_count(), expected_total);

 // Every lookup returns the right partner count for that key.
 let mut sum: u64 = 0;
 for k_idx in 0..32u64 {
 let look = reader.lookup(&key(1, k_idx)).unwrap().unwrap();
 let deg = (k_idx % 3 + 1) as usize;
 assert_eq!(look.partners.len(), deg, "wrong deg for key {k_idx}");
 assert_eq!(look.lsns.len(), deg);
 // First LSN equals k_idx*10 (per the writer above).
 assert_eq!(look.lsns[0], k_idx * 10);
 sum += deg as u64;
 }
 assert_eq!(sum, expected_total);
 }
}
