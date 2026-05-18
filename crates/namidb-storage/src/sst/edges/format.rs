//! Wire layout: file header + footer + section table.
//!
//! Defined by [RFC-002](../../../../../docs/rfc/002-sst-format.md) §3.2.1
//! and §3.2.8.

use bytes::Bytes;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};

// ─── magic + versions ───────────────────────────────────────────────────

pub const HEADER_MAGIC: [u8; 8] = *b"TGEDGE\0\0";
pub const FOOTER_MAGIC: [u8; 8] = *b"TGEDGE\xFE\xEF";
pub const HEADER_LEN: usize = 64;
pub const FORMAT_MAJOR: u8 = 1;
pub const FORMAT_MINOR: u8 = 0;

// ─── flag bits ──────────────────────────────────────────────────────────

pub const FLAG_HAS_PROPERTIES: u32 = 1 << 0;
pub const FLAG_HAS_TOMBSTONES: u32 = 1 << 1;
pub const FLAG_SKEW_BUCKETS: u32 = 1 << 2;
pub const FLAG_INVERSE_PARTNER: u32 = 1 << 3;
pub const FLAGS_RESERVED_MASK: u32 = !0u32 << 4; // bits 4..31

// ─── section kinds ──────────────────────────────────────────────────────

pub const SECTION_KEY_IDS: u16 = 0x0001;
pub const SECTION_OFFSETS: u16 = 0x0002;
pub const SECTION_PARTNERS: u16 = 0x0003;
pub const SECTION_PER_EDGE_LSN: u16 = 0x0004;
pub const SECTION_PER_EDGE_TOMBSTONES: u16 = 0x0005;
pub const SECTION_FENCE_INDEX: u16 = 0x0006;
pub const SECTION_PROPERTY_STREAM: u16 = 0x0100;

pub const CODEC_NONE: u8 = 0;
pub const CODEC_ZSTD: u8 = 1;

/// The overflow-property reserved name (mirrors the node SST convention).
pub const OVERFLOW_JSON_NAME: &str = "__overflow_json";

// ─── header ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeFileHeader {
 pub format_major: u8,
 pub format_minor: u8,
 pub flags: u32,
 pub edge_type_id: [u8; 16],
 pub src_label_id: [u8; 16],
 pub dst_label_id: [u8; 16],
}

impl EdgeFileHeader {
 pub fn new(edge_type: &str, src_label: &str, dst_label: &str, flags: u32) -> Self {
 Self {
 format_major: FORMAT_MAJOR,
 format_minor: FORMAT_MINOR,
 flags,
 edge_type_id: blake3_short(edge_type),
 src_label_id: blake3_short(src_label),
 dst_label_id: blake3_short(dst_label),
 }
 }

 pub fn encode(&self, buf: &mut Vec<u8>) {
 let start = buf.len();
 buf.extend_from_slice(&HEADER_MAGIC); // 0..8
 buf.push(self.format_major); // 8
 buf.push(self.format_minor); // 9
 buf.extend_from_slice(&(HEADER_LEN as u16).to_le_bytes()); // 10..12
 buf.extend_from_slice(&self.flags.to_le_bytes()); // 12..16
 buf.extend_from_slice(&self.edge_type_id); // 16..32
 buf.extend_from_slice(&self.src_label_id); // 32..48
 buf.extend_from_slice(&self.dst_label_id); // 48..64
 debug_assert_eq!(buf.len() - start, HEADER_LEN);
 }

 pub fn decode(buf: &[u8]) -> Result<Self> {
 if buf.len() < HEADER_LEN {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("edge SST header truncated ({} bytes)", buf.len()),
 });
 }
 if buf[..8] != HEADER_MAGIC {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: "edge SST header magic mismatch".into(),
 });
 }
 let format_major = buf[8];
 let format_minor = buf[9];
 if format_major != FORMAT_MAJOR {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("unsupported edge SST format_major={format_major}"),
 });
 }
 let header_size = u16::from_le_bytes([buf[10], buf[11]]);
 if header_size as usize != HEADER_LEN {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("edge SST header_size={header_size} != {HEADER_LEN}"),
 });
 }
 let flags = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
 if flags & FLAGS_RESERVED_MASK != 0 {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("edge SST sets unknown reserved flag bits in 0x{flags:08x}"),
 });
 }
 let edge_type_id: [u8; 16] = buf[16..32].try_into().unwrap();
 let src_label_id: [u8; 16] = buf[32..48].try_into().unwrap();
 let dst_label_id: [u8; 16] = buf[48..64].try_into().unwrap();
 Ok(Self {
 format_major,
 format_minor,
 flags,
 edge_type_id,
 src_label_id,
 dst_label_id,
 })
 }
}

fn blake3_short(s: &str) -> [u8; 16] {
 let hash = blake3::hash(s.as_bytes());
 let bytes = hash.as_bytes();
 let mut out = [0u8; 16];
 out.copy_from_slice(&bytes[..16]);
 out
}

// ─── section table ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionEntry {
 pub kind: u16,
 pub offset: u64,
 pub length: u64,
 pub codec: u8,
 pub xxhash3_64: u64,
 pub name: String,
}

impl SectionEntry {
 /// Encode a single entry into `buf`.
 pub fn encode(&self, buf: &mut Vec<u8>) -> Result<()> {
 if self.name.len() > u8::MAX as usize {
 return Err(Error::invariant(format!(
 "section name '{}' exceeds 255 bytes",
 self.name
 )));
 }
 buf.extend_from_slice(&self.kind.to_le_bytes());
 buf.extend_from_slice(&self.offset.to_le_bytes());
 buf.extend_from_slice(&self.length.to_le_bytes());
 buf.push(self.codec);
 buf.push(0); // reserved
 buf.extend_from_slice(&self.xxhash3_64.to_le_bytes());
 buf.push(self.name.len() as u8);
 buf.extend_from_slice(self.name.as_bytes());
 Ok(())
 }

 pub fn decode(buf: &[u8], cursor: usize) -> Result<(Self, usize)> {
 // Fixed prefix is 28 bytes.
 let need = 28;
 if cursor + need > buf.len() {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: "section entry header truncated".into(),
 });
 }
 let kind = u16::from_le_bytes([buf[cursor], buf[cursor + 1]]);
 let offset = u64::from_le_bytes(buf[cursor + 2..cursor + 10].try_into().unwrap());
 let length = u64::from_le_bytes(buf[cursor + 10..cursor + 18].try_into().unwrap());
 let codec = buf[cursor + 18];
 // reserved at cursor+19
 let xxhash3_64 = u64::from_le_bytes(buf[cursor + 20..cursor + 28].try_into().unwrap());
 let name_len = buf[cursor + 28] as usize;
 let name_start = cursor + 29;
 let name_end = name_start + name_len;
 if name_end > buf.len() {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: "section entry name truncated".into(),
 });
 }
 let name = std::str::from_utf8(&buf[name_start..name_end])
 .map_err(|e| Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("section entry name not utf-8: {e}"),
 })?
 .to_string();
 Ok((
 Self {
 kind,
 offset,
 length,
 codec,
 xxhash3_64,
 name,
 },
 (name_end) - cursor,
 ))
 }
}

// ─── footer body + trailer ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeFileFooter {
 pub sections: Vec<SectionEntry>,
 pub key_count: u64,
 pub edge_count: u64,
 pub offsets_bits: u8,
 pub min_key_id: [u8; 16],
 pub max_key_id: [u8; 16],
 pub min_lsn: u64,
 pub max_lsn: u64,
 pub schema_version_min: u64,
 pub schema_version_max: u64,
}

/// Trailer is fixed at 20 bytes: xxhash(8) + footer_len(4) + magic(8).
pub const FOOTER_TRAILER_LEN: usize = 20;

impl EdgeFileFooter {
 /// Encode the footer (body + trailer). Returns the total bytes appended
 /// to `buf`, which is also written into the trailer's `footer_len` field.
 pub fn encode(&self, buf: &mut Vec<u8>) -> Result<usize> {
 let start = buf.len();
 // Body
 for entry in &self.sections {
 entry.encode(buf)?;
 }
 buf.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
 buf.extend_from_slice(&self.key_count.to_le_bytes());
 buf.extend_from_slice(&self.edge_count.to_le_bytes());
 buf.push(self.offsets_bits);
 buf.extend_from_slice(&self.min_key_id);
 buf.extend_from_slice(&self.max_key_id);
 buf.extend_from_slice(&self.min_lsn.to_le_bytes());
 buf.extend_from_slice(&self.max_lsn.to_le_bytes());
 buf.extend_from_slice(&self.schema_version_min.to_le_bytes());
 buf.extend_from_slice(&self.schema_version_max.to_le_bytes());
 let body_end = buf.len();

 // Trailer
 let body_hash = xxh3_64(&buf[start..body_end]);
 buf.extend_from_slice(&body_hash.to_le_bytes());
 let footer_len_pos = buf.len();
 buf.extend_from_slice(&0u32.to_le_bytes()); // placeholder
 buf.extend_from_slice(&FOOTER_MAGIC);
 let total = buf.len() - start;
 // Backpatch footer_len.
 let fl = (total as u32).to_le_bytes();
 buf[footer_len_pos..footer_len_pos + 4].copy_from_slice(&fl);
 Ok(total)
 }

 /// Decode the footer from a fully-buffered file body. Returns the
 /// footer plus the byte offset at which the footer body starts.
 pub fn decode(body: &Bytes) -> Result<(Self, usize)> {
 let len = body.len();
 if len < FOOTER_TRAILER_LEN {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("edge SST too small ({len} bytes)"),
 });
 }
 let trailer_start = len - FOOTER_TRAILER_LEN;
 if body[len - 8..] != FOOTER_MAGIC {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: "edge SST footer magic mismatch".into(),
 });
 }
 let footer_len = u32::from_le_bytes(
 body[trailer_start + 8..trailer_start + 12]
 .try_into()
 .unwrap(),
 ) as usize;
 if footer_len > len {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("footer_len {footer_len} exceeds file size {len}"),
 });
 }
 let footer_start = len - footer_len;
 let body_end = trailer_start;
 let body_hash_stored =
 u64::from_le_bytes(body[trailer_start..trailer_start + 8].try_into().unwrap());
 let body_hash_computed = xxh3_64(&body[footer_start..body_end]);
 if body_hash_stored != body_hash_computed {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: "edge SST footer xxhash mismatch".into(),
 });
 }

 // Parse body. The section table comes first, followed by the
 // fixed-shape tail. We don't know the section count up front, so
 // we walk from `footer_start` and bound by `body_end - tail_size`.
 let tail_size: usize = 4 // section_count
 + 8 // key_count
 + 8 // edge_count
 + 1 // offsets_bits
 + 16 // min_key_id
 + 16 // max_key_id
 + 8 // min_lsn
 + 8 // max_lsn
 + 8 // schema_version_min
 + 8; // schema_version_max
 if footer_start + tail_size > body_end {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: "footer body too small to contain the fixed tail".into(),
 });
 }
 let tail_start = body_end - tail_size;
 let section_count =
 u32::from_le_bytes(body[tail_start..tail_start + 4].try_into().unwrap());
 let mut cursor = tail_start + 4;
 let key_count = u64::from_le_bytes(body[cursor..cursor + 8].try_into().unwrap());
 cursor += 8;
 let edge_count = u64::from_le_bytes(body[cursor..cursor + 8].try_into().unwrap());
 cursor += 8;
 let offsets_bits = body[cursor];
 cursor += 1;
 let min_key_id: [u8; 16] = body[cursor..cursor + 16].try_into().unwrap();
 cursor += 16;
 let max_key_id: [u8; 16] = body[cursor..cursor + 16].try_into().unwrap();
 cursor += 16;
 let min_lsn = u64::from_le_bytes(body[cursor..cursor + 8].try_into().unwrap());
 cursor += 8;
 let max_lsn = u64::from_le_bytes(body[cursor..cursor + 8].try_into().unwrap());
 cursor += 8;
 let schema_version_min = u64::from_le_bytes(body[cursor..cursor + 8].try_into().unwrap());
 cursor += 8;
 let schema_version_max = u64::from_le_bytes(body[cursor..cursor + 8].try_into().unwrap());

 // Section table walk: from footer_start up to tail_start.
 let table_end = tail_start;
 let mut sections = Vec::with_capacity(section_count as usize);
 let mut table_cursor = footer_start;
 for _ in 0..section_count {
 if table_cursor >= table_end {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: "section table ended before declared count".into(),
 });
 }
 let (entry, consumed) = SectionEntry::decode(body, table_cursor)?;
 table_cursor += consumed;
 sections.push(entry);
 }
 if table_cursor != table_end {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("section table ended at {table_cursor}, expected {table_end}"),
 });
 }

 // Validate section ranges. RFC-002 §3.2.8 requires the section
 // table on disk to be sorted ascending by `offset`; rejecting an
 // out-of-order table here is the only way to keep readers and
 // writers from drifting on that invariant. Once sortedness is
 // guaranteed, overlap detection is a single linear pass.
 let mut prev_end: u64 = HEADER_LEN as u64;
 for (i, entry) in sections.iter().enumerate() {
 if i > 0 && entry.offset < sections[i - 1].offset {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "section table not sorted by offset: entry {} at {} precedes entry {} at {}",
 i,
 entry.offset,
 i - 1,
 sections[i - 1].offset
 ),
 });
 }
 let end = entry
 .offset
 .checked_add(entry.length)
 .ok_or_else(|| Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("section '{}' offset+length overflow", entry.name),
 })?;
 if entry.offset < HEADER_LEN as u64 || end > footer_start as u64 {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "section '{}' [{}, {}) outside of file body [{}, {})",
 entry.name, entry.offset, end, HEADER_LEN, footer_start
 ),
 });
 }
 if entry.offset < prev_end {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!(
 "section '{}' overlaps a prior section: starts at {} but body before it ends at {}",
 entry.name, entry.offset, prev_end
 ),
 });
 }
 prev_end = end;
 }

 Ok((
 Self {
 sections,
 key_count,
 edge_count,
 offsets_bits,
 min_key_id,
 max_key_id,
 min_lsn,
 max_lsn,
 schema_version_min,
 schema_version_max,
 },
 footer_start,
 ))
 }

 /// Look up the first section whose `(kind, name)` matches.
 pub fn find(&self, kind: u16, name: &str) -> Option<&SectionEntry> {
 self.sections
 .iter()
 .find(|e| e.kind == kind && e.name == name)
 }

 /// Look up the first section by `kind` ignoring the name.
 pub fn find_kind(&self, kind: u16) -> Option<&SectionEntry> {
 self.sections.iter().find(|e| e.kind == kind)
 }
}

#[cfg(test)]
mod tests {
 use bytes::Bytes;

 use super::*;

 #[test]
 fn header_round_trip() {
 let h = EdgeFileHeader::new(
 "KNOWS",
 "Person",
 "Person",
 FLAG_HAS_PROPERTIES | FLAG_INVERSE_PARTNER,
 );
 let mut buf = Vec::new();
 h.encode(&mut buf);
 assert_eq!(buf.len(), HEADER_LEN);
 let back = EdgeFileHeader::decode(&buf).unwrap();
 assert_eq!(back, h);
 }

 #[test]
 fn header_rejects_bad_magic() {
 let h = EdgeFileHeader::new("X", "Y", "Z", 0);
 let mut buf = Vec::new();
 h.encode(&mut buf);
 buf[0] = b'!';
 let err = EdgeFileHeader::decode(&buf).unwrap_err();
 assert!(matches!(err, Error::Corrupted { .. }));
 }

 #[test]
 fn header_rejects_unknown_reserved_flag() {
 let h = EdgeFileHeader::new("X", "Y", "Z", 1 << 10); // reserved bit 10
 let mut buf = Vec::new();
 h.encode(&mut buf);
 let err = EdgeFileHeader::decode(&buf).unwrap_err();
 assert!(matches!(err, Error::Corrupted { .. }));
 }

 #[test]
 fn header_rejects_future_major() {
 let h = EdgeFileHeader::new("X", "Y", "Z", 0);
 let mut buf = Vec::new();
 h.encode(&mut buf);
 buf[8] = 2;
 let err = EdgeFileHeader::decode(&buf).unwrap_err();
 assert!(matches!(err, Error::Corrupted { .. }));
 }

 #[test]
 fn footer_round_trip_empty() {
 // Build a minimal "file": header + footer (no sections).
 let mut file = Vec::new();
 EdgeFileHeader::new("KNOWS", "P", "P", 0).encode(&mut file);
 let footer = EdgeFileFooter {
 sections: Vec::new(),
 key_count: 0,
 edge_count: 0,
 offsets_bits: 32,
 min_key_id: [0; 16],
 max_key_id: [0; 16],
 min_lsn: 0,
 max_lsn: 0,
 schema_version_min: 0,
 schema_version_max: 0,
 };
 footer.encode(&mut file).unwrap();
 let body = Bytes::from(file);
 let (back, footer_start) = EdgeFileFooter::decode(&body).unwrap();
 assert_eq!(back, footer);
 assert_eq!(footer_start, HEADER_LEN);
 }

 #[test]
 fn footer_round_trip_with_sections() {
 let mut file = Vec::new();
 EdgeFileHeader::new("KNOWS", "P", "P", 0).encode(&mut file);
 // Reserve some byte ranges for two fake sections.
 let s1_start = file.len() as u64;
 file.extend_from_slice(b"section-one-bytes");
 let s1_end = file.len() as u64;
 let s2_start = s1_end;
 file.extend_from_slice(b"another-section");
 let s2_end = file.len() as u64;

 let footer = EdgeFileFooter {
 sections: vec![
 SectionEntry {
 kind: SECTION_KEY_IDS,
 offset: s1_start,
 length: s1_end - s1_start,
 codec: CODEC_NONE,
 xxhash3_64: 0,
 name: String::new(),
 },
 SectionEntry {
 kind: SECTION_PROPERTY_STREAM,
 offset: s2_start,
 length: s2_end - s2_start,
 codec: CODEC_ZSTD,
 xxhash3_64: 0,
 name: "since".into(),
 },
 ],
 key_count: 10,
 edge_count: 25,
 offsets_bits: 24,
 min_key_id: [1; 16],
 max_key_id: [2; 16],
 min_lsn: 100,
 max_lsn: 200,
 schema_version_min: 3,
 schema_version_max: 3,
 };
 footer.encode(&mut file).unwrap();
 let body = Bytes::from(file);
 let (back, footer_start) = EdgeFileFooter::decode(&body).unwrap();
 assert_eq!(back, footer);
 assert!(footer_start > HEADER_LEN);
 }

 #[test]
 fn footer_rejects_bad_xxhash() {
 let mut file = Vec::new();
 EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
 let footer = EdgeFileFooter {
 sections: Vec::new(),
 key_count: 0,
 edge_count: 0,
 offsets_bits: 32,
 min_key_id: [0; 16],
 max_key_id: [0; 16],
 min_lsn: 0,
 max_lsn: 0,
 schema_version_min: 0,
 schema_version_max: 0,
 };
 footer.encode(&mut file).unwrap();
 let last = file.len() - FOOTER_TRAILER_LEN; // xxhash position
 file[last] ^= 0xff;
 let body = Bytes::from(file);
 let err = EdgeFileFooter::decode(&body).unwrap_err();
 assert!(matches!(err, Error::Corrupted { .. }));
 }

 #[test]
 fn footer_rejects_overlapping_sections() {
 let mut file = Vec::new();
 EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
 // Pad some body bytes.
 file.extend_from_slice(&[0u8; 128]);
 let body_end = file.len() as u64;
 let footer = EdgeFileFooter {
 sections: vec![
 SectionEntry {
 kind: SECTION_KEY_IDS,
 offset: HEADER_LEN as u64,
 length: 100,
 codec: CODEC_NONE,
 xxhash3_64: 0,
 name: String::new(),
 },
 SectionEntry {
 // overlaps the first section by 50 bytes
 kind: SECTION_OFFSETS,
 offset: HEADER_LEN as u64 + 50,
 length: 20,
 codec: CODEC_NONE,
 xxhash3_64: 0,
 name: String::new(),
 },
 ],
 key_count: 0,
 edge_count: 0,
 offsets_bits: 32,
 min_key_id: [0; 16],
 max_key_id: [0; 16],
 min_lsn: 0,
 max_lsn: 0,
 schema_version_min: 0,
 schema_version_max: 0,
 };
 // Sanity: body_end accommodates both sections individually.
 assert!(body_end >= HEADER_LEN as u64 + 100);
 footer.encode(&mut file).unwrap();
 let body = Bytes::from(file);
 let err = EdgeFileFooter::decode(&body).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("overlap")),
 other => panic!("expected Corrupted with overlap detail, got {other:?}"),
 }
 }

 #[test]
 fn footer_rejects_section_outside_file() {
 let mut file = Vec::new();
 EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
 let footer = EdgeFileFooter {
 sections: vec![SectionEntry {
 kind: SECTION_KEY_IDS,
 offset: 1_000_000, // way past EOF
 length: 16,
 codec: CODEC_NONE,
 xxhash3_64: 0,
 name: String::new(),
 }],
 key_count: 0,
 edge_count: 0,
 offsets_bits: 32,
 min_key_id: [0; 16],
 max_key_id: [0; 16],
 min_lsn: 0,
 max_lsn: 0,
 schema_version_min: 0,
 schema_version_max: 0,
 };
 footer.encode(&mut file).unwrap();
 let body = Bytes::from(file);
 let err = EdgeFileFooter::decode(&body).unwrap_err();
 assert!(matches!(err, Error::Corrupted { .. }));
 }

 #[test]
 fn footer_rejects_unsorted_section_table() {
 // RFC-002 §3.2.8 requires the on-disk section table to be sorted
 // ascending by `offset`. A writer that emits it out of order must
 // be rejected even if the byte ranges are otherwise valid — the
 // reader's ranged-GET prefetch logic relies on the invariant.
 let mut file = Vec::new();
 EdgeFileHeader::new("X", "Y", "Z", 0).encode(&mut file);
 // Reserve two non-overlapping body ranges in canonical order.
 let s1_start = file.len() as u64;
 file.extend_from_slice(&[0u8; 32]);
 let s1_end = file.len() as u64;
 let s2_start = s1_end;
 file.extend_from_slice(&[0u8; 32]);
 let s2_end = file.len() as u64;

 // Build a footer that lists the *later* section first.
 let footer = EdgeFileFooter {
 sections: vec![
 SectionEntry {
 kind: SECTION_OFFSETS,
 offset: s2_start,
 length: s2_end - s2_start,
 codec: CODEC_NONE,
 xxhash3_64: 0,
 name: String::new(),
 },
 SectionEntry {
 kind: SECTION_KEY_IDS,
 offset: s1_start,
 length: s1_end - s1_start,
 codec: CODEC_NONE,
 xxhash3_64: 0,
 name: String::new(),
 },
 ],
 key_count: 0,
 edge_count: 0,
 offsets_bits: 32,
 min_key_id: [0; 16],
 max_key_id: [0; 16],
 min_lsn: 0,
 max_lsn: 0,
 schema_version_min: 0,
 schema_version_max: 0,
 };
 footer.encode(&mut file).unwrap();
 let body = Bytes::from(file);
 let err = EdgeFileFooter::decode(&body).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("not sorted")),
 other => panic!("expected Corrupted with not-sorted detail, got {other:?}"),
 }
 }
}
