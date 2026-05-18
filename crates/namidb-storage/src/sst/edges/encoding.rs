//! Low-level encoders / decoders for the edge SST wire format.
//!
//! Defined by [RFC-002](../../../../../docs/rfc/002-sst-format.md) §3.2.3
//! (bitpacked offsets) and §3.2.4 (partner blocks).
//!
//! All multi-byte integers are little-endian.

use crate::error::{Error, Result};

// ─── varint (Protobuf LEB128) ───────────────────────────────────────────

/// Write a `u64` as a base-128 varint. Returns the number of bytes written.
pub fn write_varint(mut value: u64, buf: &mut Vec<u8>) -> usize {
 let mut written = 0;
 while value >= 0x80 {
 buf.push(((value as u8) & 0x7f) | 0x80);
 value >>= 7;
 written += 1;
 }
 buf.push(value as u8);
 written + 1
}

/// Read a `u64` varint from `buf` starting at `cursor`. Returns
/// `(value, bytes_consumed)`.
pub fn read_varint(buf: &[u8], cursor: usize) -> Result<(u64, usize)> {
 let mut value: u64 = 0;
 let mut shift: u32 = 0;
 let mut i = 0;
 loop {
 if cursor + i >= buf.len() {
 return Err(Error::invariant(
 "varint truncated: read past end of buffer",
 ));
 }
 let byte = buf[cursor + i];
 let chunk = (byte & 0x7f) as u64;
 // On byte 10 of a u64 varint we are at `shift = 63` and only the
 // single bit at position 63 of the encoded value can come from this
 // byte. A chunk with any bit other than the lowest set would mean
 // the encoded value is `>= 2^64` and cannot fit in `u64` — we'd
 // silently overflow otherwise (`chunk << 63` wraps to zero in
 // release mode).
 if shift == 63 && chunk > 1 {
 return Err(Error::invariant(
 "varint overflows u64: top byte sets bits beyond bit 63",
 ));
 }
 value |= chunk << shift;
 i += 1;
 if byte & 0x80 == 0 {
 return Ok((value, i));
 }
 shift += 7;
 if shift >= 64 {
 return Err(Error::invariant("varint overflows u64"));
 }
 }
}

/// Number of bytes a value would occupy when encoded as varint.
pub fn varint_len(value: u64) -> usize {
 let mut n = 1;
 let mut v = value >> 7;
 while v != 0 {
 v >>= 7;
 n += 1;
 }
 n
}

// ─── bitpacked offsets (§3.2.3) ─────────────────────────────────────────

/// Width in bits chosen by the writer for the offsets section, per the
/// table in §3.2.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetWidth {
 /// 3 bytes per offset (`< 2^24`).
 W24,
 /// 4 bytes per offset (`< 2^32`).
 W32,
 /// 5 bytes per offset (`< 2^40`).
 W40,
 /// 6 bytes per offset (`< 2^48`).
 W48,
}

impl OffsetWidth {
 pub fn bytes(self) -> usize {
 match self {
 OffsetWidth::W24 => 3,
 OffsetWidth::W32 => 4,
 OffsetWidth::W40 => 5,
 OffsetWidth::W48 => 6,
 }
 }

 pub fn as_bits(self) -> u8 {
 match self {
 OffsetWidth::W24 => 24,
 OffsetWidth::W32 => 32,
 OffsetWidth::W40 => 40,
 OffsetWidth::W48 => 48,
 }
 }

 pub fn from_bits(bits: u8) -> Result<Self> {
 Ok(match bits {
 24 => OffsetWidth::W24,
 32 => OffsetWidth::W32,
 40 => OffsetWidth::W40,
 48 => OffsetWidth::W48,
 other => {
 return Err(Error::invariant(format!(
 "unsupported offsets_bits={other}; expected one of 24/32/40/48"
 )))
 }
 })
 }

 /// Pick the smallest width that can represent every offset up to
 /// `max_value` inclusive.
 pub fn for_max(max_value: u64) -> Self {
 if max_value < (1u64 << 24) {
 OffsetWidth::W24
 } else if max_value < (1u64 << 32) {
 OffsetWidth::W32
 } else if max_value < (1u64 << 40) {
 OffsetWidth::W40
 } else if max_value < (1u64 << 48) {
 OffsetWidth::W48
 } else {
 // SST sections > 256 TiB exceed v1; defer to v2 with W64.
 panic!(
 "offset value {} exceeds 2^48; format v1 only supports W48",
 max_value
 );
 }
 }
}

/// Append a single offset to `buf` using the supplied width.
pub fn write_offset(value: u64, width: OffsetWidth, buf: &mut Vec<u8>) {
 let bytes = value.to_le_bytes();
 buf.extend_from_slice(&bytes[..width.bytes()]);
}

/// Read a single offset from `buf` starting at `cursor`.
pub fn read_offset(buf: &[u8], cursor: usize, width: OffsetWidth) -> Result<u64> {
 let need = width.bytes();
 if cursor + need > buf.len() {
 return Err(Error::invariant(format!(
 "offset read out of bounds: cursor={cursor} need={need} len={}",
 buf.len()
 )));
 }
 let mut raw = [0u8; 8];
 raw[..need].copy_from_slice(&buf[cursor..cursor + need]);
 Ok(u64::from_le_bytes(raw))
}

// ─── partner blocks (§3.2.4) ────────────────────────────────────────────

pub const TAG_SPLIT: u8 = 0x01;
pub const TAG_DENSE: u8 = 0x10;

/// Compute the byte cost of the split-encoded block for the given partner
/// list, **assuming the list is sorted ascending by the full 128-bit id**.
pub fn split_block_cost(partners: &[[u8; 16]]) -> usize {
 if partners.is_empty() {
 return 0;
 }
 let top0 = u64::from_le_bytes(partners[0][..8].try_into().unwrap());
 let mut total = varint_len(top0) + 8; // top64[0] + bot64[0]
 let mut prev_top = top0;
 for p in &partners[1..] {
 let top = u64::from_le_bytes(p[..8].try_into().unwrap());
 let delta = top.wrapping_sub(prev_top);
 total += varint_len(delta) + 8;
 prev_top = top;
 }
 total
}

/// Byte cost of the dense block (always 16 × deg).
pub fn dense_block_cost(partners: &[[u8; 16]]) -> usize {
 partners.len() * 16
}

/// Selection rule (§3.2.4): emit dense when split would not save bytes,
/// or when `deg > skew_threshold`.
pub fn pick_block_tag(partners: &[[u8; 16]], skew_threshold: usize) -> u8 {
 let deg = partners.len();
 if deg > skew_threshold {
 return TAG_DENSE;
 }
 if split_block_cost(partners) >= dense_block_cost(partners) {
 TAG_DENSE
 } else {
 TAG_SPLIT
 }
}

/// Encode `partners` as a single partner block, choosing tag per the
/// selection rule. Returns `tag`.
///
/// The block layout is:
/// ```text
/// deg: varint | tag: u8 | payload
/// ```
pub fn write_partner_block(partners: &[[u8; 16]], skew_threshold: usize, buf: &mut Vec<u8>) -> u8 {
 let tag = pick_block_tag(partners, skew_threshold);
 write_varint(partners.len() as u64, buf);
 buf.push(tag);
 match tag {
 TAG_SPLIT => write_split_payload(partners, buf),
 TAG_DENSE => write_dense_payload(partners, buf),
 _ => unreachable!(),
 }
 tag
}

fn write_split_payload(partners: &[[u8; 16]], buf: &mut Vec<u8>) {
 if partners.is_empty() {
 return;
 }
 let top0 = u64::from_le_bytes(partners[0][..8].try_into().unwrap());
 let bot0 = u64::from_le_bytes(partners[0][8..].try_into().unwrap());
 write_varint(top0, buf);
 buf.extend_from_slice(&bot0.to_le_bytes());
 let mut prev_top = top0;
 for p in &partners[1..] {
 let top = u64::from_le_bytes(p[..8].try_into().unwrap());
 let bot = u64::from_le_bytes(p[8..].try_into().unwrap());
 let delta = top.wrapping_sub(prev_top);
 write_varint(delta, buf);
 buf.extend_from_slice(&bot.to_le_bytes());
 prev_top = top;
 }
}

fn write_dense_payload(partners: &[[u8; 16]], buf: &mut Vec<u8>) {
 for p in partners {
 buf.extend_from_slice(p);
 }
}

/// Decode a single partner block from `buf` starting at `cursor`. Returns
/// the decoded partners and the number of bytes consumed.
pub fn read_partner_block(buf: &[u8], cursor: usize) -> Result<(Vec<[u8; 16]>, usize)> {
 let start = cursor;
 let (deg, n) = read_varint(buf, cursor)?;
 let mut c = cursor + n;
 if c >= buf.len() {
 return Err(Error::invariant("partner block truncated: missing tag"));
 }
 let tag = buf[c];
 c += 1;
 let deg = deg as usize;
 let partners = match tag {
 TAG_SPLIT => {
 let (out, consumed) = read_split_payload(buf, c, deg)?;
 c += consumed;
 out
 }
 TAG_DENSE => {
 let (out, consumed) = read_dense_payload(buf, c, deg)?;
 c += consumed;
 out
 }
 other => {
 return Err(Error::Corrupted {
 path: "<edges>".into(),
 detail: format!("unknown partner block tag 0x{other:02x}"),
 });
 }
 };
 Ok((partners, c - start))
}

fn read_split_payload(buf: &[u8], cursor: usize, deg: usize) -> Result<(Vec<[u8; 16]>, usize)> {
 if deg == 0 {
 return Ok((Vec::new(), 0));
 }
 let mut out = Vec::with_capacity(deg);
 let mut c = cursor;

 let (top, n) = read_varint(buf, c)?;
 c += n;
 if c + 8 > buf.len() {
 return Err(Error::invariant("split block truncated: bot64[0]"));
 }
 let bot = u64::from_le_bytes(buf[c..c + 8].try_into().unwrap());
 c += 8;
 out.push(combine_top_bot(top, bot));
 let mut prev_top = top;

 for _ in 1..deg {
 let (delta, n) = read_varint(buf, c)?;
 c += n;
 if c + 8 > buf.len() {
 return Err(Error::invariant("split block truncated: bot64[j]"));
 }
 let bot = u64::from_le_bytes(buf[c..c + 8].try_into().unwrap());
 c += 8;
 let top = prev_top.wrapping_add(delta);
 out.push(combine_top_bot(top, bot));
 prev_top = top;
 }
 Ok((out, c - cursor))
}

fn read_dense_payload(buf: &[u8], cursor: usize, deg: usize) -> Result<(Vec<[u8; 16]>, usize)> {
 let need = deg * 16;
 if cursor + need > buf.len() {
 return Err(Error::invariant(format!(
 "dense block truncated: need {need} bytes, have {}",
 buf.len() - cursor
 )));
 }
 let mut out = Vec::with_capacity(deg);
 for i in 0..deg {
 let off = cursor + i * 16;
 let arr: [u8; 16] = buf[off..off + 16].try_into().unwrap();
 out.push(arr);
 }
 Ok((out, need))
}

fn combine_top_bot(top: u64, bot: u64) -> [u8; 16] {
 let mut out = [0u8; 16];
 out[..8].copy_from_slice(&top.to_le_bytes());
 out[8..].copy_from_slice(&bot.to_le_bytes());
 out
}

#[cfg(test)]
mod tests {
 use super::*;

 // ── varint ─────────────────────────────────────────────────────────

 #[test]
 fn varint_round_trips() {
 for v in [0u64, 1, 127, 128, 0x3fff, 0x4000, u64::MAX] {
 let mut buf = Vec::new();
 let n_w = write_varint(v, &mut buf);
 assert_eq!(n_w, buf.len());
 assert_eq!(n_w, varint_len(v));
 let (back, n_r) = read_varint(&buf, 0).unwrap();
 assert_eq!(back, v);
 assert_eq!(n_r, buf.len());
 }
 }

 #[test]
 fn varint_overflow_rejected() {
 // 10 0x80 bytes form a 70-bit value → too wide for u64.
 let buf = [0x80u8; 11];
 let err = read_varint(&buf, 0).unwrap_err();
 assert!(matches!(err, Error::Invariant(_)));
 }

 #[test]
 fn varint_truncated_rejected() {
 let buf = [0x80u8, 0x80, 0x80];
 let err = read_varint(&buf, 0).unwrap_err();
 assert!(matches!(err, Error::Invariant(_)));
 }

 #[test]
 fn varint_rejects_top_byte_overflow_beyond_bit_63() {
 // 10 bytes: 9 continuation bytes encoding the low 63 bits as zero,
 // then a top byte of 0x02 which would set bit 64 of the result.
 // `2 << 63` overflows u64 — must be rejected.
 let mut buf = vec![0x80u8; 9];
 buf.push(0x02);
 let err = read_varint(&buf, 0).unwrap_err();
 assert!(matches!(err, Error::Invariant(_)));
 // Mirror with 0x7f (every bit set in chunk) — same rejection.
 let mut buf = vec![0x80u8; 9];
 buf.push(0x7f);
 let err = read_varint(&buf, 0).unwrap_err();
 assert!(matches!(err, Error::Invariant(_)));
 }

 #[test]
 fn varint_accepts_legal_u64_max() {
 // u64::MAX encoded as a 10-byte varint with top byte 0x01.
 let mut buf = vec![0xffu8; 9];
 buf.push(0x01);
 let (v, n) = read_varint(&buf, 0).unwrap();
 assert_eq!(v, u64::MAX);
 assert_eq!(n, 10);
 }

 // ── offsets ────────────────────────────────────────────────────────

 #[test]
 fn offset_width_selection() {
 assert_eq!(OffsetWidth::for_max(0), OffsetWidth::W24);
 assert_eq!(OffsetWidth::for_max((1 << 24) - 1), OffsetWidth::W24);
 assert_eq!(OffsetWidth::for_max(1 << 24), OffsetWidth::W32);
 assert_eq!(OffsetWidth::for_max((1 << 32) - 1), OffsetWidth::W32);
 assert_eq!(OffsetWidth::for_max(1 << 32), OffsetWidth::W40);
 assert_eq!(OffsetWidth::for_max(1 << 40), OffsetWidth::W48);
 }

 #[test]
 fn offset_round_trip_all_widths() {
 for &w in &[
 OffsetWidth::W24,
 OffsetWidth::W32,
 OffsetWidth::W40,
 OffsetWidth::W48,
 ] {
 for v in [0u64, 1, 255, 256, (1u64 << 16), (1u64 << w.as_bits()) - 1] {
 let mut buf = Vec::new();
 write_offset(v, w, &mut buf);
 assert_eq!(buf.len(), w.bytes());
 let back = read_offset(&buf, 0, w).unwrap();
 assert_eq!(back, v, "{w:?} v={v}");
 }
 }
 }

 // ── partner blocks ─────────────────────────────────────────────────

 fn partner(top: u64, bot: u64) -> [u8; 16] {
 let mut out = [0u8; 16];
 out[..8].copy_from_slice(&top.to_le_bytes());
 out[8..].copy_from_slice(&bot.to_le_bytes());
 out
 }

 #[test]
 fn split_payload_round_trip_clustered_top64() {
 // All partners share the same top64 → maximum compression.
 let partners: Vec<_> = (0..10).map(|i| partner(1_700_000_000_000, i)).collect();
 let mut buf = Vec::new();
 let tag = write_partner_block(&partners, /*skew*/ 10_000, &mut buf);
 assert_eq!(tag, TAG_SPLIT);
 let (back, consumed) = read_partner_block(&buf, 0).unwrap();
 assert_eq!(back, partners);
 assert_eq!(consumed, buf.len());
 // 8 bytes (top0 varint at 1.7e12 ≈ 6 bytes) + 8 bot + 9 * (1 byte delta + 8 bot)
 // ≈ 14 + 9 * 9 = 95 bytes vs raw 160 bytes.
 assert!(buf.len() < 160);
 }

 #[test]
 fn dense_payload_round_trip() {
 // Use the dense path via skew_threshold override.
 let partners: Vec<_> = (0..3).map(|i| partner(i * 1000, i * 31)).collect();
 let mut buf = Vec::new();
 let tag = write_partner_block(&partners, /*skew*/ 1, &mut buf);
 assert_eq!(tag, TAG_DENSE);
 let (back, consumed) = read_partner_block(&buf, 0).unwrap();
 assert_eq!(back, partners);
 assert_eq!(consumed, buf.len());
 // 1 byte deg + 1 byte tag + 3 * 16 bytes = 50 bytes.
 assert_eq!(buf.len(), 50);
 }

 #[test]
 fn selection_rule_falls_back_to_dense_when_split_loses() {
 // 10 partners with `top64_delta = 2^60` each. Each delta needs 9 varint
 // bytes (61 bits → ceil(61/7) = 9). Per-partner cost in split:
 // first: 1 (top0=0 varint) + 8 (bot0) = 9 bytes
 // rest: 9 (delta) + 8 (bot) = 17 bytes × 9 = 153 bytes
 // total: 162 bytes
 // Dense: 10 × 16 = 160 bytes. Split loses by 2 → fallback wins.
 let partners: Vec<_> = (0..10u64).map(|i| partner(i * (1u64 << 60), i)).collect();
 assert!(
 split_block_cost(&partners) >= dense_block_cost(&partners),
 "test precondition: expected split to lose vs dense"
 );
 let tag = pick_block_tag(&partners, 10_000);
 assert_eq!(tag, TAG_DENSE);
 }

 #[test]
 fn selection_rule_prefers_split_on_clustered() {
 let partners: Vec<_> = (0..100u64).map(|i| partner(123, i)).collect();
 let tag = pick_block_tag(&partners, 10_000);
 assert_eq!(tag, TAG_SPLIT);
 }

 #[test]
 fn unknown_tag_is_corrupted() {
 let mut buf = Vec::new();
 write_varint(1, &mut buf);
 buf.push(0xff);
 buf.extend_from_slice(&[0u8; 16]);
 let err = read_partner_block(&buf, 0).unwrap_err();
 assert!(matches!(err, Error::Corrupted { .. }));
 }
}
