//! Split-Block Bloom Filter (SBBF) + side-car wire format.
//!
//! Defined by [RFC-002](../../../../docs/rfc/002-sst-format.md) §4.2.
//!
//! ## Format
//!
//! ```text
//! offset size field
//! ─────── ──── ─────────────────────────────────────────
//! 0 8 magic b"TGBLOOM\0"
//! 8 1 format_major u8 = 1
//! 9 1 format_minor u8 = 0
//! 10 2 reserved u16 = 0
//! 12 1 bits_per_key u8 (default 10)
//! 13 1 reserved2 u8 = 0 (was k_hashes pre-rev3)
//! 14 2 reserved3 u16 = 0
//! 16 4 block_count u32 LE
//! 20 8 key_count u64 LE
//! 28 … blocks [SbbfBlock; block_count] (each = [u8; 32])
//! … 8 trailer xxhash3_64 LE over [0 .. file_size - 8)
//! ```
//!
//! Total bytes: `28 + 32 * block_count + 8`.
//!
//! ## SBBF algorithm
//!
//! - Hash a key with **xxHash3-64** (seed 0).
//! - `block_index = ((h >> 32) * block_count) >> 32` → pick a block.
//! - Within that 256-bit (8 × u32) block, set one bit per `i ∈ 0..8`,
//! chosen by `bit_index = (SALT[i] * (h as u32)) >> 27`. The salts are
//! the standard Parquet SBBF constants.
//!
//! Probe replicates the same computation and tests whether every bit the
//! mask would set is already set.

use bytes::{Buf, Bytes};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result};

/// Default key density. 10 bits/key gives ~1 % FPR for SBBF.
pub const DEFAULT_BITS_PER_KEY: u8 = 10;

/// Threshold below which the writer omits the bloom entirely (RFC-002 §4.2).
pub const BLOOM_OMIT_THRESHOLD_BYTES: u64 = 256 * 1024;

const MAGIC: [u8; 8] = *b"TGBLOOM\0";
const HEADER_LEN: usize = 28;
const TRAILER_LEN: usize = 8;
const BLOCK_BYTES: usize = 32; // 256 bits

const SALT: [u32; 8] = [
 0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
];

/// Pointer + parameters describing a bloom side-car (lives in the manifest's
/// `SstDescriptor`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BloomDescriptor {
 pub path: String,
 pub size_bytes: u32,
 pub key_count: u64,
 pub bits_per_key: u8,
 pub block_count: u32,
 pub xxhash3_64: u64,
}

impl BloomDescriptor {
 /// Build a descriptor from a freshly serialised side-car body.
 ///
 /// Performs the same structural checks the reader does (magic, major,
 /// declared size matches actual size) so a descriptor built here can be
 /// trusted by downstream code without re-parsing the body.
 pub fn from_body(path: impl Into<String>, body: &[u8]) -> Result<Self> {
 let path = path.into();
 if body.len() < HEADER_LEN + TRAILER_LEN {
 return Err(Error::Corrupted {
 path,
 detail: format!("bloom body too small ({} bytes)", body.len()),
 });
 }
 if body[..8] != MAGIC {
 return Err(Error::Corrupted {
 path,
 detail: "bloom magic mismatch".into(),
 });
 }
 let major = body[8];
 if major != 1 {
 return Err(Error::Corrupted {
 path,
 detail: format!("unsupported bloom format_major={major}"),
 });
 }
 let bits_per_key = body[12];
 let block_count = u32::from_le_bytes([body[16], body[17], body[18], body[19]]);
 let key_count = u64::from_le_bytes(body[20..28].try_into().unwrap());
 let expected_total = HEADER_LEN + BLOCK_BYTES * block_count as usize + TRAILER_LEN;
 if body.len() != expected_total {
 return Err(Error::Corrupted {
 path,
 detail: format!(
 "bloom size mismatch: header says {expected_total} bytes, got {}",
 body.len()
 ),
 });
 }
 let trailer_start = body.len() - TRAILER_LEN;
 let xxhash3_64 = u64::from_le_bytes(body[trailer_start..].try_into().unwrap());
 Ok(Self {
 path,
 size_bytes: body.len() as u32,
 key_count,
 bits_per_key,
 block_count,
 xxhash3_64,
 })
 }
}

/// In-memory SBBF used by the writer.
#[derive(Debug, Clone)]
pub struct BloomFilter {
 blocks: Vec<[u32; 8]>,
 bits_per_key: u8,
 key_count: u64,
}

impl BloomFilter {
 /// New empty filter sized for `expected_keys` at `bits_per_key`.
 pub fn with_capacity(expected_keys: u64, bits_per_key: u8) -> Self {
 let bpk = bits_per_key.max(1) as u64;
 let total_bits = expected_keys.saturating_mul(bpk).max(256); // ≥1 block
 let block_count = total_bits.div_ceil(256).max(1) as usize;
 Self {
 blocks: vec![[0u32; 8]; block_count],
 bits_per_key,
 key_count: 0,
 }
 }

 /// Insert a raw 16-byte key (e.g. a `NodeId`).
 pub fn insert(&mut self, key: &[u8]) {
 let h = xxh3_64(key);
 let block_index = self.block_index(h);
 let mask = block_mask(h as u32);
 let block = &mut self.blocks[block_index];
 for i in 0..8 {
 block[i] |= mask[i];
 }
 self.key_count = self.key_count.saturating_add(1);
 }

 /// Probe membership. False positives possible, false negatives never.
 pub fn contains(&self, key: &[u8]) -> bool {
 let h = xxh3_64(key);
 let block_index = self.block_index(h);
 let mask = block_mask(h as u32);
 let block = &self.blocks[block_index];
 for i in 0..8 {
 if (block[i] & mask[i]) != mask[i] {
 return false;
 }
 }
 true
 }

 fn block_index(&self, h: u64) -> usize {
 // Map the top 32 bits of h into [0, block_count).
 (((h >> 32) * self.blocks.len() as u64) >> 32) as usize
 }

 /// Number of blocks (each 256 bits).
 pub fn block_count(&self) -> u32 {
 self.blocks.len() as u32
 }

 pub fn key_count(&self) -> u64 {
 self.key_count
 }

 pub fn bits_per_key(&self) -> u8 {
 self.bits_per_key
 }

 /// Encode to the side-car wire format.
 pub fn to_bytes(&self) -> Bytes {
 let block_count = self.blocks.len();
 let mut buf = Vec::with_capacity(HEADER_LEN + BLOCK_BYTES * block_count + TRAILER_LEN);
 buf.extend_from_slice(&MAGIC); // 0..8
 buf.push(1); // 8 format_major
 buf.push(0); // 9 format_minor
 buf.extend_from_slice(&0u16.to_le_bytes()); // 10..12 reserved
 buf.push(self.bits_per_key); // 12
 buf.push(0); // 13 reserved2
 buf.extend_from_slice(&0u16.to_le_bytes()); // 14..16 reserved3
 buf.extend_from_slice(&(block_count as u32).to_le_bytes()); // 16..20
 buf.extend_from_slice(&self.key_count.to_le_bytes()); // 20..28
 debug_assert_eq!(buf.len(), HEADER_LEN);
 for block in &self.blocks {
 for word in block {
 buf.extend_from_slice(&word.to_le_bytes());
 }
 }
 let body_hash = xxh3_64(&buf);
 buf.extend_from_slice(&body_hash.to_le_bytes());
 Bytes::from(buf)
 }

 /// Decode a side-car body back into an in-memory filter, verifying
 /// the magic, version and trailing checksum.
 pub fn from_bytes(path: &str, body: &[u8]) -> Result<Self> {
 if body.len() < HEADER_LEN + TRAILER_LEN {
 return Err(Error::Corrupted {
 path: path.to_string(),
 detail: format!("bloom body too small ({} bytes)", body.len()),
 });
 }
 if body[..8] != MAGIC {
 return Err(Error::Corrupted {
 path: path.to_string(),
 detail: "bloom magic mismatch".into(),
 });
 }
 let major = body[8];
 if major != 1 {
 return Err(Error::Corrupted {
 path: path.to_string(),
 detail: format!("unsupported bloom format_major={major}"),
 });
 }
 // minor (body[9]) is forward-compatible.
 let bits_per_key = body[12];
 let block_count = u32::from_le_bytes([body[16], body[17], body[18], body[19]]) as usize;
 let key_count = u64::from_le_bytes(body[20..28].try_into().unwrap());
 let body_end = HEADER_LEN + BLOCK_BYTES * block_count;
 let expected_total = body_end + TRAILER_LEN;
 if body.len() != expected_total {
 return Err(Error::Corrupted {
 path: path.to_string(),
 detail: format!(
 "bloom size mismatch: expected {expected_total}, got {}",
 body.len()
 ),
 });
 }
 let body_hash_computed = xxh3_64(&body[..body_end]);
 let body_hash_stored =
 u64::from_le_bytes(body[body_end..body_end + TRAILER_LEN].try_into().unwrap());
 if body_hash_computed != body_hash_stored {
 return Err(Error::Corrupted {
 path: path.to_string(),
 detail: "bloom trailer xxhash mismatch".into(),
 });
 }
 let mut blocks = Vec::with_capacity(block_count);
 let mut cursor = &body[HEADER_LEN..body_end];
 for _ in 0..block_count {
 let mut block = [0u32; 8];
 for word in &mut block {
 *word = cursor.get_u32_le();
 }
 blocks.push(block);
 }
 Ok(Self {
 blocks,
 bits_per_key,
 key_count,
 })
 }
}

fn block_mask(key32: u32) -> [u32; 8] {
 let mut mask = [0u32; 8];
 for i in 0..8 {
 let bit_index = (SALT[i].wrapping_mul(key32)) >> 27; // 5 bits → 0..32
 mask[i] = 1u32 << (bit_index & 31);
 }
 mask
}

#[cfg(test)]
mod tests {
 use super::*;
 use uuid::Uuid;

 fn key(i: u64) -> [u8; 16] {
 let mut k = [0u8; 16];
 k[..8].copy_from_slice(&i.to_le_bytes());
 k
 }

 #[test]
 fn empty_filter_has_at_least_one_block() {
 let f = BloomFilter::with_capacity(0, DEFAULT_BITS_PER_KEY);
 assert!(f.block_count() >= 1);
 }

 #[test]
 fn inserted_keys_are_present() {
 let mut f = BloomFilter::with_capacity(1000, DEFAULT_BITS_PER_KEY);
 for i in 0..1000u64 {
 f.insert(&key(i));
 }
 for i in 0..1000u64 {
 assert!(f.contains(&key(i)), "missing key {i}");
 }
 assert_eq!(f.key_count(), 1000);
 }

 #[test]
 fn false_positive_rate_bounded() {
 let n = 10_000u64;
 let mut f = BloomFilter::with_capacity(n, DEFAULT_BITS_PER_KEY);
 for i in 0..n {
 f.insert(&key(i));
 }
 let mut fp = 0u64;
 let probes = 50_000u64;
 for i in n..(n + probes) {
 if f.contains(&key(i)) {
 fp += 1;
 }
 }
 let fpr = fp as f64 / probes as f64;
 // Theoretical for SBBF at 10 bits/key is around 1 %; allow up to 4 %
 // to absorb statistical noise on small samples.
 assert!(fpr < 0.04, "FPR {fpr} too high (fp={fp}, probes={probes})");
 }

 #[test]
 fn round_trip_through_wire_format() {
 let mut f = BloomFilter::with_capacity(500, DEFAULT_BITS_PER_KEY);
 let uuids: Vec<Uuid> = (0..500).map(|_| Uuid::now_v7()).collect();
 for u in &uuids {
 f.insert(u.as_bytes());
 }
 let bytes = f.to_bytes();
 let reloaded = BloomFilter::from_bytes("test.bloom", &bytes).unwrap();
 assert_eq!(reloaded.block_count(), f.block_count());
 assert_eq!(reloaded.key_count(), f.key_count());
 for u in &uuids {
 assert!(reloaded.contains(u.as_bytes()));
 }
 }

 #[test]
 fn rejects_bad_magic() {
 let mut bytes = BloomFilter::with_capacity(10, DEFAULT_BITS_PER_KEY)
 .to_bytes()
 .to_vec();
 bytes[0] = b'X';
 let err = BloomFilter::from_bytes("test.bloom", &bytes).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("magic")),
 other => panic!("expected Corrupted, got {other:?}"),
 }
 }

 #[test]
 fn rejects_bad_trailer_xxhash() {
 let mut f = BloomFilter::with_capacity(10, DEFAULT_BITS_PER_KEY);
 f.insert(&[0u8; 16]);
 let mut bytes = f.to_bytes().to_vec();
 let last = bytes.len() - 1;
 bytes[last] ^= 0xff;
 let err = BloomFilter::from_bytes("test.bloom", &bytes).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("xxhash")),
 other => panic!("expected Corrupted, got {other:?}"),
 }
 }

 #[test]
 fn rejects_unsupported_major() {
 let mut bytes = BloomFilter::with_capacity(10, DEFAULT_BITS_PER_KEY)
 .to_bytes()
 .to_vec();
 bytes[8] = 2; // format_major = 2
 // Trailer must remain valid for this test to isolate the major check —
 // recompute it over the modified body.
 let body_end = bytes.len() - TRAILER_LEN;
 let new_hash = xxh3_64(&bytes[..body_end]);
 bytes[body_end..].copy_from_slice(&new_hash.to_le_bytes());
 let err = BloomFilter::from_bytes("test.bloom", &bytes).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("format_major")),
 other => panic!("expected Corrupted, got {other:?}"),
 }
 }

 #[test]
 fn descriptor_from_body_matches_filter() {
 let mut f = BloomFilter::with_capacity(123, DEFAULT_BITS_PER_KEY);
 for i in 0..123 {
 f.insert(&key(i));
 }
 let bytes = f.to_bytes();
 let d = BloomDescriptor::from_body("path/x.bloom", &bytes).unwrap();
 assert_eq!(d.size_bytes as usize, bytes.len());
 assert_eq!(d.block_count, f.block_count());
 assert_eq!(d.key_count, f.key_count());
 assert_eq!(d.bits_per_key, DEFAULT_BITS_PER_KEY);
 }

 #[test]
 fn descriptor_from_body_rejects_size_mismatch() {
 let f = BloomFilter::with_capacity(10, DEFAULT_BITS_PER_KEY);
 let mut bytes = f.to_bytes().to_vec();
 // Bump the declared block_count by 1 without growing the body —
 // expected size now exceeds actual size.
 let inflated = (f.block_count() + 1).to_le_bytes();
 bytes[16..20].copy_from_slice(&inflated);
 let err = BloomDescriptor::from_body("p.bloom", &bytes).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("size mismatch")),
 other => panic!("expected Corrupted, got {other:?}"),
 }
 }

 #[test]
 fn descriptor_from_body_rejects_bad_magic() {
 let f = BloomFilter::with_capacity(10, DEFAULT_BITS_PER_KEY);
 let mut bytes = f.to_bytes().to_vec();
 bytes[0] = b'!';
 let err = BloomDescriptor::from_body("p.bloom", &bytes).unwrap_err();
 match err {
 Error::Corrupted { detail, .. } => assert!(detail.contains("magic")),
 other => panic!("expected Corrupted, got {other:?}"),
 }
 }
}
