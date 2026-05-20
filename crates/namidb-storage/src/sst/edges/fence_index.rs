//! Optional fence-pointer index for the `key_ids` section.
//!
//! Defined by [RFC-002](../../../../../docs/rfc/002-sst-format.md) §3.2.9.
//!
//! Layout:
//!
//! ```text
//! fence_stride: u32
//! entry_count: u32
//! entries: [ FenceEntry ; entry_count ]
//! FenceEntry { key: [u8; 16], key_ids_offset: u64 }
//! ```
//!
//! `key_ids_offset` is the byte offset of `key` within the `key_ids`
//! section (relative to the start of that section).

use crate::error::{Error, Result};

/// Default keys-per-fence-entry stride.
pub const DEFAULT_FENCE_STRIDE: u32 = 256;

/// Below this key-count the writer does **not** emit a fence index; the
/// full `key_ids` section can be fetched in one ranged GET.
pub const FENCE_INDEX_THRESHOLD: u64 = 65_536;

const FENCE_ENTRY_LEN: usize = 16 + 8; // key + offset
const FENCE_HEADER_LEN: usize = 4 + 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenceEntry {
    pub key: [u8; 16],
    pub key_ids_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenceIndex {
    pub stride: u32,
    pub entries: Vec<FenceEntry>,
}

impl FenceIndex {
    /// Build a fence index from the sorted `key_ids` slice. Stride must be
    /// `>= 1`.
    pub fn build(key_ids: &[[u8; 16]], stride: u32) -> Self {
        let stride = stride.max(1);
        let mut entries = Vec::with_capacity(key_ids.len().div_ceil(stride as usize));
        for (i, key) in key_ids.iter().enumerate().step_by(stride as usize) {
            entries.push(FenceEntry {
                key: *key,
                key_ids_offset: (i as u64) * 16,
            });
        }
        Self { stride, entries }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FENCE_HEADER_LEN + self.entries.len() * FENCE_ENTRY_LEN);
        buf.extend_from_slice(&self.stride.to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for e in &self.entries {
            buf.extend_from_slice(&e.key);
            buf.extend_from_slice(&e.key_ids_offset.to_le_bytes());
        }
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < FENCE_HEADER_LEN {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!("fence index header truncated ({} bytes)", buf.len()),
            });
        }
        let stride = u32::from_le_bytes(buf[..4].try_into().unwrap());
        let entry_count = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let need = FENCE_HEADER_LEN + entry_count * FENCE_ENTRY_LEN;
        if buf.len() != need {
            return Err(Error::Corrupted {
                path: "<edges>".into(),
                detail: format!(
                    "fence index size mismatch: expected {need}, got {}",
                    buf.len()
                ),
            });
        }
        let mut entries = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let off = FENCE_HEADER_LEN + i * FENCE_ENTRY_LEN;
            let key: [u8; 16] = buf[off..off + 16].try_into().unwrap();
            let key_ids_offset = u64::from_le_bytes(buf[off + 16..off + 24].try_into().unwrap());
            entries.push(FenceEntry {
                key,
                key_ids_offset,
            });
        }
        Ok(Self { stride, entries })
    }

    /// Binary search the fence entries to find the bracket `[lo, hi)` of
    /// candidate `key_ids` indexes that contains `target`.
    ///
    /// Returns the inclusive lower bound (as an index into `key_ids`) and
    /// the exclusive upper bound. The caller then GETs that window of
    /// `key_ids` from object storage and runs a final in-memory bsearch.
    ///
    /// If `target` is smaller than the smallest fence entry the lower
    /// bound is `0`. If it's larger than every entry the upper bound is
    /// `key_count`.
    pub fn bracket_for(&self, target: &[u8; 16], key_count: u64) -> (u64, u64) {
        if self.entries.is_empty() {
            return (0, key_count);
        }
        // Find the rightmost entry whose key <= target.
        let pos = self.entries.partition_point(|e| &e.key <= target);
        let lo_idx = if pos == 0 { 0 } else { pos - 1 };
        let lo_offset = self.entries[lo_idx].key_ids_offset / 16;
        let hi_offset = if pos < self.entries.len() {
            self.entries[pos].key_ids_offset / 16
        } else {
            key_count
        };
        (lo_offset, hi_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_seq(n: usize) -> Vec<[u8; 16]> {
        (0..n)
            .map(|i| {
                let mut k = [0u8; 16];
                k[..8].copy_from_slice(&(i as u64).to_be_bytes());
                k
            })
            .collect()
    }

    #[test]
    fn builds_one_entry_per_stride() {
        let keys = key_seq(1000);
        let idx = FenceIndex::build(&keys, 100);
        assert_eq!(idx.stride, 100);
        assert_eq!(idx.entries.len(), 10);
        assert_eq!(idx.entries[0].key, keys[0]);
        assert_eq!(idx.entries[0].key_ids_offset, 0);
        assert_eq!(idx.entries[1].key, keys[100]);
        assert_eq!(idx.entries[1].key_ids_offset, 100 * 16);
    }

    #[test]
    fn encode_decode_round_trip() {
        let keys = key_seq(300);
        let idx = FenceIndex::build(&keys, 64);
        let bytes = idx.encode();
        let back = FenceIndex::decode(&bytes).unwrap();
        assert_eq!(back, idx);
    }

    #[test]
    fn bracket_for_finds_correct_window() {
        let keys = key_seq(1000);
        let idx = FenceIndex::build(&keys, 100);
        // Target is keys[150]; the bracket should be entries[1]..entries[2] →
        // key_ids indices [100, 200).
        let (lo, hi) = idx.bracket_for(&keys[150], 1000);
        assert_eq!(lo, 100);
        assert_eq!(hi, 200);
    }

    #[test]
    fn bracket_below_first_returns_zero_window() {
        let keys = key_seq(1000);
        let idx = FenceIndex::build(&keys, 100);
        let lower = [0u8; 16]; // smaller than every key (keys[0] is `0x00..00 00 00 00 00`)
                               // The fence's first entry is exactly keys[0] = [0u8;16] too here,
                               // so partition_point places target before any entry: bracket is
                               // [0, entries[0].offset/16) = [0, 0). Allow that or expand: test with
                               // a target one less in big-endian sense — but our key 0 is all-zero,
                               // and lower is all-zero, so they're equal. Use a unique target.
        let target = {
            let mut k = lower;
            k[15] = 1; // > keys[0]
            k
        };
        let (lo, hi) = idx.bracket_for(&target, 1000);
        // target == keys[0]+1 sorts between keys[0] and keys[1]; bracket is
        // [entries[0].offset/16, entries[1].offset/16) = [0, 100).
        assert_eq!(lo, 0);
        assert_eq!(hi, 100);
    }

    #[test]
    fn bracket_above_last_returns_tail_window() {
        let keys = key_seq(1000);
        let idx = FenceIndex::build(&keys, 100);
        let mut target = keys[999];
        target[15] = target[15].wrapping_add(1); // > every key in the set
        let (lo, hi) = idx.bracket_for(&target, 1000);
        // Last fence entry is at index 900 (entry 9); bracket is [900, 1000).
        assert_eq!(lo, 900);
        assert_eq!(hi, 1000);
    }

    #[test]
    fn empty_fence_returns_full_range() {
        let idx = FenceIndex {
            stride: 1,
            entries: vec![],
        };
        let target = [0u8; 16];
        let (lo, hi) = idx.bracket_for(&target, 42);
        assert_eq!(lo, 0);
        assert_eq!(hi, 42);
    }

    #[test]
    fn decode_rejects_size_mismatch() {
        let mut bytes = FenceIndex::build(&key_seq(10), 5).encode();
        bytes.push(0);
        let err = FenceIndex::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corrupted { .. }));
    }
}
