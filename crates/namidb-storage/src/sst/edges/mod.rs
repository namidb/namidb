//! Edge SST (CSR binary) format.
//!
//! Defined by [RFC-002](../../../../../docs/rfc/002-sst-format.md) §3.
//!
//! Each flush emits **two** physical files per `(edge_type, level)` bucket:
//! a **forward** SST (sorted by `src_id`) and an **inverse** SST (sorted by
//! `dst_id`). Both share the same wire format, differentiated by
//! `flags.INVERSE_PARTNER`.

pub mod encoding;
pub mod fence_index;
pub mod format;
pub mod inverse;
pub mod reader;
pub mod writer;

pub use fence_index::{DEFAULT_FENCE_STRIDE, FENCE_INDEX_THRESHOLD};
pub use format::{
 EdgeFileFooter, EdgeFileHeader, SectionEntry, FLAG_HAS_PROPERTIES, FLAG_HAS_TOMBSTONES,
 FLAG_INVERSE_PARTNER, FLAG_SKEW_BUCKETS,
};
pub use reader::{EdgeLookup, EdgeRowProjection, EdgeSstReader};
pub use writer::{EdgeRecord, EdgeSstFinish, EdgeSstStats, EdgeSstWriter, EdgeSstWriterOptions};

/// Direction of an edge SST.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeDirection {
 /// `key_ids` are `src_id`; partners are `dst_id`. Reads "out-edges of `s`".
 Forward,
 /// `key_ids` are `dst_id`; partners are `src_id`. Reads "in-edges of `d`".
 Inverse,
}

impl EdgeDirection {
 pub fn flag_bit(self) -> u32 {
 match self {
 EdgeDirection::Forward => 0,
 EdgeDirection::Inverse => FLAG_INVERSE_PARTNER,
 }
 }

 pub fn from_flags(flags: u32) -> Self {
 if flags & FLAG_INVERSE_PARTNER != 0 {
 EdgeDirection::Inverse
 } else {
 EdgeDirection::Forward
 }
 }

 /// Path tag used in the SST filename (RFC-002 §1).
 pub fn path_tag(self) -> &'static str {
 match self {
 EdgeDirection::Forward => "edges-fwd",
 EdgeDirection::Inverse => "edges-inv",
 }
 }
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn direction_round_trips_through_flags() {
 assert_eq!(
 EdgeDirection::from_flags(EdgeDirection::Forward.flag_bit()),
 EdgeDirection::Forward
 );
 assert_eq!(
 EdgeDirection::from_flags(EdgeDirection::Inverse.flag_bit()),
 EdgeDirection::Inverse
 );
 }

 #[test]
 fn direction_path_tags_match_rfc() {
 assert_eq!(EdgeDirection::Forward.path_tag(), "edges-fwd");
 assert_eq!(EdgeDirection::Inverse.path_tag(), "edges-inv");
 }
}
