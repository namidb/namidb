//! Edge SST (CSR binary) format.
//!
//! Defined by [RFC-002](../../../../../docs/rfc/002-sst-format.md) §3.
//!
//! Each flush emits two authoritative physical files per `(edge_type, level)`
//! bucket: a **forward** SST (sorted by `src_id`) and an **inverse** SST
//! (sorted by `dst_id`). Both share the same wire format, differentiated by
//! `flags.INVERSE_PARTNER`. Current forward SSTs may also carry a bounded,
//! optional `.epidx` point accelerator; the CSR remains the authority.

pub mod encoding;
pub mod fence_index;
pub mod format;
pub mod inverse;
pub mod paged_reader;
pub(crate) mod point_index;
pub(crate) mod property_pages;
pub mod reader;
pub mod writer;

pub use fence_index::{DEFAULT_FENCE_STRIDE, FENCE_INDEX_THRESHOLD};
pub use format::{
    EdgeFileFooter, EdgeFileHeader, EdgePageChecksumDirectory, EdgeSstBinding, SectionEntry,
    SectionPageChecksums, EDGE_CHECKSUM_PAGE_BYTES, FLAG_HAS_PROPERTIES, FLAG_HAS_TOMBSTONES,
    FLAG_INVERSE_PARTNER, FLAG_SKEW_BUCKETS, SECTION_EDGE_ORDINALS, SECTION_PAGE_CHECKSUMS,
    SECTION_SST_BINDING,
};
pub use paged_reader::{
    PagedEdgeIoStats, PagedEdgeReader, EDGE_FOOTER_PREFETCH_BYTES, MAX_CACHED_EDGE_FENCE_BYTES,
    MAX_EDGE_FOOTER_BYTES, MAX_UNFENCED_KEY_IDS_BYTES,
};
pub use reader::{EdgeLookup, EdgePointLookup, EdgeRowProjection, EdgeSstReader};
pub use writer::{
    EdgeRecord, EdgeSstFinish, EdgeSstStats, EdgeSstWriter, EdgeSstWriterOptions,
    DEFAULT_EDGE_IN_MEMORY_FINISH_MAX_BYTES, EDGE_IN_MEMORY_FINISH_MAX_BYTES_ENV,
};

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
