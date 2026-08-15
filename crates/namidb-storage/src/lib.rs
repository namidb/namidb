//! # namidb-storage
//!
//! LSM storage engine on top of [`object_store::ObjectStore`].
//!
//! ## Modules
//!
//! - [`paths`] — canonical key derivations for namespace layouts.
//! - [`manifest`] — manifest document + CAS commit protocol.
//! - [`fence`] — single-writer epoch fencing primitives.
//! - [`error`] — storage-specific error enum.
//! - `wal`, `memtable`, `sst`, `compaction` — coming online incrementally.

#![warn(rust_2018_idioms)]
#![deny(missing_debug_implementations)]

pub mod adjacency;
pub mod backup;
pub mod cache;
pub mod cache_budget;
pub mod cancel;
pub mod compact;
pub mod error;
pub mod fence;
pub mod flush;
pub mod ingest;
pub mod janitor;
pub mod local;
pub mod manifest;
pub mod memtable;
pub mod node_cache;
pub mod parquet_loader;
pub mod paths;
pub mod pin;
pub mod property_index;
pub mod range_cache;
pub mod read;
pub mod recovery;
pub mod route_telemetry;
pub mod search_lsm;
pub(crate) mod search_lsm_flush;
pub mod search_workspace;
pub(crate) mod spooled_object;
pub mod sst;
#[cfg(test)]
pub(crate) mod test_support;
pub mod text;
pub mod unique_index;
pub mod uri;
pub mod wal;

pub use adjacency::{
    adjacency_budget_bytes, adjacency_enabled, build_adjacency, shared_adjacency_cache,
    AdjacencyCache, AdjacencyKey, EdgeAdjacency, EdgeSlice, DEFAULT_ADJACENCY_BUDGET_MIB,
};
pub use backup::{copy_namespace_snapshot, SnapshotCopyReport};
pub use cache::{
    shared_sst_cache, sst_cache_budget_bytes, sst_cache_enabled, EdgeStreamBundle, SstCache,
    DEFAULT_SST_CACHE_BUDGET_MIB,
};
pub use cache_budget::{
    cache_max_bytes, search_index_cache_max_bytes, shared_cache_capacities,
    shared_cache_capacity_bytes, shared_cache_usage_bytes, validate_cache_configuration,
    CacheCapacities, DEFAULT_CACHE_MAX_BYTES,
};
pub use compact::{
    compact_l0_to_l1, install_prepared, prepare_compaction, CompactionBasis, CompactionOutcome,
    PreparedCompaction,
};
pub use error::{Error, Result};
pub use fence::{Epoch, WriterFence};
pub use flush::{flush, EdgeWriteRecord, FlushOutcome, NodeWriteRecord};
pub use ingest::{
    clear_shared_caches, prune_shared_caches, CommitOutcome, SessionCaches, StagedValue,
    WriterSession,
};
pub use janitor::{sweep_orphans, JanitorReport};
pub use local::LocalFileObjectStore;
pub use manifest::{
    KindSpecificStats, Manifest, ManifestStore, NodePropertyPagesDescriptor, SstDescriptor,
    SstKind, SstLevel, WalSegmentDescriptor,
};
pub use memtable::{FrozenMemtable, MemEntry, MemKey, MemOp, Memtable, MemtableSnapshot};
pub use node_cache::{
    node_cache_budget_bytes, node_cache_enabled, shared_node_cache, CachedNodeView, NodeCacheKey,
    NodeViewCache, DEFAULT_NODE_CACHE_BUDGET_MIB,
};
pub use parquet_loader::{
    load_edges as load_edges_from_parquet, load_nodes as load_nodes_from_parquet, LoadOutcome,
};
pub use paths::NamespacePaths;
pub use pin::{PinLease, RetentionPin, DEFAULT_PIN_TTL};
pub use range_cache::{
    shared_range_cache, ImmutableRangeCache, ImmutableRangeKey, PinnedObjectGeneration,
    PinnedObjectRangeSource, RangeCacheConfig, RangeCacheError, RangeCacheStats,
    DEFAULT_NVME_CACHE_BLOCK_BYTES, DEFAULT_NVME_CACHE_WRITE_BUFFER_BYTES,
    DEFAULT_RAM_PAGE_CACHE_MAX_BYTES, DEFAULT_RANGE_CACHE_MAX_ENTRY_BYTES,
    DEFAULT_RANGE_CACHE_PAGE_BYTES,
};
#[cfg(feature = "vector-index")]
pub use read::VectorFilterSearch;
pub use read::{
    EdgeListView, EdgeView, NodeView, OwnedSnapshot, PinnedSnapshot, Snapshot, SnapshotCell,
};
pub use recovery::{
    recover_memtable, recover_memtable_with_snapshot, write_memtable_snapshot,
    MemtableSnapshotEntry, MemtableSnapshotFile, RecoveredMemtable, WalEntry, WalOp,
};
#[cfg(feature = "vector-index")]
pub use sst::vector::vector_filter_bitmap_searches;
pub use sst::{
    BloomDescriptor, BloomFilter, DegreeHistogram, EdgeDirection, EdgePointLookup, EdgeRecord,
    EdgeSstFinish, EdgeSstReader, EdgeSstStats, EdgeSstWriter, EdgeSstWriterOptions, NodeSstReader,
    NodeSstWriter, NodeSstWriterOptions, PropertyColumnStats, StatScalar,
};
pub use unique_index::UniqueProbe;
pub use uri::{parse_uri, UriError};
pub use wal::{WalRecord, WalSegment, WalSegmentRef, WalStore};
