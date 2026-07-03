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
pub mod read;
pub mod recovery;
pub mod sst;
pub mod text;
pub mod unique_index;
pub mod uri;
pub mod wal;

pub use adjacency::{
    adjacency_budget_bytes, adjacency_enabled, build_adjacency, AdjacencyCache, AdjacencyKey,
    EdgeAdjacency, EdgeSlice, DEFAULT_ADJACENCY_BUDGET_MIB,
};
pub use backup::{copy_namespace_snapshot, SnapshotCopyReport};
pub use cache::{
    sst_cache_budget_bytes, sst_cache_enabled, EdgeStreamBundle, SstCache,
    DEFAULT_SST_CACHE_BUDGET_MIB,
};
pub use compact::{
    compact_l0_to_l1, install_prepared, prepare_compaction, CompactionBasis, CompactionOutcome,
    PreparedCompaction,
};
pub use error::{Error, Result};
pub use fence::{Epoch, WriterFence};
pub use flush::{flush, EdgeWriteRecord, FlushOutcome, NodeWriteRecord};
pub use ingest::{CommitOutcome, WriterSession};
pub use janitor::{sweep_orphans, JanitorReport};
pub use local::LocalFileObjectStore;
pub use manifest::{
    KindSpecificStats, Manifest, ManifestStore, SstDescriptor, SstKind, SstLevel,
    WalSegmentDescriptor,
};
pub use memtable::{FrozenMemtable, MemEntry, MemKey, MemOp, Memtable, MemtableSnapshot};
pub use node_cache::{
    node_cache_budget_bytes, node_cache_enabled, CachedNodeView, NodeCacheKey, NodeViewCache,
    DEFAULT_NODE_CACHE_BUDGET_MIB,
};
pub use parquet_loader::{
    load_edges as load_edges_from_parquet, load_nodes as load_nodes_from_parquet, LoadOutcome,
};
pub use paths::NamespacePaths;
pub use pin::{PinLease, RetentionPin, DEFAULT_PIN_TTL};
pub use read::{
    EdgeListView, EdgeView, NodeView, OwnedSnapshot, PinnedSnapshot, Snapshot, SnapshotCell,
};
pub use recovery::{
    recover_memtable, recover_memtable_with_snapshot, write_memtable_snapshot,
    MemtableSnapshotEntry, MemtableSnapshotFile, RecoveredMemtable, WalEntry, WalOp,
};
pub use sst::{
    BloomDescriptor, BloomFilter, DegreeHistogram, EdgeDirection, EdgeRecord, EdgeSstFinish,
    EdgeSstReader, EdgeSstStats, EdgeSstWriter, EdgeSstWriterOptions, NodeSstReader, NodeSstWriter,
    NodeSstWriterOptions, PropertyColumnStats, StatScalar,
};
pub use unique_index::UniqueProbe;
pub use uri::{parse_uri, UriError};
pub use wal::{WalRecord, WalSegment, WalSegmentRef, WalStore};
