//! SST (sorted string table) layer.
//!
//! Defined by [RFC-002](../../../../docs/rfc/002-sst-format.md).
//!
//! - [`stats`] — `PropertyColumnStats`, `DegreeHistogram`, `HllSketchBytes`.
//! - [`hll`] — HyperLogLog cardinality sketch (RFC-010 follow-up).
//! - [`bloom`] — split-block bloom filter + side-car wire format.
//! - [`nodes`] — Parquet node SST writer + reader.
//! - [`edges`] — CSR binary edge SST (forward + inverse partners).

pub mod bloom;
pub mod edges;
pub mod hll;
pub mod nodes;
pub mod predicates;
pub mod stats;

pub use bloom::{BloomDescriptor, BloomFilter, BLOOM_OMIT_THRESHOLD_BYTES, DEFAULT_BITS_PER_KEY};
pub use edges::{
    EdgeDirection, EdgeRecord, EdgeSstFinish, EdgeSstReader, EdgeSstStats, EdgeSstWriter,
    EdgeSstWriterOptions,
};
pub use hll::{Hll, DEFAULT_PRECISION as HLL_DEFAULT_PRECISION};
pub use nodes::{
    targeted_scan_async as node_targeted_scan_async, NodeSstReader, NodeSstWriter,
    NodeSstWriterOptions, OVERFLOW_JSON, SCHEMA_VERSION,
};
pub use predicates::{eval_against_value, eval_row_group, RowGroupVerdict, ScanPredicate};
pub use stats::{DegreeHistogram, HllSketchBytes, PropertyColumnStats, StatScalar};
