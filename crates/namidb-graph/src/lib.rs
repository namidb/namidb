//! # namidb-graph
//!
//! Property-graph data structures and analytical kernels on top of
//! [`namidb_storage`]. The storage layer serves CSR-backed adjacency; this
//! crate hosts the algorithms that run over it.
//!
//! Today the crate ships [`algo`] — exact in-memory graph kernels (WCC and
//! PageRank) over a [`algo::Graph`] built from snapshot edges. These are the
//! first native graph algorithms; the CALL/YIELD Cypher surface that exposes
//! them as procedures is a fast-follow (see RFC-023 for the shortest-path
//! precedent).

#![warn(rust_2018_idioms)]

pub mod algo;
