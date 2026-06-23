//! # namidb-ann
//!
//! DiskANN/Vamana approximate-nearest-neighbor graph index — the algorithm
//! layer, storage-agnostic. See the crate README for the design rationale.
//!
//! The crate is organized so the algorithm ([`search`], [`build`]) is generic
//! over a [`VectorSpace`] and never touches a byte of object storage. A storage
//! integration wires this into `SstKind::VectorGraph` behind a Cargo feature in
//! `namidb-storage`.

#![warn(rust_2018_idioms)]

pub mod build;
pub mod graph;
pub mod search;
pub mod space;

pub use build::{build, build_with_seed, BuildParams, InitStrategy};
pub use graph::VamanaGraph;
pub use search::search;
pub use space::{F32CosineSpace, Int8Space, L2Space, VectorSpace};
