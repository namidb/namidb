//! # namidb
//!
//! Public façade crate. Re-exports the curated public surface of the
//! storage / graph / query crates so end users only depend on one crate.
//!
//! ```toml
//! namidb = "0.0.1"
//! ```
//!
//! For now we just re-export the foundational types from `namidb-core`.

#![warn(rust_2018_idioms)]

pub use namidb_core::*;
