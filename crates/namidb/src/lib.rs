//! # namidb
//!
//! Workspace façade crate. It currently re-exports the pure-data
//! [`namidb_core`] surface while the curated embedded-engine API is prepared.
//! This crate is not published to crates.io yet.
//!
//! ```toml
//! namidb = { git = "https://github.com/namidb/namidb", tag = "v2.0.5" }
//! ```
//!
//! Git/path consumers should pin an exact revision and treat this low-level
//! workspace surface as unstable until the first explicit crates.io release.
//!
//! ```
//! use namidb::{DataType, LabelDef, PropertyDef, Schema};
//!
//! fn main() -> namidb::Result<()> {
//!     let schema = Schema::builder()
//!         .label(LabelDef {
//!             name: "Person".into(),
//!             properties: vec![PropertyDef::new("name", DataType::Utf8, false)?],
//!         })?
//!         .build();
//!
//!     assert!(schema.label("Person").is_some());
//!     Ok(())
//! }
//! ```

#![warn(rust_2018_idioms)]

pub use namidb_core::*;
