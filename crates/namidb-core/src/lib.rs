//! # namidb-core
//!
//! Common types, errors and schema primitives shared by every NamiDB crate.
//!
//! Nothing in this crate touches I/O or the network: it is the pure-data
//! foundation of the engine. The expensive cousins (`namidb-storage`,
//! `namidb-graph`, `namidb-query`) build on top.

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod error;
pub mod id;
pub mod profile;
pub mod quantize;
pub mod schema;
pub mod value;

pub use error::{Error, Result};
pub use id::{EdgeId, LabelId, NamespaceId, NodeId};
pub use quantize::{dequantize_i8, quantize_i8};
pub use schema::{
    Constraint, ConstraintKind, DataType, EdgeTypeDef, LabelDef, LabelDictionary, PropertyDef,
    Schema, SchemaBuilder,
};
pub use value::Value;
