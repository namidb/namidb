//! Cost-based optimizer foundation (RFC-010).
//!
//! This module exposes three pieces that the optimizer builds on:
//!
//! 1. [`stats::StatsCatalog`] — per-label / per-edge-type aggregate
//! statistics derived from a committed [`Manifest`].
//! 2. [`selectivity::selectivity`] — pure estimate of the fraction of
//! rows a predicate retains, given the label/edge stats of the
//! bindings in scope.
//! 3. [`cardinality::estimate`] — per-operator cardinality estimate
//! walked over a [`LogicalPlan`], returning a parallel
//! [`cardinality::Cardinality`] tree.
//!
//! It only consumes the structures `PropertyColumnStats` and
//! `DegreeHistogram` that already live in every `SstDescriptor`
//!. The writer-side that populates HLL sketches is a
//! separate follow-up (see RFC-010 §"Drawbacks").
//!
//! [`Manifest`]: namidb_storage::Manifest
//! [`LogicalPlan`]: crate::plan::LogicalPlan

pub mod cardinality;
pub mod selectivity;
pub mod stats;

pub use cardinality::{estimate, BindingMeta, Cardinality};
pub use selectivity::selectivity;
pub use stats::{EdgeTypeStats, LabelStats, PropStats, StatsCatalog};
