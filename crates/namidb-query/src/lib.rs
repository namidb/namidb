//! # namidb-query
//!
//! Cypher / GQL parser, cost-based optimizer and vectorized morsel-driven
//! executor.
//!
//! The crate is organised as a pipeline:
//!
//! 1. [`parser`] — Cypher source text → AST
//! 2. [`plan`] — logical plan IR + lowering (AST → `LogicalPlan`)
//! 3. [`optimize`] — cost-based optimizer (`LogicalPlan` → optimised `LogicalPlan`)
//! 4. [`exec`] — morsel-driven vectorized executor (`LogicalPlan` → result stream)
//!
//! See [`docs/rfc/004-cypher-subset.md`](../../../docs/rfc/004-cypher-subset.md)
//! for the v0 Cypher subset.

#![warn(rust_2018_idioms)]

pub mod cost;
pub mod exec;
pub mod optimize;
pub mod parser;
pub mod plan;

pub use cost::{
 estimate, BindingMeta, Cardinality, EdgeTypeStats, LabelStats, PropStats, StatsCatalog,
};
pub use exec::{
 evaluate, execute, execute_factor_path, execute_flat_path, execute_write, factorize_enabled,
 EvalError, ExecError, Params, Row, RuntimeValue, WriteOutcome,
};
pub use optimize::{convert_cross_to_hash, normalize_filters, optimize, predicate_pushdown};
pub use parser::{parse, ParseError, ParseResult, Query};
pub use plan::{
 explain, explain_query, explain_query_raw, explain_query_raw_verbose, explain_query_verbose,
 explain_verbose, lower, AggregateExpr, LogicalPlan, LowerError, LowerErrorKind,
};

/// Lower + optimize. The convenience entry point that the executor and
/// CLI use by default. Tests that need the raw lowering should call
/// [`lower`] directly. RFC-011 §1.
pub fn plan(query: &Query, catalog: &StatsCatalog) -> Result<LogicalPlan, LowerError> {
 Ok(optimize(lower(query)?, catalog))
}
