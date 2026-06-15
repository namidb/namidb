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
pub mod pagination;
pub mod parser;
pub mod plan;
pub mod plan_cache;
pub mod profile;

pub use pagination::{
    next_cursor, next_cursor_keyset, paginate_plan, paginate_plan_keyset, Cursor, CursorError,
    CursorKeyset,
};
pub use plan_cache::{parse_lower_optimize, query_text_hash, PlanError};
pub use profile::{profile_query_tree, ProfileError};

pub use cost::{
    estimate, BindingMeta, Cardinality, EdgeTypeStats, LabelStats, PropStats, StatsCatalog,
};
pub use exec::{
    enforce_node_unique_constraints, evaluate, execute, execute_factor_path, execute_flat_path,
    execute_with_limits, execute_write, execute_write_staged, execute_write_staged_with_deadline,
    execute_write_with_deadline, factorize_enabled, EvalError, ExecError, Params, Row,
    RuntimeValue, WriteOutcome,
};
pub use optimize::{convert_cross_to_hash, normalize_filters, optimize, predicate_pushdown};
pub use parser::{parse, ParseError, ParseResult, Query};
pub use plan::{
    explain, explain_query, explain_query_raw, explain_query_raw_tree,
    explain_query_raw_tree_verbose, explain_query_raw_verbose, explain_query_tree,
    explain_query_tree_verbose, explain_query_verbose, explain_tree, explain_tree_verbose,
    explain_verbose, lower, AggregateExpr, ExplainNode, LogicalPlan, LowerError, LowerErrorKind,
    RuntimeStats,
};

/// Lower + optimize. The convenience entry point that the executor and
/// CLI use by default. Tests that need the raw lowering should call
/// [`lower`] directly. RFC-011 §1.
pub fn plan(query: &Query, catalog: &StatsCatalog) -> Result<LogicalPlan, LowerError> {
    Ok(optimize(lower(query)?, catalog))
}
