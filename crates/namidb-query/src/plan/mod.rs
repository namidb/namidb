//! Logical plan IR + lowering from the parsed AST.
//!
//! The pipeline is:
//!
//! ```text
//! ast::Query ──lower──▶ LogicalPlan ──explain/execute──▶ string / rows
//! ```
//!
//! See [`docs/rfc/008-logical-plan-ir.md`](../../../../docs/rfc/008-logical-plan-ir.md).

pub mod explain;
pub mod logical;
pub mod lower;

pub use explain::{
    collect_route_notes, explain, explain_plan_lines, explain_query, explain_query_raw,
    explain_query_raw_tree, explain_query_raw_tree_verbose, explain_query_raw_verbose,
    explain_query_tree, explain_query_tree_verbose, explain_query_verbose, explain_tree,
    explain_tree_verbose, explain_verbose, ExplainNode, RuntimeStats,
};
pub use logical::{AggregateExpr, LogicalPlan, OrderKey, ProjectionItem, RowCount, ShortestMode};
pub use lower::{lower, LowerError, LowerErrorKind};
