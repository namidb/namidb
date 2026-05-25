//! Naïve tree-walking executor for the milestone.
//!
//! Consumes a [`LogicalPlan`] and a [`Snapshot`], producing all matching
//! rows eagerly. Streaming and morsel-driven execution arrive.
//!
//! See [`docs/rfc/008-logical-plan-ir.md`](../../../../docs/rfc/008-logical-plan-ir.md)
//! §"API del executor".

pub mod expr;
pub mod factor;
pub mod leapfrog;
pub mod row;
pub mod value;
pub mod walker;
pub mod writer;

pub use expr::{evaluate, EvalError, Params};
pub use factor::{
    factorize_enabled, FactorArena, FactorIdx, FactorNode, FactorRowSet, Slot, FACTOR_ROOT,
};
pub use leapfrog::{LeapfrogIntersect, MergeSortedUnion, OrdIterator, SortedSliceIter};
pub use row::Row;
pub use value::{NodeValue, RelValue, RuntimeValue};
pub use walker::{execute, execute_factor_path, execute_flat_path, ExecError};
pub use writer::{execute_write, WriteOutcome};
