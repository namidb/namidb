//! Naïve tree-walking executor for the milestone.
//!
//! Consumes a [`LogicalPlan`] and a [`Snapshot`], producing all matching
//! rows eagerly. Streaming and morsel-driven execution arrive.
//!
//! See [`docs/rfc/008-logical-plan-ir.md`](../../../../docs/rfc/008-logical-plan-ir.md)
//! §"API del executor".

pub mod auto_index;
pub mod expr;
pub mod factor;
pub mod fusion;
pub mod leapfrog;
pub mod limits;
pub mod memory;
pub mod path_count;
pub mod row;
pub mod show;
pub mod text_scoring;
pub mod value;
pub mod walker;
pub mod writer;

pub use auto_index::{with_auto_index_hints, AutoIndexHints};
pub use expr::{evaluate, EvalError, Params};
pub use factor::{
    factorize_enabled, FactorArena, FactorIdx, FactorNode, FactorRowSet, Slot, FACTOR_ROOT,
};
pub use leapfrog::{LeapfrogIntersect, MergeSortedUnion, OrdIterator, SortedSliceIter};
pub use memory::{
    current_query_memory, parse_unreclaimable_resident, process_resident_bytes,
    resident_memory_probe, system_memory_bytes, with_query_memory, MemoryProbe, QueryMemoryBudget,
    QueryMemoryExceeded, QueryMemoryPool, QueryMemoryScope,
};
pub use row::Row;
pub use show::{
    show_constraints_rows, show_indexes_columns, show_indexes_rows, show_schema_columns,
};
pub use value::{NodeValue, RelValue, RuntimeValue};
pub use walker::{
    cancel_filter_membership_carries, eligibility_cache_hits, execute, execute_factor_path,
    execute_flat_path, execute_with_limits, filter_membership_carries,
    filter_membership_carry_pending, filter_membership_ids_for_tests, filter_membership_paths,
    filter_membership_seek_budget, filter_membership_seek_budget_declines,
    filter_membership_seek_cache_hits, filter_membership_seek_ids_for_tests,
    filter_membership_seek_speculation, filter_membership_seeks, filter_membership_sst_key_reads,
    set_filter_membership_carry_for_tests, set_filter_membership_seek_cap_for_tests,
    set_filter_membership_seek_enabled, set_filter_membership_seek_population_divisor,
    set_vector_member_scan_enabled, vector_member_scans, ExecError, FilterMembershipCarries,
};
pub use writer::{
    enforce_node_unique_constraints, execute_write, execute_write_checked_with_deadline,
    execute_write_staged, execute_write_staged_with_deadline, execute_write_with_deadline,
    WriteOutcome,
};
