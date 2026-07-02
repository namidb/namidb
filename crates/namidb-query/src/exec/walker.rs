//! Tree-walking executor.
//!
//! Eager `Vec<Row>` semantics. Single-threaded. No optimizer. Plugs into
//! `namidb_storage::Snapshot` for `lookup_node`, `out_edges`, `in_edges`
//! and `scan_label`.
//!
//! See RFC-008 §"API del executor".

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

use std::sync::Arc;

use futures::future::BoxFuture;
use futures::FutureExt;
use namidb_core::id::NodeId;
use namidb_storage::sst::predicates::eval_against_value;
use namidb_storage::{EdgeDirection, EdgeView, Snapshot};

use super::expr::{evaluate, order_for_sort, EvalError, Params};
use super::factor::{factorize_enabled, FactorArena, FactorIdx, FactorRowSet, Slot};
use super::leapfrog::{LeapfrogIntersect, SortedSliceIter};
use super::row::Row;
use super::value::{NodeValue, RelValue, RuntimeValue};
use crate::parser::{Expression, RelationshipDirection, SourceSpan};
use crate::plan::logical::{
    AggregateExpr, EdgeConstraint, LogicalPlan, NodeBinding, OrderKey, ProjectionItem, RowCount,
};

/// Top-level error produced by the executor. Wraps `EvalError`,
/// storage errors and structural runtime mismatches.
#[derive(Debug)]
pub enum ExecError {
    Eval(EvalError),
    Storage(namidb_storage::Error),
    Runtime(String),
    /// A declared schema constraint (e.g. a unique property) was violated by
    /// a write. Maps to `Neo.ClientError.Schema.ConstraintValidationFailed`.
    Constraint(String),
    /// A read query ran past its wall-clock deadline (the server's
    /// configured query timeout). Surfaced from the scan / expand loops and
    /// at operator boundaries; never raised when no deadline is in scope.
    Timeout,
    /// A read query tried to materialise more rows in one operator than the
    /// server's configured row cap allows. Carries the cap. Never raised
    /// when no cap is in scope.
    RowCap(usize),
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecError::Eval(e) => write!(f, "{}", e),
            ExecError::Storage(e) => write!(f, "storage: {}", e),
            ExecError::Runtime(m) => write!(f, "runtime: {}", m),
            ExecError::Constraint(m) => write!(f, "constraint violation: {}", m),
            ExecError::Timeout => write!(f, "query exceeded the configured timeout"),
            ExecError::RowCap(cap) => {
                write!(f, "query exceeded the configured row cap of {cap}")
            }
        }
    }
}

impl std::error::Error for ExecError {}

impl ExecError {
    /// `true` if this error is a deliberately-unsupported feature (unknown
    /// function, unimplemented expression form) rather than an internal bug.
    /// Transports use this to surface a typed "not supported" error instead
    /// of a bare 500 / generic storage/eval bucket.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            ExecError::Eval(e) if e.kind == super::expr::EvalErrorKind::Unsupported
        )
    }
}

impl From<EvalError> for ExecError {
    fn from(e: EvalError) -> Self {
        ExecError::Eval(e)
    }
}

impl From<namidb_storage::Error> for ExecError {
    fn from(e: namidb_storage::Error) -> Self {
        match e {
            // Storage raises its own Timeout when a query's deadline fires
            // mid-decode (cooperative cancellation); surface it as the
            // executor's Timeout, not a generic storage error.
            namidb_storage::Error::Timeout => ExecError::Timeout,
            other => ExecError::Storage(other),
        }
    }
}

/// Resolve a `SKIP` / `LIMIT` [`RowCount`] to a concrete `u64` at execution
/// time. A `$param` must bind to a non-negative integer; `what` names the
/// clause (`"SKIP"` / `"LIMIT"`) for the error message.
pub(crate) fn resolve_row_count(
    rc: &RowCount,
    params: &Params,
    what: &str,
) -> Result<u64, ExecError> {
    match rc {
        RowCount::Const(n) => Ok(*n),
        RowCount::Param(name) => match params.get(name) {
            Some(RuntimeValue::Integer(n)) if *n >= 0 => Ok(*n as u64),
            Some(RuntimeValue::Integer(_)) => Err(ExecError::Runtime(format!(
                "{what} parameter `${name}` must be non-negative"
            ))),
            Some(other) => Err(ExecError::Runtime(format!(
                "{what} parameter `${name}` must be an integer, got {}",
                other.type_name()
            ))),
            None => Err(ExecError::Runtime(format!(
                "{what} parameter `${name}` not provided"
            ))),
        },
    }
}

/// Execute `plan` against `snapshot`, returning all result rows.
///
/// Dispatches to the flat-row path or the factorised path depending on
/// [`factorize_enabled`] (env var `NAMIDB_FACTORIZE`). Both paths emit
/// the same row set; the factor path defers materialisation and avoids
/// the per-edge `BTreeMap` clone in multi-hop Expand chains (RFC-017).
pub async fn execute(
    plan: &LogicalPlan,
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    namidb_core::profile_scope!("walker::execute");
    if factorize_enabled() {
        execute_factor_path(plan, snapshot, params).await
    } else {
        execute_flat_path(plan, snapshot, params).await
    }
}

/// Like [`execute`], but bounded by an optional wall-clock `deadline` and
/// an optional `row_cap` (the maximum rows any single operator may
/// materialise).
///
/// The server derives both from its read query timeout and row cap. When
/// set, the executor returns [`ExecError::Timeout`] if a long scan / expand
/// or the operator dispatch crosses the deadline, or [`ExecError::RowCap`]
/// if an operator would exceed the cap. `None`/`None` is unbounded and
/// behaves exactly like [`execute`].
pub async fn execute_with_limits(
    plan: &LogicalPlan,
    snapshot: &Snapshot<'_>,
    params: &Params,
    deadline: Option<std::time::Instant>,
    row_cap: Option<usize>,
) -> Result<Vec<Row>, ExecError> {
    crate::exec::limits::with_limits(deadline, row_cap, execute(plan, snapshot, params)).await
}

/// Always-flat entry point — bypasses the `NAMIDB_FACTORIZE` env check.
/// Used by parity tests to compare the two paths side-by-side without
/// global env mutation.
pub async fn execute_flat_path(
    plan: &LogicalPlan,
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    let routing = PlanRouting::analyze(plan);
    execute_inner_with_routing(plan, snapshot, params, None, &routing).await
}

/// Always-factor entry point — bypasses the `NAMIDB_FACTORIZE` env check.
/// Executes the factor plan recursively and materialises every leaf into
/// a flat `Row` at the root sink. Used by parity tests.
pub async fn execute_factor_path(
    plan: &LogicalPlan,
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    let routing = PlanRouting::analyze(plan);
    let set = execute_factor_inner_with_routing(plan, snapshot, params, None, &routing).await?;
    let rows = set.materialize_all(None);
    crate::exec::limits::check_row_cap(rows.len())?;
    Ok(rows)
}

/// Public wrapper for callers outside `walker.rs` (e.g. `writer.rs`,
/// SemiApply subplan recursion below). Computes plan-aware routing for
/// the given subplan once and delegates. Recursive calls inside
/// `execute_inner_with_routing` reuse the parent's routing — see the
/// note in [`PlanRouting`].
pub(crate) fn execute_inner<'a>(
    plan: &'a LogicalPlan,
    snapshot: &'a Snapshot<'_>,
    params: &'a Params,
    outer: Option<&'a Row>,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    async move {
        let routing = PlanRouting::analyze(plan);
        execute_inner_with_routing(plan, snapshot, params, outer, &routing).await
    }
    .boxed()
}

pub(crate) fn execute_inner_with_routing<'a>(
    plan: &'a LogicalPlan,
    snapshot: &'a Snapshot<'_>,
    params: &'a Params,
    outer: Option<&'a Row>,
    routing: &'a PlanRouting,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    async move {
        // Deadline guard (read query timeout): one cheap check per operator
        // invocation, so a deeply recursive plan is bounded between operators
        // even before the long scan / expand loops below check it themselves.
        crate::exec::limits::check_deadline()?;
        // PROFILE hook: when the caller wrapped this `execute` in a
        // `ProfileCollector` scope, time every operator and stash the
        // result against its node pointer. The pointer is stable for
        // the duration of one `execute` because the plan is borrowed
        // for `'a` and not mutated; `profile_query_tree` consults the
        // same collector to attribute stats per operator.
        let profile_start = crate::profile::collector_present().then(std::time::Instant::now);
        let result = match plan {
            LogicalPlan::Empty => Ok(vec![Row::new()]),

            LogicalPlan::Argument { bindings } => {
                let outer = outer.ok_or_else(|| {
                    ExecError::Runtime(
 "Argument operator requires an outer row from a containing SemiApply / PatternList".into(),
 )
                })?;
                let mut row = Row::new();
                for name in bindings {
                    if let Some(v) = outer.get(name) {
                        row.set(name.clone(), v.clone());
                    }
                }
                Ok(vec![row])
            }

            LogicalPlan::SemiApply {
                input,
                subplan,
                negated,
            } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let sub_rows =
                        execute_inner_with_routing(subplan, snapshot, params, Some(&row), routing)
                            .await?;
                    let matched = !sub_rows.is_empty();
                    let keep = if *negated { !matched } else { matched };
                    if keep {
                        out.push(row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Apply { input, subplan } => {
                // Correlated lateral join: for each outer row, run the subplan
                // with that row as the correlation and emit the row combined
                // with each subplan row.
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let sub_rows =
                        execute_inner_with_routing(subplan, snapshot, params, Some(&row), routing)
                            .await?;
                    for s in sub_rows {
                        let mut merged = row.clone();
                        for (k, v) in &s.bindings {
                            merged.set(k.clone(), v.clone());
                        }
                        out.push(merged);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Create { .. }
            | LogicalPlan::Merge { .. }
            | LogicalPlan::Set { .. }
            | LogicalPlan::Remove { .. }
            | LogicalPlan::Delete { .. }
            | LogicalPlan::Foreach { .. } => Err(ExecError::Runtime(
                "write operators require execute_write(plan, &mut WriterSession, params)"
                    .to_string(),
            )),

            LogicalPlan::MultiwayJoin { .. } => Err(ExecError::Runtime(
                "MultiwayJoin requires the factorised executor; \
                 set NAMIDB_FACTORIZE=1 (RFC-024)"
                    .to_string(),
            )),

            LogicalPlan::VectorSearch {
                label,
                alias,
                property,
                query,
                k,
                distance,
                score_alias,
                post_filter,
            } => {
                flat_vector_search(
                    snapshot,
                    label.as_deref(),
                    alias,
                    property,
                    query,
                    k,
                    *distance,
                    score_alias,
                    post_filter.as_ref(),
                    params,
                )
                .await
            }

            LogicalPlan::CallProcedure {
                namespace,
                name,
                args,
                yield_items,
            } => {
                flat_call_procedure(
                    namespace.as_deref(),
                    name,
                    args,
                    yield_items,
                    snapshot,
                    params,
                )
                .await
            }

            LogicalPlan::PatternList {
                input,
                subplan,
                projection,
                alias,
            } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let inner_rows =
                        execute_inner_with_routing(subplan, snapshot, params, Some(&row), routing)
                            .await?;
                    let mut list = Vec::with_capacity(inner_rows.len());
                    for inner in inner_rows {
                        list.push(evaluate(projection, &inner, params)?);
                    }
                    let mut new_row = row;
                    new_row.set(alias.clone(), RuntimeValue::List(list));
                    out.push(new_row);
                }
                Ok(out)
            }

            LogicalPlan::NodeScan {
                label,
                alias,
                predicates,
                projection,
            } => {
                let labels = resolve_node_labels(snapshot, label.as_deref());
                let mut rows: Vec<Row> = Vec::new();
                for label_name in &labels {
                    let nodes = snapshot
                        .scan_label_with_predicates_and_projection(
                            label_name,
                            predicates,
                            projection.as_deref(),
                        )
                        .await?;
                    for n in nodes {
                        let value = RuntimeValue::Node(Box::new(NodeValue::from(n)));
                        rows.push(Row::new().with(alias.clone(), value));
                    }
                }
                Ok(rows)
            }

            LogicalPlan::NodeById {
                input,
                label,
                alias,
                id,
            } => {
                let input_rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::with_capacity(input_rows.len());
                for row in input_rows {
                    let id_value = evaluate(id, &row, params)?;
                    let node_id = node_id_from_value(&id_value, id.span)?;
                    let found = match label {
                        Some(l) => snapshot.lookup_node(l, node_id).await?,
                        None => scan_node_for_id(snapshot, node_id).await?,
                    };
                    if let Some(view) = found {
                        let mut new_row = row;
                        new_row.set(
                            alias.clone(),
                            RuntimeValue::Node(Box::new(NodeValue::from(view))),
                        );
                        out.push(new_row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::NodeByPropertyValue {
                input,
                label,
                alias,
                property,
                value,
                multi,
            } => {
                let input_rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::with_capacity(input_rows.len());
                for row in input_rows {
                    let lookup_val = evaluate(value, &row, params)?;
                    if *multi {
                        // Non-unique indexed property: fan out one row per
                        // matching node.
                        for view in lookup_nodes_by_property_via_scan(
                            snapshot,
                            label,
                            property,
                            &lookup_val,
                        )
                        .await?
                        {
                            let mut new_row = row.clone();
                            new_row.set(
                                alias.clone(),
                                RuntimeValue::Node(Box::new(NodeValue::from(view))),
                            );
                            out.push(new_row);
                        }
                    } else if let Some(view) =
                        lookup_node_by_property_via_scan(snapshot, label, property, &lookup_val)
                            .await?
                    {
                        let mut new_row = row;
                        new_row.set(
                            alias.clone(),
                            RuntimeValue::Node(Box::new(NodeValue::from(view))),
                        );
                        out.push(new_row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Filter { input, predicate } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let v = evaluate(predicate, &row, params)?;
                    if v.as_bool() == Some(true) {
                        out.push(row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Project {
                input,
                items,
                distinct,
                discard_input_bindings,
            } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let projected = project_rows(&rows, items, *discard_input_bindings, params)?;
                if *distinct {
                    Ok(dedup_rows(projected))
                } else {
                    Ok(projected)
                }
            }

            LogicalPlan::TopN {
                input,
                keys,
                skip,
                limit,
            } => {
                // Resolve `$param` SKIP/LIMIT against the bound params before
                // anything reads them numerically.
                let skip = resolve_row_count(skip, params, "SKIP")?;
                let limit = resolve_row_count(limit, params, "LIMIT")?;
                // LIMIT-pushdown: with no ORDER BY (keys empty) and a finite
                // limit, the child only needs its first `skip + limit` rows.
                // Run it under a row budget (`execute_capped`) so an
                // Expand/NodeScan can stop early instead of materialising its
                // full output before we truncate. Any plan shape the budget
                // can't safely cross falls back to full execution inside
                // `execute_capped`, so the worst case equals today's
                // behaviour. The sort/skip/take below are unchanged — they
                // still truncate the (possibly over-shooting) result exactly.
                let mut rows = if keys.is_empty() && limit != u64::MAX {
                    let cap = (skip as usize).saturating_add(limit as usize);
                    execute_capped(input, snapshot, params, outer, routing, cap).await?
                } else {
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?
                };
                if !keys.is_empty() {
                    // Bounded top-k when a finite LIMIT keeps fewer rows than the
                    // input has: O(n log k) heap instead of a full O(n log n)
                    // sort of every row (identical result — same stable order).
                    let bound = (skip as usize)
                        .saturating_add(if limit == u64::MAX { usize::MAX } else { limit as usize });
                    if bound != usize::MAX && bound < rows.len() {
                        rows = bounded_topk(rows, keys, params, bound)?;
                    } else {
                        sort_rows(&mut rows, keys, params)?;
                    }
                }
                let skip = skip as usize;
                if skip >= rows.len() {
                    return Ok(Vec::new());
                }
                let mut iter = rows.into_iter().skip(skip);
                let take = if limit == u64::MAX {
                    usize::MAX
                } else {
                    limit as usize
                };
                let mut out = Vec::with_capacity(take.min(64));
                for _ in 0..take {
                    match iter.next() {
                        Some(r) => out.push(r),
                        None => break,
                    }
                }
                Ok(out)
            }

            LogicalPlan::Distinct { input } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                Ok(dedup_rows(rows))
            }

            LogicalPlan::Union { left, right, all } => {
                let mut l =
                    execute_inner_with_routing(left, snapshot, params, outer, routing).await?;
                let r = execute_inner_with_routing(right, snapshot, params, outer, routing).await?;
                l.extend(r);
                if *all {
                    Ok(l)
                } else {
                    Ok(dedup_rows(l))
                }
            }

            LogicalPlan::CrossProduct { left, right } => {
                let l = execute_inner_with_routing(left, snapshot, params, outer, routing).await?;
                let r = execute_inner_with_routing(right, snapshot, params, outer, routing).await?;
                // Pre-check the multiplicative size before building the
                // product Vec, so a runaway cross product aborts instead of
                // allocating `l.len() * r.len()` rows first.
                crate::exec::limits::check_row_cap(l.len().saturating_mul(r.len()))?;
                Ok(cross_product(l, r))
            }

            LogicalPlan::HashJoin {
                build,
                probe,
                on,
                residual,
            } => {
                // Build phase: materialise hash table over build-side keys.
                // We use the existing row fingerprint machinery as the
                // key — it is the same canonical form used by Distinct,
                // so semantics match Cypher 3VL elsewhere.
                let build_rows =
                    execute_inner_with_routing(build, snapshot, params, outer, routing).await?;
                let mut table: std::collections::HashMap<Vec<String>, Vec<Row>> =
                    std::collections::HashMap::new();
                for row in build_rows {
                    let mut key = Vec::with_capacity(on.len());
                    let mut has_null = false;
                    for jk in on {
                        let v = evaluate(&jk.build_side, &row, params)?;
                        if matches!(v, RuntimeValue::Null) {
                            has_null = true;
                            break;
                        }
                        key.push(fingerprint_value(&v));
                    }
                    if has_null {
                        continue; // NULL key never matches (3VL).
                    }
                    table.entry(key).or_default().push(row);
                }
                // Probe phase: stream probe-side, look up each row, emit
                // joined tuples passing residual.
                let probe_rows =
                    execute_inner_with_routing(probe, snapshot, params, outer, routing).await?;
                let mut out = Vec::new();
                for prow in probe_rows {
                    let mut key = Vec::with_capacity(on.len());
                    let mut has_null = false;
                    for jk in on {
                        let v = evaluate(&jk.probe_side, &prow, params)?;
                        if matches!(v, RuntimeValue::Null) {
                            has_null = true;
                            break;
                        }
                        key.push(fingerprint_value(&v));
                    }
                    if has_null {
                        continue;
                    }
                    if let Some(matches) = table.get(&key) {
                        for brow in matches {
                            let mut combined = brow.clone();
                            for (k, v) in &prow.bindings {
                                combined.bindings.insert(k.clone(), v.clone());
                            }
                            if let Some(res) = residual {
                                // False / NULL silently drop the joined row.
                                if let RuntimeValue::Bool(true) = evaluate(res, &combined, params)?
                                {
                                    out.push(combined);
                                }
                            } else {
                                out.push(combined);
                            }
                        }
                    }
                }
                Ok(out)
            }

            LogicalPlan::HashSemiJoin {
                outer: outer_plan,
                inner: inner_plan,
                on,
                negated,
                residual,
            } => {
                // Build phase: execute inner ONCE (no outer correlation),
                // hash each row by the JoinKey::build_side fingerprint.
                let inner_rows =
                    execute_inner_with_routing(inner_plan, snapshot, params, outer, routing)
                        .await?;
                let mut key_set: std::collections::HashSet<Vec<String>> =
                    std::collections::HashSet::new();
                let mut residual_index: std::collections::HashMap<Vec<String>, Vec<Row>> =
                    std::collections::HashMap::new();
                for row in inner_rows {
                    let mut key = Vec::with_capacity(on.len());
                    let mut has_null = false;
                    for jk in on {
                        let v = evaluate(&jk.build_side, &row, params)?;
                        if matches!(v, RuntimeValue::Null) {
                            has_null = true;
                            break;
                        }
                        key.push(fingerprint_value(&v));
                    }
                    if has_null {
                        continue;
                    }
                    if residual.is_some() {
                        residual_index.entry(key.clone()).or_default().push(row);
                    }
                    key_set.insert(key);
                }

                // Probe phase: stream outer, lookup, emit per (matched,
                // negated) truth table.
                let outer_rows =
                    execute_inner_with_routing(outer_plan, snapshot, params, outer, routing)
                        .await?;
                let mut out = Vec::with_capacity(outer_rows.len());
                for orow in outer_rows {
                    let mut key = Vec::with_capacity(on.len());
                    let mut has_null = false;
                    for jk in on {
                        let v = evaluate(&jk.probe_side, &orow, params)?;
                        if matches!(v, RuntimeValue::Null) {
                            has_null = true;
                            break;
                        }
                        key.push(fingerprint_value(&v));
                    }
                    let matched = if has_null {
                        false
                    } else if let Some(res) = residual {
                        // Residual semantics: at least one inner row whose
                        // residual evaluates to true.
                        match residual_index.get(&key) {
                            Some(inner_rows) => {
                                let mut any = false;
                                for irow in inner_rows {
                                    let mut combined = irow.clone();
                                    for (k, v) in &orow.bindings {
                                        combined.bindings.insert(k.clone(), v.clone());
                                    }
                                    if let RuntimeValue::Bool(true) =
                                        evaluate(res, &combined, params)?
                                    {
                                        any = true;
                                        break;
                                    }
                                }
                                any
                            }
                            None => false,
                        }
                    } else {
                        key_set.contains(&key)
                    };
                    let keep = if *negated { !matched } else { matched };
                    if keep {
                        out.push(orow);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Unwind { input, list, alias } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::new();
                for row in rows {
                    let v = evaluate(list, &row, params)?;
                    match v {
                        RuntimeValue::List(items) => {
                            for item in items {
                                let mut new_row = row.clone();
                                new_row.set(alias.clone(), item);
                                out.push(new_row);
                            }
                        }
                        RuntimeValue::Null => {} // empty unwind
                        _ => {
                            return Err(ExecError::Runtime(format!(
                                "UNWIND requires a list; got {}",
                                v.type_name()
                            )))
                        }
                    }
                }
                Ok(out)
            }

            LogicalPlan::Expand {
                input,
                source,
                edge_type,
                direction,
                rel_alias,
                target_alias,
                target_labels,
                length,
                optional,
                back_reference,
                shortest,
                path_binding,
            } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let length = resolve_length(length, params)?;
                execute_expand(
                    rows,
                    source,
                    edge_type.as_deref(),
                    *direction,
                    rel_alias.as_deref(),
                    target_alias,
                    target_labels,
                    length.clone(),
                    *optional,
                    *back_reference,
                    *shortest,
                    path_binding.as_deref(),
                    snapshot,
                    routing.needs_properties(rel_alias.as_deref()),
                    should_skip_target_materialize(
                        snapshot,
                        routing,
                        target_alias,
                        edge_type.as_deref(),
                        *direction,
                        target_labels,
                        length,
                        *back_reference,
                    ),
                    None,
                )
                .await
            }

            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregations,
            } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                execute_aggregate(rows, group_by, aggregations, params)
            }

            LogicalPlan::EdgeTypeCount { edge_types, output } => {
                // Sum the live edge count of each listed type. Every edge
                // belongs to exactly one type, so per-type counts are
                // disjoint — no cross-type dedup. This reads only the edge
                // index of each type, skipping the NodeScan + Expand the
                // `edge_count_pushdown` pass replaced.
                let mut total: i64 = 0;
                for et in edge_types {
                    total += snapshot.count_edge_type(et).await? as i64;
                }
                Ok(vec![
                    Row::new().with(output.clone(), RuntimeValue::Integer(total))
                ])
            }
        };
        // Row-cap guard: bound the rows any single operator hands up. This
        // covers every operator uniformly (NodeScan, Expand, CrossProduct,
        // HashJoin, Unwind, ...); the most explosive producers also fail
        // fast before fully materialising (see CrossProduct's pre-check and
        // the Expand accumulation loop).
        if let Ok(rows) = &result {
            crate::exec::limits::check_row_cap(rows.len())?;
        }
        if let Some(start) = profile_start {
            if let Ok(rows) = &result {
                crate::profile::record_op(plan, start.elapsed(), rows.len() as u64);
            }
        }
        result
    }
    .boxed()
}

/// Execute `plan` under an order-insensitive row budget `cap`, used ONLY
/// by the `TopN`-with-empty-keys path (a bare `LIMIT`/`SKIP`, no
/// `ORDER BY`). It honours the cap in the three operators where a prefix
/// of the output is a valid prefix of the full result:
///   * non-`DISTINCT` `Project` — 1:1, order-preserving → pass cap through;
///   * `Expand` — 1:N, order-preserving → run its own input UNCAPPED (a
///     zero-edge seed yields no rows and must not starve the budget) and
///     stop the expansion at a seed boundary once `out.len() >= cap`;
///   * `NodeScan` — leaf → stop after `cap` rows (counter is global across
///     labels; predicates are pre-applied so truncation is safe).
/// EVERY other operator drops, reorders, dedups, expands-by-data, or
/// aggregates rows, so a prefix would be wrong — the catch-all delegates
/// to the uncapped [`execute_inner_with_routing`], which makes
/// "worst case == identical to today" true by construction. The budget is
/// valid only because no order is imposed; it is dropped the instant any
/// order-imposing / cardinality-altering operator is crossed (the
/// catch-all enforces this — nothing not whitelisted keeps the cap).
fn execute_capped<'a>(
    plan: &'a LogicalPlan,
    snapshot: &'a Snapshot<'_>,
    params: &'a Params,
    outer: Option<&'a Row>,
    routing: &'a PlanRouting,
    cap: usize,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    async move {
        match plan {
            LogicalPlan::Project {
                input,
                items,
                distinct: false,
                discard_input_bindings,
            } => {
                let rows = execute_capped(input, snapshot, params, outer, routing, cap).await?;
                project_rows(&rows, items, *discard_input_bindings, params)
            }

            LogicalPlan::Expand {
                input,
                source,
                edge_type,
                direction,
                rel_alias,
                target_alias,
                target_labels,
                length,
                optional,
                back_reference,
                shortest,
                path_binding,
            } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let length = resolve_length(length, params)?;
                execute_expand(
                    rows,
                    source,
                    edge_type.as_deref(),
                    *direction,
                    rel_alias.as_deref(),
                    target_alias,
                    target_labels,
                    length.clone(),
                    *optional,
                    *back_reference,
                    *shortest,
                    path_binding.as_deref(),
                    snapshot,
                    routing.needs_properties(rel_alias.as_deref()),
                    should_skip_target_materialize(
                        snapshot,
                        routing,
                        target_alias,
                        edge_type.as_deref(),
                        *direction,
                        target_labels,
                        length,
                        *back_reference,
                    ),
                    Some(cap),
                )
                .await
            }

            LogicalPlan::NodeScan {
                label,
                alias,
                predicates,
                projection,
            } => {
                let labels = resolve_node_labels(snapshot, label.as_deref());
                let mut rows: Vec<Row> = Vec::new();
                'scan: for label_name in &labels {
                    let nodes = snapshot
                        .scan_label_with_predicates_and_projection(
                            label_name,
                            predicates,
                            projection.as_deref(),
                        )
                        .await?;
                    for n in nodes {
                        if rows.len() >= cap {
                            break 'scan;
                        }
                        let value = RuntimeValue::Node(Box::new(NodeValue::from(n)));
                        rows.push(Row::new().with(alias.clone(), value));
                    }
                }
                Ok(rows)
            }

            // Cap unsafe to push through this operator — run it in full.
            other => execute_inner_with_routing(other, snapshot, params, outer, routing).await,
        }
    }
    .boxed()
}

// ───────────────────────── Expand ────────────────────────────────────

/// Clamp an open-ended variable-length upper bound (`max == u32::MAX`, from a
/// `*` / `*N..` pattern) to the configured hop cap. A finite bound passes
/// through unchanged.
fn clamp_hop_max(max: u32) -> u32 {
    if max == u32::MAX {
        crate::parser::ast::UNBOUNDED_VAR_LENGTH_CAP
    } else {
        max
    }
}

/// Read a variable-length bound supplied as a query parameter (`*1..$n`) as a
/// non-negative hop count.
fn resolve_hop_param(name: &str, params: &Params) -> Result<u32, ExecError> {
    match params.get(name) {
        Some(RuntimeValue::Integer(n)) if *n >= 0 => Ok(*n as u32),
        Some(RuntimeValue::Integer(_)) => Err(ExecError::Runtime(format!(
            "variable-length bound `${name}` must be non-negative"
        ))),
        Some(other) => Err(ExecError::Runtime(format!(
            "variable-length bound `${name}` must be an integer, got {}",
            other.type_name()
        ))),
        None => Err(ExecError::Runtime(format!(
            "variable-length bound parameter `${name}` not provided"
        ))),
    }
}

/// Substitute any parameter bounds in `length` with their runtime values,
/// yielding a fixed range the expand executors can consume directly.
pub(crate) fn resolve_length(
    length: &Option<crate::parser::RelationshipLength>,
    params: &Params,
) -> Result<Option<crate::parser::RelationshipLength>, ExecError> {
    let Some(l) = length else { return Ok(None) };
    let min = match &l.min_param {
        Some(p) => resolve_hop_param(p, params)?,
        None => l.min,
    };
    let max = match &l.max_param {
        Some(p) => resolve_hop_param(p, params)?,
        None => l.max,
    };
    if max < min {
        return Err(ExecError::Runtime(format!(
            "variable-length range resolves to *{min}..{max} (max < min)"
        )));
    }
    Ok(Some(crate::parser::RelationshipLength::fixed(min, max)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_expand(
    rows: Vec<Row>,
    source: &str,
    edge_type: Option<&[String]>,
    direction: RelationshipDirection,
    rel_alias: Option<&str>,
    target_alias: &str,
    target_labels: &[String],
    length: Option<crate::parser::RelationshipLength>,
    optional: bool,
    back_reference: bool,
    shortest: crate::plan::ShortestMode,
    path_binding: Option<&str>,
    snapshot: &Snapshot<'_>,
    want_properties: bool,
    skip_target_materialize: bool,
    cap: Option<usize>,
) -> Result<Vec<Row>, ExecError> {
    namidb_core::profile_scope!("walker::execute_expand");
    let edge_types = resolve_edge_types(snapshot, edge_type);
    let min = length.as_ref().map(|l| l.min).unwrap_or(1);
    let max = clamp_hop_max(length.as_ref().map(|l| l.max).unwrap_or(1));

    let mut out = Vec::new();
    for row in rows {
        // Deadline + row-cap guards: a multi-seed (or variable-length)
        // expansion is the most expensive operator, so bound it at every
        // seed boundary and fail fast before `out` grows past the cap.
        crate::exec::limits::check_deadline()?;
        crate::exec::limits::check_row_cap(out.len())?;
        // LIMIT-pushdown budget (set only via `execute_capped`, never on
        // the normal path). Checked at the seed boundary BEFORE processing
        // the next input row, so every consumed seed contributes its
        // COMPLETE edge set and `out` stays an order-preserving prefix of
        // the uncapped result. `cap == Some(0)` returns empty immediately.
        if let Some(cap) = cap {
            if out.len() >= cap {
                break;
            }
        }
        let starting = match row.get(source) {
            Some(RuntimeValue::Node(n)) => n.id,
            _ => {
                return Err(ExecError::Runtime(format!(
                    "Expand source `{}` is not a Node",
                    source
                )))
            }
        };

        // Back-reference: read the existing binding once. The
        // traversal explores the frontier freely; only paths whose
        // tail matches `existing_target_id` are kept as results.
        let existing_target_id: Option<NodeId> = if back_reference {
            match row.get(target_alias) {
                Some(RuntimeValue::Node(n)) => Some(n.id),
                Some(RuntimeValue::Null) => None,
                other => {
                    return Err(ExecError::Runtime(format!(
                        "Expand back-reference target `{}` is not a Node (got {:?})",
                        target_alias, other
                    )))
                }
            }
        } else {
            None
        };

        // Zero-length patterns (`*0..n`): the source node itself
        // counts as a valid match at hop 0. Emit it before stepping
        // out; downstream filters / target labels still apply.
        let mut hop_results: Vec<Row> = Vec::new();
        let mut matched_any = false;
        if min == 0 {
            let mut zero_row = row.clone();
            if !back_reference {
                // The target_alias must be bound for downstream
                // operators that read it. Materialise the source as
                // the target (graph-theoretic identity at hop 0).
                if let Some(RuntimeValue::Node(n)) = row.get(source) {
                    zero_row.set(target_alias.to_string(), RuntimeValue::Node(n.clone()));
                }
            }
            if let Some(name) = rel_alias {
                // No edge traversed at hop 0 → rel binding is NULL.
                zero_row.set(name, RuntimeValue::Null);
            }
            // At hop 0 the source node IS the far end, so the pattern's target
            // labels constrain it: `(a)-[:R*0..n]->(x:Label)` may bind the
            // source as `x` only if the source itself carries every target
            // label. Without this, `*0..` returned wrongly-labelled hop-0 rows
            // (the label check for hops ≥1 lives in the traversal, which hop 0
            // skips).
            let source_has_target_labels = match row.get(source) {
                Some(RuntimeValue::Node(n)) => {
                    target_labels.iter().all(|l| n.labels.contains(l))
                }
                _ => target_labels.is_empty(),
            };
            let zero_keeps = source_has_target_labels
                && match existing_target_id {
                    Some(existing) => starting == existing,
                    None => true,
                };
            if zero_keeps {
                hop_results.push(zero_row);
                matched_any = true;
            }
        }

        // Materialise the BFS trail only when shortestPath asked for
        // it. The head node opens the trail; each Expand hop appends
        // a Rel + a target Node so `RuntimeValue::Path` can be
        // assembled on the hit row (RFC-023).
        let materialise_trail = path_binding.is_some();
        let initial_trail = if materialise_trail {
            match row.get(source) {
                Some(RuntimeValue::Node(n)) => {
                    vec![RuntimeValue::Node(n.clone())]
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let mut frontier: Vec<Step> = vec![Step {
            tail: starting,
            row: row.clone(),
            trail: initial_trail,
            rels: Vec::new(),
        }];
        let hop_start = min.max(1);
        let _ = hop_start;
        for hop in 1..=max {
            let mut next_frontier = Vec::new();
            // Phase 1: pre-collect neighbours for every step so we can
            // batch-prewarm `Snapshot::lookup_node` once per hop (Fix #3b).
            // Without this, each (step, edge) pair issues its own
            // `lookup_node` SST decode — the dominant cost in cold IC09
            // (2 k+ uncached lookups × 4.2 ms each in the SF1 profile).
            let mut step_neighbours: Vec<(Step, Vec<EdgeView>)> =
                Vec::with_capacity(frontier.len());
            let mut unique_targets: Vec<NodeId> = Vec::new();
            let mut seen_targets: std::collections::HashSet<NodeId> =
                std::collections::HashSet::new();
            for step in frontier.drain(..) {
                let neighbours =
                    neighbours_of_any(snapshot, &edge_types, direction, step.tail, want_properties)
                        .await?;
                if !back_reference && !skip_target_materialize {
                    for edge in &neighbours {
                        let tid = partner_id(edge, direction, step.tail);
                        if seen_targets.insert(tid) {
                            unique_targets.push(tid);
                        }
                    }
                }
                step_neighbours.push((step, neighbours));
            }
            // Phase 2: batch prewarm. Populates L1 (and L2 if attached)
            // so the per-edge `lookup_node` below hits the cache instead
            // of decoding the SST again. We discard the returned `Vec`;
            // the cache is the only side-effect we care about.
            if !back_reference && !skip_target_materialize && !unique_targets.is_empty() {
                if let Some(label) = target_labels.first() {
                    let _ = snapshot.batch_lookup_nodes(label, &unique_targets).await?;
                }
            }
            for (step, neighbours) in step_neighbours {
                for edge in neighbours {
                    let target_id = partner_id(&edge, direction, step.tail);
                    // Cypher relationship uniqueness (trail semantics): a
                    // relationship may appear at most once per matched path.
                    // Skip an edge already traversed on this path so a
                    // variable-length pattern cannot walk the same edge back
                    // (e.g. `-[:R*2..2]-` over a single edge → a-r-b-r-a). Only
                    // enforced for multi-hop expansions; a single hop can never
                    // repeat a relationship. The identity is the STORED edge
                    // `(edge_type, src, dst)`, so a Both-direction traversal of
                    // the same edge in either orientation collapses to one key.
                    let edge_key = if max > 1 {
                        Some((edge.edge_type.clone(), edge.src, edge.dst))
                    } else {
                        None
                    };
                    if let Some(k) = &edge_key {
                        if step.rels.contains(k) {
                            continue;
                        }
                    }
                    // Back-reference fast path: skip the lookup_node
                    // (the binding's NodeView is already on the row).
                    // For non-back-reference, fetch the view so we
                    // can populate / label-filter.
                    // The far-end label(s) constrain which reached nodes are
                    // RESULTS — not which may be traversed THROUGH. For a
                    // multi-hop (`*`) expansion we therefore traverse every
                    // existing neighbour and let `target_is_result` gate the hit.
                    // Pruning the frontier on a label mismatch (the pre-fix bug)
                    // made `(s)-[:R*1..n]->(a:Label)` return empty whenever the
                    // intermediate nodes were not themselves `Label`.
                    let mut target_is_result = true;
                    let target_view_opt = if back_reference {
                        None
                    } else if skip_target_materialize {
                        // Fix #3: the binding is "transit only" — the next
                        // Expand reads only `.id`. Skip the SST decode and
                        // synthesise an id-only stub below. Schema-guaranteed
                        // dst_label means no correctness drift vs the
                        // `continue`-on-None branch below.
                        None
                    } else if let Some(label) = target_labels.first() {
                        if max > 1 {
                            // Multi-hop: traverse through any existing node; the
                            // far-end label gates only whether it is a result.
                            match scan_node_for_id(snapshot, target_id).await? {
                                Some(v) => {
                                    target_is_result =
                                        target_labels.iter().all(|l| v.labels.contains(l));
                                    Some(v)
                                }
                                None => continue,
                            }
                        } else {
                            // Single hop: the target IS the result, so a label
                            // mismatch excludes the edge (no traversal beyond it).
                            // Conjunctive multi-label: must carry EVERY label.
                            match snapshot.lookup_node(label, target_id).await? {
                                Some(v) if target_labels.iter().all(|l| v.labels.contains(l)) => {
                                    Some(v)
                                }
                                _ => continue,
                            }
                        }
                    } else {
                        match scan_node_for_id(snapshot, target_id).await? {
                            Some(v) => Some(v),
                            None => continue,
                        }
                    };
                    let rel_value = RuntimeValue::Rel(Box::new(RelValue::from(edge)));
                    let mut new_row = step.row.clone();
                    if let Some(name) = rel_alias {
                        new_row.set(name, rel_value.clone());
                    }
                    // For shortestPath trail materialisation we need a
                    // target NodeValue regardless of `skip_target_materialize`.
                    // Compute it once below and reuse for both the row binding
                    // and the trail.
                    let target_node_value: Option<NodeValue> =
                        if let Some(view) = target_view_opt.as_ref() {
                            Some(NodeValue::from(view.clone()))
                        } else if back_reference {
                            // Back-reference uses the pre-bound NodeView from
                            // the existing target_alias on the seed row.
                            match row.get(target_alias) {
                                Some(RuntimeValue::Node(n)) => Some(n.as_ref().clone()),
                                _ => None,
                            }
                        } else if skip_target_materialize {
                            Some(NodeValue {
                                id: target_id,
                                labels: target_labels.iter().map(|l| l.to_string()).collect(),
                                properties: std::collections::BTreeMap::new(),
                            })
                        } else {
                            None
                        };

                    if let Some(view) = target_view_opt {
                        new_row.set(
                            target_alias.to_string(),
                            RuntimeValue::Node(Box::new(NodeValue::from(view))),
                        );
                    } else if skip_target_materialize && !back_reference {
                        // id-only stub: enough for the next Expand to read
                        // `.id`; `label` is preserved so RuntimeValue::Node
                        // still type-checks for downstream Expand source reads.
                        if let Some(nv) = &target_node_value {
                            new_row.set(
                                target_alias.to_string(),
                                RuntimeValue::Node(Box::new(nv.clone())),
                            );
                        }
                    }
                    // Back-reference: the binding stays at the
                    // original existing target; new_row already
                    // carries it from row.clone() above.
                    let mut new_trail = step.trail.clone();
                    if materialise_trail {
                        new_trail.push(rel_value);
                        if let Some(nv) = target_node_value {
                            new_trail.push(RuntimeValue::Node(Box::new(nv)));
                        } else {
                            new_trail.push(RuntimeValue::Null);
                        }
                    }
                    let mut new_rels = step.rels.clone();
                    if let Some(k) = &edge_key {
                        new_rels.push(k.clone());
                    }
                    next_frontier.push(Step {
                        tail: target_id,
                        row: new_row.clone(),
                        trail: new_trail.clone(),
                        rels: new_rels,
                    });
                    if hop >= min.max(1) {
                        let keeps = target_is_result
                            && match existing_target_id {
                                Some(existing) => target_id == existing,
                                None => true,
                            };
                        if keeps {
                            let mut hit_row = new_row;
                            if let Some(name) = path_binding {
                                hit_row.set(name.to_string(), RuntimeValue::Path(new_trail));
                            }
                            hop_results.push(hit_row);
                            matched_any = true;
                            // shortestPath: at most one row per
                            // (source, target). Stop the whole BFS
                            // for this seed row.
                            if shortest == crate::plan::ShortestMode::First {
                                break;
                            }
                        }
                    }
                }
                if shortest == crate::plan::ShortestMode::First && matched_any {
                    break;
                }
            }
            // shortestPath: hit found this hop → don't extend the
            // frontier into hop+1.
            // allShortestPaths: hit found this hop → emit every
            // row of this length (already done above), then stop.
            if matched_any && shortest != crate::plan::ShortestMode::None {
                break;
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        if matched_any {
            out.append(&mut hop_results);
        } else if optional {
            let mut empty = row.clone();
            if let Some(name) = rel_alias {
                empty.set(name, RuntimeValue::Null);
            }
            if !back_reference {
                empty.set(target_alias.to_string(), RuntimeValue::Null);
            }
            out.push(empty);
        }
    }
    Ok(out)
}

struct Step {
    tail: NodeId,
    row: Row,
    /// Path materialisation trail for RFC-023 `shortestPath`. Empty
    /// when the Expand doesn't need to materialise a path (the
    /// common case). The trail alternates Node, Rel, Node, Rel, ...
    /// — for shortestPath we only fill it when `path_binding` is
    /// `Some(_)`.
    trail: Vec<RuntimeValue>,
    /// Relationships already traversed on THIS path, as stored edge
    /// identities `(edge_type, src, dst)`. Enforces Cypher relationship
    /// uniqueness (trail semantics): a variable-length path may not reuse a
    /// relationship, so `-[:R*2..2]-` cannot walk one edge out and back. Only
    /// populated for multi-hop expansions (`max > 1`); left empty on the
    /// single-hop hot path, where reuse is impossible.
    rels: Vec<(String, NodeId, NodeId)>,
}

/// Resolve the set of labels a `NodeScan` operator must visit.
///
/// `Some(l)` → scan only label `l`. `None` (pattern wrote `MATCH (n)`
/// without a label predicate) → enumerate every label observable through
/// the snapshot (`Snapshot::observed_labels`). Cost grows linearly with
/// the observed label count; the existing label-by-label `scan_label`
/// path is reused so per-label predicates and projections still apply.
fn resolve_node_labels(snapshot: &Snapshot<'_>, label: Option<&str>) -> Vec<String> {
    match label {
        Some(l) => vec![l.to_string()],
        None => snapshot.observed_labels(),
    }
}

/// Flat (no-index) top-k vector search over `label`'s `property` embeddings
/// (RFC-030). Scans every node of `label` (or all labels when `None`), scoring
/// each against `query`, and emits the `k` best as rows binding `alias` to the
/// node and `score_alias` to the metric value. This is the universal fallback
/// for [`LogicalPlan::VectorSearch`]; the `vector-index` feature adds an
/// index-backed path (Step 8) that calls this only when no descriptor matches.
///
/// The scan projects only the embedding column, so a large label costs one
/// decoded column per node rather than the whole property map. Candidates with
/// `CALL algo.<name>() [YIELD …]` — run a built-in graph procedure over the
/// full snapshot and emit one row per result record (RFC-008 PR1).
///
/// The kernels in `namidb_graph::algo` operate on an in-memory `Graph`; this
/// builds that graph from the snapshot — every node via `scan_label` (so
/// isolated nodes keep their own component / get a score) and every edge via
/// `scan_edge_type` (which carries properties, for edge weights) — runs the
/// kernel, then projects the canonical output columns to the YIELD bindings
/// (or the canonical names when `YIELD` was omitted).
async fn flat_call_procedure(
    namespace: Option<&str>,
    name: &str,
    args: &[Expression],
    yield_items: &[(String, String)],
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    // The `search` namespace holds text-retrieval procedures, which scan a
    // label's text property rather than the edge graph. Dispatch them before
    // the (potentially expensive) algo-graph build below.
    if namespace == Some("search") {
        return flat_search_procedure(name, args, yield_items, snapshot, params).await;
    }
    // Neo4j-compatible `CALL db.index.vector.queryNodes(indexName, k, queryVector)`.
    if namespace == Some("db.index.vector") {
        return db_index_vector_procedure(name, args, yield_items, snapshot, params).await;
    }
    if !matches!(namespace, Some("algo") | None) {
        return Err(proc_unsupported(format!(
            "unknown procedure namespace `{}` \
             (supported: `algo`, `search`, `db.index.vector`)",
            namespace.unwrap_or("")
        )));
    }

    let graph = snapshot_to_algo_graph(snapshot).await?;

    // Canonical output: column names + one RuntimeValue per column per row.
    let (cols, raw): (Vec<&'static str>, Vec<Vec<RuntimeValue>>) = match name {
        "wcc" => {
            if !args.is_empty() {
                return Err(proc_unsupported("algo.wcc takes no arguments"));
            }
            // Poll the query deadline mid-computation so a runaway CALL on a
            // huge graph is interruptible, not just at the operator boundary.
            let comps = namidb_graph::algo::weakly_connected_components_cancellable(
                &graph,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            let mut entries: Vec<(NodeId, usize)> = comps.assignment.into_iter().collect();
            // Deterministic order: by component id, then by node id.
            entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let raw = entries
                .into_iter()
                .map(|(id, c)| vec![node_runtime(id), RuntimeValue::Integer(c as i64)])
                .collect();
            (vec!["node_id", "component"], raw)
        }
        "pagerank" => {
            let opts = pagerank_options(args, params)?;
            let pr = namidb_graph::algo::pagerank_cancellable(
                &graph,
                &opts,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            let mut entries: Vec<(NodeId, f64)> = pr.scores.into_iter().collect();
            // Descending by score, then by node id for stability.
            entries.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            let raw = entries
                .into_iter()
                .map(|(id, s)| vec![node_runtime(id), RuntimeValue::Float(s)])
                .collect();
            (vec!["node_id", "score"], raw)
        }
        "degree" => {
            if !args.is_empty() {
                return Err(proc_unsupported("algo.degree takes no arguments"));
            }
            let deg = namidb_graph::algo::degrees_cancellable(
                &graph,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            let mut ids: Vec<NodeId> = graph.nodes().to_vec();
            // Deterministic: by total degree desc, then node id.
            ids.sort_by(|a, b| deg.total(b).cmp(&deg.total(a)).then_with(|| a.cmp(b)));
            let raw = ids
                .into_iter()
                .map(|id| {
                    vec![
                        node_runtime(id),
                        RuntimeValue::Integer(deg.in_degree.get(&id).copied().unwrap_or(0) as i64),
                        RuntimeValue::Integer(deg.out_degree.get(&id).copied().unwrap_or(0) as i64),
                        RuntimeValue::Integer(deg.total(&id) as i64),
                    ]
                })
                .collect();
            (vec!["node_id", "in_degree", "out_degree", "degree"], raw)
        }
        "scc" => {
            if !args.is_empty() {
                return Err(proc_unsupported("algo.scc takes no arguments"));
            }
            let comps = namidb_graph::algo::strongly_connected_components_cancellable(
                &graph,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            let mut entries: Vec<(NodeId, usize)> = comps.assignment.into_iter().collect();
            entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let raw = entries
                .into_iter()
                .map(|(id, c)| vec![node_runtime(id), RuntimeValue::Integer(c as i64)])
                .collect();
            (vec!["node_id", "component"], raw)
        }
        "triangle_count" => {
            if !args.is_empty() {
                return Err(proc_unsupported("algo.triangle_count takes no arguments"));
            }
            let tri = namidb_graph::algo::triangle_count_cancellable(
                &graph,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            let mut ids: Vec<NodeId> = graph.nodes().to_vec();
            // Deterministic: by triangle count desc, then node id.
            ids.sort_by(|a, b| {
                let ta = tri.per_node.get(a).copied().unwrap_or(0);
                let tb = tri.per_node.get(b).copied().unwrap_or(0);
                tb.cmp(&ta).then_with(|| a.cmp(b))
            });
            let raw = ids
                .into_iter()
                .map(|id| {
                    vec![
                        node_runtime(id),
                        RuntimeValue::Integer(tri.per_node.get(&id).copied().unwrap_or(0) as i64),
                        RuntimeValue::Float(tri.coefficient.get(&id).copied().unwrap_or(0.0)),
                    ]
                })
                .collect();
            (vec!["node_id", "triangles", "coefficient"], raw)
        }
        "label_propagation" => {
            let max_iters = label_propagation_options(args, params)?;
            let comm = namidb_graph::algo::label_propagation_cancellable(
                &graph,
                max_iters,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            let mut entries: Vec<(NodeId, usize)> = comm.assignment.into_iter().collect();
            entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let raw = entries
                .into_iter()
                .map(|(id, c)| vec![node_runtime(id), RuntimeValue::Integer(c as i64)])
                .collect();
            (vec!["node_id", "community"], raw)
        }
        "shortest_path" => {
            let (source, weighted) = shortest_path_options(args, params)?;
            // Dijkstra is unsound with negative weights; reject the query rather
            // than silently skipping those edges and returning wrong distances
            // (matches Neo4j GDS, which validates non-negative weights).
            if weighted && graph.has_negative_weight() {
                return Err(proc_unsupported(
                    "algo.shortest_path with weighted:true does not support negative edge weights",
                ));
            }
            let sp = namidb_graph::algo::shortest_paths_cancellable(
                &graph,
                source,
                weighted,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            let mut entries: Vec<(NodeId, f64)> = sp.distance.into_iter().collect();
            // Ascending by distance, then node id.
            entries.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            let raw = entries
                .into_iter()
                .map(|(id, dist)| {
                    vec![
                        node_runtime(id),
                        RuntimeValue::Float(dist),
                        RuntimeValue::Integer(sp.hops.get(&id).copied().unwrap_or(0) as i64),
                    ]
                })
                .collect();
            (vec!["node_id", "distance", "hops"], raw)
        }
        "fastRP" | "fast_rp" | "fastrp" => {
            let opts = fastrp_options(args, params)?;
            let result = namidb_graph::algo::fast_rp_cancellable(
                &graph,
                &opts,
                &namidb_storage::cancel::deadline_exceeded,
            )
            .map_err(|_| ExecError::Timeout)?;
            // Deterministic order by node id; the embedding is a FloatVector
            // ready to ingest straight into a vector index.
            let mut entries: Vec<(NodeId, Vec<f32>)> = result.embeddings.into_iter().collect();
            entries.sort_by_key(|e| e.0);
            let raw = entries
                .into_iter()
                .map(|(id, emb)| vec![node_runtime(id), RuntimeValue::Vector(emb)])
                .collect();
            (vec!["node_id", "embedding"], raw)
        }
        other => {
            return Err(proc_unsupported(format!(
                "unknown procedure `algo.{other}` (supported: algo.wcc, algo.scc, \
                 algo.pagerank, algo.degree, algo.triangle_count, \
                 algo.label_propagation, algo.shortest_path, algo.fastRP)"
            )));
        }
    };

    project_proc_rows(&format!("algo.{name}"), &cols, raw, yield_items)
}

/// Project a procedure's canonical columns (`cols`) and per-row values (`raw`)
/// onto the caller's YIELD bindings — or the canonical names when YIELD is
/// omitted. Shared by every `CALL` procedure. `qualified` is the full procedure
/// name (e.g. `algo.wcc`, `search.bm25`) for error messages.
fn project_proc_rows(
    qualified: &str,
    cols: &[&str],
    raw: Vec<Vec<RuntimeValue>>,
    yield_items: &[(String, String)],
) -> Result<Vec<Row>, ExecError> {
    let projection: Vec<(usize, String)> = if yield_items.is_empty() {
        cols.iter()
            .enumerate()
            .map(|(i, c)| (i, c.to_string()))
            .collect()
    } else {
        let mut out = Vec::with_capacity(yield_items.len());
        for (src, bind) in yield_items {
            match cols.iter().position(|c| *c == src.as_str()) {
                Some(i) => out.push((i, bind.clone())),
                None => {
                    return Err(proc_unsupported(format!(
                        "procedure `{qualified}` has no output column `{src}` \
                         (available: {})",
                        cols.join(", ")
                    )));
                }
            }
        }
        out
    };

    let mut rows = Vec::with_capacity(raw.len());
    for record in raw {
        let mut row = Row::new();
        for (i, bind) in &projection {
            row = row.with(bind.clone(), record[*i].clone());
        }
        rows.push(row);
    }
    Ok(rows)
}

fn proc_unsupported(msg: impl Into<String>) -> ExecError {
    ExecError::Eval(EvalError::unsupported(msg, SourceSpan::point(0)))
}

/// `CALL search.<name>(...)` — text-retrieval procedures over a label's text
/// property. Currently `search.bm25`.
async fn flat_search_procedure(
    name: &str,
    args: &[Expression],
    yield_items: &[(String, String)],
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    match name {
        "bm25" => bm25_search(args, yield_items, snapshot, params).await,
        "vector" => vector_search_procedure(args, yield_items, snapshot, params).await,
        "hybrid" => hybrid_search_procedure(args, yield_items, snapshot, params).await,
        other => Err(proc_unsupported(format!(
            "unknown procedure `search.{other}` \
             (supported: search.bm25, search.vector, search.hybrid)"
        ))),
    }
}

/// Full BM25 with real IDF (hybrid search Layer C). Scans `label`'s text
/// property/properties, builds corpus statistics (document count, average
/// length, per-query-term document frequency) in one pass, then scores every
/// candidate document with [`text_scoring::bm25_term_score`] and an IDF derived
/// from the corpus. Yields `node` (the matched node) + `score`, ordered by
/// score descending. Unlike the per-row `bm25()` scalar, this weights rare
/// terms above common ones.
///
/// `CALL search.bm25({label: 'Note', text_properties: ['body','title'],
/// query: $q, k: 10})`
async fn bm25_search(
    args: &[Expression],
    yield_items: &[(String, String)],
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    let (label, props, query, k) = bm25_search_args(args, params)?;
    let ranked = bm25_ranked(snapshot, &label, &props, &query, k).await?;

    // Hydrate each ranked id to its full node so `YIELD node` carries the
    // document's properties. A doc ranked but since deleted resolves to None.
    let mut raw = Vec::with_capacity(ranked.len());
    for (id, score) in ranked {
        if let Some(view) = snapshot.lookup_node(&label, id).await? {
            let node = RuntimeValue::Node(Box::new(NodeValue::from(view)));
            raw.push(vec![node, RuntimeValue::Float(score)]);
        }
    }
    project_proc_rows("search.bm25", &["node", "score"], raw, yield_items)
}

/// Full BM25 ranking as `(NodeId, score)` best-first (score desc, `NodeId` asc on
/// ties) — the retrieval core shared by `search.bm25` and the sparse leg of
/// `search.hybrid`. Serves from a registered full-text index when one covers
/// `(label, props)` and is authoritative for the corpus, else the exact flat
/// scan; either way the result is freshness-equivalent to the flat scan. `k`
/// caps the result (`None` = all). The empty-query case returns an empty vec.
async fn bm25_ranked(
    snapshot: &Snapshot<'_>,
    label: &str,
    props: &[String],
    query: &str,
    k: Option<usize>,
) -> Result<Vec<(NodeId, f64)>, ExecError> {
    use crate::exec::text_scoring::{bm25_idf, bm25_term_score, tokenize_counts};

    // Distinct query terms (a repeated query term is scored once), sorted for a
    // deterministic df/idf index order.
    let mut qterms: Vec<String> = tokenize_counts(query).0.into_keys().collect();
    qterms.sort();
    if qterms.is_empty() {
        return Ok(Vec::new());
    }

    // (`text-index`): serve from a registered full-text index that covers
    // (label, props) and is authoritative for the corpus; `text_search` returns
    // None (→ flat scan) when it is not, keeping fresh writes visible.
    #[cfg(feature = "text-index")]
    {
        let index_name = snapshot
            .manifest()
            .manifest
            .text_indexes
            .iter()
            .find(|d| d.matches(label, props))
            .map(|d| d.name.clone());
        if let Some(index_name) = index_name {
            if let Some(hits) = snapshot.text_search(&index_name, label, &qterms, k).await? {
                return Ok(hits);
            }
        }
    }

    let views = snapshot.scan_label(label).await?;

    // One pass: corpus stats over every document (a node with the text field)
    // and the per-query-term frequencies of the candidate documents.
    let mut n_docs = 0usize;
    let mut total_len = 0usize;
    let mut df = vec![0usize; qterms.len()];
    let mut candidates: Vec<(usize, Vec<u32>, usize)> = Vec::new(); // (view idx, tf per qterm, len)
    let mut since_check = 0usize;
    for (vi, view) in views.iter().enumerate() {
        let Some(text) = doc_text(view, props) else {
            continue; // not part of the searchable corpus
        };
        let (counts, len) = tokenize_counts(&text);
        n_docs += 1;
        total_len += len;
        let mut tfs = vec![0u32; qterms.len()];
        let mut any = false;
        for (i, qt) in qterms.iter().enumerate() {
            let tf = counts.get(qt).copied().unwrap_or(0);
            tfs[i] = tf;
            if tf > 0 {
                df[i] += 1;
                any = true;
            }
        }
        if any {
            candidates.push((vi, tfs, len));
        }
        since_check += 1;
        if since_check >= 4096 {
            since_check = 0;
            if namidb_storage::cancel::deadline_exceeded() {
                return Err(ExecError::Timeout);
            }
        }
    }

    let avg_len = if n_docs > 0 {
        total_len as f64 / n_docs as f64
    } else {
        1.0
    };
    let idf: Vec<f64> = df.iter().map(|&d| bm25_idf(n_docs, d)).collect();

    let mut scored: Vec<(usize, f64)> = candidates
        .into_iter()
        .map(|(vi, tfs, len)| {
            let s: f64 = tfs
                .iter()
                .enumerate()
                .map(|(i, &tf)| bm25_term_score(idf[i], tf, len, avg_len))
                .sum();
            (vi, s)
        })
        .collect();
    // Score descending, node id ascending for a deterministic tie-break.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| views[a.0].id.cmp(&views[b.0].id))
    });
    if let Some(k) = k {
        scored.truncate(k);
    }

    Ok(scored
        .into_iter()
        .map(|(vi, s)| (views[vi].id, s))
        .collect())
}

/// The text of a document for BM25: the configured properties' string values
/// joined by a space. `None` when the node carries none of them as a string —
/// such a node is not a member of the searchable corpus.
fn doc_text(view: &namidb_storage::NodeView, props: &[String]) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    for p in props {
        if let Some(namidb_core::Value::Str(s)) = view.properties.get(p) {
            parts.push(s.as_str());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Resolve `search.bm25` options from its single required map argument:
/// `{label, query, text_property | text_properties, k?}`. Returns
/// `(label, text_properties, query, k)`.
fn bm25_search_args(
    args: &[Expression],
    params: &Params,
) -> Result<(String, Vec<String>, String, Option<usize>), ExecError> {
    let map = match args {
        [arg] => match evaluate(arg, &Row::new(), params)? {
            RuntimeValue::Map(m) => m,
            _ => {
                return Err(proc_unsupported(
                    "search.bm25 expects a single map argument, e.g. \
                     {label: 'Note', text_property: 'body', query: $q}",
                ));
            }
        },
        _ => {
            return Err(proc_unsupported(
                "search.bm25 requires one map argument, e.g. \
                 {label: 'Note', text_property: 'body', query: $q}",
            ));
        }
    };

    let want_str = |v: &RuntimeValue| match v {
        RuntimeValue::String(s) => Some(s.clone()),
        _ => None,
    };

    let label = map
        .get("label")
        .and_then(want_str)
        .ok_or_else(|| proc_unsupported("search.bm25 requires a `label` string"))?;
    let query = map
        .get("query")
        .and_then(want_str)
        .ok_or_else(|| proc_unsupported("search.bm25 requires a `query` string"))?;

    // Either a single `text_property` or a list `text_properties`.
    let props: Vec<String> = match (map.get("text_properties"), map.get("text_property")) {
        (Some(RuntimeValue::List(items)), _) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match want_str(it) {
                    Some(s) => out.push(s),
                    None => {
                        return Err(proc_unsupported(
                            "search.bm25 `text_properties` must be a list of strings",
                        ));
                    }
                }
            }
            if out.is_empty() {
                return Err(proc_unsupported(
                    "search.bm25 `text_properties` must be non-empty",
                ));
            }
            out
        }
        (Some(_), _) => {
            return Err(proc_unsupported(
                "search.bm25 `text_properties` must be a list of strings",
            ));
        }
        (None, Some(v)) => match want_str(v) {
            Some(s) => vec![s],
            None => {
                return Err(proc_unsupported(
                    "search.bm25 `text_property` must be a string",
                ));
            }
        },
        (None, None) => {
            return Err(proc_unsupported(
                "search.bm25 requires `text_property` (string) or `text_properties` (list)",
            ));
        }
    };

    let k =
        match map.get("k") {
            None => None,
            Some(v) => Some(as_usize(v).ok_or_else(|| {
                proc_unsupported("search.bm25 `k` must be a non-negative integer")
            })?),
        };

    Ok((label, props, query, k))
}

/// Single required map argument for a procedure: `CALL proc({...})`.
fn proc_single_map(
    args: &[Expression],
    params: &Params,
    proc: &str,
) -> Result<std::collections::BTreeMap<String, RuntimeValue>, ExecError> {
    match args {
        [arg] => match evaluate(arg, &Row::new(), params)? {
            RuntimeValue::Map(m) => Ok(m),
            _ => Err(proc_unsupported(format!(
                "{proc} expects a single map argument"
            ))),
        },
        _ => Err(proc_unsupported(format!(
            "{proc} requires one map argument"
        ))),
    }
}

/// A map value as an owned `String`, or `None` if it is not a string.
fn proc_str(v: &RuntimeValue) -> Option<String> {
    match v {
        RuntimeValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Coerce a map value into an f32 query vector: a `vector()`/int8 vector, or a
/// Cypher list of numbers.
fn runtime_to_f32_vec(v: &RuntimeValue) -> Option<Vec<f32>> {
    match v {
        RuntimeValue::Vector(x) => Some(x.clone()),
        RuntimeValue::Vector8 { codes, scale } => {
            Some(codes.iter().map(|&c| c as f32 * *scale).collect())
        }
        RuntimeValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    RuntimeValue::Float(f) => out.push(*f as f32),
                    RuntimeValue::Integer(n) => out.push(*n as f32),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Parse an optional `metric` map value into a [`VectorDistance`] (default cosine).
fn proc_metric(
    v: Option<&RuntimeValue>,
) -> Result<crate::plan::logical::VectorDistance, ExecError> {
    use crate::plan::logical::VectorDistance;
    match v {
        None => Ok(VectorDistance::Cosine),
        Some(RuntimeValue::String(s)) => match s.to_ascii_lowercase().as_str() {
            "cosine" => Ok(VectorDistance::Cosine),
            "dot" | "dot_product" => Ok(VectorDistance::Dot),
            "euclidean" | "l2" => Ok(VectorDistance::Euclidean),
            other => Err(proc_unsupported(format!(
                "unknown metric `{other}` (cosine|dot|euclidean)"
            ))),
        },
        Some(_) => Err(proc_unsupported("`metric` must be a string")),
    }
}

/// Map storage `VectorMetric` (from a registered index descriptor) to the
/// engine's `VectorDistance`.
fn storage_metric_to_distance(
    m: namidb_storage::manifest::VectorMetric,
) -> crate::plan::logical::VectorDistance {
    use crate::plan::logical::VectorDistance;
    use namidb_storage::manifest::VectorMetric;
    match m {
        VectorMetric::Cosine => VectorDistance::Cosine,
        VectorMetric::Dot => VectorDistance::Dot,
        VectorMetric::Euclidean => VectorDistance::Euclidean,
    }
}

/// Extract `(NodeId, score)` from `vector_search_rows` / procedure rows (bound to
/// `node` + `score`), for fusion.
fn rows_to_id_score(rows: &[Row]) -> Vec<(NodeId, f64)> {
    rows.iter()
        .filter_map(|r| {
            let id = match r.get("node") {
                Some(RuntimeValue::Node(n)) => n.id,
                _ => return None,
            };
            let score = match r.get("score") {
                Some(RuntimeValue::Float(f)) => *f,
                Some(RuntimeValue::Integer(i)) => *i as f64,
                _ => return None,
            };
            Some((id, score))
        })
        .collect()
}

/// Wrap an [`ExpressionKind`](crate::parser::ast::ExpressionKind) into an
/// [`Expression`] at a synthetic span (the predicate is engine-built, not parsed).
fn mk_filter_expr(kind: crate::parser::ast::ExpressionKind, sp: SourceSpan) -> Expression {
    Expression { kind, span: sp }
}

/// `alias.key` property-access expression over the bound node binding.
fn proc_prop_access(alias: &str, key: &str, sp: SourceSpan) -> Expression {
    use crate::parser::ast::{ExpressionKind, Identifier, PropertyAccess};
    let target = mk_filter_expr(ExpressionKind::Variable(Identifier::new(alias, sp)), sp);
    mk_filter_expr(
        ExpressionKind::Property(Box::new(PropertyAccess {
            target,
            key: Identifier::new(key, sp),
            span: sp,
        })),
        sp,
    )
}

/// Convert a `filter` operand [`RuntimeValue`] into a literal/list [`Expression`].
fn proc_value_expr(v: &RuntimeValue, sp: SourceSpan) -> Result<Expression, ExecError> {
    use crate::parser::ast::{ExpressionKind, Literal};
    let kind = match v {
        // A null comparison is 3VL-undefined (never true), so `filter: {k: null}`
        // would silently match zero rows. The filter DSL has no IS NULL operator,
        // so reject null outright rather than produce a useless empty result.
        RuntimeValue::Null => {
            return Err(proc_unsupported(
                "`filter` values cannot be null (a null comparison matches no rows); \
                 omit the key instead",
            ))
        }
        RuntimeValue::Bool(b) => ExpressionKind::Literal(Literal::Boolean(*b)),
        RuntimeValue::Integer(n) => ExpressionKind::Literal(Literal::Integer(*n)),
        RuntimeValue::Float(f) => ExpressionKind::Literal(Literal::Float(*f)),
        RuntimeValue::String(s) => ExpressionKind::Literal(Literal::String(s.clone())),
        RuntimeValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(proc_value_expr(it, sp)?);
            }
            ExpressionKind::List(out)
        }
        _ => {
            return Err(proc_unsupported(
                "`filter` values must be scalars (string/number/bool) or lists of them",
            ))
        }
    };
    Ok(mk_filter_expr(kind, sp))
}

/// One `filter` term: a `{ op: value }` map → the matching comparison, else a
/// bare scalar → equality. Operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`.
fn proc_filter_term(
    alias: &str,
    key: &str,
    val: &RuntimeValue,
    sp: SourceSpan,
) -> Result<Expression, ExecError> {
    use crate::parser::ast::{BinaryOp, ExpressionKind};
    match val {
        // `{ op: value, … }` condition map → each operator AND-combined (so a
        // range is `{ gte: 10, lt: 100 }`). `in` works here too when the filter
        // arrives via a `$param` (runtime map keys are plain strings, not parsed).
        RuntimeValue::Map(opmap) => {
            if opmap.is_empty() {
                return Err(proc_unsupported(format!(
                    "`filter` `{key}`: a condition map needs at least one operator \
                     (eq|ne|gt|gte|lt|lte|in)"
                )));
            }
            let mut acc: Option<Expression> = None;
            for (op, operand) in opmap {
                let term = proc_filter_op(alias, key, op, operand, sp)?;
                acc = Some(match acc {
                    None => term,
                    Some(prev) => mk_filter_expr(
                        ExpressionKind::Binary {
                            op: BinaryOp::And,
                            left: Box::new(prev),
                            right: Box::new(term),
                        },
                        sp,
                    ),
                });
            }
            Ok(acc.expect("non-empty condition map"))
        }
        // A bare list → membership (`tenant_id: ['a', 'b']` ⇒ `n.tenant_id IN […]`).
        // The ergonomic inline spelling of `in`, which can't be a bare map key
        // (it is a reserved Cypher keyword that must be backtick-quoted).
        RuntimeValue::List(_) => {
            let list = proc_value_expr(val, sp)?;
            Ok(mk_filter_expr(
                ExpressionKind::In {
                    item: Box::new(proc_prop_access(alias, key, sp)),
                    list: Box::new(list),
                },
                sp,
            ))
        }
        // A bare scalar → equality.
        _ => {
            let rhs = proc_value_expr(val, sp)?;
            Ok(mk_filter_expr(
                ExpressionKind::Binary {
                    op: BinaryOp::Eq,
                    left: Box::new(proc_prop_access(alias, key, sp)),
                    right: Box::new(rhs),
                },
                sp,
            ))
        }
    }
}

/// One `prop op operand` predicate inside a condition map.
fn proc_filter_op(
    alias: &str,
    key: &str,
    op: &str,
    operand: &RuntimeValue,
    sp: SourceSpan,
) -> Result<Expression, ExecError> {
    use crate::parser::ast::{BinaryOp, ExpressionKind};
    let prop = proc_prop_access(alias, key, sp);
    if op.eq_ignore_ascii_case("in") {
        if !matches!(operand, RuntimeValue::List(_)) {
            return Err(proc_unsupported(format!(
                "`filter` `{key}.in` must be a list"
            )));
        }
        let list = proc_value_expr(operand, sp)?;
        return Ok(mk_filter_expr(
            ExpressionKind::In {
                item: Box::new(prop),
                list: Box::new(list),
            },
            sp,
        ));
    }
    let binop = match op.to_ascii_lowercase().as_str() {
        "eq" => BinaryOp::Eq,
        "ne" => BinaryOp::Ne,
        "gt" => BinaryOp::Gt,
        "gte" => BinaryOp::Ge,
        "lt" => BinaryOp::Lt,
        "lte" => BinaryOp::Le,
        other => {
            return Err(proc_unsupported(format!(
                "`filter` `{key}`: unknown operator `{other}` (eq|ne|gt|gte|lt|lte|in)"
            )))
        }
    };
    let rhs = proc_value_expr(operand, sp)?;
    Ok(mk_filter_expr(
        ExpressionKind::Binary {
            op: binop,
            left: Box::new(prop),
            right: Box::new(rhs),
        },
        sp,
    ))
}

/// Build a `post_filter` [`Expression`] from a procedure `filter` map argument so
/// the KNN procedures get the SAME index over-fetch + flat-scan fallback the
/// natural `MATCH … WHERE` form gets — instead of post-filtering an already
/// truncated top-k (which can starve a sparse tenant in a shared index to zero
/// results). Built against the bound node `alias` and handed to
/// [`vector_search_rows`] as its `post_filter`.
///
/// Shape (Qdrant-like; keys AND-combine):
/// - `{ tenant_id: "t1", status: "active" }` → `n.tenant_id = "t1" AND n.status = "active"`
/// - `{ tier: [1, 2, 3] }`                    → `n.tier IN [1, 2, 3]` (a list value ⇒ membership)
/// - `{ score: { gte: 0.5 }, region: { ne: "eu" } }` → `n.score >= 0.5 AND n.region <> "eu"`
///
/// A scalar value ⇒ equality, a list value ⇒ `IN`, a `{ op: value }` map ⇒ the
/// AND of its operators (`eq|ne|gt|gte|lt|lte|in`). The explicit `in` operator is
/// reachable only when the filter is supplied as a `$param` (a bare `in:` map key
/// can't be parsed inline — it is a reserved keyword); inline queries use the
/// list-value form.
fn proc_filter_expr(filter: &RuntimeValue, alias: &str) -> Result<Expression, ExecError> {
    use crate::parser::ast::{BinaryOp, ExpressionKind};
    let map = match filter {
        RuntimeValue::Map(m) => m,
        _ => {
            return Err(proc_unsupported(
                "`filter` must be a map of property → value or { op: value }",
            ))
        }
    };
    if map.is_empty() {
        return Err(proc_unsupported("`filter` must not be empty"));
    }
    let sp = SourceSpan::point(0);
    let mut conj: Option<Expression> = None;
    for (key, val) in map {
        let term = proc_filter_term(alias, key, val, sp)?;
        conj = Some(match conj {
            None => term,
            Some(acc) => mk_filter_expr(
                ExpressionKind::Binary {
                    op: BinaryOp::And,
                    left: Box::new(acc),
                    right: Box::new(term),
                },
                sp,
            ),
        });
    }
    Ok(conj.expect("a non-empty filter map yields at least one term"))
}

/// Parse an optional `filter` map value into a `post_filter` over the `node`
/// binding (shared by the vector / hybrid / queryNodes procedures).
fn proc_opt_filter(v: Option<&RuntimeValue>) -> Result<Option<Expression>, ExecError> {
    match v {
        None => Ok(None),
        Some(f) => Ok(Some(proc_filter_expr(f, "node")?)),
    }
}

/// `CALL search.vector({label, property, query, k?, ef?, metric?, filter?}) YIELD
/// node, score` — vector KNN served from the Vamana index (or the exact flat
/// scan), with a tunable `ef` beam width (recall vs latency) and an optional
/// `filter` (index-side over-fetch + exact fallback, not a post-truncation
/// filter). The ergonomic, EXPLAIN-free counterpart to the optimizer's KNN
/// rewrite.
async fn vector_search_procedure(
    args: &[Expression],
    yield_items: &[(String, String)],
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    let map = proc_single_map(args, params, "search.vector")?;
    let label = map
        .get("label")
        .and_then(proc_str)
        .ok_or_else(|| proc_unsupported("search.vector requires a `label` string"))?;
    let property = map
        .get("property")
        .and_then(proc_str)
        .ok_or_else(|| proc_unsupported("search.vector requires a `property` string"))?;
    let qv = map
        .get("query")
        .and_then(runtime_to_f32_vec)
        .ok_or_else(|| {
            proc_unsupported("search.vector requires a `query` vector (list or vector())")
        })?;
    let k = match map.get("k") {
        None => 10,
        Some(v) => as_usize(v)
            .ok_or_else(|| proc_unsupported("search.vector `k` must be a non-negative integer"))?,
    };
    let ef = match map.get("ef") {
        None => None,
        Some(v) => Some(as_usize(v).ok_or_else(|| {
            proc_unsupported("search.vector `ef` must be a non-negative integer")
        })?),
    };
    let distance = proc_metric(map.get("metric"))?;
    let pf = proc_opt_filter(map.get("filter"))?;

    let q = RuntimeValue::Vector(qv);
    let rows = vector_search_rows(
        snapshot,
        Some(&label),
        "node",
        &property,
        &q,
        SourceSpan::point(0),
        k,
        distance,
        "score",
        pf.as_ref(),
        ef,
        params,
    )
    .await?;
    let raw: Vec<Vec<RuntimeValue>> = rows
        .into_iter()
        .map(|r| {
            vec![
                r.get("node").cloned().unwrap_or(RuntimeValue::Null),
                r.get("score").cloned().unwrap_or(RuntimeValue::Null),
            ]
        })
        .collect();
    project_proc_rows("search.vector", &["node", "score"], raw, yield_items)
}

/// `CALL db.index.vector.queryNodes(indexName, k, queryVector [, {ef, filter}])
/// YIELD node, score` — Neo4j-compatible vector KNN. Resolves the index by NAME
/// (its descriptor supplies the label, property, and metric), then serves it
/// through the same path as `search.vector`. The optional 4th map may carry an
/// `ef` beam width and/or a `filter` (index over-fetch + exact fallback).
async fn db_index_vector_procedure(
    name: &str,
    args: &[Expression],
    yield_items: &[(String, String)],
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    if name != "queryNodes" {
        return Err(proc_unsupported(format!(
            "unknown procedure `db.index.vector.{name}` (supported: queryNodes)"
        )));
    }
    if args.len() < 3 {
        return Err(proc_unsupported(
            "db.index.vector.queryNodes(indexName, k, queryVector [, {ef: …, filter: {…}}])",
        ));
    }
    let index_name = match evaluate(&args[0], &Row::new(), params)? {
        RuntimeValue::String(s) => s,
        _ => return Err(proc_unsupported("queryNodes `indexName` must be a string")),
    };
    let k = as_usize(&evaluate(&args[1], &Row::new(), params)?)
        .ok_or_else(|| proc_unsupported("queryNodes `k` must be a non-negative integer"))?;
    let qv = runtime_to_f32_vec(&evaluate(&args[2], &Row::new(), params)?)
        .ok_or_else(|| proc_unsupported("queryNodes `queryVector` must be a list or vector()"))?;
    // The optional 4th map carries `ef` (beam width) and/or `filter` (a
    // post_filter compiled to the bound `node` binding → index over-fetch +
    // exact fallback, the same as the natural form).
    let (ef, pf) = match args.get(3) {
        None => (None, None),
        Some(a) => match evaluate(a, &Row::new(), params)? {
            RuntimeValue::Map(m) => {
                let ef = m.get("ef").and_then(as_usize);
                let pf = proc_opt_filter(m.get("filter"))?;
                (ef, pf)
            }
            _ => (None, None),
        },
    };

    // Resolve the index by name → (label, property, metric).
    let resolved = snapshot
        .manifest()
        .manifest
        .vector_indexes
        .iter()
        .find(|d| d.name == index_name)
        .map(|d| {
            (
                d.label.clone(),
                d.property.clone(),
                storage_metric_to_distance(d.metric),
            )
        });
    let (label, property, distance) = resolved
        .ok_or_else(|| proc_unsupported(format!("no vector index named `{index_name}`")))?;

    let q = RuntimeValue::Vector(qv);
    let rows = vector_search_rows(
        snapshot,
        Some(&label),
        "node",
        &property,
        &q,
        SourceSpan::point(0),
        k,
        distance,
        "score",
        pf.as_ref(),
        ef,
        params,
    )
    .await?;
    let raw: Vec<Vec<RuntimeValue>> = rows
        .into_iter()
        .map(|r| {
            vec![
                r.get("node").cloned().unwrap_or(RuntimeValue::Null),
                r.get("score").cloned().unwrap_or(RuntimeValue::Null),
            ]
        })
        .collect();
    project_proc_rows(
        "db.index.vector.queryNodes",
        &["node", "score"],
        raw,
        yield_items,
    )
}

/// Resolve the sparse leg's text properties from a `search.hybrid` map:
/// `text_properties` (non-empty list) or `text_property` (string).
fn hybrid_text_props(
    map: &std::collections::BTreeMap<String, RuntimeValue>,
) -> Result<Vec<String>, ExecError> {
    match (map.get("text_properties"), map.get("text_property")) {
        (Some(RuntimeValue::List(items)), _) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match proc_str(it) {
                    Some(s) => out.push(s),
                    None => {
                        return Err(proc_unsupported(
                            "search.hybrid `text_properties` must be a list of strings",
                        ))
                    }
                }
            }
            if out.is_empty() {
                return Err(proc_unsupported(
                    "search.hybrid `text_properties` must be non-empty",
                ));
            }
            Ok(out)
        }
        (Some(_), _) => Err(proc_unsupported(
            "search.hybrid `text_properties` must be a list of strings",
        )),
        (None, Some(v)) => proc_str(v)
            .map(|s| vec![s])
            .ok_or_else(|| proc_unsupported("search.hybrid `text_property` must be a string")),
        (None, None) => Err(proc_unsupported(
            "search.hybrid sparse leg requires `text_property` or `text_properties`",
        )),
    }
}

/// `CALL search.hybrid({label, query_text?, text_property(ies)?, query_vector?,
/// vector_property?, k?, ef?, fusion?, rrf_k?, alpha?, metric?, filter?}) YIELD node,
/// score` — fuse a dense (vector KNN) and a sparse (BM25) retrieval into one
/// ranking. Default fusion is Reciprocal Rank Fusion (rank-based, needs no score
/// calibration across the two scales); `fusion: 'linear'` does a weighted
/// min-max blend (`alpha` on the dense leg). Each leg independently serves from
/// its index or falls back to its exact flat scan, so hybrid is
/// freshness-equivalent to running the two separately and fusing. At least one
/// leg must be configured.
async fn hybrid_search_procedure(
    args: &[Expression],
    yield_items: &[(String, String)],
    snapshot: &Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    use crate::exec::fusion;

    let map = proc_single_map(args, params, "search.hybrid")?;
    let label = map
        .get("label")
        .and_then(proc_str)
        .ok_or_else(|| proc_unsupported("search.hybrid requires a `label` string"))?;
    let k = match map.get("k") {
        None => 10,
        Some(v) => as_usize(v)
            .ok_or_else(|| proc_unsupported("search.hybrid `k` must be a non-negative integer"))?,
    };
    let ef = match map.get("ef") {
        None => None,
        Some(v) => Some(as_usize(v).ok_or_else(|| {
            proc_unsupported("search.hybrid `ef` must be a non-negative integer")
        })?),
    };
    let distance = proc_metric(map.get("metric"))?;
    // Optional `filter`: applied to the dense leg as an index-side post_filter
    // (over-fetch + exact fallback) AND to the fused output (so a sparse-only
    // BM25 hit that fails the predicate is dropped too).
    let pf = proc_opt_filter(map.get("filter"))?;
    let fusion_mode = map
        .get("fusion")
        .and_then(proc_str)
        .unwrap_or_else(|| "rrf".to_string());
    // A present-but-wrong-type tuning value is an error, not a silent default.
    let opt_f64 = |key: &str| -> Result<Option<f64>, ExecError> {
        match map.get(key) {
            None => Ok(None),
            Some(v) => Ok(Some(as_f64(v).ok_or_else(|| {
                proc_unsupported(format!("search.hybrid `{key}` must be a number"))
            })?)),
        }
    };
    let opt_usize = |key: &str| -> Result<Option<usize>, ExecError> {
        match map.get(key) {
            None => Ok(None),
            Some(v) => Ok(Some(as_usize(v).ok_or_else(|| {
                proc_unsupported(format!(
                    "search.hybrid `{key}` must be a non-negative integer"
                ))
            })?)),
        }
    };
    let rrf_k = opt_f64("rrf_k")?.unwrap_or(fusion::DEFAULT_RRF_K);
    if rrf_k <= 0.0 {
        return Err(proc_unsupported("search.hybrid `rrf_k` must be > 0"));
    }
    let alpha = opt_f64("alpha")?.unwrap_or(0.5);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(proc_unsupported("search.hybrid `alpha` must be in [0, 1]"));
    }
    // Per-leg candidate depth before fusion: over-fetch so a node ranked well in
    // one leg but just outside the other's window still contributes.
    const OVERFETCH: usize = 8;
    let k_dense = opt_usize("k_dense")?.unwrap_or_else(|| k.saturating_mul(OVERFETCH).max(k));
    let k_sparse = opt_usize("k_sparse")?.unwrap_or_else(|| k.saturating_mul(OVERFETCH).max(k));

    // A dense leg needs BOTH keys; supplying exactly one is a mistake, not a
    // silently-disabled leg.
    let has_qvec = map.contains_key("query_vector");
    let has_vprop = map.contains_key("vector_property");
    if has_qvec != has_vprop {
        return Err(proc_unsupported(
            "search.hybrid dense leg needs both `query_vector` and `vector_property`",
        ));
    }
    let dense_configured = has_qvec && has_vprop;
    let sparse_configured = map.contains_key("query_text");
    if !dense_configured && !sparse_configured {
        return Err(proc_unsupported(
            "search.hybrid needs a dense leg (query_vector + vector_property) \
             and/or a sparse leg (query_text + text_property/properties)",
        ));
    }

    // Dense leg (vector KNN), best-first.
    let dense: Vec<(NodeId, f64)> = if dense_configured {
        let qv = map
            .get("query_vector")
            .and_then(runtime_to_f32_vec)
            .ok_or_else(|| {
                proc_unsupported("search.hybrid `query_vector` must be a list or vector()")
            })?;
        let vprop = map
            .get("vector_property")
            .and_then(proc_str)
            .ok_or_else(|| proc_unsupported("search.hybrid `vector_property` must be a string"))?;
        let q = RuntimeValue::Vector(qv);
        let rows = vector_search_rows(
            snapshot,
            Some(&label),
            "node",
            &vprop,
            &q,
            SourceSpan::point(0),
            k_dense,
            distance,
            "score",
            pf.as_ref(),
            ef,
            params,
        )
        .await?;
        rows_to_id_score(&rows)
    } else {
        Vec::new()
    };

    // Sparse leg (BM25), best-first.
    let sparse: Vec<(NodeId, f64)> = if sparse_configured {
        let qtext = map
            .get("query_text")
            .and_then(proc_str)
            .ok_or_else(|| proc_unsupported("search.hybrid `query_text` must be a string"))?;
        let props = hybrid_text_props(&map)?;
        // BM25 has no filter pushdown, so a residual `filter` is applied only at
        // the fused materialization below. Truncating the sparse leg to `k_sparse`
        // BEFORE that filter would starve a selective filter (the matching docs
        // can rank past `k_sparse` globally). When a filter is present, fetch a much
        // DEEPER ranking so the post-filter sees enough candidates — `k * 512`,
        // matching the dense leg's maximum widening depth (OVERFETCH_BASE ×
        // WIDEN_GROWTH^(MAX_WIDEN_ROUNDS-1) = 8 × 4³). This bounds the sparse leg,
        // the fusion structures, and the materialization probes to O(k·512) rather
        // than O(corpus) (avoiding a resource cliff on a common query term), while
        // still covering any filter that keeps ≳ 1/512 of the BM25-ranked docs.
        const SPARSE_FILTER_DEPTH: usize = 512;
        let sparse_k = if pf.is_some() {
            Some(k.saturating_mul(SPARSE_FILTER_DEPTH).max(k_sparse))
        } else {
            Some(k_sparse)
        };
        bm25_ranked(snapshot, &label, &props, &qtext, sparse_k).await?
    } else {
        Vec::new()
    };

    // RRF is rank-based, so the deeper filtered sparse leg does not change the
    // fused order (only the ranks of the surviving docs matter). `linear`
    // min-max-normalizes each leg over its own [worst, best] window, so a deeper
    // (filtered) sparse leg shifts the normalization baseline and can reorder the
    // blend — a known sensitivity of score-calibrated fusion. RRF (the default) is
    // the robust choice when a `filter` is present.
    let fused = match fusion_mode.to_ascii_lowercase().as_str() {
        "rrf" => fusion::rrf(&[&dense, &sparse], rrf_k),
        "linear" => fusion::linear(&[&dense, &sparse], &[alpha, 1.0 - alpha]),
        other => {
            return Err(proc_unsupported(format!(
                "unknown fusion `{other}` (rrf|linear)"
            )))
        }
    };

    // Materialise the fused candidates in rank order, applying the residual
    // `filter` (if any) BEFORE truncating, so the top-k is taken among the rows
    // that pass it (a node missing from the over-fetched window simply isn't
    // returned — fusion already over-fetched ×8 per leg).
    let mut raw = Vec::with_capacity(k.min(fused.len()));
    for (id, score) in fused.into_iter() {
        if raw.len() >= k {
            break;
        }
        crate::exec::limits::check_deadline()?;
        if let Some(view) = snapshot.lookup_node(&label, id).await? {
            let node_rv = RuntimeValue::Node(Box::new(NodeValue::from(view)));
            if let Some(pf) = pf.as_ref() {
                let mut row = Row::new();
                row.set("node".to_string(), node_rv.clone());
                row.set("score".to_string(), RuntimeValue::Float(score));
                if evaluate(pf, &row, params)?.as_bool() != Some(true) {
                    continue;
                }
            }
            raw.push(vec![node_rv, RuntimeValue::Float(score)]);
        }
    }
    project_proc_rows("search.hybrid", &["node", "score"], raw, yield_items)
}

/// A `node_id` YIELD value: a node carrying just its id (labels/properties
/// empty — the procedure output is about identity + score/component).
fn node_runtime(id: NodeId) -> RuntimeValue {
    RuntimeValue::Node(Box::new(NodeValue {
        id,
        labels: BTreeSet::new(),
        properties: std::collections::BTreeMap::new(),
    }))
}

/// Build an in-memory `algo::Graph` from the snapshot: every node (isolates
/// included) and every edge (with an optional `weight` property).
async fn snapshot_to_algo_graph(
    snapshot: &Snapshot<'_>,
) -> Result<namidb_graph::algo::Graph, ExecError> {
    let mut g = namidb_graph::algo::Graph::new();
    for label in snapshot.observed_labels() {
        // We only need the node ids, so scan with an EMPTY property projection:
        // this skips decoding every declared property column AND serde-parsing
        // __overflow_json per row — the dominant cost of building the algo graph
        // (millions of JSON parses on a large namespace, per CALL algo.*).
        for n in snapshot
            .scan_label_with_predicates_and_projection(&label, &[], Some(&[]))
            .await?
        {
            g.add_node(n.id);
        }
    }
    for et in snapshot.observed_edge_types() {
        for e in snapshot.scan_edge_type(&et).await? {
            let w = e.properties.get("weight").and_then(numeric_weight);
            g.add_edge(e.src, e.dst, w);
        }
    }
    Ok(g)
}

fn numeric_weight(v: &namidb_core::Value) -> Option<f64> {
    match v {
        namidb_core::Value::F64(x) => Some(*x),
        namidb_core::Value::I64(x) => Some(*x as f64),
        _ => None,
    }
}

/// Resolve `algo.pagerank` options from its optional single map argument:
/// `CALL algo.pagerank({damping: 0.9, max_iterations: 50, tolerance: 1e-6})`.
/// Omitted keys keep the engine defaults; wrong types are rejected.
fn pagerank_options(
    args: &[Expression],
    params: &Params,
) -> Result<namidb_graph::algo::PageRankOptions, ExecError> {
    let mut opts = namidb_graph::algo::PageRankOptions::default();
    match args {
        [] => {}
        [arg] => {
            let val = evaluate(arg, &Row::new(), params)?;
            let map = match val {
                RuntimeValue::Map(m) => m,
                _ => {
                    return Err(proc_unsupported(
                        "algo.pagerank expects a single map argument, e.g. {damping: 0.9}",
                    ));
                }
            };
            if let Some(d) = map.get("damping") {
                opts.damping = as_f64(d)
                    .ok_or_else(|| proc_unsupported("algo.pagerank `damping` must be a number"))?;
            }
            if let Some(m) = map.get("max_iterations") {
                opts.max_iterations = as_usize(m).ok_or_else(|| {
                    proc_unsupported(
                        "algo.pagerank `max_iterations` must be a non-negative integer",
                    )
                })?;
            }
            if let Some(t) = map.get("tolerance") {
                opts.tolerance = as_f64(t).ok_or_else(|| {
                    proc_unsupported("algo.pagerank `tolerance` must be a number")
                })?;
            }
        }
        _ => {
            return Err(proc_unsupported(
                "algo.pagerank takes at most one (map) argument",
            ));
        }
    }
    Ok(opts)
}

/// Resolve `algo.fastRP` options from an optional map:
/// `{dimension?, iterations?, iteration_weights?, normalization_strength?, seed?}`.
/// `iterations: k` sets the default `[0, 1, …, 1]` weights of length `k+1`; an
/// explicit `iteration_weights` list overrides that.
fn fastrp_options(
    args: &[Expression],
    params: &Params,
) -> Result<namidb_graph::algo::FastRpOptions, ExecError> {
    let mut opts = namidb_graph::algo::FastRpOptions::default();
    match args {
        [] => {}
        [arg] => {
            let map = match evaluate(arg, &Row::new(), params)? {
                RuntimeValue::Map(m) => m,
                _ => {
                    return Err(proc_unsupported(
                        "algo.fastRP expects a single map argument, e.g. {dimension: 256}",
                    ))
                }
            };
            if let Some(v) = map.get("dimension") {
                opts.dimension = as_usize(v).ok_or_else(|| {
                    proc_unsupported("algo.fastRP `dimension` must be a non-negative integer")
                })?;
            }
            if let Some(v) = map.get("normalization_strength") {
                opts.normalization_strength = as_f64(v).ok_or_else(|| {
                    proc_unsupported("algo.fastRP `normalization_strength` must be a number")
                })? as f32;
            }
            if let Some(v) = map.get("seed") {
                opts.seed = as_usize(v).ok_or_else(|| {
                    proc_unsupported("algo.fastRP `seed` must be a non-negative integer")
                })? as u64;
            }
            if let Some(v) = map.get("iterations") {
                let iters = as_usize(v).ok_or_else(|| {
                    proc_unsupported("algo.fastRP `iterations` must be a non-negative integer")
                })?;
                opts.iteration_weights = std::iter::once(0.0)
                    .chain(std::iter::repeat_n(1.0, iters))
                    .collect();
            }
            if let Some(v) = map.get("iteration_weights") {
                match v {
                    RuntimeValue::List(items) => {
                        let mut w = Vec::with_capacity(items.len());
                        for it in items {
                            w.push(as_f64(it).ok_or_else(|| {
                                proc_unsupported(
                                    "algo.fastRP `iteration_weights` must be a list of numbers",
                                )
                            })? as f32);
                        }
                        if w.is_empty() {
                            return Err(proc_unsupported(
                                "algo.fastRP `iteration_weights` must be non-empty",
                            ));
                        }
                        opts.iteration_weights = w;
                    }
                    _ => {
                        return Err(proc_unsupported(
                            "algo.fastRP `iteration_weights` must be a list of numbers",
                        ))
                    }
                }
            }
            if opts.dimension == 0 {
                return Err(proc_unsupported("algo.fastRP `dimension` must be >= 1"));
            }
        }
        _ => {
            return Err(proc_unsupported(
                "algo.fastRP takes at most one (map) argument",
            ));
        }
    }
    Ok(opts)
}

/// Resolve `algo.label_propagation` options from its optional single map arg:
/// `CALL algo.label_propagation({max_iterations: 20})`. Omitted → the engine
/// default iteration cap.
fn label_propagation_options(args: &[Expression], params: &Params) -> Result<usize, ExecError> {
    let mut max_iters = namidb_graph::algo::LABEL_PROPAGATION_DEFAULT_ITERS;
    match args {
        [] => {}
        [arg] => {
            let map = match evaluate(arg, &Row::new(), params)? {
                RuntimeValue::Map(m) => m,
                _ => {
                    return Err(proc_unsupported(
                        "algo.label_propagation expects a single map argument, e.g. {max_iterations: 20}",
                    ));
                }
            };
            if let Some(m) = map.get("max_iterations") {
                max_iters = as_usize(m).ok_or_else(|| {
                    proc_unsupported(
                        "algo.label_propagation `max_iterations` must be a non-negative integer",
                    )
                })?;
            }
        }
        _ => {
            return Err(proc_unsupported(
                "algo.label_propagation takes at most one (map) argument",
            ));
        }
    }
    Ok(max_iters)
}

/// Resolve `algo.shortest_path` options from its required single map arg:
/// `CALL algo.shortest_path({source: "<uuid>", weighted: true})`. `source` is
/// required (a node-id string or a node value); `weighted` defaults to false
/// (BFS hop count) and, when true, runs Dijkstra over non-negative `weight`s.
fn shortest_path_options(
    args: &[Expression],
    params: &Params,
) -> Result<(NodeId, bool), ExecError> {
    let map = match args {
        [arg] => match evaluate(arg, &Row::new(), params)? {
            RuntimeValue::Map(m) => m,
            _ => {
                return Err(proc_unsupported(
                    "algo.shortest_path expects a single map argument, e.g. {source: \"<uuid>\"}",
                ));
            }
        },
        _ => {
            return Err(proc_unsupported(
                "algo.shortest_path requires one map argument with a `source`, e.g. {source: \"<uuid>\"}",
            ));
        }
    };
    let source = match map.get("source") {
        Some(RuntimeValue::String(s)) => s.parse::<NodeId>().map_err(|_| {
            proc_unsupported(format!(
                "algo.shortest_path `source` is not a valid node id: {s}"
            ))
        })?,
        Some(RuntimeValue::Node(n)) => n.id,
        Some(_) => {
            return Err(proc_unsupported(
                "algo.shortest_path `source` must be a node-id string or a node",
            ));
        }
        None => {
            return Err(proc_unsupported(
                "algo.shortest_path requires a `source` (node-id string or node)",
            ));
        }
    };
    let weighted = match map.get("weighted") {
        None => false,
        Some(RuntimeValue::Bool(b)) => *b,
        Some(_) => {
            return Err(proc_unsupported(
                "algo.shortest_path `weighted` must be a boolean",
            ));
        }
    };
    Ok((source, weighted))
}

fn as_f64(v: &RuntimeValue) -> Option<f64> {
    match v {
        RuntimeValue::Float(x) => Some(*x),
        RuntimeValue::Integer(x) => Some(*x as f64),
        _ => None,
    }
}

fn as_usize(v: &RuntimeValue) -> Option<usize> {
    match v {
        RuntimeValue::Integer(x) => usize::try_from(*x).ok(),
        _ => None,
    }
}

/// a missing/NULL embedding or a zero-magnitude vector (undefined cosine) are
/// dropped.
#[allow(clippy::too_many_arguments)]
async fn flat_vector_search(
    snapshot: &Snapshot<'_>,
    label: Option<&str>,
    alias: &str,
    property: &str,
    query: &Expression,
    k: &RowCount,
    distance: crate::plan::logical::VectorDistance,
    score_alias: &str,
    post_filter: Option<&Expression>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    // The query expression carries no row bindings (literal vector or $param);
    // evaluate it once against an empty row.
    let q = evaluate(query, &Row::new(), params)?;
    // Resolve the LIMIT the same way TopN does: a parameterized `LIMIT $k` must
    // use the bound value, not a hardcoded default. The apply_vector_search
    // rewrite deletes the TopN that would otherwise have resolved the param, so
    // if this fell back to `unwrap_or(10)` a `LIMIT $k` silently returned 10
    // rows regardless of $k. Errors on a missing/invalid $k, exactly like TopN.
    let limit = resolve_row_count(k, params, "LIMIT")? as usize;
    // The natural/operator form has no syntax for the beam width, so it reads a
    // reserved, namespaced param `$__vector_ef` to tune recall vs latency on the
    // filtered-ANN path (the procedures take a first-class `ef`). It is the
    // recall/latency dial: a larger value widens the beam (more recall, more
    // latency), a smaller one narrows it — clamped downstream only to `≥ kprime`
    // (`e.max(kprime)`), so for a small `k` a value below the default 64 yields a
    // narrower-than-default beam (lower recall). It changes neither correctness of
    // a *selective* filter (one that under-fills `k` and forces the exact flat
    // fallback) nor a result the engine already calls exact; but for a
    // non-selective filter or a plain KNN, a narrower beam can return a different,
    // lower-recall approximate top-k — like any ANN beam knob. Absent, the index
    // default applies. NON-STABLE knob — superseded by a future `OPTIONS { ef }`
    // surface (RFC-036). Namespaced to avoid clashing with a user's own `$ef`.
    let ef_search = params.get("__vector_ef").and_then(as_usize);
    vector_search_rows(
        snapshot,
        label,
        alias,
        property,
        &q,
        query.span,
        limit,
        distance,
        score_alias,
        post_filter,
        ef_search,
        params,
    )
    .await
}

/// Core of the vector KNN: serve a pre-evaluated query vector `q` from the
/// Vamana index when one applies (freshness-equivalent to the flat scan), else
/// the exact flat scan. `ef_search` overrides the index beam width (`None` =
/// default). Shared by the `VectorSearch` operator and the `search.vector` /
/// `search.hybrid` procedures. Emits rows binding the node to `alias` and the
/// metric score to `score_alias`.
#[allow(clippy::too_many_arguments)]
async fn vector_search_rows(
    snapshot: &Snapshot<'_>,
    label: Option<&str>,
    alias: &str,
    property: &str,
    q: &RuntimeValue,
    span: SourceSpan,
    limit: usize,
    distance: crate::plan::logical::VectorDistance,
    score_alias: &str,
    post_filter: Option<&Expression>,
    ef_search: Option<usize>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    use crate::exec::expr::vector_score;

    // RFC-030 (`vector-index`): serve from the Vamana index when one exists for
    // (label, property, metric). Falls through to the flat scan otherwise (and
    // also when a residual filter is too selective for the index to satisfy).
    #[cfg(feature = "vector-index")]
    {
        if let Some(rows) = try_index_search(
            snapshot,
            label,
            alias,
            property,
            q,
            limit,
            distance,
            score_alias,
            post_filter,
            span,
            ef_search,
            params,
        )
        .await?
        {
            return Ok(rows);
        }
    }
    #[cfg(not(feature = "vector-index"))]
    let _ = ef_search;

    let labels = resolve_node_labels(snapshot, label);
    // Materialise the WHOLE node: the result binds it to `alias`, and a
    // downstream projection (`RETURN d.title`) or a procedure's `YIELD node` may
    // read any property — not just the embedding — and a `post_filter` may too.
    // (The index path likewise returns full nodes via `lookup_node`.) Projecting
    // only the embedding column here would leave those properties null.
    let projection: Option<Vec<String>> = None;

    // (sort_key, score_value, node) — sort_key is "lower is better" (higher-is-
    // better metrics are negated), so an ascending sort yields the top-k.
    let mut scored: Vec<(f64, f64, NodeValue)> = Vec::new();
    for label_name in &labels {
        let nodes = snapshot
            .scan_label_with_predicates_and_projection(label_name, &[], projection.as_deref())
            .await?;
        for n in nodes {
            crate::exec::limits::check_deadline()?;
            let node = NodeValue::from(n);
            let Some(emb) = node.properties.get(property) else {
                continue;
            };
            let Some((score, higher_is_better)) = vector_score(distance, emb, q, span)? else {
                continue;
            };
            let sort_key = if higher_is_better { -score } else { score };
            scored.push((sort_key, score, node));
        }
    }

    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    // Build rows in rank order; apply the residual filter (if any) BEFORE
    // truncating, so the top-k is taken among the rows that pass it.
    let mut out = Vec::with_capacity(limit);
    for (_sort_key, score, node) in scored {
        if out.len() >= limit {
            break;
        }
        let mut row = Row::new();
        row.set(alias.to_string(), RuntimeValue::Node(Box::new(node)));
        row.set(score_alias.to_string(), RuntimeValue::Float(score));
        if let Some(pf) = post_filter {
            if evaluate(pf, &row, params)?.as_bool() != Some(true) {
                continue;
            }
        }
        out.push(row);
    }
    Ok(out)
}

/// RFC-030 (`vector-index`): serve a `VectorSearch` from the Vamana index when
/// one is registered for `(label, property, metric)`. Returns `Ok(None)` when
/// no index applies (no label, euclidean metric, or no descriptor) so the
/// caller falls through to the flat scan.
#[cfg(feature = "vector-index")]
#[allow(clippy::too_many_arguments)]
async fn try_index_search(
    snapshot: &Snapshot<'_>,
    label: Option<&str>,
    alias: &str,
    property: &str,
    q: &RuntimeValue,
    k: usize,
    distance: crate::plan::logical::VectorDistance,
    score_alias: &str,
    post_filter: Option<&Expression>,
    span: SourceSpan,
    ef_search: Option<usize>,
    params: &Params,
) -> Result<Option<Vec<Row>>, ExecError> {
    use crate::exec::expr::vector_score;
    use crate::plan::logical::VectorDistance;
    use namidb_storage::manifest::VectorMetric;

    let Some(label) = label else {
        return Ok(None);
    };
    let metric = match distance {
        VectorDistance::Cosine => VectorMetric::Cosine,
        VectorDistance::Dot => VectorMetric::Dot,
        VectorDistance::Euclidean => VectorMetric::Euclidean,
    };
    // All three metrics now serve from the index: the `.vg` stores the original
    // vectors and reranks with the true metric, so the returned score equals the
    // flat path's (cosine similarity / raw dot / L2 distance), and the freshness
    // merge below scores deltas with the same `vector_score(distance, …)`.
    let (index_name, index_dim, index_int8) = {
        let desc = snapshot
            .manifest()
            .manifest
            .vector_indexes
            .iter()
            .find(|d| d.matches(label, property, metric));
        match desc {
            Some(d) => (
                d.name.clone(),
                d.dim as usize,
                d.quantization == namidb_storage::manifest::VectorQuantization::Int8,
            ),
            None => return Ok(None),
        }
    };

    // Freshness: a persisted `Nodes` SST newer than the index (flushed or
    // partially-merged but not yet folded in by an authoritative compaction)
    // means the `.vg` cannot see the full corpus — fall back to the exact flat
    // scan for that window (LSN-based; mirrors `text_search`).
    if snapshot.index_outrun_by_nodes(&index_name, namidb_storage::manifest::SstKind::VectorGraph) {
        return Ok(None);
    }

    // Coerce the query to f32 for the index search.
    let qv: Vec<f32> = match q {
        RuntimeValue::Vector(v) => v.clone(),
        RuntimeValue::Vector8 { codes, scale } => {
            codes.iter().map(|&c| c as f32 * *scale).collect()
        }
        _ => return Ok(None),
    };
    // Dimension parity with the index: a wrong-length query would otherwise be
    // silently prefix-scored by the index. Fall back to the flat scan, which
    // raises the canonical `vector dimension mismatch` error — same behaviour as
    // when no index exists.
    if qv.len() != index_dim {
        return Ok(None);
    }
    // A zero-magnitude query makes cosine undefined. The flat path's
    // `vector_score(Cosine, …)` returns `None` for every candidate (drop), so the
    // correct answer is empty — but the index rerank's `metric_score` returns a
    // similarity of 0.0 instead of dropping, which would diverge. Fall through to
    // the flat scan (the single source of truth) so the index path agrees with the
    // `cosine_similarity` builtin. Dot/L2 are well-defined on a zero query, so the
    // guard is cosine-only (mirrors `build_body`'s cosine-only zero-vector skip).
    if metric == VectorMetric::Cosine && qv.iter().all(|x| *x == 0.0) {
        return Ok(None);
    }

    // Fresh memtable/overlay delta the index has not absorbed: `Some(vec)` is a
    // live embedding to merge in, `None` suppresses a now-stale id (tombstoned,
    // label removed, or embedding dropped) so the merge stays equal to the flat
    // scan (RFC-030 freshness).
    let delta = snapshot
        .vector_fresh_delta(label, property)
        .map_err(ExecError::Storage)?;
    let delta_ids: BTreeSet<NodeId> = delta.iter().map(|(id, _)| *id).collect();

    // Pre-score the freshness delta ONCE: a delta's score is independent of the
    // fetch width, so the widening loop below only re-queries the index — it never
    // re-scores the delta. All three metrics serve from the index; an f32 index
    // reranks with the true metric so its score equals the flat path's, and the
    // delta's `vector_score(metric, …)` is on the same scale and orientation. An
    // int8 index returns the QUANTIZED cosine, so a fresh delta is round-tripped
    // through the same quantizer to keep both halves of the merge commensurable —
    // otherwise a node would score differently before vs after compaction folds it
    // into the index.
    let higher_is_better = !matches!(distance, VectorDistance::Euclidean);
    let mut delta_scored: Vec<(f64, NodeId)> = Vec::with_capacity(delta.len());
    if !delta.is_empty() {
        let q_rv = RuntimeValue::Vector(qv.clone());
        for (id, emb) in delta {
            if let Some(v) = emb {
                let v = if index_int8 {
                    let (codes, scale) = namidb_core::quantize::quantize_i8(&v);
                    namidb_core::quantize::dequantize_i8(&codes, scale)
                } else {
                    v
                };
                let emb_rv = RuntimeValue::Vector(v);
                if let Some((s, _higher)) = vector_score(distance, &emb_rv, &q_rv, span)? {
                    delta_scored.push((s, id));
                }
            }
        }
    }

    // Adaptive iterative widening: start at the historical ×8 over-fetch, then grow
    // `kprime`/`ef` geometrically (×4) whenever a residual `post_filter` leaves
    // fewer than k survivors, BEFORE the O(n) flat fallback — so a moderately
    // selective filter (the multi-tenant shared-index case) is served from the
    // index, not a flat scan every query. With no filter it is exactly one round at
    // mult=1: an exact top-k cannot under-fill from selectivity.
    const OVERFETCH_BASE: usize = 8;
    const WIDEN_GROWTH: usize = 4;
    const MAX_WIDEN_ROUNDS: usize = 4; // mult = 8, 32, 128, 512
    let widen = post_filter.is_some();
    let max_rounds = if widen { MAX_WIDEN_ROUNDS } else { 1 };
    let mut mult = if widen { OVERFETCH_BASE } else { 1 };

    for _ in 0..max_rounds {
        let kprime = k
            .saturating_mul(mult)
            .saturating_add(delta_ids.len())
            .max(k);
        // Beam width ≥ the fetch count so the beam can actually surface `kprime`
        // candidates. A user `$__vector_ef` is clamped only to `≥ kprime`, so for a
        // small `kprime` it can sit below the no-override default of 64 (a narrower
        // beam, lower recall); it is the recall/latency dial, not a one-way raise.
        // Correctness for a selective filter is still guaranteed by the flat
        // fallback below, regardless of the beam.
        let ef = match ef_search {
            Some(e) => e.max(kprime),
            None => kprime.max(64),
        };
        let hits = snapshot.vector_search(&index_name, &qv, kprime, ef).await?;
        // Fewer hits than asked ⇒ `kprime ≥` the corpus the index can see, so a
        // wider fetch cannot surface more (checked after using this round's hits).
        let index_exhausted = hits.len() < kprime;

        // Merge: deduped index hits not superseded by the delta, plus the
        // pre-scored delta. `seen` starts from the delta ids each round so a
        // superseded hit is dropped, and also dedups index hits a partial rebuild
        // returned twice (storage `vector_search` unions `.vg` SSTs without
        // deduping, and the wider `kprime` fetch widens that window).
        let mut seen: BTreeSet<NodeId> = delta_ids.clone();
        let mut scored: Vec<(f64, NodeId)> = Vec::with_capacity(hits.len() + delta_scored.len());
        for (id, score) in hits {
            if seen.insert(id) {
                scored.push((score as f64, id));
            }
        }
        scored.extend(delta_scored.iter().copied());
        if higher_is_better {
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        } else {
            scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        }

        // Materialise in rank order, applying the residual filter; take up to k.
        // Per-candidate deadline probe: a widened filtered ANN can do many cold
        // node lookups and must stay interruptible the way the flat scan is.
        let mut out = Vec::with_capacity(k);
        for (score, id) in scored {
            if out.len() >= k {
                break;
            }
            crate::exec::limits::check_deadline()?;
            let Some(view) = snapshot.lookup_node(label, id).await? else {
                continue;
            };
            let mut row = Row::new();
            row.set(
                alias.to_string(),
                RuntimeValue::Node(Box::new(NodeValue::from(view))),
            );
            row.set(score_alias.to_string(), RuntimeValue::Float(score));
            if let Some(pf) = post_filter {
                if evaluate(pf, &row, params)?.as_bool() != Some(true) {
                    continue;
                }
            }
            out.push(row);
        }

        if out.len() >= k {
            return Ok(Some(out));
        }
        // Once the index is drained, only the flat scan can reach k (it also covers
        // ids whose node is gone, `lookup_node` → None, and is the ground truth).
        if index_exhausted {
            break;
        }
        mult = mult.saturating_mul(WIDEN_GROWTH);
    }

    // Fewer than k survivors even after widening (a selective `post_filter`, or
    // index hits whose nodes vanished) — fall back to the exact flat scan: it
    // applies the same filter to every node and is the ground truth, never short.
    Ok(None)
}

/// Resolve the set of edge types an `Expand` operator must traverse.
///
/// `Some(types)` → traverse only the listed types (union for
/// alternation `[:A|:B]`). `None` (pattern wrote `-[r]->` without
/// a type label) → enumerate every edge type observable through the
/// snapshot (declared schema + memtable + persisted SSTs). Cost grows
/// linearly with the observed type count — EXPLAIN surfaces this so
/// users can opt back into typed expansion.
fn resolve_edge_types(snapshot: &Snapshot<'_>, edge_type: Option<&[String]>) -> Vec<String> {
    match edge_type {
        Some(types) => types.to_vec(),
        None => snapshot.observed_edge_types(),
    }
}

async fn neighbours_of_any(
    snapshot: &Snapshot<'_>,
    edge_types: &[String],
    direction: RelationshipDirection,
    node: NodeId,
    want_properties: bool,
) -> Result<Vec<EdgeView>, ExecError> {
    if edge_types.len() == 1 {
        return neighbours_of(snapshot, &edge_types[0], direction, node, want_properties).await;
    }
    let mut all = Vec::new();
    for et in edge_types {
        let edges = neighbours_of(snapshot, et, direction, node, want_properties).await?;
        all.extend(edges);
    }
    Ok(all)
}

async fn neighbours_of(
    snapshot: &Snapshot<'_>,
    edge_type: &str,
    direction: RelationshipDirection,
    node: NodeId,
    want_properties: bool,
) -> Result<Vec<EdgeView>, ExecError> {
    // Plan-aware routing (RFC-018 §4): when the rel binding the
    // Expand produces is read downstream — as `r` or as `r.prop` — we
    // force the SST path so `EdgeView.properties` is populated.
    // Otherwise we go through the default `out_edges` / `in_edges`
    // dispatch, which uses the CSR path when `NAMIDB_ADJACENCY=1` is
    // set and an adjacency cache is attached. Memtable-sourced edges
    // carry full properties on both paths.
    if want_properties {
        return match direction {
            RelationshipDirection::Right => {
                Ok(snapshot.out_edges_via_sst(edge_type, node).await?.edges)
            }
            RelationshipDirection::Left => {
                Ok(snapshot.in_edges_via_sst(edge_type, node).await?.edges)
            }
            RelationshipDirection::Both => {
                let mut out = snapshot.out_edges_via_sst(edge_type, node).await?.edges;
                // A self-loop (src == dst == node) is in BOTH out and in edges;
                // out_edges already yielded it, so drop it from the in half or
                // an undirected match returns/counts it twice.
                out.extend(
                    snapshot
                        .in_edges_via_sst(edge_type, node)
                        .await?
                        .edges
                        .into_iter()
                        .filter(|e| e.src != e.dst),
                );
                Ok(out)
            }
        };
    }
    match direction {
        RelationshipDirection::Right => Ok(snapshot.out_edges(edge_type, node).await?.edges),
        RelationshipDirection::Left => Ok(snapshot.in_edges(edge_type, node).await?.edges),
        RelationshipDirection::Both => {
            let mut out = snapshot.out_edges(edge_type, node).await?.edges;
            // Drop self-loops from the in half — out_edges already yielded them
            // (see the via_sst path above).
            out.extend(
                snapshot
                    .in_edges(edge_type, node)
                    .await?
                    .edges
                    .into_iter()
                    .filter(|e| e.src != e.dst),
            );
            Ok(out)
        }
    }
}

fn partner_id(edge: &EdgeView, direction: RelationshipDirection, source: NodeId) -> NodeId {
    match direction {
        RelationshipDirection::Right => edge.dst,
        RelationshipDirection::Left => edge.src,
        RelationshipDirection::Both => {
            if edge.src == source {
                edge.dst
            } else {
                edge.src
            }
        }
    }
}

/// Fix #3 entry point: decide whether the Expand's `target_alias`
/// binding can be stubbed (id-only) instead of materialised via
/// `lookup_node`. Five conditions must hold:
///
/// 1. `target_alias` is never read by any expression in the plan —
///    not in RETURN, WHERE, ORDER BY, projection items, join keys,
///    aggregation args, etc. Determined by [`PlanRouting::references`].
///    A `Variable(t)` or `Property(t, _)` anywhere flips this off.
/// 2. The length is single-hop (`*1..1`, the default). Variable-length
///    paths bind `target_alias` at every intermediate hop, so the
///    "transit only" assumption breaks down.
/// 3. The Expand is not a back-reference (the existing binding already
///    carries the materialised NodeView; we leave it alone).
/// 4. The edge_type is known statically (un-typed expand `(-[]-)` would
///    require enumerating every edge_type and we can't constrain the
///    target label).
/// 5. The `(edge_type, direction, target_label)` triple is
///    schema-guaranteed: the schema declares an edge_type whose
///    dst_label (Right) or src_label (Left) matches the target_label.
///    Any edge surfacing through the CSR / SST adjacency for that
///    `(edge_type, direction)` then points at a node guaranteed to be
///    of that label — the same invariant `lookup_node(label, id)`
///    enforces via its `continue`-on-None branch, but for free.
#[allow(clippy::too_many_arguments)]
fn should_skip_target_materialize(
    snapshot: &Snapshot<'_>,
    routing: &PlanRouting,
    target_alias: &str,
    edge_type: Option<&[String]>,
    direction: RelationshipDirection,
    target_labels: &[String],
    length: Option<crate::parser::RelationshipLength>,
    back_reference: bool,
) -> bool {
    if back_reference {
        return false;
    }
    if routing.references(target_alias) {
        return false;
    }
    // Single-hop only. None means "default *1..1" by lowering convention.
    let single_hop = length.map(|l| l.min == 1 && l.max == 1).unwrap_or(true);
    if !single_hop {
        return false;
    }
    let Some(edge_types) = edge_type else {
        return false;
    };
    // Skip is only safe for exactly ONE schema-guaranteed target label: the
    // optimization synthesises an id-only stub WITHOUT decoding the node, so it
    // can't confirm extra labels. An unlabelled target (len 0, legacy
    // `scan_node_for_id` path) and a multi-label target (len > 1, which needs
    // the conjunctive materialise-and-check) both fall through to the full path.
    let [target_label] = target_labels else {
        return false;
    };
    let target_label = target_label.as_str();
    let schema = &snapshot.manifest().manifest.schema;
    // Type alternation `[:A|:B]`: every listed type has to point at the
    // same target label, otherwise we'd silently drop matches where the
    // label diverges. Walking each declaration is O(types.len()), well
    // bounded in practice (the parser caps alternation at a handful).
    edge_types.iter().all(|et| {
        let Some(edge_def) = schema.edge_type(et) else {
            return false;
        };
        match direction {
            RelationshipDirection::Right => edge_def.dst_label == target_label,
            RelationshipDirection::Left => edge_def.src_label == target_label,
            RelationshipDirection::Both => {
                edge_def.dst_label == target_label && edge_def.src_label == target_label
            }
        }
    })
}

/// Walk every label in the manifest looking for a node with `id`.
/// Cypher's `Expand` doesn't carry the target label in v1 (only the
/// edge type), so we trial-search until storage provides a
/// label-index for ids.
pub(crate) async fn scan_node_for_id(
    snapshot: &Snapshot<'_>,
    id: NodeId,
) -> Result<Option<namidb_storage::NodeView>, ExecError> {
    // `observed_labels` covers the declared schema *and* labels that
    // were ever written into the memtable or any SST. Without it the
    // typeless Expand path falls back to declared-only and silently
    // drops every neighbour for namespaces that skipped `SchemaBuilder`
    // (the root cause of B1 / B7 — `MATCH ()-[r:T]->()` returning 0).
    for label in snapshot.observed_labels() {
        if let Some(view) = snapshot.lookup_node(&label, id).await? {
            return Ok(Some(view));
        }
    }
    Ok(None)
}

// ───────────────────────── Project ───────────────────────────────────

pub(crate) fn project_rows(
    rows: &[Row],
    items: &[ProjectionItem],
    discard_input_bindings: bool,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let mut new_row = if discard_input_bindings {
            Row::new()
        } else {
            row.clone()
        };
        for item in items {
            let v = evaluate(&item.expression, row, params)?;
            new_row.set(item.alias.clone(), v);
        }
        out.push(new_row);
    }
    Ok(out)
}

// ───────────────────────── Sort / Distinct ───────────────────────────

pub(crate) fn sort_rows(
    rows: &mut Vec<Row>,
    keys: &[OrderKey],
    params: &Params,
) -> Result<(), ExecError> {
    // Pre-compute key values for each row to avoid re-evaluating during
    // comparisons.
    let mut keyed: Vec<(Vec<RuntimeValue>, Row)> = Vec::with_capacity(rows.len());
    for row in rows.drain(..) {
        let mut row_keys = Vec::with_capacity(keys.len());
        for k in keys {
            row_keys.push(evaluate(&k.expression, &row, params)?);
        }
        keyed.push((row_keys, row));
    }
    keyed.sort_by(|(av, _), (bv, _)| compare_keys(av, bv, keys));
    *rows = keyed.into_iter().map(|(_, r)| r).collect();
    Ok(())
}

fn compare_keys(a: &[RuntimeValue], b: &[RuntimeValue], keys: &[OrderKey]) -> Ordering {
    for (i, k) in keys.iter().enumerate() {
        let desc = matches!(k.direction, crate::parser::OrderDirection::Desc);
        let o = order_for_sort(&a[i], &b[i], desc);
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

/// Heap element for the bounded top-k path of `TopN`. Orders by the same
/// per-key direction `compare_keys` uses (`descs[i]` is true when ORDER BY key
/// `i` is `DESC`), then breaks ties by `pos` — the element's position in the
/// input — so a max-heap that keeps the `k` smallest reproduces the stable
/// full-sort's first `k` exactly, ties and all. The heap's `peek` is the worst
/// kept candidate, evicted when a better one arrives.
struct TopNItem {
    vals: Vec<RuntimeValue>,
    leaf: crate::exec::FactorIdx,
    pos: usize,
    descs: std::sync::Arc<[bool]>,
}

impl Ord for TopNItem {
    fn cmp(&self, other: &Self) -> Ordering {
        for (i, &desc) in self.descs.iter().enumerate() {
            let o = order_for_sort(&self.vals[i], &other.vals[i], desc);
            if o != Ordering::Equal {
                return o;
            }
        }
        // Stable tiebreak: earlier input position sorts first.
        self.pos.cmp(&other.pos)
    }
}
impl PartialOrd for TopNItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for TopNItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for TopNItem {}

/// Flat-path bounded top-k heap element: like [`TopNItem`] but carries the full
/// [`Row`] instead of a factor leaf. Same key-then-position ordering.
struct FlatTopNItem {
    vals: Vec<RuntimeValue>,
    pos: usize,
    row: Row,
    descs: std::sync::Arc<[bool]>,
}
impl Ord for FlatTopNItem {
    fn cmp(&self, other: &Self) -> Ordering {
        for (i, &desc) in self.descs.iter().enumerate() {
            let o = order_for_sort(&self.vals[i], &other.vals[i], desc);
            if o != Ordering::Equal {
                return o;
            }
        }
        self.pos.cmp(&other.pos)
    }
}
impl PartialOrd for FlatTopNItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for FlatTopNItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for FlatTopNItem {}

/// Return the `bound` best rows for `keys`, already ordered, using an
/// `O(n log bound)` bounded max-heap instead of a full `O(n log n)` sort — the
/// standard `ORDER BY … LIMIT k` optimization (e.g. a KNN fallback over a 1M-row
/// label materialised only `k` rows' worth of keyed state, not 1M). Ties break
/// by input position, reproducing the stable full sort's first `bound` rows.
fn bounded_topk(
    rows: Vec<Row>,
    keys: &[OrderKey],
    params: &Params,
    bound: usize,
) -> Result<Vec<Row>, ExecError> {
    use std::collections::BinaryHeap;
    if bound == 0 {
        return Ok(Vec::new());
    }
    let descs: std::sync::Arc<[bool]> = keys
        .iter()
        .map(|k| matches!(k.direction, crate::parser::OrderDirection::Desc))
        .collect();
    let mut heap: BinaryHeap<FlatTopNItem> = BinaryHeap::with_capacity(bound + 1);
    for (pos, row) in rows.into_iter().enumerate() {
        let mut vals = Vec::with_capacity(keys.len());
        for k in keys {
            vals.push(evaluate(&k.expression, &row, params)?);
        }
        heap.push(FlatTopNItem {
            vals,
            pos,
            row,
            descs: descs.clone(),
        });
        // Max-heap: the root is the worst kept row; evict it once over budget so
        // only the `bound` best remain.
        if heap.len() > bound {
            heap.pop();
        }
    }
    let mut items = heap.into_vec();
    items.sort();
    Ok(items.into_iter().map(|it| it.row).collect())
}

pub(crate) fn cross_product(left: Vec<Row>, right: Vec<Row>) -> Vec<Row> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(left.len() * right.len());
    for l in &left {
        for r in &right {
            let mut merged = l.clone();
            for (k, v) in &r.bindings {
                merged.set(k.clone(), v.clone());
            }
            out.push(merged);
        }
    }
    out
}

pub(crate) fn dedup_rows(mut rows: Vec<Row>) -> Vec<Row> {
    // For determinism we sort by canonical key first then dedup. Since
    // RuntimeValue can hold Floats (which don't implement Ord), we use
    // a String fingerprint computed by serialising the row.
    rows.sort_by_key(row_fingerprint);
    rows.dedup();
    rows
}

fn row_fingerprint(row: &Row) -> String {
    let mut out = String::new();
    for (k, v) in &row.bindings {
        out.push_str(k);
        out.push('=');
        out.push_str(&fingerprint_value(v));
        out.push(';');
    }
    out
}

/// Content-faithful equality key for DISTINCT, GROUP BY, and hash-join
/// build/probe keys. Two [`RuntimeValue`]s must map to the same string iff they
/// are equal. Every variable-length payload is length-prefixed so distinct
/// values cannot collide through a shared separator (the old `,`-joined form
/// made `["a,s:b"]` and `["a","b"]` identical), and Float/Vector/Bytes encode
/// full content — the old length-only encoding collapsed every same-dimension
/// vector, every same-length byte string, and floats differing past 10 decimals
/// into one group / one join bucket.
fn fingerprint_value(v: &RuntimeValue) -> String {
    let mut out = String::new();
    fingerprint_into(&mut out, v);
    out
}

fn fingerprint_into(out: &mut String, v: &RuntimeValue) {
    use std::fmt::Write;
    match v {
        RuntimeValue::Null => out.push_str("Z;"),
        RuntimeValue::Bool(b) => {
            out.push('B');
            out.push(if *b { '1' } else { '0' });
        }
        RuntimeValue::Integer(n) => {
            let _ = write!(out, "I{n};");
        }
        RuntimeValue::Float(f) => {
            // Full-precision, exact: bit pattern (with -0.0 canonicalised to
            // +0.0 so the two compare equal, as `==` does). Beats `{:.10}`,
            // which merged values differing past the 10th decimal.
            let bits = if *f == 0.0 { 0u64 } else { f.to_bits() };
            let _ = write!(out, "F{bits:016x};");
        }
        RuntimeValue::String(s) => {
            // Length-prefixed content: self-delimiting, so no separator in the
            // string can be confused with a structural one.
            let _ = write!(out, "S{}:", s.len());
            out.push_str(s);
        }
        RuntimeValue::Bytes(b) => {
            let _ = write!(out, "Y{}:", b.len());
            for byte in b {
                let _ = write!(out, "{byte:02x}");
            }
        }
        RuntimeValue::Vector(vec) => {
            // Full component content (bit-exact), not just the dimension.
            let _ = write!(out, "V{}:", vec.len());
            for x in vec {
                let _ = write!(out, "{:08x}", x.to_bits());
            }
        }
        RuntimeValue::Vector8 { codes, scale } => {
            let _ = write!(out, "W{:08x}:{}:", scale.to_bits(), codes.len());
            for c in codes {
                let _ = write!(out, "{:02x}", *c as u8);
            }
        }
        RuntimeValue::List(items) => {
            let _ = write!(out, "L{}:", items.len());
            for it in items {
                fingerprint_into(out, it);
            }
        }
        RuntimeValue::Map(m) => {
            let _ = write!(out, "M{}:", m.len());
            for (k, val) in m {
                let _ = write!(out, "{}=", k.len());
                out.push_str(k);
                fingerprint_into(out, val);
            }
        }
        // A node's id is its identity; an edge's identity is (type, src, dst)
        // — the storage key for edges — with the type length-prefixed.
        RuntimeValue::Node(n) => {
            let _ = write!(out, "N{};", n.id);
        }
        RuntimeValue::Rel(r) => {
            let _ = write!(out, "R{}:{}{}>{};", r.edge_type.len(), r.edge_type, r.src, r.dst);
        }
        RuntimeValue::Date(d) => {
            let _ = write!(out, "D{d};");
        }
        RuntimeValue::DateTime(d) => {
            let _ = write!(out, "T{d};");
        }
        RuntimeValue::Path(items) => {
            let _ = write!(out, "P{}:", items.len());
            for it in items {
                fingerprint_into(out, it);
            }
        }
    }
}

// ───────────────────────── Aggregate ─────────────────────────────────

pub(crate) fn execute_aggregate(
    rows: Vec<Row>,
    group_by: &[(Expression, String)],
    aggregations: &[(String, AggregateExpr)],
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    use std::collections::BTreeMap;

    let mut groups: BTreeMap<String, (Vec<RuntimeValue>, Vec<Row>)> = BTreeMap::new();
    for row in rows {
        let mut key_vals = Vec::with_capacity(group_by.len());
        let mut fingerprint = String::new();
        for (expr, _) in group_by {
            let v = evaluate(expr, &row, params)?;
            fingerprint.push_str(&fingerprint_value(&v));
            fingerprint.push('|');
            key_vals.push(v);
        }
        groups
            .entry(fingerprint)
            .or_insert_with(|| (key_vals, Vec::new()))
            .1
            .push(row);
    }
    // Empty input + no group keys: emit one row of "empty group" so that
    // `RETURN count(*)` over zero rows yields `0`.
    if groups.is_empty() && group_by.is_empty() {
        let mut row = Row::new();
        for (alias, agg) in aggregations {
            row.set(alias.clone(), aggregate_over(&[], agg, params)?);
        }
        return Ok(vec![row]);
    }

    let mut out = Vec::with_capacity(groups.len());
    for (_, (key_vals, group_rows)) in groups {
        let mut row = Row::new();
        for ((_, alias), v) in group_by.iter().zip(key_vals) {
            row.set(alias.clone(), v);
        }
        for (alias, agg) in aggregations {
            row.set(alias.clone(), aggregate_over(&group_rows, agg, params)?);
        }
        out.push(row);
    }
    Ok(out)
}

fn aggregate_over(
    rows: &[Row],
    agg: &AggregateExpr,
    params: &Params,
) -> Result<RuntimeValue, ExecError> {
    match agg {
        AggregateExpr::Count { arg, distinct } => match arg {
            None => Ok(RuntimeValue::Integer(rows.len() as i64)),
            Some(e) => {
                let mut count: i64 = 0;
                let mut seen = BTreeSet::new();
                for row in rows {
                    let v = evaluate(e, row, params)?;
                    if v.is_null() {
                        continue;
                    }
                    if *distinct {
                        let fp = fingerprint_value(&v);
                        if seen.insert(fp) {
                            count += 1;
                        }
                    } else {
                        count += 1;
                    }
                }
                Ok(RuntimeValue::Integer(count))
            }
        },
        AggregateExpr::Sum { arg, distinct } => {
            let vals = collect_non_null(rows, arg, *distinct, params)?;
            sum_values(&vals)
        }
        AggregateExpr::Avg { arg, distinct } => {
            let vals = collect_non_null(rows, arg, *distinct, params)?;
            if vals.is_empty() {
                return Ok(RuntimeValue::Null);
            }
            let total = sum_values(&vals)?;
            match total {
                RuntimeValue::Integer(n) => Ok(RuntimeValue::Float(n as f64 / vals.len() as f64)),
                RuntimeValue::Float(f) => Ok(RuntimeValue::Float(f / vals.len() as f64)),
                _ => Ok(RuntimeValue::Null),
            }
        }
        AggregateExpr::Min { arg } => {
            let vals = collect_non_null(rows, arg, false, params)?;
            Ok(vals
                .into_iter()
                .min_by(|a, b| order_for_sort(a, b, false))
                .unwrap_or(RuntimeValue::Null))
        }
        AggregateExpr::Max { arg } => {
            let vals = collect_non_null(rows, arg, false, params)?;
            Ok(vals
                .into_iter()
                .max_by(|a, b| order_for_sort(a, b, false))
                .unwrap_or(RuntimeValue::Null))
        }
        AggregateExpr::Collect { arg, distinct } => {
            let vals = collect_non_null(rows, arg, *distinct, params)?;
            Ok(RuntimeValue::List(vals))
        }
    }
}

fn collect_non_null(
    rows: &[Row],
    arg: &Expression,
    distinct: bool,
    params: &Params,
) -> Result<Vec<RuntimeValue>, ExecError> {
    let mut out = Vec::with_capacity(rows.len());
    let mut seen = BTreeSet::new();
    for row in rows {
        let v = evaluate(arg, row, params)?;
        if v.is_null() {
            continue;
        }
        if distinct {
            let fp = fingerprint_value(&v);
            if !seen.insert(fp) {
                continue;
            }
        }
        out.push(v);
    }
    Ok(out)
}

fn sum_values(vals: &[RuntimeValue]) -> Result<RuntimeValue, ExecError> {
    let mut is_float = false;
    let mut i_total: i64 = 0;
    let mut f_total: f64 = 0.0;
    for v in vals {
        match v {
            RuntimeValue::Integer(n) => {
                if is_float {
                    f_total += *n as f64;
                } else {
                    i_total += *n;
                }
            }
            RuntimeValue::Float(f) => {
                if !is_float {
                    f_total = i_total as f64;
                    is_float = true;
                }
                f_total += *f;
            }
            _ => {
                return Err(ExecError::Runtime(format!(
                    "sum/avg requires numeric values, got {}",
                    v.type_name()
                )))
            }
        }
    }
    if vals.is_empty() {
        return Ok(RuntimeValue::Null);
    }
    if is_float {
        Ok(RuntimeValue::Float(f_total))
    } else {
        Ok(RuntimeValue::Integer(i_total))
    }
}

// ───────────────────────── NodeId conversion ─────────────────────────

pub(crate) fn node_id_from_value(v: &RuntimeValue, span: SourceSpan) -> Result<NodeId, ExecError> {
    match v {
        RuntimeValue::String(s) => match uuid::Uuid::parse_str(s) {
            Ok(u) => Ok(NodeId::from_uuid(u)),
            Err(e) => Err(ExecError::Eval(EvalError::new(
                format!("invalid NodeId `{}`: {}", s, e),
                span,
            ))),
        },
        RuntimeValue::Node(n) => Ok(n.id),
        _ => Err(ExecError::Eval(EvalError::new(
            format!(
                "NodeId must be a UUID string or Node, got {}",
                v.type_name()
            ),
            span,
        ))),
    }
}

/// Lookup a node by a unique user property via predicate-pushed scan
/// + first-match short-circuit. Used by `LogicalPlan::NodeByPropertyValue`.
///
/// The storage layer's `scan_label_with_predicates` already pushes the
/// `Eq` predicate to the row-group level (only matching row-groups are
/// decoded — bloom + min/max prune away the rest). Once it returns the
/// candidate set, we filter exactly and take the first match: per the
/// `PropertyDef::unique` contract there's at most one.
///
/// Future optimisation: a dedicated `Snapshot::lookup_node_by_property`
/// that short-circuits the *storage-side* iteration once the first
/// match is found (today the scan still materialises all matches
/// before returning; for a truly unique property that's exactly one
/// row, so the waste is bounded — for a misdeclared "unique" property
/// with multiple matches, we silently take the first).
pub(crate) async fn lookup_node_by_property_via_scan(
    snapshot: &Snapshot<'_>,
    label: &str,
    property: &str,
    value: &RuntimeValue,
) -> Result<Option<namidb_storage::NodeView>, ExecError> {
    // For v0 we only index String-valued properties (LDBC's `id`).
    // Other scalar types fall back to `scan_label_with_predicates` —
    // accurate but pays the per-row decoder overhead every call.
    if let RuntimeValue::String(s) = value {
        return snapshot
            .lookup_node_by_property(label, property, s)
            .await
            .map_err(ExecError::from);
    }

    // Fallback for non-string keys.
    let scalar = match value {
        RuntimeValue::Integer(i) => Some(namidb_storage::sst::stats::StatScalar::Int64(*i)),
        RuntimeValue::Bool(b) => Some(namidb_storage::sst::stats::StatScalar::Bool(*b)),
        RuntimeValue::Float(f) => Some(namidb_storage::sst::stats::StatScalar::Float64(*f)),
        _ => None,
    };
    let candidates = if let Some(s) = scalar {
        let pred = namidb_storage::sst::predicates::ScanPredicate::Eq {
            column: property.to_string(),
            value: s,
        };
        snapshot
            .scan_label_with_predicates(label, &[pred])
            .await
            .map_err(ExecError::from)?
    } else {
        snapshot.scan_label(label).await.map_err(ExecError::from)?
    };
    Ok(candidates.into_iter().next())
}

/// Multi-match variant of [`lookup_node_by_property_via_scan`] for a
/// non-unique `indexed` property: returns every node carrying `value`. A
/// String key resolves through the equality posting-list sidecar; other
/// scalar types fall back to a full label scan filtered by exact value
/// (no typed sidecar yet).
pub(crate) async fn lookup_nodes_by_property_via_scan(
    snapshot: &Snapshot<'_>,
    label: &str,
    property: &str,
    value: &RuntimeValue,
) -> Result<Vec<namidb_storage::NodeView>, ExecError> {
    if let RuntimeValue::String(s) = value {
        return snapshot
            .lookup_nodes_by_property(label, property, s)
            .await
            .map_err(ExecError::from);
    }
    let all = snapshot.scan_label(label).await.map_err(ExecError::from)?;
    Ok(all
        .into_iter()
        .filter(|view| {
            view.properties
                .get(property)
                .map(|cv| RuntimeValue::from(cv.clone()) == *value)
                .unwrap_or(false)
        })
        .collect())
}

// ────────────────────────── Factor path ────────────────────────
//
// `execute_factor_inner` mirrors `execute_inner` but operates on
// `FactorRowSet`. Only the operators whose chained execution benefits
// from factorisation are reimplemented here (Expand and the operators
// that typically sit on its input/output edges in IC02/IC09 plans:
// Empty, NodeScan, NodeById, Filter, Project intermediate). Operators
// not yet ported execute the flat path on a materialised input and wrap
// the result via `FactorRowSet::from_flat`. This keeps initial scope small while
// proving the factor harness end-to-end; later iterations will port the binary
// operators (CrossProduct, HashJoin) and later iterations will port sinks.

pub(crate) fn execute_factor_inner_with_routing<'a>(
    plan: &'a LogicalPlan,
    snapshot: &'a Snapshot<'_>,
    params: &'a Params,
    outer: Option<&'a Row>,
    routing: &'a PlanRouting,
) -> BoxFuture<'a, Result<FactorRowSet, ExecError>> {
    async move {
        // Deadline guard (read query timeout): one check per factor-path
        // operator invocation, mirroring the flat path.
        crate::exec::limits::check_deadline()?;
        match plan {
            // Operators that benefit directly: keep everything factorised.
            LogicalPlan::Empty => Ok(FactorRowSet::singleton_root()),

            // VectorSearch is a leaf that drives its own read; run the flat
            // fallback and wrap it (no factorisation benefit for a top-k scan).
            LogicalPlan::VectorSearch {
                label,
                alias,
                property,
                query,
                k,
                distance,
                score_alias,
                post_filter,
            } => {
                let rows = flat_vector_search(
                    snapshot,
                    label.as_deref(),
                    alias,
                    property,
                    query,
                    k,
                    *distance,
                    score_alias,
                    post_filter.as_ref(),
                    params,
                )
                .await?;
                Ok(FactorRowSet::from_flat(rows))
            }

            // CallProcedure is a source leaf; run the flat helper and wrap it.
            LogicalPlan::CallProcedure {
                namespace,
                name,
                args,
                yield_items,
            } => {
                let rows = flat_call_procedure(
                    namespace.as_deref(),
                    name,
                    args,
                    yield_items,
                    snapshot,
                    params,
                )
                .await?;
                Ok(FactorRowSet::from_flat(rows))
            }

            LogicalPlan::NodeScan {
                label,
                alias,
                predicates,
                projection,
            } => {
                // NodeScan produces N independent rows — emit each as a
                // direct child of root. No clone path; one FactorNode per
                // result. For typeless scans we fan out across every
                // observed label and concatenate the result.
                let labels = resolve_node_labels(snapshot, label.as_deref());
                let mut set = FactorRowSet::singleton_root();
                let root = set.arena.root();
                let alias_arc: Arc<str> = Arc::from(alias.as_str());
                let mut leaves: Vec<crate::exec::FactorIdx> = Vec::new();
                for label_name in &labels {
                    let nodes = snapshot
                        .scan_label_with_predicates_and_projection(
                            label_name,
                            predicates,
                            projection.as_deref(),
                        )
                        .await?;
                    for n in nodes {
                        let slot = Slot {
                            name: alias_arc.clone(),
                            value: RuntimeValue::Node(Box::new(NodeValue::from(n))),
                        };
                        leaves.push(set.arena.push(root, vec![slot]));
                    }
                }
                set.leaves = leaves;
                Ok(set)
            }

            LogicalPlan::NodeById {
                input,
                label,
                alias,
                id,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let alias_arc: Arc<str> = Arc::from(alias.as_str());
                let mut out_leaves = Vec::new();
                // We need a temporary materialised row per leaf to evaluate
                // `id` (which is an Expression over bindings). Building
                // the row only includes the bindings the Expression
                // references is a future optimisation;
                // materialising the whole chain is correct now.
                let arena_view = input_set.arena.clone();
                let mut next_arena = input_set.arena;
                for leaf in input_set.leaves {
                    let row = arena_view.materialize(leaf, None);
                    let id_value = evaluate(id, &row, params)?;
                    let node_id = node_id_from_value(&id_value, id.span)?;
                    let found = match label {
                        Some(l) => snapshot.lookup_node(l, node_id).await?,
                        None => scan_node_for_id(snapshot, node_id).await?,
                    };
                    if let Some(view) = found {
                        let slot = Slot {
                            name: alias_arc.clone(),
                            value: RuntimeValue::Node(Box::new(NodeValue::from(view))),
                        };
                        out_leaves.push(next_arena.push(leaf, vec![slot]));
                    }
                }
                Ok(FactorRowSet {
                    arena: next_arena,
                    leaves: out_leaves,
                })
            }

            LogicalPlan::NodeByPropertyValue {
                input,
                label,
                alias,
                property,
                value,
                multi,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let alias_arc: Arc<str> = Arc::from(alias.as_str());
                let mut out_leaves = Vec::new();
                let arena_view = input_set.arena.clone();
                let mut next_arena = input_set.arena;
                for leaf in input_set.leaves {
                    let row = arena_view.materialize(leaf, None);
                    let lookup_val = evaluate(value, &row, params)?;
                    if *multi {
                        for view in lookup_nodes_by_property_via_scan(
                            snapshot,
                            label,
                            property,
                            &lookup_val,
                        )
                        .await?
                        {
                            let slot = Slot {
                                name: alias_arc.clone(),
                                value: RuntimeValue::Node(Box::new(NodeValue::from(view))),
                            };
                            out_leaves.push(next_arena.push(leaf, vec![slot]));
                        }
                    } else if let Some(view) =
                        lookup_node_by_property_via_scan(snapshot, label, property, &lookup_val)
                            .await?
                    {
                        let slot = Slot {
                            name: alias_arc.clone(),
                            value: RuntimeValue::Node(Box::new(NodeValue::from(view))),
                        };
                        out_leaves.push(next_arena.push(leaf, vec![slot]));
                    }
                }
                Ok(FactorRowSet {
                    arena: next_arena,
                    leaves: out_leaves,
                })
            }

            LogicalPlan::Filter { input, predicate } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let arena_view = input_set.arena.clone();
                let mut out_leaves = Vec::with_capacity(input_set.leaves.len());
                for leaf in input_set.leaves {
                    // Materialise full row to evaluate predicate. Column-
                    // aware materialise is a follow-up; today we accept the
                    // O(depth) walk + transient Row alloc per leaf.
                    let row = arena_view.materialize(leaf, None);
                    let v = evaluate(predicate, &row, params)?;
                    if v.as_bool() == Some(true) {
                        out_leaves.push(leaf);
                    }
                }
                Ok(FactorRowSet {
                    arena: input_set.arena,
                    leaves: out_leaves,
                })
            }

            LogicalPlan::Expand {
                input,
                source,
                edge_type,
                direction,
                rel_alias,
                target_alias,
                target_labels,
                length,
                optional,
                back_reference,
                shortest,
                path_binding,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let length = resolve_length(length, params)?;
                // The factor expand executor does not materialise a path binding
                // (`p`) or a shortestPath trail. Route those to the flat executor
                // (which does) and re-wrap — otherwise `p` / `nodes(p)` downstream
                // of a factorised variable-length expand sees an unbound `p`.
                if path_binding.is_some() || !matches!(shortest, crate::plan::ShortestMode::None) {
                    let rows = input_set.materialize_all(None);
                    let out = execute_expand(
                        rows,
                        source,
                        edge_type.as_deref(),
                        *direction,
                        rel_alias.as_deref(),
                        target_alias,
                        target_labels,
                        length,
                        *optional,
                        *back_reference,
                        *shortest,
                        path_binding.as_deref(),
                        snapshot,
                        routing.needs_properties(rel_alias.as_deref()),
                        false,
                        None,
                    )
                    .await?;
                    return Ok(FactorRowSet::from_flat(out));
                }
                execute_expand_factor(
                    input_set,
                    source,
                    edge_type.as_deref(),
                    *direction,
                    rel_alias.as_deref(),
                    target_alias,
                    target_labels,
                    length.clone(),
                    *optional,
                    *back_reference,
                    snapshot,
                    routing.needs_properties(rel_alias.as_deref()),
                    should_skip_target_materialize(
                        snapshot,
                        routing,
                        target_alias,
                        edge_type.as_deref(),
                        *direction,
                        target_labels,
                        length,
                        *back_reference,
                    ),
                )
                .await
            }

            LogicalPlan::CrossProduct { left, right } => {
                let l = execute_factor_inner_with_routing(left, snapshot, params, outer, routing)
                    .await?;
                let r = execute_factor_inner_with_routing(right, snapshot, params, outer, routing)
                    .await?;
                cross_product_factor(l, r)
            }

            LogicalPlan::HashJoin {
                build,
                probe,
                on,
                residual,
            } => {
                hash_join_factor(
                    build,
                    probe,
                    on,
                    residual.as_ref(),
                    snapshot,
                    params,
                    outer,
                    routing,
                )
                .await
            }

            LogicalPlan::HashSemiJoin {
                outer: outer_plan,
                inner: inner_plan,
                on,
                negated,
                residual,
            } => {
                hash_semi_join_factor(
                    outer_plan,
                    inner_plan,
                    on,
                    *negated,
                    residual.as_ref(),
                    snapshot,
                    params,
                    outer,
                    routing,
                )
                .await
            }

            // Operators not yet ported to the factor path: execute flat
            // and wrap. The wrap inserts one FactorNode per result row,
            // each as a direct child of root; downstream operators continue
            // Sinks / pass-through operators: recurse children via the
            // factor path (so multi-hop Expand chains underneath stay
            // factorised), then materialise at the operator boundary and
            // re-use the existing flat helpers. v0 strategy — later iterations add
            // true factor-native versions (heap-by-lookup_binding for
            // TopN, fingerprint group-by without full materialise, etc.).
            LogicalPlan::Project {
                input,
                items,
                distinct,
                discard_input_bindings,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let input_rows = input_set.materialize_all(None);
                let projected = project_rows(&input_rows, items, *discard_input_bindings, params)?;
                let rows = if *distinct {
                    dedup_rows(projected)
                } else {
                    projected
                };
                Ok(FactorRowSet::from_flat(rows))
            }

            LogicalPlan::TopN {
                input,
                keys,
                skip,
                limit,
            } => {
                // TopN heap-native over the arena. Instead of
                // materialising every leaf and then sorting / taking, we:
                // 1. Compute each ORDER BY key per leaf using a thin row
                // that holds only the bindings the key expressions
                // reference (collected statically). Avoids cloning
                // unrelated NodeValue properties for IC09-shaped
                // queries (1500 leaves × 3 unused NodeValues =
                // ~4500 RuntimeValue clones avoided).
                // 2. Sort the (key_vals, leaf) pairs.
                // 3. Materialise the full row only for the `skip..skip+limit`
                // window — 20 materialisations for `LIMIT 20`
                // regardless of input cardinality.
                let skip = resolve_row_count(skip, params, "SKIP")?;
                let limit = resolve_row_count(limit, params, "LIMIT")?;
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;

                // Empty keys: stable order, just skip+take + materialise.
                if keys.is_empty() {
                    let skip = skip as usize;
                    if skip >= input_set.cardinality() {
                        return Ok(FactorRowSet::from_flat(Vec::new()));
                    }
                    let take = if limit == u64::MAX {
                        usize::MAX
                    } else {
                        limit as usize
                    };
                    let rows: Vec<Row> = input_set
                        .leaves
                        .iter()
                        .skip(skip)
                        .take(take)
                        .map(|&leaf| input_set.arena.materialize(leaf, None))
                        .collect();
                    return Ok(FactorRowSet::from_flat(rows));
                }

                // Variables referenced by ANY of the ORDER BY expressions.
                let mut needed: BTreeSet<String> = BTreeSet::new();
                for k in keys {
                    collect_referenced_variables(&k.expression, &mut needed);
                }

                // Bounded top-k fast path. When a `LIMIT` bounds the result to
                // `k = skip + limit` rows and `k < n`, keep only the `k` best in
                // a max-heap whose top is the worst kept candidate: O(n log k)
                // time and O(k) memory instead of materialising and sorting all
                // `n` keyed rows. This is the hot path for KNN
                // (`ORDER BY cosine_similarity(...) DESC LIMIT k`). The position
                // tiebreak makes the result identical to the full sort below.
                if limit != u64::MAX {
                    let k = (skip as usize).saturating_add(limit as usize);
                    if k > 0 && k < input_set.cardinality() {
                        let descs: std::sync::Arc<[bool]> = std::sync::Arc::from(
                            keys.iter()
                                .map(|key| {
                                    matches!(key.direction, crate::parser::OrderDirection::Desc)
                                })
                                .collect::<Vec<bool>>(),
                        );
                        let mut heap: std::collections::BinaryHeap<TopNItem> =
                            std::collections::BinaryHeap::with_capacity(k + 1);
                        for (pos, &leaf) in input_set.leaves.iter().enumerate() {
                            let mut thin_row = Row::new();
                            for var_name in &needed {
                                if let Some(v) = input_set.arena.lookup_binding(leaf, var_name) {
                                    thin_row.set(var_name.clone(), v.clone());
                                }
                            }
                            let mut key_vals = Vec::with_capacity(keys.len());
                            for key in keys {
                                key_vals.push(evaluate(&key.expression, &thin_row, params)?);
                            }
                            let item = TopNItem {
                                vals: key_vals,
                                leaf,
                                pos,
                                descs: descs.clone(),
                            };
                            if heap.len() < k {
                                heap.push(item);
                            } else if &item < heap.peek().expect("heap full so non-empty") {
                                // `item` sorts before the current worst kept.
                                heap.pop();
                                heap.push(item);
                            }
                        }
                        let mut kept = heap.into_vec();
                        kept.sort_unstable();
                        let rows: Vec<Row> = kept
                            .into_iter()
                            .skip(skip as usize)
                            .take(limit as usize)
                            .map(|it| input_set.arena.materialize(it.leaf, None))
                            .collect();
                        return Ok(FactorRowSet::from_flat(rows));
                    }
                }

                let mut keyed: Vec<(Vec<RuntimeValue>, crate::exec::FactorIdx)> =
                    Vec::with_capacity(input_set.cardinality());
                for &leaf in &input_set.leaves {
                    // Thin row: only the bindings the keys actually read.
                    let mut thin_row = Row::new();
                    for var_name in &needed {
                        if let Some(v) = input_set.arena.lookup_binding(leaf, var_name) {
                            thin_row.set(var_name.clone(), v.clone());
                        }
                    }
                    let mut key_vals = Vec::with_capacity(keys.len());
                    for k in keys {
                        key_vals.push(evaluate(&k.expression, &thin_row, params)?);
                    }
                    keyed.push((key_vals, leaf));
                }

                keyed.sort_by(|(av, _), (bv, _)| compare_keys(av, bv, keys));

                let skip = skip as usize;
                if skip >= keyed.len() {
                    return Ok(FactorRowSet::from_flat(Vec::new()));
                }
                let take = if limit == u64::MAX {
                    usize::MAX
                } else {
                    limit as usize
                };
                let rows: Vec<Row> = keyed
                    .into_iter()
                    .skip(skip)
                    .take(take)
                    .map(|(_, leaf)| input_set.arena.materialize(leaf, None))
                    .collect();
                Ok(FactorRowSet::from_flat(rows))
            }

            LogicalPlan::Distinct { input } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let rows = input_set.materialize_all(None);
                Ok(FactorRowSet::from_flat(dedup_rows(rows)))
            }

            LogicalPlan::Union { left, right, all } => {
                let l = execute_factor_inner_with_routing(left, snapshot, params, outer, routing)
                    .await?;
                let r = execute_factor_inner_with_routing(right, snapshot, params, outer, routing)
                    .await?;
                let mut rows = l.materialize_all(None);
                rows.extend(r.materialize_all(None));
                let out = if *all { rows } else { dedup_rows(rows) };
                Ok(FactorRowSet::from_flat(out))
            }

            LogicalPlan::Unwind { input, list, alias } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let input_rows = input_set.materialize_all(None);
                let mut out = Vec::new();
                for row in input_rows {
                    let v = evaluate(list, &row, params)?;
                    match v {
                        RuntimeValue::List(items) => {
                            for item in items {
                                let mut new_row = row.clone();
                                new_row.set(alias.clone(), item);
                                out.push(new_row);
                            }
                        }
                        RuntimeValue::Null => {}
                        _ => {
                            return Err(ExecError::Runtime(format!(
                                "UNWIND requires a list; got {}",
                                v.type_name()
                            )));
                        }
                    }
                }
                // UNWIND amplifies each input row into its list, so guard the
                // output (the flat path is bounded by the per-operator check).
                crate::exec::limits::check_row_cap(out.len())?;
                Ok(FactorRowSet::from_flat(out))
            }

            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregations,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let rows = input_set.materialize_all(None);
                let agg_rows = execute_aggregate(rows, group_by, aggregations, params)?;
                Ok(FactorRowSet::from_flat(agg_rows))
            }

            // Correlated subplan operators: the outer row is threaded into
            // the inner execute. Inner planning is read-once but the outer
            // may bind ad-hoc fields, so we materialise the outer per row
            // to thread it through `execute_inner(subplan, ..., Some(row))`.
            LogicalPlan::SemiApply {
                input,
                subplan,
                negated,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let rows = input_set.materialize_all(None);
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let sub_rows =
                        execute_inner_with_routing(subplan, snapshot, params, Some(&row), routing)
                            .await?;
                    let matched = !sub_rows.is_empty();
                    let keep = if *negated { !matched } else { matched };
                    if keep {
                        out.push(row);
                    }
                }
                Ok(FactorRowSet::from_flat(out))
            }

            LogicalPlan::Apply { input, subplan } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let rows = input_set.materialize_all(None);
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let sub_rows =
                        execute_inner_with_routing(subplan, snapshot, params, Some(&row), routing)
                            .await?;
                    for s in sub_rows {
                        let mut merged = row.clone();
                        for (k, v) in &s.bindings {
                            merged.set(k.clone(), v.clone());
                        }
                        out.push(merged);
                    }
                }
                Ok(FactorRowSet::from_flat(out))
            }

            LogicalPlan::PatternList {
                input,
                subplan,
                projection,
                alias,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                let rows = input_set.materialize_all(None);
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let inner_rows =
                        execute_inner_with_routing(subplan, snapshot, params, Some(&row), routing)
                            .await?;
                    let mut list = Vec::with_capacity(inner_rows.len());
                    for inner in inner_rows {
                        list.push(evaluate(projection, &inner, params)?);
                    }
                    let mut new_row = row;
                    new_row.set(alias.clone(), RuntimeValue::List(list));
                    out.push(new_row);
                }
                Ok(FactorRowSet::from_flat(out))
            }

            LogicalPlan::Argument { bindings } => {
                let outer = outer.ok_or_else(|| {
                    ExecError::Runtime(
 "Argument operator requires an outer row from a containing SemiApply / PatternList".into(),
 )
                })?;
                let mut row = Row::new();
                for name in bindings {
                    if let Some(v) = outer.get(name) {
                        row.set(name.clone(), v.clone());
                    }
                }
                Ok(FactorRowSet::from_flat(vec![row]))
            }

            // Write operators must go through execute_write. Surface a
            // clear error if they reach the read executor regardless of
            // path.
            LogicalPlan::Create { .. }
            | LogicalPlan::Merge { .. }
            | LogicalPlan::Set { .. }
            | LogicalPlan::Remove { .. }
            | LogicalPlan::Delete { .. }
            | LogicalPlan::Foreach { .. } => Err(ExecError::Runtime(
                "write operators require execute_write(plan, &mut WriterSession, params)"
                    .to_string(),
            )),

            LogicalPlan::EdgeTypeCount { edge_types, output } => {
                // Same per-type sum as the flat path. The edge-count pushdown
                // pass runs unconditionally in optimize(), so this leaf reaches
                // the factor executor too; erroring here regressed a common
                // `RETURN count(r)` query for every NAMIDB_FACTORIZE=1 deployment.
                let mut total: i64 = 0;
                for et in edge_types {
                    total += snapshot.count_edge_type(et).await? as i64;
                }
                Ok(FactorRowSet::from_flat(vec![Row::new()
                    .with(output.clone(), RuntimeValue::Integer(total))]))
            }

            LogicalPlan::MultiwayJoin {
                vars,
                edges,
                ordering,
                factorize_required,
            } => {
                if !*factorize_required {
                    return Err(ExecError::Runtime(
                        "MultiwayJoin v0 requires factorize_required=true".to_string(),
                    ));
                }
                if outer.is_some() {
                    return Err(ExecError::Runtime(
                        "MultiwayJoin v0 cannot run under a correlated subplan; \
                         detection pass should skip subtrees referenced from Argument"
                            .to_string(),
                    ));
                }
                execute_multiway_join_factor(vars, edges, ordering, snapshot).await
            }
        }
    }
    .boxed()
}

/// A stored edge's identity `(edge_type, src, dst)` — the key for Cypher
/// relationship uniqueness in variable-length expansion.
type EdgeIdentity = (String, NodeId, NodeId);

/// A factor-path expansion frontier entry: the factor node the path was reached
/// under, its tail node, and the relationships already traversed on the path
/// (for relationship-uniqueness; empty on the single-hop path).
type FactorFrontierEntry = (crate::exec::FactorIdx, NodeId, Vec<EdgeIdentity>);

// Pointer-based Expand that avoids the per-edge `BTreeMap` clone in
// walker.rs:544 / 554. Each surviving (parent_leaf, edge, target) tuple
// produces one new `FactorNode` carrying just the rel and target slots;
// inherited bindings (source row + any prior Expand levels) stay implicit
// via the parent chain.
#[allow(clippy::too_many_arguments)]
async fn execute_expand_factor(
    input: FactorRowSet,
    source: &str,
    edge_type: Option<&[String]>,
    direction: RelationshipDirection,
    rel_alias: Option<&str>,
    target_alias: &str,
    target_labels: &[String],
    length: Option<crate::parser::RelationshipLength>,
    optional: bool,
    back_reference: bool,
    snapshot: &Snapshot<'_>,
    want_properties: bool,
    skip_target_materialize: bool,
) -> Result<FactorRowSet, ExecError> {
    namidb_core::profile_scope!("walker::execute_expand_factor");
    let edge_types = resolve_edge_types(snapshot, edge_type);
    let min = length.as_ref().map(|l| l.min).unwrap_or(1);
    let max = clamp_hop_max(length.as_ref().map(|l| l.max).unwrap_or(1));

    let FactorRowSet {
        mut arena,
        leaves: input_leaves,
    } = input;
    let target_arc: Arc<str> = Arc::from(target_alias);
    let rel_arc: Option<Arc<str>> = rel_alias.map(Arc::from);

    let mut out_leaves: Vec<crate::exec::FactorIdx> = Vec::new();

    for parent_leaf in input_leaves {
        // Deadline + row-cap guards, mirroring the flat `execute_expand`:
        // a factor-path expansion is just as able to run long or to build
        // an unbounded arena, so bound it at every seed boundary.
        crate::exec::limits::check_deadline()?;
        crate::exec::limits::check_row_cap(out_leaves.len())?;
        // Read the source binding from the chain. lookup_binding walks
        // root-ward without materialising the whole row.
        let starting = match arena.lookup_binding(parent_leaf, source) {
            Some(RuntimeValue::Node(n)) => n.id,
            Some(_) => {
                return Err(ExecError::Runtime(format!(
                    "Expand source `{}` is not a Node",
                    source
                )))
            }
            None => {
                return Err(ExecError::Runtime(format!(
                    "Expand source `{}` is not bound in the input row",
                    source
                )))
            }
        };

        // Back-reference: pull the existing target id once. Only paths
        // ending at this id survive emission.
        let existing_target_id: Option<NodeId> = if back_reference {
            match arena.lookup_binding(parent_leaf, target_alias) {
                Some(RuntimeValue::Node(n)) => Some(n.id),
                Some(RuntimeValue::Null) => None,
                other => {
                    return Err(ExecError::Runtime(format!(
                        "Expand back-reference target `{}` is not a Node (got {:?})",
                        target_alias, other
                    )))
                }
            }
        } else {
            None
        };

        let mut hop_outputs_for_this_input: Vec<crate::exec::FactorIdx> = Vec::new();
        let mut matched_any = false;

        if min == 0 {
            // hop=0 — emit a leaf with the source itself as target (or
            // skip the target binding entirely under back_reference).
            let mut slots: Vec<Slot> = Vec::with_capacity(2);
            if let Some(name) = &rel_arc {
                slots.push(Slot {
                    name: name.clone(),
                    value: RuntimeValue::Null,
                });
            }
            // Hop 0 binds the source as the far end, so it must carry every
            // target label (mirrors the flat-path zero-hop fix).
            let source_has_target_labels =
                match arena.lookup_binding(parent_leaf, source) {
                    Some(RuntimeValue::Node(n)) => {
                        target_labels.iter().all(|l| n.labels.contains(l))
                    }
                    _ => target_labels.is_empty(),
                };
            if !back_reference {
                if let Some(RuntimeValue::Node(n)) = arena.lookup_binding(parent_leaf, source) {
                    slots.push(Slot {
                        name: target_arc.clone(),
                        value: RuntimeValue::Node(n.clone()),
                    });
                }
            }
            let zero_keeps = source_has_target_labels
                && match existing_target_id {
                    Some(existing) => starting == existing,
                    None => true,
                };
            if zero_keeps {
                let new_idx = arena.push(parent_leaf, slots);
                hop_outputs_for_this_input.push(new_idx);
                matched_any = true;
            }
        }

        // Frontier holds (factor_idx_for_this_hop, tail_id). At each hop
        // we expand neighbours and push a new FactorNode under the prior
        // frontier entry, NOT under the original parent_leaf. That keeps
        // per-step bindings (rel and intermediate target_alias) attached
        // to the correct chain for variable-length paths.
        // Third tuple element: relationships traversed on this path, as stored
        // edge identities `(edge_type, src, dst)`, for Cypher relationship
        // uniqueness (trail semantics). Only populated for multi-hop
        // expansions; empty on the single-hop path where reuse is impossible.
        let mut frontier: Vec<FactorFrontierEntry> =
            vec![(parent_leaf, starting, Vec::new())];

        for hop in 1..=max {
            let mut next_frontier: Vec<FactorFrontierEntry> = Vec::new();
            // Phase 1: pre-collect neighbours per frontier entry so the
            // batch prewarm below can populate L1 with one SST decode
            // (Fix #3b — same rationale as the flat path).
            let mut step_neighbours: Vec<(FactorFrontierEntry, Vec<EdgeView>)> =
                Vec::with_capacity(frontier.len());
            let mut unique_targets: Vec<NodeId> = Vec::new();
            let mut seen_targets: std::collections::HashSet<NodeId> =
                std::collections::HashSet::new();
            for (cur_parent, tail, rels) in frontier.drain(..) {
                let neighbours =
                    neighbours_of_any(snapshot, &edge_types, direction, tail, want_properties)
                        .await?;
                if !back_reference && !skip_target_materialize {
                    for edge in &neighbours {
                        let tid = partner_id(edge, direction, tail);
                        if seen_targets.insert(tid) {
                            unique_targets.push(tid);
                        }
                    }
                }
                step_neighbours.push(((cur_parent, tail, rels), neighbours));
            }
            // Phase 2: batch prewarm.
            if !back_reference && !skip_target_materialize && !unique_targets.is_empty() {
                if let Some(label) = target_labels.first() {
                    let _ = snapshot.batch_lookup_nodes(label, &unique_targets).await?;
                }
            }
            for ((cur_parent, tail, rels), neighbours) in step_neighbours {
                for edge in neighbours {
                    let target_id = partner_id(&edge, direction, tail);
                    // Cypher relationship uniqueness (trail semantics): skip an
                    // edge already traversed on this path so `-[:R*2..2]-` can't
                    // walk one edge out and back. Only enforced for multi-hop.
                    let edge_key = if max > 1 {
                        Some((edge.edge_type.clone(), edge.src, edge.dst))
                    } else {
                        None
                    };
                    if let Some(k) = &edge_key {
                        if rels.contains(k) {
                            continue;
                        }
                    }
                    // Far-end label(s) gate RESULTS, not traversal: for a
                    // multi-hop expansion, traverse through any node and let
                    // `target_is_result` decide the hit (see the flat-path fix).
                    let mut target_is_result = true;
                    let target_view_opt = if back_reference {
                        None
                    } else if skip_target_materialize {
                        // Fix #3: transit-only binding, see flat-path comment.
                        None
                    } else if let Some(label) = target_labels.first() {
                        if max > 1 {
                            match scan_node_for_id(snapshot, target_id).await? {
                                Some(v) => {
                                    target_is_result =
                                        target_labels.iter().all(|l| v.labels.contains(l));
                                    Some(v)
                                }
                                None => continue,
                            }
                        } else {
                            // Single hop: label mismatch excludes the edge.
                            // Conjunctive multi-label: must carry EVERY label.
                            match snapshot.lookup_node(label, target_id).await? {
                                Some(v) if target_labels.iter().all(|l| v.labels.contains(l)) => {
                                    Some(v)
                                }
                                _ => continue,
                            }
                        }
                    } else {
                        match scan_node_for_id(snapshot, target_id).await? {
                            Some(v) => Some(v),
                            None => continue,
                        }
                    };
                    let mut slots: Vec<Slot> = Vec::with_capacity(2);
                    if let Some(name) = &rel_arc {
                        slots.push(Slot {
                            name: name.clone(),
                            value: RuntimeValue::Rel(Box::new(RelValue::from(edge))),
                        });
                    }
                    if let Some(view) = target_view_opt {
                        slots.push(Slot {
                            name: target_arc.clone(),
                            value: RuntimeValue::Node(Box::new(NodeValue::from(view))),
                        });
                    } else if skip_target_materialize && !back_reference {
                        // id-only stub for the next Expand's `.id` read.
                        slots.push(Slot {
                            name: target_arc.clone(),
                            value: RuntimeValue::Node(Box::new(NodeValue {
                                id: target_id,
                                labels: target_labels.iter().map(|l| l.to_string()).collect(),
                                properties: std::collections::BTreeMap::new(),
                            })),
                        });
                    }
                    // One arena push per (parent, edge) pair. NO Row clone.
                    let new_idx = arena.push(cur_parent, slots);
                    let mut new_rels = rels.clone();
                    if let Some(k) = edge_key {
                        new_rels.push(k);
                    }
                    next_frontier.push((new_idx, target_id, new_rels));
                    if hop >= min.max(1) {
                        let keeps = target_is_result
                            && match existing_target_id {
                                Some(existing) => target_id == existing,
                                None => true,
                            };
                        if keeps {
                            hop_outputs_for_this_input.push(new_idx);
                            matched_any = true;
                        }
                    }
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        if matched_any {
            out_leaves.append(&mut hop_outputs_for_this_input);
        } else if optional {
            // Emit a leaf with NULL bindings for rel and target so OPTIONAL
            // MATCH semantics survive.
            let mut slots: Vec<Slot> = Vec::with_capacity(2);
            if let Some(name) = &rel_arc {
                slots.push(Slot {
                    name: name.clone(),
                    value: RuntimeValue::Null,
                });
            }
            if !back_reference {
                slots.push(Slot {
                    name: target_arc.clone(),
                    value: RuntimeValue::Null,
                });
            }
            out_leaves.push(arena.push(parent_leaf, slots));
        }
    }

    Ok(FactorRowSet {
        arena,
        leaves: out_leaves,
    })
}

// Worst-case optimal multiway join (RFC-024).
//
// Walks `ordering` left to right. At each level, builds the candidate
// list for the current variable by:
//
// - level 0: scan_label over the outer variable's label, with predicates
//   pushed down to storage.
// - level >0: leapfrog-intersect the `sorted_partners` lists of every
//   already-bound variable connected to this one by an `EdgeConstraint`.
//   Each surviving id is dereferenced via `lookup_node` so we can apply
//   the label filter and the in-binding predicates before recursing.
//
// At the bottom of the descent the executor pushes a single FactorNode
// under the arena root carrying one Slot per variable. No Row clones
// in the inner loop — only the final emit allocates.
fn execute_multiway_join_factor<'a>(
    vars: &'a [NodeBinding],
    edges: &'a [EdgeConstraint],
    ordering: &'a [usize],
    snapshot: &'a Snapshot<'_>,
) -> BoxFuture<'a, Result<FactorRowSet, ExecError>> {
    async move {
        namidb_core::profile_scope!("walker::execute_multiway_join_factor");
        if vars.is_empty() {
            return Ok(FactorRowSet::singleton_root());
        }
        if ordering.len() != vars.len() {
            return Err(ExecError::Runtime(format!(
                "MultiwayJoin: ordering length {} does not match vars length {}",
                ordering.len(),
                vars.len(),
            )));
        }
        let mut state = MultiwayState {
            arena: FactorArena::new(),
            leaves: Vec::new(),
            bound: vec![None; vars.len()],
            materialised: vec![None; vars.len()],
        };
        descend_multiway(&mut state, 0, vars, edges, ordering, snapshot).await?;
        Ok(FactorRowSet {
            arena: state.arena,
            leaves: state.leaves,
        })
    }
    .boxed()
}

struct MultiwayState {
    arena: FactorArena,
    leaves: Vec<FactorIdx>,
    bound: Vec<Option<NodeId>>,
    materialised: Vec<Option<NodeValue>>,
}

fn descend_multiway<'a>(
    state: &'a mut MultiwayState,
    level: usize,
    vars: &'a [NodeBinding],
    edges: &'a [EdgeConstraint],
    ordering: &'a [usize],
    snapshot: &'a Snapshot<'_>,
) -> BoxFuture<'a, Result<(), ExecError>> {
    async move {
        if level == ordering.len() {
            // Per-tuple multiplicity to match Cypher's per-path semantics
            // (and the binary executor): for each constraint, count how
            // many edges actually exist between the bound endpoints
            // across the constraint's alternation set, then emit
            // `prod_e mult_e` copies of the leaf. With single-type
            // single-edge constraints the multiplicity is 1 and the
            // arena ends up with one leaf per tuple (the original WCOJ
            // set-semantics behaviour); with `[:A|:B]` or parallel
            // edges of the same type the row count tracks the binary
            // path's `Vec<EdgeView>` fan-out exactly.
            let mut copies: usize = 1;
            for e in edges {
                let from = state.bound[e.from_idx].ok_or_else(|| {
                    ExecError::Runtime("MultiwayJoin: from_idx not bound at leaf level".into())
                })?;
                let to = state.bound[e.to_idx].ok_or_else(|| {
                    ExecError::Runtime("MultiwayJoin: to_idx not bound at leaf level".into())
                })?;
                let m =
                    count_edge_multiplicity(snapshot, from, to, &e.edge_types, e.direction).await?;
                copies = copies.saturating_mul(m);
                if copies == 0 {
                    // The leapfrog said this constraint's partner list
                    // contained `to`, but a re-scan of the SST/CSR
                    // found zero edges. The only way this happens is a
                    // tombstone slipped between the two reads — drop
                    // the tuple, which matches what the binary path
                    // would do.
                    return Ok(());
                }
            }
            // Build the leaf bindings once; push `copies` references
            // into the leaf list. `materialize_all` walks each leaf
            // index independently, so pushing the same index N times
            // yields N identical rows without N times the arena work.
            let root = state.arena.root();
            let mut slots: Vec<Slot> = Vec::with_capacity(vars.len());
            for (i, v) in vars.iter().enumerate() {
                let value = state
                    .materialised
                    .get(i)
                    .and_then(|m| m.clone())
                    .ok_or_else(|| {
                        ExecError::Runtime(format!(
                            "MultiwayJoin: variable `{}` not materialised at leaf level",
                            v.alias
                        ))
                    })?;
                slots.push(Slot::new(
                    Arc::<str>::from(v.alias.as_str()),
                    RuntimeValue::Node(Box::new(value)),
                ));
            }
            let leaf = state.arena.push(root, slots);
            for _ in 0..copies {
                state.leaves.push(leaf);
            }
            return Ok(());
        }

        let var_idx = ordering[level];
        let var = &vars[var_idx];

        // Gather the (bound_partner, storage_direction, edge_type) tuples
        // that constrain this variable to the prefix bound so far.
        // One source entry per (bound_neighbour, storage_direction,
        // edge_types_for_this_constraint). For a constraint with type
        // alternation `[:A|:B]` the executor fetches `sorted_partners`
        // once per listed type and merges them via `MergeSortedUnion`
        // into a single ascending list before the outer leapfrog
        // intersects across constraints.
        let mut sources: Vec<(NodeId, EdgeDirection, &[String])> = Vec::new();
        for e in edges {
            let (bound_idx, bound_is_from) =
                if e.from_idx == var_idx && state.bound[e.to_idx].is_some() {
                    (e.to_idx, false)
                } else if e.to_idx == var_idx && state.bound[e.from_idx].is_some() {
                    (e.from_idx, true)
                } else {
                    continue;
                };
            let dir = relationship_to_edge_direction(e.direction, bound_is_from)?;
            sources.push((
                state.bound[bound_idx].unwrap(),
                dir,
                e.edge_types.as_slice(),
            ));
        }

        if sources.is_empty() {
            // Level 0: outer scan. Pushes the label predicate to storage.
            if level != 0 {
                return Err(ExecError::Runtime(format!(
                    "MultiwayJoin: variable `{}` at level {} has no edge to a prior variable; \
                     planner emitted an unconnected ordering",
                    var.alias, level
                )));
            }
            let label = var.label.as_deref().ok_or_else(|| {
                ExecError::Runtime(format!(
                    "MultiwayJoin v0: outer variable `{}` requires a label",
                    var.alias
                ))
            })?;
            let nodes = snapshot
                .scan_label_with_predicates_and_projection(label, &var.predicates, None)
                .await?;
            for view in nodes {
                let id = view.id;
                state.bound[var_idx] = Some(id);
                state.materialised[var_idx] = Some(NodeValue::from(view));
                descend_multiway(state, level + 1, vars, edges, ordering, snapshot).await?;
            }
        } else {
            // Per-constraint partner list. For single-type constraints
            // we just call `sorted_partners` once; for alternation we
            // call it per type and union via `MergeSortedUnion` (the
            // output is sorted ascending without duplicates, which is
            // what `LeapfrogIntersect` needs).
            let mut lists: Vec<Vec<NodeId>> = Vec::with_capacity(sources.len());
            for (src, dir, edge_types) in &sources {
                if edge_types.len() == 1 {
                    lists.push(snapshot.sorted_partners(&edge_types[0], *src, *dir).await?);
                } else {
                    let mut per_type: Vec<Vec<NodeId>> = Vec::with_capacity(edge_types.len());
                    for et in *edge_types {
                        per_type.push(snapshot.sorted_partners(et, *src, *dir).await?);
                    }
                    let iters: Vec<SortedSliceIter<'_>> =
                        per_type.iter().map(|l| SortedSliceIter::new(l)).collect();
                    lists.push(crate::exec::MergeSortedUnion::new(iters).collect());
                }
            }
            let iters: Vec<SortedSliceIter<'_>> = lists
                .iter()
                .map(|l| SortedSliceIter::new(l.as_slice()))
                .collect();
            let mut lf = LeapfrogIntersect::new(iters);
            // Drain into a Vec up front so the borrow of `lists` ends
            // before the per-candidate await below moves us back to the
            // executor and would otherwise hold the slice across an await.
            let mut candidates: Vec<NodeId> = Vec::new();
            while let Some(k) = lf.key() {
                candidates.push(k);
                lf.next();
            }
            drop(lf);
            drop(lists);

            for cand_id in candidates {
                let view_opt = if let Some(label) = &var.label {
                    snapshot.lookup_node(label, cand_id).await?
                } else {
                    scan_node_for_id(snapshot, cand_id).await?
                };
                let view = match view_opt {
                    Some(v) => v,
                    None => continue,
                };
                if !var.predicates.is_empty() {
                    let matches = var.predicates.iter().all(|p| {
                        let val = view.properties.get(p.column());
                        eval_against_value(p, val)
                    });
                    if !matches {
                        continue;
                    }
                }
                state.bound[var_idx] = Some(cand_id);
                state.materialised[var_idx] = Some(NodeValue::from(view));
                descend_multiway(state, level + 1, vars, edges, ordering, snapshot).await?;
            }
        }

        state.bound[var_idx] = None;
        state.materialised[var_idx] = None;
        Ok(())
    }
    .boxed()
}

/// Count how many edges of any type in `edge_types` actually connect
/// `from` to `to` in the given direction. Used by the WCOJ leaf
/// emission to scale per-tuple set semantics back up to per-path
/// multiset semantics (the binary executor's native shape).
///
/// `sorted_partners` only tells us whether at least one edge of a
/// given type connects the two nodes; here we go through
/// `out_edges` / `in_edges`, which return `Vec<EdgeView>`, so
/// parallel edges of the same type contribute one count each. The
/// cost is `O(types * deg)` per call, paid only once per output
/// tuple — concretely, leaves are already the pruned cyclic
/// matches the leapfrog produced, so this dominates only in the
/// pathological case where the multiplicity per constraint is
/// huge anyway.
async fn count_edge_multiplicity(
    snapshot: &Snapshot<'_>,
    from: NodeId,
    to: NodeId,
    edge_types: &[String],
    direction: RelationshipDirection,
) -> Result<usize, ExecError> {
    let mut total: usize = 0;
    for et in edge_types {
        let edges = match direction {
            RelationshipDirection::Right => snapshot.out_edges(et, from).await?.edges,
            RelationshipDirection::Left => snapshot.in_edges(et, from).await?.edges,
            RelationshipDirection::Both => {
                return Err(ExecError::Runtime(
                    "MultiwayJoin v0: undirected edges not supported".into(),
                ));
            }
        };
        for e in edges {
            let partner = match direction {
                RelationshipDirection::Right => e.dst,
                RelationshipDirection::Left => e.src,
                RelationshipDirection::Both => unreachable!(),
            };
            if partner == to {
                total += 1;
            }
        }
    }
    Ok(total)
}

/// Translate the Cypher-side `RelationshipDirection` carried by an
/// `EdgeConstraint` into the storage-side `EdgeDirection` that
/// `Snapshot::sorted_partners` expects.
///
/// `bound_is_from = true` means the variable already bound at this
/// step matches the constraint's `from_idx`; the partner we want is
/// at `to_idx`. The conversion table:
///
/// | constraint dir | bound = from | bound = to |
/// |:-:|:-:|:-:|
/// | `Right` (`from -> to`) | `Forward` | `Inverse` |
/// | `Left`  (`from <- to`) | `Inverse` | `Forward` |
/// | `Both`                 | error in v0 |
fn relationship_to_edge_direction(
    dir: RelationshipDirection,
    bound_is_from: bool,
) -> Result<EdgeDirection, ExecError> {
    match (dir, bound_is_from) {
        (RelationshipDirection::Right, true) => Ok(EdgeDirection::Forward),
        (RelationshipDirection::Right, false) => Ok(EdgeDirection::Inverse),
        (RelationshipDirection::Left, true) => Ok(EdgeDirection::Inverse),
        (RelationshipDirection::Left, false) => Ok(EdgeDirection::Forward),
        (RelationshipDirection::Both, _) => Err(ExecError::Runtime(
            "MultiwayJoin v0: undirected edges (`-[]-`) are not supported; \
             cycle detection pass should canonicalise to Right or Left"
                .to_string(),
        )),
    }
}

// Cross product of two factorised sets. RFC-017 §3.2.2.
//
// For each (left_leaf, right_leaf) pair we materialise the right chain
// into a flat row of bindings, then push one "bridge" `FactorNode`
// under `left_leaf` carrying those bindings as slots. The left chain
// stays fully factorised; only the right side is flattened-per-pair.
//
// This is the bridge variant from RFC-017 §3.2.2 (vs. `splice_under`
// which would re-parent the entire right arena once and break per-pair
// independence). Bridge keeps the arena topologically simple — every
// chain still has one parent — at the cost of materialising the right
// row once per output leaf.
fn cross_product_factor(
    left: FactorRowSet,
    right: FactorRowSet,
) -> Result<FactorRowSet, ExecError> {
    if left.leaves.is_empty() || right.leaves.is_empty() {
        return Ok(FactorRowSet {
            arena: left.arena,
            leaves: Vec::new(),
        });
    }
    // Pre-check the multiplicative size before building, mirroring the flat
    // path, so a runaway cross product aborts instead of allocating first.
    crate::exec::limits::check_row_cap(left.leaves.len().saturating_mul(right.leaves.len()))?;
    let FactorRowSet {
        mut arena,
        leaves: left_leaves,
    } = left;

    let mut out_leaves = Vec::with_capacity(left_leaves.len() * right.leaves.len());
    for &l in &left_leaves {
        for &r in &right.leaves {
            let r_row = right.arena.materialize(r, None);
            let mut slots: Vec<Slot> = Vec::with_capacity(r_row.bindings.len());
            for (name, value) in r_row.bindings {
                // Skip names the left chain already binds — Cypher
                // cross-products only arise between disconnected pattern
                // parts, so the intersection should be empty, but be
                // defensive: `arena.lookup_binding` already returns the
                // ancestor value and shadowing would silently drop the
                // duplicate, so dropping here just avoids arena bloat.
                if arena.lookup_binding(l, &name).is_some() {
                    continue;
                }
                slots.push(Slot::new(Arc::from(name.as_str()), value));
            }
            out_leaves.push(arena.push(l, slots));
        }
    }
    Ok(FactorRowSet {
        arena,
        leaves: out_leaves,
    })
}

// Hash join: build side materialises to flat (the hash table requires
// indexable rows), probe side stays factorised. Per (probe_leaf,
// matched_build_row) we push one bridge FactorNode under `probe_leaf`
// with build's bindings as slots. RFC-017 §3.2.3.
//
// Residual predicate is evaluated on the materialised combined row;
// failing rows leave an orphan node in the arena (harmless — never
// referenced again) but do not appear in `out_leaves`.
#[allow(clippy::too_many_arguments)]
async fn hash_join_factor(
    build_plan: &LogicalPlan,
    probe_plan: &LogicalPlan,
    on: &[crate::plan::logical::JoinKey],
    residual: Option<&Expression>,
    snapshot: &Snapshot<'_>,
    params: &Params,
    outer: Option<&Row>,
    routing: &PlanRouting,
) -> Result<FactorRowSet, ExecError> {
    // Build phase: execute factor, materialise, hash by build_side key.
    let build_set =
        execute_factor_inner_with_routing(build_plan, snapshot, params, outer, routing).await?;
    let build_rows = build_set.materialize_all(None);
    crate::exec::limits::check_row_cap(build_rows.len())?;
    let mut table: std::collections::HashMap<Vec<String>, Vec<Row>> =
        std::collections::HashMap::new();
    for row in build_rows {
        let mut key = Vec::with_capacity(on.len());
        let mut has_null = false;
        for jk in on {
            let v = evaluate(&jk.build_side, &row, params)?;
            if matches!(v, RuntimeValue::Null) {
                has_null = true;
                break;
            }
            key.push(fingerprint_value(&v));
        }
        if has_null {
            continue;
        }
        table.entry(key).or_default().push(row);
    }

    // Probe phase: stream factor, look up, emit per matched build row.
    let probe_set =
        execute_factor_inner_with_routing(probe_plan, snapshot, params, outer, routing).await?;
    let FactorRowSet {
        mut arena,
        leaves: probe_leaves,
    } = probe_set;
    let mut out_leaves = Vec::new();
    for probe_leaf in probe_leaves {
        let probe_row = arena.materialize(probe_leaf, None);
        let mut key = Vec::with_capacity(on.len());
        let mut has_null = false;
        for jk in on {
            let v = evaluate(&jk.probe_side, &probe_row, params)?;
            if matches!(v, RuntimeValue::Null) {
                has_null = true;
                break;
            }
            key.push(fingerprint_value(&v));
        }
        if has_null {
            continue;
        }
        if let Some(matches) = table.get(&key) {
            for brow in matches {
                let mut slots: Vec<Slot> = Vec::with_capacity(brow.bindings.len());
                for (name, value) in &brow.bindings {
                    if arena.lookup_binding(probe_leaf, name).is_some() {
                        continue;
                    }
                    slots.push(Slot::new(Arc::from(name.as_str()), value.clone()));
                }
                let new_idx = arena.push(probe_leaf, slots);
                if let Some(res) = residual {
                    let combined = arena.materialize(new_idx, None);
                    if let RuntimeValue::Bool(true) = evaluate(res, &combined, params)? {
                        out_leaves.push(new_idx);
                    }
                // else: orphan node; arena holds it but we don't expose
                // a leaf to it. Acceptable for v0 — would matter only
                // for memory-bound queries that exercise this path
                // intensively.
                } else {
                    out_leaves.push(new_idx);
                }
            }
        }
    }
    Ok(FactorRowSet {
        arena,
        leaves: out_leaves,
    })
}

// Hash semi-join: outer survives, inner is a set lookup. No bindings
// flow from inner to outer (asymmetric — RFC-016 documented the no-swap
// rule). Implementation mirrors the flat path: build inner key_set
// (plus residual_index if residual present), probe outer, retain by the
// (matched XOR negated) truth table. No new FactorNodes added — just
// `leaves` filtering. RFC-017 §3.2.3.
#[allow(clippy::too_many_arguments)]
async fn hash_semi_join_factor(
    outer_plan: &LogicalPlan,
    inner_plan: &LogicalPlan,
    on: &[crate::plan::logical::JoinKey],
    negated: bool,
    residual: Option<&Expression>,
    snapshot: &Snapshot<'_>,
    params: &Params,
    outer: Option<&Row>,
    routing: &PlanRouting,
) -> Result<FactorRowSet, ExecError> {
    // Build inner: factor → materialise → hash by build_side.
    let inner_set =
        execute_factor_inner_with_routing(inner_plan, snapshot, params, outer, routing).await?;
    let inner_rows = inner_set.materialize_all(None);
    crate::exec::limits::check_row_cap(inner_rows.len())?;
    let mut key_set: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
    let mut residual_index: std::collections::HashMap<Vec<String>, Vec<Row>> =
        std::collections::HashMap::new();
    for row in inner_rows {
        let mut key = Vec::with_capacity(on.len());
        let mut has_null = false;
        for jk in on {
            let v = evaluate(&jk.build_side, &row, params)?;
            if matches!(v, RuntimeValue::Null) {
                has_null = true;
                break;
            }
            key.push(fingerprint_value(&v));
        }
        if has_null {
            continue;
        }
        if residual.is_some() {
            residual_index.entry(key.clone()).or_default().push(row);
        }
        key_set.insert(key);
    }

    // Probe outer: factor stream, retain leaf iff truth table says so.
    let outer_set =
        execute_factor_inner_with_routing(outer_plan, snapshot, params, outer, routing).await?;
    let FactorRowSet {
        arena,
        leaves: outer_leaves,
    } = outer_set;
    let mut out_leaves = Vec::with_capacity(outer_leaves.len());
    for outer_leaf in outer_leaves {
        let outer_row = arena.materialize(outer_leaf, None);
        let mut key = Vec::with_capacity(on.len());
        let mut has_null = false;
        for jk in on {
            let v = evaluate(&jk.probe_side, &outer_row, params)?;
            if matches!(v, RuntimeValue::Null) {
                has_null = true;
                break;
            }
            key.push(fingerprint_value(&v));
        }
        let matched = if has_null {
            false
        } else if let Some(res) = residual {
            match residual_index.get(&key) {
                Some(irows) => {
                    let mut any = false;
                    for irow in irows {
                        let mut combined = irow.clone();
                        for (k, v) in &outer_row.bindings {
                            combined.bindings.insert(k.clone(), v.clone());
                        }
                        if let RuntimeValue::Bool(true) = evaluate(res, &combined, params)? {
                            any = true;
                            break;
                        }
                    }
                    any
                }
                None => false,
            }
        } else {
            key_set.contains(&key)
        };
        let keep = if negated { !matched } else { matched };
        if keep {
            out_leaves.push(outer_leaf);
        }
    }
    Ok(FactorRowSet {
        arena,
        leaves: out_leaves,
    })
}

// Helper: collect the set of variable bindings an Expression
// reads, so factor sinks can build a "thin" row carrying only those
// bindings instead of materialising the full chain per leaf.
//
// Conservative: any expression form that closes over its own pattern
// (Exists, ListComprehension, PatternComprehension) contributes nothing
// — these forms are handled by the SemiApply / PatternList branches in
// `execute_factor_inner`, not by the sink. `Star` and `Parameter` and
// `Literal` similarly read no host binding.
fn collect_referenced_variables(expr: &Expression, out: &mut BTreeSet<String>) {
    use crate::parser::ExpressionKind;
    match &expr.kind {
        ExpressionKind::Variable(ident) => {
            out.insert(ident.name.to_string());
        }
        ExpressionKind::Property(pa) => {
            collect_referenced_variables(&pa.target, out);
        }
        ExpressionKind::Index { target, index } => {
            collect_referenced_variables(target, out);
            collect_referenced_variables(index, out);
        }
        ExpressionKind::Range { target, from, to } => {
            collect_referenced_variables(target, out);
            if let Some(e) = from {
                collect_referenced_variables(e, out);
            }
            if let Some(e) = to {
                collect_referenced_variables(e, out);
            }
        }
        ExpressionKind::Unary { expr, .. } => collect_referenced_variables(expr, out),
        ExpressionKind::Binary { left, right, .. } => {
            collect_referenced_variables(left, out);
            collect_referenced_variables(right, out);
        }
        ExpressionKind::In { item, list } => {
            collect_referenced_variables(item, out);
            collect_referenced_variables(list, out);
        }
        ExpressionKind::StringTest {
            target, pattern, ..
        } => {
            collect_referenced_variables(target, out);
            collect_referenced_variables(pattern, out);
        }
        ExpressionKind::IsNull { expr, .. } => collect_referenced_variables(expr, out),
        ExpressionKind::FunctionCall { args, .. } => {
            for arg in args {
                collect_referenced_variables(arg, out);
            }
        }
        ExpressionKind::Case {
            scrutinee,
            branches,
            otherwise,
        } => {
            if let Some(s) = scrutinee {
                collect_referenced_variables(s, out);
            }
            for b in branches {
                collect_referenced_variables(&b.when, out);
                collect_referenced_variables(&b.then, out);
            }
            if let Some(o) = otherwise {
                collect_referenced_variables(o, out);
            }
        }
        ExpressionKind::List(items) => {
            for e in items {
                collect_referenced_variables(e, out);
            }
        }
        ExpressionKind::Map(map) => {
            for (_k, v) in &map.entries {
                collect_referenced_variables(v, out);
            }
        }
        // Closed pattern forms — the binding reads they perform live
        // inside their own sub-plan, not in the host expression.
        ExpressionKind::Exists(_)
        | ExpressionKind::ExistsSubquery(_)
        | ExpressionKind::ListComprehension(_)
        | ExpressionKind::PatternComprehension(_)
        | ExpressionKind::Quantifier(_)
        | ExpressionKind::Literal(_)
        | ExpressionKind::Parameter(_)
        | ExpressionKind::Star => {}
    }
}

// ─────────────────── Plan-aware routing ─────────────────────
//
// RFC-018 §4 documented a properties caveat: with the slim CSR
// adjacency on, `EdgeView.properties` for SST-sourced edges comes back
// empty. The mitigation promised here is that the walker inspects the
// query plan; any `Expand` whose `rel_alias` is read downstream
// (whether as `r` or as `r.prop`) routes through the full-property SST
// path on a per-call-site basis, leaving the CSR path for the strictly
// topology-only majority of edge traversals.
//
// The analysis is a single visit at every public entry point. It
// reuses [`collect_referenced_variables`] for the per-expression work
// — `r.prop` already lands `r` in the set, which is exactly the
// invariant we need (whole-rel returns must also see populated
// properties, so they take the SST path too).
//
// Second invariant: the same set drives **target-materialise skipping**
// for chained Expands (Fix #3 — closes cold IC09 by removing the
// per-edge `lookup_node` on intermediate hops). A target_alias that
// never appears in any expression (RETURN / WHERE / ORDER BY /
// projection / join key / aggregation) is only ever read by the next
// Expand's `source` lookup, which uses only `NodeValue.id`. We can
// therefore stub the binding with an id-only `NodeValue` and avoid the
// SST decode entirely. Correctness is preserved when the
// `(edge_type, direction, target_label)` triple is schema-guaranteed
// (the dst/src label of the edge matches the declared target label) —
// any edge surfacing through neighbours_of_any then points at a node
// that *is* of the expected label.
#[derive(Debug, Default)]
pub(crate) struct PlanRouting {
    referenced_aliases: BTreeSet<String>,
}

impl PlanRouting {
    pub(crate) fn analyze(plan: &LogicalPlan) -> Self {
        let mut refs: BTreeSet<String> = BTreeSet::new();
        collect_plan_referenced_variables(plan, &mut refs);
        Self {
            referenced_aliases: refs,
        }
    }

    pub(crate) fn needs_properties(&self, rel_alias: Option<&str>) -> bool {
        match rel_alias {
            None => false,
            Some(a) => self.referenced_aliases.contains(a),
        }
    }

    /// `true` ⇔ `alias` is read anywhere in the plan — RETURN, WHERE,
    /// ORDER BY, projection, join keys, aggregation args, etc. Bare
    /// `Variable(alias)` and `Property(alias, k)` both count.
    ///
    /// An Expand whose `target_alias` returns `false` here is "transit
    /// only": the next Expand reads its `.id`, nothing else, so we can
    /// skip materialising the NodeView entirely (Fix #3).
    pub(crate) fn references(&self, alias: &str) -> bool {
        self.referenced_aliases.contains(alias)
    }
}

fn collect_plan_referenced_variables(plan: &LogicalPlan, out: &mut BTreeSet<String>) {
    use crate::plan::logical::{AggregateExpr, CreateElement, RemoveOp};

    match plan {
        LogicalPlan::Filter { predicate, .. } => {
            collect_referenced_variables(predicate, out);
        }
        LogicalPlan::Project { items, .. } => {
            for it in items {
                collect_referenced_variables(&it.expression, out);
            }
        }
        LogicalPlan::TopN { keys, .. } => {
            for k in keys {
                collect_referenced_variables(&k.expression, out);
            }
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregations,
            ..
        } => {
            for (e, _alias) in group_by {
                collect_referenced_variables(e, out);
            }
            for (_alias, agg) in aggregations {
                match agg {
                    AggregateExpr::Count { arg: Some(e), .. }
                    | AggregateExpr::Sum { arg: e, .. }
                    | AggregateExpr::Avg { arg: e, .. }
                    | AggregateExpr::Min { arg: e }
                    | AggregateExpr::Max { arg: e }
                    | AggregateExpr::Collect { arg: e, .. } => {
                        collect_referenced_variables(e, out);
                    }
                    AggregateExpr::Count { arg: None, .. } => {}
                }
            }
        }
        LogicalPlan::NodeById { id, .. } => {
            collect_referenced_variables(id, out);
        }
        LogicalPlan::NodeByPropertyValue { value, .. } => {
            collect_referenced_variables(value, out);
        }
        LogicalPlan::VectorSearch { query, .. } => {
            collect_referenced_variables(query, out);
        }
        LogicalPlan::CallProcedure { args, .. } => {
            for a in args {
                collect_referenced_variables(a, out);
            }
        }
        LogicalPlan::Unwind { list, .. } => {
            collect_referenced_variables(list, out);
        }
        LogicalPlan::Foreach { list, .. } => {
            // The body (a child) is walked by the generic children() recursion
            // below; here we only add the list's outer references.
            collect_referenced_variables(list, out);
        }
        LogicalPlan::HashJoin { on, residual, .. } => {
            for k in on {
                collect_referenced_variables(&k.build_side, out);
                collect_referenced_variables(&k.probe_side, out);
            }
            if let Some(r) = residual {
                collect_referenced_variables(r, out);
            }
        }
        LogicalPlan::HashSemiJoin { on, residual, .. } => {
            for k in on {
                collect_referenced_variables(&k.build_side, out);
                collect_referenced_variables(&k.probe_side, out);
            }
            if let Some(r) = residual {
                collect_referenced_variables(r, out);
            }
        }
        LogicalPlan::PatternList { projection, .. } => {
            collect_referenced_variables(projection, out);
        }
        LogicalPlan::Create { elements, .. } => {
            for el in elements {
                match el {
                    CreateElement::Node { properties, .. }
                    | CreateElement::Rel { properties, .. } => {
                        for (_k, e) in properties {
                            collect_referenced_variables(e, out);
                        }
                    }
                }
            }
        }
        LogicalPlan::Merge {
            pattern,
            on_match_sets,
            on_create_sets,
            ..
        } => {
            for el in pattern {
                match el {
                    CreateElement::Node { properties, .. }
                    | CreateElement::Rel { properties, .. } => {
                        for (_k, e) in properties {
                            collect_referenced_variables(e, out);
                        }
                    }
                }
            }
            for s in on_match_sets.iter().chain(on_create_sets.iter()) {
                visit_set_op(s, out);
            }
        }
        LogicalPlan::Set { items, .. } => {
            for s in items {
                visit_set_op(s, out);
            }
        }
        LogicalPlan::Remove { items, .. } => {
            for r in items {
                match r {
                    RemoveOp::Property { target_alias, .. }
                    | RemoveOp::Labels { target_alias, .. } => {
                        out.insert(target_alias.clone());
                    }
                }
            }
        }
        LogicalPlan::Delete { targets, .. } => {
            for e in targets {
                collect_referenced_variables(e, out);
            }
        }
        LogicalPlan::Expand { .. }
        | LogicalPlan::Distinct { .. }
        | LogicalPlan::Union { .. }
        | LogicalPlan::CrossProduct { .. }
        | LogicalPlan::SemiApply { .. }
        | LogicalPlan::Apply { .. }
        | LogicalPlan::NodeScan { .. }
        | LogicalPlan::Empty
        | LogicalPlan::Argument { .. }
        | LogicalPlan::MultiwayJoin { .. }
        | LogicalPlan::EdgeTypeCount { .. } => {}
    }

    for child in plan.children() {
        collect_plan_referenced_variables(child, out);
    }
}

fn visit_set_op(s: &crate::plan::logical::SetOp, out: &mut BTreeSet<String>) {
    use crate::plan::logical::SetOp;
    match s {
        SetOp::Property {
            target_alias,
            value,
            ..
        }
        | SetOp::Replace {
            target_alias,
            value,
        }
        | SetOp::Merge {
            target_alias,
            value,
        } => {
            out.insert(target_alias.clone());
            collect_referenced_variables(value, out);
        }
        SetOp::Labels { target_alias, .. } => {
            out.insert(target_alias.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use namidb_core::id::NodeId;

    #[test]
    fn node_id_from_string_value() {
        let id = NodeId::new();
        let v = RuntimeValue::String(id.to_string());
        let parsed = node_id_from_value(&v, SourceSpan::point(0)).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn fingerprint_is_content_faithful_and_collision_free() {
        // Distinct same-length vectors must NOT share a fingerprint (the old
        // length-only encoding collapsed them → wrong count(DISTINCT)/GROUP BY).
        let v1 = RuntimeValue::Vector(vec![1.0, 2.0]);
        let v2 = RuntimeValue::Vector(vec![3.0, 4.0]);
        assert_ne!(fingerprint_value(&v1), fingerprint_value(&v2));
        assert_eq!(fingerprint_value(&v1), fingerprint_value(&RuntimeValue::Vector(vec![1.0, 2.0])));

        // Distinct same-length byte strings (e.g. two 2-byte hashes).
        let b1 = RuntimeValue::Bytes(vec![0x01, 0x02]);
        let b2 = RuntimeValue::Bytes(vec![0x03, 0x04]);
        assert_ne!(fingerprint_value(&b1), fingerprint_value(&b2));

        // Floats differing past the 10th decimal must be distinct.
        let f1 = RuntimeValue::Float(1.000_000_000_01);
        let f2 = RuntimeValue::Float(1.000_000_000_02);
        assert_ne!(fingerprint_value(&f1), fingerprint_value(&f2));
        // +0.0 and -0.0 compare equal, so they must share a fingerprint.
        assert_eq!(
            fingerprint_value(&RuntimeValue::Float(0.0)),
            fingerprint_value(&RuntimeValue::Float(-0.0))
        );

        // Separator collision: a string containing the old separators must not
        // collide with a differently-structured list.
        let l1 = RuntimeValue::List(vec![RuntimeValue::String("a,s:b".into())]);
        let l2 = RuntimeValue::List(vec![
            RuntimeValue::String("a".into()),
            RuntimeValue::String("b".into()),
        ]);
        assert_ne!(fingerprint_value(&l1), fingerprint_value(&l2));
    }

    #[test]
    fn node_id_from_non_uuid_string_errors() {
        let v = RuntimeValue::String("not-a-uuid".into());
        let err = node_id_from_value(&v, SourceSpan::point(0)).unwrap_err();
        assert!(matches!(err, ExecError::Eval(_)));
    }

    #[test]
    fn fingerprint_distinguishes_distinct_values() {
        let a = fingerprint_value(&RuntimeValue::Integer(1));
        let b = fingerprint_value(&RuntimeValue::Integer(2));
        let c = fingerprint_value(&RuntimeValue::String("1".into()));
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn sum_integers_yields_integer() {
        let v = sum_values(&[RuntimeValue::Integer(1), RuntimeValue::Integer(2)]).unwrap();
        assert_eq!(v, RuntimeValue::Integer(3));
    }

    #[test]
    fn sum_mixed_promotes_to_float() {
        let v = sum_values(&[RuntimeValue::Integer(1), RuntimeValue::Float(2.5)]).unwrap();
        assert_eq!(v, RuntimeValue::Float(3.5));
    }
}
