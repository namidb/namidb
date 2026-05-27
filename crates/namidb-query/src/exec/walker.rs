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
    AggregateExpr, EdgeConstraint, LogicalPlan, NodeBinding, OrderKey, ProjectionItem,
};

/// Top-level error produced by the executor. Wraps `EvalError`,
/// storage errors and structural runtime mismatches.
#[derive(Debug)]
pub enum ExecError {
    Eval(EvalError),
    Storage(namidb_storage::Error),
    Runtime(String),
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecError::Eval(e) => write!(f, "{}", e),
            ExecError::Storage(e) => write!(f, "storage: {}", e),
            ExecError::Runtime(m) => write!(f, "runtime: {}", m),
        }
    }
}

impl std::error::Error for ExecError {}

impl From<EvalError> for ExecError {
    fn from(e: EvalError) -> Self {
        ExecError::Eval(e)
    }
}

impl From<namidb_storage::Error> for ExecError {
    fn from(e: namidb_storage::Error) -> Self {
        ExecError::Storage(e)
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
    Ok(set.materialize_all(None))
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

            LogicalPlan::Create { .. }
            | LogicalPlan::Merge { .. }
            | LogicalPlan::Set { .. }
            | LogicalPlan::Remove { .. }
            | LogicalPlan::Delete { .. } => Err(ExecError::Runtime(
                "write operators require execute_write(plan, &mut WriterSession, params)"
                    .to_string(),
            )),

            LogicalPlan::MultiwayJoin { .. } => Err(ExecError::Runtime(
                "MultiwayJoin requires the factorised executor; \
                 set NAMIDB_FACTORIZE=1 (RFC-024)"
                    .to_string(),
            )),

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
                    if let Some(view) = snapshot.lookup_node(label, node_id).await? {
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
            } => {
                let input_rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                let mut out = Vec::with_capacity(input_rows.len());
                for row in input_rows {
                    let lookup_val = evaluate(value, &row, params)?;
                    if let Some(view) =
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
                let mut rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                if !keys.is_empty() {
                    sort_rows(&mut rows, keys, params)?;
                }
                let skip = *skip as usize;
                if skip >= rows.len() {
                    return Ok(Vec::new());
                }
                let mut iter = rows.into_iter().skip(skip);
                let take = if *limit == u64::MAX {
                    usize::MAX
                } else {
                    *limit as usize
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
                target_label,
                length,
                optional,
                back_reference,
                shortest,
                path_binding,
            } => {
                let rows =
                    execute_inner_with_routing(input, snapshot, params, outer, routing).await?;
                execute_expand(
                    rows,
                    source,
                    edge_type.as_deref(),
                    *direction,
                    rel_alias.as_deref(),
                    target_alias,
                    target_label.as_deref(),
                    *length,
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
                        target_label.as_deref(),
                        *length,
                        *back_reference,
                    ),
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
        };
        if let Some(start) = profile_start {
            if let Ok(rows) = &result {
                crate::profile::record_op(plan, start.elapsed(), rows.len() as u64);
            }
        }
        result
    }
    .boxed()
}

// ───────────────────────── Expand ────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn execute_expand(
    rows: Vec<Row>,
    source: &str,
    edge_type: Option<&[String]>,
    direction: RelationshipDirection,
    rel_alias: Option<&str>,
    target_alias: &str,
    target_label: Option<&str>,
    length: Option<crate::parser::RelationshipLength>,
    optional: bool,
    back_reference: bool,
    shortest: crate::plan::ShortestMode,
    path_binding: Option<&str>,
    snapshot: &Snapshot<'_>,
    want_properties: bool,
    skip_target_materialize: bool,
) -> Result<Vec<Row>, ExecError> {
    namidb_core::profile_scope!("walker::execute_expand");
    let edge_types = resolve_edge_types(snapshot, edge_type);
    let min = length.map(|l| l.min).unwrap_or(1);
    let max = length.map(|l| l.max).unwrap_or(1);

    let mut out = Vec::new();
    for row in rows {
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
            let zero_keeps = match existing_target_id {
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
                if let Some(label) = target_label {
                    let _ = snapshot.batch_lookup_nodes(label, &unique_targets).await?;
                }
            }
            for (step, neighbours) in step_neighbours {
                for edge in neighbours {
                    let target_id = partner_id(&edge, direction, step.tail);
                    // Back-reference fast path: skip the lookup_node
                    // (the binding's NodeView is already on the row).
                    // For non-back-reference, fetch the view so we
                    // can populate / label-filter.
                    let target_view_opt = if back_reference {
                        None
                    } else if skip_target_materialize {
                        // Fix #3: the binding is "transit only" — the next
                        // Expand reads only `.id`. Skip the SST decode and
                        // synthesise an id-only stub below. Schema-guaranteed
                        // dst_label means no correctness drift vs the
                        // `continue`-on-None branch below.
                        None
                    } else if let Some(label) = target_label {
                        match snapshot.lookup_node(label, target_id).await? {
                            Some(v) => Some(v),
                            None => continue,
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
                                label: target_label.unwrap_or_default().to_string(),
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
                    next_frontier.push(Step {
                        tail: target_id,
                        row: new_row.clone(),
                        trail: new_trail.clone(),
                    });
                    if hop >= min.max(1) {
                        let keeps = match existing_target_id {
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
                out.extend(snapshot.in_edges_via_sst(edge_type, node).await?.edges);
                Ok(out)
            }
        };
    }
    match direction {
        RelationshipDirection::Right => Ok(snapshot.out_edges(edge_type, node).await?.edges),
        RelationshipDirection::Left => Ok(snapshot.in_edges(edge_type, node).await?.edges),
        RelationshipDirection::Both => {
            let mut out = snapshot.out_edges(edge_type, node).await?.edges;
            out.extend(snapshot.in_edges(edge_type, node).await?.edges);
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
    target_label: Option<&str>,
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
    let Some(target_label) = target_label else {
        // Without target_label the legacy path uses `scan_node_for_id`
        // to confirm the id resolves to *some* node. We could try
        // harder (e.g. require the schema-declared dst_label to be a
        // singleton across all edge_types), but the conservative gate
        // covers IC09 / IC02 already.
        return false;
    };
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
async fn scan_node_for_id(
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

fn fingerprint_value(v: &RuntimeValue) -> String {
    match v {
        RuntimeValue::Null => "<null>".into(),
        RuntimeValue::Bool(b) => b.to_string(),
        RuntimeValue::Integer(n) => format!("i:{}", n),
        RuntimeValue::Float(f) => format!("f:{:.10}", f),
        RuntimeValue::String(s) => format!("s:{}", s),
        RuntimeValue::List(items) => {
            let mut s = "[".to_string();
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&fingerprint_value(it));
            }
            s.push(']');
            s
        }
        RuntimeValue::Map(m) => {
            let mut s = "{".to_string();
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(k);
                s.push(':');
                s.push_str(&fingerprint_value(v));
            }
            s.push('}');
            s
        }
        RuntimeValue::Node(n) => format!("n:{}", n.id),
        RuntimeValue::Rel(r) => format!("r:{}:{}->{}", r.edge_type, r.src, r.dst),
        RuntimeValue::Date(d) => format!("d:{}", d),
        RuntimeValue::DateTime(d) => format!("dt:{}", d),
        RuntimeValue::Bytes(b) => format!("b:{}", b.len()),
        RuntimeValue::Vector(v) => format!("v:{}", v.len()),
        RuntimeValue::Path(items) => {
            let mut s = "p:[".to_string();
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&fingerprint_value(it));
            }
            s.push(']');
            s
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
            .map_err(ExecError::Storage);
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
            .map_err(ExecError::Storage)?
    } else {
        snapshot
            .scan_label(label)
            .await
            .map_err(ExecError::Storage)?
    };
    Ok(candidates.into_iter().next())
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
        match plan {
            // Operators that benefit directly: keep everything factorised.
            LogicalPlan::Empty => Ok(FactorRowSet::singleton_root()),

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
                    if let Some(view) = snapshot.lookup_node(label, node_id).await? {
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
                    if let Some(view) =
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
                target_label,
                length,
                optional,
                back_reference,
                shortest: _, // factor path: shortestPath routes via the flat path; see Plan-aware routing.
                path_binding: _,
            } => {
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;
                execute_expand_factor(
                    input_set,
                    source,
                    edge_type.as_deref(),
                    *direction,
                    rel_alias.as_deref(),
                    target_alias,
                    target_label.as_deref(),
                    *length,
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
                        target_label.as_deref(),
                        *length,
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
                Ok(cross_product_factor(l, r))
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
                let input_set =
                    execute_factor_inner_with_routing(input, snapshot, params, outer, routing)
                        .await?;

                // Empty keys: stable order, just skip+take + materialise.
                if keys.is_empty() {
                    let skip = *skip as usize;
                    if skip >= input_set.cardinality() {
                        return Ok(FactorRowSet::from_flat(Vec::new()));
                    }
                    let take = if *limit == u64::MAX {
                        usize::MAX
                    } else {
                        *limit as usize
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

                let skip = *skip as usize;
                if skip >= keyed.len() {
                    return Ok(FactorRowSet::from_flat(Vec::new()));
                }
                let take = if *limit == u64::MAX {
                    usize::MAX
                } else {
                    *limit as usize
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
            | LogicalPlan::Delete { .. } => Err(ExecError::Runtime(
                "write operators require execute_write(plan, &mut WriterSession, params)"
                    .to_string(),
            )),

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
    target_label: Option<&str>,
    length: Option<crate::parser::RelationshipLength>,
    optional: bool,
    back_reference: bool,
    snapshot: &Snapshot<'_>,
    want_properties: bool,
    skip_target_materialize: bool,
) -> Result<FactorRowSet, ExecError> {
    namidb_core::profile_scope!("walker::execute_expand_factor");
    let edge_types = resolve_edge_types(snapshot, edge_type);
    let min = length.map(|l| l.min).unwrap_or(1);
    let max = length.map(|l| l.max).unwrap_or(1);

    let FactorRowSet {
        mut arena,
        leaves: input_leaves,
    } = input;
    let target_arc: Arc<str> = Arc::from(target_alias);
    let rel_arc: Option<Arc<str>> = rel_alias.map(Arc::from);

    let mut out_leaves: Vec<crate::exec::FactorIdx> = Vec::new();

    for parent_leaf in input_leaves {
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
            if !back_reference {
                if let Some(RuntimeValue::Node(n)) = arena.lookup_binding(parent_leaf, source) {
                    slots.push(Slot {
                        name: target_arc.clone(),
                        value: RuntimeValue::Node(n.clone()),
                    });
                }
            }
            let zero_keeps = match existing_target_id {
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
        let mut frontier: Vec<(crate::exec::FactorIdx, NodeId)> = vec![(parent_leaf, starting)];

        for hop in 1..=max {
            let mut next_frontier: Vec<(crate::exec::FactorIdx, NodeId)> = Vec::new();
            // Phase 1: pre-collect neighbours per frontier entry so the
            // batch prewarm below can populate L1 with one SST decode
            // (Fix #3b — same rationale as the flat path).
            let mut step_neighbours: Vec<((crate::exec::FactorIdx, NodeId), Vec<EdgeView>)> =
                Vec::with_capacity(frontier.len());
            let mut unique_targets: Vec<NodeId> = Vec::new();
            let mut seen_targets: std::collections::HashSet<NodeId> =
                std::collections::HashSet::new();
            for (cur_parent, tail) in frontier.drain(..) {
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
                step_neighbours.push(((cur_parent, tail), neighbours));
            }
            // Phase 2: batch prewarm.
            if !back_reference && !skip_target_materialize && !unique_targets.is_empty() {
                if let Some(label) = target_label {
                    let _ = snapshot.batch_lookup_nodes(label, &unique_targets).await?;
                }
            }
            for ((cur_parent, tail), neighbours) in step_neighbours {
                for edge in neighbours {
                    let target_id = partner_id(&edge, direction, tail);
                    let target_view_opt = if back_reference {
                        None
                    } else if skip_target_materialize {
                        // Fix #3: transit-only binding, see flat-path comment.
                        None
                    } else if let Some(label) = target_label {
                        match snapshot.lookup_node(label, target_id).await? {
                            Some(v) => Some(v),
                            None => continue,
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
                                label: target_label.unwrap_or_default().to_string(),
                                properties: std::collections::BTreeMap::new(),
                            })),
                        });
                    }
                    // One arena push per (parent, edge) pair. NO Row clone.
                    let new_idx = arena.push(cur_parent, slots);
                    next_frontier.push((new_idx, target_id));
                    if hop >= min.max(1) {
                        let keeps = match existing_target_id {
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
fn cross_product_factor(left: FactorRowSet, right: FactorRowSet) -> FactorRowSet {
    if left.leaves.is_empty() || right.leaves.is_empty() {
        return FactorRowSet {
            arena: left.arena,
            leaves: Vec::new(),
        };
    }
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
    FactorRowSet {
        arena,
        leaves: out_leaves,
    }
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
        ExpressionKind::Between { target, low, high } => {
            collect_referenced_variables(target, out);
            collect_referenced_variables(low, out);
            collect_referenced_variables(high, out);
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
        | ExpressionKind::ListComprehension(_)
        | ExpressionKind::PatternComprehension(_)
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
        LogicalPlan::Unwind { list, .. } => {
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
        | LogicalPlan::NodeScan { .. }
        | LogicalPlan::Empty
        | LogicalPlan::Argument { .. }
        | LogicalPlan::MultiwayJoin { .. } => {}
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
