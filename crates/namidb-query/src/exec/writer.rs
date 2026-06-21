//! Write-aware executor.
//!
//! Sister to [`crate::exec::walker::execute`]. Drives a [`LogicalPlan`]
//! containing write operators (Create / Merge / Set / Remove / Delete)
//! against a mutable [`WriterSession`], delegating read sub-plans back to
//! the read-only walker. Auto-commits at the end of the query.
//!
//! See [`docs/rfc/009-write-clauses.md`](../../../../docs/rfc/009-write-clauses.md).
//!
//! Read-your-own-writes (RFC-026): read sub-plans run against
//! [`WriterSession::overlay_snapshot`], so a `MATCH`/`MERGE`-match/unique
//! check that follows a `CREATE` in the same statement or transaction sees
//! the staged rows. v1 wires node reads; staged edges are not yet visible
//! to traversals (RFC-026 Q1, a fast follow).
//!
//! Limitations of v0:
//!
//! - MERGE matches by single-element pattern (one node or
//! node-rel-node chain).
//! - DETACH DELETE enumerates incident edges across every edge_type
//! declared on the manifest schema.
//! - Property values must be representable as `core::Value` scalars
//! (List/Map/Node/Rel are rejected with an explicit error).

use std::collections::BTreeMap;
use std::time::Instant;

use futures::future::BoxFuture;
use futures::FutureExt;
use namidb_core::id::NodeId;
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NodeWriteRecord, WriterSession};

use super::expr::{evaluate, Params};
use super::row::Row;
use super::value::{NodeValue, RelValue, RuntimeValue};
use super::walker::{execute_inner, ExecError};
use crate::parser::{Expression, RelationshipDirection};
use crate::plan::logical::{CreateElement, LogicalPlan, RemoveOp, SetOp};

/// Result of a write-path execution.
#[derive(Debug, Clone, Default)]
pub struct WriteOutcome {
    pub rows: Vec<Row>,
    pub nodes_created: u64,
    pub edges_created: u64,
    pub nodes_deleted: u64,
    pub edges_deleted: u64,
    pub properties_set: u64,
    /// Labels added (`SET n:L`) or removed (`REMOVE n:L`).
    pub labels_set: u64,
}

/// Execute `plan` against `writer`, staging its mutations into the
/// writer's pending batch but NOT committing. The caller must then either
/// `writer.commit_batch()` to make the batch durable or
/// `writer.discard_batch()` to roll it back. Used by explicit Bolt
/// transactions, which stage several statements and commit once at COMMIT.
/// The RETURN rows are computed during the apply, so they are available
/// before the commit; a later read sub-plan in the same statement sees the
/// staged batch through the read-your-own-writes overlay (RFC-026).
pub async fn execute_write_staged(
    plan: &LogicalPlan,
    writer: &mut WriterSession,
    params: &Params,
) -> Result<WriteOutcome, ExecError> {
    execute_write_staged_with_deadline(plan, writer, params, None).await
}

/// [`execute_write_staged`] with a wall-clock `deadline`. The deadline rides
/// the shared [`namidb_storage::cancel`] task-local for the whole apply, so
/// the storage decode loops a read sub-plan drives and the per-row / per-edge
/// write loops here both probe it and abort a runaway statement mid-flight
/// (cooperative cancellation) with [`ExecError::Timeout`]. `None` runs
/// unguarded with the baseline cost — the server passes a deadline, embedded
/// callers and tests do not.
pub async fn execute_write_staged_with_deadline(
    plan: &LogicalPlan,
    writer: &mut WriterSession,
    params: &Params,
    deadline: Option<Instant>,
) -> Result<WriteOutcome, ExecError> {
    let mut outcome = WriteOutcome::default();
    let rows = crate::exec::limits::with_limits(
        deadline,
        None,
        execute_write_inner(plan, writer, params, &mut outcome),
    )
    .await?;
    outcome.rows = rows;
    Ok(outcome)
}

/// Execute `plan` against `writer` and commit. Auto-commit mode: each call
/// is its own transaction. Reads pin a snapshot per read sub-plan; writes
/// go through `writer.upsert_*` / `tombstone_*` and `writer.commit_batch()`
/// makes them durable. For a multi-statement explicit transaction use
/// [`execute_write_staged`] and commit once at the end.
pub async fn execute_write(
    plan: &LogicalPlan,
    writer: &mut WriterSession,
    params: &Params,
) -> Result<WriteOutcome, ExecError> {
    execute_write_with_deadline(plan, writer, params, None).await
}

/// [`execute_write`] with a wall-clock `deadline` bounding the apply. A write
/// that overruns is aborted with [`ExecError::Timeout`] before
/// `commit_batch`, so the pending batch is discarded and nothing partial is
/// sealed — the writer is left clean for the next statement. `None` runs
/// unbounded.
pub async fn execute_write_with_deadline(
    plan: &LogicalPlan,
    writer: &mut WriterSession,
    params: &Params,
    deadline: Option<Instant>,
) -> Result<WriteOutcome, ExecError> {
    let outcome = match execute_write_staged_with_deadline(plan, writer, params, deadline).await {
        Ok(outcome) => outcome,
        Err(e) => {
            // The statement failed (a timeout counts) after staging some
            // mutations into the pending batch (writers are long-lived and
            // shared, so the batch outlives this call). Drop them, or the
            // next write on this writer would seal them with its own commit.
            writer.discard_batch();
            return Err(e);
        }
    };
    writer.commit_batch().await.map_err(ExecError::Storage)?;
    Ok(outcome)
}

fn execute_write_inner<'a>(
    plan: &'a LogicalPlan,
    writer: &'a mut WriterSession,
    params: &'a Params,
    outcome: &'a mut WriteOutcome,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    async move {
        match plan {
            // ─── Write operators ────────────────────────────────────
            LogicalPlan::Create { input, elements } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let new_row = apply_create(elements, row, writer, params, outcome).await?;
                    out.push(new_row);
                }
                Ok(out)
            }

            LogicalPlan::Set { input, items } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let new_row = apply_sets(items, row, writer, params, outcome).await?;
                    out.push(new_row);
                }
                Ok(out)
            }

            LogicalPlan::Remove { input, items } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let new_row = apply_removes(items, row, writer, outcome)?;
                    out.push(new_row);
                }
                Ok(out)
            }

            LogicalPlan::Delete {
                input,
                targets,
                detach,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    apply_delete(targets, *detach, &row, writer, params, outcome).await?;
                    out.push(row);
                }
                Ok(out)
            }

            LogicalPlan::Merge {
                input,
                pattern,
                on_match_sets,
                on_create_sets,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                let mut out = Vec::with_capacity(rows.len().max(1));
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let merged = apply_merge(
                        pattern,
                        on_match_sets,
                        on_create_sets,
                        row,
                        writer,
                        params,
                        outcome,
                    )
                    .await?;
                    out.extend(merged);
                }
                Ok(out)
            }

            LogicalPlan::Foreach {
                input,
                variable,
                list,
                body,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                for row in &rows {
                    crate::exec::limits::check_deadline()?;
                    let items = match evaluate(list, row, params)? {
                        RuntimeValue::List(items) => items,
                        RuntimeValue::Null => continue,
                        v => {
                            return Err(ExecError::Runtime(format!(
                                "FOREACH requires a list; got {}",
                                v.type_name()
                            )));
                        }
                    };
                    for item in items {
                        let mut seed = row.clone();
                        seed.set(variable.clone(), item);
                        exec_foreach_body(body, writer, params, outcome, &seed).await?;
                    }
                }
                // FOREACH is side-effect only: the input rows pass through
                // unchanged so any following clause keeps the same cardinality.
                Ok(rows)
            }

            // ─── Read operators that may wrap a write child: handle
            // row-wise here so the write semantics run on the child first.
            LogicalPlan::Project {
                input,
                items,
                distinct,
                discard_input_bindings,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                let mut projected = crate::exec::walker::project_rows(
                    &rows,
                    items,
                    *discard_input_bindings,
                    params,
                )?;
                if *distinct {
                    projected = crate::exec::walker::dedup_rows(projected);
                }
                Ok(projected)
            }
            LogicalPlan::Filter { input, predicate } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let v = evaluate(predicate, &row, params)?;
                    if v.as_bool() == Some(true) {
                        out.push(row);
                    }
                }
                Ok(out)
            }
            LogicalPlan::TopN {
                input,
                keys,
                skip,
                limit,
            } => {
                let skip = crate::exec::walker::resolve_row_count(skip, params, "SKIP")?;
                let limit = crate::exec::walker::resolve_row_count(limit, params, "LIMIT")?;
                let mut rows = execute_write_inner(input, writer, params, outcome).await?;
                if !keys.is_empty() {
                    crate::exec::walker::sort_rows(&mut rows, keys, params)?;
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
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                Ok(crate::exec::walker::dedup_rows(rows))
            }
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregations,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
                crate::exec::walker::execute_aggregate(rows, group_by, aggregations, params)
            }
            LogicalPlan::Unwind { input, list, alias } => {
                let rows = execute_write_inner(input, writer, params, outcome).await?;
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
                        RuntimeValue::Null => {}
                        _ => {
                            return Err(ExecError::Runtime(format!(
                                "UNWIND requires a list; got {}",
                                v.type_name()
                            )));
                        }
                    }
                }
                Ok(out)
            }
            LogicalPlan::Union { left, right, all } => {
                let mut l = execute_write_inner(left, writer, params, outcome).await?;
                let r = execute_write_inner(right, writer, params, outcome).await?;
                l.extend(r);
                if *all {
                    Ok(l)
                } else {
                    Ok(crate::exec::walker::dedup_rows(l))
                }
            }
            LogicalPlan::CrossProduct { left, right } => {
                let l = execute_write_inner(left, writer, params, outcome).await?;
                let r = execute_write_inner(right, writer, params, outcome).await?;
                Ok(crate::exec::walker::cross_product(l, r))
            }

            LogicalPlan::HashJoin { .. } | LogicalPlan::HashSemiJoin { .. } => {
                // HashJoin and HashSemiJoin are read-only (their rewriters
                // never touch subtrees that contain writes). In a write
                // path we delegate to the post-write snapshot reader so
                // the executor lives in exactly one place.
                let snap = writer.overlay_snapshot();
                crate::exec::walker::execute_inner(plan, &snap, params, None).await
            }

            LogicalPlan::EdgeTypeCount { .. } => {
                // Read-only leaf: delegate to the post-write snapshot reader.
                let snap = writer.overlay_snapshot();
                crate::exec::walker::execute_inner(plan, &snap, params, None).await
            }

            // ─── NodeById can have a write-bearing input (e.g. CREATE
            // ... WITH p MATCH (f:Person {id: $fid}) ...). Recurse on
            // the input via execute_write_inner so writes commit, then
            // perform the lookup against the post-write snapshot.
            LogicalPlan::NodeById {
                input,
                label,
                alias,
                id,
            } => {
                let input_rows = execute_write_inner(input, writer, params, outcome).await?;
                let snap = writer.overlay_snapshot();
                let mut out = Vec::with_capacity(input_rows.len());
                for row in input_rows {
                    let id_value = evaluate(id, &row, params)?;
                    let node_id = crate::exec::walker::node_id_from_value(&id_value, id.span)?;
                    let found = match label {
                        Some(l) => snap
                            .lookup_node(l, node_id)
                            .await
                            .map_err(ExecError::Storage)?,
                        None => crate::exec::walker::scan_node_for_id(&snap, node_id).await?,
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

            // Same shape as NodeById: writes commit first, then the
            // unique-property lookup runs against the post-write snapshot.
            LogicalPlan::NodeByPropertyValue {
                input,
                label,
                alias,
                property,
                value,
                multi,
            } => {
                let input_rows = execute_write_inner(input, writer, params, outcome).await?;
                let snap = writer.overlay_snapshot();
                let mut out = Vec::with_capacity(input_rows.len());
                for row in input_rows {
                    let lookup_val = evaluate(value, &row, params)?;
                    if *multi {
                        for view in crate::exec::walker::lookup_nodes_by_property_via_scan(
                            &snap,
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
                        crate::exec::walker::lookup_node_by_property_via_scan(
                            &snap,
                            label,
                            property,
                            &lookup_val,
                        )
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

            // ─── An Expand whose input stages writes (RFC-026 Q1):
            // `CREATE (a)-[:R]->(b) WITH a MATCH (a)-[:R]->(x) RETURN x`. The
            // input subtree carries the CREATE, so it cannot go to the
            // read-only walker (which rejects embedded writes). Recurse the
            // input through the write executor to stage the mutations and
            // materialise the source rows, then run the traversal step against
            // the read-your-own-writes overlay so the just-staged edge is
            // visible. A pure-read Expand still falls to the read-leaf arm
            // below. `want_properties = true` / `skip_target_materialize =
            // false`: the routing optimisation that prunes those is a read-only
            // walker concern, so materialise fully here (correct, just not
            // pruned).
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
            } if input.contains_write() => {
                let input_rows = execute_write_inner(input, writer, params, outcome).await?;
                let snap = writer.overlay_snapshot();
                crate::exec::walker::execute_expand(
                    input_rows,
                    source,
                    edge_type.as_deref(),
                    *direction,
                    rel_alias.as_deref(),
                    target_alias,
                    target_labels,
                    *length,
                    *optional,
                    *back_reference,
                    *shortest,
                    path_binding.as_deref(),
                    &snap,
                    true,
                    false,
                    None,
                )
                .await
            }

            // ─── Pure read leaves and pattern-driven operators that do
            // NOT contain writes: delegate to the read-only walker on a
            // freshly pinned snapshot. v0: no read-your-own-writes.
            LogicalPlan::Empty
            | LogicalPlan::NodeScan { .. }
            | LogicalPlan::Argument { .. }
            | LogicalPlan::Expand { .. }
            | LogicalPlan::SemiApply { .. }
            | LogicalPlan::PatternList { .. }
            | LogicalPlan::MultiwayJoin { .. }
            | LogicalPlan::VectorSearch { .. }
            | LogicalPlan::CallProcedure { .. } => {
                let snap = writer.overlay_snapshot();
                execute_inner(plan, &snap, params, None).await
            }
        }
    }
    .boxed()
}

/// Execute a FOREACH body for one element, seeded with `seed` (the per-element
/// row carrying the loop variable + outer bindings). The body is a chain of
/// updating operators bottoming at an `Empty`/`Argument` leaf, which here yields
/// `seed`. Returns the produced rows (used only to thread bindings through a
/// multi-clause body); the caller discards them.
fn exec_foreach_body<'a>(
    plan: &'a LogicalPlan,
    writer: &'a mut WriterSession,
    params: &'a Params,
    outcome: &'a mut WriteOutcome,
    seed: &'a Row,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    async move {
        match plan {
            // The leaf: the per-element seed row.
            LogicalPlan::Empty | LogicalPlan::Argument { .. } => Ok(vec![seed.clone()]),
            LogicalPlan::Create { input, elements } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    out.push(apply_create(elements, row, writer, params, outcome).await?);
                }
                Ok(out)
            }
            LogicalPlan::Set { input, items } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    out.push(apply_sets(items, row, writer, params, outcome).await?);
                }
                Ok(out)
            }
            LogicalPlan::Remove { input, items } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    out.push(apply_removes(items, row, writer, outcome)?);
                }
                Ok(out)
            }
            LogicalPlan::Delete {
                input,
                targets,
                detach,
            } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                for row in &rows {
                    apply_delete(targets, *detach, row, writer, params, outcome).await?;
                }
                Ok(rows)
            }
            LogicalPlan::Merge {
                input,
                pattern,
                on_match_sets,
                on_create_sets,
            } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = Vec::new();
                for row in rows {
                    out.extend(
                        apply_merge(
                            pattern,
                            on_match_sets,
                            on_create_sets,
                            row,
                            writer,
                            params,
                            outcome,
                        )
                        .await?,
                    );
                }
                Ok(out)
            }
            LogicalPlan::Foreach {
                input,
                variable,
                list,
                body,
            } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                for row in &rows {
                    let items = match evaluate(list, row, params)? {
                        RuntimeValue::List(items) => items,
                        RuntimeValue::Null => continue,
                        v => {
                            return Err(ExecError::Runtime(format!(
                                "FOREACH requires a list; got {}",
                                v.type_name()
                            )));
                        }
                    };
                    for item in items {
                        let mut inner = row.clone();
                        inner.set(variable.clone(), item);
                        exec_foreach_body(body, writer, params, outcome, &inner).await?;
                    }
                }
                Ok(rows)
            }
            other => Err(ExecError::Runtime(format!(
                "operator `{}` is not allowed in a FOREACH body",
                other.operator_name()
            ))),
        }
    }
    .boxed()
}

// ──────────────────────────── CREATE ─────────────────────────────────

/// Evaluate a `properties_spread` expression and merge its entries
/// into the `core_props` / `runtime_props` accumulators of a CREATE.
///
/// The expression must evaluate to a `Map`; anything else is an error
/// (most commonly the caller passed `$x` where `$x` is not a map).
/// `_id` keys are extracted into `explicit_id` rather than treated as
/// stored properties so the `CREATE (n:L $params)` idiom can still
/// pin a NodeId through the spread map.
fn apply_spread_properties(
    spread_expr: &Expression,
    row: &Row,
    params: &Params,
    core_props: &mut BTreeMap<String, CoreValue>,
    runtime_props: &mut BTreeMap<String, RuntimeValue>,
    explicit_id: &mut Option<NodeId>,
) -> Result<(), ExecError> {
    let value = evaluate(spread_expr, row, params)?;
    let map = match value {
        RuntimeValue::Map(m) => m,
        other => {
            return Err(ExecError::Runtime(format!(
                "properties spread expects a MAP, got {}",
                other.type_name()
            )));
        }
    };
    for (k, v) in map {
        if k == "_id" {
            *explicit_id = Some(crate::exec::walker::node_id_from_value(
                &v,
                spread_expr.span,
            )?);
            continue;
        }
        let core = runtime_to_core(&v, spread_expr).map_err(ExecError::Runtime)?;
        core_props.insert(k.clone(), core);
        runtime_props.insert(k, v);
    }
    Ok(())
}

/// Find a node, other than `exclude`, that already holds `value` for the
/// declared-unique property `prop` on `label`. The lookup runs against the
/// read-your-own-writes overlay (RFC-026), so a value staged earlier in the
/// same uncommitted statement/transaction is seen too.
///
/// A string value uses the `O(log N)` property index. Any other type falls
/// back to a label scan and a typed-value compare, because the persisted
/// property index keys on strings; this enforces non-string unique
/// constraints correctly, at a scan's cost per check (a typed index is a
/// later optimisation).
async fn find_unique_conflict(
    writer: &WriterSession,
    label: &str,
    prop: &str,
    value: &CoreValue,
    exclude: Option<NodeId>,
) -> Result<Option<NodeId>, ExecError> {
    let snap = writer.overlay_snapshot();
    let conflict = match value {
        CoreValue::Str(v) => snap
            .lookup_node_by_property(label, prop, v)
            .await
            .map_err(ExecError::Storage)?
            .map(|node| node.id)
            .filter(|id| Some(*id) != exclude),
        other => {
            let mut found = None;
            for node in snap.scan_label(label).await.map_err(ExecError::Storage)? {
                if Some(node.id) == exclude {
                    continue;
                }
                if node.properties.get(prop) == Some(other) {
                    found = Some(node.id);
                    break;
                }
            }
            found
        }
    };
    drop(snap);
    Ok(conflict)
}

/// Enforce declared unique constraints for a node about to be created. Each
/// label's unique properties (of any type) are checked against the
/// read-your-own-writes overlay (RFC-026), so a duplicate value staged earlier
/// in the same uncommitted statement/transaction is caught too, not just one
/// already committed. Returns [`ExecError::Constraint`] on the first duplicate.
async fn enforce_unique_on_create(
    writer: &WriterSession,
    labels: &[String],
    core_props: &BTreeMap<String, CoreValue>,
) -> Result<(), ExecError> {
    // Collect the (label, property, value) checks first so the borrow of the
    // schema is released before we take a snapshot.
    let checks: Vec<(String, String, CoreValue)> = {
        let schema = writer.schema();
        let mut checks = Vec::new();
        for label in labels {
            if let Some(def) = schema.label(label) {
                for prop in &def.properties {
                    if prop.unique {
                        if let Some(v) = core_props.get(&prop.name) {
                            checks.push((label.clone(), prop.name.clone(), v.clone()));
                        }
                    }
                }
            }
        }
        checks
    };
    for (label, prop, value) in checks {
        if find_unique_conflict(writer, &label, &prop, &value, None)
            .await?
            .is_some()
        {
            return Err(ExecError::Constraint(format!(
                "{label}.{prop} = {value:?} already exists (unique constraint)"
            )));
        }
    }
    Ok(())
}

/// Enforce declared unique constraints for a node about to be staged by a
/// caller OUTSIDE the Cypher executor (the Python low-level bulk API), against
/// the read-your-own-writes overlay. Returns the conflict message on the first
/// duplicate, mirroring the check `CREATE` already runs, so the low-level path
/// cannot silently commit duplicate unique-property values.
pub async fn enforce_node_unique_constraints(
    writer: &WriterSession,
    labels: &[String],
    core_props: &BTreeMap<String, CoreValue>,
) -> Result<(), String> {
    enforce_unique_on_create(writer, labels, core_props)
        .await
        .map_err(|e| match e {
            ExecError::Constraint(msg) => msg,
            other => other.to_string(),
        })
}

/// Enforce a unique constraint when SET assigns `value` to `key` on a node.
/// If `key` is a declared unique property on any of the node's labels and a
/// different node already holds `value`, reject. Setting the node's own
/// current value (self-update) is allowed. Values of any type are checked;
/// see [`find_unique_conflict`] for how string vs non-string is resolved.
async fn enforce_unique_on_set(
    writer: &WriterSession,
    labels: &[String],
    key: &str,
    value: &CoreValue,
    self_id: NodeId,
) -> Result<(), ExecError> {
    let unique_labels: Vec<String> = {
        let schema = writer.schema();
        labels
            .iter()
            .filter(|l| {
                schema.label(l).is_some_and(|d| {
                    d.properties
                        .iter()
                        .any(|p| p.name.as_str() == key && p.unique)
                })
            })
            .cloned()
            .collect()
    };
    for label in &unique_labels {
        // Read-your-own-writes overlay (RFC-026): a SET that follows a CREATE
        // in the same statement/transaction must see the staged row. The
        // node's own row is excluded via `self_id`, so a self-update (or a
        // no-op write of the same value) is allowed.
        if find_unique_conflict(writer, label, key, value, Some(self_id))
            .await?
            .is_some()
        {
            return Err(ExecError::Constraint(format!(
                "{label}.{key} = {value:?} already held by another node (unique constraint)"
            )));
        }
    }
    Ok(())
}

/// Enforce declared NOT NULL constraints for a node that is being created or
/// is gaining new labels. For each label in `labels`, every property the
/// schema declares `nullable = false` must be present in `core_props` with a
/// non-null value; a missing property and an explicit `NULL` are both
/// violations. Returns [`ExecError::Constraint`] on the first one.
///
/// Pure schema lookup, no snapshot read: declared NOT NULL is a property of
/// the row being staged, unlike the unique checks which must consult the
/// read-your-own-writes overlay. Node-only, mirroring `enforce_unique_*`
/// (edges carry no declared-property validation today).
fn enforce_notnull_on_create(
    writer: &WriterSession,
    labels: &[String],
    core_props: &BTreeMap<String, CoreValue>,
) -> Result<(), ExecError> {
    let schema = writer.schema();
    for label in labels {
        let Some(def) = schema.label(label) else {
            continue;
        };
        for prop in &def.properties {
            if prop.nullable {
                continue;
            }
            match core_props.get(&prop.name) {
                Some(v) if !matches!(v, CoreValue::Null) => {}
                Some(_) => {
                    return Err(ExecError::Constraint(format!(
                        "{label}.{} is declared NOT NULL but was set to null (not-null constraint)",
                        prop.name
                    )));
                }
                None => {
                    return Err(ExecError::Constraint(format!(
                        "{label}.{} is declared NOT NULL but is missing (not-null constraint)",
                        prop.name
                    )));
                }
            }
        }
    }
    Ok(())
}

/// The first of the node's `labels` that declares `key` as `nullable =
/// false`, if any. Shared by the SET-to-null and REMOVE not-null guards.
fn not_null_label(writer: &WriterSession, labels: &[String], key: &str) -> Option<String> {
    let schema = writer.schema();
    labels.iter().find_map(|label| {
        schema.label(label).and_then(|def| {
            def.properties
                .iter()
                .any(|p| p.name == key && !p.nullable)
                .then(|| label.clone())
        })
    })
}

async fn apply_create(
    elements: &[CreateElement],
    mut row: Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
    for elem in elements {
        match elem {
            CreateElement::Node {
                alias,
                labels,
                properties,
                properties_spread,
            } => {
                // Back-reference: don't create if already bound.
                if row.get(alias).is_some() {
                    continue;
                }
                let mut core_props = BTreeMap::new();
                let mut runtime_props = BTreeMap::new();
                let mut explicit_id: Option<NodeId> = None;
                // `properties_spread` is the runtime-evaluated map for
                // the `CREATE (n:L $params)` idiom. Apply it first so
                // explicit `properties` overwrite collisions, matching
                // the conventional spread semantics.
                if let Some(spread_expr) = properties_spread {
                    apply_spread_properties(
                        spread_expr,
                        &row,
                        params,
                        &mut core_props,
                        &mut runtime_props,
                        &mut explicit_id,
                    )?;
                }
                for (k, expr) in properties {
                    let v = evaluate(expr, &row, params)?;
                    if k == "_id" {
                        // `{_id: ...}` becomes the storage NodeId; not stored
                        // as a regular property. The `._id` accessor (and the
                        // `id(n)` Cypher function) materialise it on read.
                        // Plain `id` is now a user-owned property.
                        explicit_id = Some(crate::exec::walker::node_id_from_value(&v, expr.span)?);
                        continue;
                    }
                    let core = runtime_to_core(&v, expr).map_err(ExecError::Runtime)?;
                    core_props.insert(k.clone(), core);
                    runtime_props.insert(k.clone(), v);
                }
                let id = match explicit_id {
                    Some(id) => id,
                    None => NodeId::new(),
                };
                // Enforce declared unique constraints against the
                // read-your-own-writes overlay (RFC-026) before staging the
                // node, so a duplicate staged earlier in the same uncommitted
                // batch is caught as well as one already committed.
                enforce_unique_on_create(writer, labels, &core_props).await?;
                enforce_notnull_on_create(writer, labels, &core_props)?;
                let record = NodeWriteRecord {
                    properties: core_props,
                    schema_version: 1,
                    ..Default::default()
                };
                writer
                    .upsert_node_with_labels(labels.iter().cloned(), id, &record)
                    .map_err(ExecError::Storage)?;
                outcome.nodes_created += 1;
                let node_value = NodeValue {
                    id,
                    labels: labels.iter().cloned().collect(),
                    properties: runtime_props,
                };
                row.set(alias.clone(), RuntimeValue::Node(Box::new(node_value)));
            }
            CreateElement::Rel {
                alias,
                edge_type,
                source_alias,
                target_alias,
                direction,
                properties,
                properties_spread,
            } => {
                let src_id = expect_node_id(&row, source_alias)?;
                let dst_id = expect_node_id(&row, target_alias)?;
                let (src, dst) = match direction {
                    RelationshipDirection::Right => (src_id, dst_id),
                    RelationshipDirection::Left => (dst_id, src_id),
                    RelationshipDirection::Both => {
                        return Err(ExecError::Runtime(
                            "CREATE relationship must be directed".into(),
                        ));
                    }
                };
                let mut core_props = BTreeMap::new();
                let mut runtime_props = BTreeMap::new();
                if let Some(spread_expr) = properties_spread {
                    // `_id` only applies to node creates; edges have no
                    // user-visible id slot.
                    let mut ignored_id: Option<NodeId> = None;
                    apply_spread_properties(
                        spread_expr,
                        &row,
                        params,
                        &mut core_props,
                        &mut runtime_props,
                        &mut ignored_id,
                    )?;
                    if ignored_id.is_some() {
                        return Err(ExecError::Runtime(
                            "_id is not valid on a relationship CREATE".into(),
                        ));
                    }
                }
                for (k, expr) in properties {
                    let v = evaluate(expr, &row, params)?;
                    let core = runtime_to_core(&v, expr).map_err(ExecError::Runtime)?;
                    core_props.insert(k.clone(), core);
                    runtime_props.insert(k.clone(), v);
                }
                let record = EdgeWriteRecord {
                    properties: core_props,
                    schema_version: 1,
                };
                writer
                    .upsert_edge(edge_type.clone(), src, dst, &record)
                    .map_err(ExecError::Storage)?;
                outcome.edges_created += 1;
                if let Some(name) = alias {
                    let rel_value = RelValue {
                        edge_type: edge_type.clone(),
                        src,
                        dst,
                        properties: runtime_props,
                    };
                    row.set(name.clone(), RuntimeValue::Rel(Box::new(rel_value)));
                }
            }
        }
    }
    Ok(row)
}

// ──────────────────────────── SET ────────────────────────────────────

async fn apply_sets(
    items: &[SetOp],
    mut row: Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
    for op in items {
        row = apply_set(op, row, writer, params, outcome).await?;
    }
    Ok(row)
}

async fn apply_set(
    op: &SetOp,
    mut row: Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
    match op {
        SetOp::Property {
            target_alias,
            key,
            value,
        } => {
            let new_val = evaluate(value, &row, params)?;
            let core = runtime_to_core(&new_val, value).map_err(ExecError::Runtime)?;
            match row.get(target_alias).cloned() {
                Some(RuntimeValue::Node(mut n)) => {
                    // Enforce unique constraints if `key` is a declared unique
                    // property on one of the node's labels. Self-update (setting
                    // the node's own value) is allowed.
                    let label_vec: Vec<String> = n.labels.iter().cloned().collect();
                    enforce_unique_on_set(writer, &label_vec, key, &core, n.id).await?;
                    if matches!(core, CoreValue::Null) {
                        if let Some(label) = not_null_label(writer, &label_vec, key) {
                            return Err(ExecError::Constraint(format!(
                                "{label}.{key} is declared NOT NULL and cannot be set to null \
                                 (not-null constraint)"
                            )));
                        }
                    }
                    let mut core_props = node_runtime_props_to_core(&n.properties)?;
                    core_props.insert(key.clone(), core);
                    let record = NodeWriteRecord {
                        properties: core_props,
                        schema_version: 1,
                        ..Default::default()
                    };
                    // Preserve the full label set on a property update; the
                    // node is keyed by id, so re-upserting with one label would
                    // silently drop the others.
                    writer
                        .upsert_node_with_labels(n.labels.iter().cloned(), n.id, &record)
                        .map_err(ExecError::Storage)?;
                    n.properties.insert(key.clone(), new_val);
                    outcome.properties_set += 1;
                    row.set(target_alias.clone(), RuntimeValue::Node(n));
                }
                Some(RuntimeValue::Rel(mut r)) => {
                    let mut core_props = node_runtime_props_to_core(&r.properties)?;
                    core_props.insert(key.clone(), core);
                    let record = EdgeWriteRecord {
                        properties: core_props,
                        schema_version: 1,
                    };
                    writer
                        .upsert_edge(r.edge_type.clone(), r.src, r.dst, &record)
                        .map_err(ExecError::Storage)?;
                    r.properties.insert(key.clone(), new_val);
                    outcome.properties_set += 1;
                    row.set(target_alias.clone(), RuntimeValue::Rel(r));
                }
                Some(other) => {
                    return Err(ExecError::Runtime(format!(
                        "SET target `{}` must be a Node or Relationship, got {}",
                        target_alias,
                        other.type_name()
                    )));
                }
                None => {
                    return Err(ExecError::Runtime(format!(
                        "SET target `{}` is not bound",
                        target_alias
                    )));
                }
            }
        }
        SetOp::Replace {
            target_alias,
            value,
        } => {
            row = apply_set_map(true, target_alias, value, row, writer, params, outcome).await?;
        }
        SetOp::Merge {
            target_alias,
            value,
        } => {
            row = apply_set_map(false, target_alias, value, row, writer, params, outcome).await?;
        }
        SetOp::Labels {
            target_alias,
            labels,
        } => match row.get(target_alias).cloned() {
            Some(RuntimeValue::Node(mut n)) => {
                // Union the new labels into the node's set, then re-upsert
                // (keyed by id) so the row carries the full set.
                let added_labels: Vec<String> = labels
                    .iter()
                    .filter(|l| n.labels.insert((*l).clone()))
                    .cloned()
                    .collect();
                let core_props = node_runtime_props_to_core(&n.properties)?;
                // A newly-added label brings its own NOT NULL contract: the
                // node must already carry a non-null value for every property
                // that label declares non-null.
                enforce_notnull_on_create(writer, &added_labels, &core_props)?;
                let record = NodeWriteRecord {
                    properties: core_props,
                    schema_version: 1,
                    ..Default::default()
                };
                writer
                    .upsert_node_with_labels(n.labels.iter().cloned(), n.id, &record)
                    .map_err(ExecError::Storage)?;
                outcome.labels_set += added_labels.len() as u64;
                row.set(target_alias.clone(), RuntimeValue::Node(n));
            }
            other => {
                return Err(ExecError::Runtime(format!(
                    "SET {}:Label target must be a Node, got {:?}",
                    target_alias, other
                )));
            }
        },
    }
    Ok(row)
}

/// Compute the post-SET property set for a map-form SET. `replace` (`SET x =
/// {..}`) starts from an empty set; otherwise (`SET x += {..}`) it starts from
/// the current properties and merges. A `null` value removes its key in both
/// forms (openCypher property-removal semantics).
fn merged_props(
    replace: bool,
    current: &BTreeMap<String, RuntimeValue>,
    incoming: &[(String, RuntimeValue)],
) -> BTreeMap<String, RuntimeValue> {
    let mut out = if replace {
        BTreeMap::new()
    } else {
        current.clone()
    };
    for (k, v) in incoming {
        if matches!(v, RuntimeValue::Null) {
            out.remove(k);
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

/// Apply a map-form SET: `SET x = {..}` (replace, `replace = true`) or
/// `SET x += {..}` (merge). The right-hand side may be a map literal, a
/// `$param` map, or another node/relationship whose properties are copied.
/// `+= null` is a no-op and `= null` clears all properties, matching Neo4j.
/// Uniqueness and NOT NULL are enforced against the FINAL property set, so a
/// `=` that drops a NOT NULL column is rejected rather than silently committed.
async fn apply_set_map(
    replace: bool,
    target_alias: &str,
    value: &Expression,
    mut row: Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
    let incoming: Vec<(String, RuntimeValue)> = match evaluate(value, &row, params)? {
        RuntimeValue::Map(m) => m.into_iter().collect(),
        RuntimeValue::Node(n) => n.properties.into_iter().collect(),
        RuntimeValue::Rel(r) => r.properties.into_iter().collect(),
        RuntimeValue::Null => {
            if !replace {
                return Ok(row); // `+= null` is a no-op.
            }
            Vec::new() // `= null` clears all properties.
        }
        other => {
            return Err(ExecError::Runtime(format!(
                "SET {target_alias} {} requires a map, node, or relationship, got {}",
                if replace { "=" } else { "+=" },
                other.type_name()
            )));
        }
    };

    match row.get(target_alias).cloned() {
        Some(RuntimeValue::Node(mut n)) => {
            let final_runtime = merged_props(replace, &n.properties, &incoming);
            let final_core = node_runtime_props_to_core(&final_runtime)?;
            let labels: Vec<String> = n.labels.iter().cloned().collect();
            // Uniqueness against the final set, excluding the node's own row so
            // a self-update is allowed; then NOT NULL so a `=` that drops a
            // required column is rejected, not silently committed.
            for (k, cv) in &final_core {
                enforce_unique_on_set(writer, &labels, k, cv, n.id).await?;
            }
            enforce_notnull_on_create(writer, &labels, &final_core)?;
            let record = NodeWriteRecord {
                properties: final_core,
                schema_version: 1,
                ..Default::default()
            };
            writer
                .upsert_node_with_labels(n.labels.iter().cloned(), n.id, &record)
                .map_err(ExecError::Storage)?;
            outcome.properties_set += incoming.len() as u64;
            n.properties = final_runtime;
            row.set(target_alias.to_string(), RuntimeValue::Node(n));
        }
        Some(RuntimeValue::Rel(mut r)) => {
            let final_runtime = merged_props(replace, &r.properties, &incoming);
            let final_core = node_runtime_props_to_core(&final_runtime)?;
            let record = EdgeWriteRecord {
                properties: final_core,
                schema_version: 1,
            };
            writer
                .upsert_edge(r.edge_type.clone(), r.src, r.dst, &record)
                .map_err(ExecError::Storage)?;
            outcome.properties_set += incoming.len() as u64;
            r.properties = final_runtime;
            row.set(target_alias.to_string(), RuntimeValue::Rel(r));
        }
        Some(other) => {
            return Err(ExecError::Runtime(format!(
                "SET target `{target_alias}` must be a Node or Relationship, got {}",
                other.type_name()
            )));
        }
        None => {
            return Err(ExecError::Runtime(format!(
                "SET target `{target_alias}` is not bound"
            )));
        }
    }
    Ok(row)
}

// ──────────────────────────── REMOVE ─────────────────────────────────

fn apply_removes(
    items: &[RemoveOp],
    mut row: Row,
    writer: &mut WriterSession,
    outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
    for op in items {
        row = apply_remove(op, row, writer, outcome)?;
    }
    Ok(row)
}

fn apply_remove(
    op: &RemoveOp,
    mut row: Row,
    writer: &mut WriterSession,
    outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
    match op {
        RemoveOp::Property { target_alias, key } => match row.get(target_alias).cloned() {
            Some(RuntimeValue::Node(mut n)) => {
                let labels: Vec<String> = n.labels.iter().cloned().collect();
                if let Some(label) = not_null_label(writer, &labels, key) {
                    return Err(ExecError::Constraint(format!(
                        "{label}.{key} is declared NOT NULL and cannot be removed \
                         (not-null constraint)"
                    )));
                }
                let mut core_props = node_runtime_props_to_core(&n.properties)?;
                core_props.remove(key);
                let record = NodeWriteRecord {
                    properties: core_props,
                    schema_version: 1,
                    ..Default::default()
                };
                // Preserve the full label set on a property removal (node is
                // keyed by id; a single-label upsert would drop the others).
                writer
                    .upsert_node_with_labels(n.labels.iter().cloned(), n.id, &record)
                    .map_err(ExecError::Storage)?;
                n.properties.remove(key);
                outcome.properties_set += 1;
                row.set(target_alias.clone(), RuntimeValue::Node(n));
            }
            Some(RuntimeValue::Rel(mut r)) => {
                let mut core_props = node_runtime_props_to_core(&r.properties)?;
                core_props.remove(key);
                let record = EdgeWriteRecord {
                    properties: core_props,
                    schema_version: 1,
                };
                writer
                    .upsert_edge(r.edge_type.clone(), r.src, r.dst, &record)
                    .map_err(ExecError::Storage)?;
                r.properties.remove(key);
                outcome.properties_set += 1;
                row.set(target_alias.clone(), RuntimeValue::Rel(r));
            }
            other => {
                return Err(ExecError::Runtime(format!(
                    "REMOVE target `{}` must be a Node or Relationship, got {:?}",
                    target_alias, other
                )));
            }
        },
        RemoveOp::Labels {
            target_alias,
            labels,
        } => match row.get(target_alias).cloned() {
            Some(RuntimeValue::Node(mut n)) => {
                // Set difference, then re-upsert (keyed by id). A node may end
                // up with zero labels — Cypher permits unlabelled nodes.
                let removed = labels.iter().filter(|l| n.labels.remove(*l)).count();
                let record = NodeWriteRecord {
                    properties: node_runtime_props_to_core(&n.properties)?,
                    schema_version: 1,
                    ..Default::default()
                };
                writer
                    .upsert_node_with_labels(n.labels.iter().cloned(), n.id, &record)
                    .map_err(ExecError::Storage)?;
                outcome.labels_set += removed as u64;
                row.set(target_alias.clone(), RuntimeValue::Node(n));
            }
            other => {
                return Err(ExecError::Runtime(format!(
                    "REMOVE {}:Label target must be a Node, got {:?}",
                    target_alias, other
                )));
            }
        },
    }
    Ok(row)
}

// ──────────────────────────── DELETE ─────────────────────────────────

async fn apply_delete(
    targets: &[Expression],
    detach: bool,
    row: &Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<(), ExecError> {
    for target in targets {
        let v = evaluate(target, row, params)?;
        match v {
            RuntimeValue::Node(n) => {
                if detach {
                    detach_incident_edges(n.id, writer, outcome).await?;
                }
                // Tombstone is keyed by id; the label arg is vestigial (a
                // tombstone removes the node from every label scan). Pass any
                // carried label for diagnostics.
                let any_label = n.labels.iter().next().cloned().unwrap_or_default();
                writer
                    .tombstone_node(any_label, n.id)
                    .map_err(ExecError::Storage)?;
                outcome.nodes_deleted += 1;
            }
            RuntimeValue::Rel(r) => {
                writer
                    .tombstone_edge(r.edge_type.clone(), r.src, r.dst)
                    .map_err(ExecError::Storage)?;
                outcome.edges_deleted += 1;
            }
            RuntimeValue::Null => {
                // Cypher: DELETE NULL is a no-op.
            }
            other => {
                return Err(ExecError::Runtime(format!(
                    "DELETE target must be a Node, Relationship or NULL, got {}",
                    other.type_name()
                )));
            }
        }
    }
    Ok(())
}

async fn detach_incident_edges(
    node: NodeId,
    writer: &mut WriterSession,
    outcome: &mut WriteOutcome,
) -> Result<(), ExecError> {
    // Enumerate every edge_type declared on the manifest schema and
    // tombstone both directions. This is O(edge_types × incident_edges)
    // — acceptable for v0; see RFC-009 §Drawbacks.
    let edge_types: Vec<String> = writer.observed_edge_types();
    for et in edge_types {
        crate::exec::limits::check_deadline()?;
        let mut to_delete: Vec<(NodeId, NodeId)> = Vec::new();
        {
            let snap = writer.overlay_snapshot();
            let out_edges = snap
                .out_edges(&et, node)
                .await
                .map_err(ExecError::Storage)?;
            for e in &out_edges.edges {
                to_delete.push((e.src, e.dst));
            }
            let in_edges = snap.in_edges(&et, node).await.map_err(ExecError::Storage)?;
            for e in &in_edges.edges {
                to_delete.push((e.src, e.dst));
            }
        }
        for (i, (src, dst)) in to_delete.into_iter().enumerate() {
            // Probe on a stride: tombstoning is a cheap memtable insert, so an
            // `Instant::now()` per edge would show on a million-edge detach.
            // The bounded read above already probes during edge decode.
            if i % namidb_storage::cancel::CHECK_STRIDE == 0 {
                crate::exec::limits::check_deadline()?;
            }
            writer
                .tombstone_edge(et.clone(), src, dst)
                .map_err(ExecError::Storage)?;
            outcome.edges_deleted += 1;
        }
    }
    Ok(())
}

// ──────────────────────────── MERGE ──────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn apply_merge(
    pattern: &[CreateElement],
    on_match_sets: &[SetOp],
    on_create_sets: &[SetOp],
    row: Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<Vec<Row>, ExecError> {
    // v0: support a single Node pattern, or a Node-Rel-Node chain.
    let matches = find_merge_matches(pattern, &row, writer, params).await?;
    if !matches.is_empty() {
        let mut out = Vec::with_capacity(matches.len());
        for mut m_row in matches {
            for op in on_match_sets {
                m_row = apply_set(op, m_row, writer, params, outcome).await?;
            }
            out.push(m_row);
        }
        Ok(out)
    } else {
        // Create branch.
        let created = apply_create(pattern, row, writer, params, outcome).await?;
        let mut created = created;
        for op in on_create_sets {
            created = apply_set(op, created, writer, params, outcome).await?;
        }
        Ok(vec![created])
    }
}

/// Try to match the MERGE pattern against the current snapshot. Returns
/// every row of bindings produced by the match (empty if no match).
///
/// `lower_create_pattern_element` emits Nodes and Rels in CREATE order
/// (target Node before its incoming Rel), so callers must NOT assume
/// positional layout. We locate the head by alias (the source of the
/// single Rel for a 1-hop pattern, or the only Node for a 0-hop one)
/// and dispatch by alias from there.
#[allow(clippy::type_complexity)] // local BTreeMap of borrowed pattern slots
async fn find_merge_matches(
    pattern: &[CreateElement],
    outer_row: &Row,
    writer: &mut WriterSession,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    // Split the pattern into Nodes (by alias) and Rels (in insertion
    // order). v0 supports either a single Node, or exactly one Rel
    // joining two Nodes.
    let mut nodes: MergeNodeMap<'_> = BTreeMap::new();
    let mut rels: Vec<&CreateElement> = Vec::new();
    for el in pattern {
        match el {
            CreateElement::Node {
                alias,
                labels,
                properties,
                properties_spread: _,
            } => {
                // Carry the full label set: `MERGE (n:A:B)` matches a node that
                // carries BOTH labels, and creates one with both on miss.
                nodes.insert(alias.as_str(), (labels.as_slice(), properties.as_slice()));
            }
            CreateElement::Rel { .. } => rels.push(el),
        }
    }

    if rels.is_empty() {
        // Single-node MERGE: pattern must contain exactly one Node.
        if nodes.len() != 1 {
            return Err(ExecError::Runtime(
                "MERGE pattern must contain at least one node".into(),
            ));
        }
        let (head_alias, (head_labels, head_props)) = nodes.into_iter().next().expect("len == 1");
        let snap = writer.overlay_snapshot();
        let candidates = snap
            .scan_label(merge_scan_label(head_labels))
            .await
            .map_err(ExecError::Storage)?;
        let mut matched_rows: Vec<Row> = Vec::new();
        for view in candidates {
            let node_val = NodeValue::from(view);
            if !node_has_all_labels(&node_val, head_labels) {
                continue;
            }
            if !merge_props_match(head_props, &node_val.properties, outer_row, params)? {
                continue;
            }
            let mut new_row = outer_row.clone();
            new_row.set(
                head_alias.to_string(),
                RuntimeValue::Node(Box::new(node_val)),
            );
            matched_rows.push(new_row);
        }
        return Ok(matched_rows);
    }

    // N-hop chain: seed matched rows from the first rel's source node,
    // then extend through each rel in insertion order. `rels` is already
    // in chain order (see `lower_create_pattern_element`).
    //
    // Each pattern node can be either:
    //   * a fresh local Node (entry in `nodes` with label + property
    //     spec) — scan its label, filter by props, bind on the row;
    //   * a back-reference to an alias already bound on the outer row
    //     (e.g. `MATCH (a), (b) MERGE (a)-[:R]->(b)`) — no scan, just
    //     keep the carried-in NodeValue.
    let snap = writer.overlay_snapshot();
    let first_head_alias = match rels[0] {
        CreateElement::Rel { source_alias, .. } => source_alias.as_str(),
        _ => unreachable!("rels only contains Rel variants"),
    };
    let mut matched_rows: Vec<Row> =
        seed_merge_head(first_head_alias, &nodes, outer_row, &snap, params).await?;

    for rel in &rels {
        let (rel_alias, rel_edge_type, rel_direction, rel_props, source_alias, target_alias) =
            match rel {
                CreateElement::Rel {
                    alias,
                    edge_type,
                    direction,
                    properties,
                    source_alias,
                    target_alias,
                    ..
                } => (
                    alias.as_deref(),
                    edge_type.as_str(),
                    *direction,
                    properties.as_slice(),
                    source_alias.as_str(),
                    target_alias.as_str(),
                ),
                _ => unreachable!("rels only contains Rel variants"),
            };
        // Resolve the tail: either a fresh pattern Node or a
        // back-reference to a binding on the outer row.
        let tail = MergeTail::resolve(target_alias, &nodes, outer_row)?;

        let mut next: Vec<Row> = Vec::new();
        for source_row in matched_rows {
            let source_node_id = match source_row.get(source_alias) {
                Some(RuntimeValue::Node(n)) => n.id,
                _ => continue,
            };
            let neighbours = match rel_direction {
                RelationshipDirection::Right => snap.out_edges(rel_edge_type, source_node_id).await,
                RelationshipDirection::Left => snap.in_edges(rel_edge_type, source_node_id).await,
                RelationshipDirection::Both => {
                    return Err(ExecError::Runtime(
                        "MERGE relationship must be directed".into(),
                    ));
                }
            }
            .map_err(ExecError::Storage)?;

            for e in neighbours.edges {
                let partner_id = match rel_direction {
                    RelationshipDirection::Right => e.dst,
                    RelationshipDirection::Left => e.src,
                    _ => unreachable!(),
                };
                let partner_node = match &tail {
                    MergeTail::Fresh { labels, props } => {
                        let view = match snap
                            .lookup_node(merge_scan_label(labels), partner_id)
                            .await
                            .map_err(ExecError::Storage)?
                        {
                            Some(v) => v,
                            None => continue,
                        };
                        let partner = NodeValue::from(view);
                        if !node_has_all_labels(&partner, labels) {
                            continue;
                        }
                        if !merge_props_match(props, &partner.properties, &source_row, params)? {
                            continue;
                        }
                        partner
                    }
                    MergeTail::BackReference { node_id, value } => {
                        if partner_id != *node_id {
                            continue;
                        }
                        (**value).clone()
                    }
                };
                let rel_value = RelValue::from(e);
                if !merge_props_match(rel_props, &rel_value.properties, &source_row, params)? {
                    continue;
                }
                let mut new_row = source_row.clone();
                new_row.set(
                    target_alias.to_string(),
                    RuntimeValue::Node(Box::new(partner_node)),
                );
                if let Some(name) = rel_alias {
                    new_row.set(name.to_string(), RuntimeValue::Rel(Box::new(rel_value)));
                }
                next.push(new_row);
            }
        }
        matched_rows = next;
    }
    Ok(matched_rows)
}

/// `alias -> (label, declared property entries)` map built once per
/// MERGE call from the lowered `CreateElement::Node`s. Lives only as
/// long as `find_merge_matches` borrows the lowered pattern.
type MergeNodeMap<'a> = BTreeMap<&'a str, (&'a [String], &'a [(String, Expression)])>;

/// The label a MERGE node scans on (its primary/first); the remaining labels
/// are confirmed per-candidate by [`node_has_all_labels`]. Empty string when
/// somehow unlabelled (lowering requires at least one label).
fn merge_scan_label(labels: &[String]) -> &str {
    labels.first().map(String::as_str).unwrap_or("")
}

/// True if `n` carries every label in `required` — the conjunctive set
/// semantics of `MERGE (n:A:B)` / `MATCH (n:A:B)`.
fn node_has_all_labels(n: &NodeValue, required: &[String]) -> bool {
    required.iter().all(|l| n.labels.contains(l))
}

/// Seed `find_merge_matches` for an N-hop chain. The "head" alias is
/// the source of the first rel. If the pattern declares it as a fresh
/// Node we scan its label; if the caller already bound it on the outer
/// row (back-reference) we lift that NodeValue verbatim.
async fn seed_merge_head(
    head_alias: &str,
    nodes: &MergeNodeMap<'_>,
    outer_row: &Row,
    snap: &namidb_storage::Snapshot<'_>,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    if let Some((head_labels, head_props)) = nodes.get(head_alias).copied() {
        let candidates = snap
            .scan_label(merge_scan_label(head_labels))
            .await
            .map_err(ExecError::Storage)?;
        let mut out = Vec::new();
        for view in candidates {
            let node_val = NodeValue::from(view);
            if !node_has_all_labels(&node_val, head_labels) {
                continue;
            }
            if !merge_props_match(head_props, &node_val.properties, outer_row, params)? {
                continue;
            }
            let mut new_row = outer_row.clone();
            new_row.set(
                head_alias.to_string(),
                RuntimeValue::Node(Box::new(node_val)),
            );
            out.push(new_row);
        }
        return Ok(out);
    }
    if let Some(RuntimeValue::Node(_)) = outer_row.get(head_alias) {
        // Back-reference: one match per outer row, carrying the
        // existing binding through unchanged.
        return Ok(vec![outer_row.clone()]);
    }
    Err(ExecError::Runtime(format!(
        "MERGE head `{}` not found in pattern or outer scope",
        head_alias
    )))
}

/// Tail-side classification for one rel inside an N-hop MERGE chain:
/// either a fresh local Node (scan label + match props) or a
/// back-reference to a NodeValue already bound on the outer row (the
/// rel must point at exactly that id).
enum MergeTail<'a> {
    Fresh {
        labels: &'a [String],
        props: &'a [(String, Expression)],
    },
    BackReference {
        node_id: NodeId,
        value: Box<NodeValue>,
    },
}

impl<'a> MergeTail<'a> {
    fn resolve(alias: &str, nodes: &MergeNodeMap<'a>, outer_row: &Row) -> Result<Self, ExecError> {
        if let Some((labels, props)) = nodes.get(alias).copied() {
            return Ok(MergeTail::Fresh { labels, props });
        }
        if let Some(RuntimeValue::Node(n)) = outer_row.get(alias) {
            return Ok(MergeTail::BackReference {
                node_id: n.id,
                value: n.clone(),
            });
        }
        Err(ExecError::Runtime(format!(
            "MERGE tail `{}` not found in pattern or outer scope",
            alias
        )))
    }
}

fn merge_props_match(
    declared: &[(String, Expression)],
    actual: &BTreeMap<String, RuntimeValue>,
    row: &Row,
    params: &Params,
) -> Result<bool, ExecError> {
    for (key, expr) in declared {
        let expected = evaluate(expr, row, params)?;
        match actual.get(key) {
            Some(v) if runtime_values_equal(v, &expected) => continue,
            _ => return Ok(false),
        }
    }
    Ok(true)
}

fn runtime_values_equal(a: &RuntimeValue, b: &RuntimeValue) -> bool {
    match (a, b) {
        (RuntimeValue::Null, RuntimeValue::Null) => true,
        (RuntimeValue::Integer(x), RuntimeValue::Integer(y)) => x == y,
        (RuntimeValue::Float(x), RuntimeValue::Float(y)) => x == y,
        (RuntimeValue::Integer(x), RuntimeValue::Float(y))
        | (RuntimeValue::Float(y), RuntimeValue::Integer(x)) => (*x as f64) == *y,
        (RuntimeValue::Bool(x), RuntimeValue::Bool(y)) => x == y,
        (RuntimeValue::String(x), RuntimeValue::String(y)) => x == y,
        (RuntimeValue::Node(x), RuntimeValue::Node(y)) => x.id == y.id,
        (RuntimeValue::Rel(x), RuntimeValue::Rel(y)) => {
            x.edge_type == y.edge_type && x.src == y.src && x.dst == y.dst
        }
        _ => false,
    }
}

// ──────────────────────────── helpers ────────────────────────────────

fn expect_node_id(row: &Row, alias: &str) -> Result<NodeId, ExecError> {
    match row.get(alias) {
        Some(RuntimeValue::Node(n)) => Ok(n.id),
        Some(other) => Err(ExecError::Runtime(format!(
            "CREATE/MERGE source/target `{}` must be a Node, got {}",
            alias,
            other.type_name()
        ))),
        None => Err(ExecError::Runtime(format!(
            "CREATE/MERGE source/target `{}` is not bound",
            alias
        ))),
    }
}

fn runtime_to_core(v: &RuntimeValue, expr: &Expression) -> Result<CoreValue, String> {
    match v {
        RuntimeValue::Null => Ok(CoreValue::Null),
        RuntimeValue::Bool(b) => Ok(CoreValue::Bool(*b)),
        RuntimeValue::Integer(n) => Ok(CoreValue::I64(*n)),
        RuntimeValue::Float(f) => Ok(CoreValue::F64(*f)),
        RuntimeValue::String(s) => Ok(CoreValue::Str(s.clone())),
        RuntimeValue::Bytes(b) => Ok(CoreValue::Bytes(b.clone())),
        RuntimeValue::Vector(v) => Ok(CoreValue::Vec(v.clone())),
        RuntimeValue::Date(d) => Ok(CoreValue::Date(*d)),
        RuntimeValue::DateTime(m) => Ok(CoreValue::DateTime(*m)),
        RuntimeValue::List(items) => {
            // Lists store through the `__overflow_json` stream as a
            // tagged JSON object; the writer cannot route them into a
            // declared columnar property yet.
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(runtime_to_core(item, expr)?);
            }
            Ok(CoreValue::List(out))
        }
        RuntimeValue::Map(entries) => {
            let mut out = BTreeMap::new();
            for (k, v) in entries {
                out.insert(k.clone(), runtime_to_core(v, expr)?);
            }
            Ok(CoreValue::Map(out))
        }
        other => Err(format!(
            "property value at `{}` is {} — only scalars, lists, and string-keyed maps are storable",
            expr,
            other.type_name()
        )),
    }
}

fn node_runtime_props_to_core(
    props: &BTreeMap<String, RuntimeValue>,
) -> Result<BTreeMap<String, CoreValue>, ExecError> {
    let mut out = BTreeMap::new();
    for (k, v) in props {
        let core = runtime_value_to_core(v).map_err(|msg| {
            ExecError::Runtime(format!("property `{k}` cannot round-trip: {msg}"))
        })?;
        out.insert(k.clone(), core);
    }
    Ok(out)
}

/// Variant of [`runtime_to_core`] without an `Expression` to anchor
/// the error to. Used when re-serialising a previously-bound node /
/// rel's properties back to the writer (SET applied to a property
/// that came from a Node value).
fn runtime_value_to_core(v: &RuntimeValue) -> Result<CoreValue, String> {
    match v {
        RuntimeValue::Null => Ok(CoreValue::Null),
        RuntimeValue::Bool(b) => Ok(CoreValue::Bool(*b)),
        RuntimeValue::Integer(n) => Ok(CoreValue::I64(*n)),
        RuntimeValue::Float(f) => Ok(CoreValue::F64(*f)),
        RuntimeValue::String(s) => Ok(CoreValue::Str(s.clone())),
        RuntimeValue::Bytes(b) => Ok(CoreValue::Bytes(b.clone())),
        RuntimeValue::Vector(v) => Ok(CoreValue::Vec(v.clone())),
        RuntimeValue::Date(d) => Ok(CoreValue::Date(*d)),
        RuntimeValue::DateTime(m) => Ok(CoreValue::DateTime(*m)),
        RuntimeValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(runtime_value_to_core(item)?);
            }
            Ok(CoreValue::List(out))
        }
        RuntimeValue::Map(entries) => {
            let mut out = BTreeMap::new();
            for (k, v) in entries {
                out.insert(k.clone(), runtime_value_to_core(v)?);
            }
            Ok(CoreValue::Map(out))
        }
        other => Err(format!(
            "{} is not storable (only scalars, lists, and string-keyed maps round-trip)",
            other.type_name()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use namidb_core::id::NamespaceId;
    use namidb_storage::NamespacePaths;
    use std::sync::Arc;

    fn store() -> Arc<dyn object_store::ObjectStore> {
        Arc::new(object_store::memory::InMemory::new())
    }

    fn paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    #[tokio::test]
    async fn create_node_persists_and_returns_binding() {
        use crate::{lower, parse, Params};

        let mut writer = WriterSession::open(store(), paths("write-create"))
            .await
            .unwrap();
        let q = parse("CREATE (a:Person {name: 'Ada'}) RETURN a").unwrap();
        let plan = lower(&q).unwrap();
        let outcome = execute_write(&plan, &mut writer, &Params::new())
            .await
            .unwrap();
        assert_eq!(outcome.nodes_created, 1);
        assert_eq!(outcome.rows.len(), 1);
        match outcome.rows[0].get("a") {
            Some(RuntimeValue::Node(n)) => {
                assert!(n.labels.contains("Person"));
                match n.properties.get("name") {
                    Some(RuntimeValue::String(s)) => assert_eq!(s, "Ada"),
                    other => panic!("unexpected: {:?}", other),
                }
            }
            other => panic!("unexpected: {:?}", other),
        }
        // Now snapshot and read back.
        let snap = writer.snapshot();
        let nodes = snap.scan_label("Person").await.unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[tokio::test]
    async fn create_node_with_params_spread_persists_entries() {
        use crate::{lower, parse, Params};

        let mut writer = WriterSession::open(store(), paths("write-create-spread"))
            .await
            .unwrap();
        let q = parse("CREATE (a:Person $props) RETURN a").unwrap();
        let plan = lower(&q).unwrap();

        let mut spread = BTreeMap::new();
        spread.insert("name".to_string(), RuntimeValue::String("Ada".into()));
        spread.insert("age".to_string(), RuntimeValue::Integer(36));
        let mut params = Params::new();
        params.insert("props".to_string(), RuntimeValue::Map(spread));

        let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
        assert_eq!(outcome.nodes_created, 1);
        match outcome.rows[0].get("a") {
            Some(RuntimeValue::Node(n)) => {
                assert!(n.labels.contains("Person"));
                assert!(matches!(
                    n.properties.get("name"),
                    Some(RuntimeValue::String(s)) if s == "Ada"
                ));
                assert!(matches!(
                    n.properties.get("age"),
                    Some(RuntimeValue::Integer(36))
                ));
            }
            other => panic!("unexpected: {:?}", other),
        }

        let snap = writer.snapshot();
        let nodes = snap.scan_label("Person").await.unwrap();
        assert_eq!(nodes.len(), 1);
        // Stored properties should match what the spread provided.
        let stored = &nodes[0].properties;
        assert!(stored.contains_key("name"));
        assert!(stored.contains_key("age"));
    }

    #[tokio::test]
    async fn create_rejects_duplicate_unique_property() {
        use crate::{lower, parse, Params};
        use namidb_core::{DataType, LabelDef, PropertyDef, SchemaBuilder};

        let mut writer = WriterSession::open(store(), paths("write-unique"))
            .await
            .unwrap();

        // Create Ada, then flush a schema that declares Person.name unique.
        // The flush persists Ada and records the schema on the manifest, so
        // the next CREATE checks against the committed snapshot.
        let q = parse("CREATE (a:Person {name: 'Ada'}) RETURN a").unwrap();
        execute_write(&lower(&q).unwrap(), &mut writer, &Params::new())
            .await
            .unwrap();

        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, true)
                    .unwrap()
                    .with_unique(true)],
            })
            .unwrap()
            .build();
        writer.flush(schema).await.unwrap();

        // A second Ada must be rejected as a unique-constraint violation.
        let dup = parse("CREATE (b:Person {name: 'Ada'}) RETURN b").unwrap();
        let err = execute_write(&lower(&dup).unwrap(), &mut writer, &Params::new())
            .await
            .expect_err("duplicate unique value must be rejected");
        assert!(
            matches!(err, ExecError::Constraint(_)),
            "expected a constraint violation, got: {err:?}"
        );

        // A different name still succeeds (no false positive).
        let ok = parse("CREATE (c:Person {name: 'Bob'}) RETURN c").unwrap();
        execute_write(&lower(&ok).unwrap(), &mut writer, &Params::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn enforce_node_unique_constraints_rejects_duplicate_for_low_level_path() {
        use crate::{lower, parse, Params};
        use namidb_core::{DataType, LabelDef, PropertyDef, SchemaBuilder};

        let mut writer = WriterSession::open(store(), paths("enforce-unique-helper"))
            .await
            .unwrap();
        execute_write(
            &lower(&parse("CREATE (a:Person {name: 'Ada'}) RETURN a").unwrap()).unwrap(),
            &mut writer,
            &Params::new(),
        )
        .await
        .unwrap();
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, true)
                    .unwrap()
                    .with_unique(true)],
            })
            .unwrap()
            .build();
        writer.flush(schema).await.unwrap();

        // The public helper the Python low-level bulk API calls must reject a
        // duplicate unique value the same way CREATE does.
        let labels = vec!["Person".to_string()];
        let mut dup = BTreeMap::new();
        dup.insert("name".to_string(), CoreValue::Str("Ada".into()));
        assert!(
            enforce_node_unique_constraints(&writer, &labels, &dup)
                .await
                .is_err(),
            "the low-level path must reject a duplicate unique value"
        );

        let mut fresh = BTreeMap::new();
        fresh.insert("name".to_string(), CoreValue::Str("Bob".into()));
        assert!(enforce_node_unique_constraints(&writer, &labels, &fresh)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn failed_write_does_not_leak_into_the_next_commit() {
        use crate::{lower, parse, Params};

        let mut writer = WriterSession::open(store(), paths("write-discard-on-error"))
            .await
            .unwrap();

        // Stages (a:Person) then fails on the second element's non-map
        // spread. Before the fix the staged Person stayed in the pending
        // batch of this long-lived writer.
        let q = parse("CREATE (a:Person {name: 'Ada'}), (b:Ghost $props) RETURN a").unwrap();
        let plan = lower(&q).unwrap();
        let mut bad = Params::new();
        bad.insert("props".to_string(), RuntimeValue::Integer(7));
        let err = execute_write(&plan, &mut writer, &bad)
            .await
            .expect_err("non-map spread should fail the statement");
        assert!(format!("{err:?}").contains("MAP"));

        // A later, unrelated write commits on the same writer.
        let q2 = parse("CREATE (c:Other {k: 1}) RETURN c").unwrap();
        let plan2 = lower(&q2).unwrap();
        execute_write(&plan2, &mut writer, &Params::new())
            .await
            .unwrap();

        // The Person staged by the failed statement must NOT have been
        // sealed by the second statement's commit.
        let snap = writer.snapshot();
        assert_eq!(
            snap.scan_label("Person").await.unwrap().len(),
            0,
            "a node staged by a failed write must not leak into the next commit"
        );
        assert_eq!(snap.scan_label("Other").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn set_rejects_duplicate_unique_property_but_allows_self() {
        use crate::{lower, parse, Params};
        use namidb_core::{DataType, LabelDef, PropertyDef, SchemaBuilder};

        let mut writer = WriterSession::open(store(), paths("write-set-unique"))
            .await
            .unwrap();

        for q in [
            "CREATE (a:Person {name: 'Ada'})",
            "CREATE (b:Person {name: 'Bob'})",
        ] {
            execute_write(
                &lower(&parse(q).unwrap()).unwrap(),
                &mut writer,
                &Params::new(),
            )
            .await
            .unwrap();
        }
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, true)
                    .unwrap()
                    .with_unique(true)],
            })
            .unwrap()
            .build();
        writer.flush(schema).await.unwrap();

        // SET Bob.name = 'Ada' collides with Ada: rejected.
        let dup = "MATCH (b:Person {name: 'Bob'}) SET b.name = 'Ada' RETURN b";
        let err = execute_write(
            &lower(&parse(dup).unwrap()).unwrap(),
            &mut writer,
            &Params::new(),
        )
        .await
        .expect_err("setting a unique property to an existing value must be rejected");
        assert!(matches!(err, ExecError::Constraint(_)), "got: {err:?}");

        // SET Ada.name = 'Ada' is a self-update: allowed.
        let same = "MATCH (a:Person {name: 'Ada'}) SET a.name = 'Ada' RETURN a";
        execute_write(
            &lower(&parse(same).unwrap()).unwrap(),
            &mut writer,
            &Params::new(),
        )
        .await
        .unwrap();

        // SET Ada.name = 'Alice' to a fresh value: allowed.
        let fresh = "MATCH (a:Person {name: 'Ada'}) SET a.name = 'Alice' RETURN a";
        execute_write(
            &lower(&parse(fresh).unwrap()).unwrap(),
            &mut writer,
            &Params::new(),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn create_rejects_spread_param_that_is_not_a_map() {
        use crate::{lower, parse, Params};

        let mut writer = WriterSession::open(store(), paths("write-create-spread-bad"))
            .await
            .unwrap();
        let q = parse("CREATE (a:Person $props) RETURN a").unwrap();
        let plan = lower(&q).unwrap();
        let mut params = Params::new();
        params.insert("props".to_string(), RuntimeValue::Integer(7));

        let err = execute_write(&plan, &mut writer, &params)
            .await
            .expect_err("non-map spread should fail");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("MAP"),
            "expected a clear type error, got: {msg}"
        );
    }
}
