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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use futures::future::BoxFuture;
use futures::FutureExt;
use namidb_core::id::NodeId;
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NodeWriteRecord, StagedValue, UniqueProbe, WriterSession};

use super::expr::{evaluate, is_equal, Params};
use super::row::Row;
use super::value::{NodeValue, RelValue, RuntimeValue};
use super::walker::{execute_inner_with_routing, ExecError, PlanRouting};
use crate::parser::{Expression, ExpressionKind, RelationshipDirection};
use crate::plan::logical::{CreateElement, LogicalPlan, RemoveOp, SetOp};

/// Number of correlated point updates hydrated at once by the write-only fast
/// path. 128 rows keeps even 1024d/large-document `NodeView`s bounded while
/// still amortising object-store sidecar and Parquet range reads.
const DEFAULT_CORRELATED_WRITE_CHUNK_ROWS: usize = 128;
/// An operator setting the environment variable must not accidentally turn
/// the bounded path back into a request-sized materialisation.
const MAX_CORRELATED_WRITE_CHUNK_ROWS: usize = 1_024;
const CORRELATED_WRITE_CHUNK_ROWS_ENV: &str = "NAMIDB_CORRELATED_WRITE_CHUNK_ROWS";

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
    /// Storage point-lookup batches issued by the bounded write-only
    /// `UNWIND ... MATCH/MERGE ... SET` paths.
    pub correlated_write_lookup_batches: u64,
    /// Maximum number of complete pre-write `NodeView`s simultaneously owned
    /// by those paths. This is an observable guard against retaining one old
    /// embedding/document per request row.
    pub correlated_write_peak_hydrated_rows: u64,
    /// Maximum number of executor/materialised input rows simultaneously
    /// retained by a bounded correlated SET/MERGE path. Wide request maps are
    /// borrowed from the parameter list and cloned into at most one live row.
    pub correlated_write_peak_materialized_rows: u64,
    /// Effective hard row bound used by the correlated write path. Zero means
    /// this statement did not use the specialised path.
    pub correlated_write_chunk_rows: u64,
    /// Repeated lookup keys resolved from the statement-local scalar overlay
    /// instead of walking the writer's accumulated staged memtable.
    pub correlated_write_local_overlay_hits: u64,
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
    // Analyse the complete statement once. Read subplans delegated from below
    // must retain write-parent context: in particular, an Expand under bare
    // `DELETE r` needs the sparse identity route instead of looking like an
    // unreferenced topology-only Expand and rebuilding a whole-type CSR.
    let routing = PlanRouting::analyze(plan);
    let rows = crate::exec::limits::with_limits(
        deadline,
        None,
        execute_write_inner(plan, writer, params, &mut outcome, &routing),
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
    if let Err(e) = writer.commit_batch().await {
        // The commit failed and this statement is aborted. `commit_batch`
        // preserves the pending batch on error (so an internal retry is
        // possible), but this writer is long-lived and shared: if we return
        // without clearing it, the staged records would be sealed by the next
        // unrelated statement's commit or by the background flush — turning a
        // negatively-acked write durable. Nothing is durable until the manifest
        // CAS lands, so discarding here is safe.
        writer.discard_batch();
        return Err(ExecError::Storage(e));
    }
    Ok(outcome)
}

fn execute_write_inner<'a>(
    plan: &'a LogicalPlan,
    writer: &'a mut WriterSession,
    params: &'a Params,
    outcome: &'a mut WriteOutcome,
    routing: &'a PlanRouting,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    execute_write_inner_mode(plan, writer, params, outcome, routing, true)
}

/// Execute one operator, optionally retaining the rows it produces.
///
/// Children still run in normal row-producing mode because their parent may
/// need those bindings to perform its write. The terminal
/// [`LogicalPlan::DiscardResult`] invokes its direct child with
/// `retain_output = false`, which lets the final write consume and release each
/// updated row instead of accumulating a batch of duplicate embeddings merely
/// to throw it away at the result boundary.
fn execute_write_inner_mode<'a>(
    plan: &'a LogicalPlan,
    writer: &'a mut WriterSession,
    params: &'a Params,
    outcome: &'a mut WriteOutcome,
    routing: &'a PlanRouting,
    retain_output: bool,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    async move {
        match plan {
            // ─── Write operators ────────────────────────────────────
            LogicalPlan::Create { input, elements } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let new_row = apply_create(elements, row, writer, params, outcome).await?;
                    if retain_output {
                        out.push(new_row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Set { input, items } => {
                if !retain_output
                    && execute_discarded_correlated_node_set(input, items, writer, params, outcome)
                        .await?
                {
                    return Ok(Vec::new());
                }
                if !retain_output {
                    if let LogicalPlan::Merge {
                        input: merge_input,
                        pattern,
                        on_match_sets,
                        on_create_sets,
                    } = input.as_ref()
                    {
                        if execute_discarded_correlated_single_node_merge(
                            merge_input,
                            pattern,
                            on_match_sets,
                            on_create_sets,
                            items,
                            writer,
                            params,
                            outcome,
                        )
                        .await?
                        {
                            return Ok(Vec::new());
                        }
                    }
                }
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let new_row = apply_sets(items, row, writer, params, outcome).await?;
                    if retain_output {
                        out.push(new_row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Remove { input, items } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let new_row = apply_removes(items, row, writer, outcome)?;
                    if retain_output {
                        out.push(new_row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Delete {
                input,
                targets,
                detach,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    apply_delete(targets, *detach, &row, writer, params, outcome).await?;
                    if retain_output {
                        out.push(row);
                    }
                }
                Ok(out)
            }

            LogicalPlan::Merge {
                input,
                pattern,
                on_match_sets,
                on_create_sets,
            } => {
                if !retain_output
                    && execute_discarded_correlated_single_node_merge(
                        input,
                        pattern,
                        on_match_sets,
                        on_create_sets,
                        &[],
                        writer,
                        params,
                        outcome,
                    )
                    .await?
                {
                    return Ok(Vec::new());
                }
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len().max(1))
                } else {
                    Vec::new()
                };
                if let Some(PreparedSingleNodeMergeBatch {
                    rows: prepared,
                    mut prefetched_nodes,
                }) = prepare_single_node_merge_batch(pattern, &rows, writer, params).await?
                {
                    debug_assert_eq!(prepared.len(), rows.len());
                    for (row, expected) in rows.into_iter().zip(prepared) {
                        crate::exec::limits::check_deadline()?;
                        let merged = apply_merge(
                            pattern,
                            on_match_sets,
                            on_create_sets,
                            row,
                            writer,
                            params,
                            outcome,
                            Some(&expected),
                            Some(&mut prefetched_nodes),
                            None,
                            None,
                        )
                        .await?;
                        if retain_output {
                            out.extend(merged);
                        }
                    }
                } else if let Some(PreparedBoundRelationshipMergeBatch {
                    rows: prepared,
                    mut prefetched_edges,
                }) =
                    prepare_bound_relationship_merge_batch(pattern, &rows, writer).await?
                {
                    debug_assert_eq!(prepared.len(), rows.len());
                    for (row, expected) in rows.into_iter().zip(prepared) {
                        crate::exec::limits::check_deadline()?;
                        let merged = apply_merge(
                            pattern,
                            on_match_sets,
                            on_create_sets,
                            row,
                            writer,
                            params,
                            outcome,
                            None,
                            None,
                            Some(&expected),
                            Some(&mut prefetched_edges),
                        )
                        .await?;
                        if retain_output {
                            out.extend(merged);
                        }
                    }
                } else {
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
                            None,
                            None,
                            None,
                            None,
                        )
                        .await?;
                        if retain_output {
                            out.extend(merged);
                        }
                    }
                }
                Ok(out)
            }

            LogicalPlan::Foreach {
                input,
                variable,
                list,
                body,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                // FOREACH is side-effect only: the input rows pass through
                // unchanged so any following clause keeps the same cardinality.
                if retain_output {
                    for row in &rows {
                        execute_foreach_row(variable, list, body, row, writer, params, outcome)
                            .await?;
                    }
                    Ok(rows)
                } else {
                    // At a terminal result sink, consume owned rows one at a
                    // time. This releases wide UNWIND bindings (notably
                    // embeddings) as soon as their side effects complete.
                    for row in rows {
                        execute_foreach_row(variable, list, body, &row, writer, params, outcome)
                            .await?;
                    }
                    Ok(Vec::new())
                }
            }

            // Correlated CALL subquery whose body writes: for each outer row,
            // run the subplan's write chain seeded with that row (its Argument
            // leaf carries the imports), then emit the outer row combined with
            // each subplan row. A read-only Apply falls through to the read
            // delegation below.
            LogicalPlan::Apply { input, subplan } if subplan.contains_write() => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let sub_rows = exec_foreach_body_mode(
                        subplan,
                        writer,
                        params,
                        outcome,
                        &row,
                        retain_output,
                    )
                    .await?;
                    if retain_output {
                        for s in sub_rows {
                            let mut merged = row.clone();
                            for (k, v) in &s.bindings {
                                merged.set(k.clone(), v.clone());
                            }
                            out.push(merged);
                        }
                    }
                }
                Ok(out)
            }

            // ─── Read operators that may wrap a write child: handle
            // row-wise here so the write semantics run on the child first.
            LogicalPlan::Project {
                input,
                items,
                distinct,
                discard_input_bindings,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut projected = crate::exec::walker::project_rows(
                    &rows,
                    items,
                    *discard_input_bindings,
                    params,
                )?;
                if *distinct {
                    projected = crate::exec::walker::dedup_rows(projected);
                }
                Ok(if retain_output { projected } else { Vec::new() })
            }
            LogicalPlan::DiscardResult { input } => {
                // Consume the child completely so every write is staged, then
                // ask the terminal operator not to retain its output batch.
                // This prevents updated embeddings from escaping in
                // `WriteOutcome` (and being cloned/expanded by a wire adapter).
                let _ = execute_write_inner_mode(input, writer, params, outcome, routing, false)
                    .await?;
                Ok(Vec::new())
            }
            LogicalPlan::Filter { input, predicate } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    let v = evaluate(predicate, &row, params)?;
                    if retain_output && v.as_bool() == Some(true) {
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
                let mut rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                if !keys.is_empty() {
                    crate::exec::walker::sort_rows(&mut rows, keys, params)?;
                }
                if !retain_output {
                    return Ok(Vec::new());
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
                if !retain_output {
                    let _ =
                        execute_write_inner_mode(input, writer, params, outcome, routing, false)
                            .await?;
                    return Ok(Vec::new());
                }
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                Ok(crate::exec::walker::dedup_rows(rows))
            }
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregations,
            } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let aggregated =
                    crate::exec::walker::execute_aggregate(rows, group_by, aggregations, params)?;
                Ok(if retain_output {
                    aggregated
                } else {
                    Vec::new()
                })
            }
            LogicalPlan::Unwind { input, list, alias } => {
                let rows = execute_write_inner(input, writer, params, outcome, routing).await?;
                let mut out = Vec::new();
                for row in rows {
                    let v = evaluate(list, &row, params)?;
                    match v {
                        RuntimeValue::List(items) if retain_output => {
                            for item in items {
                                let mut new_row = row.clone();
                                new_row.set(alias.clone(), item);
                                out.push(new_row);
                            }
                        }
                        RuntimeValue::List(_) => {}
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
                if !retain_output {
                    // UNION's combination/deduplication is observable only in
                    // its result stream. Both branches must still run, but
                    // neither branch needs to retain wide write rows.
                    let _ = execute_write_inner_mode(left, writer, params, outcome, routing, false)
                        .await?;
                    let _ =
                        execute_write_inner_mode(right, writer, params, outcome, routing, false)
                            .await?;
                    return Ok(Vec::new());
                }
                let mut l = execute_write_inner(left, writer, params, outcome, routing).await?;
                let r = execute_write_inner(right, writer, params, outcome, routing).await?;
                l.extend(r);
                if *all {
                    Ok(l)
                } else {
                    Ok(crate::exec::walker::dedup_rows(l))
                }
            }
            LogicalPlan::CrossProduct { left, right } => {
                if !retain_output {
                    // An uncorrelated unit subquery executes its right side
                    // once, independently of the left cardinality. With no
                    // consumer above this root, constructing the Cartesian
                    // output would only retain duplicate bindings.
                    let _ = execute_write_inner_mode(left, writer, params, outcome, routing, false)
                        .await?;
                    let _ =
                        execute_write_inner_mode(right, writer, params, outcome, routing, false)
                            .await?;
                    return Ok(Vec::new());
                }
                let l = execute_write_inner(left, writer, params, outcome, routing).await?;
                let r = execute_write_inner(right, writer, params, outcome, routing).await?;
                Ok(crate::exec::walker::cross_product(l, r))
            }

            LogicalPlan::HashJoin { .. } | LogicalPlan::HashSemiJoin { .. } => {
                // HashJoin and HashSemiJoin are read-only (their rewriters
                // never touch subtrees that contain writes). In a write
                // path we delegate to the post-write snapshot reader so
                // the executor lives in exactly one place.
                let snap = snapshot_for_write_read(writer, routing);
                execute_inner_with_routing(plan, &snap, params, None, routing).await
            }

            LogicalPlan::EdgeTypeCount { .. } => {
                // Read-only leaf: delegate to the post-write snapshot reader.
                let snap = snapshot_for_write_read(writer, routing);
                execute_inner_with_routing(plan, &snap, params, None, routing).await
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
                let input_rows =
                    execute_write_inner(input, writer, params, outcome, routing).await?;
                let snap = snapshot_for_write_read(writer, routing);
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
                let input_rows =
                    execute_write_inner(input, writer, params, outcome, routing).await?;
                let use_transactional =
                    routing.transactional_property_reads() || writer.has_staged_node_mutations();
                let snap = snapshot_for_write_read(writer, routing);

                // A label-agnostic correlated lookup is necessarily
                // multi-valued: equal values can belong to nodes under
                // different labels. Resolve every String probe in one global
                // posting lookup rather than scanning all nodes once per
                // input row. `snap` is the transactional overlay whenever
                // this statement must observe staged mutations, so the batch
                // retains read-your-own-writes and discard/rollback semantics.
                //
                // The result is aligned to the String subsequence, including
                // duplicates and misses. NULL still never matches and runtime
                // types without a String sidecar retain the exact scan
                // fallback. Evaluate every RHS before I/O so expression
                // errors remain eager and cannot leave a partially consumed
                // result batch.
                if *multi && label.is_empty() {
                    let evaluated_rows = input_rows
                        .into_iter()
                        .map(|row| {
                            let lookup_value = evaluate(value, &row, params)?;
                            Ok::<_, ExecError>((row, lookup_value))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let string_values = evaluated_rows
                        .iter()
                        .filter_map(|(_, lookup_value)| match lookup_value {
                            RuntimeValue::String(value) => Some(value.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let batch_results = if string_values.is_empty() {
                        Vec::new()
                    } else {
                        snap.batch_lookup_nodes_by_property_any_label(property, &string_values)
                            .await
                            .map_err(ExecError::Storage)?
                    };
                    if batch_results.len() != string_values.len() {
                        return Err(ExecError::Runtime(format!(
                            "batch property lookup returned {} results for {} values",
                            batch_results.len(),
                            string_values.len()
                        )));
                    }

                    let mut batch_results = batch_results.into_iter();
                    let mut out = Vec::new();
                    for (row, lookup_value) in evaluated_rows {
                        let matches = match lookup_value {
                            RuntimeValue::String(_) => batch_results.next().ok_or_else(|| {
                                ExecError::Runtime(
                                    "batch property lookup result alignment was lost".into(),
                                )
                            })?,
                            RuntimeValue::Null => Vec::new(),
                            other => {
                                crate::exec::walker::lookup_nodes_by_property_via_scan(
                                    &snap, label, property, &other,
                                )
                                .await?
                            }
                        };
                        for view in matches {
                            let mut new_row = row.clone();
                            new_row.set(
                                alias.clone(),
                                RuntimeValue::Node(Box::new(NodeValue::from(view))),
                            );
                            out.push(new_row);
                        }
                    }
                    return Ok(out);
                }

                // A correlated unique String lookup can resolve the whole
                // input in one storage call, including node-mutating plans
                // such as:
                //
                //   UNWIND $rows AS row
                //   MATCH (n:Doc {key: row.key})
                //   SET n.embedding = row.embedding
                //
                // Those plans require RYOW, but populating the transactional
                // tuple map with `unique_probe` would scan the full label on
                // the first row. Seed just this batch's exact hit/miss keys
                // from the immutable sidecar instead; the storage helper
                // reconciles any bounded staged overlay and journals the
                // partial keys for rollback. Multi-value and non-String
                // probes retain the exact per-row paths below.
                //
                // Evaluate every RHS first so expression errors retain
                // eager-operator semantics and no partial seed has happened
                // when one row fails.
                if !*multi {
                    let evaluated_rows = input_rows
                        .into_iter()
                        .map(|row| {
                            let lookup_value = evaluate(value, &row, params)?;
                            Ok::<_, ExecError>((row, lookup_value))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let string_values = evaluated_rows
                        .iter()
                        .filter_map(|(_, lookup_value)| match lookup_value {
                            RuntimeValue::String(value) => Some(value.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let batch_results = if string_values.is_empty() {
                        Some(Vec::new())
                    } else if use_transactional {
                        writer
                            .seed_unique_string_candidates(label, property, &string_values)
                            .await
                            .map_err(ExecError::Storage)?
                    } else {
                        Some(
                            snap.batch_lookup_nodes_by_property(label, property, &string_values)
                                .await
                                .map_err(ExecError::Storage)?,
                        )
                    };
                    if let Some(results) = &batch_results {
                        if results.len() != string_values.len() {
                            return Err(ExecError::Runtime(format!(
                                "batch property lookup returned {} results for {} values",
                                results.len(),
                                string_values.len()
                            )));
                        }
                    }

                    let mut batch_results = batch_results.map(IntoIterator::into_iter);
                    let mut out = Vec::with_capacity(evaluated_rows.len());
                    for (row, lookup_value) in evaluated_rows {
                        let found = if matches!(&lookup_value, RuntimeValue::String(_)) {
                            if let Some(results) = &mut batch_results {
                                results.next().ok_or_else(|| {
                                    ExecError::Runtime(
                                        "batch property lookup result alignment was lost".into(),
                                    )
                                })?
                            } else {
                                lookup_unique_node_for_write(
                                    writer,
                                    &snap,
                                    label,
                                    property,
                                    &lookup_value,
                                    value,
                                    use_transactional,
                                )
                                .await?
                            }
                        } else {
                            lookup_unique_node_for_write(
                                writer,
                                &snap,
                                label,
                                property,
                                &lookup_value,
                                value,
                                use_transactional,
                            )
                            .await?
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
                    return Ok(out);
                }

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
                    } else if let Some(view) = lookup_unique_node_for_write(
                        writer,
                        &snap,
                        label,
                        property,
                        &lookup_val,
                        value,
                        use_transactional,
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
            // below. `EdgeReadMode::Properties` / `skip_target_materialize =
            // false`: the routing optimisation that prunes those is a
            // read-only walker concern, so materialise fully here (correct,
            // just not pruned).
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
                let input_rows =
                    execute_write_inner(input, writer, params, outcome, routing).await?;
                let snap = snapshot_for_write_read(writer, routing);
                let length = crate::exec::walker::resolve_length(length, params)?;
                crate::exec::walker::execute_expand(
                    input_rows,
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
                    &snap,
                    crate::exec::walker::EdgeReadMode::Properties,
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
            | LogicalPlan::Apply { .. }
            | LogicalPlan::PatternList { .. }
            | LogicalPlan::MultiwayJoin { .. }
            | LogicalPlan::VectorSearch { .. }
            | LogicalPlan::CallProcedure { .. } => {
                let snap = snapshot_for_write_read(writer, routing);
                execute_inner_with_routing(plan, &snap, params, None, routing).await
            }
        }
    }
    .boxed()
}

/// Return the bounded correlated-update chunk size.
///
/// Parsing is deliberately forgiving (matching the other performance tuning
/// knobs): absent, invalid and zero values use the safe default. A hard clamp
/// prevents a deployment typo from restoring request-sized hydration.
fn correlated_write_chunk_rows() -> usize {
    std::env::var(CORRELATED_WRITE_CHUNK_ROWS_ENV)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&rows| rows > 0)
        .unwrap_or(DEFAULT_CORRELATED_WRITE_CHUNK_ROWS)
        .min(MAX_CORRELATED_WRITE_CHUNK_ROWS)
}

/// Whether the narrow correlated SET can keep lookup identity stable.
///
/// A property-only update of the matched alias is safe as long as it cannot
/// change the lookup property. Map replacement/merge and label SETs are
/// deliberately excluded: their key/label effects need the general
/// staged-aware executor.
fn correlated_set_preserves_lookup_identity(
    items: &[SetOp],
    matched_alias: &str,
    lookup_property: &str,
) -> bool {
    !items.is_empty()
        && correlated_sets_preserve_lookup_identity(items, matched_alias, lookup_property)
}

fn correlated_sets_preserve_lookup_identity(
    items: &[SetOp],
    matched_alias: &str,
    lookup_property: &str,
) -> bool {
    items.iter().all(|item| {
        matches!(
            item,
            SetOp::Property {
                target_alias,
                key,
                ..
            } if target_alias == matched_alias && key != lookup_property
        )
    })
}

fn is_direct_property_of(expr: &Expression, alias: &str) -> bool {
    let ExpressionKind::Property(access) = &expr.kind else {
        return false;
    };
    matches!(
        &access.target.kind,
        ExpressionKind::Variable(variable) if variable.name == alias
    )
}

/// Execute the common vectorisation/update shape without materialising its
/// complete intermediate rowset:
///
/// ```text
/// UNWIND $rows AS row
/// MATCH (a:Label {key: row.key})
/// SET a.embedding = row.embedding, ...
/// ```
///
/// The generic tree walker must first produce every UNWIND row, then every
/// hydrated `NodeView`, before `SET` can consume the first row. For existing
/// nodes with wide vectors this retained both the request vectors and the old
/// node vectors for the entire batch. At 2,000 × 1024d that was sufficient to
/// make a routine incremental update exhaust the process.
///
/// This path is intentionally narrow and semantics-preserving:
///
/// * it only handles a terminal/discarded `SET`;
/// * the point lookup must be unique (`multi = false`);
/// * the driver is exactly `Empty -> UNWIND $parameter`, so the parameter list
///   can be borrowed instead of cloned by `evaluate`;
/// * every lookup expression is preflighted as String/NULL before the first
///   mutation, so an unsupported type can fall back without replaying writes;
/// * candidates are point-probed/hydrated as one bounded storage batch, then
///   moved into `SET` one row at a time and dropped immediately.
///
/// The path is admitted only from a clean transaction and only when SET cannot
/// alter the matched alias's key or labels. Each chunk therefore starts from a
/// committed index batch; a scalar `key -> NodeId` map records just the keys
/// this statement already modified, and duplicate keys point-read their one
/// staged node. Work is O(total rows), rather than rescanning and decoding the
/// accumulated staged embeddings for every chunk. Errors still bubble to the
/// existing statement boundary, whose auto-commit wrapper discards the whole
/// pending batch.
async fn execute_discarded_correlated_node_set(
    input: &LogicalPlan,
    items: &[SetOp],
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<bool, ExecError> {
    let LogicalPlan::NodeByPropertyValue {
        input,
        label,
        alias,
        property,
        value,
        multi: false,
    } = input
    else {
        return Ok(false);
    };
    let LogicalPlan::Unwind {
        input,
        list,
        alias: unwind_alias,
    } = input.as_ref()
    else {
        return Ok(false);
    };
    if !matches!(input.as_ref(), LogicalPlan::Empty) {
        return Ok(false);
    }
    if !is_direct_property_of(value, unwind_alias) {
        return Ok(false);
    }
    let ExpressionKind::Parameter(parameter) = &list.kind else {
        return Ok(false);
    };
    let Some(RuntimeValue::List(parameter_rows)) = params.get(parameter) else {
        // Let the generic evaluator retain its exact missing-parameter /
        // UNWIND-type diagnostic.
        return Ok(false);
    };
    if writer.has_staged_node_mutations()
        || !correlated_set_preserves_lookup_identity(items, alias, property)
    {
        // An explicit transaction may carry prior writes, while map/label/key
        // SETs can change which node a later row should match. The generic
        // staged-aware path is authoritative for those cases.
        return Ok(false);
    }

    // Preflight every lookup before staging anything. Retain only compact
    // String keys (never the parameter maps/vectors), so discovering a
    // non-String probe can still hand the untouched statement to the generic
    // executor without replaying side effects.
    let mut lookup_keys = Vec::with_capacity(parameter_rows.len());
    for parameter_row in parameter_rows {
        let row = Row::new().with(unwind_alias.clone(), parameter_row.clone());
        outcome.correlated_write_peak_materialized_rows =
            outcome.correlated_write_peak_materialized_rows.max(1);
        match evaluate(value, &row, params)? {
            RuntimeValue::String(value) => lookup_keys.push(Some(value)),
            RuntimeValue::Null => lookup_keys.push(None),
            _ => return Ok(false),
        }
    }

    let chunk_rows = correlated_write_chunk_rows();
    outcome.correlated_write_chunk_rows =
        outcome.correlated_write_chunk_rows.max(chunk_rows as u64);
    let mut local_overlay: HashMap<String, NodeId> = HashMap::new();

    for (parameter_chunk, lookup_chunk) in parameter_rows
        .chunks(chunk_rows)
        .zip(lookup_keys.chunks(chunk_rows))
    {
        crate::exec::limits::check_deadline()?;

        let string_values = lookup_chunk
            .iter()
            .filter_map(|lookup_key| lookup_key.clone())
            .collect::<Vec<_>>();

        // Read only immutable/committed point indexes. Repeated keys are
        // reconciled below through `local_overlay`, so this never walks the
        // growing staged memtable.
        let string_results = if string_values.is_empty() {
            Vec::new()
        } else {
            outcome.correlated_write_lookup_batches += 1;
            writer
                .batch_lookup_committed_unique_string_candidates(label, property, &string_values)
                .await
                .map_err(ExecError::Storage)?
        };
        if string_results.len() != string_values.len() {
            return Err(ExecError::Runtime(format!(
                "batch property lookup returned {} results for {} values",
                string_results.len(),
                string_values.len()
            )));
        }

        // Align String point results and NULL misses to parameter rows. A key
        // already touched by this statement resolves from exactly one staged
        // point; untouched keys retain the committed batch result.
        let mut string_results = string_results.into_iter();
        let mut aligned_results = Vec::with_capacity(parameter_chunk.len());
        for lookup_key in lookup_chunk {
            let found = match lookup_key {
                Some(key) => {
                    let committed = string_results.next().ok_or_else(|| {
                        ExecError::Runtime("batch property lookup result alignment was lost".into())
                    })?;
                    match local_overlay.get(key) {
                        Some(id) => {
                            outcome.correlated_write_local_overlay_hits += 1;
                            match writer.staged_node(*id).map_err(ExecError::Storage)? {
                                StagedValue::Upsert(view) => Some(view),
                                StagedValue::Tombstone => None,
                                StagedValue::Untouched => {
                                    return Err(ExecError::Runtime(
                                        "correlated SET local overlay lost its staged node".into(),
                                    ));
                                }
                            }
                        }
                        None => committed,
                    }
                }
                None => None,
            };
            aligned_results.push(found);
        }
        if string_results.next().is_some() {
            return Err(ExecError::Runtime(
                "batch property lookup returned extra results".into(),
            ));
        }

        let hydrated_rows = aligned_results.iter().filter(|view| view.is_some()).count() as u64;
        outcome.correlated_write_peak_hydrated_rows = outcome
            .correlated_write_peak_hydrated_rows
            .max(hydrated_rows);
        debug_assert!(aligned_results.len() <= chunk_rows);

        // Move (do not clone) each old NodeView into the one live executor row.
        // `apply_sets` may build the new staged record, but the old wide view
        // and the cloned UNWIND map are released before the next row/chunk.
        for ((parameter_row, lookup_key), found) in parameter_chunk
            .iter()
            .zip(lookup_chunk)
            .zip(aligned_results)
        {
            crate::exec::limits::check_deadline()?;
            let Some(view) = found else {
                continue;
            };
            let node_id = view.id;
            let mut row = Row::new().with(unwind_alias.clone(), parameter_row.clone());
            row.set(
                alias.clone(),
                RuntimeValue::Node(Box::new(NodeValue::from(view))),
            );
            let _ = apply_sets(items, row, writer, params, outcome).await?;
            if let Some(key) = lookup_key {
                // Same rule as the MERGE fast path: only an actually staged
                // mutation may shadow the committed image, otherwise a repeated
                // key in a later chunk resolves to an `Untouched` id.
                if !matches!(
                    writer.staged_node(node_id).map_err(ExecError::Storage)?,
                    StagedValue::Untouched
                ) {
                    local_overlay.insert(key.clone(), node_id);
                }
            }
        }
    }

    Ok(true)
}

/// Metadata for the deliberately narrow terminal single-node MERGE path.
///
/// Requiring exactly one explicit unique String key (and no spread) means a
/// batch point lookup completely decides the match branch. More expressive
/// patterns retain the canonical MERGE executor and its full residual logic.
struct CorrelatedSingleNodeMergeShape<'a> {
    alias: &'a str,
    labels: &'a [String],
    lookup_label: String,
    lookup_property: &'a str,
    lookup_expression: &'a Expression,
}

fn correlated_single_node_merge_shape<'a>(
    pattern: &'a [CreateElement],
    writer: &WriterSession,
) -> Option<CorrelatedSingleNodeMergeShape<'a>> {
    let [CreateElement::Node {
        alias,
        labels,
        properties,
        properties_spread: None,
    }] = pattern
    else {
        return None;
    };
    let [(lookup_property, lookup_expression)] = properties.as_slice() else {
        return None;
    };
    if lookup_property == "_id" {
        return None;
    }

    let schema = writer.schema();
    let lookup_label = labels.iter().find_map(|label| {
        let property_unique = schema.label(label).is_some_and(|definition| {
            definition.properties.iter().any(|property| {
                property.name.as_str() == lookup_property.as_str() && property.unique
            })
        });
        let constraint_unique = schema.constraints().iter().any(|constraint| {
            constraint.kind == namidb_core::ConstraintKind::Unique
                && constraint.label.as_str() == label.as_str()
                && constraint.properties.len() == 1
                && constraint.properties[0].as_str() == lookup_property.as_str()
        });
        (property_unique || constraint_unique).then(|| label.clone())
    })?;

    Some(CorrelatedSingleNodeMergeShape {
        alias,
        labels,
        lookup_label,
        lookup_property,
        lookup_expression,
    })
}

/// Bounded write-only path for the vector-upsert shape:
///
/// ```text
/// UNWIND $rows AS row
/// MERGE (a:Label {key: row.key})
/// [ON MATCH SET ...] [ON CREATE SET ...]
/// [SET a.embedding = row.embedding, ...]
/// ```
///
/// The generic MERGE preparation retains a materialised pattern and hydrated
/// pre-write `NodeValue` for the complete UNWIND. This implementation borrows
/// the request list, preflights only compact String keys, and batches committed
/// point reads. A scalar key-to-id overlay supplies exact RYOW for duplicates
/// and newly-created keys without scanning the growing staged memtable.
///
/// Admission is intentionally conservative: clean transaction, exact
/// `Empty -> UNWIND $parameter` driver, one node with one explicit
/// single-property unique key, and property-only SETs that cannot change that
/// key or the node's labels. Any richer shape falls back before the first
/// mutation.
#[allow(clippy::too_many_arguments)]
async fn execute_discarded_correlated_single_node_merge(
    input: &LogicalPlan,
    pattern: &[CreateElement],
    on_match_sets: &[SetOp],
    on_create_sets: &[SetOp],
    trailing_sets: &[SetOp],
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<bool, ExecError> {
    let LogicalPlan::Unwind {
        input,
        list,
        alias: unwind_alias,
    } = input
    else {
        return Ok(false);
    };
    if !matches!(input.as_ref(), LogicalPlan::Empty) || writer.has_staged_node_mutations() {
        return Ok(false);
    }
    let ExpressionKind::Parameter(parameter) = &list.kind else {
        return Ok(false);
    };
    let Some(RuntimeValue::List(parameter_rows)) = params.get(parameter) else {
        return Ok(false);
    };
    let Some(shape) = correlated_single_node_merge_shape(pattern, writer) else {
        return Ok(false);
    };
    if !is_direct_property_of(shape.lookup_expression, unwind_alias) {
        return Ok(false);
    }
    if !correlated_sets_preserve_lookup_identity(on_match_sets, shape.alias, shape.lookup_property)
        || !correlated_sets_preserve_lookup_identity(
            on_create_sets,
            shape.alias,
            shape.lookup_property,
        )
        || !correlated_sets_preserve_lookup_identity(
            trailing_sets,
            shape.alias,
            shape.lookup_property,
        )
    {
        return Ok(false);
    }

    // Match the generic batch preparer's error-before-effects behaviour:
    // evaluate every MERGE key before staging the first row. Only the compact
    // String keys survive this pass; request maps and vectors are dropped
    // immediately after each expression evaluation.
    let mut lookup_keys = Vec::with_capacity(parameter_rows.len());
    for parameter_row in parameter_rows {
        let row = Row::new().with(unwind_alias.clone(), parameter_row.clone());
        outcome.correlated_write_peak_materialized_rows =
            outcome.correlated_write_peak_materialized_rows.max(1);
        match evaluate(shape.lookup_expression, &row, params)? {
            RuntimeValue::String(value) => lookup_keys.push(value),
            _ => return Ok(false),
        }
    }

    let chunk_rows = correlated_write_chunk_rows();
    outcome.correlated_write_chunk_rows =
        outcome.correlated_write_chunk_rows.max(chunk_rows as u64);
    let mut local_overlay: HashMap<String, NodeId> = HashMap::new();

    for (parameter_chunk, lookup_chunk) in parameter_rows
        .chunks(chunk_rows)
        .zip(lookup_keys.chunks(chunk_rows))
    {
        crate::exec::limits::check_deadline()?;

        // Keys touched by an earlier chunk are deliberately omitted: seeding
        // their older committed answer would overwrite the newer
        // transactional claimant. Duplicates first seen in this chunk may
        // share the same committed answer; processing-time overlay checks make
        // the second occurrence observe the first occurrence's mutation.
        let batched_positions = lookup_chunk
            .iter()
            .map(|key| !local_overlay.contains_key(key))
            .collect::<Vec<_>>();
        let batched_values = lookup_chunk
            .iter()
            .zip(&batched_positions)
            .filter(|(_, batched)| **batched)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        let committed = if batched_values.is_empty() {
            Vec::new()
        } else {
            outcome.correlated_write_lookup_batches += 1;
            writer
                .seed_unmodified_committed_unique_string_candidates(
                    &shape.lookup_label,
                    shape.lookup_property,
                    &batched_values,
                )
                .await
                .map_err(ExecError::Storage)?
        };
        if committed.len() != batched_values.len() {
            return Err(ExecError::Runtime(format!(
                "batch MERGE property lookup returned {} results for {} values",
                committed.len(),
                batched_values.len()
            )));
        }
        let hydrated_rows = committed.iter().filter(|view| view.is_some()).count() as u64;
        outcome.correlated_write_peak_hydrated_rows = outcome
            .correlated_write_peak_hydrated_rows
            .max(hydrated_rows);

        let mut committed = committed.into_iter();
        for (index, (parameter_row, lookup_key)) in
            parameter_chunk.iter().zip(lookup_chunk).enumerate()
        {
            crate::exec::limits::check_deadline()?;
            let was_batched = batched_positions[index];
            let committed_candidate = if was_batched {
                committed.next()
            } else {
                Some(None)
            };
            let committed_candidate = committed_candidate.ok_or_else(|| {
                ExecError::Runtime("batch MERGE property lookup result alignment was lost".into())
            })?;
            let candidate = match local_overlay.get(lookup_key) {
                Some(id) => {
                    outcome.correlated_write_local_overlay_hits += 1;
                    match writer.staged_node(*id).map_err(ExecError::Storage)? {
                        StagedValue::Upsert(view) => Some(view),
                        StagedValue::Tombstone => None,
                        StagedValue::Untouched => {
                            return Err(ExecError::Runtime(
                                "correlated MERGE local overlay lost its staged node".into(),
                            ));
                        }
                    }
                }
                None => {
                    debug_assert!(was_batched);
                    committed_candidate
                }
            };

            let expected = MaterializedMergeNode {
                properties: BTreeMap::from([(
                    shape.lookup_property.to_string(),
                    RuntimeValue::String(lookup_key.clone()),
                )]),
                core_properties: BTreeMap::from([(
                    shape.lookup_property.to_string(),
                    CoreValue::Str(lookup_key.clone()),
                )]),
                explicit_id: None,
            };
            outcome.correlated_write_peak_materialized_rows =
                outcome.correlated_write_peak_materialized_rows.max(1);

            let mut row = Row::new().with(unwind_alias.clone(), parameter_row.clone());
            let mut merged = match candidate
                .map(NodeValue::from)
                .filter(|node| materialized_node_matches(node, shape.labels, &expected))
            {
                Some(node) => {
                    row.set(shape.alias.to_string(), RuntimeValue::Node(Box::new(node)));
                    apply_sets(on_match_sets, row, writer, params, outcome).await?
                }
                None => {
                    let row = apply_create(pattern, row, writer, params, outcome).await?;
                    apply_sets(on_create_sets, row, writer, params, outcome).await?
                }
            };
            merged = apply_sets(trailing_sets, merged, writer, params, outcome).await?;
            let id = match merged.get(shape.alias) {
                Some(RuntimeValue::Node(node)) => node.id,
                _ => {
                    return Err(ExecError::Runtime(format!(
                        "correlated MERGE did not bind node `{}`",
                        shape.alias
                    )));
                }
            };
            // Only a row that actually staged a mutation has to shadow the
            // committed image for a later duplicate key. A MERGE that matched an
            // existing node with no ON MATCH and no trailing SET stages nothing,
            // and recording it here would make the next occurrence of that key
            // resolve to an `Untouched` id. Leaving it out keeps the key in the
            // next chunk's committed lookup, which returns the same node.
            if !matches!(
                writer.staged_node(id).map_err(ExecError::Storage)?,
                StagedValue::Untouched
            ) {
                local_overlay.insert(lookup_key.clone(), id);
            }
        }
        if committed.next().is_some() {
            return Err(ExecError::Runtime(
                "batch MERGE property lookup returned extra results".into(),
            ));
        }
    }

    Ok(true)
}

fn snapshot_for_write_read<'a>(
    writer: &'a WriterSession,
    routing: &PlanRouting,
) -> namidb_storage::Snapshot<'a> {
    if routing.transactional_property_reads() || writer.has_staged_node_mutations() {
        writer.transactional_overlay_snapshot()
    } else {
        writer.overlay_snapshot()
    }
}

/// Unique-property lookup used by write plans after their input may already
/// have staged node mutations.
///
/// The read-only walker can use committed caches/SST sidecars, but a RYOW
/// lookup must also see values claimed or freed earlier in the transaction.
/// Probe the writer-private transactional tuple index, including the small
/// I64/F64 variant set required by Cypher numeric equality, then confirm the
/// candidate against the exact overlay snapshot. Unindexable values retain the
/// scan fallback; NULL never matches under three-valued equality.
async fn lookup_unique_node_for_write(
    writer: &WriterSession,
    snapshot: &namidb_storage::Snapshot<'_>,
    label: &str,
    property: &str,
    lookup_value: &RuntimeValue,
    value_expr: &Expression,
    use_transactional: bool,
) -> Result<Option<namidb_storage::NodeView>, ExecError> {
    if lookup_value.is_null() {
        return Ok(None);
    }
    if matches!(lookup_value, RuntimeValue::String(_)) && !use_transactional {
        // Edge-only plans do not invalidate node-property state. Their
        // committed unique sidecar/cache is the true point-read path and
        // avoids a cold full-label population on a large reopened store.
        // Node-mutating plans select the transactional map below.
        return crate::exec::walker::lookup_node_by_property_via_scan(
            snapshot,
            label,
            property,
            lookup_value,
        )
        .await;
    }
    let declared_unique = !label.is_empty()
        && writer.schema().label(label).is_some_and(|def| {
            def.properties
                .iter()
                .any(|prop| prop.name == property && prop.unique)
        });
    if !declared_unique {
        return crate::exec::walker::lookup_node_by_property_via_scan(
            snapshot,
            label,
            property,
            lookup_value,
        )
        .await;
    }

    let core = match runtime_to_core(lookup_value, value_expr) {
        Ok(core) => core,
        Err(_) => {
            // A runtime Node/Rel/Path is not a storable index key. Preserve the
            // expression engine's previous comparison semantics (normally no
            // match) instead of turning a read predicate into a write error.
            return crate::exec::walker::lookup_node_by_property_via_scan(
                snapshot,
                label,
                property,
                lookup_value,
            )
            .await;
        }
    };
    let tuple = vec![(property.to_string(), core)];
    let Some(variants) = merge_unique_probe_variants(&tuple) else {
        return crate::exec::walker::lookup_node_by_property_via_scan(
            snapshot,
            label,
            property,
            lookup_value,
        )
        .await;
    };

    let mut candidate_ids = BTreeSet::new();
    for variant in variants {
        let refs: Vec<(&str, &CoreValue)> = variant
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        match writer
            .unique_probe(label, &refs, None)
            .await
            .map_err(ExecError::Storage)?
        {
            UniqueProbe::Conflict(id) => {
                candidate_ids.insert(id);
            }
            UniqueProbe::NoConflict => {}
            UniqueProbe::Unindexable => {
                return crate::exec::walker::lookup_node_by_property_via_scan(
                    snapshot,
                    label,
                    property,
                    lookup_value,
                )
                .await;
            }
        }
    }

    let mut confirmed: Option<namidb_storage::NodeView> = None;
    for id in candidate_ids {
        let Some(view) = snapshot
            .lookup_node(label, id)
            .await
            .map_err(ExecError::Storage)?
        else {
            continue;
        };
        let matches = view
            .properties
            .get(property)
            .map(|stored| is_equal(&RuntimeValue::from(stored.clone()), lookup_value))
            .unwrap_or(false);
        if !matches {
            continue;
        }
        let replace = confirmed.as_ref().is_none_or(|current| {
            view.lsn > current.lsn || (view.lsn == current.lsn && view.id < current.id)
        });
        if replace {
            confirmed = Some(view);
        }
    }
    Ok(confirmed)
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
    exec_foreach_body_mode(plan, writer, params, outcome, seed, true)
}

/// Execute a seeded write-subquery/body and optionally retain its terminal
/// rows.
///
/// Intermediate rows are preserved because they carry bindings/cardinality to
/// the next updating clause. Only the root result may be discarded. This is
/// what lets a terminal correlated unit subquery execute once per outer row
/// without retaining a second copy of each wide row, while the same subquery
/// under an outer `RETURN` still produces one row per invocation.
fn exec_foreach_body_mode<'a>(
    plan: &'a LogicalPlan,
    writer: &'a mut WriterSession,
    params: &'a Params,
    outcome: &'a mut WriteOutcome,
    seed: &'a Row,
    retain_output: bool,
) -> BoxFuture<'a, Result<Vec<Row>, ExecError>> {
    async move {
        match plan {
            // The leaf: the per-element seed row.
            LogicalPlan::Empty | LogicalPlan::Argument { .. } => {
                if retain_output {
                    Ok(vec![seed.clone()])
                } else {
                    Ok(Vec::new())
                }
            }
            LogicalPlan::Create { input, elements } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    crate::exec::limits::check_deadline()?;
                    let new_row = apply_create(elements, row, writer, params, outcome).await?;
                    if retain_output {
                        out.push(new_row);
                    }
                }
                Ok(out)
            }
            LogicalPlan::Set { input, items } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    let new_row = apply_sets(items, row, writer, params, outcome).await?;
                    if retain_output {
                        out.push(new_row);
                    }
                }
                Ok(out)
            }
            LogicalPlan::Remove { input, items } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len())
                } else {
                    Vec::new()
                };
                for row in rows {
                    let new_row = apply_removes(items, row, writer, outcome)?;
                    if retain_output {
                        out.push(new_row);
                    }
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
                if retain_output {
                    Ok(rows)
                } else {
                    Ok(Vec::new())
                }
            }
            LogicalPlan::Merge {
                input,
                pattern,
                on_match_sets,
                on_create_sets,
            } => {
                let rows = exec_foreach_body(input, writer, params, outcome, seed).await?;
                let mut out = if retain_output {
                    Vec::with_capacity(rows.len().max(1))
                } else {
                    Vec::new()
                };
                for row in rows {
                    let merged = apply_merge(
                        pattern,
                        on_match_sets,
                        on_create_sets,
                        row,
                        writer,
                        params,
                        outcome,
                        None,
                        None,
                        None,
                        None,
                    )
                    .await?;
                    if retain_output {
                        out.extend(merged);
                    }
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
                if retain_output {
                    for row in &rows {
                        execute_foreach_row(variable, list, body, row, writer, params, outcome)
                            .await?;
                    }
                    Ok(rows)
                } else {
                    for row in rows {
                        execute_foreach_row(variable, list, body, &row, writer, params, outcome)
                            .await?;
                    }
                    Ok(Vec::new())
                }
            }
            other => Err(ExecError::Runtime(format!(
                "operator `{}` is not allowed in a FOREACH body",
                other.operator_name()
            ))),
        }
    }
    .boxed()
}

async fn execute_foreach_row(
    variable: &str,
    list: &Expression,
    body: &LogicalPlan,
    row: &Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<(), ExecError> {
    crate::exec::limits::check_deadline()?;
    let items = match evaluate(list, row, params)? {
        RuntimeValue::List(items) => items,
        RuntimeValue::Null => return Ok(()),
        v => {
            return Err(ExecError::Runtime(format!(
                "FOREACH requires a list; got {}",
                v.type_name()
            )));
        }
    };
    foreach_iterations(variable, items, body, row, writer, params, outcome).await
}

/// Run a FOREACH's body once per list element. Read-modify-write mutations to
/// bindings that existed on the incoming `row` are carried across iterations
/// (so `SET c.n = c.n + i` accumulates), but bindings introduced by the body
/// (e.g. a CREATE alias) and the loop variable are not, so they cannot corrupt
/// the next iteration's seed.
async fn foreach_iterations(
    variable: &str,
    items: Vec<RuntimeValue>,
    body: &LogicalPlan,
    row: &Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<(), ExecError> {
    let original_keys: Vec<String> = row.bindings.keys().cloned().collect();
    let mut carry = row.clone();
    for item in items {
        crate::exec::limits::check_deadline()?;
        let mut seed = carry.clone();
        seed.set(variable.to_string(), item);
        let out = exec_foreach_body(body, writer, params, outcome, &seed).await?;
        if let Some(last) = out.into_iter().next_back() {
            for k in &original_keys {
                if let Some(v) = last.get(k) {
                    carry.set(k.clone(), v.clone());
                }
            }
        }
    }
    Ok(())
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
    reserved_id_error: Option<&'static str>,
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
            if let Some(message) = reserved_id_error {
                return Err(ExecError::Runtime(message.into()));
            }
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
/// Scalar values (strings, integers, floats, bools, dates, bytes) probe the
/// writer's transactional unique-value index: one label scan populates it,
/// then every check is O(1) and staged upserts/tombstones keep it current —
/// a constraint-bearing bulk write no longer re-scans the label per row.
/// Values without a canonical scalar encoding fall back to the label scan
/// with a typed-value compare, which is the source of truth.
async fn find_unique_conflict(
    writer: &WriterSession,
    label: &str,
    prop: &str,
    value: &CoreValue,
    exclude: Option<NodeId>,
) -> Result<Option<NodeId>, ExecError> {
    match writer
        .unique_probe(label, &[(prop, value)], exclude)
        .await
        .map_err(ExecError::Storage)?
    {
        UniqueProbe::Conflict(id) => Ok(Some(id)),
        UniqueProbe::NoConflict => Ok(None),
        UniqueProbe::Unindexable => {
            let snap = writer.overlay_snapshot();
            let mut found = None;
            for node in snap.scan_label(label).await.map_err(ExecError::Storage)? {
                if Some(node.id) == exclude {
                    continue;
                }
                if node.properties.get(prop) == Some(value) {
                    found = Some(node.id);
                    break;
                }
            }
            drop(snap);
            Ok(found)
        }
    }
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
    let to_msg = |e: ExecError| match e {
        ExecError::Constraint(msg) => msg,
        other => other.to_string(),
    };
    enforce_unique_on_create(writer, labels, core_props)
        .await
        .map_err(to_msg)?;
    enforce_composite_unique(writer, labels, core_props, None, None)
        .await
        .map_err(to_msg)
}

/// Enforce a unique constraint when SET assigns `value` to `key` on a node.
/// If `key` is a declared unique property on any of the node's labels and a
/// different node already holds `value`, reject. Setting the node's own
/// current value (self-update) is allowed. Values of any type are checked;
/// see [`find_unique_conflict`] for how scalar vs non-scalar is resolved.
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

/// Find a node, other than `exclude`, that already holds the same value for
/// EVERY property in `props` — a composite-uniqueness conflict. All-scalar
/// tuples probe the writer's transactional unique-value index (one label
/// scan to populate, O(1) per check thereafter); tuples containing a
/// non-scalar value scan the label against the read-your-own-writes overlay
/// (RFC-026) — the scan IS the source of truth, so the index cannot drift
/// from it. `props` is assumed complete (every value present and non-null),
/// which the caller guarantees.
async fn find_composite_conflict(
    writer: &WriterSession,
    label: &str,
    props: &[(String, CoreValue)],
    exclude: Option<NodeId>,
) -> Result<Option<NodeId>, ExecError> {
    let pairs: Vec<(&str, &CoreValue)> = props.iter().map(|(k, v)| (k.as_str(), v)).collect();
    match writer
        .unique_probe(label, &pairs, exclude)
        .await
        .map_err(ExecError::Storage)?
    {
        UniqueProbe::Conflict(id) => Ok(Some(id)),
        UniqueProbe::NoConflict => Ok(None),
        UniqueProbe::Unindexable => {
            let snap = writer.overlay_snapshot();
            let mut found = None;
            for node in snap.scan_label(label).await.map_err(ExecError::Storage)? {
                if Some(node.id) == exclude {
                    continue;
                }
                if props.iter().all(|(k, v)| node.properties.get(k) == Some(v)) {
                    found = Some(node.id);
                    break;
                }
            }
            drop(snap);
            Ok(found)
        }
    }
}

/// Enforce declared COMPOSITE uniqueness constraints (two or more properties)
/// for a node being created (`exclude = None`) or updated (`exclude =
/// Some(self_id)`, so a self-update is allowed). A node is exempt from a
/// constraint unless EVERY one of its properties is present and non-null in
/// `core_props`, matching Cypher composite-uniqueness semantics. Single-property
/// uniqueness is handled separately by [`enforce_unique_on_create`] /
/// [`enforce_unique_on_set`] via the `PropertyDef::unique` flag.
async fn enforce_composite_unique(
    writer: &WriterSession,
    labels: &[String],
    core_props: &BTreeMap<String, CoreValue>,
    exclude: Option<NodeId>,
    changed_properties: Option<&[&str]>,
) -> Result<(), ExecError> {
    // Collect the tuples to check first so the schema borrow is released before
    // we take a snapshot.
    let checks: Vec<(String, Vec<(String, CoreValue)>)> = {
        let schema = writer.schema();
        let mut checks = Vec::new();
        for c in schema.constraints() {
            if c.kind != namidb_core::ConstraintKind::Unique || c.properties.len() < 2 {
                continue;
            }
            if !labels.iter().any(|l| l == &c.label) {
                continue;
            }
            if changed_properties.is_some_and(|changed| {
                !c.properties
                    .iter()
                    .any(|property| changed.contains(&property.as_str()))
            }) {
                // Updating an unrelated property cannot create a new claimant
                // for this tuple. Avoid populating its transactional index with
                // a label scan merely to prove the unchanged node still does
                // not conflict with itself.
                continue;
            }
            let mut tuple = Vec::with_capacity(c.properties.len());
            let mut complete = true;
            for p in &c.properties {
                match core_props.get(p) {
                    Some(v) if !matches!(v, CoreValue::Null) => tuple.push((p.clone(), v.clone())),
                    _ => {
                        complete = false;
                        break;
                    }
                }
            }
            if complete {
                checks.push((c.label.clone(), tuple));
            }
        }
        checks
    };
    for (label, tuple) in checks {
        if find_composite_conflict(writer, &label, &tuple, exclude)
            .await?
            .is_some()
        {
            let desc = tuple
                .iter()
                .map(|(k, v)| format!("{k} = {v:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ExecError::Constraint(format!(
                "({desc}) already exists (composite unique constraint on {label})"
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

/// Length of a vector-valued property, or `None` if `v` is not a vector — a dense
/// `Vec`, an int8 `VecI8`, or an all-numeric `List` (a coercion candidate). A
/// heterogeneous/non-numeric list is NOT a vector.
fn vector_value_len(v: &CoreValue) -> Option<usize> {
    match v {
        CoreValue::Vec(x) => Some(x.len()),
        CoreValue::VecI8 { codes, .. } => Some(codes.len()),
        CoreValue::List(items) if items.iter().all(is_numeric_core) => Some(items.len()),
        _ => None,
    }
}

fn is_numeric_core(v: &CoreValue) -> bool {
    matches!(v, CoreValue::I64(_) | CoreValue::F64(_))
}

/// An all-numeric `List` as `Vec<f32>` (for coercion to a dense `CoreValue::Vec`).
fn numeric_list_to_f32(items: &[CoreValue]) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        match it {
            CoreValue::F64(f) => out.push(*f as f32),
            CoreValue::I64(n) => out.push(*n as f32),
            _ => return None,
        }
    }
    Some(out)
}

/// Enforce embedding dimension at write time for every registered vector index
/// covering one of `labels`. A wrong-dim value is rejected (instead of silently
/// poisoning the next index build — a single mismatched row makes `build_body`
/// error and skip the whole `.vg`), and a correct-dim numeric `List` is coerced to
/// a dense `Vec` so it is actually indexed (a `List` is skipped at build time, so
/// a bare-list embedding otherwise reads fine via flat scan yet never enters the
/// `.vg`). No-ops when no vector index exists. Node-only — vector indexes are
/// label-scoped — mirroring `enforce_notnull_on_create`. Length-only (a zero-norm
/// but correct-dim vector is accepted, matching the build's dim-only check).
///
/// `changed` scopes the check to the properties this write actually introduces:
/// `Some(keys)` validates/coerces only indexes whose property is in `keys` (so a
/// `SET d.title = …` does NOT re-validate a node's untouched, possibly legacy,
/// embedding); `None` validates every present property (used on `CREATE`, where
/// all properties are new, and on a label-add, where the existing embedding must
/// satisfy the newly-applicable label's index).
fn enforce_vector_dims(
    writer: &WriterSession,
    labels: &[String],
    core_props: &mut BTreeMap<String, CoreValue>,
    changed: Option<&[&str]>,
) -> Result<(), ExecError> {
    let indexes = writer.vector_indexes();
    if indexes.is_empty() {
        return Ok(());
    }
    for desc in indexes {
        if !labels.iter().any(|l| l == &desc.label) {
            continue;
        }
        // Only validate/coerce indexes whose property this write touches.
        if let Some(keys) = changed {
            if !keys.contains(&desc.property.as_str()) {
                continue;
            }
        }
        // A null clears the property (not-null is enforced separately); a present
        // value must be a vector of the declared dimension.
        let len = match core_props.get(&desc.property) {
            None | Some(CoreValue::Null) => continue,
            Some(v) => match vector_value_len(v) {
                Some(l) => l,
                None => {
                    return Err(ExecError::Constraint(format!(
                        "{}.{} must be a numeric vector for index `{}` (dimension constraint)",
                        desc.label, desc.property, desc.name
                    )))
                }
            },
        };
        if len != desc.dim as usize {
            return Err(ExecError::Constraint(format!(
                "{}.{} embedding has dim {len} but vector index `{}` declares {} \
                 (dimension constraint)",
                desc.label, desc.property, desc.name, desc.dim
            )));
        }
        // Reject non-finite components at the door: a stored NaN poisons every
        // distance it ever participates in, and unlike a wrong dim it cannot
        // be detected from the score a query gets back.
        let non_finite = match core_props.get(&desc.property) {
            Some(CoreValue::Vec(floats)) => floats.iter().any(|f| !f.is_finite()),
            Some(CoreValue::List(items)) => items.iter().any(|item| match item {
                CoreValue::F64(f) => !f.is_finite(),
                _ => false,
            }),
            _ => false,
        };
        if non_finite {
            return Err(ExecError::Constraint(format!(
                "{}.{} embedding contains a non-finite component (NaN or \
                 infinity) for vector index `{}`",
                desc.label, desc.property, desc.name
            )));
        }
        // Correct dim but stored as a numeric List → coerce to a dense Vec so the
        // index build covers it.
        let coerced = match core_props.get(&desc.property) {
            Some(CoreValue::List(items)) => numeric_list_to_f32(items),
            _ => None,
        };
        if let Some(floats) = coerced {
            core_props.insert(desc.property.clone(), CoreValue::Vec(floats));
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
                        None,
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
                    Some(id) => {
                        // CREATE must create a NEW node. With an explicit `_id`
                        // (literal or via a `$props`-spread map) `upsert_node`
                        // would silently OVERWRITE an existing node — a
                        // correctness and data-integrity hole (a client could
                        // clobber another node by supplying its id). Reject a
                        // collision against the read-your-own-writes overlay
                        // (committed + staged) instead.
                        let exists = {
                            let snap = writer.overlay_snapshot();
                            crate::exec::walker::scan_node_for_id(&snap, id)
                                .await?
                                .is_some()
                        };
                        if exists {
                            return Err(ExecError::Constraint(format!(
                                "CREATE with _id {id} conflicts with an existing node"
                            )));
                        }
                        id
                    }
                    None => NodeId::new(),
                };
                // Enforce declared unique constraints against the
                // read-your-own-writes overlay (RFC-026) before staging the
                // node, so a duplicate staged earlier in the same uncommitted
                // batch is caught as well as one already committed.
                enforce_unique_on_create(writer, labels, &core_props).await?;
                enforce_composite_unique(writer, labels, &core_props, None, None).await?;
                enforce_notnull_on_create(writer, labels, &core_props)?;
                // CREATE introduces every property → validate them all.
                enforce_vector_dims(writer, labels, &mut core_props, None)?;
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
                        Some("_id is not valid on a relationship CREATE"),
                    )?;
                }
                for (k, expr) in properties {
                    if k == "_id" {
                        return Err(ExecError::Runtime(
                            "_id is not valid on a relationship CREATE".into(),
                        ));
                    }
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

/// Keep aliases that reference the same physical node coherent after a SET.
///
/// Cypher permits one node to be bound under multiple names. Updating only
/// the target alias leaves the other bindings with a stale property map; a
/// later SET through that alias can then rewrite from stale state, and MERGE's
/// batch-prefetch refresh can cache the wrong clone. The storage write is
/// id-primary, so mirror its complete post-write value to every matching
/// binding in the row.
fn synchronize_node_bindings(row: &mut Row, updated: &NodeValue) {
    for value in row.bindings.values_mut() {
        if matches!(value, RuntimeValue::Node(node) if node.id == updated.id) {
            *value = RuntimeValue::Node(Box::new(updated.clone()));
        }
    }
}

/// Relationship counterpart of [`synchronize_node_bindings`].
///
/// Relationships are identity-keyed by `(type, src, dst)` in storage. Mirror
/// the complete post-write value to every alias for that key so a later SET
/// cannot rewrite it from an older property map.
fn synchronize_rel_bindings(row: &mut Row, updated: &RelValue) {
    for value in row.bindings.values_mut() {
        if matches!(
            value,
            RuntimeValue::Rel(rel)
                if rel.edge_type == updated.edge_type
                    && rel.src == updated.src
                    && rel.dst == updated.dst
        ) {
            *value = RuntimeValue::Rel(Box::new(updated.clone()));
        }
    }
}

/// Refresh the SET target from the writer-private last-write-wins memtable.
///
/// Logical operators materialize their input rows before applying writes.
/// Under `UNWIND` (and MERGE's prefetched batches), later rows therefore carry
/// clones from before an earlier row's mutation. A direct staged point lookup
/// repairs that clone in O(log pending distinct keys) without probing SSTs,
/// then synchronizes every alias of the same entity before the RHS is
/// evaluated.
fn refresh_staged_write_target(
    row: &mut Row,
    target_alias: &str,
    writer: &WriterSession,
) -> Result<(), ExecError> {
    match row.get(target_alias).cloned() {
        Some(RuntimeValue::Node(node)) => {
            match writer.staged_node(node.id).map_err(ExecError::Storage)? {
                StagedValue::Untouched => {}
                StagedValue::Tombstone => {
                    return Err(ExecError::Runtime(format!(
                        "SET target `{target_alias}` was deleted earlier in this transaction"
                    )));
                }
                StagedValue::Upsert(view) => {
                    synchronize_node_bindings(row, &NodeValue::from(view));
                }
            }
        }
        Some(RuntimeValue::Rel(rel)) => match writer
            .staged_edge(&rel.edge_type, rel.src, rel.dst)
            .map_err(ExecError::Storage)?
        {
            StagedValue::Untouched => {}
            StagedValue::Tombstone => {
                return Err(ExecError::Runtime(format!(
                    "SET target `{target_alias}` was deleted earlier in this transaction"
                )));
            }
            StagedValue::Upsert(view) => {
                synchronize_rel_bindings(row, &RelValue::from(view));
            }
        },
        // Preserve the existing type/unbound diagnostics in `apply_set`.
        Some(_) | None => {}
    }
    Ok(())
}

async fn apply_set(
    op: &SetOp,
    mut row: Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
    refresh_staged_write_target(&mut row, op.target_alias(), writer)?;
    match op {
        SetOp::Property {
            target_alias,
            key,
            value,
        } => {
            let new_val = evaluate(value, &row, params)?;
            let core = runtime_to_core(&new_val, value).map_err(ExecError::Runtime)?;
            match row.get(target_alias).cloned() {
                Some(RuntimeValue::Node(n)) => {
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
                    // `refresh_staged_write_target` made this row clone current
                    // before the RHS was evaluated, so the full-record rewrite
                    // never needs a second committed/SST lookup.
                    let mut core_props = node_runtime_props_to_core(&n.properties)?;
                    core_props.insert(key.clone(), core);
                    // Composite uniqueness is checked against the node's full
                    // post-SET property set, excluding the node itself.
                    enforce_composite_unique(
                        writer,
                        &label_vec,
                        &core_props,
                        Some(n.id),
                        Some(&[key.as_str()]),
                    )
                    .await?;
                    // Validate only the property being SET, not the node's other
                    // (possibly pre-index, legacy) embeddings.
                    enforce_vector_dims(
                        writer,
                        &label_vec,
                        &mut core_props,
                        Some(&[key.as_str()]),
                    )?;
                    let runtime_props = core_props
                        .iter()
                        .map(|(name, value)| (name.clone(), RuntimeValue::from(value.clone())))
                        .collect();
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
                    outcome.properties_set += 1;
                    let updated = NodeValue {
                        id: n.id,
                        labels: n.labels,
                        properties: runtime_props,
                    };
                    synchronize_node_bindings(&mut row, &updated);
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
                    synchronize_rel_bindings(&mut row, &r);
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
                let mut core_props = node_runtime_props_to_core(&n.properties)?;
                // A newly-added label brings its own NOT NULL contract: the
                // node must already carry a non-null value for every property
                // that label declares non-null.
                enforce_notnull_on_create(writer, &added_labels, &core_props)?;
                // …and its vector-index dimension contract: a gained label can
                // bring a vector index the node's existing embedding must satisfy
                // (scoped to the gained labels, so a pre-existing index on an
                // unchanged label does not re-validate here — `None` over the
                // added-label set checks every present property against them).
                enforce_vector_dims(writer, &added_labels, &mut core_props, None)?;
                // It also subjects the node to that label's uniqueness contracts
                // (single-property and composite): the node's existing values
                // must not collide with another node under the labels it gains.
                // Both checks scope to `added_labels` and exclude the node
                // itself, so they no-op when no new label is actually added.
                for (k, cv) in &core_props {
                    enforce_unique_on_set(writer, &added_labels, k, cv, n.id).await?;
                }
                enforce_composite_unique(writer, &added_labels, &core_props, Some(n.id), None)
                    .await?;
                let runtime_props = core_props
                    .iter()
                    .map(|(name, value)| (name.clone(), RuntimeValue::from(value.clone())))
                    .collect();
                let record = NodeWriteRecord {
                    properties: core_props,
                    schema_version: 1,
                    ..Default::default()
                };
                writer
                    .upsert_node_with_labels(n.labels.iter().cloned(), n.id, &record)
                    .map_err(ExecError::Storage)?;
                outcome.labels_set += added_labels.len() as u64;
                let updated = NodeValue {
                    id: n.id,
                    labels: n.labels,
                    properties: runtime_props,
                };
                synchronize_node_bindings(&mut row, &updated);
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
        Some(RuntimeValue::Node(n)) => {
            let final_runtime = merged_props(replace, &n.properties, &incoming);
            let mut final_core = node_runtime_props_to_core(&final_runtime)?;
            let labels: Vec<String> = n.labels.iter().cloned().collect();
            let changed: Vec<&str> = incoming.iter().map(|(k, _)| k.as_str()).collect();
            // Uniqueness against the final set, excluding the node's own row so
            // a self-update is allowed; then NOT NULL so a `=` that drops a
            // required column is rejected, not silently committed.
            for key in &changed {
                if let Some(value) = final_core.get(*key) {
                    enforce_unique_on_set(writer, &labels, key, value, n.id).await?;
                }
            }
            enforce_composite_unique(writer, &labels, &final_core, Some(n.id), Some(&changed))
                .await?;
            enforce_notnull_on_create(writer, &labels, &final_core)?;
            // Validate only the properties this SET introduces (a `+=` must not
            // re-validate the node's untouched legacy embeddings; a `=` replaces
            // the whole set, so `incoming` IS the final embedding set).
            enforce_vector_dims(writer, &labels, &mut final_core, Some(&changed))?;
            let final_runtime = final_core
                .iter()
                .map(|(name, value)| (name.clone(), RuntimeValue::from(value.clone())))
                .collect();
            let record = NodeWriteRecord {
                properties: final_core,
                schema_version: 1,
                ..Default::default()
            };
            writer
                .upsert_node_with_labels(n.labels.iter().cloned(), n.id, &record)
                .map_err(ExecError::Storage)?;
            outcome.properties_set += incoming.len() as u64;
            let updated = NodeValue {
                id: n.id,
                labels: n.labels,
                properties: final_runtime,
            };
            synchronize_node_bindings(&mut row, &updated);
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
            synchronize_rel_bindings(&mut row, &r);
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
    let target_alias = match op {
        RemoveOp::Property { target_alias, .. } | RemoveOp::Labels { target_alias, .. } => {
            target_alias.as_str()
        }
    };
    refresh_staged_write_target(&mut row, target_alias, writer)?;
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
                synchronize_node_bindings(&mut row, &n);
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
                synchronize_rel_bindings(&mut row, &r);
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
                synchronize_node_bindings(&mut row, &n);
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
                } else if let Some(edge_type) = incident_edge_type(n.id, writer).await? {
                    // openCypher/Neo4j contract: deleting a connected node
                    // without DETACH is an error, never a silent commit of
                    // dangling edges — which reads would then resurrect as
                    // half-broken traversal results.
                    return Err(ExecError::Runtime(format!(
                        "cannot DELETE a node that still has relationships \
                         (found {edge_type}); use DETACH DELETE to remove the \
                         node and its relationships"
                    )));
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

/// First edge type with a live edge incident to `node`, if any. Early-exits on
/// the first hit, so the common disconnected-node DELETE pays at most one
/// bounded adjacency probe per declared edge type.
async fn incident_edge_type(
    node: NodeId,
    writer: &mut WriterSession,
) -> Result<Option<String>, ExecError> {
    let edge_types: Vec<String> = writer.observed_edge_types();
    for et in edge_types {
        crate::exec::limits::check_deadline()?;
        let snap = writer.overlay_snapshot();
        if !snap
            .out_edges(&et, node)
            .await
            .map_err(ExecError::Storage)?
            .edges
            .is_empty()
            || !snap
                .in_edges(&et, node)
                .await
                .map_err(ExecError::Storage)?
                .edges
                .is_empty()
        {
            return Ok(Some(et));
        }
    }
    Ok(None)
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

/// Materialise and hydrate the existing-node candidates for a correlated
/// single-node MERGE in bulk.
///
/// The transactional unique index answers a hit with a `NodeId`, but the
/// complete `NodeValue` is still needed for residual properties, RETURN and
/// SET. Resolving those ids one row at a time makes a compacted id-primary SST
/// pay one targeted Parquet walk per existing key. Misses never hydrate a
/// node, which is why new-key MERGE stayed fast while idempotent replay
/// degraded with total corpus size.
///
/// Single-String unique keys first use the same sidecar-backed batch point
/// path as MATCH and seed those exact hit/miss keys into the transactional
/// index. Other key shapes retain the generic unique-probe fallback. Candidate
/// ids are then hydrated in row-group-aware batches. Per-row execution below
/// still re-probes the live transactional index, so duplicate keys and ON
/// MATCH mutations retain strict RYOW semantics. The prefetched map is
/// refreshed after every row that mutates a node.
struct PreparedSingleNodeMergeBatch {
    rows: Vec<MaterializedMergeNode>,
    prefetched_nodes: HashMap<NodeId, NodeValue>,
}

async fn prepare_single_node_merge_batch(
    pattern: &[CreateElement],
    rows: &[Row],
    writer: &WriterSession,
    params: &Params,
) -> Result<Option<PreparedSingleNodeMergeBatch>, ExecError> {
    namidb_core::profile_scope!("writer::prepare_single_node_merge_batch");
    let [CreateElement::Node {
        labels,
        properties,
        properties_spread,
        ..
    }] = pattern
    else {
        return Ok(None);
    };
    if rows.is_empty() || !merge_labels_have_unique_key(writer, labels) {
        return Ok(None);
    }

    let node_pattern = MergeNodePattern {
        labels,
        properties,
        properties_spread: properties_spread.as_ref(),
    };
    let mut prepared = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        if index % namidb_storage::cancel::CHECK_STRIDE == 0 {
            crate::exec::limits::check_deadline()?;
        }
        let expected = materialize_merge_node_pattern(node_pattern, row, params)?;
        prepared.push(expected);
    }

    let mut prefetched = HashMap::new();
    // Group all eligible probes so one sidecar pass + one node batch serves
    // the whole UNWIND. If an earlier clause already staged node mutations,
    // the storage seed reconciles that bounded LWW overlay over the committed
    // point answers without scanning the stored label.
    let mut string_groups: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for expected in &prepared {
        if let Some((label, property, value)) =
            covered_single_string_unique_key(writer, labels, expected)
        {
            string_groups
                .entry((label, property))
                .or_default()
                .push(value);
        }
    }
    for ((label, property), values) in string_groups {
        if let Some(views) = writer
            .seed_unique_string_candidates(&label, &property, &values)
            .await
            .map_err(ExecError::Storage)?
        {
            for view in views.into_iter().flatten() {
                prefetched.insert(view.id, NodeValue::from(view));
            }
        }
    }

    let mut ids_by_label: BTreeMap<String, BTreeSet<NodeId>> = BTreeMap::new();
    for expected in &prepared {
        if let Some((label, ids)) = probe_merge_unique_candidates(writer, labels, expected).await? {
            ids_by_label.entry(label).or_default().extend(ids);
        }
    }
    if !ids_by_label.is_empty() {
        let snapshot = writer.overlay_snapshot();
        for (label, ids) in ids_by_label {
            let ids: Vec<NodeId> = ids
                .into_iter()
                .filter(|id| !prefetched.contains_key(id))
                .collect();
            if ids.is_empty() {
                continue;
            }
            let views = snapshot
                .batch_lookup_nodes(&label, &ids)
                .await
                .map_err(ExecError::Storage)?;
            if views.len() != ids.len() {
                return Err(ExecError::Runtime(format!(
                    "batch MERGE hydration returned {} results for {} ids",
                    views.len(),
                    ids.len()
                )));
            }
            for (id, view) in ids.into_iter().zip(views) {
                if let Some(view) = view {
                    prefetched.insert(id, NodeValue::from(view));
                }
            }
        }
    }
    Ok(Some(PreparedSingleNodeMergeBatch {
        rows: prepared,
        prefetched_nodes: prefetched,
    }))
}

/// The sidecar-compatible unique-key subset used by batch MERGE seeding.
///
/// Composite and non-String tuples still use the canonical typed
/// transactional probe. Strings cover the loader's dominant `{key: ...}`
/// shape and are exactly the key domain supported by legacy unique sidecars.
fn covered_single_string_unique_key(
    writer: &WriterSession,
    labels: &[String],
    expected: &MaterializedMergeNode,
) -> Option<(String, String, String)> {
    covered_merge_unique_keys(writer, labels, expected)
        .into_iter()
        .find_map(|(label, tuple)| {
            let [(property, CoreValue::Str(value))] = tuple.as_slice() else {
                return None;
            };
            Some((label, property.clone(), value.clone()))
        })
}

/// Cheap schema-only guard for the bulk preparation above. Dynamic spread
/// maps may or may not actually cover the key on a given row; preparing those
/// rows is still useful because `probe_merge_unique_candidates` decides that
/// from the already-materialised map without evaluating expressions twice.
fn merge_labels_have_unique_key(writer: &WriterSession, labels: &[String]) -> bool {
    writer.schema().constraints().iter().any(|constraint| {
        constraint.kind == namidb_core::ConstraintKind::Unique
            && labels.iter().any(|label| label == &constraint.label)
    }) || labels.iter().any(|label| {
        writer
            .schema()
            .label(label)
            .is_some_and(|definition| definition.properties.iter().any(|property| property.unique))
    })
}

type EdgeMergeKey = (String, NodeId, NodeId);

/// One input row whose MERGE relationship has both endpoints already bound.
///
/// Keeping only the physical identity here is intentional: relationship
/// property expressions are still evaluated at their original per-row point,
/// preserving errors and RYOW semantics. The expensive committed lookup is
/// the part shared across the whole input batch.
struct PreparedBoundRelationshipMerge {
    key: EdgeMergeKey,
}

struct PreparedBoundRelationshipMergeBatch {
    rows: Vec<PreparedBoundRelationshipMerge>,
    prefetched_edges: HashMap<EdgeMergeKey, Option<RelValue>>,
}

/// Batch the loader's dominant `MATCH a, MATCH b, MERGE (a)-[:T]->(b)`
/// existence probes.
///
/// The CSR fallback is opened at most once per SST, while current SSTs use one
/// range-readable B+tree probe for all endpoint pairs. Only patterns whose
/// source and target are genuine outer-scope back-references are eligible;
/// fresh pattern nodes retain the expand/filter matcher below.
async fn prepare_bound_relationship_merge_batch(
    pattern: &[CreateElement],
    rows: &[Row],
    writer: &WriterSession,
) -> Result<Option<PreparedBoundRelationshipMergeBatch>, ExecError> {
    namidb_core::profile_scope!("writer::prepare_bound_relationship_merge_batch");
    if rows.is_empty() {
        return Ok(None);
    }
    let local_nodes: BTreeSet<&str> = pattern
        .iter()
        .filter_map(|element| match element {
            CreateElement::Node { alias, .. } => Some(alias.as_str()),
            CreateElement::Rel { .. } => None,
        })
        .collect();
    let mut rels = pattern.iter().filter_map(|element| match element {
        CreateElement::Rel {
            edge_type,
            direction,
            source_alias,
            target_alias,
            ..
        } => Some((
            edge_type.as_str(),
            *direction,
            source_alias.as_str(),
            target_alias.as_str(),
        )),
        CreateElement::Node { .. } => None,
    });
    let Some((edge_type, direction, source_alias, target_alias)) = rels.next() else {
        return Ok(None);
    };
    if rels.next().is_some()
        || local_nodes.contains(source_alias)
        || local_nodes.contains(target_alias)
        || matches!(direction, RelationshipDirection::Both)
    {
        return Ok(None);
    }

    let mut prepared = Vec::with_capacity(rows.len());
    let mut pairs = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        if index % namidb_storage::cancel::CHECK_STRIDE == 0 {
            crate::exec::limits::check_deadline()?;
        }
        let (Some(RuntimeValue::Node(source)), Some(RuntimeValue::Node(target))) =
            (row.get(source_alias), row.get(target_alias))
        else {
            return Ok(None);
        };
        let (src, dst) = match direction {
            RelationshipDirection::Right => (source.id, target.id),
            RelationshipDirection::Left => (target.id, source.id),
            RelationshipDirection::Both => unreachable!("undirected relationship rejected above"),
        };
        prepared.push(PreparedBoundRelationshipMerge {
            key: (edge_type.to_string(), src, dst),
        });
        pairs.push((src, dst));
    }

    let views = writer
        .overlay_snapshot()
        .batch_lookup_edges_via_sst(edge_type, &pairs)
        .await
        .map_err(ExecError::Storage)?;
    if views.len() != prepared.len() {
        return Err(ExecError::Runtime(format!(
            "batch relationship MERGE returned {} results for {} endpoint pairs",
            views.len(),
            prepared.len()
        )));
    }
    let mut prefetched_edges = HashMap::with_capacity(prepared.len());
    for (expected, view) in prepared.iter().zip(views) {
        prefetched_edges.insert(expected.key.clone(), view.map(RelValue::from));
    }
    Ok(Some(PreparedBoundRelationshipMergeBatch {
        rows: prepared,
        prefetched_edges,
    }))
}

#[allow(clippy::too_many_arguments)]
async fn apply_merge(
    pattern: &[CreateElement],
    on_match_sets: &[SetOp],
    on_create_sets: &[SetOp],
    row: Row,
    writer: &mut WriterSession,
    params: &Params,
    outcome: &mut WriteOutcome,
    prepared_node: Option<&MaterializedMergeNode>,
    mut prefetched_nodes: Option<&mut HashMap<NodeId, NodeValue>>,
    prepared_edge: Option<&PreparedBoundRelationshipMerge>,
    mut prefetched_edges: Option<&mut HashMap<EdgeMergeKey, Option<RelValue>>>,
) -> Result<Vec<Row>, ExecError> {
    // v0: support a single Node pattern, or a Node-Rel-Node chain.
    let hints = MergeMatchHints {
        prepared_node,
        prefetched_nodes: prefetched_nodes.as_deref(),
        prepared_edge,
        prefetched_edges: prefetched_edges.as_deref(),
    };
    let matches = find_merge_matches(pattern, &row, writer, params, hints).await?;
    if !matches.is_empty() {
        let mut out = Vec::with_capacity(matches.len());
        for mut m_row in matches {
            for op in on_match_sets {
                m_row = apply_set(op, m_row, writer, params, outcome).await?;
            }
            if let Some(prefetched) = prefetched_nodes.as_mut() {
                refresh_prefetched_nodes(prefetched, &m_row, pattern);
            }
            if let (Some(expected), Some(prefetched)) = (prepared_edge, prefetched_edges.as_mut()) {
                refresh_prefetched_edge(writer, expected, prefetched)?;
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
        if let Some(prefetched) = prefetched_nodes.as_mut() {
            refresh_prefetched_nodes(prefetched, &created, pattern);
        }
        if let (Some(expected), Some(prefetched)) = (prepared_edge, prefetched_edges.as_mut()) {
            refresh_prefetched_edge(writer, expected, prefetched)?;
        }
        Ok(vec![created])
    }
}

fn refresh_prefetched_edge(
    writer: &WriterSession,
    expected: &PreparedBoundRelationshipMerge,
    prefetched: &mut HashMap<EdgeMergeKey, Option<RelValue>>,
) -> Result<(), ExecError> {
    let (edge_type, src, dst) = &expected.key;
    match writer
        .staged_edge(edge_type, *src, *dst)
        .map_err(ExecError::Storage)?
    {
        StagedValue::Untouched => {}
        StagedValue::Tombstone => {
            prefetched.insert(expected.key.clone(), None);
        }
        StagedValue::Upsert(view) => {
            prefetched.insert(expected.key.clone(), Some(RelValue::from(view)));
        }
    }
    Ok(())
}

/// Refresh only the node aliases introduced by the MERGE pattern.
///
/// An input row may already bind the same physical node under another alias.
/// Those aliases are kept coherent by `synchronize_node_bindings`, but they
/// are not authoritative inputs to this cache: restricting the refresh to
/// pattern aliases prevents an unrelated/stale outer binding from replacing
/// the just-mutated MERGE value if a future row-producing operator constructs
/// bindings without going through SET synchronization.
fn refresh_prefetched_nodes(
    prefetched: &mut HashMap<NodeId, NodeValue>,
    row: &Row,
    pattern: &[CreateElement],
) {
    for alias in pattern.iter().filter_map(|element| match element {
        CreateElement::Node { alias, .. } => Some(alias),
        CreateElement::Rel { .. } => None,
    }) {
        if let Some(RuntimeValue::Node(node)) = row.get(alias) {
            prefetched.insert(node.id, (**node).clone());
        }
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
struct MergeMatchHints<'a> {
    prepared_node: Option<&'a MaterializedMergeNode>,
    prefetched_nodes: Option<&'a HashMap<NodeId, NodeValue>>,
    prepared_edge: Option<&'a PreparedBoundRelationshipMerge>,
    prefetched_edges: Option<&'a HashMap<EdgeMergeKey, Option<RelValue>>>,
}

#[allow(clippy::type_complexity)] // local BTreeMap of borrowed pattern slots
async fn find_merge_matches(
    pattern: &[CreateElement],
    outer_row: &Row,
    writer: &mut WriterSession,
    params: &Params,
    hints: MergeMatchHints<'_>,
) -> Result<Vec<Row>, ExecError> {
    let MergeMatchHints {
        prepared_node,
        prefetched_nodes,
        prepared_edge,
        prefetched_edges,
    } = hints;
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
                properties_spread,
            } => {
                // Carry the full label set: `MERGE (n:A:B)` matches a node that
                // carries BOTH labels, and creates one with both on miss.
                nodes.insert(
                    alias.as_str(),
                    MergeNodePattern {
                        labels: labels.as_slice(),
                        properties: properties.as_slice(),
                        properties_spread: properties_spread.as_ref(),
                    },
                );
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
        let (head_alias, head) = nodes.into_iter().next().expect("len == 1");
        let mut matched_rows: Vec<Row> = Vec::new();
        let candidates = match prepared_node {
            Some(expected) => {
                merge_node_candidates_materialized(head, expected, writer, prefetched_nodes).await?
            }
            None => merge_node_candidates(head, outer_row, writer, params).await?,
        };
        for node_val in candidates {
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
    let first_head_alias = match rels[0] {
        CreateElement::Rel { source_alias, .. } => source_alias.as_str(),
        _ => unreachable!("rels only contains Rel variants"),
    };
    let mut matched_rows: Vec<Row> =
        seed_merge_head(first_head_alias, &nodes, outer_row, writer, params).await?;
    let snap = writer.overlay_snapshot();

    for rel in &rels {
        let (
            rel_alias,
            rel_edge_type,
            rel_direction,
            rel_props,
            rel_properties_spread,
            source_alias,
            target_alias,
        ) = match rel {
            CreateElement::Rel {
                alias,
                edge_type,
                direction,
                properties,
                properties_spread,
                source_alias,
                target_alias,
            } => (
                alias.as_deref(),
                edge_type.as_str(),
                *direction,
                properties.as_slice(),
                properties_spread.as_ref(),
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
            let expected_rel_props = materialize_merge_rel_properties(
                rel_props,
                rel_properties_spread,
                &source_row,
                params,
            )?;
            let expected_tail = match &tail {
                MergeTail::Fresh(pattern) => Some(materialize_merge_node_pattern(
                    *pattern,
                    &source_row,
                    params,
                )?),
                MergeTail::BackReference { .. } => None,
            };
            let source_node_id = match source_row.get(source_alias) {
                Some(RuntimeValue::Node(n)) => n.id,
                _ => continue,
            };

            // Expand-Into for MERGE: when the tail is already bound, the
            // relationship's physical identity is fully known. Probe the
            // exact `(type, src, dst)` key instead of materialising every
            // relationship incident to the source and filtering by partner.
            // The storage point path reconciles memtable/SST versions and
            // tombstones, then decodes properties only for the winning row.
            if let MergeTail::BackReference { node_id, value } = &tail {
                let (physical_src, physical_dst) = match rel_direction {
                    RelationshipDirection::Right => (source_node_id, *node_id),
                    RelationshipDirection::Left => (*node_id, source_node_id),
                    RelationshipDirection::Both => {
                        return Err(ExecError::Runtime(
                            "MERGE relationship must be directed".into(),
                        ));
                    }
                };
                let prefetched_edge = prepared_edge
                    .filter(|expected| {
                        expected.key.0.as_str() == rel_edge_type
                            && expected.key.1 == physical_src
                            && expected.key.2 == physical_dst
                    })
                    .and_then(|expected| {
                        prefetched_edges.and_then(|edges| edges.get(&expected.key))
                    })
                    .cloned();
                // The common loader shape `MERGE (a)-[:R]->(b)` needs only an
                // existence answer when the relationship is anonymous and no
                // relationship properties participate in the pattern. Avoid
                // decoding the winning SST's property streams altogether.
                if expected_rel_props.is_empty() && rel_alias.is_none() {
                    let exists = match &prefetched_edge {
                        Some(edge) => edge.is_some(),
                        None => snap
                            .contains_edge_via_sst(rel_edge_type, physical_src, physical_dst)
                            .await
                            .map_err(ExecError::Storage)?,
                    };
                    if !exists {
                        continue;
                    }
                    let mut new_row = source_row;
                    new_row.set(
                        target_alias.to_string(),
                        RuntimeValue::Node(Box::new((**value).clone())),
                    );
                    next.push(new_row);
                    continue;
                }
                let rel_value = match prefetched_edge {
                    Some(Some(edge)) => edge,
                    Some(None) => continue,
                    None => {
                        let Some(edge) = snap
                            .lookup_edge_via_sst(rel_edge_type, physical_src, physical_dst)
                            .await
                            .map_err(ExecError::Storage)?
                        else {
                            continue;
                        };
                        RelValue::from(edge)
                    }
                };
                if !materialized_props_match(&expected_rel_props, &rel_value.properties) {
                    continue;
                }
                let mut new_row = source_row;
                new_row.set(
                    target_alias.to_string(),
                    RuntimeValue::Node(Box::new((**value).clone())),
                );
                if let Some(name) = rel_alias {
                    new_row.set(name.to_string(), RuntimeValue::Rel(Box::new(rel_value)));
                }
                next.push(new_row);
                continue;
            }

            // MERGE is a correlated relationship existence probe. Keep it on
            // the source-keyed SST path instead of the manifest-versioned CSR:
            // bulk loaders commit many small batches, so rebuilding the whole
            // edge type after every commit would make relationship MERGE
            // quadratic in the accumulated graph size. The full SST view is
            // also required for correctness — relationship properties may be
            // part of the MERGE pattern, returned through the alias, or used
            // by ON MATCH SET (which must preserve the existing property map).
            let neighbours = match rel_direction {
                RelationshipDirection::Right => {
                    snap.out_edges_via_sst(rel_edge_type, source_node_id).await
                }
                RelationshipDirection::Left => {
                    snap.in_edges_via_sst(rel_edge_type, source_node_id).await
                }
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
                    MergeTail::Fresh(pattern) => {
                        let view = match snap
                            .lookup_node(merge_scan_label(pattern.labels), partner_id)
                            .await
                            .map_err(ExecError::Storage)?
                        {
                            Some(v) => v,
                            None => continue,
                        };
                        let partner = NodeValue::from(view);
                        if !materialized_node_matches(
                            &partner,
                            pattern.labels,
                            expected_tail.as_ref().expect("fresh tail was materialized"),
                        ) {
                            continue;
                        }
                        partner
                    }
                    MergeTail::BackReference { .. } => {
                        unreachable!("bound tails use the exact endpoint probe above")
                    }
                };
                let rel_value = RelValue::from(e);
                if !materialized_props_match(&expected_rel_props, &rel_value.properties) {
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

/// Borrowed node-pattern metadata retained while matching a MERGE. Unlike the
/// old `(labels, properties)` tuple, this deliberately carries the optional
/// `$props` spread: ignoring it made `MERGE (n:L $props)` match every `:L`
/// node regardless of the supplied key.
#[derive(Clone, Copy)]
struct MergeNodePattern<'a> {
    labels: &'a [String],
    properties: &'a [(String, Expression)],
    properties_spread: Option<&'a Expression>,
}

/// `alias -> node pattern` map built once per MERGE call. Lives only as long
/// as `find_merge_matches` borrows the lowered pattern.
type MergeNodeMap<'a> = BTreeMap<&'a str, MergeNodePattern<'a>>;

/// One node MERGE pattern evaluated against its current outer row. Both
/// runtime and storage values are kept: runtime values preserve Cypher's
/// comparison semantics for residual predicates, while storage values are
/// the canonical keys consumed by `WriterSession::unique_probe`.
struct MaterializedMergeNode {
    properties: BTreeMap<String, RuntimeValue>,
    core_properties: BTreeMap<String, CoreValue>,
    explicit_id: Option<NodeId>,
}

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

/// Evaluate a node pattern's spread and explicit map exactly once for the
/// current outer row. Explicit entries override spread entries, matching the
/// create branch. `_id` is an engine NodeId selector rather than a stored
/// property and is therefore split out for the direct lookup path.
fn materialize_merge_node_pattern(
    pattern: MergeNodePattern<'_>,
    row: &Row,
    params: &Params,
) -> Result<MaterializedMergeNode, ExecError> {
    let mut core_properties = BTreeMap::new();
    let mut properties = BTreeMap::new();
    let mut explicit_id = None;
    if let Some(spread_expr) = pattern.properties_spread {
        apply_spread_properties(
            spread_expr,
            row,
            params,
            &mut core_properties,
            &mut properties,
            &mut explicit_id,
            None,
        )?;
    }
    for (key, expr) in pattern.properties {
        let value = evaluate(expr, row, params)?;
        if key == "_id" {
            explicit_id = Some(crate::exec::walker::node_id_from_value(&value, expr.span)?);
            continue;
        }
        let core = runtime_to_core(&value, expr).map_err(ExecError::Runtime)?;
        core_properties.insert(key.clone(), core);
        properties.insert(key.clone(), value);
    }
    Ok(MaterializedMergeNode {
        properties,
        core_properties,
        explicit_id,
    })
}

/// Evaluate a relationship MERGE property map, including `$props`. `_id`
/// remains invalid for relationships, exactly as on the create branch.
fn materialize_merge_rel_properties(
    properties: &[(String, Expression)],
    properties_spread: Option<&Expression>,
    row: &Row,
    params: &Params,
) -> Result<BTreeMap<String, RuntimeValue>, ExecError> {
    let mut core_properties = BTreeMap::new();
    let mut runtime_properties = BTreeMap::new();
    if let Some(spread_expr) = properties_spread {
        let mut invalid_id = None;
        apply_spread_properties(
            spread_expr,
            row,
            params,
            &mut core_properties,
            &mut runtime_properties,
            &mut invalid_id,
            Some("_id is not valid on a relationship MERGE"),
        )?;
    }
    for (key, expr) in properties {
        if key == "_id" {
            return Err(ExecError::Runtime(
                "_id is not valid on a relationship MERGE".into(),
            ));
        }
        let value = evaluate(expr, row, params)?;
        // Run the same storability validation as the create branch even
        // though only runtime values are needed for the residual comparison.
        let core = runtime_to_core(&value, expr).map_err(ExecError::Runtime)?;
        core_properties.insert(key.clone(), core);
        runtime_properties.insert(key.clone(), value);
    }
    Ok(runtime_properties)
}

fn materialized_props_match(
    expected: &BTreeMap<String, RuntimeValue>,
    actual: &BTreeMap<String, RuntimeValue>,
) -> bool {
    expected.iter().all(|(key, value)| {
        actual
            .get(key)
            .is_some_and(|v| runtime_values_equal(v, value))
    })
}

fn materialized_node_matches(
    node: &NodeValue,
    required_labels: &[String],
    expected: &MaterializedMergeNode,
) -> bool {
    node_has_all_labels(node, required_labels)
        && expected.explicit_id.is_none_or(|id| node.id == id)
        && materialized_props_match(&expected.properties, &node.properties)
}

/// Unique keys declared by the schema and fully covered by this MERGE node
/// pattern. Sidecar-compatible single-String keys come first so batch seeding
/// and the canonical probe both reuse their preseeded O(1) postings instead
/// of populating a composite index with a label scan. Remaining constraints
/// are ordered longest first; legacy single-property `PropertyDef::unique`
/// flags are included even when the named-constraint list predates them.
/// Labels may be secondary in
/// `MERGE (n:A:B)`: the lookup is scoped to the label that owns the
/// constraint, then all required labels are checked as residuals.
fn covered_merge_unique_keys(
    writer: &WriterSession,
    labels: &[String],
    expected: &MaterializedMergeNode,
) -> Vec<(String, Vec<(String, CoreValue)>)> {
    let schema = writer.schema();
    let mut specs: BTreeSet<(String, Vec<String>)> = BTreeSet::new();
    for constraint in schema.constraints() {
        if constraint.kind != namidb_core::ConstraintKind::Unique
            || !labels.iter().any(|l| l == &constraint.label)
        {
            continue;
        }
        let mut names = constraint.properties.clone();
        names.sort();
        if names
            .iter()
            .all(|name| expected.core_properties.contains_key(name))
        {
            specs.insert((constraint.label.clone(), names));
        }
    }
    for label in labels {
        let Some(def) = schema.label(label) else {
            continue;
        };
        for property in &def.properties {
            if property.unique && expected.core_properties.contains_key(&property.name) {
                specs.insert((label.clone(), vec![property.name.clone()]));
            }
        }
    }
    let mut out: Vec<_> = specs
        .into_iter()
        .map(|(label, names)| {
            let values: Vec<(String, CoreValue)> = names
                .into_iter()
                .map(|name| {
                    let value = expected
                        .core_properties
                        .get(&name)
                        .expect("covered key")
                        .clone();
                    (name, value)
                })
                .collect();
            (label, values)
        })
        .collect();
    out.sort_by(|a, b| {
        let a_seeded = matches!(a.1.as_slice(), [(_, CoreValue::Str(_))]);
        let b_seeded = matches!(b.1.as_slice(), [(_, CoreValue::Str(_))]);
        b_seeded
            .cmp(&a_seeded)
            .then_with(|| b.1.len().cmp(&a.1.len()))
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

/// First declared non-unique equality index covered by the pattern. String
/// values use the equality posting-list sidecar; other values deliberately
/// fall through to the typed/full scan path because storage's equality
/// sidecar is currently string-only.
fn covered_merge_equality_key(
    writer: &WriterSession,
    labels: &[String],
    expected: &MaterializedMergeNode,
) -> Option<(String, String, String)> {
    for label in labels {
        let Some(def) = writer.schema().label(label) else {
            continue;
        };
        for property in &def.properties {
            if property.indexed && !property.unique {
                if let Some(RuntimeValue::String(value)) = expected.properties.get(&property.name) {
                    return Some((label.clone(), property.name.clone(), value.clone()));
                }
            }
        }
    }
    None
}

/// Candidate selection for a fresh node in a MERGE pattern:
///
/// 1. `_id` → direct NodeId lookup.
/// 2. Fully-covered unique (including composite) key → transactional O(1)
///    probe plus direct NodeId lookup. The index sees committed + staged rows.
/// 3. Declared non-unique equality index → posting-list lookup.
/// 4. No usable index → legacy label scan.
///
/// Every fast path applies the full label/property residual afterwards, so
/// the optimization cannot weaken MERGE's whole-pattern semantics.
async fn merge_node_candidates(
    pattern: MergeNodePattern<'_>,
    row: &Row,
    writer: &WriterSession,
    params: &Params,
) -> Result<Vec<NodeValue>, ExecError> {
    let expected = materialize_merge_node_pattern(pattern, row, params)?;
    merge_node_candidates_materialized(pattern, &expected, writer, None).await
}

async fn merge_node_candidates_materialized(
    pattern: MergeNodePattern<'_>,
    expected: &MaterializedMergeNode,
    writer: &WriterSession,
    prefetched_nodes: Option<&HashMap<NodeId, NodeValue>>,
) -> Result<Vec<NodeValue>, ExecError> {
    if let Some(id) = expected.explicit_id {
        let snap = writer.overlay_snapshot();
        let found = if pattern.labels.is_empty() {
            crate::exec::walker::scan_node_for_id(&snap, id).await?
        } else {
            snap.lookup_node(merge_scan_label(pattern.labels), id)
                .await
                .map_err(ExecError::Storage)?
        };
        return Ok(found
            .map(NodeValue::from)
            .filter(|node| materialized_node_matches(node, pattern.labels, expected))
            .into_iter()
            .collect());
    }

    if let Some((label, candidate_ids)) =
        probe_merge_unique_candidates(writer, pattern.labels, expected).await?
    {
        // Absence across every Cypher-equal encoding proves that the full
        // pattern cannot match. For numeric keys this includes both strict
        // I64/F64 storage domains.
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut found = Vec::with_capacity(candidate_ids.len());
        let mut unresolved = Vec::new();
        for id in candidate_ids {
            if let Some(node) = prefetched_nodes.and_then(|nodes| nodes.get(&id)).cloned() {
                if materialized_node_matches(&node, pattern.labels, expected) {
                    found.push(node);
                }
            } else {
                unresolved.push(id);
            }
        }
        if !unresolved.is_empty() {
            let snap = writer.overlay_snapshot();
            let resolved = snap
                .batch_lookup_nodes(&label, &unresolved)
                .await
                .map_err(ExecError::Storage)?;
            for node in resolved.into_iter().flatten().map(NodeValue::from) {
                if materialized_node_matches(&node, pattern.labels, expected) {
                    found.push(node);
                }
            }
        }
        return Ok(found);
    }

    if let Some((label, property, value)) =
        covered_merge_equality_key(writer, pattern.labels, expected)
    {
        // MERGE can create/update a node, invalidating committed claimant
        // caches at every auto-commit. Use the writer-private map even before
        // the first staged mutation so one-row statements stay O(1) after one
        // population.
        let snap = writer.transactional_overlay_snapshot();
        let lookup_value = RuntimeValue::String(value);
        let candidates = crate::exec::walker::lookup_nodes_by_property_via_scan(
            &snap,
            &label,
            &property,
            &lookup_value,
        )
        .await?;
        return Ok(candidates
            .into_iter()
            .map(NodeValue::from)
            .filter(|node| materialized_node_matches(node, pattern.labels, expected))
            .collect());
    }

    let snap = writer.overlay_snapshot();
    let candidates = snap
        .scan_label(merge_scan_label(pattern.labels))
        .await
        .map_err(ExecError::Storage)?;
    Ok(candidates
        .into_iter()
        .map(NodeValue::from)
        .filter(|node| materialized_node_matches(node, pattern.labels, expected))
        .collect())
}

/// Return the authoritative candidate ids from the first fully-covered,
/// indexable unique key. `Some((label, empty))` proves a miss; `None` means no
/// covered key can safely answer this runtime value and the caller must try
/// an equality index or scan.
async fn probe_merge_unique_candidates(
    writer: &WriterSession,
    labels: &[String],
    expected: &MaterializedMergeNode,
) -> Result<Option<(String, BTreeSet<NodeId>)>, ExecError> {
    for (label, tuple) in covered_merge_unique_keys(writer, labels, expected) {
        let Some(variants) = merge_unique_probe_variants(&tuple) else {
            continue;
        };
        let mut candidate_ids = BTreeSet::new();
        let mut fully_indexable = true;
        for variant in variants {
            let refs: Vec<(&str, &CoreValue)> = variant
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect();
            match writer
                .unique_probe(&label, &refs, None)
                .await
                .map_err(ExecError::Storage)?
            {
                UniqueProbe::Conflict(id) => {
                    candidate_ids.insert(id);
                }
                UniqueProbe::NoConflict => {}
                // NULL / list / map / vector / ambiguous large-float keys
                // cannot prove absence through this index. Try another
                // covered key, then fall through to equality/label scan.
                UniqueProbe::Unindexable => {
                    fully_indexable = false;
                    break;
                }
            }
        }
        if !fully_indexable {
            continue;
        }
        return Ok(Some((label, candidate_ids)));
    }
    Ok(None)
}

/// Strict storage uniqueness distinguishes integers from floats, while Cypher
/// equality considers `1` and `1.0` equal. Produce the small Cartesian set of
/// strict tuples that can satisfy the runtime predicate so a negative result
/// remains proof of absence without giving numeric bulk loads an O(N) scan.
///
/// Integer→float has exactly one counterpart (`n as f64`). A finite integral
/// float below 2^53 has exactly one integer counterpart. Above that boundary
/// several adjacent i64 values may round to the same f64, so the safe answer
/// is `None` (fall back to a scan). The variant cap prevents pathological
/// user-defined composite constraints from creating exponential work.
fn merge_unique_probe_variants(
    tuple: &[(String, CoreValue)],
) -> Option<Vec<Vec<(String, CoreValue)>>> {
    const MAX_VARIANTS: usize = 256;
    const MAX_EXACT_INTEGER_F64: f64 = 9_007_199_254_740_992.0; // 2^53

    let mut variants: Vec<Vec<(String, CoreValue)>> = vec![Vec::with_capacity(tuple.len())];
    for (name, value) in tuple {
        let options = match value {
            CoreValue::I64(n) => vec![CoreValue::I64(*n), CoreValue::F64(*n as f64)],
            CoreValue::F64(f) if f.is_nan() => return None,
            CoreValue::F64(f)
                if f.is_finite() && f.fract() == 0.0 && f.abs() < MAX_EXACT_INTEGER_F64 =>
            {
                let n = *f as i64;
                debug_assert_eq!(n as f64, *f);
                vec![CoreValue::F64(*f), CoreValue::I64(n)]
            }
            CoreValue::F64(f)
                if f.is_finite() && f.fract() == 0.0 && f.abs() >= MAX_EXACT_INTEGER_F64 =>
            {
                return None;
            }
            _ => vec![value.clone()],
        };
        if variants.len().checked_mul(options.len())? > MAX_VARIANTS {
            return None;
        }
        let mut expanded = Vec::with_capacity(variants.len() * options.len());
        for prefix in variants {
            for option in &options {
                let mut next = prefix.clone();
                next.push((name.clone(), option.clone()));
                expanded.push(next);
            }
        }
        variants = expanded;
    }
    Some(variants)
}

/// Seed `find_merge_matches` for an N-hop chain. The "head" alias is
/// the source of the first rel. If the pattern declares it as a fresh
/// Node we use the same indexed candidate selection as single-node MERGE; if
/// the caller already bound it on the outer row (back-reference) we lift that
/// NodeValue verbatim.
async fn seed_merge_head(
    head_alias: &str,
    nodes: &MergeNodeMap<'_>,
    outer_row: &Row,
    writer: &WriterSession,
    params: &Params,
) -> Result<Vec<Row>, ExecError> {
    if let Some(pattern) = nodes.get(head_alias).copied() {
        let mut out = Vec::new();
        for node_val in merge_node_candidates(pattern, outer_row, writer, params).await? {
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
    Fresh(MergeNodePattern<'a>),
    BackReference {
        node_id: NodeId,
        value: Box<NodeValue>,
    },
}

impl<'a> MergeTail<'a> {
    fn resolve(alias: &str, nodes: &MergeNodeMap<'a>, outer_row: &Row) -> Result<Self, ExecError> {
        if let Some(pattern) = nodes.get(alias).copied() {
            return Ok(MergeTail::Fresh(pattern));
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

fn runtime_values_equal(a: &RuntimeValue, b: &RuntimeValue) -> bool {
    match (a, b) {
        // This is persisted-pattern equivalence, not the three-valued Cypher
        // `=` operator: `materialized_props_match` has already established
        // that the property key exists on both sides. Since NAMIDB currently
        // permits an explicit null to round-trip through CREATE/MERGE, it must
        // match itself or every replay would take the create branch.
        (RuntimeValue::Null, RuntimeValue::Null) => true,
        (RuntimeValue::Integer(x), RuntimeValue::Integer(y)) => x == y,
        (RuntimeValue::Float(x), RuntimeValue::Float(y)) => x == y,
        (RuntimeValue::Integer(x), RuntimeValue::Float(y))
        | (RuntimeValue::Float(y), RuntimeValue::Integer(x)) => (*x as f64) == *y,
        (RuntimeValue::Bool(x), RuntimeValue::Bool(y)) => x == y,
        (RuntimeValue::String(x), RuntimeValue::String(y)) => x == y,
        (RuntimeValue::Bytes(x), RuntimeValue::Bytes(y)) => x == y,
        (RuntimeValue::Vector(x), RuntimeValue::Vector(y)) => x == y,
        (
            RuntimeValue::Vector8 {
                codes: x_codes,
                scale: x_scale,
            },
            RuntimeValue::Vector8 {
                codes: y_codes,
                scale: y_scale,
            },
        ) => x_codes == y_codes && x_scale == y_scale,
        (RuntimeValue::Date(x), RuntimeValue::Date(y)) => x == y,
        (RuntimeValue::DateTime(x), RuntimeValue::DateTime(y)) => x == y,
        (RuntimeValue::List(x), RuntimeValue::List(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|(left, right)| runtime_values_equal(left, right))
        }
        (RuntimeValue::Map(x), RuntimeValue::Map(y)) => {
            x.len() == y.len()
                && x.iter().all(|(key, left)| {
                    y.get(key)
                        .is_some_and(|right| runtime_values_equal(left, right))
                })
        }
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
        RuntimeValue::Vector8 { codes, scale } => Ok(CoreValue::VecI8 {
            codes: codes.clone(),
            scale: *scale,
        }),
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
        RuntimeValue::Vector8 { codes, scale } => Ok(CoreValue::VecI8 {
            codes: codes.clone(),
            scale: *scale,
        }),
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

    #[test]
    fn merge_property_equivalence_covers_nested_and_typed_storage_values() {
        let left = RuntimeValue::Map(BTreeMap::from([
            (
                "nested".into(),
                RuntimeValue::List(vec![
                    RuntimeValue::Integer(7),
                    RuntimeValue::Map(BTreeMap::from([
                        ("nullable".into(), RuntimeValue::Null),
                        ("ratio".into(), RuntimeValue::Float(3.0)),
                    ])),
                ]),
            ),
            ("bytes".into(), RuntimeValue::Bytes(vec![0, 1, 255])),
            ("vector".into(), RuntimeValue::Vector(vec![0.25, -1.5])),
            ("date".into(), RuntimeValue::Date(20_000)),
            (
                "datetime".into(),
                RuntimeValue::DateTime(1_700_000_000_123_456),
            ),
        ]));
        let right = RuntimeValue::Map(BTreeMap::from([
            (
                "nested".into(),
                RuntimeValue::List(vec![
                    // Numeric coercion must apply recursively too.
                    RuntimeValue::Float(7.0),
                    RuntimeValue::Map(BTreeMap::from([
                        ("nullable".into(), RuntimeValue::Null),
                        ("ratio".into(), RuntimeValue::Integer(3)),
                    ])),
                ]),
            ),
            ("bytes".into(), RuntimeValue::Bytes(vec![0, 1, 255])),
            ("vector".into(), RuntimeValue::Vector(vec![0.25, -1.5])),
            ("date".into(), RuntimeValue::Date(20_000)),
            (
                "datetime".into(),
                RuntimeValue::DateTime(1_700_000_000_123_456),
            ),
        ]));

        assert!(runtime_values_equal(&left, &right));

        let mut changed = match right {
            RuntimeValue::Map(values) => values,
            _ => unreachable!(),
        };
        changed.insert("date".into(), RuntimeValue::Date(20_001));
        assert!(!runtime_values_equal(&left, &RuntimeValue::Map(changed)));
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
