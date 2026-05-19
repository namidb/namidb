//! Write-aware executor.
//!
//! Sister to [`crate::exec::walker::execute`]. Drives a [`LogicalPlan`]
//! containing write operators (Create / Merge / Set / Remove / Delete)
//! against a mutable [`WriterSession`], delegating read sub-plans back to
//! the read-only walker. Auto-commits at the end of the query.
//!
//! See [`docs/rfc/009-write-clauses.md`](../../../../docs/rfc/009-write-clauses.md).
//!
//! Limitations of v0:
//!
//! - No read-your-own-writes: each read sub-plan sees the pre-call
//! snapshot. Pieces created mid-query are not visible until commit.
//! - MERGE matches by single-element pattern (one node or
//! node-rel-node chain).
//! - DETACH DELETE enumerates incident edges across every edge_type
//! declared on the manifest schema.
//! - Property values must be representable as `core::Value` scalars
//! (List/Map/Node/Rel are rejected with an explicit error).

use std::collections::BTreeMap;

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
}

/// Execute `plan` against `writer`. Reads pin a snapshot per read
/// sub-plan; writes go through `writer.upsert_*` / `tombstone_*`. At the
/// end, `writer.commit_batch()` makes every mutation durable.
pub async fn execute_write(
 plan: &LogicalPlan,
 writer: &mut WriterSession,
 params: &Params,
) -> Result<WriteOutcome, ExecError> {
 let mut outcome = WriteOutcome::default();
 let rows = execute_write_inner(plan, writer, params, &mut outcome).await?;
 writer.commit_batch().await.map_err(ExecError::Storage)?;
 outcome.rows = rows;
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
 let new_row = apply_create(elements, row, writer, params, outcome)?;
 out.push(new_row);
 }
 Ok(out)
 }

 LogicalPlan::Set { input, items } => {
 let rows = execute_write_inner(input, writer, params, outcome).await?;
 let mut out = Vec::with_capacity(rows.len());
 for row in rows {
 let new_row = apply_sets(items, row, writer, params, outcome)?;
 out.push(new_row);
 }
 Ok(out)
 }

 LogicalPlan::Remove { input, items } => {
 let rows = execute_write_inner(input, writer, params, outcome).await?;
 let mut out = Vec::with_capacity(rows.len());
 for row in rows {
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
 let mut rows = execute_write_inner(input, writer, params, outcome).await?;
 if !keys.is_empty() {
 crate::exec::walker::sort_rows(&mut rows, keys, params)?;
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
 let snap = writer.snapshot();
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
 let snap = writer.snapshot();
 let mut out = Vec::with_capacity(input_rows.len());
 for row in input_rows {
 let id_value = evaluate(id, &row, params)?;
 let node_id = crate::exec::walker::node_id_from_value(&id_value, id.span)?;
 if let Some(view) = snap
 .lookup_node(label, node_id)
 .await
 .map_err(ExecError::Storage)?
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

 // ─── Pure read leaves and pattern-driven operators that do
 // NOT contain writes: delegate to the read-only walker on a
 // freshly pinned snapshot. v0: no read-your-own-writes.
 LogicalPlan::Empty
 | LogicalPlan::NodeScan { .. }
 | LogicalPlan::Argument { .. }
 | LogicalPlan::Expand { .. }
 | LogicalPlan::SemiApply { .. }
 | LogicalPlan::PatternList { .. } => {
 let snap = writer.snapshot();
 execute_inner(plan, &snap, params, None).await
 }
 }
 }
 .boxed()
}

// ──────────────────────────── CREATE ─────────────────────────────────

fn apply_create(
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
 label,
 properties,
 } => {
 // Back-reference: don't create if already bound.
 if row.get(alias).is_some() {
 continue;
 }
 let mut core_props = BTreeMap::new();
 let mut runtime_props = BTreeMap::new();
 let mut explicit_id: Option<NodeId> = None;
 for (k, expr) in properties {
 let v = evaluate(expr, &row, params)?;
 if k == "id" {
 // `{id: ...}` becomes the storage NodeId; not
 // stored as a regular property. The `.id`
 // accessor materialises the NodeId on read.
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
 let record = NodeWriteRecord {
 properties: core_props,
 schema_version: 1,
 };
 writer
 .upsert_node(label.clone(), id, &record)
 .map_err(ExecError::Storage)?;
 outcome.nodes_created += 1;
 let node_value = NodeValue {
 id,
 label: label.clone(),
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

fn apply_sets(
 items: &[SetOp],
 mut row: Row,
 writer: &mut WriterSession,
 params: &Params,
 outcome: &mut WriteOutcome,
) -> Result<Row, ExecError> {
 for op in items {
 row = apply_set(op, row, writer, params, outcome)?;
 }
 Ok(row)
}

fn apply_set(
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
 let mut core_props = node_runtime_props_to_core(&n.properties)?;
 core_props.insert(key.clone(), core);
 let record = NodeWriteRecord {
 properties: core_props,
 schema_version: 1,
 };
 writer
 .upsert_node(n.label.clone(), n.id, &record)
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
 SetOp::Replace { target_alias, .. } | SetOp::Merge { target_alias, .. } => {
 return Err(ExecError::Runtime(format!(
 "SET {} = {{...}} / += {{...}} land (use SET prop = value forms)",
 target_alias
 )));
 }
 SetOp::Labels { target_alias, .. } => {
 return Err(ExecError::Runtime(format!(
 "SET {}:Label lands (manifest schema labelset mutation)",
 target_alias
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
 let mut core_props = node_runtime_props_to_core(&n.properties)?;
 core_props.remove(key);
 let record = NodeWriteRecord {
 properties: core_props,
 schema_version: 1,
 };
 writer
 .upsert_node(n.label.clone(), n.id, &record)
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
 RemoveOp::Labels { target_alias, .. } => {
 return Err(ExecError::Runtime(format!(
 "REMOVE {}:Label lands",
 target_alias
 )));
 }
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
 writer
 .tombstone_node(n.label.clone(), n.id)
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
 let mut to_delete: Vec<(NodeId, NodeId)> = Vec::new();
 {
 let snap = writer.snapshot();
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
 for (src, dst) in to_delete {
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
 m_row = apply_set(op, m_row, writer, params, outcome)?;
 }
 out.push(m_row);
 }
 Ok(out)
 } else {
 // Create branch.
 let created = apply_create(pattern, row, writer, params, outcome)?;
 let mut created = created;
 for op in on_create_sets {
 created = apply_set(op, created, writer, params, outcome)?;
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
async fn find_merge_matches(
 pattern: &[CreateElement],
 outer_row: &Row,
 writer: &mut WriterSession,
 params: &Params,
) -> Result<Vec<Row>, ExecError> {
 // Split the pattern into Nodes (by alias) and Rels (in insertion
 // order). v0 supports either a single Node, or exactly one Rel
 // joining two Nodes.
 let mut nodes: BTreeMap<&str, (&str, &[(String, Expression)])> = BTreeMap::new();
 let mut rels: Vec<&CreateElement> = Vec::new();
 for el in pattern {
 match el {
 CreateElement::Node {
 alias,
 label,
 properties,
 } => {
 nodes.insert(alias.as_str(), (label.as_str(), properties.as_slice()));
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
 let (head_alias, (head_label, head_props)) =
 nodes.into_iter().next().expect("len == 1");
 let snap = writer.snapshot();
 let candidates = snap
 .scan_label(head_label)
 .await
 .map_err(ExecError::Storage)?;
 let mut matched_rows: Vec<Row> = Vec::new();
 for view in candidates {
 let node_val = NodeValue::from(view);
 if !merge_props_match(head_props, &node_val.properties, outer_row, params)? {
 continue;
 }
 let mut new_row = outer_row.clone();
 new_row.set(head_alias.to_string(), RuntimeValue::Node(Box::new(node_val)));
 matched_rows.push(new_row);
 }
 return Ok(matched_rows);
 }

 if rels.len() != 1 || nodes.len() != 2 {
 return Err(ExecError::Runtime(
 "MERGE patterns with more than one relationship hop are not supported yet".into(),
 ));
 }

 let (rel_edge_type, rel_direction, rel_props, head_alias, tail_alias) = match rels[0] {
 CreateElement::Rel {
 edge_type,
 direction,
 properties,
 source_alias,
 target_alias,
 ..
 } => (
 edge_type.as_str(),
 *direction,
 properties.as_slice(),
 source_alias.as_str(),
 target_alias.as_str(),
 ),
 _ => unreachable!("rels only contains Rel variants"),
 };

 let (head_label, head_props) = *nodes
 .get(head_alias)
 .ok_or_else(|| ExecError::Runtime(format!("MERGE head `{}` not found", head_alias)))?;
 let (tail_label, tail_props) = *nodes
 .get(tail_alias)
 .ok_or_else(|| ExecError::Runtime(format!("MERGE tail `{}` not found", tail_alias)))?;

 let snap = writer.snapshot();
 let candidates = snap
 .scan_label(head_label)
 .await
 .map_err(ExecError::Storage)?;

 let mut matched_rows: Vec<Row> = Vec::new();
 for view in candidates {
 let node_val = NodeValue::from(view);
 if !merge_props_match(head_props, &node_val.properties, outer_row, params)? {
 continue;
 }
 let mut new_row = outer_row.clone();
 new_row.set(head_alias.to_string(), RuntimeValue::Node(Box::new(node_val)));
 matched_rows.push(new_row);
 }

 let mut chained: Vec<Row> = Vec::new();
 for head_row in matched_rows {
 let head_node_id = match head_row.get(head_alias) {
 Some(RuntimeValue::Node(n)) => n.id,
 _ => continue,
 };
 let neighbours = match rel_direction {
 RelationshipDirection::Right => snap.out_edges(rel_edge_type, head_node_id).await,
 RelationshipDirection::Left => snap.in_edges(rel_edge_type, head_node_id).await,
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
 let partner = match snap
 .lookup_node(tail_label, partner_id)
 .await
 .map_err(ExecError::Storage)?
 {
 Some(v) => NodeValue::from(v),
 None => continue,
 };
 if !merge_props_match(tail_props, &partner.properties, &head_row, params)? {
 continue;
 }
 let rel_value = RelValue::from(e);
 if !merge_props_match(rel_props, &rel_value.properties, &head_row, params)? {
 continue;
 }
 let mut new_row = head_row.clone();
 new_row.set(tail_alias.to_string(), RuntimeValue::Node(Box::new(partner)));
 chained.push(new_row);
 }
 }
 Ok(chained)
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
 other => Err(format!(
 "property value at `{}` is {} — only scalars are storable in v0",
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
 match v {
 RuntimeValue::Null => {
 out.insert(k.clone(), CoreValue::Null);
 }
 RuntimeValue::Bool(b) => {
 out.insert(k.clone(), CoreValue::Bool(*b));
 }
 RuntimeValue::Integer(n) => {
 out.insert(k.clone(), CoreValue::I64(*n));
 }
 RuntimeValue::Float(f) => {
 out.insert(k.clone(), CoreValue::F64(*f));
 }
 RuntimeValue::String(s) => {
 out.insert(k.clone(), CoreValue::Str(s.clone()));
 }
 RuntimeValue::Bytes(b) => {
 out.insert(k.clone(), CoreValue::Bytes(b.clone()));
 }
 RuntimeValue::Vector(v) => {
 out.insert(k.clone(), CoreValue::Vec(v.clone()));
 }
 other => {
 return Err(ExecError::Runtime(format!(
 "property `{}` is {} — non-scalar values cannot round-trip through storage in v0",
 k,
 other.type_name()
 )));
 }
 }
 }
 Ok(out)
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
 assert_eq!(n.label, "Person");
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
}
