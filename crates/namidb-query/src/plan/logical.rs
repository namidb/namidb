//! Logical plan IR.
//!
//! See [`docs/rfc/008-logical-plan-ir.md`](../../../../docs/rfc/008-logical-plan-ir.md).

use namidb_storage::sst::predicates::ScanPredicate;

use crate::parser::{Expression, OrderDirection, RelationshipDirection, RelationshipLength};

/// Tree of relational/graph operators produced by lowering the AST and
/// consumed by the executor. See RFC-008.
#[derive(Clone, Debug, PartialEq)]
pub enum LogicalPlan {
 /// Scan all nodes carrying `label`. `predicates` is the list of
 /// single-column conjunctive predicates pushed by `optimize`
 /// (RFC-013) — empty when lowering, populated when a `Filter`
 /// directly above carried conjuncts the storage layer can use for
 /// row-group pruning. `projection` (RFC-015) is `None` when the
 /// query references the binding as a whole or no projection
 /// analysis ran; `Some(cols)` ⇒ storage decodes only those
 /// property columns plus the engine columns.
 NodeScan {
 /// `Some(label)` restricts the scan to one label; `None`
 /// fans out across every label observable through the
 /// snapshot (`Snapshot::observed_labels`). Lowering produces
 /// `None` for `MATCH (n)` without a label predicate.
 label: Option<String>,
 alias: String,
 predicates: Vec<ScanPredicate>,
 projection: Option<Vec<String>>,
 },

 /// Point-lookup by id. Lowering emits this when the AST contains an
 /// inline filter `{id: <expr>}` on a node pattern. The id expression
 /// is evaluated against each row produced by `input` (typically
 /// `Empty` for queries that start with the lookup, or a populated
 /// plan when the lookup follows UNWIND / WITH).
 NodeById {
 input: Box<LogicalPlan>,
 label: String,
 alias: String,
 id: Expression,
 },

 /// Expand `source` across an edge to produce `target_alias`.
 Expand {
 input: Box<LogicalPlan>,
 source: String,
 edge_type: Option<String>,
 direction: RelationshipDirection,
 rel_alias: Option<String>,
 target_alias: String,
 /// Label declared on the target node pattern, if any. The
 /// executor uses it to issue `lookup_node(label, id)` directly
 /// instead of probing every label in the schema.
 target_label: Option<String>,
 length: Option<RelationshipLength>,
 /// `true` for `OPTIONAL MATCH`: emit a row with `target=NULL`
 /// when no neighbour matches.
 optional: bool,
 /// `true` when `target_alias` was already bound in scope before
 /// this Expand (LDBC IC01-shaped path-existence patterns:
 /// `MATCH (p), (f) ... MATCH (p)-[:KNOWS*1..3]-(f)`). The
 /// executor must NOT overwrite the existing binding; instead,
 /// at every emission point it asserts the discovered tail node
 /// is identical to the bound target. Variable-length traversals
 /// still explore the frontier freely, but only paths that
 /// terminate at the existing target survive.
 back_reference: bool,
 },

 /// Selection predicate.
 Filter {
 input: Box<LogicalPlan>,
 predicate: Expression,
 },

 /// Projection (RETURN / WITH).
 Project {
 input: Box<LogicalPlan>,
 items: Vec<ProjectionItem>,
 distinct: bool,
 /// `true` for RETURN — drops every non-projected binding. `false`
 /// for WITH (keeps bindings live so the next clause can reference
 /// them, even if the parser doesn't materialise that yet).
 discard_input_bindings: bool,
 },

 /// Aggregate (group_by + aggregation expressions).
 Aggregate {
 input: Box<LogicalPlan>,
 group_by: Vec<(Expression, String)>,
 aggregations: Vec<(String, AggregateExpr)>,
 },

 /// Sort + skip + limit fused. Pure sort: `skip=0, limit=u64::MAX`.
 /// Pure limit: `keys.is_empty()`.
 TopN {
 input: Box<LogicalPlan>,
 keys: Vec<OrderKey>,
 skip: u64,
 limit: u64,
 },

 /// Distinct across every visible binding.
 Distinct { input: Box<LogicalPlan> },

 /// UNION (set) or UNION ALL (multiset).
 Union {
 left: Box<LogicalPlan>,
 right: Box<LogicalPlan>,
 all: bool,
 },

 /// Expand a list expression to one row per element.
 Unwind {
 input: Box<LogicalPlan>,
 list: Expression,
 alias: String,
 },

 /// Single empty driver row — used when the query starts with `UNWIND`
 /// or with a leading `WITH` literal.
 Empty,

 /// Cartesian product of `left` × `right`. Used to combine pattern
 /// parts inside a multi-pattern `MATCH ... , ... ` clause where the
 /// parts share no binding. Naïve nested-loop in v0; hash join when
 /// shared bindings exist arrives.
 CrossProduct {
 left: Box<LogicalPlan>,
 right: Box<LogicalPlan>,
 },

 /// Inner hash equi-join (RFC-012). Build a hash table over
 /// `build`'s rows keyed by each `JoinKey::build_side`; stream
 /// `probe` and look up matches via `JoinKey::probe_side`. Emit one
 /// row per match with bindings from both sides combined.
 ///
 /// `residual` is any non-equi predicate left over from the
 /// pre-conversion Filter (e.g. `a.z >= b.w` in
 /// `WHERE a.x = b.y AND a.z >= b.w`). Evaluated 3VL on the joined
 /// row; False / NULL drop the row.
 HashJoin {
 build: Box<LogicalPlan>,
 probe: Box<LogicalPlan>,
 on: Vec<JoinKey>,
 residual: Option<Expression>,
 },

 /// Decorrelated semi-join (RFC-014). Functionally
 /// equivalent to `SemiApply` but executes the subplan as `inner`
 /// ONCE (not per outer row) and builds a key set; the outer
 /// (`outer`) is probed against the set.
 ///
 /// `outer` bindings flow through to the output; `inner` bindings
 /// are dropped (semi-join semantics — only the existence of a
 /// match matters).
 ///
 /// `negated = true` ⇒ keep outer rows that have NO inner match
 /// (`NOT EXISTS`).
 HashSemiJoin {
 outer: Box<LogicalPlan>,
 inner: Box<LogicalPlan>,
 on: Vec<JoinKey>,
 negated: bool,
 /// Non-equi predicate evaluated on the joined row (outer ∪
 /// build's full row recovered from a side map). Optional —
 /// the rewriter ships `None` in v0.
 residual: Option<Expression>,
 },

 /// Placeholder for "the outer scope provides these bindings". Used
 /// as the leaf of a SemiApply/PatternList subplan when the inner
 /// pattern references variables bound in the outer query. Executor
 /// materialises a single row containing the outer bindings.
 Argument { bindings: Vec<String> },

 /// Existential / anti-existential semi-join. For each row produced
 /// by `input`, evaluate `subplan` parametrised by the row's
 /// bindings; keep the row iff the subplan yields at least one row
 /// (`negated=false`) or zero rows (`negated=true`).
 ///
 /// Lowering rule: `Exists(pattern)` predicates that appear at the
 /// AND-root of a WHERE expression are extracted to a chain of
 /// SemiApply operators; the residual predicates remain in `Filter`.
 SemiApply {
 input: Box<LogicalPlan>,
 subplan: Box<LogicalPlan>,
 negated: bool,
 },

 /// Materialise a pattern comprehension `[(a)-[]->(b) WHERE p | proj]`
 /// into a list-valued binding. For each row from `input`, execute
 /// `subplan` parametrised by the row, evaluate `projection` on each
 /// inner row, and bind the resulting list to `alias` on the outer row.
 PatternList {
 input: Box<LogicalPlan>,
 subplan: Box<LogicalPlan>,
 projection: Expression,
 alias: String,
 },

 // ─── Write-path operators (RFC-009) ────────────────────────
 /// `CREATE (a:Person {name: 'Ada'})-[r:KNOWS]->(b:Person {...})` —
 /// drives `input` (often `Empty` when standalone) and for each row
 /// materialises the listed elements in order. Newly created nodes
 /// and relationships are bound on the row under their aliases.
 Create {
 input: Box<LogicalPlan>,
 elements: Vec<CreateElement>,
 },

 /// `MERGE (pattern) [ON MATCH SET ...] [ON CREATE SET ...]` — try to
 /// MATCH the pattern; if at least one row matches, apply
 /// `on_match_sets`; otherwise CREATE the pattern and apply
 /// `on_create_sets`. The single pattern variant of v0 contains a
 /// single node or one node + one rel + one node chain.
 Merge {
 input: Box<LogicalPlan>,
 pattern: Vec<CreateElement>,
 on_match_sets: Vec<SetOp>,
 on_create_sets: Vec<SetOp>,
 },

 /// `SET a.prop = value` / `SET a = {...}` / `SET a += {...}` /
 /// `SET a:Label`. For each row from `input`, apply every `SetOp`.
 Set {
 input: Box<LogicalPlan>,
 items: Vec<SetOp>,
 },

 /// `REMOVE a.prop` / `REMOVE a:Label`. For each row, drop the named
 /// property or label from the bound node/rel.
 Remove {
 input: Box<LogicalPlan>,
 items: Vec<RemoveOp>,
 },

 /// `DELETE a, b` / `DETACH DELETE a`. For each row, tombstone the
 /// referenced node(s)/rel(s). With `detach=true`, nodes are first
 /// stripped of their incident edges across every edge type known
 /// to the manifest schema.
 Delete {
 input: Box<LogicalPlan>,
 targets: Vec<Expression>,
 detach: bool,
 },
}

impl LogicalPlan {
 /// Iterate over the direct child plans (0 to 2). Useful for
 /// rewriters and EXPLAIN walkers.
 pub fn children(&self) -> Vec<&LogicalPlan> {
 match self {
 LogicalPlan::NodeScan { .. } | LogicalPlan::Empty | LogicalPlan::Argument { .. } => {
 vec![]
 }
 LogicalPlan::NodeById { input, .. }
 | LogicalPlan::Expand { input, .. }
 | LogicalPlan::Filter { input, .. }
 | LogicalPlan::Project { input, .. }
 | LogicalPlan::Aggregate { input, .. }
 | LogicalPlan::TopN { input, .. }
 | LogicalPlan::Distinct { input }
 | LogicalPlan::Unwind { input, .. } => vec![input.as_ref()],
 LogicalPlan::Union { left, right, .. } | LogicalPlan::CrossProduct { left, right } => {
 vec![left.as_ref(), right.as_ref()]
 }
 LogicalPlan::HashJoin { build, probe, .. } => {
 vec![build.as_ref(), probe.as_ref()]
 }
 LogicalPlan::HashSemiJoin { outer, inner, .. } => {
 vec![outer.as_ref(), inner.as_ref()]
 }
 LogicalPlan::SemiApply { input, subplan, .. } => {
 vec![input.as_ref(), subplan.as_ref()]
 }
 LogicalPlan::PatternList { input, subplan, .. } => {
 vec![input.as_ref(), subplan.as_ref()]
 }
 LogicalPlan::Create { input, .. }
 | LogicalPlan::Merge { input, .. }
 | LogicalPlan::Set { input, .. }
 | LogicalPlan::Remove { input, .. }
 | LogicalPlan::Delete { input, .. } => vec![input.as_ref()],
 }
 }

 /// True if `self` (or any descendant) is a write-side operator.
 /// CLI dispatch and tooling use this to decide between
 /// `execute(plan, &Snapshot, _)` and
 /// `execute_write(plan, &mut WriterSession, _)`.
 pub fn contains_write(&self) -> bool {
 matches!(
 self,
 LogicalPlan::Create { .. }
 | LogicalPlan::Merge { .. }
 | LogicalPlan::Set { .. }
 | LogicalPlan::Remove { .. }
 | LogicalPlan::Delete { .. }
 ) || self.children().iter().any(|c| c.contains_write())
 }

 /// Short operator name used by EXPLAIN headers.
 pub fn operator_name(&self) -> &'static str {
 match self {
 LogicalPlan::NodeScan { .. } => "NodeScan",
 LogicalPlan::NodeById { .. } => "NodeById",
 LogicalPlan::Expand { optional, .. } => {
 if *optional {
 "OptionalExpand"
 } else {
 "Expand"
 }
 }
 LogicalPlan::Filter { .. } => "Filter",
 LogicalPlan::Project { distinct, .. } => {
 if *distinct {
 "ProjectDistinct"
 } else {
 "Project"
 }
 }
 LogicalPlan::Aggregate { .. } => "Aggregate",
 LogicalPlan::TopN { .. } => "TopN",
 LogicalPlan::Distinct { .. } => "Distinct",
 LogicalPlan::Union { all, .. } => {
 if *all {
 "UnionAll"
 } else {
 "Union"
 }
 }
 LogicalPlan::Unwind { .. } => "Unwind",
 LogicalPlan::Empty => "Empty",
 LogicalPlan::CrossProduct { .. } => "CrossProduct",
 LogicalPlan::HashJoin { .. } => "HashJoin",
 LogicalPlan::HashSemiJoin { negated, .. } => {
 if *negated {
 "AntiHashSemiJoin"
 } else {
 "HashSemiJoin"
 }
 }
 LogicalPlan::Argument { .. } => "Argument",
 LogicalPlan::SemiApply { negated, .. } => {
 if *negated {
 "AntiSemiApply"
 } else {
 "SemiApply"
 }
 }
 LogicalPlan::PatternList { .. } => "PatternList",
 LogicalPlan::Create { .. } => "Create",
 LogicalPlan::Merge { .. } => "Merge",
 LogicalPlan::Set { .. } => "Set",
 LogicalPlan::Remove { .. } => "Remove",
 LogicalPlan::Delete { detach, .. } => {
 if *detach {
 "DetachDelete"
 } else {
 "Delete"
 }
 }
 }
 }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionItem {
 pub expression: Expression,
 pub alias: String,
}

/// One equi-join pair for [`LogicalPlan::HashJoin`].
///
/// `build_side` is evaluated on each row of the build subtree to
/// compute the hash key; `probe_side` is evaluated on each probe row
/// to look up matches. Each expression references only aliases
/// produced by its respective subtree.
#[derive(Clone, Debug, PartialEq)]
pub struct JoinKey {
 pub build_side: Expression,
 pub probe_side: Expression,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrderKey {
 pub expression: Expression,
 pub direction: OrderDirection,
}

/// Aggregate function applied inside an `Aggregate` operator.
#[derive(Clone, Debug, PartialEq)]
pub enum AggregateExpr {
 /// `count(*)` if `arg = None`; `count(expr)` otherwise.
 Count {
 arg: Option<Expression>,
 distinct: bool,
 },
 Sum {
 arg: Expression,
 distinct: bool,
 },
 Avg {
 arg: Expression,
 distinct: bool,
 },
 Min {
 arg: Expression,
 },
 Max {
 arg: Expression,
 },
 Collect {
 arg: Expression,
 distinct: bool,
 },
}

/// One element of a `CREATE` / `MERGE` pattern.
///
/// `Node` introduces a fresh node bound under `alias`; `Rel` connects
/// two already-bound aliases via an edge of the given type and
/// direction. RFC-009 §"Operadores nuevos".
#[derive(Clone, Debug, PartialEq)]
pub enum CreateElement {
 Node {
 alias: String,
 label: String,
 properties: Vec<(String, Expression)>,
 },
 Rel {
 alias: Option<String>,
 edge_type: String,
 source_alias: String,
 target_alias: String,
 direction: RelationshipDirection,
 properties: Vec<(String, Expression)>,
 },
}

impl CreateElement {
 pub fn alias(&self) -> Option<&str> {
 match self {
 CreateElement::Node { alias, .. } => Some(alias.as_str()),
 CreateElement::Rel { alias, .. } => alias.as_deref(),
 }
 }
}

/// One `SET` operation. RFC-009 §"Operadores nuevos".
#[derive(Clone, Debug, PartialEq)]
pub enum SetOp {
 Property {
 target_alias: String,
 key: String,
 value: Expression,
 },
 Replace {
 target_alias: String,
 value: Expression,
 },
 Merge {
 target_alias: String,
 value: Expression,
 },
 Labels {
 target_alias: String,
 labels: Vec<String>,
 },
}

impl SetOp {
 pub fn target_alias(&self) -> &str {
 match self {
 SetOp::Property { target_alias, .. }
 | SetOp::Replace { target_alias, .. }
 | SetOp::Merge { target_alias, .. }
 | SetOp::Labels { target_alias, .. } => target_alias,
 }
 }
}

/// One `REMOVE` operation. RFC-009 §"Operadores nuevos".
#[derive(Clone, Debug, PartialEq)]
pub enum RemoveOp {
 Property {
 target_alias: String,
 key: String,
 },
 Labels {
 target_alias: String,
 labels: Vec<String>,
 },
}

impl AggregateExpr {
 /// Canonical name of the aggregate function (`"count"`, `"sum"`, ...).
 pub fn function_name(&self) -> &'static str {
 match self {
 AggregateExpr::Count { .. } => "count",
 AggregateExpr::Sum { .. } => "sum",
 AggregateExpr::Avg { .. } => "avg",
 AggregateExpr::Min { .. } => "min",
 AggregateExpr::Max { .. } => "max",
 AggregateExpr::Collect { .. } => "collect",
 }
 }

 pub fn distinct(&self) -> bool {
 match self {
 AggregateExpr::Count { distinct, .. }
 | AggregateExpr::Sum { distinct, .. }
 | AggregateExpr::Avg { distinct, .. }
 | AggregateExpr::Collect { distinct, .. } => *distinct,
 AggregateExpr::Min { .. } | AggregateExpr::Max { .. } => false,
 }
 }
}

#[cfg(test)]
mod tests {
 use super::*;
 use crate::parser::{ExpressionKind, Identifier, Literal, SourceSpan};

 fn lit(n: i64) -> Expression {
 Expression {
 kind: ExpressionKind::Literal(Literal::Integer(n)),
 span: SourceSpan::point(0),
 }
 }

 #[test]
 fn operator_names_match_explain_expectation() {
 let scan = LogicalPlan::NodeScan {
 label: Some("Person".into()),
 alias: "a".into(),
 predicates: vec![],
 projection: None,
 };
 assert_eq!(scan.operator_name(), "NodeScan");
 assert_eq!(scan.children().len(), 0);
 }

 #[test]
 fn filter_exposes_input_as_child() {
 let scan = LogicalPlan::NodeScan {
 label: Some("Person".into()),
 alias: "a".into(),
 predicates: vec![],
 projection: None,
 };
 let filter = LogicalPlan::Filter {
 input: Box::new(scan.clone()),
 predicate: lit(1),
 };
 assert_eq!(filter.operator_name(), "Filter");
 assert_eq!(filter.children().len(), 1);
 assert_eq!(filter.children()[0], &scan);
 }

 #[test]
 fn union_exposes_both_children() {
 let s = LogicalPlan::NodeScan {
 label: Some("L".into()),
 alias: "a".into(),
 predicates: vec![],
 projection: None,
 };
 let u = LogicalPlan::Union {
 left: Box::new(s.clone()),
 right: Box::new(s.clone()),
 all: true,
 };
 assert_eq!(u.operator_name(), "UnionAll");
 assert_eq!(u.children().len(), 2);
 }

 #[test]
 fn aggregate_expr_helpers() {
 let count_star = AggregateExpr::Count {
 arg: None,
 distinct: false,
 };
 let count_distinct = AggregateExpr::Count {
 arg: Some(lit(1)),
 distinct: true,
 };
 let min = AggregateExpr::Min { arg: lit(2) };
 assert_eq!(count_star.function_name(), "count");
 assert!(!count_star.distinct());
 assert!(count_distinct.distinct());
 assert_eq!(min.function_name(), "min");
 assert!(!min.distinct());
 }

 #[test]
 fn write_operator_names_and_children() {
 let dummy_input = LogicalPlan::Empty;
 let create = LogicalPlan::Create {
 input: Box::new(dummy_input.clone()),
 elements: vec![],
 };
 assert_eq!(create.operator_name(), "Create");
 assert_eq!(create.children().len(), 1);

 let delete = LogicalPlan::Delete {
 input: Box::new(dummy_input.clone()),
 targets: vec![],
 detach: false,
 };
 assert_eq!(delete.operator_name(), "Delete");
 let detach = LogicalPlan::Delete {
 input: Box::new(dummy_input.clone()),
 targets: vec![],
 detach: true,
 };
 assert_eq!(detach.operator_name(), "DetachDelete");

 let set = LogicalPlan::Set {
 input: Box::new(dummy_input.clone()),
 items: vec![],
 };
 assert_eq!(set.operator_name(), "Set");
 assert_eq!(set.children().len(), 1);
 }

 #[test]
 fn create_element_alias_helper() {
 let node = CreateElement::Node {
 alias: "a".into(),
 label: "Person".into(),
 properties: vec![],
 };
 assert_eq!(node.alias(), Some("a"));
 let rel = CreateElement::Rel {
 alias: None,
 edge_type: "KNOWS".into(),
 source_alias: "a".into(),
 target_alias: "b".into(),
 direction: crate::parser::RelationshipDirection::Right,
 properties: vec![],
 };
 assert_eq!(rel.alias(), None);
 }

 #[test]
 fn set_op_target_alias_helper() {
 let p = SetOp::Property {
 target_alias: "a".into(),
 key: "name".into(),
 value: lit(1),
 };
 assert_eq!(p.target_alias(), "a");
 let l = SetOp::Labels {
 target_alias: "b".into(),
 labels: vec!["X".into()],
 };
 assert_eq!(l.target_alias(), "b");
 }

 #[test]
 fn hash_join_exposes_build_and_probe_as_children() {
 let l = LogicalPlan::NodeScan {
 label: Some("Person".into()),
 alias: "a".into(),
 predicates: vec![],
 projection: None,
 };
 let r = LogicalPlan::NodeScan {
 label: Some("Person".into()),
 alias: "b".into(),
 predicates: vec![],
 projection: None,
 };
 let plan = LogicalPlan::HashJoin {
 build: Box::new(l.clone()),
 probe: Box::new(r.clone()),
 on: vec![JoinKey {
 build_side: lit(1),
 probe_side: lit(2),
 }],
 residual: None,
 };
 assert_eq!(plan.operator_name(), "HashJoin");
 assert_eq!(plan.children().len(), 2);
 assert_eq!(plan.children()[0], &l);
 assert_eq!(plan.children()[1], &r);
 assert!(!plan.contains_write());
 }

 #[test]
 fn projection_item_holds_alias() {
 let item = ProjectionItem {
 expression: Expression {
 kind: ExpressionKind::Variable(Identifier::new("a", SourceSpan::point(0))),
 span: SourceSpan::point(0),
 },
 alias: "n".into(),
 };
 assert_eq!(item.alias, "n");
 }
}
