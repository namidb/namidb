//! HashJoin orientation re-evaluation (RFC-016 §2 — v0 step 1).
//!
//! Walks the plan bottom-up. For every `HashJoin`, re-estimates both
//! branches against the catalog and swaps build/probe when the
//! current probe is smaller than the current build (Selinger's "build
//! over the smaller side" rule). The re-evaluation matters because
//! The optimizer picks build/probe BEFORE the rest of the pipeline runs —
//! predicate pushdown, projection pushdown and
//! decorrelation can change a branch's estimated rows after
//! the initial decision.
//!
//! v0 limited to single-node orientation swaps. Bushy DP enumeration
//! over chains of 3+ HashJoins lands (RFC-016 §"Out-of-
//! scope" — the gain on LDBC SNB queries we have today is dominated
//! by the local orientation choice).

use crate::cost::{estimate, StatsCatalog};
use crate::plan::logical::{JoinKey, LogicalPlan};

/// Reorder `HashJoin` orientations bottom-up. Each HashJoin recomputes
/// the estimate of both branches; if the probe is strictly smaller
/// than the build, the rewriter swaps them (and mirrors each
/// `JoinKey`).
///
/// Idempotent: a plan whose orientations already pick the smaller
/// side as build is left unchanged.
pub fn reorder_joins(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
 recurse(plan, catalog)
}

fn recurse(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
 match plan {
 // ─── HashJoin: recurse + re-orient ────────────────────────────
 LogicalPlan::HashJoin {
 build,
 probe,
 on,
 residual,
 } => {
 let new_build = recurse(*build, catalog);
 let new_probe = recurse(*probe, catalog);
 let b_rows = estimate(&new_build, catalog).rows;
 let p_rows = estimate(&new_probe, catalog).rows;
 if p_rows < b_rows {
 let swapped_on: Vec<JoinKey> = on
 .into_iter()
 .map(|k| JoinKey {
 build_side: k.probe_side,
 probe_side: k.build_side,
 })
 .collect();
 LogicalPlan::HashJoin {
 build: Box::new(new_probe),
 probe: Box::new(new_build),
 on: swapped_on,
 residual,
 }
 } else {
 LogicalPlan::HashJoin {
 build: Box::new(new_build),
 probe: Box::new(new_probe),
 on,
 residual,
 }
 }
 }

 // ─── Recurse on every other operator ──────────────────────────
 LogicalPlan::Empty | LogicalPlan::Argument { .. } | LogicalPlan::NodeScan { .. } => plan,
 LogicalPlan::NodeById {
 input,
 label,
 alias,
 id,
 } => LogicalPlan::NodeById {
 input: Box::new(recurse(*input, catalog)),
 label,
 alias,
 id,
 },
 LogicalPlan::NodeByPropertyValue {
 input,
 label,
 alias,
 property,
 value,
 } => LogicalPlan::NodeByPropertyValue {
 input: Box::new(recurse(*input, catalog)),
 label,
 alias,
 property,
 value,
 },
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
 } => LogicalPlan::Expand {
 input: Box::new(recurse(*input, catalog)),
 source,
 edge_type,
 direction,
 rel_alias,
 target_alias,
 target_label,
 length,
 optional,
 back_reference,
 },
 LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
 input: Box::new(recurse(*input, catalog)),
 predicate,
 },
 LogicalPlan::Project {
 input,
 items,
 distinct,
 discard_input_bindings,
 } => LogicalPlan::Project {
 input: Box::new(recurse(*input, catalog)),
 items,
 distinct,
 discard_input_bindings,
 },
 LogicalPlan::Aggregate {
 input,
 group_by,
 aggregations,
 } => LogicalPlan::Aggregate {
 input: Box::new(recurse(*input, catalog)),
 group_by,
 aggregations,
 },
 LogicalPlan::TopN {
 input,
 keys,
 skip,
 limit,
 } => LogicalPlan::TopN {
 input: Box::new(recurse(*input, catalog)),
 keys,
 skip,
 limit,
 },
 LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
 input: Box::new(recurse(*input, catalog)),
 },
 LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
 left: Box::new(recurse(*left, catalog)),
 right: Box::new(recurse(*right, catalog)),
 all,
 },
 LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
 input: Box::new(recurse(*input, catalog)),
 list,
 alias,
 },
 LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
 left: Box::new(recurse(*left, catalog)),
 right: Box::new(recurse(*right, catalog)),
 },
 LogicalPlan::HashSemiJoin {
 outer,
 inner,
 on,
 negated,
 residual,
 } => LogicalPlan::HashSemiJoin {
 outer: Box::new(recurse(*outer, catalog)),
 inner: Box::new(recurse(*inner, catalog)),
 on,
 negated,
 residual,
 },
 LogicalPlan::SemiApply {
 input,
 subplan,
 negated,
 } => LogicalPlan::SemiApply {
 input: Box::new(recurse(*input, catalog)),
 subplan: Box::new(recurse(*subplan, catalog)),
 negated,
 },
 LogicalPlan::PatternList {
 input,
 subplan,
 projection,
 alias,
 } => LogicalPlan::PatternList {
 input: Box::new(recurse(*input, catalog)),
 subplan: Box::new(recurse(*subplan, catalog)),
 projection,
 alias,
 },
 LogicalPlan::Create { input, elements } => LogicalPlan::Create {
 input: Box::new(recurse(*input, catalog)),
 elements,
 },
 LogicalPlan::Merge {
 input,
 pattern,
 on_match_sets,
 on_create_sets,
 } => LogicalPlan::Merge {
 input: Box::new(recurse(*input, catalog)),
 pattern,
 on_match_sets,
 on_create_sets,
 },
 LogicalPlan::Set { input, items } => LogicalPlan::Set {
 input: Box::new(recurse(*input, catalog)),
 items,
 },
 LogicalPlan::Remove { input, items } => LogicalPlan::Remove {
 input: Box::new(recurse(*input, catalog)),
 items,
 },
 LogicalPlan::Delete {
 input,
 targets,
 detach,
 } => LogicalPlan::Delete {
 input: Box::new(recurse(*input, catalog)),
 targets,
 detach,
 },
 }
}

#[cfg(test)]
mod tests {
 use super::*;
 use crate::cost::stats::StatsCatalog;
 use crate::parser::ast::{Expression, ExpressionKind, Identifier, PropertyAccess};
 use crate::parser::SourceSpan;

 fn span() -> SourceSpan {
 SourceSpan::point(0)
 }
 fn ident(name: &str) -> Identifier {
 Identifier {
 name: name.into(),
 span: span(),
 quoted: false,
 }
 }
 fn property(alias: &str, key: &str) -> Expression {
 Expression {
 kind: ExpressionKind::Property(Box::new(PropertyAccess {
 target: Expression {
 kind: ExpressionKind::Variable(ident(alias)),
 span: span(),
 },
 key: ident(key),
 span: span(),
 })),
 span: span(),
 }
 }
 fn scan(label: &str, alias: &str) -> LogicalPlan {
 LogicalPlan::NodeScan {
 label: Some(label.into()),
 alias: alias.into(),
 predicates: Vec::new(),
 projection: None,
 }
 }

 /// Build a catalog with one `Label` per `(name, node_count)`.
 fn catalog_with(labels: &[(&str, u64)]) -> StatsCatalog {
 StatsCatalog::with_label_counts(labels)
 }

 #[test]
 fn swap_when_probe_is_smaller_than_build() {
 // Build = Big (1000 rows), Probe = Small (10 rows).
 // After reorder: Small becomes build.
 let plan = LogicalPlan::HashJoin {
 build: Box::new(scan("Big", "b")),
 probe: Box::new(scan("Small", "s")),
 on: vec![JoinKey {
 build_side: property("b", "id"),
 probe_side: property("s", "id"),
 }],
 residual: None,
 };
 let cat = catalog_with(&[("Big", 1000), ("Small", 10)]);
 let out = reorder_joins(plan, &cat);
 match out {
 LogicalPlan::HashJoin {
 build, probe, on, ..
 } => {
 assert!(matches!(
 *build,
 LogicalPlan::NodeScan { ref label, .. } if label.as_deref() == Some("Small")
 ));
 assert!(matches!(
 *probe,
 LogicalPlan::NodeScan { ref label, .. } if label.as_deref() == Some("Big")
 ));
 // Keys must mirror — what was probe is now build.
 assert_eq!(on.len(), 1);
 // The new build_side should be the OLD probe_side, i.e.
 // `s.id` (Variable name `s`).
 if let ExpressionKind::Property(pa) = &on[0].build_side.kind {
 if let ExpressionKind::Variable(id) = &pa.target.kind {
 assert_eq!(id.name, "s");
 } else {
 panic!("expected Variable target on build_side");
 }
 }
 }
 other => panic!("expected HashJoin, got {:?}", other),
 }
 }

 #[test]
 fn no_swap_when_build_is_already_smaller() {
 let plan = LogicalPlan::HashJoin {
 build: Box::new(scan("Small", "s")),
 probe: Box::new(scan("Big", "b")),
 on: vec![JoinKey {
 build_side: property("s", "id"),
 probe_side: property("b", "id"),
 }],
 residual: None,
 };
 let cat = catalog_with(&[("Small", 10), ("Big", 1000)]);
 let out = reorder_joins(plan, &cat);
 match out {
 LogicalPlan::HashJoin { build, probe, .. } => {
 assert!(matches!(
 *build,
 LogicalPlan::NodeScan { ref label, .. } if label.as_deref() == Some("Small")
 ));
 assert!(matches!(
 *probe,
 LogicalPlan::NodeScan { ref label, .. } if label.as_deref() == Some("Big")
 ));
 }
 _ => panic!(),
 }
 }

 #[test]
 fn no_swap_when_sides_equal() {
 let plan = LogicalPlan::HashJoin {
 build: Box::new(scan("Equal", "a")),
 probe: Box::new(scan("Equal", "b")),
 on: vec![JoinKey {
 build_side: property("a", "id"),
 probe_side: property("b", "id"),
 }],
 residual: None,
 };
 let cat = catalog_with(&[("Equal", 100)]);
 let out = reorder_joins(plan.clone(), &cat);
 assert_eq!(out, plan);
 }

 #[test]
 fn reorder_is_idempotent() {
 let plan = LogicalPlan::HashJoin {
 build: Box::new(scan("Big", "b")),
 probe: Box::new(scan("Small", "s")),
 on: vec![JoinKey {
 build_side: property("b", "id"),
 probe_side: property("s", "id"),
 }],
 residual: None,
 };
 let cat = catalog_with(&[("Big", 1000), ("Small", 10)]);
 let once = reorder_joins(plan, &cat);
 let twice = reorder_joins(once.clone(), &cat);
 assert_eq!(once, twice);
 }

 #[test]
 fn recurses_into_nested_hash_joins() {
 // (Big × Small) ⋈ Tiny where each inner HashJoin has the wrong
 // orientation.
 let inner = LogicalPlan::HashJoin {
 build: Box::new(scan("Big", "b")),
 probe: Box::new(scan("Small", "s")),
 on: vec![JoinKey {
 build_side: property("b", "id"),
 probe_side: property("s", "id"),
 }],
 residual: None,
 };
 let outer = LogicalPlan::HashJoin {
 build: Box::new(inner),
 probe: Box::new(scan("Tiny", "t")),
 on: vec![JoinKey {
 build_side: property("b", "tid"),
 probe_side: property("t", "id"),
 }],
 residual: None,
 };
 let cat = catalog_with(&[("Big", 1000), ("Small", 10), ("Tiny", 1)]);
 let out = reorder_joins(outer, &cat);
 // The inner Big⋈Small should have swapped to Small⋈Big after
 // reorder (Small is smaller).
 match out {
 LogicalPlan::HashJoin { build, probe, .. } => {
 // Outer: Tiny is smallest → becomes build.
 assert!(matches!(
 *build,
 LogicalPlan::NodeScan { ref label, .. } if label.as_deref() == Some("Tiny")
 ));
 // The other side is the (now-reoriented) inner.
 match *probe {
 LogicalPlan::HashJoin {
 build: inner_build, ..
 } => assert!(matches!(
 *inner_build,
 LogicalPlan::NodeScan { ref label, .. } if label.as_deref() == Some("Small")
 )),
 _ => panic!("expected nested HashJoin"),
 }
 }
 _ => panic!(),
 }
 }

 #[test]
 fn pass_through_non_hash_join_subtrees() {
 // Filter wrapping a NodeScan — no HashJoin, pass through.
 let plan = LogicalPlan::Filter {
 input: Box::new(scan("Person", "a")),
 predicate: Expression {
 kind: ExpressionKind::Literal(crate::parser::ast::Literal::Boolean(true)),
 span: span(),
 },
 };
 let cat = StatsCatalog::empty();
 let out = reorder_joins(plan.clone(), &cat);
 assert_eq!(out, plan);
 }
}
