//! Cardinality estimator for [`LogicalPlan`] (RFC-010 §3).
//!
//! `estimate(plan, &catalog)` walks the tree bottom-up and returns a
//! parallel [`Cardinality`] tree. Each node holds the estimated number
//! of rows it emits, the children's cardinalities, and the live
//! bindings it makes available downstream (alias → label / edge_type).
//!
//! The estimator is pure: no Snapshot access, no I/O. Inputs are the
//! plan tree and the pre-built [`StatsCatalog`].
//!
//! [`LogicalPlan`]: crate::plan::LogicalPlan
//! [`StatsCatalog`]: super::stats::StatsCatalog

use std::collections::BTreeMap;

use super::selectivity::{selectivity, BindingStats};
use super::stats::StatsCatalog;
use crate::parser::ast::{Expression, ExpressionKind, Literal, RelationshipDirection};
use crate::plan::logical::{
    AggregateExpr, CreateElement, LogicalPlan, OrderKey, ProjectionItem, RemoveOp, SetOp,
};

/// Result of [`estimate`]. Parallel to the input plan tree.
#[derive(Clone, Debug)]
pub struct Cardinality {
    /// Estimated rows this node emits. Always ≥ 0.
    pub rows: f64,
    /// Cardinality of the direct children, in the same order as
    /// `plan.children()`.
    pub children: Vec<Cardinality>,
    /// Bindings that the operator leaves visible downstream. Keyed by
    /// alias. Used by upstream Filter selectivity and EXPLAIN VERBOSE.
    pub bindings: BTreeMap<String, BindingMeta>,
    /// Short label for EXPLAIN VERBOSE — same as `plan.operator_name()`.
    pub operator: &'static str,
}

/// Meta about a single binding currently in scope.
#[derive(Clone, Debug, Default)]
pub struct BindingMeta {
    pub label: Option<String>,
    pub edge_type: Option<String>,
}

/// Default fan-out for an `Expand` whose edge_type has no stats.
const DEFAULT_BRANCH_FACTOR: f64 = 2.0;
/// Cap on `branch^length` so variable-length over dense graphs doesn't
/// explode to infinity (RFC-010 §3.4).
const MAX_VARLEN_BRANCH: f64 = 10_000.0;
/// Default list length when `UNWIND` is applied to a Parameter /
/// Variable / non-literal expression.
const DEFAULT_UNWIND_LEN: f64 = 5.0;

/// Estimate the cardinality of `plan` given `catalog`.
pub fn estimate(plan: &LogicalPlan, catalog: &StatsCatalog) -> Cardinality {
    estimate_inner(plan, catalog)
}

fn estimate_inner(plan: &LogicalPlan, catalog: &StatsCatalog) -> Cardinality {
    match plan {
        LogicalPlan::Empty => leaf(plan, 1.0, BTreeMap::new()),
        LogicalPlan::Argument { bindings } => {
            let mut b = BTreeMap::new();
            for name in bindings {
                b.insert(name.clone(), BindingMeta::default());
            }
            leaf(plan, 1.0, b)
        }
        LogicalPlan::NodeScan {
            label,
            alias,
            predicates,
            projection: _, // RFC-015: projection doesn't change row count.
        } => {
            let n = match label {
                Some(l) => catalog.label(l).map(|s| s.node_count as f64).unwrap_or(0.0),
                None => catalog.total_nodes() as f64,
            };
            let mut b = BTreeMap::new();
            b.insert(
                alias.clone(),
                BindingMeta {
                    label: label.clone(),
                    ..Default::default()
                },
            );
            // Apply selectivity of any pushed `ScanPredicate` (RFC-013).
            // Reuses the existing `selectivity` machinery by synthesising
            // an `Expression` per predicate; multiplicative under
            // independence assumption (RFC-010 §3.2).
            let rows = if predicates.is_empty() {
                n
            } else {
                let mut bs = BindingStats::empty();
                if let Some(l) = label {
                    if let Some(stats) = catalog.label(l) {
                        bs = bs.with(alias.clone(), stats);
                    }
                }
                let mut acc = n;
                for p in predicates {
                    let synth = super::selectivity::scan_predicate_to_expression(p, alias);
                    acc *= selectivity(&synth, &bs);
                }
                acc
            };
            leaf(plan, rows, b)
        }
        LogicalPlan::NodeById {
            input,
            label,
            alias,
            ..
        }
        | LogicalPlan::NodeByPropertyValue {
            input,
            label,
            alias,
            ..
        } => {
            let child = estimate_inner(input, catalog);
            // Point lookup: each input row triggers at most one hit.
            let rows = child.rows.min(child.rows.max(1.0));
            let mut bindings = child.bindings.clone();
            bindings.insert(
                alias.clone(),
                BindingMeta {
                    label: Some(label.clone()),
                    ..Default::default()
                },
            );
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Expand {
            input,
            edge_type,
            direction,
            target_alias,
            target_label,
            length,
            optional,
            rel_alias,
            ..
        } => {
            let child = estimate_inner(input, catalog);
            let branch = branch_factor(catalog, edge_type.as_deref(), *direction);
            let multiplier = match length {
                Some(l) => {
                    // Variable-length walks: cap branch^max to avoid
                    // pathological estimates on dense graphs.
                    let raw = branch.powi(l.max as i32);
                    raw.min(MAX_VARLEN_BRANCH)
                }
                None => branch,
            };
            let mut rows = child.rows * multiplier;
            if *optional && multiplier == 0.0 {
                rows = child.rows;
            }
            let mut bindings = child.bindings.clone();
            bindings.insert(
                target_alias.clone(),
                BindingMeta {
                    label: target_label.clone(),
                    ..Default::default()
                },
            );
            if let Some(rel) = rel_alias {
                bindings.insert(
                    rel.clone(),
                    BindingMeta {
                        edge_type: edge_type.clone(),
                        ..Default::default()
                    },
                );
            }
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Filter { input, predicate } => {
            let child = estimate_inner(input, catalog);
            let bs = make_binding_stats(catalog, &child.bindings);
            let s = selectivity(predicate, &bs);
            let rows = (child.rows * s).max(0.0);
            let bindings = child.bindings.clone();
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => {
            let child = estimate_inner(input, catalog);
            let rows = if *distinct {
                dedup_estimate(child.rows, items, catalog, &child.bindings)
            } else {
                child.rows
            };
            let mut bindings = if *discard_input_bindings {
                BTreeMap::new()
            } else {
                child.bindings.clone()
            };
            for item in items {
                bindings.insert(item.alias.clone(), binding_for_item(item, &child.bindings));
            }
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Distinct { input } => {
            let child = estimate_inner(input, catalog);
            // Without per-binding NDV we use the same heuristic as
            // dedup_estimate's "no info" branch.
            let rows = child.rows.powf(0.7).min(child.rows);
            let bindings = child.bindings.clone();
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => {
            let child = estimate_inner(input, catalog);
            let rows = if group_by.is_empty() {
                1.0
            } else {
                let ndv = group_by
                    .iter()
                    .map(|(e, _)| ndv_for_expr(e, catalog, &child.bindings))
                    .fold(1.0, |acc, x| acc * x);
                ndv.min(child.rows).max(1.0)
            };
            // Bindings after aggregate: group keys + agg aliases.
            let mut bindings = BTreeMap::new();
            for (e, alias) in group_by {
                let meta = if let ExpressionKind::Variable(id) = &e.kind {
                    child.bindings.get(&id.name).cloned().unwrap_or_default()
                } else {
                    BindingMeta::default()
                };
                bindings.insert(alias.clone(), meta);
            }
            for (alias, _) in aggregations {
                bindings.insert(alias.clone(), BindingMeta::default());
            }
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::TopN {
            input,
            keys: _,
            skip,
            limit,
        } => {
            let child = estimate_inner(input, catalog);
            let after_skip = (child.rows - *skip as f64).max(0.0);
            let rows = if *limit == u64::MAX {
                after_skip
            } else {
                after_skip.min(*limit as f64)
            };
            let bindings = child.bindings.clone();
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Unwind { input, list, alias } => {
            let child = estimate_inner(input, catalog);
            let avg_len = match &list.kind {
                ExpressionKind::List(xs) => xs.len() as f64,
                ExpressionKind::Literal(Literal::Null) => 0.0,
                _ => DEFAULT_UNWIND_LEN,
            };
            let rows = child.rows * avg_len;
            let mut bindings = child.bindings.clone();
            bindings.insert(alias.clone(), BindingMeta::default());
            Cardinality {
                rows,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Union { left, right, all } => {
            let l = estimate_inner(left, catalog);
            let r = estimate_inner(right, catalog);
            let combined = l.rows + r.rows;
            let rows = if *all {
                combined
            } else {
                let dedup_factor = 0.5;
                l.rows.max(r.rows) + dedup_factor * l.rows.min(r.rows)
            };
            // Union bindings: intersection by alias name (both sides
            // declare the same projection schema after lowering).
            let mut bindings = BTreeMap::new();
            for (k, v) in &l.bindings {
                if r.bindings.contains_key(k) {
                    bindings.insert(k.clone(), v.clone());
                }
            }
            Cardinality {
                rows,
                bindings,
                children: vec![l, r],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::CrossProduct { left, right } => {
            let l = estimate_inner(left, catalog);
            let r = estimate_inner(right, catalog);
            let rows = l.rows * r.rows;
            let mut bindings = l.bindings.clone();
            for (k, v) in &r.bindings {
                bindings.entry(k.clone()).or_insert_with(|| v.clone());
            }
            Cardinality {
                rows,
                bindings,
                children: vec![l, r],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => {
            let b = estimate_inner(build, catalog);
            let p = estimate_inner(probe, catalog);
            // Selinger '79: inner equi-join cardinality.
            // rows = (|build| * |probe|) / max(ndv(build_key), ndv(probe_key))
            // Multi-key: assume independence, divide by product.
            let mut divisor = 1.0_f64;
            for key in on {
                let build_ndv =
                    ndv_for_expr_opt(&key.build_side, catalog, &b.bindings).unwrap_or(1.0);
                let probe_ndv =
                    ndv_for_expr_opt(&key.probe_side, catalog, &p.bindings).unwrap_or(1.0);
                divisor *= build_ndv.max(probe_ndv).max(1.0);
            }
            let mut rows = (b.rows * p.rows / divisor).max(0.0);
            // Bindings of the joined tuple.
            let mut bindings = b.bindings.clone();
            for (k, v) in &p.bindings {
                bindings.entry(k.clone()).or_insert_with(|| v.clone());
            }
            if let Some(res) = residual {
                let bs = make_binding_stats(catalog, &bindings);
                rows *= selectivity(res, &bs);
            }
            Cardinality {
                rows,
                bindings,
                children: vec![b, p],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => {
            let outer = estimate_inner(input, catalog);
            let inner = estimate_inner(subplan, catalog);
            // Probability that at least one inner row matches the outer.
            let match_prob = if outer.rows > 0.0 {
                (inner.rows / outer.rows).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let kept = if *negated {
                1.0 - match_prob
            } else {
                match_prob
            };
            let rows = (outer.rows * kept).max(0.0);
            let bindings = outer.bindings.clone();
            Cardinality {
                rows,
                bindings,
                children: vec![outer, inner],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::HashSemiJoin {
            outer,
            inner,
            on: _,
            negated,
            residual: _,
        } => {
            // Mirrors the SemiApply estimate: each outer row matches with
            // probability proportional to inner.rows / outer.rows.
            // Multi-key on= refinement is RFC-014 Open Question.
            let o = estimate_inner(outer, catalog);
            let i = estimate_inner(inner, catalog);
            let match_prob = if o.rows > 0.0 {
                (i.rows / o.rows).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let kept = if *negated {
                1.0 - match_prob
            } else {
                match_prob
            };
            let rows = (o.rows * kept).max(0.0);
            // Semi-join semantics: only outer bindings survive.
            let bindings = o.bindings.clone();
            Cardinality {
                rows,
                bindings,
                children: vec![o, i],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::PatternList {
            input,
            subplan,
            alias,
            ..
        } => {
            let outer = estimate_inner(input, catalog);
            let inner = estimate_inner(subplan, catalog);
            // PatternList emits one row per outer row; the list itself
            // is a value column.
            let rows = outer.rows;
            let mut bindings = outer.bindings.clone();
            bindings.insert(alias.clone(), BindingMeta::default());
            Cardinality {
                rows,
                bindings,
                children: vec![outer, inner],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Create { input, elements } => {
            let child = estimate_inner(input, catalog);
            // Writes are sinks — the result of `execute_write` is a
            // WriteOutcome, not rows. We still surface `child.rows` as
            // the work-driver so EXPLAIN VERBOSE can show how many
            // input rows the CREATE iterates over.
            let mut bindings = child.bindings.clone();
            for el in elements {
                if let Some(a) = el.alias() {
                    let meta = match el {
                        CreateElement::Node { label, .. } => BindingMeta {
                            label: Some(label.clone()),
                            ..Default::default()
                        },
                        CreateElement::Rel { edge_type, .. } => BindingMeta {
                            edge_type: Some(edge_type.clone()),
                            ..Default::default()
                        },
                    };
                    bindings.insert(a.to_string(), meta);
                }
            }
            Cardinality {
                rows: 0.0,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Merge { input, pattern, .. } => {
            let child = estimate_inner(input, catalog);
            let mut bindings = child.bindings.clone();
            for el in pattern {
                if let Some(a) = el.alias() {
                    let meta = match el {
                        CreateElement::Node { label, .. } => BindingMeta {
                            label: Some(label.clone()),
                            ..Default::default()
                        },
                        CreateElement::Rel { edge_type, .. } => BindingMeta {
                            edge_type: Some(edge_type.clone()),
                            ..Default::default()
                        },
                    };
                    bindings.insert(a.to_string(), meta);
                }
            }
            Cardinality {
                rows: 0.0,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Set { input, items: _ } => {
            let child = estimate_inner(input, catalog);
            let bindings = child.bindings.clone();
            Cardinality {
                rows: 0.0,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Remove { input, items: _ } => {
            let child = estimate_inner(input, catalog);
            let bindings = child.bindings.clone();
            Cardinality {
                rows: 0.0,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
        LogicalPlan::Delete { input, .. } => {
            let child = estimate_inner(input, catalog);
            let bindings = child.bindings.clone();
            Cardinality {
                rows: 0.0,
                bindings,
                children: vec![child],
                operator: plan.operator_name(),
            }
        }
    }
}

fn leaf(plan: &LogicalPlan, rows: f64, bindings: BTreeMap<String, BindingMeta>) -> Cardinality {
    Cardinality {
        rows,
        bindings,
        children: vec![],
        operator: plan.operator_name(),
    }
}

fn branch_factor(
    catalog: &StatsCatalog,
    edge_type: Option<&str>,
    direction: RelationshipDirection,
) -> f64 {
    let stats = match edge_type {
        Some(et) => catalog.edge_type(et),
        None => None,
    };
    match (stats, edge_type) {
        (Some(s), _) => match direction {
            RelationshipDirection::Right => s.avg_out_degree.max(0.0),
            RelationshipDirection::Left => s.avg_in_degree.max(0.0),
            RelationshipDirection::Both => (s.avg_out_degree + s.avg_in_degree).max(0.0),
        },
        (None, Some(_)) => DEFAULT_BRANCH_FACTOR,
        (None, None) => {
            // Anonymous edge type — sum over every known edge type.
            let mut total = 0.0;
            for name in catalog.edge_type_names() {
                if let Some(s) = catalog.edge_type(name) {
                    total += match direction {
                        RelationshipDirection::Right => s.avg_out_degree,
                        RelationshipDirection::Left => s.avg_in_degree,
                        RelationshipDirection::Both => s.avg_out_degree + s.avg_in_degree,
                    };
                }
            }
            if total > 0.0 {
                total
            } else {
                DEFAULT_BRANCH_FACTOR
            }
        }
    }
}

/// Build a `BindingStats` view from the upstream bindings + the catalog.
/// Aliases without a `label` are dropped; the selectivity engine can
/// only reason about properties of labelled bindings.
fn make_binding_stats<'a>(
    catalog: &'a StatsCatalog,
    bindings: &BTreeMap<String, BindingMeta>,
) -> BindingStats<'a> {
    let mut bs = BindingStats::empty();
    for (alias, meta) in bindings {
        if let Some(label) = &meta.label {
            if let Some(ls) = catalog.label(label) {
                bs = bs.with(alias.clone(), ls);
            }
        }
    }
    bs
}

fn dedup_estimate(
    rows: f64,
    items: &[ProjectionItem],
    catalog: &StatsCatalog,
    bindings: &BTreeMap<String, BindingMeta>,
) -> f64 {
    if rows == 0.0 {
        return 0.0;
    }
    let mut prod = 1.0_f64;
    let mut any_ndv = false;
    for item in items {
        if let Some(n) = ndv_for_expr_opt(&item.expression, catalog, bindings) {
            prod *= n;
            any_ndv = true;
        }
    }
    if any_ndv {
        prod.min(rows).max(1.0)
    } else {
        rows.powf(0.7).max(1.0)
    }
}

/// NDV estimate for an expression, defaulting to the parent row count
/// when nothing else is known. Used by Aggregate and Distinct.
fn ndv_for_expr(
    expr: &Expression,
    catalog: &StatsCatalog,
    bindings: &BTreeMap<String, BindingMeta>,
) -> f64 {
    ndv_for_expr_opt(expr, catalog, bindings).unwrap_or(1.0)
}

fn ndv_for_expr_opt(
    expr: &Expression,
    catalog: &StatsCatalog,
    bindings: &BTreeMap<String, BindingMeta>,
) -> Option<f64> {
    match &expr.kind {
        ExpressionKind::Variable(id) => {
            // Variable bound to a node alias: NDV ≈ node_count of its
            // label. Variable bound to an edge: NDV ≈ edge_count.
            let meta = bindings.get(&id.name)?;
            if let Some(label) = &meta.label {
                return Some(catalog.label(label)?.node_count as f64);
            }
            if let Some(et) = &meta.edge_type {
                return Some(catalog.edge_type(et)?.edge_count as f64);
            }
            None
        }
        ExpressionKind::Property(p) => {
            let target = &p.target;
            let ExpressionKind::Variable(alias) = &target.kind else {
                return None;
            };
            let meta = bindings.get(&alias.name)?;
            let label = meta.label.as_ref()?;
            let lstats = catalog.label(label)?;
            let pstats = lstats.properties.get(&p.key.name)?;
            pstats.ndv.map(|n| n as f64)
        }
        _ => None,
    }
}

fn binding_for_item(item: &ProjectionItem, parent: &BTreeMap<String, BindingMeta>) -> BindingMeta {
    if let ExpressionKind::Variable(id) = &item.expression.kind {
        parent.get(&id.name).cloned().unwrap_or_default()
    } else {
        BindingMeta::default()
    }
}

// Keep `OrderKey`, `SetOp`, `RemoveOp`, `AggregateExpr` imported so the
// match arms above stay synchronised with logical.rs.
#[allow(dead_code)]
fn _keep_imports_live(_o: &OrderKey, _s: &SetOp, _r: &RemoveOp, _a: &AggregateExpr) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::stats::{LabelStats, PropStats};
    use crate::parser::ast::Identifier;
    use crate::parser::SourceSpan;
    use crate::plan::logical::AggregateExpr;
    use namidb_storage::sst::stats::StatScalar;

    fn span() -> SourceSpan {
        SourceSpan::point(0)
    }

    fn make_catalog() -> StatsCatalog {
        // Build a minimal in-memory catalog without going through
        // Manifest. We reach into private fields via a helper test-only
        // constructor — emulated here by re-implementing the merge
        // logic. Simpler: build the catalog manually below.
        let mut cat = StatsCatalog::empty();
        // Mutate via from_manifest equivalent: tests in `stats.rs`
        // already cover the manifest path; here we just need numeric
        // fixtures.
        let age_prop = PropStats {
            null_count: 0,
            non_null_count: 1000,
            min: Some(StatScalar::Int64(0)),
            max: Some(StatScalar::Int64(100)),
            ndv: Some(50),
            unique: false,
        };
        let person = LabelStats {
            name: "Person".into(),
            node_count: 1000,
            properties: {
                let mut m = BTreeMap::new();
                m.insert("age".into(), age_prop);
                m
            },
        };
        let knows = crate::cost::stats::EdgeTypeStats {
            name: "KNOWS".into(),
            edge_count: 5000,
            avg_out_degree: 5.0,
            max_out_degree: 50,
            avg_in_degree: 5.0,
            max_in_degree: 50,
            src_label: Some("Person".into()),
            dst_label: Some("Person".into()),
        };
        // The `StatsCatalog` fields are private; expose via a doctest
        // pattern: emit a fake manifest by-hand isn't worth here.
        // Workaround: construct catalog via from_manifest using the
        // helpers in `stats.rs::tests`, but those are private to that
        // module. Instead we add a doctest-only helper.
        cat.__test_insert_label(person);
        cat.__test_insert_edge_type(knows);
        cat
    }

    fn person_scan() -> LogicalPlan {
        LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "p".into(),
            predicates: vec![],
            projection: None,
        }
    }

    #[test]
    fn empty_emits_one_row() {
        let cat = StatsCatalog::empty();
        let c = estimate(&LogicalPlan::Empty, &cat);
        assert_eq!(c.rows, 1.0);
        assert!(c.bindings.is_empty());
    }

    #[test]
    fn argument_emits_one_row_with_bindings() {
        let cat = StatsCatalog::empty();
        let plan = LogicalPlan::Argument {
            bindings: vec!["a".into(), "b".into()],
        };
        let c = estimate(&plan, &cat);
        assert_eq!(c.rows, 1.0);
        assert_eq!(c.bindings.len(), 2);
    }

    #[test]
    fn node_scan_uses_label_node_count() {
        let cat = make_catalog();
        let c = estimate(&person_scan(), &cat);
        assert_eq!(c.rows, 1000.0);
        assert_eq!(c.bindings["p"].label.as_deref(), Some("Person"));
    }

    #[test]
    fn node_scan_unknown_label_yields_zero() {
        let cat = StatsCatalog::empty();
        let c = estimate(&person_scan(), &cat);
        assert_eq!(c.rows, 0.0);
    }

    #[test]
    fn filter_with_eq_applies_selectivity() {
        let cat = make_catalog();
        // age = 30 → 1/ndv = 1/50 = 0.02; rows = 20.
        let pred = Expression {
            kind: ExpressionKind::Binary {
                op: crate::parser::ast::BinaryOp::Eq,
                left: Box::new(Expression {
                    kind: ExpressionKind::Property(Box::new(crate::parser::ast::PropertyAccess {
                        target: Expression {
                            kind: ExpressionKind::Variable(Identifier::new("p", span())),
                            span: span(),
                        },
                        key: Identifier::new("age", span()),
                        span: span(),
                    })),
                    span: span(),
                }),
                right: Box::new(Expression {
                    kind: ExpressionKind::Literal(Literal::Integer(30)),
                    span: span(),
                }),
            },
            span: span(),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(person_scan()),
            predicate: pred,
        };
        let c = estimate(&plan, &cat);
        assert!((c.rows - 20.0).abs() < 1e-9);
    }

    #[test]
    fn expand_multiplies_by_avg_out_degree() {
        let cat = make_catalog();
        let expand = LogicalPlan::Expand {
            input: Box::new(person_scan()),
            source: "p".into(),
            edge_type: Some("KNOWS".into()),
            direction: RelationshipDirection::Right,
            rel_alias: Some("r".into()),
            target_alias: "q".into(),
            target_label: Some("Person".into()),
            length: None,
            optional: false,
            back_reference: false,
        };
        let c = estimate(&expand, &cat);
        // 1000 * 5.0 = 5000.
        assert!((c.rows - 5000.0).abs() < 1e-9);
        assert_eq!(c.bindings["q"].label.as_deref(), Some("Person"));
        assert_eq!(c.bindings["r"].edge_type.as_deref(), Some("KNOWS"));
    }

    #[test]
    fn variable_length_expand_caps_at_max() {
        let cat = make_catalog();
        // branch=5, length max=10 → 5^10 = 9.7M; capped at 10K.
        let expand = LogicalPlan::Expand {
            input: Box::new(person_scan()),
            source: "p".into(),
            edge_type: Some("KNOWS".into()),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "q".into(),
            target_label: None,
            length: Some(crate::parser::ast::RelationshipLength { min: 1, max: 10 }),
            optional: false,
            back_reference: false,
        };
        let c = estimate(&expand, &cat);
        assert!(c.rows <= 1000.0 * MAX_VARLEN_BRANCH);
    }

    #[test]
    fn topn_clamps_to_limit() {
        let cat = make_catalog();
        let plan = LogicalPlan::TopN {
            input: Box::new(person_scan()),
            keys: vec![],
            skip: 0,
            limit: 50,
        };
        let c = estimate(&plan, &cat);
        assert!((c.rows - 50.0).abs() < 1e-9);
    }

    #[test]
    fn topn_with_skip_subtracts_first() {
        let cat = make_catalog();
        let plan = LogicalPlan::TopN {
            input: Box::new(person_scan()),
            keys: vec![],
            skip: 100,
            limit: 50,
        };
        let c = estimate(&plan, &cat);
        assert!((c.rows - 50.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_without_group_by_returns_one() {
        let cat = make_catalog();
        let plan = LogicalPlan::Aggregate {
            input: Box::new(person_scan()),
            group_by: vec![],
            aggregations: vec![(
                "n".into(),
                AggregateExpr::Count {
                    arg: None,
                    distinct: false,
                },
            )],
        };
        let c = estimate(&plan, &cat);
        assert!((c.rows - 1.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_with_group_by_uses_ndv() {
        let cat = make_catalog();
        let group_expr = Expression {
            kind: ExpressionKind::Property(Box::new(crate::parser::ast::PropertyAccess {
                target: Expression {
                    kind: ExpressionKind::Variable(Identifier::new("p", span())),
                    span: span(),
                },
                key: Identifier::new("age", span()),
                span: span(),
            })),
            span: span(),
        };
        let plan = LogicalPlan::Aggregate {
            input: Box::new(person_scan()),
            group_by: vec![(group_expr, "age".into())],
            aggregations: vec![],
        };
        let c = estimate(&plan, &cat);
        // age.ndv = 50 → group cardinality 50.
        assert!((c.rows - 50.0).abs() < 1e-9);
    }

    #[test]
    fn cross_product_multiplies() {
        let cat = make_catalog();
        let plan = LogicalPlan::CrossProduct {
            left: Box::new(person_scan()),
            right: Box::new(person_scan()),
        };
        let c = estimate(&plan, &cat);
        assert!((c.rows - 1_000_000.0).abs() < 1e-3);
    }

    #[test]
    fn semi_apply_keeps_match_probability() {
        let cat = make_catalog();
        // input 1000, subplan 500 → match_prob = 0.5 → rows 500.
        let plan = LogicalPlan::SemiApply {
            input: Box::new(person_scan()),
            subplan: Box::new(LogicalPlan::TopN {
                input: Box::new(person_scan()),
                keys: vec![],
                skip: 0,
                limit: 500,
            }),
            negated: false,
        };
        let c = estimate(&plan, &cat);
        assert!((c.rows - 500.0).abs() < 1e-9);
    }

    #[test]
    fn anti_semi_apply_complements() {
        let cat = make_catalog();
        let plan = LogicalPlan::SemiApply {
            input: Box::new(person_scan()),
            subplan: Box::new(LogicalPlan::TopN {
                input: Box::new(person_scan()),
                keys: vec![],
                skip: 0,
                limit: 200,
            }),
            negated: true,
        };
        let c = estimate(&plan, &cat);
        // match_prob = 0.2 → anti = 0.8 → rows 800.
        assert!((c.rows - 800.0).abs() < 1e-9);
    }

    #[test]
    fn write_operator_emits_zero_rows() {
        let cat = make_catalog();
        let plan = LogicalPlan::Set {
            input: Box::new(person_scan()),
            items: vec![],
        };
        let c = estimate(&plan, &cat);
        assert_eq!(c.rows, 0.0);
        // But the child cardinality is preserved.
        assert!((c.children[0].rows - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn unwind_with_literal_list() {
        let cat = StatsCatalog::empty();
        let list = Expression {
            kind: ExpressionKind::List(vec![
                Expression {
                    kind: ExpressionKind::Literal(Literal::Integer(1)),
                    span: span(),
                },
                Expression {
                    kind: ExpressionKind::Literal(Literal::Integer(2)),
                    span: span(),
                },
                Expression {
                    kind: ExpressionKind::Literal(Literal::Integer(3)),
                    span: span(),
                },
            ]),
            span: span(),
        };
        let plan = LogicalPlan::Unwind {
            input: Box::new(LogicalPlan::Empty),
            list,
            alias: "x".into(),
        };
        let c = estimate(&plan, &cat);
        assert!((c.rows - 3.0).abs() < 1e-9);
    }
}
