//! Projection pushdown / column pruning (RFC-015 §3).
//!
//! Analyses the plan top-down to compute, per alias, the set of
//! property keys actually accessed. Walks back bottom-up to annotate
//! each `NodeScan` with `projection = Some(cols)` when the bindings
//! for that alias are referenced ONLY via `Property(alias, key)`. When
//! a bare `Variable(alias)` appears anywhere (e.g. `RETURN a`), the
//! alias falls back to `RequiredProps::All` and the NodeScan keeps
//! `projection = None` (storage reads every declared column).
//!
//! Idempotent: re-running on a plan that already has projections
//! produces the same result.

use std::collections::{BTreeMap, BTreeSet};

use crate::parser::ast::{
    CaseBranch, Expression, ExpressionKind, MapLiteral, NodePattern, PatternElement,
    PatternProperties, RelationshipPattern, UnaryOp,
};
use crate::plan::logical::{AggregateExpr, CreateElement, LogicalPlan, RemoveOp, SetOp};

/// Required-properties tracker for a single alias.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RequiredProps {
    /// Specific set of property keys accessed (`alias.key`).
    Set(BTreeSet<String>),
    /// At least one expression accessed the binding as a whole
    /// (`Variable(alias)`), so every column must survive.
    All,
}

impl RequiredProps {
    fn add_key(&mut self, key: &str) {
        if let Self::Set(s) = self {
            s.insert(key.to_string());
        }
    }
}

#[derive(Default, Clone, Debug)]
struct RequiredSet {
    by_alias: BTreeMap<String, RequiredProps>,
}

impl RequiredSet {
    fn record_property(&mut self, alias: &str, key: &str) {
        self.by_alias
            .entry(alias.to_string())
            .and_modify(|p| p.add_key(key))
            .or_insert_with(|| {
                let mut s = BTreeSet::new();
                s.insert(key.to_string());
                RequiredProps::Set(s)
            });
    }

    fn record_all(&mut self, alias: &str) {
        self.by_alias
            .entry(alias.to_string())
            .and_modify(|p| *p = RequiredProps::All)
            .or_insert(RequiredProps::All);
    }
}

/// Apply projection pushdown to `plan`. Returns a plan where every
/// `NodeScan` whose alias is referenced exclusively via PropertyAccess
/// carries `projection = Some(cols)`.
pub fn apply_projection_pushdown(plan: LogicalPlan) -> LogicalPlan {
    let required = compute_required(&plan);
    rewrite(plan, &required)
}

fn compute_required(plan: &LogicalPlan) -> RequiredSet {
    let mut req = RequiredSet::default();
    collect_from_plan(plan, &mut req);
    req
}

/// Walk the entire plan, harvesting every expression's property
/// references. Subplans of SemiApply / HashSemiJoin / PatternList are
/// visited too (their NodeScans get their own projections).
fn collect_from_plan(plan: &LogicalPlan, req: &mut RequiredSet) {
    match plan {
        LogicalPlan::NodeScan {
            predicates, alias, ..
        } => {
            // Predicates already pushed need their columns to survive.
            for p in predicates {
                req.record_property(alias, p.column());
            }
        }
        LogicalPlan::NodeById { input, id, .. } => {
            collect_from_plan(input, req);
            collect_from_expr(id, req);
        }
        LogicalPlan::NodeByPropertyValue {
            input,
            alias,
            property,
            value,
            ..
        } => {
            collect_from_plan(input, req);
            // The lookup property column must survive projection pushdown
            // — `lookup_node_by_property_via_scan` reads it back to verify
            // the exact-equality match.
            req.record_property(alias, property);
            collect_from_expr(value, req);
        }
        LogicalPlan::Expand { input, source, .. } => {
            // The source binding must expose its `id` for traversal —
            // the executor reads `row.get(source).id` to call
            // `out_edges` / `in_edges`. Record `id` defensively so the
            // outer NodeScan keeps `id` (which is the engine `node_id`
            // column — always included by the engine-column carve-out;
            // belt-and-suspenders).
            req.record_property(source, "id");
            collect_from_plan(input, req);
        }
        LogicalPlan::Filter { input, predicate } => {
            collect_from_expr(predicate, req);
            collect_from_plan(input, req);
        }
        LogicalPlan::Project {
            input,
            items,
            distinct: _,
            discard_input_bindings: _,
        } => {
            for it in items {
                collect_from_expr(&it.expression, req);
            }
            collect_from_plan(input, req);
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => {
            for (e, _) in group_by {
                collect_from_expr(e, req);
            }
            for (_, agg) in aggregations {
                collect_from_aggregate(agg, req);
            }
            collect_from_plan(input, req);
        }
        LogicalPlan::TopN { input, keys, .. } => {
            for k in keys {
                collect_from_expr(&k.expression, req);
            }
            collect_from_plan(input, req);
        }
        LogicalPlan::Distinct { input } => collect_from_plan(input, req),
        LogicalPlan::Union { left, right, .. } => {
            collect_from_plan(left, req);
            collect_from_plan(right, req);
        }
        LogicalPlan::Unwind { input, list, .. } => {
            collect_from_expr(list, req);
            collect_from_plan(input, req);
        }
        LogicalPlan::CrossProduct { left, right } => {
            collect_from_plan(left, req);
            collect_from_plan(right, req);
        }
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => {
            for jk in on {
                collect_from_expr(&jk.build_side, req);
                collect_from_expr(&jk.probe_side, req);
            }
            if let Some(r) = residual {
                collect_from_expr(r, req);
            }
            collect_from_plan(build, req);
            collect_from_plan(probe, req);
        }
        LogicalPlan::HashSemiJoin {
            outer,
            inner,
            on,
            residual,
            ..
        } => {
            for jk in on {
                collect_from_expr(&jk.build_side, req);
                collect_from_expr(&jk.probe_side, req);
            }
            if let Some(r) = residual {
                collect_from_expr(r, req);
            }
            collect_from_plan(outer, req);
            collect_from_plan(inner, req);
        }
        LogicalPlan::SemiApply { input, subplan, .. } => {
            collect_from_plan(input, req);
            collect_from_plan(subplan, req);
        }
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            ..
        } => {
            collect_from_expr(projection, req);
            collect_from_plan(input, req);
            collect_from_plan(subplan, req);
        }
        LogicalPlan::Argument { .. } | LogicalPlan::Empty | LogicalPlan::EdgeTypeCount { .. } => {}
        LogicalPlan::MultiwayJoin { vars, .. } => {
            // The executor reads `id` for every variable to drive the
            // leapfrog trie. Predicates on those variables are folded
            // into `NodeBinding.predicates` by the detection pass so
            // they already mark their own columns.
            for v in vars {
                req.record_property(&v.alias, "id");
                for p in &v.predicates {
                    req.record_property(&v.alias, p.column());
                }
            }
        }
        // Writes: every binding visible to a write may be materialised
        // by the writer. Mark them All defensively. v0 doesn't push
        // projection through writes; we just collect from subtree.
        LogicalPlan::Create { input, elements } => {
            collect_from_plan(input, req);
            for e in elements {
                collect_from_create_element(e, req);
            }
        }
        LogicalPlan::Merge {
            input,
            pattern,
            on_match_sets,
            on_create_sets,
        } => {
            collect_from_plan(input, req);
            for e in pattern {
                collect_from_create_element(e, req);
            }
            for s in on_match_sets.iter().chain(on_create_sets.iter()) {
                collect_from_set_op(s, req);
            }
        }
        LogicalPlan::Set { input, items } => {
            collect_from_plan(input, req);
            for s in items {
                collect_from_set_op(s, req);
            }
        }
        LogicalPlan::Remove { input, items } => {
            collect_from_plan(input, req);
            for r in items {
                collect_from_remove_op(r, req);
            }
        }
        LogicalPlan::Delete { input, targets, .. } => {
            collect_from_plan(input, req);
            for t in targets {
                collect_from_expr(t, req);
            }
        }
    }
}

fn collect_from_aggregate(agg: &AggregateExpr, req: &mut RequiredSet) {
    match agg {
        AggregateExpr::Count { arg: Some(e), .. }
        | AggregateExpr::Sum { arg: e, .. }
        | AggregateExpr::Avg { arg: e, .. }
        | AggregateExpr::Min { arg: e }
        | AggregateExpr::Max { arg: e }
        | AggregateExpr::Collect { arg: e, .. } => collect_from_expr(e, req),
        AggregateExpr::Count { arg: None, .. } => {}
    }
}

fn collect_from_create_element(e: &CreateElement, req: &mut RequiredSet) {
    match e {
        CreateElement::Node { properties, .. } => {
            for (_, v) in properties {
                collect_from_expr(v, req);
            }
        }
        CreateElement::Rel {
            properties,
            source_alias,
            target_alias,
            ..
        } => {
            req.record_all(source_alias);
            req.record_all(target_alias);
            for (_, v) in properties {
                collect_from_expr(v, req);
            }
        }
    }
}

fn collect_from_set_op(s: &SetOp, req: &mut RequiredSet) {
    match s {
        SetOp::Property {
            target_alias,
            key: _,
            value,
        } => {
            req.record_all(target_alias);
            collect_from_expr(value, req);
        }
        SetOp::Replace {
            target_alias,
            value,
        }
        | SetOp::Merge {
            target_alias,
            value,
        } => {
            req.record_all(target_alias);
            collect_from_expr(value, req);
        }
        SetOp::Labels { target_alias, .. } => req.record_all(target_alias),
    }
}

fn collect_from_remove_op(r: &RemoveOp, req: &mut RequiredSet) {
    match r {
        RemoveOp::Property { target_alias, .. } | RemoveOp::Labels { target_alias, .. } => {
            req.record_all(target_alias);
        }
    }
}

/// Walk an expression tree harvesting alias.key pairs into `req`. A
/// bare `Variable(alias)` (or a subquery binding) records `All` for
/// that alias.
fn collect_from_expr(expr: &Expression, req: &mut RequiredSet) {
    match &expr.kind {
        ExpressionKind::Variable(id) => req.record_all(&id.name),
        ExpressionKind::Property(pa) => {
            // Drill through arbitrary `target` shapes but only treat the
            // top-level `target.key` chain as a property reference when
            // the inner target is a bare Variable.
            match &pa.target.kind {
                ExpressionKind::Variable(id) => req.record_property(&id.name, &pa.key.name),
                _ => {
                    // Compound target (e.g. `node(a).key`); fall back to
                    // recording all aliases the target references.
                    collect_from_expr(&pa.target, req);
                }
            }
        }
        ExpressionKind::Literal(_) | ExpressionKind::Parameter(_) | ExpressionKind::Star => {}
        ExpressionKind::Index { target, index } => {
            collect_from_expr(target, req);
            collect_from_expr(index, req);
        }
        ExpressionKind::Range { target, from, to } => {
            collect_from_expr(target, req);
            if let Some(f) = from {
                collect_from_expr(f, req);
            }
            if let Some(t) = to {
                collect_from_expr(t, req);
            }
        }
        ExpressionKind::Unary {
            op: UnaryOp::Not,
            expr,
        }
        | ExpressionKind::Unary {
            op: UnaryOp::Neg,
            expr,
        } => collect_from_expr(expr, req),
        ExpressionKind::Binary { op: _, left, right } => {
            collect_from_expr(left, req);
            collect_from_expr(right, req);
        }
        ExpressionKind::In { item, list } => {
            collect_from_expr(item, req);
            collect_from_expr(list, req);
        }
        ExpressionKind::Between { target, low, high } => {
            collect_from_expr(target, req);
            collect_from_expr(low, req);
            collect_from_expr(high, req);
        }
        ExpressionKind::StringTest {
            target, pattern, ..
        } => {
            collect_from_expr(target, req);
            collect_from_expr(pattern, req);
        }
        ExpressionKind::IsNull { expr, .. } => collect_from_expr(expr, req),
        ExpressionKind::FunctionCall { args, .. } => {
            for a in args {
                collect_from_expr(a, req);
            }
        }
        ExpressionKind::Case {
            scrutinee,
            branches,
            otherwise,
        } => {
            if let Some(s) = scrutinee {
                collect_from_expr(s, req);
            }
            for CaseBranch { when, then, .. } in branches {
                collect_from_expr(when, req);
                collect_from_expr(then, req);
            }
            if let Some(o) = otherwise {
                collect_from_expr(o, req);
            }
        }
        ExpressionKind::Exists(pe) => collect_from_pattern_element(pe, req),
        ExpressionKind::List(xs) => {
            for x in xs {
                collect_from_expr(x, req);
            }
        }
        ExpressionKind::Map(m) => collect_from_map(m, req),
        ExpressionKind::ListComprehension(lc) => {
            collect_from_expr(&lc.list, req);
            if let Some(p) = &lc.predicate {
                collect_from_expr(p, req);
            }
            if let Some(p) = &lc.projection {
                collect_from_expr(p, req);
            }
        }
        ExpressionKind::PatternComprehension(pc) => {
            collect_from_pattern_element(&pc.pattern, req);
            if let Some(p) = &pc.predicate {
                collect_from_expr(p, req);
            }
            collect_from_expr(&pc.projection, req);
        }
    }
}

fn collect_from_pattern_element(pe: &PatternElement, req: &mut RequiredSet) {
    collect_from_node_pattern(&pe.head, req);
    for (rel, node) in &pe.chain {
        collect_from_rel_pattern(rel, req);
        collect_from_node_pattern(node, req);
    }
}

fn collect_from_node_pattern(np: &NodePattern, req: &mut RequiredSet) {
    if let Some(b) = &np.binding {
        // Pattern binding inside subquery references the alias as a whole.
        req.record_all(&b.name);
    }
    if let Some(p) = &np.properties {
        collect_from_pattern_properties(p, req);
    }
}

fn collect_from_rel_pattern(rp: &RelationshipPattern, req: &mut RequiredSet) {
    if let Some(b) = &rp.binding {
        req.record_all(&b.name);
    }
    if let Some(p) = &rp.properties {
        collect_from_pattern_properties(p, req);
    }
}

fn collect_from_pattern_properties(p: &PatternProperties, req: &mut RequiredSet) {
    match p {
        PatternProperties::Literal(m) => collect_from_map(m, req),
        PatternProperties::Parameter { .. } => {
            // `$param` references no variable bindings.
        }
    }
}

fn collect_from_map(m: &MapLiteral, req: &mut RequiredSet) {
    for (_, v) in &m.entries {
        collect_from_expr(v, req);
    }
}

/// Rewrite each NodeScan to carry `projection = Some(sorted_cols)`
/// when its alias is fully RequiredProps::Set in `req`. Otherwise
/// leave `projection` unchanged (lowering's None).
fn rewrite(plan: LogicalPlan, req: &RequiredSet) -> LogicalPlan {
    match plan {
        LogicalPlan::NodeScan {
            label,
            alias,
            predicates,
            projection,
        } => {
            let new_projection = match req.by_alias.get(&alias) {
                Some(RequiredProps::Set(cols)) => {
                    // Sort alphabetically for deterministic EXPLAIN.
                    let mut v: Vec<String> = cols.iter().cloned().collect();
                    v.sort();
                    Some(v)
                }
                Some(RequiredProps::All) => None,
                None => {
                    // Alias never referenced — keep `None` (read all) so
                    // dead-code aliases stay literally identical. The
                    // optimizer may have removed all uses upstream.
                    None
                }
            };
            // Prefer the freshly-computed projection but never widen an
            // existing one (rewriter is idempotent over its own output).
            let final_projection = match (projection, new_projection) {
                (Some(existing), Some(new)) => Some(intersect_sorted(&existing, &new)),
                (Some(existing), None) => Some(existing),
                (None, new) => new,
            };
            LogicalPlan::NodeScan {
                label,
                alias,
                predicates,
                projection: final_projection,
            }
        }
        LogicalPlan::NodeById {
            input,
            label,
            alias,
            id,
        } => LogicalPlan::NodeById {
            input: Box::new(rewrite(*input, req)),
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
            input: Box::new(rewrite(*input, req)),
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
            shortest,
            path_binding,
        } => LogicalPlan::Expand {
            input: Box::new(rewrite(*input, req)),
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
        },
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(rewrite(*input, req)),
            predicate,
        },
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(rewrite(*input, req)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => LogicalPlan::Aggregate {
            input: Box::new(rewrite(*input, req)),
            group_by,
            aggregations,
        },
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(rewrite(*input, req)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(rewrite(*input, req)),
        },
        LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
            left: Box::new(rewrite(*left, req)),
            right: Box::new(rewrite(*right, req)),
            all,
        },
        LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
            input: Box::new(rewrite(*input, req)),
            list,
            alias,
        },
        LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
            left: Box::new(rewrite(*left, req)),
            right: Box::new(rewrite(*right, req)),
        },
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => LogicalPlan::HashJoin {
            build: Box::new(rewrite(*build, req)),
            probe: Box::new(rewrite(*probe, req)),
            on,
            residual,
        },
        LogicalPlan::HashSemiJoin {
            outer,
            inner,
            on,
            negated,
            residual,
        } => LogicalPlan::HashSemiJoin {
            outer: Box::new(rewrite(*outer, req)),
            inner: Box::new(rewrite(*inner, req)),
            on,
            negated,
            residual,
        },
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => LogicalPlan::SemiApply {
            input: Box::new(rewrite(*input, req)),
            subplan: Box::new(rewrite(*subplan, req)),
            negated,
        },
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            alias,
        } => LogicalPlan::PatternList {
            input: Box::new(rewrite(*input, req)),
            subplan: Box::new(rewrite(*subplan, req)),
            projection,
            alias,
        },
        LogicalPlan::Argument { .. }
        | LogicalPlan::Empty
        | LogicalPlan::MultiwayJoin { .. }
        | LogicalPlan::EdgeTypeCount { .. } => plan,
        LogicalPlan::Create { input, elements } => LogicalPlan::Create {
            input: Box::new(rewrite(*input, req)),
            elements,
        },
        LogicalPlan::Merge {
            input,
            pattern,
            on_match_sets,
            on_create_sets,
        } => LogicalPlan::Merge {
            input: Box::new(rewrite(*input, req)),
            pattern,
            on_match_sets,
            on_create_sets,
        },
        LogicalPlan::Set { input, items } => LogicalPlan::Set {
            input: Box::new(rewrite(*input, req)),
            items,
        },
        LogicalPlan::Remove { input, items } => LogicalPlan::Remove {
            input: Box::new(rewrite(*input, req)),
            items,
        },
        LogicalPlan::Delete {
            input,
            targets,
            detach,
        } => LogicalPlan::Delete {
            input: Box::new(rewrite(*input, req)),
            targets,
            detach,
        },
    }
}

fn intersect_sorted(a: &[String], b: &[String]) -> Vec<String> {
    let set_a: BTreeSet<&String> = a.iter().collect();
    let mut out: Vec<String> = b.iter().filter(|x| set_a.contains(x)).cloned().collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{BinaryOp, Identifier, Literal, PropertyAccess};
    use crate::parser::SourceSpan;
    use crate::plan::logical::ProjectionItem;

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
    fn variable(alias: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Variable(ident(alias)),
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
    fn project_only(input: LogicalPlan, items: Vec<(Expression, &str)>) -> LogicalPlan {
        LogicalPlan::Project {
            input: Box::new(input),
            items: items
                .into_iter()
                .map(|(e, a)| ProjectionItem {
                    expression: e,
                    alias: a.into(),
                })
                .collect(),
            distinct: false,
            discard_input_bindings: true,
        }
    }

    #[test]
    fn project_property_yields_named_projection() {
        // RETURN a.firstName over NodeScan(Person, a)
        let plan = project_only(scan("Person", "a"), vec![(property("a", "firstName"), "x")]);
        let out = apply_projection_pushdown(plan);
        match out {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::NodeScan { projection, .. } => {
                    assert_eq!(projection, Some(vec!["firstName".into()]));
                }
                other => panic!("expected NodeScan under Project, got {:?}", other),
            },
            other => panic!("expected Project at root, got {:?}", other),
        }
    }

    #[test]
    fn bare_variable_yields_no_projection() {
        // RETURN a → bare alias, every column required.
        let plan = project_only(scan("Person", "a"), vec![(variable("a"), "all")]);
        let out = apply_projection_pushdown(plan);
        match out {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::NodeScan { projection, .. } => {
                    assert!(
                        projection.is_none(),
                        "bare variable should not produce a projection, got {projection:?}",
                    );
                }
                other => panic!("expected NodeScan, got {:?}", other),
            },
            other => panic!("expected Project at root, got {:?}", other),
        }
    }

    #[test]
    fn multiple_property_accesses_aggregate_to_set() {
        // RETURN a.x, a.y → projection [x, y] (alphabetical).
        let plan = project_only(
            scan("Person", "a"),
            vec![(property("a", "y"), "y"), (property("a", "x"), "x")],
        );
        let out = apply_projection_pushdown(plan);
        match out {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::NodeScan { projection, .. } => {
                    assert_eq!(projection, Some(vec!["x".into(), "y".into()]));
                }
                other => panic!("expected NodeScan, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn rewriter_is_idempotent() {
        let plan = project_only(scan("Person", "a"), vec![(property("a", "firstName"), "x")]);
        let once = apply_projection_pushdown(plan);
        let twice = apply_projection_pushdown(once.clone());
        assert_eq!(once, twice);
    }

    #[test]
    fn predicate_columns_get_included() {
        // Filter(a.age > 30) over NodeScan; projection should retain `age`.
        let pred = Expression {
            kind: ExpressionKind::Binary {
                op: BinaryOp::Gt,
                left: Box::new(property("a", "age")),
                right: Box::new(Expression {
                    kind: ExpressionKind::Literal(Literal::Integer(30)),
                    span: span(),
                }),
            },
            span: span(),
        };
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan("Person", "a")),
                predicate: pred,
            }),
            items: vec![ProjectionItem {
                expression: property("a", "firstName"),
                alias: "name".into(),
            }],
            distinct: false,
            discard_input_bindings: true,
        };
        let out = apply_projection_pushdown(plan);
        // Walk down to NodeScan and verify both `age` and `firstName`
        // ended up in the projection (alphabetical).
        fn find_node_scan(p: &LogicalPlan) -> Option<&Option<Vec<String>>> {
            match p {
                LogicalPlan::NodeScan { projection, .. } => Some(projection),
                _ => p.children().iter().find_map(|c| find_node_scan(c)),
            }
        }
        let proj = find_node_scan(&out).cloned().flatten();
        assert_eq!(proj, Some(vec!["age".into(), "firstName".into()]));
    }

    #[test]
    fn unrelated_alias_keeps_none() {
        // Reference `a.firstName` only — alias `b`'s NodeScan stays `None`.
        let plan = project_only(
            LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Message", "b")),
            },
            vec![(property("a", "firstName"), "x")],
        );
        let out = apply_projection_pushdown(plan);
        fn walk(plan: &LogicalPlan, out: &mut Vec<(String, Option<Vec<String>>)>) {
            if let LogicalPlan::NodeScan {
                alias, projection, ..
            } = plan
            {
                out.push((alias.clone(), projection.clone()));
            }
            for c in plan.children() {
                walk(c, out);
            }
        }
        let mut scans = Vec::new();
        walk(&out, &mut scans);
        let a = scans.iter().find(|(a, _)| a == "a").unwrap();
        let b = scans.iter().find(|(a, _)| a == "b").unwrap();
        assert_eq!(a.1, Some(vec!["firstName".into()]));
        // alias `b` not referenced → None (read all). Note: if `b` is
        // wholly unused the optimizer may drop the side later; v0 keeps
        // it.
        assert!(b.1.is_none());
    }
}
