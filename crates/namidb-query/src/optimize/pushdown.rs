//! Predicate pushdown (RFC-011 §2).
//!
//! Walks the plan top-down with an accumulator of pending predicates.
//! Each operator decides which conjuncts it can absorb (push further
//! down its input) and which must stay above it (because they reference
//! aliases the operator introduces).
//!
//! Distinct/TopN/write operators are barriers — pending predicates are
//! materialised as a `Filter` above them rather than crossed.

use std::collections::{BTreeMap, BTreeSet};

use super::parquet_pushdown::classify_pending_for_scan;
use super::{and_chain, apply_filters, expression_aliases, produced_aliases, split_and_terms};
use crate::parser::ast::{Expression, ExpressionKind};
use crate::plan::logical::{CreateElement, LogicalPlan};

/// Push every `Filter` in `plan` as close to the leaves as possible.
/// See RFC-011 §2 for the per-operator rules.
pub fn predicate_pushdown(plan: LogicalPlan) -> LogicalPlan {
    pushdown_at(plan, Vec::new())
}

fn pushdown_at(plan: LogicalPlan, pending: Vec<Expression>) -> LogicalPlan {
    match plan {
        // ─── leaves: materialise pending above and stop ───────────────
        LogicalPlan::Empty
        | LogicalPlan::Argument { .. }
        | LogicalPlan::MultiwayJoin { .. }
        | LogicalPlan::EdgeTypeCount { .. }
        | LogicalPlan::VectorSearch { .. }
        | LogicalPlan::CallProcedure { .. } => {
            // The detection pass folds predicates over participating
            // variables into `NodeBinding.predicates` before emitting,
            // so by the time this pass reaches a MultiwayJoin there
            // should be nothing left to push. Anything that does
            // arrive (e.g. a Filter that references a variable bound
            // by the multiway join) stays above as a regular Filter.
            apply_filters(plan, pending)
        }

        // ─── NodeScan: try Parquet-pushdown each pending conjunct ─────
        LogicalPlan::NodeScan {
            label,
            alias,
            predicates: existing,
            projection,
        } => {
            // RFC-013 §5: classify pending against this scan's alias.
            // Pushable conjuncts merge into `predicates`; residual stays
            // as Filter above.
            let (pushed, residual) = classify_pending_for_scan(pending, &alias);
            let mut merged = existing;
            merged.extend(pushed);
            apply_filters(
                LogicalPlan::NodeScan {
                    label,
                    alias,
                    predicates: merged,
                    projection,
                },
                residual,
            )
        }

        // ─── Filter: AND-split into pending, recurse into input ───────
        LogicalPlan::Filter { input, predicate } => {
            let mut acc = pending;
            for term in split_and_terms(&predicate) {
                acc.push(term);
            }
            pushdown_at(*input, acc)
        }

        // ─── NodeById { input, alias } — introduces `alias` ───────────
        LogicalPlan::NodeById {
            input,
            label,
            alias,
            id,
        } => {
            let mut introduced = BTreeSet::new();
            introduced.insert(alias.clone());
            let (pushable, stay) = partition_by_alias_disjoint(pending, &introduced);
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::NodeById {
                    input: Box::new(new_input),
                    label,
                    alias,
                    id,
                },
                stay,
            )
        }

        // Same alias-introducing shape as NodeById.
        LogicalPlan::NodeByPropertyValue {
            input,
            label,
            alias,
            property,
            value,
            multi,
        } => {
            let mut introduced = BTreeSet::new();
            introduced.insert(alias.clone());
            let (pushable, stay) = partition_by_alias_disjoint(pending, &introduced);
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::NodeByPropertyValue {
                    input: Box::new(new_input),
                    label,
                    alias,
                    property,
                    value,
                    multi,
                },
                stay,
            )
        }

        // Same alias-introducing shape again for the composite lookup.
        LogicalPlan::NodeByPropertyTuple {
            input,
            label,
            alias,
            properties,
            values,
        } => {
            let mut introduced = BTreeSet::new();
            introduced.insert(alias.clone());
            let (pushable, stay) = partition_by_alias_disjoint(pending, &introduced);
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::NodeByPropertyTuple {
                    input: Box::new(new_input),
                    label,
                    alias,
                    properties,
                    values,
                },
                stay,
            )
        }

        // ─── Expand: introduces target_alias (+ rel_alias?) ────────────
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
            let mut introduced = BTreeSet::new();
            introduced.insert(target_alias.clone());
            if let Some(r) = &rel_alias {
                introduced.insert(r.clone());
            }
            // The path binding (`p` in `MATCH p = (a)-[*]->(b)`) is materialised
            // here too, so a `WHERE` term over `nodes(p)` must stay ABOVE the
            // Expand, not sink below the operator that produces `p`.
            if let Some(p) = &path_binding {
                introduced.insert(p.clone());
            }
            let (pushable, stay) = partition_by_alias_disjoint(pending, &introduced);
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::Expand {
                    input: Box::new(new_input),
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
                },
                stay,
            )
        }

        // ─── CrossProduct: split pending by side ───────────────────────
        LogicalPlan::CrossProduct { left, right } => {
            let left_aliases = produced_aliases(&left);
            let right_aliases = produced_aliases(&right);
            let mut to_left = Vec::new();
            let mut to_right = Vec::new();
            let mut keep_top = Vec::new();
            for term in pending {
                let refs = expression_aliases(&term);
                let hits_left = !refs.is_disjoint(&left_aliases);
                let hits_right = !refs.is_disjoint(&right_aliases);
                match (hits_left, hits_right) {
                    (true, false) => to_left.push(term),
                    (false, true) => to_right.push(term),
                    _ => keep_top.push(term),
                }
            }
            let new_left = pushdown_at(*left, to_left);
            let new_right = pushdown_at(*right, to_right);
            apply_filters(
                LogicalPlan::CrossProduct {
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                },
                keep_top,
            )
        }

        // ─── HashJoin: same split logic as CrossProduct ───────────────
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => {
            let build_aliases = produced_aliases(&build);
            let probe_aliases = produced_aliases(&probe);
            let mut to_build = Vec::new();
            let mut to_probe = Vec::new();
            let mut keep_top = Vec::new();
            for term in pending {
                let refs = expression_aliases(&term);
                let hits_build = !refs.is_disjoint(&build_aliases);
                let hits_probe = !refs.is_disjoint(&probe_aliases);
                match (hits_build, hits_probe) {
                    (true, false) => to_build.push(term),
                    (false, true) => to_probe.push(term),
                    _ => keep_top.push(term),
                }
            }
            let new_build = pushdown_at(*build, to_build);
            let new_probe = pushdown_at(*probe, to_probe);
            apply_filters(
                LogicalPlan::HashJoin {
                    build: Box::new(new_build),
                    probe: Box::new(new_probe),
                    on,
                    residual,
                },
                keep_top,
            )
        }

        // ─── HashSemiJoin: only outer bindings flow out ───────────────
        LogicalPlan::HashSemiJoin {
            outer,
            inner,
            on,
            negated,
            residual,
        } => {
            let outer_aliases = produced_aliases(&outer);
            let mut to_outer = Vec::new();
            let mut keep_top = Vec::new();
            for term in pending {
                let refs = expression_aliases(&term);
                if refs.is_subset(&outer_aliases) {
                    to_outer.push(term);
                } else {
                    keep_top.push(term);
                }
            }
            let new_outer = pushdown_at(*outer, to_outer);
            apply_filters(
                LogicalPlan::HashSemiJoin {
                    outer: Box::new(new_outer),
                    inner,
                    on,
                    negated,
                    residual,
                },
                keep_top,
            )
        }

        // ─── Project: pushable iff alias-preserving (no rename) ────────
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => {
            let preserved = identity_projection_aliases(&items);
            let (mut pushable, stay) = partition_by_alias_subset(pending, &preserved);
            // A WITH that RENAMES a property (`WITH v, p.cod_item AS c
            // WHERE c = '…'`) used to strand the predicate above this
            // Project, so the index anchor on `p` was never visible to
            // unique_lookup / anchor_inversion and the plan degenerated to
            // scanning the whole source label (fifth field report: a 160k
            // scan where the in-pattern spelling of the SAME query anchors
            // and finishes instantly). Substitute the alias back to its
            // bound expression and push the rewritten conjunct down; the
            // fixpoint loop then re-plans it as an anchored lookup.
            let (substituted, stay) = partition_substitutable(stay, &items, distinct);
            pushable.extend(substituted);
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::Project {
                    input: Box::new(new_input),
                    items,
                    distinct,
                    discard_input_bindings,
                },
                stay,
            )
        }

        // ─── Aggregate: pushable iff refs only identity group-by keys ──
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => {
            let mut preserved = BTreeSet::new();
            for (e, alias) in &group_by {
                if let ExpressionKind::Variable(id) = &e.kind {
                    if id.name == *alias {
                        preserved.insert(id.name.clone());
                    }
                }
            }
            let agg_aliases: BTreeSet<String> =
                aggregations.iter().map(|(a, _)| a.clone()).collect();
            let mut pushable = Vec::new();
            let mut stay = Vec::new();
            for term in pending {
                let refs = expression_aliases(&term);
                if refs.is_subset(&preserved) && refs.is_disjoint(&agg_aliases) {
                    pushable.push(term);
                } else {
                    stay.push(term);
                }
            }
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::Aggregate {
                    input: Box::new(new_input),
                    group_by,
                    aggregations,
                },
                stay,
            )
        }

        // ─── Union: push to both sides when aliases live in both ──────
        LogicalPlan::Union { left, right, all } => {
            let left_aliases = produced_aliases(&left);
            let right_aliases = produced_aliases(&right);
            let mut pushable = Vec::new();
            let mut stay = Vec::new();
            for term in pending {
                let refs = expression_aliases(&term);
                if refs.is_subset(&left_aliases) && refs.is_subset(&right_aliases) {
                    pushable.push(term);
                } else {
                    stay.push(term);
                }
            }
            let new_left = pushdown_at(*left, pushable.clone());
            let new_right = pushdown_at(*right, pushable);
            apply_filters(
                LogicalPlan::Union {
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                    all,
                },
                stay,
            )
        }

        // ─── Unwind: predicate over the unwind alias stays above ──────
        LogicalPlan::Unwind { input, list, alias } => {
            let mut introduced = BTreeSet::new();
            introduced.insert(alias.clone());
            let (pushable, stay) = partition_by_alias_disjoint(pending, &introduced);
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::Unwind {
                    input: Box::new(new_input),
                    list,
                    alias,
                },
                stay,
            )
        }

        // ─── SemiApply: pending flows to outer; subplan untouched ─────
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => {
            // SemiApply does not introduce any visible alias above. All
            // pending predicates flow into `input`. The subplan keeps
            // its own scope and is not visited for pushdown.
            let new_input = pushdown_at(*input, pending);
            LogicalPlan::SemiApply {
                input: Box::new(new_input),
                subplan,
                negated,
            }
        }

        // Apply exposes the subquery's RETURN bindings above it, so a pending
        // predicate may reference them — keep all pending filters above the
        // Apply and recurse into the input with none. The subplan is a
        // separate (correlated) scope and is not visited.
        LogicalPlan::Apply { input, subplan } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Apply {
                    input: Box::new(new_input),
                    subplan,
                },
                pending,
            )
        }

        // ─── PatternList: introduces the list alias ───────────────────
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            alias,
        } => {
            let mut introduced = BTreeSet::new();
            introduced.insert(alias.clone());
            let (pushable, stay) = partition_by_alias_disjoint(pending, &introduced);
            let new_input = pushdown_at(*input, pushable);
            apply_filters(
                LogicalPlan::PatternList {
                    input: Box::new(new_input),
                    subplan,
                    projection,
                    alias,
                },
                stay,
            )
        }

        // ─── TopN / Distinct: barriers (cardinality-changing) ─────────
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::TopN {
                    input: Box::new(new_input),
                    keys,
                    skip,
                    limit,
                },
                pending,
            )
        }
        LogicalPlan::Distinct { input } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Distinct {
                    input: Box::new(new_input),
                },
                pending,
            )
        }
        LogicalPlan::DiscardResult { input } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::DiscardResult {
                    input: Box::new(new_input),
                },
                pending,
            )
        }

        // ─── write ops: barriers — pending stays above, recurse input ─
        LogicalPlan::Create { input, elements } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Create {
                    input: Box::new(new_input),
                    elements,
                },
                pending,
            )
        }
        LogicalPlan::Foreach {
            input,
            variable,
            list,
            body,
        } => {
            // Push pending filters down through the input; keep the write body
            // untouched. Filters stay above FOREACH (it is a pass-through).
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Foreach {
                    input: Box::new(new_input),
                    variable,
                    list,
                    body,
                },
                pending,
            )
        }
        LogicalPlan::Merge {
            input,
            pattern,
            on_match_sets,
            on_create_sets,
        } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Merge {
                    input: Box::new(new_input),
                    pattern,
                    on_match_sets,
                    on_create_sets,
                },
                pending,
            )
        }
        LogicalPlan::Set { input, items } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Set {
                    input: Box::new(new_input),
                    items,
                },
                pending,
            )
        }
        LogicalPlan::Remove { input, items } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Remove {
                    input: Box::new(new_input),
                    items,
                },
                pending,
            )
        }
        LogicalPlan::Delete {
            input,
            targets,
            detach,
        } => {
            let new_input = pushdown_at(*input, Vec::new());
            apply_filters(
                LogicalPlan::Delete {
                    input: Box::new(new_input),
                    targets,
                    detach,
                },
                pending,
            )
        }
    }
}

fn partition_by_alias_disjoint(
    pending: Vec<Expression>,
    introduced: &BTreeSet<String>,
) -> (Vec<Expression>, Vec<Expression>) {
    let mut pushable = Vec::new();
    let mut stay = Vec::new();
    for term in pending {
        if expression_aliases(&term).is_disjoint(introduced) {
            pushable.push(term);
        } else {
            stay.push(term);
        }
    }
    (pushable, stay)
}

fn partition_by_alias_subset(
    pending: Vec<Expression>,
    preserved: &BTreeSet<String>,
) -> (Vec<Expression>, Vec<Expression>) {
    let mut pushable = Vec::new();
    let mut stay = Vec::new();
    for term in pending {
        let refs = expression_aliases(&term);
        if refs.is_subset(preserved) {
            pushable.push(term);
        } else {
            stay.push(term);
        }
    }
    (pushable, stay)
}

/// Split predicates that can be pushed BELOW an aliasing projection by
/// substituting each referenced alias with the expression the projection
/// binds it to. Returns `(rewritten_pushable, stay)`.
///
/// Only pure property/variable bindings substitute: a computed alias could
/// hide a function call whose evaluation count would change, and an
/// aggregate cannot be evaluated below its own projection. A conjunct is
/// pushed only when EVERY alias it references resolves, and no substituted
/// expression mentions a name this projection rebinds (`WITH p.x AS p`
/// means `p` below is a different value).
fn partition_substitutable(
    pending: Vec<Expression>,
    items: &[crate::plan::logical::ProjectionItem],
    distinct: bool,
) -> (Vec<Expression>, Vec<Expression>) {
    // A DISTINCT projection dedups before the filter; pushing a predicate
    // below it would change how often a non-deterministic term is
    // evaluated, so leave those alone.
    if distinct {
        return (Vec::new(), pending);
    }
    let mut substitutions: BTreeMap<String, Expression> = BTreeMap::new();
    let mut rebound: BTreeSet<String> = BTreeSet::new();
    for item in items {
        let identity = matches!(
            &item.expression.kind,
            ExpressionKind::Variable(id) if id.name == item.alias
        );
        if !identity {
            // This name means something different above than below.
            rebound.insert(item.alias.clone());
        }
        if is_substitutable_projection(&item.expression) && !identity {
            substitutions.insert(item.alias.clone(), item.expression.clone());
        }
    }
    let preserved = identity_projection_aliases(items);
    let keep_as_var = BTreeSet::new();
    let mut pushable = Vec::new();
    let mut stay = Vec::new();
    for term in pending {
        let refs = expression_aliases(&term);
        let resolvable = refs
            .iter()
            .all(|alias| preserved.contains(alias) || substitutions.contains_key(alias));
        if !resolvable {
            stay.push(term);
            continue;
        }
        let rewritten = crate::plan::lower::substitute_aliases(&term, &substitutions, &keep_as_var);
        // The rewritten predicate must only mention names whose meaning is
        // the same below this projection.
        if expression_aliases(&rewritten)
            .iter()
            .any(|alias| rebound.contains(alias))
        {
            stay.push(term);
            continue;
        }
        pushable.push(rewritten);
    }
    (pushable, stay)
}

/// A projection expression safe to evaluate below its own projection: a
/// bare variable or a property access chain rooted at one.
fn is_substitutable_projection(expr: &Expression) -> bool {
    match &expr.kind {
        ExpressionKind::Variable(_) => true,
        ExpressionKind::Property(access) => is_substitutable_projection(&access.target),
        _ => false,
    }
}

fn identity_projection_aliases(items: &[crate::plan::logical::ProjectionItem]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for it in items {
        if let ExpressionKind::Variable(id) = &it.expression.kind {
            if id.name == it.alias {
                out.insert(id.name.clone());
            }
        }
    }
    out
}

// Make the `and_chain` and `CreateElement` imports retain a use across
// the file even if a future refactor stops referencing them directly.
#[allow(dead_code)]
fn _retain_imports(_a: fn(Vec<Expression>) -> Option<Expression>, _e: &CreateElement) {
    let _ = and_chain;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::optimize;
    use crate::parser::ast::{BinaryOp, Identifier, Literal, RelationshipDirection};
    use crate::parser::SourceSpan;
    use crate::plan::logical::ShortestMode;
    use crate::plan::logical::{AggregateExpr, ProjectionItem, RowCount};

    fn span() -> SourceSpan {
        SourceSpan::point(0)
    }

    fn var(name: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Variable(Identifier::new(name, span())),
            span: span(),
        }
    }

    fn prop(alias: &str, key: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Property(Box::new(crate::parser::ast::PropertyAccess {
                target: var(alias),
                key: Identifier::new(key, span()),
                span: span(),
            })),
            span: span(),
        }
    }

    fn int(n: i64) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::Integer(n)),
            span: span(),
        }
    }

    fn binop(op: BinaryOp, l: Expression, r: Expression) -> Expression {
        Expression {
            kind: ExpressionKind::Binary {
                op,
                left: Box::new(l),
                right: Box::new(r),
            },
            span: span(),
        }
    }

    fn scan(label: &str, alias: &str) -> LogicalPlan {
        LogicalPlan::NodeScan {
            label: Some(label.into()),
            alias: alias.into(),
            predicates: vec![],
            projection: None,
        }
    }

    /// Helper: filter root predicate when input is the given operator name.
    fn root_filter(plan: &LogicalPlan) -> Option<&Expression> {
        if let LogicalPlan::Filter { predicate, .. } = plan {
            Some(predicate)
        } else {
            None
        }
    }

    fn empty_catalog() -> crate::cost::StatsCatalog {
        crate::cost::StatsCatalog::empty()
    }

    #[test]
    fn pushes_filter_below_expand_when_pred_refs_source() {
        // Filter(a.age > z) over Expand(a -> b). The non-literal RHS keeps
        // the conjunct from being absorbed into NodeScan.predicates by
        // RFC-013, so the test still observes the structural shape
        // it cares about (Filter sandwiched between Expand and NodeScan).
        let pred = binop(BinaryOp::Gt, prop("a", "age"), var("z"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan("Person", "a")),
                source: "a".into(),
                edge_type: Some(vec!["KNOWS".into()]),
                direction: RelationshipDirection::Right,
                rel_alias: None,
                target_alias: "b".into(),
                target_labels: vec![],
                length: None,
                optional: false,
                back_reference: false,
                shortest: ShortestMode::None,
                path_binding: None,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        // Expand should now sit above a Filter that wraps NodeScan.
        match optimized {
            LogicalPlan::Expand { input, .. } => match *input {
                LogicalPlan::Filter { input, .. } => {
                    assert!(matches!(*input, LogicalPlan::NodeScan { .. }));
                }
                other => panic!("expected Filter under Expand, got {:?}", other),
            },
            other => panic!("expected Expand at root, got {:?}", other),
        }
    }

    #[test]
    fn keeps_filter_above_expand_when_pred_refs_target() {
        let pred = binop(BinaryOp::Eq, prop("b", "name"), int(1));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan("Person", "a")),
                source: "a".into(),
                edge_type: Some(vec!["KNOWS".into()]),
                direction: RelationshipDirection::Right,
                rel_alias: None,
                target_alias: "b".into(),
                target_labels: vec![],
                length: None,
                optional: false,
                back_reference: false,
                shortest: ShortestMode::None,
                path_binding: None,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        // Filter must remain at the top.
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
        if let LogicalPlan::Filter { input, .. } = optimized {
            assert!(matches!(*input, LogicalPlan::Expand { .. }));
        }
    }

    #[test]
    fn keeps_filter_above_optional_expand_when_pred_refs_target() {
        // Same rule applies: rel_alias and target_alias are introduced
        // by the Expand, so pred over them stays above. (Independent of
        // optional flag — semantics is enforced by 3VL in the Filter.)
        let pred = binop(BinaryOp::Eq, prop("b", "name"), int(1));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan("Person", "a")),
                source: "a".into(),
                edge_type: Some(vec!["KNOWS".into()]),
                direction: RelationshipDirection::Right,
                rel_alias: None,
                target_alias: "b".into(),
                target_labels: vec![],
                length: None,
                optional: true,
                back_reference: false,
                shortest: ShortestMode::None,
                path_binding: None,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn splits_compound_predicate_across_expand() {
        // Filter(a.age > 30 AND b.name = 1) → Filter(b.name=1) on top,
        // Filter(a.age>30) below Expand.
        let a_pred = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let b_pred = binop(BinaryOp::Eq, prop("b", "name"), prop("b", "z_const"));
        let compound = binop(BinaryOp::And, a_pred, b_pred);
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan("Person", "a")),
                source: "a".into(),
                edge_type: Some(vec!["KNOWS".into()]),
                direction: RelationshipDirection::Right,
                rel_alias: None,
                target_alias: "b".into(),
                target_labels: vec![],
                length: None,
                optional: false,
                back_reference: false,
                shortest: ShortestMode::None,
                path_binding: None,
            }),
            predicate: compound,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::Filter { input, predicate } => {
                // Top filter retains the b.name predicate.
                let aliases = expression_aliases(&predicate);
                assert!(aliases.contains("b"));
                assert!(!aliases.contains("a"));
                match *input {
                    LogicalPlan::Expand { input, .. } => match *input {
                        LogicalPlan::Filter { predicate, .. } => {
                            let aliases = expression_aliases(&predicate);
                            assert!(aliases.contains("a"));
                            assert!(!aliases.contains("b"));
                        }
                        other => panic!("expected Filter under Expand, got {:?}", other),
                    },
                    other => panic!("expected Expand under top Filter, got {:?}", other),
                }
            }
            other => panic!("expected Filter at root, got {:?}", other),
        }
    }

    #[test]
    fn splits_filter_across_cross_product() {
        // Filter(a.age > 30 AND b.name = 1) over CrossProduct(L=a, R=b)
        let a_pred = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let b_pred = binop(BinaryOp::Eq, prop("b", "name"), prop("b", "z_const"));
        let compound = binop(BinaryOp::And, a_pred, b_pred);
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: compound,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::CrossProduct { left, right } => {
                // Each side should now carry its own Filter.
                let l_has_filter = matches!(*left, LogicalPlan::Filter { .. });
                let r_has_filter = matches!(*right, LogicalPlan::Filter { .. });
                assert!(l_has_filter, "left should carry a.age filter");
                assert!(r_has_filter, "right should carry b.name filter");
            }
            other => panic!("expected CrossProduct root, got {:?}", other),
        }
    }

    #[test]
    fn keeps_mixed_side_filter_above_cross_product() {
        // Filter(a.x = b.y) over CrossProduct(L=a, R=b) — neither side
        // owns both aliases, so the filter stays at the top.
        let pred = binop(BinaryOp::Eq, prop("a", "x"), prop("b", "y"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn pushes_through_project_when_alias_identity_preserved() {
        // Filter(a.age > 30) over Project [a=a]
        let pred = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Project {
                input: Box::new(scan("Person", "a")),
                items: vec![ProjectionItem {
                    expression: var("a"),
                    alias: "a".into(),
                }],
                distinct: false,
                discard_input_bindings: true,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::Project { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Project at root, got {:?}", other),
        }
    }

    #[test]
    fn pushes_filter_through_a_renaming_projection() {
        // Filter(x.age > 30) over Project [x=a]: `WITH a AS x WHERE …` is
        // the same query as filtering `a.age` below, so the predicate now
        // substitutes the alias and sinks into the scan. Leaving it above
        // stranded index anchors behind a WITH (sixth field report).
        let pred = binop(BinaryOp::Gt, prop("x", "age"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Project {
                input: Box::new(scan("Person", "a")),
                items: vec![ProjectionItem {
                    expression: var("a"),
                    alias: "x".into(),
                }],
                distinct: false,
                discard_input_bindings: true,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        // The Project is now the root and the rewritten predicate lives
        // below it (absorbed by the scan or as a Filter under the Project).
        let LogicalPlan::Project { input, .. } = &optimized else {
            panic!("expected Project at root, got {optimized:?}");
        };
        let rendered = format!("{input:?}");
        assert!(
            rendered.contains("\"a\"") && rendered.contains("age"),
            "the alias must be substituted back to a.age below the Project: {rendered}"
        );
    }

    #[test]
    fn keeps_filter_above_project_for_a_computed_alias() {
        // Filter(x > 30) over Project [x = a.age * 2]: a computed binding is
        // deliberately NOT substituted — only pure variable/property
        // bindings are, so the pass never duplicates arbitrary evaluation
        // below its own projection.
        let pred = binop(BinaryOp::Gt, var("x"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Project {
                input: Box::new(scan("Person", "a")),
                items: vec![ProjectionItem {
                    expression: binop(BinaryOp::Mul, prop("a", "age"), int(2)),
                    alias: "x".into(),
                }],
                distinct: false,
                discard_input_bindings: true,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn keeps_filter_above_a_distinct_projection() {
        // DISTINCT dedups before the filter; pushing below would change how
        // often a non-deterministic term is evaluated, so the pass declines.
        let pred = binop(BinaryOp::Gt, prop("x", "age"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Project {
                input: Box::new(scan("Person", "a")),
                items: vec![ProjectionItem {
                    expression: var("a"),
                    alias: "x".into(),
                }],
                distinct: true,
                discard_input_bindings: true,
            }),
            predicate: pred,
        };
        assert!(matches!(
            predicate_pushdown(plan),
            LogicalPlan::Filter { .. }
        ));
    }

    #[test]
    fn pushes_through_topn_barrier_keeps_filter_above() {
        // Filter(a.age > 30) over TopN(limit=10) over scan.
        let pred = binop(BinaryOp::Gt, prop("a", "age"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::TopN {
                input: Box::new(scan("Person", "a")),
                keys: vec![],
                skip: RowCount::Const(0),
                limit: RowCount::Const(10),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        // TopN is a barrier: Filter stays above.
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn does_not_push_through_distinct() {
        let pred = binop(BinaryOp::Gt, prop("a", "age"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Distinct {
                input: Box::new(scan("Person", "a")),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn pushes_through_unwind_when_alias_unrelated() {
        // Filter(a.age > 30) over Unwind(list=$xs alias=z) over scan(a)
        let pred = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let list_expr = Expression {
            kind: ExpressionKind::Parameter("xs".into()),
            span: span(),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Unwind {
                input: Box::new(scan("Person", "a")),
                list: list_expr,
                alias: "z".into(),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::Unwind { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Unwind at root, got {:?}", other),
        }
    }

    #[test]
    fn keeps_filter_above_unwind_when_pred_refs_unwind_alias() {
        // Filter(z > 0) over Unwind alias=z — must stay above.
        let pred = binop(BinaryOp::Gt, var("z"), int(0));
        let list_expr = Expression {
            kind: ExpressionKind::Parameter("xs".into()),
            span: span(),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Unwind {
                input: Box::new(scan("Person", "a")),
                list: list_expr,
                alias: "z".into(),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn pushes_filter_to_both_sides_of_union() {
        // Both sides project `a` (NodeScan with alias `a`).
        let pred = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Union {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "a")),
                all: true,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::Union { left, right, .. } => {
                assert!(matches!(*left, LogicalPlan::Filter { .. }));
                assert!(matches!(*right, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Union at root, got {:?}", other),
        }
    }

    #[test]
    fn flows_filter_into_outer_of_semi_apply_only() {
        // SemiApply(input=scan(a), subplan=scan(b)). Filter(a.age > 30)
        // should land on the outer side.
        let pred = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::SemiApply {
                input: Box::new(scan("Person", "a")),
                subplan: Box::new(scan("Person", "b")),
                negated: false,
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::SemiApply { input, subplan, .. } => {
                assert!(matches!(*input, LogicalPlan::Filter { .. }));
                // Subplan untouched.
                assert!(matches!(*subplan, LogicalPlan::NodeScan { .. }));
            }
            other => panic!("expected SemiApply at root, got {:?}", other),
        }
    }

    #[test]
    fn keeps_filter_above_aggregate_when_pred_refs_agg_alias() {
        // Filter(cnt > 5) over Aggregate(... aggregations=[cnt=count(*)])
        // → stays above.
        let pred = binop(BinaryOp::Gt, var("cnt"), int(5));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Aggregate {
                input: Box::new(scan("Person", "a")),
                group_by: vec![],
                aggregations: vec![(
                    "cnt".into(),
                    AggregateExpr::Count {
                        arg: None,
                        distinct: false,
                    },
                )],
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn pushes_filter_below_aggregate_when_pred_refs_identity_group_key() {
        // Filter(a.age > 30) over Aggregate(group=[a=a], aggs=[cnt=count(*)])
        // → pushable since `a` is an identity group key.
        let pred = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Aggregate {
                input: Box::new(scan("Person", "a")),
                group_by: vec![(var("a"), "a".into())],
                aggregations: vec![(
                    "cnt".into(),
                    AggregateExpr::Count {
                        arg: None,
                        distinct: false,
                    },
                )],
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::Aggregate { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Aggregate at root, got {:?}", other),
        }
    }

    #[test]
    fn write_op_acts_as_barrier_for_filter_above() {
        // Filter above Set is unusual but if it ever appears, pending
        // predicates remain above the Set (do not seep into Set's input).
        let pred = binop(BinaryOp::Gt, prop("a", "age"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Set {
                input: Box::new(scan("Person", "a")),
                items: vec![],
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
        if let LogicalPlan::Filter { input, .. } = optimized {
            assert!(matches!(*input, LogicalPlan::Set { .. }));
        }
    }

    #[test]
    fn idempotent_on_already_optimized_plan() {
        // Run pushdown twice; result should not change after the first.
        let pred = binop(BinaryOp::Gt, prop("a", "age"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan("Person", "a")),
                source: "a".into(),
                edge_type: Some(vec!["KNOWS".into()]),
                direction: RelationshipDirection::Right,
                rel_alias: None,
                target_alias: "b".into(),
                target_labels: vec![],
                length: None,
                optional: false,
                back_reference: false,
                shortest: ShortestMode::None,
                path_binding: None,
            }),
            predicate: pred,
        };
        let once = predicate_pushdown(plan);
        let twice = predicate_pushdown(once.clone());
        assert_eq!(once, twice);
    }

    #[test]
    fn nested_filter_collapses_through_pushdown() {
        // Two stacked Filters with overlapping pushable predicates.
        let p1 = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let p2 = binop(BinaryOp::Eq, prop("a", "firstName"), prop("a", "z_const"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(LogicalPlan::Expand {
                    input: Box::new(scan("Person", "a")),
                    source: "a".into(),
                    edge_type: Some(vec!["KNOWS".into()]),
                    direction: RelationshipDirection::Right,
                    rel_alias: None,
                    target_alias: "b".into(),
                    target_labels: vec![],
                    length: None,
                    optional: false,
                    back_reference: false,
                    shortest: ShortestMode::None,
                    path_binding: None,
                }),
                predicate: p1,
            }),
            predicate: p2,
        };
        let optimized = predicate_pushdown(plan);
        // Both filters should end up below the Expand.
        match optimized {
            LogicalPlan::Expand { input, .. } => {
                // Stack of Filter(p2 over Filter(p1 over NodeScan)).
                // Each level should be a Filter.
                let mut node: &LogicalPlan = &input;
                let mut depth = 0;
                while let LogicalPlan::Filter { input: deeper, .. } = node {
                    depth += 1;
                    node = deeper;
                }
                assert!(depth >= 1, "expected at least one Filter below Expand");
                assert!(matches!(node, LogicalPlan::NodeScan { .. }));
            }
            other => panic!("expected Expand at root, got {:?}", other),
        }
    }

    #[test]
    fn empty_plan_unchanged() {
        let p = LogicalPlan::Empty;
        assert_eq!(predicate_pushdown(p.clone()), p);
    }

    #[test]
    fn argument_plan_unchanged_when_no_filter() {
        let p = LogicalPlan::Argument {
            bindings: vec!["a".into()],
        };
        assert_eq!(predicate_pushdown(p.clone()), p);
    }

    #[test]
    fn pushdown_then_optimize_fixpoint() {
        let cat = empty_catalog();
        let pred = binop(BinaryOp::Gt, prop("a", "age"), int(30));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(scan("Person", "a")),
                source: "a".into(),
                edge_type: Some(vec!["KNOWS".into()]),
                direction: RelationshipDirection::Right,
                rel_alias: None,
                target_alias: "b".into(),
                target_labels: vec![],
                length: None,
                optional: false,
                back_reference: false,
                shortest: ShortestMode::None,
                path_binding: None,
            }),
            predicate: pred,
        };
        let once = optimize(plan.clone(), &cat);
        let twice = optimize(once.clone(), &cat);
        assert_eq!(once, twice, "optimize should be idempotent");
    }

    #[test]
    fn node_by_id_keeps_alias_filter_above() {
        // Filter(a.x = 1) above NodeById{alias=a, ...} should stay above
        // — `a` is the introduced alias.
        let pred = binop(BinaryOp::Eq, prop("a", "x"), int(1));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::NodeById {
                input: Box::new(LogicalPlan::Empty),
                label: Some("Person".into()),
                alias: "a".into(),
                id: int(1),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn node_by_id_pushes_predicate_referencing_outer_binding() {
        // Filter(b.y > 0) over NodeById{alias=a, input=Argument{b}}
        // pushable into the Argument-rooted input.
        let pred = binop(BinaryOp::Gt, prop("b", "y"), int(0));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::NodeById {
                input: Box::new(LogicalPlan::Argument {
                    bindings: vec!["b".into()],
                }),
                label: Some("Person".into()),
                alias: "a".into(),
                id: int(1),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::NodeById { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected NodeById at root, got {:?}", other),
        }
    }

    #[test]
    fn parameter_only_predicate_stays_above_cross_product() {
        // Filter($p > 0) — no aliases referenced; stays at top (constant).
        let p_expr = Expression {
            kind: ExpressionKind::Parameter("p".into()),
            span: span(),
        };
        let pred = binop(BinaryOp::Gt, p_expr, int(0));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let optimized = predicate_pushdown(plan);
        assert!(matches!(optimized, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn split_distributes_two_conjuncts_to_both_cross_sides() {
        // Filter(a.age > 30 AND b.age < 50) over CrossProduct(a, b).
        let a = binop(BinaryOp::Gt, prop("a", "age"), prop("a", "z_const"));
        let b = binop(BinaryOp::Lt, prop("b", "age"), prop("b", "z_const"));
        let compound = binop(BinaryOp::And, a, b);
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: compound,
        };
        let optimized = predicate_pushdown(plan);
        match optimized {
            LogicalPlan::CrossProduct { left, right } => {
                let l_pred = root_filter(&left).expect("left has filter");
                let r_pred = root_filter(&right).expect("right has filter");
                assert!(expression_aliases(l_pred).contains("a"));
                assert!(expression_aliases(r_pred).contains("b"));
            }
            other => panic!("expected CrossProduct, got {:?}", other),
        }
    }
}
