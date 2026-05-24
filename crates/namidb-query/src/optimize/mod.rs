//! Logical-plan rewrites (RFC-011).
//!
//! The optimizer pipeline takes a freshly-lowered [`LogicalPlan`] and
//! rewrites its shape to reduce the cardinality that expensive operators
//! (`Expand`, `CrossProduct`, `SemiApply`, `PatternList`) process. Today
//! the pipeline consists of two passes iterated to fixpoint:
//!
//! 1. [`predicate_pushdown`] — split each `Filter` predicate by AND and
//! push each conjunct as close to the leaves as possible.
//! 2. [`normalize_filters`] — merge adjacent `Filter` nodes, drop
//! `Filter(true)` and the defensive `__label_eq` filter that the
//! lowering emits over `Expand { target_label: Some(_) }`.
//!
//! Future rewrites (join reorder, hash conversion) will plug
//! into this pipeline.
//!
//! [`LogicalPlan`]: crate::plan::LogicalPlan

use std::collections::BTreeSet;

use crate::cost::StatsCatalog;
use crate::parser::ast::{
    BinaryOp, CaseBranch, Expression, ExpressionKind, MapLiteral, NodePattern, PatternElement,
    RelationshipPattern,
};
use crate::parser::SourceSpan;
use crate::plan::logical::{CreateElement, LogicalPlan, ProjectionItem};

pub mod decorrelation;
pub mod join_conversion;
pub mod join_reorder;
pub mod normalize;
pub mod parquet_pushdown;
pub mod projection_pushdown;
pub mod pushdown;
pub mod unique_lookup;

pub use decorrelation::convert_semi_apply_to_hash_semi_join;
pub use join_conversion::convert_cross_to_hash;
pub use join_reorder::reorder_joins;
pub use normalize::normalize_filters;
pub use parquet_pushdown::{classify_pending_for_scan, try_into_scan_predicate};
pub use projection_pushdown::apply_projection_pushdown;
pub use pushdown::predicate_pushdown;

/// Maximum number of pushdown+normalize rounds before the optimizer
/// gives up looking for a fixpoint. Each well-formed rewrite is
/// idempotent after at most 2 rounds; the cap is defensive against
/// future bugs.
const MAX_FIXPOINT_ROUNDS: usize = 8;

/// Apply the full optimizer pipeline to `plan`. Idempotent.
///
/// `catalog` is accepted (and ignored today) so that future cost-aware
/// rewrites (join reorder) can be added without breaking
/// signatures.
pub fn optimize(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    let mut current = plan;
    for _ in 0..MAX_FIXPOINT_ROUNDS {
        // RFC-pending: unique-property lookup rewrite runs FIRST so the
        // downstream pushdowns (which assume NodeScan input) see the
        // already-replaced point-lookup operator and don't re-introduce
        // a Filter on top of it.
        let unique_lookup = unique_lookup::apply_unique_property_lookup(current.clone(), catalog);
        let pushed = normalize_filters(predicate_pushdown(unique_lookup));
        let hashed = convert_cross_to_hash(pushed, catalog);
        let decorrelated = convert_semi_apply_to_hash_semi_join(hashed, catalog);
        // RFC-016: reorder HashJoin orientations using the now-final
        // estimate of each branch (predicates + structural rewrites
        // already applied).
        let reordered = reorder_joins(decorrelated, catalog);
        // RFC-015: projection pushdown runs LAST in each round so it
        // sees the final NodeScan shape (with predicates already
        // absorbed) and can compute the minimal column set.
        let next = apply_projection_pushdown(reordered);
        if next == current {
            return next;
        }
        current = next;
    }
    current
}

// ─────────────────────── shared helpers (RFC-011 §4) ───────────────────────

/// Set of aliases referenced anywhere in `expr`. Property accesses
/// contribute their target alias; pattern subqueries (`Exists`,
/// `PatternComprehension`) contribute every binding identifier they
/// mention plus everything their nested expressions reference.
///
/// Over-approximation is the safe direction: a wider alias set keeps
/// predicates higher in the plan rather than pushing them past nodes
/// that would change their semantics.
pub(crate) fn expression_aliases(expr: &Expression) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    visit_expression(expr, &mut out);
    out
}

fn visit_expression(expr: &Expression, out: &mut BTreeSet<String>) {
    match &expr.kind {
        ExpressionKind::Variable(id) => {
            out.insert(id.name.clone());
        }
        ExpressionKind::Property(pa) => {
            visit_expression(&pa.target, out);
        }
        ExpressionKind::Index { target, index } => {
            visit_expression(target, out);
            visit_expression(index, out);
        }
        ExpressionKind::Range { target, from, to } => {
            visit_expression(target, out);
            if let Some(f) = from {
                visit_expression(f, out);
            }
            if let Some(t) = to {
                visit_expression(t, out);
            }
        }
        ExpressionKind::Unary { expr, .. } => visit_expression(expr, out),
        ExpressionKind::Binary { left, right, .. } => {
            visit_expression(left, out);
            visit_expression(right, out);
        }
        ExpressionKind::In { item, list } => {
            visit_expression(item, out);
            visit_expression(list, out);
        }
        ExpressionKind::Between { target, low, high } => {
            visit_expression(target, out);
            visit_expression(low, out);
            visit_expression(high, out);
        }
        ExpressionKind::StringTest {
            target, pattern, ..
        } => {
            visit_expression(target, out);
            visit_expression(pattern, out);
        }
        ExpressionKind::IsNull { expr, .. } => visit_expression(expr, out),
        ExpressionKind::FunctionCall { args, .. } => {
            for a in args {
                visit_expression(a, out);
            }
        }
        ExpressionKind::Case {
            scrutinee,
            branches,
            otherwise,
        } => {
            if let Some(s) = scrutinee {
                visit_expression(s, out);
            }
            for CaseBranch { when, then, .. } in branches {
                visit_expression(when, out);
                visit_expression(then, out);
            }
            if let Some(o) = otherwise {
                visit_expression(o, out);
            }
        }
        ExpressionKind::Exists(pe) => visit_pattern_element(pe, out),
        ExpressionKind::List(xs) => {
            for x in xs {
                visit_expression(x, out);
            }
        }
        ExpressionKind::Map(m) => visit_map(m, out),
        ExpressionKind::ListComprehension(lc) => {
            visit_expression(&lc.list, out);
            if let Some(p) = &lc.predicate {
                visit_expression(p, out);
            }
            if let Some(p) = &lc.projection {
                visit_expression(p, out);
            }
            out.insert(lc.variable.name.clone());
        }
        ExpressionKind::PatternComprehension(pc) => {
            if let Some(b) = &pc.binding {
                out.insert(b.name.clone());
            }
            visit_pattern_element(&pc.pattern, out);
            if let Some(p) = &pc.predicate {
                visit_expression(p, out);
            }
            visit_expression(&pc.projection, out);
        }
        ExpressionKind::Literal(_) | ExpressionKind::Parameter(_) | ExpressionKind::Star => {}
    }
}

fn visit_pattern_element(pe: &PatternElement, out: &mut BTreeSet<String>) {
    visit_node_pattern(&pe.head, out);
    for (rel, node) in &pe.chain {
        visit_relationship_pattern(rel, out);
        visit_node_pattern(node, out);
    }
}

fn visit_node_pattern(np: &NodePattern, out: &mut BTreeSet<String>) {
    if let Some(b) = &np.binding {
        out.insert(b.name.clone());
    }
    if let Some(m) = &np.properties {
        visit_map(m, out);
    }
}

fn visit_relationship_pattern(rp: &RelationshipPattern, out: &mut BTreeSet<String>) {
    if let Some(b) = &rp.binding {
        out.insert(b.name.clone());
    }
    if let Some(m) = &rp.properties {
        visit_map(m, out);
    }
}

fn visit_map(m: &MapLiteral, out: &mut BTreeSet<String>) {
    for (_, v) in &m.entries {
        visit_expression(v, out);
    }
}

/// Aliases that `plan` makes visible to its parent.
pub(crate) fn produced_aliases(plan: &LogicalPlan) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    collect_produced(plan, &mut out);
    out
}

fn collect_produced(plan: &LogicalPlan, out: &mut BTreeSet<String>) {
    match plan {
        LogicalPlan::Empty => {}
        LogicalPlan::Argument { bindings } => {
            for b in bindings {
                out.insert(b.clone());
            }
        }
        LogicalPlan::NodeScan { alias, .. } => {
            out.insert(alias.clone());
        }
        LogicalPlan::NodeById { input, alias, .. }
        | LogicalPlan::NodeByPropertyValue { input, alias, .. } => {
            collect_produced(input, out);
            out.insert(alias.clone());
        }
        LogicalPlan::Expand {
            input,
            target_alias,
            rel_alias,
            ..
        } => {
            collect_produced(input, out);
            out.insert(target_alias.clone());
            if let Some(r) = rel_alias {
                out.insert(r.clone());
            }
        }
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::TopN { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::SemiApply { input, .. } => collect_produced(input, out),
        LogicalPlan::Project {
            items,
            discard_input_bindings,
            input,
            distinct: _,
        } => {
            if !discard_input_bindings {
                collect_produced(input, out);
            }
            for it in items {
                out.insert(it.alias.clone());
            }
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregations,
            ..
        } => {
            for (_, alias) in group_by {
                out.insert(alias.clone());
            }
            for (alias, _) in aggregations {
                out.insert(alias.clone());
            }
        }
        LogicalPlan::Union { left, right, .. } => {
            // Schema-aware: a parent only sees aliases declared by BOTH
            // sides (projection-matched).
            let mut l = BTreeSet::new();
            let mut r = BTreeSet::new();
            collect_produced(left, &mut l);
            collect_produced(right, &mut r);
            for a in l.intersection(&r) {
                out.insert(a.clone());
            }
        }
        LogicalPlan::CrossProduct { left, right } => {
            collect_produced(left, out);
            collect_produced(right, out);
        }
        LogicalPlan::HashJoin { build, probe, .. } => {
            collect_produced(build, out);
            collect_produced(probe, out);
        }
        LogicalPlan::HashSemiJoin { outer, .. } => {
            // Semi-join semantics: only `outer` bindings survive.
            collect_produced(outer, out);
        }
        LogicalPlan::Unwind { input, alias, .. } => {
            collect_produced(input, out);
            out.insert(alias.clone());
        }
        LogicalPlan::PatternList { input, alias, .. } => {
            collect_produced(input, out);
            out.insert(alias.clone());
        }
        LogicalPlan::Create { input, elements }
        | LogicalPlan::Merge {
            input,
            pattern: elements,
            ..
        } => {
            collect_produced(input, out);
            for el in elements {
                if let Some(a) = create_element_alias(el) {
                    out.insert(a.to_string());
                }
            }
        }
        LogicalPlan::Set { input, .. }
        | LogicalPlan::Remove { input, .. }
        | LogicalPlan::Delete { input, .. } => collect_produced(input, out),
    }
}

fn create_element_alias(el: &CreateElement) -> Option<&str> {
    match el {
        CreateElement::Node { alias, .. } => Some(alias.as_str()),
        CreateElement::Rel { alias, .. } => alias.as_deref(),
    }
}

/// AND-flatten: `a AND b AND c AND d` → `vec![a, b, c, d]`. Single
/// non-AND expressions return `vec![expr]`.
pub(crate) fn split_and_terms(expr: &Expression) -> Vec<Expression> {
    fn walk(expr: &Expression, out: &mut Vec<Expression>) {
        if let ExpressionKind::Binary {
            op: BinaryOp::And,
            left,
            right,
        } = &expr.kind
        {
            walk(left, out);
            walk(right, out);
        } else {
            out.push(expr.clone());
        }
    }
    let mut out = Vec::new();
    walk(expr, &mut out);
    out
}

/// Build a left-associative AND chain. Returns `None` for an empty
/// vector. Preserves the source-text order of `terms`.
pub(crate) fn and_chain(terms: Vec<Expression>) -> Option<Expression> {
    let mut iter = terms.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, next| {
        let span = SourceSpan::new(acc.span.start, next.span.end);
        Expression {
            kind: ExpressionKind::Binary {
                op: BinaryOp::And,
                left: Box::new(acc),
                right: Box::new(next),
            },
            span,
        }
    }))
}

/// Wrap `plan` in a `Filter` whose predicate is the AND of `terms`.
/// If `terms` is empty, return `plan` untouched.
pub(crate) fn apply_filters(plan: LogicalPlan, terms: Vec<Expression>) -> LogicalPlan {
    match and_chain(terms) {
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        },
        None => plan,
    }
}

/// True iff `predicate` is the synthetic `__label_eq(alias, "Label")`
/// call that the lowering emits over an `Expand` with a declared
/// `target_label`. Used by [`normalize_filters`] to remove redundant
/// defensive filters.
pub(crate) fn is_synthetic_label_eq(predicate: &Expression, alias: &str, label: &str) -> bool {
    if let ExpressionKind::FunctionCall { name, args, .. } = &predicate.kind {
        let name_matches = name
            .segments
            .first()
            .map(|s| s.name.eq_ignore_ascii_case("__label_eq"))
            .unwrap_or(false);
        if !name_matches || args.len() != 2 {
            return false;
        }
        let alias_arg_matches = matches!(
        &args[0].kind,
        ExpressionKind::Variable(id) if id.name == alias
        );
        let label_arg_matches = matches!(
        &args[1].kind,
        ExpressionKind::Literal(crate::parser::ast::Literal::String(s)) if s == label
        );
        return alias_arg_matches && label_arg_matches;
    }
    false
}

/// True iff `predicate` is a cross-side equality `lhs = rhs` where one
/// side's aliases are entirely on the left subtree and the other side's
/// entirely on the right. Used by EXPLAIN to flag join candidates after
/// pushdown leaves an equality above a `CrossProduct` (the future optimizer will
/// detect this same shape to materialise a HashJoin).
pub(crate) fn is_join_candidate(
    predicate: &Expression,
    left: &BTreeSet<String>,
    right: &BTreeSet<String>,
) -> bool {
    if let ExpressionKind::Binary {
        op: BinaryOp::Eq,
        left: l,
        right: r,
    } = &predicate.kind
    {
        let la = expression_aliases(l);
        let ra = expression_aliases(r);
        let l_then_r =
            la.is_subset(left) && ra.is_subset(right) && !la.is_empty() && !ra.is_empty();
        let r_then_l =
            la.is_subset(right) && ra.is_subset(left) && !la.is_empty() && !ra.is_empty();
        l_then_r || r_then_l
    } else {
        false
    }
}

#[allow(dead_code)]
fn _keep_imports_live(_p: &ProjectionItem) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{Identifier, Literal, QualifiedName};
    use crate::plan::logical::ShortestMode;

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

    fn lit_true() -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::Boolean(true)),
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

    #[test]
    fn expression_aliases_collects_simple_variables() {
        let e = binop(BinaryOp::And, var("a"), var("b"));
        let s = expression_aliases(&e);
        assert!(s.contains("a"));
        assert!(s.contains("b"));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn expression_aliases_walks_property_access() {
        let e = binop(BinaryOp::Gt, prop("a", "age"), int(30));
        let s = expression_aliases(&e);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec!["a"]);
    }

    #[test]
    fn expression_aliases_ignores_literal_and_parameter() {
        let p = Expression {
            kind: ExpressionKind::Parameter("foo".into()),
            span: span(),
        };
        let e = binop(BinaryOp::Eq, p, int(1));
        let s = expression_aliases(&e);
        assert!(s.is_empty());
    }

    #[test]
    fn split_and_terms_flattens_left_skewed_tree() {
        // (a AND b) AND c → [a, b, c]
        let inner = binop(BinaryOp::And, var("a"), var("b"));
        let outer = binop(BinaryOp::And, inner, var("c"));
        let parts = split_and_terms(&outer);
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn split_and_terms_flattens_right_skewed_tree() {
        // a AND (b AND c) → [a, b, c]
        let inner = binop(BinaryOp::And, var("b"), var("c"));
        let outer = binop(BinaryOp::And, var("a"), inner);
        let parts = split_and_terms(&outer);
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn split_and_terms_returns_single_for_non_and() {
        let e = binop(BinaryOp::Or, var("a"), var("b"));
        let parts = split_and_terms(&e);
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn and_chain_returns_none_for_empty() {
        assert!(and_chain(vec![]).is_none());
    }

    #[test]
    fn and_chain_preserves_single_term() {
        let v = var("a");
        let chained = and_chain(vec![v.clone()]).unwrap();
        assert_eq!(chained, v);
    }

    #[test]
    fn and_chain_builds_left_associative_tree() {
        let chained = and_chain(vec![var("a"), var("b"), var("c")]).unwrap();
        // ((a AND b) AND c)
        match chained.kind {
            ExpressionKind::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                assert!(matches!(
                right.kind,
                ExpressionKind::Variable(ref id) if id.name == "c"
                ));
                match left.kind {
                    ExpressionKind::Binary {
                        op: BinaryOp::And,
                        left: ll,
                        right: lr,
                    } => {
                        assert!(matches!(
                        ll.kind,
                        ExpressionKind::Variable(ref id) if id.name == "a"
                        ));
                        assert!(matches!(
                        lr.kind,
                        ExpressionKind::Variable(ref id) if id.name == "b"
                        ));
                    }
                    _ => panic!("expected nested AND"),
                }
            }
            _ => panic!("expected AND"),
        }
    }

    #[test]
    fn apply_filters_skips_empty() {
        let plan = LogicalPlan::Empty;
        let result = apply_filters(plan.clone(), vec![]);
        assert_eq!(result, plan);
    }

    #[test]
    fn apply_filters_wraps_terms() {
        let plan = LogicalPlan::Empty;
        let result = apply_filters(plan, vec![lit_true()]);
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn produced_aliases_scan_emits_alias() {
        let plan = LogicalPlan::NodeScan {
            label: Some("P".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let s = produced_aliases(&plan);
        assert!(s.contains("a"));
    }

    #[test]
    fn produced_aliases_expand_includes_target_and_rel() {
        let plan = LogicalPlan::Expand {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("P".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            source: "a".into(),
            edge_type: Some("KNOWS".into()),
            direction: crate::parser::RelationshipDirection::Right,
            rel_alias: Some("r".into()),
            target_alias: "b".into(),
            target_label: None,
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let s = produced_aliases(&plan);
        assert!(s.contains("a"));
        assert!(s.contains("b"));
        assert!(s.contains("r"));
    }

    #[test]
    fn produced_aliases_project_discard_drops_input() {
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("P".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            items: vec![ProjectionItem {
                expression: var("a"),
                alias: "x".into(),
            }],
            distinct: false,
            discard_input_bindings: true,
        };
        let s = produced_aliases(&plan);
        assert!(s.contains("x"));
        assert!(!s.contains("a"));
    }

    #[test]
    fn produced_aliases_union_intersects_sides() {
        let l = LogicalPlan::NodeScan {
            label: Some("P".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let r = LogicalPlan::NodeScan {
            label: Some("P".into()),
            alias: "b".into(),
            predicates: vec![],
            projection: None,
        };
        let plan = LogicalPlan::Union {
            left: Box::new(l),
            right: Box::new(r),
            all: true,
        };
        let s = produced_aliases(&plan);
        // Neither alias is in both sides (a vs b) → intersection empty.
        assert!(s.is_empty());
    }

    #[test]
    fn is_synthetic_label_eq_recognises_the_call() {
        let p = Expression {
            kind: ExpressionKind::FunctionCall {
                name: QualifiedName::single(Identifier::new("__label_eq", span())),
                args: vec![
                    var("b"),
                    Expression {
                        kind: ExpressionKind::Literal(Literal::String("Person".into())),
                        span: span(),
                    },
                ],
                distinct: false,
            },
            span: span(),
        };
        assert!(is_synthetic_label_eq(&p, "b", "Person"));
        assert!(!is_synthetic_label_eq(&p, "c", "Person"));
        assert!(!is_synthetic_label_eq(&p, "b", "Comment"));
    }

    #[test]
    fn is_join_candidate_detects_cross_side_equality() {
        let mut left = BTreeSet::new();
        left.insert("a".to_string());
        let mut right = BTreeSet::new();
        right.insert("b".to_string());
        let p = binop(BinaryOp::Eq, prop("a", "name"), prop("b", "name"));
        assert!(is_join_candidate(&p, &left, &right));
        // Mirror.
        let mirror = binop(BinaryOp::Eq, prop("b", "name"), prop("a", "name"));
        assert!(is_join_candidate(&mirror, &left, &right));
    }

    #[test]
    fn is_join_candidate_rejects_same_side_equality() {
        let mut left = BTreeSet::new();
        left.insert("a".to_string());
        let mut right = BTreeSet::new();
        right.insert("b".to_string());
        let p = binop(BinaryOp::Eq, prop("a", "x"), prop("a", "y"));
        assert!(!is_join_candidate(&p, &left, &right));
    }

    #[test]
    fn is_join_candidate_rejects_non_equality() {
        let mut left = BTreeSet::new();
        left.insert("a".to_string());
        let mut right = BTreeSet::new();
        right.insert("b".to_string());
        let p = binop(BinaryOp::Gt, prop("a", "x"), prop("b", "y"));
        assert!(!is_join_candidate(&p, &left, &right));
    }
}
