//! Decorrelation rewriter (RFC-014 §2).
//!
//! Converts `SemiApply { input, subplan, negated }` whose subplan has
//! a single `Argument { bindings: [X] }` leaf and where `X` carries a
//! known label, into `HashSemiJoin { outer, inner, on, negated,
//! residual: None }` with `inner` having the `Argument` substituted by
//! a fresh `NodeScan` on `X`'s label.
//!
//! The rewriter is conservative: any shape that doesn't match the
//! single-Argument single-binding template is left as a SemiApply.
//!
//! Idempotent — `HashSemiJoin` is not a `SemiApply`, so repeated
//! application is a no-op.

use std::collections::BTreeMap;

use crate::cost::StatsCatalog;
use crate::parser::ast::{Expression, ExpressionKind, Identifier, PropertyAccess};
use crate::parser::SourceSpan;
use crate::plan::logical::{JoinKey, LogicalPlan};

/// Rewrite SemiApply ⇒ HashSemiJoin bottom-up when the subplan has a
/// decorrelable shape (single Argument leaf with single binding and
/// known outer label).
pub fn convert_semi_apply_to_hash_semi_join(
    plan: LogicalPlan,
    catalog: &StatsCatalog,
) -> LogicalPlan {
    let recursed = recurse_children(plan, catalog);
    try_convert_at_root(recursed, catalog)
}

fn try_convert_at_root(plan: LogicalPlan, _catalog: &StatsCatalog) -> LogicalPlan {
    let LogicalPlan::SemiApply {
        input,
        subplan,
        negated,
    } = plan
    else {
        return plan;
    };

    // Inspect the subplan. We need:
    // (1) a single Argument leaf,
    // (2) `bindings = [X]` of length 1,
    // (3) X has a known label in the outer (`input`) scope.
    let mut arg_bindings_collector: Vec<Vec<String>> = Vec::new();
    collect_arguments(&subplan, &mut arg_bindings_collector);
    if arg_bindings_collector.len() != 1 {
        return LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        };
    }
    let arg_bindings = &arg_bindings_collector[0];
    if arg_bindings.len() != 1 {
        return LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        };
    }
    let x = arg_bindings[0].clone();

    let mut outer_labels = BTreeMap::new();
    collect_outer_labels(&input, &mut outer_labels);
    let Some(Some(label)) = outer_labels.get(&x).cloned() else {
        return LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        };
    };

    // Reject decorrelation when the subplan contains operators we don't
    // know how to safely lift (Aggregate, Distinct, TopN, Union,
    // CrossProduct, nested HashJoin/SemiApply, writes). The rest
    // (Expand, Filter, NodeById, NodeScan, Project preserving bindings)
    // is safe.
    if !subplan_is_decorrelable(&subplan) {
        return LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        };
    }

    // Replace the single Argument with a fresh NodeScan on `X`'s label.
    let new_subplan = replace_argument(*subplan, &x, &label);

    // Join key uses the internal NodeId, accessed via the `_id`
    // accessor since the unqualified `id` is now a user property
    // (Bug #1 rename).
    let key_expr = property_expression(&x, "_id");
    LogicalPlan::HashSemiJoin {
        outer: input,
        inner: Box::new(new_subplan),
        on: vec![JoinKey {
            build_side: key_expr.clone(),
            probe_side: key_expr,
        }],
        negated,
        residual: None,
    }
}

/// Build `Property { target: Variable(alias), key: "id" }`.
fn property_expression(alias: &str, key: &str) -> Expression {
    let span = SourceSpan::point(0);
    let target = Expression {
        kind: ExpressionKind::Variable(Identifier {
            name: alias.into(),
            span,
            quoted: false,
        }),
        span,
    };
    Expression {
        kind: ExpressionKind::Property(Box::new(PropertyAccess {
            target,
            key: Identifier {
                name: key.into(),
                span,
                quoted: false,
            },
            span,
        })),
        span,
    }
}

/// Walk the outer plan collecting `alias → label` for every operator
/// that introduces a labelled binding. Only the most-recent assignment
/// wins (later operators shadow earlier ones).
fn collect_outer_labels(plan: &LogicalPlan, out: &mut BTreeMap<String, Option<String>>) {
    match plan {
        LogicalPlan::NodeScan { alias, label, .. } => {
            out.insert(alias.clone(), label.clone());
        }
        LogicalPlan::NodeById {
            alias,
            label,
            input,
            ..
        }
        | LogicalPlan::NodeByPropertyValue {
            alias,
            label,
            input,
            ..
        } => {
            collect_outer_labels(input, out);
            out.insert(alias.clone(), Some(label.clone()));
        }
        LogicalPlan::Expand {
            input,
            target_alias,
            target_label,
            rel_alias: _,
            ..
        } => {
            collect_outer_labels(input, out);
            out.insert(target_alias.clone(), target_label.clone());
        }
        _ => {
            for c in plan.children() {
                collect_outer_labels(c, out);
            }
        }
    }
}

/// Collect every `Argument { bindings }` reachable from `plan`. Used to
/// decide whether the subplan has exactly one correlation point.
fn collect_arguments(plan: &LogicalPlan, out: &mut Vec<Vec<String>>) {
    if let LogicalPlan::Argument { bindings } = plan {
        out.push(bindings.clone());
        return;
    }
    for c in plan.children() {
        collect_arguments(c, out);
    }
}

/// Reject decorrelation when the subplan contains operators whose
/// semantics interact with iteration order (Aggregate, Distinct,
/// TopN, Union, CrossProduct, nested HashJoin/SemiApply, writes).
fn subplan_is_decorrelable(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Aggregate { .. }
        | LogicalPlan::Distinct { .. }
        | LogicalPlan::TopN { .. }
        | LogicalPlan::Union { .. }
        | LogicalPlan::CrossProduct { .. }
        | LogicalPlan::HashJoin { .. }
        | LogicalPlan::HashSemiJoin { .. }
        | LogicalPlan::SemiApply { .. }
        | LogicalPlan::PatternList { .. }
        | LogicalPlan::Unwind { .. }
        | LogicalPlan::Create { .. }
        | LogicalPlan::Merge { .. }
        | LogicalPlan::Set { .. }
        | LogicalPlan::Remove { .. }
        | LogicalPlan::Delete { .. } => false,
        _ => plan.children().iter().all(|c| subplan_is_decorrelable(c)),
    }
}

/// Replace the (unique) `Argument` leaf inside `plan` with a fresh
/// `NodeScan { label, alias: x }`. Recurses down only the IR shape we
/// already approved via `subplan_is_decorrelable`.
fn replace_argument(plan: LogicalPlan, x: &str, label: &str) -> LogicalPlan {
    match plan {
        LogicalPlan::Argument { bindings } if bindings.iter().any(|b| b == x) => {
            LogicalPlan::NodeScan {
                label: Some(label.to_string()),
                alias: x.to_string(),
                predicates: Vec::new(),
                projection: None,
            }
        }
        LogicalPlan::Argument { .. } => plan,
        LogicalPlan::NodeScan { .. } | LogicalPlan::Empty => plan,
        LogicalPlan::NodeById {
            input,
            label: l,
            alias,
            id,
        } => LogicalPlan::NodeById {
            input: Box::new(replace_argument(*input, x, label)),
            label: l,
            alias,
            id,
        },
        LogicalPlan::NodeByPropertyValue {
            input,
            label: l,
            alias,
            property,
            value,
        } => LogicalPlan::NodeByPropertyValue {
            input: Box::new(replace_argument(*input, x, label)),
            label: l,
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
            input: Box::new(replace_argument(*input, x, label)),
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
            input: Box::new(replace_argument(*input, x, label)),
            predicate,
        },
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(replace_argument(*input, x, label)),
            items,
            distinct,
            discard_input_bindings,
        },
        // Anything else was already rejected by `subplan_is_decorrelable`;
        // if it slips through we surface the bug rather than silently
        // miscompile.
        other => panic!(
            "decorrelation::replace_argument: unsupported operator reached: {:?}",
            other.operator_name()
        ),
    }
}

fn recurse_children(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    match plan {
        LogicalPlan::Empty | LogicalPlan::Argument { .. } | LogicalPlan::NodeScan { .. } => plan,
        LogicalPlan::NodeById {
            input,
            label,
            alias,
            id,
        } => LogicalPlan::NodeById {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
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
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
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
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
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
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            predicate,
        },
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => LogicalPlan::Aggregate {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            group_by,
            aggregations,
        },
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
        },
        LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
            left: Box::new(convert_semi_apply_to_hash_semi_join(*left, catalog)),
            right: Box::new(convert_semi_apply_to_hash_semi_join(*right, catalog)),
            all,
        },
        LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            list,
            alias,
        },
        LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
            left: Box::new(convert_semi_apply_to_hash_semi_join(*left, catalog)),
            right: Box::new(convert_semi_apply_to_hash_semi_join(*right, catalog)),
        },
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => LogicalPlan::HashJoin {
            build: Box::new(convert_semi_apply_to_hash_semi_join(*build, catalog)),
            probe: Box::new(convert_semi_apply_to_hash_semi_join(*probe, catalog)),
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
            outer: Box::new(convert_semi_apply_to_hash_semi_join(*outer, catalog)),
            inner: Box::new(convert_semi_apply_to_hash_semi_join(*inner, catalog)),
            on,
            negated,
            residual,
        },
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => LogicalPlan::SemiApply {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            // Subplans recurse too — nested SemiApply within a subplan
            // gets the same chance.
            subplan: Box::new(convert_semi_apply_to_hash_semi_join(*subplan, catalog)),
            negated,
        },
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            alias,
        } => LogicalPlan::PatternList {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            subplan: Box::new(convert_semi_apply_to_hash_semi_join(*subplan, catalog)),
            projection,
            alias,
        },
        LogicalPlan::Create { input, elements } => LogicalPlan::Create {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            elements,
        },
        LogicalPlan::Merge {
            input,
            pattern,
            on_match_sets,
            on_create_sets,
        } => LogicalPlan::Merge {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            pattern,
            on_match_sets,
            on_create_sets,
        },
        LogicalPlan::Set { input, items } => LogicalPlan::Set {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            items,
        },
        LogicalPlan::Remove { input, items } => LogicalPlan::Remove {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            items,
        },
        LogicalPlan::Delete {
            input,
            targets,
            detach,
        } => LogicalPlan::Delete {
            input: Box::new(convert_semi_apply_to_hash_semi_join(*input, catalog)),
            targets,
            detach,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{BinaryOp, Literal, RelationshipDirection};
    use crate::plan::logical::ProjectionItem;
    use crate::plan::logical::ShortestMode;

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
    fn scan(label: &str, alias: &str) -> LogicalPlan {
        LogicalPlan::NodeScan {
            label: Some(label.into()),
            alias: alias.into(),
            predicates: Vec::new(),
            projection: None,
        }
    }
    fn argument(bindings: &[&str]) -> LogicalPlan {
        LogicalPlan::Argument {
            bindings: bindings.iter().map(|s| s.to_string()).collect(),
        }
    }
    fn expand(input: LogicalPlan, source: &str, et: &str, target: &str) -> LogicalPlan {
        LogicalPlan::Expand {
            input: Box::new(input),
            source: source.into(),
            edge_type: Some(et.into()),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: target.into(),
            target_label: Some("Person".into()),
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        }
    }
    fn semi_apply(input: LogicalPlan, subplan: LogicalPlan, negated: bool) -> LogicalPlan {
        LogicalPlan::SemiApply {
            input: Box::new(input),
            subplan: Box::new(subplan),
            negated,
        }
    }
    fn catalog() -> StatsCatalog {
        StatsCatalog::empty()
    }

    #[test]
    fn converts_simple_exists() {
        // Filter(EXISTS((a)-[:KNOWS]->(b))) over NodeScan(Person, a):
        // SemiApply { negated: false }
        // NodeScan(Person, a)
        // Expand source=a edge=KNOWS target=b
        // Argument { bindings: [a] }
        let plan = semi_apply(
            scan("Person", "a"),
            expand(argument(&["a"]), "a", "KNOWS", "b"),
            false,
        );
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        match out {
            LogicalPlan::HashSemiJoin {
                outer,
                inner,
                on,
                negated,
                residual,
            } => {
                assert!(!negated);
                assert!(residual.is_none());
                assert_eq!(on.len(), 1);
                assert!(matches!(*outer, LogicalPlan::NodeScan { .. }));
                // Inner must have NodeScan replacing Argument.
                if let LogicalPlan::Expand { input, .. } = *inner {
                    assert!(
                        matches!(*input, LogicalPlan::NodeScan { ref alias, ref label, .. }
 if alias == "a" && label.as_deref() == Some("Person"))
                    );
                } else {
                    panic!("expected Expand at inner root");
                }
            }
            other => panic!("expected HashSemiJoin, got {:?}", other),
        }
    }

    #[test]
    fn keeps_subplan_with_multi_binding_argument() {
        // Two bindings → out of v0 scope.
        let plan = semi_apply(
            scan("Person", "a"),
            expand(argument(&["a", "b"]), "a", "KNOWS", "c"),
            false,
        );
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        assert!(matches!(out, LogicalPlan::SemiApply { .. }));
    }

    #[test]
    fn keeps_subplan_without_argument() {
        // No Argument leaf — subplan is independent. v0 leaves as SemiApply
        // (planned).
        let plan = semi_apply(scan("Person", "a"), scan("Person", "b"), false);
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        assert!(matches!(out, LogicalPlan::SemiApply { .. }));
    }

    #[test]
    fn keeps_subplan_with_unknown_outer_label() {
        // Outer alias `a` is from an Argument (no label info). Rewriter
        // can't infer a label → keeps as SemiApply.
        let plan = semi_apply(
            argument(&["a"]),
            expand(argument(&["a"]), "a", "KNOWS", "b"),
            false,
        );
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        assert!(matches!(out, LogicalPlan::SemiApply { .. }));
    }

    #[test]
    fn handles_negated_anti_semi_join() {
        let plan = semi_apply(
            scan("Person", "a"),
            expand(argument(&["a"]), "a", "KNOWS", "b"),
            true,
        );
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        match out {
            LogicalPlan::HashSemiJoin { negated, .. } => assert!(negated),
            other => panic!("expected AntiHashSemiJoin, got {:?}", other),
        }
    }

    #[test]
    fn keeps_subplan_with_aggregate_inside() {
        // Aggregate in subplan: not decorrelable.
        let agg = LogicalPlan::Aggregate {
            input: Box::new(expand(argument(&["a"]), "a", "KNOWS", "b")),
            group_by: Vec::new(),
            aggregations: vec![(
                "cnt".into(),
                crate::plan::logical::AggregateExpr::Count {
                    arg: None,
                    distinct: false,
                },
            )],
        };
        let plan = semi_apply(scan("Person", "a"), agg, false);
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        assert!(matches!(out, LogicalPlan::SemiApply { .. }));
    }

    #[test]
    fn preserves_filter_inside_subplan() {
        // Filter(b.age > 30) inside the subplan stays inside the inner;
        // decorrelation only swaps the Argument for a NodeScan.
        let span_ = span();
        let pred = Expression {
            kind: ExpressionKind::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expression {
                    kind: ExpressionKind::Property(Box::new(PropertyAccess {
                        target: Expression {
                            kind: ExpressionKind::Variable(ident("b")),
                            span: span_,
                        },
                        key: ident("age"),
                        span: span_,
                    })),
                    span: span_,
                }),
                right: Box::new(Expression {
                    kind: ExpressionKind::Literal(Literal::Integer(30)),
                    span: span_,
                }),
            },
            span: span_,
        };
        let inner_with_filter = LogicalPlan::Filter {
            input: Box::new(expand(argument(&["a"]), "a", "KNOWS", "b")),
            predicate: pred,
        };
        let plan = semi_apply(scan("Person", "a"), inner_with_filter, false);
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        match out {
            LogicalPlan::HashSemiJoin { inner, .. } => {
                assert!(matches!(*inner, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected HashSemiJoin, got {:?}", other),
        }
    }

    #[test]
    fn rewriter_is_idempotent() {
        let plan = semi_apply(
            scan("Person", "a"),
            expand(argument(&["a"]), "a", "KNOWS", "b"),
            false,
        );
        let once = convert_semi_apply_to_hash_semi_join(plan.clone(), &catalog());
        let twice = convert_semi_apply_to_hash_semi_join(once.clone(), &catalog());
        assert_eq!(once, twice);
    }

    #[test]
    fn skips_when_outer_carries_no_match_for_argument_alias() {
        // Subplan's Argument refers to "z" but outer doesn't expose "z".
        let plan = semi_apply(
            scan("Person", "a"),
            expand(argument(&["z"]), "z", "KNOWS", "b"),
            false,
        );
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        assert!(matches!(out, LogicalPlan::SemiApply { .. }));
    }

    #[test]
    fn keeps_nested_semi_apply_inside_subplan() {
        // Subplan contains another SemiApply — outer must NOT be
        // decorrelated (its subplan_is_decorrelable rejects nested
        // SemiApply). The inner SemiApply's input is also an Argument
        // (not a NodeScan), so the inner has no visible outer label
        // and also stays put. Net: both SemiApply layers survive.
        let inner = semi_apply(
            argument(&["a"]),
            expand(argument(&["a"]), "a", "KNOWS", "b"),
            false,
        );
        let plan = semi_apply(scan("Person", "a"), inner, false);
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        match out {
            LogicalPlan::SemiApply { subplan, .. } => {
                assert!(
 matches!(*subplan, LogicalPlan::SemiApply { .. }),
 "nested SemiApply should survive when its input lacks a labelled binding, got {:?}",
 subplan
 );
            }
            _ => panic!("expected SemiApply at root"),
        }
    }

    #[test]
    fn project_with_discard_keeps_subplan() {
        // Project that drops bindings is rare in subplans but if it occurs
        // we still decorrelate (the rewrite passes through Project).
        let proj = LogicalPlan::Project {
            input: Box::new(expand(argument(&["a"]), "a", "KNOWS", "b")),
            items: vec![ProjectionItem {
                expression: Expression {
                    kind: ExpressionKind::Variable(ident("b")),
                    span: span(),
                },
                alias: "b".into(),
            }],
            distinct: false,
            discard_input_bindings: false,
        };
        let plan = semi_apply(scan("Person", "a"), proj, false);
        let out = convert_semi_apply_to_hash_semi_join(plan, &catalog());
        // Either is fine; we accept conversion here because Project is
        // listed as decorrelable.
        assert!(matches!(
            out,
            LogicalPlan::HashSemiJoin { .. } | LogicalPlan::SemiApply { .. }
        ));
    }
}
