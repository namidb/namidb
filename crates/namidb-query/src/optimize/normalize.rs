//! Filter-tree normalization (RFC-011 §3).
//!
//! Bottom-up rewrite that tidies the shape of stacked `Filter` nodes
//! produced by [`predicate_pushdown`] and by the lowering itself:
//!
//! 1. Merge adjacent `Filter`s into a single AND.
//! 2. Drop `Filter(literal true)`.
//! 3. Drop the synthetic `__label_eq(target, "Label")` filter immediately
//! over an `Expand` whose `target_label` matches.

use super::{and_chain, is_synthetic_label_eq, split_and_terms};
use crate::parser::ast::{Expression, ExpressionKind, Literal};
use crate::plan::logical::LogicalPlan;

/// Apply the normalization rules to every node of `plan`. Bottom-up.
pub fn normalize_filters(plan: LogicalPlan) -> LogicalPlan {
    let recursed = recurse_children(plan);
    apply_local_rules(recursed)
}

fn recurse_children(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::NodeScan { .. } | LogicalPlan::Empty | LogicalPlan::Argument { .. } => plan,
        LogicalPlan::NodeById {
            input,
            label,
            alias,
            id,
        } => LogicalPlan::NodeById {
            input: Box::new(normalize_filters(*input)),
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
            input: Box::new(normalize_filters(*input)),
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
            input: Box::new(normalize_filters(*input)),
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
            input: Box::new(normalize_filters(*input)),
            predicate,
        },
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(normalize_filters(*input)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => LogicalPlan::Aggregate {
            input: Box::new(normalize_filters(*input)),
            group_by,
            aggregations,
        },
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(normalize_filters(*input)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(normalize_filters(*input)),
        },
        LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
            left: Box::new(normalize_filters(*left)),
            right: Box::new(normalize_filters(*right)),
            all,
        },
        LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
            input: Box::new(normalize_filters(*input)),
            list,
            alias,
        },
        LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
            left: Box::new(normalize_filters(*left)),
            right: Box::new(normalize_filters(*right)),
        },
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => LogicalPlan::HashJoin {
            build: Box::new(normalize_filters(*build)),
            probe: Box::new(normalize_filters(*probe)),
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
            outer: Box::new(normalize_filters(*outer)),
            inner: Box::new(normalize_filters(*inner)),
            on,
            negated,
            residual,
        },
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => LogicalPlan::SemiApply {
            input: Box::new(normalize_filters(*input)),
            // Subplans are not visited normalization (separate scope).
            subplan,
            negated,
        },
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            alias,
        } => LogicalPlan::PatternList {
            input: Box::new(normalize_filters(*input)),
            subplan,
            projection,
            alias,
        },
        LogicalPlan::Create { input, elements } => LogicalPlan::Create {
            input: Box::new(normalize_filters(*input)),
            elements,
        },
        LogicalPlan::Merge {
            input,
            pattern,
            on_match_sets,
            on_create_sets,
        } => LogicalPlan::Merge {
            input: Box::new(normalize_filters(*input)),
            pattern,
            on_match_sets,
            on_create_sets,
        },
        LogicalPlan::Set { input, items } => LogicalPlan::Set {
            input: Box::new(normalize_filters(*input)),
            items,
        },
        LogicalPlan::Remove { input, items } => LogicalPlan::Remove {
            input: Box::new(normalize_filters(*input)),
            items,
        },
        LogicalPlan::Delete {
            input,
            targets,
            detach,
        } => LogicalPlan::Delete {
            input: Box::new(normalize_filters(*input)),
            targets,
            detach,
        },
    }
}

fn apply_local_rules(plan: LogicalPlan) -> LogicalPlan {
    if let LogicalPlan::Filter { input, predicate } = plan {
        return apply_filter_rules(*input, predicate);
    }
    plan
}

fn apply_filter_rules(input: LogicalPlan, predicate: Expression) -> LogicalPlan {
    // Rule 3: drop literal true.
    if matches!(
        predicate.kind,
        ExpressionKind::Literal(Literal::Boolean(true))
    ) {
        return input;
    }
    // Rule 4: drop synthetic __label_eq when the child Expand declares the
    // same target_label.
    if let LogicalPlan::Expand {
        target_alias,
        target_label: Some(label),
        ..
    } = &input
    {
        if is_synthetic_label_eq(&predicate, target_alias, label) {
            return input;
        }
    }
    // Rule 2: merge adjacent Filters.
    if let LogicalPlan::Filter {
        input: inner_input,
        predicate: inner_predicate,
    } = input
    {
        let mut terms = split_and_terms(&inner_predicate);
        terms.extend(split_and_terms(&predicate));
        let combined = and_chain(terms).expect("at least one term present");
        // After merging, recheck rule 3 (combined may still be literal true).
        if matches!(
            combined.kind,
            ExpressionKind::Literal(Literal::Boolean(true))
        ) {
            return *inner_input;
        }
        return LogicalPlan::Filter {
            input: inner_input,
            predicate: combined,
        };
    }
    LogicalPlan::Filter {
        input: Box::new(input),
        predicate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{BinaryOp, Identifier, QualifiedName, RelationshipDirection};
    use crate::parser::SourceSpan;
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

    fn scan(label: &str, alias: &str) -> LogicalPlan {
        LogicalPlan::NodeScan {
            label: Some(label.into()),
            alias: alias.into(),
            predicates: vec![],
            projection: None,
        }
    }

    #[test]
    fn drops_filter_literal_true() {
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("Person", "a")),
            predicate: lit_true(),
        };
        let result = normalize_filters(plan);
        assert!(matches!(result, LogicalPlan::NodeScan { .. }));
    }

    #[test]
    fn merges_adjacent_filters_into_and() {
        let p1 = binop(BinaryOp::Gt, prop("a", "age"), int(30));
        let p2 = binop(BinaryOp::Eq, prop("a", "name"), int(1));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan("Person", "a")),
                predicate: p1,
            }),
            predicate: p2,
        };
        let result = normalize_filters(plan);
        // Single Filter at the root with two AND'd conjuncts.
        match result {
            LogicalPlan::Filter { input, predicate } => {
                assert!(matches!(*input, LogicalPlan::NodeScan { .. }));
                assert_eq!(split_and_terms(&predicate).len(), 2);
            }
            other => panic!("expected Filter, got {:?}", other),
        }
    }

    #[test]
    fn merges_three_stacked_filters() {
        let p1 = binop(BinaryOp::Gt, prop("a", "x"), int(1));
        let p2 = binop(BinaryOp::Gt, prop("a", "y"), int(1));
        let p3 = binop(BinaryOp::Gt, prop("a", "z"), int(1));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(LogicalPlan::Filter {
                    input: Box::new(scan("Person", "a")),
                    predicate: p1,
                }),
                predicate: p2,
            }),
            predicate: p3,
        };
        let result = normalize_filters(plan);
        match result {
            LogicalPlan::Filter { predicate, input } => {
                assert!(matches!(*input, LogicalPlan::NodeScan { .. }));
                assert_eq!(split_and_terms(&predicate).len(), 3);
            }
            other => panic!("expected Filter, got {:?}", other),
        }
    }

    #[test]
    fn drops_synthetic_label_eq_over_expand() {
        let label_eq = Expression {
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
        let expand = LogicalPlan::Expand {
            input: Box::new(scan("Person", "a")),
            source: "a".into(),
            edge_type: Some("KNOWS".into()),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: Some("Person".into()),
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(expand),
            predicate: label_eq,
        };
        let result = normalize_filters(plan);
        assert!(matches!(result, LogicalPlan::Expand { .. }));
    }

    #[test]
    fn keeps_label_eq_when_alias_mismatches() {
        // Label-eq targets `c` but Expand introduces `b` → not removed.
        let label_eq = Expression {
            kind: ExpressionKind::FunctionCall {
                name: QualifiedName::single(Identifier::new("__label_eq", span())),
                args: vec![
                    var("c"),
                    Expression {
                        kind: ExpressionKind::Literal(Literal::String("Person".into())),
                        span: span(),
                    },
                ],
                distinct: false,
            },
            span: span(),
        };
        let expand = LogicalPlan::Expand {
            input: Box::new(scan("Person", "a")),
            source: "a".into(),
            edge_type: Some("KNOWS".into()),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: Some("Person".into()),
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(expand),
            predicate: label_eq,
        };
        let result = normalize_filters(plan);
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn keeps_label_eq_when_target_label_missing() {
        // Expand has target_label=None → can't drop the defensive filter.
        let label_eq = Expression {
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
        let expand = LogicalPlan::Expand {
            input: Box::new(scan("Person", "a")),
            source: "a".into(),
            edge_type: Some("KNOWS".into()),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: None,
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(expand),
            predicate: label_eq,
        };
        let result = normalize_filters(plan);
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn recurses_into_children() {
        // Filter(true) under an Expand should be dropped even if it's
        // not at the root.
        let inner_filter = LogicalPlan::Filter {
            input: Box::new(scan("Person", "a")),
            predicate: lit_true(),
        };
        let plan = LogicalPlan::Expand {
            input: Box::new(inner_filter),
            source: "a".into(),
            edge_type: Some("KNOWS".into()),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: None,
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let result = normalize_filters(plan);
        match result {
            LogicalPlan::Expand { input, .. } => {
                assert!(matches!(*input, LogicalPlan::NodeScan { .. }));
            }
            other => panic!("expected Expand, got {:?}", other),
        }
    }

    #[test]
    fn idempotent() {
        let p1 = binop(BinaryOp::Gt, prop("a", "x"), int(1));
        let p2 = binop(BinaryOp::Gt, prop("a", "y"), int(1));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan("Person", "a")),
                predicate: p1,
            }),
            predicate: p2,
        };
        let once = normalize_filters(plan);
        let twice = normalize_filters(once.clone());
        assert_eq!(once, twice);
    }

    #[test]
    fn merged_filter_with_redundant_true_collapses() {
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan("Person", "a")),
                predicate: lit_true(),
            }),
            predicate: lit_true(),
        };
        let result = normalize_filters(plan);
        assert!(matches!(result, LogicalPlan::NodeScan { .. }));
    }

    #[test]
    fn drops_label_eq_after_merge() {
        // Filter(__label_eq) over Filter(real predicate) over Expand{
        // target_label=Person, target_alias=b}. After merge we'd get a
        // single Filter with both conjuncts — the __label_eq does NOT
        // get individually pruned, because rule 2 (merge) ran first.
        // That's expected v0: rule 4 only applies to the *exact*
        // synthetic predicate alone above an Expand.
        let label_eq = Expression {
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
        let real_pred = binop(BinaryOp::Eq, prop("b", "name"), int(1));
        let expand = LogicalPlan::Expand {
            input: Box::new(scan("Person", "a")),
            source: "a".into(),
            edge_type: Some("KNOWS".into()),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: Some("Person".into()),
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(expand),
                predicate: label_eq,
            }),
            predicate: real_pred,
        };
        let result = normalize_filters(plan);
        // After normalization we expect a single Filter (combined) over
        // the Expand. The combined predicate still has the __label_eq
        // call (rule 4 doesn't dissect inside compound predicates).
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }
}
