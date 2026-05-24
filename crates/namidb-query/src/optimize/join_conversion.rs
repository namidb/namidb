//! Convert `Filter(eq cross-side) ⇒ CrossProduct(L, R)` to `HashJoin`
//! (RFC-012).
//!
//! Bottom-up rewrite that fires after [`predicate_pushdown`] and
//! [`normalize_filters`]. By the time we see a `Filter` whose immediate
//! child is `CrossProduct`, every predicate that could have been pushed
//! to one side already has been; what remains above the cross product
//! is either a cross-side equality (our trigger) or a non-equality
//! mixer that we keep as `residual` on the resulting HashJoin.
//!
//! [`predicate_pushdown`]: super::pushdown::predicate_pushdown
//! [`normalize_filters`]: super::normalize::normalize_filters

use std::collections::BTreeSet;

use super::{and_chain, expression_aliases, produced_aliases, split_and_terms};
use crate::cost::{estimate, StatsCatalog};
use crate::parser::ast::{BinaryOp, Expression, ExpressionKind};
use crate::plan::logical::{JoinKey, LogicalPlan};

/// Rewriter: convert every reachable `Filter(eq cross-side) ⇒
/// CrossProduct` shape into a `HashJoin`. Recurses through every
/// operator. Idempotent — running it twice on the same plan yields
/// the same tree.
pub fn convert_cross_to_hash(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    let recursed = recurse_children(plan, catalog);
    try_convert_at_root(recursed, catalog)
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
            input: Box::new(convert_cross_to_hash(*input, catalog)),
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
            input: Box::new(convert_cross_to_hash(*input, catalog)),
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
            input: Box::new(convert_cross_to_hash(*input, catalog)),
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
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            predicate,
        },
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => LogicalPlan::Aggregate {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            group_by,
            aggregations,
        },
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
        },
        LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
            left: Box::new(convert_cross_to_hash(*left, catalog)),
            right: Box::new(convert_cross_to_hash(*right, catalog)),
            all,
        },
        LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            list,
            alias,
        },
        LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
            left: Box::new(convert_cross_to_hash(*left, catalog)),
            right: Box::new(convert_cross_to_hash(*right, catalog)),
        },
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => LogicalPlan::SemiApply {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            // Subplans are not visited — same scope rule as in
            // `predicate_pushdown` and `normalize_filters`.
            subplan,
            negated,
        },
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            alias,
        } => LogicalPlan::PatternList {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            subplan,
            projection,
            alias,
        },
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => LogicalPlan::HashJoin {
            build: Box::new(convert_cross_to_hash(*build, catalog)),
            probe: Box::new(convert_cross_to_hash(*probe, catalog)),
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
            outer: Box::new(convert_cross_to_hash(*outer, catalog)),
            inner: Box::new(convert_cross_to_hash(*inner, catalog)),
            on,
            negated,
            residual,
        },
        LogicalPlan::Create { input, elements } => LogicalPlan::Create {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            elements,
        },
        LogicalPlan::Merge {
            input,
            pattern,
            on_match_sets,
            on_create_sets,
        } => LogicalPlan::Merge {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            pattern,
            on_match_sets,
            on_create_sets,
        },
        LogicalPlan::Set { input, items } => LogicalPlan::Set {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            items,
        },
        LogicalPlan::Remove { input, items } => LogicalPlan::Remove {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            items,
        },
        LogicalPlan::Delete {
            input,
            targets,
            detach,
        } => LogicalPlan::Delete {
            input: Box::new(convert_cross_to_hash(*input, catalog)),
            targets,
            detach,
        },
    }
}

fn try_convert_at_root(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    let LogicalPlan::Filter { input, predicate } = plan else {
        return plan;
    };
    let LogicalPlan::CrossProduct { left, right } = *input else {
        // Restore the Filter we just opened.
        return LogicalPlan::Filter { input, predicate };
    };

    let left_aliases = produced_aliases(&left);
    let right_aliases = produced_aliases(&right);

    let conjuncts = split_and_terms(&predicate);
    let mut left_to_right_keys: Vec<(Expression, Expression)> = Vec::new();
    let mut residual_terms: Vec<Expression> = Vec::new();

    for term in conjuncts {
        match classify_conjunct(&term, &left_aliases, &right_aliases) {
            ConjunctClass::LeftEqRight { lhs, rhs } => {
                left_to_right_keys.push((lhs, rhs));
            }
            ConjunctClass::RightEqLeft { lhs, rhs } => {
                // Canonicalise so the left subtree's expression sits in
                // the first slot of the pair. We swap.
                left_to_right_keys.push((rhs, lhs));
            }
            ConjunctClass::Residual => residual_terms.push(term),
        }
    }

    if left_to_right_keys.is_empty() {
        // No cross-side equality → no conversion. Rebuild original shape.
        return LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct { left, right }),
            predicate,
        };
    }

    // Build vs probe decision: smaller side becomes `build` so the hash
    // table is compact. The `on` pairs were canonicalised as
    // (left_expr, right_expr); if left is chosen as build we keep them
    // as-is; if right is chosen as build, we swap each pair so
    // `build_side` is the right expression.
    let l_est = estimate(&left, catalog).rows;
    let r_est = estimate(&right, catalog).rows;
    let pick_left_as_build = l_est <= r_est; // deterministic tie-break.
    let (build, probe, on) = if pick_left_as_build {
        let on = left_to_right_keys
            .into_iter()
            .map(|(l, r)| JoinKey {
                build_side: l,
                probe_side: r,
            })
            .collect();
        (left, right, on)
    } else {
        let on = left_to_right_keys
            .into_iter()
            .map(|(l, r)| JoinKey {
                build_side: r,
                probe_side: l,
            })
            .collect();
        (right, left, on)
    };

    let residual = and_chain(residual_terms);

    LogicalPlan::HashJoin {
        build,
        probe,
        on,
        residual,
    }
}

enum ConjunctClass {
    /// Equality whose LHS aliases ⊆ left subtree, RHS ⊆ right subtree.
    LeftEqRight { lhs: Expression, rhs: Expression },
    /// Mirror: LHS ⊆ right, RHS ⊆ left.
    RightEqLeft { lhs: Expression, rhs: Expression },
    /// Anything else stays as a residual predicate above the HashJoin.
    Residual,
}

fn classify_conjunct(
    term: &Expression,
    left: &BTreeSet<String>,
    right: &BTreeSet<String>,
) -> ConjunctClass {
    let ExpressionKind::Binary {
        op: BinaryOp::Eq,
        left: lhs,
        right: rhs,
    } = &term.kind
    else {
        return ConjunctClass::Residual;
    };
    let lhs_aliases = expression_aliases(lhs);
    let rhs_aliases = expression_aliases(rhs);
    // Both sides must be non-empty *and* sit entirely on one subtree.
    let l_on_left = !lhs_aliases.is_empty() && lhs_aliases.is_subset(left);
    let r_on_right = !rhs_aliases.is_empty() && rhs_aliases.is_subset(right);
    let l_on_right = !lhs_aliases.is_empty() && lhs_aliases.is_subset(right);
    let r_on_left = !rhs_aliases.is_empty() && rhs_aliases.is_subset(left);
    if l_on_left && r_on_right {
        return ConjunctClass::LeftEqRight {
            lhs: lhs.as_ref().clone(),
            rhs: rhs.as_ref().clone(),
        };
    }
    if l_on_right && r_on_left {
        return ConjunctClass::RightEqLeft {
            lhs: lhs.as_ref().clone(),
            rhs: rhs.as_ref().clone(),
        };
    }
    ConjunctClass::Residual
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::stats::{EdgeTypeStats, LabelStats, PropStats};
    use crate::parser::ast::{Identifier, Literal, PropertyAccess};
    use crate::parser::SourceSpan;
    use crate::plan::logical::ProjectionItem;
        use namidb_storage::sst::stats::StatScalar;
    use std::collections::BTreeMap;

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
            kind: ExpressionKind::Property(Box::new(PropertyAccess {
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

    fn catalog_with_ndv(person_rows: u64, ndv: u64) -> StatsCatalog {
        let mut cat = StatsCatalog::empty();
        let mut props = BTreeMap::new();
        props.insert(
            "firstName".to_string(),
            PropStats {
                null_count: 0,
                non_null_count: person_rows,
                min: Some(StatScalar::Utf8("Alice".into())),
                max: Some(StatScalar::Utf8("Zoe".into())),
                ndv: Some(ndv),
                unique: false,
            },
        );
        props.insert(
            "age".to_string(),
            PropStats {
                null_count: 0,
                non_null_count: person_rows,
                min: Some(StatScalar::Int64(18)),
                max: Some(StatScalar::Int64(99)),
                ndv: Some(ndv),
                unique: false,
            },
        );
        cat.__test_insert_label(LabelStats {
            name: "Person".into(),
            node_count: person_rows,
            properties: props,
        });
        cat.__test_insert_edge_type(EdgeTypeStats {
            name: "KNOWS".into(),
            edge_count: 0,
            avg_out_degree: 0.0,
            avg_in_degree: 0.0,
            max_out_degree: 0,
            max_in_degree: 0,
            src_label: Some("Person".into()),
            dst_label: Some("Person".into()),
        });
        cat
    }

    #[test]
    fn single_cross_side_equality_converts() {
        let cat = catalog_with_ndv(100, 50);
        let pred = binop(BinaryOp::Eq, prop("a", "firstName"), prop("b", "firstName"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let optimized = convert_cross_to_hash(plan, &cat);
        match optimized {
            LogicalPlan::HashJoin { on, residual, .. } => {
                assert_eq!(on.len(), 1);
                assert!(residual.is_none());
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn no_eq_no_conversion() {
        let cat = catalog_with_ndv(100, 50);
        // No equality — `WHERE a.age > b.age` stays as Filter ⇒ CrossProduct.
        let pred = binop(BinaryOp::Gt, prop("a", "age"), prop("b", "age"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let result = convert_cross_to_hash(plan, &cat);
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn same_side_equality_stays_as_residual_if_no_cross_eq() {
        let cat = catalog_with_ndv(100, 50);
        // `WHERE a.age = a.firstName` is degenerate but legal — same side.
        // No cross-side eq → no conversion.
        let pred = binop(BinaryOp::Eq, prop("a", "age"), prop("a", "firstName"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let result = convert_cross_to_hash(plan, &cat);
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn cross_eq_with_extra_residual_keeps_residual_on_hash_join() {
        let cat = catalog_with_ndv(100, 50);
        // a.firstName = b.firstName AND a.age > b.age
        let eq = binop(BinaryOp::Eq, prop("a", "firstName"), prop("b", "firstName"));
        let gt = binop(BinaryOp::Gt, prop("a", "age"), prop("b", "age"));
        let pred = binop(BinaryOp::And, eq, gt);
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let optimized = convert_cross_to_hash(plan, &cat);
        match optimized {
            LogicalPlan::HashJoin { on, residual, .. } => {
                assert_eq!(on.len(), 1);
                assert!(residual.is_some());
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn multi_eq_coalesces_to_multi_key_hash_join() {
        let cat = catalog_with_ndv(100, 50);
        // a.firstName = b.firstName AND a.age = b.age
        let eq1 = binop(BinaryOp::Eq, prop("a", "firstName"), prop("b", "firstName"));
        let eq2 = binop(BinaryOp::Eq, prop("a", "age"), prop("b", "age"));
        let pred = binop(BinaryOp::And, eq1, eq2);
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let optimized = convert_cross_to_hash(plan, &cat);
        match optimized {
            LogicalPlan::HashJoin { on, residual, .. } => {
                assert_eq!(on.len(), 2);
                assert!(residual.is_none());
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn mirrored_equality_canonicalises_build_side() {
        let cat = catalog_with_ndv(100, 50);
        // Note: the equality is written with `b` first, `a` second —
        // the rewriter must still detect it as cross-side and emit a
        // HashJoin whose build_side aliases lie in the build subtree.
        let pred = binop(BinaryOp::Eq, prop("b", "firstName"), prop("a", "firstName"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let optimized = convert_cross_to_hash(plan, &cat);
        match optimized {
            LogicalPlan::HashJoin { on, .. } => {
                let key = &on[0];
                // The build subtree (smaller side; here L=R so left is
                // picked deterministically) introduces `a`. So
                // build_side must reference `a`.
                let bs_aliases = expression_aliases(&key.build_side);
                assert!(bs_aliases.contains("a"));
                let ps_aliases = expression_aliases(&key.probe_side);
                assert!(ps_aliases.contains("b"));
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn build_side_is_the_smaller_of_the_two() {
        // Build a catalog where left scans a label of 10 nodes and
        // right scans a label of 1000 nodes. The build side should be
        // the 10-node one.
        let mut cat = StatsCatalog::empty();
        cat.__test_insert_label(LabelStats {
            name: "Small".into(),
            node_count: 10,
            properties: {
                let mut m = BTreeMap::new();
                m.insert(
                    "k".into(),
                    PropStats {
                        null_count: 0,
                        non_null_count: 10,
                        min: None,
                        max: None,
                        ndv: Some(10),
                        unique: false,
                    },
                );
                m
            },
        });
        cat.__test_insert_label(LabelStats {
            name: "Big".into(),
            node_count: 1000,
            properties: {
                let mut m = BTreeMap::new();
                m.insert(
                    "k".into(),
                    PropStats {
                        null_count: 0,
                        non_null_count: 1000,
                        min: None,
                        max: None,
                        ndv: Some(1000),
                        unique: false,
                    },
                );
                m
            },
        });
        let pred = binop(BinaryOp::Eq, prop("s", "k"), prop("b", "k"));
        // Left = Big (1000), Right = Small (10) — build should pick Small.
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(LogicalPlan::NodeScan {
                    label: Some("Big".into()),
                    alias: "b".into(),
                    predicates: vec![],
                    projection: None,
                }),
                right: Box::new(LogicalPlan::NodeScan {
                    label: Some("Small".into()),
                    alias: "s".into(),
                    predicates: vec![],
                    projection: None,
                }),
            }),
            predicate: pred,
        };
        let optimized = convert_cross_to_hash(plan, &cat);
        match optimized {
            LogicalPlan::HashJoin { build, .. } => {
                // The chosen build subtree should be the smaller one
                // (`Small`).
                match *build {
                    LogicalPlan::NodeScan { label, .. } => {
                        assert_eq!(label.as_deref(), Some("Small"))
                    }
                    other => panic!("expected NodeScan, got {:?}", other),
                }
            }
            other => panic!("expected HashJoin, got {:?}", other),
        }
    }

    #[test]
    fn no_op_when_already_optimized() {
        let cat = catalog_with_ndv(100, 50);
        // A plan without `Filter ⇒ CrossProduct` is untouched.
        let plan = LogicalPlan::Project {
            input: Box::new(scan("Person", "a")),
            items: vec![ProjectionItem {
                expression: var("a"),
                alias: "a".into(),
            }],
            distinct: false,
            discard_input_bindings: true,
        };
        let optimized = convert_cross_to_hash(plan.clone(), &cat);
        assert_eq!(optimized, plan);
    }

    #[test]
    fn idempotent() {
        let cat = catalog_with_ndv(100, 50);
        let pred = binop(BinaryOp::Eq, prop("a", "firstName"), prop("b", "firstName"));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let once = convert_cross_to_hash(plan, &cat);
        let twice = convert_cross_to_hash(once.clone(), &cat);
        assert_eq!(once, twice);
    }

    #[test]
    fn parameter_only_predicate_is_residual() {
        let cat = catalog_with_ndv(100, 50);
        // `WHERE $p = $q` (no aliases) — not a cross-side eq;
        // expression_aliases returns empty for both sides.
        let p_left = Expression {
            kind: ExpressionKind::Parameter("p".into()),
            span: span(),
        };
        let p_right = Expression {
            kind: ExpressionKind::Parameter("q".into()),
            span: span(),
        };
        let pred = binop(BinaryOp::Eq, p_left, p_right);
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let result = convert_cross_to_hash(plan, &cat);
        // Both sides aliases are empty so neither is a "subset of one
        // side"; classify returns Residual → no conversion.
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn equality_with_literal_is_residual() {
        let cat = catalog_with_ndv(100, 50);
        // `a.x = 5` — literal one side; same-side eq, not cross.
        let pred = binop(BinaryOp::Eq, prop("a", "firstName"), int(5));
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "a")),
                right: Box::new(scan("Person", "b")),
            }),
            predicate: pred,
        };
        let result = convert_cross_to_hash(plan, &cat);
        assert!(matches!(result, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn recurses_into_nested_cross_products() {
        let cat = catalog_with_ndv(100, 50);
        let inner_eq = binop(BinaryOp::Eq, prop("b", "firstName"), prop("c", "firstName"));
        let inner = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::CrossProduct {
                left: Box::new(scan("Person", "b")),
                right: Box::new(scan("Person", "c")),
            }),
            predicate: inner_eq,
        };
        // Wrap in a Project that doesn't touch the cross.
        let plan = LogicalPlan::Project {
            input: Box::new(inner),
            items: vec![ProjectionItem {
                expression: var("b"),
                alias: "b".into(),
            }],
            distinct: false,
            discard_input_bindings: true,
        };
        let optimized = convert_cross_to_hash(plan, &cat);
        match optimized {
            LogicalPlan::Project { input, .. } => {
                assert!(matches!(*input, LogicalPlan::HashJoin { .. }));
            }
            other => panic!("expected Project, got {:?}", other),
        }
    }
}
