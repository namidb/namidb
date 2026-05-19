//! Rewrite `Filter(predicate = a.<prop> == <literal>)` over
//! `NodeScan(label = L)` into `NodeByPropertyValue` whenever `<prop>`
//! is declared `unique` in the schema (see `PropertyDef::unique`).
//!
//! Triggers a point lookup in place of a full label scan + filter for
//! the LDBC SNB anchor pattern `MATCH (p:Person {id: 'literal'})`,
//! which is the highest-leverage cold-start fix from STATUS.md
//! §"Bench A SF1" (cold IC09 went from 9 s to ~150 ms with the same
//! optimisation in Kuzu / Neo4j range indexes).
//!
//! The pass is conservative: it only fires when
//! 1. The Filter's predicate is exactly `Property(a, <prop>) == <literal>`
//!    (single conjunct; multiple conjuncts fall through unchanged —
//!    a follow-up could combine the unique lookup with residual filters).
//! 2. The immediate child is `NodeScan { label: Some(L), predicates: [],
//!    projection: None }` — no other pushed predicates or projections
//!    (we want the cheapest possible rewrite to avoid composing with
//!    later passes).
//! 3. The catalog reports `props[prop].unique == true` for `(L, prop)`.
//!
//! All other shapes pass through. Runs once per fixpoint iteration like
//! the other rewrites in `optimize::mod`.

use crate::cost::StatsCatalog;
use crate::parser::ast::{Expression, ExpressionKind};
use crate::plan::LogicalPlan;

/// Run the unique-property lookup rewrite over `plan`.
pub fn apply_unique_property_lookup(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    rewrite(plan, catalog)
}

fn rewrite(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    match plan {
        // The pattern we care about: `Filter(eq-on-prop, NodeScan(label))`.
        LogicalPlan::Filter { input, predicate } => {
            // Try to match the rewrite pattern.
            if let LogicalPlan::NodeScan {
                label: Some(label),
                alias,
                predicates,
                projection,
            } = input.as_ref()
            {
                if predicates.is_empty() && projection.is_none() {
                    if let Some((prop, value_expr)) =
                        extract_eq_on_prop(&predicate, alias)
                    {
                        if catalog
                            .label(label)
                            .and_then(|l| l.properties.get(&prop))
                            .map(|p| p.unique)
                            .unwrap_or(false)
                        {
                            return LogicalPlan::NodeByPropertyValue {
                                input: Box::new(LogicalPlan::Empty),
                                label: label.clone(),
                                alias: alias.clone(),
                                property: prop,
                                value: value_expr,
                            };
                        }
                    }
                }
            }
            // No match — recurse into the input and keep the Filter.
            LogicalPlan::Filter {
                input: Box::new(rewrite(*input, catalog)),
                predicate,
            }
        }

        // Recurse on every other operator (mechanical).
        LogicalPlan::Empty | LogicalPlan::Argument { .. } | LogicalPlan::NodeScan { .. } => plan,
        LogicalPlan::NodeById {
            input,
            label,
            alias,
            id,
        } => LogicalPlan::NodeById {
            input: Box::new(rewrite(*input, catalog)),
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
            input: Box::new(rewrite(*input, catalog)),
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
            input: Box::new(rewrite(*input, catalog)),
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
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(rewrite(*input, catalog)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => LogicalPlan::Aggregate {
            input: Box::new(rewrite(*input, catalog)),
            group_by,
            aggregations,
        },
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(rewrite(*input, catalog)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(rewrite(*input, catalog)),
        },
        LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
            input: Box::new(rewrite(*input, catalog)),
            list,
            alias,
        },
        LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
            left: Box::new(rewrite(*left, catalog)),
            right: Box::new(rewrite(*right, catalog)),
            all,
        },
        LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
            left: Box::new(rewrite(*left, catalog)),
            right: Box::new(rewrite(*right, catalog)),
        },
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => LogicalPlan::HashJoin {
            build: Box::new(rewrite(*build, catalog)),
            probe: Box::new(rewrite(*probe, catalog)),
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
            outer: Box::new(rewrite(*outer, catalog)),
            inner: Box::new(rewrite(*inner, catalog)),
            on,
            negated,
            residual,
        },
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => LogicalPlan::SemiApply {
            input: Box::new(rewrite(*input, catalog)),
            subplan: Box::new(rewrite(*subplan, catalog)),
            negated,
        },
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            alias,
        } => LogicalPlan::PatternList {
            input: Box::new(rewrite(*input, catalog)),
            subplan: Box::new(rewrite(*subplan, catalog)),
            projection,
            alias,
        },
        // Write operators — recurse on their child but never rewrite
        // the write itself (no read planning around them).
        LogicalPlan::Create { input, elements } => LogicalPlan::Create {
            input: Box::new(rewrite(*input, catalog)),
            elements,
        },
        LogicalPlan::Merge {
            input,
            pattern,
            on_match_sets,
            on_create_sets,
        } => LogicalPlan::Merge {
            input: Box::new(rewrite(*input, catalog)),
            pattern,
            on_match_sets,
            on_create_sets,
        },
        LogicalPlan::Set { input, items } => LogicalPlan::Set {
            input: Box::new(rewrite(*input, catalog)),
            items,
        },
        LogicalPlan::Remove { input, items } => LogicalPlan::Remove {
            input: Box::new(rewrite(*input, catalog)),
            items,
        },
        LogicalPlan::Delete {
            input,
            targets,
            detach,
        } => LogicalPlan::Delete {
            input: Box::new(rewrite(*input, catalog)),
            targets,
            detach,
        },
    }
}

/// If `expr` is exactly `alias.<prop> == <literal-ish>`, return
/// `(prop, value_expr)`. Otherwise `None`.
fn extract_eq_on_prop(expr: &Expression, scan_alias: &str) -> Option<(String, Expression)> {
    let ExpressionKind::Binary { op, left, right } = &expr.kind else {
        return None;
    };
    use crate::parser::ast::BinaryOp;
    if !matches!(op, BinaryOp::Eq) {
        return None;
    }
    // Either side can carry the property access; the other is the value.
    if let Some(prop) = property_on_alias(left, scan_alias) {
        return Some((prop, (**right).clone()));
    }
    if let Some(prop) = property_on_alias(right, scan_alias) {
        return Some((prop, (**left).clone()));
    }
    None
}

fn property_on_alias(expr: &Expression, alias: &str) -> Option<String> {
    let ExpressionKind::Property(pa) = &expr.kind else {
        return None;
    };
    let ExpressionKind::Variable(id) = &pa.target.kind else {
        return None;
    };
    if id.name == alias {
        Some(pa.key.name.clone())
    } else {
        None
    }
}
