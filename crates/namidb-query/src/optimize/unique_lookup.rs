//! Point-lookup rewrites over a labelled node scan, including its correlated
//! `Filter(..., CrossProduct(outer, NodeScan(...)))` form:
//!
//! - `a.<prop> == <literal>` → `NodeByPropertyValue` when `<prop>` is
//!   declared `unique` in the schema (see `PropertyDef::unique`).
//! - `elementId(a) == <expr>` / `id(a) == <expr>` → `NodeById` (a UUID
//!   point lookup), with or without a label on the scan. This is what
//!   turns a GUI's `MATCH (n) WHERE elementId(n) = $id` node fetch and
//!   expand from a full scan into a direct lookup.
//!
//! Triggers a point lookup in place of a full label scan + filter for
//! the LDBC SNB anchor pattern `MATCH (p:Person {id: 'literal'})`,
//! which is the highest-leverage cold-start fix from STATUS.md
//! §"Bench A SF1" (cold IC09 went from 9 s to ~150 ms with the same
//! optimisation in Kuzu / Neo4j range indexes).
//!
//! The pass is conservative: it only fires when
//! 1. The Filter's predicate is exactly `Property(a, <prop>) == <value>`
//!    (single conjunct; multiple conjuncts fall through unchanged —
//!    a follow-up could combine the unique lookup with residual filters).
//! 2. The immediate child is either a bare `NodeScan` or a `CrossProduct`
//!    whose right side is that scan and whose left side produces every alias
//!    referenced by `<value>`.
//! 3. The scan has no pushed predicates/projection and either (a) has an
//!    explicit label whose property is unique/equality-indexed or (b) is
//!    label-agnostic and at least one declared label indexes that property.
//!    The latter uses the id-primary global posting sidecar and always fans
//!    out, preserving equal values across different labels.
//!
//! All other shapes pass through. Runs once per fixpoint iteration like
//! the other rewrites in `optimize::mod`.

use super::{expression_aliases, produced_aliases};
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
                label,
                alias,
                predicates,
                projection,
            } = input.as_ref()
            {
                if predicates.is_empty() && projection.is_none() {
                    if let Some((prop, value_expr)) = extract_eq_on_prop(&predicate, alias) {
                        // The lookup input is `Empty`, so the scan alias is not
                        // bound while `value_expr` is evaluated. Rewriting
                        // `p.city = p.other` (or even `p.city = p.city`) would
                        // therefore turn a valid per-row comparison into an
                        // unbound-variable error. Correlated scans below have
                        // the analogous, stricter subset check against the
                        // aliases produced by their outer input.
                        if expression_aliases(&value_expr).contains(alias) {
                            return LogicalPlan::Filter {
                                input: Box::new(rewrite(*input, catalog)),
                                predicate,
                            };
                        }
                        let lookup = match label {
                            Some(label) => property_lookup_mode(catalog, label, &prop)
                                .map(|multi| (label.clone(), multi)),
                            None if global_property_lookup_available(catalog, &prop) => {
                                // Empty label is the internal label-agnostic
                                // scope. It always fans out: uniqueness is
                                // per-label, not global.
                                Some((String::new(), true))
                            }
                            None => None,
                        };
                        if let Some((label, multi)) = lookup {
                            return LogicalPlan::NodeByPropertyValue {
                                input: Box::new(LogicalPlan::Empty),
                                label,
                                alias: alias.clone(),
                                property: prop,
                                value: value_expr,
                                multi,
                            };
                        }
                    }
                }
            }

            // Correlated inline-property lookup:
            //
            //   UNWIND $keys AS key
            //   MATCH (n:Doc {key: key})-[r]->(:Target)
            //
            // Lowering must keep `key` in scope, so the node pattern becomes
            // `Filter(n.key = key, CrossProduct(Unwind, NodeScan(n)))`. The
            // old rewrite only recognised `Filter(NodeScan)` above, and the
            // later join-conversion pass consequently turned this into a
            // HashJoin that decoded the entire :Doc label once per batch.
            //
            // A NodeByPropertyValue already has an `input` precisely for this
            // correlated shape: execute the outer rows, evaluate the lookup
            // expression against each one, then bind `n` from the index. Only
            // consume the filter when every alias used by the value expression
            // is produced by the outer/left input; otherwise evaluating it
            // before `n` is bound would change semantics.
            let correlated = match input.as_ref() {
                LogicalPlan::CrossProduct { left, right } => match right.as_ref() {
                    LogicalPlan::NodeScan {
                        label,
                        alias,
                        predicates,
                        projection,
                    } if predicates.is_empty() && projection.is_none() => {
                        extract_eq_on_prop(&predicate, alias).and_then(|(prop, value_expr)| {
                            let outer_aliases = produced_aliases(left);
                            let value_aliases = expression_aliases(&value_expr);
                            if !value_aliases.is_subset(&outer_aliases) {
                                return None;
                            }
                            let lookup = match label {
                                Some(label) => property_lookup_mode(catalog, label, &prop)
                                    .map(|multi| (label.clone(), multi)),
                                None if global_property_lookup_available(catalog, &prop) => {
                                    Some((String::new(), true))
                                }
                                None => None,
                            };
                            lookup.map(|(label, multi)| {
                                (label, alias.clone(), prop, value_expr, multi)
                            })
                        })
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some((label, alias, property, value, multi)) = correlated {
                let LogicalPlan::CrossProduct { left, .. } = *input else {
                    unreachable!("correlated lookup was matched as CrossProduct")
                };
                return LogicalPlan::NodeByPropertyValue {
                    input: Box::new(rewrite(*left, catalog)),
                    label,
                    alias,
                    property,
                    value,
                    multi,
                };
            }

            // `WHERE elementId(n) = <v>` / `WHERE id(n) = <v>` over a bare
            // scan becomes a NodeById point lookup by UUID — scoped to the
            // scan's label when it has one, or fanned across observed labels
            // when it doesn't (the unlabelled `MATCH (n) WHERE elementId(n)
            // = $id` shape a GUI fetch/expand uses). Avoids a full scan.
            if let LogicalPlan::NodeScan {
                label,
                alias,
                predicates,
                projection,
            } = input.as_ref()
            {
                if predicates.is_empty() && projection.is_none() {
                    if let Some(value_expr) = extract_eq_on_node_id(&predicate, alias) {
                        return LogicalPlan::NodeById {
                            input: Box::new(LogicalPlan::Empty),
                            label: label.clone(),
                            alias: alias.clone(),
                            id: value_expr,
                        };
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
        LogicalPlan::Empty
        | LogicalPlan::Argument { .. }
        | LogicalPlan::NodeScan { .. }
        | LogicalPlan::MultiwayJoin { .. }
        | LogicalPlan::EdgeTypeCount { .. }
        | LogicalPlan::VectorSearch { .. }
        | LogicalPlan::CallProcedure { .. } => plan,
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
            multi,
        } => LogicalPlan::NodeByPropertyValue {
            input: Box::new(rewrite(*input, catalog)),
            label,
            alias,
            property,
            value,
            multi,
        },
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
        } => LogicalPlan::Expand {
            input: Box::new(rewrite(*input, catalog)),
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
        LogicalPlan::DiscardResult { input } => LogicalPlan::DiscardResult {
            input: Box::new(rewrite(*input, catalog)),
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
        LogicalPlan::Apply { input, subplan } => LogicalPlan::Apply {
            input: Box::new(rewrite(*input, catalog)),
            subplan: Box::new(rewrite(*subplan, catalog)),
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
        LogicalPlan::Foreach {
            input,
            variable,
            list,
            body,
        } => LogicalPlan::Foreach {
            input: Box::new(rewrite(*input, catalog)),
            variable,
            list,
            body,
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

/// Return the `NodeByPropertyValue::multi` flag for an indexed property.
///
/// Unique properties use a point lookup (`false`); non-unique equality indexes
/// use their posting list (`true`). `None` means the property has no lookup
/// index and the caller must preserve the scan/filter plan.
fn property_lookup_mode(catalog: &StatsCatalog, label: &str, property: &str) -> Option<bool> {
    let stats = catalog
        .label(label)
        .and_then(|l| l.properties.get(property))?;
    if stats.unique {
        Some(false)
    } else if stats.indexed {
        Some(true)
    } else {
        None
    }
}

/// Whether id-primary node SSTs carry (or legacy storage can lazily build) a
/// label-agnostic posting index for `property`.
fn global_property_lookup_available(catalog: &StatsCatalog, property: &str) -> bool {
    catalog.label_names().any(|label| {
        catalog
            .label(label)
            .and_then(|stats| stats.properties.get(property))
            .is_some_and(|stats| stats.unique || stats.indexed)
    })
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

/// Match `elementId(<alias>) == <value>` or `id(<alias>) == <value>`
/// (either argument order) and return the value expression. Both
/// functions yield a node's UUID string, which `node_id_from_value`
/// parses back into a `NodeId` at execution time.
fn extract_eq_on_node_id(expr: &Expression, scan_alias: &str) -> Option<Expression> {
    use crate::parser::ast::BinaryOp;
    let ExpressionKind::Binary { op, left, right } = &expr.kind else {
        return None;
    };
    if !matches!(op, BinaryOp::Eq) {
        return None;
    }
    if is_node_id_call(left, scan_alias) {
        return Some((**right).clone());
    }
    if is_node_id_call(right, scan_alias) {
        return Some((**left).clone());
    }
    None
}

/// `elementId(<alias>)` or `id(<alias>)` — a single-argument call of
/// either function on the scan's own variable.
fn is_node_id_call(expr: &Expression, alias: &str) -> bool {
    let ExpressionKind::FunctionCall { name, args, .. } = &expr.kind else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    let fname = name.joined().to_ascii_lowercase();
    if fname != "elementid" && fname != "id" {
        return false;
    }
    matches!(&args[0].kind, ExpressionKind::Variable(id) if id.name == alias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{LabelStats, PropStats, StatsCatalog};
    use crate::parser::parse;
    use crate::plan::lower;
    use std::collections::BTreeMap;

    fn catalog_with_city(unique: bool, indexed: bool) -> StatsCatalog {
        let mut cat = StatsCatalog::empty();
        let mut props = BTreeMap::new();
        props.insert(
            "city".to_string(),
            PropStats {
                unique,
                indexed,
                ..Default::default()
            },
        );
        cat.__test_insert_label(LabelStats {
            name: "Person".into(),
            node_count: 1000,
            properties: props,
        });
        cat
    }

    /// `multi` of the first `NodeByPropertyValue` in the plan, if any.
    fn find_lookup(plan: &LogicalPlan) -> Option<bool> {
        if let LogicalPlan::NodeByPropertyValue { multi, .. } = plan {
            return Some(*multi);
        }
        plan.children().into_iter().find_map(find_lookup)
    }

    fn plan_contains_filter(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::Filter { .. })
            || plan.children().into_iter().any(plan_contains_filter)
    }

    fn rewrite_query(q: &str, cat: &StatsCatalog) -> LogicalPlan {
        let parsed = parse(q).unwrap();
        let plan = lower(&parsed).unwrap();
        apply_unique_property_lookup(plan, cat)
    }

    #[test]
    fn indexed_non_unique_property_emits_multi_index_lookup() {
        let cat = catalog_with_city(false, true);
        let plan = rewrite_query("MATCH (p:Person {city: 'LA'}) RETURN p", &cat);
        assert_eq!(
            find_lookup(&plan),
            Some(true),
            "an indexed non-unique property should emit a multi index lookup"
        );
    }

    #[test]
    fn unique_property_emits_point_lookup() {
        let cat = catalog_with_city(true, false);
        let plan = rewrite_query("MATCH (p:Person {city: 'LA'}) RETURN p", &cat);
        assert_eq!(
            find_lookup(&plan),
            Some(false),
            "a unique property should stay a single point lookup"
        );
    }

    #[test]
    fn unique_property_comparison_to_same_scan_alias_keeps_filter() {
        let cat = catalog_with_city(true, false);
        let plan = rewrite_query("MATCH (p:Person) WHERE p.city = p.city RETURN p", &cat);
        assert_eq!(
            find_lookup(&plan),
            None,
            "the lookup value cannot read p before NodeByPropertyValue binds p"
        );
        assert!(
            plan_contains_filter(&plan),
            "the per-row self comparison must remain as a Filter: {plan:?}"
        );
    }

    #[test]
    fn indexed_property_comparison_to_scan_binding_keeps_filter() {
        let cat = catalog_with_city(false, true);
        let plan = rewrite_query("MATCH (p:Person) WHERE p.city = p RETURN p", &cat);
        assert_eq!(
            find_lookup(&plan),
            None,
            "a multi lookup must not evaluate the scanned binding against Empty"
        );
        assert!(
            plan_contains_filter(&plan),
            "the alias-dependent comparison must remain as a Filter: {plan:?}"
        );
    }

    #[test]
    fn correlated_unique_property_under_expand_emits_point_lookup() {
        let cat = catalog_with_city(true, false);
        let plan = rewrite_query(
            "UNWIND ['LA', 'NYC'] AS wanted \
             MATCH (p:Person {city: wanted})-[r]->(:Place) \
             DELETE r",
            &cat,
        );
        assert_eq!(
            find_lookup(&plan),
            Some(false),
            "the UNWIND-correlated anchor must use one unique lookup per key"
        );

        fn lookup_input(plan: &LogicalPlan) -> Option<&LogicalPlan> {
            if let LogicalPlan::NodeByPropertyValue { input, .. } = plan {
                return Some(input);
            }
            plan.children().into_iter().find_map(lookup_input)
        }
        assert!(
            matches!(lookup_input(&plan), Some(LogicalPlan::Unwind { .. })),
            "the outer UNWIND must become the lookup input, got {plan:?}"
        );
    }

    #[test]
    fn correlated_unindexed_property_keeps_scan_join() {
        let cat = catalog_with_city(false, false);
        let plan = rewrite_query(
            "UNWIND ['LA', 'NYC'] AS wanted \
             MATCH (p:Person {city: wanted})-[r]->(:Place) \
             DELETE r",
            &cat,
        );
        assert_eq!(
            find_lookup(&plan),
            None,
            "an outer binding must not make an unindexed property indexable"
        );
    }

    #[test]
    fn correlated_unlabelled_property_uses_global_posting_lookup() {
        let cat = catalog_with_city(true, false);
        let plan = rewrite_query(
            "UNWIND ['LA', 'NYC'] AS wanted \
             MATCH (p {city: wanted})-[r]->(:Place) \
             DELETE r",
            &cat,
        );
        assert_eq!(
            find_lookup(&plan),
            Some(true),
            "the label-agnostic anchor uses a global posting lookup, never a guessed unique scope"
        );
    }

    #[test]
    fn unindexed_property_stays_a_scan() {
        let cat = catalog_with_city(false, false);
        let plan = rewrite_query("MATCH (p:Person {city: 'LA'}) RETURN p", &cat);
        assert_eq!(
            find_lookup(&plan),
            None,
            "an un-indexed property must keep the full scan + filter"
        );
    }
}
