//! Point-lookup rewrites over a labelled node scan, including its correlated
//! `Filter(..., CrossProduct(outer, NodeScan(...)))` form:
//!
//! - `a.<prop> == <literal>` → `NodeByPropertyValue` when `<prop>` is
//!   declared `unique` in the schema (see `PropertyDef::unique`).
//! - `a.<m1> == <lit> AND a.<m2> == <lit> [AND ...]` → `NodeByPropertyTuple`
//!   when the conjuncts cover EVERY member of a declared composite index
//!   (`Schema::indexes`). Priority: unique single-property first (at most
//!   one row), then the longest covered composite, then non-unique
//!   single-property postings.
//! - `elementId(a) == <expr>` / `id(a) == <expr>` → `NodeById` (a UUID
//!   point lookup), with or without a label on the scan. This is what
//!   turns a GUI's `MATCH (n) WHERE elementId(n) = $id` node fetch and
//!   expand from a full scan into a direct lookup. `id(a) IN <list>` seeks
//!   every listed id through `NodeById(Distinct(Unwind(<list>)))`.
//!
//! Triggers a point lookup in place of a full label scan + filter for
//! the LDBC SNB anchor pattern `MATCH (p:Person {id: 'literal'})`,
//! which is the highest-leverage cold-start fix from STATUS.md
//! §"Bench A SF1" (cold IC09 went from 9 s to ~150 ms with the same
//! optimisation in Kuzu / Neo4j range indexes).
//!
//! The pass is conservative: it only fires when
//! 1. A top-level `AND` conjunct is `Property(a, <prop>) == <value>`.
//!    The best indexed conjunct (unique before non-unique) becomes the
//!    lookup and every other conjunct remains as a residual `Filter`.
//! 2. The immediate child is either a bare `NodeScan` or a `CrossProduct`
//!    whose right side is that scan and whose left side produces every alias
//!    referenced by `<value>`.
//! 3. The scan either (a) has an explicit label whose property is
//!    unique/equality-indexed or (b) is label-agnostic and at least one
//!    declared label indexes that property. A bare scan's pushed predicates
//!    and projection do not block it; the predicates join the residual.
//!    The latter uses the id-primary global posting sidecar and always fans
//!    out, preserving equal values across different labels.
//!
//! All other shapes pass through. Runs once per fixpoint iteration like
//! the other rewrites in `optimize::mod`.

use super::{expression_aliases, produced_aliases};
use crate::cost::StatsCatalog;
use crate::parser::ast::{Expression, ExpressionKind};
use crate::plan::LogicalPlan;
use std::collections::BTreeSet;

/// Run the unique-property lookup rewrite over `plan`.
pub fn apply_unique_property_lookup(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    rewrite(plan, catalog)
}

fn rewrite(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    match plan {
        // The pattern we care about: `Filter(eq-on-prop, NodeScan(label))`.
        LogicalPlan::Filter { input, predicate } => {
            // Try to match the rewrite pattern.
            //
            // A scan projection does not block any of the lookups below: it
            // only narrows what the scan decodes, and a lookup loads the whole
            // node anyway. Pushed scan predicates do not block them either.
            // On an Expand source the filter only reaches the scan a round
            // after pushdown moved the literal conjuncts into it, so they are
            // folded back in and stay behind as part of the residual.
            if let LogicalPlan::NodeScan {
                label,
                alias,
                predicates: scan_predicates,
                ..
            } = input.as_ref()
            {
                let predicate = with_scan_predicates(&predicate, scan_predicates, alias);
                // Lookup priority: a unique single-property conjunct
                // (at most one row) beats a declared composite index,
                // which beats a non-unique single-property posting (the
                // composite consumes MORE conjuncts, so its posting set
                // is a subset of any member's).
                let single =
                    extract_indexed_conjunct(&predicate, alias, label.as_deref(), catalog, None);
                if let Some(indexed) = &single {
                    if !indexed.multi {
                        return finish_single_lookup(single.expect("checked above"), alias);
                    }
                }
                if let Some(label) = label.as_deref() {
                    if let Some(composite) =
                        extract_composite_conjuncts(&predicate, alias, label, catalog)
                    {
                        let lookup = LogicalPlan::NodeByPropertyTuple {
                            input: Box::new(LogicalPlan::Empty),
                            label: label.to_owned(),
                            alias: alias.clone(),
                            properties: composite.properties,
                            values: composite.values,
                        };
                        return match composite.residual {
                            Some(residual) => LogicalPlan::Filter {
                                input: Box::new(lookup),
                                predicate: residual,
                            },
                            None => lookup,
                        };
                    }
                }
                if let Some(indexed) = single {
                    return finish_single_lookup(indexed, alias);
                }
                // `WHERE elementId(n) = <v>` / `WHERE id(n) = <v>` becomes a
                // NodeById point lookup by UUID, scoped to the scan's label
                // when it has one, or fanned across observed labels when it
                // doesn't (the unlabelled `MATCH (n) WHERE elementId(n) = $id`
                // shape a GUI fetch/expand uses). `id(n) IN <list>` seeks each
                // listed id the same way. Avoids a full scan.
                if let Some(seek) = extract_node_id_conjunct(&predicate, alias, None) {
                    let (lookup, residual) =
                        id_seek(label.clone(), alias, seek, LogicalPlan::Empty);
                    return with_residual(lookup, residual);
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
                        ..
                    } if predicates.is_empty() => {
                        let outer_aliases = produced_aliases(left);
                        extract_indexed_conjunct(
                            &predicate,
                            alias,
                            label.as_deref(),
                            catalog,
                            Some(&outer_aliases),
                        )
                        .map(|indexed| (alias.clone(), indexed))
                    }
                    _ => None,
                },
                _ => None,
            };
            // Same correlated shape keyed by id: `UNWIND $ids AS x MATCH (n)
            // WHERE id(n) = x` binds `n` per outer row by point lookup. Only
            // equality: an `IN` seek dedupes its ids, which would also
            // collapse duplicate outer rows here.
            let correlated_id = match input.as_ref() {
                LogicalPlan::CrossProduct { left, right } if correlated.is_none() => {
                    match right.as_ref() {
                        LogicalPlan::NodeScan {
                            label,
                            alias,
                            predicates,
                            ..
                        } if predicates.is_empty() => extract_node_id_conjunct(
                            &predicate,
                            alias,
                            Some(&produced_aliases(left)),
                        )
                        .filter(|seek| !seek.many)
                        .map(|seek| (label.clone(), alias.clone(), seek)),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some((label, alias, seek)) = correlated_id {
                let LogicalPlan::CrossProduct { left, .. } = *input else {
                    unreachable!("correlated id seek was matched as CrossProduct")
                };
                let (lookup, residual) = id_seek(label, &alias, seek, rewrite(*left, catalog));
                return with_residual(lookup, residual);
            }
            if let Some((alias, indexed)) = correlated {
                let LogicalPlan::CrossProduct { left, .. } = *input else {
                    unreachable!("correlated lookup was matched as CrossProduct")
                };
                let lookup = LogicalPlan::NodeByPropertyValue {
                    input: Box::new(rewrite(*left, catalog)),
                    label: indexed.label,
                    alias,
                    property: indexed.property,
                    value: indexed.value,
                    multi: indexed.multi,
                };
                return match indexed.residual {
                    Some(residual) => LogicalPlan::Filter {
                        input: Box::new(lookup),
                        predicate: residual,
                    },
                    None => lookup,
                };
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
        LogicalPlan::NodeByPropertyTuple {
            input,
            label,
            alias,
            properties,
            values,
        } => LogicalPlan::NodeByPropertyTuple {
            input: Box::new(rewrite(*input, catalog)),
            label,
            alias,
            properties,
            values,
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
            unique_rels,
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
            unique_rels,
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

pub(super) struct IndexedConjunct {
    pub(super) label: String,
    pub(super) property: String,
    pub(super) value: Expression,
    pub(super) multi: bool,
    pub(super) residual: Option<Expression>,
}

/// Pick the best indexable equality from a top-level conjunction. Unique
/// lookups win over non-unique postings even when they appear later, because
/// they bound the candidate set to at most one row. The remaining expression
/// tree is retained as a residual filter, so extracting the lookup never drops
/// a predicate.
///
/// `outer_aliases == None` denotes an uncorrelated scan. In that shape the
/// lookup value cannot reference any row binding because the lookup input is
/// `Empty`. A correlated scan instead requires every value alias to be
/// provided by its outer input.
pub(super) fn extract_indexed_conjunct(
    predicate: &Expression,
    scan_alias: &str,
    label: Option<&str>,
    catalog: &StatsCatalog,
    outer_aliases: Option<&BTreeSet<String>>,
) -> Option<IndexedConjunct> {
    let mut conjuncts = Vec::new();
    collect_top_level_conjuncts(predicate, &mut conjuncts);

    let mut best: Option<(usize, String, String, Expression, bool)> = None;
    for (index, conjunct) in conjuncts.iter().enumerate() {
        let Some((property, value)) = extract_eq_on_prop(conjunct, scan_alias) else {
            continue;
        };
        let value_aliases = expression_aliases(&value);
        let aliases_are_available = match outer_aliases {
            Some(outer) => value_aliases.is_subset(outer),
            None => value_aliases.is_empty(),
        };
        if !aliases_are_available {
            continue;
        }

        let lookup = match label {
            Some(label) => property_lookup_mode(catalog, label, &property)
                .map(|multi| (label.to_owned(), multi)),
            None if global_property_lookup_available(catalog, &property) => {
                // Empty label is the internal label-agnostic scope. It always
                // fans out: uniqueness is per-label, not global.
                Some((String::new(), true))
            }
            None => None,
        };
        let Some((lookup_label, multi)) = lookup else {
            continue;
        };

        // Preserve source order within the same lookup class, but replace a
        // non-unique candidate with any unique one found later.
        if best
            .as_ref()
            .is_some_and(|(_, _, _, _, best_multi)| !*best_multi || multi)
        {
            continue;
        }
        best = Some((index, lookup_label, property, value, multi));
    }

    let (index, label, property, value, multi) = best?;
    let mut leaf_index = 0usize;
    let residual = remove_top_level_conjunct(predicate, index, &mut leaf_index);
    debug_assert_eq!(
        leaf_index,
        conjuncts.len(),
        "conjunction traversal must assign stable leaf ordinals"
    );
    Some(IndexedConjunct {
        label,
        property,
        value,
        multi,
        residual,
    })
}

/// Build the `NodeByPropertyValue` (+ residual filter) for a matched
/// single-property conjunct — the tail the two priority branches share.
fn finish_single_lookup(indexed: IndexedConjunct, alias: &str) -> LogicalPlan {
    let lookup = LogicalPlan::NodeByPropertyValue {
        input: Box::new(LogicalPlan::Empty),
        label: indexed.label,
        alias: alias.to_owned(),
        property: indexed.property,
        value: indexed.value,
        multi: indexed.multi,
    };
    match indexed.residual {
        Some(residual) => LogicalPlan::Filter {
            input: Box::new(lookup),
            predicate: residual,
        },
        None => lookup,
    }
}

/// A matched composite-index lookup: members in DECLARATION order with
/// their aligned value expressions, plus whatever conjuncts remain.
struct CompositeConjunct {
    properties: Vec<String>,
    values: Vec<Expression>,
    residual: Option<Expression>,
}

/// Match the scan's top-level equality conjuncts against the declared
/// composite indexes. Fires only when EVERY member of a declared index has
/// an uncorrelated `alias.<member> = <expr>` conjunct; among matches the
/// longest member list wins (most conjuncts consumed). The matched values
/// are reordered to declaration order — the tuple key layout — and every
/// consumed conjunct leaves the residual.
fn extract_composite_conjuncts(
    predicate: &Expression,
    scan_alias: &str,
    label: &str,
    catalog: &StatsCatalog,
) -> Option<CompositeConjunct> {
    let mut conjuncts = Vec::new();
    collect_top_level_conjuncts(predicate, &mut conjuncts);

    // First equality conjunct per property; a duplicate (`a=1 AND a=2`)
    // stays in the residual and keeps filtering.
    let mut equalities: std::collections::BTreeMap<String, (usize, Expression)> =
        std::collections::BTreeMap::new();
    for (index, conjunct) in conjuncts.iter().enumerate() {
        let Some((property, value)) = extract_eq_on_prop(conjunct, scan_alias) else {
            continue;
        };
        if !expression_aliases(&value).is_empty() {
            continue;
        }
        equalities.entry(property).or_insert((index, value));
    }
    if equalities.len() < 2 {
        return None;
    }

    let mut best: Option<&namidb_core::schema::IndexDef> = None;
    for def in catalog.composite_indexes() {
        if def.label != label {
            continue;
        }
        if !def
            .properties
            .iter()
            .all(|member| equalities.contains_key(member))
        {
            continue;
        }
        if best.is_none_or(|b| def.properties.len() > b.properties.len()) {
            best = Some(def);
        }
    }
    let def = best?;

    let mut used = BTreeSet::new();
    let mut values = Vec::with_capacity(def.properties.len());
    for member in &def.properties {
        let (index, value) = &equalities[member];
        used.insert(*index);
        values.push(value.clone());
    }
    let mut leaf_index = 0usize;
    let residual = remove_top_level_conjunct_set(predicate, &used, &mut leaf_index);
    debug_assert_eq!(
        leaf_index,
        conjuncts.len(),
        "conjunction traversal must assign stable leaf ordinals"
    );
    Some(CompositeConjunct {
        properties: def.properties.clone(),
        values,
        residual,
    })
}

/// Set-removal twin of [`remove_top_level_conjunct`]: drop every leaf whose
/// ordinal is in `remove`, preserving the remaining tree shape.
fn remove_top_level_conjunct_set(
    expr: &Expression,
    remove: &BTreeSet<usize>,
    next_index: &mut usize,
) -> Option<Expression> {
    if let ExpressionKind::Binary {
        op: crate::parser::ast::BinaryOp::And,
        left,
        right,
    } = &expr.kind
    {
        let left = remove_top_level_conjunct_set(left, remove, next_index);
        let right = remove_top_level_conjunct_set(right, remove, next_index);
        return match (left, right) {
            (Some(left), Some(right)) => Some(Expression {
                span: expr.span,
                kind: ExpressionKind::Binary {
                    op: crate::parser::ast::BinaryOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                },
            }),
            (Some(remaining), None) | (None, Some(remaining)) => Some(remaining),
            (None, None) => None,
        };
    }

    let index = *next_index;
    *next_index += 1;
    (!remove.contains(&index)).then(|| expr.clone())
}

fn collect_top_level_conjuncts<'a>(expr: &'a Expression, out: &mut Vec<&'a Expression>) {
    if let ExpressionKind::Binary {
        op: crate::parser::ast::BinaryOp::And,
        left,
        right,
    } = &expr.kind
    {
        collect_top_level_conjuncts(left, out);
        collect_top_level_conjuncts(right, out);
    } else {
        out.push(expr);
    }
}

/// Clone a conjunction while removing exactly the leaf at `remove_index`.
/// Keeping the original tree shape avoids gratuitously reassociating boolean
/// evaluation and preserves source spans for the residual diagnostics.
fn remove_top_level_conjunct(
    expr: &Expression,
    remove_index: usize,
    next_index: &mut usize,
) -> Option<Expression> {
    if let ExpressionKind::Binary {
        op: crate::parser::ast::BinaryOp::And,
        left,
        right,
    } = &expr.kind
    {
        let left = remove_top_level_conjunct(left, remove_index, next_index);
        let right = remove_top_level_conjunct(right, remove_index, next_index);
        return match (left, right) {
            (Some(left), Some(right)) => Some(Expression {
                span: expr.span,
                kind: ExpressionKind::Binary {
                    op: crate::parser::ast::BinaryOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                },
            }),
            (Some(remaining), None) | (None, Some(remaining)) => Some(remaining),
            (None, None) => None,
        };
    }

    let index = *next_index;
    *next_index += 1;
    (index != remove_index).then(|| expr.clone())
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

/// A `WHERE` conjunct that pins a scan's node by id: `id(a) = <value>` or
/// `id(a) IN <value>` (`elementId` likewise, either side of `=`).
pub(super) struct IdSeek {
    /// The `id(a)` / `elementId(a)` call itself.
    call: Expression,
    pub(super) value: Expression,
    /// `IN`: `value` is a list of ids rather than one id.
    pub(super) many: bool,
    pub(super) residual: Option<Expression>,
}

/// Find the first top-level conjunct of `predicate` that seeks `scan_alias`
/// by id. Both functions yield a node's UUID string, which the `NodeById`
/// executor parses back into a `NodeId`. `outer_aliases` follows
/// [`extract_indexed_conjunct`]: `None` means the value may not reference
/// any binding, `Some` that the outer input must produce every one it does.
pub(super) fn extract_node_id_conjunct(
    predicate: &Expression,
    scan_alias: &str,
    outer_aliases: Option<&BTreeSet<String>>,
) -> Option<IdSeek> {
    use crate::parser::ast::BinaryOp;
    let mut conjuncts = Vec::new();
    collect_top_level_conjuncts(predicate, &mut conjuncts);
    let (index, call, value, many) =
        conjuncts.iter().enumerate().find_map(|(index, conjunct)| {
            let (call, value, many) = match &conjunct.kind {
                ExpressionKind::Binary {
                    op: BinaryOp::Eq,
                    left,
                    right,
                } if is_node_id_call(left, scan_alias) => (left, right, false),
                ExpressionKind::Binary {
                    op: BinaryOp::Eq,
                    left,
                    right,
                } if is_node_id_call(right, scan_alias) => (right, left, false),
                ExpressionKind::In { item, list } if is_node_id_call(item, scan_alias) => {
                    (item, list, true)
                }
                _ => return None,
            };
            let aliases = expression_aliases(value);
            let available = match outer_aliases {
                Some(outer) => aliases.is_subset(outer),
                None => aliases.is_empty(),
            };
            available.then(|| (index, (**call).clone(), (**value).clone(), many))
        })?;
    let mut leaf_index = 0usize;
    let residual = remove_top_level_conjunct(predicate, index, &mut leaf_index);
    Some(IdSeek {
        call,
        value,
        many,
        residual,
    })
}

/// Build the `NodeById` for `seek` over `input`, and hand back its residual.
/// An `IN` seek unwinds the list into a hidden binding first; the `Distinct`
/// keeps a repeated id to one row, as the filter it replaces would.
///
/// The seek resolves its key as leniently as the `{_id: ...}` sigil does (a
/// Node value, any UUID spelling), so `call = key` stays on top of it: the
/// filter it replaces compares the canonical id string and nothing else.
pub(super) fn id_seek(
    label: Option<String>,
    alias: &str,
    seek: IdSeek,
    input: LogicalPlan,
) -> (LogicalPlan, Option<Expression>) {
    let IdSeek {
        call,
        value,
        many,
        residual,
    } = seek;
    let (input, id) = if many {
        let slot = format!("__idseek_{alias}");
        let id = Expression {
            span: value.span,
            kind: ExpressionKind::Variable(crate::parser::ast::Identifier::new(
                slot.clone(),
                value.span,
            )),
        };
        let unwind = LogicalPlan::Unwind {
            input: Box::new(input),
            list: value,
            alias: slot,
        };
        (
            LogicalPlan::Distinct {
                input: Box::new(unwind),
            },
            id,
        )
    } else {
        (input, value)
    };
    let span = call.span;
    let exact = Expression {
        span,
        kind: ExpressionKind::Binary {
            op: crate::parser::ast::BinaryOp::Eq,
            left: Box::new(call),
            right: Box::new(id.clone()),
        },
    };
    let lookup = LogicalPlan::NodeById {
        input: Box::new(input),
        label,
        alias: alias.to_owned(),
        id,
    };
    (with_residual(lookup, Some(exact)), residual)
}

/// `predicate` with a scan's pushed predicates ANDed back on as filter
/// conjuncts, so a lookup that replaces the scan keeps enforcing them.
fn with_scan_predicates(
    predicate: &Expression,
    scan_predicates: &[namidb_storage::sst::predicates::ScanPredicate],
    alias: &str,
) -> Expression {
    scan_predicates
        .iter()
        .fold(predicate.clone(), |acc, pushed| {
            let pushed = crate::cost::selectivity::scan_predicate_to_expression(pushed, alias);
            Expression {
                span: acc.span,
                kind: ExpressionKind::Binary {
                    op: crate::parser::ast::BinaryOp::And,
                    left: Box::new(acc),
                    right: Box::new(pushed),
                },
            }
        })
}

fn with_residual(plan: LogicalPlan, residual: Option<Expression>) -> LogicalPlan {
    match residual {
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        },
        None => plan,
    }
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

    fn catalog_with_lookup_mix() -> StatsCatalog {
        let mut cat = StatsCatalog::empty();
        let mut props = BTreeMap::new();
        props.insert(
            "city".to_string(),
            PropStats {
                indexed: true,
                ..Default::default()
            },
        );
        props.insert(
            "external_id".to_string(),
            PropStats {
                unique: true,
                ..Default::default()
            },
        );
        props.insert("active".to_string(), PropStats::default());
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

    fn find_lookup_property(plan: &LogicalPlan) -> Option<&str> {
        if let LogicalPlan::NodeByPropertyValue { property, .. } = plan {
            return Some(property);
        }
        plan.children().into_iter().find_map(find_lookup_property)
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
    fn unique_conjunct_drives_lookup_and_preserves_residual() {
        let cat = catalog_with_lookup_mix();
        let plan = rewrite_query(
            "MATCH (p:Person) \
             WHERE p.city = 'LA' AND p.active = true AND p.external_id = 'p-1' \
             RETURN p",
            &cat,
        );
        assert_eq!(
            find_lookup_property(&plan),
            Some("external_id"),
            "the unique equality must win over an earlier non-unique posting lookup: {plan:?}"
        );
        assert_eq!(find_lookup(&plan), Some(false));
        assert!(
            plan_contains_filter(&plan),
            "unselected conjuncts must remain as a residual Filter: {plan:?}"
        );
    }

    #[test]
    fn indexed_conjunct_does_not_cross_top_level_or() {
        let cat = catalog_with_lookup_mix();
        let plan = rewrite_query(
            "MATCH (p:Person) \
             WHERE p.external_id = 'p-1' OR p.active = true \
             RETURN p",
            &cat,
        );
        assert_eq!(
            find_lookup(&plan),
            None,
            "extracting an equality from OR would drop valid matches"
        );
        assert!(plan_contains_filter(&plan));
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
    fn correlated_unique_conjunct_preserves_residual_filter() {
        let cat = catalog_with_lookup_mix();
        let plan = rewrite_query(
            "UNWIND ['p-1', 'p-2'] AS wanted \
             MATCH (p:Person) \
             WHERE p.external_id = wanted AND p.active = true \
             SET p.active = false",
            &cat,
        );
        assert_eq!(find_lookup(&plan), Some(false));
        assert_eq!(find_lookup_property(&plan), Some("external_id"));
        assert!(
            plan_contains_filter(&plan),
            "the node predicate not consumed by the lookup must survive: {plan:?}"
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

    /// `(input, id)` of the first `NodeById` in the plan, if any.
    fn find_node_by_id(plan: &LogicalPlan) -> Option<(&LogicalPlan, &Expression)> {
        if let LogicalPlan::NodeById { input, id, .. } = plan {
            return Some((input, id));
        }
        plan.children().into_iter().find_map(find_node_by_id)
    }

    fn plan_contains_node_scan(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeScan { .. })
            || plan.children().into_iter().any(plan_contains_node_scan)
    }

    /// The `Distinct(Unwind(Empty))` an `IN` seek reads its ids from.
    fn assert_in_seek(plan: &LogicalPlan) {
        let (input, id) = find_node_by_id(plan).unwrap_or_else(|| panic!("no seek: {plan:?}"));
        let LogicalPlan::Distinct { input } = input else {
            panic!("an IN seek must dedupe its ids: {plan:?}");
        };
        let LogicalPlan::Unwind { input, alias, .. } = input.as_ref() else {
            panic!("an IN seek must unwind its list: {plan:?}");
        };
        assert!(matches!(input.as_ref(), LogicalPlan::Empty));
        assert!(matches!(&id.kind, ExpressionKind::Variable(v) if v.name == *alias));
        assert!(!plan_contains_node_scan(plan), "{plan:?}");
    }

    #[test]
    fn id_in_param_lowers_to_node_by_id_over_distinct_unwind() {
        let cat = StatsCatalog::empty();
        assert_in_seek(&rewrite_query(
            "MATCH (n:Person) WHERE id(n) IN $ids RETURN n",
            &cat,
        ));
        assert_in_seek(&rewrite_query(
            "MATCH (n) WHERE elementId(n) IN ['a', 'b'] RETURN n",
            &cat,
        ));
    }

    #[test]
    fn id_eq_with_residual_conjunct_still_seeks() {
        let cat = StatsCatalog::empty();
        let plan = rewrite_query(
            "MATCH (n:Person) WHERE n.age > 1 AND id(n) = $id RETURN n",
            &cat,
        );
        assert!(find_node_by_id(&plan).is_some(), "{plan:?}");
        assert!(!plan_contains_node_scan(&plan), "{plan:?}");
        assert!(
            plan_contains_filter(&plan),
            "the other conjunct must remain as a residual Filter: {plan:?}"
        );
    }

    #[test]
    fn correlated_unwind_id_eq_seeks() {
        let cat = StatsCatalog::empty();
        let plan = rewrite_query("UNWIND $ids AS x MATCH (n) WHERE id(n) = x RETURN n", &cat);
        let (input, _) = find_node_by_id(&plan).unwrap_or_else(|| panic!("{plan:?}"));
        assert!(
            matches!(input, LogicalPlan::Unwind { .. }),
            "the outer UNWIND must drive the seek: {plan:?}"
        );
        assert!(!plan_contains_node_scan(&plan), "{plan:?}");
    }

    #[test]
    fn id_seek_on_an_expand_source_survives_projection_pushdown() {
        // The filter only lands on the scan after a pushdown round, by which
        // time projection pushdown has narrowed the scan to its id column.
        for q in [
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE id(a) = $id RETURN id(b)",
            "MATCH (a)-[:KNOWS]->(b) WHERE id(a) IN $ids RETURN b",
        ] {
            let plan = crate::optimize::optimize(
                lower(&parse(q).unwrap()).unwrap(),
                &StatsCatalog::empty(),
            );
            assert!(find_node_by_id(&plan).is_some(), "{q}: {plan:?}");
            assert!(!plan_contains_node_scan(&plan), "{q}: {plan:?}");
        }
    }

    #[test]
    fn id_seek_on_an_expand_source_keeps_pushed_conjuncts() {
        // By the round the id filter reaches the scan, pushdown has moved the
        // literal conjunct (or the inline property) into the scan itself.
        for q in [
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE id(a) = $id AND a.age > 1 RETURN b",
            "MATCH (a:Person {age: 3})-[:KNOWS]->(b) WHERE id(a) IN $ids RETURN id(b)",
        ] {
            let plan = crate::optimize::optimize(
                lower(&parse(q).unwrap()).unwrap(),
                &StatsCatalog::empty(),
            );
            assert!(find_node_by_id(&plan).is_some(), "{q}: {plan:?}");
            assert!(!plan_contains_node_scan(&plan), "{q}: {plan:?}");
            assert!(
                format!("{plan:?}").contains("\"age\""),
                "the pushed conjunct must stay as a residual: {q}: {plan:?}"
            );
        }
    }

    #[test]
    fn unique_lookup_on_an_expand_source_read_only_by_identity() {
        // Projection pushdown narrows a scan read only through id() or
        // count() before the parameter filter reaches it.
        let cat = catalog_with_city(true, false);
        for q in [
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.city = $c RETURN id(a), id(b)",
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.city = $c RETURN count(a)",
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.city = $c AND a.age > 1 RETURN labels(a)",
        ] {
            let plan = crate::optimize::optimize(lower(&parse(q).unwrap()).unwrap(), &cat);
            assert_eq!(find_lookup(&plan), Some(false), "{q}: {plan:?}");
            assert!(!plan_contains_node_scan(&plan), "{q}: {plan:?}");
        }
    }

    #[test]
    fn id_seek_on_other_alias_or_self_reference_does_not_fire() {
        let cat = StatsCatalog::empty();
        for q in [
            "MATCH (n:Person) WHERE id(n) = n.ref RETURN n",
            "MATCH (n:Person) WHERE id(n) IN n.refs RETURN n",
            "MATCH (n:Person) WHERE id(n) = $a OR n.age > 1 RETURN n",
            // An IN seek dedupes its ids, so it cannot sit under outer rows.
            "UNWIND $lists AS xs MATCH (n) WHERE id(n) IN xs RETURN n",
        ] {
            let plan = rewrite_query(q, &cat);
            assert!(find_node_by_id(&plan).is_none(), "{q}: {plan:?}");
        }
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
