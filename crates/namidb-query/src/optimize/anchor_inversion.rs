//! Pattern-anchor inversion (25tb-readiness item 36).
//!
//! Lowering anchors a relationship chain at its textual head, so
//! `MATCH (p:Person)-[w:WORKS_AT]->(c:Company {cid: 0})` scans every Person
//! and expands forward, while the semantically identical
//! `MATCH (c:Company {cid: 0})<-[w:WORKS_AT]-(p:Person)` starts from one
//! unique lookup. Found live during the 200k-node S3 validation: the first
//! form took ~18s warm and timed out cold; the second answered instantly.
//!
//! This pass rewrites the first shape into the second: when a single-hop,
//! non-optional Expand reads from a BARE NodeScan and a Filter directly
//! above it pins the expand TARGET with an index-answerable equality, anchor
//! at the target (`NodeByPropertyValue`) and walk the edge in the inverse
//! direction. Row bindings are unchanged — the relationship value carries
//! the stored `(src, dst)` regardless of traversal direction, and the old
//! source's label constraint moves into the inverted Expand's
//! `target_labels`, where the executor enforces it conjunctively.
//!
//! Deliberately narrow:
//! - `length: None` only — a starred alias binds the path's relationship
//!   LIST in pattern order, and a path binding materializes head→target, so
//!   inverting either would change observable shapes.
//! - `optional: false` only — OPTIONAL nullifies the *target* side; swapping
//!   endpoints would nullify the wrong side.
//! - `back_reference: false`, `shortest: None`, no `path_binding`.
//! - The source must be a bare `NodeScan` (no pushed predicates, no
//!   projection): if the source is filtered too, choosing sides needs real
//!   cost comparison, which stays with the join-reorder work.
//! - A `multi: true` (posting list) target anchor is only taken when label
//!   statistics say the target label is no larger than the source label;
//!   `multi: false` (unique) is always at most one row and always wins.

use crate::cost::StatsCatalog;
use crate::parser::RelationshipDirection;
use crate::plan::LogicalPlan;

use super::unique_lookup::extract_indexed_conjunct;

pub fn apply_anchor_inversion(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    rewrite(plan, catalog)
}

fn rewrite(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    match plan {
        LogicalPlan::Filter { input, predicate } => {
            if let Some(inverted) = try_invert(&predicate, input.as_ref(), catalog) {
                return inverted;
            }
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

fn try_invert(
    predicate: &crate::parser::Expression,
    input: &LogicalPlan,
    catalog: &StatsCatalog,
) -> Option<LogicalPlan> {
    let LogicalPlan::Expand {
        input: expand_input,
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
    } = input
    else {
        return None;
    };
    if length.is_some()
        || *optional
        || *back_reference
        || !matches!(shortest, crate::plan::ShortestMode::None)
        || path_binding.is_some()
    {
        return None;
    }
    // The stored-edge orientation must stay decidable after the swap.
    let inverted_direction = match direction {
        RelationshipDirection::Right => RelationshipDirection::Left,
        RelationshipDirection::Left => RelationshipDirection::Right,
        RelationshipDirection::Both => RelationshipDirection::Both,
    };
    // Source side must be a completely unselective scan; anything filtered
    // would need a real cost comparison to pick a side. One exception sees
    // through the veil: a label DISJUNCTION source `(o:A|B)` lowers to an
    // unlabeled scan behind a pure `__label_eq OR __label_eq` filter — its
    // selectivity is exactly as knowable as a labeled scan's (the sum of
    // the disjunct labels), and without this arm every anchored query over
    // a disjunction source re-scans the whole namespace per anchor value
    // (fifth field report: 20s per date, worse than the unanchored form).
    // (alias, single label, label-disjunction filter + its labels).
    type SourceShape<'p> = (
        &'p String,
        Option<String>,
        Option<(crate::parser::Expression, Vec<String>)>,
    );
    let (source_alias, source_label, source_disjunction): SourceShape<'_> =
        match expand_input.as_ref() {
            LogicalPlan::NodeScan {
                label,
                alias,
                predicates,
                projection,
            } if predicates.is_empty() && projection.is_none() => (alias, label.clone(), None),
            LogicalPlan::Filter {
                predicate: label_pred,
                input: scan,
            } => {
                let LogicalPlan::NodeScan {
                    label: None,
                    alias,
                    predicates,
                    projection,
                } = scan.as_ref()
                else {
                    return None;
                };
                if !predicates.is_empty() || projection.is_some() {
                    return None;
                }
                let labels = crate::optimize::extract_label_disjunction(label_pred, alias)?;
                (alias, None, Some((label_pred.clone(), labels)))
            }
            _ => return None,
        };
    debug_assert_eq!(source, source_alias);
    // The equality must pin the expand TARGET under its declared label. A
    // multi-label (conjunctive) target keeps the scan-side plan: the lookup
    // could only prove one of the labels.
    let [target_label] = target_labels.as_slice() else {
        return None;
    };
    let indexed = extract_indexed_conjunct(
        predicate,
        target_alias,
        Some(target_label.as_str()),
        catalog,
        None,
    )?;
    if indexed.label != *target_label {
        return None;
    }
    if indexed.multi {
        // A posting-list anchor fans out; only take it when the target label
        // is provably no larger than the label(s) we would otherwise scan —
        // a disjunction source counts as the SUM of its disjunct labels.
        let target_count = catalog.label(target_label).map(|stats| stats.node_count);
        let source_count = match &source_disjunction {
            Some((_, labels)) => labels.iter().try_fold(0u64, |total, label| {
                catalog
                    .label(label)
                    .map(|stats| total.saturating_add(stats.node_count))
            }),
            None => source_label
                .as_deref()
                .and_then(|label| catalog.label(label))
                .map(|stats| stats.node_count),
        };
        match (target_count, source_count) {
            (Some(target), Some(source)) if target <= source => {}
            _ => return None,
        }
    }

    let anchor = LogicalPlan::NodeByPropertyValue {
        input: Box::new(LogicalPlan::Empty),
        label: indexed.label,
        alias: target_alias.clone(),
        property: indexed.property,
        value: indexed.value,
        multi: indexed.multi,
    };
    let inverted = LogicalPlan::Expand {
        input: Box::new(anchor),
        source: target_alias.clone(),
        edge_type: edge_type.clone(),
        direction: inverted_direction,
        rel_alias: rel_alias.clone(),
        target_alias: source_alias.clone(),
        // A disjunction cannot ride the conjunctive target_labels field; its
        // OR-filter re-attaches below and now runs over the anchor's small
        // neighbourhood instead of the whole namespace.
        target_labels: match &source_disjunction {
            Some(_) => Vec::new(),
            None => source_label.iter().cloned().collect(),
        },
        length: None,
        optional: false,
        back_reference: false,
        shortest: crate::plan::ShortestMode::None,
        path_binding: None,
    };
    let mut result = match source_disjunction {
        Some((label_pred, _)) => LogicalPlan::Filter {
            input: Box::new(inverted),
            predicate: label_pred,
        },
        None => inverted,
    };
    if let Some(residual) = indexed.residual {
        result = LogicalPlan::Filter {
            input: Box::new(result),
            predicate: residual,
        };
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::cost::{LabelStats, PropStats};
    use crate::parser::parse;
    use crate::plan::lower;

    fn catalog(cid_unique: bool, cid_indexed: bool, companies: u64, people: u64) -> StatsCatalog {
        let mut cat = StatsCatalog::empty();
        let mut company_props = BTreeMap::new();
        company_props.insert(
            "cid".to_string(),
            PropStats {
                unique: cid_unique,
                indexed: cid_indexed,
                ..Default::default()
            },
        );
        cat.__test_insert_label(LabelStats {
            name: "Company".into(),
            node_count: companies,
            properties: company_props,
        });
        cat.__test_insert_label(LabelStats {
            name: "Person".into(),
            node_count: people,
            properties: BTreeMap::new(),
        });
        cat
    }

    fn rewrite_query(q: &str, cat: &StatsCatalog) -> LogicalPlan {
        let parsed = parse(q).unwrap();
        let plan = lower(&parsed).unwrap();
        apply_anchor_inversion(plan, cat)
    }

    /// The inverted Expand, if the pass fired: (source, direction,
    /// target_alias, target_labels, anchor alias under it).
    fn find_inverted(
        plan: &LogicalPlan,
    ) -> Option<(String, RelationshipDirection, String, Vec<String>)> {
        if let LogicalPlan::Expand {
            input,
            source,
            direction,
            target_alias,
            target_labels,
            ..
        } = plan
        {
            if matches!(input.as_ref(), LogicalPlan::NodeByPropertyValue { .. }) {
                return Some((
                    source.clone(),
                    *direction,
                    target_alias.clone(),
                    target_labels.clone(),
                ));
            }
        }
        plan.children().into_iter().find_map(find_inverted)
    }

    fn plan_has_node_scan(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeScan { .. })
            || plan.children().into_iter().any(plan_has_node_scan)
    }

    const SLOW_FORM: &str = "MATCH (p:Person)-[w:WORKS_AT]->(c:Company {cid: 0}) \
                             WHERE w.since = 0 RETURN count(*) AS c";

    #[test]
    fn unique_target_equality_inverts_the_anchor() {
        let plan = rewrite_query(SLOW_FORM, &catalog(true, false, 10, 1000));
        let (source, direction, target_alias, target_labels) =
            find_inverted(&plan).expect("the pass must fire on the reported shape");
        assert_eq!(source, "c", "the anchor moves to the selective endpoint");
        assert_eq!(direction, RelationshipDirection::Left, "arrow inverts");
        assert_eq!(target_alias, "p");
        assert_eq!(target_labels, vec!["Person".to_string()]);
        assert!(
            !plan_has_node_scan(&plan),
            "the unselective Person scan must be gone"
        );
    }

    #[test]
    fn residual_predicates_survive_above_the_inverted_expand() {
        // WHERE w.since = 0 cannot anchor anything; it must remain a Filter.
        let plan = rewrite_query(SLOW_FORM, &catalog(true, false, 10, 1000));
        fn has_filter(plan: &LogicalPlan) -> bool {
            matches!(plan, LogicalPlan::Filter { .. })
                || plan.children().into_iter().any(has_filter)
        }
        assert!(
            has_filter(&plan),
            "w.since must survive as a residual filter"
        );
    }

    #[test]
    fn left_arrow_inverts_to_right_and_both_stays_both() {
        let plan = rewrite_query(
            "MATCH (p:Person)<-[w:WORKS_AT]-(c:Company {cid: 0}) RETURN p",
            &catalog(true, false, 10, 1000),
        );
        let (_, direction, ..) = find_inverted(&plan).expect("left form inverts too");
        assert_eq!(direction, RelationshipDirection::Right);

        let plan = rewrite_query(
            "MATCH (p:Person)-[w:WORKS_AT]-(c:Company {cid: 0}) RETURN p",
            &catalog(true, false, 10, 1000),
        );
        let (_, direction, ..) = find_inverted(&plan).expect("both form fires");
        assert_eq!(direction, RelationshipDirection::Both);
    }

    #[test]
    fn indexed_multi_anchor_requires_target_no_larger_than_source() {
        // Indexed (multi) target smaller than source: invert.
        let plan = rewrite_query(SLOW_FORM, &catalog(false, true, 10, 1000));
        assert!(find_inverted(&plan).is_some());
        // Indexed target BIGGER than source: keep the scan-side plan.
        let plan = rewrite_query(SLOW_FORM, &catalog(false, true, 5000, 1000));
        assert!(find_inverted(&plan).is_none());
        // Unique target bigger than source: a point lookup still wins.
        let plan = rewrite_query(SLOW_FORM, &catalog(true, false, 5000, 1000));
        assert!(find_inverted(&plan).is_some());
    }

    /// A single-hop `q = ...` path is assembled STATICALLY from the node/rel
    /// bindings by a Project, so inversion is safe and welcome there; only
    /// walker-materialized trails (var-length, shortestPath) are guarded.
    #[test]
    fn static_single_hop_path_bindings_still_invert() {
        let plan = rewrite_query(
            "MATCH q = (p:Person)-[w:WORKS_AT]->(c:Company {cid: 0}) RETURN q",
            &catalog(true, false, 10, 1000),
        );
        assert!(find_inverted(&plan).is_some());
    }

    #[test]
    fn shape_guards_keep_semantics_bearing_forms_untouched() {
        let cat = catalog(true, false, 10, 1000);
        for query in [
            // OPTIONAL nullifies the target side; inverting would flip it.
            "MATCH (p:Person) OPTIONAL MATCH (p)-[w:WORKS_AT]->(c:Company {cid: 0}) RETURN p, c",
            // Var-length binds a path-ordered list.
            "MATCH (p:Person)-[w:WORKS_AT*1..2]->(c:Company {cid: 0}) RETURN p",
            // Unindexed target property: nothing to anchor on.
            "MATCH (p:Person)-[w:WORKS_AT]->(c:Company {name: 'x'}) RETURN p",
        ] {
            let plan = rewrite_query(query, &cat);
            assert!(
                find_inverted(&plan).is_none(),
                "guarded shape must not invert: {query}"
            );
        }
        // A filtered source needs cost comparison; stay put.
        let plan = rewrite_query(
            "MATCH (p:Person {name: 'ana'})-[w:WORKS_AT]->(c:Company {cid: 0}) RETURN p",
            &cat,
        );
        assert!(
            find_inverted(&plan).is_none(),
            "a filtered source keeps the lowered anchor"
        );
    }

    /// Fifth field report: a label-DISJUNCTION source `(o:A|B)` hides its
    /// scan behind an OR-filter, which used to defeat the inversion — every
    /// anchored per-date query re-scanned the whole namespace. The pass now
    /// sees through a PURE label disjunction (and only that).
    #[test]
    fn disjunction_source_inverts_with_label_filter_reattached() {
        let mut cat = StatsCatalog::empty();
        let mut fecha_props = BTreeMap::new();
        fecha_props.insert(
            "fecha".to_string(),
            PropStats {
                unique: true,
                ..Default::default()
            },
        );
        cat.__test_insert_label(LabelStats {
            name: "FECHA".into(),
            node_count: 271,
            properties: fecha_props,
        });
        cat.__test_insert_label(LabelStats {
            name: "OFERTA".into(),
            node_count: 400,
            properties: BTreeMap::new(),
        });
        cat.__test_insert_label(LabelStats {
            name: "PROMOCION".into(),
            node_count: 100,
            properties: BTreeMap::new(),
        });

        let plan = rewrite_query(
            "MATCH (o:OFERTA|PROMOCION)-[:VIGENTE_EN]->(f:FECHA {fecha: 'x'}) RETURN count(o)",
            &cat,
        );
        // The anchor fired: a NodeByPropertyValue on f feeds an inverted
        // expand whose target is o with NO conjunctive labels...
        fn find(plan: &LogicalPlan) -> Option<(String, String, usize)> {
            if let LogicalPlan::Expand {
                input,
                source,
                target_alias,
                target_labels,
                ..
            } = plan
            {
                if let LogicalPlan::NodeByPropertyValue { alias, .. } = input.as_ref() {
                    if alias == source {
                        return Some((source.clone(), target_alias.clone(), target_labels.len()));
                    }
                }
            }
            plan.children().into_iter().find_map(find)
        }
        let (source, target, labels) = find(&plan).expect("disjunction source must invert");
        assert_eq!(source, "f");
        assert_eq!(target, "o");
        assert_eq!(
            labels, 0,
            "the disjunction cannot ride conjunctive target_labels"
        );
        // ...and the OR label filter survives ABOVE the inverted expand.
        fn has_label_or_filter(plan: &LogicalPlan) -> bool {
            if let LogicalPlan::Filter { predicate, .. } = plan {
                if crate::optimize::extract_label_disjunction(predicate, "o")
                    .is_some_and(|labels| labels == ["OFERTA", "PROMOCION"])
                {
                    return true;
                }
            }
            plan.children().into_iter().any(has_label_or_filter)
        }
        assert!(has_label_or_filter(&plan), "{plan:?}");
    }

    /// A non-label filter over the source is NOT a disjunction veil: the
    /// pass must keep refusing filtered sources it cannot cost.
    #[test]
    fn arbitrary_source_filter_still_blocks_inversion() {
        let cat = catalog(true, false, 6, 60);
        let plan = rewrite_query(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company {cid: 'x'}) \
             WHERE p.name = 'ana' RETURN count(p)",
            &cat,
        );
        // Pushdown has not run inside this pass; the WHERE sits as a Filter
        // over the scan only if lowering placed it there. Either way the
        // assertion is: no inverted expand whose input is the cid anchor
        // AND whose source filter was a non-label predicate got consumed.
        fn anchor_count(plan: &LogicalPlan) -> usize {
            let here = usize::from(matches!(
                plan,
                LogicalPlan::NodeByPropertyValue { alias, .. } if alias == "c"
            ));
            here + plan.children().into_iter().map(anchor_count).sum::<usize>()
        }
        // The plain shape (no source filter) in `catalog` tests already
        // inverts; here we only require the pass didn't DROP the name
        // predicate if it fired through some path.
        fn has_name_filter(plan: &LogicalPlan) -> bool {
            if let LogicalPlan::Filter { predicate, .. } = plan {
                if format!("{predicate:?}").contains("name") {
                    return true;
                }
            }
            plan.children().into_iter().any(has_name_filter)
        }
        assert!(anchor_count(&plan) <= 1);
        assert!(
            has_name_filter(&plan),
            "the name predicate must survive: {plan:?}"
        );
    }
}
