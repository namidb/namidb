//! Cost-based choice between an id anchor and a label scan.
//!
//! `WHERE elementId(d) IN $ids` (or `id(d) = $id`) turns `d` into a point
//! lookup, and lowering then walks the pattern away from it. That is the
//! right start for an ordinary node, and the wrong one for a hub: anchored
//! at a node with 1.7M incoming `ON_DEVICE` edges,
//! `MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $ids`
//! reads and label-checks every one of those neighbours to keep the few
//! thousand that are `:Call`, while scanning the 2,958 `:Call` nodes and
//! following their single outgoing edge each touches a few thousand rows.
//!
//! This module holds the cost comparison and the rewrite that uses it:
//!
//! - [`id_anchor_wins`] compares the rows the two starts touch, the C_out
//!   sum (every row each step produces, RFC-010) of the walk from the id
//!   anchor against the walk from the cheapest label scan, from the per
//!   edge-type, per-direction average degrees of the statistics catalog.
//!   The anchor keeps winning unless the scan is clearly cheaper
//!   ([`SCAN_MARGIN`]) and the catalog knows every number involved: with no
//!   statistics nothing changes. Anchor inversion uses it before it
//!   re-anchors a scan at an id-pinned target, so the two passes agree and
//!   cannot undo each other.
//! - [`apply_hub_deanchor`] rewrites a single hop that starts at an id seek
//!   into a label scan of the hop's target plus the inverted hop back to the
//!   pinned node, with the id predicate as a filter on top, when
//!   [`id_anchor_wins`] says so.
//!
//! The statistics are averages, so the choice is per edge type and
//! direction, not per node: a store with many ordinary nodes and a few hubs
//! of the same label averages the hubs away. In the shapes this targets
//! (a handful of hubs holding most of an edge type) the average is the hub
//! degree.

use crate::cost::cardinality::branch_factor;
use crate::cost::StatsCatalog;
use crate::parser::ast::{Expression, ExpressionKind};
use crate::parser::RelationshipDirection;
use crate::plan::{LogicalPlan, ShortestMode};

use super::{and_chain, split_and_terms};

/// The scan must touch fewer than `1 / SCAN_MARGIN` of the rows the id
/// anchor touches before it replaces the anchor. An id lookup is a point
/// read, a label scan decodes every row it emits, and the estimate is only
/// an average; a factor of four keeps every borderline case on the plan the
/// query spelled.
pub(crate) const SCAN_MARGIN: f64 = 4.0;

/// One hop of a walk, as the cost model needs it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CostHop<'a> {
    pub edge_type: Option<&'a [String]>,
    /// Direction of travel when walking AWAY from the id anchor.
    pub direction_from_anchor: RelationshipDirection,
}

pub(crate) fn invert(direction: RelationshipDirection) -> RelationshipDirection {
    match direction {
        RelationshipDirection::Right => RelationshipDirection::Left,
        RelationshipDirection::Left => RelationshipDirection::Right,
        RelationshipDirection::Both => RelationshipDirection::Both,
    }
}

/// Rows the id anchor contributes, from its id expression: the length of a
/// literal list, else one (a parameter list of unknown length counts as one
/// id, which favours the anchor).
pub(crate) fn id_rows(value: &Expression, many: bool) -> f64 {
    match &value.kind {
        ExpressionKind::List(items) if many => items.len().max(1) as f64,
        _ => 1.0,
    }
}

fn known_degree(
    catalog: &StatsCatalog,
    edge_type: Option<&[String]>,
    direction: RelationshipDirection,
) -> Option<f64> {
    // Only typed hops whose every type the catalog has measured: a missing
    // entry would fall back to a made-up default fan-out.
    let types = edge_type.filter(|types| !types.is_empty())?;
    if types.iter().any(|t| catalog.edge_type(t).is_none()) {
        return None;
    }
    Some(branch_factor(catalog, Some(types), direction))
}

/// C_out of a walk: `start` rows, then each hop multiplies by its degree.
fn walk_cost<'h>(
    catalog: &StatsCatalog,
    start: f64,
    hops: impl Iterator<Item = (Option<&'h [String]>, RelationshipDirection)>,
) -> Option<f64> {
    let mut rows = start;
    let mut cost = start;
    for (edge_type, direction) in hops {
        rows *= known_degree(catalog, edge_type, direction)?;
        cost += rows;
    }
    Some(cost)
}

/// `true` when starting at the id anchor is the plan to keep: the walk from
/// `anchor_rows` ids along `hops` (nearest the anchor first) costs no more
/// than [`SCAN_MARGIN`] times the walk from `scan_rows` nodes back along the
/// same hops, or when the catalog lacks a number either side needs.
pub(crate) fn id_anchor_wins(
    catalog: &StatsCatalog,
    anchor_rows: f64,
    hops: &[CostHop<'_>],
    scan_rows: Option<u64>,
) -> bool {
    let Some(scan_rows) = scan_rows else {
        return true;
    };
    let anchor = walk_cost(
        catalog,
        anchor_rows,
        hops.iter().map(|h| (h.edge_type, h.direction_from_anchor)),
    );
    let scan = walk_cost(
        catalog,
        scan_rows as f64,
        hops.iter()
            .rev()
            .map(|h| (h.edge_type, invert(h.direction_from_anchor))),
    );
    match (anchor, scan) {
        (Some(anchor), Some(scan)) => scan * SCAN_MARGIN >= anchor,
        _ => true,
    }
}

/// The node count of the label `catalog` knows to be smallest among
/// `labels`, with that label. `None` when no label is known.
pub(crate) fn cheapest_known_label<'l>(
    labels: &'l [String],
    catalog: &StatsCatalog,
) -> Option<(&'l String, u64)> {
    labels
        .iter()
        .filter_map(|l| catalog.label(l).map(|s| (l, s.node_count)))
        .min_by_key(|(_, n)| *n)
}

/// Rewrite every id-anchored hop the cost model says should start from the
/// other side. See the module docs.
pub fn apply_hub_deanchor(mut plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    visit(&mut plan, catalog);
    plan
}

fn visit(plan: &mut LogicalPlan, catalog: &StatsCatalog) {
    for child in plan.children_mut() {
        visit(child, catalog);
    }
    if let Some(rewritten) = deanchor_one(plan, catalog) {
        *plan = rewritten;
    }
}

/// An id seek as `unique_lookup::id_seek` builds it: `Filter` (holding the
/// exact `call = key` check and maybe other conjuncts on the alias) over
/// `NodeById`, over `Empty` for one id or `Distinct(Unwind(list))` for `IN`.
struct SeekShape {
    alias: String,
    label: Option<String>,
    /// The original predicate on the alias: `call = value` / `call IN list`
    /// plus every other conjunct of the filter.
    predicate: Expression,
    rows: f64,
}

fn match_seek(plan: &LogicalPlan) -> Option<SeekShape> {
    let LogicalPlan::Filter { input, predicate } = plan else {
        return None;
    };
    let LogicalPlan::NodeById {
        input: seek_input,
        label,
        alias,
        id,
    } = input.as_ref()
    else {
        return None;
    };
    // `Some(list)` for the `IN` form.
    let list = match seek_input.as_ref() {
        LogicalPlan::Empty => None,
        LogicalPlan::Distinct { input } => match input.as_ref() {
            LogicalPlan::Unwind {
                input,
                list,
                alias: slot,
            } if matches!(input.as_ref(), LogicalPlan::Empty)
                && matches!(&id.kind, ExpressionKind::Variable(v) if v.name == *slot) =>
            {
                Some(list.clone())
            }
            _ => return None,
        },
        _ => return None,
    };
    let mut found = false;
    let mut terms = Vec::new();
    for term in split_and_terms(predicate) {
        let exact = match &term.kind {
            ExpressionKind::Binary {
                op: crate::parser::ast::BinaryOp::Eq,
                left,
                right,
            } if !found && right.as_ref() == id && is_id_call(left, alias) => Some(left.clone()),
            _ => None,
        };
        match exact {
            Some(call) => {
                found = true;
                let span = term.span;
                terms.push(match &list {
                    None => term,
                    Some(list) => Expression {
                        kind: ExpressionKind::In {
                            item: call,
                            list: Box::new(list.clone()),
                        },
                        span,
                    },
                });
            }
            None => {
                // Anything else must be about the anchor alone: the hidden
                // seek slot disappears with the rewrite.
                let aliases = super::expression_aliases(&term);
                if aliases.iter().any(|a| a != alias) {
                    return None;
                }
                terms.push(term);
            }
        }
    }
    if !found {
        return None;
    }
    let rows = match &list {
        Some(list) => id_rows(list, true),
        None => 1.0,
    };
    Some(SeekShape {
        alias: alias.clone(),
        label: label.clone(),
        predicate: and_chain(terms)?,
        rows,
    })
}

fn is_id_call(expr: &Expression, alias: &str) -> bool {
    let ExpressionKind::FunctionCall { name, args, .. } = &expr.kind else {
        return false;
    };
    let fname = name.joined().to_ascii_lowercase();
    args.len() == 1
        && (fname == "elementid" || fname == "id")
        && matches!(&args[0].kind, ExpressionKind::Variable(v) if v.name == alias)
}

fn deanchor_one(plan: &LogicalPlan, catalog: &StatsCatalog) -> Option<LogicalPlan> {
    let LogicalPlan::Expand {
        input,
        source,
        edge_type,
        direction,
        rel_alias,
        target_alias,
        target_labels,
        length: None,
        optional: false,
        back_reference: false,
        shortest: ShortestMode::None,
        path_binding: None,
        unique_rels,
    } = plan
    else {
        return None;
    };
    // The bottom hop of its chain: nothing below it to stay unique against.
    if !unique_rels.is_empty() {
        return None;
    }
    let seek = match_seek(input)?;
    if seek.alias != *source || target_alias == source {
        return None;
    }
    let (scan_label, scan_rows) = cheapest_known_label(target_labels, catalog)?;
    let hop = CostHop {
        edge_type: edge_type.as_deref(),
        direction_from_anchor: *direction,
    };
    if id_anchor_wins(catalog, seek.rows, &[hop], Some(scan_rows)) {
        return None;
    }
    let mut scan = LogicalPlan::NodeScan {
        label: Some(scan_label.clone()),
        alias: target_alias.clone(),
        predicates: Vec::new(),
        projection: None,
    };
    let other_labels: Vec<Expression> = target_labels
        .iter()
        .filter(|l| *l != scan_label)
        .map(|l| crate::plan::lower::build_label_eq(target_alias, l, seek.predicate.span))
        .collect();
    if let Some(predicate) = and_chain(other_labels) {
        scan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate,
        };
    }
    let walk = LogicalPlan::Expand {
        input: Box::new(scan),
        source: target_alias.clone(),
        edge_type: edge_type.clone(),
        direction: invert(*direction),
        rel_alias: rel_alias.clone(),
        target_alias: seek.alias.clone(),
        target_labels: seek.label.iter().cloned().collect(),
        length: None,
        optional: false,
        back_reference: false,
        shortest: ShortestMode::None,
        path_binding: None,
        unique_rels: Vec::new(),
    };
    Some(LogicalPlan::Filter {
        input: Box::new(walk),
        predicate: seek.predicate,
    })
}

#[cfg(test)]
mod tests {
    use crate::cost::{EdgeTypeStats, LabelStats, StatsCatalog};
    use crate::plan::explain;

    /// The reported shape: 4 `:Device` hubs holding every `ON_DEVICE` edge
    /// (one per non-device node), 2,958 `:Call` nodes.
    fn catalog(avg_in_degree: f64) -> StatsCatalog {
        let mut cat = StatsCatalog::empty();
        for (name, count) in [
            ("Device", 4),
            ("Call", 2_958),
            ("Recorded", 100_000),
            ("Contact", 400_000),
            ("Message", 1_400_000),
        ] {
            cat.__test_insert_label(LabelStats {
                name: name.into(),
                node_count: count,
                properties: Default::default(),
            });
        }
        cat.__test_insert_edge_type(EdgeTypeStats {
            name: "ON_DEVICE".into(),
            edge_count: 2_399_996,
            avg_out_degree: 1.0,
            max_out_degree: 1,
            avg_in_degree,
            max_in_degree: 1_700_000,
            src_label: None,
            dst_label: None,
        });
        cat.__test_insert_edge_type(EdgeTypeStats {
            name: "TO".into(),
            edge_count: 2_958,
            avg_out_degree: 1.0,
            max_out_degree: 1,
            avg_in_degree: 1.0,
            max_in_degree: 3,
            src_label: None,
            dst_label: None,
        });
        cat
    }

    fn hubs() -> StatsCatalog {
        catalog(600_000.0)
    }

    fn plan_with(q: &str, cat: &StatsCatalog) -> String {
        let parsed = crate::parse(q).unwrap();
        explain(&crate::plan(&parsed, cat).unwrap())
    }

    fn assert_starts_from_call_scan(q: &str, cat: &StatsCatalog) -> String {
        let p = plan_with(q, cat);
        assert!(!p.contains("NodeById"), "{q}\n{p}");
        assert!(p.contains("NodeScan label=Call alias=c"), "{q}\n{p}");
        assert!(
            p.contains("Expand source=c edge_type=ON_DEVICE dir=->"),
            "{q}\n{p}"
        );
        assert!(p.contains(" target=d"), "{q}\n{p}");
        p
    }

    #[test]
    fn hub_id_anchor_starts_from_the_small_label_instead() {
        for q in [
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs RETURN c.ts",
            "MATCH (c:Call)-[:ON_DEVICE]->(d:Device) WHERE elementId(d) IN $hubs RETURN c.ts",
            "MATCH (d)<-[:ON_DEVICE]-(c) WHERE elementId(d) IN $hubs AND c:Call RETURN c.ts",
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) = $hub RETURN c.ts",
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE id(d) = $hub RETURN c",
            "MATCH (d:Device)<-[r:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs RETURN r, c, d",
        ] {
            let p = assert_starts_from_call_scan(q, &hubs());
            assert!(
                p.contains("Filter ((elementId(d) ") || p.contains("Filter ((id(d) "),
                "{q}\n{p}"
            );
        }
    }

    #[test]
    fn the_id_predicate_survives_with_its_list_and_other_anchor_conjuncts() {
        let p = assert_starts_from_call_scan(
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) \
             WHERE elementId(d) IN $hubs AND d.name <> 'x' RETURN c.ts",
            &hubs(),
        );
        assert!(p.contains("elementId(d) IN $hubs"), "{p}");
        assert!(p.contains("d.name"), "{p}");
        assert!(!p.contains("__idseek"), "{p}");
        assert!(p.contains("target=d:Device"), "{p}");
    }

    #[test]
    fn ordinary_degree_keeps_the_id_anchor() {
        // Average in-degree 3: the anchor touches a handful of rows.
        for q in [
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs RETURN c.ts",
            "MATCH (c:Call)-[:ON_DEVICE]->(d:Device) WHERE elementId(d) = $hub RETURN c.ts",
        ] {
            let p = plan_with(q, &catalog(3.0));
            assert!(p.contains("NodeById label=Device alias=d"), "{q}\n{p}");
            assert!(!p.contains("NodeScan label=Call"), "{q}\n{p}");
        }
    }

    #[test]
    fn missing_statistics_keep_the_id_anchor() {
        let mut cat = StatsCatalog::empty();
        for (name, count) in [("Device", 4), ("Call", 2_958)] {
            cat.__test_insert_label(LabelStats {
                name: name.into(),
                node_count: count,
                properties: Default::default(),
            });
        }
        let p = plan_with(
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs RETURN c.ts",
            &cat,
        );
        assert!(p.contains("NodeById"), "{p}");
        // And an unknown target label leaves nothing to scan.
        let p = plan_with(
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Unknown) WHERE elementId(d) IN $hubs RETURN c.ts",
            &hubs(),
        );
        assert!(p.contains("NodeById"), "{p}");
    }

    #[test]
    fn an_untyped_or_unlabelled_hop_keeps_the_id_anchor() {
        for q in [
            "MATCH (d:Device)<--(c:Call) WHERE elementId(d) IN $hubs RETURN c.ts",
            "MATCH (d:Device)<-[:ON_DEVICE]-(c) WHERE elementId(d) IN $hubs RETURN c.ts",
        ] {
            let p = plan_with(q, &hubs());
            assert!(p.contains("NodeById"), "{q}\n{p}");
        }
    }

    #[test]
    fn a_literal_id_list_is_costed_by_its_length() {
        // 4 ids × 600k vs 2,958 calls: scan.
        let p = plan_with(
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN ['a', 'b', 'c', 'd'] \
             RETURN c.ts",
            &hubs(),
        );
        assert!(p.contains("NodeScan label=Call"), "{p}");
        // 3 ids × 1,000 = 3,000 rows vs 2,958 × 2 × margin 4: anchor.
        let p = plan_with(
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN ['a', 'b', 'c'] \
             RETURN c.ts",
            &catalog(1_000.0),
        );
        assert!(p.contains("NodeById"), "{p}");
    }

    #[test]
    fn a_multi_label_target_scans_its_smallest_label() {
        let p = plan_with(
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Recorded:Call) WHERE elementId(d) IN $hubs \
             RETURN c.ts",
            &hubs(),
        );
        assert!(p.contains("NodeScan label=Call alias=c"), "{p}");
        assert!(
            p.contains("__label_eq(c, 'Recorded')") || p.contains("Recorded"),
            "{p}"
        );
    }

    #[test]
    fn longer_chains_start_from_the_scan_too() {
        let p = plan_with(
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call)-[:TO]->(x:Contact) \
             WHERE elementId(d) IN $hubs RETURN x",
            &hubs(),
        );
        assert!(!p.contains("NodeById"), "{p}");
        assert!(p.contains("NodeScan label=Call alias=c"), "{p}");
        assert!(
            p.contains("Expand source=c edge_type=TO dir=-> target=x:Contact"),
            "{p}"
        );
    }

    #[test]
    fn optional_match_from_the_anchor_is_left_alone() {
        let p = plan_with(
            "MATCH (d:Device) WHERE elementId(d) IN $hubs \
             OPTIONAL MATCH (d)<-[:ON_DEVICE]-(c:Call) RETURN d, c",
            &hubs(),
        );
        assert!(p.contains("NodeById"), "{p}");
    }

    #[test]
    fn deanchoring_is_a_fixpoint() {
        for q in [
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs RETURN c.ts",
            "MATCH (c:Call)-[:ON_DEVICE]->(d:Device) WHERE elementId(d) = $hub RETURN c",
            "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call)-[:TO]->(x:Contact) \
             WHERE elementId(d) IN $hubs RETURN x",
        ] {
            let parsed = crate::parse(q).unwrap();
            let once = crate::plan(&parsed, &hubs()).unwrap();
            let twice = crate::optimize(once.clone(), &hubs());
            assert_eq!(once, twice, "{q}");
        }
    }
}
