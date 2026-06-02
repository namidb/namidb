//! Edge-type-count pushdown.
//!
//! A global `count(*)` / `count(r)` over a directed, single-hop,
//! unfiltered typed `Expand` whose source is an unfiltered `NodeScan`
//! does not need to visit a single node: the answer is exactly the
//! number of live edges of that type. This pass rewrites that shape into
//! a [`LogicalPlan::EdgeTypeCount`] leaf, which the executor answers with
//! [`namidb_storage::read::Snapshot::count_edge_type`] (memtable + edge
//! SSTs, merged, tombstones pruned) instead of `NodeScan + Expand +
//! Aggregate` over the whole node set.
//!
//! ## Why it is correct
//!
//! For a directed pattern `()-[r:T]->()` (or `()<-[r:T]-()`) every edge
//! of type `T` has exactly one source (resp. target) node, so scanning
//! all nodes and expanding visits each edge exactly once. Hence
//! `count(*) == count(r) == count(DISTINCT r) ==` the live-edge count.
//! Undirected (`-`) is excluded: Cypher emits each edge in both
//! directions, so its count is not the edge count. Alternation
//! `[:A|:B]` sums per-type counts; an edge belongs to exactly one type
//! so the per-type counts are disjoint.
//!
//! ## What disables it (falls back to the ordinary plan)
//!
//! Anything that makes the row count differ from the live-edge count: a
//! labelled or predicated source `NodeScan`, a target label, a non-empty
//! `WHERE`, variable-length (`[*..]`), `OPTIONAL`, `shortestPath`, an
//! untyped edge, a back-referenced target, a `GROUP BY`, more than one
//! aggregation, or a `count` argument that is not the relationship
//! binding. The pass only descends linear result-shaping wrappers
//! (`Project` / `TopN` / `Distinct`) from the root, so it never rewrites
//! inside a correlated subplan where the global count would be wrong.

use crate::parser::ast::ExpressionKind;
use crate::parser::{Expression, RelationshipDirection};
use crate::plan::logical::{AggregateExpr, LogicalPlan, ShortestMode};

/// Rewrite eligible global edge-type counts into [`LogicalPlan::EdgeTypeCount`].
pub fn apply_edge_count_pushdown(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        // Descend the linear result-shaping wrappers lowering puts above
        // the Aggregate (RETURN projection, ORDER BY, DISTINCT). These do
        // not change the global-count semantics.
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(apply_edge_count_pushdown(*input)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(apply_edge_count_pushdown(*input)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(apply_edge_count_pushdown(*input)),
        },
        // The pattern itself.
        agg @ LogicalPlan::Aggregate { .. } => match try_match(&agg) {
            Some(rewritten) => rewritten,
            None => agg,
        },
        // Any other operator: leave it (and everything below) untouched.
        other => other,
    }
}

/// Return `Some(EdgeTypeCount)` if `plan` is an eligible global edge-type
/// count, else `None`.
fn try_match(plan: &LogicalPlan) -> Option<LogicalPlan> {
    let LogicalPlan::Aggregate {
        input,
        group_by,
        aggregations,
    } = plan
    else {
        return None;
    };
    // Global, single COUNT only.
    if !group_by.is_empty() || aggregations.len() != 1 {
        return None;
    }
    let (output, agg) = &aggregations[0];
    let AggregateExpr::Count { arg, .. } = agg else {
        return None;
    };

    // Directed, single-hop, unfiltered, plain typed Expand.
    let LogicalPlan::Expand {
        input: expand_input,
        source,
        edge_type,
        direction,
        rel_alias,
        target_labels,
        length,
        optional,
        back_reference,
        shortest,
        path_binding,
        ..
    } = input.as_ref()
    else {
        return None;
    };
    if *optional
        || *back_reference
        || path_binding.is_some()
        || length.is_some()
        || !target_labels.is_empty()
        || !matches!(shortest, ShortestMode::None)
    {
        return None;
    }
    // Undirected (`Both`) counts each edge twice in Cypher; only `->`/`<-`
    // map to the live-edge count.
    if !matches!(
        direction,
        RelationshipDirection::Left | RelationshipDirection::Right
    ) {
        return None;
    }
    let Some(edge_types) = edge_type else {
        return None;
    };
    if edge_types.is_empty() {
        return None;
    }
    // `count(*)` or `count(<rel_alias>)`. A count over a node binding or
    // an arbitrary expression is not the edge count.
    if let Some(expr) = arg {
        let Some(ra) = rel_alias else {
            return None;
        };
        if !is_variable(expr, ra) {
            return None;
        }
    }

    // Source must be an unfiltered NodeScan over every node, and it must
    // be the node the Expand starts from.
    let LogicalPlan::NodeScan {
        label,
        alias,
        predicates,
        ..
    } = expand_input.as_ref()
    else {
        return None;
    };
    if label.is_some() || !predicates.is_empty() || alias != source {
        return None;
    }

    Some(LogicalPlan::EdgeTypeCount {
        edge_types: edge_types.clone(),
        output: output.clone(),
    })
}

fn is_variable(expr: &Expression, name: &str) -> bool {
    matches!(&expr.kind, ExpressionKind::Variable(id) if id.name == name)
}

#[cfg(test)]
mod tests {
    use crate::cost::StatsCatalog;
    use crate::plan::logical::LogicalPlan;
    use crate::plan_cache::parse_lower_optimize;

    fn has_edge_type_count(plan: &LogicalPlan) -> Option<(Vec<String>, String)> {
        if let LogicalPlan::EdgeTypeCount { edge_types, output } = plan {
            return Some((edge_types.clone(), output.clone()));
        }
        plan.children().iter().find_map(|c| has_edge_type_count(c))
    }

    fn has_node_scan(plan: &LogicalPlan) -> bool {
        if matches!(plan, LogicalPlan::NodeScan { .. }) {
            return true;
        }
        plan.children().iter().any(|c| has_node_scan(c))
    }

    fn optimized(q: &str) -> LogicalPlan {
        parse_lower_optimize(q, &StatsCatalog::empty()).unwrap()
    }

    #[test]
    fn global_directed_typed_count_becomes_edge_type_count() {
        let plan = optimized("MATCH ()-[r:KNOWS]->() RETURN count(r) AS n");
        let (types, _out) =
            has_edge_type_count(&plan).expect("count over directed typed expand must push down");
        assert_eq!(types, vec!["KNOWS".to_string()]);
        assert!(!has_node_scan(&plan), "the NodeScan must be eliminated");
    }

    #[test]
    fn count_star_also_pushes_down() {
        let plan = optimized("MATCH ()-[:KNOWS]->() RETURN count(*) AS n");
        assert!(has_edge_type_count(&plan).is_some());
        assert!(!has_node_scan(&plan));
    }

    #[test]
    fn incoming_direction_pushes_down() {
        let plan = optimized("MATCH ()<-[r:KNOWS]-() RETURN count(r) AS n");
        assert!(has_edge_type_count(&plan).is_some());
    }

    #[test]
    fn alternation_lists_every_branch() {
        let plan = optimized("MATCH ()-[r:KNOWS|FOLLOWS]->() RETURN count(r) AS n");
        let (types, _) = has_edge_type_count(&plan).expect("alternation must push down");
        assert_eq!(types, vec!["KNOWS".to_string(), "FOLLOWS".to_string()]);
    }

    #[test]
    fn labelled_source_does_not_push_down() {
        // count of KNOWS edges FROM Person nodes != all KNOWS edges.
        let plan = optimized("MATCH (a:Person)-[r:KNOWS]->() RETURN count(r) AS n");
        assert!(has_edge_type_count(&plan).is_none());
        assert!(has_node_scan(&plan));
    }

    #[test]
    fn target_label_does_not_push_down() {
        let plan = optimized("MATCH ()-[r:KNOWS]->(b:Person) RETURN count(r) AS n");
        assert!(has_edge_type_count(&plan).is_none());
    }

    #[test]
    fn undirected_does_not_push_down() {
        // Cypher counts each undirected edge twice; the edge count would be wrong.
        let plan = optimized("MATCH ()-[r:KNOWS]-() RETURN count(r) AS n");
        assert!(has_edge_type_count(&plan).is_none());
    }

    #[test]
    fn untyped_does_not_push_down() {
        let plan = optimized("MATCH ()-[r]->() RETURN count(r) AS n");
        assert!(has_edge_type_count(&plan).is_none());
    }

    #[test]
    fn variable_length_does_not_push_down() {
        let plan = optimized("MATCH ()-[r:KNOWS*1..3]->() RETURN count(r) AS n");
        assert!(has_edge_type_count(&plan).is_none());
    }

    #[test]
    fn grouped_count_does_not_push_down() {
        // count per source node is not a global edge count.
        let plan = optimized("MATCH (a)-[r:KNOWS]->() RETURN a, count(r) AS n");
        assert!(has_edge_type_count(&plan).is_none());
    }
}
