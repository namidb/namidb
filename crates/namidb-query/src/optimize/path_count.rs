//! Aggregation pushdown through a linear chain: grouped path counts.
//!
//! `MATCH (d:Doc)<-[:AMENDS]-(x) RETURN d.title, count(x)` asks, per `Doc`,
//! for its number of incoming `AMENDS` edges — its in-degree. The naive plan
//! scans every `Doc`, probes each one's adjacency, materialises one row per
//! edge and groups them. On a whole graph that is one adjacency probe per
//! node and one row per edge, which is what kept such aggregations from
//! finishing. The same holds, multiplied, for multi-hop counts like
//! `MATCH (a:Org)-[:ISSUED]->(d:Doc)-[:HAS]->(s:Section) RETURN a.name,
//! count(s)`.
//!
//! This pass rewrites
//!
//! ```text
//! Aggregate(group = keys over ONE chain node g, aggs = count(*) / count(v))
//!   Expand(v_{k-1} -> v_k) ... Expand(v0 -> v1)
//!     NodeScan(v0)
//! ```
//!
//! into
//!
//! ```text
//! Aggregate(group = same keys, aggs = sum(__pathcount))
//!   PathCount(v0 .. vk, group = g)
//! ```
//!
//! `PathCount` (see `exec::path_count`) reads each hop's edge list once and
//! folds the number of chain matches through every node of `g`; it emits one
//! row per `g` node with a non-zero count. The `Aggregate` keeps the original
//! keys, so a key that maps several nodes to one group (`d.title`) still sums
//! correctly, and the output order and every downstream operator
//! (`ORDER BY c DESC LIMIT 5`) see the same rows as before.
//!
//! ## When it applies
//!
//! - every aggregation is a non-distinct `count(*)` or `count(<binding>)` of
//!   a chain node or relationship (never NULL in a non-optional match);
//! - every grouping key reads at most one chain NODE binding (none for a
//!   global count);
//! - the input is exactly a chain of single-hop, non-optional Expands (no
//!   variable length, no shortest path, no path binding, no back-reference)
//!   over a `NodeScan`, each Expand starting where the previous one ended;
//!   filters other than the scan's pushed predicates and the Expands' label
//!   checks decline;
//! - a chain of two or more hops has typed hops with pairwise disjoint type
//!   sets, so Cypher relationship uniqueness cannot remove a path the count
//!   would include;
//! - it is cheaper: with statistics, the edges `PathCount` reads must not
//!   exceed `READ_MARGIN` times the rows the naive walk produces; without
//!   statistics, only an unfiltered head scan qualifies (a filtered head is
//!   usually selective, and the naive walk from it is then the cheaper plan);
//! - not inside a correlated subplan (`Apply` / `SemiApply` / pattern list),
//!   where it would re-read whole edge types per outer row.
//!
//! `NAMIDB_PATH_COUNT=0` turns the rewrite off.

use std::collections::BTreeSet;

use crate::cost::cardinality::branch_factor;
use crate::cost::{estimate, StatsCatalog};
use crate::parser::ast::{Expression, ExpressionKind};
use crate::plan::logical::{AggregateExpr, LogicalPlan, PathCountHop, PathCountNode, ShortestMode};

use super::expression_aliases;

/// Column the rewritten `Aggregate` sums.
pub(crate) const PATH_COUNT_COLUMN: &str = "__pathcount";

/// `PathCount` reads every edge of each hop's types once; the naive walk
/// produces (and probes for) every row it emits. Reading an edge in a
/// sequential pass is far cheaper than producing a row, so the rewrite is
/// taken unless it would read more than this many edges per naive row.
const READ_MARGIN: f64 = 4.0;

fn enabled() -> bool {
    !std::env::var("NAMIDB_PATH_COUNT").is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// Rewrite every eligible grouped count. See the module docs.
pub fn apply_path_count(mut plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    if !enabled() {
        return plan;
    }
    visit(&mut plan, catalog);
    plan
}

fn visit(plan: &mut LogicalPlan, catalog: &StatsCatalog) {
    match plan {
        // Correlated subplans run once per outer row: only their input side
        // is rewritten.
        LogicalPlan::Apply { input, .. }
        | LogicalPlan::SemiApply { input, .. }
        | LogicalPlan::PatternList { input, .. } => {
            visit(input, catalog);
            return;
        }
        _ => {}
    }
    if let Some(rewritten) = try_rewrite(plan, catalog) {
        *plan = rewritten;
        return;
    }
    for child in plan.children_mut() {
        visit(child, catalog);
    }
}

struct Chain {
    nodes: Vec<PathCountNode>,
    hops: Vec<PathCountHop>,
    rel_aliases: Vec<String>,
    head_projection: Option<Vec<String>>,
}

/// Parse `Expand* (NodeScan)` into a chain, or `None`.
fn parse_chain(plan: &LogicalPlan) -> Option<Chain> {
    let mut expands = Vec::new();
    let mut cur = plan;
    let (head, head_projection) = loop {
        match cur {
            LogicalPlan::Expand {
                input,
                length: None,
                optional: false,
                back_reference: false,
                shortest: ShortestMode::None,
                path_binding: None,
                ..
            } => {
                expands.push(cur);
                cur = input;
            }
            LogicalPlan::NodeScan {
                label,
                alias,
                predicates,
                projection,
            } => {
                break (
                    PathCountNode {
                        alias: alias.clone(),
                        labels: label.iter().cloned().collect(),
                        predicates: predicates.clone(),
                    },
                    projection.clone(),
                )
            }
            _ => return None,
        }
    };
    if expands.is_empty() {
        return None;
    }
    expands.reverse();
    let mut nodes = vec![head];
    let mut hops = Vec::new();
    let mut rel_aliases = Vec::new();
    for e in expands {
        let LogicalPlan::Expand {
            source,
            edge_type,
            direction,
            rel_alias,
            target_alias,
            target_labels,
            unique_rels,
            ..
        } = e
        else {
            return None;
        };
        if source != &nodes.last()?.alias {
            return None;
        }
        if nodes.iter().any(|n| &n.alias == target_alias) {
            return None;
        }
        if matches!(edge_type, Some(ts) if ts.is_empty()) {
            return None;
        }
        // Relationship uniqueness may only name relationships of this
        // chain; with disjoint hop types (checked below) it never fires.
        if unique_rels.iter().any(|u| !rel_aliases.contains(u)) {
            return None;
        }
        if let Some(r) = rel_alias {
            rel_aliases.push(r.clone());
        }
        hops.push(PathCountHop {
            edge_type: edge_type.clone(),
            direction: *direction,
        });
        nodes.push(PathCountNode {
            alias: target_alias.clone(),
            labels: target_labels.clone(),
            predicates: Vec::new(),
        });
    }
    if hops.len() >= 2 {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for hop in &hops {
            let types = hop.edge_type.as_ref()?;
            for t in types {
                if !seen.insert(t.as_str()) {
                    return None;
                }
            }
        }
    }
    Some(Chain {
        nodes,
        hops,
        rel_aliases,
        head_projection,
    })
}

fn is_count_of_chain_binding(agg: &AggregateExpr, chain: &Chain) -> bool {
    match agg {
        AggregateExpr::Count {
            arg: None,
            distinct: false,
        } => true,
        AggregateExpr::Count {
            arg: Some(e),
            distinct: false,
        } => match &e.kind {
            ExpressionKind::Variable(id) => {
                chain.nodes.iter().any(|n| n.alias == id.name)
                    || chain.rel_aliases.contains(&id.name)
            }
            _ => false,
        },
        _ => false,
    }
}

/// Property projection the group node's keys need when it is materialised
/// by `PathCount` itself: `Some(props)` when every reference is `g.prop`,
/// `None` (whole node) otherwise.
fn group_projection(keys: &[(Expression, String)], alias: &str) -> Option<Vec<String>> {
    fn walk(expr: &Expression, alias: &str, props: &mut BTreeSet<String>) -> bool {
        match &expr.kind {
            ExpressionKind::Property(pa) => {
                if let ExpressionKind::Variable(id) = &pa.target.kind {
                    if id.name == alias {
                        props.insert(pa.key.name.clone());
                        return true;
                    }
                }
                walk(&pa.target, alias, props)
            }
            ExpressionKind::Variable(id) => id.name != alias,
            _ => {
                // Any other form: fine only if it does not mention the alias
                // at all (a constant or parameter); otherwise read the whole
                // node.
                !expression_aliases(expr).contains(alias)
            }
        }
    }
    let mut props = BTreeSet::new();
    for (e, _) in keys {
        if !walk(e, alias, &mut props) {
            return None;
        }
    }
    Some(props.into_iter().collect())
}

fn cheap_enough(chain: &Chain, input: &LogicalPlan, catalog: &StatsCatalog) -> bool {
    let head = &chain.nodes[0];
    let typed_known = chain.hops.iter().all(|h| {
        h.edge_type
            .as_ref()
            .is_some_and(|ts| ts.iter().all(|t| catalog.edge_type(t).is_some()))
    });
    let label_known = head.labels.iter().all(|l| catalog.label(l).is_some())
        && (catalog.total_nodes() > 0 || !head.labels.is_empty());
    if !(typed_known && label_known) {
        // Unknown statistics: only a whole-label / whole-graph head.
        return head.predicates.is_empty();
    }
    // Rows of the naive walk (C_out): head rows, then each hop multiplies.
    let mut cur = input;
    while let LogicalPlan::Expand { input, .. } = cur {
        cur = input;
    }
    let mut rows = estimate(cur, catalog).rows.max(1.0);
    let mut naive = rows;
    for hop in &chain.hops {
        rows *= branch_factor(catalog, hop.edge_type.as_deref(), hop.direction);
        naive += rows;
    }
    let read: f64 = chain
        .hops
        .iter()
        .flat_map(|h| h.edge_type.iter().flatten())
        .filter_map(|t| catalog.edge_type(t).map(|s| s.edge_count as f64))
        .sum();
    read <= naive * READ_MARGIN
}

fn try_rewrite(plan: &LogicalPlan, catalog: &StatsCatalog) -> Option<LogicalPlan> {
    let LogicalPlan::Aggregate {
        input,
        group_by,
        aggregations,
    } = plan
    else {
        return None;
    };
    if aggregations.is_empty() {
        return None;
    }
    let chain = parse_chain(input)?;
    if !aggregations
        .iter()
        .all(|(_, a)| is_count_of_chain_binding(a, &chain))
    {
        return None;
    }
    // The single chain node every key reads.
    let mut key_aliases: BTreeSet<String> = BTreeSet::new();
    for (e, _) in group_by {
        key_aliases.extend(expression_aliases(e));
    }
    let group = match key_aliases.len() {
        0 if group_by.is_empty() => None,
        // Constant keys only: a single group over the whole chain, but the
        // Aggregate still needs one row per path to carry the key. Rare;
        // decline rather than special-case.
        0 => return None,
        1 => {
            let alias = key_aliases.iter().next()?;
            Some(chain.nodes.iter().position(|n| &n.alias == alias)?)
        }
        _ => return None,
    };
    if !cheap_enough(&chain, input, catalog) {
        return None;
    }
    let projection = match group {
        Some(0) => chain.head_projection.clone(),
        Some(g) => group_projection(group_by, &chain.nodes[g].alias),
        None => Some(Vec::new()),
    };
    let sum_arg = Expression {
        kind: ExpressionKind::Variable(crate::parser::ast::Identifier {
            name: PATH_COUNT_COLUMN.to_string(),
            span: crate::parser::SourceSpan::point(0),
            quoted: false,
        }),
        span: crate::parser::SourceSpan::point(0),
    };
    let aggregations = aggregations
        .iter()
        .map(|(name, _)| {
            (
                name.clone(),
                AggregateExpr::Sum {
                    arg: sum_arg.clone(),
                    distinct: false,
                },
            )
        })
        .collect();
    Some(LogicalPlan::Aggregate {
        input: Box::new(LogicalPlan::PathCount {
            nodes: chain.nodes,
            hops: chain.hops,
            group,
            projection,
            output: PATH_COUNT_COLUMN.to_string(),
        }),
        group_by: group_by.clone(),
        aggregations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_cache::parse_lower_optimize;

    fn path_count(plan: &LogicalPlan) -> Option<(Vec<String>, Option<usize>)> {
        if let LogicalPlan::PathCount { nodes, group, .. } = plan {
            return Some((nodes.iter().map(|n| n.alias.clone()).collect(), *group));
        }
        plan.children().into_iter().find_map(path_count)
    }

    fn optimized(q: &str) -> LogicalPlan {
        parse_lower_optimize(q, &StatsCatalog::empty()).unwrap()
    }

    #[test]
    fn grouped_one_hop_count_reads_degrees() {
        let plan = optimized(
            "MATCH (d:Doc)<-[:AMENDS]-(x) RETURN d.title AS t, count(x) AS c ORDER BY c DESC LIMIT 5",
        );
        assert_eq!(
            path_count(&plan),
            Some((vec!["d".into(), "x".into()], Some(0)))
        );
    }

    #[test]
    fn group_key_on_the_far_node_and_multi_hop() {
        let plan = optimized("MATCH (s:Section)-[:OF]->(d:Doc) RETURN d.id AS id, count(s) AS c");
        assert_eq!(
            path_count(&plan),
            Some((vec!["s".into(), "d".into()], Some(1)))
        );
        let plan = optimized(
            "MATCH (a:Org)-[:ISSUED]->(d:Doc)-[:HAS]->(s:Section)-[:VERSION]->(v) \
             RETURN a.name AS a, count(v) AS c",
        );
        assert_eq!(
            path_count(&plan),
            Some((
                vec!["a".into(), "d".into(), "s".into(), "v".into()],
                Some(0)
            ))
        );
        let plan =
            optimized("MATCH (a:Org)-[:ISSUED]->(d:Doc)-[:HAS]->(s:Section) RETURN count(*) AS c");
        assert_eq!(path_count(&plan).map(|(_, g)| g), Some(None));
    }

    #[test]
    fn semantics_bearing_shapes_do_not_rewrite() {
        for q in [
            // A repeated type could reuse one relationship: uniqueness.
            "MATCH (a:Doc)-[:R]->(b)-[:R]->(c) RETURN a.k, count(*)",
            // DISTINCT, a nullable argument, a non-count aggregate.
            "MATCH (a:Doc)-[:R]->(b) RETURN a.k, count(DISTINCT b)",
            "MATCH (a:Doc)-[:R]->(b) RETURN a.k, count(b.k)",
            "MATCH (a:Doc)-[:R]->(b) RETURN a.k, collect(b)",
            // Keys over two chain nodes.
            "MATCH (a:Doc)-[:R]->(b) RETURN a.k, b.k, count(*)",
            // Optional / variable-length / filtered middle.
            "MATCH (a:Doc) OPTIONAL MATCH (a)-[:R]->(b) RETURN a.k, count(b)",
            "MATCH (a:Doc)-[:R*1..2]->(b) RETURN a.k, count(*)",
            "MATCH (a:Doc)-[:R]->(b)-[:S]->(c) WHERE b.k > 1 RETURN a.k, count(*)",
            // A filtered head with no statistics: the walk from it is cheaper.
            "MATCH (a:Doc) WHERE a.k = 1 MATCH (a)-[:R]->(b) RETURN a.k, count(*)",
        ] {
            assert_eq!(path_count(&optimized(q)), None, "must not rewrite: {q}");
        }
    }

    #[test]
    fn kill_switch_is_read_from_the_environment() {
        // Only the parser of the switch; flipping the process env in a
        // parallel test run would race other tests.
        assert!(enabled() || std::env::var("NAMIDB_PATH_COUNT").is_ok());
    }
}
