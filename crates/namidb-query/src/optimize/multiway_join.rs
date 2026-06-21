//! RFC-024: cycle detection + rewrite pass for worst-case optimal joins.
//!
//! Walks the plan top-down. At each level, if the subtree is a
//! contiguous `Expand` chain rooted at a `NodeScan` and the harvested
//! constraint graph has a cycle, replace it with a single
//! `LogicalPlan::MultiwayJoin` carrying the participating aliases,
//! edges, and an executor-friendly variable ordering. Otherwise recurse
//! into children.
//!
//! Gated by `NAMIDB_WCOJ=1`. When the flag is set but
//! `NAMIDB_FACTORIZE` is not, the pass logs a warning and skips
//! rewriting (the executor refuses to run a `MultiwayJoin` outside the
//! factor path; the binary plan is the safe fallback). RFC-024 §"Feature
//! flag matrix" wanted a hard `OptimizeError::ConfigurationConflict`
//! here, but that requires threading `Result` through every `optimize`
//! caller for a config check that's already user-actionable from a log
//! line — divergence noted on purpose.
//!
//! v0 preconditions for a chain to qualify (any failure drops the
//! rewrite for the subtree and recurses normally):
//!
//! - Chain is contiguous `Expand` operators with no `Filter` in between
//!   (after `normalize_filters` runs upstream, defensive label filters
//!   are gone — any remaining `Filter` carries user predicates that v0
//!   does not yet know how to relocate).
//! - Each `Expand` has `length == None` (single hop), a typed edge
//!   (`edge_type == Some(_)`), a non-`Both` direction, and no
//!   `rel_alias` (the executor does not materialise rel bindings yet).
//! - The chain terminates in a `NodeScan` with a declared label.
//! - Every participating alias has a known label (head's `label` or the
//!   Expand's `target_label`).
//! - `back_reference: true` is allowed and is what supplies the closing
//!   edge of the cycle; the executor treats it identically to any
//!   other constraint.

use std::collections::{BTreeMap, BTreeSet};

use crate::exec::factor::factorize_enabled;
use crate::parser::RelationshipDirection;
use crate::plan::logical::{EdgeConstraint, LogicalPlan, NodeBinding};

/// Read the `NAMIDB_WCOJ` env var. Empty / unset / "0" / "false" /
/// "no" / "off" → off. Any of "1" / "true" / "yes" / "on"
/// (case-insensitive) → on.
pub fn wcoj_enabled() -> bool {
    match std::env::var("NAMIDB_WCOJ") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// Public entry point. Returns `plan` unchanged when WCOJ is disabled
/// or when the FACTORIZE prerequisite is missing.
pub fn detect_multiway_join(plan: LogicalPlan) -> LogicalPlan {
    if !wcoj_enabled() {
        return plan;
    }
    if !factorize_enabled() {
        tracing::warn!("NAMIDB_WCOJ=1 requires NAMIDB_FACTORIZE=1; multiway join pass skipped");
        return plan;
    }
    rewrite(plan)
}

fn rewrite(plan: LogicalPlan) -> LogicalPlan {
    if let Some(rewritten) = try_rewrite_chain(&plan) {
        // MultiwayJoin is a leaf — nothing further to recurse into.
        return rewritten;
    }
    recurse_children(plan)
}

// ─────────────────────── chain harvest + rewrite ────────────────────────

/// Walk `plan` downward as long as it is an `Expand`. Collect the
/// participating aliases, labels, and edges. Returns `None` when the
/// chain does not satisfy the v0 preconditions or when the resulting
/// constraint graph has no cycle.
fn try_rewrite_chain(plan: &LogicalPlan) -> Option<LogicalPlan> {
    let head = match plan {
        LogicalPlan::Expand { .. } => plan,
        _ => return None,
    };

    // First pass — collect raw expand metadata top-down. We validate
    // per-operator preconditions here but defer alias/label assignment
    // until after we know the chain bottoms out in a NodeScan, because
    // the back-reference Expand at the top has `target_label: None`
    // and only validates against the alias the NodeScan introduces at
    // the bottom.
    let mut raw_expands: Vec<RawExpand> = Vec::new();
    let mut cursor = head;
    while let LogicalPlan::Expand {
        input,
        source,
        edge_type,
        direction,
        rel_alias,
        target_alias,
        target_labels,
        length,
        optional,
        back_reference: _,
        shortest,
        path_binding,
    } = cursor
    {
        if length.is_some()
            || edge_type.is_none()
            || rel_alias.is_some()
            || *optional
            || !matches!(shortest, crate::plan::logical::ShortestMode::None)
            || path_binding.is_some()
            || matches!(direction, RelationshipDirection::Both)
            // Multi-label targets need a conjunctive label check the leapfrog
            // executor does not do (it scans/confirms one label). Fall back to
            // the regular Expand executor, which enforces the full set.
            || target_labels.len() > 1
        {
            return None;
        }
        raw_expands.push(RawExpand {
            source: source.clone(),
            target: target_alias.clone(),
            target_label: target_labels.first().cloned(),
            edge_types: edge_type.as_ref().unwrap().clone(),
            direction: *direction,
        });
        cursor = input.as_ref();
    }

    // The chain terminates in a NodeScan with a declared label.
    let (head_alias, head_label, head_predicates) = match cursor {
        LogicalPlan::NodeScan {
            label: Some(label),
            alias,
            predicates,
            projection: _,
        } => (alias.clone(), label.clone(), predicates.clone()),
        _ => return None,
    };

    // Second pass — assign aliases and labels. Seed with the head, then
    // process expansions bottom-up (raw_expands is in top-down order,
    // so iterate in reverse) so each Expand's `source` is already
    // registered when we look at it.
    let mut alias_order: Vec<String> = Vec::new();
    let mut alias_index: BTreeMap<String, usize> = BTreeMap::new();
    let mut alias_label: BTreeMap<String, String> = BTreeMap::new();
    upsert_alias(
        head_alias.clone(),
        Some(head_label.clone()),
        &mut alias_order,
        &mut alias_index,
        &mut alias_label,
    )?;
    for ex in raw_expands.iter().rev() {
        if !alias_index.contains_key(&ex.source) {
            // Chain references something bound outside the subtree.
            return None;
        }
        // Back-reference Expands carry `target_label: None`
        // (lower.rs:914) because the alias was already labelled when it
        // was first introduced. For non-back-refs a missing label means
        // the user wrote `()` without a label, which v0 rejects because
        // the executor's outer scan and per-candidate `lookup_node`
        // both need one.
        let label_for_target = match &ex.target_label {
            Some(l) => Some(l.clone()),
            None if alias_label.contains_key(&ex.target) => None,
            None => return None,
        };
        upsert_alias(
            ex.target.clone(),
            label_for_target,
            &mut alias_order,
            &mut alias_index,
            &mut alias_label,
        )?;
    }

    let raw_edges: Vec<RawEdge> = raw_expands
        .iter()
        .map(|ex| RawEdge {
            from: ex.source.clone(),
            to: ex.target.clone(),
            edge_types: ex.edge_types.clone(),
            direction: ex.direction,
        })
        .collect();

    // Cycle detection (treat constraints as undirected for this check;
    // direction is handled by the executor). v vertices + e edges with
    // e >= v ⇒ at least one cycle, but a chain may still be acyclic if
    // edges duplicate; do a real union-find sweep.
    if !has_cycle(alias_order.len(), &raw_edges, &alias_index) {
        return None;
    }

    // Build NodeBinding list in the canonical order (`alias_order`).
    // The head NodeScan's predicates attach to its own binding; every
    // other binding carries no predicates in v0 (defensive label
    // filters are already absorbed via `target_label`).
    let vars: Vec<NodeBinding> = alias_order
        .iter()
        .map(|alias| NodeBinding {
            alias: alias.clone(),
            label: alias_label.get(alias).cloned(),
            predicates: if *alias == head_alias {
                head_predicates.clone()
            } else {
                Vec::new()
            },
        })
        .collect();

    let edges: Vec<EdgeConstraint> = raw_edges
        .iter()
        .map(|e| EdgeConstraint {
            from_idx: alias_index[&e.from],
            to_idx: alias_index[&e.to],
            edge_types: e.edge_types.clone(),
            direction: e.direction,
        })
        .collect();

    let ordering = variable_ordering(&vars, &edges, &head_alias);

    Some(LogicalPlan::MultiwayJoin {
        vars,
        edges,
        ordering,
        factorize_required: true,
    })
}

#[derive(Debug)]
struct RawExpand {
    source: String,
    target: String,
    target_label: Option<String>,
    edge_types: Vec<String>,
    direction: RelationshipDirection,
}

#[derive(Debug)]
struct RawEdge {
    from: String,
    to: String,
    edge_types: Vec<String>,
    direction: RelationshipDirection,
}

fn upsert_alias(
    alias: String,
    label: Option<String>,
    order: &mut Vec<String>,
    index: &mut BTreeMap<String, usize>,
    labels: &mut BTreeMap<String, String>,
) -> Option<()> {
    match index.get(&alias) {
        Some(_) => {
            if let Some(new_label) = label {
                match labels.get(&alias) {
                    Some(existing) if existing != &new_label => {
                        // Same alias bound under conflicting labels —
                        // the chain is malformed for our purposes.
                        return None;
                    }
                    Some(_) => {}
                    None => {
                        labels.insert(alias.clone(), new_label);
                    }
                }
            }
        }
        None => {
            let idx = order.len();
            order.push(alias.clone());
            index.insert(alias.clone(), idx);
            if let Some(l) = label {
                labels.insert(alias, l);
            }
        }
    }
    Some(())
}

fn has_cycle(
    num_vertices: usize,
    edges: &[RawEdge],
    alias_index: &BTreeMap<String, usize>,
) -> bool {
    let mut parent: Vec<usize> = (0..num_vertices).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] == x {
            return x;
        }
        let r = find(parent, parent[x]);
        parent[x] = r;
        r
    }
    let mut already_seen: BTreeSet<(usize, usize)> = BTreeSet::new();
    for e in edges {
        let a = alias_index[&e.from];
        let b = alias_index[&e.to];
        let key = if a < b { (a, b) } else { (b, a) };
        if !already_seen.insert(key) {
            // Two edges between the same pair of aliases also count as
            // a cycle (parallel edges). The leapfrog will simply
            // intersect more lists at one level.
            return true;
        }
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb {
            return true;
        }
        parent[ra] = rb;
    }
    false
}

// ─────────────────────── variable ordering ──────────────────────────────

/// v0 heuristic: keep the head NodeScan alias as the outer-most
/// variable (so the executor reuses its label scan and any pushed
/// predicates), then order the rest by degree in the constraint graph
/// descending so high-connectivity variables are bound early and the
/// leapfrog intersections at deeper levels stay narrow. Ties broken by
/// alias name lex for determinism.
///
/// The RFC §"Variable ordering" sketches a more elaborate AGM-aware
/// pass; we leave it for v0.1 once a bench surfaces this heuristic
/// regressing.
fn variable_ordering(
    vars: &[NodeBinding],
    edges: &[EdgeConstraint],
    head_alias: &str,
) -> Vec<usize> {
    let mut degree = vec![0usize; vars.len()];
    for e in edges {
        degree[e.from_idx] += 1;
        degree[e.to_idx] += 1;
    }
    let head_idx = vars
        .iter()
        .position(|v| v.alias == head_alias)
        .expect("head alias must be in vars");

    let mut rest: Vec<usize> = (0..vars.len()).filter(|&i| i != head_idx).collect();
    rest.sort_by(|&a, &b| {
        degree[b]
            .cmp(&degree[a])
            .then_with(|| vars[a].alias.cmp(&vars[b].alias))
    });

    let mut order = Vec::with_capacity(vars.len());
    order.push(head_idx);
    order.extend(rest);
    order
}

// ─────────────────────── mechanical recursion ───────────────────────────

fn recurse_children(plan: LogicalPlan) -> LogicalPlan {
    match plan {
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
            input: Box::new(rewrite(*input)),
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
            input: Box::new(rewrite(*input)),
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
            input: Box::new(rewrite(*input)),
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
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(rewrite(*input)),
            predicate,
        },
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(rewrite(*input)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => LogicalPlan::Aggregate {
            input: Box::new(rewrite(*input)),
            group_by,
            aggregations,
        },
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(rewrite(*input)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(rewrite(*input)),
        },
        LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
            left: Box::new(rewrite(*left)),
            right: Box::new(rewrite(*right)),
            all,
        },
        LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
            input: Box::new(rewrite(*input)),
            list,
            alias,
        },
        LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
            left: Box::new(rewrite(*left)),
            right: Box::new(rewrite(*right)),
        },
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => LogicalPlan::HashJoin {
            build: Box::new(rewrite(*build)),
            probe: Box::new(rewrite(*probe)),
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
            outer: Box::new(rewrite(*outer)),
            inner: Box::new(rewrite(*inner)),
            on,
            negated,
            residual,
        },
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => LogicalPlan::SemiApply {
            input: Box::new(rewrite(*input)),
            // Do NOT recurse into `subplan`: a MultiwayJoin under a
            // correlated subplan would inherit bindings from the outer
            // row, which v0 explicitly refuses (see walker.rs).
            subplan,
            negated,
        },
        LogicalPlan::PatternList {
            input,
            subplan,
            projection,
            alias,
        } => LogicalPlan::PatternList {
            input: Box::new(rewrite(*input)),
            // Same as SemiApply: subplan stays binary.
            subplan,
            projection,
            alias,
        },
        LogicalPlan::Create { input, elements } => LogicalPlan::Create {
            input: Box::new(rewrite(*input)),
            elements,
        },
        LogicalPlan::Foreach {
            input,
            variable,
            list,
            body,
        } => LogicalPlan::Foreach {
            input: Box::new(rewrite(*input)),
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
            input: Box::new(rewrite(*input)),
            pattern,
            on_match_sets,
            on_create_sets,
        },
        LogicalPlan::Set { input, items } => LogicalPlan::Set {
            input: Box::new(rewrite(*input)),
            items,
        },
        LogicalPlan::Remove { input, items } => LogicalPlan::Remove {
            input: Box::new(rewrite(*input)),
            items,
        },
        LogicalPlan::Delete {
            input,
            targets,
            detach,
        } => LogicalPlan::Delete {
            input: Box::new(rewrite(*input)),
            targets,
            detach,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::ShortestMode;

    fn scan(alias: &str, label: &str) -> LogicalPlan {
        LogicalPlan::NodeScan {
            label: Some(label.into()),
            alias: alias.into(),
            predicates: Vec::new(),
            projection: None,
        }
    }

    fn expand(
        input: LogicalPlan,
        source: &str,
        target: &str,
        target_label: &str,
        back_reference: bool,
    ) -> LogicalPlan {
        LogicalPlan::Expand {
            input: Box::new(input),
            source: source.into(),
            edge_type: Some(vec!["KNOWS".into()]),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: target.into(),
            target_labels: vec![target_label.into()],
            length: None,
            optional: false,
            back_reference,
            shortest: ShortestMode::None,
            path_binding: None,
        }
    }

    #[test]
    fn triangle_chain_rewrites_to_multiway_join() {
        let plan = expand(
            expand(
                expand(scan("a", "Person"), "a", "b", "Person", false),
                "b",
                "c",
                "Person",
                false,
            ),
            "c",
            "a",
            "Person",
            true,
        );
        let rewritten = try_rewrite_chain(&plan).expect("triangle must rewrite");
        match rewritten {
            LogicalPlan::MultiwayJoin {
                vars,
                edges,
                ordering,
                factorize_required,
            } => {
                assert!(factorize_required);
                let aliases: Vec<&str> = vars.iter().map(|v| v.alias.as_str()).collect();
                assert!(aliases.contains(&"a"));
                assert!(aliases.contains(&"b"));
                assert!(aliases.contains(&"c"));
                assert_eq!(edges.len(), 3);
                assert_eq!(ordering.len(), 3);
                // Head alias `a` is the outer-most variable.
                assert_eq!(vars[ordering[0]].alias, "a");
            }
            other => panic!("expected MultiwayJoin, got {:?}", other.operator_name()),
        }
    }

    #[test]
    fn open_chain_stays_binary() {
        // a -> b -> c, no closing edge.
        let plan = expand(
            expand(scan("a", "Person"), "a", "b", "Person", false),
            "b",
            "c",
            "Person",
            false,
        );
        assert!(try_rewrite_chain(&plan).is_none());
    }

    #[test]
    fn variable_length_chain_rejected() {
        let mut plan = expand(scan("a", "Person"), "a", "b", "Person", false);
        if let LogicalPlan::Expand { length, .. } = &mut plan {
            *length = Some(crate::parser::RelationshipLength { min: 1, max: 3 });
        }
        let closed = expand(plan, "b", "a", "Person", true);
        assert!(try_rewrite_chain(&closed).is_none());
    }

    #[test]
    fn untyped_edge_chain_rejected() {
        let mut inner = expand(scan("a", "Person"), "a", "b", "Person", false);
        if let LogicalPlan::Expand { edge_type, .. } = &mut inner {
            *edge_type = None;
        }
        let closed = expand(inner, "b", "a", "Person", true);
        assert!(try_rewrite_chain(&closed).is_none());
    }

    #[test]
    fn both_direction_chain_rejected() {
        let mut inner = expand(scan("a", "Person"), "a", "b", "Person", false);
        if let LogicalPlan::Expand { direction, .. } = &mut inner {
            *direction = RelationshipDirection::Both;
        }
        let closed = expand(inner, "b", "a", "Person", true);
        assert!(try_rewrite_chain(&closed).is_none());
    }

    #[test]
    fn rel_alias_chain_rejected() {
        let mut inner = expand(scan("a", "Person"), "a", "b", "Person", false);
        if let LogicalPlan::Expand { rel_alias, .. } = &mut inner {
            *rel_alias = Some("r".into());
        }
        let closed = expand(inner, "b", "a", "Person", true);
        assert!(try_rewrite_chain(&closed).is_none());
    }

    #[test]
    fn missing_target_label_rejected() {
        let mut inner = expand(scan("a", "Person"), "a", "b", "Person", false);
        if let LogicalPlan::Expand { target_labels, .. } = &mut inner {
            *target_labels = vec![];
        }
        let closed = expand(inner, "b", "a", "Person", true);
        assert!(try_rewrite_chain(&closed).is_none());
    }

    #[test]
    fn cycle_detected_correctly() {
        let mut alias_index = BTreeMap::new();
        alias_index.insert("a".to_string(), 0);
        alias_index.insert("b".to_string(), 1);
        alias_index.insert("c".to_string(), 2);

        let acyclic = vec![
            RawEdge {
                from: "a".into(),
                to: "b".into(),
                edge_types: vec!["X".into()],
                direction: RelationshipDirection::Right,
            },
            RawEdge {
                from: "b".into(),
                to: "c".into(),
                edge_types: vec!["X".into()],
                direction: RelationshipDirection::Right,
            },
        ];
        assert!(!has_cycle(3, &acyclic, &alias_index));

        let cyclic = vec![
            RawEdge {
                from: "a".into(),
                to: "b".into(),
                edge_types: vec!["X".into()],
                direction: RelationshipDirection::Right,
            },
            RawEdge {
                from: "b".into(),
                to: "c".into(),
                edge_types: vec!["X".into()],
                direction: RelationshipDirection::Right,
            },
            RawEdge {
                from: "c".into(),
                to: "a".into(),
                edge_types: vec!["X".into()],
                direction: RelationshipDirection::Right,
            },
        ];
        assert!(has_cycle(3, &cyclic, &alias_index));
    }
}
