//! Label predicates in `WHERE` become label scans.
//!
//! `MATCH (n:Device)` lowers to a label-scoped `NodeScan`, but the same
//! constraint written as a predicate — `MATCH (n) WHERE n:Device`, or a
//! `(n:A|B)` disjunction pattern — lowers to an unlabelled `NodeScan`
//! (`label=*`, every node of every label) under a `Filter` holding
//! `__label_eq(n, 'Device')`. On a large store that reads and decodes every
//! node to keep a handful, which is both slow and enough memory to take a
//! process down.
//!
//! Two rewrites, both on a `Filter` sitting directly on an unlabelled,
//! still-unscoped `NodeScan` (the place predicate pushdown leaves a
//! single-alias predicate):
//!
//! 1. **Conjunction** ([`scope_label_conjunct`]): a top-level conjunct
//!    `__label_eq(n, L)` scopes the scan to `L`. With several (`n:A:B`,
//!    `n:A AND n:B`) the label with the fewest nodes is chosen and the others
//!    stay as filters, as do `NOT n:L` and everything else. Runs inside the
//!    optimizer fixpoint, so later passes (unique lookups, anchor inversion,
//!    join planning) see the labelled scan exactly as if the label had been
//!    written in the pattern.
//!
//! 2. **Disjunction** ([`split_label_disjunction`]): a top-level conjunct
//!    `D1 OR … OR Dk` in which every `Di` requires some label `Li` of `n`
//!    becomes a `UNION ALL` of `k` label scans. Branch `i` scans `Li`, keeps
//!    `Di`, and drops the rows an earlier branch already produced with
//!    `NOT coalesce(Dj, false)` for every `j < i` — so the branches are
//!    disjoint, the union needs no de-duplication (and no memory for it),
//!    and every node appears exactly as often as under the original filter.
//!    The remaining conjuncts stay above the union; pushdown sinks them into
//!    every branch on the next round. Runs once the first fixpoint has
//!    settled, so the passes that understand the lowered `(n:A|B)` filter
//!    shape (anchor inversion) get their chance first.
//!
//! What stays a full scan: a predicate that can hold for a node without any
//! particular label (`NOT n:L` alone, `n:A OR n.x = 1`) — there is no label
//! to start from, so reading everything is the only correct plan.

use crate::cost::StatsCatalog;
use crate::parser::ast::{BinaryOp, Expression, ExpressionKind, Literal, QualifiedName, UnaryOp};
use crate::plan::LogicalPlan;

use super::{and_chain, split_and_terms, synthetic_label_eq_label};

/// Rewrite 1 over the whole plan. See the module docs.
pub fn scope_label_conjunct(mut plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    visit(&mut plan, &mut |node| scope_one(node, catalog));
    plan
}

/// Rewrite 2 over the whole plan. See the module docs.
pub fn split_label_disjunction(mut plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    visit(&mut plan, &mut |node| split_one(node, catalog));
    plan
}

/// Post-order walk: children first, then `f` on the node itself.
fn visit(plan: &mut LogicalPlan, f: &mut dyn FnMut(&mut LogicalPlan)) {
    for child in plan.children_mut() {
        visit(child, f);
    }
    f(plan);
}

/// `(alias, Filter predicate)` when `plan` is `Filter` over an unlabelled
/// `NodeScan`.
fn filter_over_unlabelled_scan(plan: &LogicalPlan) -> Option<(&str, &Expression)> {
    let LogicalPlan::Filter { input, predicate } = plan else {
        return None;
    };
    let LogicalPlan::NodeScan {
        label: None, alias, ..
    } = input.as_ref()
    else {
        return None;
    };
    Some((alias.as_str(), predicate))
}

fn label_count(catalog: &StatsCatalog, label: &str) -> u64 {
    // A label the catalog has never seen holds no committed node; it only
    // has memtable rows, if any, so it is the cheapest possible scan.
    catalog.label(label).map_or(0, |stats| stats.node_count)
}

/// The cheapest label among `labels` (fewest nodes, first on ties).
fn cheapest<'a>(
    labels: impl Iterator<Item = &'a String>,
    catalog: &StatsCatalog,
) -> Option<&'a String> {
    labels.min_by_key(|label| label_count(catalog, label))
}

fn scope_one(plan: &mut LogicalPlan, catalog: &StatsCatalog) {
    let Some((alias, predicate)) = filter_over_unlabelled_scan(plan) else {
        return;
    };
    let terms = split_and_terms(predicate);
    let labels: Vec<String> = terms
        .iter()
        .filter_map(|term| synthetic_label_eq_label(term, alias))
        .collect();
    let Some(chosen) = cheapest(labels.iter(), catalog).cloned() else {
        return;
    };
    let mut removed = false;
    let rest: Vec<Expression> = terms
        .into_iter()
        .filter(|term| {
            if !removed && synthetic_label_eq_label(term, alias).as_deref() == Some(chosen.as_str())
            {
                removed = true;
                false
            } else {
                true
            }
        })
        .collect();
    let LogicalPlan::Filter { input, .. } = std::mem::replace(plan, LogicalPlan::Empty) else {
        unreachable!("checked above");
    };
    let mut scan = *input;
    if let LogicalPlan::NodeScan { label, .. } = &mut scan {
        *label = Some(chosen);
    }
    *plan = match and_chain(rest) {
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(scan),
            predicate,
        },
        None => scan,
    };
}

/// Flatten an `OR` chain into its disjuncts.
fn split_or_terms(expr: &Expression) -> Vec<Expression> {
    fn walk(expr: &Expression, out: &mut Vec<Expression>) {
        if let ExpressionKind::Binary {
            op: BinaryOp::Or,
            left,
            right,
        } = &expr.kind
        {
            walk(left, out);
            walk(right, out);
        } else {
            out.push(expr.clone());
        }
    }
    let mut out = Vec::new();
    walk(expr, &mut out);
    out
}

/// One branch of a label disjunction: the label it scans and what of the
/// disjunct is left to check once the scan guarantees that label.
struct Branch {
    label: String,
    residual: Vec<Expression>,
    /// The whole disjunct, to exclude its rows from later branches.
    disjunct: Expression,
    /// `true` when the disjunct is made of label tests only, so it is never
    /// NULL for a scanned node and plain `NOT` excludes it exactly.
    labels_only: bool,
}

/// Every disjunct of `term` pins a label of `alias`: the branches, else
/// `None`.
fn label_branches(term: &Expression, alias: &str, catalog: &StatsCatalog) -> Option<Vec<Branch>> {
    let disjuncts = split_or_terms(term);
    if disjuncts.len() < 2 {
        return None;
    }
    disjuncts
        .into_iter()
        .map(|disjunct| {
            let conjuncts = split_and_terms(&disjunct);
            let labels: Vec<String> = conjuncts
                .iter()
                .filter_map(|c| synthetic_label_eq_label(c, alias))
                .collect();
            let label = cheapest(labels.iter(), catalog)?.clone();
            let labels_only = labels.len() == conjuncts.len();
            let mut removed = false;
            let residual = conjuncts
                .into_iter()
                .filter(|c| {
                    if !removed
                        && synthetic_label_eq_label(c, alias).as_deref() == Some(label.as_str())
                    {
                        removed = true;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            Some(Branch {
                label,
                residual,
                disjunct,
                labels_only,
            })
        })
        .collect()
}

fn not(expr: Expression) -> Expression {
    let span = expr.span;
    Expression {
        kind: ExpressionKind::Unary {
            op: UnaryOp::Not,
            expr: Box::new(expr),
        },
        span,
    }
}

/// `coalesce(expr, false)`: TRUE stays TRUE, FALSE and NULL become FALSE.
fn true_or_false(expr: Expression) -> Expression {
    let span = expr.span;
    Expression {
        kind: ExpressionKind::FunctionCall {
            name: QualifiedName::single(crate::parser::ast::Identifier::new("coalesce", span)),
            args: vec![
                expr,
                Expression {
                    kind: ExpressionKind::Literal(Literal::Boolean(false)),
                    span,
                },
            ],
            distinct: false,
        },
        span,
    }
}

fn split_one(plan: &mut LogicalPlan, catalog: &StatsCatalog) {
    let Some((alias, predicate)) = filter_over_unlabelled_scan(plan) else {
        return;
    };
    let alias = alias.to_string();
    let terms = split_and_terms(predicate);
    let Some((index, branches)) = terms
        .iter()
        .enumerate()
        .find_map(|(i, term)| label_branches(term, &alias, catalog).map(|b| (i, b)))
    else {
        return;
    };
    let LogicalPlan::Filter { input, .. } = std::mem::replace(plan, LogicalPlan::Empty) else {
        unreachable!("checked above");
    };
    let LogicalPlan::NodeScan {
        predicates,
        projection,
        ..
    } = *input
    else {
        unreachable!("checked above");
    };
    let mut union: Option<LogicalPlan> = None;
    for (i, branch) in branches.iter().enumerate() {
        let mut checks = branch.residual.clone();
        for earlier in &branches[..i] {
            checks.push(if earlier.labels_only {
                not(earlier.disjunct.clone())
            } else {
                not(true_or_false(earlier.disjunct.clone()))
            });
        }
        let scan = LogicalPlan::NodeScan {
            label: Some(branch.label.clone()),
            alias: alias.clone(),
            predicates: predicates.clone(),
            projection: projection.clone(),
        };
        let part = match and_chain(checks) {
            Some(predicate) => LogicalPlan::Filter {
                input: Box::new(scan),
                predicate,
            },
            None => scan,
        };
        union = Some(match union {
            None => part,
            Some(left) => LogicalPlan::Union {
                left: Box::new(left),
                right: Box::new(part),
                all: true,
            },
        });
    }
    let union = union.expect("a disjunction has at least two branches");
    let rest: Vec<Expression> = terms
        .into_iter()
        .enumerate()
        .filter_map(|(i, term)| (i != index).then_some(term))
        .collect();
    *plan = match and_chain(rest) {
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(union),
            predicate,
        },
        None => union,
    };
}

/// Fold a `WHERE m:L` on the target of a plain single-hop `Expand` into the
/// Expand's own target labels. The Expand then proves the label on the
/// compact label sidecar before it reads any neighbour, instead of reading
/// every neighbour of a hub and filtering afterwards. Only for a
/// non-optional, fixed-length hop that introduces its target: there the
/// label constraint on the target and the filter above are the same thing.
pub fn fold_expand_target_labels(mut plan: LogicalPlan) -> LogicalPlan {
    visit(&mut plan, &mut fold_one);
    plan
}

fn fold_one(plan: &mut LogicalPlan) {
    let LogicalPlan::Filter { input, predicate } = plan else {
        return;
    };
    let LogicalPlan::Expand {
        target_alias,
        target_labels,
        length: None,
        optional: false,
        back_reference: false,
        shortest: crate::plan::ShortestMode::None,
        path_binding: None,
        ..
    } = input.as_mut()
    else {
        return;
    };
    let terms = split_and_terms(predicate);
    let mut rest = Vec::with_capacity(terms.len());
    let mut changed = false;
    for term in terms {
        match synthetic_label_eq_label(&term, target_alias) {
            Some(label) => {
                if !target_labels.contains(&label) {
                    target_labels.push(label);
                }
                changed = true;
            }
            None => rest.push(term),
        }
    }
    if !changed {
        return;
    }
    let LogicalPlan::Filter { input, .. } = std::mem::replace(plan, LogicalPlan::Empty) else {
        unreachable!("matched above");
    };
    *plan = match and_chain(rest) {
        Some(predicate) => LogicalPlan::Filter { input, predicate },
        None => *input,
    };
}

#[cfg(test)]
mod tests {
    use crate::cost::{LabelStats, StatsCatalog};
    use crate::plan::explain;

    fn catalog() -> StatsCatalog {
        let mut cat = StatsCatalog::empty();
        for (name, count) in [("Device", 4), ("Call", 2_958), ("Message", 1_400_000)] {
            cat.__test_insert_label(LabelStats {
                name: name.into(),
                node_count: count,
                properties: Default::default(),
            });
        }
        cat
    }

    fn plan(q: &str) -> String {
        let parsed = crate::parse(q).unwrap();
        explain(&crate::plan(&parsed, &catalog()).unwrap())
    }

    fn assert_no_full_scan(q: &str) -> String {
        let p = plan(q);
        assert!(!p.contains("label=*"), "{q}\n{p}");
        p
    }

    #[test]
    fn where_label_scopes_the_scan() {
        let p = assert_no_full_scan("MATCH (n) WHERE n:Device RETURN count(n)");
        assert!(p.contains("NodeScan label=Device alias=n"), "{p}");
        assert!(!p.contains("__label_eq"), "{p}");
    }

    #[test]
    fn where_label_conjunction_picks_the_smallest_label() {
        for q in [
            "MATCH (n) WHERE n:Message AND n:Device RETURN n",
            "MATCH (n) WHERE n:Message:Device RETURN n",
            "MATCH (n) WHERE n.x = 1 AND n:Message AND n:Device RETURN n",
        ] {
            let p = assert_no_full_scan(q);
            assert!(p.contains("NodeScan label=Device"), "{q}\n{p}");
            assert!(
                p.contains("__label_eq(n, 'Message')") || p.contains("__label_eq(n, \"Message\")"),
                "{q}\n{p}"
            );
        }
    }

    #[test]
    fn negated_label_filters_but_another_label_anchors() {
        let p = assert_no_full_scan("MATCH (n) WHERE n:Call AND NOT n:Device RETURN n");
        assert!(p.contains("NodeScan label=Call"), "{p}");
        assert!(p.contains("NOT"), "{p}");
    }

    #[test]
    fn negated_label_alone_still_scans_everything() {
        let p = plan("MATCH (n) WHERE NOT n:Device RETURN n");
        assert!(p.contains("label=*"), "{p}");
    }

    #[test]
    fn label_disjunction_becomes_disjoint_union_of_label_scans() {
        for q in [
            "MATCH (n) WHERE n:Device OR n:Call RETURN count(n)",
            "MATCH (n:Device|Call) RETURN count(n)",
            "MATCH (n) WHERE (n:Device OR n:Call) AND n.x > 1 RETURN n",
            "MATCH (n) WHERE (n:Device AND n.x = 1) OR n:Call RETURN n",
        ] {
            let p = assert_no_full_scan(q);
            assert!(p.contains("UnionAll"), "{q}\n{p}");
            assert!(p.contains("NodeScan label=Device"), "{q}\n{p}");
            assert!(p.contains("NodeScan label=Call"), "{q}\n{p}");
        }
    }

    #[test]
    fn mixed_disjunction_without_a_label_per_branch_stays_a_scan() {
        let p = plan("MATCH (n) WHERE n:Device OR n.x = 1 RETURN n");
        assert!(p.contains("label=*"), "{p}");
    }

    #[test]
    fn label_on_any_pattern_variable_after_with_and_in_optional_match() {
        for q in [
            "MATCH (a:Call) WITH a MATCH (n) WHERE n:Device RETURN a, n",
            "MATCH (a:Call) OPTIONAL MATCH (n) WHERE n:Device RETURN a, n",
            "MATCH (a:Call), (n) WHERE n:Device RETURN a, n",
            "MATCH (a:Call) WITH a MATCH (n) WHERE n:Device OR n:Message RETURN a, n",
        ] {
            assert_no_full_scan(q);
        }
    }

    #[test]
    fn label_on_expand_target_becomes_expand_label() {
        let p = assert_no_full_scan("MATCH (d:Device)<-[:ON_DEVICE]-(x) WHERE x:Call RETURN x");
        assert!(!p.contains("__label_eq"), "{p}");
        assert!(p.contains("target=x:Call"), "{p}");
    }

    #[test]
    fn label_inside_optional_match_folds_under_the_optional_apply() {
        // The WHERE belongs to the OPTIONAL MATCH: the subplan under the
        // OptionalApply may drop non-matching rows, the apply then emits the
        // NULL row. Folding the label into that subplan's Expand is exact.
        let p =
            plan("MATCH (d:Device) OPTIONAL MATCH (d)<-[:ON_DEVICE]-(x) WHERE x:Call RETURN d, x");
        assert!(p.contains("OptionalApply"), "{p}");
        assert!(p.contains("target=x:Call"), "{p}");
        assert!(!p.contains("label=*"), "{p}");
    }

    #[test]
    fn parenthesised_label_predicates_parse_and_rewrite() {
        for q in [
            "MATCH (n) WHERE (n:Device) RETURN n",
            "MATCH (n) WHERE (n:Device OR n:Call) RETURN n",
            "MATCH (n) WHERE (n:Device AND n.x = 1) RETURN n",
            "MATCH (n) WHERE NOT (n:Device) AND (n:Call) RETURN n",
        ] {
            assert_no_full_scan(q);
        }
    }

    #[test]
    fn rewrite_is_idempotent() {
        for q in [
            "MATCH (n) WHERE n:Device OR n:Call RETURN n",
            "MATCH (n) WHERE n:Device RETURN n",
        ] {
            let parsed = crate::parse(q).unwrap();
            let once = crate::plan(&parsed, &catalog()).unwrap();
            let twice = crate::optimize(once.clone(), &catalog());
            assert_eq!(once, twice, "{q}");
        }
    }
}
