//! `vector_search` rewrite (RFC-030, `vector-index`): turn a flat KNN shape
//! into a [`LogicalPlan::VectorSearch`] when a matching Vamana index exists.
//!
//! A KNN lowers to one of two shapes depending on how the query ends:
//!
//! - **terminal `RETURN`** — `MATCH (d:L) [WHERE …] RETURN d.x AS t,
//!   cosine_similarity(d.emb, $q) AS score ORDER BY score DESC LIMIT k` lowers to
//!   `Project[t, score]{ TopN{ [Filter] { NodeScan{L, d} } } }` — the projection
//!   is the **outer** node. This is the common, natural way to write a KNN.
//! - **non-terminal `WITH`** — `MATCH (d:L) WITH d ORDER BY cosine_similarity(
//!   d.emb,$q) DESC LIMIT k …` lowers to `TopN{ Project[.., dist AS score]{
//!   [Filter]{ NodeScan{L, d} } } }` — the projection is **inside** the TopN.
//!
//! When the catalog has a `VectorIndexDescriptor` for `(L, prop, metric)`, the
//! `TopN[ [Filter] NodeScan ]` ranking sub-tree collapses to a `VectorSearch`
//! leaf the executor serves from the index (falling back to the flat scan when
//! no index matches); any outer `Project` is preserved on top. A `WHERE` the
//! rewrite folds in is captured as the `VectorSearch`'s `post_filter` so the
//! index path survives a filter (RFC-030 filtered ANN). The rewrite reaches a
//! KNN nested in a UNION branch, a `CALL {}` subquery, a join, or an aggregate
//! (bottom-up recursion).
//!
//! The order key must be in the metric's *nearest-first* direction — `DESC` for
//! the higher-is-closer metrics (`cosine_similarity`, `dot_product`), `ASC` for
//! `euclidean_distance` (lower distance is closer); the wrong direction asks for
//! the *farthest* k, which the index does not compute. Conservative: any `SKIP`,
//! an unbounded `ORDER BY` with no `LIMIT` (k = `u64::MAX`), multiple keys, a
//! non-vector key function, a `DISTINCT` projection, a cross-binding filter, a
//! predicate already pushed into `NodeScan.predicates`, or a missing index leaves
//! the plan unchanged.
//!
//! Registered in `optimize::mod` right after `unique_lookup`, so the downstream
//! pushdowns (which treat `VectorSearch` as an opaque leaf) see the new
//! operator and don't re-introduce a Filter above it.

use crate::cost::StatsCatalog;
use crate::parser::ast::{BinaryOp, OrderDirection};
use crate::parser::{Expression, ExpressionKind};
use crate::plan::logical::{LogicalPlan, OrderKey, ProjectionItem, RowCount, VectorDistance};

/// Run the rewrite over `plan`. No-op when no index matches.
pub fn apply_vector_search(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    // Rewrite nested sub-plans first (so a KNN inside a UNION branch, a `CALL {}`
    // subquery, a join, or an aggregate is collapsed too), then attempt the KNN
    // shapes at this node.
    let plan = recurse(plan, catalog);
    try_rewrite_here(plan, catalog)
}

/// Attempt the KNN→`VectorSearch` rewrite at the *root* of `plan` (its children
/// are already rewritten). Tries the three lowered shapes; returns `plan`
/// unchanged when none match or no index backs it.
fn try_rewrite_here(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    // Terminal-`RETURN` KNN: `Project[ … ]{ TopN{ [Filter] NodeScan } }`. Rewrite
    // the TopN ranking sub-tree to a `VectorSearch` and keep the outer Project
    // (it re-projects `d.x`/`score` from the node the index returns). The score
    // column's alias is read from the outer Project so the operator names it the
    // same way the projection expects.
    if let LogicalPlan::Project {
        items,
        input,
        distinct,
        discard_input_bindings,
    } = plan
    {
        if let LogicalPlan::TopN { keys, .. } = input.as_ref() {
            let sa = outer_score_alias_of(&items, keys);
            if let Some(vs) = try_match(&input, catalog, None, sa.as_deref()) {
                return LogicalPlan::Project {
                    items,
                    input: Box::new(vs),
                    distinct,
                    discard_input_bindings,
                };
            }
        }
        // Not a KNN under this Project — reassemble unchanged.
        return LogicalPlan::Project {
            items,
            input,
            distinct,
            discard_input_bindings,
        };
    }

    // A threshold / filter on the *ranked* output (e.g. `WHERE score >= 0.86`)
    // lowers to a Filter wrapping the TopN. Fold it into the VectorSearch's
    // `post_filter` when it references only the searched binding; otherwise leave
    // the plan alone so the filter still runs (flat path).
    if let LogicalPlan::Filter { predicate, input } = &plan {
        if matches!(input.as_ref(), LogicalPlan::TopN { .. }) {
            if let Some(vs) = try_match(input, catalog, Some(predicate), None) {
                return vs;
            }
        }
    }

    // Bare TopN at the root (non-terminal `WITH`, or a hand-built plan).
    if let Some(vs) = try_match(&plan, catalog, None, None) {
        vs
    } else {
        plan
    }
}

/// The alias the outer Project gives the distance column, matched structurally
/// (span-insensitively) against the TopN's ranking key, so the rewritten
/// `VectorSearch` names its score column the way the projection refers to it.
fn outer_score_alias_of(items: &[ProjectionItem], keys: &[OrderKey]) -> Option<String> {
    let (km, kp, ka, _) = single_distance_key(keys)?;
    items.iter().find_map(|it| {
        let (m, p, a, _) = distance_call_parts(&it.expression)?;
        (m == km && p == kp && a == ka).then(|| it.alias.clone())
    })
}

/// Descend into a plan's children so a KNN nested anywhere gets rewritten.
///
/// The KNN-chain wrappers (TopN / Project / Filter) are rebuilt **structurally**
/// (`recurse` on the child) so the multi-level shapes — `Project{TopN}` (terminal
/// RETURN) and `Filter{TopN}` (similarity threshold) — survive intact for
/// `try_rewrite_here` to match as a unit at the parent. Every other operator that
/// can *contain* a KNN sub-plan (UNION branches, `CALL {}` subqueries lowered to
/// Apply/SemiApply, joins, cross products, aggregates, unwind/distinct) hands each
/// child to the full `apply_vector_search`, which independently optimizes that
/// sub-plan from its own root. Leaves and write operators fall through unchanged.
fn recurse(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    match plan {
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(recurse(*input, catalog)),
            keys,
            skip,
            limit,
        },
        LogicalPlan::Project {
            input,
            items,
            distinct,
            discard_input_bindings,
        } => LogicalPlan::Project {
            input: Box::new(recurse(*input, catalog)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(recurse(*input, catalog)),
            predicate,
        },
        // Binary operators that can host a complete KNN sub-plan in a branch.
        LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
            left: Box::new(apply_vector_search(*left, catalog)),
            right: Box::new(apply_vector_search(*right, catalog)),
            all,
        },
        LogicalPlan::CrossProduct { left, right } => LogicalPlan::CrossProduct {
            left: Box::new(apply_vector_search(*left, catalog)),
            right: Box::new(apply_vector_search(*right, catalog)),
        },
        LogicalPlan::Apply { input, subplan } => LogicalPlan::Apply {
            input: Box::new(apply_vector_search(*input, catalog)),
            subplan: Box::new(apply_vector_search(*subplan, catalog)),
        },
        LogicalPlan::SemiApply {
            input,
            subplan,
            negated,
        } => LogicalPlan::SemiApply {
            input: Box::new(apply_vector_search(*input, catalog)),
            subplan: Box::new(apply_vector_search(*subplan, catalog)),
            negated,
        },
        LogicalPlan::HashJoin {
            build,
            probe,
            on,
            residual,
        } => LogicalPlan::HashJoin {
            build: Box::new(apply_vector_search(*build, catalog)),
            probe: Box::new(apply_vector_search(*probe, catalog)),
            on,
            residual,
        },
        // Single-input read wrappers a KNN sub-plan can sit beneath.
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregations,
        } => LogicalPlan::Aggregate {
            input: Box::new(apply_vector_search(*input, catalog)),
            group_by,
            aggregations,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(apply_vector_search(*input, catalog)),
        },
        LogicalPlan::Unwind { input, list, alias } => LogicalPlan::Unwind {
            input: Box::new(apply_vector_search(*input, catalog)),
            list,
            alias,
        },
        other => other,
    }
}

/// Borrow-based matcher: if `plan` is the KNN chain AND a backing index exists,
/// return the replacement `VectorSearch`. `above` is an optional predicate from
/// a Filter wrapping the TopN (a threshold on the ranked output) to fold into
/// `post_filter`.
fn try_match(
    plan: &LogicalPlan,
    catalog: &StatsCatalog,
    above: Option<&Expression>,
    outer_score_alias: Option<&str>,
) -> Option<LogicalPlan> {
    let LogicalPlan::TopN {
        keys,
        skip,
        limit,
        input,
    } = plan
    else {
        return None;
    };
    // No SKIP; exactly one correctly-oriented key on a vector-distance function.
    if !matches!(skip, RowCount::Const(0)) {
        return None;
    }
    // An unbounded `ORDER BY` with no `LIMIT` lowers to `k = u64::MAX`; copying
    // that into `VectorSearch.k` would overflow `Vec::with_capacity(k)` in the
    // executor. Leave such a plan as the flat TopN (which streams without
    // pre-allocating k).
    if matches!(limit, RowCount::Const(u64::MAX)) {
        return None;
    }
    let (distance, prop, alias, query) = single_distance_key(keys)?;

    // The TopN's input is either an inner Project that computes the score and
    // carries the binding through (non-terminal `WITH` form), or — for a terminal
    // `RETURN`, whose projection sits *outside* the TopN — a `[Filter →] NodeScan`
    // directly. Resolve the score alias and peel an optional `WHERE` Filter in
    // both cases.
    let (score_alias, between, scan) = match input.as_ref() {
        LogicalPlan::Project {
            items,
            distinct: false,
            input: proj_input,
            ..
        } => {
            let sa = items
                .iter()
                .find_map(|it| {
                    let (m, p, a, _) = distance_call_parts(&it.expression)?;
                    (m == distance && p == prop && a == alias).then(|| it.alias.clone())
                })
                .or_else(|| outer_score_alias.map(str::to_string))
                .unwrap_or_else(|| "score".to_string());
            let (scan, between) = peel_filter(proj_input.as_ref());
            (sa, between, scan)
        }
        other => {
            let sa = outer_score_alias
                .map(str::to_string)
                .unwrap_or_else(|| "score".to_string());
            let (scan, between) = peel_filter(other);
            (sa, between, scan)
        }
    };

    let LogicalPlan::NodeScan {
        label: Some(label),
        alias: scan_alias,
        predicates,
        ..
    } = scan
    else {
        return None;
    };
    if scan_alias != &alias {
        return None;
    }
    // `predicate_pushdown` can fold a `WHERE` into `NodeScan.predicates` before
    // this rewrite runs (the fixpoint interleaves the passes). Those storage-level
    // predicates cannot be reconstructed into a `post_filter` Expression here, so
    // swallowing the scan into a `VectorSearch` would silently drop them. Refuse
    // the rewrite when any are present — the flat path keeps and honours them. The
    // common filtered-ANN case is unaffected: a `Filter` directly above the scan
    // is captured via `peel_filter` before pushdown moves it.
    if !predicates.is_empty() {
        return None;
    }

    // Index must exist for (label, prop, metric).
    catalog.vector_index_for(label, &prop, metric_to_storage(distance))?;

    // Fold the captured predicate(s) into `post_filter`, but ONLY when they
    // reference solely the searched binding (`alias`) / the score column
    // (`score_alias`). A predicate that touches another binding must NOT be
    // swallowed — bail so the plan keeps its Filter and runs via the flat path.
    let mut post_filter: Option<Expression> = None;
    if let Some(b) = between {
        if !aliases_within(&b, &[&alias]) {
            return None;
        }
        post_filter = Some(b);
    }
    if let Some(a) = above {
        if !aliases_within(a, &[&alias, &score_alias]) {
            return None;
        }
        post_filter = Some(match post_filter {
            Some(p) => and_expr(p, a.clone()),
            None => a.clone(),
        });
    }

    Some(LogicalPlan::VectorSearch {
        label: Some(label.clone()),
        alias,
        property: prop,
        query,
        k: limit.clone(),
        distance,
        score_alias,
        post_filter,
    })
}

/// `true` if every binding `expr` references is in `allowed`. Used to keep a
/// captured `WHERE` from swallowing a cross-binding predicate.
fn aliases_within(expr: &Expression, allowed: &[&str]) -> bool {
    super::expression_aliases(expr)
        .iter()
        .all(|a| allowed.contains(&a.as_str()))
}

/// `a AND b` as one predicate Expression.
fn and_expr(a: Expression, b: Expression) -> Expression {
    let span = a.span;
    Expression {
        kind: ExpressionKind::Binary {
            op: BinaryOp::And,
            left: Box::new(a),
            right: Box::new(b),
        },
        span,
    }
}

/// `true` when a larger metric value means a closer match — cosine similarity
/// and raw dot product. Euclidean is a distance: smaller is closer.
fn metric_higher_is_better(metric: VectorDistance) -> bool {
    !matches!(metric, VectorDistance::Euclidean)
}

/// From `keys`, if it's a single key whose expression is
/// `dist_fn(Property(Variable(alias), prop), query)` ordered in the metric's
/// *nearest-first* direction, return `(metric, prop, alias, query)`. Nearest-first
/// is `DESC` for the higher-is-closer metrics (cosine, dot) and `ASC` for
/// euclidean (lower distance is closer). A query ordered the other way asks for
/// the *farthest* k — which `VectorSearch` does not compute — so it is left as the
/// flat TopN.
fn single_distance_key(keys: &[OrderKey]) -> Option<(VectorDistance, String, String, Expression)> {
    if keys.len() != 1 {
        return None;
    }
    let key = &keys[0];
    let parts = distance_call_parts(&key.expression)?;
    let want = if metric_higher_is_better(parts.0) {
        OrderDirection::Desc
    } else {
        OrderDirection::Asc
    };
    if key.direction != want {
        return None;
    }
    Some(parts)
}

/// `dist_fn(Property(Variable(alias), prop), query)` →
/// `(metric, prop, alias, query)`. The same shape whether it drives an
/// `ORDER BY` key or a projection item, so the score column's alias can be
/// matched structurally (span-insensitively) across the two.
fn distance_call_parts(expr: &Expression) -> Option<(VectorDistance, String, String, Expression)> {
    let metric = distance_metric(expr)?;
    let ExpressionKind::FunctionCall { args, .. } = &expr.kind else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    // Exactly one arg is Property(Variable(alias), prop); the other is the query.
    let (alias, prop, query) = match (extract_property(&args[0]), extract_property(&args[1])) {
        (Some((a, p)), None) => (a, p, args[1].clone()),
        (None, Some((a, p))) => (a, p, args[0].clone()),
        _ => return None,
    };
    Some((metric, prop, alias, query))
}

/// Peel an optional `WHERE` Filter directly above the NodeScan, returning the
/// inner plan and the captured predicate (e.g. `d.emb IS NOT NULL`,
/// `cosine_similarity(d.emb,$q) >= 0.86`, or a label/property predicate).
fn peel_filter(plan: &LogicalPlan) -> (&LogicalPlan, Option<Expression>) {
    match plan {
        LogicalPlan::Filter { input, predicate } => (input.as_ref(), Some(predicate.clone())),
        other => (other, None),
    }
}

/// `dist_fn(...)` → its [`VectorDistance`], or `None` for a non-vector function.
fn distance_metric(expr: &Expression) -> Option<VectorDistance> {
    let ExpressionKind::FunctionCall { name, .. } = &expr.kind else {
        return None;
    };
    match name.joined().to_ascii_lowercase().as_str() {
        "cosine_similarity" => Some(VectorDistance::Cosine),
        "dot_product" => Some(VectorDistance::Dot),
        "euclidean_distance" => Some(VectorDistance::Euclidean),
        _ => None,
    }
}

/// `Property(Variable(alias), key=prop)` → `(alias, prop)`.
fn extract_property(expr: &Expression) -> Option<(String, String)> {
    let ExpressionKind::Property(pa) = &expr.kind else {
        return None;
    };
    let target = &pa.target;
    let ExpressionKind::Variable(ident) = &target.kind else {
        return None;
    };
    Some((ident.name.clone(), pa.key.name.clone()))
}

fn metric_to_storage(d: VectorDistance) -> namidb_storage::manifest::VectorMetric {
    match d {
        VectorDistance::Cosine => namidb_storage::manifest::VectorMetric::Cosine,
        VectorDistance::Dot => namidb_storage::manifest::VectorMetric::Dot,
        VectorDistance::Euclidean => namidb_storage::manifest::VectorMetric::Euclidean,
    }
}

#[cfg(all(test, feature = "vector-index"))]
mod tests {
    use super::*;
    use crate::cost::StatsCatalog;
    use crate::parser::ast::{OrderDirection, QualifiedName};
    use crate::parser::{Identifier, PropertyAccess, SourceSpan};
    use crate::plan::logical::{RowCount, VectorDistance};
    use crate::plan::{LogicalPlan, OrderKey, ProjectionItem};
    use namidb_storage::manifest::{VectorIndexDescriptor, VectorMetric};

    fn sp() -> SourceSpan {
        SourceSpan::point(0)
    }

    /// `cosine_similarity(d.emb, $q)` as an Expression.
    fn dist_call(metric_fn: &str) -> Expression {
        let prop = Expression {
            kind: ExpressionKind::Property(Box::new(PropertyAccess {
                target: Expression {
                    kind: ExpressionKind::Variable(Identifier::new("d", sp())),
                    span: sp(),
                },
                key: Identifier::new("emb", sp()),
                span: sp(),
            })),
            span: sp(),
        };
        let query = Expression {
            kind: ExpressionKind::Parameter("q".into()),
            span: sp(),
        };
        Expression {
            kind: ExpressionKind::FunctionCall {
                name: QualifiedName::single(Identifier::new(metric_fn, sp())),
                args: vec![prop, query],
                distinct: false,
            },
            span: sp(),
        }
    }

    fn knn_plan(metric_fn: &str) -> LogicalPlan {
        let call = dist_call(metric_fn);
        let scan = LogicalPlan::NodeScan {
            label: Some("Doc".into()),
            alias: "d".into(),
            predicates: vec![],
            projection: None,
        };
        let project = LogicalPlan::Project {
            input: Box::new(scan),
            items: vec![
                ProjectionItem {
                    expression: call.clone(),
                    alias: "score".into(),
                },
                ProjectionItem {
                    expression: Expression {
                        kind: ExpressionKind::Variable(Identifier::new("d", sp())),
                        span: sp(),
                    },
                    alias: "d".into(),
                },
            ],
            distinct: false,
            discard_input_bindings: true,
        };
        LogicalPlan::TopN {
            input: Box::new(project),
            keys: vec![OrderKey {
                expression: call,
                direction: OrderDirection::Desc,
            }],
            skip: RowCount::Const(0),
            limit: RowCount::Const(10),
        }
    }

    fn catalog_with_index(metric: VectorMetric) -> StatsCatalog {
        let mut m = namidb_storage::Manifest::empty(namidb_storage::Epoch::ZERO, uuid::Uuid::nil());
        m.vector_indexes.push(VectorIndexDescriptor {
            name: "doc_emb".into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim: 16,
            metric,
            r: 32,
            l_build: 64,
            alpha: 1.2,
        });
        StatsCatalog::from_manifest(&m)
    }

    #[test]
    fn rewrites_knn_to_vector_search_when_indexed() {
        let plan = knn_plan("cosine_similarity");
        let cat = catalog_with_index(VectorMetric::Cosine);
        let out = apply_vector_search(plan, &cat);
        match out {
            LogicalPlan::VectorSearch {
                label,
                alias,
                property,
                distance,
                score_alias,
                k,
                ..
            } => {
                assert_eq!(label.as_deref(), Some("Doc"));
                assert_eq!(alias, "d");
                assert_eq!(property, "emb");
                assert_eq!(distance, VectorDistance::Cosine);
                assert_eq!(score_alias, "score");
                assert_eq!(k, RowCount::Const(10));
            }
            other => panic!("expected VectorSearch, got {:?}", other.operator_name()),
        }
    }

    #[test]
    fn leaves_plan_unchanged_when_no_index() {
        let plan = knn_plan("cosine_similarity");
        let cat = StatsCatalog::empty();
        let out = apply_vector_search(plan.clone(), &cat);
        assert!(
            matches!(out, LogicalPlan::TopN { .. }),
            "no index → unchanged"
        );
    }

    #[test]
    fn metric_mismatch_leaves_plan_unchanged() {
        // cosine query, but only a dot index exists → no rewrite.
        let plan = knn_plan("cosine_similarity");
        let cat = catalog_with_index(VectorMetric::Dot);
        let out = apply_vector_search(plan, &cat);
        assert!(matches!(out, LogicalPlan::TopN { .. }));
    }

    #[test]
    fn dot_metric_rewrites_when_dot_index() {
        let plan = knn_plan("dot_product");
        let cat = catalog_with_index(VectorMetric::Dot);
        let out = apply_vector_search(plan, &cat);
        match out {
            LogicalPlan::VectorSearch { distance, .. } => {
                assert_eq!(distance, VectorDistance::Dot);
            }
            other => panic!("expected VectorSearch, got {:?}", other.operator_name()),
        }
    }

    /// `knn_plan` with a `Filter(predicate)` between the Project and the
    /// NodeScan (a `WHERE` on the match).
    fn knn_plan_with_filter(metric_fn: &str, predicate: Expression) -> LogicalPlan {
        let call = dist_call(metric_fn);
        let scan = LogicalPlan::NodeScan {
            label: Some("Doc".into()),
            alias: "d".into(),
            predicates: vec![],
            projection: None,
        };
        let filter = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate,
        };
        let project = LogicalPlan::Project {
            input: Box::new(filter),
            items: vec![
                ProjectionItem {
                    expression: call.clone(),
                    alias: "score".into(),
                },
                ProjectionItem {
                    expression: Expression {
                        kind: ExpressionKind::Variable(Identifier::new("d", sp())),
                        span: sp(),
                    },
                    alias: "d".into(),
                },
            ],
            distinct: false,
            discard_input_bindings: true,
        };
        LogicalPlan::TopN {
            input: Box::new(project),
            keys: vec![OrderKey {
                expression: call,
                direction: OrderDirection::Desc,
            }],
            skip: RowCount::Const(0),
            limit: RowCount::Const(10),
        }
    }

    #[test]
    fn captures_scan_filter_into_post_filter() {
        // A WHERE that references only the searched binding is folded into
        // `post_filter` instead of being dropped.
        let pred = dist_call("cosine_similarity"); // references only `d`
        let plan = knn_plan_with_filter("cosine_similarity", pred);
        let cat = catalog_with_index(VectorMetric::Cosine);
        match apply_vector_search(plan, &cat) {
            LogicalPlan::VectorSearch { post_filter, .. } => {
                assert!(post_filter.is_some(), "scan WHERE must be captured");
            }
            other => panic!("expected VectorSearch, got {:?}", other.operator_name()),
        }
    }

    #[test]
    fn cross_binding_filter_blocks_rewrite() {
        // A predicate touching another binding must NOT be swallowed: bail so
        // the Filter survives and runs via the flat path.
        let pred = Expression {
            kind: ExpressionKind::Variable(Identifier::new("other", sp())),
            span: sp(),
        };
        let plan = knn_plan_with_filter("cosine_similarity", pred);
        let cat = catalog_with_index(VectorMetric::Cosine);
        let out = apply_vector_search(plan, &cat);
        assert!(
            matches!(out, LogicalPlan::TopN { .. }),
            "cross-binding filter → no rewrite"
        );
    }

    #[test]
    fn plain_knn_has_no_post_filter() {
        let plan = knn_plan("cosine_similarity");
        let cat = catalog_with_index(VectorMetric::Cosine);
        match apply_vector_search(plan, &cat) {
            LogicalPlan::VectorSearch { post_filter, .. } => assert!(post_filter.is_none()),
            other => panic!("expected VectorSearch, got {:?}", other.operator_name()),
        }
    }

    /// Terminal-`RETURN` shape (the common KNN): the Project sits *outside* the
    /// TopN — `Project[title, dist AS score]{ TopN{ [Filter] NodeScan{Doc, d} } }`.
    fn knn_plan_terminal_return(metric_fn: &str, filter: Option<Expression>) -> LogicalPlan {
        let call = dist_call(metric_fn);
        let scan = LogicalPlan::NodeScan {
            label: Some("Doc".into()),
            alias: "d".into(),
            predicates: vec![],
            projection: None,
        };
        let topn_input = match filter {
            Some(predicate) => LogicalPlan::Filter {
                input: Box::new(scan),
                predicate,
            },
            None => scan,
        };
        let topn = LogicalPlan::TopN {
            input: Box::new(topn_input),
            keys: vec![OrderKey {
                expression: call.clone(),
                direction: OrderDirection::Desc,
            }],
            skip: RowCount::Const(0),
            limit: RowCount::Const(10),
        };
        LogicalPlan::Project {
            input: Box::new(topn),
            items: vec![
                // A non-distance projected column (`RETURN d.title AS title`),
                // here a bare binding ref — the matcher must ignore it.
                ProjectionItem {
                    expression: Expression {
                        kind: ExpressionKind::Variable(Identifier::new("d", sp())),
                        span: sp(),
                    },
                    alias: "title".into(),
                },
                ProjectionItem {
                    expression: call,
                    alias: "score".into(),
                },
            ],
            distinct: false,
            discard_input_bindings: true,
        }
    }

    #[test]
    fn rewrites_terminal_return_knn() {
        // The TopN ranking sub-tree collapses to VectorSearch; the outer Project
        // (the RETURN) is preserved on top, and the score column keeps its alias.
        let plan = knn_plan_terminal_return("cosine_similarity", None);
        let cat = catalog_with_index(VectorMetric::Cosine);
        match apply_vector_search(plan, &cat) {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::VectorSearch {
                    alias,
                    property,
                    score_alias,
                    post_filter,
                    ..
                } => {
                    assert_eq!(alias, "d");
                    assert_eq!(property, "emb");
                    assert_eq!(score_alias, "score");
                    assert!(post_filter.is_none());
                }
                other => panic!("expected VectorSearch, got {:?}", other.operator_name()),
            },
            other => panic!(
                "expected Project(VectorSearch), got {:?}",
                other.operator_name()
            ),
        }
    }

    #[test]
    fn terminal_return_folds_where_into_post_filter() {
        // A `WHERE` on the match (e.g. `d.emb IS NOT NULL`, or a `>= 0.86`
        // threshold) sits under the TopN; it must be captured, not dropped.
        let pred = dist_call("cosine_similarity"); // references only `d`
        let plan = knn_plan_terminal_return("cosine_similarity", Some(pred));
        let cat = catalog_with_index(VectorMetric::Cosine);
        match apply_vector_search(plan, &cat) {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::VectorSearch { post_filter, .. } => {
                    assert!(post_filter.is_some(), "match WHERE must be captured");
                }
                other => panic!("expected VectorSearch, got {:?}", other.operator_name()),
            },
            other => panic!(
                "expected Project(VectorSearch), got {:?}",
                other.operator_name()
            ),
        }
    }

    /// `knn_plan` in the non-terminal WITH shape with an explicit metric fn and
    /// order direction, for the orientation tests.
    fn knn_plan_dir(metric_fn: &str, dir: OrderDirection) -> LogicalPlan {
        let mut plan = knn_plan(metric_fn);
        if let LogicalPlan::TopN { ref mut keys, .. } = plan {
            keys[0].direction = dir;
        }
        plan
    }

    #[test]
    fn euclidean_rewrites_asc_not_desc() {
        // Euclidean is nearest-first ASC; the ANN serves nearest-k, so only ASC
        // (with a euclidean index) rewrites. DESC asks for the farthest k.
        let cat = catalog_with_index(VectorMetric::Euclidean);
        let asc = knn_plan_dir("euclidean_distance", OrderDirection::Asc);
        assert!(
            matches!(
                apply_vector_search(asc, &cat),
                LogicalPlan::VectorSearch { .. }
            ),
            "euclidean ASC + index → VectorSearch"
        );
        let desc = knn_plan_dir("euclidean_distance", OrderDirection::Desc);
        assert!(
            matches!(apply_vector_search(desc, &cat), LogicalPlan::TopN { .. }),
            "euclidean DESC (farthest-k) → unchanged"
        );
    }

    #[test]
    fn cosine_asc_does_not_rewrite() {
        // Cosine nearest-first is DESC; ASC asks for the least-similar k.
        let cat = catalog_with_index(VectorMetric::Cosine);
        let asc = knn_plan_dir("cosine_similarity", OrderDirection::Asc);
        assert!(matches!(
            apply_vector_search(asc, &cat),
            LogicalPlan::TopN { .. }
        ));
    }

    #[test]
    fn unbounded_limit_is_not_rewritten() {
        // `ORDER BY … DESC` with no `LIMIT` lowers to k = u64::MAX; rewriting it
        // would overflow `Vec::with_capacity(k)` in the executor. Stay flat.
        let mut plan = knn_plan("cosine_similarity");
        if let LogicalPlan::TopN { ref mut limit, .. } = plan {
            *limit = RowCount::Const(u64::MAX);
        }
        let cat = catalog_with_index(VectorMetric::Cosine);
        assert!(matches!(
            apply_vector_search(plan, &cat),
            LogicalPlan::TopN { .. }
        ));
    }

    #[test]
    fn pushed_scan_predicate_blocks_rewrite() {
        use namidb_storage::sst::predicates::ScanPredicate;
        // A predicate already folded into NodeScan.predicates (by predicate
        // pushdown) must not be silently dropped — refuse the rewrite so the flat
        // path keeps honouring it.
        let mut plan = knn_plan("cosine_similarity");
        if let LogicalPlan::TopN { ref mut input, .. } = plan {
            if let LogicalPlan::Project { ref mut input, .. } = **input {
                if let LogicalPlan::NodeScan {
                    ref mut predicates, ..
                } = **input
                {
                    predicates.push(ScanPredicate::IsNotNull {
                        column: "emb".into(),
                    });
                }
            }
        }
        let cat = catalog_with_index(VectorMetric::Cosine);
        assert!(
            matches!(apply_vector_search(plan, &cat), LogicalPlan::TopN { .. }),
            "pushed scan predicate → no rewrite (flat path honours it)"
        );
    }

    #[test]
    fn rewrites_knn_nested_in_union_branch() {
        // A KNN in a UNION branch is collapsed too (bottom-up recursion).
        let cat = catalog_with_index(VectorMetric::Cosine);
        let plan = LogicalPlan::Union {
            left: Box::new(knn_plan("cosine_similarity")),
            right: Box::new(knn_plan("cosine_similarity")),
            all: true,
        };
        match apply_vector_search(plan, &cat) {
            LogicalPlan::Union { left, right, .. } => {
                assert!(matches!(*left, LogicalPlan::VectorSearch { .. }));
                assert!(matches!(*right, LogicalPlan::VectorSearch { .. }));
            }
            other => panic!("expected Union, got {:?}", other.operator_name()),
        }
    }

    #[test]
    fn terminal_return_cross_binding_filter_blocks_rewrite() {
        // A predicate touching another binding must bail: the plan stays
        // Project(TopN(...)), running via the flat path.
        let pred = Expression {
            kind: ExpressionKind::Variable(Identifier::new("other", sp())),
            span: sp(),
        };
        let plan = knn_plan_terminal_return("cosine_similarity", Some(pred));
        let cat = catalog_with_index(VectorMetric::Cosine);
        match apply_vector_search(plan, &cat) {
            LogicalPlan::Project { input, .. } => {
                assert!(
                    matches!(*input, LogicalPlan::TopN { .. }),
                    "cross-binding filter → TopN preserved (flat path)"
                );
            }
            other => panic!("expected Project(TopN), got {:?}", other.operator_name()),
        }
    }
}
