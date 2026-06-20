//! `vector_search` rewrite (RFC-030, `vector-index`): turn a flat KNN shape
//! into a [`LogicalPlan::VectorSearch`] when a matching Vamana index exists.
//!
//! The flat KNN shape lowered from
//! `MATCH (n:L) WITH n ORDER BY cosine_similarity(n.emb, $q) DESC LIMIT 10` is
//! `TopN{ keys:[dist(prop,$q) DESC], limit:k, Project{ [.., dist AS score],
//! Filter{ NodeScan{L, n} } } }`. When the catalog has a `VectorIndexDescriptor`
//! for `(L, prop, metric)`, the whole chain collapses to a `VectorSearch` leaf
//! that the executor serves from the index (falling back to flat scan when no
//! index matches). Conservative: any `SKIP`, a non-DESC key, multiple keys, a
//! non-vector key function, a `DISTINCT` projection, or a missing index leaves
//! the plan unchanged.
//!
//! Registered in `optimize::mod` right after `unique_lookup`, so the downstream
//! pushdowns (which treat `VectorSearch` as an opaque leaf) see the new
//! operator and don't re-introduce a Filter above it.

use crate::cost::StatsCatalog;
use crate::parser::ast::OrderDirection;
use crate::parser::{Expression, ExpressionKind};
use crate::plan::logical::{LogicalPlan, OrderKey, RowCount, VectorDistance};

/// Run the rewrite over `plan`. No-op when no index matches.
pub fn apply_vector_search(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
    let plan = recurse(plan);
    if let Some(vs) = try_match(&plan, catalog) {
        vs
    } else {
        plan
    }
}

/// Bottom-up recursion through the single-input operators that wrap a KNN
/// (TopN / Project / Filter). A KNN is a single-chain shape, so this is enough
/// to catch one nested under an outer clause. (Pure structural rebuild; the
/// index match is decided in `try_match` against the catalog.)
fn recurse(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } => LogicalPlan::TopN {
            input: Box::new(recurse(*input)),
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
            input: Box::new(recurse(*input)),
            items,
            distinct,
            discard_input_bindings,
        },
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(recurse(*input)),
            predicate,
        },
        other => other,
    }
}

/// Borrow-based matcher: if `plan` is the KNN chain AND a backing index exists,
/// return the replacement `VectorSearch`.
fn try_match(plan: &LogicalPlan, catalog: &StatsCatalog) -> Option<LogicalPlan> {
    let LogicalPlan::TopN {
        keys,
        skip,
        limit,
        input,
    } = plan
    else {
        return None;
    };
    // No SKIP; exactly one DESC key on a vector-distance function.
    if !matches!(skip, RowCount::Const(0)) {
        return None;
    }
    let (distance, prop, alias, query) = single_distance_key(keys)?;

    let LogicalPlan::Project {
        items,
        distinct: false,
        discard_input_bindings: _,
        input: proj_input,
    } = input.as_ref()
    else {
        return None;
    };

    // Score column alias = the projection item equal to the distance call.
    let score_alias = items
        .iter()
        .find(|it| distance_key_matches(&it.expression, keys))
        .map(|it| it.alias.clone())
        .unwrap_or_else(|| "score".to_string());

    // Peel an optional Filter (e.g. `WHERE n.emb IS NOT NULL`).
    let scan = match proj_input.as_ref() {
        LogicalPlan::Filter { input, .. } => input.as_ref(),
        other => other,
    };
    let LogicalPlan::NodeScan {
        label: Some(label),
        alias: scan_alias,
        ..
    } = scan
    else {
        return None;
    };
    if scan_alias != &alias {
        return None;
    }

    // Index must exist for (label, prop, metric).
    catalog.vector_index_for(label, &prop, metric_to_storage(distance))?;

    Some(LogicalPlan::VectorSearch {
        label: Some(label.clone()),
        alias,
        property: prop,
        query,
        k: limit.clone(),
        distance,
        score_alias,
    })
}

/// From `keys`, if it's a single DESC key whose expression is
/// `dist_fn(Property(Variable(alias), prop), query)`, return
/// `(metric, prop, alias, query)`.
fn single_distance_key(
    keys: &[OrderKey],
) -> Option<(VectorDistance, String, String, Expression)> {
    if keys.len() != 1 {
        return None;
    }
    let key = &keys[0];
    if !matches!(key.direction, OrderDirection::Desc) {
        return None;
    }
    let metric = distance_metric(&key.expression)?;
    let ExpressionKind::FunctionCall { args, .. } = &key.expression.kind else {
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

/// `true` if `expr` is the same distance-function call that drives `keys[0]`.
fn distance_key_matches(expr: &Expression, keys: &[OrderKey]) -> bool {
    if keys.len() != 1 {
        return false;
    }
    expr == &keys[0].expression
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
        let mut m = namidb_storage::Manifest::empty(
            namidb_storage::Epoch::ZERO,
            uuid::Uuid::nil(),
        );
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
        assert!(matches!(out, LogicalPlan::TopN { .. }), "no index → unchanged");
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
}
