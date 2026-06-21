//! Predicate selectivity estimation (RFC-010 §2).
//!
//! `selectivity(expr, &bindings) -> f64` returns the fraction of input
//! rows that the predicate retains. The estimator is fail-soft: any
//! sub-expression it cannot reason about falls back to the table of
//! defaults below.
//!
//! Defaults (RFC-010 §2):
//!
//! | Term | Default |
//! |-------------------------------|---------|
//! | `prop = literal` (no NDV) | 0.10 |
//! | `prop < / > literal` | 0.33 |
//! | `IS NULL` | 0.05 |
//! | `STARTS WITH` / `CONTAINS` | 0.10 |
//! | unknown | 0.50 |
//!
//! Selectivity is always clamped to `[0.0, 1.0]`.

use std::collections::BTreeMap;

use namidb_storage::sst::stats::StatScalar;

use super::stats::{LabelStats, PropStats, StatsCatalog};
use crate::parser::ast::{BinaryOp, Expression, ExpressionKind, Literal, StringOp, UnaryOp};

/// Bindings visible to the predicate, mapping alias → label stats. The
/// optimizer fills this from the upstream plan's [`crate::cost::Cardinality::bindings`].
#[derive(Debug, Default, Clone)]
pub struct BindingStats<'a> {
    pub by_alias: BTreeMap<String, &'a LabelStats>,
    /// Optional catalog for rules that need namespace-wide counts not tied to
    /// a bound alias — currently the secondary-label fraction for `__label_eq`
    /// (`MATCH (n:A:B)`). `None` falls back to the old defensive behaviour.
    pub catalog: Option<&'a StatsCatalog>,
}

impl<'a> BindingStats<'a> {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn with(mut self, alias: impl Into<String>, stats: &'a LabelStats) -> Self {
        self.by_alias.insert(alias.into(), stats);
        self
    }

    pub fn with_catalog(mut self, catalog: &'a StatsCatalog) -> Self {
        self.catalog = Some(catalog);
        self
    }

    pub fn prop_stats(&self, alias: &str, prop: &str) -> Option<&'a PropStats> {
        self.by_alias
            .get(alias)
            .and_then(|l| l.properties.get(prop))
    }
}

const FALLBACK_EQ: f64 = 0.10;
const FALLBACK_RANGE: f64 = 0.33;
const FALLBACK_IS_NULL: f64 = 0.05;
const FALLBACK_STRING_TEST: f64 = 0.10;
const FALLBACK_UNKNOWN: f64 = 0.50;
const FALLBACK_IN_PER_ELEM: f64 = 0.10;

/// Estimate the fraction of input rows that satisfy `expr`.
pub fn selectivity(expr: &Expression, bindings: &BindingStats<'_>) -> f64 {
    let s = sel_inner(expr, bindings);
    clamp01(s)
}

fn sel_inner(expr: &Expression, bindings: &BindingStats<'_>) -> f64 {
    match &expr.kind {
        // ─── boolean combinators ───────────────────────────────────────
        ExpressionKind::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => sel_inner(left, bindings) * sel_inner(right, bindings),
        ExpressionKind::Binary {
            op: BinaryOp::Or,
            left,
            right,
        } => {
            let a = sel_inner(left, bindings);
            let b = sel_inner(right, bindings);
            a + b - a * b
        }
        ExpressionKind::Binary {
            op: BinaryOp::Xor,
            left,
            right,
        } => {
            let a = sel_inner(left, bindings);
            let b = sel_inner(right, bindings);
            a + b - 2.0 * a * b
        }
        ExpressionKind::Unary {
            op: UnaryOp::Not,
            expr,
        } => 1.0 - sel_inner(expr, bindings),

        // ─── comparison ────────────────────────────────────────────────
        ExpressionKind::Binary { op, left, right }
            if matches!(
                op,
                BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
            ) =>
        {
            sel_comparison(*op, left, right, bindings)
        }

        // ─── IS NULL / IS NOT NULL ─────────────────────────────────────
        ExpressionKind::IsNull { expr, negated } => {
            let base = sel_is_null(expr, bindings);
            if *negated {
                1.0 - base
            } else {
                base
            }
        }

        // ─── IN / string tests ─────────────────────────────────────────
        ExpressionKind::In { item, list } => sel_in(item, list, bindings),
        ExpressionKind::StringTest { .. } => FALLBACK_STRING_TEST,

        // ─── literals & special cases ──────────────────────────────────
        ExpressionKind::Literal(Literal::Boolean(true)) => 1.0,
        ExpressionKind::Literal(Literal::Boolean(false)) => 0.0,
        ExpressionKind::Literal(Literal::Null) => 0.0,
        ExpressionKind::Exists(_) | ExpressionKind::ExistsSubquery(_) => FALLBACK_UNKNOWN,

        // ─── synthetic engine functions ────────────────────────────────
        ExpressionKind::FunctionCall { name, args, .. }
            if name
                .segments
                .first()
                .map(|s| s.name.eq_ignore_ascii_case("__label_eq"))
                .unwrap_or(false) =>
        {
            sel_label_eq(args, bindings)
        }

        // Anything else (function calls, CASE, list/map literals, etc.).
        _ => FALLBACK_UNKNOWN,
    }
}

/// Selectivity of a synthetic `__label_eq(alias, "L")` filter: the fraction of
/// nodes carrying label `L`, approximated as `node_count(L) / total_nodes`
/// (independence). `MATCH (n:A:B)` scans `A` and applies this for `B`, so the
/// estimate becomes `node_count(A) * node_count(B)/total_nodes` — an
/// intersection guess rather than the old `node_count(A)` (which ignored `B`).
/// Falls back to 1.0 (no shrink) when the catalog or label is unknown.
fn sel_label_eq(args: &[Expression], bindings: &BindingStats<'_>) -> f64 {
    let (Some(catalog), Some(label)) = (bindings.catalog, label_literal(args)) else {
        return 1.0;
    };
    let total = catalog.total_nodes();
    if total == 0 {
        return 1.0;
    }
    match catalog.label(label) {
        Some(ls) => ls.node_count as f64 / total as f64,
        None => 1.0,
    }
}

/// Pull the label string out of `__label_eq(target, "Label")`'s second arg.
fn label_literal(args: &[Expression]) -> Option<&str> {
    match args.get(1).map(|e| &e.kind) {
        Some(ExpressionKind::Literal(Literal::String(s))) => Some(s.as_str()),
        _ => None,
    }
}

/// Selectivity for an `IS NULL` test. We only know the stats when the
/// target is a `Property(alias.prop)` and the binding is in scope.
fn sel_is_null(expr: &Expression, bindings: &BindingStats<'_>) -> f64 {
    if let Some((alias, prop)) = property_access(expr) {
        if let Some(ps) = bindings.prop_stats(&alias, &prop) {
            let total = ps.null_count + ps.non_null_count;
            if total == 0 {
                return FALLBACK_IS_NULL;
            }
            return ps.null_count as f64 / total as f64;
        }
    }
    FALLBACK_IS_NULL
}

fn sel_comparison(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    bindings: &BindingStats<'_>,
) -> f64 {
    // Try both orderings — `prop = lit` and `lit = prop` should yield
    // identical estimates.
    if let Some((alias, prop, lit)) = property_op_literal(left, right) {
        if let Some(ps) = bindings.prop_stats(&alias, &prop) {
            return sel_comparison_on_stats(op, ps, &lit);
        }
    }
    if let Some((alias, prop, lit)) = property_op_literal(right, left) {
        // Operator is mirrored when the literal sits on the left.
        let mirrored = mirror_binary_op(op);
        if let Some(ps) = bindings.prop_stats(&alias, &prop) {
            return sel_comparison_on_stats(mirrored, ps, &lit);
        }
    }
    op_fallback(op)
}

fn sel_comparison_on_stats(op: BinaryOp, ps: &PropStats, lit: &Literal) -> f64 {
    let Some(lit_scalar) = literal_to_scalar(lit) else {
        return op_fallback(op);
    };
    match op {
        BinaryOp::Eq => {
            if let Some(ndv) = ps.ndv {
                if ndv == 0 {
                    return FALLBACK_EQ;
                }
                return 1.0 / ndv as f64;
            }
            FALLBACK_EQ
        }
        BinaryOp::Ne => 1.0 - sel_comparison_on_stats(BinaryOp::Eq, ps, lit),
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
            range_selectivity(op, &lit_scalar, ps).unwrap_or(FALLBACK_RANGE)
        }
        _ => FALLBACK_UNKNOWN,
    }
}

fn op_fallback(op: BinaryOp) -> f64 {
    match op {
        BinaryOp::Eq => FALLBACK_EQ,
        BinaryOp::Ne => 1.0 - FALLBACK_EQ,
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => FALLBACK_RANGE,
        _ => FALLBACK_UNKNOWN,
    }
}

fn range_selectivity(op: BinaryOp, lit: &StatScalar, ps: &PropStats) -> Option<f64> {
    let min = ps.min.as_ref()?;
    let max = ps.max.as_ref()?;
    let min_f = scalar_to_f64(min)?;
    let max_f = scalar_to_f64(max)?;
    let lit_f = scalar_to_f64(lit)?;
    if max_f <= min_f {
        // Degenerate column.
        return Some(match op {
            BinaryOp::Lt | BinaryOp::Le => {
                if lit_f >= max_f {
                    1.0
                } else {
                    0.0
                }
            }
            BinaryOp::Gt | BinaryOp::Ge => {
                if lit_f <= min_f {
                    1.0
                } else {
                    0.0
                }
            }
            _ => FALLBACK_RANGE,
        });
    }
    let range = max_f - min_f;
    let below = ((lit_f - min_f) / range).clamp(0.0, 1.0);
    Some(match op {
        BinaryOp::Lt | BinaryOp::Le => below,
        BinaryOp::Gt | BinaryOp::Ge => 1.0 - below,
        _ => FALLBACK_RANGE,
    })
}

fn sel_in(item: &Expression, list: &Expression, bindings: &BindingStats<'_>) -> f64 {
    let list_len = match &list.kind {
        ExpressionKind::List(items) => items.len() as f64,
        _ => return FALLBACK_UNKNOWN,
    };
    if let Some((alias, prop)) = property_access(item) {
        if let Some(ps) = bindings.prop_stats(&alias, &prop) {
            if let Some(ndv) = ps.ndv {
                if ndv > 0 {
                    return (list_len / ndv as f64).min(1.0);
                }
            }
        }
    }
    (list_len * FALLBACK_IN_PER_ELEM).min(1.0)
}

// ─────────────── shape extraction helpers ──────────────────────────────

/// Return `(alias, prop_name)` if `expr` is a property access shaped
/// like `alias.prop` over a single hop.
fn property_access(expr: &Expression) -> Option<(String, String)> {
    let ExpressionKind::Property(p) = &expr.kind else {
        return None;
    };
    let target = &p.target;
    let ExpressionKind::Variable(alias) = &target.kind else {
        return None;
    };
    Some((alias.name.clone(), p.key.name.clone()))
}

fn property_op_literal(a: &Expression, b: &Expression) -> Option<(String, String, Literal)> {
    let (alias, prop) = property_access(a)?;
    let lit = literal_from_expr(b)?;
    Some((alias, prop, lit))
}

fn literal_from_expr(expr: &Expression) -> Option<Literal> {
    match &expr.kind {
        ExpressionKind::Literal(l) => Some(l.clone()),
        _ => None,
    }
}

fn literal_to_scalar(lit: &Literal) -> Option<StatScalar> {
    match lit {
        Literal::Boolean(b) => Some(StatScalar::Bool(*b)),
        Literal::Integer(i) => Some(StatScalar::Int64(*i)),
        Literal::Float(f) => Some(StatScalar::Float64(*f)),
        Literal::String(s) => Some(StatScalar::Utf8(s.clone())),
        Literal::Null => None,
    }
}

fn scalar_to_f64(s: &StatScalar) -> Option<f64> {
    match s {
        StatScalar::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        StatScalar::Int32(i) => Some(*i as f64),
        StatScalar::Int64(i) => Some(*i as f64),
        StatScalar::Float32(f) => Some(*f as f64),
        StatScalar::Float64(f) => Some(*f),
        StatScalar::Date32(d) => Some(*d as f64),
        StatScalar::TimestampMicrosUtc(t) => Some(*t as f64),
        // Strings/binary don't map to a numeric range; the caller falls
        // back to FALLBACK_RANGE.
        StatScalar::Utf8(_) | StatScalar::LargeUtf8(_) | StatScalar::Binary(_) => None,
    }
}

fn mirror_binary_op(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::Le => BinaryOp::Ge,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::Ge => BinaryOp::Le,
        other => other,
    }
}

fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        FALLBACK_UNKNOWN
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// Synthesise an `Expression` equivalent to `predicate` against the
/// given `alias`. Used by the cardinality estimator (RFC-013 §7) to
/// reuse the existing selectivity heuristics for predicates that have
/// been pushed into a `NodeScan.predicates`. The synthesised tree is
/// not parser-emitted — its `SourceSpan` is a `point(0)` placeholder.
pub fn scan_predicate_to_expression(
    predicate: &namidb_storage::sst::predicates::ScanPredicate,
    alias: &str,
) -> Expression {
    use namidb_storage::sst::predicates::ScanPredicate as P;
    let span = crate::parser::SourceSpan::point(0);
    let prop = |column: &str| Expression {
        kind: ExpressionKind::Property(Box::new(crate::parser::ast::PropertyAccess {
            target: Expression {
                kind: ExpressionKind::Variable(crate::parser::ast::Identifier {
                    name: alias.into(),
                    span,
                    quoted: false,
                }),
                span,
            },
            key: crate::parser::ast::Identifier {
                name: column.into(),
                span,
                quoted: false,
            },
            span,
        })),
        span,
    };
    let lit = |l: Literal| Expression {
        kind: ExpressionKind::Literal(l),
        span,
    };
    match predicate {
        P::Eq { column, value } => binary(
            BinaryOp::Eq,
            prop(column),
            lit(stat_to_literal(value)),
            span,
        ),
        P::Lt { column, value } => binary(
            BinaryOp::Lt,
            prop(column),
            lit(stat_to_literal(value)),
            span,
        ),
        P::LtEq { column, value } => binary(
            BinaryOp::Le,
            prop(column),
            lit(stat_to_literal(value)),
            span,
        ),
        P::Gt { column, value } => binary(
            BinaryOp::Gt,
            prop(column),
            lit(stat_to_literal(value)),
            span,
        ),
        P::GtEq { column, value } => binary(
            BinaryOp::Ge,
            prop(column),
            lit(stat_to_literal(value)),
            span,
        ),
        P::IsNull { column } => Expression {
            kind: ExpressionKind::IsNull {
                expr: Box::new(prop(column)),
                negated: false,
            },
            span,
        },
        P::IsNotNull { column } => Expression {
            kind: ExpressionKind::IsNull {
                expr: Box::new(prop(column)),
                negated: true,
            },
            span,
        },
        P::In { column, values } => {
            let items: Vec<Expression> = values.iter().map(|v| lit(stat_to_literal(v))).collect();
            Expression {
                kind: ExpressionKind::In {
                    item: Box::new(prop(column)),
                    list: Box::new(Expression {
                        kind: ExpressionKind::List(items),
                        span,
                    }),
                },
                span,
            }
        }
    }
}

fn binary(
    op: BinaryOp,
    l: Expression,
    r: Expression,
    span: crate::parser::SourceSpan,
) -> Expression {
    Expression {
        kind: ExpressionKind::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        },
        span,
    }
}

fn stat_to_literal(s: &StatScalar) -> Literal {
    match s {
        StatScalar::Bool(b) => Literal::Boolean(*b),
        StatScalar::Int32(n) => Literal::Integer(*n as i64),
        StatScalar::Int64(n) => Literal::Integer(*n),
        StatScalar::Float32(f) => Literal::Float(*f as f64),
        StatScalar::Float64(f) => Literal::Float(*f),
        StatScalar::Utf8(s) => Literal::String(s.clone()),
        StatScalar::LargeUtf8(s) => Literal::String(s.clone()),
        // Bytes / Date / Timestamp project onto a string fallback so the
        // existing literal_to_scalar path still resolves to a usable
        // StatScalar comparison.
        StatScalar::Binary(b) => Literal::String(String::from_utf8_lossy(b).into_owned()),
        StatScalar::Date32(n) => Literal::Integer(*n as i64),
        StatScalar::TimestampMicrosUtc(n) => Literal::Integer(*n),
    }
}

// Keep StringOp imported so the matcher above stays warning-free even
// when fallback-only.
#[allow(dead_code)]
fn _unused_string_op(_k: StringOp) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{Identifier, PropertyAccess, QualifiedName};
    use crate::parser::SourceSpan;

    fn span() -> SourceSpan {
        SourceSpan::point(0)
    }

    fn var(name: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Variable(Identifier::new(name, span())),
            span: span(),
        }
    }

    fn lit_int(n: i64) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::Integer(n)),
            span: span(),
        }
    }

    fn lit_str(s: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::String(s.to_string())),
            span: span(),
        }
    }

    fn prop_access(alias: &str, prop: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Property(Box::new(PropertyAccess {
                target: var(alias),
                key: Identifier::new(prop, span()),
                span: span(),
            })),
            span: span(),
        }
    }

    fn binary(op: BinaryOp, l: Expression, r: Expression) -> Expression {
        Expression {
            kind: ExpressionKind::Binary {
                op,
                left: Box::new(l),
                right: Box::new(r),
            },
            span: span(),
        }
    }

    fn label_with_age_stats(
        min: i64,
        max: i64,
        nulls: u64,
        non_nulls: u64,
        ndv: Option<u64>,
    ) -> LabelStats {
        let mut props = BTreeMap::new();
        props.insert(
            "age".into(),
            PropStats {
                null_count: nulls,
                non_null_count: non_nulls,
                min: Some(StatScalar::Int64(min)),
                max: Some(StatScalar::Int64(max)),
                ndv,
                unique: false,
                indexed: false,
            },
        );
        LabelStats {
            name: "Person".into(),
            node_count: nulls + non_nulls,
            properties: props,
        }
    }

    #[test]
    fn eq_with_ndv_uses_one_over_ndv() {
        let stats = label_with_age_stats(0, 100, 0, 1000, Some(10));
        let b = BindingStats::empty().with("a", &stats);
        let expr = binary(BinaryOp::Eq, prop_access("a", "age"), lit_int(30));
        let s = selectivity(&expr, &b);
        assert!((s - 0.1).abs() < 1e-12);
    }

    #[test]
    fn eq_without_ndv_falls_back_to_default() {
        let stats = label_with_age_stats(0, 100, 0, 1000, None);
        let b = BindingStats::empty().with("a", &stats);
        let expr = binary(BinaryOp::Eq, prop_access("a", "age"), lit_int(30));
        let s = selectivity(&expr, &b);
        assert!((s - FALLBACK_EQ).abs() < 1e-12);
    }

    #[test]
    fn range_lt_uses_min_max() {
        // age min=0, max=100. age < 25 → 0.25.
        let stats = label_with_age_stats(0, 100, 0, 1000, None);
        let b = BindingStats::empty().with("a", &stats);
        let expr = binary(BinaryOp::Lt, prop_access("a", "age"), lit_int(25));
        let s = selectivity(&expr, &b);
        assert!((s - 0.25).abs() < 1e-12);
    }

    #[test]
    fn range_gt_uses_min_max() {
        let stats = label_with_age_stats(0, 100, 0, 1000, None);
        let b = BindingStats::empty().with("a", &stats);
        let expr = binary(BinaryOp::Gt, prop_access("a", "age"), lit_int(75));
        let s = selectivity(&expr, &b);
        assert!((s - 0.25).abs() < 1e-12);
    }

    #[test]
    fn mirrored_literal_left_uses_correct_branch() {
        // 75 < age ≡ age > 75 → 0.25.
        let stats = label_with_age_stats(0, 100, 0, 1000, None);
        let b = BindingStats::empty().with("a", &stats);
        let expr = binary(BinaryOp::Lt, lit_int(75), prop_access("a", "age"));
        let s = selectivity(&expr, &b);
        assert!((s - 0.25).abs() < 1e-12);
    }

    #[test]
    fn and_multiplies_branches() {
        let stats = label_with_age_stats(0, 100, 0, 1000, Some(10));
        let b = BindingStats::empty().with("a", &stats);
        let eq = binary(BinaryOp::Eq, prop_access("a", "age"), lit_int(30));
        let lt = binary(BinaryOp::Lt, prop_access("a", "age"), lit_int(50));
        let and = binary(BinaryOp::And, eq, lt);
        let s = selectivity(&and, &b);
        // 0.1 * 0.5 = 0.05.
        assert!((s - 0.05).abs() < 1e-12);
    }

    #[test]
    fn or_uses_inclusion_exclusion() {
        let stats = label_with_age_stats(0, 100, 0, 1000, Some(10));
        let b = BindingStats::empty().with("a", &stats);
        let lt = binary(BinaryOp::Lt, prop_access("a", "age"), lit_int(25)); // 0.25
        let gt = binary(BinaryOp::Gt, prop_access("a", "age"), lit_int(75)); // 0.25
        let or = binary(BinaryOp::Or, lt, gt);
        let s = selectivity(&or, &b);
        // 0.25 + 0.25 - 0.0625 = 0.4375.
        assert!((s - 0.4375).abs() < 1e-12);
    }

    #[test]
    fn not_inverts() {
        let stats = label_with_age_stats(0, 100, 0, 1000, Some(10));
        let b = BindingStats::empty().with("a", &stats);
        let eq = binary(BinaryOp::Eq, prop_access("a", "age"), lit_int(30));
        let not = Expression {
            kind: ExpressionKind::Unary {
                op: UnaryOp::Not,
                expr: Box::new(eq),
            },
            span: span(),
        };
        let s = selectivity(&not, &b);
        assert!((s - 0.9).abs() < 1e-12);
    }

    #[test]
    fn is_null_uses_null_count() {
        let stats = label_with_age_stats(0, 100, 50, 50, None);
        let b = BindingStats::empty().with("a", &stats);
        let expr = Expression {
            kind: ExpressionKind::IsNull {
                expr: Box::new(prop_access("a", "age")),
                negated: false,
            },
            span: span(),
        };
        let s = selectivity(&expr, &b);
        assert!((s - 0.5).abs() < 1e-12);
    }

    #[test]
    fn is_not_null_complements() {
        let stats = label_with_age_stats(0, 100, 50, 50, None);
        let b = BindingStats::empty().with("a", &stats);
        let expr = Expression {
            kind: ExpressionKind::IsNull {
                expr: Box::new(prop_access("a", "age")),
                negated: true,
            },
            span: span(),
        };
        let s = selectivity(&expr, &b);
        assert!((s - 0.5).abs() < 1e-12);
    }

    #[test]
    fn in_with_ndv() {
        let stats = label_with_age_stats(0, 100, 0, 1000, Some(10));
        let b = BindingStats::empty().with("a", &stats);
        let list = Expression {
            kind: ExpressionKind::List(vec![lit_int(1), lit_int(2), lit_int(3)]),
            span: span(),
        };
        let expr = Expression {
            kind: ExpressionKind::In {
                item: Box::new(prop_access("a", "age")),
                list: Box::new(list),
            },
            span: span(),
        };
        let s = selectivity(&expr, &b);
        // 3 / 10 = 0.3.
        assert!((s - 0.3).abs() < 1e-12);
    }

    #[test]
    fn in_without_ndv_uses_fallback() {
        let stats = label_with_age_stats(0, 100, 0, 1000, None);
        let b = BindingStats::empty().with("a", &stats);
        let list = Expression {
            kind: ExpressionKind::List(vec![lit_int(1), lit_int(2)]),
            span: span(),
        };
        let expr = Expression {
            kind: ExpressionKind::In {
                item: Box::new(prop_access("a", "age")),
                list: Box::new(list),
            },
            span: span(),
        };
        let s = selectivity(&expr, &b);
        // 2 * 0.1 = 0.2 capped.
        assert!((s - 0.2).abs() < 1e-12);
    }

    #[test]
    fn literal_true_is_one() {
        let b = BindingStats::empty();
        let expr = Expression {
            kind: ExpressionKind::Literal(Literal::Boolean(true)),
            span: span(),
        };
        assert!((selectivity(&expr, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn literal_false_is_zero() {
        let b = BindingStats::empty();
        let expr = Expression {
            kind: ExpressionKind::Literal(Literal::Boolean(false)),
            span: span(),
        };
        assert!(selectivity(&expr, &b) < 1e-12);
    }

    #[test]
    fn unknown_expression_falls_back_to_half() {
        let b = BindingStats::empty();
        // A bare property access (treated as truthiness) is unknown.
        let expr = prop_access("a", "age");
        let s = selectivity(&expr, &b);
        assert!((s - FALLBACK_UNKNOWN).abs() < 1e-12);
    }

    #[test]
    fn string_test_uses_fixed_fallback() {
        let b = BindingStats::empty();
        let expr = Expression {
            kind: ExpressionKind::StringTest {
                op: StringOp::StartsWith,
                target: Box::new(prop_access("a", "name")),
                pattern: Box::new(lit_str("Al")),
            },
            span: span(),
        };
        let s = selectivity(&expr, &b);
        assert!((s - FALLBACK_STRING_TEST).abs() < 1e-12);
    }

    #[test]
    fn unknown_alias_falls_back_to_eq_default() {
        let b = BindingStats::empty();
        let expr = binary(BinaryOp::Eq, prop_access("zzz", "age"), lit_int(30));
        let s = selectivity(&expr, &b);
        assert!((s - FALLBACK_EQ).abs() < 1e-12);
    }

    #[test]
    fn label_eq_is_treated_as_one() {
        let b = BindingStats::empty();
        let expr = Expression {
            kind: ExpressionKind::FunctionCall {
                name: QualifiedName::single(Identifier::new("__label_eq", span())),
                args: vec![var("a"), lit_str("Person")],
                distinct: false,
            },
            span: span(),
        };
        let s = selectivity(&expr, &b);
        assert!((s - 1.0).abs() < 1e-12);
    }
}
