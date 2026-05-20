//! Storage-level predicate pushdown (RFC-013 §5).
//!
//! Converts WHERE conjuncts that compare a single property of one
//! NodeScan alias against a literal to `ScanPredicate` values the
//! storage layer can use for row-group pruning. Non-pushable conjuncts
//! survive as a `Filter` above the NodeScan.
//!
//! This module is called from `optimize::pushdown::pushdown_at` at the
//! NodeScan leaf — at that point, `pending` holds every conjunct the
//! caller wanted to materialise as a Filter above the scan.
//!
//! Conservatism: when in doubt, leave the conjunct as `Expression`
//! (the executor's `Filter` is correct for any expression). The
//! storage layer always falls back to a full row-group scan on
//! ambiguous predicates, so a missed pushdown is correctness-safe.

use namidb_storage::sst::predicates::ScanPredicate;
use namidb_storage::sst::stats::StatScalar;

use crate::parser::ast::{BinaryOp, Expression, ExpressionKind, Literal};

/// Split `pending` into (a) conjuncts that translate to `ScanPredicate`
/// against `alias` and (b) the residual that must remain as a Filter.
///
/// The order of the residual preserves the input order so EXPLAIN
/// output stays predictable.
pub fn classify_pending_for_scan(
    pending: Vec<Expression>,
    alias: &str,
) -> (Vec<ScanPredicate>, Vec<Expression>) {
    let mut pushable = Vec::new();
    let mut residual = Vec::new();
    for expr in pending {
        match try_into_scan_predicate(&expr, alias) {
            Some(p) => pushable.push(p),
            None => residual.push(expr),
        }
    }
    (pushable, residual)
}

/// Try to convert a single conjunct into a `ScanPredicate` against
/// `alias`. Returns `None` for any expression the storage layer cannot
/// evaluate against per-row-group statistics:
///
/// - Property reference is to a different alias.
/// - One side is not a literal (parameter, function call, arithmetic).
/// - Comparison is `!=` (no single-sided row-group verdict).
/// - Literal is NULL (use IS [NOT] NULL instead).
pub fn try_into_scan_predicate(expr: &Expression, alias: &str) -> Option<ScanPredicate> {
    match &expr.kind {
        ExpressionKind::Binary { op, left, right } => {
            try_binary_to_predicate(*op, left, right, alias)
        }
        ExpressionKind::Between { target, low, high } => {
            let column = property_column_for_alias(target, alias)?;
            let lo = literal_to_stat_scalar(literal_of(low)?)?;
            let hi = literal_to_stat_scalar(literal_of(high)?)?;
            Some(ScanPredicate::Between {
                column,
                low: lo,
                high: hi,
            })
        }
        ExpressionKind::IsNull {
            expr: inner,
            negated,
        } => {
            let column = property_column_for_alias(inner, alias)?;
            Some(if *negated {
                ScanPredicate::IsNotNull { column }
            } else {
                ScanPredicate::IsNull { column }
            })
        }
        ExpressionKind::In { item, list } => {
            let column = property_column_for_alias(item, alias)?;
            let values = literal_list(list)?;
            Some(ScanPredicate::In { column, values })
        }
        _ => None,
    }
}

fn try_binary_to_predicate(
    op: BinaryOp,
    left: &Expression,
    right: &Expression,
    alias: &str,
) -> Option<ScanPredicate> {
    // Try (Property OP Literal). Then try mirror (Literal OP Property).
    if let (Some(column), Some(lit)) = (property_column_for_alias(left, alias), literal_of(right)) {
        let value = literal_to_stat_scalar(lit)?;
        return predicate_for_binary(op, column, value, /*literal_on_right=*/ true);
    }
    if let (Some(column), Some(lit)) = (property_column_for_alias(right, alias), literal_of(left)) {
        let value = literal_to_stat_scalar(lit)?;
        return predicate_for_binary(op, column, value, /*literal_on_right=*/ false);
    }
    None
}

/// Build a predicate for the symmetric `Property OP Literal` form,
/// flipping the operator if the literal is on the left.
fn predicate_for_binary(
    op: BinaryOp,
    column: String,
    value: StatScalar,
    literal_on_right: bool,
) -> Option<ScanPredicate> {
    let canonical = if literal_on_right {
        op
    } else {
        // `Literal OP Property` ≡ `Property OP' Literal` with OP' = flip(OP).
        flip_binary(op)?
    };
    match canonical {
        BinaryOp::Eq => Some(ScanPredicate::Eq { column, value }),
        BinaryOp::Lt => Some(ScanPredicate::Lt { column, value }),
        BinaryOp::Le => Some(ScanPredicate::LtEq { column, value }),
        BinaryOp::Gt => Some(ScanPredicate::Gt { column, value }),
        BinaryOp::Ge => Some(ScanPredicate::GtEq { column, value }),
        _ => None,
    }
}

fn flip_binary(op: BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Eq => BinaryOp::Eq,
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::Le => BinaryOp::Ge,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::Ge => BinaryOp::Le,
        _ => return None,
    })
}

/// If `expr` is `<alias>.<key>`, return the declared `key` string.
/// Otherwise return `None`.
fn property_column_for_alias(expr: &Expression, alias: &str) -> Option<String> {
    let prop = match &expr.kind {
        ExpressionKind::Property(p) => p,
        _ => return None,
    };
    match &prop.target.kind {
        ExpressionKind::Variable(id) if id.name == alias => Some(prop.key.name.clone()),
        _ => None,
    }
}

/// Pluck the `Literal` out of `expr` if `expr` is exactly a literal
/// node (no arithmetic, no function, no parameter).
fn literal_of(expr: &Expression) -> Option<&Literal> {
    match &expr.kind {
        ExpressionKind::Literal(lit) => Some(lit),
        _ => None,
    }
}

/// Pluck a list of literals out of `[1, 2, 3]`-style expressions for
/// `IN` pushdown. Returns `None` if any element is non-literal or NULL
/// (NULL collapses the IN to inconclusive — defer to the residual
/// Filter for 3VL).
fn literal_list(expr: &Expression) -> Option<Vec<StatScalar>> {
    let items = match &expr.kind {
        ExpressionKind::List(items) => items,
        _ => return None,
    };
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let lit = literal_of(it)?;
        out.push(literal_to_stat_scalar(lit)?);
    }
    Some(out)
}

/// Translate a parser `Literal` into the storage-side `StatScalar`.
/// Returns `None` for NULL (use IsNull instead).
fn literal_to_stat_scalar(lit: &Literal) -> Option<StatScalar> {
    match lit {
        Literal::Integer(n) => Some(StatScalar::Int64(*n)),
        Literal::Float(f) => Some(StatScalar::Float64(*f)),
        Literal::String(s) => Some(StatScalar::Utf8(s.clone())),
        Literal::Boolean(b) => Some(StatScalar::Bool(*b)),
        Literal::Null => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ast::{Identifier, PropertyAccess};
    use crate::parser::error::SourceSpan;

    fn span() -> SourceSpan {
        SourceSpan::point(0)
    }

    fn ident(name: &str) -> Identifier {
        Identifier {
            name: name.into(),
            span: span(),
            quoted: false,
        }
    }

    fn variable(name: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Variable(ident(name)),
            span: span(),
        }
    }

    fn property(alias: &str, key: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Property(Box::new(PropertyAccess {
                target: variable(alias),
                key: ident(key),
                span: span(),
            })),
            span: span(),
        }
    }

    fn int_lit(n: i64) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::Integer(n)),
            span: span(),
        }
    }

    fn str_lit(s: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::String(s.into())),
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

    #[test]
    fn eq_property_on_left_translates() {
        let e = binary(BinaryOp::Eq, property("a", "age"), int_lit(30));
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::Eq {
                column: "age".into(),
                value: StatScalar::Int64(30),
            })
        );
    }

    #[test]
    fn eq_property_on_right_translates() {
        let e = binary(BinaryOp::Eq, int_lit(30), property("a", "age"));
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::Eq {
                column: "age".into(),
                value: StatScalar::Int64(30),
            })
        );
    }

    #[test]
    fn lt_property_on_left_translates() {
        let e = binary(BinaryOp::Lt, property("a", "age"), int_lit(30));
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::Lt {
                column: "age".into(),
                value: StatScalar::Int64(30),
            })
        );
    }

    #[test]
    fn lt_with_literal_on_left_flips_to_gt() {
        // `30 < a.age` ≡ `a.age > 30`.
        let e = binary(BinaryOp::Lt, int_lit(30), property("a", "age"));
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::Gt {
                column: "age".into(),
                value: StatScalar::Int64(30),
            })
        );
    }

    #[test]
    fn le_translates_to_lteq() {
        let e = binary(BinaryOp::Le, property("a", "age"), int_lit(30));
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::LtEq {
                column: "age".into(),
                value: StatScalar::Int64(30),
            })
        );
    }

    #[test]
    fn ge_translates_to_gteq() {
        let e = binary(BinaryOp::Ge, property("a", "age"), int_lit(30));
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::GtEq {
                column: "age".into(),
                value: StatScalar::Int64(30),
            })
        );
    }

    #[test]
    fn ne_is_not_pushable() {
        let e = binary(BinaryOp::Ne, property("a", "age"), int_lit(30));
        assert!(try_into_scan_predicate(&e, "a").is_none());
    }

    #[test]
    fn property_on_different_alias_rejects() {
        let e = binary(BinaryOp::Eq, property("b", "age"), int_lit(30));
        assert!(try_into_scan_predicate(&e, "a").is_none());
    }

    #[test]
    fn non_literal_right_rejects() {
        let e = binary(BinaryOp::Eq, property("a", "age"), variable("b"));
        assert!(try_into_scan_predicate(&e, "a").is_none());
    }

    #[test]
    fn null_literal_rejects() {
        let e = binary(
            BinaryOp::Eq,
            property("a", "age"),
            Expression {
                kind: ExpressionKind::Literal(Literal::Null),
                span: span(),
            },
        );
        assert!(try_into_scan_predicate(&e, "a").is_none());
    }

    #[test]
    fn string_literal_translates() {
        let e = binary(BinaryOp::Eq, property("a", "name"), str_lit("Alice"));
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::Eq {
                column: "name".into(),
                value: StatScalar::Utf8("Alice".into()),
            })
        );
    }

    #[test]
    fn between_translates() {
        let e = Expression {
            kind: ExpressionKind::Between {
                target: Box::new(property("a", "age")),
                low: Box::new(int_lit(20)),
                high: Box::new(int_lit(40)),
            },
            span: span(),
        };
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::Between {
                column: "age".into(),
                low: StatScalar::Int64(20),
                high: StatScalar::Int64(40),
            })
        );
    }

    #[test]
    fn is_null_translates() {
        let e = Expression {
            kind: ExpressionKind::IsNull {
                expr: Box::new(property("a", "age")),
                negated: false,
            },
            span: span(),
        };
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::IsNull {
                column: "age".into(),
            })
        );
    }

    #[test]
    fn is_not_null_translates() {
        let e = Expression {
            kind: ExpressionKind::IsNull {
                expr: Box::new(property("a", "age")),
                negated: true,
            },
            span: span(),
        };
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::IsNotNull {
                column: "age".into(),
            })
        );
    }

    #[test]
    fn in_with_literals_translates() {
        let e = Expression {
            kind: ExpressionKind::In {
                item: Box::new(property("a", "age")),
                list: Box::new(Expression {
                    kind: ExpressionKind::List(vec![int_lit(10), int_lit(20), int_lit(30)]),
                    span: span(),
                }),
            },
            span: span(),
        };
        assert_eq!(
            try_into_scan_predicate(&e, "a"),
            Some(ScanPredicate::In {
                column: "age".into(),
                values: vec![
                    StatScalar::Int64(10),
                    StatScalar::Int64(20),
                    StatScalar::Int64(30),
                ],
            })
        );
    }

    #[test]
    fn in_with_non_literal_rejects() {
        let e = Expression {
            kind: ExpressionKind::In {
                item: Box::new(property("a", "age")),
                list: Box::new(Expression {
                    kind: ExpressionKind::List(vec![int_lit(10), variable("b")]),
                    span: span(),
                }),
            },
            span: span(),
        };
        assert!(try_into_scan_predicate(&e, "a").is_none());
    }

    #[test]
    fn classify_partitions_correctly() {
        let pushable_eq = binary(BinaryOp::Eq, property("a", "age"), int_lit(30));
        let cross_alias = binary(BinaryOp::Eq, property("b", "age"), int_lit(30));
        let non_literal = binary(BinaryOp::Eq, property("a", "age"), variable("x"));

        let (pushed, residual) = classify_pending_for_scan(
            vec![
                pushable_eq.clone(),
                cross_alias.clone(),
                non_literal.clone(),
            ],
            "a",
        );
        assert_eq!(pushed.len(), 1);
        assert_eq!(residual.len(), 2);
    }

    #[test]
    fn classify_preserves_residual_order() {
        let r1 = binary(BinaryOp::Eq, property("b", "x"), int_lit(1));
        let p = binary(BinaryOp::Eq, property("a", "age"), int_lit(30));
        let r2 = binary(BinaryOp::Eq, property("c", "y"), int_lit(2));

        let (pushed, residual) =
            classify_pending_for_scan(vec![r1.clone(), p.clone(), r2.clone()], "a");
        assert_eq!(pushed.len(), 1);
        // Residual conjuncts preserve input order so EXPLAIN stays
        // predictable.
        assert_eq!(residual.len(), 2);
        assert_eq!(residual[0], r1);
        assert_eq!(residual[1], r2);
    }

    #[test]
    fn parameter_is_not_pushable() {
        let e = binary(
            BinaryOp::Eq,
            property("a", "age"),
            Expression {
                kind: ExpressionKind::Parameter("minAge".into()),
                span: span(),
            },
        );
        assert!(try_into_scan_predicate(&e, "a").is_none());
    }
}
