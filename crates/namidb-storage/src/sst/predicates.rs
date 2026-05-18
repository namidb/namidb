//! Scan-level predicate IR + row-group verdict evaluator (RFC-013).
//!
//! `ScanPredicate` is a single-column conjunctive predicate the SST
//! reader can evaluate against per-row-group statistics
//! (`PropertyColumnStats`) to skip entire row-groups without decoding
//! them. Conservatism is mandatory: only return `Absent` when the
//! stats *prove* no row in the row-group can satisfy the predicate.
//! Missing min/max, type mismatch, or null-only columns all yield
//! `MaybePresent` so the row-group decodes and the executor's `Filter`
//! catches the per-row case via 3VL.
//!
//! Cross-type comparisons (e.g. `Int32` predicate vs `Float64` stats)
//! return `MaybePresent` defensively — the optimizer never generates
//! cross-type predicates because the schema declares property types,
//! but a faulty caller cannot drop legitimate rows.

use std::cmp::Ordering;

use namidb_core::Value;

use super::stats::{PropertyColumnStats, StatScalar};

/// A single-column predicate pushable to the SST reader. Each variant
/// references a property column by its declared name (not by Parquet
/// leaf path); the reader resolves the leaf index at scan time.
#[derive(Clone, Debug, PartialEq)]
pub enum ScanPredicate {
 /// `column == value`.
 Eq { column: String, value: StatScalar },
 /// `column < value`.
 Lt { column: String, value: StatScalar },
 /// `column <= value`.
 LtEq { column: String, value: StatScalar },
 /// `column > value`.
 Gt { column: String, value: StatScalar },
 /// `column >= value`.
 GtEq { column: String, value: StatScalar },
 /// `low <= column <= high` (inclusive both sides).
 Between {
 column: String,
 low: StatScalar,
 high: StatScalar,
 },
 /// `column IS NULL`.
 IsNull { column: String },
 /// `column IS NOT NULL`.
 IsNotNull { column: String },
 /// `column IN (v1, v2, ...)`. v0 supports literal-only lists.
 In {
 column: String,
 values: Vec<StatScalar>,
 },
}

impl ScanPredicate {
 /// The declared property column name this predicate filters on.
 pub fn column(&self) -> &str {
 match self {
 ScanPredicate::Eq { column, .. }
 | ScanPredicate::Lt { column, .. }
 | ScanPredicate::LtEq { column, .. }
 | ScanPredicate::Gt { column, .. }
 | ScanPredicate::GtEq { column, .. }
 | ScanPredicate::Between { column, .. }
 | ScanPredicate::IsNull { column }
 | ScanPredicate::IsNotNull { column }
 | ScanPredicate::In { column, .. } => column,
 }
 }
}

/// Outcome of evaluating a `ScanPredicate` against the stats of a
/// single row-group. Conservatism: `Absent` only when the stats
/// **prove** no row matches; otherwise `MaybePresent` (decode the
/// row-group and let the executor's Filter handle row-level NULL
/// semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowGroupVerdict {
 /// Stats prove no row in this row-group can satisfy the predicate.
 /// The reader skips the row-group entirely.
 Absent,
 /// Stats are inconclusive or overlap the predicate. The reader
 /// decodes the row-group and downstream operators handle per-row
 /// evaluation.
 MaybePresent,
}

/// Evaluate `predicate` against per-row-group statistics `stats`.
///
/// Returns `Absent` only when the stats *prove* no row matches.
/// `MaybePresent` covers: missing min/max, type mismatch, NULL-only
/// column with an ordered predicate, or genuine overlap. The reader
/// must decode any row-group with `MaybePresent` and rely on the
/// executor's `Filter` to drop non-matching rows.
///
/// Precondition: `predicate.column() == stats.name`. Violating it is
/// programmer error; we still return `MaybePresent` to fail safe.
pub fn eval_row_group(predicate: &ScanPredicate, stats: &PropertyColumnStats) -> RowGroupVerdict {
 if predicate.column() != stats.name {
 return RowGroupVerdict::MaybePresent;
 }

 use RowGroupVerdict::*;
 use ScanPredicate as P;

 match predicate {
 P::Eq { value, .. } => match (&stats.min, &stats.max) {
 (Some(lo), Some(hi)) => match (scalar_cmp(lo, value), scalar_cmp(value, hi)) {
 (
 Some(Ordering::Less | Ordering::Equal),
 Some(Ordering::Less | Ordering::Equal),
 ) => MaybePresent,
 (Some(_), Some(_)) => Absent,
 _ => MaybePresent,
 },
 _ => MaybePresent,
 },

 P::Lt { value, .. } => match &stats.min {
 // min < value ⇒ at least one row may be < value.
 Some(lo) => match scalar_cmp(lo, value) {
 Some(Ordering::Less) => MaybePresent,
 Some(_) => Absent,
 None => MaybePresent,
 },
 None => MaybePresent,
 },

 P::LtEq { value, .. } => match &stats.min {
 Some(lo) => match scalar_cmp(lo, value) {
 Some(Ordering::Less | Ordering::Equal) => MaybePresent,
 Some(_) => Absent,
 None => MaybePresent,
 },
 None => MaybePresent,
 },

 P::Gt { value, .. } => match &stats.max {
 Some(hi) => match scalar_cmp(hi, value) {
 Some(Ordering::Greater) => MaybePresent,
 Some(_) => Absent,
 None => MaybePresent,
 },
 None => MaybePresent,
 },

 P::GtEq { value, .. } => match &stats.max {
 Some(hi) => match scalar_cmp(hi, value) {
 Some(Ordering::Greater | Ordering::Equal) => MaybePresent,
 Some(_) => Absent,
 None => MaybePresent,
 },
 None => MaybePresent,
 },

 P::Between {
 column: _,
 low,
 high,
 } => {
 let lower = eval_row_group(
 &P::GtEq {
 column: predicate.column().to_string(),
 value: low.clone(),
 },
 stats,
 );
 if lower == Absent {
 return Absent;
 }
 let upper = eval_row_group(
 &P::LtEq {
 column: predicate.column().to_string(),
 value: high.clone(),
 },
 stats,
 );
 if upper == Absent {
 return Absent;
 }
 MaybePresent
 }

 P::IsNull { .. } => {
 if stats.null_count > 0 {
 MaybePresent
 } else {
 Absent
 }
 }

 P::IsNotNull { .. } => {
 // Conservative: we don't know the row-group's row_count
 // here, so we cannot prove every row is NULL. Always
 // decode; the Filter above drops nulls at row level.
 MaybePresent
 }

 P::In { values, .. } => {
 // Build the closed interval [min(values), max(values)] and
 // reuse the Between path. False-positives (a value
 // inside [low, high] but not in the list) are caught by
 // the residual Filter the optimizer leaves intact when
 // `In` is partial.
 if values.is_empty() {
 return Absent;
 }
 let (lo, hi) = match in_value_range(values) {
 Some(r) => r,
 None => return MaybePresent,
 };
 let lower = eval_row_group(
 &P::GtEq {
 column: predicate.column().to_string(),
 value: lo,
 },
 stats,
 );
 if lower == Absent {
 return Absent;
 }
 let upper = eval_row_group(
 &P::LtEq {
 column: predicate.column().to_string(),
 value: hi,
 },
 stats,
 );
 if upper == Absent {
 return Absent;
 }
 MaybePresent
 }
 }
}

/// Compare two `StatScalar` values of the same variant. Returns `None`
/// when the variants don't match (cross-type comparison — defensive
/// MaybePresent at the caller). NaN floats compare as `Equal` for
/// determinism (the writer canonicalises NaN before hashing; pruning
/// is conservative regardless).
fn scalar_cmp(a: &StatScalar, b: &StatScalar) -> Option<Ordering> {
 use StatScalar::*;
 match (a, b) {
 (Bool(x), Bool(y)) => Some(x.cmp(y)),
 (Int32(x), Int32(y)) => Some(x.cmp(y)),
 (Int64(x), Int64(y)) => Some(x.cmp(y)),
 (Float32(x), Float32(y)) => Some(float_cmp_f32(*x, *y)),
 (Float64(x), Float64(y)) => Some(float_cmp_f64(*x, *y)),
 (Utf8(x), Utf8(y)) => Some(x.cmp(y)),
 (LargeUtf8(x), LargeUtf8(y)) => Some(x.cmp(y)),
 (Binary(x), Binary(y)) => Some(x.cmp(y)),
 (Date32(x), Date32(y)) => Some(x.cmp(y)),
 (TimestampMicrosUtc(x), TimestampMicrosUtc(y)) => Some(x.cmp(y)),
 _ => None,
 }
}

fn float_cmp_f32(a: f32, b: f32) -> Ordering {
 a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn float_cmp_f64(a: f64, b: f64) -> Ordering {
 a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// Evaluate `predicate` against a single property value loaded into
/// memory (memtable row, point lookup result, etc.). Uses 3VL: `None`
/// or `Value::Null` plus an ordered predicate returns `false`; `IsNull`
/// returns `true` on missing or null, `IsNotNull` returns `false`.
///
/// Cross-type comparison (e.g. predicate `Utf8` vs value `I64`) returns
/// `false` — same conservative behaviour as the cost-model selectivity.
/// This is safe for `Snapshot::scan_label_with_predicates` because the
/// row-group pruning is independent; the per-value path only narrows
/// what survives.
pub fn eval_against_value(predicate: &ScanPredicate, value: Option<&Value>) -> bool {
 use ScanPredicate as P;

 // IS NULL / IS NOT NULL — handle absence first.
 if let P::IsNull { .. } = predicate {
 return value.map(|v| v.is_null()).unwrap_or(true);
 }
 if let P::IsNotNull { .. } = predicate {
 return value.map(|v| !v.is_null()).unwrap_or(false);
 }

 // Ordered predicates: missing or null evaluates to `false` (3VL).
 let Some(v) = value else { return false };
 if v.is_null() {
 return false;
 }

 match predicate {
 P::Eq { value: target, .. } => value_cmp(v, target)
 .map(|o| o == Ordering::Equal)
 .unwrap_or(false),
 P::Lt { value: target, .. } => value_cmp(v, target)
 .map(|o| o == Ordering::Less)
 .unwrap_or(false),
 P::LtEq { value: target, .. } => value_cmp(v, target)
 .map(|o| matches!(o, Ordering::Less | Ordering::Equal))
 .unwrap_or(false),
 P::Gt { value: target, .. } => value_cmp(v, target)
 .map(|o| o == Ordering::Greater)
 .unwrap_or(false),
 P::GtEq { value: target, .. } => value_cmp(v, target)
 .map(|o| matches!(o, Ordering::Greater | Ordering::Equal))
 .unwrap_or(false),
 P::Between { low, high, .. } => {
 let lo_ok = value_cmp(v, low)
 .map(|o| matches!(o, Ordering::Greater | Ordering::Equal))
 .unwrap_or(false);
 let hi_ok = value_cmp(v, high)
 .map(|o| matches!(o, Ordering::Less | Ordering::Equal))
 .unwrap_or(false);
 lo_ok && hi_ok
 }
 P::In { values, .. } => values.iter().any(|target| {
 value_cmp(v, target)
 .map(|o| o == Ordering::Equal)
 .unwrap_or(false)
 }),
 P::IsNull { .. } | P::IsNotNull { .. } => unreachable!("handled above"),
 }
}

/// Compare a memtable `Value` against a predicate-side `StatScalar`.
/// Returns `None` for cross-type comparisons; defensive behaviour at
/// the caller.
fn value_cmp(v: &Value, s: &StatScalar) -> Option<Ordering> {
 use StatScalar as S;
 use Value as V;
 match (v, s) {
 (V::Bool(a), S::Bool(b)) => Some(a.cmp(b)),
 // Value::I64 is the canonical integer slot. Compare against
 // Int32/Date32 by widening to i64.
 (V::I64(a), S::Int32(b)) => Some(a.cmp(&(*b as i64))),
 (V::I64(a), S::Int64(b)) => Some(a.cmp(b)),
 (V::I64(a), S::Date32(b)) => Some(a.cmp(&(*b as i64))),
 (V::I64(a), S::TimestampMicrosUtc(b)) => Some(a.cmp(b)),
 // Value::F64 ↔ Float32/Float64. NaN compares as Equal for
 // determinism (writer canonicalises NaN during HLL hashing).
 (V::F64(a), S::Float32(b)) => Some(float_cmp_f64(*a, *b as f64)),
 (V::F64(a), S::Float64(b)) => Some(float_cmp_f64(*a, *b)),
 (V::Str(a), S::Utf8(b)) => Some(a.as_str().cmp(b.as_str())),
 (V::Str(a), S::LargeUtf8(b)) => Some(a.as_str().cmp(b.as_str())),
 (V::Bytes(a), S::Binary(b)) => Some(a.as_slice().cmp(b.as_slice())),
 _ => None,
 }
}

/// Return `(min, max)` over a non-empty `Vec<StatScalar>` assuming all
/// elements share the same variant. Returns `None` when variants
/// disagree (cross-type IN list — defensive MaybePresent at the caller).
fn in_value_range(values: &[StatScalar]) -> Option<(StatScalar, StatScalar)> {
 let mut min = values[0].clone();
 let mut max = values[0].clone();
 for v in &values[1..] {
 match scalar_cmp(v, &min) {
 Some(Ordering::Less) => min = v.clone(),
 Some(_) => {}
 None => return None,
 }
 match scalar_cmp(v, &max) {
 Some(Ordering::Greater) => max = v.clone(),
 Some(_) => {}
 None => return None,
 }
 }
 Some((min, max))
}

#[cfg(test)]
mod tests {
 use super::*;

 fn stats(
 name: &str,
 min: Option<StatScalar>,
 max: Option<StatScalar>,
 nulls: u64,
 ) -> PropertyColumnStats {
 PropertyColumnStats {
 name: name.to_string(),
 null_count: nulls,
 min,
 max,
 ndv_estimate: None,
 }
 }

 #[test]
 fn eq_in_range_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(25),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn eq_below_min_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(5),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn eq_above_max_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(99),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn eq_at_boundary_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p_min = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(10),
 };
 let p_max = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(50),
 };
 assert_eq!(eval_row_group(&p_min, &s), RowGroupVerdict::MaybePresent);
 assert_eq!(eval_row_group(&p_max, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn lt_above_min_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Lt {
 column: "age".into(),
 value: StatScalar::Int64(30),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn lt_at_or_below_min_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p_eq = ScanPredicate::Lt {
 column: "age".into(),
 value: StatScalar::Int64(10),
 };
 let p_lt = ScanPredicate::Lt {
 column: "age".into(),
 value: StatScalar::Int64(5),
 };
 // Lt(10): min=10 means every row is ≥10; none is <10. Absent.
 assert_eq!(eval_row_group(&p_eq, &s), RowGroupVerdict::Absent);
 assert_eq!(eval_row_group(&p_lt, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn lteq_at_min_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::LtEq {
 column: "age".into(),
 value: StatScalar::Int64(10),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn gt_below_max_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Gt {
 column: "age".into(),
 value: StatScalar::Int64(30),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn gt_at_or_above_max_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p_eq = ScanPredicate::Gt {
 column: "age".into(),
 value: StatScalar::Int64(50),
 };
 let p_gt = ScanPredicate::Gt {
 column: "age".into(),
 value: StatScalar::Int64(99),
 };
 assert_eq!(eval_row_group(&p_eq, &s), RowGroupVerdict::Absent);
 assert_eq!(eval_row_group(&p_gt, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn gteq_at_max_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::GtEq {
 column: "age".into(),
 value: StatScalar::Int64(50),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn between_overlap_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Between {
 column: "age".into(),
 low: StatScalar::Int64(20),
 high: StatScalar::Int64(40),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn between_disjoint_above_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Between {
 column: "age".into(),
 low: StatScalar::Int64(60),
 high: StatScalar::Int64(80),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn between_disjoint_below_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Between {
 column: "age".into(),
 low: StatScalar::Int64(1),
 high: StatScalar::Int64(5),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn is_null_with_nulls_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 3,
 );
 let p = ScanPredicate::IsNull {
 column: "age".into(),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn is_null_without_nulls_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::IsNull {
 column: "age".into(),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn is_not_null_is_always_maybe_present() {
 // v0 conservative: even null_count==0 returns MaybePresent;
 // executor's Filter handles per-row NULL drops.
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::IsNotNull {
 column: "age".into(),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn in_overlap_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::In {
 column: "age".into(),
 values: vec![StatScalar::Int64(15), StatScalar::Int64(45)],
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn in_disjoint_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::In {
 column: "age".into(),
 values: vec![StatScalar::Int64(80), StatScalar::Int64(90)],
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn in_empty_is_absent() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::In {
 column: "age".into(),
 values: vec![],
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn missing_min_max_is_maybe_present() {
 let s = stats("age", None, None, 0);
 for p in [
 ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(5),
 },
 ScanPredicate::Lt {
 column: "age".into(),
 value: StatScalar::Int64(5),
 },
 ScanPredicate::Gt {
 column: "age".into(),
 value: StatScalar::Int64(5),
 },
 ScanPredicate::Between {
 column: "age".into(),
 low: StatScalar::Int64(1),
 high: StatScalar::Int64(2),
 },
 ] {
 assert_eq!(
 eval_row_group(&p, &s),
 RowGroupVerdict::MaybePresent,
 "{p:?}"
 );
 }
 }

 #[test]
 fn cross_type_is_maybe_present() {
 // stats are Int64 but predicate carries Utf8 — defensive fallthrough.
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Utf8("hello".into()),
 };
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn column_mismatch_is_maybe_present() {
 let s = stats(
 "age",
 Some(StatScalar::Int64(10)),
 Some(StatScalar::Int64(50)),
 0,
 );
 let p = ScanPredicate::Eq {
 column: "name".into(),
 value: StatScalar::Int64(5),
 };
 // column() != stats.name ⇒ defensive MaybePresent.
 assert_eq!(eval_row_group(&p, &s), RowGroupVerdict::MaybePresent);
 }

 #[test]
 fn utf8_range_pruning_works() {
 let s = stats(
 "name",
 Some(StatScalar::Utf8("Alice".into())),
 Some(StatScalar::Utf8("David".into())),
 0,
 );
 let p_in = ScanPredicate::Eq {
 column: "name".into(),
 value: StatScalar::Utf8("Bob".into()),
 };
 let p_out = ScanPredicate::Eq {
 column: "name".into(),
 value: StatScalar::Utf8("Zoe".into()),
 };
 assert_eq!(eval_row_group(&p_in, &s), RowGroupVerdict::MaybePresent);
 assert_eq!(eval_row_group(&p_out, &s), RowGroupVerdict::Absent);
 }

 #[test]
 fn eval_against_value_eq_matches() {
 let p = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(30),
 };
 assert!(eval_against_value(&p, Some(&Value::I64(30))));
 assert!(!eval_against_value(&p, Some(&Value::I64(31))));
 assert!(!eval_against_value(&p, Some(&Value::Null)));
 assert!(!eval_against_value(&p, None));
 }

 #[test]
 fn eval_against_value_lt_works() {
 let p = ScanPredicate::Lt {
 column: "age".into(),
 value: StatScalar::Int64(30),
 };
 assert!(eval_against_value(&p, Some(&Value::I64(20))));
 assert!(!eval_against_value(&p, Some(&Value::I64(30))));
 assert!(!eval_against_value(&p, Some(&Value::I64(40))));
 }

 #[test]
 fn eval_against_value_between_works() {
 let p = ScanPredicate::Between {
 column: "age".into(),
 low: StatScalar::Int64(20),
 high: StatScalar::Int64(40),
 };
 assert!(eval_against_value(&p, Some(&Value::I64(20))));
 assert!(eval_against_value(&p, Some(&Value::I64(30))));
 assert!(eval_against_value(&p, Some(&Value::I64(40))));
 assert!(!eval_against_value(&p, Some(&Value::I64(50))));
 assert!(!eval_against_value(&p, Some(&Value::I64(10))));
 }

 #[test]
 fn eval_against_value_is_null_matches_missing_and_null() {
 let p = ScanPredicate::IsNull {
 column: "age".into(),
 };
 assert!(eval_against_value(&p, None));
 assert!(eval_against_value(&p, Some(&Value::Null)));
 assert!(!eval_against_value(&p, Some(&Value::I64(0))));
 }

 #[test]
 fn eval_against_value_is_not_null_drops_missing_and_null() {
 let p = ScanPredicate::IsNotNull {
 column: "age".into(),
 };
 assert!(!eval_against_value(&p, None));
 assert!(!eval_against_value(&p, Some(&Value::Null)));
 assert!(eval_against_value(&p, Some(&Value::I64(0))));
 }

 #[test]
 fn eval_against_value_in_matches_any_listed() {
 let p = ScanPredicate::In {
 column: "name".into(),
 values: vec![
 StatScalar::Utf8("Alice".into()),
 StatScalar::Utf8("Bob".into()),
 ],
 };
 assert!(eval_against_value(&p, Some(&Value::Str("Alice".into()))));
 assert!(eval_against_value(&p, Some(&Value::Str("Bob".into()))));
 assert!(!eval_against_value(&p, Some(&Value::Str("Carol".into()))));
 assert!(!eval_against_value(&p, None));
 }

 #[test]
 fn eval_against_value_cross_type_returns_false() {
 let p = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Utf8("30".into()),
 };
 // Value::I64 vs StatScalar::Utf8 → cross-type → false (3VL drop).
 assert!(!eval_against_value(&p, Some(&Value::I64(30))));
 }

 #[test]
 fn predicate_column_accessor_is_consistent() {
 let p1 = ScanPredicate::Eq {
 column: "age".into(),
 value: StatScalar::Int64(5),
 };
 let p2 = ScanPredicate::IsNull {
 column: "name".into(),
 };
 let p3 = ScanPredicate::In {
 column: "age".into(),
 values: vec![StatScalar::Int64(1)],
 };
 assert_eq!(p1.column(), "age");
 assert_eq!(p2.column(), "name");
 assert_eq!(p3.column(), "age");
 }
}
