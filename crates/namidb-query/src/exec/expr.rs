//! Cypher expression evaluator.
//!
//! Implements three-valued logic per RFC-008 §"Semántica NULL".

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use regex::Regex;

use super::row::Row;
use super::value::RuntimeValue;
use crate::parser::{BinaryOp, Expression, ExpressionKind, Literal, SourceSpan, StringOp, UnaryOp};

/// Classification of an [`EvalError`]. `Unsupported` distinguishes a feature
/// the engine deliberately does not implement (e.g. an unknown function) from
/// a genuine internal bug, so transports can surface a typed "not supported"
/// error instead of a bare 500.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalErrorKind {
    /// Anything not specifically classified — treated as an internal error.
    Generic,
    /// A feature the engine does not support (unknown function, unsupported
    /// expression form). Maps to a "not supported" error on every transport.
    Unsupported,
}

/// Runtime error returned by expression / executor code. Carries a span
/// for diagnostics and a [`EvalErrorKind`] for typed classification.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalError {
    pub message: String,
    pub span: SourceSpan,
    pub kind: EvalErrorKind,
}

impl EvalError {
    pub fn new(msg: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            message: msg.into(),
            span,
            kind: EvalErrorKind::Generic,
        }
    }

    /// Construct an `Unsupported` error (unknown function, unimplemented form).
    pub fn unsupported(msg: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            message: msg.into(),
            span,
            kind: EvalErrorKind::Unsupported,
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EvalError: {} at {}", self.message, self.span)
    }
}

impl std::error::Error for EvalError {}

/// Parameters bound at execution time (e.g. `$personId`).
pub type Params = BTreeMap<String, RuntimeValue>;

/// Evaluate `expr` against the current `row` and `params`. Returns
/// `RuntimeValue::Null` for any operation where a NULL propagates per
/// three-valued logic.
pub fn evaluate(expr: &Expression, row: &Row, params: &Params) -> Result<RuntimeValue, EvalError> {
    let span = expr.span;
    match &expr.kind {
        ExpressionKind::Literal(l) => Ok(literal_to_runtime(l)),
        ExpressionKind::Star => Err(EvalError::new("`*` is only valid inside `count(*)`", span)),
        ExpressionKind::Variable(id) => row
            .get(&id.name)
            .cloned()
            .ok_or_else(|| EvalError::new(format!("binding `{}` not bound", id.name), id.span)),
        ExpressionKind::Parameter(name) => params
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::new(format!("parameter `${}` not provided", name), span)),
        ExpressionKind::Property(p) => {
            let target = evaluate(&p.target, row, params)?;
            Ok(property_access(&target, &p.key.name))
        }
        ExpressionKind::Index { target, index } => {
            let t = evaluate(target, row, params)?;
            let i = evaluate(index, row, params)?;
            Ok(index_into(&t, &i))
        }
        ExpressionKind::Range { target, from, to } => {
            let t = evaluate(target, row, params)?;
            let from_v = match from {
                Some(e) => Some(evaluate(e, row, params)?),
                None => None,
            };
            let to_v = match to {
                Some(e) => Some(evaluate(e, row, params)?),
                None => None,
            };
            Ok(range_into(&t, from_v.as_ref(), to_v.as_ref()))
        }
        ExpressionKind::Unary { op, expr: inner } => {
            let v = evaluate(inner, row, params)?;
            Ok(eval_unary(*op, &v))
        }
        ExpressionKind::Binary { op, left, right } => {
            let l = evaluate(left, row, params)?;
            let r = evaluate(right, row, params)?;
            eval_binary(*op, &l, &r, span)
        }
        ExpressionKind::In { item, list } => {
            let item_v = evaluate(item, row, params)?;
            let list_v = evaluate(list, row, params)?;
            Ok(eval_in(&item_v, &list_v))
        }
        ExpressionKind::StringTest {
            op,
            target,
            pattern,
        } => {
            let t = evaluate(target, row, params)?;
            let p = evaluate(pattern, row, params)?;
            Ok(eval_string_test(*op, &t, &p))
        }
        ExpressionKind::IsNull {
            expr: inner,
            negated,
        } => {
            let v = evaluate(inner, row, params)?;
            let is_null = v.is_null();
            Ok(RuntimeValue::Bool(if *negated {
                !is_null
            } else {
                is_null
            }))
        }
        ExpressionKind::FunctionCall {
            name,
            args,
            distinct: _,
        } => {
            let name_str = name.joined().to_ascii_lowercase();
            let evaluated_args: Vec<RuntimeValue> = args
                .iter()
                .map(|a| evaluate(a, row, params))
                .collect::<Result<_, _>>()?;
            call_scalar_function(&name_str, &evaluated_args, span)
        }
        ExpressionKind::Case {
            scrutinee,
            branches,
            otherwise,
        } => {
            let scrut_v = match scrutinee {
                Some(s) => Some(evaluate(s, row, params)?),
                None => None,
            };
            for b in branches {
                let when_v = evaluate(&b.when, row, params)?;
                let matched = if let Some(s) = &scrut_v {
                    is_equal(s, &when_v)
                } else {
                    when_v.as_bool().unwrap_or(false)
                };
                if matched {
                    return evaluate(&b.then, row, params);
                }
            }
            match otherwise {
                Some(e) => evaluate(e, row, params),
                None => Ok(RuntimeValue::Null),
            }
        }
        ExpressionKind::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(evaluate(it, row, params)?);
            }
            Ok(RuntimeValue::List(out))
        }
        ExpressionKind::Map(m) => {
            let mut out = BTreeMap::new();
            for (k, v) in &m.entries {
                out.insert(k.name.clone(), evaluate(v, row, params)?);
            }
            Ok(RuntimeValue::Map(out))
        }
        ExpressionKind::ListComprehension(lc) => eval_list_comprehension(lc, row, params),
        ExpressionKind::PatternComprehension(_) => Err(EvalError::new(
            "pattern comprehension evaluation requires storage access — \
 must be hoisted to a PatternList operator before evaluate()",
            span,
        )),
        ExpressionKind::Exists(_) => Err(EvalError::new(
            "EXISTS pattern predicates require storage access — \
 must be hoisted to a SemiApply operator before evaluate()",
            span,
        )),
    }
}

fn literal_to_runtime(l: &Literal) -> RuntimeValue {
    match l {
        Literal::Integer(n) => RuntimeValue::Integer(*n),
        Literal::Float(f) => RuntimeValue::Float(*f),
        Literal::String(s) => RuntimeValue::String(s.clone()),
        Literal::Boolean(b) => RuntimeValue::Bool(*b),
        Literal::Null => RuntimeValue::Null,
    }
}

fn property_access(target: &RuntimeValue, key: &str) -> RuntimeValue {
    match target {
        RuntimeValue::Null => RuntimeValue::Null,
        RuntimeValue::Node(n) => match key {
            // `_id` materialises the internal NodeId. Plain `id` falls
            // through to the property map so users can store their own.
            "_id" => RuntimeValue::String(n.id.to_string()),
            _ => n.properties.get(key).cloned().unwrap_or(RuntimeValue::Null),
        },
        RuntimeValue::Rel(r) => match key {
            // Rel internal identifier — see Node accessor above for the
            // rationale behind the `_id` sigil.
            "_id" => RuntimeValue::String(format!("{}:{}", r.src, r.dst)),
            _ => r.properties.get(key).cloned().unwrap_or(RuntimeValue::Null),
        },
        RuntimeValue::Map(m) => m.get(key).cloned().unwrap_or(RuntimeValue::Null),
        _ => RuntimeValue::Null,
    }
}

fn index_into(target: &RuntimeValue, index: &RuntimeValue) -> RuntimeValue {
    match (target, index) {
        (RuntimeValue::Null, _) | (_, RuntimeValue::Null) => RuntimeValue::Null,
        (RuntimeValue::List(items), RuntimeValue::Integer(i)) => {
            let len = items.len() as i64;
            let resolved = if *i < 0 { len + i } else { *i };
            if resolved < 0 || resolved >= len {
                RuntimeValue::Null
            } else {
                items[resolved as usize].clone()
            }
        }
        (RuntimeValue::Map(m), RuntimeValue::String(k)) => {
            m.get(k).cloned().unwrap_or(RuntimeValue::Null)
        }
        _ => RuntimeValue::Null,
    }
}

fn range_into(
    target: &RuntimeValue,
    from: Option<&RuntimeValue>,
    to: Option<&RuntimeValue>,
) -> RuntimeValue {
    match target {
        RuntimeValue::List(items) => {
            let len = items.len() as i64;
            let resolve = |v: Option<&RuntimeValue>, default: i64| -> i64 {
                match v {
                    Some(RuntimeValue::Integer(n)) => {
                        if *n < 0 {
                            (len + n).max(0)
                        } else {
                            (*n).min(len)
                        }
                    }
                    _ => default,
                }
            };
            let lo = resolve(from, 0);
            let hi = resolve(to, len);
            if lo >= hi {
                RuntimeValue::List(Vec::new())
            } else {
                RuntimeValue::List(items[lo as usize..hi as usize].to_vec())
            }
        }
        _ => RuntimeValue::Null,
    }
}

fn eval_unary(op: UnaryOp, v: &RuntimeValue) -> RuntimeValue {
    if v.is_null() {
        return RuntimeValue::Null;
    }
    match op {
        UnaryOp::Neg => match v {
            RuntimeValue::Integer(n) => RuntimeValue::Integer(-*n),
            RuntimeValue::Float(f) => RuntimeValue::Float(-*f),
            _ => RuntimeValue::Null,
        },
        UnaryOp::Not => match v.as_bool() {
            Some(b) => RuntimeValue::Bool(!b),
            None => RuntimeValue::Null,
        },
    }
}

fn eval_binary(
    op: BinaryOp,
    a: &RuntimeValue,
    b: &RuntimeValue,
    span: SourceSpan,
) -> Result<RuntimeValue, EvalError> {
    match op {
        // 3VL logical: special NULL handling.
        BinaryOp::And => match (a.as_bool(), b.as_bool()) {
            (Some(false), _) | (_, Some(false)) => Ok(RuntimeValue::Bool(false)),
            (Some(true), Some(true)) => Ok(RuntimeValue::Bool(true)),
            _ => Ok(RuntimeValue::Null),
        },
        BinaryOp::Or => match (a.as_bool(), b.as_bool()) {
            (Some(true), _) | (_, Some(true)) => Ok(RuntimeValue::Bool(true)),
            (Some(false), Some(false)) => Ok(RuntimeValue::Bool(false)),
            _ => Ok(RuntimeValue::Null),
        },
        BinaryOp::Xor => match (a.as_bool(), b.as_bool()) {
            (Some(x), Some(y)) => Ok(RuntimeValue::Bool(x ^ y)),
            _ => Ok(RuntimeValue::Null),
        },
        _ => {
            if a.is_null() || b.is_null() {
                return Ok(RuntimeValue::Null);
            }
            match op {
                BinaryOp::Add => arith(a, b, span, "+", |x, y| x + y, |x, y| x + y, true),
                BinaryOp::Sub => arith(a, b, span, "-", |x, y| x - y, |x, y| x - y, false),
                BinaryOp::Mul => arith(a, b, span, "*", |x, y| x * y, |x, y| x * y, false),
                BinaryOp::Div => arith_div(a, b, span),
                BinaryOp::Mod => {
                    arith(a, b, span, "%", |x, y| x.rem_euclid(y), |x, y| x % y, false)
                }
                BinaryOp::Pow => arith_pow(a, b, span),
                BinaryOp::Eq => Ok(RuntimeValue::Bool(is_equal(a, b))),
                BinaryOp::Ne => Ok(RuntimeValue::Bool(!is_equal(a, b))),
                BinaryOp::Lt => order_cmp(a, b, |o| o == Ordering::Less),
                BinaryOp::Gt => order_cmp(a, b, |o| o == Ordering::Greater),
                BinaryOp::Le => order_cmp(a, b, |o| o != Ordering::Greater),
                BinaryOp::Ge => order_cmp(a, b, |o| o != Ordering::Less),
                BinaryOp::RegexMatch => regex_match(a, b, span),
                BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => unreachable!(),
            }
        }
    }
}

fn arith(
    a: &RuntimeValue,
    b: &RuntimeValue,
    span: SourceSpan,
    op_label: &'static str,
    f_int: impl Fn(i64, i64) -> i64,
    f_float: impl Fn(f64, f64) -> f64,
    string_concat: bool,
) -> Result<RuntimeValue, EvalError> {
    match (a, b) {
        (RuntimeValue::Integer(x), RuntimeValue::Integer(y)) => {
            Ok(RuntimeValue::Integer(f_int(*x, *y)))
        }
        (RuntimeValue::Float(x), RuntimeValue::Float(y)) => {
            Ok(RuntimeValue::Float(f_float(*x, *y)))
        }
        (RuntimeValue::Integer(x), RuntimeValue::Float(y)) => {
            Ok(RuntimeValue::Float(f_float(*x as f64, *y)))
        }
        (RuntimeValue::Float(x), RuntimeValue::Integer(y)) => {
            Ok(RuntimeValue::Float(f_float(*x, *y as f64)))
        }
        (RuntimeValue::String(x), RuntimeValue::String(y)) if string_concat => {
            Ok(RuntimeValue::String(format!("{}{}", x, y)))
        }
        (RuntimeValue::String(x), other) if string_concat => Ok(RuntimeValue::String(format!(
            "{}{}",
            x,
            runtime_to_string_concat(other)
        ))),
        (other, RuntimeValue::String(y)) if string_concat => Ok(RuntimeValue::String(format!(
            "{}{}",
            runtime_to_string_concat(other),
            y
        ))),
        (RuntimeValue::List(xs), RuntimeValue::List(ys)) if string_concat => {
            let mut out = xs.clone();
            out.extend_from_slice(ys);
            Ok(RuntimeValue::List(out))
        }
        _ => Err(EvalError::new(
            format!(
                "cannot apply `{}` between {} and {}",
                op_label,
                a.type_name(),
                b.type_name()
            ),
            span,
        )),
    }
}

fn arith_div(
    a: &RuntimeValue,
    b: &RuntimeValue,
    span: SourceSpan,
) -> Result<RuntimeValue, EvalError> {
    match (a, b) {
        (_, RuntimeValue::Integer(0)) => Err(EvalError::new("integer division by zero", span)),
        (RuntimeValue::Integer(x), RuntimeValue::Integer(y)) => Ok(RuntimeValue::Integer(x / y)),
        (RuntimeValue::Float(x), RuntimeValue::Float(y)) => Ok(RuntimeValue::Float(x / y)),
        (RuntimeValue::Integer(x), RuntimeValue::Float(y)) => {
            Ok(RuntimeValue::Float(*x as f64 / *y))
        }
        (RuntimeValue::Float(x), RuntimeValue::Integer(y)) => {
            Ok(RuntimeValue::Float(*x / *y as f64))
        }
        _ => Err(EvalError::new(
            format!(
                "cannot apply `/` between {} and {}",
                a.type_name(),
                b.type_name()
            ),
            span,
        )),
    }
}

fn arith_pow(
    a: &RuntimeValue,
    b: &RuntimeValue,
    span: SourceSpan,
) -> Result<RuntimeValue, EvalError> {
    let to_f64 = |v: &RuntimeValue| -> Option<f64> {
        match v {
            RuntimeValue::Integer(n) => Some(*n as f64),
            RuntimeValue::Float(f) => Some(*f),
            _ => None,
        }
    };
    match (to_f64(a), to_f64(b)) {
        (Some(x), Some(y)) => Ok(RuntimeValue::Float(x.powf(y))),
        _ => Err(EvalError::new(
            format!(
                "cannot apply `^` between {} and {}",
                a.type_name(),
                b.type_name()
            ),
            span,
        )),
    }
}

fn runtime_to_string_concat(v: &RuntimeValue) -> String {
    match v {
        RuntimeValue::Null => "".to_string(),
        RuntimeValue::Integer(n) => n.to_string(),
        RuntimeValue::Float(f) => f.to_string(),
        RuntimeValue::String(s) => s.clone(),
        RuntimeValue::Bool(b) => b.to_string(),
        _ => format!("{:?}", v),
    }
}

fn is_equal(a: &RuntimeValue, b: &RuntimeValue) -> bool {
    match (a, b) {
        (RuntimeValue::Null, _) | (_, RuntimeValue::Null) => false,
        (RuntimeValue::Integer(x), RuntimeValue::Integer(y)) => x == y,
        (RuntimeValue::Float(x), RuntimeValue::Float(y)) => x == y,
        (RuntimeValue::Integer(x), RuntimeValue::Float(y))
        | (RuntimeValue::Float(y), RuntimeValue::Integer(x)) => (*x as f64) == *y,
        (RuntimeValue::Bool(x), RuntimeValue::Bool(y)) => x == y,
        (RuntimeValue::String(x), RuntimeValue::String(y)) => x == y,
        (RuntimeValue::List(x), RuntimeValue::List(y)) => x == y,
        (RuntimeValue::Map(x), RuntimeValue::Map(y)) => x == y,
        (RuntimeValue::Node(x), RuntimeValue::Node(y)) => x.id == y.id,
        (RuntimeValue::Rel(x), RuntimeValue::Rel(y)) => {
            x.edge_type == y.edge_type && x.src == y.src && x.dst == y.dst
        }
        (RuntimeValue::Date(x), RuntimeValue::Date(y)) => x == y,
        (RuntimeValue::DateTime(x), RuntimeValue::DateTime(y)) => x == y,
        _ => false,
    }
}

fn order_cmp(
    a: &RuntimeValue,
    b: &RuntimeValue,
    predicate: impl Fn(Ordering) -> bool,
) -> Result<RuntimeValue, EvalError> {
    match compare(a, b) {
        Some(o) => Ok(RuntimeValue::Bool(predicate(o))),
        None => Ok(RuntimeValue::Null),
    }
}

fn compare(a: &RuntimeValue, b: &RuntimeValue) -> Option<Ordering> {
    match (a, b) {
        (RuntimeValue::Null, _) | (_, RuntimeValue::Null) => None,
        (RuntimeValue::Integer(x), RuntimeValue::Integer(y)) => Some(x.cmp(y)),
        (RuntimeValue::Float(x), RuntimeValue::Float(y)) => x.partial_cmp(y),
        (RuntimeValue::Integer(x), RuntimeValue::Float(y)) => (*x as f64).partial_cmp(y),
        (RuntimeValue::Float(x), RuntimeValue::Integer(y)) => x.partial_cmp(&(*y as f64)),
        (RuntimeValue::String(x), RuntimeValue::String(y)) => Some(x.cmp(y)),
        (RuntimeValue::Bool(x), RuntimeValue::Bool(y)) => Some(x.cmp(y)),
        (RuntimeValue::Date(x), RuntimeValue::Date(y)) => Some(x.cmp(y)),
        (RuntimeValue::DateTime(x), RuntimeValue::DateTime(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Public comparator used by Sort/TopN. Returns `Ordering::Equal` for
/// uncomparable values to preserve input order (stable sort).
pub fn order_for_sort(a: &RuntimeValue, b: &RuntimeValue, direction_desc: bool) -> Ordering {
    // NULL sorts last by default (Cypher semantics).
    match (a.is_null(), b.is_null()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            let o = compare(a, b).unwrap_or(Ordering::Equal);
            if direction_desc {
                o.reverse()
            } else {
                o
            }
        }
    }
}

fn eval_in(item: &RuntimeValue, list: &RuntimeValue) -> RuntimeValue {
    if item.is_null() || list.is_null() {
        return RuntimeValue::Null;
    }
    match list {
        RuntimeValue::List(items) => {
            let mut has_null = false;
            for v in items {
                if v.is_null() {
                    has_null = true;
                    continue;
                }
                if is_equal(item, v) {
                    return RuntimeValue::Bool(true);
                }
            }
            if has_null {
                RuntimeValue::Null
            } else {
                RuntimeValue::Bool(false)
            }
        }
        _ => RuntimeValue::Null,
    }
}

fn eval_string_test(op: StringOp, a: &RuntimeValue, b: &RuntimeValue) -> RuntimeValue {
    match (a, b) {
        (RuntimeValue::Null, _) | (_, RuntimeValue::Null) => RuntimeValue::Null,
        (RuntimeValue::String(s), RuntimeValue::String(p)) => RuntimeValue::Bool(match op {
            StringOp::StartsWith => s.starts_with(p.as_str()),
            StringOp::EndsWith => s.ends_with(p.as_str()),
            StringOp::Contains => s.contains(p.as_str()),
        }),
        _ => RuntimeValue::Null,
    }
}

/// Cypher `=~`: whole-string regular-expression match, following Neo4j
/// semantics where the pattern must match the *entire* string (like Java's
/// `String.matches`), not a substring. The pattern is anchored with
/// `^(?:…)$` so a top-level alternation (`a|b`) still binds under both
/// anchors; redundant user anchors are harmless. Returns Null when either
/// operand is not a string, and raises an error for an invalid pattern
/// rather than silently matching nothing.
fn regex_match(
    a: &RuntimeValue,
    b: &RuntimeValue,
    span: SourceSpan,
) -> Result<RuntimeValue, EvalError> {
    let (subject, pattern) = match (a, b) {
        (RuntimeValue::String(s), RuntimeValue::String(p)) => (s, p),
        _ => return Ok(RuntimeValue::Null),
    };
    let re = Regex::new(&format!("^(?:{})$", pattern)).map_err(|e| {
        EvalError::new(
            format!("invalid regular expression `{}`: {}", pattern, e),
            span,
        )
    })?;
    Ok(RuntimeValue::Bool(re.is_match(subject)))
}

fn call_scalar_function(
    name: &str,
    args: &[RuntimeValue],
    span: SourceSpan,
) -> Result<RuntimeValue, EvalError> {
    match name {
        // --- Identity / graph projection
        "id" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Node(n) => RuntimeValue::String(n.id.to_string()),
            RuntimeValue::Rel(r) => RuntimeValue::String(format!("{}:{}", r.src, r.dst)),
            _ => RuntimeValue::Null,
        }),
        // `elementId()` returns the exact id string the Bolt layer emits
        // as `element_id`, so a GUI's `WHERE elementId(x) = $id` (G.V()'s
        // node/edge fetch and expand) round-trips. Nodes: the UUID;
        // relationships: the synthetic `<type>-<src>-><dst>` form. These
        // mirror `uuid_to_bolt_ids` / `synthetic_edge_id` in namidb-bolt.
        "elementid" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Node(n) => RuntimeValue::String(n.id.to_string()),
            RuntimeValue::Rel(r) => {
                RuntimeValue::String(format!("{}-{}->{}", r.edge_type, r.src.0, r.dst.0))
            }
            _ => RuntimeValue::Null,
        }),
        "labels" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Node(n) => RuntimeValue::List(
                n.labels
                    .iter()
                    .map(|l| RuntimeValue::String(l.clone()))
                    .collect(),
            ),
            _ => RuntimeValue::Null,
        }),
        "type" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Rel(r) => RuntimeValue::String(r.edge_type.clone()),
            _ => RuntimeValue::Null,
        }),
        "keys" => single_arg(name, args, span).map(|v| {
            let keys: Vec<String> = match v {
                RuntimeValue::Node(n) => n.properties.keys().cloned().collect(),
                RuntimeValue::Rel(r) => r.properties.keys().cloned().collect(),
                RuntimeValue::Map(m) => m.keys().cloned().collect(),
                _ => return RuntimeValue::Null,
            };
            RuntimeValue::List(
                keys.into_iter()
                    .map(RuntimeValue::String)
                    .collect::<Vec<_>>(),
            )
        }),
        "properties" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Node(n) => RuntimeValue::Map(n.properties.clone()),
            RuntimeValue::Rel(r) => RuntimeValue::Map(r.properties.clone()),
            RuntimeValue::Map(m) => RuntimeValue::Map(m.clone()),
            _ => RuntimeValue::Null,
        }),

        // --- Path element functions (RFC-004). A path is the alternating
        // sequence `[Node, Rel, Node, Rel, ..., Node]`; `nodes` returns the node
        // elements (even positions), `relationships` the rel elements (odd
        // positions). NULL padding from shortestPath is dropped so the lists
        // stay well-typed for downstream `[x IN nodes(p) WHERE ...]` filtering.
        "nodes" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Path(items) => RuntimeValue::List(
                items
                    .into_iter()
                    .step_by(2)
                    .filter(|x| matches!(x, RuntimeValue::Node(_)))
                    .collect(),
            ),
            _ => RuntimeValue::Null,
        }),
        "relationships" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Path(items) => RuntimeValue::List(
                items
                    .into_iter()
                    .skip(1)
                    .step_by(2)
                    .filter(|x| matches!(x, RuntimeValue::Rel(_)))
                    .collect(),
            ),
            _ => RuntimeValue::Null,
        }),

        // --- Collection ops
        "size" | "length" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::String(s) => RuntimeValue::Integer(s.chars().count() as i64),
            RuntimeValue::List(items) => RuntimeValue::Integer(items.len() as i64),
            RuntimeValue::Map(m) => RuntimeValue::Integer(m.len() as i64),
            // A vector's size is its dimension, so `size(n.embedding)` answers
            // "how many dimensions" without a dedicated builtin.
            RuntimeValue::Vector(v) => RuntimeValue::Integer(v.len() as i64),
            RuntimeValue::Vector8 { codes, .. } => RuntimeValue::Integer(codes.len() as i64),
            // Path is `[Node, Rel, Node, Rel, ..., Node]` so the
            // relationship count is `(len - 1) / 2`. The shortestPath
            // lower fills missing rel/target bindings with NULL,
            // which still preserves the position-based count.
            RuntimeValue::Path(items) if !items.is_empty() => {
                RuntimeValue::Integer((items.len() as i64 - 1) / 2)
            }
            _ => RuntimeValue::Null,
        }),
        "head" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::List(items) => items.first().cloned().unwrap_or(RuntimeValue::Null),
            _ => RuntimeValue::Null,
        }),
        "last" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::List(items) => items.last().cloned().unwrap_or(RuntimeValue::Null),
            _ => RuntimeValue::Null,
        }),
        "tail" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::List(items) if !items.is_empty() => {
                RuntimeValue::List(items[1..].to_vec())
            }
            RuntimeValue::List(_) => RuntimeValue::List(Vec::new()),
            _ => RuntimeValue::Null,
        }),
        "coalesce" => {
            for a in args {
                if !a.is_null() {
                    return Ok(a.clone());
                }
            }
            Ok(RuntimeValue::Null)
        }
        "range" => range_fn(args, span),

        // --- String functions
        "tolower" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::String(s) => RuntimeValue::String(s.to_lowercase()),
            _ => RuntimeValue::Null,
        }),
        "toupper" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::String(s) => RuntimeValue::String(s.to_uppercase()),
            _ => RuntimeValue::Null,
        }),
        "trim" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::String(s) => RuntimeValue::String(s.trim().to_string()),
            _ => RuntimeValue::Null,
        }),
        "ltrim" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::String(s) => RuntimeValue::String(s.trim_start().to_string()),
            _ => RuntimeValue::Null,
        }),
        "rtrim" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::String(s) => RuntimeValue::String(s.trim_end().to_string()),
            _ => RuntimeValue::Null,
        }),
        // `reverse` flips a string (by character) or a list.
        "reverse" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::String(s) => RuntimeValue::String(s.chars().rev().collect()),
            RuntimeValue::List(mut items) => {
                items.reverse();
                RuntimeValue::List(items)
            }
            _ => RuntimeValue::Null,
        }),
        "left" => str_left_right(args, span, true),
        "right" => str_left_right(args, span, false),
        "substring" => str_substring(args, span),
        "replace" => str_replace(args, span),
        "split" => str_split(args, span),
        "tostring" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Null => RuntimeValue::Null,
            other => RuntimeValue::String(runtime_to_string_concat(&other)),
        }),
        "tointeger" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Integer(n) => RuntimeValue::Integer(n),
            RuntimeValue::Float(f) => RuntimeValue::Integer(f as i64),
            RuntimeValue::String(s) => match s.trim().parse::<i64>() {
                Ok(n) => RuntimeValue::Integer(n),
                Err(_) => RuntimeValue::Null,
            },
            RuntimeValue::Bool(b) => RuntimeValue::Integer(if b { 1 } else { 0 }),
            _ => RuntimeValue::Null,
        }),
        "tofloat" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Float(f) => RuntimeValue::Float(f),
            RuntimeValue::Integer(n) => RuntimeValue::Float(n as f64),
            RuntimeValue::String(s) => match s.trim().parse::<f64>() {
                Ok(f) => RuntimeValue::Float(f),
                Err(_) => RuntimeValue::Null,
            },
            _ => RuntimeValue::Null,
        }),
        // `toBoolean`: parse "true"/"false" (any case), pass a Bool through,
        // and map 0/1; anything else is Null (Cypher `toBooleanOrNull`).
        "toboolean" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Bool(b) => RuntimeValue::Bool(b),
            RuntimeValue::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "true" => RuntimeValue::Bool(true),
                "false" => RuntimeValue::Bool(false),
                _ => RuntimeValue::Null,
            },
            RuntimeValue::Integer(0) => RuntimeValue::Bool(false),
            RuntimeValue::Integer(1) => RuntimeValue::Bool(true),
            _ => RuntimeValue::Null,
        }),

        // --- Numeric
        "abs" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Integer(n) => RuntimeValue::Integer(n.abs()),
            RuntimeValue::Float(f) => RuntimeValue::Float(f.abs()),
            _ => RuntimeValue::Null,
        }),
        // `round`/`floor`/`ceil`/`sqrt` always return a Float, matching
        // Neo4j; an integer argument is promoted first.
        "round" => single_arg(name, args, span).map(|v| num_to_float(&v, f64::round)),
        "floor" => single_arg(name, args, span).map(|v| num_to_float(&v, f64::floor)),
        "ceil" => single_arg(name, args, span).map(|v| num_to_float(&v, f64::ceil)),
        "sqrt" => single_arg(name, args, span).map(|v| num_to_float(&v, f64::sqrt)),
        // `sign` returns -1, 0 or 1 as an Integer.
        "sign" => single_arg(name, args, span).map(|v| match v {
            RuntimeValue::Integer(n) => RuntimeValue::Integer(n.signum()),
            RuntimeValue::Float(f) => RuntimeValue::Integer(if f > 0.0 {
                1
            } else if f < 0.0 {
                -1
            } else {
                0
            }),
            _ => RuntimeValue::Null,
        }),

        // --- Vector constructor
        // `vector([x, y, …])` lifts a numeric list into a first-class
        // `Vector(Vec<f32>)`, which round-trips through `runtime_to_core`
        // → `CoreValue::Vec` → Parquet column. Without this builtin, a
        // bare `[0.1, 0.2]` literal stays a `List` and the writer rejects
        // it with "only scalars are storable in v0".
        "vector" => {
            let v = single_arg(name, args, span)?;
            match v {
                RuntimeValue::Null => Ok(RuntimeValue::Null),
                RuntimeValue::Vector(items) => Ok(RuntimeValue::Vector(items)),
                RuntimeValue::List(items) => {
                    let mut out = Vec::with_capacity(items.len());
                    for (idx, item) in items.into_iter().enumerate() {
                        let coerced = match item {
                            RuntimeValue::Float(f) => f as f32,
                            RuntimeValue::Integer(n) => n as f32,
                            other => {
                                return Err(EvalError::new(
                                    format!(
                                        "vector() requires numeric elements; got {} at index {}",
                                        other.type_name(),
                                        idx
                                    ),
                                    span,
                                ));
                            }
                        };
                        out.push(coerced);
                    }
                    Ok(RuntimeValue::Vector(out))
                }
                other => Err(EvalError::new(
                    format!(
                        "vector() requires a list of numbers; got {}",
                        other.type_name()
                    ),
                    span,
                )),
            }
        }

        // --- Vector similarity / distance
        // These power KNN over stored embeddings without a dedicated operator:
        // a query like
        //   MATCH (n:Note) WHERE n.embedding IS NOT NULL
        //   RETURN n ORDER BY cosine_similarity(n.embedding, $q) DESC LIMIT 10
        // expresses semantic search through the existing scan + sort + limit
        // path, with a WHERE on labels/properties acting as a pre-filter on the
        // candidate set. Each takes two operands that are a stored `Vector` or a
        // numeric `List` (so a `$param` array works without an explicit
        // `vector()` wrap). NULL in either operand propagates to NULL; a
        // dimension mismatch is a usage error.
        "cosine_similarity" | "dot_product" | "euclidean_distance" => {
            match vector_pair(name, args, span)? {
                None => Ok(RuntimeValue::Null),
                Some((a, b)) => Ok(match name {
                    "dot_product" => RuntimeValue::Float(vec_dot_f64(&a, &b)),
                    "euclidean_distance" => {
                        let sum: f64 = a
                            .iter()
                            .zip(&b)
                            .map(|(x, y)| {
                                let d = *x as f64 - *y as f64;
                                d * d
                            })
                            .sum();
                        RuntimeValue::Float(sum.sqrt())
                    }
                    // cosine_similarity: undefined (NULL) when either vector has
                    // zero magnitude, since the denominator would be zero.
                    _ => {
                        let na = vec_dot_f64(&a, &a).sqrt();
                        let nb = vec_dot_f64(&b, &b).sqrt();
                        if na == 0.0 || nb == 0.0 {
                            RuntimeValue::Null
                        } else {
                            RuntimeValue::Float(vec_dot_f64(&a, &b) / (na * nb))
                        }
                    }
                }),
            }
        }

        // BM25 lexical relevance (hybrid search, Item 13 Layer B):
        // `bm25(document, query) -> Float`. NULL if either arg is NULL
        // (three-valued logic). See `exec::text_scoring`.
        "bm25" => match args {
            [RuntimeValue::Null, _] | [_, RuntimeValue::Null] => Ok(RuntimeValue::Null),
            [RuntimeValue::String(doc), RuntimeValue::String(query)] => Ok(RuntimeValue::Float(
                crate::exec::text_scoring::bm25_score(doc, query),
            )),
            _ => Err(EvalError::new(
                "bm25(document, query) expects two strings",
                span,
            )),
        },

        // --- Lowering helpers (internal)
        "__path" => Ok(RuntimeValue::Path(args.to_vec())),
        "__label_eq" => match args {
            // Label-membership test: `MATCH (n:A:B)` lowers to one
            // `__label_eq(n, "A")` per label, ANDed together. A node carrying
            // the label set {A, B, ...} passes iff it contains the asked label.
            [target, RuntimeValue::String(label)] => Ok(match target {
                RuntimeValue::Node(n) => RuntimeValue::Bool(n.labels.contains(label)),
                RuntimeValue::Null => RuntimeValue::Null,
                _ => RuntimeValue::Bool(false),
            }),
            _ => Err(EvalError::new("__label_eq expects (node, string)", span)),
        },

        // `timestamp()` — milliseconds since the Unix epoch. A common
        // Cypher helper for stamping writes (`SET n.updated = timestamp()`).
        // Evaluated per call from the wall clock.
        "timestamp" => {
            if !args.is_empty() {
                return Err(EvalError::new("timestamp() takes no arguments", span));
            }
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            Ok(RuntimeValue::Integer(ms))
        }

        // `datetime()` — Cypher UTC datetime, microseconds since the epoch
        // (RuntimeValue::DateTime). Zero args returns the current instant
        // (like Neo4j/Memgraph `datetime()`); a single ISO-8601 / RFC3339
        // string parses to the same representation; NULL propagates.
        "datetime" | "localdatetime" => match args {
            [] => Ok(RuntimeValue::DateTime(now_micros())),
            [RuntimeValue::Null] => Ok(RuntimeValue::Null),
            [RuntimeValue::DateTime(m)] => Ok(RuntimeValue::DateTime(*m)),
            [RuntimeValue::String(s)] => parse_iso_datetime(s, span),
            _ => Err(EvalError::new(
                format!("{name}() takes no arguments or a single ISO-8601 string"),
                span,
            )),
        },

        // `date()` — Cypher calendar date, days since the epoch
        // (RuntimeValue::Date). Zero args returns today (UTC); a single
        // `YYYY-MM-DD` string parses to the same; NULL propagates.
        "date" => match args {
            [] => Ok(RuntimeValue::Date((now_micros() / 86_400_000_000) as i32)),
            [RuntimeValue::Null] => Ok(RuntimeValue::Null),
            [RuntimeValue::Date(d)] => Ok(RuntimeValue::Date(*d)),
            [RuntimeValue::String(s)] => parse_iso_date(s, span),
            _ => Err(EvalError::new(
                "date() takes no arguments or a single YYYY-MM-DD string",
                span,
            )),
        },

        _ => Err(EvalError::unsupported(
            format!("function `{}` is not supported in v0", name),
            span,
        )),
    }
}

/// Microseconds since the Unix epoch, read from the wall clock. Shared by the
/// zero-argument `datetime()` / `date()` constructors.
fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Parse an ISO-8601 / RFC3339 datetime into microseconds since the epoch.
/// Accepts an offset-qualified RFC3339 string, a bare `YYYY-MM-DDTHH:MM:SS`
/// (interpreted as UTC), and a bare `YYYY-MM-DD` (promoted to midnight UTC).
fn parse_iso_datetime(s: &str, span: SourceSpan) -> Result<RuntimeValue, EvalError> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(RuntimeValue::DateTime(
            dt.with_timezone(&chrono::Utc).timestamp_micros(),
        ));
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(RuntimeValue::DateTime(ndt.and_utc().timestamp_micros()));
        }
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let ndt = d.and_hms_opt(0, 0, 0).expect("midnight is always valid");
        return Ok(RuntimeValue::DateTime(ndt.and_utc().timestamp_micros()));
    }
    Err(EvalError::new(
        format!("datetime(): could not parse `{s}` as an ISO-8601 datetime"),
        span,
    ))
}

/// Parse a `YYYY-MM-DD` string into days since the epoch.
fn parse_iso_date(s: &str, span: SourceSpan) -> Result<RuntimeValue, EvalError> {
    match chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        Ok(d) => {
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
            Ok(RuntimeValue::Date(
                d.signed_duration_since(epoch).num_days() as i32,
            ))
        }
        Err(_) => Err(EvalError::new(
            format!("date(): could not parse `{s}` as YYYY-MM-DD"),
            span,
        )),
    }
}

fn single_arg(
    name: &str,
    args: &[RuntimeValue],
    span: SourceSpan,
) -> Result<RuntimeValue, EvalError> {
    match args {
        [single] => Ok(single.clone()),
        _ => Err(EvalError::new(
            format!("`{}` expects exactly 1 argument", name),
            span,
        )),
    }
}

/// Coerce a builtin argument that must be an integer. `Null` yields
/// `Ok(None)` so the caller can propagate Null; a non-integer is an error.
fn want_int(v: &RuntimeValue, fname: &str, span: SourceSpan) -> Result<Option<i64>, EvalError> {
    match v {
        RuntimeValue::Null => Ok(None),
        RuntimeValue::Integer(n) => Ok(Some(*n)),
        other => Err(EvalError::new(
            format!("{} expects an integer, got {}", fname, other.type_name()),
            span,
        )),
    }
}

/// Apply a float transform, promoting an Integer first; non-numbers -> Null.
fn num_to_float(v: &RuntimeValue, f: impl Fn(f64) -> f64) -> RuntimeValue {
    match v {
        RuntimeValue::Integer(n) => RuntimeValue::Float(f(*n as f64)),
        RuntimeValue::Float(x) => RuntimeValue::Float(f(*x)),
        _ => RuntimeValue::Null,
    }
}

/// Sum of products in f64. Embeddings are stored as f32; accumulating in f64
/// keeps the dot product (and the norms derived from it) numerically stable
/// over high-dimensional vectors.
fn vec_dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// Score a candidate vector against a query for the `VectorSearch` flat
/// fallback (RFC-030). Mirrors the `cosine_similarity` / `dot_product` /
/// `euclidean_distance` builtins but returns the value the score column should
/// hold **plus** `true` when higher-is-better (cosine/dot similarity) so the
/// caller can rank correctly. `None` ⇒ drop the candidate (NULL operand, or a
/// zero-magnitude vector making cosine undefined). A dimension mismatch is a
/// usage error.
pub(crate) fn vector_score(
    distance: crate::plan::logical::VectorDistance,
    a: &RuntimeValue,
    b: &RuntimeValue,
    span: SourceSpan,
) -> Result<Option<(f64, bool)>, EvalError> {
    use crate::plan::logical::VectorDistance;
    let Some(av) = coerce_vector(a, "vector", span)? else {
        return Ok(None);
    };
    let Some(bv) = coerce_vector(b, "vector", span)? else {
        return Ok(None);
    };
    if av.len() != bv.len() {
        return Err(EvalError::new(
            format!("vector dimension mismatch ({} vs {})", av.len(), bv.len()),
            span,
        ));
    }
    Ok(Some(match distance {
        VectorDistance::Dot => (vec_dot_f64(&av, &bv), true),
        VectorDistance::Euclidean => {
            let sum: f64 = av
                .iter()
                .zip(&bv)
                .map(|(x, y)| {
                    let d = *x as f64 - *y as f64;
                    d * d
                })
                .sum();
            (sum.sqrt(), false)
        }
        VectorDistance::Cosine => {
            let na = vec_dot_f64(&av, &av).sqrt();
            let nb = vec_dot_f64(&bv, &bv).sqrt();
            if na == 0.0 || nb == 0.0 {
                return Ok(None);
            }
            (vec_dot_f64(&av, &bv) / (na * nb), true)
        }
    }))
}

/// Coerce a similarity/distance operand into a vector. Accepts a stored
/// `Vector` or a numeric `List` (so a `$param` array works without an explicit
/// `vector()` wrap). `Null` yields `Ok(None)` so the caller can propagate NULL;
/// a non-numeric element or a non-vector value is an error.
fn coerce_vector(
    v: &RuntimeValue,
    fname: &str,
    span: SourceSpan,
) -> Result<Option<Vec<f32>>, EvalError> {
    match v {
        RuntimeValue::Null => Ok(None),
        RuntimeValue::Vector(items) => Ok(Some(items.clone())),
        // Asymmetric scoring: a stored int8 vector dequantizes to f32 here, so a
        // similarity against an f32 query (or another vector) is computed in f32
        // with f64 accumulation. `code * scale` recovers the approximation the
        // recall harness measured (RFC int8 quantization).
        RuntimeValue::Vector8 { codes, scale } => {
            Ok(Some(codes.iter().map(|&c| c as f32 * *scale).collect()))
        }
        RuntimeValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (idx, it) in items.iter().enumerate() {
                match it {
                    RuntimeValue::Float(f) => out.push(*f as f32),
                    RuntimeValue::Integer(n) => out.push(*n as f32),
                    other => {
                        return Err(EvalError::new(
                            format!(
                                "{}: vector elements must be numeric; got {} at index {}",
                                fname,
                                other.type_name(),
                                idx
                            ),
                            span,
                        ));
                    }
                }
            }
            Ok(Some(out))
        }
        other => Err(EvalError::new(
            format!("{} requires vectors; got {}", fname, other.type_name()),
            span,
        )),
    }
}

/// A coerced pair of equal-length vector operands, or `None` when either
/// operand was NULL (so the caller propagates NULL).
type VectorPair = Option<(Vec<f32>, Vec<f32>)>;

/// Pull the two operands of a vector similarity/distance builtin, returning
/// `Ok(None)` when either is NULL so the caller yields NULL. Errors on arity, a
/// non-vector operand, or a dimension mismatch.
fn vector_pair(
    name: &str,
    args: &[RuntimeValue],
    span: SourceSpan,
) -> Result<VectorPair, EvalError> {
    let (a, b) = match args {
        [a, b] => (a, b),
        _ => {
            return Err(EvalError::new(
                format!("`{}` expects exactly 2 vectors", name),
                span,
            ))
        }
    };
    let va = match coerce_vector(a, name, span)? {
        Some(v) => v,
        None => return Ok(None),
    };
    let vb = match coerce_vector(b, name, span)? {
        Some(v) => v,
        None => return Ok(None),
    };
    if va.len() != vb.len() {
        return Err(EvalError::new(
            format!(
                "{}: vectors must have the same dimension ({} vs {})",
                name,
                va.len(),
                vb.len()
            ),
            span,
        ));
    }
    Ok(Some((va, vb)))
}

/// `left(s, n)` / `right(s, n)`: the first / last `n` characters.
fn str_left_right(
    args: &[RuntimeValue],
    span: SourceSpan,
    left: bool,
) -> Result<RuntimeValue, EvalError> {
    let fname = if left {
        "left(string, length)"
    } else {
        "right(string, length)"
    };
    let (s, n) = match args {
        [RuntimeValue::String(s), n] => (s, n),
        [RuntimeValue::Null, _] => return Ok(RuntimeValue::Null),
        _ => {
            return Err(EvalError::new(
                format!("{} expects a string and an integer", fname),
                span,
            ))
        }
    };
    let n = match want_int(n, fname, span)? {
        Some(n) => n,
        None => return Ok(RuntimeValue::Null),
    };
    if n < 0 {
        return Err(EvalError::new(
            format!("{}: length must be non-negative", fname),
            span,
        ));
    }
    let chars: Vec<char> = s.chars().collect();
    let n = (n as usize).min(chars.len());
    let out: String = if left {
        chars[..n].iter().collect()
    } else {
        chars[chars.len() - n..].iter().collect()
    };
    Ok(RuntimeValue::String(out))
}

/// `substring(s, start[, length])`: 0-based, by character (Neo4j semantics).
fn str_substring(args: &[RuntimeValue], span: SourceSpan) -> Result<RuntimeValue, EvalError> {
    let fname = "substring(string, start[, length])";
    let (s, start, len) = match args {
        [RuntimeValue::String(s), start] => (s, start, None),
        [RuntimeValue::String(s), start, len] => (s, start, Some(len)),
        [RuntimeValue::Null, ..] => return Ok(RuntimeValue::Null),
        _ => {
            return Err(EvalError::new(
                format!("{} expects a string and integer offsets", fname),
                span,
            ))
        }
    };
    let start = match want_int(start, fname, span)? {
        Some(n) => n,
        None => return Ok(RuntimeValue::Null),
    };
    if start < 0 {
        return Err(EvalError::new(
            format!("{}: start must be non-negative", fname),
            span,
        ));
    }
    let chars: Vec<char> = s.chars().collect();
    let start = (start as usize).min(chars.len());
    let end = match len {
        None => chars.len(),
        Some(l) => {
            let l = match want_int(l, fname, span)? {
                Some(l) => l,
                None => return Ok(RuntimeValue::Null),
            };
            if l < 0 {
                return Err(EvalError::new(
                    format!("{}: length must be non-negative", fname),
                    span,
                ));
            }
            start.saturating_add(l as usize).min(chars.len())
        }
    };
    Ok(RuntimeValue::String(chars[start..end].iter().collect()))
}

/// `replace(s, search, replacement)`: replace every occurrence.
fn str_replace(args: &[RuntimeValue], span: SourceSpan) -> Result<RuntimeValue, EvalError> {
    match args {
        [RuntimeValue::String(s), RuntimeValue::String(search), RuntimeValue::String(rep)] => {
            Ok(RuntimeValue::String(s.replace(search.as_str(), rep)))
        }
        [RuntimeValue::Null, _, _] | [_, RuntimeValue::Null, _] | [_, _, RuntimeValue::Null] => {
            Ok(RuntimeValue::Null)
        }
        _ => Err(EvalError::new(
            "replace(string, search, replacement) expects three strings",
            span,
        )),
    }
}

/// `split(s, delimiter)`: an empty delimiter splits into characters.
fn str_split(args: &[RuntimeValue], span: SourceSpan) -> Result<RuntimeValue, EvalError> {
    match args {
        [RuntimeValue::String(s), RuntimeValue::String(delim)] => {
            let parts: Vec<RuntimeValue> = if delim.is_empty() {
                s.chars()
                    .map(|c| RuntimeValue::String(c.to_string()))
                    .collect()
            } else {
                s.split(delim.as_str())
                    .map(|p| RuntimeValue::String(p.to_string()))
                    .collect()
            };
            Ok(RuntimeValue::List(parts))
        }
        [RuntimeValue::Null, _] | [_, RuntimeValue::Null] => Ok(RuntimeValue::Null),
        _ => Err(EvalError::new(
            "split(string, delimiter) expects two strings",
            span,
        )),
    }
}

/// `range(start, end[, step])`: an inclusive integer list (Neo4j semantics).
fn range_fn(args: &[RuntimeValue], span: SourceSpan) -> Result<RuntimeValue, EvalError> {
    let (start, end, step) = match args {
        [a, b] => (a, b, None),
        [a, b, c] => (a, b, Some(c)),
        _ => {
            return Err(EvalError::new(
                "range(start, end[, step]) expects two or three integers",
                span,
            ))
        }
    };
    let (start, end) = match (
        want_int(start, "range()", span)?,
        want_int(end, "range()", span)?,
    ) {
        (Some(s), Some(e)) => (s, e),
        _ => return Ok(RuntimeValue::Null),
    };
    let step = match step {
        None => 1,
        Some(v) => match want_int(v, "range()", span)? {
            Some(s) => s,
            None => return Ok(RuntimeValue::Null),
        },
    };
    if step == 0 {
        return Err(EvalError::new("range(): step must be non-zero", span));
    }
    let mut out = Vec::new();
    let mut i = start;
    loop {
        if (step > 0 && i > end) || (step < 0 && i < end) {
            break;
        }
        out.push(RuntimeValue::Integer(i));
        match i.checked_add(step) {
            Some(next) => i = next,
            None => break,
        }
    }
    Ok(RuntimeValue::List(out))
}

fn eval_list_comprehension(
    lc: &crate::parser::ListComprehension,
    row: &Row,
    params: &Params,
) -> Result<RuntimeValue, EvalError> {
    let list_v = evaluate(&lc.list, row, params)?;
    let items = match list_v {
        RuntimeValue::List(items) => items,
        RuntimeValue::Null => return Ok(RuntimeValue::Null),
        other => {
            return Err(EvalError::new(
                format!(
                    "list comprehension requires a list, got {}",
                    other.type_name()
                ),
                lc.list.span,
            ));
        }
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let mut local = row.clone();
        local.set(lc.variable.name.clone(), item.clone());
        if let Some(pred) = &lc.predicate {
            let v = evaluate(pred, &local, params)?;
            if v.as_bool() != Some(true) {
                continue;
            }
        }
        let projected = match &lc.projection {
            Some(p) => evaluate(p, &local, params)?,
            None => item,
        };
        out.push(projected);
    }
    Ok(RuntimeValue::List(out))
}

#[cfg(test)]
mod tests {
    use super::super::value::NodeValue;
    use super::*;
    use crate::parser::parse;

    fn row_with(bindings: &[(&str, RuntimeValue)]) -> Row {
        let mut r = Row::new();
        for (k, v) in bindings {
            r.set(*k, v.clone());
        }
        r
    }

    fn eval_str(src: &str, row: &Row, params: &Params) -> RuntimeValue {
        let q = parse(&format!("RETURN {} AS r", src)).unwrap();
        let item = match &q.head.clauses[0] {
            crate::parser::Clause::Return(r) => &r.items[0].expression,
            _ => panic!(),
        };
        evaluate(item, row, params).unwrap()
    }

    #[test]
    fn arith_int_add() {
        let r = eval_str("1 + 2", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Integer(3));
    }

    #[test]
    fn timestamp_returns_epoch_millis() {
        let r = eval_str("timestamp()", &Row::new(), &Params::new());
        match r {
            // Sanity: after 2023-01-01T00:00:00Z in milliseconds.
            RuntimeValue::Integer(ms) => assert!(ms > 1_672_531_200_000, "too small: {ms}"),
            other => panic!("timestamp() should be an integer, got {other:?}"),
        }
    }

    #[test]
    fn arith_int_float_promotion() {
        let r = eval_str("1 + 2.5", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Float(3.5));
    }

    #[test]
    fn null_propagates_through_arith() {
        let r = eval_str("1 + NULL", &Row::new(), &Params::new());
        assert!(r.is_null());
    }

    #[test]
    fn three_valued_and_or() {
        assert_eq!(
            eval_str("TRUE AND FALSE", &Row::new(), &Params::new()),
            RuntimeValue::Bool(false)
        );
        assert_eq!(
            eval_str("TRUE AND NULL", &Row::new(), &Params::new()),
            RuntimeValue::Null
        );
        assert_eq!(
            eval_str("FALSE AND NULL", &Row::new(), &Params::new()),
            RuntimeValue::Bool(false)
        );
        assert_eq!(
            eval_str("TRUE OR NULL", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
        assert_eq!(
            eval_str("NULL OR NULL", &Row::new(), &Params::new()),
            RuntimeValue::Null
        );
    }

    #[test]
    fn in_list_with_null() {
        assert_eq!(
            eval_str("1 IN [1, 2, 3]", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
        assert_eq!(
            eval_str("5 IN [1, 2, 3]", &Row::new(), &Params::new()),
            RuntimeValue::Bool(false)
        );
        assert_eq!(
            eval_str("5 IN [1, NULL, 3]", &Row::new(), &Params::new()),
            RuntimeValue::Null
        );
    }

    #[test]
    fn is_null_returns_bool() {
        assert_eq!(
            eval_str("NULL IS NULL", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
        assert_eq!(
            eval_str("1 IS NOT NULL", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
    }

    #[test]
    fn property_access_on_node() {
        let mut props = BTreeMap::new();
        props.insert("name".into(), RuntimeValue::String("Ada".into()));
        let node = RuntimeValue::Node(Box::new(NodeValue {
            id: namidb_core::id::NodeId::new(),
            labels: std::iter::once("Person".to_string()).collect(),
            properties: props,
        }));
        let row = row_with(&[("a", node)]);
        let r = eval_str("a.name", &row, &Params::new());
        assert_eq!(r, RuntimeValue::String("Ada".into()));
        let r = eval_str("a.missing", &row, &Params::new());
        assert!(r.is_null());
    }

    #[test]
    fn parameter_lookup() {
        let mut params = Params::new();
        params.insert("x".into(), RuntimeValue::Integer(7));
        let r = eval_str("$x", &Row::new(), &params);
        assert_eq!(r, RuntimeValue::Integer(7));
    }

    #[test]
    fn order_cmp_int_vs_float() {
        assert_eq!(
            eval_str("1 < 2.5", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
    }

    #[test]
    fn string_concat_via_plus() {
        assert_eq!(
            eval_str("'hello' + ' ' + 'world'", &Row::new(), &Params::new()),
            RuntimeValue::String("hello world".into())
        );
    }

    #[test]
    fn coalesce_picks_first_non_null() {
        let r = eval_str("coalesce(NULL, NULL, 42)", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Integer(42));
    }

    #[test]
    fn integer_division_by_zero_errors() {
        let q = parse("RETURN 1 / 0 AS r").unwrap();
        let item = match &q.head.clauses[0] {
            crate::parser::Clause::Return(r) => &r.items[0].expression,
            _ => panic!(),
        };
        let err = evaluate(item, &Row::new(), &Params::new()).unwrap_err();
        assert!(err.message.contains("division"));
    }

    #[test]
    fn case_expression_matches_branch() {
        let r = eval_str(
            "CASE WHEN 1 = 2 THEN 'a' WHEN 2 > 1 THEN 'b' ELSE 'c' END",
            &Row::new(),
            &Params::new(),
        );
        assert_eq!(r, RuntimeValue::String("b".into()));
    }

    // ─── `=~` regex match (Neo4j whole-string semantics) ──────────

    #[test]
    fn regex_match_is_whole_string_not_substring() {
        // The regression: `=~` used to be substring `contains`, so this
        // bare-substring pattern wrongly matched. It must now be false.
        assert_eq!(
            eval_str("'hello' =~ 'ell'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(false)
        );
        // A pattern that spans the whole string matches.
        assert_eq!(
            eval_str("'hello' =~ 'h.*o'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
        assert_eq!(
            eval_str("'hello' =~ 'hello'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
    }

    #[test]
    fn regex_match_classes_and_quantifiers() {
        assert_eq!(
            eval_str("'abc123' =~ '[a-z]+[0-9]+'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
        assert_eq!(
            eval_str("'abc' =~ '[0-9]+'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(false)
        );
    }

    #[test]
    fn regex_match_alternation_is_anchored() {
        // `^(?:cat|dog)$`: a top-level alternation binds under both anchors.
        assert_eq!(
            eval_str("'cat' =~ 'cat|dog'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
        // Trailing char fails because the match must cover the whole string.
        assert_eq!(
            eval_str("'cats' =~ 'cat|dog'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(false)
        );
    }

    #[test]
    fn regex_match_inline_case_insensitive_flag() {
        assert_eq!(
            eval_str("'HELLO' =~ '(?i)hello'", &Row::new(), &Params::new()),
            RuntimeValue::Bool(true)
        );
    }

    #[test]
    fn regex_match_non_string_operand_is_null() {
        assert!(eval_str("123 =~ '1'", &Row::new(), &Params::new()).is_null());
    }

    #[test]
    fn regex_match_invalid_pattern_errors() {
        let err = eval_expr_err("'x' =~ '['");
        assert!(
            err.message.contains("invalid regular expression"),
            "unexpected message: {}",
            err.message
        );
    }

    // ─── scalar string / math / list builtins ─────────────────────

    fn s(src: &str) -> RuntimeValue {
        eval_str(src, &Row::new(), &Params::new())
    }
    fn ints(xs: &[i64]) -> RuntimeValue {
        RuntimeValue::List(xs.iter().map(|n| RuntimeValue::Integer(*n)).collect())
    }
    fn strs(xs: &[&str]) -> RuntimeValue {
        RuntimeValue::List(
            xs.iter()
                .map(|x| RuntimeValue::String((*x).into()))
                .collect(),
        )
    }

    #[test]
    fn builtin_substring() {
        assert_eq!(
            s("substring('hello', 1, 3)"),
            RuntimeValue::String("ell".into())
        );
        assert_eq!(
            s("substring('hello', 2)"),
            RuntimeValue::String("llo".into())
        );
        // length past the end clamps to the end rather than erroring.
        assert_eq!(
            s("substring('hi', 1, 99)"),
            RuntimeValue::String("i".into())
        );
        assert!(s("substring(NULL, 0)").is_null());
        assert!(eval_expr_err("substring('x', -1)")
            .message
            .contains("non-negative"));
    }

    #[test]
    fn builtin_left_right() {
        assert_eq!(s("left('hello', 3)"), RuntimeValue::String("hel".into()));
        assert_eq!(s("right('hello', 3)"), RuntimeValue::String("llo".into()));
        // n beyond the length returns the whole string.
        assert_eq!(s("left('hi', 9)"), RuntimeValue::String("hi".into()));
    }

    #[test]
    fn builtin_trims_and_reverse() {
        assert_eq!(s("ltrim('  hi ')"), RuntimeValue::String("hi ".into()));
        assert_eq!(s("rtrim(' hi  ')"), RuntimeValue::String(" hi".into()));
        assert_eq!(s("reverse('abc')"), RuntimeValue::String("cba".into()));
        assert_eq!(s("reverse([1, 2, 3])"), ints(&[3, 2, 1]));
    }

    #[test]
    fn builtin_replace_and_split() {
        assert_eq!(
            s("replace('a-b-c', '-', ':')"),
            RuntimeValue::String("a:b:c".into())
        );
        assert_eq!(s("split('a,b,c', ',')"), strs(&["a", "b", "c"]));
        assert_eq!(s("split('abc', '')"), strs(&["a", "b", "c"]));
    }

    #[test]
    fn builtin_round_floor_ceil_sqrt() {
        assert_eq!(s("round(2.4)"), RuntimeValue::Float(2.0));
        assert_eq!(s("round(2.6)"), RuntimeValue::Float(3.0));
        assert_eq!(s("floor(2.9)"), RuntimeValue::Float(2.0));
        assert_eq!(s("ceil(2.1)"), RuntimeValue::Float(3.0));
        // integer argument is promoted to Float first.
        assert_eq!(s("sqrt(9)"), RuntimeValue::Float(3.0));
    }

    #[test]
    fn builtin_sign() {
        assert_eq!(s("sign(-5)"), RuntimeValue::Integer(-1));
        assert_eq!(s("sign(0)"), RuntimeValue::Integer(0));
        assert_eq!(s("sign(3.2)"), RuntimeValue::Integer(1));
    }

    #[test]
    fn builtin_toboolean() {
        assert_eq!(s("toBoolean('TRUE')"), RuntimeValue::Bool(true));
        assert_eq!(s("toBoolean('false')"), RuntimeValue::Bool(false));
        assert_eq!(s("toBoolean(0)"), RuntimeValue::Bool(false));
        assert!(s("toBoolean('nope')").is_null());
    }

    #[test]
    fn builtin_range() {
        assert_eq!(s("range(0, 3)"), ints(&[0, 1, 2, 3]));
        assert_eq!(s("range(0, 10, 2)"), ints(&[0, 2, 4, 6, 8, 10]));
        assert_eq!(s("range(3, 0, -1)"), ints(&[3, 2, 1, 0]));
        assert!(eval_expr_err("range(0, 5, 0)").message.contains("non-zero"));
    }

    #[test]
    fn list_index_negative() {
        let r = eval_str("[10, 20, 30][-1]", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Integer(30));
    }

    #[test]
    fn list_range_slice() {
        let r = eval_str("[1, 2, 3, 4, 5][1..3]", &Row::new(), &Params::new());
        assert_eq!(
            r,
            RuntimeValue::List(vec![RuntimeValue::Integer(2), RuntimeValue::Integer(3)])
        );
    }

    #[test]
    fn list_comprehension_projection_only() {
        let r = eval_str("[x IN [1, 2, 3] | x * 2]", &Row::new(), &Params::new());
        assert_eq!(
            r,
            RuntimeValue::List(vec![
                RuntimeValue::Integer(2),
                RuntimeValue::Integer(4),
                RuntimeValue::Integer(6),
            ])
        );
    }

    #[test]
    fn list_comprehension_predicate_only() {
        let r = eval_str(
            "[x IN [1, 2, 3, 4] WHERE x > 2]",
            &Row::new(),
            &Params::new(),
        );
        assert_eq!(
            r,
            RuntimeValue::List(vec![RuntimeValue::Integer(3), RuntimeValue::Integer(4)])
        );
    }

    #[test]
    fn list_comprehension_predicate_and_projection() {
        let r = eval_str(
            "[x IN [1, 2, 3, 4] WHERE x % 2 = 0 | x * 10]",
            &Row::new(),
            &Params::new(),
        );
        assert_eq!(
            r,
            RuntimeValue::List(vec![RuntimeValue::Integer(20), RuntimeValue::Integer(40)])
        );
    }

    #[test]
    fn list_comprehension_null_list_returns_null() {
        let r = eval_str("[x IN NULL | x]", &Row::new(), &Params::new());
        assert!(r.is_null());
    }

    #[test]
    fn datetime_no_args_returns_datetime_value() {
        let r = eval_str("datetime()", &Row::new(), &Params::new());
        assert!(matches!(r, RuntimeValue::DateTime(_)));
    }

    #[test]
    fn datetime_parses_rfc3339_string() {
        let r = eval_str(
            "datetime('2024-01-02T03:04:05Z')",
            &Row::new(),
            &Params::new(),
        );
        // 2024-01-02T03:04:05Z == 1_704_164_645 s since the epoch.
        assert_eq!(r, RuntimeValue::DateTime(1_704_164_645_000_000));
    }

    #[test]
    fn datetime_of_null_is_null() {
        let r = eval_str("datetime(null)", &Row::new(), &Params::new());
        assert!(r.is_null());
    }

    #[test]
    fn date_parses_iso_date_to_days_since_epoch() {
        let r = eval_str("date('1970-01-02')", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Date(1));
    }

    #[test]
    fn date_no_args_returns_date_value() {
        let r = eval_str("date()", &Row::new(), &Params::new());
        assert!(matches!(r, RuntimeValue::Date(_)));
    }

    #[test]
    fn list_comprehension_can_reference_outer_row() {
        let row = row_with(&[("threshold", RuntimeValue::Integer(2))]);
        let r = eval_str(
            "[x IN [1, 2, 3, 4] WHERE x > threshold | x]",
            &row,
            &Params::new(),
        );
        assert_eq!(
            r,
            RuntimeValue::List(vec![RuntimeValue::Integer(3), RuntimeValue::Integer(4)])
        );
    }

    #[test]
    fn dot_underscore_id_on_node_returns_uuid_string() {
        let id = namidb_core::id::NodeId::new();
        let node = RuntimeValue::Node(Box::new(NodeValue {
            id,
            labels: std::iter::once("Person".to_string()).collect(),
            properties: BTreeMap::new(),
        }));
        let row = row_with(&[("a", node)]);
        let r = eval_str("a._id", &row, &Params::new());
        assert_eq!(r, RuntimeValue::String(id.to_string()));
    }

    #[test]
    fn dot_id_on_node_returns_user_property_not_internal_id() {
        // Regression: `id` was previously aliased to the internal NodeId.
        // After the rename to `_id`, plain `id` must surface the
        // user-owned property value verbatim.
        let id = namidb_core::id::NodeId::new();
        let mut props = BTreeMap::new();
        props.insert("id".into(), RuntimeValue::String("external-42".into()));
        let node = RuntimeValue::Node(Box::new(NodeValue {
            id,
            labels: std::iter::once("Person".to_string()).collect(),
            properties: props,
        }));
        let row = row_with(&[("a", node)]);
        let r = eval_str("a.id", &row, &Params::new());
        assert_eq!(r, RuntimeValue::String("external-42".into()));
    }

    #[test]
    fn dot_id_on_node_without_property_returns_null() {
        // No user-defined `id` property → accessor must yield Null, not
        // fall back to the internal NodeId.
        let node = RuntimeValue::Node(Box::new(NodeValue {
            id: namidb_core::id::NodeId::new(),
            labels: std::iter::once("Person".to_string()).collect(),
            properties: BTreeMap::new(),
        }));
        let row = row_with(&[("a", node)]);
        let r = eval_str("a.id", &row, &Params::new());
        assert_eq!(r, RuntimeValue::Null);
    }

    #[test]
    fn label_eq_internal_function() {
        let node = RuntimeValue::Node(Box::new(NodeValue {
            id: namidb_core::id::NodeId::new(),
            labels: std::iter::once("Person".to_string()).collect(),
            properties: BTreeMap::new(),
        }));
        let row = row_with(&[("a", node)]);
        let q = parse("RETURN __label_eq(a, 'Person') AS r").unwrap();
        let expr = match &q.head.clauses[0] {
            crate::parser::Clause::Return(r) => &r.items[0].expression,
            _ => panic!(),
        };
        let r = evaluate(expr, &row, &Params::new()).unwrap();
        assert_eq!(r, RuntimeValue::Bool(true));
    }

    // ─── vector() constructor ─────────────────────────────────────

    fn eval_expr_err(src: &str) -> EvalError {
        let q = parse(&format!("RETURN {} AS r", src)).unwrap();
        let item = match &q.head.clauses[0] {
            crate::parser::Clause::Return(r) => &r.items[0].expression,
            _ => panic!(),
        };
        evaluate(item, &Row::new(), &Params::new()).unwrap_err()
    }

    #[test]
    fn vector_from_float_list() {
        let r = eval_str("vector([0.1, 0.2, 0.3])", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Vector(vec![0.1_f32, 0.2_f32, 0.3_f32]));
    }

    #[test]
    fn vector_from_integer_list_promotes_to_f32() {
        let r = eval_str("vector([1, 2, 3])", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Vector(vec![1.0_f32, 2.0_f32, 3.0_f32]));
    }

    #[test]
    fn vector_from_mixed_int_float_list() {
        let r = eval_str("vector([1, 2.5, 3])", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Vector(vec![1.0_f32, 2.5_f32, 3.0_f32]));
    }

    #[test]
    fn vector_empty_list() {
        let r = eval_str("vector([])", &Row::new(), &Params::new());
        assert_eq!(r, RuntimeValue::Vector(Vec::new()));
    }

    #[test]
    fn vector_null_passthrough() {
        let r = eval_str("vector(NULL)", &Row::new(), &Params::new());
        assert!(r.is_null());
    }

    #[test]
    fn vector_idempotent_on_existing_vector() {
        // `vector(vec)` where `vec` is already a Vector — uses a parameter
        // because there is no Cypher literal for Vector. Idempotency keeps
        // composition (e.g. `vector(vector(x))`) safe.
        let mut params = Params::new();
        params.insert("v".into(), RuntimeValue::Vector(vec![1.0_f32, 2.0_f32]));
        let q = parse("RETURN vector($v) AS r").unwrap();
        let item = match &q.head.clauses[0] {
            crate::parser::Clause::Return(r) => &r.items[0].expression,
            _ => panic!(),
        };
        let r = evaluate(item, &Row::new(), &params).unwrap();
        assert_eq!(r, RuntimeValue::Vector(vec![1.0_f32, 2.0_f32]));
    }

    #[test]
    fn vector_rejects_non_numeric_element() {
        let err = eval_expr_err(r#"vector([1, "x", 3])"#);
        assert!(
            err.message.contains("vector()")
                && err.message.contains("STRING")
                && err.message.contains("index 1"),
            "unexpected message: {}",
            err.message
        );
    }

    #[test]
    fn vector_rejects_null_element() {
        let err = eval_expr_err("vector([1.0, NULL, 3.0])");
        assert!(
            err.message.contains("vector()")
                && err.message.contains("NULL")
                && err.message.contains("index 1"),
            "unexpected message: {}",
            err.message
        );
    }

    #[test]
    fn vector_rejects_non_list_argument() {
        let err = eval_expr_err(r#"vector("not a list")"#);
        assert!(
            err.message.contains("vector()") && err.message.contains("STRING"),
            "unexpected message: {}",
            err.message
        );
    }

    #[test]
    fn vector_requires_single_argument() {
        let err = eval_expr_err("vector([1.0], [2.0])");
        assert!(
            err.message.contains("vector") && err.message.contains("exactly 1"),
            "unexpected message: {}",
            err.message
        );
    }

    // ─── vector similarity / distance builtins ────────────────────

    #[test]
    fn builtin_vector_distances() {
        // Identical unit vectors: cosine 1.0.
        assert_eq!(
            s("cosine_similarity(vector([1.0, 0.0, 0.0]), vector([1.0, 0.0, 0.0]))"),
            RuntimeValue::Float(1.0)
        );
        // Orthogonal: cosine 0.0.
        assert_eq!(
            s("cosine_similarity(vector([1.0, 0.0]), vector([0.0, 1.0]))"),
            RuntimeValue::Float(0.0)
        );
        // Opposite direction: cosine -1.0.
        assert_eq!(
            s("cosine_similarity(vector([1.0, 0.0]), vector([-1.0, 0.0]))"),
            RuntimeValue::Float(-1.0)
        );
        // dot_product and euclidean_distance on exact-representable inputs.
        assert_eq!(
            s("dot_product(vector([1.0, 2.0, 3.0]), vector([4.0, 5.0, 6.0]))"),
            RuntimeValue::Float(32.0)
        );
        assert_eq!(
            s("euclidean_distance(vector([0.0, 0.0]), vector([3.0, 4.0]))"),
            RuntimeValue::Float(5.0)
        );
    }

    #[test]
    fn builtin_vector_accepts_bare_numeric_lists() {
        // A `$param` array arrives as a List, not a Vector, so the distance
        // builtins coerce numeric lists directly (no explicit `vector()` wrap).
        assert_eq!(
            s("cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])"),
            RuntimeValue::Float(1.0)
        );
        assert_eq!(
            s("euclidean_distance([0, 0], [3, 4])"),
            RuntimeValue::Float(5.0)
        );
    }

    #[test]
    fn cosine_over_int8_approximates_exact_f32() {
        use namidb_core::quantize::quantize_i8;
        // A stored vector quantized to int8, scored against an f32 query: the
        // asymmetric scorer dequantizes and must land close to the exact f32
        // cosine (the recall the int8 harness validated). Vectors arrive via
        // params because there is no Cypher literal for a Vector8.
        let stored = vec![0.2f32, -0.5, 0.9, 0.1, -0.3, 0.7, -0.15];
        let query = vec![0.1f32, -0.4, 0.8, 0.2, -0.2, 0.6, -0.1];
        let (codes, scale) = quantize_i8(&stored);

        let mut p_i8 = Params::new();
        p_i8.insert("emb".into(), RuntimeValue::Vector8 { codes, scale });
        p_i8.insert("q".into(), RuntimeValue::Vector(query.clone()));
        let int8_cos = eval_str("cosine_similarity($emb, $q)", &Row::new(), &p_i8);

        let mut p_f = Params::new();
        p_f.insert("emb".into(), RuntimeValue::Vector(stored));
        p_f.insert("q".into(), RuntimeValue::Vector(query));
        let exact_cos = eval_str("cosine_similarity($emb, $q)", &Row::new(), &p_f);

        match (int8_cos, exact_cos) {
            (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                assert!((a - b).abs() < 0.01, "int8 cosine {a} vs exact {b}");
            }
            other => panic!("expected floats, got {other:?}"),
        }

        // size() over an int8 vector returns its dimension.
        let (codes, scale) = quantize_i8(&[0.1f32, 0.2, 0.3]);
        let mut p = Params::new();
        p.insert("emb".into(), RuntimeValue::Vector8 { codes, scale });
        assert_eq!(
            eval_str("size($emb)", &Row::new(), &p),
            RuntimeValue::Integer(3)
        );
    }

    #[test]
    fn builtin_vector_null_propagates() {
        assert!(s("cosine_similarity(NULL, vector([1.0, 0.0]))").is_null());
        assert!(s("dot_product(vector([1.0]), NULL)").is_null());
    }

    #[test]
    fn builtin_vector_zero_magnitude_is_null() {
        // Cosine is undefined when a vector has zero magnitude.
        assert!(s("cosine_similarity(vector([0.0, 0.0]), vector([1.0, 1.0]))").is_null());
    }

    #[test]
    fn builtin_bm25_scores_and_propagates_null() {
        // A matching term scores positive; no match is zero.
        match s("bm25('the quick brown fox', 'fox')") {
            RuntimeValue::Float(x) => assert!(x > 0.0, "expected positive, got {x}"),
            other => panic!("expected Float, got {other:?}"),
        }
        match s("bm25('the quick brown fox', 'elephant')") {
            RuntimeValue::Float(x) => assert_eq!(x, 0.0),
            other => panic!("expected Float, got {other:?}"),
        }
        // NULL propagates (three-valued logic).
        assert!(s("bm25(NULL, 'fox')").is_null());
        assert!(s("bm25('fox', NULL)").is_null());
    }

    #[test]
    fn builtin_bm25_type_error() {
        let err = eval_expr_err("bm25('doc', 42)");
        assert!(
            err.message.contains("two strings"),
            "unexpected message: {}",
            err.message
        );
    }

    #[test]
    fn builtin_vector_dimension_mismatch_errors() {
        let err = eval_expr_err("cosine_similarity(vector([1.0, 2.0]), vector([1.0]))");
        assert!(
            err.message.contains("same dimension"),
            "unexpected message: {}",
            err.message
        );
    }

    #[test]
    fn builtin_vector_wrong_arity_errors() {
        let err = eval_expr_err("dot_product(vector([1.0]))");
        assert!(
            err.message.contains("exactly 2"),
            "unexpected message: {}",
            err.message
        );
    }

    #[test]
    fn builtin_size_of_vector_is_dimension() {
        assert_eq!(s("size(vector([1.0, 2.0, 3.0]))"), RuntimeValue::Integer(3));
    }
}
