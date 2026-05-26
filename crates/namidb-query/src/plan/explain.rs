//! Pretty-printer for [`LogicalPlan`].
//!
//! Two output shapes:
//!
//! * `explain*` — indented tree string, stable enough to assert on in
//!   tests and to paste into bug reports.
//! * `explain_tree*` — `ExplainNode` struct (`Serialize`) for callers
//!   that want to render the plan in their own UI. The crate does not
//!   depend on `serde_json`; downstreams (e.g. the cloud worker) handle
//!   the JSON conversion themselves.

use std::fmt::Write;

use serde::Serialize;

use namidb_storage::sst::predicates::ScanPredicate;
use namidb_storage::sst::stats::StatScalar;

use super::logical::{AggregateExpr, CreateElement, LogicalPlan, RemoveOp, SetOp};
use super::lower::{lower, LowerError};
use crate::cost::{estimate, Cardinality, StatsCatalog};
use crate::optimize::{is_join_candidate, optimize, produced_aliases};
use crate::parser::{OrderDirection, Query, RelationshipDirection};

/// Structured rendering of a [`LogicalPlan`] node. `summary` is the
/// same one-line shape that `explain` emits per operator; the optional
/// fields are populated by the `*_verbose` variants when a cost catalog
/// is available.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ExplainNode {
    /// One-line operator description (`NodeScan label=Person alias=a`).
    pub summary: String,
    /// Cardinality estimate for this operator. Only populated by the
    /// verbose tree builders.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_rows: Option<u64>,
    /// Sum of estimates from the root downwards. Populated only on the
    /// root node of a verbose tree.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_total_work: Option<u64>,
    /// `true` when a `Filter` over `CrossProduct` could be turned into a
    /// hash join by [`crate::optimize::convert_cross_to_hash`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub join_candidate: Option<bool>,
    /// `true` when the optimiser had no catalog stats for this
    /// operator's label / edge type. Lets callers warn the user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_stats: Option<bool>,
    pub children: Vec<ExplainNode>,
}

/// Render `plan` as an indented tree string. Each operator takes one
/// line; children are indented by two spaces.
pub fn explain(plan: &LogicalPlan) -> String {
    let mut out = String::new();
    write_node(plan, 0, &mut out);
    out
}

/// Lower `query` and render its plan tree. Convenience wrapper for the
/// `EXPLAIN <query>` syntax — callers do not need to import the lowerer
/// separately.
pub fn explain_query(query: &Query) -> Result<String, LowerError> {
    let plan = lower(query)?;
    Ok(explain(&plan))
}

/// Render `plan` with per-operator estimated row counts derived from
/// `catalog` (RFC-010). The total estimate is emitted as a
/// header comment line.
///
/// Operators whose label/edge-type is missing from the catalog are
/// flagged with ` (no stats)` so callers can spot the gap.
pub fn explain_verbose(plan: &LogicalPlan, catalog: &StatsCatalog) -> String {
    let card = estimate(plan, catalog);
    let mut out = String::new();
    let _ = writeln!(&mut out, "# Estimated rows: {}", format_rows(card.rows));
    let _ = writeln!(
        &mut out,
        "# Estimated total work: {}",
        format_rows(sum_rows(&card))
    );
    write_node_verbose(plan, &card, catalog, 0, &mut out);
    out
}

/// Lower, optimize, and render with cardinality. This is what
/// `EXPLAIN VERBOSE <query>` invokes (RFC-011 §6.1). For the literal
/// lowering pre-optimize, see [`explain_query_raw_verbose`].
pub fn explain_query_verbose(query: &Query, catalog: &StatsCatalog) -> Result<String, LowerError> {
    let plan = optimize(lower(query)?, catalog);
    Ok(explain_verbose(&plan, catalog))
}

/// Render the lowering verbatim (no optimizer pipeline). Used by
/// `EXPLAIN RAW <query>` (RFC-011 §6.2). Without VERBOSE there are no
/// estimates, so the output matches the `EXPLAIN` shape.
pub fn explain_query_raw(query: &Query) -> Result<String, LowerError> {
    let plan = lower(query)?;
    Ok(explain(&plan))
}

/// Render the lowering verbatim with cardinality. Used by
/// `EXPLAIN RAW VERBOSE <query>` (RFC-011 §6.2). Lets the caller
/// compare estimates of the pre-optimize plan against the optimized
/// version (`explain_query_verbose`).
pub fn explain_query_raw_verbose(
    query: &Query,
    catalog: &StatsCatalog,
) -> Result<String, LowerError> {
    let plan = lower(query)?;
    Ok(explain_verbose(&plan, catalog))
}

/// Build an [`ExplainNode`] tree from `plan`. Same shape as
/// [`explain`] but structured — each operator gets its own node so
/// callers can render the plan in tables, JSON, or whatever wire
/// format they prefer.
pub fn explain_tree(plan: &LogicalPlan) -> ExplainNode {
    node_to_tree(plan)
}

/// Build a verbose [`ExplainNode`] tree with cardinality estimates,
/// `join_candidate` flags, and `no_stats` markers. Only the root node
/// carries `estimated_total_work`.
pub fn explain_tree_verbose(plan: &LogicalPlan, catalog: &StatsCatalog) -> ExplainNode {
    let card = estimate(plan, catalog);
    let mut root = node_to_tree_verbose(plan, &card, catalog);
    root.estimated_total_work = Some(format_rows(sum_rows(&card)));
    root
}

/// Lower `query` and build a structured tree. Convenience wrapper for
/// the equivalent of `EXPLAIN <query>` returning JSON-friendly data.
pub fn explain_query_tree(query: &Query) -> Result<ExplainNode, LowerError> {
    let plan = lower(query)?;
    Ok(explain_tree(&plan))
}

/// Lower, optimize, and render a verbose structured tree. The
/// equivalent of `EXPLAIN VERBOSE <query>` returning data instead of a
/// formatted string.
pub fn explain_query_tree_verbose(
    query: &Query,
    catalog: &StatsCatalog,
) -> Result<ExplainNode, LowerError> {
    let plan = optimize(lower(query)?, catalog);
    Ok(explain_tree_verbose(&plan, catalog))
}

/// Lower without the optimiser pipeline and return a structured tree.
/// Equivalent of `EXPLAIN RAW <query>`.
pub fn explain_query_raw_tree(query: &Query) -> Result<ExplainNode, LowerError> {
    let plan = lower(query)?;
    Ok(explain_tree(&plan))
}

/// Lower without the optimiser pipeline and return a verbose
/// structured tree. Equivalent of `EXPLAIN RAW VERBOSE <query>`.
pub fn explain_query_raw_tree_verbose(
    query: &Query,
    catalog: &StatsCatalog,
) -> Result<ExplainNode, LowerError> {
    let plan = lower(query)?;
    Ok(explain_tree_verbose(&plan, catalog))
}

fn node_to_tree(plan: &LogicalPlan) -> ExplainNode {
    let mut summary = String::new();
    write_header(plan, &mut summary);
    let children = plan.children().iter().map(|c| node_to_tree(c)).collect();
    ExplainNode {
        summary,
        estimated_rows: None,
        estimated_total_work: None,
        join_candidate: None,
        no_stats: None,
        children,
    }
}

fn node_to_tree_verbose(
    plan: &LogicalPlan,
    card: &Cardinality,
    catalog: &StatsCatalog,
) -> ExplainNode {
    let mut summary = String::new();
    write_header(plan, &mut summary);
    let join_candidate = if let LogicalPlan::Filter { input, predicate } = plan {
        if let LogicalPlan::CrossProduct { left, right } = input.as_ref() {
            if is_join_candidate(predicate, &produced_aliases(left), &produced_aliases(right)) {
                Some(true)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    let no_stats = if plan_has_stats(plan, catalog) {
        None
    } else {
        Some(true)
    };
    let child_plans = plan.children();
    let children = child_plans
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let child_card = card
                .children
                .get(i)
                .cloned()
                .unwrap_or_else(|| Cardinality {
                    rows: 0.0,
                    children: vec![],
                    bindings: Default::default(),
                    operator: "?",
                });
            node_to_tree_verbose(c, &child_card, catalog)
        })
        .collect();
    ExplainNode {
        summary,
        estimated_rows: Some(format_rows(card.rows)),
        estimated_total_work: None,
        join_candidate,
        no_stats,
        children,
    }
}

fn write_node(plan: &LogicalPlan, depth: usize, out: &mut String) {
    for _ in 0..depth {
        out.push(' ');
    }
    write_header(plan, out);
    out.push('\n');
    for child in plan.children() {
        write_node(child, depth + 1, out);
    }
}

fn write_node_verbose(
    plan: &LogicalPlan,
    card: &Cardinality,
    catalog: &StatsCatalog,
    depth: usize,
    out: &mut String,
) {
    for _ in 0..depth {
        out.push(' ');
    }
    write_header(plan, out);
    if let LogicalPlan::Filter { input, predicate } = plan {
        if let LogicalPlan::CrossProduct { left, right } = input.as_ref() {
            if is_join_candidate(predicate, &produced_aliases(left), &produced_aliases(right)) {
                out.push_str(" [join candidate]");
            }
        }
    }
    let _ = write!(out, " (est={}", format_rows(card.rows));
    if !plan_has_stats(plan, catalog) {
        out.push_str(", no stats");
    }
    out.push(')');
    out.push('\n');
    let children = plan.children();
    for (i, child) in children.iter().enumerate() {
        let child_card = card
            .children
            .get(i)
            .cloned()
            .unwrap_or_else(|| Cardinality {
                rows: 0.0,
                children: vec![],
                bindings: Default::default(),
                operator: "?",
            });
        write_node_verbose(child, &child_card, catalog, depth + 1, out);
    }
}

/// Round to the nearest non-negative integer, with `ceil` for
/// fractions in (0, 1) so an operator that emits "some" rows never
/// renders as `est=0`.
fn format_rows(r: f64) -> u64 {
    if !r.is_finite() {
        return 0;
    }
    if r > 0.0 && r < 1.0 {
        return 1;
    }
    r.round().max(0.0) as u64
}

fn sum_rows(card: &Cardinality) -> f64 {
    let mut total = card.rows.max(0.0);
    for child in &card.children {
        total += sum_rows(child);
    }
    total
}

fn plan_has_stats(plan: &LogicalPlan, catalog: &StatsCatalog) -> bool {
    match plan {
        LogicalPlan::NodeScan { label, .. } => label
            .as_deref()
            .and_then(|l| catalog.label(l))
            .map(|l| l.node_count > 0)
            .unwrap_or(false),
        LogicalPlan::NodeById { label, .. } | LogicalPlan::NodeByPropertyValue { label, .. } => {
            catalog
                .label(label)
                .map(|l| l.node_count > 0)
                .unwrap_or(false)
        }
        LogicalPlan::Expand {
            edge_type: Some(ets),
            ..
        } => ets.iter().any(|et| {
            catalog
                .edge_type(et)
                .map(|e| e.edge_count > 0)
                .unwrap_or(false)
        }),
        _ => true,
    }
}

fn write_header(plan: &LogicalPlan, out: &mut String) {
    match plan {
        LogicalPlan::NodeScan {
            label,
            alias,
            predicates,
            projection,
        } => {
            let label_str = label.as_deref().unwrap_or("*");
            let _ = write!(out, "NodeScan label={} alias={}", label_str, alias);
            if let Some(cols) = projection {
                let _ = write!(out, " projection=[{}]", cols.join(", "));
            }
            if !predicates.is_empty() {
                let rendered: Vec<String> = predicates
                    .iter()
                    .map(|p| format_scan_predicate(p, alias))
                    .collect();
                let _ = write!(out, " predicates=[{}]", rendered.join(", "));
            }
        }
        LogicalPlan::NodeById {
            label, alias, id, ..
        } => {
            let _ = write!(out, "NodeById label={} alias={} id={}", label, alias, id);
        }
        LogicalPlan::NodeByPropertyValue {
            label,
            alias,
            property,
            value,
            ..
        } => {
            let _ = write!(
                out,
                "NodeByPropertyValue label={} alias={} {}={}",
                label, alias, property, value
            );
        }
        LogicalPlan::Expand {
            source,
            edge_type,
            direction,
            rel_alias,
            target_alias,
            length,
            optional,
            ..
        } => {
            out.push_str(if *optional {
                "OptionalExpand "
            } else {
                "Expand "
            });
            let _ = write!(out, "source={}", source);
            if let Some(ts) = edge_type {
                let _ = write!(out, " edge_type={}", ts.join("|"));
            }
            let _ = write!(out, " dir={}", direction_label(*direction));
            if let Some(r) = rel_alias {
                let _ = write!(out, " rel={}", r);
            }
            let _ = write!(out, " target={}", target_alias);
            if let Some(len) = length {
                if len.min == len.max {
                    let _ = write!(out, " length={}", len.min);
                } else {
                    let _ = write!(out, " length={}..{}", len.min, len.max);
                }
            }
        }
        LogicalPlan::Filter { predicate, .. } => {
            let _ = write!(out, "Filter ({})", predicate);
        }
        LogicalPlan::Project {
            items,
            distinct,
            discard_input_bindings,
            ..
        } => {
            out.push_str(if *distinct {
                "ProjectDistinct "
            } else {
                "Project "
            });
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}={}", item.alias, item.expression);
            }
            out.push(']');
            if !*discard_input_bindings {
                out.push_str(" (keep)");
            }
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregations,
            ..
        } => {
            out.push_str("Aggregate");
            if !group_by.is_empty() {
                out.push_str(" group=[");
                for (i, (e, alias)) in group_by.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}={}", alias, e);
                }
                out.push(']');
            }
            out.push_str(" aggs=[");
            for (i, (alias, agg)) in aggregations.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}={}", alias, format_aggregate(agg));
            }
            out.push(']');
        }
        LogicalPlan::TopN {
            keys, skip, limit, ..
        } => {
            out.push_str("TopN");
            if !keys.is_empty() {
                out.push_str(" keys=[");
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(
                        out,
                        "{} {}",
                        k.expression,
                        match k.direction {
                            OrderDirection::Asc => "ASC",
                            OrderDirection::Desc => "DESC",
                        }
                    );
                }
                out.push(']');
            }
            if *skip > 0 {
                let _ = write!(out, " skip={}", skip);
            }
            if *limit != u64::MAX {
                let _ = write!(out, " limit={}", limit);
            }
        }
        LogicalPlan::Distinct { .. } => {
            out.push_str("Distinct");
        }
        LogicalPlan::Union { all, .. } => {
            out.push_str(if *all { "UnionAll" } else { "Union" });
        }
        LogicalPlan::Unwind { list, alias, .. } => {
            let _ = write!(out, "Unwind list={} alias={}", list, alias);
        }
        LogicalPlan::Empty => {
            out.push_str("Empty");
        }
        LogicalPlan::CrossProduct { .. } => {
            out.push_str("CrossProduct");
        }
        LogicalPlan::HashJoin { on, residual, .. } => {
            out.push_str("HashJoin on=[");
            for (i, k) in on.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "({}, {})", k.build_side, k.probe_side);
            }
            out.push(']');
            if let Some(r) = residual {
                let _ = write!(out, " residual=({})", r);
            }
        }
        LogicalPlan::HashSemiJoin {
            on,
            negated,
            residual,
            ..
        } => {
            out.push_str(if *negated {
                "AntiHashSemiJoin on=["
            } else {
                "HashSemiJoin on=["
            });
            for (i, k) in on.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "({}, {})", k.build_side, k.probe_side);
            }
            out.push(']');
            if let Some(r) = residual {
                let _ = write!(out, " residual=({})", r);
            }
        }
        LogicalPlan::Argument { bindings } => {
            out.push_str("Argument");
            if !bindings.is_empty() {
                out.push_str(" bindings=[");
                for (i, b) in bindings.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(b);
                }
                out.push(']');
            }
        }
        LogicalPlan::SemiApply { negated, .. } => {
            out.push_str(if *negated {
                "AntiSemiApply"
            } else {
                "SemiApply"
            });
        }
        LogicalPlan::PatternList {
            projection, alias, ..
        } => {
            let _ = write!(out, "PatternList alias={} projection={}", alias, projection);
        }
        LogicalPlan::Create { elements, .. } => {
            out.push_str("Create [");
            for (i, e) in elements.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_create_element(e, out);
            }
            out.push(']');
        }
        LogicalPlan::Merge {
            pattern,
            on_match_sets,
            on_create_sets,
            ..
        } => {
            out.push_str("Merge [");
            for (i, e) in pattern.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_create_element(e, out);
            }
            out.push(']');
            if !on_match_sets.is_empty() {
                out.push_str(" ON MATCH ");
                write_set_ops(on_match_sets, out);
            }
            if !on_create_sets.is_empty() {
                out.push_str(" ON CREATE ");
                write_set_ops(on_create_sets, out);
            }
        }
        LogicalPlan::Set { items, .. } => {
            out.push_str("Set ");
            write_set_ops(items, out);
        }
        LogicalPlan::Remove { items, .. } => {
            out.push_str("Remove [");
            for (i, op) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                match op {
                    RemoveOp::Property { target_alias, key } => {
                        let _ = write!(out, "{}.{}", target_alias, key);
                    }
                    RemoveOp::Labels {
                        target_alias,
                        labels,
                    } => {
                        let _ = write!(out, "{}", target_alias);
                        for l in labels {
                            let _ = write!(out, ":{}", l);
                        }
                    }
                }
            }
            out.push(']');
        }
        LogicalPlan::Delete {
            targets, detach, ..
        } => {
            out.push_str(if *detach {
                "DetachDelete ["
            } else {
                "Delete ["
            });
            for (i, t) in targets.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}", t);
            }
            out.push(']');
        }
        LogicalPlan::MultiwayJoin {
            vars,
            edges,
            ordering,
            ..
        } => {
            out.push_str("MultiwayJoin vars=[");
            for (i, idx) in ordering.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let v = &vars[*idx];
                if let Some(label) = &v.label {
                    let _ = write!(out, "{}:{}", v.alias, label);
                } else {
                    let _ = write!(out, "{}", v.alias);
                }
            }
            out.push_str("] edges=[");
            for (i, e) in edges.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(
                    out,
                    "{}-[:{}]{}-{}",
                    vars[e.from_idx].alias,
                    e.edge_types.join("|"),
                    direction_label(e.direction),
                    vars[e.to_idx].alias
                );
            }
            out.push(']');
        }
    }
}

fn write_create_element(e: &CreateElement, out: &mut String) {
    match e {
        CreateElement::Node {
            alias,
            label,
            properties,
            properties_spread,
        } => {
            let _ = write!(out, "({}:{}", alias, label);
            if !properties.is_empty() {
                out.push_str(" {");
                for (i, (k, v)) in properties.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}: {}", k, v);
                }
                out.push('}');
            }
            if let Some(spread) = properties_spread {
                let _ = write!(out, " spread={}", spread);
            }
            out.push(')');
        }
        CreateElement::Rel {
            alias,
            edge_type,
            source_alias,
            target_alias,
            direction,
            properties,
            properties_spread,
        } => {
            let arrow_l = matches!(direction, RelationshipDirection::Left);
            let arrow_r = matches!(direction, RelationshipDirection::Right);
            let _ = write!(out, "({})", source_alias);
            out.push_str(if arrow_l { "<-[" } else { "-[" });
            if let Some(a) = alias {
                let _ = write!(out, "{}", a);
            }
            let _ = write!(out, ":{}", edge_type);
            if !properties.is_empty() {
                out.push_str(" {");
                for (i, (k, v)) in properties.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}: {}", k, v);
                }
                out.push('}');
            }
            if let Some(spread) = properties_spread {
                let _ = write!(out, " spread={}", spread);
            }
            out.push_str(if arrow_r { "]->" } else { "]-" });
            let _ = write!(out, "({})", target_alias);
        }
    }
}

fn write_set_ops(items: &[SetOp], out: &mut String) {
    out.push('[');
    for (i, op) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match op {
            SetOp::Property {
                target_alias,
                key,
                value,
            } => {
                let _ = write!(out, "{}.{} = {}", target_alias, key, value);
            }
            SetOp::Replace {
                target_alias,
                value,
            } => {
                let _ = write!(out, "{} = {}", target_alias, value);
            }
            SetOp::Merge {
                target_alias,
                value,
            } => {
                let _ = write!(out, "{} += {}", target_alias, value);
            }
            SetOp::Labels {
                target_alias,
                labels,
            } => {
                let _ = write!(out, "{}", target_alias);
                for l in labels {
                    let _ = write!(out, ":{}", l);
                }
            }
        }
    }
    out.push(']');
}

fn direction_label(d: RelationshipDirection) -> &'static str {
    match d {
        RelationshipDirection::Right => "->",
        RelationshipDirection::Left => "<-",
        RelationshipDirection::Both => "--",
    }
}

/// Render a `ScanPredicate` in the same alias.property syntax used by
/// the source Cypher. The literal is rendered via its `StatScalar`.
fn format_scan_predicate(p: &ScanPredicate, alias: &str) -> String {
    match p {
        ScanPredicate::Eq { column, value } => {
            format!("{alias}.{column} = {}", format_stat_scalar(value))
        }
        ScanPredicate::Lt { column, value } => {
            format!("{alias}.{column} < {}", format_stat_scalar(value))
        }
        ScanPredicate::LtEq { column, value } => {
            format!("{alias}.{column} <= {}", format_stat_scalar(value))
        }
        ScanPredicate::Gt { column, value } => {
            format!("{alias}.{column} > {}", format_stat_scalar(value))
        }
        ScanPredicate::GtEq { column, value } => {
            format!("{alias}.{column} >= {}", format_stat_scalar(value))
        }
        ScanPredicate::Between { column, low, high } => {
            format!(
                "{alias}.{column} BETWEEN {} AND {}",
                format_stat_scalar(low),
                format_stat_scalar(high)
            )
        }
        ScanPredicate::IsNull { column } => format!("{alias}.{column} IS NULL"),
        ScanPredicate::IsNotNull { column } => format!("{alias}.{column} IS NOT NULL"),
        ScanPredicate::In { column, values } => {
            let rendered: Vec<String> = values.iter().map(format_stat_scalar).collect();
            format!("{alias}.{column} IN [{}]", rendered.join(", "))
        }
    }
}

fn format_stat_scalar(s: &StatScalar) -> String {
    match s {
        StatScalar::Bool(b) => b.to_string(),
        StatScalar::Int32(n) => n.to_string(),
        StatScalar::Int64(n) => n.to_string(),
        StatScalar::Float32(f) => f.to_string(),
        StatScalar::Float64(f) => f.to_string(),
        StatScalar::Utf8(s) => format!("\"{}\"", s),
        StatScalar::LargeUtf8(s) => format!("\"{}\"", s),
        StatScalar::Binary(b) => format!("0x{}", hex_encode(b)),
        StatScalar::Date32(n) => format!("date({n})"),
        StatScalar::TimestampMicrosUtc(n) => format!("timestamp({n})"),
    }
}

fn hex_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        let _ = write!(s, "{:02x}", byte);
    }
    s
}

fn format_aggregate(agg: &AggregateExpr) -> String {
    match agg {
        AggregateExpr::Count {
            arg: None,
            distinct: false,
        } => "count(*)".into(),
        AggregateExpr::Count {
            arg: None,
            distinct: true,
        } => "count(DISTINCT *)".into(),
        AggregateExpr::Count {
            arg: Some(e),
            distinct,
        } => {
            if *distinct {
                format!("count(DISTINCT {})", e)
            } else {
                format!("count({})", e)
            }
        }
        AggregateExpr::Sum { arg, distinct } => format_args("sum", arg, *distinct),
        AggregateExpr::Avg { arg, distinct } => format_args("avg", arg, *distinct),
        AggregateExpr::Min { arg } => format!("min({})", arg),
        AggregateExpr::Max { arg } => format!("max({})", arg),
        AggregateExpr::Collect { arg, distinct } => format_args("collect", arg, *distinct),
    }
}

fn format_args(name: &str, arg: &crate::parser::Expression, distinct: bool) -> String {
    if distinct {
        format!("{}(DISTINCT {})", name, arg)
    } else {
        format!("{}({})", name, arg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{
        Expression, ExpressionKind, Identifier, Literal, OrderDirection, RelationshipDirection,
        SourceSpan,
    };
    use crate::plan::logical::ShortestMode;
    use crate::plan::logical::{LogicalPlan, OrderKey, ProjectionItem};

    fn ident_expr(name: &str) -> Expression {
        Expression {
            kind: ExpressionKind::Variable(Identifier::new(name, SourceSpan::point(0))),
            span: SourceSpan::point(0),
        }
    }

    fn int(n: i64) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::Integer(n)),
            span: SourceSpan::point(0),
        }
    }

    #[test]
    fn explain_renders_scan_only() {
        let p = LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        assert_eq!(explain(&p), "NodeScan label=Person alias=a\n");
    }

    #[test]
    fn explain_renders_filter_over_scan() {
        let p = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("Person".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            predicate: int(1),
        };
        let s = explain(&p);
        assert_eq!(s, "Filter (1)\n NodeScan label=Person alias=a\n");
    }

    #[test]
    fn explain_renders_full_chain() {
        let scan = LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let expand = LogicalPlan::Expand {
            input: Box::new(scan),
            source: "a".into(),
            edge_type: Some(vec!["KNOWS".into()]),
            direction: RelationshipDirection::Right,
            rel_alias: Some("r".into()),
            target_alias: "b".into(),
            target_label: None,
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let project = LogicalPlan::Project {
            input: Box::new(expand),
            items: vec![ProjectionItem {
                expression: ident_expr("b"),
                alias: "b".into(),
            }],
            distinct: false,
            discard_input_bindings: true,
        };
        let topn = LogicalPlan::TopN {
            input: Box::new(project),
            keys: vec![OrderKey {
                expression: ident_expr("b"),
                direction: OrderDirection::Desc,
            }],
            skip: 0,
            limit: 10,
        };
        let out = explain(&topn);
        let expected = "\
TopN keys=[b DESC] limit=10
 Project [b=b]
  Expand source=a edge_type=KNOWS dir=-> rel=r target=b
   NodeScan label=Person alias=a
";
        assert_eq!(out, expected);
    }

    #[test]
    fn explain_renders_optional_expand() {
        let p = LogicalPlan::Expand {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("P".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            source: "a".into(),
            edge_type: Some(vec!["KNOWS".into()]),
            direction: RelationshipDirection::Both,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: None,
            length: None,
            optional: true,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let s = explain(&p);
        assert!(s.starts_with("OptionalExpand"));
    }

    fn micro_catalog() -> StatsCatalog {
        use crate::cost::stats::{EdgeTypeStats, LabelStats, PropStats};
        use namidb_storage::sst::stats::StatScalar;
        let mut cat = StatsCatalog::empty();
        let age = PropStats {
            null_count: 0,
            non_null_count: 100,
            min: Some(StatScalar::Int64(18)),
            max: Some(StatScalar::Int64(99)),
            ndv: Some(50),
            unique: false,
        };
        let mut props = std::collections::BTreeMap::new();
        props.insert("age".into(), age);
        cat.__test_insert_label(LabelStats {
            name: "Person".into(),
            node_count: 100,
            properties: props,
        });
        cat.__test_insert_edge_type(EdgeTypeStats {
            name: "KNOWS".into(),
            edge_count: 500,
            avg_out_degree: 5.0,
            avg_in_degree: 5.0,
            max_out_degree: 20,
            max_in_degree: 20,
            src_label: Some("Person".into()),
            dst_label: Some("Person".into()),
        });
        cat
    }

    #[test]
    fn explain_verbose_emits_estimates() {
        let cat = micro_catalog();
        let plan = LogicalPlan::Expand {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("Person".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            source: "a".into(),
            edge_type: Some(vec!["KNOWS".into()]),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: Some("Person".into()),
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let s = explain_verbose(&plan, &cat);
        assert!(s.starts_with("# Estimated rows: 500"));
        assert!(s.contains("Expand "));
        assert!(s.contains("(est=500)"));
        assert!(s.contains("NodeScan"));
        assert!(s.contains("(est=100)"));
        // Total work = 500 (expand) + 100 (scan) = 600.
        assert!(s.contains("# Estimated total work: 600"));
    }

    #[test]
    fn explain_verbose_marks_missing_stats() {
        let cat = StatsCatalog::empty();
        let plan = LogicalPlan::NodeScan {
            label: Some("Unknown".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let s = explain_verbose(&plan, &cat);
        assert!(s.contains("(est=0, no stats)"));
    }

    #[test]
    fn explain_verbose_promotes_subunit_to_one_row() {
        // A predicate that filters to < 1 row should still render as
        // est=1 rather than est=0.
        let mut cat = micro_catalog();
        // Make the eq even narrower: ndv=1000 → 0.001 → 100 * 0.001 = 0.1.
        let mut props = std::collections::BTreeMap::new();
        let age = crate::cost::stats::PropStats {
            null_count: 0,
            non_null_count: 100,
            min: None,
            max: None,
            ndv: Some(1000),
            unique: false,
        };
        props.insert("age".into(), age);
        cat.__test_insert_label(crate::cost::stats::LabelStats {
            name: "Person".into(),
            node_count: 100,
            properties: props,
        });
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("Person".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            predicate: Expression {
                kind: crate::parser::ast::ExpressionKind::Binary {
                    op: crate::parser::ast::BinaryOp::Eq,
                    left: Box::new(Expression {
                        kind: crate::parser::ast::ExpressionKind::Property(Box::new(
                            crate::parser::ast::PropertyAccess {
                                target: ident_expr("a"),
                                key: crate::parser::ast::Identifier::new(
                                    "age",
                                    SourceSpan::point(0),
                                ),
                                span: SourceSpan::point(0),
                            },
                        )),
                        span: SourceSpan::point(0),
                    }),
                    right: Box::new(int(30)),
                },
                span: SourceSpan::point(0),
            },
        };
        let s = explain_verbose(&plan, &cat);
        // 100 * 0.001 = 0.1 → format_rows promotes to 1.
        assert!(s.contains("Filter "));
        assert!(s.contains("(est=1)"));
    }

    #[test]
    fn explain_query_verbose_lowers_and_renders() {
        let cat = micro_catalog();
        let query = crate::parser::parse("MATCH (a:Person) RETURN a").unwrap();
        let s = explain_query_verbose(&query, &cat).unwrap();
        assert!(s.contains("# Estimated rows:"));
        assert!(s.contains("NodeScan"));
    }

    #[test]
    fn explain_renders_aggregate() {
        use crate::plan::logical::AggregateExpr;
        let p = LogicalPlan::Aggregate {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("P".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            group_by: vec![(ident_expr("a"), "a".into())],
            aggregations: vec![(
                "n".into(),
                AggregateExpr::Count {
                    arg: None,
                    distinct: false,
                },
            )],
        };
        let s = explain(&p);
        assert!(s.contains("Aggregate"));
        assert!(s.contains("count(*)"));
    }

    #[test]
    fn explain_tree_returns_structured_nodes() {
        let scan = LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let expand = LogicalPlan::Expand {
            input: Box::new(scan),
            source: "a".into(),
            edge_type: Some(vec!["KNOWS".into()]),
            direction: RelationshipDirection::Right,
            rel_alias: Some("r".into()),
            target_alias: "b".into(),
            target_label: None,
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let tree = explain_tree(&expand);
        assert_eq!(
            tree.summary,
            "Expand source=a edge_type=KNOWS dir=-> rel=r target=b"
        );
        assert_eq!(tree.estimated_rows, None);
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].summary, "NodeScan label=Person alias=a");
        assert!(tree.children[0].children.is_empty());
    }

    #[test]
    fn explain_tree_verbose_carries_estimates() {
        let cat = micro_catalog();
        let plan = LogicalPlan::Expand {
            input: Box::new(LogicalPlan::NodeScan {
                label: Some("Person".into()),
                alias: "a".into(),
                predicates: vec![],
                projection: None,
            }),
            source: "a".into(),
            edge_type: Some(vec!["KNOWS".into()]),
            direction: RelationshipDirection::Right,
            rel_alias: None,
            target_alias: "b".into(),
            target_label: Some("Person".into()),
            length: None,
            optional: false,
            back_reference: false,
            shortest: ShortestMode::None,
            path_binding: None,
        };
        let tree = explain_tree_verbose(&plan, &cat);
        assert_eq!(tree.estimated_rows, Some(500));
        assert_eq!(tree.estimated_total_work, Some(600));
        assert_eq!(tree.no_stats, None);
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].estimated_rows, Some(100));
        // Only the root carries total_work.
        assert_eq!(tree.children[0].estimated_total_work, None);
    }

    #[test]
    fn explain_tree_verbose_marks_missing_stats() {
        let cat = StatsCatalog::empty();
        let plan = LogicalPlan::NodeScan {
            label: Some("Unknown".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let tree = explain_tree_verbose(&plan, &cat);
        assert_eq!(tree.no_stats, Some(true));
    }

    #[test]
    fn explain_query_tree_lowers_and_returns_root() {
        let q = crate::parser::parse("MATCH (a:Person) RETURN a").unwrap();
        let tree = explain_query_tree(&q).unwrap();
        assert!(tree.summary.contains("Project"));
    }
}
