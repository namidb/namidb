//! Row builders for `SHOW CONSTRAINTS` / `SHOW INDEXES`.
//!
//! These schema-introspection commands are intercepted out-of-band by the
//! server (and by the embedded Python client) — they never lower to a
//! `LogicalPlan`. The functions here turn the manifest schema into the same
//! [`Row`] shape an ordinary read produces, so the existing result-serialisation
//! paths (HTTP JSON, Bolt records, Python `QueryResult`) carry them unchanged.

use namidb_core::schema::{Constraint, ConstraintKind, Schema};
use namidb_storage::Manifest;

use super::row::Row;
use super::value::RuntimeValue;

/// Column order for both `SHOW` statements (a Neo4j-compatible subset).
pub fn show_schema_columns() -> Vec<String> {
    ["name", "type", "entityType", "labelsOrTypes", "properties"]
        .into_iter()
        .map(String::from)
        .collect()
}

fn str_list(items: &[String]) -> RuntimeValue {
    RuntimeValue::List(items.iter().cloned().map(RuntimeValue::String).collect())
}

fn schema_row(name: String, kind: &str, label: String, properties: &[String]) -> Row {
    Row::new()
        .with("name", RuntimeValue::String(name))
        .with("type", RuntimeValue::String(kind.to_string()))
        .with("entityType", RuntimeValue::String("NODE".into()))
        .with("labelsOrTypes", str_list(&[label]))
        .with("properties", str_list(properties))
}

/// Rows for `SHOW CONSTRAINTS`, built from the manifest schema. Lists every
/// named constraint plus a synthesized row for any legacy single-property
/// `unique` flag that predates named constraints, so older manifests still
/// report. Sorted by name for a stable result order.
pub fn show_constraints_rows(schema: &Schema) -> Vec<Row> {
    let mut entries: Vec<(String, &'static str, String, Vec<String>)> = Vec::new();
    for c in schema.constraints() {
        entries.push((
            c.name.clone(),
            c.kind.as_str(),
            c.label.clone(),
            c.properties.clone(),
        ));
    }
    for (label, def) in &schema.labels {
        for p in &def.properties {
            if p.unique {
                let props = vec![p.name.clone()];
                if schema
                    .constraint_matching(label, &props, ConstraintKind::Unique)
                    .is_none()
                {
                    let name = Constraint::default_name(label, &props, ConstraintKind::Unique);
                    entries.push((name, ConstraintKind::Unique.as_str(), label.clone(), props));
                }
            }
        }
    }
    entries.sort();
    entries
        .into_iter()
        .map(|(name, kind, label, props)| schema_row(name, kind, label, &props))
        .collect()
}

/// Rows for `SHOW INDEXES`, built from the full manifest: equality (property)
/// indexes from the schema, plus the named vector and full-text descriptors.
/// Sorted by name for a stable result order.
pub fn show_indexes_rows(manifest: &Manifest) -> Vec<Row> {
    let mut entries: Vec<(String, &'static str, String, Vec<String>)> = Vec::new();
    for (label, def) in &manifest.schema.labels {
        for p in &def.properties {
            if p.indexed {
                let name = format!("index_{label}_{}", p.name);
                entries.push((name, "RANGE", label.clone(), vec![p.name.clone()]));
            }
        }
    }
    for vi in &manifest.vector_indexes {
        entries.push((
            vi.name.clone(),
            "VECTOR",
            vi.label.clone(),
            vec![vi.property.clone()],
        ));
    }
    for ti in &manifest.text_indexes {
        entries.push((
            ti.name.clone(),
            "FULLTEXT",
            ti.label.clone(),
            ti.properties.clone(),
        ));
    }
    entries.sort();
    entries
        .into_iter()
        .map(|(name, kind, label, props)| schema_row(name, kind, label, &props))
        .collect()
}
