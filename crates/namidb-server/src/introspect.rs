//! Memgraph-flavoured schema introspection for Bolt GUI clients.
//!
//! NamiDB's Cypher dialect has no `CALL`/`SHOW` clause, so the schema
//! procedures a graph GUI fires on connect would be rejected at the
//! parser with a `SyntaxError` before they ever reach the engine. A
//! Neo4j/Memgraph-compatible client (e.g. G.V()/gdotv) needs those to
//! answer, or its schema sidebar stays empty.
//!
//! Rather than grow a procedure subsystem in the query crate, we
//! intercept the handful of introspection strings here, on the Bolt
//! boundary, and synthesise the answer from the live [`Snapshot`]
//! (memtable + SSTs, so freshly-written data shows up without a flush).
//!
//! The exact set is the queries G.V()/gdotv issues for a **Memgraph**
//! connection, read verbatim from its bundled `neo4j-java-driver`
//! backend:
//!
//! ```text
//! CALL meta_util.schema() YIELD *;
//! CALL schema.node_type_properties() YIELD *
//! CALL schema.rel_type_properties() YIELD *
//! CALL mg.procedures() YIELD name, signature
//! CALL mg.functions() YIELD name, signature
//! ```
//!
//! Anything else returns `None` and falls through to the real parser.
//!
//! Property type detail is best-effort: declared/persisted property
//! types come from the manifest cheaply, and ad-hoc (schemaless) ones
//! are picked up by sampling live nodes. Edge-property sampling and a
//! bounded (non-materialising) node scan are deliberate follow-ups; see
//! the notes at [`sample_label`].

use std::collections::BTreeMap;

use namidb_bolt::{BackendError, RunOutcome, StatementType};
use namidb_core::{DataType, Value};
use namidb_query::{Row, RuntimeValue};
use namidb_storage::Snapshot;
use tracing::warn;

/// How many nodes per label we look at when sampling ad-hoc property
/// names. The scan itself still walks the label (see follow-up note);
/// this only caps how many rows we inspect for property keys.
const SAMPLE_NODES: usize = 256;

/// How many edges per type we inspect for relationship property names.
const SAMPLE_EDGES: usize = 256;

/// Intercept a Memgraph-style introspection query. Returns `None` for
/// anything that isn't one of the known procedures, in which case the
/// caller proceeds to the normal parse/plan/execute path.
pub async fn try_introspect(
    cypher: &str,
    snap: &Snapshot<'_>,
) -> Option<Result<RunOutcome, BackendError>> {
    let norm = normalize(cypher);
    let outcome = if norm.starts_with("call meta_util.schema(") {
        // Memgraph connection type.
        meta_util_schema(snap).await
    } else if norm.starts_with("call schema.node_type_properties(") {
        node_type_properties(snap).await
    } else if norm.starts_with("call schema.rel_type_properties(") {
        rel_type_properties(snap).await
    } else if norm.starts_with("call mg.procedures(") {
        mg_procedures()
    } else if norm.starts_with("call mg.functions(") {
        mg_functions()
    } else if norm.starts_with("call db.labels(") {
        // Neo4j connection type (db.*, apoc.meta.*, SHOW, dbms.*).
        db_labels(snap)
    } else if norm.starts_with("call db.relationshiptypes(") {
        db_relationship_types(snap)
    } else if norm.starts_with("call db.propertykeys(") {
        db_property_keys(snap).await
    } else if norm.starts_with("call apoc.meta.nodetypeproperties(") {
        apoc_node_type_properties(snap).await
    } else if norm.starts_with("call apoc.meta.reltypeproperties(") {
        apoc_rel_type_properties(snap).await
    } else if norm.starts_with("call dbms.components(") {
        dbms_components()
    } else if norm.starts_with("show procedures") || norm.starts_with("call dbms.procedures(") {
        show_procedures()
    } else if norm.starts_with("show functions") || norm.starts_with("call dbms.functions(") {
        show_functions()
    } else if norm.starts_with("show databases") {
        show_databases()
    } else {
        return None;
    };
    Some(Ok(outcome))
}

/// Collapse whitespace, drop a trailing `;`, and lowercase so the
/// match is robust to the small formatting differences between clients
/// (`YIELD *` vs not, trailing semicolon, extra spaces).
fn normalize(s: &str) -> String {
    let trimmed = s.trim().trim_end_matches(';').trim();
    trimmed
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// `CALL meta_util.schema() YIELD *` — the primary schema source for a
/// Memgraph client. Returns a single row, single column whose value is
/// a map `{nodes: [...], relationships: [...]}`. Each node is a map
/// `{id, labels, count, properties}` and each relationship a map
/// `{id, start, end, label, count, properties}`, with `start`/`end`
/// referencing node `id`s — the exact shape G.V() deserialises.
async fn meta_util_schema(snap: &Snapshot<'_>) -> RunOutcome {
    let labels = snap.observed_labels();

    // Stable id per label so relationship endpoints can reference them.
    let mut label_id: BTreeMap<String, i64> = BTreeMap::new();
    let mut nodes: Vec<RuntimeValue> = Vec::with_capacity(labels.len());
    for (i, label) in labels.iter().enumerate() {
        let id = i as i64;
        label_id.insert(label.clone(), id);
        let (count, props) = sample_label(snap, label).await;
        let mut node = BTreeMap::new();
        node.insert("id".to_string(), RuntimeValue::Integer(id));
        node.insert(
            "labels".to_string(),
            RuntimeValue::List(vec![RuntimeValue::String(label.clone())]),
        );
        node.insert("count".to_string(), RuntimeValue::Integer(count));
        node.insert("properties".to_string(), props_to_map(props));
        nodes.push(RuntimeValue::Map(node));
    }

    let mut rels: Vec<RuntimeValue> = Vec::new();
    match snap.observed_edge_endpoints().await {
        Ok(endpoints) => {
            for (j, ep) in endpoints.iter().enumerate() {
                let start = ep.src_label.as_ref().and_then(|l| label_id.get(l)).copied();
                let end = ep.dst_label.as_ref().and_then(|l| label_id.get(l)).copied();
                // Keep start/end pointing at a node that exists in the
                // list above; an unresolved endpoint (edge type created
                // without inferable labels) falls back to node id 0
                // rather than dangling, which G.V() can't resolve.
                let mut rel = BTreeMap::new();
                rel.insert("id".to_string(), RuntimeValue::Integer(j as i64));
                rel.insert(
                    "start".to_string(),
                    RuntimeValue::Integer(start.unwrap_or(0)),
                );
                rel.insert("end".to_string(), RuntimeValue::Integer(end.unwrap_or(0)));
                rel.insert(
                    "label".to_string(),
                    RuntimeValue::String(ep.edge_type.clone()),
                );
                rel.insert("count".to_string(), RuntimeValue::Integer(0));
                rel.insert("properties".to_string(), RuntimeValue::Map(BTreeMap::new()));
                rels.push(RuntimeValue::Map(rel));
            }
        }
        Err(e) => warn!(error = %e, "introspect: observed_edge_endpoints failed"),
    }

    let mut schema = BTreeMap::new();
    schema.insert("nodes".to_string(), RuntimeValue::List(nodes));
    schema.insert("relationships".to_string(), RuntimeValue::List(rels));

    let row = Row::new().with("schema", RuntimeValue::Map(schema));
    read_outcome(vec!["schema".to_string()], vec![row])
}

/// `CALL schema.node_type_properties()` — one row per (label, property)
/// with Memgraph's column set. G.V() reads `nodeLabels`, `propertyName`
/// and `propertyTypes`.
async fn node_type_properties(snap: &Snapshot<'_>) -> RunOutcome {
    let fields = vec![
        "nodeType".to_string(),
        "nodeLabels".to_string(),
        "mandatory".to_string(),
        "propertyName".to_string(),
        "propertyTypes".to_string(),
    ];
    let mut rows = Vec::new();
    for label in snap.observed_labels() {
        let (_count, props) = sample_label(snap, &label).await;
        let node_type = format!(":`{}`", label);
        let node_labels = RuntimeValue::List(vec![RuntimeValue::String(label.clone())]);
        if props.is_empty() {
            rows.push(node_prop_row(&node_type, node_labels, "", ""));
        } else {
            for (name, ty) in props {
                rows.push(node_prop_row(&node_type, node_labels.clone(), &name, &ty));
            }
        }
    }
    read_outcome(fields, rows)
}

/// `CALL schema.rel_type_properties()` — one row per (edge type,
/// property), with property names and types sampled from live edges.
/// Edge types with no sampled properties still get one row so the type
/// shows up in the client's relationship list.
async fn rel_type_properties(snap: &Snapshot<'_>) -> RunOutcome {
    let fields = vec![
        "relType".to_string(),
        "mandatory".to_string(),
        "propertyName".to_string(),
        "propertyTypes".to_string(),
    ];
    let mut rows = Vec::new();
    for edge_type in snap.observed_edge_types() {
        let rel_type = format!(":`{}`", edge_type);
        let props = sample_edge_type(snap, &edge_type).await;
        if props.is_empty() {
            rows.push(rel_prop_row(&rel_type, "", ""));
        } else {
            for (name, ty) in props {
                rows.push(rel_prop_row(&rel_type, &name, &ty));
            }
        }
    }
    read_outcome(fields, rows)
}

fn rel_prop_row(rel_type: &str, prop_name: &str, prop_type: &str) -> Row {
    Row::new()
        .with("relType", RuntimeValue::String(rel_type.to_string()))
        .with("mandatory", RuntimeValue::Bool(false))
        .with("propertyName", RuntimeValue::String(prop_name.to_string()))
        .with("propertyTypes", RuntimeValue::String(prop_type.to_string()))
}

// ── Neo4j connection type ────────────────────────────────────────────

/// `CALL db.labels()` — one `label` column per node label.
fn db_labels(snap: &Snapshot<'_>) -> RunOutcome {
    let rows = snap
        .observed_labels()
        .into_iter()
        .map(|l| Row::new().with("label", RuntimeValue::String(l)))
        .collect();
    read_outcome(vec!["label".to_string()], rows)
}

/// `CALL db.relationshipTypes()` — one `relationshipType` column per type.
fn db_relationship_types(snap: &Snapshot<'_>) -> RunOutcome {
    let rows = snap
        .observed_edge_types()
        .into_iter()
        .map(|t| Row::new().with("relationshipType", RuntimeValue::String(t)))
        .collect();
    read_outcome(vec!["relationshipType".to_string()], rows)
}

/// `CALL db.propertyKeys()` — distinct property keys across all labels.
async fn db_property_keys(snap: &Snapshot<'_>) -> RunOutcome {
    let mut keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for label in snap.observed_labels() {
        let (_count, props) = sample_label(snap, &label).await;
        keys.extend(props.into_keys());
    }
    let rows = keys
        .into_iter()
        .map(|k| Row::new().with("propertyKey", RuntimeValue::String(k)))
        .collect();
    read_outcome(vec!["propertyKey".to_string()], rows)
}

/// `CALL dbms.components()` — product name, version, edition. Drivers
/// read the version from the Bolt `HELLO` already; this is for display.
fn dbms_components() -> RunOutcome {
    let row = Row::new()
        .with("name", RuntimeValue::String("Neo4j Kernel".to_string()))
        .with(
            "versions",
            RuntimeValue::List(vec![RuntimeValue::String("5.13.0".to_string())]),
        )
        .with("edition", RuntimeValue::String("community".to_string()));
    read_outcome(
        vec![
            "name".to_string(),
            "versions".to_string(),
            "edition".to_string(),
        ],
        vec![row],
    )
}

/// `SHOW PROCEDURES` / `CALL dbms.procedures()` — the procedures this
/// shim answers, in Neo4j's column shape.
fn show_procedures() -> RunOutcome {
    let procs = [
        ("db.labels", "db.labels() :: (label :: STRING)"),
        (
            "db.relationshipTypes",
            "db.relationshipTypes() :: (relationshipType :: STRING)",
        ),
        ("db.propertyKeys", "db.propertyKeys() :: (propertyKey :: STRING)"),
        (
            "apoc.meta.nodeTypeProperties",
            "apoc.meta.nodeTypeProperties() :: (nodeType :: STRING, nodeLabels :: LIST, propertyName :: STRING, propertyTypes :: LIST, mandatory :: BOOLEAN)",
        ),
        (
            "apoc.meta.relTypeProperties",
            "apoc.meta.relTypeProperties() :: (relType :: STRING, sourceNodeLabels :: LIST, targetNodeLabels :: LIST, propertyName :: STRING, propertyTypes :: LIST, mandatory :: BOOLEAN)",
        ),
    ];
    let rows = procs
        .iter()
        .map(|(name, sig)| {
            Row::new()
                .with("name", RuntimeValue::String((*name).to_string()))
                .with("description", RuntimeValue::String(String::new()))
                .with("signature", RuntimeValue::String((*sig).to_string()))
        })
        .collect();
    read_outcome(
        vec![
            "name".to_string(),
            "description".to_string(),
            "signature".to_string(),
        ],
        rows,
    )
}

/// `SHOW FUNCTIONS` / `CALL dbms.functions()` — no user functions to
/// advertise; the column shape matters more than the rows.
fn show_functions() -> RunOutcome {
    read_outcome(
        vec![
            "name".to_string(),
            "description".to_string(),
            "signature".to_string(),
        ],
        Vec::new(),
    )
}

/// `SHOW DATABASES` — NamiDB serves a single namespace, presented as the
/// default `neo4j` database so the client's database picker resolves.
fn show_databases() -> RunOutcome {
    let row = Row::new()
        .with("name", RuntimeValue::String("neo4j".to_string()))
        .with("type", RuntimeValue::String("standard".to_string()))
        .with("access", RuntimeValue::String("read-write".to_string()))
        .with(
            "address",
            RuntimeValue::String("localhost:7687".to_string()),
        )
        .with("role", RuntimeValue::String("primary".to_string()))
        .with(
            "requestedStatus",
            RuntimeValue::String("online".to_string()),
        )
        .with("currentStatus", RuntimeValue::String("online".to_string()))
        .with("default", RuntimeValue::Bool(true))
        .with("home", RuntimeValue::Bool(true));
    read_outcome(
        vec![
            "name".to_string(),
            "type".to_string(),
            "access".to_string(),
            "address".to_string(),
            "role".to_string(),
            "requestedStatus".to_string(),
            "currentStatus".to_string(),
            "default".to_string(),
            "home".to_string(),
        ],
        vec![row],
    )
}

/// `CALL apoc.meta.nodeTypeProperties()` — Neo4j/APOC node schema. Same
/// data as `schema.node_type_properties` but APOC's column shape, where
/// `propertyTypes` is a list. The `{includeLabels: [$label]}` variant is
/// answered with the full set; the client keys on `nodeLabels`.
async fn apoc_node_type_properties(snap: &Snapshot<'_>) -> RunOutcome {
    let fields = vec![
        "nodeType".to_string(),
        "nodeLabels".to_string(),
        "propertyName".to_string(),
        "propertyTypes".to_string(),
        "mandatory".to_string(),
    ];
    let mut rows = Vec::new();
    for label in snap.observed_labels() {
        let (_count, props) = sample_label(snap, &label).await;
        let node_type = format!(":`{}`", label);
        let node_labels = vec![RuntimeValue::String(label.clone())];
        if props.is_empty() {
            rows.push(apoc_node_row(&node_type, &node_labels, "", None));
        } else {
            for (name, ty) in props {
                rows.push(apoc_node_row(&node_type, &node_labels, &name, Some(&ty)));
            }
        }
    }
    read_outcome(fields, rows)
}

fn apoc_node_row(
    node_type: &str,
    node_labels: &[RuntimeValue],
    prop_name: &str,
    prop_type: Option<&str>,
) -> Row {
    Row::new()
        .with("nodeType", RuntimeValue::String(node_type.to_string()))
        .with("nodeLabels", RuntimeValue::List(node_labels.to_vec()))
        .with("propertyName", RuntimeValue::String(prop_name.to_string()))
        .with("propertyTypes", type_list(prop_type))
        .with("mandatory", RuntimeValue::Bool(false))
}

/// `CALL apoc.meta.relTypeProperties()` — Neo4j/APOC relationship schema
/// with endpoint labels and sampled property types.
async fn apoc_rel_type_properties(snap: &Snapshot<'_>) -> RunOutcome {
    let fields = vec![
        "relType".to_string(),
        "sourceNodeLabels".to_string(),
        "targetNodeLabels".to_string(),
        "propertyName".to_string(),
        "propertyTypes".to_string(),
        "mandatory".to_string(),
    ];
    let endpoints = edge_endpoints(snap).await;
    let mut rows = Vec::new();
    for edge_type in snap.observed_edge_types() {
        let rel_type = format!(":`{}`", edge_type);
        let (src, dst) = endpoints.get(&edge_type).cloned().unwrap_or((None, None));
        let src_labels = label_list(src);
        let dst_labels = label_list(dst);
        let props = sample_edge_type(snap, &edge_type).await;
        if props.is_empty() {
            rows.push(apoc_rel_row(&rel_type, &src_labels, &dst_labels, "", None));
        } else {
            for (name, ty) in props {
                rows.push(apoc_rel_row(
                    &rel_type,
                    &src_labels,
                    &dst_labels,
                    &name,
                    Some(&ty),
                ));
            }
        }
    }
    read_outcome(fields, rows)
}

fn apoc_rel_row(
    rel_type: &str,
    src_labels: &[RuntimeValue],
    dst_labels: &[RuntimeValue],
    prop_name: &str,
    prop_type: Option<&str>,
) -> Row {
    Row::new()
        .with("relType", RuntimeValue::String(rel_type.to_string()))
        .with("sourceNodeLabels", RuntimeValue::List(src_labels.to_vec()))
        .with("targetNodeLabels", RuntimeValue::List(dst_labels.to_vec()))
        .with("propertyName", RuntimeValue::String(prop_name.to_string()))
        .with("propertyTypes", type_list(prop_type))
        .with("mandatory", RuntimeValue::Bool(false))
}

fn type_list(prop_type: Option<&str>) -> RuntimeValue {
    match prop_type {
        Some(t) => RuntimeValue::List(vec![RuntimeValue::String(t.to_string())]),
        None => RuntimeValue::List(Vec::new()),
    }
}

fn label_list(label: Option<String>) -> Vec<RuntimeValue> {
    label
        .map(|l| vec![RuntimeValue::String(l)])
        .unwrap_or_default()
}

/// `CALL mg.procedures()` — advertise the procedures this shim answers
/// so the client's autocomplete/catalog isn't empty.
fn mg_procedures() -> RunOutcome {
    let procs = [
        ("meta_util.schema", "meta_util.schema() :: (schema :: MAP)"),
        (
            "schema.node_type_properties",
            "schema.node_type_properties() :: (nodeType :: STRING, nodeLabels :: LIST, mandatory :: BOOLEAN, propertyName :: STRING, propertyTypes :: STRING)",
        ),
        (
            "schema.rel_type_properties",
            "schema.rel_type_properties() :: (relType :: STRING, mandatory :: BOOLEAN, propertyName :: STRING, propertyTypes :: STRING)",
        ),
        ("mg.procedures", "mg.procedures() :: (name :: STRING, signature :: STRING)"),
        ("mg.functions", "mg.functions() :: (name :: STRING, signature :: STRING)"),
    ];
    let rows = procs
        .iter()
        .map(|(name, sig)| {
            Row::new()
                .with("name", RuntimeValue::String((*name).to_string()))
                .with("signature", RuntimeValue::String((*sig).to_string()))
        })
        .collect();
    read_outcome(vec!["name".to_string(), "signature".to_string()], rows)
}

/// `CALL mg.functions()` — no user-defined functions to advertise yet.
fn mg_functions() -> RunOutcome {
    read_outcome(
        vec!["name".to_string(), "signature".to_string()],
        Vec::new(),
    )
}

// --- helpers ---------------------------------------------------------

/// Label property names mapped to a Memgraph-style type label. Declared
/// and persisted types come from the manifest for free; ad-hoc
/// (schemaless) properties that only live in `__overflow_json` are not
/// typed there, so we also sample live nodes to surface them.
///
/// Follow-up: `scan_label` materialises the whole label. Introspection
/// runs rarely, but on a large undeclared label this is a full scan per
/// schema refresh. A bounded streaming scan (stop after `SAMPLE_NODES`)
/// would cap it; for declared/flushed schemas the manifest path alone is
/// enough and the sample mostly adds nothing.
async fn sample_label(snap: &Snapshot<'_>, label: &str) -> (i64, BTreeMap<String, String>) {
    let mut props: BTreeMap<String, String> = snap
        .observed_property_types_for_label(label)
        .into_iter()
        .map(|(name, dt)| (name, datatype_name(&dt).to_string()))
        .collect();

    let mut count: i64 = 0;
    match snap.scan_label(label).await {
        Ok(views) => {
            count = views.len() as i64;
            for view in views.iter().take(SAMPLE_NODES) {
                for (name, value) in &view.properties {
                    if matches!(value, Value::Null) {
                        continue;
                    }
                    props
                        .entry(name.clone())
                        .or_insert_with(|| value_type_name(value).to_string());
                }
            }
        }
        Err(e) => warn!(error = %e, label, "introspect: scan_label failed"),
    }
    (count, props)
}

/// Sampled property names of an edge type, mapped to a type label.
async fn sample_edge_type(snap: &Snapshot<'_>, edge_type: &str) -> BTreeMap<String, String> {
    let mut props: BTreeMap<String, String> = BTreeMap::new();
    match snap.scan_edge_type(edge_type).await {
        Ok(edges) => {
            for edge in edges.iter().take(SAMPLE_EDGES) {
                for (name, value) in &edge.properties {
                    if matches!(value, Value::Null) {
                        continue;
                    }
                    props
                        .entry(name.clone())
                        .or_insert_with(|| value_type_name(value).to_string());
                }
            }
        }
        Err(e) => warn!(error = %e, edge_type, "introspect: scan_edge_type failed"),
    }
    props
}

/// Inferred endpoint labels per edge type (`edge_type -> (src, dst)`).
async fn edge_endpoints(snap: &Snapshot<'_>) -> BTreeMap<String, (Option<String>, Option<String>)> {
    let mut map = BTreeMap::new();
    match snap.observed_edge_endpoints().await {
        Ok(endpoints) => {
            for ep in endpoints {
                map.insert(ep.edge_type, (ep.src_label, ep.dst_label));
            }
        }
        Err(e) => warn!(error = %e, "introspect: observed_edge_endpoints failed"),
    }
    map
}

fn props_to_map(props: BTreeMap<String, String>) -> RuntimeValue {
    RuntimeValue::Map(
        props
            .into_iter()
            .map(|(name, ty)| (name, RuntimeValue::String(ty)))
            .collect(),
    )
}

fn node_prop_row(
    node_type: &str,
    node_labels: RuntimeValue,
    prop_name: &str,
    prop_type: &str,
) -> Row {
    Row::new()
        .with("nodeType", RuntimeValue::String(node_type.to_string()))
        .with("nodeLabels", node_labels)
        .with("mandatory", RuntimeValue::Bool(false))
        .with("propertyName", RuntimeValue::String(prop_name.to_string()))
        .with("propertyTypes", RuntimeValue::String(prop_type.to_string()))
}

fn read_outcome(fields: Vec<String>, rows: Vec<Row>) -> RunOutcome {
    RunOutcome {
        fields,
        rows,
        statement_type: StatementType::Read,
        counters: Default::default(),
    }
}

/// Manifest-declared/persisted Arrow type → Memgraph-ish type label.
fn datatype_name(dt: &DataType) -> &'static str {
    match dt {
        DataType::Bool => "Bool",
        DataType::Int32 | DataType::Int64 => "Int",
        DataType::Float32 | DataType::Float64 => "Float",
        DataType::Utf8 | DataType::LargeUtf8 => "String",
        DataType::Binary => "String",
        DataType::Date32 => "Date",
        DataType::TimestampMicrosUtc => "LocalDateTime",
        DataType::FloatVector { .. } => "List",
        DataType::Json => "Map",
    }
}

/// Sampled runtime value → Memgraph-ish type label.
fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "Null",
        Value::Bool(_) => "Bool",
        Value::I64(_) => "Int",
        Value::F64(_) => "Float",
        Value::Str(_) => "String",
        Value::Bytes(_) => "String",
        Value::Vec(_) => "List",
        Value::Date(_) => "Date",
        Value::DateTime(_) => "LocalDateTime",
        Value::List(_) => "List",
        _ => "Map",
    }
}
