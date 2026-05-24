//! Conversions between [`namidb_query::RuntimeValue`] and Bolt
//! [`Value`].
//!
//! Bolt has slightly different value model than the executor:
//!
//! - Bolt has no `Vector` type. We encode `Vector(Vec<f32>)` as a
//!   `List<Float>` so drivers receive plain numbers.
//! - Bolt represents `Node` / `Relationship` / `Path` as PackStream
//!   structs with positional fields. The structs change shape between
//!   v4 and v5 (v5 adds `element_id`), so the conversion takes a
//!   `WithElementId` flag.
//! - NodeIds are UUIDv7 (`u128`) on our side but `i64` on Bolt's
//!   legacy `id` field. We pack the low 64 bits of the UUID into the
//!   `id` slot and carry the full UUID as the `element_id` string.

use std::collections::BTreeMap;

use namidb_query::exec::{NodeValue, RelValue};
use namidb_query::RuntimeValue;

use crate::value::{struct_tag, Node, Relationship, Value};

/// Whether the negotiated Bolt version carries `element_id` fields on
/// node / relationship structs (v5.0+).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementIdMode {
    /// Bolt 4.x — no `element_id` field; legacy `id` only.
    None,
    /// Bolt 5.x — include `element_id`, `start_element_id`,
    /// `end_element_id`.
    Include,
}

impl ElementIdMode {
    pub fn from_major(major: u8) -> Self {
        if major >= 5 {
            ElementIdMode::Include
        } else {
            ElementIdMode::None
        }
    }

    fn is_include(self) -> bool {
        matches!(self, ElementIdMode::Include)
    }
}

/// Convert a runtime value to a Bolt wire value.
pub fn runtime_to_bolt(v: &RuntimeValue, mode: ElementIdMode) -> Value {
    match v {
        RuntimeValue::Null => Value::Null,
        RuntimeValue::Bool(b) => Value::Bool(*b),
        RuntimeValue::Integer(n) => Value::Int(*n),
        RuntimeValue::Float(f) => Value::Float(*f),
        RuntimeValue::String(s) => Value::String(s.clone()),
        RuntimeValue::Bytes(b) => Value::Bytes(b.clone()),
        RuntimeValue::Vector(v) => Value::List(v.iter().map(|x| Value::Float(*x as f64)).collect()),
        RuntimeValue::List(items) => {
            Value::List(items.iter().map(|v| runtime_to_bolt(v, mode)).collect())
        }
        RuntimeValue::Map(m) => Value::Map(
            m.iter()
                .map(|(k, v)| (k.clone(), runtime_to_bolt(v, mode)))
                .collect(),
        ),
        RuntimeValue::Date(days) => Value::Struct {
            tag: struct_tag::DATE,
            fields: vec![Value::Int(*days as i64)],
        },
        RuntimeValue::DateTime(micros) => {
            // LocalDateTime (Bolt tag 0x64, 2 fields: seconds, nanos
            // since 1970-01-01T00:00:00, no zone). We split micros
            // into both. RuntimeValue::DateTime is UTC; the v5
            // DateTime (0x49) form would also work but carries an
            // extra tz offset we don't track today.
            let seconds = micros.div_euclid(1_000_000);
            let nanos = micros.rem_euclid(1_000_000) * 1_000;
            Value::Struct {
                tag: struct_tag::LOCAL_DATETIME,
                fields: vec![Value::Int(seconds), Value::Int(nanos)],
            }
        }
        RuntimeValue::Node(n) => node_to_bolt(n, mode),
        RuntimeValue::Rel(r) => rel_to_bolt(r, mode),
        RuntimeValue::Path(items) => path_to_bolt(items, mode),
    }
}

fn node_to_bolt(n: &NodeValue, mode: ElementIdMode) -> Value {
    let (legacy_id, element_id) = uuid_to_bolt_ids(n.id.0);
    let labels = vec![Value::String(n.label.clone())];
    let properties: BTreeMap<String, Value> = n
        .properties
        .iter()
        .map(|(k, v)| (k.clone(), runtime_to_bolt(v, mode)))
        .collect();
    let node = Node {
        id: legacy_id,
        labels: labels
            .into_iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            })
            .collect(),
        properties,
        element_id: Some(element_id),
    };
    node.to_struct(mode.is_include())
}

fn rel_to_bolt(r: &RelValue, mode: ElementIdMode) -> Value {
    let (id, element_id) = synthetic_edge_id(&r.src, &r.dst, &r.edge_type);
    let (start_id, start_element_id) = uuid_to_bolt_ids(r.src.0);
    let (end_id, end_element_id) = uuid_to_bolt_ids(r.dst.0);
    let properties: BTreeMap<String, Value> = r
        .properties
        .iter()
        .map(|(k, v)| (k.clone(), runtime_to_bolt(v, mode)))
        .collect();
    let rel = Relationship {
        id,
        start_id,
        end_id,
        rel_type: r.edge_type.clone(),
        properties,
        element_id: Some(element_id),
        start_element_id: Some(start_element_id),
        end_element_id: Some(end_element_id),
    };
    rel.to_struct(mode.is_include())
}

fn path_to_bolt(items: &[RuntimeValue], mode: ElementIdMode) -> Value {
    // Bolt Path struct (tag 0x50, 3 fields):
    //   { nodes:    [Node, ...],
    //     rels:     [UnboundRelationship, ...],
    //     indices:  [i64, ...]   (zig-zag of rel-index, node-index) }
    //
    // The executor's Path is a flat alternating sequence
    // `[Node, Rel, Node, Rel, ..., Node]`. We deduplicate nodes and
    // rels and emit indices into the dedup'd lists, following the
    // Bolt spec's positive-forward / negative-reverse convention.
    use std::collections::HashMap;

    let mut nodes: Vec<NodeValue> = Vec::new();
    let mut node_index: HashMap<namidb_core::id::NodeId, i64> = HashMap::new();
    let mut rels: Vec<RelValue> = Vec::new();
    let mut indices: Vec<i64> = Vec::new();

    let mut prev_node_id: Option<namidb_core::id::NodeId> = None;
    let mut pending_rel: Option<&RelValue> = None;
    for v in items {
        match v {
            RuntimeValue::Node(boxed) => {
                let node = boxed.as_ref();
                let idx = match node_index.get(&node.id) {
                    Some(&i) => i,
                    None => {
                        let i = nodes.len() as i64;
                        node_index.insert(node.id, i);
                        nodes.push(node.clone());
                        i
                    }
                };
                if let Some(rel) = pending_rel.take() {
                    let rel_idx = rels.len() as i64 + 1;
                    let forward = prev_node_id == Some(rel.src);
                    rels.push(rel.clone());
                    indices.push(if forward { rel_idx } else { -rel_idx });
                    indices.push(idx);
                }
                prev_node_id = Some(node.id);
            }
            RuntimeValue::Rel(boxed) => {
                pending_rel = Some(boxed.as_ref());
            }
            _ => {
                // Path carried something that is not a Node or Rel;
                // fall back to a plain List of converted values.
                return Value::List(items.iter().map(|v| runtime_to_bolt(v, mode)).collect());
            }
        }
    }

    let nodes_v = Value::List(nodes.iter().map(|n| node_to_bolt(n, mode)).collect());
    let rels_v = Value::List(rels.iter().map(|r| unbound_rel(r, mode)).collect());
    let indices_v = Value::List(indices.into_iter().map(Value::Int).collect());

    Value::Struct {
        tag: struct_tag::PATH,
        fields: vec![nodes_v, rels_v, indices_v],
    }
}

fn unbound_rel(r: &RelValue, mode: ElementIdMode) -> Value {
    // UnboundRelationship struct tag 0x72, v5 layout:
    //   { id, type, properties, element_id }
    let (id, element_id) = synthetic_edge_id(&r.src, &r.dst, &r.edge_type);
    let mut fields = vec![
        Value::Int(id),
        Value::String(r.edge_type.clone()),
        Value::Map(
            r.properties
                .iter()
                .map(|(k, v)| (k.clone(), runtime_to_bolt(v, mode)))
                .collect(),
        ),
    ];
    if mode.is_include() {
        fields.push(Value::String(element_id));
    }
    Value::Struct {
        tag: struct_tag::UNBOUND_RELATIONSHIP,
        fields,
    }
}

/// Convert a Bolt parameter value back into a runtime value. Used
/// when the client sends a parameter map with `RUN`.
pub fn bolt_to_runtime(v: &Value) -> RuntimeValue {
    match v {
        Value::Null => RuntimeValue::Null,
        Value::Bool(b) => RuntimeValue::Bool(*b),
        Value::Int(n) => RuntimeValue::Integer(*n),
        Value::Float(f) => RuntimeValue::Float(*f),
        Value::String(s) => RuntimeValue::String(s.clone()),
        Value::Bytes(b) => RuntimeValue::Bytes(b.clone()),
        Value::List(items) => RuntimeValue::List(items.iter().map(bolt_to_runtime).collect()),
        Value::Map(m) => RuntimeValue::Map(
            m.iter()
                .map(|(k, v)| (k.clone(), bolt_to_runtime(v)))
                .collect(),
        ),
        Value::Struct { tag, fields } => decode_struct_param(*tag, fields),
    }
}

fn decode_struct_param(tag: u8, fields: &[Value]) -> RuntimeValue {
    let seconds_nanos = || -> Option<(i64, i64)> {
        let seconds = match fields.first()? {
            Value::Int(n) => *n,
            _ => return None,
        };
        let nanos = match fields.get(1) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        Some((seconds, nanos))
    };
    match tag {
        struct_tag::DATE => match fields.first() {
            Some(Value::Int(days)) => RuntimeValue::Date(*days as i32),
            _ => fallback_struct_map(tag, fields),
        },
        struct_tag::LOCAL_DATETIME => match seconds_nanos() {
            Some((s, n)) => RuntimeValue::DateTime(s * 1_000_000 + n / 1_000),
            None => fallback_struct_map(tag, fields),
        },
        // DateTime (v5+, UTC seconds + nanos + tz_offset_seconds).
        // RuntimeValue::DateTime is UTC micros, so the offset is
        // informational only — we keep the UTC seconds verbatim.
        struct_tag::DATETIME => match seconds_nanos() {
            Some((s, n)) => RuntimeValue::DateTime(s * 1_000_000 + n / 1_000),
            None => fallback_struct_map(tag, fields),
        },
        // DateTime (v4 legacy, wall-clock seconds at the offset).
        // Convert to UTC by subtracting the offset.
        struct_tag::DATETIME_LEGACY => match (seconds_nanos(), fields.get(2)) {
            (Some((s, n)), Some(Value::Int(off))) => {
                RuntimeValue::DateTime((s - off) * 1_000_000 + n / 1_000)
            }
            (Some((s, n)), _) => RuntimeValue::DateTime(s * 1_000_000 + n / 1_000),
            _ => fallback_struct_map(tag, fields),
        },
        // DateTimeZoneId carries a zone-id string; we currently
        // collapse to UTC like the offset form.
        struct_tag::DATETIME_ZONE_ID => match seconds_nanos() {
            Some((s, n)) => RuntimeValue::DateTime(s * 1_000_000 + n / 1_000),
            None => fallback_struct_map(tag, fields),
        },
        _ => fallback_struct_map(tag, fields),
    }
}

fn fallback_struct_map(tag: u8, fields: &[Value]) -> RuntimeValue {
    let mut m = BTreeMap::new();
    m.insert("_bolt_struct_tag".into(), RuntimeValue::Integer(tag as i64));
    m.insert(
        "_bolt_struct_fields".into(),
        RuntimeValue::List(fields.iter().map(bolt_to_runtime).collect()),
    );
    RuntimeValue::Map(m)
}

/// Convert a Bolt parameter map into a `namidb_query::Params`.
pub fn params_from_bolt_map(m: &BTreeMap<String, Value>) -> namidb_query::Params {
    let mut params = namidb_query::Params::new();
    for (k, v) in m {
        params.insert(k.clone(), bolt_to_runtime(v));
    }
    params
}

/// Pack the low 64 bits of a UUIDv7 into an `i64` and the full UUID
/// string into the `element_id` slot.
fn uuid_to_bolt_ids(u: uuid::Uuid) -> (i64, String) {
    let bytes = u.as_bytes();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[8..16]);
    let low = i64::from_be_bytes(buf);
    (low, u.to_string())
}

/// Synthesise a stable edge identifier from `(src, edge_type, dst)`.
/// We currently lack a first-class EdgeId in the runtime so we hash
/// the components into the `id` slot and emit a structured
/// `element_id` for the v5 path.
fn synthetic_edge_id(
    src: &namidb_core::id::NodeId,
    dst: &namidb_core::id::NodeId,
    edge_type: &str,
) -> (i64, String) {
    use xxhash_rust::xxh3::xxh3_64;
    let mut buf = Vec::with_capacity(16 + 16 + edge_type.len());
    buf.extend_from_slice(src.as_bytes());
    buf.extend_from_slice(dst.as_bytes());
    buf.extend_from_slice(edge_type.as_bytes());
    let h = xxh3_64(&buf) as i64;
    let element_id = format!("{}-{}->{}", edge_type, src.0, dst.0);
    (h, element_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use namidb_core::id::NodeId;
    use uuid::Uuid;

    #[test]
    fn scalars_roundtrip() {
        let cases = vec![
            (RuntimeValue::Null, Value::Null),
            (RuntimeValue::Bool(true), Value::Bool(true)),
            (RuntimeValue::Integer(42), Value::Int(42)),
            (RuntimeValue::Float(1.5), Value::Float(1.5)),
            (RuntimeValue::String("x".into()), Value::String("x".into())),
        ];
        for (rt, bolt) in cases {
            assert_eq!(runtime_to_bolt(&rt, ElementIdMode::Include), bolt);
            assert_eq!(bolt_to_runtime(&bolt), rt);
        }
    }

    #[test]
    fn vector_becomes_list_of_floats() {
        let v = RuntimeValue::Vector(vec![0.1, 0.2, 0.3]);
        let b = runtime_to_bolt(&v, ElementIdMode::Include);
        match b {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                for item in items {
                    assert!(matches!(item, Value::Float(_)));
                }
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn node_carries_element_id_on_v5() {
        let u = Uuid::now_v7();
        let node = NodeValue {
            id: NodeId::from_uuid(u),
            label: "Person".into(),
            properties: {
                let mut m = BTreeMap::new();
                m.insert("name".into(), RuntimeValue::String("Alice".into()));
                m
            },
        };
        let rt = RuntimeValue::Node(Box::new(node));
        let bolt = runtime_to_bolt(&rt, ElementIdMode::Include);
        let (tag, fields) = match bolt {
            Value::Struct { tag, fields } => (tag, fields),
            other => panic!("expected struct, got {:?}", other),
        };
        assert_eq!(tag, struct_tag::NODE);
        assert_eq!(fields.len(), 4); // id, labels, properties, element_id
        assert_eq!(fields[3], Value::String(u.to_string()));
    }

    #[test]
    fn node_omits_element_id_on_v4() {
        let u = Uuid::now_v7();
        let node = NodeValue {
            id: NodeId::from_uuid(u),
            label: "Person".into(),
            properties: BTreeMap::new(),
        };
        let rt = RuntimeValue::Node(Box::new(node));
        let bolt = runtime_to_bolt(&rt, ElementIdMode::None);
        let fields = match bolt {
            Value::Struct { fields, .. } => fields,
            other => panic!("expected struct, got {:?}", other),
        };
        assert_eq!(fields.len(), 3); // id, labels, properties only
    }

    #[test]
    fn datetime_uses_local_datetime_struct() {
        let micros = 1_700_000_000_000_001; // arbitrary
        let v = runtime_to_bolt(&RuntimeValue::DateTime(micros), ElementIdMode::Include);
        let (tag, fields) = match v {
            Value::Struct { tag, fields } => (tag, fields),
            other => panic!("expected struct, got {:?}", other),
        };
        assert_eq!(tag, struct_tag::LOCAL_DATETIME);
        assert_eq!(fields.len(), 2);
        // Round-trip: encode then decode the seconds+nanos back to micros.
        let seconds = match fields[0] {
            Value::Int(n) => n,
            _ => panic!(),
        };
        let nanos = match fields[1] {
            Value::Int(n) => n,
            _ => panic!(),
        };
        assert_eq!(seconds * 1_000_000 + nanos / 1_000, micros);
    }

    #[test]
    fn params_from_bolt_map_basic() {
        let mut m = BTreeMap::new();
        m.insert("k".into(), Value::Int(7));
        m.insert("s".into(), Value::String("hi".into()));
        let params = params_from_bolt_map(&m);
        assert_eq!(params.get("k"), Some(&RuntimeValue::Integer(7)));
        assert_eq!(params.get("s"), Some(&RuntimeValue::String("hi".into())));
    }
}
