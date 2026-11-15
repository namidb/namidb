//! Conservative heap-size estimates for materialised read output.
//!
//! Used by the [`crate::cancel::RetainMeter`] call sites so a query's memory
//! account hears about the rows a storage read keeps for it while the read
//! is still running, not only when the read hands its whole result back.
//! The numbers model a 64-bit target with a glibc-like allocator (8-byte
//! header, 16-byte rounding, 32-byte minimum block) and B-tree nodes at
//! minimum fill, so they err on the high side, like the query layer's own
//! estimators.

use std::collections::{BTreeMap, BTreeSet};

use namidb_core::Value;

use crate::read::{EdgeView, NodeView};

const BTREE_CAPACITY: usize = 11;
const BTREE_MIN_LEN: usize = 5;
const BTREE_NODE_HEADER: usize = 16;

#[inline]
fn alloc(size: usize) -> usize {
    if size == 0 {
        0
    } else {
        size.saturating_add(8)
            .div_ceil(16)
            .saturating_mul(16)
            .max(32)
    }
}

/// Bytes of B-tree nodes holding `len` entries of `slot` bytes each.
#[inline]
fn btree(len: usize, slot: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let nodes = len.div_ceil(BTREE_MIN_LEN);
    nodes.saturating_mul(alloc(
        BTREE_NODE_HEADER.saturating_add(BTREE_CAPACITY.saturating_mul(slot)),
    ))
}

/// Heap bytes owned by `value` (its inline size is the caller's).
pub fn value_heap_bytes(value: &Value) -> usize {
    match value {
        Value::Str(s) => alloc(s.capacity()),
        Value::Bytes(b) => alloc(b.capacity()),
        Value::Vec(v) => alloc(v.capacity().saturating_mul(std::mem::size_of::<f32>())),
        Value::VecI8 { codes, .. } => alloc(codes.capacity()),
        Value::List(items) => items.iter().fold(
            alloc(
                items
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Value>()),
            ),
            |acc, item| acc.saturating_add(value_heap_bytes(item)),
        ),
        Value::Map(map) => map_heap_bytes(map),
        _ => 0,
    }
}

/// Heap bytes of a property map: B-tree nodes, keys and values.
pub fn map_heap_bytes(map: &BTreeMap<String, Value>) -> usize {
    let slot = std::mem::size_of::<String>() + std::mem::size_of::<Value>();
    map.iter()
        .fold(btree(map.len(), slot), |acc, (key, value)| {
            acc.saturating_add(alloc(key.capacity()))
                .saturating_add(value_heap_bytes(value))
        })
}

fn label_set_bytes(labels: &BTreeSet<String>) -> usize {
    labels.iter().fold(
        btree(labels.len(), std::mem::size_of::<String>()),
        |acc, label| acc.saturating_add(alloc(label.capacity())),
    )
}

impl NodeView {
    /// Conservative estimate of the bytes this view occupies, inline size
    /// included.
    pub fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<NodeView>()
            .saturating_add(label_set_bytes(&self.labels))
            .saturating_add(map_heap_bytes(&self.properties))
    }
}

impl EdgeView {
    /// Conservative estimate of the bytes this view occupies, inline size
    /// included.
    pub fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<EdgeView>()
            .saturating_add(alloc(self.edge_type.capacity()))
            .saturating_add(map_heap_bytes(&self.properties))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_list_properties_dominate_the_estimate() {
        let mut properties = BTreeMap::new();
        properties.insert(
            "emb".to_string(),
            Value::List((0..1024).map(|i| Value::F64(i as f64)).collect()),
        );
        let view = NodeView {
            id: namidb_core::NodeId::from_uuid(uuid::Uuid::nil()),
            labels: BTreeSet::from(["Item".to_string()]),
            properties,
            lsn: 1,
            schema_version: 0,
        };
        let bytes = view.estimated_bytes();
        assert!(bytes >= 1024 * std::mem::size_of::<Value>(), "{bytes}");
        assert!(bytes < 4 * 1024 * std::mem::size_of::<Value>(), "{bytes}");
    }
}
