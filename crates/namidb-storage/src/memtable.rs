//! In-memory write buffer between a successful WAL append and a future SST
//! flush.
//!
//! Single-writer model: a [`Memtable`] is owned by exactly one writer task,
//! so we get away with a `BTreeMap` and no synchronisation. Reads come
//! through the same task, or through immutable snapshots (later).
//!
//! ## Semantics
//!
//! - Each ([`MemKey`]) maps to the latest [`MemEntry`] that was applied.
//! - A later `apply` with the same key replaces the value; the byte-size
//! accounting is updated accordingly.
//! - Tombstones (deletes) are stored as [`MemOp::Tombstone`]. They are kept
//! until they are merged into an SST, so subsequent reads correctly see
//! "absent" rather than the stale pre-delete value when the memtable is
//! later combined with cold SSTs at read time.
//! - The memtable does **not** know about the WAL. The writer is expected to
//! call [`Memtable::apply`] only after the WAL append for that record has
//! returned success.

use std::collections::BTreeMap;
use std::ops::Bound;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use namidb_core::NodeId;

/// Key into the memtable.
///
/// Nodes are keyed by `id` alone (id-primary): a node's label set rides in the
/// value (the encoded `NodeWriteRecord`), not the key, so the same id is one
/// row regardless of how many labels it carries. Edges still key by
/// `(edge_type, src, dst)`. Node keys sort before edge keys (variant order),
/// and within nodes by `id`, which keeps flush output id-ascending for free.
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum MemKey {
    Node {
        id: NodeId,
    },
    Edge {
        edge_type: String,
        src: NodeId,
        dst: NodeId,
    },
}

impl MemKey {
    /// Scope string used by the flush/manifest layer. Nodes are no longer
    /// partitioned by label, so their scope is empty; edges scope by type.
    pub fn scope(&self) -> &str {
        match self {
            MemKey::Node { .. } => "",
            MemKey::Edge { edge_type, .. } => edge_type,
        }
    }
    fn approx_bytes(&self) -> usize {
        match self {
            MemKey::Node { .. } => 16,
            MemKey::Edge { edge_type, .. } => edge_type.len() + 32,
        }
    }
}

/// Operation stored against a [`MemKey`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemOp {
    Upsert(Bytes),
    Tombstone,
}

impl MemOp {
    fn approx_bytes(&self) -> usize {
        match self {
            MemOp::Upsert(b) => b.len(),
            MemOp::Tombstone => 0,
        }
    }
}

/// Value stored against a [`MemKey`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemEntry {
    pub lsn: u64,
    pub op: MemOp,
}

/// In-memory write buffer.
///
/// `Memtable` is the writer-side, mutable form: the writer keeps a
/// single instance and applies records into it. Readers consume the
/// memtable through a [`MemtableSnapshot`] taken at a well-defined
/// moment (commit_batch / flush) and shared via `Arc` so multiple
/// concurrent reads see the same point-in-time view without holding
/// any writer lock. See RFC-021.
#[derive(Debug, Default)]
pub struct Memtable {
    inner: BTreeMap<MemKey, MemEntry>,
    bytes: usize,
}

impl Memtable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Best-effort byte size; used by the writer to decide when to flush.
    pub fn bytes_estimate(&self) -> usize {
        self.bytes
    }

    /// Apply a single mutation. Returns the previous [`MemEntry`] if the key
    /// was already present (mostly useful for tests / metrics).
    pub fn apply(&mut self, key: MemKey, lsn: u64, op: MemOp) -> Option<MemEntry> {
        let new_bytes = key.approx_bytes() + op.approx_bytes() + 24 /* MemEntry overhead */;
        let prev = self.inner.insert(key.clone(), MemEntry { lsn, op });
        let old_bytes = prev
            .as_ref()
            .map(|e| key.approx_bytes() + e.op.approx_bytes() + 24)
            .unwrap_or(0);
        self.bytes = self.bytes + new_bytes - old_bytes;
        prev
    }

    pub fn get(&self, key: &MemKey) -> Option<&MemEntry> {
        self.inner.get(key)
    }

    /// Iterate every entry in key order. Cheap; the memtable owns the keys.
    pub fn iter(&self) -> impl Iterator<Item = (&MemKey, &MemEntry)> {
        self.inner.iter()
    }

    /// Iterate every node entry (all `MemKey::Node`) in id order. Nodes sort
    /// before edges and the read path filters by decoding each value's label
    /// set, since the label is no longer in the key.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (&MemKey, &MemEntry)> {
        let start = MemKey::Node {
            id: NodeId::from_uuid(uuid::Uuid::nil()),
        };
        let end = MemKey::Node {
            id: NodeId::from_uuid(uuid::Uuid::max()),
        };
        self.inner
            .range((Bound::Included(start), Bound::Included(end)))
    }

    /// Iterate entries restricted to a single edge type.
    pub fn iter_edge_type<'a>(
        &'a self,
        edge_type: &'a str,
    ) -> impl Iterator<Item = (&'a MemKey, &'a MemEntry)> + 'a {
        // We cannot tightly bound the BTreeMap range across the (src, dst)
        // pair without overflow gymnastics; a filtering scan is fine for the
        // memtable since it is bounded by the flush threshold.
        self.inner.iter().filter(
            move |(k, _)| matches!(k, MemKey::Edge { edge_type: et, .. } if et == edge_type),
        )
    }

    /// Swap out the contents into a frozen [`FrozenMemtable`], leaving `self`
    /// empty. Used right before SST flush so writers can keep accepting
    /// records while the previous batch is being written out.
    pub fn freeze(&mut self) -> FrozenMemtable {
        let inner = std::mem::take(&mut self.inner);
        let bytes = std::mem::take(&mut self.bytes);
        FrozenMemtable { inner, bytes }
    }

    /// Build an immutable [`MemtableSnapshot`] of the current state.
    ///
    /// The snapshot owns its own copy of the `BTreeMap`, so the
    /// returned value lives independently of the writer that produced
    /// it. Readers consume the snapshot via `Arc<MemtableSnapshot>` so
    /// many concurrent reads share the same allocation without locking
    /// the writer (RFC-021).
    pub fn snapshot_view(&self) -> MemtableSnapshot {
        MemtableSnapshot {
            inner: self.inner.clone(),
        }
    }
}

/// Immutable, read-only view of a memtable at a point in time.
///
/// Built by [`Memtable::snapshot_view`] and shared across concurrent
/// readers via `Arc`. Exposes the same read API a `&Memtable` did
/// (`iter`, `get`, `iter_nodes`, `iter_edge_type`), with no mutation
/// surface. See RFC-021.
#[derive(Debug, Default, Clone)]
pub struct MemtableSnapshot {
    inner: BTreeMap<MemKey, MemEntry>,
}

impl MemtableSnapshot {
    /// Empty snapshot. Used as the initial value of the published cell
    /// before any commit has happened.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn get(&self, key: &MemKey) -> Option<&MemEntry> {
        self.inner.get(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&MemKey, &MemEntry)> {
        self.inner.iter()
    }

    /// Iterate every node entry (all `MemKey::Node`) in id order. The read
    /// path decodes each value's label set to filter by label.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (&MemKey, &MemEntry)> {
        let start = MemKey::Node {
            id: NodeId::from_uuid(uuid::Uuid::nil()),
        };
        let end = MemKey::Node {
            id: NodeId::from_uuid(uuid::Uuid::max()),
        };
        self.inner
            .range((Bound::Included(start), Bound::Included(end)))
    }

    pub fn iter_edge_type<'a>(
        &'a self,
        edge_type: &'a str,
    ) -> impl Iterator<Item = (&'a MemKey, &'a MemEntry)> + 'a {
        self.inner.iter().filter(
            move |(k, _)| matches!(k, MemKey::Edge { edge_type: et, .. } if et == edge_type),
        )
    }
}

/// An immutable, ordered snapshot of a memtable, ready to be turned into
/// SSTs.
#[derive(Debug, Clone)]
pub struct FrozenMemtable {
    inner: BTreeMap<MemKey, MemEntry>,
    bytes: usize,
}

impl FrozenMemtable {
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn bytes_estimate(&self) -> usize {
        self.bytes
    }
    pub fn iter(&self) -> impl Iterator<Item = (&MemKey, &MemEntry)> {
        self.inner.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(byte: u8) -> NodeId {
        NodeId::from_uuid(uuid::Uuid::from_bytes([byte; 16]))
    }

    #[test]
    fn upsert_replaces_previous_value() {
        let mut mt = Memtable::new();
        let key = MemKey::Node { id: nid(1) };
        let prev = mt.apply(key.clone(), 1, MemOp::Upsert(Bytes::from_static(b"v1")));
        assert!(prev.is_none());

        let prev = mt.apply(
            key.clone(),
            2,
            MemOp::Upsert(Bytes::from_static(b"v2-longer")),
        );
        assert!(prev.is_some());
        assert_eq!(prev.unwrap().op, MemOp::Upsert(Bytes::from_static(b"v1")));

        let got = mt.get(&key).unwrap();
        assert_eq!(got.lsn, 2);
        assert_eq!(got.op, MemOp::Upsert(Bytes::from_static(b"v2-longer")));
    }

    #[test]
    fn tombstone_overrides_upsert() {
        let mut mt = Memtable::new();
        let key = MemKey::Node { id: nid(1) };
        mt.apply(key.clone(), 1, MemOp::Upsert(Bytes::from_static(b"v1")));
        mt.apply(key.clone(), 2, MemOp::Tombstone);
        assert_eq!(mt.get(&key).unwrap().op, MemOp::Tombstone);
    }

    #[test]
    fn bytes_estimate_tracks_replacements_and_deletes() {
        let mut mt = Memtable::new();
        let key = MemKey::Node { id: nid(1) };
        mt.apply(key.clone(), 1, MemOp::Upsert(Bytes::from_static(b"v1")));
        let after_first = mt.bytes_estimate();
        mt.apply(
            key.clone(),
            2,
            MemOp::Upsert(Bytes::from_static(b"v2_is_longer_than_v1")),
        );
        assert!(mt.bytes_estimate() > after_first);
        mt.apply(key.clone(), 3, MemOp::Tombstone);
        // Tombstone reclaims the payload bytes but keeps the key entry, so
        // the new estimate must be smaller than after the long upsert.
        assert!(mt.bytes_estimate() < after_first + 20);
    }

    #[test]
    fn iter_yields_keys_in_order() {
        let mut mt = Memtable::new();
        let keys = [
            MemKey::Node { id: nid(3) },
            MemKey::Node { id: nid(1) },
            MemKey::Node { id: nid(2) },
        ];
        for (i, k) in keys.iter().enumerate() {
            mt.apply(k.clone(), i as u64, MemOp::Upsert(Bytes::from_static(b"x")));
        }
        let observed: Vec<u8> = mt
            .iter()
            .map(|(k, _)| match k {
                MemKey::Node { id, .. } => id.as_bytes()[0],
                MemKey::Edge { .. } => unreachable!(),
            })
            .collect();
        assert_eq!(observed, vec![1, 2, 3]);
    }

    #[test]
    fn iter_nodes_and_edge_type_scope_correctly() {
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: nid(1) },
            1,
            MemOp::Upsert(Bytes::from_static(b"a")),
        );
        mt.apply(
            MemKey::Node { id: nid(2) },
            2,
            MemOp::Upsert(Bytes::from_static(b"b")),
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: nid(1),
                dst: nid(3),
            },
            3,
            MemOp::Upsert(Bytes::from_static(b"c")),
        );

        // Nodes are no longer scoped by label in the key; iter_nodes yields
        // every node entry regardless of label (two distinct ids inserted).
        assert_eq!(mt.iter_nodes().count(), 2);
        assert_eq!(mt.iter_edge_type("KNOWS").count(), 1);
        assert_eq!(mt.iter_edge_type("OTHER").count(), 0);
    }

    #[test]
    fn freeze_resets_memtable_and_returns_data() {
        let mut mt = Memtable::new();
        for i in 0..5 {
            mt.apply(
                MemKey::Node { id: nid(i) },
                i as u64,
                MemOp::Upsert(Bytes::from_static(b"x")),
            );
        }
        let snapshot = mt.freeze();
        assert_eq!(snapshot.len(), 5);
        assert_eq!(mt.len(), 0);
        assert_eq!(mt.bytes_estimate(), 0);
        // Snapshot is iterable and preserves ordering.
        let mut prev: Option<&MemKey> = None;
        for (k, _) in snapshot.iter() {
            if let Some(p) = prev {
                assert!(p < k);
            }
            prev = Some(k);
        }
    }
}
