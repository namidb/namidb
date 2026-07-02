//! Per-writer transactional index over declared-unique property values.
//!
//! [`crate::WriterSession::unique_probe`] answers "which node currently
//! holds this value tuple for `(label, properties)`?" in O(1) after a
//! one-time label scan, instead of re-scanning the label for every row a
//! constraint-bearing bulk write stages. Unlike
//! [`crate::property_index::PropertyIndexCache`] (committed state only,
//! shared across snapshots), this index is private to one `WriterSession`
//! and tracks committed **plus staged** state: it is populated lazily from
//! the read-your-own-writes overlay — the same source the flat scan uses —
//! and then kept current by every staged node upsert/tombstone, so a value
//! freed or claimed earlier in the same uncommitted batch is visible to the
//! next check.
//!
//! Consistency contract: a populated `(label, property-set)` map must agree
//! with a fresh `scan_label` over the overlay snapshot at all times. Any
//! event that can change node content outside the staged-write chokepoints
//! (commit, flush, SST attach, DDL, batch discard) resets the index; the
//! next probe repopulates from a fresh scan.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use namidb_core::id::NodeId;
use namidb_core::Value;

/// Outcome of probing the writer's unique-value index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniqueProbe {
    /// A node other than the excluded one currently holds the value tuple.
    Conflict(NodeId),
    /// No other node holds the value tuple.
    NoConflict,
    /// At least one probed value has no canonical scalar encoding
    /// (vector/list/map/null/NaN); the caller must fall back to the
    /// scan-based check, which IS the source of truth.
    Unindexable,
}

/// Canonical, hashable form of one scalar property value. Key equality must
/// match [`Value`]'s derived `PartialEq` exactly — that is what the flat
/// scan compares with — so `I64(1)` and `F64(1.0)` stay distinct, `-0.0`
/// is folded into `0.0` (`f64::eq` treats them as equal, bit patterns do
/// not), and NaN is rejected at encode time (`NaN != NaN` means a scan can
/// never observe a NaN conflict).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum UniqueKeyPart {
    Str(String),
    I64(i64),
    Bool(bool),
    /// Bit pattern of a non-NaN f64 with -0.0 normalised to 0.0.
    F64(u64),
    Bytes(Vec<u8>),
    Date(i32),
    DateTime(i64),
}

pub(crate) type UniqueKey = Vec<UniqueKeyPart>;

fn key_part(v: &Value) -> Option<UniqueKeyPart> {
    match v {
        Value::Str(s) => Some(UniqueKeyPart::Str(s.clone())),
        Value::I64(n) => Some(UniqueKeyPart::I64(*n)),
        Value::Bool(b) => Some(UniqueKeyPart::Bool(*b)),
        Value::F64(f) if !f.is_nan() => {
            let f = if *f == 0.0 { 0.0 } else { *f };
            Some(UniqueKeyPart::F64(f.to_bits()))
        }
        Value::Bytes(b) => Some(UniqueKeyPart::Bytes(b.clone())),
        Value::Date(d) => Some(UniqueKeyPart::Date(*d)),
        Value::DateTime(m) => Some(UniqueKeyPart::DateTime(*m)),
        Value::F64(_)
        | Value::Null
        | Value::Vec(_)
        | Value::VecI8 { .. }
        | Value::List(_)
        | Value::Map(_) => None,
    }
}

/// Encode the probe values themselves (already paired with their sorted
/// property names). `None` when any value is unindexable.
pub(crate) fn encode_probe_key(values: &[&Value]) -> Option<UniqueKey> {
    values.iter().map(|v| key_part(v)).collect()
}

/// Encode a node's key for a constraint over `names` (sorted) from its
/// property map. `None` when any property is absent or unindexable — such a
/// node cannot equal an indexable probe tuple, so it is simply not filed.
fn encode_node_key(names: &[String], props: &BTreeMap<String, Value>) -> Option<UniqueKey> {
    names.iter().map(|n| props.get(n).and_then(key_part)).collect()
}

/// One populated `(label, property-set)` constraint map.
///
/// `holders` keeps EVERY node currently carrying a value tuple (normally one,
/// but pre-existing duplicates — e.g. a constraint declared over data that
/// already violates it — must keep answering "conflict" exactly like the
/// scan would). `by_node` is the reverse edge that makes staged upserts and
/// tombstones O(1): a full-record upsert first detaches the node from its
/// previous tuple, then files it under the new one.
#[derive(Debug, Default)]
struct ConstraintMap {
    holders: HashMap<UniqueKey, Vec<NodeId>>,
    by_node: HashMap<NodeId, UniqueKey>,
}

impl ConstraintMap {
    fn detach(&mut self, id: NodeId) {
        if let Some(old) = self.by_node.remove(&id) {
            if let Some(ids) = self.holders.get_mut(&old) {
                ids.retain(|x| *x != id);
                if ids.is_empty() {
                    self.holders.remove(&old);
                }
            }
        }
    }

    fn file(&mut self, id: NodeId, key: UniqueKey) {
        let ids = self.holders.entry(key.clone()).or_default();
        if !ids.contains(&id) {
            ids.push(id);
        }
        self.by_node.insert(id, key);
    }
}

/// The per-writer index: `(label, sorted property names) → ConstraintMap`.
/// Interior mutability because probes run under `&WriterSession` while
/// staged-write maintenance runs under `&mut` — the writer is single-owner,
/// so the mutex is uncontended.
#[derive(Debug, Default)]
pub struct UniqueConstraintIndex {
    maps: Mutex<HashMap<(String, Vec<String>), ConstraintMap>>,
    /// Label scans performed to populate a constraint map. Exposed so tests
    /// can assert a bulk write pays exactly one scan, not one per row.
    populate_scans: AtomicU64,
    /// Probes answered from a populated map (i.e. without scanning).
    probes: AtomicU64,
}

impl UniqueConstraintIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Probe a populated map. `None` when `(label, names)` has not been
    /// populated yet — the caller scans and calls [`Self::populate`].
    pub(crate) fn probe(
        &self,
        label: &str,
        names: &[String],
        key: &UniqueKey,
        exclude: Option<NodeId>,
    ) -> Option<UniqueProbe> {
        let maps = self.maps.lock().expect("unique index lock");
        let map = maps.get(&(label.to_string(), names.to_vec()))?;
        self.probes.fetch_add(1, Ordering::Relaxed);
        let conflict = map
            .holders
            .get(key)
            .and_then(|ids| ids.iter().find(|id| Some(**id) != exclude));
        Some(match conflict {
            Some(id) => UniqueProbe::Conflict(*id),
            None => UniqueProbe::NoConflict,
        })
    }

    /// Install the `(label, names)` map from a label scan over the overlay
    /// snapshot. `entries` yields every live node of the label with its
    /// current property map.
    pub(crate) fn populate<'a>(
        &self,
        label: &str,
        names: &[String],
        entries: impl Iterator<Item = (NodeId, &'a BTreeMap<String, Value>)>,
    ) {
        let mut map = ConstraintMap::default();
        for (id, props) in entries {
            if let Some(key) = encode_node_key(names, props) {
                map.file(id, key);
            }
        }
        self.populate_scans.fetch_add(1, Ordering::Relaxed);
        self.maps
            .lock()
            .expect("unique index lock")
            .insert((label.to_string(), names.to_vec()), map);
    }

    /// Maintain every populated map for a staged full-record node upsert:
    /// the node's previous tuple (if any) is freed, and it is re-filed under
    /// each constraint whose label it carries and whose properties are all
    /// present and indexable in the new record.
    pub(crate) fn apply_upsert(
        &self,
        id: NodeId,
        labels: &[&str],
        props: &BTreeMap<String, Value>,
    ) {
        let mut maps = self.maps.lock().expect("unique index lock");
        for ((clabel, cnames), map) in maps.iter_mut() {
            map.detach(id);
            if !labels.iter().any(|l| l == clabel) {
                continue;
            }
            if let Some(key) = encode_node_key(cnames, props) {
                map.file(id, key);
            }
        }
    }

    /// Maintain every populated map for a staged node tombstone.
    pub(crate) fn apply_tombstone(&self, id: NodeId) {
        let mut maps = self.maps.lock().expect("unique index lock");
        for map in maps.values_mut() {
            map.detach(id);
        }
    }

    /// Drop every populated map; the next probe repopulates from a scan.
    pub(crate) fn reset(&self) {
        self.maps.lock().expect("unique index lock").clear();
    }

    /// Number of populating label scans performed so far.
    pub fn populate_scans(&self) -> u64 {
        self.populate_scans.load(Ordering::Relaxed)
    }

    /// Number of probes answered from a populated map (no scan).
    pub fn probes(&self) -> u64 {
        self.probes.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(n: u8) -> NodeId {
        NodeId::from_uuid(uuid::Uuid::from_bytes([n; 16]))
    }

    fn props(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn key_encoding_matches_value_equality_semantics() {
        // I64 and F64 stay distinct, exactly like `Value`'s PartialEq.
        assert_ne!(key_part(&Value::I64(1)), key_part(&Value::F64(1.0)));
        // -0.0 and 0.0 compare equal as f64, so they must share a key.
        assert_eq!(key_part(&Value::F64(0.0)), key_part(&Value::F64(-0.0)));
        // NaN never equals anything under the scan; refuse to index it.
        assert_eq!(key_part(&Value::F64(f64::NAN)), None);
        // Non-scalar values are unindexable and force the scan fallback.
        assert_eq!(key_part(&Value::List(vec![Value::I64(1)])), None);
        assert_eq!(key_part(&Value::Null), None);
    }

    #[test]
    fn upsert_rehomes_and_tombstone_frees() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["email".to_string()];
        let a = props(&[("email", Value::Str("a@x".into()))]);
        idx.populate("User", &names, vec![(nid(1), &a)].into_iter());

        let key = encode_probe_key(&[&Value::Str("a@x".into())]).unwrap();
        assert_eq!(
            idx.probe("User", &names, &key, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );
        // Self-exclusion: the holder rewriting its own value is not a conflict.
        assert_eq!(
            idx.probe("User", &names, &key, Some(nid(1))),
            Some(UniqueProbe::NoConflict)
        );

        // Full-record upsert moves the node to a new value, freeing the old.
        let b = props(&[("email", Value::Str("b@x".into()))]);
        idx.apply_upsert(nid(1), &["User"], &b);
        assert_eq!(
            idx.probe("User", &names, &key, None),
            Some(UniqueProbe::NoConflict)
        );
        let key_b = encode_probe_key(&[&Value::Str("b@x".into())]).unwrap();
        assert_eq!(
            idx.probe("User", &names, &key_b, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );

        idx.apply_tombstone(nid(1));
        assert_eq!(
            idx.probe("User", &names, &key_b, None),
            Some(UniqueProbe::NoConflict)
        );
    }

    #[test]
    fn preexisting_duplicates_keep_conflicting_like_the_scan() {
        // Two nodes already carry the same value (constraint declared over
        // violating data). Excluding one must still surface the other.
        let idx = UniqueConstraintIndex::new();
        let names = vec!["code".to_string()];
        let v = props(&[("code", Value::I64(7))]);
        idx.populate("A", &names, vec![(nid(1), &v), (nid(2), &v)].into_iter());
        let key = encode_probe_key(&[&Value::I64(7)]).unwrap();
        assert_eq!(
            idx.probe("A", &names, &key, Some(nid(1))),
            Some(UniqueProbe::Conflict(nid(2)))
        );
    }

    #[test]
    fn label_scoping_and_incomplete_tuples() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["a".to_string(), "b".to_string()];
        idx.populate("L", &names, std::iter::empty());

        // A node not carrying the constraint's label is never filed.
        let p = props(&[("a", Value::I64(1)), ("b", Value::I64(2))]);
        idx.apply_upsert(nid(1), &["Other"], &p);
        let key = encode_probe_key(&[&Value::I64(1), &Value::I64(2)]).unwrap();
        assert_eq!(
            idx.probe("L", &names, &key, None),
            Some(UniqueProbe::NoConflict)
        );

        // Missing tuple element → not filed either.
        let partial = props(&[("a", Value::I64(1))]);
        idx.apply_upsert(nid(2), &["L"], &partial);
        assert_eq!(
            idx.probe("L", &names, &key, None),
            Some(UniqueProbe::NoConflict)
        );

        // Complete tuple on the right label → conflict.
        idx.apply_upsert(nid(3), &["L"], &p);
        assert_eq!(
            idx.probe("L", &names, &key, None),
            Some(UniqueProbe::Conflict(nid(3)))
        );
    }

    #[test]
    fn unpopulated_probe_reports_none() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["x".to_string()];
        let key = encode_probe_key(&[&Value::I64(1)]).unwrap();
        assert_eq!(idx.probe("L", &names, &key, None), None);
        assert_eq!(idx.probes(), 0);
    }
}
