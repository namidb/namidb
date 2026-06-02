//! WAL replay: rebuild a [`Memtable`] from the WAL segments referenced
//! by a manifest.
//!
//! ## Where this fits
//!
//! After a writer crashes (or after a cold start in a new process), the
//! durable record of in-flight mutations lives in the WAL segments
//! referenced by the latest manifest. `recover_memtable` walks those
//! segments in `seq` order, decodes each [`WalEntry`] inside, and
//! replays it into a fresh `Memtable`.
//!
//! Once the caller holds the reconstructed memtable, the normal flush
//! path can run against it and durably retire those WAL segments.
//!
//! ## Wire format
//!
//! Each [`crate::wal::WalRecord`] frames a single [`WalEntry`] inside
//! its `payload` field. Encoding is `bincode`:
//!
//! ```text
//! WalEntry { key: MemKey, op: WalOp, lsn: u64 }
//! WalOp = Upsert(Vec<u8>) | Tombstone
//! ```
//!
//! `WalOp` mirrors [`MemOp`] but owns `Vec<u8>` instead of [`bytes::Bytes`]
//! because `Bytes` does not derive `serde::Serialize`. Conversion is
//! zero-copy in one direction (the `Vec` is wrapped) and copy-once in
//! the other (the `Bytes::to_vec()` happens once per WAL append).
//!
//! `MemKey` and the `NodeId` it contains both already derive
//! `Serialize`/`Deserialize`, so the envelope serialises straightforwardly.
//! bincode 1.x rejects `deserialize_any` and that bites the flush-time
//! [`crate::flush::NodeWriteRecord`] (which transitively includes the
//! untagged [`namidb_core::Value`]), but the WAL envelope only owns
//! tagged enums and concrete primitives, so bincode is the right tool here.

use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::error::{Error, Result};
use crate::manifest::Manifest;
use crate::memtable::{MemKey, MemOp, Memtable};
use crate::paths::NamespacePaths;
use crate::wal::WalStore;

/// Serializable mirror of [`MemOp`]. See module docs for the rationale.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalOp {
    Upsert(Vec<u8>),
    Tombstone,
}

/// Envelope written inside each [`crate::wal::WalRecord::payload`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalEntry {
    pub key: MemKey,
    pub op: WalOp,
    pub lsn: u64,
}

impl WalEntry {
    /// Build a [`WalEntry`] from the same triple `Memtable::apply` would
    /// receive. The bytes inside `op` are copied once.
    pub fn from_apply(key: MemKey, lsn: u64, op: &MemOp) -> Self {
        let op = match op {
            MemOp::Upsert(b) => WalOp::Upsert(b.to_vec()),
            MemOp::Tombstone => WalOp::Tombstone,
        };
        Self { key, op, lsn }
    }

    pub fn encode(&self) -> Result<Bytes> {
        let bytes = bincode::serialize(self)
            .map_err(|e| Error::invariant(format!("bincode encode WalEntry: {e}")))?;
        Ok(Bytes::from(bytes))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| Error::invariant(format!("bincode decode WalEntry: {e}")))
    }

    /// Decompose into the `(key, lsn, op)` triple `Memtable::apply` takes.
    pub fn into_memtable_apply(self) -> (MemKey, u64, MemOp) {
        let op = match self.op {
            WalOp::Upsert(v) => MemOp::Upsert(Bytes::from(v)),
            WalOp::Tombstone => MemOp::Tombstone,
        };
        (self.key, self.lsn, op)
    }
}

/// Outcome of [`recover_memtable`].
#[derive(Debug)]
pub struct RecoveredMemtable {
    pub memtable: Memtable,
    /// Largest LSN observed across every replayed WAL record. `0` when
    /// the manifest had no WAL segments to replay.
    pub max_lsn: u64,
    /// Number of records actually applied to the memtable.
    pub records_replayed: usize,
    /// `true` when the cold-start path skipped at least one WAL
    /// record because a memtable snapshot already covered it.
    /// Diagnostic only — surfaced for benchmark assertions.
    pub used_snapshot: bool,
}

/// Bincode-serialised checkpoint of the memtable, persisted to
/// `paths.memtable_snapshot()` so a cold-starting writer can skip the
/// linear WAL replay for everything it covers.
///
/// `last_lsn` is the largest LSN already present in `entries`; the
/// recovery path only re-applies WAL records whose `lsn` is strictly
/// greater than this value.
#[derive(Debug, Serialize, Deserialize)]
pub struct MemtableSnapshotFile {
    /// Wire-format version. Bumped if `entries` ever changes shape.
    pub version: u32,
    pub last_lsn: u64,
    pub entries: Vec<MemtableSnapshotEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemtableSnapshotEntry {
    pub key: MemKey,
    pub lsn: u64,
    pub op: WalOp,
}

const MEMTABLE_SNAPSHOT_VERSION: u32 = 1;

impl MemtableSnapshotFile {
    /// Build a snapshot file from the current `(MemKey, lsn, MemOp)`
    /// view of a live memtable. The caller passes the iterator
    /// directly so the file does not require interior knowledge of
    /// the memtable representation.
    pub fn from_iter<I>(last_lsn: u64, iter: I) -> Self
    where
        I: IntoIterator<Item = (MemKey, u64, MemOp)>,
    {
        let entries = iter
            .into_iter()
            .map(|(key, lsn, op)| {
                let op = match op {
                    MemOp::Upsert(b) => WalOp::Upsert(b.to_vec()),
                    MemOp::Tombstone => WalOp::Tombstone,
                };
                MemtableSnapshotEntry { key, lsn, op }
            })
            .collect();
        Self {
            version: MEMTABLE_SNAPSHOT_VERSION,
            last_lsn,
            entries,
        }
    }
}

/// Persist `snapshot` to the configured object store path. Uses
/// `PutMode::Overwrite` so a fresh snapshot replaces the previous
/// one in a single PUT.
pub async fn write_memtable_snapshot(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    snapshot: &MemtableSnapshotFile,
) -> Result<()> {
    let bytes = bincode::serialize(snapshot)
        .map_err(|e| Error::invariant(format!("bincode encode memtable snapshot: {e}")))?;
    let path = paths.memtable_snapshot();
    let opts = PutOptions::from(PutMode::Overwrite);
    store
        .put_opts(&path, PutPayload::from(bytes), opts)
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

async fn try_read_memtable_snapshot(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
) -> Result<Option<MemtableSnapshotFile>> {
    match store.get(path).await {
        Ok(get_result) => {
            let bytes = get_result.bytes().await.map_err(Error::ObjectStore)?;
            let snap: MemtableSnapshotFile = bincode::deserialize(&bytes)
                .map_err(|e| Error::invariant(format!("bincode decode memtable snapshot: {e}")))?;
            if snap.version != MEMTABLE_SNAPSHOT_VERSION {
                // Future-proofing: a snapshot from a newer engine is
                // skipped rather than rejected. Callers fall back to
                // the full WAL replay.
                debug!(
                    version = snap.version,
                    expected = MEMTABLE_SNAPSHOT_VERSION,
                    "ignoring memtable snapshot with unknown version"
                );
                return Ok(None);
            }
            Ok(Some(snap))
        }
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(other) => Err(Error::ObjectStore(other)),
    }
}

/// Replay every WAL segment referenced by `manifest` and return the
/// resulting in-memory state.
#[instrument(
 skip(manifest, wal_store),
 fields(
 namespace = %wal_store.paths().namespace(),
 segments = manifest.wal_segments.len(),
 )
)]
pub async fn recover_memtable(
    manifest: &Manifest,
    wal_store: &WalStore,
) -> Result<RecoveredMemtable> {
    recover_memtable_with_snapshot(manifest, wal_store, None).await
}

/// Same shape as [`recover_memtable`], plus an optional object-store
/// handle used to look for a `memtable_snapshot.bin` checkpoint at
/// `paths.memtable_snapshot()`. If found and its version is supported,
/// the snapshot is loaded into the memtable and the WAL replay skips
/// every record whose LSN is already covered.
pub async fn recover_memtable_with_snapshot(
    manifest: &Manifest,
    wal_store: &WalStore,
    snapshot_store: Option<&Arc<dyn ObjectStore>>,
) -> Result<RecoveredMemtable> {
    let mut memtable = Memtable::new();
    let mut max_lsn: u64 = 0;
    let mut records_replayed = 0usize;

    // Phase 0: seed from a checkpoint if available.
    let mut used_snapshot = false;
    let mut snapshot_floor: u64 = 0;
    if let Some(store) = snapshot_store {
        let snap_path = wal_store.paths().memtable_snapshot();
        if let Some(snap) = try_read_memtable_snapshot(store, &snap_path).await? {
            debug!(
                last_lsn = snap.last_lsn,
                entries = snap.entries.len(),
                "seeding recovery from memtable snapshot"
            );
            for entry in snap.entries {
                let op = match entry.op {
                    WalOp::Upsert(v) => MemOp::Upsert(Bytes::from(v)),
                    WalOp::Tombstone => MemOp::Tombstone,
                };
                memtable.apply(entry.key, entry.lsn, op);
            }
            max_lsn = max_lsn.max(snap.last_lsn);
            snapshot_floor = snap.last_lsn;
            used_snapshot = true;
        }
    }

    if manifest.wal_segments.is_empty() {
        debug!("manifest has no WAL segments; recovery is a no-op");
        return Ok(RecoveredMemtable {
            memtable,
            max_lsn,
            records_replayed,
            used_snapshot,
        });
    }

    // Read segments in seq order so LSNs (which are monotonic per writer)
    // replay in their original sequence and `Memtable::apply` sees the
    // "last write wins" view we want.
    let mut segments: Vec<_> = manifest.wal_segments.iter().collect();
    segments.sort_by_key(|s| s.seq);

    for seg_desc in segments {
        // Fast path: if every record in this segment is already
        // covered by the snapshot, skip the GET entirely. WAL records
        // are LSN-ascending within a segment and the descriptor's
        // last_lsn is its high-water mark.
        if seg_desc.last_lsn <= snapshot_floor {
            continue;
        }
        let segment = wal_store.read_segment(seg_desc.seq).await?;
        let actual_last_lsn = segment.last_lsn();
        if actual_last_lsn != seg_desc.last_lsn {
            // Asymmetric semantics matter here: `actual > declared` means
            // the writer raced the manifest (a record landed after the
            // descriptor was prepared); `actual < declared` means the
            // segment body was truncated between writer ack and now.
            // Both leave the namespace in an inconsistent state we must
            // refuse to read past — the manifest is the source of truth
            // for "what should have been durable" and the segment body
            // is the source of truth for "what actually is durable".
            return Err(Error::Corrupted {
                path: seg_desc.path.clone(),
                detail: format!(
                    "wal segment {} declared last_lsn={} in manifest but body carries last_lsn={}",
                    seg_desc.seq, seg_desc.last_lsn, actual_last_lsn
                ),
            });
        }
        for record in segment.records {
            if record.lsn <= snapshot_floor {
                continue;
            }
            let entry = WalEntry::decode(&record.payload)?;
            if entry.lsn != record.lsn {
                return Err(Error::Corrupted {
                    path: seg_desc.path.clone(),
                    detail: format!(
                        "wal segment {}: WalEntry.lsn={} differs from WalRecord.lsn={}",
                        seg_desc.seq, entry.lsn, record.lsn
                    ),
                });
            }
            let (key, lsn, op) = entry.into_memtable_apply();
            memtable.apply(key, lsn, op);
            max_lsn = max_lsn.max(lsn);
            records_replayed += 1;
        }
    }

    Ok(RecoveredMemtable {
        memtable,
        max_lsn,
        records_replayed,
        used_snapshot,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use namidb_core::{NamespaceId, NodeId};
    use object_store::memory::InMemory;
    use object_store::ObjectStore;
    use uuid::Uuid;

    use super::*;
    use crate::fence::Epoch;
    use crate::manifest::WalSegmentDescriptor;
    use crate::paths::NamespacePaths;
    use crate::wal::{WalRecord, WalSegment};

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    fn nid(byte: u8) -> NodeId {
        let mut b = [0u8; 16];
        b[15] = byte;
        NodeId::from_uuid(Uuid::from_bytes(b))
    }

    #[test]
    fn wal_entry_round_trip_upsert() {
        let entry = WalEntry {
            key: MemKey::Node { id: nid(1) },
            op: WalOp::Upsert(b"payload-bytes".to_vec()),
            lsn: 7,
        };
        let bytes = entry.encode().unwrap();
        let back = WalEntry::decode(&bytes).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn wal_entry_round_trip_tombstone() {
        let entry = WalEntry {
            key: MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: nid(1),
                dst: nid(2),
            },
            op: WalOp::Tombstone,
            lsn: 42,
        };
        let bytes = entry.encode().unwrap();
        let back = WalEntry::decode(&bytes).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn from_apply_converts_memop() {
        let key = MemKey::Node { id: nid(3) };
        let upsert = MemOp::Upsert(Bytes::from_static(b"x"));
        let entry = WalEntry::from_apply(key.clone(), 5, &upsert);
        match entry.op {
            WalOp::Upsert(v) => assert_eq!(v, b"x"),
            _ => panic!("expected Upsert"),
        }
        assert_eq!(entry.lsn, 5);

        let tomb = WalEntry::from_apply(key, 6, &MemOp::Tombstone);
        assert!(matches!(tomb.op, WalOp::Tombstone));
        assert_eq!(tomb.lsn, 6);
    }

    #[tokio::test]
    async fn recover_empty_manifest_returns_empty_memtable() {
        let store = store();
        let wal = WalStore::new(store, paths("rec-empty"));
        let manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        let out = recover_memtable(&manifest, &wal).await.unwrap();
        assert!(out.memtable.is_empty());
        assert_eq!(out.max_lsn, 0);
        assert_eq!(out.records_replayed, 0);
    }

    #[tokio::test]
    async fn recover_replays_single_segment_in_record_order() {
        let store = store();
        let paths = paths("rec-single");
        let wal = WalStore::new(store, paths);

        // Build a segment with 3 records: insert Alice, insert Bob,
        // tombstone Alice.
        let mut seg = WalSegment::new(1);
        let alice_id = nid(1);
        let bob_id = nid(2);

        let e1 = WalEntry {
            key: MemKey::Node { id: alice_id },
            op: WalOp::Upsert(b"alice-v1".to_vec()),
            lsn: 10,
        };
        let e2 = WalEntry {
            key: MemKey::Node { id: bob_id },
            op: WalOp::Upsert(b"bob-v1".to_vec()),
            lsn: 11,
        };
        let e3 = WalEntry {
            key: MemKey::Node { id: alice_id },
            op: WalOp::Tombstone,
            lsn: 12,
        };
        for e in [&e1, &e2, &e3] {
            seg.push(WalRecord {
                lsn: e.lsn,
                payload: e.encode().unwrap(),
            });
        }
        wal.append_segment(&seg).await.unwrap();

        // Manifest that knows about this segment.
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: seg.seq,
            path: format!("tenants/rec-single/wal/{:016x}.wal", seg.seq),
            last_lsn: seg.last_lsn(),
        });

        let out = recover_memtable(&manifest, &wal).await.unwrap();
        assert_eq!(out.records_replayed, 3);
        assert_eq!(out.max_lsn, 12);
        assert_eq!(out.memtable.len(), 2);

        // Alice's last op was the tombstone.
        let alice_key = MemKey::Node { id: alice_id };
        let alice = out.memtable.get(&alice_key).unwrap();
        assert_eq!(alice.lsn, 12);
        assert_eq!(alice.op, MemOp::Tombstone);

        // Bob is still an upsert.
        let bob_key = MemKey::Node { id: bob_id };
        let bob = out.memtable.get(&bob_key).unwrap();
        assert_eq!(bob.lsn, 11);
        match &bob.op {
            MemOp::Upsert(b) => assert_eq!(b.as_ref(), b"bob-v1"),
            _ => panic!("expected Upsert"),
        }
    }

    #[tokio::test]
    async fn recover_walks_multiple_segments_in_seq_order() {
        let store = store();
        let paths = paths("rec-multi");
        let wal = WalStore::new(store, paths);

        // Segment 2 carries the older write (LSN 1), segment 1 carries
        // a tombstone overwriting it (LSN 5). With seq-ordered replay
        // segment 1 should apply first and the tombstone in segment 2
        // is the durable end state. (LSNs in this test are intentionally
        // not strictly increasing with seq to prove we trust seq order
        // and the Memtable's "last write wins" semantics, not a sort
        // by LSN.)
        let key = MemKey::Node { id: nid(7) };

        let mut seg_first = WalSegment::new(1);
        seg_first.push(WalRecord {
            lsn: 5,
            payload: WalEntry {
                key: key.clone(),
                op: WalOp::Upsert(b"first".to_vec()),
                lsn: 5,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&seg_first).await.unwrap();

        let mut seg_second = WalSegment::new(2);
        seg_second.push(WalRecord {
            lsn: 6,
            payload: WalEntry {
                key: key.clone(),
                op: WalOp::Tombstone,
                lsn: 6,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&seg_second).await.unwrap();

        // Manifest references the segments in reverse order to make sure
        // recovery still walks seq ascending, not manifest order.
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 2,
            path: format!("tenants/rec-multi/wal/{:016x}.wal", 2),
            last_lsn: 6,
        });
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: format!("tenants/rec-multi/wal/{:016x}.wal", 1),
            last_lsn: 5,
        });

        let out = recover_memtable(&manifest, &wal).await.unwrap();
        assert_eq!(out.records_replayed, 2);
        assert_eq!(out.max_lsn, 6);
        let entry = out.memtable.get(&key).unwrap();
        // Last apply wins → tombstone from seg=2.
        assert_eq!(entry.lsn, 6);
        assert_eq!(entry.op, MemOp::Tombstone);
    }

    #[tokio::test]
    async fn recover_detects_lsn_mismatch_between_envelope_and_frame() {
        let store = store();
        let wal = WalStore::new(store, paths("rec-lsnmismatch"));

        let mut seg = WalSegment::new(1);
        // Frame LSN is 1; envelope claims 999.
        seg.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: MemKey::Node { id: nid(9) },
                op: WalOp::Upsert(b"x".to_vec()),
                lsn: 999,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&seg).await.unwrap();

        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: "tenants/rec-lsnmismatch/wal/0000000000000001.wal".into(),
            last_lsn: 1,
        });

        let err = recover_memtable(&manifest, &wal).await.unwrap_err();
        match err {
            Error::Corrupted { detail, .. } => {
                assert!(detail.contains("WalEntry.lsn=999"));
                assert!(detail.contains("WalRecord.lsn=1"));
            }
            other => panic!("expected Corrupted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recover_detects_segment_last_lsn_below_declared() {
        // I6 (bug audit): the symmetric case to `above_declared`. If the
        // segment body in object storage carries fewer records than the
        // manifest promised, the segment was truncated between writer
        // ack and now. Silently accepting that hides data loss.
        let store = store();
        let wal = WalStore::new(store, paths("rec-lsnunder"));

        let mut seg = WalSegment::new(4);
        seg.push(WalRecord {
            lsn: 10,
            payload: WalEntry {
                key: MemKey::Node { id: nid(1) },
                op: WalOp::Upsert(b"x".to_vec()),
                lsn: 10,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&seg).await.unwrap();

        // Manifest claims last_lsn=50 but the segment only carries up to 10.
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 4,
            path: "tenants/rec-lsnunder/wal/0000000000000004.wal".into(),
            last_lsn: 50,
        });

        let err = recover_memtable(&manifest, &wal).await.unwrap_err();
        match err {
            Error::Corrupted { detail, .. } => {
                assert!(detail.contains("declared last_lsn=50"));
                assert!(detail.contains("last_lsn=10"));
            }
            other => panic!("expected Corrupted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recover_detects_segment_last_lsn_above_declared() {
        let store = store();
        let wal = WalStore::new(store, paths("rec-lsnover"));

        let mut seg = WalSegment::new(3);
        seg.push(WalRecord {
            lsn: 100,
            payload: WalEntry {
                key: MemKey::Node { id: nid(1) },
                op: WalOp::Tombstone,
                lsn: 100,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&seg).await.unwrap();

        // Manifest claims last_lsn=50 but the segment carries 100.
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 3,
            path: "tenants/rec-lsnover/wal/0000000000000003.wal".into(),
            last_lsn: 50,
        });

        let err = recover_memtable(&manifest, &wal).await.unwrap_err();
        match err {
            Error::Corrupted { detail, .. } => {
                assert!(detail.contains("declared last_lsn=50"));
                assert!(detail.contains("last_lsn=100"));
            }
            other => panic!("expected Corrupted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recover_with_snapshot_short_circuits_covered_segments() {
        // Layout for the test:
        //   * snapshot covers LSNs 1..=10 (one Person upsert at LSN 1).
        //   * WAL segment seq=0 has LSNs 1..=10 (already covered).
        //   * WAL segment seq=1 has LSN 11 (new record).
        // Recovery should skip seg 0 entirely, decode only seg 1, and
        // report `used_snapshot = true`.
        let store = store();
        let paths = paths("rec-snap-skip");
        let wal = WalStore::new(store.clone(), paths.clone());

        // Snapshot file.
        let snap = MemtableSnapshotFile::from_iter(
            10,
            vec![(
                MemKey::Node { id: nid(1) },
                1,
                MemOp::Upsert(Bytes::from_static(b"ada-v1")),
            )],
        );
        write_memtable_snapshot(&store, &paths, &snap)
            .await
            .unwrap();

        let new_record = WalEntry {
            key: MemKey::Node { id: nid(2) },
            op: WalOp::Upsert(b"bob-v1".to_vec()),
            lsn: 11,
        }
        .encode()
        .unwrap();
        let mut seg0 = WalSegment::new(0);
        seg0.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: MemKey::Node { id: nid(1) },
                op: WalOp::Upsert(b"ada-v1".to_vec()),
                lsn: 1,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&seg0).await.unwrap();
        let mut seg1 = WalSegment::new(1);
        seg1.push(WalRecord {
            lsn: 11,
            payload: new_record,
        });
        wal.append_segment(&seg1).await.unwrap();

        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 0,
            path: format!("wal#{}", 0),
            last_lsn: 1,
        });
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: format!("wal#{}", 1),
            last_lsn: 11,
        });

        let out = recover_memtable_with_snapshot(&manifest, &wal, Some(&store))
            .await
            .unwrap();
        assert!(out.used_snapshot);
        // Only the LSN-11 record replayed; the LSN-1 record came from
        // the snapshot.
        assert_eq!(out.records_replayed, 1);
        assert_eq!(out.max_lsn, 11);
        assert!(!out.memtable.is_empty());
    }

    #[tokio::test]
    async fn recover_without_snapshot_store_falls_back_to_full_replay() {
        // Same WAL layout as the previous test, but the caller does not
        // pass a snapshot store. The fast path is bypassed and every
        // record is replayed.
        let store = store();
        let paths = paths("rec-snap-fallback");
        let wal = WalStore::new(store.clone(), paths);

        let mut seg = WalSegment::new(0);
        seg.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: MemKey::Node { id: nid(1) },
                op: WalOp::Upsert(b"ada-v1".to_vec()),
                lsn: 1,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&seg).await.unwrap();
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 0,
            path: "wal#0".into(),
            last_lsn: 1,
        });

        let out = recover_memtable(&manifest, &wal).await.unwrap();
        assert!(!out.used_snapshot);
        assert_eq!(out.records_replayed, 1);
        assert_eq!(out.max_lsn, 1);
    }
}
