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
use tracing::{debug, instrument, warn};

use crate::error::{Error, Result};
use crate::manifest::{Manifest, WalSegmentDescriptor};
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

/// Checkpoint of the memtable persisted to `paths.memtable_snapshot()`.
///
/// Engine-owned version 2 envelopes are bound to the exact manifest WAL
/// descriptor vector they materialise. A writer claim, DDL commit, or SST
/// compaction may advance the manifest version while preserving that vector,
/// so version/epoch are deliberately not part of the binding. A data commit
/// changes the vector and a flush clears it, making an older checkpoint
/// ineligible by construction.
///
/// The v2 wire starts with this struct's raw bincode encoding and appends a
/// checksummed private `NAMIMS02` binding. Keeping the historical three-field
/// value as the prefix makes rollback safe: a 2.0.4 reader decodes it, sees
/// the unknown `version == 2`, and falls back to WAL while ignoring the
/// trailing appendix. Callers must use [`write_memtable_snapshot`] rather
/// than serialising the value directly.
#[derive(Debug, Serialize, Deserialize)]
pub struct MemtableSnapshotFile {
    /// Public-prefix version. Legacy v1 bodies and v2 bodies are both decoded
    /// through this unchanged field before the private appendix is considered.
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

pub(crate) const MEMTABLE_SNAPSHOT_VERSION: u32 = 2;
const MEMTABLE_SNAPSHOT_MAGIC: &[u8; 8] = b"NAMIMS02";
const MEMTABLE_SNAPSHOT_HEADER_BYTES: usize = 8 + 8;
const MEMTABLE_SNAPSHOT_FOOTER_BYTES: usize = 8;

/// Private v2 appendix. The wire starts with a raw bincode encoding of
/// [`MemtableSnapshotFile`] and appends this binding after `NAMIMS02`.
/// Bincode 1.x's legacy `deserialize` permits trailing bytes, so a 2.0.4
/// reader decodes the unchanged public prefix, observes `version == 2`, and
/// safely falls back to WAL without understanding the appendix.
#[derive(Debug, Serialize, Deserialize)]
struct MemtableSnapshotBindingV2 {
    wal_segments: Option<Vec<WalSegmentBindingV2>>,
}

/// Bincode-safe mirror of [`WalSegmentDescriptor`].
///
/// The manifest descriptor omits `xxh3: None` for compact JSON. Reusing that
/// `skip_serializing_if` shape in a positional format leaves the decoder
/// expecting a field that was never emitted. This private wire always carries
/// all four fields.
#[derive(Debug, Serialize, Deserialize)]
struct WalSegmentBindingV2 {
    seq: u64,
    path: String,
    last_lsn: u64,
    xxh3: Option<u64>,
}

impl WalSegmentBindingV2 {
    fn from_descriptor(descriptor: &WalSegmentDescriptor) -> Self {
        Self {
            seq: descriptor.seq,
            path: descriptor.path.clone(),
            last_lsn: descriptor.last_lsn,
            xxh3: descriptor.xxh3,
        }
    }

    fn matches(&self, descriptor: &WalSegmentDescriptor) -> bool {
        self.seq == descriptor.seq
            && self.path == descriptor.path
            && self.last_lsn == descriptor.last_lsn
            && self.xxh3 == descriptor.xxh3
    }
}

#[derive(Debug)]
pub(crate) struct DecodedMemtableSnapshot {
    snapshot: MemtableSnapshotFile,
    wal_segments: Option<Vec<WalSegmentBindingV2>>,
}

impl DecodedMemtableSnapshot {
    pub(crate) fn covers_manifest(&self, manifest: &Manifest) -> bool {
        let binding_matches = self.wal_segments.as_ref().is_some_and(|segments| {
            segments.len() == manifest.wal_segments.len()
                && segments
                    .iter()
                    .zip(&manifest.wal_segments)
                    .all(|(bound, current)| bound.matches(current))
        });
        if self.snapshot.version != MEMTABLE_SNAPSHOT_VERSION || !binding_matches {
            return false;
        }
        let expected_last_lsn = manifest
            .wal_segments
            .iter()
            .map(|segment| segment.last_lsn)
            .max()
            .unwrap_or(0);
        self.snapshot.last_lsn == expected_last_lsn
            && self
                .snapshot
                .entries
                .iter()
                .all(|entry| entry.lsn <= self.snapshot.last_lsn)
            && (!manifest.wal_segments.is_empty() || self.snapshot.entries.is_empty())
    }
}

impl MemtableSnapshotFile {
    fn collect_entries<I>(iter: I) -> Vec<MemtableSnapshotEntry>
    where
        I: IntoIterator<Item = (MemKey, u64, MemOp)>,
    {
        iter.into_iter()
            .map(|(key, lsn, op)| {
                let op = match op {
                    MemOp::Upsert(b) => WalOp::Upsert(b.to_vec()),
                    MemOp::Tombstone => WalOp::Tombstone,
                };
                MemtableSnapshotEntry { key, lsn, op }
            })
            .collect()
    }

    /// Build an exact recovery checkpoint for `manifest` from the current
    /// `(MemKey, lsn, MemOp)` view of its live memtable.
    pub(crate) fn from_manifest<I>(manifest: &Manifest, iter: I) -> Self
    where
        I: IntoIterator<Item = (MemKey, u64, MemOp)>,
    {
        let last_lsn = manifest
            .wal_segments
            .iter()
            .map(|segment| segment.last_lsn)
            .max()
            .unwrap_or(0);
        Self {
            version: MEMTABLE_SNAPSHOT_VERSION,
            last_lsn,
            entries: Self::collect_entries(iter),
        }
    }

    /// Build a snapshot value with the historical public shape.
    ///
    /// Persisting it through [`write_memtable_snapshot`] deliberately creates
    /// an unbound cache entry, which recovery ignores because the public value
    /// cannot prove which manifest WAL closure it represents. The engine's
    /// writer uses a crate-private bound write helper.
    pub fn from_iter<I>(last_lsn: u64, iter: I) -> Self
    where
        I: IntoIterator<Item = (MemKey, u64, MemOp)>,
    {
        Self {
            version: MEMTABLE_SNAPSHOT_VERSION,
            last_lsn,
            entries: Self::collect_entries(iter),
        }
    }
}

/// Persist an unbound `snapshot` to the configured object store path.
///
/// This preserves the historical public API without allowing
/// caller-constructed entries to become recovery authority: the reader
/// treats this form as a cache miss and replays WAL. Engine-owned snapshots
/// use [`write_memtable_snapshot_for_manifest`] to add the exact private
/// binding. Both forms use `PutMode::Overwrite`.
pub async fn write_memtable_snapshot(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    snapshot: &MemtableSnapshotFile,
) -> Result<()> {
    write_memtable_snapshot_with_binding(store, paths, snapshot, None).await
}

/// Persist an engine-owned checkpoint bound to the exact WAL closure of
/// `manifest`. Kept crate-private so the legacy public API cannot accidentally
/// bless caller-constructed entries as authoritative.
pub(crate) async fn write_memtable_snapshot_for_manifest(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    manifest: &Manifest,
    snapshot: &MemtableSnapshotFile,
) -> Result<()> {
    write_memtable_snapshot_with_binding(
        store,
        paths,
        snapshot,
        Some(manifest.wal_segments.as_slice()),
    )
    .await
}

async fn write_memtable_snapshot_with_binding(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    snapshot: &MemtableSnapshotFile,
    wal_segments: Option<&[WalSegmentDescriptor]>,
) -> Result<()> {
    let prefix = bincode::serialize(snapshot)
        .map_err(|e| Error::invariant(format!("bincode encode memtable snapshot: {e}")))?;
    let binding = MemtableSnapshotBindingV2 {
        wal_segments: wal_segments.map(|segments| {
            segments
                .iter()
                .map(WalSegmentBindingV2::from_descriptor)
                .collect()
        }),
    };
    let binding = bincode::serialize(&binding)
        .map_err(|e| Error::invariant(format!("bincode encode snapshot binding: {e}")))?;
    let binding_len = u64::try_from(binding.len())
        .map_err(|_| Error::invariant("memtable snapshot binding exceeds u64 wire length"))?;
    let mut bytes = Vec::with_capacity(
        prefix
            .len()
            .saturating_add(MEMTABLE_SNAPSHOT_HEADER_BYTES)
            .saturating_add(binding.len())
            .saturating_add(MEMTABLE_SNAPSHOT_FOOTER_BYTES),
    );
    bytes.extend_from_slice(&prefix);
    bytes.extend_from_slice(MEMTABLE_SNAPSHOT_MAGIC);
    bytes.extend_from_slice(&binding_len.to_le_bytes());
    bytes.extend_from_slice(&binding);
    // Bind both the public recovery state and its private WAL closure. Any
    // prefix, header, or appendix mutation therefore turns this disposable
    // cache into a miss rather than recovery authority.
    let checksum = xxhash_rust::xxh3::xxh3_64(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    let path = paths.memtable_snapshot();
    let opts = PutOptions::from(PutMode::Overwrite);
    store
        .put_opts(&path, PutPayload::from(bytes), opts)
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

pub(crate) fn decode_memtable_snapshot(
    bytes: &[u8],
) -> std::result::Result<DecodedMemtableSnapshot, String> {
    // Decode through a cursor so its final position gives us the exact end of
    // the rollback-compatible public prefix. Re-serialising `entries` merely
    // to recover that offset would transiently duplicate a potentially large
    // memtable snapshot during cold start.
    let mut cursor = std::io::Cursor::new(bytes);
    let snapshot: MemtableSnapshotFile = bincode::deserialize_from(&mut cursor)
        .map_err(|error| format!("bincode decode memtable snapshot prefix: {error}"))?;
    if snapshot.version != MEMTABLE_SNAPSHOT_VERSION {
        return Err(format!(
            "snapshot payload version {} is unsupported (expected {})",
            snapshot.version, MEMTABLE_SNAPSHOT_VERSION
        ));
    }

    // Bincode's legacy reader intentionally accepts trailing bytes. Validate
    // the private appendix and the checksum over the original prefix bytes.
    let prefix_len = usize::try_from(cursor.position())
        .map_err(|_| "memtable snapshot prefix exceeds usize".to_string())?;
    let appendix = &bytes[prefix_len..];
    if appendix.len() < MEMTABLE_SNAPSHOT_HEADER_BYTES + MEMTABLE_SNAPSHOT_FOOTER_BYTES {
        return Err("body is too short for the v2 appendix".into());
    }
    if appendix[..8] != MEMTABLE_SNAPSHOT_MAGIC[..] {
        return Err("missing or unknown snapshot appendix magic".into());
    }
    let declared_len = u64::from_le_bytes(
        appendix[8..16]
            .try_into()
            .expect("snapshot length slice is fixed"),
    );
    let binding_len = usize::try_from(declared_len)
        .map_err(|_| "declared snapshot binding exceeds usize".to_string())?;
    let binding_start = prefix_len
        .checked_add(MEMTABLE_SNAPSHOT_HEADER_BYTES)
        .ok_or_else(|| "snapshot binding start overflow".to_string())?;
    let binding_end = binding_start
        .checked_add(binding_len)
        .ok_or_else(|| "snapshot binding end overflow".to_string())?;
    let expected_len = binding_end
        .checked_add(MEMTABLE_SNAPSHOT_FOOTER_BYTES)
        .ok_or_else(|| "snapshot wire length overflow".to_string())?;
    if bytes.len() != expected_len {
        return Err(format!(
            "snapshot wire length mismatch: declared {expected_len}, actual {}",
            bytes.len()
        ));
    }
    let declared_checksum = u64::from_le_bytes(
        bytes[binding_end..expected_len]
            .try_into()
            .expect("snapshot checksum slice is fixed"),
    );
    let actual_checksum = xxhash_rust::xxh3::xxh3_64(&bytes[..binding_end]);
    if declared_checksum != actual_checksum {
        return Err(format!(
            "snapshot checksum mismatch: declared {declared_checksum:#x}, actual {actual_checksum:#x}"
        ));
    }
    let binding: MemtableSnapshotBindingV2 =
        bincode::deserialize(&bytes[binding_start..binding_end])
            .map_err(|error| format!("bincode decode snapshot binding: {error}"))?;
    Ok(DecodedMemtableSnapshot {
        snapshot,
        wal_segments: binding.wal_segments,
    })
}

async fn try_read_memtable_snapshot(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
) -> Result<Option<DecodedMemtableSnapshot>> {
    match store.get(path).await {
        Ok(get_result) => {
            let bytes = match get_result.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    warn!(path = %path, %error, "memtable snapshot read failed; replaying WAL");
                    return Ok(None);
                }
            };
            match decode_memtable_snapshot(&bytes) {
                Ok(snapshot) => Ok(Some(snapshot)),
                Err(detail) => {
                    // The snapshot is a disposable cache. Legacy v1 bodies,
                    // truncation, bit-rot, and future formats all fall back to
                    // the authoritative WAL rather than preventing startup.
                    warn!(path = %path, %detail, "ignoring unusable memtable snapshot");
                    Ok(None)
                }
            }
        }
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(error) => {
            warn!(path = %path, %error, "memtable snapshot GET failed; replaying WAL");
            Ok(None)
        }
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
        match try_read_memtable_snapshot(store, &snap_path).await? {
            Some(snap) if !snap.covers_manifest(manifest) => {
                debug!(
                    snap_last_lsn = snap.snapshot.last_lsn,
                    manifest_wal_segments = manifest.wal_segments.len(),
                    "ignoring memtable snapshot bound to a different WAL closure"
                );
            }
            Some(snap) => {
                let snapshot = snap.snapshot;
                debug!(
                    last_lsn = snapshot.last_lsn,
                    entries = snapshot.entries.len(),
                    "seeding recovery from memtable snapshot"
                );
                for entry in snapshot.entries {
                    let op = match entry.op {
                        WalOp::Upsert(v) => MemOp::Upsert(Bytes::from(v)),
                        WalOp::Tombstone => MemOp::Tombstone,
                    };
                    memtable.apply(entry.key, entry.lsn, op);
                }
                max_lsn = max_lsn.max(snapshot.last_lsn);
                snapshot_floor = snapshot.last_lsn;
                used_snapshot = !manifest.wal_segments.is_empty();
            }
            None => {}
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
            xxh3: None,
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
            xxh3: None,
        });
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: format!("tenants/rec-multi/wal/{:016x}.wal", 1),
            last_lsn: 5,
            xxh3: None,
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
            xxh3: None,
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
            xxh3: None,
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
            xxh3: None,
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
    async fn pre_flush_snapshot_is_ignored_even_after_gc_lowers_sst_hwm() {
        // Model the dangerous post-GC shape: a valid old snapshot contains an
        // upsert and is bound to the pre-flush WAL, while the current manifest
        // has no WAL and no node SST because a later DELETE+tombstone were
        // fully garbage-collected. The old `snap.last_lsn > max_sst_lsn`
        // heuristic accepted this body (1 > 0) and resurrected the node.
        let store = store();
        let paths = paths("rec-snap-stale");
        let wal = WalStore::new(store.clone(), paths.clone());

        let mut pre_flush = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        pre_flush.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 1,
            xxh3: None,
        });
        let snap = MemtableSnapshotFile::from_manifest(
            &pre_flush,
            vec![(
                MemKey::Node { id: nid(1) },
                1,
                MemOp::Upsert(Bytes::from_static(b"resurrected")),
            )],
        );
        write_memtable_snapshot_for_manifest(&store, &paths, &pre_flush, &snap)
            .await
            .unwrap();

        let manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());

        let out = recover_memtable_with_snapshot(&manifest, &wal, Some(&store))
            .await
            .unwrap();
        assert!(!out.used_snapshot, "stale snapshot must be ignored");
        assert!(
            out.memtable.get(&MemKey::Node { id: nid(1) }).is_none(),
            "the stale row must NOT be re-seeded into the memtable"
        );
    }

    #[tokio::test]
    async fn public_snapshot_literal_remains_source_compatible_and_unbound() {
        let store = store();
        let paths = paths("rec-snap-public-unbound");
        // Keep this as a literal: adding a public field to the patch-release
        // API would make the regression fail to compile downstream too.
        let snapshot = MemtableSnapshotFile {
            version: MEMTABLE_SNAPSHOT_VERSION,
            last_lsn: 0,
            entries: Vec::new(),
        };
        write_memtable_snapshot(&store, &paths, &snapshot)
            .await
            .unwrap();
        let body = store
            .get(&paths.memtable_snapshot())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let decoded = decode_memtable_snapshot(&body).unwrap();
        let manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        assert!(
            !decoded.covers_manifest(&manifest),
            "the legacy public writer must never bless caller data as recovery authority"
        );
    }

    #[tokio::test]
    async fn v2_wire_is_ignored_cleanly_by_204_decoder() {
        // This deliberately mirrors the public three-field snapshot and
        // version gate used by 2.0.4. `bincode::deserialize` accepts trailing
        // bytes, so the private v2 appendix must not turn rollback into a
        // decode error.
        #[derive(Debug, serde::Deserialize)]
        struct MemtableSnapshotFile204 {
            version: u32,
            last_lsn: u64,
            entries: Vec<MemtableSnapshotEntry>,
        }

        fn decode_as_204(
            bytes: &[u8],
        ) -> std::result::Result<Option<MemtableSnapshotFile204>, bincode::Error> {
            let snapshot: MemtableSnapshotFile204 = bincode::deserialize(bytes)?;
            if snapshot.version != 1 {
                return Ok(None);
            }
            Ok(Some(snapshot))
        }

        let store = store();
        let paths = paths("rec-snap-rollback-204");
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 7,
            path: paths.wal_segment(7).as_ref().to_string(),
            last_lsn: 42,
            xxh3: Some(99),
        });
        let snapshot = MemtableSnapshotFile::from_manifest(
            &manifest,
            vec![(
                MemKey::Node { id: nid(7) },
                42,
                MemOp::Upsert(Bytes::from_static(b"rollback-safe")),
            )],
        );
        write_memtable_snapshot_for_manifest(&store, &paths, &manifest, &snapshot)
            .await
            .unwrap();
        let body = store
            .get(&paths.memtable_snapshot())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        let legacy_shape: MemtableSnapshotFile204 =
            bincode::deserialize(&body).expect("2.0.4 must decode the public prefix");
        assert_eq!(legacy_shape.version, MEMTABLE_SNAPSHOT_VERSION);
        assert_eq!(legacy_shape.last_lsn, 42);
        assert_eq!(legacy_shape.entries.len(), 1);
        assert!(
            decode_as_204(&body).unwrap().is_none(),
            "2.0.4 must ignore unknown v2 and replay WAL, not fail decoding"
        );
        assert!(
            decode_memtable_snapshot(&body)
                .unwrap()
                .covers_manifest(&manifest),
            "the same wire must remain authoritative for the v2 reader"
        );
    }

    #[tokio::test]
    async fn recover_with_exact_snapshot_skips_its_wal_closure() {
        // Layout for the test:
        //   * WAL segments seq=0/1 carry Ada@1 and Bob@11.
        //   * the snapshot is bound to exactly both descriptors and contains
        //     their reconciled memtable state.
        // Recovery should issue no WAL replay and report `used_snapshot`.
        let store = store();
        let paths = paths("rec-snap-skip");
        let wal = WalStore::new(store.clone(), paths.clone());

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
            xxh3: None,
        });
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: format!("wal#{}", 1),
            last_lsn: 11,
            xxh3: None,
        });

        let snap = MemtableSnapshotFile::from_manifest(
            &manifest,
            vec![
                (
                    MemKey::Node { id: nid(1) },
                    1,
                    MemOp::Upsert(Bytes::from_static(b"ada-v1")),
                ),
                (
                    MemKey::Node { id: nid(2) },
                    11,
                    MemOp::Upsert(Bytes::from_static(b"bob-v1")),
                ),
            ],
        );
        write_memtable_snapshot_for_manifest(&store, &paths, &manifest, &snap)
            .await
            .unwrap();
        let body = store
            .get(&paths.memtable_snapshot())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert!(
            decode_memtable_snapshot(&body)
                .unwrap()
                .covers_manifest(&manifest),
            "the freshly written exact snapshot must cover its WAL closure"
        );

        let out = recover_memtable_with_snapshot(&manifest, &wal, Some(&store))
            .await
            .unwrap();
        assert!(out.used_snapshot);
        assert_eq!(out.records_replayed, 0);
        assert_eq!(out.max_lsn, 11);
        assert!(out.memtable.get(&MemKey::Node { id: nid(1) }).is_some());
        assert!(out.memtable.get(&MemKey::Node { id: nid(2) }).is_some());
    }

    #[tokio::test]
    async fn writer_claim_keeps_snapshot_valid_when_wal_closure_is_unchanged() {
        let store = store();
        let paths = paths("rec-snap-claim");
        let wal = WalStore::new(store.clone(), paths.clone());
        let key = MemKey::Node { id: nid(1) };
        let mut before_claim = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        before_claim.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 1,
            xxh3: Some(7),
        });
        let snapshot = MemtableSnapshotFile::from_manifest(
            &before_claim,
            vec![(
                key.clone(),
                1,
                MemOp::Upsert(Bytes::from_static(b"snapshot")),
            )],
        );
        write_memtable_snapshot_for_manifest(&store, &paths, &before_claim, &snapshot)
            .await
            .unwrap();

        // `WriterSession::open` claims the writer before recovery. That
        // metadata-only manifest commit changes both version and epoch while
        // cloning the WAL descriptor vector exactly.
        let mut after_claim = before_claim.next_version(Uuid::now_v7());
        after_claim.epoch = before_claim.epoch.next();
        assert_ne!(after_claim.version, before_claim.version);
        assert_ne!(after_claim.epoch, before_claim.epoch);
        assert_eq!(after_claim.wal_segments, before_claim.wal_segments);

        // Deliberately do not PUT the WAL object: successful recovery proves
        // it accepted the exact-bound snapshot rather than replaying.
        let recovered = recover_memtable_with_snapshot(&after_claim, &wal, Some(&store))
            .await
            .unwrap();
        assert!(recovered.used_snapshot);
        assert_eq!(recovered.records_replayed, 0);
        assert_eq!(
            recovered.memtable.get(&key).unwrap().op,
            MemOp::Upsert(Bytes::from_static(b"snapshot"))
        );
    }

    #[tokio::test]
    async fn corrupt_snapshot_is_a_cache_miss_and_wal_replays() {
        let store = store();
        let paths = paths("rec-snap-corrupt");
        let wal = WalStore::new(store.clone(), paths.clone());
        let key = MemKey::Node { id: nid(1) };
        let payload = WalEntry {
            key: key.clone(),
            op: WalOp::Upsert(b"authoritative-wal".to_vec()),
            lsn: 1,
        }
        .encode()
        .unwrap();
        let mut segment = WalSegment::new(1);
        segment.push(WalRecord { lsn: 1, payload });
        wal.append_segment(&segment).await.unwrap();

        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 1,
            xxh3: None,
        });
        let snapshot = MemtableSnapshotFile::from_manifest(
            &manifest,
            vec![(
                key.clone(),
                1,
                MemOp::Upsert(Bytes::from_static(b"snapshot-copy")),
            )],
        );
        write_memtable_snapshot_for_manifest(&store, &paths, &manifest, &snapshot)
            .await
            .unwrap();
        let snapshot_path = paths.memtable_snapshot();
        let mut corrupt = store
            .get(&snapshot_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap()
            .to_vec();
        let prefix_len = bincode::serialize(&snapshot).unwrap().len();
        corrupt[prefix_len + MEMTABLE_SNAPSHOT_HEADER_BYTES] ^= 0x80;
        store
            .put(&snapshot_path, PutPayload::from(corrupt))
            .await
            .unwrap();

        let recovered = recover_memtable_with_snapshot(&manifest, &wal, Some(&store))
            .await
            .unwrap();
        assert!(!recovered.used_snapshot);
        assert_eq!(recovered.records_replayed, 1);
        assert_eq!(
            recovered.memtable.get(&key).unwrap().op,
            MemOp::Upsert(Bytes::from_static(b"authoritative-wal"))
        );
    }

    #[tokio::test]
    async fn truncated_snapshot_is_a_cache_miss_and_wal_replays() {
        let store = store();
        let paths = paths("rec-snap-truncated");
        let wal = WalStore::new(store.clone(), paths.clone());
        let key = MemKey::Node { id: nid(1) };
        let mut segment = WalSegment::new(1);
        segment.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: key.clone(),
                op: WalOp::Upsert(b"authoritative-wal".to_vec()),
                lsn: 1,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&segment).await.unwrap();

        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 1,
            xxh3: None,
        });
        let snapshot = MemtableSnapshotFile::from_manifest(
            &manifest,
            vec![(
                key.clone(),
                1,
                MemOp::Upsert(Bytes::from_static(b"snapshot-copy")),
            )],
        );
        write_memtable_snapshot_for_manifest(&store, &paths, &manifest, &snapshot)
            .await
            .unwrap();
        let snapshot_path = paths.memtable_snapshot();
        let mut truncated = store
            .get(&snapshot_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap()
            .to_vec();
        truncated.truncate(truncated.len() - 4);
        store
            .put(&snapshot_path, PutPayload::from(truncated))
            .await
            .unwrap();

        let recovered = recover_memtable_with_snapshot(&manifest, &wal, Some(&store))
            .await
            .unwrap();
        assert!(!recovered.used_snapshot);
        assert_eq!(recovered.records_replayed, 1);
        assert_eq!(
            recovered.memtable.get(&key).unwrap().op,
            MemOp::Upsert(Bytes::from_static(b"authoritative-wal"))
        );
    }

    #[tokio::test]
    async fn legacy_v1_snapshot_is_a_cache_miss_and_wal_replays() {
        #[derive(serde::Serialize)]
        struct LegacySnapshotV1 {
            version: u32,
            last_lsn: u64,
            entries: Vec<MemtableSnapshotEntry>,
        }

        let store = store();
        let paths = paths("rec-snap-v1");
        let wal = WalStore::new(store.clone(), paths.clone());
        let key = MemKey::Node { id: nid(1) };
        let mut segment = WalSegment::new(1);
        segment.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: key.clone(),
                op: WalOp::Upsert(b"wal-v2-reader".to_vec()),
                lsn: 1,
            }
            .encode()
            .unwrap(),
        });
        wal.append_segment(&segment).await.unwrap();
        let mut manifest = Manifest::empty(Epoch::ZERO, Uuid::now_v7());
        manifest.wal_segments.push(WalSegmentDescriptor {
            seq: 1,
            path: paths.wal_segment(1).as_ref().to_string(),
            last_lsn: 1,
            xxh3: None,
        });

        let legacy = LegacySnapshotV1 {
            version: 1,
            last_lsn: 1,
            entries: vec![MemtableSnapshotEntry {
                key: key.clone(),
                lsn: 1,
                op: WalOp::Upsert(b"legacy-cache".to_vec()),
            }],
        };
        store
            .put(
                &paths.memtable_snapshot(),
                PutPayload::from(bincode::serialize(&legacy).unwrap()),
            )
            .await
            .unwrap();

        let recovered = recover_memtable_with_snapshot(&manifest, &wal, Some(&store))
            .await
            .unwrap();
        assert!(!recovered.used_snapshot);
        assert_eq!(recovered.records_replayed, 1);
        assert_eq!(
            recovered.memtable.get(&key).unwrap().op,
            MemOp::Upsert(Bytes::from_static(b"wal-v2-reader"))
        );
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
            xxh3: None,
        });

        let out = recover_memtable(&manifest, &wal).await.unwrap();
        assert!(!out.used_snapshot);
        assert_eq!(out.records_replayed, 1);
        assert_eq!(out.max_lsn, 1);
    }
}
