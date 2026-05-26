//! Write side of the storage engine.
//!
//! A [`WriterSession`] owns the in-process state a single writer needs
//! to durably ingest mutations: WAL sequencing, batch building,
//! manifest CAS bumps, memtable application, snapshot handout. It is
//! the orchestration counterpart to [`crate::flush`] (memtable → SSTs)
//! and [`crate::recovery`] (WAL → memtable on cold start) and
//! [`crate::read::Snapshot`] (manifest-pinned query view).
//!
//! ## Loop
//!
//! 1. [`WriterSession::open`] either bootstraps the namespace or claims
//! it (bumping the epoch so any prior writer is fenced). It then
//! runs [`crate::recovery::recover_memtable`] against the current
//! manifest's WAL refs and seeds the in-process LSN + WAL-seq
//! counters from there.
//! 2. `upsert_node` / `tombstone_node` / `upsert_edge` / `tombstone_edge`
//! allocate a fresh LSN and append a [`WalEntry`] to the pending
//! batch in memory. These methods do NOT touch object storage and
//! do NOT mutate the memtable yet.
//! 3. [`WriterSession::commit_batch`] is the durability boundary:
//! a. Seal the pending WAL segment and PUT it with `PutMode::Create`.
//! b. Commit a new manifest version that records the segment under
//! `wal_segments` (CAS).
//! c. Only AFTER both succeed, apply the records to the live
//! memtable. ACK to the caller.
//! On any failure before the manifest CAS lands, the memtable
//! remains untouched and the caller can retry (the pending batch
//! is also preserved on PUT failure).
//! 4. [`WriterSession::flush`] runs [`crate::flush::flush`] against
//! the frozen memtable; on success it clears the WAL refs from
//! the new manifest version (existing behaviour) so future opens
//! won't replay obsolete WAL data.
//! 5. [`WriterSession::snapshot`] hands out a [`Snapshot<'_>`]
//! borrowing the live memtable and pinning the current manifest.
//!
//! ## What's deliberately not here
//!
//! - Auto-batching by time/size threshold. Callers explicitly invoke
//! `commit_batch`; auto-flush land later when the query layer
//! surfaces real workload patterns.
//! - Concurrent writer handover beyond `claim_writer`'s epoch bump.
//! Two simultaneous `open` calls each fence the other but the
//! resulting "ABA" dance is documented as a follow-up; for the
//! single-writer model it's correct.
//! - Background flush / compaction. The caller drives them manually.

use std::sync::Arc;

use bytes::Bytes;
use object_store::ObjectStore;
use tracing::{debug, instrument};
use uuid::Uuid;

use namidb_core::{NodeId, Schema};

use crate::adjacency::{adjacency_budget_bytes, adjacency_enabled, AdjacencyCache};
use crate::cache::{sst_cache_budget_bytes, sst_cache_enabled, SstCache};
use crate::compact::{compact_l0_to_l1, CompactionOutcome};
use crate::error::{Error, Result};
use crate::fence::WriterFence;
use crate::flush::{flush, EdgeWriteRecord, FlushOutcome, NodeWriteRecord};
use crate::manifest::{LoadedManifest, ManifestStore, WalSegmentDescriptor};
use crate::memtable::{MemKey, MemOp, Memtable, MemtableSnapshot};
use crate::node_cache::{node_cache_budget_bytes, node_cache_enabled, NodeViewCache};
use crate::paths::NamespacePaths;
use crate::read::Snapshot;
use crate::recovery::{recover_memtable, WalEntry, WalOp};
use crate::wal::{WalRecord, WalSegment, WalStore};

/// Outcome of [`WriterSession::commit_batch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitOutcome {
    /// Pending batch was empty; nothing was written.
    Empty,
    /// Segment durable + manifest committed + memtable applied.
    Committed {
        wal_seq: u64,
        last_lsn: u64,
        records: usize,
        manifest_version: u64,
    },
}

/// Single-writer ingest session.
pub struct WriterSession {
    manifest_store: ManifestStore,
    wal_store: WalStore,
    fence: WriterFence,
    current: LoadedManifest,
    memtable: Memtable,
    /// Immutable snapshot of `memtable` taken at the last successful
    /// commit / flush / open. Readers consume it via `Arc` so multiple
    /// concurrent snapshots share the same allocation without locking
    /// the writer (RFC-021). Refreshed by [`Self::refresh_published`].
    published_memtable: Arc<MemtableSnapshot>,
    next_lsn: u64,
    next_wal_seq: u64,
    pending: WalSegment,
    pending_payloads: Vec<(MemKey, u64, MemOp)>,
    /// CSR adjacency cache shared across every `Snapshot` this writer
    /// emits (RFC-018). `Some` when `NAMIDB_ADJACENCY=1` at
    /// `open` time; otherwise `None` and edge lookups walk the legacy
    /// SST path. The cache is `Arc`-shared so query bursts amortise
    /// the per-`(manifest_version, edge_type, direction)` build cost.
    adjacency_cache: Option<Arc<AdjacencyCache>>,
    /// Cross-snapshot NodeView cache (RFC-019). `Some` when
    /// `NAMIDB_NODE_CACHE=1` at `open` time. Attached to every
    /// `Snapshot` this writer emits; the 3-tier `lookup_node` consults
    /// it between the per-snapshot L1 and the L3 SST walk.
    node_cache: Option<Arc<NodeViewCache>>,
    /// Process-wide [`SstCache`]. Default ON since the cache
    /// now also stores decoded edge property streams, which IC07 at SF1
    /// surfaced as the dominant per-call cost of `edge_lookup_via_sst`.
    /// Set `NAMIDB_SST_CACHE=0` to disable.
    sst_cache: Option<SstCache>,
    /// Cross-snapshot lazy index over `(label, property) → value → NodeId`
    /// (RFC-pending). Always constructed (cheap empty map); the
    /// per-snapshot `Snapshot::lookup_node_by_property` populates it on
    /// the first miss and reuses it from every subsequent snapshot.
    /// Reset on `flush` because a flush bumps the manifest version and
    /// can introduce new nodes.
    property_index_cache: Arc<crate::property_index::PropertyIndexCache>,
}

impl std::fmt::Debug for WriterSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriterSession")
            .field("manifest_version", &self.current.manifest.version)
            .field("epoch", &self.fence.epoch)
            .field("next_lsn", &self.next_lsn)
            .field("next_wal_seq", &self.next_wal_seq)
            .field("pending_records", &self.pending.records.len())
            .field("memtable_entries", &self.memtable.len())
            .finish()
    }
}

impl WriterSession {
    /// Open or claim a namespace. If the namespace's manifest pointer
    /// does not yet exist, bootstrap it; otherwise bump the epoch via
    /// `claim_writer` to fence any prior writer. After either path,
    /// replay the WAL segments the manifest references and seed
    /// counters so the next allocated LSN follows the last durable one.
    #[instrument(skip(store, paths), fields(namespace = %paths.namespace()))]
    pub async fn open(store: Arc<dyn ObjectStore>, paths: NamespacePaths) -> Result<Self> {
        let manifest_store = ManifestStore::new(store.clone(), paths.clone());
        let wal_store = WalStore::new(store, paths);

        let (current, fence) = match manifest_store.load_current().await {
            Ok(_) => manifest_store.claim_writer().await?,
            Err(Error::ObjectStore(object_store::Error::NotFound { .. })) => {
                let writer_id = Uuid::now_v7();
                let loaded = manifest_store.bootstrap(writer_id).await?;
                let fence = WriterFence::new(loaded.manifest.epoch);
                (loaded, fence)
            }
            Err(other) => return Err(other),
        };

        let recovered = recover_memtable(&current.manifest, &wal_store).await?;
        let next_lsn = recovered.max_lsn.saturating_add(1).max(1);

        // Pick a WAL seq strictly greater than every segment we can see
        // in the object store, not just those the manifest references.
        // Orphan segments (PUT but never committed) must not be
        // re-used because `PutMode::Create` would still refuse them.
        let listed = wal_store.list_segments().await?;
        let next_wal_seq = listed.last().map(|r| r.seq.saturating_add(1)).unwrap_or(1);

        let adjacency_cache = if adjacency_enabled() {
            Some(Arc::new(AdjacencyCache::new(adjacency_budget_bytes())))
        } else {
            None
        };
        let node_cache = if node_cache_enabled() {
            Some(Arc::new(NodeViewCache::new(node_cache_budget_bytes())))
        } else {
            None
        };
        let sst_cache = sst_cache_enabled().then(|| SstCache::new(sst_cache_budget_bytes()));

        let published_memtable = Arc::new(recovered.memtable.snapshot_view());
        Ok(Self {
            manifest_store,
            wal_store,
            fence,
            current,
            memtable: recovered.memtable,
            published_memtable,
            next_lsn,
            next_wal_seq,
            pending: WalSegment::new(next_wal_seq),
            pending_payloads: Vec::new(),
            adjacency_cache,
            node_cache,
            sst_cache,
            property_index_cache: Arc::new(crate::property_index::PropertyIndexCache::new()),
        })
    }

    /// Refresh [`Self::published_memtable`] from the current `memtable`.
    /// Called by `commit_batch` / `flush` after a successful CAS so
    /// readers picking up a fresh snapshot see the newly-durable
    /// records. O(memtable_size) for the BTreeMap clone.
    fn refresh_published(&mut self) {
        self.published_memtable = Arc::new(self.memtable.snapshot_view());
    }

    /// Cross-snapshot lazy property index. Hand it to every `Snapshot`
    /// the writer emits (`Snapshot::with_property_index_cache`) so
    /// warm-path `lookup_node_by_property` calls hit the same `HashMap`.
    pub fn property_index_cache(&self) -> &Arc<crate::property_index::PropertyIndexCache> {
        &self.property_index_cache
    }

    /// Adjacency cache attached to this writer (RFC-018). `None`
    /// when `NAMIDB_ADJACENCY` was not set at `open` time. Exposed so
    /// tests can probe hit/miss/build counters and assert that the CSR
    /// path actually ran.
    pub fn adjacency_cache(&self) -> Option<&Arc<AdjacencyCache>> {
        self.adjacency_cache.as_ref()
    }

    /// Cross-snapshot NodeView cache attached to this writer (RFC-019).
    /// `None` when `NAMIDB_NODE_CACHE` was not set at `open` time.
    /// Exposed for tests/observability — hit/miss/insert stats.
    pub fn node_cache(&self) -> Option<&Arc<NodeViewCache>> {
        self.node_cache.as_ref()
    }

    /// Process-wide SST body / metadata / edge-stream cache attached to
    /// this writer. `None` when `NAMIDB_SST_CACHE=0` at `open` time.
    /// Exposed for tests/observability.
    pub fn sst_cache(&self) -> Option<&SstCache> {
        self.sst_cache.as_ref()
    }

    /// LSN the next mutation will be assigned.
    pub fn next_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// Current manifest version visible to this writer.
    pub fn manifest_version(&self) -> u64 {
        self.current.manifest.version
    }

    /// Number of mutations queued and not yet durable.
    pub fn pending_len(&self) -> usize {
        self.pending.records.len()
    }

    /// Every edge type known to this writer — declared in the manifest
    /// schema, present in the current memtable, or persisted in at
    /// least one SST descriptor. Used by the query executor's
    /// `DETACH DELETE` to enumerate incident edges across types.
    pub fn observed_edge_types(&self) -> Vec<String> {
        use std::collections::BTreeSet;

        let mut set: BTreeSet<String> = self
            .current
            .manifest
            .schema
            .edge_types
            .keys()
            .cloned()
            .collect();
        for (key, _) in self.memtable.iter() {
            if let crate::memtable::MemKey::Edge { edge_type, .. } = key {
                set.insert(edge_type.clone());
            }
        }
        for sst in &self.current.manifest.ssts {
            if matches!(
                sst.kind,
                crate::manifest::SstKind::EdgesFwd | crate::manifest::SstKind::EdgesInv
            ) {
                set.insert(sst.scope.clone());
            }
        }
        set.into_iter().collect()
    }

    /// Build an [`OwnedSnapshot`] pointing at the same published
    /// memtable + manifest [`Self::snapshot`] uses, but without a
    /// borrowed lifetime. Wrap in `Arc` and hand to a
    /// [`SnapshotCell`] so concurrent readers consume it without
    /// holding the writer mutex (RFC-021).
    pub fn owned_snapshot(&self) -> Arc<crate::read::OwnedSnapshot> {
        Arc::new(crate::read::OwnedSnapshot {
            manifest: self.current.clone(),
            memtable: Arc::clone(&self.published_memtable),
            store: self.manifest_store.store().clone(),
            paths: self.manifest_store.paths().clone(),
            cache: self.sst_cache.clone(),
            ranged_mode: crate::read::RangedMode::Auto,
            ranged_threshold_bytes: crate::read::DEFAULT_RANGED_THRESHOLD_BYTES,
            adjacency_cache: self.adjacency_cache.clone(),
            shared_node_cache: self.node_cache.clone(),
            property_index_cache: Some(self.property_index_cache.clone()),
        })
    }

    /// Snapshot view of the namespace as of the last successful
    /// [`commit_batch`] / [`flush`] / [`open`]. The snapshot does NOT
    /// see records that have only been queued via `upsert_*` /
    /// `tombstone_*` (they aren't durable yet).
    pub fn snapshot(&self) -> Snapshot<'_> {
        let mut snap = Snapshot::new(
            self.current.clone(),
            &self.published_memtable,
            self.manifest_store.store().clone(),
            self.manifest_store.paths().clone(),
        );
        if let Some(cache) = &self.sst_cache {
            snap = snap.with_cache(cache.clone());
        }
        if let Some(cache) = &self.adjacency_cache {
            snap = snap.with_adjacency_cache(cache.clone());
        }
        if let Some(cache) = &self.node_cache {
            snap = snap.with_shared_node_cache(cache.clone());
        }
        snap = snap.with_property_index_cache(self.property_index_cache.clone());
        snap
    }

    /// Queue a node upsert. Allocates an LSN and appends the entry to
    /// the pending WAL batch. Returns the LSN.
    pub fn upsert_node(
        &mut self,
        label: impl Into<String>,
        id: NodeId,
        record: &NodeWriteRecord,
    ) -> Result<u64> {
        let lsn = self.alloc_lsn();
        let key = MemKey::Node {
            label: label.into(),
            id,
        };
        let payload = record.encode()?;
        let entry = WalEntry {
            key: key.clone(),
            op: WalOp::Upsert(payload.to_vec()),
            lsn,
        };
        self.append_pending(entry, MemOp::Upsert(payload), lsn, key)?;
        Ok(lsn)
    }

    /// Queue a node tombstone. Returns the LSN.
    pub fn tombstone_node(&mut self, label: impl Into<String>, id: NodeId) -> Result<u64> {
        let lsn = self.alloc_lsn();
        let key = MemKey::Node {
            label: label.into(),
            id,
        };
        let entry = WalEntry {
            key: key.clone(),
            op: WalOp::Tombstone,
            lsn,
        };
        self.append_pending(entry, MemOp::Tombstone, lsn, key)?;
        Ok(lsn)
    }

    /// Queue an edge upsert. Returns the LSN.
    pub fn upsert_edge(
        &mut self,
        edge_type: impl Into<String>,
        src: NodeId,
        dst: NodeId,
        record: &EdgeWriteRecord,
    ) -> Result<u64> {
        let lsn = self.alloc_lsn();
        let key = MemKey::Edge {
            edge_type: edge_type.into(),
            src,
            dst,
        };
        let payload = record.encode()?;
        let entry = WalEntry {
            key: key.clone(),
            op: WalOp::Upsert(payload.to_vec()),
            lsn,
        };
        self.append_pending(entry, MemOp::Upsert(payload), lsn, key)?;
        Ok(lsn)
    }

    /// Queue an edge tombstone. Returns the LSN.
    pub fn tombstone_edge(
        &mut self,
        edge_type: impl Into<String>,
        src: NodeId,
        dst: NodeId,
    ) -> Result<u64> {
        let lsn = self.alloc_lsn();
        let key = MemKey::Edge {
            edge_type: edge_type.into(),
            src,
            dst,
        };
        let entry = WalEntry {
            key: key.clone(),
            op: WalOp::Tombstone,
            lsn,
        };
        self.append_pending(entry, MemOp::Tombstone, lsn, key)?;
        Ok(lsn)
    }

    /// Durability boundary: seal the pending batch into a WAL segment,
    /// PUT it, CAS a new manifest version that references it, and
    /// only THEN apply the records to the live memtable.
    ///
    /// ## Cadence trade-off
    ///
    /// Each call costs **two object-store round-trips**: one WAL PUT and
    /// one manifest CAS PUT. On loopback / in-memory stores that's
    /// invisible; on real S3 it's the dominant cost of an ingest loop.
    /// Measured against Cloudflare R2 from a laptop the round-trip is
    /// ~750 ms; against same-region EC2 it's ~5–15 ms.
    ///
    /// The caller decides how often to invoke this:
    ///
    /// - **Tight cadence** (e.g. one commit per 1 K rows) → small loss
    /// window if the writer crashes (~1 K records of pending work
    /// re-issued on the next `WriterSession::open`), high network
    /// overhead on slow links.
    /// - **Loose cadence** (e.g. one commit per 100 K rows) → 100 K-row
    /// loss window, ~100 × less network overhead.
    ///
    /// As a calibration point: against R2 from a laptop, 1 M nodes with
    /// `commit_batch` every 10 K rows yields ~6.6 K elem/s (100
    /// round-trips × 750 ms = 75 s of network alone). The same workload
    /// with `commit_batch` every 100 K rows clears 10 K elem/s.
    ///
    /// The engine does not pick a cadence for you. A bulk loader should
    /// commit at memtable-flush boundaries (or larger); an interactive
    /// writer that needs read-your-writes durability should commit
    /// more frequently and pay the round-trips.
    ///
    /// ## Failure modes
    ///
    /// - PUT failure / WAL seq collision → returns the error; the
    /// pending batch is preserved and the writer can retry. The
    /// pending WAL `seq` is unchanged so a retry hits the same
    /// object path; a successful retry by another writer with the
    /// same `seq` is impossible because [`crate::wal::WalStore::append_segment`]
    /// uses `PutMode::Create`.
    /// - Manifest CAS loss → returns `ManifestCommitCas`. The segment
    /// is durable in object storage but the manifest does not yet
    /// reference it. Caller must reload the manifest and retry; a
    /// later `claim_writer` either fences this session (in which
    /// case the orphan segment is collected by the janitor) or
    /// succeeds at which point the segment becomes reachable again.
    /// For the single-writer model the simpler answer is for
    /// the caller to drop the session.
    #[instrument(skip(self), fields(
 manifest_version = self.current.manifest.version,
 pending = self.pending.records.len(),
 ))]
    pub async fn commit_batch(&mut self) -> Result<CommitOutcome> {
        if self.pending.is_empty() {
            return Ok(CommitOutcome::Empty);
        }

        let seq = self.pending.seq;
        let last_lsn = self.pending.last_lsn();
        let records = self.pending.records.len();

        // The WAL segment path is fully determined by `seq`, so we can
        // build the next manifest body before the WAL PUT lands and
        // pipeline the two writes. That turns the per-commit critical
        // path from `WAL + manifest body + pointer CAS` (three round
        // trips) into `max(WAL, manifest body) + pointer CAS` (two).
        // If the WAL append fails, the body PUT is harmless: the
        // pointer never moves, the next manifest commit overwrites
        // the reference, and the janitor sweeps the orphan.
        let segment_path = self.wal_store.paths().wal_segment(seq);
        let mut next = self.current.manifest.next_version(self.fence.writer_id);
        next.wal_segments.push(WalSegmentDescriptor {
            seq,
            path: segment_path.as_ref().to_string(),
            last_lsn,
        });

        let (wal_result, body_result) = tokio::join!(
            self.wal_store.append_segment(&self.pending),
            self.manifest_store
                .put_body(&self.fence, &self.current, &next),
        );
        let _wal_path = wal_result?;
        let pointer = body_result?;
        let new_current = self
            .manifest_store
            .cas_pointer(&self.fence, &self.current, next, pointer)
            .await?;

        // Durability achieved. Drain the queued payloads into the
        // memtable in LSN order (they are already in insertion order
        // because each `append_pending` call appends).
        let drained = std::mem::take(&mut self.pending_payloads);
        for (key, lsn, op) in drained {
            self.memtable.apply(key, lsn, op);
        }
        self.pending = WalSegment::new(seq.saturating_add(1));

        self.current = new_current;
        self.next_wal_seq = seq.saturating_add(1);
        // Publish the new memtable snapshot so subsequent reads (HTTP,
        // Bolt, embedded) pick up the just-committed records without
        // taking the writer lock. See RFC-021.
        self.refresh_published();

        debug!(
            wal_seq = seq,
            last_lsn,
            manifest_version = self.current.manifest.version,
            "commit_batch sealed"
        );

        Ok(CommitOutcome::Committed {
            wal_seq: seq,
            last_lsn,
            records,
            manifest_version: self.current.manifest.version,
        })
    }

    /// Compact every `(kind, scope)` bucket in L0 that holds more than
    /// one SST into a single L1 SST. No-op if every bucket has ≤1 SST.
    /// Does NOT touch the memtable or the pending batch.
    #[instrument(skip(self, schema), fields(
 manifest_version = self.current.manifest.version,
 ))]
    pub async fn compact_l0(&mut self, schema: &Schema) -> Result<CompactionOutcome> {
        let outcome =
            compact_l0_to_l1(&self.manifest_store, &self.fence, &self.current, schema).await?;
        self.current = outcome.committed.clone();
        Ok(outcome)
    }

    /// Flush the live memtable iff the current manifest already references
    /// `max_wal_segments` or more WAL segments. No-op below the threshold.
    /// Returns `true` when a flush ran.
    ///
    /// Callers (the cloud worker, the bench loaders) use this after each
    /// `commit_batch` to keep the cold-start cost bounded — without it,
    /// every commit appends a WAL segment to the manifest forever and
    /// `recover_memtable` re-replays the entire history on the next mount.
    /// The engine does NOT auto-flush inside `commit_batch` itself because
    /// some workloads (single-shot LDBC bulk load) prefer to batch the
    /// flush at the end and skip the intermediate SSTs.
    ///
    /// The schema is taken from the current manifest; pass `flush(schema)`
    /// explicitly if you need to flush with a *different* schema version.
    #[instrument(skip(self), fields(
 manifest_version = self.current.manifest.version,
 wal_segments = self.current.manifest.wal_segments.len(),
 ))]
    pub async fn maybe_flush(&mut self, max_wal_segments: usize) -> Result<bool> {
        // `0` is the sentinel for "auto-flush disabled" so the caller can
        // express that in config without a `None`-wrapping ceremony.
        if max_wal_segments == 0 || self.current.manifest.wal_segments.len() < max_wal_segments {
            return Ok(false);
        }
        let schema = self.current.manifest.schema.clone();
        self.flush(schema).await?;
        Ok(true)
    }

    /// Freeze the live memtable and run the SST flush path.
    /// Implicitly commits any pending batch first so the caller doesn't
    /// strand records mid-flush.
    #[instrument(skip(self, schema), fields(
 manifest_version = self.current.manifest.version,
 ))]
    pub async fn flush(&mut self, schema: Schema) -> Result<FlushOutcome> {
        let _ = self.commit_batch().await?;
        let frozen = self.memtable.freeze();
        let outcome = flush(
            &self.manifest_store,
            &self.fence,
            &self.current,
            &frozen,
            schema,
        )
        .await?;
        self.current = outcome.committed.clone();
        // The flush emptied the live memtable (memtable.freeze() drained
        // it), so the published snapshot must reset to empty too.
        self.refresh_published();
        // Invalidate the cross-snapshot property index — a flush can
        // promote new nodes from the memtable into SSTs, and the cached
        // value→NodeId map is built off a snapshot that pre-dates the
        // new manifest version. Subsequent snapshots will rebuild on
        // their first miss.
        self.property_index_cache.reset();
        Ok(outcome)
    }

    fn alloc_lsn(&mut self) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn = self.next_lsn.saturating_add(1);
        lsn
    }

    fn append_pending(&mut self, entry: WalEntry, op: MemOp, lsn: u64, key: MemKey) -> Result<()> {
        let payload = entry.encode()?;
        self.pending.push(WalRecord { lsn, payload });
        self.pending_payloads.push((key, lsn, op));
        Ok(())
    }
}

/// Convenience: the WAL payload bytes a `WalEntry` produces. Useful
/// for diagnostics or external WAL consumers; the writer itself
/// emits these inside `commit_batch`.
pub fn encode_wal_entry(entry: &WalEntry) -> Result<Bytes> {
    entry.encode()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use namidb_core::{
        DataType, EdgeTypeDef, LabelDef, NamespaceId, PropertyDef, SchemaBuilder, Value,
    };
    use object_store::memory::InMemory;

    use super::*;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn make_paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    fn person_label() -> LabelDef {
        LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("age", DataType::Int32, true).unwrap(),
            ],
        }
    }

    fn knows_edge() -> EdgeTypeDef {
        EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![],
        }
    }

    fn sorted_node_id(b: u8) -> NodeId {
        let mut bytes = [0u8; 16];
        bytes[15] = b;
        NodeId::from_uuid(Uuid::from_bytes(bytes))
    }

    fn node_record(name: &str, age: Option<i32>) -> NodeWriteRecord {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        if let Some(a) = age {
            props.insert("age".into(), Value::I64(a as i64));
        }
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
        }
    }

    fn edge_record() -> EdgeWriteRecord {
        EdgeWriteRecord {
            properties: BTreeMap::new(),
            schema_version: 1,
        }
    }

    fn schema() -> Schema {
        SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build()
    }

    #[tokio::test]
    async fn open_bootstraps_fresh_namespace() {
        let store = make_store();
        let paths = make_paths("ingest-bootstrap");
        let session = WriterSession::open(store, paths).await.unwrap();
        assert_eq!(session.manifest_version(), 0);
        assert_eq!(session.next_lsn(), 1);
        assert_eq!(session.pending_len(), 0);
    }

    #[tokio::test]
    async fn upsert_then_commit_makes_data_visible_via_snapshot() {
        let store = make_store();
        let paths = make_paths("ingest-upsert");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        let alice = sorted_node_id(1);
        let lsn = session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        assert_eq!(lsn, 1);
        assert_eq!(session.pending_len(), 1);

        // Snapshot BEFORE commit: not visible (the queued record lives
        // only inside pending, not in the memtable).
        let pre = session.snapshot();
        assert!(pre.lookup_node("Person", alice).await.unwrap().is_none());
        drop(pre);

        let outcome = session.commit_batch().await.unwrap();
        match outcome {
            CommitOutcome::Committed {
                wal_seq,
                last_lsn,
                records,
                manifest_version,
            } => {
                assert_eq!(wal_seq, 1);
                assert_eq!(last_lsn, 1);
                assert_eq!(records, 1);
                assert_eq!(manifest_version, 1);
            }
            other => panic!("expected Committed, got {other:?}"),
        }

        let post = session.snapshot();
        let view = post.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(view.lsn, 1);
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
    }

    #[tokio::test]
    async fn empty_commit_batch_is_noop() {
        let store = make_store();
        let paths = make_paths("ingest-empty");
        let mut session = WriterSession::open(store, paths).await.unwrap();
        let out = session.commit_batch().await.unwrap();
        assert_eq!(out, CommitOutcome::Empty);
        assert_eq!(session.manifest_version(), 0);
    }

    #[tokio::test]
    async fn multiple_operations_in_single_batch_apply_atomically() {
        let store = make_store();
        let paths = make_paths("ingest-multi");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);

        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        session
            .upsert_node("Person", bob, &node_record("Bob", None))
            .unwrap();
        session
            .upsert_edge("KNOWS", alice, bob, &edge_record())
            .unwrap();
        session.tombstone_node("Person", alice).unwrap();

        assert_eq!(session.pending_len(), 4);
        let out = session.commit_batch().await.unwrap();
        match out {
            CommitOutcome::Committed {
                records, last_lsn, ..
            } => {
                assert_eq!(records, 4);
                assert_eq!(last_lsn, 4);
            }
            other => panic!("expected Committed, got {other:?}"),
        }

        let snap = session.snapshot();
        // Alice was tombstoned at the highest LSN inside the batch.
        assert!(snap.lookup_node("Person", alice).await.unwrap().is_none());
        // Bob is visible.
        let bob_view = snap.lookup_node("Person", bob).await.unwrap().unwrap();
        assert_eq!(
            bob_view.properties.get("name"),
            Some(&Value::Str("Bob".into()))
        );
        // The edge survives (its src tombstone affects node-side lookups,
        // not the edge SST itself).
        let out_edges = snap.out_edges("KNOWS", alice).await.unwrap();
        assert_eq!(out_edges.edges.len(), 1);
        assert_eq!(out_edges.edges[0].dst, bob);
    }

    #[tokio::test]
    async fn flush_durably_promotes_pending_then_committed_records() {
        let store = make_store();
        let paths = make_paths("ingest-flush");
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();

        let alice = sorted_node_id(1);
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        // Note: skip the explicit commit_batch — `flush` should fold any
        // pending records in first.
        let outcome = session.flush(schema()).await.unwrap();
        assert_eq!(outcome.ssts_written, 1);
        assert!(outcome.committed.manifest.wal_segments.is_empty());

        // Now reopen the namespace in a fresh session (cold-start path).
        // The flush cleared the WAL refs, so recovery has nothing to
        // replay and the snapshot still sees Alice via the SST.
        let session2 = WriterSession::open(store, paths).await.unwrap();
        assert_eq!(session2.next_lsn(), 1, "no WAL refs → counter resets");
        let snap = session2.snapshot();
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
    }

    #[tokio::test]
    async fn reopen_replays_uncommitted_wal_segments() {
        let store = make_store();
        let paths = make_paths("ingest-recovery");

        let alice = sorted_node_id(1);
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        let _ = session.commit_batch().await.unwrap();
        // Note: we did NOT flush. The WAL segment is referenced by the
        // manifest but the data is not yet in any SST.
        drop(session);

        let session2 = WriterSession::open(store, paths).await.unwrap();
        // The new session bumped epoch via claim_writer; manifest is at
        // v2 (v1 was the wal_segments commit, v2 is the claim).
        assert!(session2.manifest_version() >= 2);
        assert_eq!(
            session2.next_lsn(),
            2,
            "recovery saw lsn=1, next should be 2"
        );
        let snap = session2.snapshot();
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(view.lsn, 1);
    }

    #[tokio::test]
    async fn wal_seq_skips_over_orphan_segments_on_reopen() {
        // Simulate a writer crashing between WAL PUT and manifest
        // commit by appending a segment directly to the WAL store and
        // never referencing it from the manifest. A fresh
        // WriterSession::open must NOT try to reuse that seq.
        let store = make_store();
        let paths = make_paths("ingest-orphan");

        // Bootstrap so the manifest exists.
        let session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        drop(session);

        // Manually PUT seq=1 to simulate the orphan.
        let wal_store = WalStore::new(store.clone(), paths.clone());
        let mut orphan_seg = WalSegment::new(1);
        orphan_seg.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: MemKey::Node {
                    label: "Person".into(),
                    id: sorted_node_id(99),
                },
                op: WalOp::Upsert(b"ghost".to_vec()),
                lsn: 1,
            }
            .encode()
            .unwrap(),
        });
        wal_store.append_segment(&orphan_seg).await.unwrap();

        // Reopen: the new session must claim seq=2, not 1.
        let mut session = WriterSession::open(store, paths).await.unwrap();
        let alice = sorted_node_id(1);
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        let out = session.commit_batch().await.unwrap();
        match out {
            CommitOutcome::Committed { wal_seq, .. } => {
                assert_eq!(wal_seq, 2, "must skip over orphan seq=1");
            }
            other => panic!("expected Committed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn compact_l0_after_two_flushes_collapses_to_single_l1() {
        let store = make_store();
        let paths = make_paths("ingest-compact");
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);

        // Flush #1: alice.
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        let _ = session.flush(schema()).await.unwrap();

        // Flush #2: bob.
        session
            .upsert_node("Person", bob, &node_record("Bob", None))
            .unwrap();
        let _ = session.flush(schema()).await.unwrap();
        assert_eq!(session.current.manifest.ssts.len(), 2);

        // Compaction collapses the two L0 nodes SSTs into one L1.
        let outcome = session.compact_l0(&schema()).await.unwrap();
        assert_eq!(outcome.source_ssts_removed, 2);
        assert_eq!(outcome.new_ssts_written, 1);
        assert_eq!(session.current.manifest.ssts.len(), 1);
        assert_eq!(
            session.current.manifest.ssts[0].level,
            crate::manifest::SstLevel(1)
        );

        // Snapshot still resolves both rows correctly.
        let snap = session.snapshot();
        let v_alice = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(
            v_alice.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
        let v_bob = snap.lookup_node("Person", bob).await.unwrap().unwrap();
        assert_eq!(
            v_bob.properties.get("name"),
            Some(&Value::Str("Bob".into()))
        );
    }

    #[tokio::test]
    async fn maybe_flush_is_a_noop_below_threshold() {
        let store = make_store();
        let paths = make_paths("ingest-maybe-flush-below");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        // One commit → one wal_segment in the manifest.
        session
            .upsert_node("Person", sorted_node_id(1), &node_record("Alice", Some(30)))
            .unwrap();
        session.commit_batch().await.unwrap();
        assert_eq!(session.current.manifest.wal_segments.len(), 1);

        // Threshold 4 > 1 → no-op, no SST flush.
        let flushed = session.maybe_flush(4).await.unwrap();
        assert!(!flushed);
        assert_eq!(session.current.manifest.wal_segments.len(), 1);
        assert_eq!(session.current.manifest.ssts.len(), 0);

        // Threshold 0 → "disabled", still a no-op even when the manifest
        // has segments. Lets callers express "off" without an `Option`.
        let flushed = session.maybe_flush(0).await.unwrap();
        assert!(!flushed);
        assert_eq!(session.current.manifest.wal_segments.len(), 1);
    }

    #[tokio::test]
    async fn maybe_flush_truncates_wal_when_threshold_crossed() {
        // N1 regression: without this, every commit_batch leaves a wal
        // segment in the manifest forever and cold-start replays them
        // all on the next mount. Verify maybe_flush clears them once a
        // configurable threshold is crossed.
        let store = make_store();
        let paths = make_paths("ingest-maybe-flush-threshold");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        for i in 0..3 {
            session
                .upsert_node(
                    "Person",
                    sorted_node_id(i),
                    &node_record(&format!("p{}", i), Some(20 + i as i32)),
                )
                .unwrap();
            session.commit_batch().await.unwrap();
        }
        assert_eq!(session.current.manifest.wal_segments.len(), 3);
        assert_eq!(session.current.manifest.ssts.len(), 0);

        let flushed = session.maybe_flush(3).await.unwrap();
        assert!(flushed);
        // Flush retired the in-flight WAL into SSTs and cleared the
        // segment list — cold-start now just opens the manifest, no
        // replay needed.
        assert_eq!(session.current.manifest.wal_segments.len(), 0);
        assert!(!session.current.manifest.ssts.is_empty());
    }

    #[tokio::test]
    async fn second_open_fences_first() {
        let store = make_store();
        let paths = make_paths("ingest-fence");

        let mut session_a = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let session_b = WriterSession::open(store, paths).await.unwrap();
        // The fresh session_b bumped the epoch; session_a's fence is
        // now stale and any commit_batch must fail with Fenced.
        let alice = sorted_node_id(1);
        session_a
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        let err = session_a.commit_batch().await.unwrap_err();
        match err {
            Error::Fenced { mine, current } => {
                assert!(mine < current);
            }
            other => panic!("expected Fenced, got {other:?}"),
        }
        // session_b can still ingest cleanly.
        drop(session_b);
    }
}
