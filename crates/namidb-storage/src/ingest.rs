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

use namidb_core::{
    Constraint, ConstraintKind, DataType, LabelDef, LabelDictionary, NodeId, PropertyDef, Schema,
    Value,
};

use crate::adjacency::{adjacency_budget_bytes, adjacency_enabled, AdjacencyCache};
use crate::cache::{sst_cache_budget_bytes, sst_cache_enabled, SstCache};
use crate::compact::{compact_l0_to_l1, CompactionOutcome};
use crate::error::{Error, Result};
use crate::fence::WriterFence;
use crate::flush::{flush, EdgeWriteRecord, FlushOutcome, NodeWriteRecord};
use crate::manifest::{
    LoadedManifest, Manifest, ManifestStore, SstKind, SstLevel, TextIndexDescriptor,
    VectorIndexDescriptor, WalSegmentDescriptor,
};
use crate::memtable::{MemKey, MemOp, Memtable, MemtableSnapshot};
use crate::node_cache::{node_cache_budget_bytes, node_cache_enabled, NodeViewCache};
use crate::paths::NamespacePaths;
use crate::read::Snapshot;
use crate::recovery::{
    recover_memtable_with_snapshot, write_memtable_snapshot, MemtableSnapshotFile, WalEntry, WalOp,
};
use crate::wal::{WalRecord, WalSegment, WalStore};

/// Default number of commits between automatic memtable snapshots.
/// Zero disables auto-snapshotting; callers can still drive it manually
/// via [`WriterSession::write_memtable_snapshot_now`].
const DEFAULT_AUTO_SNAPSHOT_EVERY: u64 = 0;

/// Parse `NAMIDB_MEMTABLE_SNAPSHOT_EVERY` for the auto-snapshot cadence.
/// `0` (or unset) disables; any positive value `N` snapshots after
/// every `N` successful commits.
fn auto_snapshot_every() -> u64 {
    std::env::var("NAMIDB_MEMTABLE_SNAPSHOT_EVERY")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_AUTO_SNAPSHOT_EVERY)
}

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
    /// Object store handle kept around so [`Self::commit_batch`] can
    /// fire the auto-snapshot path without re-deriving it from the
    /// manifest store. Same `Arc` we hand `Snapshot::new`.
    store: Arc<dyn ObjectStore>,
    /// Auto-snapshot cadence resolved at `open` time. Zero disables.
    auto_snapshot_every: u64,
    /// Commits ago since the last successful snapshot write. Reset
    /// when a snapshot lands; bumped by every non-empty `commit_batch`.
    commits_since_snapshot: u64,
    /// Set when a `commit_batch` retry hits a terminal error it cannot
    /// resolve in-session (an orphan-WAL collision whose re-seq retry
    /// still loses the manifest CAS). The single-writer contract is to
    /// drop the session and reopen on a terminal commit error; this flag
    /// enforces it in code so a contract-violating re-entry issues no
    /// further object-store writes and mints no new orphan segments.
    /// Only a fresh [`Self::open`] clears it.
    poisoned: bool,
    /// Namespace label dictionary. Seeded from the manifest at `open` and
    /// extended as `upsert_node*` interns new label names; stamped onto every
    /// committed manifest so a node's on-row `LabelId`s always resolve to names.
    label_dict: LabelDictionary,
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
        let wal_store = WalStore::new(store.clone(), paths);

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

        // Cold-start fast path: if a memtable snapshot is present at
        // `paths.memtable_snapshot()`, seed the in-process memtable
        // from it and skip every WAL record it already covers.
        let recovered =
            recover_memtable_with_snapshot(&current.manifest, &wal_store, Some(&store)).await?;
        // Rebase the LSN counter past the highest LSN durably held in any
        // SST, not just what recovery saw in the WAL + memtable snapshot.
        // Once a namespace flushes all its WAL into SSTs, `recovered.max_lsn`
        // drops to 0 — `recover_memtable_with_snapshot` only scans WAL
        // segments and the snapshot, never `manifest.ssts`. Without this
        // max, a cold-reopened all-SST namespace would restart at lsn=1 and
        // the next online write would be silently shadowed by its own older
        // SST row (reads pick the strictly-higher LSN). The SST high-water
        // is the true floor for the next LSN.
        let max_sst_lsn = current
            .manifest
            .ssts
            .iter()
            .map(|sst| sst.max_lsn)
            .max()
            .unwrap_or(0);
        let next_lsn = recovered.max_lsn.max(max_sst_lsn).saturating_add(1).max(1);

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
            label_dict: current.manifest.label_dict.clone(),
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
            store,
            auto_snapshot_every: auto_snapshot_every(),
            commits_since_snapshot: 0,
            poisoned: false,
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

    /// The largest number of L0 SSTs in any single `(kind, scope)` bucket
    /// of the current manifest. A point lookup on a bucket pays an
    /// `O(L0 count)` candidate scan, so this is the worst-case read
    /// amplification right now. The server uses it to trigger compaction
    /// reactively (rather than only on the periodic tick) and to apply a
    /// soft write stall when L0 outpaces compaction (RFC-027 P5).
    pub fn max_l0_bucket_len(&self) -> usize {
        let mut counts: std::collections::HashMap<(SstKind, &str), usize> =
            std::collections::HashMap::new();
        for sst in &self.current.manifest.ssts {
            if sst.level == SstLevel::L0 {
                *counts.entry((sst.kind, sst.scope.as_str())).or_insert(0) += 1;
            }
        }
        counts.values().copied().max().unwrap_or(0)
    }

    /// Number of mutations queued and not yet durable.
    pub fn pending_len(&self) -> usize {
        self.pending.records.len()
    }

    /// Drop the uncommitted batch without making it durable, returning the
    /// number of mutations discarded. Used to roll back an explicit
    /// transaction. Safe because staged writes only touch `pending` /
    /// `pending_payloads` (queued) and never the memtable until
    /// `commit_batch` drains them after the manifest CAS, so there is
    /// nothing durable or in-memory to unwind. The WAL sequence is reused
    /// (the discarded segment was never persisted); a skipped `next_lsn` is
    /// harmless (reads always pick the strictly-higher LSN).
    pub fn discard_batch(&mut self) -> usize {
        let discarded = self.pending.records.len();
        self.pending = WalSegment::new(self.pending.seq);
        self.pending_payloads.clear();
        discarded
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

    /// Read-your-own-writes snapshot (RFC-026): the committed
    /// [`snapshot`](Self::snapshot) with this writer's staged-but-
    /// uncommitted batch overlaid on top, so a read sub-plan that runs
    /// after a `CREATE`/`MERGE`/`SET`/`DELETE` in the same statement or
    /// transaction sees the staged work. Reads outside a write context
    /// keep using [`snapshot`](Self::snapshot); there is nothing staged
    /// for them to see.
    ///
    /// When nothing is staged this is exactly [`snapshot`](Self::snapshot)
    /// (same caches, no overlay), so a write statement's first read before
    /// any mutation pays nothing.
    ///
    /// The overlay is built from `pending_payloads`, so it reflects exactly
    /// what [`commit_batch`](Self::commit_batch) would make durable. The
    /// cross-snapshot NodeView and property-index caches are deliberately
    /// NOT attached: they are keyed by manifest version and shared across
    /// sessions, so caching a staged (uncommitted) row in them would leak
    /// it to a concurrent reader pinned at the same version. The immutable
    /// SST/adjacency body caches are safe to keep.
    pub fn overlay_snapshot(&self) -> Snapshot<'_> {
        if self.pending_payloads.is_empty() {
            return self.snapshot();
        }
        // Materialise the staged batch as a second memtable. `apply` in
        // pending (LSN-ascending) order leaves each key at its highest-LSN
        // op, exactly as `commit_batch` would drain it into the live
        // memtable.
        let mut staged = Memtable::new();
        for (key, lsn, op) in &self.pending_payloads {
            staged.apply(key.clone(), *lsn, op.clone());
        }
        // Resolve labels through the writer's live dictionary, not the
        // committed manifest's: a node staged in this batch may carry a
        // label name first interned in the same batch, which the committed
        // `current.manifest.label_dict` does not know about yet. The live
        // dict is a superset, so committed labels still resolve.
        let mut current = self.current.clone();
        current.manifest.label_dict = self.label_dict.clone();
        let mut snap = Snapshot::new(
            current,
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
        snap.with_overlay(staged.snapshot_view())
    }

    /// Schema of the current manifest version. The write path consults it to
    /// enforce declared constraints (e.g. unique properties) before staging
    /// a mutation.
    pub fn schema(&self) -> &namidb_core::Schema {
        &self.current.manifest.schema
    }

    /// The registered vector indexes (committed manifest). The write path consults
    /// these to enforce embedding dimension at write time: a `register_vector_index`
    /// commits immediately and is never part of a staged batch, so `self.current`
    /// is authoritative when a mutation is applied.
    pub fn vector_indexes(&self) -> &[crate::manifest::VectorIndexDescriptor] {
        &self.current.manifest.vector_indexes
    }

    /// Queue a single-label node upsert. Convenience wrapper over
    /// [`upsert_node_with_labels`](Self::upsert_node_with_labels); kept so the
    /// many single-label call sites stay unchanged.
    pub fn upsert_node(
        &mut self,
        label: impl Into<String>,
        id: NodeId,
        record: &NodeWriteRecord,
    ) -> Result<u64> {
        self.upsert_node_with_labels(std::iter::once(label.into()), id, record)
    }

    /// Queue a node upsert carrying a full label set. Allocates an LSN, interns
    /// every label name into the namespace dictionary, stamps the resulting
    /// (sorted, deduped) [`LabelId`](namidb_core::LabelId) values onto the
    /// record, keys the row by `id` alone, and appends to the pending WAL
    /// batch. Returns the LSN.
    pub fn upsert_node_with_labels<I>(
        &mut self,
        labels: I,
        id: NodeId,
        record: &NodeWriteRecord,
    ) -> Result<u64>
    where
        I: IntoIterator<Item = String>,
    {
        let lsn = self.alloc_lsn();
        let mut label_ids: Vec<u32> = labels
            .into_iter()
            .map(|name| self.label_dict.intern(&name).get())
            .collect();
        label_ids.sort_unstable();
        label_ids.dedup();

        let mut record = record.clone();
        record.labels = label_ids;
        let key = MemKey::Node { id };
        let payload = record.encode()?;
        let entry = WalEntry {
            key: key.clone(),
            op: WalOp::Upsert(payload.to_vec()),
            lsn,
        };
        self.append_pending(entry, MemOp::Upsert(payload), lsn, key)?;
        Ok(lsn)
    }

    /// Queue a node tombstone. Keyed by `id` alone: a tombstone removes the node
    /// from every label scan regardless of which labels it carried, so the
    /// `label` argument is vestigial (kept for call-site compatibility).
    pub fn tombstone_node(&mut self, _label: impl Into<String>, id: NodeId) -> Result<u64> {
        let lsn = self.alloc_lsn();
        let key = MemKey::Node { id };
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
    /// - Transient PUT failure → returns the error; the pending batch is
    /// preserved and the writer can retry. The pending WAL `seq` is
    /// unchanged so a retry hits the same object path.
    /// - WAL seq collision (`PutMode::Create` → `AlreadyExists` →
    /// [`Error::Precondition`]) means an orphan segment already sits at
    /// our `seq`: either our own from a prior attempt whose commit failed
    /// after the WAL PUT, or one injected externally. We re-pick a seq
    /// strictly above every segment now visible and retry the commit
    /// ONCE, body-first. The retry recovers only when the manifest body
    /// at `base+1` is still free (i.e. the prior attempt's body PUT had
    /// failed); otherwise it terminates with `ManifestCommitCas`. A
    /// terminal retry failure poisons the session (see [`Self::open`]).
    /// - Manifest CAS loss → returns `ManifestCommitCas`. The segment is
    /// durable in object storage but the manifest does not yet reference
    /// it; the janitor sweeps the orphan. For the single-writer model the
    /// answer is for the caller to drop the session and reopen, at which
    /// point [`Self::open`]'s defensive `list_segments` picks a fresh seq.
    ///
    /// Callers must treat BOTH `Precondition` and `ManifestCommitCas` from
    /// this method as "drop the session and reopen". No data is lost: the
    /// memtable is untouched until the pointer CAS lands, so a failed
    /// commit never ACKs records it did not make durable and referenced.
    #[instrument(skip(self), fields(
 manifest_version = self.current.manifest.version,
 pending = self.pending.records.len(),
 ))]
    pub async fn commit_batch(&mut self) -> Result<CommitOutcome> {
        // A terminal commit failure poisons the session: the single-writer
        // contract is to drop it and reopen. Enforce that here so a
        // contract-violating retry issues no further object-store writes
        // and mints no new orphan segments.
        if self.poisoned {
            return Err(Error::precondition(
                "writer session poisoned by a prior terminal commit failure; drop and reopen",
            ));
        }
        if self.pending.is_empty() {
            return Ok(CommitOutcome::Empty);
        }

        let base_seq = self.pending.seq;
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
        let next = self.build_next(base_seq, last_lsn);

        let (wal_result, body_result) = tokio::join!(
            self.wal_store.append_segment(&self.pending),
            self.manifest_store
                .put_body(&self.fence, &self.current, &next),
        );

        // Inspect `wal_result` by value WITHOUT `?` so an orphan-WAL
        // `Precondition` routes to the retry instead of propagating.
        let (committed_seq, new_current) = match wal_result {
            Ok(_) => {
                let pointer = body_result?;
                let new_current = self
                    .manifest_store
                    .cas_pointer(&self.fence, &self.current, next, pointer)
                    .await?;
                (base_seq, new_current)
            }
            Err(Error::Precondition(_)) => {
                // Orphan segment already occupies `base_seq`. The body PUT
                // above may or may not have landed; we discard its result
                // and rebuild from scratch at a fresh seq. Re-pick a seq
                // strictly above every segment now visible (same rule as
                // `open`) and retry the commit ONCE, body-first so the
                // common "base+1 already taken" case fails fast as
                // `ManifestCommitCas` without minting a new WAL orphan.
                let _ = body_result;
                let listed = self.wal_store.list_segments().await?;
                let fresh = listed
                    .last()
                    .map(|r| r.seq.saturating_add(1))
                    .unwrap_or(base_seq.saturating_add(1));
                // Re-seq the in-memory segment in place; records are
                // preserved because only `seq` changes.
                self.pending.seq = fresh;
                let next = self.build_next(fresh, last_lsn);
                match self.commit_body_first(next).await {
                    Ok(new_current) => (fresh, new_current),
                    Err(err) => {
                        // Terminal: restore the original seq so a
                        // contract-violating re-entry mints no new orphan,
                        // and poison the session.
                        self.pending.seq = base_seq;
                        self.poisoned = true;
                        return Err(err);
                    }
                }
            }
            Err(other) => return Err(other),
        };

        // Durability achieved. Drain the queued payloads into the
        // memtable in LSN order (they are already in insertion order
        // because each `append_pending` call appends).
        let drained = std::mem::take(&mut self.pending_payloads);
        for (key, lsn, op) in drained {
            self.memtable.apply(key, lsn, op);
        }
        self.pending = WalSegment::new(committed_seq.saturating_add(1));

        self.current = new_current;
        self.next_wal_seq = committed_seq.saturating_add(1);
        // Publish the new memtable snapshot so subsequent reads (HTTP,
        // Bolt, embedded) pick up the just-committed records without
        // taking the writer lock. See RFC-021.
        self.refresh_published();
        // Invalidate the cross-snapshot property index. A commit adds new
        // nodes to the live memtable, but a previously warmed value→NodeId
        // map was frozen against an older snapshot and would otherwise hide
        // the just-committed records from `lookup_node_by_property`
        // (read-after-write bug). Subsequent snapshots rebuild on their
        // first miss. Mirrors the reset in `flush`/`attach_ssts`.
        self.property_index_cache.reset();

        // Auto-snapshot tick. Best effort: a snapshot is a cache, the
        // WAL is the source of truth. Log the failure and keep going
        // so a temporary object-store hiccup never poisons the commit
        // that just completed.
        self.commits_since_snapshot = self.commits_since_snapshot.saturating_add(1);
        if self.auto_snapshot_every > 0 && self.commits_since_snapshot >= self.auto_snapshot_every {
            match self.write_memtable_snapshot_inner(last_lsn).await {
                Ok(()) => {
                    self.commits_since_snapshot = 0;
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "memtable snapshot write failed; continuing without it"
                    );
                }
            }
        }

        debug!(
            wal_seq = committed_seq,
            last_lsn,
            manifest_version = self.current.manifest.version,
            "commit_batch sealed"
        );

        Ok(CommitOutcome::Committed {
            wal_seq: committed_seq,
            last_lsn,
            records,
            manifest_version: self.current.manifest.version,
        })
    }

    /// Build the next manifest version that records the pending WAL
    /// segment at `seq`. Always derived from `self.current` (unchanged on
    /// a failed commit attempt), so the resulting version is `base + 1`
    /// and carries exactly one fresh `wal_segments` descriptor at `seq`.
    fn build_next(&self, seq: u64, last_lsn: u64) -> Manifest {
        let segment_path = self.wal_store.paths().wal_segment(seq);
        let mut next = self.current.manifest.next_version(self.fence.writer_id);
        // Persist any label names interned since the last commit.
        next.label_dict = self.label_dict.clone();
        next.wal_segments.push(WalSegmentDescriptor {
            seq,
            path: segment_path.as_ref().to_string(),
            last_lsn,
        });
        next
    }

    /// Sequential commit used by the orphan-WAL retry: PUT the manifest
    /// body FIRST, then the WAL segment, then CAS the pointer. Body-first
    /// ordering means the common "manifest body `base+1` already exists"
    /// case fails fast as `ManifestCommitCas` before we ever PUT a WAL
    /// segment, so a doomed retry mints no new orphan WAL at `fresh`.
    /// `self.pending.seq` must already point at the fresh seq.
    async fn commit_body_first(&self, next: Manifest) -> Result<LoadedManifest> {
        let pointer = self
            .manifest_store
            .put_body(&self.fence, &self.current, &next)
            .await?;
        self.wal_store.append_segment(&self.pending).await?;
        self.manifest_store
            .cas_pointer(&self.fence, &self.current, next, pointer)
            .await
    }

    /// Persist the current memtable to `paths.memtable_snapshot()` so
    /// the next cold start can skip the WAL records the snapshot
    /// already covers. Public so callers that drive the cadence
    /// themselves (cloud worker policies, CLI maintenance commands)
    /// can request a snapshot independently of the auto-tick.
    pub async fn write_memtable_snapshot_now(&mut self) -> Result<()> {
        let last_lsn = self.next_lsn.saturating_sub(1);
        self.write_memtable_snapshot_inner(last_lsn).await?;
        self.commits_since_snapshot = 0;
        Ok(())
    }

    async fn write_memtable_snapshot_inner(&self, last_lsn: u64) -> Result<()> {
        let entries = self
            .memtable
            .iter()
            .map(|(key, entry)| (key.clone(), entry.lsn, entry.op.clone()));
        let snapshot = MemtableSnapshotFile::from_iter(last_lsn, entries);
        write_memtable_snapshot(&self.store, self.manifest_store.paths(), &snapshot).await
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

    /// Attach offline-built SSTs into a FRESH namespace as ONE new manifest
    /// version (RFC-023). PUTs every body + bloom + unique-property sidecar
    /// via the size-adaptive uploader, then commits `next_version` with
    /// `schema` REPLACING the base schema and `.ssts` extended with the built
    /// descriptors — exactly the "PUT N immutable SSTs, then one CAS" shape
    /// `flush` already uses, minus the in-process memtable build.
    ///
    /// **Fresh-namespace-only.** The schema is a *replace* and the SST set an
    /// *extend-from-empty*, both safe only on a bootstrap namespace with no
    /// SSTs and no WAL segments; importing into a populated namespace would
    /// clobber other labels' schema and orphan their indexes.
    ///
    /// Error contract mirrors `flush`/`commit`: [`Error::Fenced`] ⇒ abort and
    /// drop the session; a lost manifest CAS ⇒ retryable (already-PUT bodies
    /// are harmless orphans the janitor sweeps).
    pub async fn attach_ssts(
        &mut self,
        built: Vec<crate::flush::builder::BuiltSst>,
        schema: Schema,
    ) -> Result<FlushOutcome> {
        self.fence.assert_alive(self.current.manifest.epoch)?;
        if !self.current.manifest.ssts.is_empty() || !self.current.manifest.wal_segments.is_empty()
        {
            return Err(Error::invariant(
                "attach_ssts requires a fresh namespace (no SSTs, no WAL segments)",
            ));
        }
        if built.is_empty() {
            return Ok(FlushOutcome {
                committed: self.current.clone(),
                ssts_written: 0,
                bloom_sidecars_written: 0,
            });
        }

        // 1. Decompose every BuiltSst into its PUT bodies + manifest
        //    descriptor. `into_parts` is the in-crate accessor that reads
        //    PendingSst's private fields (defined inside flush::builder).
        let store = self.manifest_store.store().clone();
        let mut descriptors = Vec::with_capacity(built.len());
        let mut put_futures: Vec<_> = Vec::with_capacity(built.len() * 2);
        let mut bloom_count = 0usize;
        let mut attached_max_lsn = 0u64;
        for b in built {
            // Install the SST's label names into the namespace dictionary so the
            // on-row LabelIds (baked at build time) resolve. Fresh-namespace
            // attach starts from an empty dict, so a single-label-per-SST batch
            // keeps id 0 == that label, matching the baked `__labels`.
            for name in b.label_names() {
                self.label_dict.intern(&name);
            }
            let (body_path, body, bloom, sidecars, descriptor) = b.into_parts();
            attached_max_lsn = attached_max_lsn.max(descriptor.max_lsn);
            let store_ref = store.clone();
            put_futures.push(Box::pin(async move {
                crate::flush::put_object(store_ref, &body_path, body).await
            })
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<()>> + Send>,
                >);
            if let Some((bloom_path, bloom_body)) = bloom {
                bloom_count += 1;
                let store_ref = store.clone();
                put_futures.push(Box::pin(async move {
                    crate::flush::put_object(store_ref, &bloom_path, bloom_body).await
                }));
            }
            for (sidecar_path, sidecar_body) in sidecars {
                let store_ref = store.clone();
                put_futures.push(Box::pin(async move {
                    crate::flush::put_object(store_ref, &sidecar_path, sidecar_body).await
                }));
            }
            descriptors.push(descriptor);
        }
        let ssts_written = descriptors.len();

        // 2. I/O phase — concurrent PUTs; first error short-circuits.
        futures::future::try_join_all(put_futures).await?;

        // 3. Commit phase — one new manifest version (mirrors flush()).
        let mut next = self.current.manifest.next_version(self.fence.writer_id);
        next.schema = schema; // REPLACE (fresh-namespace-only)
        next.label_dict = self.label_dict.clone();
        next.ssts.extend(descriptors); // extend == set on a fresh base
        next.wal_segments.clear();
        let committed = self
            .manifest_store
            .commit(&self.fence, &self.current, next)
            .await?;

        // 4. Post-commit fixup (mirror flush()) + the PR#40 next_lsn rebase:
        //    seed past the attached SST high-water so a later online write
        //    cannot be silently shadowed by an attached row.
        self.current = committed.clone();
        self.next_lsn = self.next_lsn.max(attached_max_lsn.saturating_add(1)).max(1);
        self.refresh_published();
        self.property_index_cache.reset();

        Ok(FlushOutcome {
            committed,
            ssts_written,
            bloom_sidecars_written: bloom_count,
        })
    }

    /// Register a DiskANN/Vamana vector index — the execution half of
    /// `CREATE VECTOR INDEX` (RFC-030).
    ///
    /// This is a **metadata-only** manifest commit: it appends `desc` to
    /// [`Manifest::vector_indexes`] and commits one new manifest version,
    /// staging **no** memtable rows and writing **no** WAL segment. (It is
    /// DDL, not a row write; routing it through [`commit_batch`](Self::commit_batch)
    /// would stage an empty batch and burn a WAL seq for nothing.) The
    /// compaction build hook materializes the matching `SstKind::VectorGraph`
    /// body lazily on the next sweep — registering the descriptor does not
    /// build the graph.
    ///
    /// Rejects a duplicate index (same `name`, or the same
    /// `(label, property, metric)` target already covered) so two
    /// `CREATE VECTOR INDEX` statements for one property cannot race to build
    /// two graphs over it. With `if_not_exists`, a duplicate is instead a no-op
    /// success returning the current manifest version (`IF NOT EXISTS`); the
    /// int8/cosine misconfiguration check is never suppressed.
    ///
    /// Returns the new manifest version. Error contract mirrors
    /// [`attach_ssts`](Self::attach_ssts): [`Error::Fenced`] ⇒ abort and drop
    /// the session; a lost manifest CAS ⇒ retryable.
    pub async fn register_vector_index(
        &mut self,
        desc: VectorIndexDescriptor,
        if_not_exists: bool,
    ) -> Result<u64> {
        self.fence.assert_alive(self.current.manifest.epoch)?;
        // int8 quantization is cosine-only (the scale-invariant Int8Space).
        // Reject the misconfiguration here — fail-fast — rather than committing a
        // descriptor whose `build_body` would later error and wedge EVERY
        // compaction for the namespace (the descriptor lives in the manifest).
        // `IF NOT EXISTS` suppresses only *existence* conflicts, not this
        // misconfiguration, so the check stays unconditional.
        if desc.quantization == crate::manifest::VectorQuantization::Int8
            && desc.metric != crate::manifest::VectorMetric::Cosine
        {
            return Err(Error::precondition(format!(
                "vector index `{}`: int8 quantization requires metric cosine",
                desc.name
            )));
        }
        for existing in &self.current.manifest.vector_indexes {
            if existing.name == desc.name {
                // A same-name index already exists: idempotent no-op under
                // `IF NOT EXISTS`, otherwise an error (mirrors
                // `create_property_index_named`).
                if if_not_exists {
                    return Ok(self.current.manifest.version);
                }
                return Err(Error::precondition(format!(
                    "a vector index named `{}` already exists",
                    desc.name
                )));
            }
            if existing.matches(&desc.label, &desc.property, desc.metric) {
                if if_not_exists {
                    return Ok(self.current.manifest.version);
                }
                return Err(Error::precondition(format!(
                    "a vector index on ({}:{}) with metric `{}` already exists: `{}`",
                    desc.label,
                    desc.property,
                    desc.metric.builtin_name(),
                    existing.name
                )));
            }
        }

        // Mirror attach_ssts/flush: derive the next version (next_version
        // clones vector_indexes forward), push the descriptor, commit, then
        // refresh the published view so subsequent reads plan against the new
        // catalog.
        let mut next = self.current.manifest.next_version(self.fence.writer_id);
        next.vector_indexes.push(desc);
        let committed = self
            .manifest_store
            .commit(&self.fence, &self.current, next)
            .await?;
        let version = committed.manifest.version;
        self.current = committed;
        self.refresh_published();
        self.property_index_cache.reset();
        Ok(version)
    }

    /// Register a full-text (BM25) index. A **metadata-only** manifest commit
    /// (mirrors [`register_vector_index`](Self::register_vector_index)): appends
    /// `desc` to [`Manifest::text_indexes`], stages no rows and no WAL. The
    /// compaction build hook materializes the `SstKind::TextIndex` body lazily on
    /// the next sweep. Rejects a duplicate by name or by `(label, properties)`
    /// target; with `if_not_exists`, a duplicate is a no-op success returning the
    /// current manifest version. Returns the new manifest version.
    pub async fn register_text_index(
        &mut self,
        desc: TextIndexDescriptor,
        if_not_exists: bool,
    ) -> Result<u64> {
        self.fence.assert_alive(self.current.manifest.epoch)?;
        for existing in &self.current.manifest.text_indexes {
            if existing.name == desc.name {
                // Idempotent no-op under `IF NOT EXISTS`, else an error.
                if if_not_exists {
                    return Ok(self.current.manifest.version);
                }
                return Err(Error::precondition(format!(
                    "a text index named `{}` already exists",
                    desc.name
                )));
            }
            if existing.matches(&desc.label, &desc.properties) {
                if if_not_exists {
                    return Ok(self.current.manifest.version);
                }
                return Err(Error::precondition(format!(
                    "a text index on ({}:{}) already exists: `{}`",
                    desc.label,
                    desc.properties.join(","),
                    existing.name
                )));
            }
        }

        let mut next = self.current.manifest.next_version(self.fence.writer_id);
        next.text_indexes.push(desc);
        let committed = self
            .manifest_store
            .commit(&self.fence, &self.current, next)
            .await?;
        let version = committed.manifest.version;
        self.current = committed;
        self.refresh_published();
        self.property_index_cache.reset();
        Ok(version)
    }

    /// `CREATE CONSTRAINT … IS UNIQUE`: declare `(label, property)` unique so the
    /// write path rejects duplicate values (`CREATE`/`MERGE`/`SET` and the bulk
    /// API all consult `PropertyDef::unique`). Validates the existing data first
    /// — if a duplicate is already present, the constraint is rejected with
    /// [`Error::Precondition`] (mirroring Neo4j) rather than silently leaving a
    /// violated constraint. A metadata-only schema commit; the next flush emits
    /// the unique sidecar.
    pub async fn create_unique_constraint(&mut self, label: &str, property: &str) -> Result<u64> {
        let props = [property.to_string()];
        self.create_unique_constraint_named(None, label, &props, false)
            .await
    }

    /// `CREATE CONSTRAINT [name] [IF NOT EXISTS] FOR (n:Label) REQUIRE (n.p, …)
    /// IS UNIQUE`. Single-property uniqueness sets [`PropertyDef::unique`] (so
    /// the planner point-lookup + equality sidecar keep working) AND records a
    /// named [`Constraint`]; composite uniqueness records only the
    /// [`Constraint`] and is enforced by a tuple scan on write. Validates the
    /// existing data first — a pre-existing duplicate is rejected with
    /// [`Error::Precondition`]. With `if_not_exists`, an already-present
    /// constraint (by name, by the same label+property-set, or a legacy
    /// single-property `unique` flag) is a no-op returning the current version;
    /// without it, that case is an error.
    pub async fn create_unique_constraint_named(
        &mut self,
        name: Option<&str>,
        label: &str,
        properties: &[String],
        if_not_exists: bool,
    ) -> Result<u64> {
        self.fence.assert_alive(self.current.manifest.epoch)?;

        if properties.is_empty() {
            return Err(Error::precondition(
                "a uniqueness constraint requires at least one property",
            ));
        }
        {
            let mut seen = std::collections::HashSet::new();
            for p in properties {
                if !seen.insert(p.as_str()) {
                    return Err(Error::precondition(format!(
                        "property '{p}' is listed twice in the same constraint"
                    )));
                }
            }
        }

        let kind = ConstraintKind::Unique;

        // ── Existence / name-collision checks (borrow the schema, then drop) ──
        let (exists, declared_type) = {
            let schema = &self.current.manifest.schema;
            if let Some(n) = name {
                if let Some(existing) = schema.constraint_named(n) {
                    if !existing.matches(label, properties, kind) {
                        return Err(Error::precondition(format!(
                            "a constraint named '{n}' already exists with a different definition"
                        )));
                    }
                }
            }
            let def_exists = schema
                .constraint_matching(label, properties, kind)
                .is_some();
            // Legacy single-property unique: the flag is set on a manifest that
            // predates named constraints, so there is no list entry yet.
            let legacy_single_exists = properties.len() == 1
                && !def_exists
                && schema
                    .label(label)
                    .and_then(|l| l.properties.iter().find(|p| p.name == properties[0]))
                    .is_some_and(|p| p.unique);
            let declared_type = if properties.len() == 1 {
                schema
                    .label(label)
                    .and_then(|l| l.properties.iter().find(|p| p.name == properties[0]))
                    .map(|p| p.data_type.clone())
            } else {
                None
            };
            (def_exists || legacy_single_exists, declared_type)
        };
        if exists {
            if if_not_exists {
                return Ok(self.current.manifest.version);
            }
            return Err(Error::precondition(format!(
                "a uniqueness constraint on {label}({}) already exists",
                properties.join(", ")
            )));
        }

        // ── Validate existing data; infer the single-property type ───────────
        let single = properties.len() == 1;
        let mut inferred: Option<DataType> = None;
        {
            let snap = self.snapshot();
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            for node in snap.scan_label(label).await? {
                // A row is exempt unless every property is present and non-null.
                let mut key_parts: Vec<String> = Vec::with_capacity(properties.len());
                let mut complete = true;
                for p in properties {
                    match node.properties.get(p) {
                        Some(v) if !matches!(v, Value::Null) => key_parts.push(format!("{v:?}")),
                        _ => {
                            complete = false;
                            break;
                        }
                    }
                }
                if !complete {
                    continue;
                }
                if single && declared_type.is_none() && inferred.is_none() {
                    inferred = value_datatype(node.properties.get(&properties[0]).unwrap());
                }
                // Separate the parts with a control byte so distinct tuples
                // cannot alias (each `{v:?}` part already quotes strings).
                let key = key_parts.join("\u{1}");
                if !seen.insert(key) {
                    let desc = properties
                        .iter()
                        .map(|p| format!("{p}={:?}", node.properties.get(p).unwrap()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(Error::precondition(format!(
                        "cannot create unique constraint on {label}({}): \
                         duplicate ({desc}) already exists",
                        properties.join(", ")
                    )));
                }
            }
        }

        // ── Commit the schema change ─────────────────────────────────────────
        let mut schema = self.current.manifest.schema.clone();
        if single {
            let dtype = declared_type.unwrap_or_else(|| inferred.unwrap_or(DataType::Utf8));
            upsert_property_flags(&mut schema, label, &properties[0], dtype, true, false)?;
        }
        let cname = name
            .map(str::to_string)
            .unwrap_or_else(|| Constraint::default_name(label, properties, kind));
        schema.constraints.push(Constraint {
            name: cname,
            label: label.to_string(),
            properties: properties.to_vec(),
            kind,
        });

        let mut next = self.current.manifest.next_version(self.fence.writer_id);
        next.schema = schema;
        let committed = self
            .manifest_store
            .commit(&self.fence, &self.current, next)
            .await?;
        let version = committed.manifest.version;
        self.current = committed;
        self.refresh_published();
        self.property_index_cache.reset();
        Ok(version)
    }

    /// `CREATE INDEX … ON :Label(prop)`: declare `(label, property)` indexed so
    /// the flush layer emits a secondary equality-index sidecar (faster
    /// `MATCH (n:Label {prop: …})`). Non-unique; no data validation. A
    /// metadata-only schema commit.
    pub async fn create_property_index(&mut self, label: &str, property: &str) -> Result<u64> {
        self.create_property_index_named(None, label, property, false)
            .await
    }

    /// `CREATE INDEX [name] [IF NOT EXISTS] FOR (n:Label) ON (n.prop)`. The
    /// `name` is accepted for Cypher parity but equality-index names are not
    /// persisted (the index is keyed by `(label, property)`); `SHOW INDEXES`
    /// synthesizes a deterministic name. With `if_not_exists`, an already-indexed
    /// property is a no-op; without it, re-declaring one is an error.
    pub async fn create_property_index_named(
        &mut self,
        _name: Option<&str>,
        label: &str,
        property: &str,
        if_not_exists: bool,
    ) -> Result<u64> {
        self.fence.assert_alive(self.current.manifest.epoch)?;
        let already = self
            .current
            .manifest
            .schema
            .label(label)
            .and_then(|l| l.properties.iter().find(|p| p.name == property))
            .is_some_and(|p| p.indexed);
        if already {
            if if_not_exists {
                return Ok(self.current.manifest.version);
            }
            return Err(Error::precondition(format!(
                "an index on {label}({property}) already exists"
            )));
        }
        self.alter_property_for_ddl(label, property, false, true)
            .await
    }

    /// Shared body of the index DDL: scan the label to infer the property type
    /// when it is not already declared, then commit a schema that marks the
    /// property indexed. (Uniqueness DDL has its own path in
    /// [`create_unique_constraint_named`], which also records the named
    /// [`Constraint`].)
    async fn alter_property_for_ddl(
        &mut self,
        label: &str,
        property: &str,
        unique: bool,
        indexed: bool,
    ) -> Result<u64> {
        self.fence.assert_alive(self.current.manifest.epoch)?;

        // The declared type wins; otherwise infer it from the first live value.
        let declared_type = self
            .current
            .manifest
            .schema
            .label(label)
            .and_then(|l| l.properties.iter().find(|p| p.name == property))
            .map(|p| p.data_type.clone());

        let mut inferred: Option<DataType> = None;
        if declared_type.is_none() {
            let snap = self.snapshot();
            for node in snap.scan_label(label).await? {
                let Some(v) = node.properties.get(property) else {
                    continue;
                };
                if matches!(v, Value::Null) {
                    continue;
                }
                inferred = value_datatype(v);
                break;
            }
        }

        let dtype = declared_type.unwrap_or_else(|| inferred.unwrap_or(DataType::Utf8));
        let mut schema = self.current.manifest.schema.clone();
        upsert_property_flags(&mut schema, label, property, dtype, unique, indexed)?;

        let mut next = self.current.manifest.next_version(self.fence.writer_id);
        next.schema = schema;
        let committed = self
            .manifest_store
            .commit(&self.fence, &self.current, next)
            .await?;
        let version = committed.manifest.version;
        self.current = committed;
        self.refresh_published();
        self.property_index_cache.reset();
        Ok(version)
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

/// Best-effort [`DataType`] for a runtime [`Value`], used to type a property
/// that a constraint/index DDL declares but the schema did not. `None` for
/// values that cannot back a typed column (vectors, lists, maps), where the
/// caller falls back to `Utf8` — write-time uniqueness enforcement is
/// type-agnostic regardless.
fn value_datatype(v: &Value) -> Option<DataType> {
    match v {
        Value::Bool(_) => Some(DataType::Bool),
        Value::I64(_) => Some(DataType::Int64),
        Value::F64(_) => Some(DataType::Float64),
        Value::Str(_) => Some(DataType::Utf8),
        Value::Bytes(_) => Some(DataType::Binary),
        Value::Date(_) => Some(DataType::Date32),
        Value::DateTime(_) => Some(DataType::TimestampMicrosUtc),
        Value::Null | Value::Vec(_) | Value::VecI8 { .. } | Value::List(_) | Value::Map(_) => None,
    }
}

/// Mark `(label, property)` unique and/or indexed in `schema`, creating the
/// `LabelDef`/`PropertyDef` when absent (the engine is schemaless, so a
/// constraint may target a property no manifest has declared yet). Existing
/// flags are preserved — a second DDL only ORs its flag in.
fn upsert_property_flags(
    schema: &mut Schema,
    label: &str,
    property: &str,
    dtype: DataType,
    unique: bool,
    indexed: bool,
) -> Result<()> {
    let label_def = schema
        .labels
        .entry(label.to_string())
        .or_insert_with(|| LabelDef {
            name: label.to_string(),
            properties: Vec::new(),
        });
    if let Some(p) = label_def.properties.iter_mut().find(|p| p.name == property) {
        p.unique = p.unique || unique;
        p.indexed = p.indexed || indexed;
    } else {
        let mut p = PropertyDef::new(property, dtype, true).map_err(Error::Core)?;
        p.unique = unique;
        p.indexed = indexed;
        label_def.properties.push(p);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use namidb_core::{
        DataType, EdgeTypeDef, LabelDef, NamespaceId, PropertyDef, SchemaBuilder, Value,
    };
    use object_store::memory::InMemory;
    use object_store::ObjectStoreExt;

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
            ..Default::default()
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
    async fn register_vector_index_commits_descriptor_and_rejects_duplicates() {
        use crate::manifest::{VectorMetric, VectorQuantization};

        let store = make_store();
        let paths = make_paths("ingest-vecidx");
        let mut session = WriterSession::open(store, paths).await.unwrap();
        assert!(
            session
                .snapshot()
                .manifest()
                .manifest
                .vector_indexes
                .is_empty(),
            "fresh namespace has no vector indexes"
        );

        let desc = VectorIndexDescriptor {
            name: "doc_emb".into(),
            label: "Doc".into(),
            property: "emb".into(),
            dim: 16,
            metric: VectorMetric::Cosine,
            r: 32,
            l_build: 64,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        };
        // Metadata-only commit: no rows staged, manifest version bumps to 1.
        assert_eq!(session.pending_len(), 0);
        let v = session
            .register_vector_index(desc.clone(), false)
            .await
            .unwrap();
        assert_eq!(v, 1);
        assert_eq!(session.pending_len(), 0, "DDL stages no memtable rows");
        let snap = session.snapshot();
        let registered = &snap.manifest().manifest.vector_indexes;
        assert_eq!(registered.len(), 1);
        assert_eq!(registered[0].name, "doc_emb");
        assert_eq!(registered[0].metric, VectorMetric::Cosine);
        drop(snap);

        // Same name → rejected, version unchanged.
        let err = session
            .register_vector_index(desc.clone(), false)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "{err:?}");
        assert_eq!(session.manifest_version(), 1);

        // Same name with `IF NOT EXISTS` → idempotent no-op: Ok(current version),
        // no extra index registered.
        let v_ine = session
            .register_vector_index(desc.clone(), true)
            .await
            .unwrap();
        assert_eq!(v_ine, 1, "IF NOT EXISTS returns the current version");
        assert_eq!(session.manifest_version(), 1);
        assert_eq!(
            session.snapshot().manifest().manifest.vector_indexes.len(),
            1
        );

        // Same (label, property, metric) target under a new name → rejected.
        let dup_target = VectorIndexDescriptor {
            name: "doc_emb2".into(),
            ..desc.clone()
        };
        let err = session
            .register_vector_index(dup_target.clone(), false)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "{err:?}");
        // …but `IF NOT EXISTS` over the same target is a no-op success.
        let v_dup = session
            .register_vector_index(dup_target, true)
            .await
            .unwrap();
        assert_eq!(v_dup, 1, "IF NOT EXISTS target collision is a no-op");

        // A different metric over the same property is a distinct index.
        let other = VectorIndexDescriptor {
            name: "doc_dot".into(),
            metric: VectorMetric::Dot,
            ..desc.clone()
        };
        let v2 = session.register_vector_index(other, false).await.unwrap();
        assert_eq!(v2, 2);
        let snap = session.snapshot();
        assert_eq!(snap.manifest().manifest.vector_indexes.len(), 2);

        // int8 quantization requires cosine — a dot/int8 descriptor is rejected at
        // registration (else it would wedge every later compaction).
        let bad = VectorIndexDescriptor {
            name: "doc_i8_dot".into(),
            property: "emb2".into(),
            metric: VectorMetric::Dot,
            quantization: VectorQuantization::Int8,
            ..desc
        };
        let err = session
            .register_vector_index(bad.clone(), false)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "{err:?}");
        // int8/cosine misconfiguration is NOT suppressed by IF NOT EXISTS.
        let err = session.register_vector_index(bad, true).await.unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "{err:?}");
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
    async fn overlay_snapshot_sees_staged_batch_but_plain_snapshot_does_not() {
        // RFC-026 read-your-own-writes at the storage layer: a writer's
        // staged-but-uncommitted batch is visible through `overlay_snapshot`
        // (a staged upsert appears, a staged tombstone hides a committed
        // row) while `snapshot` keeps showing only committed state.
        let store = make_store();
        let paths = make_paths("ingest-overlay");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        // Commit Alice so there is committed state to overlay on top of.
        let alice = sorted_node_id(1);
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        session.commit_batch().await.unwrap();

        // Stage Bob (create) and a tombstone of Alice (delete), no commit.
        let bob = sorted_node_id(2);
        session
            .upsert_node("Person", bob, &node_record("Bob", Some(40)))
            .unwrap();
        session.tombstone_node("Person", alice).unwrap();

        // Plain snapshot: committed only — Alice visible, Bob not.
        let committed = session.snapshot();
        assert!(committed
            .lookup_node("Person", alice)
            .await
            .unwrap()
            .is_some());
        assert!(committed
            .lookup_node("Person", bob)
            .await
            .unwrap()
            .is_none());
        assert_eq!(committed.scan_label("Person").await.unwrap().len(), 1);
        drop(committed);

        // Overlay snapshot: Bob is visible; Alice is hidden by the staged
        // tombstone; the label scan reflects exactly the staged batch.
        let overlay = session.overlay_snapshot();
        assert!(overlay.lookup_node("Person", bob).await.unwrap().is_some());
        assert!(overlay
            .lookup_node("Person", alice)
            .await
            .unwrap()
            .is_none());
        let scanned = overlay.scan_label("Person").await.unwrap();
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].id, bob);
        drop(overlay);

        // Nothing staged: overlay_snapshot collapses to the committed view.
        session.discard_batch();
        let after = session.overlay_snapshot();
        assert!(after.lookup_node("Person", alice).await.unwrap().is_some());
        assert!(after.lookup_node("Person", bob).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn flushed_edge_to_missing_endpoint_is_accepted_and_resolves_lazily() {
        // Load-bearing invariant for multi-window / multi-shard bulk loads
        // (RFC-023): an edge may reference an endpoint node not present in the
        // same batch — or anywhere yet. It must be accepted, survive flush,
        // and resolve lazily (None) at query time, never be rejected. A future
        // referential-integrity feature must not silently break this.
        let store = make_store();
        let paths = make_paths("ingest-dangling-edge");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        let alice = sorted_node_id(1);
        let ghost = sorted_node_id(99); // never upserted as a node
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        // Edge to a non-existent endpoint — must be accepted, not rejected.
        session
            .upsert_edge("KNOWS", alice, ghost, &edge_record())
            .unwrap();
        let _ = session.commit_batch().await.unwrap();
        let outcome = session.flush(schema()).await.unwrap();
        assert!(outcome.committed.manifest.wal_segments.is_empty());

        let snap = session.snapshot();
        // The edge exists and points at the ghost endpoint…
        let out = snap.out_edges("KNOWS", alice).await.unwrap();
        assert_eq!(out.edges.len(), 1);
        assert_eq!(out.edges[0].dst, ghost);
        // …but the endpoint node itself does not resolve.
        assert!(snap.lookup_node("Person", ghost).await.unwrap().is_none());
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

    // Regression: a node committed via `commit_batch` (no flush) must be
    // visible to `lookup_node_by_property` even after the cross-snapshot
    // property index has been warmed for that (label, property) pair.
    // Before the fix `commit_batch` did not reset the cache, so a warmed
    // value→NodeId map (frozen on an older snapshot) hid the just-committed
    // record and the lookup returned `None` (read-after-write bug).
    #[tokio::test]
    async fn commit_batch_resets_property_index_for_read_after_write() {
        let store = make_store();
        let paths = make_paths("ingest-ryow-prop-index");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        // Commit Alice.
        let alice = sorted_node_id(1);
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        let _ = session.commit_batch().await.unwrap();

        // Warm the property-index cache with a miss on a not-yet-existing
        // value, freezing the (Person, name) -> {Alice} map.
        {
            let snap = session.snapshot();
            let miss = snap
                .lookup_node_by_property("Person", "name", "Bob")
                .await
                .unwrap();
            assert!(miss.is_none(), "Bob does not exist yet");
        }

        // Commit Bob via the normal write path (no flush).
        let bob = sorted_node_id(2);
        session
            .upsert_node("Person", bob, &node_record("Bob", Some(40)))
            .unwrap();
        let _ = session.commit_batch().await.unwrap();

        // Bob must now be visible BOTH to scan and to property lookup.
        let snap = session.snapshot();
        let scanned = snap.scan_label("Person").await.unwrap();
        assert!(
            scanned.iter().any(|v| v.id == bob),
            "Bob must be visible to scan_label after commit"
        );
        let found = snap
            .lookup_node_by_property("Person", "name", "Bob")
            .await
            .unwrap();
        assert!(
            found.is_some(),
            "Bob committed via commit_batch must be visible to \
             lookup_node_by_property (property index cache reset on commit)"
        );
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
    async fn discard_batch_drops_uncommitted_writes() {
        let store = make_store();
        let paths = make_paths("ingest-discard");
        let mut session = WriterSession::open(store, paths).await.unwrap();

        session
            .upsert_node("Person", sorted_node_id(1), &node_record("Alice", Some(30)))
            .unwrap();
        session
            .upsert_node("Person", sorted_node_id(2), &node_record("Bob", None))
            .unwrap();
        assert_eq!(session.pending_len(), 2);

        assert_eq!(session.discard_batch(), 2);
        assert_eq!(session.pending_len(), 0);

        // The discarded ops never reached the memtable, so a commit now is a
        // no-op and nothing is visible.
        assert_eq!(session.commit_batch().await.unwrap(), CommitOutcome::Empty);
        let snap = session.snapshot();
        assert!(snap
            .lookup_node("Person", sorted_node_id(1))
            .await
            .unwrap()
            .is_none());

        // The session is still usable: a fresh write commits and persists.
        session
            .upsert_node("Person", sorted_node_id(3), &node_record("Cara", Some(20)))
            .unwrap();
        let out = session.commit_batch().await.unwrap();
        assert!(matches!(out, CommitOutcome::Committed { records: 1, .. }));
        let snap = session.snapshot();
        assert!(snap
            .lookup_node("Person", sorted_node_id(3))
            .await
            .unwrap()
            .is_some());
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
        // The WAL was cleared by the flush, but Alice's row lives in an SST
        // at lsn=1. The counter must rebase to 2 (past the SST high-water),
        // NOT reset to 1 — restarting at 1 would let the next online write
        // collide with / be shadowed by that flushed SST row.
        assert_eq!(
            session2.next_lsn(),
            2,
            "no WAL refs, but next_lsn must rebase past the flushed SST's max_lsn (1)"
        );
        let snap = session2.snapshot();
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
    }

    #[tokio::test]
    async fn reopen_after_flush_rebases_next_lsn_and_online_write_wins() {
        // Regression for the silent-shadow bug: after a namespace flushes
        // all its WAL into SSTs and is cold-reopened, the LSN counter must
        // continue past the SST high-water, and a fresh upsert to an
        // already-flushed node must WIN rather than be shadowed by its
        // older SST row.
        let store = make_store();
        let paths = make_paths("ingest-lsn-rebase");

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        let _ = session.commit_batch().await.unwrap();
        session
            .upsert_node("Person", bob, &node_record("Bob", Some(25)))
            .unwrap();
        let _ = session.commit_batch().await.unwrap();
        let outcome = session.flush(schema()).await.unwrap();
        assert!(outcome.committed.manifest.wal_segments.is_empty());

        // The highest LSN (Bob, lsn=2) now lives only in an SST.
        let max_sst_lsn = outcome
            .committed
            .manifest
            .ssts
            .iter()
            .map(|sst| sst.max_lsn)
            .max()
            .unwrap();
        assert_eq!(max_sst_lsn, 2);

        // Cold-reopen: next_lsn rebases past the SST high-water (not to 1).
        let mut session2 = WriterSession::open(store, paths).await.unwrap();
        assert_eq!(session2.next_lsn(), max_sst_lsn + 1);

        // A fresh upsert to the already-flushed Alice must win.
        session2
            .upsert_node("Person", alice, &node_record("Alice2", Some(31)))
            .unwrap();
        let _ = session2.commit_batch().await.unwrap();
        let snap = session2.snapshot();
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice2".into())),
            "online upsert after cold-reopen must win over the flushed SST row"
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

    /// `ObjectStore` wrapper that fails the next `put_opts` whose key
    /// contains a configured substring, exactly once, with a transient
    /// (non-`AlreadyExists`) error. Used to simulate a crash in the
    /// narrow window between the WAL PUT and the manifest body PUT.
    #[derive(Debug)]
    struct FaultStore {
        inner: Arc<dyn ObjectStore>,
        fail_next_put_on: std::sync::Mutex<Option<String>>,
    }

    impl FaultStore {
        fn new(inner: Arc<dyn ObjectStore>) -> Self {
            Self {
                inner,
                fail_next_put_on: std::sync::Mutex::new(None),
            }
        }
        fn fail_next_put_containing(&self, needle: &str) {
            *self.fail_next_put_on.lock().unwrap() = Some(needle.to_string());
        }
    }

    impl std::fmt::Display for FaultStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FaultStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for FaultStore {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            let hit = {
                let mut guard = self.fail_next_put_on.lock().unwrap();
                match guard.as_deref() {
                    Some(needle) if location.as_ref().contains(needle) => {
                        *guard = None;
                        true
                    }
                    _ => false,
                }
            };
            if hit {
                return Err(object_store::Error::Generic {
                    store: "FaultStore",
                    source: "injected transient put failure".into(),
                });
            }
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &object_store::path::Path,
            to: &object_store::path::Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }
    }

    #[tokio::test]
    async fn commit_batch_reseqs_and_recovers_when_base_plus_one_is_free() {
        // An orphan WAL sits at the seq this session picked, AND the
        // attempt-0 manifest body PUT fails transiently (so version
        // base+1 stays free). Fix D must re-seq past the orphan and
        // complete the commit at the fresh seq, in the same call.
        let fault = Arc::new(FaultStore::new(Arc::new(InMemory::new())));
        let store: Arc<dyn ObjectStore> = fault.clone();
        let paths = make_paths("ingest-orphan-recover");

        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();

        // Orphan at seq=1 (the seq a fresh bootstrap picks).
        let wal_store = WalStore::new(store.clone(), paths.clone());
        let mut orphan = WalSegment::new(1);
        orphan.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: MemKey::Node {
                    id: sorted_node_id(99),
                },
                op: WalOp::Upsert(b"ghost".to_vec()),
                lsn: 1,
            }
            .encode()
            .unwrap(),
        });
        wal_store.append_segment(&orphan).await.unwrap();

        let alice = sorted_node_id(1);
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();

        // Make attempt-0's body PUT (manifest v1) fail once so base+1
        // is free when the retry re-runs put_body.
        let v1 = paths.manifest_version(1);
        fault.fail_next_put_containing(v1.as_ref());

        let out = session.commit_batch().await.unwrap();
        match out {
            CommitOutcome::Committed { wal_seq, .. } => {
                assert_eq!(
                    wal_seq, 2,
                    "Fix D must re-seq past the orphan and commit at 2"
                );
            }
            other => panic!("expected Committed, got {other:?}"),
        }

        // The committed manifest references ONLY the fresh seq, never the
        // orphan seq (rebuilding `next` from scratch, not appending).
        assert_eq!(session.current.manifest.wal_segments.len(), 1);
        assert_eq!(session.current.manifest.wal_segments[0].seq, 2);

        // The real record is durable and visible.
        let snap = session.snapshot();
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(view.lsn, 1);
    }

    #[tokio::test]
    async fn commit_batch_orphan_collision_becomes_clean_terminal_cas_and_poisons() {
        // An orphan WAL sits at the session's seq and attempt-0's body
        // PUT succeeds (occupying base+1). The retry cannot reuse base+1,
        // so it terminates with a clean ManifestCommitCas (NOT a bare
        // Precondition), restores the original seq, and poisons the
        // session so a contract-violating re-entry writes nothing.
        let store = make_store();
        let paths = make_paths("ingest-orphan-terminal");

        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();

        let wal_store = WalStore::new(store.clone(), paths.clone());
        let mut orphan = WalSegment::new(1);
        orphan.push(WalRecord {
            lsn: 1,
            payload: WalEntry {
                key: MemKey::Node {
                    id: sorted_node_id(99),
                },
                op: WalOp::Upsert(b"ghost".to_vec()),
                lsn: 1,
            }
            .encode()
            .unwrap(),
        });
        wal_store.append_segment(&orphan).await.unwrap();

        let alice = sorted_node_id(1);
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();

        let err = session.commit_batch().await.unwrap_err();
        assert!(
            matches!(err, Error::ManifestCommitCas { .. }),
            "orphan collision with an occupied base+1 must surface as a clean terminal CAS, got {err:?}"
        );

        // MUST-FIX 3: the original seq is restored.
        assert_eq!(session.pending.seq, 1);

        // MUST-FIX 4: the session is poisoned; a re-entry short-circuits.
        match session.commit_batch().await.unwrap_err() {
            Error::Precondition(msg) => {
                assert!(
                    msg.contains("poisoned"),
                    "expected poison message, got {msg}"
                )
            }
            other => panic!("expected poisoned Precondition, got {other:?}"),
        }

        // Nothing was applied to the memtable: the batch never ACKed.
        let snap = session.snapshot();
        assert!(snap.lookup_node("Person", alice).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn max_l0_bucket_len_counts_l0_and_drops_after_compaction() {
        // RFC-027 P5 signal: each flush adds one L0 SST to the node bucket,
        // so the worst-bucket L0 count grows with flushes and collapses to
        // zero L0 once compaction folds them into an L1.
        let store = make_store();
        let paths = make_paths("ingest-l0-count");
        let mut session = WriterSession::open(store, paths).await.unwrap();
        assert_eq!(session.max_l0_bucket_len(), 0);

        for i in 0..3u8 {
            session
                .upsert_node("Person", sorted_node_id(i), &node_record("p", None))
                .unwrap();
            session.flush(schema()).await.unwrap();
        }
        assert_eq!(
            session.max_l0_bucket_len(),
            3,
            "three flushes leave three L0 SSTs in the node bucket"
        );

        session.compact_l0(&schema()).await.unwrap();
        assert_eq!(
            session.max_l0_bucket_len(),
            0,
            "compaction folds the L0 SSTs into an L1, clearing the L0 count"
        );
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

    #[tokio::test]
    async fn write_memtable_snapshot_now_persists_and_speeds_cold_start() {
        // Writer 1: upsert + commit + snapshot. Writer 2 (cold start)
        // must see the data without replaying any WAL records — the
        // snapshot covers them.
        let store = make_store();
        let paths = make_paths("ingest-auto-snap");

        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let alice = sorted_node_id(1);
        session
            .upsert_node("Person", alice, &node_record("Alice", Some(30)))
            .unwrap();
        session.commit_batch().await.unwrap();
        session.write_memtable_snapshot_now().await.unwrap();

        // The blob lives at the canonical path. Use the store directly
        // so the test stays storage-agnostic.
        let head = store.head(&paths.memtable_snapshot()).await.unwrap();
        assert!(head.size > 0);

        // Cold start: a new writer recovers from the snapshot rather
        // than the WAL. Data must still be visible.
        drop(session);
        let session2 = WriterSession::open(store, paths).await.unwrap();
        let snap = session2.snapshot();
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
        drop(snap);
        // next_lsn must continue from the snapshot's last_lsn, not
        // restart from 1.
        assert!(session2.next_lsn() > 1);
    }

    // ── Offline SST builder + attach (RFC-023) ──────────────────────────

    #[tokio::test]
    async fn attach_built_node_sst_is_queryable_on_fresh_namespace() {
        use crate::flush::builder::{build_node_sst, NodeInput};
        let store = make_store();
        let paths = make_paths("attach-nodes");

        let id_bytes = {
            let mut b = [0u8; 16];
            b[15] = 1;
            b
        };
        let mut props = BTreeMap::new();
        props.insert("name".to_string(), Value::Str("Alice".into()));
        // Build the SST OUT-OF-BAND — no live writer involved.
        let built = build_node_sst(
            &paths,
            &schema(),
            "Person",
            vec![NodeInput {
                id: id_bytes,
                properties: props,
                tombstone: false,
            }],
        )
        .unwrap()
        .expect("non-empty rows yield a BuiltSst");

        // Attach into a FRESH session on the same paths + store.
        let mut session = WriterSession::open(store, paths).await.unwrap();
        let outcome = session.attach_ssts(vec![built], schema()).await.unwrap();
        assert_eq!(outcome.ssts_written, 1);
        assert_eq!(outcome.committed.manifest.ssts.len(), 1);
        assert!(outcome.committed.manifest.wal_segments.is_empty());

        // The attached node is queryable.
        let alice = NodeId::from_uuid(Uuid::from_bytes(id_bytes));
        let snap = session.snapshot();
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
    }

    #[tokio::test]
    async fn attach_into_nonfresh_namespace_is_rejected() {
        use crate::flush::builder::{build_node_sst, NodeInput};
        let store = make_store();
        let paths = make_paths("attach-nonfresh");
        let mut session = WriterSession::open(store, paths.clone()).await.unwrap();
        // Make the namespace non-fresh (a committed WAL segment).
        session
            .upsert_node("Person", sorted_node_id(1), &node_record("Alice", Some(30)))
            .unwrap();
        let _ = session.commit_batch().await.unwrap();

        let built = build_node_sst(
            &paths,
            &schema(),
            "Person",
            vec![NodeInput {
                id: [0u8; 16],
                properties: BTreeMap::new(),
                tombstone: false,
            }],
        )
        .unwrap()
        .unwrap();
        let err = session
            .attach_ssts(vec![built], schema())
            .await
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("fresh"),
            "expected fresh-namespace guard, got {err:?}"
        );
    }

    #[tokio::test]
    async fn builder_dedups_duplicate_node_ids_keeping_last() {
        use crate::flush::builder::{build_node_sst, NodeInput};
        let store = make_store();
        let paths = make_paths("attach-dedup");
        let id = {
            let mut b = [0u8; 16];
            b[15] = 7;
            b
        };
        let mk = |name: &str| {
            let mut p = BTreeMap::new();
            p.insert("name".to_string(), Value::Str(name.into()));
            NodeInput {
                id,
                properties: p,
                tombstone: false,
            }
        };
        // Same id twice — "Old" then "New". Keep-LAST ⇒ "New".
        let built = build_node_sst(&paths, &schema(), "Person", vec![mk("Old"), mk("New")])
            .unwrap()
            .unwrap();
        assert_eq!(built.row_count(), 1, "duplicate ids collapse to one row");

        let mut session = WriterSession::open(store, paths).await.unwrap();
        session.attach_ssts(vec![built], schema()).await.unwrap();
        let node = NodeId::from_uuid(Uuid::from_bytes(id));
        let snap = session.snapshot();
        let view = snap.lookup_node("Person", node).await.unwrap().unwrap();
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("New".into())),
            "keep-last upsert semantics"
        );
    }

    #[tokio::test]
    async fn attach_built_nodes_and_edges_is_traversable() {
        use crate::flush::builder::{build_edge_ssts, build_node_sst, EdgeInput, NodeInput};
        let store = make_store();
        let paths = make_paths("attach-graph");
        let a = {
            let mut b = [0u8; 16];
            b[15] = 1;
            b
        };
        let z = {
            let mut b = [0u8; 16];
            b[15] = 2;
            b
        };
        let node = |id: [u8; 16], name: &str| {
            let mut p = BTreeMap::new();
            p.insert("name".to_string(), Value::Str(name.into()));
            NodeInput {
                id,
                properties: p,
                tombstone: false,
            }
        };
        let nodes = build_node_sst(
            &paths,
            &schema(),
            "Person",
            vec![node(a, "A"), node(z, "Z")],
        )
        .unwrap()
        .unwrap();
        let edges = build_edge_ssts(
            &paths,
            &schema(),
            "KNOWS",
            vec![EdgeInput {
                src: a,
                dst: z,
                properties: BTreeMap::new(),
                tombstone: false,
            }],
        )
        .unwrap();
        assert_eq!(edges.len(), 2, "forward + inverse CSR");

        let mut all = vec![nodes];
        all.extend(edges);
        let mut session = WriterSession::open(store, paths).await.unwrap();
        let outcome = session.attach_ssts(all, schema()).await.unwrap();
        assert_eq!(outcome.ssts_written, 3, "1 node + 2 edge SSTs");

        // The graph is traversable: A -KNOWS-> Z.
        let aid = NodeId::from_uuid(Uuid::from_bytes(a));
        let zid = NodeId::from_uuid(Uuid::from_bytes(z));
        let snap = session.snapshot();
        let out = snap.out_edges("KNOWS", aid).await.unwrap();
        assert_eq!(out.edges.len(), 1);
        assert_eq!(out.edges[0].dst, zid);
    }

    #[tokio::test]
    async fn create_unique_constraint_sets_flag_and_create_index_marks_indexed() {
        let store = make_store();
        let paths = make_paths("ingest-constraint");
        let mut s = WriterSession::open(store, paths).await.unwrap();
        s.upsert_node("Person", NodeId::new(), &node_record("Alice", Some(30)))
            .unwrap();
        s.upsert_node("Person", NodeId::new(), &node_record("Bob", Some(40)))
            .unwrap();
        s.commit_batch().await.unwrap();

        // No duplicates → the unique constraint commits and flips the schema flag.
        let v = s.create_unique_constraint("Person", "name").await.unwrap();
        assert!(v >= 1);
        assert_eq!(s.pending_len(), 0, "DDL stages no memtable rows");
        let snap = s.snapshot();
        let name = snap
            .manifest()
            .manifest
            .schema
            .label("Person")
            .unwrap()
            .properties
            .iter()
            .find(|p| p.name == "name")
            .unwrap();
        assert!(name.unique);
        drop(snap);

        // CREATE INDEX marks the property indexed (non-unique).
        s.create_property_index("Person", "age").await.unwrap();
        let snap = s.snapshot();
        let age = snap
            .manifest()
            .manifest
            .schema
            .label("Person")
            .unwrap()
            .properties
            .iter()
            .find(|p| p.name == "age")
            .unwrap();
        assert!(age.indexed);
    }

    #[tokio::test]
    async fn create_unique_constraint_rejects_when_data_already_violates() {
        let store = make_store();
        let paths = make_paths("ingest-constraint-dup");
        let mut s = WriterSession::open(store, paths).await.unwrap();
        s.upsert_node("Person", NodeId::new(), &node_record("Dup", Some(1)))
            .unwrap();
        s.upsert_node("Person", NodeId::new(), &node_record("Dup", Some(2)))
            .unwrap();
        s.commit_batch().await.unwrap();

        let err = s
            .create_unique_constraint("Person", "name")
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "got {err:?}");
        // Rejected before any commit: schema unchanged.
        assert!(s
            .snapshot()
            .manifest()
            .manifest
            .schema
            .label("Person")
            .and_then(|l| l.properties.iter().find(|p| p.name == "name"))
            .map(|p| !p.unique)
            .unwrap_or(true));
    }

    #[tokio::test]
    async fn composite_unique_constraint_rejects_existing_duplicate_tuple() {
        let store = make_store();
        let paths = make_paths("ingest-composite-dup");
        let mut s = WriterSession::open(store, paths).await.unwrap();
        // Two nodes share the same (name, age) tuple.
        s.upsert_node("Person", NodeId::new(), &node_record("Dup", Some(7)))
            .unwrap();
        s.upsert_node("Person", NodeId::new(), &node_record("Dup", Some(7)))
            .unwrap();
        s.commit_batch().await.unwrap();

        let props = vec!["name".to_string(), "age".to_string()];
        let err = s
            .create_unique_constraint_named(None, "Person", &props, false)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "got {err:?}");
        // Nothing recorded: rejected before any commit.
        assert!(s
            .snapshot()
            .manifest()
            .manifest
            .schema
            .constraints
            .is_empty());
    }

    #[tokio::test]
    async fn composite_unique_constraint_allows_distinct_tuples_and_records_it() {
        let store = make_store();
        let paths = make_paths("ingest-composite-ok");
        let mut s = WriterSession::open(store, paths).await.unwrap();
        s.upsert_node("Person", NodeId::new(), &node_record("Ann", Some(7)))
            .unwrap();
        // Same name, different age → the tuple is distinct, so it is allowed.
        s.upsert_node("Person", NodeId::new(), &node_record("Ann", Some(8)))
            .unwrap();
        s.commit_batch().await.unwrap();

        let props = vec!["name".to_string(), "age".to_string()];
        s.create_unique_constraint_named(Some("uq_pa"), "Person", &props, false)
            .await
            .unwrap();
        let snap = s.snapshot();
        let schema = &snap.manifest().manifest.schema;
        assert_eq!(schema.constraints.len(), 1);
        assert_eq!(schema.constraints[0].name, "uq_pa");
        assert_eq!(schema.constraints[0].properties, props);
        // A composite constraint must NOT flip the single-property unique flag.
        let name_unique = schema
            .label("Person")
            .and_then(|l| l.properties.iter().find(|p| p.name == "name"))
            .map(|p| p.unique)
            .unwrap_or(false);
        assert!(!name_unique, "composite must not mark `name` itself unique");
    }

    #[tokio::test]
    async fn unique_constraint_if_not_exists_is_idempotent_else_errors() {
        let store = make_store();
        let paths = make_paths("ingest-ine");
        let mut s = WriterSession::open(store, paths).await.unwrap();
        s.upsert_node("Person", NodeId::new(), &node_record("Ann", Some(1)))
            .unwrap();
        s.commit_batch().await.unwrap();

        let props = vec!["name".to_string()];
        let v1 = s
            .create_unique_constraint_named(None, "Person", &props, false)
            .await
            .unwrap();
        // Re-declaring WITHOUT `IF NOT EXISTS` is an error.
        let err = s
            .create_unique_constraint_named(None, "Person", &props, false)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "got {err:?}");
        // Re-declaring WITH `IF NOT EXISTS` is a no-op (no new manifest version).
        let v2 = s
            .create_unique_constraint_named(None, "Person", &props, true)
            .await
            .unwrap();
        assert_eq!(v2, v1, "IF NOT EXISTS must not commit a new version");
    }
}
