//! Limited node scan over overlapping node SSTs.
//!
//! With node rows in the memtable, or two node SSTs whose id ranges overlap
//! (random ids make that the norm once there is more than one SST), no single
//! SST holds a prefix of the scan. The limited scan used to reconcile every
//! node and truncate, which on a large store decodes every row and its
//! overflow JSON to return ten. This route merges the SSTs' id-ordered rows
//! instead, reading only the structural columns, settles last-writer-wins
//! per id as it goes, and point-reads the winners in batches until it has
//! `limit` rows. The rows and their order are the ones the full scan
//! returns, since that scan also emits winners in id order.
//!
//! [`NodeScanCursor`] keeps the same merge open between calls, so a caller
//! can pull a label one batch at a time.

use std::collections::{BTreeMap, VecDeque};

use arrow_array::{BooleanArray, FixedSizeBinaryArray, UInt64Array};
use object_store::path::Path;
use uuid::Uuid;

use namidb_core::{LabelId, NodeId};

use super::{
    node_scan_limited_async, node_sst_can_contain_label, node_view_from_payload, row_carries_label,
    NodeView, RangedMode, Snapshot,
};
use crate::error::{Error, Result};
use crate::memtable::{MemKey, MemOp};
use crate::sst::nodes::{RangedNodeBatchStream, COL_LSN, COL_NODE_ID, COL_TOMBSTONE};
use crate::sst::predicates::ScanPredicate;

/// One SST's structural rows, pulled a row group at a time.
struct SstCursor {
    stream: RangedNodeBatchStream,
    /// `(id, lsn, live and carries the label)`, in id order.
    rows: VecDeque<(NodeId, u64, bool)>,
    last: Option<NodeId>,
    done: bool,
}

impl Snapshot<'_> {
    /// Whether a `_limited` node scan with `predicates` can stop before
    /// reading every node: always on disjoint node SSTs, and through the
    /// merge below otherwise. The merge only takes unfiltered scans and
    /// needs ranged reads, so a filtered scan or the full-body diagnostic
    /// mode keeps the reconcile-and-truncate fallback.
    pub fn node_scan_stops_early(&self, predicates: &[ScanPredicate]) -> bool {
        self.node_scan_is_disjoint()
            || (predicates.is_empty() && !matches!(self.ranged_mode, RangedMode::Force(false)))
    }

    /// The first `limit` live nodes (optionally in `label`, matching every
    /// predicate) in id order, without reading the rest of the store.
    ///
    /// `None` leaves the caller on the full reconciliation: a filtered scan,
    /// the full-body diagnostic mode, or an SST whose rows are not in id
    /// order. A filter could only be checked after the point read, so a
    /// selective one would cost a point read per node in every SST holding
    /// a version of it, where the reconcile reads the projected columns
    /// once.
    pub(super) async fn try_scan_overlapping_nodes_limited(
        &self,
        label: Option<&str>,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
        limit: usize,
    ) -> Result<Option<Vec<NodeView>>> {
        if !predicates.is_empty() {
            return Ok(None);
        }
        let Some(mut merge) = self.open_node_merge(label).await? else {
            return Ok(None);
        };
        let mut out: Vec<NodeView> = Vec::with_capacity(limit.min(1024));
        let mut pending: Vec<NodeId> = Vec::new();
        loop {
            let exhausted = match merge.next_live(self).await? {
                Merged::Live(id) => {
                    pending.push(id);
                    false
                }
                Merged::End => true,
                Merged::Unordered => return Ok(None),
            };
            if !pending.is_empty() && (exhausted || pending.len() >= limit - out.len()) {
                out.extend(self.lookup_merged(label, projection, &pending).await?);
                out.truncate(limit);
                if out.len() == limit {
                    return Ok(Some(out));
                }
                pending.clear();
            }
            if exhausted {
                return Ok(Some(out));
            }
        }
    }

    /// Open a cursor over every live node of `label` at this snapshot, in id
    /// order, that reads the store one batch at a time. Every call to
    /// [`NodeScanCursor::next_batch`] must pass this same snapshot.
    pub async fn label_scan_cursor(&self, label: &str) -> Result<NodeScanCursor> {
        Ok(NodeScanCursor {
            label: label.to_string(),
            merge: self.open_node_merge(Some(label)).await?,
            fallback: None,
            last: None,
            failed: false,
        })
    }

    /// Start the id-ordered merge of the memtable and every node SST, or
    /// `None` in the full-body diagnostic mode, which has no ranged reads.
    async fn open_node_merge(&self, label: Option<&str>) -> Result<Option<NodeMerge>> {
        if matches!(self.ranged_mode, RangedMode::Force(false)) {
            return Ok(None);
        }
        let dict = &self.manifest.manifest.label_dict;

        // The memtable side, one winner per id. Equal LSNs keep the first
        // version seen, the same tie rule as `update_node_winner`.
        let mut memtable: BTreeMap<NodeId, (u64, bool)> = BTreeMap::new();
        for (key, entry) in self.node_entries() {
            let MemKey::Node { id } = key else {
                continue;
            };
            if memtable.get(id).is_some_and(|(lsn, _)| *lsn >= entry.lsn) {
                continue;
            }
            let keep = match &entry.op {
                MemOp::Tombstone => false,
                MemOp::Upsert(payload) => match label {
                    None => true,
                    Some(label) => node_view_from_payload(*id, entry.lsn, payload, dict, "")?
                        .labels
                        .iter()
                        .any(|l| l == label),
                },
            };
            memtable.insert(*id, (entry.lsn, keep));
        }

        let mut cursors = Vec::new();
        for idx in self.manifest.index.node_descriptors() {
            // An SST that cannot hold the label has no candidates. A newer
            // version there that dropped the label is caught by the point
            // read, which resolves every candidate over all SSTs.
            if label.is_some_and(|label| {
                !node_sst_can_contain_label(&self.manifest.manifest, idx, label)
            }) {
                continue;
            }
            let desc = &self.manifest.manifest.ssts[idx];
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            let metadata = self.node_sst_metadata(desc, &absolute).await?;
            let (stream, _) = node_scan_limited_async(
                self.store.clone(),
                Path::from(absolute),
                desc.size_bytes,
                &self.label_def_for_node_sst(desc),
                &[],
                Some(&[]),
                Some(metadata),
            )
            .await?;
            cursors.push((
                idx,
                SstCursor {
                    stream,
                    rows: VecDeque::new(),
                    last: None,
                    done: false,
                },
            ));
        }
        Ok(Some(NodeMerge {
            label: label.map(str::to_string),
            label_id: label.and_then(|label| dict.id(label)),
            memtable: memtable.into_iter().peekable(),
            cursors,
        }))
    }

    /// Point-read merged winners, dropping any whose stored version lost
    /// `label`, and trim them to `projection`.
    async fn lookup_merged(
        &self,
        label: Option<&str>,
        projection: Option<&[String]>,
        ids: &[NodeId],
    ) -> Result<Vec<NodeView>> {
        let mut out = Vec::with_capacity(ids.len());
        for mut view in self
            .batch_lookup_nodes("", ids)
            .await?
            .into_iter()
            .flatten()
        {
            if label.is_some_and(|label| !view.labels.iter().any(|l| l == label)) {
                continue;
            }
            if let Some(requested) = projection {
                view.properties
                    .retain(|property, _| requested.iter().any(|r| r == property));
            }
            out.push(view);
        }
        Ok(out)
    }
}

/// What [`NodeMerge::next_live`] found.
enum Merged {
    Live(NodeId),
    End,
    /// An SST's rows are not in id order, so the merge cannot go on.
    Unordered,
}

/// The id-ordered merge of the memtable and every node SST, reading only
/// the structural columns. It owns its streams and borrows nothing from the
/// snapshot that opened it, so it can be kept between reads as long as the
/// same snapshot is passed back.
struct NodeMerge {
    label: Option<String>,
    label_id: Option<LabelId>,
    memtable: std::iter::Peekable<std::collections::btree_map::IntoIter<NodeId, (u64, bool)>>,
    cursors: Vec<(usize, SstCursor)>,
}

impl NodeMerge {
    /// The next id whose newest version is live (and carries the label).
    async fn next_live(&mut self, snap: &Snapshot<'_>) -> Result<Merged> {
        let dict = &snap.manifest.manifest.label_dict;
        let label = self.label.as_deref();
        loop {
            for (idx, cursor) in &mut self.cursors {
                while cursor.rows.is_empty() && !cursor.done {
                    crate::cancel::check()?;
                    let desc = &snap.manifest.manifest.ssts[*idx];
                    let Some(reader) = cursor.stream.next_row_group().await? else {
                        cursor.done = true;
                        break;
                    };
                    let mut decoded = 0;
                    for batch in reader {
                        let batch = batch.map_err(|error| {
                            Error::invariant(format!("structural node scan read: {error}"))
                        })?;
                        crate::cancel::check()?;
                        let id_col = batch
                            .column_by_name(COL_NODE_ID)
                            .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
                            .ok_or_else(|| Error::invariant("node_id column missing"))?;
                        let tomb_col = batch
                            .column_by_name(COL_TOMBSTONE)
                            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
                            .ok_or_else(|| Error::invariant("tombstone column missing"))?;
                        let lsn_col = batch
                            .column_by_name(COL_LSN)
                            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                            .ok_or_else(|| Error::invariant("lsn column missing"))?;
                        decoded += batch.num_rows();
                        for row in 0..batch.num_rows() {
                            let id_bytes: [u8; 16] = id_col
                                .value(row)
                                .try_into()
                                .map_err(|_| Error::invariant("node_id row length != 16"))?;
                            let id = NodeId::from_uuid(Uuid::from_bytes(id_bytes));
                            // The merge needs each SST strictly in id order.
                            if cursor.last.is_some_and(|last| last >= id) {
                                return Ok(Merged::Unordered);
                            }
                            cursor.last = Some(id);
                            let keep = !tomb_col.value(row)
                                && label.is_none_or(|label| {
                                    row_carries_label(
                                        &batch,
                                        row,
                                        label,
                                        self.label_id,
                                        dict,
                                        &desc.scope,
                                    )
                                });
                            cursor.rows.push_back((id, lsn_col.value(row), keep));
                        }
                    }
                    if let Some(cache) = &snap.cache {
                        cache.record_limited_node_scan_work(decoded, 0, 0, 1, 0);
                    }
                }
            }

            let next = self
                .cursors
                .iter()
                .filter_map(|(_, cursor)| cursor.rows.front().map(|row| row.0))
                .chain(self.memtable.peek().map(|(id, _)| *id))
                .min();
            let Some(id) = next else {
                return Ok(Merged::End);
            };
            let mut winner: Option<(u64, bool)> = None;
            if self
                .memtable
                .peek()
                .is_some_and(|(mem_id, _)| *mem_id == id)
            {
                winner = self.memtable.next().map(|(_, version)| version);
            }
            for (_, cursor) in &mut self.cursors {
                if cursor.rows.front().is_some_and(|row| row.0 == id) {
                    let (_, lsn, keep) = cursor.rows.pop_front().expect("front checked");
                    if winner.is_none_or(|(best, _)| lsn > best) {
                        winner = Some((lsn, keep));
                    }
                }
            }
            if winner.is_some_and(|(_, keep)| keep) {
                return Ok(Merged::Live(id));
            }
        }
    }
}

/// A label scan read one batch at a time, from
/// [`Snapshot::label_scan_cursor`]. Each batch costs about its own rows:
/// the structural columns of the row groups it reaches, and one batched
/// point read for the rows it returns. Rows come in id order, the order
/// [`Snapshot::scan_label`] returns them in.
pub struct NodeScanCursor {
    label: String,
    /// `None` when the merge cannot run (full-body diagnostic mode).
    merge: Option<NodeMerge>,
    /// Rows left from a full reconciliation, once the merge gave up.
    fallback: Option<std::vec::IntoIter<NodeView>>,
    /// The last id returned.
    last: Option<NodeId>,
    /// An earlier batch failed.
    failed: bool,
}

impl std::fmt::Debug for NodeScanCursor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NodeScanCursor")
            .field("label", &self.label)
            .field("last", &self.last)
            .finish_non_exhaustive()
    }
}

impl NodeScanCursor {
    /// Up to `max` more nodes; empty once the label is exhausted. `snap`
    /// must be the snapshot that opened the cursor. After an error every
    /// later call fails too, since the failed batch's rows are lost and
    /// going on would skip them.
    pub async fn next_batch(&mut self, snap: &Snapshot<'_>, max: usize) -> Result<Vec<NodeView>> {
        if self.failed {
            return Err(Error::precondition(
                "an earlier batch of this label scan failed; open a new cursor",
            ));
        }
        let batch = self.read_batch(snap, max).await;
        self.failed = batch.is_err();
        batch
    }

    async fn read_batch(&mut self, snap: &Snapshot<'_>, max: usize) -> Result<Vec<NodeView>> {
        loop {
            if let Some(rest) = &mut self.fallback {
                return Ok(rest.by_ref().take(max).collect());
            }
            if let Some(merge) = &mut self.merge {
                if let Some(rows) = Self::merged_batch(merge, snap, &self.label, max).await? {
                    if let Some(row) = rows.last() {
                        self.last = Some(row.id);
                    }
                    return Ok(rows);
                }
                // The merge may have settled an id below `last` from a stale
                // version before it saw the SST out of order, and rows it
                // already returned cannot be taken back.
                if self.last.is_some() {
                    return Err(Error::invariant(format!(
                        "a node SST is not in id order, so the scan of `{}` cannot go on; \
                         use scan_label",
                        self.label
                    )));
                }
            }
            // ponytail: an SST out of id order (or no ranged reads) costs one
            // full reconcile here, like the limited scan's fallback.
            self.fallback = Some(snap.scan_label(&self.label).await?.into_iter());
        }
    }

    /// `None` when the merge hit an SST out of id order.
    async fn merged_batch(
        merge: &mut NodeMerge,
        snap: &Snapshot<'_>,
        label: &str,
        max: usize,
    ) -> Result<Option<Vec<NodeView>>> {
        let mut out = Vec::new();
        while out.len() < max {
            let mut pending = Vec::new();
            let mut end = false;
            while pending.len() < max - out.len() {
                match merge.next_live(snap).await? {
                    Merged::Live(id) => pending.push(id),
                    Merged::End => {
                        end = true;
                        break;
                    }
                    Merged::Unordered => return Ok(None),
                }
            }
            out.extend(snap.lookup_merged(Some(label), None, &pending).await?);
            if end {
                break;
            }
        }
        Ok(Some(out))
    }
}
