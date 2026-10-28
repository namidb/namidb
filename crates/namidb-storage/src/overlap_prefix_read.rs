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

use std::collections::{BTreeMap, VecDeque};

use arrow_array::{BooleanArray, FixedSizeBinaryArray, UInt64Array};
use object_store::path::Path;
use uuid::Uuid;

use namidb_core::NodeId;

use super::{
    node_scan_limited_async, node_view_from_payload, node_view_matches_predicates,
    row_carries_label, NodeView, RangedMode, Snapshot,
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
    /// Whether a `_limited` node scan can stop before reading every node:
    /// always on disjoint node SSTs, and through the merge below otherwise.
    /// The merge needs ranged reads, so the full-body diagnostic mode keeps
    /// the old reconcile-and-truncate fallback.
    pub fn node_scan_stops_early(&self) -> bool {
        self.node_scan_is_disjoint() || !matches!(self.ranged_mode, RangedMode::Force(false))
    }

    /// The first `limit` live nodes (optionally in `label`, matching every
    /// predicate) in id order, without reading the rest of the store.
    ///
    /// `None` leaves the caller on the full reconciliation: the full-body
    /// diagnostic mode, or an SST whose rows are not in id order.
    pub(super) async fn try_scan_overlapping_nodes_limited(
        &self,
        label: Option<&str>,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
        limit: usize,
    ) -> Result<Option<Vec<NodeView>>> {
        if matches!(self.ranged_mode, RangedMode::Force(false)) {
            return Ok(None);
        }
        let dict = &self.manifest.manifest.label_dict;
        let label_id = label.and_then(|label| dict.id(label));

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
        let mut memtable = memtable.into_iter().peekable();

        let mut cursors = Vec::new();
        for idx in self.manifest.index.node_descriptors() {
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

        let remaining = |out: &Vec<NodeView>| {
            let left = limit - out.len();
            // A predicate can reject what the label already admitted, so
            // fetch a few more per round rather than one batch per survivor.
            if predicates.is_empty() {
                left
            } else {
                left.max(256)
            }
        };
        let mut out: Vec<NodeView> = Vec::with_capacity(limit.min(1024));
        let mut pending: Vec<NodeId> = Vec::new();
        loop {
            for (idx, cursor) in &mut cursors {
                while cursor.rows.is_empty() && !cursor.done {
                    let desc = &self.manifest.manifest.ssts[*idx];
                    let Some(reader) = cursor.stream.next_row_group().await? else {
                        cursor.done = true;
                        break;
                    };
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
                        for row in 0..batch.num_rows() {
                            let id_bytes: [u8; 16] = id_col
                                .value(row)
                                .try_into()
                                .map_err(|_| Error::invariant("node_id row length != 16"))?;
                            let id = NodeId::from_uuid(Uuid::from_bytes(id_bytes));
                            // The merge needs each SST strictly in id order.
                            if cursor.last.is_some_and(|last| last >= id) {
                                return Ok(None);
                            }
                            cursor.last = Some(id);
                            let keep = !tomb_col.value(row)
                                && label.is_none_or(|label| {
                                    row_carries_label(
                                        &batch,
                                        row,
                                        label,
                                        label_id,
                                        dict,
                                        &desc.scope,
                                    )
                                });
                            cursor.rows.push_back((id, lsn_col.value(row), keep));
                        }
                    }
                }
            }

            let next = cursors
                .iter()
                .filter_map(|(_, cursor)| cursor.rows.front().map(|row| row.0))
                .chain(memtable.peek().map(|(id, _)| *id))
                .min();
            let exhausted = next.is_none();
            if let Some(id) = next {
                let mut winner: Option<(u64, bool)> = None;
                if memtable.peek().is_some_and(|(mem_id, _)| *mem_id == id) {
                    winner = memtable.next().map(|(_, version)| version);
                }
                for (_, cursor) in &mut cursors {
                    if cursor.rows.front().is_some_and(|row| row.0 == id) {
                        let (_, lsn, keep) = cursor.rows.pop_front().expect("front checked");
                        if winner.is_none_or(|(best, _)| lsn > best) {
                            winner = Some((lsn, keep));
                        }
                    }
                }
                if winner.is_some_and(|(_, keep)| keep) {
                    pending.push(id);
                }
            }

            if !pending.is_empty() && (exhausted || pending.len() >= remaining(&out)) {
                for view in self.batch_lookup_nodes("", &pending).await?.into_iter().flatten() {
                    if label.is_some_and(|label| !view.labels.iter().any(|l| l == label))
                        || !node_view_matches_predicates(&view, predicates)
                    {
                        continue;
                    }
                    let mut view = view;
                    if let Some(requested) = projection {
                        view.properties
                            .retain(|property, _| requested.iter().any(|r| r == property));
                    }
                    out.push(view);
                    if out.len() == limit {
                        return Ok(Some(out));
                    }
                }
                pending.clear();
            }
            if exhausted {
                return Ok(Some(out));
            }
        }
    }
}
