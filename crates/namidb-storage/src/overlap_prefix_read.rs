//! Limited node scan over overlapping node SSTs.
//!
//! With node rows in the memtable, or two node SSTs whose id ranges overlap
//! (random ids make that the norm once there is more than one SST), no single
//! SST holds a prefix of the scan. The limited scan used to reconcile every
//! node and truncate, which on a large store decodes every row and its
//! overflow JSON to return ten. This route merges the SSTs' id-ordered rows
//! instead, reading only the structural columns plus whatever the predicates
//! need, settles last-writer-wins per id as it goes, tests the predicates on
//! the winning version, and stops at `limit` winners. Their properties are
//! then read from the SST row that won, through the same property pages or
//! Parquet row selection as the two-phase label scan. The rows and their
//! order are the ones the full scan returns, since that scan also emits
//! winners in id order and tests predicates after last-writer-wins.
//!
//! [`NodeScanCursor`] runs a structural-only merge that stays open between
//! calls, so a caller can pull a label one batch at a time.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use arrow_array::{BooleanArray, FixedSizeBinaryArray, RecordBatch, UInt64Array};
use bytes::Bytes;
use object_store::path::Path;
use uuid::Uuid;

use namidb_core::{LabelDef, LabelId, NodeId};

use super::{
    decode_node_labels, decode_node_row_properties, node_scan_limited_async,
    node_sst_can_contain_label, node_view_from_payload, node_view_matches_predicates,
    row_carries_label, LabelScanRow, NodePropertyPageReader, NodeView, RangedMode, Snapshot,
};
use crate::error::{Error, Result};
use crate::memtable::{MemKey, MemOp};
use crate::sst::nodes::{
    RangedNodeBatchStream, COL_LSN, COL_NODE_ID, COL_TOMBSTONE, SCHEMA_VERSION,
};
use crate::sst::predicates::ScanPredicate;

/// One SST's rows, pulled a row group at a time.
struct SstCursor {
    stream: RangedNodeBatchStream,
    label_def: LabelDef,
    /// Property pages for the predicate columns, when the SST has them.
    pages: Option<Arc<NodePropertyPageReader>>,
    /// The current row group's batches, which `rows` point into.
    batches: Vec<RecordBatch>,
    /// In id order.
    rows: VecDeque<SstRow>,
    /// Physical ordinal of the next row read.
    ordinal: u64,
    last: Option<NodeId>,
    done: bool,
}

struct SstRow {
    id: NodeId,
    lsn: u64,
    /// Live, carries the label and passes every predicate.
    keep: bool,
    batch: usize,
    row: usize,
    ordinal: u64,
}

impl Snapshot<'_> {
    /// Whether a `_limited` node scan can stop before reading every node:
    /// always on disjoint node SSTs, and through the merge below otherwise.
    /// The merge needs ranged reads, so the full-body diagnostic mode keeps
    /// the reconcile-and-truncate fallback.
    pub fn node_scan_stops_early(&self) -> bool {
        self.node_scan_is_disjoint() || !matches!(self.ranged_mode, RangedMode::Force(false))
    }

    /// The first `limit` live nodes (optionally in `label`, matching every
    /// predicate) in id order, without reading the rest of the store.
    ///
    /// `None` leaves the caller on the full reconciliation: the full-body
    /// diagnostic mode, a failed property page probe, or an SST whose rows
    /// are not in id order or disagree with the manifest.
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
        let predicate_columns: Vec<String> = predicates
            .iter()
            .map(|predicate| predicate.column().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let predicate_set: BTreeSet<&str> = predicate_columns.iter().map(String::as_str).collect();

        // The memtable side, one version per id, decoded only when the merge
        // reaches it. Equal LSNs keep the first version seen, the same tie
        // rule as `update_node_winner`.
        let mut memtable: BTreeMap<NodeId, (u64, Option<&Bytes>)> = BTreeMap::new();
        for (key, entry) in self.node_entries() {
            let MemKey::Node { id } = key else {
                continue;
            };
            if memtable.get(id).is_some_and(|(lsn, _)| *lsn >= entry.lsn) {
                continue;
            }
            let payload = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => Some(payload),
            };
            memtable.insert(*id, (entry.lsn, payload));
        }
        let mut memtable = memtable.into_iter().peekable();

        let mut cursors = Vec::new();
        for idx in self.manifest.index.node_descriptors() {
            let desc = &self.manifest.manifest.ssts[idx];
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            let metadata = self.node_sst_metadata(desc, &absolute).await?;
            let label_def = self.label_def_for_node_sst(desc);
            let pages = if predicates.is_empty() {
                None
            } else {
                self.node_property_reader(desc).await?
            };
            // Predicate columns come from property pages when the SST has
            // them, so the overflow JSON (embeddings included) stays unread.
            let columns: &[String] = if pages.is_some() {
                &[]
            } else {
                &predicate_columns
            };
            let (stream, _) = node_scan_limited_async(
                self.store.clone(),
                Path::from(absolute),
                desc.size_bytes,
                &label_def,
                &[],
                Some(columns),
                Some(metadata),
            )
            .await?;
            cursors.push((
                idx,
                SstCursor {
                    stream,
                    label_def,
                    pages,
                    batches: Vec::new(),
                    rows: VecDeque::new(),
                    ordinal: 0,
                    last: None,
                    done: false,
                },
            ));
        }

        let mut out: Vec<NodeView> = Vec::with_capacity(limit.min(1024));
        let mut by_sst: BTreeMap<usize, Vec<(usize, u64)>> = BTreeMap::new();
        let mut decoded_rows = 0usize;
        let mut examined_rows = 0usize;
        let mut row_groups = 0usize;
        while out.len() < limit {
            for (idx, cursor) in &mut cursors {
                while cursor.rows.is_empty() && !cursor.done {
                    let desc = &self.manifest.manifest.ssts[*idx];
                    let Some(reader) = cursor.stream.next_row_group().await? else {
                        if cursor.ordinal != desc.row_count {
                            return Ok(None);
                        }
                        cursor.done = true;
                        break;
                    };
                    row_groups += 1;
                    cursor.batches.clear();
                    for batch in reader {
                        let batch = batch.map_err(|error| {
                            Error::invariant(format!("structural node scan read: {error}"))
                        })?;
                        crate::cancel::check()?;
                        decoded_rows += batch.num_rows();
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
                        // Predicate verdicts per row, from property pages or
                        // the projected columns of this batch.
                        let paged = match &cursor.pages {
                            Some(pages) => {
                                let Some(candidates) = self
                                    .project_node_property_batch(
                                        pages,
                                        desc,
                                        &predicate_columns,
                                        &batch,
                                        cursor.ordinal,
                                        label,
                                    )
                                    .await?
                                else {
                                    return Ok(None);
                                };
                                Some(
                                    candidates
                                        .into_iter()
                                        .map(|(_, _, view)| {
                                            view.is_some_and(|view| {
                                                node_view_matches_predicates(&view, predicates)
                                            })
                                        })
                                        .collect::<Vec<_>>(),
                                )
                            }
                            None => None,
                        };
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
                            let keep = match &paged {
                                Some(paged) => paged[row],
                                None => {
                                    !tomb_col.value(row)
                                        && label.is_none_or(|label| {
                                            row_carries_label(
                                                &batch,
                                                row,
                                                label,
                                                label_id,
                                                dict,
                                                &desc.scope,
                                            )
                                        })
                                        && (predicates.is_empty() || {
                                            let view = NodeView {
                                                id,
                                                labels: BTreeSet::new(),
                                                properties: decode_node_row_properties(
                                                    &batch,
                                                    row,
                                                    &cursor.label_def,
                                                    Some(&predicate_set),
                                                )?,
                                                lsn: 0,
                                                schema_version: 0,
                                            };
                                            node_view_matches_predicates(&view, predicates)
                                        })
                                }
                            };
                            cursor.rows.push_back(SstRow {
                                id,
                                lsn: lsn_col.value(row),
                                keep,
                                batch: cursor.batches.len(),
                                row,
                                ordinal: cursor.ordinal + row as u64,
                            });
                        }
                        cursor.ordinal += batch.num_rows() as u64;
                        cursor.batches.push(batch);
                    }
                }
            }

            let next = cursors
                .iter()
                .filter_map(|(_, cursor)| cursor.rows.front().map(|row| row.id))
                .chain(memtable.peek().map(|(id, _)| *id))
                .min();
            let Some(id) = next else {
                break;
            };
            examined_rows += 1;
            let mut winner: Option<(u64, Option<LabelScanRow>)> = None;
            if memtable.peek().is_some_and(|(mem_id, _)| *mem_id == id) {
                let (_, (lsn, payload)) = memtable.next().expect("peeked");
                let row = match payload {
                    None => None,
                    Some(payload) => {
                        let view = node_view_from_payload(id, lsn, payload, dict, "")?;
                        (label.is_none_or(|label| view.labels.contains(label))
                            && node_view_matches_predicates(&view, predicates))
                        .then_some(LabelScanRow {
                            view,
                            sst_row: None,
                        })
                    }
                };
                winner = Some((lsn, row));
            }
            for (idx, cursor) in &mut cursors {
                if cursor.rows.front().is_none_or(|row| row.id != id) {
                    continue;
                }
                let row = cursor.rows.pop_front().expect("front checked");
                if winner.as_ref().is_some_and(|(best, _)| row.lsn <= *best) {
                    continue;
                }
                let scan_row = if row.keep {
                    let desc = &self.manifest.manifest.ssts[*idx];
                    let batch = &cursor.batches[row.batch];
                    let schema_version = batch
                        .column_by_name(SCHEMA_VERSION)
                        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                        .ok_or_else(|| Error::invariant("__schema_version column missing"))?
                        .value(row.row);
                    Some(LabelScanRow {
                        view: NodeView {
                            id,
                            labels: decode_node_labels(batch, row.row, dict, &desc.scope),
                            properties: BTreeMap::new(),
                            lsn: row.lsn,
                            schema_version,
                        },
                        sst_row: Some((*idx, row.ordinal)),
                    })
                } else {
                    None
                };
                winner = Some((row.lsn, scan_row));
            }
            if let Some((_, Some(row))) = winner {
                if let Some((idx, ordinal)) = row.sst_row {
                    by_sst.entry(idx).or_default().push((out.len(), ordinal));
                }
                out.push(row.view);
            }
        }
        let range_bytes = cursors
            .iter()
            .map(|(_, cursor)| cursor.stream.stats().bytes_read())
            .sum();
        drop(cursors);

        // Within one SST, id order is physical order, so each list is sorted
        // by ordinal as the property readers expect.
        if !self.fill_label_rows(&by_sst, projection, &mut out).await? {
            return Ok(None);
        }
        if let Some(requested) = projection {
            for view in &mut out {
                view.properties
                    .retain(|property, _| requested.iter().any(|r| r == property));
            }
        }
        if let Some(cache) = &self.cache {
            cache.record_limited_node_scan_work(
                decoded_rows,
                examined_rows,
                out.len(),
                row_groups,
                range_bytes,
            );
        }
        Ok(Some(out))
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

        let ssts = &self.manifest.manifest.ssts;
        let descriptors = self.manifest.index.node_descriptors();
        let can_hold = |idx: usize| {
            label
                .is_none_or(|label| node_sst_can_contain_label(&self.manifest.manifest, idx, label))
        };
        let oldest_candidate = descriptors
            .iter()
            .filter(|idx| can_hold(**idx))
            .map(|idx| ssts[*idx].min_lsn)
            .min();
        let mut cursors = Vec::new();
        for idx in descriptors {
            // An SST that cannot hold the label has no candidates, only
            // versions that may hide one. When all of them are older than
            // every candidate they hide nothing, so the merge skips it.
            // Skipping is always safe, since the point read resolves each
            // candidate over all SSTs; keeping a newer one saves that read
            // for every member it relabelled or deleted.
            if !can_hold(idx) && oldest_candidate.is_none_or(|oldest| ssts[idx].max_lsn < oldest) {
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
                MergeCursor {
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

/// One SST's structural rows for [`NodeMerge`], pulled a row group at a time.
struct MergeCursor {
    stream: RangedNodeBatchStream,
    /// `(id, lsn, live and carries the label)`, in id order.
    rows: VecDeque<(NodeId, u64, bool)>,
    last: Option<NodeId>,
    done: bool,
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
    cursors: Vec<(usize, MergeCursor)>,
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
