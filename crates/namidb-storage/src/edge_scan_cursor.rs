//! Edge type scan read one batch at a time.
//!
//! [`Snapshot::scan_edge_type`] merges every forward SST of the type into
//! one map of decoded edges before it returns. The cursor merges the same
//! sources in `(src, dst)` order as it goes and decodes the properties of
//! the edges it returns only, so a caller walking a large type holds one
//! batch of edges at a time.

use std::collections::{BTreeMap, VecDeque};

use uuid::Uuid;

use namidb_core::NodeId;

use super::{
    decode_edge_properties, load_declared_streams, update_edge_winner, EdgeView, Snapshot,
};
use crate::error::Result;
use crate::flush::EdgeWriteRecord;
use crate::manifest::SstKind;
use crate::memtable::{MemKey, MemOp};
use crate::sst::edges::reader::EdgeSstReader;

type EdgeWinners = BTreeMap<(NodeId, NodeId), (u64, Option<EdgeView>)>;

/// One forward SST, walked a key at a time.
struct EdgeSstCursor {
    reader: EdgeSstReader,
    overflow: Option<Vec<Option<String>>>,
    declared: Vec<(String, Vec<Option<String>>)>,
    next_key: usize,
    /// `(src, dst, lsn, tombstone, edge index)` of the current key, by dst.
    rows: VecDeque<(NodeId, NodeId, u64, bool, usize)>,
}

impl EdgeSstCursor {
    fn front(&mut self) -> Result<Option<(NodeId, NodeId)>> {
        while self.rows.is_empty() && (self.next_key as u64) < self.reader.key_count() {
            let first = self.reader.cumulative_edges()[self.next_key] as usize;
            let mut rows: Vec<_> = self
                .reader
                .key_edges(self.next_key)?
                .into_iter()
                .enumerate()
                .map(|(i, row)| {
                    (
                        NodeId::from_uuid(Uuid::from_bytes(row.key_id)),
                        NodeId::from_uuid(Uuid::from_bytes(row.partner_id)),
                        row.lsn,
                        row.tombstone,
                        first + i,
                    )
                })
                .collect();
            rows.sort_by_key(|row| row.1);
            self.rows.extend(rows);
            self.next_key += 1;
        }
        Ok(self.rows.front().map(|row| (row.0, row.1)))
    }
}

/// An edge type scan read one batch at a time, from
/// [`Snapshot::edge_scan_cursor`]. Edges come in the order
/// [`Snapshot::scan_edge_type`] returns them in, `(src, dst)` ascending.
pub struct EdgeScanCursor {
    edge_type: String,
    memtable: std::iter::Peekable<<EdgeWinners as IntoIterator>::IntoIter>,
    ssts: Vec<EdgeSstCursor>,
}

impl std::fmt::Debug for EdgeScanCursor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EdgeScanCursor")
            .field("edge_type", &self.edge_type)
            .finish_non_exhaustive()
    }
}

impl Snapshot<'_> {
    /// Open a cursor over every live edge of `edge_type` at this snapshot.
    ///
    /// ponytail: the forward SSTs have no ranged scan, so their bodies and
    /// property streams are read here in full; only the per-edge decode is
    /// deferred to each batch. A paged key-range read would bound the I/O
    /// too.
    pub async fn edge_scan_cursor(&self, edge_type: &str) -> Result<EdgeScanCursor> {
        let mut memtable: EdgeWinners = BTreeMap::new();
        for (mk, entry) in self.edge_mem_entries_for_type(edge_type) {
            let MemKey::Edge {
                edge_type: et,
                src,
                dst,
            } = mk
            else {
                continue;
            };
            if et != edge_type {
                continue;
            }
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => Some(EdgeView {
                    edge_type: edge_type.to_string(),
                    src: *src,
                    dst: *dst,
                    properties: EdgeWriteRecord::decode(payload)?.properties,
                    lsn: entry.lsn,
                }),
            };
            update_edge_winner(&mut memtable, (*src, *dst), entry.lsn, view);
        }

        let declared_property_names: Vec<String> = self
            .manifest
            .manifest
            .schema
            .edge_type(edge_type)
            .map(|def| def.properties.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();
        let mut ssts = Vec::new();
        for &idx in self
            .manifest
            .index
            .scope_descriptors(SstKind::EdgesFwd, edge_type)
        {
            let desc = &self.manifest.manifest.ssts[idx];
            let reader = EdgeSstReader::open(self.get_sst_body(desc).await?)?;
            ssts.push(EdgeSstCursor {
                overflow: reader.read_overflow_strings()?,
                declared: load_declared_streams(&reader, &declared_property_names)?,
                reader,
                next_key: 0,
                rows: VecDeque::new(),
            });
        }
        Ok(EdgeScanCursor {
            edge_type: edge_type.to_string(),
            memtable: memtable.into_iter().peekable(),
            ssts,
        })
    }
}

impl EdgeScanCursor {
    /// Up to `max` more edges; empty once the type is exhausted.
    pub fn next_batch(&mut self, max: usize) -> Result<Vec<EdgeView>> {
        let mut out = Vec::new();
        while out.len() < max {
            let mut next = self.memtable.peek().map(|(key, _)| *key);
            for sst in &mut self.ssts {
                if let Some(key) = sst.front()? {
                    next = Some(next.map_or(key, |next| next.min(key)));
                }
            }
            let Some(key) = next else {
                break;
            };
            crate::cancel::check()?;
            // Same tie rule as `update_edge_winner`: memtable first, then
            // SSTs in descriptor order, and an equal LSN keeps the first.
            let mut winner: Option<(u64, Option<EdgeView>)> = None;
            if self.memtable.peek().is_some_and(|(k, _)| *k == key) {
                winner = self.memtable.next().map(|(_, version)| version);
            }
            for sst in &mut self.ssts {
                while sst.front()? == Some(key) {
                    let (src, dst, lsn, tombstone, i) =
                        sst.rows.pop_front().expect("front checked");
                    if winner.as_ref().is_some_and(|(best, _)| *best >= lsn) {
                        continue;
                    }
                    let view = if tombstone {
                        None
                    } else {
                        Some(EdgeView {
                            edge_type: self.edge_type.clone(),
                            src,
                            dst,
                            properties: decode_edge_properties(
                                sst.overflow.as_ref().and_then(|v| v.get(i)),
                                &sst.declared,
                                i,
                            )?,
                            lsn,
                        })
                    };
                    winner = Some((lsn, view));
                }
            }
            if let Some((_, Some(view))) = winner {
                out.push(view);
            }
        }
        Ok(out)
    }
}
