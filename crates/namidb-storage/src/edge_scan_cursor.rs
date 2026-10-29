//! Edge type scan read one batch at a time.
//!
//! [`Snapshot::scan_edge_type`] merges every forward SST of the type into
//! one map of decoded edges before it returns. The cursor merges the same
//! sources in `(src, dst)` order as it goes, reading each SST through ranged
//! reads about [`SCAN_CHUNK_EDGES`] edges at a time, and builds the views of
//! the edges it returns only. A batch costs about its own rows plus one
//! chunk per SST, and the cursor holds one chunk per SST.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use uuid::Uuid;

use namidb_core::NodeId;

use super::{decode_edge_properties, update_edge_winner, EdgeView, Snapshot};
use crate::error::{Error, Result};
use crate::flush::EdgeWriteRecord;
use crate::manifest::SstKind;
use crate::memtable::{MemKey, MemOp};
use crate::sst::edges::format::OVERFLOW_JSON_NAME;
use crate::sst::edges::paged_reader::PagedEdgeReader;

type EdgeWinners = BTreeMap<(NodeId, NodeId), (u64, Option<EdgeView>)>;

/// Edges each SST reads per ranged chunk, unless one key has more.
const SCAN_CHUNK_EDGES: u64 = 1024;

/// One forward SST, read a chunk of keys at a time.
struct EdgeSstCursor {
    reader: Arc<PagedEdgeReader>,
    next_key: u64,
    /// `(src, dst, lsn, tombstone, row in the chunk)`, in `(src, dst)` order.
    rows: VecDeque<(NodeId, NodeId, u64, bool, usize)>,
    /// The current chunk's property strings, which `rows` index.
    overflow: Option<Vec<Option<String>>>,
    declared: Vec<(String, Vec<Option<String>>)>,
}

impl EdgeSstCursor {
    async fn front(&mut self, declared_names: &[String]) -> Result<Option<(NodeId, NodeId)>> {
        while self.rows.is_empty() && self.next_key < self.reader.key_count() {
            crate::cancel::check()?;
            let (edges, first_edge, next_key) = self
                .reader
                .read_key_chunk(self.next_key, SCAN_CHUNK_EDGES)
                .await?;
            let chunk = first_edge..first_edge + edges.len() as u64;
            self.overflow = self
                .reader
                .read_property_rows(OVERFLOW_JSON_NAME, chunk.clone())
                .await?;
            self.declared.clear();
            for name in declared_names {
                if let Some(values) = self.reader.read_property_rows(name, chunk.clone()).await? {
                    self.declared.push((name.clone(), values));
                }
            }
            let mut rows: Vec<_> = edges
                .into_iter()
                .enumerate()
                .map(|(i, row)| {
                    (
                        NodeId::from_uuid(Uuid::from_bytes(row.key_id)),
                        NodeId::from_uuid(Uuid::from_bytes(row.partner_id)),
                        row.lsn,
                        row.tombstone,
                        i,
                    )
                })
                .collect();
            // Keys are stored in order, a key's partners may not be.
            rows.sort_by_key(|row| (row.0, row.1));
            self.rows.extend(rows);
            self.next_key = next_key;
        }
        Ok(self.rows.front().map(|row| (row.0, row.1)))
    }
}

/// An edge type scan read one batch at a time, from
/// [`Snapshot::edge_scan_cursor`]. Edges come in the order
/// [`Snapshot::scan_edge_type`] returns them in, `(src, dst)` ascending.
pub struct EdgeScanCursor {
    edge_type: String,
    declared_names: Vec<String>,
    memtable: std::iter::Peekable<<EdgeWinners as IntoIterator>::IntoIter>,
    ssts: Vec<EdgeSstCursor>,
    /// An earlier batch failed.
    failed: bool,
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
    /// It reads the SSTs as it goes, so the caller must keep the snapshot's
    /// files alive until the cursor is dropped.
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

        let declared_names: Vec<String> = self
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
            crate::cancel::check()?;
            let desc = &self.manifest.manifest.ssts[idx];
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            ssts.push(EdgeSstCursor {
                reader: self.fetch_paged_edge_reader(&absolute).await?,
                next_key: 0,
                rows: VecDeque::new(),
                overflow: None,
                declared: Vec::new(),
            });
        }
        Ok(EdgeScanCursor {
            edge_type: edge_type.to_string(),
            declared_names,
            memtable: memtable.into_iter().peekable(),
            ssts,
            failed: false,
        })
    }
}

impl EdgeScanCursor {
    /// Up to `max` more edges; empty once the type is exhausted. After an
    /// error every later call fails too, so no edge is skipped silently.
    pub async fn next_batch(&mut self, max: usize) -> Result<Vec<EdgeView>> {
        if self.failed {
            return Err(Error::precondition(
                "an earlier batch of this edge scan failed; open a new cursor",
            ));
        }
        let batch = self.read_batch(max).await;
        self.failed = batch.is_err();
        batch
    }

    async fn read_batch(&mut self, max: usize) -> Result<Vec<EdgeView>> {
        let mut out = Vec::new();
        while out.len() < max {
            let mut next = self.memtable.peek().map(|(key, _)| *key);
            for sst in &mut self.ssts {
                if let Some(key) = sst.front(&self.declared_names).await? {
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
            // A chunk ends on a key boundary, so every row of `key` is in the
            // chunk already loaded, and the next chunk waits for a later batch.
            for sst in &mut self.ssts {
                while sst.rows.front().is_some_and(|row| (row.0, row.1) == key) {
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
