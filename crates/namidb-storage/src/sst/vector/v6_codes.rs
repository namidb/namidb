//! Resident int8 codes of VG6 delta segments.
//!
//! A VG6 delta is exact f32, and [`VectorV6Reader::search_exact`] reads,
//! decompresses and scores every page of it on every query. Under sustained
//! writes an index carries tens of thousands of such rows, so each query paid
//! O(delta bytes) of decompression and f64 arithmetic.
//!
//! [`VectorV6Reader::search_coded`] keeps, per segment and process, an int8
//! code of every live row together with the norm of its quantisation residual
//! (built once from one full scan, bounded by a process-wide byte budget, and
//! keyed by the immutable segment id). A query bounds every eligible row's
//! exact score from the codes (see [`crate::vector_tail`]), and reads and
//! scores in f32 only the pages holding rows whose upper bound reaches the
//! `k`-th best lower bound. The hits are the same rows, with the same exact
//! scores and tie order, as the full exact scan.

use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use uuid::Uuid;

use super::{
    compare_hits, decode_block, decode_vector_page, metric_score, validate_query, RankedHit,
    VectorV6Hit, VectorV6Reader, VectorV6SearchResult,
};
use crate::error::Result;
use crate::manifest::VectorMetric;
use crate::sst::search_delta::SearchFilterValue;
use crate::vector_tail::{encode_code, retain_possible_top_k, PreparedQuery, TailCode};

/// One live row of a segment.
#[derive(Debug)]
struct CodedRow {
    ordinal: u64,
    /// Index of the page holding the row.
    page: u32,
    /// `None` when the code cannot bound the score; always rescored.
    code: Option<TailCode>,
}

#[derive(Debug)]
struct SegmentCodes {
    metric: VectorMetric,
    dim: u32,
    rows: Vec<CodedRow>,
    bytes: usize,
}

/// Default process-wide budget for resident VG6 codes: about a quarter of the
/// f32 bytes it replaces at 1024 dimensions, so 64 MiB keeps ~55k delta rows
/// resident. Shared-cache reclamation under memory pressure empties it.
const DEFAULT_CODE_CACHE_BYTES: usize = 64 * 1024 * 1024;

fn code_cache_bytes() -> usize {
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("NAMIDB_VECTOR_CODE_CACHE_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_CODE_CACHE_BYTES)
    })
}

#[derive(Default)]
struct CodeCache {
    entries: HashMap<Uuid, (Arc<SegmentCodes>, u64)>,
    bytes: usize,
    tick: u64,
}

impl CodeCache {
    fn get(&mut self, id: &Uuid) -> Option<Arc<SegmentCodes>> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.get_mut(id).map(|(codes, used)| {
            *used = tick;
            Arc::clone(codes)
        })
    }

    fn insert(&mut self, id: Uuid, codes: Arc<SegmentCodes>, budget: usize) {
        if codes.bytes > budget {
            return;
        }
        self.tick += 1;
        if let Some((old, _)) = self.entries.insert(id, (Arc::clone(&codes), self.tick)) {
            self.bytes -= old.bytes;
        }
        self.bytes += codes.bytes;
        while self.bytes > budget {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(id, _)| *id)
            else {
                break;
            };
            if let Some((evicted, _)) = self.entries.remove(&oldest) {
                self.bytes -= evicted.bytes;
            }
        }
    }
}

fn cache() -> &'static Mutex<CodeCache> {
    static CACHE: OnceLock<Mutex<CodeCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(CodeCache::default()))
}

/// Drop every resident code; the next query per segment rebuilds them.
pub(crate) fn clear_resident_codes() {
    if let Ok(mut cache) = cache().lock() {
        cache.entries.clear();
        cache.bytes = 0;
    }
}

impl VectorV6Reader {
    async fn segment_codes(&self) -> Result<Option<Arc<SegmentCodes>>> {
        let id = self.footer.binding.segment.sst_id;
        if let Some(codes) = cache().lock().ok().and_then(|mut cache| cache.get(&id)) {
            if codes.metric == self.footer.metric && codes.dim == self.footer.dim {
                return Ok(Some(codes));
            }
        }
        let budget = code_cache_bytes();
        if budget == 0 {
            return Ok(None);
        }
        let mut rows = Vec::new();
        let mut bytes = 0_usize;
        for (page_index, page) in self.footer.pages.iter().enumerate() {
            crate::cancel::check()?;
            let compressed = self.source.read_range(page.wire.range()?).await?;
            let raw = decode_block(&compressed, &page.wire, "vector page")?;
            drop(compressed);
            for row in decode_vector_page(&raw, page, self.footer.dim)? {
                let code = encode_code(&row.vector, self.footer.metric);
                bytes += std::mem::size_of::<CodedRow>() + code.as_ref().map_or(0, TailCode::bytes);
                rows.push(CodedRow {
                    ordinal: row.ordinal,
                    page: page_index as u32,
                    code,
                });
            }
            if bytes > budget {
                // Too large to keep resident: scan exactly instead.
                return Ok(None);
            }
        }
        let codes = Arc::new(SegmentCodes {
            metric: self.footer.metric,
            dim: self.footer.dim,
            rows,
            bytes,
        });
        if let Ok(mut cache) = cache().lock() {
            cache.insert(id, Arc::clone(&codes), budget);
        }
        Ok(Some(codes))
    }

    /// [`Self::search_exact`] served from resident codes: the same hits with
    /// the same exact scores, reading only the pages that can hold them.
    /// `None` when the segment's codes cannot be kept resident; the caller
    /// then runs the exact scan.
    pub(crate) async fn search_coded(
        &self,
        query: &[f32],
        k: usize,
        groups: &[(String, Vec<SearchFilterValue>)],
    ) -> Result<Option<VectorV6SearchResult>> {
        validate_query(query, self.footer.dim)?;
        if k == 0 || self.footer.binding.segment.live_payload_count == 0 {
            return Ok(None);
        }
        let Some(codes) = self.segment_codes().await? else {
            return Ok(None);
        };
        let (allowed, applied_filter_groups) = self.load_filter_mask(groups).await?;
        let prepared = PreparedQuery::new(query, self.footer.metric);
        let mut always: Vec<usize> = Vec::new();
        let mut bounded: Vec<(f64, f64, usize)> = Vec::new();
        let mut eligible_rows_seen = 0_usize;
        for (index, row) in codes.rows.iter().enumerate() {
            if allowed
                .as_ref()
                .is_some_and(|mask| !mask.contains(row.ordinal))
            {
                continue;
            }
            eligible_rows_seen += 1;
            match row.code.as_ref().and_then(|code| prepared.bounds(code)) {
                Some((lower, upper)) => bounded.push((lower, upper, index)),
                None => always.push(index),
            }
        }
        retain_possible_top_k(&mut bounded, k);
        always.extend(bounded.into_iter().map(|(_, _, index)| index));

        // Rescore the candidates exactly, one read per page that holds any.
        let mut by_page: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
        for index in always {
            let row = &codes.rows[index];
            by_page.entry(row.page).or_default().push(row.ordinal);
        }
        let pages: Vec<u32> = by_page.keys().copied().collect();
        let ranges = pages
            .iter()
            .map(|page| self.footer.pages[*page as usize].wire.range())
            .collect::<Result<Vec<_>>>()?;
        let wires = self.source.read_ranges(&ranges).await?;
        let higher_is_better = self.footer.metric != VectorMetric::Euclidean;
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        let mut peak_live_bytes = 0_usize;
        for (page, compressed) in pages.iter().zip(wires) {
            crate::cancel::check()?;
            let reference = &self.footer.pages[*page as usize];
            let raw = decode_block(&compressed, &reference.wire, "vector page")?;
            peak_live_bytes = peak_live_bytes.max(raw.len());
            let wanted = &by_page[page];
            for row in decode_vector_page(&raw, reference, self.footer.dim)? {
                if !wanted.contains(&row.ordinal) {
                    continue;
                }
                let score = metric_score(self.footer.metric, &row.vector, query);
                heap.push(RankedHit {
                    hit: VectorV6Hit {
                        node_id: row.node_id,
                        lsn: row.lsn,
                        payload_fingerprint: row.payload_fingerprint,
                        score,
                    },
                    higher_is_better,
                });
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        let mut ranked = heap
            .into_iter()
            .map(|candidate| candidate.hit)
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| compare_hits(left, right, higher_is_better));
        Ok(Some(VectorV6SearchResult {
            hits: ranked,
            applied_filter_groups,
            // Every page was accounted for through its codes.
            scanned_pages: self.footer.pages.len(),
            eligible_rows_seen,
            peak_live_bytes,
        }))
    }
}
