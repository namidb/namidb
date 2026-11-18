//! Resident int8 codes of VG6 delta segments.
//!
//! A VG6 delta is exact f32, and [`VectorV6Reader::search_exact`] reads,
//! decompresses and scores every page of it on every query. Under sustained
//! writes an index carries tens of thousands of such rows, so each query paid
//! O(delta bytes) of decompression and f64 arithmetic.
//!
//! [`VectorV6Reader::search_coded`] keeps, per segment and process, an int8
//! code of every live row together with the norm of its quantisation residual
//! (built by the first query's exact scan, bounded by a process-wide byte
//! budget that never evicts live entries, and keyed by the immutable segment
//! id). A query bounds every eligible row's
//! exact score from the codes (see [`crate::vector_tail`]), and reads and
//! scores in f32 only the pages holding rows whose upper bound reaches the
//! `k`-th best lower bound. The hits are the same rows, with the same exact
//! scores and tie order, as the full exact scan.

use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use uuid::Uuid;

use super::{
    compare_hits, decode_block, decode_vector_page, metric_score, validate_query, RankedHit,
    VectorPageRow, VectorV6Hit, VectorV6Reader, VectorV6SearchResult,
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

/// Codes not used for this long are dropped first: a delta merged away by
/// compaction is never asked for again.
const CODE_IDLE: Duration = Duration::from_secs(120);

#[derive(Default)]
struct CodeCache {
    entries: HashMap<Uuid, (Arc<SegmentCodes>, Instant)>,
    bytes: usize,
}

impl CodeCache {
    fn get(&mut self, id: &Uuid) -> Option<Arc<SegmentCodes>> {
        self.entries.get_mut(id).map(|(codes, used)| {
            *used = Instant::now();
            Arc::clone(codes)
        })
    }

    /// Whether `bytes` more fit once idle entries are dropped. Live entries
    /// are never evicted to make room: deltas are read in turn by every
    /// query, so evicting one to admit another only moves the miss (every
    /// query would rebuild codes and none would use them).
    fn admits(&mut self, bytes: usize, budget: usize) -> bool {
        if self.bytes.saturating_add(bytes) > budget {
            let now = Instant::now();
            let idle: Vec<Uuid> = self
                .entries
                .iter()
                .filter(|(_, (_, used))| now.duration_since(*used) >= CODE_IDLE)
                .map(|(id, _)| *id)
                .collect();
            for id in idle {
                if let Some((evicted, _)) = self.entries.remove(&id) {
                    self.bytes -= evicted.bytes;
                }
            }
        }
        self.bytes.saturating_add(bytes) <= budget
    }

    fn insert(&mut self, id: Uuid, codes: Arc<SegmentCodes>, budget: usize) {
        if !self.admits(codes.bytes, budget) {
            return;
        }
        if let Some((old, _)) = self
            .entries
            .insert(id, (Arc::clone(&codes), Instant::now()))
        {
            self.bytes -= old.bytes;
        }
        self.bytes += codes.bytes;
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

/// Resident bytes of one row's code at `dim` dimensions.
fn coded_row_bytes(dim: usize) -> usize {
    std::mem::size_of::<CodedRow>() + std::mem::size_of::<TailCode>() + dim
}

impl VectorV6Reader {
    fn cached_codes(&self) -> Option<Arc<SegmentCodes>> {
        let id = self.footer.binding.segment.sst_id;
        cache()
            .lock()
            .ok()
            .and_then(|mut cache| cache.get(&id))
            .filter(|codes| codes.metric == self.footer.metric && codes.dim == self.footer.dim)
    }

    /// Whether this segment's codes would be admitted now.
    fn codes_admitted(&self) -> bool {
        let budget = code_cache_bytes();
        let estimate = usize::try_from(self.footer.binding.segment.live_payload_count)
            .unwrap_or(usize::MAX)
            .saturating_mul(coded_row_bytes(self.footer.dim as usize));
        budget > 0
            && cache()
                .lock()
                .is_ok_and(|mut cache| cache.admits(estimate, budget))
    }

    /// [`Self::search_exact`] served from resident codes: the same hits with
    /// the same exact scores, reading only the pages that can hold them.
    ///
    /// Without resident codes, a segment the cache admits is answered by one
    /// exact pass that also encodes every row (so the first query costs what
    /// the exact scan costs and later ones read only candidate pages). `None`
    /// when the cache cannot admit it; the caller then runs the exact scan.
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
        let codes = match self.cached_codes() {
            Some(codes) => codes,
            None if self.codes_admitted() => {
                return self
                    .search_exact_and_encode(query, k, groups)
                    .await
                    .map(Some)
            }
            None => return Ok(None),
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
                push_ranked(
                    &mut heap,
                    k,
                    higher_is_better,
                    self.footer.metric,
                    &row,
                    query,
                );
            }
        }
        Ok(Some(finish_ranked(
            heap,
            higher_is_better,
            applied_filter_groups,
            self.footer.pages.len(),
            eligible_rows_seen,
            peak_live_bytes,
        )))
    }

    /// The exact scan of [`Self::search_exact`], encoding every row on the
    /// way and installing the codes.
    async fn search_exact_and_encode(
        &self,
        query: &[f32],
        k: usize,
        groups: &[(String, Vec<SearchFilterValue>)],
    ) -> Result<VectorV6SearchResult> {
        let (allowed, applied_filter_groups) = self.load_filter_mask(groups).await?;
        let higher_is_better = self.footer.metric != VectorMetric::Euclidean;
        let mut heap = BinaryHeap::with_capacity(k.saturating_add(1));
        let mut eligible_rows_seen = 0_usize;
        let mut peak_live_bytes = 0_usize;
        let mut rows = Vec::new();
        let mut bytes = 0_usize;
        for (page_index, page) in self.footer.pages.iter().enumerate() {
            crate::cancel::check()?;
            let compressed = self.source.read_range(page.wire.range()?).await?;
            let raw = decode_block(&compressed, &page.wire, "vector page")?;
            drop(compressed);
            peak_live_bytes = peak_live_bytes.max(raw.len());
            for row in decode_vector_page(&raw, page, self.footer.dim)? {
                let code = encode_code(&row.vector, self.footer.metric);
                bytes += std::mem::size_of::<CodedRow>() + code.as_ref().map_or(0, TailCode::bytes);
                rows.push(CodedRow {
                    ordinal: row.ordinal,
                    page: page_index as u32,
                    code,
                });
                if allowed
                    .as_ref()
                    .is_some_and(|mask| !mask.contains(row.ordinal))
                {
                    continue;
                }
                eligible_rows_seen += 1;
                push_ranked(
                    &mut heap,
                    k,
                    higher_is_better,
                    self.footer.metric,
                    &row,
                    query,
                );
            }
        }
        let codes = Arc::new(SegmentCodes {
            metric: self.footer.metric,
            dim: self.footer.dim,
            rows,
            bytes,
        });
        if let Ok(mut cache) = cache().lock() {
            cache.insert(
                self.footer.binding.segment.sst_id,
                codes,
                code_cache_bytes(),
            );
        }
        Ok(finish_ranked(
            heap,
            higher_is_better,
            applied_filter_groups,
            self.footer.pages.len(),
            eligible_rows_seen,
            peak_live_bytes,
        ))
    }
}

fn push_ranked(
    heap: &mut BinaryHeap<RankedHit>,
    k: usize,
    higher_is_better: bool,
    metric: VectorMetric,
    row: &VectorPageRow,
    query: &[f32],
) {
    heap.push(RankedHit {
        hit: VectorV6Hit {
            node_id: row.node_id,
            lsn: row.lsn,
            payload_fingerprint: row.payload_fingerprint,
            score: metric_score(metric, &row.vector, query),
        },
        higher_is_better,
    });
    if heap.len() > k {
        heap.pop();
    }
}

fn finish_ranked(
    heap: BinaryHeap<RankedHit>,
    higher_is_better: bool,
    applied_filter_groups: usize,
    scanned_pages: usize,
    eligible_rows_seen: usize,
    peak_live_bytes: usize,
) -> VectorV6SearchResult {
    let mut ranked = heap
        .into_iter()
        .map(|candidate| candidate.hit)
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| compare_hits(left, right, higher_is_better));
    VectorV6SearchResult {
        hits: ranked,
        applied_filter_groups,
        // Every page was read or accounted for through its codes.
        scanned_pages,
        eligible_rows_seen,
        peak_live_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes(bytes: usize) -> Arc<SegmentCodes> {
        Arc::new(SegmentCodes {
            metric: VectorMetric::Cosine,
            dim: 4,
            rows: Vec::new(),
            bytes,
        })
    }

    #[test]
    fn live_codes_are_never_evicted_to_admit_others() {
        let mut cache = CodeCache::default();
        let (a, b, c) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        cache.insert(a, codes(60), 100);
        // B does not fit next to a live A: it is refused, A stays.
        cache.insert(b, codes(60), 100);
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&b).is_none());
        cache.insert(c, codes(40), 100);
        assert!(cache.get(&c).is_some());
        assert_eq!(cache.bytes, 100);

        // Once A goes idle it makes room.
        cache.entries.get_mut(&a).unwrap().1 = Instant::now() - CODE_IDLE;
        cache.insert(b, codes(60), 100);
        assert!(cache.get(&a).is_none());
        assert!(cache.get(&b).is_some());
        assert_eq!(cache.bytes, 100);
    }
}
