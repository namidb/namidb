//! Object-store-native vector index (`NAMIVG05`).
//!
//! V4 is a single bincode value and therefore has to be downloaded and decoded
//! in full. V5 is an immutable, range-readable hierarchical IVF index:
//!
//! ```text
//! +----------+----------------------+----------------------+--------+---------+
//! | magic    | zstd nav pages (i8)  | zstd exact pages     | footer | trailer |
//! +----------+----------------------+----------------------+--------+---------+
//! ```
//!
//! The compact footer contains a balanced int8-quantized centroid tree and page
//! directory.
//! Opening an index retains only that metadata. A query probes a bounded number
//! of leaves, evaluates complete local filter bitmaps before approximate top-k,
//! and fetches full-precision pages only for rerank candidates. Consequently a
//! cold query transfers pages proportional to `nprobe`, not the corpus.

pub mod external;

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::io::{Cursor, Read};
use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bincode::Options;
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use namidb_core::quantize::quantize_i8;
use namidb_core::Value;

use super::{
    encode_vector_filter_value, metric_name, metric_score, VectorFilterPostings,
    VectorGraphBuildStats,
};
use crate::error::Error;
use crate::manifest::{VectorIndexDescriptor, VectorMetric};
use crate::search_workspace::shared_search_workspace;

/// V5 object magic. V4 remains the default writer until the engine integration
/// explicitly opts into this independent format.
pub const MAGIC_V5: &[u8; 8] = b"NAMIVG05";
const TRAILER_MAGIC: &[u8; 8] = b"NVG5END!";
const TRAILER_LEN: usize = 8 + 8 + 4;
const FORMAT_VERSION: u16 = 5;
const MAX_FOOTER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_COMPRESSED_BLOCK_BYTES: u64 = 512 * 1024 * 1024;
const MAX_RAW_BLOCK_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_DIM: u32 = 1_048_576;
const MAX_NAV_BATCH_PAGES: usize = 16;
/// Bincode's tree/string containers have allocator overhead in addition to
/// the retained decompressed wire. Reserving six raw bytes per encoded byte is
/// intentionally conservative and keeps the process-wide workspace cap above
/// both buffers while a page is scored.
const NAV_DECODE_WORKSPACE_FACTOR: usize = 6;

/// Immutable byte-range source for one `.vg` object.
///
/// Object-store integrations should override `read_ranges` with a concurrent or
/// coalescing implementation. Cache and manifest concerns deliberately remain
/// outside the format reader.
#[async_trait]
pub trait VectorV5RangeSource: Send + Sync {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error>;

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
        let mut out = Vec::with_capacity(ranges.len());
        for range in ranges {
            out.push(self.read_range(range.clone()).await?);
        }
        Ok(out)
    }
}

/// Deterministic build controls. They are arguments rather than process-global
/// environment variables so tests and concurrent index builds cannot interfere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorV5BuildOptions {
    /// Maximum rows in one independently addressable nav/exact page pair.
    pub target_rows_per_page: usize,
    /// Maximum fan-out of one centroid-tree node.
    pub branch_factor: usize,
    /// Zstd level applied independently to every page.
    pub compression_level: i32,
}

impl Default for VectorV5BuildOptions {
    fn default() -> Self {
        Self {
            target_rows_per_page: 512,
            branch_factor: 8,
            compression_level: 3,
        }
    }
}

/// Query controls for hierarchical probing and exact reranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorV5SearchOptions {
    /// Initial number of leaf pages to probe.
    pub nprobe: usize,
    /// Maximum pages after adaptive widening for selective native filters.
    pub max_nprobe: usize,
    /// Approximate candidates retained per requested result.
    pub rerank_factor: usize,
}

impl Default for VectorV5SearchOptions {
    fn default() -> Self {
        Self {
            nprobe: 4,
            max_nprobe: 64,
            rerank_factor: 8,
        }
    }
}

/// Search output plus the number of requested filter groups applied natively.
///
/// Unsupported groups are intentionally not interpreted as empty: the caller
/// can apply them as residual predicates or select its exact fallback.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorV5SearchResult {
    pub hits: Vec<([u8; 16], f32)>,
    pub applied_filter_groups: usize,
    pub probed_pages: usize,
    /// Eligible rows observed in the pages actually probed, before approximate
    /// rerank/truncation. This is an exact corpus count only when
    /// `probed_pages == VectorV5Reader::page_count()`.
    pub eligible_rows_seen: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlockRef {
    offset: u64,
    len: u32,
    raw_len: u32,
    crc32: u32,
}

impl BlockRef {
    fn range(&self) -> Result<Range<u64>, Error> {
        let end = self
            .offset
            .checked_add(self.len as u64)
            .ok_or_else(|| Error::invariant("vector v5 block range overflows u64"))?;
        Ok(self.offset..end)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PageRef {
    row_count: u32,
    nav: BlockRef,
    exact: BlockRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CentroidNode {
    /// Per-centroid symmetric int8 quantization. Keeping only `dim + 4` bytes
    /// per tree node holds the complete 10M-row/1024d navigation tier below
    /// ~32 MiB at the default 512 rows/page; f32 would require ~90 MiB.
    codes: Vec<i8>,
    scale: f32,
    children: Vec<u32>,
    page: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Footer {
    format_version: u16,
    dim: u32,
    metric: String,
    point_count: u64,
    min_node_id: [u8; 16],
    max_node_id: [u8; 16],
    root: u32,
    nodes: Vec<CentroidNode>,
    pages: Vec<PageRef>,
    filter_properties: Vec<String>,
    target_rows_per_page: u32,
    branch_factor: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NavPage {
    ids: Vec<[u8; 16]>,
    /// Row-major quantized coordinates (`ids.len() * dim` bytes).
    codes: Vec<i8>,
    scales: Vec<f32>,
    /// Complete local bitmaps. Property alternatives OR; groups AND.
    filters: BTreeMap<String, BTreeMap<String, Vec<u64>>>,
}

#[derive(Debug)]
struct BuildRow {
    source_ordinal: u32,
    id: [u8; 16],
    vector: Vec<f32>,
}

#[derive(Debug)]
struct BuildTree {
    nodes: Vec<CentroidNode>,
    leaves: Vec<Vec<usize>>,
}

/// Metadata-only reader. It never stores document IDs, quantized rows, or f32
/// corpus pages beyond one query's local futures/results.
pub struct VectorV5Reader {
    source: Arc<dyn VectorV5RangeSource>,
    file_len: u64,
    footer_offset: u64,
    footer: Footer,
    metric: VectorMetric,
}

impl std::fmt::Debug for VectorV5Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorV5Reader")
            .field("file_len", &self.file_len)
            .field("footer_offset", &self.footer_offset)
            .field("dim", &self.footer.dim)
            .field("metric", &self.footer.metric)
            .field("point_count", &self.footer.point_count)
            .field("centroid_nodes", &self.footer.nodes.len())
            .field("pages", &self.footer.pages.len())
            .finish()
    }
}

/// Build a V5 body without native metadata postings.
pub fn build_body_v5(
    desc: &VectorIndexDescriptor,
    members: Vec<([u8; 16], Vec<f32>)>,
    options: VectorV5BuildOptions,
) -> Result<Option<(Bytes, VectorGraphBuildStats)>, Error> {
    build_body_v5_with_filter_postings(desc, members, VectorFilterPostings::new(), options)
}

/// Build a V5 body with complete postings collected by node compaction.
pub(crate) fn build_body_v5_with_filter_postings(
    desc: &VectorIndexDescriptor,
    members: Vec<([u8; 16], Vec<f32>)>,
    filter_postings: VectorFilterPostings,
    options: VectorV5BuildOptions,
) -> Result<Option<(Bytes, VectorGraphBuildStats)>, Error> {
    validate_build_options(desc, options)?;
    if members.is_empty() {
        return Ok(None);
    }
    let dim = desc.dim as usize;
    let mut rows = Vec::with_capacity(members.len());
    for (ordinal, (id, vector)) in members.into_iter().enumerate() {
        if vector.len() != dim {
            return Err(Error::invariant(format!(
                "vector v5 index `{}`: embedding dim {} != declared {}",
                desc.name,
                vector.len(),
                dim
            )));
        }
        if vector.iter().any(|component| !component.is_finite()) {
            return Err(Error::invariant(format!(
                "vector v5 index `{}` contains a non-finite component",
                desc.name
            )));
        }
        if desc.metric == VectorMetric::Cosine && vector.iter().all(|v| *v == 0.0) {
            continue;
        }
        rows.push(BuildRow {
            source_ordinal: u32::try_from(ordinal)
                .map_err(|_| Error::invariant("vector v5 exceeds u32 source ordinals"))?,
            id,
            vector,
        });
    }
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() > u32::MAX as usize {
        return Err(Error::invariant("vector v5 exceeds u32 row ordinals"));
    }
    rows.sort_by_key(|row| row.id);
    if rows.windows(2).any(|pair| pair[0].id == pair[1].id) {
        return Err(Error::invariant("vector v5 contains duplicate node ids"));
    }

    let mut tree = BuildTree {
        nodes: Vec::new(),
        leaves: Vec::new(),
    };
    let root = partition_rows(
        &rows,
        (0..rows.len()).collect(),
        options.target_rows_per_page,
        options.branch_factor,
        desc.metric,
        &mut tree,
    )?;
    // A present property means complete coverage even when every posting is
    // empty. Preserve that distinction so an empty native filter returns zero
    // rather than being mistaken for an unsupported residual predicate.
    let filter_properties: Vec<String> = filter_postings.keys().cloned().collect();
    let reverse_filters = reverse_filter_postings(filter_postings)?;

    let mut body = MAGIC_V5.to_vec();
    let mut page_refs = Vec::with_capacity(tree.leaves.len());
    for leaf in &tree.leaves {
        let nav = make_nav_page(&rows, leaf, dim, &reverse_filters)?;
        let nav_raw = serialize(&nav, "navigation page")?;
        let nav_ref = append_compressed_block(&mut body, &nav_raw, options.compression_level)?;

        let exact_raw_len = leaf
            .len()
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::invariant("vector v5 exact page size overflows usize"))?;
        let mut exact_raw = Vec::with_capacity(exact_raw_len);
        for &row_index in leaf {
            for component in &rows[row_index].vector {
                exact_raw.extend_from_slice(&component.to_le_bytes());
            }
        }
        let exact_ref = append_compressed_block(&mut body, &exact_raw, options.compression_level)?;
        page_refs.push(PageRef {
            row_count: u32::try_from(leaf.len())
                .map_err(|_| Error::invariant("vector v5 page exceeds u32 rows"))?,
            nav: nav_ref,
            exact: exact_ref,
        });
    }

    let min_node_id = rows
        .first()
        .map(|row| row.id)
        .ok_or_else(|| Error::invariant("vector v5 lost all rows"))?;
    let max_node_id = rows
        .last()
        .map(|row| row.id)
        .ok_or_else(|| Error::invariant("vector v5 lost all rows"))?;
    let footer = Footer {
        format_version: FORMAT_VERSION,
        dim: desc.dim,
        metric: metric_name(desc.metric).to_string(),
        point_count: rows.len() as u64,
        min_node_id,
        max_node_id,
        root,
        nodes: tree.nodes,
        pages: page_refs,
        filter_properties,
        target_rows_per_page: u32::try_from(options.target_rows_per_page)
            .map_err(|_| Error::invariant("vector v5 target page rows exceed u32"))?,
        branch_factor: u16::try_from(options.branch_factor)
            .map_err(|_| Error::invariant("vector v5 branch factor exceeds u16"))?,
    };
    let footer_bytes = serialize(&footer, "footer")?;
    if footer_bytes.is_empty() || footer_bytes.len() as u64 > MAX_FOOTER_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 footer is too large ({} bytes)",
            footer_bytes.len()
        )));
    }
    body.extend_from_slice(&footer_bytes);
    body.extend_from_slice(TRAILER_MAGIC);
    body.extend_from_slice(&(footer_bytes.len() as u64).to_le_bytes());
    body.extend_from_slice(&crc32fast::hash(&footer_bytes).to_le_bytes());

    let stats = VectorGraphBuildStats {
        dim: desc.dim,
        metric: metric_name(desc.metric).to_string(),
        point_count: rows.len() as u64,
        min_node_id,
        max_node_id,
        r: desc.r,
        l_build: desc.l_build,
        alpha: desc.alpha,
        entry_medoid: root,
    };
    Ok(Some((Bytes::from(body), stats)))
}

impl VectorV5Reader {
    /// Open by reading only magic, trailer, and compact centroid footer.
    pub async fn open(source: Arc<dyn VectorV5RangeSource>, file_len: u64) -> Result<Self, Error> {
        let minimum = (MAGIC_V5.len() + TRAILER_LEN) as u64;
        if file_len < minimum {
            return Err(Error::invariant("vector v5 body too short"));
        }
        let trailer_start = file_len - TRAILER_LEN as u64;
        let probes = source
            .read_ranges(&[0..MAGIC_V5.len() as u64, trailer_start..file_len])
            .await?;
        if probes.len() != 2 {
            return Err(Error::invariant(
                "vector v5 range source returned the wrong probe count",
            ));
        }
        require_len(&probes[0], MAGIC_V5.len(), "header")?;
        if probes[0].as_ref() != MAGIC_V5 {
            return Err(Error::invariant("vector v5 magic mismatch"));
        }
        require_len(&probes[1], TRAILER_LEN, "trailer")?;
        let (footer_len, footer_crc) = decode_trailer(&probes[1])?;
        if footer_len == 0 || footer_len > MAX_FOOTER_BYTES {
            return Err(Error::invariant(format!(
                "vector v5 footer length {footer_len} is invalid"
            )));
        }
        let footer_offset = trailer_start
            .checked_sub(footer_len)
            .ok_or_else(|| Error::invariant("vector v5 footer starts before object"))?;
        if footer_offset < MAGIC_V5.len() as u64 {
            return Err(Error::invariant("vector v5 footer overlaps header"));
        }
        let footer_bytes = source.read_range(footer_offset..trailer_start).await?;
        require_len(
            &footer_bytes,
            usize::try_from(footer_len)
                .map_err(|_| Error::invariant("vector v5 footer does not fit usize"))?,
            "footer",
        )?;
        if crc32fast::hash(&footer_bytes) != footer_crc {
            return Err(Error::invariant("vector v5 footer checksum mismatch"));
        }
        let footer: Footer = deserialize(&footer_bytes, footer_len, "footer")?;
        let metric = parse_metric(&footer.metric)?;
        validate_footer(&footer, footer_offset)?;
        Ok(Self {
            source,
            file_len,
            footer_offset,
            footer,
            metric,
        })
    }

    pub fn point_count(&self) -> u64 {
        self.footer.point_count
    }

    pub fn dim(&self) -> u32 {
        self.footer.dim
    }

    pub fn metric(&self) -> VectorMetric {
        self.metric
    }

    pub fn higher_is_better(&self) -> bool {
        self.metric != VectorMetric::Euclidean
    }

    pub fn page_count(&self) -> usize {
        self.footer.pages.len()
    }

    pub fn supports_filter_property(&self, property: &str) -> bool {
        self.footer
            .filter_properties
            .binary_search_by(|candidate| candidate.as_str().cmp(property))
            .is_ok()
    }

    pub fn node_id_bounds(&self) -> ([u8; 16], [u8; 16]) {
        (self.footer.min_node_id, self.footer.max_node_id)
    }

    /// Approximate resident allocation attributable to decoded metadata. This
    /// intentionally excludes the source/caches and is useful for admission
    /// accounting and regression tests.
    pub fn resident_metadata_bytes(&self) -> usize {
        let centroids = self
            .footer
            .nodes
            .iter()
            .map(|node| {
                node.codes.len() * std::mem::size_of::<i8>()
                    + std::mem::size_of::<f32>()
                    + node.children.len() * std::mem::size_of::<u32>()
                    + std::mem::size_of::<CentroidNode>()
            })
            .sum::<usize>();
        centroids
            .saturating_add(self.footer.pages.len() * std::mem::size_of::<PageRef>())
            .saturating_add(
                self.footer
                    .filter_properties
                    .iter()
                    .map(|value| value.len())
                    .sum::<usize>(),
            )
    }

    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        options: VectorV5SearchOptions,
    ) -> Result<Vec<([u8; 16], f32)>, Error> {
        Ok(self
            .search_filter_groups(query, k, options, &[])
            .await?
            .hits)
    }

    /// Search with `(property, OR-values)` groups combined with AND.
    ///
    /// Every supported group is evaluated against local page bitmaps before
    /// approximate scoring and before top-k truncation. Unsupported groups are
    /// reported through `applied_filter_groups`, allowing a residual/fallback.
    pub async fn search_filter_groups(
        &self,
        query: &[f32],
        k: usize,
        options: VectorV5SearchOptions,
        groups: &[(String, Vec<Value>)],
    ) -> Result<VectorV5SearchResult, Error> {
        if query.len() != self.footer.dim as usize {
            return Err(Error::invariant(format!(
                "vector v5 query dimension {} != index dimension {}",
                query.len(),
                self.footer.dim
            )));
        }
        if query.iter().any(|component| !component.is_finite()) {
            return Err(Error::invariant(
                "vector v5 query contains a non-finite component",
            ));
        }
        if k == 0 || self.footer.point_count == 0 {
            return Ok(VectorV5SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                probed_pages: 0,
                eligible_rows_seen: 0,
            });
        }
        let initial_nprobe = options.nprobe.max(1).min(self.footer.pages.len());
        let max_nprobe = options
            .max_nprobe
            .max(initial_nprobe)
            .min(self.footer.pages.len());
        let point_count = usize::try_from(self.footer.point_count)
            .map_err(|_| Error::invariant("vector v5 point count does not fit usize"))?;
        let rerank_k = k
            .saturating_mul(options.rerank_factor.max(1))
            .max(k)
            .min(point_count);
        let final_k = k.min(rerank_k);
        let workspace_plan = vector_workspace_plan(&self.footer, rerank_k, final_k, groups)?;
        let _workspace = shared_search_workspace()
            .reserve("vector v5 search", workspace_plan.required_bytes)
            .await?;
        let supported_groups = prepare_groups(&self.footer.filter_properties, groups);
        let applied_filter_groups = supported_groups.len();
        let leaf_order = self.rank_leaf_pages(query)?;

        let mut approximate = BoundedApproxCandidates::new(rerank_k);
        let mut eligible_rows_seen = 0usize;
        let mut visit_order = 0u64;
        let mut fetched = 0usize;
        let mut target = initial_nprobe;
        loop {
            let batch = &leaf_order[fetched..target];
            for page_batch in batch.chunks(workspace_plan.nav_batch_pages) {
                let ranges = page_batch
                    .iter()
                    .map(|page| self.footer.pages[*page as usize].nav.range())
                    .collect::<Result<Vec<_>, _>>()?;
                let blocks = self.source.read_ranges(&ranges).await?;
                if blocks.len() != ranges.len() {
                    return Err(Error::invariant(
                        "vector v5 range source returned wrong nav block count",
                    ));
                }
                for ((&page, wire), range) in page_batch.iter().zip(blocks).zip(ranges) {
                    require_len(
                        &wire,
                        usize::try_from(range.end - range.start).map_err(|_| {
                            Error::invariant("vector v5 range length overflows usize")
                        })?,
                        "navigation block",
                    )?;
                    let page_ref = &self.footer.pages[page as usize];
                    let nav =
                        decode_nav_page(&wire, &page_ref.nav, page_ref.row_count, self.footer.dim)?;
                    score_nav_page_bounded(
                        self.metric,
                        query,
                        page,
                        &nav,
                        &supported_groups,
                        &mut approximate,
                        &mut eligible_rows_seen,
                        &mut visit_order,
                    )?;
                }
            }
            fetched = target;
            // Widen only when a selective native filter has not produced enough
            // eligible rows. Unfiltered queries obey their explicit nprobe.
            if applied_filter_groups == 0 || eligible_rows_seen >= k || fetched >= max_nprobe {
                break;
            }
            target = fetched.saturating_mul(2).max(fetched + 1).min(max_nprobe);
        }

        let mut approximate = approximate.into_sorted_candidates(self.metric);
        // Group candidates in-place so one exact f32 page can be read,
        // scored and discarded before the next page arrives.
        approximate.sort_by(|left, right| {
            left.page
                .cmp(&right.page)
                .then_with(|| left.row.cmp(&right.row))
                .then_with(|| left.visit_order.cmp(&right.visit_order))
        });
        let mut exact_hits = BoundedExactHits::new(final_k);
        let mut candidate_start = 0usize;
        while candidate_start < approximate.len() {
            let page = approximate[candidate_start].page;
            let mut candidate_end = candidate_start + 1;
            while candidate_end < approximate.len() && approximate[candidate_end].page == page {
                candidate_end += 1;
            }
            let range = self.footer.pages[page as usize].exact.range()?;
            let wire = self.source.read_range(range.clone()).await?;
            require_len(
                &wire,
                usize::try_from(range.end - range.start)
                    .map_err(|_| Error::invariant("vector v5 range length overflows usize"))?,
                "exact block",
            )?;
            let page_ref = &self.footer.pages[page as usize];
            let raw = decode_block(&wire, &page_ref.exact, "exact page")?;
            let expected = (page_ref.row_count as usize)
                .checked_mul(self.footer.dim as usize)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| Error::invariant("vector v5 exact page size overflows"))?;
            if raw.len() != expected {
                return Err(Error::invariant(format!(
                    "vector v5 exact page length {} != expected {expected}",
                    raw.len()
                )));
            }
            let row_bytes = (self.footer.dim as usize)
                .checked_mul(4)
                .ok_or_else(|| Error::invariant("vector v5 row byte size overflows"))?;
            for candidate in &approximate[candidate_start..candidate_end] {
                let start = (candidate.row as usize)
                    .checked_mul(row_bytes)
                    .ok_or_else(|| Error::invariant("vector v5 exact row offset overflows"))?;
                let end = start
                    .checked_add(row_bytes)
                    .ok_or_else(|| Error::invariant("vector v5 exact row end overflows"))?;
                let row = raw
                    .get(start..end)
                    .ok_or_else(|| Error::invariant("vector v5 exact row out of range"))?;
                let mut vector = Vec::with_capacity(self.footer.dim as usize);
                for bytes in row.chunks_exact(4) {
                    vector.push(f32::from_le_bytes(
                        bytes
                            .try_into()
                            .map_err(|_| Error::invariant("vector v5 invalid f32 bytes"))?,
                    ));
                }
                let score = metric_score(self.metric, &vector, query).0 as f32;
                exact_hits.insert(self.metric, candidate.id, score);
            }
            candidate_start = candidate_end;
        }
        let hits = exact_hits.into_sorted_hits(self.metric);
        Ok(VectorV5SearchResult {
            hits,
            applied_filter_groups,
            probed_pages: fetched,
            eligible_rows_seen,
        })
    }

    /// Exhaustively score one authoritative V5 base with bounded memory.
    ///
    /// This is the correctness path for a Search-LSM base followed by exact
    /// VG6 deltas: every clustered page is visited, native filters are applied
    /// before decoding/scoring its f32 rows, and only one page plus an `O(k)`
    /// heap is retained. It therefore proves exact final-corpus parity without
    /// turning `rerank_factor` into an accidental corpus-sized allocation. It
    /// is not the serving ANN path and its scan cost must not be reported as
    /// object-native query latency or byte amplification.
    pub async fn search_exact_filter_groups(
        &self,
        query: &[f32],
        k: usize,
        groups: &[(String, Vec<Value>)],
    ) -> Result<VectorV5SearchResult, Error> {
        if query.len() != self.footer.dim as usize {
            return Err(Error::invariant(format!(
                "vector v5 exact query dimension {} != index dimension {}",
                query.len(),
                self.footer.dim
            )));
        }
        if query.iter().any(|component| !component.is_finite()) {
            return Err(Error::invariant(
                "vector v5 exact query contains a non-finite component",
            ));
        }
        if k == 0 || self.footer.point_count == 0 {
            return Ok(VectorV5SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                probed_pages: 0,
                eligible_rows_seen: 0,
            });
        }

        let supported_groups = prepare_groups(&self.footer.filter_properties, groups);
        let applied_filter_groups = supported_groups.len();
        let workspace_bytes = vector_exact_workspace_bytes(&self.footer, k, groups)?;
        let _workspace = shared_search_workspace()
            .reserve("vector v5 exact base search", workspace_bytes)
            .await?;
        let dim = self.footer.dim as usize;
        let row_bytes = dim
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::invariant("vector v5 exact row byte size overflows"))?;
        let mut exact_hits = BoundedExactHits::new(k);
        let mut eligible_rows_seen = 0usize;
        let mut vector = Vec::with_capacity(dim);

        for page_ref in &self.footer.pages {
            let nav_range = page_ref.nav.range()?;
            let exact_range = page_ref.exact.range()?;
            let blocks = self
                .source
                .read_ranges(&[nav_range.clone(), exact_range.clone()])
                .await?;
            if blocks.len() != 2 {
                return Err(Error::invariant(
                    "vector v5 exact range source returned wrong block count",
                ));
            }
            require_len(
                &blocks[0],
                usize::try_from(nav_range.end - nav_range.start)
                    .map_err(|_| Error::invariant("vector v5 nav range overflows usize"))?,
                "navigation block",
            )?;
            require_len(
                &blocks[1],
                usize::try_from(exact_range.end - exact_range.start)
                    .map_err(|_| Error::invariant("vector v5 exact range overflows usize"))?,
                "exact block",
            )?;
            let nav = decode_nav_page(
                &blocks[0],
                &page_ref.nav,
                page_ref.row_count,
                self.footer.dim,
            )?;
            let raw = decode_block(&blocks[1], &page_ref.exact, "exact page")?;
            let expected = (page_ref.row_count as usize)
                .checked_mul(row_bytes)
                .ok_or_else(|| Error::invariant("vector v5 exact page size overflows"))?;
            if raw.len() != expected {
                return Err(Error::invariant(format!(
                    "vector v5 exact page length {} != expected {expected}",
                    raw.len()
                )));
            }

            for row_index in 0..nav.ids.len() {
                if !row_matches_filters(row_index, &nav.filters, &supported_groups) {
                    continue;
                }
                eligible_rows_seen = eligible_rows_seen
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("vector v5 eligible rows overflow usize"))?;
                let start = row_index
                    .checked_mul(row_bytes)
                    .ok_or_else(|| Error::invariant("vector v5 exact row offset overflows"))?;
                let end = start
                    .checked_add(row_bytes)
                    .ok_or_else(|| Error::invariant("vector v5 exact row end overflows"))?;
                let bytes = raw
                    .get(start..end)
                    .ok_or_else(|| Error::invariant("vector v5 exact row leaves page"))?;
                vector.clear();
                for component in bytes.chunks_exact(size_of::<f32>()) {
                    vector.push(f32::from_le_bytes(component.try_into().map_err(|_| {
                        Error::invariant("vector v5 exact component is truncated")
                    })?));
                }
                let score = metric_score(self.metric, &vector, query).0 as f32;
                exact_hits.insert(self.metric, nav.ids[row_index], score);
            }
        }

        Ok(VectorV5SearchResult {
            hits: exact_hits.into_sorted_hits(self.metric),
            applied_filter_groups,
            probed_pages: self.footer.pages.len(),
            eligible_rows_seen,
        })
    }

    fn rank_leaf_pages(&self, query: &[f32]) -> Result<Vec<u32>, Error> {
        // A parent centroid's score is not an upper bound on any descendant
        // (especially for inner product), so comparing nodes from different
        // depths in one best-first heap can prune the truly closest leaf. Leaf
        // centroids are deliberately the small resident navigation tier: score
        // all C of their int8 representations, then range-read only `nprobe`
        // pages. At 10M rows / 512 rows per page C is ~19.5k, still tiny beside
        // scanning even one corpus dimension tier, and it preserves metric
        // recall without making an invalid geometric-bound assumption. Result
        // scores remain exact because candidates are reranked from f32 pages.
        let mut leaves: Vec<(u32, f64)> = self
            .footer
            .nodes
            .iter()
            .filter_map(|node| {
                node.page.map(|page| {
                    (
                        page,
                        quality(
                            self.metric,
                            approximate_score(self.metric, query, &node.codes, node.scale) as f64,
                        ),
                    )
                })
            })
            .collect();
        if leaves.len() != self.footer.pages.len() {
            return Err(Error::invariant(
                "vector v5 centroid traversal did not reach every page",
            ));
        }
        leaves.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        Ok(leaves.into_iter().map(|(page, _)| page).collect())
    }
}

#[derive(Debug, Clone)]
struct ApproxCandidate {
    page: u32,
    row: u32,
    id: [u8; 16],
    score: f32,
    visit_order: u64,
}

#[derive(Debug)]
struct ApproxHeapEntry {
    quality: f32,
    candidate: ApproxCandidate,
}

impl PartialEq for ApproxHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ApproxHeapEntry {}

impl PartialOrd for ApproxHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// BinaryHeap's root is the worst retained candidate, making replacement
/// O(log rerank_k). Greater means worse.
impl Ord for ApproxHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .quality
            .total_cmp(&self.quality)
            .then_with(|| self.candidate.id.cmp(&other.candidate.id))
            .then_with(|| self.candidate.visit_order.cmp(&other.candidate.visit_order))
    }
}

#[derive(Debug)]
struct BoundedApproxCandidates {
    capacity: usize,
    heap: BinaryHeap<ApproxHeapEntry>,
}

impl BoundedApproxCandidates {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    fn insert(&mut self, metric: VectorMetric, candidate: ApproxCandidate) {
        if self.capacity == 0 {
            return;
        }
        let entry = ApproxHeapEntry {
            quality: quality(metric, candidate.score as f64) as f32,
            candidate,
        };
        if self.heap.len() < self.capacity {
            self.heap.push(entry);
            return;
        }
        let Some(worst) = self.heap.peek() else {
            return;
        };
        if approx_entry_is_better(&entry, worst) {
            let _ = self.heap.pop();
            self.heap.push(entry);
        }
    }

    fn into_sorted_candidates(self, metric: VectorMetric) -> Vec<ApproxCandidate> {
        let mut candidates: Vec<_> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|entry| entry.candidate)
            .collect();
        sort_approx(metric, &mut candidates);
        candidates
    }
}

fn approx_entry_is_better(candidate: &ApproxHeapEntry, current_worst: &ApproxHeapEntry) -> bool {
    candidate
        .quality
        .total_cmp(&current_worst.quality)
        .then_with(|| current_worst.candidate.id.cmp(&candidate.candidate.id))
        .then_with(|| {
            current_worst
                .candidate
                .visit_order
                .cmp(&candidate.candidate.visit_order)
        })
        == Ordering::Greater
}

#[derive(Debug)]
struct ExactHeapEntry {
    quality: f32,
    id: [u8; 16],
    score: f32,
}

impl PartialEq for ExactHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ExactHeapEntry {}

impl PartialOrd for ExactHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExactHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .quality
            .total_cmp(&self.quality)
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[derive(Debug)]
struct BoundedExactHits {
    capacity: usize,
    heap: BinaryHeap<ExactHeapEntry>,
}

impl BoundedExactHits {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    fn insert(&mut self, metric: VectorMetric, id: [u8; 16], score: f32) {
        if self.capacity == 0 {
            return;
        }
        let entry = ExactHeapEntry {
            quality: quality(metric, score as f64) as f32,
            id,
            score,
        };
        if self.heap.len() < self.capacity {
            self.heap.push(entry);
            return;
        }
        let Some(worst) = self.heap.peek() else {
            return;
        };
        let better = entry
            .quality
            .total_cmp(&worst.quality)
            .then_with(|| worst.id.cmp(&entry.id))
            == Ordering::Greater;
        if better {
            let _ = self.heap.pop();
            self.heap.push(entry);
        }
    }

    fn into_sorted_hits(self, metric: VectorMetric) -> Vec<([u8; 16], f32)> {
        let mut hits: Vec<_> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|entry| (entry.id, entry.score))
            .collect();
        sort_hits(metric, &mut hits);
        hits
    }
}

#[derive(Debug, Clone, Copy)]
struct VectorWorkspacePlan {
    nav_batch_pages: usize,
    required_bytes: usize,
}

fn vector_exact_workspace_bytes(
    footer: &Footer,
    k: usize,
    groups: &[(String, Vec<Value>)],
) -> Result<usize, Error> {
    let heap_bytes = checked_mul(
        k,
        size_of::<ExactHeapEntry>(),
        "vector v5 exact base heap workspace",
    )?;
    let group_bytes = groups
        .iter()
        .try_fold(0usize, |total, (property, values)| {
            let values_bytes = values.iter().try_fold(0usize, |subtotal, value| {
                let encoded_bytes = encode_vector_filter_value(value)
                    .map_or(size_of::<Value>(), |encoded| {
                        size_of::<String>().saturating_add(encoded.len())
                    });
                checked_add(subtotal, encoded_bytes, "vector v5 exact filter workspace")
            })?;
            checked_add(
                total,
                property
                    .len()
                    .saturating_add(values_bytes)
                    .saturating_add(size_of::<(String, Vec<Value>)>()),
                "vector v5 exact filter workspace",
            )
        })?;
    let row_vector_bytes = checked_mul(
        footer.dim as usize,
        size_of::<f32>(),
        "vector v5 exact base row workspace",
    )?;
    let page_peak = footer.pages.iter().try_fold(0usize, |peak, page| {
        let nav_decode = checked_mul(
            page.nav.raw_len as usize,
            NAV_DECODE_WORKSPACE_FACTOR,
            "vector v5 exact nav decode workspace",
        )?;
        let page_bytes = [
            page.nav.len as usize,
            page.exact.len as usize,
            nav_decode,
            page.exact.raw_len as usize,
            row_vector_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, value| {
            checked_add(total, value, "vector v5 exact page workspace")
        })?;
        Ok::<usize, Error>(peak.max(page_bytes))
    })?;
    [heap_bytes, group_bytes, row_vector_bytes, page_peak]
        .into_iter()
        .try_fold(0usize, |total, value| {
            checked_add(total, value, "vector v5 exact base total workspace")
        })
}

fn vector_workspace_plan(
    footer: &Footer,
    rerank_k: usize,
    final_k: usize,
    groups: &[(String, Vec<Value>)],
) -> Result<VectorWorkspacePlan, Error> {
    let candidate_bytes = checked_mul(
        rerank_k,
        size_of::<ApproxHeapEntry>().saturating_add(size_of::<ApproxCandidate>()),
        "vector v5 rerank candidate workspace",
    )?;
    let exact_heap_bytes = checked_mul(
        final_k,
        size_of::<ExactHeapEntry>(),
        "vector v5 exact heap workspace",
    )?;
    let leaf_rank_bytes = checked_mul(
        footer.pages.len(),
        size_of::<(u32, f64)>().saturating_add(size_of::<u32>()),
        "vector v5 leaf ranking workspace",
    )?;
    let group_bytes = groups
        .iter()
        .try_fold(0usize, |total, (property, values)| {
            let values_bytes = values.iter().try_fold(0usize, |subtotal, value| {
                let encoded_bytes = encode_vector_filter_value(value)
                    .map_or(size_of::<Value>(), |encoded| {
                        size_of::<String>().saturating_add(encoded.len())
                    });
                checked_add(subtotal, encoded_bytes, "vector v5 filter workspace")
            })?;
            checked_add(
                total,
                property
                    .len()
                    .saturating_add(values_bytes)
                    .saturating_add(size_of::<(String, Vec<Value>)>()),
                "vector v5 filter workspace",
            )
        })?;
    let row_vector_bytes = checked_mul(
        footer.dim as usize,
        size_of::<f32>(),
        "vector v5 exact row workspace",
    )?;
    let base = [
        candidate_bytes,
        exact_heap_bytes,
        leaf_rank_bytes,
        group_bytes,
        row_vector_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, value| {
        checked_add(total, value, "vector v5 base query workspace")
    })?;

    let max_nav_raw = footer
        .pages
        .iter()
        .map(|page| page.nav.raw_len as usize)
        .max()
        .unwrap_or(0);
    let nav_decode = checked_mul(
        max_nav_raw,
        NAV_DECODE_WORKSPACE_FACTOR,
        "vector v5 navigation decode workspace",
    )?;
    let exact_page_peak = footer
        .pages
        .iter()
        .map(|page| {
            (page.exact.len as usize)
                .saturating_add(page.exact.raw_len as usize)
                .saturating_add(row_vector_bytes)
        })
        .max()
        .unwrap_or(0);
    let capacity = shared_search_workspace().capacity_bytes();

    let maximum_batch = MAX_NAV_BATCH_PAGES.min(footer.pages.len()).max(1);
    let mut fallback_required = usize::MAX;
    for batch_pages in (1..=maximum_batch).rev() {
        // Any probed set is bounded by the sum of the `batch_pages` largest
        // compressed nav blocks plus one decoded page.
        let mut largest = Vec::<usize>::with_capacity(batch_pages);
        for page in &footer.pages {
            let len = page.nav.len as usize;
            let position = largest.partition_point(|candidate| *candidate >= len);
            if position < batch_pages {
                largest.insert(position, len);
                if largest.len() > batch_pages {
                    let _ = largest.pop();
                }
            }
        }
        let compressed_batch = largest.into_iter().try_fold(0usize, |total, value| {
            checked_add(total, value, "vector v5 compressed nav batch workspace")
        })?;
        let nav_peak = checked_add(
            compressed_batch,
            nav_decode,
            "vector v5 navigation workspace",
        )?;
        let transient_peak = nav_peak.max(exact_page_peak);
        let required = checked_add(base, transient_peak, "vector v5 total query workspace")?;
        fallback_required = fallback_required.min(required);
        if required <= capacity {
            return Ok(VectorWorkspacePlan {
                nav_batch_pages: batch_pages,
                required_bytes: required,
            });
        }
    }
    Ok(VectorWorkspacePlan {
        nav_batch_pages: 1,
        required_bytes: fallback_required,
    })
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Error> {
    left.checked_add(right)
        .ok_or_else(|| Error::invariant(format!("{label} overflows usize")))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, Error> {
    left.checked_mul(right)
        .ok_or_else(|| Error::invariant(format!("{label} overflows usize")))
}

fn validate_build_options(
    desc: &VectorIndexDescriptor,
    options: VectorV5BuildOptions,
) -> Result<(), Error> {
    if desc.dim == 0 || desc.dim > MAX_DIM {
        return Err(Error::invariant(format!(
            "vector v5 dimension {} is invalid",
            desc.dim
        )));
    }
    if options.target_rows_per_page == 0 || options.target_rows_per_page > u32::MAX as usize {
        return Err(Error::invariant(
            "vector v5 target_rows_per_page must be in 1..=u32::MAX",
        ));
    }
    if !(2..=u16::MAX as usize).contains(&options.branch_factor) {
        return Err(Error::invariant(
            "vector v5 branch_factor must be in 2..=u16::MAX",
        ));
    }
    if !(-7..=22).contains(&options.compression_level) {
        return Err(Error::invariant(
            "vector v5 zstd compression level must be in -7..=22",
        ));
    }
    Ok(())
}

fn partition_rows(
    rows: &[BuildRow],
    mut indices: Vec<usize>,
    target: usize,
    branch: usize,
    metric: VectorMetric,
    tree: &mut BuildTree,
) -> Result<u32, Error> {
    if indices.is_empty() {
        return Err(Error::invariant("vector v5 cannot partition an empty set"));
    }
    let centroid = centroid(rows, &indices, metric)?;
    let (codes, scale) = quantize_i8(&centroid);
    let node_index = u32::try_from(tree.nodes.len())
        .map_err(|_| Error::invariant("vector v5 centroid tree exceeds u32 nodes"))?;
    tree.nodes.push(CentroidNode {
        codes,
        scale,
        children: Vec::new(),
        page: None,
    });
    if indices.len() <= target {
        indices.sort_by_key(|index| rows[*index].id);
        let page = u32::try_from(tree.leaves.len())
            .map_err(|_| Error::invariant("vector v5 exceeds u32 pages"))?;
        tree.leaves.push(indices);
        tree.nodes[node_index as usize].page = Some(page);
        return Ok(node_index);
    }

    let axis = highest_variance_axis(rows, &indices)?;
    let group_count = branch.min(indices.len().div_ceil(target)).max(2);
    let mut children = Vec::with_capacity(group_count);
    // Balanced recursive multi-selection is O(N log B) for fan-out B, unlike
    // assigning every row to every centroid (O(N*B)) or sorting the whole set
    // at each tree level (O(N log N log C)). Across the tree this gives the
    // intended O(N log C * dim) construction bound.
    for chunk in balanced_axis_groups(rows, indices, axis, group_count) {
        children.push(partition_rows(rows, chunk, target, branch, metric, tree)?);
    }
    tree.nodes[node_index as usize].children = children;
    Ok(node_index)
}

fn balanced_axis_groups(
    rows: &[BuildRow],
    mut indices: Vec<usize>,
    axis: usize,
    groups: usize,
) -> Vec<Vec<usize>> {
    if groups <= 1 || indices.len() <= 1 {
        return vec![indices];
    }
    let left_groups = groups / 2;
    let right_groups = groups - left_groups;
    // Proportional split keeps every eventual bucket within one row of N/B.
    let split = indices.len() * left_groups / groups;
    indices.select_nth_unstable_by(split, |left, right| {
        rows[*left].vector[axis]
            .total_cmp(&rows[*right].vector[axis])
            .then_with(|| rows[*left].id.cmp(&rows[*right].id))
    });
    let right = indices.split_off(split);
    let mut out = balanced_axis_groups(rows, indices, axis, left_groups);
    out.extend(balanced_axis_groups(rows, right, axis, right_groups));
    out
}

fn centroid(rows: &[BuildRow], indices: &[usize], metric: VectorMetric) -> Result<Vec<f32>, Error> {
    let dim = rows
        .get(indices[0])
        .ok_or_else(|| Error::invariant("vector v5 centroid row missing"))?
        .vector
        .len();
    let mut center = vec![0.0f64; dim];
    for &index in indices {
        for (dst, value) in center.iter_mut().zip(&rows[index].vector) {
            *dst += *value as f64;
        }
    }
    let scale = 1.0 / indices.len() as f64;
    let mut out: Vec<f32> = center.into_iter().map(|sum| (sum * scale) as f32).collect();
    if metric == VectorMetric::Cosine {
        let norm = out.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut out {
                *value /= norm;
            }
        }
    }
    Ok(out)
}

fn highest_variance_axis(rows: &[BuildRow], indices: &[usize]) -> Result<usize, Error> {
    let dim = rows
        .get(indices[0])
        .ok_or_else(|| Error::invariant("vector v5 variance row missing"))?
        .vector
        .len();
    let mut sum = vec![0.0f64; dim];
    let mut sum_sq = vec![0.0f64; dim];
    for &index in indices {
        for (axis, value) in rows[index].vector.iter().enumerate() {
            let value = *value as f64;
            sum[axis] += value;
            sum_sq[axis] += value * value;
        }
    }
    let n = indices.len() as f64;
    Ok((0..dim)
        .max_by(|left, right| {
            let vl = sum_sq[*left] / n - (sum[*left] / n).powi(2);
            let vr = sum_sq[*right] / n - (sum[*right] / n).powi(2);
            vl.total_cmp(&vr).then_with(|| right.cmp(left))
        })
        .unwrap_or(0))
}

fn reverse_filter_postings(
    postings: VectorFilterPostings,
) -> Result<BTreeMap<u32, Vec<(String, String)>>, Error> {
    let mut reverse: BTreeMap<u32, Vec<(String, String)>> = BTreeMap::new();
    for (property, values) in postings {
        for (value, posting) in values {
            for ordinal in posting.into_sorted_ordinals() {
                reverse
                    .entry(ordinal)
                    .or_default()
                    .push((property.clone(), value.clone()));
            }
        }
    }
    for entries in reverse.values_mut() {
        entries.sort();
        entries.dedup();
    }
    Ok(reverse)
}

fn make_nav_page(
    rows: &[BuildRow],
    leaf: &[usize],
    dim: usize,
    reverse_filters: &BTreeMap<u32, Vec<(String, String)>>,
) -> Result<NavPage, Error> {
    let code_len = leaf
        .len()
        .checked_mul(dim)
        .ok_or_else(|| Error::invariant("vector v5 nav page size overflows usize"))?;
    let mut ids = Vec::with_capacity(leaf.len());
    let mut codes = Vec::with_capacity(code_len);
    let mut scales = Vec::with_capacity(leaf.len());
    let mut filters: BTreeMap<String, BTreeMap<String, Vec<u64>>> = BTreeMap::new();
    let words = leaf.len().div_ceil(64);
    for (local, &row_index) in leaf.iter().enumerate() {
        let row = &rows[row_index];
        ids.push(row.id);
        let (row_codes, scale) = quantize_i8(&row.vector);
        if row_codes.len() != dim || !scale.is_finite() || scale < 0.0 {
            return Err(Error::invariant("vector v5 quantizer returned invalid row"));
        }
        codes.extend(row_codes);
        scales.push(scale);
        if let Some(entries) = reverse_filters.get(&row.source_ordinal) {
            for (property, value) in entries {
                let bitmap = filters
                    .entry(property.clone())
                    .or_default()
                    .entry(value.clone())
                    .or_insert_with(|| vec![0; words]);
                bitmap[local / 64] |= 1u64 << (local % 64);
            }
        }
    }
    Ok(NavPage {
        ids,
        codes,
        scales,
        filters,
    })
}

fn prepare_groups(
    supported_properties: &[String],
    groups: &[(String, Vec<Value>)],
) -> Vec<(String, Vec<String>)> {
    groups
        .iter()
        .filter(|(property, _)| {
            supported_properties
                .binary_search_by(|candidate| candidate.as_str().cmp(property.as_str()))
                .is_ok()
        })
        .filter_map(|(property, values)| {
            let encoded: Option<Vec<String>> = values
                .iter()
                .map(|value| encode_vector_filter_value(value).map(|value| value.into_owned()))
                .collect();
            encoded.map(|values| (property.clone(), values))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn score_nav_page_bounded(
    metric: VectorMetric,
    query: &[f32],
    page: u32,
    nav: &NavPage,
    groups: &[(String, Vec<String>)],
    out: &mut BoundedApproxCandidates,
    eligible_rows_seen: &mut usize,
    visit_order: &mut u64,
) -> Result<(), Error> {
    let dim = query.len();
    for row in 0..nav.ids.len() {
        if !row_matches_filters(row, &nav.filters, groups) {
            continue;
        }
        *eligible_rows_seen = eligible_rows_seen
            .checked_add(1)
            .ok_or_else(|| Error::invariant("vector v5 eligible row count overflows usize"))?;
        let start = row
            .checked_mul(dim)
            .ok_or_else(|| Error::invariant("vector v5 nav row offset overflows"))?;
        let end = start
            .checked_add(dim)
            .ok_or_else(|| Error::invariant("vector v5 nav row end overflows"))?;
        let codes = nav
            .codes
            .get(start..end)
            .ok_or_else(|| Error::invariant("vector v5 nav codes out of range"))?;
        let scale = *nav
            .scales
            .get(row)
            .ok_or_else(|| Error::invariant("vector v5 nav scale out of range"))?;
        let score = approximate_score(metric, query, codes, scale);
        out.insert(
            metric,
            ApproxCandidate {
                page,
                row: u32::try_from(row)
                    .map_err(|_| Error::invariant("vector v5 row exceeds u32"))?,
                id: nav.ids[row],
                score,
                visit_order: *visit_order,
            },
        );
        *visit_order = visit_order
            .checked_add(1)
            .ok_or_else(|| Error::invariant("vector v5 visit order overflows u64"))?;
    }
    Ok(())
}

fn row_matches_filters(
    row: usize,
    filters: &BTreeMap<String, BTreeMap<String, Vec<u64>>>,
    groups: &[(String, Vec<String>)],
) -> bool {
    groups.iter().all(|(property, alternatives)| {
        let Some(values) = filters.get(property) else {
            return false;
        };
        alternatives.iter().any(|value| {
            values
                .get(value)
                .and_then(|words| words.get(row / 64))
                .is_some_and(|word| word & (1u64 << (row % 64)) != 0)
        })
    })
}

fn approximate_score(metric: VectorMetric, query: &[f32], codes: &[i8], scale: f32) -> f32 {
    match metric {
        VectorMetric::Dot => query
            .iter()
            .zip(codes)
            .map(|(query, code)| *query * (*code as f32 * scale))
            .sum(),
        VectorMetric::Cosine => {
            let mut dot = 0.0f32;
            let mut query_norm = 0.0f32;
            let mut vector_norm = 0.0f32;
            for (query, code) in query.iter().zip(codes) {
                let vector = *code as f32 * scale;
                dot += *query * vector;
                query_norm += *query * *query;
                vector_norm += vector * vector;
            }
            if query_norm == 0.0 || vector_norm == 0.0 {
                0.0
            } else {
                dot / (query_norm.sqrt() * vector_norm.sqrt())
            }
        }
        VectorMetric::Euclidean => query
            .iter()
            .zip(codes)
            .map(|(query, code)| {
                let delta = *query - *code as f32 * scale;
                delta * delta
            })
            .sum::<f32>()
            .sqrt(),
    }
}

fn sort_approx(metric: VectorMetric, hits: &mut [ApproxCandidate]) {
    if metric == VectorMetric::Euclidean {
        hits.sort_by(|left, right| {
            left.score
                .total_cmp(&right.score)
                .then_with(|| left.id.cmp(&right.id))
                .then_with(|| left.visit_order.cmp(&right.visit_order))
        });
    } else {
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
                .then_with(|| left.visit_order.cmp(&right.visit_order))
        });
    }
}

fn sort_hits(metric: VectorMetric, hits: &mut [([u8; 16], f32)]) {
    if metric == VectorMetric::Euclidean {
        hits.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
    } else {
        hits.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
    }
}

fn quality(metric: VectorMetric, score: f64) -> f64 {
    if metric == VectorMetric::Euclidean {
        -score
    } else {
        score
    }
}

fn append_compressed_block(
    output: &mut Vec<u8>,
    raw: &[u8],
    compression_level: i32,
) -> Result<BlockRef, Error> {
    if raw.len() as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 raw block too large ({} bytes)",
            raw.len()
        )));
    }
    let wire = zstd::stream::encode_all(Cursor::new(raw), compression_level)
        .map_err(|error| Error::invariant(format!("vector v5 zstd encode failed: {error}")))?;
    if wire.len() as u64 > MAX_COMPRESSED_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 compressed block too large ({} bytes)",
            wire.len()
        )));
    }
    let block = BlockRef {
        offset: output.len() as u64,
        len: u32::try_from(wire.len())
            .map_err(|_| Error::invariant("vector v5 compressed block exceeds u32"))?,
        raw_len: u32::try_from(raw.len())
            .map_err(|_| Error::invariant("vector v5 raw block exceeds u32"))?,
        crc32: crc32fast::hash(&wire),
    };
    output.extend_from_slice(&wire);
    Ok(block)
}

fn decode_nav_page(
    wire: &[u8],
    block: &BlockRef,
    expected_rows: u32,
    dim: u32,
) -> Result<NavPage, Error> {
    let raw = decode_block(wire, block, "navigation page")?;
    let nav: NavPage = deserialize(&raw, block.raw_len as u64, "navigation page")?;
    let rows = expected_rows as usize;
    let expected_codes = rows
        .checked_mul(dim as usize)
        .ok_or_else(|| Error::invariant("vector v5 nav code length overflows"))?;
    if nav.ids.len() != rows || nav.scales.len() != rows || nav.codes.len() != expected_codes {
        return Err(Error::invariant(
            "vector v5 navigation page length mismatch",
        ));
    }
    if nav
        .scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.0)
    {
        return Err(Error::invariant(
            "vector v5 navigation page has invalid scale",
        ));
    }
    let words = rows.div_ceil(64);
    for values in nav.filters.values() {
        for bitmap in values.values() {
            if bitmap.len() != words {
                return Err(Error::invariant(
                    "vector v5 navigation filter bitmap length mismatch",
                ));
            }
            if rows % 64 != 0
                && bitmap.last().is_some_and(|word| {
                    let mask = (1u64 << (rows % 64)) - 1;
                    word & !mask != 0
                })
            {
                return Err(Error::invariant(
                    "vector v5 navigation filter has out-of-range bits",
                ));
            }
        }
    }
    Ok(nav)
}

fn decode_block(wire: &[u8], block: &BlockRef, label: &str) -> Result<Vec<u8>, Error> {
    if wire.len() != block.len as usize {
        return Err(Error::invariant(format!(
            "vector v5 {label} compressed length mismatch"
        )));
    }
    if block.len as u64 > MAX_COMPRESSED_BLOCK_BYTES || block.raw_len as u64 > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 {label} declares an oversized block"
        )));
    }
    if crc32fast::hash(wire) != block.crc32 {
        return Err(Error::invariant(format!(
            "vector v5 {label} checksum mismatch"
        )));
    }
    let decoder = zstd::stream::read::Decoder::new(Cursor::new(wire))
        .map_err(|error| Error::invariant(format!("vector v5 {label} decode failed: {error}")))?;
    // Never trust the zstd frame's advertised content size. Bound expansion by
    // the checksummed directory's already-capped raw length (+1 to detect a
    // lying frame) before allocating the output.
    let mut limited = decoder.take(block.raw_len as u64 + 1);
    let mut raw = Vec::with_capacity(block.raw_len as usize);
    limited
        .read_to_end(&mut raw)
        .map_err(|error| Error::invariant(format!("vector v5 {label} decode failed: {error}")))?;
    if raw.len() != block.raw_len as usize {
        return Err(Error::invariant(format!(
            "vector v5 {label} raw length mismatch"
        )));
    }
    Ok(raw)
}

fn serialize<T: Serialize>(value: &T, label: &str) -> Result<Vec<u8>, Error> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(value)
        .map_err(|error| Error::invariant(format!("vector v5 {label} encode failed: {error}")))
}

fn deserialize<T: DeserializeOwned>(bytes: &[u8], limit: u64, label: &str) -> Result<T, Error> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(limit)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| Error::invariant(format!("vector v5 {label} decode failed: {error}")))
}

fn decode_trailer(bytes: &[u8]) -> Result<(u64, u32), Error> {
    require_len(bytes, TRAILER_LEN, "trailer")?;
    if &bytes[..8] != TRAILER_MAGIC {
        return Err(Error::invariant("vector v5 trailer magic mismatch"));
    }
    let footer_len = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| Error::invariant("vector v5 invalid footer length bytes"))?,
    );
    let crc = u32::from_le_bytes(
        bytes[16..20]
            .try_into()
            .map_err(|_| Error::invariant("vector v5 invalid footer crc bytes"))?,
    );
    Ok((footer_len, crc))
}

fn require_len(bytes: &[u8], expected: usize, label: &str) -> Result<(), Error> {
    if bytes.len() != expected {
        return Err(Error::invariant(format!(
            "vector v5 {label} length {} != expected {expected}",
            bytes.len()
        )));
    }
    Ok(())
}

fn parse_metric(name: &str) -> Result<VectorMetric, Error> {
    match name {
        "cosine" => Ok(VectorMetric::Cosine),
        "dot" => Ok(VectorMetric::Dot),
        "euclidean" => Ok(VectorMetric::Euclidean),
        _ => Err(Error::invariant(format!(
            "vector v5 unknown metric `{name}`"
        ))),
    }
}

fn validate_footer(footer: &Footer, footer_offset: u64) -> Result<(), Error> {
    if footer.format_version != FORMAT_VERSION {
        return Err(Error::invariant(format!(
            "vector v5 unsupported format version {}",
            footer.format_version
        )));
    }
    if footer.dim == 0 || footer.dim > MAX_DIM {
        return Err(Error::invariant("vector v5 footer has invalid dimension"));
    }
    if footer.point_count == 0 || footer.point_count > u32::MAX as u64 {
        return Err(Error::invariant("vector v5 footer has invalid point count"));
    }
    if footer.nodes.is_empty() || footer.pages.is_empty() {
        return Err(Error::invariant(
            "vector v5 footer has no centroid tree/pages",
        ));
    }
    if footer.root as usize >= footer.nodes.len() {
        return Err(Error::invariant("vector v5 footer root out of range"));
    }
    if footer.target_rows_per_page == 0 || footer.branch_factor < 2 {
        return Err(Error::invariant(
            "vector v5 footer has invalid build parameters",
        ));
    }
    if footer
        .filter_properties
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(Error::invariant(
            "vector v5 filter properties are not strictly sorted",
        ));
    }
    let mut row_sum = 0u64;
    let mut previous_end = MAGIC_V5.len() as u64;
    for page in &footer.pages {
        if page.row_count == 0 || page.row_count > footer.target_rows_per_page {
            return Err(Error::invariant("vector v5 page has invalid row count"));
        }
        row_sum = row_sum
            .checked_add(page.row_count as u64)
            .ok_or_else(|| Error::invariant("vector v5 page row sum overflows"))?;
        for block in [&page.nav, &page.exact] {
            if block.len == 0
                || block.raw_len == 0
                || block.len as u64 > MAX_COMPRESSED_BLOCK_BYTES
                || block.raw_len as u64 > MAX_RAW_BLOCK_BYTES
            {
                return Err(Error::invariant("vector v5 page block size is invalid"));
            }
            let range = block.range()?;
            if range.start < previous_end || range.end > footer_offset {
                return Err(Error::invariant(
                    "vector v5 page blocks overlap or leave object bounds",
                ));
            }
            previous_end = range.end;
        }
        let exact_len = (page.row_count as u64)
            .checked_mul(footer.dim as u64)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| Error::invariant("vector v5 exact page size overflows"))?;
        if page.exact.raw_len as u64 != exact_len {
            return Err(Error::invariant(
                "vector v5 exact page declared length mismatch",
            ));
        }
    }
    if row_sum != footer.point_count {
        return Err(Error::invariant(
            "vector v5 page rows do not equal point count",
        ));
    }
    for node in &footer.nodes {
        if node.codes.len() != footer.dim as usize
            || !node.scale.is_finite()
            || node.scale < 0.0
            || (node.scale == 0.0 && node.codes.iter().any(|value| *value != 0))
        {
            return Err(Error::invariant(
                "vector v5 centroid has invalid coordinates",
            ));
        }
        match (node.page, node.children.is_empty()) {
            (Some(page), true) if page as usize >= footer.pages.len() => {
                return Err(Error::invariant("vector v5 leaf page out of range"));
            }
            (Some(_), true) => {}
            (None, false) => {
                if node.children.len() > footer.branch_factor as usize
                    || node
                        .children
                        .iter()
                        .any(|child| *child as usize >= footer.nodes.len())
                {
                    return Err(Error::invariant("vector v5 centroid children are invalid"));
                }
            }
            _ => {
                return Err(Error::invariant(
                    "vector v5 centroid must be either branch or leaf",
                ));
            }
        }
    }
    validate_tree_reachability(footer)?;
    Ok(())
}

fn validate_tree_reachability(footer: &Footer) -> Result<(), Error> {
    let mut state = vec![0u8; footer.nodes.len()];
    let mut stack = vec![(footer.root, false)];
    let mut pages = vec![false; footer.pages.len()];
    while let Some((node_index, exiting)) = stack.pop() {
        let slot = state
            .get_mut(node_index as usize)
            .ok_or_else(|| Error::invariant("vector v5 tree node out of range"))?;
        if exiting {
            *slot = 2;
            continue;
        }
        if *slot == 1 {
            return Err(Error::invariant("vector v5 centroid tree contains a cycle"));
        }
        if *slot == 2 {
            return Err(Error::invariant("vector v5 centroid tree shares a child"));
        }
        *slot = 1;
        stack.push((node_index, true));
        let node = &footer.nodes[node_index as usize];
        if let Some(page) = node.page {
            if std::mem::replace(&mut pages[page as usize], true) {
                return Err(Error::invariant(
                    "vector v5 page is referenced by multiple leaves",
                ));
            }
        } else {
            for &child in node.children.iter().rev() {
                stack.push((child, false));
            }
        }
    }
    if state.iter().any(|value| *value != 2) || pages.iter().any(|value| !*value) {
        return Err(Error::invariant(
            "vector v5 centroid tree has unreachable nodes/pages",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::manifest::VectorQuantization;
    use crate::sst::vector::VectorFilterPosting;

    #[derive(Debug)]
    struct TrackingSource {
        bytes: Bytes,
        ranges: Mutex<Vec<Range<u64>>>,
        batch_sizes: Mutex<Vec<usize>>,
    }

    impl TrackingSource {
        fn new(bytes: Bytes) -> Self {
            Self {
                bytes,
                ranges: Mutex::new(Vec::new()),
                batch_sizes: Mutex::new(Vec::new()),
            }
        }

        fn bytes_read(&self) -> u64 {
            self.ranges
                .lock()
                .expect("tracking lock")
                .iter()
                .map(|range| range.end - range.start)
                .sum()
        }

        fn range_count(&self) -> usize {
            self.ranges.lock().expect("tracking lock").len()
        }

        fn max_batch_size(&self) -> usize {
            self.batch_sizes
                .lock()
                .expect("batch tracking lock")
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
        }
    }

    #[async_trait]
    impl VectorV5RangeSource for TrackingSource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error> {
            if range.start > range.end || range.end > self.bytes.len() as u64 {
                return Err(Error::invariant("test range source request out of bounds"));
            }
            self.ranges
                .lock()
                .expect("tracking lock")
                .push(range.clone());
            Ok(self.bytes.slice(range.start as usize..range.end as usize))
        }

        async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
            self.batch_sizes
                .lock()
                .expect("batch tracking lock")
                .push(ranges.len());
            let mut out = Vec::with_capacity(ranges.len());
            for range in ranges {
                out.push(self.read_range(range.clone()).await?);
            }
            Ok(out)
        }
    }

    fn descriptor(metric: VectorMetric, dim: u32) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: format!("v5-{metric:?}"),
            label: "Doc".to_string(),
            property: "embedding".to_string(),
            dim,
            metric,
            r: 32,
            l_build: 64,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        }
    }

    fn id(value: u32) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[..4].copy_from_slice(&value.to_le_bytes());
        id
    }

    fn id_value(value: &[u8; 16]) -> u32 {
        u32::from_le_bytes(value[..4].try_into().expect("id bytes"))
    }

    fn normalize(vector: &mut [f32]) {
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in vector {
                *value /= norm;
            }
        }
    }

    fn all_probe_options(page_count: usize) -> VectorV5SearchOptions {
        VectorV5SearchOptions {
            nprobe: page_count,
            max_nprobe: page_count,
            rerank_factor: 32,
        }
    }

    #[test]
    fn bounded_heaps_match_stable_full_sort_for_every_metric() {
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let candidates: Vec<_> = (0..257u32)
                .map(|ordinal| ApproxCandidate {
                    page: ordinal % 17,
                    row: ordinal / 17,
                    id: id((ordinal * 37) % 257),
                    score: if ordinal % 11 == 0 {
                        0.5
                    } else {
                        (ordinal as f32 * 0.03125).sin()
                    },
                    visit_order: ordinal as u64,
                })
                .collect();
            let mut expected = candidates.clone();
            sort_approx(metric, &mut expected);
            expected.truncate(23);

            let mut bounded = BoundedApproxCandidates::new(23);
            for candidate in candidates {
                bounded.insert(metric, candidate);
            }
            let actual = bounded.into_sorted_candidates(metric);
            assert_eq!(
                actual.iter().map(|hit| hit.id).collect::<Vec<_>>(),
                expected.iter().map(|hit| hit.id).collect::<Vec<_>>()
            );
            assert_eq!(
                actual.iter().map(|hit| hit.score).collect::<Vec<_>>(),
                expected.iter().map(|hit| hit.score).collect::<Vec<_>>()
            );

            let mut exact: Vec<_> = (0..257u32)
                .map(|ordinal| {
                    (
                        id((ordinal * 53) % 257),
                        if ordinal % 13 == 0 {
                            -0.25
                        } else {
                            (ordinal as f32 * 0.0625).cos()
                        },
                    )
                })
                .collect();
            let mut expected_exact = exact.clone();
            sort_hits(metric, &mut expected_exact);
            expected_exact.truncate(19);
            let mut bounded_exact = BoundedExactHits::new(19);
            for (node_id, score) in exact.drain(..) {
                bounded_exact.insert(metric, node_id, score);
            }
            assert_eq!(
                bounded_exact.into_sorted_hits(metric),
                expected_exact,
                "{metric:?}"
            );
        }
    }

    #[tokio::test]
    async fn roundtrip_returns_metric_faithful_scores_for_all_metrics() {
        let members = vec![
            (id(0), vec![1.0, 0.0, 0.0]),
            (id(1), vec![0.0, 1.0, 0.0]),
            (id(2), vec![2.0, 0.0, 0.0]),
            (id(3), vec![-1.0, 0.0, 0.0]),
            (id(4), vec![1.0, 0.25, 0.0]),
            (id(5), vec![0.1, -0.2, 0.7]),
        ];
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let (body, stats) = build_body_v5(
                &descriptor(metric, 3),
                members.clone(),
                VectorV5BuildOptions {
                    target_rows_per_page: 2,
                    branch_factor: 2,
                    compression_level: 1,
                },
            )
            .expect("build")
            .expect("body");
            assert_eq!(stats.metric, metric_name(metric));
            let source = Arc::new(TrackingSource::new(body.clone()));
            let reader = VectorV5Reader::open(source, body.len() as u64)
                .await
                .expect("open");
            let hits = reader
                .search(&[1.0, 0.0, 0.0], 3, all_probe_options(reader.page_count()))
                .await
                .expect("search");
            assert_eq!(hits.len(), 3);
            let expected_first = match metric {
                VectorMetric::Cosine => 0,
                VectorMetric::Dot => 2,
                VectorMetric::Euclidean => 0,
            };
            assert_eq!(id_value(&hits[0].0), expected_first);
            let expected_score = match metric {
                VectorMetric::Cosine => 1.0,
                VectorMetric::Dot => 2.0,
                VectorMetric::Euclidean => 0.0,
            };
            assert!(
                (hits[0].1 - expected_score).abs() < 1e-6,
                "{metric:?}: {:?}",
                hits
            );
        }
    }

    #[tokio::test]
    async fn singleton_roundtrip_searches_every_metric_with_native_filters() {
        let node_id = id(17);
        let vector = vec![2.0, -1.0, 0.5];
        let query = vector.clone();
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let postings = BTreeMap::from([
                (
                    "active".to_string(),
                    BTreeMap::from([("b:1".to_string(), VectorFilterPosting::Sparse(vec![0]))]),
                ),
                (
                    "kind".to_string(),
                    BTreeMap::from([("s:law".to_string(), VectorFilterPosting::Sparse(vec![0]))]),
                ),
            ]);
            let (body, stats) = build_body_v5_with_filter_postings(
                &descriptor(metric, 3),
                vec![(node_id, vector.clone())],
                postings,
                VectorV5BuildOptions {
                    target_rows_per_page: 1,
                    branch_factor: 2,
                    compression_level: 1,
                },
            )
            .expect("singleton build")
            .expect("singleton body");
            assert_eq!(stats.point_count, 1, "{metric:?}");
            assert_eq!(stats.min_node_id, node_id, "{metric:?}");
            assert_eq!(stats.max_node_id, node_id, "{metric:?}");

            let source = Arc::new(TrackingSource::new(body.clone()));
            let reader = VectorV5Reader::open(source, body.len() as u64)
                .await
                .expect("singleton open");
            assert_eq!(reader.point_count(), 1, "{metric:?}");
            assert_eq!(reader.page_count(), 1, "{metric:?}");
            assert_eq!(reader.node_id_bounds(), (node_id, node_id), "{metric:?}");
            assert_eq!(reader.footer.root, 0, "{metric:?}");
            assert_eq!(reader.footer.nodes.len(), 1, "{metric:?}");
            assert_eq!(reader.footer.nodes[0].page, Some(0), "{metric:?}");
            assert!(reader.footer.nodes[0].children.is_empty(), "{metric:?}");
            assert_eq!(reader.footer.pages[0].row_count, 1, "{metric:?}");

            let groups = [
                ("active".to_string(), vec![Value::Bool(true)]),
                ("kind".to_string(), vec![Value::Str("law".to_string())]),
                (
                    "unsupported".to_string(),
                    vec![Value::Str("residual".to_string())],
                ),
            ];
            let filtered = reader
                .search_filter_groups(&query, 5, all_probe_options(reader.page_count()), &groups)
                .await
                .expect("singleton filtered search");
            let expected_score = metric_score(metric, &vector, &query).0 as f32;
            assert_eq!(filtered.applied_filter_groups, 2, "{metric:?}");
            assert_eq!(filtered.probed_pages, 1, "{metric:?}");
            assert_eq!(filtered.eligible_rows_seen, 1, "{metric:?}");
            assert_eq!(filtered.hits.len(), 1, "{metric:?}");
            assert_eq!(filtered.hits[0].0, node_id, "{metric:?}");
            assert!(
                (filtered.hits[0].1 - expected_score).abs() < 1e-6,
                "{metric:?}: {:?}",
                filtered.hits
            );

            let exact = reader
                .search_exact_filter_groups(&query, 5, &groups)
                .await
                .expect("singleton exact filtered search");
            assert_eq!(exact, filtered, "{metric:?}");

            let rejected = reader
                .search_filter_groups(
                    &query,
                    1,
                    all_probe_options(reader.page_count()),
                    &[("active".to_string(), vec![Value::Bool(false)])],
                )
                .await
                .expect("singleton rejecting filter");
            assert_eq!(rejected.applied_filter_groups, 1, "{metric:?}");
            assert_eq!(rejected.probed_pages, 1, "{metric:?}");
            assert_eq!(rejected.eligible_rows_seen, 0, "{metric:?}");
            assert!(rejected.hits.is_empty(), "{metric:?}");
        }
    }

    #[test]
    fn singleton_cosine_zero_is_empty_while_other_metrics_materialize_it() {
        let options = VectorV5BuildOptions {
            target_rows_per_page: 1,
            branch_factor: 2,
            compression_level: 1,
        };
        assert!(build_body_v5(
            &descriptor(VectorMetric::Cosine, 3),
            vec![(id(1), vec![0.0; 3])],
            options,
        )
        .expect("cosine zero build")
        .is_none());
        for metric in [VectorMetric::Dot, VectorMetric::Euclidean] {
            let (_, stats) =
                build_body_v5(&descriptor(metric, 3), vec![(id(1), vec![0.0; 3])], options)
                    .expect("zero build")
                    .expect("singleton zero body");
            assert_eq!(stats.point_count, 1, "{metric:?}");
        }
    }

    #[tokio::test]
    async fn singleton_page_corruption_is_detected_when_the_page_is_read() {
        let (body, _) = build_body_v5(
            &descriptor(VectorMetric::Dot, 2),
            vec![(id(1), vec![1.0, 2.0])],
            VectorV5BuildOptions {
                target_rows_per_page: 1,
                branch_factor: 2,
                compression_level: 1,
            },
        )
        .expect("singleton build")
        .expect("singleton body");
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source, body.len() as u64)
            .await
            .expect("singleton metadata");
        let nav_offset = reader.footer.pages[0].nav.offset as usize;

        let mut corrupt = body.to_vec();
        corrupt[nav_offset] ^= 0x01;
        let corrupt = Bytes::from(corrupt);
        let source = Arc::new(TrackingSource::new(corrupt.clone()));
        let reader = VectorV5Reader::open(source, corrupt.len() as u64)
            .await
            .expect("footer remains intact");
        assert!(reader
            .search(&[1.0, 2.0], 1, all_probe_options(reader.page_count()))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn native_filters_apply_or_within_group_and_and_between_groups_before_k() {
        let members: Vec<_> = (0..192u32)
            .map(|ordinal| {
                (
                    id(ordinal),
                    vec![1.0 - ordinal as f32 / 1_000.0, ordinal as f32 / 10_000.0],
                )
            })
            .collect();
        let red: Vec<u32> = (0..192).filter(|ordinal| ordinal % 3 == 0).collect();
        let blue: Vec<u32> = (0..192).filter(|ordinal| ordinal % 3 == 1).collect();
        let active: Vec<u32> = (96..192).collect();
        let postings = BTreeMap::from([
            (
                "color".to_string(),
                BTreeMap::from([
                    ("s:red".to_string(), VectorFilterPosting::Sparse(red)),
                    ("s:blue".to_string(), VectorFilterPosting::Sparse(blue)),
                ]),
            ),
            (
                "active".to_string(),
                BTreeMap::from([("b:1".to_string(), VectorFilterPosting::Sparse(active))]),
            ),
        ]);
        let (body, _) = build_body_v5_with_filter_postings(
            &descriptor(VectorMetric::Dot, 2),
            members,
            postings,
            VectorV5BuildOptions {
                target_rows_per_page: 12,
                branch_factor: 4,
                compression_level: 1,
            },
        )
        .expect("build")
        .expect("body");
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source, body.len() as u64)
            .await
            .expect("open");
        let result = reader
            .search_filter_groups(
                &[1.0, 0.0],
                15,
                VectorV5SearchOptions {
                    nprobe: 1,
                    max_nprobe: reader.page_count(),
                    rerank_factor: 8,
                },
                &[
                    (
                        "color".to_string(),
                        vec![Value::Str("red".into()), Value::Str("blue".into())],
                    ),
                    ("active".to_string(), vec![Value::Bool(true)]),
                    ("residual".to_string(), vec![Value::Str("ignored".into())]),
                ],
            )
            .await
            .expect("search");
        assert_eq!(result.applied_filter_groups, 2);
        assert_eq!(result.hits.len(), 15);
        assert!(result.eligible_rows_seen >= result.hits.len());
        assert!(result.probed_pages > 1, "selective filter should widen");
        for (node_id, _) in result.hits {
            let ordinal = id_value(&node_id);
            assert!(ordinal >= 96);
            assert!(ordinal % 3 == 0 || ordinal % 3 == 1);
        }
    }

    #[tokio::test]
    async fn exact_base_scan_is_page_bounded_and_matches_filtered_brute_force() {
        let members: Vec<_> = (0..192u32)
            .map(|ordinal| {
                (
                    id(ordinal),
                    vec![1.0 - ordinal as f32 / 1_000.0, ordinal as f32 / 10_000.0],
                )
            })
            .collect();
        let postings = BTreeMap::from([
            (
                "color".to_string(),
                BTreeMap::from([
                    (
                        "s:red".to_string(),
                        VectorFilterPosting::Sparse(
                            (0..192).filter(|ordinal| ordinal % 3 == 0).collect(),
                        ),
                    ),
                    (
                        "s:blue".to_string(),
                        VectorFilterPosting::Sparse(
                            (0..192).filter(|ordinal| ordinal % 3 == 1).collect(),
                        ),
                    ),
                ]),
            ),
            (
                "active".to_string(),
                BTreeMap::from([(
                    "b:1".to_string(),
                    VectorFilterPosting::Sparse((96..192).collect()),
                )]),
            ),
        ]);
        let (body, _) = build_body_v5_with_filter_postings(
            &descriptor(VectorMetric::Dot, 2),
            members.clone(),
            postings,
            VectorV5BuildOptions {
                target_rows_per_page: 12,
                branch_factor: 4,
                compression_level: 1,
            },
        )
        .expect("build")
        .expect("body");
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source.clone(), body.len() as u64)
            .await
            .expect("open");
        let groups = [
            (
                "color".to_string(),
                vec![Value::Str("red".into()), Value::Str("blue".into())],
            ),
            ("active".to_string(), vec![Value::Bool(true)]),
        ];
        let actual = reader
            .search_exact_filter_groups(&[1.0, 0.0], 15, &groups)
            .await
            .expect("exact base search");

        let mut expected = members
            .iter()
            .enumerate()
            .filter(|(ordinal, _)| *ordinal >= 96 && (*ordinal % 3 == 0 || *ordinal % 3 == 1))
            .map(|(_, (node_id, vector))| {
                (
                    *node_id,
                    metric_score(VectorMetric::Dot, vector, &[1.0, 0.0]).0 as f32,
                )
            })
            .collect::<Vec<_>>();
        sort_hits(VectorMetric::Dot, &mut expected);
        expected.truncate(15);

        assert_eq!(actual.hits, expected);
        assert_eq!(actual.applied_filter_groups, 2);
        assert_eq!(actual.probed_pages, reader.page_count());
        assert_eq!(actual.eligible_rows_seen, 64);
        assert_eq!(source.max_batch_size(), 2);
    }

    #[tokio::test]
    async fn deterministic_build_is_independent_of_unfiltered_input_order() {
        let members: Vec<_> = (0..96u32)
            .map(|ordinal| {
                (
                    id(ordinal),
                    vec![ordinal as f32 / 97.0, ((ordinal * 17) % 31) as f32 / 31.0],
                )
            })
            .collect();
        let options = VectorV5BuildOptions {
            target_rows_per_page: 11,
            branch_factor: 3,
            compression_level: 1,
        };
        let a = build_body_v5(
            &descriptor(VectorMetric::Cosine, 2),
            members.clone(),
            options,
        )
        .expect("build a")
        .expect("body a")
        .0;
        let mut reversed = members;
        reversed.reverse();
        let b = build_body_v5(&descriptor(VectorMetric::Cosine, 2), reversed, options)
            .expect("build b")
            .expect("body b")
            .0;
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn corrupt_truncated_and_oversized_footer_are_rejected_without_panics() {
        let members: Vec<_> = (0..32u32)
            .map(|ordinal| (id(ordinal), vec![ordinal as f32 + 1.0, 1.0]))
            .collect();
        let (body, _) = build_body_v5(
            &descriptor(VectorMetric::Dot, 2),
            members,
            VectorV5BuildOptions {
                target_rows_per_page: 8,
                branch_factor: 2,
                compression_level: 1,
            },
        )
        .expect("build")
        .expect("body");

        let truncated = body.slice(..body.len() - 5);
        let source = Arc::new(TrackingSource::new(truncated.clone()));
        assert!(VectorV5Reader::open(source, truncated.len() as u64)
            .await
            .is_err());

        let mut bad_crc = body.to_vec();
        let trailer = body.len() - TRAILER_LEN;
        let footer_len =
            u64::from_le_bytes(bad_crc[trailer + 8..trailer + 16].try_into().unwrap()) as usize;
        let footer_start = trailer - footer_len;
        bad_crc[footer_start] ^= 0x55;
        let bad_crc = Bytes::from(bad_crc);
        let source = Arc::new(TrackingSource::new(bad_crc.clone()));
        assert!(VectorV5Reader::open(source, bad_crc.len() as u64)
            .await
            .is_err());

        let mut huge = body.to_vec();
        huge[trailer + 8..trailer + 16].copy_from_slice(&(MAX_FOOTER_BYTES + 1).to_le_bytes());
        let huge = Bytes::from(huge);
        let source = Arc::new(TrackingSource::new(huge.clone()));
        assert!(VectorV5Reader::open(source, huge.len() as u64)
            .await
            .is_err());

        // A page corruption is detected lazily when that range is consumed.
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source, body.len() as u64)
            .await
            .expect("open valid");
        let nav_offset = reader.footer.pages[0].nav.offset as usize;
        let mut page_corrupt = body.to_vec();
        page_corrupt[nav_offset] ^= 0x01;
        let page_corrupt = Bytes::from(page_corrupt);
        let source = Arc::new(TrackingSource::new(page_corrupt.clone()));
        let reader = VectorV5Reader::open(source, page_corrupt.len() as u64)
            .await
            .expect("metadata remains valid");
        assert!(reader
            .search(&[1.0, 1.0], 3, all_probe_options(reader.page_count()))
            .await
            .is_err());
    }

    fn clustered_members(count: u32, dim: usize, clusters: u32) -> Vec<([u8; 16], Vec<f32>)> {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed_0005);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|cluster| {
                let mut center = vec![0.0; dim];
                center[cluster as usize % dim] = 1.0;
                center[(cluster as usize * 7 + 3) % dim] = 0.45;
                normalize(&mut center);
                center
            })
            .collect();
        (0..count)
            .map(|ordinal| {
                let center = &centers[(ordinal % clusters) as usize];
                let mut vector: Vec<f32> = center
                    .iter()
                    .map(|value| *value + rng.gen_range(-0.025..0.025))
                    .collect();
                normalize(&mut vector);
                (id(ordinal), vector)
            })
            .collect()
    }

    #[tokio::test]
    async fn cold_query_reads_sublinear_bytes_and_reader_memory_is_metadata_only() {
        let members = clustered_members(4_096, 32, 32);
        let (body, _) = build_body_v5(
            &descriptor(VectorMetric::Cosine, 32),
            members.clone(),
            VectorV5BuildOptions {
                target_rows_per_page: 64,
                branch_factor: 4,
                compression_level: 1,
            },
        )
        .expect("build")
        .expect("body");
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source.clone(), body.len() as u64)
            .await
            .expect("open");
        let metadata_bytes = reader.resident_metadata_bytes();
        assert!(metadata_bytes < members.len() * 16, "{metadata_bytes}");
        let query = members[777].1.clone();
        let hits = reader
            .search(
                &query,
                10,
                VectorV5SearchOptions {
                    nprobe: 1,
                    max_nprobe: 1,
                    rerank_factor: 8,
                },
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 10);
        assert!(
            source.bytes_read() < body.len() as u64 / 3,
            "read {} of {} bytes in {} ranges",
            source.bytes_read(),
            body.len(),
            source.range_count()
        );
        assert!(source.range_count() <= 6);
    }

    #[tokio::test]
    async fn hierarchical_probe_has_reasonable_clustered_recall() {
        let members = clustered_members(2_048, 24, 16);
        let query = members[913].1.clone();
        let mut exact: Vec<_> = members
            .iter()
            .map(|(node_id, vector)| {
                (
                    *node_id,
                    metric_score(VectorMetric::Cosine, vector, &query).0 as f32,
                )
            })
            .collect();
        sort_hits(VectorMetric::Cosine, &mut exact);
        exact.truncate(10);
        let exact_ids: BTreeSet<_> = exact.iter().map(|hit| hit.0).collect();

        let (body, _) = build_body_v5(
            &descriptor(VectorMetric::Cosine, 24),
            members,
            VectorV5BuildOptions {
                target_rows_per_page: 64,
                branch_factor: 4,
                compression_level: 1,
            },
        )
        .expect("build")
        .expect("body");
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source, body.len() as u64)
            .await
            .expect("open");
        let approximate = reader
            .search(
                &query,
                10,
                VectorV5SearchOptions {
                    nprobe: 4,
                    max_nprobe: 4,
                    rerank_factor: 16,
                },
            )
            .await
            .expect("search");
        let overlap = approximate
            .iter()
            .filter(|hit| exact_ids.contains(&hit.0))
            .count();
        assert!(overlap >= 7, "recall@10={overlap}/10; {approximate:?}");
    }

    #[tokio::test]
    async fn all_page_probe_batches_navigation_and_streams_exact_pages() {
        let members = clustered_members(2_048, 24, 32);
        let query = members[777].1.clone();
        let (body, _) = build_body_v5(
            &descriptor(VectorMetric::Cosine, 24),
            members,
            VectorV5BuildOptions {
                target_rows_per_page: 16,
                branch_factor: 4,
                compression_level: 1,
            },
        )
        .expect("build")
        .expect("body");
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source.clone(), body.len() as u64)
            .await
            .expect("open");
        assert!(reader.page_count() > MAX_NAV_BATCH_PAGES);
        let hits = reader
            .search(
                &query,
                20,
                VectorV5SearchOptions {
                    nprobe: reader.page_count(),
                    max_nprobe: reader.page_count(),
                    rerank_factor: 8,
                },
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 20);
        assert!(
            source.max_batch_size() <= MAX_NAV_BATCH_PAGES,
            "held {} compressed navigation pages at once",
            source.max_batch_size()
        );
    }
}
