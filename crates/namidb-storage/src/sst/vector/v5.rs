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
mod rabitq;

#[doc(hidden)]
pub use rabitq::override_epsilon_for_benchmarks as override_rabitq_epsilon_for_benchmarks;
pub use rabitq::VectorV5CodePrefetch;
pub(crate) use rabitq::{
    base_lacks_section as base_lacks_rabitq, note_base_without_section as note_base_without_rabitq,
};

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::io::Write;
use std::mem::size_of;
use std::ops::Range;
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};
use std::sync::Arc;

use async_trait::async_trait;
use bincode::Options;
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use namidb_core::quantize::quantize_i8;
use namidb_core::Value;

use super::{
    decode_vector_filter_value, encode_vector_filter_value, metric_name, metric_score,
    VectorFilterPostings, VectorGraphBuildStats,
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
const MAX_NAV_BATCH_PAGES: usize = 64;
/// Upper bound on exact f32 pages fetched concurrently for one rerank wave.
const MAX_EXACT_BATCH_PAGES: usize = 64;
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

    /// Self-contained blocks every query reads with the same bounds
    /// (navigation pages). A caching source can key them by their exact
    /// range instead of rounding each up to a large page.
    async fn read_blocks(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
        self.read_ranges(ranges).await
    }

    /// Small scattered reads (frame headers and candidate rows of the exact
    /// re-rank). A caching source can serve them from small pages.
    async fn read_small(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
        self.read_ranges(ranges).await
    }

    /// [`Self::read_blocks`] of many neighbouring blocks (sorted by offset)
    /// read ahead of their use: a caching source can fetch them with a few
    /// large requests and keep each as the entry `read_blocks` later finds.
    async fn prime_blocks(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
        self.read_blocks(ranges).await
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
    /// Rows whose full-precision vector was read and scored exactly.
    pub reranked_rows: usize,
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

/// One row of a V5 page as it was built: NodeId, full-precision vector, and
/// its native-filter values in property order (absent properties omitted).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VectorV5StoredRow {
    pub(crate) id: [u8; 16],
    pub(crate) vector: Vec<f32>,
    pub(crate) filters: Vec<(String, Value)>,
}

/// Metadata-only reader. It never stores document IDs, quantized rows, or f32
/// corpus pages beyond one query's local futures/results.
pub struct VectorV5Reader {
    source: Arc<dyn VectorV5RangeSource>,
    file_len: u64,
    footer_offset: u64,
    footer: Footer,
    metric: VectorMetric,
    /// Per page: whether its exact block is a confirmed raw-block frame
    /// (`RAW_UNKNOWN` until a query reads its frame header).
    raw_pages: Box<[AtomicU8]>,
    /// The RaBitQ navigation section, when the base carries a valid one.
    rabitq: RabitqState,
}

/// Whether a base carries a RaBitQ section ([`rabitq`]).
#[derive(Debug)]
enum RabitqState {
    Present(rabitq::RabitqIndex),
    /// Written before the section existed (or with it disabled).
    Absent,
    /// A locator was found but the section did not validate; served by the
    /// int8 path.
    Invalid,
}

/// Size of a base's RaBitQ section, for capacity reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorV5RabitqStats {
    /// Rotated (padded) dimension: code bits per row.
    pub padded_dim: usize,
    /// Bytes of the code blocks every probe reads (codes and factors).
    pub code_block_bytes: u64,
    /// Bytes of the whole section (codes, ids, filters, directory, locator).
    pub section_bytes: u64,
    /// Decoded directory and rotation retained by the reader.
    pub resident_metadata_bytes: usize,
}

const RAW_UNKNOWN: u8 = 0;
const RAW_YES: u8 = 1;
const RAW_NO: u8 = 2;

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
            .field("rabitq", &matches!(self.rabitq, RabitqState::Present(_)))
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
    let mut page_nodes = vec![usize::MAX; tree.leaves.len()];
    for (index, node) in tree.nodes.iter().enumerate() {
        if let Some(page) = node.page {
            page_nodes[page as usize] = index;
        }
    }
    let mut section = rabitq::section_enabled()
        .then(|| {
            Ok::<_, Error>((
                rabitq::Encoder::new(desc.metric, dim)?,
                rabitq::SectionBuilder::default(),
                Vec::new(),
            ))
        })
        .transpose()?;
    for (page, leaf) in tree.leaves.iter().enumerate() {
        let nav = make_nav_page(&rows, leaf, dim, &reverse_filters)?;
        if let Some((encoder, builder, bytes)) = section.as_mut() {
            let node = tree
                .nodes
                .get(page_nodes[page])
                .ok_or_else(|| Error::invariant("vector v5 page has no leaf centroid"))?;
            let mut centroid = Vec::with_capacity(dim);
            rabitq::leaf_center(&node.codes, node.scale, 1.0, &mut centroid);
            let mut center_scale = rabitq::CenterScale::default();
            for &row_index in leaf {
                center_scale.push(desc.metric, &rows[row_index].vector, &centroid);
            }
            let mut page_encoder = rabitq::PageEncoder::new(
                encoder,
                &node.codes,
                node.scale,
                center_scale.finish(&centroid),
                leaf.len(),
            );
            for &row_index in leaf {
                page_encoder.push(encoder, &rows[row_index].id, &rows[row_index].vector)?;
            }
            let filters = (!filter_properties.is_empty())
                .then(|| rabitq::serialize_filters(&nav.filters))
                .transpose()?;
            builder.push_page(bytes, &page_encoder.finish(filters))?;
        }
        let nav_raw = serialize(&nav, "navigation page")?;
        #[cfg(test)]
        let nav_ref = if tests::COMPRESS_EXACT.with(|flag| flag.get()) {
            append_compressed_block(&mut body, &nav_raw, options.compression_level)?
        } else {
            append_raw_frame_block(&mut body, &nav_raw)?
        };
        #[cfg(not(test))]
        let nav_ref = append_raw_frame_block(&mut body, &nav_raw)?;

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
        #[cfg(test)]
        let exact_ref = if tests::COMPRESS_EXACT.with(|flag| flag.get()) {
            append_compressed_block(&mut body, &exact_raw, options.compression_level)?
        } else {
            append_raw_frame_block(&mut body, &exact_raw)?
        };
        #[cfg(not(test))]
        let exact_ref = append_raw_frame_block(&mut body, &exact_raw)?;
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
    // k-means splits may fan out wider than `branch_factor`; readers bound
    // a node's children by the footer's value.
    let fan_out = tree
        .nodes
        .iter()
        .map(|node| node.children.len())
        .max()
        .unwrap_or(0)
        .max(options.branch_factor);
    if let Some((encoder, builder, bytes)) = section {
        let section_offset = body.len() as u64;
        body.extend_from_slice(&bytes);
        builder.finish(
            &mut body,
            section_offset,
            &encoder,
            metric_name(desc.metric),
        )?;
    }
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
        branch_factor: u16::try_from(fan_out)
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
        entry_medoid: layout_tag(desc.metric, root),
    };
    Ok(Some((Bytes::from(body), stats)))
}

/// Bytes of footer `VectorV5Reader::open` reads with the trailer. A footer
/// is about 1/400 of its base (one int8 centroid and a few page references
/// per page of full-precision rows), so 1/128 covers it with room to spare,
/// bounded so a probe of a huge base stays a few MiB.
fn speculative_footer_bytes(file_len: u64) -> u64 {
    (file_len / 128).clamp(64 * 1024, 8 * 1024 * 1024)
}

impl VectorV5Reader {
    /// Open by reading only magic, trailer, and compact centroid footer.
    pub async fn open(source: Arc<dyn VectorV5RangeSource>, file_len: u64) -> Result<Self, Error> {
        let minimum = (MAGIC_V5.len() + TRAILER_LEN) as u64;
        if file_len < minimum {
            return Err(Error::invariant("vector v5 body too short"));
        }
        let trailer_start = file_len - TRAILER_LEN as u64;
        // One round trip for the header and a speculative tail that holds the
        // trailer and, for nearly every base, the whole footer; only a footer
        // larger than the guess costs a second read.
        let tail_start = trailer_start
            .saturating_sub(speculative_footer_bytes(file_len))
            .max(MAGIC_V5.len() as u64);
        let probes = source
            .read_ranges(&[0..MAGIC_V5.len() as u64, tail_start..file_len])
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
        let tail = &probes[1];
        require_len(
            tail,
            usize::try_from(file_len - tail_start)
                .map_err(|_| Error::invariant("vector v5 tail does not fit usize"))?,
            "tail",
        )?;
        let (footer_len, footer_crc) = decode_trailer(&tail[tail.len() - TRAILER_LEN..])?;
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
        let footer_bytes = if footer_offset >= tail_start {
            let start = usize::try_from(footer_offset - tail_start)
                .map_err(|_| Error::invariant("vector v5 footer does not fit usize"))?;
            tail.slice(start..tail.len() - TRAILER_LEN)
        } else {
            source.read_range(footer_offset..trailer_start).await?
        };
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
        let raw_pages = footer
            .pages
            .iter()
            .map(|page| {
                AtomicU8::new(if raw_frame_candidate(&page.exact) {
                    RAW_UNKNOWN
                } else {
                    RAW_NO
                })
            })
            .collect();
        let rabitq = Self::open_rabitq(
            source.as_ref(),
            &footer,
            metric,
            footer_offset,
            tail_start,
            tail,
        )
        .await?;
        Ok(Self {
            source,
            file_len,
            footer_offset,
            footer,
            metric,
            raw_pages,
            rabitq,
        })
    }

    /// Locate and validate the RaBitQ section. Read errors propagate; a
    /// section that does not validate leaves the base on the int8 path.
    async fn open_rabitq(
        source: &dyn VectorV5RangeSource,
        footer: &Footer,
        metric: VectorMetric,
        footer_offset: u64,
        tail_start: u64,
        tail: &Bytes,
    ) -> Result<RabitqState, Error> {
        let Some(locator_range) = rabitq::locator_range(footer, footer_offset) else {
            return Ok(RabitqState::Absent);
        };
        let read = |range: Range<u64>| async move {
            if range.start >= tail_start {
                let start = usize::try_from(range.start - tail_start)
                    .map_err(|_| Error::invariant("vector v5 tail offset overflows usize"))?;
                let end = usize::try_from(range.end - tail_start)
                    .map_err(|_| Error::invariant("vector v5 tail offset overflows usize"))?;
                Ok::<Bytes, Error>(tail.slice(start..end))
            } else {
                source.read_range(range).await
            }
        };
        let locator = read(locator_range).await?;
        let Some((directory_offset, directory_len, directory_crc)) =
            rabitq::parse_locator(&locator)
        else {
            return Ok(RabitqState::Absent);
        };
        let invalid = |reason: &str| {
            tracing::warn!(
                reason,
                "vector v5 RaBitQ section ignored; using int8 navigation"
            );
            Ok(RabitqState::Invalid)
        };
        let Some(directory_range) =
            rabitq::directory_range(footer, footer_offset, directory_offset, directory_len)
        else {
            return invalid("directory range is out of bounds");
        };
        let directory = read(directory_range).await?;
        if directory.len() as u64 != directory_len || crc32fast::hash(&directory) != directory_crc {
            return invalid("directory checksum mismatch");
        }
        match rabitq::RabitqIndex::open(footer, metric, &directory, directory_offset) {
            Ok(index) => Ok(RabitqState::Present(index)),
            Err(error) => invalid(&error.to_string()),
        }
    }

    /// Whether queries use the RaBitQ section.
    pub fn has_rabitq(&self) -> bool {
        matches!(self.rabitq, RabitqState::Present(_))
    }

    /// Whether the base was written without a RaBitQ section (as releases
    /// before it wrote every base); a rebuild adds one.
    pub fn rabitq_absent(&self) -> bool {
        matches!(self.rabitq, RabitqState::Absent)
    }

    pub fn rabitq_stats(&self) -> Option<VectorV5RabitqStats> {
        let RabitqState::Present(index) = &self.rabitq else {
            return None;
        };
        Some(VectorV5RabitqStats {
            padded_dim: index.padded_dim(),
            code_block_bytes: index.code_block_bytes(),
            section_bytes: index.section_bytes(self.footer_offset),
            resident_metadata_bytes: index.resident_bytes(),
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
        let rabitq = match &self.rabitq {
            RabitqState::Present(index) => index.resident_bytes(),
            _ => 0,
        };
        centroids
            .saturating_add(rabitq)
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
    /// Byte length of the object this reader was opened over.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Native-filter properties this body advertises as complete.
    pub(crate) fn filter_properties(&self) -> &[String] {
        &self.footer.filter_properties
    }

    /// Decode every row of page `page` (in page order, not NodeId order):
    /// the f32 vectors from its exact block and the filter values from its
    /// navigation bitmaps. One page is resident at a time.
    pub(crate) async fn read_page_rows(
        &self,
        page: usize,
    ) -> Result<Vec<VectorV5StoredRow>, Error> {
        let page_ref = self
            .footer
            .pages
            .get(page)
            .ok_or_else(|| Error::invariant("vector v5 page ordinal is out of range"))?;
        let wires = self
            .source
            .read_ranges(&[page_ref.nav.range()?, page_ref.exact.range()?])
            .await?;
        let [nav_wire, exact_wire] = wires.as_slice() else {
            return Err(Error::invariant(
                "vector v5 range source returned the wrong page block count",
            ));
        };
        let nav = decode_nav_page(nav_wire, &page_ref.nav, page_ref.row_count, self.footer.dim)?;
        let raw = decode_block(exact_wire, &page_ref.exact, "exact page")?;
        let dim = self.footer.dim as usize;
        let row_bytes = dim
            .checked_mul(4)
            .ok_or_else(|| Error::invariant("vector v5 row byte size overflows"))?;
        if raw.len()
            != row_bytes
                .checked_mul(nav.ids.len())
                .ok_or_else(|| Error::invariant("vector v5 exact page size overflows"))?
        {
            return Err(Error::invariant(
                "vector v5 exact page length disagrees with its navigation page",
            ));
        }
        let mut rows = Vec::with_capacity(nav.ids.len());
        for (id, exact) in nav.ids.iter().zip(raw.chunks_exact(row_bytes.max(1))) {
            let mut vector = Vec::with_capacity(dim);
            for bytes in exact.chunks_exact(4) {
                let component = f32::from_le_bytes(
                    bytes
                        .try_into()
                        .map_err(|_| Error::invariant("vector v5 invalid f32 bytes"))?,
                );
                if !component.is_finite() {
                    return Err(Error::invariant(
                        "vector v5 exact page contains a non-finite component",
                    ));
                }
                vector.push(component);
            }
            rows.push(VectorV5StoredRow {
                id: *id,
                vector,
                filters: Vec::new(),
            });
        }
        for (property, values) in nav.filters {
            if !self.supports_filter_property(&property) {
                return Err(Error::invariant(
                    "vector v5 navigation page carries an unadvertised filter",
                ));
            }
            for (key, bitmap) in values {
                let value = decode_vector_filter_value(&key)
                    .ok_or_else(|| Error::invariant("vector v5 filter key is not decodable"))?;
                for (word_index, mut word) in bitmap.into_iter().enumerate() {
                    while word != 0 {
                        let row = word_index * 64 + word.trailing_zeros() as usize;
                        word &= word - 1;
                        let filters = &mut rows
                            .get_mut(row)
                            .ok_or_else(|| {
                                Error::invariant("vector v5 filter bitmap leaves its page")
                            })?
                            .filters;
                        if filters.last().is_some_and(|(last, _)| *last == property) {
                            return Err(Error::invariant(
                                "vector v5 row holds two values of one filter property",
                            ));
                        }
                        filters.push((property.clone(), value.clone()));
                    }
                }
            }
        }
        Ok(rows)
    }

    pub async fn search_filter_groups(
        &self,
        query: &[f32],
        k: usize,
        options: VectorV5SearchOptions,
        groups: &[(String, Vec<Value>)],
    ) -> Result<VectorV5SearchResult, Error> {
        if let Some(empty) = self.validate_query(query, k)? {
            return Ok(empty);
        }
        // A zero cosine query scores every row 0; the int8 path returns that.
        let zero_cosine =
            self.metric == VectorMetric::Cosine && query.iter().all(|value| *value == 0.0);
        match &self.rabitq {
            RabitqState::Present(index) if !zero_cosine => {
                self.search_rabitq(index, query, k, options, groups).await
            }
            _ => self.search_int8(query, k, options, groups).await,
        }
    }

    /// Search through the int8 navigation pages even when the base carries a
    /// RaBitQ section (the path every base without one takes).
    pub async fn search_filter_groups_int8(
        &self,
        query: &[f32],
        k: usize,
        options: VectorV5SearchOptions,
        groups: &[(String, Vec<Value>)],
    ) -> Result<VectorV5SearchResult, Error> {
        if let Some(empty) = self.validate_query(query, k)? {
            return Ok(empty);
        }
        self.search_int8(query, k, options, groups).await
    }

    /// Reject malformed queries; `Some` is the empty result of a query that
    /// can match nothing.
    fn validate_query(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Option<VectorV5SearchResult>, Error> {
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
            return Ok(Some(VectorV5SearchResult {
                hits: Vec::new(),
                applied_filter_groups: 0,
                probed_pages: 0,
                eligible_rows_seen: 0,
                reranked_rows: 0,
            }));
        }
        Ok(None)
    }

    async fn search_int8(
        &self,
        query: &[f32],
        k: usize,
        options: VectorV5SearchOptions,
        groups: &[(String, Vec<Value>)],
    ) -> Result<VectorV5SearchResult, Error> {
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
        let workspace_plan = vector_workspace_plan(
            &self.footer,
            rerank_k,
            final_k,
            groups,
            rabitq::search_planning_budget(),
        )?;
        let _workspace = shared_search_workspace()
            .reserve("vector v5 search", workspace_plan.required_bytes)
            .await?;
        let supported_groups = prepare_groups(&self.footer.filter_properties, groups);
        let applied_filter_groups = supported_groups.len();
        let leaf_order = {
            namidb_core::profile_scope!("v5.rank_leaves");
            self.rank_leaf_pages(query)?
        };

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
                let blocks = {
                    namidb_core::profile_scope!("v5.nav.read");
                    self.source.read_blocks(&ranges).await?
                };
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
                    let nav = {
                        namidb_core::profile_scope!("v5.nav.decode");
                        decode_nav_page(&wire, &page_ref.nav, page_ref.row_count, self.footer.dim)?
                    };
                    namidb_core::profile_scope!("v5.nav.score");
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
        let reranked_rows = approximate.len();
        // Group candidates in-place so one exact f32 page can be read,
        // scored and discarded before the next page arrives.
        approximate.sort_by(|left, right| {
            left.page
                .cmp(&right.page)
                .then_with(|| left.row.cmp(&right.row))
                .then_with(|| left.visit_order.cmp(&right.visit_order))
        });
        let mut exact_hits = BoundedExactHits::new(final_k);
        // `(page, first candidate, end candidate)` per distinct rerank page.
        let mut page_groups = Vec::new();
        let mut candidate_start = 0usize;
        while candidate_start < approximate.len() {
            let page = approximate[candidate_start].page;
            let mut candidate_end = candidate_start + 1;
            while candidate_end < approximate.len() && approximate[candidate_end].page == page {
                candidate_end += 1;
            }
            page_groups.push((page, candidate_start, candidate_end));
            candidate_start = candidate_end;
        }
        let row_bytes = (self.footer.dim as usize)
            .checked_mul(4)
            .ok_or_else(|| Error::invariant("vector v5 row byte size overflows"))?;
        let query_norms = QueryNorms::of(query);
        for wave in page_groups.chunks(workspace_plan.exact_batch_pages.max(1)) {
            // Pages stored as raw-block frames serve just the candidate rows;
            // a page whose layout or rows do not check out is read whole.
            let mut whole = Vec::new();
            let mut spans = Vec::new();
            let mut plans = Vec::new();
            for &(page, candidate_start, candidate_end) in wave {
                let exact = &self.footer.pages[page as usize].exact;
                let state = self.raw_pages[page as usize].load(AtomicOrdering::Relaxed);
                if state == RAW_NO {
                    whole.push((page, candidate_start, candidate_end));
                    continue;
                }
                let header = (state == RAW_UNKNOWN).then(|| {
                    spans.push(raw_frame_header_range(exact));
                    spans.len() - 1
                });
                let rows = plan_raw_rows(
                    exact,
                    approximate[candidate_start..candidate_end]
                        .iter()
                        .map(|candidate| candidate.row),
                    row_bytes as u64,
                    &mut spans,
                );
                plans.push((page, candidate_start, candidate_end, header, rows));
            }
            if !spans.is_empty() {
                let parts = {
                    namidb_core::profile_scope!("v5.exact.read_rows");
                    self.source.read_small(&spans).await?
                };
                if parts.len() != spans.len() {
                    return Err(Error::invariant(
                        "vector v5 range source returned wrong exact row count",
                    ));
                }
                namidb_core::profile_scope!("v5.exact.score_rows");
                for (page, candidate_start, candidate_end, header, rows) in plans {
                    let exact = &self.footer.pages[page as usize].exact;
                    if let Some(header) = header {
                        let raw = parts[header].as_ref() == raw_frame_header(exact.raw_len);
                        self.raw_pages[page as usize]
                            .store(if raw { RAW_YES } else { RAW_NO }, AtomicOrdering::Relaxed);
                        if !raw {
                            whole.push((page, candidate_start, candidate_end));
                            continue;
                        }
                    }
                    let scored = score_raw_rows(
                        self.metric,
                        query,
                        &query_norms,
                        &parts,
                        &spans,
                        &approximate[candidate_start..candidate_end],
                        &rows,
                        row_bytes,
                    );
                    match scored {
                        Some(scored) => {
                            for (id, score) in scored {
                                exact_hits.insert(self.metric, id, score);
                            }
                        }
                        None => whole.push((page, candidate_start, candidate_end)),
                    }
                }
            }
            if whole.is_empty() {
                continue;
            }
            let ranges = whole
                .iter()
                .map(|(page, _, _)| self.footer.pages[*page as usize].exact.range())
                .collect::<Result<Vec<_>, _>>()?;
            let wires = {
                namidb_core::profile_scope!("v5.exact.read");
                self.source.read_ranges(&ranges).await?
            };
            if wires.len() != ranges.len() {
                return Err(Error::invariant(
                    "vector v5 range source returned wrong exact block count",
                ));
            }
            for ((&(page, candidate_start, candidate_end), wire), range) in
                whole.iter().zip(wires).zip(ranges)
            {
                require_len(
                    &wire,
                    usize::try_from(range.end - range.start)
                        .map_err(|_| Error::invariant("vector v5 range length overflows usize"))?,
                    "exact block",
                )?;
                let page_ref = &self.footer.pages[page as usize];
                namidb_core::profile_scope!("v5.exact.decode_score");
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
                    let vector = f32_row(row);
                    let score = metric_score(self.metric, &vector, query).0 as f32;
                    exact_hits.insert(self.metric, candidate.id, score);
                }
            }
        }
        let hits = exact_hits.into_sorted_hits(self.metric);
        Ok(VectorV5SearchResult {
            hits,
            applied_filter_groups,
            probed_pages: fetched,
            eligible_rows_seen,
            reranked_rows,
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
                reranked_rows: 0,
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
            reranked_rows: eligible_rows_seen,
        })
    }

    fn rank_leaf_pages(&self, query: &[f32]) -> Result<Vec<u32>, Error> {
        let query_norm = lane_norm(query);
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
                            approximate_score(
                                self.metric,
                                query,
                                query_norm,
                                &node.codes,
                                node.scale,
                            ) as f64,
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
    /// Quantization step of the row's int8 code.
    scale: f32,
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
    /// Exact f32 pages fetched (concurrently) and scored per rerank wave.
    exact_batch_pages: usize,
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
    budget: usize,
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
    // Batches beyond one page are read-ahead: they are sized against
    // `budget`, this search's share of the workspace, not the whole pool.
    // One page of each (the fallback below) is admitted whenever it fits
    // the pool.

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
        if required <= budget {
            // Rerank pages are independent range reads: fetch as many at once
            // as the workspace holds (one round trip per wave instead of one
            // per page), never fewer than the single page already admitted.
            let mut exact_batch_pages = 1;
            for wave in (2..=MAX_EXACT_BATCH_PAGES).rev() {
                let exact_peak = checked_mul(
                    exact_page_peak,
                    wave,
                    "vector v5 exact rerank wave workspace",
                )?;
                let wave_required = checked_add(
                    base,
                    nav_peak.max(exact_peak),
                    "vector v5 total query workspace",
                )?;
                if wave_required <= budget {
                    exact_batch_pages = wave;
                    break;
                }
            }
            let required = checked_add(
                base,
                nav_peak.max(checked_mul(
                    exact_page_peak,
                    exact_batch_pages,
                    "vector v5 exact rerank wave workspace",
                )?),
                "vector v5 total query workspace",
            )?;
            return Ok(VectorWorkspacePlan {
                nav_batch_pages: batch_pages,
                exact_batch_pages,
                required_bytes: required,
            });
        }
    }
    Ok(VectorWorkspacePlan {
        nav_batch_pages: 1,
        exact_batch_pages: 1,
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

    if metric != VectorMetric::Dot && kmeans_enabled() {
        if let Some(groups) = kmeans_split_indices(rows, &indices, target, metric) {
            let mut children = Vec::with_capacity(groups.len());
            for group in groups {
                children.push(partition_rows(rows, group, target, branch, metric, tree)?);
            }
            tree.nodes[node_index as usize].children = children;
            return Ok(node_index);
        }
    }

    let axis = highest_variance_axis(rows, &indices)?;
    let group_count = branch.min(indices.len().div_ceil(target)).max(2);
    let mut children = Vec::with_capacity(group_count);
    // Split along the principal direction of a bounded sample (falling back
    // to the highest-variance axis). In high dimensions a single coordinate
    // separates natural clusters poorly, so neighbours ended up spread over
    // many leaves and recall needed far more probes.
    let sample = principal_sample_indices(&indices);
    let sample_rows = sample
        .iter()
        .map(|index| rows[*index].vector.as_slice())
        .collect::<Vec<_>>();
    // Dot-product scores also grow with a row's norm, which the centred
    // principal direction ignores: dot-metric bases keep the axis split.
    let direction = (metric != VectorMetric::Dot)
        .then(|| principal_direction(&sample_rows, axis))
        .flatten();
    let keyed = match direction {
        Some(direction) => indices
            .into_iter()
            .map(|index| (dot_f32(&rows[index].vector, &direction), index))
            .collect::<Vec<_>>(),
        None => indices
            .into_iter()
            .map(|index| (rows[index].vector[axis], index))
            .collect::<Vec<_>>(),
    };
    // Balanced recursive multi-selection is O(N log B) for fan-out B, unlike
    // assigning every row to every centroid (O(N*B)) or sorting the whole set
    // at each tree level (O(N log N log C)). Across the tree this gives the
    // intended O(N log C * dim) construction bound.
    for chunk in balanced_key_groups(rows, keyed, group_count) {
        children.push(partition_rows(rows, chunk, target, branch, metric, tree)?);
    }
    tree.nodes[node_index as usize].children = children;
    Ok(node_index)
}

/// Nearest-centroid groups of `indices` from a k-means split trained on an
/// even stride of them, or `None` when the split makes no progress (all
/// rows equal, or one group would keep every row).
fn kmeans_split_indices(
    rows: &[BuildRow],
    indices: &[usize],
    target: usize,
    metric: VectorMetric,
) -> Option<Vec<Vec<usize>>> {
    let groups = kmeans_group_count(indices.len(), target);
    let sample_rows = kmeans_sample_rows(indices.len(), groups);
    let step = indices.len().div_ceil(sample_rows).max(1);
    let sample: Vec<&[f32]> = indices
        .iter()
        .step_by(step)
        .map(|index| rows[*index].vector.as_slice())
        .collect();
    let seed = indices.len() as u64 ^ u64::from_le_bytes(rows[indices[0]].id[..8].try_into().ok()?);
    let centroids = kmeans_centroids(&sample, groups, metric, seed)?;
    let norms: Vec<f32> = centroids.iter().map(|c| dot_lanes(c, c)).collect();
    let mut out = vec![Vec::new(); centroids.len()];
    for index in indices {
        out[nearest_centroid(&rows[*index].vector, &centroids, &norms, metric)].push(*index);
    }
    out.retain(|group| !group.is_empty());
    (out.len() >= 2).then_some(out)
}

fn balanced_key_groups(
    rows: &[BuildRow],
    mut keyed: Vec<(f32, usize)>,
    groups: usize,
) -> Vec<Vec<usize>> {
    if groups <= 1 || keyed.len() <= 1 {
        return vec![keyed.into_iter().map(|(_, index)| index).collect()];
    }
    let left_groups = groups / 2;
    let right_groups = groups - left_groups;
    // Proportional split keeps every eventual bucket within one row of N/B.
    let split = keyed.len() * left_groups / groups;
    keyed.select_nth_unstable_by(split, |left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| rows[left.1].id.cmp(&rows[right.1].id))
    });
    let right = keyed.split_off(split);
    let mut out = balanced_key_groups(rows, keyed, left_groups);
    out.extend(balanced_key_groups(rows, right, right_groups));
    out
}

/// Rows sampled to estimate a split direction: an even stride over the set.
const PRINCIPAL_SAMPLE_ROWS: usize = 256;
const PRINCIPAL_ITERATIONS: usize = 8;

fn principal_sample_indices(indices: &[usize]) -> Vec<usize> {
    let step = indices.len().div_ceil(PRINCIPAL_SAMPLE_ROWS).max(1);
    indices.iter().step_by(step).copied().collect()
}

pub(super) fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

/// Dot product with eight independent lanes, which the compiler vectorises;
/// the k-means splits below spend nearly all their time here.
pub(super) fn dot_lanes(left: &[f32], right: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut left_chunks = left.chunks_exact(8);
    let mut right_chunks = right.chunks_exact(8);
    for (l, r) in (&mut left_chunks).zip(&mut right_chunks) {
        for lane in 0..8 {
            lanes[lane] += l[lane] * r[lane];
        }
    }
    let tail: f32 = left_chunks
        .remainder()
        .iter()
        .zip(right_chunks.remainder())
        .map(|(l, r)| l * r)
        .sum();
    lanes.iter().sum::<f32>() + tail
}

/// Most children one k-means split makes. A node with more rows than
/// `KMEANS_MAX_GROUPS * target` splits again below.
pub(super) const KMEANS_MAX_GROUPS: usize = 64;

/// Layout tag a V5 base built with k-means splits carries in its
/// descriptor's `entry_medoid` (a V5 body's root is always node 0 and no
/// reader takes it from the descriptor). Older releases ignore it; the
/// compaction planner rebuilds a cosine or Euclidean base without it once,
/// so bases written before k-means splits get the new layout.
pub(crate) const VECTOR_V5_KMEANS_LAYOUT: u32 = 0x4B4D_0001;

/// The descriptor tag for a base built now: k-means bases carry
/// [`VECTOR_V5_KMEANS_LAYOUT`], dot-product bases (direction splits) their
/// root.
pub(crate) fn layout_tag(metric: VectorMetric, root: u32) -> u32 {
    if metric != VectorMetric::Dot && kmeans_enabled() {
        VECTOR_V5_KMEANS_LAYOUT
    } else {
        root
    }
}

/// Test-only switch back to the direction split, for before/after sweeps.
#[cfg(test)]
pub(super) static KMEANS_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(super) fn kmeans_enabled() -> bool {
    #[cfg(test)]
    {
        !KMEANS_OFF.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        true
    }
}
/// Training rows per child and in all for one k-means split.
pub(super) const KMEANS_SAMPLE_PER_GROUP: usize = 48;
pub(super) const KMEANS_MAX_SAMPLE: usize = 8192;
const KMEANS_ITERATIONS: usize = 12;

/// Children of a k-means split of `rows` rows into pages of `target`.
pub(super) fn kmeans_group_count(rows: usize, target: usize) -> usize {
    rows.div_ceil(target.max(1)).clamp(2, KMEANS_MAX_GROUPS)
}

/// Training rows for a split into `groups` children of a node of `rows`.
pub(super) fn kmeans_sample_rows(rows: usize, groups: usize) -> usize {
    groups
        .saturating_mul(KMEANS_SAMPLE_PER_GROUP)
        .clamp(256, KMEANS_MAX_SAMPLE)
        .min(rows)
}

/// Split centroids trained by k-means (k-means++ seeding, Lloyd iterations)
/// on `sample`. Cosine trains on unit rows and keeps unit centroids
/// (spherical k-means); Euclidean uses plain means. Deterministic for a given
/// sample and `seed`. `None` when the sample cannot support `groups`
/// distinct centroids. Memory is the sample plus `groups * dim` floats.
///
/// Leaves made by nearest-centroid assignment hold a query's true neighbours
/// in far fewer leaves than equal-count slabs along one direction, which is
/// what the leaf ranking at query time relies on.
pub(super) fn kmeans_centroids(
    sample: &[&[f32]],
    groups: usize,
    metric: VectorMetric,
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let dim = sample.first()?.len();
    if groups < 2 || sample.len() < groups || dim == 0 {
        return None;
    }
    let cosine = metric == VectorMetric::Cosine;
    let points: Vec<Vec<f32>> = sample
        .iter()
        .map(|row| {
            let mut row = row.to_vec();
            if cosine {
                let norm = dot_lanes(&row, &row).sqrt();
                if norm > 0.0 {
                    row.iter_mut().for_each(|value| *value /= norm);
                }
            }
            row
        })
        .collect();
    let norms: Vec<f32> = points.iter().map(|row| dot_lanes(row, row)).collect();
    // Distance used for seeding and empty-cluster repair: 1 - cos for unit
    // rows, squared L2 otherwise.
    let distance = |point: usize, centre: &[f32], centre_norm: f32| -> f32 {
        let dot = dot_lanes(&points[point], centre);
        if cosine {
            (1.0 - dot).max(0.0)
        } else {
            (norms[point] + centre_norm - 2.0 * dot).max(0.0)
        }
    };
    // splitmix64: deterministic and dependency-free.
    let mut state = seed;
    let mut next_unit = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(groups);
    let first = ((next_unit() * points.len() as f64) as usize).min(points.len() - 1);
    centroids.push(points[first].clone());
    let mut nearest: Vec<f32> = (0..points.len())
        .map(|point| distance(point, &centroids[0], norms[first]))
        .collect();
    while centroids.len() < groups {
        let total: f64 = nearest.iter().map(|d| f64::from(*d)).sum();
        if total.is_nan() || total <= 0.0 {
            return None;
        }
        let mut target = next_unit() * total;
        let mut chosen = points.len() - 1;
        for (point, d) in nearest.iter().enumerate() {
            target -= f64::from(*d);
            if target <= 0.0 {
                chosen = point;
                break;
            }
        }
        let centre = points[chosen].clone();
        let centre_norm = norms[chosen];
        for (point, d) in nearest.iter_mut().enumerate() {
            *d = d.min(distance(point, &centre, centre_norm));
        }
        centroids.push(centre);
    }
    let mut assignment = vec![0usize; points.len()];
    for _ in 0..KMEANS_ITERATIONS {
        let centre_norms: Vec<f32> = centroids.iter().map(|c| dot_lanes(c, c)).collect();
        let mut changed = false;
        let mut worst: Vec<(f32, usize)> = Vec::with_capacity(points.len());
        for (point, slot) in assignment.iter_mut().enumerate() {
            let best = nearest_centroid(&points[point], &centroids, &centre_norms, metric);
            if *slot != best {
                *slot = best;
                changed = true;
            }
            worst.push((distance(point, &centroids[best], centre_norms[best]), point));
        }
        let mut sums = vec![vec![0.0f64; dim]; groups];
        let mut counts = vec![0usize; groups];
        for (point, group) in assignment.iter().enumerate() {
            counts[*group] += 1;
            for (sum, value) in sums[*group].iter_mut().zip(&points[point]) {
                *sum += f64::from(*value);
            }
        }
        // An empty cluster takes the point farthest from its centroid.
        worst.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut spare = worst.into_iter().map(|(_, point)| point);
        for group in 0..groups {
            if counts[group] == 0 {
                let point = spare.next()?;
                centroids[group] = points[point].clone();
                changed = true;
                continue;
            }
            let n = counts[group] as f64;
            let mut centre: Vec<f32> = sums[group].iter().map(|sum| (sum / n) as f32).collect();
            if cosine {
                let norm = dot_lanes(&centre, &centre).sqrt();
                if norm > 0.0 {
                    centre.iter_mut().for_each(|value| *value /= norm);
                }
            }
            centroids[group] = centre;
        }
        if !changed {
            break;
        }
    }
    Some(centroids)
}

/// Index of the centroid nearest to `vector` under `metric` (cosine: the
/// centroids are unit, so the largest dot product; Euclidean: the smallest
/// `|c|^2 - 2 v.c`). Ties go to the lower index.
pub(super) fn nearest_centroid(
    vector: &[f32],
    centroids: &[Vec<f32>],
    centroid_norms: &[f32],
    metric: VectorMetric,
) -> usize {
    let mut best = 0usize;
    let mut best_key = f32::NEG_INFINITY;
    for (index, centre) in centroids.iter().enumerate() {
        let dot = dot_lanes(vector, centre);
        let key = if metric == VectorMetric::Cosine {
            dot
        } else {
            2.0 * dot - centroid_norms[index]
        };
        if key > best_key {
            best_key = key;
            best = index;
        }
    }
    best
}

/// Top principal direction of `sample` by power iteration, started from
/// `axis` so it is deterministic and never worse-conditioned than the axis.
/// `None` when the sample has no spread. Memory is `O(dim)` beyond the
/// sample; cost is `O(sample * dim * iterations)`.
pub(super) fn principal_direction(sample: &[&[f32]], axis: usize) -> Option<Vec<f32>> {
    let dim = sample.first()?.len();
    if sample.len() < 2 || axis >= dim {
        return None;
    }
    let mut mean = vec![0.0f64; dim];
    for row in sample {
        for (sum, value) in mean.iter_mut().zip(row.iter()) {
            *sum += f64::from(*value);
        }
    }
    let n = sample.len() as f64;
    let mean = mean.into_iter().map(|sum| sum / n).collect::<Vec<_>>();
    let mut direction = vec![0.0f64; dim];
    direction[axis] = 1.0;
    for _ in 0..PRINCIPAL_ITERATIONS {
        let mut next = vec![0.0f64; dim];
        for row in sample {
            let projection = row
                .iter()
                .zip(&mean)
                .zip(&direction)
                .map(|((value, mean), dir)| (f64::from(*value) - mean) * dir)
                .sum::<f64>();
            for ((acc, value), mean) in next.iter_mut().zip(row.iter()).zip(&mean) {
                *acc += (f64::from(*value) - mean) * projection;
            }
        }
        let norm = next.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return None;
        }
        direction = next.into_iter().map(|value| value / norm).collect();
    }
    Some(direction.into_iter().map(|value| value as f32).collect())
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
    let query_norm = lane_norm(query);
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
        let score = approximate_score(metric, query, query_norm, codes, scale);
        out.insert(
            metric,
            ApproxCandidate {
                page,
                row: u32::try_from(row)
                    .map_err(|_| Error::invariant("vector v5 row exceeds u32"))?,
                id: nav.ids[row],
                score,
                scale,
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

/// `‖query‖`, summed in the same fixed lanes as [`approximate_score`].
fn lane_norm(query: &[f32]) -> f32 {
    const LANES: usize = 8;
    let split = query.len() - query.len() % LANES;
    let mut lanes = [0.0f32; LANES];
    for q in query[..split].chunks_exact(LANES) {
        for lane in 0..LANES {
            lanes[lane] += q[lane] * q[lane];
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for q in &query[split..] {
        sum += *q * *q;
    }
    sum.sqrt()
}

/// Score of `query` (whose [`lane_norm`] is `query_norm`) against the int8
/// code `codes * scale`. The sums run in eight fixed lanes so the compiler
/// can vectorise them while the result stays deterministic on every platform
/// (no reassociation is left to the compiler).
fn approximate_score(
    metric: VectorMetric,
    query: &[f32],
    query_norm: f32,
    codes: &[i8],
    scale: f32,
) -> f32 {
    const LANES: usize = 8;
    let len = query.len().min(codes.len());
    let (query, codes) = (&query[..len], &codes[..len]);
    let split = len - len % LANES;
    match metric {
        VectorMetric::Dot => {
            let mut dot = [0.0f32; LANES];
            for (q, c) in query[..split]
                .chunks_exact(LANES)
                .zip(codes[..split].chunks_exact(LANES))
            {
                for lane in 0..LANES {
                    dot[lane] += q[lane] * c[lane] as f32;
                }
            }
            let mut dot_sum = dot.iter().sum::<f32>();
            for (q, c) in query[split..].iter().zip(&codes[split..]) {
                dot_sum += *q * *c as f32;
            }
            dot_sum * scale
        }
        VectorMetric::Cosine => {
            let mut dot = [0.0f32; LANES];
            let mut code_norm = [0i32; LANES];
            for (q, c) in query[..split]
                .chunks_exact(LANES)
                .zip(codes[..split].chunks_exact(LANES))
            {
                for lane in 0..LANES {
                    dot[lane] += q[lane] * c[lane] as f32;
                    code_norm[lane] += c[lane] as i32 * c[lane] as i32;
                }
            }
            let mut dot_sum = dot.iter().sum::<f32>();
            let mut code_sum = code_norm.iter().map(|v| *v as i64).sum::<i64>();
            for (q, c) in query[split..].iter().zip(&codes[split..]) {
                dot_sum += *q * *c as f32;
                code_sum += *c as i64 * *c as i64;
            }
            // cos(q, s·c) = q·c / (|q| |c|) for any positive scale s.
            if query_norm == 0.0 || code_sum == 0 || scale == 0.0 {
                0.0
            } else {
                dot_sum / (query_norm * (code_sum as f32).sqrt())
            }
        }
        VectorMetric::Euclidean => {
            let mut sum = [0.0f32; LANES];
            for (q, c) in query[..split]
                .chunks_exact(LANES)
                .zip(codes[..split].chunks_exact(LANES))
            {
                for lane in 0..LANES {
                    let delta = q[lane] - c[lane] as f32 * scale;
                    sum[lane] += delta * delta;
                }
            }
            let mut total = sum.iter().sum::<f32>();
            for (q, c) in query[split..].iter().zip(&codes[split..]) {
                let delta = *q - *c as f32 * scale;
                total += delta * delta;
            }
            total.sqrt()
        }
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

/// How pages were written before 3.5 (tests build such bases to cover them).
#[cfg(test)]
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
    let wire = zstd::stream::encode_all(std::io::Cursor::new(raw), compression_level)
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
    let nav = decode_nav_raw(&raw)?;
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

/// Decode a navigation page's bincode body (fixint, little endian, no
/// trailing bytes) without a per-element serde visit: the code and scale
/// arrays, which are nearly all of the page, are copied in bulk. Accepts and
/// rejects exactly what [`deserialize`] does for [`NavPage`].
fn decode_nav_raw(raw: &[u8]) -> Result<NavPage, Error> {
    struct Cursor<'a> {
        bytes: &'a [u8],
    }
    fn short() -> Error {
        Error::invariant("vector v5 navigation page decode failed: truncated")
    }
    impl<'a> Cursor<'a> {
        fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
            if self.bytes.len() < n {
                return Err(short());
            }
            let (head, tail) = self.bytes.split_at(n);
            self.bytes = tail;
            Ok(head)
        }
        fn len(&mut self, element_bytes: usize) -> Result<usize, Error> {
            let len = u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes"));
            // A length whose elements cannot fit in what is left is truncated
            // input, checked before any allocation.
            let len = usize::try_from(len).map_err(|_| short())?;
            if len
                .checked_mul(element_bytes)
                .is_none_or(|n| n > self.bytes.len())
            {
                return Err(short());
            }
            Ok(len)
        }
        fn string(&mut self) -> Result<String, Error> {
            let len = self.len(1)?;
            String::from_utf8(self.take(len)?.to_vec()).map_err(|_| {
                Error::invariant("vector v5 navigation page decode failed: invalid utf-8")
            })
        }
    }
    let mut cursor = Cursor { bytes: raw };
    let rows = cursor.len(16)?;
    let ids = cursor
        .take(rows * 16)?
        .chunks_exact(16)
        .map(|id| id.try_into().expect("16 bytes"))
        .collect();
    let code_len = cursor.len(1)?;
    let codes = cursor
        .take(code_len)?
        .iter()
        .map(|byte| *byte as i8)
        .collect();
    let scale_len = cursor.len(4)?;
    let scales = cursor
        .take(scale_len * 4)?
        .chunks_exact(4)
        .map(|scale| f32::from_le_bytes(scale.try_into().expect("4 bytes")))
        .collect();
    let mut filters = BTreeMap::new();
    let properties = cursor.len(8 + 8)?;
    for _ in 0..properties {
        let property = cursor.string()?;
        let mut values = BTreeMap::new();
        let value_count = cursor.len(8 + 8)?;
        for _ in 0..value_count {
            let value = cursor.string()?;
            let words = cursor.len(8)?;
            let bitmap = cursor
                .take(words * 8)?
                .chunks_exact(8)
                .map(|word| u64::from_le_bytes(word.try_into().expect("8 bytes")))
                .collect();
            values.insert(value, bitmap);
        }
        filters.insert(property, values);
    }
    if !cursor.bytes.is_empty() {
        return Err(Error::invariant(
            "vector v5 navigation page decode failed: trailing bytes",
        ));
    }
    Ok(NavPage {
        ids,
        codes,
        scales,
        filters,
    })
}

/// Raw-content bytes per block of a stored exact page (the zstd maximum).
const RAW_FRAME_BLOCK_BYTES: u64 = 128 * 1024;
/// Magic, a single-segment descriptor with a 4-byte content size, the size.
const RAW_FRAME_HEADER_BYTES: u64 = 9;
const RAW_FRAME_BLOCK_HEADER_BYTES: u64 = 3;

/// Exact f32 pages are written as zstd frames made only of raw (stored)
/// blocks. Any zstd decoder, including every earlier V5 reader, decodes them
/// like any other page, while a reader that recognises the layout can range
/// read a single row instead of the whole page: re-ranking needs a handful of
/// rows per page, and f32 embeddings barely compress anyway.
fn raw_frame_header(raw_len: u32) -> [u8; RAW_FRAME_HEADER_BYTES as usize] {
    let mut header = [0u8; RAW_FRAME_HEADER_BYTES as usize];
    header[..4].copy_from_slice(&0xFD2F_B528u32.to_le_bytes());
    // Frame_Content_Size_flag = 2 (4 bytes), Single_Segment_flag = 1.
    header[4] = 0b1010_0000;
    header[5..].copy_from_slice(&raw_len.to_le_bytes());
    header
}

fn raw_frame_block_count(raw_len: u64) -> u64 {
    raw_len.div_ceil(RAW_FRAME_BLOCK_BYTES).max(1)
}

fn raw_frame_block_header(raw_len: u64, block: u64) -> [u8; RAW_FRAME_BLOCK_HEADER_BYTES as usize] {
    let size = raw_len
        .saturating_sub(block * RAW_FRAME_BLOCK_BYTES)
        .min(RAW_FRAME_BLOCK_BYTES) as u32;
    let last = u32::from(block + 1 == raw_frame_block_count(raw_len));
    // Last_Block bit, Block_Type 0 (raw), Block_Size.
    let value = last | (size << 3);
    let bytes = value.to_le_bytes();
    [bytes[0], bytes[1], bytes[2]]
}

/// Wire length of a raw-block frame holding `raw_len` bytes.
fn raw_frame_len(raw_len: u64) -> u64 {
    RAW_FRAME_HEADER_BYTES + raw_frame_block_count(raw_len) * RAW_FRAME_BLOCK_HEADER_BYTES + raw_len
}

/// `Write` adapter that emits exactly `raw_len` bytes as a raw-block frame.
struct RawFrameEncoder<W: Write> {
    inner: W,
    raw_len: u64,
    written: u64,
}

impl<W: Write> RawFrameEncoder<W> {
    fn new(mut inner: W, raw_len: u64) -> Result<Self, Error> {
        let header_len = u32::try_from(raw_len)
            .map_err(|_| Error::invariant("vector v5 exact page exceeds u32 bytes"))?;
        inner.write_all(&raw_frame_header(header_len))?;
        if raw_len == 0 {
            inner.write_all(&raw_frame_block_header(0, 0))?;
        }
        Ok(Self {
            inner,
            raw_len,
            written: 0,
        })
    }

    fn finish(self) -> Result<W, Error> {
        if self.written != self.raw_len {
            return Err(Error::invariant(format!(
                "vector v5 exact page wrote {} of {} bytes",
                self.written, self.raw_len
            )));
        }
        Ok(self.inner)
    }
}

impl<W: Write> Write for RawFrameEncoder<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        if self.written >= self.raw_len {
            return Err(std::io::Error::other(
                "raw frame content exceeds its length",
            ));
        }
        let offset_in_block = self.written % RAW_FRAME_BLOCK_BYTES;
        if offset_in_block == 0 {
            let block = self.written / RAW_FRAME_BLOCK_BYTES;
            self.inner
                .write_all(&raw_frame_block_header(self.raw_len, block))?;
        }
        let room = (RAW_FRAME_BLOCK_BYTES - offset_in_block).min(self.raw_len - self.written);
        let take = bytes.len().min(room as usize);
        self.inner.write_all(&bytes[..take])?;
        self.written += take as u64;
        Ok(take)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Append `raw` as a raw-block frame (an exact or navigation page).
fn append_raw_frame_block(output: &mut Vec<u8>, raw: &[u8]) -> Result<BlockRef, Error> {
    let raw_len = raw.len() as u64;
    if raw_len > MAX_RAW_BLOCK_BYTES || raw_frame_len(raw_len) > MAX_COMPRESSED_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 exact page too large ({raw_len} bytes)"
        )));
    }
    let offset = output.len();
    let mut encoder = RawFrameEncoder::new(&mut *output, raw_len)?;
    encoder.write_all(raw)?;
    encoder.finish()?;
    let wire = &output[offset..];
    Ok(BlockRef {
        offset: offset as u64,
        len: u32::try_from(wire.len())
            .map_err(|_| Error::invariant("vector v5 exact page exceeds u32"))?,
        raw_len: u32::try_from(raw_len)
            .map_err(|_| Error::invariant("vector v5 raw block exceeds u32"))?,
        crc32: crc32fast::hash(wire),
    })
}

/// Whether `block`'s length is that of a raw-block frame of its raw length.
/// The frame header is then checked once per page and every row against its
/// code; anything else takes the whole-page path.
fn raw_frame_candidate(block: &BlockRef) -> bool {
    block.raw_len > 0 && block.len as u64 == raw_frame_len(block.raw_len as u64)
}

fn raw_frame_header_range(block: &BlockRef) -> Range<u64> {
    block.offset..block.offset + RAW_FRAME_HEADER_BYTES
}

/// Absolute object ranges holding raw bytes `raw` of a raw-block frame (more
/// than one when the bytes straddle a block header).
fn raw_content_ranges(block: &BlockRef, raw: Range<u64>) -> Vec<Range<u64>> {
    let raw_len = block.raw_len as u64;
    let mut ranges = Vec::with_capacity(1);
    let mut cursor = raw.start;
    while cursor < raw.end {
        let index = cursor / RAW_FRAME_BLOCK_BYTES;
        let block_raw_end = ((index + 1) * RAW_FRAME_BLOCK_BYTES).min(raw_len);
        let content_start = block.offset
            + RAW_FRAME_HEADER_BYTES
            + index * (RAW_FRAME_BLOCK_HEADER_BYTES + RAW_FRAME_BLOCK_BYTES)
            + RAW_FRAME_BLOCK_HEADER_BYTES;
        let end = raw.end.min(block_raw_end);
        ranges.push(
            content_start + (cursor - index * RAW_FRAME_BLOCK_BYTES)
                ..content_start + (end - index * RAW_FRAME_BLOCK_BYTES),
        );
        cursor = end;
    }
    ranges
}

/// Rows closer than this are fetched in one range.
const RAW_ROW_COALESCE_GAP: u64 = 32 * 1024;

/// One row's bytes: `(span, offset in span, length)` pieces.
type RawRowPieces = Vec<(usize, usize, usize)>;

/// Append the ranges holding `candidate_rows` (ascending) to `spans`,
/// merging nearby rows, and return where each row's pieces live.
fn plan_raw_rows(
    exact: &BlockRef,
    candidate_rows: impl ExactSizeIterator<Item = u32>,
    row_bytes: u64,
    spans: &mut Vec<Range<u64>>,
) -> Vec<RawRowPieces> {
    let first_span = spans.len();
    let mut rows = Vec::with_capacity(candidate_rows.len());
    for candidate_row in candidate_rows {
        let raw_start = candidate_row as u64 * row_bytes;
        let mut pieces = Vec::with_capacity(1);
        for range in raw_content_ranges(exact, raw_start..raw_start + row_bytes) {
            let merge = spans.len() > first_span
                && spans.last().is_some_and(|last| {
                    range.start >= last.start && range.start <= last.end + RAW_ROW_COALESCE_GAP
                });
            if merge {
                let last = spans.last_mut().expect("checked");
                last.end = last.end.max(range.end);
            } else {
                spans.push(range.clone());
            }
            let span = spans.len() - 1;
            pieces.push((
                span,
                (range.start - spans[span].start) as usize,
                (range.end - range.start) as usize,
            ));
        }
        rows.push(pieces);
    }
    rows
}

fn f32_row(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("4 bytes")))
        .collect()
}

/// `‖q‖₁` and `‖q‖₂` of a query, for the code-consistency check.
struct QueryNorms {
    l1: f64,
    l2: f64,
}

impl QueryNorms {
    fn of(query: &[f32]) -> Self {
        Self {
            l1: query.iter().map(|v| (*v as f64).abs()).sum(),
            l2: query
                .iter()
                .map(|v| *v as f64 * *v as f64)
                .sum::<f64>()
                .sqrt(),
        }
    }
}

/// Whether an exact row read by range agrees with the checksummed int8 code
/// it was ranked by. Every coordinate of a code is within half a step of the
/// row (`|x_i − ŝc_i| ≤ s/2`, no clamping since `s = max|x|/127`), which bounds
/// how far the exact score can be from the approximate one. A row read from
/// the wrong place or damaged bytes fails it and the page is re-read whole
/// under its checksum. The tolerance is generous: this guards the layout, the
/// whole-page checksum still guards the bits.
fn exact_row_consistent(
    metric: VectorMetric,
    norms: &QueryNorms,
    row: &[f32],
    exact: f64,
    candidate: &ApproxCandidate,
) -> bool {
    if !exact.is_finite() || row.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let half_step = candidate.scale as f64 * 0.5 * (1.0 + 1e-3) + 1e-9;
    let error_l2 = half_step * (row.len() as f64).sqrt();
    let row_norm = row
        .iter()
        .map(|v| *v as f64 * *v as f64)
        .sum::<f64>()
        .sqrt();
    let approx = candidate.score as f64;
    let (bound, slack) = match metric {
        VectorMetric::Dot => (half_step * norms.l1, 1e-3 * norms.l2 * row_norm + 1e-6),
        VectorMetric::Cosine => {
            if row_norm == 0.0 || norms.l2 == 0.0 {
                return exact == 0.0 && approx == 0.0;
            }
            (2.0 * error_l2 / row_norm, 1e-3)
        }
        VectorMetric::Euclidean => (error_l2, 1e-3 * (norms.l2 + row_norm) + 1e-6),
    };
    (exact - approx).abs() <= bound + slack
}

/// Exact scores of `candidates` from rows read by range out of a raw-block
/// frame, or `None` when a row's length or its agreement with its code does
/// not check out.
#[allow(clippy::too_many_arguments)]
fn score_raw_rows(
    metric: VectorMetric,
    query: &[f32],
    norms: &QueryNorms,
    parts: &[Bytes],
    spans: &[Range<u64>],
    candidates: &[ApproxCandidate],
    rows: &[RawRowPieces],
    row_bytes: usize,
) -> Option<Vec<([u8; 16], f32)>> {
    let mut scored = Vec::with_capacity(candidates.len());
    let mut row = Vec::with_capacity(row_bytes);
    for (candidate, pieces) in candidates.iter().zip(rows) {
        row.clear();
        for &(span, offset, len) in pieces {
            let part = parts.get(span)?;
            if part.len() as u64 != spans[span].end - spans[span].start {
                return None;
            }
            row.extend_from_slice(part.get(offset..offset + len)?);
        }
        if row.len() != row_bytes {
            return None;
        }
        let vector = f32_row(&row);
        let score = metric_score(metric, &vector, query).0;
        if !exact_row_consistent(metric, norms, &vector, score, candidate) {
            return None;
        }
        scored.push((candidate.id, score as f32));
    }
    Some(scored)
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
    // Never trust the zstd frame's advertised content size: the output buffer
    // is exactly the checksummed directory's (already capped) raw length, and
    // a frame that would expand past it fails instead of growing it. One
    // bulk call decodes several times faster than a streaming reader.
    let raw = zstd::bulk::decompress(wire, block.raw_len as usize)
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
    use std::io::Read;
    use std::sync::Mutex;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::manifest::VectorQuantization;
    use crate::sst::vector::VectorFilterPosting;

    thread_local! {
        /// Build exact pages the way 3.4 and earlier did (zstd-compressed).
        pub(super) static COMPRESS_EXACT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    #[derive(Debug)]
    struct TrackingSource {
        bytes: Bytes,
        ranges: Mutex<Vec<Range<u64>>>,
        batch_sizes: Mutex<Vec<usize>>,
        nav_batch_sizes: Mutex<Vec<usize>>,
    }

    impl TrackingSource {
        fn new(bytes: Bytes) -> Self {
            Self {
                bytes,
                ranges: Mutex::new(Vec::new()),
                batch_sizes: Mutex::new(Vec::new()),
                nav_batch_sizes: Mutex::new(Vec::new()),
            }
        }

        fn max_nav_batch_size(&self) -> usize {
            self.nav_batch_sizes
                .lock()
                .expect("batch tracking lock")
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
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

        fn longest_range_since(&self, first: usize) -> u64 {
            self.ranges.lock().expect("tracking lock")[first..]
                .iter()
                .map(|range| range.end - range.start)
                .max()
                .unwrap_or(0)
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

        async fn read_blocks(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
            self.nav_batch_sizes
                .lock()
                .expect("batch tracking lock")
                .push(ranges.len());
            self.read_ranges(ranges).await
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
                    scale: 0.0,
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
            .search_filter_groups_int8(&[1.0, 2.0], 1, all_probe_options(reader.page_count()), &[])
            .await
            .is_err());
        assert_rabitq_code_corruption_is_detected(&body, &[1.0, 2.0], 1).await;
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
            .search_filter_groups_int8(&[1.0, 1.0], 3, all_probe_options(reader.page_count()), &[])
            .await
            .is_err());
        assert_rabitq_code_corruption_is_detected(&body, &[1.0, 1.0], 3).await;
    }

    /// The RaBitQ path never reads navigation pages; a flipped code byte
    /// must fail its block CRC when the page is probed.
    async fn assert_rabitq_code_corruption_is_detected(body: &Bytes, query: &[f32], k: usize) {
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source, body.len() as u64)
            .await
            .expect("open valid");
        let codes = reader.rabitq_code_block_range(0).expect("section present");
        let mut corrupt = body.to_vec();
        corrupt[codes.start as usize] ^= 0x01;
        let corrupt = Bytes::from(corrupt);
        let source = Arc::new(TrackingSource::new(corrupt.clone()));
        let reader = VectorV5Reader::open(source, corrupt.len() as u64)
            .await
            .expect("directory remains valid");
        assert!(reader.has_rabitq());
        assert!(reader
            .search(query, k, all_probe_options(reader.page_count()))
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

    /// Recall@10 over a bench-like corpus (clusters of ~6k rows around random
    /// centroids, Gaussian noise of norm `spread`) for each `(nprobe,
    /// rerank)`, built with the in-memory or the spooling (compaction) builder.
    async fn clustered_recall(
        rows: u32,
        cluster_rows: u32,
        dim: usize,
        external: bool,
        combos: &[(usize, usize)],
    ) -> Vec<f64> {
        let clusters = (rows / cluster_rows).max(1);
        let spread = 0.35f32;
        let gauss = |rng: &mut ChaCha8Rng| -> f32 {
            let u1: f32 = rng.gen_range(f32::MIN_POSITIVE..1.0);
            let u2: f32 = rng.gen_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| {
                let mut c: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect();
                normalize(&mut c);
                c
            })
            .collect();
        let noise = spread / (dim as f32).sqrt();
        let around = |rng: &mut ChaCha8Rng, c: &[f32]| {
            let mut v: Vec<f32> = c.iter().map(|x| x + gauss(rng) * noise).collect();
            normalize(&mut v);
            v
        };
        let members: Vec<_> = (0..rows)
            .map(|i| (id(i), around(&mut rng, &centers[(i % clusters) as usize])))
            .collect();
        let queries: Vec<Vec<f32>> = (0..40)
            .map(|q| {
                let c = &centers[q % clusters as usize];
                around(&mut rng, c)
            })
            .collect();
        let truth: Vec<BTreeSet<[u8; 16]>> = queries
            .iter()
            .map(|query| {
                let mut exact: Vec<_> = members
                    .iter()
                    .map(|(node_id, vector)| {
                        (
                            *node_id,
                            metric_score(VectorMetric::Cosine, vector, query).0 as f32,
                        )
                    })
                    .collect();
                sort_hits(VectorMetric::Cosine, &mut exact);
                exact.truncate(10);
                exact.into_iter().map(|hit| hit.0).collect()
            })
            .collect();
        let started = std::time::Instant::now();
        let body = if external {
            let mut members = members;
            members.sort_by_key(|(node_id, _)| *node_id);
            let mut collector = external::VectorV5ExternalCollector::new_base(
                external::VectorV5ExternalBuildConfig::default(),
            )
            .expect("collector");
            for (node_id, vector) in &members {
                collector
                    .push(*node_id, vector, &BTreeMap::new())
                    .expect("push");
            }
            let mut artifact = collector
                .finish(
                    &descriptor(VectorMetric::Cosine, dim as u32),
                    VectorV5BuildOptions::default(),
                )
                .expect("build")
                .expect("artifact");
            artifact.rewind().expect("rewind");
            let mut body = Vec::new();
            std::io::Read::read_to_end(&mut artifact.file, &mut body).expect("read");
            Bytes::from(body)
        } else {
            build_body_v5(
                &descriptor(VectorMetric::Cosine, dim as u32),
                members,
                VectorV5BuildOptions::default(),
            )
            .expect("build")
            .expect("body")
            .0
        };
        eprintln!("built {} bytes in {:?}", body.len(), started.elapsed());
        let mut recalls = Vec::with_capacity(combos.len());
        for &(nprobe, rerank) in combos {
            let source = Arc::new(TrackingSource::new(body.clone()));
            let reader = VectorV5Reader::open(source.clone(), body.len() as u64)
                .await
                .expect("open");
            let pages = reader.page_count();
            let opened = source.range_count();
            let mut recall = 0usize;
            for (query, truth) in queries.iter().zip(&truth) {
                let hits = reader
                    .search(
                        query,
                        10,
                        VectorV5SearchOptions {
                            nprobe: nprobe.min(pages),
                            max_nprobe: nprobe.min(pages),
                            rerank_factor: rerank,
                        },
                    )
                    .await
                    .expect("search");
                recall += hits.iter().filter(|hit| truth.contains(&hit.0)).count();
            }
            let recall = recall as f64 / (10.0 * queries.len() as f64);
            eprintln!(
                "pages={pages} nprobe={nprobe} rerank={rerank} recall@10={recall:.3} ranges/query={:.1}",
                (source.range_count() - opened) as f64 / queries.len() as f64
            );
            recalls.push(recall);
        }
        recalls
    }

    /// Recall/IO sweep. Run by hand:
    /// `NAMIDB_V5_SWEEP_ROWS=377000 cargo test ... v5_recall_sweep -- --ignored --nocapture`
    /// (`NAMIDB_V5_SWEEP_EXTERNAL=1` for the spooling builder).
    /// Recall@10 against exact search on production-shaped embeddings:
    /// Zipf-sized clusters with per-row spread, new query vectors drawn the
    /// same way. `NAMIDB_SWEEP_ROWS` (default 378532), `NAMIDB_SWEEP_OLD=1`
    /// builds with the direction split instead of k-means.
    #[tokio::test]
    #[ignore]
    async fn v5_prodshape_recall_sweep() {
        let rows: usize = std::env::var("NAMIDB_SWEEP_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(378_532);
        let old = std::env::var_os("NAMIDB_SWEEP_OLD").is_some_and(|v| v == "1");
        KMEANS_OFF.store(old, std::sync::atomic::Ordering::Relaxed);
        let dim = 1024usize;
        let clusters = 1500usize;
        let gauss = |rng: &mut ChaCha8Rng| -> f32 {
            let u1: f32 = rng.gen_range(f32::MIN_POSITIVE..1.0);
            let u2: f32 = rng.gen_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| {
                let mut c: Vec<f32> = (0..dim).map(|_| gauss(&mut rng)).collect();
                normalize(&mut c);
                c
            })
            .collect();
        let weights: Vec<f64> = (1..=clusters).map(|r| 1.0 / (r as f64).powf(0.8)).collect();
        let total: f64 = weights.iter().sum();
        let mut cumulative = Vec::with_capacity(clusters);
        let mut acc = 0.0;
        for w in &weights {
            acc += w / total;
            cumulative.push(acc);
        }
        let draw = |rng: &mut ChaCha8Rng| -> Vec<f32> {
            let u: f64 = rng.gen_range(0.0..1.0);
            let c = cumulative.partition_point(|x| *x < u).min(clusters - 1);
            let spread: f32 = rng.gen_range(0.45..0.8) / (dim as f32).sqrt();
            let mut v: Vec<f32> = centers[c].iter().map(|x| x + gauss(rng) * spread).collect();
            normalize(&mut v);
            v
        };
        let members: Vec<([u8; 16], Vec<f32>)> =
            (0..rows as u32).map(|i| (id(i), draw(&mut rng))).collect();
        let queries: Vec<Vec<f32>> = (0..100).map(|_| draw(&mut rng)).collect();
        let started = std::time::Instant::now();
        let truth: Vec<BTreeSet<[u8; 16]>> = queries
            .iter()
            .map(|query| {
                let mut exact: Vec<_> = members
                    .iter()
                    .map(|(node_id, vector)| (*node_id, dot_lanes(vector, query)))
                    .collect();
                exact.select_nth_unstable_by(10, |a, b| b.1.total_cmp(&a.1));
                exact.truncate(10);
                exact.into_iter().map(|hit| hit.0).collect()
            })
            .collect();
        eprintln!("truth in {:?}", started.elapsed());
        let started = std::time::Instant::now();
        let mut members = members;
        members.sort_by_key(|(node_id, _)| *node_id);
        let mut collector = external::VectorV5ExternalCollector::new_base(
            external::VectorV5ExternalBuildConfig::default(),
        )
        .expect("collector");
        for (node_id, vector) in &members {
            collector
                .push(*node_id, vector, &BTreeMap::new())
                .expect("push");
        }
        drop(members);
        let mut artifact = collector
            .finish(
                &descriptor(VectorMetric::Cosine, dim as u32),
                VectorV5BuildOptions::default(),
            )
            .expect("build")
            .expect("artifact");
        eprintln!(
            "built (old={old}) in {:?}: {:?}",
            started.elapsed(),
            artifact.stats
        );
        artifact.rewind().expect("rewind");
        let mut body = Vec::new();
        std::io::Read::read_to_end(&mut artifact.file, &mut body).expect("read");
        let body = Bytes::from(body);
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source, body.len() as u64)
            .await
            .expect("open");
        let pages = reader.page_count();
        for nprobe in [16usize, 24, 32, 48, 64, 96, 128, 192] {
            for rerank in [8usize, 16] {
                let started = std::time::Instant::now();
                let mut recall = 0usize;
                for (query, truth) in queries.iter().zip(&truth) {
                    let hits = reader
                        .search(
                            query,
                            10,
                            VectorV5SearchOptions {
                                nprobe: nprobe.min(pages),
                                max_nprobe: nprobe.min(pages),
                                rerank_factor: rerank,
                            },
                        )
                        .await
                        .expect("search");
                    recall += hits.iter().filter(|hit| truth.contains(&hit.0)).count();
                }
                eprintln!(
                    "old={old} pages={pages} nprobe={nprobe} rerank={rerank} recall@10={:.3} cpu/query={:?}",
                    recall as f64 / 1000.0,
                    started.elapsed() / queries.len() as u32
                );
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn v5_recall_sweep() {
        let rows: u32 = std::env::var("NAMIDB_V5_SWEEP_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(48_000);
        let dim: usize = std::env::var("NAMIDB_V5_SWEEP_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        let external = std::env::var_os("NAMIDB_V5_SWEEP_EXTERNAL").is_some_and(|v| v == "1");
        clustered_recall(
            rows,
            5_900,
            dim,
            external,
            &[
                (4, 8),
                (8, 8),
                (16, 8),
                (24, 16),
                (32, 16),
                (48, 16),
                (64, 16),
                (128, 16),
            ],
        )
        .await;
    }

    /// Leaves split along a principal direction keep a natural cluster's
    /// neighbours in few leaves. Axis-aligned splits (3.4.0) scattered them:
    /// recall@10 0.78 here, against 0.98.
    #[tokio::test]
    async fn clustered_neighbours_stay_in_few_leaves() {
        for external in [false, true] {
            // 64 clusters of 1000 rows in 128 leaves; 4 probes.
            let recall = clustered_recall(64_000, 1_000, 512, external, &[(4, 8)]).await;
            assert!(
                recall[0] >= 0.9,
                "external={external} recall@10={} at nprobe 4",
                recall[0]
            );
        }
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
            source.max_nav_batch_size() <= MAX_NAV_BATCH_PAGES,
            "held {} compressed navigation pages at once",
            source.max_nav_batch_size()
        );
    }

    #[test]
    fn fast_nav_decoder_matches_bincode_and_rejects_what_it_rejects() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for (rows, dim, filter_props) in [
            (0usize, 4usize, 0usize),
            (1, 3, 0),
            (37, 8, 2),
            (512, 16, 1),
        ] {
            let words = rows.div_ceil(64);
            let mut filters = BTreeMap::new();
            for p in 0..filter_props {
                let mut values = BTreeMap::new();
                for v in 0..3 {
                    values.insert(
                        format!("válue-{v}"),
                        (0..words).map(|_| rng.gen::<u64>()).collect::<Vec<_>>(),
                    );
                }
                filters.insert(format!("prop{p}"), values);
            }
            let page = NavPage {
                ids: (0..rows).map(|r| id(r as u32)).collect(),
                codes: (0..rows * dim).map(|_| rng.gen::<i8>()).collect(),
                scales: (0..rows).map(|_| rng.gen::<f32>()).collect(),
                filters,
            };
            let raw = serialize(&page, "test").unwrap();
            let slow: NavPage = deserialize(&raw, raw.len() as u64, "test").unwrap();
            let fast = decode_nav_raw(&raw).unwrap();
            assert_eq!(fast.ids, slow.ids);
            assert_eq!(fast.codes, slow.codes);
            assert_eq!(
                fast.scales.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                slow.scales.iter().map(|s| s.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(fast.filters, slow.filters);
            for cut in [0, 1, 7, 8, raw.len() / 2, raw.len().saturating_sub(1)] {
                if cut < raw.len() {
                    assert!(decode_nav_raw(&raw[..cut]).is_err(), "cut {cut}");
                    assert!(deserialize::<NavPage>(&raw[..cut], raw.len() as u64, "t").is_err());
                }
            }
            let mut trailing = raw.clone();
            trailing.push(0);
            assert!(decode_nav_raw(&trailing).is_err());
        }
        // A declared length larger than the input fails without allocating it.
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend([0u8; 32]);
        assert!(decode_nav_raw(&huge).is_err());
    }

    #[test]
    fn raw_frames_decode_with_every_zstd_reader() {
        for len in [1usize, 4096, 131_071, 131_072, 131_073, 2 * 1024 * 1024 + 5] {
            let raw: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            let mut body = vec![7u8; 3];
            let block = append_raw_frame_block(&mut body, &raw).unwrap();
            assert!(raw_frame_candidate(&block));
            let wire = &body[block.offset as usize..];
            assert_eq!(wire.len() as u64, raw_frame_len(len as u64));
            assert_eq!(decode_block(wire, &block, "t").unwrap(), raw);
            // The streaming decoder earlier releases use.
            let mut streamed = Vec::new();
            zstd::stream::read::Decoder::new(std::io::Cursor::new(wire))
                .unwrap()
                .read_to_end(&mut streamed)
                .unwrap();
            assert_eq!(streamed, raw);
            // Every row range maps back onto the raw content.
            for (start, end) in [(0u64, 1u64), (len as u64 - 1, len as u64), (0, len as u64)] {
                let mut bytes = Vec::new();
                for range in raw_content_ranges(&block, start..end) {
                    bytes.extend_from_slice(&body[range.start as usize..range.end as usize]);
                }
                assert_eq!(bytes, raw[start as usize..end as usize]);
            }
        }
    }

    fn rerank_members(rows: u32, dim: usize) -> Vec<([u8; 16], Vec<f32>)> {
        let mut rng = ChaCha8Rng::seed_from_u64(29);
        (0..rows)
            .map(|ordinal| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                normalize(&mut v);
                (id(ordinal), v)
            })
            .collect()
    }

    async fn build_reader(
        metric: VectorMetric,
        dim: usize,
        members: Vec<([u8; 16], Vec<f32>)>,
        compress_exact: bool,
    ) -> (VectorV5Reader, Arc<TrackingSource>, Bytes) {
        COMPRESS_EXACT.with(|flag| flag.set(compress_exact));
        let built = build_body_v5(
            &descriptor(metric, dim as u32),
            members,
            VectorV5BuildOptions {
                target_rows_per_page: if dim >= 1024 { 512 } else { 64 },
                branch_factor: 4,
                compression_level: 1,
            },
        );
        COMPRESS_EXACT.with(|flag| flag.set(false));
        let (body, _) = built.expect("build").expect("body");
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source.clone(), body.len() as u64)
            .await
            .expect("open");
        (reader, source, body)
    }

    #[tokio::test]
    async fn open_reads_header_trailer_and_footer_in_one_round_trip() {
        let (_, _, body) =
            build_reader(VectorMetric::Cosine, 96, rerank_members(600, 96), false).await;
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source.clone(), body.len() as u64)
            .await
            .expect("open");
        assert!(reader.page_count() > 1);
        // One batched request (header + tail), no separate footer read.
        assert_eq!(source.range_count(), 2);
        assert_eq!(*source.batch_sizes.lock().unwrap(), vec![2]);
        // The guess is bounded both ways.
        assert!(speculative_footer_bytes(body.len() as u64) >= 64 * 1024);
        assert!(speculative_footer_bytes(u64::MAX) == 8 * 1024 * 1024);
    }

    #[tokio::test]
    async fn row_reads_return_what_whole_compressed_pages_return() {
        // 3.4 bases (compressed exact pages) and 3.5 bases (raw frames) give
        // the same hits and scores; the 3.5 base reads only candidate rows.
        // dim 96 makes rows straddle the 128 KiB block boundaries.
        for dim in [96usize, 1024] {
            for metric in [
                VectorMetric::Cosine,
                VectorMetric::Dot,
                VectorMetric::Euclidean,
            ] {
                let members = rerank_members(if dim >= 1024 { 2_000 } else { 600 }, dim);
                let queries: Vec<Vec<f32>> =
                    members.iter().step_by(97).map(|m| m.1.clone()).collect();
                let (old, old_source, _) = build_reader(metric, dim, members.clone(), true).await;
                let (new, new_source, _) = build_reader(metric, dim, members, false).await;
                let options = VectorV5SearchOptions {
                    nprobe: 3,
                    max_nprobe: 3,
                    rerank_factor: 8,
                };
                let page_bytes = if dim >= 1024 { 512 } else { 64 } * dim as u64 * 4;
                for query in &queries {
                    let before_old = old_source.bytes_read();
                    let before_new = new_source.bytes_read();
                    let first_new = new_source.range_count();
                    // The int8 tier's fixed rerank_factor fan-out isolates the
                    // exact-page format (RaBitQ fan-out is bound-driven).
                    let a = old
                        .search_filter_groups_int8(query, 10, options, &[])
                        .await
                        .expect("old")
                        .hits;
                    let b = new
                        .search_filter_groups_int8(query, 10, options, &[])
                        .await
                        .expect("new")
                        .hits;
                    assert_eq!(a, b, "{metric:?} dim {dim}");
                    let old_bytes = old_source.bytes_read() - before_old;
                    let new_bytes = new_source.bytes_read() - before_new;
                    // Small rows coalesce into whole-page spans; wide ones
                    // must cost a fraction of the pages.
                    if dim >= 1024 {
                        assert!(
                            new_bytes < old_bytes,
                            "{metric:?} dim {dim}: rows read {new_bytes} vs pages {old_bytes}"
                        );
                    }
                    // No exact page was read whole.
                    assert!(new_source.longest_range_since(first_new) < page_bytes);
                }
            }
        }
    }

    #[tokio::test]
    async fn a_damaged_row_is_never_scored_silently() {
        let dim = 64usize;
        let members = rerank_members(300, dim);
        let query = members[5].1.clone();
        let (reader, _, body) = build_reader(VectorMetric::Cosine, dim, members, false).await;
        let options = VectorV5SearchOptions {
            nprobe: reader.page_count(),
            max_nprobe: reader.page_count(),
            rerank_factor: 32,
        };
        let hits = reader.search(&query, 5, options).await.expect("search");
        assert_eq!(id_value(&hits[0].0), 5);
        // Flip the exponent byte of one coordinate of row 5 wherever it lives.
        let needle: Vec<u8> = query.iter().flat_map(|v| v.to_le_bytes()).collect();
        let at = body
            .windows(needle.len())
            .position(|window| window == needle.as_slice())
            .expect("row 5 is stored contiguously");
        let mut damaged = body.to_vec();
        damaged[at + 4 * 7 + 3] ^= 0x40;
        let damaged = Bytes::from(damaged);
        let source = Arc::new(TrackingSource::new(damaged.clone()));
        let reader = VectorV5Reader::open(source, damaged.len() as u64)
            .await
            .expect("open");
        let result = reader.search(&query, 5, options).await;
        assert!(
            result.is_err(),
            "damaged row must fail the page checksum: {result:?}"
        );
    }

    /// Warm-cache CPU cost of one serving query on a realistic base:
    /// `cargo test --release -p namidb-storage --features vector-index --lib
    /// v5_warm_query_cost -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn v5_warm_query_cost() {
        let rows: u32 = std::env::var("V5_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);
        let dim = 1024usize;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let members: Vec<_> = (0..rows)
            .map(|ordinal| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                normalize(&mut v);
                (id(ordinal), v)
            })
            .collect();
        let started = std::time::Instant::now();
        let (body, _) = build_body_v5(
            &descriptor(VectorMetric::Cosine, dim as u32),
            members,
            VectorV5BuildOptions::default(),
        )
        .expect("build")
        .expect("body");
        eprintln!("built {} bytes in {:?}", body.len(), started.elapsed());
        let source = Arc::new(TrackingSource::new(body.clone()));
        let reader = VectorV5Reader::open(source, body.len() as u64)
            .await
            .expect("open");
        let pages = reader.page_count();
        let options = VectorV5SearchOptions {
            nprobe: 24.min(pages),
            max_nprobe: 192.min(pages),
            rerank_factor: 8,
        };
        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                normalize(&mut v);
                v
            })
            .collect();
        for q in &queries[..5] {
            reader.search(q, 10, options).await.expect("search");
        }
        let started = std::time::Instant::now();
        for q in &queries {
            reader.search(q, 10, options).await.expect("search");
        }
        eprintln!(
            "pages={pages} per query {:?} (nprobe {}, rerank x{})",
            started.elapsed() / queries.len() as u32,
            options.nprobe,
            options.rerank_factor
        );
        eprintln!("{}", namidb_core::profile::dump_table());
    }
}
