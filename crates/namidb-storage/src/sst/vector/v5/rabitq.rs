//! RaBitQ 1-bit navigation tier for `NAMIVG05` bases.
//!
//! The int8 navigation pages of a V5 base cost `dim + 4` bytes per row (plus
//! its NodeId), and a warm query wants every probed page resident. This
//! module adds an optional section with one bit per (rotated) dimension and
//! three correction factors per row, the estimator of Gao & Long, *RaBitQ:
//! Quantizing High-Dimensional Vectors with a Theoretical Error Bound for
//! Approximate Nearest Neighbor Search* (SIGMOD 2024):
//!
//! ```text
//! +-------+-----------+-------------+---------------------------+-----------+---------+--------+---------+
//! | magic | nav pages | exact pages | RaBitQ page blocks        | directory | locator | footer | trailer |
//! +-------+-----------+-------------+---------------------------+-----------+---------+--------+---------+
//!                                    per page: codes | ids | filters
//! ```
//!
//! * **Compatibility.** Every V5 reader bounds page blocks by the footer
//!   offset (`range.end <= footer_offset`), never requires the last exact page
//!   to end at it, and reads only the blocks its footer names. The section
//!   lives in that gap, so a reader that predates it serves the base through
//!   the unchanged int8 path. The trailer and footer are byte-for-byte what
//!   they were. A new reader looks for the fixed-size [`LOCATOR_LEN`] locator
//!   only when the gap is large enough to hold it, and validates magic,
//!   checksums and every offset before using the section; a base without one
//!   (or with one it cannot validate) keeps the int8 path.
//! * **Rotation.** A seeded randomized block Hadamard transform: `rounds`
//!   times, flip signs, permute, and apply a normalized Walsh-Hadamard
//!   transform to each block (the largest power of two dividing the padded
//!   dimension, so 1024 is one block and 1536 three of 512). Every step is
//!   orthogonal and the padded dimension is only rounded up to a multiple of
//!   64, so a 1536-d row costs 192 code bytes, not 256.
//! * **Center.** A row is coded relative to its leaf centroid: the footer's
//!   int8 centroid dequantized and multiplied by a per-page scalar stored in
//!   the directory (the least-squares scale of that direction for the leaf's
//!   rows; for cosine bases this turns the unit-norm direction back into the
//!   mean of the normalized rows). Builder and reader compute it with the same
//!   function from the same bytes.
//! * **Estimator.** With `r = x - c`, `o = r/|r|`, rotated sign code `ō` and
//!   `α = <ō, P o>`, `<r, y>` is estimated as `|r|/α · <ō, P y>` for
//!   `y = q - c`, and with probability about `1 - 2·exp(-c0·ε0²)` its error is
//!   below `ε0 · |r|·sqrt(1 - α²)/α · |y| / sqrt(D - 1)` (Theorem 3.2).
//!   Inner products follow from `<x, q> = <c, q> + <r, c> + <r, q - c>`, and
//!   squared distances from `|x - q|² = |r|² + |q - c|² - 2<r, q - c>`.
//! * **Candidates.** A query keeps the `k`-th largest lower bound `τ` of the
//!   rows it scanned, and re-ranks (in exact f32) every row whose upper bound
//!   reaches `τ`: a row whose upper bound stays below `τ` cannot be in the
//!   exact top `k` unless a bound failed.
//!
//! Only the code blocks (`D/8 + 12` bytes per row) are read by every probe;
//! ids are read for the pages holding re-rank candidates and filter bitmaps
//! only by filtered queries. All three go through the reader's range source,
//! so whatever stays resident is charged to the range cache's budget.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::io::Write;
use std::ops::Range;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Mutex, OnceLock};

use bytes::Bytes;
use namidb_core::Value;
use serde::{Deserialize, Serialize};

use super::{
    deserialize, metric_score, plan_raw_rows, prepare_groups, raw_frame_header,
    raw_frame_header_range, require_len, row_matches_filters, serialize, shared_search_workspace,
    BoundedExactHits, Footer, RabitqState, VectorMetric, VectorV5Reader, VectorV5SearchOptions,
    VectorV5SearchResult, MAX_EXACT_BATCH_PAGES, MAX_NAV_BATCH_PAGES, NAV_DECODE_WORKSPACE_FACTOR,
    RAW_NO, RAW_UNKNOWN, RAW_YES,
};
use crate::error::Error;

/// Magic of the fixed-size locator written immediately before the footer.
pub(crate) const LOCATOR_MAGIC: &[u8; 8] = b"NVRBQ1LC";
/// `magic | directory offset u64 | directory length u64 | directory crc32 |
/// crc32 of the preceding 28 bytes`.
pub(crate) const LOCATOR_LEN: usize = 32;
const SECTION_VERSION: u16 = 1;
/// Bits per rotated coordinate held in a code block.
const CODE_BITS: u8 = 1;
const ROTATION_SEED: u64 = 0x5241_4249_5451_0001;
const ROTATION_ROUNDS: u8 = 4;
const MAX_ROTATION_ROUNDS: u8 = 16;
/// The rotated (padded) dimension is a multiple of this.
const PAD_MULTIPLE: usize = 64;
/// `[a, b, e]` per row, see [`encode_row`].
const FACTORS_PER_ROW: usize = 3;
const FACTOR_BYTES: usize = FACTORS_PER_ROW * 4;
const ID_BYTES: usize = 16;
const MAX_DIRECTORY_BYTES: u64 = 512 * 1024 * 1024;
/// Confidence parameter of the error bound. Gao & Long use 1.9: the bound
/// then fails for a few percent of rows at most (fewer for the close rows a
/// top-k query cares about, whose error shrinks with `sqrt(1 - <o, q>²)`),
/// and a true neighbour is lost only when its upper bound falls below the
/// `k`-th lower bound, which needs a failure of about twice that size.
pub(crate) const EPSILON0: f32 = 1.9;

/// Benchmark-only override of [`EPSILON0`] (stored as f32 bits; 0 = unset).
static EPSILON_OVERRIDE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Set the bound's confidence parameter for this process (benchmarks that
/// sweep it). Not a serving knob: the default is what the engine ships.
#[doc(hidden)]
pub fn override_epsilon_for_benchmarks(epsilon: f32) {
    EPSILON_OVERRIDE.store(epsilon.to_bits(), AtomicOrdering::Relaxed);
}

fn epsilon() -> f32 {
    match EPSILON_OVERRIDE.load(AtomicOrdering::Relaxed) {
        0 => EPSILON0,
        bits => f32::from_bits(bits),
    }
}
/// Re-rank candidates are bounded by `rerank_factor · k · CANDIDATE_CAP_FACTOR`
/// rows (the int8 path re-ranks `rerank_factor · k`). The bound alone selects
/// fewer on every dataset measured; the cap only bounds a pathological query.
const CANDIDATE_CAP_FACTOR: usize = 8;

#[cfg(test)]
thread_local! {
    /// Test-only: build bases as releases before the section did.
    pub(crate) static SKIP_SECTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether builders write the section (always, outside tests that need a
/// base as earlier releases wrote it).
pub(crate) fn section_enabled() -> bool {
    #[cfg(test)]
    {
        !SKIP_SECTION.with(|flag| flag.get())
    }
    #[cfg(not(test))]
    {
        true
    }
}

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

/// SplitMix64: a tiny, fully specified generator, so the rotation a base was
/// coded with is reproducible from its seed on every platform and release.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        // Bounds here are < 2^32, so the modulo bias is below 2^-32.
        (self.next() % bound as u64) as usize
    }
}

#[derive(Debug, Clone)]
struct RotationRound {
    signs: Vec<f32>,
    perm: Vec<u32>,
}

/// Seeded randomized block Hadamard rotation of `dim`-vectors into
/// `padded`-vectors (zero padded; norms and inner products are preserved).
#[derive(Debug, Clone)]
pub(crate) struct Rotation {
    dim: usize,
    padded: usize,
    block: usize,
    rounds: Vec<RotationRound>,
}

impl Rotation {
    pub(crate) fn new(dim: usize, seed: u64, rounds: u8) -> Result<Self, Error> {
        if dim == 0 || rounds == 0 || rounds > MAX_ROTATION_ROUNDS {
            return Err(Error::invariant("vector v5 rabitq rotation is invalid"));
        }
        let padded = dim
            .checked_next_multiple_of(PAD_MULTIPLE)
            .ok_or_else(|| Error::invariant("vector v5 rabitq dimension overflows"))?;
        let block = 1usize << padded.trailing_zeros();
        let mut rng = SplitMix64(seed ^ (padded as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let rounds = (0..rounds)
            .map(|_| {
                let signs = (0..padded)
                    .map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })
                    .collect();
                let mut perm: Vec<u32> = (0..padded as u32).collect();
                for i in (1..padded).rev() {
                    perm.swap(i, rng.below(i + 1));
                }
                RotationRound { signs, perm }
            })
            .collect();
        Ok(Self {
            dim,
            padded,
            block,
            rounds,
        })
    }

    #[cfg(test)]
    pub(crate) fn padded_dim(&self) -> usize {
        self.padded
    }

    fn resident_bytes(&self) -> usize {
        self.rounds.len() * self.padded * (size_of::<f32>() + size_of::<u32>())
    }

    /// Rotate `input` (`dim` values) into `out` (`padded` values). `scratch`
    /// must also hold `padded` values.
    fn apply(&self, input: &[f32], out: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(input.len(), self.dim);
        out[..self.dim].copy_from_slice(input);
        out[self.dim..].fill(0.0);
        let norm = 1.0 / (self.block as f32).sqrt();
        for round in &self.rounds {
            for (dst, &src) in scratch.iter_mut().zip(&round.perm) {
                let src = src as usize;
                *dst = out[src] * round.signs[src];
            }
            for block in scratch.chunks_exact_mut(self.block) {
                fwht(block);
            }
            for (dst, src) in out.iter_mut().zip(scratch.iter()) {
                *dst = *src * norm;
            }
        }
    }
}

/// Unnormalized in-place fast Walsh-Hadamard transform (`len` a power of two).
fn fwht(values: &mut [f32]) {
    let len = values.len();
    let mut half = 1;
    while half < len {
        for start in (0..len).step_by(half * 2) {
            let (left, right) = values[start..start + half * 2].split_at_mut(half);
            for (a, b) in left.iter_mut().zip(right.iter_mut()) {
                let (x, y) = (*a, *b);
                *a = x + y;
                *b = x - y;
            }
        }
        half *= 2;
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Vector a row or query is compared as: unit length for cosine (whose score
/// is the inner product of normalized vectors), itself otherwise.
fn effective_into(metric: VectorMetric, vector: &[f32], out: &mut Vec<f32>) {
    out.clear();
    if metric == VectorMetric::Cosine {
        let norm = vector
            .iter()
            .map(|v| *v as f64 * *v as f64)
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            out.extend(vector.iter().map(|v| (*v as f64 / norm) as f32));
        } else {
            out.resize(vector.len(), 0.0);
        }
    } else {
        out.extend_from_slice(vector);
    }
}

/// The center rows of a page are coded against: the footer's int8 leaf
/// centroid, dequantized, times the page's `center_scale`.
pub(crate) fn leaf_center(codes: &[i8], scale: f32, center_scale: f32, out: &mut Vec<f32>) {
    out.clear();
    out.extend(codes.iter().map(|c| (*c as f32 * scale) * center_scale));
}

/// Least-squares scale of a leaf's dequantized centroid direction for its
/// rows: `Σ <x, c> / (n |c|²)`.
#[derive(Debug, Default)]
pub(crate) struct CenterScale {
    sum_dot: f64,
    rows: u64,
    effective: Vec<f32>,
}

impl CenterScale {
    pub(crate) fn push(&mut self, metric: VectorMetric, vector: &[f32], centroid: &[f32]) {
        effective_into(metric, vector, &mut self.effective);
        self.sum_dot += self
            .effective
            .iter()
            .zip(centroid)
            .map(|(x, c)| *x as f64 * *c as f64)
            .sum::<f64>();
        self.rows += 1;
    }

    pub(crate) fn finish(&self, centroid: &[f32]) -> f32 {
        let norm_sq = centroid.iter().map(|c| *c as f64 * *c as f64).sum::<f64>();
        if self.rows == 0 || norm_sq == 0.0 {
            return 0.0;
        }
        let scale = (self.sum_dot / (self.rows as f64 * norm_sq)) as f32;
        if scale.is_finite() {
            scale
        } else {
            0.0
        }
    }
}

/// Code bytes per row for a rotation.
fn code_bytes(rotation: &Rotation) -> usize {
    rotation.padded / 8
}

/// Per-row quantities that can be recomputed from the exact row to check that
/// a range-read row is the one the code was built from.
#[derive(Debug, Clone, Copy)]
struct ResidualStats {
    /// `|r|`.
    norm: f64,
    /// `<r, c>` (inner-product metrics) or `|r|²` (Euclidean).
    additive: f64,
}

fn residual_into(
    metric: VectorMetric,
    vector: &[f32],
    center: &[f32],
    effective: &mut Vec<f32>,
    residual: &mut Vec<f32>,
) -> ResidualStats {
    effective_into(metric, vector, effective);
    residual.clear();
    residual.extend(effective.iter().zip(center).map(|(x, c)| *x - *c));
    let norm_sq = residual.iter().map(|v| *v as f64 * *v as f64).sum::<f64>();
    let additive = if metric == VectorMetric::Euclidean {
        norm_sq
    } else {
        residual
            .iter()
            .zip(center)
            .map(|(r, c)| *r as f64 * *c as f64)
            .sum::<f64>()
    };
    ResidualStats {
        norm: norm_sq.sqrt(),
        additive,
    }
}

/// Reusable buffers for coding rows.
#[derive(Debug)]
pub(crate) struct Encoder {
    metric: VectorMetric,
    rotation: Rotation,
    effective: Vec<f32>,
    residual: Vec<f32>,
    rotated: Vec<f32>,
    scratch: Vec<f32>,
}

impl Encoder {
    pub(crate) fn new(metric: VectorMetric, dim: usize) -> Result<Self, Error> {
        let rotation = Rotation::new(dim, ROTATION_SEED, ROTATION_ROUNDS)?;
        let padded = rotation.padded;
        Ok(Self {
            metric,
            rotation,
            effective: Vec::with_capacity(dim),
            residual: Vec::with_capacity(dim),
            rotated: vec![0.0; padded],
            scratch: vec![0.0; padded],
        })
    }

    pub(crate) fn code_bytes(&self) -> usize {
        code_bytes(&self.rotation)
    }

    /// Append `vector`'s sign code to `codes` and its factors
    /// `[a, b, e]` to `factors`, where `a` is `<r, c>` (`|r|²` for
    /// Euclidean), `b = |r|/α` scales `<ō, P y>` into `<r, y>`, and
    /// `e = |r|·sqrt(1-α²)/α` is the error scale of that estimate.
    fn encode_row(
        &mut self,
        vector: &[f32],
        center: &[f32],
        codes: &mut Vec<u8>,
        factors: &mut Vec<f32>,
    ) -> Result<(), Error> {
        let stats = residual_into(
            self.metric,
            vector,
            center,
            &mut self.effective,
            &mut self.residual,
        );
        let code_bytes = self.code_bytes();
        if stats.norm == 0.0 {
            codes.resize(codes.len() + code_bytes, 0);
            factors.extend_from_slice(&[stats.additive as f32, 0.0, 0.0]);
            return Ok(());
        }
        self.rotation
            .apply(&self.residual, &mut self.rotated, &mut self.scratch);
        let l1 = self.rotated.iter().map(|v| v.abs() as f64).sum::<f64>();
        let alpha = (l1 / (stats.norm * (self.rotation.padded as f64).sqrt())).min(1.0);
        // Also rejects NaN.
        if alpha.is_nan() || alpha <= 0.0 {
            return Err(Error::invariant("vector v5 rabitq code has no alignment"));
        }
        for chunk in self.rotated.chunks_exact(8) {
            let mut byte = 0u8;
            for (bit, value) in chunk.iter().enumerate() {
                if *value > 0.0 {
                    byte |= 1 << bit;
                }
            }
            codes.push(byte);
        }
        let b = stats.norm / alpha;
        let e = stats.norm * (1.0 - alpha * alpha).max(0.0).sqrt() / alpha;
        let row = [stats.additive as f32, b as f32, e as f32];
        if row.iter().any(|value| !value.is_finite()) {
            return Err(Error::invariant("vector v5 rabitq factors are not finite"));
        }
        factors.extend_from_slice(&row);
        Ok(())
    }
}

/// The blocks of one page, before placement in the object.
#[derive(Debug, Default)]
pub(crate) struct PageBlocks {
    center_scale: f32,
    codes: Vec<u8>,
    factors: Vec<f32>,
    ids: Vec<u8>,
    filters: Option<Vec<u8>>,
}

impl PageBlocks {
    pub(crate) fn logical_bytes(&self) -> usize {
        self.codes
            .capacity()
            .saturating_add(self.factors.capacity().saturating_mul(4))
            .saturating_add(self.ids.capacity())
            .saturating_add(self.filters.as_ref().map_or(0, Vec::capacity))
    }
}

/// Codes one page: create with the page's center, push its rows in page
/// order, then [`finish`](Self::finish).
#[derive(Debug)]
pub(crate) struct PageEncoder {
    blocks: PageBlocks,
    center: Vec<f32>,
}

impl PageEncoder {
    /// `centroid` is the dequantized leaf centroid and `center_scale` the
    /// page's [`CenterScale`].
    pub(crate) fn new(
        encoder: &Encoder,
        centroid_codes: &[i8],
        centroid_scale: f32,
        center_scale: f32,
        rows: usize,
    ) -> Self {
        let mut center = Vec::with_capacity(centroid_codes.len());
        leaf_center(centroid_codes, centroid_scale, center_scale, &mut center);
        Self {
            blocks: PageBlocks {
                center_scale,
                codes: Vec::with_capacity(rows.saturating_mul(encoder.code_bytes())),
                factors: Vec::with_capacity(rows.saturating_mul(FACTORS_PER_ROW)),
                ids: Vec::with_capacity(rows.saturating_mul(ID_BYTES)),
                filters: None,
            },
            center,
        }
    }

    pub(crate) fn push(
        &mut self,
        encoder: &mut Encoder,
        id: &[u8; 16],
        vector: &[f32],
    ) -> Result<(), Error> {
        encoder.encode_row(
            vector,
            &self.center,
            &mut self.blocks.codes,
            &mut self.blocks.factors,
        )?;
        self.blocks.ids.extend_from_slice(id);
        Ok(())
    }

    pub(crate) fn logical_bytes(&self) -> usize {
        self.blocks
            .logical_bytes()
            .saturating_add(self.center.capacity().saturating_mul(4))
    }

    /// `filters` is the page's serialized filter bitmaps when the base
    /// advertises filter properties.
    pub(crate) fn finish(mut self, filters: Option<Vec<u8>>) -> PageBlocks {
        self.blocks.filters = filters;
        self.blocks
    }
}

/// Serialized bitmaps of one page, the same map its navigation page holds.
pub(crate) fn serialize_filters(
    filters: &BTreeMap<String, BTreeMap<String, Vec<u64>>>,
) -> Result<Vec<u8>, Error> {
    serialize(filters, "rabitq filter block")
}

// ---------------------------------------------------------------------------
// Section layout
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SectionBlock {
    offset: u64,
    len: u32,
    crc32: u32,
}

impl SectionBlock {
    fn range(&self) -> Range<u64> {
        self.offset..self.offset + self.len as u64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PageEntry {
    center_scale: f32,
    codes: SectionBlock,
    ids: SectionBlock,
    /// Empty (`len == 0`) when the base has no filter properties.
    filters: SectionBlock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Directory {
    version: u16,
    dim: u32,
    padded_dim: u32,
    bits: u8,
    rotation_seed: u64,
    rotation_rounds: u8,
    metric: String,
    pages: Vec<PageEntry>,
}

/// Places page blocks contiguously (offsets relative to the section start)
/// and writes the directory and locator after them.
#[derive(Debug, Default)]
pub(crate) struct SectionBuilder {
    pages: Vec<PageEntry>,
    written: u64,
}

impl SectionBuilder {
    pub(crate) fn push_page<W: Write>(
        &mut self,
        out: &mut W,
        blocks: &PageBlocks,
    ) -> Result<(), Error> {
        let mut code_block =
            Vec::with_capacity(blocks.codes.len() + blocks.factors.len() * size_of::<f32>());
        code_block.extend_from_slice(&blocks.codes);
        for factor in &blocks.factors {
            code_block.extend_from_slice(&factor.to_le_bytes());
        }
        let codes = self.append(out, &code_block)?;
        let ids = self.append(out, &blocks.ids)?;
        let filters = match &blocks.filters {
            Some(bytes) => self.append(out, bytes)?,
            None => SectionBlock {
                offset: self.written,
                len: 0,
                crc32: 0,
            },
        };
        self.pages.push(PageEntry {
            center_scale: blocks.center_scale,
            codes,
            ids,
            filters,
        });
        Ok(())
    }

    fn append<W: Write>(&mut self, out: &mut W, bytes: &[u8]) -> Result<SectionBlock, Error> {
        let len = u32::try_from(bytes.len())
            .map_err(|_| Error::invariant("vector v5 rabitq block exceeds u32"))?;
        out.write_all(bytes)?;
        let block = SectionBlock {
            offset: self.written,
            len,
            crc32: crc32fast::hash(bytes),
        };
        self.written = self
            .written
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::invariant("vector v5 rabitq section overflows u64"))?;
        Ok(block)
    }

    /// Bytes of page blocks pushed so far.
    pub(crate) fn blocks_len(&self) -> u64 {
        self.written
    }

    /// Resident size of the directory being accumulated.
    pub(crate) fn metadata_bytes(&self) -> usize {
        self.pages.capacity() * size_of::<PageEntry>()
    }

    /// Write the directory and locator. The page blocks were written at
    /// `section_offset` of the object, which `out` continues right after
    /// them. Returns the bytes written by this call.
    pub(crate) fn finish<W: Write>(
        mut self,
        out: &mut W,
        section_offset: u64,
        encoder: &Encoder,
        metric_name: &str,
    ) -> Result<u64, Error> {
        for page in &mut self.pages {
            for block in [&mut page.codes, &mut page.ids, &mut page.filters] {
                block.offset = block
                    .offset
                    .checked_add(section_offset)
                    .ok_or_else(|| Error::invariant("vector v5 rabitq offset overflows"))?;
            }
        }
        let directory_offset = section_offset
            .checked_add(self.written)
            .ok_or_else(|| Error::invariant("vector v5 rabitq offset overflows"))?;
        let directory = Directory {
            version: SECTION_VERSION,
            dim: u32::try_from(encoder.rotation.dim)
                .map_err(|_| Error::invariant("vector v5 rabitq dimension exceeds u32"))?,
            padded_dim: u32::try_from(encoder.rotation.padded)
                .map_err(|_| Error::invariant("vector v5 rabitq dimension exceeds u32"))?,
            bits: CODE_BITS,
            rotation_seed: ROTATION_SEED,
            rotation_rounds: ROTATION_ROUNDS,
            metric: metric_name.to_string(),
            pages: self.pages,
        };
        let bytes = serialize(&directory, "rabitq directory")?;
        if bytes.len() as u64 > MAX_DIRECTORY_BYTES {
            return Err(Error::invariant("vector v5 rabitq directory is too large"));
        }
        out.write_all(&bytes)?;
        out.write_all(&locator(
            directory_offset,
            bytes.len() as u64,
            crc32fast::hash(&bytes),
        ))?;
        Ok(bytes.len() as u64 + LOCATOR_LEN as u64)
    }
}

fn locator(directory_offset: u64, directory_len: u64, directory_crc: u32) -> [u8; LOCATOR_LEN] {
    let mut out = [0u8; LOCATOR_LEN];
    out[..8].copy_from_slice(LOCATOR_MAGIC);
    out[8..16].copy_from_slice(&directory_offset.to_le_bytes());
    out[16..24].copy_from_slice(&directory_len.to_le_bytes());
    out[24..28].copy_from_slice(&directory_crc.to_le_bytes());
    let crc = crc32fast::hash(&out[..28]);
    out[28..32].copy_from_slice(&crc.to_le_bytes());
    out
}

/// `(directory offset, length, crc32)` of a well-formed locator.
pub(crate) fn parse_locator(bytes: &[u8]) -> Option<(u64, u64, u32)> {
    if bytes.len() != LOCATOR_LEN || &bytes[..8] != LOCATOR_MAGIC {
        return None;
    }
    let crc = u32::from_le_bytes(bytes[28..32].try_into().ok()?);
    if crc32fast::hash(&bytes[..28]) != crc {
        return None;
    }
    Some((
        u64::from_le_bytes(bytes[8..16].try_into().ok()?),
        u64::from_le_bytes(bytes[16..24].try_into().ok()?),
        u32::from_le_bytes(bytes[24..28].try_into().ok()?),
    ))
}

/// Where a reader looks for the section: the locator range just below the
/// footer, if the gap after the last page block can hold one.
pub(crate) fn locator_range(footer: &Footer, footer_offset: u64) -> Option<Range<u64>> {
    let data_end = page_data_end(footer);
    let start = footer_offset.checked_sub(LOCATOR_LEN as u64)?;
    (start >= data_end).then_some(start..footer_offset)
}

/// End of the last navigation/exact block the footer names.
fn page_data_end(footer: &Footer) -> u64 {
    footer
        .pages
        .iter()
        .flat_map(|page| [&page.nav, &page.exact])
        .map(|block| block.offset.saturating_add(block.len as u64))
        .max()
        .unwrap_or(super::MAGIC_V5.len() as u64)
}

/// Directory range named by a locator, if it lies between the page data and
/// the locator itself.
pub(crate) fn directory_range(
    footer: &Footer,
    footer_offset: u64,
    directory_offset: u64,
    directory_len: u64,
) -> Option<Range<u64>> {
    let locator_start = footer_offset.checked_sub(LOCATOR_LEN as u64)?;
    let end = directory_offset.checked_add(directory_len)?;
    (directory_len > 0
        && directory_len <= MAX_DIRECTORY_BYTES
        && directory_offset >= page_data_end(footer)
        && end == locator_start)
        .then_some(directory_offset..end)
}

/// A validated section, retained by the reader.
#[derive(Debug)]
pub(crate) struct RabitqIndex {
    directory: Directory,
    rotation: Rotation,
    /// Footer node of each page's leaf.
    page_leaf: Vec<u32>,
    code_bytes: usize,
}

impl RabitqIndex {
    /// Decode and validate a directory against its footer. Any inconsistency
    /// is an error the caller turns into "no section" (the int8 path).
    pub(crate) fn open(
        footer: &Footer,
        metric: VectorMetric,
        directory_bytes: &[u8],
        directory_offset: u64,
    ) -> Result<Self, Error> {
        let directory: Directory = deserialize(
            directory_bytes,
            directory_bytes.len() as u64,
            "rabitq directory",
        )?;
        if directory.version != SECTION_VERSION || directory.bits != CODE_BITS {
            return Err(Error::invariant(
                "vector v5 rabitq section version unsupported",
            ));
        }
        if directory.dim != footer.dim || directory.metric != super::metric_name(metric) {
            return Err(Error::invariant(
                "vector v5 rabitq section disagrees with footer",
            ));
        }
        let rotation = Rotation::new(
            directory.dim as usize,
            directory.rotation_seed,
            directory.rotation_rounds,
        )?;
        if rotation.padded != directory.padded_dim as usize
            || directory.pages.len() != footer.pages.len()
        {
            return Err(Error::invariant("vector v5 rabitq section shape mismatch"));
        }
        let code_bytes = code_bytes(&rotation);
        let filtered = !footer.filter_properties.is_empty();
        let mut previous_end = page_data_end(footer);
        for (entry, page) in directory.pages.iter().zip(&footer.pages) {
            let rows = page.row_count as u64;
            let expected_codes = rows * (code_bytes + FACTOR_BYTES) as u64;
            if entry.codes.len as u64 != expected_codes
                || entry.ids.len as u64 != rows * ID_BYTES as u64
                || (entry.filters.len == 0) == filtered
                || !entry.center_scale.is_finite()
            {
                return Err(Error::invariant("vector v5 rabitq page entry is invalid"));
            }
            for block in [&entry.codes, &entry.ids, &entry.filters] {
                let range = block.range();
                if range.start < previous_end || range.end > directory_offset {
                    return Err(Error::invariant(
                        "vector v5 rabitq blocks overlap or leave the section",
                    ));
                }
                previous_end = range.end;
            }
        }
        let mut page_leaf = vec![u32::MAX; footer.pages.len()];
        for (index, node) in footer.nodes.iter().enumerate() {
            if let Some(page) = node.page {
                page_leaf[page as usize] = index as u32;
            }
        }
        if page_leaf.contains(&u32::MAX) {
            return Err(Error::invariant("vector v5 rabitq page has no leaf"));
        }
        Ok(Self {
            directory,
            rotation,
            page_leaf,
            code_bytes,
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.directory.pages.capacity() * size_of::<PageEntry>()
            + self.page_leaf.capacity() * size_of::<u32>()
            + self.rotation.resident_bytes()
            + size_of::<Self>()
    }

    pub(crate) fn padded_dim(&self) -> usize {
        self.rotation.padded
    }

    pub(crate) fn code_block_bytes(&self) -> u64 {
        self.directory
            .pages
            .iter()
            .map(|page| page.codes.len as u64)
            .sum()
    }

    pub(crate) fn section_bytes(&self, footer_offset: u64) -> u64 {
        let start = self
            .directory
            .pages
            .first()
            .map_or(footer_offset, |page| page.codes.offset);
        footer_offset.saturating_sub(start)
    }

    fn center(&self, footer: &Footer, page: usize, out: &mut Vec<f32>) {
        let node = &footer.nodes[self.page_leaf[page] as usize];
        leaf_center(
            &node.codes,
            node.scale,
            self.directory.pages[page].center_scale,
            out,
        );
    }
}

// ---------------------------------------------------------------------------
// Backfill registry
// ---------------------------------------------------------------------------

/// Bounded set of base objects this process opened without a RaBitQ
/// section, keyed by object file name (it embeds the SST id, so it is
/// unique and never reused, and every read path knows it: the Search-LSM
/// path from the descriptor, the reader cache from its key). The compaction
/// planner treats such a base as due for its (memory-gated, one-at-a-time)
/// rebuild, which writes the section; the old base serves through the int8
/// path until the new one installs.
const MAX_TRACKED_BASES: usize = 65_536;

fn bases_without_section() -> &'static Mutex<HashSet<String>> {
    static SET: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Last path segment of an object path or cache key.
fn object_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Record that the base at `path` (an object path or a cache key ending in
/// it) has no section.
pub(crate) fn note_base_without_section(path: &str) {
    let name = object_name(path);
    let mut set = bases_without_section()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if set.contains(name) {
        return;
    }
    if set.len() >= MAX_TRACKED_BASES {
        set.clear();
    }
    set.insert(name.to_owned());
    tracing::info!(
        object = name,
        "vector base has no RaBitQ section; scheduling its rebuild (int8 navigation serves until then)"
    );
}

pub(crate) fn base_lacks_section(path: &str) -> bool {
    bases_without_section()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .contains(object_name(path))
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// Per-query state: the effective query and the lookup table of the page
/// being scanned.
struct QueryScorer {
    metric: VectorMetric,
    query: Vec<f32>,
    residual: Vec<f32>,
    rotated: Vec<f32>,
    scratch: Vec<f32>,
    /// `code_bytes · 256` partial sums: entry `j·256 + v` is the sum of the
    /// rotated query coordinates `8j + t` whose bit `t` is set in `v`.
    lut: Vec<f32>,
    /// Per page: `<c, q>` (inner products) or `|q - c|²` (Euclidean).
    base: f32,
    sum_rotated: f32,
    inv_sqrt_d: f32,
    /// `ε0 · |q - c| / sqrt(D - 1)`, doubled for squared distances.
    bound_scale: f32,
}

impl QueryScorer {
    fn new(metric: VectorMetric, query: &[f32], padded: usize) -> Self {
        let mut effective = Vec::with_capacity(query.len());
        effective_into(metric, query, &mut effective);
        Self {
            metric,
            query: effective,
            residual: Vec::with_capacity(query.len()),
            rotated: vec![0.0; padded],
            scratch: vec![0.0; padded],
            lut: vec![0.0; padded / 8 * 256],
            base: 0.0,
            sum_rotated: 0.0,
            inv_sqrt_d: 1.0 / (padded as f32).sqrt(),
            bound_scale: 0.0,
        }
    }

    fn workspace_bytes(dim: usize, padded: usize) -> usize {
        (dim * 2 + padded * 2 + padded / 8 * 256) * size_of::<f32>()
    }

    fn prepare_page(&mut self, rotation: &Rotation, center: &[f32]) {
        self.residual.clear();
        self.residual
            .extend(self.query.iter().zip(center).map(|(q, c)| *q - *c));
        let residual_sq = self
            .residual
            .iter()
            .map(|v| *v as f64 * *v as f64)
            .sum::<f64>();
        self.base = if self.metric == VectorMetric::Euclidean {
            residual_sq as f32
        } else {
            center
                .iter()
                .zip(&self.query)
                .map(|(c, q)| *c as f64 * *q as f64)
                .sum::<f64>() as f32
        };
        let padded = rotation.padded as f32;
        let mut scale = epsilon() * residual_sq.sqrt() as f32 / (padded - 1.0).max(1.0).sqrt();
        if self.metric == VectorMetric::Euclidean {
            scale *= 2.0;
        }
        self.bound_scale = scale;
        rotation.apply(&self.residual, &mut self.rotated, &mut self.scratch);
        self.sum_rotated = self.rotated.iter().sum();
        for (table, coords) in self
            .lut
            .chunks_exact_mut(256)
            .zip(self.rotated.chunks_exact(8))
        {
            table[0] = 0.0;
            for value in 1..256usize {
                let low = value.trailing_zeros() as usize;
                table[value] = table[value & (value - 1)] + coords[low];
            }
        }
    }

    /// `<ō, P(q - c)>` of one code.
    #[inline]
    fn code_dot(&self, code: &[u8]) -> f32 {
        let mut sums = [0.0f32; 4];
        let mut chunks = code.chunks_exact(4);
        let mut base = 0usize;
        for chunk in &mut chunks {
            for lane in 0..4 {
                sums[lane] += self.lut[(base + lane) * 256 + chunk[lane] as usize];
            }
            base += 4;
        }
        let mut total = sums[0] + sums[1] + sums[2] + sums[3];
        for (offset, byte) in chunks.remainder().iter().enumerate() {
            total += self.lut[(base + offset) * 256 + *byte as usize];
        }
        (2.0 * total - self.sum_rotated) * self.inv_sqrt_d
    }

    /// `(estimate, bound)` of the row's quality (higher is better: inner
    /// product, or the negated squared distance).
    #[inline]
    fn estimate(&self, code: &[u8], factors: [f32; 3]) -> (f32, f32) {
        let [additive, b, e] = factors;
        let inner = b * self.code_dot(code);
        let estimate = if self.metric == VectorMetric::Euclidean {
            -(additive + self.base - 2.0 * inner)
        } else {
            self.base + additive + inner
        };
        (estimate, e * self.bound_scale)
    }
}

fn read_factors(block: &[u8], offset: usize) -> [f32; 3] {
    let value = |index: usize| {
        let start = offset + index * 4;
        f32::from_le_bytes([
            block[start],
            block[start + 1],
            block[start + 2],
            block[start + 3],
        ])
    };
    [value(0), value(1), value(2)]
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    page: u32,
    row: u32,
    estimate: f32,
    upper: f32,
    /// The row's code factors, to check the exact row read for it.
    factors: [f32; 3],
}

/// `f32` ordered by `total_cmp`, smallest at the heap root.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MinF32(f32);

impl Eq for MinF32 {}

impl PartialOrd for MinF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0)
    }
}

/// Bound-driven candidate set: rows whose upper bound reaches the `k`-th
/// largest lower bound seen, at most `cap` of them (highest estimates).
struct Selector {
    k: usize,
    cap: usize,
    lower: BinaryHeap<MinF32>,
    threshold: f32,
    candidates: Vec<Candidate>,
}

impl Selector {
    fn new(k: usize, cap: usize) -> Self {
        Self {
            k,
            cap,
            lower: BinaryHeap::with_capacity(k + 1),
            threshold: f32::NEG_INFINITY,
            candidates: Vec::with_capacity(cap.min(4096)),
        }
    }

    fn workspace_bytes(k: usize, cap: usize) -> usize {
        (k + 1) * size_of::<MinF32>() + cap.saturating_mul(4) * size_of::<Candidate>()
    }

    #[inline]
    fn offer(&mut self, page: u32, row: u32, estimate: f32, bound: f32, factors: [f32; 3]) {
        let lower = estimate - bound;
        let upper = estimate + bound;
        if self.lower.len() < self.k {
            self.lower.push(MinF32(lower));
            if self.lower.len() == self.k {
                self.threshold = self.lower.peek().map_or(f32::NEG_INFINITY, |v| v.0);
            }
        } else if lower > self.threshold {
            self.lower.pop();
            self.lower.push(MinF32(lower));
            self.threshold = self.lower.peek().map_or(f32::NEG_INFINITY, |v| v.0);
        }
        if upper < self.threshold {
            return;
        }
        self.candidates.push(Candidate {
            page,
            row,
            estimate,
            upper,
            factors,
        });
        if self.candidates.len() >= self.cap.saturating_mul(4).max(64) {
            self.prune(self.cap.saturating_mul(2));
        }
    }

    fn prune(&mut self, keep: usize) {
        let threshold = self.threshold;
        self.candidates
            .retain(|candidate| candidate.upper >= threshold);
        if self.candidates.len() > keep {
            self.candidates
                .select_nth_unstable_by(keep, |a, b| b.estimate.total_cmp(&a.estimate));
            self.candidates.truncate(keep);
        }
    }

    /// Final candidates, ordered by `(page, row)`.
    fn finish(mut self) -> Vec<Candidate> {
        let cap = self.cap;
        self.prune(cap);
        self.candidates
            .sort_unstable_by(|a, b| a.page.cmp(&b.page).then(a.row.cmp(&b.row)));
        self.candidates
    }
}

/// The `k`-th best exact quality seen by a re-rank, in the estimator's
/// units (see [`estimated_quality`]).
#[derive(Debug)]
struct KthQuality {
    k: usize,
    best: BinaryHeap<MinF32>,
}

impl KthQuality {
    fn new(k: usize) -> Self {
        Self {
            k,
            best: BinaryHeap::with_capacity(k + 1),
        }
    }

    fn offer(&mut self, quality: f32) {
        if self.k == 0 || !quality.is_finite() {
            return;
        }
        if self.best.len() < self.k {
            self.best.push(MinF32(quality));
        } else if self.best.peek().is_some_and(|worst| quality > worst.0) {
            self.best.pop();
            self.best.push(MinF32(quality));
        }
    }

    /// `None` until `k` rows were scored.
    fn value(&self) -> Option<f32> {
        (self.best.len() == self.k)
            .then(|| self.best.peek().map(|worst| worst.0))
            .flatten()
    }
}

/// The exact value [`QueryScorer::estimate`] estimates for a row: the inner
/// product of the effective (for cosine, normalized) row and query, or the
/// negated squared distance.
fn estimated_quality(metric: VectorMetric, row: &[f32], query: &[f32]) -> f32 {
    match metric {
        VectorMetric::Euclidean => {
            -(row
                .iter()
                .zip(query)
                .map(|(x, q)| {
                    let d = *x as f64 - *q as f64;
                    d * d
                })
                .sum::<f64>() as f32)
        }
        VectorMetric::Cosine => {
            let (mut dot, mut xx, mut qq) = (0.0f64, 0.0f64, 0.0f64);
            for (x, q) in row.iter().zip(query) {
                let (x, q) = (*x as f64, *q as f64);
                dot += x * q;
                xx += x * x;
                qq += q * q;
            }
            if xx > 0.0 && qq > 0.0 {
                (dot / (xx.sqrt() * qq.sqrt())) as f32
            } else {
                0.0
            }
        }
        _ => row
            .iter()
            .zip(query)
            .map(|(x, q)| *x as f64 * *q as f64)
            .sum::<f64>() as f32,
    }
}

/// Most candidate pages one re-rank wave reads concurrently. Row reads of
/// raw-block pages are small; the range source bounds in-flight requests.
const MAX_RERANK_WAVE_PAGES: usize = 256;

/// Batch sizes for one RaBitQ query and the workspace they reserve.
struct RabitqPlan {
    code_batch_pages: usize,
    exact_batch_pages: usize,
    required_bytes: usize,
}

fn rabitq_plan(
    footer: &Footer,
    index: &RabitqIndex,
    final_k: usize,
    cap: usize,
    groups: &[(String, Vec<Value>)],
    filtered: bool,
) -> RabitqPlan {
    let dim = footer.dim as usize;
    let group_bytes = groups
        .iter()
        .map(|(property, values)| property.len() + values.len() * 64 + 64)
        .sum::<usize>();
    let base = QueryScorer::workspace_bytes(dim, index.rotation.padded)
        + Selector::workspace_bytes(final_k, cap)
        + final_k * 32
        + footer.pages.len() * (size_of::<(u32, f64)>() + size_of::<u32>())
        + group_bytes
        + dim * size_of::<f32>() * 4;
    let page_read = |page: &PageEntry| {
        let mut bytes = page.codes.len as usize;
        if filtered {
            bytes += page.filters.len as usize * (1 + NAV_DECODE_WORKSPACE_FACTOR);
        }
        bytes
    };
    let mut reads: Vec<usize> = index.directory.pages.iter().map(page_read).collect();
    reads.sort_unstable_by(|a, b| b.cmp(a));
    let exact_page_peak = footer
        .pages
        .iter()
        .zip(&index.directory.pages)
        .map(|(page, entry)| {
            (page.exact.len as usize)
                .saturating_add(page.exact.raw_len as usize)
                .saturating_add(entry.ids.len as usize)
                .saturating_add(dim * size_of::<f32>())
        })
        .max()
        .unwrap_or(0);
    let capacity = shared_search_workspace().capacity_bytes();
    let mut code_batch_pages = 1;
    let mut code_peak = reads.first().copied().unwrap_or(0);
    let mut sum = 0usize;
    for (count, bytes) in reads.iter().take(MAX_NAV_BATCH_PAGES).enumerate() {
        sum = sum.saturating_add(*bytes);
        if base.saturating_add(sum.max(exact_page_peak)) <= capacity {
            code_batch_pages = count + 1;
            code_peak = sum;
        } else {
            break;
        }
    }
    let mut exact_batch_pages = 1;
    for wave in (2..=MAX_EXACT_BATCH_PAGES).rev() {
        if base.saturating_add(code_peak.max(exact_page_peak.saturating_mul(wave))) <= capacity {
            exact_batch_pages = wave;
            break;
        }
    }
    RabitqPlan {
        code_batch_pages,
        exact_batch_pages,
        required_bytes: base
            .saturating_add(code_peak.max(exact_page_peak.saturating_mul(exact_batch_pages))),
    }
}

fn verify_block<'a>(wire: &'a Bytes, block: &SectionBlock, label: &str) -> Result<&'a [u8], Error> {
    require_len(wire, block.len as usize, label)?;
    if crc32fast::hash(wire) != block.crc32 {
        return Err(Error::invariant(format!(
            "vector v5 rabitq {label} checksum mismatch"
        )));
    }
    Ok(wire.as_ref())
}

fn decode_filters(
    wire: &Bytes,
    block: &SectionBlock,
    rows: usize,
) -> Result<BTreeMap<String, BTreeMap<String, Vec<u64>>>, Error> {
    let bytes = verify_block(wire, block, "filter block")?;
    let filters: BTreeMap<String, BTreeMap<String, Vec<u64>>> =
        deserialize(bytes, bytes.len() as u64, "rabitq filter block")?;
    let words = rows.div_ceil(64);
    for values in filters.values() {
        for bitmap in values.values() {
            if bitmap.len() != words
                || (rows % 64 != 0
                    && bitmap
                        .last()
                        .is_some_and(|word| word & !((1u64 << (rows % 64)) - 1) != 0))
            {
                return Err(Error::invariant(
                    "vector v5 rabitq filter bitmap is malformed",
                ));
            }
        }
    }
    Ok(filters)
}

/// Whether an exact row read by range is the row a code was built from: the
/// residual statistics recomputed from it must equal the checksummed
/// factors (they are computed by the same code from the same bytes, so only
/// a row from the wrong place or damaged bytes differ).
fn exact_row_matches_code(
    metric: VectorMetric,
    row: &[f32],
    center: &[f32],
    factors: [f32; 3],
    effective: &mut Vec<f32>,
    residual: &mut Vec<f32>,
) -> bool {
    if row.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let stats = residual_into(metric, row, center, effective, residual);
    let [additive, b, e] = factors.map(|v| v as f64);
    let stored_norm = (b * b - e * e).max(0.0).sqrt();
    let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-4 * scale.max(1e-6) + 1e-6;
    close(
        stats.additive,
        additive,
        stats.additive.abs().max(stats.norm),
    ) && close(stats.norm, stored_norm, stats.norm)
}

impl VectorV5Reader {
    /// RaBitQ-navigated search. Same probe policy, widening, filters and
    /// exact final scores as the int8 path; only the candidate selection
    /// differs.
    pub(super) async fn search_rabitq(
        &self,
        index: &RabitqIndex,
        query: &[f32],
        k: usize,
        options: VectorV5SearchOptions,
        groups: &[(String, Vec<Value>)],
    ) -> Result<VectorV5SearchResult, Error> {
        let pages = self.footer.pages.len();
        let initial_nprobe = options.nprobe.max(1).min(pages);
        let max_nprobe = options.max_nprobe.max(initial_nprobe).min(pages);
        let point_count = usize::try_from(self.footer.point_count)
            .map_err(|_| Error::invariant("vector v5 point count does not fit usize"))?;
        let final_k = k.min(point_count);
        let rerank_k = k
            .saturating_mul(options.rerank_factor.max(1))
            .max(k)
            .min(point_count);
        let cap = rerank_k
            .saturating_mul(CANDIDATE_CAP_FACTOR)
            .max(final_k)
            .min(point_count);
        let supported_groups = prepare_groups(&self.footer.filter_properties, groups);
        let applied_filter_groups = supported_groups.len();
        let filtered = applied_filter_groups > 0;
        let plan = rabitq_plan(&self.footer, index, final_k, cap, groups, filtered);
        let _workspace = shared_search_workspace()
            .reserve("vector v5 rabitq search", plan.required_bytes)
            .await?;
        let leaf_order = {
            namidb_core::profile_scope!("v5.rank_leaves");
            self.rank_leaf_pages(query)?
        };

        let mut scorer = QueryScorer::new(self.metric, query, index.rotation.padded);
        let mut selector = Selector::new(final_k, cap);
        let mut center = Vec::with_capacity(self.footer.dim as usize);
        let code_bytes = index.code_bytes;
        let mut eligible_rows_seen = 0usize;
        let mut fetched = 0usize;
        let mut target = initial_nprobe;
        loop {
            for batch in leaf_order[fetched..target].chunks(plan.code_batch_pages.max(1)) {
                let mut ranges = Vec::with_capacity(batch.len() * 2);
                for &page in batch {
                    let entry = &index.directory.pages[page as usize];
                    ranges.push(entry.codes.range());
                    if filtered {
                        ranges.push(entry.filters.range());
                    }
                }
                let blocks = {
                    namidb_core::profile_scope!("v5.rabitq.read");
                    self.source.read_blocks(&ranges).await?
                };
                if blocks.len() != ranges.len() {
                    return Err(Error::invariant(
                        "vector v5 range source returned wrong rabitq block count",
                    ));
                }
                let per_page = if filtered { 2 } else { 1 };
                namidb_core::profile_scope!("v5.rabitq.score");
                for (&page, wires) in batch.iter().zip(blocks.chunks_exact(per_page)) {
                    let entry = &index.directory.pages[page as usize];
                    let rows = self.footer.pages[page as usize].row_count as usize;
                    let block = verify_block(&wires[0], &entry.codes, "code block")?;
                    let filters = if filtered {
                        Some(decode_filters(&wires[1], &entry.filters, rows)?)
                    } else {
                        None
                    };
                    index.center(&self.footer, page as usize, &mut center);
                    scorer.prepare_page(&index.rotation, &center);
                    let factor_base = rows * code_bytes;
                    for row in 0..rows {
                        if let Some(filters) = &filters {
                            if !row_matches_filters(row, filters, &supported_groups) {
                                continue;
                            }
                        }
                        eligible_rows_seen += 1;
                        let code = &block[row * code_bytes..(row + 1) * code_bytes];
                        let factors = read_factors(block, factor_base + row * FACTOR_BYTES);
                        let (estimate, bound) = scorer.estimate(code, factors);
                        if !estimate.is_finite() || !bound.is_finite() {
                            return Err(Error::invariant(
                                "vector v5 rabitq estimate is not finite",
                            ));
                        }
                        selector.offer(page, row as u32, estimate, bound, factors);
                    }
                }
            }
            fetched = target;
            if applied_filter_groups == 0 || eligible_rows_seen >= k || fetched >= max_nprobe {
                break;
            }
            target = fetched.saturating_mul(2).max(fetched + 1).min(max_nprobe);
        }

        let mut candidates = selector.finish();
        // Re-rank the best `rerank_k` estimates first. Their exact scores
        // give a k-th best quality far tighter than the estimates' k-th lower
        // bound, so a remaining candidate is read only when its upper bound
        // still reaches it (the same bound that admitted it). One extra round
        // trip when anything remains, and usually far fewer rows read.
        let mut exact_hits = BoundedExactHits::new(final_k);
        let mut kth = KthQuality::new(final_k);
        let mut rest = Vec::new();
        if candidates.len() > rerank_k {
            candidates.sort_unstable_by(|a, b| b.estimate.total_cmp(&a.estimate));
            rest = candidates.split_off(rerank_k);
            candidates.sort_unstable_by(|a, b| a.page.cmp(&b.page).then(a.row.cmp(&b.row)));
        }
        let mut reranked_rows = candidates.len();
        self.rerank_rabitq(
            index,
            query,
            plan.exact_batch_pages,
            &candidates,
            &mut exact_hits,
            &mut kth,
        )
        .await?;
        if let Some(threshold) = kth.value() {
            rest.retain(|candidate| candidate.upper >= threshold);
        }
        if !rest.is_empty() {
            rest.sort_unstable_by(|a, b| a.page.cmp(&b.page).then(a.row.cmp(&b.row)));
            reranked_rows += rest.len();
            self.rerank_rabitq(
                index,
                query,
                plan.exact_batch_pages,
                &rest,
                &mut exact_hits,
                &mut kth,
            )
            .await?;
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

    /// Exact f32 scores of `candidates` (sorted by page and row): rows are
    /// range read from raw-block exact pages (whole pages otherwise), NodeIds
    /// from the pages' checksummed id blocks, in one round trip per wave.
    async fn rerank_rabitq(
        &self,
        index: &RabitqIndex,
        query: &[f32],
        exact_batch_pages: usize,
        candidates: &[Candidate],
        exact_hits: &mut BoundedExactHits,
        kth: &mut KthQuality,
    ) -> Result<(), Error> {
        let mut groups = Vec::new();
        let mut start = 0usize;
        while start < candidates.len() {
            let page = candidates[start].page;
            let mut end = start + 1;
            while end < candidates.len() && candidates[end].page == page {
                end += 1;
            }
            groups.push((page, start, end));
            start = end;
        }
        let dim = self.footer.dim as usize;
        let row_bytes = dim * size_of::<f32>();
        let mut center = Vec::with_capacity(dim);
        let mut effective = Vec::with_capacity(dim);
        let mut residual = Vec::with_capacity(dim);
        let mut row = Vec::with_capacity(row_bytes);
        // Pack pages into waves by what they will actually read: candidate
        // rows of a raw-block page cost about their bytes (plus coalescing
        // gaps), a page that must be read whole costs the page. One whole
        // page more stays reserved for a page whose rows do not check out.
        let page_peak = self
            .footer
            .pages
            .iter()
            .map(|page| page.exact.len as usize + page.exact.raw_len as usize)
            .max()
            .unwrap_or(0);
        let budget = page_peak
            .saturating_mul(exact_batch_pages.max(1))
            .saturating_sub(page_peak);
        let mut waves = Vec::new();
        let mut wave_start = 0usize;
        let mut wave_cost = 0usize;
        for (position, &(page, start, end)) in groups.iter().enumerate() {
            let page_ref = &self.footer.pages[page as usize];
            let whole = page_ref.exact.len as usize + page_ref.exact.raw_len as usize;
            let cost = if self.raw_pages[page as usize].load(AtomicOrdering::Relaxed) == RAW_NO {
                whole
            } else {
                ((end - start) * (row_bytes + super::RAW_ROW_COALESCE_GAP as usize)).min(whole)
            } + index.directory.pages[page as usize].ids.len as usize;
            let pages = position - wave_start;
            if pages > 0 && (pages >= MAX_RERANK_WAVE_PAGES || wave_cost + cost > budget) {
                waves.push(wave_start..position);
                wave_start = position;
                wave_cost = 0;
            }
            wave_cost += cost;
        }
        if wave_start < groups.len() {
            waves.push(wave_start..groups.len());
        }
        for wave in waves.into_iter().map(|range| &groups[range]) {
            let id_ranges: Vec<Range<u64>> = wave
                .iter()
                .map(|(page, _, _)| index.directory.pages[*page as usize].ids.range())
                .collect();
            let mut spans = Vec::new();
            let mut plans = Vec::new();
            let mut whole = Vec::new();
            for (slot, &(page, start, end)) in wave.iter().enumerate() {
                let exact = &self.footer.pages[page as usize].exact;
                let state = self.raw_pages[page as usize].load(AtomicOrdering::Relaxed);
                if state == RAW_NO {
                    whole.push(slot);
                    continue;
                }
                // A page stored at exactly the raw-block frame length of its
                // rows is read as raw without first reading its frame header:
                // every row is checked against its code below, and a page
                // whose rows do not check out is re-read whole.
                let speculative = state == RAW_UNKNOWN
                    && exact.len as u64 == super::raw_frame_len(exact.raw_len as u64);
                let header = (state == RAW_UNKNOWN && !speculative).then(|| {
                    spans.push(raw_frame_header_range(exact));
                    spans.len() - 1
                });
                let rows = plan_raw_rows(
                    exact,
                    candidates[start..end].iter().map(|c| c.row),
                    row_bytes as u64,
                    &mut spans,
                );
                plans.push((slot, header, speculative, rows));
            }
            let whole_ranges = whole
                .iter()
                .map(|slot| self.footer.pages[wave[*slot].0 as usize].exact.range())
                .collect::<Result<Vec<_>, _>>()?;
            let (id_wires, parts, whole_wires) = {
                namidb_core::profile_scope!("v5.rabitq.rerank.read");
                futures::try_join!(
                    self.source.read_blocks(&id_ranges),
                    async {
                        if spans.is_empty() {
                            Ok(Vec::new())
                        } else {
                            self.source.read_small(&spans).await
                        }
                    },
                    async {
                        if whole_ranges.is_empty() {
                            Ok(Vec::new())
                        } else {
                            self.source.read_ranges(&whole_ranges).await
                        }
                    },
                )?
            };
            if id_wires.len() != id_ranges.len()
                || parts.len() != spans.len()
                || whole_wires.len() != whole_ranges.len()
            {
                return Err(Error::invariant(
                    "vector v5 range source returned wrong rabitq rerank count",
                ));
            }
            let mut ids = Vec::with_capacity(wave.len());
            for ((page, _, _), wire) in wave.iter().zip(&id_wires) {
                ids.push(verify_block(
                    wire,
                    &index.directory.pages[*page as usize].ids,
                    "id block",
                )?);
            }
            let id_of = |slot: usize, row: u32| -> Result<[u8; 16], Error> {
                let start = row as usize * ID_BYTES;
                ids[slot]
                    .get(start..start + ID_BYTES)
                    .and_then(|bytes| bytes.try_into().ok())
                    .ok_or_else(|| Error::invariant("vector v5 rabitq id out of range"))
            };
            namidb_core::profile_scope!("v5.rabitq.rerank.score");
            // Pages whose rows did not check out are re-read whole.
            let mut retry = Vec::new();
            for (slot, header, speculative, pieces) in plans {
                let (page, start, end) = wave[slot];
                let exact = &self.footer.pages[page as usize].exact;
                if let Some(header) = header {
                    let raw = parts[header].as_ref() == raw_frame_header(exact.raw_len);
                    self.raw_pages[page as usize]
                        .store(if raw { RAW_YES } else { RAW_NO }, AtomicOrdering::Relaxed);
                    if !raw {
                        retry.push(slot);
                        continue;
                    }
                }
                index.center(&self.footer, page as usize, &mut center);
                let mut scored = Vec::with_capacity(end - start);
                let mut consistent = true;
                for (candidate, pieces) in candidates[start..end].iter().zip(&pieces) {
                    row.clear();
                    for &(span, offset, len) in pieces {
                        match parts.get(span).and_then(|part| {
                            (part.len() as u64 == spans[span].end - spans[span].start)
                                .then(|| part.get(offset..offset + len))
                                .flatten()
                        }) {
                            Some(bytes) => row.extend_from_slice(bytes),
                            None => consistent = false,
                        }
                    }
                    if !consistent || row.len() != row_bytes {
                        consistent = false;
                        break;
                    }
                    let vector = super::f32_row(&row);
                    if !exact_row_matches_code(
                        self.metric,
                        &vector,
                        &center,
                        candidate.factors,
                        &mut effective,
                        &mut residual,
                    ) {
                        consistent = false;
                        break;
                    }
                    let score = metric_score(self.metric, &vector, query).0 as f32;
                    scored.push((
                        id_of(slot, candidate.row)?,
                        score,
                        estimated_quality(self.metric, &vector, query),
                    ));
                }
                if consistent {
                    if speculative {
                        self.raw_pages[page as usize].store(RAW_YES, AtomicOrdering::Relaxed);
                    }
                    for (id, score, exact_quality) in scored {
                        exact_hits.insert(self.metric, id, score);
                        kth.offer(exact_quality);
                    }
                } else {
                    retry.push(slot);
                }
            }
            let mut whole_pages: Vec<(usize, Bytes)> = whole.into_iter().zip(whole_wires).collect();
            if !retry.is_empty() {
                let ranges = retry
                    .iter()
                    .map(|slot| self.footer.pages[wave[*slot].0 as usize].exact.range())
                    .collect::<Result<Vec<_>, _>>()?;
                let wires = self.source.read_ranges(&ranges).await?;
                if wires.len() != ranges.len() {
                    return Err(Error::invariant(
                        "vector v5 range source returned wrong exact block count",
                    ));
                }
                whole_pages.extend(retry.into_iter().zip(wires));
            }
            for (slot, wire) in whole_pages {
                let (page, start, end) = wave[slot];
                let page_ref = &self.footer.pages[page as usize];
                let raw = super::decode_block(&wire, &page_ref.exact, "exact page")?;
                if raw.len() != page_ref.row_count as usize * row_bytes {
                    return Err(Error::invariant(
                        "vector v5 exact page length disagrees with its directory",
                    ));
                }
                for candidate in &candidates[start..end] {
                    let offset = candidate.row as usize * row_bytes;
                    let bytes = raw
                        .get(offset..offset + row_bytes)
                        .ok_or_else(|| Error::invariant("vector v5 exact row out of range"))?;
                    let vector = super::f32_row(bytes);
                    let score = metric_score(self.metric, &vector, query).0 as f32;
                    exact_hits.insert(self.metric, id_of(slot, candidate.row)?, score);
                    kth.offer(estimated_quality(self.metric, &vector, query));
                }
            }
        }
        Ok(())
    }
}

/// Blocks per `prime_blocks` call while prefetching the resident tier: one
/// call covers a few 8 MiB spans, which bounds the bytes held at once.
const PREFETCH_BLOCKS_PER_CALL: usize = 2048;

/// Outcome of [`VectorV5Reader::prefetch_resident_codes`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VectorV5CodePrefetch {
    /// The base carries a valid RaBitQ section (otherwise nothing is read).
    pub section_present: bool,
    /// Bytes of all code blocks (1-bit codes plus per-row factors).
    pub code_bytes: u64,
    /// Bytes the warm-up reads and compares with its budget: the code
    /// blocks plus the id blocks (and filter blocks, when present).
    pub block_bytes: u64,
    /// Bytes read through the source by this call.
    pub loaded_bytes: u64,
    /// `block_bytes` exceeded `max_bytes`, so nothing was read.
    pub skipped: bool,
}

impl VectorV5Reader {
    /// Warm the resident tier: every page's code block (1-bit codes and
    /// factors) and id block (the NodeIds a re-rank reports), plus its filter
    /// block when the base has filter properties. They are read through
    /// [`VectorV5RangeSource::prime_blocks`]: the section stores them
    /// contiguously, so a caching source fetches them with a few large
    /// requests and admits each as the exact entry a search later reads.
    ///
    /// `max_bytes` is the caller's share of the cache for this base (for
    /// example half the RAM page cache). When the blocks are larger than
    /// that, nothing is read and `skipped` is set: a partial warm-up would
    /// only displace other entries. Blocks are CRC-checked. Bases without the
    /// section return immediately.
    pub async fn prefetch_resident_codes(
        &self,
        max_bytes: u64,
    ) -> Result<VectorV5CodePrefetch, Error> {
        let RabitqState::Present(index) = &self.rabitq else {
            return Ok(VectorV5CodePrefetch::default());
        };
        let mut report = VectorV5CodePrefetch {
            section_present: true,
            code_bytes: index.code_block_bytes(),
            ..VectorV5CodePrefetch::default()
        };
        let mut blocks: Vec<(&SectionBlock, &'static str)> =
            Vec::with_capacity(index.directory.pages.len() * 3);
        for entry in &index.directory.pages {
            blocks.push((&entry.codes, "code block"));
            blocks.push((&entry.ids, "id block"));
            if entry.filters.len > 0 {
                blocks.push((&entry.filters, "filter block"));
            }
        }
        report.block_bytes = blocks
            .iter()
            .map(|(block, _)| block.len as u64)
            .sum::<u64>();
        if report.block_bytes > max_bytes {
            report.skipped = true;
            return Ok(report);
        }
        blocks.sort_by_key(|(block, _)| block.offset);
        for chunk in blocks.chunks(PREFETCH_BLOCKS_PER_CALL) {
            let ranges: Vec<Range<u64>> = chunk.iter().map(|(block, _)| block.range()).collect();
            let wires = self.source.prime_blocks(&ranges).await?;
            if wires.len() != ranges.len() {
                return Err(Error::invariant(
                    "vector v5 range source returned wrong rabitq block count",
                ));
            }
            for ((block, label), wire) in chunk.iter().zip(&wires) {
                verify_block(wire, block, label)?;
                report.loaded_bytes = report.loaded_bytes.saturating_add(wire.len() as u64);
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
impl VectorV5Reader {
    /// Absolute range of page `page`'s code block, when the section loaded.
    pub(crate) fn rabitq_code_block_range(&self, page: usize) -> Option<Range<u64>> {
        match &self.rabitq {
            RabitqState::Present(index) => index.directory.pages.get(page).map(|p| p.codes.range()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use uuid::Uuid;

    use super::super::{build_body_v5, build_body_v5_with_filter_postings, VectorV5BuildOptions};
    use super::*;
    use crate::manifest::{VectorIndexDescriptor, VectorQuantization};
    use crate::sst::vector::v5::VectorV5RangeSource;
    use crate::sst::vector::{VectorFilterPosting, VectorFilterPostings};

    #[derive(Debug)]
    struct MemorySource(Bytes);

    #[async_trait::async_trait]
    impl VectorV5RangeSource for MemorySource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error> {
            if range.start > range.end || range.end > self.0.len() as u64 {
                return Err(Error::invariant("test range out of bounds"));
            }
            Ok(self.0.slice(range.start as usize..range.end as usize))
        }
    }

    fn descriptor(metric: VectorMetric, dim: u32) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: format!("rabitq-{metric:?}"),
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
        id[12..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn gaussian(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|_| {
                // Box-Muller.
                let u: f64 = rng.gen_range(1e-12..1.0);
                let v: f64 = rng.gen_range(0.0..1.0);
                ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
            })
            .collect()
    }

    fn normalized(mut vector: Vec<f32>) -> Vec<f32> {
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter_mut().for_each(|v| *v /= norm);
        }
        vector
    }

    /// Clustered unit vectors with anisotropic noise and near-duplicates.
    fn clustered(count: u32, dim: usize, clusters: usize, seed: u64) -> Vec<([u8; 16], Vec<f32>)> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| normalized(gaussian(&mut rng, dim)))
            .collect();
        (0..count)
            .map(|ordinal| {
                let center = &centers[rng.gen_range(0..clusters)];
                let noise = gaussian(&mut rng, dim);
                let spread = rng.gen_range(0.2..1.2f32);
                let vector: Vec<f32> = center
                    .iter()
                    .zip(noise)
                    .enumerate()
                    .map(|(axis, (c, n))| {
                        c + n * spread * (1.0 + axis as f32 / dim as f32) / (dim as f32).sqrt()
                    })
                    .collect();
                (id(ordinal), normalized(vector))
            })
            .collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    #[test]
    fn rotation_is_orthogonal_deterministic_and_pads_to_64() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for dim in [1usize, 7, 100, 384, 1024, 1536] {
            let rotation = Rotation::new(dim, ROTATION_SEED, ROTATION_ROUNDS).unwrap();
            assert_eq!(rotation.padded_dim(), dim.next_multiple_of(64));
            assert_eq!(rotation.padded_dim() % rotation.block, 0);
            assert!(rotation.block.is_power_of_two() && rotation.block >= 64);
            let again = Rotation::new(dim, ROTATION_SEED, ROTATION_ROUNDS).unwrap();
            let a = gaussian(&mut rng, dim);
            let b = gaussian(&mut rng, dim);
            let padded = rotation.padded_dim();
            let (mut ra, mut rb, mut rc, mut scratch) = (
                vec![0.0; padded],
                vec![0.0; padded],
                vec![0.0; padded],
                vec![0.0; padded],
            );
            rotation.apply(&a, &mut ra, &mut scratch);
            rotation.apply(&b, &mut rb, &mut scratch);
            again.apply(&a, &mut rc, &mut scratch);
            assert_eq!(ra, rc, "same seed, same rotation");
            let (na, nra) = (dot(&a, &a), dot(&ra, &ra));
            assert!((na - nra).abs() <= 1e-4 * na, "norm preserved at {dim}");
            let (ab, rab) = (dot(&a, &b), dot(&ra, &rb));
            assert!((ab - rab).abs() <= 1e-4 * na.sqrt() * dot(&b, &b).sqrt());
            let other = Rotation::new(dim, ROTATION_SEED + 1, ROTATION_ROUNDS).unwrap();
            other.apply(&a, &mut rc, &mut scratch);
            assert_ne!(ra, rc, "another seed rotates differently");
        }
    }

    /// One coded row against one query: `(estimate - exact, bound at ε0)`.
    fn errors(metric: VectorMetric, dim: usize, pairs: usize, seed: u64) -> Vec<(f64, f64)> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut encoder = Encoder::new(metric, dim).unwrap();
        let padded = encoder.rotation.padded_dim();
        let mut scorer_center = Vec::new();
        let mut out = Vec::with_capacity(pairs);
        for _ in 0..pairs {
            let center: Vec<f32> = gaussian(&mut rng, dim).iter().map(|v| v * 0.3).collect();
            let x: Vec<f32> = center
                .iter()
                .zip(gaussian(&mut rng, dim))
                .map(|(c, n)| c + n * 0.2)
                .collect();
            let q: Vec<f32> = center
                .iter()
                .zip(gaussian(&mut rng, dim))
                .map(|(c, n)| c + n * 0.25)
                .collect();
            // The cosine path codes the normalized row, so its center lives
            // in the normalized space too (a leaf centroid of unit rows,
            // scaled by its CenterScale).
            let center = if metric == VectorMetric::Cosine {
                let norm = dot(&center, &center).sqrt() as f32;
                center.iter().map(|v| v / norm * 0.85).collect()
            } else {
                center
            };
            let (mut codes, mut factors) = (Vec::new(), Vec::new());
            encoder
                .encode_row(&x, &center, &mut codes, &mut factors)
                .unwrap();
            let mut scorer = QueryScorer::new(metric, &q, padded);
            scorer_center.clear();
            scorer_center.extend_from_slice(&center);
            scorer.prepare_page(&encoder.rotation, &scorer_center);
            let (estimate, bound) = scorer.estimate(&codes, [factors[0], factors[1], factors[2]]);
            let exact = match metric {
                VectorMetric::Dot => dot(&x, &q),
                VectorMetric::Cosine => dot(&x, &q) / (dot(&x, &x) * dot(&q, &q)).sqrt(),
                VectorMetric::Euclidean => -x
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| ((a - b) as f64).powi(2))
                    .sum::<f64>(),
            };
            out.push((estimate as f64 - exact, bound as f64));
        }
        out
    }

    #[test]
    fn estimator_is_unbiased_and_its_bound_holds_at_the_stated_rate() {
        for metric in [
            VectorMetric::Dot,
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
        ] {
            for (dim, seed) in [(128usize, 11u64), (1024, 12), (1536, 13)] {
                let samples = errors(metric, dim, 2_000, seed);
                let n = samples.len() as f64;
                // Unbiased: the mean error is within 4 standard errors of 0.
                let mean = samples.iter().map(|(e, _)| e).sum::<f64>() / n;
                let var = samples.iter().map(|(e, _)| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
                assert!(
                    mean.abs() <= 4.0 * (var / n).sqrt() + 1e-6,
                    "{metric:?}/{dim}: mean error {mean} (sd {})",
                    var.sqrt()
                );
                // The normalized error is asymptotically standard normal
                // (Gao & Long, §3.2), so the bound at ε0 fails with
                // probability about P(|Z| > ε0) (5.7% at 1.9, 0.27% at 3)
                // or less. Allow three standard errors of sampling noise
                // and one point for the finite dimension.
                let violations = |epsilon: f64| {
                    samples
                        .iter()
                        .filter(|(e, bound)| e.abs() > bound * epsilon / EPSILON0 as f64)
                        .count() as f64
                        / n
                };
                let at_default = violations(EPSILON0 as f64);
                let at_three = violations(3.0);
                let allowed = |p: f64| p + 3.0 * (p * (1.0 - p) / n).sqrt() + 0.01;
                eprintln!(
                    "{metric:?}/{dim}: bias {mean:+.2e} (sd {:.2e}), bound failures {:.2}% at ε0=1.9, {:.2}% at 3",
                    var.sqrt(),
                    at_default * 100.0,
                    at_three * 100.0
                );
                assert!(
                    at_default <= allowed(0.0574),
                    "{metric:?}/{dim}: ε0 bound failed {at_default}"
                );
                assert!(
                    at_three <= allowed(0.0027),
                    "{metric:?}/{dim}: 3σ bound failed {at_three}"
                );
                // The bound is not vacuous: errors use a good share of it.
                assert!(
                    violations(EPSILON0 as f64 / 4.0) > 0.2,
                    "{metric:?}/{dim}: bound too loose"
                );
            }
        }
    }

    #[test]
    fn the_rerank_threshold_uses_the_estimators_units() {
        // A second-wave candidate is dropped when its upper bound is below
        // the k-th exact quality, so the exact quality must be the value the
        // estimate approximates, for every metric and unnormalized inputs.
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        for metric in [
            VectorMetric::Dot,
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
        ] {
            let dim = 256;
            let mut encoder = Encoder::new(metric, dim).unwrap();
            let padded = encoder.rotation.padded_dim();
            let mut within = 0usize;
            let trials = 400;
            for _ in 0..trials {
                let x: Vec<f32> = gaussian(&mut rng, dim).iter().map(|v| v * 3.0).collect();
                let q: Vec<f32> = x
                    .iter()
                    .zip(gaussian(&mut rng, dim))
                    .map(|(x, n)| (x + n) * 0.5)
                    .collect();
                let mut center = Vec::new();
                effective_into(metric, &x, &mut center);
                for value in &mut center {
                    *value *= 0.8;
                }
                let (mut codes, mut factors) = (Vec::new(), Vec::new());
                encoder
                    .encode_row(&x, &center, &mut codes, &mut factors)
                    .unwrap();
                let mut scorer = QueryScorer::new(metric, &q, padded);
                scorer.prepare_page(&encoder.rotation, &center);
                let (estimate, bound) =
                    scorer.estimate(&codes, [factors[0], factors[1], factors[2]]);
                let exact = estimated_quality(metric, &x, &q);
                if (estimate - exact).abs() <= bound * 1.5 {
                    within += 1;
                }
            }
            assert!(
                within as f64 >= 0.9 * trials as f64,
                "{metric:?}: only {within}/{trials} exact qualities within the bound"
            );
        }
        let mut kth = KthQuality::new(3);
        for value in [0.1, 0.9, 0.5] {
            assert_eq!(kth.value(), None);
            kth.offer(value);
        }
        assert_eq!(kth.value(), Some(0.1));
        kth.offer(0.7);
        kth.offer(f32::NAN);
        assert_eq!(kth.value(), Some(0.5));
    }

    #[test]
    fn codes_and_factors_round_trip_the_residual() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for metric in [
            VectorMetric::Dot,
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
        ] {
            let dim = 100;
            let mut encoder = Encoder::new(metric, dim).unwrap();
            let center: Vec<f32> = gaussian(&mut rng, dim).iter().map(|v| v * 0.1).collect();
            let x = gaussian(&mut rng, dim);
            let (mut codes, mut factors) = (Vec::new(), Vec::new());
            encoder
                .encode_row(&x, &center, &mut codes, &mut factors)
                .unwrap();
            assert_eq!(codes.len(), encoder.code_bytes());
            assert_eq!(codes.len(), 128 / 8);
            // Bits are the signs of the rotated residual.
            let (mut effective, mut residual) = (Vec::new(), Vec::new());
            let stats = residual_into(metric, &x, &center, &mut effective, &mut residual);
            let padded = encoder.rotation.padded_dim();
            let (mut rotated, mut scratch) = (vec![0.0; padded], vec![0.0; padded]);
            encoder
                .rotation
                .apply(&residual, &mut rotated, &mut scratch);
            for (index, value) in rotated.iter().enumerate() {
                let bit = codes[index / 8] >> (index % 8) & 1;
                assert_eq!(bit == 1, *value > 0.0);
            }
            let factors = [factors[0], factors[1], factors[2]];
            assert!(exact_row_matches_code(
                metric,
                &x,
                &center,
                factors,
                &mut effective,
                &mut residual
            ));
            let [a, b, e] = factors.map(|v| v as f64);
            assert!(((b * b - e * e).sqrt() - stats.norm).abs() < 1e-4 * stats.norm);
            assert!((a - stats.additive).abs() < 1e-4 * stats.additive.abs().max(1e-3));
            // Another row does not pass for this one.
            let y = gaussian(&mut rng, dim);
            assert!(!exact_row_matches_code(
                metric,
                &y,
                &center,
                factors,
                &mut effective,
                &mut residual
            ));
            // A row equal to its center codes exactly.
            let (mut codes, mut factors) = (Vec::new(), Vec::new());
            let centered = if metric == VectorMetric::Cosine {
                normalized(center.clone())
            } else {
                center.clone()
            };
            encoder
                .encode_row(&centered, &centered, &mut codes, &mut factors)
                .unwrap();
            if metric == VectorMetric::Cosine {
                // Renormalizing a unit row moves it by rounding only.
                assert!(factors[1] < 1e-5 && factors[2] < 1e-5);
            } else {
                assert!(codes.iter().all(|b| *b == 0));
                assert_eq!(&factors[1..], &[0.0, 0.0]);
            }
        }
    }

    #[test]
    fn locator_round_trips_and_rejects_damage() {
        let bytes = locator(123_456, 789, 0xDEAD_BEEF);
        assert_eq!(parse_locator(&bytes), Some((123_456, 789, 0xDEAD_BEEF)));
        for position in 0..LOCATOR_LEN {
            let mut damaged = bytes;
            damaged[position] ^= 0x10;
            assert_eq!(parse_locator(&damaged), None, "byte {position}");
        }
        assert_eq!(parse_locator(&bytes[1..]), None);
        assert_eq!(parse_locator(&[0u8; LOCATOR_LEN]), None);
    }

    fn build(
        metric: VectorMetric,
        members: &[([u8; 16], Vec<f32>)],
        with_section: bool,
        filters: VectorFilterPostings,
    ) -> Bytes {
        SKIP_SECTION.with(|flag| flag.set(!with_section));
        let built = build_body_v5_with_filter_postings(
            &descriptor(metric, members[0].1.len() as u32),
            members.to_vec(),
            filters,
            VectorV5BuildOptions {
                target_rows_per_page: 64,
                branch_factor: 4,
                compression_level: 1,
            },
        );
        SKIP_SECTION.with(|flag| flag.set(false));
        built.unwrap().unwrap().0
    }

    async fn open(bytes: &Bytes) -> VectorV5Reader {
        VectorV5Reader::open(Arc::new(MemorySource(bytes.clone())), bytes.len() as u64)
            .await
            .unwrap()
    }

    /// Footer offset and footer bytes of a body.
    fn footer_of(bytes: &[u8]) -> (usize, Vec<u8>) {
        let trailer = &bytes[bytes.len() - 20..];
        let len = u64::from_le_bytes(trailer[8..16].try_into().unwrap()) as usize;
        let start = bytes.len() - 20 - len;
        (start, bytes[start..bytes.len() - 20].to_vec())
    }

    #[tokio::test]
    async fn section_leaves_every_byte_an_older_reader_reads_unchanged() {
        let members = clustered(1_000, 48, 6, 7);
        for metric in [
            VectorMetric::Dot,
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
        ] {
            let old = build(metric, &members, false, VectorFilterPostings::new());
            let new = build(metric, &members, true, VectorFilterPostings::new());
            let (old_footer_at, old_footer) = footer_of(&old);
            let (new_footer_at, new_footer) = footer_of(&new);
            // Footer, trailer and every page block are byte-identical; the
            // section only occupies the gap older readers never read.
            assert_eq!(old_footer, new_footer);
            assert_eq!(old[..old_footer_at], new[..old_footer_at]);
            assert_eq!(old[old_footer_at..], new[new_footer_at..]);
            assert!(new_footer_at > old_footer_at);

            let old_reader = open(&old).await;
            let new_reader = open(&new).await;
            assert!(!old_reader.has_rabitq() && old_reader.rabitq_absent());
            assert!(new_reader.has_rabitq() && !new_reader.rabitq_absent());
            let stats = new_reader.rabitq_stats().unwrap();
            assert_eq!(stats.padded_dim, 64);
            assert_eq!(stats.code_block_bytes, 1_000 * (8 + 12));
            // The int8 path (all an older reader has) gives identical results
            // on both bodies, and the RaBitQ path agrees with exact search.
            let options = VectorV5SearchOptions {
                nprobe: 4,
                max_nprobe: 64,
                rerank_factor: 8,
            };
            for (index, (_, query)) in members.iter().enumerate().step_by(97) {
                let a = old_reader
                    .search_filter_groups(query, 10, options, &[])
                    .await
                    .unwrap();
                let b = new_reader
                    .search_filter_groups_int8(query, 10, options, &[])
                    .await
                    .unwrap();
                assert_eq!(a, b, "{metric:?} query {index}");
            }
        }
    }

    async fn overlap(
        reader: &VectorV5Reader,
        members: &[([u8; 16], Vec<f32>)],
        metric: VectorMetric,
    ) -> (f64, f64, f64) {
        let pages = reader.page_count();
        let options = VectorV5SearchOptions {
            nprobe: pages,
            max_nprobe: pages,
            rerank_factor: 8,
        };
        let (mut rabitq, mut int8, mut reranked) = (0usize, 0usize, 0usize);
        let mut queries = 0usize;
        for (_, query) in members.iter().step_by(37) {
            let exact = reader
                .search_exact_filter_groups(query, 10, &[])
                .await
                .unwrap();
            let truth: HashSet<[u8; 16]> = exact.hits.iter().map(|(id, _)| *id).collect();
            let fast = reader
                .search_filter_groups(query, 10, options, &[])
                .await
                .unwrap();
            let slow = reader
                .search_filter_groups_int8(query, 10, options, &[])
                .await
                .unwrap();
            rabitq += fast
                .hits
                .iter()
                .filter(|(id, _)| truth.contains(id))
                .count();
            int8 += slow
                .hits
                .iter()
                .filter(|(id, _)| truth.contains(id))
                .count();
            reranked += fast.reranked_rows;
            queries += 1;
            // Exact scores: a hit's score equals the exact search's score.
            for (id, score) in &fast.hits {
                if let Some((_, expected)) = exact.hits.iter().find(|(other, _)| other == id) {
                    assert!((score - expected).abs() < 1e-5, "{metric:?}");
                }
            }
        }
        let denominator = (queries * 10) as f64;
        (
            rabitq as f64 / denominator,
            int8 as f64 / denominator,
            reranked as f64 / queries as f64,
        )
    }

    #[tokio::test]
    async fn rabitq_search_matches_exact_with_every_page_probed() {
        let members = clustered(3_000, 96, 12, 21);
        for metric in [
            VectorMetric::Dot,
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
        ] {
            let body = build(metric, &members, true, VectorFilterPostings::new());
            let reader = open(&body).await;
            assert!(reader.has_rabitq());
            let (rabitq, int8, reranked) = overlap(&reader, &members, metric).await;
            assert!(
                rabitq >= 0.99,
                "{metric:?}: rabitq overlap {rabitq} (int8 {int8})"
            );
            assert!(reranked <= 8.0 * 10.0 * CANDIDATE_CAP_FACTOR as f64);
        }
    }

    #[tokio::test]
    async fn filters_apply_before_k_on_the_rabitq_path() {
        let members = clustered(1_200, 32, 4, 3);
        let (rare, common): (Vec<u32>, Vec<u32>) =
            (0..members.len() as u32).partition(|o| o % 7 == 0);
        let mut postings = VectorFilterPostings::new();
        postings.insert(
            "tag".to_string(),
            BTreeMap::from([
                ("s:rare".to_string(), VectorFilterPosting::Sparse(rare)),
                ("s:common".to_string(), VectorFilterPosting::Sparse(common)),
            ]),
        );
        let body = build(VectorMetric::Cosine, &members, true, postings);
        let reader = open(&body).await;
        assert!(reader.has_rabitq());
        let groups = vec![("tag".to_string(), vec![Value::Str("rare".to_string())])];
        let options = VectorV5SearchOptions {
            nprobe: 2,
            max_nprobe: reader.page_count(),
            rerank_factor: 8,
        };
        for (ordinal, (_, query)) in members.iter().enumerate().step_by(101) {
            let fast = reader
                .search_filter_groups(query, 10, options, &groups)
                .await
                .unwrap();
            let slow = reader
                .search_filter_groups_int8(query, 10, options, &groups)
                .await
                .unwrap();
            assert_eq!(fast.applied_filter_groups, 1);
            assert_eq!(fast.hits.len(), 10, "query {ordinal}");
            for (id, _) in &fast.hits {
                let row = u32::from_be_bytes(id[12..].try_into().unwrap()) as usize;
                assert_eq!(row % 7, 0, "only rare rows qualify");
            }
            assert_eq!(fast.eligible_rows_seen, slow.eligible_rows_seen);
            assert_eq!(fast.probed_pages, slow.probed_pages);
        }
    }

    #[tokio::test]
    async fn a_damaged_section_falls_back_to_int8_navigation() {
        let members = clustered(800, 32, 4, 9);
        let body = build(
            VectorMetric::Euclidean,
            &members,
            true,
            VectorFilterPostings::new(),
        );
        let (footer_at, _) = footer_of(&body);
        let options = VectorV5SearchOptions::default();
        let query = &members[17].1;
        let expected = open(&body)
            .await
            .search_filter_groups_int8(query, 5, options, &[])
            .await
            .unwrap();
        // Damage the directory (just below the locator): the locator still
        // parses, the directory checksum does not.
        let mut damaged = body.to_vec();
        damaged[footer_at - LOCATOR_LEN - 3] ^= 0xFF;
        let reader = open(&Bytes::from(damaged)).await;
        assert!(!reader.has_rabitq());
        assert!(
            !reader.rabitq_absent(),
            "invalid, not absent: no rebuild loop"
        );
        let got = reader
            .search_filter_groups(query, 5, options, &[])
            .await
            .unwrap();
        assert_eq!(got, expected);
        // Damage a code block: queries fail loudly instead of trusting it.
        let reader = open(&body).await;
        let RabitqState::Present(index) = &reader.rabitq else {
            panic!("section expected");
        };
        let at = index.directory.pages[0].codes.offset as usize;
        let mut damaged = body.to_vec();
        damaged[at] ^= 0x01;
        let reader = open(&Bytes::from(damaged)).await;
        assert!(reader.has_rabitq());
        let pages = reader.page_count();
        let all = VectorV5SearchOptions {
            nprobe: pages,
            max_nprobe: pages,
            rerank_factor: 8,
        };
        assert!(reader
            .search_filter_groups(query, 5, all, &[])
            .await
            .is_err());
    }

    #[tokio::test]
    async fn bases_without_a_section_are_served_by_int8_and_reported_absent() {
        let members = clustered(500, 16, 3, 2);
        let body = build(
            VectorMetric::Dot,
            &members,
            false,
            VectorFilterPostings::new(),
        );
        let reader = open(&body).await;
        assert!(reader.rabitq_absent() && reader.rabitq_stats().is_none());
        let options = VectorV5SearchOptions::default();
        let a = reader
            .search_filter_groups(&members[3].1, 7, options, &[])
            .await
            .unwrap();
        let b = reader
            .search_filter_groups_int8(&members[3].1, 7, options, &[])
            .await
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.reranked_rows, b.reranked_rows);
        let name = format!("{}-vector-graph-doc.vg", Uuid::new_v4().simple());
        assert!(!base_lacks_section(&format!("sst/level1/{name}")));
        // A cache key (namespace prefix + path) marks the same object.
        note_base_without_section(&format!("ns/sst/level1/{name}"));
        assert!(base_lacks_section(&format!("sst/level1/{name}")));
        // The in-memory builder still writes a section by default.
        let body = build_body_v5(
            &descriptor(VectorMetric::Dot, 16),
            members,
            VectorV5BuildOptions::default(),
        )
        .unwrap()
        .unwrap()
        .0;
        assert!(open(&body).await.has_rabitq());
    }

    #[tokio::test]
    async fn prefetch_reads_every_code_and_id_block_once_within_its_budget() {
        #[derive(Debug)]
        struct Counting(Bytes, std::sync::atomic::AtomicU64);
        #[async_trait::async_trait]
        impl VectorV5RangeSource for Counting {
            async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error> {
                self.1.fetch_add(
                    range.end - range.start,
                    std::sync::atomic::Ordering::Relaxed,
                );
                Ok(self.0.slice(range.start as usize..range.end as usize))
            }
        }
        let members = clustered(900, 32, 4, 9);
        let body = build(
            VectorMetric::Cosine,
            &members,
            true,
            VectorFilterPostings::new(),
        );
        let source = Arc::new(Counting(body.clone(), Default::default()));
        let reader = VectorV5Reader::open(source.clone(), body.len() as u64)
            .await
            .unwrap();
        let stats = reader.rabitq_stats().unwrap();
        let before = source.1.load(std::sync::atomic::Ordering::Relaxed);
        // Too small a share: nothing is read.
        let skipped = reader.prefetch_resident_codes(0).await.unwrap();
        assert!(skipped.section_present && skipped.skipped && skipped.loaded_bytes == 0);
        // Codes and ids (this base has no filter properties).
        let ids = members.len() as u64 * ID_BYTES as u64;
        assert_eq!(skipped.block_bytes, stats.code_block_bytes + ids);
        let skipped = reader
            .prefetch_resident_codes(skipped.block_bytes - 1)
            .await
            .unwrap();
        assert!(skipped.skipped && skipped.loaded_bytes == 0);
        assert_eq!(source.1.load(std::sync::atomic::Ordering::Relaxed), before);
        let done = reader
            .prefetch_resident_codes(skipped.block_bytes)
            .await
            .unwrap();
        assert!(!done.skipped);
        assert_eq!(done.code_bytes, stats.code_block_bytes);
        assert_eq!(done.loaded_bytes, done.block_bytes);
        assert_eq!(
            source.1.load(std::sync::atomic::Ordering::Relaxed) - before,
            done.block_bytes
        );
        // A base without the section reads nothing.
        let body = build(
            VectorMetric::Cosine,
            &members,
            false,
            VectorFilterPostings::new(),
        );
        let reader = open(&body).await;
        assert_eq!(
            reader.prefetch_resident_codes(u64::MAX).await.unwrap(),
            VectorV5CodePrefetch::default()
        );
    }

    #[tokio::test]
    async fn zero_cosine_query_and_single_row_bases_work() {
        let members = vec![(id(1), vec![0.5f32, -0.25, 1.0])];
        for metric in [
            VectorMetric::Dot,
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
        ] {
            let body = build(metric, &members, true, VectorFilterPostings::new());
            let reader = open(&body).await;
            assert!(reader.has_rabitq());
            let hits = reader
                .search_filter_groups(&[0.0, 0.0, 0.0], 3, VectorV5SearchOptions::default(), &[])
                .await
                .unwrap()
                .hits;
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].0, id(1));
        }
    }
}
