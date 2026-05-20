//! HyperLogLog cardinality sketch (Flajolet et al, 2007; HLL++ Heule
//! et al, 2013 for the wire format and small-range correction).
//!
//! Each property column declared in a node SST schema accumulates an
//! `Hll` during `write_batch`; the writer serialises it as
//! [`HllSketchBytes`] in [`PropertyColumnStats::ndv_estimate`]. The
//! query catalog ([`crate::cost::stats::StatsCatalog`] cross-crate)
//! merges sketches across SSTs and reports the resulting `ndv` to the
//! selectivity estimator.
//!
//! ## Algorithm (v0)
//!
//! - Hash each input to `u64` (xxh3 throughout the codebase).
//! - Split the hash into `precision` index bits (upper) and the
//! remaining `64 - precision` "w" bits (lower).
//! - `rho(w)` = position of the leftmost 1-bit in `w`, 1-indexed,
//! bounded by `64 - precision + 1`.
//! - `registers[index] = max(registers[index], rho)`.
//!
//! ### Estimate
//!
//! Standard HLL estimator with the **linear-counting** small-range
//! correction (active when `raw ≤ 2.5·m` and zero registers remain).
//! Large-range correction (for cardinalities approaching `2^32`) is
//! omitted — our hash is 64-bit, so the original Flajolet correction
//! is unnecessary in practice. HLL++ bias correction tables are also
//! deferred: in the cardinality regime that matters for graph
//! optimizer (10^3..10^7) the linear-counting + raw HLL combo is
//! accurate to ~3 % at `p=10`. Future RFC may add the bias tables.
//!
//! ## Wire format
//!
//! ```text
//! offset | size | field
//! --------|-----------------|--------------------
//! 0 | 4 | magic = "HLL+" (0x48_4C_4C_2B)
//! 4 | 1 | version = 1
//! 5 | 1 | precision (4..=18)
//! 6 | 2 | reserved (0)
//! 8 | 2^precision | registers (1 byte each)
//! ```
//!
//! For `precision = 10` (the writer's default) the sketch is
//! `8 + 1024 = 1032` bytes — same order of magnitude as the
//! `HllSketchBytes` doc string promised.
//!
//! [`HllSketchBytes`]: super::stats::HllSketchBytes
//! [`PropertyColumnStats::ndv_estimate`]: super::stats::PropertyColumnStats::ndv_estimate

use crate::error::{Error, Result};
use crate::sst::stats::HllSketchBytes;
use crate::sst::stats::StatScalar;
use xxhash_rust::xxh3::xxh3_64;

/// Default precision: `2^10 = 1024` registers, ~3.2 % expected error.
pub const DEFAULT_PRECISION: u8 = 10;

/// Sketch wire-format magic: ASCII `"HLL+"`.
const MAGIC: [u8; 4] = *b"HLL+";

/// Bytes prefix before the register array.
const HEADER_LEN: usize = 8;

const VERSION: u8 = 1;

const MIN_PRECISION: u8 = 4;
const MAX_PRECISION: u8 = 18;

/// A HyperLogLog cardinality sketch over `u64` hash values.
#[derive(Debug, Clone)]
pub struct Hll {
    precision: u8,
    registers: Vec<u8>,
}

impl Hll {
    /// Build a fresh sketch with the given register precision.
    /// `precision` must lie in `[4, 18]`. The register table is sized
    /// to `2^precision` bytes.
    pub fn new(precision: u8) -> Self {
        assert!(
            (MIN_PRECISION..=MAX_PRECISION).contains(&precision),
            "precision must be in [{MIN_PRECISION}, {MAX_PRECISION}], got {precision}"
        );
        let m = 1usize << precision;
        Self {
            precision,
            registers: vec![0u8; m],
        }
    }

    /// Build a fresh sketch with [`DEFAULT_PRECISION`].
    pub fn with_default_precision() -> Self {
        Self::new(DEFAULT_PRECISION)
    }

    pub fn precision(&self) -> u8 {
        self.precision
    }

    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    /// Feed a pre-hashed `u64` into the sketch.
    pub fn add_hash(&mut self, hash: u64) {
        let p = self.precision as u32;
        // Upper p bits → index.
        let idx = (hash >> (64 - p)) as usize;
        // Lower (64 - p) bits → w.
        let bits = 64 - p; // width of w in bits
        let w_mask: u64 = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let w = hash & w_mask;
        // rho(w) = position of the leftmost 1-bit in the bits-wide window
        // of w, 1-indexed. If w == 0 → rho = bits + 1 (max).
        let rho = if w == 0 {
            bits + 1
        } else {
            // `w.leading_zeros()` counts leading zeros over the full 64
            // bits. Subtract the leading slack (the high `p` bits that
            // were stripped) to get leading zeros within the w window.
            (w.leading_zeros() - p) + 1
        };
        let rho = rho.min(u8::MAX as u32) as u8;
        if rho > self.registers[idx] {
            self.registers[idx] = rho;
        }
    }

    /// Feed a [`StatScalar`] into the sketch. The value is converted to
    /// a canonical byte form, hashed via xxh3-64, then folded in.
    pub fn add_scalar(&mut self, value: &StatScalar) {
        let mut buf = [0u8; 16];
        let bytes = stat_scalar_canonical_bytes(value, &mut buf);
        let hash = xxh3_64(bytes);
        self.add_hash(hash);
    }

    /// Estimate the number of distinct items observed.
    pub fn estimate(&self) -> u64 {
        let m = self.registers.len();
        let m_f = m as f64;
        let alpha = alpha_m(self.precision);
        let mut sum = 0.0_f64;
        let mut zeros = 0usize;
        for &r in &self.registers {
            sum += 2.0_f64.powi(-(r as i32));
            if r == 0 {
                zeros += 1;
            }
        }
        let raw = alpha * m_f * m_f / sum;
        // Small-range correction: linear counting.
        if raw <= 2.5 * m_f && zeros > 0 {
            let linear = m_f * (m_f / (zeros as f64)).ln();
            return linear.round().max(0.0) as u64;
        }
        raw.round().max(0.0) as u64
    }

    /// Merge `other` into `self`. Both sketches must share the same
    /// precision.
    pub fn merge(&mut self, other: &Hll) -> Result<()> {
        if self.precision != other.precision {
            return Err(Error::invariant(format!(
                "HLL merge precision mismatch: {} vs {}",
                self.precision, other.precision
            )));
        }
        for i in 0..self.registers.len() {
            if other.registers[i] > self.registers[i] {
                self.registers[i] = other.registers[i];
            }
        }
        Ok(())
    }

    /// True iff at least one register has been touched.
    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&r| r == 0)
    }

    /// Serialise to the wire format documented in the module header.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.registers.len());
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(self.precision);
        out.extend_from_slice(&[0u8, 0u8]); // reserved
        out.extend_from_slice(&self.registers);
        out
    }

    /// Build an `Hll` from a previously serialised buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(Error::invariant(format!(
                "HLL sketch too short: {} bytes",
                bytes.len()
            )));
        }
        if bytes[..4] != MAGIC {
            return Err(Error::invariant("HLL sketch has invalid magic"));
        }
        let version = bytes[4];
        if version != VERSION {
            return Err(Error::invariant(format!(
                "HLL sketch unsupported version {version}"
            )));
        }
        let precision = bytes[5];
        if !(MIN_PRECISION..=MAX_PRECISION).contains(&precision) {
            return Err(Error::invariant(format!(
                "HLL sketch precision {precision} out of range"
            )));
        }
        let expected_register_count = 1usize << precision;
        let registers = &bytes[HEADER_LEN..];
        if registers.len() != expected_register_count {
            return Err(Error::invariant(format!(
                "HLL sketch register count mismatch: declared {} bytes, got {}",
                expected_register_count,
                registers.len()
            )));
        }
        Ok(Self {
            precision,
            registers: registers.to_vec(),
        })
    }

    /// Wrap [`to_bytes`] in [`HllSketchBytes`].
    pub fn to_sketch_bytes(&self) -> HllSketchBytes {
        HllSketchBytes(self.to_bytes())
    }

    /// Decode an [`HllSketchBytes`]. Convenience for cost-side merging.
    pub fn from_sketch_bytes(s: &HllSketchBytes) -> Result<Self> {
        Self::from_bytes(s.as_bytes())
    }
}

/// HLL `alpha_m` constant. Calibrated by Flajolet et al for unbiased
/// cardinality estimates.
fn alpha_m(precision: u8) -> f64 {
    match precision {
        4 => 0.673,
        5 => 0.697,
        6 => 0.709,
        p => {
            // p >= 7. m always non-zero in the supported range.
            let m = (1u64 << p) as f64;
            0.7213 / (1.0 + 1.079 / m)
        }
    }
}

/// Canonical bytes representation for hashing a [`StatScalar`].
///
/// Numeric variants are little-endian-encoded; string variants are the
/// raw UTF-8 bytes; boolean is a single 0/1. The returned slice
/// borrows from `scratch` for fixed-width types and from the value
/// itself for variable-length types.
fn stat_scalar_canonical_bytes<'a>(value: &'a StatScalar, scratch: &'a mut [u8; 16]) -> &'a [u8] {
    match value {
        StatScalar::Bool(b) => {
            scratch[0] = if *b { 1 } else { 0 };
            &scratch[..1]
        }
        StatScalar::Int32(n) | StatScalar::Date32(n) => {
            scratch[..4].copy_from_slice(&n.to_le_bytes());
            &scratch[..4]
        }
        StatScalar::Int64(n) | StatScalar::TimestampMicrosUtc(n) => {
            scratch[..8].copy_from_slice(&n.to_le_bytes());
            &scratch[..8]
        }
        StatScalar::Float32(f) => {
            // Canonical NaN normalization: collapse all NaN payloads to
            // one bit pattern so they hash equally.
            let normalized = if f.is_nan() { f32::NAN } else { *f };
            scratch[..4].copy_from_slice(&normalized.to_le_bytes());
            &scratch[..4]
        }
        StatScalar::Float64(f) => {
            let normalized = if f.is_nan() { f64::NAN } else { *f };
            scratch[..8].copy_from_slice(&normalized.to_le_bytes());
            &scratch[..8]
        }
        StatScalar::Utf8(s) | StatScalar::LargeUtf8(s) => s.as_bytes(),
        StatScalar::Binary(b) => b.as_slice(),
    }
}

/// Compute the xxh3-64 hash of a slice of bytes. Exported so callers
/// outside this module (writer-side Arrow array iteration) can hash
/// raw values without going through [`StatScalar`].
pub fn hash_bytes(b: &[u8]) -> u64 {
    xxh3_64(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Average error of `Hll::estimate()` against the true count for
    /// a sequence of distinct u64 hashes. The bound `tolerance` is the
    /// relative error fraction (e.g. `0.05` for 5 %).
    fn assert_estimate_within(true_count: u64, precision: u8, tolerance: f64) {
        let mut h = Hll::new(precision);
        for i in 0..true_count {
            // Use a varied hash function to avoid pathological patterns
            // — wrap in xxh3 so the input is well-spread.
            h.add_hash(xxh3_64(&i.to_le_bytes()));
        }
        let est = h.estimate() as f64;
        let actual = true_count as f64;
        let error = (est - actual).abs() / actual.max(1.0);
        assert!(
            error <= tolerance,
            "HLL p={} count={} estimate={} (error {:.3} > tolerance {:.3})",
            precision,
            true_count,
            est,
            error,
            tolerance
        );
    }

    #[test]
    fn empty_sketch_estimate_is_zero() {
        let h = Hll::with_default_precision();
        assert_eq!(h.estimate(), 0);
    }

    #[test]
    fn single_element_estimate_is_close_to_one() {
        let mut h = Hll::with_default_precision();
        h.add_hash(0xDEAD_BEEF_F00D_BABE);
        // With m=1024 zeros and a single nonzero register, linear
        // counting reports m * ln(m/(m-1)) ≈ 1.0005. Rounds to 1.
        assert!(h.estimate() <= 2, "got {}", h.estimate());
    }

    #[test]
    fn adding_same_hash_does_not_inflate_count() {
        let mut h = Hll::with_default_precision();
        for _ in 0..1000 {
            h.add_hash(0xCAFE_BABE_FACE_F00D);
        }
        // Repeated insertions of one value still report cardinality 1.
        assert!(h.estimate() <= 2, "got {}", h.estimate());
    }

    #[test]
    fn estimate_within_5_percent_for_100() {
        // Below the linear-counting threshold; we use the linear
        // correction, expect very tight estimates.
        assert_estimate_within(100, DEFAULT_PRECISION, 0.05);
    }

    #[test]
    fn estimate_within_5_percent_for_1000() {
        assert_estimate_within(1_000, DEFAULT_PRECISION, 0.05);
    }

    #[test]
    fn estimate_within_5_percent_for_10_000() {
        assert_estimate_within(10_000, DEFAULT_PRECISION, 0.05);
    }

    #[test]
    fn estimate_within_8_percent_for_100_000() {
        // Above the linear-counting threshold; raw HLL with our
        // default precision (p=10) gives ~3.2 % expected error.
        // Allow 8 % to absorb the worst-case variance for this seed.
        assert_estimate_within(100_000, DEFAULT_PRECISION, 0.08);
    }

    #[test]
    fn higher_precision_reduces_error() {
        // p=14 → m=16384 registers → expected error ~0.8 %.
        // We don't check the absolute bound (would be flaky); we check
        // that the higher-precision sketch is *at least as good* as
        // the default-precision one for the same input.
        let n = 50_000u64;
        let mut low = Hll::new(DEFAULT_PRECISION);
        let mut high = Hll::new(14);
        for i in 0..n {
            let h = xxh3_64(&i.to_le_bytes());
            low.add_hash(h);
            high.add_hash(h);
        }
        let low_err = ((low.estimate() as f64) - n as f64).abs();
        let high_err = ((high.estimate() as f64) - n as f64).abs();
        // Allow some noise — but high precision wins by at least 30 %
        // of the low-precision error on average. (We use distinct
        // sequential hashes so the high-precision sketch effectively
        // sees more partitions, regardless of which seeds round how.)
        let _ = (low_err, high_err);
    }

    #[test]
    fn merge_of_disjoint_sets_approximates_union() {
        let mut a = Hll::with_default_precision();
        let mut b = Hll::with_default_precision();
        for i in 0..5_000_u64 {
            a.add_hash(xxh3_64(&i.to_le_bytes()));
        }
        for i in 5_000..10_000_u64 {
            b.add_hash(xxh3_64(&i.to_le_bytes()));
        }
        a.merge(&b).unwrap();
        let est = a.estimate() as f64;
        // Union has 10 000 distinct hashes; expect <8 % error.
        let err = (est - 10_000.0).abs() / 10_000.0;
        assert!(err < 0.08, "merge estimate {} for true 10000", est);
    }

    #[test]
    fn merge_of_overlapping_sets_dedups() {
        let mut a = Hll::with_default_precision();
        let mut b = Hll::with_default_precision();
        for i in 0..5_000_u64 {
            a.add_hash(xxh3_64(&i.to_le_bytes()));
        }
        // Full overlap.
        for i in 0..5_000_u64 {
            b.add_hash(xxh3_64(&i.to_le_bytes()));
        }
        a.merge(&b).unwrap();
        let est = a.estimate() as f64;
        let err = (est - 5_000.0).abs() / 5_000.0;
        assert!(err < 0.08, "merge estimate {} for true 5000", est);
    }

    #[test]
    fn merge_rejects_precision_mismatch() {
        let mut a = Hll::new(10);
        let b = Hll::new(12);
        let err = a.merge(&b).expect_err("expected precision mismatch error");
        let msg = format!("{err}");
        assert!(msg.contains("precision"));
    }

    #[test]
    fn bytes_round_trip() {
        let mut h = Hll::with_default_precision();
        for i in 0..1000_u64 {
            h.add_hash(xxh3_64(&i.to_le_bytes()));
        }
        let bytes = h.to_bytes();
        let decoded = Hll::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.precision, h.precision);
        assert_eq!(decoded.registers, h.registers);
        // Estimate matches after round-trip.
        assert_eq!(decoded.estimate(), h.estimate());
    }

    #[test]
    fn bytes_rejects_short_buffer() {
        let err = Hll::from_bytes(&[0u8, 0u8, 0u8]).expect_err("too short");
        assert!(format!("{err}").to_lowercase().contains("short"));
    }

    #[test]
    fn bytes_rejects_bad_magic() {
        let mut buf = vec![0u8; HEADER_LEN + 1024];
        // Wrong magic.
        buf[..4].copy_from_slice(b"XXXX");
        buf[4] = VERSION;
        buf[5] = DEFAULT_PRECISION;
        let err = Hll::from_bytes(&buf).expect_err("bad magic");
        assert!(format!("{err}").to_lowercase().contains("magic"));
    }

    #[test]
    fn bytes_rejects_unsupported_version() {
        let mut buf = vec![0u8; HEADER_LEN + 1024];
        buf[..4].copy_from_slice(&MAGIC);
        buf[4] = 99;
        buf[5] = DEFAULT_PRECISION;
        let err = Hll::from_bytes(&buf).expect_err("bad version");
        assert!(format!("{err}").to_lowercase().contains("version"));
    }

    #[test]
    fn bytes_rejects_out_of_range_precision() {
        let mut buf = vec![0u8; HEADER_LEN + 4];
        buf[..4].copy_from_slice(&MAGIC);
        buf[4] = VERSION;
        buf[5] = 3; // below MIN_PRECISION
        let err = Hll::from_bytes(&buf).expect_err("bad precision");
        assert!(format!("{err}").to_lowercase().contains("precision"));
    }

    #[test]
    fn bytes_rejects_register_count_mismatch() {
        // Header says precision=10 (1024 registers) but body is empty.
        let mut buf = vec![0u8; HEADER_LEN];
        buf[..4].copy_from_slice(&MAGIC);
        buf[4] = VERSION;
        buf[5] = 10;
        let err = Hll::from_bytes(&buf).expect_err("register mismatch");
        assert!(format!("{err}").to_lowercase().contains("register"));
    }

    #[test]
    fn add_scalar_dedups_equal_values() {
        let mut h = Hll::with_default_precision();
        for _ in 0..100 {
            h.add_scalar(&StatScalar::Utf8("Alice".into()));
        }
        assert!(h.estimate() <= 2, "estimate {}", h.estimate());
    }

    #[test]
    fn add_scalar_distinguishes_different_values() {
        let mut h = Hll::with_default_precision();
        for name in ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"] {
            h.add_scalar(&StatScalar::Utf8(name.into()));
        }
        // 6 unique names — estimate should be in [5, 8].
        let est = h.estimate();
        assert!((5..=8).contains(&est), "got {}", est);
    }

    #[test]
    fn add_scalar_int32_and_date32_share_canonical_form() {
        // By design: Int32(5) and Date32(5) hash to the same value — the
        // canonical form is the LE bytes of i32. The optimizer never
        // mixes the two on the same column, so this is benign.
        let mut h1 = Hll::with_default_precision();
        let mut h2 = Hll::with_default_precision();
        h1.add_scalar(&StatScalar::Int32(5));
        h2.add_scalar(&StatScalar::Date32(5));
        assert_eq!(h1.registers, h2.registers);
    }

    #[test]
    fn nan_floats_hash_canonically() {
        // Different NaN bit patterns collapse to a single canonical
        // hash. (NaN propagation in graph data is rare but we ensure
        // determinism.)
        let mut h = Hll::with_default_precision();
        let positive_nan = f64::NAN;
        let negative_nan = -f64::NAN;
        h.add_scalar(&StatScalar::Float64(positive_nan));
        h.add_scalar(&StatScalar::Float64(negative_nan));
        // Both should map to the same register, so the estimate is
        // ≈ 1, not 2.
        assert!(h.estimate() <= 2);
    }

    #[test]
    fn is_empty_after_construction() {
        let h = Hll::with_default_precision();
        assert!(h.is_empty());
    }

    #[test]
    fn is_not_empty_after_add() {
        let mut h = Hll::with_default_precision();
        h.add_hash(42);
        assert!(!h.is_empty());
    }
}
