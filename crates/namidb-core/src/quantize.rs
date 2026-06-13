//! Symmetric per-vector int8 quantization for embedding vectors.
//!
//! Each vector is quantized with its own max-abs scale, so the full int8 range
//! [-127, 127] is used no matter the dimension. A single fixed scale wastes
//! almost all of the range for high-dimensional unit vectors — their
//! components are ~1/sqrt(dim) and tiny, so `x * 127` lands in single digits
//! and recall collapses (the `namidb-bench vector-recall` harness measures
//! exactly this: fixed-scale recall@10 falls to ~0.87 at dim 1536, per-vector
//! scaling restores it). Quantization is lossy; the harness reports recall@k.
//!
//! Stored form is `(codes: Vec<i8>, scale: f32)`, with `x_i ≈ codes_i * scale`.
//! The asymmetric scorer keeps the query in f32 and folds the scale into the
//! dot product: `dot(query, stored) = scale * Σ query_i * codes_i` — so the
//! stored side is never expanded back into an f32 vector. There is exactly one
//! definition of this mapping, shared by the write path, the scorer, and the
//! harness.

/// Quantize a vector with a per-vector symmetric max-abs scale. Returns the
/// int8 codes and the scale `s` such that `x_i ≈ codes_i * s`. A zero vector
/// (or empty) yields all-zero codes and `scale = 0.0`, which dequantizes back
/// to zeros.
///
/// Non-finite components (`NaN`/`±Inf`) are excluded from the scale and coded
/// as `0`, so the returned scale is always finite. Otherwise a single `Inf`
/// would make `max_abs` (and the scale) `Inf`, persisting a poisoned vector
/// that dequantizes to `NaN` — the storage layer relies on a finite scale.
pub fn quantize_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v
        .iter()
        .filter(|x| x.is_finite())
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    if max_abs == 0.0 {
        return (vec![0i8; v.len()], 0.0);
    }
    let scale = max_abs / 127.0;
    let codes = v
        .iter()
        .map(|&x| {
            if x.is_finite() {
                (x / scale).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            }
        })
        .collect();
    (codes, scale)
}

/// Recover the f32 vector from its int8 codes and scale (`codes_i * scale`).
pub fn dequantize_i8(codes: &[i8], scale: f32) -> Vec<f32> {
    codes.iter().map(|&c| c as f32 * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_bounds_the_error_per_vector() {
        // Per-vector error is at most half a step = 0.5 * scale, where
        // scale = max|x| / 127. So the bound scales with the vector's range.
        let v: Vec<f32> = (0..256).map(|i| (i as f32 / 255.0) * 0.2 - 0.1).collect();
        let (codes, scale) = quantize_i8(&v);
        let back = dequantize_i8(&codes, scale);
        for (x, y) in v.iter().zip(&back) {
            assert!(
                (x - y).abs() <= 0.5 * scale + 1e-6,
                "component {x} round-tripped to {y} (scale {scale})"
            );
        }
    }

    #[test]
    fn max_abs_component_is_exact() {
        // The component equal to max|x| maps to ±127 and dequantizes exactly.
        let (codes, scale) = quantize_i8(&[0.5, -0.25, 0.0]);
        assert_eq!(codes[0], 127);
        let back = dequantize_i8(&codes, scale);
        assert!((back[0] - 0.5).abs() < 1e-6);
        assert_eq!(codes[2], 0);
    }

    #[test]
    fn full_range_used_regardless_of_magnitude() {
        // Tiny components (high-dim unit vector) still span the int8 range,
        // which is the whole point of per-vector scaling.
        let (codes, _scale) = quantize_i8(&[0.02, -0.02, 0.01]);
        assert_eq!(codes[0], 127);
        assert_eq!(codes[1], -127);
    }

    #[test]
    fn zero_vector_is_all_zero() {
        let (codes, scale) = quantize_i8(&[0.0, 0.0, 0.0]);
        assert_eq!(codes, vec![0, 0, 0]);
        assert_eq!(scale, 0.0);
        assert_eq!(dequantize_i8(&codes, scale), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn negation_is_symmetric() {
        let v: Vec<f32> = [0.1, -0.37, 0.9, -0.04, 0.5].to_vec();
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        let (qv, _) = quantize_i8(&v);
        let (qn, _) = quantize_i8(&neg);
        for (a, b) in qv.iter().zip(&qn) {
            assert_eq!(*a, -*b);
        }
    }

    #[test]
    fn non_finite_components_stay_finite_and_zero_coded() {
        // An Inf/NaN must not poison the scale: it stays finite and the
        // offending components code to 0, so dequantize never yields NaN.
        let (codes, scale) = quantize_i8(&[0.5, f32::INFINITY, -0.25, f32::NAN]);
        assert!(scale.is_finite(), "scale must be finite, got {scale}");
        assert_eq!(codes[0], 127, "0.5 is the max finite component");
        assert_eq!(codes[1], 0, "Inf codes to 0");
        assert_eq!(codes[3], 0, "NaN codes to 0");
        assert!(dequantize_i8(&codes, scale).iter().all(|x| x.is_finite()));
    }

    #[test]
    fn all_non_finite_degrades_to_zero_vector() {
        let (codes, scale) = quantize_i8(&[f32::INFINITY, f32::NAN, f32::NEG_INFINITY]);
        assert_eq!(codes, vec![0, 0, 0]);
        assert_eq!(scale, 0.0);
    }
}
