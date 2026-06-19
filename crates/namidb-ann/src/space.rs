//! The [`VectorSpace`] abstraction: how the build/search algorithms reach the
//! stored vectors and score them, without knowing how they're laid out.
//!
//! Two concrete impls ship:
//! - [`F32CosineSpace`] — full f32 unit-normalized vectors, cosine distance.
//!   The recall-golden path (exact distances; used to validate the graph).
//! - [`Int8Space`] — per-vector int8 codes + scale (as written by the storage
//!   layer), scored with the shared `namidb_core::quantize` primitives. The
//!   shipped path. Cosine on int8 is **scale-invariant**: the per-vector scale
//!   appears identically in both numerator (dot) and denominator (norm) of the
//!   cosine, so it cancels — the impl computes it with the primitives anyway so
//!   there's one definition of the score.

use namidb_core::quantize::{dot_i8_asymmetric, norm_i8};

/// A collection of vectors the ANN algorithm can index and search. `Id`s are
/// dense `0..len()` indices.
///
/// All distances follow **"lower is closer"** semantics and **must be finite**
/// — the beam-search heaps use total ordering and a converged-search comparison
/// that assumes no `NaN`. Cosine distance (`1 - similarity`, similarity in
/// `[-1, 1]`) is finite and in `[0, 2]` for unit vectors; impls here enforce
/// finiteness via construction.
pub trait VectorSpace {
    /// Number of stored vectors. `Id`s are `0..len()`.
    fn len(&self) -> usize;

    /// `true` iff [`len`](Self::len) is `0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimensionality (all members share it).
    fn dim(&self) -> usize;

    /// Distance between two stored members — used by the build (NN-graph init
    /// + member search + robust-prune). Lower is closer; must be finite.
    fn pair_distance(&self, a: u32, b: u32) -> f32;

    /// Distance from an external f32 query to a stored member — used by search.
    /// Lower is closer; must be finite. The query need not be unit-normalized
    /// (the impl handles any normalization the metric needs).
    fn query_distance(&self, query: &[f32], b: u32) -> f32;
}

// ---------------------------------------------------------------------------
// f32 cosine space — the recall-golden path.
// ---------------------------------------------------------------------------

/// Full-precision f32 vectors scored by **cosine distance** (`1 - dot`).
///
/// Vectors are stored as given; cosine is scale-invariant, so normalization is
/// optional. For embedding recall workloads the caller pre-normalizes, but the
/// math is identical either way because the `|x|·|y|` factor cancels into the
/// ranking only through the per-query constant `|q|`, which cosine divides out.
#[derive(Clone)]
pub struct F32CosineSpace {
    vecs: Vec<Vec<f32>>,
}

impl F32CosineSpace {
    /// Build a space from owned f32 vectors. All must share the same length;
    /// an empty space is allowed (zero vectors, zero dim).
    pub fn new(vecs: Vec<Vec<f32>>) -> Self {
        Self { vecs }
    }

    /// Reference to the f32 vector for `id` (build-time introspection).
    pub fn vector(&self, id: u32) -> &[f32] {
        &self.vecs[id as usize]
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn norm(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

impl VectorSpace for F32CosineSpace {
    fn len(&self) -> usize {
        self.vecs.len()
    }

    fn dim(&self) -> usize {
        self.vecs.first().map(|v| v.len()).unwrap_or(0)
    }

    fn pair_distance(&self, a: u32, b: u32) -> f32 {
        let (va, vb) = (&self.vecs[a as usize], &self.vecs[b as usize]);
        let denom = Self::norm(va) * Self::norm(vb);
        if denom == 0.0 {
            // Two zero vectors are "identical" (distance 0); a zero vs nonzero
            // is orthogonal/maximally-distant-but-finite (distance 1.0).
            return if va.iter().all(|x| *x == 0.0) && vb.iter().all(|x| *x == 0.0) {
                0.0
            } else {
                1.0
            };
        }
        let cos = Self::dot(va, vb) / denom;
        // Clamp to [-1, 1] before `1 -` so floating error can't push the
        // distance slightly negative or above 2.
        1.0 - cos.clamp(-1.0, 1.0)
    }

    fn query_distance(&self, query: &[f32], b: u32) -> f32 {
        let vb = &self.vecs[b as usize];
        let denom = Self::norm(query) * Self::norm(vb);
        if denom == 0.0 {
            return if query.iter().all(|x| *x == 0.0) && vb.iter().all(|x| *x == 0.0) {
                0.0
            } else {
                1.0
            };
        }
        let cos = Self::dot(query, vb) / denom;
        1.0 - cos.clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// int8 cosine space — the shipped path.
// ---------------------------------------------------------------------------

/// int8-quantized vectors (`(codes, scale)` as written by the storage layer),
/// scored by **cosine distance** using the shared asymmetric scorer.
///
/// Cosine on per-vector-scaled int8 is scale-invariant: similarity is
/// `dot(q, x) / (|q|·|x|)`, and `|x| = scale·sqrt(Σ codes²)` while
/// `dot = scale·Σ q·code`, so `scale` cancels. The impl computes both with the
/// `quantize` primitives (one definition of the score) and lets the scale
/// cancel in the division — exact in f32, since the same `scale` multiplies
/// both terms.
#[derive(Clone)]
pub struct Int8Space {
    /// `(codes, scale)` per member.
    members: Vec<(Vec<i8>, f32)>,
    dim: usize,
}

impl Int8Space {
    /// Build from the stored form. All members must share `dim` codes; an empty
    /// space is allowed.
    pub fn new(members: Vec<(Vec<i8>, f32)>) -> Self {
        let dim = members.first().map(|(c, _)| c.len()).unwrap_or(0);
        Self { members, dim }
    }

    /// `(codes, scale)` for `id`.
    pub fn member(&self, id: u32) -> &(Vec<i8>, f32) {
        &self.members[id as usize]
    }

    /// int8×int8 dot (no scale) — the scale-invariant numerator piece of cosine.
    fn dot_i8_i8(a: &[i8], b: &[i8]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x as f32 * y as f32).sum()
    }

    /// `sqrt(Σ code²)` (no scale) — the scale-invariant denominator piece.
    fn l2_i8(a: &[i8]) -> f32 {
        a.iter().map(|&x| (x as f32) * (x as f32)).sum::<f32>().sqrt()
    }

    fn cosine(a: &[i8], b: &[i8]) -> f32 {
        let denom = Self::l2_i8(a) * Self::l2_i8(b);
        if denom == 0.0 {
            return if a.iter().all(|&x| x == 0) && b.iter().all(|&x| x == 0) {
                1.0
            } else {
                0.0
            };
        }
        (Self::dot_i8_i8(a, b) / denom).clamp(-1.0, 1.0)
    }
}

impl VectorSpace for Int8Space {
    fn len(&self) -> usize {
        self.members.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn pair_distance(&self, a: u32, b: u32) -> f32 {
        let (ca, _) = &self.members[a as usize];
        let (cb, _) = &self.members[b as usize];
        1.0 - Self::cosine(ca, cb)
    }

    fn query_distance(&self, query: &[f32], b: u32) -> f32 {
        // Cosine(query, x): scale cancels between dot_i8_asymmetric (×scale) and
        // norm_i8 (×scale), so the result is identical to dividing the unscaled
        // sum by the unscaled norm. Computed with the primitives for one truth.
        let (codes, scale) = &self.members[b as usize];
        let dot = dot_i8_asymmetric(query, codes, *scale);
        let norm = norm_i8(codes, *scale);
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let denom = q_norm * norm;
        if denom == 0.0 {
            return if dot == 0.0 { 0.0 } else { 1.0 };
        }
        let cos = (dot / denom).clamp(-1.0, 1.0);
        1.0 - cos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn f32_self_distance_is_zero() {
        let s = F32CosineSpace::new(vec![vec![0.6, 0.8], vec![1.0, 0.0]]);
        assert!(approx(s.pair_distance(0, 0), 0.0));
        assert!(approx(s.query_distance(s.vector(0), 0), 0.0));
    }

    #[test]
    fn f32_orthogonal_is_one() {
        let s = F32CosineSpace::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert!(approx(s.pair_distance(0, 1), 1.0));
    }

    #[test]
    fn f32_antipodal_is_two() {
        let s = F32CosineSpace::new(vec![vec![1.0, 0.0], vec![-1.0, 0.0]]);
        assert!(approx(s.pair_distance(0, 1), 2.0));
    }

    #[test]
    fn f32_scale_invariant() {
        // Same direction, different magnitude → distance 0.
        let s = F32CosineSpace::new(vec![vec![3.0, 4.0], vec![6.0, 8.0]]);
        assert!(approx(s.pair_distance(0, 1), 0.0));
    }

    #[test]
    fn int8_matches_f32_cosine_within_quant_error() {
        // int8 cosine should track the f32 cosine to within quantization error.
        let v: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                let a = (i as f32) * 0.13 - 0.3;
                let b = (i as f32) * 0.07 + 0.1;
                vec![a, b, a - b, a + b]
            })
            .collect();
        let f32s = F32CosineSpace::new(v.clone());
        let members: Vec<(Vec<i8>, f32)> = v.iter().map(|x| namidb_core::quantize::quantize_i8(x)).collect();
        let i8s = Int8Space::new(members);

        for a in 0..6 {
            for b in 0..6 {
                let d_f = f32s.pair_distance(a as u32, b as u32);
                let d_i = i8s.pair_distance(a as u32, b as u32);
                // Quantization perturbs directions slightly; allow generous slack.
                assert!(
                    (d_f - d_i).abs() < 0.05,
                    "pair ({a},{b}): f32={d_f:.4} int8={d_i:.4}"
                );
            }
        }
    }

    #[test]
    fn int8_zero_vector_is_finite() {
        let s = Int8Space::new(vec![(vec![0, 0, 0], 0.0), (vec![1, -1, 2], 0.5)]);
        assert!(s.pair_distance(0, 1).is_finite());
        assert!(s.query_distance(&[0.1, 0.2, 0.3], 0).is_finite());
        // zero vs zero → distance 0.
        assert_eq!(s.pair_distance(0, 0), 0.0);
    }
}
