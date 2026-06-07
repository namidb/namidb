//! A pluggable text embedder, with a dependency-free default.
//!
//! Phase 1 ingest needs *some* way to turn note text into a vector so the
//! `cosine_similarity` query path and the MCP `vector_search` tool have data to
//! rank. The default [`HashingEmbedder`] is deterministic, offline, and adds no
//! dependency: it applies the signed hashing trick over word tokens and
//! L2-normalizes the result, so cosine similarity reflects shared vocabulary.
//!
//! It is lexical, not semantic: it matches notes that share words, not notes
//! that share meaning. Swap in a real model/API embedder by implementing
//! [`Embedder`] and passing it via [`crate::LoadOptions::embedder`]; the stored
//! vectors and the whole query path stay the same, only the numbers improve.

use std::fmt::Debug;

/// Turn text into a fixed-dimension embedding vector.
pub trait Embedder: Debug + Send + Sync {
    /// The dimension of every vector this embedder produces.
    fn dim(&self) -> usize;
    /// Embed `text` into a `dim()`-length vector. Empty/whitespace-only text
    /// yields a zero vector (cosine similarity against it is undefined and the
    /// query layer surfaces that as NULL).
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// 256 dimensions: low token-collision rate for note-sized text while staying
/// small to store alongside each node and fast to scan.
pub const DEFAULT_EMBED_DIM: usize = 256;

/// Default embedder: the signed hashing trick over word tokens, L2-normalized.
/// Deterministic and dependency-free; lexical similarity only.
#[derive(Debug, Clone)]
pub struct HashingEmbedder {
    dim: usize,
}

impl HashingEmbedder {
    /// A new embedder producing `dim`-length vectors. `dim` is clamped to at
    /// least 1.
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self::new(DEFAULT_EMBED_DIM)
    }
}

impl Embedder for HashingEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for token in tokenize(text) {
            let h = fnv1a(token.as_bytes());
            let bucket = (h % self.dim as u64) as usize;
            // A separate bit picks the sign so colliding tokens can cancel
            // rather than always reinforce (the signed hashing trick), which
            // keeps the dot product a less biased similarity estimate.
            let sign = if (h >> 63) & 1 == 1 { 1.0 } else { -1.0 };
            v[bucket] += sign;
        }
        l2_normalize(&mut v);
        v
    }
}

/// Lowercase alphanumeric word tokens. Splitting on every non-alphanumeric char
/// keeps markdown punctuation, wikilink brackets and code fences out of the
/// vocabulary.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
}

/// FNV-1a, 64-bit. Small, fast, and stable across runs and platforms (no random
/// seed), so embeddings stored on one load stay comparable on the next.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        // Both vectors are unit-norm, so the dot product is the cosine.
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    #[test]
    fn dim_and_determinism() {
        let e = HashingEmbedder::default();
        assert_eq!(e.dim(), DEFAULT_EMBED_DIM);
        let a = e.embed("the quick brown fox");
        let b = e.embed("the quick brown fox");
        assert_eq!(a, b, "same text must embed identically");
        assert_eq!(a.len(), DEFAULT_EMBED_DIM);
    }

    #[test]
    fn nonempty_text_is_unit_norm() {
        let e = HashingEmbedder::default();
        let v = e.embed("graph database on object storage");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn empty_text_is_zero_vector() {
        let e = HashingEmbedder::default();
        let v = e.embed("   \n  ");
        assert_eq!(v.len(), DEFAULT_EMBED_DIM);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn shared_vocabulary_is_closer_than_disjoint() {
        let e = HashingEmbedder::default();
        let base = e.embed("rust graph database vector search");
        let near = e.embed("vector search in a rust graph database engine");
        let far = e.embed("banana smoothie recipe with yogurt");
        assert!(
            cosine(&base, &near) > cosine(&base, &far),
            "a note sharing vocabulary must rank closer than a disjoint one"
        );
    }

    #[test]
    fn custom_dimension_is_respected() {
        let e = HashingEmbedder::new(64);
        assert_eq!(e.dim(), 64);
        assert_eq!(e.embed("hello world").len(), 64);
    }
}
