//! A pluggable text embedder, with a dependency-free default.
//!
//! Phase 1 ingest needs *some* way to turn note text into a vector so the
//! `cosine_similarity` query path and the MCP `vector_search` tool have data to
//! rank. The default [`HashingEmbedder`] is deterministic, offline, and adds no
//! dependency: it applies the signed hashing trick over word tokens and
//! L2-normalizes the result, so cosine similarity reflects shared vocabulary.
//! It is lexical, not semantic: it matches notes that share words, not meaning.
//!
//! For real semantic search, build with `--features remote-embedder` and set
//! the `NAMIDB_EMBED_*` env vars (see [`crate::remote`]); [`embedder_from_env`]
//! then returns an API-backed embedder (OpenAI, Voyage, Cohere, Gemini, Jina)
//! and falls back to [`HashingEmbedder`] when nothing is configured.
//!
//! ## The one rule that bites
//!
//! `cosine_similarity` compares raw `f32` vectors. A namespace must be embedded
//! by exactly one embedder at one dimension: a 256-dim local namespace is not
//! comparable to a 1024-dim remote one, and even same-dim vectors from
//! different models live in different spaces. Switching the embedder, model, or
//! dimension requires a full re-embed of the vault (a prune-load).

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;

/// 256 dimensions: low token-collision rate for note-sized text while staying
/// small to store alongside each node and fast to scan.
pub const DEFAULT_EMBED_DIM: usize = 256;

/// Turn text into fixed-dimension embedding vectors.
///
/// Implementors guarantee every returned vector has length [`Embedder::dim`]
/// and is L2-normalized (so downstream cosine similarity equals the dot
/// product). Empty or whitespace-only text yields a zero vector.
///
/// `embed_batch` is the document/ingest side; the default [`Embedder::embed`]
/// is the query side. Remote embedders that distinguish the two (Voyage,
/// Cohere, Gemini, Jina) tag the request accordingly.
#[async_trait]
pub trait Embedder: Debug + Send + Sync {
    /// Dimension of every vector this embedder produces. Constant for its life.
    fn dim(&self) -> usize;

    /// Embed a batch of texts in one round-trip. The output is 1:1 with
    /// `texts` and in the same order. An empty input yields an empty output
    /// without any network call.
    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed a single query string. The default delegates to
    /// [`Embedder::embed_batch`]; remote embedders override it to tag the
    /// request as a query rather than a document.
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let one = [text.to_string()];
        let mut out = self.embed_batch(&one).await?;
        Ok(out.pop().unwrap_or_else(|| vec![0.0; self.dim()]))
    }
}

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

    /// The synchronous core. Kept separate so the async trait impl is a thin
    /// wrapper and callers that already hold a `HashingEmbedder` can embed
    /// without an executor (used by tests).
    pub fn embed_sync(&self, text: &str) -> Vec<f32> {
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

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self::new(DEFAULT_EMBED_DIM)
    }
}

#[async_trait]
impl Embedder for HashingEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_sync(t)).collect())
    }
}

/// Build the embedder selected by the environment, falling back to the local
/// [`HashingEmbedder`].
///
/// With `--features remote-embedder` and `NAMIDB_EMBEDDER=remote`, this reads
/// `NAMIDB_EMBED_PROVIDER` / `_MODEL` / `_DIM` / `_API_KEY` / `_URL` (see
/// [`crate::remote`]) and returns an API-backed embedder. If that config is
/// missing or invalid, it logs a loud warning and falls back to the local
/// 256-dim embedder, because a silent fallback would corrupt similarity in a
/// namespace the operator expected to be remote.
pub fn embedder_from_env() -> Arc<dyn Embedder> {
    #[cfg(feature = "remote-embedder")]
    {
        match crate::remote::build_remote_from_env() {
            Ok(Some(e)) => {
                tracing::info!(dim = e.dim(), "namidb: using remote embedder");
                return Arc::new(e);
            }
            Ok(None) => {} // not opted in; use the local default below
            Err(e) => tracing::warn!(
                "NAMIDB_EMBEDDER=remote but the embedder config failed: {e}. \
                 Falling back to the local {DEFAULT_EMBED_DIM}-dim HashingEmbedder. \
                 A remotely-embedded namespace is NOT comparable to a local one; \
                 fix the config and re-embed the vault if you meant to use remote."
            ),
        }
    }
    Arc::new(HashingEmbedder::default())
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

/// L2-normalize in place. No-op for a zero vector (norm 0). Shared with the
/// remote embedder, which must renormalize because some providers do not
/// renormalize Matryoshka-truncated vectors.
pub(crate) fn l2_normalize(v: &mut [f32]) {
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

    #[tokio::test]
    async fn dim_and_determinism() {
        let e = HashingEmbedder::default();
        assert_eq!(e.dim(), DEFAULT_EMBED_DIM);
        let a = e.embed("the quick brown fox").await.unwrap();
        let b = e.embed("the quick brown fox").await.unwrap();
        assert_eq!(a, b, "same text must embed identically");
        assert_eq!(a.len(), DEFAULT_EMBED_DIM);
    }

    #[tokio::test]
    async fn async_embed_matches_sync_core() {
        let e = HashingEmbedder::default();
        let via_async = e.embed("rust graph database").await.unwrap();
        let via_sync = e.embed_sync("rust graph database");
        assert_eq!(via_async, via_sync);
    }

    #[tokio::test]
    async fn batch_preserves_order_and_count() {
        let e = HashingEmbedder::default();
        let texts: Vec<String> = ["alpha", "beta", "gamma"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let batch = e.embed_batch(&texts).await.unwrap();
        assert_eq!(batch.len(), 3);
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(batch[i], e.embed_sync(t), "row {i} must match its text");
        }
    }

    #[tokio::test]
    async fn empty_batch_is_empty() {
        let e = HashingEmbedder::default();
        let out = e.embed_batch(&[]).await.unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn nonempty_text_is_unit_norm() {
        let e = HashingEmbedder::default();
        let v = e.embed_sync("graph database on object storage");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn empty_text_is_zero_vector() {
        let e = HashingEmbedder::default();
        let v = e.embed_sync("   \n  ");
        assert_eq!(v.len(), DEFAULT_EMBED_DIM);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn shared_vocabulary_is_closer_than_disjoint() {
        let e = HashingEmbedder::default();
        let base = e.embed_sync("rust graph database vector search");
        let near = e.embed_sync("vector search in a rust graph database engine");
        let far = e.embed_sync("banana smoothie recipe with yogurt");
        assert!(
            cosine(&base, &near) > cosine(&base, &far),
            "a note sharing vocabulary must rank closer than a disjoint one"
        );
    }

    #[test]
    fn custom_dimension_is_respected() {
        let e = HashingEmbedder::new(64);
        assert_eq!(e.dim(), 64);
        assert_eq!(e.embed_sync("hello world").len(), 64);
    }
}
