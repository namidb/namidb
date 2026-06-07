//! Remote, API-backed embedders for real semantic search.
//!
//! Compiled only with `--features remote-embedder`. Supports the newest text
//! embedding APIs as of mid-2026: OpenAI, Voyage 4, Cohere v4, Gemini, and
//! Jina v4. Each is a thin HTTP client over one shared [`reqwest::Client`];
//! [`RemoteEmbedder`] implements the same [`crate::embed::Embedder`] trait as
//! the local hashing embedder, so the load and query paths do not change.
//!
//! Two correctness invariants every provider shares:
//! - Output is L2-normalized here regardless of provider (some, e.g.
//!   gemini-embedding-001, do not renormalize Matryoshka-truncated vectors), so
//!   downstream cosine similarity equals the dot product.
//! - A namespace must be embedded by one provider/model/dim. Switching any of
//!   them requires a full re-embed (a prune-load); vectors from different models
//!   or dimensions are not comparable.
//!
//! ## Configuration (env)
//!
//! - `NAMIDB_EMBEDDER=remote` to opt in (else the local embedder is used).
//! - `NAMIDB_EMBED_PROVIDER` = openai | voyage | cohere | gemini | jina.
//! - `NAMIDB_EMBED_MODEL` = exact model id (default per provider).
//! - `NAMIDB_EMBED_DIM` = output dimension (default 1024; must be in the model's
//!   allowed Matryoshka set).
//! - `NAMIDB_EMBED_API_KEY`, or the provider-conventional var (OPENAI_API_KEY,
//!   VOYAGE_API_KEY, CO_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, JINA_API_KEY).
//! - `NAMIDB_EMBED_URL` = override base URL (OpenAI-compatible backends).

use std::env;
use std::time::Duration;

use anyhow::{bail, ensure, Context};
use async_trait::async_trait;
use serde::Deserialize;

use crate::embed::{l2_normalize, Embedder};

/// Universally-supported Matryoshka dimension; small enough to store per note
/// and accepted by every provider's truncation parameter.
const DEFAULT_REMOTE_DIM: usize = 1024;

/// Which side of an asymmetric retrieval model a text is. Providers that
/// distinguish documents from queries embed them differently.
#[derive(Debug, Clone, Copy)]
enum InputType {
    Document,
    Query,
}

/// The supported embedding providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    Voyage,
    Cohere,
    Gemini,
    Jina,
}

impl Provider {
    fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "openai" => Provider::OpenAi,
            "voyage" | "voyageai" => Provider::Voyage,
            "cohere" => Provider::Cohere,
            "gemini" | "google" => Provider::Gemini,
            "jina" | "jinaai" => Provider::Jina,
            other => bail!(
                "unknown NAMIDB_EMBED_PROVIDER `{other}` \
                 (expected openai|voyage|cohere|gemini|jina)"
            ),
        })
    }

    fn default_model(self) -> &'static str {
        match self {
            Provider::OpenAi => "text-embedding-3-large",
            Provider::Voyage => "voyage-4-large",
            Provider::Cohere => "embed-v4.0",
            Provider::Gemini => "gemini-embedding-001",
            Provider::Jina => "jina-embeddings-v4",
        }
    }

    /// Max inputs per HTTP call. Tuned below each provider's hard cap to stay
    /// under per-request token limits for note-sized text.
    fn max_batch(self) -> usize {
        match self {
            Provider::Cohere => 96,  // hard cap 96
            Provider::Gemini => 100, // hard cap 100 requests/call
            _ => 128,
        }
    }

    /// Provider-conventional API-key env vars, tried after `NAMIDB_EMBED_API_KEY`.
    fn key_envs(self) -> &'static [&'static str] {
        match self {
            Provider::OpenAi => &["OPENAI_API_KEY"],
            Provider::Voyage => &["VOYAGE_API_KEY"],
            Provider::Cohere => &["CO_API_KEY"],
            Provider::Gemini => &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            Provider::Jina => &["JINA_API_KEY"],
        }
    }

    /// The provider's wire value for a document vs a query. Empty for OpenAI,
    /// which has no such distinction.
    fn input_type(self, t: InputType) -> &'static str {
        match (self, t) {
            (Provider::Voyage, InputType::Document) => "document",
            (Provider::Voyage, InputType::Query) => "query",
            (Provider::Cohere, InputType::Document) => "search_document",
            (Provider::Cohere, InputType::Query) => "search_query",
            (Provider::Gemini, InputType::Document) => "RETRIEVAL_DOCUMENT",
            (Provider::Gemini, InputType::Query) => "RETRIEVAL_QUERY",
            (Provider::Jina, InputType::Document) => "retrieval.passage",
            (Provider::Jina, InputType::Query) => "retrieval.query",
            (Provider::OpenAi, _) => "",
        }
    }
}

/// An API-backed embedder. One per server/load; holds a pooled HTTP client.
pub struct RemoteEmbedder {
    client: reqwest::Client,
    provider: Provider,
    model: String,
    dim: usize,
    api_key: String,
    endpoint_override: Option<String>,
    max_retries: u32,
}

// Hand-written so the API key never lands in a log line or panic message.
impl std::fmt::Debug for RemoteEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteEmbedder")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("dim", &self.dim)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl RemoteEmbedder {
    fn new(
        provider: Provider,
        model: String,
        dim: usize,
        api_key: String,
        endpoint_override: Option<String>,
    ) -> anyhow::Result<Self> {
        ensure!(dim > 0, "embedding dimension must be positive");
        ensure!(!api_key.is_empty(), "embedding API key is empty");
        // reqwest has no default timeout; without these a hung provider would
        // stall the whole load loop forever.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("building the HTTP client")?;
        Ok(Self {
            client,
            provider,
            model,
            dim,
            api_key,
            endpoint_override,
            max_retries: 4,
        })
    }

    fn endpoint_or(&self, default: &str) -> String {
        self.endpoint_override
            .clone()
            .unwrap_or_else(|| default.to_string())
    }

    /// Build the (url, json body) for one chunk under a given input type.
    fn build_request(&self, chunk: &[String], input: InputType) -> (String, serde_json::Value) {
        use serde_json::json;
        let it = self.provider.input_type(input);
        match self.provider {
            Provider::OpenAi => (
                self.endpoint_or("https://api.openai.com/v1/embeddings"),
                json!({
                    "model": self.model,
                    "input": chunk,
                    "encoding_format": "float",
                    "dimensions": self.dim,
                }),
            ),
            Provider::Voyage => (
                self.endpoint_or("https://api.voyageai.com/v1/embeddings"),
                json!({
                    "model": self.model,
                    "input": chunk,
                    "input_type": it,
                    "output_dimension": self.dim,
                    "output_dtype": "float",
                }),
            ),
            Provider::Cohere => (
                self.endpoint_or("https://api.cohere.com/v2/embed"),
                json!({
                    "model": self.model,
                    "texts": chunk,
                    "input_type": it,
                    "embedding_types": ["float"],
                    "output_dimension": self.dim,
                }),
            ),
            Provider::Gemini => {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents",
                    self.model
                );
                let requests: Vec<_> = chunk
                    .iter()
                    .map(|t| {
                        json!({
                            "model": format!("models/{}", self.model),
                            "content": { "parts": [ { "text": t } ] },
                            "taskType": it,
                            "outputDimensionality": self.dim,
                        })
                    })
                    .collect();
                (url, json!({ "requests": requests }))
            }
            Provider::Jina => {
                let input: Vec<_> = chunk.iter().map(|t| json!({ "text": t })).collect();
                (
                    self.endpoint_or("https://api.jina.ai/v1/embeddings"),
                    json!({
                        "model": self.model,
                        "task": it,
                        "dimensions": self.dim,
                        "input": input,
                    }),
                )
            }
        }
    }

    /// Pull the float arrays out of a provider response, in input order.
    fn parse_response(&self, v: serde_json::Value) -> anyhow::Result<Vec<Vec<f32>>> {
        match self.provider {
            // OpenAI, Voyage and Jina all return `{ "data": [ { "embedding": [..] } ] }`.
            Provider::OpenAi | Provider::Voyage | Provider::Jina => {
                let r: DataResp =
                    serde_json::from_value(v).context("parsing data[].embedding response")?;
                Ok(r.data.into_iter().map(|d| d.embedding).collect())
            }
            // Cohere v2: `{ "embeddings": { "float": [[..]] } }`.
            Provider::Cohere => {
                let r: CohereResp = serde_json::from_value(v)
                    .context("parsing cohere embeddings.float response")?;
                Ok(r.embeddings.float)
            }
            // Gemini batchEmbedContents: `{ "embeddings": [ { "values": [..] } ] }`.
            Provider::Gemini => {
                let r: GeminiResp = serde_json::from_value(v)
                    .context("parsing gemini embeddings[].values response")?;
                Ok(r.embeddings.into_iter().map(|e| e.values).collect())
            }
        }
    }

    /// Enforce the per-vector invariants: exact configured dimension, and
    /// L2-normalized (so cosine == dot downstream regardless of provider).
    fn finalize(&self, mut vecs: Vec<Vec<f32>>) -> anyhow::Result<Vec<Vec<f32>>> {
        for v in &mut vecs {
            ensure!(
                v.len() == self.dim,
                "{:?} returned a {}-dim vector but {} was configured",
                self.provider,
                v.len(),
                self.dim
            );
            l2_normalize(v);
        }
        Ok(vecs)
    }

    async fn embed_chunk(
        &self,
        chunk: &[String],
        input: InputType,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let (url, body) = self.build_request(chunk, input);
        let value = self.post_with_retry(&url, &body).await?;
        let vecs = self.parse_response(value)?;
        ensure!(
            vecs.len() == chunk.len(),
            "{:?} returned {} vectors for {} inputs",
            self.provider,
            vecs.len(),
            chunk.len()
        );
        self.finalize(vecs)
    }

    async fn embed_with(
        &self,
        texts: &[String],
        input: InputType,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.provider.max_batch()) {
            out.extend(self.embed_chunk(chunk, input).await?);
        }
        Ok(out)
    }

    /// POST the body, retrying 429/5xx and connect/timeout errors with capped
    /// exponential backoff. Other 4xx (bad key, bad request) fail immediately.
    async fn post_with_retry(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let mut attempt = 0u32;
        loop {
            let req = match self.provider {
                Provider::Gemini => self
                    .client
                    .post(url)
                    .header("x-goog-api-key", &self.api_key),
                _ => self.client.post(url).bearer_auth(&self.api_key),
            };
            match req.json(body).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return resp
                            .json::<serde_json::Value>()
                            .await
                            .context("decoding embedding response JSON");
                    }
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if retryable && attempt < self.max_retries {
                        let delay = retry_after(&resp).unwrap_or_else(|| backoff(attempt));
                        attempt += 1;
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    let detail = resp.text().await.unwrap_or_default();
                    bail!(
                        "{:?} embedding request failed: HTTP {} {}",
                        self.provider,
                        status.as_u16(),
                        truncate(&detail, 300)
                    );
                }
                Err(e) if attempt < self.max_retries && (e.is_timeout() || e.is_connect()) => {
                    attempt += 1;
                    tokio::time::sleep(backoff(attempt - 1)).await;
                }
                Err(e) => return Err(e).context("sending embedding request"),
            }
        }
    }
}

#[async_trait]
impl Embedder for RemoteEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embed_with(texts, InputType::Document).await
    }

    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let one = [text.to_string()];
        let mut v = self.embed_with(&one, InputType::Query).await?;
        Ok(v.pop().unwrap_or_else(|| vec![0.0; self.dim]))
    }
}

/// Build the configured remote embedder from the environment, or `Ok(None)`
/// when `NAMIDB_EMBEDDER` is not `remote` (the caller then uses the local one).
/// Returns `Err` only when remote was requested but misconfigured.
pub fn build_remote_from_env() -> anyhow::Result<Option<RemoteEmbedder>> {
    if env::var("NAMIDB_EMBEDDER").ok().as_deref() != Some("remote") {
        return Ok(None);
    }
    let provider = Provider::parse(
        env::var("NAMIDB_EMBED_PROVIDER")
            .as_deref()
            .unwrap_or("openai"),
    )?;
    let model = env::var("NAMIDB_EMBED_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| provider.default_model().to_string());
    let dim = match env::var("NAMIDB_EMBED_DIM") {
        Ok(s) => s
            .trim()
            .parse::<usize>()
            .context("NAMIDB_EMBED_DIM must be a positive integer")?,
        Err(_) => DEFAULT_REMOTE_DIM,
    };
    let api_key = resolve_key(provider)?;
    let endpoint_override = env::var("NAMIDB_EMBED_URL").ok().filter(|s| !s.is_empty());
    Ok(Some(RemoteEmbedder::new(
        provider,
        model,
        dim,
        api_key,
        endpoint_override,
    )?))
}

fn resolve_key(provider: Provider) -> anyhow::Result<String> {
    if let Ok(k) = env::var("NAMIDB_EMBED_API_KEY") {
        if !k.is_empty() {
            return Ok(k);
        }
    }
    for var in provider.key_envs() {
        if let Ok(k) = env::var(var) {
            if !k.is_empty() {
                return Ok(k);
            }
        }
    }
    bail!(
        "no API key for {:?}: set NAMIDB_EMBED_API_KEY or {}",
        provider,
        provider.key_envs().join(" / ")
    )
}

/// Capped exponential backoff: 0.5s, 1s, 2s, 4s, ... up to 8s.
fn backoff(attempt: u32) -> Duration {
    let secs = (0.5_f64 * 2f64.powi(attempt as i32)).min(8.0);
    Duration::from_millis((secs * 1000.0) as u64)
}

/// Honor a `Retry-After: <seconds>` header when the provider sends one.
fn retry_after(resp: &reqwest::Response) -> Option<Duration> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

#[derive(Deserialize)]
struct DataResp {
    data: Vec<DataItem>,
}
#[derive(Deserialize)]
struct DataItem {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct CohereResp {
    embeddings: CohereFloats,
}
#[derive(Deserialize)]
struct CohereFloats {
    float: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct GeminiResp {
    embeddings: Vec<GeminiValues>,
}
#[derive(Deserialize)]
struct GeminiValues {
    values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedder(provider: Provider, dim: usize) -> RemoteEmbedder {
        RemoteEmbedder::new(
            provider,
            provider.default_model().into(),
            dim,
            "test-key".into(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn provider_parse_and_defaults() {
        assert_eq!(Provider::parse("OpenAI").unwrap(), Provider::OpenAi);
        assert_eq!(Provider::parse("voyageai").unwrap(), Provider::Voyage);
        assert_eq!(Provider::parse("google").unwrap(), Provider::Gemini);
        assert!(Provider::parse("nope").is_err());
        assert_eq!(Provider::Cohere.default_model(), "embed-v4.0");
        assert_eq!(Provider::Cohere.max_batch(), 96);
        assert_eq!(Provider::Gemini.max_batch(), 100);
    }

    #[test]
    fn openai_request_shape() {
        let e = embedder(Provider::OpenAi, 1024);
        let (url, body) = e.build_request(&["hi".into()], InputType::Document);
        assert_eq!(url, "https://api.openai.com/v1/embeddings");
        assert_eq!(body["dimensions"], 1024);
        assert_eq!(body["input"][0], "hi");
        assert!(body.get("input_type").is_none(), "OpenAI has no input_type");
    }

    #[test]
    fn input_type_differs_between_document_and_query() {
        let e = embedder(Provider::Cohere, 1024);
        let (_, doc) = e.build_request(&["x".into()], InputType::Document);
        let (_, qry) = e.build_request(&["x".into()], InputType::Query);
        assert_eq!(doc["input_type"], "search_document");
        assert_eq!(qry["input_type"], "search_query");
        assert_eq!(doc["texts"][0], "x"); // cohere uses `texts`, not `input`
    }

    #[test]
    fn gemini_request_uses_batch_endpoint_and_task_type() {
        let e = embedder(Provider::Gemini, 768);
        let (url, body) = e.build_request(&["a".into(), "b".into()], InputType::Query);
        assert!(url.ends_with(":batchEmbedContents"));
        assert_eq!(body["requests"].as_array().unwrap().len(), 2);
        assert_eq!(body["requests"][0]["taskType"], "RETRIEVAL_QUERY");
        assert_eq!(body["requests"][0]["outputDimensionality"], 768);
    }

    #[test]
    fn parse_openai_voyage_jina_shape() {
        let e = embedder(Provider::OpenAi, 3);
        let v: serde_json::Value = serde_json::from_str(
            r#"{"data":[{"embedding":[1.0,0.0,0.0]},{"embedding":[0.0,2.0,0.0]}]}"#,
        )
        .unwrap();
        let vecs = e.parse_response(v).unwrap();
        assert_eq!(vecs, vec![vec![1.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]]);
    }

    #[test]
    fn parse_cohere_shape() {
        let e = embedder(Provider::Cohere, 2);
        let v: serde_json::Value =
            serde_json::from_str(r#"{"embeddings":{"float":[[3.0,4.0]]}}"#).unwrap();
        assert_eq!(e.parse_response(v).unwrap(), vec![vec![3.0, 4.0]]);
    }

    #[test]
    fn parse_gemini_shape() {
        let e = embedder(Provider::Gemini, 2);
        let v: serde_json::Value =
            serde_json::from_str(r#"{"embeddings":[{"values":[6.0,8.0]}]}"#).unwrap();
        assert_eq!(e.parse_response(v).unwrap(), vec![vec![6.0, 8.0]]);
    }

    #[test]
    fn finalize_normalizes_and_checks_dim() {
        let e = embedder(Provider::OpenAi, 2);
        // [3,4] has norm 5 -> normalizes to [0.6, 0.8].
        let out = e.finalize(vec![vec![3.0, 4.0]]).unwrap();
        let norm: f32 = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        // Wrong dimension is a hard error, not a silent mismatch.
        assert!(e.finalize(vec![vec![1.0, 0.0, 0.0]]).is_err());
    }

    #[test]
    fn backoff_is_capped() {
        assert!(backoff(0) < backoff(1));
        assert_eq!(backoff(20), Duration::from_secs(8));
    }
}
