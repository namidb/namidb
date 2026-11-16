//! `live-load`: a write workload shaped like a document-graph ingestion
//! service, followed by an embedding backfill and search.
//!
//! The shape is the one that stresses incremental search indexes hardest:
//! one client sends one `UNWIND $rows AS r MERGE (x:Label {id: r.id})
//! SET x += r WITH x, r MATCH (p:Parent {id: r.parent}) MERGE (p)-[:REL]->(x)`
//! batch at a time, with a short pause between batches, for tens of minutes,
//! into labels that carry full-text indexes over kilobytes of text and
//! non-unique RANGE indexes on `id`. Some indexes are created after their
//! label already holds data. A second phase then sets one embedding per
//! existing full-text-indexed node (`MATCH ... SET a.embedding = r.v`): only
//! the vector changes.
//!
//! Phases (`--phase`):
//!
//! - `text`: the time-bounded text ingestion. Labels load in sequence
//!   (`Collection`, `Document`, `Passage`, `Revision`); each stops at its row target
//!   or at its share of `--text-minutes`, whichever comes first.
//! - `vectors`: `--vectors` embedding-only writes onto the existing `Document`
//!   nodes (`UNWIND $rows AS r MATCH (a:Document {id:r.id}) SET a.embedding =
//!   r.v`), cycling over them when there are more vectors than documents,
//!   capped at `--vector-minutes`.
//! - `search`: vector recall@k / latency against brute-force ground truth over
//!   the vectors actually loaded, plus BM25 latency on each full-text index.
//!
//! Every request is appended to `<out>/requests.csv`
//! (`epoch_ms,phase,label,rows,latency_ms,status,attempts`), and each phase
//! writes `<out>/<phase>.json`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};

use crate::http_load::VectorGen;

/// Which phase of `live-load` to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LivePhase {
    Text,
    Vectors,
    Search,
}

#[derive(Debug, Clone)]
pub struct LiveConfig {
    pub url: String,
    pub phase: LivePhase,
    pub out: PathBuf,
    pub text_minutes: f64,
    pub batch: usize,
    pub pause_ms: u64,
    pub vectors: usize,
    pub vector_batch: usize,
    pub vector_minutes: f64,
    pub dim: usize,
    pub clusters: usize,
    pub queries: usize,
    pub k: usize,
    pub ef: usize,
    pub seed: u64,
}

/// One label of the text phase.
struct LabelPlan {
    label: &'static str,
    prefix: &'static str,
    /// Row target.
    rows: usize,
    /// Cumulative share of `--text-minutes` at which this label stops.
    until_share: f64,
    text_bytes: usize,
    parent: Option<(&'static str, &'static str, &'static str, &'static str)>,
    /// DDL issued when this label starts (after earlier labels hold data).
    ddl: &'static [&'static str],
}

const PLAN: [LabelPlan; 4] = [
    LabelPlan {
        label: "Collection",
        prefix: "coll",
        rows: 10_000,
        until_share: 0.08,
        text_bytes: 2000,
        parent: None,
        ddl: &[
            "CREATE INDEX collection_id IF NOT EXISTS FOR (n:Collection) ON (n.id)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (n:Document) ON (n.id)",
            "CREATE INDEX passage_id IF NOT EXISTS FOR (n:Passage) ON (n.id)",
            "CREATE INDEX revision_id IF NOT EXISTS FOR (n:Revision) ON (n.id)",
            "CREATE FULLTEXT INDEX collection_ft IF NOT EXISTS FOR (n:Collection) ON EACH [n.title, n.body]",
        ],
    },
    LabelPlan {
        label: "Document",
        prefix: "doc",
        rows: 60_000,
        until_share: 0.35,
        text_bytes: 3000,
        parent: Some(("Collection", "coll", "collection_id", "HAS_DOCUMENT")),
        ddl: &["CREATE FULLTEXT INDEX document_ft IF NOT EXISTS FOR (n:Document) ON EACH [n.body]"],
    },
    LabelPlan {
        label: "Passage",
        prefix: "passage",
        rows: 100_000,
        until_share: 0.75,
        text_bytes: 2000,
        parent: Some(("Document", "doc", "document_id", "HAS_PASSAGE")),
        ddl: &[
            // A reference-key index added once `Document` holds data changes
            // document_ft's native-filter set: its generation must be rebuilt.
            "CREATE INDEX document_collection IF NOT EXISTS FOR (n:Document) ON (n.collection_id)",
            "CREATE FULLTEXT INDEX passage_ft IF NOT EXISTS FOR (n:Passage) ON EACH [n.body]",
            // Created once `Document` holds data, like the full-text one above.
            "CREATE VECTOR INDEX document_emb IF NOT EXISTS ON :Document(embedding) METRIC cosine DIMENSION {dim}",
        ],
    },
    LabelPlan {
        label: "Revision",
        prefix: "rev",
        rows: 1_000_000,
        until_share: 1.0,
        text_bytes: 3000,
        parent: Some(("Document", "doc", "document_id", "HAS_REVISION")),
        ddl: &[
            "CREATE INDEX passage_document IF NOT EXISTS FOR (n:Passage) ON (n.document_id)",
            "CREATE FULLTEXT INDEX revision_ft IF NOT EXISTS FOR (n:Revision) ON EACH [n.body]",
        ],
    },
];

const FULLTEXT: [(&str, &str); 4] = [
    ("Collection", "body"),
    ("Document", "body"),
    ("Passage", "body"),
    ("Revision", "body"),
];

/// A Zipf-distributed vocabulary of pseudo-words, so posting lists have the
/// long tail of real prose (a few very common terms, very many rare ones).
struct Vocabulary {
    words: Vec<String>,
    cumulative: Vec<f64>,
}

impl Vocabulary {
    fn new(size: usize, seed: u64) -> Self {
        const SYLLABLES: [&str; 32] = [
            "la", "de", "ta", "ri", "con", "es", "ma", "no", "ci", "on", "tra", "per", "ven", "sa",
            "lo", "mi", "re", "gi", "pu", "dor", "ar", "ti", "cu", "bo", "len", "fa", "pro", "ge",
            "ni", "vas", "qui", "mes",
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x766f_6361_62);
        let mut words = Vec::with_capacity(size);
        let mut seen = std::collections::HashSet::with_capacity(size);
        while words.len() < size {
            let syllables = rng.gen_range(2..=5);
            let word: String = (0..syllables)
                .map(|_| SYLLABLES[rng.gen_range(0..SYLLABLES.len())])
                .collect();
            if seen.insert(word.clone()) {
                words.push(word);
            }
        }
        let mut total = 0.0;
        let cumulative = (1..=size)
            .map(|rank| {
                total += 1.0 / (rank as f64).powf(1.07);
                total
            })
            .collect();
        Self { words, cumulative }
    }

    fn sample(&self, rng: &mut ChaCha8Rng) -> &str {
        let total = *self.cumulative.last().expect("non-empty vocabulary");
        let target = rng.gen::<f64>() * total;
        let rank = self.cumulative.partition_point(|&c| c < target);
        &self.words[rank.min(self.words.len() - 1)]
    }

    fn text(&self, rng: &mut ChaCha8Rng, bytes: usize) -> String {
        let mut out = String::with_capacity(bytes + 16);
        while out.len() < bytes {
            out.push_str(self.sample(rng));
            out.push(' ');
        }
        out.truncate(bytes);
        out
    }
}

struct Client {
    http: reqwest::Client,
    endpoint: String,
    log: std::sync::Mutex<std::fs::File>,
}

struct Outcome {
    latency_ms: f64,
    attempts: usize,
    body: Value,
}

impl Client {
    fn new(config: &LiveConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.out)?;
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(config.out.join("requests.csv"))?;
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(3600))
                .build()?,
            endpoint: format!("{}/v0/cypher", config.url.trim_end_matches('/')),
            log: std::sync::Mutex::new(log),
        })
    }

    async fn post(&self, query: &str, params: &Value) -> Result<(u16, Value)> {
        let body = serde_json::to_vec(&json!({ "query": query, "params": params }))?;
        let response = self
            .http
            .post(&self.endpoint)
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await?;
        let status = response.status().as_u16();
        let text = response.text().await?;
        Ok((
            status,
            serde_json::from_str(&text).unwrap_or(Value::String(text)),
        ))
    }

    /// Send until committed, retrying 503s and transport errors like a client
    /// library would. The latency is the wall time of the whole exchange.
    async fn commit(
        &self,
        phase: &str,
        label: &str,
        rows: usize,
        query: &str,
        params: &Value,
    ) -> Result<Outcome> {
        let started = Instant::now();
        let epoch_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let mut attempts = 0;
        loop {
            attempts += 1;
            let failure = match self.post(query, params).await {
                Ok((200, body)) => {
                    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
                    writeln!(
                        self.log.lock().unwrap(),
                        "{epoch_ms},{phase},{label},{rows},{latency_ms:.1},200,{attempts}"
                    )?;
                    return Ok(Outcome {
                        latency_ms,
                        attempts,
                        body,
                    });
                }
                Ok((503, body)) => format!("HTTP 503 {body}"),
                // The server stopped the query at its memory ceiling: the
                // batch did not apply and is safe to resend. Log the failed
                // attempt so the report counts it.
                Ok((500, body)) if body.to_string().contains("memory") => {
                    writeln!(
                        self.log.lock().unwrap(),
                        "{epoch_ms},{phase},{label},{rows},{:.1},500,{attempts}",
                        started.elapsed().as_secs_f64() * 1000.0
                    )?;
                    format!("HTTP 500 {}", truncate(&body.to_string()))
                }
                Ok((status, body)) => {
                    writeln!(
                        self.log.lock().unwrap(),
                        "{epoch_ms},{phase},{label},{rows},{:.1},{status},{attempts}",
                        started.elapsed().as_secs_f64() * 1000.0
                    )?;
                    bail!("HTTP {status} for {label}: {}", truncate(&body.to_string()));
                }
                Err(error) => error.to_string(),
            };
            if attempts >= 200 {
                bail!("{label}: gave up after {attempts} attempts: {failure}");
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

fn truncate(s: &str) -> String {
    s.chars().take(400).collect()
}

#[derive(Default)]
struct Latencies(Vec<f64>);

impl Latencies {
    fn summary(&mut self) -> Value {
        self.0.sort_by(|a, b| a.total_cmp(b));
        json!({
            "requests": self.0.len(),
            "p50_ms": percentile(&self.0, 50.0),
            "p95_ms": percentile(&self.0, 95.0),
            "p99_ms": percentile(&self.0, 99.0),
            "max_ms": percentile(&self.0, 100.0),
        })
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn write_report(out: &Path, name: &str, report: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(report)?;
    std::fs::write(out.join(format!("{name}.json")), &text)?;
    println!("{text}");
    Ok(())
}

pub async fn live_load(config: LiveConfig) -> Result<()> {
    let client = Client::new(&config)?;
    match config.phase {
        LivePhase::Text => text_phase(&client, &config).await,
        LivePhase::Vectors => vector_phase(&client, &config).await,
        LivePhase::Search => search_phase(&client, &config).await,
    }
}

fn label_query(plan: &LabelPlan) -> String {
    match plan.parent {
        None => format!(
            "UNWIND $rows AS r MERGE (x:{} {{id: r.id}}) SET x += r",
            plan.label
        ),
        Some((parent, _, key, rel)) => format!(
            "UNWIND $rows AS r MERGE (x:{label} {{id: r.id}}) SET x += r \
             WITH x, r MATCH (p:{parent} {{id: r.{key}}}) MERGE (p)-[:{rel}]->(x)",
            label = plan.label
        ),
    }
}

async fn text_phase(client: &Client, config: &LiveConfig) -> Result<()> {
    let vocabulary = Vocabulary::new(60_000, config.seed);
    let budget = Duration::from_secs_f64(config.text_minutes * 60.0);
    let started = Instant::now();
    let mut loaded = [0usize; PLAN.len()];
    let mut per_label = Vec::new();
    let mut all = Latencies::default();
    let mut retried = 0usize;
    for (index, plan) in PLAN.iter().enumerate() {
        for ddl in plan.ddl {
            let ddl = ddl.replace("{dim}", &config.dim.to_string());
            let (status, body) = client.post(&ddl, &json!({})).await?;
            eprintln!(
                "ddl `{ddl}` -> HTTP {status} {}",
                truncate(&body.to_string())
            );
        }
        let query = label_query(plan);
        let deadline = budget.mul_f64(plan.until_share);
        let mut latencies = Latencies::default();
        let label_started = Instant::now();
        while loaded[index] < plan.rows && started.elapsed() < deadline {
            let first = loaded[index];
            let count = config.batch.min(plan.rows - first);
            let rows: Vec<Value> = (first..first + count)
                .map(|i| text_row(&vocabulary, config.seed, plan, i, &loaded[..index]))
                .collect();
            let outcome = client
                .commit("text", plan.label, count, &query, &json!({ "rows": rows }))
                .await?;
            retried += outcome.attempts - 1;
            latencies.0.push(outcome.latency_ms);
            all.0.push(outcome.latency_ms);
            loaded[index] += count;
            if loaded[index] % (config.batch * 20) < config.batch {
                eprintln!(
                    "  {:>8.0}s {} {}/{} last {:.0} ms",
                    started.elapsed().as_secs_f64(),
                    plan.label,
                    loaded[index],
                    plan.rows,
                    outcome.latency_ms
                );
            }
            tokio::time::sleep(Duration::from_millis(config.pause_ms)).await;
        }
        let secs = label_started.elapsed().as_secs_f64();
        let mut summary = latencies.summary();
        summary["label"] = json!(plan.label);
        summary["rows"] = json!(loaded[index]);
        summary["secs"] = json!(secs);
        summary["rows_per_sec"] = json!(loaded[index] as f64 / secs.max(1e-9));
        eprintln!("{summary}");
        per_label.push(summary);
    }
    let mut report = all.summary();
    report["labels"] = Value::Array(per_label);
    report["secs"] = json!(started.elapsed().as_secs_f64());
    report["retried_requests"] = json!(retried);
    report["rows"] = json!(loaded.iter().sum::<usize>());
    write_report(&config.out, "text", &report)
}

fn text_row(
    vocabulary: &Vocabulary,
    seed: u64,
    plan: &LabelPlan,
    i: usize,
    earlier: &[usize],
) -> Value {
    let mut rng =
        ChaCha8Rng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9) ^ fxhash(plan.label));
    // Real legal text has a few very long documents among many short ones.
    let bytes = if rng.gen_range(0..1000) == 0 {
        plan.text_bytes * 40
    } else {
        rng.gen_range(plan.text_bytes / 2..plan.text_bytes * 3 / 2)
    };
    let mut row = json!({
        "id": format!("{}-{i}", plan.prefix),
        "body": vocabulary.text(&mut rng, bytes),
        "numero": i as i64,
        "status": STATUSES[i % 3],
    });
    if plan.label == "Collection" {
        row["title"] = json!(vocabulary.text(&mut rng, 80));
    }
    if let Some((parent, parent_prefix, key, _)) = plan.parent {
        let parents = PLAN
            .iter()
            .position(|p| p.label == parent)
            .and_then(|position| earlier.get(position).copied())
            .unwrap_or(1)
            .max(1);
        row[key] = json!(format!("{parent_prefix}-{}", rng.gen_range(0..parents)));
    }
    row
}

const STATUSES: [&str; 3] = ["open", "closed", "draft"];

fn fxhash(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The embedding backfill exactly as the ingestion service sends it: only the
/// vector changes on nodes that already carry full-text-indexed text.
const VECTOR_QUERY: &str = "UNWIND $rows AS r MATCH (a:Document {id:r.id}) SET a.embedding = r.v";

/// Write `i` targets document `i % documents`; a run with more vectors than
/// documents re-sets embeddings, which is still an embedding-only write.
fn vector_target(i: usize, documents: usize) -> usize {
    i % documents
}

async fn document_count(client: &Client) -> Result<usize> {
    let outcome = client
        .commit(
            "vectors",
            "count",
            1,
            "MATCH (n:Document) RETURN count(n) AS c",
            &json!({}),
        )
        .await?;
    outcome.body["rows"][0]["c"]
        .as_u64()
        .map(|c| c as usize)
        .ok_or_else(|| anyhow!("no Document count in {}", outcome.body))
}

async fn vector_phase(client: &Client, config: &LiveConfig) -> Result<()> {
    let gen = VectorGen::new(config.dim, config.clusters, config.seed);
    let documents = document_count(client).await?;
    if documents == 0 {
        return Err(anyhow!(
            "the vectors phase needs Document nodes from the text phase"
        ));
    }
    eprintln!("vectors onto {documents} Document nodes");
    let budget = Duration::from_secs_f64(config.vector_minutes * 60.0);
    let started = Instant::now();
    let mut loaded = 0usize;
    let mut latencies = Latencies::default();
    let mut retried = 0usize;
    let mut needing_retry = 0usize;
    while loaded < config.vectors && started.elapsed() < budget {
        let count = config.vector_batch.min(config.vectors - loaded);
        let rows: Vec<Value> = (loaded..loaded + count)
            .map(|i| {
                json!({
                    "id": format!("doc-{}", vector_target(i, documents)),
                    "v": gen.vector(i),
                })
            })
            .collect();
        let outcome = client
            .commit(
                "vectors",
                "Document",
                count,
                VECTOR_QUERY,
                &json!({ "rows": rows }),
            )
            .await?;
        retried += outcome.attempts - 1;
        needing_retry += usize::from(outcome.attempts > 1);
        latencies.0.push(outcome.latency_ms);
        loaded += count;
        if loaded % (config.vector_batch * 20) < config.vector_batch {
            eprintln!(
                "  {:>8.0}s vectors {loaded}/{} last {:.0} ms",
                started.elapsed().as_secs_f64(),
                config.vectors,
                outcome.latency_ms
            );
        }
    }
    let secs = started.elapsed().as_secs_f64();
    let mut report = latencies.summary();
    report["vectors"] = json!(loaded);
    report["documents"] = json!(documents);
    report["secs"] = json!(secs);
    report["vectors_per_sec"] = json!(loaded as f64 / secs.max(1e-9));
    report["retried_requests"] = json!(retried);
    report["requests_needing_retry"] = json!(needing_retry);
    std::fs::write(
        config.out.join("vectors.loaded"),
        format!("{loaded} {documents}"),
    )?;
    write_report(&config.out, "vectors", &report)
}

async fn search_phase(client: &Client, config: &LiveConfig) -> Result<()> {
    let marker = std::fs::read_to_string(config.out.join("vectors.loaded"))
        .context("search needs the vectors phase's vectors.loaded")?;
    let mut fields = marker.split_whitespace().map(str::parse::<usize>);
    let loaded = fields
        .next()
        .ok_or_else(|| anyhow!("empty vectors.loaded"))??;
    let documents = fields
        .next()
        .ok_or_else(|| anyhow!("vectors.loaded lacks documents"))??;
    // The vector a document holds is the last one written to it.
    let last_write = |document: usize| -> Option<usize> {
        (document < loaded.min(documents))
            .then(|| document + documents * ((loaded - 1 - document) / documents))
    };
    let indexed = loaded.min(documents);
    let gen = Arc::new(VectorGen::new(config.dim, config.clusters, config.seed));
    let queries: Vec<Vec<f32>> = (0..config.queries).map(|q| gen.query(q)).collect();
    let k = config.k;
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let chunk = indexed.div_ceil(threads).max(1);
    let partials: Vec<Vec<Vec<(f32, usize)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let gen = Arc::clone(&gen);
                let queries = &queries;
                scope.spawn(move || {
                    let mut tops: Vec<Vec<(f32, usize)>> = vec![Vec::new(); queries.len()];
                    for i in (t * chunk)..((t + 1) * chunk).min(indexed) {
                        let v = gen.vector(last_write(i).expect("indexed document"));
                        for (q, query) in queries.iter().enumerate() {
                            let score: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                            let top = &mut tops[q];
                            if top.len() < k || score > top[top.len() - 1].0 {
                                top.push((score, i));
                                top.sort_by(|a, b| b.0.total_cmp(&a.0));
                                top.truncate(k);
                            }
                        }
                    }
                    tops
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let truth: Vec<Vec<usize>> = (0..queries.len())
        .map(|q| {
            let mut all: Vec<(f32, usize)> = partials.iter().flat_map(|p| p[q].clone()).collect();
            all.sort_by(|a, b| b.0.total_cmp(&a.0));
            all.truncate(k);
            all.into_iter().map(|(_, i)| i).collect()
        })
        .collect();

    let vector_query = "CALL search.vector({label: 'Document', property: 'embedding', query: $q, \
                        k: $k, ef: $ef}) YIELD node, score RETURN node.id AS id, score";
    let mut passes = Vec::new();
    for pass in ["cold", "warm"] {
        let mut latencies = Latencies::default();
        let mut recall_sum = 0.0;
        for (q, query) in queries.iter().enumerate() {
            let outcome = client
                .commit(
                    "search",
                    "vector",
                    1,
                    vector_query,
                    &json!({ "q": query, "k": k, "ef": config.ef }),
                )
                .await?;
            latencies.0.push(outcome.latency_ms);
            let rows = outcome.body["rows"]
                .as_array()
                .ok_or_else(|| anyhow!("no rows in {}", outcome.body))?;
            let got: std::collections::HashSet<usize> = rows
                .iter()
                .filter_map(|row| row["id"].as_str())
                .filter_map(|id| id.strip_prefix("doc-"))
                .filter_map(|n| n.parse().ok())
                .collect();
            let hits = truth[q].iter().filter(|i| got.contains(i)).count();
            recall_sum += hits as f64 / k as f64;
        }
        let mut summary = latencies.summary();
        summary["pass"] = json!(pass);
        summary["recall_at_k"] = json!(recall_sum / queries.len().max(1) as f64);
        eprintln!("vector search {summary}");
        passes.push(summary);
    }

    let vocabulary = Vocabulary::new(60_000, config.seed);
    let mut fulltext = Vec::new();
    for (label, property) in FULLTEXT {
        let mut latencies = Latencies::default();
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed ^ fxhash(label));
        for _ in 0..30 {
            // Mid-frequency terms: neither stop-word-like nor absent.
            let terms = format!(
                "{} {}",
                vocabulary.words[rng.gen_range(20..2000)],
                vocabulary.words[rng.gen_range(20..2000)]
            );
            let query = format!(
                "CALL search.bm25({{label: '{label}', text_property: '{property}', \
                 query: $q, k: 10}}) YIELD node, score RETURN node.id AS id, score"
            );
            let outcome = client
                .commit("search", label, 1, &query, &json!({ "q": terms }))
                .await?;
            latencies.0.push(outcome.latency_ms);
        }
        let mut summary = latencies.summary();
        summary["label"] = json!(label);
        eprintln!("bm25 {summary}");
        fulltext.push(summary);
    }
    write_report(
        &config.out,
        "search",
        &json!({
            "vectors": loaded,
            "documents": indexed,
            "ef": config.ef,
            "vector": passes,
            "bm25": fulltext
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_is_zipfian_and_deterministic() {
        let vocabulary = Vocabulary::new(5_000, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let text = vocabulary.text(&mut rng, 20_000);
        assert_eq!(text.len(), 20_000);
        let common = vocabulary.words[0].as_str();
        let count = text.split(' ').filter(|word| *word == common).count();
        assert!(count > 20, "{count}");
        let again = Vocabulary::new(5_000, 1);
        assert_eq!(again.words[..10], vocabulary.words[..10]);
    }

    #[test]
    fn rows_link_to_loaded_parents() {
        let vocabulary = Vocabulary::new(1_000, 1);
        let row = text_row(&vocabulary, 1, &PLAN[1], 5, &[10]);
        let parent = row["collection_id"].as_str().unwrap();
        let n: usize = parent.strip_prefix("coll-").unwrap().parse().unwrap();
        assert!(n < 10);
        assert!(label_query(&PLAN[1]).contains("MATCH (p:Collection {id: r.collection_id})"));
    }
}
