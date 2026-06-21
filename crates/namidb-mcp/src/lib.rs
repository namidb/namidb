//! Local MCP server over a NamiDB graph namespace.
//!
//! Speaks the Model Context Protocol (JSON-RPC 2.0 over newline-delimited
//! stdio) so an agent like Claude Code can query a graph with real traversals
//! instead of grepping flat files. Pointed at a namespace where a markdown
//! vault was loaded (see `namidb-markdown` / `namidb load-vault`), it exposes
//! read-only tools: list/get notes, backlinks, neighbors, orphans, full-text
//! substring search, semantic vector search (K-NN over stored embeddings via
//! cosine similarity), tag tools (list tags, notes by tag including nested
//! children, tags of a note, subtags of a tag), and an escape-hatch read-only
//! `cypher` tool.
//!
//! This is the single-user local server. Multi-tenant hosting belongs in the
//! cloud layer and must be weighed against the license's anti-DBaaS grant.
//!
//! The dispatch surface ([`Server::dispatch`]) is transport-free so it can be
//! unit tested without wiring real stdio; [`serve_stdio`] is the thin I/O loop
//! the binary runs.

#![warn(rust_2018_idioms)]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use namidb_markdown::Embedder;
use namidb_graph::algo::{
    degrees, label_propagation, pagerank, shortest_paths, strongly_connected_components,
    triangle_count, weakly_connected_components, Graph, PageRankOptions,
    LABEL_PROPAGATION_DEFAULT_ITERS,
};
use namidb_query::exec::{NodeValue, RelValue};
use namidb_query::{
    execute, parse as cypher_parse, plan as build_plan, Params, ParseError, Row, RuntimeValue,
    StatsCatalog,
};
use namidb_storage::{SnapshotCell, SstCache, WriterSession};

/// MCP protocol version this server reports at `initialize`.
const PROTOCOL_VERSION: &str = "2024-11-05";

/// A NamiDB-backed MCP server. Holds one writer session (the single writer)
/// behind a mutex, a shared SST cache, and a [`SnapshotCell`] that read queries
/// serve from without taking the writer lock, so an ongoing vault load or sync
/// never blocks an agent's reads.
pub struct Server {
    session: Arc<Mutex<WriterSession>>,
    cache: SstCache,
    snapshot: Arc<SnapshotCell>,
    /// One embedder for the whole server: it embeds notes on load (document
    /// side) and the `vector_search` query string (query side), so both live in
    /// the same vector space. Chosen by `embedder_from_env` (remote when
    /// configured, else the local hashing embedder).
    embedder: Arc<dyn Embedder>,
    /// Query-embedding cache for the `vector_search` hot path, keyed by
    /// `embedder_id\0text`. A query repeated in a RAG loop is embedded once,
    /// which matters most for remote embedders (one API call instead of N).
    /// Bounded; cleared wholesale when it crosses the cap. A `std` mutex over a
    /// short, await-free critical section.
    query_embeddings: std::sync::Mutex<std::collections::HashMap<String, Vec<f32>>>,
    /// The namespace's stored embedder id, fetched once and memoised: it is
    /// immutable while the server holds one embedder, so the mismatch guard
    /// need not re-query it on every search. `None` until first observed.
    stored_embedder: std::sync::Mutex<Option<String>>,
}

/// Cap for [`Server::query_embeddings`]; past this it is cleared wholesale (a
/// RAG working set is far smaller, so this rarely trips).
const QUERY_EMBED_CACHE_CAP: usize = 512;

impl Server {
    /// Open the namespace at `store_uri` (any scheme `namidb_storage::parse_uri`
    /// accepts: `memory://`, `file://`, `s3://`, `gs://`, `az://`).
    pub async fn open(store_uri: &str) -> anyhow::Result<Self> {
        let (store, paths) =
            namidb_storage::parse_uri(store_uri).map_err(|e| anyhow::anyhow!("{e}"))?;
        let session = WriterSession::open(store, paths).await?;
        let snapshot = Arc::new(SnapshotCell::new(session.owned_snapshot()));
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            cache: SstCache::new(64 * 1024 * 1024),
            snapshot,
            embedder: namidb_markdown::embedder_from_env(),
            query_embeddings: std::sync::Mutex::new(std::collections::HashMap::new()),
            stored_embedder: std::sync::Mutex::new(None),
        })
    }

    /// Publish the writer's latest committed state so subsequent reads see it.
    /// Call after every commit (`guard` is the held writer lock).
    fn publish(&self, guard: &WriterSession) {
        self.snapshot.store(guard.owned_snapshot());
    }

    /// Load a markdown vault into the namespace before serving. Mirrors the
    /// vault (prune on) so a restart over a durable store reflects the current
    /// files instead of accumulating stale notes, then commits so the graph is
    /// durable and immediately queryable.
    ///
    /// `placeholders` matches the CLI/Python flag of the same name: when set,
    /// links and embeds to a missing note become stub `:Note` nodes (marked
    /// `placeholder: true`) so unresolved references show in the graph. The
    /// note-listing tools keep these stubs out of their results; the `cypher`
    /// tool can still reach them via `WHERE n.placeholder = true`.
    pub async fn load_vault(
        &self,
        dir: &Path,
        placeholders: bool,
    ) -> anyhow::Result<namidb_markdown::VaultLoadOutcome> {
        let opts = namidb_markdown::LoadOptions {
            prune: true,
            placeholders,
            // Embed notes on load so `vector_search` does semantic KNN over the
            // vault, using the server's embedder so notes and queries share one
            // vector space.
            embedder: Some(self.embedder.clone()),
            ..Default::default()
        };
        let mut guard = self.session.lock().await;
        let outcome = namidb_markdown::load_vault(dir, &mut guard, &opts).await?;
        guard.commit_batch().await?;
        self.publish(&guard);
        Ok(outcome)
    }

    /// Spawn a background task that watches `dir` and keeps the graph synced
    /// with it, so an agent's queries reflect edits made while the server runs.
    /// Returns immediately; the task runs until the process exits. Each change
    /// takes the writer lock only for its incremental sync and commit, then
    /// republishes the snapshot, so reads (which never take that lock) keep
    /// serving throughout. The debounced batch is only a trigger: the sync
    /// re-walks and re-hashes, so a missed or coalesced event never desyncs the
    /// graph.
    pub fn watch_vault(&self, dir: &Path, placeholders: bool) -> anyhow::Result<()> {
        use notify::{RecursiveMode, Watcher};
        use notify_debouncer_full::new_debouncer;
        use std::time::Duration;

        let session = self.session.clone();
        let snapshot = self.snapshot.clone();
        let dir = dir.to_path_buf();
        let opts = namidb_markdown::LoadOptions {
            prune: true,
            placeholders,
            // Embed notes on load so `vector_search` does semantic KNN over the
            // vault, using the server's embedder so notes and queries share one
            // vector space.
            embedder: Some(self.embedder.clone()),
            ..Default::default()
        };

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut debouncer = new_debouncer(Duration::from_millis(400), None, move |res| {
            let _ = tx.send(res);
        })
        .map_err(|e| anyhow::anyhow!("watcher: {e}"))?;
        debouncer
            .watcher()
            .watch(&dir, RecursiveMode::Recursive)
            .map_err(|e| anyhow::anyhow!("watch {}: {e}", dir.display()))?;

        tokio::spawn(async move {
            // Keep the debouncer alive for the lifetime of the task; dropping it
            // stops the watch.
            let _debouncer = debouncer;
            while let Some(event) = rx.recv().await {
                let batch = match event {
                    Ok(batch) => batch,
                    Err(errs) => {
                        eprintln!("watch error: {errs:?}");
                        continue;
                    }
                };
                let _ = batch; // advisory only; the sync re-walks the vault
                let mut guard = session.lock().await;
                let out = match namidb_markdown::sync_vault(&dir, &mut guard, &opts).await {
                    Ok(out) => out,
                    Err(e) => {
                        eprintln!("watch sync failed: {e}");
                        continue;
                    }
                };
                if let Err(e) = guard.commit_batch().await {
                    eprintln!("watch sync commit failed: {e}");
                    continue;
                }
                snapshot.store(guard.owned_snapshot());
                drop(guard);
                if out.notes_added + out.notes_modified + out.notes_deleted > 0 {
                    eprintln!(
                        "watch sync: +{} ~{} -{} ={}",
                        out.notes_added, out.notes_modified, out.notes_deleted, out.notes_unchanged,
                    );
                }
            }
        });
        Ok(())
    }

    /// Handle one JSON-RPC method and return its `result` value, or an
    /// [`RpcError`]. Notifications (methods under `notifications/`) return
    /// `Ok(Value::Null)`; the caller drops the value because notifications
    /// carry no id.
    pub async fn dispatch(&self, method: &str, params: &Value) -> Result<Value, RpcError> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "namidb-mcp", "version": env!("CARGO_PKG_VERSION") },
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tool_specs() })),
            "tools/call" => self.handle_tools_call(params).await,
            m if m.starts_with("notifications/") => Ok(Value::Null),
            other => Err(RpcError::method_not_found(other)),
        }
    }

    async fn handle_tools_call(&self, params: &Value) -> Result<Value, RpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("missing 'name'"))?;
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        // Tool-level failures travel as `isError` content, not JSON-RPC errors
        // (per the MCP convention), so the model sees and can react to them.
        match self.call_tool(name, &args).await {
            Ok(text) => Ok(json!({
                "content": [{ "type": "text", "text": text }],
                "isError": false,
            })),
            Err(msg) => Ok(json!({
                "content": [{ "type": "text", "text": msg }],
                "isError": true,
            })),
        }
    }

    /// Run a named tool, returning a text payload (JSON rows) or an error
    /// message string.
    async fn call_tool(&self, name: &str, args: &Value) -> Result<String, String> {
        // vector_search embeds the query and guards against an embedder
        // mismatch, so it runs its own query rather than the shared path below.
        if name == "vector_search" {
            return self.vector_search(args).await;
        }
        if name == "hybrid_search" {
            return self.hybrid_search(args).await;
        }
        if name == "graph_algorithm" {
            return self.graph_algorithm(args).await;
        }
        let (cypher, params): (String, Params) = match name {
            "list_notes" => (
                // `placeholder IS NULL` keeps unresolved-reference stubs (which
                // have no path/body) out of the real-note listing; the `cypher`
                // tool can still reach them via `WHERE n.placeholder = true`.
                "MATCH (n:Note) WHERE n.placeholder IS NULL \
                 RETURN n.title AS title, n.path AS path ORDER BY n.title LIMIT 500"
                    .to_string(),
                Params::new(),
            ),
            "orphans" => (
                // A note is an orphan only if nothing links to or embeds it
                // (and it links/embeds nothing), so span both edge types.
                "MATCH (n:Note) WHERE NOT EXISTS((n)-[:LINKS_TO|:EMBEDS]-()) \
                 RETURN n.title AS title, n.path AS path"
                    .to_string(),
                Params::new(),
            ),
            "backlinks" => {
                let note = str_arg(args, "note")?;
                let (cond, p) = note_match("t", &note);
                (
                    // Embeds are references too, so backlinks span both types.
                    // DISTINCT because alternation keeps per-edge multiplicity:
                    // a note that both links and embeds the target has two
                    // parallel edges and would otherwise be listed twice.
                    format!(
                        "MATCH (src:Note)-[:LINKS_TO|:EMBEDS]->(t:Note) WHERE {cond} \
                         RETURN DISTINCT src.title AS title, src.path AS path"
                    ),
                    p,
                )
            }
            "neighbors" => {
                let note = str_arg(args, "note")?;
                // Hop count must be a literal in the pattern, so clamp and
                // interpolate rather than bind it as a parameter.
                let hops = args
                    .get("hops")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .clamp(1, 5);
                let (cond, p) = note_match("s", &note);
                (
                    // Exclude placeholder stubs from the returned neighbors so a
                    // dangling `[[ref]]` does not surface as a pathless note.
                    format!(
                        "MATCH (s:Note)-[:LINKS_TO|:EMBEDS*1..{hops}]-(n:Note) \
                         WHERE ({cond}) AND n.placeholder IS NULL \
                         RETURN DISTINCT n.title AS title, n.path AS path"
                    ),
                    p,
                )
            }
            "search" => {
                let text = str_arg(args, "text")?;
                let mut p = Params::new();
                p.insert("text".to_string(), RuntimeValue::String(text));
                (
                    // Stubs have no body and a kebab-cased title; keep them out
                    // of search hits.
                    "MATCH (n:Note) \
                     WHERE (n.body CONTAINS $text OR n.title CONTAINS $text) \
                       AND n.placeholder IS NULL \
                     RETURN n.title AS title, n.path AS path LIMIT 100"
                        .to_string(),
                    p,
                )
            }
            "get_note" => {
                let note = str_arg(args, "note")?;
                let (cond, p) = note_match("n", &note);
                (
                    // ORDER BY before LIMIT 1 so the winner is deterministic
                    // when more than one note matches the disjunction. The
                    // placeholder guard means a get_note on an unresolved
                    // reference returns nothing rather than a pathless stub.
                    format!(
                        "MATCH (n:Note) WHERE ({cond}) AND n.placeholder IS NULL \
                         RETURN n.title AS title, n.path AS path, n.body AS body \
                         ORDER BY n.path LIMIT 1"
                    ),
                    p,
                )
            }
            "list_tags" => (
                "MATCH (t:Tag) RETURN t.name AS tag ORDER BY t.name LIMIT 500".to_string(),
                Params::new(),
            ),
            "notes_by_tag" => {
                let raw = str_arg(args, "tag")?;
                // Tags are stored without the leading '#'; accept it either way.
                let tag = raw.strip_prefix('#').unwrap_or(&raw).to_string();
                let mut p = Params::new();
                p.insert(
                    "prefix".to_string(),
                    RuntimeValue::String(format!("{tag}/")),
                );
                p.insert("tag".to_string(), RuntimeValue::String(tag));
                (
                    // The tag itself plus anything nested under it: `area` also
                    // returns notes tagged `area/db`, matched by name prefix.
                    // DISTINCT because a note tagged both `area` and `area/db`
                    // would otherwise be listed twice.
                    "MATCH (n:Note)-[:TAGGED]->(t:Tag) \
                     WHERE t.name = $tag OR t.name STARTS WITH $prefix \
                     RETURN DISTINCT n.title AS title, n.path AS path ORDER BY n.title"
                        .to_string(),
                    p,
                )
            }
            "subtags" => {
                let raw = str_arg(args, "tag")?;
                let tag = raw.strip_prefix('#').unwrap_or(&raw).to_string();
                let mut p = Params::new();
                p.insert("tag".to_string(), RuntimeValue::String(tag));
                (
                    // Immediate child tags of the given tag (incoming
                    // `:SUBTAG_OF`), so an agent can walk the tag tree.
                    "MATCH (child:Tag)-[:SUBTAG_OF]->(t:Tag) WHERE t.name = $tag \
                     RETURN child.name AS tag ORDER BY child.name"
                        .to_string(),
                    p,
                )
            }
            "tags_of" => {
                let note = str_arg(args, "note")?;
                let (cond, p) = note_match("n", &note);
                (
                    format!(
                        "MATCH (n:Note)-[:TAGGED]->(t:Tag) WHERE {cond} \
                         RETURN DISTINCT t.name AS tag ORDER BY t.name"
                    ),
                    p,
                )
            }
            "cypher" => (str_arg(args, "query")?, Params::new()),
            other => return Err(format!("unknown tool: {other}")),
        };

        let rows = self
            .run_read_query(&cypher, &params)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())
    }

    /// Semantic K-NN: embed the query text with the server's embedder and rank
    /// nodes by cosine similarity. Guards against an embedder mismatch first so
    /// a vault embedded by a different model is refused rather than ranked
    /// wrongly.
    async fn vector_search(&self, args: &Value) -> Result<String, String> {
        let text = str_arg(args, "query")?;
        let label = ident_arg(args, "label", "Note")?;
        let property = ident_arg(args, "property", "embedding")?;
        // LIMIT takes a literal, so clamp and interpolate `k` rather than bind.
        let k = args
            .get("k")
            .and_then(Value::as_u64)
            .unwrap_or(10)
            .clamp(1, 100);
        // Optional metadata pre-filter: a Cypher predicate spliced into the
        // WHERE before ranking, so the candidate set is narrowed *before* top-k
        // rather than truncated after it. Still read-only — `run_read_query`
        // rejects writes, and the agent can already run arbitrary reads via the
        // `cypher` tool, so this grants no new reach.
        let filter = args
            .get("where")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());

        // Mismatch guard: notes carry the id of the embedder that indexed them.
        // If that differs from the server's current embedder, cosine over two
        // different spaces would be silently wrong, so refuse with guidance.
        let current = self.embedder.id();
        if let Some(stored) = self.cached_stored_embedder_id(&label, &property).await? {
            if stored != current {
                return Err(format!(
                    "this namespace was embedded with `{stored}`, but the server's \
                     embedder is `{current}`. Re-embed the vault (load it with the \
                     matching embedder) before searching."
                ));
            }
        }

        // Embed the query as a query (vs a document) in the same space as the
        // stored notes; memoised by embedder id + text.
        let vector = self.embed_query_cached(&text).await?;
        let mut p = Params::new();
        p.insert("query".to_string(), RuntimeValue::Vector(vector));

        // Pre-filter on label + "has an embedding" (and drop placeholder stubs)
        // narrows the candidate set, plus the optional user predicate, then rank
        // by cosine. `label`/`property` are validated identifiers, so
        // interpolating them is injection-safe.
        let where_extra = match filter {
            Some(f) => format!(" AND ({f})"),
            None => String::new(),
        };
        let cypher = format!(
            "MATCH (n:{label}) \
             WHERE n.{property} IS NOT NULL AND n.placeholder IS NULL{where_extra} \
             RETURN n.title AS title, n.path AS path, \
                    cosine_similarity(n.{property}, $query) AS score \
             ORDER BY score DESC LIMIT {k}"
        );
        let rows = self
            .run_read_query(&cypher, &p)
            .await
            .map_err(|e| e.to_string())?;
        serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())
    }

    /// [`Self::stored_embedder_id`] memoised. The stored id is immutable while
    /// the server holds one embedder, so it is fetched at most once. A `None`
    /// (no embeddings yet) is deliberately not cached, so the guard starts
    /// working the moment a vault is embedded in this session.
    async fn cached_stored_embedder_id(
        &self,
        label: &str,
        property: &str,
    ) -> Result<Option<String>, String> {
        let cached = self.stored_embedder.lock().unwrap().clone();
        if cached.is_some() {
            return Ok(cached);
        }
        let observed = self.stored_embedder_id(label, property).await?;
        if let Some(id) = &observed {
            *self.stored_embedder.lock().unwrap() = Some(id.clone());
        }
        Ok(observed)
    }

    /// [`Embedder::embed`] for the query side, memoised by `embedder_id\0text`
    /// so a query repeated in a RAG loop is embedded once (one API call instead
    /// of N for a remote embedder). The lock is never held across the embed.
    async fn embed_query_cached(&self, text: &str) -> Result<Vec<f32>, String> {
        let key = format!("{}\0{text}", self.embedder.id());
        let cached = self.query_embeddings.lock().unwrap().get(&key).cloned();
        if let Some(v) = cached {
            return Ok(v);
        }
        let vector = self
            .embedder
            .embed(text)
            .await
            .map_err(|e| format!("embedding the query failed: {e}"))?;
        let mut cache = self.query_embeddings.lock().unwrap();
        // Bounded: a RAG working set fits well under the cap; clear wholesale if
        // a long-running session accumulates many distinct queries.
        if cache.len() >= QUERY_EMBED_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, vector.clone());
        Ok(vector)
    }

    /// The embedder id stamped on one embedded node of `label`, if any. `None`
    /// means the namespace has no embeddings yet, or they predate stamping (in
    /// which case the mismatch guard can't fire and the search proceeds).
    async fn stored_embedder_id(
        &self,
        label: &str,
        property: &str,
    ) -> Result<Option<String>, String> {
        let cypher = format!(
            "MATCH (n:{label}) WHERE n.{property} IS NOT NULL \
             RETURN n.embedding_model AS model LIMIT 1"
        );
        let rows = self
            .run_read_query(&cypher, &Params::new())
            .await
            .map_err(|e| e.to_string())?;
        Ok(rows
            .first()
            .and_then(|r| r.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string))
    }

    /// Hybrid search: run lexical and dense channels, then fuse via Reciprocal Rank Fusion (RRF).
    /// RRF formula: `score(channel, rank) = weight / (k + rank)` where k=60 is the standard constant.
    /// Each channel is queried with `LIMIT 3 * k` to get enough candidates for fusion.
    async fn hybrid_search(&self, args: &Value) -> Result<String, String> {
        let text = str_arg(args, "query")?;
        let label = ident_arg(args, "label", "Note")?;
        let property = ident_arg(args, "property", "embedding")?;
        let k = args
            .get("k")
            .and_then(Value::as_u64)
            .unwrap_or(10)
            .clamp(1, 100);
        let lexical_weight = args
            .get("lexical_weight")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let semantic_weight = args
            .get("semantic_weight")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let filter = args
            .get("where")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());

        // Validate weights are positive
        if lexical_weight <= 0.0 || semantic_weight <= 0.0 {
            return Err("lexical_weight and semantic_weight must be positive".to_string());
        }

        // Embedder mismatch guard (same as vector_search)
        let current = self.embedder.id();
        if let Some(stored) = self.cached_stored_embedder_id(&label, &property).await? {
            if stored != current {
                return Err(format!(
                    "this namespace was embedded with `{stored}`, but the server's \
                     embedder is `{current}`. Re-embed the vault (load it with the \
                     matching embedder) before searching."
                ));
            }
        }

        // Run both channels concurrently (they are independent reads of the
        // same snapshot) instead of sequentially — halves the latency of a
        // hybrid query on a remote embedder or a large candidate set.
        let (lexical_res, semantic_res) = tokio::join!(
            self.lexical_channel(&text, &label, k, filter),
            self.semantic_channel(&text, &label, &property, k, filter),
        );
        let lexical_results = lexical_res?;
        let semantic_results = semantic_res?;

        // Fuse via RRF
        let fused = rrf_fuse(
            &lexical_results,
            &semantic_results,
            60, // k constant for RRF
            lexical_weight,
            semantic_weight,
        );

        // Return top-k results
        let top_k: Vec<_> = fused.into_iter().take(k as usize).collect();
        serde_json::to_string_pretty(&top_k).map_err(|e| e.to_string())
    }

    /// Lexical channel: substring search in title/body.
    async fn lexical_channel(
        &self,
        text: &str,
        label: &str,
        k: u64,
        filter: Option<&str>,
    ) -> Result<Vec<FusedResult>, String> {
        let candidate_limit = (3 * k) as usize;

        // 1. Allowed set: non-placeholder nodes, plus the optional caller filter.
        // Resolving the filter here (against the `n` binding) keeps the caller's
        // `where` predicate working — `CALL ... YIELD` has no WHERE clause, so the
        // filter cannot ride along with the ranking query below.
        let where_extra = match filter {
            Some(f) => format!(" AND ({f})"),
            None => String::new(),
        };
        let allowed_cypher = format!(
            "MATCH (n:{label}) WHERE n.placeholder IS NULL{where_extra} RETURN id(n) AS id"
        );
        let allowed_rows = self
            .run_read_query(&allowed_cypher, &Params::new())
            .await
            .map_err(|e| e.to_string())?;
        let allowed: std::collections::HashSet<String> = allowed_rows
            .iter()
            .filter_map(|r| r.get("id").and_then(Value::as_str).map(str::to_string))
            .collect();

        // 2. Rank the whole corpus by full BM25 with real IDF (Layer C):
        // `CALL search.bm25` builds corpus statistics (document frequency, average
        // length) and weights rare query terms above common ones — the corpus
        // signal the per-row `bm25()` scalar cannot see. `label` is a validated
        // identifier; the query text rides as a parameter.
        let mut p = Params::new();
        p.insert("text".to_string(), RuntimeValue::String(text.to_string()));
        let rank_cypher = format!(
            "CALL search.bm25({{label: '{label}', text_properties: ['body', 'title'], query: $text}}) \
             YIELD node, score \
             RETURN id(node) AS id, node.title AS title, node.path AS path, score \
             ORDER BY score DESC"
        );
        let rows = self
            .run_read_query(&rank_cypher, &p)
            .await
            .map_err(|e| e.to_string())?;

        // 3. Keep allowed ids in rank order (already score-descending), then cap.
        Ok(rows
            .into_iter()
            .filter(|r| {
                r.get("id")
                    .and_then(Value::as_str)
                    .map(|id| allowed.contains(id))
                    .unwrap_or(false)
            })
            .take(candidate_limit)
            .enumerate()
            .map(|(rank, row)| FusedResult {
                title: row.get("title").and_then(Value::as_str).unwrap_or("").to_string(),
                path: row.get("path").and_then(Value::as_str).unwrap_or("").to_string(),
                lexical_rank: Some(rank + 1), // 1-indexed
                semantic_rank: None,
                score: 0.0,
            })
            .collect())
    }

    /// Semantic channel: dense vector cosine similarity search.
    async fn semantic_channel(
        &self,
        text: &str,
        label: &str,
        property: &str,
        k: u64,
        filter: Option<&str>,
    ) -> Result<Vec<FusedResult>, String> {
        let candidate_limit = 3 * k;
        let vector = self.embed_query_cached(text).await?;
        let mut p = Params::new();
        p.insert("query".to_string(), RuntimeValue::Vector(vector));

        let where_extra = match filter {
            Some(f) => format!(" AND ({f})"),
            None => String::new(),
        };

        let cypher = format!(
            "MATCH (n:{label}) \
             WHERE n.{property} IS NOT NULL AND n.placeholder IS NULL{where_extra} \
             RETURN n.title AS title, n.path AS path, \
                    cosine_similarity(n.{property}, $query) AS score \
             ORDER BY score DESC LIMIT {candidate_limit}"
        );

        let rows = self
            .run_read_query(&cypher, &p)
            .await
            .map_err(|e| e.to_string())?;

        Ok(rows
            .into_iter()
            .enumerate()
            .map(|(rank, row)| {
                let score = row.get("score").and_then(Value::as_f64).unwrap_or(0.0);
                FusedResult {
                    title: row.get("title").and_then(Value::as_str).unwrap_or("").to_string(),
                    path: row.get("path").and_then(Value::as_str).unwrap_or("").to_string(),
                    lexical_rank: None,
                    semantic_rank: Some(rank + 1), // 1-indexed
                    score,
                }
            })
            .collect())
    }

    /// Graph algorithm tool: run an analytical kernel (WCC or PageRank) over
    /// the subgraph induced by `label` nodes. Builds an in-memory
    /// [`Graph`] from snapshot edges, runs the kernel, and returns results
    /// joined back to readable node titles/paths.
    ///
    /// This is the MCP wedge for native graph algorithms (the CALL/YIELD
    /// Cypher surface is a fast-follow). Read-only like every MCP tool.
    async fn graph_algorithm(&self, args: &Value) -> Result<String, String> {
        let algorithm = str_arg(args, "algorithm")?;
        let label = ident_arg(args, "label", "Note")?;
        // Optional edge-type allowlist. When omitted, every edge type between
        // `label` nodes is included.
        let edge_types: Vec<String> = args
            .get("edge_types")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let top_k = args
            .get("k")
            .and_then(Value::as_u64)
            .unwrap_or(10)
            .clamp(1, 1000) as usize;

        // Validate edge type names as identifiers (they interpolate into the
        // query text).
        for et in &edge_types {
            let mut chars = et.chars();
            let ok = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !ok {
                return Err("edge_types entries must be simple identifiers".to_string());
            }
        }

        // 1. Node set: id + readable title/path. Drop placeholder stubs
        // (unresolved-reference nodes with no real body/path); they are not
        // real graph members and would otherwise pollute components/ranks.
        // `n.placeholder IS NULL` is true for any node lacking the property,
        // so this is safe on labels that never set it.
        let nodes_cypher = format!(
            "MATCH (n:{label}) WHERE n.placeholder IS NULL \
             RETURN id(n) AS id, n.title AS title, n.path AS path"
        );
        let node_rows = self
            .run_read_query(&nodes_cypher, &Params::new())
            .await
            .map_err(|e| e.to_string())?;

        // id string -> (NodeId, title, path) for result joining.
        let mut id_meta: HashMap<String, (String, String)> = HashMap::new();
        for row in &node_rows {
            let id = row.get("id").and_then(Value::as_str).unwrap_or("").to_string();
            let title = row.get("title").and_then(Value::as_str).unwrap_or("").to_string();
            let path = row.get("path").and_then(Value::as_str).unwrap_or("").to_string();
            id_meta.insert(id, (title, path));
        }

        // 2. Edges: all types between `label` nodes (optionally filtered).
        let type_filter = if edge_types.is_empty() {
            String::new()
        } else {
            let alts = edge_types
                .iter()
                .map(|t| format!(":{t}"))
                .collect::<Vec<_>>()
                .join("|");
            format!("[{alts}]")
        };
        let edges_cypher = format!(
            "MATCH (a:{label})-{type_filter}->(b:{label}) \
             WHERE a.placeholder IS NULL AND b.placeholder IS NULL \
             RETURN id(a) AS src, id(b) AS dst"
        );
        let edge_rows = self
            .run_read_query(&edges_cypher, &Params::new())
            .await
            .map_err(|e| e.to_string())?;

        // 3. Build the in-memory graph.
        let mut graph = Graph::new();
        for row in &node_rows {
            if let Some(id_str) = row.get("id").and_then(Value::as_str) {
                if let Ok(nid) = namidb_core::NodeId::from_str(id_str) {
                    graph.add_node(nid);
                }
            }
        }
        for row in &edge_rows {
            let src = row.get("src").and_then(Value::as_str);
            let dst = row.get("dst").and_then(Value::as_str);
            if let (Some(s), Some(d)) = (src, dst) {
                if let (Ok(src_id), Ok(dst_id)) = (
                    namidb_core::NodeId::from_str(s),
                    namidb_core::NodeId::from_str(d),
                ) {
                    graph.add_edge(src_id, dst_id, None);
                }
            }
        }

        // 4. Run the kernel.
        let algo = algorithm.as_str();
        let result = match algo {
            "wcc" | "weakly_connected_components" => {
                let comps = weakly_connected_components(&graph);
                // Group nodes by component, then return the top-K largest.
                let components = group_top_k(&comps.assignment, &id_meta, top_k);
                json!({
                    "algorithm": "wcc",
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "component_count": comps.count,
                    "components": components,
                })
            }
            "pagerank" | "page_rank" => {
                let mut opts = PageRankOptions::default();
                if let Some(d) = args.get("damping").and_then(Value::as_f64) {
                    opts.damping = d;
                }
                if let Some(m) = args.get("max_iterations").and_then(Value::as_u64) {
                    opts.max_iterations = m as usize;
                }
                let pr = pagerank(&graph, &opts);
                let mut ranked: Vec<(String, f64)> = id_meta
                    .keys()
                    .filter_map(|id| {
                        namidb_core::NodeId::from_str(id)
                            .ok()
                            .and_then(|nid| pr.scores.get(&nid).map(|&s| (id.clone(), s)))
                    })
                    .collect();
                // Score desc, node id asc for a deterministic tie-break (so the
                // top-k truncation is reproducible when scores are equal).
                ranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                let scores: Vec<Value> = ranked
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(id, score)| {
                        id_meta.get(&id).map(|(title, path)| {
                            json!({
                                "id": id,
                                "title": title,
                                "path": path,
                                "score": score,
                            })
                        })
                    })
                    .collect();
                json!({
                    "algorithm": "pagerank",
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "iterations": pr.iterations,
                    "converged": pr.converged,
                    "scores": scores,
                })
            }
            "scc" | "strongly_connected_components" => {
                let comps = strongly_connected_components(&graph);
                let components = group_top_k(&comps.assignment, &id_meta, top_k);
                json!({
                    "algorithm": "scc",
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "component_count": comps.count,
                    "components": components,
                })
            }
            "label_propagation" | "community" => {
                let iters = args
                    .get("max_iterations")
                    .and_then(Value::as_u64)
                    .map(|m| m as usize)
                    .unwrap_or(LABEL_PROPAGATION_DEFAULT_ITERS);
                let comm = label_propagation(&graph, iters);
                let communities = group_top_k(&comm.assignment, &id_meta, top_k);
                json!({
                    "algorithm": "label_propagation",
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "community_count": comm.count,
                    "communities": communities,
                })
            }
            "degree" | "degree_centrality" => {
                let deg = degrees(&graph);
                let mut ranked: Vec<(String, usize, usize, usize)> = id_meta
                    .keys()
                    .filter_map(|id| {
                        namidb_core::NodeId::from_str(id).ok().map(|nid| {
                            let ind = deg.in_degree.get(&nid).copied().unwrap_or(0);
                            let outd = deg.out_degree.get(&nid).copied().unwrap_or(0);
                            (id.clone(), ind, outd, ind + outd)
                        })
                    })
                    .collect();
                // Total degree desc, node id asc for a deterministic tie-break.
                ranked.sort_by(|a, b| b.3.cmp(&a.3).then_with(|| a.0.cmp(&b.0)));
                let scores: Vec<Value> = ranked
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(id, ind, outd, tot)| {
                        id_meta.get(&id).map(|(title, path)| {
                            json!({
                                "id": id, "title": title, "path": path,
                                "in_degree": ind, "out_degree": outd, "degree": tot,
                            })
                        })
                    })
                    .collect();
                json!({
                    "algorithm": "degree",
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "scores": scores,
                })
            }
            "triangle_count" | "triangles" => {
                let tri = triangle_count(&graph);
                let mut ranked: Vec<(String, usize, f64)> = id_meta
                    .keys()
                    .filter_map(|id| {
                        namidb_core::NodeId::from_str(id).ok().map(|nid| {
                            let t = tri.per_node.get(&nid).copied().unwrap_or(0);
                            let c = tri.coefficient.get(&nid).copied().unwrap_or(0.0);
                            (id.clone(), t, c)
                        })
                    })
                    .collect();
                // Triangle count desc, node id asc for a deterministic tie-break.
                ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let scores: Vec<Value> = ranked
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(id, t, c)| {
                        id_meta.get(&id).map(|(title, path)| {
                            json!({
                                "id": id, "title": title, "path": path,
                                "triangles": t, "coefficient": c,
                            })
                        })
                    })
                    .collect();
                json!({
                    "algorithm": "triangle_count",
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "total_triangles": tri.total,
                    "scores": scores,
                })
            }
            "shortest_path" | "shortest_paths" => {
                let source_str = args
                    .get("source")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "shortest_path requires a `source` node id".to_string())?;
                let source = namidb_core::NodeId::from_str(source_str)
                    .map_err(|_| format!("`source` is not a valid node id: {source_str}"))?;
                // The induced subgraph carries no edge weights, so this is BFS
                // (hop count); each reachable node appears once.
                let sp = shortest_paths(&graph, source, false);
                let mut ranked: Vec<(String, f64, usize)> = id_meta
                    .keys()
                    .filter_map(|id| {
                        namidb_core::NodeId::from_str(id).ok().and_then(|nid| {
                            sp.distance.get(&nid).map(|&d| {
                                (id.clone(), d, sp.hops.get(&nid).copied().unwrap_or(0))
                            })
                        })
                    })
                    .collect();
                // Distance asc, node id asc for a deterministic tie-break.
                ranked.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                let paths: Vec<Value> = ranked
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(id, d, h)| {
                        id_meta.get(&id).map(|(title, path)| {
                            json!({
                                "id": id, "title": title, "path": path,
                                "distance": d, "hops": h,
                            })
                        })
                    })
                    .collect();
                json!({
                    "algorithm": "shortest_path",
                    "source": source_str,
                    "node_count": graph.node_count(),
                    "edge_count": graph.edge_count(),
                    "reachable": paths.len(),
                    "paths": paths,
                })
            }
            other => {
                return Err(format!(
                    "unknown algorithm `{other}`; supported: wcc, scc, pagerank, degree, \
                     triangle_count, label_propagation, shortest_path"
                ));
            }
        };
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())
    }

    /// Parse, plan, and execute a read-only Cypher query, returning rows as
    /// JSON objects. Rejects write plans.
    async fn run_read_query(&self, cypher: &str, params: &Params) -> anyhow::Result<Vec<Value>> {
        let parsed = cypher_parse(cypher).map_err(|errs| anyhow::anyhow!(fmt_parse_errs(&errs)))?;
        // Serve from the published snapshot rather than the writer lock, so a
        // concurrent load/sync (which holds the lock to write) never blocks a
        // read. The snapshot is a consistent committed view.
        let owned = self.snapshot.load();
        let snap = owned.borrow().with_cache(self.cache.clone());
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let plan = build_plan(&parsed, &catalog).map_err(|e| anyhow::anyhow!("{e}"))?;
        if plan.contains_write() {
            anyhow::bail!("this MCP server is read-only; write queries are rejected");
        }
        let rows = execute(&plan, &snap, params)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(rows.iter().map(row_to_json).collect())
    }
}

/// Drive the stdio JSON-RPC loop until EOF. Reads one JSON message per line,
/// writes one response line per request, and stays silent for notifications.
pub async fn serve_stdio(server: Server) -> anyhow::Result<()> {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(response) = handle_line(&server, line).await {
            let mut bytes = serde_json::to_vec(&response)?;
            bytes.push(b'\n');
            stdout.write_all(&bytes).await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct Incoming {
    /// Absent (or null) for notifications; present for requests.
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

async fn handle_line(server: &Server, line: &str) -> Option<Value> {
    let incoming: Incoming = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return Some(error_response(
                Value::Null,
                -32700,
                &format!("parse error: {e}"),
            ))
        }
    };
    let result = server.dispatch(&incoming.method, &incoming.params).await;
    // No id means a notification: never reply, even on error.
    let id = incoming.id?;
    Some(match result {
        Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
        Err(err) => error_response(id, err.code, &err.message),
    })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// JSON-RPC error returned by [`Server::dispatch`].
#[derive(Debug, Clone)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

impl RpcError {
    fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
        }
    }
    fn invalid_params(message: &str) -> Self {
        Self {
            code: -32602,
            message: message.to_string(),
        }
    }
}

fn str_arg(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing required string argument '{key}'"))
}

/// Read an optional identifier argument (a label or property name), falling
/// back to `default`. Restricted to `[A-Za-z_][A-Za-z0-9_]*` so it can be
/// interpolated into the query text without opening an injection hole.
fn ident_arg(args: &Value, key: &str, default: &str) -> Result<String, String> {
    let raw = args.get(key).and_then(Value::as_str).unwrap_or(default);
    let mut chars = raw.chars();
    let ok = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if ok {
        Ok(raw.to_string())
    } else {
        Err(format!(
            "'{key}' must be a simple identifier (letters, digits, underscore)"
        ))
    }
}

/// Build a `(predicate, params)` pair resolving the note bound as `var` by
/// name: the normalized key, the exact title, or the path. `var` is a fixed
/// pattern variable ("t" / "s" / "n"), never user input.
///
/// The `key` disjunct is emitted only when the input actually normalizes to a
/// non-empty key, so an empty or punctuation-only name (which normalizes to
/// "") can't match a note whose own key is empty (e.g. a file named `-.md`).
fn note_match(var: &str, note: &str) -> (String, Params) {
    let key = namidb_markdown::normalize_key(note);
    let mut p = Params::new();
    p.insert("note".to_string(), RuntimeValue::String(note.to_string()));
    if key.is_empty() {
        (format!("{var}.title = $note OR {var}.path = $note"), p)
    } else {
        p.insert("key".to_string(), RuntimeValue::String(key));
        (
            format!("{var}.key = $key OR {var}.title = $note OR {var}.path = $note"),
            p,
        )
    }
}

fn fmt_parse_errs(errs: &[ParseError]) -> String {
    match errs.first() {
        Some(e) => format!("parse error [{:?}]: {} at {}", e.code, e.message, e.span),
        None => "parse error".to_string(),
    }
}

/// Result from a single search channel (lexical or semantic) for RRF fusion.
#[derive(Debug, Clone, serde::Serialize)]
struct FusedResult {
    /// Note title.
    title: String,
    /// Note path (key).
    path: String,
    /// Rank in lexical channel (1-indexed), if present.
    lexical_rank: Option<usize>,
    /// Rank in semantic channel (1-indexed), if present.
    semantic_rank: Option<usize>,
    /// Fused RRF score (populated after fusion).
    score: f64,
}

/// Group a component/community assignment into the `top_k` largest groups,
/// each joined to readable titles/paths (≤ 50 members shown). Shared by the
/// `wcc`, `scc`, and `label_propagation` graph algorithms.
fn group_top_k(
    assignment: &HashMap<namidb_core::NodeId, usize>,
    id_meta: &HashMap<String, (String, String)>,
    top_k: usize,
) -> Vec<Value> {
    let mut by_comp: HashMap<usize, Vec<String>> = HashMap::new();
    for id_str in id_meta.keys() {
        if let Ok(nid) = namidb_core::NodeId::from_str(id_str) {
            if let Some(&c) = assignment.get(&nid) {
                by_comp.entry(c).or_default().push(id_str.clone());
            }
        }
    }
    let mut groups: Vec<(usize, Vec<String>)> = by_comp.into_iter().collect();
    // Sort members (stable node-id strings) within each group first, so both
    // the tie-break below and the `take(50)` member cap are reproducible.
    for g in &mut groups {
        g.1.sort();
    }
    // Size desc, then smallest member id asc. The member-id tie-break is fully
    // deterministic regardless of HashMap iteration order or the arbitrary
    // component-id integers; without it, equal-sized groups would be emitted in
    // randomized order and the `take(top_k)` truncation could return a different
    // SET of groups each run, not merely a different order.
    groups.sort_by(|a, b| {
        b.1.len()
            .cmp(&a.1.len())
            .then_with(|| a.1.first().cmp(&b.1.first()))
    });
    groups
        .into_iter()
        .take(top_k)
        .map(|(_c, ids)| {
            let members: Vec<Value> = ids
                .iter()
                .take(50)
                .filter_map(|id| {
                    id_meta
                        .get(id)
                        .map(|(title, path)| json!({ "id": id, "title": title, "path": path }))
                })
                .collect();
            json!({ "size": ids.len(), "members": members })
        })
        .collect()
}

/// Reciprocal Rank Fusion (RRF): combine ranked lists from multiple search channels.
///
/// RRF formula: `score(rank) = weight / (k + rank)` where k=60 is the standard constant.
/// This rank-based fusion sidesteps comparing incompatible score ranges (BM25 vs cosine).
///
/// Returns results sorted by fused score (descending).
fn rrf_fuse(
    lexical: &[FusedResult],
    semantic: &[FusedResult],
    k_constant: usize,
    lexical_weight: f64,
    semantic_weight: f64,
) -> Vec<FusedResult> {
    let mut fused: HashMap<String, FusedResult> = HashMap::new();

    // Process lexical channel. RRF: score = weight / (k + rank), rank 1-indexed.
    for result in lexical {
        let key = format!("{}\0{}", result.title, result.path);
        let rank = result.lexical_rank.unwrap_or(usize::MAX);
        let score = lexical_weight / (k_constant as f64 + rank as f64);

        fused.entry(key).or_insert_with(|| result.clone()).score += score;
    }

    // Process semantic channel
    for result in semantic {
        let key = format!("{}\0{}", result.title, result.path);
        let rank = result.semantic_rank.unwrap_or(usize::MAX);
        let score = semantic_weight / (k_constant as f64 + rank as f64);

        fused.entry(key).or_insert_with(|| result.clone()).score += score;
    }

    // Sort by fused score descending
    let mut results: Vec<_> = fused.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn tool_specs() -> Vec<Value> {
    let note_arg = json!({
        "type": "object",
        "properties": { "note": { "type": "string", "description": "Note name (file stem), title, or path. The name match ignores case and -/_/space differences, so 'User Role', 'user-role' and 'user_role' all resolve." } },
        "required": ["note"],
    });
    vec![
        json!({
            "name": "list_notes",
            "description": "List all notes (title and path), up to 500.",
            "inputSchema": { "type": "object", "properties": {} },
        }),
        json!({
            "name": "get_note",
            "description": "Return a single note's title, path and full markdown body.",
            "inputSchema": note_arg,
        }),
        json!({
            "name": "backlinks",
            "description": "Notes that link to or embed the given note (incoming :LINKS_TO or :EMBEDS edges).",
            "inputSchema": note_arg,
        }),
        json!({
            "name": "neighbors",
            "description": "Notes within N hops of the given note via links or embeds (undirected, default 1, max 5).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "note": { "type": "string", "description": "Note name (file stem), title, or path; the name match ignores case and -/_/space differences" },
                    "hops": { "type": "integer", "minimum": 1, "maximum": 5, "description": "Hop distance (default 1)" },
                },
                "required": ["note"],
            },
        }),
        json!({
            "name": "orphans",
            "description": "Notes with no links or embeds in or out.",
            "inputSchema": { "type": "object", "properties": {} },
        }),
        json!({
            "name": "search",
            "description": "Notes whose title or body contains the given substring (case-sensitive), up to 100.",
            "inputSchema": {
                "type": "object",
                "properties": { "text": { "type": "string", "description": "Substring to search for" } },
                "required": ["text"],
            },
        }),
        json!({
            "name": "list_tags",
            "description": "List all tags in the graph (the `:Tag` nodes), up to 500.",
            "inputSchema": { "type": "object", "properties": {} },
        }),
        json!({
            "name": "notes_by_tag",
            "description": "Notes carrying the given tag or any tag nested under it, so `area` also returns notes tagged `area/db`. Tag names are case-sensitive.",
            "inputSchema": {
                "type": "object",
                "properties": { "tag": { "type": "string", "description": "Tag name (without '#'); matches the tag and its nested children" } },
                "required": ["tag"],
            },
        }),
        json!({
            "name": "subtags",
            "description": "Immediate child tags of the given tag in the nested-tag tree (incoming `:SUBTAG_OF` edges), e.g. `area` -> `area/db`, `area/web`.",
            "inputSchema": {
                "type": "object",
                "properties": { "tag": { "type": "string", "description": "Parent tag name (without '#')" } },
                "required": ["tag"],
            },
        }),
        json!({
            "name": "tags_of",
            "description": "Tags on the given note (outgoing `:TAGGED` edges).",
            "inputSchema": {
                "type": "object",
                "properties": { "note": { "type": "string", "description": "Note name (file stem), title, or path" } },
                "required": ["note"],
            },
        }),
        json!({
            "name": "vector_search",
            "description": "Semantic search: rank notes by meaning, not keywords. Give a natural-language `query`; the server embeds it with the same model used to index the vault and returns the K nearest notes by cosine similarity. Optional: `label` (default \"Note\"), `property` (default \"embedding\"), `k` (default 10, max 100), and `where` (a Cypher predicate to pre-filter on metadata before ranking). Returns title, path and score, highest first. Notes without an embedding are skipped (load the vault with embeddings enabled).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural-language search text; embedded server-side" },
                    "label": { "type": "string", "description": "Node label to search (default \"Note\")" },
                    "property": { "type": "string", "description": "Embedding property name (default \"embedding\")" },
                    "k": { "type": "integer", "minimum": 1, "maximum": 100, "description": "Number of results to return (default 10)" },
                    "where": { "type": "string", "description": "Optional Cypher predicate over the matched node `n` to pre-filter the candidate set before ranking, e.g. `n.path STARTS WITH 'work/'`. Read-only." }
                },
                "required": ["query"],
            },
        }),
        json!({
            "name": "hybrid_search",
            "description": "Hybrid search: combine lexical (keyword) and semantic (meaning) search using Reciprocal Rank Fusion (RRF). The lexical channel matches substring in title/body; the semantic channel uses vector cosine similarity. Both channels retrieve 3*k candidates and are fused by rank with configurable weights. Returns title, path, and fused RRF score, highest first. This is the recommended tool for RAG/agent workloads where both keyword precision and semantic recall matter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search text for both lexical and semantic channels" },
                    "label": { "type": "string", "description": "Node label to search (default \"Note\")" },
                    "property": { "type": "string", "description": "Embedding property name for semantic channel (default \"embedding\")" },
                    "k": { "type": "integer", "minimum": 1, "maximum": 100, "description": "Number of results to return (default 10)" },
                    "lexical_weight": { "type": "number", "description": "Weight for lexical channel in RRF fusion (default 1.0)" },
                    "semantic_weight": { "type": "number", "description": "Weight for semantic channel in RRF fusion (default 1.0)" },
                    "where": { "type": "string", "description": "Optional Cypher predicate over the matched node `n` to pre-filter the candidate set before ranking, e.g. `n.path STARTS WITH 'work/'`. Read-only." }
                },
                "required": ["query"],
            },
        }),
        json!({
            "name": "graph_algorithm",
            "description": "Run a native graph algorithm over the subgraph induced by a node label. Algorithms: `wcc` (Weakly Connected Components — undirected reachability clusters), `scc` (Strongly Connected Components — directed cycles), `pagerank` (structural importance via power iteration), `degree` (in/out/total degree centrality), `triangle_count` (triangles + local clustering coefficient), `label_propagation` (community detection), and `shortest_path` (BFS hop distances from a `source` node id). Returns algorithm-specific results joined to readable titles/paths. Read-only. The same algorithms are available in Cypher via `CALL algo.<name>()`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "algorithm": { "type": "string", "enum": ["wcc", "scc", "pagerank", "degree", "triangle_count", "label_propagation", "shortest_path"], "description": "Algorithm to run" },
                    "label": { "type": "string", "description": "Node label to build the subgraph from (default \"Note\")" },
                    "edge_types": { "type": "array", "items": { "type": "string" }, "description": "Optional allowlist of edge types to traverse (default: all edge types between label nodes)" },
                    "k": { "type": "integer", "minimum": 1, "maximum": 1000, "description": "For partitioning algos (wcc/scc/label_propagation): number of largest groups to return. For ranking algos (pagerank/degree/triangle_count/shortest_path): number of top results to return. Default 10." },
                    "damping": { "type": "number", "description": "PageRank damping factor (default 0.85)." },
                    "max_iterations": { "type": "integer", "description": "Iteration cap for pagerank (default 100) and label_propagation (default 10)." },
                    "source": { "type": "string", "description": "Required for shortest_path: the source node id to compute hop distances from." }
                },
                "required": ["algorithm"],
            },
        }),
        json!({
            "name": "cypher",
            "description": "Run an arbitrary read-only Cypher query against the graph (nodes `:Note`/`:Tag`, edges `:LINKS_TO`/`:EMBEDS`/`:TAGGED`/`:SUBTAG_OF`). Write queries are rejected.",
            "inputSchema": {
                "type": "object",
                "properties": { "query": { "type": "string", "description": "A read-only Cypher query" } },
                "required": ["query"],
            },
        }),
    ]
}

// ── RuntimeValue → JSON ────────────────────────────────────────────────

fn row_to_json(row: &Row) -> Value {
    let mut obj = serde_json::Map::new();
    for (k, v) in &row.bindings {
        obj.insert(k.clone(), rv_to_json(v));
    }
    Value::Object(obj)
}

fn props_to_json(props: &BTreeMap<String, RuntimeValue>) -> Value {
    Value::Object(
        props
            .iter()
            .map(|(k, v)| (k.clone(), rv_to_json(v)))
            .collect(),
    )
}

fn rv_to_json(v: &RuntimeValue) -> Value {
    match v {
        RuntimeValue::Null => Value::Null,
        RuntimeValue::Bool(b) => json!(b),
        RuntimeValue::Integer(i) => json!(i),
        RuntimeValue::Float(f) => json!(f),
        RuntimeValue::String(s) => json!(s),
        RuntimeValue::List(items) | RuntimeValue::Path(items) => {
            Value::Array(items.iter().map(rv_to_json).collect())
        }
        RuntimeValue::Map(m) => props_to_json(m),
        RuntimeValue::Node(n) => node_to_json(n),
        RuntimeValue::Rel(r) => rel_to_json(r),
        RuntimeValue::Date(days) => json!({ "date_days": days }),
        RuntimeValue::DateTime(micros) => json!({ "datetime_micros": micros }),
        RuntimeValue::Bytes(b) => json!({ "bytes_len": b.len() }),
        RuntimeValue::Vector(v) => json!(v),
        RuntimeValue::Vector8 { codes, scale } => {
            // Dequantize int8 to floats so clients see a float vector.
            let f: Vec<f32> = codes.iter().map(|&c| c as f32 * *scale).collect();
            json!(f)
        }
    }
}

fn node_to_json(n: &NodeValue) -> Value {
    // `label` = representative (first) for back-compat; `labels` = full set.
    json!({
        "id": n.id.to_string(),
        "label": n.labels.iter().next().cloned().unwrap_or_default(),
        "labels": n.labels.iter().cloned().collect::<Vec<String>>(),
        "properties": props_to_json(&n.properties),
    })
}

fn rel_to_json(r: &RelValue) -> Value {
    json!({
        "edge_type": r.edge_type,
        "src": r.src.to_string(),
        "dst": r.dst.to_string(),
        "properties": props_to_json(&r.properties),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::{json, Value};

    use super::*;

    fn write(dir: &Path, rel: &str, content: &str) {
        std::fs::write(dir.join(rel), content).unwrap();
    }

    async fn server_with_vault() -> Server {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "Alpha.md", "links to [[Beta]] and [[Gamma]]\n");
        write(dir.path(), "Beta.md", "back to [[Alpha]]\n");
        write(dir.path(), "Gamma.md", "leaf with a [[Missing]] link\n");
        // An island: no links in or out, so it is the sole orphan.
        write(dir.path(), "Delta.md", "an isolated note with no links\n");
        let server = Server::open("memory://mcp-test").await.unwrap();
        let outcome = server.load_vault(dir.path(), false).await.unwrap();
        assert_eq!(outcome.notes_loaded, 4);
        // Keep the tempdir alive until after the load.
        drop(dir);
        server
    }

    async fn call(server: &Server, name: &str, args: Value) -> Value {
        let res = server
            .dispatch("tools/call", &json!({ "name": name, "arguments": args }))
            .await
            .unwrap();
        assert_eq!(res["isError"], json!(false), "tool {name} errored: {res}");
        serde_json::from_str(res["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    #[tokio::test]
    async fn initialize_and_tools_list() {
        let server = Server::open("memory://mcp-init").await.unwrap();
        let init = server.dispatch("initialize", &Value::Null).await.unwrap();
        assert_eq!(init["protocolVersion"], json!(PROTOCOL_VERSION));
        let list = server.dispatch("tools/list", &Value::Null).await.unwrap();
        assert_eq!(list["tools"].as_array().unwrap().len(), 14);
    }

    #[tokio::test]
    async fn tag_tools_query_the_tag_graph() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "---\ntags: [rust, db]\n---\nbody\n");
        write(dir.path(), "B.md", "uses #rust inline\n");
        let server = Server::open("memory://mcp-tagtools").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let tags: Vec<String> = call(&server, "list_tags", json!({}))
            .await
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["tag"].as_str().map(str::to_string))
            .collect();
        assert!(tags.contains(&"rust".to_string()) && tags.contains(&"db".to_string()));

        // A leading '#' on the query is accepted (tags store without it).
        for query in ["rust", "#rust"] {
            let tagged: Vec<String> = call(&server, "notes_by_tag", json!({ "tag": query }))
                .await
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|r| r["title"].as_str().map(str::to_string))
                .collect();
            assert_eq!(tagged, vec!["A", "B"], "{query} -> both notes tagged rust");
        }

        let a_tags: Vec<String> = call(&server, "tags_of", json!({ "note": "A" }))
            .await
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["tag"].as_str().map(str::to_string))
            .collect();
        assert_eq!(a_tags, vec!["db", "rust"], "A's tags, sorted");
    }

    #[tokio::test]
    async fn nested_tags_query_by_parent_and_list_subtags() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "#area/db inline\n");
        write(dir.path(), "B.md", "#area/web inline\n");
        write(dir.path(), "C.md", "#area directly\n");
        write(dir.path(), "D.md", "#other unrelated\n");
        let server = Server::open("memory://mcp-nested-tags").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        // notes_by_tag("area") returns the note tagged #area directly plus the
        // ones tagged the nested #area/db and #area/web, but not #other.
        let by_area = call(&server, "notes_by_tag", json!({ "tag": "area" })).await;
        let mut titles: Vec<&str> = by_area
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        titles.sort();
        assert_eq!(titles, vec!["A", "B", "C"], "direct + nested, not #other");

        // subtags("area") lists the immediate children of the tag tree.
        let subs = call(&server, "subtags", json!({ "tag": "area" })).await;
        let mut names: Vec<&str> = subs
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["tag"].as_str())
            .collect();
        names.sort();
        assert_eq!(names, vec!["area/db", "area/web"]);
    }

    #[tokio::test]
    async fn backlinks_and_orphans_and_neighbors() {
        let server = server_with_vault().await;

        let backlinks = call(&server, "backlinks", json!({ "note": "Beta" })).await;
        let titles: Vec<&str> = backlinks
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert_eq!(titles, vec!["Alpha"], "only Alpha links to Beta");

        // Alpha->Gamma gives Gamma an incoming edge, so Gamma is not an
        // orphan; only Delta (no links in or out) is. Missing was never
        // created as a node, so it never appears.
        let orphans = call(&server, "orphans", json!({})).await;
        let orphan_titles: Vec<&str> = orphans
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert_eq!(orphan_titles, vec!["Delta"]);

        let neighbors = call(&server, "neighbors", json!({ "note": "Alpha", "hops": 1 })).await;
        let n: Vec<&str> = neighbors
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert!(n.contains(&"Beta") && n.contains(&"Gamma"));
    }

    #[tokio::test]
    async fn graph_algorithm_wcc_finds_components() {
        // A -> B, A -> C, B -> C  (a connected cluster); D isolated.
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[B]] and [[C]]\n");
        write(dir.path(), "B.md", "links to [[C]]\n");
        write(dir.path(), "C.md", "a leaf\n");
        write(dir.path(), "D.md", "isolated\n");
        let server = Server::open("memory://mcp-wcc").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let res = call(&server, "graph_algorithm", json!({ "algorithm": "wcc", "label": "Note" })).await;
        // Two components: {A,B,C} and {D}.
        assert_eq!(res["algorithm"], "wcc");
        assert_eq!(res["component_count"], 2);
        assert_eq!(res["edge_count"], 3);
        // The largest component has size 3 (A, B, C).
        assert_eq!(res["components"][0]["size"], 3);
        assert_eq!(res["components"][1]["size"], 1);
    }

    #[tokio::test]
    async fn graph_algorithm_pagerank_ranks_hub_highest() {
        // Same graph: C has two in-links (from A and B) → highest PageRank.
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[B]] and [[C]]\n");
        write(dir.path(), "B.md", "links to [[C]]\n");
        write(dir.path(), "C.md", "a leaf\n");
        let server = Server::open("memory://mcp-pr").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let res = call(
            &server,
            "graph_algorithm",
            json!({ "algorithm": "pagerank", "label": "Note", "k": 3 }),
        )
        .await;
        assert_eq!(res["algorithm"], "pagerank");
        assert_eq!(res["converged"], true);
        // Top-ranked node is C (two in-links).
        assert_eq!(res["scores"][0]["title"], "C");
    }

    #[tokio::test]
    async fn graph_algorithm_degree_ranks_hub_highest() {
        // C is linked from both A and B → highest total degree.
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[B]] and [[C]]\n");
        write(dir.path(), "B.md", "links to [[C]]\n");
        write(dir.path(), "C.md", "a leaf\n");
        let server = Server::open("memory://mcp-degree").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let res = call(
            &server,
            "graph_algorithm",
            json!({ "algorithm": "degree", "label": "Note", "k": 3 }),
        )
        .await;
        assert_eq!(res["algorithm"], "degree");
        // C is linked from both A and B → in_degree 2. (Total degree ties at 2
        // across all three nodes, so locate C by title rather than by rank.)
        let c = res["scores"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["title"] == "C")
            .expect("C present");
        assert_eq!(c["in_degree"], 2);
        assert_eq!(c["out_degree"], 0);
    }

    #[tokio::test]
    async fn graph_algorithm_scc_separates_acyclic_links() {
        // A -> B -> C with no back-edges: three singleton SCCs.
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[B]]\n");
        write(dir.path(), "B.md", "links to [[C]]\n");
        write(dir.path(), "C.md", "a leaf\n");
        let server = Server::open("memory://mcp-scc").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let res = call(
            &server,
            "graph_algorithm",
            json!({ "algorithm": "scc", "label": "Note" }),
        )
        .await;
        assert_eq!(res["algorithm"], "scc");
        // No directed cycles → every node is its own strongly connected component.
        assert_eq!(res["component_count"], 3);
    }

    #[tokio::test]
    async fn graph_algorithm_shortest_path_requires_source() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[B]]\n");
        write(dir.path(), "B.md", "body\n");
        let server = Server::open("memory://mcp-sp-nosrc").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let res = server
            .dispatch(
                "tools/call",
                &json!({ "name": "graph_algorithm", "arguments": { "algorithm": "shortest_path", "label": "Note" } }),
            )
            .await
            .unwrap();
        assert_eq!(res["isError"], json!(true));
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("source"), "got: {text}");
    }

    #[tokio::test]
    async fn hybrid_search_lexical_channel_runs_real_bm25() {
        // The lexical channel routes through `CALL search.bm25` (real IDF). This
        // exercises that wiring end to end: the fused result for "fox" must
        // surface the note that actually contains the rare term.
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "Fox.md", "the quick brown fox jumps over the lazy dog\n");
        write(dir.path(), "Cat.md", "the common cat sleeps all day long\n");
        write(dir.path(), "Dog.md", "the common dog barks at the moon\n");
        let server = Server::open("memory://mcp-hybrid").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let res = call(&server, "hybrid_search", json!({ "query": "fox", "k": 3 })).await;
        let titles: Vec<&str> = res
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert!(
            titles.contains(&"Fox"),
            "the note containing the rare term must surface; got {titles:?}"
        );
    }

    #[tokio::test]
    async fn graph_algorithm_rejects_unknown_algorithm() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[B]]\n");
        write(dir.path(), "B.md", "body\n");
        let server = Server::open("memory://mcp-badalgo").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let res = server
            .dispatch(
                "tools/call",
                &json!({ "name": "graph_algorithm", "arguments": { "algorithm": "bogus" } }),
            )
            .await
            .unwrap();
        assert_eq!(res["isError"], json!(true));
        let text = res["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("unknown algorithm"), "got: {text}");
    }

    #[tokio::test]
    async fn tools_resolve_notes_by_normalized_name() {
        let dir = tempfile::tempdir().unwrap();
        // Snake-cased filename, linked to from another note.
        write(dir.path(), "user_role.md", "see the founder\n");
        write(dir.path(), "Project.md", "owned by [[user_role]]\n");
        let server = Server::open("memory://mcp-resolve").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        // Caller does not know the exact stem: kebab and spaced spellings of
        // the same name must all resolve to user_role.md.
        for spelling in ["user_role", "user-role", "User Role"] {
            let got = call(&server, "get_note", json!({ "note": spelling })).await;
            let title = got.as_array().unwrap()[0]["title"].as_str().unwrap();
            assert_eq!(title, "user_role", "{spelling} should resolve");

            let back = call(&server, "backlinks", json!({ "note": spelling })).await;
            let srcs: Vec<&str> = back
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|r| r["title"].as_str())
                .collect();
            assert_eq!(srcs, vec!["Project"], "{spelling} backlinks");
        }
    }

    #[tokio::test]
    async fn empty_name_does_not_match_empty_key_note() {
        let dir = tempfile::tempdir().unwrap();
        // A punctuation-only stem normalizes to an empty key.
        write(dir.path(), "-.md", "punctuation-only stem\n");
        write(dir.path(), "Real.md", "a real note\n");
        let server = Server::open("memory://mcp-emptykey").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        // An empty / whitespace query normalizes to an empty key, which must
        // NOT fire the key disjunct and match the empty-key note. (A literal
        // "-" still legitimately matches that note by exact title, so it is
        // not part of this check.)
        for name in ["", "   "] {
            let rows = call(&server, "get_note", json!({ "note": name })).await;
            assert!(
                rows.as_array().unwrap().is_empty(),
                "name {name:?} must not resolve to the empty-key note"
            );
        }
    }

    #[tokio::test]
    async fn backlinks_include_embedders() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "embeds ![[B]]\n");
        write(dir.path(), "B.md", "b\n");
        // C both links AND embeds B: two parallel edges, must list once.
        write(dir.path(), "C.md", "link [[B]] and embed ![[B]]\n");
        let server = Server::open("memory://mcp-embed").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        // Backlinks of B span both edge types; an embedder counts, and a node
        // that both links and embeds B appears exactly once (DISTINCT).
        let back = call(&server, "backlinks", json!({ "note": "B" })).await;
        let mut srcs: Vec<&str> = back
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        srcs.sort();
        assert_eq!(
            srcs,
            vec!["A", "C"],
            "embedder counts; dual link+embed once"
        );
    }

    #[tokio::test]
    async fn placeholders_flag_creates_stub_reachable_via_cypher() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[Missing]]\n");
        let server = Server::open("memory://mcp-ph-flag").await.unwrap();
        let outcome = server.load_vault(dir.path(), true).await.unwrap();
        assert_eq!(outcome.placeholders_created, 1, "Missing stub created");

        // The escape-hatch cypher tool can still reach the stub.
        let stubs = call(
            &server,
            "cypher",
            json!({ "query": "MATCH (n:Note) WHERE n.placeholder = true RETURN n.title AS title" }),
        )
        .await;
        let titles: Vec<&str> = stubs
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert_eq!(titles, vec!["missing"], "stub reachable via cypher");
    }

    #[tokio::test]
    async fn note_listing_tools_hide_placeholder_stubs() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "links to [[Missing]]\n");
        let server = Server::open("memory://mcp-ph-guard").await.unwrap();
        server.load_vault(dir.path(), true).await.unwrap();

        // list_notes shows only the real note, not the `missing` stub.
        let notes = call(&server, "list_notes", json!({})).await;
        let listed: Vec<&str> = notes
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert_eq!(listed, vec!["A"], "stub excluded from list_notes");

        // get_note on the unresolved reference returns nothing, not a stub.
        let got = call(&server, "get_note", json!({ "note": "Missing" })).await;
        assert!(
            got.as_array().unwrap().is_empty(),
            "get_note must not return a placeholder stub"
        );

        // search on the stub's own (kebab) title finds nothing.
        let hits = call(&server, "search", json!({ "text": "missing" })).await;
        assert!(
            hits.as_array().unwrap().is_empty(),
            "search must not surface a placeholder stub"
        );

        // neighbors of A excludes the pathless stub.
        let neighbors = call(&server, "neighbors", json!({ "note": "A" })).await;
        assert!(
            neighbors.as_array().unwrap().is_empty(),
            "a dangling ref must not appear as a neighbor"
        );
    }

    #[tokio::test]
    async fn reads_see_the_latest_published_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "alpha\n");
        let server = Server::open("memory://mcp-publish").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let before = call(&server, "list_notes", json!({})).await;
        assert_eq!(before.as_array().unwrap().len(), 1, "A visible");

        // Add a note and reload; the lock-free read path must see the new
        // commit without any explicit refresh, because load_vault republishes.
        write(dir.path(), "B.md", "beta\n");
        server.load_vault(dir.path(), false).await.unwrap();

        let after = call(&server, "list_notes", json!({})).await;
        assert_eq!(
            after.as_array().unwrap().len(),
            2,
            "A and B visible after reload"
        );
    }

    #[tokio::test]
    async fn cypher_tool_rejects_writes() {
        let server = Server::open("memory://mcp-write").await.unwrap();
        let res = server
            .dispatch(
                "tools/call",
                &json!({ "name": "cypher", "arguments": { "query": "CREATE (:Note {title:'x'})" } }),
            )
            .await
            .unwrap();
        assert_eq!(res["isError"], json!(true));
        assert!(res["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("read-only"));
    }

    #[tokio::test]
    async fn vector_search_ranks_notes_by_embedding_similarity() {
        // The server embeds notes on load AND embeds the query text, both with
        // the same embedder, so semantic search works end to end: a query about
        // databases must rank the database note above the recipe.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Rust.md",
            "rust graph database engine on object storage\n",
        );
        write(
            dir.path(),
            "Cooking.md",
            "banana smoothie recipe with yogurt and honey\n",
        );
        let server = Server::open("memory://mcp-vsearch").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        let rows = call(
            &server,
            "vector_search",
            json!({ "query": "graph database storage", "k": 1 }),
        )
        .await;
        let titles: Vec<&str> = rows
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert_eq!(titles, vec!["Rust"], "the database note must rank first");
    }

    #[tokio::test]
    async fn vector_search_where_pre_filters_before_ranking() {
        // The closest note to the query is excluded by `where`, so the predicate
        // must narrow the candidate set BEFORE top-k rather than truncate after:
        // the returned note is the best match *within* the filter.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "Postgres.md",
            "rust graph database engine on object storage\n",
        );
        write(
            dir.path(),
            "Sqlite.md",
            "small embedded relational database file\n",
        );
        let server = Server::open("memory://mcp-vsearch-where").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();

        // Unfiltered, the closer note wins.
        let unfiltered = call(
            &server,
            "vector_search",
            json!({ "query": "graph database storage", "k": 1 }),
        )
        .await;
        assert_eq!(unfiltered[0]["title"], json!("Postgres"));

        // A `where` that excludes the closer note returns the best match left.
        let filtered = call(
            &server,
            "vector_search",
            json!({ "query": "graph database storage", "k": 1, "where": "n.title = 'Sqlite'" }),
        )
        .await;
        let titles: Vec<&str> = filtered
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|r| r["title"].as_str())
            .collect();
        assert_eq!(
            titles,
            vec!["Sqlite"],
            "the where must pre-filter to Sqlite"
        );
    }

    #[tokio::test]
    async fn vector_search_rejects_unsafe_label() {
        // A label carrying Cypher syntax must be rejected by the identifier
        // guard, never interpolated into the query text.
        let server = server_with_vault().await;
        let res = server
            .dispatch(
                "tools/call",
                &json!({
                    "name": "vector_search",
                    "arguments": { "query": "hello", "label": "Note) RETURN 1 //" }
                }),
            )
            .await
            .unwrap();
        assert_eq!(res["isError"], json!(true));
        assert!(res["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("identifier"));
    }

    #[tokio::test]
    async fn vector_search_requires_query_text() {
        let server = server_with_vault().await;
        let res = server
            .dispatch(
                "tools/call",
                &json!({ "name": "vector_search", "arguments": {} }),
            )
            .await
            .unwrap();
        assert_eq!(res["isError"], json!(true));
    }

    #[tokio::test]
    async fn vector_search_refuses_on_embedder_mismatch() {
        // Index a note with the server's default embedder (id hashing-v1:256),
        // then simulate a server configured with a different embedder. The
        // stamped id no longer matches, so the search must refuse rather than
        // rank across two incompatible vector spaces.
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "A.md", "rust graph database\n");
        let mut server = Server::open("memory://mcp-mismatch").await.unwrap();
        server.load_vault(dir.path(), false).await.unwrap();
        server.embedder = std::sync::Arc::new(namidb_markdown::HashingEmbedder::new(64));

        let res = server
            .dispatch(
                "tools/call",
                &json!({ "name": "vector_search", "arguments": { "query": "db" } }),
            )
            .await
            .unwrap();
        assert_eq!(res["isError"], json!(true));
        assert!(res["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Re-embed"));
    }

    #[tokio::test]
    async fn unknown_method_is_method_not_found() {
        let server = Server::open("memory://mcp-unknown").await.unwrap();
        let err = server
            .dispatch("does/not/exist", &Value::Null)
            .await
            .unwrap_err();
        assert_eq!(err.code, -32601);
    }
}
