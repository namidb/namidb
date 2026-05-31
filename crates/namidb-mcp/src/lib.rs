//! Local MCP server over a NamiDB graph namespace.
//!
//! Speaks the Model Context Protocol (JSON-RPC 2.0 over newline-delimited
//! stdio) so an agent like Claude Code can query a graph with real traversals
//! instead of grepping flat files. Pointed at a namespace where a markdown
//! vault was loaded (see `namidb-markdown` / `namidb load-vault`), it exposes
//! read-only tools: list/get notes, backlinks, neighbors, orphans, full-text
//! substring search, tag tools (list tags, notes by tag, tags of a note), and
//! an escape-hatch read-only `cypher` tool.
//!
//! This is the single-user local server. Multi-tenant hosting belongs in the
//! cloud layer and must be weighed against the license's anti-DBaaS grant.
//!
//! The dispatch surface ([`Server::dispatch`]) is transport-free so it can be
//! unit tested without wiring real stdio; [`serve_stdio`] is the thin I/O loop
//! the binary runs.

#![warn(rust_2018_idioms)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

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
}

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
                p.insert("tag".to_string(), RuntimeValue::String(tag));
                (
                    "MATCH (n:Note)-[:TAGGED]->(t:Tag) WHERE t.name = $tag \
                     RETURN n.title AS title, n.path AS path ORDER BY n.title"
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
            "description": "Notes carrying the given tag (incoming `:TAGGED` edges). Tag names are matched exactly (case-sensitive).",
            "inputSchema": {
                "type": "object",
                "properties": { "tag": { "type": "string", "description": "Exact tag name (without '#')" } },
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
            "name": "cypher",
            "description": "Run an arbitrary read-only Cypher query against the graph (nodes `:Note`/`:Tag`, edges `:LINKS_TO`/`:EMBEDS`/`:TAGGED`). Write queries are rejected.",
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
    }
}

fn node_to_json(n: &NodeValue) -> Value {
    json!({
        "id": n.id.to_string(),
        "label": n.label,
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
        assert_eq!(list["tools"].as_array().unwrap().len(), 10);
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
    async fn unknown_method_is_method_not_found() {
        let server = Server::open("memory://mcp-unknown").await.unwrap();
        let err = server
            .dispatch("does/not/exist", &Value::Null)
            .await
            .unwrap_err();
        assert_eq!(err.code, -32601);
    }
}
