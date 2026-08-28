//! `namidb` CLI.
//!
//! Operational subcommands:
//! - `version` — build info.
//! - `namespace-check <name>` — validate a namespace identifier.
//! - `parse <cypher>` — parse a Cypher query; print round-trip form.
//! - `explain <cypher>` — parse + lower; print the logical plan tree.
//! - `run [--store <uri>] [--namespace <ns>] <cypher>` — open a
//! namespace, execute the query, print rows or `WriteOutcome`.
//! With no `--store`, runs against an ephemeral `memory://`
//! namespace. With `--store file:///path?ns=…` or any other
//! supported scheme (s3, gs, az), state is durable on the
//! configured backend.

// Cosmetic doc-rendering nit (lazy markdown list continuation); allow it.
#![allow(clippy::doc_lazy_continuation)]

use std::sync::Arc;

use clap::{Parser, Subcommand};
use namidb_core::{id::NamespaceId, value::Value as CoreValue};
use namidb_markdown::{embedder_from_env, load_vault, sync_vault, LoadOptions};
use namidb_query::{
    execute, execute_write, explain_query, explain_query_raw, explain_query_raw_verbose,
    explain_query_verbose, parse, plan as build_plan, Params, RuntimeValue, StatsCatalog,
    WriteOutcome,
};
use namidb_storage::{copy_namespace_snapshot, parse_uri, NamespacePaths, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

#[derive(Parser, Debug)]
#[command(name = "namidb", version, about = "NamiDB CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print build information.
    Version,
    /// Validate a namespace identifier.
    NamespaceCheck {
        /// The candidate namespace name.
        name: String,
    },
    /// Parse a Cypher query and print the round-trip canonical form.
    Parse {
        /// Cypher source. Wrap multi-word queries in quotes.
        query: String,
    },
    /// Lower a Cypher query and print its logical plan tree.
    /// An explicit `EXPLAIN` prefix is allowed but optional. With
    /// `--verbose` (or the `EXPLAIN VERBOSE` prefix), each operator
    /// is annotated with its estimated row count. With `--raw` (or
    /// the `EXPLAIN RAW` prefix), the optimizer is skipped and the
    /// plan is rendered exactly as the lowering produced it
    /// (RFC-011 §6.2).
    Explain {
        /// Show cardinality estimates next to each operator.
        #[arg(short, long, default_value_t = false)]
        verbose: bool,
        /// Skip the optimizer pipeline; render the lowering verbatim.
        #[arg(long, default_value_t = false)]
        raw: bool,
        /// Cypher source. Wrap multi-word queries in quotes.
        query: String,
    },
    /// Run a Cypher query against a NamiDB namespace and print rows
    /// (for read queries) or the `WriteOutcome` (for write queries).
    ///
    /// Accepts a `;`-separated multi-statement script (semicolons inside
    /// strings, backticks, and comments do not split): statements run
    /// sequentially against one session and stop at the first error, and
    /// `CREATE CONSTRAINT` / `CREATE INDEX` execute as schema commands —
    /// so a pasted schema-bootstrap script just works.
    ///
    /// Without `--store`, the command opens an ephemeral in-memory
    /// namespace whose state vanishes on exit. With `--store <uri>`,
    /// the namespace is durable on the configured backend
    /// (`file://`, `s3://`, `gs://`, `az://`, or `memory://`).
    Run {
        /// Storage URI. Examples:
        ///
        ///   memory://acme
        ///   file:///var/lib/namidb?ns=prod
        ///   s3://my-bucket/data?ns=prod&region=us-east-1
        ///   gs://my-bucket?ns=prod
        ///   az://acct/container?ns=prod
        #[arg(long)]
        store: Option<String>,
        /// Namespace name when `--store` is not supplied (defaults to
        /// `default`; ignored when `--store` is set because the URI
        /// carries its own `?ns=` parameter).
        #[arg(short, long, default_value = "default")]
        namespace: String,
        /// Cypher source. Wrap multi-word queries in quotes.
        query: String,
    },
    /// Load an Obsidian-style markdown vault as a graph: each `.md` note
    /// becomes a `:Note` node, each `[[wikilink]]` a `:LINKS_TO` edge, and
    /// YAML frontmatter becomes node properties. The note body is kept as a
    /// `body` property, so the files stay the source of truth and the graph
    /// is a derived index you can rebuild.
    ///
    /// Point `--store` at a durable backend to keep the result; without it
    /// the load runs against an ephemeral in-memory namespace (useful only to
    /// check the counts).
    LoadVault {
        /// Storage URI (see `run --help` for the scheme reference). Durable
        /// backends (`file://`, `s3://`, `gs://`, `az://`) persist the graph.
        #[arg(long)]
        store: Option<String>,
        /// Namespace name when `--store` is not supplied.
        #[arg(short, long, default_value = "default")]
        namespace: String,
        /// Node label for notes.
        #[arg(long, default_value = "Note")]
        label: String,
        /// Edge type for wikilinks.
        #[arg(long, default_value = "LINKS_TO")]
        edge_type: String,
        /// Mirror the vault: tombstone notes and links no longer present.
        /// Use when re-loading a vault that changed, so the graph stays a
        /// faithful index instead of accumulating stale nodes and edges.
        #[arg(long, default_value_t = false)]
        prune: bool,
        /// Create stub `:Note` nodes for links/embeds whose target does not
        /// exist, so unresolved references show up in the graph.
        #[arg(long, default_value_t = false)]
        placeholders: bool,
        /// Compute a text embedding for each note (title + body) and store it as
        /// an `embedding` property, so `cosine_similarity(...)` queries and the
        /// MCP `vector_search` tool can rank notes by similarity. Uses a local,
        /// deterministic, offline embedder.
        #[arg(long, default_value_t = false)]
        embed: bool,
        /// Watch the vault and re-sync incrementally on every change, keeping
        /// the graph live until interrupted (Ctrl-C). A watch always mirrors
        /// the vault, so `--prune` is implied.
        #[arg(long, default_value_t = false)]
        watch: bool,
        /// Path to the vault directory.
        path: String,
    },
    /// Copy a consistent snapshot of a namespace to another location.
    ///
    /// Pins a manifest version and copies its closure — the manifest, every
    /// SST and its side-cars, and the WAL segments still needed for recovery.
    /// All are immutable, so the snapshot is consistent by construction. The
    /// destination is left as a self-contained, openable namespace.
    ///
    /// Safe against a LIVE source: the copy pins a manifest version with a
    /// durable retention lease that the server's janitor honours, so a
    /// concurrent compaction + orphan sweep cannot reclaim pinned objects;
    /// the result is a point-in-time snapshot at the pinned version. The
    /// residual pin/sweep race fails loudly (NotFound) rather than
    /// truncating — just re-run the backup.
    Backup {
        /// Source namespace URI to back up (see `run --help` for schemes).
        #[arg(long)]
        from: String,
        /// Destination namespace URI for the snapshot (a fresh location).
        #[arg(long)]
        to: String,
        /// Manifest version to pin. Defaults to the current committed version.
        #[arg(long)]
        version: Option<u64>,
        /// Proceed even if the destination already holds a namespace.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Re-open the destination after the copy and HEAD every referenced
        /// SST/WAL object, failing if any is missing or empty.
        #[arg(long, default_value_t = false)]
        verify: bool,
    },
    /// Restore a namespace from a backup made with `backup`.
    ///
    /// The same consistent copy as `backup`, in the recovery direction. The
    /// destination must be offline (there is no fencing against a concurrent
    /// writer); restoring over an existing namespace requires `--force`.
    Restore {
        /// Backup namespace URI to restore from.
        #[arg(long)]
        from: String,
        /// Destination namespace URI to restore into.
        #[arg(long)]
        to: String,
        /// Manifest version to pin. Defaults to the current committed version.
        #[arg(long)]
        version: Option<u64>,
        /// Overwrite the destination even if it already holds a namespace.
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Re-open the destination after the copy and HEAD every referenced
        /// SST/WAL object, failing if any is missing or empty.
        #[arg(long, default_value_t = false)]
        verify: bool,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Version => {
            println!("namidb {}", env!("CARGO_PKG_VERSION"));
        }
        Cmd::NamespaceCheck { name } => {
            let ns = NamespaceId::new(&name)?;
            println!("ok: {ns}");
        }
        Cmd::Parse { query } => {
            let q = parse(&query).map_err(|errs| parse_err(&errs))?;
            println!("{}", q);
        }
        Cmd::Explain {
            query,
            verbose,
            raw,
        } => {
            let q = parse(&query).map_err(|errs| parse_err(&errs))?;
            let want_verbose = verbose || q.explain_verbose;
            let want_raw = raw || q.explain_raw;
            let tree = match (want_raw, want_verbose) {
                (true, true) => {
                    let catalog = StatsCatalog::empty();
                    explain_query_raw_verbose(&q, &catalog).map_err(|e| anyhow::anyhow!("{}", e))?
                }
                (true, false) => explain_query_raw(&q).map_err(|e| anyhow::anyhow!("{}", e))?,
                (false, true) => {
                    let catalog = StatsCatalog::empty();
                    explain_query_verbose(&q, &catalog).map_err(|e| anyhow::anyhow!("{}", e))?
                }
                (false, false) => explain_query(&q).map_err(|e| anyhow::anyhow!("{}", e))?,
            };
            print!("{}", tree);
        }
        Cmd::Run {
            store,
            namespace,
            query,
        } => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(run_query(store.as_deref(), &namespace, &query))?;
        }
        Cmd::LoadVault {
            store,
            namespace,
            label,
            edge_type,
            prune,
            placeholders,
            embed,
            watch,
            path,
        } => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(load_vault_cmd(
                store.as_deref(),
                &namespace,
                &label,
                &edge_type,
                prune,
                placeholders,
                embed,
                watch,
                &path,
            ))?;
        }
        Cmd::Backup {
            from,
            to,
            version,
            force,
            verify,
        } => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(copy_namespace_cmd(
                "backed up",
                &from,
                &to,
                version,
                force,
                verify,
            ))?;
        }
        Cmd::Restore {
            from,
            to,
            version,
            force,
            verify,
        } => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            rt.block_on(copy_namespace_cmd(
                "restored", &from, &to, version, force, verify,
            ))?;
        }
    }
    Ok(())
}

/// Shared driver for `backup` and `restore`: both copy a consistent namespace
/// snapshot from `from` to `to`. `verb` only changes the success line.
async fn copy_namespace_cmd(
    verb: &str,
    from: &str,
    to: &str,
    version: Option<u64>,
    force: bool,
    verify: bool,
) -> anyhow::Result<()> {
    let (src_store, src_paths) = parse_uri(from)?;
    let (dst_store, dst_paths) = parse_uri(to)?;
    let report = copy_namespace_snapshot(
        src_store, src_paths, dst_store, dst_paths, version, force, verify,
    )
    .await?;
    println!(
        "{verb} {from} (source version {}) -> {to}",
        report.source_version
    );
    println!(
        "  {} objects, {} bytes",
        report.objects_copied, report.bytes_copied
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn load_vault_cmd(
    store_uri: Option<&str>,
    namespace: &str,
    label: &str,
    edge_type: &str,
    prune: bool,
    placeholders: bool,
    embed: bool,
    watch: bool,
    path: &str,
) -> anyhow::Result<()> {
    let (store, paths): (Arc<dyn ObjectStore>, NamespacePaths) = match store_uri {
        Some(uri) => parse_uri(uri).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => {
            let ns = NamespaceId::new(namespace)?;
            (
                Arc::new(InMemory::new()),
                NamespacePaths::new("tenants", ns),
            )
        }
    };

    let mut writer = WriterSession::open(store, paths).await?;
    let opts = LoadOptions {
        label: label.to_string(),
        edge_type: edge_type.to_string(),
        // A watch mirrors the vault on every sync, so prune is implied.
        prune: prune || watch,
        placeholders,
        // `--embed` picks the embedder from the environment: a remote provider
        // when NAMIDB_EMBED_* is set (with --features remote-embedder), else the
        // local HashingEmbedder.
        embedder: embed.then(embedder_from_env),
        ..Default::default()
    };

    if watch {
        if store_uri.is_none() {
            eprintln!("(in-memory namespace; a watch is only useful with --store <uri> to persist the graph)");
        }
        return watch_vault_cmd(std::path::Path::new(path), &mut writer, &opts).await;
    }

    let outcome = load_vault(std::path::Path::new(path), &mut writer, &opts).await?;
    // Flush the tail the loader leaves pending so the graph is durable.
    writer.commit_batch().await?;

    println!("{}", "─".repeat(48));
    println!("notes loaded    : {}", outcome.notes_loaded);
    println!("links resolved  : {}", outcome.links_resolved);
    println!("links dangling  : {}", outcome.links_dangling);
    println!("embeds resolved : {}", outcome.embeds_resolved);
    println!("embeds dangling : {}", outcome.embeds_dangling);
    println!("name collisions : {}", outcome.name_collisions);
    if outcome.aliases_registered > 0 {
        println!("aliases         : {}", outcome.aliases_registered);
    }
    println!("tags loaded     : {}", outcome.tags_loaded);
    println!("tag links       : {}", outcome.tag_links);
    if outcome.subtag_edges > 0 {
        println!("subtag edges    : {}", outcome.subtag_edges);
    }
    if placeholders {
        println!("placeholders    : {}", outcome.placeholders_created);
    }
    if prune {
        println!("notes pruned    : {}", outcome.notes_pruned);
        println!("links pruned    : {}", outcome.links_pruned);
        println!("embeds pruned   : {}", outcome.embeds_pruned);
        println!("tags pruned     : {}", outcome.tags_pruned);
        println!("tag links pruned: {}", outcome.tag_links_pruned);
        println!("subtag pruned   : {}", outcome.subtag_edges_pruned);
    }
    println!("{}", "─".repeat(48));
    if store_uri.is_none() {
        println!("(in-memory namespace; pass --store <uri> to persist the graph)");
    }
    Ok(())
}

/// Do an initial mirrored sync, then watch `dir` and re-sync on every debounced
/// change until Ctrl-C, so the graph stays a live index of the vault.
async fn watch_vault_cmd(
    dir: &std::path::Path,
    writer: &mut WriterSession,
    opts: &LoadOptions,
) -> anyhow::Result<()> {
    use notify::{RecursiveMode, Watcher};
    use notify_debouncer_full::new_debouncer;
    use std::time::Duration;

    // Initial sync: over an empty namespace every note classifies as added, so
    // this behaves like a full load; over a populated store it reconciles
    // whatever already exists, including offline edits made while not watching.
    let out = sync_vault(dir, writer, opts).await?;
    writer.commit_batch().await?;
    eprintln!(
        "synced {}: +{} ~{} -{} ={} (links {}, tags {})",
        dir.display(),
        out.notes_added,
        out.notes_modified,
        out.notes_deleted,
        out.notes_unchanged,
        out.load.links_resolved,
        out.load.tags_loaded,
    );

    // The debouncer runs the OS watcher on its own thread and coalesces a burst
    // of edits (editors write-then-rename, multi-file paste) into one batch.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut debouncer = new_debouncer(Duration::from_millis(400), None, move |res| {
        let _ = tx.send(res);
    })
    .map_err(|e| anyhow::anyhow!("watcher: {e}"))?;
    debouncer
        .watcher()
        .watch(dir, RecursiveMode::Recursive)
        .map_err(|e| anyhow::anyhow!("watch {}: {e}", dir.display()))?;

    eprintln!("watching {} for changes (Ctrl-C to stop)", dir.display());
    loop {
        tokio::select! {
            event = rx.recv() => match event {
                // The batch is only a trigger: sync re-walks and re-hashes the
                // vault, so event paths are never trusted for correctness.
                Some(Ok(_batch)) => {
                    let out = sync_vault(dir, writer, opts).await?;
                    writer.commit_batch().await?;
                    if out.notes_added + out.notes_modified + out.notes_deleted > 0 {
                        eprintln!(
                            "sync: +{} ~{} -{} ={}",
                            out.notes_added,
                            out.notes_modified,
                            out.notes_deleted,
                            out.notes_unchanged,
                        );
                    }
                }
                Some(Err(errs)) => eprintln!("watch error: {errs:?}"),
                None => break,
            },
            _ = tokio::signal::ctrl_c() => {
                eprintln!("stopping watch");
                break;
            }
        }
    }
    Ok(())
}

async fn run_query(store_uri: Option<&str>, namespace: &str, query: &str) -> anyhow::Result<()> {
    // Multi-statement scripts (first field report, item 42): split on `;`
    // outside strings/backticks/comments and run sequentially against ONE
    // session, stopping at the first error. This is what makes a pasted
    // schema-bootstrap script (constraints + indexes + seed writes) work.
    let statements = split_statements(query);
    if statements.is_empty() {
        anyhow::bail!("no statements in input");
    }

    let (store, paths): (Arc<dyn ObjectStore>, NamespacePaths) = match store_uri {
        Some(uri) => parse_uri(uri).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => {
            let ns = NamespaceId::new(namespace)?;
            let paths = NamespacePaths::new("tenants", ns);
            (Arc::new(InMemory::new()), paths)
        }
    };
    let mut writer = WriterSession::open(store, paths).await?;

    let total = statements.len();
    for (index, statement) in statements.iter().enumerate() {
        if total > 1 {
            println!("== statement {}/{total}", index + 1);
        }
        run_statement(&mut writer, statement)
            .await
            .map_err(|e| anyhow::anyhow!("statement {}/{total}: {e}", index + 1))?;
    }
    Ok(())
}

/// One statement against the shared session. Schema DDL is intercepted the
/// same way the server does it — `CREATE CONSTRAINT` / `CREATE INDEX`
/// execute out-of-band through the writer's schema commit, never through
/// the planner (which has no DDL operators).
async fn run_statement(writer: &mut WriterSession, statement: &str) -> anyhow::Result<()> {
    let q = parse(statement).map_err(|errs| parse_err(&errs))?;

    if let Some(c) = q.as_create_constraint() {
        let properties: Vec<String> = c.properties.iter().map(|p| p.name.clone()).collect();
        let version = writer
            .create_unique_constraint_named(
                c.name.as_ref().map(|n| n.name.as_str()),
                &c.label.name,
                &properties,
                c.if_not_exists,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("constraint applied (manifest v{version})");
        return Ok(());
    }
    if let Some(ix) = q.as_create_index() {
        let properties: Vec<String> = ix.properties.iter().map(|p| p.name.clone()).collect();
        let name = ix.name.as_ref().map(|n| n.name.as_str());
        let version = if properties.len() > 1 {
            writer
                .create_composite_index_named(name, &ix.label.name, &properties, ix.if_not_exists)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            writer
                .create_property_index_named(name, &ix.label.name, &properties[0], ix.if_not_exists)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        println!("index applied (manifest v{version})");
        return Ok(());
    }

    // The catalog is rebuilt per statement: an earlier DDL or write in the
    // same script changes what the optimizer should know.
    let catalog = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    let plan = build_plan(&q, &catalog).map_err(|e| anyhow::anyhow!("{}", e))?;

    // `EXPLAIN [RAW] [VERBOSE] <query>`: render the plan (real catalog +
    // `# route:` footer, unlike the offline `explain` subcommand's empty
    // catalog) instead of executing — an `EXPLAIN CREATE ...` must never
    // write.
    if q.explain {
        let snap = writer.snapshot();
        for line in namidb_query::explain_plan_lines(&q, &plan, &snap, &catalog) {
            println!("{line}");
        }
        return Ok(());
    }

    if plan.contains_write() {
        let outcome = execute_write(&plan, writer, &Params::new())
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        print_write_outcome(&outcome);
    } else {
        let snap = writer.snapshot();
        let rows = execute(&plan, &snap, &Params::new())
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        print_rows(&rows);
    }
    Ok(())
}

/// Split a `;`-separated Cypher script into statements, honouring single-
/// and double-quoted strings (with backslash escapes), backtick-quoted
/// identifiers, and `//` line / `/* */` block comments (the lexer strips
/// comments itself, so they pass through unsplit). Empty fragments (stray
/// or trailing `;`) are dropped.
fn split_statements(input: &str) -> Vec<String> {
    #[derive(PartialEq)]
    enum Mode {
        Plain,
        Single,
        Double,
        Backtick,
        LineComment,
        BlockComment,
    }
    let mut statements = Vec::new();
    let mut current = String::new();
    let mut mode = Mode::Plain;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match mode {
            Mode::Plain => match c {
                ';' => {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        statements.push(trimmed.to_string());
                    }
                    current.clear();
                    continue;
                }
                '\'' => mode = Mode::Single,
                '"' => mode = Mode::Double,
                '`' => mode = Mode::Backtick,
                '/' if chars.peek() == Some(&'/') => mode = Mode::LineComment,
                '/' if chars.peek() == Some(&'*') => mode = Mode::BlockComment,
                _ => {}
            },
            Mode::Single => match c {
                '\\' => {
                    current.push(c);
                    if let Some(escaped) = chars.next() {
                        current.push(escaped);
                    }
                    continue;
                }
                '\'' => mode = Mode::Plain,
                _ => {}
            },
            Mode::Double => match c {
                '\\' => {
                    current.push(c);
                    if let Some(escaped) = chars.next() {
                        current.push(escaped);
                    }
                    continue;
                }
                '"' => mode = Mode::Plain,
                _ => {}
            },
            Mode::Backtick => {
                if c == '`' {
                    mode = Mode::Plain;
                }
            }
            Mode::LineComment => {
                if c == '\n' {
                    mode = Mode::Plain;
                }
            }
            Mode::BlockComment => {
                if c == '*' && chars.peek() == Some(&'/') {
                    current.push(c);
                    current.push(chars.next().expect("peeked"));
                    mode = Mode::Plain;
                    continue;
                }
            }
        }
        current.push(c);
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        statements.push(trimmed.to_string());
    }
    statements
}

fn parse_err(errs: &[namidb_query::ParseError]) -> anyhow::Error {
    let first = &errs[0];
    anyhow::anyhow!("{:?}: {} at {}", first.code, first.message, first.span)
}

fn print_write_outcome(outcome: &WriteOutcome) {
    println!("{}", "─".repeat(48));
    println!("nodes created : {}", outcome.nodes_created);
    println!("edges created : {}", outcome.edges_created);
    println!("nodes deleted : {}", outcome.nodes_deleted);
    println!("edges deleted : {}", outcome.edges_deleted);
    println!("properties set : {}", outcome.properties_set);
    println!("returned rows : {}", outcome.rows.len());
    println!("{}", "─".repeat(48));
    print_rows(&outcome.rows);
}

fn print_rows(rows: &[namidb_query::Row]) {
    if rows.is_empty() {
        println!("(no rows)");
        return;
    }
    let columns: Vec<&String> = rows[0].bindings.keys().collect();
    println!(
        "{}",
        columns
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!(
        "{}",
        "-".repeat(columns.iter().map(|c| c.len() + 3).sum::<usize>().max(8))
    );
    for row in rows {
        let cells: Vec<String> = columns
            .iter()
            .map(|c| {
                row.bindings
                    .get(c.as_str())
                    .map(format_runtime)
                    .unwrap_or_else(|| "null".to_string())
            })
            .collect();
        println!("{}", cells.join(" | "));
    }
}

fn format_runtime(v: &RuntimeValue) -> String {
    match v {
        RuntimeValue::Null => "null".to_string(),
        RuntimeValue::Bool(b) => b.to_string(),
        RuntimeValue::Integer(n) => n.to_string(),
        RuntimeValue::Float(f) => f.to_string(),
        RuntimeValue::String(s) => format!("\"{}\"", s),
        RuntimeValue::List(items) => {
            let inner: Vec<String> = items.iter().map(format_runtime).collect();
            format!("[{}]", inner.join(", "))
        }
        RuntimeValue::Map(m) => {
            let inner: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{}: {}", k, format_runtime(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        RuntimeValue::Node(n) => {
            let props: Vec<String> = n
                .properties
                .iter()
                .map(|(k, v)| format!("{}: {}", k, format_runtime(v)))
                .collect();
            let labels: String = n.labels.iter().map(|l| format!(":{}", l)).collect();
            format!(
                "({}{} {{{}}})",
                &n.id.to_string()[..8],
                labels,
                props.join(", ")
            )
        }
        RuntimeValue::Rel(r) => format!("[:{}]", r.edge_type),
        RuntimeValue::Path(items) => {
            let inner: Vec<String> = items.iter().map(format_runtime).collect();
            format!("PATH[{}]", inner.join(" → "))
        }
        RuntimeValue::Date(d) => format!("date({})", d),
        RuntimeValue::DateTime(d) => format!("datetime({})", d),
        RuntimeValue::Bytes(b) => format!("bytes({} bytes)", b.len()),
        RuntimeValue::Vector(v) => format!("vec[{}]", v.len()),
        RuntimeValue::Vector8 { codes, .. } => format!("vec8[{}]", codes.len()),
    }
}

// Keep this used to silence unused warning in the binary if the
// closure-style logging dispatch ever changes.
#[allow(dead_code)]
fn _suppress_core_value_unused(_v: CoreValue) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_honours_strings_backticks_and_comments() {
        let script = "CREATE (:P {name: 'a;b'}); // trailing; comment\n\
                      MATCH (`weird;name`) RETURN 1; /* block; \n comment */ RETURN \"x;y\";;";
        let statements = split_statements(script);
        assert_eq!(statements.len(), 3, "{statements:?}");
        assert!(statements[0].contains("'a;b'"));
        assert!(statements[1].contains("`weird;name`"));
        assert!(statements[2].contains("\"x;y\""));
        // Escaped quote inside a string hides the terminator.
        let statements = split_statements(r"RETURN 'don\'t; split'; RETURN 2");
        assert_eq!(statements.len(), 2, "{statements:?}");
        assert!(statements[0].contains(r"don\'t; split"));
        // Single statement without any semicolon.
        assert_eq!(split_statements("RETURN 1").len(), 1);
        assert!(split_statements(" ;; ; ").is_empty());
    }

    #[tokio::test]
    async fn run_query_executes_a_multi_statement_schema_script() {
        // DDL + writes + a read in one script against one session; the
        // MERGE proves the constraint from statement 1 is live for
        // statement 3 (stop-on-first-error would surface any failure).
        run_query(
            None,
            "cli-script",
            "CREATE CONSTRAINT person_email IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE; \
             CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name); \
             CREATE INDEX person_pair IF NOT EXISTS FOR (p:Person) ON (p.city, p.age); \
             CREATE (:Person {email: 'a@x', name: 'Ada'}); \
             MERGE (p:Person {email: 'a@x'}) RETURN p.name AS name",
        )
        .await
        .expect("schema script must run end to end");

        // Stop-on-first-error: the second statement is garbage, the call fails.
        let err = run_query(None, "cli-script-err", "RETURN 1; THIS IS NOT CYPHER")
            .await
            .expect_err("bad second statement must fail the script");
        assert!(err.to_string().contains("statement 2/2"), "{err}");
    }

    /// `EXPLAIN` in a `run` script renders instead of executing: before
    /// this interception the CLI silently EXECUTED the query, so an
    /// `EXPLAIN CREATE ...` wrote a row (the same trap the server fixed
    /// on its own surface).
    #[tokio::test]
    async fn explain_in_run_renders_and_never_executes() {
        let ns = NamespaceId::new("cli-explain").unwrap();
        let paths = NamespacePaths::new("tenants", ns);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let mut writer = WriterSession::open(store, paths).await.unwrap();

        run_statement(&mut writer, "CREATE INDEX pair FOR (x:X) ON (x.a, x.b)")
            .await
            .unwrap();
        run_statement(&mut writer, "EXPLAIN CREATE (:X {a: 1, b: 2})")
            .await
            .unwrap();
        let snap = writer.snapshot();
        assert!(
            snap.scan_label("X").await.unwrap().is_empty(),
            "EXPLAIN of a write must create nothing"
        );

        // A covered composite conjunct EXPLAINs to the tuple operator
        // (the shared renderer also appends its `# route:` note, asserted
        // in the server's explain_surface test).
        run_statement(
            &mut writer,
            "EXPLAIN MATCH (x:X) WHERE x.a = 1 AND x.b = 2 RETURN x",
        )
        .await
        .unwrap();
    }
}
