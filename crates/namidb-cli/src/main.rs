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

use std::sync::Arc;

use clap::{Parser, Subcommand};
use namidb_core::{id::NamespaceId, value::Value as CoreValue};
use namidb_markdown::{embedder_from_env, load_vault, sync_vault, LoadOptions};
use namidb_query::{
    execute, execute_write, explain_query, explain_query_raw, explain_query_raw_verbose,
    explain_query_verbose, parse, plan as build_plan, Params, RuntimeValue, StatsCatalog,
    WriteOutcome,
};
use namidb_storage::{parse_uri, NamespacePaths, WriterSession};
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
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
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
    let q = parse(query).map_err(|errs| parse_err(&errs))?;

    let (store, paths): (Arc<dyn ObjectStore>, NamespacePaths) = match store_uri {
        Some(uri) => parse_uri(uri).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => {
            let ns = NamespaceId::new(namespace)?;
            let paths = NamespacePaths::new("tenants", ns);
            (Arc::new(InMemory::new()), paths)
        }
    };

    let mut writer = WriterSession::open(store, paths).await?;
    let catalog = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    let plan = build_plan(&q, &catalog).map_err(|e| anyhow::anyhow!("{}", e))?;

    if plan.contains_write() {
        let outcome = execute_write(&plan, &mut writer, &Params::new())
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
    }
}

// Keep this used to silence unused warning in the binary if the
// closure-style logging dispatch ever changes.
#[allow(dead_code)]
fn _suppress_core_value_unused(_v: CoreValue) {}
