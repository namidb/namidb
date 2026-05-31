//! `namidb-mcp` binary: a local MCP server over a NamiDB graph namespace.
//!
//! Reads JSON-RPC from stdin and writes responses to stdout. Logs go to
//! stderr so they never corrupt the protocol channel.

use clap::Parser;

/// Local MCP server (stdio) exposing a NamiDB graph namespace to agents.
#[derive(Parser, Debug)]
#[command(name = "namidb-mcp", version, about = "NamiDB MCP server over stdio")]
struct Args {
    /// Storage URI of the namespace to serve (see the storage crate for the
    /// scheme reference). Defaults to an ephemeral in-memory namespace, which
    /// is only useful together with `--vault`.
    #[arg(long, default_value = "memory://mcp")]
    store: String,
    /// Optional markdown vault to load into the namespace before serving.
    #[arg(long)]
    vault: Option<String>,
    /// Create stub `:Note` nodes for links/embeds whose target does not exist,
    /// so unresolved references show up in the graph. Matches `load-vault
    /// --placeholders`. Only meaningful with `--vault`.
    #[arg(long, default_value_t = false)]
    placeholders: bool,
    /// Keep the graph live: after the initial load, watch the vault and
    /// re-sync incrementally on every change while serving. Requires `--vault`.
    #[arg(long, default_value_t = false, requires = "vault")]
    watch: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // stdout is the JSON-RPC channel; route logs to stderr to keep it clean.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse();
    let server = namidb_mcp::Server::open(&args.store).await?;
    if let Some(vault) = args.vault.as_deref() {
        let outcome = server
            .load_vault(std::path::Path::new(vault), args.placeholders)
            .await?;
        eprintln!(
            "loaded vault: {} notes, {} links ({} dangling), {} placeholders",
            outcome.notes_loaded,
            outcome.links_resolved,
            outcome.links_dangling,
            outcome.placeholders_created
        );
        if args.watch {
            server.watch_vault(std::path::Path::new(vault), args.placeholders)?;
            eprintln!("watching {vault} for changes (graph stays live while serving)");
        }
    }
    namidb_mcp::serve_stdio(server).await
}
