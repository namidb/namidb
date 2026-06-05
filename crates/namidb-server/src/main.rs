//! Entry point for the `namidb-server` binary.
//!
//! Parses CLI flags + env vars, calls [`namidb_server::run`].

use std::time::Duration;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "namidb-server",
    version,
    about = "HTTP server exposing a NamiDB namespace over REST"
)]
struct Cli {
    /// Storage URI. Examples:
    ///   file:///var/lib/namidb?ns=prod
    ///   s3://my-bucket/data?ns=prod&region=us-east-1
    ///   gs://my-bucket?ns=prod
    ///   az://acct/container?ns=prod
    ///   memory://demo
    #[arg(long, env = "NAMIDB_STORE")]
    store: String,

    /// Address to bind. Defaults to `0.0.0.0:8080`.
    #[arg(long, env = "NAMIDB_LISTEN", default_value = "0.0.0.0:8080")]
    listen: std::net::SocketAddr,

    /// Bearer token required for `/v0/cypher` and `/v0/admin/*`.
    /// When unset, the server starts in unauthenticated mode and
    /// logs a loud warning.
    #[arg(long, env = "NAMIDB_AUTH_TOKEN")]
    auth_token: Option<String>,

    /// Interval at which the memtable is flushed to L0 SSTs in the
    /// background. Set to `0s` to disable periodic flush (callers
    /// must POST /v0/admin/flush manually).
    #[arg(
        long,
        env = "NAMIDB_FLUSH_INTERVAL",
        default_value = "30s",
        value_parser = humantime::parse_duration,
    )]
    flush_interval: Duration,

    /// Interval at which the background maintenance task compacts L0 SSTs
    /// (collapsing each bucket to a single L1 SST to keep read
    /// amplification bounded) and then sweeps orphaned SST bodies. Set to
    /// `0s` to disable maintenance entirely.
    #[arg(
        long,
        env = "NAMIDB_COMPACTION_INTERVAL",
        default_value = "300s",
        value_parser = humantime::parse_duration,
    )]
    compaction_interval: Duration,

    /// Minimum age an orphaned SST body must reach before the sweep may
    /// delete it. This is the only guard against removing a file a slow
    /// reader's pinned snapshot still references, so keep it comfortably
    /// above the longest expected query/snapshot lifetime.
    #[arg(
        long,
        env = "NAMIDB_SWEEP_MIN_AGE",
        default_value = "24h",
        value_parser = humantime::parse_duration,
    )]
    sweep_min_age: Duration,

    /// Actually delete orphaned SST bodies during the sweep. Off by
    /// default: the sweep runs as a dry-run and only logs what it would
    /// free, so an operator can review the volume before opting in.
    #[arg(long, env = "NAMIDB_SWEEP_DELETE", default_value_t = false)]
    sweep_delete: bool,

    /// Address for the Bolt protocol listener (Neo4j driver
    /// compatibility). When omitted the protocol is off and the
    /// server is HTTP-only. The canonical Bolt port is 7687.
    #[arg(long, env = "NAMIDB_BOLT_LISTEN")]
    bolt_listen: Option<std::net::SocketAddr>,

    /// Idle timeout for an open Bolt explicit transaction. The writer lock
    /// is held for the life of a transaction, so an idle client would pin
    /// it; after this long without a message the transaction is rolled back
    /// and failed. Set to `0s` to allow transactions to stay open forever.
    #[arg(
        long,
        env = "NAMIDB_BOLT_TX_TIMEOUT",
        default_value = "30s",
        value_parser = humantime::parse_duration,
    )]
    bolt_tx_timeout: Duration,

    /// Wall-clock budget for a single read query (HTTP and Bolt, including
    /// in-transaction reads). A runaway scan or expansion is aborted with a
    /// timeout error instead of pinning a worker. Set to `0s` to allow read
    /// queries to run unbounded.
    #[arg(
        long,
        env = "NAMIDB_QUERY_TIMEOUT",
        default_value = "30s",
        value_parser = humantime::parse_duration,
    )]
    query_timeout: Duration,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let config = namidb_server::Config {
        store_uri: cli.store,
        listen: cli.listen,
        auth_token: cli.auth_token,
        flush_interval: cli.flush_interval,
        compaction_interval: cli.compaction_interval,
        sweep_min_age: cli.sweep_min_age,
        sweep_delete: cli.sweep_delete,
        bolt_listen: cli.bolt_listen,
        bolt_tx_timeout: cli.bolt_tx_timeout,
        query_timeout: cli.query_timeout,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(namidb_server::run(config))
}
