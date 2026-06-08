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

    /// Bearer token required for `/v0/cypher` and `/v0/admin/*`. Grants
    /// read-write access. When unset (and no `--auth-tokens-file`), the server
    /// starts in unauthenticated mode and logs a loud warning.
    #[arg(long, env = "NAMIDB_AUTH_TOKEN")]
    auth_token: Option<String>,

    /// Path to a JSON file of tokens, each with a `read-only` or `read-write`
    /// role, e.g. `{ "tokens": [{ "name": "ci", "token": "…", "role":
    /// "read-write" }, { "token": "…", "role": "read-only" }] }`. Takes
    /// precedence over `--auth-token`; lets you hand out read-only tokens and
    /// keep secrets out of the process arguments.
    #[arg(long, env = "NAMIDB_AUTH_TOKENS_FILE")]
    auth_tokens_file: Option<std::path::PathBuf>,

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

    /// Delete orphaned SST bodies during the sweep. On by default: the
    /// retention horizon (RFC-027) makes deletion safe by construction (an
    /// object referenced by no manifest version from the horizon to current
    /// is unreachable by any reader). Set to `false` for a dry-run that only
    /// logs what it would free.
    #[arg(
        long,
        env = "NAMIDB_SWEEP_DELETE",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
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

    /// Maximum rows a single read-query operator may materialise. A query
    /// whose operator output would exceed this aborts with a row-cap error
    /// instead of risking an out-of-memory blow-up. Set to `0` to allow read
    /// queries to materialise without limit.
    #[arg(long, env = "NAMIDB_QUERY_ROW_CAP", default_value_t = 0)]
    query_row_cap: usize,

    /// L0-count high-water mark per bucket that triggers a compaction as
    /// soon as a flush crosses it, instead of waiting for the periodic
    /// compaction tick. Bounds read amplification under sustained writes.
    /// Set to `0` to disable the reactive trigger.
    #[arg(long, env = "NAMIDB_COMPACTION_L0_TRIGGER", default_value_t = 8)]
    compaction_l0_trigger: usize,

    /// L0-count per bucket above which a committed write is softly stalled
    /// by `--write-stall-delay`, so the writer cannot outrun compaction
    /// without bound. Set to `0` (the default) to disable the stall.
    #[arg(long, env = "NAMIDB_WRITE_STALL_L0", default_value_t = 0)]
    write_stall_l0: usize,

    /// Delay applied to a committed write while L0 is above
    /// `--write-stall-l0`. Ignored when the stall is disabled.
    #[arg(
        long,
        env = "NAMIDB_WRITE_STALL_DELAY",
        default_value = "50ms",
        value_parser = humantime::parse_duration,
    )]
    write_stall_delay: Duration,

    /// PEM certificate-chain file. Set together with `--tls-key` to serve the
    /// HTTP and Bolt listeners over TLS; omit both to serve plaintext.
    #[arg(long, env = "NAMIDB_TLS_CERT")]
    tls_cert: Option<std::path::PathBuf>,

    /// PEM private-key file paired with `--tls-cert`.
    #[arg(long, env = "NAMIDB_TLS_KEY")]
    tls_key: Option<std::path::PathBuf>,

    /// Wall-clock at or above which a query is logged at WARN as a slow query
    /// (the statement text, never its parameters). The Prometheus counters and
    /// latency histograms at `/v0/metrics` are always on regardless of this.
    /// Set to `0s` to turn the slow-query log off.
    #[arg(
        long,
        env = "NAMIDB_SLOW_QUERY_THRESHOLD",
        default_value = "1s",
        value_parser = humantime::parse_duration,
    )]
    slow_query_threshold: Duration,
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
        auth_tokens_file: cli.auth_tokens_file,
        flush_interval: cli.flush_interval,
        compaction_interval: cli.compaction_interval,
        sweep_min_age: cli.sweep_min_age,
        sweep_delete: cli.sweep_delete,
        bolt_listen: cli.bolt_listen,
        bolt_tx_timeout: cli.bolt_tx_timeout,
        query_timeout: cli.query_timeout,
        query_row_cap: cli.query_row_cap,
        compaction_l0_trigger: cli.compaction_l0_trigger,
        write_stall_l0: cli.write_stall_l0,
        write_stall_delay: cli.write_stall_delay,
        tls_cert: cli.tls_cert,
        tls_key: cli.tls_key,
        slow_query_threshold: cli.slow_query_threshold,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(namidb_server::run(config))
}
