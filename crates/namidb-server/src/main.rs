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
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(namidb_server::run(config))
}
