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
 format!(
 "({}:{} {{{}}})",
 &n.id.to_string()[..8],
 n.label,
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
