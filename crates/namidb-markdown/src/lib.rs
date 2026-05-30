//! # namidb-markdown
//!
//! Ingest an Obsidian-style markdown vault into NamiDB as a graph: every `.md`
//! note becomes a node, every `[[wikilink]]` becomes an edge, and frontmatter
//! becomes node properties. The note body is kept verbatim as a `body`
//! property so the original markdown stays the source of truth and the graph
//! is a derived, rebuildable index over it.
//!
//! The markdown and YAML parsers live here, not in the core storage/query
//! crates, so the engine stays free of these dependencies and this
//! product-shaped (and churn-prone) mapping is contained to one crate.
//!
//! ```no_run
//! # async fn run(writer: &mut namidb_storage::WriterSession) -> anyhow::Result<()> {
//! use std::path::Path;
//! use namidb_markdown::{load_vault, LoadOptions};
//!
//! let outcome = load_vault(Path::new("./my-vault"), writer, &LoadOptions::default()).await?;
//! writer.commit_batch().await?; // flush the tail the loader left pending
//! println!("{} notes, {} links", outcome.notes_loaded, outcome.links_resolved);
//! # Ok(())
//! # }
//! ```
//!
//! See [`parse`] for the deliberate v1 subset of Obsidian behaviour this
//! covers (and what it does not).

#![warn(rust_2018_idioms)]

pub mod id;
pub mod load;
pub mod parse;

pub use id::{normalize_key, stable_node_id};
pub use load::{load_graph, load_vault, LoadOptions, VaultLoadOutcome};
pub use parse::{parse_note, parse_vault, ParsedNote, VaultGraph};
