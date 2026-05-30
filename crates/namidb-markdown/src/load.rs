//! Ingest a parsed vault into a [`WriterSession`] as a graph.
//!
//! Each note becomes a node (default label `Note`); each resolved wikilink
//! becomes an edge (default type `LINKS_TO`). Dangling links (targets with no
//! matching note) are counted but produce no edge, so every edge endpoint is a
//! real note and queries like "orphan notes" or "backlinks" stay clean.
//!
//! Follows the same cadence contract as the parquet loader: the loader leaves
//! the final batch pending and the caller decides when to `commit_batch`.

use std::collections::HashSet;
use std::path::Path;

use namidb_storage::{EdgeWriteRecord, NodeWriteRecord, WriterSession};

use crate::id::stable_node_id;
use crate::parse::{parse_vault, VaultGraph};

const DEFAULT_LABEL: &str = "Note";
const DEFAULT_EDGE_TYPE: &str = "LINKS_TO";
/// Rows (nodes + edges) between in-loop commits. Picked to match the parquet
/// loader's order of magnitude; vaults are small so most loads commit once.
const DEFAULT_COMMIT_EVERY: usize = 1000;

/// How to map a vault onto the graph.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Node label for notes.
    pub label: String,
    /// Edge type for wikilinks.
    pub edge_type: String,
    /// Rows allowed to accumulate between `commit_batch` calls. `0` leaves
    /// everything pending for the caller's final flush.
    pub commit_every: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            label: DEFAULT_LABEL.to_string(),
            edge_type: DEFAULT_EDGE_TYPE.to_string(),
            commit_every: DEFAULT_COMMIT_EVERY,
        }
    }
}

/// Outcome of a vault load.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VaultLoadOutcome {
    /// Notes upserted as nodes.
    pub notes_loaded: usize,
    /// Wikilinks that resolved to a known note and produced an edge.
    pub links_resolved: usize,
    /// Wikilinks whose target was not a known note (no edge written).
    pub links_dangling: usize,
    /// Notes whose normalized key collided with an earlier note (last write
    /// wins; surfaced so silent overwrites are visible).
    pub name_collisions: usize,
    /// `commit_batch` calls fired during the load (excludes the caller's
    /// final flush).
    pub commit_batches: usize,
}

/// Parse the vault at `dir` and ingest it through `writer`.
pub async fn load_vault(
    dir: &Path,
    writer: &mut WriterSession,
    opts: &LoadOptions,
) -> anyhow::Result<VaultLoadOutcome> {
    let graph = parse_vault(dir)?;
    load_graph(&graph, writer, opts).await
}

/// Ingest an already-parsed [`VaultGraph`]. Split out so the resolution + write
/// path is testable from an in-memory graph without filesystem I/O.
pub async fn load_graph(
    graph: &VaultGraph,
    writer: &mut WriterSession,
    opts: &LoadOptions,
) -> anyhow::Result<VaultLoadOutcome> {
    // The set of normalized keys that exist as real notes, used to tell a
    // resolved link from a dangling one.
    let mut known: HashSet<&str> = HashSet::with_capacity(graph.notes.len());
    let mut collisions = 0usize;
    for note in &graph.notes {
        if !known.insert(note.key.as_str()) {
            collisions += 1;
        }
    }

    let mut outcome = VaultLoadOutcome {
        name_collisions: collisions,
        ..Default::default()
    };
    let mut rows_since_commit = 0usize;
    let edge_record = EdgeWriteRecord::default();

    for note in &graph.notes {
        let record = NodeWriteRecord {
            properties: note.properties.clone(),
            schema_version: 1,
        };
        writer.upsert_node(opts.label.clone(), note.id, &record)?;
        outcome.notes_loaded += 1;
        rows_since_commit += 1;

        for target in &note.links {
            if known.contains(target.as_str()) {
                let dst = stable_node_id(target);
                writer.upsert_edge(opts.edge_type.clone(), note.id, dst, &edge_record)?;
                outcome.links_resolved += 1;
                rows_since_commit += 1;
            } else {
                outcome.links_dangling += 1;
            }
        }

        if opts.commit_every > 0 && rows_since_commit >= opts.commit_every {
            writer.commit_batch().await?;
            outcome.commit_batches += 1;
            rows_since_commit = 0;
        }
    }

    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use namidb_core::NamespaceId;
    use namidb_storage::{NamespacePaths, WriterSession};
    use object_store::memory::InMemory;
    use tempfile::TempDir;

    use super::*;

    fn write(dir: &Path, rel: &str, content: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[tokio::test]
    async fn loads_small_vault_through_writer_session() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(
            dir,
            "Alpha.md",
            "---\ntag: x\n---\nlinks to [[Beta]] and [[Gamma]]\n",
        );
        write(dir, "Beta.md", "back to [[Alpha]]\n");
        write(dir, "Gamma.md", "to a [[Missing]] note\n");
        // Both of these must be skipped by the walker.
        write(dir, ".obsidian/workspace.md", "[[Alpha]]\n");
        write(dir, "_templates/Tmpl.md", "[[Alpha]]\n");

        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let paths = NamespacePaths::new("test", NamespaceId::new("vault-load").unwrap());
        let mut writer = WriterSession::open(store, paths).await.unwrap();

        let outcome = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        // Flush the tail the loader left pending.
        writer.commit_batch().await.unwrap();

        assert_eq!(
            outcome.notes_loaded, 3,
            "3 real notes; .obsidian and _templates excluded"
        );
        // Alpha->Beta, Alpha->Gamma, Beta->Alpha resolve; Gamma->Missing dangles.
        assert_eq!(outcome.links_resolved, 3);
        assert_eq!(outcome.links_dangling, 1);
        assert_eq!(outcome.name_collisions, 0);
    }
}
