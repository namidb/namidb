//! Ingest a parsed vault into a [`WriterSession`] as a graph.
//!
//! Each note becomes a node (default label `Note`); each resolved wikilink
//! becomes an edge (default type `LINKS_TO`). Dangling links (targets with no
//! matching note) are counted but produce no edge, so every edge endpoint is a
//! real note and queries like "orphan notes" or "backlinks" stay clean.
//!
//! By default a load is additive (upsert-only): nodes overwrite in place
//! thanks to deterministic ids, but a note or link removed from the vault is
//! left behind. Set [`LoadOptions::prune`] to mirror the vault instead: before
//! upserting, the loader tombstones any node/edge of the configured
//! label/type that the vault no longer contains, so the graph becomes a
//! faithful, rebuildable index of the current vault.
//!
//! Follows the same cadence contract as the parquet loader: the loader leaves
//! the final batch pending and the caller decides when to `commit_batch`.

use std::collections::HashSet;
use std::path::Path;

use namidb_core::NodeId;
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
    /// Mirror the vault instead of merging into it: tombstone every node of
    /// `label` and every edge of `edge_type` that the vault no longer
    /// contains. `false` (default) keeps the load additive. Pruning only ever
    /// touches the configured `label` / `edge_type`, so unrelated data in the
    /// namespace is left alone.
    pub prune: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            label: DEFAULT_LABEL.to_string(),
            edge_type: DEFAULT_EDGE_TYPE.to_string(),
            commit_every: DEFAULT_COMMIT_EVERY,
            prune: false,
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
    /// Stale nodes tombstoned because they are no longer in the vault (only
    /// non-zero when [`LoadOptions::prune`] is set).
    pub notes_pruned: usize,
    /// Stale edges tombstoned because they are no longer in the vault (only
    /// non-zero when [`LoadOptions::prune`] is set).
    pub links_pruned: usize,
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

/// Ingest an already-parsed [`VaultGraph`]. Split out so the resolution +
/// reconcile + write path is testable from an in-memory graph without
/// filesystem I/O.
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

    // Resolve every wikilink once: the edge list the vault wants, plus a count
    // of links that point nowhere.
    let mut desired_edges: Vec<(NodeId, NodeId)> = Vec::new();
    let mut dangling = 0usize;
    for note in &graph.notes {
        for target in &note.links {
            if known.contains(target.as_str()) {
                desired_edges.push((note.id, stable_node_id(target)));
            } else {
                dangling += 1;
            }
        }
    }

    let mut outcome = VaultLoadOutcome {
        name_collisions: collisions,
        links_dangling: dangling,
        ..Default::default()
    };
    let mut rows_since_commit = 0usize;

    // Reconcile deletions first, against the last committed state (nothing
    // from this load is pending yet, so the snapshot is accurate).
    if opts.prune {
        let desired_nodes: HashSet<NodeId> = graph.notes.iter().map(|n| n.id).collect();
        let desired_edge_set: HashSet<(NodeId, NodeId)> = desired_edges.iter().copied().collect();

        let (existing_nodes, existing_edges): (Vec<NodeId>, Vec<(NodeId, NodeId)>) = {
            let snap = writer.snapshot();
            let nodes = snap
                .scan_label(&opts.label)
                .await
                .map_err(|e| anyhow::anyhow!("scan {} nodes: {e}", opts.label))?;
            let edges = snap
                .scan_edge_type(&opts.edge_type)
                .await
                .map_err(|e| anyhow::anyhow!("scan {} edges: {e}", opts.edge_type))?;
            (
                nodes.iter().map(|n| n.id).collect(),
                edges.iter().map(|e| (e.src, e.dst)).collect(),
            )
        };

        for id in existing_nodes {
            if !desired_nodes.contains(&id) {
                writer.tombstone_node(opts.label.clone(), id)?;
                outcome.notes_pruned += 1;
                rows_since_commit += 1;
                maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
            }
        }
        for (src, dst) in existing_edges {
            if !desired_edge_set.contains(&(src, dst)) {
                writer.tombstone_edge(opts.edge_type.clone(), src, dst)?;
                outcome.links_pruned += 1;
                rows_since_commit += 1;
                maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
            }
        }
    }

    // Upsert the current vault: nodes, then resolved edges.
    for note in &graph.notes {
        let record = NodeWriteRecord {
            properties: note.properties.clone(),
            schema_version: 1,
        };
        writer.upsert_node(opts.label.clone(), note.id, &record)?;
        outcome.notes_loaded += 1;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }

    let edge_record = EdgeWriteRecord::default();
    for (src, dst) in &desired_edges {
        writer.upsert_edge(opts.edge_type.clone(), *src, *dst, &edge_record)?;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }
    outcome.links_resolved = desired_edges.len();

    Ok(outcome)
}

/// Fire a `commit_batch` once the pending row count reaches the cadence.
async fn maybe_commit(
    writer: &mut WriterSession,
    opts: &LoadOptions,
    rows_since_commit: &mut usize,
    outcome: &mut VaultLoadOutcome,
) -> anyhow::Result<()> {
    if opts.commit_every > 0 && *rows_since_commit >= opts.commit_every {
        writer.commit_batch().await?;
        outcome.commit_batches += 1;
        *rows_since_commit = 0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use namidb_core::{NamespaceId, Value};
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

    async fn open(ns: &str) -> WriterSession {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let paths = NamespacePaths::new("test", NamespaceId::new(ns).unwrap());
        WriterSession::open(store, paths).await.unwrap()
    }

    async fn note_titles(writer: &WriterSession) -> Vec<String> {
        let snap = writer.snapshot();
        let nodes = snap.scan_label("Note").await.unwrap();
        let mut titles: Vec<String> = nodes
            .iter()
            .filter_map(|n| match n.properties.get("title") {
                Some(Value::Str(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        titles.sort();
        titles
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

        let mut writer = open("vault-load").await;
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
        assert_eq!(outcome.notes_pruned, 0);
        assert_eq!(outcome.links_pruned, 0);
    }

    #[tokio::test]
    async fn prune_mirrors_the_vault_on_reingest() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "Alpha.md", "links to [[Beta]] and [[Gamma]]\n");
        write(dir, "Beta.md", "back to [[Alpha]]\n");
        write(dir, "Gamma.md", "a leaf\n");

        let mut writer = open("vault-prune").await;

        // First load: Alpha, Beta, Gamma with 3 edges.
        let first = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();
        assert_eq!(first.notes_loaded, 3);
        assert_eq!(first.links_resolved, 3);
        assert_eq!(note_titles(&writer).await, vec!["Alpha", "Beta", "Gamma"]);

        // Edit the vault: delete Gamma and drop Alpha's link to it.
        std::fs::remove_file(dir.join("Gamma.md")).unwrap();
        write(dir, "Alpha.md", "links to [[Beta]] only\n");

        // Re-load WITH prune: the graph must now mirror the new vault.
        let opts = LoadOptions {
            prune: true,
            ..Default::default()
        };
        let second = load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(second.notes_loaded, 2, "Alpha + Beta upserted");
        assert_eq!(second.notes_pruned, 1, "Gamma tombstoned");
        assert_eq!(second.links_pruned, 1, "stale Alpha->Gamma tombstoned");
        assert_eq!(second.links_resolved, 2, "Alpha->Beta, Beta->Alpha");
        assert_eq!(note_titles(&writer).await, vec!["Alpha", "Beta"]);
    }

    #[tokio::test]
    async fn additive_reingest_leaves_stale_data() {
        // The default (no prune) keeps removed notes, which is exactly why
        // prune exists. Pinned so the contrast stays intentional.
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "Alpha.md", "[[Beta]]\n");
        write(dir, "Beta.md", "leaf\n");

        let mut writer = open("vault-additive").await;
        load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        std::fs::remove_file(dir.join("Beta.md")).unwrap();
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.notes_pruned, 0);
        assert_eq!(
            note_titles(&writer).await,
            vec!["Alpha", "Beta"],
            "Beta lingers"
        );
    }
}
