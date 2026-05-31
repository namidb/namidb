//! Ingest a parsed vault into a [`WriterSession`] as a graph.
//!
//! Each note becomes a node (default label `Note`); each resolved link becomes
//! a `LINKS_TO` edge and each resolved embed (`![[X]]`) a distinct `EMBEDS`
//! edge. Dangling targets (no matching note) are counted and, by default,
//! produce no edge so every edge endpoint is a real note. With
//! [`LoadOptions::placeholders`] a dangling target instead gets a stub `:Note`
//! node (`placeholder: true`, no `path`/`body`) and a real edge to it, so
//! unresolved references appear in the graph like Obsidian's graph view.
//!
//! Each distinct string tag on a note becomes a shared `:Tag` node (one per
//! tag name), linked from the note by a `:TAGGED` edge, so tag-traversal
//! queries ("notes tagged X", "notes that share a tag") run on the graph.
//! Tag names are matched as written (case-sensitive), so `Rust` and `rust`
//! are two distinct tag nodes.
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

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use namidb_core::{NodeId, Value};
use namidb_storage::{EdgeWriteRecord, NodeWriteRecord, WriterSession};

use crate::id::stable_node_id;
use crate::parse::{parse_vault, VaultGraph};

const DEFAULT_LABEL: &str = "Note";
const DEFAULT_EDGE_TYPE: &str = "LINKS_TO";
/// Edge type for note embeds (`![[X]]`), kept distinct from `LINKS_TO`. Fixed.
const EMBED_EDGE_TYPE: &str = "EMBEDS";
/// Label for tag nodes and the edge type linking a note to its tags. These are
/// fixed (not configurable) so the tag sub-graph has a stable shape.
const TAG_LABEL: &str = "Tag";
const TAG_EDGE_TYPE: &str = "TAGGED";
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
    /// Mirror the vault instead of merging into it: tombstone every node and
    /// edge the vault no longer contains, across the note graph (`label` /
    /// `edge_type` / `EMBEDS`) and the tag graph (`Tag` / `TAGGED`). `false`
    /// (default) keeps the load additive. Pruning only touches those
    /// labels/types, so unrelated data in the namespace is left alone.
    pub prune: bool,
    /// Create stub `:Note` nodes (marked `placeholder: true`) for links/embeds
    /// whose target has no real note, so unresolved references show up in the
    /// graph like Obsidian's graph view. `false` (default) just counts them.
    pub placeholders: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            label: DEFAULT_LABEL.to_string(),
            edge_type: DEFAULT_EDGE_TYPE.to_string(),
            commit_every: DEFAULT_COMMIT_EVERY,
            prune: false,
            placeholders: false,
        }
    }
}

/// Outcome of a vault load.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VaultLoadOutcome {
    /// Notes upserted as nodes.
    pub notes_loaded: usize,
    /// Non-embed wikilinks/markdown links that resolved to a known note and
    /// produced a `LINKS_TO` edge.
    pub links_resolved: usize,
    /// Link targets that were not a known note (no edge written).
    pub links_dangling: usize,
    /// Embeds (`![[X]]`) that resolved to a known note and produced an
    /// `EMBEDS` edge.
    pub embeds_resolved: usize,
    /// Embed targets that were not a known note (no edge written).
    pub embeds_dangling: usize,
    /// Notes whose normalized key collided with an earlier note (last write
    /// wins; surfaced so silent overwrites are visible).
    pub name_collisions: usize,
    /// Stale nodes tombstoned because they are no longer in the vault (only
    /// non-zero when [`LoadOptions::prune`] is set).
    pub notes_pruned: usize,
    /// Stale `LINKS_TO` edges tombstoned (prune only).
    pub links_pruned: usize,
    /// Stale `EMBEDS` edges tombstoned (prune only).
    pub embeds_pruned: usize,
    /// Distinct `:Tag` nodes upserted (the union of all notes' tags).
    pub tags_loaded: usize,
    /// `:TAGGED` edges upserted (note -> tag).
    pub tag_links: usize,
    /// Stale `:Tag` nodes tombstoned (no longer used by any note; prune only).
    pub tags_pruned: usize,
    /// Stale `:TAGGED` edges tombstoned (prune only).
    pub tag_links_pruned: usize,
    /// Placeholder stub nodes upserted for unresolved targets (only non-zero
    /// when [`LoadOptions::placeholders`] is set).
    pub placeholders_created: usize,
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

    // Resolve links and embeds once: the edge lists the vault wants, plus
    // counts of targets that point nowhere. With `placeholders`, a dangling
    // target also gets a stub `:Note` node (keyed the same as the real note
    // would be, so creating that note later just upserts over the stub) and a
    // real edge to it, so the graph shows unresolved references like Obsidian.
    let mut desired_edges: Vec<(NodeId, NodeId)> = Vec::new();
    let mut desired_embed_edges: Vec<(NodeId, NodeId)> = Vec::new();
    let mut desired_placeholders: BTreeMap<NodeId, String> = BTreeMap::new();
    let mut dangling = 0usize;
    let mut embeds_dangling = 0usize;
    for note in &graph.notes {
        for target in &note.links {
            let tid = stable_node_id(target);
            if known.contains(target.as_str()) {
                desired_edges.push((note.id, tid));
            } else {
                dangling += 1;
                if opts.placeholders {
                    desired_edges.push((note.id, tid));
                    desired_placeholders
                        .entry(tid)
                        .or_insert_with(|| target.clone());
                }
            }
        }
        for target in &note.embeds {
            let tid = stable_node_id(target);
            if known.contains(target.as_str()) {
                desired_embed_edges.push((note.id, tid));
            } else {
                embeds_dangling += 1;
                if opts.placeholders {
                    desired_embed_edges.push((note.id, tid));
                    desired_placeholders
                        .entry(tid)
                        .or_insert_with(|| target.clone());
                }
            }
        }
    }

    // Resolve tags: one shared `:Tag` node per distinct tag name (the union
    // across notes), and a `:TAGGED` edge from each note to each of its tags.
    let mut desired_tags: BTreeMap<NodeId, String> = BTreeMap::new();
    let mut desired_tagged: Vec<(NodeId, NodeId)> = Vec::new();
    for note in &graph.notes {
        for tag in &note.tags {
            let tag_id = tag_node_id(tag);
            desired_tags.entry(tag_id).or_insert_with(|| tag.clone());
            desired_tagged.push((note.id, tag_id));
        }
    }

    let mut outcome = VaultLoadOutcome {
        name_collisions: collisions,
        links_dangling: dangling,
        embeds_dangling,
        ..Default::default()
    };
    let mut rows_since_commit = 0usize;

    // Reconcile deletions first, against the last committed state (nothing
    // from this load is pending yet, so the snapshot is accurate).
    if opts.prune {
        // Real notes plus any placeholder stubs still referenced this load.
        let desired_nodes: HashSet<NodeId> = graph
            .notes
            .iter()
            .map(|n| n.id)
            .chain(desired_placeholders.keys().copied())
            .collect();
        let desired_edge_set: HashSet<(NodeId, NodeId)> = desired_edges.iter().copied().collect();
        let desired_embed_set: HashSet<(NodeId, NodeId)> =
            desired_embed_edges.iter().copied().collect();
        let desired_tag_set: HashSet<NodeId> = desired_tags.keys().copied().collect();
        let desired_tagged_set: HashSet<(NodeId, NodeId)> =
            desired_tagged.iter().copied().collect();

        type IdVec = Vec<NodeId>;
        type PairVec = Vec<(NodeId, NodeId)>;
        #[allow(clippy::type_complexity)]
        let (existing_nodes, existing_edges, existing_embeds, existing_tags, existing_tagged): (
            IdVec,
            PairVec,
            PairVec,
            IdVec,
            PairVec,
        ) = {
            let snap = writer.snapshot();
            let nodes = snap
                .scan_label(&opts.label)
                .await
                .map_err(|e| anyhow::anyhow!("scan {} nodes: {e}", opts.label))?;
            let edges = snap
                .scan_edge_type(&opts.edge_type)
                .await
                .map_err(|e| anyhow::anyhow!("scan {} edges: {e}", opts.edge_type))?;
            let embeds = snap
                .scan_edge_type(EMBED_EDGE_TYPE)
                .await
                .map_err(|e| anyhow::anyhow!("scan {EMBED_EDGE_TYPE} edges: {e}"))?;
            let tags = snap
                .scan_label(TAG_LABEL)
                .await
                .map_err(|e| anyhow::anyhow!("scan {TAG_LABEL} nodes: {e}"))?;
            let tagged = snap
                .scan_edge_type(TAG_EDGE_TYPE)
                .await
                .map_err(|e| anyhow::anyhow!("scan {TAG_EDGE_TYPE} edges: {e}"))?;
            (
                nodes.iter().map(|n| n.id).collect(),
                edges.iter().map(|e| (e.src, e.dst)).collect(),
                embeds.iter().map(|e| (e.src, e.dst)).collect(),
                tags.iter().map(|n| n.id).collect(),
                tagged.iter().map(|e| (e.src, e.dst)).collect(),
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
        for (src, dst) in existing_embeds {
            if !desired_embed_set.contains(&(src, dst)) {
                writer.tombstone_edge(EMBED_EDGE_TYPE, src, dst)?;
                outcome.embeds_pruned += 1;
                rows_since_commit += 1;
                maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
            }
        }
        for id in existing_tags {
            if !desired_tag_set.contains(&id) {
                writer.tombstone_node(TAG_LABEL, id)?;
                outcome.tags_pruned += 1;
                rows_since_commit += 1;
                maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
            }
        }
        for (src, dst) in existing_tagged {
            if !desired_tagged_set.contains(&(src, dst)) {
                writer.tombstone_edge(TAG_EDGE_TYPE, src, dst)?;
                outcome.tag_links_pruned += 1;
                rows_since_commit += 1;
                maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
            }
        }
    }

    // Upsert the current vault: note nodes, link edges, tag nodes, tag edges.
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

    // Stub `:Note` nodes for unresolved targets (only when `placeholders` is
    // on). Marked `placeholder: true` and without `path`/`body`; creating the
    // real note later upserts over the stub (same label + id).
    for (id, name) in &desired_placeholders {
        let mut props = BTreeMap::new();
        props.insert("key".to_string(), Value::Str(name.clone()));
        props.insert("title".to_string(), Value::Str(name.clone()));
        props.insert("placeholder".to_string(), Value::Bool(true));
        let record = NodeWriteRecord {
            properties: props,
            schema_version: 1,
        };
        writer.upsert_node(opts.label.clone(), *id, &record)?;
        outcome.placeholders_created += 1;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }

    let edge_record = EdgeWriteRecord::default();
    for (src, dst) in &desired_edges {
        writer.upsert_edge(opts.edge_type.clone(), *src, *dst, &edge_record)?;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }
    // desired_edges holds real-note edges plus, when placeholders is on, one
    // edge per dangling link; subtract those to count only real resolutions.
    outcome.links_resolved = desired_edges.len() - if opts.placeholders { dangling } else { 0 };

    for (src, dst) in &desired_embed_edges {
        writer.upsert_edge(EMBED_EDGE_TYPE, *src, *dst, &edge_record)?;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }
    outcome.embeds_resolved = desired_embed_edges.len()
        - if opts.placeholders {
            embeds_dangling
        } else {
            0
        };

    for (tag_id, name) in &desired_tags {
        let mut props = BTreeMap::new();
        props.insert("name".to_string(), Value::Str(name.clone()));
        let record = NodeWriteRecord {
            properties: props,
            schema_version: 1,
        };
        writer.upsert_node(TAG_LABEL, *tag_id, &record)?;
        outcome.tags_loaded += 1;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }
    for (src, dst) in &desired_tagged {
        writer.upsert_edge(TAG_EDGE_TYPE, *src, *dst, &edge_record)?;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }
    outcome.tag_links = desired_tagged.len();

    Ok(outcome)
}

/// Stable id for a tag node, namespaced with NUL bytes so a tag never collides
/// with a note whose key is the same text (note keys derive from filenames and
/// cannot contain NUL).
fn tag_node_id(tag: &str) -> NodeId {
    stable_node_id(&format!("\u{0}tag\u{0}{tag}"))
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

    async fn tag_names(writer: &WriterSession) -> Vec<String> {
        let snap = writer.snapshot();
        let nodes = snap.scan_label("Tag").await.unwrap();
        let mut names: Vec<String> = nodes
            .iter()
            .filter_map(|n| match n.properties.get("name") {
                Some(Value::Str(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn tags_become_shared_nodes_and_edges() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "---\ntags: [rust, db]\n---\nbody\n");
        write(dir, "B.md", "uses #rust inline\n");

        let mut writer = open("vault-tags").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        // Two distinct tags; `rust` is shared by A and B (one node, two edges).
        assert_eq!(out.tags_loaded, 2);
        assert_eq!(out.tag_links, 3, "A->rust, A->db, B->rust");
        assert_eq!(tag_names(&writer).await, vec!["db", "rust"]);

        // "notes tagged rust" is a reverse traversal of the shared tag node.
        let snap = writer.snapshot();
        let taggers = snap.in_edges("TAGGED", tag_node_id("rust")).await.unwrap();
        assert_eq!(taggers.edges.len(), 2, "A and B both tagged rust");
    }

    #[tokio::test]
    async fn duplicate_frontmatter_tags_count_once() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "---\ntags: [rust, rust]\n---\nx\n");

        let mut writer = open("vault-duptag").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.tags_loaded, 1);
        assert_eq!(out.tag_links, 1, "a tag listed twice still links once");
        let snap = writer.snapshot();
        assert_eq!(
            snap.in_edges("TAGGED", tag_node_id("rust"))
                .await
                .unwrap()
                .edges
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn prune_removes_unused_tags() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "---\ntags: [rust, db]\n---\nx\n");

        let mut writer = open("vault-tagprune").await;
        load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();
        assert_eq!(tag_names(&writer).await, vec!["db", "rust"]);

        // Drop the `db` tag; reload with prune.
        write(dir, "A.md", "---\ntags: [rust]\n---\nx\n");
        let opts = LoadOptions {
            prune: true,
            ..Default::default()
        };
        let out = load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.tags_pruned, 1, "unused db tag node removed");
        assert_eq!(out.tag_links_pruned, 1, "A->db edge removed");
        assert_eq!(tag_names(&writer).await, vec!["rust"]);
    }

    #[tokio::test]
    async fn embeds_use_a_distinct_edge_type() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "link [[B]] and embed ![[C]]\n");
        write(dir, "B.md", "b\n");
        write(dir, "C.md", "c\n");

        let mut writer = open("vault-embeds").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.links_resolved, 1, "A->B is a LINKS_TO");
        assert_eq!(out.embeds_resolved, 1, "A->C is an EMBEDS");

        let snap = writer.snapshot();
        let a = stable_node_id("a");
        let links = snap.out_edges("LINKS_TO", a).await.unwrap();
        assert_eq!(links.edges.len(), 1);
        assert_eq!(links.edges[0].dst, stable_node_id("b"));
        let embeds = snap.out_edges("EMBEDS", a).await.unwrap();
        assert_eq!(embeds.edges.len(), 1);
        assert_eq!(embeds.edges[0].dst, stable_node_id("c"));
    }

    #[tokio::test]
    async fn prune_removes_stale_embeds() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "embed ![[B]]\n");
        write(dir, "B.md", "b\n");

        let mut writer = open("vault-embedprune").await;
        load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        // Drop the embed; reload with prune.
        write(dir, "A.md", "no embed now\n");
        let opts = LoadOptions {
            prune: true,
            ..Default::default()
        };
        let out = load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.embeds_resolved, 0);
        assert_eq!(out.embeds_pruned, 1, "stale A->B embed tombstoned");
        let snap = writer.snapshot();
        assert_eq!(
            snap.out_edges("EMBEDS", stable_node_id("a"))
                .await
                .unwrap()
                .edges
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn placeholders_materialize_unresolved_targets() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "links to [[Missing]] and embeds ![[Gone]]\n");

        let mut writer = open("vault-ph").await;
        let opts = LoadOptions {
            placeholders: true,
            ..Default::default()
        };
        let out = load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.links_resolved, 0, "no real link targets");
        assert_eq!(out.links_dangling, 1);
        assert_eq!(out.embeds_dangling, 1);
        assert_eq!(out.placeholders_created, 2, "Missing + Gone stubs");

        let snap = writer.snapshot();
        let missing = snap
            .lookup_node("Note", stable_node_id("missing"))
            .await
            .unwrap()
            .expect("placeholder node exists");
        assert_eq!(
            missing.properties.get("placeholder"),
            Some(&Value::Bool(true))
        );
        assert!(!missing.properties.contains_key("path"), "stub has no path");
        // The dangling link became a real edge to the stub.
        let back = snap
            .in_edges("LINKS_TO", stable_node_id("missing"))
            .await
            .unwrap();
        assert_eq!(back.edges.len(), 1, "A -> Missing placeholder");
    }

    #[tokio::test]
    async fn default_off_makes_no_placeholder_node() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "[[Missing]]\n");

        let mut writer = open("vault-noph").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.links_dangling, 1);
        assert_eq!(out.placeholders_created, 0);
        let snap = writer.snapshot();
        assert!(
            snap.lookup_node("Note", stable_node_id("missing"))
                .await
                .unwrap()
                .is_none(),
            "no stub by default"
        );
    }

    #[tokio::test]
    async fn placeholder_is_promoted_when_the_real_note_appears() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "[[Missing]]\n");

        let mut writer = open("vault-phpromote").await;
        let opts = LoadOptions {
            placeholders: true,
            ..Default::default()
        };
        load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        // Create the previously-missing note; reload with prune.
        write(dir, "Missing.md", "now real\n");
        let opts2 = LoadOptions {
            placeholders: true,
            prune: true,
            ..Default::default()
        };
        load_vault(dir, &mut writer, &opts2).await.unwrap();
        writer.commit_batch().await.unwrap();

        // Same id, so the stub is upserted into a real note (path set, mark gone).
        let snap = writer.snapshot();
        let m = snap
            .lookup_node("Note", stable_node_id("missing"))
            .await
            .unwrap()
            .expect("note exists");
        assert_eq!(
            m.properties.get("placeholder"),
            None,
            "promoted to real note"
        );
        assert!(m.properties.contains_key("path"), "real note has a path");
    }

    #[tokio::test]
    async fn deleting_a_still_linked_note_leaves_a_stub_under_prune() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "[[B]]\n");
        write(dir, "B.md", "b\n");

        let mut writer = open("vault-phdel").await;
        let opts = LoadOptions {
            placeholders: true,
            prune: true,
            ..Default::default()
        };
        load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        // Delete B but keep A's link to it: the reference is now unresolved, so
        // B should survive prune as a placeholder stub (intended semantics).
        std::fs::remove_file(dir.join("B.md")).unwrap();
        let out = load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.placeholders_created, 1, "B becomes a stub");
        let snap = writer.snapshot();
        let b = snap
            .lookup_node("Note", stable_node_id("b"))
            .await
            .unwrap()
            .expect("B survives as a stub");
        assert_eq!(b.properties.get("placeholder"), Some(&Value::Bool(true)));
    }
}
