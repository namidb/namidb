//! Ingest a parsed vault into a [`WriterSession`] as a graph.
//!
//! Each note becomes a node (default label `Note`); each resolved link becomes
//! a `LINKS_TO` edge and each resolved embed (`![[X]]`) a distinct `EMBEDS`
//! edge. A link/embed target resolves by note key or by a frontmatter `aliases`
//! entry (a real note key wins over an alias). Dangling targets (no matching
//! note or alias) are counted and, by default,
//! produce no edge so every edge endpoint is a real note. With
//! [`LoadOptions::placeholders`] a dangling target instead gets a stub `:Note`
//! node (`placeholder: true`, no `path`/`body`) and a real edge to it, so
//! unresolved references appear in the graph like Obsidian's graph view.
//!
//! Each distinct string tag on a note becomes a shared `:Tag` node (one per
//! tag name), linked from the note by a `:TAGGED` edge, so tag-traversal
//! queries ("notes tagged X", "notes that share a tag") run on the graph.
//! Tag names are matched as written (case-sensitive), so `Rust` and `rust`
//! are two distinct tag nodes. A nested tag (`area/db`) also gets its ancestor
//! `:Tag` nodes and a child-to-parent `:SUBTAG_OF` edge per level, so the tag
//! tree is a real sub-graph (the note stays `:TAGGED` to the leaf it wrote).
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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use namidb_core::{NodeId, Value};
use namidb_storage::{EdgeWriteRecord, NodeWriteRecord, WriterSession};

use crate::embed::Embedder;
use crate::id::stable_node_id;
use crate::parse::{parse_vault, ParsedNote, VaultGraph};

const DEFAULT_LABEL: &str = "Note";
const DEFAULT_EDGE_TYPE: &str = "LINKS_TO";
/// Edge type for note embeds (`![[X]]`), kept distinct from `LINKS_TO`. Fixed.
const EMBED_EDGE_TYPE: &str = "EMBEDS";
/// Label for tag nodes and the edge type linking a note to its tags. These are
/// fixed (not configurable) so the tag sub-graph has a stable shape.
const TAG_LABEL: &str = "Tag";
const TAG_EDGE_TYPE: &str = "TAGGED";
/// Edge type linking a nested tag to its immediate parent (`area/db` ->
/// `area`), so the tag tree is a real sub-graph: child `:SUBTAG_OF` parent.
const SUBTAG_EDGE_TYPE: &str = "SUBTAG_OF";
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
    /// Optional text embedder. When set, every (re-)written note gets an
    /// `embedding` property (a `Value::Vec`) computed from its title + body, so
    /// `cosine_similarity(...)` queries and the MCP `vector_search` tool have
    /// vectors to rank. `None` (default) writes no embeddings. A sync only
    /// re-embeds notes whose content actually changed.
    pub embedder: Option<Arc<dyn Embedder>>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            label: DEFAULT_LABEL.to_string(),
            edge_type: DEFAULT_EDGE_TYPE.to_string(),
            commit_every: DEFAULT_COMMIT_EVERY,
            prune: false,
            placeholders: false,
            embedder: None,
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
    /// Distinct frontmatter aliases registered as resolvable names (excludes
    /// aliases shadowed by a real note key or by an earlier alias).
    pub aliases_registered: usize,
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
    /// `:SUBTAG_OF` edges upserted (nested tag -> immediate parent).
    pub subtag_edges: usize,
    /// Stale `:SUBTAG_OF` edges tombstoned (prune only).
    pub subtag_edges_pruned: usize,
    /// Placeholder stub nodes upserted for unresolved targets (only non-zero
    /// when [`LoadOptions::placeholders`] is set).
    pub placeholders_created: usize,
    /// `commit_batch` calls fired during the load (excludes the caller's
    /// final flush).
    pub commit_batches: usize,
}

/// Outcome of an incremental [`sync_vault`]/[`sync_graph`]. Wraps the same
/// write counts a load reports, plus the change classification that drove the
/// sync.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VaultSyncOutcome {
    /// The underlying write counts (edges/tags/prune), as for a load. Its
    /// `notes_loaded` counts only the note bodies actually (re-)written, so an
    /// all-unchanged sync writes zero notes.
    pub load: VaultLoadOutcome,
    /// Notes new since the last sync.
    pub notes_added: usize,
    /// Notes whose content hash changed since the last sync.
    pub notes_modified: usize,
    /// Notes left untouched because their content hash was unchanged (no body
    /// re-write).
    pub notes_unchanged: usize,
    /// Notes gone from disk since the last sync.
    pub notes_deleted: usize,
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
    load_graph_inner(graph, writer, opts, None).await
}

/// Shared write path for both a full load and an incremental sync. `prev_state`
/// is `None` for a load and `Some` for a sync. In sync mode the vault is always
/// mirrored (prune on), the existing node ids are read through a column
/// projection so note bodies are never loaded, and a note whose content hash
/// matches the prior state is left in place rather than re-written. Everything
/// else (edge/tag/stub reconcile) is identical to a prune-load, so a sync
/// converges to the same graph a fresh prune-load of the same disk would.
async fn load_graph_inner(
    graph: &VaultGraph,
    writer: &mut WriterSession,
    opts: &LoadOptions,
    prev_state: Option<&VaultState>,
) -> anyhow::Result<VaultLoadOutcome> {
    // A sync always mirrors the vault; a plain load prunes only if asked.
    let prune = opts.prune || prev_state.is_some();
    // The set of normalized keys that exist as real notes, used to tell a
    // resolved link from a dangling one.
    let mut known: HashSet<&str> = HashSet::with_capacity(graph.notes.len());
    let mut collisions = 0usize;
    for note in &graph.notes {
        if !known.insert(note.key.as_str()) {
            collisions += 1;
        }
    }

    // Frontmatter aliases: a `[[Alias]]` resolves to the note that declares the
    // alias. A real note key always wins over an alias, and the first note (in
    // path order) to declare an alias wins a clash, so resolution is
    // deterministic.
    let mut alias_map: HashMap<&str, NodeId> = HashMap::new();
    let mut aliases_registered = 0usize;
    for note in &graph.notes {
        for alias in &note.aliases {
            if known.contains(alias.as_str()) {
                continue; // a real note already owns this name
            }
            if let std::collections::hash_map::Entry::Vacant(slot) = alias_map.entry(alias.as_str())
            {
                slot.insert(note.id);
                aliases_registered += 1;
            }
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
            if known.contains(target.as_str()) {
                desired_edges.push((note.id, stable_node_id(target)));
            } else if let Some(&dst) = alias_map.get(target.as_str()) {
                desired_edges.push((note.id, dst)); // resolved via an alias
            } else {
                dangling += 1;
                if opts.placeholders {
                    let tid = stable_node_id(target);
                    desired_edges.push((note.id, tid));
                    desired_placeholders
                        .entry(tid)
                        .or_insert_with(|| target.clone());
                }
            }
        }
        for target in &note.embeds {
            if known.contains(target.as_str()) {
                desired_embed_edges.push((note.id, stable_node_id(target)));
            } else if let Some(&dst) = alias_map.get(target.as_str()) {
                desired_embed_edges.push((note.id, dst)); // resolved via an alias
            } else {
                embeds_dangling += 1;
                if opts.placeholders {
                    let tid = stable_node_id(target);
                    desired_embed_edges.push((note.id, tid));
                    desired_placeholders
                        .entry(tid)
                        .or_insert_with(|| target.clone());
                }
            }
        }
    }

    // Two distinct link targets can resolve to the same note (e.g. a note
    // linked by two of its aliases, or by both its real name and an alias),
    // producing duplicate edge pairs. Dedup so each edge is written and counted
    // once; the dangling-stub count is unaffected (stub targets are distinct).
    dedup_pairs(&mut desired_edges);
    dedup_pairs(&mut desired_embed_edges);

    // Resolve tags: one shared `:Tag` node per distinct tag name (the union
    // across notes), and a `:TAGGED` edge from each note to each of its tags.
    // A nested tag (`area/db`) also materializes its ancestor `:Tag` nodes and
    // a child-to-parent `:SUBTAG_OF` edge per level, so the tag tree is a real
    // sub-graph. The note stays `:TAGGED` to the leaf it wrote, not the
    // ancestors, so `tags_of` reflects what the author typed.
    let mut desired_tags: BTreeMap<NodeId, String> = BTreeMap::new();
    let mut desired_tagged: Vec<(NodeId, NodeId)> = Vec::new();
    let mut desired_subtags: BTreeSet<(NodeId, NodeId)> = BTreeSet::new();
    for note in &graph.notes {
        for tag in &note.tags {
            let tag_id = tag_node_id(tag);
            desired_tags.entry(tag_id).or_insert_with(|| tag.clone());
            desired_tagged.push((note.id, tag_id));
            register_tag_hierarchy(tag, &mut desired_tags, &mut desired_subtags);
        }
    }

    let mut outcome = VaultLoadOutcome {
        name_collisions: collisions,
        aliases_registered,
        links_dangling: dangling,
        embeds_dangling,
        ..Default::default()
    };
    let mut rows_since_commit = 0usize;

    // Reconcile deletions first, against the last committed state (nothing
    // from this load is pending yet, so the snapshot is accurate).
    if prune {
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
        let desired_subtag_set: HashSet<(NodeId, NodeId)> =
            desired_subtags.iter().copied().collect();

        type IdVec = Vec<NodeId>;
        type PairVec = Vec<(NodeId, NodeId)>;
        #[allow(clippy::type_complexity)]
        let (
            existing_nodes,
            existing_edges,
            existing_embeds,
            existing_tags,
            existing_tagged,
            existing_subtags,
        ): (IdVec, PairVec, PairVec, IdVec, PairVec, PairVec) = {
            let snap = writer.snapshot();
            // Only the ids are needed for the deletion diff, so project a
            // single column rather than materializing note bodies.
            let nodes = snap
                .scan_label_with_predicates_and_projection(
                    &opts.label,
                    &[],
                    Some(&["key".to_string()]),
                )
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
            let subtags = snap
                .scan_edge_type(SUBTAG_EDGE_TYPE)
                .await
                .map_err(|e| anyhow::anyhow!("scan {SUBTAG_EDGE_TYPE} edges: {e}"))?;
            (
                nodes.iter().map(|n| n.id).collect(),
                edges.iter().map(|e| (e.src, e.dst)).collect(),
                embeds.iter().map(|e| (e.src, e.dst)).collect(),
                tags.iter().map(|n| n.id).collect(),
                tagged.iter().map(|e| (e.src, e.dst)).collect(),
                subtags.iter().map(|e| (e.src, e.dst)).collect(),
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
        for (src, dst) in existing_subtags {
            if !desired_subtag_set.contains(&(src, dst)) {
                writer.tombstone_edge(SUBTAG_EDGE_TYPE, src, dst)?;
                outcome.subtag_edges_pruned += 1;
                rows_since_commit += 1;
                maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
            }
        }
    }

    // Upsert the current vault's note nodes, then edges, tag nodes and tag
    // edges. In sync mode a note whose stored content hash still matches is
    // left in place: its node is already byte-identical, so re-writing the
    // body would be wasted work (the expensive part of a load).
    let to_write: Vec<&ParsedNote> = graph
        .notes
        .iter()
        .filter(|note| match prev_state {
            Some(prev) => !matches!(
                (prev.get(&note.key), note_hash(note)),
                (Some(Some(stored)), Some(cur)) if stored.as_str() == cur
            ),
            None => true,
        })
        .collect();

    // One batched embedding round-trip for the whole write set, so a remote
    // embedder issues a single HTTP call (chunked internally) instead of one
    // per note. In sync mode only changed notes are embedded, so steady-state
    // cost tracks edits, not vault size. Output is 1:1 with `to_write`.
    let embeddings: Vec<Option<Vec<f32>>> = match &opts.embedder {
        Some(embedder) => {
            let texts: Vec<String> = to_write.iter().map(|n| note_embedding_text(n)).collect();
            embedder
                .embed_batch(&texts)
                .await?
                .into_iter()
                .map(Some)
                .collect()
        }
        None => vec![None; to_write.len()],
    };

    for (note, embedding) in to_write.into_iter().zip(embeddings) {
        let mut properties = note.properties.clone();
        if let Some(vec) = embedding {
            properties.insert("embedding".to_string(), Value::Vec(vec));
        }
        let record = NodeWriteRecord {
            properties,
            schema_version: 1,
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

    for (child, parent) in &desired_subtags {
        writer.upsert_edge(SUBTAG_EDGE_TYPE, *child, *parent, &edge_record)?;
        rows_since_commit += 1;
        maybe_commit(writer, opts, &mut rows_since_commit, &mut outcome).await?;
    }
    outcome.subtag_edges = desired_subtags.len();

    Ok(outcome)
}

/// Sync an already-parsed [`VaultGraph`] against the prior [`VaultState`],
/// re-indexing only what changed. Bodies of unchanged notes are not re-written
/// and note bodies are never loaded to detect deletions, but edges and tags are
/// reconciled exactly as a prune-load, so the resulting graph is identical to a
/// fresh prune-load of the same disk state.
pub async fn sync_graph(
    graph: &VaultGraph,
    prev_state: &VaultState,
    writer: &mut WriterSession,
    opts: &LoadOptions,
) -> anyhow::Result<VaultSyncOutcome> {
    let diff = diff_vault(graph, prev_state);
    let counts = VaultSyncOutcome {
        notes_added: diff.added.len(),
        notes_modified: diff.modified.len(),
        notes_unchanged: diff.unchanged.len(),
        notes_deleted: diff.deleted.len(),
        load: VaultLoadOutcome::default(),
    };
    let load = load_graph_inner(graph, writer, opts, Some(prev_state)).await?;
    Ok(VaultSyncOutcome { load, ..counts })
}

/// Parse the vault at `dir`, read the prior [`VaultState`] from the graph, and
/// sync incrementally. The first sync over an empty namespace classifies every
/// note as added, so it behaves like a full load.
pub async fn sync_vault(
    dir: &Path,
    writer: &mut WriterSession,
    opts: &LoadOptions,
) -> anyhow::Result<VaultSyncOutcome> {
    let prev_state = read_vault_state(writer, &opts.label).await?;
    let graph = parse_vault(dir)?;
    sync_graph(&graph, &prev_state, writer, opts).await
}

/// The stored content hash of a parsed note, if present.
fn note_hash(note: &ParsedNote) -> Option<&str> {
    match note.properties.get("content_hash") {
        Some(Value::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// The text an embedder sees for a note: its title, then its body. The title is
/// included because short notes (and ones reached only by their filename)
/// otherwise carry too few body tokens to rank well.
fn note_embedding_text(note: &ParsedNote) -> String {
    match note.properties.get("body") {
        Some(Value::Str(body)) if !body.is_empty() => format!("{}\n{}", note.title, body),
        _ => note.title.clone(),
    }
}

/// Drop duplicate `(src, dst)` pairs in place, preserving first-seen order, so
/// an edge that several link targets resolve to is written and counted once.
fn dedup_pairs(pairs: &mut Vec<(NodeId, NodeId)>) {
    let mut seen = HashSet::new();
    pairs.retain(|pair| seen.insert(*pair));
}

/// Stable id for a tag node, namespaced with NUL bytes so a tag never collides
/// with a note whose key is the same text (note keys derive from filenames and
/// cannot contain NUL).
fn tag_node_id(tag: &str) -> NodeId {
    stable_node_id(&format!("\u{0}tag\u{0}{tag}"))
}

/// Register the ancestor tag nodes and child-to-parent `:SUBTAG_OF` edges of a
/// nested tag. For `area/db/x` this adds nodes `area/db` and `area` and edges
/// `area/db/x -> area/db` and `area/db -> area`. The leaf node itself is added
/// by the caller. A non-nested tag (no `/`) contributes nothing.
fn register_tag_hierarchy(
    tag: &str,
    desired_tags: &mut BTreeMap<NodeId, String>,
    desired_subtags: &mut BTreeSet<(NodeId, NodeId)>,
) {
    let mut child = tag;
    while let Some(slash) = child.rfind('/') {
        let parent = &child[..slash];
        // A leading/empty segment (e.g. `/x` or `a//b`) is not a real parent.
        if parent.is_empty() {
            break;
        }
        let child_id = tag_node_id(child);
        let parent_id = tag_node_id(parent);
        desired_tags
            .entry(parent_id)
            .or_insert_with(|| parent.to_string());
        desired_subtags.insert((child_id, parent_id));
        child = parent;
    }
}

/// The last-loaded state for an incremental sync: each real note's stored
/// `content_hash`, keyed by its normalized key. `None` marks a real note that
/// predates the hash (loaded before `content_hash` existed); it is treated as
/// "changed" so the next sync backfills it.
pub type VaultState = HashMap<String, Option<String>>;

/// Read the last-loaded [`VaultState`] from the graph through a column
/// projection, so note bodies are never materialized (only `key`, `path` and
/// `content_hash` columns are read). Placeholder stubs are excluded by their
/// missing `path`, so a dangling-reference stub is never mistaken for a real
/// note that a sync would have to delete.
pub async fn read_vault_state(writer: &WriterSession, label: &str) -> anyhow::Result<VaultState> {
    let projection = [
        "key".to_string(),
        "path".to_string(),
        "content_hash".to_string(),
    ];
    let snap = writer.snapshot();
    let nodes = snap
        .scan_label_with_predicates_and_projection(label, &[], Some(&projection))
        .await
        .map_err(|e| anyhow::anyhow!("scan {label} state: {e}"))?;
    let mut state = VaultState::with_capacity(nodes.len());
    for node in &nodes {
        // A real note has both a key and a path; a stub has a key but no path.
        let (Some(Value::Str(key)), Some(Value::Str(_path))) =
            (node.properties.get("key"), node.properties.get("path"))
        else {
            continue;
        };
        let hash = match node.properties.get("content_hash") {
            Some(Value::Str(h)) => Some(h.clone()),
            _ => None,
        };
        state.insert(key.clone(), hash);
    }
    Ok(state)
}

/// How a re-parsed vault differs from the last-loaded [`VaultState`]. Indices
/// point into the parsed graph's `notes`; `deleted` holds the keys of real
/// notes that are gone from disk.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct VaultDiff {
    /// Notes whose key was not in the previous state.
    pub added: Vec<usize>,
    /// Notes whose key was present but whose content hash changed (or was
    /// missing, e.g. a pre-hash note).
    pub modified: Vec<usize>,
    /// Notes whose content hash matched the previous state.
    pub unchanged: Vec<usize>,
    /// Keys of real notes that were in the previous state but no longer on disk.
    pub deleted: Vec<String>,
}

/// Classify each note in a freshly parsed `graph` against `prev` by comparing
/// `content_hash`, and find the keys deleted from disk. A note counts as
/// unchanged only when both the stored and the current hash are present and
/// equal; anything else (added key, changed hash, missing hash) is re-indexed.
pub fn diff_vault(graph: &VaultGraph, prev: &VaultState) -> VaultDiff {
    let mut diff = VaultDiff::default();
    let mut on_disk: HashSet<&str> = HashSet::with_capacity(graph.notes.len());
    for (i, note) in graph.notes.iter().enumerate() {
        on_disk.insert(note.key.as_str());
        let cur = match note.properties.get("content_hash") {
            Some(Value::Str(s)) => Some(s.as_str()),
            _ => None,
        };
        match prev.get(&note.key) {
            None => diff.added.push(i),
            Some(prev_hash) => match (prev_hash.as_deref(), cur) {
                (Some(p), Some(c)) if p == c => diff.unchanged.push(i),
                _ => diff.modified.push(i),
            },
        }
    }
    for key in prev.keys() {
        if !on_disk.contains(key.as_str()) {
            diff.deleted.push(key.clone());
        }
    }
    diff
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
    async fn frontmatter_links_become_edges() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        // A wikilink in a frontmatter property, not in the body.
        write(
            dir,
            "Child.md",
            "---\nup: \"[[Parent]]\"\n---\njust a child\n",
        );
        write(dir, "Parent.md", "the parent\n");

        let mut writer = open("vault-fmlink").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.links_resolved, 1, "Child -> Parent via frontmatter");
        let snap = writer.snapshot();
        let edges = snap
            .out_edges("LINKS_TO", stable_node_id("child"))
            .await
            .unwrap();
        assert_eq!(edges.edges.len(), 1);
        assert_eq!(edges.edges[0].dst, stable_node_id("parent"));
    }

    #[tokio::test]
    async fn frontmatter_aliases_resolve_links() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(
            dir,
            "User Role.md",
            "---\naliases: [\"U-R\"]\n---\nthe role\n",
        );
        write(
            dir,
            "Project X.md",
            "---\naliases: [\"px\"]\n---\nthe project\n",
        );
        write(dir, "Other.md", "see [[U-R]] and [[px]] and [[Nope]]\n");

        let mut writer = open("vault-alias").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.aliases_registered, 2, "U-R and px");
        // The two aliases resolve to their real notes; Nope dangles.
        assert_eq!(out.links_resolved, 2);
        assert_eq!(out.links_dangling, 1, "Nope");
        let snap = writer.snapshot();
        let mut dsts: Vec<_> = snap
            .out_edges("LINKS_TO", stable_node_id("other"))
            .await
            .unwrap()
            .edges
            .iter()
            .map(|e| e.dst)
            .collect();
        dsts.sort();
        let mut want = vec![stable_node_id("user-role"), stable_node_id("project-x")];
        want.sort();
        assert_eq!(dsts, want, "aliases resolve to the real notes");
    }

    #[tokio::test]
    async fn two_aliases_of_one_note_make_a_single_edge() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        // Target is known by two aliases; Source links both.
        write(
            dir,
            "Target.md",
            "---\naliases: [\"Foo\", \"Bar\"]\n---\nt\n",
        );
        write(dir, "Source.md", "see [[Foo]] and [[Bar]]\n");

        let mut writer = open("vault-alias-fanin").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        // Both aliases collapse to one edge, counted once.
        assert_eq!(out.links_resolved, 1, "one physical edge, counted once");
        let snap = writer.snapshot();
        let edges = snap
            .out_edges("LINKS_TO", stable_node_id("source"))
            .await
            .unwrap();
        assert_eq!(edges.edges.len(), 1);
        assert_eq!(edges.edges[0].dst, stable_node_id("target"));
    }

    #[tokio::test]
    async fn a_real_note_key_wins_over_an_alias() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "Foo.md", "the real foo\n");
        write(
            dir,
            "Bar.md",
            "---\naliases: [\"Foo\"]\n---\nbar aliases foo\n",
        );
        write(dir, "Link.md", "see [[Foo]]\n");

        let mut writer = open("vault-alias-shadow").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(
            out.aliases_registered, 0,
            "alias Foo shadowed by the real note"
        );
        let snap = writer.snapshot();
        let edges = snap
            .out_edges("LINKS_TO", stable_node_id("link"))
            .await
            .unwrap();
        assert_eq!(edges.edges.len(), 1);
        assert_eq!(edges.edges[0].dst, stable_node_id("foo"), "real Foo wins");
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
    async fn nested_tags_build_a_subtag_hierarchy() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "uses #area/db/x and #area/web\n");

        let mut writer = open("vault-nested-tags").await;
        let out = load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        // Distinct tags: the two leaves plus ancestors area/db and area.
        assert_eq!(out.tags_loaded, 4, "area/db/x, area/web, area/db, area");
        // area/db/x->area/db, area/db->area, area/web->area.
        assert_eq!(out.subtag_edges, 3);

        let snap = writer.snapshot();
        // area has two direct children (incoming SUBTAG_OF).
        assert_eq!(
            snap.in_edges("SUBTAG_OF", tag_node_id("area"))
                .await
                .unwrap()
                .edges
                .len(),
            2,
            "area/db and area/web point at area"
        );
        // The note is TAGGED to the leaves it wrote, never the synthetic
        // ancestors.
        assert_eq!(
            snap.in_edges("TAGGED", tag_node_id("area"))
                .await
                .unwrap()
                .edges
                .len(),
            0,
            "no direct TAGGED to the ancestor area"
        );
        assert_eq!(
            snap.in_edges("TAGGED", tag_node_id("area/db/x"))
                .await
                .unwrap()
                .edges
                .len(),
            1,
            "A tagged the leaf area/db/x"
        );
    }

    #[tokio::test]
    async fn prune_reconciles_the_subtag_hierarchy() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "#area/db and #area/web\n");

        let mut writer = open("vault-nested-prune").await;
        load_vault(dir, &mut writer, &LoadOptions::default())
            .await
            .unwrap();
        writer.commit_batch().await.unwrap();

        // Drop area/web; reload with prune. `area` stays (area/db still uses
        // it), but area/web and its SUBTAG_OF edge go.
        write(dir, "A.md", "#area/db only\n");
        let opts = LoadOptions {
            prune: true,
            ..Default::default()
        };
        let out = load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        assert_eq!(out.subtag_edges_pruned, 1, "area/web->area removed");
        let snap = writer.snapshot();
        assert!(
            snap.lookup_node("Tag", tag_node_id("area/web"))
                .await
                .unwrap()
                .is_none(),
            "area/web tag node gone"
        );
        assert!(
            snap.lookup_node("Tag", tag_node_id("area"))
                .await
                .unwrap()
                .is_some(),
            "area kept, area/db still uses it"
        );
        assert_eq!(
            snap.in_edges("SUBTAG_OF", tag_node_id("area"))
                .await
                .unwrap()
                .edges
                .len(),
            1,
            "only area/db->area remains"
        );
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

    #[test]
    fn diff_classifies_notes_against_prev_state() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "alpha\n");
        write(dir, "B.md", "beta\n");
        write(dir, "C.md", "gamma\n");
        let graph = parse_vault(dir).unwrap();

        let hash_of = |key: &str| -> String {
            let note = graph.notes.iter().find(|n| n.key == key).unwrap();
            match note.properties.get("content_hash") {
                Some(Value::Str(s)) => s.clone(),
                _ => panic!("note {key} has no content_hash"),
            }
        };
        let mut prev = VaultState::new();
        prev.insert("a".into(), Some(hash_of("a"))); // unchanged
        prev.insert("b".into(), Some("stale".into())); // modified (hash differs)
        prev.insert("d".into(), Some("gone".into())); // deleted (absent on disk)
                                                      // "c" is absent from prev, so it is added.

        let diff = diff_vault(&graph, &prev);
        let keys = |idx: &[usize]| -> Vec<String> {
            idx.iter().map(|&i| graph.notes[i].key.clone()).collect()
        };
        assert_eq!(keys(&diff.added), vec!["c"]);
        assert_eq!(keys(&diff.modified), vec!["b"]);
        assert_eq!(keys(&diff.unchanged), vec!["a"]);
        assert_eq!(diff.deleted, vec!["d".to_string()]);
    }

    #[tokio::test]
    async fn read_vault_state_excludes_stubs_and_carries_hashes() {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        write(dir, "A.md", "links to [[Missing]]\n");

        let mut writer = open("vault-state").await;
        let opts = LoadOptions {
            placeholders: true,
            ..Default::default()
        };
        load_vault(dir, &mut writer, &opts).await.unwrap();
        writer.commit_batch().await.unwrap();

        let state = read_vault_state(&writer, "Note").await.unwrap();
        // Only the real note A is in the state; the Missing stub is excluded
        // (no path), and A carries a content hash read via the projection.
        assert_eq!(state.len(), 1, "stub excluded from state");
        assert!(!state.contains_key("missing"));
        assert!(
            matches!(state.get("a"), Some(Some(_))),
            "A carries a content hash"
        );
    }

    /// Every node of `label`, as (id, sorted property debug pairs), sorted.
    /// Captures the full live property set so a comparison is byte-exact.
    async fn canon_nodes(
        writer: &WriterSession,
        label: &str,
    ) -> Vec<(String, Vec<(String, String)>)> {
        let snap = writer.snapshot();
        let mut out: Vec<(String, Vec<(String, String)>)> = snap
            .scan_label(label)
            .await
            .unwrap()
            .iter()
            .map(|n| {
                let mut props: Vec<(String, String)> = n
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), format!("{v:?}")))
                    .collect();
                props.sort();
                (n.id.to_string(), props)
            })
            .collect();
        out.sort();
        out
    }

    /// Every live edge of `edge_type` as sorted (src, dst) id pairs.
    async fn canon_edges(writer: &WriterSession, edge_type: &str) -> Vec<(String, String)> {
        let snap = writer.snapshot();
        let mut out: Vec<(String, String)> = snap
            .scan_edge_type(edge_type)
            .await
            .unwrap()
            .iter()
            .map(|e| (e.src.to_string(), e.dst.to_string()))
            .collect();
        out.sort();
        out
    }

    /// The correctness contract for incremental sync: after a sync, the graph
    /// is byte-identical to a fresh prune-load of the same disk state.
    async fn assert_sync_matches_fresh_load(placeholders: bool) {
        let vault = TempDir::new().unwrap();
        let dir = vault.path();
        // v1.
        write(
            dir,
            "A.md",
            "---\ntags: [proj]\n---\nlinks [[B]] embeds ![[C]] and [[Missing]] #area/db\n",
        );
        write(dir, "B.md", "beta #shared\n");
        write(dir, "C.md", "gamma #shared\n");
        write(dir, "D.md", "delta, to be deleted\n");
        write(dir, "E.md", "echo, unchanged #solo\n");

        let opts = LoadOptions {
            placeholders,
            ..Default::default()
        };

        let mut synced = open(&format!("oracle-sync-{placeholders}")).await;
        load_vault(dir, &mut synced, &opts).await.unwrap();
        synced.commit_batch().await.unwrap();

        // Mutate disk to v2: A modified (new tag, new link/embed targets, drops
        // Missing), C modified (tag change), D deleted, F added; B and E
        // untouched.
        write(
            dir,
            "A.md",
            "---\ntags: [proj, more]\n---\nlinks [[E]] embeds ![[B]] #area/web\n",
        );
        write(dir, "C.md", "gamma #renamed\n");
        std::fs::remove_file(dir.join("D.md")).unwrap();
        write(dir, "F.md", "foxtrot [[A]] #shared\n");

        let out = sync_vault(dir, &mut synced, &opts).await.unwrap();
        synced.commit_batch().await.unwrap();

        assert_eq!(out.notes_added, 1, "F added");
        assert_eq!(out.notes_modified, 2, "A and C modified");
        assert_eq!(out.notes_deleted, 1, "D deleted");
        assert_eq!(out.notes_unchanged, 2, "B and E unchanged");
        assert_eq!(
            out.load.notes_loaded, 3,
            "only A, C and F bodies are (re)written; B and E are skipped"
        );

        // Fresh prune-load of v2 into a separate writer.
        let mut fresh = open(&format!("oracle-fresh-{placeholders}")).await;
        let fresh_opts = LoadOptions {
            prune: true,
            placeholders,
            ..Default::default()
        };
        load_vault(dir, &mut fresh, &fresh_opts).await.unwrap();
        fresh.commit_batch().await.unwrap();

        for label in ["Note", "Tag"] {
            assert_eq!(
                canon_nodes(&synced, label).await,
                canon_nodes(&fresh, label).await,
                "{label} nodes diverge (placeholders={placeholders})"
            );
        }
        for edge_type in ["LINKS_TO", "EMBEDS", "TAGGED", "SUBTAG_OF"] {
            assert_eq!(
                canon_edges(&synced, edge_type).await,
                canon_edges(&fresh, edge_type).await,
                "{edge_type} edges diverge (placeholders={placeholders})"
            );
        }
    }

    #[tokio::test]
    async fn sync_matches_fresh_prune_load_placeholders_off() {
        assert_sync_matches_fresh_load(false).await;
    }

    #[tokio::test]
    async fn sync_matches_fresh_prune_load_placeholders_on() {
        assert_sync_matches_fresh_load(true).await;
    }
}
