//! Executor for [`LogicalPlan::PathCount`]: grouped path counts over a
//! linear chain, folded from whole-edge-type topology.
//!
//! The naive plan for `MATCH (a:A)-[:R]->(b)-[:S]->(c) RETURN a.k, count(*)`
//! enumerates every path: one row per `(a, b, c)`, each a map of bound
//! values, then groups them. The number of paths through a node is a product
//! of two independent sums, though — the paths reaching it from the head of
//! the chain and the paths leaving it toward the tail — and each sum folds
//! hop by hop over the hop's edge list:
//!
//! ```text
//! R_k(v)   = 1                         for v that passes node k's filter
//! R_i(u)   = Σ_{(u -> v) in hop i} R_{i+1}(v)   then node i's filter
//! L_0(u)   = multiplicity of u in v0's scan
//! L_{i+1}(v) = Σ_{(u -> v) in hop i} L_i(u)     then node i+1's filter
//! count(g) = L_g(g) * R_g(g)
//! ```
//!
//! Each hop's live `(src, dst)` pairs are read once, sequentially
//! ([`Snapshot::live_edge_pairs`]); no node is probed per path and no row is
//! built per path. Endpoint filters are proven with the same batched label /
//! existence checks the replaced `Expand`s use
//! ([`prepare_expand_targets`]), so the result is exactly the naive one:
//! same groups, same counts.

use std::collections::HashMap;

use namidb_core::id::NodeId;
use namidb_storage::Snapshot;

use crate::exec::row::Row;
use crate::exec::value::{NodeValue, RuntimeValue};
use crate::exec::walker::{
    prepare_expand_targets, resolve_edge_types, scan_node_views, ExecError, ExpandTargets,
};
use crate::parser::RelationshipDirection;
use crate::plan::logical::{PathCountHop, PathCountNode};

/// Node ids are server-generated UUIDs, so a multiply-xor hash over their
/// 8-byte halves spreads them well; a path count hashes every endpoint of
/// every edge it folds, and SipHash was a measurable share of that.
#[derive(Default, Clone, Copy)]
struct IdHasher(u64);

impl std::hash::Hasher for IdHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(word))
                .wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }

    fn write_usize(&mut self, n: usize) {
        self.write(&(n as u64).to_le_bytes());
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type IdBuild = std::hash::BuildHasherDefault<IdHasher>;
type Weights = HashMap<NodeId, u64, IdBuild>;

fn weights_with_capacity(n: usize) -> Weights {
    Weights::with_capacity_and_hasher(n, IdBuild::default())
}

fn overflow() -> ExecError {
    ExecError::Runtime("path count overflowed a 64-bit integer".into())
}

/// Live pairs of one hop, oriented `(near, far)` for travel from
/// `nodes[i]` (near) to `nodes[i + 1]` (far). An undirected hop yields each
/// edge in both orientations except a self-loop, which the undirected
/// `Expand` also yields once.
async fn hop_pairs(
    snapshot: &Snapshot<'_>,
    hop: &PathCountHop,
) -> Result<Vec<(NodeId, NodeId)>, ExecError> {
    let types = resolve_edge_types(snapshot, hop.edge_type.as_deref());
    let mut out = Vec::new();
    let mut probe = crate::exec::limits::DeadlineProbe::new();
    for et in &types {
        let pairs = snapshot.live_edge_pairs(et).await?;
        match hop.direction {
            RelationshipDirection::Right => out.extend(pairs),
            RelationshipDirection::Left => out.extend(pairs.into_iter().map(|(s, d)| (d, s))),
            RelationshipDirection::Both => {
                out.reserve(pairs.len() * 2);
                for (s, d) in pairs {
                    probe.tick()?;
                    out.push((s, d));
                    if s != d {
                        out.push((d, s));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// A batch at least this large may be proven by scanning the node's label
/// instead of probing each id.
const SCAN_PROOF_MIN_IDS: usize = 4_096;

/// A scan reads every member of the label; a probe reads per id, and costs
/// roughly this many scanned members each (a probe touches a sidecar page or
/// a row group per id; a scan decodes the label's id column once). Measured
/// on a corpus whose node rows carry multi-kilobyte text and embeddings,
/// where a probe is the expensive side.
const SCAN_PROOF_RATIO: u64 = 10;

/// Keep the entries of `raw` whose node passes `node`'s filter: label
/// membership (or plain existence for an unlabelled node), exactly as
/// `Expand` proves its endpoints.
///
/// A large batch scans the node's label (or every node, unlabelled) with an
/// empty projection — the same read a `NodeScan` makes, so the same live
/// set — and keeps the members it meets. A small one probes per id like
/// `Expand` does.
async fn keep_members(
    snapshot: &Snapshot<'_>,
    node: &PathCountNode,
    raw: Weights,
) -> Result<Weights, ExecError> {
    if raw.is_empty() {
        return Ok(raw);
    }
    let scan_label = node.labels.first().map(String::as_str);
    // `NAMIDB_PATH_COUNT_SCAN_PROOF=always|never` pins the strategy (both
    // are exact; the differential tests run each under both).
    let scan = match std::env::var("NAMIDB_PATH_COUNT_SCAN_PROOF").as_deref() {
        Ok("always") => true,
        Ok("never") => false,
        _ => {
            if raw.len() < SCAN_PROOF_MIN_IDS {
                false
            } else {
                // Exact and cached; with overlapping node files the first
                // call reconciles ids without reading properties.
                let population = match snapshot.metadata_node_count(scan_label) {
                    Some(population) => population,
                    None => snapshot.count_nodes(scan_label).await?,
                };
                (raw.len() as u64).saturating_mul(SCAN_PROOF_RATIO) >= population
            }
        }
    };
    if scan {
        let views = scan_node_views(snapshot, scan_label, &[], Some(&[]), None).await?;
        let mut probe = crate::exec::limits::DeadlineProbe::new();
        let mut out = weights_with_capacity(raw.len().min(views.len()));
        for view in views {
            probe.tick()?;
            if let Some(w) = raw.get(&view.id) {
                if node.labels.iter().all(|l| view.labels.contains(l)) {
                    out.insert(view.id, *w);
                }
            }
        }
        return Ok(out);
    }
    let ids: Vec<NodeId> = raw.keys().copied().collect();
    match prepare_expand_targets(snapshot, &node.labels, &ids, true, false, false).await? {
        ExpandTargets::Membership(live) => Ok(raw
            .into_iter()
            .filter(|(id, _)| live.get(id).copied().unwrap_or(false))
            .collect()),
        _ => Err(ExecError::Runtime(
            "path count endpoint proof returned materialised views".into(),
        )),
    }
}

/// `v0`'s scan as a multiplicity map: how many rows the original `NodeScan`
/// emits per node id. Only needed when `v0` carries scan predicates; an
/// unfiltered `v0` is proven like any other node.
async fn scan_multiplicity(
    snapshot: &Snapshot<'_>,
    node: &PathCountNode,
) -> Result<Weights, ExecError> {
    let views = scan_node_views(
        snapshot,
        node.labels.first().map(String::as_str),
        &node.predicates,
        Some(&[]),
        None,
    )
    .await?;
    let mut out = Weights::default();
    for view in views {
        if node.labels.iter().all(|l| view.labels.contains(l)) {
            *out.entry(view.id).or_insert(0) += 1;
        }
    }
    Ok(out)
}

/// Apply node `i`'s filter to a weight map (drop failing ids). Node 0 with
/// predicates multiplies by its scan multiplicity instead.
async fn filter_weights(
    snapshot: &Snapshot<'_>,
    node: &PathCountNode,
    is_head: bool,
    raw: Weights,
) -> Result<Weights, ExecError> {
    if is_head && !node.predicates.is_empty() {
        let mult = scan_multiplicity(snapshot, node).await?;
        let mut out = weights_with_capacity(raw.len().min(mult.len()));
        for (id, w) in raw {
            if let Some(m) = mult.get(&id) {
                out.insert(id, w.checked_mul(*m).ok_or_else(overflow)?);
            }
        }
        return Ok(out);
    }
    keep_members(snapshot, node, raw).await
}

/// Fold one hop: `out(near) += w(far)` over `(near, far)` pairs.
fn fold(
    pairs: &[(NodeId, NodeId)],
    from_far: &Weights,
    toward_near: bool,
) -> Result<Weights, ExecError> {
    let mut out = Weights::default();
    let mut probe = crate::exec::limits::DeadlineProbe::new();
    for &(near, far) in pairs {
        probe.tick()?;
        let (src, dst) = if toward_near {
            (far, near)
        } else {
            (near, far)
        };
        if let Some(w) = from_far.get(&src) {
            let slot = out.entry(dst).or_insert(0);
            *slot = slot.checked_add(*w).ok_or_else(overflow)?;
        }
    }
    Ok(out)
}

/// The initial (unfiltered) weight map of the endpoint node on one side:
/// every id that appears on that side of the adjacent hop, weight 1.
fn endpoint_ones(pairs: &[(NodeId, NodeId)], far_side: bool) -> Weights {
    let mut out = Weights::default();
    for &(near, far) in pairs {
        out.insert(if far_side { far } else { near }, 1);
    }
    out
}

/// Execute a `PathCount` leaf. See the module docs.
pub(crate) async fn execute_path_count(
    snapshot: &Snapshot<'_>,
    nodes: &[PathCountNode],
    hops: &[PathCountHop],
    group: Option<usize>,
    projection: Option<&[String]>,
    output: &str,
) -> Result<Vec<Row>, ExecError> {
    namidb_core::profile_scope!("walker::execute_path_count");
    let k = hops.len();
    if k == 0 || nodes.len() != k + 1 || group.is_some_and(|g| g > k) {
        return Err(ExecError::Runtime("malformed PathCount chain".into()));
    }
    // Every hop's topology, read once. Hop type sets are disjoint (the
    // rewrite requires it for chains), so no type is read twice.
    let mut pairs: Vec<Vec<(NodeId, NodeId)>> = Vec::with_capacity(k);
    for hop in hops {
        pairs.push(hop_pairs(snapshot, hop).await?);
    }
    let g = group.unwrap_or(0);

    // Right fold: tail (node k) back to node g. `right` ends as the
    // UNFILTERED sum at node g (g's filter is applied at emission).
    let right: Option<Weights> = if g < k {
        let mut w = endpoint_ones(&pairs[k - 1], true);
        w = filter_weights(snapshot, &nodes[k], k == 0, w).await?;
        for i in (g..k).rev() {
            w = fold(&pairs[i], &w, true)?;
            if i > g {
                w = filter_weights(snapshot, &nodes[i], i == 0, w).await?;
            }
        }
        Some(w)
    } else {
        None
    };
    // Left fold: head (node 0) forward to node g, same convention.
    let left: Option<Weights> = if g > 0 {
        let mut w = endpoint_ones(&pairs[0], false);
        w = filter_weights(snapshot, &nodes[0], true, w).await?;
        for (i, hop_pairs) in pairs.iter().enumerate().take(g) {
            w = fold(hop_pairs, &w, false)?;
            if i + 1 < g {
                w = filter_weights(snapshot, &nodes[i + 1], false, w).await?;
            }
        }
        Some(w)
    } else {
        None
    };
    let mut weights: Weights = match (left, right) {
        (Some(l), Some(r)) => {
            let (small, big) = if l.len() <= r.len() { (l, r) } else { (r, l) };
            let mut out = weights_with_capacity(small.len());
            for (id, a) in small {
                if let Some(b) = big.get(&id) {
                    out.insert(id, a.checked_mul(*b).ok_or_else(overflow)?);
                }
            }
            out
        }
        (Some(w), None) | (None, Some(w)) => w,
        (None, None) => unreachable!("k >= 1 means at least one side folds"),
    };
    weights.retain(|_, w| *w > 0);
    let group_node = &nodes[g];

    let Some(g) = group else {
        // Global count: node 0's filter, then the total.
        let filtered = filter_weights(snapshot, group_node, true, weights).await?;
        let mut total: u64 = 0;
        for w in filtered.values() {
            total = total.checked_add(*w).ok_or_else(overflow)?;
        }
        if total == 0 {
            return Ok(Vec::new());
        }
        let total = i64::try_from(total).map_err(|_| overflow())?;
        return Ok(vec![Row::new().with(output, RuntimeValue::Integer(total))]);
    };

    let emit = |view: namidb_storage::NodeView, w: u64| -> Result<Row, ExecError> {
        let w = i64::try_from(w).map_err(|_| overflow())?;
        Ok(Row::new()
            .with(
                group_node.alias.clone(),
                RuntimeValue::Node(Box::new(NodeValue::from(view))),
            )
            .with(output, RuntimeValue::Integer(w)))
    };
    let mut out = Vec::new();
    let mut probe = crate::exec::limits::DeadlineProbe::new();
    if g == 0 || !group_node.labels.is_empty() {
        // The original scan of the group node (v0), or a scan of the group
        // node's first label with the projection its keys need. Either way a
        // row per live member, filtered to the members that have paths.
        let scan_predicates = if g == 0 {
            group_node.predicates.as_slice()
        } else {
            &[]
        };
        if weights.is_empty() && g != 0 {
            return Ok(out);
        }
        let views = scan_node_views(
            snapshot,
            group_node.labels.first().map(String::as_str),
            scan_predicates,
            projection,
            None,
        )
        .await?;
        for view in views {
            probe.tick()?;
            let Some(w) = weights.get(&view.id).copied() else {
                continue;
            };
            if !group_node.labels.iter().all(|l| view.labels.contains(l)) {
                continue;
            }
            out.push(emit(view, w)?);
        }
    } else {
        // An unlabelled interior/tail group node: hydrate exactly the nodes
        // that have paths (what the replaced Expand materialised).
        let mut ids: Vec<NodeId> = weights.keys().copied().collect();
        ids.sort_unstable();
        let views = snapshot.batch_lookup_nodes("", &ids).await?;
        for (id, view) in ids.into_iter().zip(views) {
            probe.tick()?;
            if let Some(view) = view {
                out.push(emit(view, weights[&id])?);
            }
        }
    }
    Ok(out)
}
