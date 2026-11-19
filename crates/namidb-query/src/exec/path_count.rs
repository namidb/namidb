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

/// Heap bytes of a weight map: a `(NodeId, u64)` slot plus a control byte
/// per bucket at the 7/8 load factor, rounded up.
fn weights_bytes(w: &Weights) -> usize {
    w.capacity().saturating_mul(32)
}

const PAIR_BYTES: usize = std::mem::size_of::<(NodeId, NodeId)>();

/// The fold's working state (oriented edge lists beyond what storage
/// already charged, weight maps, id batches), charged to the query's memory
/// budget. The check counts the storage-retained edge pairs as well, so a
/// grouped count over a graph larger than the budget fails with a memory
/// error instead of growing past it. Inactive (free) without a budget.
struct StateMeter {
    active: bool,
    bytes: usize,
    checked: usize,
}

impl StateMeter {
    fn new() -> Self {
        Self {
            active: crate::exec::memory::memory_scoped(),
            bytes: 0,
            checked: 0,
        }
    }

    fn add(&mut self, bytes: usize) -> Result<(), ExecError> {
        if !self.active {
            return Ok(());
        }
        self.bytes = self.bytes.saturating_add(bytes);
        if self.bytes - self.checked >= crate::exec::memory::METER_FLUSH_BYTES {
            self.checked = self.bytes;
            crate::exec::memory::check_transient_over_storage(self.bytes)?;
        }
        Ok(())
    }

    fn release(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_sub(bytes);
        self.checked = self.checked.min(self.bytes);
    }

    /// Account `w` growing from `before` bytes to its current size.
    fn grew(&mut self, before: usize, w: &Weights) -> Result<usize, ExecError> {
        let now = weights_bytes(w);
        if now > before {
            self.add(now - before)?;
        }
        Ok(now)
    }
}

fn overflow() -> ExecError {
    ExecError::Runtime("path count overflowed a 64-bit integer".into())
}

/// Live pairs of one hop, oriented `(near, far)` for travel from
/// `nodes[i]` (near) to `nodes[i + 1]` (far). An undirected hop yields each
/// edge in both orientations except a self-loop, which the undirected
/// `Expand` also yields once.
///
/// Storage charges the pairs it returns while it reads them; the meter is
/// charged for the second orientation of an undirected hop as it is built.
/// The caller then moves the whole edge lists' charge onto the meter.
async fn hop_pairs(
    snapshot: &Snapshot<'_>,
    hop: &PathCountHop,
    meter: &mut StateMeter,
) -> Result<Vec<(NodeId, NodeId)>, ExecError> {
    let types = resolve_edge_types(snapshot, hop.edge_type.as_deref());
    let mut out: Vec<(NodeId, NodeId)> = Vec::new();
    let mut probe = crate::exec::limits::DeadlineProbe::new();
    for et in &types {
        let pairs = snapshot.live_edge_pairs(et).await?;
        match hop.direction {
            RelationshipDirection::Right if out.is_empty() => out = pairs,
            RelationshipDirection::Right => out.extend(pairs),
            RelationshipDirection::Left if out.is_empty() => {
                // Same layout, so the collect reuses the buffer in place.
                out = pairs.into_iter().map(|(s, d)| (d, s)).collect();
            }
            RelationshipDirection::Left => out.extend(pairs.into_iter().map(|(s, d)| (d, s))),
            RelationshipDirection::Both => {
                let before = out.capacity();
                out.reserve(pairs.len() * 2);
                meter.add((out.capacity() - before).saturating_mul(PAIR_BYTES))?;
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
    meter: &mut StateMeter,
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
        meter.add(weights_bytes(&out))?;
        for view in views {
            probe.tick()?;
            if let Some(w) = raw.get(&view.id) {
                if node.labels.iter().all(|l| view.labels.contains(l)) {
                    out.insert(view.id, *w);
                }
            }
        }
        // The scanned views are gone; only `out` (charged above) remains.
        crate::exec::memory::release_storage_rows();
        return Ok(out);
    }
    let ids: Vec<NodeId> = raw.keys().copied().collect();
    let ids_bytes = ids.capacity().saturating_mul(std::mem::size_of::<NodeId>());
    meter.add(ids_bytes)?;
    let proven = prepare_expand_targets(snapshot, &node.labels, &ids, true, false, false).await?;
    drop(ids);
    meter.release(ids_bytes);
    // Whatever the proof read is dropped; its membership map is charged
    // below.
    crate::exec::memory::release_storage_rows();
    match proven {
        ExpandTargets::Membership(live) => {
            // The membership map is as large as the batch; it lives until
            // the filter below has run.
            let live_bytes = live.capacity().saturating_mul(32);
            meter.add(live_bytes)?;
            let mut out = weights_with_capacity(0);
            let mut held = 0usize;
            for (id, w) in raw {
                if live.get(&id).copied().unwrap_or(false) {
                    out.insert(id, w);
                    held = meter.grew(held, &out)?;
                }
            }
            meter.release(live_bytes);
            Ok(out)
        }
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
    meter: &mut StateMeter,
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
    let mut held = 0usize;
    for view in views {
        if node.labels.iter().all(|l| view.labels.contains(l)) {
            *out.entry(view.id).or_insert(0) += 1;
            held = meter.grew(held, &out)?;
        }
    }
    crate::exec::memory::release_storage_rows();
    Ok(out)
}

/// Apply node `i`'s filter to a weight map (drop failing ids). Node 0 with
/// predicates multiplies by its scan multiplicity instead.
///
/// The input map's bytes are released once it is consumed; the result's
/// stay charged (the caller releases them when it drops the map).
async fn filter_weights(
    snapshot: &Snapshot<'_>,
    node: &PathCountNode,
    is_head: bool,
    raw: Weights,
    meter: &mut StateMeter,
) -> Result<Weights, ExecError> {
    let raw_bytes = weights_bytes(&raw);
    let out = if is_head && !node.predicates.is_empty() {
        let mult = scan_multiplicity(snapshot, node, meter).await?;
        let mult_bytes = weights_bytes(&mult);
        let mut out = weights_with_capacity(raw.len().min(mult.len()));
        meter.add(weights_bytes(&out))?;
        for (id, w) in raw {
            if let Some(m) = mult.get(&id) {
                out.insert(id, w.checked_mul(*m).ok_or_else(overflow)?);
            }
        }
        meter.release(mult_bytes);
        out
    } else {
        keep_members(snapshot, node, raw, meter).await?
    };
    meter.release(raw_bytes);
    Ok(out)
}

/// Fold one hop: `out(near) += w(far)` over `(near, far)` pairs.
///
/// The result's bytes are charged to `meter` as it grows (a rehash is the
/// only growth), so an oversized fold stops mid-flight.
fn fold(
    pairs: &[(NodeId, NodeId)],
    from_far: &Weights,
    toward_near: bool,
    meter: &mut StateMeter,
) -> Result<Weights, ExecError> {
    let mut out = Weights::default();
    let mut held = 0usize;
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
            if out.capacity() * 32 != held {
                held = meter.grew(held, &out)?;
            }
        }
    }
    Ok(out)
}

/// The initial (unfiltered) weight map of the endpoint node on one side:
/// every id that appears on that side of the adjacent hop, weight 1.
fn endpoint_ones(
    pairs: &[(NodeId, NodeId)],
    far_side: bool,
    meter: &mut StateMeter,
) -> Result<Weights, ExecError> {
    let mut out = Weights::default();
    let mut held = 0usize;
    for &(near, far) in pairs {
        out.insert(if far_side { far } else { near }, 1);
        if out.capacity() * 32 != held {
            held = meter.grew(held, &out)?;
        }
    }
    Ok(out)
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
    let mut meter = StateMeter::new();
    let mut pairs: Vec<Vec<(NodeId, NodeId)>> = Vec::with_capacity(k);
    for hop in hops {
        pairs.push(hop_pairs(snapshot, hop, &mut meter).await?);
    }
    // Storage charged the edge lists while it read them, and keeps that
    // charge until the operator settles. Move it onto the meter (which
    // releases as the fold drops state) so later reads (endpoint proofs)
    // are not charged on top of lists they replaced.
    meter.release(meter.bytes);
    crate::exec::memory::release_storage_rows();
    let held: usize = pairs
        .iter()
        .map(|p| p.capacity().saturating_mul(PAIR_BYTES))
        .sum();
    meter.add(held)?;
    let g = group.unwrap_or(0);

    // Right fold: tail (node k) back to node g. `right` ends as the
    // UNFILTERED sum at node g (g's filter is applied at emission).
    let right: Option<Weights> = if g < k {
        let mut w = endpoint_ones(&pairs[k - 1], true, &mut meter)?;
        w = filter_weights(snapshot, &nodes[k], k == 0, w, &mut meter).await?;
        for i in (g..k).rev() {
            let prev = weights_bytes(&w);
            w = fold(&pairs[i], &w, true, &mut meter)?;
            meter.release(prev);
            if i > g {
                w = filter_weights(snapshot, &nodes[i], i == 0, w, &mut meter).await?;
            }
        }
        Some(w)
    } else {
        None
    };
    // Left fold: head (node 0) forward to node g, same convention.
    let left: Option<Weights> = if g > 0 {
        let mut w = endpoint_ones(&pairs[0], false, &mut meter)?;
        w = filter_weights(snapshot, &nodes[0], true, w, &mut meter).await?;
        for (i, hop_pairs) in pairs.iter().enumerate().take(g) {
            let prev = weights_bytes(&w);
            w = fold(hop_pairs, &w, false, &mut meter)?;
            meter.release(prev);
            if i + 1 < g {
                w = filter_weights(snapshot, &nodes[i + 1], false, w, &mut meter).await?;
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
            meter.add(weights_bytes(&out))?;
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
        let filtered = filter_weights(snapshot, group_node, true, weights, &mut meter).await?;
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
