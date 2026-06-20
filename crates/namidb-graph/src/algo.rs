//! Graph algorithms over an in-memory adjacency representation.
//!
//! The storage layer serves CSR-backed adjacency via
//! [`namidb_storage::Snapshot::out_edges`]; this module runs analytical
//! kernels over a [`Graph`] built from those edges. Two kernels ship first:
//!
//! - **WCC** (Weakly Connected Components): union-find over an undirected
//!   view of the edges. Cheapest, most useful — partitions the graph into
//!   reachability clusters.
//! - **PageRank**: classic power iteration over a directed edge-weighted
//!   graph with a damping factor (default 0.85), a configurable iteration
//!   cap, and an L1-convergence stop.
//!
//! Both are exact, single-pass-over-edges kernels — no sampling, no
//! approximation. They are intentionally simple and allocation-light: a
//! RAG/agent workload on a markdown vault is well inside the size where the
//! exact algorithm is the right one (the HNSW-on-CSR ANN work is the home
//! for scale; these kernels are the home for correctness).

use std::collections::HashMap;

use namidb_core::NodeId;

/// How often (in loop iterations) a cancellable kernel polls its cancel
/// callback. Polling every node/iteration would add measurable overhead to a
/// tight CPU loop; every 4096 keeps cancellation latency low while the poll
/// cost stays in the noise.
const CANCEL_CHECK_STRIDE: usize = 4096;

/// Returned by the `*_cancellable` kernels when their cancel callback fired
/// (e.g. the query deadline was exceeded mid-computation). The caller maps it
/// to its own timeout error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("graph algorithm cancelled (deadline exceeded)")
    }
}

impl std::error::Error for Cancelled {}

/// A directed multigraph keyed by [`NodeId`], built from edge lists.
///
/// Self-loops and duplicate edges are tolerated (they do not change WCC; in
/// PageRank duplicate out-edges split the out-mass, matching the convention
/// that each emitted edge is one unit of probability). Isolated nodes (no
/// edges) are kept so they appear as their own singleton component / land on
/// the PageRank base rank.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    /// Out-adjacency: `node -> [(neighbor, weight)]`. Weight defaults to 1.0.
    out: HashMap<NodeId, Vec<(NodeId, f64)>>,
    /// Every node known to the graph (including isolates), in insertion order.
    nodes: Vec<NodeId>,
    /// Set view of `nodes` for O(1) membership + dedup on insertion.
    seen: HashMap<NodeId, ()>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a node as present even if it has no edges (an isolate).
    /// Idempotent: re-inserting a known node is a no-op.
    pub fn add_node(&mut self, id: NodeId) {
        if self.seen.insert(id, ()).is_none() {
            self.nodes.push(id);
            self.out.entry(id).or_default();
        }
    }

    /// Add a directed edge `src -> dst` with weight `w` (default 1.0 when
    /// omitted). Both endpoints are registered as nodes.
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId, w: Option<f64>) {
        let weight = w.unwrap_or(1.0);
        self.add_node(src);
        self.add_node(dst);
        self.out.entry(src).or_default().push((dst, weight));
    }

    /// Total node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total edge count (sum of out-degrees).
    pub fn edge_count(&self) -> usize {
        self.out.values().map(|v| v.len()).sum()
    }

    /// Iterate nodes in insertion order.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// Out-edges of `src`, if any.
    pub fn out_edges(&self, src: NodeId) -> Option<&[(NodeId, f64)]> {
        self.out.get(&src).map(Vec::as_slice)
    }
}

/// Result of a weakly-connected-components computation: each node mapped to
/// a component id in `[0, n)` where `n` is the number of distinct components.
/// Two nodes are in the same component iff they share an id.
#[derive(Debug, Clone)]
pub struct Components {
    /// `node -> component id`.
    pub assignment: HashMap<NodeId, usize>,
    /// Number of distinct components.
    pub count: usize,
}

/// Compute Weakly Connected Components via union-find over an *undirected*
/// view of the edges (direction ignored). `O(n α(n) + m)`.
///
/// Returns the component assignment and the number of distinct components.
/// Isolated nodes each form their own singleton component.
pub fn weakly_connected_components(graph: &Graph) -> Components {
    // Infallible: a never-cancel callback can't return Err.
    weakly_connected_components_cancellable(graph, &|| false).expect("never cancels")
}

/// [`weakly_connected_components`] that polls `cancel` periodically and returns
/// [`Cancelled`] if it fires — so a long-running `CALL algo.wcc()` honours the
/// query deadline mid-computation, not just at the operator boundary.
pub fn weakly_connected_components_cancellable(
    graph: &Graph,
    cancel: &dyn Fn() -> bool,
) -> Result<Components, Cancelled> {
    // Map each node id to a dense index for the union-find arrays.
    let mut index: HashMap<NodeId, usize> = HashMap::with_capacity(graph.node_count());
    for (i, &n) in graph.nodes().iter().enumerate() {
        index.insert(n, i);
    }
    let n = index.len();

    let mut uf = UnionFind::new(n);
    let mut since_check = 0usize;
    // Every undirected edge merges its endpoints' sets.
    for (&src, nbrs) in &graph.out {
        let Some(&si) = index.get(&src) else {
            continue;
        };
        for &(dst, _) in nbrs {
            if let Some(&di) = index.get(&dst) {
                uf.union(si, di);
            }
            since_check += 1;
            if since_check >= CANCEL_CHECK_STRIDE {
                since_check = 0;
                if cancel() {
                    return Err(Cancelled);
                }
            }
        }
    }

    // Assign dense component ids by canonical root.
    let mut root_to_comp: HashMap<usize, usize> = HashMap::new();
    let mut assignment = HashMap::with_capacity(n);
    for (&node, &i) in &index {
        let root = uf.find(i);
        let comp = match root_to_comp.get(&root) {
            Some(&c) => c,
            None => {
                let c = root_to_comp.len();
                root_to_comp.insert(root, c);
                c
            }
        };
        assignment.insert(node, comp);
    }
    Ok(Components {
        assignment,
        count: root_to_comp.len(),
    })
}

/// Result of a PageRank computation.
#[derive(Debug, Clone)]
pub struct PageRank {
    /// `node -> PageRank score`. Scores sum to ~1.0 (within the L1 tolerance).
    pub scores: HashMap<NodeId, f64>,
    /// Number of iterations actually run (may be less than the cap if it
    /// converged early).
    pub iterations: usize,
    /// Whether the L1 norm dropped below `tolerance` before the iteration cap.
    pub converged: bool,
}

/// PageRank options.
#[derive(Debug, Clone)]
pub struct PageRankOptions {
    /// Damping factor (probability of following a link vs. teleporting).
    /// Default 0.85, the textbook value.
    pub damping: f64,
    /// Maximum iterations regardless of convergence. Default 100.
    pub max_iterations: usize,
    /// L1 convergence tolerance: stop when `Σ|new - old| < tolerance`.
    /// Default 1e-6.
    pub tolerance: f64,
}

impl Default for PageRankOptions {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Compute PageRank via power iteration.
///
/// The standard update with damping `d`:
/// ```text
/// PR(v) = (1 - d)/N + d * Σ_{u -> v} PR(u) / outdeg(u)
/// ```
/// **Dangling nodes** (no out-edges) redistribute their mass uniformly each
/// iteration so total probability is conserved — the correction that keeps
/// the scores summing to 1 on real graphs (which always have sinks).
///
/// Returns the per-node scores, the iteration count, and whether it converged.
pub fn pagerank(graph: &Graph, opts: &PageRankOptions) -> PageRank {
    // Infallible: a never-cancel callback can't return Err.
    pagerank_cancellable(graph, opts, &|| false).expect("never cancels")
}

/// [`pagerank`] that polls `cancel` once per power-iteration and returns
/// [`Cancelled`] if it fires — so a long-running `CALL algo.pagerank()` honours
/// the query deadline mid-computation (each iteration is O(V+E), so per-
/// iteration polling bounds cancellation latency to one iteration).
pub fn pagerank_cancellable(
    graph: &Graph,
    opts: &PageRankOptions,
    cancel: &dyn Fn() -> bool,
) -> Result<PageRank, Cancelled> {
    let n = graph.node_count();
    let mut scores: HashMap<NodeId, f64> = HashMap::with_capacity(n);
    if n == 0 {
        return Ok(PageRank {
            scores,
            iterations: 0,
            converged: true,
        });
    }
    // Uniform initial distribution.
    let init = 1.0 / n as f64;
    for &node in graph.nodes() {
        scores.insert(node, init);
    }

    let d = opts.damping;
    let teleport = (1.0 - d) / n as f64;

    let mut iterations = 0;
    let mut converged = false;
    for _ in 0..opts.max_iterations {
        // Each iteration is O(V+E); poll the deadline before doing the work.
        if cancel() {
            return Err(Cancelled);
        }
        iterations += 1;

        // Collect dangling mass and redistribute uniformly so probability is
        // conserved. A node is "dangling" if it has NO usable out-mass path:
        // no out-edges, OR a non-positive out-edge weight sum (a degenerate
        // case the push loop below skips). Counting the latter as dangling is
        // load-bearing — without it that node's rank would simply vanish each
        // iteration and the scores would stop summing to 1.
        let is_dangling = |n: &NodeId| match graph.out_edges(*n) {
            None | Some([]) => true,
            Some(e) => e.iter().map(|&(_, w)| w).sum::<f64>() <= 0.0,
        };
        let dangling_mass: f64 = graph
            .nodes()
            .iter()
            .filter(|n| is_dangling(n))
            .map(|n| scores[n])
            .sum();
        let dangling_share = d * dangling_mass / n as f64;

        let mut new_scores: HashMap<NodeId, f64> = HashMap::with_capacity(n);
        for &node in graph.nodes() {
            new_scores.insert(node, teleport + dangling_share);
        }
        // Push each node's rank along its out-edges. Total out-mass must be
        // exactly `d * PR(src)` to conserve probability, so the per-edge share
        // is `d * PR(src) * (w / Σw)` — normalized by the WEIGHT SUM, not the
        // edge count. (Dividing by count and multiplying by w leaks mass when
        // weights are not all 1.0.) A non-positive weight sum is degenerate and
        // was already counted as dangling above (its mass is redistributed), so
        // skip it here rather than divide by zero / produce NaN.
        for (&src, nbrs) in &graph.out {
            if nbrs.is_empty() {
                continue;
            }
            let weight_sum: f64 = nbrs.iter().map(|&(_, w)| w).sum();
            if weight_sum <= 0.0 {
                continue;
            }
            let base = d * scores[&src] / weight_sum;
            for &(dst, w) in nbrs {
                *new_scores.get_mut(&dst).unwrap() += base * w;
            }
        }

        // L1 convergence check.
        let l1: f64 = graph
            .nodes()
            .iter()
            .map(|n| (new_scores[n] - scores[n]).abs())
            .sum();
        scores = new_scores;
        if l1 < opts.tolerance {
            converged = true;
            break;
        }
    }

    Ok(PageRank {
        scores,
        iterations,
        converged,
    })
}

/// Iterative union-find (disjoint-set) with path halving and union by rank.
/// Indices are dense `[0, n)`.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        // Path halving: point each node at its grandparent on the way up.
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        // Attach the shorter tree under the taller; equal ranks bump the rank.
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(b: [u8; 16]) -> NodeId {
        NodeId::from_uuid(uuid::Uuid::from_bytes(b))
    }

    #[test]
    fn wcc_single_component_chain() {
        // a -> b -> c -> d  (undirected: all one component).
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(b, c, None);
        g.add_edge(c, d, None);

        let comps = weakly_connected_components(&g);
        assert_eq!(comps.count, 1);
        assert_eq!(comps.assignment[&a], comps.assignment[&d]);
    }

    #[test]
    fn cancellable_kernels_honour_the_cancel_callback() {
        // Build a graph with enough edges to cross the cancel-check stride.
        let mut g = Graph::new();
        let hub = nid([0; 16]);
        for i in 1u128..=(CANCEL_CHECK_STRIDE as u128 + 50) {
            g.add_edge(hub, nid(i.to_le_bytes()), None);
        }
        // A cancel callback that always fires must stop both kernels.
        let always = || true;
        assert_eq!(
            weakly_connected_components_cancellable(&g, &always).err(),
            Some(Cancelled)
        );
        assert_eq!(
            pagerank_cancellable(&g, &PageRankOptions::default(), &always).err(),
            Some(Cancelled)
        );
        // A never-cancel callback returns the same result as the public fn.
        let never = || false;
        assert!(weakly_connected_components_cancellable(&g, &never).is_ok());
        assert!(pagerank_cancellable(&g, &PageRankOptions::default(), &never).is_ok());
    }

    #[test]
    fn wcc_two_components_with_isolate() {
        // {a-b} and {c} (isolate) and {d-e}.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let e = nid([5; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_node(c); // isolate
        g.add_edge(d, e, None);

        let comps = weakly_connected_components(&g);
        assert_eq!(comps.count, 3);
        assert_eq!(comps.assignment[&a], comps.assignment[&b]);
        assert_ne!(comps.assignment[&a], comps.assignment[&c]);
        assert_ne!(comps.assignment[&a], comps.assignment[&d]);
        assert_eq!(comps.assignment[&d], comps.assignment[&e]);
    }

    #[test]
    fn wcc_direction_ignored() {
        // a -> b and c -> b: undirected, all reachable → one component.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(c, b, None);
        let comps = weakly_connected_components(&g);
        assert_eq!(comps.count, 1);
    }

    #[test]
    fn wcc_empty_graph() {
        let g = Graph::new();
        let comps = weakly_connected_components(&g);
        assert_eq!(comps.count, 0);
        assert!(comps.assignment.is_empty());
    }

    #[test]
    fn pagerank_sums_to_one() {
        // A small graph with a dangling node to exercise the sink fix.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(a, c, None);
        g.add_edge(b, c, None);
        // c is a dangling node (no out-edges).

        let pr = pagerank(&g, &PageRankOptions::default());
        let total: f64 = pr.scores.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "PageRank scores should sum to ~1.0, got {total}"
        );
        assert!(pr.converged, "should converge within the iteration cap");
    }

    #[test]
    fn pagerank_conserves_mass_with_nonpositive_weight_sum() {
        // A node whose only out-edge has a non-positive weight has no usable
        // out-mass path; it must be treated as dangling (its rank redistributed)
        // or probability leaks and the scores stop summing to 1.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, Some(1.0));
        // `a` also points to c with a negative weight; b -> a with weight 0.
        g.add_edge(c, a, Some(-2.0)); // c's weight sum is negative → degenerate
        g.add_edge(b, a, Some(0.0)); // b's weight sum is zero → degenerate
        g.add_node(c);

        let pr = pagerank(&g, &PageRankOptions::default());
        let total: f64 = pr.scores.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "PageRank must conserve mass even with non-positive weight sums, got {total}"
        );
        // No NaN/negative leaked in.
        assert!(pr.scores.values().all(|s| s.is_finite() && *s >= 0.0));
    }

    #[test]
    fn pagerank_hub_scores_higher() {
        // a is pointed to by b and c → a should outrank the others.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(b, a, None);
        g.add_edge(c, a, None);
        g.add_edge(a, d, None);

        let pr = pagerank(&g, &PageRankOptions::default());
        assert!(
            pr.scores[&a] > pr.scores[&b] && pr.scores[&a] > pr.scores[&c],
            "the node with two in-links should rank highest"
        );
    }

    #[test]
    fn pagerank_empty_graph() {
        let g = Graph::new();
        let pr = pagerank(&g, &PageRankOptions::default());
        assert!(pr.scores.is_empty());
        assert!(pr.converged);
    }

    #[test]
    fn pagerank_all_isolates_uniform() {
        // No edges → uniform distribution (each node a teleport target).
        let mut g = Graph::new();
        g.add_node(nid([1; 16]));
        g.add_node(nid([2; 16]));
        g.add_node(nid([3; 16]));
        let pr = pagerank(&g, &PageRankOptions::default());
        let vals: Vec<f64> = pr.scores.values().cloned().collect();
        let total: f64 = vals.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "should sum to 1.0, got {total}");
        // All equal.
        assert!((vals[0] - vals[1]).abs() < 1e-9);
        assert!((vals[1] - vals[2]).abs() < 1e-9);
    }

    #[test]
    fn pagerank_weighted_edges_conserve_mass() {
        // Unequal weights: total out-mass must still be `d * PR(src)`, so the
        // scores sum to ~1.0. The pre-fix code normalized by edge COUNT and
        // leaked mass whenever weights were not all 1.0 (sum diverged from 1).
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, Some(3.0));
        g.add_edge(a, c, Some(1.0));
        g.add_edge(b, c, Some(2.0));
        let pr = pagerank(&g, &PageRankOptions::default());
        let total: f64 = pr.scores.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "weighted PageRank must conserve mass (sum to 1.0), got {total}"
        );
        // All scores finite.
        for v in pr.scores.values() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn pagerank_zero_weight_does_not_nan() {
        // Degenerate: a zero-weight edge must not produce NaN/inf.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, Some(0.0));
        let pr = pagerank(&g, &PageRankOptions::default());
        for v in pr.scores.values() {
            assert!(v.is_finite(), "score must be finite, got {v}");
        }
    }
}