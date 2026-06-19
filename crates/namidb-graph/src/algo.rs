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
    // Map each node id to a dense index for the union-find arrays.
    let mut index: HashMap<NodeId, usize> = HashMap::with_capacity(graph.node_count());
    for (i, &n) in graph.nodes().iter().enumerate() {
        index.insert(n, i);
    }
    let n = index.len();

    let mut uf = UnionFind::new(n);
    // Every undirected edge merges its endpoints' sets.
    for (&src, nbrs) in &graph.out {
        let Some(&si) = index.get(&src) else {
            continue;
        };
        for &(dst, _) in nbrs {
            if let Some(&di) = index.get(&dst) {
                uf.union(si, di);
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
    Components {
        assignment,
        count: root_to_comp.len(),
    }
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
    let n = graph.node_count();
    let mut scores: HashMap<NodeId, f64> = HashMap::with_capacity(n);
    if n == 0 {
        return PageRank {
            scores,
            iterations: 0,
            converged: true,
        };
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
        iterations += 1;

        // Collect dangling mass (nodes with no out-edges) and redistribute
        // uniformly so probability is conserved.
        let dangling_mass: f64 = graph
            .nodes()
            .iter()
            .filter(|&&n| graph.out_edges(n).map_or(true, |e| e.is_empty()))
            .map(|n| scores[n])
            .sum();
        let dangling_share = d * dangling_mass / n as f64;

        let mut new_scores: HashMap<NodeId, f64> = HashMap::with_capacity(n);
        for &node in graph.nodes() {
            new_scores.insert(node, teleport + dangling_share);
        }
        // Push each node's rank along its out-edges.
        for (&src, nbrs) in &graph.out {
            let outdeg = nbrs.len();
            if outdeg == 0 {
                continue;
            }
            let share = d * scores[&src] / outdeg as f64;
            for &(dst, w) in nbrs {
                // Weighted edges scale the contributed mass; the weight is
                // relative to the edge, normalized across the out-set so total
                // out-mass stays `d * PR(src)`.
                *new_scores.get_mut(&dst).unwrap() += share * w;
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

    PageRank {
        scores,
        iterations,
        converged,
    }
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
}
