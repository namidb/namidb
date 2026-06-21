//! Graph algorithms over an in-memory adjacency representation.
//!
//! The storage layer serves CSR-backed adjacency via
//! [`namidb_storage::Snapshot::out_edges`]; this module runs analytical
//! kernels over a [`Graph`] built from those edges. The kernels cover the
//! questions a graph workload actually asks:
//!
//! - **WCC** ([`weakly_connected_components`]) — *what is connected?*
//!   Union-find over an undirected view of the edges. Cheapest, most useful —
//!   partitions the graph into reachability clusters.
//! - **SCC** ([`strongly_connected_components`]) — *what cycles together?*
//!   Tarjan's algorithm (iterative, no recursion limit) over the *directed*
//!   edges — the directed counterpart to WCC.
//! - **PageRank** ([`pagerank`]) — *what is authoritative?* Classic power
//!   iteration over a directed edge-weighted graph with a damping factor
//!   (default 0.85), a configurable iteration cap, and an L1-convergence stop.
//! - **Degree centrality** ([`degrees`]) — *what is a hub?* In/out/total
//!   degree per node, one pass over the edges.
//! - **Triangle count** ([`triangle_count`]) — *how clustered is a node?*
//!   Triangles through each node plus the local clustering coefficient over
//!   the undirected simple-graph view.
//! - **Label propagation** ([`label_propagation`]) — *what are the
//!   communities?* Near-linear community detection that finds sub-structure
//!   WCC cannot (communities inside one connected component).
//! - **Shortest paths** ([`shortest_paths`]) — *how do I get from A to B?*
//!   BFS (unweighted hop count) or Dijkstra (non-negative edge weights) from
//!   a single source, following the directed edges.
//!
//! All are exact kernels — no sampling, no approximation. They are
//! intentionally simple and allocation-light: a RAG/agent workload on a
//! markdown vault is well inside the size where the exact algorithm is the
//! right one (the HNSW-on-CSR ANN work is the home for scale; these kernels
//! are the home for correctness). Every kernel has a `*_cancellable` variant
//! that polls a cancel callback so a runaway `CALL algo.*` on a huge graph
//! honours the query deadline mid-computation, not just at the operator
//! boundary.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

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

    // Assign dense component ids by canonical root. This is a second O(N) pass,
    // so it also polls the deadline (a high-node-count sparse graph spends real
    // time here, not just in the union phase above).
    let mut root_to_comp: HashMap<usize, usize> = HashMap::new();
    let mut assignment = HashMap::with_capacity(n);
    since_check = 0;
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
        since_check += 1;
        if since_check >= CANCEL_CHECK_STRIDE {
            since_check = 0;
            if cancel() {
                return Err(Cancelled);
            }
        }
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

// ===========================================================================
// Degree centrality
// ===========================================================================

/// Per-node degree counts over the directed multigraph. `total = in + out`;
/// self-loops count toward both in and out, and parallel (duplicate) edges
/// count each time — degree is over the edges as stored, not a simple-graph
/// view.
#[derive(Debug, Clone, Default)]
pub struct Degrees {
    /// `node -> number of incoming edges`.
    pub in_degree: HashMap<NodeId, usize>,
    /// `node -> number of outgoing edges`.
    pub out_degree: HashMap<NodeId, usize>,
}

impl Degrees {
    /// Total degree (in + out) of a node; 0 if the node is unknown.
    pub fn total(&self, node: &NodeId) -> usize {
        self.in_degree.get(node).copied().unwrap_or(0)
            + self.out_degree.get(node).copied().unwrap_or(0)
    }
}

/// Compute in/out degree for every node (isolates included, at 0). `O(V + E)`.
pub fn degrees(graph: &Graph) -> Degrees {
    degrees_cancellable(graph, &|| false).expect("never cancels")
}

/// [`degrees`] that polls `cancel` periodically.
pub fn degrees_cancellable(graph: &Graph, cancel: &dyn Fn() -> bool) -> Result<Degrees, Cancelled> {
    let mut in_degree: HashMap<NodeId, usize> = HashMap::with_capacity(graph.node_count());
    let mut out_degree: HashMap<NodeId, usize> = HashMap::with_capacity(graph.node_count());
    // Seed every known node at 0 so isolates report degree 0 rather than absent.
    for &n in graph.nodes() {
        in_degree.insert(n, 0);
        out_degree.insert(n, 0);
    }
    let mut since_check = 0usize;
    for (&src, nbrs) in &graph.out {
        out_degree.insert(src, nbrs.len());
        for &(dst, _) in nbrs {
            *in_degree.entry(dst).or_insert(0) += 1;
            since_check += 1;
            if since_check >= CANCEL_CHECK_STRIDE {
                since_check = 0;
                if cancel() {
                    return Err(Cancelled);
                }
            }
        }
    }
    Ok(Degrees {
        in_degree,
        out_degree,
    })
}

// ===========================================================================
// Strongly Connected Components (Tarjan, iterative)
// ===========================================================================

/// Strongly Connected Components via Tarjan's algorithm — the *directed*
/// counterpart to [`weakly_connected_components`]. Two nodes share a component
/// iff each is reachable from the other following edge direction. `O(V + E)`.
///
/// Implemented iteratively with an explicit DFS stack so a deep or pathological
/// graph cannot overflow the call stack. Reuses [`Components`].
pub fn strongly_connected_components(graph: &Graph) -> Components {
    strongly_connected_components_cancellable(graph, &|| false).expect("never cancels")
}

/// [`strongly_connected_components`] that polls `cancel` periodically.
pub fn strongly_connected_components_cancellable(
    graph: &Graph,
    cancel: &dyn Fn() -> bool,
) -> Result<Components, Cancelled> {
    let index = dense_index(graph);
    let n = graph.node_count();
    let nodes = graph.nodes();

    // Directed adjacency in dense indices.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (&src, nbrs) in &graph.out {
        if let Some(&si) = index.get(&src) {
            for &(dst, _) in nbrs {
                if let Some(&di) = index.get(&dst) {
                    adj[si].push(di);
                }
            }
        }
    }

    const UNVISITED: i64 = -1;
    let mut disc = vec![UNVISITED; n]; // discovery index per node
    let mut low = vec![0i64; n]; // lowest discovery index reachable
    let mut on_stack = vec![false; n];
    let mut comp = vec![0usize; n];
    let mut scc_stack: Vec<usize> = Vec::new();
    let mut next_disc: i64 = 0;
    let mut comp_count = 0usize;
    let mut since_check = 0usize;

    // Explicit DFS. Each frame is (node, cursor) where cursor is the index of
    // the next child to visit. cursor == 0 marks the node's first visit.
    for start in 0..n {
        if disc[start] != UNVISITED {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some(&(v, cursor)) = call.last() {
            if cursor == 0 {
                disc[v] = next_disc;
                low[v] = next_disc;
                next_disc += 1;
                scc_stack.push(v);
                on_stack[v] = true;
            }
            since_check += 1;
            if since_check >= CANCEL_CHECK_STRIDE {
                since_check = 0;
                if cancel() {
                    return Err(Cancelled);
                }
            }
            if cursor < adj[v].len() {
                call.last_mut().unwrap().1 += 1;
                let w = adj[v][cursor];
                if disc[w] == UNVISITED {
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                // All children visited: if v is an SCC root, pop its members.
                if low[v] == disc[v] {
                    let c = comp_count;
                    comp_count += 1;
                    loop {
                        let w = scc_stack.pop().expect("scc stack non-empty at root");
                        on_stack[w] = false;
                        comp[w] = c;
                        if w == v {
                            break;
                        }
                    }
                }
                call.pop();
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[v]);
                }
            }
        }
    }

    let mut assignment = HashMap::with_capacity(n);
    for (i, &node) in nodes.iter().enumerate() {
        assignment.insert(node, comp[i]);
    }
    Ok(Components {
        assignment,
        count: comp_count,
    })
}

// ===========================================================================
// Triangle count + local clustering coefficient
// ===========================================================================

/// Result of a triangle-count computation over the undirected simple-graph view.
#[derive(Debug, Clone, Default)]
pub struct Triangles {
    /// `node -> number of triangles the node participates in`.
    pub per_node: HashMap<NodeId, usize>,
    /// `node -> local clustering coefficient in [0, 1]` (0 when degree < 2).
    pub coefficient: HashMap<NodeId, f64>,
    /// Total number of distinct triangles in the graph.
    pub total: usize,
}

/// Count triangles through each node and the local clustering coefficient, over
/// the undirected simple-graph view (direction ignored, self-loops and parallel
/// edges collapsed). The clustering coefficient of `v` is
/// `2·T(v) / (deg(v)·(deg(v)−1))`, where `T(v)` is the number of edges among
/// `v`'s neighbours.
pub fn triangle_count(graph: &Graph) -> Triangles {
    triangle_count_cancellable(graph, &|| false).expect("never cancels")
}

/// [`triangle_count`] that polls `cancel` periodically.
pub fn triangle_count_cancellable(
    graph: &Graph,
    cancel: &dyn Fn() -> bool,
) -> Result<Triangles, Cancelled> {
    let index = dense_index(graph);
    let n = graph.node_count();
    let nodes = graph.nodes();
    let adj = undirected_adjacency(graph, &index);

    let mut per_node: HashMap<NodeId, usize> = HashMap::with_capacity(n);
    let mut coefficient: HashMap<NodeId, f64> = HashMap::with_capacity(n);
    let mut triple_total = 0usize; // each triangle counted 3× (once per vertex)
    let mut since_check = 0usize;

    for v in 0..n {
        let neigh: Vec<usize> = adj[v].iter().copied().collect();
        let deg = neigh.len();
        let mut t = 0usize;
        for i in 0..neigh.len() {
            for j in (i + 1)..neigh.len() {
                if adj[neigh[i]].contains(&neigh[j]) {
                    t += 1;
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
        per_node.insert(nodes[v], t);
        let coef = if deg < 2 {
            0.0
        } else {
            2.0 * t as f64 / (deg as f64 * (deg as f64 - 1.0))
        };
        coefficient.insert(nodes[v], coef);
        triple_total += t;
    }

    Ok(Triangles {
        per_node,
        coefficient,
        total: triple_total / 3,
    })
}

// ===========================================================================
// Label propagation (community detection)
// ===========================================================================

/// Default iteration cap for [`label_propagation`].
pub const LABEL_PROPAGATION_DEFAULT_ITERS: usize = 10;

/// Community detection via asynchronous label propagation (Raghavan et al.,
/// 2007) over the undirected view. Each sweep visits nodes in a fixed order and
/// updates labels *in place*, so a node sees its neighbours' labels as updated
/// earlier in the same sweep — the asynchronous schedule that avoids the
/// flip-flop oscillation a synchronous update suffers on bipartite structures
/// (e.g. a single connected pair). A node adopts the label most common among
/// its neighbours, keeps its own label when that label is already a maximum,
/// and otherwise breaks ties toward the smallest label id — so the result is
/// fully deterministic (no randomness). Runs until no node changes or
/// `max_iterations` is reached, then relabels to dense component ids in
/// `[0, count)`.
///
/// Like all label propagation, it can merge two cohesive groups joined only by
/// a weak bridge into one community (the well-known "monster community" effect).
/// For a strict connectivity partition use [`weakly_connected_components`] or
/// [`strongly_connected_components`]; this finds the looser community structure.
pub fn label_propagation(graph: &Graph, max_iterations: usize) -> Components {
    label_propagation_cancellable(graph, max_iterations, &|| false).expect("never cancels")
}

/// [`label_propagation`] that polls `cancel` periodically.
pub fn label_propagation_cancellable(
    graph: &Graph,
    max_iterations: usize,
    cancel: &dyn Fn() -> bool,
) -> Result<Components, Cancelled> {
    let index = dense_index(graph);
    let n = graph.node_count();
    let nodes = graph.nodes();
    let adj = undirected_adjacency(graph, &index);

    // Each node starts in its own community (its dense index as the label).
    let mut label: Vec<usize> = (0..n).collect();
    let mut since_check = 0usize;

    for _ in 0..max_iterations {
        let mut changed = false;
        // In-place (asynchronous) sweep in node order: a node sees neighbours
        // already updated this sweep, which is what damps oscillation.
        for v in 0..n {
            if adj[v].is_empty() {
                continue;
            }
            let mut freq: HashMap<usize, usize> = HashMap::new();
            for &w in &adj[v] {
                *freq.entry(label[w]).or_insert(0) += 1;
            }
            // freq is non-empty (adj[v] is non-empty), so max() is safe.
            let max_count = *freq.values().max().unwrap();
            let keep_current = freq.get(&label[v]).copied().unwrap_or(0) == max_count;
            let new_label = if keep_current {
                label[v]
            } else {
                // Smallest label id achieving the maximum frequency.
                freq.iter()
                    .filter(|&(_, &c)| c == max_count)
                    .map(|(&l, _)| l)
                    .min()
                    .unwrap()
            };
            if new_label != label[v] {
                label[v] = new_label;
                changed = true;
            }
            since_check += 1;
            if since_check >= CANCEL_CHECK_STRIDE {
                since_check = 0;
                if cancel() {
                    return Err(Cancelled);
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Relabel raw labels to dense component ids in first-seen (node) order.
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut assignment = HashMap::with_capacity(n);
    for (v, &node) in nodes.iter().enumerate() {
        let raw = label[v];
        let c = match remap.get(&raw) {
            Some(&c) => c,
            None => {
                let c = remap.len();
                remap.insert(raw, c);
                c
            }
        };
        assignment.insert(node, c);
    }
    Ok(Components {
        assignment,
        count: remap.len(),
    })
}

// ===========================================================================
// Single-source shortest paths (BFS / Dijkstra)
// ===========================================================================

/// Result of a single-source shortest-path computation. Only nodes reachable
/// from the source appear (the source itself at distance 0); unreachable nodes
/// are absent.
#[derive(Debug, Clone, Default)]
pub struct ShortestPaths {
    /// `node -> shortest distance from the source` — hop count when unweighted,
    /// summed edge weight when weighted.
    pub distance: HashMap<NodeId, f64>,
    /// `node -> number of edges on the shortest path from the source`.
    pub hops: HashMap<NodeId, usize>,
}

/// Single-source shortest paths from `source`, following directed edges.
/// `weighted = false` runs BFS (every edge costs one hop). `weighted = true`
/// runs Dijkstra over the edge weights, which **must be non-negative**
/// (negative-weight edges are skipped, since Dijkstra's greedy choice is unsound
/// with them). Returns an empty result when `source` is not in the graph.
pub fn shortest_paths(graph: &Graph, source: NodeId, weighted: bool) -> ShortestPaths {
    shortest_paths_cancellable(graph, source, weighted, &|| false).expect("never cancels")
}

/// [`shortest_paths`] that polls `cancel` periodically.
pub fn shortest_paths_cancellable(
    graph: &Graph,
    source: NodeId,
    weighted: bool,
    cancel: &dyn Fn() -> bool,
) -> Result<ShortestPaths, Cancelled> {
    let mut distance: HashMap<NodeId, f64> = HashMap::new();
    let mut hops: HashMap<NodeId, usize> = HashMap::new();
    // Unknown source → no rows.
    if graph.out_edges(source).is_none() {
        return Ok(ShortestPaths { distance, hops });
    }
    let mut since_check = 0usize;

    if weighted {
        // Dijkstra with a binary min-heap. The NodeId tie-breaks equal
        // distances so the pop order — and thus the result — is deterministic.
        distance.insert(source, 0.0);
        hops.insert(source, 0);
        let mut heap: BinaryHeap<Reverse<(MinF64, NodeId)>> = BinaryHeap::new();
        heap.push(Reverse((MinF64(0.0), source)));
        while let Some(Reverse((MinF64(d), u))) = heap.pop() {
            // Skip stale heap entries left by an earlier, worse relaxation.
            if d > *distance.get(&u).unwrap_or(&f64::INFINITY) {
                continue;
            }
            let cur_hops = hops.get(&u).copied().unwrap_or(0);
            if let Some(edges) = graph.out_edges(u) {
                for &(v, w) in edges {
                    if w < 0.0 {
                        continue; // Dijkstra is unsound with negative weights
                    }
                    let nd = d + w;
                    if nd < *distance.get(&v).unwrap_or(&f64::INFINITY) {
                        distance.insert(v, nd);
                        hops.insert(v, cur_hops + 1);
                        heap.push(Reverse((MinF64(nd), v)));
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
        }
    } else {
        // BFS: first time we reach a node is its shortest hop distance.
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        distance.insert(source, 0.0);
        hops.insert(source, 0);
        visited.insert(source);
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            let du = distance[&u];
            let hu = hops[&u];
            if let Some(edges) = graph.out_edges(u) {
                for &(v, _) in edges {
                    if visited.insert(v) {
                        distance.insert(v, du + 1.0);
                        hops.insert(v, hu + 1);
                        queue.push_back(v);
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
        }
    }

    Ok(ShortestPaths { distance, hops })
}

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Build a dense `node -> index` map in the graph's insertion order.
fn dense_index(graph: &Graph) -> HashMap<NodeId, usize> {
    let mut index = HashMap::with_capacity(graph.node_count());
    for (i, &n) in graph.nodes().iter().enumerate() {
        index.insert(n, i);
    }
    index
}

/// Undirected simple-graph adjacency in dense indices: symmetric, deduplicated,
/// self-loops removed. Used by the undirected-view kernels (triangle count,
/// label propagation).
fn undirected_adjacency(graph: &Graph, index: &HashMap<NodeId, usize>) -> Vec<HashSet<usize>> {
    let n = graph.node_count();
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (&src, nbrs) in &graph.out {
        let Some(&si) = index.get(&src) else {
            continue;
        };
        for &(dst, _) in nbrs {
            let Some(&di) = index.get(&dst) else {
                continue;
            };
            if si == di {
                continue; // drop self-loops
            }
            adj[si].insert(di);
            adj[di].insert(si);
        }
    }
    adj
}

/// Total order over `f64` for use as a min-heap key (via [`Reverse`]). Distances
/// here are finite sums of non-negative weights, so NaN never arises; `total_cmp`
/// keeps it well-defined regardless.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MinF64(f64);
impl Eq for MinF64 {}
impl PartialOrd for MinF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
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

    // --- degree centrality ---------------------------------------------------

    #[test]
    fn degree_counts_in_out_and_isolates() {
        // a -> b, a -> c, b -> a, plus isolate d.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(a, c, None);
        g.add_edge(b, a, None);
        g.add_node(d);

        let deg = degrees(&g);
        assert_eq!(deg.out_degree[&a], 2);
        assert_eq!(deg.in_degree[&a], 1);
        assert_eq!(deg.total(&a), 3);
        assert_eq!(deg.in_degree[&b], 1);
        assert_eq!(deg.out_degree[&b], 1);
        assert_eq!(deg.in_degree[&c], 1);
        assert_eq!(deg.out_degree[&c], 0);
        // Isolate present at zero.
        assert_eq!(deg.total(&d), 0);
        assert_eq!(deg.in_degree[&d], 0);
        assert_eq!(deg.out_degree[&d], 0);
    }

    #[test]
    fn degree_self_loop_counts_both_sides() {
        let a = nid([1; 16]);
        let mut g = Graph::new();
        g.add_edge(a, a, None);
        let deg = degrees(&g);
        assert_eq!(deg.in_degree[&a], 1);
        assert_eq!(deg.out_degree[&a], 1);
        assert_eq!(deg.total(&a), 2);
    }

    // --- strongly connected components --------------------------------------

    #[test]
    fn scc_directed_cycle_is_one_component() {
        // a -> b -> c -> a : a single SCC of size 3.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(b, c, None);
        g.add_edge(c, a, None);
        let scc = strongly_connected_components(&g);
        assert_eq!(scc.count, 1);
        assert_eq!(scc.assignment[&a], scc.assignment[&b]);
        assert_eq!(scc.assignment[&b], scc.assignment[&c]);
    }

    #[test]
    fn scc_dag_each_node_its_own_component() {
        // a -> b -> c (acyclic): three singleton SCCs even though WCC is one.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(b, c, None);
        let scc = strongly_connected_components(&g);
        assert_eq!(scc.count, 3);
        assert_ne!(scc.assignment[&a], scc.assignment[&b]);
        assert_ne!(scc.assignment[&b], scc.assignment[&c]);
        // Sanity: WCC collapses the same graph into one component.
        assert_eq!(weakly_connected_components(&g).count, 1);
    }

    #[test]
    fn scc_two_cycles_joined_by_a_bridge() {
        // {a<->b} ==bridge==> {c<->d} : two SCCs, the bridge does not merge them.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(b, a, None);
        g.add_edge(b, c, None); // one-way bridge
        g.add_edge(c, d, None);
        g.add_edge(d, c, None);
        let scc = strongly_connected_components(&g);
        assert_eq!(scc.count, 2);
        assert_eq!(scc.assignment[&a], scc.assignment[&b]);
        assert_eq!(scc.assignment[&c], scc.assignment[&d]);
        assert_ne!(scc.assignment[&a], scc.assignment[&c]);
    }

    #[test]
    fn scc_self_loop_and_isolate() {
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let mut g = Graph::new();
        g.add_edge(a, a, None); // self-loop SCC of size 1
        g.add_node(b); // isolate
        let scc = strongly_connected_components(&g);
        assert_eq!(scc.count, 2);
        assert_ne!(scc.assignment[&a], scc.assignment[&b]);
    }

    #[test]
    fn scc_deep_chain_does_not_overflow_stack() {
        // A long directed chain would blow a recursive DFS; the iterative one
        // must handle it. 50k nodes, all singleton SCCs.
        let mut g = Graph::new();
        let mut prev = nid(1u128.to_le_bytes());
        g.add_node(prev);
        for i in 2u128..=50_000 {
            let cur = nid(i.to_le_bytes());
            g.add_edge(prev, cur, None);
            prev = cur;
        }
        let scc = strongly_connected_components(&g);
        assert_eq!(scc.count, 50_000);
    }

    // --- triangle count ------------------------------------------------------

    #[test]
    fn triangle_count_triangle_and_coefficient() {
        // Triangle a-b-c plus a pendant d off a.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(b, c, None);
        g.add_edge(c, a, None);
        g.add_edge(a, d, None);
        let tri = triangle_count(&g);
        assert_eq!(tri.total, 1);
        // a, b, c each in the one triangle.
        assert_eq!(tri.per_node[&a], 1);
        assert_eq!(tri.per_node[&b], 1);
        assert_eq!(tri.per_node[&c], 1);
        assert_eq!(tri.per_node[&d], 0);
        // b has neighbours {a, c} which are connected → coefficient 1.0.
        assert!((tri.coefficient[&b] - 1.0).abs() < 1e-9);
        // a has neighbours {b, c, d}; only b-c connected → 2*1/(3*2) = 1/3.
        assert!((tri.coefficient[&a] - (1.0 / 3.0)).abs() < 1e-9);
        // d has degree 1 → coefficient 0.
        assert_eq!(tri.coefficient[&d], 0.0);
    }

    #[test]
    fn triangle_count_none_in_a_line() {
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(b, c, None);
        let tri = triangle_count(&g);
        assert_eq!(tri.total, 0);
        assert!(tri.per_node.values().all(|&t| t == 0));
    }

    #[test]
    fn triangle_count_k4_has_four_triangles() {
        // Complete graph on 4 nodes: C(4,3) = 4 triangles, each node in 3.
        let ids: Vec<NodeId> = (1u128..=4).map(|i| nid(i.to_le_bytes())).collect();
        let mut g = Graph::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                g.add_edge(ids[i], ids[j], None);
            }
        }
        let tri = triangle_count(&g);
        assert_eq!(tri.total, 4);
        for id in &ids {
            assert_eq!(tri.per_node[id], 3);
            assert!((tri.coefficient[id] - 1.0).abs() < 1e-9);
        }
    }

    // --- label propagation ---------------------------------------------------

    #[test]
    fn label_propagation_separates_disconnected_cliques() {
        // Two disconnected triangles → two communities, each collapsing to one
        // label. (A single weak bridge between them could legitimately merge
        // them under any label propagation, so the cliques are disconnected.)
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let x = nid([11; 16]);
        let y = nid([12; 16]);
        let z = nid([13; 16]);
        let mut g = Graph::new();
        for (s, t) in [(a, b), (b, c), (c, a)] {
            g.add_edge(s, t, None);
        }
        for (s, t) in [(x, y), (y, z), (z, x)] {
            g.add_edge(s, t, None);
        }
        let comm = label_propagation(&g, LABEL_PROPAGATION_DEFAULT_ITERS);
        assert_eq!(comm.assignment[&a], comm.assignment[&b]);
        assert_eq!(comm.assignment[&b], comm.assignment[&c]);
        assert_eq!(comm.assignment[&x], comm.assignment[&y]);
        assert_eq!(comm.assignment[&y], comm.assignment[&z]);
        assert_eq!(comm.count, 2);
        assert_ne!(comm.assignment[&a], comm.assignment[&x]);
    }

    #[test]
    fn label_propagation_connected_pair_is_one_community() {
        // The async schedule must NOT flip-flop a single edge into two labels.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        let comm = label_propagation(&g, LABEL_PROPAGATION_DEFAULT_ITERS);
        assert_eq!(comm.count, 1);
        assert_eq!(comm.assignment[&a], comm.assignment[&b]);
    }

    #[test]
    fn label_propagation_is_deterministic() {
        // Same graph twice → identical assignment (no randomness).
        let mut g = Graph::new();
        for i in 1u128..=20 {
            let s = nid(i.to_le_bytes());
            let t = nid((i % 20 + 1).to_le_bytes());
            g.add_edge(s, t, None);
        }
        let first = label_propagation(&g, 20);
        let second = label_propagation(&g, 20);
        assert_eq!(first.assignment, second.assignment);
        assert_eq!(first.count, second.count);
    }

    #[test]
    fn label_propagation_isolates_are_singletons() {
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let mut g = Graph::new();
        g.add_node(a);
        g.add_node(b);
        let comm = label_propagation(&g, 5);
        assert_eq!(comm.count, 2);
        assert_ne!(comm.assignment[&a], comm.assignment[&b]);
    }

    // --- shortest paths ------------------------------------------------------

    #[test]
    fn shortest_path_bfs_hop_counts() {
        // a -> b -> c -> d and a -> d (a shortcut): d is one hop away.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(b, c, None);
        g.add_edge(c, d, None);
        g.add_edge(a, d, None);
        let sp = shortest_paths(&g, a, false);
        assert_eq!(sp.distance[&a], 0.0);
        assert_eq!(sp.distance[&b], 1.0);
        assert_eq!(sp.distance[&c], 2.0);
        assert_eq!(sp.distance[&d], 1.0); // via the shortcut, not 3 hops
        assert_eq!(sp.hops[&d], 1);
    }

    #[test]
    fn shortest_path_dijkstra_prefers_cheaper_route() {
        // a ->(10) d  and  a ->(1) b ->(1) c ->(1) d : weighted picks 3.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(a, d, Some(10.0));
        g.add_edge(a, b, Some(1.0));
        g.add_edge(b, c, Some(1.0));
        g.add_edge(c, d, Some(1.0));
        let sp = shortest_paths(&g, a, true);
        assert!((sp.distance[&d] - 3.0).abs() < 1e-9);
        assert_eq!(sp.hops[&d], 3); // three cheap hops, not the one expensive one
        assert!((sp.distance[&b] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn shortest_path_unreachable_nodes_absent() {
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let island = nid([9; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_node(island); // not reachable from a
        let sp = shortest_paths(&g, a, false);
        assert!(sp.distance.contains_key(&a));
        assert!(sp.distance.contains_key(&b));
        assert!(!sp.distance.contains_key(&island));
    }

    #[test]
    fn shortest_path_unknown_source_is_empty() {
        let a = nid([1; 16]);
        let mut g = Graph::new();
        g.add_node(a);
        let ghost = nid([200; 16]);
        let sp = shortest_paths(&g, ghost, false);
        assert!(sp.distance.is_empty());
        assert!(sp.hops.is_empty());
    }

    #[test]
    fn shortest_path_source_only_when_no_out_edges() {
        // An isolate source yields just itself at distance 0.
        let a = nid([1; 16]);
        let mut g = Graph::new();
        g.add_node(a);
        let sp = shortest_paths(&g, a, true);
        assert_eq!(sp.distance.len(), 1);
        assert_eq!(sp.distance[&a], 0.0);
    }

    // --- cancellation of the new kernels ------------------------------------

    #[test]
    fn new_kernels_honour_cancel() {
        // A graph large enough to cross the cancel-check stride in each kernel.
        let mut g = Graph::new();
        let hub = nid([0; 16]);
        for i in 1u128..=(CANCEL_CHECK_STRIDE as u128 + 50) {
            let other = nid(i.to_le_bytes());
            g.add_edge(hub, other, None);
            g.add_edge(other, hub, None); // make degrees non-trivial + a cycle
        }
        let always = || true;
        assert_eq!(degrees_cancellable(&g, &always).err(), Some(Cancelled));
        assert_eq!(
            strongly_connected_components_cancellable(&g, &always).err(),
            Some(Cancelled)
        );
        assert_eq!(
            triangle_count_cancellable(&g, &always).err(),
            Some(Cancelled)
        );
        assert_eq!(
            label_propagation_cancellable(&g, 10, &always).err(),
            Some(Cancelled)
        );
        assert_eq!(
            shortest_paths_cancellable(&g, hub, true, &always).err(),
            Some(Cancelled)
        );
        // never-cancel returns Ok for all.
        let never = || false;
        assert!(degrees_cancellable(&g, &never).is_ok());
        assert!(strongly_connected_components_cancellable(&g, &never).is_ok());
        assert!(triangle_count_cancellable(&g, &never).is_ok());
        assert!(label_propagation_cancellable(&g, 10, &never).is_ok());
        assert!(shortest_paths_cancellable(&g, hub, false, &never).is_ok());
    }
}
