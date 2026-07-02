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

    /// `true` if any edge carries a negative weight. Weighted shortest-path
    /// (Dijkstra) is unsound on negative weights, so the caller rejects the
    /// query rather than returning silently-wrong distances.
    pub fn has_negative_weight(&self) -> bool {
        self.out
            .values()
            .any(|nbrs| nbrs.iter().any(|&(_, w)| w < 0.0))
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
    //
    // Iterate in node-INSERTION order (`graph.nodes()`), not `&index` HashMap
    // order: component ids are allocated first-come, so iterating the randomized
    // HashMap would label the same partition differently every run. The partition
    // is always correct either way, but stable ids are required for snapshot
    // tests, cross-kernel joins, and reproducible output — matching the
    // deterministic relabel that `label_propagation` already does.
    let mut root_to_comp: HashMap<usize, usize> = HashMap::new();
    let mut assignment = HashMap::with_capacity(n);
    since_check = 0;
    for &node in graph.nodes() {
        let i = index[&node];
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
        // no out-edges, OR no positive out-edge weight (a degenerate case the
        // push loop below skips). Counting the latter as dangling is load-
        // bearing — without it that node's rank would simply vanish each
        // iteration and the scores would stop summing to 1.
        //
        // PageRank is only defined for non-negative weights, so a NEGATIVE edge
        // is treated as absent: the weight sum and the per-edge push both use
        // `max(w, 0)`. Keying the guards on the raw signed sum was a bug — a node
        // mixing signs but summing positive (e.g. +3 and −2) passed both guards
        // and the −2 edge injected negative mass, producing negative (and
        // compensating >1) scores.
        let positive_sum =
            |e: &[(NodeId, f64)]| -> f64 { e.iter().map(|&(_, w)| w.max(0.0)).sum::<f64>() };
        let is_dangling = |n: &NodeId| match graph.out_edges(*n) {
            None | Some([]) => true,
            Some(e) => positive_sum(e) <= 0.0,
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
        // Push each node's rank along its (positive-weight) out-edges. Total
        // out-mass must be exactly `d * PR(src)` to conserve probability, so the
        // per-edge share is `d * PR(src) * (w / Σw⁺)` — normalized by the POSITIVE
        // weight sum, not the edge count. A non-positive weight sum is degenerate
        // and was already counted as dangling above (its mass is redistributed),
        // so skip it here rather than divide by zero / produce NaN.
        // Iterate sources in `nodes()` (insertion) order, NOT `graph.out`
        // HashMap order: f64 addition is non-associative, so accumulating each
        // dst's incoming mass in HashMap-iteration order (randomised per process)
        // made scores — and thus near-tie rankings — differ run-to-run for the
        // same graph. Insertion order is deterministic, so results are now
        // reproducible.
        for &src in graph.nodes() {
            let nbrs = match graph.out.get(&src) {
                Some(n) if !n.is_empty() => n,
                _ => continue,
            };
            let weight_sum = positive_sum(nbrs);
            if weight_sum <= 0.0 {
                continue;
            }
            let base = d * scores[&src] / weight_sum;
            for &(dst, w) in nbrs {
                if w <= 0.0 {
                    continue;
                }
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
// FastRP — Fast Random Projection node embeddings
// ===========================================================================

/// Knobs for [`fast_rp`]. Defaults follow Neo4j GDS: dimension 256, four
/// propagation hops with `[0, 1, 1, 1]` weights (hop 0 — the raw random
/// projection — is dropped), undirected mean propagation.
#[derive(Clone, Debug)]
pub struct FastRpOptions {
    /// Embedding dimensionality `d`.
    pub dimension: usize,
    /// One weight per hop, length `iterations + 1`. `iteration_weights[0]`
    /// weights the initial random projection (hop 0); `[k]` weights the k-th
    /// propagation. The embedding is `Σ_k w[k] · R_k`.
    pub iteration_weights: Vec<f32>,
    /// Degree-normalization exponent `β` on the *source* neighbour: a message
    /// from `j` to `i` is scaled by `deg(j)^β / deg(i)`. `0.0` (default) is plain
    /// mean propagation (`D⁻¹A`).
    pub normalization_strength: f32,
    /// Seed for the deterministic sparse random projection — same
    /// `(graph, options, seed)` always yields the same embeddings.
    pub seed: u64,
}

impl Default for FastRpOptions {
    fn default() -> Self {
        Self {
            dimension: 256,
            iteration_weights: vec![0.0, 1.0, 1.0, 1.0],
            normalization_strength: 0.0,
            seed: 42,
        }
    }
}

/// FastRP result: a `d`-dimensional f32 embedding per node — exactly the
/// `(NodeId, Vec<f32>)` shape the vector index ingests, so structural embeddings
/// can be written straight into a `.vg`.
#[derive(Debug, Clone)]
pub struct FastRp {
    pub embeddings: HashMap<NodeId, Vec<f32>>,
    pub dimension: usize,
}

/// SplitMix64 — a tiny deterministic PRNG for the sparse random projection (no
/// external RNG dependency, fully reproducible across platforms).
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// L2-normalise a vector in place (no-op for a zero vector).
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

/// FastRP node embeddings over the undirected graph. See [`fast_rp_cancellable`].
pub fn fast_rp(graph: &Graph, opts: &FastRpOptions) -> FastRp {
    fast_rp_cancellable(graph, opts, &|| false).expect("non-cancelling closure cannot fail")
}

/// FastRP (Chen et al., "Fast and Accurate Network Embeddings via Very Sparse
/// Random Projection", CIKM 2019). Each node starts from a very-sparse random
/// projection (`±√s` with probability `1/(2s)` each, else 0, with `s = 3`),
/// L2-normalised; that signal is propagated over the degree-normalised
/// undirected adjacency for `iterations` hops, and the hops are combined with
/// `iteration_weights`. Near-linear (`O(iterations · (E·d))`), deterministic for
/// a fixed seed, and cancellable per hop.
///
/// The node iteration order is the graph's insertion order (`graph.nodes()`),
/// never HashMap order, so embeddings are reproducible. Parallel/both-direction
/// edges raise a node's effective degree (a multigraph view), matching how the
/// other kernels here treat the edge multiset.
pub fn fast_rp_cancellable(
    graph: &Graph,
    opts: &FastRpOptions,
    cancel: &dyn Fn() -> bool,
) -> Result<FastRp, Cancelled> {
    let nodes = graph.nodes();
    let n = nodes.len();
    let d = opts.dimension;
    if n == 0 || d == 0 {
        return Ok(FastRp {
            embeddings: HashMap::new(),
            dimension: d,
        });
    }
    let pos: HashMap<NodeId, usize> = nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Undirected adjacency (each stored out-edge links both directions). Built
    // in node-INSERTION order (`nodes`), not `graph.out` HashMap order, so each
    // `adj[i]` neighbour list is deterministic — f32 propagation sums are
    // non-associative, so a different neighbour order would yield different
    // embeddings run-to-run / across platforms, breaking the seed determinism.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &src in nodes {
        let si = pos[&src];
        if let Some(nbrs) = graph.out_edges(src) {
            for &(dst, _) in nbrs {
                if let Some(&di) = pos.get(&dst) {
                    adj[si].push(di);
                    adj[di].push(si);
                }
            }
        }
    }
    let deg: Vec<f64> = adj.iter().map(|a| a.len() as f64).collect();

    // Hop 0: very-sparse random projection (s = 3), L2-normalised per node.
    let scale = (3.0f32).sqrt();
    let mut r_prev: Vec<Vec<f32>> = (0..n)
        .map(|p| {
            let node_seed = splitmix64(opts.seed ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut v = vec![0.0f32; d];
            for (dim, slot) in v.iter_mut().enumerate() {
                let h = splitmix64(node_seed.wrapping_add(dim as u64));
                // Uniform in [0, 1) from the top 53 bits.
                let u = (h >> 11) as f64 / (1u64 << 53) as f64;
                *slot = if u < 1.0 / 6.0 {
                    scale
                } else if u < 1.0 / 3.0 {
                    -scale
                } else {
                    0.0
                };
            }
            l2_normalize(&mut v);
            v
        })
        .collect();

    let iterations = opts.iteration_weights.len().saturating_sub(1);
    let mut emb: Vec<Vec<f32>> = vec![vec![0.0f32; d]; n];
    let w0 = opts.iteration_weights.first().copied().unwrap_or(0.0);
    if w0 != 0.0 {
        for i in 0..n {
            for x in 0..d {
                emb[i][x] += w0 * r_prev[i][x];
            }
        }
    }

    let beta = opts.normalization_strength as f64;
    for k in 1..=iterations {
        if cancel() {
            return Err(Cancelled);
        }
        let mut r_next: Vec<Vec<f32>> = vec![vec![0.0f32; d]; n];
        for i in 0..n {
            let di = deg[i].max(1.0);
            let row = &mut r_next[i];
            for &j in &adj[i] {
                // Propagate from the previous hop (`r_prev`), so writing `row`
                // (= r_next[i]) never aliases the source.
                let cij = (deg[j].max(1.0).powf(beta) / di) as f32;
                for x in 0..d {
                    row[x] += cij * r_prev[j][x];
                }
            }
            l2_normalize(row);
        }
        let wk = opts.iteration_weights[k];
        if wk != 0.0 {
            for i in 0..n {
                for x in 0..d {
                    emb[i][x] += wk * r_next[i][x];
                }
            }
        }
        r_prev = r_next;
    }

    let embeddings = nodes
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, std::mem::take(&mut emb[i])))
        .collect();
    Ok(FastRp {
        embeddings,
        dimension: d,
    })
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

// ===========================================================================
// Louvain community detection
// ===========================================================================

/// Knobs for [`louvain`]. Defaults follow Neo4j GDS: up to 10 aggregation
/// levels, 10 local-move sweeps per level, and a modularity-gain tolerance of
/// 1e-4 between levels for early convergence.
#[derive(Clone, Debug)]
pub struct LouvainOptions {
    /// Maximum dendrogram height (levels of community aggregation).
    pub max_levels: usize,
    /// Maximum local-move sweeps within one level.
    pub max_iterations: usize,
    /// Stop adding levels once a level improves modularity by less than this.
    pub tolerance: f64,
}

impl Default for LouvainOptions {
    fn default() -> Self {
        Self {
            max_levels: 10,
            max_iterations: 10,
            tolerance: 1e-4,
        }
    }
}

/// Result of a Louvain run: the final community per node plus the modularity
/// of that partition. Like WCC, Louvain works on the *undirected* view of the
/// edges (each stored directed edge is one undirected edge of its weight).
#[derive(Debug, Clone)]
pub struct Louvain {
    /// `node -> community id`, dense in `[0, count)`.
    pub assignment: HashMap<NodeId, usize>,
    /// Number of distinct communities.
    pub count: usize,
    /// Modularity of the returned partition.
    pub modularity: f64,
}

/// Louvain modularity-based community detection over the undirected view of
/// the graph (Blondel et al. 2008): repeated local-move sweeps that greedily
/// maximise modularity, then community aggregation into super-nodes, until the
/// modularity gain between levels drops below `tolerance` or `max_levels` is
/// hit. Deterministic: nodes are swept in insertion order and ties prefer the
/// current community, then the lowest community id.
pub fn louvain(graph: &Graph, opts: &LouvainOptions) -> Louvain {
    louvain_cancellable(graph, opts, &|| false).expect("never cancels")
}

/// [`louvain`] that polls `cancel` periodically.
pub fn louvain_cancellable(
    graph: &Graph,
    opts: &LouvainOptions,
    cancel: &dyn Fn() -> bool,
) -> Result<Louvain, Cancelled> {
    let n = graph.node_count();
    let mut index: HashMap<NodeId, usize> = HashMap::with_capacity(n);
    for (i, &node) in graph.nodes().iter().enumerate() {
        index.insert(node, i);
    }

    // Level-0 undirected working graph: non-loop incidences in BOTH directions
    // plus per-node self-loop weight (each self-loop counted once).
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut loops: Vec<f64> = vec![0.0; n];
    let mut m = 0.0f64;
    for (&src, nbrs) in &graph.out {
        let si = index[&src];
        for &(dst, w) in nbrs {
            let di = index[&dst];
            m += w;
            if si == di {
                loops[si] += w;
            } else {
                adj[si].push((di, w));
                adj[di].push((si, w));
            }
        }
    }
    if m <= 0.0 || n == 0 {
        // Edgeless graph: every node is its own community, modularity 0.
        let assignment = graph
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();
        return Ok(Louvain {
            assignment,
            count: n,
            modularity: 0.0,
        });
    }

    // `membership[orig]` tracks each original node's community through the
    // levels; level-local ids are composed into it after each aggregation.
    let mut membership: Vec<usize> = (0..n).collect();
    let mut prev_q = f64::NEG_INFINITY;
    let mut since_check = 0usize;

    for _level in 0..opts.max_levels.max(1) {
        let ln = adj.len();
        // Weighted degree: incident non-loop weight + 2x self-loops.
        let k: Vec<f64> = (0..ln)
            .map(|i| adj[i].iter().map(|&(_, w)| w).sum::<f64>() + 2.0 * loops[i])
            .collect();
        let mut comm: Vec<usize> = (0..ln).collect();
        let mut sigma_tot = k.clone();
        let two_m = 2.0 * m;

        // Local-move phase: sweep nodes in index order until a full sweep
        // makes no move (or the sweep cap is hit).
        let mut moved_any = false;
        for _sweep in 0..opts.max_iterations.max(1) {
            let mut moved = false;
            let mut neigh: HashMap<usize, f64> = HashMap::new();
            for i in 0..ln {
                let c0 = comm[i];
                neigh.clear();
                for &(j, w) in &adj[i] {
                    *neigh.entry(comm[j]).or_insert(0.0) += w;
                    since_check += 1;
                    if since_check >= CANCEL_CHECK_STRIDE {
                        since_check = 0;
                        if cancel() {
                            return Err(Cancelled);
                        }
                    }
                }
                sigma_tot[c0] -= k[i];
                // Gain of joining community c (up to terms constant across
                // choices): k_i_in(c) - sigma_tot(c) * k_i / 2m. Scan
                // candidates in ascending community id and require a strictly
                // better gain, so ties keep the current community first and
                // the lowest id otherwise — deterministic regardless of
                // HashMap iteration order.
                let stay = neigh.get(&c0).copied().unwrap_or(0.0) - sigma_tot[c0] * k[i] / two_m;
                let mut candidates: Vec<usize> = neigh.keys().copied().collect();
                candidates.sort_unstable();
                let (mut best_c, mut best_gain) = (c0, stay);
                for c in candidates {
                    if c == c0 {
                        continue;
                    }
                    let gain = neigh[&c] - sigma_tot[c] * k[i] / two_m;
                    if gain > best_gain + f64::EPSILON {
                        best_c = c;
                        best_gain = gain;
                    }
                }
                sigma_tot[best_c] += k[i];
                if best_c != c0 {
                    comm[i] = best_c;
                    moved = true;
                    moved_any = true;
                }
            }
            if !moved {
                break;
            }
        }

        // Renumber communities densely by first occurrence (node order).
        let mut renumber: HashMap<usize, usize> = HashMap::new();
        for c in comm.iter_mut() {
            let next = renumber.len();
            *c = *renumber.entry(*c).or_insert(next);
        }
        let cn = renumber.len();
        for slot in membership.iter_mut() {
            *slot = comm[*slot];
        }

        // Modularity of this level's partition (networkx convention:
        // Q = sum_c internal(c)/m - (sigma_tot(c)/2m)^2, internal counting
        // each undirected edge and each self-loop once).
        let mut internal = vec![0.0f64; cn];
        let mut tot = vec![0.0f64; cn];
        for i in 0..ln {
            tot[comm[i]] += k[i];
            internal[comm[i]] += loops[i];
            for &(j, w) in &adj[i] {
                if comm[j] == comm[i] {
                    internal[comm[i]] += w / 2.0; // both directions stored
                }
            }
        }
        let q: f64 = (0..cn)
            .map(|c| internal[c] / m - (tot[c] / two_m).powi(2))
            .sum();

        let done = !moved_any || cn == ln || q - prev_q < opts.tolerance;
        prev_q = q;
        if done {
            break;
        }

        // Aggregate: communities become super-nodes; inter-community weights
        // sum (both directions preserved), intra-community weight becomes the
        // super-node's self-loop.
        let mut new_adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); cn];
        let mut new_loops = vec![0.0f64; cn];
        for i in 0..ln {
            new_loops[comm[i]] += loops[i];
            for &(j, w) in &adj[i] {
                if comm[j] == comm[i] {
                    new_loops[comm[i]] += w / 2.0;
                } else {
                    *new_adj[comm[i]].entry(comm[j]).or_insert(0.0) += w;
                }
            }
        }
        // new_loops double-added intra weight (w/2 from each direction) — that
        // is exactly once per undirected edge, as required.
        adj = new_adj
            .into_iter()
            .map(|mp| {
                let mut v: Vec<(usize, f64)> = mp.into_iter().collect();
                v.sort_unstable_by_key(|&(c, _)| c);
                v
            })
            .collect();
        loops = new_loops;
    }

    // Final relabel: dense ids by first occurrence in node-insertion order,
    // matching WCC/label_propagation's deterministic output.
    let mut relabel: HashMap<usize, usize> = HashMap::new();
    let mut assignment = HashMap::with_capacity(n);
    for (i, &node) in graph.nodes().iter().enumerate() {
        let next = relabel.len();
        let c = *relabel.entry(membership[i]).or_insert(next);
        assignment.insert(node, c);
    }
    Ok(Louvain {
        assignment,
        count: relabel.len(),
        modularity: prev_q.max(0.0),
    })
}

// ===========================================================================
// Betweenness centrality (Brandes)
// ===========================================================================

/// Result of [`betweenness`]: raw (unnormalised) betweenness centrality per
/// node over directed unit-cost shortest paths, Neo4j GDS's default. On an
/// undirected projection every s→t pair is counted in both directions, so raw
/// scores come out doubled — the caller halves them for undirected semantics.
#[derive(Debug, Clone)]
pub struct Betweenness {
    /// `node -> centrality score`.
    pub scores: HashMap<NodeId, f64>,
}

/// Brandes' exact betweenness centrality (2001): one BFS + dependency
/// accumulation per source, `O(V·E)` total for unweighted graphs.
/// Deterministic: sources are processed in insertion order and the algorithm's
/// result is order-independent.
pub fn betweenness(graph: &Graph) -> Betweenness {
    betweenness_cancellable(graph, &|| false).expect("never cancels")
}

/// [`betweenness`] that polls `cancel` periodically (per source and inside the
/// BFS inner loop).
pub fn betweenness_cancellable(
    graph: &Graph,
    cancel: &dyn Fn() -> bool,
) -> Result<Betweenness, Cancelled> {
    let n = graph.node_count();
    let mut index: HashMap<NodeId, usize> = HashMap::with_capacity(n);
    for (i, &node) in graph.nodes().iter().enumerate() {
        index.insert(node, i);
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (&src, nbrs) in &graph.out {
        let si = index[&src];
        for &(dst, _) in nbrs {
            adj[si].push(index[&dst]);
        }
    }

    let mut bc = vec![0.0f64; n];
    // Per-source scratch, reset via the visit stack instead of re-allocating.
    let mut sigma = vec![0.0f64; n];
    let mut dist = vec![-1i64; n];
    let mut delta = vec![0.0f64; n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut since_check = 0usize;

    for s in 0..n {
        let mut stack: Vec<usize> = Vec::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        sigma[s] = 1.0;
        dist[s] = 0;
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            stack.push(u);
            for &v in &adj[u] {
                if dist[v] < 0 {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
                if dist[v] == dist[u] + 1 {
                    sigma[v] += sigma[u];
                    pred[v].push(u);
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
        // Dependency accumulation in reverse BFS order.
        for &w in stack.iter().rev() {
            for &v in &pred[w] {
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w]);
            }
            if w != s {
                bc[w] += delta[w];
            }
        }
        // Reset only what this source touched.
        for &w in &stack {
            sigma[w] = 0.0;
            dist[w] = -1;
            delta[w] = 0.0;
            pred[w].clear();
        }
        if cancel() {
            return Err(Cancelled);
        }
    }

    let scores = graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, bc[i]))
        .collect();
    Ok(Betweenness { scores })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(b: [u8; 16]) -> NodeId {
        NodeId::from_uuid(uuid::Uuid::from_bytes(b))
    }

    #[test]
    fn pagerank_mixed_sign_edges_leak_no_negative_mass() {
        // `a` has two out-edges: +3 to b and −2 to c (raw sum +1, so the old
        // signed-sum guards let the −2 edge inject negative mass). The negative
        // edge must be ignored: no score is negative and mass is conserved.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, Some(3.0));
        g.add_edge(a, c, Some(-2.0));
        g.add_node(c);

        let pr = pagerank(&g, &PageRankOptions::default());
        let total: f64 = pr.scores.values().sum();
        assert!((total - 1.0).abs() < 1e-3, "mass not conserved: {total}");
        assert!(
            pr.scores.values().all(|s| s.is_finite() && *s >= 0.0),
            "negative/NaN score leaked: {:?}",
            pr.scores
        );
    }

    #[test]
    fn fastrp_is_deterministic_and_separates_communities() {
        // Two triangles joined by a single bridge edge. Nodes inside a triangle
        // should embed more similarly to each other than to the far triangle.
        let t1 = [nid([1; 16]), nid([2; 16]), nid([3; 16])];
        let t2 = [nid([4; 16]), nid([5; 16]), nid([6; 16])];
        let mut g = Graph::new();
        for tri in [&t1, &t2] {
            g.add_edge(tri[0], tri[1], None);
            g.add_edge(tri[1], tri[2], None);
            g.add_edge(tri[2], tri[0], None);
        }
        g.add_edge(t1[0], t2[0], None); // bridge

        let opts = FastRpOptions {
            dimension: 64,
            ..Default::default()
        };
        let a = fast_rp(&g, &opts);
        let b = fast_rp(&g, &opts);
        assert_eq!(a.embeddings, b.embeddings, "same seed → same embeddings");
        assert!(a.embeddings.values().all(|v| v.len() == 64));

        let cos = |x: &[f32], y: &[f32]| -> f32 {
            let dot: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
            let nx = x.iter().map(|p| p * p).sum::<f32>().sqrt();
            let ny = y.iter().map(|p| p * p).sum::<f32>().sqrt();
            if nx == 0.0 || ny == 0.0 {
                0.0
            } else {
                dot / (nx * ny)
            }
        };
        // Within-triangle similarity exceeds the cross-triangle similarity for
        // the two non-bridge nodes.
        let within = cos(&a.embeddings[&t1[1]], &a.embeddings[&t1[2]]);
        let across = cos(&a.embeddings[&t1[1]], &a.embeddings[&t2[1]]);
        assert!(
            within > across,
            "within {within} should exceed across {across}"
        );
    }

    #[test]
    fn wcc_component_ids_are_deterministic() {
        // 40 disjoint two-node components; the integer ids must be identical
        // across runs (no dependence on HashMap iteration order).
        let build = || {
            let mut g = Graph::new();
            for i in 0..40u8 {
                let x = nid([i; 16]);
                let y = nid([100 + i; 16]);
                g.add_edge(x, y, None);
            }
            weakly_connected_components(&g).assignment
        };
        assert_eq!(build(), build(), "WCC component ids must be deterministic");
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

    #[test]
    fn has_negative_weight_detects_negative_edges() {
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, Some(5.0));
        assert!(!g.has_negative_weight());
        g.add_edge(b, a, Some(-1.0));
        assert!(g.has_negative_weight());
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
        assert!(louvain_cancellable(&g, &LouvainOptions::default(), &never).is_ok());
        assert!(betweenness_cancellable(&g, &never).is_ok());
        let always = || true;
        assert_eq!(
            louvain_cancellable(&g, &LouvainOptions::default(), &always).err(),
            Some(Cancelled)
        );
        assert_eq!(betweenness_cancellable(&g, &always).err(), Some(Cancelled));
    }

    #[test]
    fn louvain_separates_two_bridged_cliques() {
        // Two 4-cliques joined by one bridge edge: the canonical Louvain
        // fixture. Each clique must land in one community, distinct from the
        // other clique's, with clearly positive modularity.
        let c1: Vec<NodeId> = (1u8..=4).map(|b| nid([b; 16])).collect();
        let c2: Vec<NodeId> = (5u8..=8).map(|b| nid([b; 16])).collect();
        let mut g = Graph::new();
        for cl in [&c1, &c2] {
            for i in 0..cl.len() {
                for j in (i + 1)..cl.len() {
                    g.add_edge(cl[i], cl[j], None);
                }
            }
        }
        g.add_edge(c1[0], c2[0], None);

        let res = louvain(&g, &LouvainOptions::default());
        assert_eq!(res.count, 2, "two cliques → two communities");
        let comm1 = res.assignment[&c1[0]];
        let comm2 = res.assignment[&c2[0]];
        assert_ne!(comm1, comm2);
        assert!(c1.iter().all(|n| res.assignment[n] == comm1), "clique 1 intact");
        assert!(c2.iter().all(|n| res.assignment[n] == comm2), "clique 2 intact");
        assert!(
            res.modularity > 0.3,
            "bridged cliques have high modularity, got {}",
            res.modularity
        );
    }

    #[test]
    fn louvain_edgeless_graph_is_all_singletons() {
        let mut g = Graph::new();
        let nodes: Vec<NodeId> = (1u8..=3).map(|b| nid([b; 16])).collect();
        for &n in &nodes {
            g.add_node(n);
        }
        let res = louvain(&g, &LouvainOptions::default());
        assert_eq!(res.count, 3);
        assert_eq!(res.modularity, 0.0);
        let distinct: HashSet<usize> = res.assignment.values().copied().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn louvain_is_deterministic() {
        let c1: Vec<NodeId> = (1u8..=5).map(|b| nid([b; 16])).collect();
        let c2: Vec<NodeId> = (6u8..=10).map(|b| nid([b; 16])).collect();
        let mut g = Graph::new();
        for cl in [&c1, &c2] {
            for i in 0..cl.len() {
                for j in (i + 1)..cl.len() {
                    g.add_edge(cl[i], cl[j], None);
                }
            }
        }
        g.add_edge(c1[4], c2[0], None);
        let a = louvain(&g, &LouvainOptions::default());
        for _ in 0..5 {
            let b = louvain(&g, &LouvainOptions::default());
            assert_eq!(a.assignment, b.assignment);
            assert_eq!(a.modularity, b.modularity);
        }
    }

    #[test]
    fn betweenness_directed_path_exact_values() {
        // a→b→c→d→e: interior nodes carry all pass-through pairs.
        // b: a→{c,d,e} = 3; c: {a,b}→{d,e} = 4; d: {a,b,c}→e = 3; ends 0.
        let ids: Vec<NodeId> = (1u8..=5).map(|b| nid([b; 16])).collect();
        let mut g = Graph::new();
        for w in ids.windows(2) {
            g.add_edge(w[0], w[1], None);
        }
        let bc = betweenness(&g);
        assert_eq!(bc.scores[&ids[0]], 0.0);
        assert_eq!(bc.scores[&ids[1]], 3.0);
        assert_eq!(bc.scores[&ids[2]], 4.0);
        assert_eq!(bc.scores[&ids[3]], 3.0);
        assert_eq!(bc.scores[&ids[4]], 0.0);
    }

    #[test]
    fn betweenness_splits_over_equal_shortest_paths() {
        // Diamond a→{b,c}→d: two equal-length paths, so b and c each carry
        // half of the single a→d dependency.
        let a = nid([1; 16]);
        let b = nid([2; 16]);
        let c = nid([3; 16]);
        let d = nid([4; 16]);
        let mut g = Graph::new();
        g.add_edge(a, b, None);
        g.add_edge(a, c, None);
        g.add_edge(b, d, None);
        g.add_edge(c, d, None);
        let bc = betweenness(&g);
        assert_eq!(bc.scores[&b], 0.5);
        assert_eq!(bc.scores[&c], 0.5);
        assert_eq!(bc.scores[&a], 0.0);
        assert_eq!(bc.scores[&d], 0.0);
    }
}
