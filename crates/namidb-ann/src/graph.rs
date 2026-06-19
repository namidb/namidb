//! [`VamanaGraph`] — the bounded-degree search graph produced by the Vamana
//! build, plus its invariants. The search machinery lives in [`crate::search`].

use serde::{Deserialize, Serialize};

/// A Vamana/DiskANN search graph: for each member `id`, the ids of its graph
/// out-neighbours (`adjacency[id]`), and a single entry point (`entry`) —
/// conventionally the approximate medoid of the set.
///
/// `id`s are dense `0..adjacency.len()` and the graph is undirected in
/// *intent* (the build links both directions) but stored as directed adjacency
/// lists; search only ever follows out-edges. Neighbour lists are bounded by
/// the build degree `R` (robust-prune caps them); the field is public so the
/// storage layer can encode it directly.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VamanaGraph {
    /// `adjacency[id]` = out-neighbours of `id`. Empty for an empty graph.
    pub adjacency: Vec<Vec<u32>>,
    /// Entry-point medoid. `0` for an empty/1-node graph.
    pub entry: u32,
}

impl VamanaGraph {
    /// Construct from built adjacency + entry. Cheap; does not validate degree
    /// bounds (the build enforces those) but checks the entry is in range.
    #[inline]
    pub fn new(adjacency: Vec<Vec<u32>>, entry: u32) -> Self {
        debug_assert!(
            adjacency.is_empty() || (entry as usize) < adjacency.len(),
            "entry {entry} out of range for {} nodes",
            adjacency.len()
        );
        Self { adjacency, entry }
    }

    /// Number of members (nodes) in the indexed set.
    #[inline]
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// `true` iff the graph indexes zero members.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// Out-neighbours of `id`. Panics if `id` is out of range (callers iterate
    /// `0..len()`).
    #[inline]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        &self.adjacency[id as usize]
    }

    /// Maximum out-degree over all nodes — a health check that the build's `R`
    /// bound held (transient overshoot during build is pruned before finish).
    pub fn max_degree(&self) -> usize {
        self.adjacency.iter().map(|n| n.len()).max().unwrap_or(0)
    }

    /// Total number of directed edges (`Σ |adjacency[i]|`).
    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(|n| n.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors() {
        let g = VamanaGraph::new(vec![vec![1, 2], vec![0], vec![0]], 0);
        assert_eq!(g.len(), 3);
        assert!(!g.is_empty());
        assert_eq!(g.neighbors(0), &[1, 2]);
        assert_eq!(g.entry, 0);
        assert_eq!(g.max_degree(), 2);
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn empty_graph() {
        let g = VamanaGraph::new(vec![], 0);
        assert!(g.is_empty());
        assert_eq!(g.max_degree(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn single_node() {
        let g = VamanaGraph::new(vec![vec![]], 0);
        assert_eq!(g.len(), 1);
        assert!(g.neighbors(0).is_empty());
    }

    #[test]
    fn serde_round_trip() {
        let g = VamanaGraph::new(vec![vec![2], vec![2], vec![0, 1]], 2);
        let json = serde_json::to_string(&g).unwrap();
        let back: VamanaGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.adjacency, g.adjacency);
        assert_eq!(back.entry, g.entry);
    }
}
