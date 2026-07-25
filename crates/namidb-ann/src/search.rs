//! Greedy best-first beam search over a [`VamanaGraph`].
//!
//! The core [`search_with`] takes a distance *closure* `dist(id) -> f32`, so
//! the build (member-to-member, via [`VectorSpace::pair_distance`]) and the
//! query path (via [`VectorSpace::query_distance`]) share one loop. The public
//! [`search`] wraps it for the query case and returns the top-`k` neighbours.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::graph::VamanaGraph;
use crate::space::VectorSpace;

/// A search hit: a member `id` and its (lower-is-closer) distance to the query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub dist: f32,
}

/// A distance-tagged node with a **total** order on `dist` (then `id`). Distances
/// are finite by the [`VectorSpace`] contract, so `total_cmp` never sees `NaN`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DistNode {
    dist: f32,
    id: u32,
}

impl Eq for DistNode {}

impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.dist.total_cmp(&other.dist) {
            Ordering::Equal => self.id.cmp(&other.id),
            o => o,
        }
    }
}

impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable visited-state for [`beam_search_with_scratch`].
///
/// The build invokes beam search once per indexed point. Clearing a dense
/// `Vec<bool>` of length `n` before every invocation turns that loop into
/// `O(n²)` memory writes even when `l_build` is small. Dense epoch marks pay
/// one `O(n)` allocation for the whole build and then reset in `O(1)`.
///
/// One-off public queries use the sparse variant: its memory and clearing work
/// scale with the nodes actually reached by the beam, not with the full corpus.
pub(crate) struct BeamScratch {
    marks: VisitMarks,
}

enum VisitMarks {
    /// Fast repeated probes over one fixed corpus (the Vamana build).
    Dense {
        epochs: Vec<u32>,
        current_epoch: u32,
    },
    /// One-off probes where allocating `O(n)` state would dominate a small beam.
    Sparse(HashSet<u32>),
}

impl BeamScratch {
    /// Dense epoch marks for a sequence of searches over `n` graph nodes.
    pub(crate) fn dense(n: usize) -> Self {
        Self {
            marks: VisitMarks::Dense {
                epochs: vec![0; n],
                current_epoch: 0,
            },
        }
    }

    /// Sparse marks for a one-off search. `expected` is only an allocation hint;
    /// the set grows if the beam reaches more nodes.
    fn sparse(expected: usize) -> Self {
        Self {
            marks: VisitMarks::Sparse(HashSet::with_capacity(expected)),
        }
    }

    /// Start a logically empty visited set for a graph of `n` nodes.
    fn begin(&mut self, n: usize) {
        match &mut self.marks {
            VisitMarks::Dense {
                epochs,
                current_epoch,
            } => {
                // Build currently fixes `n`, but resize keeps this scratch safe
                // for future callers that reuse it with a larger graph.
                if epochs.len() < n {
                    epochs.resize(n, 0);
                }
                *current_epoch = current_epoch.wrapping_add(1);
                if *current_epoch == 0 {
                    // One full clear per 2^32 searches, rather than per search.
                    epochs.fill(0);
                    *current_epoch = 1;
                }
            }
            VisitMarks::Sparse(visited) => visited.clear(),
        }
    }

    /// Mark `id`, returning `true` only on its first visit in this search.
    #[inline]
    fn mark_if_new(&mut self, id: u32) -> bool {
        match &mut self.marks {
            VisitMarks::Dense {
                epochs,
                current_epoch,
            } => {
                let mark = &mut epochs[id as usize];
                if *mark == *current_epoch {
                    false
                } else {
                    *mark = *current_epoch;
                    true
                }
            }
            VisitMarks::Sparse(visited) => visited.insert(id),
        }
    }

    #[cfg(test)]
    fn resident_marks(&self) -> usize {
        match &self.marks {
            VisitMarks::Dense { epochs, .. } => epochs.len(),
            VisitMarks::Sparse(visited) => visited.len(),
        }
    }

    #[cfg(test)]
    fn dense_allocation_ptr(&self) -> Option<*const u32> {
        match &self.marks {
            VisitMarks::Dense { epochs, .. } => Some(epochs.as_ptr()),
            VisitMarks::Sparse(_) => None,
        }
    }
}

/// Beam search core, parameterized by the adjacency slice. Shared by the query
/// path ([`search`], over a finished [`VamanaGraph`]) and the build (over the
/// in-progress adjacency `&adj`), so the build never clones the graph-so-far.
///
/// Starting at `entry`, greedily expand the closest unseen candidate, keeping
/// the `ef` closest results seen so far, until no unexplored candidate can
/// improve the result set. Returns the `k` nearest (sorted ascending by
/// distance). `ef` is clamped to `≥ k` and `≤ n`.
pub(crate) fn beam_search(
    adjacency: &[Vec<u32>],
    n: usize,
    entry: u32,
    k: usize,
    ef: usize,
    dist: impl Fn(u32) -> f32,
) -> Vec<Neighbor> {
    // Public searches are normally tiny relative to the corpus (`ef=64` by
    // default). A sparse scratch avoids allocating/zeroing `n` slots per query.
    // Cap the eager reservation: `ef` is caller-controlled, and an extreme
    // value must not turn into a giant hash-table allocation before traversal.
    const MAX_EAGER_SPARSE_MARKS: usize = 4_096;
    let expected_visits = ef.max(k).min(n).min(MAX_EAGER_SPARSE_MARKS);
    let mut scratch = BeamScratch::sparse(expected_visits);
    beam_search_with_scratch(adjacency, n, entry, k, ef, dist, &mut scratch)
}

/// Beam search using caller-owned visited state. The Vamana builder reuses one
/// dense epoch scratch across all `n` refinement searches.
pub(crate) fn beam_search_with_scratch(
    adjacency: &[Vec<u32>],
    n: usize,
    entry: u32,
    k: usize,
    ef: usize,
    dist: impl Fn(u32) -> f32,
    scratch: &mut BeamScratch,
) -> Vec<Neighbor> {
    if n == 0 || k == 0 {
        return Vec::new();
    }
    // The entry point comes from a stored graph that may have been decoded from
    // object storage (the `.vg` body has no checksum), so an out-of-range entry
    // is possible on a corrupt/foreign file. Neighbour ids are already bounds-
    // checked in the expansion loop; guard the entry the same way rather than
    // indexing `visited[entry]` / `adjacency[entry]` and panicking.
    if entry as usize >= n {
        return Vec::new();
    }
    let k = k.min(n);
    let ef = ef.max(k).min(n);

    scratch.begin(n);

    // `candidates`: min-heap (closest-first) of things to expand.
    let mut candidates: BinaryHeap<std::cmp::Reverse<DistNode>> = BinaryHeap::with_capacity(ef + 1);
    // `results`: max-heap of the ef closest seen; peek() = the current farthest.
    let mut results: BinaryHeap<DistNode> = BinaryHeap::with_capacity(ef + 1);

    scratch.mark_if_new(entry);
    let d0 = dist(entry);
    candidates.push(std::cmp::Reverse(DistNode {
        dist: d0,
        id: entry,
    }));
    results.push(DistNode {
        dist: d0,
        id: entry,
    });

    while let Some(std::cmp::Reverse(DistNode { dist: d_c, id: c })) = candidates.pop() {
        // Converged: the closest unexplored candidate is already farther than
        // our worst result and the beam is full — nothing left can improve it.
        let worst = results.peek().map(|d| d.dist).unwrap_or(f32::INFINITY);
        if results.len() == ef && d_c > worst {
            break;
        }
        for &nb in adjacency[c as usize].as_slice() {
            let nbi = nb as usize;
            if nbi >= n || !scratch.mark_if_new(nb) {
                continue;
            }
            let d_n = dist(nb);
            // Admit if the beam isn't full, or this beats the current farthest.
            if results.len() < ef || d_n < worst {
                candidates.push(std::cmp::Reverse(DistNode { dist: d_n, id: nb }));
                results.push(DistNode { dist: d_n, id: nb });
                if results.len() > ef {
                    results.pop(); // evict the (now) farthest
                }
            }
        }
    }

    let mut out: Vec<Neighbor> = results
        .into_iter()
        .map(|d| Neighbor {
            id: d.id,
            dist: d.dist,
        })
        .collect();
    out.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
    out.truncate(k);
    out
}

/// Member-to-member / query search over a finished graph. See [`beam_search`].
pub fn search_with<S: VectorSpace>(
    _space: &S,
    graph: &VamanaGraph,
    k: usize,
    ef: usize,
    dist: impl Fn(u32) -> f32,
) -> Vec<Neighbor> {
    beam_search(&graph.adjacency, graph.len(), graph.entry, k, ef, dist)
}

/// Search for the `k` nearest members to `query`. `ef` (beam width) trades
/// recall for latency; `ef ≥ k` is enforced internally.
pub fn search<S: VectorSpace>(
    space: &S,
    graph: &VamanaGraph,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Vec<Neighbor> {
    search_with(space, graph, k, ef, |id| space.query_distance(query, id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::F32CosineSpace;

    /// A tiny ring + centre graph to exercise expansion + convergence.
    fn ring_space() -> F32CosineSpace {
        F32CosineSpace::new(vec![
            vec![1.0, 0.0],  // 0
            vec![0.0, 1.0],  // 1
            vec![-1.0, 0.0], // 2
            vec![0.0, -1.0], // 3
        ])
    }

    #[test]
    fn empty_graph_returns_empty() {
        let s = F32CosineSpace::new(vec![]);
        let g = VamanaGraph::new(vec![], 0);
        assert!(search(&s, &g, &[1.0, 0.0], 3, 4).is_empty());
    }

    #[test]
    fn k_zero_returns_empty() {
        let s = ring_space();
        let g = VamanaGraph::new(vec![vec![], vec![], vec![], vec![]], 0);
        assert!(search(&s, &g, &[1.0, 0.0], 0, 4).is_empty());
    }

    #[test]
    fn nearest_of_query_is_returned_first() {
        // Fully-connected 4-node graph; query along +x → node 0 is nearest.
        let s = ring_space();
        let full = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];
        let g = VamanaGraph::new(full, 0);
        let out = search(&s, &g, &[1.0, 0.0], 1, 4);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 0);
    }

    #[test]
    fn search_reaches_distant_node_via_edges() {
        // A *path* graph 0-1-2-3 with entry at 0. Query near node 3 must still
        // reach it by walking the chain (tests that expansion actually happens).
        let s = ring_space();
        let chain = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let g = VamanaGraph::new(chain, 0);
        let out = search(&s, &g, &[0.0, -1.0], 1, 4);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 3, "must walk the chain to reach node 3");
    }

    #[test]
    fn out_of_range_entry_returns_empty_not_panic() {
        // A graph decoded from a corrupt/foreign body could carry an entry that
        // is out of range for its adjacency. Search must return empty, not panic
        // on `visited[entry]`.
        let s = ring_space();
        let g = VamanaGraph {
            adjacency: vec![vec![1], vec![0], vec![3], vec![2]],
            entry: 99,
        };
        assert!(search(&s, &g, &[1.0, 0.0], 2, 4).is_empty());
    }

    #[test]
    fn results_sorted_ascending_and_trimmed() {
        let s = ring_space();
        let full = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];
        let g = VamanaGraph::new(full, 0);
        let out = search(&s, &g, &[1.0, 0.0], 3, 4);
        assert_eq!(out.len(), 3);
        for w in out.windows(2) {
            assert!(w[0].dist <= w[1].dist + 1e-6, "not sorted: {:?}", out);
        }
    }

    #[test]
    fn sparse_scratch_tracks_visits_not_corpus_size() {
        // This models a tiny beam over a very large corpus without allocating
        // that corpus. The old `vec![false; n]` path reserved ten million
        // entries before touching its first node.
        let mut scratch = BeamScratch::sparse(4);
        scratch.begin(10_000_000);
        assert!(scratch.mark_if_new(9_999_999));
        assert!(!scratch.mark_if_new(9_999_999));
        assert_eq!(scratch.resident_marks(), 1);
    }

    #[test]
    fn dense_epoch_scratch_resets_without_reallocating() {
        let full = vec![vec![1, 2, 3], vec![0, 2, 3], vec![0, 1, 3], vec![0, 1, 2]];
        let mut scratch = BeamScratch::dense(full.len());
        let allocation = scratch.dense_allocation_ptr();

        // First search visits the whole connected graph.
        let first =
            beam_search_with_scratch(&full, full.len(), 0, 1, 4, |id| id as f32, &mut scratch);
        assert_eq!(first[0].id, 0);

        // A fresh epoch must make nodes visited above eligible again. Starting
        // from 3 still reaches 0, the best node; stale marks would leave 3.
        let second =
            beam_search_with_scratch(&full, full.len(), 3, 1, 4, |id| id as f32, &mut scratch);
        assert_eq!(second[0].id, 0);
        assert_eq!(scratch.dense_allocation_ptr(), allocation);
    }
}
