//! Leapfrog triejoin primitives (RFC-024).
//!
//! Two pieces:
//!
//! - [`OrdIterator`], the cursor abstraction every input list exposes,
//!   plus [`SortedSliceIter`] which adapts a `&[NodeId]` produced by
//!   [`crate::exec::walker`] from a sorted CSR partner slice (or its
//!   memtable-merged equivalent, see `Snapshot::sorted_partners`).
//!
//! - [`LeapfrogIntersect`], the classical Veldhuizen 2014 algorithm
//!   over a `Vec<OrdIterator>`. It emits the keys present in every
//!   input list in ascending order, taking `O(k log d)` work per
//!   emitted key where `k` is the number of inputs and `d` is the
//!   gap jumped by the longest seek.
//!
//! The crate-level executor in [`crate::exec::walker`] plugs these
//! into `execute_multiway_join_factor`: one [`LeapfrogIntersect`]
//! per trie level, each fed by [`SortedSliceIter`]s over the partner
//! lists of variables already bound at that level.

use namidb_core::id::NodeId;

/// A monotonically ascending cursor over a multiset of `NodeId`s.
///
/// Implementations have to honour two invariants:
///
/// - [`key`](Self::key) returns the current key or `None` at end.
///   Once `None`, [`at_end`](Self::at_end) is `true` and stays `true`.
/// - [`seek`](Self::seek) advances the cursor to the first key `>=
///   target`. If no such key exists the iterator finishes.
pub trait OrdIterator {
    fn key(&self) -> Option<NodeId>;
    fn next(&mut self);
    fn seek(&mut self, target: NodeId);
    fn at_end(&self) -> bool;
}

/// Cursor over an externally owned, sorted slice of `NodeId`s.
///
/// The slice has to be sorted ascending; the storage layer guarantees
/// that for both the raw CSR `EdgeSlice::partners` and the memtable-
/// merged `Snapshot::sorted_partners` output, so [`LeapfrogIntersect`]
/// can rely on it without re-sorting.
pub struct SortedSliceIter<'a> {
    partners: &'a [NodeId],
    cursor: usize,
}

impl<'a> SortedSliceIter<'a> {
    pub fn new(partners: &'a [NodeId]) -> Self {
        Self {
            partners,
            cursor: 0,
        }
    }
}

impl<'a> OrdIterator for SortedSliceIter<'a> {
    fn key(&self) -> Option<NodeId> {
        self.partners.get(self.cursor).copied()
    }

    fn next(&mut self) {
        if self.cursor < self.partners.len() {
            self.cursor += 1;
        }
    }

    fn seek(&mut self, target: NodeId) {
        if self.cursor >= self.partners.len() {
            return;
        }
        if self.partners[self.cursor] >= target {
            return;
        }
        // Exponential probe to find a window `[lo, hi)` that contains the
        // first key >= target. Cost is `O(log d)` where `d` is the gap
        // jumped, not `O(log n)`.
        let mut step = 1usize;
        let mut lo = self.cursor;
        let mut hi = self.cursor.saturating_add(step);
        while hi < self.partners.len() && self.partners[hi] < target {
            lo = hi;
            step = step.saturating_mul(2);
            hi = self.cursor.saturating_add(step);
        }
        let hi = hi.min(self.partners.len());
        let offset = self.partners[lo..hi].partition_point(|x| *x < target);
        self.cursor = lo + offset;
    }

    fn at_end(&self) -> bool {
        self.cursor >= self.partners.len()
    }
}

/// Multi-way intersection of `k` ascending `OrdIterator`s.
///
/// Usage:
///
/// ```ignore
/// let mut lf = LeapfrogIntersect::new(vec![it_a, it_b, it_c]);
/// while let Some(k) = lf.key() {
///     handle(k);
///     lf.next();
/// }
/// ```
///
/// Construction sorts the iterators by their current key, then runs
/// the search loop once so [`key`](Self::key) is immediately the first
/// element of the intersection (or `None` if the intersection is
/// empty). Each [`next`](Self::next) advances the rotating iterator
/// and re-runs the search loop.
pub struct LeapfrogIntersect<I: OrdIterator> {
    iters: Vec<I>,
    p: usize,
    finished: bool,
    current: Option<NodeId>,
}

impl<I: OrdIterator> LeapfrogIntersect<I> {
    pub fn new(mut iters: Vec<I>) -> Self {
        if iters.is_empty() || iters.iter().any(|i| i.at_end()) {
            return Self {
                iters,
                p: 0,
                finished: true,
                current: None,
            };
        }
        // Sort iterators by their starting key. The unwrap is safe: we
        // checked `at_end` above, so every iterator has a key.
        iters.sort_by_key(|i| i.key().unwrap());
        let mut this = Self {
            iters,
            p: 0,
            finished: false,
            current: None,
        };
        this.search();
        this
    }

    pub fn key(&self) -> Option<NodeId> {
        if self.finished {
            None
        } else {
            self.current
        }
    }

    pub fn next(&mut self) {
        if self.finished {
            return;
        }
        self.iters[self.p].next();
        if self.iters[self.p].at_end() {
            self.finished = true;
            self.current = None;
            return;
        }
        self.p = (self.p + 1) % self.iters.len();
        self.search();
    }

    fn search(&mut self) {
        let k = self.iters.len();
        loop {
            let cur = match self.iters[self.p].key() {
                Some(v) => v,
                None => {
                    self.finished = true;
                    self.current = None;
                    return;
                }
            };
            let prev = (self.p + k - 1) % k;
            let max = match self.iters[prev].key() {
                Some(v) => v,
                None => {
                    self.finished = true;
                    self.current = None;
                    return;
                }
            };
            if cur == max {
                self.current = Some(cur);
                return;
            }
            self.iters[self.p].seek(max);
            if self.iters[self.p].at_end() {
                self.finished = true;
                self.current = None;
                return;
            }
            self.p = (self.p + 1) % k;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn nid(n: u64) -> NodeId {
        let mut bytes = [0u8; 16];
        bytes[8..].copy_from_slice(&n.to_be_bytes());
        NodeId::from_uuid(Uuid::from_bytes(bytes))
    }

    fn drain<I: OrdIterator>(mut lf: LeapfrogIntersect<I>) -> Vec<NodeId> {
        let mut out = Vec::new();
        while let Some(k) = lf.key() {
            out.push(k);
            lf.next();
        }
        out
    }

    #[test]
    fn single_iter_passthrough() {
        let xs = vec![nid(1), nid(3), nid(5)];
        let lf = LeapfrogIntersect::new(vec![SortedSliceIter::new(&xs)]);
        assert_eq!(drain(lf), xs);
    }

    #[test]
    fn two_way_partial_overlap() {
        let xs = vec![nid(1), nid(3), nid(5), nid(7)];
        let ys = vec![nid(2), nid(3), nid(7), nid(9)];
        let lf = LeapfrogIntersect::new(vec![SortedSliceIter::new(&xs), SortedSliceIter::new(&ys)]);
        assert_eq!(drain(lf), vec![nid(3), nid(7)]);
    }

    #[test]
    fn three_way_empty_intersection() {
        let a = vec![nid(1), nid(2), nid(3)];
        let b = vec![nid(4), nid(5), nid(6)];
        let c = vec![nid(7), nid(8), nid(9)];
        let lf = LeapfrogIntersect::new(vec![
            SortedSliceIter::new(&a),
            SortedSliceIter::new(&b),
            SortedSliceIter::new(&c),
        ]);
        assert!(drain(lf).is_empty());
    }

    #[test]
    fn seek_beyond_end_marks_at_end() {
        let xs = vec![nid(1), nid(3), nid(5)];
        let mut it = SortedSliceIter::new(&xs);
        it.seek(nid(100));
        assert!(it.at_end());
        assert_eq!(it.key(), None);
    }

    #[test]
    fn seek_to_present_value_stops_at_hit() {
        let xs = vec![nid(1), nid(3), nid(5), nid(7), nid(9)];
        let mut it = SortedSliceIter::new(&xs);
        it.seek(nid(5));
        assert_eq!(it.key(), Some(nid(5)));
    }

    #[test]
    fn three_way_identical_lists() {
        let xs = vec![nid(1), nid(2), nid(3)];
        let ys = xs.clone();
        let zs = xs.clone();
        let lf = LeapfrogIntersect::new(vec![
            SortedSliceIter::new(&xs),
            SortedSliceIter::new(&ys),
            SortedSliceIter::new(&zs),
        ]);
        assert_eq!(drain(lf), xs);
    }

    #[test]
    fn empty_input_list_finishes_immediately() {
        let xs: Vec<NodeId> = Vec::new();
        let ys = vec![nid(1), nid(2)];
        let lf = LeapfrogIntersect::new(vec![SortedSliceIter::new(&xs), SortedSliceIter::new(&ys)]);
        assert!(drain(lf).is_empty());
    }

    #[test]
    fn exponential_seek_jumps_far_keys() {
        let xs: Vec<NodeId> = (0u64..1000).map(nid).collect();
        let ys = vec![nid(0), nid(999)];
        let lf = LeapfrogIntersect::new(vec![SortedSliceIter::new(&xs), SortedSliceIter::new(&ys)]);
        assert_eq!(drain(lf), ys);
    }
}
