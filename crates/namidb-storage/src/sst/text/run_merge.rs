//! Bounded fan-in k-way merge shared by the external full-text sorters.
//!
//! Spilled runs are kept as a flat list instead of binary-counter levels. A
//! build merges them once, at finish, with a heap over at most
//! [`MERGE_FAN_IN`] readers that streams records straight into the consumer.
//! Each spilled byte is therefore written once (to its initial run) and read
//! once. Only when there are more runs than the fan-in are groups of the
//! smallest runs folded into intermediate runs first, so open files and
//! reader buffers stay bounded however large the corpus grows.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{Error, Result};

/// Upper bound on the runs read concurrently by one merge.
pub(super) const MERGE_FAN_IN: usize = 64;
/// Smallest read buffer a merge reader gets. A budget too small to give every
/// reader this much lowers the fan-in instead.
pub(super) const MIN_MERGE_READER_BUFFER_BYTES: usize = 4 * 1024;
/// Largest read buffer a merge reader gets; more buys no sequential I/O.
pub(super) const MAX_MERGE_READER_BUFFER_BYTES: usize = 1024 * 1024;
/// Pending runs, as a multiple of the fan-in, at which a build folds its
/// smallest runs while still ingesting. This bounds open spool files during
/// the build; below it every run waits for the single final merge.
const PENDING_RUNS_PER_FAN_IN: usize = 4;

/// Read-buffer budget of one merge and the fan-in it affords.
#[derive(Debug, Clone, Copy)]
pub(super) struct MergeBudget {
    reader_bytes: usize,
}

impl MergeBudget {
    /// `reader_bytes` is the memory the merge readers may share.
    pub(super) fn new(reader_bytes: usize) -> Self {
        Self { reader_bytes }
    }

    /// Runs one merge reads at once: [`MERGE_FAN_IN`], lowered so every
    /// reader keeps at least [`MIN_MERGE_READER_BUFFER_BYTES`], and never
    /// below two.
    pub(super) fn fan_in(self) -> usize {
        (self.reader_bytes / MIN_MERGE_READER_BUFFER_BYTES).clamp(2, MERGE_FAN_IN)
    }

    /// Read buffer for each of `readers` concurrently open runs. For
    /// `readers <= fan_in()` the total stays within the budget unless the
    /// budget itself is below the floor of two readers.
    pub(super) fn reader_buffer_bytes(self, readers: usize) -> usize {
        (self.reader_bytes / readers.max(1))
            .clamp(MIN_MERGE_READER_BUFFER_BYTES, MAX_MERGE_READER_BUFFER_BYTES)
    }

    /// Pending runs at which ingestion folds a group instead of waiting.
    pub(super) fn max_pending_runs(self) -> usize {
        self.fan_in() * PENDING_RUNS_PER_FAN_IN
    }

    /// Size of the next intermediate merge for `pending` runs, or `None` when
    /// the final merge can read them all directly. A full group removes
    /// `fan_in - 1` runs; the first, possibly partial, group is sized so the
    /// remaining runs fit the final merge exactly, which rewrites the fewest
    /// bytes.
    pub(super) fn next_group(self, pending: usize) -> Option<usize> {
        let fan_in = self.fan_in();
        (pending > fan_in).then(|| fan_in.min(pending - fan_in + 1))
    }
}

/// Remove the `count` shortest runs from `runs` (ties keep spill order).
pub(super) fn take_smallest<R>(runs: &mut Vec<R>, count: usize, len: impl Fn(&R) -> u64) -> Vec<R> {
    runs.sort_by_key(|run| len(run));
    runs.drain(..count.min(runs.len())).collect()
}

/// Record type of a sorted run.
pub(super) trait MergeRecord: Ord {
    /// True when two records collide on the sort key the format forbids
    /// repeating (it may be coarser than full equality).
    fn same_key(&self, other: &Self) -> bool;
}

/// One checksummed sorted run being read.
pub(super) trait SortedRunSource {
    type Record: MergeRecord;

    /// The next record, or `None` once the run is exhausted and verified.
    fn next_record(&mut self) -> Result<Option<Self::Record>>;

    /// True once the run's length and checksum were verified at its end.
    fn verified(&self) -> bool;
}

/// Format-specific invariant messages of a merge.
pub(super) struct MergeMessages {
    pub(super) duplicate: &'static str,
    pub(super) non_increasing: &'static str,
    pub(super) unverified: &'static str,
}

struct HeapEntry<T> {
    record: T,
    source: usize,
}

impl<T: Ord> Ord for HeapEntry<T> {
    // `BinaryHeap` is a max-heap: invert so the smallest record pops first,
    // the lower source index winning ties.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .record
            .cmp(&self.record)
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl<T: Ord> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T: Ord> Eq for HeapEntry<T> {}

/// Heap-based merge of sorted runs into one strictly increasing stream.
///
/// It holds one current record per run plus the last record it returned, and
/// rejects a repeated key or a non-increasing record. After the last record
/// every run has been read to its end and checksum-verified.
pub(super) struct KWayMerge<S: SortedRunSource> {
    sources: Vec<S>,
    heap: BinaryHeap<HeapEntry<S::Record>>,
    current: Option<S::Record>,
    messages: &'static MergeMessages,
}

impl<S: SortedRunSource> KWayMerge<S> {
    pub(super) fn new(mut sources: Vec<S>, messages: &'static MergeMessages) -> Result<Self> {
        let mut heap = BinaryHeap::with_capacity(sources.len());
        for (source, reader) in sources.iter_mut().enumerate() {
            if let Some(record) = reader.next_record()? {
                heap.push(HeapEntry { record, source });
            }
        }
        Ok(Self {
            sources,
            heap,
            current: None,
            messages,
        })
    }

    /// The next record in merged order, borrowed until the following call.
    pub(super) fn next(&mut self) -> Result<Option<&S::Record>> {
        crate::background::pace();
        let Some(HeapEntry { record, source }) = self.heap.pop() else {
            if !self.sources.iter().all(SortedRunSource::verified) {
                return Err(Error::invariant(self.messages.unverified));
            }
            return Ok(None);
        };
        if let Some(next) = self.sources[source].next_record()? {
            self.heap.push(HeapEntry {
                record: next,
                source,
            });
        }
        if let Some(previous) = self.current.as_ref() {
            if previous.same_key(&record) {
                return Err(Error::invariant(self.messages.duplicate));
            }
            if previous >= &record {
                return Err(Error::invariant(self.messages.non_increasing));
            }
        }
        self.current = Some(record);
        Ok(self.current.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_bounds_fan_in_buffers_and_groups() {
        let roomy = MergeBudget::new(16 * 1024 * 1024);
        assert_eq!(roomy.fan_in(), MERGE_FAN_IN);
        assert_eq!(roomy.reader_buffer_bytes(64), 256 * 1024);
        assert_eq!(roomy.reader_buffer_bytes(1), MAX_MERGE_READER_BUFFER_BYTES);
        assert_eq!(roomy.next_group(64), None);
        assert_eq!(roomy.next_group(65), Some(2));
        assert_eq!(roomy.next_group(100), Some(37));
        assert_eq!(roomy.next_group(1_000), Some(64));

        let small = MergeBudget::new(64 * 1024);
        assert_eq!(small.fan_in(), 16);
        assert!(small.fan_in() * small.reader_buffer_bytes(small.fan_in()) <= 64 * 1024);

        let tiny = MergeBudget::new(96);
        assert_eq!(tiny.fan_in(), 2);
        assert_eq!(tiny.reader_buffer_bytes(2), MIN_MERGE_READER_BUFFER_BYTES);
        assert_eq!(tiny.next_group(3), Some(2));
    }

    #[test]
    fn take_smallest_prefers_short_runs_in_spill_order() {
        let mut runs = vec![(0, 9u64), (1, 3), (2, 3), (3, 1), (4, 7)];
        let taken = take_smallest(&mut runs, 3, |run| run.1);
        assert_eq!(taken, vec![(3, 1), (1, 3), (2, 3)]);
        assert_eq!(runs, vec![(4, 7), (0, 9)]);
    }
}
