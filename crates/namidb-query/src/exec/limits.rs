//! Read-query execution guards: a wall-clock timeout and a row cap.
//!
//! The deadline rides a tokio task-local in [`namidb_storage::cancel`] so the
//! storage crate's CPU-bound SST decode and merge loops can probe it too and
//! abort a single long operator mid-flight (cooperative cancellation), not
//! only at the query operator boundaries here. The row cap is a query-local
//! concern on a separate task-local. The regular query path (no guard in
//! scope: tests, the write executor, profiling) keeps its baseline cost.
//!
//! - The deadline is checked at operator boundaries, inside the long scan /
//!   expand loops, and inside the storage decode loops. Operators whose loops
//!   do pure CPU work over already-materialised rows (cross product, joins,
//!   filter, projection, aggregation, sort, DISTINCT, factor
//!   materialisation) probe it through an amortised [`DeadlineProbe`], so a
//!   runaway `MATCH (a), (b)` honours the timeout and an admin cancel
//!   mid-operator instead of only at the next operator boundary.
//! - The row cap bounds the rows any single operator materialises: the
//!   executor aborts a query whose operator output would exceed it, plus a
//!   fast pre-check on the multiplicative `CrossProduct` and a fast-fail
//!   inside the `Expand` accumulation loop so a runaway never fully
//!   materialises.
//!
//! A row cap counts rows, not bytes: the working-memory budget in
//! [`super::memory`] bounds wide rows and operator state under the cap, and
//! is scoped separately with [`super::memory::with_query_memory`].
//!
//! The server scopes both from its configuration (env `NAMIDB_QUERY_TIMEOUT`
//! and `NAMIDB_QUERY_ROW_CAP`). Writes carry only a deadline (no row cap):
//! the write executor scopes it through [`with_limits`] in
//! `execute_write_with_deadline` and probes it in its per-row loops, so a
//! runaway statement aborts before commit. The bare `execute_write` and an
//! open Bolt transaction's own idle timeout remain separate, unrelated
//! bounds.

use std::cmp::Ordering;
use std::future::Future;
use std::time::Instant;

use super::walker::ExecError;

tokio::task_local! {
    /// Maximum rows a single operator may materialise, when a cap is in scope.
    static ROW_CAP: usize;
}

/// Run `fut` with an optional deadline and row cap scoped on the current
/// task. The deadline is installed on [`namidb_storage::cancel`]'s task-local
/// so storage decode loops observe it; the row cap on a query-local one. When
/// both are `None` the future runs unguarded (no overhead).
pub(crate) async fn with_limits<F: Future>(
    deadline: Option<Instant>,
    row_cap: Option<usize>,
    fut: F,
) -> F::Output {
    let capped = async move {
        match row_cap {
            Some(cap) => ROW_CAP.scope(cap, fut).await,
            None => fut.await,
        }
    };
    namidb_storage::cancel::with_deadline(deadline, capped).await
}

/// `Err(ExecError::Timeout)` when a deadline is in scope and has passed,
/// `Ok(())` otherwise. Delegates to the shared storage task-local so the
/// query and storage layers observe the same deadline. A no-op when none is
/// scoped.
#[inline]
pub(crate) fn check_deadline() -> Result<(), ExecError> {
    use namidb_storage::cancel::Interrupt;
    match namidb_storage::cancel::interrupted() {
        None => Ok(()),
        Some(Interrupt::Timeout) => Err(ExecError::Timeout),
        Some(Interrupt::Cancelled) => Err(ExecError::Cancelled),
    }
}

/// `Err(ExecError::RowCap)` when a row cap is in scope and `len` exceeds it,
/// `Ok(())` otherwise. `len` is the row count an operator produced (or is
/// about to produce). A no-op when no cap is scoped.
/// The row cap in scope, if any — for expression-level materialization
/// guards (a `range()` longer than the cap is rejected before it allocates).
#[inline]
pub(crate) fn scoped_row_cap() -> Option<usize> {
    ROW_CAP.try_with(|cap| *cap).ok()
}

#[inline]
pub(crate) fn check_row_cap(len: usize) -> Result<(), ExecError> {
    ROW_CAP
        .try_with(|cap| {
            if len > *cap {
                Err(ExecError::RowCap(*cap))
            } else {
                Ok(())
            }
        })
        .unwrap_or(Ok(()))
}

/// Units of work between two probes of a [`DeadlineProbe`]. A probe reads a
/// task-local, an atomic and the clock (tens of nanoseconds); at this stride
/// that is noise next to the per-row work of every loop that uses it, while
/// still firing within well under a millisecond of the deadline.
pub(crate) const PROBE_STRIDE: u32 = 1024;

/// Amortised cooperative-cancellation probe for tight CPU loops that produce
/// no I/O (and so never reach a storage probe): call [`tick`](Self::tick)
/// once per unit of work and it checks the deadline / cancel flag every
/// [`PROBE_STRIDE`] ticks. Returns the existing `Timeout` / `Cancelled`
/// errors. Without a guard in scope a check is a task-local miss.
#[derive(Debug, Default)]
pub(crate) struct DeadlineProbe {
    ticks: u32,
}

impl DeadlineProbe {
    pub(crate) const fn new() -> Self {
        Self { ticks: 0 }
    }

    /// Count one unit of work; probe the guards every [`PROBE_STRIDE`].
    #[inline]
    pub(crate) fn tick(&mut self) -> Result<(), ExecError> {
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks % PROBE_STRIDE == 0 {
            check_deadline()
        } else {
            Ok(())
        }
    }
}

/// Elements sorted per run by [`sort_by_interruptible`] before the runs are
/// merged with probes between steps.
const SORT_RUN: usize = 1 << 14;

/// Stable sort that honours the deadline / cancel flag. `slice::sort_by`
/// cannot be interrupted (and a comparator that starts returning `Equal`
/// to bail out breaks its total-order contract), so a guarded sort of more
/// than one run sorts fixed-size runs with the standard sort, probing
/// between runs, then merges adjacent runs bottom-up with an amortised probe
/// per merged element. Ties keep the left run's element first, so the result
/// is identical to `sort_by`. Unguarded (no deadline, no cancel flag) or
/// small inputs take the plain `sort_by` with no overhead.
pub(crate) fn sort_by_interruptible<T>(
    items: &mut Vec<T>,
    cmp: impl Fn(&T, &T) -> Ordering,
) -> Result<(), ExecError> {
    check_deadline()?;
    if !sort_merges_runs(items.len()) {
        items.sort_by(&cmp);
        return Ok(());
    }
    let mut runs: Vec<Vec<T>> = Vec::with_capacity(items.len().div_ceil(SORT_RUN));
    let mut drain = items.drain(..);
    loop {
        let mut run: Vec<T> = drain.by_ref().take(SORT_RUN).collect();
        if run.is_empty() {
            break;
        }
        check_deadline()?;
        run.sort_by(&cmp);
        runs.push(run);
    }
    drop(drain);
    let mut probe = DeadlineProbe::new();
    while runs.len() > 1 {
        let mut merged = Vec::with_capacity(runs.len().div_ceil(2));
        let mut pairs = runs.into_iter();
        while let Some(left) = pairs.next() {
            match pairs.next() {
                Some(right) => merged.push(merge_runs(left, right, &cmp, &mut probe)?),
                None => merged.push(left),
            }
        }
        runs = merged;
    }
    *items = runs.pop().unwrap_or_default();
    Ok(())
}

/// Whether [`sort_by_interruptible`] takes its run-and-merge path for `len`
/// elements (and so briefly holds a second buffer of that many elements,
/// which callers metering working memory account for).
#[inline]
pub(crate) fn sort_merges_runs(len: usize) -> bool {
    len > SORT_RUN && namidb_storage::cancel::guarded()
}

/// Stable merge of two sorted runs (left wins ties), probing the guards.
fn merge_runs<T>(
    left: Vec<T>,
    right: Vec<T>,
    cmp: &impl Fn(&T, &T) -> Ordering,
    probe: &mut DeadlineProbe,
) -> Result<Vec<T>, ExecError> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut left = left.into_iter().peekable();
    let mut right = right.into_iter().peekable();
    loop {
        probe.tick()?;
        let take_left = match (left.peek(), right.peek()) {
            (Some(l), Some(r)) => cmp(l, r) != Ordering::Greater,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let next = if take_left { left.next() } else { right.next() };
        out.extend(next);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Duration;

    fn pseudo_random(n: usize) -> Vec<(u32, usize)> {
        // Few distinct keys so stability is observable through the index.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x % 97) as u32, i)
            })
            .collect()
    }

    #[tokio::test]
    async fn interruptible_sort_matches_the_stable_sort() {
        for n in [0, 1, 5, SORT_RUN, SORT_RUN + 1, 3 * SORT_RUN + 17] {
            let mut expected = pseudo_random(n);
            expected.sort_by(|a, b| a.0.cmp(&b.0));
            let later = Instant::now() + Duration::from_secs(600);
            let got = namidb_storage::cancel::with_deadline(Some(later), async {
                let mut items = pseudo_random(n);
                sort_by_interruptible(&mut items, |a, b| a.0.cmp(&b.0)).unwrap();
                items
            })
            .await;
            assert_eq!(got, expected, "n = {n}");
        }
        // Unguarded takes the plain sort.
        let mut items = pseudo_random(2 * SORT_RUN);
        sort_by_interruptible(&mut items, |a, b| a.0.cmp(&b.0)).unwrap();
        assert!(items.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[tokio::test]
    async fn interruptible_sort_stops_on_cancel() {
        let flag = Arc::new(AtomicBool::new(false));
        let err = namidb_storage::cancel::with_cancel_flag(flag.clone(), async {
            let mut items = pseudo_random(4 * SORT_RUN);
            let calls = std::cell::Cell::new(0usize);
            sort_by_interruptible(&mut items, |a, b| {
                calls.set(calls.get() + 1);
                if calls.get() == 10 {
                    flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                a.0.cmp(&b.0)
            })
            .unwrap_err()
        })
        .await;
        assert!(matches!(err, ExecError::Cancelled), "got {err:?}");
    }

    #[tokio::test]
    async fn probe_fires_within_one_stride() {
        let past = Instant::now() - Duration::from_millis(1);
        namidb_storage::cancel::with_deadline(Some(past), async {
            let mut probe = DeadlineProbe::new();
            let mut ok = 0u32;
            while probe.tick().is_ok() {
                ok += 1;
            }
            assert_eq!(ok, PROBE_STRIDE - 1);
        })
        .await;
        // No guard: never fires.
        let mut probe = DeadlineProbe::new();
        for _ in 0..(4 * PROBE_STRIDE) {
            probe.tick().unwrap();
        }
    }
}
