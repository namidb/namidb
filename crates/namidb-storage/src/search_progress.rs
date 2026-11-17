//! Process-wide state of search index generations, for `SHOW INDEXES` and
//! `/metrics`.
//!
//! Two things are tracked:
//!
//! - **Base builds in flight**, keyed by the Search-LSM generation they
//!   build. A build registers itself with the number of Nodes rows it will
//!   scan and advances as it scans them; the registration ends when the build
//!   finishes or fails. A Building generation's progress is read from here.
//! - **Generation states per published snapshot cell**: how many vector and
//!   full-text indexes each open namespace serves natively (online) and how
//!   many still fall back to a scan (building). The server's snapshot cell
//!   records it on every publish and removes it when the cell drops.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use uuid::Uuid;

use crate::manifest::Manifest;
use crate::search_lsm::{SearchLsmKind, SearchLsmStatus};

/// Share of `populationPercent` the corpus scan accounts for; the rest is
/// the graph or postings build and the install that follow it.
const SCAN_SHARE: f64 = 0.9;

#[derive(Debug, Default)]
struct BuildProgress {
    scanned: AtomicU64,
    total: u64,
    scan_done: std::sync::atomic::AtomicBool,
}

fn builds() -> &'static Mutex<HashMap<Uuid, Arc<BuildProgress>>> {
    static BUILDS: OnceLock<Mutex<HashMap<Uuid, Arc<BuildProgress>>>> = OnceLock::new();
    BUILDS.get_or_init(Default::default)
}

/// One registered base build. Dropping it ends the registration.
#[derive(Debug)]
pub struct BuildGuard {
    generations: Vec<Uuid>,
    progress: Arc<BuildProgress>,
}

impl BuildGuard {
    /// Count `rows` more scanned Nodes rows.
    pub fn scanned(&self, rows: u64) {
        self.progress.scanned.fetch_add(rows, Ordering::Relaxed);
    }

    /// The corpus scan is over; the build's finishing phase runs next.
    pub fn scan_finished(&self) {
        self.progress.scan_done.store(true, Ordering::Relaxed);
    }
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        let mut map = builds().lock().unwrap_or_else(|poison| poison.into_inner());
        for generation in &self.generations {
            if map
                .get(generation)
                .is_some_and(|progress| Arc::ptr_eq(progress, &self.progress))
            {
                map.remove(generation);
            }
        }
    }
}

/// Register a base build of `generations` that will scan `total_rows` Nodes
/// rows.
pub fn begin_build(generations: impl IntoIterator<Item = Uuid>, total_rows: u64) -> BuildGuard {
    let progress = Arc::new(BuildProgress {
        total: total_rows,
        ..BuildProgress::default()
    });
    let generations = generations.into_iter().collect::<Vec<_>>();
    let mut map = builds().lock().unwrap_or_else(|poison| poison.into_inner());
    for generation in &generations {
        map.insert(*generation, Arc::clone(&progress));
    }
    BuildGuard {
        generations,
        progress,
    }
}

/// Percent complete (0..100) of the base build running for `generation`, or
/// `None` when no build of it is in flight in this process.
pub fn build_percent(generation: Uuid) -> Option<f64> {
    let map = builds().lock().unwrap_or_else(|poison| poison.into_inner());
    let progress = map.get(&generation)?;
    if progress.scan_done.load(Ordering::Relaxed) {
        return Some(SCAN_SHARE * 100.0);
    }
    let scanned = progress.scanned.load(Ordering::Relaxed) as f64;
    let total = progress.total.max(1) as f64;
    Some((scanned / total).min(1.0) * SCAN_SHARE * 100.0)
}

/// Number of base builds in flight in this process.
pub fn builds_in_flight() -> usize {
    let map = builds().lock().unwrap_or_else(|poison| poison.into_inner());
    let mut distinct = map.values().map(Arc::as_ptr).collect::<Vec<_>>();
    distinct.sort_unstable();
    distinct.dedup();
    distinct.len()
}

/// Serving state of one vector or full-text index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexState {
    /// The index answers searches natively.
    Online,
    /// The index is created but a search falls back to a scan until its
    /// generation activates.
    Building,
}

impl IndexState {
    pub fn as_str(self) -> &'static str {
        match self {
            IndexState::Online => "ONLINE",
            IndexState::Building => "BUILDING",
        }
    }
}

/// The serving state of index `name` of `kind` in `manifest`, from the same
/// read plan the search route uses, and its percent complete: 100 when
/// online, otherwise the in-flight build's progress (0 when none runs yet).
pub fn index_state(manifest: &Manifest, kind: SearchLsmKind, name: &str) -> (IndexState, f64) {
    use crate::search_lsm::{select_search_read_plan, SearchReadPlan};
    match select_search_read_plan(manifest, kind, name) {
        SearchReadPlan::FlatFallback(_) => {
            let percent = manifest
                .search_lsm
                .iter()
                .find(|state| state.kind == kind && state.index_name == name)
                .and_then(|state| build_percent(state.generation_id))
                .unwrap_or(0.0);
            (IndexState::Building, percent)
        }
        _ => (IndexState::Online, 100.0),
    }
}

/// Counts of indexes by kind and state in one namespace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexStateCounts {
    pub vector_online: u64,
    pub vector_building: u64,
    pub text_online: u64,
    pub text_building: u64,
}

impl IndexStateCounts {
    /// Cheap per-publish classification: a generation that is Active serves
    /// natively; one that is Building, or an index without a generation, does
    /// not. (`SHOW INDEXES` applies the full read plan.)
    pub fn of(manifest: &Manifest) -> Self {
        let active = |kind: SearchLsmKind, name: &str| {
            manifest.search_lsm.iter().any(|state| {
                state.kind == kind
                    && state.index_name == name
                    && state.status == SearchLsmStatus::Active
            })
        };
        let mut counts = Self::default();
        for descriptor in &manifest.vector_indexes {
            if active(SearchLsmKind::Vector, &descriptor.name) {
                counts.vector_online += 1;
            } else {
                counts.vector_building += 1;
            }
        }
        for descriptor in &manifest.text_indexes {
            if active(SearchLsmKind::Text, &descriptor.name) {
                counts.text_online += 1;
            } else {
                counts.text_building += 1;
            }
        }
        counts
    }

    fn add(&mut self, other: &Self) {
        self.vector_online += other.vector_online;
        self.vector_building += other.vector_building;
        self.text_online += other.text_online;
        self.text_building += other.text_building;
    }
}

fn cells() -> &'static Mutex<HashMap<u64, IndexStateCounts>> {
    static CELLS: OnceLock<Mutex<HashMap<u64, IndexStateCounts>>> = OnceLock::new();
    CELLS.get_or_init(Default::default)
}

/// A process-unique id for a snapshot cell.
pub(crate) fn next_cell_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn record_cell(cell: u64, manifest: &Manifest) {
    let counts = IndexStateCounts::of(manifest);
    let mut map = cells().lock().unwrap_or_else(|poison| poison.into_inner());
    if counts == IndexStateCounts::default() {
        map.remove(&cell);
    } else {
        map.insert(cell, counts);
    }
}

pub(crate) fn forget_cell(cell: u64) {
    cells()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .remove(&cell);
}

/// Index state counts summed over every open namespace in this process.
pub fn index_state_counts() -> IndexStateCounts {
    let map = cells().lock().unwrap_or_else(|poison| poison.into_inner());
    let mut total = IndexStateCounts::default();
    for counts in map.values() {
        total.add(counts);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_build_reports_progress_until_it_drops() {
        let generation = Uuid::now_v7();
        assert_eq!(build_percent(generation), None);
        let guard = begin_build([generation], 200);
        assert_eq!(build_percent(generation), Some(0.0));
        guard.scanned(100);
        let half = build_percent(generation).unwrap();
        assert!((half - 45.0).abs() < 1e-9, "{half}");
        guard.scanned(1_000);
        assert_eq!(build_percent(generation), Some(90.0));
        guard.scan_finished();
        assert_eq!(build_percent(generation), Some(90.0));
        assert!(builds_in_flight() >= 1);
        drop(guard);
        assert_eq!(build_percent(generation), None);
    }
}
