//! Process-wide transient workspace governor for object-native search.
//!
//! Search accelerators must bound not only retained caches but also compressed
//! range responses, decompressed pages, decoded postings/candidates, and
//! result heaps. This pool serialises or throttles concurrent queries against
//! one byte budget without changing their requested recall or semantics.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::Error;

/// Exact-byte process-wide transient search-workspace setting.
pub(crate) const SEARCH_WORKSPACE_MAX_BYTES_ENV: &str = "NAMIDB_SEARCH_WORKSPACE_MAX_BYTES";
/// Exact-byte result cap used by APIs whose `k = None` contract materialises
/// every hit.
pub(crate) const SEARCH_MAX_RESULT_BYTES_ENV: &str = "NAMIDB_SEARCH_MAX_RESULT_BYTES";
/// Maximum UTF-8 bytes across the configured string fields of one document
/// analysed by the exact BM25 fallback.
pub(crate) const BM25_MAX_DOCUMENT_BYTES_ENV: &str = "NAMIDB_BM25_MAX_DOCUMENT_BYTES";
/// Default transient workspace shared by all concurrent searches: 256 MiB.
pub(crate) const DEFAULT_SEARCH_WORKSPACE_MAX_BYTES: usize = 256 * 1024 * 1024;
/// Default materialised-result allowance for an unbounded search: 64 MiB.
pub(crate) const DEFAULT_SEARCH_MAX_RESULT_BYTES: usize = 64 * 1024 * 1024;
/// Default exact-fallback document ceiling: 1 MiB of configured string fields.
pub(crate) const DEFAULT_BM25_MAX_DOCUMENT_BYTES: usize = 1024 * 1024;
/// Conservative transient-workspace multiplier for streaming one BM25
/// document.
///
/// One UTF-8 input byte can account for at most one emitted token. The dominant
/// adversarial shape is therefore thousands of unique, very short terms: allow
/// 128 bytes per hash-table entry (owned `String`, count, buckets, allocator
/// slack, and a resize peak), then another 32 bytes per input byte for Unicode
/// lowercase expansion, the rolling tokenizer scratch, and allocator slack.
pub(crate) const BM25_DOCUMENT_WORKSPACE_MULTIPLIER: usize = 160;
#[cfg(test)]
const BM25_UNIQUE_TERM_ENTRY_ESTIMATE_BYTES: usize = 128;
/// Conservative retained/transient allowance for one materialised text hit.
///
/// The query layer may temporarily hold a ranking entry, an output tuple, a
/// deduplication-map entry, and allocator/hash-table overhead for the same
/// logical hit. Keeping this estimate central makes the range-readable,
/// legacy, and exact-fallback paths enforce one contract.
pub(crate) const MATERIALISED_TEXT_RESULT_BYTES_PER_HIT: usize = 320;
/// Semaphore accounting granularity. Every reservation rounds upward.
pub(crate) const SEARCH_WORKSPACE_UNIT_BYTES: usize = 4 * 1024;

/// Snapshot of process-wide query-workspace counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchWorkspaceMetrics {
    /// Effective byte capacity after rounding down to 4-KiB units.
    pub capacity_bytes: usize,
    /// Bytes currently reserved by active query phases.
    pub reserved_bytes: usize,
    /// Process-lifetime high-water mark of reserved bytes.
    pub peak_reserved_bytes: usize,
    /// Reservations granted, including zero-byte reservations.
    pub successful_reservations: u64,
    /// Reservations that observed insufficient immediately available permits
    /// and therefore had to await the fair semaphore.
    pub contended_reservations: u64,
    /// Requests rejected because one query's estimate exceeded total capacity.
    pub rejected_reservations: u64,
}

/// A byte-accounted, fair Tokio semaphore shared by search implementations.
#[derive(Debug)]
pub(crate) struct SearchWorkspacePool {
    semaphore: Arc<Semaphore>,
    capacity_bytes: usize,
    reserved_bytes: Arc<AtomicUsize>,
    peak_reserved_bytes: Arc<AtomicUsize>,
    successful_reservations: Arc<AtomicU64>,
    contended_reservations: Arc<AtomicU64>,
    rejected_reservations: Arc<AtomicU64>,
}

static SHARED_SEARCH_WORKSPACE: OnceLock<SearchWorkspacePool> = OnceLock::new();

impl SearchWorkspacePool {
    pub(crate) fn new(configured_capacity_bytes: usize) -> Self {
        // Round down so one semaphore unit can never make the effective
        // assignment exceed the operator's exact-byte ceiling.
        let units =
            (configured_capacity_bytes / SEARCH_WORKSPACE_UNIT_BYTES).min(u32::MAX as usize);
        let capacity_bytes = units.saturating_mul(SEARCH_WORKSPACE_UNIT_BYTES);
        Self {
            semaphore: Arc::new(Semaphore::new(units)),
            capacity_bytes,
            reserved_bytes: Arc::new(AtomicUsize::new(0)),
            peak_reserved_bytes: Arc::new(AtomicUsize::new(0)),
            successful_reservations: Arc::new(AtomicU64::new(0)),
            contended_reservations: Arc::new(AtomicU64::new(0)),
            rejected_reservations: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Reserve enough 4-KiB units for `required_bytes`.
    ///
    /// Requests larger than the entire pool fail immediately. Requests that
    /// individually fit wait fairly behind active queries and therefore never
    /// need to lower `k`, probing effort, or any other accuracy control.
    pub(crate) async fn reserve(
        &self,
        operation: &'static str,
        required_bytes: usize,
    ) -> Result<SearchWorkspaceReservation, Error> {
        let units = required_bytes
            .checked_add(SEARCH_WORKSPACE_UNIT_BYTES - 1)
            .map(|bytes| bytes / SEARCH_WORKSPACE_UNIT_BYTES)
            .filter(|units| *units <= u32::MAX as usize);
        let Some(units) = units else {
            self.rejected_reservations.fetch_add(1, Ordering::Relaxed);
            return Err(self.exceeded(operation, required_bytes));
        };
        let reserved_bytes = units.saturating_mul(SEARCH_WORKSPACE_UNIT_BYTES);
        if reserved_bytes > self.capacity_bytes {
            self.rejected_reservations.fetch_add(1, Ordering::Relaxed);
            return Err(self.exceeded(operation, required_bytes));
        }

        if units > self.semaphore.available_permits() {
            self.contended_reservations.fetch_add(1, Ordering::Relaxed);
        }
        let permit = if units == 0 {
            None
        } else {
            Some(
                Arc::clone(&self.semaphore)
                    .acquire_many_owned(units as u32)
                    .await
                    .map_err(|_| {
                        Error::invariant("search workspace pool is unexpectedly closed")
                    })?,
            )
        };
        let previous = self
            .reserved_bytes
            .fetch_add(reserved_bytes, Ordering::AcqRel);
        update_peak(
            &self.peak_reserved_bytes,
            previous.saturating_add(reserved_bytes),
        );
        self.successful_reservations.fetch_add(1, Ordering::Relaxed);

        Ok(SearchWorkspaceReservation {
            permit,
            reserved_bytes,
            current: Arc::clone(&self.reserved_bytes),
        })
    }

    pub(crate) fn metrics(&self) -> SearchWorkspaceMetrics {
        SearchWorkspaceMetrics {
            capacity_bytes: self.capacity_bytes,
            reserved_bytes: self.reserved_bytes.load(Ordering::Acquire),
            peak_reserved_bytes: self.peak_reserved_bytes.load(Ordering::Acquire),
            successful_reservations: self.successful_reservations.load(Ordering::Relaxed),
            contended_reservations: self.contended_reservations.load(Ordering::Relaxed),
            rejected_reservations: self.rejected_reservations.load(Ordering::Relaxed),
        }
    }

    fn exceeded(&self, operation: &'static str, required_bytes: usize) -> Error {
        Error::QueryWorkspaceExceeded {
            operation,
            required_bytes,
            capacity_bytes: self.capacity_bytes,
        }
    }
}

/// RAII permit returned by [`SearchWorkspacePool::reserve`].
#[derive(Debug)]
pub struct SearchWorkspaceReservation {
    // Dropping this field returns semaphore permits.
    permit: Option<OwnedSemaphorePermit>,
    reserved_bytes: usize,
    current: Arc<AtomicUsize>,
}

impl Drop for SearchWorkspaceReservation {
    fn drop(&mut self) {
        self.current
            .fetch_sub(self.reserved_bytes, Ordering::AcqRel);
        // Keep the permit alive until after accounting has been decremented.
        self.permit.take();
    }
}

pub(crate) fn shared_search_workspace() -> &'static SearchWorkspacePool {
    SHARED_SEARCH_WORKSPACE.get_or_init(|| {
        let configured = env_bytes(
            SEARCH_WORKSPACE_MAX_BYTES_ENV,
            DEFAULT_SEARCH_WORKSPACE_MAX_BYTES,
        );
        let pool = SearchWorkspacePool::new(configured);
        tracing::info!(
            configured_bytes = configured,
            effective_bytes = pool.capacity_bytes(),
            unit_bytes = SEARCH_WORKSPACE_UNIT_BYTES,
            "resolved process-wide search workspace"
        );
        pool
    })
}

/// Return process-wide search-workspace counters if a search already
/// initialised the pool.
///
/// Metrics scrapes deliberately do not initialise the pool or sample its
/// environment configuration as a side effect.
pub fn search_workspace_metrics() -> Option<SearchWorkspaceMetrics> {
    SHARED_SEARCH_WORKSPACE
        .get()
        .map(SearchWorkspacePool::metrics)
}

/// Exact configured ceiling for APIs that materialise every search result.
pub fn search_max_result_bytes() -> usize {
    env_bytes(SEARCH_MAX_RESULT_BYTES_ENV, DEFAULT_SEARCH_MAX_RESULT_BYTES)
}

/// Maximum number of materialised text hits admitted by the process result
/// budget. Query-layer exact fallbacks use the same ceiling as persisted FTS
/// readers so changing execution paths cannot bypass the memory contract.
pub fn search_max_text_result_hits() -> usize {
    search_max_result_bytes() / MATERIALISED_TEXT_RESULT_BYTES_PER_HIT
}

/// Conservative retained/transient estimate for materialised text results.
pub fn estimated_text_result_bytes(hits: usize) -> usize {
    hits.saturating_mul(MATERIALISED_TEXT_RESULT_BYTES_PER_HIT)
}

/// UTF-8 document ceiling for the exact BM25 corpus fallback.
pub fn bm25_max_document_bytes() -> usize {
    env_bytes(BM25_MAX_DOCUMENT_BYTES_ENV, DEFAULT_BM25_MAX_DOCUMENT_BYTES)
}

/// Conservative peak workspace reserved for one streaming BM25 document.
pub fn estimated_bm25_document_workspace_bytes(document_bytes: usize) -> usize {
    document_bytes.saturating_mul(BM25_DOCUMENT_WORKSPACE_MULTIPLIER)
}

/// Acquire process-wide transient workspace for a search phase implemented
/// outside the storage crate (for example the exact Cypher fallback).
pub async fn reserve_search_workspace(
    operation: &'static str,
    required_bytes: usize,
) -> Result<SearchWorkspaceReservation, Error> {
    shared_search_workspace()
        .reserve(operation, required_bytes)
        .await
}

fn env_bytes(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!(name, value, %error, default, "ignoring invalid byte setting");
                default
            }
        },
        Err(std::env::VarError::NotPresent) => default,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(name, default, "ignoring non-UTF-8 byte setting");
            default
        }
    }
}

fn update_peak(peak: &AtomicUsize, value: usize) {
    let mut observed = peak.load(Ordering::Relaxed);
    while value > observed {
        match peak.compare_exchange_weak(observed, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => observed = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fmt::Write;

    #[tokio::test]
    async fn reservations_round_up_and_return_permits() {
        let pool = SearchWorkspacePool::new(3 * SEARCH_WORKSPACE_UNIT_BYTES);
        let reservation = pool.reserve("test search", 1).await.unwrap();
        assert_eq!(reservation.reserved_bytes, SEARCH_WORKSPACE_UNIT_BYTES);
        assert_eq!(pool.metrics().reserved_bytes, SEARCH_WORKSPACE_UNIT_BYTES);
        drop(reservation);
        assert_eq!(pool.metrics().reserved_bytes, 0);
    }

    #[tokio::test]
    async fn oversized_reservation_is_explicit_and_does_not_change_usage() {
        let pool = SearchWorkspacePool::new(SEARCH_WORKSPACE_UNIT_BYTES);
        let error = pool
            .reserve("exact vector search", SEARCH_WORKSPACE_UNIT_BYTES + 1)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            Error::QueryWorkspaceExceeded {
                operation: "exact vector search",
                required_bytes,
                capacity_bytes,
            } if required_bytes == SEARCH_WORKSPACE_UNIT_BYTES + 1
                && capacity_bytes == SEARCH_WORKSPACE_UNIT_BYTES
        ));
        let metrics = pool.metrics();
        assert_eq!(metrics.reserved_bytes, 0);
        assert_eq!(metrics.rejected_reservations, 1);
    }

    #[tokio::test]
    async fn concurrent_reservations_never_exceed_the_pool() {
        let pool = Arc::new(SearchWorkspacePool::new(2 * SEARCH_WORKSPACE_UNIT_BYTES));
        let first = pool
            .reserve("first", 2 * SEARCH_WORKSPACE_UNIT_BYTES)
            .await
            .unwrap();
        let waiting_pool = Arc::clone(&pool);
        let waiter = tokio::spawn(async move {
            waiting_pool
                .reserve("second", SEARCH_WORKSPACE_UNIT_BYTES)
                .await
                .unwrap()
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(first);
        let second = waiter.await.unwrap();
        assert_eq!(pool.metrics().peak_reserved_bytes, pool.capacity_bytes());
        drop(second);
        assert_eq!(pool.metrics().reserved_bytes, 0);
        assert_eq!(pool.metrics().contended_reservations, 1);
    }

    #[test]
    fn default_bm25_document_and_unbounded_results_fit_one_reservation() {
        let document = estimated_bm25_document_workspace_bytes(DEFAULT_BM25_MAX_DOCUMENT_BYTES);
        // The flat fallback retains one sentinel hit beyond the unbounded
        // result ceiling so it can report truncation rather than return it.
        let results =
            DEFAULT_SEARCH_MAX_RESULT_BYTES.saturating_add(MATERIALISED_TEXT_RESULT_BYTES_PER_HIT);
        assert!(
            document.saturating_add(results) <= DEFAULT_SEARCH_WORKSPACE_MAX_BYTES,
            "defaults must not reject every exact unbounded BM25 query"
        );
    }

    #[test]
    fn bm25_document_multiplier_covers_adversarial_unique_terms_and_cjk() {
        fn assert_model_fits(text: &str) {
            let tokens = crate::text::tokenize(text);
            let unique: HashSet<&str> = tokens.iter().map(String::as_str).collect();
            let unique_map_model = unique.iter().fold(0usize, |bytes, token| {
                bytes.saturating_add(
                    BM25_UNIQUE_TERM_ENTRY_ESTIMATE_BYTES.saturating_add(token.len()),
                )
            });
            let reservation = estimated_bm25_document_workspace_bytes(text.len());
            assert!(
                unique_map_model <= reservation,
                "{} unique tokens require modeled {unique_map_model} bytes, \
                 reservation is {reservation}",
                unique.len()
            );
        }

        // 4,095 distinct overlapping bigrams from one separator-free run.
        let cjk: String = (0x3400..0x4400)
            .map(|codepoint| char::from_u32(codepoint).unwrap())
            .collect();
        assert_model_fits(&cjk);

        // Thousands of small unique terms maximise map-entry overhead relative
        // to retained token bytes.
        let mut short_unique = String::new();
        for i in 0..8_192 {
            write!(&mut short_unique, "t{i:x} ").unwrap();
        }
        assert_model_fits(&short_unique);
    }
}
