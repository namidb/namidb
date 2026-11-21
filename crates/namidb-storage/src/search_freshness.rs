//! Bounds on serving a search index while unflushed writes touch its corpus.
//!
//! A full-text read merges the memtable's documents of the indexed label into
//! the index answer (scored from their memtable values, with the corpus
//! statistics corrected for them), so a write no longer turns the next
//! `search.bm25` into an `O(corpus)` scan. That merge costs `O(fresh
//! documents)` per query; past [`fresh_text_max_docs`] the read takes the
//! exact flat scan instead, counts it, logs it once per snapshot, and asks
//! the namespace's flush loop for an early flush, which folds the documents
//! into the index and restores the native route.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Default fresh-document bound of one full-text read.
pub const DEFAULT_FRESH_TEXT_MAX_DOCS: usize = 5_000;

/// Fresh documents a full-text read merges before it takes the flat scan
/// (`NAMIDB_SEARCH_FRESH_TEXT_MAX_DOCS`, default 5 000; 0 never merges).
pub fn fresh_text_max_docs() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NAMIDB_SEARCH_FRESH_TEXT_MAX_DOCS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_FRESH_TEXT_MAX_DOCS)
    })
}

fn flush_notify() -> &'static tokio::sync::Notify {
    static NOTIFY: OnceLock<tokio::sync::Notify> = OnceLock::new();
    NOTIFY.get_or_init(tokio::sync::Notify::new)
}

fn flush_requests() -> &'static Mutex<HashSet<String>> {
    static REQUESTS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REQUESTS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Ask the flush loop of `namespace` for an early flush. Non-blocking and
/// coalesced: repeated requests before the loop wakes are one flush. A
/// request made while the loop is busy stays recorded, and the next request
/// wakes it again.
pub fn request_flush(namespace: &str) {
    flush_requests()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(namespace.to_string());
    flush_notify().notify_waiters();
}

/// Resolves when some namespace asked for an early flush; the loop then
/// calls [`take_flush_request`] for its own namespace.
pub async fn flush_requested() {
    flush_notify().notified().await;
}

/// Whether `namespace` asked for an early flush since the last call (and
/// clears the request).
pub fn take_flush_request(namespace: &str) -> bool {
    flush_requests()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(namespace)
}

/// Record a read that found more fresh documents than it merges: counted
/// in `/metrics`, logged once per `(namespace, snapshot)`, and an early flush
/// requested.
#[cfg(feature = "text-index")]
pub(crate) fn note_overflow(
    namespace: &str,
    index: &str,
    snapshot_key: (u64, usize),
    fresh: usize,
) {
    crate::route_telemetry::record_text_fresh_overflow();
    static LOGGED: OnceLock<Mutex<Option<(String, u64, usize)>>> = OnceLock::new();
    let key = (namespace.to_string(), snapshot_key.0, snapshot_key.1);
    let first = {
        let mut logged = LOGGED
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if logged.as_ref() == Some(&key) {
            false
        } else {
            *logged = Some(key);
            true
        }
    };
    if first {
        tracing::info!(
            namespace,
            index,
            fresh_documents = fresh,
            limit = fresh_text_max_docs(),
            "full-text read takes the exact scan: more unflushed documents than it merges; \
             early flush requested"
        );
    }
    request_flush(namespace);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_flush_request_wakes_the_loop_once_per_namespace() {
        let waiter = tokio::spawn(flush_requested());
        tokio::task::yield_now().await;
        request_flush("fresh-a");
        request_flush("fresh-a");
        tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
            .await
            .expect("woken")
            .unwrap();
        assert!(take_flush_request("fresh-a"));
        assert!(!take_flush_request("fresh-a"));
        assert!(!take_flush_request("fresh-b"));
    }
}
