//! Unindexed-filter hints for automatic indexing.
//!
//! While a statement runs, the executor notes each `(label, property)` that a
//! search filter or a pushed-down `WHERE` equality/range had to evaluate by
//! scanning, because no property index covers it. The server scopes a
//! [`AutoIndexHints`] sink around read statements and, when it is enabled,
//! schedules a background index build for what the sink collected. Nothing
//! is recorded outside such a scope.

use std::collections::BTreeSet;
use std::future::Future;
use std::sync::{Arc, Mutex};

use namidb_storage::Snapshot;

/// Distinct pairs one statement records at most.
const MAX_HINTS_PER_STATEMENT: usize = 16;

/// Collected `(label, property)` pairs of one statement.
#[derive(Clone, Debug, Default)]
pub struct AutoIndexHints(Arc<Mutex<BTreeSet<(String, String)>>>);

impl AutoIndexHints {
    pub fn new() -> Self {
        Self::default()
    }

    /// The pairs recorded so far, leaving the sink empty.
    pub fn take(&self) -> Vec<(String, String)> {
        std::mem::take(
            &mut *self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
        .into_iter()
        .collect()
    }

    fn insert(&self, label: &str, property: &str) {
        let mut hints = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if hints.len() < MAX_HINTS_PER_STATEMENT {
            hints.insert((label.to_string(), property.to_string()));
        }
    }
}

tokio::task_local! {
    static HINTS: AutoIndexHints;
}

/// Run `fut` with `hints` collecting its unindexed filter properties.
pub async fn with_auto_index_hints<F: Future>(hints: AutoIndexHints, fut: F) -> F::Output {
    HINTS.scope(hints, fut).await
}

/// Note that `property` of `label` was filtered by a scan. Ignored when no
/// sink is in scope or when an index or unique constraint already covers it.
pub(crate) fn note_unindexed(snapshot: &Snapshot<'_>, label: &str, property: &str) {
    let _ = HINTS.try_with(|hints| {
        let covered = snapshot
            .manifest()
            .manifest
            .schema
            .label(label)
            .and_then(|def| def.properties.iter().find(|p| p.name == property))
            .is_some_and(|p| p.indexed || p.unique);
        if !covered && !property.is_empty() {
            hints.insert(label, property);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hints_are_scoped_deduplicated_and_bounded() {
        let hints = AutoIndexHints::new();
        with_auto_index_hints(hints.clone(), async {
            let _ = HINTS.try_with(|sink| {
                for i in 0..40 {
                    sink.insert("L", &format!("p{}", i % 20));
                }
            });
        })
        .await;
        let taken = hints.take();
        assert_eq!(taken.len(), MAX_HINTS_PER_STATEMENT);
        assert!(hints.take().is_empty());
        // Outside a scope nothing is recorded and nothing panics.
        assert!(HINTS.try_with(|_| ()).is_err());
    }
}
