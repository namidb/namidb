//! Run a query against a snapshot and return its [`ExplainNode`] tree
//! decorated with real measurements (`profile`).
//!
//! For now this is the "stop-watch around `execute`" version: we time
//! the whole query and report `rows_returned` + `elapsed_us` on the
//! root node. Per-operator timings need a streaming executor we do
//! not have yet — see the [`RuntimeStats`] doc for the trade-off.

use namidb_storage::Snapshot;

use crate::cost::StatsCatalog;
use crate::exec::{execute, ExecError, Params};
use crate::optimize::optimize;
use crate::parser::Query;
use crate::plan::{explain_tree_verbose, ExplainNode, RuntimeStats};
use crate::plan::{lower, LowerError};

/// Error surfaced by [`profile_query_tree`].
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("lowering failed: {0}")]
    Lower(#[from] LowerError),
    #[error("execution failed: {0}")]
    Exec(#[from] ExecError),
}

/// Lower, optimise, execute, and decorate the resulting verbose
/// `ExplainNode` with real runtime measurements. The cloud worker's
/// `/v1/cypher/profile` endpoint can hand the returned tree straight
/// back to the caller — it already carries both estimates and
/// actuals.
pub async fn profile_query_tree(
    query: &Query,
    snapshot: &Snapshot<'_>,
    params: &Params,
    catalog: &StatsCatalog,
) -> Result<ExplainNode, ProfileError> {
    let plan = optimize(lower(query)?, catalog);
    let start = std::time::Instant::now();
    let rows = execute(&plan, snapshot, params).await?;
    let elapsed_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
    let mut tree = explain_tree_verbose(&plan, catalog);
    tree.profile = Some(RuntimeStats {
        rows_returned: rows.len() as u64,
        elapsed_us,
    });
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use namidb_core::id::{NamespaceId, NodeId};
    use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
    use object_store::memory::InMemory;
    use object_store::ObjectStore;

    use super::*;
    use crate::parser::parse;

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    #[tokio::test]
    async fn profile_root_carries_rows_returned_and_elapsed() {
        use namidb_core::value::Value as CoreValue;
        use std::collections::BTreeMap;

        let mut writer = WriterSession::open(store(), paths("profile-basic"))
            .await
            .unwrap();
        let alice = NodeId::new();
        let mut p = BTreeMap::new();
        p.insert("name".into(), CoreValue::Str("Alice".into()));
        writer
            .upsert_node(
                "Person",
                alice,
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                },
            )
            .unwrap();
        writer.commit_batch().await.unwrap();
        let snap = writer.snapshot();

        let q = parse("MATCH (a:Person) RETURN a.name AS name").unwrap();
        let catalog = StatsCatalog::empty();
        let tree = profile_query_tree(&q, &snap, &Params::new(), &catalog)
            .await
            .unwrap();
        let profile = tree.profile.expect("expected profile on root");
        assert_eq!(profile.rows_returned, 1);
        // Elapsed is a positive integer; we cannot assert an upper
        // bound because CI runners vary, but any successful call must
        // have spent a few microseconds at minimum.
        assert!(
            profile.elapsed_us < 60_000_000,
            "elapsed should be sub-minute"
        );
    }

    #[tokio::test]
    async fn profile_returns_zero_rows_for_empty_match() {
        let writer = WriterSession::open(store(), paths("profile-empty"))
            .await
            .unwrap();
        let snap = writer.snapshot();
        let q = parse("MATCH (a:Nonexistent) RETURN a").unwrap();
        let catalog = StatsCatalog::empty();
        let tree = profile_query_tree(&q, &snap, &Params::new(), &catalog)
            .await
            .unwrap();
        assert_eq!(tree.profile.as_ref().unwrap().rows_returned, 0);
        // Children still carry estimates from explain_tree_verbose; no
        // child carries a profile (per-op timings out of scope).
        fn assert_no_child_profile(node: &ExplainNode) {
            for c in &node.children {
                assert!(c.profile.is_none());
                assert_no_child_profile(c);
            }
        }
        assert_no_child_profile(&tree);
    }
}
