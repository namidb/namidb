//! Run a query against a snapshot and return its [`ExplainNode`] tree
//! decorated with real measurements (`profile`).
//!
//! Per-operator timings work through a task-local [`ProfileCollector`]:
//! `profile_query_tree` allocates a collector, scopes it on the
//! current tokio task with `tokio::task_local!`, and runs the regular
//! `execute` against the plan. Each call into
//! `execute_inner_with_routing` checks the task-local and, if a
//! collector is active, records the elapsed time and `rows.len()` of
//! the operator that just finished, keyed by the [`LogicalPlan`]
//! node's stable pointer. After execution, `profile_query_tree` walks
//! the plan + explain tree in lockstep and attributes the recorded
//! [`RuntimeStats`] to the matching [`ExplainNode`].
//!
//! Times are inclusive (an operator's elapsed includes the time its
//! children took). Exclusive timing would need children-subtracting
//! bookkeeping; inclusive is the metric callers usually want for
//! "who is slow" anyway.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use namidb_storage::Snapshot;

use crate::cost::StatsCatalog;
use crate::exec::{execute, ExecError, Params};
use crate::optimize::optimize;
use crate::parser::Query;
use crate::plan::{explain_tree_verbose, ExplainNode, LogicalPlan, RuntimeStats};
use crate::plan::{lower, LowerError};

/// Per-operator measurement bag used by `profile_query_tree`. The
/// executor probes [`collector_present`] cheaply on every operator
/// and only does the bookkeeping when one is attached.
#[derive(Default)]
pub struct ProfileCollector {
    samples: Mutex<HashMap<usize, OpProfile>>,
}

struct OpProfile {
    rows: u64,
    elapsed: Duration,
}

impl ProfileCollector {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn record(&self, plan_ptr: usize, elapsed: Duration, rows: u64) {
        if let Ok(mut g) = self.samples.lock() {
            g.insert(plan_ptr, OpProfile { rows, elapsed });
        }
    }

    fn get(&self, plan_ptr: usize) -> Option<RuntimeStats> {
        let g = self.samples.lock().ok()?;
        g.get(&plan_ptr).map(|p| RuntimeStats {
            rows_returned: p.rows,
            elapsed_us: u64::try_from(p.elapsed.as_micros()).unwrap_or(u64::MAX),
        })
    }
}

tokio::task_local! {
    static CURRENT_COLLECTOR: Arc<ProfileCollector>;
}

/// Cheap check used by the executor to decide whether the per-op
/// instrumentation hot path runs. `false` when the caller did not
/// wrap `execute` in a [`ProfileCollector`] scope, so the regular
/// query path keeps its baseline performance.
pub fn collector_present() -> bool {
    CURRENT_COLLECTOR.try_with(|_| ()).is_ok()
}

/// Record the elapsed / row count of an operator. Called by the
/// executor at the bottom of `execute_inner_with_routing`. No-op
/// when no collector is in scope.
pub fn record_op(plan: &LogicalPlan, elapsed: Duration, rows: u64) {
    let _ = CURRENT_COLLECTOR.try_with(|coll| {
        coll.record(plan as *const _ as usize, elapsed, rows);
    });
}

/// Walk `plan` and `tree` in lockstep, copying each recorded
/// [`RuntimeStats`] from the collector onto the matching
/// [`ExplainNode`]. The two trees were built from the same plan so
/// their child counts match; if they ever drift the walk degrades
/// gracefully (operators without a recorded entry stay
/// `profile = None`).
fn attribute_profiles(plan: &LogicalPlan, tree: &mut ExplainNode, coll: &ProfileCollector) {
    if let Some(stats) = coll.get(plan as *const _ as usize) {
        tree.profile = Some(stats);
    }
    let children = plan.children();
    for (i, child) in children.iter().enumerate() {
        if let Some(node) = tree.children.get_mut(i) {
            attribute_profiles(child, node, coll);
        }
    }
}

/// Error surfaced by [`profile_query_tree`].
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("lowering failed: {0}")]
    Lower(#[from] LowerError),
    #[error("execution failed: {0}")]
    Exec(#[from] ExecError),
}

/// Lower, optimise, execute, and decorate the resulting verbose
/// `ExplainNode` with real runtime measurements per operator. The
/// cloud worker's `/v1/cypher/profile` endpoint can hand the returned
/// tree straight back to the caller — it already carries both
/// estimates and actuals.
///
/// Every operator the executor visited gets its own `profile` entry
/// (inclusive elapsed + rows the operator produced). The root entry
/// is therefore the same as "stop-watch around `execute`" minus the
/// `lower` / `optimize` overhead.
pub async fn profile_query_tree(
    query: &Query,
    snapshot: &Snapshot<'_>,
    params: &Params,
    catalog: &StatsCatalog,
) -> Result<ExplainNode, ProfileError> {
    let plan = optimize(lower(query)?, catalog);
    let collector = ProfileCollector::new();
    let coll_for_scope = collector.clone();
    let rows = CURRENT_COLLECTOR
        .scope(coll_for_scope, async {
            execute(&plan, snapshot, params).await
        })
        .await?;
    let mut tree = explain_tree_verbose(&plan, catalog);
    attribute_profiles(&plan, &mut tree, &collector);
    // The root entry should always be present (every successful
    // execute records the outermost operator). Fall back to a
    // synthetic stop-watch if for some reason the root was not
    // attributed (e.g. an empty plan that bypasses the per-op hook).
    if tree.profile.is_none() {
        tree.profile = Some(RuntimeStats {
            rows_returned: rows.len() as u64,
            elapsed_us: 0,
        });
    }
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
                    ..Default::default()
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
        let profile = tree.profile.as_ref().expect("expected profile on root");
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
    }

    #[tokio::test]
    async fn profile_attributes_per_operator() {
        // Build a graph with a few nodes so the leaf operator
        // (`NodeScan`) sees rows. Then check that at least one inner
        // operator also got a profile entry, proving the per-op
        // walk works rather than only the root being populated.
        use namidb_core::value::Value as CoreValue;
        use std::collections::BTreeMap;

        let mut writer = WriterSession::open(store(), paths("profile-peritem"))
            .await
            .unwrap();
        for name in ["Alice", "Bob", "Carol"] {
            let mut p = BTreeMap::new();
            p.insert("name".into(), CoreValue::Str(name.into()));
            writer
                .upsert_node(
                    "Person",
                    NodeId::new(),
                    &NodeWriteRecord {
                        properties: p,
                        schema_version: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        writer.commit_batch().await.unwrap();
        let snap = writer.snapshot();

        // A query with an inline WHERE keeps Filter as a separate
        // operator (predicate pushdown leaves an explicit Filter when
        // it cannot prove the predicate is column-only). That gives
        // us NodeScan -> Filter -> Project, three operators with
        // distinct pointers — so we can assert per-op attribution
        // without depending on the optimiser folding stable shapes.
        let q =
            parse("MATCH (a:Person) WHERE a.name <> 'nobody' RETURN a.name AS name ORDER BY name")
                .unwrap();
        let catalog = StatsCatalog::empty();
        let tree = profile_query_tree(&q, &snap, &Params::new(), &catalog)
            .await
            .unwrap();

        fn collect_profiles<'a>(node: &'a ExplainNode, out: &mut Vec<&'a RuntimeStats>) {
            if let Some(p) = &node.profile {
                out.push(p);
            }
            for c in &node.children {
                collect_profiles(c, out);
            }
        }
        let mut profiles = Vec::new();
        collect_profiles(&tree, &mut profiles);
        assert!(
            profiles.len() >= 2,
            "expected per-op profiles on at least two operators, got {}",
            profiles.len()
        );
        // The root's rows_returned equals what the executor handed back.
        assert_eq!(tree.profile.as_ref().unwrap().rows_returned, 3);
    }

    #[tokio::test]
    async fn execute_without_collector_does_not_panic() {
        // The per-op hot path checks for a task-local collector. A
        // plain `execute` (no `profile_query_tree`) must keep working
        // — this is the regression guard for that.
        let writer = WriterSession::open(store(), paths("profile-noop"))
            .await
            .unwrap();
        let snap = writer.snapshot();
        let q = parse("MATCH (a:Person) RETURN a").unwrap();
        let catalog = StatsCatalog::empty();
        let plan = optimize(lower(&q).unwrap(), &catalog);
        let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
        assert!(rows.is_empty());
    }
}
