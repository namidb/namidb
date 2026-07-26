//! Bolt listener for `namidb-server`.
//!
//! Wires [`namidb_bolt::Session`] up to the writer session that the
//! HTTP router already owns, so both protocols share one
//! `WriterSession` per process (single-writer invariant from RFC-001).
//!
//! Most of the heavy lifting lives in `namidb-bolt`. This module
//! supplies the [`Backend`] adapter and the `accept()` loop.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use namidb_bolt::{
    AuthPolicy, Authenticator, Backend, BackendError, RunCancellation, RunOutcome, ServerInfo,
    Session, StatementType, Value,
};
use namidb_query::{
    execute_with_limits, execute_write_staged_with_deadline, parse as cypher_parse,
    plan as build_plan, ExecError, LowerError, Params, ParseError, Row, WriteOutcome,
};
use namidb_storage::WriterSession;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::auth::{AuthConfig, Principal};
use crate::metrics::{Protocol, QueryKind, WriterLockKind};
use crate::AppState;

/// One executed Bolt query, classified for metrics: read vs write (`None` if
/// it failed before planning), the wall-clock it took (up to the end of
/// execution, excluding any write-stall sleep), and the outcome the protocol
/// returns to the driver.
/// The uniform transient failure for a foreground writer-lock timeout.
fn writer_busy_error() -> BackendError {
    BackendError::Storage(
        "writer is busy: could not acquire the write lock within the configured bound; retry"
            .into(),
    )
}

fn memory_pressure_error(pressure: crate::memory::MemoryPressure) -> BackendError {
    BackendError::Storage(format!(
        "process memory pressure: resident {} bytes reached configured maximum {} bytes; \
         reconstructible caches were reclaimed, retry after memory falls",
        pressure.resident_bytes, pressure.max_bytes
    ))
}

struct RunObservation {
    kind: Option<QueryKind>,
    elapsed: std::time::Duration,
    result: std::result::Result<RunOutcome, BackendError>,
}

fn disconnected_observation(
    started: std::time::Instant,
    kind: Option<QueryKind>,
) -> RunObservation {
    RunObservation {
        kind,
        elapsed: started.elapsed(),
        result: Err(BackendError::Other(
            "Bolt client disconnected while RUN was executing".into(),
        )),
    }
}

/// In-flight explicit transaction (BEGIN..COMMIT/ROLLBACK). Holds the
/// global writer lock for the whole transaction so no other writer — nor
/// the flush / compaction tasks — can commit a half-built batch in the
/// middle of it. Staged statements live in the writer's pending batch and
/// are made durable in one commit at COMMIT, or dropped at ROLLBACK.
struct TxState {
    writer: OwnedMutexGuard<WriterSession>,
    /// Whether any statement staged a mutation, so ROLLBACK only discards
    /// when there is something to discard.
    staged: bool,
}

/// Adapter that drives Bolt `RUN` requests against the shared
/// [`WriterSession`]. One is created per connection.
pub struct ServerBackend {
    state: AppState,
    /// Per-connection explicit-transaction slot. `None` outside BEGIN..END.
    tx: Mutex<Option<TxState>>,
    /// The authenticated principal for this connection, set by the paired
    /// [`TokenAuthenticator`] at LOGON. `None` until authenticated (open mode
    /// leaves it `None`, which `principal()` resolves to an anonymous
    /// read-write caller). A `std::sync::Mutex` so the write gate reads it
    /// without an `.await`; per-connection, so never contended.
    principal: Arc<std::sync::Mutex<Option<Principal>>>,
}

impl ServerBackend {
    pub fn new(state: AppState, principal: Arc<std::sync::Mutex<Option<Principal>>>) -> Self {
        Self {
            state,
            tx: Mutex::new(None),
            principal,
        }
    }

    /// The connection's authenticated principal, or an anonymous read-write
    /// principal when unauthenticated (open mode).
    fn principal(&self) -> Principal {
        self.principal
            .lock()
            .expect("bolt principal lock poisoned")
            .clone()
            .unwrap_or_else(Principal::anonymous_rw)
    }

    /// Consult the pre-execution authorization hook for `plan`. Returns
    /// `Some(Forbidden)` to deny (mapped to a Bolt `Forbidden` error), `None`
    /// to allow. Mirrors the HTTP `authz.check` call so the Bolt path is NOT a
    /// policy bypass (NoOp default ⇒ always allows).
    async fn authz_denied(&self, plan: &namidb_query::LogicalPlan) -> Option<BackendError> {
        match self.state.authz.check(&self.principal(), plan).await {
            Ok(()) => None,
            Err(denied) => Some(BackendError::Forbidden(denied.to_string())),
        }
    }

    /// `Some(error)` when the connection's principal may not write, to reject a
    /// write before it touches the writer lock.
    fn write_forbidden(&self) -> Option<BackendError> {
        (!self.principal().allows_write()).then(|| {
            BackendError::Forbidden("this token is read-only; write queries are forbidden".into())
        })
    }

    /// Bolt shape for `CREATE VECTOR INDEX`: gate on role, run the DDL via
    /// the shared storage helper, return an empty `RunOutcome` tagged
    /// `Schema`. Auto-commit only — the in-transaction path rejects DDL
    /// (a schema command commits immediately and cannot be rolled back).
    #[cfg(feature = "vector-index")]
    async fn run_create_vector_index(
        &self,
        cvi: &namidb_query::parser::ast::CreateVectorIndexClause,
        started: std::time::Instant,
    ) -> RunObservation {
        if let Some(err) = self.write_forbidden() {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        // Authorization hook for the schema op (DDL is intercepted pre-plan).
        let op = crate::authz::SchemaOp::CreateVectorIndex {
            name: &cvi.name.name,
            label: &cvi.label.name,
            property: &cvi.property.name,
        };
        if let Err(denied) = self.state.authz.check_schema(&self.principal(), op).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Forbidden(denied.to_string())),
            };
        }
        let Some(mut writer) = self.state.lock_writer_bounded(WriterLockKind::Bolt).await else {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(writer_busy_error()),
            };
        };
        let result = crate::apply_create_vector_index(&mut writer, &self.state.snapshot, cvi).await;
        if let Err(e) = &result {
            // Reopen a fenced/poisoned session in place under the held lock
            // (no-op for user errors like a duplicate index name).
            crate::recovery::recover_writer_if_needed(
                &mut writer,
                &self.state.snapshot,
                &self.state.writer_health,
                &self.state.namespace,
                e,
            )
            .await;
        }
        drop(writer);
        let elapsed = started.elapsed();
        match result {
            Ok(_) => RunObservation {
                kind: Some(QueryKind::Write),
                elapsed,
                result: Ok(RunOutcome {
                    fields: vec![],
                    rows: vec![],
                    statement_type: StatementType::Schema,
                    counters: BTreeMap::new(),
                }),
            },
            Err(e) => {
                // A duplicate name/target is a user (semantic) error; a fence
                // or lost CAS is a transient storage error.
                let is_user = matches!(
                    &e,
                    namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_)
                );
                let err = if is_user {
                    BackendError::Semantic(e.to_string())
                } else {
                    map_storage_err(e)
                };
                RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed,
                    result: Err(err),
                }
            }
        }
    }

    /// Bolt shape for `CREATE FULLTEXT INDEX` (mirrors `run_create_vector_index`).
    #[cfg(feature = "text-index")]
    async fn run_create_fulltext_index(
        &self,
        cfi: &namidb_query::parser::ast::CreateFulltextIndexClause,
        started: std::time::Instant,
    ) -> RunObservation {
        if let Some(err) = self.write_forbidden() {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        let props: Vec<String> = cfi.properties.iter().map(|p| p.name.clone()).collect();
        let op = crate::authz::SchemaOp::CreateFulltextIndex {
            name: &cfi.name.name,
            label: &cfi.label.name,
            properties: &props,
        };
        if let Err(denied) = self.state.authz.check_schema(&self.principal(), op).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Forbidden(denied.to_string())),
            };
        }
        let Some(mut writer) = self.state.lock_writer_bounded(WriterLockKind::Bolt).await else {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(writer_busy_error()),
            };
        };
        let result =
            crate::apply_create_fulltext_index(&mut writer, &self.state.snapshot, cfi).await;
        drop(writer);
        let elapsed = started.elapsed();
        match result {
            Ok(_) => RunObservation {
                kind: Some(QueryKind::Write),
                elapsed,
                result: Ok(RunOutcome {
                    fields: vec![],
                    rows: vec![],
                    statement_type: StatementType::Schema,
                    counters: BTreeMap::new(),
                }),
            },
            Err(e) => {
                let is_user = matches!(
                    &e,
                    namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_)
                );
                let err = if is_user {
                    BackendError::Semantic(e.to_string())
                } else {
                    map_storage_err(e)
                };
                RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed,
                    result: Err(err),
                }
            }
        }
    }

    /// Bolt shape for `DROP VECTOR INDEX` (mirrors `run_create_vector_index`,
    /// same authz treatment as CREATE).
    #[cfg(feature = "vector-index")]
    async fn run_drop_vector_index(
        &self,
        dvi: &namidb_query::parser::ast::DropVectorIndexClause,
        started: std::time::Instant,
    ) -> RunObservation {
        if let Some(err) = self.write_forbidden() {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        let op = crate::authz::SchemaOp::DropVectorIndex {
            name: &dvi.name.name,
        };
        if let Err(denied) = self.state.authz.check_schema(&self.principal(), op).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Forbidden(denied.to_string())),
            };
        }
        let Some(mut writer) = self.state.lock_writer_bounded(WriterLockKind::Bolt).await else {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(writer_busy_error()),
            };
        };
        let result = crate::apply_drop_vector_index(&mut writer, &self.state.snapshot, dvi).await;
        if let Err(e) = &result {
            // Reopen a fenced/poisoned session in place under the held lock
            // (no-op for user errors like a duplicate index name).
            crate::recovery::recover_writer_if_needed(
                &mut writer,
                &self.state.snapshot,
                &self.state.writer_health,
                &self.state.namespace,
                e,
            )
            .await;
        }
        drop(writer);
        let elapsed = started.elapsed();
        match result {
            Ok(_) => RunObservation {
                kind: Some(QueryKind::Write),
                elapsed,
                result: Ok(RunOutcome {
                    fields: vec![],
                    rows: vec![],
                    statement_type: StatementType::Schema,
                    counters: BTreeMap::new(),
                }),
            },
            Err(e) => {
                // A missing index (without IF EXISTS) is a user (semantic)
                // error; a fence or lost CAS is a transient storage error.
                let is_user = matches!(
                    &e,
                    namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_)
                );
                let err = if is_user {
                    BackendError::Semantic(e.to_string())
                } else {
                    map_storage_err(e)
                };
                RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed,
                    result: Err(err),
                }
            }
        }
    }

    /// Bolt shape for `DROP INDEX` / `DROP FULLTEXT INDEX` (mirrors
    /// `run_drop_vector_index`).
    #[cfg(feature = "text-index")]
    async fn run_drop_fulltext_index(
        &self,
        dfi: &namidb_query::parser::ast::DropFulltextIndexClause,
        started: std::time::Instant,
    ) -> RunObservation {
        if let Some(err) = self.write_forbidden() {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        let op = crate::authz::SchemaOp::DropFulltextIndex {
            name: &dfi.name.name,
        };
        if let Err(denied) = self.state.authz.check_schema(&self.principal(), op).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Forbidden(denied.to_string())),
            };
        }
        let Some(mut writer) = self.state.lock_writer_bounded(WriterLockKind::Bolt).await else {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(writer_busy_error()),
            };
        };
        let result = crate::apply_drop_fulltext_index(&mut writer, &self.state.snapshot, dfi).await;
        if let Err(e) = &result {
            // Reopen a fenced/poisoned session in place under the held lock
            // (no-op for user errors like a duplicate index name).
            crate::recovery::recover_writer_if_needed(
                &mut writer,
                &self.state.snapshot,
                &self.state.writer_health,
                &self.state.namespace,
                e,
            )
            .await;
        }
        drop(writer);
        let elapsed = started.elapsed();
        match result {
            Ok(_) => RunObservation {
                kind: Some(QueryKind::Write),
                elapsed,
                result: Ok(RunOutcome {
                    fields: vec![],
                    rows: vec![],
                    statement_type: StatementType::Schema,
                    counters: BTreeMap::new(),
                }),
            },
            Err(e) => {
                let is_user = matches!(
                    &e,
                    namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_)
                );
                let err = if is_user {
                    BackendError::Semantic(e.to_string())
                } else {
                    map_storage_err(e)
                };
                RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed,
                    result: Err(err),
                }
            }
        }
    }

    /// Bolt shape for `CREATE CONSTRAINT`/`CREATE INDEX` (always-on schema DDL).
    async fn run_create_property_ddl(
        &self,
        name: Option<&str>,
        label: &str,
        properties: &[String],
        unique: bool,
        if_not_exists: bool,
        started: std::time::Instant,
    ) -> RunObservation {
        if let Some(err) = self.write_forbidden() {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        let op = if unique {
            crate::authz::SchemaOp::CreateConstraint { label, properties }
        } else {
            crate::authz::SchemaOp::CreateIndex {
                label,
                property: &properties[0],
            }
        };
        if let Err(denied) = self.state.authz.check_schema(&self.principal(), op).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Forbidden(denied.to_string())),
            };
        }
        let Some(mut writer) = self.state.lock_writer_bounded(WriterLockKind::Bolt).await else {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(writer_busy_error()),
            };
        };
        let result = if unique {
            crate::apply_create_constraint(
                &mut writer,
                &self.state.snapshot,
                name,
                label,
                properties,
                if_not_exists,
            )
            .await
        } else {
            crate::apply_create_index(
                &mut writer,
                &self.state.snapshot,
                name,
                label,
                &properties[0],
                if_not_exists,
            )
            .await
        };
        if let Err(e) = &result {
            crate::recovery::recover_writer_if_needed(
                &mut writer,
                &self.state.snapshot,
                &self.state.writer_health,
                &self.state.namespace,
                e,
            )
            .await;
        }
        drop(writer);
        let elapsed = started.elapsed();
        match result {
            Ok(_) => RunObservation {
                kind: Some(QueryKind::Write),
                elapsed,
                result: Ok(RunOutcome {
                    fields: vec![],
                    rows: vec![],
                    statement_type: StatementType::Schema,
                    counters: BTreeMap::new(),
                }),
            },
            Err(e) => {
                let is_user = matches!(
                    &e,
                    namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_)
                );
                let err = if is_user {
                    BackendError::Semantic(e.to_string())
                } else {
                    map_storage_err(e)
                };
                RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed,
                    result: Err(err),
                }
            }
        }
    }

    /// Auto-commit query: parse, plan, and execute against the published
    /// snapshot (reads) or the writer lock (writes), timing the work for the
    /// metrics. Mirrors the HTTP `run_cypher`. The stopwatch stops at the end
    /// of execution, before the optional write-stall sleep, so backpressure is
    /// not counted as query latency.
    async fn run_query(
        &self,
        cypher: &str,
        params: Params,
        cancellation: &RunCancellation,
    ) -> RunObservation {
        let started = std::time::Instant::now();
        if cancellation.is_cancelled() {
            return disconnected_observation(started, None);
        }

        let parsed = match cypher_parse(cypher) {
            Ok(p) => p,
            Err(errs) => {
                let first = &errs[0];
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(BackendError::Syntax(format!(
                        "{} at {}",
                        first.message, first.span
                    ))),
                };
            }
        };
        // `CREATE VECTOR INDEX` is schema DDL: intercept before planning.
        #[cfg(feature = "vector-index")]
        if let Some(cvi) = parsed.as_create_vector_index() {
            return self.run_create_vector_index(cvi, started).await;
        }

        // `CREATE FULLTEXT INDEX` is schema DDL: intercept before planning.
        #[cfg(feature = "text-index")]
        if let Some(cfi) = parsed.as_create_fulltext_index() {
            return self.run_create_fulltext_index(cfi, started).await;
        }

        // `DROP VECTOR INDEX` is schema DDL: intercept before planning.
        #[cfg(feature = "vector-index")]
        if let Some(dvi) = parsed.as_drop_vector_index() {
            return self.run_drop_vector_index(dvi, started).await;
        }

        // `DROP INDEX` / `DROP FULLTEXT INDEX`: schema DDL, intercept pre-plan.
        #[cfg(feature = "text-index")]
        if let Some(dfi) = parsed.as_drop_fulltext_index() {
            return self.run_drop_fulltext_index(dfi, started).await;
        }

        // `CREATE CONSTRAINT` / `CREATE INDEX`: schema DDL, intercept pre-plan.
        if let Some(c) = parsed.as_create_constraint() {
            let properties: Vec<String> = c.properties.iter().map(|p| p.name.clone()).collect();
            return self
                .run_create_property_ddl(
                    c.name.as_ref().map(|n| n.name.as_str()),
                    &c.label.name,
                    &properties,
                    true,
                    c.if_not_exists,
                    started,
                )
                .await;
        }
        if let Some(c) = parsed.as_create_index() {
            let properties = [c.property.name.clone()];
            return self
                .run_create_property_ddl(
                    c.name.as_ref().map(|n| n.name.as_str()),
                    &c.label.name,
                    &properties,
                    false,
                    c.if_not_exists,
                    started,
                )
                .await;
        }

        // `SHOW CONSTRAINTS` / `SHOW INDEXES`: schema introspection, answer from
        // the published manifest (a read; no writer lock).
        if let Some(c) = parsed.as_show_schema() {
            use namidb_query::parser::ast::ShowKind;
            let owned = self.state.snapshot.load();
            let manifest = &owned.manifest().manifest;
            let rows = match c.kind {
                ShowKind::Constraints => namidb_query::show_constraints_rows(&manifest.schema),
                ShowKind::Indexes => namidb_query::show_indexes_rows(manifest),
            };
            return RunObservation {
                kind: Some(QueryKind::Read),
                elapsed: started.elapsed(),
                result: Ok(show_run_outcome(rows)),
            };
        }

        // Plan against the latest published snapshot — no writer lock.
        let owned = self.state.snapshot.load();
        let catalog = self.state.catalog_for(&owned.manifest().manifest);
        let plan = match build_plan(&parsed, &catalog).map_err(map_lower_err) {
            Ok(p) => p,
            Err(e) => {
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(e),
                };
            }
        };

        // Pre-execution authorization hook (RFC-015 Wave B): a policy may deny
        // before execution. NoOp by default. Mirrors the HTTP path so Bolt is
        // not a policy bypass.
        if let Some(err) = self.authz_denied(&plan).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        if cancellation.is_cancelled() {
            return disconnected_observation(
                started,
                Some(if plan.contains_write() {
                    QueryKind::Write
                } else {
                    QueryKind::Read
                }),
            );
        }

        if plan.contains_write() {
            // A read-only token may not write — reject before the writer lock.
            if let Some(err) = self.write_forbidden() {
                return RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed: started.elapsed(),
                    result: Err(err),
                };
            }
            // Writes still take the writer lock (single-writer invariant),
            // bounded so queued writes fail fast behind a stuck writer.
            // On success we refresh the snapshot cell so subsequent reads
            // see the just-committed records (RFC-021).
            let writer_lock = self.state.lock_writer_bounded(WriterLockKind::Bolt);
            tokio::pin!(writer_lock);
            let writer = tokio::select! {
                writer = &mut writer_lock => writer,
                _ = cancellation.cancelled() => {
                    return disconnected_observation(started, Some(QueryKind::Write));
                }
            };
            let Some(mut writer) = writer else {
                return RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed: started.elapsed(),
                    result: Err(writer_busy_error()),
                };
            };

            // Applying a write only mutates the in-memory pending batch, so it
            // is cancellation-safe: on EOF drop the apply future, discard the
            // partial batch, and release the writer. The durability commit
            // below is deliberately NOT selected against cancellation — once
            // WAL/manifest publication starts it must run to a definite
            // outcome before this guard can be released.
            let staged = {
                let apply = execute_write_staged_with_deadline(
                    &plan,
                    &mut writer,
                    &params,
                    self.state.write_deadline(),
                );
                tokio::pin!(apply);
                tokio::select! {
                    result = &mut apply => Some(result),
                    _ = cancellation.cancelled() => None,
                }
            };
            let outcome = match staged {
                Some(Ok(outcome)) => outcome,
                Some(Err(error)) => {
                    writer.discard_batch();
                    crate::recovery::recover_after_write_error(
                        &mut writer,
                        &self.state.snapshot,
                        &self.state.writer_health,
                        &self.state.namespace,
                        &error,
                    )
                    .await;
                    drop(writer);
                    return RunObservation {
                        kind: Some(QueryKind::Write),
                        elapsed: started.elapsed(),
                        result: Err(map_exec_err(error)),
                    };
                }
                None => {
                    writer.discard_batch();
                    drop(writer);
                    return disconnected_observation(started, Some(QueryKind::Write));
                }
            };

            if cancellation.is_cancelled() {
                writer.discard_batch();
                drop(writer);
                return disconnected_observation(started, Some(QueryKind::Write));
            }

            match writer.commit_batch().await {
                Ok(_) => {
                    self.state.snapshot.store(writer.owned_snapshot());
                    // Soft write stall (RFC-027 P5): sample under the lock,
                    // release, then back off this request if L0 (or the
                    // memtable byte budget) is piling up.
                    let stall = self.state.after_commit_backpressure(&writer);
                    drop(writer);
                    let elapsed = started.elapsed();
                    if cancellation.is_cancelled() {
                        return disconnected_observation(started, Some(QueryKind::Write));
                    }
                    if let Some(delay) = stall {
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {}
                            _ = cancellation.cancelled() => {
                                return disconnected_observation(
                                    started,
                                    Some(QueryKind::Write),
                                );
                            }
                        }
                    }
                    RunObservation {
                        kind: Some(QueryKind::Write),
                        elapsed,
                        result: Ok(write_run_outcome(outcome)),
                    }
                }
                Err(error) => {
                    writer.discard_batch();
                    // A fenced/poisoned session would fail every later write
                    // on both protocols; reopen it in place under the lock.
                    crate::recovery::recover_writer_if_needed(
                        &mut writer,
                        &self.state.snapshot,
                        &self.state.writer_health,
                        &self.state.namespace,
                        &error,
                    )
                    .await;
                    drop(writer);
                    RunObservation {
                        kind: Some(QueryKind::Write),
                        elapsed: started.elapsed(),
                        result: Err(map_storage_err(error)),
                    }
                }
            }
        } else {
            // Read path: borrow a short-lived `Snapshot` from the owned
            // snapshot; the Arc keeps the underlying memtable alive for
            // the duration of the query, no writer lock needed.
            let snap = owned.borrow();
            let read = execute_with_limits(
                &plan,
                &snap,
                &params,
                self.state.query_deadline(),
                self.state.query_row_cap(),
            );
            tokio::pin!(read);
            let rows = tokio::select! {
                rows = &mut read => rows,
                _ = cancellation.cancelled() => {
                    return disconnected_observation(started, Some(QueryKind::Read));
                }
            };
            let elapsed = started.elapsed();
            RunObservation {
                kind: Some(QueryKind::Read),
                elapsed,
                result: rows.map(read_run_outcome).map_err(map_exec_err),
            }
        }
    }

    /// In-transaction query: stage writes into the held transaction's writer
    /// (no commit) or read with the staged batch overlaid (RFC-026), timing the
    /// work for the metrics. Mirrors the auto-commit `run_query`.
    async fn run_query_in_tx(
        &self,
        cypher: &str,
        params: Params,
        cancellation: &RunCancellation,
    ) -> RunObservation {
        let started = std::time::Instant::now();
        if cancellation.is_cancelled() {
            return disconnected_observation(started, None);
        }

        let parsed = match cypher_parse(cypher) {
            Ok(p) => p,
            Err(errs) => {
                let first = &errs[0];
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(BackendError::Syntax(format!(
                        "{} at {}",
                        first.message, first.span
                    ))),
                };
            }
        };
        // DDL commits immediately and cannot be rolled back, so it is
        // rejected inside an explicit transaction (auto-commit only).
        #[cfg(feature = "vector-index")]
        if parsed.as_create_vector_index().is_some() {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Unsupported(
                    "CREATE VECTOR INDEX cannot run inside a transaction".into(),
                )),
            };
        }
        #[cfg(feature = "text-index")]
        if parsed.as_create_fulltext_index().is_some() {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Unsupported(
                    "CREATE FULLTEXT INDEX cannot run inside a transaction".into(),
                )),
            };
        }
        #[cfg(feature = "vector-index")]
        if parsed.as_drop_vector_index().is_some() {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Unsupported(
                    "DROP VECTOR INDEX cannot run inside a transaction".into(),
                )),
            };
        }
        #[cfg(feature = "text-index")]
        if parsed.as_drop_fulltext_index().is_some() {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Unsupported(
                    "DROP INDEX cannot run inside a transaction".into(),
                )),
            };
        }
        if parsed.as_create_constraint().is_some() || parsed.as_create_index().is_some() {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Unsupported(
                    "CREATE CONSTRAINT / CREATE INDEX cannot run inside a transaction".into(),
                )),
            };
        }
        let owned = self.state.snapshot.load();
        let catalog = self.state.catalog_for(&owned.manifest().manifest);
        let plan = match build_plan(&parsed, &catalog).map_err(map_lower_err) {
            Ok(p) => p,
            Err(e) => {
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(e),
                };
            }
        };

        // Pre-execution authorization hook (RFC-015 Wave B); NoOp by default.
        if let Some(err) = self.authz_denied(&plan).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        if cancellation.is_cancelled() {
            return disconnected_observation(
                started,
                Some(if plan.contains_write() {
                    QueryKind::Write
                } else {
                    QueryKind::Read
                }),
            );
        }

        if plan.contains_write() {
            // A read-only token may not write, even inside an open transaction.
            if let Some(err) = self.write_forbidden() {
                return RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed: started.elapsed(),
                    result: Err(err),
                };
            }
            // Stage into the transaction's held writer; do NOT commit. The
            // RETURN rows are computed during the apply, so they stream now.
            let tx_lock = self.tx.lock();
            tokio::pin!(tx_lock);
            let mut slot = tokio::select! {
                slot = &mut tx_lock => slot,
                _ = cancellation.cancelled() => {
                    return disconnected_observation(started, Some(QueryKind::Write));
                }
            };
            let tx = match slot.as_mut() {
                Some(tx) => tx,
                None => {
                    // A protocol-state error, not an executed query: keep it out
                    // of the latency histogram (kind None) like a parse/plan
                    // error. It still counts toward queries_total status=error.
                    return RunObservation {
                        kind: None,
                        elapsed: started.elapsed(),
                        result: Err(BackendError::Other("no open transaction".into())),
                    };
                }
            };
            let staged = {
                let apply = execute_write_staged_with_deadline(
                    &plan,
                    &mut tx.writer,
                    &params,
                    self.state.write_deadline(),
                );
                tokio::pin!(apply);
                tokio::select! {
                    result = &mut apply => Some(result),
                    _ = cancellation.cancelled() => None,
                }
            };
            let result = match staged {
                Some(Ok(outcome)) => {
                    tx.staged = true;
                    Ok(write_run_outcome(outcome))
                }
                Some(Err(e)) => {
                    // A failed statement aborts the transaction. Drop whatever
                    // it (or an earlier statement) staged so a stray COMMIT
                    // cannot seal a partial write; the session moves to FAILED
                    // and the client must ROLLBACK / RESET.
                    tx.writer.discard_batch();
                    Err(map_exec_err(e))
                }
                None => {
                    tx.writer.discard_batch();
                    tx.staged = false;
                    Err(BackendError::Other(
                        "Bolt client disconnected while RUN was executing".into(),
                    ))
                }
            };
            RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result,
            }
        } else {
            // Read against the transaction's own writer so the staged batch
            // is visible (RFC-026). The writer pins the committed state at
            // tx-begin (no commit happens mid-tx while we hold the lock) and
            // overlays everything statements 1..N-1 staged.
            let tx_lock = self.tx.lock();
            tokio::pin!(tx_lock);
            let mut slot = tokio::select! {
                slot = &mut tx_lock => slot,
                _ = cancellation.cancelled() => {
                    return disconnected_observation(started, Some(QueryKind::Read));
                }
            };
            let tx = match slot.as_mut() {
                Some(tx) => tx,
                None => {
                    // See the write branch: a no-open-transaction error is a
                    // protocol-state error, not an executed query.
                    return RunObservation {
                        kind: None,
                        elapsed: started.elapsed(),
                        result: Err(BackendError::Other("no open transaction".into())),
                    };
                }
            };
            let snap = tx.writer.overlay_snapshot();
            let read = execute_with_limits(
                &plan,
                &snap,
                &params,
                self.state.query_deadline(),
                self.state.query_row_cap(),
            );
            tokio::pin!(read);
            let rows = tokio::select! {
                rows = &mut read => rows,
                _ = cancellation.cancelled() => {
                    return disconnected_observation(started, Some(QueryKind::Read));
                }
            };
            let elapsed = started.elapsed();
            RunObservation {
                kind: Some(QueryKind::Read),
                elapsed,
                result: rows.map(read_run_outcome).map_err(map_exec_err),
            }
        }
    }

    async fn run_observed(
        &self,
        cypher: &str,
        params: Params,
        cancellation: RunCancellation,
    ) -> std::result::Result<RunOutcome, BackendError> {
        let admission_started = std::time::Instant::now();
        if let Err(pressure) = self.state.memory.admit_query().await {
            let error = memory_pressure_error(pressure);
            self.state.metrics.observe_query(
                Protocol::Bolt,
                None,
                false,
                admission_started.elapsed(),
                cypher,
            );
            return Err(error);
        }

        // Memgraph-style schema introspection (gdotv and other Bolt GUIs)
        // bypasses the Cypher parser. It is a short metadata probe and does
        // not acquire the writer. Admission deliberately precedes it so this
        // parser bypass cannot evade process-wide memory pressure.
        {
            let owned = self.state.snapshot.load();
            let snap = owned.borrow();
            if let Some(result) = crate::introspect::try_introspect(cypher, &snap).await {
                return result;
            }
        }

        let _in_flight = self.state.metrics.track_in_flight();
        let obs = self.run_query(cypher, params, &cancellation).await;
        self.state.metrics.observe_query(
            Protocol::Bolt,
            obs.kind,
            obs.result.is_ok(),
            obs.elapsed,
            cypher,
        );
        obs.result
    }

    async fn run_in_tx_observed(
        &self,
        cypher: &str,
        params: Params,
        cancellation: RunCancellation,
    ) -> std::result::Result<RunOutcome, BackendError> {
        let admission_started = std::time::Instant::now();
        if let Err(pressure) = self.state.memory.admit_query().await {
            let error = memory_pressure_error(pressure);
            self.state.metrics.observe_query(
                Protocol::Bolt,
                None,
                false,
                admission_started.elapsed(),
                cypher,
            );
            return Err(error);
        }

        // Introspection stays on the published schema snapshot; data reads
        // below use the transaction overlay. Keep admission ahead of this
        // parser bypass for the same reason as the auto-commit path.
        {
            let owned = self.state.snapshot.load();
            let snap = owned.borrow();
            if let Some(result) = crate::introspect::try_introspect(cypher, &snap).await {
                return result;
            }
        }

        let _in_flight = self.state.metrics.track_in_flight();
        let obs = self.run_query_in_tx(cypher, params, &cancellation).await;
        self.state.metrics.observe_query(
            Protocol::Bolt,
            obs.kind,
            obs.result.is_ok(),
            obs.elapsed,
            cypher,
        );
        obs.result
    }
}

#[async_trait]
impl Backend for ServerBackend {
    async fn run(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError> {
        self.run_observed(cypher, params, RunCancellation::new())
            .await
    }

    async fn run_with_cancellation(
        &self,
        cypher: &str,
        params: Params,
        cancellation: RunCancellation,
    ) -> std::result::Result<RunOutcome, BackendError> {
        self.run_observed(cypher, params, cancellation).await
    }

    async fn logoff(&self) {
        // Clear the per-connection identity so a subsequent request on this
        // connection cannot reuse the logged-off principal. Without auth
        // re-established it falls back to anonymous (open mode) or is rejected
        // at the next write/authz gate. (The Authenticator trait documents
        // that an embedder binding identity out-of-band should reset it here.)
        *self.principal.lock().expect("bolt principal lock poisoned") = None;
    }

    async fn begin_tx(&self) -> std::result::Result<(), BackendError> {
        let mut slot = self.tx.lock().await;
        if slot.is_some() {
            return Err(BackendError::Other("a transaction is already open".into()));
        }
        // BEGIN admits a new unit of work before it queues for (and then pins)
        // the global writer. COMMIT and ROLLBACK intentionally remain
        // available under pressure so an already-open transaction can always
        // release the lock and its staged memory.
        if let Err(pressure) = self.state.memory.admit_query().await {
            return Err(memory_pressure_error(pressure));
        }
        // Take the global writer lock for the whole transaction, bounded so
        // a BEGIN queued behind a stuck/long transaction fails fast instead
        // of pinning the connection. Held across RUNs (and client
        // think-time) until COMMIT/ROLLBACK — see TxState.
        let timeout = self.state.writer_lock_timeout();
        let lock = self.state.writer.clone().lock_owned();
        let lock_started = std::time::Instant::now();
        let writer = if timeout.is_zero() {
            lock.await
        } else {
            match tokio::time::timeout(timeout, lock).await {
                Ok(guard) => guard,
                Err(_) => {
                    self.state.metrics.observe_writer_lock(
                        WriterLockKind::BoltTransaction,
                        lock_started.elapsed(),
                        false,
                    );
                    return Err(writer_busy_error());
                }
            }
        };
        self.state.metrics.observe_writer_lock(
            WriterLockKind::BoltTransaction,
            lock_started.elapsed(),
            true,
        );
        *slot = Some(TxState {
            writer,
            staged: false,
        });
        Ok(())
    }

    async fn run_in_tx(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError> {
        self.run_in_tx_observed(cypher, params, RunCancellation::new())
            .await
    }

    async fn run_in_tx_with_cancellation(
        &self,
        cypher: &str,
        params: Params,
        cancellation: RunCancellation,
    ) -> std::result::Result<RunOutcome, BackendError> {
        self.run_in_tx_observed(cypher, params, cancellation).await
    }

    async fn commit_tx(&self) -> std::result::Result<(), BackendError> {
        let mut slot = self.tx.lock().await;
        let mut tx = slot
            .take()
            .ok_or_else(|| BackendError::Other("no open transaction".into()))?;
        // One manifest CAS makes the whole transaction durable; then
        // republish so reads see it. Dropping `tx` releases the writer lock.
        match tx.writer.commit_batch().await {
            Ok(_) => {
                self.state.snapshot.store(tx.writer.owned_snapshot());
                Ok(())
            }
            Err(e) => {
                // The COMMIT failed, so the transaction is over: drop the
                // staged batch (we have already `take()`n the tx slot, so a
                // later ROLLBACK/RESET can no longer reach it, and nothing is
                // durable until the manifest CAS lands) so the aborted writes
                // can never be sealed by the next unrelated commit — then
                // reopen a fenced/poisoned session in place while we still
                // hold the writer lock.
                tx.writer.discard_batch();
                crate::recovery::recover_writer_if_needed(
                    &mut tx.writer,
                    &self.state.snapshot,
                    &self.state.writer_health,
                    &self.state.namespace,
                    &e,
                )
                .await;
                Err(map_storage_err(e))
            }
        }
    }

    async fn rollback_tx(&self) -> std::result::Result<(), BackendError> {
        let mut slot = self.tx.lock().await;
        if let Some(mut tx) = slot.take() {
            // Always discard: a statement that failed before `staged` was set
            // can still have left mutations in the pending batch. Discarding
            // an empty batch is a no-op. Dropping `tx` releases the writer.
            tx.writer.discard_batch();
        }
        Ok(())
    }

    async fn current_bookmark(&self) -> Option<String> {
        Some(format!(
            "namidb:v{}",
            self.state.snapshot.manifest_version()
        ))
    }
}

fn classify_write(o: &namidb_query::WriteOutcome) -> StatementType {
    let any_read = !o.rows.is_empty();
    let any_write = o.nodes_created > 0
        || o.nodes_deleted > 0
        || o.edges_created > 0
        || o.edges_deleted > 0
        || o.properties_set > 0;
    match (any_read, any_write) {
        (true, true) => StatementType::ReadWrite,
        (false, true) => StatementType::Write,
        (true, false) => StatementType::Read,
        (false, false) => StatementType::Write,
    }
}

fn field_list(rows: &[namidb_query::Row]) -> Vec<String> {
    rows.first()
        .map(|r| r.bindings.keys().cloned().collect())
        .unwrap_or_default()
}

/// Build the Bolt `RunOutcome` for a write statement (auto-commit or staged
/// in a transaction): the result rows plus the update counters.
fn write_run_outcome(outcome: WriteOutcome) -> RunOutcome {
    let stype = classify_write(&outcome);
    let fields = field_list(&outcome.rows);
    let mut counters = std::collections::BTreeMap::new();
    counters.insert("nodes-created".into(), outcome.nodes_created as i64);
    counters.insert("nodes-deleted".into(), outcome.nodes_deleted as i64);
    counters.insert("relationships-created".into(), outcome.edges_created as i64);
    counters.insert("relationships-deleted".into(), outcome.edges_deleted as i64);
    counters.insert("properties-set".into(), outcome.properties_set as i64);
    RunOutcome {
        fields,
        rows: outcome.rows,
        statement_type: stype,
        counters,
    }
}

/// Build the Bolt `RunOutcome` for a read statement.
fn read_run_outcome(rows: Vec<Row>) -> RunOutcome {
    let fields = field_list(&rows);
    RunOutcome {
        fields,
        rows,
        statement_type: StatementType::Read,
        counters: Default::default(),
    }
}

/// Build the Bolt `RunOutcome` for a `SHOW CONSTRAINTS`/`SHOW INDEXES` result.
/// The session emits each row's values by looking them up by field name, so the
/// canonical SHOW column order is used as the field list (and is surfaced even
/// when there are no rows).
fn show_run_outcome(rows: Vec<Row>) -> RunOutcome {
    RunOutcome {
        fields: namidb_query::show_schema_columns(),
        rows,
        statement_type: StatementType::Read,
        counters: Default::default(),
    }
}

/// Map a storage commit failure to a Bolt error. A failed manifest CAS
/// fences/poisons the `WriterSession` (its contract is "drop and reopen");
/// the commit paths run [`crate::recovery::recover_writer_if_needed`] before
/// returning, so the client sees a retryable storage error and the retry
/// lands on a reopened session.
fn map_storage_err(e: namidb_storage::Error) -> BackendError {
    BackendError::Storage(format!("{e}"))
}

fn map_lower_err(e: LowerError) -> BackendError {
    use namidb_query::LowerErrorKind;
    match e.kind {
        LowerErrorKind::UnsupportedFeature => BackendError::Unsupported(e.message),
        _ => BackendError::Semantic(e.message),
    }
}

fn map_exec_err(e: ExecError) -> BackendError {
    // A deliberately-unsupported feature surfaces as the typed
    // `BackendError::Unsupported` (Neo.ClientError.Statement.NotSupported),
    // not a generic eval/storage bucket — so a driver can tell "not
    // implemented" from a genuine internal bug. This is the exec-side twin
    // of `map_lower_err`'s UnsupportedFeature arm.
    if e.is_unsupported() {
        return BackendError::Unsupported(e.to_string());
    }
    match e {
        // A constraint violation has its own Neo4j error class so drivers
        // can distinguish it from an ordinary evaluation error.
        ExecError::Constraint(m) => BackendError::Constraint(m),
        // The rest are opaque from outside the crate; format and bucket as
        // either an eval or a storage error on a best-effort substring match.
        other => {
            let text = format!("{other}");
            if text.contains("storage") || text.contains("manifest") {
                BackendError::Storage(text)
            } else {
                BackendError::Eval(text)
            }
        }
    }
}

/// Bolt `Custom` authenticator backed by the server's token set. On a
/// successful LOGON it records the resolved [`Principal`] into the
/// per-connection cell the paired [`ServerBackend`] reads to gate writes — the
/// "out of band" per-connection context the [`Authenticator`] contract
/// describes.
struct TokenAuthenticator {
    auth: Arc<AuthConfig>,
    principal: Arc<std::sync::Mutex<Option<Principal>>>,
}

#[async_trait]
impl Authenticator for TokenAuthenticator {
    async fn authenticate(
        &self,
        extra: &BTreeMap<String, Value>,
    ) -> std::result::Result<(), String> {
        let str_field = |key: &str| {
            extra.get(key).and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            })
        };
        let scheme = str_field("scheme").unwrap_or("none");
        if scheme != "basic" && scheme != "bearer" {
            return Err(format!("unsupported auth scheme `{scheme}`"));
        }
        match str_field("credentials").and_then(|c| self.auth.principal_for(c)) {
            Some(p) => {
                *self.principal.lock().expect("bolt principal lock poisoned") = Some(p);
                Ok(())
            }
            None => Err("invalid credentials".into()),
        }
    }
}

/// Build the per-connection [`AuthPolicy`]: `Open` when no tokens are
/// configured, otherwise a [`TokenAuthenticator`] that records the resolved
/// principal for the backend's write gate.
fn make_policy(
    auth: &Arc<AuthConfig>,
    principal: Arc<std::sync::Mutex<Option<Principal>>>,
) -> AuthPolicy {
    if auth.is_open() {
        AuthPolicy::Open
    } else {
        AuthPolicy::Custom(Arc::new(TokenAuthenticator {
            auth: auth.clone(),
            principal,
        }))
    }
}

/// Bind the Bolt listener and serve sessions until the process exits.
pub async fn serve(
    state: AppState,
    listen: std::net::SocketAddr,
    auth: Arc<AuthConfig>,
    tx_timeout: std::time::Duration,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
    tls: Option<tokio_rustls::TlsAcceptor>,
) -> anyhow::Result<()> {
    // `Duration::ZERO` disables the per-transaction idle timeout.
    let tx_idle_timeout = (!tx_timeout.is_zero()).then_some(tx_timeout);
    let listener = TcpListener::bind(listen).await?;
    info!(addr = %listen, "namidb bolt listening");
    // The HELLO `server` agent must look like a Neo4j build or the
    // official drivers (and GUIs built on them: gdotv, Neo4j Browser,
    // Bloom) reject the connection with "Server does not identify as a
    // genuine Neo4j instance". Memgraph and Amazon Neptune present a
    // `Neo4j/<version>` agent for exactly this reason; the Bolt endpoint
    // exists for driver compatibility, so we default to one too.
    // Override via `NAMIDB_BOLT_SERVER_AGENT` (e.g. to the honest
    // `NamiDB/<version>` when talking to a lenient client).
    let agent =
        std::env::var("NAMIDB_BOLT_SERVER_AGENT").unwrap_or_else(|_| "Neo4j/5.13.0".to_string());
    info!(server_agent = %agent, "bolt server agent");

    // Cap concurrent Bolt connections so a flood of idle/slowloris sockets
    // cannot exhaust file descriptors, tasks, and memory. Overridable via
    // `NAMIDB_BOLT_MAX_CONNECTIONS`; a rejected connection is closed immediately
    // rather than queued. The handshake timeout bounds how long a connection
    // may occupy a permit before completing the (unauthenticated) handshake.
    let max_conns = std::env::var("NAMIDB_BOLT_MAX_CONNECTIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1024);
    let conn_limit = Arc::new(tokio::sync::Semaphore::new(max_conns));
    let handshake_timeout = Some(std::time::Duration::from_secs(10));
    info!(max_connections = max_conns, "bolt connection cap");
    loop {
        let (socket, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(p) => p,
                Err(e) => {
                    error!(error = %e, "bolt accept failed");
                    continue;
                }
            },
            // Stop accepting new connections on shutdown (SIGTERM/SIGINT). The
            // HTTP server drains in parallel; in-flight Bolt sessions finish on
            // their own tasks.
            _ = shutdown.wait_for(|stop| *stop) => {
                info!("shutdown signalled, bolt listener stopping");
                break;
            }
        };
        // Acquire a connection permit before doing any per-connection work.
        // At the cap the socket is dropped (closed) instead of spawning an
        // unbounded task — hard backpressure against connection floods.
        let permit = match conn_limit.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                warn!(%peer, max = max_conns, "bolt connection cap reached; rejecting");
                continue;
            }
        };
        if let Err(e) = socket.set_nodelay(true) {
            warn!(error = %e, %peer, "set_nodelay failed");
        }
        let state = state.clone();
        // One principal cell per connection, shared between the authenticator
        // (which sets it at LOGON) and the backend (which reads it on every
        // write). `None` until authenticated; open mode leaves it `None`.
        let principal = Arc::new(std::sync::Mutex::new(None));
        let policy = make_policy(&auth, principal.clone());
        let info = ServerInfo {
            agent: agent.clone(),
            connection_id: Uuid::now_v7().to_string(),
        };
        let tls = tls.clone();
        tokio::spawn(async move {
            // Hold the permit for the whole connection lifetime; dropping the
            // task (any exit path) releases it back to the semaphore.
            let _permit = permit;
            let backend: Arc<dyn Backend> = Arc::new(ServerBackend::new(state, principal));
            // `Session` is generic over the transport, so the only fork is the
            // optional TLS handshake on the accepted socket.
            match tls {
                Some(acceptor) => {
                    // Bound the TLS handshake too: a client that opens the
                    // socket but never drives the handshake must not pin the
                    // permit/task indefinitely.
                    let accepted = match handshake_timeout {
                        Some(t) => match tokio::time::timeout(t, acceptor.accept(socket)).await {
                            Ok(r) => r,
                            Err(_) => {
                                warn!(%peer, "bolt TLS handshake timed out");
                                return;
                            }
                        },
                        None => acceptor.accept(socket).await,
                    };
                    match accepted {
                        Ok(stream) => {
                            run_session(
                                stream,
                                info,
                                policy,
                                backend,
                                tx_idle_timeout,
                                handshake_timeout,
                                peer,
                            )
                            .await
                        }
                        Err(e) => warn!(error = %e, %peer, "bolt TLS handshake failed"),
                    }
                }
                None => {
                    run_session(
                        socket,
                        info,
                        policy,
                        backend,
                        tx_idle_timeout,
                        handshake_timeout,
                        peer,
                    )
                    .await
                }
            }
        });
    }
    Ok(())
}

/// Build and run one Bolt session over any byte stream — a plain `TcpStream`
/// or a TLS stream. `Session` is generic over the transport, so TLS adds only
/// a handshake in front of the same session loop.
async fn run_session<S>(
    socket: S,
    info: ServerInfo,
    policy: AuthPolicy,
    backend: Arc<dyn Backend>,
    tx_idle_timeout: Option<std::time::Duration>,
    handshake_timeout: Option<std::time::Duration>,
    peer: std::net::SocketAddr,
) where
    S: tokio::io::AsyncReadExt + tokio::io::AsyncWriteExt + Unpin,
{
    // Cap total transaction lifetime so a client that stays under the idle
    // timeout cannot pin the shared writer forever. Overridable; default 5 min.
    let max_tx_lifetime = std::env::var("NAMIDB_BOLT_MAX_TX_LIFETIME")
        .ok()
        .and_then(|s| humantime::parse_duration(&s).ok())
        .or_else(|| Some(std::time::Duration::from_secs(300)));
    let session = Session::new(socket, info, policy, backend)
        .with_tx_idle_timeout(tx_idle_timeout)
        .with_handshake_timeout(handshake_timeout)
        .with_max_tx_lifetime(max_tx_lifetime);
    if let Err(e) = session.run().await {
        warn!(error = %e, %peer, "bolt session ended with error");
    }
}

// `ParseError` is included for callers that want a custom Bolt error
// shape; today we collapse to a single `Syntax(String)` above.
#[allow(dead_code)]
fn parse_err_to_string(e: &ParseError) -> String {
    format!("{} at {}", e.message, e.span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Role;
    use namidb_core::{NodeId, Schema, Value as CoreValue};
    use namidb_storage::{NodeWriteRecord, SessionCaches};
    use object_store::ObjectStore;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::sync::Notify;

    /// Blocks exactly one cold SST read so a cancellation test can stop a
    /// write after it has acquired the global writer and staged an earlier
    /// clause, without relying on wall-clock timing.
    #[derive(Debug)]
    struct BlockOneSstGet {
        inner: Arc<dyn ObjectStore>,
        should_block: AtomicBool,
        started: Notify,
        release: Notify,
    }

    impl std::fmt::Display for BlockOneSstGet {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BlockOneSstGet({})", self.inner)
        }
    }

    #[async_trait]
    impl ObjectStore for BlockOneSstGet {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            if location.as_ref().contains("/sst/")
                && self.should_block.swap(false, Ordering::SeqCst)
            {
                self.started.notify_one();
                self.release.notified().await;
            }
            self.inner.get_opts(location, options).await
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &object_store::path::Path,
            to: &object_store::path::Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }
    }

    // A hook that denies everything — to prove the Bolt path consults it (the
    // gap the adversarial review found: Bolt used to skip the AuthzHook).
    struct DenyAll;
    #[async_trait]
    impl crate::authz::AuthzHook for DenyAll {
        async fn check(
            &self,
            _p: &Principal,
            _plan: &namidb_query::LogicalPlan,
        ) -> Result<(), crate::authz::Denied> {
            Err(crate::authz::Denied::new("denied by test policy"))
        }
    }

    async fn backend_with_authz(authz: Arc<dyn crate::authz::AuthzHook>) -> ServerBackend {
        let (store, paths) = namidb_storage::parse_uri("memory://bolt-authz-test").unwrap();
        let writer = namidb_storage::WriterSession::open(store, paths)
            .await
            .unwrap();
        let state = AppState::new(writer, None, "test".into()).with_authz(authz);
        // Authenticated read-write principal, so the deny can't be attributed
        // to the role gate — it must come from the AuthzHook.
        let principal = Arc::new(std::sync::Mutex::new(Some(Principal {
            subject: "tester".into(),
            role: Role::ReadWrite,
            groups: vec![],
        })));
        ServerBackend::new(state, principal)
    }

    #[tokio::test]
    async fn bolt_run_query_consults_authz_hook_and_can_deny_reads() {
        let backend = backend_with_authz(Arc::new(DenyAll)).await;
        // A plain READ — the role gate would allow it; the hook must deny.
        let err = backend
            .run("MATCH (n) RETURN n", Params::new())
            .await
            .expect_err("deny-all hook must reject the read over Bolt");
        assert!(matches!(err, BackendError::Forbidden(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn bolt_default_authz_allows_reads() {
        // NoOp default must not change behavior: the read succeeds.
        let backend = backend_with_authz(Arc::new(crate::authz::NoOpAuthz)).await;
        let out = backend.run("MATCH (n) RETURN n", Params::new()).await;
        assert!(out.is_ok(), "default authz should allow: {out:?}");
    }

    #[tokio::test]
    async fn bolt_memory_admission_precedes_introspection_and_begin_lock() {
        let (store, paths) = namidb_storage::parse_uri("memory://bolt-memory-admission").unwrap();
        let writer = namidb_storage::WriterSession::open(store, paths)
            .await
            .unwrap();
        let governor = Arc::new(crate::memory::MemoryGovernor::new(1));
        let state = AppState::new(writer, None, "bolt-memory".into())
            .with_memory_governor(Arc::clone(&governor));
        let backend = ServerBackend::new(state.clone(), Arc::new(std::sync::Mutex::new(None)));

        // This query normally bypasses the parser/executor entirely. It must
        // still be rejected before it reads the schema snapshot.
        let introspection = backend
            .run("CALL db.labels()", Params::new())
            .await
            .expect_err("introspection must not bypass memory admission");
        assert!(
            matches!(&introspection, BackendError::Storage(text) if text.contains("memory pressure")),
            "got {introspection:?}"
        );

        // BEGIN is also a new unit of work and must fail before taking the
        // long-lived global writer lock.
        let begin = backend
            .begin_tx()
            .await
            .expect_err("BEGIN must be rejected under memory pressure");
        assert!(
            matches!(&begin, BackendError::Storage(text) if text.contains("memory pressure")),
            "got {begin:?}"
        );
        assert!(
            state.writer.try_lock().is_ok(),
            "a rejected BEGIN must not pin the writer"
        );
        assert_eq!(governor.rejected_queries(), 2);
    }

    #[tokio::test]
    async fn bolt_commit_and_rollback_remain_available_under_memory_pressure() {
        let (store, paths) = namidb_storage::parse_uri("memory://bolt-memory-close-tx").unwrap();
        let writer = namidb_storage::WriterSession::open(store, paths)
            .await
            .unwrap();
        let governor = Arc::new(crate::memory::MemoryGovernor::new(1));
        let state = AppState::new(writer, None, "bolt-memory-close".into())
            .with_memory_governor(Arc::clone(&governor));
        let backend = ServerBackend::new(state.clone(), Arc::new(std::sync::Mutex::new(None)));

        // Model a transaction that was admitted before pressure rose. Closing
        // it must not consult admission: COMMIT needs to release staged memory
        // and the writer lock, and ROLLBACK is the emergency escape hatch.
        let mut commit_writer = state.writer.clone().lock_owned().await;
        commit_writer
            .upsert_node(
                "T",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([("k".into(), CoreValue::I64(1))]),
                    schema_version: 0,
                    labels: vec![],
                },
            )
            .unwrap();
        *backend.tx.lock().await = Some(TxState {
            writer: commit_writer,
            staged: true,
        });
        backend
            .commit_tx()
            .await
            .expect("COMMIT must remain available under pressure");

        let mut rollback_writer = state.writer.clone().lock_owned().await;
        rollback_writer
            .upsert_node(
                "T",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([("k".into(), CoreValue::I64(2))]),
                    schema_version: 0,
                    labels: vec![],
                },
            )
            .unwrap();
        *backend.tx.lock().await = Some(TxState {
            writer: rollback_writer,
            staged: true,
        });
        backend
            .rollback_tx()
            .await
            .expect("ROLLBACK must remain available under pressure");
        assert!(
            state.writer.try_lock().is_ok(),
            "closing either transaction must release the writer"
        );
        assert_eq!(
            governor.rejected_queries(),
            0,
            "COMMIT/ROLLBACK must not run admission"
        );
    }

    /// Regression for the fenced-writer dead end over Bolt: a COMMIT that
    /// hits the fence must trigger the automatic reopen so the next
    /// transaction on this server commits — no restart required.
    #[tokio::test]
    async fn bolt_commit_tx_recovers_after_writer_is_fenced() {
        let (store, paths) = namidb_storage::parse_uri("memory://bolt-fence-recover").unwrap();
        let writer = namidb_storage::WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let state = AppState::new(writer, None, "bolt-fence".into());
        let principal = Arc::new(std::sync::Mutex::new(None));
        let backend = ServerBackend::new(state.clone(), principal);

        // An interloper claims the namespace, fencing the server's writer.
        let interloper = namidb_storage::WriterSession::open(store, paths)
            .await
            .unwrap();
        drop(interloper);

        // An explicit transaction stages a write; COMMIT hits the fence.
        backend.begin_tx().await.unwrap();
        backend
            .run_in_tx("CREATE (:T {k: 1})", Params::new())
            .await
            .unwrap();
        let err = backend
            .commit_tx()
            .await
            .expect_err("the commit must fail on the fence");
        assert!(matches!(err, BackendError::Storage(_)), "got {err:?}");

        // The failed COMMIT ran the reopen: the next transaction commits.
        backend.begin_tx().await.unwrap();
        backend
            .run_in_tx("CREATE (:T {k: 2})", Params::new())
            .await
            .unwrap();
        backend
            .commit_tx()
            .await
            .expect("the Bolt commit path must recover after the reopen");
        assert_eq!(state.writer_health.status(), "ok");

        // Only the recovered commit is visible; the fenced (never-ACKed)
        // transaction did not resurrect.
        let out = backend
            .run("MATCH (t:T) RETURN t.k AS k", Params::new())
            .await
            .unwrap();
        assert_eq!(out.rows.len(), 1, "exactly the recovered write is durable");
    }

    /// A client EOF during an unbounded write must cancel the reversible apply
    /// phase, discard clauses already staged by that statement, and release
    /// the single writer even when the configured lock timeout is disabled.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_run_discards_partial_batch_and_releases_writer() {
        let (_unused, paths) =
            namidb_storage::parse_uri("memory://bolt-cancel-releases-writer").unwrap();
        let blocking_store = Arc::new(BlockOneSstGet {
            inner: Arc::new(object_store::memory::InMemory::new()),
            should_block: AtomicBool::new(false),
            started: Notify::new(),
            release: Notify::new(),
        });
        let store: Arc<dyn ObjectStore> = blocking_store.clone();

        // Seed one cold SST without read caches, then reopen the writer so the
        // MATCH below has to reach the object store after staging `Partial`.
        let mut seeder = namidb_storage::WriterSession::open_with_caches(
            store.clone(),
            paths.clone(),
            SessionCaches::none(),
        )
        .await
        .unwrap();
        seeder
            .upsert_node(
                "Seed",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([("k".into(), CoreValue::I64(1))]),
                    schema_version: 0,
                    labels: vec![],
                },
            )
            .unwrap();
        seeder.commit_batch().await.unwrap();
        seeder.flush(Schema::empty()).await.unwrap();
        drop(seeder);

        let writer =
            namidb_storage::WriterSession::open_with_caches(store, paths, SessionCaches::none())
                .await
                .unwrap();
        let state = AppState::new(writer, None, "bolt-cancel".into());
        let backend = Arc::new(ServerBackend::new(
            state.clone(),
            Arc::new(std::sync::Mutex::new(None)),
        ));

        blocking_store.should_block.store(true, Ordering::SeqCst);
        let cancellation = RunCancellation::new();
        let task = {
            let backend = Arc::clone(&backend);
            let cancellation = cancellation.clone();
            tokio::spawn(async move {
                backend
                    .run_with_cancellation(
                        "CREATE (p:Partial {k: 9}) \
                         WITH p MATCH (s:Seed) RETURN s.k AS k",
                        Params::new(),
                        cancellation,
                    )
                    .await
            })
        };

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            blocking_store.started.notified(),
        )
        .await
        .expect("write did not reach the blocked SST read");
        assert!(
            state.writer.try_lock().is_err(),
            "RUN must hold the writer while its apply phase is blocked"
        );

        cancellation.cancel();
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("cancelled RUN kept the writer pinned")
            .expect("cancelled RUN task panicked");
        assert!(
            matches!(result, Err(BackendError::Other(_))),
            "disconnect cancellation must terminate RUN, got {result:?}"
        );

        {
            let writer =
                tokio::time::timeout(std::time::Duration::from_secs(1), state.writer.lock())
                    .await
                    .expect("writer was not released after cancellation");
            assert_eq!(
                writer.pending_len(),
                0,
                "the clause staged before cancellation leaked into the next commit"
            );
        }

        backend
            .run("CREATE (:AfterCancel {k: 2})", Params::new())
            .await
            .expect("the next Bolt writer must make progress without restart");
        let partial = backend
            .run("MATCH (p:Partial) RETURN p", Params::new())
            .await
            .unwrap();
        assert!(
            partial.rows.is_empty(),
            "cancelled statement became visible later"
        );
        let after = backend
            .run("MATCH (n:AfterCancel) RETURN n.k AS k", Params::new())
            .await
            .unwrap();
        assert_eq!(after.rows.len(), 1, "subsequent committed write is visible");

        // No blocked future remains, but wake defensively if an object-store
        // implementation polls cancellation differently.
        blocking_store.release.notify_waiters();
    }

    #[tokio::test]
    async fn bolt_logoff_clears_principal() {
        let backend = backend_with_authz(Arc::new(crate::authz::NoOpAuthz)).await;
        // A principal is set; after LOGOFF it must be cleared (falls back to
        // anonymous, so a stale identity can't be reused).
        assert_eq!(backend.principal().subject, "tester");
        backend.logoff().await;
        assert_eq!(
            backend.principal().subject,
            Principal::anonymous_rw().subject,
            "logoff must clear the per-connection principal"
        );
    }
}
