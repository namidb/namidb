//! Python bindings for the NamiDB storage engine.
//!
//! See `pyproject.toml` for the maturin build configuration. The
//! published wheel exposes a single top-level module `namidb` whose
//! main entry points are [`Client`] (open/insert/lookup + Cypher) and
//! [`QueryResult`] (cursor over a Cypher result set).

// pyo3 0.22 generates a lot of `Py<T>::into(Py<T>)` shapes inside the
// `#[pymethods]` macro that clippy 1.93 flags as `useless_conversion`.
// Suppress at the module level rather than littering every method.
#![allow(clippy::useless_conversion)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use object_store::ObjectStore;
use pyo3::exceptions::{PyPermissionError, PyRuntimeError, PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use uuid::Uuid;

use namidb_core::{NodeId, Value};
use namidb_query::exec::{NodeValue, RelValue};
use namidb_query::{
    execute_with_limits, execute_write_with_deadline, parse as cypher_parse, plan as build_plan,
    show_constraints_rows, show_indexes_rows, show_schema_columns, ExecError, LowerError, Params,
    ParseError, Query, Row, RuntimeValue, StatsCatalog,
};
use namidb_storage::{
    sweep_orphans, CommitOutcome, EdgeListView, EdgeView, EdgeWriteRecord, JanitorReport,
    ManifestStore, NodeView, NodeWriteRecord, OwnedSnapshot, PinnedSnapshot, ReaderSession,
    SnapshotCell, SstCache, WriterSession,
};

// pyo3 0.22's macro expands a `gil-refs` feature check this crate does not
// declare.
#[allow(unexpected_cfgs)]
mod query_errors {
    use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};

    pyo3::create_exception!(
        namidb,
        QueryTimeoutError,
        PyTimeoutError,
        "A Cypher query ran past its `timeout_ms`."
    );
    pyo3::create_exception!(
        namidb,
        QueryRowCapError,
        PyRuntimeError,
        "A Cypher read produced more rows than its `max_rows`."
    );
}
use query_errors::{QueryRowCapError, QueryTimeoutError};

/// What a [`Client`] holds: the namespace's writer, or a reader that never
/// claims the writer epoch (`read_only=True`).
///
/// A writer also carries a [`SnapshotCell`]: reads run on their snapshot
/// outside the writer lock, so `sweep()` needs the cell's registry to know
/// which older versions a read in another thread still holds.
#[derive(Clone)]
enum Session {
    Writer(Arc<Mutex<WriterSession>>, Arc<SnapshotCell>),
    Reader(Arc<Mutex<ReaderSession>>),
}

/// The snapshot one read runs on. A writer's is registered with the sweep's
/// retention horizon until it drops. A reader's is a plain `Arc`, because
/// [`ReaderSession`] releases a retired version's pin once nothing else
/// holds its snapshot.
enum ReadSnapshot {
    Pinned(PinnedSnapshot),
    Shared(Arc<OwnedSnapshot>),
}

impl std::ops::Deref for ReadSnapshot {
    type Target = OwnedSnapshot;
    fn deref(&self) -> &OwnedSnapshot {
        match self {
            ReadSnapshot::Pinned(p) => p,
            ReadSnapshot::Shared(s) => s,
        }
    }
}

impl Session {
    /// The committed snapshot reads run on. The lock is held only to clone
    /// it, so a long read never blocks writers or other reads. Waiting for
    /// the lock gives up at `deadline`.
    async fn snapshot(&self, deadline: Option<Instant>) -> PyResult<ReadSnapshot> {
        Ok(match self {
            Session::Writer(w, reads) => {
                // Publish and register under the writer lock, which a sweep
                // also holds while it reads the horizon: a read either
                // registered before the sweep or sees its result.
                let w = lock_until(w, deadline).await?;
                reads.store(w.owned_snapshot());
                ReadSnapshot::Pinned(reads.load())
            }
            Session::Reader(r) => {
                let mut r = lock_until(r, deadline).await?;
                r.keep_alive().await;
                ReadSnapshot::Shared(r.snapshot())
            }
        })
    }

    async fn close(&self) -> PyResult<()> {
        match self {
            Session::Writer(..) => Ok(()),
            Session::Reader(r) => r.lock().await.release().await.map_err(map_storage_err),
        }
    }

    fn writer(&self) -> PyResult<&Arc<Mutex<WriterSession>>> {
        Ok(self.writer_and_reads()?.0)
    }

    fn writer_and_reads(&self) -> PyResult<(&Arc<Mutex<WriterSession>>, &Arc<SnapshotCell>)> {
        match self {
            Session::Writer(w, reads) => Ok((w, reads)),
            Session::Reader(_) => Err(PyPermissionError::new_err(
                "this Client was opened with read_only=True",
            )),
        }
    }
}

/// Lock `mutex`, giving up with `QueryTimeoutError` once `deadline` passes.
async fn lock_until<T>(
    mutex: &Mutex<T>,
    deadline: Option<Instant>,
) -> PyResult<tokio::sync::MutexGuard<'_, T>> {
    match deadline {
        None => Ok(mutex.lock().await),
        Some(deadline) => tokio::time::timeout_at(deadline.into(), mutex.lock())
            .await
            .map_err(|_| QueryTimeoutError::new_err("query timed out waiting for the client lock")),
    }
}

/// A namespace handle: object store + writer (or reader) session + a tokio
/// runtime that drives every async storage call from synchronous Python.
/// Every blocking call releases the GIL while it waits.
#[pyclass(module = "namidb")]
pub struct Client {
    runtime: Runtime,
    session: Session,
    /// Serialises compaction passes without sharing the writer mutex. Two
    /// concurrent `acompact()` calls must not prepare from the same basis and
    /// race to remove the same immutable inputs.
    compaction_lock: Arc<Mutex<()>>,
    store: Arc<dyn ObjectStore>,
    paths: namidb_storage::NamespacePaths,
    cache: SstCache,
    /// Orphan age for the sweep that follows an applied compaction; `None`
    /// leaves reclaiming to explicit `sweep()` calls.
    auto_sweep_min_age: Option<Duration>,
    /// `flush()` compacts inline once a bucket holds this many L0 SSTs
    /// (`0` disables).
    compaction_l0_trigger: usize,
}

#[pymethods]
impl Client {
    /// Open (or bootstrap) a namespace.
    ///
    /// `uri` accepts:
    /// - `memory://<namespace>` — in-process store (testing).
    /// - `file:///abs/dir?ns=<ns>` or `file://./rel?ns=<ns>` — local
    /// filesystem with full manifest CAS (via flock + atomic rename).
    /// - `s3://<bucket>[/<prefix>]?ns=<ns>[&region=...][&endpoint=...][&allow_http=true|false]`
    /// — any S3-compatible service (AWS S3, Cloudflare R2, MinIO,
    /// Tigris, LocalStack, …). Credentials come from the standard
    /// AWS env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    /// optional `AWS_SESSION_TOKEN`).
    /// - `gs://<bucket>[/<prefix>]?ns=<ns>[&service_account=/path/key.json]`
    /// — Google Cloud Storage. Auth from `GOOGLE_APPLICATION_CREDENTIALS`
    /// or `?service_account=`.
    /// - `az://<account>/<container>[/<prefix>]?ns=<ns>[&endpoint=...][&allow_http=true][&use_emulator=true]`
    /// — Azure Blob Storage. Auth from `AZURE_STORAGE_ACCOUNT_NAME`
    /// + `AZURE_STORAGE_ACCESS_KEY` (or SAS token via env).
    ///
    /// With `read_only=True` the client never claims the namespace, so a
    /// writer in another process keeps running. It serves the newest commit
    /// as of open (or the last [`Client::refresh`], at most an hour back)
    /// and every mutator raises `PermissionError`. The namespace must
    /// already exist. [`Client::close`] drops its retention lease.
    ///
    /// `spool_dir` is where compaction writes its scratch files. It defaults
    /// to `NAMIDB_SPOOL_DIR`, then to `<root>/.namidb-spool` for `file://`
    /// URIs. `auto_sweep_min_age_secs` reclaims objects an applied compaction
    /// retired once they were written that long ago (`None` disables; it is
    /// no grace period for a reader in another process), and `flush()`
    /// compacts once a bucket reaches `compaction_l0_trigger` L0 SSTs (`0`
    /// disables). A `read_only` client never compacts or sweeps, so these
    /// options do nothing there and no spool dir is created.
    #[new]
    #[pyo3(signature = (uri, cache_bytes=None, *, read_only=false, spool_dir=None, auto_sweep_min_age_secs=Some(60), compaction_l0_trigger=8))]
    fn new(
        py: Python<'_>,
        uri: &str,
        cache_bytes: Option<usize>,
        read_only: bool,
        spool_dir: Option<String>,
        auto_sweep_min_age_secs: Option<u64>,
        compaction_l0_trigger: usize,
    ) -> PyResult<Self> {
        let (store, paths) = parse_uri(uri)?;
        if !read_only {
            configure_spool_dir(uri, spool_dir)?;
        }
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
        let cache = SstCache::new(cache_bytes.unwrap_or(64 * 1024 * 1024));
        let session = py
            .allow_threads(|| {
                runtime.block_on(async {
                    Ok::<_, namidb_storage::Error>(if read_only {
                        let reader =
                            ReaderSession::open(store.clone(), paths.clone(), Some(cache.clone()))
                                .await?;
                        Session::Reader(Arc::new(Mutex::new(reader)))
                    } else {
                        let writer = WriterSession::open(store.clone(), paths.clone()).await?;
                        let reads = Arc::new(SnapshotCell::new(writer.owned_snapshot()));
                        Session::Writer(Arc::new(Mutex::new(writer)), reads)
                    })
                })
            })
            .map_err(map_storage_err)?;
        Ok(Self {
            runtime,
            session,
            compaction_lock: Arc::new(Mutex::new(())),
            store,
            paths,
            cache,
            auto_sweep_min_age: auto_sweep_min_age_secs
                .filter(|_| !read_only)
                .map(Duration::from_secs),
            compaction_l0_trigger: if read_only { 0 } else { compaction_l0_trigger },
        })
    }

    /// Stage a node upsert into the current batch.
    fn upsert_node(
        &self,
        py: Python<'_>,
        label: &str,
        id: &str,
        properties: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let id = parse_node_id(id)?;
        let props = py_dict_to_value_map(properties)?;
        let record = NodeWriteRecord {
            properties: props,
            schema_version: 0,
            ..Default::default()
        };
        let session = self.session.writer()?;
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                let labels = [label.to_string()];
                namidb_query::enforce_node_unique_constraints(
                    &session,
                    &labels,
                    &record.properties,
                    Some(id),
                )
                .await
                .map_err(PyValueError::new_err)?;
                session
                    .upsert_node(label, id, &record)
                    .map_err(map_storage_err)
            })
        })?;
        Ok(())
    }

    /// Stage a multi-label node upsert into the current batch. `labels` is the
    /// full label set the node carries; the node is keyed by `id` alone, so a
    /// later upsert with a different set replaces it (last-write-wins).
    fn upsert_node_with_labels(
        &self,
        py: Python<'_>,
        labels: Vec<String>,
        id: &str,
        properties: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let id = parse_node_id(id)?;
        let props = py_dict_to_value_map(properties)?;
        let record = NodeWriteRecord {
            properties: props,
            schema_version: 0,
            ..Default::default()
        };
        let session = self.session.writer()?;
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                namidb_query::enforce_node_unique_constraints(
                    &session,
                    &labels,
                    &record.properties,
                    Some(id),
                )
                .await
                .map_err(PyValueError::new_err)?;
                session
                    .upsert_node_with_labels(labels, id, &record)
                    .map_err(map_storage_err)
            })
        })?;
        Ok(())
    }

    /// Stage a node tombstone into the current batch.
    fn tombstone_node(&self, py: Python<'_>, label: &str, id: &str) -> PyResult<()> {
        let id = parse_node_id(id)?;
        let session = self.session.writer()?;
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                session.tombstone_node(label, id).map_err(map_storage_err)
            })
        })?;
        Ok(())
    }

    /// Stage an edge upsert into the current batch.
    fn upsert_edge(
        &self,
        py: Python<'_>,
        edge_type: &str,
        src: &str,
        dst: &str,
        properties: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let src = parse_node_id(src)?;
        let dst = parse_node_id(dst)?;
        let props = py_dict_to_value_map(properties)?;
        let record = EdgeWriteRecord {
            properties: props,
            schema_version: 0,
        };
        let session = self.session.writer()?;
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                session
                    .upsert_edge(edge_type, src, dst, &record)
                    .map_err(map_storage_err)
            })
        })?;
        Ok(())
    }

    /// Bulk-load nodes from a Parquet file. Required column `node_id`
    /// (`FixedSizeBinary(16)` raw UUID); every other column becomes a
    /// property. `commit_every` rows per durability boundary (`0` leaves
    /// everything pending for a final `commit()` / `flush()`). Returns the
    /// number of rows loaded. This is the file-to-graph fast path, no
    /// per-row dict construction.
    #[pyo3(signature = (label, path, commit_every=0))]
    fn load_parquet_nodes(
        &self,
        py: Python<'_>,
        label: &str,
        path: &str,
        commit_every: usize,
    ) -> PyResult<usize> {
        let path = std::path::PathBuf::from(path);
        let session = self.session.writer()?;
        let outcome = py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                namidb_storage::load_nodes_from_parquet(&path, &mut session, label, commit_every)
                    .await
                    .map_err(map_storage_err)
            })
        })?;
        Ok(outcome.rows_loaded)
    }

    /// Bulk-load edges from a Parquet file. Required columns `src` / `dst`
    /// (`FixedSizeBinary(16)` endpoint UUIDs); every other column becomes a
    /// property. Endpoint nodes must already exist, so load the node files
    /// first. Returns the number of edges loaded.
    #[pyo3(signature = (edge_type, path, commit_every=0))]
    fn load_parquet_edges(
        &self,
        py: Python<'_>,
        edge_type: &str,
        path: &str,
        commit_every: usize,
    ) -> PyResult<usize> {
        let path = std::path::PathBuf::from(path);
        let session = self.session.writer()?;
        let outcome = py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                namidb_storage::load_edges_from_parquet(
                    &path,
                    &mut session,
                    edge_type,
                    commit_every,
                )
                .await
                .map_err(map_storage_err)
            })
        })?;
        Ok(outcome.rows_loaded)
    }

    /// Stage an edge tombstone into the current batch.
    fn tombstone_edge(
        &self,
        py: Python<'_>,
        edge_type: &str,
        src: &str,
        dst: &str,
    ) -> PyResult<()> {
        let src = parse_node_id(src)?;
        let dst = parse_node_id(dst)?;
        let session = self.session.writer()?;
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                session
                    .tombstone_edge(edge_type, src, dst)
                    .map_err(map_storage_err)
            })
        })?;
        Ok(())
    }

    /// Durably commit the current batch (WAL append + manifest CAS).
    /// Returns the last LSN persisted, or `None` when the batch was empty.
    fn commit(&self, py: Python<'_>) -> PyResult<Option<u64>> {
        let session = self.session.writer()?;
        let last_lsn = py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                let outcome = session.commit_batch().await.map_err(map_storage_err)?;
                Ok::<Option<u64>, PyErr>(match outcome {
                    CommitOutcome::Empty => None,
                    CommitOutcome::Committed { last_lsn, .. } => Some(last_lsn),
                })
            })
        })?;
        Ok(last_lsn)
    }

    /// Flush the memtable into L0 SSTs, then run a compaction pass when a
    /// bucket has reached `compaction_l0_trigger` L0 SSTs.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.writer()?;
        let l0 = py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut session = session.lock().await;
                let schema = session.snapshot().manifest().manifest.schema.clone();
                session.flush(schema).await.map_err(map_storage_err)?;
                Ok::<_, PyErr>(session.max_l0_bucket_len())
            })
        })?;
        // Embedded mode has no maintenance loop, so without this L0 grows
        // until someone calls compact() and every read fans out across it.
        // The flush is already durable, so a failed pass is only a warning:
        // raising would read as "data not persisted" to a retrying caller.
        if self.compaction_l0_trigger > 0 && l0 >= self.compaction_l0_trigger {
            let session = session.clone();
            let compaction_lock = self.compaction_lock.clone();
            let auto_sweep = self.auto_sweep();
            let compacted = py.allow_threads(|| {
                self.runtime
                    .block_on(run_compaction_inner(session, compaction_lock, auto_sweep))
            });
            if let Err(e) = compacted {
                PyErr::warn_bound(
                    py,
                    &py.get_type_bound::<PyRuntimeWarning>(),
                    &format!("flush succeeded, but the compaction it triggered failed: {e}"),
                    1,
                )?;
            }
        }
        Ok(())
    }

    /// Run one leveled compaction pass.
    ///
    /// The expensive prepare phase (object reads, row merging, vector/full-text
    /// index builds, and object writes) runs without the writer mutex. The
    /// mutex is held only while capturing a basis and while installing the
    /// prepared manifest update. Returns a dict describing the pass.
    fn compact(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let session = self.session.writer()?.clone();
        let compaction_lock = self.compaction_lock.clone();
        let auto_sweep = self.auto_sweep();
        let report = py.allow_threads(|| {
            self.runtime
                .block_on(run_compaction_inner(session, compaction_lock, auto_sweep))
        })?;
        compaction_report_to_py(py, &report)
    }

    /// Async sibling of [`Client::compact`]. Returns the same report dict.
    fn acompact<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let session = self.session.writer()?.clone();
        let compaction_lock = self.compaction_lock.clone();
        let auto_sweep = self.auto_sweep();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let report = run_compaction_inner(session, compaction_lock, auto_sweep).await?;
            Python::with_gil(|py| compaction_report_to_py(py, &report))
        })
    }

    /// Delete objects no manifest version references any more and that are
    /// at least `min_age_secs` old: SSTs a compaction merged away, WAL
    /// segments a flush retired, superseded manifests. With `dry_run=True`
    /// nothing is deleted and the dict reports what would be.
    #[pyo3(signature = (min_age_secs=60, dry_run=false))]
    fn sweep(&self, py: Python<'_>, min_age_secs: u64, dry_run: bool) -> PyResult<Py<PyDict>> {
        let (session, reads) = self.session.writer_and_reads()?;
        let manifest_store = ManifestStore::new(self.store.clone(), self.paths.clone());
        let report = py.allow_threads(|| {
            self.runtime.block_on(async {
                let _compaction_guard = self.compaction_lock.lock().await;
                let guard = session.lock().await;
                sweep_locked(
                    &guard,
                    reads,
                    &manifest_store,
                    Duration::from_secs(min_age_secs),
                    !dry_run,
                )
                .await
            })
        })?;
        let d = PyDict::new_bound(py);
        d.set_item("orphans_found", report.orphans_found)?;
        d.set_item("orphans_deleted", report.orphans_deleted)?;
        d.set_item("bytes_freed", report.bytes_freed)?;
        d.set_item("skipped_too_young", report.skipped_too_young)?;
        d.set_item("wal_segments_reclaimed", report.wal_segments_reclaimed)?;
        d.set_item("wal_bytes_freed", report.wal_bytes_freed)?;
        d.set_item(
            "manifest_snapshots_reclaimed",
            report.manifest_snapshots_reclaimed,
        )?;
        d.set_item("manifest_bytes_freed", report.manifest_bytes_freed)?;
        d.set_item("pointer_files_reclaimed", report.pointer_files_reclaimed)?;
        d.set_item("pointer_bytes_freed", report.pointer_bytes_freed)?;
        d.set_item(
            "memtable_snapshots_reclaimed",
            report.memtable_snapshots_reclaimed,
        )?;
        d.set_item(
            "memtable_snapshot_bytes_freed",
            report.memtable_snapshot_bytes_freed,
        )?;
        d.set_item("expired_pins_reclaimed", report.expired_pins_reclaimed)?;
        d.set_item("pins_honoured", report.pins_honoured)?;
        // A pin lease that landed mid-sweep deletes nothing; run it again.
        d.set_item("aborted_by_pin", report.aborted_by_pin)?;
        Ok(d.into())
    }

    /// Look up a single node by `(label, id)`.
    fn lookup_node<'py>(
        &self,
        py: Python<'py>,
        label: &str,
        id: &str,
    ) -> PyResult<Option<Py<PyDict>>> {
        let id = parse_node_id(id)?;
        let view: Option<NodeView> = py.allow_threads(|| {
            self.runtime.block_on(async {
                let owned = self.session.snapshot(None).await?;
                let snap = owned.borrow().with_cache(self.cache.clone());
                snap.lookup_node(label, id).await.map_err(map_storage_err)
            })
        })?;
        match view {
            Some(v) => Ok(Some(node_view_to_py(py, &v)?.into())),
            None => Ok(None),
        }
    }

    /// All outgoing edges of `(edge_type, src)`.
    fn out_edges<'py>(&self, py: Python<'py>, edge_type: &str, src: &str) -> PyResult<Py<PyList>> {
        let src = parse_node_id(src)?;
        let list: EdgeListView = py.allow_threads(|| {
            self.runtime.block_on(async {
                let owned = self.session.snapshot(None).await?;
                let snap = owned.borrow().with_cache(self.cache.clone());
                snap.out_edges(edge_type, src)
                    .await
                    .map_err(map_storage_err)
            })
        })?;
        edge_list_to_py(py, &list)
    }

    /// All incoming edges of `(edge_type, dst)`.
    fn in_edges<'py>(&self, py: Python<'py>, edge_type: &str, dst: &str) -> PyResult<Py<PyList>> {
        let dst = parse_node_id(dst)?;
        let list: EdgeListView = py.allow_threads(|| {
            self.runtime.block_on(async {
                let owned = self.session.snapshot(None).await?;
                let snap = owned.borrow().with_cache(self.cache.clone());
                snap.in_edges(edge_type, dst).await.map_err(map_storage_err)
            })
        })?;
        edge_list_to_py(py, &list)
    }

    /// All nodes under `label` (memtable + SSTs, last-write-wins). `limit`
    /// stops after that many nodes.
    #[pyo3(signature = (label, limit=None))]
    fn scan_label<'py>(
        &self,
        py: Python<'py>,
        label: &str,
        limit: Option<usize>,
    ) -> PyResult<Py<PyList>> {
        let views = self.scan_nodes(py, label, limit)?;
        let list = PyList::empty_bound(py);
        for v in &views {
            list.append(node_view_to_py(py, v)?)?;
        }
        Ok(list.into())
    }

    /// All edges of `edge_type` (memtable + SSTs, last-write-wins). `limit`
    /// stops after that many edges.
    #[pyo3(signature = (edge_type, limit=None))]
    fn scan_edge_type<'py>(
        &self,
        py: Python<'py>,
        edge_type: &str,
        limit: Option<usize>,
    ) -> PyResult<Py<PyList>> {
        let views = self.scan_edges(py, edge_type, limit)?;
        let list = PyList::empty_bound(py);
        for v in &views {
            list.append(edge_view_to_py(py, v)?)?;
        }
        Ok(list.into())
    }

    /// Like [`Client::scan_label`], but returns an iterator that builds one
    /// dict per step instead of a list holding every node at once.
    #[pyo3(signature = (label, limit=None))]
    fn iter_label(&self, py: Python<'_>, label: &str, limit: Option<usize>) -> PyResult<NodeIter> {
        Ok(NodeIter {
            views: self.scan_nodes(py, label, limit)?.into_iter(),
        })
    }

    /// Like [`Client::scan_edge_type`], but returns an iterator that builds
    /// one dict per step.
    #[pyo3(signature = (edge_type, limit=None))]
    fn iter_edge_type(
        &self,
        py: Python<'_>,
        edge_type: &str,
        limit: Option<usize>,
    ) -> PyResult<EdgeIter> {
        Ok(EdgeIter {
            views: self.scan_edges(py, edge_type, limit)?.into_iter(),
        })
    }

    /// Move a `read_only` client to the newest committed version. Returns
    /// `True` when it advanced. A writer client always reads its own latest
    /// commit, so this is a no-op returning `False` there.
    fn refresh(&self, py: Python<'_>) -> PyResult<bool> {
        let Session::Reader(reader) = &self.session else {
            return Ok(false);
        };
        py.allow_threads(|| {
            self.runtime.block_on(async {
                // ponytail: reads wait on this lock while the WAL tail
                // replays; swap in the new snapshot outside it if refreshes
                // get slow enough to matter.
                let mut reader = reader.lock().await;
                reader.refresh().await.map_err(map_storage_err)
            })
        })
    }

    /// Release the retention lease a `read_only` client holds on the
    /// version it serves, so the writer's cleanup can reclaim it right away
    /// instead of when the lease expires. The client keeps working; its next
    /// `refresh()` that advances takes a new lease. A no-op on a writer.
    /// `with Client(...) as c:` calls this on exit.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.runtime.block_on(self.session.close()))
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }

    /// Stage a batch of node upserts that merge into the stored nodes. Each
    /// `row` is a `dict[str, Any]` keyed by `"_id"` (UUID string), or by the
    /// legacy `"id"`, which is then also kept as an `id` property. Every other
    /// key is merged into the node's current properties: unmentioned keys are
    /// kept, a `None` value removes the key, and `label` is added to the
    /// node's labels. Rows repeating an id in one batch merge in order.
    ///
    /// This reads each node before writing it; use [`Client::put_nodes`] for
    /// first loads. Returns the count of staged rows. Like
    /// [`Client::upsert_node`], the mutations are durable only after a
    /// subsequent [`Client::commit`].
    fn merge_nodes(
        &self,
        py: Python<'_>,
        label: &str,
        rows: &Bound<'_, PyList>,
    ) -> PyResult<usize> {
        self.stage_nodes(py, label, rows, true)
    }

    /// Stage a batch of node upserts that replace the stored nodes (no read).
    /// Rows take the same shape as [`Client::merge_nodes`], but each row is
    /// the node's full property set and `label` its only label.
    fn put_nodes(&self, py: Python<'_>, label: &str, rows: &Bound<'_, PyList>) -> PyResult<usize> {
        self.stage_nodes(py, label, rows, false)
    }

    /// Stage a batch of edge upserts. Each row needs `"src"` and `"dst"`
    /// UUID strings; remaining keys become edge properties. Like
    /// [`Client::upsert_edge`], the mutations are durable only after a
    /// subsequent [`Client::commit`].
    fn merge_edges(
        &self,
        py: Python<'_>,
        edge_type: &str,
        rows: &Bound<'_, PyList>,
    ) -> PyResult<usize> {
        let mut parsed: Vec<(NodeId, NodeId, BTreeMap<String, Value>)> =
            Vec::with_capacity(rows.len());
        for (idx, row_obj) in rows.iter().enumerate() {
            let row = row_obj
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err(format!("row #{idx} is not a dict")))?;
            let src_obj = row
                .get_item("src")?
                .ok_or_else(|| PyValueError::new_err(format!("row #{idx} is missing 'src'")))?;
            let dst_obj = row
                .get_item("dst")?
                .ok_or_else(|| PyValueError::new_err(format!("row #{idx} is missing 'dst'")))?;
            let src_str: String = src_obj.extract()?;
            let dst_str: String = dst_obj.extract()?;
            let src = parse_node_id(&src_str)?;
            let dst = parse_node_id(&dst_str)?;
            let mut props = BTreeMap::new();
            for (k, v) in row.iter() {
                let key: String = k.extract()?;
                if key == "src" || key == "dst" {
                    continue;
                }
                props.insert(key, py_any_to_value(&v)?);
            }
            parsed.push((src, dst, props));
        }
        let edge_type = edge_type.to_string();
        let session = self.session.writer()?.clone();
        let count = py.allow_threads(|| {
            self.runtime.block_on(async move {
                let mut guard = session.lock().await;
                for (src, dst, props) in &parsed {
                    let record = EdgeWriteRecord {
                        properties: props.clone(),
                        schema_version: 0,
                    };
                    guard
                        .upsert_edge(&edge_type, *src, *dst, &record)
                        .map_err(map_storage_err)?;
                }
                Ok::<_, PyErr>(parsed.len())
            })
        })?;
        Ok(count)
    }

    /// Scan every node under `label` and return the result as a
    /// `pyarrow.Table`. Columns: `id`, `label`, `lsn`,
    /// `schema_version`, then the union of property keys across the
    /// scanned views (missing keys are filled with `None`). `limit` stops
    /// after that many nodes.
    #[pyo3(signature = (label, limit=None))]
    fn scan_label_arrow<'py>(
        &self,
        py: Python<'py>,
        label: &str,
        limit: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let views = self.scan_nodes(py, label, limit)?;
        let mut prop_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for v in &views {
            for k in v.properties.keys() {
                prop_keys.insert(k.clone());
            }
        }
        let prop_keys: Vec<String> = prop_keys.into_iter().collect();
        build_node_views_table(py, &views, &prop_keys)
    }

    /// Load an Obsidian-style markdown vault into this namespace as a graph.
    ///
    /// Each `.md` note under `path` becomes a node (default label `Note`),
    /// each `[[wikilink]]` an edge (default type `LINKS_TO`), and YAML
    /// frontmatter becomes node properties; the raw note body is kept as a
    /// `body` property. Unlike `upsert_node` / `upsert_edge`, this commits
    /// the load before returning, so the graph is durable on exit.
    ///
    /// With `prune=True` the load mirrors the vault: notes and links removed
    /// from `path` since a previous load are tombstoned, so the graph stays a
    /// faithful index. The default (`prune=False`) is additive.
    ///
    /// Each note's string tags also become shared `:Tag` nodes linked by
    /// `:TAGGED` edges, so tag traversals run on the graph. With
    /// `placeholders=True`, links/embeds whose target has no real note get a
    /// stub `:Note` (`placeholder: true`) so unresolved references show up.
    ///
    /// Returns a dict with `notes_loaded`, `links_resolved`, `links_dangling`,
    /// `embeds_resolved`, `embeds_dangling`, `name_collisions`,
    /// `aliases_registered`, `notes_pruned`, `links_pruned`, `embeds_pruned`,
    /// `tags_loaded`, `tag_links`, `tags_pruned`, `tag_links_pruned`,
    /// `subtag_edges`, `subtag_edges_pruned`, `placeholders_created` and
    /// `commit_batches`.
    // Wide arity is inherent to the Python keyword-argument surface; the
    // `#[pyo3(signature = ...)]` above gives every extra parameter a default.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, label="Note", edge_type="LINKS_TO", commit_every=1000, prune=false, placeholders=false))]
    fn load_vault(
        &self,
        py: Python<'_>,
        path: &str,
        label: &str,
        edge_type: &str,
        commit_every: usize,
        prune: bool,
        placeholders: bool,
    ) -> PyResult<Py<PyDict>> {
        let opts = namidb_markdown::LoadOptions {
            label: label.to_string(),
            edge_type: edge_type.to_string(),
            commit_every,
            prune,
            placeholders,
            ..Default::default()
        };
        let dir = std::path::PathBuf::from(path);
        let session = self.session.writer()?.clone();
        let outcome = py.allow_threads(|| {
            self.runtime.block_on(async move {
                let mut guard = session.lock().await;
                let outcome = namidb_markdown::load_vault(&dir, &mut guard, &opts)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("vault load failed: {e}")))?;
                guard.commit_batch().await.map_err(map_storage_err)?;
                Ok::<_, PyErr>(outcome)
            })
        })?;

        let d = PyDict::new_bound(py);
        d.set_item("notes_loaded", outcome.notes_loaded)?;
        d.set_item("links_resolved", outcome.links_resolved)?;
        d.set_item("links_dangling", outcome.links_dangling)?;
        d.set_item("embeds_resolved", outcome.embeds_resolved)?;
        d.set_item("embeds_dangling", outcome.embeds_dangling)?;
        d.set_item("name_collisions", outcome.name_collisions)?;
        d.set_item("aliases_registered", outcome.aliases_registered)?;
        d.set_item("notes_pruned", outcome.notes_pruned)?;
        d.set_item("links_pruned", outcome.links_pruned)?;
        d.set_item("embeds_pruned", outcome.embeds_pruned)?;
        d.set_item("placeholders_created", outcome.placeholders_created)?;
        d.set_item("tags_loaded", outcome.tags_loaded)?;
        d.set_item("tag_links", outcome.tag_links)?;
        d.set_item("tags_pruned", outcome.tags_pruned)?;
        d.set_item("tag_links_pruned", outcome.tag_links_pruned)?;
        d.set_item("subtag_edges", outcome.subtag_edges)?;
        d.set_item("subtag_edges_pruned", outcome.subtag_edges_pruned)?;
        d.set_item("commit_batches", outcome.commit_batches)?;
        Ok(d.into())
    }

    /// Run a Cypher query synchronously and return a [`QueryResult`].
    ///
    /// `params` is an optional `dict[str, Any]` whose values are
    /// converted to [`namidb_query::RuntimeValue`] (`None`, `bool`,
    /// `int`, `float`, `str`, `bytes`, `list`, `dict`,
    /// `datetime.date`, `datetime.datetime` are all accepted; pass
    /// `list[float]` for vector parameters).
    ///
    /// Writes (CREATE / SET / DELETE / MERGE / REMOVE) are durably
    /// committed (WAL append + manifest CAS) **before this method
    /// returns** — the executor calls `commit_batch()` internally at
    /// the end of every write plan. Call `client.flush()` to push the
    /// memtable into L0 SSTs once enough writes have accumulated.
    ///
    /// Reads run on the latest committed snapshot without holding the
    /// writer lock, so several threads can query one client at once.
    ///
    /// `timeout_ms` aborts the query with `QueryTimeoutError` once that much
    /// wall time has passed since the call, waiting for the client's lock
    /// included. A write that times out is discarded, not committed.
    /// `max_rows` aborts a read with `QueryRowCapError` as soon as any step
    /// produces more rows than that; writes ignore it.
    #[pyo3(signature = (query, params=None, *, timeout_ms=None, max_rows=None))]
    fn cypher(
        &self,
        py: Python<'_>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
        max_rows: Option<usize>,
    ) -> PyResult<Py<QueryResult>> {
        let params = match params {
            Some(d) => py_dict_to_params(d)?,
            None => Params::new(),
        };
        let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms));
        let result: QueryResult = py.allow_threads(|| {
            self.runtime.block_on(run_cypher(
                &self.session,
                query,
                &params,
                self.cache.clone(),
                deadline,
                max_rows,
            ))
        })?;
        Py::new(py, result)
    }

    /// Async sibling of [`Client::cypher`]. Returns a Python coroutine
    /// that resolves to a [`QueryResult`].
    #[pyo3(signature = (query, params=None, *, timeout_ms=None, max_rows=None))]
    fn acypher<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
        timeout_ms: Option<u64>,
        max_rows: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = match params {
            Some(d) => py_dict_to_params(d)?,
            None => Params::new(),
        };
        let deadline = timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms));
        let query = query.to_string();
        let session = self.session.clone();
        let cache = self.cache.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = run_cypher(&session, &query, &params, cache, deadline, max_rows).await?;
            Python::with_gil(|py| Py::new(py, result))
        })
    }

    /// Return `(hits, misses, inserts, usage_bytes)` for the cache.
    fn cache_stats(&self) -> (u64, u64, u64, usize) {
        (
            self.cache.hits(),
            self.cache.misses(),
            self.cache.inserts(),
            self.cache.usage(),
        )
    }

    fn namespace_prefix(&self) -> String {
        self.paths.namespace_prefix().as_ref().to_string()
    }

    /// Inspect the absolute URI form of the underlying store, for
    /// debugging.
    fn store_repr(&self) -> String {
        format!("{:?}", self.store)
    }
}

impl Client {
    /// What an applied compaction should sweep afterwards, if anything.
    fn auto_sweep(&self) -> Option<AutoSweep> {
        let (_, reads) = self.session.writer_and_reads().ok()?;
        self.auto_sweep_min_age.map(|age| {
            (
                ManifestStore::new(self.store.clone(), self.paths.clone()),
                reads.clone(),
                age,
            )
        })
    }

    /// Shared body of [`Client::merge_nodes`] (`merge = true`) and
    /// [`Client::put_nodes`].
    fn stage_nodes(
        &self,
        py: Python<'_>,
        label: &str,
        rows: &Bound<'_, PyList>,
        merge: bool,
    ) -> PyResult<usize> {
        let mut parsed: Vec<(NodeId, BTreeMap<String, Value>)> = Vec::with_capacity(rows.len());
        for (idx, row_obj) in rows.iter().enumerate() {
            let row = row_obj
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err(format!("row #{idx} is not a dict")))?;
            // `_id` is the internal node id, as in Cypher. A legacy row keyed by
            // `id` still works, and its `id` stays a property so `n.id` reads it.
            let (id_key, id_obj) = match row.get_item("_id")? {
                Some(v) => ("_id", v),
                None => (
                    "id",
                    row.get_item("id")?.ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "row #{idx} is missing the required '_id' key (UUID string)"
                        ))
                    })?,
                ),
            };
            let id_str: String = id_obj.extract().map_err(|_| {
                PyValueError::new_err(format!("row #{idx} '{id_key}' is not a string"))
            })?;
            let id = parse_node_id(&id_str)?;
            let mut props = BTreeMap::new();
            for (k, v) in row.iter() {
                let key: String = k.extract()?;
                if key == "_id" {
                    continue;
                }
                props.insert(key, py_any_to_value(&v)?);
            }
            parsed.push((id, props));
        }
        let count = parsed.len();
        let label = label.to_string();
        let session = self.session.writer()?;
        let cache = self.cache.clone();
        py.allow_threads(|| {
            self.runtime.block_on(async move {
                let mut guard = session.lock().await;
                // Unique property names declared for this label (empty if none).
                let unique_props: Vec<String> = guard
                    .schema()
                    .label(&label)
                    .map(|d| {
                        d.properties
                            .iter()
                            .filter(|p| p.unique)
                            .map(|p| p.name.clone())
                            .collect()
                    })
                    .unwrap_or_default();
                let mut nodes: Vec<(NodeId, Vec<String>, BTreeMap<String, Value>)> =
                    Vec::with_capacity(count);
                // Rows repeating an id fold into one node, so the unique checks
                // below see only each id's final value.
                let mut slot: std::collections::HashMap<NodeId, usize> =
                    std::collections::HashMap::new();
                if merge {
                    // Read-modify-write against the RYOW overlay, so nodes staged
                    // by earlier calls count too.
                    let snap = guard.overlay_snapshot().with_cache(cache);
                    // One batched read warms the decoded row groups, so the
                    // per-id lookups below (label-agnostic, to union labels)
                    // stay in memory instead of decoding an SST per row.
                    let mut distinct: Vec<NodeId> = parsed.iter().map(|(id, _)| *id).collect();
                    distinct.sort_unstable();
                    distinct.dedup();
                    snap.batch_lookup_nodes(&label, &distinct)
                        .await
                        .map_err(map_storage_err)?;
                    for (id, props) in parsed {
                        let i = match slot.get(&id) {
                            Some(&i) => i,
                            None => {
                                let (labels, base) = match snap
                                    .lookup_node_by_id(id)
                                    .await
                                    .map_err(map_storage_err)?
                                {
                                    Some(n) => (n.labels.into_iter().collect(), n.properties),
                                    None => (Vec::new(), BTreeMap::new()),
                                };
                                slot.insert(id, nodes.len());
                                nodes.push((id, labels, base));
                                nodes.len() - 1
                            }
                        };
                        let (_, labels, current) = &mut nodes[i];
                        if !labels.contains(&label) {
                            labels.push(label.clone());
                        }
                        for (k, v) in props {
                            if matches!(v, Value::Null) {
                                current.remove(&k);
                            } else {
                                current.insert(k, v);
                            }
                        }
                    }
                } else {
                    // Replace semantics: the last row for an id wins.
                    for (id, props) in parsed {
                        match slot.get(&id) {
                            Some(&i) => nodes[i].2 = props,
                            None => {
                                slot.insert(id, nodes.len());
                                nodes.push((id, vec![label.clone()], props));
                            }
                        }
                    }
                }
                let mut seen: std::collections::HashMap<(String, String), NodeId> =
                    std::collections::HashMap::new();
                for (id, labels, props) in nodes {
                    // Committed + already-staged duplicates, via the RYOW overlay.
                    // The node's own row is excluded: this write replaces it.
                    namidb_query::enforce_node_unique_constraints(
                        &guard,
                        &labels,
                        &props,
                        Some(id),
                    )
                    .await
                    .map_err(PyValueError::new_err)?;
                    // Two rows in THIS batch carrying the same unique value.
                    for uprop in &unique_props {
                        if let Some(v) = props.get(uprop) {
                            let key = (uprop.clone(), format!("{v:?}"));
                            if let Some(prev) = seen.insert(key, id) {
                                if prev != id {
                                    return Err(PyValueError::new_err(format!(
                                        "{label}.{uprop} = {v:?} appears on two ids in the same \
                                     batch (unique constraint)"
                                    )));
                                }
                            }
                        }
                    }
                    let record = NodeWriteRecord {
                        properties: props,
                        schema_version: 0,
                        ..Default::default()
                    };
                    guard
                        .upsert_node_with_labels(labels, id, &record)
                        .map_err(map_storage_err)?;
                }
                Ok::<_, PyErr>(())
            })
        })?;
        Ok(count)
    }

    fn scan_nodes(
        &self,
        py: Python<'_>,
        label: &str,
        limit: Option<usize>,
    ) -> PyResult<Vec<NodeView>> {
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let owned = self.session.snapshot(None).await?;
                let snap = owned.borrow().with_cache(self.cache.clone());
                match limit {
                    Some(limit) => {
                        snap.scan_label_with_predicates_and_projection_limited(
                            label,
                            &[],
                            None,
                            limit,
                        )
                        .await
                    }
                    None => snap.scan_label(label).await,
                }
                .map_err(map_storage_err)
            })
        })
    }

    fn scan_edges(
        &self,
        py: Python<'_>,
        edge_type: &str,
        limit: Option<usize>,
    ) -> PyResult<Vec<EdgeView>> {
        let mut views = py.allow_threads(|| {
            self.runtime.block_on(async {
                let owned = self.session.snapshot(None).await?;
                let snap = owned.borrow().with_cache(self.cache.clone());
                snap.scan_edge_type(edge_type)
                    .await
                    .map_err(map_storage_err)
            })
        })?;
        // ponytail: storage has no limited edge scan, so `limit` bounds what
        // reaches Python, not what is read. Add a limited merge in read.rs
        // when edge scans need to stop early.
        if let Some(limit) = limit {
            views.truncate(limit);
        }
        Ok(views)
    }
}

/// Iterator returned by [`Client::iter_label`]. Each step converts one
/// node to a dict, so Python never holds the whole scan as dicts.
#[pyclass(module = "namidb")]
pub struct NodeIter {
    views: std::vec::IntoIter<NodeView>,
}

#[pymethods]
impl NodeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.views
            .next()
            .map(|v| node_view_to_py(py, &v))
            .transpose()
    }
}

/// Iterator returned by [`Client::iter_edge_type`].
#[pyclass(module = "namidb")]
pub struct EdgeIter {
    views: std::vec::IntoIter<EdgeView>,
}

#[pymethods]
impl EdgeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.views
            .next()
            .map(|v| edge_view_to_py(py, &v))
            .transpose()
    }
}

// ── Embedded maintenance ──────────────────────────────────────────────

#[derive(Debug)]
struct CompactionReport {
    applied: bool,
    manifest_version_before: u64,
    manifest_version_after: u64,
    l0_before: usize,
    l0_after: usize,
    source_ssts_removed: usize,
    new_ssts_written: usize,
    bloom_sidecars_written: usize,
}

/// Compact one immutable basis without retaining the foreground writer lock.
///
/// A separate mutex serialises compaction passes, while ordinary writes remain
/// free to acquire `session` throughout the expensive prepare phase. Storage's
/// install step folds those newer writes/flushes into the current manifest.
async fn run_compaction_inner(
    session: Arc<Mutex<WriterSession>>,
    compaction_lock: Arc<Mutex<()>>,
    auto_sweep: Option<AutoSweep>,
) -> PyResult<CompactionReport> {
    let _compaction_guard = compaction_lock.lock().await;
    let basis = {
        let guard = session.lock().await;
        guard.compaction_basis()
    };
    let manifest_version_before = basis.manifest_version();
    let l0_before = basis.max_l0_bucket_len();

    if !basis.needs_compaction() {
        // Re-sample after re-acquiring the writer: a concurrent foreground
        // flush may have advanced the manifest while this no-op pass ran.
        let guard = session.lock().await;
        return Ok(CompactionReport {
            applied: false,
            manifest_version_before,
            manifest_version_after: guard.snapshot().manifest().manifest.version,
            l0_before,
            l0_after: guard.max_l0_bucket_len(),
            source_ssts_removed: 0,
            new_ssts_written: 0,
            bloom_sidecars_written: 0,
        });
    }

    let schema = basis.schema().clone();
    // No WriterSession guard is alive across this await: all expensive object
    // I/O, CPU merging, and vector/text index construction happen off-lock.
    let prepared = basis.prepare(&schema).await.map_err(map_storage_err)?;

    // Let already-ready foreground work join the FIFO writer queue before the
    // brief install section, mirroring the server maintenance loop.
    tokio::task::yield_now().await;
    let mut guard = session.lock().await;
    let outcome = guard
        .install_prepared_compaction(prepared)
        .await
        .map_err(map_storage_err)?;
    let l0_after = guard.max_l0_bucket_len();
    let applied = outcome.source_ssts_removed > 0;
    if let Some((manifest_store, reads, min_age)) = auto_sweep.filter(|_| applied) {
        // The merged-away inputs and the WAL earlier flushes retired stay on
        // disk until a sweep. The pass is already installed, so a failed sweep
        // only leaves them for the next one.
        let _ = sweep_locked(&guard, &reads, &manifest_store, min_age, true).await;
    }

    Ok(CompactionReport {
        applied,
        manifest_version_before,
        manifest_version_after: outcome.committed.manifest.version,
        l0_before,
        l0_after,
        source_ssts_removed: outcome.source_ssts_removed,
        new_ssts_written: outcome.new_ssts_written,
        bloom_sidecars_written: outcome.bloom_sidecars_written,
    })
}

/// What an applied compaction sweeps with: the store, the writer's read
/// registry and the orphan age.
type AutoSweep = (ManifestStore, Arc<SnapshotCell>, Duration);

/// Sweep orphaned objects. The caller holds both the compaction lock and the
/// writer guard, so no prepared compaction output is waiting to be installed
/// and no flush or commit sits between its object PUT and the manifest CAS.
/// Reads run on snapshots outside the writer guard, so the horizon is the
/// oldest version a read in this process still holds (`reads` registers each
/// one under the guard), or the current version when none is running.
/// Readers in other processes are covered only by pin leases: `min_age`
/// counts from an object's write, not from when it stopped being referenced.
async fn sweep_locked(
    guard: &WriterSession,
    reads: &SnapshotCell,
    manifest_store: &ManifestStore,
    min_age: Duration,
    delete: bool,
) -> PyResult<JanitorReport> {
    reads.store(guard.owned_snapshot());
    let horizon = reads.retention_horizon();
    sweep_orphans(manifest_store, horizon, min_age, 1, delete)
        .await
        .map_err(map_storage_err)
}

/// Point compaction scratch at `spool_dir`, else `NAMIDB_SPOOL_DIR`, else
/// `<root>/.namidb-spool` for a `file://` URI, so a big compaction fills the
/// graph volume instead of `/var/tmp`. The leading dot keeps it from ever
/// parsing as a namespace. A default that cannot be created (a read-only
/// root) keeps the platform fallback; an explicit `spool_dir` must work.
/// The setting is process-wide and every client resets it, so a later
/// `memory://` or `s3://` client does not inherit a `file://` graph's dir.
fn configure_spool_dir(uri: &str, spool_dir: Option<String>) -> PyResult<()> {
    let explicit = spool_dir.is_some();
    let dir = spool_dir
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("NAMIDB_SPOOL_DIR")
                .filter(|v| !v.is_empty())
                .map(std::path::PathBuf::from)
        })
        .or_else(|| {
            let rest = uri.strip_prefix("file://")?;
            let root = rest.split_once('?').map_or(rest, |(path, _)| path);
            Some(std::path::Path::new(root).join(".namidb-spool"))
        });
    let dir = match dir {
        Some(dir) => match std::fs::create_dir_all(&dir) {
            Ok(()) => Some(dir),
            Err(e) if explicit => {
                return Err(PyValueError::new_err(format!(
                    "spool_dir {}: {e}",
                    dir.display()
                )))
            }
            Err(_) => None,
        },
        None => None,
    };
    namidb_storage::set_default_spool_dir(dir);
    Ok(())
}

fn compaction_report_to_py(py: Python<'_>, report: &CompactionReport) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("applied", report.applied)?;
    d.set_item("manifest_version_before", report.manifest_version_before)?;
    d.set_item("manifest_version_after", report.manifest_version_after)?;
    d.set_item("l0_before", report.l0_before)?;
    d.set_item("l0_after", report.l0_after)?;
    d.set_item("source_ssts_removed", report.source_ssts_removed)?;
    d.set_item("new_ssts_written", report.new_ssts_written)?;
    d.set_item("bloom_sidecars_written", report.bloom_sidecars_written)?;
    Ok(d.into())
}

// ── Cypher execution path (shared between sync + async) ───────────────

#[cfg(feature = "vector-index")]
fn vector_index_descriptor_from(
    cvi: &namidb_query::parser::ast::CreateVectorIndexClause,
) -> namidb_storage::manifest::VectorIndexDescriptor {
    use namidb_query::parser::ast::{VectorMetric as M, VectorQuantization as Q};

    let metric = match cvi.metric {
        M::Cosine => namidb_storage::manifest::VectorMetric::Cosine,
        M::Dot => namidb_storage::manifest::VectorMetric::Dot,
        M::Euclidean => namidb_storage::manifest::VectorMetric::Euclidean,
    };
    let quantization = match cvi.quantization {
        Q::None => namidb_storage::manifest::VectorQuantization::None,
        Q::Int8 => namidb_storage::manifest::VectorQuantization::Int8,
    };
    // Keep these defaults byte-for-byte aligned with the server and
    // `namidb_ann::BuildParams::default()`.
    namidb_storage::manifest::VectorIndexDescriptor {
        name: cvi.name.name.clone(),
        label: cvi.label.name.clone(),
        property: cvi.property.name.clone(),
        dim: cvi.dim,
        metric,
        r: cvi.r.unwrap_or(64),
        l_build: cvi.l_build.unwrap_or(128),
        alpha: cvi.alpha.unwrap_or(1.2),
        quantization,
    }
}

fn empty_query_result() -> QueryResult {
    QueryResult {
        columns: vec![],
        rows: vec![],
    }
}

/// Parse, plan and run one Cypher statement. Reads run on a committed
/// snapshot without the writer lock; schema DDL and write plans take the
/// lock and are planned again under it, because a plan built from the
/// lock-free snapshot can be stale by the time the lock is held.
async fn run_cypher(
    session: &Session,
    query: &str,
    params: &Params,
    cache: SstCache,
    deadline: Option<Instant>,
    row_cap: Option<usize>,
) -> PyResult<QueryResult> {
    let parsed = cypher_parse(query).map_err(parse_errs_to_pyerr)?;
    if !is_schema_ddl(&parsed) {
        let owned = session.snapshot(deadline).await?;
        let snap = owned.borrow().with_cache(cache);
        if let Some(result) = run_read(&parsed, &snap, params, deadline, row_cap).await? {
            return Ok(result);
        }
    }
    let writer = session.writer()?;
    let mut guard = lock_until(writer, deadline).await?;
    run_write(&mut guard, &parsed, params, deadline).await
}

fn is_schema_ddl(parsed: &Query) -> bool {
    parsed.as_create_vector_index().is_some()
        || parsed.as_create_fulltext_index().is_some()
        || parsed.as_drop_vector_index().is_some()
        || parsed.as_drop_fulltext_index().is_some()
        || parsed.as_create_constraint().is_some()
        || parsed.as_create_index().is_some()
}

/// The read side of [`run_cypher`]: SHOW, EXPLAIN and read plans. Returns
/// `None` when the statement writes.
async fn run_read(
    parsed: &Query,
    snap: &namidb_storage::Snapshot<'_>,
    params: &Params,
    deadline: Option<Instant>,
    row_cap: Option<usize>,
) -> PyResult<Option<QueryResult>> {
    let manifest = &snap.manifest().manifest;
    if let Some(c) = parsed.as_show_schema() {
        use namidb_query::parser::ast::ShowKind;
        let rows = match c.kind {
            ShowKind::Constraints => show_constraints_rows(&manifest.schema),
            ShowKind::Indexes => show_indexes_rows(manifest),
        };
        return Ok(Some(QueryResult {
            columns: show_schema_columns(),
            rows,
        }));
    }

    let catalog = StatsCatalog::from_manifest(manifest);
    let plan = build_plan(parsed, &catalog).map_err(lower_err_to_pyerr)?;

    // `EXPLAIN [RAW] [VERBOSE] <query>`: render the plan instead of
    // executing — the same rendering (tree + `# route:` footer) the server
    // serves. Without this check the embedded client silently EXECUTED the
    // query, so an `EXPLAIN CREATE ...` wrote data.
    if parsed.explain {
        let rows = namidb_query::explain_plan_lines(parsed, &plan, snap, &catalog)
            .into_iter()
            .map(|line| {
                let mut row = Row::new();
                row.set("plan", RuntimeValue::String(line));
                row
            })
            .collect();
        return Ok(Some(QueryResult {
            columns: vec!["plan".into()],
            rows,
        }));
    }
    if plan.contains_write() {
        return Ok(None);
    }

    let rows = execute_with_limits(&plan, snap, params, deadline, row_cap)
        .await
        .map_err(exec_err_to_pyerr)?;
    // One shared implementation across the embedded client, HTTP and Bolt:
    // three surfaces disagreeing about column order is its own bug.
    let columns = namidb_query::result_columns(&plan, &rows);
    Ok(Some(QueryResult { columns, rows }))
}

/// The write side of [`run_cypher`], run under the held writer guard.
async fn run_write(
    guard: &mut WriterSession,
    parsed: &Query,
    params: &Params,
    deadline: Option<Instant>,
) -> PyResult<QueryResult> {
    // Schema DDL / introspection is intercepted before planning — these clauses
    // never lower to a `LogicalPlan`. Running them as the sole statement here
    // lets the embedded client issue them directly (no Bolt/HTTP round-trip).
    #[cfg(feature = "vector-index")]
    if let Some(cvi) = parsed.as_create_vector_index() {
        let desc = vector_index_descriptor_from(cvi);
        guard
            .register_vector_index(desc, cvi.if_not_exists)
            .await
            .map_err(map_storage_err)?;
        return Ok(empty_query_result());
    }
    #[cfg(feature = "text-index")]
    if let Some(cfi) = parsed.as_create_fulltext_index() {
        let desc = namidb_storage::manifest::TextIndexDescriptor::new(
            cfi.name.name.clone(),
            cfi.label.name.clone(),
            cfi.properties.iter().map(|p| p.name.clone()).collect(),
        );
        guard
            .register_text_index(desc, cfi.if_not_exists)
            .await
            .map_err(map_storage_err)?;
        return Ok(empty_query_result());
    }
    #[cfg(feature = "vector-index")]
    if let Some(dvi) = parsed.as_drop_vector_index() {
        guard
            .drop_vector_index(&dvi.name.name, dvi.if_exists)
            .await
            .map_err(map_storage_err)?;
        return Ok(empty_query_result());
    }
    #[cfg(feature = "text-index")]
    if let Some(dfi) = parsed.as_drop_fulltext_index() {
        guard
            .drop_text_index(&dfi.name.name, dfi.if_exists)
            .await
            .map_err(map_storage_err)?;
        return Ok(empty_query_result());
    }
    if let Some(c) = parsed.as_create_constraint() {
        let properties: Vec<String> = c.properties.iter().map(|p| p.name.clone()).collect();
        guard
            .create_unique_constraint_named(
                c.name.as_ref().map(|n| n.name.as_str()),
                &c.label.name,
                &properties,
                c.if_not_exists,
            )
            .await
            .map_err(map_storage_err)?;
        return Ok(empty_query_result());
    }
    if let Some(c) = parsed.as_create_index() {
        let properties: Vec<String> = c.properties.iter().map(|p| p.name.clone()).collect();
        let name = c.name.as_ref().map(|n| n.name.as_str());
        if properties.len() > 1 {
            guard
                .create_composite_index_named(name, &c.label.name, &properties, c.if_not_exists)
                .await
                .map_err(map_storage_err)?;
        } else {
            guard
                .create_property_index_named(name, &c.label.name, &properties[0], c.if_not_exists)
                .await
                .map_err(map_storage_err)?;
        }
        return Ok(empty_query_result());
    }
    let catalog = StatsCatalog::from_manifest(&guard.snapshot().manifest().manifest);
    let plan = build_plan(parsed, &catalog).map_err(lower_err_to_pyerr)?;
    let rows = execute_write_with_deadline(&plan, guard, params, deadline)
        .await
        .map_err(exec_err_to_pyerr)?
        .rows;
    let columns = namidb_query::result_columns(&plan, &rows);
    Ok(QueryResult { columns, rows })
}

// ── QueryResult ───────────────────────────────────────────────────────

/// Result of a Cypher query. Wraps `Vec<namidb_query::Row>` and exposes
/// it to Python as a list-of-dicts and as Arrow / pandas /
/// polars.
#[pyclass(module = "namidb")]
pub struct QueryResult {
    columns: Vec<String>,
    rows: Vec<Row>,
}

#[pymethods]
impl QueryResult {
    /// Column names in the projection order of the first row. Empty
    /// when the query returned zero rows (Cypher does not surface a
    /// schema independently of the row-set in v0).
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    /// Number of rows in the result.
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryResult(rows={}, columns={:?})",
            self.rows.len(),
            self.columns
        )
    }

    /// Materialise the result as a `list[dict[str, Any]]`. Each row is
    /// a dict keyed by binding name; values are converted from
    /// [`RuntimeValue`] following the Cypher-to-Python mapping in the
    /// crate-level docstring.
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Py<PyList>> {
        let list = PyList::empty_bound(py);
        for row in &self.rows {
            list.append(row_to_py_dict(py, row, Some(&self.columns))?)?;
        }
        Ok(list.into())
    }

    /// First row as `dict[str, Any]` or `None` if the result is empty.
    fn first<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        match self.rows.first() {
            Some(r) => Ok(Some(row_to_py_dict(py, r, Some(&self.columns))?.into())),
            None => Ok(None),
        }
    }

    /// Materialise the result as a `pyarrow.Table`. Column types are
    /// inferred by `pyarrow.array` from the Python-side row values
    /// (`int → int64`, `float → float64`, `bool → bool_`, `str →
    /// string`, `bytes → binary`, `datetime.date → date32`,
    /// `datetime.datetime` UTC `→ timestamp[us, UTC]`, etc.). Columns
    /// holding `Node` / `Rel` / `Map` / `Path` values land as
    /// pyarrow `struct` / `list` types. Mixed-type columns may fail
    /// at inference time and propagate the pyarrow error as
    /// `RuntimeError`.
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pyarrow = import_pyarrow(py)?;
        let arrays = PyList::empty_bound(py);
        for col in &self.columns {
            let pylist = PyList::empty_bound(py);
            for row in &self.rows {
                let py_val = match row.bindings.get(col) {
                    Some(v) => runtime_value_to_py(py, v)?,
                    None => py.None(),
                };
                pylist.append(py_val)?;
            }
            let arr = pyarrow.call_method1("array", (pylist,))?;
            arrays.append(arr)?;
        }
        let table_cls = pyarrow.getattr("Table")?;
        let names = self.columns.clone();
        let table = table_cls.call_method1("from_arrays", (arrays, names))?;
        Ok(table.into())
    }

    /// `pandas.DataFrame` view of the result. Delegates to
    /// [`Self::to_arrow`] + `pyarrow.Table.to_pandas()`.
    fn to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let df = table.bind(py).call_method0("to_pandas")?;
        Ok(df.into())
    }

    /// `polars.DataFrame` view of the result. Delegates to
    /// [`Self::to_arrow`] + `polars.from_arrow(...)`. Requires the
    /// optional `polars` extra (`pip install 'namidb[polars]'`).
    fn to_polars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let polars = py.import_bound("polars").map_err(|e| {
            pyo3::exceptions::PyImportError::new_err(format!(
                "polars is required for QueryResult.to_polars(). \
 Install it with `pip install 'namidb[polars]'`. \
 Original error: {e}"
            ))
        })?;
        let table = self.to_arrow(py)?;
        let df = polars.call_method1("from_arrow", (table,))?;
        Ok(df.into())
    }
}

// ── URI parsing ───────────────────────────────────────────────────────

/// Thin wrapper over [`namidb_storage::parse_uri`] that maps
/// [`UriError`] variants onto pyo3 exceptions. See the storage crate
/// for the canonical scheme reference.
fn parse_uri(uri: &str) -> PyResult<(Arc<dyn ObjectStore>, namidb_storage::NamespacePaths)> {
    namidb_storage::parse_uri(uri).map_err(|e| match e {
        namidb_storage::UriError::BackendInit { .. } => PyRuntimeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    })
}

fn parse_node_id(id: &str) -> PyResult<NodeId> {
    let uuid = Uuid::parse_str(id)
        .map_err(|e| PyValueError::new_err(format!("invalid UUID '{id}': {e}")))?;
    Ok(NodeId::from_uuid(uuid))
}

// ── Value <-> Python conversions (storage-side Value) ─────────────────

fn py_dict_to_value_map(d: &Bound<'_, PyDict>) -> PyResult<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let value = py_any_to_value(&v)?;
        out.insert(key, value);
    }
    Ok(out)
}

fn py_any_to_value(v: &Bound<'_, PyAny>) -> PyResult<Value> {
    use pyo3::types::PyBytes;
    if v.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(Value::I64(i));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(Value::F64(f));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(Value::Str(s));
    }
    // Check actual Python `bytes` type rather than relying on `Vec<u8>`
    // extraction — a Python list-of-small-ints like `[1, 2, 3]` will
    // also extract as `Vec<u8>` and silently turn into Bytes.
    if v.is_instance_of::<PyBytes>() {
        return Ok(Value::Bytes(v.extract::<Vec<u8>>()?));
    }
    if let Ok(vec) = v.extract::<Vec<f32>>() {
        return Ok(Value::Vec(vec));
    }
    Err(PyValueError::new_err(format!(
        "unsupported Python value: {}",
        v.get_type().name()?
    )))
}

fn value_to_py(py: Python<'_>, v: &Value) -> PyResult<Py<PyAny>> {
    use pyo3::types::{PyBytes, PyFloat, PyString};
    use pyo3::IntoPy;
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::I64(i) => i.into_py(py),
        Value::F64(f) => PyFloat::new_bound(py, *f).into_any().unbind(),
        Value::Str(s) => PyString::new_bound(py, s).into_any().unbind(),
        Value::Bytes(b) => PyBytes::new_bound(py, b).into_any().unbind(),
        Value::Vec(v) => {
            let list = PyList::empty_bound(py);
            for x in v {
                list.append(*x)?;
            }
            list.into_any().unbind()
        }
        Value::VecI8 { codes, scale } => {
            // Dequantize int8 to floats so Python sees a float vector.
            let list = PyList::empty_bound(py);
            for &c in codes {
                list.append(c as f32 * *scale)?;
            }
            list.into_any().unbind()
        }
        Value::Date(days) => {
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("static epoch");
            let d = epoch + chrono::Duration::days(*days as i64);
            d.into_py(py)
        }
        Value::DateTime(micros) => {
            let secs = micros.div_euclid(1_000_000);
            let extra_nanos = (micros.rem_euclid(1_000_000) as u32) * 1000;
            let dt = Utc
                .timestamp_opt(secs, extra_nanos)
                .single()
                .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
            dt.into_py(py)
        }
        Value::List(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Map(entries) => {
            let dict = PyDict::new_bound(py);
            for (k, val) in entries {
                dict.set_item(k, value_to_py(py, val)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

fn node_view_to_py<'py>(py: Python<'py>, v: &NodeView) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("id", v.id.to_string())?;
    // `labels` is the full set; `label` stays as the representative (first)
    // label for backward compatibility with single-label consumers.
    d.set_item(
        "label",
        v.labels.iter().next().map(String::as_str).unwrap_or(""),
    )?;
    d.set_item("labels", v.labels.iter().cloned().collect::<Vec<String>>())?;
    d.set_item("lsn", v.lsn)?;
    d.set_item("schema_version", v.schema_version)?;
    let props = PyDict::new_bound(py);
    for (k, val) in &v.properties {
        props.set_item(k, value_to_py(py, val)?)?;
    }
    d.set_item("properties", props)?;
    Ok(d)
}

fn edge_view_to_py<'py>(py: Python<'py>, v: &EdgeView) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("edge_type", &v.edge_type)?;
    d.set_item("src", v.src.to_string())?;
    d.set_item("dst", v.dst.to_string())?;
    d.set_item("lsn", v.lsn)?;
    let props = PyDict::new_bound(py);
    for (k, val) in &v.properties {
        props.set_item(k, value_to_py(py, val)?)?;
    }
    d.set_item("properties", props)?;
    Ok(d)
}

fn edge_list_to_py<'py>(py: Python<'py>, list: &EdgeListView) -> PyResult<Py<PyList>> {
    let out = PyList::empty_bound(py);
    for e in &list.edges {
        out.append(edge_view_to_py(py, e)?)?;
    }
    Ok(out.into())
}

// ── RuntimeValue <-> Python conversions (query-side, richer set) ──────

fn py_dict_to_params(d: &Bound<'_, PyDict>) -> PyResult<Params> {
    let mut out = Params::new();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        out.insert(key, py_any_to_runtime_value(&v)?);
    }
    Ok(out)
}

fn py_any_to_runtime_value(v: &Bound<'_, PyAny>) -> PyResult<RuntimeValue> {
    use pyo3::types::PyBytes;
    if v.is_none() {
        return Ok(RuntimeValue::Null);
    }
    // bool MUST be checked before i64 — Python `True`/`False` extract
    // successfully as `i64` (1 / 0).
    if let Ok(b) = v.extract::<bool>() {
        return Ok(RuntimeValue::Bool(b));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(RuntimeValue::Integer(i));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(RuntimeValue::Float(f));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(RuntimeValue::String(s));
    }
    // Check actual Python `bytes` type — a Python list-of-small-ints
    // like `[1, 2, 3]` would otherwise extract as `Vec<u8>` and turn
    // into Bytes instead of List.
    if v.is_instance_of::<PyBytes>() {
        return Ok(RuntimeValue::Bytes(v.extract::<Vec<u8>>()?));
    }
    // datetime/date before list/dict because they are *also* iterable.
    if let Ok(dt) = v.extract::<DateTime<Utc>>() {
        return Ok(RuntimeValue::DateTime(dt.timestamp_micros()));
    }
    if let Ok(d) = v.extract::<NaiveDate>() {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("static epoch");
        let days = d.signed_duration_since(epoch).num_days() as i32;
        return Ok(RuntimeValue::Date(days));
    }
    if let Ok(list) = v.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_any_to_runtime_value(&item)?);
        }
        return Ok(RuntimeValue::List(out));
    }
    if let Ok(dict) = v.downcast::<PyDict>() {
        let mut out = BTreeMap::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            out.insert(key, py_any_to_runtime_value(&v)?);
        }
        return Ok(RuntimeValue::Map(out));
    }
    Err(PyValueError::new_err(format!(
        "unsupported Python value for Cypher parameter: {}",
        v.get_type().name()?
    )))
}

fn runtime_value_to_py(py: Python<'_>, v: &RuntimeValue) -> PyResult<Py<PyAny>> {
    use pyo3::types::{PyBytes, PyFloat, PyString};
    use pyo3::IntoPy;
    Ok(match v {
        RuntimeValue::Null => py.None(),
        RuntimeValue::Bool(b) => b.into_py(py),
        RuntimeValue::Integer(i) => i.into_py(py),
        RuntimeValue::Float(f) => PyFloat::new_bound(py, *f).into_any().unbind(),
        RuntimeValue::String(s) => PyString::new_bound(py, s).into_any().unbind(),
        RuntimeValue::Bytes(b) => PyBytes::new_bound(py, b).into_any().unbind(),
        RuntimeValue::Vector(v) => {
            let list = PyList::empty_bound(py);
            for x in v {
                list.append(*x)?;
            }
            list.into_any().unbind()
        }
        RuntimeValue::Vector8 { codes, scale } => {
            // Dequantize int8 to floats so Python sees a float vector.
            let list = PyList::empty_bound(py);
            for &c in codes {
                list.append(c as f32 * *scale)?;
            }
            list.into_any().unbind()
        }
        RuntimeValue::List(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                list.append(runtime_value_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        RuntimeValue::Map(m) => {
            let d = PyDict::new_bound(py);
            for (k, val) in m {
                d.set_item(k, runtime_value_to_py(py, val)?)?;
            }
            d.into_any().unbind()
        }
        RuntimeValue::Node(n) => node_value_to_py(py, n)?.into_any().unbind(),
        RuntimeValue::Rel(r) => rel_value_to_py(py, r)?.into_any().unbind(),
        RuntimeValue::Date(days) => {
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("static epoch");
            let d = epoch + chrono::Duration::days(*days as i64);
            d.into_py(py)
        }
        RuntimeValue::DateTime(micros) => {
            let secs = micros.div_euclid(1_000_000);
            let extra_nanos = (micros.rem_euclid(1_000_000) as u32) * 1000;
            let dt = Utc
                .timestamp_opt(secs, extra_nanos)
                .single()
                .unwrap_or_else(|| Utc.timestamp_opt(0, 0).unwrap());
            dt.into_py(py)
        }
        RuntimeValue::Path(items) => {
            let list = PyList::empty_bound(py);
            for item in items {
                list.append(runtime_value_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
    })
}

fn node_value_to_py<'py>(py: Python<'py>, n: &NodeValue) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("_kind", "node")?;
    d.set_item("id", n.id.to_string())?;
    // `label` = representative (first) for back-compat; `labels` = full set.
    d.set_item(
        "label",
        n.labels.iter().next().map(String::as_str).unwrap_or(""),
    )?;
    d.set_item("labels", n.labels.iter().cloned().collect::<Vec<String>>())?;
    let props = PyDict::new_bound(py);
    for (k, val) in &n.properties {
        props.set_item(k, runtime_value_to_py(py, val)?)?;
    }
    d.set_item("properties", props)?;
    Ok(d)
}

fn rel_value_to_py<'py>(py: Python<'py>, r: &RelValue) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("_kind", "rel")?;
    d.set_item("edge_type", &r.edge_type)?;
    d.set_item("src", r.src.to_string())?;
    d.set_item("dst", r.dst.to_string())?;
    let props = PyDict::new_bound(py);
    for (k, val) in &r.properties {
        props.set_item(k, runtime_value_to_py(py, val)?)?;
    }
    d.set_item("properties", props)?;
    Ok(d)
}

fn row_to_py_dict<'py>(
    py: Python<'py>,
    row: &Row,
    columns: Option<&[String]>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    if let Some(cols) = columns {
        for col in cols {
            let py_val = match row.bindings.get(col) {
                Some(v) => runtime_value_to_py(py, v)?,
                None => py.None(),
            };
            d.set_item(col, py_val)?;
        }
    } else {
        for (k, v) in &row.bindings {
            d.set_item(k, runtime_value_to_py(py, v)?)?;
        }
    }
    Ok(d)
}

// ── Error mapping ─────────────────────────────────────────────────────

fn map_storage_err(e: namidb_storage::Error) -> PyErr {
    PyRuntimeError::new_err(format!("namidb storage error: {e}"))
}

fn parse_errs_to_pyerr(errs: Vec<ParseError>) -> PyErr {
    match errs.first() {
        Some(first) => PyValueError::new_err(format!(
            "Cypher parse error [{:?}]: {} at {}",
            first.code, first.message, first.span
        )),
        None => PyValueError::new_err("Cypher parse error (empty diagnostic list)"),
    }
}

fn lower_err_to_pyerr(e: LowerError) -> PyErr {
    PyValueError::new_err(format!("Cypher lower error: {e}"))
}

fn exec_err_to_pyerr(e: ExecError) -> PyErr {
    match e {
        // Python has no admin cancel path, so a cancel can only come from
        // the same deadline machinery.
        // A deadline that fires in a write's commit tail arrives as a storage
        // error.
        ExecError::Timeout
        | ExecError::Cancelled
        | ExecError::Storage(namidb_storage::Error::Timeout | namidb_storage::Error::Cancelled) => {
            QueryTimeoutError::new_err(e.to_string())
        }
        ExecError::RowCap(_) => QueryRowCapError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(format!("Cypher execution error: {e}")),
    }
}

// ── pyarrow helpers ───────────────────────────────────────────────────

fn import_pyarrow(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import_bound("pyarrow").map_err(|e| {
        pyo3::exceptions::PyImportError::new_err(format!(
            "pyarrow is required for Arrow / pandas / polars output. \
 It is a hard dependency of `namidb` — if you are seeing this, \
 your environment is broken. Original error: {e}"
        ))
    })
}

/// Shared table-builder used by [`Client::scan_label_arrow`]. Caller
/// passes the pre-computed union of property keys across `views`;
/// missing keys are filled with `None`. Metadata columns (`id`,
/// `label`, `lsn`, `schema_version`) come first, then property
/// columns in `prop_keys` order.
fn build_node_views_table(
    py: Python<'_>,
    views: &[NodeView],
    prop_keys: &[String],
) -> PyResult<Py<PyAny>> {
    let pyarrow = import_pyarrow(py)?;
    let arrays = PyList::empty_bound(py);
    let mut all_names: Vec<String> = vec![
        "id".to_string(),
        "label".to_string(),
        "labels".to_string(),
        "lsn".to_string(),
        "schema_version".to_string(),
    ];
    // A property named like a metadata column (`id` kept from a legacy row
    // keyed by `id`, or a business key) would make the schema ambiguous.
    // Drop it when it only echoes the node id, else rename it `property.<key>`.
    let prop_columns: Vec<(String, &String)> = prop_keys
        .iter()
        .filter_map(|key| {
            if !all_names.contains(key) {
                return Some((key.clone(), key));
            }
            // Compared as parsed ids: a legacy row may spell its UUID in
            // upper case, braced or without hyphens.
            let echoes_id = key == "id"
                && views.iter().all(|v| match v.properties.get(key) {
                    None => true,
                    Some(Value::Str(s)) => s.parse::<NodeId>().is_ok_and(|id| id == v.id),
                    Some(_) => false,
                });
            (!echoes_id).then(|| (format!("property.{key}"), key))
        })
        .collect();
    all_names.extend(prop_columns.iter().map(|(name, _)| name.clone()));

    let ids = PyList::empty_bound(py);
    for v in views {
        ids.append(v.id.to_string())?;
    }
    arrays.append(pyarrow.call_method1("array", (ids,))?)?;

    // `label` (representative, back-compat) and `labels` (full set as a
    // list<string> column) sit side by side.
    let labels = PyList::empty_bound(py);
    for v in views {
        labels.append(v.labels.iter().next().map(String::as_str).unwrap_or(""))?;
    }
    arrays.append(pyarrow.call_method1("array", (labels,))?)?;

    let label_lists = PyList::empty_bound(py);
    for v in views {
        let inner = PyList::empty_bound(py);
        for l in &v.labels {
            inner.append(l.as_str())?;
        }
        label_lists.append(inner)?;
    }
    arrays.append(pyarrow.call_method1("array", (label_lists,))?)?;

    let lsns = PyList::empty_bound(py);
    for v in views {
        lsns.append(v.lsn)?;
    }
    arrays.append(pyarrow.call_method1("array", (lsns,))?)?;

    let svs = PyList::empty_bound(py);
    for v in views {
        svs.append(v.schema_version)?;
    }
    arrays.append(pyarrow.call_method1("array", (svs,))?)?;

    for (_, key) in &prop_columns {
        let pylist = PyList::empty_bound(py);
        for v in views {
            let pyval = match v.properties.get(*key) {
                Some(val) => value_to_py(py, val)?,
                None => py.None(),
            };
            pylist.append(pyval)?;
        }
        arrays.append(pyarrow.call_method1("array", (pylist,))?)?;
    }

    let table_cls = pyarrow.getattr("Table")?;
    let table = table_cls.call_method1("from_arrays", (arrays, all_names))?;
    Ok(table.into())
}

/// Surface every public name on the Rust extension; the wrapper at
/// `python/namidb/__init__.py` re-exports them as `namidb.<name>`.
#[pymodule]
fn _lib(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<QueryResult>()?;
    m.add_class::<NodeIter>()?;
    m.add_class::<EdgeIter>()?;
    m.add(
        "QueryTimeoutError",
        py.get_type_bound::<QueryTimeoutError>(),
    )?;
    m.add("QueryRowCapError", py.get_type_bound::<QueryRowCapError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
