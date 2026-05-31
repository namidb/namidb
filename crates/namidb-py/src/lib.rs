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

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use object_store::ObjectStore;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use uuid::Uuid;

use namidb_core::{NodeId, Value};
use namidb_query::exec::{NodeValue, RelValue};
use namidb_query::{
    execute, execute_write, parse as cypher_parse, plan as build_plan, ExecError, LogicalPlan,
    LowerError, Params, ParseError, Row, RuntimeValue, StatsCatalog,
};
use namidb_storage::{
    CommitOutcome, EdgeListView, EdgeView, EdgeWriteRecord, NodeView, NodeWriteRecord, SstCache,
    WriterSession,
};

/// A namespace handle: object store + writer session + a tokio runtime
/// that drives every async storage call from synchronous Python.
#[pyclass(module = "namidb")]
pub struct Client {
    runtime: Runtime,
    session: Arc<Mutex<WriterSession>>,
    store: Arc<dyn ObjectStore>,
    paths: namidb_storage::NamespacePaths,
    cache: SstCache,
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
    #[new]
    #[pyo3(signature = (uri, cache_bytes=None))]
    fn new(uri: &str, cache_bytes: Option<usize>) -> PyResult<Self> {
        let (store, paths) = parse_uri(uri)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
        let session = runtime
            .block_on(WriterSession::open(store.clone(), paths.clone()))
            .map_err(map_storage_err)?;
        let cache = SstCache::new(cache_bytes.unwrap_or(64 * 1024 * 1024));
        Ok(Self {
            runtime,
            session: Arc::new(Mutex::new(session)),
            store,
            paths,
            cache,
        })
    }

    /// Stage a node upsert into the current batch.
    fn upsert_node(&self, label: &str, id: &str, properties: &Bound<'_, PyDict>) -> PyResult<()> {
        let id = parse_node_id(id)?;
        let props = py_dict_to_value_map(properties)?;
        let record = NodeWriteRecord {
            properties: props,
            schema_version: 0,
        };
        self.runtime.block_on(async {
            let mut session = self.session.lock().await;
            session
                .upsert_node(label, id, &record)
                .map_err(map_storage_err)
        })?;
        Ok(())
    }

    /// Stage a node tombstone into the current batch.
    fn tombstone_node(&self, label: &str, id: &str) -> PyResult<()> {
        let id = parse_node_id(id)?;
        self.runtime.block_on(async {
            let mut session = self.session.lock().await;
            session.tombstone_node(label, id).map_err(map_storage_err)
        })?;
        Ok(())
    }

    /// Stage an edge upsert into the current batch.
    fn upsert_edge(
        &self,
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
        self.runtime.block_on(async {
            let mut session = self.session.lock().await;
            session
                .upsert_edge(edge_type, src, dst, &record)
                .map_err(map_storage_err)
        })?;
        Ok(())
    }

    /// Stage an edge tombstone into the current batch.
    fn tombstone_edge(&self, edge_type: &str, src: &str, dst: &str) -> PyResult<()> {
        let src = parse_node_id(src)?;
        let dst = parse_node_id(dst)?;
        self.runtime.block_on(async {
            let mut session = self.session.lock().await;
            session
                .tombstone_edge(edge_type, src, dst)
                .map_err(map_storage_err)
        })?;
        Ok(())
    }

    /// Durably commit the current batch (WAL append + manifest CAS).
    /// Returns the last LSN persisted, or `None` when the batch was empty.
    fn commit(&self) -> PyResult<Option<u64>> {
        let last_lsn = self.runtime.block_on(async {
            let mut session = self.session.lock().await;
            let outcome = session.commit_batch().await.map_err(map_storage_err)?;
            Ok::<Option<u64>, PyErr>(match outcome {
                CommitOutcome::Empty => None,
                CommitOutcome::Committed { last_lsn, .. } => Some(last_lsn),
            })
        })?;
        Ok(last_lsn)
    }

    /// Flush the memtable into L0 SSTs.
    fn flush(&self) -> PyResult<()> {
        self.runtime.block_on(async {
            let mut session = self.session.lock().await;
            let schema = session.snapshot().manifest().manifest.schema.clone();
            session.flush(schema).await.map_err(map_storage_err)
        })?;
        Ok(())
    }

    /// Look up a single node by `(label, id)`.
    fn lookup_node<'py>(
        &self,
        py: Python<'py>,
        label: &str,
        id: &str,
    ) -> PyResult<Option<Py<PyDict>>> {
        let id = parse_node_id(id)?;
        let view: Option<NodeView> = self.runtime.block_on(async {
            let session = self.session.lock().await;
            let snap = session.snapshot().with_cache(self.cache.clone());
            snap.lookup_node(label, id).await.map_err(map_storage_err)
        })?;
        match view {
            Some(v) => Ok(Some(node_view_to_py(py, &v)?.into())),
            None => Ok(None),
        }
    }

    /// All outgoing edges of `(edge_type, src)`.
    fn out_edges<'py>(&self, py: Python<'py>, edge_type: &str, src: &str) -> PyResult<Py<PyList>> {
        let src = parse_node_id(src)?;
        let list: EdgeListView = self.runtime.block_on(async {
            let session = self.session.lock().await;
            let snap = session.snapshot().with_cache(self.cache.clone());
            snap.out_edges(edge_type, src)
                .await
                .map_err(map_storage_err)
        })?;
        edge_list_to_py(py, &list)
    }

    /// All incoming edges of `(edge_type, dst)`.
    fn in_edges<'py>(&self, py: Python<'py>, edge_type: &str, dst: &str) -> PyResult<Py<PyList>> {
        let dst = parse_node_id(dst)?;
        let list: EdgeListView = self.runtime.block_on(async {
            let session = self.session.lock().await;
            let snap = session.snapshot().with_cache(self.cache.clone());
            snap.in_edges(edge_type, dst).await.map_err(map_storage_err)
        })?;
        edge_list_to_py(py, &list)
    }

    /// All nodes under `label` (memtable + SSTs, last-write-wins).
    fn scan_label<'py>(&self, py: Python<'py>, label: &str) -> PyResult<Py<PyList>> {
        let views: Vec<NodeView> = self.runtime.block_on(async {
            let session = self.session.lock().await;
            let snap = session.snapshot().with_cache(self.cache.clone());
            snap.scan_label(label).await.map_err(map_storage_err)
        })?;
        let list = PyList::empty_bound(py);
        for v in &views {
            list.append(node_view_to_py(py, v)?)?;
        }
        Ok(list.into())
    }

    /// All edges of `edge_type` (memtable + SSTs, last-write-wins).
    fn scan_edge_type<'py>(&self, py: Python<'py>, edge_type: &str) -> PyResult<Py<PyList>> {
        let views: Vec<EdgeView> = self.runtime.block_on(async {
            let session = self.session.lock().await;
            let snap = session.snapshot().with_cache(self.cache.clone());
            snap.scan_edge_type(edge_type)
                .await
                .map_err(map_storage_err)
        })?;
        let list = PyList::empty_bound(py);
        for v in &views {
            list.append(edge_view_to_py(py, v)?)?;
        }
        Ok(list.into())
    }

    /// Stage a batch of node upserts under a single tokio runtime + lock
    /// hop. Each `row` is a `dict[str, Any]` that must contain a string
    /// `"id"` (UUID); all other keys become node properties.
    ///
    /// Returns the count of staged rows. Like [`Client::upsert_node`],
    /// the mutations are durable only after a subsequent
    /// [`Client::commit`].
    fn merge_nodes(&self, label: &str, rows: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut parsed: Vec<(NodeId, BTreeMap<String, Value>)> = Vec::with_capacity(rows.len());
        for (idx, row_obj) in rows.iter().enumerate() {
            let row = row_obj
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err(format!("row #{idx} is not a dict")))?;
            let id_obj = row.get_item("id")?.ok_or_else(|| {
                PyValueError::new_err(format!(
                    "row #{idx} is missing the required 'id' key (UUID string)"
                ))
            })?;
            let id_str: String = id_obj
                .extract()
                .map_err(|_| PyValueError::new_err(format!("row #{idx} 'id' is not a string")))?;
            let id = parse_node_id(&id_str)?;
            let mut props = BTreeMap::new();
            for (k, v) in row.iter() {
                let key: String = k.extract()?;
                if key == "id" {
                    continue;
                }
                props.insert(key, py_any_to_value(&v)?);
            }
            parsed.push((id, props));
        }
        let label = label.to_string();
        let session = self.session.clone();
        let count = self.runtime.block_on(async move {
            let mut guard = session.lock().await;
            for (id, props) in &parsed {
                let record = NodeWriteRecord {
                    properties: props.clone(),
                    schema_version: 0,
                };
                guard
                    .upsert_node(&label, *id, &record)
                    .map_err(map_storage_err)?;
            }
            Ok::<_, PyErr>(parsed.len())
        })?;
        Ok(count)
    }

    /// Stage a batch of edge upserts. Each row needs `"src"` and `"dst"`
    /// UUID strings; remaining keys become edge properties. Like
    /// [`Client::upsert_edge`], the mutations are durable only after a
    /// subsequent [`Client::commit`].
    fn merge_edges(&self, edge_type: &str, rows: &Bound<'_, PyList>) -> PyResult<usize> {
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
        let session = self.session.clone();
        let count = self.runtime.block_on(async move {
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
        })?;
        Ok(count)
    }

    /// Scan every node under `label` and return the result as a
    /// `pyarrow.Table`. Columns: `id`, `label`, `lsn`,
    /// `schema_version`, then the union of property keys across the
    /// scanned views (missing keys are filled with `None`).
    fn scan_label_arrow<'py>(&self, py: Python<'py>, label: &str) -> PyResult<Py<PyAny>> {
        let views: Vec<NodeView> = self.runtime.block_on(async {
            let session = self.session.lock().await;
            let snap = session.snapshot().with_cache(self.cache.clone());
            snap.scan_label(label).await.map_err(map_storage_err)
        })?;
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
        };
        let dir = std::path::PathBuf::from(path);
        let session = self.session.clone();
        let outcome = self.runtime.block_on(async move {
            let mut guard = session.lock().await;
            let outcome = namidb_markdown::load_vault(&dir, &mut guard, &opts)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("vault load failed: {e}")))?;
            guard.commit_batch().await.map_err(map_storage_err)?;
            Ok::<_, PyErr>(outcome)
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
    #[pyo3(signature = (query, params=None))]
    fn cypher(
        &self,
        py: Python<'_>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<QueryResult>> {
        let params = match params {
            Some(d) => py_dict_to_params(d)?,
            None => Params::new(),
        };
        let session = self.session.clone();
        let cache = self.cache.clone();
        let query_owned = query.to_string();
        let result: QueryResult = self.runtime.block_on(async move {
            let mut guard = session.lock().await;
            run_cypher_inner(&mut guard, &query_owned, &params, cache).await
        })?;
        Py::new(py, result)
    }

    /// Async sibling of [`Client::cypher`]. Returns a Python coroutine
    /// that resolves to a [`QueryResult`].
    #[pyo3(signature = (query, params=None))]
    fn acypher<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = match params {
            Some(d) => py_dict_to_params(d)?,
            None => Params::new(),
        };
        let query = query.to_string();
        let session = self.session.clone();
        let cache = self.cache.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = session.lock().await;
            let result = run_cypher_inner(&mut guard, &query, &params, cache).await?;
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

// ── Cypher execution path (shared between sync + async) ───────────────

/// Drive a Cypher query under a held `WriterSession` guard. Caller is
/// responsible for acquiring the `tokio::sync::Mutex` lock; this
/// helper just performs parse + lower + optimize + execute, then
/// extracts the row-set + projected column list.
async fn run_cypher_inner(
    guard: &mut WriterSession,
    query: &str,
    params: &Params,
    cache: SstCache,
) -> PyResult<QueryResult> {
    let parsed = cypher_parse(query).map_err(parse_errs_to_pyerr)?;
    let catalog = StatsCatalog::from_manifest(&guard.snapshot().manifest().manifest);
    let plan = build_plan(&parsed, &catalog).map_err(lower_err_to_pyerr)?;
    let plan_columns = extract_column_order(&plan);
    let rows = if plan.contains_write() {
        let outcome = execute_write(&plan, guard, params)
            .await
            .map_err(exec_err_to_pyerr)?;
        outcome.rows
    } else {
        let snap = guard.snapshot().with_cache(cache);
        execute(&plan, &snap, params)
            .await
            .map_err(exec_err_to_pyerr)?
    };
    // Prefer the column order declared by the RETURN / WITH projection;
    // fall back to the first row's BTreeMap order if the plan top is
    // some shape we don't recognise (e.g. a future operator).
    let columns: Vec<String> = plan_columns.unwrap_or_else(|| {
        rows.first()
            .map(|r| r.bindings.keys().cloned().collect())
            .unwrap_or_default()
    });
    Ok(QueryResult { columns, rows })
}

/// Walk down the plan top through order-preserving wrappers
/// (`Distinct`, `TopN`) until we hit a `Project` and return its
/// projected alias order. Returns `None` if no `Project` is reached
/// (which would be unusual for a lowered Cypher query — the parser
/// only emits `RETURN` clauses, and `RETURN` always lowers to
/// `Project`).
fn extract_column_order(plan: &LogicalPlan) -> Option<Vec<String>> {
    let mut current = plan;
    loop {
        match current {
            LogicalPlan::Project { items, .. } => {
                return Some(items.iter().map(|i| i.alias.clone()).collect());
            }
            LogicalPlan::Distinct { input } => current = input,
            LogicalPlan::TopN { input, .. } => current = input,
            _ => return None,
        }
    }
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
    d.set_item("label", &v.label)?;
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
    d.set_item("label", &n.label)?;
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
    PyRuntimeError::new_err(format!("Cypher execution error: {e}"))
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
        "lsn".to_string(),
        "schema_version".to_string(),
    ];
    all_names.extend(prop_keys.iter().cloned());

    let ids = PyList::empty_bound(py);
    for v in views {
        ids.append(v.id.to_string())?;
    }
    arrays.append(pyarrow.call_method1("array", (ids,))?)?;

    let labels = PyList::empty_bound(py);
    for v in views {
        labels.append(&v.label)?;
    }
    arrays.append(pyarrow.call_method1("array", (labels,))?)?;

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

    for key in prop_keys {
        let pylist = PyList::empty_bound(py);
        for v in views {
            let pyval = match v.properties.get(key) {
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
fn _lib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<QueryResult>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
