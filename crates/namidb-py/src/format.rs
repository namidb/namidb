//! Storage format inspection and the explicit node-layout migration behind
//! `Client.storage_format()` and `Client.migrate()`.

use std::sync::Arc;
use std::time::Instant;

use object_store::ObjectStore;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

use namidb_storage::{
    inspect_storage_format, FormatInspection, Manifest, NamespacePaths, StorageFormat,
    STORAGE_FORMAT_VERSION,
};

use crate::Writer;

// pyo3 0.22's macro expands a `gil-refs` feature check this crate does not
// declare.
#[allow(unexpected_cfgs)]
mod errors {
    use pyo3::exceptions::PyRuntimeError;

    pyo3::create_exception!(
        namidb,
        StorageFormatError,
        PyRuntimeError,
        "The namespace's storage format does not allow the requested access: it was \
         written by a newer NamiDB, or opening it for writing would upgrade data an \
         older release still reads (pass `upgrade=True` to allow that)."
    );
}
pub(crate) use errors::StorageFormatError;

/// Map a storage error, giving format refusals their own exception type.
pub(crate) fn map_format_err(e: &namidb_storage::Error) -> Option<PyErr> {
    match e {
        namidb_storage::Error::NewerStorageFormat { .. }
        | namidb_storage::Error::StorageFormatUpgradeRequired { .. } => {
            Some(StorageFormatError::new_err(e.to_string()))
        }
        _ => None,
    }
}

fn stamp_dict<'py>(py: Python<'py>, stamp: &StorageFormat) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("version", stamp.version)?;
    d.set_item("min_reader_version", stamp.min_reader_version)?;
    d.set_item("min_writer_version", stamp.min_writer_version)?;
    d.set_item("written_by", &stamp.written_by)?;
    Ok(d)
}

pub(crate) fn inspection_to_py(
    py: Python<'_>,
    inspection: &FormatInspection,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("manifest_version", inspection.manifest_version)?;
    d.set_item("supported_version", STORAGE_FORMAT_VERSION)?;
    match &inspection.stamp {
        Some(stamp) => d.set_item("stamp", stamp_dict(py, stamp)?)?,
        None => d.set_item("stamp", py.None())?,
    }
    d.set_item("effective_version", inspection.effective_format())?;
    let legacy: Vec<_> = inspection.legacy_node_ssts().collect();
    d.set_item("node_ssts", inspection.node_ssts.len())?;
    d.set_item("legacy_node_ssts", legacy.len())?;
    d.set_item(
        "legacy_node_rows",
        legacy.iter().map(|sst| sst.row_count).sum::<u64>(),
    )?;
    d.set_item(
        "legacy_node_bytes",
        legacy.iter().map(|sst| sst.size_bytes).sum::<u64>(),
    )?;
    d.set_item("legacy_wal_segments", inspection.legacy_wal_segments)?;
    d.set_item("needs_migration", !legacy.is_empty())?;
    Ok(d.unbind())
}

pub(crate) async fn inspect(
    store: Arc<dyn ObjectStore>,
    paths: NamespacePaths,
    manifest: Manifest,
) -> namidb_storage::Result<FormatInspection> {
    inspect_storage_format(&store, &paths, &manifest).await
}

/// One rewritten SST, as the progress callback and the report see it.
struct Step {
    done: usize,
    total: usize,
    rows: u64,
    source_bytes: u64,
    output_bytes: u64,
    level: u32,
}

fn step_dict<'py>(py: Python<'py>, step: &Step, started: Instant) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("done", step.done)?;
    d.set_item("total", step.total)?;
    d.set_item("rows", step.rows)?;
    d.set_item("source_bytes", step.source_bytes)?;
    d.set_item("output_bytes", step.output_bytes)?;
    d.set_item("level", step.level)?;
    d.set_item("elapsed_secs", started.elapsed().as_secs_f64())?;
    Ok(d)
}

/// Rewrite every 2.x-layout node SST into the current layout, one SST per
/// manifest commit. Each SST is prepared and verified off the writer lock,
/// and installed under it, like a compaction pass. Interrupted at any point
/// (an exception, Ctrl-C, a crash), the namespace keeps every SST it had
/// committed so far; running it again resumes with the ones left.
pub(crate) fn migrate(
    py: Python<'_>,
    runtime: &Runtime,
    writer: &Arc<Writer>,
    compaction_lock: &Arc<Mutex<()>>,
    progress: Option<&Bound<'_, PyAny>>,
    dry_run: bool,
) -> PyResult<Py<PyDict>> {
    let started = Instant::now();
    // Hold the compaction lock for the whole run, so no automatic pass
    // prepares against an SST this run is replacing.
    let compaction_guard =
        py.allow_threads(|| runtime.block_on(compaction_lock.clone().lock_owned()));
    let basis =
        py.allow_threads(|| runtime.block_on(async { writer.lock().await.compaction_basis() }));
    let legacy = py
        .allow_threads(|| runtime.block_on(basis.legacy_node_ssts()))
        .map_err(crate::map_storage_err)?;
    let total = legacy.len();
    let report = PyDict::new_bound(py);
    report.set_item("legacy_node_ssts", total)?;
    report.set_item(
        "legacy_node_rows",
        legacy.iter().map(|sst| sst.row_count).sum::<u64>(),
    )?;
    report.set_item(
        "legacy_node_bytes",
        legacy.iter().map(|sst| sst.size_bytes).sum::<u64>(),
    )?;
    report.set_item("dry_run", dry_run)?;
    let steps = PyList::empty_bound(py);
    let mut migrated = 0usize;
    let mut skipped = 0usize;
    if !dry_run {
        for sst in &legacy {
            // Ctrl-C between SSTs stops here with everything before it
            // committed.
            py.check_signals()?;
            let outcome = py.allow_threads(|| {
                runtime.block_on(async {
                    let basis = writer.lock().await.compaction_basis();
                    let (prepared, migration) =
                        match basis.prepare_node_layout_migration(sst.id).await {
                            Ok(done) => done,
                            // A compaction merged it away since the listing:
                            // its rows are already in the current layout.
                            Err(namidb_storage::Error::Precondition(_)) => return Ok(None),
                            Err(e) => return Err(e),
                        };
                    let mut guard = writer.lock().await;
                    match guard.install_prepared_compaction(prepared).await {
                        Ok(_) => Ok(Some(migration)),
                        Err(namidb_storage::Error::Precondition(_)) => Ok(None),
                        Err(e) => Err(e),
                    }
                })
            });
            let Some(migration) = outcome.map_err(crate::map_storage_err)? else {
                skipped += 1;
                continue;
            };
            migrated += 1;
            let step = Step {
                done: migrated + skipped,
                total,
                rows: migration.rows,
                source_bytes: migration.source_bytes,
                output_bytes: migration.output_bytes,
                level: migration.level,
            };
            let d = step_dict(py, &step, started)?;
            steps.append(&d)?;
            if let Some(callback) = progress {
                callback.call1((d,))?;
            }
        }
    }
    drop(compaction_guard);
    let basis =
        py.allow_threads(|| runtime.block_on(async { writer.lock().await.compaction_basis() }));
    let remaining = if dry_run {
        total
    } else {
        py.allow_threads(|| runtime.block_on(basis.legacy_node_ssts()))
            .map_err(crate::map_storage_err)?
            .len()
    };
    report.set_item("migrated", migrated)?;
    report.set_item("skipped", skipped)?;
    report.set_item("remaining", remaining)?;
    report.set_item("steps", steps)?;
    report.set_item("manifest_version", basis.manifest_version())?;
    report.set_item("seconds", started.elapsed().as_secs_f64())?;
    Ok(report.unbind())
}
