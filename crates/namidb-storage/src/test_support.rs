//! Cross-module test coordination for process-global configuration.
//!
//! The Search-LSM compaction policy is sampled from environment variables,
//! which are process-global: a test that sets `NAMIDB_SEARCH_LSM_*` leaks the
//! values to every test running concurrently, not only to itself. Any test
//! that either MUTATES those variables or ASSERTS behavior that depends on
//! their defaults (e.g. "the incremental policy retains deltas", "no legacy
//! rebuild marker is minted") must hold [`SEARCH_COMPACTION_ENV`] for its
//! whole body. Mutating tests additionally construct
//! [`SearchCompactionEnvRestore`] so the previous values return on drop even
//! when the test panics.

use std::sync::Arc;

use object_store::ObjectStore;

/// Serialises every test that touches or observes the Search-LSM policy env.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
pub(crate) static SEARCH_COMPACTION_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII: force a deterministic consolidation-friendly policy, restoring the
/// caller's environment on drop.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
pub(crate) struct SearchCompactionEnvRestore {
    max_segments: Option<std::ffi::OsString>,
    compact_segments: Option<std::ffi::OsString>,
    base_bytes: Option<std::ffi::OsString>,
    base_stale_percent: Option<std::ffi::OsString>,
    force_base: Option<std::ffi::OsString>,
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl SearchCompactionEnvRestore {
    pub(crate) fn configure() -> Self {
        let restore = Self {
            max_segments: std::env::var_os("NAMIDB_SEARCH_LSM_MAX_SEGMENTS"),
            compact_segments: std::env::var_os("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS"),
            base_bytes: std::env::var_os("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES"),
            base_stale_percent: std::env::var_os("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT"),
            force_base: std::env::var_os("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION"),
        };
        std::env::set_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS", "8");
        std::env::set_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS", "3");
        std::env::set_var("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES", u64::MAX.to_string());
        std::env::set_var("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT", u64::MAX.to_string());
        std::env::set_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION", "true");
        restore
    }

    pub(crate) fn select_delta_runs(&self, trigger: usize) {
        std::env::set_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS", trigger.to_string());
        std::env::set_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION", "false");
    }
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
impl Drop for SearchCompactionEnvRestore {
    fn drop(&mut self) {
        match self.max_segments.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS"),
        }
        match self.compact_segments.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS"),
        }
        match self.base_bytes.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES"),
        }
        match self.base_stale_percent.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT"),
        }
        match self.force_base.take() {
            Some(value) => std::env::set_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION", value),
            None => std::env::remove_var("NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION"),
        }
    }
}

/// `put_opts` exactly once. Used to simulate both transient commit faults
/// and cancellation while an immutable flush body is in flight.
#[derive(Debug)]
pub(crate) struct FaultStore {
    inner: Arc<dyn ObjectStore>,
    fail_next_put_on: std::sync::Mutex<Option<String>>,
    block_next_put_on: std::sync::Mutex<Option<String>>,
    blocked_put: tokio::sync::Notify,
    release_put: tokio::sync::Notify,
}

impl FaultStore {
    pub(crate) fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self {
            inner,
            fail_next_put_on: std::sync::Mutex::new(None),
            block_next_put_on: std::sync::Mutex::new(None),
            blocked_put: tokio::sync::Notify::new(),
            release_put: tokio::sync::Notify::new(),
        }
    }
    pub(crate) fn fail_next_put_containing(&self, needle: &str) {
        *self.fail_next_put_on.lock().unwrap() = Some(needle.to_string());
    }

    pub(crate) fn block_next_put_containing(&self, needle: &str) {
        *self.block_next_put_on.lock().unwrap() = Some(needle.to_string());
    }

    pub(crate) async fn wait_for_blocked_put(&self) {
        self.blocked_put.notified().await;
    }
}

impl std::fmt::Display for FaultStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaultStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for FaultStore {
    async fn put_opts(
        &self,
        location: &object_store::path::Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        let hit = {
            let mut guard = self.fail_next_put_on.lock().unwrap();
            match guard.as_deref() {
                Some(needle) if location.as_ref().contains(needle) => {
                    *guard = None;
                    true
                }
                _ => false,
            }
        };
        if hit {
            return Err(object_store::Error::Generic {
                store: "FaultStore",
                source: "injected transient put failure".into(),
            });
        }
        let block = {
            let mut guard = self.block_next_put_on.lock().unwrap();
            match guard.as_deref() {
                Some(needle) if location.as_ref().contains(needle) => {
                    *guard = None;
                    true
                }
                _ => false,
            }
        };
        if block {
            // `notify_one` retains a permit if the test has not started
            // waiting yet. The matching PUT then stays pending until the
            // flush future is deliberately dropped.
            self.blocked_put.notify_one();
            self.release_put.notified().await;
        }
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
        self.inner.get_opts(location, options).await
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
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
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }
}
