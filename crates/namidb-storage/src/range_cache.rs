//! Bounded RAM + NVMe cache for immutable object-store ranges.
//!
//! Object storage remains authoritative. This cache only retains reconstructible
//! byte ranges and can be deleted at any time. The cache key includes the
//! immutable object path, a caller-supplied generation pin (ETag/version), and
//! the exact byte range. Reusing a path for new contents therefore cannot serve
//! stale bytes as long as callers advance the generation pin.
//!
//! The NVMe tier is optional. With no `NAMIDB_NVME_CACHE_PATH` and no explicit
//! RAM page-cache budget, [`shared_range_cache`] returns `None` and the existing
//! storage behaviour is unchanged. When NVMe is enabled, Foyer writes admitted
//! entries on insertion, recovers them after restart, and supplies single-flight
//! misses through [`ImmutableRangeCache::get_or_fetch`].
//!
//! ## Memory accounting
//!
//! `NAMIDB_RAM_PAGE_CACHE_MAX_BYTES` is a request *inside*
//! `NAMIDB_CACHE_MAX_BYTES`. For a hybrid instance its effective assignment is
//! divided between Foyer's resident memory cache and its bounded disk write
//! buffers and a conservative bound for its persistent hash index.
//! `NAMIDB_NVME_CACHE_MAX_BYTES` is an independent on-disk device ceiling.
//! Hybrid admission accepts canonical page-sized entries so a large device
//! cannot create an unbounded in-memory index from millions of tiny ranges.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use foyer::{
    BlockEngineConfig, Cache, CacheBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache,
    HybridCacheBuilder, HybridCachePolicy, RecoverMode, Source,
};
use fs2::FileExt as _;
use futures::{stream, StreamExt, TryStreamExt};
use object_store::path::Path as ObjectPath;
use object_store::{GetOptions, ObjectMeta, ObjectStore, ObjectStoreExt};
use thiserror::Error;
use tokio::sync::Semaphore;

use crate::cache_budget::shared_cache_capacities;

const PAGE_BYTES: usize = 4096;
const CACHE_KEY_FORMAT: &[u8] = b"namidb-immutable-range-v1";
const STORAGE_ENTRY_HEADROOM_BYTES: usize = PAGE_BYTES;
const STORAGE_KEY_BYTES: usize = 102;
const MAX_PARALLEL_PAGE_FETCHES: usize = 16;
const MEMORY_CACHE_SHARDS: usize = 8;

/// Maximum number of authoritative range GETs issued concurrently by one
/// pinned source, including cache misses from overlapping batch reads.
pub const MAX_PINNED_OBJECT_RANGE_READS: usize = 16;
const FOYER_INDEXER_SHARDS: usize = 16;
/// Conservative `HashMap<u64, Index>` capacity, control-byte, allocator and
/// lock overhead per persistent entry.
const FOYER_INDEX_BYTES_PER_ENTRY: usize = 128;
const FOYER_INDEX_SHARD_OVERHEAD_BYTES: usize = PAGE_BYTES;

/// Default RAM request when a complete NVMe configuration is supplied without
/// an explicit `NAMIDB_RAM_PAGE_CACHE_MAX_BYTES`.
pub const DEFAULT_RAM_PAGE_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;
/// Largest range admitted by default.
pub const DEFAULT_RANGE_CACHE_MAX_ENTRY_BYTES: usize = 4 * 1024 * 1024;
/// Canonical fetch/cache page. Normalizing arbitrary reads to this boundary
/// makes adjacent and overlapping requests share one Foyer key and one remote
/// fetch.
pub const DEFAULT_RANGE_CACHE_PAGE_BYTES: usize = 256 * 1024;
/// Default Foyer block size. A block is the disk eviction unit and bounds the
/// largest serialized entry.
pub const DEFAULT_NVME_CACHE_BLOCK_BYTES: usize = 16 * 1024 * 1024;
/// Maximum default size of each Foyer write buffer carved from the effective
/// RAM assignment.
pub const DEFAULT_NVME_CACHE_WRITE_BUFFER_BYTES: usize = 16 * 1024 * 1024;
/// Foyer's block engine owns one active and one rotating write buffer.
///
/// `with_buffer_pool_size` configures each buffer, not their aggregate.
const FOYER_WRITE_BUFFER_COUNT: usize = 2;

/// Immutable object identity and exact half-open byte range.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImmutableRangeKey {
    object_path: String,
    generation: String,
    range: Range<u64>,
}

impl ImmutableRangeKey {
    /// Create a key pinned to an object-store generation (ETag/version).
    pub fn new(
        object_path: impl Into<String>,
        generation: impl Into<String>,
        range: Range<u64>,
    ) -> Result<Self, RangeCacheError> {
        let key = Self {
            object_path: object_path.into(),
            generation: generation.into(),
            range,
        };
        key.validate()?;
        Ok(key)
    }

    /// Create a key for a path whose name itself is immutable (for example an
    /// UUID-addressed SST). This explicit constructor avoids silently omitting
    /// the generation from mutable object paths.
    pub fn from_immutable_path(
        object_path: impl Into<String>,
        range: Range<u64>,
    ) -> Result<Self, RangeCacheError> {
        Self::new(object_path, "immutable-path-v1", range)
    }

    pub fn object_path(&self) -> &str {
        &self.object_path
    }

    pub fn generation(&self) -> &str {
        &self.generation
    }

    pub fn range(&self) -> Range<u64> {
        self.range.clone()
    }

    pub fn len(&self) -> u64 {
        self.range.end - self.range.start
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    fn validate(&self) -> Result<(), RangeCacheError> {
        if self.object_path.is_empty() {
            return Err(RangeCacheError::InvalidKey(
                "object path must not be empty".into(),
            ));
        }
        if self.generation.is_empty() {
            return Err(RangeCacheError::InvalidKey(
                "generation pin must not be empty".into(),
            ));
        }
        if self.range.start >= self.range.end {
            return Err(RangeCacheError::InvalidKey(format!(
                "range must be non-empty and ordered, got {}..{}",
                self.range.start, self.range.end
            )));
        }
        usize::try_from(self.len()).map_err(|_| {
            RangeCacheError::InvalidKey(format!(
                "range length {} does not fit this platform",
                self.len()
            ))
        })?;
        Ok(())
    }
}

/// Generation precondition used both in the persistent cache key and the
/// authoritative object-store request.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PinnedObjectGeneration {
    /// Native object-store version ID.
    Version(String),
    /// HTTP entity tag, enforced with `If-Match`.
    ETag(String),
    /// The object path itself is content-addressed/immutable. No remote
    /// precondition is necessary, but this explicit variant documents that
    /// the caller has made that guarantee.
    ImmutablePath,
}

impl PinnedObjectGeneration {
    /// Prefer a native version ID, otherwise use ETag. Mutable objects with
    /// neither cannot be safely persisted in a cross-restart cache.
    pub fn from_meta(meta: &ObjectMeta) -> Result<Self, RangeCacheError> {
        if let Some(version) = meta.version.as_ref().filter(|value| !value.is_empty()) {
            return Ok(Self::Version(version.clone()));
        }
        if let Some(etag) = meta.e_tag.as_ref().filter(|value| !value.is_empty()) {
            return Ok(Self::ETag(etag.clone()));
        }
        Err(RangeCacheError::InvalidKey(format!(
            "object {} has neither version nor ETag; use ImmutablePath only for a \
             content-addressed path",
            meta.location
        )))
    }

    fn cache_token(&self) -> Result<String, RangeCacheError> {
        match self {
            Self::Version(version) if !version.is_empty() => Ok(format!("version:{version}")),
            Self::ETag(etag) if !etag.is_empty() => Ok(format!("etag:{etag}")),
            Self::ImmutablePath => Ok("immutable-path-v1".into()),
            Self::Version(_) => Err(RangeCacheError::InvalidKey(
                "object version must not be empty".into(),
            )),
            Self::ETag(_) => Err(RangeCacheError::InvalidKey(
                "object ETag must not be empty".into(),
            )),
        }
    }

    fn get_options(&self, range: Range<u64>) -> GetOptions {
        let options = GetOptions::new().with_range(Some(range));
        match self {
            Self::Version(version) => options.with_version(Some(version.clone())),
            Self::ETag(etag) => options.with_if_match(Some(etag.clone())),
            Self::ImmutablePath => options,
        }
    }
}

fn validate_get_result(
    result: &object_store::GetResult,
    location: &ObjectPath,
    generation: &PinnedObjectGeneration,
    object_size: u64,
    expected_range: &Range<u64>,
) -> Result<(), RangeCacheError> {
    if result.meta.location != *location {
        return Err(RangeCacheError::ResponseValidation(format!(
            "authoritative range response location {} != requested {}",
            result.meta.location, location
        )));
    }
    if result.meta.size != object_size {
        return Err(RangeCacheError::ResponseValidation(format!(
            "authoritative range response object size {} != pinned {object_size} for {}",
            result.meta.size, location
        )));
    }
    if result.range != *expected_range {
        return Err(RangeCacheError::ResponseValidation(format!(
            "authoritative range response {}..{} != requested {}..{} for {}",
            result.range.start,
            result.range.end,
            expected_range.start,
            expected_range.end,
            location
        )));
    }
    match generation {
        PinnedObjectGeneration::Version(expected) => {
            if result.meta.version.as_deref() != Some(expected.as_str()) {
                return Err(RangeCacheError::ResponseValidation(format!(
                    "authoritative range response version {:?} != pinned {:?} for {}",
                    result.meta.version, expected, location
                )));
            }
        }
        PinnedObjectGeneration::ETag(expected) => {
            if result.meta.e_tag.as_deref() != Some(expected.as_str()) {
                return Err(RangeCacheError::ResponseValidation(format!(
                    "authoritative range response ETag {:?} != pinned {:?} for {}",
                    result.meta.e_tag, expected, location
                )));
            }
        }
        PinnedObjectGeneration::ImmutablePath => {}
    }
    Ok(())
}

/// Complete configuration for one immutable range cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeCacheConfig {
    /// Total effective RAM assignment. For a hybrid cache this includes
    /// both Foyer write buffers; the remainder is the resident memory cache.
    pub ram_budget_bytes: usize,
    /// Optional persistent cache directory.
    pub disk_path: Option<PathBuf>,
    /// Fixed Foyer device capacity. Must be zero without `disk_path`.
    pub disk_capacity_bytes: usize,
    /// Maximum admitted payload bytes.
    pub max_entry_bytes: usize,
    /// Canonical aligned object-store range size used by
    /// [`ImmutableRangeCache::read_object_range`].
    pub page_size_bytes: usize,
    /// Foyer disk eviction unit and serialized-entry ceiling.
    pub block_size_bytes: usize,
    /// Size of each active/rotating disk write buffer. Both are carved from
    /// `ram_budget_bytes`.
    pub write_buffer_bytes: usize,
    /// Stable deployment/store namespace mixed into every on-disk key.
    pub key_namespace: String,
}

impl RangeCacheConfig {
    /// Construct a bounded memory-only range cache.
    pub fn memory_only(ram_budget_bytes: usize) -> Self {
        let max_entry_bytes = DEFAULT_RANGE_CACHE_MAX_ENTRY_BYTES.min(ram_budget_bytes);
        Self {
            ram_budget_bytes,
            disk_path: None,
            disk_capacity_bytes: 0,
            max_entry_bytes,
            page_size_bytes: DEFAULT_RANGE_CACHE_PAGE_BYTES.min(max_entry_bytes),
            block_size_bytes: 0,
            write_buffer_bytes: 0,
            key_namespace: "default".into(),
        }
    }

    /// Construct a hybrid range cache with production defaults.
    pub fn hybrid(
        ram_budget_bytes: usize,
        disk_path: impl Into<PathBuf>,
        disk_capacity_bytes: usize,
    ) -> Self {
        let write_buffer_bytes = default_write_buffer_bytes(ram_budget_bytes);
        Self {
            ram_budget_bytes,
            disk_path: Some(disk_path.into()),
            disk_capacity_bytes,
            max_entry_bytes: DEFAULT_RANGE_CACHE_MAX_ENTRY_BYTES,
            page_size_bytes: DEFAULT_RANGE_CACHE_PAGE_BYTES,
            block_size_bytes: DEFAULT_NVME_CACHE_BLOCK_BYTES,
            write_buffer_bytes,
            // Persistent data can outlive one store client. Force callers to
            // provide an identity that includes deployment/bucket.
            key_namespace: String::new(),
        }
    }

    /// Resolve configuration from the environment and the already-scaled
    /// process cache plan.
    ///
    /// The NVMe path and capacity are a pair. Supplying only one is rejected
    /// rather than accidentally allocating 80% of the filesystem (Foyer's
    /// direct default when no capacity is specified).
    pub fn from_env() -> Result<Option<Self>, RangeCacheError> {
        let ram_budget_bytes = shared_cache_capacities().range_page_capacity_bytes();
        let disk_path = optional_nonempty_env("NAMIDB_NVME_CACHE_PATH")?.map(PathBuf::from);
        let disk_capacity = optional_usize_env("NAMIDB_NVME_CACHE_MAX_BYTES")?;
        let disk_capacity_bytes = disk_capacity.unwrap_or(0);

        match (&disk_path, disk_capacity_bytes) {
            (None, 0) if ram_budget_bytes == 0 => return Ok(None),
            (None, 0) => {}
            (Some(_), 0) => {
                return Err(RangeCacheError::Configuration(
                    "NAMIDB_NVME_CACHE_PATH requires a non-zero \
                     NAMIDB_NVME_CACHE_MAX_BYTES"
                        .into(),
                ));
            }
            (None, _) => {
                return Err(RangeCacheError::Configuration(
                    "NAMIDB_NVME_CACHE_MAX_BYTES requires NAMIDB_NVME_CACHE_PATH".into(),
                ));
            }
            (Some(_), _) if ram_budget_bytes == 0 => {
                // Respect NAMIDB_CACHE_MAX_BYTES=0 and proportional plans that
                // assign no bytes to this optional tier.
                tracing::warn!(
                    disk_capacity_bytes,
                    "NVMe range cache disabled because its effective RAM assignment is zero"
                );
                return Ok(None);
            }
            (Some(_), _) => {}
        }

        let explicit_max_entry = optional_usize_env("NAMIDB_RANGE_CACHE_MAX_ENTRY_BYTES")?;
        let max_entry_bytes = explicit_max_entry.unwrap_or_else(|| {
            if disk_path.is_some() {
                DEFAULT_RANGE_CACHE_MAX_ENTRY_BYTES
            } else {
                DEFAULT_RANGE_CACHE_MAX_ENTRY_BYTES.min(ram_budget_bytes)
            }
        });
        let page_size_bytes = optional_usize_env("NAMIDB_RANGE_CACHE_PAGE_BYTES")?
            .unwrap_or_else(|| DEFAULT_RANGE_CACHE_PAGE_BYTES.min(max_entry_bytes));
        let block_size_bytes = if disk_path.is_some() {
            optional_usize_env("NAMIDB_NVME_CACHE_BLOCK_BYTES")?
                .unwrap_or(DEFAULT_NVME_CACHE_BLOCK_BYTES)
        } else {
            0
        };
        let write_buffer_bytes = if disk_path.is_some() {
            optional_usize_env("NAMIDB_NVME_CACHE_WRITE_BUFFER_BYTES")?
                .unwrap_or_else(|| default_write_buffer_bytes(ram_budget_bytes))
        } else {
            0
        };
        let key_namespace = match (
            disk_path.is_some(),
            optional_nonempty_env("NAMIDB_RANGE_CACHE_NAMESPACE")?,
        ) {
            (true, Some(namespace)) => namespace,
            (true, None) => {
                return Err(RangeCacheError::Configuration(
                    "NAMIDB_RANGE_CACHE_NAMESPACE is required with \
                     NAMIDB_NVME_CACHE_PATH; use a stable deployment+bucket identity"
                        .into(),
                ));
            }
            (false, Some(namespace)) => namespace,
            (false, None) => "memory-only".into(),
        };

        let config = Self {
            ram_budget_bytes,
            disk_path,
            disk_capacity_bytes,
            max_entry_bytes,
            page_size_bytes,
            block_size_bytes,
            write_buffer_bytes,
            key_namespace,
        };
        config.validate()?;
        Ok(Some(config))
    }

    /// Conservative bytes reserved for Foyer's persistent hash index.
    pub fn disk_index_estimate_bytes(&self) -> usize {
        if self.disk_path.is_none() || self.disk_capacity_bytes == 0 || self.page_size_bytes == 0 {
            return 0;
        }
        self.disk_capacity_bytes
            .div_ceil(self.page_size_bytes)
            .saturating_mul(FOYER_INDEX_BYTES_PER_ENTRY)
            .saturating_add(FOYER_INDEXER_SHARDS.saturating_mul(FOYER_INDEX_SHARD_OVERHEAD_BYTES))
    }

    pub fn write_buffer_reservation_bytes(&self) -> usize {
        self.write_buffer_bytes
            .saturating_mul(FOYER_WRITE_BUFFER_COUNT)
    }

    /// Bytes available to Foyer's resident entries after reserving both disk
    /// write buffers and the conservative persistent-index bound.
    pub fn memory_capacity_bytes(&self) -> usize {
        self.ram_budget_bytes
            .saturating_sub(self.write_buffer_reservation_bytes())
            .saturating_sub(self.disk_index_estimate_bytes())
    }

    pub fn is_hybrid(&self) -> bool {
        self.disk_path.is_some()
    }

    pub fn validate(&self) -> Result<(), RangeCacheError> {
        if self.ram_budget_bytes == 0 {
            return Err(RangeCacheError::Configuration(
                "range cache RAM budget must be non-zero".into(),
            ));
        }
        if self.max_entry_bytes == 0 {
            return Err(RangeCacheError::Configuration(
                "range cache max entry bytes must be non-zero".into(),
            ));
        }
        if self.page_size_bytes < PAGE_BYTES {
            return Err(RangeCacheError::Configuration(format!(
                "range cache page bytes must be at least {PAGE_BYTES}"
            )));
        }
        if self.page_size_bytes % PAGE_BYTES != 0 {
            return Err(RangeCacheError::Configuration(format!(
                "range cache page bytes must be aligned to {PAGE_BYTES}"
            )));
        }
        if self.page_size_bytes > self.max_entry_bytes {
            return Err(RangeCacheError::Configuration(format!(
                "range cache page size {} exceeds max entry {}",
                self.page_size_bytes, self.max_entry_bytes
            )));
        }
        if self.key_namespace.is_empty() {
            return Err(RangeCacheError::Configuration(
                "range cache namespace must not be empty".into(),
            ));
        }
        if self.key_namespace.len() > 1024 {
            return Err(RangeCacheError::Configuration(
                "range cache namespace must be at most 1024 bytes".into(),
            ));
        }

        match &self.disk_path {
            None => {
                if self.disk_capacity_bytes != 0
                    || self.block_size_bytes != 0
                    || self.write_buffer_bytes != 0
                {
                    return Err(RangeCacheError::Configuration(
                        "memory-only range cache must have zero disk capacity, block size, \
                         and write buffer"
                            .into(),
                    ));
                }
                if self.max_entry_bytes > self.ram_budget_bytes {
                    return Err(RangeCacheError::Configuration(format!(
                        "memory-only max entry {} exceeds RAM budget {}",
                        self.max_entry_bytes, self.ram_budget_bytes
                    )));
                }
            }
            Some(path) => {
                if path.as_os_str().is_empty() {
                    return Err(RangeCacheError::Configuration(
                        "NVMe cache path must not be empty".into(),
                    ));
                }
                if self.disk_capacity_bytes < PAGE_BYTES {
                    return Err(RangeCacheError::Configuration(format!(
                        "NVMe cache capacity must be at least {PAGE_BYTES} bytes"
                    )));
                }
                if self.block_size_bytes < PAGE_BYTES {
                    return Err(RangeCacheError::Configuration(format!(
                        "NVMe cache block size must be at least {PAGE_BYTES} bytes"
                    )));
                }
                let block_size = align_up(self.block_size_bytes, PAGE_BYTES)?;
                let minimum_device = block_size.checked_mul(8).ok_or_else(|| {
                    RangeCacheError::Configuration("NVMe block-size calculation overflow".into())
                })?;
                if self.disk_capacity_bytes < minimum_device {
                    return Err(RangeCacheError::Configuration(format!(
                        "NVMe cache capacity {} must hold at least eight blocks ({minimum_device})",
                        self.disk_capacity_bytes
                    )));
                }
                let serialized_ceiling = self
                    .max_entry_bytes
                    .checked_add(STORAGE_ENTRY_HEADROOM_BYTES)
                    .ok_or_else(|| {
                        RangeCacheError::Configuration(
                            "range cache max-entry calculation overflow".into(),
                        )
                    })?;
                if serialized_ceiling > block_size {
                    return Err(RangeCacheError::Configuration(format!(
                        "max entry {} plus serialization headroom exceeds NVMe block size \
                         {block_size}",
                        self.max_entry_bytes
                    )));
                }
                if self.write_buffer_bytes < PAGE_BYTES {
                    return Err(RangeCacheError::Configuration(format!(
                        "NVMe write buffer must be at least {PAGE_BYTES} bytes"
                    )));
                }
                if self.write_buffer_bytes % PAGE_BYTES != 0 {
                    return Err(RangeCacheError::Configuration(format!(
                        "NVMe write buffer must be aligned to {PAGE_BYTES}"
                    )));
                }
                let aggregate_write_buffers = self.write_buffer_reservation_bytes();
                let disk_index_estimate = self.disk_index_estimate_bytes();
                let reserved = aggregate_write_buffers
                    .checked_add(disk_index_estimate)
                    .ok_or_else(|| {
                        RangeCacheError::Configuration(
                            "NVMe cache RAM reservation overflows usize".into(),
                        )
                    })?;
                if reserved >= self.ram_budget_bytes {
                    return Err(RangeCacheError::Configuration(format!(
                        "Foyer's {FOYER_WRITE_BUFFER_COUNT} NVMe write buffers reserve \
                         {aggregate_write_buffers} bytes and its persistent index is estimated \
                         at {disk_index_estimate} bytes; their {reserved}-byte total must be \
                         smaller than the range-cache RAM budget {}",
                        self.ram_budget_bytes
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Errors are isolated from authoritative object-store data: callers may
/// discard/reopen the cache after any Foyer or local-I/O failure.
#[derive(Debug, Error)]
pub enum RangeCacheError {
    #[error("invalid immutable range cache configuration: {0}")]
    Configuration(String),

    #[error("invalid immutable range key: {0}")]
    InvalidKey(String),

    #[error(
        "range cache value length mismatch for {path} {start}..{end}: expected {expected}, \
         found {actual}"
    )]
    ValueLengthMismatch {
        path: String,
        start: u64,
        end: u64,
        expected: usize,
        actual: usize,
    },

    #[error("range cache directory {path} is already in use: {source}")]
    DirectoryBusy {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("range cache I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Foyer range cache error: {0}")]
    Foyer(#[from] foyer::Error),

    #[error("invalid authoritative range response: {0}")]
    ResponseValidation(String),

    #[error("authoritative range fetch failed: {0}")]
    Fetch(#[source] anyhow::Error),
}

impl RangeCacheError {
    /// Whether retrying directly against object storage is a safe bypass.
    ///
    /// Only failures owned by the reconstructible local cache qualify. An
    /// external Foyer error is the cache's wrapped read-through future (for
    /// example an S3 precondition failure) and must not be retried as though it
    /// were an NVMe fault: doing so can duplicate remote work and obscure an
    /// object-generation change.
    pub fn is_local_failure(&self) -> bool {
        match self {
            Self::DirectoryBusy { .. } | Self::Io(_) | Self::ValueLengthMismatch { .. } => true,
            Self::Foyer(error) => error.kind() != foyer::ErrorKind::External,
            Self::Configuration(_)
            | Self::InvalidKey(_)
            | Self::ResponseValidation(_)
            | Self::Fetch(_) => false,
        }
    }

    fn is_response_validation_failure(&self) -> bool {
        match self {
            Self::InvalidKey(_)
            | Self::ValueLengthMismatch { .. }
            | Self::ResponseValidation(_) => true,
            Self::Fetch(source) => source.downcast_ref::<RangeCacheError>().map_or_else(
                || source.downcast_ref::<object_store::Error>().is_none(),
                RangeCacheError::is_response_validation_failure,
            ),
            Self::Foyer(error) => error
                .downcast_ref::<RangeCacheError>()
                .is_some_and(RangeCacheError::is_response_validation_failure),
            Self::Configuration(_) | Self::DirectoryBusy { .. } | Self::Io(_) => false,
        }
    }
}

#[derive(Default)]
struct RangeCacheCounters {
    memory_hits: AtomicU64,
    disk_hits: AtomicU64,
    misses: AtomicU64,
    outer_fetches: AtomicU64,
    inserts: AtomicU64,
    admission_rejections: AtomicU64,
    corrupt_entries: AtomicU64,
}

/// Stable counter snapshot for diagnostics and tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RangeCacheStats {
    pub memory_hits: u64,
    pub disk_hits: u64,
    pub misses: u64,
    pub outer_fetches: u64,
    pub inserts: u64,
    pub admission_rejections: u64,
    pub corrupt_entries: u64,
}

/// Synchronous, side-effect-free view of the process-wide range cache.
///
/// Observability surfaces use this instead of [`shared_range_cache`] so a
/// Prometheus scrape never initializes an NVMe device, acquires its directory
/// lock, or samples configuration before the first real range read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SharedRangeCacheSnapshot {
    pub ram_budget_bytes: usize,
    pub memory_capacity_bytes: usize,
    pub memory_usage_bytes: usize,
    pub write_buffer_reservation_bytes: usize,
    pub disk_index_estimate_bytes: usize,
    pub disk_capacity_bytes: usize,
    pub disk_read_bytes: usize,
    pub disk_write_bytes: usize,
    pub stats: RangeCacheStats,
}

/// Backing store for the range cache.
///
/// A memory-only cache is deliberately *not* a `HybridCache` with no device.
/// Foyer's hybrid cache starts background workers on whichever tokio runtime
/// first builds it, and this cache is reached through a process-global
/// `OnceCell`. Any host that creates and drops runtimes — every `#[tokio::test]`
/// does, and so does an embedded caller that opens a runtime per call — would
/// leave every later user awaiting workers that no longer exist. The pure
/// in-memory cache has no workers and therefore no runtime affinity, so the
/// default (RAM-only) tier is safe to share for the life of the process.
#[derive(Clone)]
enum RangeCacheBackend {
    Memory(Cache<String, Bytes>),
    Hybrid(HybridCache<String, Bytes>),
}

impl RangeCacheBackend {
    fn usage(&self) -> usize {
        match self {
            Self::Memory(cache) => cache.usage(),
            Self::Hybrid(cache) => cache.memory().usage(),
        }
    }

    fn evict_all(&self) {
        match self {
            Self::Memory(cache) => cache.evict_all(),
            Self::Hybrid(cache) => cache.memory().evict_all(),
        }
    }

    fn disk_read_bytes(&self) -> usize {
        match self {
            Self::Memory(_) => 0,
            Self::Hybrid(cache) => cache.statistics().disk_read_bytes(),
        }
    }

    fn disk_write_bytes(&self) -> usize {
        match self {
            Self::Memory(_) => 0,
            Self::Hybrid(cache) => cache.statistics().disk_write_bytes(),
        }
    }

    fn insert(&self, key: String, value: Bytes) {
        match self {
            Self::Memory(cache) => {
                cache.insert(key, value);
            }
            Self::Hybrid(cache) => {
                cache.insert(key, value);
            }
        }
    }

    fn remove(&self, key: &str) {
        match self {
            Self::Memory(cache) => {
                cache.remove(key);
            }
            Self::Hybrid(cache) => cache.remove(key),
        }
    }

    /// Look up a resident value and where it came from.
    async fn get(&self, key: &str) -> Result<Option<(Bytes, Source)>, RangeCacheError> {
        match self {
            Self::Memory(cache) => Ok(cache
                .get(key)
                .map(|entry| (entry.value().clone(), entry.source()))),
            Self::Hybrid(cache) => Ok(cache
                .get(key)
                .await?
                .map(|entry| (entry.value().clone(), entry.source()))),
        }
    }

    /// Single-flight read-through. Both backends deduplicate concurrent
    /// fetches of the same cold key, so the caller's fetch runs at most once.
    async fn get_or_fetch<F, FU>(
        &self,
        key: &str,
        fetch: F,
    ) -> Result<(Bytes, Source), RangeCacheError>
    where
        F: FnOnce() -> FU,
        FU: Future<Output = Result<Bytes, anyhow::Error>> + Send + 'static,
    {
        match self {
            Self::Memory(cache) => {
                let entry = cache.get_or_fetch(key, fetch).await?;
                Ok((entry.value().clone(), entry.source()))
            }
            Self::Hybrid(cache) => {
                let entry = cache.get_or_fetch(key, fetch).await?;
                Ok((entry.value().clone(), entry.source()))
            }
        }
    }

    async fn destroy_storage(&self) -> Result<(), RangeCacheError> {
        match self {
            // Nothing is persisted, so `clear_memory` already emptied it.
            Self::Memory(_) => Ok(()),
            Self::Hybrid(cache) => {
                cache.storage().destroy().await?;
                Ok(())
            }
        }
    }

    async fn close(&self) -> Result<(), RangeCacheError> {
        match self {
            Self::Memory(_) => Ok(()),
            Self::Hybrid(cache) => {
                cache.close().await?;
                Ok(())
            }
        }
    }
}

/// Cloneable process cache for immutable object-store ranges.
#[derive(Clone)]
pub struct ImmutableRangeCache {
    cache: RangeCacheBackend,
    config: Arc<RangeCacheConfig>,
    counters: Arc<RangeCacheCounters>,
    // Holding the file keeps the cache directory process-exclusive. Foyer's
    // filesystem device does not lock the directory itself.
    _directory_lock: Option<Arc<File>>,
}

impl fmt::Debug for ImmutableRangeCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImmutableRangeCache")
            .field("ram_budget_bytes", &self.config.ram_budget_bytes)
            .field("memory_capacity_bytes", &self.memory_capacity_bytes())
            .field("memory_usage_bytes", &self.memory_usage_bytes())
            .field("disk_path", &self.config.disk_path)
            .field("disk_capacity_bytes", &self.config.disk_capacity_bytes)
            .field("max_entry_bytes", &self.config.max_entry_bytes)
            .field("hybrid", &self.is_hybrid())
            .field("stats", &self.stats())
            .finish()
    }
}

impl ImmutableRangeCache {
    /// Open a memory-only or persistent hybrid cache.
    pub async fn open(mut config: RangeCacheConfig) -> Result<Self, RangeCacheError> {
        config.validate()?;
        if config.disk_path.is_some() {
            config.block_size_bytes = align_up(config.block_size_bytes, PAGE_BYTES)?;
            config.disk_capacity_bytes = align_down(config.disk_capacity_bytes, PAGE_BYTES);
        }

        let memory_capacity = config.memory_capacity_bytes();
        let logical_memory_capacity = memory_capacity;
        let max_entry_bytes = config.max_entry_bytes;
        // Foyer divides the capacity evenly between shards. Keep every shard
        // large enough for the heaviest admissible resident entry; otherwise
        // a valid page in a small cache would be inserted and immediately
        // evicted merely because its shard is smaller than the page.
        let maximum_resident_weight = max_entry_bytes
            .saturating_add(STORAGE_KEY_BYTES)
            .min(memory_capacity)
            .max(1);
        let memory_shards =
            (memory_capacity / maximum_resident_weight).clamp(1, MEMORY_CACHE_SHARDS);
        let weighter = |key: &String, value: &Bytes| key.len().saturating_add(value.len());
        let admits = move |key: &String, value: &Bytes| {
            let weight = key.len().saturating_add(value.len());
            logical_memory_capacity > 0
                && value.len() <= max_entry_bytes
                && weight <= logical_memory_capacity
        };

        // No disk tier means no Foyer storage engine, and therefore no
        // background workers bound to the calling runtime. See
        // `RangeCacheBackend` for why that distinction has to be structural
        // rather than a hybrid cache configured with an empty device.
        let Some(disk_path) = config.disk_path.clone() else {
            let cache = CacheBuilder::new(memory_capacity.max(1))
                .with_shards(memory_shards)
                .with_weighter(weighter)
                .with_filter(admits)
                .build();
            tracing::info!(
                range_cache_ram_budget_bytes = config.ram_budget_bytes,
                range_cache_memory_bytes = config.memory_capacity_bytes(),
                range_cache_max_entry_bytes = config.max_entry_bytes,
                range_cache_hybrid = false,
                "opened immutable object-range cache"
            );
            return Ok(Self {
                cache: RangeCacheBackend::Memory(cache),
                config: Arc::new(config),
                counters: Arc::new(RangeCacheCounters::default()),
                _directory_lock: None,
            });
        };

        let mut builder = HybridCacheBuilder::new()
            .with_name("namidb-range-cache")
            .with_policy(HybridCachePolicy::WriteOnInsertion)
            .with_flush_on_close(true)
            .memory(memory_capacity.max(1))
            .with_shards(memory_shards)
            .with_weighter(weighter)
            .with_filter(admits)
            .storage();

        let directory_lock = {
            let path = disk_path.as_path();
            std::fs::create_dir_all(path)?;
            let lock = lock_cache_directory(path)?;
            let device = FsDeviceBuilder::new(path)
                .with_capacity(config.disk_capacity_bytes)
                .build()?;
            let engine = BlockEngineConfig::new(device)
                .with_block_size(config.block_size_bytes)
                .with_indexer_shards(FOYER_INDEXER_SHARDS)
                .with_recover_concurrency(4)
                .with_flushers(1)
                .with_reclaimers(1)
                .with_buffer_pool_size(config.write_buffer_bytes)
                .with_submit_queue_size_threshold(config.write_buffer_bytes)
                // Removes/clear are rare for immutable generations, but a
                // tombstone log prevents deleted cache entries reappearing
                // after an otherwise clean restart.
                .with_tombstone_log(true);
            builder = builder
                .with_recover_mode(RecoverMode::Quiet)
                .with_engine_config(engine);
            Some(Arc::new(lock))
        };

        let cache = RangeCacheBackend::Hybrid(builder.build().await?);
        tracing::info!(
            range_cache_ram_budget_bytes = config.ram_budget_bytes,
            range_cache_memory_bytes = config.memory_capacity_bytes(),
            range_cache_write_buffer_bytes = config.write_buffer_reservation_bytes(),
            range_cache_disk_index_estimate_bytes = config.disk_index_estimate_bytes(),
            range_cache_disk_bytes = config.disk_capacity_bytes,
            range_cache_max_entry_bytes = config.max_entry_bytes,
            range_cache_hybrid = config.is_hybrid(),
            "opened immutable object-range cache"
        );
        Ok(Self {
            cache,
            config: Arc::new(config),
            counters: Arc::new(RangeCacheCounters::default()),
            _directory_lock: directory_lock,
        })
    }

    pub fn config(&self) -> &RangeCacheConfig {
        &self.config
    }

    pub fn is_hybrid(&self) -> bool {
        // Foyer 0.22.3's `HybridCache::is_hybrid()` misidentifies its noop
        // storage engine as enabled because its `TypeId` check observes the
        // erased Arc. Our validated configuration is the authoritative mode.
        self.config.is_hybrid()
    }

    pub fn ram_budget_bytes(&self) -> usize {
        self.config.ram_budget_bytes
    }

    pub fn memory_capacity_bytes(&self) -> usize {
        self.config.memory_capacity_bytes()
    }

    pub fn memory_usage_bytes(&self) -> usize {
        self.cache.usage()
    }

    pub fn write_buffer_reservation_bytes(&self) -> usize {
        self.config.write_buffer_reservation_bytes()
    }

    pub fn disk_index_estimate_bytes(&self) -> usize {
        self.config.disk_index_estimate_bytes()
    }

    pub fn disk_capacity_bytes(&self) -> usize {
        self.config.disk_capacity_bytes
    }

    pub fn disk_read_bytes(&self) -> usize {
        self.cache.disk_read_bytes()
    }

    pub fn disk_write_bytes(&self) -> usize {
        self.cache.disk_write_bytes()
    }

    /// Reclaim only the RAM tier while preserving persistent NVMe pages.
    ///
    /// `evict_all` is intentional: Foyer 0.22's bulk `clear` drops entries but
    /// leaves its shard usage counter stale, whereas eviction decrements both
    /// resident bytes and accounting. Write-on-insertion mode has no eviction
    /// pipe, so already-persisted pages are not redundantly rewritten.
    pub fn clear_memory(&self) {
        self.cache.evict_all();
    }

    pub fn stats(&self) -> RangeCacheStats {
        RangeCacheStats {
            memory_hits: self.counters.memory_hits.load(Ordering::Relaxed),
            disk_hits: self.counters.disk_hits.load(Ordering::Relaxed),
            misses: self.counters.misses.load(Ordering::Relaxed),
            outer_fetches: self.counters.outer_fetches.load(Ordering::Relaxed),
            inserts: self.counters.inserts.load(Ordering::Relaxed),
            admission_rejections: self.counters.admission_rejections.load(Ordering::Relaxed),
            corrupt_entries: self.counters.corrupt_entries.load(Ordering::Relaxed),
        }
    }

    /// Read an exact byte range from an object pinned by metadata obtained
    /// from HEAD/list. Arbitrary requests are normalized into fixed aligned
    /// pages, so overlapping callers coalesce onto identical Foyer
    /// single-flight keys.
    pub async fn read_meta_range(
        &self,
        store: Arc<dyn ObjectStore>,
        meta: &ObjectMeta,
        range: Range<u64>,
    ) -> Result<Bytes, RangeCacheError> {
        let generation = PinnedObjectGeneration::from_meta(meta)?;
        self.read_object_range(store, &meta.location, &generation, meta.size, range)
            .await
    }

    /// Read an exact object range with an explicit generation precondition.
    ///
    /// `object_size` must come from the same pinned metadata snapshot. The
    /// method issues only page-aligned `GetOptions` range reads and sets either
    /// `version` or `If-Match` on every miss. Up to sixteen independent pages
    /// are fetched concurrently; requests for the same page share Foyer's
    /// single-flight future.
    pub async fn read_object_range(
        &self,
        store: Arc<dyn ObjectStore>,
        location: &ObjectPath,
        generation: &PinnedObjectGeneration,
        object_size: u64,
        range: Range<u64>,
    ) -> Result<Bytes, RangeCacheError> {
        self.read_object_range_limited(store, location, generation, object_size, range, None, None)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn read_object_range_limited(
        &self,
        store: Arc<dyn ObjectStore>,
        location: &ObjectPath,
        generation: &PinnedObjectGeneration,
        object_size: u64,
        range: Range<u64>,
        remote_limit: Option<Arc<Semaphore>>,
        source_counters: Option<Arc<PinnedObjectRangeCounters>>,
    ) -> Result<Bytes, RangeCacheError> {
        if range.start > range.end || range.end > object_size {
            return Err(RangeCacheError::InvalidKey(format!(
                "requested range {}..{} is outside object {} of size {object_size}",
                range.start, range.end, location
            )));
        }
        if range.is_empty() {
            return Ok(Bytes::new());
        }
        let requested_len = usize::try_from(range.end - range.start).map_err(|_| {
            RangeCacheError::InvalidKey(format!(
                "requested range length {} does not fit this platform",
                range.end - range.start
            ))
        })?;
        let page_size = u64::try_from(self.config.page_size_bytes).map_err(|_| {
            RangeCacheError::Configuration(
                "configured range-cache page size does not fit u64".into(),
            )
        })?;
        let first_page = range.start / page_size * page_size;
        let path_string = location.to_string();
        let generation_token = generation.cache_token()?;
        let pages = (first_page..range.end)
            .step_by(self.config.page_size_bytes)
            .map(|start| start..start.saturating_add(page_size).min(object_size));

        let fetched: Vec<(Range<u64>, Bytes)> = stream::iter(pages)
            .map(|page| {
                let cache = self.clone();
                let store = store.clone();
                let location = location.clone();
                let path_string = path_string.clone();
                let generation = generation.clone();
                let generation_token = generation_token.clone();
                let remote_limit = remote_limit.clone();
                let source_counters = source_counters.clone();
                async move {
                    let fetch_page = page.clone();
                    let value = cache
                        .get_or_fetch(
                            &path_string,
                            &generation_token,
                            page.clone(),
                            move || async move {
                                let _permit = match remote_limit {
                                    Some(limit) => Some(
                                        limit
                                            .acquire_owned()
                                            .await
                                            .expect("pinned range semaphore is never closed"),
                                    ),
                                    None => None,
                                };
                                // Count only requests that progress past the concurrency gate.
                                // A cancelled future waiting for a permit never reaches the
                                // authoritative store and must not inflate remote I/O metrics.
                                if let Some(counters) = source_counters.as_ref() {
                                    counters.remote_requests.fetch_add(1, Ordering::Relaxed);
                                }
                                let options = generation.get_options(fetch_page.clone());
                                let result = store
                                    .get_opts(&location, options)
                                    .await
                                    .map_err(anyhow::Error::from)?;
                                validate_get_result(
                                    &result,
                                    &location,
                                    &generation,
                                    object_size,
                                    &fetch_page,
                                )?;
                                let bytes = result.bytes().await.map_err(anyhow::Error::from)?;
                                let expected = usize::try_from(fetch_page.end - fetch_page.start)
                                    .map_err(|_| {
                                    anyhow::Error::new(RangeCacheError::ResponseValidation(
                                        format!(
                                            "authoritative range length {} does not fit this \
                                             platform for {}",
                                            fetch_page.end - fetch_page.start,
                                            location
                                        ),
                                    ))
                                })?;
                                if bytes.len() != expected {
                                    return Err(anyhow::Error::new(
                                        RangeCacheError::ResponseValidation(format!(
                                            "authoritative range {}..{} for {} returned {} bytes, \
                                             expected {expected}",
                                            fetch_page.start,
                                            fetch_page.end,
                                            location,
                                            bytes.len()
                                        )),
                                    ));
                                }
                                if let Some(counters) = source_counters.as_ref() {
                                    counters
                                        .remote_bytes
                                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                                }
                                Ok(bytes)
                            },
                        )
                        .await?;
                    Ok::<_, RangeCacheError>((page, value))
                }
            })
            .buffered(MAX_PARALLEL_PAGE_FETCHES)
            .try_collect()
            .await?;

        if let [(page, value)] = fetched.as_slice() {
            let start = usize::try_from(range.start - page.start).expect("page offset fits usize");
            let end = start + requested_len;
            return Ok(value.slice(start..end));
        }

        let mut output = BytesMut::with_capacity(requested_len);
        for (page, value) in fetched {
            let copy_start = range.start.max(page.start);
            let copy_end = range.end.min(page.end);
            if copy_start >= copy_end {
                continue;
            }
            let local_start =
                usize::try_from(copy_start - page.start).expect("page offset fits usize");
            let local_end = usize::try_from(copy_end - page.start).expect("page offset fits usize");
            output.extend_from_slice(&value[local_start..local_end]);
        }
        if output.len() != requested_len {
            return Err(RangeCacheError::InvalidKey(format!(
                "page assembly for {} {}..{} produced {} bytes, expected {requested_len}",
                location,
                range.start,
                range.end,
                output.len()
            )));
        }
        Ok(output.freeze())
    }

    /// Read one generation-pinned range.
    pub async fn get(
        &self,
        object_path: &str,
        generation: &str,
        range: Range<u64>,
    ) -> Result<Option<Bytes>, RangeCacheError> {
        let key = ImmutableRangeKey::new(object_path, generation, range)?;
        self.get_key(&key).await
    }

    /// Read one prevalidated range key.
    pub async fn get_key(&self, key: &ImmutableRangeKey) -> Result<Option<Bytes>, RangeCacheError> {
        key.validate()?;
        if !self.can_cache_key(key) {
            self.counters
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            self.counters.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let storage_key = self.storage_key(key);
        let Some((value, source)) = self.cache.get(&storage_key).await? else {
            self.counters.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        self.record_source(source);
        if self.validate_value(key, &value).is_err() {
            self.cache.remove(&storage_key);
            self.counters
                .corrupt_entries
                .fetch_add(1, Ordering::Relaxed);
            self.counters.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        Ok(Some(value))
    }

    /// Insert one generation-pinned range. `Ok(false)` means the value is
    /// valid but outside this cache's hard admission limits.
    pub fn insert(
        &self,
        object_path: &str,
        generation: &str,
        range: Range<u64>,
        value: Bytes,
    ) -> Result<bool, RangeCacheError> {
        let key = ImmutableRangeKey::new(object_path, generation, range)?;
        self.insert_key(&key, value)
    }

    pub fn insert_key(
        &self,
        key: &ImmutableRangeKey,
        value: Bytes,
    ) -> Result<bool, RangeCacheError> {
        key.validate()?;
        self.validate_value(key, &value)?;
        if !self.can_cache_key(key) {
            self.counters
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        self.cache.insert(self.storage_key(key), value);
        self.counters.inserts.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Read-through cache with Foyer single-flight on a cold key.
    ///
    /// The supplied future fetches the authoritative exact range. Its result
    /// must have precisely the key's length; a short/long object-store response
    /// is rejected and never admitted.
    pub async fn get_or_fetch<F, Fut, E>(
        &self,
        object_path: &str,
        generation: &str,
        range: Range<u64>,
        fetch: F,
    ) -> Result<Bytes, RangeCacheError>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<Bytes, E>> + Send + 'static,
        E: Into<anyhow::Error>,
    {
        let key = ImmutableRangeKey::new(object_path, generation, range)?;
        self.get_or_fetch_key(&key, fetch).await
    }

    pub async fn get_or_fetch_key<F, Fut, E>(
        &self,
        key: &ImmutableRangeKey,
        fetch: F,
    ) -> Result<Bytes, RangeCacheError>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<Bytes, E>> + Send + 'static,
        E: Into<anyhow::Error>,
    {
        key.validate()?;
        let expected = usize::try_from(key.len()).expect("key validation checked usize");
        if !self.can_cache_key(key) {
            self.counters
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            let value = fetch()
                .await
                .map_err(|error| RangeCacheError::Fetch(error.into()))?;
            self.validate_value(key, &value)?;
            self.counters.outer_fetches.fetch_add(1, Ordering::Relaxed);
            return Ok(value);
        }

        let storage_key = self.storage_key(key);
        let path = key.object_path.clone();
        let start = key.range.start;
        let end = key.range.end;
        let counters = self.counters.clone();
        let (value, source) = self
            .cache
            .get_or_fetch(&storage_key, move || async move {
                counters.outer_fetches.fetch_add(1, Ordering::Relaxed);
                let value = fetch().await.map_err(Into::into)?;
                if value.len() != expected {
                    return Err(anyhow::anyhow!(
                        "authoritative range length mismatch for {path} {start}..{end}: \
                         expected {expected}, found {}",
                        value.len()
                    ));
                }
                Ok::<Bytes, anyhow::Error>(value)
            })
            .await?;
        self.record_source(source);
        if let Err(error) = self.validate_value(key, &value) {
            self.cache.remove(&storage_key);
            self.counters
                .corrupt_entries
                .fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }
        Ok(value)
    }

    /// Remove the exact generation/range key from both tiers.
    pub fn remove(
        &self,
        object_path: &str,
        generation: &str,
        range: Range<u64>,
    ) -> Result<(), RangeCacheError> {
        let key = ImmutableRangeKey::new(object_path, generation, range)?;
        self.cache.remove(&self.storage_key(&key));
        Ok(())
    }

    /// Destroy all resident and on-disk entries. The cache remains a
    /// reconstructible optimization; authoritative objects are untouched.
    pub async fn clear(&self) -> Result<(), RangeCacheError> {
        // Evict first so Foyer's in-memory usage counters are decremented.
        // `HybridCache::clear` currently delegates to `Cache::clear`, whose
        // accounting is stale in Foyer 0.22 after draining the shards.
        self.clear_memory();
        self.cache.destroy_storage().await?;
        Ok(())
    }

    /// Gracefully flush/close the Foyer store. Drop all clones before reopening
    /// the same directory so the process-exclusive directory lock is released.
    pub async fn close(&self) -> Result<(), RangeCacheError> {
        self.cache.close().await?;
        Ok(())
    }

    fn can_cache_key(&self, key: &ImmutableRangeKey) -> bool {
        let Ok(len) = usize::try_from(key.len()) else {
            return false;
        };
        if len == 0 || len > self.config.max_entry_bytes {
            return false;
        }
        // The persistent index itself lives in RAM. Admitting arbitrary tiny
        // values would make its entry count proportional to device bytes
        // rather than canonical pages and invalidate the configured cap.
        if self.is_hybrid() && len < self.config.page_size_bytes {
            return false;
        }
        self.is_hybrid()
            || self.storage_key(key).len().saturating_add(len) <= self.memory_capacity_bytes()
    }

    fn validate_value(
        &self,
        key: &ImmutableRangeKey,
        value: &Bytes,
    ) -> Result<(), RangeCacheError> {
        let expected = usize::try_from(key.len()).expect("key validation checked usize");
        if value.len() == expected {
            return Ok(());
        }
        Err(RangeCacheError::ValueLengthMismatch {
            path: key.object_path.clone(),
            start: key.range.start,
            end: key.range.end,
            expected,
            actual: value.len(),
        })
    }

    fn storage_key(&self, key: &ImmutableRangeKey) -> String {
        let mut hasher = blake3::Hasher::new();
        hash_part(&mut hasher, CACHE_KEY_FORMAT);
        hash_part(&mut hasher, self.config.key_namespace.as_bytes());
        hash_part(&mut hasher, key.object_path.as_bytes());
        hash_part(&mut hasher, key.generation.as_bytes());
        hasher.update(&key.range.start.to_le_bytes());
        hasher.update(&key.range.end.to_le_bytes());
        format!(
            "nr1:{}:{:016x}:{:016x}",
            hasher.finalize().to_hex(),
            key.range.start,
            key.range.end
        )
    }

    fn record_source(&self, source: Source) {
        match source {
            Source::Memory => {
                self.counters.memory_hits.fetch_add(1, Ordering::Relaxed);
            }
            Source::Disk => {
                self.counters.disk_hits.fetch_add(1, Ordering::Relaxed);
            }
            Source::Outer => {
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Generation-pinned, range-only view of one object-store object.
///
/// Construction performs (or consumes) one HEAD and freezes its location,
/// size, and version/ETag. Every authoritative GET carries that generation
/// precondition and its response shape is checked before bytes can reach a
/// decoder or the shared RAM/NVMe cache. Cache failures fall back to a direct
/// GET only when [`RangeCacheError::is_local_failure`] identifies them as
/// reconstructible local failures.
#[derive(Default)]
struct PinnedObjectRangeCounters {
    logical_requests: AtomicU64,
    logical_bytes: AtomicU64,
    remote_requests: AtomicU64,
    remote_bytes: AtomicU64,
}

/// Per-source I/O accounting. Logical bytes are exact caller-visible ranges;
/// remote bytes are authoritative object-store payloads and may be larger due
/// to aligned cache pages. Cache hits increase only the logical counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PinnedObjectRangeStats {
    pub logical_requests: u64,
    pub logical_bytes: u64,
    pub remote_requests: u64,
    pub remote_bytes: u64,
}

#[derive(Clone)]
pub struct PinnedObjectRangeSource {
    store: Arc<dyn ObjectStore>,
    meta: ObjectMeta,
    generation: PinnedObjectGeneration,
    range_cache: Option<ImmutableRangeCache>,
    remote_limit: Arc<Semaphore>,
    counters: Arc<PinnedObjectRangeCounters>,
}

impl fmt::Debug for PinnedObjectRangeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedObjectRangeSource")
            .field("location", &self.meta.location)
            .field("object_size", &self.meta.size)
            .field("generation", &self.generation)
            .field("range_cache", &self.range_cache.is_some())
            .field("remote_concurrency", &MAX_PINNED_OBJECT_RANGE_READS)
            .field("stats", &self.stats())
            .finish()
    }
}

impl PinnedObjectRangeSource {
    /// HEAD `location` exactly once and open it against the process-wide
    /// RAM/NVMe page cache.
    pub async fn open(
        store: Arc<dyn ObjectStore>,
        location: ObjectPath,
    ) -> crate::error::Result<Self> {
        let meta = store.head(&location).await?;
        if meta.location != location {
            return Err(crate::error::Error::Corrupted {
                path: location.to_string(),
                detail: format!(
                    "HEAD returned metadata for unexpected location {}",
                    meta.location
                ),
            });
        }
        Self::from_meta(store, meta).await
    }

    /// Open from metadata already fixed by a caller's HEAD/list operation.
    pub async fn from_meta(
        store: Arc<dyn ObjectStore>,
        meta: ObjectMeta,
    ) -> crate::error::Result<Self> {
        let range_cache = resolved_shared_range_cache(&meta).await?;
        Self::from_meta_with_cache(store, meta, range_cache)
    }

    /// Open metadata for a content-addressed/create-only object path.
    ///
    /// Some local/object-store adapters do not expose an ETag or version even
    /// though NamiDB SST names contain a fresh UUID and are never overwritten.
    /// Callers must opt into that stronger path-level invariant explicitly;
    /// mutable locations must use [`Self::from_meta`].
    pub async fn from_immutable_meta(
        store: Arc<dyn ObjectStore>,
        meta: ObjectMeta,
    ) -> crate::error::Result<Self> {
        let range_cache = resolved_shared_range_cache(&meta).await?;
        Self::from_meta_with_cache_and_generation(
            store,
            meta,
            PinnedObjectGeneration::ImmutablePath,
            range_cache,
        )
    }

    /// Open a NamiDB create-only UUID object, preferring a backend generation
    /// token and falling back to the explicit immutable-path invariant only
    /// when metadata exposes neither version nor ETag.
    pub async fn from_create_only_meta(
        store: Arc<dyn ObjectStore>,
        meta: ObjectMeta,
    ) -> crate::error::Result<Self> {
        match PinnedObjectGeneration::from_meta(&meta) {
            Ok(generation) => {
                let range_cache = resolved_shared_range_cache(&meta).await?;
                Self::from_meta_with_cache_and_generation(store, meta, generation, range_cache)
            }
            // Neither version nor ETag: immutability is inferred from NamiDB's
            // create-only naming, not asserted by a backend generation. The
            // shared cache would key these reads by path+range alone — a
            // constant token — so two contents at one path would collide, and
            // no remote precondition exists to catch a violated assumption.
            // Read authoritatively and never touch the shared cache; callers
            // that can genuinely assert path immutability use
            // `from_immutable_meta` instead.
            Err(_) => Self::from_meta_with_cache_and_generation(
                store,
                meta,
                PinnedObjectGeneration::ImmutablePath,
                None,
            ),
        }
    }

    /// Explicit cache injection for readers that already own cache policy and
    /// for deterministic tests.
    pub(crate) fn from_meta_with_cache(
        store: Arc<dyn ObjectStore>,
        meta: ObjectMeta,
        range_cache: Option<ImmutableRangeCache>,
    ) -> crate::error::Result<Self> {
        let generation = PinnedObjectGeneration::from_meta(&meta).map_err(|error| {
            crate::error::Error::precondition(format!(
                "pinned range reads require an object version or ETag for {}: {error}",
                meta.location
            ))
        })?;
        Self::from_meta_with_cache_and_generation(store, meta, generation, range_cache)
    }

    fn from_meta_with_cache_and_generation(
        store: Arc<dyn ObjectStore>,
        meta: ObjectMeta,
        generation: PinnedObjectGeneration,
        range_cache: Option<ImmutableRangeCache>,
    ) -> crate::error::Result<Self> {
        if meta.location.as_ref().is_empty() {
            return Err(crate::error::Error::precondition(
                "pinned object range source requires a non-empty location",
            ));
        }
        generation.cache_token().map_err(|error| {
            crate::error::Error::precondition(format!(
                "invalid generation pin for {}: {error}",
                meta.location
            ))
        })?;
        Ok(Self {
            store,
            meta,
            generation,
            range_cache,
            remote_limit: Arc::new(Semaphore::new(MAX_PINNED_OBJECT_RANGE_READS)),
            counters: Arc::new(PinnedObjectRangeCounters::default()),
        })
    }

    pub fn metadata(&self) -> &ObjectMeta {
        &self.meta
    }

    pub fn location(&self) -> &ObjectPath {
        &self.meta.location
    }

    pub fn object_size(&self) -> u64 {
        self.meta.size
    }

    pub fn generation(&self) -> &PinnedObjectGeneration {
        &self.generation
    }

    /// Whether reads may consult the shared range cache. False for inferred
    /// immutability, whose constructor refuses the cache by contract. A test
    /// observability hook: production code never branches on it.
    #[cfg(test)]
    pub(crate) fn has_range_cache(&self) -> bool {
        self.range_cache.is_some()
    }

    pub fn stats(&self) -> PinnedObjectRangeStats {
        PinnedObjectRangeStats {
            logical_requests: self.counters.logical_requests.load(Ordering::Relaxed),
            logical_bytes: self.counters.logical_bytes.load(Ordering::Relaxed),
            remote_requests: self.counters.remote_requests.load(Ordering::Relaxed),
            remote_bytes: self.counters.remote_bytes.load(Ordering::Relaxed),
        }
    }

    /// Read one exact half-open range from the frozen object generation.
    pub async fn read_range(&self, range: Range<u64>) -> crate::error::Result<Bytes> {
        self.validate_range(&range)?;
        self.counters
            .logical_requests
            .fetch_add(1, Ordering::Relaxed);
        if range.is_empty() {
            return Ok(Bytes::new());
        }

        let bytes = if let Some(cache) = &self.range_cache {
            match cache
                .read_object_range_limited(
                    self.store.clone(),
                    &self.meta.location,
                    &self.generation,
                    self.meta.size,
                    range.clone(),
                    Some(self.remote_limit.clone()),
                    Some(self.counters.clone()),
                )
                .await
            {
                Ok(bytes) => self.validate_length(range.clone(), bytes)?,
                Err(error) if error.is_local_failure() => {
                    tracing::warn!(
                        path = %self.meta.location,
                        start = range.start,
                        end = range.end,
                        error = %error,
                        "local range-cache failure; retrying pinned authoritative range"
                    );
                    self.direct_read(range.clone()).await?
                }
                Err(error) => {
                    return Err(pinned_range_cache_error(&self.meta.location, error));
                }
            }
        } else {
            self.direct_read(range.clone()).await?
        };
        self.counters
            .logical_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    /// Read a batch in input order. Physical authoritative GETs across the
    /// complete source are additionally guarded by the shared 16-permit
    /// semaphore, including page-normalized cache misses.
    pub async fn read_ranges(&self, ranges: &[Range<u64>]) -> crate::error::Result<Vec<Bytes>> {
        stream::iter(ranges.iter().cloned())
            .map(|range| self.read_range(range))
            .buffered(MAX_PINNED_OBJECT_RANGE_READS)
            .try_collect()
            .await
    }

    fn validate_range(&self, range: &Range<u64>) -> crate::error::Result<()> {
        if range.start > range.end || range.end > self.meta.size {
            return Err(crate::error::Error::Corrupted {
                path: self.meta.location.to_string(),
                detail: format!(
                    "requested range {}..{} is outside pinned object size {}",
                    range.start, range.end, self.meta.size
                ),
            });
        }
        usize::try_from(range.end - range.start).map_err(|_| crate::error::Error::Corrupted {
            path: self.meta.location.to_string(),
            detail: format!(
                "requested range length {} does not fit this platform",
                range.end - range.start
            ),
        })?;
        Ok(())
    }

    async fn direct_read(&self, range: Range<u64>) -> crate::error::Result<Bytes> {
        let _permit = self
            .remote_limit
            .acquire()
            .await
            .expect("pinned range semaphore is never closed");
        self.counters
            .remote_requests
            .fetch_add(1, Ordering::Relaxed);
        let options = self.generation.get_options(range.clone());
        let result = self.store.get_opts(&self.meta.location, options).await?;
        validate_get_result(
            &result,
            &self.meta.location,
            &self.generation,
            self.meta.size,
            &range,
        )
        .map_err(|error| pinned_range_cache_error(&self.meta.location, error))?;
        let bytes = result.bytes().await?;
        let bytes = self.validate_length(range, bytes)?;
        self.counters
            .remote_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    fn validate_length(&self, range: Range<u64>, bytes: Bytes) -> crate::error::Result<Bytes> {
        let expected = usize::try_from(range.end - range.start)
            .expect("source range validation checked usize");
        if bytes.len() != expected {
            return Err(crate::error::Error::Corrupted {
                path: self.meta.location.to_string(),
                detail: format!(
                    "range {}..{} returned {} bytes, expected {expected}",
                    range.start,
                    range.end,
                    bytes.len()
                ),
            });
        }
        Ok(bytes)
    }
}

async fn resolved_shared_range_cache(
    meta: &ObjectMeta,
) -> crate::error::Result<Option<ImmutableRangeCache>> {
    match shared_range_cache().await {
        Ok(cache) => Ok(cache),
        Err(error) if error.is_local_failure() => {
            tracing::warn!(
                path = %meta.location,
                error = %error,
                "shared range cache unavailable; using pinned authoritative ranges"
            );
            Ok(None)
        }
        Err(error) => Err(pinned_range_cache_error(&meta.location, error)),
    }
}

fn pinned_range_cache_error(location: &ObjectPath, error: RangeCacheError) -> crate::error::Error {
    if error.is_response_validation_failure() {
        return crate::error::Error::Corrupted {
            path: location.to_string(),
            detail: error.to_string(),
        };
    }
    crate::error::Error::ObjectStore(object_store::Error::Generic {
        store: "pinned-object-range",
        source: Box::new(error),
    })
}

static SHARED_RANGE_CACHE: tokio::sync::OnceCell<Option<ImmutableRangeCache>> =
    tokio::sync::OnceCell::const_new();

/// Lazily open the process-wide immutable range cache.
///
/// No environment configuration yields `Ok(None)`. Configuration is sampled
/// once on the first successful initialization, matching the other shared
/// cache tiers.
pub async fn shared_range_cache() -> Result<Option<ImmutableRangeCache>, RangeCacheError> {
    SHARED_RANGE_CACHE
        .get_or_try_init(|| async {
            let Some(config) = RangeCacheConfig::from_env()? else {
                return Ok(None);
            };
            ImmutableRangeCache::open(config).await.map(Some)
        })
        .await
        .cloned()
}

/// Synchronous usage probe for the aggregate cache-budget reporter. It never
/// initializes the async range cache as a side effect.
pub(crate) fn shared_range_cache_usage_bytes() -> usize {
    SHARED_RANGE_CACHE
        .get()
        .and_then(Option::as_ref)
        .map_or(0, ImmutableRangeCache::memory_usage_bytes)
}

/// Snapshot an already-initialized shared cache without creating one.
///
/// `None` means either no range read has initialized the tier yet or the
/// sampled configuration disabled it. Both states deliberately render as
/// `enabled=0` at the server metrics layer.
pub fn shared_range_cache_snapshot() -> Option<SharedRangeCacheSnapshot> {
    SHARED_RANGE_CACHE
        .get()
        .and_then(Option::as_ref)
        .map(|cache| SharedRangeCacheSnapshot {
            ram_budget_bytes: cache.ram_budget_bytes(),
            memory_capacity_bytes: cache.memory_capacity_bytes(),
            memory_usage_bytes: cache.memory_usage_bytes(),
            write_buffer_reservation_bytes: cache.write_buffer_reservation_bytes(),
            disk_index_estimate_bytes: cache.disk_index_estimate_bytes(),
            disk_capacity_bytes: cache.disk_capacity_bytes(),
            disk_read_bytes: cache.disk_read_bytes(),
            disk_write_bytes: cache.disk_write_bytes(),
            stats: cache.stats(),
        })
}

/// Reclaim the shared range cache's RAM tier without initializing it and
/// without destroying its persistent NVMe contents.
pub(crate) fn clear_shared_range_cache_memory() {
    if let Some(cache) = SHARED_RANGE_CACHE.get().and_then(Option::as_ref) {
        cache.clear_memory();
    }
}

fn lock_cache_directory(path: &Path) -> Result<File, RangeCacheError> {
    let lock_path = path.join(".namidb-range-cache.lock");
    // The lock file's contents are irrelevant, but it must never be truncated:
    // a concurrent holder may already be blocked on it.
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    file.try_lock_exclusive()
        .map_err(|source| RangeCacheError::DirectoryBusy {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(file)
}

fn hash_part(hasher: &mut blake3::Hasher, part: &[u8]) {
    hasher.update(&(part.len() as u64).to_le_bytes());
    hasher.update(part);
}

fn default_write_buffer_bytes(ram_budget_bytes: usize) -> usize {
    if ram_budget_bytes == 0 {
        return 0;
    }
    (ram_budget_bytes / 4)
        .clamp(PAGE_BYTES, DEFAULT_NVME_CACHE_WRITE_BUFFER_BYTES)
        .min(ram_budget_bytes)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, RangeCacheError> {
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .ok_or_else(|| RangeCacheError::Configuration("alignment calculation overflow".into()))
}

fn align_down(value: usize, alignment: usize) -> usize {
    value - value % alignment
}

fn optional_nonempty_env(name: &str) -> Result<Option<String>, RangeCacheError> {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.trim();
            Ok((!value.is_empty()).then(|| value.to_owned()))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(RangeCacheError::Configuration(format!(
            "{name} is not valid UTF-8"
        ))),
    }
}

fn optional_usize_env(name: &str) -> Result<Option<usize>, RangeCacheError> {
    match std::env::var(name) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => value.trim().parse::<usize>().map(Some).map_err(|error| {
            RangeCacheError::Configuration(format!("{name} must be an exact byte count: {error}"))
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(RangeCacheError::Configuration(format!(
            "{name} is not valid UTF-8"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use object_store::memory::InMemory;
    use object_store::{ObjectStoreExt, PutPayload};

    use super::*;

    fn memory_config(bytes: usize) -> RangeCacheConfig {
        let mut config = RangeCacheConfig::memory_only(bytes);
        config.max_entry_bytes = bytes / 2;
        config.page_size_bytes = config.max_entry_bytes.min(4096);
        config.key_namespace = "test-memory".into();
        config
    }

    fn hybrid_config(path: &Path) -> RangeCacheConfig {
        let mut config = RangeCacheConfig::hybrid(2 * 1024 * 1024, path, 32 * 1024 * 1024);
        config.max_entry_bytes = 64 * 1024;
        config.page_size_bytes = 16 * 1024;
        config.block_size_bytes = 1024 * 1024;
        config.write_buffer_bytes = 256 * 1024;
        config.key_namespace = "test-hybrid".into();
        config
    }

    #[test]
    fn generation_and_range_are_part_of_the_stable_key() {
        let config = memory_config(1024 * 1024);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let cache = runtime.block_on(ImmutableRangeCache::open(config)).unwrap();
        let a = ImmutableRangeKey::new("bucket/a", "etag-1", 0..4096).unwrap();
        let b = ImmutableRangeKey::new("bucket/a", "etag-2", 0..4096).unwrap();
        let c = ImmutableRangeKey::new("bucket/a", "etag-1", 4096..8192).unwrap();
        assert_ne!(cache.storage_key(&a), cache.storage_key(&b));
        assert_ne!(cache.storage_key(&a), cache.storage_key(&c));
        assert_eq!(cache.storage_key(&a), cache.storage_key(&a));

        let mut other_config = memory_config(1024 * 1024);
        other_config.key_namespace = "another-store".into();
        let other = runtime
            .block_on(ImmutableRangeCache::open(other_config))
            .unwrap();
        assert_ne!(
            cache.storage_key(&a),
            other.storage_key(&a),
            "store namespace must isolate identical paths and generations"
        );
    }

    #[tokio::test]
    async fn memory_only_cache_is_bounded_and_rejects_oversized_ranges() {
        let cache = ImmutableRangeCache::open(memory_config(16 * 1024))
            .await
            .unwrap();
        assert!(!cache.is_hybrid());
        let first = Bytes::from(vec![1; 4096]);
        assert!(cache
            .insert("objects/a", "v1", 0..4096, first.clone())
            .unwrap());
        assert_eq!(
            cache.get("objects/a", "v1", 0..4096).await.unwrap(),
            Some(first)
        );

        let too_large = Bytes::from(vec![2; 8193]);
        assert!(!cache.insert("objects/b", "v1", 0..8193, too_large).unwrap());
        assert!(cache.memory_usage_bytes() <= cache.memory_capacity_bytes());
        assert_eq!(cache.stats().admission_rejections, 1);

        cache.clear_memory();
        assert_eq!(cache.memory_usage_bytes(), 0);
        assert!(cache
            .get("objects/a", "v1", 0..4096)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn mismatched_range_payload_is_never_admitted() {
        let cache = ImmutableRangeCache::open(memory_config(64 * 1024))
            .await
            .unwrap();
        let error = cache
            .insert("objects/a", "v1", 10..20, Bytes::from_static(b"short"))
            .unwrap_err();
        assert!(matches!(
            error,
            RangeCacheError::ValueLengthMismatch {
                expected: 10,
                actual: 5,
                ..
            }
        ));
        assert!(cache
            .get("objects/a", "v1", 10..20)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_read_through_miss_is_single_flight() {
        let cache = ImmutableRangeCache::open(memory_config(1024 * 1024))
            .await
            .unwrap();
        let fetches = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..16 {
            let cache = cache.clone();
            let fetches = fetches.clone();
            tasks.push(tokio::spawn(async move {
                cache
                    .get_or_fetch("objects/a", "v1", 0..4096, move || async move {
                        fetches.fetch_add(1, AtomicOrdering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        Ok::<_, anyhow::Error>(Bytes::from(vec![7; 4096]))
                    })
                    .await
                    .unwrap()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap().len(), 4096);
        }
        assert_eq!(fetches.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(cache.stats().outer_fetches, 1);
        assert!(cache.memory_usage_bytes() <= cache.memory_capacity_bytes());
    }

    #[tokio::test]
    async fn object_reader_pins_generation_and_coalesces_overlapping_ranges() {
        let mut config = memory_config(1024 * 1024);
        config.page_size_bytes = 4096;
        let cache = ImmutableRangeCache::open(config).await.unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectPath::from("objects/paged");
        let body = Bytes::from((0..16 * 1024).map(|value| value as u8).collect::<Vec<_>>());
        store
            .put(&location, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let meta = store.head(&location).await.unwrap();

        let first = cache
            .read_meta_range(store.clone(), &meta, 100..300)
            .await
            .unwrap();
        assert_eq!(first, body.slice(100..300));
        let second = cache
            .read_meta_range(store.clone(), &meta, 250..500)
            .await
            .unwrap();
        assert_eq!(second, body.slice(250..500));
        assert_eq!(
            cache.stats().outer_fetches,
            1,
            "both subranges normalize to the same 4 KiB page"
        );
        assert_eq!(cache.stats().memory_hits, 1);

        let crossing = cache
            .read_meta_range(store.clone(), &meta, 4000..5000)
            .await
            .unwrap();
        assert_eq!(crossing, body.slice(4000..5000));
        assert_eq!(
            cache.stats().outer_fetches,
            2,
            "the already-hot first page is reused and only the next page is fetched"
        );

        let wrong = PinnedObjectGeneration::ETag("\"not-the-object-etag\"".into());
        let error = cache
            .read_object_range(store, &location, &wrong, meta.size, 8192..8200)
            .await
            .unwrap_err();
        assert!(
            matches!(error, RangeCacheError::Foyer(_)),
            "object-store precondition failure is surfaced, got {error:?}"
        );
    }

    #[tokio::test]
    async fn create_only_source_supports_backends_without_generation_metadata() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectPath::from("immutable/uuid-sidecar");
        let body = Bytes::from_static(b"0123456789abcdef");
        store
            .put(&location, PutPayload::from_bytes(body.clone()))
            .await
            .unwrap();
        let mut meta = store.head(&location).await.unwrap();
        meta.e_tag = None;
        meta.version = None;

        assert!(
            PinnedObjectRangeSource::from_meta(store.clone(), meta.clone())
                .await
                .is_err(),
            "mutable-path construction must still require a remote generation pin"
        );
        let source = PinnedObjectRangeSource::from_create_only_meta(store, meta)
            .await
            .unwrap();
        assert_eq!(source.read_range(3..11).await.unwrap(), body.slice(3..11));
        assert_eq!(source.generation(), &PinnedObjectGeneration::ImmutablePath);
    }

    /// An inferred pin has no backend generation and no remote precondition,
    /// so the shared cache — whose key would collapse to path+range — must
    /// never be consulted. Pass-through reads make a violated create-only
    /// assumption observable instead of serving a stale collision.
    #[tokio::test]
    async fn inferred_immutable_pin_is_never_cached() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectPath::from(format!("immutable/{}/inferred-pin", uuid::Uuid::now_v7()));
        let first = Bytes::from_static(b"first-generation");
        store
            .put(&location, PutPayload::from_bytes(first.clone()))
            .await
            .unwrap();
        let mut meta = store.head(&location).await.unwrap();
        meta.e_tag = None;
        meta.version = None;

        let source = PinnedObjectRangeSource::from_create_only_meta(store.clone(), meta)
            .await
            .unwrap();
        assert_eq!(source.generation(), &PinnedObjectGeneration::ImmutablePath);
        assert!(
            !source.has_range_cache(),
            "inferred immutability must refuse the shared range cache"
        );
        assert_eq!(source.read_range(0..16).await.unwrap(), first);

        // Same store, same path, same length: only pass-through reads can
        // observe the replacement. A cached inferred pin would serve `first`.
        let second = Bytes::from_static(b"other-generation");
        assert_eq!(second.len(), first.len());
        store
            .put(&location, PutPayload::from_bytes(second.clone()))
            .await
            .unwrap();
        assert_eq!(source.read_range(0..16).await.unwrap(), second);
    }

    #[tokio::test]
    async fn hybrid_cache_recovers_range_from_disk_after_restart() {
        let directory = tempfile::tempdir().unwrap();
        let config = hybrid_config(directory.path());
        let value = Bytes::from(vec![9; 32 * 1024]);

        let cache = ImmutableRangeCache::open(config.clone()).await.unwrap();
        assert!(cache.is_hybrid());
        assert_eq!(
            cache.memory_capacity_bytes()
                + cache.write_buffer_reservation_bytes()
                + cache.disk_index_estimate_bytes(),
            cache.ram_budget_bytes()
        );
        assert!(
            !cache
                .insert(
                    "objects/tiny",
                    "etag-tiny",
                    0..1024,
                    Bytes::from(vec![1; 1024])
                )
                .unwrap(),
            "tiny persistent entries would invalidate the disk-index RAM bound"
        );
        assert!(cache
            .insert(
                "objects/a",
                "etag-a",
                100..(100 + value.len() as u64),
                value.clone()
            )
            .unwrap());
        cache.close().await.unwrap();
        drop(cache);

        let reopened = ImmutableRangeCache::open(config).await.unwrap();
        let got = reopened
            .get("objects/a", "etag-a", 100..(100 + value.len() as u64))
            .await
            .unwrap();
        assert_eq!(got, Some(value));
        assert_eq!(reopened.stats().disk_hits, 1);
        assert!(reopened.memory_usage_bytes() <= reopened.memory_capacity_bytes());
        reopened.close().await.unwrap();
    }

    #[tokio::test]
    async fn hybrid_directory_is_process_exclusive() {
        let directory = tempfile::tempdir().unwrap();
        let config = hybrid_config(directory.path());
        let first = ImmutableRangeCache::open(config.clone()).await.unwrap();
        let error = ImmutableRangeCache::open(config).await.unwrap_err();
        assert!(matches!(error, RangeCacheError::DirectoryBusy { .. }));
        first.close().await.unwrap();
    }

    #[test]
    fn invalid_hybrid_limits_fail_before_foyer_allocates() {
        let directory = tempfile::tempdir().unwrap();
        let config = RangeCacheConfig::hybrid(2 * 1024 * 1024, directory.path(), 32 * 1024 * 1024);
        assert!(
            config.validate().is_err(),
            "persistent caches require an explicit store namespace"
        );

        let mut config = hybrid_config(directory.path());
        config.max_entry_bytes = config.block_size_bytes;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, RangeCacheError::Configuration(_)));

        let mut config = hybrid_config(directory.path());
        config.page_size_bytes = 1;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, RangeCacheError::Configuration(_)));

        let mut config = hybrid_config(directory.path());
        config.write_buffer_bytes += 1;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, RangeCacheError::Configuration(_)));

        let mut config = hybrid_config(directory.path());
        config.write_buffer_bytes = config.ram_budget_bytes + 1;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, RangeCacheError::Configuration(_)));

        let mut config = hybrid_config(directory.path());
        config.write_buffer_bytes = config.ram_budget_bytes / FOYER_WRITE_BUFFER_COUNT;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, RangeCacheError::Configuration(_)));
    }
}
