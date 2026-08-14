//! Bounded-memory external builder for [`super::VectorV5Reader`].
//!
//! Compaction already produces authoritative winners in `NodeId` order. This
//! collector makes that ordering part of its contract: [`push`](VectorV5ExternalCollector::push)
//! rejects duplicates and out-of-order IDs, eliminating a corpus-sized sort.
//! Rows are checksummed into a scratch spool. `finish` recursively scans and
//! partitions that spool, retaining only `O(dim)`, a bounded quantile sample,
//! and one output page in memory.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use bincode::Options;
use crc32fast::Hasher;
use namidb_core::quantize::quantize_i8;
use namidb_core::Value;
use tempfile::NamedTempFile;
use xxhash_rust::xxh3::xxh3_64;

use super::{
    encode_vector_filter_value, metric_name, validate_build_options, BlockRef, CentroidNode,
    Footer, NavPage, PageRef, VectorGraphBuildStats, VectorIndexDescriptor, VectorMetric,
    VectorV5BuildOptions, FORMAT_VERSION, MAGIC_V5, MAX_COMPRESSED_BLOCK_BYTES, MAX_FOOTER_BYTES,
    MAX_RAW_BLOCK_BYTES, TRAILER_MAGIC,
};
use crate::error::Error;

const SPOOL_MAGIC: &[u8; 8] = b"NV5SPL01";
const SPOOL_HEADER_LEN: u64 = 12;
const DEFAULT_MEMORY_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_SAMPLE_ROWS: usize = 16_384;
const MIN_MEMORY_BYTES: usize = 64 * 1024;
const MAX_EXTERNAL_BRANCH_FACTOR: usize = 64;
const MAX_FILTER_COMPONENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_FILTERS_PER_ROW: usize = 65_536;
const MAX_PARTITION_DEPTH: usize = 64;

/// Explicit scratch and memory controls for an external vector build.
///
/// Production callers normally use [`Self::from_env`]. Tests should pass an
/// explicit configuration and never mutate process-global environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorV5ExternalBuildConfig {
    pub scratch_dir: PathBuf,
    pub memory_budget_bytes: usize,
    pub quantile_sample_rows: usize,
}

impl Default for VectorV5ExternalBuildConfig {
    fn default() -> Self {
        Self {
            scratch_dir: std::env::temp_dir(),
            memory_budget_bytes: DEFAULT_MEMORY_BYTES,
            quantile_sample_rows: DEFAULT_SAMPLE_ROWS,
        }
    }
}

impl VectorV5ExternalBuildConfig {
    /// Read `NAMIDB_INDEX_BUILD_SPOOL_DIR` (index-specific override),
    /// `NAMIDB_COMPACTION_SPOOL_DIR` (legacy/requested alias), then the engine
    /// wide `NAMIDB_SPOOL_DIR`, plus `NAMIDB_INDEX_BUILD_MEMORY_BYTES`.
    pub fn from_env() -> Result<Self, Error> {
        let mut config = Self::default();
        if let Some(path) = std::env::var_os("NAMIDB_INDEX_BUILD_SPOOL_DIR")
            .or_else(|| std::env::var_os("NAMIDB_COMPACTION_SPOOL_DIR"))
            .or_else(|| std::env::var_os("NAMIDB_SPOOL_DIR"))
        {
            config.scratch_dir = PathBuf::from(path);
        }
        if let Ok(raw) = std::env::var("NAMIDB_INDEX_BUILD_MEMORY_BYTES") {
            config.memory_budget_bytes = raw.parse::<usize>().map_err(|error| {
                Error::invariant(format!(
                    "NAMIDB_INDEX_BUILD_MEMORY_BYTES must be an integer: {error}"
                ))
            })?;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Error> {
        if self.memory_budget_bytes < MIN_MEMORY_BYTES {
            return Err(Error::invariant(format!(
                "vector v5 external memory budget {} is below the minimum {MIN_MEMORY_BYTES}",
                self.memory_budget_bytes
            )));
        }
        if self.quantile_sample_rows < 2 {
            return Err(Error::invariant(
                "vector v5 external quantile_sample_rows must be at least 2",
            ));
        }
        std::fs::create_dir_all(&self.scratch_dir).map_err(|error| {
            Error::invariant(format!(
                "cannot create vector v5 spool directory {}: {error}",
                self.scratch_dir.display()
            ))
        })?;
        if !self.scratch_dir.is_dir() {
            return Err(Error::invariant(format!(
                "vector v5 spool path {} is not a directory",
                self.scratch_dir.display()
            )));
        }
        Ok(())
    }
}

/// Logical resource counters from a completed external build.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VectorV5ExternalBuildMetrics {
    pub input_rows: u64,
    pub indexed_rows: u64,
    pub page_count: u64,
    pub partition_count: u64,
    pub max_partition_depth: usize,
    pub peak_sample_rows: usize,
    /// Conservative accounting of builder-owned live allocations. It excludes
    /// the caller-owned vector passed to `push` and OS page cache.
    pub peak_logical_memory_bytes: usize,
    /// Peak transient scan/sample/page workspace, excluding the final compact
    /// footer metadata. This value is independent of corpus cardinality.
    pub peak_working_memory_bytes: usize,
    /// Final centroid tree + page directory retained by the reader.
    pub resident_metadata_bytes: usize,
    pub scratch_bytes_written: u64,
    pub artifact_bytes_written: u64,
    pub effective_target_rows_per_page: usize,
}

impl VectorV5ExternalBuildMetrics {
    fn observe_working(&mut self, bytes: usize) {
        self.peak_working_memory_bytes = self.peak_working_memory_bytes.max(bytes);
        self.peak_logical_memory_bytes = self
            .peak_logical_memory_bytes
            .max(self.resident_metadata_bytes.saturating_add(bytes));
    }
}

/// Seekable, unlinked temporary artifact ready for upload to object storage.
#[derive(Debug)]
pub struct VectorV5ExternalArtifact {
    pub file: File,
    pub len: u64,
    pub stats: VectorGraphBuildStats,
    pub metrics: VectorV5ExternalBuildMetrics,
}

impl VectorV5ExternalArtifact {
    /// Rewind before handing the file to a streaming uploader.
    pub fn rewind(&mut self) -> Result<(), Error> {
        self.file.seek(SeekFrom::Start(0))?;
        Ok(())
    }
}

/// Winner-stream collector. IDs must be strictly increasing.
pub struct VectorV5ExternalCollector {
    config: VectorV5ExternalBuildConfig,
    spool: NamedTempFile,
    dim: Option<usize>,
    input_rows: u64,
    last_id: Option<[u8; 16]>,
    filter_properties: BTreeSet<String>,
    filter_property_bytes: usize,
    max_record_wire_bytes: usize,
    peak_logical_memory_bytes: usize,
    scratch_bytes_written: u64,
}

impl std::fmt::Debug for VectorV5ExternalCollector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorV5ExternalCollector")
            .field("scratch_dir", &self.config.scratch_dir)
            .field("memory_budget_bytes", &self.config.memory_budget_bytes)
            .field("dim", &self.dim)
            .field("input_rows", &self.input_rows)
            .finish_non_exhaustive()
    }
}

impl VectorV5ExternalCollector {
    /// Create an authoritative clustered V5 base builder.
    ///
    /// V5 has no delta wire mode: every completed artifact is an absolute
    /// snapshot of the NodeId-sorted winners streamed through [`Self::push`].
    /// This explicit constructor lets Search-LSM compaction and acceptance
    /// fixtures state that contract without routing a base through the VG6
    /// delta builder. It is otherwise identical to [`Self::new`] and preserves
    /// the same bounded-memory, scratch, metrics, and footer validation
    /// guarantees.
    pub fn new_base(config: VectorV5ExternalBuildConfig) -> Result<Self, Error> {
        Self::new(config)
    }

    pub fn new(config: VectorV5ExternalBuildConfig) -> Result<Self, Error> {
        config.validate()?;
        let spool = new_named_spool(&config.scratch_dir)?;
        Ok(Self {
            config,
            spool,
            dim: None,
            input_rows: 0,
            last_id: None,
            filter_properties: BTreeSet::new(),
            filter_property_bytes: 0,
            max_record_wire_bytes: 0,
            peak_logical_memory_bytes: 0,
            scratch_bytes_written: 0,
        })
    }

    /// Append one authoritative winner.
    ///
    /// `filters` may contain String, Bool, or Null. Null creates no posting;
    /// other value types are rejected so a caller cannot silently believe a
    /// numeric predicate was materialised by the current typed-key format.
    pub fn push(
        &mut self,
        id: [u8; 16],
        vector: &[f32],
        filters: &BTreeMap<String, Value>,
    ) -> Result<(), Error> {
        if let Some(last) = self.last_id {
            if id <= last {
                return Err(Error::invariant(
                    "vector v5 external winner stream must be strictly NodeId ascending",
                ));
            }
        }
        if vector.is_empty() {
            return Err(Error::invariant(
                "vector v5 external rows cannot have dimension zero",
            ));
        }
        if vector.iter().any(|component| !component.is_finite()) {
            return Err(Error::invariant(
                "vector v5 external row contains a non-finite component",
            ));
        }
        match self.dim {
            None => {
                self.dim = Some(vector.len());
                write_spool_header(self.spool.as_file_mut(), vector.len())?;
                self.scratch_bytes_written =
                    self.scratch_bytes_written.saturating_add(SPOOL_HEADER_LEN);
            }
            Some(dim) if dim != vector.len() => {
                return Err(Error::invariant(format!(
                    "vector v5 external row dimension {} != first row dimension {dim}",
                    vector.len()
                )));
            }
            Some(_) => {}
        }

        let (encoded_filters, observed_properties) = encode_filters(filters)?;
        let payload = encode_record_payload(id, vector, &encoded_filters)?;
        let new_property_bytes = observed_properties
            .iter()
            .filter(|property| !self.filter_properties.contains(*property))
            .map(|property| {
                property
                    .len()
                    .saturating_add(std::mem::size_of::<String>() + 48)
            })
            .sum::<usize>();
        let projected_property_bytes = self
            .filter_property_bytes
            .saturating_add(new_property_bytes);
        let wire_bytes = payload
            .len()
            .checked_add(8)
            .ok_or_else(|| Error::invariant("vector v5 spool record size overflows"))?;
        if wire_bytes.saturating_add(projected_property_bytes) > self.config.memory_budget_bytes {
            return Err(Error::invariant(format!(
                "vector v5 spool row + filter schema requires {} bytes, above the configured \
                 memory budget {}",
                wire_bytes.saturating_add(projected_property_bytes),
                self.config.memory_budget_bytes
            )));
        }
        write_record_payload(self.spool.as_file_mut(), &payload)?;
        for property in observed_properties {
            self.filter_properties.insert(property);
        }
        self.filter_property_bytes = projected_property_bytes;
        self.max_record_wire_bytes = self.max_record_wire_bytes.max(wire_bytes);
        self.peak_logical_memory_bytes = self.peak_logical_memory_bytes.max(
            payload
                .capacity()
                .saturating_add(self.filter_property_bytes),
        );
        self.scratch_bytes_written = self.scratch_bytes_written.saturating_add(wire_bytes as u64);
        self.input_rows = self
            .input_rows
            .checked_add(1)
            .ok_or_else(|| Error::invariant("vector v5 external input count overflows"))?;
        self.last_id = Some(id);
        Ok(())
    }

    pub fn input_rows(&self) -> u64 {
        self.input_rows
    }

    /// Convert the spool into one range-readable V5 artifact.
    pub fn finish(
        mut self,
        desc: &VectorIndexDescriptor,
        options: VectorV5BuildOptions,
    ) -> Result<Option<VectorV5ExternalArtifact>, Error> {
        validate_build_options(desc, options)?;
        if options.branch_factor > MAX_EXTERNAL_BRANCH_FACTOR {
            return Err(Error::invariant(format!(
                "vector v5 external branch_factor {} exceeds the file-descriptor-safe maximum \
                 {MAX_EXTERNAL_BRANCH_FACTOR}",
                options.branch_factor
            )));
        }
        let Some(dim) = self.dim else {
            return Ok(None);
        };
        if dim != desc.dim as usize {
            return Err(Error::invariant(format!(
                "vector v5 external spool dimension {dim} != descriptor dimension {}",
                desc.dim
            )));
        }
        self.spool.as_file_mut().flush()?;

        let mut metrics = VectorV5ExternalBuildMetrics {
            input_rows: self.input_rows,
            peak_logical_memory_bytes: self.peak_logical_memory_bytes,
            peak_working_memory_bytes: self.peak_logical_memory_bytes,
            scratch_bytes_written: self.scratch_bytes_written,
            ..VectorV5ExternalBuildMetrics::default()
        };
        let prepared = prepare_root_partition(
            &mut self.spool,
            &self.config.scratch_dir,
            dim,
            desc.metric,
            self.config.memory_budget_bytes,
            &mut metrics,
        )?;
        let Some((root_partition, min_node_id, max_node_id, max_record_wire_bytes)) = prepared
        else {
            return Ok(None);
        };
        metrics.indexed_rows = root_partition.rows;

        let effective_target = effective_target_rows(
            options.target_rows_per_page,
            dim,
            max_record_wire_bytes.max(self.max_record_wire_bytes),
            self.config.memory_budget_bytes,
        )?;
        metrics.effective_target_rows_per_page = effective_target;

        let mut artifact = tempfile::tempfile_in(&self.config.scratch_dir)?;
        artifact.write_all(MAGIC_V5)?;
        let mut context = ExternalBuildContext {
            config: &self.config,
            desc,
            options,
            effective_target,
            output: &mut artifact,
            nodes: Vec::new(),
            pages: Vec::new(),
            metadata_bytes: 0,
            metrics: &mut metrics,
        };
        let root = build_partition(root_partition, 0, &mut context)?;
        let nodes = std::mem::take(&mut context.nodes);
        let pages = std::mem::take(&mut context.pages);
        drop(context);
        let filter_metadata_bytes = self
            .filter_properties
            .iter()
            .map(|property| {
                property
                    .capacity()
                    .saturating_add(std::mem::size_of::<String>())
            })
            .sum::<usize>();
        metrics.resident_metadata_bytes = metrics
            .resident_metadata_bytes
            .saturating_add(filter_metadata_bytes);
        metrics.peak_logical_memory_bytes = metrics.peak_logical_memory_bytes.max(
            metrics
                .resident_metadata_bytes
                .saturating_add(metrics.peak_working_memory_bytes),
        );
        if metrics.peak_logical_memory_bytes > self.config.memory_budget_bytes {
            return Err(Error::invariant(format!(
                "vector v5 footer + working set requires {} logical bytes, above budget {}",
                metrics.peak_logical_memory_bytes, self.config.memory_budget_bytes
            )));
        }
        let footer = Footer {
            format_version: FORMAT_VERSION,
            dim: desc.dim,
            metric: metric_name(desc.metric).to_string(),
            point_count: metrics.indexed_rows,
            min_node_id,
            max_node_id,
            root,
            nodes,
            pages,
            filter_properties: self.filter_properties.into_iter().collect(),
            target_rows_per_page: u32::try_from(effective_target)
                .map_err(|_| Error::invariant("vector v5 effective page rows exceed u32"))?,
            branch_factor: u16::try_from(options.branch_factor)
                .map_err(|_| Error::invariant("vector v5 branch factor exceeds u16"))?,
        };
        write_footer(&mut artifact, &footer)?;
        let len = artifact.stream_position()?;
        artifact.seek(SeekFrom::Start(0))?;
        metrics.page_count = footer.pages.len() as u64;
        metrics.artifact_bytes_written = len;

        let stats = VectorGraphBuildStats {
            dim: desc.dim,
            metric: metric_name(desc.metric).to_string(),
            point_count: metrics.indexed_rows,
            min_node_id,
            max_node_id,
            r: desc.r,
            l_build: desc.l_build,
            alpha: desc.alpha,
            entry_medoid: root,
        };
        Ok(Some(VectorV5ExternalArtifact {
            file: artifact,
            len,
            stats,
            metrics,
        }))
    }
}

fn new_named_spool(directory: &Path) -> Result<NamedTempFile, Error> {
    tempfile::Builder::new()
        .prefix("namivg05-spool-")
        .tempfile_in(directory)
        .map_err(Error::from)
}

fn encode_filters(
    filters: &BTreeMap<String, Value>,
) -> Result<(Vec<(String, String)>, Vec<String>), Error> {
    if filters.len() > MAX_FILTERS_PER_ROW {
        return Err(Error::invariant("vector v5 row has too many filters"));
    }
    let mut encoded = Vec::with_capacity(filters.len());
    let mut properties = Vec::with_capacity(filters.len());
    for (property, value) in filters {
        if property.len() > MAX_FILTER_COMPONENT_BYTES {
            return Err(Error::invariant("vector v5 filter property is too large"));
        }
        match value {
            Value::Null => {
                properties.push(property.clone());
            }
            Value::Bool(_) | Value::Str(_) => {
                let key = encode_vector_filter_value(value)
                    .ok_or_else(|| Error::invariant("vector v5 filter encoding failed"))?
                    .into_owned();
                if key.len() > MAX_FILTER_COMPONENT_BYTES {
                    return Err(Error::invariant("vector v5 filter value is too large"));
                }
                properties.push(property.clone());
                encoded.push((property.clone(), key));
            }
            _ => {
                return Err(Error::invariant(format!(
                    "vector v5 native filter `{property}` must be String, Bool, or Null"
                )));
            }
        }
    }
    Ok((encoded, properties))
}

fn write_spool_header(file: &mut File, dim: usize) -> Result<(), Error> {
    file.write_all(SPOOL_MAGIC)?;
    file.write_all(
        &u32::try_from(dim)
            .map_err(|_| Error::invariant("vector v5 spool dimension exceeds u32"))?
            .to_le_bytes(),
    )?;
    Ok(())
}

fn validate_spool_header(file: &mut File, expected_dim: usize) -> Result<(), Error> {
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; SPOOL_HEADER_LEN as usize];
    file.read_exact(&mut header).map_err(|error| {
        Error::invariant(format!("vector v5 spool header is truncated: {error}"))
    })?;
    if &header[..8] != SPOOL_MAGIC {
        return Err(Error::invariant("vector v5 spool magic mismatch"));
    }
    let dim = u32::from_le_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| Error::invariant("vector v5 spool dimension bytes are invalid"))?,
    ) as usize;
    if dim != expected_dim {
        return Err(Error::invariant(format!(
            "vector v5 spool dimension {dim} != expected {expected_dim}"
        )));
    }
    Ok(())
}

fn encode_record_payload(
    id: [u8; 16],
    vector: &[f32],
    filters: &[(String, String)],
) -> Result<Vec<u8>, Error> {
    let vector_bytes = vector
        .len()
        .checked_mul(4)
        .ok_or_else(|| Error::invariant("vector v5 spool vector size overflows"))?;
    let filter_bytes = filters
        .iter()
        .try_fold(0usize, |total, (property, value)| {
            total
                .checked_add(8)
                .and_then(|sum| sum.checked_add(property.len()))
                .and_then(|sum| sum.checked_add(value.len()))
                .ok_or_else(|| Error::invariant("vector v5 spool filter size overflows"))
        })?;
    let capacity = 16usize
        .checked_add(vector_bytes)
        .and_then(|sum| sum.checked_add(4))
        .and_then(|sum| sum.checked_add(filter_bytes))
        .ok_or_else(|| Error::invariant("vector v5 spool payload size overflows"))?;
    let mut payload = Vec::with_capacity(capacity);
    payload.extend_from_slice(&id);
    for component in vector {
        payload.extend_from_slice(&component.to_le_bytes());
    }
    payload.extend_from_slice(
        &u32::try_from(filters.len())
            .map_err(|_| Error::invariant("vector v5 spool filter count exceeds u32"))?
            .to_le_bytes(),
    );
    for (property, value) in filters {
        payload.extend_from_slice(
            &u32::try_from(property.len())
                .map_err(|_| Error::invariant("vector v5 filter property exceeds u32"))?
                .to_le_bytes(),
        );
        payload.extend_from_slice(property.as_bytes());
        payload.extend_from_slice(
            &u32::try_from(value.len())
                .map_err(|_| Error::invariant("vector v5 filter value exceeds u32"))?
                .to_le_bytes(),
        );
        payload.extend_from_slice(value.as_bytes());
    }
    Ok(payload)
}

fn write_record_payload(file: &mut File, payload: &[u8]) -> Result<(), Error> {
    let len = u32::try_from(payload.len())
        .map_err(|_| Error::invariant("vector v5 spool row exceeds u32 bytes"))?;
    file.write_all(&len.to_le_bytes())?;
    file.write_all(payload)?;
    file.write_all(&crc32fast::hash(payload).to_le_bytes())?;
    Ok(())
}

#[derive(Debug)]
struct SpoolRecord {
    id: [u8; 16],
    vector: Vec<f32>,
    filters: Vec<(String, String)>,
    payload: Vec<u8>,
}

impl SpoolRecord {
    fn logical_bytes(&self) -> usize {
        self.payload
            .capacity()
            .saturating_add(self.vector.capacity().saturating_mul(4))
            .saturating_add(
                self.filters
                    .iter()
                    .map(|(property, value)| {
                        property
                            .capacity()
                            .saturating_add(value.capacity())
                            .saturating_add(std::mem::size_of::<(String, String)>())
                    })
                    .sum::<usize>(),
            )
    }

    fn wire_bytes(&self) -> usize {
        self.payload.len().saturating_add(8)
    }
}

fn read_record(
    file: &mut File,
    dim: usize,
    memory_budget: usize,
) -> Result<Option<SpoolRecord>, Error> {
    let mut len_bytes = [0u8; 4];
    match file.read(&mut len_bytes) {
        Ok(0) => return Ok(None),
        Ok(4) => {}
        Ok(read) => {
            return Err(Error::invariant(format!(
                "vector v5 spool row length is truncated ({read}/4 bytes)"
            )));
        }
        Err(error) => return Err(Error::from(error)),
    }
    let payload_len = u32::from_le_bytes(len_bytes) as usize;
    let minimum = 16usize
        .checked_add(
            dim.checked_mul(4)
                .ok_or_else(|| Error::invariant("vector v5 spool dimension overflows"))?,
        )
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| Error::invariant("vector v5 spool minimum row size overflows"))?;
    if payload_len < minimum || payload_len.saturating_add(8) > memory_budget {
        return Err(Error::invariant(format!(
            "vector v5 spool row length {payload_len} is invalid for dim {dim}"
        )));
    }
    let mut payload = vec![0u8; payload_len];
    file.read_exact(&mut payload).map_err(|error| {
        Error::invariant(format!("vector v5 spool row payload is truncated: {error}"))
    })?;
    let mut crc_bytes = [0u8; 4];
    file.read_exact(&mut crc_bytes)
        .map_err(|error| Error::invariant(format!("vector v5 spool CRC is truncated: {error}")))?;
    let expected_crc = u32::from_le_bytes(crc_bytes);
    if crc32fast::hash(&payload) != expected_crc {
        return Err(Error::invariant("vector v5 spool row checksum mismatch"));
    }

    let mut cursor = 0usize;
    let id: [u8; 16] = take_bytes(&payload, &mut cursor, 16, "node id")?
        .try_into()
        .map_err(|_| Error::invariant("vector v5 spool node id is invalid"))?;
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        let bytes: [u8; 4] = take_bytes(&payload, &mut cursor, 4, "vector component")?
            .try_into()
            .map_err(|_| Error::invariant("vector v5 spool component is invalid"))?;
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            return Err(Error::invariant(
                "vector v5 spool contains a non-finite component",
            ));
        }
        vector.push(value);
    }
    let filter_count = read_u32(&payload, &mut cursor, "filter count")? as usize;
    if filter_count > MAX_FILTERS_PER_ROW {
        return Err(Error::invariant("vector v5 spool filter count is invalid"));
    }
    let mut filters = Vec::with_capacity(filter_count);
    for _ in 0..filter_count {
        let property_len = read_u32(&payload, &mut cursor, "filter property length")? as usize;
        if property_len > MAX_FILTER_COMPONENT_BYTES {
            return Err(Error::invariant(
                "vector v5 spool filter property is oversized",
            ));
        }
        let property = String::from_utf8(
            take_bytes(&payload, &mut cursor, property_len, "filter property")?.to_vec(),
        )
        .map_err(|error| {
            Error::invariant(format!(
                "vector v5 spool filter property is not UTF-8: {error}"
            ))
        })?;
        let value_len = read_u32(&payload, &mut cursor, "filter value length")? as usize;
        if value_len > MAX_FILTER_COMPONENT_BYTES {
            return Err(Error::invariant(
                "vector v5 spool filter value is oversized",
            ));
        }
        let value = String::from_utf8(
            take_bytes(&payload, &mut cursor, value_len, "filter value")?.to_vec(),
        )
        .map_err(|error| {
            Error::invariant(format!(
                "vector v5 spool filter value is not UTF-8: {error}"
            ))
        })?;
        filters.push((property, value));
    }
    if cursor != payload.len() {
        return Err(Error::invariant("vector v5 spool row has trailing bytes"));
    }
    Ok(Some(SpoolRecord {
        id,
        vector,
        filters,
        payload,
    }))
}

fn take_bytes<'a>(
    payload: &'a [u8],
    cursor: &mut usize,
    len: usize,
    label: &str,
) -> Result<&'a [u8], Error> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| Error::invariant(format!("vector v5 spool {label} offset overflows")))?;
    let bytes = payload
        .get(*cursor..end)
        .ok_or_else(|| Error::invariant(format!("vector v5 spool {label} is truncated")))?;
    *cursor = end;
    Ok(bytes)
}

fn read_u32(payload: &[u8], cursor: &mut usize, label: &str) -> Result<u32, Error> {
    Ok(u32::from_le_bytes(
        take_bytes(payload, cursor, 4, label)?
            .try_into()
            .map_err(|_| Error::invariant(format!("vector v5 spool {label} is invalid")))?,
    ))
}

#[derive(Debug)]
struct Partition {
    spool: NamedTempFile,
    rows: u64,
    bytes: u64,
}

impl Partition {
    fn create(directory: &Path, dim: usize) -> Result<Self, Error> {
        let mut spool = new_named_spool(directory)?;
        write_spool_header(spool.as_file_mut(), dim)?;
        Ok(Self {
            spool,
            rows: 0,
            bytes: SPOOL_HEADER_LEN,
        })
    }

    fn rewind(&mut self, dim: usize) -> Result<(), Error> {
        validate_spool_header(self.spool.as_file_mut(), dim)
    }

    fn append(&mut self, record: &SpoolRecord) -> Result<(), Error> {
        write_record_payload(self.spool.as_file_mut(), &record.payload)?;
        self.rows = self
            .rows
            .checked_add(1)
            .ok_or_else(|| Error::invariant("vector v5 partition row count overflows"))?;
        self.bytes = self
            .bytes
            .checked_add(record.wire_bytes() as u64)
            .ok_or_else(|| Error::invariant("vector v5 partition byte count overflows"))?;
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.spool.as_file_mut().flush()?;
        Ok(())
    }
}

fn prepare_root_partition(
    source: &mut NamedTempFile,
    scratch_dir: &Path,
    dim: usize,
    metric: VectorMetric,
    memory_budget: usize,
    metrics: &mut VectorV5ExternalBuildMetrics,
) -> Result<Option<(Partition, [u8; 16], [u8; 16], usize)>, Error> {
    validate_spool_header(source.as_file_mut(), dim)?;
    let mut root = Partition::create(scratch_dir, dim)?;
    metrics.scratch_bytes_written = metrics
        .scratch_bytes_written
        .saturating_add(SPOOL_HEADER_LEN);
    let mut min_id = None;
    let mut max_id = None;
    let mut max_wire = 0usize;
    while let Some(record) = read_record(source.as_file_mut(), dim, memory_budget)? {
        metrics.observe_working(record.logical_bytes());
        if metric == VectorMetric::Cosine && record.vector.iter().all(|value| *value == 0.0) {
            continue;
        }
        min_id.get_or_insert(record.id);
        max_id = Some(record.id);
        max_wire = max_wire.max(record.wire_bytes());
        root.append(&record)?;
        metrics.scratch_bytes_written = metrics
            .scratch_bytes_written
            .saturating_add(record.wire_bytes() as u64);
    }
    root.flush()?;
    match (min_id, max_id) {
        (Some(min), Some(max)) => Ok(Some((root, min, max, max_wire))),
        _ => Ok(None),
    }
}

fn effective_target_rows(
    requested: usize,
    dim: usize,
    max_record_wire_bytes: usize,
    budget: usize,
) -> Result<usize, Error> {
    // During leaf emission we retain one decoded record plus IDs, int8 codes,
    // scales and conservative filter-map overhead for every row. Exact f32 is
    // streamed in a second pass and is never page-resident.
    let fixed = dim
        .checked_mul(16)
        .and_then(|value| value.checked_add(max_record_wire_bytes.saturating_mul(2)))
        .ok_or_else(|| Error::invariant("vector v5 external fixed memory estimate overflows"))?;
    if fixed >= budget {
        return Err(Error::invariant(format!(
            "vector v5 external dimension/row requires at least {} bytes, above budget {budget}",
            fixed.saturating_add(1)
        )));
    }
    let per_row = dim
        .checked_add(16 + 4 + 192)
        .and_then(|value| value.checked_add(max_record_wire_bytes))
        .ok_or_else(|| Error::invariant("vector v5 external per-row estimate overflows"))?;
    let rows = ((budget - fixed) / per_row).max(1).min(requested);
    Ok(rows)
}

struct ExternalBuildContext<'a> {
    config: &'a VectorV5ExternalBuildConfig,
    desc: &'a VectorIndexDescriptor,
    options: VectorV5BuildOptions,
    effective_target: usize,
    output: &'a mut File,
    nodes: Vec<CentroidNode>,
    pages: Vec<PageRef>,
    metadata_bytes: usize,
    metrics: &'a mut VectorV5ExternalBuildMetrics,
}

impl ExternalBuildContext<'_> {
    fn account_metadata(&mut self) -> Result<(), Error> {
        self.metrics.resident_metadata_bytes = self.metadata_bytes;
        self.metrics.peak_logical_memory_bytes = self.metrics.peak_logical_memory_bytes.max(
            self.metrics
                .resident_metadata_bytes
                .saturating_add(self.metrics.peak_working_memory_bytes),
        );
        if self.metrics.peak_logical_memory_bytes > self.config.memory_budget_bytes {
            return Err(Error::invariant(format!(
                "vector v5 external metadata + workspace requires {} bytes, above budget {}",
                self.metrics.peak_logical_memory_bytes, self.config.memory_budget_bytes
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PartitionStats {
    centroid: Vec<f32>,
    split_axis: usize,
    max_variance: f64,
}

fn scan_partition(
    partition: &mut Partition,
    dim: usize,
    metric: VectorMetric,
    budget: usize,
    metrics: &mut VectorV5ExternalBuildMetrics,
) -> Result<PartitionStats, Error> {
    partition.rewind(dim)?;
    let mut sums = vec![0.0f64; dim];
    let mut sums_sq = vec![0.0f64; dim];
    let accum_bytes = dim
        .checked_mul(16)
        .ok_or_else(|| Error::invariant("vector v5 variance accumulator size overflows"))?;
    if accum_bytes >= budget {
        return Err(Error::invariant(format!(
            "vector v5 variance accumulators require {accum_bytes} bytes, above budget {budget}"
        )));
    }
    let mut seen = 0u64;
    while let Some(record) = read_record(partition.spool.as_file_mut(), dim, budget)? {
        for (axis, value) in record.vector.iter().enumerate() {
            let value = *value as f64;
            sums[axis] += value;
            sums_sq[axis] += value * value;
        }
        metrics.observe_working(accum_bytes.saturating_add(record.logical_bytes()));
        seen = seen
            .checked_add(1)
            .ok_or_else(|| Error::invariant("vector v5 scan row count overflows"))?;
    }
    if seen != partition.rows || seen == 0 {
        return Err(Error::invariant(format!(
            "vector v5 partition row count mismatch: scanned {seen}, expected {}",
            partition.rows
        )));
    }
    let n = seen as f64;
    let mut split_axis = 0usize;
    let mut max_variance = f64::NEG_INFINITY;
    for axis in 0..dim {
        let variance = (sums_sq[axis] / n - (sums[axis] / n).powi(2)).max(0.0);
        if variance > max_variance {
            max_variance = variance;
            split_axis = axis;
        }
    }
    let mut centroid: Vec<f32> = sums.into_iter().map(|sum| (sum / n) as f32).collect();
    if metric == VectorMetric::Cosine {
        let norm = centroid
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for value in &mut centroid {
                *value /= norm;
            }
        }
    }
    Ok(PartitionStats {
        centroid,
        split_axis,
        max_variance,
    })
}

fn build_partition(
    mut partition: Partition,
    depth: usize,
    context: &mut ExternalBuildContext<'_>,
) -> Result<u32, Error> {
    if depth > MAX_PARTITION_DEPTH {
        return Err(Error::invariant(
            "vector v5 external partition depth exceeded safety limit",
        ));
    }
    context.metrics.max_partition_depth = context.metrics.max_partition_depth.max(depth);
    context.metrics.partition_count = context
        .metrics
        .partition_count
        .checked_add(1)
        .ok_or_else(|| Error::invariant("vector v5 partition counter overflows"))?;
    let dim = context.desc.dim as usize;
    let stats = scan_partition(
        &mut partition,
        dim,
        context.desc.metric,
        context.config.memory_budget_bytes,
        context.metrics,
    )?;
    let (codes, scale) = quantize_i8(&stats.centroid);
    let node = u32::try_from(context.nodes.len())
        .map_err(|_| Error::invariant("vector v5 external tree exceeds u32 nodes"))?;
    let node_capacity_before = context.nodes.capacity();
    let code_bytes = codes.capacity();
    context.nodes.push(CentroidNode {
        codes,
        scale,
        children: Vec::new(),
        page: None,
    });
    context.metadata_bytes = context
        .metadata_bytes
        .saturating_add(code_bytes)
        .saturating_add(
            context
                .nodes
                .capacity()
                .saturating_sub(node_capacity_before)
                .saturating_mul(std::mem::size_of::<CentroidNode>()),
        );
    context.account_metadata()?;

    if partition.rows as usize <= context.effective_target {
        let page = write_leaf_page(&mut partition, context)?;
        context.nodes[node as usize].page = Some(page);
        return Ok(node);
    }

    let requested_groups = context
        .options
        .branch_factor
        .min((partition.rows as usize).div_ceil(context.effective_target))
        .max(2);
    let children = partition_multiway(&mut partition, &stats, requested_groups, depth, context)?;
    if children.len() < 2 {
        return Err(Error::invariant(
            "vector v5 external partition failed to make progress",
        ));
    }
    let mut child_nodes = Vec::with_capacity(children.len());
    for child in children {
        child_nodes.push(build_partition(child, depth + 1, context)?);
    }
    context.metadata_bytes = context
        .metadata_bytes
        .saturating_add(child_nodes.capacity().saturating_mul(4));
    context.nodes[node as usize].children = child_nodes;
    context.account_metadata()?;
    Ok(node)
}

#[derive(Debug, Clone)]
struct SamplePoint {
    sample_hash: u64,
    value: f32,
    tie_hash: u64,
    id: [u8; 16],
}

impl PartialEq for SamplePoint {
    fn eq(&self, other: &Self) -> bool {
        self.sample_hash == other.sample_hash && self.id == other.id
    }
}

impl Eq for SamplePoint {}

impl PartialOrd for SamplePoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SamplePoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sample_hash
            .cmp(&other.sample_hash)
            .then_with(|| self.id.cmp(&other.id))
    }
}

fn partition_multiway(
    partition: &mut Partition,
    stats: &PartitionStats,
    requested_groups: usize,
    depth: usize,
    context: &mut ExternalBuildContext<'_>,
) -> Result<Vec<Partition>, Error> {
    let dim = context.desc.dim as usize;
    // For an all-equal partition, quantiles carry no information. The stream is
    // NodeId ordered, so exact rank ranges are a deterministic NodeId tie-break
    // and guarantee balanced progress.
    if stats.max_variance <= f64::EPSILON {
        return partition_by_rank(partition, requested_groups, dim, context);
    }

    let sample_capacity_by_budget = context
        .config
        .memory_budget_bytes
        .saturating_div(std::mem::size_of::<SamplePoint>().max(1) * 4)
        .max(2);
    let sample_limit = context
        .config
        .quantile_sample_rows
        .min(sample_capacity_by_budget)
        .min(partition.rows as usize)
        .max(2);
    let mut heap = BinaryHeap::with_capacity(sample_limit);
    partition.rewind(dim)?;
    while let Some(record) = read_record(
        partition.spool.as_file_mut(),
        dim,
        context.config.memory_budget_bytes,
    )? {
        let point = SamplePoint {
            sample_hash: stable_hash(&record.id, stats.split_axis, depth, 0x51),
            value: record.vector[stats.split_axis],
            tie_hash: stable_hash(&record.id, stats.split_axis, depth, 0xA7),
            id: record.id,
        };
        if heap.len() < sample_limit {
            heap.push(point);
        } else if heap.peek().is_some_and(|largest| point < *largest) {
            heap.pop();
            heap.push(point);
        }
        context.metrics.observe_working(
            heap.capacity()
                .saturating_mul(std::mem::size_of::<SamplePoint>())
                .saturating_add(record.logical_bytes()),
        );
    }
    context.metrics.peak_sample_rows = context.metrics.peak_sample_rows.max(heap.len());
    let mut sample = heap.into_vec();
    sample.sort_by(split_key_order);
    let groups = requested_groups.min(sample.len()).max(2);
    let mut cuts = Vec::with_capacity(groups - 1);
    for group in 1..groups {
        let index = group
            .checked_mul(sample.len())
            .ok_or_else(|| Error::invariant("vector v5 sample quantile index overflows"))?
            / groups;
        cuts.push(sample[index.min(sample.len() - 1)].clone());
    }

    let mut children =
        create_partitions(groups, &context.config.scratch_dir, dim, context.metrics)?;
    partition.rewind(dim)?;
    while let Some(record) = read_record(
        partition.spool.as_file_mut(),
        dim,
        context.config.memory_budget_bytes,
    )? {
        let key = SamplePoint {
            sample_hash: 0,
            value: record.vector[stats.split_axis],
            tie_hash: stable_hash(&record.id, stats.split_axis, depth, 0xA7),
            id: record.id,
        };
        let child = cuts.partition_point(|cut| split_key_order(cut, &key) == Ordering::Less);
        children[child].append(&record)?;
        context.metrics.scratch_bytes_written = context
            .metrics
            .scratch_bytes_written
            .saturating_add(record.wire_bytes() as u64);
    }
    for child in &mut children {
        child.flush()?;
    }
    children.retain(|child| child.rows > 0);
    if children.len() < 2 || children.iter().any(|child| child.rows == partition.rows) {
        // A bounded sample can miss an extreme skew. Rank partitioning is the
        // deterministic progress fallback and remains sequential-I/O.
        return partition_by_rank(partition, requested_groups, dim, context);
    }
    Ok(children)
}

fn split_key_order(left: &SamplePoint, right: &SamplePoint) -> Ordering {
    left.value
        .total_cmp(&right.value)
        .then_with(|| left.tie_hash.cmp(&right.tie_hash))
        .then_with(|| left.id.cmp(&right.id))
}

fn stable_hash(id: &[u8; 16], axis: usize, depth: usize, domain: u8) -> u64 {
    let mut bytes = [0u8; 16 + 8 + 8 + 1];
    bytes[..16].copy_from_slice(id);
    bytes[16..24].copy_from_slice(&(axis as u64).to_le_bytes());
    bytes[24..32].copy_from_slice(&(depth as u64).to_le_bytes());
    bytes[32] = domain;
    xxh3_64(&bytes)
}

fn create_partitions(
    count: usize,
    scratch_dir: &Path,
    dim: usize,
    metrics: &mut VectorV5ExternalBuildMetrics,
) -> Result<Vec<Partition>, Error> {
    let mut partitions = Vec::with_capacity(count);
    for _ in 0..count {
        partitions.push(Partition::create(scratch_dir, dim)?);
        metrics.scratch_bytes_written = metrics
            .scratch_bytes_written
            .saturating_add(SPOOL_HEADER_LEN);
    }
    Ok(partitions)
}

fn partition_by_rank(
    partition: &mut Partition,
    groups: usize,
    dim: usize,
    context: &mut ExternalBuildContext<'_>,
) -> Result<Vec<Partition>, Error> {
    let mut children =
        create_partitions(groups, &context.config.scratch_dir, dim, context.metrics)?;
    partition.rewind(dim)?;
    let mut ordinal = 0u64;
    while let Some(record) = read_record(
        partition.spool.as_file_mut(),
        dim,
        context.config.memory_budget_bytes,
    )? {
        let child = ((ordinal as u128 * groups as u128) / partition.rows as u128) as usize;
        children[child.min(groups - 1)].append(&record)?;
        context.metrics.scratch_bytes_written = context
            .metrics
            .scratch_bytes_written
            .saturating_add(record.wire_bytes() as u64);
        ordinal += 1;
    }
    for child in &mut children {
        child.flush()?;
    }
    children.retain(|child| child.rows > 0);
    Ok(children)
}

fn write_leaf_page(
    partition: &mut Partition,
    context: &mut ExternalBuildContext<'_>,
) -> Result<u32, Error> {
    let dim = context.desc.dim as usize;
    let rows = usize::try_from(partition.rows)
        .map_err(|_| Error::invariant("vector v5 leaf row count does not fit usize"))?;
    let code_capacity = rows
        .checked_mul(dim)
        .ok_or_else(|| Error::invariant("vector v5 leaf code capacity overflows"))?;
    let mut nav = NavPage {
        ids: Vec::with_capacity(rows),
        codes: Vec::with_capacity(code_capacity),
        scales: Vec::with_capacity(rows),
        filters: BTreeMap::new(),
    };
    let words = rows.div_ceil(64);
    partition.rewind(dim)?;
    let mut ordinal = 0usize;
    while let Some(record) = read_record(
        partition.spool.as_file_mut(),
        dim,
        context.config.memory_budget_bytes,
    )? {
        nav.ids.push(record.id);
        let (codes, scale) = quantize_i8(&record.vector);
        if codes.len() != dim || !scale.is_finite() || scale < 0.0 {
            return Err(Error::invariant(
                "vector v5 external quantizer returned an invalid row",
            ));
        }
        nav.codes.extend(codes);
        nav.scales.push(scale);
        for (property, value) in &record.filters {
            let bitmap = nav
                .filters
                .entry(property.clone())
                .or_default()
                .entry(value.clone())
                .or_insert_with(|| vec![0u64; words]);
            bitmap[ordinal / 64] |= 1u64 << (ordinal % 64);
        }
        ordinal += 1;
        let nav_estimate = nav
            .ids
            .capacity()
            .saturating_mul(16)
            .saturating_add(nav.codes.capacity())
            .saturating_add(nav.scales.capacity().saturating_mul(4))
            .saturating_add(estimate_filter_map_bytes(&nav.filters))
            .saturating_add(record.logical_bytes());
        context.metrics.observe_working(nav_estimate);
        if nav_estimate > context.config.memory_budget_bytes {
            return Err(Error::invariant(format!(
                "vector v5 leaf requires {nav_estimate} logical bytes, above budget {}",
                context.config.memory_budget_bytes
            )));
        }
    }
    if ordinal != rows {
        return Err(Error::invariant(
            "vector v5 leaf row count changed while emitting nav page",
        ));
    }
    let nav_ref = write_nav_block(context.output, &nav, context.options.compression_level)?;
    // Drop page-local navigation allocations before the full-precision pass.
    drop(nav);
    let exact_ref = write_exact_block(
        context.output,
        partition,
        dim,
        context.options.compression_level,
        context.config.memory_budget_bytes,
        context.metrics,
    )?;
    let page = u32::try_from(context.pages.len())
        .map_err(|_| Error::invariant("vector v5 external page count exceeds u32"))?;
    let page_capacity_before = context.pages.capacity();
    context.pages.push(PageRef {
        row_count: u32::try_from(rows)
            .map_err(|_| Error::invariant("vector v5 leaf rows exceed u32"))?,
        nav: nav_ref,
        exact: exact_ref,
    });
    context.metadata_bytes = context.metadata_bytes.saturating_add(
        context
            .pages
            .capacity()
            .saturating_sub(page_capacity_before)
            .saturating_mul(std::mem::size_of::<PageRef>()),
    );
    context.account_metadata()?;
    Ok(page)
}

fn estimate_filter_map_bytes(filters: &BTreeMap<String, BTreeMap<String, Vec<u64>>>) -> usize {
    filters
        .iter()
        .map(|(property, values)| {
            property
                .capacity()
                .saturating_add(std::mem::size_of::<String>() + 48)
                .saturating_add(
                    values
                        .iter()
                        .map(|(value, bitmap)| {
                            value
                                .capacity()
                                .saturating_add(bitmap.capacity().saturating_mul(8))
                                .saturating_add(std::mem::size_of::<String>() + 48)
                        })
                        .sum::<usize>(),
                )
        })
        .sum()
}

struct CrcCountingWriter<'a> {
    inner: &'a mut File,
    hasher: Hasher,
    written: u64,
}

impl<'a> CrcCountingWriter<'a> {
    fn new(inner: &'a mut File) -> Self {
        Self {
            inner,
            hasher: Hasher::new(),
            written: 0,
        }
    }

    fn finish(self) -> (u64, u32) {
        (self.written, self.hasher.finalize())
    }
}

impl Write for CrcCountingWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hasher.update(&bytes[..written]);
        self.written = self.written.saturating_add(written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn write_nav_block(
    output: &mut File,
    nav: &NavPage,
    compression_level: i32,
) -> Result<BlockRef, Error> {
    let raw_len = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialized_size(nav)
        .map_err(|error| Error::invariant(format!("vector v5 nav size failed: {error}")))?;
    validate_block_lengths(raw_len, "navigation page")?;
    let offset = output.stream_position()?;
    let writer = CrcCountingWriter::new(output);
    let mut encoder = zstd::stream::write::Encoder::new(writer, compression_level)
        .map_err(|error| Error::invariant(format!("vector v5 nav zstd init failed: {error}")))?;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize_into(&mut encoder, nav)
        .map_err(|error| Error::invariant(format!("vector v5 nav encode failed: {error}")))?;
    let writer = encoder
        .finish()
        .map_err(|error| Error::invariant(format!("vector v5 nav zstd finish failed: {error}")))?;
    let (wire_len, crc32) = writer.finish();
    make_block_ref(offset, wire_len, raw_len, crc32, "navigation page")
}

fn write_exact_block(
    output: &mut File,
    partition: &mut Partition,
    dim: usize,
    compression_level: i32,
    memory_budget: usize,
    metrics: &mut VectorV5ExternalBuildMetrics,
) -> Result<BlockRef, Error> {
    let raw_len = partition
        .rows
        .checked_mul(dim as u64)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| Error::invariant("vector v5 exact block raw length overflows"))?;
    validate_block_lengths(raw_len, "exact page")?;
    let offset = output.stream_position()?;
    let writer = CrcCountingWriter::new(output);
    let mut encoder = zstd::stream::write::Encoder::new(writer, compression_level)
        .map_err(|error| Error::invariant(format!("vector v5 exact zstd init failed: {error}")))?;
    partition.rewind(dim)?;
    let mut rows = 0u64;
    while let Some(record) = read_record(partition.spool.as_file_mut(), dim, memory_budget)? {
        for component in &record.vector {
            encoder.write_all(&component.to_le_bytes())?;
        }
        metrics.observe_working(record.logical_bytes());
        rows += 1;
    }
    if rows != partition.rows {
        return Err(Error::invariant(
            "vector v5 leaf row count changed while emitting exact page",
        ));
    }
    let writer = encoder.finish().map_err(|error| {
        Error::invariant(format!("vector v5 exact zstd finish failed: {error}"))
    })?;
    let (wire_len, crc32) = writer.finish();
    make_block_ref(offset, wire_len, raw_len, crc32, "exact page")
}

fn validate_block_lengths(raw_len: u64, label: &str) -> Result<(), Error> {
    if raw_len == 0 || raw_len > MAX_RAW_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 external {label} raw length {raw_len} is invalid"
        )));
    }
    Ok(())
}

fn make_block_ref(
    offset: u64,
    wire_len: u64,
    raw_len: u64,
    crc32: u32,
    label: &str,
) -> Result<BlockRef, Error> {
    if wire_len == 0 || wire_len > MAX_COMPRESSED_BLOCK_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 external {label} compressed length {wire_len} is invalid"
        )));
    }
    Ok(BlockRef {
        offset,
        len: u32::try_from(wire_len)
            .map_err(|_| Error::invariant(format!("vector v5 {label} wire length exceeds u32")))?,
        raw_len: u32::try_from(raw_len)
            .map_err(|_| Error::invariant(format!("vector v5 {label} raw length exceeds u32")))?,
        crc32,
    })
}

fn write_footer(output: &mut File, footer: &Footer) -> Result<(), Error> {
    let footer_len = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialized_size(footer)
        .map_err(|error| Error::invariant(format!("vector v5 footer size failed: {error}")))?;
    if footer_len == 0 || footer_len > MAX_FOOTER_BYTES {
        return Err(Error::invariant(format!(
            "vector v5 external footer length {footer_len} is invalid"
        )));
    }
    let writer = CrcCountingWriter::new(output);
    let mut writer = writer;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize_into(&mut writer, footer)
        .map_err(|error| Error::invariant(format!("vector v5 footer encode failed: {error}")))?;
    let (actual_len, crc32) = writer.finish();
    if actual_len != footer_len {
        return Err(Error::invariant(
            "vector v5 footer serialized length changed unexpectedly",
        ));
    }
    output.write_all(TRAILER_MAGIC)?;
    output.write_all(&footer_len.to_le_bytes())?;
    output.write_all(&crc32.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use async_trait::async_trait;
    use bytes::Bytes;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::manifest::VectorQuantization;
    use crate::sst::vector::v5::{
        build_body_v5, VectorV5RangeSource, VectorV5Reader, VectorV5SearchOptions,
    };

    #[derive(Debug)]
    struct MemorySource(Bytes);

    #[async_trait]
    impl VectorV5RangeSource for MemorySource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error> {
            if range.start > range.end || range.end > self.0.len() as u64 {
                return Err(Error::invariant("test range is outside artifact"));
            }
            Ok(self.0.slice(range.start as usize..range.end as usize))
        }
    }

    fn descriptor(metric: VectorMetric, dim: u32) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: format!("external-{metric:?}"),
            label: "Doc".to_string(),
            property: "embedding".to_string(),
            dim,
            metric,
            r: 32,
            l_build: 64,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        }
    }

    fn node_id(value: u32) -> [u8; 16] {
        let mut id = [0u8; 16];
        // Big-endian keeps the fixture's numeric order equal to the collector's
        // required bytewise NodeId order across 255 -> 256.
        id[..4].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn normalize(vector: &mut [f32]) {
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in vector {
                *value /= norm;
            }
        }
    }

    fn rows(count: u32, dim: usize) -> Vec<([u8; 16], Vec<f32>)> {
        let mut rng = ChaCha8Rng::seed_from_u64(0xE571_0005);
        (0..count)
            .map(|ordinal| {
                let cluster = ordinal % 8;
                let mut vector = vec![0.0; dim];
                vector[cluster as usize % dim] = 1.0;
                for value in &mut vector {
                    *value += rng.gen_range(-0.03..0.03);
                }
                normalize(&mut vector);
                (node_id(ordinal), vector)
            })
            .collect()
    }

    fn config(directory: &Path, memory_budget_bytes: usize) -> VectorV5ExternalBuildConfig {
        VectorV5ExternalBuildConfig {
            scratch_dir: directory.to_path_buf(),
            memory_budget_bytes,
            quantile_sample_rows: 32,
        }
    }

    fn options(target: usize) -> VectorV5BuildOptions {
        VectorV5BuildOptions {
            target_rows_per_page: target,
            branch_factor: 4,
            compression_level: 1,
        }
    }

    fn build_external(
        directory: &Path,
        desc: &VectorIndexDescriptor,
        members: &[([u8; 16], Vec<f32>)],
        target: usize,
        memory: usize,
    ) -> VectorV5ExternalArtifact {
        let mut collector =
            VectorV5ExternalCollector::new(config(directory, memory)).expect("collector");
        for (id, vector) in members {
            collector.push(*id, vector, &BTreeMap::new()).expect("push");
        }
        collector
            .finish(desc, options(target))
            .expect("finish")
            .expect("artifact")
    }

    fn artifact_bytes(artifact: &mut VectorV5ExternalArtifact) -> Bytes {
        artifact.rewind().expect("rewind");
        let mut bytes = Vec::with_capacity(artifact.len as usize);
        artifact
            .file
            .read_to_end(&mut bytes)
            .expect("read artifact");
        assert_eq!(bytes.len() as u64, artifact.len);
        Bytes::from(bytes)
    }

    async fn open_artifact(artifact: &mut VectorV5ExternalArtifact) -> (Bytes, VectorV5Reader) {
        let bytes = artifact_bytes(artifact);
        let source = Arc::new(MemorySource(bytes.clone()));
        let reader = VectorV5Reader::open(source, bytes.len() as u64)
            .await
            .expect("open artifact");
        (bytes, reader)
    }

    #[tokio::test]
    async fn exact_all_page_results_match_in_memory_builder_for_all_metrics() {
        let members = rows(384, 16);
        let scratch = tempfile::tempdir().expect("scratch");
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let desc = descriptor(metric, 16);
            let mut external = build_external(scratch.path(), &desc, &members, 24, 256 * 1024);
            let (_, external_reader) = open_artifact(&mut external).await;
            let (memory_body, _) = build_body_v5(&desc, members.clone(), options(24))
                .expect("memory build")
                .expect("memory body");
            let memory_source = Arc::new(MemorySource(memory_body.clone()));
            let memory_reader = VectorV5Reader::open(memory_source, memory_body.len() as u64)
                .await
                .expect("memory open");
            let query = members[137].1.clone();
            let external_hits = external_reader
                .search(
                    &query,
                    20,
                    VectorV5SearchOptions {
                        nprobe: external_reader.page_count(),
                        max_nprobe: external_reader.page_count(),
                        rerank_factor: 32,
                    },
                )
                .await
                .expect("external search");
            let memory_hits = memory_reader
                .search(
                    &query,
                    20,
                    VectorV5SearchOptions {
                        nprobe: memory_reader.page_count(),
                        max_nprobe: memory_reader.page_count(),
                        rerank_factor: 32,
                    },
                )
                .await
                .expect("memory search");
            assert_eq!(external_hits, memory_hits, "{metric:?}");
            assert_eq!(external.stats.point_count, members.len() as u64);
        }
    }

    #[tokio::test]
    async fn singleton_base_roundtrips_all_metrics_with_filters_and_exact_metrics() {
        let scratch = tempfile::tempdir().expect("scratch");
        let node_id = node_id(7);
        let vector = vec![2.0, -1.0, 0.5];
        let filters = BTreeMap::from([
            ("active".to_string(), Value::Bool(true)),
            ("kind".to_string(), Value::Str("law".to_string())),
        ]);
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let desc = descriptor(metric, 3);
            let budget = MIN_MEMORY_BYTES;
            let mut collector = VectorV5ExternalCollector::new_base(config(scratch.path(), budget))
                .expect("base collector");
            collector
                .push(node_id, &vector, &filters)
                .expect("singleton row");
            let mut artifact = collector
                .finish(&desc, options(1))
                .expect("singleton finish")
                .expect("singleton artifact");

            assert_eq!(artifact.stats.point_count, 1, "{metric:?}");
            assert_eq!(artifact.stats.min_node_id, node_id, "{metric:?}");
            assert_eq!(artifact.stats.max_node_id, node_id, "{metric:?}");
            assert_eq!(artifact.stats.entry_medoid, 0, "{metric:?}");
            assert_eq!(artifact.metrics.input_rows, 1, "{metric:?}");
            assert_eq!(artifact.metrics.indexed_rows, 1, "{metric:?}");
            assert_eq!(artifact.metrics.page_count, 1, "{metric:?}");
            assert_eq!(artifact.metrics.partition_count, 1, "{metric:?}");
            assert_eq!(artifact.metrics.max_partition_depth, 0, "{metric:?}");
            assert!(
                artifact.metrics.peak_logical_memory_bytes <= budget,
                "{metric:?}: {:?}",
                artifact.metrics
            );

            let (_, reader) = open_artifact(&mut artifact).await;
            assert_eq!(reader.point_count(), 1, "{metric:?}");
            assert_eq!(reader.page_count(), 1, "{metric:?}");
            assert_eq!(reader.node_id_bounds(), (node_id, node_id), "{metric:?}");
            assert!(reader.supports_filter_property("active"), "{metric:?}");
            assert!(reader.supports_filter_property("kind"), "{metric:?}");
            let groups = [
                ("active".to_string(), vec![Value::Bool(true)]),
                ("kind".to_string(), vec![Value::Str("law".to_string())]),
            ];
            let options = VectorV5SearchOptions {
                nprobe: 1,
                max_nprobe: 1,
                rerank_factor: 8,
            };
            let filtered = reader
                .search_filter_groups(&vector, 4, options, &groups)
                .await
                .expect("singleton filtered search");
            assert_eq!(filtered.applied_filter_groups, 2, "{metric:?}");
            assert_eq!(filtered.probed_pages, 1, "{metric:?}");
            assert_eq!(filtered.eligible_rows_seen, 1, "{metric:?}");
            assert_eq!(filtered.hits.len(), 1, "{metric:?}");
            assert_eq!(filtered.hits[0].0, node_id, "{metric:?}");
            let expected_score = super::super::metric_score(metric, &vector, &vector).0 as f32;
            assert!(
                (filtered.hits[0].1 - expected_score).abs() < 1e-6,
                "{metric:?}: {:?}",
                filtered.hits
            );
            let exact = reader
                .search_exact_filter_groups(&vector, 4, &groups)
                .await
                .expect("singleton exact search");
            assert_eq!(exact, filtered, "{metric:?}");
        }
    }

    #[test]
    fn singleton_cosine_zero_is_empty_but_other_metrics_are_materialized() {
        let scratch = tempfile::tempdir().expect("scratch");
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let desc = descriptor(metric, 3);
            let mut collector =
                VectorV5ExternalCollector::new_base(config(scratch.path(), MIN_MEMORY_BYTES))
                    .expect("base collector");
            collector
                .push(node_id(1), &[0.0; 3], &BTreeMap::new())
                .expect("zero row");
            let artifact = collector.finish(&desc, options(1)).expect("finish");
            if metric == VectorMetric::Cosine {
                assert!(artifact.is_none(), "{metric:?}");
            } else {
                assert_eq!(
                    artifact.expect("zero singleton artifact").stats.point_count,
                    1,
                    "{metric:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn typed_filters_are_spooled_and_applied_page_locally_before_k() {
        let members = rows(256, 8);
        let scratch = tempfile::tempdir().expect("scratch");
        let desc = descriptor(VectorMetric::Cosine, 8);
        let mut collector =
            VectorV5ExternalCollector::new(config(scratch.path(), 128 * 1024)).expect("collector");
        for (ordinal, (id, vector)) in members.iter().enumerate() {
            let filters = BTreeMap::from([
                (
                    "color".to_string(),
                    Value::Str(if ordinal % 3 == 0 { "red" } else { "blue" }.to_string()),
                ),
                ("active".to_string(), Value::Bool(ordinal >= 128)),
            ]);
            collector.push(*id, vector, &filters).expect("push");
        }
        let mut artifact = collector
            .finish(&desc, options(16))
            .expect("finish")
            .expect("artifact");
        let (_, reader) = open_artifact(&mut artifact).await;
        assert!(reader.supports_filter_property("color"));
        assert!(reader.supports_filter_property("active"));
        let result = reader
            .search_filter_groups(
                &members[3].1,
                12,
                VectorV5SearchOptions {
                    nprobe: 1,
                    max_nprobe: reader.page_count(),
                    rerank_factor: 16,
                },
                &[
                    ("color".to_string(), vec![Value::Str("red".to_string())]),
                    ("active".to_string(), vec![Value::Bool(true)]),
                ],
            )
            .await
            .expect("filtered search");
        assert_eq!(result.applied_filter_groups, 2);
        assert_eq!(result.hits.len(), 12);
        for (id, _) in result.hits {
            let ordinal = u32::from_be_bytes(id[..4].try_into().unwrap()) as usize;
            assert!(ordinal >= 128);
            assert_eq!(ordinal % 3, 0);
        }
    }

    #[test]
    fn artifact_is_deterministic_and_stream_contract_rejects_reordering() {
        let members = rows(192, 8);
        let scratch = tempfile::tempdir().expect("scratch");
        let desc = descriptor(VectorMetric::Dot, 8);
        let mut first = build_external(scratch.path(), &desc, &members, 7, 96 * 1024);
        let mut second = build_external(scratch.path(), &desc, &members, 7, 96 * 1024);
        assert_eq!(artifact_bytes(&mut first), artifact_bytes(&mut second));
        assert_eq!(
            std::fs::read_dir(scratch.path())
                .expect("successful scratch cleanup")
                .count(),
            0,
            "only anonymous artifact files may survive finish"
        );

        let mut collector =
            VectorV5ExternalCollector::new(config(scratch.path(), 96 * 1024)).expect("collector");
        collector
            .push(members[1].0, &members[1].1, &BTreeMap::new())
            .expect("first");
        assert!(collector
            .push(members[0].0, &members[0].1, &BTreeMap::new())
            .is_err());
    }

    #[test]
    fn explicit_base_constructor_emits_the_existing_bounded_v5_artifact() {
        let members = rows(192, 8);
        let scratch = tempfile::tempdir().expect("scratch");
        let desc = descriptor(VectorMetric::Cosine, 8);
        let build_config = config(scratch.path(), 128 * 1024);

        let mut explicit =
            VectorV5ExternalCollector::new_base(build_config.clone()).expect("base collector");
        let mut legacy =
            VectorV5ExternalCollector::new(build_config).expect("legacy collector alias");
        for (id, vector) in &members {
            explicit
                .push(*id, vector, &BTreeMap::new())
                .expect("explicit base row");
            legacy
                .push(*id, vector, &BTreeMap::new())
                .expect("legacy base row");
        }
        let mut explicit = explicit
            .finish(&desc, options(16))
            .expect("finish explicit base")
            .expect("explicit base artifact");
        let mut legacy = legacy
            .finish(&desc, options(16))
            .expect("finish legacy base")
            .expect("legacy base artifact");

        assert_eq!(artifact_bytes(&mut explicit), artifact_bytes(&mut legacy));
        assert_eq!(explicit.stats.dim, legacy.stats.dim);
        assert_eq!(explicit.stats.metric, legacy.stats.metric);
        assert_eq!(explicit.stats.point_count, legacy.stats.point_count);
        assert_eq!(explicit.stats.min_node_id, legacy.stats.min_node_id);
        assert_eq!(explicit.stats.max_node_id, legacy.stats.max_node_id);
        assert_eq!(explicit.stats.r, legacy.stats.r);
        assert_eq!(explicit.stats.l_build, legacy.stats.l_build);
        assert_eq!(explicit.stats.alpha, legacy.stats.alpha);
        assert_eq!(explicit.stats.entry_medoid, legacy.stats.entry_medoid);
        assert_eq!(explicit.metrics, legacy.metrics);
        assert_eq!(explicit.metrics.indexed_rows, members.len() as u64);
        assert!(explicit.metrics.peak_logical_memory_bytes <= 128 * 1024);
    }

    #[test]
    fn validates_rows_and_cosine_zero_filter_preserves_indexed_bounds() {
        let scratch = tempfile::tempdir().expect("scratch");
        let desc = descriptor(VectorMetric::Cosine, 4);
        let mut collector =
            VectorV5ExternalCollector::new(config(scratch.path(), MIN_MEMORY_BYTES))
                .expect("collector");
        collector
            .push(node_id(0), &[0.0; 4], &BTreeMap::new())
            .expect("zero is accepted into metric-agnostic spool");
        collector
            .push(node_id(1), &[1.0, 0.0, 0.0, 0.0], &BTreeMap::new())
            .expect("first indexed");
        assert!(collector
            .push(node_id(2), &[f32::NAN, 0.0, 0.0, 0.0], &BTreeMap::new())
            .is_err());
        assert!(collector
            .push(node_id(2), &[1.0, 0.0, 0.0], &BTreeMap::new())
            .is_err());
        collector
            .push(node_id(2), &[0.0, 1.0, 0.0, 0.0], &BTreeMap::new())
            .expect("second indexed");
        let artifact = collector
            .finish(&desc, options(2))
            .expect("finish")
            .expect("artifact");
        assert_eq!(artifact.stats.point_count, 2);
        assert_eq!(artifact.stats.min_node_id, node_id(1));
        assert_eq!(artifact.stats.max_node_id, node_id(2));
        assert_eq!(artifact.metrics.input_rows, 3);
        assert_eq!(artifact.metrics.indexed_rows, 2);
    }

    #[tokio::test]
    async fn tiny_pages_and_all_equal_axes_make_balanced_progress() {
        let members: Vec<_> = (0..97u32)
            .map(|ordinal| (node_id(ordinal), vec![1.0, 1.0, 1.0, 1.0]))
            .collect();
        let scratch = tempfile::tempdir().expect("scratch");
        let desc = descriptor(VectorMetric::Dot, 4);
        let mut artifact = build_external(scratch.path(), &desc, &members, 2, MIN_MEMORY_BYTES);
        assert!(artifact.metrics.partition_count > artifact.metrics.page_count);
        assert!(artifact.metrics.max_partition_depth < 16);
        let (_, reader) = open_artifact(&mut artifact).await;
        assert!(reader.page_count() >= members.len().div_ceil(2));
        let hits = reader
            .search(
                &[1.0; 4],
                5,
                VectorV5SearchOptions {
                    nprobe: reader.page_count(),
                    max_nprobe: reader.page_count(),
                    rerank_factor: 32,
                },
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 5);
    }

    #[test]
    fn spool_corruption_is_detected_and_all_named_scratch_is_cleaned() {
        let scratch = tempfile::tempdir().expect("scratch");
        let desc = descriptor(VectorMetric::Cosine, 4);
        let members = rows(16, 4);
        let mut collector =
            VectorV5ExternalCollector::new(config(scratch.path(), MIN_MEMORY_BYTES))
                .expect("collector");
        for (id, vector) in &members {
            collector.push(*id, vector, &BTreeMap::new()).expect("push");
        }
        assert!(std::fs::read_dir(scratch.path())
            .expect("list")
            .next()
            .is_some());
        collector
            .spool
            .as_file_mut()
            .seek(SeekFrom::Start(SPOOL_HEADER_LEN + 4 + 3))
            .expect("seek");
        collector
            .spool
            .as_file_mut()
            .write_all(&[0xA5])
            .expect("corrupt");
        assert!(collector.finish(&desc, options(4)).is_err());
        assert_eq!(
            std::fs::read_dir(scratch.path())
                .expect("list after")
                .count(),
            0
        );
    }

    #[test]
    fn logical_peak_is_budget_bounded_and_independent_of_corpus_size() {
        let scratch = tempfile::tempdir().expect("scratch");
        let desc = descriptor(VectorMetric::Cosine, 16);
        let small_rows = rows(128, 16);
        let large_rows = rows(4_096, 16);
        // 4,096 rows at 8 rows/page deliberately create ~512 leaf descriptors;
        // 128 KiB is below that footer's measured 140 KiB requirement and must
        // be rejected. 256 KiB leaves room for both metadata and fixed working
        // set while still being tiny relative to corpus payload.
        let budget = 256 * 1024;
        let small = build_external(scratch.path(), &desc, &small_rows, 8, budget);
        let large = build_external(scratch.path(), &desc, &large_rows, 8, budget);
        assert!(small.metrics.peak_logical_memory_bytes <= budget);
        assert!(large.metrics.peak_logical_memory_bytes <= budget);
        assert!(
            large.metrics.peak_working_memory_bytes
                <= small
                    .metrics
                    .peak_working_memory_bytes
                    .saturating_mul(2)
                    .saturating_add(4096),
            "small={} large={}",
            small.metrics.peak_working_memory_bytes,
            large.metrics.peak_working_memory_bytes
        );
        assert!(large.metrics.resident_metadata_bytes > small.metrics.resident_metadata_bytes);
        assert!(large.metrics.partition_count > small.metrics.partition_count);
    }

    #[test]
    fn quantized_footer_navigation_tier_meets_ten_million_row_target() {
        let dim = 1_024usize;
        let rows = 10_000_000usize;
        let page_rows = 512usize;
        let leaves = rows.div_ceil(page_rows);
        let branch = 8usize;
        let nodes = leaves
            .checked_mul(branch)
            .expect("node estimate")
            .div_ceil(branch - 1);
        let estimated = nodes
            .checked_mul(dim + 4 + std::mem::size_of::<CentroidNode>() + 4)
            .expect("metadata estimate");
        assert!(
            estimated <= 32 * 1024 * 1024,
            "estimated centroid tier is {} MiB",
            estimated / (1024 * 1024)
        );
    }
}
