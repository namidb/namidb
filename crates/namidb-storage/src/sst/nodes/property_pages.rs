//! Object-native, independently ranged node-property pages.
//!
//! This optional sidecar is deliberately standalone from the legacy Parquet
//! and `.nloc2` readers. A future manifest generation can point at it without
//! changing the existing wire contracts. Builders consume NodeId-sorted rows,
//! external-sort all schemaless cells through one bounded scratch pipeline,
//! and emit a catalog, fixed-size page index, and independently compressed
//! value pages.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use namidb_core::Value;
use uuid::Uuid;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use crate::error::{Error, Result};
use crate::range_cache::{PinnedObjectRangeSource, PinnedObjectRangeStats};
use crate::sst::paged_index::create_spool_file;

const HEADER_MAGIC: [u8; 8] = *b"NAMINPP1";
const FOOTER_MAGIC: [u8; 8] = *b"NPPFTR01";
pub const NODE_PROPERTY_PAGES_FORMAT_VERSION: u16 = 1;
const HEADER_LEN: usize = 192;
const FOOTER_LEN: usize = 80;
const INDEX_ENTRY_LEN: usize = 100;
const RUN_RECORD_FIXED_LEN: usize = 52;
const CATALOG_ENTRY_FIXED_LEN: usize = 92;
const PAGE_HEADER_MAGIC: [u8; 8] = *b"NPPCELL1";
const PAGE_HEADER_LEN: usize = 16;
const PAGE_ROW_FIXED_LEN: usize = 32;
const COPY_BUFFER_BYTES: usize = 64 * 1024;
const VALUE_STREAM_BUFFER_BYTES: usize = 16 * 1024;
const RUN_FAN_IN: usize = 32;
const FLAG_COMPLETE: u32 = 1;
const CODEC_RAW: u8 = 0;
const CODEC_ZSTD: u8 = 1;

pub const DEFAULT_NODE_PROPERTY_SORT_MEMORY_BYTES: usize = 8 * 1024 * 1024;
pub const DEFAULT_NODE_PROPERTY_PAGE_TARGET_BYTES: usize = 1024 * 1024;
pub const DEFAULT_NODE_PROPERTY_MAX_VALUE_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_NODE_PROPERTY_MAX_PAGE_BYTES: usize = 65 * 1024 * 1024;
pub const DEFAULT_NODE_PROPERTY_MAX_CATALOG_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_NODE_PROPERTY_MAX_PROPERTIES: u32 = 65_536;
pub const DEFAULT_NODE_PROPERTY_MAX_VALUE_DEPTH: u16 = 32;
pub const DEFAULT_NODE_PROPERTY_MAX_VALUE_ELEMENTS: u64 = 1_000_000;
pub const DEFAULT_NODE_PROPERTY_MAX_INDEX_QUERY_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_NODE_PROPERTY_MAX_QUERY_ROWS: usize = 1_000_000;

/// Explicit build/query ceilings. Every ceiling is authoritative: crossing it
/// returns an error and never drops a cell, property, or page.
#[derive(Debug, Clone)]
pub struct NodePropertyPageConfig {
    pub sort_memory_bytes: usize,
    pub page_target_bytes: usize,
    pub max_value_bytes: usize,
    pub max_page_decoded_bytes: usize,
    pub max_catalog_bytes: usize,
    pub max_properties: u32,
    pub max_value_depth: u16,
    pub max_value_elements: u64,
    pub max_property_name_bytes: usize,
    pub max_index_query_bytes: usize,
    pub max_query_rows: usize,
    pub compress_pages: bool,
}

impl Default for NodePropertyPageConfig {
    fn default() -> Self {
        Self {
            sort_memory_bytes: DEFAULT_NODE_PROPERTY_SORT_MEMORY_BYTES,
            page_target_bytes: DEFAULT_NODE_PROPERTY_PAGE_TARGET_BYTES,
            max_value_bytes: DEFAULT_NODE_PROPERTY_MAX_VALUE_BYTES,
            max_page_decoded_bytes: DEFAULT_NODE_PROPERTY_MAX_PAGE_BYTES,
            max_catalog_bytes: DEFAULT_NODE_PROPERTY_MAX_CATALOG_BYTES,
            max_properties: DEFAULT_NODE_PROPERTY_MAX_PROPERTIES,
            max_value_depth: DEFAULT_NODE_PROPERTY_MAX_VALUE_DEPTH,
            max_value_elements: DEFAULT_NODE_PROPERTY_MAX_VALUE_ELEMENTS,
            max_property_name_bytes: 1024,
            max_index_query_bytes: DEFAULT_NODE_PROPERTY_MAX_INDEX_QUERY_BYTES,
            max_query_rows: DEFAULT_NODE_PROPERTY_MAX_QUERY_ROWS,
            compress_pages: true,
        }
    }
}

impl NodePropertyPageConfig {
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        config.sort_memory_bytes = env_usize(
            "NAMIDB_NODE_PROPERTY_SORT_MEMORY_BYTES",
            config.sort_memory_bytes,
        )?;
        config.page_target_bytes = env_usize(
            "NAMIDB_NODE_PROPERTY_PAGE_TARGET_BYTES",
            config.page_target_bytes,
        )?;
        config.max_value_bytes = env_usize(
            "NAMIDB_NODE_PROPERTY_MAX_VALUE_BYTES",
            config.max_value_bytes,
        )?;
        config.max_page_decoded_bytes = env_usize(
            "NAMIDB_NODE_PROPERTY_MAX_PAGE_BYTES",
            config.max_page_decoded_bytes,
        )?;
        config.max_catalog_bytes = env_usize(
            "NAMIDB_NODE_PROPERTY_MAX_CATALOG_BYTES",
            config.max_catalog_bytes,
        )?;
        config.max_properties =
            env_u32("NAMIDB_NODE_PROPERTY_MAX_PROPERTIES", config.max_properties)?;
        config.max_value_depth = env_u16(
            "NAMIDB_NODE_PROPERTY_MAX_VALUE_DEPTH",
            config.max_value_depth,
        )?;
        config.max_value_elements = env_u64(
            "NAMIDB_NODE_PROPERTY_MAX_VALUE_ELEMENTS",
            config.max_value_elements,
        )?;
        config.max_property_name_bytes = env_usize(
            "NAMIDB_NODE_PROPERTY_MAX_NAME_BYTES",
            config.max_property_name_bytes,
        )?;
        config.max_index_query_bytes = env_usize(
            "NAMIDB_NODE_PROPERTY_MAX_INDEX_QUERY_BYTES",
            config.max_index_query_bytes,
        )?;
        config.max_query_rows =
            env_usize("NAMIDB_NODE_PROPERTY_MAX_QUERY_ROWS", config.max_query_rows)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.sort_memory_bytes < 64 * 1024 {
            return Err(Error::precondition(
                "node property sort memory must be at least 64 KiB",
            ));
        }
        if self.page_target_bytes == 0
            || self.max_value_bytes == 0
            || self.max_page_decoded_bytes == 0
            || self.max_catalog_bytes == 0
            || self.max_properties == 0
            || self.max_value_depth == 0
            || self.max_value_elements == 0
            || self.max_property_name_bytes == 0
            || self.max_index_query_bytes == 0
            || self.max_query_rows == 0
        {
            return Err(Error::precondition(
                "node property page limits must all be non-zero",
            ));
        }
        if self.page_target_bytes > self.max_page_decoded_bytes {
            return Err(Error::precondition(
                "node property page target exceeds its decoded-page ceiling",
            ));
        }
        let minimum_value_page = self
            .max_value_bytes
            .checked_add(PAGE_HEADER_LEN + PAGE_ROW_FIXED_LEN)
            .ok_or_else(|| Error::precondition("node property value page ceiling overflows"))?;
        if minimum_value_page > self.max_page_decoded_bytes {
            return Err(Error::precondition(
                "node property decoded-page ceiling cannot hold one maximum-sized value",
            ));
        }
        if self.max_value_bytes > u32::MAX as usize {
            return Err(Error::precondition(
                "node property value ceiling exceeds the u32 wire limit",
            ));
        }
        if self.max_property_name_bytes > u16::MAX as usize {
            return Err(Error::precondition(
                "node property name ceiling exceeds the u16 wire limit",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct DiskSpool {
    file: Option<File>,
    len: u64,
}

impl DiskSpool {
    fn ensure(&mut self) -> std::io::Result<&mut File> {
        if self.file.is_none() {
            self.file = Some(create_spool_file()?);
        }
        Ok(self.file.as_mut().expect("spool file installed"))
    }

    fn rewind(&mut self) -> std::io::Result<()> {
        self.ensure()?.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    fn into_file(mut self) -> Result<(File, u64)> {
        self.ensure()?;
        let mut file = self.file.take().expect("spool file installed");
        file.flush()?;
        file.sync_data()?;
        let actual = file.metadata()?.len();
        if actual != self.len {
            return Err(Error::invariant(format!(
                "node property spool length {} != file length {actual}",
                self.len
            )));
        }
        file.seek(SeekFrom::Start(0))?;
        Ok((file, self.len))
    }
}

impl Write for DiskSpool {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.ensure()?.write(bytes)?;
        self.len = self
            .len
            .checked_add(written as u64)
            .ok_or_else(|| std::io::Error::other("node property spool exceeds u64"))?;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.ensure()?.flush()
    }
}

/// Distinguishes a missing key from an explicit `Value::Null`.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyCell {
    Absent,
    Null,
    Value(Value),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodePropertyProjection {
    pub node_id: [u8; 16],
    pub ordinal: Option<u64>,
    pub properties: BTreeMap<String, PropertyCell>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodePropertyPredicate {
    EqString {
        property: String,
        value: String,
    },
    InString {
        property: String,
        values: Vec<String>,
    },
    EqBool {
        property: String,
        value: bool,
    },
    InBool {
        property: String,
        values: Vec<bool>,
    },
    /// Kept explicit so callers never mistake an unimplemented predicate for
    /// a complete empty result.
    Unsupported {
        description: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodePropertyFilterResult {
    Matches(Vec<NodePropertyProjection>),
    Unsupported { reason: String },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NodePropertyProbeStats {
    pub range_requests: u64,
    pub logical_bytes: u64,
    pub payload_pages: u64,
    pub payload_bytes: u64,
}

impl NodePropertyProbeStats {
    fn between(
        before: PinnedObjectRangeStats,
        after: PinnedObjectRangeStats,
        payload_pages: u64,
        payload_bytes: u64,
    ) -> Self {
        Self {
            range_requests: after.logical_requests - before.logical_requests,
            logical_bytes: after.logical_bytes - before.logical_bytes,
            payload_pages,
            payload_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SortCell {
    property: Vec<u8>,
    node_id: [u8; 16],
    ordinal: u64,
    state: u8,
    value: ValueExtent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValueExtent {
    offset: u64,
    len: u32,
    crc32: u32,
}

impl Ord for SortCell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.property, self.node_id, self.ordinal).cmp(&(
            &other.property,
            other.node_id,
            other.ordinal,
        ))
    }
}

impl PartialOrd for SortCell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct SortRun {
    file: File,
    count: u64,
}

#[derive(Debug)]
struct CellSorter {
    memory_limit: usize,
    key_memory_limit: usize,
    buffered_bytes: usize,
    peak_buffered_bytes: usize,
    max_record_bytes: usize,
    buffered: Vec<SortCell>,
    levels: Vec<Vec<SortRun>>,
    values: DiskSpool,
}

impl CellSorter {
    fn new(memory_limit: usize) -> Self {
        Self {
            memory_limit,
            key_memory_limit: memory_limit.saturating_sub(VALUE_STREAM_BUFFER_BYTES),
            buffered_bytes: 0,
            peak_buffered_bytes: 0,
            max_record_bytes: 0,
            buffered: Vec::new(),
            levels: Vec::new(),
            values: DiskSpool::default(),
        }
    }

    fn push(&mut self, cell: SortCell) -> Result<()> {
        let retained = 96usize
            .checked_add(cell.property.capacity())
            .ok_or_else(|| Error::invariant("node property sort record exceeds usize"))?;
        if retained > self.key_memory_limit / 2 {
            return Err(Error::precondition(format!(
                "one node property cell requires {retained} retained sort bytes; the \
                 two-way merge needs twice that within \
                 NAMIDB_NODE_PROPERTY_SORT_MEMORY_BYTES={}",
                self.memory_limit
            )));
        }
        if !self.buffered.is_empty()
            && self.buffered_bytes.saturating_add(retained) > self.key_memory_limit
        {
            self.flush_buffer()?;
        }
        self.buffered.push(cell);
        self.buffered_bytes = self
            .buffered_bytes
            .checked_add(retained)
            .ok_or_else(|| Error::invariant("node property sort memory exceeds usize"))?;
        self.max_record_bytes = self.max_record_bytes.max(retained);
        self.peak_buffered_bytes = self.peak_buffered_bytes.max(self.buffered_bytes);
        Ok(())
    }

    fn append_value(
        &mut self,
        value: &Value,
        config: &NodePropertyPageConfig,
    ) -> Result<ValueExtent> {
        append_canonical_value(&mut self.values, value, config)
    }

    fn null_extent(&self) -> ValueExtent {
        ValueExtent {
            offset: self.values.len,
            len: 0,
            crc32: 0,
        }
    }

    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffered.is_empty() {
            return Ok(());
        }
        self.buffered.sort_unstable();
        let mut file = create_spool_file()?;
        for cell in &self.buffered {
            write_run_cell(&mut file, cell)?;
        }
        file.sync_data()?;
        file.seek(SeekFrom::Start(0))?;
        let run = SortRun {
            file,
            count: self.buffered.len() as u64,
        };
        self.buffered.clear();
        self.buffered_bytes = 0;
        self.insert_run(0, run)
    }

    fn insert_run(&mut self, level: usize, run: SortRun) -> Result<()> {
        if self.levels.len() <= level {
            self.levels.resize_with(level + 1, Vec::new);
        }
        self.levels[level].push(run);
        let fan_in = self.merge_fan_in();
        if self.levels[level].len() < fan_in {
            return Ok(());
        }
        let inputs = self.levels[level].drain(..fan_in).collect();
        let merged = merge_runs(inputs)?;
        self.insert_run(level + 1, merged)
    }

    fn finish(mut self) -> Result<(CellMerge, DiskSpool, usize)> {
        self.flush_buffer()?;
        self.values.flush()?;
        let fan_in = self.merge_fan_in();
        let max_record_bytes = self.max_record_bytes;
        let peak_buffered_bytes = self.peak_buffered_bytes;
        let values = std::mem::take(&mut self.values);
        let mut runs: Vec<SortRun> = self.levels.into_iter().flatten().collect();
        while runs.len() > fan_in {
            let mut output = Vec::with_capacity(runs.len().div_ceil(fan_in));
            let mut inputs = runs.into_iter();
            loop {
                let group: Vec<_> = inputs.by_ref().take(fan_in).collect();
                if group.is_empty() {
                    break;
                }
                output.push(merge_runs(group)?);
            }
            runs = output;
        }
        let merge_memory = max_record_bytes.saturating_mul(runs.len());
        Ok((
            CellMerge::new(runs)?,
            values,
            peak_buffered_bytes
                .saturating_add(VALUE_STREAM_BUFFER_BYTES)
                .max(merge_memory),
        ))
    }

    fn merge_fan_in(&self) -> usize {
        if self.max_record_bytes == 0 {
            return RUN_FAN_IN;
        }
        (self.key_memory_limit / self.max_record_bytes).clamp(2, RUN_FAN_IN)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct HeapCell {
    cell: SortCell,
    run: usize,
}

impl Ord for HeapCell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cell
            .cmp(&other.cell)
            .then_with(|| self.run.cmp(&other.run))
    }
}

impl PartialOrd for HeapCell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct RunCursor {
    file: File,
    remaining: u64,
}

#[derive(Debug)]
struct CellMerge {
    cursors: Vec<RunCursor>,
    heap: BinaryHeap<Reverse<HeapCell>>,
    pending_run: Option<usize>,
}

impl CellMerge {
    fn new(runs: Vec<SortRun>) -> Result<Self> {
        let mut cursors: Vec<RunCursor> = runs
            .into_iter()
            .map(|run| RunCursor {
                file: run.file,
                remaining: run.count,
            })
            .collect();
        let mut heap = BinaryHeap::new();
        for (index, cursor) in cursors.iter_mut().enumerate() {
            if let Some(cell) = read_next_cell(cursor)? {
                heap.push(Reverse(HeapCell { cell, run: index }));
            }
        }
        Ok(Self {
            cursors,
            heap,
            pending_run: None,
        })
    }

    fn next_cell(&mut self) -> Result<Option<SortCell>> {
        if let Some(run) = self.pending_run.take() {
            if let Some(next) = read_next_cell(&mut self.cursors[run])? {
                self.heap.push(Reverse(HeapCell { cell: next, run }));
            }
        }
        let Some(Reverse(item)) = self.heap.pop() else {
            return Ok(None);
        };
        let cell = item.cell;
        self.pending_run = Some(item.run);
        Ok(Some(cell))
    }
}

fn merge_runs(runs: Vec<SortRun>) -> Result<SortRun> {
    let count = runs.iter().try_fold(0u64, |count, run| {
        count
            .checked_add(run.count)
            .ok_or_else(|| Error::invariant("node property run count exceeds u64"))
    })?;
    let mut merge = CellMerge::new(runs)?;
    let mut output = create_spool_file()?;
    while let Some(cell) = merge.next_cell()? {
        write_run_cell(&mut output, &cell)?;
    }
    output.sync_data()?;
    output.seek(SeekFrom::Start(0))?;
    Ok(SortRun {
        file: output,
        count,
    })
}

fn write_run_cell(writer: &mut File, cell: &SortCell) -> Result<()> {
    let property_len = u32::try_from(cell.property.len())
        .map_err(|_| Error::invariant("node property name exceeds u32"))?;
    let mut fixed = [0u8; RUN_RECORD_FIXED_LEN];
    fixed[..4].copy_from_slice(&property_len.to_le_bytes());
    fixed[4..20].copy_from_slice(&cell.node_id);
    fixed[20..28].copy_from_slice(&cell.ordinal.to_le_bytes());
    fixed[28] = cell.state;
    fixed[32..40].copy_from_slice(&cell.value.offset.to_le_bytes());
    fixed[40..44].copy_from_slice(&cell.value.len.to_le_bytes());
    fixed[44..48].copy_from_slice(&cell.value.crc32.to_le_bytes());
    let mut crc = crc32fast::Hasher::new();
    crc.update(&fixed[..48]);
    crc.update(&cell.property);
    fixed[48..52].copy_from_slice(&crc.finalize().to_le_bytes());
    writer.write_all(&fixed)?;
    writer.write_all(&cell.property)?;
    Ok(())
}

fn read_next_cell(cursor: &mut RunCursor) -> Result<Option<SortCell>> {
    if cursor.remaining == 0 {
        return Ok(None);
    }
    let mut fixed = [0u8; RUN_RECORD_FIXED_LEN];
    cursor.file.read_exact(&mut fixed)?;
    let property_len = u32::from_le_bytes(fixed[..4].try_into().unwrap()) as usize;
    let node_id = fixed[4..20].try_into().unwrap();
    let ordinal = u64::from_le_bytes(fixed[20..28].try_into().unwrap());
    let state = fixed[28];
    let value = ValueExtent {
        offset: u64::from_le_bytes(fixed[32..40].try_into().unwrap()),
        len: u32::from_le_bytes(fixed[40..44].try_into().unwrap()),
        crc32: u32::from_le_bytes(fixed[44..48].try_into().unwrap()),
    };
    if property_len == 0
        || property_len > u16::MAX as usize
        || fixed[29..32] != [0, 0, 0]
        || state > 1
        || (state == 0 && (value.len != 0 || value.crc32 != 0))
        || (state == 1 && value.len == 0)
    {
        return Err(corrupted("node property scratch record flags invalid"));
    }
    let expected_crc = u32::from_le_bytes(fixed[48..52].try_into().unwrap());
    let mut property = vec![0u8; property_len];
    cursor.file.read_exact(&mut property)?;
    let mut crc = crc32fast::Hasher::new();
    crc.update(&fixed[..48]);
    crc.update(&property);
    if crc.finalize() != expected_crc {
        return Err(corrupted("node property scratch record CRC mismatch"));
    }
    cursor.remaining -= 1;
    Ok(Some(SortCell {
        property,
        node_id,
        ordinal,
        state,
        value,
    }))
}

fn append_canonical_value(
    output: &mut DiskSpool,
    value: &Value,
    config: &NodePropertyPageConfig,
) -> Result<ValueExtent> {
    let mut elements = 0u64;
    let encoded_len = measure_value(value, 0, &mut elements, config)?;
    if encoded_len == 0 || encoded_len > config.max_value_bytes {
        return Err(Error::precondition(format!(
            "encoded node property value requires {encoded_len} bytes, above \
             NAMIDB_NODE_PROPERTY_MAX_VALUE_BYTES={}",
            config.max_value_bytes
        )));
    }
    let len = u32::try_from(encoded_len)
        .map_err(|_| Error::precondition("encoded node property Value exceeds u32"))?;
    let offset = output.len;
    let mut writer = CanonicalValueWriter {
        output,
        crc: crc32fast::Hasher::new(),
        written: 0,
    };
    encode_value_streaming(value, &mut writer)?;
    if writer.written != u64::from(len) {
        return Err(Error::invariant(format!(
            "streamed node property Value wrote {} bytes, measured {len}",
            writer.written
        )));
    }
    let actual_end = writer.output.len;
    let crc32 = writer.crc.finalize();
    let expected_end = offset
        .checked_add(u64::from(len))
        .ok_or_else(|| Error::invariant("node property value extent exceeds u64"))?;
    if actual_end != expected_end {
        return Err(Error::invariant(
            "node property value spool position disagrees with its extent",
        ));
    }
    Ok(ValueExtent { offset, len, crc32 })
}

fn measure_value(
    value: &Value,
    depth: u16,
    elements: &mut u64,
    config: &NodePropertyPageConfig,
) -> Result<usize> {
    if depth > config.max_value_depth {
        return Err(Error::precondition(format!(
            "node property Value depth exceeds {}",
            config.max_value_depth
        )));
    }
    *elements = elements
        .checked_add(1)
        .ok_or_else(|| Error::invariant("node property Value elements exceed u64"))?;
    if *elements > config.max_value_elements {
        return Err(Error::precondition(format!(
            "node property Value elements exceed {}",
            config.max_value_elements
        )));
    }
    let len = match value {
        Value::Null => 1,
        Value::Bool(_) => 2,
        Value::I64(_) | Value::F64(_) | Value::DateTime(_) => 9,
        Value::Date(_) => 5,
        Value::Str(value) => {
            require_u32_len(value.len(), "node property String")?;
            wire_sum(&[1, 4, value.len()])?
        }
        Value::Bytes(value) => {
            require_u32_len(value.len(), "node property Bytes")?;
            wire_sum(&[1, 4, value.len()])?
        }
        Value::Vec(value) => {
            add_value_elements(elements, value.len(), config)?;
            require_u32_len(value.len(), "node property vector")?;
            wire_sum(&[
                1,
                4,
                value
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| Error::precondition("vector wire length exceeds usize"))?,
            ])?
        }
        Value::VecI8 { codes, .. } => {
            add_value_elements(elements, codes.len(), config)?;
            require_u32_len(codes.len(), "node property int8 vector")?;
            wire_sum(&[1, 4, codes.len(), 4])?
        }
        Value::List(values) => {
            require_u32_len(values.len(), "node property List")?;
            let mut len = 5usize;
            for value in values {
                len = len
                    .checked_add(measure_value(value, depth + 1, elements, config)?)
                    .ok_or_else(|| Error::precondition("List wire length exceeds usize"))?;
            }
            len
        }
        Value::Map(values) => {
            require_u32_len(values.len(), "node property Map")?;
            let mut len = 5usize;
            for (key, value) in values {
                if key.len() > config.max_property_name_bytes {
                    return Err(Error::precondition(format!(
                        "node property Map key requires {} bytes, above {}",
                        key.len(),
                        config.max_property_name_bytes
                    )));
                }
                require_u32_len(key.len(), "node property Map key")?;
                let value_len = measure_value(value, depth + 1, elements, config)?;
                len = len
                    .checked_add(4)
                    .and_then(|len| len.checked_add(key.len()))
                    .and_then(|len| len.checked_add(value_len))
                    .ok_or_else(|| Error::precondition("Map wire length exceeds usize"))?;
            }
            len
        }
    };
    if len > config.max_value_bytes {
        return Err(Error::precondition(format!(
            "encoded node property value exceeds {} bytes",
            config.max_value_bytes
        )));
    }
    Ok(len)
}

fn add_value_elements(
    elements: &mut u64,
    count: usize,
    config: &NodePropertyPageConfig,
) -> Result<()> {
    *elements = elements
        .checked_add(count as u64)
        .ok_or_else(|| Error::invariant("node property Value elements exceed u64"))?;
    if *elements > config.max_value_elements {
        return Err(Error::precondition(format!(
            "node property Value elements exceed {}",
            config.max_value_elements
        )));
    }
    Ok(())
}

fn require_u32_len(len: usize, kind: &str) -> Result<u32> {
    u32::try_from(len).map_err(|_| Error::precondition(format!("{kind} length exceeds u32")))
}

fn wire_sum(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(0usize, |total, part| {
        total
            .checked_add(*part)
            .ok_or_else(|| Error::precondition("node property Value wire length exceeds usize"))
    })
}

struct CanonicalValueWriter<'a> {
    output: &'a mut DiskSpool,
    crc: crc32fast::Hasher,
    written: u64,
}

impl CanonicalValueWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.output.write_all(bytes)?;
        self.crc.update(bytes);
        self.written = self
            .written
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::invariant("streamed node property Value exceeds u64"))?;
        Ok(())
    }

    fn write_len(&mut self, len: usize, kind: &str) -> Result<()> {
        self.write(&require_u32_len(len, kind)?.to_le_bytes())
    }
}

fn encode_value_streaming(value: &Value, output: &mut CanonicalValueWriter<'_>) -> Result<()> {
    match value {
        Value::Null => output.write(&[0])?,
        Value::Bool(value) => output.write(&[1, u8::from(*value)])?,
        Value::I64(value) => {
            output.write(&[2])?;
            output.write(&value.to_le_bytes())?;
        }
        Value::F64(value) => {
            output.write(&[3])?;
            output.write(&value.to_bits().to_le_bytes())?;
        }
        Value::Str(value) => {
            output.write(&[4])?;
            output.write_len(value.len(), "node property String")?;
            output.write(value.as_bytes())?;
        }
        Value::Bytes(value) => {
            output.write(&[5])?;
            output.write_len(value.len(), "node property Bytes")?;
            output.write(value)?;
        }
        Value::Vec(values) => {
            output.write(&[6])?;
            output.write_len(values.len(), "node property vector")?;
            let mut buffer = [0u8; VALUE_STREAM_BUFFER_BYTES];
            for chunk in values.chunks(buffer.len() / 4) {
                for (index, component) in chunk.iter().enumerate() {
                    buffer[index * 4..index * 4 + 4]
                        .copy_from_slice(&component.to_bits().to_le_bytes());
                }
                output.write(&buffer[..chunk.len() * 4])?;
            }
        }
        Value::VecI8 { codes, scale } => {
            output.write(&[7])?;
            output.write_len(codes.len(), "node property int8 vector")?;
            let mut buffer = [0u8; VALUE_STREAM_BUFFER_BYTES];
            for chunk in codes.chunks(buffer.len()) {
                for (destination, code) in buffer.iter_mut().zip(chunk) {
                    *destination = *code as u8;
                }
                output.write(&buffer[..chunk.len()])?;
            }
            output.write(&scale.to_bits().to_le_bytes())?;
        }
        Value::Date(value) => {
            output.write(&[8])?;
            output.write(&value.to_le_bytes())?;
        }
        Value::DateTime(value) => {
            output.write(&[9])?;
            output.write(&value.to_le_bytes())?;
        }
        Value::List(values) => {
            output.write(&[10])?;
            output.write_len(values.len(), "node property List")?;
            for value in values {
                encode_value_streaming(value, output)?;
            }
        }
        Value::Map(values) => {
            output.write(&[11])?;
            output.write_len(values.len(), "node property Map")?;
            for (key, value) in values {
                output.write_len(key.len(), "node property Map key")?;
                output.write(key.as_bytes())?;
                encode_value_streaming(value, output)?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PageIndexEntry {
    property_id: u32,
    page_ordinal: u32,
    first_node_id: [u8; 16],
    last_node_id: [u8; 16],
    first_ordinal: u64,
    last_ordinal: u64,
    payload_offset: u64,
    encoded_len: u64,
    decoded_len: u64,
    row_count: u32,
    codec: u8,
    crc32: u32,
    salted_xxh3: u64,
}

impl PageIndexEntry {
    fn encode(&self, output: &mut Vec<u8>) {
        let start = output.len();
        output.extend_from_slice(&self.property_id.to_le_bytes());
        output.extend_from_slice(&self.page_ordinal.to_le_bytes());
        output.extend_from_slice(&self.first_node_id);
        output.extend_from_slice(&self.last_node_id);
        output.extend_from_slice(&self.first_ordinal.to_le_bytes());
        output.extend_from_slice(&self.last_ordinal.to_le_bytes());
        output.extend_from_slice(&self.payload_offset.to_le_bytes());
        output.extend_from_slice(&self.encoded_len.to_le_bytes());
        output.extend_from_slice(&self.decoded_len.to_le_bytes());
        output.extend_from_slice(&self.row_count.to_le_bytes());
        output.push(self.codec);
        output.extend_from_slice(&[0, 0, 0]);
        output.extend_from_slice(&self.crc32.to_le_bytes());
        output.extend_from_slice(&self.salted_xxh3.to_le_bytes());
        debug_assert_eq!(output.len() - start, INDEX_ENTRY_LEN);
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != INDEX_ENTRY_LEN {
            return Err(corrupted("node property page-index entry truncated"));
        }
        let codec = bytes[84];
        if bytes[85..88] != [0, 0, 0] || !matches!(codec, CODEC_RAW | CODEC_ZSTD) {
            return Err(corrupted("node property page-index flags invalid"));
        }
        Ok(Self {
            property_id: u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            page_ordinal: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            first_node_id: bytes[8..24].try_into().unwrap(),
            last_node_id: bytes[24..40].try_into().unwrap(),
            first_ordinal: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            last_ordinal: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            payload_offset: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
            encoded_len: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            decoded_len: u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
            row_count: u32::from_le_bytes(bytes[80..84].try_into().unwrap()),
            codec,
            crc32: u32::from_le_bytes(bytes[88..92].try_into().unwrap()),
            salted_xxh3: u64::from_le_bytes(bytes[92..100].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CatalogEntry {
    name: String,
    property_id: u32,
    index_offset: u64,
    page_count: u32,
    index_crc32: u32,
    index_xxh3: u64,
    entry_count: u64,
    min_node_id: [u8; 16],
    max_node_id: [u8; 16],
    min_ordinal: u64,
    max_ordinal: u64,
}

impl CatalogEntry {
    fn encode(&self, output: &mut Vec<u8>) -> Result<()> {
        let name_len = u16::try_from(self.name.len())
            .map_err(|_| Error::invariant("node property catalog name exceeds u16"))?;
        output.extend_from_slice(&name_len.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&self.property_id.to_le_bytes());
        output.extend_from_slice(&self.index_offset.to_le_bytes());
        output.extend_from_slice(&self.page_count.to_le_bytes());
        output.extend_from_slice(&self.index_crc32.to_le_bytes());
        output.extend_from_slice(&self.index_xxh3.to_le_bytes());
        output.extend_from_slice(&self.entry_count.to_le_bytes());
        output.extend_from_slice(&self.min_node_id);
        output.extend_from_slice(&self.max_node_id);
        output.extend_from_slice(&self.min_ordinal.to_le_bytes());
        output.extend_from_slice(&self.max_ordinal.to_le_bytes());
        output.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(output.len(), CATALOG_ENTRY_FIXED_LEN);
        output.extend_from_slice(self.name.as_bytes());
        Ok(())
    }

    fn decode(bytes: &[u8], cursor: &mut usize) -> Result<Self> {
        let fixed_end = cursor
            .checked_add(CATALOG_ENTRY_FIXED_LEN)
            .ok_or_else(|| corrupted("node property catalog cursor exceeds usize"))?;
        let fixed = bytes
            .get(*cursor..fixed_end)
            .ok_or_else(|| corrupted("node property catalog entry truncated"))?;
        let name_len = u16::from_le_bytes(fixed[..2].try_into().unwrap()) as usize;
        if fixed[2..4] != [0, 0] || fixed[88..92] != [0, 0, 0, 0] {
            return Err(corrupted("node property catalog reserved bytes invalid"));
        }
        let name_end = fixed_end
            .checked_add(name_len)
            .ok_or_else(|| corrupted("node property catalog name end exceeds usize"))?;
        let name = std::str::from_utf8(
            bytes
                .get(fixed_end..name_end)
                .ok_or_else(|| corrupted("node property catalog name truncated"))?,
        )
        .map_err(|error| corrupted(format!("node property name is not UTF-8: {error}")))?
        .to_string();
        *cursor = name_end;
        Ok(Self {
            name,
            property_id: u32::from_le_bytes(fixed[4..8].try_into().unwrap()),
            index_offset: u64::from_le_bytes(fixed[8..16].try_into().unwrap()),
            page_count: u32::from_le_bytes(fixed[16..20].try_into().unwrap()),
            index_crc32: u32::from_le_bytes(fixed[20..24].try_into().unwrap()),
            index_xxh3: u64::from_le_bytes(fixed[24..32].try_into().unwrap()),
            entry_count: u64::from_le_bytes(fixed[32..40].try_into().unwrap()),
            min_node_id: fixed[40..56].try_into().unwrap(),
            max_node_id: fixed[56..72].try_into().unwrap(),
            min_ordinal: u64::from_le_bytes(fixed[72..80].try_into().unwrap()),
            max_ordinal: u64::from_le_bytes(fixed[80..88].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Clone)]
struct Header {
    flags: u32,
    sst_id: [u8; 16],
    node_count: u64,
    cell_count: u64,
    property_count: u32,
    page_count: u32,
    catalog_offset: u64,
    catalog_len: u64,
    index_offset: u64,
    index_len: u64,
    payload_offset: u64,
    payload_len: u64,
    footer_offset: u64,
    footer_len: u64,
    catalog_crc32: u32,
    index_crc32: u32,
    payload_crc32: u32,
    catalog_xxh3: u64,
    index_xxh3: u64,
    payload_xxh3: u64,
}

impl Header {
    fn encode(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(HEADER_LEN);
        output.extend_from_slice(&HEADER_MAGIC);
        output.extend_from_slice(&NODE_PROPERTY_PAGES_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        output.extend_from_slice(&self.flags.to_le_bytes());
        output.extend_from_slice(&self.sst_id);
        output.extend_from_slice(&self.node_count.to_le_bytes());
        output.extend_from_slice(&self.cell_count.to_le_bytes());
        output.extend_from_slice(&self.property_count.to_le_bytes());
        output.extend_from_slice(&self.page_count.to_le_bytes());
        for value in [
            self.catalog_offset,
            self.catalog_len,
            self.index_offset,
            self.index_len,
            self.payload_offset,
            self.payload_len,
            self.footer_offset,
            self.footer_len,
        ] {
            output.extend_from_slice(&value.to_le_bytes());
        }
        output.extend_from_slice(&self.catalog_crc32.to_le_bytes());
        output.extend_from_slice(&self.index_crc32.to_le_bytes());
        output.extend_from_slice(&self.payload_crc32.to_le_bytes());
        output.extend_from_slice(&0u32.to_le_bytes());
        output.extend_from_slice(&self.catalog_xxh3.to_le_bytes());
        output.extend_from_slice(&self.index_xxh3.to_le_bytes());
        output.extend_from_slice(&self.payload_xxh3.to_le_bytes());
        output.resize(HEADER_LEN - 4, 0);
        let crc = crc32fast::hash(&output);
        output.extend_from_slice(&crc.to_le_bytes());
        debug_assert_eq!(output.len(), HEADER_LEN);
        output
    }

    fn decode(bytes: &[u8]) -> Result<(Self, u32)> {
        if bytes.len() != HEADER_LEN {
            return Err(corrupted("node property header length mismatch"));
        }
        if bytes[..8] != HEADER_MAGIC {
            return Err(corrupted("node property header magic mismatch"));
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        let header_len = u16::from_le_bytes(bytes[10..12].try_into().unwrap()) as usize;
        if version != NODE_PROPERTY_PAGES_FORMAT_VERSION || header_len != HEADER_LEN {
            return Err(corrupted(format!(
                "unsupported node property header version={version} len={header_len}"
            )));
        }
        let expected_crc = u32::from_le_bytes(bytes[HEADER_LEN - 4..].try_into().unwrap());
        if crc32fast::hash(&bytes[..HEADER_LEN - 4]) != expected_crc {
            return Err(corrupted("node property header CRC mismatch"));
        }
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if flags & !FLAG_COMPLETE != 0
            || bytes[132..136].iter().any(|byte| *byte != 0)
            || bytes[160..HEADER_LEN - 4].iter().any(|byte| *byte != 0)
        {
            return Err(corrupted("node property header reserved bytes non-zero"));
        }
        Ok((
            Self {
                flags,
                sst_id: bytes[16..32].try_into().unwrap(),
                node_count: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
                cell_count: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
                property_count: u32::from_le_bytes(bytes[48..52].try_into().unwrap()),
                page_count: u32::from_le_bytes(bytes[52..56].try_into().unwrap()),
                catalog_offset: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
                catalog_len: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
                index_offset: u64::from_le_bytes(bytes[72..80].try_into().unwrap()),
                index_len: u64::from_le_bytes(bytes[80..88].try_into().unwrap()),
                payload_offset: u64::from_le_bytes(bytes[88..96].try_into().unwrap()),
                payload_len: u64::from_le_bytes(bytes[96..104].try_into().unwrap()),
                footer_offset: u64::from_le_bytes(bytes[104..112].try_into().unwrap()),
                footer_len: u64::from_le_bytes(bytes[112..120].try_into().unwrap()),
                catalog_crc32: u32::from_le_bytes(bytes[120..124].try_into().unwrap()),
                index_crc32: u32::from_le_bytes(bytes[124..128].try_into().unwrap()),
                payload_crc32: u32::from_le_bytes(bytes[128..132].try_into().unwrap()),
                catalog_xxh3: u64::from_le_bytes(bytes[136..144].try_into().unwrap()),
                index_xxh3: u64::from_le_bytes(bytes[144..152].try_into().unwrap()),
                payload_xxh3: u64::from_le_bytes(bytes[152..160].try_into().unwrap()),
            },
            expected_crc,
        ))
    }
}

#[derive(Debug, Clone)]
struct Footer {
    sst_id: [u8; 16],
    header_crc32: u32,
    catalog_crc32: u32,
    index_crc32: u32,
    payload_crc32: u32,
    aggregate_xxh3: u64,
    object_len: u64,
    property_count: u32,
    page_count: u32,
    node_count: u64,
}

impl Footer {
    fn encode(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(FOOTER_LEN);
        output.extend_from_slice(&FOOTER_MAGIC);
        output.extend_from_slice(&self.sst_id);
        output.extend_from_slice(&self.header_crc32.to_le_bytes());
        output.extend_from_slice(&self.catalog_crc32.to_le_bytes());
        output.extend_from_slice(&self.index_crc32.to_le_bytes());
        output.extend_from_slice(&self.payload_crc32.to_le_bytes());
        output.extend_from_slice(&self.aggregate_xxh3.to_le_bytes());
        output.extend_from_slice(&self.object_len.to_le_bytes());
        output.extend_from_slice(&self.property_count.to_le_bytes());
        output.extend_from_slice(&self.page_count.to_le_bytes());
        output.extend_from_slice(&self.node_count.to_le_bytes());
        output.extend_from_slice(&0u32.to_le_bytes());
        let crc = crc32fast::hash(&output);
        output.extend_from_slice(&crc.to_le_bytes());
        debug_assert_eq!(output.len(), FOOTER_LEN);
        output
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FOOTER_LEN || bytes[..8] != FOOTER_MAGIC {
            return Err(corrupted("node property footer magic/length mismatch"));
        }
        if bytes[72..76] != [0, 0, 0, 0] {
            return Err(corrupted("node property footer reserved bytes non-zero"));
        }
        let expected_crc = u32::from_le_bytes(bytes[76..80].try_into().unwrap());
        if crc32fast::hash(&bytes[..76]) != expected_crc {
            return Err(corrupted("node property footer CRC mismatch"));
        }
        Ok(Self {
            sst_id: bytes[8..24].try_into().unwrap(),
            header_crc32: u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            catalog_crc32: u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            index_crc32: u32::from_le_bytes(bytes[32..36].try_into().unwrap()),
            payload_crc32: u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
            aggregate_xxh3: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            object_len: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            property_count: u32::from_le_bytes(bytes[56..60].try_into().unwrap()),
            page_count: u32::from_le_bytes(bytes[60..64].try_into().unwrap()),
            node_count: u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NodePropertyBuildStats {
    pub node_count: u64,
    pub cell_count: u64,
    pub property_count: u32,
    pub page_count: u32,
    /// Canonical bytes streamed through the temporary value spool. This may
    /// be arbitrarily larger than the retained sort workspace.
    pub value_spool_bytes: u64,
    pub peak_sort_memory_bytes: usize,
    pub peak_page_memory_bytes: usize,
    pub size_bytes: u64,
}

/// Scatter/gather upload product. Files are in exact wire order and anonymous,
/// so cancellation/drop closes and removes all scratch state.
#[derive(Debug)]
pub struct NodePropertyPageUpload {
    files: Vec<File>,
    size_bytes: u64,
    stats: NodePropertyBuildStats,
    sst_id: Uuid,
    content_xxh3: u64,
}

impl NodePropertyPageUpload {
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn stats(&self) -> NodePropertyBuildStats {
        self.stats
    }

    pub fn sst_id(&self) -> Uuid {
        self.sst_id
    }

    pub fn content_xxh3(&self) -> u64 {
        self.content_xxh3
    }

    pub fn into_files(self) -> (Vec<File>, u64) {
        (self.files, self.size_bytes)
    }
}

/// Schemaless node-property builder. `push_sorted` never opens one spool per
/// property; all cells share one bounded external sorter.
#[derive(Debug)]
pub struct NodePropertyPageBuilder {
    config: NodePropertyPageConfig,
    sst_id: Uuid,
    sorter: CellSorter,
    last_node_id: Option<[u8; 16]>,
    last_ordinal: Option<u64>,
    node_count: u64,
    cell_count: u64,
}

impl NodePropertyPageBuilder {
    pub fn new(config: NodePropertyPageConfig) -> Result<Self> {
        Self::new_bound(config, Uuid::now_v7())
    }

    /// Construct pages whose wire UUID is the owning Nodes SST UUID.
    pub fn new_bound(config: NodePropertyPageConfig, sst_id: Uuid) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            sorter: CellSorter::new(config.sort_memory_bytes),
            config,
            sst_id,
            last_node_id: None,
            last_ordinal: None,
            node_count: 0,
            cell_count: 0,
        })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(NodePropertyPageConfig::from_env()?)
    }

    pub fn sst_id(&self) -> Uuid {
        self.sst_id
    }

    /// Append one complete node property map. NodeIds and physical ordinals
    /// must both be strictly ascending.
    pub fn push_sorted(
        &mut self,
        node_id: [u8; 16],
        ordinal: u64,
        properties: &BTreeMap<String, Value>,
    ) -> Result<()> {
        if self
            .last_node_id
            .is_some_and(|previous| node_id <= previous)
        {
            return Err(Error::precondition(
                "node property rows must be strictly NodeId-sorted",
            ));
        }
        if self
            .last_ordinal
            .is_some_and(|previous| ordinal <= previous)
        {
            return Err(Error::precondition(
                "node property ordinals must be strictly ascending",
            ));
        }
        for (property, value) in properties {
            if property.is_empty() || property.len() > self.config.max_property_name_bytes {
                return Err(Error::precondition(format!(
                    "node property name requires {} bytes; allowed range is 1..={}",
                    property.len(),
                    self.config.max_property_name_bytes
                )));
            }
            let (state, extent) = if matches!(value, Value::Null) {
                (0, self.sorter.null_extent())
            } else {
                (1, self.sorter.append_value(value, &self.config)?)
            };
            self.sorter.push(SortCell {
                property: property.as_bytes().to_vec(),
                node_id,
                ordinal,
                state,
                value: extent,
            })?;
            self.cell_count = self
                .cell_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("node property cell count exceeds u64"))?;
        }
        self.node_count = self
            .node_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("node property node count exceeds u64"))?;
        self.last_node_id = Some(node_id);
        self.last_ordinal = Some(ordinal);
        Ok(())
    }

    pub fn finish(self) -> Result<NodePropertyPageUpload> {
        let NodePropertyPageBuilder {
            config,
            sst_id,
            sorter,
            node_count,
            cell_count,
            ..
        } = self;
        let (mut cells, mut values, peak_sort_memory_bytes) = sorter.finish()?;
        let value_spool_bytes = values.len;
        let mut catalog = DiskSpool::default();
        let mut index = DiskSpool::default();
        let mut payload = DiskSpool::default();
        let mut payload_crc = crc32fast::Hasher::new();
        let mut payload_hash = Xxh3::new();
        let mut property_count = 0u32;
        let mut page_count = 0u32;
        let mut copy_buffer = vec![0u8; COPY_BUFFER_BYTES];
        let mut peak_page_memory_bytes = copy_buffer.len();
        let mut current: Option<PropertyBuildState> = None;

        while let Some(cell) = cells.next_cell()? {
            let changed = current
                .as_ref()
                .is_some_and(|property| property.name.as_bytes() != cell.property);
            if changed {
                let finished = current.take().expect("property exists when changed");
                finish_property(
                    finished,
                    &config,
                    *sst_id.as_bytes(),
                    &mut catalog,
                    &mut index,
                    &mut payload,
                    &mut payload_crc,
                    &mut payload_hash,
                    &mut page_count,
                    &mut copy_buffer,
                    &mut peak_page_memory_bytes,
                )?;
            }
            if current.is_none() {
                property_count = property_count
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("node property count exceeds u32"))?;
                if property_count > config.max_properties {
                    return Err(Error::precondition(format!(
                        "node property catalog exceeds NAMIDB_NODE_PROPERTY_MAX_PROPERTIES={}",
                        config.max_properties
                    )));
                }
                let name = String::from_utf8(cell.property.clone())
                    .map_err(|error| corrupted(format!("sorted property is not UTF-8: {error}")))?;
                current = Some(PropertyBuildState::new(name, property_count - 1, index.len));
            }
            current.as_mut().expect("property installed").push(
                cell,
                &config,
                &mut values,
                *sst_id.as_bytes(),
                &mut index,
                &mut payload,
                &mut payload_crc,
                &mut payload_hash,
                &mut page_count,
                &mut copy_buffer,
                &mut peak_page_memory_bytes,
            )?;
        }
        if let Some(finished) = current {
            finish_property(
                finished,
                &config,
                *sst_id.as_bytes(),
                &mut catalog,
                &mut index,
                &mut payload,
                &mut payload_crc,
                &mut payload_hash,
                &mut page_count,
                &mut copy_buffer,
                &mut peak_page_memory_bytes,
            )?;
        }

        if catalog.len > config.max_catalog_bytes as u64 {
            return Err(Error::precondition(format!(
                "node property catalog requires {} bytes, above \
                 NAMIDB_NODE_PROPERTY_MAX_CATALOG_BYTES={}",
                catalog.len, config.max_catalog_bytes
            )));
        }
        let (catalog_crc32, catalog_xxh3) = spool_checksums(&mut catalog)?;
        let (index_crc32, index_xxh3) = spool_checksums(&mut index)?;
        let payload_crc32 = payload_crc.finalize();
        let payload_xxh3 = payload_hash.digest();

        let catalog_offset = HEADER_LEN as u64;
        let index_offset = catalog_offset
            .checked_add(catalog.len)
            .ok_or_else(|| Error::invariant("node property index offset exceeds u64"))?;
        let payload_offset = index_offset
            .checked_add(index.len)
            .ok_or_else(|| Error::invariant("node property payload offset exceeds u64"))?;
        let footer_offset = payload_offset
            .checked_add(payload.len)
            .ok_or_else(|| Error::invariant("node property footer offset exceeds u64"))?;
        let object_len = footer_offset
            .checked_add(FOOTER_LEN as u64)
            .ok_or_else(|| Error::invariant("node property object length exceeds u64"))?;
        let header = Header {
            flags: FLAG_COMPLETE,
            sst_id: *sst_id.as_bytes(),
            node_count,
            cell_count,
            property_count,
            page_count,
            catalog_offset,
            catalog_len: catalog.len,
            index_offset,
            index_len: index.len,
            payload_offset,
            payload_len: payload.len,
            footer_offset,
            footer_len: FOOTER_LEN as u64,
            catalog_crc32,
            index_crc32,
            payload_crc32,
            catalog_xxh3,
            index_xxh3,
            payload_xxh3,
        };
        let header_bytes = header.encode();
        let header_crc32 = u32::from_le_bytes(header_bytes[HEADER_LEN - 4..].try_into().unwrap());
        let aggregate_xxh3 = aggregate_hash(&header_bytes, &header);
        let footer = Footer {
            sst_id: *sst_id.as_bytes(),
            header_crc32,
            catalog_crc32,
            index_crc32,
            payload_crc32,
            aggregate_xxh3,
            object_len,
            property_count,
            page_count,
            node_count,
        }
        .encode();

        let mut header_spool = DiskSpool::default();
        header_spool.write_all(&header_bytes)?;
        let mut footer_spool = DiskSpool::default();
        footer_spool.write_all(&footer)?;
        let (header_file, _) = header_spool.into_file()?;
        let (catalog_file, _) = catalog.into_file()?;
        let (index_file, _) = index.into_file()?;
        let (payload_file, _) = payload.into_file()?;
        let (footer_file, _) = footer_spool.into_file()?;
        let stats = NodePropertyBuildStats {
            node_count,
            cell_count,
            property_count,
            page_count,
            value_spool_bytes,
            peak_sort_memory_bytes,
            peak_page_memory_bytes,
            size_bytes: object_len,
        };
        Ok(NodePropertyPageUpload {
            files: vec![
                header_file,
                catalog_file,
                index_file,
                payload_file,
                footer_file,
            ],
            size_bytes: object_len,
            stats,
            sst_id,
            content_xxh3: aggregate_xxh3,
        })
    }
}

struct PropertyBuildState {
    name: String,
    property_id: u32,
    index_offset: u64,
    page_count: u32,
    entry_count: u64,
    min_node_id: Option<[u8; 16]>,
    max_node_id: Option<[u8; 16]>,
    min_ordinal: u64,
    max_ordinal: u64,
    index_crc: crc32fast::Hasher,
    index_hash: Xxh3,
    page: PageBuildState,
}

impl PropertyBuildState {
    fn new(name: String, property_id: u32, index_offset: u64) -> Self {
        Self {
            name,
            property_id,
            index_offset,
            page_count: 0,
            entry_count: 0,
            min_node_id: None,
            max_node_id: None,
            min_ordinal: u64::MAX,
            max_ordinal: 0,
            index_crc: crc32fast::Hasher::new(),
            index_hash: Xxh3::new(),
            page: PageBuildState::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        cell: SortCell,
        config: &NodePropertyPageConfig,
        values: &mut DiskSpool,
        sst_id: [u8; 16],
        index: &mut DiskSpool,
        payload: &mut DiskSpool,
        payload_crc: &mut crc32fast::Hasher,
        payload_hash: &mut Xxh3,
        total_page_count: &mut u32,
        copy_buffer: &mut [u8],
        peak_page_memory_bytes: &mut usize,
    ) -> Result<()> {
        let row_bytes = PAGE_ROW_FIXED_LEN
            .checked_add(cell.value.len as usize)
            .ok_or_else(|| Error::invariant("node property page row exceeds usize"))?;
        let mut projected = PAGE_HEADER_LEN
            .checked_add(self.page.rows.len as usize)
            .and_then(|bytes| bytes.checked_add(row_bytes))
            .ok_or_else(|| Error::invariant("node property page length exceeds usize"))?;
        if self.page.row_count > 0 && projected > config.page_target_bytes {
            flush_property_page(
                self,
                config,
                sst_id,
                index,
                payload,
                payload_crc,
                payload_hash,
                total_page_count,
                copy_buffer,
                peak_page_memory_bytes,
            )?;
            projected = PAGE_HEADER_LEN
                .checked_add(row_bytes)
                .ok_or_else(|| Error::invariant("node property page length exceeds usize"))?;
        }
        if projected > config.max_page_decoded_bytes {
            return Err(Error::precondition(format!(
                "one node property page requires {projected} bytes, above \
                 NAMIDB_NODE_PROPERTY_MAX_PAGE_BYTES={}",
                config.max_page_decoded_bytes
            )));
        }
        self.page.push(&cell, values, copy_buffer)?;
        *peak_page_memory_bytes = (*peak_page_memory_bytes).max(copy_buffer.len());
        self.entry_count = self
            .entry_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("node property entry count exceeds u64"))?;
        self.min_node_id.get_or_insert(cell.node_id);
        self.max_node_id = Some(cell.node_id);
        self.min_ordinal = self.min_ordinal.min(cell.ordinal);
        self.max_ordinal = self.max_ordinal.max(cell.ordinal);
        Ok(())
    }
}

#[derive(Debug, Default)]
struct PageBuildState {
    rows: DiskSpool,
    row_count: u32,
    first_node_id: Option<[u8; 16]>,
    last_node_id: Option<[u8; 16]>,
    first_ordinal: u64,
    last_ordinal: u64,
}

impl PageBuildState {
    fn push(
        &mut self,
        cell: &SortCell,
        values: &mut DiskSpool,
        copy_buffer: &mut [u8],
    ) -> Result<()> {
        if self.row_count == 0 {
            self.first_node_id = Some(cell.node_id);
            self.first_ordinal = cell.ordinal;
        }
        self.last_node_id = Some(cell.node_id);
        self.last_ordinal = cell.ordinal;
        self.rows.write_all(&cell.node_id)?;
        self.rows.write_all(&cell.ordinal.to_le_bytes())?;
        self.rows.write_all(&[cell.state, 0, 0, 0])?;
        self.rows.write_all(&cell.value.len.to_le_bytes())?;
        copy_value_extent(values, &mut self.rows, cell.state, cell.value, copy_buffer)?;
        self.row_count = self
            .row_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("node property page rows exceed u32"))?;
        Ok(())
    }
}

fn copy_value_extent(
    source: &mut DiskSpool,
    destination: &mut DiskSpool,
    state: u8,
    extent: ValueExtent,
    buffer: &mut [u8],
) -> Result<()> {
    if state == 0 {
        if extent.len != 0 || extent.crc32 != 0 || extent.offset > source.len {
            return Err(corrupted("Null value extent is invalid"));
        }
        return Ok(());
    }
    if state != 1 || extent.len == 0 || buffer.is_empty() {
        return Err(corrupted("non-null value extent is invalid"));
    }
    let end = extent
        .offset
        .checked_add(u64::from(extent.len))
        .ok_or_else(|| corrupted("node property value extent end overflows"))?;
    if end > source.len {
        return Err(corrupted(
            "node property value extent exceeds its canonical spool",
        ));
    }
    let reader = source
        .file
        .as_mut()
        .ok_or_else(|| corrupted("node property canonical value spool disappeared"))?;
    reader.seek(SeekFrom::Start(extent.offset))?;
    let mut remaining = u64::from(extent.len);
    let mut crc = crc32fast::Hasher::new();
    while remaining > 0 {
        let take = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("node property value copy buffer fits usize");
        reader.read_exact(&mut buffer[..take])?;
        crc.update(&buffer[..take]);
        destination.write_all(&buffer[..take])?;
        remaining -= take as u64;
    }
    if crc.finalize() != extent.crc32 {
        return Err(corrupted(
            "node property canonical value extent CRC mismatch",
        ));
    }
    Ok(())
}

/// Ranged reader for the standalone node-property sidecar. Opening retains
/// only the bounded sparse catalog; page indexes and payloads stay in the
/// immutable object store until a named property is requested.
#[derive(Debug, Clone)]
pub struct NodePropertyPageReader {
    source: Arc<PinnedObjectRangeSource>,
    config: NodePropertyPageConfig,
    header: Header,
    catalog: Vec<CatalogEntry>,
}

impl NodePropertyPageReader {
    pub async fn open(
        source: Arc<PinnedObjectRangeSource>,
        expected_sst_id: Uuid,
        config: NodePropertyPageConfig,
    ) -> Result<Self> {
        config.validate()?;
        let object_len = source.object_size();
        let minimum_len = (HEADER_LEN + FOOTER_LEN) as u64;
        if object_len < minimum_len {
            return Err(corrupted(format!(
                "node property object has {object_len} bytes, below {minimum_len}"
            )));
        }
        let header_bytes = source.read_range(0..HEADER_LEN as u64).await?;
        let (header, header_crc32) = Header::decode(&header_bytes)?;
        if header.sst_id != *expected_sst_id.as_bytes() {
            return Err(corrupted(format!(
                "node property UUID {} does not match expected {expected_sst_id}",
                Uuid::from_bytes(header.sst_id)
            )));
        }
        validate_header_geometry(&header, object_len, &config)?;

        let footer_bytes = source.read_range(header.footer_offset..object_len).await?;
        let footer = Footer::decode(&footer_bytes)?;
        validate_footer_binding(&footer, &header, &header_bytes, header_crc32, object_len)?;

        let catalog_bytes = if header.catalog_len == 0 {
            Bytes::new()
        } else {
            source
                .read_range(
                    header.catalog_offset
                        ..header
                            .catalog_offset
                            .checked_add(header.catalog_len)
                            .ok_or_else(|| corrupted("node property catalog range overflows"))?,
                )
                .await?
        };
        if crc32fast::hash(&catalog_bytes) != header.catalog_crc32
            || xxh3_64(&catalog_bytes) != header.catalog_xxh3
        {
            return Err(corrupted("node property catalog checksum mismatch"));
        }
        let catalog = decode_catalog(&catalog_bytes, &header, &config)?;
        Ok(Self {
            source,
            config,
            header,
            catalog,
        })
    }

    pub fn sst_id(&self) -> Uuid {
        Uuid::from_bytes(self.header.sst_id)
    }

    pub fn node_count(&self) -> u64 {
        self.header.node_count
    }

    pub fn property_count(&self) -> u32 {
        self.header.property_count
    }

    pub fn cell_count(&self) -> u64 {
        self.header.cell_count
    }

    pub fn page_count(&self) -> u32 {
        self.header.page_count
    }

    pub fn content_xxh3(&self) -> u64 {
        aggregate_hash(&self.header.encode(), &self.header)
    }

    pub fn is_complete(&self) -> bool {
        self.header.flags & FLAG_COMPLETE != 0
    }

    pub fn resident_metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(
            self.catalog
                .iter()
                .map(|entry| {
                    entry
                        .name
                        .capacity()
                        .saturating_add(std::mem::size_of::<CatalogEntry>())
                })
                .sum::<usize>(),
        )
    }

    /// Cumulative logical range-I/O issued by this reader. Exposed for
    /// production telemetry and projection-regression tests; it never reveals
    /// cached bytes or retains payloads.
    pub fn range_stats(&self) -> PinnedObjectRangeStats {
        self.source.stats()
    }

    /// Project named properties for an explicit NodeId batch. Input order and
    /// duplicates are preserved. Unknown/absent properties are explicit
    /// [`PropertyCell::Absent`] values.
    pub async fn project_node_ids(
        &self,
        properties: &[String],
        node_ids: &[[u8; 16]],
    ) -> Result<(Vec<NodePropertyProjection>, NodePropertyProbeStats)> {
        if node_ids.len() > self.config.max_query_rows {
            return Err(Error::precondition(format!(
                "node property batch has {} rows, above \
                 NAMIDB_NODE_PROPERTY_MAX_QUERY_ROWS={}",
                node_ids.len(),
                self.config.max_query_rows
            )));
        }
        validate_projection_names(properties, &self.config)?;
        let before = self.source.stats();
        let unique_properties = properties.iter().cloned().collect::<BTreeSet<_>>();
        let mut projections = node_ids
            .iter()
            .map(|node_id| NodePropertyProjection {
                node_id: *node_id,
                ordinal: None,
                properties: unique_properties
                    .iter()
                    .cloned()
                    .map(|property| (property, PropertyCell::Absent))
                    .collect(),
            })
            .collect::<Vec<_>>();
        if projections.is_empty() || unique_properties.is_empty() {
            return Ok((
                projections,
                NodePropertyProbeStats::between(before, self.source.stats(), 0, 0),
            ));
        }
        let mut positions = BTreeMap::<[u8; 16], Vec<usize>>::new();
        for (position, node_id) in node_ids.iter().copied().enumerate() {
            positions.entry(node_id).or_default().push(position);
        }
        let wanted = positions.keys().copied().collect::<BTreeSet<_>>();
        let mut payload_pages = 0u64;
        let mut payload_bytes = 0u64;
        for property_name in unique_properties {
            let Some(catalog) = self.catalog_entry(&property_name) else {
                continue;
            };
            let pages = self.load_property_index(catalog).await?;
            let selected_pages = pages
                .iter()
                .filter(|page| {
                    wanted
                        .range(page.first_node_id..=page.last_node_id)
                        .next()
                        .is_some()
                })
                .collect::<Vec<_>>();
            let mut cursor = 0;
            while cursor < selected_pages.len() {
                let end = next_payload_batch_end(&selected_pages, cursor, &self.config);
                let encoded_pages = self
                    .read_payload_pages(&selected_pages[cursor..end])
                    .await?;
                for (page, encoded) in selected_pages[cursor..end]
                    .iter()
                    .copied()
                    .zip(encoded_pages)
                {
                    payload_pages += 1;
                    payload_bytes = payload_bytes
                        .checked_add(page.encoded_len)
                        .ok_or_else(|| Error::invariant("node property probe bytes exceed u64"))?;
                    let rows = decode_page_selected(
                        &encoded,
                        page,
                        self.header.sst_id,
                        &self.config,
                        |node_id, _| wanted.contains(&node_id),
                    )?;
                    for (node_id, ordinal, cell) in rows {
                        let row_positions = positions.get(&node_id).ok_or_else(|| {
                            Error::invariant("node property page returned an unrequested NodeId")
                        })?;
                        for position in row_positions {
                            let projection = &mut projections[*position];
                            if let Some(previous) = projection.ordinal {
                                if previous != ordinal {
                                    return Err(corrupted(
                                        "node property ordinal differs across projected properties",
                                    ));
                                }
                            } else {
                                projection.ordinal = Some(ordinal);
                            }
                            let slot =
                                projection
                                    .properties
                                    .get_mut(&property_name)
                                    .ok_or_else(|| {
                                        Error::invariant(
                                            "node property projection slot disappeared",
                                        )
                                    })?;
                            if !matches!(slot, PropertyCell::Absent) {
                                return Err(corrupted(
                                    "node property page contains duplicate NodeId/property rows",
                                ));
                            }
                            *slot = cell.clone();
                        }
                    }
                }
                cursor = end;
            }
        }
        Ok((
            projections,
            NodePropertyProbeStats::between(
                before,
                self.source.stats(),
                payload_pages,
                payload_bytes,
            ),
        ))
    }

    /// Sparse range projection. It returns nodes that have at least one of the
    /// requested properties in the half-open physical ordinal range. Callers
    /// that need rows whose entire projection is absent must supply NodeIds via
    /// [`Self::project_node_ids`].
    pub async fn project_ordinal_range(
        &self,
        properties: &[String],
        ordinals: Range<u64>,
    ) -> Result<(Vec<NodePropertyProjection>, NodePropertyProbeStats)> {
        if ordinals.start > ordinals.end {
            return Err(Error::precondition(
                "node property ordinal range starts after its end",
            ));
        }
        validate_projection_names(properties, &self.config)?;
        let before = self.source.stats();
        if ordinals.is_empty() || properties.is_empty() {
            return Ok((
                Vec::new(),
                NodePropertyProbeStats::between(before, self.source.stats(), 0, 0),
            ));
        }
        let unique_properties = properties.iter().cloned().collect::<BTreeSet<_>>();
        let mut projections = BTreeMap::<[u8; 16], NodePropertyProjection>::new();
        let mut payload_pages = 0u64;
        let mut payload_bytes = 0u64;
        for property_name in &unique_properties {
            let Some(catalog) = self.catalog_entry(property_name) else {
                continue;
            };
            if catalog.max_ordinal < ordinals.start || catalog.min_ordinal >= ordinals.end {
                continue;
            }
            let pages = self.load_property_index(catalog).await?;
            let selected_pages = pages
                .iter()
                .filter(|page| {
                    page.last_ordinal >= ordinals.start && page.first_ordinal < ordinals.end
                })
                .collect::<Vec<_>>();
            let mut cursor = 0;
            while cursor < selected_pages.len() {
                let end = next_payload_batch_end(&selected_pages, cursor, &self.config);
                let encoded_pages = self
                    .read_payload_pages(&selected_pages[cursor..end])
                    .await?;
                for (page, encoded) in selected_pages[cursor..end]
                    .iter()
                    .copied()
                    .zip(encoded_pages)
                {
                    payload_pages += 1;
                    payload_bytes = payload_bytes
                        .checked_add(page.encoded_len)
                        .ok_or_else(|| Error::invariant("node property probe bytes exceed u64"))?;
                    let rows = decode_page_selected(
                        &encoded,
                        page,
                        self.header.sst_id,
                        &self.config,
                        |_, ordinal| ordinals.contains(&ordinal),
                    )?;
                    for (node_id, ordinal, cell) in rows {
                        if !projections.contains_key(&node_id)
                            && projections.len() >= self.config.max_query_rows
                        {
                            return Err(Error::precondition(format!(
                                "node property range result exceeds \
                                 NAMIDB_NODE_PROPERTY_MAX_QUERY_ROWS={}",
                                self.config.max_query_rows
                            )));
                        }
                        let projection =
                            projections
                                .entry(node_id)
                                .or_insert_with(|| NodePropertyProjection {
                                    node_id,
                                    ordinal: Some(ordinal),
                                    properties: unique_properties
                                        .iter()
                                        .cloned()
                                        .map(|property| (property, PropertyCell::Absent))
                                        .collect(),
                                });
                        if projection.ordinal != Some(ordinal) {
                            return Err(corrupted(
                                "node property ordinal differs across range properties",
                            ));
                        }
                        let slot =
                            projection
                                .properties
                                .get_mut(property_name)
                                .ok_or_else(|| {
                                    Error::invariant(
                                        "node property range projection slot disappeared",
                                    )
                                })?;
                        if !matches!(slot, PropertyCell::Absent) {
                            return Err(corrupted(
                                "node property page contains duplicate range rows",
                            ));
                        }
                        *slot = cell;
                    }
                }
                cursor = end;
            }
        }
        let mut projections = projections.into_values().collect::<Vec<_>>();
        projections
            .sort_by_key(|projection| (projection.ordinal.unwrap_or(u64::MAX), projection.node_id));
        Ok((
            projections,
            NodePropertyProbeStats::between(
                before,
                self.source.stats(),
                payload_pages,
                payload_bytes,
            ),
        ))
    }

    /// Evaluate only exact String/Bool equality or IN predicates. Any unknown
    /// predicate or incomplete sidecar returns `Unsupported`, never a partial
    /// match subset.
    pub async fn filter_node_ids(
        &self,
        predicates: &[NodePropertyPredicate],
        projection: &[String],
        node_ids: &[[u8; 16]],
    ) -> Result<(NodePropertyFilterResult, NodePropertyProbeStats)> {
        let before = self.source.stats();
        if !self.is_complete() {
            return Ok((
                NodePropertyFilterResult::Unsupported {
                    reason: "node property sidecar is not marked complete".into(),
                },
                NodePropertyProbeStats::between(before, self.source.stats(), 0, 0),
            ));
        }
        if let Some(reason) = predicates.iter().find_map(|predicate| match predicate {
            NodePropertyPredicate::Unsupported { description } => Some(description.clone()),
            _ => None,
        }) {
            return Ok((
                NodePropertyFilterResult::Unsupported { reason },
                NodePropertyProbeStats::between(before, self.source.stats(), 0, 0),
            ));
        }
        validate_predicates(predicates, &self.config)?;
        validate_projection_names(projection, &self.config)?;
        let mut loaded_properties = projection.iter().cloned().collect::<BTreeSet<_>>();
        loaded_properties.extend(
            predicates
                .iter()
                .filter_map(predicate_property)
                .map(str::to_owned),
        );
        let loaded_properties = loaded_properties.into_iter().collect::<Vec<_>>();
        let (rows, stats) = self.project_node_ids(&loaded_properties, node_ids).await?;
        let projected = projection.iter().cloned().collect::<BTreeSet<_>>();
        let mut matches = Vec::new();
        for mut row in rows {
            if predicates
                .iter()
                .all(|predicate| predicate_matches(predicate, &row.properties))
            {
                row.properties
                    .retain(|property, _| projected.contains(property));
                matches.push(row);
            }
        }
        Ok((NodePropertyFilterResult::Matches(matches), stats))
    }

    /// Validate every index and value page belonging to the requested
    /// properties without retaining rows. Unknown properties are valid empty
    /// projections. This is the preflight used by a streaming visitor: after
    /// it succeeds, an optional-sidecar corruption cannot be discovered only
    /// after visitor-visible rows have already escaped.
    pub async fn verify_properties(&self, properties: &[String]) -> Result<NodePropertyProbeStats> {
        validate_projection_names(properties, &self.config)?;
        let before = self.source.stats();
        let mut payload_pages = 0u64;
        let mut payload_bytes = 0u64;
        for property_name in properties.iter().collect::<BTreeSet<_>>() {
            let Some(catalog) = self.catalog_entry(property_name) else {
                continue;
            };
            let pages = self.load_property_index(catalog).await?;
            let page_refs = pages.iter().collect::<Vec<_>>();
            let mut cursor = 0;
            while cursor < page_refs.len() {
                let end = next_payload_batch_end(&page_refs, cursor, &self.config);
                let encoded_pages = self.read_payload_pages(&page_refs[cursor..end]).await?;
                for (page, encoded) in page_refs[cursor..end].iter().zip(encoded_pages) {
                    let rows = decode_page_selected(
                        &encoded,
                        page,
                        self.header.sst_id,
                        &self.config,
                        |_, _| true,
                    )?;
                    if rows.len() != page.row_count as usize {
                        return Err(corrupted(
                            "node property projection scrub did not decode every row",
                        ));
                    }
                    payload_pages = payload_pages
                        .checked_add(1)
                        .ok_or_else(|| corrupted("node property scrub page count overflows"))?;
                    payload_bytes = payload_bytes
                        .checked_add(page.encoded_len)
                        .ok_or_else(|| corrupted("node property scrub bytes overflow"))?;
                }
                cursor = end;
            }
        }
        Ok(NodePropertyProbeStats::between(
            before,
            self.source.stats(),
            payload_pages,
            payload_bytes,
        ))
    }

    /// Bounded integrity scrub. Catalog, every per-property index, every page
    /// checksum, and every canonical Value are validated without retaining the
    /// corpus or more than one decoded page.
    pub async fn verify_all(&self) -> Result<NodePropertyProbeStats> {
        let before = self.source.stats();
        let mut index_crc = crc32fast::Hasher::new();
        let mut index_hash = Xxh3::new();
        let mut payload_crc = crc32fast::Hasher::new();
        let mut payload_hash = Xxh3::new();
        let mut expected_payload_offset = 0u64;
        let mut payload_pages = 0u64;
        let mut payload_bytes = 0u64;
        for catalog in &self.catalog {
            let pages = self.load_property_index(catalog).await?;
            for page in pages {
                if page.payload_offset != expected_payload_offset {
                    return Err(corrupted(
                        "node property payload pages are not globally contiguous",
                    ));
                }
                let mut raw_index = Vec::with_capacity(INDEX_ENTRY_LEN);
                page.encode(&mut raw_index);
                index_crc.update(&raw_index);
                index_hash.update(&raw_index);
                let encoded = self.read_payload_page(&page).await?;
                payload_crc.update(&encoded);
                payload_hash.update(&encoded);
                let rows = decode_page_selected(
                    &encoded,
                    &page,
                    self.header.sst_id,
                    &self.config,
                    |_, _| true,
                )?;
                if rows.len() != page.row_count as usize {
                    return Err(corrupted(
                        "node property integrity scrub did not decode every row",
                    ));
                }
                expected_payload_offset = expected_payload_offset
                    .checked_add(page.encoded_len)
                    .ok_or_else(|| corrupted("node property payload position overflows"))?;
                payload_pages += 1;
                payload_bytes = payload_bytes
                    .checked_add(page.encoded_len)
                    .ok_or_else(|| corrupted("node property scrub bytes overflow"))?;
            }
        }
        if expected_payload_offset != self.header.payload_len
            || index_crc.finalize() != self.header.index_crc32
            || index_hash.digest() != self.header.index_xxh3
            || payload_crc.finalize() != self.header.payload_crc32
            || payload_hash.digest() != self.header.payload_xxh3
        {
            return Err(corrupted(
                "node property full-section integrity checksum mismatch",
            ));
        }
        Ok(NodePropertyProbeStats::between(
            before,
            self.source.stats(),
            payload_pages,
            payload_bytes,
        ))
    }

    async fn load_property_index(&self, catalog: &CatalogEntry) -> Result<Vec<PageIndexEntry>> {
        let len = usize::try_from(catalog.page_count)
            .ok()
            .and_then(|count| count.checked_mul(INDEX_ENTRY_LEN))
            .ok_or_else(|| corrupted("node property page-index span exceeds usize"))?;
        if len > self.config.max_index_query_bytes {
            return Err(Error::precondition(format!(
                "property '{}' needs {len} page-index bytes, above \
                 NAMIDB_NODE_PROPERTY_MAX_INDEX_QUERY_BYTES={}",
                catalog.name, self.config.max_index_query_bytes
            )));
        }
        let relative_end = catalog
            .index_offset
            .checked_add(len as u64)
            .ok_or_else(|| corrupted("node property page-index range overflows"))?;
        if relative_end > self.header.index_len {
            return Err(corrupted(
                "node property page-index range exceeds index section",
            ));
        }
        let start = self
            .header
            .index_offset
            .checked_add(catalog.index_offset)
            .ok_or_else(|| corrupted("node property absolute index offset overflows"))?;
        let bytes = self.source.read_range(start..start + len as u64).await?;
        if crc32fast::hash(&bytes) != catalog.index_crc32 || xxh3_64(&bytes) != catalog.index_xxh3 {
            return Err(corrupted(format!(
                "node property page index for '{}' failed checksum",
                catalog.name
            )));
        }
        let mut pages = Vec::with_capacity(catalog.page_count as usize);
        let mut row_count = 0u64;
        for (page_ordinal, raw) in bytes.chunks_exact(INDEX_ENTRY_LEN).enumerate() {
            let page = PageIndexEntry::decode(raw)?;
            validate_page_index_entry(
                &page,
                catalog,
                page_ordinal,
                pages.last(),
                self.header.payload_len,
                &self.config,
            )?;
            row_count = row_count
                .checked_add(u64::from(page.row_count))
                .ok_or_else(|| corrupted("node property indexed row count exceeds u64"))?;
            pages.push(page);
        }
        if row_count != catalog.entry_count {
            return Err(corrupted(format!(
                "node property '{}' index has {row_count} rows, catalog says {}",
                catalog.name, catalog.entry_count
            )));
        }
        let first = pages
            .first()
            .ok_or_else(|| corrupted("node property catalog has no indexed page"))?;
        let last = pages
            .last()
            .ok_or_else(|| corrupted("node property catalog has no indexed page"))?;
        if first.first_node_id != catalog.min_node_id
            || last.last_node_id != catalog.max_node_id
            || first.first_ordinal != catalog.min_ordinal
            || last.last_ordinal != catalog.max_ordinal
        {
            return Err(corrupted(format!(
                "node property '{}' catalog range disagrees with its pages",
                catalog.name
            )));
        }
        Ok(pages)
    }

    async fn read_payload_page(&self, page: &PageIndexEntry) -> Result<Bytes> {
        let start = self
            .header
            .payload_offset
            .checked_add(page.payload_offset)
            .ok_or_else(|| corrupted("node property absolute payload offset overflows"))?;
        let end = start
            .checked_add(page.encoded_len)
            .ok_or_else(|| corrupted("node property payload range overflows"))?;
        self.source.read_range(start..end).await
    }

    async fn read_payload_pages(&self, pages: &[&PageIndexEntry]) -> Result<Vec<Bytes>> {
        let mut ranges = Vec::with_capacity(pages.len());
        for page in pages {
            let start = self
                .header
                .payload_offset
                .checked_add(page.payload_offset)
                .ok_or_else(|| corrupted("node property absolute payload offset overflows"))?;
            let end = start
                .checked_add(page.encoded_len)
                .ok_or_else(|| corrupted("node property payload range overflows"))?;
            ranges.push(start..end);
        }
        self.source.read_ranges(&ranges).await
    }

    fn catalog_entry(&self, property: &str) -> Option<&CatalogEntry> {
        self.catalog
            .binary_search_by(|entry| entry.name.as_str().cmp(property))
            .ok()
            .map(|index| &self.catalog[index])
    }
}

/// Keep parallel object-store I/O useful without letting one query retain an
/// unbounded vector of encoded pages. Decoding one page beside this batch is
/// bounded by roughly twice `max_page_decoded_bytes`.
fn next_payload_batch_end(
    pages: &[&PageIndexEntry],
    start: usize,
    config: &NodePropertyPageConfig,
) -> usize {
    let mut end = start;
    let mut encoded_bytes = 0u64;
    let budget = config.max_page_decoded_bytes as u64;
    while end < pages.len() && end - start < 16 {
        let next = pages[end].encoded_len;
        if end > start && encoded_bytes.saturating_add(next) > budget {
            break;
        }
        encoded_bytes = encoded_bytes.saturating_add(next);
        end += 1;
    }
    end.max(start + 1)
}

fn validate_header_geometry(
    header: &Header,
    object_len: u64,
    config: &NodePropertyPageConfig,
) -> Result<()> {
    if header.property_count > config.max_properties {
        return Err(corrupted(format!(
            "node property count {} exceeds configured ceiling {}",
            header.property_count, config.max_properties
        )));
    }
    if header.catalog_len > config.max_catalog_bytes as u64 {
        return Err(corrupted(format!(
            "node property catalog length {} exceeds configured ceiling {}",
            header.catalog_len, config.max_catalog_bytes
        )));
    }
    let expected_index_len = u64::from(header.page_count)
        .checked_mul(INDEX_ENTRY_LEN as u64)
        .ok_or_else(|| corrupted("node property index length exceeds u64"))?;
    if header.catalog_offset != HEADER_LEN as u64
        || header.index_len != expected_index_len
        || header.index_offset
            != header
                .catalog_offset
                .checked_add(header.catalog_len)
                .ok_or_else(|| corrupted("node property catalog end overflows"))?
        || header.payload_offset
            != header
                .index_offset
                .checked_add(header.index_len)
                .ok_or_else(|| corrupted("node property index end overflows"))?
        || header.footer_offset
            != header
                .payload_offset
                .checked_add(header.payload_len)
                .ok_or_else(|| corrupted("node property payload end overflows"))?
        || header.footer_len != FOOTER_LEN as u64
        || header
            .footer_offset
            .checked_add(header.footer_len)
            .ok_or_else(|| corrupted("node property footer end overflows"))?
            != object_len
    {
        return Err(corrupted(
            "node property sections are not exact and contiguous",
        ));
    }
    let empty = header.property_count == 0;
    if empty
        != (header.cell_count == 0
            && header.page_count == 0
            && header.catalog_len == 0
            && header.index_len == 0
            && header.payload_len == 0)
    {
        return Err(corrupted(
            "node property empty/non-empty section accounting disagrees",
        ));
    }
    if !empty && (header.cell_count == 0 || header.page_count == 0 || header.payload_len == 0) {
        return Err(corrupted(
            "non-empty node property object has an empty required section",
        ));
    }
    Ok(())
}

fn validate_footer_binding(
    footer: &Footer,
    header: &Header,
    header_bytes: &[u8],
    header_crc32: u32,
    object_len: u64,
) -> Result<()> {
    if footer.sst_id != header.sst_id
        || footer.header_crc32 != header_crc32
        || footer.catalog_crc32 != header.catalog_crc32
        || footer.index_crc32 != header.index_crc32
        || footer.payload_crc32 != header.payload_crc32
        || footer.aggregate_xxh3 != aggregate_hash(header_bytes, header)
        || footer.object_len != object_len
        || footer.property_count != header.property_count
        || footer.page_count != header.page_count
        || footer.node_count != header.node_count
    {
        return Err(corrupted(
            "node property footer does not bind the header and object",
        ));
    }
    Ok(())
}

fn decode_catalog(
    bytes: &[u8],
    header: &Header,
    config: &NodePropertyPageConfig,
) -> Result<Vec<CatalogEntry>> {
    let mut cursor = 0usize;
    let mut catalog = Vec::with_capacity(header.property_count as usize);
    let mut expected_index_offset = 0u64;
    let mut total_pages = 0u32;
    let mut total_entries = 0u64;
    let mut previous_name: Option<String> = None;
    for expected_property_id in 0..header.property_count {
        let entry = CatalogEntry::decode(bytes, &mut cursor)?;
        if entry.name.is_empty() || entry.name.len() > config.max_property_name_bytes {
            return Err(corrupted(format!(
                "node property catalog name has invalid {}-byte length",
                entry.name.len()
            )));
        }
        if previous_name
            .as_ref()
            .is_some_and(|previous| previous.as_bytes() >= entry.name.as_bytes())
            || entry.property_id != expected_property_id
            || entry.page_count == 0
            || entry.entry_count == 0
            || entry.min_node_id > entry.max_node_id
            || entry.min_ordinal > entry.max_ordinal
            || entry.index_offset != expected_index_offset
        {
            return Err(corrupted(
                "node property catalog order, identity, or range is invalid",
            ));
        }
        let index_len = u64::from(entry.page_count)
            .checked_mul(INDEX_ENTRY_LEN as u64)
            .ok_or_else(|| corrupted("node property catalog index span overflows"))?;
        expected_index_offset = expected_index_offset
            .checked_add(index_len)
            .ok_or_else(|| corrupted("node property catalog index end overflows"))?;
        if expected_index_offset > header.index_len {
            return Err(corrupted(
                "node property catalog index span exceeds index section",
            ));
        }
        total_pages = total_pages
            .checked_add(entry.page_count)
            .ok_or_else(|| corrupted("node property catalog page count overflows"))?;
        total_entries = total_entries
            .checked_add(entry.entry_count)
            .ok_or_else(|| corrupted("node property catalog entry count overflows"))?;
        previous_name = Some(entry.name.clone());
        catalog.push(entry);
    }
    if cursor != bytes.len()
        || expected_index_offset != header.index_len
        || total_pages != header.page_count
        || total_entries != header.cell_count
    {
        return Err(corrupted(
            "node property catalog totals or trailing bytes are invalid",
        ));
    }
    Ok(catalog)
}

fn validate_projection_names(properties: &[String], config: &NodePropertyPageConfig) -> Result<()> {
    if properties.len() > config.max_properties as usize {
        return Err(Error::precondition(format!(
            "node property projection has {} names, above {}",
            properties.len(),
            config.max_properties
        )));
    }
    if let Some(property) = properties
        .iter()
        .find(|property| property.is_empty() || property.len() > config.max_property_name_bytes)
    {
        return Err(Error::precondition(format!(
            "node property projection name has invalid {}-byte length",
            property.len()
        )));
    }
    Ok(())
}

fn predicate_property(predicate: &NodePropertyPredicate) -> Option<&str> {
    match predicate {
        NodePropertyPredicate::EqString { property, .. }
        | NodePropertyPredicate::InString { property, .. }
        | NodePropertyPredicate::EqBool { property, .. }
        | NodePropertyPredicate::InBool { property, .. } => Some(property),
        NodePropertyPredicate::Unsupported { .. } => None,
    }
}

fn validate_predicates(
    predicates: &[NodePropertyPredicate],
    config: &NodePropertyPageConfig,
) -> Result<()> {
    if predicates.len() > config.max_properties as usize {
        return Err(Error::precondition(format!(
            "node property filter has {} predicates, above {}",
            predicates.len(),
            config.max_properties
        )));
    }
    let mut alternatives = 0u64;
    for predicate in predicates {
        let Some(property) = predicate_property(predicate) else {
            continue;
        };
        if property.is_empty() || property.len() > config.max_property_name_bytes {
            return Err(Error::precondition(format!(
                "node property predicate name has invalid {}-byte length",
                property.len()
            )));
        }
        match predicate {
            NodePropertyPredicate::EqString { value, .. } => {
                alternatives += 1;
                if value.len() > config.max_value_bytes {
                    return Err(Error::precondition(
                        "node property predicate String exceeds value ceiling",
                    ));
                }
            }
            NodePropertyPredicate::InString { values, .. } => {
                alternatives = alternatives
                    .checked_add(values.len() as u64)
                    .ok_or_else(|| Error::precondition("filter alternatives exceed u64"))?;
                if values
                    .iter()
                    .any(|value| value.len() > config.max_value_bytes)
                {
                    return Err(Error::precondition(
                        "node property predicate String exceeds value ceiling",
                    ));
                }
            }
            NodePropertyPredicate::EqBool { .. } => alternatives += 1,
            NodePropertyPredicate::InBool { values, .. } => {
                alternatives = alternatives
                    .checked_add(values.len() as u64)
                    .ok_or_else(|| Error::precondition("filter alternatives exceed u64"))?;
            }
            NodePropertyPredicate::Unsupported { .. } => {}
        }
        if alternatives > config.max_value_elements {
            return Err(Error::precondition(format!(
                "node property filter alternatives exceed {}",
                config.max_value_elements
            )));
        }
    }
    Ok(())
}

fn predicate_matches(
    predicate: &NodePropertyPredicate,
    properties: &BTreeMap<String, PropertyCell>,
) -> bool {
    match predicate {
        NodePropertyPredicate::EqString { property, value } => matches!(
            properties.get(property),
            Some(PropertyCell::Value(Value::Str(candidate))) if candidate == value
        ),
        NodePropertyPredicate::InString { property, values } => matches!(
            properties.get(property),
            Some(PropertyCell::Value(Value::Str(candidate))) if values.contains(candidate)
        ),
        NodePropertyPredicate::EqBool { property, value } => matches!(
            properties.get(property),
            Some(PropertyCell::Value(Value::Bool(candidate))) if candidate == value
        ),
        NodePropertyPredicate::InBool { property, values } => matches!(
            properties.get(property),
            Some(PropertyCell::Value(Value::Bool(candidate))) if values.contains(candidate)
        ),
        NodePropertyPredicate::Unsupported { .. } => false,
    }
}

fn validate_page_index_entry(
    page: &PageIndexEntry,
    catalog: &CatalogEntry,
    expected_page_ordinal: usize,
    previous: Option<&PageIndexEntry>,
    payload_len: u64,
    config: &NodePropertyPageConfig,
) -> Result<()> {
    let expected_page_ordinal = u32::try_from(expected_page_ordinal)
        .map_err(|_| corrupted("node property page ordinal exceeds u32"))?;
    let minimum_decoded = (PAGE_HEADER_LEN as u64)
        .checked_add(
            u64::from(page.row_count)
                .checked_mul(PAGE_ROW_FIXED_LEN as u64)
                .ok_or_else(|| corrupted("node property minimum page length overflows"))?,
        )
        .ok_or_else(|| corrupted("node property minimum page length overflows"))?;
    let payload_end = page
        .payload_offset
        .checked_add(page.encoded_len)
        .ok_or_else(|| corrupted("node property page payload end overflows"))?;
    if page.property_id != catalog.property_id
        || page.page_ordinal != expected_page_ordinal
        || page.row_count == 0
        || page.first_node_id > page.last_node_id
        || page.first_ordinal > page.last_ordinal
        || page.encoded_len == 0
        || page.decoded_len < minimum_decoded
        || page.decoded_len > config.max_page_decoded_bytes as u64
        || page.encoded_len > config.max_page_decoded_bytes as u64
        || (page.codec == CODEC_RAW && page.encoded_len != page.decoded_len)
        || payload_end > payload_len
    {
        return Err(corrupted(format!(
            "node property page index entry {expected_page_ordinal} is invalid"
        )));
    }
    if let Some(previous) = previous {
        let previous_payload_end = previous
            .payload_offset
            .checked_add(previous.encoded_len)
            .ok_or_else(|| corrupted("node property previous payload end overflows"))?;
        if previous.last_node_id >= page.first_node_id
            || previous.last_ordinal >= page.first_ordinal
            || previous_payload_end != page.payload_offset
        {
            return Err(corrupted(
                "node property page ranges overlap or payload is not contiguous",
            ));
        }
    }
    Ok(())
}

fn decode_page_selected(
    encoded: &[u8],
    page: &PageIndexEntry,
    sst_id: [u8; 16],
    config: &NodePropertyPageConfig,
    mut selected: impl FnMut([u8; 16], u64) -> bool,
) -> Result<Vec<([u8; 16], u64, PropertyCell)>> {
    if encoded.len() as u64 != page.encoded_len
        || crc32fast::hash(encoded) != page.crc32
        || salted_page_hash(encoded, sst_id, page.property_id, page.page_ordinal)
            != page.salted_xxh3
    {
        return Err(corrupted("node property payload page checksum mismatch"));
    }
    let decoded_storage;
    let decoded = match page.codec {
        CODEC_RAW => encoded,
        CODEC_ZSTD => {
            let expected_len = usize::try_from(page.decoded_len)
                .map_err(|_| corrupted("node property decoded page exceeds usize"))?;
            let decoder =
                zstd::stream::read::Decoder::new(Cursor::new(encoded)).map_err(|error| {
                    corrupted(format!("node property zstd page init failed: {error}"))
                })?;
            let mut limited = decoder.take(page.decoded_len.saturating_add(1));
            let mut bytes = Vec::with_capacity(expected_len.min(config.page_target_bytes));
            limited.read_to_end(&mut bytes).map_err(|error| {
                corrupted(format!("node property zstd page decode failed: {error}"))
            })?;
            if bytes.len() != expected_len {
                return Err(corrupted(format!(
                    "node property page decoded to {} bytes, expected {expected_len}",
                    bytes.len()
                )));
            }
            decoded_storage = bytes;
            decoded_storage.as_slice()
        }
        _ => return Err(corrupted("node property page codec is unknown")),
    };
    if decoded.len() != page.decoded_len as usize
        || decoded.len() < PAGE_HEADER_LEN
        || decoded[..8] != PAGE_HEADER_MAGIC
        || decoded[12..16] != [0, 0, 0, 0]
    {
        return Err(corrupted("node property decoded page header is invalid"));
    }
    let row_count = u32::from_le_bytes(decoded[8..12].try_into().unwrap());
    if row_count != page.row_count {
        return Err(corrupted(
            "node property decoded page row count disagrees with index",
        ));
    }
    let mut cursor = PAGE_HEADER_LEN;
    let mut rows = Vec::new();
    let mut previous_node_id = None;
    let mut previous_ordinal = None;
    let mut first_node_id = None;
    let mut first_ordinal = None;
    let mut last_node_id = None;
    let mut last_ordinal = None;
    for _ in 0..row_count {
        let fixed_end = cursor
            .checked_add(PAGE_ROW_FIXED_LEN)
            .ok_or_else(|| corrupted("node property page row cursor overflows"))?;
        let fixed = decoded
            .get(cursor..fixed_end)
            .ok_or_else(|| corrupted("node property page row is truncated"))?;
        let node_id: [u8; 16] = fixed[..16].try_into().unwrap();
        let ordinal = u64::from_le_bytes(fixed[16..24].try_into().unwrap());
        let state = fixed[24];
        if fixed[25..28] != [0, 0, 0] || state > 1 {
            return Err(corrupted("node property page row flags are invalid"));
        }
        let value_len = u32::from_le_bytes(fixed[28..32].try_into().unwrap()) as usize;
        if value_len > config.max_value_bytes || (state == 0 && value_len != 0) {
            return Err(corrupted(
                "node property page row value length/state is invalid",
            ));
        }
        let value_end = fixed_end
            .checked_add(value_len)
            .ok_or_else(|| corrupted("node property page value end overflows"))?;
        let value = decoded
            .get(fixed_end..value_end)
            .ok_or_else(|| corrupted("node property page value is truncated"))?;
        if previous_node_id.is_some_and(|previous| previous >= node_id)
            || previous_ordinal.is_some_and(|previous| previous >= ordinal)
        {
            return Err(corrupted("node property page rows are not strictly sorted"));
        }
        first_node_id.get_or_insert(node_id);
        first_ordinal.get_or_insert(ordinal);
        last_node_id = Some(node_id);
        last_ordinal = Some(ordinal);
        if selected(node_id, ordinal) {
            let cell = if state == 0 {
                PropertyCell::Null
            } else {
                PropertyCell::Value(decode_top_level_value(value, config)?)
            };
            rows.push((node_id, ordinal, cell));
        }
        previous_node_id = Some(node_id);
        previous_ordinal = Some(ordinal);
        cursor = value_end;
    }
    if cursor != decoded.len()
        || first_node_id != Some(page.first_node_id)
        || last_node_id != Some(page.last_node_id)
        || first_ordinal != Some(page.first_ordinal)
        || last_ordinal != Some(page.last_ordinal)
    {
        return Err(corrupted(
            "node property page boundary or trailing-byte mismatch",
        ));
    }
    Ok(rows)
}

fn salted_page_hash(encoded: &[u8], sst_id: [u8; 16], property_id: u32, page_ordinal: u32) -> u64 {
    let mut hash = Xxh3::new();
    hash.update(&sst_id);
    hash.update(&property_id.to_le_bytes());
    hash.update(&page_ordinal.to_le_bytes());
    hash.update(encoded);
    hash.digest()
}

fn decode_top_level_value(bytes: &[u8], config: &NodePropertyPageConfig) -> Result<Value> {
    if bytes.is_empty() || bytes.len() > config.max_value_bytes {
        return Err(corrupted(
            "node property encoded Value has an invalid length",
        ));
    }
    let mut decoder = ValueDecoder {
        bytes,
        cursor: 0,
        elements: 0,
        config,
    };
    let value = decoder.decode(0)?;
    if decoder.cursor != bytes.len() {
        return Err(corrupted("node property Value has trailing bytes"));
    }
    if matches!(value, Value::Null) {
        return Err(corrupted(
            "top-level Null must use the explicit property-cell state",
        ));
    }
    Ok(value)
}

struct ValueDecoder<'a> {
    bytes: &'a [u8],
    cursor: usize,
    elements: u64,
    config: &'a NodePropertyPageConfig,
}

impl ValueDecoder<'_> {
    fn decode(&mut self, depth: u16) -> Result<Value> {
        if depth > self.config.max_value_depth {
            return Err(corrupted(format!(
                "node property Value depth exceeds {}",
                self.config.max_value_depth
            )));
        }
        self.elements = self
            .elements
            .checked_add(1)
            .ok_or_else(|| corrupted("node property Value element count overflows"))?;
        if self.elements > self.config.max_value_elements {
            return Err(corrupted(format!(
                "node property Value elements exceed {}",
                self.config.max_value_elements
            )));
        }
        let tag = self.read_u8()?;
        match tag {
            0 => Ok(Value::Null),
            1 => match self.read_u8()? {
                0 => Ok(Value::Bool(false)),
                1 => Ok(Value::Bool(true)),
                _ => Err(corrupted("node property Bool has a non-canonical byte")),
            },
            2 => Ok(Value::I64(i64::from_le_bytes(self.read_array()?))),
            3 => Ok(Value::F64(f64::from_bits(u64::from_le_bytes(
                self.read_array()?,
            )))),
            4 => {
                let bytes = self.read_length_prefixed()?;
                Ok(Value::Str(
                    std::str::from_utf8(bytes)
                        .map_err(|error| {
                            corrupted(format!("node property String is not UTF-8: {error}"))
                        })?
                        .to_string(),
                ))
            }
            5 => Ok(Value::Bytes(self.read_length_prefixed()?.to_vec())),
            6 => {
                let count = self.read_collection_count()?;
                self.consume_leaf_elements(count)?;
                let byte_len = count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| corrupted("node property vector byte length overflows"))?;
                let raw = self.read_exact(byte_len)?;
                let mut values = Vec::with_capacity(count);
                for component in raw.chunks_exact(4) {
                    values.push(f32::from_bits(u32::from_le_bytes(
                        component.try_into().unwrap(),
                    )));
                }
                Ok(Value::Vec(values))
            }
            7 => {
                let count = self.read_collection_count()?;
                self.consume_leaf_elements(count)?;
                let raw = self.read_exact(count)?;
                let codes = raw.iter().map(|code| *code as i8).collect();
                let scale = f32::from_bits(u32::from_le_bytes(self.read_array()?));
                Ok(Value::VecI8 { codes, scale })
            }
            8 => Ok(Value::Date(i32::from_le_bytes(self.read_array()?))),
            9 => Ok(Value::DateTime(i64::from_le_bytes(self.read_array()?))),
            10 => {
                let count = self.read_collection_count()?;
                self.require_future_elements(count)?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.decode(depth + 1)?);
                }
                Ok(Value::List(values))
            }
            11 => {
                let count = self.read_collection_count()?;
                self.require_future_elements(count)?;
                let mut values = BTreeMap::new();
                let mut previous_key: Option<String> = None;
                let max_key_bytes = self.config.max_property_name_bytes;
                for _ in 0..count {
                    let key_bytes = self.read_length_prefixed()?;
                    if key_bytes.len() > max_key_bytes {
                        return Err(corrupted(
                            "node property Map key exceeds configured name ceiling",
                        ));
                    }
                    let key = std::str::from_utf8(key_bytes)
                        .map_err(|error| {
                            corrupted(format!("node property Map key is not UTF-8: {error}"))
                        })?
                        .to_string();
                    if previous_key
                        .as_ref()
                        .is_some_and(|previous| previous.as_bytes() >= key.as_bytes())
                    {
                        return Err(corrupted("node property Map keys are not strictly sorted"));
                    }
                    let value = self.decode(depth + 1)?;
                    previous_key = Some(key.clone());
                    values.insert(key, value);
                }
                Ok(Value::Map(values))
            }
            other => Err(corrupted(format!(
                "node property Value has unknown tag {other}"
            ))),
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        let byte = *self
            .bytes
            .get(self.cursor)
            .ok_or_else(|| corrupted("node property Value is truncated"))?;
        self.cursor += 1;
        Ok(byte)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self
            .read_exact(N)?
            .try_into()
            .expect("exact node property slice length"))
    }

    fn read_exact(&mut self, len: usize) -> Result<&[u8]> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| corrupted("node property Value cursor overflows"))?;
        let bytes = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| corrupted("node property Value is truncated"))?;
        self.cursor = end;
        Ok(bytes)
    }

    fn read_length_prefixed(&mut self) -> Result<&[u8]> {
        let len = u32::from_le_bytes(self.read_array()?) as usize;
        self.read_exact(len)
    }

    fn read_collection_count(&mut self) -> Result<usize> {
        let count = u32::from_le_bytes(self.read_array()?) as usize;
        if count as u64 > self.config.max_value_elements {
            return Err(corrupted(format!(
                "node property Value collection exceeds {} elements",
                self.config.max_value_elements
            )));
        }
        Ok(count)
    }

    fn require_future_elements(&self, count: usize) -> Result<()> {
        if self
            .elements
            .checked_add(count as u64)
            .is_none_or(|elements| elements > self.config.max_value_elements)
        {
            return Err(corrupted(format!(
                "node property Value elements exceed {}",
                self.config.max_value_elements
            )));
        }
        Ok(())
    }

    fn consume_leaf_elements(&mut self, count: usize) -> Result<()> {
        self.elements = self
            .elements
            .checked_add(count as u64)
            .ok_or_else(|| corrupted("node property Value element count overflows"))?;
        if self.elements > self.config.max_value_elements {
            return Err(corrupted(format!(
                "node property Value elements exceed {}",
                self.config.max_value_elements
            )));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn flush_property_page(
    property: &mut PropertyBuildState,
    config: &NodePropertyPageConfig,
    sst_id: [u8; 16],
    index: &mut DiskSpool,
    payload: &mut DiskSpool,
    payload_crc: &mut crc32fast::Hasher,
    payload_hash: &mut Xxh3,
    total_page_count: &mut u32,
    copy_buffer: &mut [u8],
    peak_page_memory_bytes: &mut usize,
) -> Result<()> {
    if property.page.row_count == 0 {
        return Ok(());
    }
    let page = std::mem::take(&mut property.page);
    let decoded_len = (PAGE_HEADER_LEN as u64)
        .checked_add(page.rows.len)
        .ok_or_else(|| Error::invariant("node property decoded page length exceeds u64"))?;
    if decoded_len > config.max_page_decoded_bytes as u64 {
        return Err(Error::precondition(format!(
            "node property page requires {decoded_len} decoded bytes, above \
             NAMIDB_NODE_PROPERTY_MAX_PAGE_BYTES={}",
            config.max_page_decoded_bytes
        )));
    }
    let mut raw = DiskSpool::default();
    raw.write_all(&PAGE_HEADER_MAGIC)?;
    raw.write_all(&page.row_count.to_le_bytes())?;
    raw.write_all(&0u32.to_le_bytes())?;
    let mut rows = page.rows;
    rows.rewind()?;
    *peak_page_memory_bytes = (*peak_page_memory_bytes).max(copy_buffer.len());
    copy_exact_spool(&mut rows, &mut raw, copy_buffer)?;
    if raw.len != decoded_len {
        return Err(Error::invariant(
            "node property decoded page accounting mismatch",
        ));
    }

    let (mut encoded, codec) = if config.compress_pages {
        raw.rewind()?;
        let mut encoder =
            zstd::stream::write::Encoder::new(DiskSpool::default(), 3).map_err(|error| {
                Error::invariant(format!("node property page zstd init failed: {error}"))
            })?;
        copy_exact_reader(
            raw.file
                .as_mut()
                .ok_or_else(|| Error::invariant("node property raw page spool disappeared"))?,
            raw.len,
            &mut encoder,
            copy_buffer,
        )?;
        let compressed = encoder.finish().map_err(|error| {
            Error::invariant(format!("node property page zstd finish failed: {error}"))
        })?;
        if compressed.len < raw.len {
            (compressed, CODEC_ZSTD)
        } else {
            (raw, CODEC_RAW)
        }
    } else {
        (raw, CODEC_RAW)
    };
    if encoded.len == 0 || encoded.len > config.max_page_decoded_bytes as u64 {
        return Err(Error::precondition(format!(
            "node property encoded page requires {} bytes, above its {}-byte ceiling",
            encoded.len, config.max_page_decoded_bytes
        )));
    }

    let page_ordinal = property.page_count;
    let payload_offset = payload.len;
    let mut page_crc = crc32fast::Hasher::new();
    let mut page_hash = Xxh3::new();
    page_hash.update(&sst_id);
    page_hash.update(&property.property_id.to_le_bytes());
    page_hash.update(&page_ordinal.to_le_bytes());
    encoded.rewind()?;
    let encoded_len = encoded.len;
    let encoded_file = encoded
        .file
        .as_mut()
        .ok_or_else(|| Error::invariant("node property encoded page spool disappeared"))?;
    let mut remaining = encoded_len;
    while remaining > 0 {
        let take = usize::try_from(remaining.min(copy_buffer.len() as u64))
            .expect("node property copy buffer fits usize");
        encoded_file.read_exact(&mut copy_buffer[..take])?;
        let bytes = &copy_buffer[..take];
        page_crc.update(bytes);
        page_hash.update(bytes);
        payload_crc.update(bytes);
        payload_hash.update(bytes);
        payload.write_all(bytes)?;
        remaining -= take as u64;
    }

    let entry = PageIndexEntry {
        property_id: property.property_id,
        page_ordinal,
        first_node_id: page
            .first_node_id
            .ok_or_else(|| Error::invariant("node property page has no first NodeId"))?,
        last_node_id: page
            .last_node_id
            .ok_or_else(|| Error::invariant("node property page has no last NodeId"))?,
        first_ordinal: page.first_ordinal,
        last_ordinal: page.last_ordinal,
        payload_offset,
        encoded_len,
        decoded_len,
        row_count: page.row_count,
        codec,
        crc32: page_crc.finalize(),
        salted_xxh3: page_hash.digest(),
    };
    let mut raw_entry = Vec::with_capacity(INDEX_ENTRY_LEN);
    entry.encode(&mut raw_entry);
    property.index_crc.update(&raw_entry);
    property.index_hash.update(&raw_entry);
    index.write_all(&raw_entry)?;
    property.page_count = property
        .page_count
        .checked_add(1)
        .ok_or_else(|| Error::invariant("node property page count exceeds u32"))?;
    *total_page_count = total_page_count
        .checked_add(1)
        .ok_or_else(|| Error::invariant("total node property page count exceeds u32"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finish_property(
    mut property: PropertyBuildState,
    config: &NodePropertyPageConfig,
    sst_id: [u8; 16],
    catalog: &mut DiskSpool,
    index: &mut DiskSpool,
    payload: &mut DiskSpool,
    payload_crc: &mut crc32fast::Hasher,
    payload_hash: &mut Xxh3,
    total_page_count: &mut u32,
    copy_buffer: &mut [u8],
    peak_page_memory_bytes: &mut usize,
) -> Result<()> {
    flush_property_page(
        &mut property,
        config,
        sst_id,
        index,
        payload,
        payload_crc,
        payload_hash,
        total_page_count,
        copy_buffer,
        peak_page_memory_bytes,
    )?;
    if property.entry_count == 0 || property.page_count == 0 {
        return Err(Error::invariant(
            "node property catalog cannot contain an empty property",
        ));
    }
    let expected_index_len = u64::from(property.page_count)
        .checked_mul(INDEX_ENTRY_LEN as u64)
        .and_then(|len| property.index_offset.checked_add(len))
        .ok_or_else(|| Error::invariant("node property index span exceeds u64"))?;
    if expected_index_len != index.len {
        return Err(Error::invariant(
            "node property catalog index span is not contiguous",
        ));
    }
    let entry = CatalogEntry {
        name: property.name,
        property_id: property.property_id,
        index_offset: property.index_offset,
        page_count: property.page_count,
        index_crc32: property.index_crc.finalize(),
        index_xxh3: property.index_hash.digest(),
        entry_count: property.entry_count,
        min_node_id: property
            .min_node_id
            .ok_or_else(|| Error::invariant("node property has no minimum NodeId"))?,
        max_node_id: property
            .max_node_id
            .ok_or_else(|| Error::invariant("node property has no maximum NodeId"))?,
        min_ordinal: property.min_ordinal,
        max_ordinal: property.max_ordinal,
    };
    let mut bytes = Vec::with_capacity(CATALOG_ENTRY_FIXED_LEN + entry.name.len());
    entry.encode(&mut bytes)?;
    let projected = catalog
        .len
        .checked_add(bytes.len() as u64)
        .ok_or_else(|| Error::invariant("node property catalog length exceeds u64"))?;
    if projected > config.max_catalog_bytes as u64 {
        return Err(Error::precondition(format!(
            "node property catalog requires at least {projected} bytes, above \
             NAMIDB_NODE_PROPERTY_MAX_CATALOG_BYTES={}",
            config.max_catalog_bytes
        )));
    }
    catalog.write_all(&bytes)?;
    Ok(())
}

fn copy_exact_spool(
    source: &mut DiskSpool,
    destination: &mut DiskSpool,
    buffer: &mut [u8],
) -> Result<()> {
    let len = source.len;
    let reader = source
        .file
        .as_mut()
        .ok_or_else(|| Error::invariant("node property source spool disappeared"))?;
    copy_exact_reader(reader, len, destination, buffer)
}

fn copy_exact_reader(
    source: &mut impl Read,
    len: u64,
    destination: &mut impl Write,
    buffer: &mut [u8],
) -> Result<()> {
    let mut remaining = len;
    while remaining > 0 {
        let take = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("node property copy buffer fits usize");
        source.read_exact(&mut buffer[..take])?;
        destination.write_all(&buffer[..take])?;
        remaining -= take as u64;
    }
    Ok(())
}

fn spool_checksums(spool: &mut DiskSpool) -> Result<(u32, u64)> {
    spool.rewind()?;
    let mut crc = crc32fast::Hasher::new();
    let mut hash = Xxh3::new();
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let len = spool.len;
    let reader = spool
        .file
        .as_mut()
        .ok_or_else(|| Error::invariant("node property checksum spool disappeared"))?;
    let mut remaining = len;
    while remaining > 0 {
        let take = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("node property checksum buffer fits usize");
        reader.read_exact(&mut buffer[..take])?;
        crc.update(&buffer[..take]);
        hash.update(&buffer[..take]);
        remaining -= take as u64;
    }
    spool.rewind()?;
    Ok((crc.finalize(), hash.digest()))
}

fn aggregate_hash(header_bytes: &[u8], header: &Header) -> u64 {
    let mut hash = Xxh3::new();
    hash.update(&header.sst_id);
    hash.update(header_bytes);
    hash.update(&header.catalog_xxh3.to_le_bytes());
    hash.update(&header.index_xxh3.to_le_bytes());
    hash.update(&header.payload_xxh3.to_le_bytes());
    hash.digest()
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    env_number(name, default)
}

fn env_u32(name: &str, default: u32) -> Result<u32> {
    env_number(name, default)
}

fn env_u16(name: &str, default: u16) -> Result<u16> {
    env_number(name, default)
}

fn env_u64(name: &str, default: u64) -> Result<u64> {
    env_number(name, default)
}

fn env_number<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) => value.parse::<T>().map_err(|error| {
            Error::precondition(format!("{name} must be an exact integer: {error}"))
        }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::precondition(format!("{name} is not valid UTF-8")))
        }
    }
}

fn corrupted(detail: impl Into<String>) -> Error {
    Error::Corrupted {
        path: "<node-property-pages>".into(),
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

    fn node_id(value: u64) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[8..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn collect_upload(upload: NodePropertyPageUpload) -> (Uuid, Vec<u8>, NodePropertyBuildStats) {
        let id = upload.sst_id();
        let stats = upload.stats();
        let expected_len = upload.size_bytes();
        let (files, len) = upload.into_files();
        assert_eq!(len, expected_len);
        assert_eq!(files.len(), 5);
        let mut body = Vec::with_capacity(len as usize);
        for mut file in files {
            file.read_to_end(&mut body).unwrap();
        }
        assert_eq!(body.len() as u64, len);
        (id, body, stats)
    }

    async fn open_body(
        body: Vec<u8>,
        expected_id: Uuid,
        config: NodePropertyPageConfig,
    ) -> Result<NodePropertyPageReader> {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        // Fresh path per open: InMemory ETags are per-store counters, so
        // rewriting different bytes at one path across fresh stores would
        // collide in the process-global range cache if it is ever enabled.
        // The reader validates the header's sst_id, not the path.
        let path = Path::from(format!(
            "node-properties/{}/{expected_id}.npp",
            Uuid::now_v7()
        ));
        store.put(&path, PutPayload::from(body)).await.unwrap();
        let meta = store.head(&path).await.unwrap();
        let source = Arc::new(
            PinnedObjectRangeSource::from_create_only_meta(store, meta)
                .await
                .unwrap(),
        );
        NodePropertyPageReader::open(source, expected_id, config).await
    }

    fn compact_test_config() -> NodePropertyPageConfig {
        NodePropertyPageConfig {
            sort_memory_bytes: 256 * 1024,
            page_target_bytes: 16 * 1024,
            max_value_bytes: 1024 * 1024,
            max_page_decoded_bytes: 2 * 1024 * 1024,
            max_catalog_bytes: 1024 * 1024,
            max_properties: 256,
            max_value_depth: 16,
            max_value_elements: 100_000,
            max_property_name_bytes: 256,
            max_index_query_bytes: 1024 * 1024,
            max_query_rows: 100_000,
            compress_pages: true,
        }
    }

    #[tokio::test]
    async fn round_trips_exact_values_null_absent_projection_and_filters() {
        let config = compact_test_config();
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        let mut nested_map = BTreeMap::new();
        nested_map.insert("alpha".into(), Value::I64(7));
        nested_map.insert("beta".into(), Value::Null);
        let mut first = BTreeMap::new();
        first.insert("bool".into(), Value::Bool(true));
        first.insert("bytes".into(), Value::Bytes(vec![0, 1, 255]));
        first.insert("compressible".into(), Value::Str("L".repeat(10_000)));
        first.insert("date".into(), Value::Date(20_000));
        first.insert("datetime".into(), Value::DateTime(1_700_000_000));
        first.insert("f64".into(), Value::F64(-12.5));
        first.insert("i64".into(), Value::I64(-99));
        first.insert(
            "list".into(),
            Value::List(vec![Value::Str("x".into()), Value::Null]),
        );
        first.insert("map".into(), Value::Map(nested_map.clone()));
        first.insert("null".into(), Value::Null);
        first.insert("str".into(), Value::Str("derecho".into()));
        first.insert("vec".into(), Value::Vec(vec![1.0, -2.5, 3.25]));
        first.insert(
            "veci8".into(),
            Value::VecI8 {
                codes: vec![-128, 0, 127],
                scale: 0.25,
            },
        );
        builder.push_sorted(node_id(1), 10, &first).unwrap();
        builder
            .push_sorted(
                node_id(2),
                11,
                &BTreeMap::from([("str".into(), Value::Str("mercantil".into()))]),
            )
            .unwrap();
        let (sst_id, body, _) = collect_upload(builder.finish().unwrap());
        let reader = open_body(body, sst_id, config).await.unwrap();
        let names = first
            .keys()
            .cloned()
            .chain(std::iter::once("missing".into()))
            .collect::<Vec<_>>();
        let (rows, stats) = reader
            .project_node_ids(&names, &[node_id(1), node_id(2), node_id(1)])
            .await
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], rows[2]);
        assert_eq!(rows[0].ordinal, Some(10));
        assert_eq!(rows[0].properties["null"], PropertyCell::Null);
        assert_eq!(
            rows[0].properties["map"],
            PropertyCell::Value(Value::Map(nested_map))
        );
        for (name, value) in &first {
            if !matches!(value, Value::Null) {
                assert_eq!(
                    rows[0].properties[name],
                    PropertyCell::Value(value.clone()),
                    "property {name}"
                );
            }
        }
        assert_eq!(rows[0].properties["missing"], PropertyCell::Absent);
        assert_eq!(rows[1].properties["bool"], PropertyCell::Absent);
        assert!(stats.payload_pages > 0);
        let compressed_pages = reader
            .load_property_index(reader.catalog_entry("compressible").unwrap())
            .await
            .unwrap();
        assert_eq!(compressed_pages.len(), 1);
        assert_eq!(compressed_pages[0].codec, CODEC_ZSTD);

        let (filtered, _) = reader
            .filter_node_ids(
                &[
                    NodePropertyPredicate::EqBool {
                        property: "bool".into(),
                        value: true,
                    },
                    NodePropertyPredicate::InString {
                        property: "str".into(),
                        values: vec!["derecho".into()],
                    },
                ],
                &["str".into()],
                &[node_id(1), node_id(2)],
            )
            .await
            .unwrap();
        let NodePropertyFilterResult::Matches(filtered) = filtered else {
            panic!("supported predicates must not return Unsupported");
        };
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].node_id, node_id(1));
        assert_eq!(filtered[0].properties.len(), 1);
        let (unsupported, unsupported_stats) = reader
            .filter_node_ids(
                &[NodePropertyPredicate::Unsupported {
                    description: "numeric range".into(),
                }],
                &[],
                &[node_id(1)],
            )
            .await
            .unwrap();
        assert!(matches!(
            unsupported,
            NodePropertyFilterResult::Unsupported { .. }
        ));
        assert_eq!(unsupported_stats, NodePropertyProbeStats::default());
        reader.verify_all().await.unwrap();
    }

    #[tokio::test]
    async fn title_projection_does_not_read_embedding_pages() {
        let mut config = compact_test_config();
        config.compress_pages = false;
        config.page_target_bytes = 64 * 1024;
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        for row in 0..100u64 {
            let properties = BTreeMap::from([
                (
                    "embedding".into(),
                    Value::Vec((0..1024).map(|value| value as f32 + row as f32).collect()),
                ),
                ("title".into(), Value::Str(format!("Articulo {row}"))),
            ]);
            builder
                .push_sorted(node_id(row + 1), row, &properties)
                .unwrap();
        }
        let (sst_id, body, _) = collect_upload(builder.finish().unwrap());
        let object_bytes = body.len() as u64;
        let reader = open_body(body, sst_id, config).await.unwrap();
        let (rows, stats) = reader
            .project_node_ids(&["title".into()], &[node_id(50)])
            .await
            .unwrap();
        assert_eq!(
            rows[0].properties["title"],
            PropertyCell::Value(Value::Str("Articulo 49".into()))
        );
        assert_eq!(stats.payload_pages, 1);
        assert!(
            stats.payload_bytes < 10_000,
            "title read unexpectedly fetched {} payload bytes",
            stats.payload_bytes
        );
        assert!(stats.payload_bytes.saturating_mul(20) < object_bytes);
        let (range, range_stats) = reader
            .project_ordinal_range(&["title".into()], 48..51)
            .await
            .unwrap();
        assert_eq!(
            range.iter().map(|row| row.ordinal).collect::<Vec<_>>(),
            vec![Some(48), Some(49), Some(50)]
        );
        assert_eq!(range_stats.payload_pages, 1);
    }

    #[tokio::test]
    async fn giant_value_is_spooled_and_round_trips_with_explicit_ceilings() {
        let mut config = compact_test_config();
        config.sort_memory_bytes = 2 * 1024 * 1024;
        config.page_target_bytes = 128 * 1024;
        config.max_value_bytes = 512 * 1024;
        config.max_page_decoded_bytes = 1024 * 1024;
        config.compress_pages = false;
        let giant = vec![0xA5; 400 * 1024];
        let giant_string = "derecho-".repeat(40_000);
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        builder
            .push_sorted(
                node_id(1),
                1,
                &BTreeMap::from([
                    ("blob".into(), Value::Bytes(giant.clone())),
                    ("text".into(), Value::Str(giant_string.clone())),
                ]),
            )
            .unwrap();
        let (sst_id, body, stats) = collect_upload(builder.finish().unwrap());
        assert!(stats.peak_sort_memory_bytes <= config.sort_memory_bytes);
        assert!(stats.peak_page_memory_bytes <= config.max_page_decoded_bytes);
        let reader = open_body(body, sst_id, config).await.unwrap();
        let (rows, _) = reader
            .project_node_ids(&["blob".into(), "text".into()], &[node_id(1)])
            .await
            .unwrap();
        assert_eq!(
            rows[0].properties["blob"],
            PropertyCell::Value(Value::Bytes(giant))
        );
        assert_eq!(
            rows[0].properties["text"],
            PropertyCell::Value(Value::Str(giant_string))
        );
    }

    #[tokio::test]
    async fn value_larger_than_default_sort_budget_is_key_only_spooled() {
        let config = NodePropertyPageConfig::default();
        let giant_len = config.sort_memory_bytes + 1024 * 1024 + 17;
        let properties = BTreeMap::from([("blob".into(), Value::Bytes(vec![0x5A; giant_len]))]);
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        builder.push_sorted(node_id(1), 1, &properties).unwrap();
        let (sst_id, body, stats) = collect_upload(builder.finish().unwrap());
        assert!(stats.value_spool_bytes > config.sort_memory_bytes as u64);
        assert!(stats.peak_sort_memory_bytes <= config.sort_memory_bytes);
        assert!(stats.peak_page_memory_bytes <= COPY_BUFFER_BYTES);

        let reader = open_body(body, sst_id, config).await.unwrap();
        let (rows, _) = reader
            .project_node_ids(&["blob".into()], &[node_id(1)])
            .await
            .unwrap();
        let PropertyCell::Value(Value::Bytes(decoded)) = &rows[0].properties["blob"] else {
            panic!("giant value changed type");
        };
        assert_eq!(decoded.len(), giant_len);
        assert!(decoded.iter().all(|byte| *byte == 0x5A));
    }

    #[tokio::test]
    async fn corruption_truncation_and_uuid_mismatch_fail_closed() {
        let mut config = compact_test_config();
        config.compress_pages = false;
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        builder
            .push_sorted(
                node_id(1),
                1,
                &BTreeMap::from([("title".into(), Value::Str("Ley".into()))]),
            )
            .unwrap();
        let (sst_id, body, _) = collect_upload(builder.finish().unwrap());

        let wrong = open_body(body.clone(), Uuid::now_v7(), config.clone()).await;
        assert!(matches!(wrong, Err(Error::Corrupted { .. })));

        let mut truncated = body.clone();
        truncated.pop();
        let truncated = open_body(truncated, sst_id, config.clone()).await;
        assert!(matches!(truncated, Err(Error::Corrupted { .. })));

        let (header, _) = Header::decode(&body[..HEADER_LEN]).unwrap();
        let mut corrupt_catalog = body.clone();
        corrupt_catalog[header.catalog_offset as usize] ^= 0x80;
        let corrupt_catalog = open_body(corrupt_catalog, sst_id, config.clone()).await;
        assert!(matches!(corrupt_catalog, Err(Error::Corrupted { .. })));

        let mut corrupt_payload = body;
        corrupt_payload[header.payload_offset as usize] ^= 0x01;
        let reader = open_body(corrupt_payload, sst_id, config).await.unwrap();
        let result = reader
            .project_node_ids(&["title".into()], &[node_id(1)])
            .await;
        assert!(matches!(result, Err(Error::Corrupted { .. })));
    }

    #[test]
    fn builder_memory_accounting_stays_within_configured_bounds() {
        let mut config = compact_test_config();
        config.sort_memory_bytes = 64 * 1024;
        config.page_target_bytes = 8 * 1024;
        config.compress_pages = false;
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        for row in 0..5_000u64 {
            builder
                .push_sorted(
                    node_id(row + 1),
                    row,
                    &BTreeMap::from([
                        ("embedding".into(), Value::Vec(vec![row as f32; 64])),
                        ("flag".into(), Value::Bool(row % 2 == 0)),
                        ("title".into(), Value::Str(format!("Articulo {row}"))),
                    ]),
                )
                .unwrap();
        }
        let (_, _, stats) = collect_upload(builder.finish().unwrap());
        assert_eq!(stats.node_count, 5_000);
        assert_eq!(stats.cell_count, 15_000);
        assert!(stats.page_count > 3);
        assert!(
            stats.peak_sort_memory_bytes <= config.sort_memory_bytes,
            "{} > {}",
            stats.peak_sort_memory_bytes,
            config.sort_memory_bytes
        );
        assert!(stats.peak_page_memory_bytes <= config.max_page_decoded_bytes);
    }

    #[test]
    fn limits_and_order_are_fail_closed() {
        let mut config = compact_test_config();
        config.max_value_depth = 2;
        config.max_value_elements = 4;
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        let deep = Value::List(vec![Value::List(vec![Value::List(vec![Value::I64(1)])])]);
        assert!(builder
            .push_sorted(node_id(1), 1, &BTreeMap::from([("deep".into(), deep)]))
            .is_err());

        let mut builder = NodePropertyPageBuilder::new(config).unwrap();
        builder
            .push_sorted(node_id(2), 2, &BTreeMap::new())
            .unwrap();
        assert!(builder
            .push_sorted(node_id(1), 3, &BTreeMap::new())
            .is_err());
    }

    /// Slow release/soak coverage for the stated production shape. The builder
    /// receives one 1024d row at a time and the assertions prove its retained
    /// workspace remains independent of the 400+ MiB corpus.
    #[test]
    #[ignore = "large 100k x 1024d bounded-memory soak"]
    fn soak_100k_rows_1024d_is_disk_spooled() {
        let config = NodePropertyPageConfig {
            compress_pages: false,
            max_value_bytes: 4 * 1024 * 1024,
            max_page_decoded_bytes: 8 * 1024 * 1024,
            ..Default::default()
        };
        let mut builder = NodePropertyPageBuilder::new(config.clone()).unwrap();
        for row in 0..100_000u64 {
            builder
                .push_sorted(
                    node_id(row + 1),
                    row,
                    &BTreeMap::from([("embedding".into(), Value::Vec(vec![row as f32; 1024]))]),
                )
                .unwrap();
        }
        let (_, _, stats) = collect_upload(builder.finish().unwrap());
        assert_eq!(stats.node_count, 100_000);
        assert!(stats.peak_sort_memory_bytes <= config.sort_memory_bytes);
        assert!(stats.peak_page_memory_bytes <= config.max_page_decoded_bytes);
    }
}
