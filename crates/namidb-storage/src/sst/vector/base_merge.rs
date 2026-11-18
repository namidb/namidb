//! Clustered-base rebuild from a vector generation's own segments.
//!
//! A vector generation is one optional clustered `NAMIVG05` base followed by
//! exact `NAMIVG06` deltas. Both keep every live row at full precision: the
//! base stores the f32 vector and the native-filter values of each row, a
//! delta stores the f32 vector, the version record (NodeId, LSN, live or
//! suppressed, payload fingerprint) and complete native-filter postings. A new
//! base can therefore be assembled from the segments alone, without reading
//! the node corpus:
//!
//! 1. the base rows (stored in cluster order) are sorted by NodeId through a
//!    bounded external sort;
//! 2. the deltas' version tables are k-way merged by NodeId together with that
//!    sorted base stream. A node with any delta record takes its highest-LSN
//!    record (equal LSN with different content is corruption) and the base
//!    row is shadowed, exactly as queries reconcile a base with its deltas; a
//!    node without one keeps its base row. Suppressions are dropped;
//! 3. each live winner's vector comes from its segment's payload pages (read
//!    in lockstep with its version table) and its filter values from the
//!    segment's postings; a delta winner must re-fingerprint to its version
//!    record, so a reconstructed payload is proven identical to the flushed
//!    one;
//! 4. the winners stream, in NodeId order, into the same
//!    [`VectorV5ExternalCollector`] the node-scan rebuild feeds, so the output
//!    is the base a node scan over the same corpus produces (byte-identical
//!    under the same build configuration).
//!
//! Resident state: the external sort's buffer, one version page and one
//! payload page per delta, the delta filter columns (4 bytes per live row and
//! filter property), and the collector's own bounded working set, which runs
//! after the merge state is released.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use futures::executor::block_on;
use namidb_core::Value;
use xxhash_rust::xxh3::Xxh3;

use super::v5::external::{
    VectorV5ExternalArtifact, VectorV5ExternalBuildConfig, VectorV5ExternalCollector,
};
use super::v5::{VectorV5BuildOptions, VectorV5Reader};
use super::v6::{
    vector_v6_payload_fingerprint, VectorPageRow, VectorV6FilterColumn, VectorV6Payload,
    VectorV6Reader,
};
use crate::error::{Error, Result};
use crate::manifest::{VectorIndexDescriptor, VectorMetric};
use crate::sst::external_pairs::{ExternalPairMerge, ExternalPairSorter};
use crate::sst::search_delta::{
    reconcile_node_versions, SearchFilterValue, SearchVersionOperation, SearchVersionRecord,
};

const MIN_SORT_MEMORY_BYTES: usize = 64 * 1024;
const FILTER_ABSENT: u8 = 0;
const FILTER_FALSE: u8 = 1;
const FILTER_TRUE: u8 = 2;
const FILTER_STRING: u8 = 3;

/// Segments of one vector generation and the base build configuration.
#[derive(Debug)]
pub struct VectorBaseMergeInput<'a> {
    pub descriptor: &'a VectorIndexDescriptor,
    /// The generation's native-filter properties (sorted). Every source must
    /// advertise each of them as complete.
    pub filter_properties: &'a [String],
    /// The generation's clustered base, if it has one.
    pub base: Option<&'a VectorV5Reader>,
    /// The generation's deltas, oldest first.
    pub deltas: &'a [VectorV6Reader],
    pub collector: VectorV5ExternalBuildConfig,
    pub options: VectorV5BuildOptions,
}

/// Result of [`merge_vector_base`].
#[derive(Debug)]
pub struct VectorBaseMergeOutput {
    /// `None` when the collector produced no body (no live winner, or fewer
    /// rows than a V5 body needs; see `live_rows`).
    pub artifact: Option<VectorV5ExternalArtifact>,
    /// Base rows plus delta version records read.
    pub input_rows: u64,
    /// Distinct NodeIds across the sources.
    pub touched_rows: u64,
    /// Live winners pushed into the collector.
    pub live_rows: u64,
    /// Order-sensitive digest of every winner; non-zero.
    pub winner_digest: u64,
    /// Peak merge-side workspace (sort buffer, filter columns, decoded pages),
    /// before the collector's own build.
    pub peak_merge_workspace_bytes: usize,
}

/// Merge `input`'s segments into one clustered base. Readers' async methods
/// are driven with `block_on`: run this on a blocking thread over locally
/// readable sources.
pub fn merge_vector_base(input: VectorBaseMergeInput<'_>) -> Result<VectorBaseMergeOutput> {
    let VectorBaseMergeInput {
        descriptor,
        filter_properties,
        base,
        deltas,
        collector,
        options,
    } = input;
    if base.is_none() && deltas.is_empty() {
        return Err(Error::invariant(
            "vector base merge requires at least one source",
        ));
    }
    if filter_properties.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::invariant(
            "vector base merge filter properties are not sorted and distinct",
        ));
    }
    let workspace_budget = collector.memory_budget_bytes;
    let sort_budget = (workspace_budget / 4).max(MIN_SORT_MEMORY_BYTES);
    let dim = descriptor.dim as usize;

    // 1. Base rows into NodeId order.
    let mut base_rows = 0u64;
    let base_stream = match base {
        Some(base) => {
            if base.dim() != descriptor.dim || base.metric() != descriptor.metric {
                return Err(Error::precondition(
                    "vector base merge: base dimension or metric differs from its catalog entry",
                ));
            }
            if base.filter_properties() != filter_properties {
                return Err(Error::precondition(
                    "vector base merge: base filter properties differ from the generation's",
                ));
            }
            let mut sorter = ExternalPairSorter::with_memory_limit(sort_budget)?;
            let mut key = Vec::new();
            for page in 0..base.page_count() {
                for row in block_on(base.read_page_rows(page))? {
                    encode_base_row(
                        &row.id,
                        &row.vector,
                        &row.filters,
                        filter_properties,
                        &mut key,
                    )?;
                    sorter.push(0, &key, row.id)?;
                    base_rows += 1;
                }
            }
            if base_rows != base.point_count() {
                return Err(Error::invariant(
                    "vector base merge: base rows disagree with its point count",
                ));
            }
            Some(sorter.finish()?)
        }
        None => None,
    };
    let mut base_stream = BaseStream::new(base_stream, dim, filter_properties)?;

    // 2. Delta cursors with their filter columns.
    let mut cursors = Vec::with_capacity(deltas.len());
    let mut column_bytes = 0usize;
    for reader in deltas {
        if reader.dim() != descriptor.dim || reader.metric() != descriptor.metric {
            return Err(Error::precondition(
                "vector base merge: delta dimension or metric differs from its catalog entry",
            ));
        }
        let mut columns = Vec::with_capacity(filter_properties.len());
        for property in filter_properties {
            let column = block_on(reader.live_filter_column(property))?.ok_or_else(|| {
                Error::precondition(format!(
                    "vector base merge: a delta does not advertise native filter '{property}'"
                ))
            })?;
            column_bytes = column_bytes.saturating_add(column.resident_bytes());
            if column_bytes > workspace_budget / 4 {
                return Err(Error::precondition(format!(
                    "vector base merge: delta filter columns need more than {} bytes",
                    workspace_budget / 4
                )));
            }
            columns.push(column);
        }
        cursors.push(DeltaCursor::open(reader, columns)?);
    }
    let mut heap = BinaryHeap::with_capacity(cursors.len());
    for (source, cursor) in cursors.iter().enumerate() {
        if let Some(record) = cursor.current() {
            heap.push(Reverse((record.node_id, source)));
        }
    }

    // 3. Winners, in NodeId order, into the base collector.
    let mut builder = VectorV5ExternalCollector::new_base(collector)?;
    let mut digest = Xxh3::new();
    digest.update(b"NamiDB/VectorBaseSegmentMerge/v1");
    let mut touched_rows = 0u64;
    let mut live_rows = 0u64;
    let mut peak_merge_workspace_bytes = 0usize;
    let mut filters = BTreeMap::new();
    loop {
        let delta_min = heap.peek().map(|Reverse((node_id, _))| *node_id);
        let base_min = base_stream.head().map(|row| row.id);
        let node_id = match (delta_min, base_min) {
            (None, None) => break,
            (Some(delta), Some(base)) => delta.min(base),
            (Some(delta), None) => delta,
            (None, Some(base)) => base,
        };
        touched_rows = touched_rows
            .checked_add(1)
            .ok_or_else(|| Error::invariant("vector base merge touched rows overflow"))?;
        let base_row = if base_min == Some(node_id) {
            base_stream.take()?
        } else {
            None
        };
        let mut group: Vec<(usize, SearchVersionRecord, Option<VectorPageRow>)> = Vec::new();
        while let Some(Reverse((candidate, source))) = heap.peek().copied() {
            if candidate != node_id {
                break;
            }
            heap.pop();
            let (record, payload) = cursors[source].take()?;
            if record.node_id != node_id {
                return Err(Error::invariant(
                    "vector base merge heap key disagrees with its version record",
                ));
            }
            group.push((source, record, payload));
            if let Some(next) = cursors[source].current() {
                heap.push(Reverse((next.node_id, source)));
            }
        }
        peak_merge_workspace_bytes = peak_merge_workspace_bytes.max(
            sort_budget
                .saturating_add(column_bytes)
                .saturating_add(cursors.iter().map(DeltaCursor::resident_bytes).sum()),
        );

        let (vector, row_filters) = if group.is_empty() {
            let row =
                base_row.ok_or_else(|| Error::invariant("vector base merge lost its base row"))?;
            update_digest(&mut digest, node_id, 0, true, 0);
            (row.vector, row.filters)
        } else {
            // Any delta record shadows the base row: the base predates every
            // delta of the generation.
            let winner = reconcile_node_versions(group.iter().map(|entry| entry.1))
                .map_err(|error| Error::Corrupted {
                    path: "<search-delta-version-tables>".into(),
                    detail: error.to_string(),
                })?
                .ok_or_else(|| Error::invariant("vector base merge group has no winner"))?;
            let live = matches!(winner.operation, SearchVersionOperation::Live { .. });
            update_digest(
                &mut digest,
                node_id,
                winner.lsn,
                live,
                winner.payload_fingerprint,
            );
            if !live {
                continue;
            }
            // Identical records in several sources carry identical payloads;
            // take the newest source's.
            let (source, record, payload) = group
                .into_iter()
                .rev()
                .find(|(_, record, _)| *record == winner)
                .ok_or_else(|| Error::invariant("vector base merge lost its winning source"))?;
            let row = payload
                .ok_or_else(|| Error::invariant("vector base merge live winner has no payload"))?;
            let mut payload = VectorV6Payload {
                vector: row.vector,
                filters: BTreeMap::new(),
            };
            for (property, column) in filter_properties.iter().zip(&cursors[source].filters) {
                if let Some(value) = column.value(row.ordinal)? {
                    payload.filters.insert(property.clone(), value.clone());
                }
            }
            if vector_v6_payload_fingerprint(&payload)? != record.payload_fingerprint {
                return Err(Error::Corrupted {
                    path: "<search-delta-segments>".into(),
                    detail: format!(
                        "vector payload of NodeId {node_id:02x?} reassembled from its segment \
                         does not match its version record"
                    ),
                });
            }
            let mut row_filters = Vec::with_capacity(payload.filters.len());
            for (property, value) in payload.filters {
                row_filters.push((property, filter_value(value)?));
            }
            (payload.vector, row_filters)
        };
        // The V5 builder drops zero vectors under cosine (not rankable); a
        // flush already records them as suppressions.
        if descriptor.metric == VectorMetric::Cosine && vector.iter().all(|value| *value == 0.0) {
            continue;
        }
        filters.clear();
        for property in filter_properties {
            filters.insert(property.clone(), Value::Null);
        }
        for (property, value) in row_filters {
            let slot = filters.get_mut(&property).ok_or_else(|| {
                Error::invariant("vector base merge row carries an unadvertised filter")
            })?;
            *slot = value;
        }
        builder.push(node_id, &vector, &filters)?;
        live_rows += 1;
    }

    let mut input_rows = base_rows;
    for cursor in &cursors {
        cursor.finish_check()?;
        input_rows = input_rows
            .checked_add(cursor.records)
            .ok_or_else(|| Error::invariant("vector base merge input rows overflow"))?;
    }
    base_stream.finish_check()?;
    // Release the merge state before the collector's own working set.
    drop(cursors);
    drop(base_stream);
    if builder.input_rows() != live_rows {
        return Err(Error::invariant(
            "vector base merge collector count disagrees with its winners",
        ));
    }
    let artifact = builder.finish(descriptor, options)?;
    Ok(VectorBaseMergeOutput {
        artifact,
        input_rows,
        touched_rows,
        live_rows,
        winner_digest: digest.digest().max(1),
        peak_merge_workspace_bytes,
    })
}

fn update_digest(digest: &mut Xxh3, node_id: [u8; 16], lsn: u64, live: bool, fingerprint: u64) {
    digest.update(&node_id);
    digest.update(&lsn.to_le_bytes());
    digest.update(&[u8::from(live)]);
    digest.update(&fingerprint.to_le_bytes());
}

/// The node-scan rebuild hands the collector Bool/String values only (the
/// catalog's native-filter types); anything else is not a V5 filter.
fn filter_value(value: SearchFilterValue) -> Result<Value> {
    match value {
        SearchFilterValue::Bool(value) => Ok(Value::Bool(value)),
        SearchFilterValue::String(value) => Ok(Value::Str(value)),
        _ => Err(Error::precondition(
            "vector base merge: a native filter value is neither Bool nor String",
        )),
    }
}

/// Sort key of one base row: its NodeId (the sort order), its vector, then one
/// tagged value per generation filter property.
fn encode_base_row(
    id: &[u8; 16],
    vector: &[f32],
    filters: &[(String, Value)],
    filter_properties: &[String],
    key: &mut Vec<u8>,
) -> Result<()> {
    key.clear();
    key.extend_from_slice(id);
    for component in vector {
        key.extend_from_slice(&component.to_le_bytes());
    }
    let mut stored = filters.iter().peekable();
    for property in filter_properties {
        match stored.next_if(|(stored, _)| stored == property) {
            None => key.push(FILTER_ABSENT),
            Some((_, Value::Bool(false))) => key.push(FILTER_FALSE),
            Some((_, Value::Bool(true))) => key.push(FILTER_TRUE),
            Some((_, Value::Str(value))) => {
                key.push(FILTER_STRING);
                key.extend_from_slice(
                    &u32::try_from(value.len())
                        .map_err(|_| Error::invariant("vector base filter value exceeds u32"))?
                        .to_le_bytes(),
                );
                key.extend_from_slice(value.as_bytes());
            }
            Some(_) => {
                return Err(Error::invariant(
                    "vector base row carries a non Bool/String filter value",
                ))
            }
        }
    }
    if stored.next().is_some() {
        return Err(Error::invariant(
            "vector base row carries an unadvertised filter",
        ));
    }
    Ok(())
}

struct BaseRow {
    id: [u8; 16],
    vector: Vec<f32>,
    filters: Vec<(String, Value)>,
}

/// The NodeId-sorted base rows, one row ahead.
struct BaseStream<'a> {
    merge: Option<ExternalPairMerge>,
    head: Option<BaseRow>,
    dim: usize,
    filter_properties: &'a [String],
}

impl<'a> BaseStream<'a> {
    fn new(
        merge: Option<ExternalPairMerge>,
        dim: usize,
        filter_properties: &'a [String],
    ) -> Result<Self> {
        let mut stream = BaseStream {
            merge,
            head: None,
            dim,
            filter_properties,
        };
        stream.head = stream.next_row()?;
        Ok(stream)
    }

    fn head(&self) -> Option<&BaseRow> {
        self.head.as_ref()
    }

    fn take(&mut self) -> Result<Option<BaseRow>> {
        let row = self.head.take();
        if let Some(row) = &row {
            let next = self.next_row()?;
            if next.as_ref().is_some_and(|next| next.id <= row.id) {
                return Err(Error::invariant(
                    "vector base merge base rows are not NodeId-sorted and distinct",
                ));
            }
            self.head = next;
        }
        Ok(row)
    }

    fn finish_check(&self) -> Result<()> {
        if self.head.is_some() {
            return Err(Error::invariant("vector base merge left base rows unread"));
        }
        Ok(())
    }

    fn next_row(&mut self) -> Result<Option<BaseRow>> {
        let Some(merge) = self.merge.as_mut() else {
            return Ok(None);
        };
        let Some(pair) = merge.next_pair()? else {
            return Ok(None);
        };
        decode_base_row(&pair.key, pair.id, self.dim, self.filter_properties).map(Some)
    }
}

/// Inverse of [`encode_base_row`].
fn decode_base_row(
    key: &[u8],
    id: [u8; 16],
    dim: usize,
    filter_properties: &[String],
) -> Result<BaseRow> {
    let corrupt = || Error::invariant("vector base merge sort record is malformed");
    let vector_end = dim
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(corrupt)?;
    if key.len() < vector_end || key[..16] != id {
        return Err(corrupt());
    }
    let vector = key[16..vector_end]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    let mut cursor = vector_end;
    let mut filters = Vec::new();
    for property in filter_properties {
        let tag = *key.get(cursor).ok_or_else(corrupt)?;
        cursor += 1;
        let value = match tag {
            FILTER_ABSENT => continue,
            FILTER_FALSE => Value::Bool(false),
            FILTER_TRUE => Value::Bool(true),
            FILTER_STRING => {
                let len = key.get(cursor..cursor + 4).ok_or_else(corrupt)?;
                let len = u32::from_le_bytes([len[0], len[1], len[2], len[3]]) as usize;
                cursor += 4;
                let end = cursor.checked_add(len).ok_or_else(corrupt)?;
                let bytes = key.get(cursor..end).ok_or_else(corrupt)?;
                cursor = end;
                Value::Str(String::from_utf8(bytes.to_vec()).map_err(|_| corrupt())?)
            }
            _ => return Err(corrupt()),
        };
        filters.push((property.clone(), value));
    }
    if cursor != key.len() {
        return Err(corrupt());
    }
    Ok(BaseRow {
        id,
        vector,
        filters,
    })
}

/// One delta's version table and live payload pages, read in lockstep.
struct DeltaCursor<'a> {
    reader: &'a VectorV6Reader,
    versions: Vec<SearchVersionRecord>,
    version_page: usize,
    version_offset: usize,
    payloads: std::vec::IntoIter<VectorPageRow>,
    payload_page: usize,
    payload_page_bytes: usize,
    next_ordinal: u64,
    filters: Vec<VectorV6FilterColumn>,
    records: u64,
}

impl<'a> DeltaCursor<'a> {
    fn open(reader: &'a VectorV6Reader, filters: Vec<VectorV6FilterColumn>) -> Result<Self> {
        let versions = reader.version_reader();
        let records = if versions.page_count() == 0 {
            Vec::new()
        } else {
            block_on(versions.read_page(0))?
        };
        Ok(Self {
            reader,
            versions: records,
            version_page: 0,
            version_offset: 0,
            payloads: Vec::new().into_iter(),
            payload_page: 0,
            payload_page_bytes: 0,
            next_ordinal: 0,
            filters,
            records: 0,
        })
    }

    fn current(&self) -> Option<&SearchVersionRecord> {
        self.versions.get(self.version_offset)
    }

    fn resident_bytes(&self) -> usize {
        self.versions
            .capacity()
            .saturating_mul(std::mem::size_of::<SearchVersionRecord>())
            .saturating_add(self.payload_page_bytes)
    }

    /// The current version record and, when it is live, its payload row.
    fn take(&mut self) -> Result<(SearchVersionRecord, Option<VectorPageRow>)> {
        let record = *self
            .current()
            .ok_or_else(|| Error::invariant("vector base merge read past a version table"))?;
        let payload = match record.operation {
            SearchVersionOperation::Live { payload_ordinal } => {
                if payload_ordinal != self.next_ordinal {
                    return Err(corrupt_delta("live ordinals are not dense in NodeId order"));
                }
                let row = self.next_payload()?;
                if row.ordinal != payload_ordinal
                    || row.node_id != record.node_id
                    || row.lsn != record.lsn
                    || row.payload_fingerprint != record.payload_fingerprint
                {
                    return Err(corrupt_delta(
                        "a live payload disagrees with its version record",
                    ));
                }
                self.next_ordinal += 1;
                Some(row)
            }
            SearchVersionOperation::Suppress => None,
        };
        self.records += 1;
        self.version_offset += 1;
        if self.version_offset == self.versions.len() {
            self.version_page += 1;
            self.version_offset = 0;
            let versions = self.reader.version_reader();
            self.versions = if self.version_page < versions.page_count() {
                let page = block_on(versions.read_page(self.version_page))?;
                if page.is_empty() {
                    return Err(corrupt_delta("a version page is empty"));
                }
                page
            } else {
                Vec::new()
            };
        }
        if self
            .current()
            .is_some_and(|next| next.node_id <= record.node_id)
        {
            return Err(corrupt_delta("a version table is not NodeId-sorted"));
        }
        Ok((record, payload))
    }

    fn next_payload(&mut self) -> Result<VectorPageRow> {
        loop {
            if let Some(row) = self.payloads.next() {
                return Ok(row);
            }
            if self.payload_page == self.reader.page_count() {
                return Err(corrupt_delta("a live version record has no payload row"));
            }
            let rows = block_on(self.reader.read_live_page(self.payload_page))?;
            self.payload_page += 1;
            self.payload_page_bytes = rows
                .iter()
                .map(|row| std::mem::size_of::<VectorPageRow>() + row.vector.capacity() * 4)
                .sum();
            self.payloads = rows.into_iter();
        }
    }

    fn finish_check(&self) -> Result<()> {
        let reference = self.reader.version_reader().reference();
        if self.current().is_some()
            || self.records != reference.record_count
            || self.next_ordinal != reference.live_count
            || self.payload_page != self.reader.page_count()
            || self.payloads.len() != 0
        {
            return Err(corrupt_delta(
                "version records and live payloads were not consumed exactly",
            ));
        }
        Ok(())
    }
}

fn corrupt_delta(detail: &str) -> Error {
    Error::Corrupted {
        path: "<vector-delta-segment>".into(),
        detail: detail.into(),
    }
}
