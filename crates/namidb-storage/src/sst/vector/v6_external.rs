//! Bounded-memory streaming builder for one `NAMIVG06` delta.
//!
//! Flush input is already ordered by `NodeId`. The common winner table and
//! exact-vector pages are therefore written incrementally. Native-filter
//! observations take the only different order required by the wire
//! (`property,value,ordinal`) and are folded through the shared external pair
//! sorter. The finished object remains exactly the V6 layout consumed by the
//! existing reader; this module changes construction only.

use std::fs::File;
use std::io::{Cursor, Seek, Write};
use std::mem::size_of;

use super::*;
use crate::sst::external_pairs::{ExternalPair, ExternalPairSorter};
use crate::sst::paged_index::create_spool_file;

const DEFAULT_MEMORY_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const MIN_MEMORY_BUDGET_BYTES: usize = 512 * 1024;
const DEFAULT_MAX_FILTER_DISTINCT: usize = 4_096;

/// Explicit aggregate workspace controls for one production VG6 build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorV6ExternalBuildConfig {
    pub memory_budget_bytes: usize,
    pub max_filter_distinct_per_property: usize,
    pub wire: VectorV6BuildOptions,
}

impl Default for VectorV6ExternalBuildConfig {
    fn default() -> Self {
        Self {
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            max_filter_distinct_per_property: DEFAULT_MAX_FILTER_DISTINCT,
            wire: VectorV6BuildOptions::default(),
        }
    }
}

impl VectorV6ExternalBuildConfig {
    pub fn from_env(wire: VectorV6BuildOptions) -> Result<Self> {
        let memory_budget_bytes = match std::env::var(INDEX_BUILD_MEMORY_ENV) {
            Ok(value) => value.parse::<usize>().map_err(|error| {
                Error::precondition(format!(
                    "{INDEX_BUILD_MEMORY_ENV} must be an exact byte count: {error}"
                ))
            })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_MEMORY_BUDGET_BYTES,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(Error::precondition(format!(
                    "{INDEX_BUILD_MEMORY_ENV} is not valid UTF-8"
                )));
            }
        };
        let config = Self {
            memory_budget_bytes,
            wire,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.memory_budget_bytes < MIN_MEMORY_BUDGET_BYTES {
            return Err(Error::precondition(format!(
                "vector v6 external memory budget {} is below minimum {MIN_MEMORY_BUDGET_BYTES}",
                self.memory_budget_bytes
            )));
        }
        if self.max_filter_distinct_per_property == 0 || self.wire.rows_per_page == 0 {
            return Err(Error::precondition(
                "vector v6 external cardinality limits must be positive",
            ));
        }
        Ok(())
    }
}

/// Conservative logical high-water counters for one completed build.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VectorV6ExternalBuildMetrics {
    pub memory_budget_bytes: usize,
    pub effective_mutations: u64,
    pub live_payloads: u64,
    pub suppressions: u64,
    pub vector_pages: u32,
    pub filter_values: u64,
    pub max_page_bytes: usize,
    pub max_filter_value_bytes: usize,
    pub peak_logical_memory_bytes: usize,
}

/// File-backed immutable V6 object ready for bounded multipart upload.
#[derive(Debug)]
pub struct VectorV6ExternalArtifact {
    pub file: File,
    pub len: u64,
    pub output: VectorV6BuildOutput,
    pub metrics: VectorV6ExternalBuildMetrics,
}

#[derive(Debug)]
struct OwnedPageRow {
    ordinal: u64,
    node_id: [u8; 16],
    lsn: u64,
    payload_fingerprint: u64,
    vector: Vec<f32>,
}

/// Incremental builder. `push` requires strictly increasing NodeIds.
pub struct VectorV6ExternalBuilder {
    state: SearchLsmState,
    descriptor: VectorIndexDescriptor,
    context: VectorV6BuildContext,
    config: VectorV6ExternalBuildConfig,
    version_writer: Option<SearchVersionTableWriter<File>>,
    page_file: File,
    pages: Vec<VectorPageRef>,
    page_rows: Vec<OwnedPageRow>,
    filter_sorter: Option<ExternalPairSorter>,
    last_node_id: Option<[u8; 16]>,
    live_count: u64,
    suppress_count: u64,
    live_count_delta: i64,
    reconciled: bool,
    metrics: VectorV6ExternalBuildMetrics,
    poisoned: bool,
}

impl std::fmt::Debug for VectorV6ExternalBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorV6ExternalBuilder")
            .field("index_name", &self.state.index_name)
            .field("live_count", &self.live_count)
            .field("suppress_count", &self.suppress_count)
            .field("pages", &self.pages.len())
            .field("buffered_page_rows", &self.page_rows.len())
            .field("metrics", &self.metrics)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl VectorV6ExternalBuilder {
    pub fn new(
        state: &SearchLsmState,
        descriptor: &VectorIndexDescriptor,
        context: VectorV6BuildContext,
    ) -> Result<Self> {
        Self::with_config(
            state,
            descriptor,
            context,
            VectorV6ExternalBuildConfig::from_env(VectorV6BuildOptions::default())?,
        )
    }

    pub fn with_config(
        state: &SearchLsmState,
        descriptor: &VectorIndexDescriptor,
        context: VectorV6BuildContext,
        config: VectorV6ExternalBuildConfig,
    ) -> Result<Self> {
        Self::with_config_inner(state, descriptor, context, config, None)
    }

    /// Build one already-reconciled DeltaRun.
    ///
    /// `authenticated_live_count_delta` is the exact signed sum from the
    /// selected source footers. It cannot be reconstructed from the collapsed
    /// winner payloads: repeated updates may reduce several source mutations to
    /// one final live record without changing their aggregate statistic.
    pub fn with_config_reconciled(
        state: &SearchLsmState,
        descriptor: &VectorIndexDescriptor,
        context: VectorV6BuildContext,
        authenticated_live_count_delta: i64,
        config: VectorV6ExternalBuildConfig,
    ) -> Result<Self> {
        Self::with_config_inner(
            state,
            descriptor,
            context,
            config,
            Some(authenticated_live_count_delta),
        )
    }

    fn with_config_inner(
        state: &SearchLsmState,
        descriptor: &VectorIndexDescriptor,
        mut context: VectorV6BuildContext,
        mut config: VectorV6ExternalBuildConfig,
        authenticated_live_count_delta: Option<i64>,
    ) -> Result<Self> {
        config.validate()?;

        // Keep one raw page, its compressed output, and the filter sorter
        // inside the aggregate budget. Page cardinality is a physical tuning
        // choice, not a semantic format change, so reducing it is safe.
        let row_bytes = vector_row_len(descriptor.dim)?;
        if row_bytes.saturating_mul(3) > config.memory_budget_bytes {
            return Err(Error::precondition(format!(
                "one vector v6 row requires an estimated {} bytes of page/compression \
                 workspace, above the configured {}-byte budget",
                row_bytes.saturating_mul(3),
                config.memory_budget_bytes
            )));
        }
        let page_budget = (config.memory_budget_bytes / 4).max(row_bytes);
        config.wire.rows_per_page = config
            .wire
            .rows_per_page
            .min((page_budget / row_bytes).max(1));

        let mut validation = Cursor::new(Vec::new());
        validate_build_configuration(
            &mut validation,
            state,
            descriptor,
            &mut context,
            config.wire,
        )?;

        let mut version_file = create_spool_file()?;
        version_file.write_all(MAGIC_V6)?;
        let version_writer = SearchVersionTableWriter::new(version_file)?;
        let page_file = create_spool_file()?;
        let sort_budget = (config.memory_budget_bytes / 4).max(64 * 1024);
        let filter_sorter = ExternalPairSorter::with_memory_limit(sort_budget)?;
        Ok(Self {
            state: state.clone(),
            descriptor: descriptor.clone(),
            context,
            metrics: VectorV6ExternalBuildMetrics {
                memory_budget_bytes: config.memory_budget_bytes,
                ..Default::default()
            },
            page_rows: Vec::with_capacity(config.wire.rows_per_page),
            config,
            version_writer: Some(version_writer),
            page_file,
            pages: Vec::new(),
            filter_sorter: Some(filter_sorter),
            last_node_id: None,
            live_count: 0,
            suppress_count: 0,
            live_count_delta: authenticated_live_count_delta.unwrap_or(0),
            reconciled: authenticated_live_count_delta.is_some(),
            poisoned: false,
        })
    }

    pub fn metrics(&self) -> VectorV6ExternalBuildMetrics {
        self.metrics
    }

    pub fn push(&mut self, mutation: VectorV6Mutation) -> Result<()> {
        if self.poisoned {
            return Err(Error::precondition(
                "vector v6 external builder is poisoned by an earlier push failure",
            ));
        }
        if self.reconciled {
            return Err(Error::precondition(
                "reconciled vector v6 builder requires push_reconciled",
            ));
        }
        let result = self.push_inner(mutation);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn push_inner(&mut self, mutation: VectorV6Mutation) -> Result<()> {
        if mutation.lsn == 0 {
            return Err(Error::precondition(
                "vector v6 external mutation uses reserved LSN zero",
            ));
        }
        if self
            .last_node_id
            .is_some_and(|previous| mutation.node_id <= previous)
        {
            return Err(Error::precondition(
                "vector v6 external builder requires strictly ascending NodeIds",
            ));
        }
        if let Some(before) = &mutation.before {
            validate_payload(
                before,
                &self.descriptor,
                &self.context.complete_filter_properties,
            )?;
        }
        if let Some(after) = &mutation.after {
            validate_payload(
                after,
                &self.descriptor,
                &self.context.complete_filter_properties,
            )?;
        }
        if mutation.before == mutation.after {
            // Preserve the ordinary builder's stream-order contract even when
            // a logically unchanged mutation emits no physical record.
            self.last_node_id = Some(mutation.node_id);
            return Ok(());
        }
        let before_live = mutation.before.is_some();
        let payload_fingerprint = match &mutation.after {
            Some(after) => vector_v6_payload_fingerprint(after)?,
            None => search_suppress_fingerprint(),
        };
        let record = match &mutation.after {
            Some(_) => {
                SearchVersionRecord::live(mutation.node_id, mutation.lsn, payload_fingerprint, 0)
            }
            None => {
                SearchVersionRecord::suppress(mutation.node_id, mutation.lsn, payload_fingerprint)
            }
        };
        match &mutation.after {
            Some(_) => {
                self.live_count_delta = self
                    .live_count_delta
                    .checked_add(if before_live { 0 } else { 1 })
                    .ok_or_else(|| Error::invariant("vector live delta overflows"))?;
            }
            None => {
                self.live_count_delta = self
                    .live_count_delta
                    .checked_sub(if before_live { 1 } else { 0 })
                    .ok_or_else(|| Error::invariant("vector live delta underflows"))?;
            }
        }
        self.push_effect(record, mutation.after)
    }

    /// Append one winner selected from source NAMISV01 tables.
    ///
    /// The output preserves the winner's NodeId, LSN, operation class, and
    /// payload fingerprint exactly. Live ordinals are rewritten because they
    /// are local to the new object. The optional payload is only the body
    /// resolved from the captured Nodes snapshot and must fingerprint to the
    /// selected record; its newer Node LSN is intentionally irrelevant.
    pub fn push_reconciled(
        &mut self,
        record: SearchVersionRecord,
        after: Option<VectorV6Payload>,
    ) -> Result<()> {
        if self.poisoned {
            return Err(Error::precondition(
                "vector v6 external builder is poisoned by an earlier push failure",
            ));
        }
        if !self.reconciled {
            return Err(Error::precondition(
                "ordinary vector v6 builder requires push",
            ));
        }
        let result = self.push_reconciled_inner(record, after);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn push_reconciled_inner(
        &mut self,
        record: SearchVersionRecord,
        after: Option<VectorV6Payload>,
    ) -> Result<()> {
        match (record.operation, &after) {
            (SearchVersionOperation::Live { .. }, Some(payload)) => {
                validate_payload(
                    payload,
                    &self.descriptor,
                    &self.context.complete_filter_properties,
                )?;
                if vector_v6_payload_fingerprint(payload)? != record.payload_fingerprint {
                    return Err(Error::invariant(
                        "reconciled vector payload fingerprint differs from its source winner",
                    ));
                }
            }
            (SearchVersionOperation::Suppress, None)
                if record.payload_fingerprint == search_suppress_fingerprint() => {}
            (SearchVersionOperation::Suppress, None) => {
                return Err(Error::invariant(
                    "reconciled vector suppression has a non-canonical fingerprint",
                ));
            }
            (SearchVersionOperation::Live { .. }, None)
            | (SearchVersionOperation::Suppress, Some(_)) => {
                return Err(Error::invariant(
                    "reconciled vector operation and payload presence disagree",
                ));
            }
        }
        self.push_effect(record, after)
    }

    fn push_effect(
        &mut self,
        record: SearchVersionRecord,
        after: Option<VectorV6Payload>,
    ) -> Result<()> {
        if record.lsn == 0 {
            return Err(Error::precondition(
                "vector v6 external mutation uses reserved LSN zero",
            ));
        }
        if self
            .last_node_id
            .is_some_and(|previous| record.node_id <= previous)
        {
            return Err(Error::precondition(
                "vector v6 external builder requires strictly ascending NodeIds",
            ));
        }

        // Both the common winner-table directory and the page/footer
        // directories are sparse but grow with the number of physical pages.
        // Reject before extending them if their conservative resident estimate
        // would leave the builder's assigned budget.
        let next_mutations = self.metrics.effective_mutations.saturating_add(1);
        let version_pages = next_mutations.div_ceil(512) as usize;
        let vector_pages = self
            .live_count
            .saturating_add(if after.is_some() { 1 } else { 0 })
            .div_ceil(self.config.wire.rows_per_page as u64) as usize;
        let directory_estimate = version_pages
            .saturating_mul(128)
            .saturating_add(vector_pages.saturating_mul(128));
        if directory_estimate > self.config.memory_budget_bytes / 4 {
            return Err(Error::precondition(format!(
                "vector v6 sparse directories require an estimated {directory_estimate} bytes, \
                 above their configured workspace"
            )));
        }

        match after {
            Some(after) => {
                let ordinal = self.live_count;
                self.version_writer
                    .as_mut()
                    .ok_or_else(|| Error::invariant("vector v6 version writer is absent"))?
                    .push(SearchVersionRecord::live(
                        record.node_id,
                        record.lsn,
                        record.payload_fingerprint,
                        ordinal,
                    ))?;
                for (property, value) in &after.filters {
                    let property = self
                        .context
                        .complete_filter_properties
                        .binary_search(property)
                        .map_err(|_| {
                            Error::invariant("vector v6 payload contains an unadvertised filter")
                        })?;
                    let property = u32::try_from(property)
                        .map_err(|_| Error::invariant("vector filter ordinal exceeds u32"))?;
                    let encoded = encode_filter_value(value)?;
                    self.filter_sorter
                        .as_mut()
                        .ok_or_else(|| Error::invariant("vector filter sorter is absent"))?
                        .push(property, &encoded, u128::from(ordinal).to_be_bytes())?;
                }
                self.page_rows.push(OwnedPageRow {
                    ordinal,
                    node_id: record.node_id,
                    lsn: record.lsn,
                    payload_fingerprint: record.payload_fingerprint,
                    vector: after.vector,
                });
                self.live_count = self
                    .live_count
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("vector v6 live count overflows"))?;
                if self.page_rows.len() == self.config.wire.rows_per_page {
                    self.flush_page()?;
                }
            }
            None => {
                self.version_writer
                    .as_mut()
                    .ok_or_else(|| Error::invariant("vector v6 version writer is absent"))?
                    .push(SearchVersionRecord::suppress(
                        record.node_id,
                        record.lsn,
                        record.payload_fingerprint,
                    ))?;
                self.suppress_count = self
                    .suppress_count
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("vector v6 suppress count overflows"))?;
            }
        }
        self.metrics.effective_mutations = self
            .metrics
            .effective_mutations
            .checked_add(1)
            .ok_or_else(|| Error::invariant("vector mutation count overflows"))?;
        self.last_node_id = Some(record.node_id);
        Ok(())
    }

    fn flush_page(&mut self) -> Result<()> {
        if self.page_rows.is_empty() {
            return Ok(());
        }
        let inputs = self
            .page_rows
            .iter()
            .map(|row| VectorPageInput {
                ordinal: row.ordinal,
                node_id: row.node_id,
                lsn: row.lsn,
                payload_fingerprint: row.payload_fingerprint,
                vector: &row.vector,
            })
            .collect::<Vec<_>>();
        let raw_bytes = inputs
            .len()
            .saturating_mul(vector_row_len(self.descriptor.dim)?)
            .saturating_add(VECTOR_PAGE_HEADER_LEN);
        self.metrics.max_page_bytes = self.metrics.max_page_bytes.max(raw_bytes);
        self.metrics.peak_logical_memory_bytes =
            self.metrics.peak_logical_memory_bytes.max(raw_bytes);
        self.pages.push(write_vector_page(
            &mut self.page_file,
            &inputs,
            self.descriptor.dim,
            self.config.wire.compression_level,
        )?);
        self.page_rows.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<Option<VectorV6ExternalArtifact>> {
        if self.poisoned {
            return Err(Error::precondition(
                "cannot finish a poisoned vector v6 external builder",
            ));
        }
        if self.metrics.effective_mutations == 0 {
            return Ok(None);
        }
        self.flush_page()?;
        let version_writer = self
            .version_writer
            .take()
            .ok_or_else(|| Error::invariant("vector v6 version writer is absent"))?;
        let (mut version_file, version_table) = version_writer.finish()?;
        if version_table.live_count != self.live_count
            || version_table.suppress_count != self.suppress_count
        {
            return Err(Error::invariant(
                "vector v6 external version counts diverged",
            ));
        }

        let mut output = create_spool_file()?;
        version_file.rewind()?;
        let version_len = std::io::copy(&mut version_file, &mut output)?;
        self.page_file.rewind()?;
        let page_len = std::io::copy(&mut self.page_file, &mut output)?;
        for page in &mut self.pages {
            page.wire.offset = page
                .wire
                .offset
                .checked_add(version_len)
                .ok_or_else(|| Error::invariant("vector page offset overflows"))?;
        }
        if output.stream_position()?
            != version_len
                .checked_add(page_len)
                .ok_or_else(|| Error::invariant("vector spool length overflows"))?
        {
            return Err(Error::invariant(
                "vector v6 external spool copy length changed",
            ));
        }

        let filter_sorter = self
            .filter_sorter
            .take()
            .ok_or_else(|| Error::invariant("vector filter sorter is absent"))?;
        let filters = write_external_filters(
            &mut output,
            filter_sorter,
            self.live_count,
            &self.context.complete_filter_properties,
            &self.config,
            &mut self.metrics,
        )?;
        let content_xxh3 = content_digest(
            &version_table,
            self.descriptor.dim,
            self.descriptor.metric,
            self.live_count_delta,
            &self.pages,
            &filters,
        )?;
        let segment = SearchSegmentRef {
            sst_id: self.context.sst_id,
            role: SearchSegmentRole::Delta,
            format: SearchSegmentFormat::VectorV6,
            payload: SearchSegmentPayload::Complete,
            event_ranges: self.context.event_ranges,
            min_lsn: version_table.min_lsn,
            max_lsn: version_table.max_lsn,
            mutation_count: version_table.record_count,
            live_payload_count: version_table.live_count,
            suppress_count: version_table.suppress_count,
            content_xxh3,
            complete_filter_properties: self.context.complete_filter_properties,
            stats: SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(self.live_count_delta),
            },
            equal_lsn_conflict_count: 0,
        };
        let binding = SearchSegmentWireBinding::new(&self.state, &segment, version_table.clone())?;
        let footer = Footer {
            footer_version: FOOTER_VERSION,
            mode: VectorV6Mode::FlatExact,
            binding,
            dim: self.descriptor.dim,
            metric: self.descriptor.metric,
            live_count_delta: self.live_count_delta,
            pages: self.pages,
            filters,
        };
        let footer_bytes = serialize_bounded(&footer, MAX_FOOTER_BYTES, "vector v6 footer")?;
        let directory_bytes = footer_bytes.len();
        if directory_bytes > self.config.memory_budget_bytes / 2 {
            return Err(Error::precondition(format!(
                "vector v6 footer requires {directory_bytes} bytes, above its configured workspace"
            )));
        }
        self.metrics.peak_logical_memory_bytes =
            self.metrics.peak_logical_memory_bytes.max(directory_bytes);
        output.write_all(&footer_bytes)?;
        output.write_all(TRAILER_MAGIC)?;
        output.write_all(&(footer_bytes.len() as u64).to_le_bytes())?;
        output.write_all(&crc32fast::hash(&footer_bytes).to_le_bytes())?;
        output.sync_data()?;
        let object_len = output.stream_position()?;
        output.rewind()?;

        self.metrics.live_payloads = self.live_count;
        self.metrics.suppressions = self.suppress_count;
        self.metrics.vector_pages = u32::try_from(footer.pages.len())
            .map_err(|_| Error::invariant("vector page count exceeds u32"))?;
        if self.metrics.peak_logical_memory_bytes > self.config.memory_budget_bytes {
            return Err(Error::precondition(format!(
                "vector v6 external build exceeded its {}-byte memory contract",
                self.config.memory_budget_bytes
            )));
        }
        Ok(Some(VectorV6ExternalArtifact {
            file: output,
            len: object_len,
            output: VectorV6BuildOutput {
                segment,
                object_len,
                page_count: self.metrics.vector_pages,
                version_table,
            },
            metrics: self.metrics,
        }))
    }
}

fn encode_filter_value(value: &SearchFilterValue) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    match value {
        SearchFilterValue::Bool(value) => {
            encoded.extend_from_slice(&[0, u8::from(*value)]);
        }
        SearchFilterValue::I64(value) => {
            encoded.push(1);
            encoded.extend_from_slice(&((*value as u64) ^ (1_u64 << 63)).to_be_bytes());
        }
        SearchFilterValue::F64Bits(value) => {
            encoded.push(2);
            encoded.extend_from_slice(&value.to_be_bytes());
        }
        SearchFilterValue::String(value) => {
            encoded.push(3);
            encoded.extend_from_slice(value.as_bytes());
        }
        SearchFilterValue::Bytes(value) => {
            encoded.push(4);
            encoded.extend_from_slice(value);
        }
        SearchFilterValue::Date(value) => {
            encoded.push(5);
            encoded.extend_from_slice(&((*value as u32) ^ (1_u32 << 31)).to_be_bytes());
        }
        SearchFilterValue::DateTime(value) => {
            encoded.push(6);
            encoded.extend_from_slice(&((*value as u64) ^ (1_u64 << 63)).to_be_bytes());
        }
    }
    Ok(encoded)
}

fn decode_filter_value(bytes: &[u8]) -> Result<SearchFilterValue> {
    let (&tag, body) = bytes
        .split_first()
        .ok_or_else(|| Error::invariant("vector filter value is empty"))?;
    match tag {
        0 if body == [0] => Ok(SearchFilterValue::Bool(false)),
        0 if body == [1] => Ok(SearchFilterValue::Bool(true)),
        1 if body.len() == 8 => {
            let ordered = u64::from_be_bytes(body.try_into().expect("checked i64 bytes"));
            Ok(SearchFilterValue::I64((ordered ^ (1_u64 << 63)) as i64))
        }
        2 if body.len() == 8 => Ok(SearchFilterValue::F64Bits(u64::from_be_bytes(
            body.try_into().expect("checked f64 bits"),
        ))),
        3 => Ok(SearchFilterValue::String(
            String::from_utf8(body.to_vec()).map_err(|error| {
                Error::invariant(format!("vector filter string is not UTF-8: {error}"))
            })?,
        )),
        4 => Ok(SearchFilterValue::Bytes(body.to_vec())),
        5 if body.len() == 4 => {
            let ordered = u32::from_be_bytes(body.try_into().expect("checked date bytes"));
            Ok(SearchFilterValue::Date((ordered ^ (1_u32 << 31)) as i32))
        }
        6 if body.len() == 8 => {
            let ordered = u64::from_be_bytes(body.try_into().expect("checked datetime bytes"));
            Ok(SearchFilterValue::DateTime(
                (ordered ^ (1_u64 << 63)) as i64,
            ))
        }
        _ => Err(Error::invariant(
            "vector filter value has an invalid ordered encoding",
        )),
    }
}

fn pair_ordinal(pair: &ExternalPair) -> Result<u64> {
    let raw = u128::from_be_bytes(pair.id);
    u64::try_from(raw).map_err(|_| Error::invariant("vector filter ordinal exceeds u64"))
}

fn write_external_filters<W: Write + Seek>(
    writer: &mut W,
    sorter: ExternalPairSorter,
    row_count: u64,
    properties: &[String],
    config: &VectorV6ExternalBuildConfig,
    metrics: &mut VectorV6ExternalBuildMetrics,
) -> Result<Vec<FilterBlockRef>> {
    let mut merge = sorter.finish()?;
    let dense_bytes = bitmap_words(row_count)?
        .checked_mul(size_of::<u64>())
        .ok_or_else(|| Error::invariant("vector filter bitmap length overflows"))?;
    let per_value_limit = (config.memory_budget_bytes / 4).max(4 * 1024);
    if dense_bytes > per_value_limit {
        return Err(Error::precondition(format!(
            "one vector native-filter bitmap requires {dense_bytes} bytes, above its {per_value_limit}-byte workspace"
        )));
    }

    let mut result = properties
        .iter()
        .map(|property| FilterBlockRef {
            property: property.clone(),
            row_count,
            values: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut current: Option<(u32, Vec<u8>, AdaptiveOrdinals)> = None;
    while let Some(pair) = merge.next_pair()? {
        let ordinal = pair_ordinal(&pair)?;
        if ordinal >= row_count {
            return Err(Error::invariant(
                "vector filter observation leaves live ordinals",
            ));
        }
        let same = current
            .as_ref()
            .is_some_and(|(property, key, _)| *property == pair.property && *key == pair.key);
        if !same {
            if let Some((property, key, ordinals)) = current.take() {
                finish_filter_value(
                    writer,
                    &mut result,
                    property,
                    key,
                    ordinals,
                    config,
                    metrics,
                )?;
            }
            current = Some((
                pair.property,
                pair.key,
                AdaptiveOrdinals::new(row_count, dense_bytes),
            ));
        }
        current
            .as_mut()
            .expect("current filter value installed")
            .2
            .push(ordinal)?;
    }
    if let Some((property, key, ordinals)) = current {
        finish_filter_value(
            writer,
            &mut result,
            property,
            key,
            ordinals,
            config,
            metrics,
        )?;
    }
    Ok(result)
}

/// Streaming counterpart of [`encode_filter_posting`]. It must produce the
/// byte-identical posting: the segment's `content_xxh3` covers these bytes,
/// and the in-memory builder computes them through the canonical encoder. The
/// canonical sparse→dense flip compares *encoded varint bytes* against the
/// dense bitmap length, so this accumulator encodes deltas incrementally and
/// flips at exactly that point — sizing the decision by resident `u64`s
/// (as this once did) flipped to Dense far earlier and forked the fingerprint.
#[derive(Debug)]
enum AdaptiveOrdinals {
    Sparse {
        /// Canonical delta-varint stream of everything pushed so far.
        encoded: Vec<u8>,
        cardinality: u64,
        last_ordinal: Option<u64>,
        dense_bytes: usize,
        row_count: u64,
    },
    Dense {
        bitmap: Vec<u8>,
        count: u64,
        row_count: u64,
    },
}

impl AdaptiveOrdinals {
    fn new(row_count: u64, dense_bytes: usize) -> Self {
        Self::Sparse {
            encoded: Vec::new(),
            cardinality: 0,
            last_ordinal: None,
            dense_bytes,
            row_count,
        }
    }

    fn push(&mut self, ordinal: u64) -> Result<()> {
        match self {
            Self::Sparse {
                encoded,
                cardinality,
                last_ordinal,
                dense_bytes,
                row_count,
            } => {
                if last_ordinal.is_some_and(|previous| previous >= ordinal) {
                    return Err(Error::invariant(
                        "vector filter ordinals are not strictly increasing",
                    ));
                }
                let delta = last_ordinal.map_or(ordinal, |previous| ordinal - previous);
                encode_u64_varint(delta, encoded);
                *last_ordinal = Some(ordinal);
                *cardinality = cardinality
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("vector filter count overflows"))?;
                if encoded.len() >= *dense_bytes {
                    // Rebuild the bitmap from the canonical stream itself so
                    // the flip cannot drift from what was actually encoded.
                    let mut bitmap = vec![0_u8; *dense_bytes];
                    let mut cursor = 0usize;
                    let mut previous = 0u64;
                    let mut first = true;
                    while cursor < encoded.len() {
                        let delta = decode_u64_varint(encoded, &mut cursor)?;
                        let ordinal = if first { delta } else { previous + delta };
                        set_dense_filter_bit(&mut bitmap, ordinal, *row_count)?;
                        previous = ordinal;
                        first = false;
                    }
                    let count = *cardinality;
                    *self = Self::Dense {
                        bitmap,
                        count,
                        row_count: *row_count,
                    };
                }
            }
            Self::Dense {
                bitmap,
                count,
                row_count,
            } => {
                set_dense_filter_bit(bitmap, ordinal, *row_count)?;
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| Error::invariant("vector filter count overflows"))?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<(FilterPostingEncoding, Vec<u8>, u64)> {
        match self {
            Self::Sparse {
                encoded,
                cardinality,
                ..
            } => {
                if cardinality == 0 {
                    return Err(Error::invariant(
                        "vector v6 filter posting input is inconsistent",
                    ));
                }
                Ok((
                    FilterPostingEncoding::SparseDeltaVarint,
                    encoded,
                    cardinality,
                ))
            }
            Self::Dense { bitmap, count, .. } => {
                Ok((FilterPostingEncoding::DenseBitmap, bitmap, count))
            }
        }
    }
}

fn finish_filter_value<W: Write + Seek>(
    writer: &mut W,
    filters: &mut [FilterBlockRef],
    property: u32,
    key: Vec<u8>,
    ordinals: AdaptiveOrdinals,
    config: &VectorV6ExternalBuildConfig,
    metrics: &mut VectorV6ExternalBuildMetrics,
) -> Result<()> {
    let property = usize::try_from(property)
        .map_err(|_| Error::invariant("vector filter property exceeds usize"))?;
    let filter = filters
        .get_mut(property)
        .ok_or_else(|| Error::invariant("vector filter property ordinal is out of bounds"))?;
    if filter.values.len() == config.max_filter_distinct_per_property {
        return Err(Error::precondition(format!(
            "vector native filter '{}' exceeds {} distinct values",
            filter.property, config.max_filter_distinct_per_property
        )));
    }
    let value = decode_filter_value(&key)?;
    let (encoding, raw, cardinality) = ordinals.finish()?;
    metrics.max_filter_value_bytes = metrics.max_filter_value_bytes.max(raw.len());
    metrics.peak_logical_memory_bytes = metrics.peak_logical_memory_bytes.max(raw.len());
    filter.values.push(FilterValueRef {
        value,
        cardinality,
        encoding,
        wire: write_compressed_block(
            writer,
            &raw,
            config.wire.compression_level,
            "vector external filter posting block",
        )?,
    });
    metrics.filter_values = metrics
        .filter_values
        .checked_add(1)
        .ok_or_else(|| Error::invariant("vector filter value count overflows"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{VectorMetric, VectorQuantization};
    use crate::search_lsm::{SearchEventRange, SearchLsmKind, SearchLsmStatus};
    use crate::sst::search_delta::SearchFilterValue;
    use std::collections::BTreeMap;
    use std::io::Read;

    fn state() -> SearchLsmState {
        SearchLsmState {
            index_name: "emb".into(),
            kind: SearchLsmKind::Vector,
            catalog_signature: "sig".into(),
            generation_id: Uuid::from_u128(1),
            status: SearchLsmStatus::Active,
            next_event_seq: 0,
            ..Default::default()
        }
    }

    fn descriptor() -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            name: "emb".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 8,
            l_build: 16,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        }
    }

    fn payload(x: f32, active: bool) -> VectorV6Payload {
        VectorV6Payload {
            vector: vec![x, 1.0 - x],
            filters: BTreeMap::from([("active".into(), SearchFilterValue::Bool(active))]),
        }
    }

    #[test]
    fn streaming_builder_matches_in_memory_segment_semantics() {
        let state = state();
        let descriptor = descriptor();
        let context = VectorV6BuildContext {
            sst_id: Uuid::from_u128(2),
            event_ranges: vec![SearchEventRange::new(0, 1)],
            complete_filter_properties: vec!["active".into()],
        };
        let mutations = vec![
            VectorV6Mutation {
                node_id: 1_u128.to_be_bytes(),
                lsn: 10,
                before: None,
                after: Some(payload(0.1, true)),
            },
            VectorV6Mutation {
                node_id: 2_u128.to_be_bytes(),
                lsn: 11,
                before: Some(payload(0.2, false)),
                after: None,
            },
        ];
        let mut builder =
            VectorV6ExternalBuilder::new(&state, &descriptor, context.clone()).unwrap();
        for mutation in mutations.clone() {
            builder.push(mutation).unwrap();
        }
        let mut external = builder.finish().unwrap().unwrap();
        let memory = build_delta_v6(
            &state,
            &descriptor,
            context,
            mutations,
            VectorV6BuildOptions::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(external.output.segment, memory.output.segment);
        assert_eq!(
            external.output.version_table.content_xxh3,
            memory.output.version_table.content_xxh3
        );
        let mut body = Vec::new();
        external.file.read_to_end(&mut body).unwrap();
        assert_eq!(body.len() as u64, external.len);
        assert_eq!(&body[..8], MAGIC_V6);
        assert!(external.metrics.peak_logical_memory_bytes <= external.metrics.memory_budget_bytes);
    }

    #[test]
    fn no_op_returns_none_and_input_must_be_ordered() {
        let state = state();
        let descriptor = descriptor();
        let context = VectorV6BuildContext {
            sst_id: Uuid::from_u128(3),
            event_ranges: vec![SearchEventRange::new(0, 1)],
            complete_filter_properties: vec!["active".into()],
        };
        let same = payload(0.4, true);
        let mut builder = VectorV6ExternalBuilder::new(&state, &descriptor, context).unwrap();
        builder
            .push(VectorV6Mutation {
                node_id: 2_u128.to_be_bytes(),
                lsn: 10,
                before: Some(same.clone()),
                after: Some(same),
            })
            .unwrap();
        assert!(builder.finish().unwrap().is_none());

        let context = VectorV6BuildContext {
            sst_id: Uuid::from_u128(4),
            event_ranges: vec![SearchEventRange::new(0, 1)],
            complete_filter_properties: vec!["active".into()],
        };
        let mut unordered = VectorV6ExternalBuilder::new(&state, &descriptor, context).unwrap();
        unordered
            .push(VectorV6Mutation {
                node_id: 2_u128.to_be_bytes(),
                lsn: 10,
                before: None,
                after: Some(payload(0.2, true)),
            })
            .unwrap();
        assert!(unordered
            .push(VectorV6Mutation {
                node_id: 1_u128.to_be_bytes(),
                lsn: 11,
                before: None,
                after: Some(payload(0.1, true)),
            })
            .is_err());
    }

    #[test]
    fn reconciled_builder_preserves_winner_identity_and_authenticated_stats() {
        let state = state();
        let descriptor = descriptor();
        let context = VectorV6BuildContext {
            sst_id: Uuid::from_u128(5),
            event_ranges: vec![SearchEventRange::new(2, 7)],
            complete_filter_properties: vec!["active".into()],
        };
        let live_payload = payload(0.25, true);
        let live_fingerprint = vector_v6_payload_fingerprint(&live_payload).unwrap();
        let mut builder = VectorV6ExternalBuilder::with_config_reconciled(
            &state,
            &descriptor,
            context,
            -7,
            VectorV6ExternalBuildConfig::default(),
        )
        .unwrap();
        builder
            .push_reconciled(
                SearchVersionRecord::live(1_u128.to_be_bytes(), 31, live_fingerprint, 99_999),
                Some(live_payload),
            )
            .unwrap();
        builder
            .push_reconciled(
                SearchVersionRecord::suppress(
                    2_u128.to_be_bytes(),
                    44,
                    search_suppress_fingerprint(),
                ),
                None,
            )
            .unwrap();
        let artifact = builder.finish().unwrap().unwrap();
        assert_eq!(artifact.output.version_table.record_count, 2);
        assert_eq!(artifact.output.version_table.min_lsn, 31);
        assert_eq!(artifact.output.version_table.max_lsn, 44);
        assert_eq!(
            artifact.output.segment.stats,
            SearchSegmentStats::Vector {
                live_count: SearchStatValue::Delta(-7)
            }
        );
        assert_eq!(artifact.output.segment.live_payload_count, 1);
        assert_eq!(artifact.output.segment.suppress_count, 1);

        let context = VectorV6BuildContext {
            sst_id: Uuid::from_u128(6),
            event_ranges: vec![SearchEventRange::new(2, 7)],
            complete_filter_properties: vec!["active".into()],
        };
        let mut mismatched = VectorV6ExternalBuilder::with_config_reconciled(
            &state,
            &descriptor,
            context,
            0,
            VectorV6ExternalBuildConfig::default(),
        )
        .unwrap();
        assert!(mismatched
            .push_reconciled(
                SearchVersionRecord::live(1_u128.to_be_bytes(), 9, 123, 0),
                Some(payload(0.5, false)),
            )
            .is_err());
    }
}
