//! Bounded-memory external builder for `NAMIFT04` deltas.
//!
//! The caller streams NodeId-sorted before/after images. Token occurrences and
//! signed df events are folded into checksummed external-sort runs. The final
//! run is consumed one term at a time, so neither the corpus postings nor all
//! term entries coexist in RAM. Complete native filters travel through the same
//! external sort and are emitted as independently range-readable adaptive
//! postings: delta-varint ordinals while sparse, bitmaps only when dense.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};

use bincode::Options;
use xxhash_rust::xxh3::Xxh3;

use super::{
    bincode_options, bitmap_words, content_digest, decode_u64_varint, encode_doc_record,
    encode_posting_block, encode_u64_varint, filter_value_resident_bytes, non_zero_digest,
    prepare_payload, prepared_payload_fingerprint, search_suppress_fingerprint, serialize_bounded,
    text_segment_stats, validate_build_configuration, write_compressed_block, DictionaryBlockRef,
    DocRecord, DocTableRef, FilterBlockRef, FilterPostingEncoding, FilterValueRef, Footer, Posting,
    PostingBlockRef, RegionRef, SearchFilterValue, SearchLsmState, SearchSegmentPayload,
    SearchSegmentRef, SearchSegmentRole, SearchVersionOperation, SearchVersionRecord,
    SearchVersionTableWriter, TermEntry, TextV4BuildContext, TextV4BuildOptions, TextV4BuildOutput,
    TextV4Mutation, TextV4Payload, FOOTER_VERSION, MAGIC_V4, MAX_FOOTER_BYTES, MAX_RAW_BLOCK_BYTES,
    POSTINGS_REGION_DOMAIN, TRAILER_LEN, TRAILER_MAGIC,
};
use crate::error::{Error, Result};
use crate::search_lsm::SearchSegmentFormat;
use crate::sst::search_delta::SearchSegmentWireBinding;

const DEFAULT_MEMORY_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const MIN_MEMORY_BUDGET_BYTES: usize = 256 * 1024;
const DEFAULT_MAX_FILTER_DISTINCT: usize = 4_096;
const IO_BUFFER_BYTES: usize = 16 * 1024;
const RUN_MAGIC: &[u8; 8] = b"NFT4RUN1";
const RUN_HEADER_LEN: u64 = 8 + 8 + 8 + 4;
const TERM_SPOOL_HEADER_LEN: u64 = 4 + 4;
const RECORD_FIXED_BYTES: usize = 1 + 8 + 4 + 4 + 1;
const FILTER_KEY_PREFIX: &str = "\0filter:";
const INDEX_BUILD_MEMORY_ENV: &str = "NAMIDB_INDEX_BUILD_MEMORY_BYTES";
const INDEX_BUILD_SPOOL_DIR_ENV: &str = "NAMIDB_INDEX_BUILD_SPOOL_DIR";
const COMPACTION_SPOOL_DIR_ENV: &str = "NAMIDB_COMPACTION_SPOOL_DIR";
const SHARED_SPOOL_DIR_ENV: &str = "NAMIDB_SPOOL_DIR";

/// Explicit memory/scratch controls for a production FT4 delta build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextV4ExternalBuildConfig {
    pub memory_budget_bytes: usize,
    pub spool_directory: Option<PathBuf>,
    pub max_filter_distinct_per_property: usize,
    pub wire: TextV4BuildOptions,
}

impl Default for TextV4ExternalBuildConfig {
    fn default() -> Self {
        Self {
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            spool_directory: None,
            max_filter_distinct_per_property: DEFAULT_MAX_FILTER_DISTINCT,
            wire: TextV4BuildOptions::default(),
        }
    }
}

impl TextV4ExternalBuildConfig {
    pub fn from_env(wire: TextV4BuildOptions) -> Result<Self> {
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
        let spool_directory = std::env::var_os(INDEX_BUILD_SPOOL_DIR_ENV)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                std::env::var_os(COMPACTION_SPOOL_DIR_ENV).filter(|value| !value.is_empty())
            })
            .or_else(|| std::env::var_os(SHARED_SPOOL_DIR_ENV).filter(|value| !value.is_empty()))
            .map(PathBuf::from);
        let config = Self {
            memory_budget_bytes,
            spool_directory,
            wire,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.memory_budget_bytes < MIN_MEMORY_BUDGET_BYTES {
            return Err(Error::precondition(format!(
                "text v4 external memory budget {} is below minimum {MIN_MEMORY_BUDGET_BYTES}",
                self.memory_budget_bytes
            )));
        }
        if self.max_filter_distinct_per_property == 0
            || self.wire.postings_per_block == 0
            || self.wire.terms_per_dictionary_block == 0
        {
            return Err(Error::precondition(
                "text v4 external cardinality limits must be positive",
            ));
        }
        if let Some(directory) = self.spool_directory.as_deref() {
            validate_spool_directory(directory)?;
        }
        Ok(())
    }

    fn sort_budget(&self) -> usize {
        self.memory_budget_bytes / 2
    }

    fn block_budget(&self) -> usize {
        (self.memory_budget_bytes / 8).max(4 * 1024)
    }

    fn directory_budget(&self) -> usize {
        self.memory_budget_bytes / 8
    }
}

/// Conservative logical allocation/I/O counters from one completed build.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TextV4ExternalBuildMetrics {
    pub memory_budget_bytes: usize,
    pub max_sort_buffer_bytes: usize,
    pub max_filter_bytes: usize,
    pub max_posting_block_bytes: usize,
    pub max_dictionary_block_bytes: usize,
    pub max_directory_bytes: usize,
    pub peak_logical_memory_bytes: usize,
    pub initial_run_count: u64,
    pub run_merge_count: u64,
    pub spool_bytes_written: u64,
    pub filter_spool_bytes: u64,
    pub filter_value_count: u64,
}

/// Authenticated coarse statistics for a reconciled FT4 DeltaRun.
///
/// These are sums over the source runs being replaced. They deliberately do
/// not describe the final after-images streamed to the builder: one final
/// winner can summarize several source events whose net corpus effect differs
/// from a fresh insertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconciledTextDeltaStats {
    pub doc_count_delta: i64,
    pub total_len_delta: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExternalBuildMode {
    Derived,
    Reconciled(ReconciledTextDeltaStats),
}

impl TextV4ExternalBuildMetrics {
    fn observe_push(&mut self, sort_bytes: usize) {
        self.max_sort_buffer_bytes = self.max_sort_buffer_bytes.max(sort_bytes);
        self.peak_logical_memory_bytes = self.peak_logical_memory_bytes.max(sort_bytes);
    }

    fn observe_finish(&mut self, workspace_bytes: usize) {
        self.peak_logical_memory_bytes = self.peak_logical_memory_bytes.max(workspace_bytes);
    }
}

/// File-backed, unlinked FT4 artifact ready for multipart/object upload.
#[derive(Debug)]
pub struct TextV4ExternalArtifact {
    pub file: File,
    pub len: u64,
    pub output: TextV4BuildOutput,
    pub metrics: TextV4ExternalBuildMetrics,
}

/// Incremental external-sort FT4 builder.
pub struct TextV4ExternalBuilder {
    state: SearchLsmState,
    role: SearchSegmentRole,
    mode: ExternalBuildMode,
    context: TextV4BuildContext,
    config: TextV4ExternalBuildConfig,
    spool: SpoolFactory,
    version_writer: Option<SearchVersionTableWriter<File>>,
    documents: File,
    document_hasher: Xxh3,
    records: Vec<TermRecord>,
    record_term_bytes: usize,
    sort_buffer_bytes: usize,
    levels: Vec<Option<RunFile>>,
    last_node_id: Option<[u8; 16]>,
    effective_count: u64,
    live_count: u64,
    suppress_count: u64,
    min_lsn: u64,
    max_lsn: u64,
    delta_docs: i64,
    delta_total_len: i64,
    final_total_len: u64,
    last_term_delta: Option<String>,
    metrics: TextV4ExternalBuildMetrics,
    poisoned: bool,
}

impl std::fmt::Debug for TextV4ExternalBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TextV4ExternalBuilder")
            .field("index_name", &self.state.index_name)
            .field("role", &self.role)
            .field("mode", &self.mode)
            .field("effective_count", &self.effective_count)
            .field("live_count", &self.live_count)
            .field("suppress_count", &self.suppress_count)
            .field("sort_buffer_bytes", &self.sort_buffer_bytes)
            .field("live_run_levels", &self.levels.len())
            .field("metrics", &self.metrics)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl TextV4ExternalBuilder {
    /// Create a signed delta builder.
    pub fn new(state: &SearchLsmState, context: TextV4BuildContext) -> Result<Self> {
        Self::with_config(
            state,
            context,
            TextV4ExternalBuildConfig::from_env(TextV4BuildOptions::default())?,
        )
    }

    pub fn with_config(
        state: &SearchLsmState,
        context: TextV4BuildContext,
        config: TextV4ExternalBuildConfig,
    ) -> Result<Self> {
        Self::with_config_for_role(
            state,
            context,
            config,
            SearchSegmentRole::Delta,
            ExternalBuildMode::Derived,
        )
    }

    /// Create a DeltaRun builder whose signed corpus statistics are the
    /// authenticated sums of the source runs being replaced.
    ///
    /// Final winner records and payloads must be supplied with
    /// [`Self::push_reconciled`], while exact signed per-term sums are supplied
    /// with [`Self::push_term_delta`]. Neither statistic is inferred from the
    /// final after-images.
    pub fn with_config_reconciled(
        state: &SearchLsmState,
        context: TextV4BuildContext,
        config: TextV4ExternalBuildConfig,
        stats: ReconciledTextDeltaStats,
    ) -> Result<Self> {
        Self::with_config_for_role(
            state,
            context,
            config,
            SearchSegmentRole::Delta,
            ExternalBuildMode::Reconciled(stats),
        )
    }

    /// Create an authoritative base builder for physical Search-LSM
    /// compaction. Inputs must be live after-images with no `before` payload.
    pub fn new_base(state: &SearchLsmState, context: TextV4BuildContext) -> Result<Self> {
        Self::with_config_base(
            state,
            context,
            TextV4ExternalBuildConfig::from_env(TextV4BuildOptions::default())?,
        )
    }

    /// Create an authoritative base builder with explicit memory/scratch
    /// controls.
    pub fn with_config_base(
        state: &SearchLsmState,
        context: TextV4BuildContext,
        config: TextV4ExternalBuildConfig,
    ) -> Result<Self> {
        Self::with_config_for_role(
            state,
            context,
            config,
            SearchSegmentRole::Base,
            ExternalBuildMode::Derived,
        )
    }

    fn with_config_for_role(
        state: &SearchLsmState,
        mut context: TextV4BuildContext,
        config: TextV4ExternalBuildConfig,
        role: SearchSegmentRole,
        mode: ExternalBuildMode,
    ) -> Result<Self> {
        config.validate()?;
        if role == SearchSegmentRole::Base && matches!(mode, ExternalBuildMode::Reconciled(_)) {
            return Err(Error::precondition(
                "text v4 reconciled statistics are valid only for DeltaRun output",
            ));
        }
        let mut validation_cursor = Cursor::new(Vec::new());
        validate_build_configuration(&mut validation_cursor, state, &mut context, config.wire)?;
        let spool = SpoolFactory::new(config.spool_directory.clone());
        let version_file = spool.create()?;
        let version_writer = SearchVersionTableWriter::new(version_file)?;
        let documents = spool.create()?;
        Ok(Self {
            state: state.clone(),
            role,
            mode,
            context,
            metrics: TextV4ExternalBuildMetrics {
                memory_budget_bytes: config.memory_budget_bytes,
                ..Default::default()
            },
            config,
            spool,
            version_writer: Some(version_writer),
            documents,
            document_hasher: Xxh3::new(),
            records: Vec::new(),
            record_term_bytes: 0,
            sort_buffer_bytes: 0,
            levels: Vec::new(),
            last_node_id: None,
            effective_count: 0,
            live_count: 0,
            suppress_count: 0,
            min_lsn: u64::MAX,
            max_lsn: 0,
            delta_docs: match mode {
                ExternalBuildMode::Derived => 0,
                ExternalBuildMode::Reconciled(stats) => stats.doc_count_delta,
            },
            delta_total_len: match mode {
                ExternalBuildMode::Derived => 0,
                ExternalBuildMode::Reconciled(stats) => stats.total_len_delta,
            },
            final_total_len: 0,
            last_term_delta: None,
            poisoned: false,
        })
    }

    pub fn metrics(&self) -> TextV4ExternalBuildMetrics {
        self.metrics
    }

    pub fn push(&mut self, mutation: TextV4Mutation) -> Result<()> {
        if self.poisoned {
            return Err(Error::precondition(
                "text v4 external builder is poisoned by an earlier push failure",
            ));
        }
        let result = self.push_inner(mutation);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Append one final reconciled winner to a DeltaRun.
    ///
    /// The source payload ordinal is segment-local and is replaced with this
    /// output's ordinal. NodeId, LSN, logical operation and authenticated
    /// payload fingerprint are preserved exactly.
    pub fn push_reconciled(
        &mut self,
        record: SearchVersionRecord,
        after: Option<TextV4Payload>,
    ) -> Result<()> {
        if self.poisoned {
            return Err(Error::precondition(
                "text v4 external builder is poisoned by an earlier push failure",
            ));
        }
        let result = self.push_reconciled_inner(record, after);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Append one exact authenticated per-term sum to a DeltaRun.
    ///
    /// Terms must be supplied once in strict lexical order. Records share the
    /// existing bounded external sorter with postings, so calls may be
    /// interleaved with [`Self::push_reconciled`] without retaining the
    /// vocabulary in memory.
    pub fn push_term_delta(&mut self, term: String, delta_df: i64) -> Result<()> {
        if self.poisoned {
            return Err(Error::precondition(
                "text v4 external builder is poisoned by an earlier push failure",
            ));
        }
        let result = self.push_term_delta_inner(term, delta_df);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn push_inner(&mut self, mutation: TextV4Mutation) -> Result<()> {
        if !matches!(self.mode, ExternalBuildMode::Derived) {
            return Err(Error::precondition(
                "text v4 reconciled DeltaRun requires push_reconciled",
            ));
        }
        if mutation.lsn == 0 {
            return Err(Error::precondition(
                "text v4 external mutation uses reserved LSN zero",
            ));
        }
        if self
            .last_node_id
            .is_some_and(|previous| mutation.node_id <= previous)
        {
            return Err(Error::precondition(
                "text v4 external builder requires strictly ascending NodeIds",
            ));
        }
        if self.role == SearchSegmentRole::Base
            && (mutation.before.is_some() || mutation.after.is_none())
        {
            return Err(Error::precondition(
                "text v4 base requires before=None and one live after-image",
            ));
        }
        self.last_node_id = Some(mutation.node_id);
        self.preflight_text(&mutation)?;
        let before = mutation.before.as_ref().map(prepare_payload).transpose()?;
        let after = mutation.after.as_ref().map(prepare_payload).transpose()?;
        self.validate_filter_keys(before.as_ref())?;
        self.validate_filter_keys(after.as_ref())?;
        if before == after {
            return Ok(());
        }

        let payload_fingerprint = match &after {
            Some(payload) => prepared_payload_fingerprint(payload)?,
            None => search_suppress_fingerprint(),
        };
        let ordinal = self.live_count;
        let record = match &after {
            Some(_) => SearchVersionRecord::live(
                mutation.node_id,
                mutation.lsn,
                payload_fingerprint,
                ordinal,
            ),
            None => {
                SearchVersionRecord::suppress(mutation.node_id, mutation.lsn, payload_fingerprint)
            }
        };
        self.version_writer
            .as_mut()
            .ok_or_else(|| Error::invariant("text v4 version writer is absent"))?
            .push(record)?;

        let event = self.effective_count;
        if let Some(before) = &before {
            let unique = before
                .tokens
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            for term in unique {
                self.push_record(TermRecord::df(term.to_owned(), event, -1))?;
            }
        }
        if let Some(after) = &after {
            let doc_len = u32::try_from(after.tokens.len())
                .map_err(|_| Error::precondition("text v4 document length exceeds u32"))?;
            let doc_record = DocRecord {
                node_id: mutation.node_id,
                lsn: mutation.lsn,
                payload_fingerprint,
                doc_len,
            };
            let encoded = encode_doc_record(doc_record);
            self.documents.write_all(&encoded)?;
            self.document_hasher.update(&encoded);
            let unique = after
                .tokens
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            for term in unique {
                self.push_record(TermRecord::df(term.to_owned(), event, 1))?;
            }
            for (position, term) in after.tokens.iter().enumerate() {
                self.push_record(TermRecord::occurrence(
                    term.clone(),
                    ordinal,
                    u32::try_from(position)
                        .map_err(|_| Error::precondition("text v4 token position exceeds u32"))?,
                    doc_len,
                ))?;
            }
            self.push_filters(ordinal, &after.filters)?;
            self.live_count = self
                .live_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 live count overflows"))?;
        } else {
            self.suppress_count = self
                .suppress_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 suppress count overflows"))?;
        }

        let before_len = before
            .as_ref()
            .map(|payload| payload.tokens.len())
            .unwrap_or(0);
        let after_len = after
            .as_ref()
            .map(|payload| payload.tokens.len())
            .unwrap_or(0);
        self.delta_docs = self
            .delta_docs
            .checked_add(if after.is_some() { 1 } else { 0 })
            .and_then(|value| value.checked_sub(if before.is_some() { 1 } else { 0 }))
            .ok_or_else(|| Error::invariant("text v4 signed document count overflows"))?;
        self.delta_total_len = self
            .delta_total_len
            .checked_add(
                i64::try_from(after_len)
                    .map_err(|_| Error::invariant("text v4 after length exceeds i64"))?,
            )
            .and_then(|value| {
                i64::try_from(before_len)
                    .ok()
                    .and_then(|before| value.checked_sub(before))
            })
            .ok_or_else(|| Error::invariant("text v4 signed total length overflows"))?;
        self.effective_count = self
            .effective_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text v4 mutation count overflows"))?;
        self.min_lsn = self.min_lsn.min(mutation.lsn);
        self.max_lsn = self.max_lsn.max(mutation.lsn);
        self.observe_memory()?;
        Ok(())
    }

    fn push_reconciled_inner(
        &mut self,
        source: SearchVersionRecord,
        after: Option<TextV4Payload>,
    ) -> Result<()> {
        if !matches!(self.mode, ExternalBuildMode::Reconciled(_)) {
            return Err(Error::precondition(
                "push_reconciled requires a reconciled text v4 DeltaRun builder",
            ));
        }
        if source.lsn == 0 {
            return Err(Error::precondition(
                "text v4 reconciled version uses reserved LSN zero",
            ));
        }
        if self
            .last_node_id
            .is_some_and(|previous| source.node_id <= previous)
        {
            return Err(Error::precondition(
                "text v4 reconciled builder requires strictly ascending NodeIds",
            ));
        }
        if after.as_ref().is_some_and(|payload| {
            payload.text.len().saturating_mul(64) > self.config.sort_budget()
        }) {
            return Err(Error::precondition(format!(
                "one text v4 reconciled payload can require more than the {}-byte per-record workspace",
                self.config.sort_budget()
            )));
        }
        let after = after.as_ref().map(prepare_payload).transpose()?;
        self.validate_filter_keys(after.as_ref())?;
        let output_record = match (source.operation, after.as_ref()) {
            (SearchVersionOperation::Live { payload_ordinal }, Some(payload)) => {
                if payload_ordinal == u64::MAX {
                    return Err(Error::precondition(
                        "text v4 reconciled live record uses the suppress ordinal",
                    ));
                }
                let fingerprint = prepared_payload_fingerprint(payload)?;
                if fingerprint != source.payload_fingerprint {
                    return Err(Error::precondition(
                        "text v4 reconciled live payload fingerprint disagrees with its winner",
                    ));
                }
                SearchVersionRecord::live(
                    source.node_id,
                    source.lsn,
                    source.payload_fingerprint,
                    self.live_count,
                )
            }
            (SearchVersionOperation::Suppress, None) => {
                if source.payload_fingerprint != search_suppress_fingerprint() {
                    return Err(Error::precondition(
                        "text v4 reconciled suppress fingerprint is not canonical",
                    ));
                }
                SearchVersionRecord::suppress(
                    source.node_id,
                    source.lsn,
                    source.payload_fingerprint,
                )
            }
            (SearchVersionOperation::Live { .. }, None) => {
                return Err(Error::precondition(
                    "text v4 reconciled live winner has no payload",
                ));
            }
            (SearchVersionOperation::Suppress, Some(_)) => {
                return Err(Error::precondition(
                    "text v4 reconciled suppress winner carries a live payload",
                ));
            }
        };

        self.last_node_id = Some(source.node_id);
        self.version_writer
            .as_mut()
            .ok_or_else(|| Error::invariant("text v4 version writer is absent"))?
            .push(output_record)?;
        let ordinal = self.live_count;
        if let Some(after) = &after {
            let doc_len = u32::try_from(after.tokens.len())
                .map_err(|_| Error::precondition("text v4 document length exceeds u32"))?;
            let doc_record = DocRecord {
                node_id: source.node_id,
                lsn: source.lsn,
                payload_fingerprint: source.payload_fingerprint,
                doc_len,
            };
            let encoded = encode_doc_record(doc_record);
            self.documents.write_all(&encoded)?;
            self.document_hasher.update(&encoded);
            for (position, term) in after.tokens.iter().enumerate() {
                self.push_record(TermRecord::occurrence(
                    term.clone(),
                    ordinal,
                    u32::try_from(position)
                        .map_err(|_| Error::precondition("text v4 token position exceeds u32"))?,
                    doc_len,
                ))?;
            }
            self.push_filters(ordinal, &after.filters)?;
            self.live_count = self
                .live_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 live count overflows"))?;
            self.final_total_len = self
                .final_total_len
                .checked_add(u64::from(doc_len))
                .ok_or_else(|| Error::invariant("text v4 final document length overflows"))?;
        } else {
            self.suppress_count = self
                .suppress_count
                .checked_add(1)
                .ok_or_else(|| Error::invariant("text v4 suppress count overflows"))?;
        }
        self.effective_count = self
            .effective_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text v4 mutation count overflows"))?;
        self.min_lsn = self.min_lsn.min(source.lsn);
        self.max_lsn = self.max_lsn.max(source.lsn);
        self.observe_memory()
    }

    fn push_term_delta_inner(&mut self, term: String, delta_df: i64) -> Result<()> {
        if !matches!(self.mode, ExternalBuildMode::Reconciled(_)) {
            return Err(Error::precondition(
                "push_term_delta requires a reconciled text v4 DeltaRun builder",
            ));
        }
        if term.is_empty() || term.starts_with(FILTER_KEY_PREFIX) {
            return Err(Error::precondition(
                "text v4 reconciled term statistic has an invalid term",
            ));
        }
        if self
            .last_term_delta
            .as_ref()
            .is_some_and(|previous| previous >= &term)
        {
            return Err(Error::precondition(
                "text v4 reconciled term statistics must be strictly increasing",
            ));
        }
        self.push_record(TermRecord::explicit_df(term.clone(), delta_df))?;
        self.last_term_delta = Some(term);
        self.observe_memory()
    }

    fn preflight_text(&self, mutation: &TextV4Mutation) -> Result<()> {
        let raw = mutation
            .before
            .as_ref()
            .map(|payload| payload.text.len())
            .unwrap_or(0)
            .checked_add(
                mutation
                    .after
                    .as_ref()
                    .map(|payload| payload.text.len())
                    .unwrap_or(0),
            )
            .ok_or_else(|| Error::precondition("text v4 mutation text length overflows"))?;
        // Shared tokenization can allocate one String header per emitted token.
        // 64× input bytes is a conservative preflight for Latin/CJK expansion.
        if raw.saturating_mul(64) > self.config.sort_budget() {
            return Err(Error::precondition(format!(
                "one text v4 mutation can require more than the {}-byte per-record workspace",
                self.config.sort_budget()
            )));
        }
        Ok(())
    }

    fn validate_filter_keys(&self, payload: Option<&super::PreparedPayload>) -> Result<()> {
        if payload.is_some_and(|payload| {
            payload.filters.keys().any(|property| {
                self.context
                    .complete_filter_properties
                    .binary_search(property)
                    .is_err()
            })
        }) {
            return Err(Error::precondition(
                "text v4 payload contains an unadvertised native-filter property",
            ));
        }
        Ok(())
    }

    fn push_record(&mut self, record: TermRecord) -> Result<()> {
        let record_bytes = size_of::<TermRecord>()
            .checked_add(record.term.capacity())
            .ok_or_else(|| Error::invariant("text v4 record accounting overflows"))?;
        if record_bytes > self.config.block_budget() {
            return Err(Error::precondition(format!(
                "one text v4 sort record requires {record_bytes} bytes, above the per-record budget"
            )));
        }
        let projected = self
            .records
            .len()
            .saturating_add(1)
            .checked_mul(size_of::<TermRecord>())
            .and_then(|bytes| bytes.checked_add(self.record_term_bytes))
            .and_then(|bytes| bytes.checked_add(record.term.capacity()))
            .ok_or_else(|| Error::invariant("text v4 sort buffer accounting overflows"))?;
        if projected > self.config.sort_budget() {
            self.flush_records()?;
        }
        if self.records.len() == self.records.capacity() {
            self.records.reserve_exact(1);
        }
        self.record_term_bytes = self
            .record_term_bytes
            .checked_add(record.term.capacity())
            .ok_or_else(|| Error::invariant("text v4 sort term bytes overflow"))?;
        self.records.push(record);
        self.sort_buffer_bytes = self
            .records
            .capacity()
            .checked_mul(size_of::<TermRecord>())
            .and_then(|bytes| bytes.checked_add(self.record_term_bytes))
            .ok_or_else(|| Error::invariant("text v4 sort buffer accounting overflows"))?;
        self.observe_memory()
    }

    fn push_filters(
        &mut self,
        ordinal: u64,
        values: &BTreeMap<String, SearchFilterValue>,
    ) -> Result<()> {
        for (property, value) in values {
            self.push_record(TermRecord::filter(
                filter_sort_key(property, value)?,
                ordinal,
            ))?;
        }
        self.observe_memory()
    }

    fn observe_memory(&mut self) -> Result<()> {
        self.metrics.observe_push(self.sort_buffer_bytes);
        self.metrics.peak_logical_memory_bytes = self.metrics.peak_logical_memory_bytes.max(
            self.sort_buffer_bytes.saturating_add(
                self.last_term_delta
                    .as_ref()
                    .map(|term| term.capacity())
                    .unwrap_or(0),
            ),
        );
        if self.metrics.peak_logical_memory_bytes > self.config.memory_budget_bytes {
            return Err(Error::invariant(
                "text v4 external logical memory exceeded configured budget",
            ));
        }
        Ok(())
    }

    fn flush_records(&mut self) -> Result<()> {
        if self.records.is_empty() {
            return Ok(());
        }
        self.records.sort_unstable();
        if self
            .records
            .windows(2)
            .any(|pair| pair[0].same_key(&pair[1]))
        {
            return Err(Error::invariant(
                "text v4 external sort contains duplicate records",
            ));
        }
        let records = std::mem::take(&mut self.records);
        self.record_term_bytes = 0;
        self.sort_buffer_bytes = 0;
        let mut writer = RunWriter::new(&self.spool)?;
        for record in &records {
            writer.push(record)?;
        }
        let run = writer.finish()?;
        self.metrics.initial_run_count = self.metrics.initial_run_count.saturating_add(1);
        self.metrics.spool_bytes_written = self.metrics.spool_bytes_written.saturating_add(run.len);
        self.insert_run(0, run)
    }

    fn insert_run(&mut self, mut level: usize, mut run: RunFile) -> Result<()> {
        loop {
            if self.levels.len() <= level {
                self.levels.resize_with(level + 1, || None);
            }
            let Some(existing) = self.levels[level].take() else {
                self.levels[level] = Some(run);
                return Ok(());
            };
            run = merge_runs(existing, run, &self.spool, self.config.block_budget())?;
            self.metrics.run_merge_count = self.metrics.run_merge_count.saturating_add(1);
            self.metrics.spool_bytes_written =
                self.metrics.spool_bytes_written.saturating_add(run.len);
            level += 1;
        }
    }

    fn collapse_runs(&mut self) -> Result<Option<RunFile>> {
        let mut final_run = None;
        for run in std::mem::take(&mut self.levels).into_iter().flatten() {
            final_run = Some(match final_run {
                None => run,
                Some(previous) => {
                    let merged =
                        merge_runs(previous, run, &self.spool, self.config.block_budget())?;
                    self.metrics.run_merge_count = self.metrics.run_merge_count.saturating_add(1);
                    self.metrics.spool_bytes_written =
                        self.metrics.spool_bytes_written.saturating_add(merged.len);
                    merged
                }
            });
        }
        Ok(final_run)
    }

    fn validate_reconciled_stats(&self) -> Result<()> {
        let ExternalBuildMode::Reconciled(stats) = self.mode else {
            return Ok(());
        };
        let doc_delta = i128::from(stats.doc_count_delta);
        let minimum_doc_delta = -i128::from(self.suppress_count);
        let maximum_doc_delta = i128::from(self.live_count);
        if doc_delta < minimum_doc_delta || doc_delta > maximum_doc_delta {
            return Err(Error::precondition(format!(
                "text v4 reconciled document delta {} leaves feasible [{minimum_doc_delta}, \
                 {maximum_doc_delta}] bounds for {} live and {} suppress winners",
                stats.doc_count_delta, self.live_count, self.suppress_count
            )));
        }
        if i128::from(stats.total_len_delta) > i128::from(self.final_total_len) {
            return Err(Error::precondition(format!(
                "text v4 reconciled total-length delta {} exceeds final live length {}",
                stats.total_len_delta, self.final_total_len
            )));
        }
        Ok(())
    }
}

impl TextV4ExternalBuilder {
    pub fn finish(mut self) -> Result<Option<TextV4ExternalArtifact>> {
        if self.poisoned {
            return Err(Error::precondition(
                "cannot finish a poisoned text v4 external builder",
            ));
        }
        if self.effective_count == 0 {
            return Ok(None);
        }
        self.validate_reconciled_stats()?;
        self.flush_records()?;
        let final_run = self.collapse_runs()?;

        let version_writer = self
            .version_writer
            .take()
            .ok_or_else(|| Error::invariant("text v4 version writer is absent"))?;
        let (mut version_file, mut version_table) = version_writer.finish()?;
        version_file.rewind()?;

        let mut output = self.spool.create()?;
        output.write_all(MAGIC_V4)?;
        let copied = std::io::copy(&mut version_file, &mut output)?;
        if copied != version_table.len {
            return Err(Error::invariant(
                "text v4 external version spool length changed",
            ));
        }
        version_table.offset = MAGIC_V4.len() as u64;

        self.documents.flush()?;
        let document_len = self
            .live_count
            .checked_mul(super::DOC_RECORD_LEN)
            .ok_or_else(|| Error::invariant("text v4 document table length overflows"))?;
        if self.documents.metadata()?.len() != document_len {
            return Err(Error::invariant(
                "text v4 external document spool length changed",
            ));
        }
        self.documents.rewind()?;
        let doc_offset = output.stream_position()?;
        let copied = std::io::copy(&mut self.documents, &mut output)?;
        if copied != document_len {
            return Err(Error::invariant(
                "text v4 external document copy length changed",
            ));
        }
        let doc_table = DocTableRef {
            offset: doc_offset,
            len: document_len,
            row_count: self.live_count,
            content_xxh3: non_zero_digest(self.document_hasher),
        };

        let postings_start = output.stream_position()?;
        let term_spool = self.spool.create()?;
        let filter_spool = self.spool.create()?;
        let mut assembler = ExternalAssembler::new(
            &mut output,
            term_spool,
            filter_spool,
            self.live_count,
            self.effective_count,
            matches!(self.mode, ExternalBuildMode::Reconciled(_)),
            self.config.wire,
            self.config.block_budget(),
            self.config.directory_budget(),
            self.config.max_filter_distinct_per_property,
        )?;
        if let Some(run) = final_run {
            assembler.consume(run, self.config.block_budget())?;
        }
        let assembly = assembler.finish()?;
        self.metrics.max_posting_block_bytes = assembly.max_posting_block_bytes;
        self.metrics.spool_bytes_written = self
            .metrics
            .spool_bytes_written
            .saturating_add(assembly.term_spool_bytes);
        self.metrics.filter_spool_bytes = assembly.filter_spool_bytes;
        self.metrics.filter_value_count = assembly.filter_entries.len() as u64;
        self.metrics.max_filter_bytes = assembly.max_filter_bytes;
        self.metrics.max_directory_bytes = assembly.filter_directory_bytes;
        self.metrics.observe_finish(
            assembly
                .max_filter_bytes
                .saturating_add(assembly.filter_directory_bytes),
        );
        self.metrics.spool_bytes_written = self
            .metrics
            .spool_bytes_written
            .saturating_add(assembly.filter_spool_bytes);
        self.metrics.observe_finish(
            assembly
                .filter_directory_bytes
                .saturating_add(assembly.max_posting_block_bytes.saturating_mul(3))
                .saturating_add(self.config.block_budget()),
        );
        let postings_end = output.stream_position()?;
        if postings_start != assembly.postings_start || postings_end != assembly.postings_end {
            return Err(Error::invariant(
                "text v4 external postings region position drifted",
            ));
        }
        let postings_region = RegionRef {
            offset: postings_start,
            len: postings_end
                .checked_sub(postings_start)
                .ok_or_else(|| Error::invariant("text v4 postings region underflows"))?,
            block_count: assembly.posting_block_count,
            metadata_xxh3: assembly.postings_metadata_xxh3,
        };

        let (dictionary, max_dictionary_block_bytes, dictionary_directory_bytes) =
            emit_dictionary_blocks(
                &mut output,
                assembly.term_spool,
                assembly.term_count,
                self.config.wire,
                self.config.block_budget(),
                self.config.directory_budget(),
            )?;
        self.metrics.max_dictionary_block_bytes = max_dictionary_block_bytes;
        self.metrics.max_directory_bytes = self.metrics.max_directory_bytes.max(
            assembly
                .filter_directory_bytes
                .saturating_add(dictionary_directory_bytes),
        );
        self.metrics.observe_finish(
            assembly
                .filter_directory_bytes
                .saturating_add(dictionary_directory_bytes)
                .saturating_add(max_dictionary_block_bytes.saturating_mul(3)),
        );

        let (filters, max_filter_emit_bytes, filter_output_directory_bytes) = emit_filter_blocks(
            &mut output,
            assembly.filter_spool,
            assembly.filter_entries,
            &self.context.complete_filter_properties,
            self.live_count,
            &self.config,
        )?;
        self.metrics.max_filter_bytes = self.metrics.max_filter_bytes.max(max_filter_emit_bytes);
        self.metrics.max_directory_bytes = self
            .metrics
            .max_directory_bytes
            .max(dictionary_directory_bytes.saturating_add(filter_output_directory_bytes));
        self.metrics.observe_finish(
            assembly
                .filter_directory_bytes
                .saturating_add(dictionary_directory_bytes)
                .saturating_add(filter_output_directory_bytes)
                .saturating_add(max_filter_emit_bytes.saturating_mul(3)),
        );
        let content_xxh3 = content_digest(
            &version_table,
            self.delta_docs,
            self.delta_total_len,
            &doc_table,
            &postings_region,
            &dictionary,
            &filters,
        )?;
        let stats = text_segment_stats(self.role, self.delta_docs, self.delta_total_len)?;
        let segment = SearchSegmentRef {
            sst_id: self.context.sst_id,
            role: self.role,
            format: SearchSegmentFormat::TextV4,
            payload: SearchSegmentPayload::Complete,
            event_ranges: self.context.event_ranges,
            min_lsn: self.min_lsn,
            max_lsn: self.max_lsn,
            mutation_count: self.effective_count,
            live_payload_count: self.live_count,
            suppress_count: self.suppress_count,
            content_xxh3,
            complete_filter_properties: self.context.complete_filter_properties,
            stats,
            equal_lsn_conflict_count: 0,
        };
        let binding = SearchSegmentWireBinding::new(&self.state, &segment, version_table.clone())?;
        let footer = Footer {
            footer_version: FOOTER_VERSION,
            binding,
            delta_docs: self.delta_docs,
            delta_total_len: self.delta_total_len,
            doc_table,
            postings_region,
            dictionary,
            filters,
        };
        let footer_wire_len = bincode_options(MAX_FOOTER_BYTES)
            .serialized_size(&footer)
            .map_err(|error| Error::precondition(format!("text v4 footer size failed: {error}")))?;
        if footer_wire_len > self.config.block_budget() as u64 {
            return Err(Error::precondition(format!(
                "text v4 footer requires {footer_wire_len} bytes, above the configured footer workspace"
            )));
        }
        let footer_bytes = serialize_bounded(&footer, MAX_FOOTER_BYTES, "text v4 footer")?;
        self.metrics.observe_finish(
            self.metrics
                .max_directory_bytes
                .saturating_add(footer_bytes.len()),
        );
        let footer_offset = output.stream_position()?;
        output.write_all(&footer_bytes)?;
        output.write_all(TRAILER_MAGIC)?;
        output.write_all(&(footer_bytes.len() as u64).to_le_bytes())?;
        output.write_all(&crc32fast::hash(&footer_bytes).to_le_bytes())?;
        output.flush()?;
        output.sync_data()?;
        let len = output.stream_position()?;
        if len
            != footer_offset
                .checked_add(footer_bytes.len() as u64)
                .and_then(|offset| offset.checked_add(TRAILER_LEN as u64))
                .ok_or_else(|| Error::invariant("text v4 external object length overflows"))?
            || output.metadata()?.len() != len
        {
            return Err(Error::invariant("text v4 external artifact length changed"));
        }
        if self.metrics.peak_logical_memory_bytes > self.config.memory_budget_bytes {
            return Err(Error::invariant(
                "text v4 external peak accounting exceeds configured budget",
            ));
        }
        output.rewind()?;
        let dictionary_block_count = u32::try_from(footer.dictionary.len())
            .map_err(|_| Error::invariant("text v4 dictionary block count exceeds u32"))?;
        Ok(Some(TextV4ExternalArtifact {
            file: output,
            len,
            output: TextV4BuildOutput {
                segment,
                object_len: len,
                dictionary_block_count,
                version_table,
            },
            metrics: self.metrics,
        }))
    }
}

fn filter_sort_key(property: &str, value: &SearchFilterValue) -> Result<String> {
    let encoded = serialize_bounded(
        &(property, value),
        MAX_RAW_BLOCK_BYTES,
        "text v4 filter sort key",
    )?;
    let capacity = FILTER_KEY_PREFIX
        .len()
        .checked_add(
            encoded
                .len()
                .checked_mul(2)
                .ok_or_else(|| Error::precondition("text v4 filter sort key length overflows"))?,
        )
        .ok_or_else(|| Error::precondition("text v4 filter sort key length overflows"))?;
    let mut key = String::with_capacity(capacity);
    key.push_str(FILTER_KEY_PREFIX);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in encoded {
        key.push(HEX[(byte >> 4) as usize] as char);
        key.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(key)
}

fn decode_filter_sort_key(key: &str) -> Result<(String, SearchFilterValue)> {
    let hex = key
        .strip_prefix(FILTER_KEY_PREFIX)
        .ok_or_else(|| Error::invariant("text v4 filter sort key prefix is invalid"))?;
    if hex.len() % 2 != 0 {
        return Err(Error::invariant(
            "text v4 filter sort key has odd hexadecimal length",
        ));
    }
    let mut encoded = Vec::with_capacity(hex.len() / 2);
    for pair in hex.as_bytes().chunks_exact(2) {
        let high = decode_hex_nibble(pair[0])?;
        let low = decode_hex_nibble(pair[1])?;
        encoded.push(high << 4 | low);
    }
    let decoded: (String, SearchFilterValue) =
        super::deserialize_bounded(&encoded, MAX_RAW_BLOCK_BYTES, "text v4 filter sort key")?;
    if decoded.0.is_empty() || filter_sort_key(&decoded.0, &decoded.1)? != key {
        return Err(Error::invariant("text v4 filter sort key is not canonical"));
    }
    Ok(decoded)
}

fn decode_hex_nibble(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(Error::invariant(
            "text v4 filter sort key is not lowercase hexadecimal",
        )),
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum TermRecordKind {
    Filter {
        ordinal: u64,
    },
    Df {
        event: u64,
        delta: i8,
    },
    ExplicitDf {
        delta: i64,
    },
    Occurrence {
        doc: u64,
        position: u32,
        doc_len: u32,
    },
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct TermRecord {
    term: String,
    kind: TermRecordKind,
}

impl TermRecord {
    fn filter(term: String, ordinal: u64) -> Self {
        Self {
            term,
            kind: TermRecordKind::Filter { ordinal },
        }
    }

    fn df(term: String, event: u64, delta: i8) -> Self {
        Self {
            term,
            kind: TermRecordKind::Df { event, delta },
        }
    }

    fn explicit_df(term: String, delta: i64) -> Self {
        Self {
            term,
            kind: TermRecordKind::ExplicitDf { delta },
        }
    }

    fn occurrence(term: String, doc: u64, position: u32, doc_len: u32) -> Self {
        Self {
            term,
            kind: TermRecordKind::Occurrence {
                doc,
                position,
                doc_len,
            },
        }
    }

    fn same_key(&self, other: &Self) -> bool {
        self == other
    }
}

impl Ord for TermRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.term
            .cmp(&other.term)
            .then_with(|| match (&self.kind, &other.kind) {
                (
                    TermRecordKind::Filter {
                        ordinal: left_ordinal,
                    },
                    TermRecordKind::Filter {
                        ordinal: right_ordinal,
                    },
                ) => left_ordinal.cmp(right_ordinal),
                (TermRecordKind::Filter { .. }, _) => Ordering::Less,
                (_, TermRecordKind::Filter { .. }) => Ordering::Greater,
                (
                    TermRecordKind::Df {
                        event: left_event,
                        delta: left_delta,
                    },
                    TermRecordKind::Df {
                        event: right_event,
                        delta: right_delta,
                    },
                ) => left_event
                    .cmp(right_event)
                    .then_with(|| left_delta.cmp(right_delta)),
                (
                    TermRecordKind::ExplicitDf { delta: left },
                    TermRecordKind::ExplicitDf { delta: right },
                ) => left.cmp(right),
                (TermRecordKind::Df { .. }, TermRecordKind::ExplicitDf { .. }) => Ordering::Less,
                (TermRecordKind::ExplicitDf { .. }, TermRecordKind::Df { .. }) => Ordering::Greater,
                (
                    TermRecordKind::Df { .. } | TermRecordKind::ExplicitDf { .. },
                    TermRecordKind::Occurrence { .. },
                ) => Ordering::Less,
                (
                    TermRecordKind::Occurrence { .. },
                    TermRecordKind::Df { .. } | TermRecordKind::ExplicitDf { .. },
                ) => Ordering::Greater,
                (
                    TermRecordKind::Occurrence {
                        doc: left_doc,
                        position: left_position,
                        doc_len: left_len,
                    },
                    TermRecordKind::Occurrence {
                        doc: right_doc,
                        position: right_position,
                        doc_len: right_len,
                    },
                ) => left_doc
                    .cmp(right_doc)
                    .then_with(|| left_position.cmp(right_position))
                    .then_with(|| left_len.cmp(right_len)),
            })
    }
}

impl PartialOrd for TermRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct SpoolFactory {
    directory: Option<PathBuf>,
}

impl SpoolFactory {
    fn new(directory: Option<PathBuf>) -> Self {
        Self { directory }
    }

    fn create(&self) -> Result<File> {
        if let Some(directory) = self.directory.as_deref() {
            return tempfile::tempfile_in(directory).map_err(Error::from);
        }
        #[cfg(unix)]
        {
            if let Ok(file) = tempfile::tempfile_in("/var/tmp") {
                return Ok(file);
            }
        }
        tempfile::tempfile().map_err(Error::from)
    }
}

fn validate_spool_directory(directory: &Path) -> Result<()> {
    let metadata = std::fs::metadata(directory).map_err(|error| {
        Error::precondition(format!(
            "text v4 spool directory {} is inaccessible: {error}",
            directory.display()
        ))
    })?;
    if !metadata.is_dir() {
        return Err(Error::precondition(format!(
            "text v4 spool path {} is not a directory",
            directory.display()
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct RunFile {
    file: File,
    len: u64,
}

struct RunWriter {
    writer: BufWriter<File>,
    count: u64,
    payload_len: u64,
    payload_crc: crc32fast::Hasher,
}

impl RunWriter {
    fn new(spool: &SpoolFactory) -> Result<Self> {
        let mut file = spool.create()?;
        file.write_all(&[0; RUN_HEADER_LEN as usize])?;
        Ok(Self {
            writer: BufWriter::with_capacity(IO_BUFFER_BYTES, file),
            count: 0,
            payload_len: 0,
            payload_crc: crc32fast::Hasher::new(),
        })
    }

    fn push(&mut self, record: &TermRecord) -> Result<()> {
        let term_len = u32::try_from(record.term.len())
            .map_err(|_| Error::precondition("text v4 run term exceeds u32"))?;
        self.write_payload(&term_len.to_le_bytes())?;
        self.write_payload(record.term.as_bytes())?;
        match record.kind {
            TermRecordKind::Filter { ordinal } => {
                self.write_payload(&[3])?;
                self.write_payload(&ordinal.to_le_bytes())?;
                self.write_payload(&0u32.to_le_bytes())?;
                self.write_payload(&0u32.to_le_bytes())?;
                self.write_payload(&[0])?;
            }
            TermRecordKind::Df { event, delta } => {
                self.write_payload(&[1])?;
                self.write_payload(&event.to_le_bytes())?;
                self.write_payload(&0u32.to_le_bytes())?;
                self.write_payload(&0u32.to_le_bytes())?;
                self.write_payload(&delta.to_le_bytes())?;
            }
            TermRecordKind::ExplicitDf { delta } => {
                self.write_payload(&[4])?;
                self.write_payload(&delta.to_le_bytes())?;
                self.write_payload(&0u32.to_le_bytes())?;
                self.write_payload(&0u32.to_le_bytes())?;
                self.write_payload(&[0])?;
            }
            TermRecordKind::Occurrence {
                doc,
                position,
                doc_len,
            } => {
                self.write_payload(&[2])?;
                self.write_payload(&doc.to_le_bytes())?;
                self.write_payload(&position.to_le_bytes())?;
                self.write_payload(&doc_len.to_le_bytes())?;
                self.write_payload(&[0])?;
            }
        }
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text v4 run count overflows"))?;
        Ok(())
    }

    fn write_payload(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.payload_crc.update(bytes);
        self.payload_len = self
            .payload_len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::invariant("text v4 run length overflows"))?;
        Ok(())
    }

    fn finish(mut self) -> Result<RunFile> {
        self.writer.flush()?;
        let mut file = self
            .writer
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        let crc = self.payload_crc.finalize();
        file.rewind()?;
        file.write_all(RUN_MAGIC)?;
        file.write_all(&self.count.to_le_bytes())?;
        file.write_all(&self.payload_len.to_le_bytes())?;
        file.write_all(&crc.to_le_bytes())?;
        let len = RUN_HEADER_LEN
            .checked_add(self.payload_len)
            .ok_or_else(|| Error::invariant("text v4 run total length overflows"))?;
        if file.metadata()?.len() != len {
            return Err(Error::invariant("text v4 run file length mismatch"));
        }
        file.rewind()?;
        Ok(RunFile { file, len })
    }
}

struct RunReader {
    reader: BufReader<File>,
    expected_count: u64,
    expected_payload_len: u64,
    expected_crc: u32,
    read_count: u64,
    read_payload_len: u64,
    payload_crc: crc32fast::Hasher,
    record_budget: usize,
    verified: bool,
}

impl RunReader {
    fn open(mut run: RunFile, record_budget: usize) -> Result<Self> {
        run.file.rewind()?;
        let mut magic = [0u8; 8];
        run.file.read_exact(&mut magic)?;
        if &magic != RUN_MAGIC {
            return Err(Error::invariant("text v4 run magic mismatch"));
        }
        let expected_count = read_u64_io(&mut run.file)?;
        let expected_payload_len = read_u64_io(&mut run.file)?;
        let expected_crc = read_u32_io(&mut run.file)?;
        let expected_len = RUN_HEADER_LEN
            .checked_add(expected_payload_len)
            .ok_or_else(|| Error::invariant("text v4 run length overflows"))?;
        if expected_len != run.len || run.file.metadata()?.len() != expected_len {
            return Err(Error::invariant("text v4 run header length mismatch"));
        }
        Ok(Self {
            reader: BufReader::with_capacity(IO_BUFFER_BYTES, run.file),
            expected_count,
            expected_payload_len,
            expected_crc,
            read_count: 0,
            read_payload_len: 0,
            payload_crc: crc32fast::Hasher::new(),
            record_budget,
            verified: false,
        })
    }

    fn next(&mut self) -> Result<Option<TermRecord>> {
        if self.read_count == self.expected_count {
            self.verify_end()?;
            return Ok(None);
        }
        let term_len = self.read_payload_u32()? as usize;
        if term_len
            .saturating_add(size_of::<TermRecord>())
            .saturating_add(RECORD_FIXED_BYTES)
            > self.record_budget
        {
            return Err(Error::invariant(
                "text v4 run record exceeds configured build budget",
            ));
        }
        if term_len as u64
            > self
                .remaining_payload()
                .saturating_sub(RECORD_FIXED_BYTES as u64)
        {
            return Err(Error::invariant(
                "text v4 run term exceeds remaining payload",
            ));
        }
        let mut term = vec![0; term_len];
        self.read_payload_exact(&mut term)?;
        let term = String::from_utf8(term)
            .map_err(|_| Error::invariant("text v4 run term is not UTF-8"))?;
        let tag = self.read_payload_byte()?;
        let primary = self.read_payload_u64()?;
        let secondary = self.read_payload_u32()?;
        let tertiary = self.read_payload_u32()?;
        let signed = self.read_payload_byte()? as i8;
        let kind = match tag {
            3 if secondary == 0 && tertiary == 0 && signed == 0 => {
                TermRecordKind::Filter { ordinal: primary }
            }
            1 if secondary == 0 && tertiary == 0 && matches!(signed, -1 | 1) => {
                TermRecordKind::Df {
                    event: primary,
                    delta: signed,
                }
            }
            4 if secondary == 0 && tertiary == 0 && signed == 0 => TermRecordKind::ExplicitDf {
                delta: i64::from_le_bytes(primary.to_le_bytes()),
            },
            2 if signed == 0 && secondary < tertiary => TermRecordKind::Occurrence {
                doc: primary,
                position: secondary,
                doc_len: tertiary,
            },
            _ => return Err(Error::invariant("text v4 run record tag/body is invalid")),
        };
        self.read_count += 1;
        Ok(Some(TermRecord { term, kind }))
    }

    fn read_payload_byte(&mut self) -> Result<u8> {
        let mut byte = [0];
        self.read_payload_exact(&mut byte)?;
        Ok(byte[0])
    }

    fn read_payload_u32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        self.read_payload_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_payload_u64(&mut self) -> Result<u64> {
        let mut bytes = [0; 8];
        self.read_payload_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_payload_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() as u64 > self.remaining_payload() {
            return Err(Error::invariant("text v4 run record is truncated"));
        }
        self.reader.read_exact(bytes)?;
        self.payload_crc.update(bytes);
        self.read_payload_len += bytes.len() as u64;
        Ok(())
    }

    fn remaining_payload(&self) -> u64 {
        self.expected_payload_len - self.read_payload_len
    }

    fn verify_end(&mut self) -> Result<()> {
        if self.verified {
            return Ok(());
        }
        if self.read_payload_len != self.expected_payload_len {
            return Err(Error::invariant("text v4 run has trailing payload"));
        }
        let actual = std::mem::replace(&mut self.payload_crc, crc32fast::Hasher::new()).finalize();
        if actual != self.expected_crc {
            return Err(Error::invariant("text v4 run checksum mismatch"));
        }
        self.verified = true;
        Ok(())
    }
}

fn merge_runs(
    left: RunFile,
    right: RunFile,
    spool: &SpoolFactory,
    record_budget: usize,
) -> Result<RunFile> {
    let mut left = RunReader::open(left, record_budget)?;
    let mut right = RunReader::open(right, record_budget)?;
    let mut output = RunWriter::new(spool)?;
    let mut a = left.next()?;
    let mut b = right.next()?;
    let mut previous: Option<TermRecord> = None;
    while a.is_some() || b.is_some() {
        let next = match (&a, &b) {
            (Some(left_record), Some(right_record)) => {
                if left_record.same_key(right_record) {
                    return Err(Error::invariant(
                        "duplicate text v4 record encountered while merging",
                    ));
                }
                if left_record <= right_record {
                    let value = a.take().expect("left record present");
                    a = left.next()?;
                    value
                } else {
                    let value = b.take().expect("right record present");
                    b = right.next()?;
                    value
                }
            }
            (Some(_), None) => {
                let value = a.take().expect("left record present");
                a = left.next()?;
                value
            }
            (None, Some(_)) => {
                let value = b.take().expect("right record present");
                b = right.next()?;
                value
            }
            (None, None) => break,
        };
        if previous.as_ref().is_some_and(|record| record >= &next) {
            return Err(Error::invariant(
                "text v4 run merge produced non-increasing records",
            ));
        }
        output.push(&next)?;
        previous = Some(next);
    }
    output.finish()
}

struct ExternalAssembler<'a> {
    output: &'a mut File,
    term_spool: BufWriter<File>,
    term_spool_bytes: u64,
    term_count: u64,
    filter_spool: BufWriter<File>,
    filter_spool_bytes: u64,
    filter_entries: Vec<FilterSpoolEntry>,
    filter_directory_bytes: usize,
    max_filter_distinct_per_property: usize,
    directory_budget: usize,
    row_count: u64,
    mutation_count: u64,
    reconciled: bool,
    wire: TextV4BuildOptions,
    block_budget: usize,
    postings_start: u64,
    current_filter_key: Option<String>,
    current_filter_property: Option<String>,
    current_filter_value: Option<SearchFilterValue>,
    current_filter: Option<FilterAccumulator>,
    saw_term_records: bool,
    current_term: Option<String>,
    delta_df: i64,
    saw_explicit_df: bool,
    current_doc: Option<u64>,
    current_doc_len: u32,
    current_positions: Vec<u32>,
    posting_block: Vec<Posting>,
    posting_block_bytes: usize,
    term_blocks: Vec<PostingBlockRef>,
    term_block_bytes: usize,
    posting_block_count: u64,
    metadata_hasher: Xxh3,
    max_posting_block_bytes: usize,
    max_filter_bytes: usize,
}

struct ExternalAssembly {
    term_spool: File,
    term_spool_bytes: u64,
    term_count: u64,
    postings_start: u64,
    postings_end: u64,
    posting_block_count: u64,
    postings_metadata_xxh3: u64,
    max_posting_block_bytes: usize,
    filter_spool: File,
    filter_spool_bytes: u64,
    filter_entries: Vec<FilterSpoolEntry>,
    filter_directory_bytes: usize,
    max_filter_bytes: usize,
}

#[derive(Debug)]
struct FilterSpoolEntry {
    property: String,
    value: SearchFilterValue,
    cardinality: u64,
    encoding: FilterPostingEncoding,
    offset: u64,
    len: u32,
    crc32: u32,
}

#[derive(Debug)]
struct FilterAccumulator {
    encoding: FilterPostingEncoding,
    raw: Vec<u8>,
    cardinality: u64,
    last_ordinal: Option<u64>,
}

impl FilterAccumulator {
    fn new() -> Self {
        Self {
            encoding: FilterPostingEncoding::SparseDeltaVarint,
            raw: Vec::new(),
            cardinality: 0,
            last_ordinal: None,
        }
    }

    fn push(&mut self, ordinal: u64, row_count: u64, block_budget: usize) -> Result<usize> {
        if ordinal >= row_count
            || self
                .last_ordinal
                .is_some_and(|previous| previous >= ordinal)
        {
            return Err(Error::invariant(
                "text v4 filter ordinals are not strictly increasing",
            ));
        }
        let mut peak = self.raw.capacity();
        match self.encoding {
            FilterPostingEncoding::SparseDeltaVarint => {
                let delta = self
                    .last_ordinal
                    .map_or(ordinal, |previous| ordinal - previous);
                let mut encoded_delta = Vec::with_capacity(10);
                encode_u64_varint(delta, &mut encoded_delta);
                let projected = self
                    .raw
                    .len()
                    .checked_add(encoded_delta.len())
                    .ok_or_else(|| Error::precondition("text v4 sparse filter size overflows"))?;
                let dense_len = bitmap_words(row_count)?
                    .checked_mul(size_of::<u64>())
                    .ok_or_else(|| Error::precondition("text v4 filter bitmap size overflows"))?;
                if projected >= dense_len || projected > block_budget {
                    if dense_len == 0 || dense_len > block_budget {
                        return Err(Error::precondition(format!(
                            "one text v4 filter posting exceeds the {block_budget}-byte block workspace"
                        )));
                    }
                    let mut dense = vec![0u8; dense_len];
                    let mut cursor = 0usize;
                    let mut previous = 0u64;
                    for index in 0..self.cardinality {
                        let delta = decode_u64_varint(&self.raw, &mut cursor)?;
                        let decoded = previous
                            .checked_add(delta)
                            .ok_or_else(|| Error::invariant("text v4 filter ordinal overflows"))?;
                        if index > 0 && delta == 0 {
                            return Err(Error::invariant(
                                "text v4 sparse filter ordinals are duplicated",
                            ));
                        }
                        set_dense_filter_bit(&mut dense, decoded, row_count)?;
                        previous = decoded;
                    }
                    if cursor != self.raw.len() {
                        return Err(Error::invariant(
                            "text v4 sparse filter accumulator has trailing bytes",
                        ));
                    }
                    set_dense_filter_bit(&mut dense, ordinal, row_count)?;
                    peak = peak.saturating_add(dense.capacity());
                    self.raw = dense;
                    self.encoding = FilterPostingEncoding::DenseBitmap;
                } else {
                    self.raw.extend_from_slice(&encoded_delta);
                    peak = peak.max(self.raw.capacity());
                }
            }
            FilterPostingEncoding::DenseBitmap => {
                set_dense_filter_bit(&mut self.raw, ordinal, row_count)?;
            }
        }
        self.cardinality = self
            .cardinality
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text v4 filter cardinality overflows"))?;
        self.last_ordinal = Some(ordinal);
        Ok(peak.max(self.raw.capacity()))
    }
}

fn set_dense_filter_bit(bitmap: &mut [u8], ordinal: u64, row_count: u64) -> Result<()> {
    if ordinal >= row_count {
        return Err(Error::invariant(
            "text v4 filter ordinal leaves document table",
        ));
    }
    let byte = usize::try_from(ordinal / 8)
        .map_err(|_| Error::invariant("text v4 filter ordinal exceeds usize"))?;
    let target = bitmap
        .get_mut(byte)
        .ok_or_else(|| Error::invariant("text v4 filter ordinal leaves bitmap"))?;
    *target |= 1u8 << (ordinal % 8);
    Ok(())
}

impl<'a> ExternalAssembler<'a> {
    fn new(
        output: &'a mut File,
        term_spool: File,
        filter_spool: File,
        row_count: u64,
        mutation_count: u64,
        reconciled: bool,
        wire: TextV4BuildOptions,
        block_budget: usize,
        directory_budget: usize,
        max_filter_distinct_per_property: usize,
    ) -> Result<Self> {
        let postings_start = output.stream_position()?;
        let mut metadata_hasher = Xxh3::new();
        metadata_hasher.update(POSTINGS_REGION_DOMAIN);
        Ok(Self {
            output,
            term_spool: BufWriter::with_capacity(IO_BUFFER_BYTES, term_spool),
            term_spool_bytes: 0,
            term_count: 0,
            filter_spool: BufWriter::with_capacity(IO_BUFFER_BYTES, filter_spool),
            filter_spool_bytes: 0,
            filter_entries: Vec::new(),
            filter_directory_bytes: 0,
            max_filter_distinct_per_property,
            directory_budget,
            row_count,
            mutation_count,
            reconciled,
            wire,
            block_budget,
            postings_start,
            current_filter_key: None,
            current_filter_property: None,
            current_filter_value: None,
            current_filter: None,
            saw_term_records: false,
            current_term: None,
            delta_df: 0,
            saw_explicit_df: false,
            current_doc: None,
            current_doc_len: 0,
            current_positions: Vec::new(),
            posting_block: Vec::new(),
            posting_block_bytes: 0,
            term_blocks: Vec::new(),
            term_block_bytes: 0,
            posting_block_count: 0,
            metadata_hasher,
            max_posting_block_bytes: 0,
            max_filter_bytes: 0,
        })
    }

    fn consume(&mut self, run: RunFile, record_budget: usize) -> Result<()> {
        let mut reader = RunReader::open(run, record_budget)?;
        let mut previous = None;
        while let Some(record) = reader.next()? {
            if previous.as_ref().is_some_and(|prior| prior >= &record) {
                return Err(Error::invariant("final text v4 run is not strictly sorted"));
            }
            self.accept(&record)?;
            previous = Some(record);
        }
        self.finish_filter()?;
        self.finish_term()
    }

    fn accept(&mut self, record: &TermRecord) -> Result<()> {
        if let TermRecordKind::Filter { ordinal } = record.kind {
            if self.saw_term_records || !record.term.starts_with(FILTER_KEY_PREFIX) {
                return Err(Error::invariant(
                    "text v4 filter record appears outside the filter keyspace",
                ));
            }
            if self
                .current_filter_key
                .as_ref()
                .is_some_and(|key| key != &record.term)
            {
                self.finish_filter()?;
            }
            if self.current_filter_key.is_none() {
                let (property, value) = decode_filter_sort_key(&record.term)?;
                self.current_filter_key = Some(record.term.clone());
                self.current_filter_property = Some(property);
                self.current_filter_value = Some(value);
                self.current_filter = Some(FilterAccumulator::new());
            }
            let peak = self
                .current_filter
                .as_mut()
                .ok_or_else(|| Error::invariant("text v4 filter accumulator is absent"))?
                .push(ordinal, self.row_count, self.block_budget)?;
            self.max_filter_bytes = self.max_filter_bytes.max(peak);
            return Ok(());
        }
        if record.term.starts_with(FILTER_KEY_PREFIX) {
            return Err(Error::invariant(
                "text v4 term collides with the reserved filter keyspace",
            ));
        }
        self.finish_filter()?;
        self.saw_term_records = true;
        if self
            .current_term
            .as_ref()
            .is_some_and(|term| term != &record.term)
        {
            self.finish_term()?;
        }
        if self.current_term.is_none() {
            self.current_term = Some(record.term.clone());
        }
        match record.kind {
            TermRecordKind::Filter { .. } => unreachable!("filter handled above"),
            TermRecordKind::Df { delta, .. } => {
                if self.reconciled {
                    return Err(Error::invariant(
                        "text v4 reconciled run contains derived df events",
                    ));
                }
                if self.current_doc.is_some() {
                    return Err(Error::invariant(
                        "text v4 df event appeared after term occurrences",
                    ));
                }
                self.delta_df = self
                    .delta_df
                    .checked_add(i64::from(delta))
                    .ok_or_else(|| Error::invariant("text v4 delta_df overflows"))?;
            }
            TermRecordKind::ExplicitDf { delta } => {
                if !self.reconciled {
                    return Err(Error::invariant(
                        "text v4 derived run contains explicit reconciled df",
                    ));
                }
                if self.current_doc.is_some() || self.saw_explicit_df {
                    return Err(Error::invariant(
                        "text v4 reconciled term has duplicate or late explicit df",
                    ));
                }
                self.delta_df = delta;
                self.saw_explicit_df = true;
            }
            TermRecordKind::Occurrence {
                doc,
                position,
                doc_len,
            } => self.accept_occurrence(doc, position, doc_len)?,
        }
        Ok(())
    }

    fn finish_filter(&mut self) -> Result<()> {
        let Some(accumulator) = self.current_filter.take() else {
            if self.current_filter_key.is_some()
                || self.current_filter_property.is_some()
                || self.current_filter_value.is_some()
            {
                return Err(Error::invariant(
                    "text v4 filter assembler state is incomplete",
                ));
            }
            return Ok(());
        };
        let property = self
            .current_filter_property
            .take()
            .ok_or_else(|| Error::invariant("text v4 filter property is absent"))?;
        let value = self
            .current_filter_value
            .take()
            .ok_or_else(|| Error::invariant("text v4 filter value is absent"))?;
        self.current_filter_key = None;
        if accumulator.cardinality == 0 || accumulator.raw.is_empty() {
            return Err(Error::invariant("text v4 filter posting is empty"));
        }
        if accumulator.raw.len() > self.block_budget {
            return Err(Error::precondition(
                "one text v4 filter posting exceeds its block workspace",
            ));
        }
        let offset = self.filter_spool_bytes;
        self.filter_spool.write_all(&accumulator.raw)?;
        self.filter_spool_bytes = self
            .filter_spool_bytes
            .checked_add(accumulator.raw.len() as u64)
            .ok_or_else(|| Error::invariant("text v4 filter spool length overflows"))?;
        let entry = FilterSpoolEntry {
            property,
            value,
            cardinality: accumulator.cardinality,
            encoding: accumulator.encoding,
            offset,
            len: u32::try_from(accumulator.raw.len())
                .map_err(|_| Error::precondition("text v4 filter posting exceeds u32"))?,
            crc32: crc32fast::hash(&accumulator.raw),
        };
        self.filter_directory_bytes = self
            .filter_directory_bytes
            .saturating_add(size_of::<FilterSpoolEntry>())
            .saturating_add(entry.property.capacity())
            .saturating_add(filter_value_resident_bytes(&entry.value));
        if self.filter_directory_bytes > self.directory_budget {
            return Err(Error::precondition(
                "text v4 filter directory exceeds configured resident budget",
            ));
        }
        self.filter_entries.push(entry);
        Ok(())
    }

    fn accept_occurrence(&mut self, doc: u64, position: u32, doc_len: u32) -> Result<()> {
        if self.current_doc.is_some_and(|current| current != doc) {
            self.finish_posting()?;
        }
        match self.current_doc {
            None => {
                self.current_doc = Some(doc);
                self.current_doc_len = doc_len;
            }
            Some(current) if current == doc && self.current_doc_len == doc_len => {}
            Some(_) => {
                return Err(Error::invariant(
                    "text v4 occurrence document metadata is inconsistent",
                ));
            }
        }
        if self
            .current_positions
            .last()
            .is_some_and(|previous| *previous >= position)
        {
            return Err(Error::invariant(
                "text v4 occurrence positions are not increasing",
            ));
        }
        let projected = self
            .current_positions
            .len()
            .saturating_add(1)
            .saturating_mul(size_of::<u32>())
            .saturating_add(size_of::<Posting>());
        if projected > self.block_budget {
            return Err(Error::precondition(format!(
                "one text v4 posting exceeds the {}-byte block workspace",
                self.block_budget
            )));
        }
        self.current_positions.push(position);
        Ok(())
    }

    fn finish_posting(&mut self) -> Result<()> {
        let Some(doc) = self.current_doc.take() else {
            return Ok(());
        };
        if self.current_positions.is_empty() {
            return Err(Error::invariant("text v4 posting has no positions"));
        }
        let posting_bytes = size_of::<Posting>()
            .saturating_add(self.current_positions.capacity() * size_of::<u32>());
        if !self.posting_block.is_empty()
            && (self.posting_block.len() == self.wire.postings_per_block
                || self.posting_block_bytes.saturating_add(posting_bytes) > self.block_budget)
        {
            self.flush_posting_block()?;
        }
        self.posting_block.push(Posting {
            doc,
            doc_len: self.current_doc_len,
            positions: std::mem::take(&mut self.current_positions),
        });
        self.posting_block_bytes = self.posting_block_bytes.saturating_add(posting_bytes);
        Ok(())
    }

    fn flush_posting_block(&mut self) -> Result<()> {
        if self.posting_block.is_empty() {
            return Ok(());
        }
        let raw = encode_posting_block(&self.posting_block)?;
        if raw.len() > self.block_budget {
            return Err(Error::precondition(
                "text v4 encoded posting block exceeds its workspace budget",
            ));
        }
        self.max_posting_block_bytes = self.max_posting_block_bytes.max(raw.len());
        let wire = write_compressed_block(
            self.output,
            &raw,
            self.wire.compression_level,
            "posting block",
        )?;
        let reference = PostingBlockRef {
            first_doc: self
                .posting_block
                .first()
                .map(|posting| posting.doc)
                .unwrap_or(0),
            last_doc: self
                .posting_block
                .last()
                .map(|posting| posting.doc)
                .unwrap_or(0),
            posting_count: u32::try_from(self.posting_block.len())
                .map_err(|_| Error::invariant("text v4 posting block count exceeds u32"))?,
            max_tf: self
                .posting_block
                .iter()
                .map(|posting| posting.positions.len() as u32)
                .max()
                .unwrap_or(0),
            min_doc_len: self
                .posting_block
                .iter()
                .map(|posting| posting.doc_len)
                .min()
                .unwrap_or(0),
            wire,
        };
        self.metadata_hasher.update(&serialize_bounded(
            &reference,
            MAX_RAW_BLOCK_BYTES,
            "posting metadata digest",
        )?);
        self.term_block_bytes = self
            .term_block_bytes
            .saturating_add(size_of::<PostingBlockRef>());
        if self.term_block_bytes > self.block_budget {
            return Err(Error::precondition(
                "one text v4 term has too many posting-block references",
            ));
        }
        self.term_blocks.push(reference);
        self.posting_block_count = self.posting_block_count.saturating_add(1);
        self.posting_block.clear();
        self.posting_block_bytes = 0;
        Ok(())
    }

    fn finish_term(&mut self) -> Result<()> {
        let Some(term) = self.current_term.take() else {
            return Ok(());
        };
        self.finish_posting()?;
        self.flush_posting_block()?;
        let live_doc_freq = self
            .term_blocks
            .iter()
            .try_fold(0_u64, |total, block| {
                total.checked_add(u64::from(block.posting_count))
            })
            .ok_or_else(|| Error::invariant("text v4 term live df overflows"))?;
        if self.reconciled {
            if !self.saw_explicit_df {
                return Err(Error::precondition(format!(
                    "text v4 reconciled postings for term {term:?} have no authenticated df"
                )));
            }
            let live_doc_freq = i128::from(live_doc_freq);
            let minimum = live_doc_freq - i128::from(self.mutation_count);
            let delta_df = i128::from(self.delta_df);
            if delta_df < minimum || delta_df > live_doc_freq {
                return Err(Error::precondition(format!(
                    "text v4 reconciled df {} for term {term:?} leaves feasible \
                     [{minimum}, {live_doc_freq}] bounds",
                    self.delta_df
                )));
            }
        }
        if self.delta_df == 0 && live_doc_freq == 0 {
            self.reset_term();
            return Ok(());
        }
        let entry = TermEntry {
            term,
            delta_df: self.delta_df,
            live_doc_freq,
            blocks: std::mem::take(&mut self.term_blocks),
        };
        let encoded = serialize_bounded(&entry, MAX_RAW_BLOCK_BYTES, "text v4 term spool entry")?;
        if encoded.len() > self.block_budget {
            return Err(Error::precondition(
                "one text v4 term directory entry exceeds its workspace budget",
            ));
        }
        self.term_spool
            .write_all(&(encoded.len() as u32).to_le_bytes())?;
        self.term_spool
            .write_all(&crc32fast::hash(&encoded).to_le_bytes())?;
        self.term_spool.write_all(&encoded)?;
        self.term_spool_bytes = self
            .term_spool_bytes
            .checked_add(TERM_SPOOL_HEADER_LEN + encoded.len() as u64)
            .ok_or_else(|| Error::invariant("text v4 term spool length overflows"))?;
        self.term_count = self
            .term_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text v4 term count overflows"))?;
        self.reset_term();
        Ok(())
    }

    fn reset_term(&mut self) {
        self.delta_df = 0;
        self.saw_explicit_df = false;
        self.current_doc = None;
        self.current_doc_len = 0;
        self.current_positions.clear();
        self.posting_block.clear();
        self.posting_block_bytes = 0;
        self.term_blocks.clear();
        self.term_block_bytes = 0;
    }

    fn finish(mut self) -> Result<ExternalAssembly> {
        if self.current_filter_key.is_some()
            || self.current_filter_property.is_some()
            || self.current_filter_value.is_some()
            || self.current_filter.is_some()
            || self.current_term.is_some()
            || self.saw_explicit_df
            || self.current_doc.is_some()
            || !self.current_positions.is_empty()
            || !self.posting_block.is_empty()
            || !self.term_blocks.is_empty()
        {
            return Err(Error::invariant(
                "text v4 assembler retained unfinished state",
            ));
        }
        self.term_spool.flush()?;
        let mut term_spool = self
            .term_spool
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        if term_spool.metadata()?.len() != self.term_spool_bytes {
            return Err(Error::invariant("text v4 term spool length changed"));
        }
        term_spool.rewind()?;
        self.filter_spool.flush()?;
        let mut filter_spool = self
            .filter_spool
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        if filter_spool.metadata()?.len() != self.filter_spool_bytes {
            return Err(Error::invariant("text v4 filter spool length changed"));
        }
        filter_spool.rewind()?;
        self.filter_entries.sort_by(|left, right| {
            left.property
                .cmp(&right.property)
                .then_with(|| left.value.cmp(&right.value))
        });
        if self
            .filter_entries
            .windows(2)
            .any(|pair| pair[0].property == pair[1].property && pair[0].value == pair[1].value)
        {
            return Err(Error::invariant(
                "text v4 filter spool contains duplicate property/value postings",
            ));
        }
        let mut previous_property: Option<&str> = None;
        let mut property_distinct = 0usize;
        for entry in &self.filter_entries {
            if previous_property != Some(entry.property.as_str()) {
                previous_property = Some(&entry.property);
                property_distinct = 0;
            }
            property_distinct = property_distinct
                .checked_add(1)
                .ok_or_else(|| Error::precondition("text v4 filter distinct count overflows"))?;
            if property_distinct > self.max_filter_distinct_per_property {
                return Err(Error::precondition(format!(
                    "text v4 filter property {} exceeds explicit distinct-value cap {}",
                    entry.property, self.max_filter_distinct_per_property
                )));
            }
        }
        let postings_end = self.output.stream_position()?;
        Ok(ExternalAssembly {
            term_spool,
            term_spool_bytes: self.term_spool_bytes,
            term_count: self.term_count,
            postings_start: self.postings_start,
            postings_end,
            posting_block_count: self.posting_block_count,
            postings_metadata_xxh3: non_zero_digest(self.metadata_hasher),
            max_posting_block_bytes: self.max_posting_block_bytes,
            filter_spool,
            filter_spool_bytes: self.filter_spool_bytes,
            filter_entries: self.filter_entries,
            filter_directory_bytes: self.filter_directory_bytes,
            max_filter_bytes: self.max_filter_bytes,
        })
    }
}

fn emit_dictionary_blocks(
    output: &mut File,
    mut term_spool: File,
    term_count: u64,
    wire: TextV4BuildOptions,
    block_budget: usize,
    directory_budget: usize,
) -> Result<(Vec<DictionaryBlockRef>, usize, usize)> {
    term_spool.rewind()?;
    let spool_len = term_spool.metadata()?.len();
    let mut consumed = 0u64;
    let mut remaining = term_count;
    let mut directory = Vec::new();
    let mut directory_bytes = 0usize;
    let mut max_block_bytes = 0usize;
    let mut previous_term: Option<String> = None;
    while remaining > 0 {
        let mut entries = Vec::<TermEntry>::new();
        let mut logical_bytes = 8usize;
        while remaining > 0 && entries.len() < wire.terms_per_dictionary_block {
            let record_start = term_spool.stream_position()?;
            let len = read_u32_io(&mut term_spool)? as usize;
            let expected_crc = read_u32_io(&mut term_spool)?;
            if len == 0 || len > block_budget {
                return Err(Error::invariant(
                    "text v4 term spool record length is invalid",
                ));
            }
            if !entries.is_empty()
                && logical_bytes
                    .saturating_add(len)
                    .saturating_add(size_of::<TermEntry>())
                    > block_budget
            {
                term_spool.seek(SeekFrom::Start(record_start))?;
                break;
            }
            let mut encoded = vec![0; len];
            term_spool.read_exact(&mut encoded)?;
            if crc32fast::hash(&encoded) != expected_crc {
                return Err(Error::invariant("text v4 term spool checksum mismatch"));
            }
            let entry: TermEntry =
                super::deserialize_bounded(&encoded, MAX_RAW_BLOCK_BYTES, "term spool entry")?;
            if previous_term
                .as_ref()
                .is_some_and(|previous| previous >= &entry.term)
            {
                return Err(Error::invariant(
                    "text v4 term spool is not strictly sorted",
                ));
            }
            previous_term = Some(entry.term.clone());
            logical_bytes = logical_bytes
                .saturating_add(len)
                .saturating_add(size_of::<TermEntry>());
            entries.push(entry);
            remaining -= 1;
            consumed = term_spool.stream_position()?;
        }
        if entries.is_empty() {
            return Err(Error::precondition(
                "one text v4 dictionary entry cannot fit the configured block workspace",
            ));
        }
        let raw = serialize_bounded(&entries, MAX_RAW_BLOCK_BYTES, "dictionary block")?;
        if raw.len() > block_budget {
            return Err(Error::precondition(
                "text v4 dictionary block exceeds configured workspace",
            ));
        }
        max_block_bytes = max_block_bytes.max(raw.len());
        let reference = DictionaryBlockRef {
            first_term: entries
                .first()
                .map(|entry| entry.term.clone())
                .ok_or_else(|| Error::invariant("text v4 dictionary block is empty"))?,
            last_term: entries
                .last()
                .map(|entry| entry.term.clone())
                .ok_or_else(|| Error::invariant("text v4 dictionary block is empty"))?,
            term_count: u32::try_from(entries.len())
                .map_err(|_| Error::invariant("text v4 dictionary term count exceeds u32"))?,
            wire: write_compressed_block(output, &raw, wire.compression_level, "dictionary block")?,
        };
        directory_bytes = directory_bytes
            .saturating_add(size_of::<DictionaryBlockRef>())
            .saturating_add(reference.first_term.capacity())
            .saturating_add(reference.last_term.capacity());
        if directory_bytes > directory_budget {
            return Err(Error::precondition(
                "text v4 sparse dictionary metadata exceeds configured resident budget",
            ));
        }
        directory.push(reference);
    }
    if consumed != spool_len || term_spool.stream_position()? != spool_len {
        return Err(Error::invariant("text v4 term spool has trailing bytes"));
    }
    Ok((directory, max_block_bytes, directory_bytes))
}

fn emit_filter_blocks(
    output: &mut File,
    mut spool: File,
    entries: Vec<FilterSpoolEntry>,
    complete_filter_properties: &[String],
    row_count: u64,
    config: &TextV4ExternalBuildConfig,
) -> Result<(Vec<FilterBlockRef>, usize, usize)> {
    let spool_len = spool.metadata()?.len();
    let mut entries = entries.into_iter().peekable();
    let mut references = Vec::with_capacity(complete_filter_properties.len());
    let mut max_raw_bytes = 0usize;
    let mut directory_bytes = 0usize;
    for property in complete_filter_properties {
        let mut values = Vec::new();
        while entries
            .peek()
            .is_some_and(|entry| entry.property.as_str() == property)
        {
            let entry = entries
                .next()
                .ok_or_else(|| Error::invariant("text v4 filter entry disappeared"))?;
            let end = entry
                .offset
                .checked_add(u64::from(entry.len))
                .ok_or_else(|| Error::invariant("text v4 filter spool range overflows"))?;
            if entry.len == 0
                || usize::try_from(entry.len).unwrap_or(usize::MAX) > config.block_budget()
                || end > spool_len
            {
                return Err(Error::invariant(
                    "text v4 filter spool reference is invalid",
                ));
            }
            spool.seek(SeekFrom::Start(entry.offset))?;
            let mut raw = vec![0u8; entry.len as usize];
            spool.read_exact(&mut raw)?;
            if crc32fast::hash(&raw) != entry.crc32 {
                return Err(Error::invariant("text v4 filter spool checksum mismatch"));
            }
            validate_filter_spool_posting(&raw, entry.encoding, entry.cardinality, row_count)?;
            max_raw_bytes = max_raw_bytes.max(raw.len());
            let reference = FilterValueRef {
                value: entry.value,
                cardinality: entry.cardinality,
                encoding: entry.encoding,
                wire: write_compressed_block(
                    output,
                    &raw,
                    config.wire.compression_level,
                    "filter posting block",
                )?,
            };
            directory_bytes = directory_bytes
                .saturating_add(size_of::<FilterValueRef>())
                .saturating_add(filter_value_resident_bytes(&reference.value));
            if directory_bytes > config.directory_budget() {
                return Err(Error::precondition(
                    "text v4 filter output directory exceeds configured resident budget",
                ));
            }
            values.push(reference);
        }
        directory_bytes = directory_bytes
            .saturating_add(size_of::<FilterBlockRef>())
            .saturating_add(property.capacity());
        if directory_bytes > config.directory_budget() {
            return Err(Error::precondition(
                "text v4 filter output directory exceeds configured resident budget",
            ));
        }
        references.push(FilterBlockRef {
            property: property.clone(),
            row_count,
            values,
        });
    }
    if let Some(entry) = entries.next() {
        return Err(Error::invariant(format!(
            "text v4 filter spool contains unadvertised property {}",
            entry.property
        )));
    }
    Ok((references, max_raw_bytes, directory_bytes))
}

fn validate_filter_spool_posting(
    raw: &[u8],
    encoding: FilterPostingEncoding,
    cardinality: u64,
    row_count: u64,
) -> Result<()> {
    if cardinality == 0 || cardinality > row_count || raw.is_empty() {
        return Err(Error::invariant(
            "text v4 filter spool cardinality is invalid",
        ));
    }
    match encoding {
        FilterPostingEncoding::SparseDeltaVarint => {
            let mut cursor = 0usize;
            let mut previous = 0u64;
            for index in 0..cardinality {
                let delta = decode_u64_varint(raw, &mut cursor)?;
                if index > 0 && delta == 0 {
                    return Err(Error::invariant(
                        "text v4 filter spool ordinals are duplicated",
                    ));
                }
                let ordinal = previous
                    .checked_add(delta)
                    .ok_or_else(|| Error::invariant("text v4 filter spool ordinal overflows"))?;
                if ordinal >= row_count {
                    return Err(Error::invariant(
                        "text v4 filter spool ordinal leaves document table",
                    ));
                }
                previous = ordinal;
            }
            if cursor != raw.len() {
                return Err(Error::invariant(
                    "text v4 sparse filter spool has trailing bytes",
                ));
            }
        }
        FilterPostingEncoding::DenseBitmap => {
            let expected = bitmap_words(row_count)?
                .checked_mul(size_of::<u64>())
                .ok_or_else(|| Error::invariant("text v4 dense filter size overflows"))?;
            if raw.len() != expected {
                return Err(Error::invariant(
                    "text v4 dense filter spool length is invalid",
                ));
            }
            let mut count = 0u64;
            let mut last_word = 0u64;
            for bytes in raw.chunks_exact(8) {
                last_word = u64::from_le_bytes(bytes.try_into().expect("fixed filter bitmap word"));
                count += u64::from(last_word.count_ones());
            }
            let remainder = row_count % 64;
            if remainder != 0 && last_word & (!0u64 << remainder) != 0 {
                return Err(Error::invariant(
                    "text v4 dense filter has bits beyond document table",
                ));
            }
            if count != cardinality {
                return Err(Error::invariant(
                    "text v4 dense filter/cardinality mismatch",
                ));
            }
        }
    }
    Ok(())
}

fn read_u32_io(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_io(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::io::Read as _;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::search_lsm::{SearchEventRange, SearchLsmKind, SearchLsmStatus};
    use crate::sst::search_delta::SearchVersionRangeSource;
    use async_trait::async_trait;
    use bytes::Bytes;
    use uuid::Uuid;

    #[derive(Debug)]
    struct MemorySource {
        body: Bytes,
        ranges: Mutex<Vec<Range<u64>>>,
    }

    #[async_trait]
    impl SearchVersionRangeSource for MemorySource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
            self.ranges.lock().unwrap().push(range.clone());
            let start = usize::try_from(range.start)
                .map_err(|_| Error::invariant("test range start does not fit usize"))?;
            let end = usize::try_from(range.end)
                .map_err(|_| Error::invariant("test range end does not fit usize"))?;
            self.body
                .get(start..end)
                .map(Bytes::copy_from_slice)
                .ok_or_else(|| Error::invariant("test range leaves text v4 body"))
        }
    }

    fn node(value: u64) -> [u8; 16] {
        let mut id = [0; 16];
        id[8..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn state() -> SearchLsmState {
        SearchLsmState {
            index_name: "external_fts".into(),
            kind: SearchLsmKind::Text,
            catalog_signature: "external-catalog".into(),
            generation_id: Uuid::from_u128(501),
            status: SearchLsmStatus::Building,
            ..SearchLsmState::default()
        }
    }

    fn context() -> TextV4BuildContext {
        TextV4BuildContext {
            sst_id: Uuid::from_u128(502),
            event_ranges: vec![SearchEventRange::new(1, 10_000)],
            complete_filter_properties: vec!["vigente".into()],
        }
    }

    fn payload(text: String, vigente: bool) -> super::super::TextV4Payload {
        super::super::TextV4Payload {
            text,
            filters: BTreeMap::from([("vigente".into(), SearchFilterValue::Bool(vigente))]),
        }
    }

    fn checked_sum(values: &[i64]) -> i64 {
        values
            .iter()
            .try_fold(0_i64, |sum, value| sum.checked_add(*value))
            .expect("test statistic sum")
    }

    fn live_record(
        node_id: [u8; 16],
        lsn: u64,
        source_ordinal: u64,
        payload: &TextV4Payload,
    ) -> SearchVersionRecord {
        SearchVersionRecord::live(
            node_id,
            lsn,
            super::super::text_v4_payload_fingerprint(payload).unwrap(),
            source_ordinal,
        )
    }

    async fn open_artifact(mut artifact: TextV4ExternalArtifact) -> super::super::TextV4Reader {
        let mut body = Vec::new();
        artifact.file.read_to_end(&mut body).unwrap();
        let segment = artifact.output.segment;
        super::super::TextV4Reader::open(
            Arc::new(MemorySource {
                body: Bytes::from(body),
                ranges: Mutex::new(Vec::new()),
            }),
            artifact.len,
            &state(),
            &segment,
        )
        .await
        .unwrap()
    }

    #[test]
    fn tiny_budget_forces_external_runs_and_peak_stays_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let config = TextV4ExternalBuildConfig {
            memory_budget_bytes: MIN_MEMORY_BUDGET_BYTES,
            spool_directory: Some(directory.path().into()),
            max_filter_distinct_per_property: 8,
            wire: TextV4BuildOptions {
                postings_per_block: 8,
                terms_per_dictionary_block: 4,
                compression_level: 1,
            },
        };
        let mut builder =
            TextV4ExternalBuilder::with_config(&state(), context(), config.clone()).unwrap();
        for value in 1..=800u64 {
            builder
                .push(TextV4Mutation {
                    node_id: node(value),
                    lsn: value,
                    before: None,
                    after: Some(payload(
                        format!(
                            "contrato laboral articulo termino_{value} repetido repetido vigente"
                        ),
                        value % 2 == 0,
                    )),
                })
                .unwrap();
        }
        let artifact = builder.finish().unwrap().unwrap();
        assert!(artifact.metrics.initial_run_count > 1);
        assert!(artifact.metrics.run_merge_count > 0);
        assert!(
            artifact.metrics.peak_logical_memory_bytes <= config.memory_budget_bytes,
            "{:?}",
            artifact.metrics
        );
        assert!(artifact.metrics.spool_bytes_written > 0);
        assert_eq!(artifact.metrics.filter_value_count, 2);
        assert_eq!(artifact.output.segment.live_payload_count, 800);
    }

    #[test]
    fn external_output_is_deterministic_and_invalid_single_record_fails_without_oom() {
        fn build() -> (Vec<u8>, TextV4ExternalBuildMetrics) {
            let directory = tempfile::tempdir().unwrap();
            let config = TextV4ExternalBuildConfig {
                memory_budget_bytes: MIN_MEMORY_BUDGET_BYTES,
                spool_directory: Some(directory.path().into()),
                max_filter_distinct_per_property: 8,
                wire: TextV4BuildOptions::default(),
            };
            let mut builder =
                TextV4ExternalBuilder::with_config(&state(), context(), config).unwrap();
            for value in 1..=32u64 {
                builder
                    .push(TextV4Mutation {
                        node_id: node(value),
                        lsn: value,
                        before: None,
                        after: Some(payload(format!("norma laboral numero {value}"), true)),
                    })
                    .unwrap();
            }
            let mut artifact = builder.finish().unwrap().unwrap();
            let mut body = Vec::new();
            artifact.file.read_to_end(&mut body).unwrap();
            (body, artifact.metrics)
        }
        let (first, first_metrics) = build();
        let (second, second_metrics) = build();
        assert_eq!(first, second);
        assert_eq!(first_metrics, second_metrics);

        let directory = tempfile::tempdir().unwrap();
        let config = TextV4ExternalBuildConfig {
            memory_budget_bytes: MIN_MEMORY_BUDGET_BYTES,
            spool_directory: Some(directory.path().into()),
            ..TextV4ExternalBuildConfig::default()
        };
        let mut builder = TextV4ExternalBuilder::with_config(&state(), context(), config).unwrap();
        let huge = "x ".repeat(MIN_MEMORY_BUDGET_BYTES);
        let error = builder
            .push(TextV4Mutation {
                node_id: node(1),
                lsn: 1,
                before: None,
                after: Some(payload(huge, true)),
            })
            .unwrap_err();
        assert!(error.to_string().contains("per-record workspace"));
    }

    #[tokio::test]
    async fn reconciled_delta_preserves_authenticated_sums_and_stats_only_terms() {
        let directory = tempfile::tempdir().unwrap();
        let config = TextV4ExternalBuildConfig {
            memory_budget_bytes: MIN_MEMORY_BUDGET_BYTES,
            spool_directory: Some(directory.path().into()),
            max_filter_distinct_per_property: 8,
            wire: TextV4BuildOptions {
                postings_per_block: 1,
                terms_per_dictionary_block: 2,
                compression_level: 1,
            },
        };
        let stats = ReconciledTextDeltaStats {
            doc_count_delta: checked_sum(&[1, -1]),
            total_len_delta: checked_sum(&[2, -5]),
        };
        let mut builder =
            TextV4ExternalBuilder::with_config_reconciled(&state(), context(), config, stats)
                .unwrap();
        let live = payload("alpha gamma".into(), true);
        let winner = live_record(node(1), 101, 9_999, &live);
        builder.push_reconciled(winner, Some(live)).unwrap();
        builder
            .push_term_delta("alpha".into(), checked_sum(&[1, -1]))
            .unwrap();
        builder
            .push_reconciled(
                SearchVersionRecord::suppress(node(2), 202, search_suppress_fingerprint()),
                None,
            )
            .unwrap();
        builder
            .push_term_delta("desaparecido".into(), checked_sum(&[-1]))
            .unwrap();
        builder
            .push_term_delta("gamma".into(), checked_sum(&[1, 1, -1]))
            .unwrap();
        builder.push_term_delta("zero".into(), 0).unwrap();

        let artifact = builder.finish().unwrap().unwrap();
        assert_eq!(artifact.output.segment.mutation_count, 2);
        assert_eq!(artifact.output.segment.live_payload_count, 1);
        assert_eq!(artifact.output.segment.suppress_count, 1);
        assert_eq!(artifact.output.segment.min_lsn, 101);
        assert_eq!(artifact.output.segment.max_lsn, 202);
        assert_eq!(
            artifact.output.segment.stats,
            crate::search_lsm::SearchSegmentStats::Text {
                doc_count: crate::search_lsm::SearchStatValue::Delta(0),
                total_len: crate::search_lsm::SearchStatValue::Delta(-3),
                term_df_violation_count: 0,
            }
        );

        let reader = open_artifact(artifact).await;
        assert_eq!(reader.delta_docs(), 0);
        assert_eq!(reader.delta_total_len(), -3);
        let mut deltas = Vec::new();
        for block in 0..reader.term_delta_block_count() {
            deltas.extend(reader.read_term_delta_block(block).await.unwrap());
        }
        assert_eq!(
            deltas,
            vec![
                super::super::TextV4TermDelta {
                    term: "alpha".into(),
                    delta_df: 0,
                },
                super::super::TextV4TermDelta {
                    term: "desaparecido".into(),
                    delta_df: -1,
                },
                super::super::TextV4TermDelta {
                    term: "gamma".into(),
                    delta_df: 1,
                },
            ],
            "zero-without-postings is omitted but negative stats-only terms survive"
        );
        let live_version = reader
            .version_reader()
            .point_probe(node(1))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(live_version.lsn, 101);
        assert_eq!(live_version.payload_fingerprint, winner.payload_fingerprint);
        assert!(matches!(
            live_version.operation,
            SearchVersionOperation::Live { payload_ordinal: 0 }
        ));
        assert!(matches!(
            reader
                .version_reader()
                .point_probe(node(2))
                .await
                .unwrap()
                .unwrap()
                .operation,
            SearchVersionOperation::Suppress
        ));
        reader.verify_all().await.unwrap();
    }

    #[test]
    fn reconciled_delta_rejects_missing_impossible_and_mismatched_statistics() {
        fn config(directory: &tempfile::TempDir) -> TextV4ExternalBuildConfig {
            TextV4ExternalBuildConfig {
                memory_budget_bytes: MIN_MEMORY_BUDGET_BYTES,
                spool_directory: Some(directory.path().into()),
                max_filter_distinct_per_property: 8,
                wire: TextV4BuildOptions::default(),
            }
        }

        let directory = tempfile::tempdir().unwrap();
        let mut missing = TextV4ExternalBuilder::with_config_reconciled(
            &state(),
            context(),
            config(&directory),
            ReconciledTextDeltaStats {
                doc_count_delta: 1,
                total_len_delta: 1,
            },
        )
        .unwrap();
        let alpha = payload("alpha".into(), true);
        missing
            .push_reconciled(live_record(node(1), 1, 17, &alpha), Some(alpha))
            .unwrap();
        assert!(missing
            .finish()
            .unwrap_err()
            .to_string()
            .contains("no authenticated df"));

        for impossible in [i64::MIN, i64::MAX] {
            let directory = tempfile::tempdir().unwrap();
            let mut builder = TextV4ExternalBuilder::with_config_reconciled(
                &state(),
                context(),
                config(&directory),
                ReconciledTextDeltaStats {
                    doc_count_delta: 1,
                    total_len_delta: 1,
                },
            )
            .unwrap();
            let alpha = payload("alpha".into(), true);
            builder
                .push_reconciled(live_record(node(1), 1, 0, &alpha), Some(alpha))
                .unwrap();
            builder.push_term_delta("alpha".into(), impossible).unwrap();
            assert!(builder
                .finish()
                .unwrap_err()
                .to_string()
                .contains("leaves feasible"));
        }

        let directory = tempfile::tempdir().unwrap();
        let mut coarse = TextV4ExternalBuilder::with_config_reconciled(
            &state(),
            context(),
            config(&directory),
            ReconciledTextDeltaStats {
                doc_count_delta: 1,
                total_len_delta: 0,
            },
        )
        .unwrap();
        coarse
            .push_reconciled(
                SearchVersionRecord::suppress(node(1), 1, search_suppress_fingerprint()),
                None,
            )
            .unwrap();
        assert!(coarse
            .finish()
            .unwrap_err()
            .to_string()
            .contains("document delta"));

        let directory = tempfile::tempdir().unwrap();
        let mut fingerprint = TextV4ExternalBuilder::with_config_reconciled(
            &state(),
            context(),
            config(&directory),
            ReconciledTextDeltaStats {
                doc_count_delta: 1,
                total_len_delta: 1,
            },
        )
        .unwrap();
        let alpha = payload("alpha".into(), true);
        let mut winner = live_record(node(1), 1, 0, &alpha);
        winner.payload_fingerprint ^= 1;
        assert!(fingerprint
            .push_reconciled(winner, Some(alpha))
            .unwrap_err()
            .to_string()
            .contains("fingerprint"));
    }

    #[tokio::test]
    async fn reconciled_delta_external_sort_and_dictionary_reads_stay_bounded() {
        const DOCUMENTS: u64 = 800;
        let directory = tempfile::tempdir().unwrap();
        let config = TextV4ExternalBuildConfig {
            memory_budget_bytes: MIN_MEMORY_BUDGET_BYTES,
            spool_directory: Some(directory.path().into()),
            max_filter_distinct_per_property: 8,
            wire: TextV4BuildOptions {
                postings_per_block: 8,
                terms_per_dictionary_block: 16,
                compression_level: 1,
            },
        };
        let mut builder = TextV4ExternalBuilder::with_config_reconciled(
            &state(),
            context(),
            config.clone(),
            ReconciledTextDeltaStats {
                doc_count_delta: DOCUMENTS as i64,
                total_len_delta: DOCUMENTS as i64,
            },
        )
        .unwrap();
        for value in 1..=DOCUMENTS {
            let document = payload(format!("term{value:05}"), value % 2 == 0);
            let winner = live_record(node(value), value, value + 50_000, &document);
            builder.push_reconciled(winner, Some(document)).unwrap();
        }
        for value in 1..=DOCUMENTS {
            builder
                .push_term_delta(format!("term{value:05}"), 1)
                .unwrap();
        }
        let artifact = builder.finish().unwrap().unwrap();
        assert!(artifact.metrics.initial_run_count > 1);
        assert!(artifact.metrics.run_merge_count > 0);
        assert!(artifact.metrics.peak_logical_memory_bytes <= config.memory_budget_bytes);
        let reader = open_artifact(artifact).await;
        let mut seen = 0usize;
        for block in 0..reader.term_delta_block_count() {
            let page = reader.read_term_delta_block(block).await.unwrap();
            assert!(page.len() <= config.wire.terms_per_dictionary_block);
            seen += page.len();
        }
        assert_eq!(seen, DOCUMENTS as usize);
    }

    #[test]
    fn ten_thousand_unique_filter_values_are_sparse_linear_and_never_omitted() {
        let directory = tempfile::tempdir().unwrap();
        let mut high_cardinality_context = context();
        high_cardinality_context.complete_filter_properties = vec!["codigo".into()];
        let config = TextV4ExternalBuildConfig {
            memory_budget_bytes: 16 * 1024 * 1024,
            spool_directory: Some(directory.path().into()),
            max_filter_distinct_per_property: 10_000,
            wire: TextV4BuildOptions {
                postings_per_block: 64,
                terms_per_dictionary_block: 32,
                compression_level: 1,
            },
        };
        let mut builder =
            TextV4ExternalBuilder::with_config(&state(), high_cardinality_context, config.clone())
                .unwrap();
        for value in 0..10_000u64 {
            builder
                .push(TextV4Mutation {
                    node_id: node(value + 1),
                    lsn: value + 1,
                    before: None,
                    after: Some(super::super::TextV4Payload {
                        text: String::new(),
                        filters: BTreeMap::from([(
                            "codigo".into(),
                            SearchFilterValue::String(format!("codigo-{value:05}")),
                        )]),
                    }),
                })
                .unwrap();
        }
        let artifact = builder.finish().unwrap().unwrap();
        assert_eq!(artifact.metrics.filter_value_count, 10_000);
        assert_eq!(
            artifact.output.segment.complete_filter_properties,
            vec!["codigo".to_owned()]
        );
        assert!(
            artifact.metrics.filter_spool_bytes < 40_000,
            "{:?}",
            artifact.metrics
        );
        assert!(
            artifact.len < 16 * 1024 * 1024,
            "high-cardinality wire grew non-linearly: {} bytes",
            artifact.len
        );
        assert!(artifact.metrics.peak_logical_memory_bytes <= config.memory_budget_bytes);
    }

    #[test]
    fn explicit_filter_cardinality_cap_fails_instead_of_degrading_completeness() {
        let directory = tempfile::tempdir().unwrap();
        let mut capped_context = context();
        capped_context.complete_filter_properties = vec!["codigo".into()];
        let config = TextV4ExternalBuildConfig {
            memory_budget_bytes: MIN_MEMORY_BUDGET_BYTES,
            spool_directory: Some(directory.path().into()),
            max_filter_distinct_per_property: 4,
            ..TextV4ExternalBuildConfig::default()
        };
        let mut builder =
            TextV4ExternalBuilder::with_config(&state(), capped_context, config).unwrap();
        for value in 0..5u64 {
            builder
                .push(TextV4Mutation {
                    node_id: node(value + 1),
                    lsn: value + 1,
                    before: None,
                    after: Some(super::super::TextV4Payload {
                        text: String::new(),
                        filters: BTreeMap::from([(
                            "codigo".into(),
                            SearchFilterValue::String(value.to_string()),
                        )]),
                    }),
                })
                .unwrap();
        }
        let error = builder.finish().unwrap_err();
        assert!(error.to_string().contains("explicit distinct-value cap"));
    }
}
