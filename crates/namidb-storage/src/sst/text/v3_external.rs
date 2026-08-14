//! Bounded-memory external builder for the `NAMIFT03` wire format.
//!
//! Documents are already NodeId-sorted when they reach this builder. The fixed
//! document table is written directly to the final artifact, while token
//! occurrences are sorted in bounded in-memory runs. Runs are folded like a
//! binary counter, so the number of live spool files is logarithmic rather
//! than proportional to the corpus. The final run is consumed term-by-term to
//! emit the exact posting/dictionary block layout used by the in-memory v3
//! builder.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};

use serde::ser::{Error as _, SerializeSeq};
use serde::{Serialize, Serializer};

use super::{
    BlockRef, DictionaryBlockRef, PostingBlockRef, DOC_RECORD_LEN, FORMAT_VERSION, MAGIC_V3,
    MAX_BLOCK_BYTES, MAX_FOOTER_BYTES, MAX_RAW_BLOCK_BYTES, POSTINGS_PER_BLOCK,
    TERMS_PER_DICTIONARY_BLOCK, TRAILER_MAGIC, ZSTD_LEVEL,
};
use crate::error::{Error, Result};
use crate::sst::text::TextIndexBuildStats;

const DEFAULT_INDEX_BUILD_MEMORY_BYTES: usize = 64 * 1024 * 1024;
const RUN_MAGIC: &[u8; 8] = b"NFTXRUN1";
const RUN_HEADER_LEN: u64 = 8 + 8 + 8 + 4;
const DIRECTORY_RECORD_HEADER_LEN: u64 = 4 + 4;
const IO_BUFFER_BYTES: usize = 16 * 1024;
const SHARED_SPOOL_DIR_ENV: &str = "NAMIDB_SPOOL_DIR";

/// Environment variable controlling the logical occurrence-sort buffer.
pub const INDEX_BUILD_MEMORY_ENV: &str = "NAMIDB_INDEX_BUILD_MEMORY_BYTES";
/// Preferred disk-backed directory shared by all external search-index builds.
pub const INDEX_BUILD_SPOOL_DIR_ENV: &str = "NAMIDB_INDEX_BUILD_SPOOL_DIR";
/// Environment variable selecting a disk-backed compaction spool directory.
pub const COMPACTION_SPOOL_DIR_ENV: &str = "NAMIDB_COMPACTION_SPOOL_DIR";

/// Options for one external full-text build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalTextIndexBuildOptions {
    /// Maximum logical bytes retained by the sortable occurrence buffer.
    ///
    /// A single occurrence larger than this limit is rejected, rather than
    /// silently violating the memory contract. Posting and dictionary blocks
    /// have independent, wire-format bounds (`MAX_RAW_BLOCK_BYTES`) and are
    /// emitted one at a time.
    pub memory_budget_bytes: usize,
    /// Directory for anonymous temporary files. `None` prefers `/var/tmp` on
    /// Unix and otherwise uses the platform's secure tempfile directory.
    pub spool_directory: Option<PathBuf>,
}

impl ExternalTextIndexBuildOptions {
    pub fn from_env() -> Result<Self> {
        let memory_budget_bytes = match env::var(INDEX_BUILD_MEMORY_ENV) {
            Ok(value) if value.trim().is_empty() => DEFAULT_INDEX_BUILD_MEMORY_BYTES,
            Ok(value) => value.trim().parse::<usize>().map_err(|error| {
                Error::precondition(format!(
                    "{INDEX_BUILD_MEMORY_ENV} must be an exact positive byte count: {error}"
                ))
            })?,
            Err(env::VarError::NotPresent) => DEFAULT_INDEX_BUILD_MEMORY_BYTES,
            Err(env::VarError::NotUnicode(_)) => {
                return Err(Error::precondition(format!(
                    "{INDEX_BUILD_MEMORY_ENV} is not valid UTF-8"
                )));
            }
        };
        if memory_budget_bytes == 0 {
            return Err(Error::precondition(format!(
                "{INDEX_BUILD_MEMORY_ENV} must be greater than zero"
            )));
        }

        let spool_directory = env::var_os(INDEX_BUILD_SPOOL_DIR_ENV)
            .filter(|value| !value.is_empty())
            .or_else(|| env::var_os(COMPACTION_SPOOL_DIR_ENV).filter(|value| !value.is_empty()))
            .or_else(|| env::var_os(SHARED_SPOOL_DIR_ENV).filter(|value| !value.is_empty()))
            .map(PathBuf::from);
        if let Some(directory) = spool_directory.as_deref() {
            validate_spool_directory(directory)?;
        }
        Ok(Self {
            memory_budget_bytes,
            spool_directory,
        })
    }

    #[cfg(test)]
    fn for_test(memory_budget_bytes: usize, spool_directory: &Path) -> Self {
        Self {
            memory_budget_bytes,
            spool_directory: Some(spool_directory.to_path_buf()),
        }
    }
}

impl Default for ExternalTextIndexBuildOptions {
    fn default() -> Self {
        Self {
            memory_budget_bytes: DEFAULT_INDEX_BUILD_MEMORY_BYTES,
            spool_directory: None,
        }
    }
}

/// Logical resource counters collected without sampling process RSS.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExternalTextIndexBuildMetrics {
    /// Configured occurrence-sort memory ceiling.
    pub memory_budget_bytes: usize,
    /// Maximum accounted occurrence bytes held before a run flush.
    pub max_buffer_bytes: usize,
    /// Initial sorted runs produced from the bounded buffer.
    pub initial_run_count: u64,
    /// Streaming two-way merges performed while folding runs.
    pub run_merge_count: u64,
    /// Bytes written to occurrence and sparse-directory spool files, including
    /// merge rewrites. This is diagnostic I/O, not final artifact size.
    pub spool_bytes_written: u64,
}

/// File-backed `NAMIFT03` artifact ready for streaming or multipart upload.
///
/// The file cursor is rewound to zero. Dropping the artifact closes and
/// removes its anonymous tempfile.
#[derive(Debug)]
pub struct TextIndexFileArtifact {
    file: File,
    len: u64,
    stats: TextIndexBuildStats,
    metrics: ExternalTextIndexBuildMetrics,
}

impl TextIndexFileArtifact {
    pub fn file(&self) -> &File {
        &self.file
    }

    pub fn file_mut(&mut self) -> &mut File {
        &mut self.file
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn stats(&self) -> &TextIndexBuildStats {
        &self.stats
    }

    pub fn metrics(&self) -> ExternalTextIndexBuildMetrics {
        self.metrics
    }

    pub fn into_parts(
        self,
    ) -> (
        File,
        u64,
        TextIndexBuildStats,
        ExternalTextIndexBuildMetrics,
    ) {
        (self.file, self.len, self.stats, self.metrics)
    }
}

/// Synchronous, bounded-memory collector for sorted `(NodeId, text)` rows.
pub struct TextIndexExternalBuilder {
    options: ExternalTextIndexBuildOptions,
    spool: SpoolFactory,
    output: File,
    occurrences: Vec<Occurrence>,
    occurrence_term_bytes: usize,
    occurrence_buffer_bytes: usize,
    levels: Vec<Option<RunFile>>,
    doc_count: u32,
    total_len: u64,
    min_node_id: Option<[u8; 16]>,
    max_node_id: Option<[u8; 16]>,
    last_node_id: Option<[u8; 16]>,
    metrics: ExternalTextIndexBuildMetrics,
    poisoned: bool,
}

impl std::fmt::Debug for TextIndexExternalBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextIndexExternalBuilder")
            .field("options", &self.options)
            .field("doc_count", &self.doc_count)
            .field("total_len", &self.total_len)
            .field("buffered_occurrences", &self.occurrences.len())
            .field("occurrence_buffer_bytes", &self.occurrence_buffer_bytes)
            .field("live_run_levels", &self.levels.len())
            .field("metrics", &self.metrics)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl TextIndexExternalBuilder {
    /// Create a builder from `NAMIDB_INDEX_BUILD_MEMORY_BYTES`. Spool
    /// precedence is `NAMIDB_INDEX_BUILD_SPOOL_DIR`, then
    /// `NAMIDB_COMPACTION_SPOOL_DIR`, then `NAMIDB_SPOOL_DIR`.
    pub fn new() -> Result<Self> {
        Self::with_options(ExternalTextIndexBuildOptions::from_env()?)
    }

    /// Explicit options are useful for embedding and deterministic tests.
    pub fn with_options(options: ExternalTextIndexBuildOptions) -> Result<Self> {
        if options.memory_budget_bytes == 0 {
            return Err(Error::precondition(
                "text external build memory budget must be greater than zero",
            ));
        }
        if let Some(directory) = options.spool_directory.as_deref() {
            validate_spool_directory(directory)?;
        }
        let spool = SpoolFactory::new(options.spool_directory.clone());
        let mut output = spool.create()?;
        output.write_all(MAGIC_V3)?;
        Ok(Self {
            metrics: ExternalTextIndexBuildMetrics {
                memory_budget_bytes: options.memory_budget_bytes,
                ..Default::default()
            },
            options,
            spool,
            output,
            occurrences: Vec::new(),
            occurrence_term_bytes: 0,
            occurrence_buffer_bytes: 0,
            levels: Vec::new(),
            doc_count: 0,
            total_len: 0,
            min_node_id: None,
            max_node_id: None,
            last_node_id: None,
            poisoned: false,
        })
    }

    /// Append one document. NodeIds must be strictly ascending.
    pub fn push(&mut self, member: ([u8; 16], String)) -> Result<()> {
        if self.poisoned {
            return Err(Error::precondition(
                "text external builder is poisoned by an earlier push failure",
            ));
        }
        let result = self.push_inner(member);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    pub fn metrics(&self) -> ExternalTextIndexBuildMetrics {
        self.metrics
    }

    /// Finish the exact v3 object. An empty input produces no index.
    pub fn finish(mut self) -> Result<Option<TextIndexFileArtifact>> {
        if self.poisoned {
            return Err(Error::precondition(
                "cannot finish a poisoned text external builder",
            ));
        }
        if self.doc_count == 0 {
            return Ok(None);
        }
        self.flush_occurrence_buffer()?;
        let final_run = self.collapse_runs()?;

        let doc_table_offset = MAGIC_V3.len() as u64;
        let doc_table_len = u64::from(self.doc_count)
            .checked_mul(DOC_RECORD_LEN)
            .ok_or_else(|| Error::invariant("text v3 document table length overflows u64"))?;
        let expected_position = doc_table_offset
            .checked_add(doc_table_len)
            .ok_or_else(|| Error::invariant("text v3 document table end overflows u64"))?;
        if self.output.stream_position()? != expected_position {
            return Err(Error::invariant(
                "text external builder document table position drifted",
            ));
        }

        let mut assembler = OutputAssembler::new(&mut self.output, &self.spool)?;
        if let Some(run) = final_run {
            assembler.consume(run, self.options.memory_budget_bytes)?;
        }
        let term_count = assembler.term_count;
        let assembled = assembler.finish()?;
        let directory = assembled.directory;
        self.metrics.spool_bytes_written = self
            .metrics
            .spool_bytes_written
            .checked_add(assembled.term_spool_bytes)
            .and_then(|bytes| bytes.checked_add(directory.bytes_written))
            .ok_or_else(|| Error::invariant("text spool byte counter overflows u64"))?;

        let footer_offset = self.output.stream_position()?;
        let serializable_directory = directory.into_serializable()?;
        let footer = StreamingFooter {
            format_version: FORMAT_VERSION,
            n_docs: self.doc_count,
            total_len: self.total_len,
            doc_table_offset,
            doc_table_len,
            term_count,
            min_node_id: self.min_node_id.expect("non-empty builder has min id"),
            max_node_id: self.max_node_id.expect("non-empty builder has max id"),
            dictionary: &serializable_directory,
        };
        let mut footer_writer = CrcCountingWriter::new(&mut self.output);
        bincode::serialize_into(&mut footer_writer, &footer)
            .map_err(|error| Error::invariant(format!("text v3 footer encode failed: {error}")))?;
        footer_writer.flush()?;
        let (footer_len, footer_crc) = footer_writer.finish();
        if footer_len == 0 || footer_len > MAX_FOOTER_BYTES {
            return Err(Error::invariant(format!(
                "text v3 footer length {footer_len} exceeds the format limit"
            )));
        }
        if self.output.stream_position()? != footer_offset + footer_len {
            return Err(Error::invariant(
                "text external builder footer position drifted",
            ));
        }
        self.output.write_all(&footer_len.to_le_bytes())?;
        self.output.write_all(TRAILER_MAGIC)?;
        self.output.write_all(&footer_crc.to_le_bytes())?;
        self.output.flush()?;
        self.output.sync_data()?;
        let len = self.output.stream_position()?;
        if self.output.metadata()?.len() != len {
            return Err(Error::invariant(
                "text external artifact length changed before handoff",
            ));
        }
        self.output.rewind()?;

        let stats = TextIndexBuildStats {
            doc_count: self.doc_count as u64,
            term_count,
            total_len: self.total_len,
            min_node_id: self.min_node_id.expect("non-empty builder has min id"),
            max_node_id: self.max_node_id.expect("non-empty builder has max id"),
        };
        Ok(Some(TextIndexFileArtifact {
            file: self.output,
            len,
            stats,
            metrics: self.metrics,
        }))
    }

    fn push_inner(&mut self, (id, text): ([u8; 16], String)) -> Result<()> {
        if self.last_node_id.is_some_and(|last| id <= last) {
            return Err(Error::precondition(
                "text external builder requires strictly ascending NodeIds",
            ));
        }
        if self.doc_count == u32::MAX {
            return Err(Error::invariant("text v3 document count exceeds u32"));
        }

        let token_count = count_tokens(&text)?;
        let doc_len = u32::try_from(token_count)
            .map_err(|_| Error::invariant("text v3 document token count exceeds u32"))?;
        let new_total_len = self
            .total_len
            .checked_add(doc_len as u64)
            .ok_or_else(|| Error::invariant("text v3 total token count overflows u64"))?;
        let doc = self.doc_count;

        self.output.write_all(&id)?;
        self.output.write_all(&doc_len.to_le_bytes())?;

        let mut emitted = 0u64;
        for_each_token(&text, |term| {
            let position = u32::try_from(emitted)
                .map_err(|_| Error::invariant("text v3 token position exceeds u32"))?;
            self.push_occurrence(Occurrence {
                term,
                doc,
                position,
                doc_len,
            })?;
            emitted += 1;
            Ok(())
        })?;
        if emitted != token_count {
            return Err(Error::invariant(
                "streaming text tokenizer produced inconsistent token counts",
            ));
        }

        self.doc_count += 1;
        self.total_len = new_total_len;
        self.min_node_id.get_or_insert(id);
        self.max_node_id = Some(id);
        self.last_node_id = Some(id);
        Ok(())
    }

    fn push_occurrence(&mut self, occurrence: Occurrence) -> Result<()> {
        let term_bytes = occurrence.term.capacity();
        let minimum_bytes = size_of::<Occurrence>()
            .checked_add(term_bytes)
            .ok_or_else(|| Error::invariant("text occurrence size accounting overflows"))?;
        if minimum_bytes > self.options.memory_budget_bytes {
            return Err(Error::precondition(format!(
                "one text occurrence requires {minimum_bytes} logical bytes, exceeding \
                 {INDEX_BUILD_MEMORY_ENV}={}",
                self.options.memory_budget_bytes
            )));
        }

        let required_capacity = if self.occurrences.len() == self.occurrences.capacity() {
            self.occurrences.len().saturating_add(1)
        } else {
            self.occurrences.capacity()
        };
        let projected = required_capacity
            .checked_mul(size_of::<Occurrence>())
            .and_then(|bytes| bytes.checked_add(self.occurrence_term_bytes))
            .and_then(|bytes| bytes.checked_add(term_bytes))
            .ok_or_else(|| Error::invariant("text occurrence buffer accounting overflows"))?;
        if projected > self.options.memory_budget_bytes {
            self.flush_occurrence_buffer()?;
        }

        if self.occurrences.len() == self.occurrences.capacity() {
            self.occurrences.reserve_exact(1);
        }
        self.occurrence_term_bytes = self
            .occurrence_term_bytes
            .checked_add(term_bytes)
            .ok_or_else(|| Error::invariant("text occurrence buffer accounting overflows"))?;
        self.occurrences.push(occurrence);
        self.occurrence_buffer_bytes = self
            .occurrences
            .capacity()
            .checked_mul(size_of::<Occurrence>())
            .and_then(|bytes| bytes.checked_add(self.occurrence_term_bytes))
            .ok_or_else(|| Error::invariant("text occurrence buffer accounting overflows"))?;
        if self.occurrence_buffer_bytes > self.options.memory_budget_bytes {
            return Err(Error::invariant(
                "text occurrence Vec capacity exceeded its configured logical memory budget",
            ));
        }
        self.metrics.max_buffer_bytes = self
            .metrics
            .max_buffer_bytes
            .max(self.occurrence_buffer_bytes);
        Ok(())
    }

    fn flush_occurrence_buffer(&mut self) -> Result<()> {
        if self.occurrences.is_empty() {
            return Ok(());
        }
        self.occurrences.sort_unstable();
        if self
            .occurrences
            .windows(2)
            .any(|pair| pair[0].same_key(&pair[1]))
        {
            return Err(Error::invariant(
                "text tokenizer emitted a duplicate term/doc/position occurrence",
            ));
        }
        let records = std::mem::take(&mut self.occurrences);
        self.occurrence_term_bytes = 0;
        self.occurrence_buffer_bytes = 0;
        let mut writer = RunWriter::new(&self.spool)?;
        for occurrence in &records {
            writer.write_occurrence(occurrence)?;
        }
        drop(records);
        let run = writer.finish()?;
        self.metrics.initial_run_count += 1;
        self.add_spool_bytes(run.len)?;
        self.insert_run(0, run)?;
        Ok(())
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
            run = merge_runs(existing, run, &self.spool, self.options.memory_budget_bytes)?;
            self.metrics.run_merge_count += 1;
            self.add_spool_bytes(run.len)?;
            level += 1;
        }
    }

    fn collapse_runs(&mut self) -> Result<Option<RunFile>> {
        let mut merged: Option<RunFile> = None;
        let levels = std::mem::take(&mut self.levels);
        for run in levels.into_iter().flatten() {
            merged = Some(match merged {
                None => run,
                Some(previous) => {
                    let run =
                        merge_runs(previous, run, &self.spool, self.options.memory_budget_bytes)?;
                    self.metrics.run_merge_count += 1;
                    self.add_spool_bytes(run.len)?;
                    run
                }
            });
        }
        Ok(merged)
    }

    fn add_spool_bytes(&mut self, bytes: u64) -> Result<()> {
        self.metrics.spool_bytes_written = self
            .metrics
            .spool_bytes_written
            .checked_add(bytes)
            .ok_or_else(|| Error::invariant("text spool byte counter overflows u64"))?;
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Occurrence {
    term: String,
    doc: u32,
    position: u32,
    doc_len: u32,
}

impl Occurrence {
    fn same_key(&self, other: &Self) -> bool {
        self.term == other.term && self.doc == other.doc && self.position == other.position
    }
}

impl Ord for Occurrence {
    fn cmp(&self, other: &Self) -> Ordering {
        self.term
            .cmp(&other.term)
            .then_with(|| self.doc.cmp(&other.doc))
            .then_with(|| self.position.cmp(&other.position))
            .then_with(|| self.doc_len.cmp(&other.doc_len))
    }
}

impl PartialOrd for Occurrence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct SpoolFactory {
    explicit_directory: Option<PathBuf>,
}

impl SpoolFactory {
    fn new(explicit_directory: Option<PathBuf>) -> Self {
        Self { explicit_directory }
    }

    fn create(&self) -> Result<File> {
        if let Some(directory) = self.explicit_directory.as_deref() {
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
            "text compaction spool directory {} is not accessible: {error}",
            directory.display()
        ))
    })?;
    if !metadata.is_dir() {
        return Err(Error::precondition(format!(
            "text compaction spool path {} is not a directory",
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
        file.write_all(&[0u8; RUN_HEADER_LEN as usize])?;
        Ok(Self {
            writer: BufWriter::with_capacity(IO_BUFFER_BYTES, file),
            count: 0,
            payload_len: 0,
            payload_crc: crc32fast::Hasher::new(),
        })
    }

    fn write_occurrence(&mut self, occurrence: &Occurrence) -> Result<()> {
        let term_len = u32::try_from(occurrence.term.len())
            .map_err(|_| Error::invariant("text occurrence term exceeds u32 bytes"))?;
        self.write_payload(&term_len.to_le_bytes())?;
        self.write_payload(occurrence.term.as_bytes())?;
        self.write_payload(&occurrence.doc.to_le_bytes())?;
        self.write_payload(&occurrence.position.to_le_bytes())?;
        self.write_payload(&occurrence.doc_len.to_le_bytes())?;
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text occurrence run count overflows u64"))?;
        Ok(())
    }

    fn write_payload(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.payload_crc.update(bytes);
        self.payload_len = self
            .payload_len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::invariant("text occurrence run length overflows u64"))?;
        Ok(())
    }

    fn finish(mut self) -> Result<RunFile> {
        self.writer.flush()?;
        let mut file = self
            .writer
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        let payload_crc = self.payload_crc.finalize();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(RUN_MAGIC)?;
        file.write_all(&self.count.to_le_bytes())?;
        file.write_all(&self.payload_len.to_le_bytes())?;
        file.write_all(&payload_crc.to_le_bytes())?;
        let len = RUN_HEADER_LEN
            .checked_add(self.payload_len)
            .ok_or_else(|| Error::invariant("text occurrence run file length overflows u64"))?;
        if file.metadata()?.len() != len {
            return Err(Error::invariant(
                "text occurrence run length does not match its header",
            ));
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
    max_record_bytes: usize,
    verified: bool,
}

impl RunReader {
    fn open(mut run: RunFile, max_record_bytes: usize) -> Result<Self> {
        run.file.rewind()?;
        let mut magic = [0u8; 8];
        run.file.read_exact(&mut magic)?;
        if &magic != RUN_MAGIC {
            return Err(Error::invariant("text occurrence run magic mismatch"));
        }
        let expected_count = read_u64_io(&mut run.file)?;
        let expected_payload_len = read_u64_io(&mut run.file)?;
        let expected_crc = read_u32_io(&mut run.file)?;
        let expected_len = RUN_HEADER_LEN
            .checked_add(expected_payload_len)
            .ok_or_else(|| Error::invariant("text occurrence run length overflows u64"))?;
        if run.len != expected_len || run.file.metadata()?.len() != expected_len {
            return Err(Error::invariant(
                "text occurrence run payload length is inconsistent",
            ));
        }
        Ok(Self {
            reader: BufReader::with_capacity(IO_BUFFER_BYTES, run.file),
            expected_count,
            expected_payload_len,
            expected_crc,
            read_count: 0,
            read_payload_len: 0,
            payload_crc: crc32fast::Hasher::new(),
            max_record_bytes,
            verified: false,
        })
    }

    fn next_occurrence(&mut self) -> Result<Option<Occurrence>> {
        if self.read_count == self.expected_count {
            self.verify_end()?;
            return Ok(None);
        }
        let term_len = self.read_payload_u32()? as usize;
        let record_bytes = size_of::<Occurrence>().saturating_add(term_len);
        if record_bytes > self.max_record_bytes {
            return Err(Error::invariant(format!(
                "text occurrence run record requires {record_bytes} bytes, exceeding its build limit"
            )));
        }
        let remaining_fixed = 12u64;
        if term_len as u64 > self.remaining_payload().saturating_sub(remaining_fixed) {
            return Err(Error::invariant(
                "text occurrence run term length exceeds remaining payload",
            ));
        }
        let mut term = vec![0u8; term_len];
        self.read_payload_exact(&mut term)?;
        let term = String::from_utf8(term)
            .map_err(|_| Error::invariant("text occurrence run term is not UTF-8"))?;
        let doc = self.read_payload_u32()?;
        let position = self.read_payload_u32()?;
        let doc_len = self.read_payload_u32()?;
        if position >= doc_len {
            return Err(Error::invariant(
                "text occurrence run position exceeds document length",
            ));
        }
        self.read_count += 1;
        Ok(Some(Occurrence {
            term,
            doc,
            position,
            doc_len,
        }))
    }

    fn read_payload_u32(&mut self) -> Result<u32> {
        let mut bytes = [0u8; 4];
        self.read_payload_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_payload_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() as u64 > self.remaining_payload() {
            return Err(Error::invariant("text occurrence run record is truncated"));
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
            return Err(Error::invariant(
                "text occurrence run has trailing or missing payload bytes",
            ));
        }
        let actual = std::mem::replace(&mut self.payload_crc, crc32fast::Hasher::new()).finalize();
        if actual != self.expected_crc {
            return Err(Error::invariant("text occurrence run checksum mismatch"));
        }
        self.verified = true;
        Ok(())
    }
}

fn merge_runs(
    left: RunFile,
    right: RunFile,
    spool: &SpoolFactory,
    max_record_bytes: usize,
) -> Result<RunFile> {
    let mut left = RunReader::open(left, max_record_bytes)?;
    let mut right = RunReader::open(right, max_record_bytes)?;
    let mut writer = RunWriter::new(spool)?;
    let mut a = left.next_occurrence()?;
    let mut b = right.next_occurrence()?;
    let mut previous: Option<Occurrence> = None;
    while a.is_some() || b.is_some() {
        let next = match (&a, &b) {
            (Some(a_value), Some(b_value)) => {
                if a_value.same_key(b_value) {
                    return Err(Error::invariant(
                        "duplicate text occurrence encountered while merging runs",
                    ));
                }
                if a_value <= b_value {
                    let value = a.take().expect("left occurrence present");
                    a = left.next_occurrence()?;
                    value
                } else {
                    let value = b.take().expect("right occurrence present");
                    b = right.next_occurrence()?;
                    value
                }
            }
            (Some(_), None) => {
                let value = a.take().expect("left occurrence present");
                a = left.next_occurrence()?;
                value
            }
            (None, Some(_)) => {
                let value = b.take().expect("right occurrence present");
                b = right.next_occurrence()?;
                value
            }
            (None, None) => break,
        };
        if previous.as_ref().is_some_and(|prior| prior >= &next) {
            return Err(Error::invariant(
                "text occurrence merge produced non-increasing records",
            ));
        }
        writer.write_occurrence(&next)?;
        previous = Some(next);
    }
    writer.finish()
}

struct OutputAssembler<'a> {
    output: &'a mut File,
    directory: DirectorySpool,
    terms: TermSpool,
    current_term: Option<String>,
    current_doc: Option<u32>,
    current_doc_len: u32,
    current_positions: ReusableSpool,
    current_position_count: u32,
    current_last_position: Option<u32>,
    posting_block: ReusableSpool,
    posting_block_count: u32,
    posting_block_first_doc: Option<u32>,
    posting_block_last_doc: Option<u32>,
    posting_block_max_tf: u32,
    posting_block_min_doc_len: u32,
    term_blocks: ReusableSpool,
    term_block_count: u64,
    term_doc_freq: u32,
    term_count: u64,
}

impl<'a> OutputAssembler<'a> {
    fn new(output: &'a mut File, spool: &SpoolFactory) -> Result<Self> {
        Ok(Self {
            output,
            directory: DirectorySpool::new(spool)?,
            terms: TermSpool::new(spool)?,
            current_term: None,
            current_doc: None,
            current_doc_len: 0,
            current_positions: ReusableSpool::new(spool)?,
            current_position_count: 0,
            current_last_position: None,
            posting_block: ReusableSpool::new(spool)?,
            posting_block_count: 0,
            posting_block_first_doc: None,
            posting_block_last_doc: None,
            posting_block_max_tf: 0,
            posting_block_min_doc_len: u32::MAX,
            term_blocks: ReusableSpool::new(spool)?,
            term_block_count: 0,
            term_doc_freq: 0,
            term_count: 0,
        })
    }

    fn consume(&mut self, run: RunFile, max_record_bytes: usize) -> Result<()> {
        let mut reader = RunReader::open(run, max_record_bytes)?;
        let mut previous: Option<Occurrence> = None;
        while let Some(occurrence) = reader.next_occurrence()? {
            if previous.as_ref().is_some_and(|prior| prior >= &occurrence) {
                return Err(Error::invariant(
                    "final text occurrence run is not strictly sorted",
                ));
            }
            self.accept(&occurrence)?;
            previous = Some(occurrence);
        }
        self.finish_term()?;
        Ok(())
    }

    fn accept(&mut self, occurrence: &Occurrence) -> Result<()> {
        if self
            .current_term
            .as_ref()
            .is_some_and(|term| term != &occurrence.term)
        {
            self.finish_term()?;
        }
        if self.current_term.is_none() {
            self.current_term = Some(occurrence.term.clone());
        }

        if self.current_doc.is_some_and(|doc| doc != occurrence.doc) {
            self.finish_posting()?;
        }
        match self.current_doc {
            None => {
                self.current_doc = Some(occurrence.doc);
                self.current_doc_len = occurrence.doc_len;
            }
            Some(doc) => {
                if doc != occurrence.doc || self.current_doc_len != occurrence.doc_len {
                    return Err(Error::invariant(
                        "text occurrence document metadata is inconsistent",
                    ));
                }
            }
        }
        if self
            .current_last_position
            .is_some_and(|last| last >= occurrence.position)
        {
            return Err(Error::invariant(
                "text occurrence positions are not strictly increasing",
            ));
        }
        let delta = match self.current_last_position {
            Some(previous) => occurrence.position - previous,
            None => occurrence.position,
        };
        self.current_positions.write_varint(delta as u64)?;
        self.current_position_count = self
            .current_position_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text posting term frequency exceeds u32"))?;
        self.current_last_position = Some(occurrence.position);
        Ok(())
    }

    fn finish_posting(&mut self) -> Result<()> {
        let Some(doc) = self.current_doc.take() else {
            return Ok(());
        };
        let tf = std::mem::take(&mut self.current_position_count);
        if tf == 0 {
            return Err(Error::invariant("text posting has no positions"));
        }
        let document_delta = match self.posting_block_last_doc {
            Some(previous) => doc
                .checked_sub(previous)
                .filter(|delta| *delta > 0)
                .ok_or_else(|| Error::invariant("text v3 builder received unsorted postings"))?,
            None => doc,
        };
        self.posting_block.write_varint(document_delta as u64)?;
        self.posting_block.write_varint(tf as u64)?;
        self.posting_block
            .write_varint(self.current_doc_len as u64)?;
        self.posting_block.write_varint(tf as u64)?;
        self.posting_block
            .append_spool(&mut self.current_positions)?;
        self.current_last_position = None;

        self.posting_block_count = self
            .posting_block_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text posting block count exceeds u32"))?;
        self.posting_block_first_doc.get_or_insert(doc);
        self.posting_block_last_doc = Some(doc);
        self.posting_block_max_tf = self.posting_block_max_tf.max(tf);
        self.posting_block_min_doc_len = self.posting_block_min_doc_len.min(self.current_doc_len);
        self.term_doc_freq = self
            .term_doc_freq
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text term document frequency exceeds u32"))?;
        if self.posting_block_count as usize == POSTINGS_PER_BLOCK {
            self.flush_posting_block()?;
        }
        Ok(())
    }

    fn flush_posting_block(&mut self) -> Result<()> {
        if self.posting_block_count == 0 {
            return Ok(());
        }
        let mut prefix = Vec::with_capacity(10);
        put_varint_local(self.posting_block_count as u64, &mut prefix);
        let wire = append_compressed_spooled_block_file(
            self.output,
            &prefix,
            &mut self.posting_block,
            "posting",
        )?;
        let block = PostingBlockRef {
            wire,
            first_doc: self
                .posting_block_first_doc
                .take()
                .expect("non-empty posting block has first document"),
            last_doc: self
                .posting_block_last_doc
                .take()
                .expect("non-empty posting block has last document"),
            max_tf: std::mem::take(&mut self.posting_block_max_tf),
            min_doc_len: std::mem::replace(&mut self.posting_block_min_doc_len, u32::MAX),
        };
        let encoded = bincode::serialize(&block).map_err(|error| {
            Error::invariant(format!(
                "text posting-block reference encode failed: {error}"
            ))
        })?;
        self.term_blocks.write_all(&encoded)?;
        self.term_block_count = self
            .term_block_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text term posting-block count overflows u64"))?;
        self.posting_block_count = 0;
        Ok(())
    }

    fn finish_term(&mut self) -> Result<()> {
        let Some(term) = self.current_term.take() else {
            return Ok(());
        };
        self.finish_posting()?;
        self.flush_posting_block()?;
        let block_count = std::mem::take(&mut self.term_block_count);
        if self.term_doc_freq == 0 || block_count == 0 || self.term_blocks.is_empty() {
            return Err(Error::invariant(
                "text external builder produced an empty term",
            ));
        }
        self.terms.push_spooled(
            &term,
            std::mem::take(&mut self.term_doc_freq),
            block_count,
            &mut self.term_blocks,
        )?;
        self.term_count = self
            .term_count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text term count overflows u64"))?;
        Ok(())
    }

    fn finish(mut self) -> Result<OutputAssembly> {
        if self.current_term.is_some()
            || self.current_doc.is_some()
            || !self.current_positions.is_empty()
            || self.current_position_count != 0
            || self.current_last_position.is_some()
            || !self.posting_block.is_empty()
            || self.posting_block_count != 0
            || self.posting_block_first_doc.is_some()
            || self.posting_block_last_doc.is_some()
            || !self.term_blocks.is_empty()
            || self.term_block_count != 0
        {
            return Err(Error::invariant(
                "text output assembler retained unfinished state",
            ));
        }
        let term_spool_bytes = self
            .terms
            .emit_dictionary_blocks(self.output, &mut self.directory)?;
        Ok(OutputAssembly {
            directory: self.directory,
            term_spool_bytes,
        })
    }
}

struct OutputAssembly {
    directory: DirectorySpool,
    term_spool_bytes: u64,
}

/// Reusable anonymous file with a small fixed userspace buffer. Posting
/// positions and encoded posting payloads can be arbitrarily large on disk
/// without ever becoming a corpus- or document-sized allocation.
struct ReusableSpool {
    writer: BufWriter<File>,
    len: u64,
}

impl ReusableSpool {
    fn new(spool: &SpoolFactory) -> Result<Self> {
        Ok(Self {
            writer: BufWriter::with_capacity(IO_BUFFER_BYTES, spool.create()?),
            len: 0,
        })
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.len = self
            .len
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::invariant("text posting spool length overflows u64"))?;
        if self.len > MAX_RAW_BLOCK_BYTES {
            return Err(Error::precondition(format!(
                "one text posting block requires more than {MAX_RAW_BLOCK_BYTES} encoded bytes"
            )));
        }
        Ok(())
    }

    fn write_varint(&mut self, value: u64) -> Result<()> {
        let mut bytes = Vec::with_capacity(10);
        put_varint_local(value, &mut bytes);
        self.write_all(&bytes)
    }

    fn append_spool(&mut self, source: &mut Self) -> Result<()> {
        let source_len = source.len;
        let projected = self
            .len
            .checked_add(source_len)
            .ok_or_else(|| Error::invariant("text posting spool length overflows u64"))?;
        if projected > MAX_RAW_BLOCK_BYTES {
            return Err(Error::precondition(format!(
                "one text posting block requires {projected} encoded bytes, exceeding the \
                 wire-format limit {MAX_RAW_BLOCK_BYTES}"
            )));
        }
        source.drain_into(&mut self.writer)?;
        self.len = projected;
        Ok(())
    }

    fn drain_into(&mut self, destination: &mut impl Write) -> Result<()> {
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.rewind()?;
        let copied = std::io::copy(file, destination)?;
        if copied != self.len {
            return Err(Error::invariant(
                "text posting spool length changed while streaming",
            ));
        }
        file.rewind()?;
        file.set_len(0)?;
        self.len = 0;
        Ok(())
    }
}

fn put_varint_local(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn append_compressed_spooled_block_file(
    file: &mut File,
    prefix: &[u8],
    raw: &mut ReusableSpool,
    kind: &str,
) -> Result<BlockRef> {
    let raw_len = (prefix.len() as u64)
        .checked_add(raw.len)
        .ok_or_else(|| Error::invariant(format!("text v3 {kind} raw length overflows u64")))?;
    if raw_len > MAX_RAW_BLOCK_BYTES || raw_len > u32::MAX as u64 {
        return Err(Error::precondition(format!(
            "text v3 {kind} block requires {raw_len} raw bytes, exceeding the wire-format limit"
        )));
    }

    let offset = file.stream_position()?;
    let counting = CrcCountingWriter::new(&mut *file);
    let mut encoder = zstd::stream::write::Encoder::new(counting, ZSTD_LEVEL)
        .map_err(|error| Error::invariant(format!("text v3 {kind} compression failed: {error}")))?;
    encoder
        .write_all(prefix)
        .map_err(|error| Error::invariant(format!("text v3 {kind} compression failed: {error}")))?;
    raw.drain_into(&mut encoder)?;
    let counting = encoder
        .finish()
        .map_err(|error| Error::invariant(format!("text v3 {kind} compression failed: {error}")))?;
    let (compressed_len, crc32) = counting.finish();
    if compressed_len > MAX_BLOCK_BYTES || compressed_len > u32::MAX as u64 {
        return Err(Error::precondition(format!(
            "text v3 {kind} block requires {compressed_len} wire bytes, exceeding the \
             wire-format limit {MAX_BLOCK_BYTES}"
        )));
    }
    Ok(BlockRef {
        offset,
        len: compressed_len as u32,
        raw_len: raw_len as u32,
        crc32,
    })
}

/// Compress a bincode `Vec<TermEntry>` directly from exact serialized term
/// records. The eight-byte vector length prefix is the only framing bincode
/// adds around the concatenated entries, so neither a common term's block
/// references nor a 128-term dictionary block need to exist in memory.
fn append_compressed_term_records_file(
    output: &mut File,
    spool: &mut File,
    records: &[TermRecordRef],
) -> Result<BlockRef> {
    if records.is_empty() || records.len() > TERMS_PER_DICTIONARY_BLOCK {
        return Err(Error::invariant(
            "text dictionary spool chunk cardinality is invalid",
        ));
    }
    let raw_len = records.iter().try_fold(8u64, |total, record| {
        total
            .checked_add(record.len)
            .ok_or_else(|| Error::invariant("text dictionary raw length overflows u64"))
    })?;
    if raw_len > MAX_RAW_BLOCK_BYTES || raw_len > u32::MAX as u64 {
        return Err(Error::precondition(format!(
            "text v3 dictionary block requires {raw_len} raw bytes, exceeding the wire-format \
             limit {MAX_RAW_BLOCK_BYTES}"
        )));
    }

    let sequence_len = u64::try_from(records.len())
        .map_err(|_| Error::invariant("text dictionary term count exceeds u64"))?;
    let sequence_prefix = bincode::serialize(&sequence_len).map_err(|error| {
        Error::invariant(format!(
            "text dictionary sequence prefix encode failed: {error}"
        ))
    })?;
    if sequence_prefix.len() != 8 {
        return Err(Error::invariant(
            "text dictionary bincode sequence prefix is not eight bytes",
        ));
    }

    let offset = output.stream_position()?;
    let counting = CrcCountingWriter::new(&mut *output);
    let mut encoder = zstd::stream::write::Encoder::new(counting, ZSTD_LEVEL).map_err(|error| {
        Error::invariant(format!("text v3 dictionary compression failed: {error}"))
    })?;
    encoder.write_all(&sequence_prefix).map_err(|error| {
        Error::invariant(format!("text v3 dictionary compression failed: {error}"))
    })?;

    let mut buffer = [0u8; IO_BUFFER_BYTES];
    for record in records {
        spool.seek(SeekFrom::Start(record.payload_offset))?;
        let mut remaining = record.len;
        let mut crc = crc32fast::Hasher::new();
        while remaining > 0 {
            let requested = usize::try_from(remaining.min(buffer.len() as u64))
                .expect("bounded by the fixed buffer length");
            spool.read_exact(&mut buffer[..requested])?;
            crc.update(&buffer[..requested]);
            encoder.write_all(&buffer[..requested]).map_err(|error| {
                Error::invariant(format!("text v3 dictionary compression failed: {error}"))
            })?;
            remaining -= requested as u64;
        }
        if crc.finalize() != record.crc32 {
            return Err(Error::invariant("text term spool checksum mismatch"));
        }
    }

    let counting = encoder.finish().map_err(|error| {
        Error::invariant(format!("text v3 dictionary compression failed: {error}"))
    })?;
    let (compressed_len, crc32) = counting.finish();
    if compressed_len > MAX_BLOCK_BYTES || compressed_len > u32::MAX as u64 {
        return Err(Error::precondition(format!(
            "text v3 dictionary block requires {compressed_len} wire bytes, exceeding the \
             wire-format limit {MAX_BLOCK_BYTES}"
        )));
    }
    Ok(BlockRef {
        offset,
        len: compressed_len as u32,
        raw_len: raw_len as u32,
        crc32,
    })
}

struct TermSpool {
    writer: BufWriter<File>,
    count: u64,
    bytes_written: u64,
}

impl TermSpool {
    fn new(spool: &SpoolFactory) -> Result<Self> {
        Ok(Self {
            writer: BufWriter::with_capacity(IO_BUFFER_BYTES, spool.create()?),
            count: 0,
            bytes_written: 0,
        })
    }

    fn push_spooled(
        &mut self,
        term: &str,
        doc_freq: u32,
        block_count: u64,
        blocks: &mut ReusableSpool,
    ) -> Result<()> {
        if block_count == 0 || blocks.is_empty() {
            return Err(Error::invariant(
                "text term spool received an empty posting-block list",
            ));
        }

        // Bincode encodes a struct as its fields in declaration order and a
        // Vec as a u64 element count followed by the serialized elements.
        // Writing the small prefix and then copying the already serialized
        // PostingBlockRefs therefore produces the exact TermEntry wire bytes
        // without materialising a potentially multi-million-document Vec.
        let mut prefix = Vec::with_capacity(term.len().saturating_add(24));
        bincode::serialize_into(&mut prefix, term)
            .and_then(|_| bincode::serialize_into(&mut prefix, &doc_freq))
            .and_then(|_| bincode::serialize_into(&mut prefix, &block_count))
            .map_err(|error| Error::invariant(format!("text term spool encode failed: {error}")))?;
        let payload_len = (prefix.len() as u64)
            .checked_add(blocks.len)
            .ok_or_else(|| Error::invariant("text term spool record length overflows u64"))?;
        if payload_len > MAX_RAW_BLOCK_BYTES || payload_len > u32::MAX as u64 {
            return Err(Error::precondition(format!(
                "one text term directory entry requires {payload_len} bytes, exceeding the \
                 wire-format limit"
            )));
        }

        let record_start = self.writer.stream_position()?;
        self.writer
            .write_all(&[0u8; DIRECTORY_RECORD_HEADER_LEN as usize])?;
        let (written, crc32) = {
            let mut counting = CrcCountingWriter::new(&mut self.writer);
            counting.write_all(&prefix)?;
            blocks.drain_into(&mut counting)?;
            counting.flush()?;
            counting.finish()
        };
        if written != payload_len {
            return Err(Error::invariant(
                "text term spool payload length changed while streaming",
            ));
        }
        let record_end = self.writer.stream_position()?;
        self.writer.seek(SeekFrom::Start(record_start))?;
        self.writer.write_all(&(payload_len as u32).to_le_bytes())?;
        self.writer.write_all(&crc32.to_le_bytes())?;
        self.writer.seek(SeekFrom::Start(record_end))?;

        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text term spool count overflows u64"))?;
        self.bytes_written = self
            .bytes_written
            .checked_add(DIRECTORY_RECORD_HEADER_LEN + payload_len)
            .ok_or_else(|| Error::invariant("text term spool length overflows u64"))?;
        Ok(())
    }

    fn emit_dictionary_blocks(
        mut self,
        output: &mut File,
        directory: &mut DirectorySpool,
    ) -> Result<u64> {
        self.writer.flush()?;
        let mut file = self
            .writer
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        if file.metadata()?.len() != self.bytes_written {
            return Err(Error::invariant("text term spool length is inconsistent"));
        }
        file.rewind()?;

        let mut remaining = self.count;
        let mut previous_term: Option<String> = None;
        while remaining > 0 {
            let chunk_len = remaining.min(TERMS_PER_DICTIONARY_BLOCK as u64) as usize;
            let mut records = Vec::with_capacity(chunk_len);
            let mut first_term = None;
            let mut last_term = None;
            for _ in 0..chunk_len {
                let len = read_u32_io(&mut file)? as u64;
                if len > self.bytes_written || len > MAX_RAW_BLOCK_BYTES {
                    return Err(Error::invariant("text term spool record length is invalid"));
                }
                let expected_crc = read_u32_io(&mut file)?;
                let payload_offset = file.stream_position()?;
                let payload_end = payload_offset
                    .checked_add(len)
                    .filter(|end| *end <= self.bytes_written)
                    .ok_or_else(|| Error::invariant("text term spool record range is invalid"))?;
                let term_len = read_u64_io(&mut file)?;
                if term_len > len.saturating_sub(8) || term_len > MAX_RAW_BLOCK_BYTES {
                    return Err(Error::invariant("text term spool term length is invalid"));
                }
                let term_len = usize::try_from(term_len)
                    .map_err(|_| Error::invariant("text term length exceeds usize"))?;
                let mut term_bytes = vec![0u8; term_len];
                file.read_exact(&mut term_bytes)?;
                let term = String::from_utf8(term_bytes)
                    .map_err(|error| Error::invariant(format!("text term is not utf8: {error}")))?;
                if previous_term
                    .as_ref()
                    .is_some_and(|previous| previous >= &term)
                {
                    return Err(Error::invariant("text term spool is not strictly sorted"));
                }
                first_term.get_or_insert_with(|| term.clone());
                last_term = Some(term.clone());
                previous_term = Some(term);
                file.seek(SeekFrom::Start(payload_end))?;
                records.push(TermRecordRef {
                    payload_offset,
                    len,
                    crc32: expected_crc,
                });
            }

            let wire = append_compressed_term_records_file(output, &mut file, &records)?;
            directory.push(&DictionaryBlockRef {
                first_term: first_term.expect("non-empty dictionary block has first term"),
                last_term: last_term.expect("non-empty dictionary block has last term"),
                wire,
            })?;
            remaining -= chunk_len as u64;
        }
        if file.stream_position()? != self.bytes_written {
            return Err(Error::invariant("text term spool has trailing bytes"));
        }
        Ok(self.bytes_written)
    }
}

struct TermRecordRef {
    payload_offset: u64,
    len: u64,
    crc32: u32,
}

struct DirectorySpool {
    writer: BufWriter<File>,
    count: u64,
    bytes_written: u64,
}

impl DirectorySpool {
    fn new(spool: &SpoolFactory) -> Result<Self> {
        Ok(Self {
            writer: BufWriter::with_capacity(IO_BUFFER_BYTES, spool.create()?),
            count: 0,
            bytes_written: 0,
        })
    }

    fn push(&mut self, entry: &DictionaryBlockRef) -> Result<()> {
        let encoded = bincode::serialize(entry).map_err(|error| {
            Error::invariant(format!("text sparse directory encode failed: {error}"))
        })?;
        let len = u32::try_from(encoded.len())
            .map_err(|_| Error::invariant("text sparse directory record exceeds u32"))?;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer
            .write_all(&crc32fast::hash(&encoded).to_le_bytes())?;
        self.writer.write_all(&encoded)?;
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text sparse directory count overflows u64"))?;
        self.bytes_written = self
            .bytes_written
            .checked_add(DIRECTORY_RECORD_HEADER_LEN + encoded.len() as u64)
            .ok_or_else(|| Error::invariant("text sparse directory length overflows u64"))?;
        Ok(())
    }

    fn into_serializable(mut self) -> Result<SerializableDirectory> {
        self.writer.flush()?;
        let mut file = self
            .writer
            .into_inner()
            .map_err(|error| Error::Io(error.into_error()))?;
        if file.metadata()?.len() != self.bytes_written {
            return Err(Error::invariant(
                "text sparse directory spool length is inconsistent",
            ));
        }
        file.rewind()?;
        Ok(SerializableDirectory {
            file: RefCell::new(file),
            count: self.count,
            bytes: self.bytes_written,
        })
    }
}

struct SerializableDirectory {
    file: RefCell<File>,
    count: u64,
    bytes: u64,
}

impl Serialize for SerializableDirectory {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let count = usize::try_from(self.count)
            .map_err(|_| S::Error::custom("text sparse directory count exceeds usize"))?;
        let mut sequence = serializer.serialize_seq(Some(count))?;
        let mut file = self.file.borrow_mut();
        file.rewind().map_err(S::Error::custom)?;
        for _ in 0..count {
            let len = read_u32_io(&mut *file).map_err(S::Error::custom)? as usize;
            if len as u64 > self.bytes || len as u64 > MAX_RAW_BLOCK_BYTES {
                return Err(S::Error::custom(
                    "text sparse directory record length is invalid",
                ));
            }
            let crc = read_u32_io(&mut *file).map_err(S::Error::custom)?;
            let mut encoded = vec![0u8; len];
            file.read_exact(&mut encoded).map_err(S::Error::custom)?;
            if crc32fast::hash(&encoded) != crc {
                return Err(S::Error::custom("text sparse directory checksum mismatch"));
            }
            let entry: DictionaryBlockRef =
                bincode::deserialize(&encoded).map_err(S::Error::custom)?;
            sequence.serialize_element(&entry)?;
        }
        if file.stream_position().map_err(S::Error::custom)? != self.bytes {
            return Err(S::Error::custom("text sparse directory has trailing bytes"));
        }
        sequence.end()
    }
}

#[derive(Serialize)]
struct StreamingFooter<'a> {
    format_version: u16,
    n_docs: u32,
    total_len: u64,
    doc_table_offset: u64,
    doc_table_len: u64,
    term_count: u64,
    min_node_id: [u8; 16],
    max_node_id: [u8; 16],
    dictionary: &'a SerializableDirectory,
}

struct CrcCountingWriter<W> {
    inner: W,
    len: u64,
    crc: crc32fast::Hasher,
}

impl<W> CrcCountingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            len: 0,
            crc: crc32fast::Hasher::new(),
        }
    }

    fn finish(self) -> (u64, u32) {
        (self.len, self.crc.finalize())
    }
}

impl<W: Write> Write for CrcCountingWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.crc.update(&bytes[..written]);
        self.len = self.len.saturating_add(written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn read_u64_io(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32_io(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn count_tokens(text: &str) -> Result<u64> {
    let mut count = 0u64;
    for_each_token(text, |_| {
        count = count
            .checked_add(1)
            .ok_or_else(|| Error::invariant("text token count overflows u64"))?;
        Ok(())
    })?;
    Ok(count)
}

/// Streaming equivalent of `crate::text::tokenize`.
fn for_each_token(mut text: &str, mut emit: impl FnMut(String) -> Result<()>) -> Result<()> {
    while !text.is_empty() {
        let Some((start, first)) = text.char_indices().find(|(_, ch)| ch.is_alphanumeric()) else {
            break;
        };
        text = &text[start..];
        if is_cjk(first) {
            let mut chars = text.char_indices();
            let (_, mut previous) = chars.next().expect("first character exists");
            let mut emitted_bigram = false;
            let mut consumed = previous.len_utf8();
            for (offset, current) in chars {
                if !current.is_alphanumeric() || !is_cjk(current) {
                    consumed = offset;
                    break;
                }
                let mut pair = String::with_capacity(previous.len_utf8() + current.len_utf8());
                pair.push(previous);
                pair.push(current);
                emit(pair.to_lowercase())?;
                emitted_bigram = true;
                previous = current;
                consumed = offset + current.len_utf8();
            }
            if !emitted_bigram {
                emit(previous.to_lowercase().to_string())?;
            }
            text = &text[consumed..];
        } else {
            let mut consumed = 0usize;
            for (offset, current) in text.char_indices() {
                if !current.is_alphanumeric() || is_cjk(current) {
                    consumed = offset;
                    break;
                }
                consumed = offset + current.len_utf8();
            }
            let token = &text[..consumed];
            emit(token.to_lowercase())?;
            text = &text[consumed..];
        }
    }
    Ok(())
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0xAC00..=0xD7AF
        | 0x20000..=0x2A6DF
    )
}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Seek as _, Write as _};

    use super::*;
    use crate::sst::text::{build_body, TextIndex};
    use crate::text::{parse_query, tokenize};

    fn id(value: u16) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[14..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn fixture() -> Vec<([u8; 16], String)> {
        vec![
            (id(1), "Graph database dataflow 東京大学".into()),
            (id(2), "database for graph datasets".into()),
            (id(3), "graph database dataset dataset".into()),
            (id(4), "texto jurídico café CAFÉ".into()),
        ]
    }

    fn build_external(
        docs: Vec<([u8; 16], String)>,
        budget: usize,
        directory: &Path,
    ) -> (Vec<u8>, ExternalTextIndexBuildMetrics) {
        let mut builder = TextIndexExternalBuilder::with_options(
            ExternalTextIndexBuildOptions::for_test(budget, directory),
        )
        .unwrap();
        for doc in docs {
            builder.push(doc).unwrap();
        }
        let artifact = builder.finish().unwrap().unwrap();
        let metrics = artifact.metrics();
        let (mut file, len, _, _) = artifact.into_parts();
        let mut body = Vec::new();
        file.read_to_end(&mut body).unwrap();
        assert_eq!(body.len() as u64, len);
        (body, metrics)
    }

    #[test]
    fn streaming_tokenizer_matches_shared_semantics() {
        for text in [
            "",
            "alpha beta",
            "CAFÉ—ÜBER Привет",
            "東京大学",
            "abc東京大学DEF",
            "한글테스트 mixed_42",
            "a 中 b",
        ] {
            let mut actual = Vec::new();
            for_each_token(text, |token| {
                actual.push(token);
                Ok(())
            })
            .unwrap();
            assert_eq!(actual, tokenize(text), "{text:?}");
        }
    }

    #[test]
    fn external_builder_is_bit_identical_and_query_equivalent() {
        let directory = tempfile::tempdir().unwrap();
        let docs = fixture();
        let expected = build_body(docs.clone()).unwrap().unwrap().0;
        let (actual, _) = build_external(docs, 64 * 1024, directory.path());
        assert_eq!(actual.as_slice(), expected.as_ref());

        let expected = TextIndex::decode(&expected).unwrap();
        let actual = TextIndex::decode(&actual).unwrap();
        for query in ["graph", "\"graph database\" data*", "\"東京大学\"", "café"] {
            let query = parse_query(query);
            assert_eq!(
                actual.search_query(&query, Some(10)),
                expected.search_query(&query, Some(10))
            );
        }
    }

    #[test]
    fn tiny_runs_are_deterministic_and_logically_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let mut docs = fixture();
        for value in 10..80 {
            docs.push((
                id(value),
                format!("term{value:03} common common phrase token"),
            ));
        }
        let (first, metrics) = build_external(docs.clone(), 96, directory.path());
        let (second, second_metrics) = build_external(docs, 96, directory.path());
        assert_eq!(first, second);
        assert_eq!(metrics.max_buffer_bytes, second_metrics.max_buffer_bytes);
        assert!(metrics.initial_run_count > 10, "{metrics:?}");
        assert!(metrics.run_merge_count > 0, "{metrics:?}");
        assert!(metrics.max_buffer_bytes <= 96, "{metrics:?}");
        TextIndex::decode(&first).unwrap();
    }

    #[test]
    fn external_builder_matches_multiple_posting_and_dictionary_blocks() {
        let directory = tempfile::tempdir().unwrap();
        let docs = (0..300u16)
            .map(|value| {
                (
                    id(value),
                    format!("common vocabulary{value:03} phrase token"),
                )
            })
            .collect::<Vec<_>>();
        let expected = build_body(docs.clone()).unwrap().unwrap().0;
        let (actual, metrics) = build_external(docs, 2 * 1024, directory.path());
        assert_eq!(actual.as_slice(), expected.as_ref());
        assert!(metrics.initial_run_count > 1, "{metrics:?}");
    }

    #[test]
    fn high_term_frequency_is_spooled_instead_of_materialised() {
        let directory = tempfile::tempdir().unwrap();
        let repeated = "repeat ".repeat(20_000);
        let docs = vec![(id(1), repeated)];
        let expected = build_body(docs.clone()).unwrap().unwrap().0;
        let (actual, metrics) = build_external(docs, 64 * 1024, directory.path());

        // Streaming the position deltas through anonymous files preserves the
        // exact wire format while the logical occurrence buffer remains below
        // its configured ceiling.
        assert_eq!(actual.as_slice(), expected.as_ref());
        assert!(metrics.max_buffer_bytes <= 64 * 1024, "{metrics:?}");
        let decoded = TextIndex::decode(&actual).unwrap();
        let hits = decoded.search_query(&parse_query("\"repeat repeat\""), Some(10));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id(1));
        assert!(hits[0].1.is_finite());
    }

    #[test]
    fn rejects_unsorted_input_and_oversized_single_occurrence() {
        let directory = tempfile::tempdir().unwrap();
        let mut builder = TextIndexExternalBuilder::with_options(
            ExternalTextIndexBuildOptions::for_test(128, directory.path()),
        )
        .unwrap();
        builder.push((id(2), "alpha".into())).unwrap();
        assert!(builder.push((id(1), "beta".into())).is_err());
        assert!(builder.finish().is_err());

        let mut builder = TextIndexExternalBuilder::with_options(
            ExternalTextIndexBuildOptions::for_test(48, directory.path()),
        )
        .unwrap();
        assert!(builder
            .push((
                id(1),
                "atermthatcannotpossiblyfitinsidefortyeightbytes".into()
            ))
            .is_err());
    }

    #[test]
    fn corrupt_run_is_rejected_and_tempfiles_clean_up() {
        let directory = tempfile::tempdir().unwrap();
        let spool = SpoolFactory::new(Some(directory.path().to_path_buf()));
        let mut writer = RunWriter::new(&spool).unwrap();
        writer
            .write_occurrence(&Occurrence {
                term: "alpha".into(),
                doc: 0,
                position: 0,
                doc_len: 1,
            })
            .unwrap();
        let mut run = writer.finish().unwrap();
        run.file.seek(SeekFrom::Start(RUN_HEADER_LEN + 4)).unwrap();
        run.file.write_all(b"z").unwrap();
        run.file.flush().unwrap();
        let mut reader = RunReader::open(run, 1024).unwrap();
        assert!(reader.next_occurrence().is_ok());
        let error = reader.next_occurrence().unwrap_err().to_string();
        assert!(error.contains("checksum"), "{error}");

        drop(reader);
        assert_eq!(
            std::fs::read_dir(directory.path()).unwrap().count(),
            0,
            "anonymous spool files must be removed on close"
        );
    }
}
