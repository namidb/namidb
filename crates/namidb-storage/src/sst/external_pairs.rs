//! Bounded-memory external sort for secondary-index observations.
//!
//! Node SST input is ordered by `NodeId`, while property indexes require
//! `(property, encoded_value, NodeId)` order. Keeping a `BTreeMap` per
//! property therefore grows with the corpus. This scratch sorter buffers a
//! configurable amount, writes sorted/checksummed runs, and performs bounded
//! fan-in merge passes. Its final iterator owns at most `MERGE_FAN_IN` current
//! records regardless of corpus size.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{Read, Seek, Write};

use crate::error::{Error, Result};
use crate::sst::paged_index::create_spool_file;

const DEFAULT_SORT_MEMORY_BYTES: usize = 8 * 1024 * 1024;
const MIN_SORT_MEMORY_BYTES: usize = 64 * 1024;
const MERGE_FAN_IN: usize = 32;
const RECORD_FIXED_BYTES: usize = 4 + 4 + 16 + 4;
const RECORD_MEMORY_OVERHEAD: usize = 64;

/// One sorted secondary-index observation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ExternalPair {
    pub(crate) property: u32,
    pub(crate) key: Vec<u8>,
    pub(crate) id: [u8; 16],
}

#[derive(Debug)]
struct SortRun {
    file: std::fs::File,
    count: u64,
}

/// Incremental bounded-memory run generator.
#[derive(Debug)]
pub(crate) struct ExternalPairSorter {
    memory_limit: usize,
    buffered_bytes: usize,
    buffered: Vec<ExternalPair>,
    /// Tiered runs prevent both an unbounded file-descriptor set and repeated
    /// rewriting of one ever-growing run. Each level contains fewer than
    /// `MERGE_FAN_IN` files during observation.
    levels: Vec<Vec<SortRun>>,
}

impl ExternalPairSorter {
    pub(crate) fn from_env() -> Result<Self> {
        let memory_limit = match std::env::var("NAMIDB_SIDECAR_SORT_MEMORY_BYTES") {
            Ok(raw) => raw.parse::<usize>().map_err(|error| {
                Error::precondition(format!(
                    "invalid NAMIDB_SIDECAR_SORT_MEMORY_BYTES={raw:?}: {error}"
                ))
            })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_SORT_MEMORY_BYTES,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(Error::precondition(
                    "NAMIDB_SIDECAR_SORT_MEMORY_BYTES is not valid UTF-8",
                ));
            }
        };
        Self::with_memory_limit(memory_limit)
    }

    /// Construct a sorter under a caller-owned aggregate memory budget.
    ///
    /// Search-index builders use this instead of the sidecar-specific
    /// environment variable so `NAMIDB_INDEX_BUILD_MEMORY_BYTES` remains the
    /// single accounting boundary for their page and external-sort workspace.
    pub(crate) fn with_memory_limit(memory_limit: usize) -> Result<Self> {
        if memory_limit < MIN_SORT_MEMORY_BYTES {
            return Err(Error::precondition(format!(
                "external pair sort memory must be at least {MIN_SORT_MEMORY_BYTES} bytes"
            )));
        }
        Ok(Self {
            memory_limit,
            buffered_bytes: 0,
            buffered: Vec::new(),
            levels: Vec::new(),
        })
    }

    pub(crate) fn push(&mut self, property: u32, key: &[u8], id: [u8; 16]) -> Result<()> {
        if key.len() > u16::MAX as usize {
            return Err(Error::invariant(
                "secondary-index key exceeds paged wire limit",
            ));
        }
        let record_bytes = RECORD_MEMORY_OVERHEAD
            .checked_add(key.len())
            .ok_or_else(|| Error::invariant("secondary-index sort record exceeds usize"))?;
        if !self.buffered.is_empty()
            && self.buffered_bytes.saturating_add(record_bytes) > self.memory_limit
        {
            self.flush_buffer()?;
        }
        self.buffered.push(ExternalPair {
            property,
            key: key.to_vec(),
            id,
        });
        self.buffered_bytes = self
            .buffered_bytes
            .checked_add(record_bytes)
            .ok_or_else(|| Error::invariant("secondary-index sort buffer exceeds usize"))?;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<ExternalPairMerge> {
        self.flush_buffer()?;
        let mut runs: Vec<SortRun> = self.levels.into_iter().flatten().collect();
        if runs.is_empty() {
            return ExternalPairMerge::new(Vec::new());
        }
        while runs.len() > MERGE_FAN_IN {
            let mut next = Vec::with_capacity(runs.len().div_ceil(MERGE_FAN_IN));
            let mut pending = runs.into_iter();
            loop {
                let group: Vec<_> = pending.by_ref().take(MERGE_FAN_IN).collect();
                if group.is_empty() {
                    break;
                }
                next.push(merge_runs(group)?);
            }
            runs = next;
        }
        ExternalPairMerge::new(runs)
    }

    fn flush_buffer(&mut self) -> Result<()> {
        if self.buffered.is_empty() {
            return Ok(());
        }
        self.buffered.sort_unstable();
        let count = self.buffered.len() as u64;
        let mut file = create_spool_file()?;
        for record in &self.buffered {
            write_record(&mut file, record)?;
        }
        file.sync_data()?;
        file.rewind()?;
        self.buffered.clear();
        self.buffered_bytes = 0;
        self.insert_run(0, SortRun { file, count })
    }

    fn insert_run(&mut self, level: usize, run: SortRun) -> Result<()> {
        if self.levels.len() <= level {
            self.levels.resize_with(level + 1, Vec::new);
        }
        self.levels[level].push(run);
        if self.levels[level].len() < MERGE_FAN_IN {
            return Ok(());
        }
        let inputs = std::mem::take(&mut self.levels[level]);
        let merged = merge_runs(inputs)?;
        self.insert_run(level + 1, merged)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct HeapRecord {
    pair: ExternalPair,
    run: usize,
}

impl PartialOrd for HeapRecord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapRecord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pair
            .cmp(&other.pair)
            .then_with(|| self.run.cmp(&other.run))
    }
}

#[derive(Debug)]
struct RunReader {
    file: std::fs::File,
    remaining: u64,
}

/// Bounded fan-in merge iterator. Scratch corruption is an error rather than
/// a silently incomplete immutable sidecar.
#[derive(Debug)]
pub(crate) struct ExternalPairMerge {
    runs: Vec<RunReader>,
    heap: BinaryHeap<Reverse<HeapRecord>>,
}

impl ExternalPairMerge {
    fn new(runs: Vec<SortRun>) -> Result<Self> {
        let mut this = Self {
            runs: runs
                .into_iter()
                .map(|run| RunReader {
                    file: run.file,
                    remaining: run.count,
                })
                .collect(),
            heap: BinaryHeap::new(),
        };
        for run in 0..this.runs.len() {
            this.refill(run)?;
        }
        Ok(this)
    }

    pub(crate) fn next_pair(&mut self) -> Result<Option<ExternalPair>> {
        crate::cancel::check()?;
        let Some(Reverse(item)) = self.heap.pop() else {
            return Ok(None);
        };
        self.refill(item.run)?;
        Ok(Some(item.pair))
    }

    fn refill(&mut self, run: usize) -> Result<()> {
        let reader = &mut self.runs[run];
        if reader.remaining == 0 {
            let mut extra = [0_u8; 1];
            if reader.file.read(&mut extra)? != 0 {
                return Err(Error::invariant(
                    "secondary-index scratch run has trailing bytes",
                ));
            }
            return Ok(());
        }
        let pair = read_record(&mut reader.file)?;
        reader.remaining -= 1;
        self.heap.push(Reverse(HeapRecord { pair, run }));
        Ok(())
    }
}

fn merge_runs(runs: Vec<SortRun>) -> Result<SortRun> {
    let count = runs.iter().try_fold(0_u64, |total, run| {
        total
            .checked_add(run.count)
            .ok_or_else(|| Error::invariant("secondary-index run count exceeds u64"))
    })?;
    let mut merge = ExternalPairMerge::new(runs)?;
    let mut file = create_spool_file()?;
    let mut written = 0_u64;
    while let Some(record) = merge.next_pair()? {
        write_record(&mut file, &record)?;
        written += 1;
    }
    if written != count {
        return Err(Error::invariant(
            "secondary-index merge lost scratch records",
        ));
    }
    file.sync_data()?;
    file.rewind()?;
    Ok(SortRun { file, count })
}

fn write_record(file: &mut std::fs::File, record: &ExternalPair) -> Result<()> {
    let key_len = u32::try_from(record.key.len())
        .map_err(|_| Error::invariant("secondary-index scratch key exceeds u32"))?;
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&record.property.to_le_bytes());
    checksum.update(&record.id);
    checksum.update(&record.key);
    file.write_all(&record.property.to_le_bytes())?;
    file.write_all(&key_len.to_le_bytes())?;
    file.write_all(&record.id)?;
    file.write_all(&checksum.finalize().to_le_bytes())?;
    file.write_all(&record.key)?;
    Ok(())
}

fn read_record(file: &mut std::fs::File) -> Result<ExternalPair> {
    let mut header = [0_u8; RECORD_FIXED_BYTES];
    file.read_exact(&mut header)?;
    let property = u32::from_le_bytes(
        header[..4]
            .try_into()
            .map_err(|_| Error::invariant("invalid secondary-index property ordinal"))?,
    );
    let key_len = u32::from_le_bytes(
        header[4..8]
            .try_into()
            .map_err(|_| Error::invariant("invalid secondary-index key length"))?,
    ) as usize;
    if key_len > u16::MAX as usize {
        return Err(Error::invariant(
            "secondary-index scratch key exceeds paged wire limit",
        ));
    }
    let id: [u8; 16] = header[8..24]
        .try_into()
        .map_err(|_| Error::invariant("invalid secondary-index node id"))?;
    let expected_checksum = u32::from_le_bytes(
        header[24..28]
            .try_into()
            .map_err(|_| Error::invariant("invalid secondary-index scratch checksum"))?,
    );
    let mut key = vec![0_u8; key_len];
    file.read_exact(&mut key)?;
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&property.to_le_bytes());
    checksum.update(&id);
    checksum.update(&key);
    if checksum.finalize() != expected_checksum {
        return Err(Error::invariant(
            "secondary-index scratch checksum mismatch",
        ));
    }
    Ok(ExternalPair { property, key, id })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_pairs_sort_across_bounded_runs() {
        let mut sorter = ExternalPairSorter {
            memory_limit: MIN_SORT_MEMORY_BYTES,
            buffered_bytes: 0,
            buffered: Vec::new(),
            levels: Vec::new(),
        };
        for value in (0_u32..10_000).rev() {
            let property = value % 3;
            sorter
                .push(
                    property,
                    format!("{value:08}").as_bytes(),
                    (value as u128).to_be_bytes(),
                )
                .unwrap();
        }
        let mut merge = sorter.finish().unwrap();
        let mut previous = None;
        let mut count = 0;
        while let Some(pair) = merge.next_pair().unwrap() {
            assert!(previous.as_ref().is_none_or(|item| item < &pair));
            previous = Some(pair);
            count += 1;
        }
        assert_eq!(count, 10_000);
    }
}
