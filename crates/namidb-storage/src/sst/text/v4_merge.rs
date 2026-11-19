//! Segment merge for `NAMIFT04`: fold several FT4 segments of one generation
//! into one, working on postings instead of document text.
//!
//! A merged segment is what a fresh build over the same winners would
//! produce, but it is assembled from the sources' own structures:
//!
//! 1. the version tables are k-way merged by `NodeId`; the highest-LSN
//!    record of each node wins (equal-LSN conflicts are corruption) and a
//!    live winner keeps the document record (length, fingerprint) of the
//!    source that holds it. Each source gets an ordinal remap: its live
//!    ordinal → the output ordinal, or nothing when a newer source shadows it;
//! 2. native-filter postings are merged per sort key and remapped;
//! 3. the dictionaries are k-way merged by term; for each term the sources'
//!    posting lists are merged by output ordinal, dropping shadowed documents.
//!
//! No document is re-read from the node corpus and no text is re-tokenized,
//! so the cost is linear in the bytes of the merged segments and the resident
//! state is one dictionary block and one posting block per source, plus the
//! ordinal remaps (8 bytes per source document).
//!
//! A delta output keeps suppressions (an older, unmerged segment may still
//! hold the node) and takes the authenticated signed sums of its sources'
//! statistics. A base output (the merge covers the generation's whole event
//! prefix) drops suppressions and derives absolute statistics from the merged
//! live documents and postings.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use futures::executor::block_on;

use super::external::{
    MergeSink, ReconciledTextDeltaStats, TextV4ExternalArtifact, TextV4ExternalBuildConfig,
    TextV4ExternalBuilder,
};
use super::{
    decode_block, decode_filter_query_mask, decode_posting_block, Posting, QueryMask, TermEntry,
    TextV4BuildContext, TextV4Reader,
};
use crate::error::{Error, Result};
use crate::search_lsm::SearchLsmState;
use crate::sst::search_delta::{
    reconcile_node_versions, SearchVersionOperation, SearchVersionRecord,
};

/// Ordinal of a source document that the merge drops.
const DROPPED: u64 = u64::MAX;
/// Winners resolved per batch of document-record reads.
const WINNER_BATCH: usize = 512;

/// Result of [`merge_text_v4_segments`].
#[derive(Debug)]
pub struct TextV4MergeOutput {
    /// `None` when no winner remains (a base whose every winner is a
    /// suppression).
    pub artifact: Option<TextV4ExternalArtifact>,
    /// Version records across the sources.
    pub input_rows: u64,
    /// Distinct nodes across the sources (one winner each).
    pub touched_rows: u64,
    /// Winners written to the output.
    pub written_rows: u64,
    /// Sum of the live output documents' lengths.
    pub live_total_len: u64,
    /// Order-sensitive digest of every winner; non-zero.
    pub winner_digest: u64,
}

/// Merge `sources` (oldest first) into one FT4 segment of `state`.
///
/// `stats` selects the output role: `Some` is a delta carrying those
/// authenticated signed sums, `None` a base. `context.complete_filter_properties`
/// must be advertised by every source. The readers' async methods are driven
/// with `block_on`; callers run this on a blocking thread over locally
/// readable sources.
pub fn merge_text_v4_segments(
    state: &SearchLsmState,
    context: TextV4BuildContext,
    config: TextV4ExternalBuildConfig,
    stats: Option<ReconciledTextDeltaStats>,
    sources: &[TextV4Reader],
) -> Result<TextV4MergeOutput> {
    if sources.is_empty() {
        return Err(Error::invariant(
            "text v4 merge requires at least one source",
        ));
    }
    for property in &context.complete_filter_properties {
        if sources
            .iter()
            .any(|source| !source.supports_filter_property(property))
        {
            return Err(Error::precondition(format!(
                "text v4 merge output advertises native filter '{property}' that a source omits"
            )));
        }
    }
    let base = stats.is_none();
    let filters = context.complete_filter_properties.clone();
    let mut builder = TextV4ExternalBuilder::with_config_presorted(state, context, config, stats)?;
    let winners = merge_winners(sources, &mut builder, base)?;
    let remaps = winners.remaps;
    let mut feed = |sink: &mut MergeSink<'_, '_>| -> Result<()> {
        merge_filters(sources, &remaps, &filters, sink)?;
        merge_terms(sources, &remaps, base, sink)
    };
    let artifact = builder.finish_presorted(&mut feed)?;
    Ok(TextV4MergeOutput {
        artifact,
        input_rows: winners.input_rows,
        touched_rows: winners.touched_rows,
        written_rows: winners.written_rows,
        live_total_len: winners.live_total_len,
        winner_digest: winners.digest.digest().max(1),
    })
}

struct Winners {
    remaps: Vec<Vec<u64>>,
    input_rows: u64,
    touched_rows: u64,
    written_rows: u64,
    live_total_len: u64,
    digest: xxhash_rust::xxh3::Xxh3,
}

struct VersionCursor {
    page: usize,
    offset: usize,
    records: Vec<SearchVersionRecord>,
}

fn merge_winners(
    sources: &[TextV4Reader],
    builder: &mut TextV4ExternalBuilder,
    base: bool,
) -> Result<Winners> {
    let mut remaps = Vec::with_capacity(sources.len());
    let mut cursors = Vec::with_capacity(sources.len());
    let mut heap = BinaryHeap::with_capacity(sources.len());
    let mut input_rows = 0u64;
    for (index, source) in sources.iter().enumerate() {
        let live = usize::try_from(source.live_document_count())
            .map_err(|_| Error::invariant("text v4 merge source is too large to remap"))?;
        remaps.push(vec![DROPPED; live]);
        let versions = source.version_reader();
        input_rows = input_rows
            .checked_add(versions.reference().record_count)
            .ok_or_else(|| Error::invariant("text v4 merge input rows overflow"))?;
        let records = if versions.page_count() == 0 {
            Vec::new()
        } else {
            block_on(versions.read_page(0))?
        };
        if let Some(first) = records.first() {
            heap.push(Reverse((first.node_id, index)));
        }
        cursors.push(VersionCursor {
            page: 0,
            offset: 0,
            records,
        });
    }

    let mut digest = xxhash_rust::xxh3::Xxh3::new();
    digest.update(b"NamiDB/TextV4SegmentMerge/v1");
    let mut touched_rows = 0u64;
    let mut written_rows = 0u64;
    let mut live_total_len = 0u64;
    let mut batch: Vec<(usize, SearchVersionRecord)> = Vec::with_capacity(WINNER_BATCH);
    while let Some(Reverse((node_id, _))) = heap.peek().copied() {
        let mut group: Vec<(usize, SearchVersionRecord)> = Vec::new();
        while let Some(Reverse((candidate, index))) = heap.peek().copied() {
            if candidate != node_id {
                break;
            }
            heap.pop();
            let cursor = &mut cursors[index];
            let record = cursor.records[cursor.offset];
            if record.node_id != node_id {
                return Err(Error::invariant(
                    "text v4 merge heap key disagrees with its version record",
                ));
            }
            group.push((index, record));
            cursor.offset += 1;
            if cursor.offset == cursor.records.len() {
                cursor.page += 1;
                cursor.offset = 0;
                let versions = sources[index].version_reader();
                cursor.records = if cursor.page < versions.page_count() {
                    block_on(versions.read_page(cursor.page))?
                } else {
                    Vec::new()
                };
            }
            if let Some(next) = cursor.records.get(cursor.offset) {
                if next.node_id <= node_id {
                    return Err(Error::invariant(
                        "text v4 merge source version table is not NodeId-sorted",
                    ));
                }
                heap.push(Reverse((next.node_id, index)));
            }
        }
        // Equal-LSN records must agree (validated here); among them the
        // newest source's copy is used.
        let winner = reconcile_node_versions(group.iter().map(|(_, record)| *record))?
            .ok_or_else(|| Error::invariant("text v4 merge version group has no winner"))?;
        let (source, record) = group
            .iter()
            .rev()
            .find(|(_, record)| record.lsn == winner.lsn)
            .copied()
            .ok_or_else(|| Error::invariant("text v4 merge lost its winning source"))?;
        touched_rows += 1;
        digest.update(&record.node_id);
        digest.update(&record.lsn.to_le_bytes());
        digest.update(&record.payload_fingerprint.to_le_bytes());
        digest.update(&[u8::from(matches!(
            record.operation,
            SearchVersionOperation::Live { .. }
        ))]);
        if base && matches!(record.operation, SearchVersionOperation::Suppress) {
            continue;
        }
        batch.push((source, record));
        if batch.len() == WINNER_BATCH {
            let (written, total_len) = flush_winners(sources, &mut remaps, builder, &mut batch)?;
            written_rows += written;
            live_total_len = live_total_len
                .checked_add(total_len)
                .ok_or_else(|| Error::invariant("text v4 merge total length overflows"))?;
        }
    }
    let (written, total_len) = flush_winners(sources, &mut remaps, builder, &mut batch)?;
    written_rows += written;
    live_total_len = live_total_len
        .checked_add(total_len)
        .ok_or_else(|| Error::invariant("text v4 merge total length overflows"))?;
    Ok(Winners {
        remaps,
        input_rows,
        touched_rows,
        written_rows,
        live_total_len,
        digest,
    })
}

/// Resolve the document records of a batch of live winners and append the
/// batch to the builder in `NodeId` order.
fn flush_winners(
    sources: &[TextV4Reader],
    remaps: &mut [Vec<u64>],
    builder: &mut TextV4ExternalBuilder,
    batch: &mut Vec<(usize, SearchVersionRecord)>,
) -> Result<(u64, u64)> {
    let mut lengths = vec![None; batch.len()];
    for (index, source) in sources.iter().enumerate() {
        let positions = batch
            .iter()
            .enumerate()
            .filter(|(_, (owner, record))| {
                *owner == index && matches!(record.operation, SearchVersionOperation::Live { .. })
            })
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        if positions.is_empty() {
            continue;
        }
        let ordinals = positions
            .iter()
            .map(|position| match batch[*position].1.operation {
                SearchVersionOperation::Live { payload_ordinal } => payload_ordinal,
                SearchVersionOperation::Suppress => unreachable!("filtered to live records"),
            })
            .collect::<Vec<_>>();
        let documents = block_on(source.read_doc_records(&ordinals))?;
        for (position, document) in positions.into_iter().zip(documents) {
            let record = batch[position].1;
            if document.node_id != record.node_id
                || document.lsn != record.lsn
                || document.payload_fingerprint != record.payload_fingerprint
            {
                return Err(Error::invariant(
                    "text v4 merge document record disagrees with its version record",
                ));
            }
            lengths[position] = Some(document.doc_len);
        }
    }
    let mut written = 0u64;
    let mut total_len = 0u64;
    for ((source, record), doc_len) in batch.drain(..).zip(lengths) {
        let ordinal = builder.push_merged_winner(record, doc_len)?;
        written += 1;
        if let (Some(ordinal), SearchVersionOperation::Live { payload_ordinal }) =
            (ordinal, record.operation)
        {
            let slot = usize::try_from(payload_ordinal)
                .ok()
                .and_then(|slot| remaps[source].get_mut(slot))
                .ok_or_else(|| {
                    Error::invariant("text v4 merge payload ordinal leaves its source")
                })?;
            *slot = ordinal;
            total_len += u64::from(doc_len.unwrap_or(0));
        }
    }
    Ok((written, total_len))
}

/// Merge the native-filter postings of every advertised property.
fn merge_filters(
    sources: &[TextV4Reader],
    remaps: &[Vec<u64>],
    properties: &[String],
    sink: &mut MergeSink<'_, '_>,
) -> Result<()> {
    // (sort key, source, value index within the source's property block).
    let mut keys: Vec<(String, usize, usize, usize)> = Vec::new();
    for property in properties {
        for (index, source) in sources.iter().enumerate() {
            let block = source
                .footer
                .filters
                .binary_search_by(|candidate| candidate.property.as_str().cmp(property))
                .map_err(|_| Error::invariant("text v4 merge source lost a filter property"))?;
            for (value, reference) in source.footer.filters[block].values.iter().enumerate() {
                keys.push((
                    MergeSink::filter_key(property, &reference.value)?,
                    index,
                    block,
                    value,
                ));
            }
        }
    }
    keys.sort();
    let mut start = 0usize;
    while start < keys.len() {
        let end = start
            + keys[start..]
                .iter()
                .take_while(|entry| entry.0 == keys[start].0)
                .count();
        let mut ordinals = Vec::new();
        for (_, index, block, value) in &keys[start..end] {
            let source = &sources[*index];
            let filter = &source.footer.filters[*block];
            let reference = &filter.values[*value];
            let compressed = block_on(source.source.read_range(reference.wire.range()?))?;
            let raw = decode_block(&compressed, &reference.wire, "filter posting block")?;
            let mask = decode_filter_query_mask(raw, reference, filter.row_count)?;
            for_each_mask_ordinal(&mask, filter.row_count, |ordinal| {
                let mapped = usize::try_from(ordinal)
                    .ok()
                    .and_then(|slot| remaps[*index].get(slot))
                    .copied()
                    .ok_or_else(|| {
                        Error::invariant("text v4 merge filter ordinal leaves its source")
                    })?;
                if mapped != DROPPED {
                    ordinals.push(mapped);
                }
                Ok(())
            })?;
        }
        ordinals.sort_unstable();
        if ordinals.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::invariant(
                "text v4 merge maps two filter postings onto one document",
            ));
        }
        for ordinal in ordinals {
            sink.filter(&keys[start].0, ordinal)?;
        }
        start = end;
    }
    Ok(())
}

fn for_each_mask_ordinal(
    mask: &QueryMask,
    row_count: u64,
    mut visit: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    match mask {
        QueryMask::Sparse(ordinals) => ordinals.iter().try_for_each(|ordinal| visit(*ordinal)),
        QueryMask::Dense(bitmap) => {
            for (byte_index, byte) in bitmap.iter().copied().enumerate() {
                let mut remaining = byte;
                while remaining != 0 {
                    let ordinal = (byte_index as u64) * 8 + u64::from(remaining.trailing_zeros());
                    if ordinal >= row_count {
                        return Err(Error::invariant(
                            "text v4 merge filter bitmap has bits past its row count",
                        ));
                    }
                    visit(ordinal)?;
                    remaining &= remaining - 1;
                }
            }
            Ok(())
        }
    }
}

struct DictionaryCursor {
    block: usize,
    offset: usize,
    entries: std::sync::Arc<Vec<TermEntry>>,
}

/// Postings of one source for the term being merged, one decoded block at a
/// time.
struct PostingStream<'r> {
    source: usize,
    reader: &'r TextV4Reader,
    entry: TermEntry,
    block: usize,
    postings: std::vec::IntoIter<Posting>,
}

impl PostingStream<'_> {
    /// Next posting of a document the merge keeps, with its output ordinal.
    fn next_kept(&mut self, remap: &[u64]) -> Result<Option<(u64, Posting)>> {
        loop {
            if let Some(posting) = self.postings.next() {
                let mapped = usize::try_from(posting.doc)
                    .ok()
                    .and_then(|slot| remap.get(slot))
                    .copied()
                    .ok_or_else(|| Error::invariant("text v4 merge posting leaves its source"))?;
                if mapped != DROPPED {
                    return Ok(Some((mapped, posting)));
                }
                continue;
            }
            let Some(reference) = self.entry.blocks.get(self.block) else {
                return Ok(None);
            };
            self.block += 1;
            let compressed = block_on(self.reader.source.read_range(reference.wire.range()?))?;
            let raw = decode_block(&compressed, &reference.wire, "posting block")?;
            // A merge re-encodes positions: keep them.
            self.postings = decode_posting_block(&raw, reference, true)?.into_iter();
        }
    }
}

fn merge_terms(
    sources: &[TextV4Reader],
    remaps: &[Vec<u64>],
    base: bool,
    sink: &mut MergeSink<'_, '_>,
) -> Result<()> {
    let mut cursors = Vec::with_capacity(sources.len());
    let mut heap: BinaryHeap<Reverse<(String, usize)>> = BinaryHeap::new();
    for (index, source) in sources.iter().enumerate() {
        let entries = match source.footer.dictionary.first() {
            Some(reference) => block_on(source.read_dictionary_block(reference))?,
            None => std::sync::Arc::new(Vec::new()),
        };
        if let Some(first) = entries.first() {
            heap.push(Reverse((first.term.clone(), index)));
        }
        cursors.push(DictionaryCursor {
            block: 0,
            offset: 0,
            entries,
        });
    }
    while let Some(Reverse((term, _))) = heap.peek().cloned() {
        let mut streams = Vec::new();
        let mut delta_df = 0i64;
        while let Some(Reverse((candidate, index))) = heap.peek().cloned() {
            if candidate != term {
                break;
            }
            heap.pop();
            let cursor = &mut cursors[index];
            let entry = cursor.entries[cursor.offset].clone();
            delta_df = delta_df.checked_add(entry.delta_df).ok_or_else(|| {
                Error::invariant(format!("text v4 merge df for term '{term}' overflows"))
            })?;
            streams.push(PostingStream {
                source: index,
                reader: &sources[index],
                entry,
                block: 0,
                postings: Vec::new().into_iter(),
            });
            cursor.offset += 1;
            if cursor.offset == cursor.entries.len() {
                cursor.block += 1;
                cursor.offset = 0;
                cursor.entries = match sources[index].footer.dictionary.get(cursor.block) {
                    Some(reference) => block_on(sources[index].read_dictionary_block(reference))?,
                    None => std::sync::Arc::new(Vec::new()),
                };
            }
            if let Some(next) = cursor.entries.get(cursor.offset) {
                if next.term <= term {
                    return Err(Error::invariant(
                        "text v4 merge source dictionary is not term-sorted",
                    ));
                }
                heap.push(Reverse((next.term.clone(), index)));
            }
        }
        sink.term(&term, (!base).then_some(delta_df))?;
        // k-way merge of the kept postings by output ordinal.
        let mut heads: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
        let mut current: Vec<Option<Posting>> = Vec::with_capacity(streams.len());
        for (slot, stream) in streams.iter_mut().enumerate() {
            match stream.next_kept(&remaps[stream.source])? {
                Some((ordinal, posting)) => {
                    heads.push(Reverse((ordinal, slot)));
                    current.push(Some(posting));
                }
                None => current.push(None),
            }
        }
        while let Some(Reverse((ordinal, slot))) = heads.pop() {
            let posting = current[slot]
                .take()
                .ok_or_else(|| Error::invariant("text v4 merge lost a posting head"))?;
            sink.posting(ordinal, posting.doc_len, &posting.positions)?;
            let owner = streams[slot].source;
            if let Some((next, posting)) = streams[slot].next_kept(&remaps[owner])? {
                heads.push(Reverse((next, slot)));
                current[slot] = Some(posting);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::io::Read;
    use std::ops::Range;
    use std::sync::Arc;

    use async_trait::async_trait;
    use bytes::Bytes;
    use uuid::Uuid;

    use super::*;
    use crate::search_lsm::{SearchEventRange, SearchLsmKind, SearchLsmStatus};
    use crate::sst::search_delta::{search_suppress_fingerprint, SearchVersionRangeSource};
    use crate::sst::text::v4::{
        text_v4_payload_fingerprint, SearchFilterValue, TextV4BuildOptions, TextV4Mutation,
        TextV4Payload,
    };

    #[derive(Debug)]
    struct MemorySource(Bytes);

    #[async_trait]
    impl SearchVersionRangeSource for MemorySource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
            Ok(self.0.slice(range.start as usize..range.end as usize))
        }
    }

    fn node(value: u64) -> [u8; 16] {
        let mut id = [0; 16];
        id[8..].copy_from_slice(&value.to_be_bytes());
        id
    }

    fn state() -> SearchLsmState {
        SearchLsmState {
            index_name: "merge_fts".into(),
            kind: SearchLsmKind::Text,
            catalog_signature: "merge-catalog".into(),
            generation_id: Uuid::from_u128(701),
            status: SearchLsmStatus::Active,
            ..SearchLsmState::default()
        }
    }

    fn context(id: u128, events: Range<u64>) -> TextV4BuildContext {
        TextV4BuildContext {
            analyzer: crate::text::TextAnalyzer::default(),
            sst_id: Uuid::from_u128(id),
            event_ranges: vec![SearchEventRange::new(events.start, events.end)],
            complete_filter_properties: vec!["status".into(), "open".into()],
        }
    }

    fn config(directory: &std::path::Path) -> TextV4ExternalBuildConfig {
        TextV4ExternalBuildConfig {
            memory_budget_bytes: 256 * 1024,
            spool_directory: Some(directory.into()),
            max_filter_distinct_per_property: 64,
            wire: TextV4BuildOptions {
                postings_per_block: 3,
                terms_per_dictionary_block: 4,
                compression_level: 1,
            },
        }
    }

    const WORDS: [&str; 12] = [
        "amber", "birch", "cedar", "delta", "ember", "fjord", "glade", "harbor", "island",
        "juniper", "kestrel", "lagoon",
    ];

    fn payload(node: u64, version: u64) -> TextV4Payload {
        let mut words = Vec::new();
        let mut state = node.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ version.wrapping_mul(31);
        for _ in 0..(3 + (node + version) % 9) {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            words.push(WORDS[(state >> 33) as usize % WORDS.len()]);
        }
        words.push(if version % 2 == 0 { "even" } else { "odd" });
        // Live documents with no tokens: a fresh build keeps them with
        // length zero and no postings, and so must a merge.
        let text = if matches!((node, version), (5, 2) | (61, 1)) {
            "-- !!".to_owned()
        } else {
            words.join(" ")
        };
        TextV4Payload {
            text,
            filters: BTreeMap::from([
                (
                    "status".into(),
                    SearchFilterValue::String(format!("e{}", (node + version) % 3)),
                ),
                ("open".into(), SearchFilterValue::Bool(node % 4 != 0)),
            ]),
        }
    }

    fn read_artifact(
        mut artifact: TextV4ExternalArtifact,
    ) -> (Vec<u8>, crate::search_lsm::SearchSegmentRef) {
        let mut body = Vec::new();
        artifact.file.read_to_end(&mut body).unwrap();
        (body, artifact.output.segment)
    }

    async fn open(body: &[u8], segment: &crate::search_lsm::SearchSegmentRef) -> TextV4Reader {
        TextV4Reader::open(
            Arc::new(MemorySource(Bytes::copy_from_slice(body))),
            body.len() as u64,
            &state(),
            segment,
        )
        .await
        .unwrap()
    }

    type Corpus = BTreeMap<u64, (u64, Option<TextV4Payload>)>;

    /// Build a delta from `(node, version)` changes; version `None` deletes.
    fn delta(
        directory: &std::path::Path,
        id: u128,
        events: Range<u64>,
        lsn: u64,
        corpus: &mut Corpus,
        changes: &BTreeMap<u64, Option<u64>>,
    ) -> (Vec<u8>, crate::search_lsm::SearchSegmentRef) {
        let mut builder =
            TextV4ExternalBuilder::with_config(&state(), context(id, events), config(directory))
                .unwrap();
        for (node_value, version) in changes {
            let before = corpus
                .get(node_value)
                .and_then(|(_, payload)| payload.clone());
            let after = version.map(|version| payload(*node_value, version));
            assert!(before != after, "test changes must change the document");
            builder
                .push(TextV4Mutation {
                    node_id: node(*node_value),
                    lsn: lsn + node_value,
                    before,
                    after: after.clone(),
                })
                .unwrap();
            corpus.insert(*node_value, (lsn + node_value, after));
        }
        read_artifact(builder.finish().unwrap().unwrap())
    }

    #[tokio::test]
    async fn merged_base_and_delta_equal_fresh_builds_byte_for_byte() {
        let directory = tempfile::tempdir().unwrap();
        let dir = directory.path();
        let mut corpus = Corpus::new();
        let mut base =
            TextV4ExternalBuilder::with_config_base(&state(), context(1, 0..1), config(dir))
                .unwrap();
        for value in 1..=60u64 {
            let after = payload(value, 0);
            base.push(TextV4Mutation {
                node_id: node(value),
                lsn: 100 + value,
                before: None,
                after: Some(after.clone()),
            })
            .unwrap();
            corpus.insert(value, (100 + value, Some(after)));
        }
        let base = read_artifact(base.finish().unwrap().unwrap());

        let mut first = BTreeMap::new();
        for value in (3..=60).step_by(3) {
            first.insert(value, Some(1));
        }
        for value in (1..=60).step_by(7) {
            first.insert(value, None);
        }
        for value in 61..=80 {
            first.insert(value, Some(1));
        }
        let first = delta(dir, 2, 1..2, 1_000, &mut corpus, &first);
        let mut second = BTreeMap::new();
        for value in (6..=60).step_by(6) {
            second.insert(value, Some(2));
        }
        second.insert(5, Some(2));
        second.insert(8, Some(3));
        second.insert(16, None);
        second.insert(70, None);
        second.insert(71, Some(2));
        second.insert(85, Some(2));
        let second = delta(dir, 3, 2..3, 2_000, &mut corpus, &second);

        let readers = vec![
            open(&base.0, &base.1).await,
            open(&first.0, &first.1).await,
            open(&second.0, &second.1).await,
        ];
        // DeltaRun: the two deltas into one signed delta.
        let stats = ReconciledTextDeltaStats {
            doc_count_delta: readers[1].delta_docs() + readers[2].delta_docs(),
            total_len_delta: readers[1].delta_total_len() + readers[2].delta_total_len(),
        };
        let merged_delta = merge_text_v4_segments(
            &state(),
            context(9, 1..3),
            config(dir),
            Some(stats),
            &readers[1..],
        )
        .unwrap();
        let mut fresh = TextV4ExternalBuilder::with_config_reconciled(
            &state(),
            context(9, 1..3),
            config(dir),
            stats,
        )
        .unwrap();
        let mut term_deltas = BTreeMap::<String, i64>::new();
        for reader in &readers[1..] {
            for block in 0..reader.term_delta_block_count() {
                for entry in reader.read_term_delta_block(block).await.unwrap() {
                    *term_deltas.entry(entry.term).or_default() += entry.delta_df;
                }
            }
        }
        let mut touched = 0u64;
        for (value, (lsn, after)) in &corpus {
            if *lsn < 1_000 {
                continue;
            }
            touched += 1;
            let record = match after {
                Some(after) => SearchVersionRecord::live(
                    node(*value),
                    *lsn,
                    text_v4_payload_fingerprint(after, crate::text::TextAnalyzer::default())
                        .unwrap(),
                    0,
                ),
                None => {
                    SearchVersionRecord::suppress(node(*value), *lsn, search_suppress_fingerprint())
                }
            };
            fresh.push_reconciled(record, after.clone()).unwrap();
        }
        for (term, delta_df) in term_deltas {
            fresh.push_term_delta(term, delta_df).unwrap();
        }
        assert_eq!(merged_delta.touched_rows, touched);
        let fresh_delta = read_artifact(fresh.finish().unwrap().unwrap());
        let merged_delta = read_artifact(merged_delta.artifact.unwrap());
        assert_eq!(merged_delta.1, fresh_delta.1);
        assert!(merged_delta.0 == fresh_delta.0, "merged delta bytes differ");
        open(&merged_delta.0, &merged_delta.1)
            .await
            .verify_all()
            .await
            .unwrap();

        // BasePrefix: base + both deltas into one absolute base.
        let merged_base =
            merge_text_v4_segments(&state(), context(10, 0..3), config(dir), None, &readers)
                .unwrap();
        let mut fresh =
            TextV4ExternalBuilder::with_config_base(&state(), context(10, 0..3), config(dir))
                .unwrap();
        let mut live = 0u64;
        for (value, (lsn, after)) in &corpus {
            if let Some(after) = after {
                live += 1;
                fresh
                    .push(TextV4Mutation {
                        node_id: node(*value),
                        lsn: *lsn,
                        before: None,
                        after: Some(after.clone()),
                    })
                    .unwrap();
            }
        }
        assert_eq!(merged_base.written_rows, live);
        let fresh_base = read_artifact(fresh.finish().unwrap().unwrap());
        let merged_base = read_artifact(merged_base.artifact.unwrap());
        assert_eq!(merged_base.1, fresh_base.1);
        assert!(merged_base.0 == fresh_base.0, "merged base bytes differ");
        open(&merged_base.0, &merged_base.1)
            .await
            .verify_all()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn merge_rejects_conflicting_equal_lsn_versions() {
        let directory = tempfile::tempdir().unwrap();
        let dir = directory.path();
        let mut corpus = Corpus::new();
        let left = delta(
            dir,
            1,
            0..1,
            10,
            &mut corpus,
            &BTreeMap::from([(1, Some(1))]),
        );
        let mut other = Corpus::new();
        let right = delta(
            dir,
            2,
            1..2,
            10,
            &mut other,
            &BTreeMap::from([(1, Some(2))]),
        );
        let readers = vec![open(&left.0, &left.1).await, open(&right.0, &right.1).await];
        let stats = ReconciledTextDeltaStats {
            doc_count_delta: 2,
            total_len_delta: readers[0].delta_total_len() + readers[1].delta_total_len(),
        };
        let error = merge_text_v4_segments(
            &state(),
            context(3, 0..2),
            config(dir),
            Some(stats),
            &readers,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("conflicting search versions"),
            "{error}"
        );
    }
}
