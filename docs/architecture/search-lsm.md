# Search LSM: incremental vector and full-text indexes

**Status:** Implemented and wired — `flush.rs::flush` prepares one delta per
registered index (`search_lsm_flush.rs`), `compact.rs` selects and installs
base/delta compactions (`search_lsm_compact.rs`), and `Snapshot` serves searches
from the active segment chain (`search_lsm_read.rs`). Unreleased: this landed
after the 2.0.6 tag.  
**Code audit:** 2026-07-27  
**Scope:** `VectorGraph` (`NAMIVG05` and successor), `TextIndex`
(`NAMIFT03` and successor), manifest/flush/compaction/read/query integration  
**Amends:** RFC-030 and RFC-035. In particular, this document replaces
RFC-035's `max(max_lsn)` coverage argument.

## Decision

NamiDB should maintain vector and full-text indexes as immutable **base +
ordered delta segments**, committed in the same manifest CAS as the `Nodes`
SSTs they cover. A search generation is usable only when the manifest proves
that every visible `Nodes` SST is covered by either:

1. a search event materialized in a segment, or
2. a `ProvenEmpty` marker produced by comparing the before/after search state.

Every non-empty delta carries a sorted `NodeId -> version` table. Updates,
deletes, relabels, removal of the indexed property, and native-filter changes
therefore shadow older payloads without mutating old objects. Query execution
searches all active segments, resolves the winning `NodeId + LSN`, and widens
before it is allowed to return an under-filled result.

The default BM25 mode remains globally exact. Delta segments store signed
corpus-stat changes, and every segment is scored with the same reconstructed
global `N`, `avgdl`, and per-query-term `df`. A Lucene-like
"deleted documents remain in statistics until merge" mode would be simpler,
but would violate the current exact-flat-scan parity contract and is not the
default proposed here.

The implementation must distinguish two notions of correctness:

- **snapshot/freshness correctness:** no write, delete, relabel, or segment is
  silently omitted; no stale version is returned;
- **ANN recall:** vector membership remains approximate unless every vector
  page is probed. The LSM must add no new false negatives beyond the selected
  ANN recall setting. An exact vector mode may exhaust pages or flat-scan.

Any missing coverage, unsupported format, corrupt page, incomplete native
filter, invalid statistic, or failed exhaustiveness proof selects the
authoritative node flat scan. It is never interpreted as an empty segment.

## Audit of the pre-Search-LSM baseline

The following facts describe the tree as it stood before this design was
implemented. They are recorded because they are what the design had to work
against; several of them — the single-body search requirement, the full-corpus
rebuild in `prepare_leveled`, and the absence of per-SST coverage — no longer
hold and are superseded by the sections below.

- `Manifest` contains index catalog descriptors, physical search SSTs, and a
  single `SearchIndexBuildState` high-water marker. It has no per-`Nodes`-SST
  coverage or segment lineage.
- `flush.rs::flush` writes `Nodes`/edge SSTs and their sidecars, then appends
  all descriptors in one manifest CAS. It does not build search deltas.
- `compact.rs::prepare_leveled` collects the complete vector/text corpus only
  during a deepest, single-node-scope merge. It then replaces every prior
  search body for an index. This is correct but creates a full-corpus rebuild
  and a flat-scan freshness window.
- `Snapshot::try_vector_search_with_point_count` and
  `Snapshot::text_search` currently require exactly one physical body. Zero or
  multiple bodies return `None`, selecting the exact fallback. This is a good
  compatibility boundary for introducing an explicit multi-segment state.
- `Snapshot::index_outrun_by_nodes` compares newer `Nodes` SSTs against index
  LSN/range metadata. This is a conservative freshness gate for one
  full-corpus body, but cannot prove that a set of independent delta bodies
  covers every write.
- The memtable vector delta already represents a touched node as
  `Some(vector)` or `None` (suppress), and the walker removes persisted hits
  shadowed by it. Full text currently falls back whenever a relevant memtable
  change alters corpus statistics.
- `NAMIVG05` is range-readable: a sparse centroid tree is resident, navigation
  and exact-rerank vectors live in independently compressed pages, and native
  filter bitmaps are local to pages.
- `NAMIFT03` is range-readable: sparse dictionary metadata is resident, while
  dictionaries, postings, positions, and document IDs are fetched by range.
  Its posting block descriptors already contain `max_tf` and `min_doc_len`,
  which are the raw ingredients for safe block-max bounds under global BM25
  statistics.
- Manifest commits use immutable object PUTs followed by the create-only
  versioned pointer CAS. The orphan repair only adopts a manifest after every
  newly referenced SST exists. Historical manifests are retained behind the
  reader pin horizon.
- Index DROP already removes every physical search descriptor with the
  matching `(kind, scope)` in the same metadata commit.

### Why `max(max_lsn)` is not coverage

Suppose base segment `B` contains node `x@10`. A later `Nodes` SST contains
`x@20` relabeled away from the index, while an unrelated delta segment reaches
LSN 30. Taking the maximum index LSN reports 30 and can incorrectly declare the
index current even though no search artifact shadows `x@10`.

The same failure appears when an updated text no longer contains a query term:
the new document is absent from that term's postings, so unioning postings
cannot discover that the old posting is stale. Coverage and a version table
are necessary; a high-water scalar is not sufficient.

## Terminology

- **catalog signature:** deterministic hash of the index descriptor, analyzer
  or metric/quantization version, and complete native-filter schema.
- **search generation:** one catalog signature plus its base/delta chain.
- **search event:** the evaluation of one newly visible `Nodes` SST against one
  registered index. Events receive a monotonically increasing sequence within
  a search generation.
- **base segment:** an absolute snapshot through an event frontier.
- **delta segment:** before/after changes for a contiguous event range after
  the base.
- **version record:** `(NodeId, LSN, operation, payload fingerprint)` for a
  search-relevant mutation.
- **suppress record:** a version record whose after-state is not a member
  (delete, relabel, missing property, invalid/zero vector under the index's
  membership rules, or text with no indexed string).
- **empty marker:** durable proof that every row in a `Nodes` SST leaves this
  index's logical before/after state unchanged.
- **shadow-only segment:** a valid version segment whose search payload could
  not be built. It preserves coverage and forces exact fallback until repaired;
  it is not an empty marker.

## Manifest model

Add a backward-compatible top-level field:

```rust
#[serde(default)]
pub search_lsm: Vec<SearchLsmState>;
```

Conceptual types follow. Exact Rust names may change, but the information and
validation rules must not.

```rust
struct SearchLsmState {
    index_name: String,
    kind: SearchKind,                 // Vector or Text
    catalog_signature: String,
    generation_id: Uuid,
    status: SearchLsmStatus,          // Building or Active
    next_event_seq: u64,
    base_frontier: Option<u64>,
    segments: Vec<SearchSegmentRef>,  // sorted by event range
    coverage: Vec<SearchCoverage>,    // one per visible Nodes SST
    compat_barrier_sst_id: Uuid,
}

struct SearchSegmentRef {
    sst_id: Uuid,                     // points into Manifest::ssts
    role: SegmentRole,                // Base, Delta, ShadowOnly
    format: SearchSegmentFormat,      // Vg5Base, Vg6, Ft3Base, Ft4
    event_ranges: Vec<SeqRange>,      // sorted, disjoint, normally one
    min_lsn: u64,
    max_lsn: u64,
    mutation_count: u64,
    live_payload_count: u64,
    suppress_count: u64,
    content_xxh3: u64,
    complete_filter_properties: Vec<String>,
    stats: SearchSegmentStats,
}

enum SearchSegmentStats {
    Vector {
        live_count: SignedOrAbsolute,
    },
    Text {
        doc_count: SignedOrAbsolute,
        total_len: SignedOrAbsolute,
        // Per-term absolute/signed df lives range-readable in the object.
    },
}

struct SearchCoverage {
    node_sst_id: Uuid,
    node_sst_max_lsn: u64,
    event_ranges: Vec<SeqRange>,
    disposition: CoverageDisposition,
}

enum CoverageDisposition {
    Segment { event_seq: u64 },
    ProvenEmpty {
        classifier_version: u16,
        before_after_digest: u64,
    },
    LogicalRewrite {
        // Node compaction introduced no new logical write. These compressed
        // ranges are inherited from the compacted input coverage entries.
        inherited_event_ranges: Vec<SeqRange>,
        input_coverage_digest: u64,
    },
}
```

`SearchLsmStatus::Building` is never queried. It allows DDL/migration to publish
intent while reads continue to use a valid legacy body or the exact flat scan.
`Active` is queried only after all invariants below validate.

The physical segment remains an ordinary `SstDescriptor` with
`kind = VectorGraph | TextIndex` and `scope = index_name`. Keeping every
physical object in `Manifest::ssts` means the current orphan repair, backup,
pin retention, rollback, and janitor live-set traversal continue to work.
`SearchSegmentRef` contains the LSM-only semantics that do not fit
`KindSpecificStats`.

### Compatibility barrier

Older 2.0.6 readers ignore unknown JSON fields, but they understand existing
`SstKind` variants. More importantly, an old reader would treat exactly one
partial delta as a full-corpus index. Every active LSM generation therefore
includes a tiny physical **compatibility barrier** descriptor of the same
`kind` and `scope`.

The barrier body has an intentionally unsupported magic (`NAMISLB1`), a
bounded checksummed footer that repeats the complete `SearchLsmState`, and
valid zero-valued kind statistics. It is identified by
`compat_barrier_sst_id`; its object path ends in `.slb`, which together with
the bounded zero-valued descriptor lets maintenance distinguish it from a
legitimate empty V5/V3 data body after an old writer drops the state. The new
reader verifies its declared object size,
magic, CRC, format version, and byte-for-byte state binding before it opens
the base. An old reader sees:

- data segment + barrier: multiple bodies, so its existing code flat-scans;
- barrier only: one undecodable body, so it flat-scans.

Thus a rolling downgrade is slow but never incomplete. An old compactor may
drop all bodies and write one legacy full-corpus body; because it also drops
the unknown top-level state when serializing its own `Manifest`, a subsequent
new reader safely recognizes the ordinary legacy generation. Old DROP INDEX
already removes every descriptor in the scope, including the barrier.

### Active-generation invariants

Validation is cheap manifest work plus footer checks when a segment is opened.
Failure of any item makes the generation unavailable:

1. The catalog contains exactly one matching descriptor and its computed
   signature equals `catalog_signature`.
2. There is exactly one LSM state for `(kind, index_name)`.
3. The barrier descriptor exists and is not listed as a data segment.
4. Every segment UUID resolves to one physical descriptor of the correct
   `kind` and `scope`. Every physical descriptor in that scope is either the
   declared barrier or is listed exactly once as a segment; an unlisted body
   invalidates the generation.
5. Segment event ranges are ordered and non-overlapping. There is at most one
   base. Deltas begin after its frontier.
6. Every visible `Nodes` SST has exactly one coverage entry with the same UUID
   and `max_lsn`.
7. The union of physical segment event ranges and `ProvenEmpty` ranges covers
   all event ranges inherited by current `Nodes` SST coverage. There is no
   uncovered or multiply owned non-empty event.
8. Base statistics are absolute; delta statistics are signed. Their sum cannot
   overflow or produce negative `N`, `total_len`, `df`, or live-vector count.
9. Every VG6/FT4 footer repeats and matches `generation_id`, catalog
   signature, segment event range, LSN bounds, and content checksum. A wrapped
   V5/V3 migration base instead uses the checksummed barrier as its lineage
   footer, plus a deterministic descriptor/native-footer fingerprint. The
   native reader still validates the legacy object's own footer checksum,
   counts, bounds, vector configuration, and complete-filter metadata.
10. Duplicate maximum LSNs for one `NodeId` must have the same operation and
    payload fingerprint. Existing online writes allocate unique increasing
    LSNs; a conflicting offline/import tie is conservatively unavailable
    rather than inventing a tie-break different from the node reader.

These checks replace `descriptor_count == 1` and
`index_outrun_by_nodes` for active LSM generations. Both legacy paths remain
unchanged when `search_lsm` is absent.

## Segment wire formats

### Common `NAMISV01` version table

Milestone 2 now has a format-neutral, range-readable winner table shared by
VG6 and FT4. Every non-empty segment embeds one table:

```text
+------------------+--------------------------+----------------------+
| 160-byte header  | 48-byte version records  | sparse page directory|
+------------------+--------------------------+----------------------+
```

Each record is `(NodeId, LSN, Live(payload_ordinal)|Suppress,
payload_fingerprint)`. Records are strictly sorted by `NodeId`, reconciled to
one winning LSN before writing, and grouped into 512-record pages (about
24 KiB). Equal-LSN repeats are accepted only when operation class and payload
fingerprint agree; payload ordinals are deliberately excluded because they
are local to each segment.

The writer accepts any `Write + Seek` spool and retains one page plus a bounded
sparse directory, never the corpus. At 10 million records the directory is
about 1.5 MiB; the uncompressed table is about 480 MB, small relative to a
1024d f32 vector corpus and independently range-readable. Opening fetches only
the fixed header and directory. One exact winner probe fetches one page, while
batch probes coalesce NodeIds by page and cap each I/O batch.

Integrity is layered:

- CRC32 over the fixed header and every independently fetched record page;
- XXH3 over the sparse directory;
- a semantic XXH3 over all ordered record bytes, verifiable by a bounded
  streaming scrub;
- exact footer/reference agreement on byte range, record/live/suppress counts,
  NodeId bounds, LSN bounds, page count, and semantic checksum.

`SearchSegmentWireBinding` is the common native-footer lineage block. It
repeats the generation UUID, index name, catalog signature, complete
`SearchSegmentRef`, and version-table reference. It accepts only VG6/FT4 and
checks the manifest mutation/live/suppress counts and LSN bounds. The
format-specific footer will add vector or postings directories and bind its
own complete-object checksum.

An empty logical event does not produce an empty table: it uses
`ProvenEmpty` coverage. This keeps "builder omitted the payload" distinct from
"exact comparison proved no search change."

### `NAMIVG06`

The first V6 production mode is the exact flat delta used by flush-sized
segments:

```text
+----------+----------------+----------------------+-------------------+
| NAMIVG06 | NAMISV01      | zstd exact f32 pages | filter blocks     |
+----------+----------------+----------------------+-------------------+
| checksummed lineage/config/page directories footer             | trailer |
+-----------------------------------------------------------------+---------+
```

`write_delta_v6` streams this object to any `Write + Seek` spool. It sorts and
classifies exact before/after images, elides canonical no-ops, emits live rows
or suppressions, and retains only one exact page plus sparse filter events for
the bounded flush batch. It does not clone vectors into the page builder.
`NAMIDB_INDEX_BUILD_MEMORY_BYTES` rejects an oversized batch before auxiliary
state is built; the in-memory `build_delta_v6` remains a convenience for small
deltas and tests.

Each exact page repeats `(NodeId, LSN, payload_fingerprint)` beside the f32
vector, is independently zstd-compressed, and has CRC32 plus raw XXH3. The
reader can exhaustively score pages using cosine, dot, or Euclidean distance
with a bounded top-k heap. `verify_all` additionally checks every live page
row against its `NAMISV01` ordinal.

Every advertised native-filter property has a sorted typed-value directory.
Each value points to its own independently compressed range: increasing
ordinals use delta-varints while sparse and switch to a bitmap only when the
bitmap is smaller. Wire and build cost are therefore `O(rows)`, not
`O(distinct values × rows)`. Presence means complete coverage, including an
authoritative property with no values. Query masks also remain sparse unless
density justifies a bitmap; equality alternatives are ORed, groups are ANDed,
and unsupported properties are explicitly left for exact residual handling.
The scalar key domain covers Bool, i64, finite canonical f64, String, Bytes,
Date, and DateTime; null/complex/vector values remain residual predicates.

Unique identity keys are not implicit filter obligations. Only explicitly
indexed, non-unique Bool/String properties enter the vector catalog signature
and complete-filter list; this keeps keys out of every segment unless a future
dedicated opt-in says otherwise.

The footer binds format mode, vector configuration, signed live-count delta,
all section checksums, the common lineage block, and a canonical non-circular
content digest repeated by `SearchSegmentRef`. Clustered i8 navigation pages
remain the next V6 base/large-delta mode; small deltas never pay ANN build
amplification.

V5 may be wrapped as a read-only base during vector migration because a newer
V6 version table can shadow its candidates. A full V6 rebuild is still needed
before segment-only point membership and incremental compaction are maximally
efficient.

Updates append the new vector as a live row and shadow the old row through the
version table. Deletes, relabels, and property removal append only a suppress
record. Graph ordinals are local to a segment and never reused across objects.

### `NAMIFT04`

FT4 now has one concrete role-aware segment layout:

```text
+----------+-----------+------------+----------+------------+---------+
| NAMIFT04 | NAMISV01 | fixed docs | postings | sparse dict| filters |
+----------+-----------+------------+----------+------------+---------+
| role-relative stats + lineage + section-checksum footer | trailer |
+----------------------------------------------------------+---------+
```

`TextV4ExternalBuilder` consumes NodeId-sorted documents one at a time. Delta
mode accepts classified before/after mutations and publishes signed
statistics. Base mode accepts only authoritative live after-images, prohibits
suppressions, and publishes absolute `N`, `total_len`, and per-term `df`.
Both modes write one fixed live-document record
`(NodeId, LSN, fingerprint, doc_len)` per payload ordinal and fold token
occurrences, df events, and `(property,value,ordinal)` filter events through
checksummed external-sort runs. A binary-counter merge keeps the number of
live runs logarithmic. The final run is consumed one term or filter value at a
time; term entries and filter payloads are replayed from local spools, so the
corpus is never resident.

The bounded convenience `write_delta_v4` rejects a caller-owned mutation
vector above its cap; production flushes push directly into the external
builder. Operational cardinality, dictionary-directory, single-posting, or
scratch limits fail explicitly. They never silently remove an advertised
complete filter.

The decoder provides exact signed term lookups and an exhaustive phrase/prefix
BM25 scorer that requires caller-supplied snapshot-wide `N`, `total_len`, and
`df`. It never substitutes segment-local IDF. Each query advances one posting
block per term in document order, evaluates phrase positions at the current
document, and retains only those blocks plus a top-k heap. A common term is
therefore exhaustively scored without `Vec<all postings>` or a global candidate
set. Native-filter postings use the same adaptive sparse/bitmap representation
as VG6 and are range-read only for requested alternatives. Resumable block-max
upper bounds remain a later latency optimization, not a prerequisite for
bounded-memory correctness.

CRC32 protects compressed blocks, raw XXH3 protects decoded pages, the fixed
document table and postings metadata have streaming digests, and
`verify_all` checks every live document against its `NAMISV01` ordinal plus all
dictionary/posting/filter invariants.

For one changed document:

```text
delta_docs      = member(after) - member(before)
delta_total_len = len(after) - len(before)
delta_df(term)  = contains(after, term) - contains(before, term)
```

Term frequency and positions belong only to the new live document's postings.
An unchanged term can have `delta_df = 0` while its TF, positions, or document
length changes; the new document still receives a live version record and new
postings.

An FT4 base carries absolute `N`, `total_len`, and `df`; its footer verifier
also requires `mutation_count == live_payload_count`, zero suppressions,
absolute document count equal to the fixed document table, per-term absolute
df equal to live postings, and total length equal to the scrubbed document
records. Summing the ordered signed deltas reconstructs exact snapshot-wide
statistics without scanning all
postings. Arithmetic uses checked signed intermediates and is validated before
conversion to unsigned values.

## Building a delta at flush

The search delta must be derived from **before and after images**, not merely
from the new node row:

1. Freeze the memtable. Multiple writes to one node are already reconciled to
   the highest LSN; preserve the first before-image and final after-image.
2. For each active index and touched `NodeId`, obtain the state immediately
   before this frozen batch. This can be captured when the node is first
   modified or batch-read from the pinned base snapshot using the exact node
   locator. It must not be inferred from optional search payloads.
3. Canonicalize before/after membership with the same label dictionary,
   property order, analyzer, vector dimension/zero rules, and filter encoding
   used by queries.
4. If every touched node has identical logical search state, write a
   `ProvenEmpty` coverage event.
5. Otherwise spool version records and live payloads, external-sort by the
   required keys, build one delta object (or `ShadowOnly` on a deterministic
   payload-build rejection), and write it alongside the node SST.
6. PUT the node body, sidecars, search segment(s), and any first-generation
   barrier. Only after all PUTs complete, append all descriptors, coverage,
   and segment state in one manifest CAS.

The delta builder must use bounded memory. Vectors, text, postings, and filter
events go to the existing local spool/external builders; memory is bounded by
the configured build budget, while temporary disk is an explicit operational
requirement. No `Vec<all vectors>` or `Vec<all documents>` is acceptable on
the flush or compaction path.

An object-store or spool I/O failure fails the flush just as a failed node SST
upload does. A deterministic ANN/postings build rejection may publish a
`ShadowOnly` segment so node writes are not permanently wedged, but that search
generation flat-scans until repair replaces it with a complete payload.

### What counts as an empty event

An empty marker is allowed only after exact before/after comparison:

- an unrelated-label upsert is not automatically empty; it may be a relabel
  away from the indexed label;
- a tombstone is not automatically empty; the node may have been a member;
- changing a native-filter property is a vector/text change even when the
  vector/text itself is byte-identical;
- changing an unindexed property is empty when label, indexed payload, and all
  native-filter values are unchanged;
- a newly observed unrelated node is empty only when the before state proves
  it was not a member.

## Query algorithms

### Generation selection

1. Look up the catalog descriptor.
2. If a matching `Active` LSM state validates, use it.
3. Else, if no LSM state exists and exactly one fresh legacy body exists, use
   the current legacy path.
4. Otherwise use the exact node flat scan.

All data segments are opened from immutable, generation-pinned paths through
the RAM/NVMe/range cache. Missing or corrupt optional search data invalidates
the whole generation for that query; it is never skipped.

### Winner oracle: dedupe by `NodeId + LSN`

Each query owns a small candidate winner cache. For a candidate
`(NodeId, candidate_lsn, segment_seq)`:

1. Check the staged/committed memtable delta. If touched there, that value or
   suppress record wins.
2. Point-probe version tables from newest relevant delta to oldest, pruning by
   LSN and NodeId range. The highest LSN wins.
3. If no delta touches the ID, a base candidate is current.
4. Accept only when the winning record is live and its
   `(LSN, fingerprint, segment)` matches the candidate payload.

A candidate from an old segment can therefore never survive a delete, relabel,
or update merely because the new content does not match the query. A newer live
payload is independently searchable in its own segment.

Version point probes are range reads and are page-cacheable. Search compaction
and a hard live-segment cap bound their fan-out.

### Vector search

For each segment:

1. Apply every supported equality group inside its page bitmaps before local
   top-k. Native filtering is enabled only if every live-payload segment
   advertises the complete requested property.
2. Probe clustered V5/V6 pages (or exact flat pages for small deltas).
3. Rerank candidates from exact f32 pages.

The coordinator merges segment candidates according to the index metric,
checks the winner oracle, removes memtable-suppressed IDs, merges exact
memtable live vectors, and deduplicates by `NodeId`.

If shadowed or residual-filtered candidates leave fewer than `k`, geometrically
widen only non-exhausted segments. Prefer segments whose next centroid/page
bound can still beat the current kth score. A finite `eligible_count` is an
exhaustiveness proof only after all relevant pages have been read. If the
widening/resource cap is reached without `k` or an exhaustion proof, use the
exact vector flat scan; never return a short page caused by stale candidates.

For the normal approximate mode, every segment is searched and no segment is
lost, but neighbor recall remains governed by `nprobe`/rerank settings. An
`exact` mode probes every exact vector page and provides true brute-force
top-k with the same winner reconciliation.

### Globally exact BM25

Reconstruct one set of global statistics:

```text
N         = base.N         + sum(delta_docs)
total_len = base.total_len + sum(delta_total_len)
df(t)     = base.df(t)     + sum(delta_df(t))
avgdl     = total_len / N
```

All segments score with those same values. This is essential: merging top-k
lists scored with segment-local IDF is not globally correct.

Prefix expansion is also global. K-way merge segment term dictionaries in
lexicographic order, sum signed `df`, skip terms whose resulting `df == 0`, and
take the same corpus-wide expansion limit used by the flat scorer. Phrase
positions remain segment-local.

For finite `k`, start a resumable block-max/WAND cursor on every payload
segment with the global statistics:

1. Fetch a small ranked batch and a safe upper bound for each segment's unseen
   candidates.
2. Globally merge candidates, reject stale versions through the winner oracle,
   and apply complete native filters or residual filters.
3. Stop only when `k` live results exist and every unseen upper bound is no
   greater than the kth live score.
4. Otherwise resume/widen the segments whose bounds can still compete.
5. If a safe bound is unavailable or a resource cap is reached, use the exact
   two-pass node scorer.

Stale postings may increase work, but cannot cause a false result: their block
bounds overestimate possible live scores, which is safe. `k = None` exhausts
all candidate streams and applies the winner oracle.

BM25 corpus statistics are always unfiltered. Native filter bitmaps restrict
candidates before scoring/WAND, but do not change `N`, `avgdl`, or `df`,
matching the existing flat scorer.

### Explicitly rejected BM25 shortcut

Summing per-segment `doc_freq` while ignoring updated/deleted older documents,
or retaining deleted documents in `df` until merge, is operationally common
and often acceptable for ranking. It is not exact NamiDB BM25. If a future
`segment_approx` mode is added, it must be opt-in and observable in query
metadata/metrics; it cannot silently replace the default contract.

## Search-segment compaction

Search compaction is independent of node-body compaction:

- compact only adjacent event ranges;
- newest `(NodeId, LSN)` wins inside the selected range;
- preserve a suppress record when an older unselected segment may still carry
  that ID;
- merge/sum signed FTS statistics with checked arithmetic;
- rebuild vector clusters/filter pages from the selected live winners using
  bounded external spools;
- merge FTS postings/positions/term vectors with streaming external sorts;
- replace selected descriptors with one output descriptor in a manifest CAS;
- transfer their event ranges to the output without changing logical sequence.

Compacting base plus a contiguous delta prefix emits a new absolute base at the
new frontier. Compacting deltas only emits one signed delta. A periodic full
base rebuild removes all stale payloads and suppression records, recomputes
centroids/ANN quality, recomputes exact BM25 statistics, and may reset the
generation's event numbering and coverage entries atomically to cap manifest
history.

Node-body compaction has a separate but equally important obligation. Replacing
one or more `Nodes` SSTs must replace their coverage entries with one
`LogicalRewrite` entry for the output SST (or remove the entries when the
authoritative output is empty). The rewrite inherits the exact union of input
event ranges and a deterministic digest of the input coverage; it never
allocates a new event or changes a search winner. Because the compatibility
barrier binds the complete state, the same manifest CAS must also publish a new
barrier and retire the previous one.

An off-lock node compaction may be outrun by ordinary flushes. Installation
must therefore rebase the captured coverage rewrite onto the current state when
the catalog, generation, captured input coverages, and captured segment prefix
still agree. Concurrently appended segments, proven-empty ranges, and coverage
entries are retained. Requiring byte-for-byte equality with the captured
`search_lsm` state would starve compaction under a sustained node write stream;
silently installing the old state would lose search events. If the prefix
proof fails, the prepared output is abandoned and reclaimed as an orphan.
Encoding and uploading the small rebased barrier is the only search-specific
work allowed in the short install phase; all corpus work remains outside the
foreground writer lock.

Recommended triggers are:

- delta segment count (hard cap first, e.g. 8-16);
- total delta bytes relative to base;
- suppress/stale ratio;
- vector recall/cluster drift;
- FTS WAND work amplification;
- native-filter schema/catalog signature change.

### Interaction with node compaction

Node compaction is a logical no-op: it reconciles already committed rows and
may garbage-collect an authoritative tombstone, but introduces no new write.
When it replaces input `Nodes` SSTs:

1. require valid coverage entries for every input;
2. remove those entries;
3. attach one `LogicalRewrite` coverage entry to the output with the compressed
   union of inherited event ranges and a digest of the inputs;
4. preserve the search segments unchanged.

This update happens in the same node-compaction manifest install. A newly
arrived flush and its coverage survive a prepared-compaction rebase. Search
compaction similarly verifies that every selected segment is still referenced
and preserves any newer appended deltas. Catalog signature changes abort both
prepared installs.

## Correctness argument

Let `W(x)` be the highest-LSN node state for `NodeId x` in the pinned snapshot.

**Coverage lemma.** In an active generation, every persisted node mutation
relevant to the index belongs to a covered search event. Non-empty events have
a segment version record; empty events have an exact before/after equality
proof. This follows from active invariant 6-7 and the atomic flush CAS.

**Winner lemma.** For any candidate payload for `x`, the winner oracle accepts
it iff it represents `W(x)` and `W(x)` is an index member. Every later change
has a version/suppress record by the coverage lemma, and the oracle chooses the
highest LSN. Therefore stale payloads cannot be returned.

**Live-payload lemma.** If `W(x)` is a member and differs from its prior state,
its event contains a live payload in some segment; if it is unchanged, the
previous accepted payload remains valid. Therefore every current member is
represented exactly once after winner reconciliation.

**Vector conclusion.** Searching every live-payload segment and merging after
winner reconciliation introduces no LSM-specific omission. Approximate recall
is solely the ANN probe's contract; exhaustive-page mode is exact.

**BM25-stat lemma.** Per-event signed changes are exact before/after
differences, so telescoping from the absolute base yields the exact snapshot
`N`, `total_len`, and `df(t)`.

**BM25-ranking conclusion.** Every live document is represented, every segment
uses identical global statistics, stale documents are rejected, and execution
stops only under a safe unseen-score bound. The returned top-k therefore equals
the existing exact flat scorer (including phrase/prefix semantics and NodeId
tie-breaks).

If any premise cannot be established, generation selection chooses the flat
scan, so uncertainty cannot become a false negative or stale result.

## Crash, CAS, rollback, and garbage collection

- Segment paths are UUIDv7 immutable objects.
- Flush/compaction PUT every object before writing the manifest body/pointer.
- A crash before CAS leaves unreferenced objects; the janitor reclaims them.
- A crash after the manifest body but before the pointer is handled by existing
  stalled-commit repair. Because every segment/barrier has an
  `SstDescriptor`, orphan adoption verifies its object exists.
- A lost flush CAS rebuilds its event sequence and before-image comparison
  against the new current manifest; it must not blindly rebase signed stats.
- A search-compaction CAS may rebase only when all selected segment IDs,
  generation/catalog signature, and coverage digest still match. Newer
  unselected deltas are preserved.
- Historical manifests pin their exact segment set. The janitor's retained
  manifest union keeps those objects alive, so rollback reads the old base and
  deltas without consulting current state.
- DROP removes LSM state, barrier, and every scoped data descriptor in one CAS.
- CREATE with the same name but a different signature receives a new
  `generation_id`; old build markers or segment bodies are never inherited.

## Migration and 2.0.6 compatibility

1. `search_lsm` defaults to empty, so old manifests decode unchanged.
   This is JSON/wire compatibility. Adding a public Rust struct field does
   require downstream crates that construct `Manifest` with a literal to add
   the field; callers using `Manifest::empty`/`next_version` are unaffected.
2. With no LSM state, the new reader retains the existing exactly-one-body
   legacy path and freshness gates.
3. To migrate a fresh legacy full-corpus body:
   - pin the manifest and verify the legacy body is not outrun;
   - range-probe its V5/V3 magic and retain the immutable data object;
   - create coverage entries for every visible `Nodes` SST;
   - PUT a fresh checksummed compatibility barrier, then add its descriptor and
     activate the state in one CAS. No corpus rewrite is required.
4. Vector V5 may temporarily serve as a base. Text V3 must either gain a
   caller-supplied global-stat scoring cursor or be rebuilt to FT4 before text
   deltas can be served exactly.
5. If node writes race the migration, keep status `Building` and either build
   catch-up deltas or retry activation against the new manifest. Never stamp a
   body forward by LSN.
6. An older 2.0.6 reader sees the barrier and flat-scans, as described above.
7. An older writer that rewrites the manifest drops the unknown LSM state; the
   remaining multiple bodies still cause exact fallback. If it also replaces
   or removes the barrier while retaining one fresh legacy full body, the next
   new maintenance pass re-adopts that body with a new barrier rather than
   rebuilding the corpus.

No in-place object rewrite is required. Rollback is selecting an older
manifest version.

## Minimum safe implementation path

The following order is deliberately useful at every commit and never exposes a
partially correct multi-segment index:

### Milestone 1: state, validator, and downgrade barrier

- Implemented: manifest types with `serde(default)`, cloning in
  `next_version`, canonical catalog signatures, and the pure validator.
- Implemented: checksummed barrier creation/removal as an ordinary
  same-kind/scope SST descriptor, so existing manifest pin, backup, orphan,
  and janitor traversal retain it without a parallel discovery mechanism.
- Implemented: metadata-only adoption of a fresh V5/V3 full-corpus body as one
  active base covering every current `Nodes` SST. A concurrent node commit
  cancels activation and leaves exact fallback; it never stamps coverage
  forward.
- Implemented: new readers validate barrier and native legacy footer binding
  before using the one base. Missing/corrupt/ambiguous artifacts and all
  multi-segment generations still flat-scan. Old readers flat-scan because of
  the barrier.

### Milestone 2: vector deltas

- Implemented foundation: common `NAMISV01` streaming version/suppress table,
  sparse point-probe directory, bounded batch probes/scrub, and native-footer
  lineage binding used by both VG6 and FT4.
- Implemented wire: VG6 exact flat-page deltas, typed complete-filter blocks,
  signed live-count statistics, streaming spool writer, range reader, exact
  scorer, and full integrity scrub.
- Pending engine integration: build the delta atomically in flush and search
  every active vector segment through the winner oracle.
- Pending large-segment mode: clustered i8 navigation pages above the
  small-delta threshold.
- Build the vector delta atomically in flush from exact before/after images.
- Search every vector segment, apply native filters, winner-check candidates,
  and widen. Under-fill or any incomplete proof falls back to the exact scan.
- Keep current memtable-delta merge.

This is the smallest materially useful incremental feature. It removes the
post-flush vector flat-scan window while handling update/delete/relabel without
stale results. It does not claim to improve ANN recall by itself.

### Milestone 3: FTS deltas without unsafe ranking

- Implemented wire: FT4 fixed documents, compressed positional postings,
  sparse signed-`df` dictionary, typed complete-filter blocks, exact global-stat
  scorer, and full integrity scrub.
- Pending engine integration: attach exact before/after FT4 deltas in flush.
- Until global-stat cursors and safe unseen bounds are implemented, manifests
  may contain covered FT4 deltas but `text_search` must flat-scan whenever
  `segments.len() > 1` or any segment is `ShadowOnly`.

This stage already makes crash-safe catch-up/compaction possible and has zero
false negatives, because it does not serve an unproven segmented ranking.

### Milestone 4: globally exact segmented BM25

- Add global term-stat reconstruction, prefix dictionary merge, resumable
  WAND/block-max cursors, winner reconciliation, and native filters.
- Turn on segmented serving only after randomized parity tests pass.

### Milestone 5: independent segment compaction

- Tiered delta compaction, base-prefix rebuild, lineage transfer during node
  compaction, bounded external builders, metrics, and hard segment caps.

The unsafe shortcut to avoid is enabling the old "loop all bodies, concatenate
top-k, and trust max LSN" behavior. It fails specifically for existing-key
updates, deletes, relabels, and text-stat comparability.

## Implementation map

- `manifest.rs`: LSM types, serialization defaults, active validator, descriptor
  lookup by UUID.
- `ingest.rs`: CREATE/DROP generation lifecycle and barrier handling.
- `flush.rs`: before-image batch lookup, bounded delta spools, atomic
  descriptors/coverage commit.
- `sst/vector/v6.rs`: version table, delta flat pages, footer metadata, point
  probe; reuse V5 page/tree codecs where possible.
- `sst/text/v4.rs`: version table, signed term stats, global-stat cursor,
  resumable block-max search; reuse V3 posting/dictionary codecs.
- `read.rs`: generation selection, segment opening, winner oracle, multi-segment
  vector and BM25 coordinator. Storage should return already reconciled hits;
  the query executor should not duplicate manifest semantics.
- `compact.rs`: remove full-corpus search rebuilding from node compaction,
  transfer coverage for logical rewrites, and add an independent search
  compactor.
- `walker.rs`: retain residual hydration/widening and exact fallback; consume
  explicit storage exhaustiveness/availability results.
- `janitor.rs`, `backup.rs`, metrics: no new object-discovery mechanism; verify
  that all physical artifacts remain ordinary SST descriptors and expose
  fallback/segment/range-read counters.

## Tests and release gates

### Manifest and compatibility

- JSON fixture from 2.0.6 decodes with empty LSM state.
- New manifest round-trip preserves every state field.
- Validator rejects missing/duplicate coverage, overlapping event ranges,
  wrong UUID/kind/scope/signature/barrier, invalid stats, and unlisted objects.
- A 2.0.6-compatible reader fixture sees `data + barrier` and selects flat
  scan; barrier-only also selects flat scan.
- Old-writer-style manifest rewrite (unknown field dropped) remains safe.

### Mutation correctness

For both vector and text, exercise:

- insert, update in place, repeated update before flush;
- delete;
- relabel into and out of the index label;
- indexed-property removal and restoration;
- native-filter-only change;
- unrelated-property change producing `ProvenEmpty`;
- same NodeId across base and several deltas;
- empty corpus, one member, and `k > live corpus`;
- conflicting equal-LSN fingerprints select fallback.

For every manifest step, compare returned IDs/scores with a pinned flat scan.
Assert no stale NodeId/version is returned.

### BM25 exactness

- Randomized operation sequences compared with the existing two-pass scorer.
- Exact score parity for terms, multi-term queries, phrases, prefix expansion,
  `k = 0/1/N/None`, ties, deletes, and relabels.
- Global `N`, `total_len`, and sampled `df` equal a flat recount after every
  delta and compaction.
- Native filters do not change corpus statistics.
- Stale high-scoring postings force cursor widening and never starve a lower
  live result.
- Unsafe/missing upper bounds force flat fallback.

### Vector quality and filtering

- Recall@k against exact brute force per metric on base-only, many deltas,
  update-heavy, delete-heavy, and post-compaction states.
- LSM recall is no worse than the configured per-segment ANN floor beyond a
  stated tolerance; exact-page mode is identical to brute force.
- Selective native filters return `k` when at least `k` eligible vectors exist,
  or prove exact exhaustion; incomplete filter coverage is `Unsupported`.
- Stale candidates cause widening/fallback, not a short page.

### Crash/CAS/rollback

Inject failure after each:

- node PUT;
- search segment PUT;
- barrier PUT;
- manifest body PUT;
- pointer CAS;
- search-compaction output PUT.

Verify orphan cleanup/adoption, retry idempotence, no visible half-generation,
prepared-compaction CAS loss, concurrent flush preservation, pinned historical
reads, and rollback after base/delta compaction.

### Resource and object-store gates

- Cold query bytes/range requests are proportional to probed pages/terms, not
  corpus size.
- Warm query hits RAM/NVMe cache and performs no body-sized GET.
- Builder peak RSS stays within the configured memory budget on a corpus much
  larger than RAM; temp-disk use and cleanup are bounded/observable.
- Reader resident metadata is bounded by segment cap and sparse directories.
- Segment count hard cap/backpressure works under a long write stream.
- Ten-million-node benchmark records vector recall/p50/p95/p99, FTS parity and
  latency, cold/warm object-store bytes, compaction throughput, RSS, NVMe use,
  and update/delete amplification.

### Mandatory release gates

- all feature combinations compile;
- legacy fixtures and downgrade-safety tests pass;
- randomized flat-parity suite passes with no stale/missing mutation;
- crash matrix passes;
- long update soak has bounded RSS and segment count;
- published benchmark meets the chosen recall/latency/object-read budgets.

Only after these gates should the LSM path become the default and be included
in a production tag/image.

## Operational metrics

At minimum expose:

- active/building/invalid generation count;
- coverage gaps and validator failure reason;
- base/delta/shadow-only segment count and bytes;
- stale candidates rejected and winner point probes;
- vector page probes, widening rounds, exact fallbacks, recall benchmark;
- FTS dictionary/posting blocks, WAND advances, unseen-bound fallbacks;
- signed-stat validation failures;
- search compaction backlog, bytes read/written, stale ratio;
- range-cache RAM/NVMe hit rate and object-store bytes by vector/text;
- build spool bytes, peak builder memory, and cleanup failures.

These distinguish an expected cold range read from a correctness fallback or a
runaway delta chain.
