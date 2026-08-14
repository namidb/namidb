# Object-native storage contract

## Decision

NamiDB can serve a ten-million-node property graph, vector search, and full
text search with object storage as the durable source of truth and without
keeping an `O(corpus)` data structure in DRAM. It cannot, and does not claim
to, run with literally zero memory. Query state, decoded pages, result heaps,
network buffers, the active write batch, and a small amount of routing
metadata must be resident while work is in flight.

The production contract is therefore:

- R2/S3 contains every durable WAL, manifest, graph, vector, text, and
  secondary-index artifact.
- Local NVMe and RAM contain only reconstructible, byte-bounded cache entries
  and bounded transient workspaces.
- A cold query is valid without preloading a namespace. It performs immutable,
  generation-pinned range reads and may be slower than a warm query.
- No reader, builder, flush, or compaction path may materialize the corpus,
  all vectors, all postings, all edges, or all index values in memory.
- A configured bound is enforced before allocation. Exceeding it returns a
  deterministic error or applies backpressure; it never silently drops an
  accelerator and falls back to an `O(N)` scan.

This is the same broad separation used by object-native search systems:
durable blob storage, a local SSD cache for locality, and memory for the
hottest pages and active query state. Turbopuffer publicly describes direct
small range reads on a cold namespace and RAM/NVMe as caches rather than the
authoritative database.[^tpuf-architecture]

## Why the physical indexes differ

One object layout is not optimal for all three workloads.

### Vectors

Graph ANN structures are excellent when their navigation graph is local, but
each dependent hop can become another high-latency object-store request.
NamiDB's object-native vector format instead uses a bounded hierarchical
centroid/routing layer and independently range-readable cluster pages:

1. descend the centroid hierarchy, reading at most one bounded batch per
   visited level;
2. select clusters, incorporating cluster-level attribute summaries;
3. fetch selected cluster pages in one bounded parallel batch;
4. apply exact row-level native filters and score compact per-row
   representations while decoding;
5. fetch full-precision vectors only for a bounded candidate set and rerank it
   with the requested metric;
6. widen deterministically when stale Search-LSM candidates or selective
   filters underfill `k`.

SPANN separates a small centroid structure from large posting lists on
secondary storage.[^spann] SPFresh adds local partition split/merge and
boundary reassignment so updates do not require recurring global
rebuilds.[^spfresh] Those are the relevant ideas; NamiDB's on-disk wire,
manifest lineage, and update reconciliation remain NamiDB-specific.

Turbopuffer's 2026 ANN v3 description independently reinforces the same
hierarchical-clustering and approximation/refinement shape: hierarchy bounds
cold object-store round trips, compact vectors narrow the candidates, and
full-precision vectors are fetched only for exact reranking.[^tpuf-ann-v3]
NamiDB currently uses signed int8 navigation/data codes rather than RaBitQ.
One-bit or product quantization is therefore an optional future wire version,
not a silent change to the distance semantics: it may ship only after
per-metric recall, filtered recall, size, and cold-byte gates beat the int8
baseline.

Native filtering is part of candidate generation, not a post-filter. Each
filterable value has both cluster-level summaries and exact local bitmaps.
This lets the planner skip irrelevant clusters and still return `k` matching
rows when they exist. The address model is `(cluster, local ordinal)`, matching
the central idea in object-native filtered ANN designs.[^tpuf-filtering]

### Full text

Full text uses a sparse term dictionary and independently checksummed,
compressed posting blocks. Blocks have a target posting count rather than
being tied to vector clusters. This avoids thousands of tiny objects/ranges
for Zipfian vocabularies; fixed-size posting blocks are also the direction
reported by Turbopuffer's second-generation FTS design.[^tpuf-fts]

Every block carries enough impact metadata for a resumable Block-Max
WAND/MaxScore cursor. Snapshot-wide `N`, total document length, and signed
per-term document-frequency deltas are reconciled before scoring; using
segment-local IDF would change ranking after every incremental update. Until
an unseen-score upper bound proves safe early termination, the implementation
must use an exact exhaustive fallback rather than return an approximate BM25
ordering. WAND/MaxScore and auxiliary document-length data are established
inverted-index techniques.[^pisa]

The canonical target is 256 postings per normal block, with explicit byte
limits for pathological positions. Delta coding is required; SIMD bitpacking
is a performance optimization that may replace the scalar codec without
changing term/block boundaries or BM25 results. This mirrors the fixed
128–512 posting envelope described for Turbopuffer FTS v2 while keeping
NamiDB's checksum and Search-LSM contracts independent.[^tpuf-fts]

Positions are stored for phrase queries. Prefix expansion is deterministic
and bounded. Builders external-sort occurrences and stream positions,
postings, term entries, dictionaries, and filters through temporary files;
one common term or one large document cannot create an unbounded `Vec`.

### Property graph

Nodes are row-grouped Parquet plus range-readable exact-node, label, unique,
equality, and ordered-property sidecars. Edges are chunked ordered adjacency
in both directions (CSR and CSC semantics), with:

- paged topology, offsets, LSNs, and tombstones;
- exact `(source, type, destination)` point lookup;
- separately addressable property records;
- checksummed pages and a footer bound to the immutable object generation.

GraphAr likewise chunks vertices and edges, supports source- and
destination-aligned adjacency, and separates property groups so irrelevant
columns need not be read.[^graphar] NamiDB uses that physical principle, but
retains its own transactional LSN and tombstone semantics.

High degree is not a license to build a full in-memory adjacency cache. Normal
traversal merges sorted paged SST slices with the bounded memtable delta. A CSR
cache remains an explicit latency optimization, disabled in the low-memory
profile and charged to the process-wide cache budget when enabled.

## Incremental consistency: Search-LSM

Every committed node mutation receives one monotonically ordered search event.
Vector and text indexes contain an immutable base plus bounded immutable delta
segments. A segment stores:

- its exact event ranges and node-SST coverage;
- a sorted version table of `(NodeId, LSN, operation, fingerprint, ordinal)`;
- complete after-images or explicit suppress/tombstone records;
- exact before/after statistic deltas;
- complete native-filter postings;
- a footer bound to catalog signature, generation, segment reference, and
  content hashes.

A query probes version tables newest-to-oldest in batches and accepts only the
snapshot winner for a `NodeId`. Stale ANN or BM25 candidates are discarded and
the segment cursors widen until `k` live winners are proven or every source is
exhausted. Equal-LSN disagreement is corruption, not an arbitrary tie break.

Flush publishes node data, search deltas (or a proven-empty range), and the
next manifest in one CAS. Search compaction consolidates deltas or rebuilds a
base using external runs. A hard live-segment limit applies write
backpressure, so query read amplification cannot grow without bound.

## Memory hierarchy and accounting

The memory limit is a composition of explicit pools, not one optimistic cache
number:

```text
total process ceiling
  ├─ retained object/page/decoded caches
  │    ├─ RAM pages
  │    ├─ cache metadata and admission rules
  │    ├─ optional node views
  │    └─ optional adjacency pages/CSR
  ├─ concurrent search workspace
  ├─ Bolt frame/decode headroom
  ├─ active memtable and transaction
  ├─ one flush or compaction build chunk
  └─ runtime/allocator/native-library reserve
```

`NAMIDB_CACHE_MAX_BYTES` bounds retained caches, including their deep payload
and metadata. It does not mean RSS. `NAMIDB_MEMORY_MAX_BYTES` is the independent
RSS/working-set admission ceiling. The watchdog clears reconstructible caches
at pressure, rejects new work at the hard limit, and never treats `flush` as
memory-free: flush and compaction have their own serialization and workspace
gates.

The persistent range cache is keyed by namespace, immutable path, generation
token, and canonical page range. A reader performs one `HEAD`, pins ETag,
version, or create-only identity, and sends that precondition with every
bounded range request. RAM and NVMe eviction never affects correctness.

## Ten-million-node deployment envelope

Capacity depends more on embeddings, edge density, indexed text, and
concurrency than on node count alone. For planning, calculate durable data
before choosing compute:

```text
raw vector bytes = vector_count × dimensions × bytes_per_component
raw endpoint bytes = edge_count × 32
text/posting bytes = corpus-dependent; measure on a representative sample
object-store reserve >= live artifacts + compaction output + WAL + migration
local scratch >= largest compaction input/output working set
```

For ten million 1,024-dimensional `f32` vectors, raw vectors alone are about
40.96 GB decimal (38.15 GiB), before IDs, filters, routing, and checksums. Int8
payloads are about one quarter of that, with a recall/precision trade-off that
must pass the workload gate. The graph may be much smaller or much larger than
the vectors depending on edge density.

A practical initial single-query-node profile is:

- 8–16 vCPU;
- 16–32 GiB RAM with a deliberately smaller NamiDB process ceiling;
- 250–500 GB local NVMe for cache plus build/compaction scratch;
- R2/S3 capacity for at least two complete generations plus WAL and 30% safety
  margin;
- one or more stateless query replicas when concurrency, not corpus size,
  requires them.

This is an operating starting point, not a minimum encoded in the engine. A
low-concurrency deployment can use substantially less RAM by shrinking RAM
cache and search workspace, accepting colder latency. It still needs enough
memory for one query's selected vector/text/graph pages and result heap. A
high-QPS deployment buys RAM/NVMe to improve hit rate; object storage reduces
durable storage cost, not the laws of active working sets.

## Release gates

The object-native release is blocked until all of these pass:

1. **Correctness**
   - update/delete/relabel winners agree with an exact node scan;
   - filtered vector search returns `k` matches when `k` exist;
   - BM25 IDs and scores agree with an exact global-stat scorer;
   - graph forward/inverse traversal and exact edge properties agree with the
     mutation log;
   - object replacement, truncated ranges, bad checksums, and manifest/footer
     drift fail closed.
2. **Boundedness**
   - builders report and enforce peak logical workspace;
   - a high-degree vertex, common term, giant property, and selective filter do
     not allocate proportional to the corpus;
   - cache metadata falls after natural eviction;
   - 1M and 10M capacity runs remain below configured RSS and scratch limits.
3. **Object-store behavior**
   - one metadata pin per object reader;
   - no unplanned full-object GET on native paths;
   - at most 16 in-flight range requests per batch;
   - cold fetched-byte ratio and request count meet the checked benchmark
     thresholds;
   - multipart cancellation aborts incomplete uploads.
4. **Updates and maintenance**
   - correlated `MATCH`/`MERGE` uses batch point probes;
   - one compaction pass drains its captured L0 backlog;
   - expensive compaction runs outside the foreground writer lock;
   - live search-segment count is hard-bounded.
5. **Compatibility and release**
   - legacy V5/V3 and graph wires have exact fallback or a tested migration;
   - format version, changelog, Docker, Python wheel, SBOM, provenance, and
     immutable tag all refer to the same commit;
   - the release is published only after the complete CI and benchmark matrix.

[^tpuf-architecture]: <https://turbopuffer.com/docs/architecture>
[^tpuf-filtering]: <https://turbopuffer.com/blog/native-filtering>
[^tpuf-fts]: <https://turbopuffer.com/blog/fts-v2-postings>
[^tpuf-ann-v3]: <https://turbopuffer.com/blog/ann-v3>
[^spann]: <https://arxiv.org/abs/2111.08566>
[^spfresh]: <https://arxiv.org/abs/2410.14452>
[^pisa]: <https://pisa.readthedocs.io/en/stable/query_index.html>
[^graphar]: <https://graphar.apache.org/docs/specification/format/>
