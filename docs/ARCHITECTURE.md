# NamiDB: Architecture and Internals

**A technical report on the design and implementation of a cloud-native graph database.**

| | |
|---|---|
| **Version** | 2.0.0 |
| **Scope** | The complete engine as implemented on the `main` branch: storage, durability, compaction, query execution, vector and full-text search, graph algorithms, and the service layer. |
| **Audience** | Systems engineers, database researchers, and graduate students. This document is written to be cited and taught from; it explains mechanism, complexity, and correctness arguments rather than usage. |
| **Companion material** | The design-proposal record lives in [`docs/rfc/`](./rfc/) (RFCs 001–036); operational guidance in [`docs/multi-tenancy.md`](./multi-tenancy.md); user-facing documentation in the top-level [`README.md`](../README.md). |
| **How to read it** | Sections are self-contained and cross-reference each other. §1 states the thesis and the system model; §2–§4 are the storage substrate; §5 is the query engine; §6–§7 are the search and analytics layers; §8 is the service and concurrency layer; §9–§10 describe how the system was built and how it is evaluated. |

> **On accuracy.** Every mechanism below is grounded in the source as it exists at 2.0.0, with inline file references (e.g. `crates/namidb-storage/src/manifest.rs`). Where a design RFC and the shipped code diverge, this report documents the **code**. Constants, budgets, and format magic numbers are read out of the source, not assumed.

---

## Contents

1. Introduction and Design Philosophy
2. Storage Engine Core: Durability and the On-Object Format
3. Compaction, Space Reclamation, and Consistent Online Backup
4. The Read Path: Snapshot Isolation, Read-Your-Own-Writes, and Merge-on-Read
5. The Cypher/GQL Query Engine
6. Vector Search: the DiskANN/Vamana ANN Index
7. Full-Text Search and Graph-Algorithm Kernels
8. Interfaces, Service-Layer Concurrency, and Multi-Tenant Operation
9. Engineering Methodology
10. Evaluation and Benchmarking Methodology
11. References
- Appendix A: Glossary
- Appendix B: Selected Constants and Defaults

---

## 1. Introduction and Design Philosophy

NamiDB is a property-graph database with native vector and full-text search whose sole durable substrate is a commodity object store — Amazon S3 or any API-compatible service (Google Cloud Storage, Azure Blob, Cloudflare R2, Tigris, MinIO), and, for embedded use, the local filesystem. The engine is written in Rust and ships in a single codebase as an embeddable library (Rust and Python), a standalone HTTP/Bolt server, and a Model Context Protocol (MCP) server for language-model agents. There is no separate metadata service, no consensus cluster, and no external lock manager: the bucket is the database.

### 1.1 Thesis: coordination through object-store primitives

The central design decision is that **all coordination is expressed through the object store's own conditional-write primitives**, and nothing else. Two guarantees, offered by every targeted store, are sufficient to build a linearizable single-writer log-structured merge-tree (LSM-tree):

1. **Conditional create as compare-and-swap.** A `PUT-if-absent` (`PutMode::Create`; HTTP `If-None-Match: *`) either creates the object or fails because it already exists. Among concurrent writers of the same key, exactly one succeeds. This is the *only* conditional operation the commit path depends on (RFC-029), chosen for universal support — unlike conditional overwrite (`If-Match`), which stores implement unevenly.
2. **Read-after-write consistency on a named key.** A `GET`/`HEAD` of a specific key observes the last write to that key, even on stores whose bucket *listing* is only eventually consistent. Because every content-bearing object is written exactly once to a version- or UUID-named path, "read what was just written" never depends on `LIST` freshness; listing is used only to discover the highest version present, and its staleness is repaired by bounded forward probing (§2).

From these two primitives the engine derives a monotonic commit log (a Create-only pointer family), single-writer mutual exclusion (epoch fencing), snapshot-isolated reads, and safe concurrent space reclamation. This lineage — monotonic commit files plus a writer epoch — is shared with systems such as Delta Lake and SlateDB; NamiDB's contribution is to carry it through a *graph* data model with columnar adjacency, an in-graph approximate-nearest-neighbour index, and a full-text index, while preserving the invariant that an index answer is always identical to a brute-force scan of the same committed state.

### 1.2 Disaggregated storage and its consequences

Separating durability (the bucket) from compute (a stateless process holding caches and a write buffer) is what makes the system cloud-native: any number of *reader* processes can attach to a namespace and serve snapshot-isolated queries with no coordination, and a *writer* is a single ordinary process that can crash and be replaced without data loss because every acknowledged write is already durable in the bucket before the acknowledgement is sent. The price is latency: a commit costs object-store round-trips rather than a local `fsync`, and a cold read costs a `GET`. The architecture is therefore built around two mitigations that recur throughout this report — **pipelining** independent object writes so a commit costs `max(WAL, manifest) + pointer` round-trips rather than their sum (§2), and a **layered cache hierarchy** (byte-budgeted SST bodies, decoded row groups, node-view and adjacency caches) that turns a warm read into an `Arc::clone` (§4).

### 1.3 Design tenets

Five invariants run through every subsystem and are the throughline of this document:

- **Immutability of load-bearing objects.** Every WAL segment, manifest body, and SST is written once to a unique path and never mutated. The single exception is one advisory discovery hint (`current.json`), which is never trusted for correctness. Immutability is what makes read-after-write sufficient, makes caches never go stale, and makes a crashed writer's partial output reclaimable garbage rather than corruption.
- **Single writer per namespace, enforced without a lock.** At most one process may mutate a namespace at a time. This is not enforced by a lease or a lock server but by a monotonic **epoch** stamped into the manifest: a superseded writer loses its next compare-and-swap and *fences itself* (§2, §8). Serial writes make the correctness arguments tractable; readers never participate in the exclusion.
- **Snapshot isolation for readers, read-your-own-writes for the writer.** A reader observes a consistent, committed point-in-time view published as an immutable snapshot, taken with no lock (RFC-021); a writer additionally sees its own uncommitted, staged batch layered over that snapshot (RFC-026).
- **Index–scan equivalence (freshness).** An answer served from a secondary index (vector, full-text, or a property sidecar) must be *identical* to the answer a brute-force scan of the same committed state would produce. Two mechanisms enforce this: unioning the in-memory delta the index has not yet absorbed, or gating on a log-sequence-number comparison and falling back to the exact scan (§4, §6, §7). This tenet is what lets the system offer approximate-nearest-neighbour search without ever returning a stale or wrong row relative to the ground truth.
- **Determinism.** Graph-algorithm kernels, equality keys, and index builds are deterministic across runs — sums are accumulated in a fixed node order, communities are relabelled by first occurrence rather than hash-map order, and the equality key used for `DISTINCT`/`GROUP BY`/hash-join is a bit-exact, length-prefixed encoding. Determinism is a correctness property (it makes tests meaningful) and a usability property (repeatable analytics).

### 1.4 The data model in one paragraph

NamiDB stores a directed, labelled property multigraph. Nodes are **id-primary**: a node is one row keyed by a 128-bit identifier (UUIDv7), and its labels live *in the row* (a packed list column) rather than in the key, so a node with several labels is still a single row and a re-label is a single-row update. Edges are typed and stored twice — a forward adjacency keyed by source and an inverse keyed by destination — giving O(deg(v)) traversal in either direction. Properties are schema-optional: the engine enforces only the constraints and indexes a user declares, and any undeclared property is preserved losslessly in a per-row JSON overflow column. Queries are written in a Cypher/GQL subset; the same physical layout backs graph traversal, vector K-nearest-neighbour, and BM25 full-text retrieval.

### 1.5 System model and failure assumptions

The engine assumes: (i) the object store provides the two primitives of §1.1 and durably persists a successful `PUT`; (ii) processes are fail-stop (they may crash at any point, including between two object writes, but do not act maliciously or corrupt already-written objects); (iii) at most one writer per namespace is *intended*, but the engine remains correct if a second writer is accidentally started — the epoch mechanism fences the older one, and the newer one repairs any partial commit the crash left behind (§2). Object corruption in transit is caught by per-record and per-segment CRCs (WAL), content hashes (SST footers, WAL adoption), and format magic numbers; a body that fails to decode is treated as absent and the read falls back to the ground-truth scan rather than returning a wrong answer. The remainder of this report makes each of these mechanisms precise.

## 2. Storage Engine Core: Durability and the On-Object Format

NamiDB is an LSM-tree (log-structured merge-tree) whose source of truth is an object store (S3 and compatibles). All coordination is expressed through object-store conditional writes; there is no external consensus service, lock manager, or catalog. This section documents the durability path — write-ahead log (WAL), memtable, commit protocol, manifest — and the immutable columnar SST (sorted string table) formats those writes eventually materialise into, as implemented in `crates/namidb-storage`.

### Object-store consistency model (the foundation)

Every correctness argument below reduces to two object-store guarantees (RFC-001, RFC-029):

1. **Conditional create = compare-and-swap.** `PutMode::Create` (HTTP `If-None-Match: *`, PUT-if-absent) either creates the object or fails with `AlreadyExists`. Two processes racing to write the same key: exactly one wins. This is the *only* conditional primitive the commit path uses after RFC-029 — chosen because it is universally supported (S3, GCS, Azure Blob, R2, Tigris, MinIO, and `LocalFileSystem` via `O_CREAT|O_EXCL`), unlike `If-Match` conditional overwrite, which several stores implement unevenly.
2. **Read-after-write on a specific key.** GET/HEAD of a named key is read-after-write consistent everywhere targeted, even where *LIST* is only eventually consistent. Because every mutable-content object is written write-once to a UUID/version-named path, "read what was just written" never depends on LIST freshness. LIST is used only to find the *highest* version present, and its staleness is repaired by bounded forward HEAD probes (below).

The design writes nothing in place except one advisory hint (`current.json`); every load-bearing object is immutable once created.

### Namespace object layout and naming

`crates/namidb-storage/src/paths.rs` renders every key under `<root>/<namespace>/`. Manifest versions and WAL segments are **zero-padded 16-digit hex** so lexical LIST order equals numeric order; SST files are UUIDv7 (time-ordered). The canonical tree:

```
manifest/current.json              advisory pointer (read fallback only)
manifest/v<16hex>.json             immutable manifest body, one per version
manifest/pointer/p<16hex>.json     Create-only pointer family (RFC-029)
manifest/pins/<uuid>.json          retention pin leases
wal/<16hex>.wal                    immutable append-once segments
sst/level<L>/<uuidv7>-<kind>-<scope>.<ext>
memtable_snapshot.bin              optional cold-start checkpoint
```

### The write path — WAL segment format

`crates/namidb-storage/src/wal.rs` defines a WAL as a sequence of **immutable, append-once segment objects** at `wal/<seq>.wal`. There is no mutable open segment; the writer buffers records in memory and, at a commit boundary, seals the whole batch into one segment and PUTs it with `PutMode::Create`. Binary framing (little-endian, CRC32-IEEE via `crc32fast`):

```
Segment ::= Header Records+ Footer
Header  ::= "TGWL"(4) | version:u16 | reserved:u16 | record_count:u32 | first_lsn:u64
Record  ::= length:u32 | crc32:u32 | lsn:u64 | payload:[length]
Footer  ::= "TGEL"(4) | crc32(header+records):u32
```

`FORMAT_VERSION = 1` (the multi-label era). Framing is version-independent, so `decode` accepts any `version <= FORMAT_VERSION` and rejects only a strictly newer one (`Error::Corrupted`); an older binary refuses a v1 segment outright rather than silently dropping the label field. Each record carries a per-record CRC over `lsn ‖ payload`, and the footer carries a segment-wide CRC over header+records, so both torn writes and intra-record corruption are caught. `append_segment` maps `AlreadyExists` to `Error::Precondition` — a seq collision means a competitor (or an orphan from a prior attempt) already claimed that slot; the loser is fenced. `lsn` (log sequence number) is assigned monotonically per namespace and is the durable order of operations. Each `payload` is a bincode-encoded `WalEntry { key: MemKey, op: WalOp, lsn }` (`recovery.rs`).

### The memtable

`crates/namidb-storage/src/memtable.rs` buffers committed-but-unflushed writes. Its backing store is a **persistent (immutable) ordered map, `imbl::OrdMap<MemKey, MemEntry>`**, not a `std::collections::BTreeMap`. The reason is snapshot publication cost: readers consume the memtable through `Arc<MemtableSnapshot>` taken at each commit (RFC-021). With `OrdMap`, `snapshot_view()` is `self.inner.clone()` — **O(1)** structural sharing; two snapshots with no write between them share the same root (`ptr_eq`), and subsequent writes copy-on-write only the touched tree chunks. A `BTreeMap` would require an O(n) deep clone per commit, making per-commit cost grow linearly with memtable size (quadratic over a flush interval) while holding the writer lock. Values are `Bytes` (reference-counted), so a chunk copy never duplicates payloads.

Types:
- `MemKey` — an enum: `Node { id: NodeId }` (id-primary: labels ride in the value, so one id is one row regardless of label count) or `Edge { edge_type: String, src, dst }`. Variant order makes all nodes sort before all edges, and node keys sort by `id`, so a flush emits rows id-ascending "for free".
- `MemOp` — `Upsert(Bytes)` or `Tombstone`. Tombstones are retained (not deleted) until a compaction proves no older SST/WAL references the key; they override stale pre-delete values at read-merge time.
- `MemEntry { lsn: u64, op: MemOp }` — last write wins per key; `apply` maintains a running `bytes` estimate used by the flush trigger.

`freeze()` swaps the map out into a `FrozenMemtable` (leaving the live map empty so writes continue during flush); `restore()` folds a frozen batch back after a *failed* flush, newest-wins by LSN, so acked writes stay visible and get retried rather than being lost when the next successful flush clears their WAL refs.

### commit_batch — the end-to-end commit protocol

`WriterSession::commit_batch` (`crates/namidb-storage/src/ingest.rs`) is the durability boundary. Records are staged (WAL bytes in `pending`, memtable triples in `pending_payloads`) by `upsert_node/upsert_edge/tombstone_*`, which allocate LSNs via `alloc_lsn`. The commit:

```
1. Poison/empty guards. Build next manifest body (build_next):
   base.next_version(), append one WalSegmentDescriptor{seq, path,
   last_lsn, xxh3 = xxh3_64(pending.encode())}, carry label_dict.
2. Pipeline two INDEPENDENT writes with tokio::join!:
      a. wal_store.append_segment(pending)     (PutMode::Create)
      b. manifest_store.put_body(fence, base, next)  (PutMode::Create)
   The WAL path is fully determined by seq, so both can fly at once —
   turning WAL + body + pointer (3 RTs) into max(WAL,body) + pointer (2).
3. On WAL Ok: pointer = body_result?; cas_pointer(...) publishes p<v+1>.
4. Only AFTER the pointer CAS lands: drain pending_payloads into the live
   memtable, advance current/seq, refresh_published() (new Arc snapshot).
```

The ordering — **WAL durable and manifest committed before the memtable is touched and before the client is acked** — is the core invariant: a failed commit never ACKs a record it did not make durable and reference, and the memtable is untouched until the pointer CAS succeeds, so recovery never resurrects an unacked write nor drops an acked one.

Failure handling is precise. A WAL `Precondition` (seq collision with an orphan) triggers a single **body-first** retry (`commit_body_first`) at a freshly listed seq: PUT body → PUT WAL → CAS pointer, so the common "body `base+1` already taken" case fails fast as `ManifestCommitCas` without minting a second orphan WAL. A terminal retry failure sets `poisoned = true` — the single-writer contract is to drop the session and reopen. `put_body` tolerates `AlreadyExists` when the durable body is provably *ours* (`existing_body_is_ours`: byte-identical, or equal modulo the audit-only `created_at`, which the embedded `writer_id` makes unforgeable); otherwise it reloads and either returns `Fenced` (epoch advanced) or `ManifestCommitCas`.

### The manifest and the versioned-pointer CAS protocol (RFC-029)

`crates/namidb-storage/src/manifest.rs`. A `Manifest` is a self-describing snapshot: `version`, `epoch`, `writer_id`, `schema`, `ssts: Vec<SstDescriptor>`, `wal_segments: Vec<WalSegmentDescriptor>`, `label_dict`, and registered `vector_indexes`/`text_indexes`. It is written write-once to `manifest/v<N>.json` (`put_body`, `PutMode::Create`).

The linearization point is the **Create-only pointer family** `manifest/pointer/p<N>.json`, where `N` equals the manifest version and the *current pointer is the highest `N` present*. `cas_pointer` creates `p<v+1>.json` with `PutMode::Create`: whoever creates it first owns version `v+1`. Both commit phases now use the same primitive (PUT-if-absent), removing the engine's former dependency on `If-Match` overwrite. This matches the Delta Lake / SlateDB model (monotonic commit files + LIST for current + writer epoch).

`load_current` → `load_pointer`: `max_pointer_version` LISTs `pointer/`, takes the max `N`, then `probe_pointer_forward` galloping HEADs `p<N+1>, p<N+2>, …` (bounded `MAX_PROBE = 8192`) to close the window where a just-created pointer has not yet appeared in an eventually-consistent LIST. If LIST is empty, it falls back to a HEAD of `p0` (fresh namespace) and then to reading the advisory `current.json` (aged namespace whose low pointers the janitor reclaimed) — trusting the advisory only if the pointer at its version still HEADs. If the probe exhausts its window it returns the *retryable* `PointerResolveStale`. The advisory `current.json` is a plain unconditional PUT written on every commit/bootstrap; it is explicitly **not** part of the CAS contract, only a non-LIST discovery hint that prevents `WriterSession::open` from re-bootstrapping (and thereby erasing) a live namespace on an EC-LIST store.

Monotonicity invariants enforced in `put_body`: `new.version == base.version + 1`, `new.epoch >= base.epoch`, and `fence.assert_alive(base.epoch)` before any write.

### Single-writer enforcement — epoch fencing and claim_writer

There is no lock. Single-writer is enforced by a monotonic `Epoch(u64)` bumped in the manifest (`crates/namidb-storage/src/fence.rs`). At `open`, if the namespace exists, `claim_writer` commits a zero-op manifest whose `epoch = base.epoch.next()`, minting a `WriterFence { epoch, writer_id: Uuid::now_v7() }`. Every subsequent `put_body`/`cas_pointer`/`flush` calls `fence.assert_alive(current_epoch)`, which returns `Error::Fenced { mine, current }` iff `current > mine`. A superseded ("zombie") writer therefore loses its next CAS (the pointer moved and the epoch is higher) and fences itself. `claim_writer` is a bounded loop: it retries while the pointer version advances (genuine live race, resolves in a couple of rounds), but a stall at a fixed version (`MAX_STALLED_ROUNDS = 8` consecutive CAS losses with no advance) is the signature of an orphan manifest body, triggering repair (up to `MAX_REPAIR_PASSES = 4`) or a terminal `OrphanManifestBody`.

### Stalled-commit repair

An orphan is a `v<N+1>.json` body whose pointer `p<N+1>` was never created (writer crashed between the two PUTs). Because versions are Create-only, nobody can supersede it, so the namespace would wedge forever. `repair_stalled_commit` decides adopt-vs-delete via `orphan_ready_to_publish`:

- **Adopt** (publish the exact pointer bytes the interrupted writer would have written — deterministic content, so it is safe even if that writer is still alive: its own `cas_pointer` merely observes `AlreadyExists`) **iff** the body parses as a manifest extending the lineage (`version == N+1`, `epoch >= base.epoch`), *every* SST it adds over the base still HEADs durable, and *every* new WAL segment it references is durable with the declared `last_lsn` **and** a matching content hash.
- **Delete** otherwise, discarding only never-acked records.

The content check is the subtle part (`wal_slot_matches`). An lsn-range match alone is insufficient: WAL slots are Create-once, so a **fenced peer's** segment could occupy the same slot with a coincident lsn range; adopting it would durably commit writes whose client saw the commit fail. So adoption additionally requires `xxh3_64(segment_bytes) == descriptor.xxh3` — proving the durable object is *this* commit's own segment. A descriptor lacking a hash (pre-hash manifest) is unverifiable and is refused (delete path). If the slot is *empty*, the repairer first **fences** it by Create-ing an empty sentinel segment (`last_lsn = 0`, which no real commit declares): winning that create guarantees the interrupted commit's own WAL PUT can never later succeed, so its `cas_pointer` (which runs only after a WAL success) can never fire — making deletion safe against a concurrent live writer.

### SST kinds and columnar layout (RFC-002)

`SstKind` = `Nodes | EdgesFwd | EdgesInv | VectorGraph | TextIndex`. `flush.rs` builds every body in RAM, PUTs them concurrently (`try_join_all`, each to a fresh UUIDv7 path), then commits one manifest that appends the new `SstDescriptor`s and `clears wal_segments` (their records are now durable in SSTs). Each descriptor carries embedded stats (`min_key/max_key` base64, `min_lsn/max_lsn`, `row_count`, `KindSpecificStats`, per-label/property stats) so the read path prunes candidates with zero extra GETs, plus optional side-car pointers (bloom, unique/equality/label indices).

- **Nodes** (`sst/nodes.rs`, Parquet): id-primary, one identity-partitioned SST spanning all labels. Fixed Arrow schema, column order `node_id: FixedSizeBinary(16)`, `tombstone: Boolean`, `lsn: UInt64`, `__labels: List<UInt32>` (packed `LabelId`s), `prop_<p>` (declared properties; in production none — everything rides in overflow), `__overflow_json: Utf8` (always present; undeclared properties as JSON, never dropped), `__schema_version: UInt64`. Rows sorted by `node_id` ascending. Defaults: ZSTD level 6, row-group target 128 Ki rows (`NAMIDB_NODE_SST_ROW_GROUP_ROWS`), 1 MiB data pages, 8192-row write batches, page index + page checksums on. Tombstones are rows with `tombstone = true` and null properties.
- **Edges** (`sst/edges/`, native CSR): two files per `(edge_type, level)` — forward (`edges-fwd`, keyed by `src_id`) and inverse (`edges-inv`, keyed by `dst_id`) — sharing one wire format distinguished by `FLAG_INVERSE_PARTNER`, giving O(deg(v)) expansion in both directions. 64-byte frozen header (`HEADER_MAGIC = "TGEDGE\0\0"`, `format_major=1`, `format_minor=0`, `flags`, and `blake3(edge_type/src_label/dst_label)[..16]`). Independently addressable sections: `key_ids` (0x0001), `offsets` (0x0002, bitpacked u24/u32/u40/u48), `partners` (0x0003, per-group split-top64/bottom64 `TAG_SPLIT=0x01` or dense `TAG_DENSE=0x10`), `per_edge_lsn` (0x0004), `per_edge_tombstones` (0x0005, omitted when none), `fence_index` (0x0006, emitted when `key_count > FENCE_INDEX_THRESHOLD = 65 536`, stride `DEFAULT_FENCE_STRIDE = 256`), and per-property streams (0x0100, Zstd Arrow IPC). Footer: section table + body fields + a fixed 20-byte trailer (`FOOTER_TRAILER_LEN`) = xxh3-64(body) ‖ footer_len:u32 ‖ `FOOTER_MAGIC = "TGEDGE\xFE\xEF"`.
- **Bloom side-cars** (`sst/bloom.rs`): split-block bloom filter (SBBF) over the key. `MAGIC = "TGBLOOM\0"`, 28-byte header, 32-byte (256-bit) blocks, 8-byte xxh3 trailer, `DEFAULT_BITS_PER_KEY = 10` (~1% FPR). Omitted when `size_bytes < BLOOM_OMIT_THRESHOLD_BYTES = 256 KiB` (scanning beats probing).
- **VectorGraph** (`.vg`, `sst/vector.rs`, `vector-index` feature): 8-byte magic `"NAMIVG03"` + bincode `VectorGraphBody` (DiskANN/Vamana graph + f32 or int8-quantized vectors). Self-contained; a reader that mismatches the magic falls back to flat scan.
- **TextIndex** (`.ft`, `sst/text.rs`, `text-index` feature): 8-byte magic `"NAMIFT02"` + bincode `TextIndexBody` (BM25 inverted index: term → postings of (doc, tf, positions), plus corpus stats).

### Recovery on open

`recover_memtable_with_snapshot` (`recovery.rs`) rebuilds the live memtable from the manifest's `wal_segments`. Phase 0 optionally seeds from `memtable_snapshot.bin` (a bincode `MemtableSnapshotFile { version, last_lsn, entries }`). A snapshot is **ignored when stale**: if `snap.last_lsn <= flushed_hwm` (the max `max_lsn` over `manifest.ssts`), the snapshot is subsumed by flushed SSTs and trusting it would resurrect rows later deleted-and-GC'd — so recovery rebuilds from SSTs + WAL instead. Otherwise the snapshot seeds the map and its `last_lsn` becomes a `snapshot_floor` below which WAL records (and whole segments) are skipped. WAL segments are replayed in seq order; each segment's actual `last_lsn` is checked against the descriptor (mismatch → `Error::Corrupted`, refusing both writer-raced-manifest and truncated-body cases), and each `WalEntry.lsn` against its `WalRecord.lsn`. `next_lsn` is derived as `max(recovered.max_lsn, max_sst_lsn) + 1` (at least 1) — folding in the SST high-water so a cold-reopened all-flushed namespace does not restart at 1 and shadow new writes under their own older SST rows. `next_wal_seq` is `max(listed segment seq) + 1` over *every* segment visible via LIST (not just manifest-referenced ones), so orphan segments are never re-used under `PutMode::Create`.

## 3. Compaction, Space Reclamation, and Consistent Online Backup

NamiDB is a log-structured merge (LSM) engine whose sorted string tables (SSTs) are immutable Arrow/Parquet objects in object storage, indexed by an append-only **manifest** (a versioned JSON object committed by a compare-and-swap pointer protocol under epoch fencing). Every flush of the writer's memtable appends one L0 SST per `(kind, scope)` **bucket** that had rows, where `kind ∈ {Nodes, EdgesFwd, EdgesInv, VectorGraph, TextIndex}` and `scope` is the label (nodes) or edge type (edges). Without reorganisation, L0 grows monotonically and every point lookup pays an O(L0-count) candidate scan. This subsystem bounds read, write, and space amplification; the mechanisms live in `crates/namidb-storage/src/{compact.rs, janitor.rs, pin.rs, backup.rs}` and are driven by the maintenance loops in `crates/namidb-server/src/{lib.rs, registry.rs}`. The design is RFC-027 (pieces P1–P5 landed as "leveled-lite").

### Leveled-lite compaction

The compactor keeps **one SST per `(kind, scope, level)`** across levels L1..Lk with a per-level byte budget `budget(Li) = base · ratio^(i−1)` (`level_budget_bytes`, saturating). Constants (`compact.rs`): `DEFAULT_COMPACTION_BASE_BYTES = 8 MiB` (`NAMIDB_COMPACTION_BASE_BYTES`) and `DEFAULT_COMPACTION_LEVEL_RATIO = 10` (`NAMIDB_COMPACTION_LEVEL_RATIO`, floored at 2). So L1 ≈ 8 MiB, L2 ≈ 80 MiB, L3 ≈ 800 MiB, forming a 10× cascade.

`plan_bucket_merge(sources, base, ratio) -> Option<BucketPlan>` decides one bucket's merge. `BucketPlan{ inputs, target_level, is_deepest }`:

```
partition sources into l0 (level 0) and leveled[level] (levels ≥ 1)
deepest_present = max level in leveled (0 if none)
inputs = l0;  cum = Σ size(l0);  target = 1
loop:
    if leveled[target] exists: push it into inputs, add its bytes to cum
    if cum ≤ budget(target): break            # fits — stop here
    if target < deepest_present: target += 1; continue   # cascade deeper
    if deepest_present ≥ 1: target += 1        # spill into one fresh deeper level
    break
if inputs.len() < 2: return None               # nothing worth rewriting
is_deepest = target ≥ deepest_present
```

L0s always drain into L1. A merge **cascades** into the next deeper occupied level only when the accumulated bytes exceed a level's budget, so large base levels are rewritten rarely — this is what bounds write amplification while read amplification stays ≈ the number of levels. A brand-new bucket lands its first SST in L1 and cascades only on a later sweep once a shallow level overflows. The output at `target_level` is the bucket's **deepest occupied level** iff `is_deepest`, which gates all garbage collection below. `needs_compaction()` (`CompactionBasis`) is a metadata-only mirror (`any_bucket_plans`) that runs `plan_bucket_merge` over the manifest with no object-store I/O, so an idle maintenance tick skips the expensive phase; only `Nodes`, `EdgesFwd`, `EdgesInv` buckets are planned (vector/text SSTs are rebuilt, not merged).

### Tombstone GC and index-rebuild authority

A **tombstone** (a delete marker row) or a superseded version may be physically dropped only when no un-merged deeper level can still hold a live row it shadows. The rule (`prepare_leveled`) is:

```
gc = node_gc_safe AND plan.is_deepest
```

Both conditions are required. `plan.is_deepest` invokes the LSM invariant: a shallower level always holds the newer LSN (log sequence number) for a key, so any copy in a deeper level is older and loses at read time; dropping a **deepest-level** tombstone can never resurrect a row. `node_gc_safe = (#distinct node scopes ≤ 1)` handles the **mixed-scope truncation hazard**: nodes are id-primary, so a node key can live in any node SST regardless of scope. If a legacy per-label node SST coexists with the id-primary `""` scope, a single bucket's deepest-level merge is *not* authoritative for the whole key space, and dropping a tombstone could resurrect a live row from the other scope. Node GC is therefore restricted to the single-scope case. Edges are keyed within `(edge_type, direction)`, so an edge bucket is self-authoritative and only `is_deepest` applies (`gc_tombstones` passed to `merge_edge_sources`). A reader pinned at an older manifest version still observes the delete through the retained source bodies, never through the new SST.

The same `gc` predicate gates **index rebuilds**. A Vamana vector graph or BM25 text index is not row-mergeable, so compaction rebuilds it from the merged corpus. The `NodeMergeIndexSpecs` are populated (from `manifest.vector_indexes` / `text_indexes`) only when `gc` holds; on a partial merge the member lists stay empty, the existing `.vg`/`.ft` is left untouched, and the freshness gate `Snapshot::index_outrun_by_nodes` routes reads to the exact flat scan until an authoritative merge rebuilds. Treating a per-bucket deepest merge in a mixed-scope namespace as corpus-complete would permanently truncate the index — the identical hazard, resolved by the identical rule.

### Prepare / commit split

A sweep is split so the single writer lock is held only for the cheap manifest CAS. `WriterSession::compaction_basis()` (`ingest.rs`) clones a `CompactionBasis{ manifest_store, fence, base }` under the lock (two Arc-backed clones plus a manifest clone). `CompactionBasis::prepare` → `prepare_leveled` then runs **off-lock**: it plans every bucket, issues every input GET (bodies are held only as compressed bytes; their sum is bounded by the level budget), performs the k-way merges and the vector/text index rebuilds on the blocking pool (`run_cpu` via `spawn_blocking`), and PUTs every output body, bloom, and sidecar at immutable UUID-v7-derived paths (`{uuid.simple}-{tag}-{scope}.{parquet|csr|vg|ft}`, `put_create` with `PutMode::Create`). The result `PreparedCompaction{ new_descs, removed_ids, bloom_count, base_version }` references no manifest.

`install_prepared` re-takes the lock only to fold the plan into the manifest **current at commit time** (which may have advanced) and run the fence-checked CAS. It asserts the epoch is still alive, then verifies **every** `removed_id` is still referenced by `current` (writes and flushes only ADD SSTs, so a missing input means another compaction already merged it away); a missing input aborts with `Error::Precondition`, leaving the manifest untouched. Otherwise it builds `next = current.next_version(writer_id)`, retains SSTs not in `removed_set`, extends with `new_descs`, and commits. A flush that interleaved during the prepare simply contributed new L0 SSTs that survive into `next` and merge on a later sweep — and any index rebuilt by this prepare is older than that L0, so the LSN freshness gate already routes those reads to the flat scan. An **abandoned prepare is safe by construction**: its outputs are unreferenced UUID-named garbage that the orphan sweep reclaims once past `min_age`.

### The k-way streaming merge

`merge_node_sources` / `merge_edge_sources` never materialise the whole decoded bucket. Each source gets a cursor (`NodeSourceCursor` decodes one Parquet row group at a time — the writer keeps `node_id` strictly ascending, so cursor order is key order; `EdgeSourceCursor` walks one partner block plus one IPC mini-batch per property stream, decoding zstd through a streaming `Read`). A `BinaryHeap<Reverse<HeapEntry>>` is keyed **(key asc, lsn desc, source order)** — for nodes `key = node_id`, for edges `key = (key_id, partner_id)`. Popping yields, per key, the highest-LSN observation first (its winner), exact ties broken toward the earlier source, reproducing the stable `sort_by(key, lsn desc)` a materialised merge would produce. The first pop of a key is the winner: it is materialised (for nodes, the JSON property-map re-encode, typically 3–10× the Parquet size, is paid **only** for winners) and pushed into an `IncrementalNodeSstWriter` in bounded chunks of `NAMIDB_COMPACTION_MERGE_CHUNK_ROWS` rows (default `NODE_SST_BATCH_ROWS = 16·1024 = 16384`). Shadowed duplicates and, when `gc`, winning tombstones are skipped without ever being converted (`cursor.advance()` / `pop(false)` still steps the property streams in lockstep). Complexity is O(N log k) for N total input rows across k sources. Streaming collectors observe the winner stream in one pass: `UniqueSidecarCollector`, `EqualitySidecarCollector`, `LabelIndexCollector`, `PerLabelStatsCollector` (RFC-025), and the `VectorMemberCollector`/`TextMemberCollector` member harvesters. **Residual memory per bucket** (by design): the compressed source bodies (Σ ≤ level budget), one decoded row group per node source, one winner chunk, the sidecar maps, and — the true lower bound — the embeddings/documents a vector/text rebuild collects, which the Vamana/BM25 builders inherently need in full.

### Space reclamation: the horizon-aware sweep

`sweep_orphans(manifest_store, retention_horizon, min_age, max_level, delete)` (`janitor.rs`) deletes only objects reachable by no live reader. The **retention horizon** is the oldest manifest version any live reader is pinned to: `SnapshotCell::retention_horizon()` returns `min(SnapshotRegistry::min_live(), current)`, where the registry is a `BTreeMap<version, refcount>` maintained by `PinnedSnapshot` acquire/drop (`read.rs`); it is unioned in the sweep with the pin-lease floor and clamped to `current_version`. The sweep then builds the **live set** as the union of objects referenced by every retained manifest version from `horizon` to `current` inclusive — SST bodies, bloom / unique / equality / label-index sidecars (all under the same `sst/level{N}/` prefix), and WAL seqs. It scans `sst/level0..scan_max_level` (where `scan_max_level = max(max_level, deepest level any retained manifest occupies)` — a hardcoded floor of 1 previously leaked every L2+ rewrite forever), and any listed object not in the live set and older than `min_age` is an orphan. It also reclaims `manifest/v{N}.json` and `manifest/pointer/p{N}.json` strictly below the horizon (RFC-029), dead WAL segments referenced by no retained version, and a stale `memtable_snapshot.bin` once `last_lsn ≤ horizon_flushed_hwm`. Safety is **by construction, not wall-clock**: an object the sweep deletes is referenced by no version at or above the horizon, so no reader can reach it; `min_age` (default 24 h) is only a secondary guard for the body-PUT-then-CAS race where a just-written object is not yet referenced by any committed version. The interaction with compaction is exact: compaction removes source descriptors from the manifest but leaves the bodies; a reader pinned at the pre-compaction version holds the horizon down, so those bodies stay live until it drops, then become reclaimable. The server drives this at `compaction_interval` (default 300 s), after compaction, with `sweep_delete` default `true`; the reactive trigger `compaction_l0_trigger` (default 8) compacts on flush when `max_l0_bucket_len() ≥ 8`, and a soft write stall (`write_stall_l0` default 24, `write_stall_delay` 50 ms) throttles the writer above three times the trigger.

### Backup pins and streaming online copy

In-process reader tracking cannot see a cross-process reader such as a running backup, so `pin.rs` adds a durable lease. `RetentionPin::acquire` writes `manifest/pins/<uuid>.json` = `PinLease{ version, expires_at_unix }` with `DEFAULT_PIN_TTL = 15 min`. `sweep_orphans` lists `manifest/pins/` **before** classifying anything, lowers its horizon to every unexpired lease's `version` (`pin_floor`), and deletes expired leases (so a crashed holder cannot pin forever). `renew_if_due` rewrites the lease once at least **half the TTL** has elapsed (`renew_due_at = now + ttl/2`) — cheap enough to call once per copied object. A **pre-delete pin re-check** (`current_pin_floor`) reloads the leases just before any deletion; a lease that landed mid-sweep and pins below the horizon aborts the pass (`aborted_by_pin`), and the holder's own post-acquire root re-check closes the residual window.

`copy_namespace_snapshot` (`backup.rs`) is the single primitive behind backup and restore. Because every referenced object is immutable (compaction/flush only add; the sweep only deletes unreferenced objects), copying the closure a pinned manifest names is **consistent by construction**. It loads the manifest, `RetentionPin::acquire`s at `source_version`, then re-HEADs the manifest body *after* the lease is visible (closing the load-then-pin race — a `NotFound` fails loudly rather than truncating). It copies SST bodies and their sidecars, then WAL segments, renewing the pin per object; writes the manifest body renumbered to a self-contained **version 0 / `Epoch::ZERO`**; and writes the pointer **last** so an interrupted copy leaves an un-pointed, ignored object set. `memtable_snapshot.bin` is deliberately skipped (not manifest-referenced; a stale copy would silence WAL replay). Each object streams through `copy_object_with_part_size`: ≤ `COPY_PART_SIZE = 8 MiB` takes a buffered PUT, larger objects a multipart upload capped at `COPY_MAX_IN_FLIGHT_PARTS = 4` in-flight parts (bounding per-object memory ≈ 4 × 8 MiB; 8 MiB clears S3's 5 MiB minimum non-final part and, with 10k parts, bounds one object at 80 GB), so a multi-GB compacted SST cannot OOM the process.

## 4. The Read Path: Snapshot Isolation, Read-Your-Own-Writes, and Merge-on-Read

NamiDB is a single-writer-per-namespace LSM engine, so the read path never sees a partially-applied write: a reader resolves a *logical* node or edge by merging one in-memory delta against the immutable Sorted String Tables (SSTs, columnar Parquet/CSR files) named by a manifest version, applying last-LSN-wins (highest log-sequence-number wins) with tombstones. This section describes how a reader obtains a consistent committed view without taking the writer lock, how a writer sees its own uncommitted work, and the process-wide cache tiers that make warm reads an `Arc::clone`.

### Snapshot isolation without the writer lock (RFC-021)

The engine separates *mutable writer state* from an *immutable published read view*. The writer owns a `Memtable`, whose payload map is a persistent `imbl::OrdMap<MemKey, MemEntry>` (`crates/namidb-storage/src/memtable.rs`). Because `OrdMap` is a copy-on-write balanced tree, `Memtable::snapshot_view()` clones only the root pointer: publication is **O(1)** with structural sharing, and later writes copy only the touched tree chunks (values are `Bytes`, refcounted, so a chunk copy never duplicates payloads). The unit test `snapshot_view_is_structural_sharing_not_a_clone` asserts `a.inner.ptr_eq(&b.inner)` for two no-write views. This is the mechanism that lets the writer republish on *every* commit without the O(memtable) tree clone that would otherwise hold the writer lock for a time linear in memtable size.

After each successful `commit_batch`/`flush`, `WriterSession::refresh_published` rebuilds `published_memtable: Arc<MemtableSnapshot>` (`crates/namidb-storage/src/ingest.rs`). `owned_snapshot()` packages `{manifest, Arc<MemtableSnapshot>, store, paths, caches}` into an `OwnedSnapshot` (`read.rs:3503`) with no lifetime parameter. Readers pick it up through a `SnapshotCell` (`read.rs:3668`): a `std::sync::Mutex<Arc<OwnedSnapshot>>` whose `load()` critical section is one pointer copy plus an `Arc` strong-count bump (tens of ns). A slow or in-flight write is invisible: `store()` republishes only after the manifest CAS lands, so readers keep serving the previous committed version — exactly the snapshot isolation the manifest CAS already guarantees. Each read repackages the owned state into a short-lived borrowed `Snapshot<'mt>` via `OwnedSnapshot::borrow()`, so there is one read engine, not two; the per-query scratch caches (`node_cache`, `decoded_node_row_groups`) live on that temporary borrow and drop at query end.

`SnapshotCell::load()` also registers the snapshot's manifest version in a `SnapshotRegistry` (a `Mutex<BTreeMap<u64, usize>>` of version→live-reader-count) and returns a `PinnedSnapshot` whose `Drop` decrements it (RFC-027). `retention_horizon()` returns `min(registry.min_live(), current_version)` — the oldest version any live reader can still reach. **This is the load-bearing interaction between compaction and the read snapshot**: the janitor/compactor sweep (`crates/namidb-storage/src/janitor.rs:262`) clamps its reclamation floor to this horizon, so an object a 60-second-old reader still needs is never GC'd, while a version below the horizon with no readers becomes collectable. Registration happens while the cell lock is held, so version selection and pinning are atomic and the horizon can never exclude a version a reader is about to read.

### Merge-on-read: reconstructing a logical row

A read materialises `NodeView { id, labels: BTreeSet<String>, properties: BTreeMap<String,Value>, lsn, schema_version }` or `EdgeView { edge_type, src, dst, properties, lsn }` by aggregating candidate versions across the memtable and every relevant SST into a per-key winner map keyed on the raw 16-byte id, keeping the highest LSN and discarding a winner that is a tombstone (`update_node_winner`/`update_partner_winner`, `read.rs`). Nodes are **id-primary**: `MemKey::Node{id}` carries no label (the label set rides in the value), so a node is one row regardless of label count, and flush output stays id-ascending for free.

Point lookup `lookup_node_by_id` (`read.rs:1440`) probes the memtable, then the node SSTs whose `[min_key, max_key]` straddles the target (`Manifest::node_candidates`), each gated by a bloom side-car (`bloom_admits`; omitted for bodies under `BLOOM_OMIT_THRESHOLD_BYTES = 256 KiB`). `scan_label_with_predicates_and_projection` (`read.rs:1639`) materialises every node across the label-agnostic memtable plus all node SSTs into a `BTreeMap<NodeId, (lsn, Option<NodeView>)>`, then filters `labels.contains(label)` at the end — a failing predicate produces `None` at that LSN so a lower-LSN SST row is not spuriously surfaced. In contrast, `scan_all_node_ids` (`read.rs:1809`) is a single label-agnostic pass that decodes only id/tombstone/lsn (projection `Some(&[])`), giving the whole-graph node set for `CALL algo.*` in O(nodes) rather than O(labels × nodes). Edges merge forward SSTs only (`SstKind::EdgesFwd`); the inverse partner duplicates the same `(src,dst,lsn)` tuples, so one direction keeps the merge unambiguous (`scan_edge_type`, `count_edge_type`).

### Read-your-own-writes overlay (RFC-026)

A writer's staged-but-uncommitted batch lives in `pending_payloads: Vec<(MemKey, u64, MemOp)>`, absent from `published_memtable` by design. `WriterSession::overlay_snapshot()` (`ingest.rs:631`) replays that batch (LSN-ascending) into a second `Memtable`, freezes it, and attaches it as `Snapshot::overlay: Option<MemtableSnapshot>`. The read paths chain the two sources: `node_entries()` returns `memtable.iter_nodes().chain(overlay…)`, and `node_mem_entry(id)` compares the two LSNs directly. Correctness rests on one invariant: **staged LSNs are strictly greater than any committed LSN** (the writer seeds `next_lsn` past every committed LSN on open), so the existing last-LSN-wins merge resolves a staged upsert over the committed row and a staged tombstone hides it — no separate read engine. The code guards this with `debug_assert!(s.lsn > c.lsn)` yet still takes `s.lsn >= c.lsn`, degrading gracefully to last-LSN-wins if allocation ever regressed. The overlay covers nodes and edges (`edge_mem_entries` feeds the SST and CSR edge paths and `sorted_partners`). Crucially, `overlay_snapshot()` attaches the immutable body/adjacency caches but **not** the cross-snapshot NodeView or property-index caches: those are keyed by manifest version and shared across sessions, so caching a staged row would leak uncommitted data to a concurrent reader pinned at the same version.

### Ranged reads and pushdown (RFC-003/011/013/015)

`RangedMode` (`read.rs:229`) is `Auto` or `Force(bool)`; `Auto` issues a full-body GET when `desc.size_bytes < ranged_threshold_bytes` (default `DEFAULT_RANGED_THRESHOLD_BYTES = 16 MiB`) and a ranged GET otherwise — full-body wins when RTT dominates transfer (small SSTs), ranged wins when transfer dominates (large SSTs). A cold ranged lookup fetches footer + Parquet page index + only the column pages of the straddling row group (~50–100 KB / 3–4 round trips vs a whole-body GET). Row-group pruning (`row_groups_for_keys`, `sst/nodes.rs:714`) exploits the writer's strict-ascending `node_id` (default row group = `128 Ki` rows): for a sorted probe set it uses `partition_point` against each group's `min_bytes/max_bytes` stats to admit only groups that can contain a key. Predicate pushdown (`scan_with_predicates_and_projection`, `sst/nodes.rs:474`) synthesises per-column stats and skips any row group where `eval_row_group == Absent`; surviving rows still get a per-row three-valued-logic recheck because pruning is conservative. Projection pushdown builds a `ProjectionMask::leaves` over the requested property columns plus the six engine leaves (`node_id`, `tombstone`, `lsn`, `__schema_version`, `__overflow_json`, `__labels`); an **empty projection** (`Some(&[])`) reads engine columns only and additionally skips the per-row `serde_json` overflow parse — the id-only fast path.

### The cache hierarchy and its memory model

Four process-wide singletons, each byte-budgeted, keyed to be namespace-safe:

| Cache | Tier / structure | Default budget | Eviction |
|---|---|---|---|
| `SstCache.inner` (`cache.rs`) | foyer `Cache<String,Bytes>` body cache, weight `key.len()+value.len()` | 256 MiB (`NAMIDB_SST_CACHE_BUDGET_MIB`) | S3-FIFO |
| `SstCache.decoded_node_row_groups` | foyer `Cache<(String,usize),Arc<Vec<RecordBatch>>>`, weight = key + Arrow `get_array_memory_size` | 256 MiB (`NAMIDB_DECODED_NODE_RG_..`) | S3-FIFO |
| `SstCache` side maps | `Arc<Mutex<HashMap<String,…>>>` for parsed `ParquetMetaData`, `EdgeStreamBundle`, `EdgeSstReader`, decoded `.ft`/`.vg` indexes | unbounded | pruned by `retain_paths` |
| `NodeViewCache` (RFC-019) | `HashMap<NodeCacheKey,(CachedNodeView,seq)>` + `BTreeMap<(version,seq),key>` order index | 256 MiB (`NAMIDB_NODE_CACHE_..`) | oldest manifest version, O(log n) |
| `AdjacencyCache` (RFC-018) | `Mutex<HashMap<AdjacencyKey,Arc<EdgeAdjacency>>>` | 512 MiB (`NAMIDB_ADJACENCY_..`) | lowest manifest version |

`lookup_node` is a 3-tier lookup: **L1** per-snapshot `Mutex<HashMap<(String,NodeId),Option<NodeView>>>`; **L2** the shared `NodeViewCache` (promoted into L1 on hit); **L3** the SST walk (inserted into both). The `NodeViewCache` caches negatives (`None` = absent/tombstoned) because the key embeds `manifest_version` — a committed write mints a fresh slot at the next version. Its eviction is O(log n): `Inner.order` is a `BTreeMap` whose `pop_first` yields the (oldest version, oldest insertion) victim without a full-map scan, and an overwrite removes the stale order entry by its stored `seq`. The `AdjacencyCache` materialises, per `(edge_type, direction)`, a CSR of five parallel arrays — `keys: Vec<NodeId>`, `offsets: Vec<u32>`, `partners: Vec<NodeId>`, `lsns: Vec<u64>`, `tombstones: Vec<bool>` — where `partners[offsets[i]..offsets[i+1]]` are key i's edges; `lookup` is O(log K) binary search + O(deg). `build_adjacency` folds every scope SST through a `BTreeMap<([u8;16],[u8;16]),(u64,bool)>` (last-LSN-wins), costing O(E log E) once per version; `get_or_build` runs the build outside the lock and discards the loser of a build race.

**Multi-tenant sharing (docs/multi-tenancy.md).** One host serves N namespaces under one set of budgets. The `SstCache` is namespace-safe *by construction* — every key is an absolute object-store path (namespace-prefixed) or `(path, row-group)`. The `NodeViewCache` and `AdjacencyCache` are **not**: the bare `(manifest_version, label/edge_type, …)` triple collides across tenants because per-namespace manifest versions both start at 1. `NodeCacheKey` and `AdjacencyKey` therefore embed `namespace: Arc<str>` (the `<root>/<ns>` prefix, rendered once per snapshot as `cache_namespace` so per-key clones are pointer-cheap); tests `same_triple_different_namespace_is_a_distinct_slot` pin this. `retain_paths(namespace_prefix, live)` prunes side-map entries **scoped to one namespace** (normalised to a `/` boundary so `tenants/a` never claims `tenants/a2`) on each manifest commit — a global retain would evict every other tenant's warm state per flush. `prune_namespace` eagerly drops a namespace's entries from all four caches on eviction; the byte-budgeted foyer tiers reclaim lazily since they are strictly bounded.

### Index-vs-flat freshness equivalence

The invariant: an SST-backed secondary index (`.vg` DiskANN/Vamana vector, `.ft` BM25 text) must return exactly what the flat scan would at this snapshot. Two mechanisms enforce it. (1) **Union the memtable delta.** `vector_fresh_delta` (`read.rs:2683`) returns every node id the `.vg` has not absorbed (committed memtable + staged overlay, highest-LSN-wins): `Some(embedding)` to merge into the KNN, `None` to *suppress* an id that is tombstoned, no longer carries the label, or dropped its embedding; the executor unions this with the ANN result. (2) **Gate and fall back.** `index_outrun_by_nodes(index_name, kind)` (`read.rs:2639`) reports whether any persisted `Nodes` SST has `max_lsn` greater than the index's — comparing **LSNs, not levels**, which closes the partial-compaction window where a shallow merge rewrites a subset to L1 and empties L0. The lockstep `Nodes` SST written by the same authoritative (deepest-level) rebuild shares the index's `max_lsn` exactly, so it is never flagged. `text_search` returns `Ok(None)` — "fall back to the exact flat scan" — when there is no index SST, when `index_outrun_by_nodes` is true, when an unflushed upsert carries the indexed label (a live corpus delta that would shift BM25's corpus-wide N/avgdl/df stats), or when a tombstoned/relabelled id is itself an indexed document. The label-scoped check means an unflushed write to an *unrelated* label no longer demotes every `search.bm25` to an O(corpus) scan.

## 5. The Cypher/GQL Query Engine

NamiDB's query engine lives in `crates/namidb-query` and is organized as a four-stage pipeline whose stages are named in `crates/namidb-query/src/lib.rs`: (1) `parser` turns source text into an abstract syntax tree (AST); (2) `plan` *lowers* the AST to a `LogicalPlan` intermediate representation (IR); (3) `optimize` rewrites that IR to a semantically equivalent but cheaper tree; (4) `exec` evaluates the tree against a read `Snapshot` (read queries) or a `WriterSession` (write queries). The convenience entry point `plan(query, catalog) = optimize(lower(query)?, catalog)` composes stages 2–3; `plan_cache::parse_lower_optimize` composes all of 1–3 and is the unit the cloud worker caches keyed by an `xxh3-64` hash of the whitespace-normalized query text (`crates/namidb-query/src/plan_cache.rs`).

### Parsing

The parser (RFC-004) is a two-pass, hand-written front end. `parser::lexer::lex` is a byte-by-byte tokenizer producing `Vec<Spanned<Token>>` with comments and whitespace stripped and keywords case-folded to a fixed `Token` enum (`crates/namidb-query/src/parser/lexer.rs`). `parser::grammar::parse_query` is recursive descent for clauses with an embedded Pratt (precedence-climbing) parser for expressions (`crates/namidb-query/src/parser/grammar.rs`). Expression nesting is bounded by `MAX_EXPRESSION_DEPTH = 128` to prevent stack overflow on adversarial input; because an evaluated expression can be no deeper than the accepted AST, this also bounds evaluator recursion. The accepted surface is the Cypher-25/GQL v0 subset of RFC-004: `MATCH`/`OPTIONAL MATCH`, `WHERE`, `WITH`, `RETURN`, `ORDER BY`/`SKIP`/`LIMIT`, `UNWIND`, `CREATE`/`MERGE`/`SET`/`REMOVE`/`DELETE`/`DETACH DELETE`/`FOREACH`, `UNION`/`UNION ALL`, `CALL`, list/pattern comprehensions, quantifiers (`all`/`any`/`none`/`single`), and `shortestPath`/`allShortestPaths`.

### The logical-plan IR

`LogicalPlan` (RFC-008, `crates/namidb-query/src/plan/logical.rs`) is a `Serialize`/`Deserialize` enum of relational-plus-graph operators, so a plan round-trips bit-for-bit through a cross-process cache. Read/shape operators: `NodeScan { label, alias, predicates, projection }`, `NodeById`, `NodeByPropertyValue { multi }` (unique vs indexed point lookups), `Expand` (single- and variable-length traversal, carrying `direction`, `edge_type`, `target_labels`, `optional`, `back_reference`, `shortest: ShortestMode`, `path_binding`), `Filter`, `Project { distinct, discard_input_bindings }`, `Aggregate { group_by, aggregations }`, `TopN { keys, skip, limit }` (fused sort+skip+limit), `Distinct`, `Union { all }`, `Unwind`, `Empty`, `CrossProduct`, `HashJoin { build, probe, on, residual }`, `HashSemiJoin { negated, residual }`, `Argument`, `SemiApply`, `Apply`, `PatternList`, plus the specialized leaves `MultiwayJoin` (RFC-024), `EdgeTypeCount`, `VectorSearch` (RFC-030), and `CallProcedure`. Write operators: `Create`, `Merge`, `Set`, `Remove`, `Delete { detach }`, `Foreach`. `SKIP`/`LIMIT` counts are a `RowCount` enum of `Const(u64)` or `Param(name)`, so the parameter name — not its value — is part of the cached plan; `RowCount::as_const()` returns `None` for `Param`, forcing count-dependent rewrites to treat it conservatively.

### Lowering

`plan::lower` (RFC-008, `crates/namidb-query/src/plan/lower.rs`) walks the AST clause list, threading a `BTreeSet<String>` of visible bindings and emitting `LowerError::{BindingNotFound, DoubleBinding, InvalidPattern, ...}` on scope violations. A `MATCH` pattern part lowers to a `NodeScan` seed followed by one `Expand` per relationship hop; an inline `{prop: v}` on a node lowers to a `Filter` (later possibly rewritten to a point lookup), and a labelled `Expand` target emits a synthetic `__label_eq(alias, "L")` guard filter that `normalize_filters` later removes when the `Expand.target_labels` already enforces it. Top-level `UNION` becomes a right-leaning chain of `Union` nodes.

### The optimizer

`optimize::optimize(plan, catalog)` (`crates/namidb-query/src/optimize/mod.rs`) first runs the structural `apply_edge_count_pushdown` once, then iterates a rewrite battery to a fixpoint capped at `MAX_FIXPOINT_ROUNDS = 8` (each well-formed rewrite is idempotent within two rounds; the cap is defensive). Per round, in order: (1) `unique_lookup` collapses `Filter(prop = literal) → NodeByPropertyValue` when the property is schema-`unique` (point lookup) or non-unique-`indexed` (`multi:true` posting-list lookup), and `WHERE elementId(n)=…` → `NodeById`; (2) `vector_search` (feature `vector-index`) rewrites a KNN shape to `VectorSearch`; (3) `predicate_pushdown` splits each `Filter` on `AND` and sinks each conjunct toward the leaves, folding pushable single-column conjuncts into `NodeScan.predicates` (RFC-013 Parquet row-group pruning) with `Distinct`/`TopN`/write operators acting as barriers; (4) `normalize_filters` merges adjacent filters and drops `Filter(true)` and redundant `__label_eq`; (5) `convert_cross_to_hash` (RFC-012); (6) `convert_semi_apply_to_hash_semi_join` (RFC-014 decorrelation); (7) `reorder_joins` (RFC-016); (8) `detect_multiway_join` (RFC-024); (9) `apply_projection_pushdown` (RFC-015) runs last so it sees the final `NodeScan` shape. Reaching the KNN→`VectorSearch` rewrite *below* `WITH`/`Project`/`Filter` chains matters because `vector_search::recurse` rebuilds the `Project{TopN}` and `Filter{TopN}` wrappers structurally and re-attempts the match at each parent; without descending, only the single-stage terminal-`RETURN` form would ever reach the index, and a similarity threshold (`WHERE score >= 0.86`) is folded into the operator's `post_filter` (RFC-030 filtered ANN) rather than forcing a flat scan.

### Statistics and the cost model

The optimizer's cost inputs are a `StatsCatalog` (RFC-010/025, `crates/namidb-query/src/cost/stats.rs`) built in `O(|ssts|)` from a committed `Manifest`, holding per-label `node_count`; per-property `PropStats { null_count, non_null_count, min, max, ndv, unique, indexed }`; and per-edge-type `EdgeTypeStats { edge_count, avg/max out/in degree }`. Number-of-distinct-values (`ndv`) comes from HyperLogLog (HLL) sketches merged across SSTs (`absorb_hll_sketch`); if *any* contributing SST lacks a sketch the merge degrades to `Incomplete` and `ndv` stays `None`, at which point equality selectivity falls back to a constant. Per-label property statistics (RFC-025) are read from an `per_label_property_stats` sidecar on id-primary node SSTs (which span many labels in one `__overflow_json` column), resolving each `label_id` through the namespace `LabelDictionary`. The memtable is deliberately not consulted; single-writer flush cadence bounds the under-estimate.

`cost::selectivity` (`crates/namidb-query/src/cost/selectivity.rs`) returns the fraction of rows a predicate keeps, clamped to `[0,1]`, `NaN`→`0.5`. Boolean combinators use independence: `AND` multiplies, `OR` uses inclusion–exclusion `a+b−ab`, `XOR` uses `a+b−2ab`, `NOT` is `1−s`. Equality is `1/ndv` when `ndv` is known, else `FALLBACK_EQ = 0.10`; range comparisons interpolate linearly across `[min,max]`, else `FALLBACK_RANGE = 0.33`; `IS NULL` uses `null_count/(null+non_null)` else `0.05`; `IN` is `list_len/ndv` capped at 1, else `list_len × 0.10`; string tests `0.10`; unknown `0.50`. `cost::cardinality::estimate` (`crates/namidb-query/src/cost/cardinality.rs`) walks the plan bottom-up producing a parallel `Cardinality` tree carrying estimated `rows` and live `bindings` (alias → label/edge metadata). `Expand` multiplies by an average `branch_factor` (summed over the alternation set / all types for untyped), with variable-length capped at `MAX_VARLEN_BRANCH = 10 000`; `HashJoin` uses the Selinger-1979 estimate `|build|·|probe| / max(ndv_build, ndv_probe)` per key (product across keys, independence), then multiplies by any residual's selectivity; `CrossProduct` multiplies; `SemiApply`/`HashSemiJoin` keep `outer × clamp(inner/outer)` (or its complement when negated); writes report 0 output rows but preserve child cardinality; `UNWIND` uses list length or `DEFAULT_UNWIND_LEN = 5`. Default constants: `DEFAULT_BRANCH_FACTOR = 2.0`, `DEFAULT_REL_CARDINALITY = 1000`.

### Join execution and planning

`convert_cross_to_hash` fires after pushdown/normalize: at a `Filter(cross-side eq) → CrossProduct` it partitions the AND-conjuncts into `LeftEqRight`/`RightEqLeft` equi-keys (each side's aliases wholly on one subtree) and residuals; it emits a `HashJoin` whose `build` is the estimated-smaller subtree and whose `JoinKey`s are canonicalized so `build_side` references build-subtree aliases, carrying the residual `AND`-chain as `residual`. `reorder_joins` (RFC-016) then re-estimates both branches bottom-up and swaps `build`/`probe` (mirroring each `JoinKey`) whenever `probe_rows < build_rows`, because the initial orientation was chosen before pushdown/decorrelation reshaped the branches; it is idempotent. Execution (`crates/namidb-query/src/exec/walker.rs`, `LogicalPlan::HashJoin`): the build subtree is materialized into a `HashMap<Vec<String>, Vec<Row>>` keyed by the `fingerprint_value` of each build key; any key column evaluating to `NULL` skips the row (three-valued logic: a NULL key never matches). The probe streams, looks up its key, and for each build match emits the union of bindings, dropping rows whose `residual` is not `Bool(true)`. Complexity is O(|build| + |probe| + |output|). `HashSemiJoin` (RFC-014) executes the `inner` subplan **once** (not per outer row — the decorrelation win), builds a `HashSet` of keys (plus a side index when a residual is present), then streams the `outer`, keeping rows per the `(matched, negated)` truth table and propagating only outer bindings. Decorrelation (`convert_semi_apply_to_hash_semi_join`) is conservative: it fires only when the subplan has exactly one `Argument` leaf with a single binding `X` whose label is known in the outer scope, replacing the `Argument` with a fresh `NodeScan(label)` and joining on `X._id`; subplans containing `Aggregate`/`Distinct`/`TopN`/`Union`/nested joins/writes are left as `SemiApply`.

### Worst-case-optimal join (WCOJ)

`detect_multiway_join` (RFC-024, `crates/namidb-query/src/optimize/multiway_join.rs`) is gated by `NAMIDB_WCOJ=1` and requires `NAMIDB_FACTORIZE=1` (else it logs a warning and no-ops, because the executor refuses a `MultiwayJoin` on the flat path). It walks a contiguous `Expand` chain rooted at a labelled `NodeScan`, requiring every hop to be single-hop, typed, non-`Both`, `rel_alias`-free, non-optional, with at most one target label; it harvests `NodeBinding`s and `EdgeConstraint`s, runs a union-find cycle check (a cyclic constraint graph — triangle, k-clique, k-cycle, or parallel edges — is the trigger), and emits one `MultiwayJoin { vars, edges, ordering, factorize_required: true }`. The variable `ordering` keeps the head scan outermost then orders the rest by descending constraint-graph degree (ties by alias). Its AGM (Atserias–Grohe–Marx) cardinality bound `|Q| ≤ ∏_e |R_e|^(w_e)` uses the greedy fractional edge cover `w_e = 1/min(deg(from(e)), deg(to(e)))`, which is the LP optimum for regular shapes and an upper bound otherwise, clipped by the cartesian product of label sizes. Execution (`execute_multiway_join_factor` → `descend_multiway`) is a trie descent: level 0 scans the head label (pushing predicates to storage); each deeper level gathers the partner lists of every already-bound constraint via `Snapshot::sorted_partners`, unions the per-type lists of an alternation `[:A|:B]` with `MergeSortedUnion` (a k-way min-heap merge, `O(log k)` per key), and intersects across constraints with `LeapfrogIntersect` (Veldhuizen-2014 leapfrog triejoin, `crates/namidb-query/src/exec/leapfrog.rs`). `SortedSliceIter::seek` uses an exponential (galloping) probe so a jump of gap `d` costs `O(log d)`, giving the intersection `O(k log d)` per emitted key. At the leaf, `count_edge_multiplicity` emits `∏_e mult_e` copies so per-path multiplicity matches the binary executor exactly.

### Factorization

Factorization (RFC-017, `crates/namidb-query/src/exec/factor.rs`) replaces `Vec<Row>` between operators with a `FactorRowSet { arena: FactorArena, leaves: Vec<FactorIdx> }`. A `FactorArena` is an append-only `Vec<FactorNode>`; a `FactorNode { parent: Option<FactorIdx>, slots: Vec<Slot> }` records only the bindings *added at that level*, and inherited bindings are recovered by walking `leaf → root`. `FactorIdx = u32` (4 bytes/node; index 0 is the pre-allocated empty `FACTOR_ROOT`), and a `Slot { name: Arc<str>, value: RuntimeValue }` shares binding names across the thousands of rows an expansion produces. This compresses many-to-many expansions: a hop that fans one parent to `d` neighbours pushes `d` child nodes all sharing the parent chain, instead of cloning the parent's `BTreeMap` `d` times. Merging two inputs uses `splice_from` (index-translating append) and `splice_under` (re-parenting). Materialization to a flat `Row` happens only at sinks (`materialize`, with child slots shadowing same-named parents — Cypher `WITH`-rebind semantics), optionally column-pruned by a projection list. Factorization is opt-in via `NAMIDB_FACTORIZE` (default off); `execute` dispatches to `execute_factor_path` or `execute_flat_path` accordingly. The correctness contract is **row parity**: every operator's factor implementation (e.g. `execute_expand_factor`) has a test asserting `row_set(flat) == row_set(fact)`, and the default flips on only when all operators have factor implementations and parity is green.

### Variable-length and shortest paths

`execute_expand` (RFC-023, `crates/namidb-query/src/exec/walker.rs:1032`) implements both single- and variable-length traversal as a per-seed BFS over a `Vec<Step>` frontier, where `Step { tail, row, trail, rels }`. Hop bounds come from `RelationshipLength`; an unbounded `*` is clamped to `UNBOUNDED_VAR_LENGTH_CAP = 64` hops (`clamp_hop_max`). Cypher relationship-uniqueness (trail semantics) is enforced by `Step.rels`, the list of stored edge identities `(edge_type, src, dst)` already traversed on this path; an edge whose key is in `rels` is skipped, so `-[:R*2..2]-` cannot walk one edge out and back. This is only populated for multi-hop expansions (`max > 1`); the single-hop hot path leaves it empty. `back_reference` reads the pre-bound target once and keeps only paths whose tail equals it. For `shortestPath` (`ShortestMode::First`) and `allShortestPaths` (`ShortestMode::All`), a visited-set BFS prune is engaged **only for `min ≤ 1`**: a `HashSet<NodeId>` seeded with the source drops any neighbour reached at an earlier level, because every prefix of an unweighted shortest path is itself shortest, so discarding re-visits preserves *all* shortest paths while bounding the frontier at `O(V)` per level instead of the `deg^hop` walk enumeration; the set is sealed at level end so `All` admits every same-level arrival (each a distinct shortest path), and `First` additionally keeps one walk per node per level and stops the whole BFS at the first hit. When `path_binding` is set (`MATCH p = shortestPath(...)`), the alternating Node/Rel `trail` is materialized into a `RuntimeValue::Path` on the hit row so `length(p)`/`nodes(p)` work.

### The write path

Write execution (RFC-009, `crates/namidb-query/src/exec/writer.rs`) drives a `LogicalPlan` against a mutable `WriterSession`, delegating read sub-plans to the read walker over `WriterSession::overlay_snapshot`, which reflects the staged batch so a `MATCH`/`MERGE`-probe/unique-check sees the writer's own uncommitted rows (read-your-own-writes, RFC-026). `execute_write_inner` iterates input rows and applies each write operator per row: `apply_create` builds `core_props`, resolves `{_id: …}` to the storage `NodeId` (rejecting a collision against the overlay), enforces unique/composite-unique/NOT-NULL/vector-dimension constraints, then calls `upsert_node_with_labels` / `upsert_edge`; `apply_sets`/`apply_removes` re-upsert the mutated node; `apply_delete` tombstones the target (with `detach=true` first stripping incident edges across every edge type in the manifest schema via `detach_incident_edges`); `MERGE` (`apply_merge`) probes the pattern (single node or one node-rel-node chain) against the overlay and applies `ON MATCH SET` on a hit or `apply_create` + `ON CREATE SET` on a miss. Unique-constraint checking (`find_unique_conflict`/`find_composite_conflict`) probes the per-writer transactional unique-value index via `writer.unique_probe`, which returns `Conflict(id)`/`NoConflict`/`Unindexable`; `Unindexable` (non-scalar value) falls back to a label scan over the overlay, the source of truth. A write commit becomes a memtable+WAL commit in `WriterSession::commit_batch`: staged mutations accumulate as WAL records and `pending_payloads` `(MemKey, LSN, MemOp)`; commit seals the pending WAL segment, PUTs it with `PutMode::Create`, CAS-swaps the `Manifest` to register the segment (epoch-fenced), and only after the CAS lands drains `pending_payloads` into the in-memory memtable; any failure before the CAS leaves the memtable untouched, and `execute_write` `discard_batch`es on error so a shared long-lived writer is not left with orphan records.

### The value and equality model

The executor's value type is `RuntimeValue` (`crates/namidb-query/src/exec/value.rs`): `Null`, `Bool`, `Integer(i64)`, `Float(f64)`, `String`, `List`, `Map`, `Node`, `Rel`, `Date(i32)`, `DateTime(i64)`, `Bytes`, `Vector(Vec<f32>)`, `Vector8 { codes, scale }` (int8-quantized), and `Path`. The canonical equality key for `DISTINCT`, `GROUP BY`, `count(DISTINCT …)`, and hash-join build/probe keys is `fingerprint_value` (`crates/namidb-query/src/exec/walker.rs:4177`), which serializes a value to a string that is equal iff the values are equal. Every variable-length payload is length-prefixed (`S<len>:<bytes>`, `L<n>:`, `M<n>:`, `V<dim>:`, `Y<len>:`), so distinct values cannot collide through a shared separator; floats encode their exact IEEE-754 bit pattern with `-0.0` canonicalized to `+0.0` (so `==` and the fingerprint agree); nodes fingerprint by `id`, relationships by `(type, src, dst)`. This bit-exact, length-prefixed encoding fixed earlier collisions where same-dimension vectors, same-length byte strings, or floats differing past ten decimals hashed together. NULL handling follows three-valued logic throughout: a NULL join key is dropped (never matches), a NULL residual drops the joined row, and `Aggregate` skips NULL arguments; sort/`min`/`max` use a total `order_for_sort` so NULL ordering is deterministic.

## 6. Vector Search: the DiskANN/Vamana ANN Index

NamiDB accelerates top-k nearest-neighbour retrieval over node embeddings with a DiskANN/Vamana approximate-nearest-neighbour (ANN) graph index, gated behind the `vector-index` Cargo feature and documented in RFC-030. The design principle stated in RFC-030 and enforced throughout the code is that the index is *an acceleration of the flat scan, never a different answer*: a set of freshness gates, a memtable/overlay delta merge, adaptive over-fetch, and an exact flat-scan fallback together guarantee the indexed path returns exactly what a brute-force scan would (to f32 tolerance on scores). The algorithm layer (`crates/namidb-ann`) is storage-agnostic and generic over a `VectorSpace` trait; the storage layer (`crates/namidb-storage/src/sst/vector.rs`) wraps it in a self-contained `.vg` SST body; the query layer (`crates/namidb-query`) rewrites KNN-shaped Cypher into a `VectorSearch` operator and merges index results with fresh writes.

### The VectorSpace abstraction

`VectorSpace` (`crates/namidb-ann/src/space.rs`) is the only way the build and search algorithms reach stored vectors. It exposes `len()`, `dim()`, `pair_distance(a: u32, b: u32)` (member-to-member, used by build), and `query_distance(query: &[f32], b: u32)` (external f32 query to member, used by search). Members are addressed by dense ordinals `0..len()`. The contract — stated in the trait doc — is **"lower is closer" and every distance must be finite**, because the beam-search heaps use a total order (`f32::total_cmp`) and a convergence comparison that assume no `NaN`. Three implementations ship:

| Space | Metric | Distance | Use |
|---|---|---|---|
| `F32CosineSpace` | cosine | `1 − dot/(‖a‖·‖b‖)`, cosine clamped to `[-1,1]` before `1 −` | recall-golden f32 path (cosine and, via MIPS, dot) |
| `L2Space` | euclidean | `sqrt(Σ(aᵢ−bᵢ)²)` | euclidean indexes (magnitude-sensitive) |
| `Int8Space` | cosine | `1 − cosine` over int8 codes | shipped quantized path (~4× smaller) |

Zero-vector semantics are made explicit and consistent: in both cosine spaces, zero-vs-zero is distance `0.0` (identical) and zero-vs-nonzero is `1.0` (orthogonal/maximally distant but finite). The `Int8Space` query path keys this on `q_norm == 0 && norm == 0`, never on the dot product — the dot is forced to 0 in every zero-norm case and cannot distinguish the two (a documented regression fix in `space.rs`).

### Vamana graph construction

The build (`crates/namidb-ann/src/build.rs`, DiskANN Algorithm 2) produces a `VamanaGraph { adjacency: Vec<Vec<u32>>, entry: u32 }` (`crates/namidb-ann/src/graph.rs`) — bounded-degree directed out-adjacency lists plus one entry point. `BuildParams` defaults are **R = 64** (max out-degree), **L_build = 128** (build beam width), **α = 1.2** (prune diversification), and `init = Auto`. These defaults are duplicated in `BuildParams::default()` and in the server DDL `unwrap_or(64/128/1.2)` (`crates/namidb-server/src/lib.rs`), a sync hazard RFC-030 calls out explicitly.

The build proceeds:

```
build(space, params):
  n = space.len(); if n==0 → empty; if n==1 → one node, no edges
  l_build = max(l_build, r + 1)                 # prune needs ≥ r candidates
  entry = approximate_medoid(space)             # exact for n ≤ 256, else 256-sample
  adj = init(n, r)                              # BruteForce if n ≤ 4000 else Random (Auto)
  for i in random_permutation(0..n):
      found = beam_search(adj, entry, k=l_build, ef=l_build, dist=pair_distance(i,·))
      new = robust_prune(i, found\{i}, alpha, r)
      adj[i] = new
      for j in new:                            # add reverse edges
          adj[j].push(i) if absent
          if |adj[j]| > r: adj[j] = robust_prune(j, adj[j], alpha, r)
```

`approximate_medoid` samples `MEDOID_SAMPLE = 256` ordinals (all of them when `n ≤ 256`) and picks the one minimizing total intra-sample distance. `InitStrategy::Auto` uses exact brute-force R-NN init below `AUTO_BRUTEFORCE_MAX = 4_000` and a random init above it; `random_init` draws `min(r, n−1)` distinct neighbours per node in `O(r)` via Floyd/partial-Fisher-Yates sampling (`rand::seq::index::sample`), replacing an earlier `O(N²)` full-shuffle that stalled large-corpus compaction.

**`robust_prune` (DiskANN Algorithm 1) with α-diversification.** Given a candidate set of `(distance-to-anchor, id)` pairs, it excludes the anchor, sorts ascending by distance (id tie-break), dedups by id keeping the closest, then greedily keeps the nearest remaining candidate `p*` and *occludes* any later `p''` that lies in `p*`'s shadow — the diversification test `α · d(p*, p'') ≤ d(anchor, p'')` — capping the kept list at `R`. A larger α occludes fewer candidates, yielding more diverse, higher-recall (but denser) neighbour lists. A subtle correctness carve-out: the occlusion is skipped when `d(p*, p'') == 0.0` (exact duplicates), because a zero distance makes the test trivially true and would make every duplicate copy unreachable — a query that exactly matches a duplicated vector would then retrieve only one copy. The guard is `d_star_pp > 0.0 && alpha * d_star_pp <= d_anchor_pp`.

**Query-time traversal.** `beam_search(adjacency, n, entry, k, ef, dist)` (`crates/namidb-ann/src/search.rs`) is a greedy best-first beam shared by build and query, parameterized only by a `dist: Fn(u32) -> f32` closure. It maintains a min-heap of candidates to expand (closest first), a max-heap of the `ef` closest results seen (`peek()` = current farthest), and an `O(n)` `visited` bit-vector. It expands the closest unexpanded candidate, admitting a neighbour when the beam is not yet full or the neighbour beats the current farthest; it converges when the beam is full (`results.len() == ef`) and the closest unexpanded candidate is farther than the worst kept result. `k` is clamped to `k.min(n)` and `ef` to `ef.max(k).min(n)`; an out-of-range `entry` (possible from a corrupt/checksum-less body) returns empty rather than panicking.

**Complexity.** Per query, navigation visits O(ef) nodes, each expanding up to R neighbours at O(dim) per distance plus O(log ef) heap work — O(ef · R · dim) plus an O(ef · dim) f64 rerank. The build runs one beam search (ef = L_build) and one robust_prune (≈ O(R · L_build · dim)) per point, so it is ≈ **O(n · L_build · R · dim)** for random init, dominated by O(n² · dim) when brute-force init fires (n ≤ 4000). The build is deterministic: `build_with_seed` runs over a `ChaCha8Rng` seeded (in storage) from `xxh3_64(index_name)`, so the same `(data, descriptor, name)` always yields the same graph.

### Distance metrics and the MIPS reduction for dot

All three metrics are indexable. `cosine` navigates with `F32CosineSpace`; `euclidean` navigates with `L2Space` (a cosine graph would mis-rank whenever magnitudes vary); `dot` (maximum inner product) navigates with cosine over **MIPS-augmented** vectors. Plain cosine navigation is magnitude-blind, but a dot top-k is dominated by large-norm vectors — exactly why a user picks `dot`. `build_body` therefore applies the Bachrach et al. (2014) MIPS→cosine reduction (`mips_augment`, `crates/namidb-storage/src/sst/vector.rs`): with `M² = max corpus ‖x‖²`, every vector `x` gets one appended coordinate `sqrt(M² − ‖x‖²)` (clamped ≥ 0), making every augmented vector have norm exactly `M`. Navigation uses the **zero-augmented query** `mips_query` (raw query with a trailing 0), whose dot with the augmentation coordinate vanishes; cosine over the augmented set then orders exactly by the raw inner product. The body records this in its metric tag: current dot bodies are stamped **`"dot-mips"`** and legacy pre-MIPS bodies remain **`"dot"`** (plain-cosine navigation) so they keep serving until an authoritative compaction rebuilds them. Crucially, the body always stores the *original, un-augmented* vectors, so decode reconstructs the augmentation deterministically and the rerank uses the true metric.

### int8 quantization

`crates/namidb-core/src/quantize.rs` is the single shared definition of per-vector symmetric max-abs int8 quantization. `quantize_i8(v) → (codes: Vec<i8>, scale: f32)` takes the max absolute value over *finite* components only (non-finite components code to 0 so the scale can never poison to `NaN`), sets `scale = max_abs / 127`, and rounds/clamps each `xᵢ/scale` to `[−127, 127]`; an all-zero or zero-max input yields all-zero codes with `scale = 0.0`. Per-vector scaling is essential at high dimension: a single fixed scale wastes almost all of the int8 range on unit vectors whose components are ~1/√dim, collapsing recall (~0.87 at dim 1536 per the bench harness). The stored form costs ≈ `dim + 4` bytes/vector versus ≈ `4·dim` for f32 — the ~4× reduction. Scoring is *asymmetric*: `dot_i8_asymmetric(query_f32, codes, scale) = scale · Σ qᵢ·codeᵢ` keeps the query in f32 and never expands the stored side, and `norm_i8(codes, scale) = scale · sqrt(Σ codeᵢ²)`. `Int8Space` builds cosine from these primitives; the per-vector `scale` appears in both numerator and denominator and cancels, so int8 cosine is scale-invariant (which is why int8 is cosine-only — `build_body` rejects int8 with any other metric).

Because the graph navigates on the *quantized* cosine, served scores would otherwise drift from the flat scan. The executor closes this: for an int8 index, `try_index_search` fetches an over-fetch pool (`INT8_RESCORE_POOL = 4`, i.e. 4k candidates even with no filter) and **rescores every candidate with the exact f32 metric from its stored embedding before ranking and truncation**, so both served scores and top-k membership match the flat scan and a node scores identically before vs after compaction folds it into the index. The quantization error is confined to *beam recall* — which nodes are visited — not to the reported score.

### The `.vg` SST body format and decoded-index cache

A `.vg` body is self-contained so a top-k needs exactly one object fetch. Layout is an 8-byte magic `MAGIC = b"NAMIVG03"` (`NAMI` `VG` `\0` major=3) followed by a bincode-serialized `VectorGraphBody { dim: u32, metric: String, ids: Vec<[u8;16]>, storage: VectorStorage, graph: VamanaGraph }`, where `VectorStorage` is `F32(Vec<Vec<f32>>)` or `Int8 { codes: Vec<Vec<i8>>, scales: Vec<f32> }`. The `ids`, `storage`, and `graph.adjacency` are parallel per node ordinal. **There is no checksum**: `VectorGraphIndex::decode` therefore validates a magic match, a known metric name, equal lengths across `storage`/`ids`/`adjacency`, an in-range entry point, and matching int8 `codes.len()`/`scales.len()`. `build_body` returns `Ok(None)` when fewer than 2 members exist (the caller keeps the flat scan), validates each vector against `desc.dim`, and — **for cosine only** — excludes all-zero vectors so the indexed corpus matches the flat scan's `vector_score(Cosine, …)` NULL-drop. Any decode error is treated as "index absent" by the read path, which then falls back to the flat scan rather than erroring a query.

`VectorGraphIndex::search(query, k, ef)` navigates the graph for up to `ef` candidates in the navigation space (clamps `ef = ef.max(k)`), then reranks: f32 indexes recompute the true metric in f64 via `metric_score` from the original vectors; int8 uses `1.0 − dist` (the quantized cosine). Results sort by `higher_is_better() = !Euclidean`. Because decoding deserializes every vector plus the full adjacency *and* clones the vectors into the navigation space, the storage layer caches the decoded `Arc<VectorGraphIndex>` process-wide, keyed by absolute SST path, in `SstCache::vector_indexes: Arc<Mutex<HashMap<String, Arc<VectorGraphIndex>>>>` (`crates/namidb-storage/src/cache.rs`; fetched in `read.rs::fetch_vector_index`). SSTs are immutable per UUIDv7-keyed path, so cached indexes never go stale; superseded paths are pruned. Without it, every KNN and each widening round would pay O(index size).

### Freshness, filtered ANN, and correctness

The index is rebuilt (not merged — a Vamana graph is not row-mergeable) only on an **authoritative** (deepest-level) compaction whose merged rows span the full label corpus; the new `.vg` descriptor is stamped with `max_lsn = corpus_max_lsn`. `try_index_search` (`crates/namidb-query/src/exec/walker.rs`) enforces freshness equivalence with the flat scan through several gates before serving from the index:

- **Freshness gate** — `Snapshot::index_outrun_by_nodes(index_name, VectorGraph)` (`read.rs`) returns true when any persisted `Nodes` SST has a `max_lsn` greater than the index's, meaning a node was persisted but not yet folded into an authoritative `.vg`. Comparing *LSNs, not levels* closes the partial-compaction truncation window (a shallow merge that empties L0 no longer hides rows); the lockstep `Nodes` SST from the same authoritative merge shares the index `max_lsn` exactly, so it is never flagged. When no `.vg` exists yet but some `Nodes` SST does, it reports "outrun" to force the flat scan.
- **Memtable/overlay delta union** — `vector_fresh_delta(label, property)` returns the committed memtable plus staged overlay entries the `.vg` has not absorbed (highest-LSN-per-id wins): `Some(emb)` is a live embedding merged into the KNN, `None` suppresses a now-stale id (tombstoned, label removed, or embedding dropped). The delta is pre-scored once with the same `vector_score(distance, …)` used by the flat scan, deduped against index hits (a `seen` set seeded from `delta_ids` each round), so a just-written vector is found immediately and a deleted one disappears.
- **Zero-vector cosine semantics** — a zero-magnitude cosine query is dropped by the flat path (`vector_score` returns `None`), but the index rerank would score similarity `0.0`; the cosine-only guard `metric == Cosine && qv.all(== 0)` flat-falls-back so the index agrees with the `cosine_similarity` builtin's NULL semantics.
- **Dimension enforcement** — a query whose length ≠ `index_dim` flat-falls-back (raising the canonical dimension-mismatch error rather than a silently prefix-scored answer); write-time `enforce_vector_dims` (`writer.rs`) rejects a wrong-dimension embedding as a `Constraint` error, keeping the corpus uniform.

**Filtered ANN (RFC-032, adaptive geometric widening).** A residual `WHERE` alongside a KNN is folded into the `VectorSearch`'s `post_filter`, but the index navigates *without* the predicate (it is filter-unaware — `beam_search` sees only ordinals and distances). Because a selective filter can leave fewer than `k` survivors from an over-fetched pool, the executor widens geometrically before the O(n) flat fallback, with constants `OVERFETCH_BASE = 8`, `WIDEN_GROWTH = 4`, `MAX_WIDEN_ROUNDS = 4` (multipliers **8 → 32 → 128 → 512**). Each round fetches `kprime = max(k, k·mult + |delta_ids|)` with beam `ef = ef_search.max(kprime)` or default `max(kprime, 64)`, merges + filters, and returns if ≥ k survivors remain; it stops early once `hits.len() < kprime` signals the index is exhausted, then returns `Ok(None)` to trigger the exact flat scan (the ground truth). With **no** filter there is exactly one round at `mult = 1` — an exact top-k cannot under-fill from selectivity. Widening only ever *grows* the beam, so recall rises and correctness never changes; a per-candidate `check_deadline()` keeps a widened filtered ANN interruptible. RFC-032 proposes true pre-filtering (attribute bitmaps materialized into the `.vg`, keyed by `max_lsn`, feeding a `beam_search_filtered`), which is not yet implemented.

**Beam-width surface.** The procedures (`search.vector`, `search.hybrid`, `db.index.vector.queryNodes`) take a first-class `ef`; the natural/operator form reads a namespaced, explicitly non-stable parameter `$__vector_ef` threaded into the same `ef_search` slot. Because it is only ever clamped up (`ef.max(kprime)`), it cannot corrupt results. RFC-036 proposes replacing it with a stable `OPTIONS { ef: … }` clause.

## 7. Full-Text Search and Graph-Algorithm Kernels

This section documents two analytical subsystems that sit above the LSM storage engine: the BM25 full-text retrieval path, and the exact in-memory graph-algorithm kernels invoked by `CALL algo.*`. Both are deliberately built on a single shared source of truth for their arithmetic, so that a precomputed index and a query-time scan return bit-identical results, and so that community outputs are reproducible across runs.

### Shared BM25 primitives

All BM25 arithmetic and tokenization lives in one module, `crates/namidb-storage/src/text.rs`, and is re-exported to the query engine through `crates/namidb-query/src/exec/text_scoring.rs` (`pub use namidb_storage::text::{...}`). This is a correctness decision, not a code-hygiene one: the persistent inverted index (`crates/namidb-storage/src/sst/text.rs`) and the query-time flat scan (`bm25_ranked` in `walker.rs`) must agree exactly, so they call the same `tokenize`, `parse_query`, `bm25_idf`, and `bm25_term_score`.

The scoring model. For a query term `t` in document `d`, the score contribution is the Okapi BM25 term with Lucene-form IDF:

```text
score(d,t) = idf(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 - b + b·|d|/avgdl) )
idf(t)     = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )
```

with the constants read directly from the code: `K1 = 1.5` and `B = 0.75` (`text.rs:33,35`). `N` is the document count, `df(t)` the number of documents containing `t`, and `avgdl` the average document length in tokens. The `+1` inside the IDF logarithm is the Lucene variant that keeps IDF non-negative even for a term present in more than half the corpus (classic Robertson–Spärck-Jones IDF goes negative past N/2). `bm25_term_score` returns 0 for `tf == 0` and clamps `avgdl` to `≥ 1.0` so the length factor never divides by zero (`text.rs:277-284`). A document's per-term contributions are summed; a whole document's score is the sum over the distinct scored terms.

Two evaluation surfaces exist. The per-row scalar `bm25(document, query)` (`text_scoring.rs:51`) is corpus-free: it fixes `idf = 1.0` and a neutral reference `AVG_LEN = 120.0` (`text_scoring.rs:47`), so it composes anywhere in a projection but cannot weight rare terms. The `CALL search.bm25` procedure and the persistent index both use real corpus IDF and the true `avgdl`.

### Tokenization

`tokenize` (`text.rs:208`) splits on maximal non-alphanumeric runs, applies no stemming and no stopword removal, and folds case with Unicode-aware `char::to_lowercase` rather than `to_ascii_lowercase`. This matters: an ASCII-only fold left every non-ASCII capital unfolded, silently breaking case-insensitive search for all non-English text — `CAFÉ → café`, `ÜBER → über`, `ПРИВЕТ → привет` now match their lowercase forms.

CJK segmentation. Scripts written without inter-word spaces (Hiragana/Katakana `U+3040–30FF`, CJK Unified Ideographs and Extensions A/B, CJK Compatibility Ideographs, Hangul syllables — the exact ranges are in `is_cjk`, `text.rs:181`) would otherwise collapse an entire run into one token no realistic query types. `emit_segment_tokens` (`text.rs:221`) therefore emits overlapping bigrams for a maximal CJK run — `東京大学 → 東京, 京大, 大学` — the dictionary-free Lucene CJKAnalyzer approach; a length-1 run degrades to a unigram. A mixed segment like `iPhone東京` yields the Latin word `iphone` plus CJK bigrams. Because both index-build and query tokenize identically, `東京` (itself one bigram) is findable. Token positions are the emission ordinal of each token, so CJK positions are per bigram — critical for phrase adjacency.

### Query syntax: `parse_query`

`parse_query` (`text.rs:110`) turns a raw string into a `TextQuery { terms, phrases, prefixes }`, all interpreted identically by the index and the flat scan:

- **Phrases** — a double-quoted span is tokenized and pushed as a `Vec<String>` token sequence; its tokens must appear at *adjacent* positions in a candidate document. An unclosed quote runs to end-of-string; `*` inside quotes is ordinary punctuation; empty quotes are dropped. A single-token phrase degrades to a required-containment constraint.
- **Prefixes** — a `*` immediately after an alphanumeric run marks that run's *last* emitted token as a prefix pattern (earlier tokens stay plain terms; a bare `*` with no preceding alphanumeric is ignored). For a CJK run `東京大*`, the last bigram `京大` becomes the prefix and `東京` a plain term.
- **Terms** — everything else is bag-of-words. A query with no `"` and no `*` parses to plain terms only, preserving historical behaviour.

`terms` and `prefixes` are collected into `BTreeSet`s (distinct, sorted); `base_terms()` returns the sorted union of plain terms and all phrase tokens, so both consumers accumulate per-term contributions in one deterministic order and produce bit-identical floats even though f64 addition is non-associative.

### The persistent inverted index (`.ft`, magic `NAMIFT02`)

A `.ft` body is an 8-byte magic (`b"NAMIFT02"`, `sst/text.rs:37`) followed by a bincode-serialised `TextIndexBody` (`sst/text.rs:41`):

| Field | Type | Meaning |
|---|---|---|
| `n_docs` | `u32` | number of indexed documents (N) |
| `total_len` | `u64` | Σ document lengths in tokens (→ `avgdl`) |
| `doc_ids` | `Vec<[u8;16]>` | NodeId per document index `i` |
| `doc_lens` | `Vec<u32>` | token count per document `i` |
| `postings` | `BTreeMap<String, Vec<(u32,u32,Vec<u32>)>>` | term → list of `(doc_index, tf, ascending token positions)` |

`postings[t].len()` is exactly `df(t)`; the per-posting position vector is what makes phrase adjacency answerable from the index alone. `build_body` (`sst/text.rs:71`) runs during compaction over `(NodeId, concatenated-text)` pairs, tokenizes each, and appends postings in ascending document order so every list is pre-sorted by document index — deterministic and binary-searchable. A document with zero tokens still counts toward N and `total_len` (it is a document) but contributes no postings. An empty member set returns `Ok(None)` — nothing to index, keep the flat fallback.

The magic version is load-bearing. Version `02` added positions; a legacy `NAMIFT01` body (position-less `(u32,u32)` postings) fails `TextIndex::decode` (`sst/text.rs:129`) rather than being silently misparsed, and the read path maps any decode failure to "index absent" and serves the flat scan until the next authoritative compaction rebuilds the SST — a format bump degrades performance, never correctness. On decode, `TextIndex` also keeps a `sorted_ids: Vec<[u8;16]>` (a sorted copy of `doc_ids`) for O(log n) membership probes (`contains_doc`, `sst/text.rs:152`).

### Phrase and prefix evaluation on the index

`TextIndex::search_query` (`sst/text.rs:178`) assembles the scored-term set as `base_terms()` plus, per prefix, a bounded vocabulary expansion: it takes a `postings.range((Included(prefix), Unbounded))`, `take_while(|t| t.starts_with(prefix))`, capped at `PREFIX_EXPANSION_LIMIT = 64` (`text.rs:43`). This yields the lexicographically-first 64 matching terms — a short prefix over a large vocabulary can never explode into thousands of scored terms, and the pick is deterministic.

Phrases are a hard candidacy constraint. `phrase_docs` (`sst/text.rs:243`) computes, for each phrase, the set of documents where the tokens occur at adjacent positions: it walks the first token's postings, binary-searches each subsequent token's posting list for the same document, and then checks whether some start offset `p` in the first list has `p+1+j` present in token `j`'s position list (again by binary search — legal because positions are stored ascending). The per-phrase document sets are intersected; an empty intersection short-circuits to no results. Scoring then iterates the scored terms, computes `bm25_idf(N, df)` per term, and adds `bm25_term_score` for every posting whose document survives the `allowed` phrase set. Crucially, adjacency *gates candidacy but does not change the formula*: a passing document scores the phrase's tokens as ordinary BM25 terms. Results are sorted score-descending with a NodeId ascending tie-break, then truncated to `k` (`None` = all).

### The label-scoped freshness gate

The interesting correctness interaction is between the compacted index and uncompacted writes. `Snapshot::text_search` (`read.rs:2736`) decides whether the index is authoritative for the live corpus, and returns `Ok(None)` (fall back to flat scan) whenever it is not. The gate has three layers:

1. **An index SST must exist** for this index name (`kind == TextIndex && scope == index_name`).
2. **No unabsorbed node delta.** `index_outrun_by_nodes` (`read.rs:2639`) compares the index SST's `max_lsn` against every `Nodes` SST's `max_lsn`; a `Nodes` SST with a strictly higher LSN means flushed-but-not-yet-folded-in rows the index has not absorbed. Comparing LSNs, not levels, closes the partial-compaction truncation window (a shallow merge that rewrites a subset to L1 cannot hide rows just because L0 emptied). The lockstep `Nodes` SST written by the same authoritative merge shares the index's LSN exactly, so it is never (`>`) flagged.
3. **Label-scoped memtable/overlay check.** For each in-memory node entry: a `Tombstone` is pushed to a `dirty` id list; an `Upsert` whose record *carries* `label` (`record_carries_label`, `read.rs:3072`, checks the label id against the record's label set via the namespace `LabelDictionary`) is a live document delta and immediately forces `Ok(None)`; an `Upsert` that does *not* carry `label` (a possible relabel *away* from the corpus) is pushed to `dirty`. Then each `dirty` id is probed with `idx.contains_doc(id)` — if a dirty id is one of the indexed documents, the index would still serve a stale doc and its removal would shift the corpus stats, so `Ok(None)`.

Why label-scoping is essential: an unflushed write to an *unrelated* label does not carry `label`, so it lands only in `dirty`, and its id is not among the index's documents, so `contains_doc` returns false and the index still serves. Before this scoping, any unflushed write anywhere disabled the index globally, turning every `search.bm25` under live mixed traffic into an O(corpus) flat scan. Why the gate cannot merge suppressions the way the vector index does (RFC-030): a KNN result is per-node independent, so fresh embeddings can simply be unioned in. BM25 scores depend on corpus-wide N, `avgdl`, and `df` — adding or removing one document shifts *every* document's score — so the correct answer when a dirty id touches the corpus is a full flat recompute, not a delta merge. The gate is therefore conservative by design.

### Flat-scan fallback and exact parity

`bm25_ranked` (`walker.rs:1867`) is the retrieval core shared by `search.bm25` and the sparse leg of `search.hybrid`. It parses the query once, consults the index via `text_search`, and on `None` scans the label. `doc_text` (`walker.rs:2030`) forms each document by joining the configured string properties with a space; a node carrying none of them is not in the corpus. A single pass computes N, `total_len`, per-fixed-term `df`, prefix-matched-term `df` (`expanded_df`), and the phrase constraint via the shared `contains_phrase` (`text.rs:167`, a sliding-window match). Prefix expansion re-uses the same `BTreeMap::range` + `take_while` + `PREFIX_EXPANSION_LIMIT` pick over the discovered vocabulary, so the flat path expands to the same lexicographic head as the index. Scored terms are accumulated in a `BTreeMap` so summation order matches the index. The one-pass loop polls `namidb_storage::cancel::deadline_exceeded()` every 4096 documents and returns `ExecError::Timeout`, so a large flat scan honours the query deadline.

### The in-memory Graph and projection

The graph kernels operate on `namidb_graph::algo::Graph` (`algo.rs:71`), a directed multigraph: `out: HashMap<NodeId, Vec<(NodeId, f64)>>` (weight defaults 1.0), `nodes: Vec<NodeId>` in insertion order (isolates included), and `seen: HashMap<NodeId,()>` for O(1) dedup. Self-loops and parallel edges are tolerated. `snapshot_to_algo_graph` (`walker.rs:2990`) builds it from a snapshot under an `AlgoProjection { labels, edge_types, direction }` (defaults: all observed labels/types, `Natural` direction). With no label filter it does one label-agnostic `scan_all_node_ids` pass; with labels it validates each against the observed set (a typo errors rather than silently projecting empty) and scans with an empty property projection (node ids only). Edges come from `scan_edge_type`; when labels are given, the *induced* subgraph is taken — an edge survives only if both endpoints did (GDS semantics). Direction maps each stored edge as `Natural` (src→dst), `Reverse` (dst→src), or `Undirected` (both, self-loop added once). A `weight` edge property, if numeric, becomes the edge weight.

### Kernels

Every kernel has a `*_cancellable` variant; the public wrapper passes a never-cancel closure. Complexities (V nodes, E edges):

| Kernel | Algorithm | Complexity | Key constants |
|---|---|---|---|
| `weakly_connected_components` | union-find (path halving + union by rank) over undirected view | O(V·α(V) + E) | — |
| `strongly_connected_components` | Tarjan, iterative explicit-stack DFS, directed | O(V + E) | — |
| `pagerank` | power iteration, dangling-mass redistribution | O(iters·(V+E)) | damping 0.85, max 100 iters, L1 tol 1e-6 |
| `degrees` | one edge pass, in/out per node | O(V + E) | — |
| `triangle_count` | compact-forward (Latapy 2008), degree ranking + merge intersection | O(E^1.5) | — |
| `label_propagation` | asynchronous in-place sweeps, deterministic tie-break | O(iters·(V+E)) | default 10 iters |
| `louvain` | modularity local-move + aggregation levels | O(levels·sweeps·(V+E)) | 10 levels, 10 sweeps, tol 1e-4 |
| `betweenness` | Brandes: BFS + dependency accumulation | O(V·E) | undirected halved by caller |
| `shortest_paths` | BFS (unweighted) / Dijkstra (non-neg weights) | O(V+E) / O((V+E)log V) | — |
| `fast_rp` | sparse random projection + hop propagation | O(iters·E·d) | dim 256, weights [0,1,1,1], sparsity s=3, seed 42 |

**WCC** (`algo.rs:158`) maps nodes to dense indices, unions endpoints of every edge (direction ignored), then assigns dense component ids by canonical root iterating in *insertion* order — not HashMap order — so ids are stable across runs. The `UnionFind` (`algo.rs:1144`) uses path halving in `find` and union-by-rank.

**SCC** (`algo.rs:470`) is Tarjan (not Kosaraju), implemented iteratively with an explicit `(node, cursor)` DFS stack so a pathological deep graph cannot overflow the call stack; each node carries a discovery index and a low-link, and an SCC root (`low[v] == disc[v]`) pops its members off the component stack.

**PageRank** (`algo.rs:284`) initialises uniformly `1/N`, uses teleport `(1-d)/N`. Dangling nodes — those with no out-edge or no *positive* out-weight — have their mass collected and redistributed uniformly each iteration so probability is conserved. Only non-negative weights contribute: the weight sum and per-edge push both use `max(w,0)`, because keying the dangling guard on a raw signed sum let a node mixing `+3` and `−2` (sum `+1`) inject negative mass and produce negative or `>1` scores. Sources are iterated in insertion order so the non-associative f64 accumulation into each destination is reproducible. It stops on L1 `Σ|new−old| < tolerance`.

**Triangle count** (`algo.rs:594`) uses the compact-forward algorithm: rank nodes by `(degree, index)`, build a *forward* adjacency of each node's strictly higher-ranked neighbours sorted by rank, and count each triangle once from its lowest-ranked vertex by a linear merge-intersection of two sorted forward lists. This is O(E^1.5); it replaced a naive per-node neighbour-pair probe costing Σ deg(v)², which degenerates on power-law graphs (a single 50k-degree hub alone is ~1.25e9 hash probes). The local clustering coefficient is `2·T(v)/(deg(v)·(deg(v)−1))`, 0 for degree < 2.

**Label propagation** (`algo.rs:714`, Raghavan 2007) sweeps nodes in fixed order and updates labels *in place* (asynchronous schedule), damping the flip-flop oscillation a synchronous update suffers on bipartite structures. A node adopts the most frequent neighbour label, keeps its own when that is already a maximum, else breaks ties toward the smallest label id — fully deterministic. It relabels to dense ids in first-seen order.

**Louvain** (`algo.rs:1234`, Blondel 2008) works on the undirected view; per level it computes weighted degrees, runs local-move sweeps that greedily maximise modularity gain `k_i_in(c) − Σ_tot(c)·k_i/2m` (candidates scanned in ascending community id, a strictly-better gain `> best + EPSILON` required, so ties keep the current then lowest id), then aggregates communities into super-nodes with intra-community weight becoming self-loops. It reports the partition's modularity `Q = Σ_c internal(c)/m − (Σ_tot(c)/2m)²` and stops when a level fails to improve `Q` by `tolerance`, makes no move, or cannot coarsen further.

**Betweenness** (`algo.rs:1449`) is Brandes' exact algorithm: per source a BFS builds `σ` (shortest-path counts) and predecessor lists, then dependency `δ` is accumulated in reverse-BFS order — O(V·E) unweighted. Scratch arrays are reset only over the touched stack, not re-allocated. Raw directed scores double on an undirected projection, so the caller multiplies by 0.5.

**Shortest paths** (`algo.rs:817`) runs BFS for unweighted (first arrival is the hop distance) or Dijkstra with a binary min-heap for weighted; negative-weight edges are skipped as unsound, and the CALL layer rejects `weighted:true` outright when `graph.has_negative_weight()`. A `MinF64` newtype gives a `total_cmp` total order for the heap; the NodeId tie-breaks equal distances so pop order is deterministic. (Note: this single-source SSSP kernel is distinct from the Cypher `shortestPath()`/`allShortestPaths()` pattern functions of RFC-023, which compile to a hop-by-hop variable-length `Expand` BFS over CSR adjacency rather than to this module.)

**FastRP** (`algo.rs:978`, Chen et al. 2019) seeds each node with a very-sparse random projection (`±√3` with probability `1/6` each, else 0), L2-normalised, using a dependency-free SplitMix64 PRNG seeded from `(seed, node index)` so `(graph, options, seed)` is fully reproducible. That signal propagates over the degree-normalised undirected adjacency for `iteration_weights.len()−1` hops (message from j to i scaled by `deg(j)^β/deg(i)`, β = `normalization_strength`, default 0 = plain mean), each hop L2-normalised and combined with its weight; hop 0 (weight 0 by default) is dropped. Adjacency is built in insertion order so the non-associative f32 sums are stable across platforms. The result is a `HashMap<NodeId, Vec<f32>>` of dimension 256 — exactly the shape the vector index ingests, so structural embeddings can be written straight into a `.vg`.

### Determinism and cancellation

Two cross-cutting properties hold across all kernels. **Determinism**: every kernel iterates `graph.nodes()` (insertion order) and relabels communities/components by first occurrence, never by randomised HashMap order — required because f64/f32 accumulation is non-associative and because snapshot tests, cross-kernel joins, and reproducible output demand stable ids. **Cancellation**: each `*_cancellable` kernel polls its `cancel` closure every `CANCEL_CHECK_STRIDE = 4096` inner-loop iterations (`algo.rs:47`), or once per iteration for the O(V+E)-per-iteration kernels (PageRank, FastRP), and returns a `Cancelled` marker the CALL layer maps to `ExecError::Timeout`. The closure passed in is `namidb_storage::cancel::deadline_exceeded`, so a runaway `CALL algo.*` on a huge graph honours the query timeout mid-computation, not merely at the operator boundary. The stride is chosen so cancellation latency stays low while the poll cost stays in the noise of a tight CPU loop.

## 8. Interfaces, Service-Layer Concurrency, and Multi-Tenant Operation

NamiDB exposes one storage/query core through five surfaces — an HTTP/JSON endpoint, a Bolt (Neo4j wire) listener, an MCP server, a PyO3 Python client, and a Rust facade — that all bind to the same primitive: exactly one `WriterSession` per namespace behind a `tokio::sync::Mutex`, plus a lock-free published read snapshot. This section defines the concurrency and consistency contract those surfaces share and the mechanisms that keep it correct.

### Single writer, lock-free readers

The unit of service state is `AppState` (`crates/namidb-server/src/lib.rs:177`): `writer: Arc<Mutex<WriterSession>>` and `snapshot: Arc<SnapshotCell>`. Every mutation in the process serializes on that one mutex; the engine forbids a *second* `WriterSession` because a second session would claim a new epoch and fence the first (RFC-034; RFC-001 epoch fencing). Reads never take the writer lock: they load an `Arc<OwnedSnapshot>` from the `SnapshotCell` and execute against a frozen, `Arc<MemtableSnapshot>`-backed view (RFC-021). Consequently the service provides **snapshot isolation** for reads and **serial** writes per namespace; a slow or stuck writer cannot degrade read latency, and read throughput scales across cores.

Consistency across the two write protocols is guaranteed by both HTTP and Bolt taking the *same* mutex and, after a successful `commit_batch`, republishing via `state.snapshot.store(writer.owned_snapshot())` while still holding the lock (`lib.rs:2248`, `bolt.rs:670`). This is the cross-request read-your-own-writes invariant (RFC-021/026): a write is not ACKed until its effects are both durable (WAL append + manifest CAS) and visible in the published snapshot, so any *later* request that loads the snapshot observes it. The commit critical section is IO-bound — two object-store round-trips (pipelined WAL+manifest-body PUT, then the pointer CAS which is the linearization point) — so per-namespace write throughput is bounded by roughly `1/(commit_RTT + in-lock read IO)`. RFC-034 characterizes this ceiling as architecturally real but unmeasured, and proposes group commit (many stagers, one committer) as the redesign that preserves the single-epoch invariant.

Foreground writers acquire the lock with a **bound**: `lock_writer_bounded` (`lib.rs:1178`) wraps `writer.lock()` in `tokio::time::timeout(NAMIDB_WRITER_LOCK_TIMEOUT, …)`; on timeout the request fails fast with HTTP 503 (`writer_busy_response`) or a transient Bolt failure, so request queues stay bounded behind a stuck writer. The CLI default is `30s` (`main.rs:238`); `Duration::ZERO` disables it. Background tasks (flush, compaction, recovery) deliberately use an *unbounded* `writer.lock().await` — a queued write behind a broken writer could only fail, so it is better to wait for a recovered session. The bounded sites are the HTTP/Bolt auto-commit write paths (`lib.rs:2231`, `lib.rs:2718`, `bolt.rs:654`) and Bolt `BEGIN` (`bolt.rs:959`).

### Writer resilience and recovery

`recovery.rs` owns the "drop the session and reopen" contract. `Error::requires_writer_reopen()` returns true only for `Error::Fenced { .. }` and `Error::ManifestCommitCas { .. }` (`crates/namidb-storage/src/error.rs:111`); a third state — a session *poisoned* by a terminal commit failure — surfaces as `Error::Precondition` (ambiguous with user errors) and must be tested separately via `WriterSession::is_poisoned()`. `recover_writer_if_needed` (`recovery.rs:148`), called *under the held writer lock* on every write/DDL/flush/compaction-install failure, marks `WriterHealth` degraded and retries `WriterSession::reopen()` up to `REOPEN_ATTEMPTS = 3` times with linear backoff `REOPEN_BACKOFF * attempt` (base 50 ms). `reopen()` (`ingest.rs:494`) re-runs `open_with_caches` against the same store/paths: it claims a **fresh epoch** (fencing whoever fenced it), rebuilds the memtable by WAL replay, empties the pending batch, and clears poison; uncommitted (never-ACKed) records are dropped, so nothing durable is lost. On success it republishes the recovered snapshot and marks health OK.

`WriterHealth` (`recovery.rs:36`) is a `Mutex<Option<String>>` read lock-free by `/v0/health`. A degraded writer turns readiness to HTTP 503 while reads still succeed — so an orchestrator stops routing writes to a server whose writer can only fail. A second, lock-free signal is the **read-side fence probe** `probe_read_fence` (`recovery.rs:111`), run from the maintenance loops: it does one advisory `load_current()` GET and calls `observe_peer_epoch(observed, local)`. If the store's manifest epoch outranks the published snapshot's, a peer writer has fenced this node and readiness drops; crucially, epoch parity clears *only* a probe-set reason (`FENCE_PROBE_PREFIX`), never a commit-failure reason the recovery path owns. This closes the split-brain window where a fenced zombie kept serving stale reads under a green health check.

### Health: liveness vs readiness

`/v0/livez` takes no lock and reads no state (`lib.rs:1257`) — a long write or compaction holding the writer lock never makes it hang; it is the container liveness target. `/v0/health` is readiness: it loads the published snapshot lock-free and reports `manifest_version`, `epoch`, the `writer: ok|degraded` status (degraded ⇒ 503), and the memtable-bytes gauge (`health_response`, `lib.rs:1227`).

### The Bolt protocol

`namidb-bolt` implements Bolt 4.4/5.0/5.4 with no dependency on the server crate (RFC-022). The PackStream codec (`codec.rs`) is a marker-by-marker translation with a critical pre-auth defense: `decode_inner` threads a `depth` counter and rejects nesting beyond `MAX_NESTING_DEPTH = 128` with a clean `NestingTooDeep` failure (`codec.rs:60,227`), because a malformed message of nested `List`/`Map`/`Struct` markers could otherwise drive the recursive decoder into a non-unwindable worker-thread stack overflow that aborts the process — reachable before authentication. Container sizes are bounded by `DEFAULT_MAX_LEN = 16 MiB`, and the message decoder is fed a lower cap pre-auth (`PRE_AUTH_MESSAGE_BYTES = 64 KiB`) versus `POST_AUTH_MESSAGE_BYTES = 16 MiB` after LOGON (`message.rs:19`).

The per-connection state machine (`namidb-bolt/src/state.rs`) walks `Negotiation → Connected → Authentication → Ready`, branching to `Streaming`/`TxReady`/`TxStreaming`, with `Failed` (only `RESET`/`GOODBYE` recover) and `Defunct`. Result delivery is **demand-driven**: `RUN` replies `SUCCESS { fields }` and buffers rows without emitting them; each `PULL { n }` pops up to `n` buffered rows as `RECORD`s and, if a remainder is left, answers `SUCCESS { has_more: true }` and stays `Streaming` so a driver's fetch_size actually pages a large result; `DISCARD` drops the same rows unsent (`session.rs:660`). Explicit transactions take the writer lock via `lock_owned()` at `BEGIN` and hold it across every in-transaction `RUN` and client think-time until `COMMIT`/`ROLLBACK` (`bolt.rs:950`); in-transaction reads run against `tx.writer.overlay_snapshot()` so they see the transaction's own staged batch (RFC-026). Three timeouts bound that held lock: the acquisition itself is bounded by `NAMIDB_WRITER_LOCK_TIMEOUT`; an idle client past `tx_idle_timeout` is rolled back at the next message boundary; and total transaction lifetime is capped by `NAMIDB_BOLT_MAX_TX_LIFETIME` (default 300 s), enforced at message boundaries (`session.rs:345`). A client that drops mid-transaction is rolled back on the read's `UnexpectedEof` so its staged batch cannot be sealed by an unrelated later commit. The listener defends against connection floods and slowloris: a `Semaphore` of `NAMIDB_BOLT_MAX_CONNECTIONS` permits (default 1024) acquired with `try_acquire_owned` — over the cap the socket is closed, not queued — and a 10 s handshake timeout on the version handshake, the TLS handshake, and every pre-auth read (`bolt.rs:1254`, `session.rs:391`). Auth maps `LOGON {scheme}`: `basic`/`bearer` check credentials against the shared `AuthConfig` (constant-time), other schemes are rejected; Bolt is single-namespace (`principal_for`, namespace-agnostic).

### HTTP surface and DDL interception

`/v0/cypher` parses, then **intercepts schema DDL before planning** — `CREATE/DROP VECTOR INDEX`, `CREATE/DROP FULLTEXT INDEX`, `CREATE CONSTRAINT/INDEX`, and `SHOW` — since these never lower to a `LogicalPlan` (`lib.rs:2043`). DDL is a metadata-only manifest commit that republishes the snapshot; it is gated on the write role and the authz hook, and is rejected inside a Bolt transaction (it cannot be rolled back). Otherwise the handler plans against the published snapshot, and branches on `plan.contains_write()`: writes take the bounded writer lock, run `execute_write_with_deadline`, republish, sample soft backpressure in-lock and sleep *after* releasing; reads borrow the snapshot and run under `execute_with_limits` (per-query deadline + operator row cap). The router adds a global `TimeoutLayer` (`NAMIDB_HTTP_REQUEST_TIMEOUT`, default 120 s) and a `GlobalConcurrencyLimitLayer` (`NAMIDB_HTTP_MAX_CONCURRENCY`, default 1024) (`lib.rs:476`). Backpressure is byte- and L0-based: a committed write whose memtable crosses `memtable_flush_bytes` nudges the flush task via `Notify`; crossing `memtable_stall_bytes` or the L0 high-water mark returns a stall delay applied out-of-lock so throttling hits the request, not the mutex (`after_commit_backpressure`, `lib.rs:344`).

### Multi-tenancy

`--multi-tenant` routes `/:namespace/v0/...` (and unprefixed `/v0/...`) to a `NamespaceRegistry` (`registry.rs`) that lazily opens one `NamespaceState` per tenant — its own `WriterSession`, `SnapshotCell`, catalog cache, `WriterHealth`, and per-namespace flush/compaction/orphan-sweep tasks. Namespace selection is security-load-bearing: `resolve_request_namespace` reads axum/matchit's captured `:namespace` path param (falling back to the `X-NamiDB-Namespace` header, then default), the same value the handler serves, closing the `/v0/v0/...` cross-tenant bypass class (`lib.rs:556`); `require_auth_multi` then rejects a token not scoped to that namespace with 401 *before* the query runs (`lib.rs:1053`, `principal_for_in`). Capacity is bounded by `--max-namespaces` (default 100); at capacity, idle namespaces (measured by a monotonic anchor diff against `--namespace-idle-timeout`, default 1 h) are evicted oldest-first. Eviction sends a `watch` cancellation token so both maintenance loops exit promptly (no zombie second writer) and eagerly calls `prune_shared_caches` for that namespace's prefix — necessary because the SST/NodeView/adjacency caches are process-wide with global budgets shared across tenants. Evicting drops the in-memory session only; ACKed writes are WAL-durable and recovered by `reopen` on next access. JWT scoping mirrors static-token `namespaces`: with `--jwt-namespaces-claim`, the claim (array or single string) must name the requested namespace (RS/ES algorithms only; JWKS refreshed hourly).

### Embedding surfaces

The **Python** client (`namidb-py`, PyO3/maturin) is an in-process embedding: `Client` owns a private tokio `Runtime` and one `Arc<Mutex<WriterSession>>`, and `block_on`s each call; `client.cypher()` commits writes durably before returning, DDL is intercepted the same way (`lib.rs:636`). The **MCP** server (`namidb-mcp`) is deliberately **read-only**: it holds one `WriterSession` (for vault load/`watch_vault` sync) but every tool routes through `run_read_query`, which serves from the published snapshot and rejects any plan where `contains_write()` (`lib.rs:1148`) — 14 tools including graph algorithms, vector/hybrid search, and an escape-hatch `cypher` tool. All surfaces therefore reduce to the same primitive: mutations serialize on the one `WriterSession`; reads run lock-free against an `OwnedSnapshot`.

## 9. Engineering Methodology

This section documents *how the system was built* — its module decomposition, its design process, and the testing and verification discipline that the correctness arguments in the preceding sections rely on. It is included because a technical report intended for teaching should be reproducible as a process, not only as an artefact.

### 9.1 Module decomposition

The engine is a Rust workspace of focused crates, layered so that a dependency edge always points from a higher-level concern to a lower-level one:

| Crate | Responsibility |
|---|---|
| `namidb-core` | Shared value types, identifiers, schema, error taxonomy, int8 quantization. |
| `namidb-storage` | The LSM engine: WAL, memtable, manifest/CAS, SST formats, compaction, janitor, backup, the read path, and the caches. Largest crate (~38 kLOC). |
| `namidb-graph` | The in-memory `Graph` and the analytical kernels (§7). |
| `namidb-ann` | The DiskANN/Vamana build and search, independent of storage (§6). |
| `namidb-query` | The Cypher/GQL parser, logical plan IR, cost-based optimizer, executor, and BM25 scoring (~38 kLOC). |
| `namidb-bolt` | The Bolt wire protocol (PackStream codec, framing, session state machine), with no dependency on the server. |
| `namidb-markdown` | Obsidian/Markdown vault ingestion and optional remote embedders. |
| `namidb-server` | The HTTP + Bolt daemon: routing, auth/JWT/PDP, multi-tenancy, writer recovery, maintenance loops. |
| `namidb-mcp`, `namidb-py`, `namidb-cli` | The MCP server, the PyO3/maturin Python bindings, and the command-line tool. |
| `namidb` | A façade crate re-exporting the stable surface. |
| `namidb-bench` | An LDBC-shaped synthetic benchmark harness. |

The separation of `namidb-ann` and `namidb-graph` from `namidb-storage` is deliberate: the vector index and the graph kernels are pure in-memory algorithms over owned data and are unit-testable without any object store, which is what allows their correctness (recall, determinism, complexity) to be pinned independently of the durability machinery.

### 9.2 RFC-driven development

Every non-trivial change is preceded by a design proposal — a **Request for Comments** — recorded in `docs/rfc/`. The set spans RFC-001 (the storage engine) through RFC-036 (the query beam-width surface); the numbering is chronological and the documents are the primary design record. An RFC states the problem, the considered alternatives, the chosen mechanism, and the invariants it must preserve, before code is written. This report treats the RFCs as the *intent* and the code as the *ground truth*: where they diverge (an RFC describing an earlier design that the implementation moved past), the report follows the code and the divergence is noted. The practical effect for a reader is that each subsystem here can be read alongside its RFC to see both the "why" (the RFC) and the "what, exactly" (this report and the source).

### 9.3 Testing and verification discipline

The engine's correctness rests on four classes of test, in increasing order of what they can catch:

- **Unit tests** co-located with each module, exercising a single mechanism against an in-memory object store (`object_store::memory::InMemory`), so a full LSM lifecycle — write, commit, flush, compact, reopen, recover — runs in milliseconds with no external service.
- **Integration tests** per crate that drive the public surface end to end: parse → lower → optimize → execute for the query engine; open → write → flush → compact → reopen for storage; the full Bolt handshake/framing/codec for the protocol.
- **Equivalence (parity) tests**, the load-bearing class for this system's central tenet (§1.3). Because a secondary index must return exactly what a scan would, the index path and the flat-scan path are run against the same committed state and asserted equal — for vector K-nearest-neighbour (index vs. brute-force cosine), BM25 (index vs. flat scan, including phrase and prefix queries), and factorized vs. flat query execution. A parity test is only meaningful if it confirms the index path was *actually taken*, so these tests additionally assert that the storage layer returned an index answer (rather than silently falling back), or that the optimized plan contains the index operator.
- **Differential and property-style tests** that pin a fast implementation to a reference: the compact-forward triangle counter against a naïve neighbour-pair reference over a pseudo-random multigraph; the streaming compaction merge against the previous materialize-everything merge, byte-for-byte on the output SSTs modulo the UUID names; Brandes betweenness against exact hand-computed values on canonical fixtures.

Determinism is itself a tested property: kernels are asserted to produce identical output across repeated runs (community labels, embedding vectors), and the memtable's persistent map is asserted to share structure (`ptr_eq`) between snapshots with no intervening write.

### 9.4 Continuous integration gates

The CI pipeline (`.github/workflows/ci.yml`) enforces, on every change to the main branch: `cargo fmt` formatting; Clippy with `-D warnings` under the **default feature set** (deliberately *without* the optional features, so a feature-gated item unused by a default build is caught as dead code); the full test suite on Linux and macOS with default features and again with the `vector-index,text-index` features enabled (the configuration the shipped server binary uses); and a minimum-supported-Rust-version (`1.85`) compilation check. The dual clippy/test runs — features off and on — are load-bearing because roughly half the search code is behind `#[cfg(feature = ...)]` gates and is compiled out of the default build.

### 9.5 The 2.0.0 hardening audit

The 2.0.0 release was produced by a systematic audit of the vector/text/graph search stack and the durability path, followed by full remediation. The audit combined an adversarial review — enumerating failure scenarios (crash windows, fenced-writer split-brain, unbounded memory, index-vs-scan divergence) and confirming each against the code — with a mechanical fix-and-verify loop in which every confirmed defect was corrected together with a regression test demonstrated to fail without the fix. The workspace test count grew from 1,545 to roughly 1,700 across the effort. The classes of defect found and closed are documented as fixes throughout §2–§8; the recurring lesson is that in a disaggregated, crash-consistent design the subtle bugs live in the *interaction* between mechanisms — a stalled commit's orphan body versus a concurrent writer's claim (§2), a compaction's index rebuild versus the read snapshot's freshness gate (§4, §6), a background task's cache invalidation versus a sibling tenant's live entries (§4) — which is why this report emphasizes those interactions.

## 10. Evaluation and Benchmarking Methodology

This report does not publish performance results as headline numbers, because for the two subsystems where a single number is most tempting — approximate-nearest-neighbour recall and query throughput — a single number without its full parameter vector is misleading. This section instead documents the *methodology* the engine uses to measure itself, so that a reader can reproduce a result and know exactly what it does and does not claim. The harness lives in `crates/namidb-bench`.

### 10.1 Graph and query workload

The query-side harness is LDBC-shaped (`loader.rs`, `queries.rs`, `runner.rs`): a synthetic social-network graph at a configurable scale factor with the skewed degree distribution and multi-hop navigational queries characteristic of the Linked Data Benchmark Council's Social Network Benchmark. It is used to exercise the executor's join and expansion paths, the cost-based optimizer's plan choices, and the read-path caches under realistic fan-out, rather than to produce a competitive throughput figure.

### 10.2 ANN recall methodology (RFC-031)

The vector index is evaluated against the only question that determines its usefulness: **does recall hold as the corpus grows?** The protocol (`ann_bench.rs`, RFC-031) is deliberately self-contained and reproducible with no external data:

- A synthetic corpus of clustered unit-norm vectors is generated from a fixed seed. The **ground truth** is the exact top-k computed by brute-force distance over the same vectors — the index is measured against the same flat scan the engine falls back to, which is also the compute floor the index must beat.
- The metric is `recall@k = |index_topk ∩ flat_topk| / k`, averaged over a query set, reported together with the full parameter vector `(dim, num_vectors, clusters, spread, ef, k, metric, quantization)`.

The methodology's governing principle, stated in RFC-031, is that **recall on a graph index is a function of its parameters, not a constant**: the same engine measures `recall@10 ≈ 0.68` on a hard, high-dimensional corpus and the `namidb-ann` unit test asserts `recall@10 ≥ 0.90` on an easy clustered one, and *both are correct* — the difference is entirely the workload and the beam width `ef`. Any recall figure is therefore only meaningful accompanied by its parameters, and this report deliberately cites none as a guarantee. RFC-031 separates two tiers explicitly: **synthetic data for continuous integration and iteration** (fast, deterministic, no download), and the external `ann-benchmarks` HDF5 datasets (`sift-128`, `glove-100-angular`, `gist-960`) reserved for a *publishable* recall-vs-QPS curve comparable to other systems. Neither replaces the other.

### 10.3 Quantization arithmetic

A separate track (`vector_recall.rs`) isolates the error introduced by int8 quantization *alone* — the arithmetic of encoding, dequantizing, and the asymmetric quantized distance — from the graph-navigation error, so the two sources of approximation can be attributed independently. This matters because the engine rescan-corrects quantized scores with the exact f32 metric on the candidate pool (§6): the served top-k membership and scores are the exact ones, and quantization affects only which candidates the graph walk visits, i.e. recall, not correctness.

### 10.4 What the measurements are for

The harness is an engineering instrument, not a marketing one: it exists to catch regressions (a recall floor that a code change drops below), to guide parameter defaults (the Vamana degree `R`, build beam `L`, and diversification `α`; the compaction level budgets), and to make the cost of a design decision visible before it ships. A researcher reproducing a NamiDB number should report the parameter vector alongside it and should distinguish an internal synthetic measurement from an external-dataset one, exactly as RFC-031 requires of the project itself.

## 11. References

### 11.1 Design RFCs

The project's design record. Each is at `docs/rfc/<n>-*.md`.

| RFC | Title |
|---|---|
| 001 | Storage Engine Architecture |
| 002 | SST Format — Property Columnar + CSR Adjacency |
| 003 | Read-path ranged reads + Parquet page index |
| 004 | Cypher subset compatibility scope |
| 008 | Logical Plan IR |
| 009 | Write clauses + execution model |
| 010 | Cost-Based Optimizer — Foundation |
| 011 | Predicate Pushdown + Filter Normalization |
| 012 | HashJoin |
| 013 | Parquet predicate pushdown |
| 014 | HashSemiJoin via decorrelation |
| 015 | Projection pushdown / column pruning |
| 016 | Join reorder (DP / greedy) |
| 017 | Factorized intermediate results |
| 018 | CSR-style adjacency materialised in-snapshot |
| 019 | Cross-snapshot NodeView cache |
| 020 | Cross-snapshot edge SST caches |
| 021 | Concurrent reads without the single-writer mutex |
| 022 | Bolt protocol — wire compatibility with Neo4j drivers |
| 023 | `shortestPath` and `allShortestPaths` |
| 024 | Worst-case-optimal join via leapfrog triejoin |
| 025 | Per-label property statistics under id-primary |
| 026 | Read-your-own-writes within a statement and transaction |
| 027 | Multi-level compaction, tombstone GC, and safe space reclamation |
| 029 | Create-only versioned manifest pointer |
| 030 | DiskANN/Vamana vector index |
| 031 | ANN recall@k-vs-QPS benchmark methodology |
| 032 | Filtered ANN pre-filtering (filtered-DiskANN) |
| 033 | Rich nested/array properties and named/sparse/multi-vector indexes |
| 034 | Writer concurrency — the single-writer-per-namespace mutex |
| 035 | Incremental `.vg` maintenance and int8-by-default navigation |
| 036 | First-class query beam-width (`ef`) surface |

### 11.2 External work referenced by the design

- **DiskANN / Vamana** — the graph-based ANN index and its `robust_prune` α-diversification (Subramanya et al., *DiskANN*, NeurIPS 2019). §6.
- **Maximum-inner-product-search reduction** — the norm-augmentation that reduces MIPS to cosine/L2 nearest-neighbour (Bachrach et al., *Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product Spaces*, RecSys 2014). §6.
- **BM25** — the Okapi ranking function with term-frequency saturation and length normalization (Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009). §7.
- **Brandes betweenness** — the O(V·E) betweenness-centrality algorithm (Brandes, *A Faster Algorithm for Betweenness Centrality*, 2001). §7.
- **Louvain** — modularity-optimizing community detection with level aggregation (Blondel et al., 2008). §7.
- **Compact-forward triangle counting** — the O(E^1.5) degree-ordered algorithm (Latapy, 2008). §7.
- **Worst-case-optimal joins** — leapfrog triejoin (Veldhuizen, 2014); the AGM bound (Atserias, Grohe, Marx, 2013). §5.
- **Factorized query processing** — factorized representations of relational results (Olteanu & Závodný, 2015). §5.
- **Reciprocal Rank Fusion** — rank-based fusion of retrieval result lists (Cormack et al., 2009). §6–§7.
- **LSM lineage on object storage** — the monotonic-commit-file-plus-writer-epoch model shared with Delta Lake and SlateDB. §1–§2.

---

## Appendix A: Glossary

| Term | Meaning |
|---|---|
| **SST** | Sorted string table: an immutable, sorted, columnar data object (Parquet for nodes, a native CSR format for edges, bincode for `.vg`/`.ft`). |
| **WAL** | Write-ahead log: immutable append-once segment objects sealing each committed batch before it is applied. |
| **Manifest** | The versioned metadata object listing the SSTs, WAL segments, schema, and indexes that constitute a namespace at one version. |
| **Epoch** | A monotonic per-namespace counter in the manifest; a writer with a lower epoch than the current one is *fenced* and self-excludes on its next commit. |
| **LSN** | Log sequence number: the monotonic, per-namespace durable order of operations; last-LSN-wins resolves a key seen in multiple sources. |
| **Memtable** | The in-memory buffer of committed-but-unflushed writes, backed by a persistent ordered map for O(1) snapshot publication. |
| **Snapshot** | An immutable, consistent, committed point-in-time view a reader executes against without taking the writer lock. |
| **Tombstone** | A deletion marker retained until compaction proves no older data references the key. |
| **Retention horizon** | The oldest manifest version any live reader (or backup pin) is pinned to; objects unreferenced from the horizon to current are reclaimable. |
| **Freshness gate** | The mechanism ensuring an index answer equals a scan of the same committed state — either delta-union or gate-and-fall-back. |
| **Vamana** | The DiskANN graph structure the vector index navigates; built with degree bound `R`, build beam `L`, and diversification `α`. |
| **`ef`** | The beam width of the vector index's greedy search: larger `ef` raises recall and cost. |
| **CSR** | Compressed sparse row: the adjacency layout giving O(deg(v)) neighbour enumeration. |
| **Factorization** | A compressed representation of many-to-many intermediate query results avoiding a Cartesian materialization. |

## Appendix B: Selected Constants and Defaults

Read from the source at 2.0.0. Runtime-tunable values give their environment variable.

| Subsystem | Constant | Value |
|---|---|---|
| WAL | format version; segment/footer magic | 1; `TGWL` / `TGEL` |
| Manifest | stall signature; repair passes; pointer forward-probe bound | 8 rounds; 4; 8192 |
| Node SST | compression; row-group target; page size; write batch | ZSTD-6; 128 Ki rows (`NAMIDB_NODE_SST_ROW_GROUP_ROWS`); 1 MiB; 8192 rows |
| Bloom | bits/key; omit threshold; block size | 10 (~1% FPR); 256 KiB; 256-bit |
| Edge SST | fence-index threshold; fence stride; header magic | 65,536 keys; 256; `TGEDGE\0\0` |
| Format magic | vector `.vg`; text `.ft`; bloom | `NAMIVG03`; `NAMIFT02`; `TGBLOOM\0` |
| Compaction | L1 byte budget; per-level growth; merge chunk | ~8 MiB; 10×; 16 Ki rows (`NAMIDB_COMPACTION_MERGE_CHUNK_ROWS`) |
| Vector index | build defaults `R` / `L_build` / `α` | 64 / 128 / 1.2 |
| BM25 | `k1`; `b` | 1.5; 0.75 |
| Hybrid | RRF `k` | 60 |
| PageRank | damping; max iterations; tolerance | 0.85; 100; 1e-6 |
| Louvain | max levels; sweeps/level; tolerance | 10; 10; 1e-4 |
| FastRP | dimension; iteration weights; sparsity; seed | 256; [0,1,1,1]; s=3; 42 |
| Kernels | deadline poll stride | 4096 iterations |
| Bolt | max nesting depth; max frame; pre/post-auth message cap; connection cap; max tx lifetime | 128; 16 MiB; 64 KiB / 16 MiB; 1024 (`NAMIDB_BOLT_MAX_CONNECTIONS`); 300 s |
| Server | writer-lock timeout; flush/stall bytes; HTTP timeout; HTTP concurrency; max namespaces; idle timeout | 30 s (`NAMIDB_WRITER_LOCK_TIMEOUT`); 64 MiB / 256 MiB; 120 s; 1024; 100; 1 h |
| Caches (process-wide, global budget) | SST body; decoded row-group tier | 256 MiB (`NAMIDB_SST_CACHE_BUDGET_MIB`); 256 MiB (`NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB`) |

---

<sub>NamiDB is developed by NamiDB, Inc. (Delaware, USA) and licensed under the Business Source License 1.1. This report describes version 2.0.0. Corrections and questions: the source is the authority; every claim here is traceable to a file path cited inline.</sub>
