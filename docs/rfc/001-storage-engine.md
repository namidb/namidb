# RFC 001: Storage Engine Architecture

**Status:** draft
**Author(s):** NamiDB founding team

## Summary

NamiDB stores property-graph data in an LSM-tree-shaped storage engine whose **source of truth is object storage** (S3 and compatibles). Coordination among writers and readers is provided conditional writes (`If-Match` / `If-None-Match` / ETag) rather than by an external consensus service. The engine is single-writer per namespace, with **epoch fencing** enforced by manifest CAS. Reads are served from a three-tier hybrid cache (memory + NVMe + optional S3 Express One Zone) over the immutable, columnar SSTs that the storage layer produces.

This RFC defines the on-disk layout, the manifest protocol, the write path (WAL → memtable → SST flush), the read path, and the compaction strategy.

## Motivation

Existing graph databases fall into two camps:

1. **Single-node embedded / RAM-bound** (Kuzu, LadybugDB, HelixDB, Memgraph, FalkorDB). These store data on local disk or RAM. None can scale a single namespace beyond one machine, none has a "your S3 bucket is the source of truth" story, and none supports scale-to-zero pricing.
2. **Shared-nothing distributed** (Neo4j, TigerGraph, NebulaGraph, Neptune). These require operational expertise to run, are expensive, and tie compute to storage.

We want the architecture proven by **turbopuffer** (vector), **SlateDB** (KV), **WarpStream** (Kafka), and **Neon** (Postgres) — compute and storage fully separated, object storage as durable substrate, single-writer fencing via cloud CAS — applied to **property graphs**. As of May 2026 nobody has shipped this.

Hard requirements:

- **Durability of S3** (11 nines) with no extra coordination service.
- **Scale-to-zero per namespace** for SaaS economics.
- **Cold query < 500 ms p50** at 10M edges.
- **Warm query < 10 ms p50.**
- **Snapshot isolation** for reads + ability to branch a graph at a point in time.
- **Single binary** to run in embedded, server, and SaaS modes.

## Design

### Tiered storage and access model

```
┌─────────────────────────────────────┐
│ Memory cache (Arrow batches) │ Sub-ms p50, w-TinyLFU / SIEVE
├─────────────────────────────────────┤
│ NVMe disk cache (foyer-rs) │ ~1-10 ms p50
├─────────────────────────────────────┤
│ S3 Express One Zone (optional) │ Single-digit ms, hot tier
├─────────────────────────────────────┤
│ S3 Standard / R2 / GCS / MinIO │ Source of truth, 11-nines durability
└─────────────────────────────────────┘
```

### Logical layout in object storage

```
<bucket>/<namespace>/
├── manifest/
│ ├── current.json # tiny pointer file: { "version": v, "etag": "..." }
│ └── v00000001.json # immutable manifest snapshot
│ └── v00000002.json
│ └── ...
├── wal/
│ ├── 00000001.wal # 64MB segments, append-only
│ ├── 00000002.wal
│ └── ...
├── sst/
│ ├── level0/
│ │ ├── 01J5XY...-nodes-Person.parquet
│ │ ├── 01J5XY...-edges-KNOWS.csr
│ │ └── 01J5XY...-vector-Document.lance
│ ├── level1/
│ │ └── ...
│ └── ...
└── snapshots/
 └── 2026-01-01T00:00:00Z.json # optional named snapshots / branches
```

Within a namespace:

- Filenames are **ULIDs** (`uuid::Uuid::now_v7()`) for natural ordering by creation time.
- Each SST belongs to one **(label, edge type, or vector index, level)** combination.
- Manifests are **fully self-describing**: they list every SST currently part of version `v`, along with statistics (row count, byte size, key range, bloom filter, partition tag, histogram).

### Manifest protocol (the heart of the design)

The manifest is the single object that determines "what is the current state of the database". All writers race to update it; only one wins per epoch.

#### Manifest file format (`v<N>.json`)

```jsonc
{
 "version": 42,
 "epoch": 7,
 "writer_id": "uuid-of-writer-process",
 "created_at": "2026-01-15T10:00:00.000Z",
 "schema_version": 11,
 "labels": [
 {
 "name": "Person",
 "node_id_type": "Uuid",
 "properties": [
 { "name": "name", "type": "Utf8", "nullable": false },
 { "name": "age", "type": "Int32", "nullable": true }
 ]
 }
 ],
 "edge_types": [
 {
 "name": "KNOWS",
 "src_label": "Person",
 "dst_label": "Person",
 "properties": [{ "name": "since", "type": "Date32", "nullable": true }]
 }
 ],
 "ssts": [
 {
 "id": "01J5XY7K...",
 "kind": "Nodes",
 "label": "Person",
 "level": 0,
 "path": "sst/level0/01J5XY7K...-nodes-Person.parquet",
 "size_bytes": 134217728,
 "row_count": 1048576,
 "min_key": "00...",
 "max_key": "ff...",
 "created_at": "2026-01-15T10:00:00Z"
 }
 ],
 "wal_segments": [
 { "id": "00000042.wal", "path": "wal/00000042.wal", "last_lsn": 1234567 }
 ]
}
```

#### `current.json` (pointer file)

```jsonc
{
 "version": 42,
 "manifest_path": "manifest/v00000042.json",
 "manifest_etag": "etag-of-v42-object"
}
```

#### CAS protocol for committing a new manifest

When the writer wants to advance the database from version `v` to `v+1`:

1. Read `current.json` (with its ETag) → `current_etag`.
2. Read `manifest/v<v>.json` → previous state.
3. Build the new manifest in memory, incrementing `version` to `v+1` and updating `epoch` if needed.
4. **PUT** `manifest/v<v+1>.json` with `PutMode::Create` (i.e. `If-None-Match: *`). If this fails, another writer has the same version assigned; abort, reload, retry.
5. **PUT** the pointer. _Amended by RFC-029:_ the pointer is now a Create-only
   versioned family — **PUT** `manifest/pointer/p<v+1>.json` with
   `PutMode::Create` (`If-None-Match: *`), and the current pointer is the
   highest `N` present, found via LIST. The original design wrote
   `current.json` with `PutMode::Update` (`If-Match: <current_etag>`); that
   depended on the spottily-supported `If-Match` overwrite, so it was replaced
   with the same PUT-if-absent primitive step 4 uses. If the create fails,
   another writer raced ahead; abort, reload, retry. See RFC-029.

This sequence ensures:

- The manifest file for any given version `v` is written **at most once** (write-once contents).
- The `current.json` pointer is the linearization point. Whoever wins the `If-Match` swap is the sole owner of version `v+1`.

#### Epoch fencing

Each writer process picks an `epoch` at startup, taken from `current_manifest.epoch + 1` and immediately committed via the CAS protocol above (a "zero-op" manifest update that only bumps epoch). After that, every other writer that tries to advance the version against this `current.json` will lose the CAS race and discover the new epoch on retry — at which point its operations must be rejected by the local writer's epoch check.

In code:

```rust
pub struct WriterFence {
 pub epoch: u64,
 pub writer_id: Uuid,
}

impl WriterFence {
 pub fn assert_alive(&self, current_epoch: u64) -> Result<()> {
 if current_epoch > self.epoch {
 return Err(Error::Fenced { mine: self.epoch, current: current_epoch });
 }
 Ok(())
 }
}
```

Every WAL append, every memtable flush, every SST commit calls `assert_alive` against the latest known manifest before its CAS. This is how single-writer is enforced **without** Raft, ZooKeeper, or any local file lock.

### Write path

```
client write API
 │
 ▼
[1] WriterFence.assert_alive(current_epoch)
 │
 ▼
[2] Buffer in WAL batcher (group commit, 100ms or 1MB)
 │
 ▼
[3] WAL flush: PUT wal/<next>.wal segment to object store
 │ (PutMode::Create on first byte to detect concurrent writer)
 │
 ▼
[4] Apply to in-memory memtable (Arrow-backed skiplist)
 │
 ▼
[5] Acknowledge to client (durability == WAL acknowledged)
 │
 ▼ (in background, when memtable > threshold)
[6] Freeze memtable → flush to SST(s) in level 0
 │
 ▼
[7] Manifest CAS: add new SSTs, mark WAL segments as flushed
 │
 ▼ (in background, scheduled)
[8] Compaction worker: merge L0 → L1, manifest CAS, GC obsolete SSTs
```

WAL durability is the user-facing acknowledgement. After a WAL group commit returns success, the data is durable.

### Read path

```
query API
 │
 ▼
[1] Read current.json → snapshot version v
 │
 ▼
[2] Optimizer builds plan against manifest v (immutable for this query)
 │
 ▼
[3] Operators issue async fetches against cached SSTs / WAL
 │ (foyer tries memory → disk → S3 Express → S3 Standard)
 ▼
[4] Stream Arrow batches to client
```

The **manifest is immutable for the lifetime of the query**, giving snapshot isolation for free. SSTs are immutable until GC. The only mutable thing in the system is `current.json`.

### Branching (Neon-style)

Branching is "named manifest aliases":

```
manifest/branches/my-branch.json → { "version": 42, "manifest_path": "manifest/v00000042.json" }
```

A branch shares SSTs with its parent (CoW); new writes go to SSTs owned by the branch. GC respects branch references.

### Compaction strategy

- **Default: leveled compaction.** Level 0 = output of memtable flush (overlapping ranges allowed). Level `L > 0` is partitioned by key range (no overlap).
- **Trigger:** L0 has > 4 SSTs, or `bytes(L_i) > 10 * bytes(L_{i-1})`.
- **Worker:** stateless. A compaction worker reads SSTs, merges them, writes new SSTs, then does a manifest CAS to swap. If the CAS fails (manifest moved underneath), the worker discards its output and retries.
- **GC:** SSTs unreferenced by `current.json` for > `retention_window` (default 24h) are deleted. Branches extend retention for their snapshots.

### Failure modes

| Failure | Behavior |
|---|---|
| Writer crashes mid-WAL flush | WAL segment is partial; recovery treats the last malformed record as torn and discards it. Manifest unchanged → no data visible. |
| Writer crashes after WAL flush, before manifest CAS | WAL segment is durable; on next startup, recovery replays WAL segments not referenced by manifest into a fresh memtable. |
| Two writers race | Loser fails CAS, refreshes manifest, discovers higher epoch, fences itself, fails subsequent writes. |
| Stale reader | Sees old manifest version `v`; queries are still consistent against `v` (snapshot isolation). New `current.json` reads pick up newer versions. |
| Corrupted SST | CRC mismatch detected on read; query fails with `CorruptedSst { sst_id }`. SSTs can be re-derived from WAL replay if WAL retention permits. |

### Concurrency for readers

Readers are **lock-free**:

- Each query opens a snapshot of the manifest at query start.
- All SSTs referenced by that manifest are guaranteed to exist until GC reclaims them (GC respects active queries via ref counting at the cache layer).
- No locking against the writer; manifest swap is atomic via CAS.

### Why not just use SlateDB?

SlateDB is a KV store. We need:

- **Property graph schema** (typed nodes, typed edges, label-scoped indexes).
- **CSR adjacency layout** in SSTs (not generic KV).
- **Vector index integration** (Lance v2 format).
- **Graph-shaped statistics** (degree distributions, label histograms) for the optimizer.
- **Multi-SST commit atomicity** so a single graph mutation can update nodes + edges + vector index in one manifest.

We borrow SlateDB's protocol shape (WAL → memtable → SST → manifest CAS) and reimplement it with graph-aware SST format. SlateDB will be used in tests as a baseline.

## Alternatives considered

### A. Local disk + replication (Memgraph / Neo4j shape)

Rejected: forces operators to run replicas, lose cloud economics, lose scale-to-zero.

### B. Postgres + extension (Apache AGE, pg-ivm)

Rejected: inherits Postgres operational story; no path to S3-native; column store via Citus is awkward.

### C. ClickHouse-shaped MergeTree

Rejected: MergeTree is optimized for OLAP scans, not multi-hop graph traversals. CSR adjacency does not fit naturally.

### D. Raft / paxos for coordination

Rejected: adds operational complexity and a separate failure domain. S3 conditional writes give us linearizable CAS for free.

## Drawbacks

1. **S3 PUT latency floor (~30-100ms Standard).** Writes feel slower than local-disk databases. Mitigations: group commit (100ms/1MB), optional S3 Express One Zone tier (single-digit ms).
2. **Single-writer per namespace.** Genuinely multi-master writes are not supported in v1. For most workloads (especially analytical / KG / RAG) this is fine; for high-throughput OLTP it isn't.
3. **Compaction write amplification on S3 costs money.** Tuning compaction policy + columnar compression is critical; we will benchmark continuously.
4. **CAS livelock under high writer contention** for the same namespace. Mitigation: only one writer per namespace by design; concurrent CAS losers fence themselves quickly.

## Open questions

- Bloom filter format inside the manifest vs as side-car files per SST.
- WAL segment size: 64MB or 16MB or 4MB? Smaller = lower commit latency, more PUTs (more $). Bench-driven.
- Compression level for WAL: `zstd -3` (default) vs uncompressed (latency win, $ loss).
- Whether to support multi-writer with merge semantics for CRDT-friendly use cases (agent memory). Probably v2.
- Manifest format: JSON now (simple) vs Arrow IPC (smaller, faster) — switch when manifest hits ~10MB.

## References

- Verbitski et al., **Amazon Aurora** (SIGMOD 2017).
- Dageville et al., **Snowflake** (SIGMOD 2016).
- Armbrust et al., **Delta Lake** (VLDB 2020).
- **SlateDB design overview**, https://slatedb.io/docs/design/overview/.
- **turbopuffer architecture**, https://turbopuffer.com/docs/architecture.
- **turbopuffer object-storage queue** blog, https://turbopuffer.com/blog/object-storage-queue.
- Jin et al., **Kùzu** (CIDR 2023).
- Leis et al., **Morsel-driven parallelism** (SIGMOD 2014).
- Neumann & Freitag, **Umbra** (CIDR 2020).
- AWS, **S3 conditional writes** (Aug + Nov 2024 launches).
- AWS, **Kafka KIP-1150 Diskless Topics** (Mar 2026).
