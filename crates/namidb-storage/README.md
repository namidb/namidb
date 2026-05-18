# namidb-storage

LSM storage engine for [NamiDB](https://github.com/namidb/namidb) on
object storage. Single-writer-per-namespace with epoch fencing,
Parquet node SSTs, custom edge-SST format with CSR adjacency, and
tiered caches for cross-snapshot reuse.

This crate is the **source of truth on disk**. Coordination among
writers and readers is provided by S3 conditional writes
(`If-Match` / `If-None-Match` / ETag) rather than by an external
consensus service.

## What lives here

- **Write path** — `WriterSession`, batch building, WAL append,
  manifest CAS, memtable application
  ([RFC-001](../../docs/rfc/001-storage-engine.md)).
- **Flush / compaction** — memtable to L0 SSTs, L0 to L1 merge.
- **Read path** — `Snapshot` over the manifest, ranged reads
  ([RFC-003](../../docs/rfc/003-read-path-ranged-reads.md)),
  predicate-aware scans.
- **SST format** — Parquet node SSTs and custom edge SSTs with CSR
  adjacency ([RFC-002](../../docs/rfc/002-sst-format.md)).
- **Caches** — `AdjacencyCache` (CSR,
  [RFC-018](../../docs/rfc/018-csr-adjacency.md)), `NodeViewCache`
  ([RFC-019](../../docs/rfc/019-node-view-cache-shared.md)),
  `SstCache` ([RFC-020](../../docs/rfc/020-edge-sst-caches.md)).
- **Recovery** — WAL replay on cold open, fence collection.
- **Parquet ingest** — bulk-load nodes from a Parquet file through
  the public writer surface.

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for design rationale.

## License

[Business Source License 1.1](../../LICENSE) — © Fonles Studios, Corp.
