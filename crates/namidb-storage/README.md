# namidb-storage

The LSM storage engine for [NamiDB](https://github.com/namidb/namidb) on
object storage. One writer per namespace with epoch fencing, Parquet
node SSTs, a custom edge-SST format with CSR adjacency, and tiered caches
for cross-snapshot reuse.

This crate is the **source of truth on disk**. Coordination between
writers and readers comes from a single object-store conditional-write
primitive — PUT-if-absent (`If-None-Match: *`, `PutMode::Create`; RFC-029)
on write-once manifest bodies, versioned pointers, and WAL segments —
rather than an external consensus service. Conditional *overwrite*
(`If-Match`) is **not** required; the read path uses ETags only for
conditional GETs.

## What lives here

- **Write path.** `WriterSession`, batch building, WAL append, manifest
  CAS, memtable application
  ([RFC-001](../../docs/rfc/001-storage-engine.md)).
- **Flush and compaction.** Memtable into L0 SSTs, then L0 into L1.
- **Read path.** `Snapshot` over the manifest, ranged reads
  ([RFC-003](../../docs/rfc/003-read-path-ranged-reads.md)), and
  predicate-aware scans.
- **SST format.** Parquet node SSTs plus custom edge SSTs with CSR
  adjacency ([RFC-002](../../docs/rfc/002-sst-format.md)).
- **Caches.** `AdjacencyCache` (CSR,
  [RFC-018](../../docs/rfc/018-csr-adjacency.md)), `NodeViewCache`
  ([RFC-019](../../docs/rfc/019-node-view-cache-shared.md)), and
  `SstCache` ([RFC-020](../../docs/rfc/020-edge-sst-caches.md)).
- **Recovery.** WAL replay on cold open, fence collection.
- **Parquet ingest.** Bulk-load nodes from a Parquet file through the
  public writer surface.

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for the design rationale.

## License

[Business Source License 1.1](../../LICENSE), © NamiDB, Inc.
