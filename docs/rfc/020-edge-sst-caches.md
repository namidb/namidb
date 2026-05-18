# RFC 020: Cross-snapshot edge SST caches

**Status:** accepted
**Author(s):** Matías Fonseca <info@namidb.com>
**Supersedes:** none

## Summary

Two cross-snapshot caches in `SstCache` that close the gate at
SF10. Both work the same way: `Arc<HashMap<absolute_path, Arc<T>>>`
guarded by a `Mutex`, populated lazily on first read of each SST,
shared across every `Snapshot` the namespace emits. SSTs are
immutable per UUIDv7-keyed path so cached entries never go stale.

- **`edge_streams: HashMap<String, Arc<EdgeStreamBundle>>`** —
 decoded `__overflow_json` + every declared property column
 (`Vec<Option<String>>` per column). Eliminates the `O(edge_count)`
 zstd-decompress + JSON-parse done on every `edge_lookup_via_sst`
 call.
- **`edge_readers: HashMap<String, Arc<EdgeSstReader>>`** — parsed
 header + footer + fence index + precomputed `cumulative_edges`
 prefix sum. Eliminates the `O(edge_count)` partner-block walk done
 by `EdgeSstReader::open` on every call.

Together they take `edge_lookup_via_sst` from `O(edge_count)` to
`O(deg + log edge_count)` in the warm path, which is what shipping
plan-aware routing and the gate at SF10 demanded.

## Motivation

After plan-aware routing closed the property caveat (queries that read
`r.prop` route through the SST path, queries that only need topology
route through CSR), the SST path became a hot path for any query whose
plan reads a relationship's properties.

Profile data from `NAMIDB_PROFILE_DUMP=1` on IC07 at SF1 + SF10
showed every `edge_lookup_via_sst` call doing two pieces of
`O(edge_count)` work:

1. **`read_overflow_strings` + `load_declared_streams`** — pulled
 every property stream off the SST, zstd-decoded each one, parsed
 JSON for every row. For LIKES at SF1 (100K edges) this was
 ~1.4 ms/call. The original comment on the code already named the
 fix: *"the foyer-rs follow-up will cache the parsed vector per SST
 id"*.
2. **`EdgeSstReader::open`** — walked every partner block to build
 the `cumulative_edges: Vec<u64>` prefix sum that the binary search
 in `EdgeSstReader::lookup` indexes into. For LIKES at SF10 (1M
 edges) this dominated the call.

The edge-stream cache added (1). The edge-reader cache added (2).
Together they cut IC07 from 9942 µs to 2262 µs at SF10 (4.4×
warm-path speedup, gate ratio 7.70× → 1.75×).

## Design

### Data shape

```rust
// crates/namidb-storage/src/cache.rs

pub struct EdgeStreamBundle {
 pub overflow: Option<Vec<Option<String>>>, // RFC-002 __overflow_json
 pub declared: Vec<(String, Vec<Option<String>>)>, // RFC-002 §3.2.7
}

pub struct SstCache {
 inner: Arc<foyer::Cache<String, Bytes>>, // body cache
 metadata: Arc<Mutex<HashMap<String, Arc<ParquetMetaData>>>>, // RFC-003
 edge_streams: Arc<Mutex<HashMap<String, Arc<EdgeStreamBundle>>>>,
 edge_readers: Arc<Mutex<HashMap<String, Arc<EdgeSstReader>>>>,
 stats: Arc<CacheStats>,
}
```

### Lookup flow

```rust
async fn edge_lookup_via_sst(
 &self,
 edge_type: &str,
 key: NodeId,
 direction: EdgeDirection,
) -> Result<EdgeListView> {
 for idx in candidates {
 let desc = &self.manifest.manifest.ssts[idx];
 if !self.bloom_admits(desc, &key_bytes).await? { continue; }
 let absolute = format!("{}/{}", self.paths.namespace_prefix(), desc.path);

 // Arc<EdgeSstReader> from cache (build only on miss).
 let reader = self.fetch_edge_reader(&absolute).await?;

 let Some(lookup) = reader.lookup(&key_bytes)? else { continue; };

 // Arc<EdgeStreamBundle> from cache (decode only on miss).
 let streams = self.fetch_edge_streams(&absolute, edge_type, &reader)?;

 // ... O(deg) loop over lookup.partners, decoding from streams.* ...
 }
 Ok(EdgeListView { edges: /* ... */ })
}
```

### Helper functions

Both helpers live on `Snapshot`. They short-circuit on cache hit and
do the expensive work + insertion on miss. Multiple callers can race
on miss (no per-key locking) — the slow second writer's insert
simply overwrites the same `Arc`, which is harmless because the data
is content-addressable by SST path.

```rust
async fn fetch_edge_reader(&self, absolute: &str) -> Result<Arc<EdgeSstReader>> {
 namidb_core::profile_scope!("Snapshot::fetch_edge_reader");
 if let Some(cache) = self.cache.as_ref() {
 if let Some(reader) = cache.get_edge_reader(absolute) {
 return Ok(reader);
 }
 }
 let body = self.fetch_bytes(absolute).await?;
 let reader = Arc::new(EdgeSstReader::open(body)?);
 if let Some(cache) = self.cache.as_ref() {
 cache.insert_edge_reader(absolute.to_string(), reader.clone());
 }
 Ok(reader)
}

fn fetch_edge_streams(&self, absolute: &str, edge_type: &str, reader: &EdgeSstReader)
 -> Result<Arc<EdgeStreamBundle>>
{
 namidb_core::profile_scope!("Snapshot::fetch_edge_streams");
 if let Some(cache) = self.cache.as_ref() {
 if let Some(bundle) = cache.get_edge_streams(absolute) {
 return Ok(bundle);
 }
 }
 let declared_property_names = /* read from manifest schema */;
 let bundle = Arc::new(EdgeStreamBundle {
 overflow: reader.read_overflow_strings()?,
 declared: load_declared_streams(reader, &declared_property_names)?,
 });
 if let Some(cache) = self.cache.as_ref() {
 cache.insert_edge_streams(absolute.to_string(), bundle.clone());
 }
 Ok(bundle)
}
```

### Memory footprint

- **`EdgeStreamBundle`**: roughly `n_edges × n_declared_columns × avg_json_size`.
 For LIKES at SF10 with one declared `creationDate` (Int64): 1M × 1 × ~10 B
 JSON = ~10 MB per SST. The overflow column is empty when all props are
 declared.
- **`EdgeSstReader`**: ~8 B per edge for `cumulative_edges` plus the SST
 body's `Bytes` refcount. For SF10 LIKES that is ~8 MB per SST.

The total across the LDBC SF10 dataset (7 edge types × ~1 SST each
post-bulk-load) is below 100 MB — comfortably below the default
`NAMIDB_SST_CACHE_BUDGET_MIB=256`.

### Invalidation

None needed. SST paths are UUIDv7-derived and never overwritten;
compaction emits new SSTs and atomically swaps the manifest. A
cached entry whose backing SST was compacted away is harmless dead
weight in the HashMap. Eviction-by-LRU is a TODO when the cache size
becomes a real concern.

## Alternatives considered

### A. Foyer hybrid cache for everything

The `SstCache.inner` already uses `foyer::Cache` for raw
bodies. Putting `Arc<EdgeSstReader>` into the same `foyer` cache
would give automatic eviction-by-LRU and weight accounting.

**Rejected for v0**: `foyer::Cache` requires `Send + Sync + 'static`
values with `Weighter` traits. `Arc<EdgeSstReader>` is `Send + Sync`
but the weighter needs to read its `cumulative_edges.len()` — doable
but introduces a tighter coupling between the cache and reader
internals. The plain `Mutex<HashMap>` is 30 lines of code and
matches the lifetime story (immutable-per-path).

### B. Cache the decoded streams inside `EdgeSstReader` via `OnceCell`

Make `read_overflow_strings` and `read_declared_property_strings`
memoize their results inside the reader itself. Combined with B's
reader cache, the streams come along for the ride.

**Rejected**: would require changing the `EdgeSstReader` public API
from `read_*` returning `Result<Option<Vec<...>>>` to either
`Result<&Option<Vec<...>>>` (lifetime ties results to reader borrow)
or owning + `Arc<Vec<...>>`. The two-cache approach keeps the reader
side stateless and the cache responsibility scoped to one module.

### C. Compaction-side baked layout (RFC-005 follow-up)

If the SST writer pre-built the `cumulative_edges` prefix sum and
serialised it into the footer as a separate section, `open` would be
`O(section_read)` instead of `O(edge_count)`.

**Deferred** but not rejected. The cache makes the warm path free
today; the on-disk layout change is the right v1 once the per-SST
size grows beyond what fits comfortably in RAM. It is a write-time
+ format-version change.

## Drawbacks

- **Memory cost**: unbounded HashMap maps grow until the namespace is
 closed. For long-running multi-tenant servers we will need LRU
 eviction tied to `SstCache.inner`'s weighter. Tracked as a
 follow-up.
- **Race-on-miss**: two concurrent readers that both miss the cache
 will both decode + insert. The work is duplicated but the result
 is identical and the cache state stays consistent (the second
 writer overwrites the first's `Arc` with content-identical data).
 No correctness issue, minor wasted CPU.
- **Cold first query is unchanged**: the first SST scan still pays
 the `O(edge_count)` decode + reader build. Pre-warming on
 `WriterSession::open` would amortise this, but it complicates the
 open path and is best left as an optional follow-up.

## Open questions

- **LRU eviction tied to the SST body cache budget.** When the
 body cache (foyer) evicts an SST body, should `edge_streams[k]`
 and `edge_readers[k]` evict in lockstep? Probably yes, but it
 requires foyer eviction callbacks we have not wired yet.
- **`NodeSstReader` analogue.** Nodes go through Parquet which has
 its own metadata cache (RFC-003). Is there a `lookup_node_via_sst`
 hot path that would benefit from a third cache? Profiling so far
 says no — the L1/L2 `NodeViewCache` covers the node side.

## Bench impact

Before S17.3 + S18.B (SF10, IC07, p50 of 3 params):

```
Query NamiDB p50 Kùzu p50 Ratio
IC07 9942 µs 1292 µs 7.70x ← FAIL gate (2x)
```

After both caches (default ON, no other changes):

```
Query NamiDB p50 Kùzu p50 Ratio
IC07 2262 µs 1292 µs 1.75x ← PASS gate (2x)
```

Other queries (IC02 / IC08 / IC09) are unaffected because their
plans either avoid the SST path (CSR routing) or read
nodes more than edges.

Test count: +0 (caches are perf, no semantics change; the existing
LDBC + storage unit tests cover correctness).
