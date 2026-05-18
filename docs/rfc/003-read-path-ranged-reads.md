# RFC 003: Read-path ranged reads + Parquet page index

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Supersedes:** —

## Summary

Replace the full-body `object_store::get()` that every cold `lookup_node`/`edge_lookup`
issues today with a byte-ranged fetch driven by the Parquet **page index** and the
existing per-row-group min/max stats. The goal is to bring cold `lookup_node` p50 on
real S3 (50–100 MB/s, ~1 ms RTT from a co-located EC2 instance; far worse from a
developer laptop) inside the the envelope of `<500 ms p50` at 10 M nodes —
which the in-process / LocalStack bench cannot exercise because localhost bandwidth
hides the issue.

## Motivation

A previous iteration closed the bench gate against LocalStack: row-group pruning on the
`node_id` column reduced per-lookup decode from O(rows_in_sst) to
O(rows_per_row_group), and the resulting numbers at 10 M nodes are:

| Metric | Target | Measured (LocalStack, MacBook) |
|---|---|---|
| Cold `lookup_node` p50 | <500 ms | **381 ms** |
| Warm `lookup_node` p50 | <10 ms | **9.27 ms** |

Both gates pass — but the cold number is misleading because `Snapshot::get_sst_body`
still fetches the **entire** Parquet body (currently 300–500 MB for a 10 M-node SST
with zstd compression). LocalStack on a single host moves that in ~350 ms; real S3
moves it in 2–10 s depending on co-location. The same code path against
`s3.us-east-1.amazonaws.com` from a developer laptop would consistently violate the
gate, even though the test harness reports green.

The root mismatch is structural: a point lookup needs ~tens of KB of column data
from a single row group, not the whole SST. The Parquet 2.0 page index gives us
exactly that — per-column-chunk per-page min/max offset + length — but we currently
ignore it.

Cost of doing nothing: any production deploy against real S3 ships with a hidden
~10× regression on cold lookups vs the bench gate. That blocks the SaaS
demo and the public launch.

## Design

### Surface change

`NodeSstReader::open` and `EdgeSstReader::open` today take `body: Bytes`. The new
path takes an `Arc<dyn ObjectStore>` + `Path` + `ObjectMeta` (size known from the
manifest descriptor) and uses `parquet::arrow::async_reader::ParquetObjectReader`
under the hood. The existing `Bytes`-backed constructors stay for the in-process
test path and for the eager `scan_label` use case (which already needs every row
group).

```rust
// New constructor (additive).
impl NodeSstReader {
 pub async fn open_async(
 label: LabelDef,
 store: Arc<dyn ObjectStore>,
 path: Path,
 size_hint: u64,
 ) -> Result<NodeSstAsyncReader> { /* ... */ }
}
```

The async reader exposes a parallel `targeted_scan_async(&[u8; 16]) -> RecordBatch`
that:

1. **Footer fetch.** `ParquetObjectReader` issues one `get_range` for the trailing
 ~8 KB of the SST to read the Parquet footer + column-chunk metadata. For a 500 MB
 SST this is ~one round-trip and ~8 KB transferred.
2. **Row-group pruning.** Same min/max stats check we already have in
 `targeted_scan`. Pick the single row group that straddles the target key (writer
 guarantees strict ascending `node_id` so there is at most one).
3. **Page index fetch.** If `with_page_index(true)`, the reader fetches the
 `OffsetIndex` + `ColumnIndex` for the chosen row group (~few KB). Combined with
 the per-page min/max from the column index, we identify the single data page in
 the `node_id` column that contains the target.
4. **Page fetch.** A single `get_range` of the chosen page's bytes (~1–8 KB
 depending on rows-per-page). Decode, find the row offset within the page,
 project the same row offset across the other columns' pages — each is one more
 `get_range`. For a `Person` label with ~6 declared properties + 2 system columns,
 that's 8 ranged GETs of ~1–8 KB each, or a `get_ranges` batched call.

Total wire footprint per cold lookup: **~50–100 KB** (vs ~500 MB today) and **3–4
round trips** (footer, page index, batched column pages). On S3 us-east-1 from
EC2 (~1 ms RTT), that's ~5–20 ms. From a laptop (~30 ms RTT), ~100–150 ms — both
inside the 500 ms gate with comfortable margin.

### Cache integration

The current `SstCache` keys on the full path and stores the entire body. Under the
new design we shift to **range-keyed caching**: keys become `(path, offset, length)`
or a normalised `(path, kind)` for the three structurally-fixed regions:

- `<path>:footer` — the trailing footer + column metadata block (size known after
 the first fetch).
- `<path>:row_group_<rg_idx>:column_<col_idx>` — per-column-chunk pages for the
 hot row group.

Warm lookups against the same SST and same row group hit memory without re-fetching.
This is essentially a buffer pool keyed by Parquet's logical units instead of by
file. Foyer continues to back it; the `weighter` adds the key length plus the value
length and the budget stays in real bytes.

### Edge SST counterpart

The edge SST format is custom (RFC-002 §3) and already ships a fence-pointer index
for `key_count > 65 536`. The same idea applies: today `EdgeSstReader::open` reads
the full body; the async variant reads the footer + fence index + the per-key
partner block. Wire format is unchanged — only the reader navigates differently.

### Manifest descriptor extension

`SstDescriptor.size_bytes: u64` already exists in the manifest. No schema change
needed; the reader passes that as the `size_hint` so `ParquetObjectReader` can
position the trailing footer read without a HEAD request.

## Alternatives considered

### A. Persist a separate "index" side-car per SST

A `.idx` blob with `node_id → (row_group, offset)` mapping, written at flush time.
Cold lookup = 1 GET of the side-car + 1 GET of the chosen row group. Rejected: the
side-car would essentially duplicate the Parquet column index, and we'd carry two
sources of truth that must stay in sync. The Parquet page index is already on disk
inside the body — re-using it is free.

### B. Maintain a sorted in-memory key→row-group map per SST

Build it on `open()` and cache. Cold lookup pays one full-SST decode the first
time, warm is instant. Rejected: the first lookup is what we're trying to fix.
Building the map requires reading the footer + column index anyway, so we may as
well consume that information directly instead of caching it in a parallel
structure.

### C. Smaller row groups (e.g., 4 K rows)

Today's row group is 128 K rows. Smaller groups would amortise less per-group
overhead and let us decode less per pruned hit. Rejected as a complete solution:
ratio improvement is linear in the row-group shrink but at some point per-group
metadata cost dominates the body. Real fix is page-level granularity, not finer
row groups.

### D. Materialise hot keys into a separate SST per layer

LSM-style "block index" promoted to its own file. Rejected: adds a writer-side
component (when to promote? what to evict?) and another manifest descriptor. The
Parquet page index already gives us per-page granularity for free; promoting hot
keys is premature.

## Drawbacks

1. **Two read paths to maintain.** The async ranged path coexists with the eager
 `Bytes`-backed path used by `scan_label` / `scan_edge_type` / compaction. We
 accept the surface area because compaction genuinely needs every row group and
 would issue worse access patterns if forced through the ranged reader.
2. **Foyer cache keying changes.** Existing tests that assert `SstCache.usage() > 0`
 after a warm cycle keep working (the cache holds page bytes instead of body
 bytes) but the bytes-per-entry distribution shifts dramatically — smaller
 entries, more of them. Eviction tuning may need a second pass.
3. **Round-trip count on S3.** A cold lookup goes from 1 wide GET to ~3 narrow
 GETs. For backends with HEAD+GET RTT penalties (some self-hosted gateways) this
 could regress wall-clock time despite the bandwidth win. Mitigation: support
 `object_store::get_ranges` (which coalesces) and benchmark explicitly against
 real S3 + LocalStack before declaring victory.
4. **Bench harness debt.** `benches/read_latency.rs` today exercises the cached
 `Bytes` path. The harness needs a new bench (`cold_ranged_from_s3`) that
 exercises the async reader and reports both the LocalStack and real-S3
 numbers — otherwise we re-introduce the LocalStack-only blind spot this RFC
 was written to close.

## Open questions

1. **Coalescing strategy for column pages.** `object_store::get_ranges` issues a
 single multi-range request when the backend supports it; for backends that don't
 (some S3 gateways), it falls back to parallel single-range GETs. Need to measure
 which dominates for our typical 8-column projection.
2. **Page index always-on?** The writer can produce the page index unconditionally
 (~negligible footer overhead) or only when row count exceeds a threshold. Cheap
 to always emit — recommend on by default and revisit only if footer size becomes
 a problem.
3. **Bloom filter probe ordering.** Today: manifest min/max → bloom → body GET.
 New flow: manifest min/max → bloom → footer GET (cheap) → row-group prune →
 page GET. Bloom still saves a footer round trip on a true miss, so keep it
 first. But if the bloom misses are rare in practice (well-tuned FPR), we may
 want to skip it and go straight to the footer fetch which is similarly small.
4. **Property-stream evolution interaction.** RFC-002 §3.2.7 (declared edge
 property streams) is a follow-up. When per-property streams ship, the ranged
 read pattern extends naturally: one extra `get_range` per requested property
 stream. No new design needed, just one more knob on the column projection.
5. **`scan_label` / `scan_edge_type` retention of the eager path.** Confirm that
 range scans always stay on the body-fetch path or whether they should also use
 ranged reads when the result set is small. Probably "always eager for now,
 revisit when the query engine surfaces predicate push-down."

## References

- RFC-002 §4.1 (SstDescriptor format, `size_bytes` already in the manifest).
- Apache Parquet [Page Index spec](https://github.com/apache/parquet-format/blob/master/PageIndex.md).
- `object_store::ObjectStore::get_ranges` ([docs.rs](https://docs.rs/object_store/0.13.0/object_store/trait.ObjectStore.html#method.get_ranges)).
- `parquet::arrow::async_reader::ParquetObjectReader` ([docs.rs](https://docs.rs/parquet/55.2.0/parquet/arrow/async_reader/struct.ParquetObjectReader.html)).
