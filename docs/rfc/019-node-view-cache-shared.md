# RFC 019: Cross-snapshot NodeView cache

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Builds on:** RFC-018 (CSR adjacency confirmed `lookup_node` as the
dominant remaining cost), NodeView intra-snapshot cache (preserved as
the L1 of the 3-tier lookup)
**Supersedes:** —

## Summary

A previous step introduced a per-`Snapshot` `Mutex<HashMap<(label,
NodeId), Option<NodeView>>>` and harvested ~12% in IC09 cosechando el
reuse intra-query (~10× repeated access for friends-of-friends, joins
probing the same node from both sides). The profile run
(`NAMIDB_PROFILE_DUMP=1 NAMIDB_ADJACENCY=1`, IC09 scale=0.1, 3 params ×
51 runs = 153 query executions) made the next lever obvious:

```
stage count total_ms avg_us
Snapshot::lookup_node 253,317 87,030.871 343.565
Snapshot::lookup_node_uncached 229,908 86,938.611 378.145
Snapshot::lookup_node.cache_hit 23,409 0.000 0.000
```

`lookup_node` is **99.4% of the IC09 wall-clock** and the intra-snapshot
cache hits exactly **9% of calls** (23.4K hits / 253K total). The remaining
91% pays the full SST candidate walk + bloom probe + parquet decode
(~378 µs each on average).

The reason for the low hit rate is structural, not algorithmic: the bench
runner builds a **fresh `Snapshot` per query execution** (`runner.rs:107`,
`writer.snapshot()` inside the warm-run loop). The intra-snapshot cache
fills during one query and is dropped at the next. **Cross-snapshot the
LDBC fixture has fewer than 5K unique nodes**; if the cache survived
across snapshots tied to the same `manifest_version`, the post-warmup hit
rate would be ~99% and lookup_node calls would collapse from 253K to
~5K.

Esta RFC introduces **`NodeViewCache`** — an `Arc`-shared, cross-snapshot
cache keyed by `(manifest_version, label, NodeId)` and storing
`Option<NodeView>` (yes, including cached "not found" so subsequent
lookups for a tombstoned or missing key skip the SST walk too).

The shape is intentionally a near-clone of `AdjacencyCache` (RFC-018 §3):
same eviction policy, same invalidation contract, same wiring pattern,
same env-var-gated feature flag. The two caches compose orthogonally —
together they cover ~99.5% of the IC09 wall-clock that the CSR
adjacency plus this node cache can reach.

### Alcance v0

- **`NodeViewCache`** — `HashMap<NodeCacheKey, Option<NodeView>>` guarded
 by `Mutex`. `Option<NodeView>` so misses (tombstones, absent rows) are
 cached too (negative-cache; same correctness contract because the key
 includes `manifest_version`).
- **3-tier `Snapshot::lookup_node`**:
 1. **L1 (intra-snap)**: existing `node_cache: Mutex<HashMap>` — fast
 short-circuit when the same `(label, NodeId)` was hit earlier in
 **this** query.
 2. **L2 (cross-snap)**: `NodeViewCache` shared `Arc`. Promotes the
 answer into L1 on hit so subsequent intra-snap calls bypass L2.
 3. **L3 (cold path)**: `lookup_node_uncached` (the existing SST walk).
 Inserts result into both L2 and L1.
- **Memory budget** configurable via `NAMIDB_NODE_CACHE_BUDGET_MIB`
 (default 256 MiB). For LDBC scale=0.1 (~5K nodes × ~1 KiB / NodeView)
 ~5 MiB; for SF1 (~500K nodes) ~500 MiB — operator knob.
- **Routing**: `NAMIDB_NODE_CACHE=1` enables the L2; default OFF
 preserves the previous L1-only behaviour exactly. Same env-var pattern
 as `NAMIDB_ADJACENCY` and `NAMIDB_FACTORIZE`.
- **Parity tests** — `tests/node_cache_parity.rs` mirror the CSR
 adjacency pattern: same Snapshot, two public APIs
 (`lookup_node_via_uncached` vs the default 3-tier path) compared.

### Out-of-scope v0

- **Negative-cache TTL or invalidation beyond manifest_version**. v0
 treats `None` results identically to `Some` — both cached, both
 invalidated when the manifest advances. If a writer commits then
 another reader queries the same key under the new version, the new
 version forces a fresh L2 entry (new key, new lookup). Edge case
 for LDBC: nonexistent.
- **Pre-warming on `WriterSession::open`**. The cache fills lazily.
 Production interactive workloads pay a single cold-query penalty
 per `(manifest_version, label, node_id)` triple — same as
 AdjacencyCache.
- **Disk-tier overflow**. Memory-only. When budget bites, evict by
 oldest `manifest_version` first (same FIFO-by-version as
 `AdjacencyCache`).
- **Per-property invalidation**. v0 caches the full `NodeView` (every
 declared + ad-hoc property). When the writer commits an upsert
 changing a single property, the new `manifest_version` invalidates
 the whole entry — coarse but correct.

## Motivation

Already covered by the profile data in the Summary. Repeating the
expected impact:

**Pre-rewrite (only CSR adjacency, NAMIDB_ADJACENCY=1):**
- IC09 p50 = **520 ms**.
- `lookup_node` = 99.4% of wall-clock.
- L1 hit rate = 9% (23K / 253K).

**Post-rewrite (NAMIDB_ADJACENCY=1 + NAMIDB_NODE_CACHE=1):**
- Estimated L2 hit rate post-warmup = **~98-99%** because LDBC IC* has
 ~5K unique person/post/comment nodes and 153 runs touch the same
 ~hundreds repeatedly.
- L3 (cold path) calls collapse from 230K to ~5K. Wall-clock saved:
 ~85 seconds across 153 runs = ~555 ms per run = **IC09 ~50-80 ms p50**.
- Gate vs Kùzu IC09 estimated **30-50×** (was 317× post-CSR).
- Other queries (IC02/07/08) reap similar relative wins since they're
 also lookup_node-dominated.

**If the bench delivers <60 ms IC09**, this would be the **first
iteration to cross the order-of-magnitude line** vs Kùzu in NamiDB
history.

## Design

### 1. Tipos de datos

```rust
// crates/namidb-storage/src/node_cache.rs

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use namidb_core::NodeId;
use crate::read::NodeView;

/// Compound key. (manifest_version, label, node_id). Two snapshots that
/// share the manifest version share the cache slot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeCacheKey {
 pub manifest_version: u64,
 pub label: String,
 pub node_id: NodeId,
}

/// Cached NodeView outcome. `None` means the cold path resolved to
/// "absent / tombstoned" — we cache the negative answer to avoid
/// repeating the SST walk.
pub type CachedNodeView = Option<NodeView>;

#[derive(Debug, Default)]
struct CacheStats {
 hits: AtomicU64,
 misses: AtomicU64,
 inserts: AtomicU64,
 evictions: AtomicU64,
}

pub struct NodeViewCache {
 inner: Mutex<HashMap<NodeCacheKey, CachedNodeView>>,
 capacity_bytes: usize,
 used_bytes: Mutex<usize>,
 stats: Arc<CacheStats>,
}
```

### 2. API

```rust
impl NodeViewCache {
 pub fn new(capacity_bytes: usize) -> Self;
 pub fn get(&self, key: &NodeCacheKey) -> Option<CachedNodeView>;
 pub fn insert(&self, key: NodeCacheKey, view: CachedNodeView);
 pub fn hits(&self) -> u64;
 pub fn misses(&self) -> u64;
 pub fn inserts(&self) -> u64;
 pub fn evictions(&self) -> u64;
 pub fn entries(&self) -> usize;
 pub fn used_bytes(&self) -> usize;
}
```

`get` returns `Option<CachedNodeView>` which is `Option<Option<NodeView>>`:
- `None` → cache miss, caller goes to L3.
- `Some(Some(view))` → cached hit, view available.
- `Some(None)` → cached miss, key was absent at this manifest version.

### 3. 3-tier `Snapshot::lookup_node`

```rust
pub async fn lookup_node(&self, label: &str, id: NodeId) -> Result<Option<NodeView>> {
 namidb_core::profile_scope!("Snapshot::lookup_node");

 // L1: intra-snapshot cache.
 let intra_key = (label.to_string(), id);
 if let Some(cached) = self.node_cache.lock().unwrap().get(&intra_key).cloned() {
 namidb_core::profile::record("Snapshot::lookup_node.l1_hit", 0);
 return Ok(cached);
 }

 // L2: cross-snapshot cache. Optional — controlled by
 // NAMIDB_NODE_CACHE + WriterSession-supplied Arc.
 if let Some(shared) = &self.shared_node_cache {
 let shared_key = NodeCacheKey {
 manifest_version: self.manifest.manifest.version,
 label: label.to_string(),
 node_id: id,
 };
 if let Some(cached) = shared.get(&shared_key) {
 namidb_core::profile::record("Snapshot::lookup_node.l2_hit", 0);
 // Promote into L1 for the rest of this snapshot's life.
 self.node_cache.lock().unwrap().insert(intra_key, cached.clone());
 return Ok(cached);
 }
 }

 // L3: cold SST walk.
 let result = self.lookup_node_uncached(label, id).await?;
 // Insert into L1.
 self.node_cache.lock().unwrap().insert(intra_key, result.clone());
 // Insert into L2 (if attached).
 if let Some(shared) = &self.shared_node_cache {
 let shared_key = NodeCacheKey {
 manifest_version: self.manifest.manifest.version,
 label: label.to_string(),
 node_id: id,
 };
 shared.insert(shared_key, result.clone());
 }
 Ok(result)
}
```

### 4. `WriterSession` wiring

```rust
pub struct WriterSession {
 // ... existing
 adjacency_cache: Option<Arc<AdjacencyCache>>,
 node_cache: Option<Arc<NodeViewCache>>, // ← NEW
}

impl WriterSession {
 pub async fn open(...) -> Result<Self> {
 // ... existing
 let adjacency_cache = adjacency_enabled().then(...);
 let node_cache = node_cache_enabled().then(|| {
 Arc::new(NodeViewCache::new(node_cache_budget_bytes()))
 });
 Ok(Self { ..., adjacency_cache, node_cache })
 }

 pub fn snapshot(&self) -> Snapshot<'_> {
 let mut snap = Snapshot::new(...);
 if let Some(c) = &self.adjacency_cache {
 snap = snap.with_adjacency_cache(c.clone());
 }
 if let Some(c) = &self.node_cache {
 snap = snap.with_shared_node_cache(c.clone());
 }
 snap
 }
}
```

### 5. Memory accounting

```rust
fn approx_size(view: &CachedNodeView) -> usize {
 match view {
 None => 32, // overhead allowance
 Some(v) => {
 v.label.capacity()
 + v.properties.iter().map(|(k, _)| k.capacity() + 64).sum::<usize>()
 + 128 // NodeId + lsn + schema_version + Box/Map overhead
 }
 }
}
```

The estimate is conservative. For the LDBC fixture, ~1 KiB per cached
NodeView × ~5K unique nodes × 2 labels (Person, Post, Comment all small)
= a few MiB. Far below the 256 MiB default budget.

When `used_bytes + new_entry_size > capacity_bytes`, evict by oldest
`manifest_version` first (same FIFO-by-version as AdjacencyCache). For
LDBC the eviction path is rare — only triggers when the manifest
advances rapidly.

### 6. Tests

#### 6.1 Unit (`node_cache.rs::tests`, ~4 tests)

- `cache_get_miss_returns_none`.
- `cache_insert_then_get_returns_view`.
- `negative_cache_returns_inner_none_on_hit` — insert `Some(None)`, get
 returns `Some(None)` (not `None` of the outer Option).
- `cache_evicts_oldest_version_when_over_budget`.

#### 6.2 Integration (`tests/node_cache_parity.rs`, ~3 tests)

Same shape as `tests/csr_adjacency_parity.rs`. Helpers:

```rust
async fn lookup_via_uncached(snap, label, id) -> Result<Option<NodeView>>;
async fn lookup_via_tiered (snap, label, id) -> Result<Option<NodeView>>;
```

- `node_cache_parity_pure_sst` — flush some nodes, lookup via both paths,
 assert equal.
- `node_cache_parity_with_tombstone_overlay` — memtable tombstone hides
 SST upsert; cache promotes the negative answer; subsequent calls hit
 L2.
- `node_cache_reuses_across_snapshots` — snapshot1 misses + inserts;
 snapshot2 (same manifest_version) hits.

### 7. Bench plan

**Triple run, scale=0.1, 50 warm runs, 3 params:**

1. **Baseline** (no flags) — intra-snapshot L1 only.
2. **CSR only** (`NAMIDB_ADJACENCY=1`) — RFC-018 path.
3. **CSR + NodeCache** (`NAMIDB_ADJACENCY=1 NAMIDB_NODE_CACHE=1`) — this RFC.

Expected delta vs baseline:
- IC02: 64 → ~25 ms.
- IC07: 7 → ~3 ms (already near Kùzu).
- IC08: 7 → ~3 ms.
- IC09: 596 → ~50-80 ms (~8-12× mejora). **Gate vs Kùzu 30-50×**.

Profile dump (`NAMIDB_PROFILE_DUMP=1`) confirma:
- `lookup_node.l2_hit` count >> `lookup_node_uncached` count
- post-warmup hit rate ~98-99%.

## Alternatives considered

### A. Make NodeView cache `'static` on the namespace, not per-WriterSession

Pro: any tool building snapshots against the same namespace shares the
cache (CLI, future REST API). Con: cross-process invalidation requires
real coordination (manifest version is monotonic but the cache lives in
RAM; restart a process, lose the cache). v0 stays per-WriterSession.
Adequate for the bench harness; upgrade .

### B. Negative-cache-as-policy (don't cache misses)

Rejected. Misses are the EXPENSIVE path. Caching them is the entire
point of the L2: a Snapshot probing for a deleted node should NOT redo
the SST walk after the first time. Same correctness contract as positive
caching because the key includes `manifest_version`.

### C. Per-label sharding to reduce mutex contention

Premature. Single mutex contention at 1500 concurrent `lookup_node`
calls × 343µs avg = ~500 µs of held-mutex time per second per snapshot.
For a single tokio runtime executor that's fine. If contention shows in
multi-core production, switch to a `DashMap` or per-label `parking_lot`
shards.

### D. Lift the cache into `SstCache::metadata`-style HashMap

Rejected for same reasons as RFC-018 §"Alternative D": SstCache is
path-keyed (`String`), NodeViewCache is semantic-keyed
(`(manifest_version, label, NodeId)`). Mixing types in one cache
muddles the abstraction.

## Drawbacks

1. **First-query latency unchanged** — cold start is still
 `lookup_node_uncached` cost (~378 µs each, ~5K calls = ~2 s
 warmup for the full LDBC fixture). Mitigation: eager `pre_warm`
 helper on `WriterSession::open` that walks the manifest and
 pre-loads `NodeView`s. Pospuesto a v1.

2. **Cache memory pressure under schema churn**. If the writer flushes
 N times during one read burst, N×current_entries get retained until
 eviction kicks in. Budget guard limits the damage but each
 `(manifest_version, label, node_id)` slot is a separate entry.
 Mitigation: aggressive FIFO-by-version eviction (same as
 AdjacencyCache).

3. **Negative-cache amplification on schema explorations**. A query
 that probes lots of "does this exist?" gets every miss pinned.
 Long-running interactive sessions may grow the cache to dataset
 size faster than the read working set. Memory budget keeps this
 bounded.

4. **3-tier path adds branches in the hot path**. Two extra `if let
 Some(...)` checks per lookup_node call. Negligible (~5-10 ns) but
 present. Profile shows current `lookup_node` at ~343 µs avg; the
 additional branches are ~0.002% overhead.

5. **Tests now have THREE pathways** to maintain parity over:
 `_uncached` (cold), `intra-snap` (L1 only), `tiered` (L1+L2+L3).
 Mitigated by exposing the path-forcing public APIs and writing
 targeted parity tests.

## Open questions

- **Q1: Default flag — when to flip?** Mi propuesta: default OFF
 inicialmente (validation phase), flip default ON once property-aware
 routing and additional bench validation cierren.
- **Q2: Should AdjacencyCache and NodeViewCache share a global
 memory budget?** v0 keeps them separate (512 MiB + 256 MiB). If
 operator complaints come in, unify under a single configurable
 pool.
- **Q3: Drop the per-Snapshot L1?** Once L2 is enabled, L1 is mostly
 redundant (after the first L2 promotion). But L1 hit is free (no
 hash + mutex), so keeping it as a fast-path is cheap.

## References

- **Profile data** — `/tmp/bench-profile-ic09.stderr` reproducible via
 `NAMIDB_PROFILE_DUMP=1 NAMIDB_ADJACENCY=1 cargo run --release -p
 namidb-bench -- run --only ic09 --scale 0.1 --warm-runs 50
 --param-count 3`.
- **RFC-018** — same Arc-shared cache pattern for adjacency.
- **Intra-snapshot `node_cache`** — original per-snapshot design
 retained as L1.
- **Kùzu CIDR 2023 §3.2** — "Node tables, materialized in an
 in-memory page-cached buffer; lookups are direct array index on
 internal NodeOffset". NamiDB's equivalent at slice-of-Snapshot vs
 Kùzu's lifetime-of-database is the analogous trade-off; we keep
 manifest-version-bounded freshness instead of paying for
 invalidation protocols.
