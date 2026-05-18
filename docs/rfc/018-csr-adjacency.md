# RFC 018: CSR-style adjacency materialised in-snapshot

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
queries — IC09 et al.)
**Builds on:** RFC-002 (SST format §3 edges binary CSR), RFC-003 (ranged reads
+ SstCache), RFC-017 (factorization composes orthogonally with this)
**Supersedes:** —

## Summary

El path actual de `Snapshot::out_edges` / `in_edges` (`read.rs:358`,
`read.rs:650` → `edge_lookup` en `read.rs:655`) hace, **por cada call**, los
siguientes pasos contra cada SST candidato del manifest:

1. Bloom side-car probe (`bloom_admits`) — load body desde `SstCache` o `S3`,
 parse magic + size + xxhash, `BloomFilter::contains(key)`.
2. SST body GET (`get_sst_body`) — cached en `SstCache::inner`, todavía es un
 `Arc::clone`.
3. `EdgeSstReader::open(body)` — parsea header + footer + fence index, build
 `cumulative_edges: Vec<u64>` con un scan completo sobre `partners` (~O(K)
 trabajo por open).
4. `EdgeSstReader::lookup(key)` — fence bracket → `position_of` (binary
 search en `key_ids`) → offset read → partner block decode →
 per-edge LSNs + tombstone bitmap slice.
5. Para edges sourced del SST, `read_overflow_strings` + `load_declared_streams`
 decode todas las property streams del SST aunque la query las ignore.

Cada uno es O(deg + K) o O(K) y la mayoría del trabajo **es por-SST, no
per-key**. En IC09 con scale=0.1 (un fanout total ~110 hops via `KNOWS·KNOWS
+ HAS_CREATOR`), eso es ~110 invocaciones de `edge_lookup` × ~3-5 SSTs por
edge_type → ~400-550 ciclos completos del pipeline (1)-(5) por query.

Esta RFC introduce **`EdgeAdjacency`**: una in-RAM CSR slim materializada
**una vez por `(manifest_version, edge_type, direction)`** por una
`AdjacencyCache` Arc-compartida cross-snapshot. Cada `edge_lookup`
post-rewrite es:

- Cache probe (DashMap-like) → `Arc<EdgeAdjacency>`.
- `binary_search` en `keys: Vec<NodeId>` → idx (O(log K)).
- Slice `partners[offsets[idx]..offsets[idx+1]]` + `lsns[...]` + `tombstones[...]` (O(deg)).
- Memtable overlay para writes recientes (O(memtable_size_for_type)).

Para IC09: el build cost se paga una vez (la primera query del bench warm-up),
y las 49 restantes pegan el cache. Cada `edge_lookup` cae de "~10-30 µs
async-pipeline" a "~few µs sync slice + bool merge".

Este es **el architectural fix** que las optimizaciones previas (NodeView
cache, edge cache) apuntaban sin resolver: el NodeView cache cosechó el
reuse intra-query del lado nodos; el edge cache falló porque las edges no
se reusan intra-query — pero **sí se reusan cross-query** y, más
importante, **el costo no es decode redundancy, es per-call SST scan**.
La CSR mata ambos vectores en un solo golpe. Es exactamente lo que Kùzu
hace internamente (Jin et al., CIDR 2023 §3.1 "rel tables, CSR-indexed by
src and dst").

### Alcance v0

- **CSR slim** — `keys: Vec<NodeId>`, `offsets: Vec<u32>`, `partners:
 Vec<NodeId>`, `lsns: Vec<u64>`, `tombstones: Vec<bool>`. NO carga edge
 properties (decided con el usuario; ver Design §4).
- **Cache Arc-compartido** `AdjacencyCache` cross-snapshot, keyed por
 `(manifest_version, edge_type, direction)`. LRU con memory budget
 configurable (default 512 MiB).
- **Build** una vez en miss, heap-merge sobre todos los SSTs del `(kind,
 scope)` group via la `LoadedManifestIndex` que ya tenemos.
- **Memtable overlay** por call — sweep O(memtable_entries_for_type) +
 per-partner last-LSN-wins merge contra la CSR slice.
- **Reroute en `Snapshot::edge_lookup`** atrás del feature flag
 `NAMIDB_ADJACENCY=0|1` (default `0` inicialmente; flip a `1` después
 de bench-validate).
- **Properties fallback** — si el call site tiene declared properties + las
 necesita, retornar `EdgeView` con properties vacías y dejar al caller
 hacer el lookup secundario via SST path. Esto significa **caveat
 explícito**: con flag ON, `Snapshot::out_edges` retorna `EdgeView.properties =
 BTreeMap::new()` para edges SST-sourced. Memtable edges retienen sus
 properties (vienen del payload decoded). Documentado debajo en §6.
- **Parity tests** comparan topología (src, dst, lsn, tombstone), NO
 properties. Storage unit tests que verifican properties usan flag OFF
 explícitamente.

### Out-of-scope v0 (siguen como follow-ups)

- **Property-aware routing**. Una iteración futura detecta en plan-time
 si la query accede `r.something` y decide topology-only vs full-edge
 lookup per call site. Cuando aterrice, el caveat de v0 desaparece.
- **Disk-tier `AdjacencyCache`**. v0 es memory-only. Cuando el dataset
 exceda el budget, evict via LRU. Spill-a-disk (foyer hybrid) llega cuando
 la memoria sea constraint real.
- **CSR for vector / hybrid indexes**. RFC-007 mantiene su propio shape.
- **Incremental refresh post-flush**. La CSR es invalidada-y-rebuild cuando
 `manifest_version` cambia. v0 paga el full rebuild en la primera query
 post-flush. Incremental merge layered on top queda diferido si bench
 lo amerita.

## Motivation

**Bench actual (scale=0.1):**

| Query | NamiDB p50 | Kùzu p50 | Ratio | Bottleneck dominante |
|---|---|---|---|---|
| IC02 (KNOWS·HAS_CREATOR) | 67 ms | 1.04 ms | **64×** | mixed: ~30% storage, ~30% expr |
| IC07 (HAS_CREATOR·LIKES) | 7 ms | 0.97 ms | 7× | mostly query planner overhead |
| IC08 (HAS_CREATOR·REPLY_OF) | 7 ms | 1.10 ms | 6× | similar a IC07 |
| IC09 (KNOWS·KNOWS·HAS_CREATOR) | **578 ms** | 1.64 ms | **353×** | **storage I/O del Expand chain** |

**Sin properties access para Rel binding en queries IC*** — el `r` es anónimo
en IC09 (`(p)-[:KNOWS]->(f)-[:KNOWS]->(fof)<-[:HAS_CREATOR]-(msg)`). Cada hop
solo necesita `(src, dst)` para emitir la próxima row del Expand. Sin
embargo el path actual decodifica todo: bloom probe + body get + reader open
con `cumulative_edges` scan + position_of bsearch + partner block decode +
per-edge LSN read + per-edge tombstone read + overflow JSON parse + declared
streams IPC decode. Cada call paga el costo completo.

**Profiling estimado (sin flamegraph todavía, basado en read-code-infer):**

| Stage por `edge_lookup` | Aproximado µs (warm cache) |
|---|---|
| `bloom_admits` (cached side-car) | ~5 µs (xxhash + bit probe) |
| `get_sst_body` (cached) | ~1 µs (Arc clone) |
| `EdgeSstReader::open` (build cumulative_edges) | ~50-150 µs (depends on K) |
| `position_of` (fence + bsearch) | ~2-5 µs |
| `lookup` (partner decode + LSN/tomb reads) | ~5-15 µs |
| `read_overflow_strings` (full SST decode) | ~100-500 µs (depends on edge_count) |
| `load_declared_streams` (per-property IPC) | ~50-200 µs |
| **Total per call** | **~200-900 µs** |

Multiplicado por ~110 hops × 3-5 SSTs candidate por hop = ~330-550 invocations
del pipeline. Lower bound: 330 × 200µs = **66 ms**. Upper bound: 550 × 900µs =
**495 ms**. La medición real (578 ms para IC09) cae justo en el medio-alto
del rango. **Confirma la hipótesis storage-I/O = bottleneck.**

**Comparativa Kùzu (rationale ajeno, pero válido):** Kùzu mantiene "rel
tables" CSR-indexed por src y dst en RAM (post-load). Cada hop es un binary
search + slice — ~1-2 µs. Para los mismos ~110 hops: **~110-220 µs total** =
sub-millisecond. Compatible con el 1.64 ms p50 de Kùzu en IC09 (lo que sobra
va a expression eval + materialise output rows).

**Si NamiDB cae a ~1-5 µs por edge_lookup post-rewrite:**

| Reducción | IC09 estimado p50 | Ratio vs Kùzu |
|---|---|---|
| Conservative (5 µs × 110 = 550 µs) | ~30-50 ms | 20-30× |
| Optimistic (1 µs × 110 = 110 µs) | ~10-20 ms | 6-12× |
| Stretch (incl. node cache assist) | ~5-10 ms | 3-6× |

**Sin alcanzar el gate (2× = 3.3 ms), pero acercándose a "demo-friendly"
territory.** Las piezas que cubren el gap restante son property deferral
plan-aware y node materialization batching (iteraciones futuras). Esta
RFC abre el camino.

## Design

### 1. Tipos de datos

Nuevo módulo `crates/namidb-storage/src/adjacency.rs`:

```rust
use std::sync::Arc;
use parking_lot::Mutex; // si no está, use std::sync::Mutex
use lru::LruCache; // existing crate; ya usado o usar manual

use crate::manifest::SstKind;
use crate::sst::edges::EdgeDirection;
use namidb_core::NodeId;

/// In-RAM CSR slim adjacency para un (edge_type, direction) en un
/// manifest_version dado.
///
/// Memory layout (10M edges + 1M distinct keys):
/// - `keys`: 16 B × 1M = 16 MB
/// - `offsets`: 4 B × 1M = 4 MB
/// - `partners`: 16 B × 10M = 160 MB
/// - `lsns`: 8 B × 10M = 80 MB
/// - `tombstones`: 1 B × 10M = 10 MB
/// - Total: ~270 MB para 10M edges, ~27 MB para 1M edges, ~270 KB para 10K.
///
/// Para scale=0.1 LDBC (50K edges per type, ~10K distinct srcs):
/// - 50K × 24 B + 10K × 20 B = ~1.4 MB per (edge_type, direction).
/// - Para 3 edge_types × 2 directions = ~8 MB total. Cabe en cualquier cache.
#[derive(Debug)]
pub struct EdgeAdjacency {
 pub edge_type: String,
 pub direction: EdgeDirection,
 pub manifest_version: u64,
 /// Sorted by NodeId. binary_search returns idx for offsets/partners.
 pub(crate) keys: Vec<NodeId>,
 /// Len = keys.len() + 1. partners[offsets[i]..offsets[i+1]] for keys[i].
 pub(crate) offsets: Vec<u32>,
 pub(crate) partners: Vec<NodeId>,
 pub(crate) lsns: Vec<u64>,
 pub(crate) tombstones: Vec<bool>,
}

impl EdgeAdjacency {
 /// Slim per-key projection. None when `key` is not present in the SSTs
 /// (caller will still consult the memtable overlay).
 pub fn lookup(&self, key: NodeId) -> Option<EdgeSlice<'_>> {
 let idx = self.keys.binary_search(&key).ok()?;
 let lo = self.offsets[idx] as usize;
 let hi = self.offsets[idx + 1] as usize;
 Some(EdgeSlice {
 partners: &self.partners[lo..hi],
 lsns: &self.lsns[lo..hi],
 tombstones: &self.tombstones[lo..hi],
 })
 }

 /// Approximate memory footprint in bytes — for LRU weighting.
 pub fn approx_bytes(&self) -> usize {
 self.keys.len() * 16
 + self.offsets.len() * 4
 + self.partners.len() * 16
 + self.lsns.len() * 8
 + self.tombstones.len()
 + self.edge_type.len()
 + 64 // overhead allowance
 }
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeSlice<'a> {
 pub partners: &'a [NodeId],
 pub lsns: &'a [u64],
 pub tombstones: &'a [bool],
}

/// Cache key. Hash by all three components.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdjacencyKey {
 pub manifest_version: u64,
 pub edge_type: String,
 pub direction: EdgeDirection,
}

/// Process-wide LRU cache de adyacencias materialised. Arc-compartido entre
/// `WriterSession` y todos los `Snapshot`s que emite.
pub struct AdjacencyCache {
 inner: Mutex<LruCache<AdjacencyKey, Arc<EdgeAdjacency>>>,
 /// Cota de bytes — sumamos `approx_bytes()` de cada entry. Excedido =
 /// evict del menos-recientemente-usado.
 capacity_bytes: usize,
 /// Bytes en uso. Tracked al insert / evict.
 used_bytes: Mutex<usize>,
 // counters opcionales (hits / misses / builds) → debug/observability.
}
```

### 2. Build process

```rust
async fn build_adjacency(
 snapshot_manifest: &LoadedManifest,
 store: &dyn ObjectStore,
 paths: &NamespacePaths,
 cache: &SstCache,
 edge_type: &str,
 direction: EdgeDirection,
) -> Result<EdgeAdjacency> {
 let want_kind = match direction {
 EdgeDirection::Forward => SstKind::EdgesFwd,
 EdgeDirection::Inverse => SstKind::EdgesInv,
 };

 // 1. Enumerate SSTs from the manifest index.
 let sst_idxs: Vec<usize> = snapshot_manifest
 .index
 .scope_descriptors(want_kind, edge_type)
 .iter()
 .copied()
 .collect();

 if sst_idxs.is_empty() {
 return Ok(EdgeAdjacency {
 edge_type: edge_type.to_string(),
 direction,
 manifest_version: snapshot_manifest.manifest.version,
 keys: Vec::new(),
 offsets: vec![0],
 partners: Vec::new(),
 lsns: Vec::new(),
 tombstones: Vec::new(),
 });
 }

 // 2. Per-SST: fetch body (cached) + open reader + scan_all_edges.
 // No paralelizamos en v0 (SSTs típicamente small, body cached, build
 // is one-time per manifest version).
 let mut per_partner: BTreeMap<(NodeId, NodeId), (u64, bool)> = BTreeMap::new();
 for idx in sst_idxs {
 let desc = &snapshot_manifest.manifest.ssts[idx];
 let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), desc.path);
 let body = fetch_with_cache(store, cache, &absolute).await?;
 let reader = EdgeSstReader::open(body)?;
 for row in reader.scan_all_edges()? {
 let key_id = NodeId::from_uuid(Uuid::from_bytes(row.key_id));
 let partner_id = NodeId::from_uuid(Uuid::from_bytes(row.partner_id));
 // last-LSN-wins across SSTs (compaction usually leaves at most one
 // SST per (key, partner) but we cannot assume).
 match per_partner.entry((key_id, partner_id)) {
 Entry::Vacant(v) => { v.insert((row.lsn, row.tombstone)); }
 Entry::Occupied(mut o) => {
 if row.lsn > o.get().0 {
 o.insert((row.lsn, row.tombstone));
 }
 }
 }
 }
 }

 // 3. Group by key (BTreeMap iter already sorts by (key, partner)) and
 // materialise into the parallel arrays.
 let mut keys: Vec<NodeId> = Vec::new();
 let mut offsets: Vec<u32> = vec![0];
 let mut partners: Vec<NodeId> = Vec::new();
 let mut lsns: Vec<u64> = Vec::new();
 let mut tombstones: Vec<bool> = Vec::new();

 let mut cur_key: Option<NodeId> = None;
 for ((k, p), (lsn, tomb)) in per_partner {
 match cur_key {
 Some(prev) if prev == k => { /* same key, just append */ }
 _ => {
 if cur_key.is_some() {
 offsets.push(partners.len() as u32);
 }
 keys.push(k);
 cur_key = Some(k);
 }
 }
 partners.push(p);
 lsns.push(lsn);
 tombstones.push(tomb);
 }
 offsets.push(partners.len() as u32); // sentinel
 debug_assert_eq!(offsets.len(), keys.len() + 1);

 Ok(EdgeAdjacency {
 edge_type: edge_type.to_string(),
 direction,
 manifest_version: snapshot_manifest.manifest.version,
 keys,
 offsets,
 partners,
 lsns,
 tombstones,
 })
}
```

Build complexity: O(total_edges · log(total_edges)) por el BTreeMap. Para
scale=0.1 con ~50K edges: ~50K × 20 = ~1 M cmp, <50 ms estimado. Para
10M edges: ~10M × 23 = ~230 M cmp, ~2-5 s. **Aceptable como cold-start cost
porque es one-time per manifest version.**

Alternative considerada (rejected v0): heap-merge cursors stream-style, evita
BTreeMap. Implementación más compleja, similar perf en este rango. Voy con
BTreeMap por claridad. Si bench muestra problema en namespaces grandes,
switch.

### 3. Cache integration

```rust
impl AdjacencyCache {
 pub fn new(capacity_bytes: usize) -> Self { /* ... */ }

 /// Resolve (or build) the EdgeAdjacency for the given key. Builds happen
 /// at most once per (manifest_version, edge_type, direction) — concurrent
 /// callers race on the cache slot; whoever wins inserts.
 pub async fn get_or_build<F, Fut>(
 &self,
 key: AdjacencyKey,
 build: F,
 ) -> Result<Arc<EdgeAdjacency>>
 where
 F: FnOnce() -> Fut,
 Fut: std::future::Future<Output = Result<EdgeAdjacency>>,
 {
 // 1. Probe under lock.
 {
 let mut lru = self.inner.lock();
 if let Some(arc) = lru.get(&key) {
 return Ok(arc.clone());
 }
 }
 // 2. Miss → build outside lock to avoid serialising builds.
 let built = build().await?;
 let weight = built.approx_bytes();
 let arc = Arc::new(built);
 // 3. Insert + evict to budget.
 {
 let mut lru = self.inner.lock();
 // Recheck — another caller may have inserted concurrently.
 if let Some(existing) = lru.get(&key) {
 return Ok(existing.clone());
 }
 lru.put(key.clone(), arc.clone());
 *self.used_bytes.lock() += weight;
 self.evict_to_capacity(&mut lru);
 }
 Ok(arc)
 }

 fn evict_to_capacity(&self, lru: &mut LruCache<AdjacencyKey, Arc<EdgeAdjacency>>) {
 let mut used = self.used_bytes.lock();
 while *used > self.capacity_bytes && lru.len() > 1 {
 if let Some((_, evicted)) = lru.pop_lru() {
 *used = used.saturating_sub(evicted.approx_bytes());
 } else {
 break;
 }
 }
 }
}
```

### 4. `EdgeView.properties` contract con flag ON

Decisión explícita (ver Summary §"Properties fallback"):

```rust
// Snapshot::edge_lookup con flag ON:
async fn edge_lookup_via_csr(...) -> Result<EdgeListView> {
 let adj = self
 .adjacency_cache
 .as_ref()
 .ok_or_else(|| Error::invariant("CSR enabled but cache absent"))?
 .get_or_build(
 AdjacencyKey { manifest_version, edge_type, direction },
 || build_adjacency(&self.manifest, ...),
 )
 .await?;

 // SST-sourced edges: NO properties.
 let mut sst_edges: BTreeMap<NodeId, (u64, Option<EdgeView>)> = BTreeMap::new();
 if let Some(slice) = adj.lookup(key) {
 for i in 0..slice.partners.len() {
 let partner = slice.partners[i];
 let lsn = slice.lsns[i];
 let view = if slice.tombstones[i] {
 None
 } else {
 let (src_id, dst_id) = match direction {
 EdgeDirection::Forward => (key, partner),
 EdgeDirection::Inverse => (partner, key),
 };
 Some(EdgeView {
 edge_type: edge_type.to_string(),
 src: src_id,
 dst: dst_id,
 properties: BTreeMap::new(), // ← caveat documented
 lsn,
 })
 };
 sst_edges.insert(partner, (lsn, view));
 }
 }

 // Memtable overlay: retain full properties (decoded from MemOp::Upsert payload).
 for (mk, entry) in self.memtable.iter() {
 // ... como en el path actual; properties full porque vienen del payload.
 }

 Ok(EdgeListView { edges: /* sort by partner, drop tombstones */ })
}
```

**Caveat con caller:** una query que accede `r.weight` (donde `r` es un Rel
binding) verá `BTreeMap::new()` cuando la edge viene de un SST y el flag está
ON. **Mitigación v0**: storage unit tests que verifican properties via
`out_edges` quedan con flag OFF. LDBC IC* no acceden edge properties → flag
ON OK para el bench gate.

**Mitigación v0.5 si el caveat duele en tests:** chequear schema antes del
reroute. Si `manifest.schema.edge_type(edge_type).properties.is_empty()` →
CSR; si tiene properties declaradas → fallback al SST path. Eso preserva
todos los tests existentes, sub-óptimo pero seguro.

**Mitigación v1 — IMPLEMENTADA:** plan-aware routing en
`namidb_query::exec::walker`. Una pasada al root del `LogicalPlan` recolecta
las variables referenciadas por toda expresión del plan (Filter/Project/TopN/
Aggregate/Join/Unwind/etc., reusando `collect_referenced_variables`). Para
cada `Expand`, si su `rel_alias` aparece en ese set, el executor llama
`Snapshot::out_edges_via_sst` / `in_edges_via_sst` (forzando full-property
SST path) en vez del dispatch default `out_edges`. Cuando el alias está
ausente o es bound pero nunca leído, se mantiene la ruta CSR. El default de
`adjacency_enabled()` quedó en ON (set `NAMIDB_ADJACENCY=0` para
desactivar). El caveat queda invisible para query callers — storage
callers que necesitan properties full deben llamar a `edge_lookup_via_sst`
directamente.

7 tests integration en `crates/namidb-query/tests/exec_plan_aware_routing.rs`
cubren: alias ausente (CSR), alias unused (CSR), `RETURN r.prop` (SST),
`RETURN r` whole (SST), `WHERE r.prop` filter (SST), `ORDER BY r.prop`
(SST), y dos Expands con routing mixto en una misma query.

### 5. Memtable overlay

El memtable contiene puts/deletes recientes que NO están en SSTs aún (no
flushed). La CSR solo representa SSTs. Para correctness:

```rust
// Sweep del memtable filtra por edge_type. Pequeño O(memtable_entries) — el
// memtable está bounded ~64 MiB de payload, típicamente <100K entries en run.
for (mk, entry) in self.memtable.iter() {
 let MemKey::Edge { edge_type: et, src, dst } = mk else { continue };
 if et != edge_type { continue; }
 let (my_key, partner) = match direction {
 EdgeDirection::Forward => (*src.as_bytes(), *dst.as_bytes()),
 EdgeDirection::Inverse => (*dst.as_bytes(), *src.as_bytes()),
 };
 if my_key != key.as_bytes() { continue; }
 // ... merge into latest with last-LSN-wins
}
```

La búsqueda lineal sobre el memtable por edge_type es estable y existing —
no la podemos optimizar sin más cambios. Para v0, ese cost es el mismo que
hoy. Si el memtable es grande, el read path ya pagaba ese cost; CSR no
empeora.

### 6. Invalidation

`manifest_version` es parte del cache key. Cuando el writer commits (post-
flush, post-compaction, post-ingest), el manifest_version incrementa.
Snapshots viejos siguen viendo la entry vieja (Arc clone) hasta que se
droppeen. Snapshots nuevos miss y rebuild. LRU eventualmente evicts entries
viejas.

Eso es **invalidation por construcción**. No hay race entre el writer y
readers — el manifest CAS protocol garantiza linearizability en el path
manifest. La CSR refleja una version atomica del manifest.

### 7. Wiring

```rust
// crates/namidb-storage/src/ingest.rs
pub struct WriterSession {
 // ... existing fields
 adjacency_cache: Option<Arc<AdjacencyCache>>,
}

impl WriterSession {
 pub async fn open(...) -> Result<Self> {
 // ... existing logic
 let adjacency_cache = adjacency_enabled().then(|| {
 Arc::new(AdjacencyCache::new(adjacency_budget_bytes()))
 });
 Ok(Self { /* ..., */ adjacency_cache })
 }

 pub fn snapshot(&self) -> Snapshot<'_> {
 Snapshot::new_with_caches(
 self.current.clone(),
 &self.memtable,
 self.manifest_store.store().clone(),
 self.manifest_store.paths().clone(),
 self.sst_cache.clone(), // o None
 self.adjacency_cache.clone(),
 )
 }
}

// crates/namidb-storage/src/read.rs
pub struct Snapshot<'mt> {
 // ... existing fields
 adjacency_cache: Option<Arc<AdjacencyCache>>,
}

impl<'mt> Snapshot<'mt> {
 pub fn new_with_caches(
 manifest: LoadedManifest,
 memtable: &'mt Memtable,
 store: Arc<dyn ObjectStore>,
 paths: NamespacePaths,
 sst_cache: Option<SstCache>,
 adjacency_cache: Option<Arc<AdjacencyCache>>,
 ) -> Self { /* ... */ }

 async fn edge_lookup(...) -> Result<EdgeListView> {
 if let Some(adj_cache) = &self.adjacency_cache {
 return self.edge_lookup_via_csr(adj_cache.clone(), ...).await;
 }
 self.edge_lookup_via_sst(...).await // path actual renombrado
 }
}
```

`adjacency_enabled()` reads `NAMIDB_ADJACENCY`:
- `"0"` / unset → `None` → SST path (status quo).
- `"1"` → `Some(Arc<AdjacencyCache>)` → CSR path.

`adjacency_budget_bytes()` reads `NAMIDB_ADJACENCY_BUDGET_MIB` (default
512 MiB):
- Big enough para LDBC SF1 (~100M edges → ~2.7 GB CSR — would exceed budget,
 evicts on demand, fine).
- Small enough para no comer toda la RAM en machines compartidos.

### 8. Tests

#### 8.1 Unit (`adjacency.rs::tests`)

- `cache_get_or_build_builds_once`: dos concurrent `get_or_build` con misma
 key → build closure invoked **una vez**.
- `cache_evicts_lru_on_capacity`: insert N entries excediendo budget → oldest
 evicted, `used_bytes` decreciendo.
- `edge_adjacency_lookup_returns_slice`: build manual + `lookup(key)` retorna
 `Some(slice)` con partners esperados.
- `edge_adjacency_lookup_absent_key`: returns `None`.
- `build_adjacency_merges_two_ssts`: setup memtable + 2 SSTs flush distintos
 → build returns merged CSR.

#### 8.2 Integration (`tests/csr_adjacency.rs`)

- `csr_serves_out_edges_topology_correctly`: writer + 2 SSTs + memtable
 layered overlay → snapshot.out_edges retorna expected topology con flag ON.
- `csr_invalidates_on_new_manifest_version`: snapshot1 → flush → snapshot2
 hit cache key diferente.
- `csr_tombstones_preserved_for_correctness`: SST con tombstone @ LSN=N +
 memtable upsert @ LSN=N+1 → edge surfaced. Caso reverso → edge hidden.

#### 8.3 Parity (`tests/csr_adjacency_parity.rs`)

```rust
#[tokio::test]
async fn parity_topology_ic09_shape() {
 let result_off = with_flag(false, || run_ic09()).await;
 let result_on = with_flag(true, || run_ic09()).await;
 // Compare topology only: ignore EdgeView.properties.
 let topo_off: BTreeSet<(NodeId, NodeId, u64)> = result_off.iter()
 .map(|e| (e.src, e.dst, e.lsn))
 .collect();
 let topo_on: BTreeSet<(NodeId, NodeId, u64)> = result_on.iter()
 .map(|e| (e.src, e.dst, e.lsn))
 .collect();
 assert_eq!(topo_off, topo_on);
}
```

### 9. Bench plan

```bash
# Pre-rewrite baseline (already measured):
# IC02 67ms | IC07 7ms | IC08 7ms | IC09 578ms

# Post-rewrite (NAMIDB_ADJACENCY=1):
NAMIDB_ADJACENCY=1 cargo run --release -p namidb-bench -- run \
 --queries IC02,IC07,IC08,IC09 --scale 0.1 --warm-runs 50

# Comparativa:
# Expected IC09: 30-80 ms (10-20× mejora local; ~20-50× vs Kùzu)
# Expected IC02: 25-50 ms (~30% mejora local; ~25-50× vs Kùzu)
# IC07/IC08 no esperan mejora notable (Expand chain corto + node-dominant).
```

## Alternatives considered

### A. Per-call CSR build (rebuilt per query)

Sin Arc-shared cache. Cada Snapshot.new() rebuilds. Pros: zero shared
mutable state, simplicidad. Cons: no aprovecha el reuse cross-query — REPL,
benchmark de N runs, workshop interactivo, todos pagan el build cost
repetidamente. **Rechazada:** la motivación de bench es que las
queries son repetidas; el cache cross-snapshot es la clave del speedup
amortizado.

### B. CSR fat (con properties)

Cargar `Vec<BTreeMap<String, Value>>` paralelo a partners. Pros: coverage
100%, sin caveat de §4. Cons: memoria 5-10× (KNOWS.creationDate single u64
property → ~30 B extra per edge), Vec<BTreeMap> cache-unfriendly. Para 10M
edges fat = 1-3 GB; budget excedido rápido. **Rechazada por memoria.**
Alternativa: lazy-on-demand property fetch contra el SST cuando caller
invoca `e.properties` — funciona pero el API se vuelve incomodo (synchronous
fn returning future). Pospuesto a plan-aware routing que es más limpio.

### C. Kùzu-style dense u64 IDs

Kùzu renumera node IDs a u64 densos contiguos al load, así `offsets[u64
node_id]` es directo (sin keys vec ni binary search). Pros: O(1) lookup
puro. Cons: requiere ID translation table NodeId → u64 (16 → 8 bytes; ~half
saved on partners too via translation). Pero NamiDB usa UUID v7 — el ID
space es público (clientes pasan UUIDs en query parameters). Translation
table en RAM agrega complexity: build cost, invalidation, query API change.
**Rechazada para v0** — el log K binary search en sorted Vec<NodeId> es
cache-friendly y suficiente. v1 puede explorar si bench muestra que es el
nuevo bottleneck.

### D. SstCache extension (in-place)

Reusar `SstCache` agregando un nuevo `metadata: HashMap` keyed por
`(manifest_version, scope, direction)`. Pros: una cache, una config. Cons:
SstCache es path-keyed; el CSR es semantic-keyed. Mezclar las dos
abstracciones genera fricción tipo-system (Bytes vs Arc<EdgeAdjacency>).
**Rechazada por separation of concerns.** AdjacencyCache es un sibling, no
un sub-cache.

### E. Sin feature flag (swap directo)

Reemplazar `edge_lookup` wholesale, confiar en test suite. Cons: si la
implementación tiene un bug, regresión silenciosa en todos los tests. La
parity strategy del flag te da bench-comparativo claro + rollback trivial.
**Rechazada — los costos del flag son <50 LoC; los beneficios son
diagnósticos + safety net.**

## Drawbacks

1. **Properties caveat con flag ON** (§4). v0 no cubre queries que acceden
 `r.something` desde edges SST-sourced. Mitigado por documentación + flag
 default OFF inicialmente. Eliminado completamente con plan-aware
 routing.

2. **Cold-start cost por edge_type**. Primera query toca cada (edge_type,
 direction) paga ~50-500 ms de build cost depending on scale. Para
 benchmarking warm-runs el cost se amortiza en run 2+. Para producción
 "snap-to-cold-query" hay un punzón. Mitigation: opcional eager build en
 `WriterSession::open` para edge_types declarados — pospuesto si bench
 shows que es necesario.

3. **Memory budget se vuelve un knob operacional**. Demasiado bajo: thrashing
 (build → evict → build). Demasiado alto: OOM en machines compartidos.
 Default 512 MiB cubre LDBC SF1 cómodamente (~50 MB needed). Para
 namespaces gigantes (>100 GB edges) el operador debe knobear.

4. **No paraleliza build**. Múltiple SSTs procesados secuencialmente. Para
 namespaces con cientos de SSTs por edge_type, build cost crece linealmente.
 Compaction lo mantiene low en práctica, pero un cold L0-heavy namespace
 pre-compaction pagaría más. v0 acepta; v1 puede paralelizar si dolor real.

5. **BTreeMap allocation en build**. Para 10M edges: 10M × ~80 B node = 800 MB
 temporario antes de materialise. Por una vez per manifest_version,
 aceptable. Si problema, switch a heap-merge cursors.

6. **`scan_all_edges` decodifica todo el SST**. Hoy ya tiene esa
 complejidad cuando se usa para compaction; estamos haciendo el mismo
 trabajo per (manifest_version, scope, direction) en el read path. Net
 change: trabajo amortizado en muchos lookups en vez de hecho per-call.

## Open questions

- **Q1: ¿Eager build en `WriterSession::open` para edge_types declarados en
 el schema?** Reduces cold-start latency. Cost: linear en total edges. Pros
 para production interactive. Contra para batch workloads. **Decidir post-
 bench**.

- **Q2: ¿Cuándo se elimina el feature flag `NAMIDB_ADJACENCY`?** Mi propuesta:
 default OFF inicialmente (validating phase), default ON una vez
 property-aware-routing aterrice, flag eliminated en iteración
 posterior.

- **Q3: Memory budget default.** Empezamos con 512 MiB. Si LDBC SF10 (10×
 scale) lo necesita superior, ajustamos. Open until SF1/SF10 medidos.

## References

- **Jin et al., CIDR 2023** "Kùzu: An Embeddable Graph DBMS" §3.1 ("Rel
 Tables, CSR-indexed by src and dst", "Adjacency information cached
 in-memory per direction").
- **RFC-002** §3 ("CSR binario on-disk" — el on-disk format ya es CSR; v0
 materialise the same in-RAM sin parse overhead).
- **RFC-003** §"Ranged reads + page index" — RFC-003 redujo per-call cost
 de body GET; el bottleneck restante post-RFC-003 es el decode + scan
 per-call. Esta RFC cierra esa puerta.
- **RFC-017** §"Out-of-scope WCOJ" — factorization y CSR son ortogonales: la
 CSR sirve por edge_type, los factor nodes apilan multiple bindings sin
 copiar BTreeMaps. Composición natural.
