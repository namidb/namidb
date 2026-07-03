//! CSR-style adjacency materialised in RAM for fast `Snapshot::edge_lookup`.
//!
//! See [`docs/rfc/018-csr-adjacency.md`](../../../docs/rfc/018-csr-adjacency.md)
//! for motivation, design, and the bench-validated rationale (CSR + cache layers
//! profiling identified per-call SST decode as the dominant cost in IC09 →
//! ~578 ms p50, 353× vs Kùzu).
//!
//! ## Shape
//!
//! For each `(manifest_version, edge_type, direction)` triple, an
//! [`EdgeAdjacency`] holds five parallel arrays — a slim CSR mirror of the
//! on-disk format ([`crate::sst::edges`]):
//!
//! - `keys`: sorted `NodeId`s present as source/dst keys.
//! - `offsets`: `partners[offsets[i]..offsets[i+1]]` are key `i`'s edges.
//! - `partners`: counterpart `NodeId` of every edge (dst for fwd, src for inv).
//! - `lsns`: per-edge LSN.
//! - `tombstones`: per-edge bit; true ⇒ this edge has been deleted at `lsn`.
//!
//! The [`AdjacencyCache`] is an `Arc`-shared, bounded-budget process-wide
//! cache. Snapshots that share a `manifest_version` see the same
//! `Arc<EdgeAdjacency>` and pay the build cost exactly once across queries.
//!
//! ## Slim, not fat
//!
//! v0 stores topology only. `EdgeView.properties` for SST-sourced edges
//! comes back as an empty map. Memtable-sourced edges keep their full
//! property maps (decoded from the upsert payload, like today). The
//! routing decision lives behind the `NAMIDB_ADJACENCY` env var so the
//! path remains usable for tests that need full edge properties.
//! Plan-aware routing will eliminate the caveat by inspecting
//! whether the query accesses `r.*` and falling back to the SST path
//! per call site when it does.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use uuid::Uuid;

use namidb_core::NodeId;

use crate::cache::SstCache;
use crate::error::{Error, Result};
use crate::manifest::{LoadedManifest, SstKind};
use crate::paths::NamespacePaths;
use crate::sst::edges::reader::EdgeSstReader;
use crate::sst::edges::EdgeDirection;

/// Default budget for an [`AdjacencyCache`]: 512 MiB. Override via
/// `NAMIDB_ADJACENCY_BUDGET_MIB`.
pub const DEFAULT_ADJACENCY_BUDGET_MIB: usize = 512;

/// Read `NAMIDB_ADJACENCY` and return `false` only for `"0"`. Anything
/// else (unset, `"1"`, garbage) returns `true`. The default flipped in
/// Once plan-aware routing (`namidb_query::exec::walker`) made the
/// properties caveat from RFC-018 §4 invisible to query callers — set
/// `NAMIDB_ADJACENCY=0` to force the legacy SST path for benchmarking or
/// troubleshooting.
pub fn adjacency_enabled() -> bool {
    std::env::var("NAMIDB_ADJACENCY")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Read `NAMIDB_ADJACENCY_BUDGET_MIB` or fall back to [`DEFAULT_ADJACENCY_BUDGET_MIB`].
pub fn adjacency_budget_bytes() -> usize {
    let mib = std::env::var("NAMIDB_ADJACENCY_BUDGET_MIB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ADJACENCY_BUDGET_MIB);
    mib.saturating_mul(1024 * 1024)
}

/// Process-wide shared [`AdjacencyCache`]: one instance for every
/// [`crate::WriterSession`] the process opens, so
/// `NAMIDB_ADJACENCY_BUDGET_MIB` bounds the PROCESS, not each session.
/// Sharing is only sound because [`AdjacencyKey`] embeds the namespace
/// prefix — the bare `(manifest_version, edge_type, direction)` triple
/// collides across tenants (per-namespace manifest versions both start
/// at 1) and would serve one tenant's CSR to another.
///
/// The enable flag and budget are read once, on first use. Returns `None`
/// when `NAMIDB_ADJACENCY=0` at first use. Callers needing a private
/// instance inject one via
/// [`crate::ingest::WriterSession::open_with_caches`].
pub fn shared_adjacency_cache() -> Option<Arc<AdjacencyCache>> {
    static SHARED: OnceLock<Option<Arc<AdjacencyCache>>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            adjacency_enabled().then(|| Arc::new(AdjacencyCache::new(adjacency_budget_bytes())))
        })
        .clone()
}

/// Per-key projection of an [`EdgeAdjacency`]. Lifetime-bound to the
/// adjacency it came from — callers copy out the partner / lsn / tombstone
/// they care about while still holding the `Arc<EdgeAdjacency>`.
#[derive(Debug, Clone, Copy)]
pub struct EdgeSlice<'a> {
    pub partners: &'a [NodeId],
    pub lsns: &'a [u64],
    pub tombstones: &'a [bool],
}

impl<'a> EdgeSlice<'a> {
    pub fn len(&self) -> usize {
        self.partners.len()
    }
    pub fn is_empty(&self) -> bool {
        self.partners.is_empty()
    }
}

/// In-RAM CSR slim adjacency for a single
/// `(manifest_version, edge_type, direction)` triple.
#[derive(Debug)]
pub struct EdgeAdjacency {
    pub edge_type: String,
    pub direction: EdgeDirection,
    pub manifest_version: u64,
    keys: Vec<NodeId>,
    offsets: Vec<u32>,
    partners: Vec<NodeId>,
    lsns: Vec<u64>,
    tombstones: Vec<bool>,
}

impl EdgeAdjacency {
    /// Empty adjacency for a `(scope, direction)` group with no SSTs in
    /// the manifest. Memtable overlay still works against it.
    pub fn empty(edge_type: String, direction: EdgeDirection, manifest_version: u64) -> Self {
        Self {
            edge_type,
            direction,
            manifest_version,
            keys: Vec::new(),
            offsets: vec![0],
            partners: Vec::new(),
            lsns: Vec::new(),
            tombstones: Vec::new(),
        }
    }

    /// Count of distinct keys (src for forward, dst for inverse).
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Total edges across all keys.
    pub fn edge_count(&self) -> usize {
        self.partners.len()
    }

    /// Look up `key` and return the parallel slices of its edges.
    /// Returns `None` when the key has zero rows in the SSTs.
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

    /// Approximate memory footprint used by [`AdjacencyCache`] for budget
    /// accounting. Counts the buffer-backing storage of each Vec plus a
    /// small overhead allowance.
    pub fn approx_bytes(&self) -> usize {
        self.keys.capacity() * 16
            + self.offsets.capacity() * 4
            + self.partners.capacity() * 16
            + self.lsns.capacity() * 8
            + self.tombstones.capacity()
            + self.edge_type.capacity()
            + 64
    }
}

/// Hash key for the adjacency cache. Must mirror exactly the tuple that
/// uniquely identifies one CSR — collisions across different
/// `manifest_version`s (or, now that the cache is shared process-wide,
/// across namespaces) would surface stale or foreign data.
///
/// `namespace` is the object-store namespace prefix (`<root>/<ns>`, no
/// trailing slash — [`crate::paths::NamespacePaths::namespace_prefix`]);
/// see [`crate::node_cache::NodeCacheKey`] for the rationale.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdjacencyKey {
    pub namespace: Arc<str>,
    pub manifest_version: u64,
    pub edge_type: String,
    pub direction: EdgeDirection,
}

impl AdjacencyKey {
    pub fn new(
        namespace: impl Into<Arc<str>>,
        manifest_version: u64,
        edge_type: impl Into<String>,
        direction: EdgeDirection,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            manifest_version,
            edge_type: edge_type.into(),
            direction,
        }
    }
}

/// Counters surfaced for diagnostics + tests. The cache itself relies only
/// on `used_bytes` for budget gating.
#[derive(Debug, Default)]
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    builds: AtomicU64,
    evictions: AtomicU64,
}

/// Process-wide CSR cache shared across `Snapshot`s — by default across
/// every `WriterSession` in the process ([`shared_adjacency_cache`]), so
/// the budget is global; keys embed the namespace so tenants never
/// collide.
///
/// v0 implementation: a `HashMap` guarded by a `Mutex` plus a manual
/// budget check on insert. When the inserted entry would push
/// `used_bytes > capacity_bytes`, we drop the entry with the **smallest**
/// `manifest_version` first (oldest), then any remaining entries until we
/// fit. This is enough for LDBC SF1 where the cache contains a handful of
/// `(edge_type, direction)` slots and version churn is low.
pub struct AdjacencyCache {
    inner: Mutex<HashMap<AdjacencyKey, Arc<EdgeAdjacency>>>,
    capacity_bytes: usize,
    used_bytes: Mutex<usize>,
    stats: Arc<CacheStats>,
}

impl std::fmt::Debug for AdjacencyCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entries = self.inner.lock().unwrap().len();
        let used = *self.used_bytes.lock().unwrap();
        f.debug_struct("AdjacencyCache")
            .field("entries", &entries)
            .field("used_bytes", &used)
            .field("capacity_bytes", &self.capacity_bytes)
            .field("hits", &self.stats.hits.load(Ordering::Relaxed))
            .field("misses", &self.stats.misses.load(Ordering::Relaxed))
            .field("builds", &self.stats.builds.load(Ordering::Relaxed))
            .field("evictions", &self.stats.evictions.load(Ordering::Relaxed))
            .finish()
    }
}

impl AdjacencyCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            capacity_bytes: capacity_bytes.max(1),
            used_bytes: Mutex::new(0),
            stats: Arc::new(CacheStats::default()),
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn used_bytes(&self) -> usize {
        *self.used_bytes.lock().unwrap()
    }

    pub fn entries(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn hits(&self) -> u64 {
        self.stats.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.stats.misses.load(Ordering::Relaxed)
    }
    pub fn builds(&self) -> u64 {
        self.stats.builds.load(Ordering::Relaxed)
    }
    pub fn evictions(&self) -> u64 {
        self.stats.evictions.load(Ordering::Relaxed)
    }

    /// Probe-only fast path. Used internally by `get_or_build`; exposed
    /// so unit tests can verify a build closure was not invoked.
    pub fn get(&self, key: &AdjacencyKey) -> Option<Arc<EdgeAdjacency>> {
        let map = self.inner.lock().unwrap();
        match map.get(key) {
            Some(arc) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                Some(arc.clone())
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Resolve `key`, building via `build` on miss. Build closure runs
    /// **outside the lock** so a slow build does not block readers of
    /// other keys; on the race where two builders for the same key
    /// finish concurrently, the second insert is a no-op (the first
    /// winner stays).
    pub async fn get_or_build<F, Fut>(
        &self,
        key: AdjacencyKey,
        build: F,
    ) -> Result<Arc<EdgeAdjacency>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<EdgeAdjacency>>,
    {
        if let Some(hit) = self.get(&key) {
            return Ok(hit);
        }
        let built = build().await?;
        self.stats.builds.fetch_add(1, Ordering::Relaxed);
        let arc = Arc::new(built);
        let weight = arc.approx_bytes();

        let mut map = self.inner.lock().unwrap();
        if let Some(existing) = map.get(&key) {
            // Another caller won the race; discard our build.
            return Ok(existing.clone());
        }
        // Evict to budget if necessary. Drop oldest (lowest
        // manifest_version) first.
        let mut used = self.used_bytes.lock().unwrap();
        while *used + weight > self.capacity_bytes && !map.is_empty() {
            // Pop the oldest manifest_version first; among ties, any entry
            // is acceptable. We pick by manifest_version + edge_type only
            // (no Ord on EdgeDirection) so two directions of the same
            // version do not have a defined order — fine for a tie-break.
            let victim_key = map
                .keys()
                .min_by_key(|k| (k.manifest_version, k.edge_type.clone()))
                .cloned();
            let Some(vk) = victim_key else { break };
            if let Some(victim) = map.remove(&vk) {
                *used = used.saturating_sub(victim.approx_bytes());
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        // If after fully evicting the entry STILL doesn't fit, we still
        // insert (a single CSR larger than the budget is acceptable as
        // the only resident entry — alternative is failing the lookup
        // which is worse for correctness).
        map.insert(key, arc.clone());
        *used = used.saturating_add(weight);
        Ok(arc)
    }

    /// Eagerly drop every CSR belonging to `namespace` (the namespace
    /// prefix embedded in [`AdjacencyKey::namespace`]). Called when a
    /// multi-tenant host evicts a namespace — CSRs are the largest cache
    /// entries per byte, so reclaiming them eagerly matters most here.
    pub fn prune_namespace(&self, namespace: &str) {
        let mut map = self.inner.lock().unwrap();
        let mut used = self.used_bytes.lock().unwrap();
        let victims: Vec<AdjacencyKey> = map
            .keys()
            .filter(|k| k.namespace.as_ref() == namespace)
            .cloned()
            .collect();
        for key in victims {
            if let Some(victim) = map.remove(&key) {
                *used = used.saturating_sub(victim.approx_bytes());
            }
        }
    }

    /// Count of CSR entries whose key belongs to `namespace`. Observability
    /// + test probe: with the process-wide shared cache, the global
    /// [`Self::builds`] counter can no longer prove "this namespace built
    /// no CSR", but an entry count scoped to the namespace can.
    pub fn namespace_entries(&self, namespace: &str) -> usize {
        self.inner
            .lock()
            .unwrap()
            .keys()
            .filter(|k| k.namespace.as_ref() == namespace)
            .count()
    }
}

/// Build an [`EdgeAdjacency`] by scanning every SST belonging to
/// `(want_kind, edge_type)` in the manifest. Each SST is fetched through
/// the supplied `SstCache` (so the second build of the same body is a
/// `Arc::clone`). Per `(key_id, partner_id)`, last-LSN wins across SSTs.
///
/// Complexity: `O(total_edges · log total_edges)` due to the intermediate
/// `BTreeMap`. For LDBC scale=0.1 (~50K edges per edge_type) this is
/// well under 50 ms; for SF1 (~500K per type) under 500 ms. Both are
/// one-time per `(manifest_version, edge_type, direction)`.
pub async fn build_adjacency(
    snapshot_manifest: &LoadedManifest,
    store: &dyn object_store::ObjectStore,
    paths: &NamespacePaths,
    cache: Option<&SstCache>,
    edge_type: &str,
    direction: EdgeDirection,
) -> Result<EdgeAdjacency> {
    namidb_core::profile_scope!("adjacency::build_adjacency");
    use std::collections::BTreeMap;

    let want_kind = match direction {
        EdgeDirection::Forward => SstKind::EdgesFwd,
        EdgeDirection::Inverse => SstKind::EdgesInv,
    };

    let sst_idxs: Vec<usize> = snapshot_manifest
        .index
        .scope_descriptors(want_kind, edge_type)
        .to_vec();

    let manifest_version = snapshot_manifest.manifest.version;

    if sst_idxs.is_empty() {
        return Ok(EdgeAdjacency::empty(
            edge_type.to_string(),
            direction,
            manifest_version,
        ));
    }

    // (key_id_bytes, partner_id_bytes) → (lsn, tombstone). Last-LSN-wins.
    // We use raw bytes so the BTreeMap sort order matches the on-disk
    // partner-sort convention (both fwd and inv SSTs sort partners by
    // their counterpart bytes ascending).
    let mut per_partner: BTreeMap<([u8; 16], [u8; 16]), (u64, bool)> = BTreeMap::new();

    for idx in sst_idxs {
        let desc = &snapshot_manifest.manifest.ssts[idx];
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), desc.path);
        let body = fetch_body(store, cache, &absolute).await?;
        let reader = EdgeSstReader::open(body)?;
        for row in reader.scan_all_edges()? {
            let entry = per_partner.entry((row.key_id, row.partner_id));
            entry
                .and_modify(|(lsn, tomb)| {
                    if row.lsn > *lsn {
                        *lsn = row.lsn;
                        *tomb = row.tombstone;
                    }
                })
                .or_insert((row.lsn, row.tombstone));
        }
    }

    // Materialise into parallel arrays. BTreeMap iteration is ascending by
    // (key_id, partner_id), so we get the CSR layout for free.
    let cap = per_partner.len();
    let mut keys: Vec<NodeId> = Vec::new();
    let mut offsets: Vec<u32> = vec![0];
    let mut partners: Vec<NodeId> = Vec::with_capacity(cap);
    let mut lsns: Vec<u64> = Vec::with_capacity(cap);
    let mut tombstones: Vec<bool> = Vec::with_capacity(cap);

    let mut cur_key_bytes: Option<[u8; 16]> = None;
    for ((k_bytes, p_bytes), (lsn, tomb)) in per_partner {
        let is_new_key = cur_key_bytes != Some(k_bytes);
        if is_new_key {
            if cur_key_bytes.is_some() {
                offsets.push(partners.len() as u32);
            }
            keys.push(NodeId::from_uuid(Uuid::from_bytes(k_bytes)));
            cur_key_bytes = Some(k_bytes);
        }
        partners.push(NodeId::from_uuid(Uuid::from_bytes(p_bytes)));
        lsns.push(lsn);
        tombstones.push(tomb);
    }
    // Sentinel.
    offsets.push(partners.len() as u32);
    debug_assert_eq!(offsets.len(), keys.len() + 1);

    Ok(EdgeAdjacency {
        edge_type: edge_type.to_string(),
        direction,
        manifest_version,
        keys,
        offsets,
        partners,
        lsns,
        tombstones,
    })
}

/// Cache-aware byte fetch. Mirrors `Snapshot::fetch_bytes` but takes the
/// cache by reference so callers don't need to clone it.
async fn fetch_body(
    store: &dyn object_store::ObjectStore,
    cache: Option<&SstCache>,
    absolute: &str,
) -> Result<bytes::Bytes> {
    use object_store::ObjectStoreExt;
    if let Some(c) = cache {
        if let Some(hit) = c.get(absolute) {
            return Ok(hit);
        }
    }
    let path = object_store::path::Path::from(absolute);
    let res = store.get(&path).await.map_err(Error::from)?;
    let body = res.bytes().await.map_err(Error::from)?;
    if let Some(c) = cache {
        c.insert(absolute.to_string(), body.clone());
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(byte: u8) -> NodeId {
        // Deterministic UUID with `byte` in the most significant slot, so
        // `NodeId::Ord` reflects the byte ordering for test predictability.
        let mut b = [0u8; 16];
        b[0] = byte;
        NodeId::from_uuid(Uuid::from_bytes(b))
    }

    fn build_adj_manual(
        edge_type: &str,
        direction: EdgeDirection,
        manifest_version: u64,
        rows: &[(NodeId, NodeId, u64, bool)],
    ) -> EdgeAdjacency {
        // Manually emulate the build process: sort rows, last-LSN wins,
        // materialise into the parallel arrays.
        use std::collections::BTreeMap;
        let mut per_partner: BTreeMap<(NodeId, NodeId), (u64, bool)> = BTreeMap::new();
        for (k, p, lsn, tomb) in rows {
            let entry = per_partner.entry((*k, *p));
            entry
                .and_modify(|(l, t)| {
                    if *lsn > *l {
                        *l = *lsn;
                        *t = *tomb;
                    }
                })
                .or_insert((*lsn, *tomb));
        }
        let mut keys = Vec::new();
        let mut offsets = vec![0];
        let mut partners = Vec::new();
        let mut lsns = Vec::new();
        let mut tombs = Vec::new();
        let mut cur = None;
        for ((k, p), (lsn, tomb)) in per_partner {
            if cur != Some(k) {
                if cur.is_some() {
                    offsets.push(partners.len() as u32);
                }
                keys.push(k);
                cur = Some(k);
            }
            partners.push(p);
            lsns.push(lsn);
            tombs.push(tomb);
        }
        offsets.push(partners.len() as u32);
        EdgeAdjacency {
            edge_type: edge_type.to_string(),
            direction,
            manifest_version,
            keys,
            offsets,
            partners,
            lsns,
            tombstones: tombs,
        }
    }

    #[test]
    fn empty_adjacency_returns_no_edges() {
        let adj = EdgeAdjacency::empty("KNOWS".to_string(), EdgeDirection::Forward, 1);
        assert_eq!(adj.key_count(), 0);
        assert_eq!(adj.edge_count(), 0);
        assert!(adj.lookup(nid(1)).is_none());
    }

    #[test]
    fn lookup_returns_partner_slices() {
        // src=1 → dst=2, dst=3; src=2 → dst=3
        let adj = build_adj_manual(
            "KNOWS",
            EdgeDirection::Forward,
            1,
            &[
                (nid(1), nid(2), 10, false),
                (nid(1), nid(3), 11, false),
                (nid(2), nid(3), 12, false),
            ],
        );
        assert_eq!(adj.key_count(), 2);
        assert_eq!(adj.edge_count(), 3);

        let slice1 = adj.lookup(nid(1)).expect("src=1 present");
        assert_eq!(slice1.partners, &[nid(2), nid(3)]);
        assert_eq!(slice1.lsns, &[10, 11]);
        assert_eq!(slice1.tombstones, &[false, false]);

        let slice2 = adj.lookup(nid(2)).expect("src=2 present");
        assert_eq!(slice2.partners, &[nid(3)]);
        assert_eq!(slice2.lsns, &[12]);

        assert!(adj.lookup(nid(99)).is_none());
    }

    #[test]
    fn build_manual_picks_last_lsn() {
        // Two rows for (1, 2): lsn=10 (tomb=false), lsn=20 (tomb=true).
        // The lsn=20 must win.
        let adj = build_adj_manual(
            "KNOWS",
            EdgeDirection::Forward,
            1,
            &[(nid(1), nid(2), 10, false), (nid(1), nid(2), 20, true)],
        );
        let slice = adj.lookup(nid(1)).unwrap();
        assert_eq!(slice.partners, &[nid(2)]);
        assert_eq!(slice.lsns, &[20]);
        assert_eq!(slice.tombstones, &[true]);
    }

    #[tokio::test]
    async fn cache_returns_same_arc_on_hit() {
        let cache = AdjacencyCache::new(1024 * 1024);
        let key = AdjacencyKey::new("tenants/acme", 1, "KNOWS", EdgeDirection::Forward);
        let key2 = key.clone();
        let built = cache
            .get_or_build(key, || async {
                Ok(EdgeAdjacency::empty(
                    "KNOWS".to_string(),
                    EdgeDirection::Forward,
                    1,
                ))
            })
            .await
            .unwrap();

        // Second resolve must reuse the same `Arc` (the closure should
        // never be invoked here; if it were, it would panic via the
        // assertion below).
        let again = cache
            .get_or_build(key2, || async {
                panic!("build closure must not be invoked on cache hit");
            })
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&built, &again));
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.builds(), 1);
        assert_eq!(cache.entries(), 1);
    }

    #[tokio::test]
    async fn cache_evicts_oldest_when_over_budget() {
        // Tiny cache: 256 B. Every empty EdgeAdjacency carries ~64 B of
        // overhead allowance + a few bytes for the edge_type label and
        // offsets sentinel, so a handful of inserts must overflow.
        let cache = AdjacencyCache::new(256);
        for v in 1..=10u64 {
            cache
                .get_or_build(
                    AdjacencyKey::new("tenants/acme", v, "KNOWS", EdgeDirection::Forward),
                    || async move {
                        let mut adj =
                            EdgeAdjacency::empty("KNOWS".to_string(), EdgeDirection::Forward, v);
                        // Inflate beyond 64 B overhead so the budget bites.
                        adj.tombstones = vec![false; 64];
                        Ok(adj)
                    },
                )
                .await
                .unwrap();
        }
        assert!(
            cache.evictions() > 0,
            "expected at least one eviction, got {}",
            cache.evictions()
        );
        // Highest manifest_version still present (FIFO-by-version).
        let hit = cache.get(&AdjacencyKey::new(
            "tenants/acme",
            10,
            "KNOWS",
            EdgeDirection::Forward,
        ));
        assert!(hit.is_some(), "newest entry must survive");
    }

    #[tokio::test]
    async fn same_triple_different_namespace_is_a_distinct_slot() {
        // Two tenants at the same manifest version with the same
        // (edge_type, direction) must resolve to their OWN CSR — the
        // shared cache must never hand tenant A's topology to tenant B.
        let cache = AdjacencyCache::new(1024 * 1024);
        let build = |ns: &'static str, partner: u8| {
            let key = AdjacencyKey::new(ns, 2, "KNOWS", EdgeDirection::Forward);
            let adj = build_adj_manual(
                "KNOWS",
                EdgeDirection::Forward,
                2,
                &[(nid(1), nid(partner), 10, false)],
            );
            (key, adj)
        };
        let (ka, adj_a) = build("tenants/a", 2);
        let (kb, adj_b) = build("tenants/b", 3);
        cache
            .get_or_build(ka.clone(), || async { Ok(adj_a) })
            .await
            .unwrap();
        cache
            .get_or_build(kb.clone(), || async { Ok(adj_b) })
            .await
            .unwrap();

        let got_a = cache.get(&ka).expect("a hit");
        let got_b = cache.get(&kb).expect("b hit");
        assert_eq!(got_a.lookup(nid(1)).unwrap().partners, &[nid(2)]);
        assert_eq!(got_b.lookup(nid(1)).unwrap().partners, &[nid(3)]);
    }

    #[tokio::test]
    async fn prune_namespace_removes_only_that_namespace() {
        let cache = AdjacencyCache::new(1024 * 1024);
        for (ns, v) in [("tenants/a", 1u64), ("tenants/a", 2), ("tenants/b", 1)] {
            cache
                .get_or_build(
                    AdjacencyKey::new(ns, v, "KNOWS", EdgeDirection::Forward),
                    || async move {
                        Ok(EdgeAdjacency::empty(
                            "KNOWS".to_string(),
                            EdgeDirection::Forward,
                            v,
                        ))
                    },
                )
                .await
                .unwrap();
        }
        assert_eq!(cache.namespace_entries("tenants/a"), 2);
        assert_eq!(cache.namespace_entries("tenants/b"), 1);
        let used_before = cache.used_bytes();

        cache.prune_namespace("tenants/a");
        assert_eq!(cache.namespace_entries("tenants/a"), 0);
        assert_eq!(cache.namespace_entries("tenants/b"), 1);
        assert!(
            cache.used_bytes() < used_before,
            "pruning must release budget bytes"
        );
    }

    #[test]
    fn adjacency_enabled_reads_env_var() {
        // Snapshot the var; serial within test so we don't race other tests.
        let original = std::env::var("NAMIDB_ADJACENCY").ok();
        std::env::set_var("NAMIDB_ADJACENCY", "1");
        assert!(adjacency_enabled());
        std::env::set_var("NAMIDB_ADJACENCY", "0");
        assert!(!adjacency_enabled());
        std::env::remove_var("NAMIDB_ADJACENCY");
        // The default was flipped the default to ON — anything but the literal
        // string "0" leaves the CSR adjacency path enabled.
        assert!(adjacency_enabled());
        // Restore.
        match original {
            Some(v) => std::env::set_var("NAMIDB_ADJACENCY", v),
            None => std::env::remove_var("NAMIDB_ADJACENCY"),
        }
    }
}
