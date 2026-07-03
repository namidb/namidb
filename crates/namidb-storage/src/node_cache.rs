//! Cross-snapshot `NodeView` cache (RFC-019).
//!
//! Mirror of [`crate::adjacency::AdjacencyCache`] but for the node side.
//! Profile data (`NAMIDB_PROFILE_DUMP=1 NAMIDB_ADJACENCY=1`
//! on IC09) showed `Snapshot::lookup_node` was 99.4% of the wall-clock
//! while the existing per-snapshot cache only hit 9% of calls — the
//! intra-snapshot scope drops the answers after every query and the
//! bench (and any interactive workload) builds a fresh `Snapshot` per
//! query. Cross-snapshot sharing, keyed by `(namespace, manifest_version,
//! label, node_id)`, lets a warmup pay the SST walk once and amortise it
//! across every subsequent query against the same manifest version. The
//! namespace component makes the process-wide shared instance
//! ([`shared_node_cache`]) safe across tenants.
//!
//! ## Negative caching
//!
//! `CachedNodeView` is `Option<NodeView>`. We **also cache `None`** —
//! a successful resolution to "absent / tombstoned" is still expensive
//! (it pays the same bloom probe + body walk + LSN merge). Caching it
//! is correct because the cache key includes `manifest_version`: once
//! the writer commits, the cache slot for the next version is fresh.
//!
//! ## L1 + L2
//!
//! The full lookup path is three tiers:
//!
//! 1. **L1** — `Snapshot::node_cache` (per-snapshot `Mutex<HashMap>`),
//! introduced.
//! 2. **L2** — this cache (`Arc`-shared across snapshots).
//! 3. **L3** — `Snapshot::lookup_node_uncached` (the SST walk).
//!
//! On L2 hit we promote into L1 so the rest of the snapshot bypasses L2
//! entirely. On L3 hit we insert into both L1 and L2.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use namidb_core::NodeId;

use crate::read::NodeView;

/// Default budget for a [`NodeViewCache`]: 256 MiB. Override via
/// `NAMIDB_NODE_CACHE_BUDGET_MIB`.
pub const DEFAULT_NODE_CACHE_BUDGET_MIB: usize = 256;

/// Read `NAMIDB_NODE_CACHE` and return `false` only for `"0"`. Anything
/// else (unset, `"1"`, garbage) returns `true`. Default flipped
/// — see [`crate::adjacency::adjacency_enabled`] for the rationale.
pub fn node_cache_enabled() -> bool {
    std::env::var("NAMIDB_NODE_CACHE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Read `NAMIDB_NODE_CACHE_BUDGET_MIB` or fall back to
/// [`DEFAULT_NODE_CACHE_BUDGET_MIB`].
pub fn node_cache_budget_bytes() -> usize {
    let mib = std::env::var("NAMIDB_NODE_CACHE_BUDGET_MIB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_NODE_CACHE_BUDGET_MIB);
    mib.saturating_mul(1024 * 1024)
}

/// Process-wide shared [`NodeViewCache`]: one instance for every
/// [`crate::WriterSession`] the process opens, so
/// `NAMIDB_NODE_CACHE_BUDGET_MIB` bounds the PROCESS, not each session.
/// Unlike the [`crate::cache::SstCache`] (whose keys are absolute paths and
/// therefore namespace-safe by construction), sharing this cache is only
/// sound because [`NodeCacheKey`] embeds the namespace prefix — the bare
/// `(manifest_version, label, node_id)` triple collides across tenants.
///
/// The enable flag and budget are read once, on first use. Returns `None`
/// when `NAMIDB_NODE_CACHE=0` at first use. Callers needing a private
/// instance inject one via
/// [`crate::ingest::WriterSession::open_with_caches`].
pub fn shared_node_cache() -> Option<Arc<NodeViewCache>> {
    static SHARED: OnceLock<Option<Arc<NodeViewCache>>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            node_cache_enabled().then(|| Arc::new(NodeViewCache::new(node_cache_budget_bytes())))
        })
        .clone()
}

/// Compound cache key. Hash by all four fields so two snapshots that share
/// `namespace` + `manifest_version` see the same slot for the same
/// `(label, node_id)`.
///
/// `namespace` is the object-store namespace prefix (`<root>/<ns>`, no
/// trailing slash — [`crate::paths::NamespacePaths::namespace_prefix`]).
/// It is part of the key because the cache is shared process-wide: two
/// tenants can hold the same `(manifest_version, label, node_id)` triple
/// with different data (manifest versions count per namespace and node
/// ids are caller-supplied), so omitting it would serve one tenant's rows
/// to another. `Arc<str>` keeps per-entry clones at pointer cost — every
/// key of one snapshot shares the same allocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeCacheKey {
    pub namespace: Arc<str>,
    pub manifest_version: u64,
    pub label: String,
    pub node_id: NodeId,
}

impl NodeCacheKey {
    pub fn new(
        namespace: impl Into<Arc<str>>,
        manifest_version: u64,
        label: impl Into<String>,
        node_id: NodeId,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            manifest_version,
            label: label.into(),
            node_id,
        }
    }
}

/// `None` ⇔ "absent / tombstoned at this manifest version". `Some(view)`
/// ⇔ "materialised NodeView".
pub type CachedNodeView = Option<NodeView>;

#[derive(Debug, Default)]
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
}

/// Inner cache state, guarded by one mutex so the map, its eviction order, and
/// the byte accounting stay consistent.
struct Inner {
    /// key → (cached view, insertion sequence). The sequence disambiguates the
    /// eviction-order entry so an overwrite can remove the stale one in O(log n).
    map: HashMap<NodeCacheKey, (CachedNodeView, u64)>,
    /// Eviction order: `(manifest_version, seq) → key`. `pop_first` yields the
    /// victim (oldest manifest version, then oldest insertion) in O(log n) — no
    /// full-map scan or per-key String clone.
    order: BTreeMap<(u64, u64), NodeCacheKey>,
    next_seq: u64,
    used_bytes: usize,
}

/// Process-wide cross-snapshot NodeView cache.
pub struct NodeViewCache {
    inner: Mutex<Inner>,
    capacity_bytes: usize,
    stats: Arc<CacheStats>,
}

impl std::fmt::Debug for NodeViewCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        let entries = inner.map.len();
        let used = inner.used_bytes;
        f.debug_struct("NodeViewCache")
            .field("entries", &entries)
            .field("used_bytes", &used)
            .field("capacity_bytes", &self.capacity_bytes)
            .field("hits", &self.stats.hits.load(Ordering::Relaxed))
            .field("misses", &self.stats.misses.load(Ordering::Relaxed))
            .field("inserts", &self.stats.inserts.load(Ordering::Relaxed))
            .field("evictions", &self.stats.evictions.load(Ordering::Relaxed))
            .finish()
    }
}

impl NodeViewCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: BTreeMap::new(),
                next_seq: 0,
                used_bytes: 0,
            }),
            capacity_bytes: capacity_bytes.max(1),
            stats: Arc::new(CacheStats::default()),
        }
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn used_bytes(&self) -> usize {
        self.inner.lock().unwrap().used_bytes
    }

    pub fn entries(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }

    pub fn hits(&self) -> u64 {
        self.stats.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.stats.misses.load(Ordering::Relaxed)
    }
    pub fn inserts(&self) -> u64 {
        self.stats.inserts.load(Ordering::Relaxed)
    }
    pub fn evictions(&self) -> u64 {
        self.stats.evictions.load(Ordering::Relaxed)
    }

    /// Probe the cache. Returns `Some(cached)` on hit (positive or
    /// negative), `None` on miss. Increments hit/miss counters.
    pub fn get(&self, key: &NodeCacheKey) -> Option<CachedNodeView> {
        let inner = self.inner.lock().unwrap();
        match inner.map.get(key) {
            Some((view, _seq)) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                Some(view.clone())
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert (or overwrite) the entry for `key`. Evicts oldest
    /// `manifest_version` entries to fit `capacity_bytes` if necessary.
    pub fn insert(&self, key: NodeCacheKey, view: CachedNodeView) {
        let weight = entry_weight(&key, &view);
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        let inner = &mut *self.inner.lock().unwrap();

        // If we're overwriting an existing entry, reclaim its weight and drop
        // its stale eviction-order entry (keyed by its old seq) first.
        if let Some((prev, prev_seq)) = inner.map.get(&key) {
            let prev_weight = entry_weight(&key, prev);
            let prev_order_key = (key.manifest_version, *prev_seq);
            inner.used_bytes = inner.used_bytes.saturating_sub(prev_weight);
            inner.order.remove(&prev_order_key);
        }

        // Evict oldest (manifest_version, seq) entries in O(log n) each until the
        // new entry fits. The new key is not yet in `order`, so it is never a
        // victim of its own insert.
        while inner.used_bytes + weight > self.capacity_bytes {
            let Some((&victim_ord, _)) = inner.order.iter().next() else {
                break; // nothing left to evict
            };
            let victim_key = inner.order.remove(&victim_ord).unwrap();
            if let Some((victim, _)) = inner.map.remove(&victim_key) {
                let victim_weight = entry_weight(&victim_key, &victim);
                inner.used_bytes = inner.used_bytes.saturating_sub(victim_weight);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        let seq = inner.next_seq;
        inner.next_seq += 1;
        inner.order.insert((key.manifest_version, seq), key.clone());
        inner.map.insert(key, (view, seq));
        inner.used_bytes = inner.used_bytes.saturating_add(weight);
    }

    /// Eagerly drop every entry belonging to `namespace` (the namespace
    /// prefix embedded in [`NodeCacheKey::namespace`]). Called when a
    /// multi-tenant host evicts a namespace — its entries in the shared
    /// cache are dead weight (a later reopen continues the same manifest
    /// lineage, so leftover entries would still be CORRECT; this is a
    /// memory-reclaim, not a correctness, operation).
    pub fn prune_namespace(&self, namespace: &str) {
        let inner = &mut *self.inner.lock().unwrap();
        let victims: Vec<(NodeCacheKey, u64)> = inner
            .map
            .iter()
            .filter(|(k, _)| k.namespace.as_ref() == namespace)
            .map(|(k, (_, seq))| (k.clone(), *seq))
            .collect();
        for (key, seq) in victims {
            inner.order.remove(&(key.manifest_version, seq));
            if let Some((view, _)) = inner.map.remove(&key) {
                inner.used_bytes = inner
                    .used_bytes
                    .saturating_sub(entry_weight(&key, &view));
            }
        }
    }

    /// Count of entries whose key belongs to `namespace`. Observability +
    /// test probe for namespace isolation and [`Self::prune_namespace`].
    pub fn namespace_entries(&self, namespace: &str) -> usize {
        self.inner
            .lock()
            .unwrap()
            .map
            .keys()
            .filter(|k| k.namespace.as_ref() == namespace)
            .count()
    }
}

/// Budget weight of one cache entry: the view estimate plus the key's own
/// heap footprint. The `namespace` component counts pointer-size only —
/// the `Arc<str>` buffer is shared by every key of a snapshot.
fn entry_weight(key: &NodeCacheKey, view: &CachedNodeView) -> usize {
    approx_size(view) + key.label.capacity() + std::mem::size_of::<Arc<str>>() + 32
}

/// Conservative size estimate for a [`CachedNodeView`]. Counts labels +
/// property name + per-value allowance + invariant overhead. Used for
/// budget accounting — exact tracking would require deep `Value` walks
/// which are not worth it.
fn approx_size(view: &CachedNodeView) -> usize {
    match view {
        None => 32, // overhead allowance for cached-miss
        Some(v) => {
            let prop_bytes: usize = v
                .properties
                .keys()
                .map(|k| k.capacity() + 64) // 64 = rough Value enum size
                .sum();
            v.labels.iter().map(|l| l.capacity()).sum::<usize>() + prop_bytes + 128
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use namidb_core::Value;
    use uuid::Uuid;

    use super::*;

    fn nid(byte: u8) -> NodeId {
        let mut b = [0u8; 16];
        b[0] = byte;
        NodeId::from_uuid(Uuid::from_bytes(b))
    }

    fn make_view(name: &str) -> NodeView {
        let mut props = BTreeMap::new();
        props.insert("name".to_string(), Value::Str(name.into()));
        NodeView {
            id: nid(1),
            labels: BTreeSet::from(["Person".to_string()]),
            properties: props,
            lsn: 10,
            schema_version: 1,
        }
    }

    const NS: &str = "tenants/acme";

    #[test]
    fn miss_returns_none_increments_misses() {
        let c = NodeViewCache::new(1024 * 1024);
        let k = NodeCacheKey::new(NS, 1, "Person", nid(1));
        assert!(c.get(&k).is_none());
        assert_eq!(c.misses(), 1);
        assert_eq!(c.hits(), 0);
    }

    #[test]
    fn insert_then_get_returns_view() {
        let c = NodeViewCache::new(1024 * 1024);
        let k = NodeCacheKey::new(NS, 1, "Person", nid(1));
        c.insert(k.clone(), Some(make_view("Alice")));
        let got = c.get(&k).expect("hit");
        assert_eq!(got.as_ref().map(|v| v.properties.len()), Some(1));
        assert_eq!(c.hits(), 1);
        assert_eq!(c.inserts(), 1);
    }

    #[test]
    fn negative_cache_returns_inner_none_on_hit() {
        // Cache a negative (key resolved to "absent"). The L2 hit must
        // surface `Some(None)`, not be confused with a cache miss.
        let c = NodeViewCache::new(1024 * 1024);
        let k = NodeCacheKey::new(NS, 1, "Person", nid(7));
        c.insert(k.clone(), None);
        let got = c.get(&k).expect("hit on negative cache");
        assert!(got.is_none(), "cached negative should still hit");
        assert_eq!(c.hits(), 1);
        assert_eq!(c.misses(), 0);
    }

    #[test]
    fn evicts_oldest_manifest_version_when_over_budget() {
        // Tight budget so a few inserts overflow. Each insert with a
        // distinct (version, label) tuple is its own entry.
        let c = NodeViewCache::new(2048);
        for v in 1..=20u64 {
            let k = NodeCacheKey::new(NS, v, "Person", nid(1));
            let mut view = make_view("padding-padding-padding-padding-padding");
            // Inflate the view so each entry is meaningful in bytes.
            for i in 0..8 {
                view.properties
                    .insert(format!("k_{i}"), Value::Str("v".repeat(32)));
            }
            c.insert(k, Some(view));
        }
        assert!(
            c.evictions() > 0,
            "expected at least one eviction, got {}",
            c.evictions()
        );
        // Most-recently-inserted version must survive.
        let k_recent = NodeCacheKey::new(NS, 20, "Person", nid(1));
        assert!(c.get(&k_recent).is_some(), "newest version must survive");
    }

    #[test]
    fn same_triple_different_namespace_is_a_distinct_slot() {
        // Two tenants at the same manifest version with the same
        // (label, node_id) must never see each other's rows.
        let c = NodeViewCache::new(1024 * 1024);
        let ka = NodeCacheKey::new("tenants/a", 2, "Person", nid(1));
        let kb = NodeCacheKey::new("tenants/b", 2, "Person", nid(1));
        c.insert(ka.clone(), Some(make_view("from-a")));
        c.insert(kb.clone(), Some(make_view("from-b")));

        let name = |v: CachedNodeView| match v.unwrap().properties.get("name") {
            Some(Value::Str(s)) => s.clone(),
            other => panic!("unexpected name: {other:?}"),
        };
        assert_eq!(name(c.get(&ka).expect("a hit")), "from-a");
        assert_eq!(name(c.get(&kb).expect("b hit")), "from-b");
    }

    #[test]
    fn prune_namespace_removes_only_that_namespace() {
        let c = NodeViewCache::new(1024 * 1024);
        c.insert(NodeCacheKey::new("tenants/a", 2, "Person", nid(1)), Some(make_view("a1")));
        c.insert(NodeCacheKey::new("tenants/a", 3, "Person", nid(2)), Some(make_view("a2")));
        c.insert(NodeCacheKey::new("tenants/b", 2, "Person", nid(1)), Some(make_view("b1")));
        assert_eq!(c.namespace_entries("tenants/a"), 2);
        assert_eq!(c.namespace_entries("tenants/b"), 1);
        let used_before = c.used_bytes();

        c.prune_namespace("tenants/a");
        assert_eq!(c.namespace_entries("tenants/a"), 0);
        assert_eq!(c.namespace_entries("tenants/b"), 1);
        assert!(
            c.used_bytes() < used_before,
            "pruning must release budget bytes"
        );
        assert!(
            c.get(&NodeCacheKey::new("tenants/b", 2, "Person", nid(1)))
                .is_some(),
            "sibling namespace survives the prune"
        );
        // The pruned entries' order slots are gone too: filling the cache
        // again must not underflow or double-free the byte accounting.
        for v in 1..=5u64 {
            c.insert(NodeCacheKey::new("tenants/a", v, "Person", nid(3)), Some(make_view("x")));
        }
        assert_eq!(c.namespace_entries("tenants/a"), 5);
    }

    #[test]
    fn env_var_helpers() {
        let original = std::env::var("NAMIDB_NODE_CACHE").ok();
        std::env::set_var("NAMIDB_NODE_CACHE", "1");
        assert!(node_cache_enabled());
        std::env::set_var("NAMIDB_NODE_CACHE", "0");
        assert!(!node_cache_enabled());
        std::env::remove_var("NAMIDB_NODE_CACHE");
        // flipped the default to ON.
        assert!(node_cache_enabled());
        match original {
            Some(v) => std::env::set_var("NAMIDB_NODE_CACHE", v),
            None => std::env::remove_var("NAMIDB_NODE_CACHE"),
        }
    }
}
