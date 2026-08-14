//! Process-wide cache capacity planning.
//!
//! The storage caches historically exposed independent per-tier MiB knobs.
//! Those knobs remain ceilings, but [`NAMIDB_CACHE_MAX_BYTES`] now places one
//! deterministic upper bound over every shared cache tier. When the requested
//! ceilings do not fit, all active tiers are scaled proportionally and the
//! rounding remainder is assigned in a stable tier order.
//!
//! [`NAMIDB_CACHE_MAX_BYTES`]: https://github.com/namidb/namidb#configuration

use std::sync::OnceLock;

use crate::adjacency::DEFAULT_ADJACENCY_BUDGET_MIB;
use crate::cache::{
    DEFAULT_BLOOM_FILTER_CACHE_BUDGET_MIB, DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB,
    DEFAULT_EDGE_READER_CACHE_BUDGET_MIB, DEFAULT_EDGE_STREAM_CACHE_BUDGET_MIB,
    DEFAULT_PATH_REGISTRY_CACHE_BUDGET_MIB, DEFAULT_PROPERTY_SIDECAR_CACHE_BUDGET_MIB,
    DEFAULT_SST_CACHE_BUDGET_MIB, DEFAULT_SST_METADATA_CACHE_BUDGET_MIB,
    DEFAULT_TEXT_INDEX_CACHE_BUDGET_MIB, DEFAULT_VECTOR_INDEX_CACHE_BUDGET_MIB,
};
use crate::node_cache::DEFAULT_NODE_CACHE_BUDGET_MIB;
use crate::range_cache::DEFAULT_RAM_PAGE_CACHE_MAX_BYTES;

/// Default aggregate capacity of all process-wide caches: 1 GiB.
pub const DEFAULT_CACHE_MAX_BYTES: usize = 1024 * 1024 * 1024;

const CACHE_TIER_COUNT: usize = 12;

/// Effective capacities after applying the process-wide maximum.
///
/// Individual fields are the hard admission ceilings used by their cache
/// tier. Their sum is always at most [`Self::max_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheCapacities {
    /// Configured aggregate maximum.
    pub max_bytes: usize,
    pub(crate) sst_body_bytes: usize,
    pub(crate) decoded_node_row_group_bytes: usize,
    pub(crate) property_sidecar_bytes: usize,
    pub(crate) sst_metadata_bytes: usize,
    pub(crate) edge_stream_bytes: usize,
    pub(crate) edge_reader_bytes: usize,
    pub(crate) bloom_filter_bytes: usize,
    /// Shared decoded-search-index pool. Vector and full-text indexes are
    /// mutually evictable so either workload can use the whole assignment
    /// instead of being rejected by an artificial per-kind split.
    pub(crate) search_index_bytes: usize,
    /// Path/admission metadata used to keep every `SstCache` tier in sync with
    /// manifests and natural Foyer evictions. This is a real cache tier: its
    /// strings and hash-table nodes are retained independently from values and
    /// therefore must be carved out of the process-wide maximum.
    pub(crate) path_registry_bytes: usize,
    /// RAM portion of the immutable object-range cache. This is kept inside
    /// `NAMIDB_CACHE_MAX_BYTES`; the independently configured NVMe device
    /// capacity is deliberately not part of the process memory budget.
    pub(crate) range_page_bytes: usize,
    pub(crate) node_view_bytes: usize,
    pub(crate) adjacency_bytes: usize,
}

impl CacheCapacities {
    fn requested_from_env(max_bytes: usize) -> Self {
        let sst_enabled = env_enabled("NAMIDB_SST_CACHE");
        let node_enabled = env_enabled("NAMIDB_NODE_CACHE");
        // A complete CSR is O(edge_count) resident state. Unlike the small,
        // bounded SST/node caches it is an explicit latency-for-memory opt-in.
        let adjacency_enabled = env_opt_in("NAMIDB_ADJACENCY");

        let sst = |name, default_mib| {
            if sst_enabled {
                legacy_budget_bytes(name, default_mib)
            } else {
                0
            }
        };

        Self {
            max_bytes,
            sst_body_bytes: sst("NAMIDB_SST_CACHE_BUDGET_MIB", DEFAULT_SST_CACHE_BUDGET_MIB),
            decoded_node_row_group_bytes: sst(
                "NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB",
                DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB,
            ),
            property_sidecar_bytes: sst(
                "NAMIDB_PROPERTY_SIDECAR_CACHE_BUDGET_MIB",
                DEFAULT_PROPERTY_SIDECAR_CACHE_BUDGET_MIB,
            ),
            sst_metadata_bytes: sst(
                "NAMIDB_SST_METADATA_CACHE_BUDGET_MIB",
                DEFAULT_SST_METADATA_CACHE_BUDGET_MIB,
            ),
            edge_stream_bytes: sst(
                "NAMIDB_EDGE_STREAM_CACHE_BUDGET_MIB",
                DEFAULT_EDGE_STREAM_CACHE_BUDGET_MIB,
            ),
            edge_reader_bytes: sst(
                "NAMIDB_EDGE_READER_CACHE_BUDGET_MIB",
                DEFAULT_EDGE_READER_CACHE_BUDGET_MIB,
            ),
            bloom_filter_bytes: sst(
                "NAMIDB_BLOOM_FILTER_CACHE_BUDGET_MIB",
                DEFAULT_BLOOM_FILTER_CACHE_BUDGET_MIB,
            ),
            search_index_bytes: (if cfg!(feature = "text-index") {
                sst(
                    "NAMIDB_TEXT_INDEX_CACHE_BUDGET_MIB",
                    DEFAULT_TEXT_INDEX_CACHE_BUDGET_MIB,
                )
            } else {
                0
            })
            .saturating_add(if cfg!(feature = "vector-index") {
                sst(
                    "NAMIDB_VECTOR_INDEX_CACHE_BUDGET_MIB",
                    DEFAULT_VECTOR_INDEX_CACHE_BUDGET_MIB,
                )
            } else {
                0
            }),
            path_registry_bytes: sst(
                "NAMIDB_CACHE_PATH_REGISTRY_BUDGET_MIB",
                DEFAULT_PATH_REGISTRY_CACHE_BUDGET_MIB,
            ),
            range_page_bytes: requested_ram_page_cache_bytes(),
            node_view_bytes: if node_enabled {
                legacy_budget_bytes(
                    "NAMIDB_NODE_CACHE_BUDGET_MIB",
                    DEFAULT_NODE_CACHE_BUDGET_MIB,
                )
            } else {
                0
            },
            adjacency_bytes: if adjacency_enabled {
                legacy_budget_bytes("NAMIDB_ADJACENCY_BUDGET_MIB", DEFAULT_ADJACENCY_BUDGET_MIB)
            } else {
                0
            },
        }
    }

    fn scaled_to(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        let requested = self.as_array();
        let requested_total = requested
            .iter()
            .fold(0u128, |sum, value| sum + *value as u128);

        if max_bytes == 0 {
            return Self::from_array(max_bytes, [0; CACHE_TIER_COUNT]);
        }
        if requested_total <= max_bytes as u128 {
            return self;
        }

        let mut scaled = [0usize; CACHE_TIER_COUNT];
        for (slot, requested_bytes) in scaled.iter_mut().zip(requested) {
            *slot = ((requested_bytes as u128 * max_bytes as u128) / requested_total) as usize;
        }

        // Flooring loses fewer than CACHE_TIER_COUNT bytes. Assign those bytes
        // in stable tier order so the result is exact and reproducible.
        let mut remainder = max_bytes.saturating_sub(scaled.iter().sum());
        while remainder > 0 {
            let mut progressed = false;
            for (slot, requested_bytes) in scaled.iter_mut().zip(requested) {
                if remainder == 0 {
                    break;
                }
                if *slot < requested_bytes {
                    *slot += 1;
                    remainder -= 1;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }

        Self::from_array(max_bytes, scaled)
    }

    /// Resolve an optional dedicated search-index assignment without ever
    /// exceeding the aggregate maximum.
    ///
    /// An explicit `NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES` is a reservation
    /// inside (not in addition to) `NAMIDB_CACHE_MAX_BYTES`: search gets that
    /// amount first and every other enabled tier is proportionally fitted into
    /// the remainder. Without the override, search participates in the legacy
    /// proportional plan using the sum of the old vector/text requests.
    fn scaled_with_search_reservation(
        mut self,
        max_bytes: usize,
        search_reservation: Option<usize>,
    ) -> Self {
        let Some(search_reservation) = search_reservation else {
            return self.scaled_to(max_bytes);
        };
        if self.search_index_bytes == 0 || max_bytes == 0 {
            return self.scaled_to(max_bytes);
        }

        let reserved = search_reservation.min(max_bytes);
        self.search_index_bytes = 0;
        let mut resolved = self.scaled_to(max_bytes.saturating_sub(reserved));
        resolved.max_bytes = max_bytes;
        resolved.search_index_bytes = reserved;
        resolved
    }

    fn as_array(self) -> [usize; CACHE_TIER_COUNT] {
        [
            self.sst_body_bytes,
            self.decoded_node_row_group_bytes,
            self.property_sidecar_bytes,
            self.sst_metadata_bytes,
            self.edge_stream_bytes,
            self.edge_reader_bytes,
            self.bloom_filter_bytes,
            self.search_index_bytes,
            self.path_registry_bytes,
            self.range_page_bytes,
            self.node_view_bytes,
            self.adjacency_bytes,
        ]
    }

    fn from_array(max_bytes: usize, values: [usize; CACHE_TIER_COUNT]) -> Self {
        Self {
            max_bytes,
            sst_body_bytes: values[0],
            decoded_node_row_group_bytes: values[1],
            property_sidecar_bytes: values[2],
            sst_metadata_bytes: values[3],
            edge_stream_bytes: values[4],
            edge_reader_bytes: values[5],
            bloom_filter_bytes: values[6],
            search_index_bytes: values[7],
            path_registry_bytes: values[8],
            range_page_bytes: values[9],
            node_view_bytes: values[10],
            adjacency_bytes: values[11],
        }
    }

    /// Sum of the nine active `SstCache` tier ceilings, including the bounded
    /// path/admission registry.
    pub fn sst_capacity_bytes(&self) -> usize {
        self.as_array()[..9]
            .iter()
            .fold(0usize, |sum, value| sum.saturating_add(*value))
    }

    /// Effective shared decoded vector/full-text index ceiling.
    pub fn search_index_capacity_bytes(&self) -> usize {
        self.search_index_bytes
    }

    /// Effective ceiling for path/admission registry metadata.
    pub fn path_registry_capacity_bytes(&self) -> usize {
        self.path_registry_bytes
    }

    /// Effective RAM budget assigned to immutable object-range pages.
    ///
    /// For a hybrid cache this includes the Foyer write buffer as well as the
    /// resident memory tier, so enabling NVMe never adds an unaccounted fixed
    /// buffer on top of `NAMIDB_CACHE_MAX_BYTES`.
    pub fn range_page_capacity_bytes(&self) -> usize {
        self.range_page_bytes
    }

    /// Effective `NodeViewCache` ceiling.
    pub fn node_view_capacity_bytes(&self) -> usize {
        self.node_view_bytes
    }

    /// Effective `AdjacencyCache` ceiling.
    pub fn adjacency_capacity_bytes(&self) -> usize {
        self.adjacency_bytes
    }

    /// Sum of every effective shared-cache ceiling.
    pub fn total_capacity_bytes(&self) -> usize {
        self.as_array()
            .iter()
            .fold(0usize, |sum, value| sum.saturating_add(*value))
    }
}

fn parse_optional_exact_bytes(
    name: &str,
    raw: Option<std::ffi::OsString>,
    empty_is_unset: bool,
) -> Result<Option<usize>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let raw = raw
        .into_string()
        .map_err(|_| format!("{name} is not valid UTF-8"))?;
    let raw = raw.trim();
    if raw.is_empty() && empty_is_unset {
        return Ok(None);
    }
    raw.parse::<usize>()
        .map(Some)
        .map_err(|error| format!("{name} must be an exact non-negative byte count: {error}"))
}

/// Validate the aggregate cache configuration before any process-wide cache
/// singleton samples it.
///
/// Storage remains usable as an embedded library when an application does not
/// call this function, but the official server invokes it at startup and
/// fails closed. That prevents a typo in a production memory rail from
/// silently selecting the 1 GiB default.
pub fn validate_cache_configuration() -> Result<(), String> {
    parse_optional_exact_bytes(
        "NAMIDB_CACHE_MAX_BYTES",
        std::env::var_os("NAMIDB_CACHE_MAX_BYTES"),
        false,
    )?;
    parse_optional_exact_bytes(
        "NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES",
        std::env::var_os("NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES"),
        true,
    )?;
    // The remaining exact-byte memory rails resolve through their own lazily
    // sampled parsers, each of which falls back to a much larger default on a
    // malformed value. Validating them here keeps a typo in any one of them
    // from silently widening the process memory plan.
    for name in [
        "NAMIDB_RAM_PAGE_CACHE_MAX_BYTES",
        crate::search_workspace::SEARCH_WORKSPACE_MAX_BYTES_ENV,
        crate::search_workspace::SEARCH_MAX_RESULT_BYTES_ENV,
        crate::search_workspace::BM25_MAX_DOCUMENT_BYTES_ENV,
    ] {
        parse_optional_exact_bytes(name, std::env::var_os(name), true)?;
    }
    Ok(())
}

/// Read the exact-byte aggregate cache limit.
///
/// An unset value uses [`DEFAULT_CACHE_MAX_BYTES`]. The official server
/// validates malformed values before opening storage. Embedded callers that
/// bypass startup validation retain the historical fallback, but receive an
/// explicit error log instead of a silent, potentially surprising 1 GiB cap.
/// `0` disables all process-wide shared caches.
pub fn cache_max_bytes() -> usize {
    match parse_optional_exact_bytes(
        "NAMIDB_CACHE_MAX_BYTES",
        std::env::var_os("NAMIDB_CACHE_MAX_BYTES"),
        false,
    ) {
        Ok(Some(bytes)) => bytes,
        Ok(None) => DEFAULT_CACHE_MAX_BYTES,
        Err(error) => {
            tracing::error!(
                %error,
                fallback_bytes = DEFAULT_CACHE_MAX_BYTES,
                "invalid aggregate cache limit; embedded caller bypassed startup validation"
            );
            DEFAULT_CACHE_MAX_BYTES
        }
    }
}

/// Optional exact-byte reservation for the shared decoded vector/full-text
/// index pool. The value is carved out of [`cache_max_bytes`], never added on
/// top of it. `None` retains proportional legacy allocation; `Some(0)`
/// deliberately disables decoded search-index caching.
pub fn search_index_cache_max_bytes() -> Option<usize> {
    match parse_optional_exact_bytes(
        "NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES",
        std::env::var_os("NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES"),
        true,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::error!(
                %error,
                "invalid search-index cache reservation; embedded caller bypassed startup validation"
            );
            None
        }
    }
}

/// Requested exact-byte RAM budget for immutable object-range pages.
///
/// An explicit `NAMIDB_RAM_PAGE_CACHE_MAX_BYTES` wins. When only the paired
/// NVMe settings are present, a conservative 64 MiB request is supplied so
/// the hybrid cache has a bounded memory/write-buffer tier. With no new
/// settings this returns zero; see `default_ram_page_cache_request` for why
/// that is still the default despite the paged read paths needing it.
pub fn requested_ram_page_cache_bytes() -> usize {
    match parse_optional_exact_bytes(
        "NAMIDB_RAM_PAGE_CACHE_MAX_BYTES",
        std::env::var_os("NAMIDB_RAM_PAGE_CACHE_MAX_BYTES"),
        true,
    ) {
        Ok(Some(bytes)) => bytes,
        Ok(None) => default_ram_page_cache_request(),
        Err(error) => {
            // Resolving to zero here would silently disable an explicitly
            // requested memory tier, which is the opposite of the operator's
            // intent. The official server rejects this at startup; an embedded
            // caller that skipped validation gets the unset behaviour instead.
            tracing::error!(
                %error,
                "invalid RAM page cache reservation; embedded caller bypassed startup validation"
            );
            default_ram_page_cache_request()
        }
    }
}

fn default_ram_page_cache_request() -> usize {
    // This tier SHOULD be on by default: the paged readers — V5/VG6 vector
    // bodies, FT4 postings, paged property indexes and the paged edge CSR —
    // serve point lookups out of small authenticated ranges, and without it
    // every one of those is a real object-store round trip. It is a carve-out
    // of `NAMIDB_CACHE_MAX_BYTES` rather than an addition to it, so enabling
    // it would not raise the process memory bound.
    //
    // Both correctness blockers that kept this tier off are fixed: a RAM-only
    // tier is a pure in-memory cache with no background workers and no runtime
    // affinity (see `RangeCacheBackend`), and an *inferred* immutability pin —
    // a store exposing neither version nor ETag — now refuses the shared cache
    // outright instead of colliding on a constant path+range token (see
    // `PinnedObjectRangeSource::from_create_only_meta`).
    //
    // What still gates the flip is test debt, not a defect: several
    // namidb-query integration tests (`exec_hybrid_search.rs`
    // indexed_sparse_filter among them) pin *exact* object-store GET counts
    // that a warm cache legitimately reduces. Audit and re-anchor those
    // assertions to what they actually protect, then return
    // `DEFAULT_RAM_PAGE_CACHE_MAX_BYTES` unconditionally here.
    let has_path = std::env::var("NAMIDB_NVME_CACHE_PATH")
        .ok()
        .is_some_and(|path| !path.trim().is_empty());
    let has_capacity = std::env::var("NAMIDB_NVME_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .is_some_and(|bytes| bytes > 0);
    if has_path && has_capacity {
        DEFAULT_RAM_PAGE_CACHE_MAX_BYTES
    } else {
        0
    }
}

/// Process-wide effective capacity plan. Environment configuration is sampled
/// once because the shared cache instances cannot be resized after creation.
pub fn shared_cache_capacities() -> CacheCapacities {
    static CAPACITIES: OnceLock<CacheCapacities> = OnceLock::new();
    *CAPACITIES.get_or_init(|| {
        let max_bytes = cache_max_bytes();
        let capacities = CacheCapacities::requested_from_env(max_bytes)
            .scaled_with_search_reservation(max_bytes, search_index_cache_max_bytes());
        tracing::info!(
            cache_max_bytes = capacities.max_bytes,
            cache_assigned_bytes = capacities.total_capacity_bytes(),
            sst_cache_bytes = capacities.sst_capacity_bytes(),
            search_index_cache_bytes = capacities.search_index_capacity_bytes(),
            cache_path_registry_bytes = capacities.path_registry_capacity_bytes(),
            range_page_cache_bytes = capacities.range_page_capacity_bytes(),
            node_cache_bytes = capacities.node_view_capacity_bytes(),
            adjacency_cache_bytes = capacities.adjacency_capacity_bytes(),
            "resolved process-wide cache capacities"
        );
        capacities
    })
}

/// Sum of the effective capacities assigned to all shared cache tiers.
pub fn shared_cache_capacity_bytes() -> usize {
    shared_cache_capacities().total_capacity_bytes()
}

/// Sum of the currently resident, cache-accounted bytes in all shared tiers.
///
/// This reports cache residency only; objects cloned into active queries may
/// outlive their cache entry and are outside this P0 counter.
pub fn shared_cache_usage_bytes() -> usize {
    let sst = crate::cache::shared_sst_cache()
        .as_ref()
        .map_or(0, crate::cache::SstCache::aggregate_usage_bytes);
    let nodes = crate::node_cache::shared_node_cache()
        .as_ref()
        .map_or(0, |cache| cache.used_bytes());
    let adjacency = crate::adjacency::shared_adjacency_cache()
        .as_ref()
        .map_or(0, |cache| cache.used_bytes());
    let range_pages = crate::range_cache::shared_range_cache_usage_bytes();
    sst.saturating_add(nodes)
        .saturating_add(adjacency)
        .saturating_add(range_pages)
}

pub(crate) fn legacy_budget_bytes(name: &str, default_mib: usize) -> usize {
    let mib = std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_mib);
    mib.saturating_mul(1024 * 1024)
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| value != "0")
        .unwrap_or(true)
}

fn env_opt_in(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requested(values: [usize; CACHE_TIER_COUNT]) -> CacheCapacities {
        CacheCapacities::from_array(usize::MAX, values)
    }

    #[test]
    fn proportional_scaling_never_exceeds_global_max() {
        let mib = 1024 * 1024;
        let legacy = [
            256 * mib,
            256 * mib,
            512 * mib,
            64 * mib,
            256 * mib,
            256 * mib,
            64 * mib,
            1024 * mib,
            32 * mib,
            0,
            256 * mib,
            512 * mib,
        ];
        let plan = requested(legacy).scaled_to(DEFAULT_CACHE_MAX_BYTES);

        assert_eq!(plan.total_capacity_bytes(), DEFAULT_CACHE_MAX_BYTES);
        assert!(plan
            .as_array()
            .iter()
            .zip(legacy)
            .all(|(effective, ceiling)| *effective <= ceiling));
        assert_eq!(
            plan,
            requested(legacy).scaled_to(DEFAULT_CACHE_MAX_BYTES),
            "rounding allocation must be deterministic"
        );
    }

    #[test]
    fn ceilings_are_preserved_when_they_fit() {
        let legacy = [100, 90, 80, 70, 60, 50, 40, 30, 25, 0, 20, 10];
        let plan = requested(legacy).scaled_to(1024);
        assert_eq!(plan.as_array(), legacy);
        assert_eq!(plan.total_capacity_bytes(), legacy.iter().sum::<usize>());
        assert_eq!(
            plan.sst_capacity_bytes(),
            legacy[..9].iter().sum::<usize>(),
            "the path registry is part of the SST/global capacity plan"
        );
    }

    #[test]
    fn zero_max_disables_every_tier() {
        let plan = requested([usize::MAX; CACHE_TIER_COUNT]).scaled_to(0);
        assert_eq!(plan.total_capacity_bytes(), 0);
        assert_eq!(plan.sst_capacity_bytes(), 0);
        assert_eq!(plan.node_view_capacity_bytes(), 0);
        assert_eq!(plan.adjacency_capacity_bytes(), 0);
    }

    #[test]
    fn sub_tier_byte_max_is_distributed_without_overflow() {
        let plan = requested([1; CACHE_TIER_COUNT]).scaled_to(4);
        assert_eq!(plan.as_array(), [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(plan.total_capacity_bytes(), 4);
    }

    #[test]
    fn explicit_search_reservation_is_carved_out_before_other_tiers() {
        let requested = requested([100; CACHE_TIER_COUNT]);
        let plan = requested.scaled_with_search_reservation(1_000, Some(700));

        assert_eq!(plan.search_index_capacity_bytes(), 700);
        assert_eq!(plan.total_capacity_bytes(), 1_000);
        assert_eq!(
            plan.as_array()
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != 7)
                .map(|(_, bytes)| *bytes)
                .sum::<usize>(),
            300
        );
    }

    #[test]
    fn explicit_search_reservation_is_clamped_to_global_max() {
        let plan =
            requested([100; CACHE_TIER_COUNT]).scaled_with_search_reservation(512, Some(4_096));
        assert_eq!(plan.search_index_capacity_bytes(), 512);
        assert_eq!(plan.total_capacity_bytes(), 512);
    }

    #[test]
    fn exact_byte_parser_is_strict_but_allows_empty_optional_reservation() {
        assert_eq!(
            parse_optional_exact_bytes(
                "NAMIDB_CACHE_MAX_BYTES",
                Some(" 1073741824 ".into()),
                false,
            ),
            Ok(Some(1_073_741_824))
        );
        assert_eq!(
            parse_optional_exact_bytes(
                "NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES",
                Some("".into()),
                true,
            ),
            Ok(None)
        );
        assert!(
            parse_optional_exact_bytes("NAMIDB_CACHE_MAX_BYTES", Some("2GiB".into()), false,)
                .unwrap_err()
                .contains("exact non-negative byte count")
        );
        assert!(
            parse_optional_exact_bytes("NAMIDB_CACHE_MAX_BYTES", Some("".into()), false,).is_err()
        );
    }
}
