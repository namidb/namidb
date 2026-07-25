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
    DEFAULT_PROPERTY_SIDECAR_CACHE_BUDGET_MIB, DEFAULT_SST_CACHE_BUDGET_MIB,
    DEFAULT_SST_METADATA_CACHE_BUDGET_MIB, DEFAULT_TEXT_INDEX_CACHE_BUDGET_MIB,
    DEFAULT_VECTOR_INDEX_CACHE_BUDGET_MIB,
};
use crate::node_cache::DEFAULT_NODE_CACHE_BUDGET_MIB;

/// Default aggregate capacity of all process-wide caches: 1 GiB.
pub const DEFAULT_CACHE_MAX_BYTES: usize = 1024 * 1024 * 1024;

const CACHE_TIER_COUNT: usize = 11;

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
    pub(crate) text_index_bytes: usize,
    pub(crate) vector_index_bytes: usize,
    pub(crate) node_view_bytes: usize,
    pub(crate) adjacency_bytes: usize,
}

impl CacheCapacities {
    fn requested_from_env(max_bytes: usize) -> Self {
        let sst_enabled = env_enabled("NAMIDB_SST_CACHE");
        let node_enabled = env_enabled("NAMIDB_NODE_CACHE");
        let adjacency_enabled = env_enabled("NAMIDB_ADJACENCY");

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
            text_index_bytes: if cfg!(feature = "text-index") {
                sst(
                    "NAMIDB_TEXT_INDEX_CACHE_BUDGET_MIB",
                    DEFAULT_TEXT_INDEX_CACHE_BUDGET_MIB,
                )
            } else {
                0
            },
            vector_index_bytes: if cfg!(feature = "vector-index") {
                sst(
                    "NAMIDB_VECTOR_INDEX_CACHE_BUDGET_MIB",
                    DEFAULT_VECTOR_INDEX_CACHE_BUDGET_MIB,
                )
            } else {
                0
            },
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

    fn as_array(self) -> [usize; CACHE_TIER_COUNT] {
        [
            self.sst_body_bytes,
            self.decoded_node_row_group_bytes,
            self.property_sidecar_bytes,
            self.sst_metadata_bytes,
            self.edge_stream_bytes,
            self.edge_reader_bytes,
            self.bloom_filter_bytes,
            self.text_index_bytes,
            self.vector_index_bytes,
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
            text_index_bytes: values[7],
            vector_index_bytes: values[8],
            node_view_bytes: values[9],
            adjacency_bytes: values[10],
        }
    }

    /// Sum of the nine active `SstCache` tier ceilings.
    pub fn sst_capacity_bytes(&self) -> usize {
        self.as_array()[..9]
            .iter()
            .fold(0usize, |sum, value| sum.saturating_add(*value))
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

/// Read the exact-byte aggregate cache limit.
///
/// An unset or malformed value uses [`DEFAULT_CACHE_MAX_BYTES`]. `0` disables
/// all process-wide shared caches.
pub fn cache_max_bytes() -> usize {
    std::env::var("NAMIDB_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CACHE_MAX_BYTES)
}

/// Process-wide effective capacity plan. Environment configuration is sampled
/// once because the shared cache instances cannot be resized after creation.
pub fn shared_cache_capacities() -> CacheCapacities {
    static CAPACITIES: OnceLock<CacheCapacities> = OnceLock::new();
    *CAPACITIES.get_or_init(|| {
        let max_bytes = cache_max_bytes();
        let capacities = CacheCapacities::requested_from_env(max_bytes).scaled_to(max_bytes);
        tracing::info!(
            cache_max_bytes = capacities.max_bytes,
            cache_assigned_bytes = capacities.total_capacity_bytes(),
            sst_cache_bytes = capacities.sst_capacity_bytes(),
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
    sst.saturating_add(nodes).saturating_add(adjacency)
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
            512 * mib,
            512 * mib,
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
        let legacy = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5];
        let plan = requested(legacy).scaled_to(1024);
        assert_eq!(plan.as_array(), legacy);
        assert_eq!(plan.total_capacity_bytes(), legacy.iter().sum::<usize>());
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
        assert_eq!(plan.as_array(), [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(plan.total_capacity_bytes(), 4);
    }
}
