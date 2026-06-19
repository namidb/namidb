//! Per-namespace aggregate statistics catalog (RFC-010 §1).
//!
//! Built once per `Snapshot` from a committed [`Manifest`], the catalog
//! is the read-only input that the selectivity and cardinality routines
//! consume. It deliberately holds only the *aggregates*: per-label
//! `node_count`, per-property `min/max/null/ndv`, per-edge-type
//! `edge_count`, and per-edge-type `avg/max` degree (both directions).
//! Raw `SstDescriptor` data is not retained.
//!
//! Memtable is intentionally not consulted — single-writer flush
//! cadence keeps the under-estimate bounded (RFC-010 §"Drawbacks 5").
//!
//! [`Manifest`]: namidb_storage::Manifest

use std::collections::BTreeMap;

use namidb_core::{LabelDictionary, LabelId, Schema};
use namidb_storage::manifest::KindSpecificStats;
use namidb_storage::sst::hll::Hll;
use namidb_storage::sst::stats::{HllSketchBytes, StatScalar};
use namidb_storage::{Manifest, SstDescriptor, SstKind};

/// Per-namespace stats consumed by the optimizer.
///
/// Construct via [`StatsCatalog::from_manifest`] (preferred) or
/// [`StatsCatalog::empty`] (for CLI / tests without committed data).
#[derive(Debug, Clone, Default)]
pub struct StatsCatalog {
    labels: BTreeMap<String, LabelStats>,
    edge_types: BTreeMap<String, EdgeTypeStats>,
    total_nodes: u64,
    total_edges: u64,
}

/// Per-label aggregate stats.
#[derive(Debug, Clone, Default)]
pub struct LabelStats {
    pub name: String,
    /// `Σ (row_count - tombstone_count)` over `Nodes` SSTs of this label.
    pub node_count: u64,
    /// Property name (logical, without `prop_` prefix) → aggregated
    /// [`PropStats`].
    pub properties: BTreeMap<String, PropStats>,
}

/// Per-property aggregate stats.
///
/// `null_count` is the sum across SSTs; `non_null_count` is derived as
/// `node_count - null_count` after the merge. `min`/`max` are computed
/// across all observed SSTs; `ndv` is `None` until the writer starts
/// emitting HLL sketches.
#[derive(Debug, Clone, Default)]
pub struct PropStats {
    pub null_count: u64,
    pub non_null_count: u64,
    pub min: Option<StatScalar>,
    pub max: Option<StatScalar>,
    pub ndv: Option<u64>,
    /// Mirrors `PropertyDef::unique` from the schema — the optimizer
    /// reads this to rewrite `Filter(prop = literal)` on top of
    /// `NodeScan(label)` into `NodeByPropertyValue` (point lookup).
    pub unique: bool,
    /// Mirrors `PropertyDef::indexed` — the optimizer reads this to
    /// rewrite an equality filter on a non-unique indexed property into a
    /// `NodeByPropertyValue { multi: true }` index lookup.
    pub indexed: bool,
}

/// Per-edge-type aggregate stats.
#[derive(Debug, Clone, Default)]
pub struct EdgeTypeStats {
    pub name: String,
    /// `Σ (row_count - tombstone_count)` over `EdgesFwd` SSTs of this type.
    pub edge_count: u64,
    pub avg_out_degree: f64,
    pub max_out_degree: u64,
    pub avg_in_degree: f64,
    pub max_in_degree: u64,
    pub src_label: Option<String>,
    pub dst_label: Option<String>,
}

impl StatsCatalog {
    /// Empty catalog — used when there is no Manifest yet (CLI ephemeral
    /// runs, unit tests).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Construct a catalog directly from `(label_name, node_count)`
    /// pairs. Intended for unit tests that need predictable
    /// cardinality estimates without spinning up a writer / flush.
    /// The resulting catalog has no property or edge stats — those
    /// are out of scope for the consumers that use this constructor.
    #[doc(hidden)]
    pub fn with_label_counts(entries: &[(&str, u64)]) -> Self {
        let mut labels: BTreeMap<String, LabelStats> = BTreeMap::new();
        let mut total_nodes: u64 = 0;
        for (name, count) in entries {
            total_nodes += *count;
            labels.insert(
                (*name).to_string(),
                LabelStats {
                    name: (*name).to_string(),
                    node_count: *count,
                    properties: Default::default(),
                },
            );
        }
        Self {
            labels,
            edge_types: BTreeMap::new(),
            total_nodes,
            total_edges: 0,
        }
    }

    /// Build a catalog from a committed [`Manifest`].
    ///
    /// Total cost is `O(|ssts|)`; SF1 LDBC sits at ~10³ descriptors
    /// (sub-millisecond). The result is owned data — safe to cache
    /// alongside a `LoadedManifest`.
    pub fn from_manifest(m: &Manifest) -> Self {
        let mut labels: BTreeMap<String, LabelStats> = BTreeMap::new();
        let mut edge_types: BTreeMap<String, EdgeTypeStats> = BTreeMap::new();
        let mut total_nodes: u64 = 0;
        let mut total_edges: u64 = 0;

        // Seed entries for every label/edge_type declared in the schema
        // even if no SST exists yet — keeps `LabelStats::name` always
        // populated and lets the optimizer distinguish "no SST" from
        // "label unknown".
        for (name, ldef) in &m.schema.labels {
            let entry = labels.entry(name.clone()).or_insert_with(|| LabelStats {
                name: name.clone(),
                ..Default::default()
            });
            // Seed the `unique` bit from the schema for every declared
            // property. The merge_node_sst pass populates the numeric
            // stats below; uniqueness is metadata, not a stat.
            for pd in &ldef.properties {
                entry
                    .properties
                    .entry(pd.name.clone())
                    .or_insert_with(|| PropStats {
                        unique: pd.unique,
                        indexed: pd.indexed,
                        ..Default::default()
                    });
            }
        }
        for (name, et) in &m.schema.edge_types {
            edge_types
                .entry(name.clone())
                .or_insert_with(|| EdgeTypeStats {
                    name: name.clone(),
                    src_label: Some(et.src_label.clone()),
                    dst_label: Some(et.dst_label.clone()),
                    ..Default::default()
                });
        }

        // Per-edge-type intermediates needed to compute averages after
        // we've seen every SST.
        let mut fwd_sum_degree: BTreeMap<String, u64> = BTreeMap::new();
        let mut fwd_key_count: BTreeMap<String, u64> = BTreeMap::new();
        let mut inv_sum_degree: BTreeMap<String, u64> = BTreeMap::new();
        let mut inv_key_count: BTreeMap<String, u64> = BTreeMap::new();
        // (label, prop) → merged HLL state. `None` value means at least
        // one SST carried no sketch for that column — we propagate that
        // gap and emit `ndv = None` (eq selectivity falls back to 0.1).
        let mut hll_merge: BTreeMap<(String, String), HllMerge> = BTreeMap::new();

        for sst in &m.ssts {
            match sst.kind {
                SstKind::Nodes => merge_node_sst(
                    sst,
                    &m.schema,
                    &m.label_dict,
                    &mut labels,
                    &mut total_nodes,
                    &mut hll_merge,
                ),
                SstKind::EdgesFwd => merge_edge_fwd_sst(
                    sst,
                    &m.schema,
                    &mut edge_types,
                    &mut fwd_sum_degree,
                    &mut fwd_key_count,
                    &mut total_edges,
                ),
                SstKind::EdgesInv => merge_edge_inv_sst(
                    sst,
                    &mut edge_types,
                    &mut inv_sum_degree,
                    &mut inv_key_count,
                ),
                // A VectorGraph SST (RFC-030) holds a Vamana search graph, not
                // node/edge rows — it contributes nothing to the label/edge/
                // property stats the cost model uses for selectivity.
                SstKind::VectorGraph => {}
            }
        }

        // Finalise averages.
        for (name, sums) in fwd_sum_degree {
            let keys = fwd_key_count.get(&name).copied().unwrap_or(0);
            if keys > 0 {
                let entry = edge_types
                    .entry(name.clone())
                    .or_insert_with(|| EdgeTypeStats {
                        name: name.clone(),
                        ..Default::default()
                    });
                entry.avg_out_degree = sums as f64 / keys as f64;
            }
        }
        for (name, sums) in inv_sum_degree {
            let keys = inv_key_count.get(&name).copied().unwrap_or(0);
            if keys > 0 {
                let entry = edge_types
                    .entry(name.clone())
                    .or_insert_with(|| EdgeTypeStats {
                        name: name.clone(),
                        ..Default::default()
                    });
                entry.avg_in_degree = sums as f64 / keys as f64;
            }
        }

        // Backfill non_null_count for property stats: needs the
        // finalised node_count which only becomes accurate after the
        // last SST has been folded in.
        for label_stats in labels.values_mut() {
            let denom = label_stats.node_count;
            for prop in label_stats.properties.values_mut() {
                prop.non_null_count = denom.saturating_sub(prop.null_count);
            }
        }

        // Materialise merged HLL → PropStats.ndv. Only emit `Some` when
        // every SST that touched this column shipped a sketch (any
        // missing sketch → ndv stays None, optimizer falls back to the
        // eq fallback).
        for ((label, prop), merge_state) in hll_merge {
            if let HllMerge::Complete(hll) = merge_state {
                if let Some(label_entry) = labels.get_mut(&label) {
                    if let Some(prop_entry) = label_entry.properties.get_mut(&prop) {
                        prop_entry.ndv = Some(hll.estimate());
                    }
                }
            }
        }

        Self {
            labels,
            edge_types,
            total_nodes,
            total_edges,
        }
    }

    pub fn label(&self, name: &str) -> Option<&LabelStats> {
        self.labels.get(name)
    }

    pub fn edge_type(&self, name: &str) -> Option<&EdgeTypeStats> {
        self.edge_types.get(name)
    }

    pub fn total_nodes(&self) -> u64 {
        self.total_nodes
    }

    pub fn total_edges(&self) -> u64 {
        self.total_edges
    }

    /// Iterator over all known label names (declared or observed).
    pub fn label_names(&self) -> impl Iterator<Item = &str> {
        self.labels.keys().map(String::as_str)
    }

    /// Iterator over all known edge-type names (declared or observed).
    pub fn edge_type_names(&self) -> impl Iterator<Item = &str> {
        self.edge_types.keys().map(String::as_str)
    }

    /// Test-only constructor: insert/replace a label entry without
    /// going through `from_manifest`. Used by sibling modules
    /// (`cardinality`, `selectivity`) to build fixtures inline.
    #[cfg(test)]
    pub(crate) fn __test_insert_label(&mut self, label: LabelStats) {
        let count = label.node_count;
        self.labels.insert(label.name.clone(), label);
        self.total_nodes = self.total_nodes.saturating_add(count);
    }

    /// Test-only constructor: insert/replace an edge-type entry.
    #[cfg(test)]
    pub(crate) fn __test_insert_edge_type(&mut self, et: EdgeTypeStats) {
        let count = et.edge_count;
        self.edge_types.insert(et.name.clone(), et);
        self.total_edges = self.total_edges.saturating_add(count);
    }
}

fn merge_node_sst(
    sst: &SstDescriptor,
    schema: &Schema,
    label_dict: &LabelDictionary,
    labels: &mut BTreeMap<String, LabelStats>,
    total_nodes: &mut u64,
    hll_merge: &mut BTreeMap<(String, String), HllMerge>,
) {
    let tombstones = match sst.kind_specific {
        KindSpecificStats::Nodes { tombstone_count } => tombstone_count,
        _ => 0,
    };
    let live = sst.row_count.saturating_sub(tombstones);
    // `total_nodes` is the distinct node-row count (one row per node) — a
    // multi-label node counts once here even though it appears under each of
    // its labels below. This keeps `node_count(L) <= total_nodes` for every L.
    *total_nodes = total_nodes.saturating_add(live);

    // Per-label `node_count`. id-primary node SSTs are no longer partitioned by
    // label (`scope == ""`); they carry per-label live counts in the
    // label-index sidecar descriptor, keyed by `LabelId`. Sum those across SSTs,
    // resolving each id through the namespace dictionary. Legacy single-label
    // SSTs predate the sidecar (`label_index == None`) and name their one label
    // in `scope`, so attribute `live` there.
    match sst.label_index.as_ref() {
        Some(li) if !li.per_label_counts.is_empty() => {
            for &(lid, count) in &li.per_label_counts {
                let Some(name) = label_dict.name(LabelId(lid)) else {
                    // Id absent from the dictionary (corrupt/forward-version
                    // manifest): we can't name the label, so skip rather than
                    // bucket it under a wrong name.
                    continue;
                };
                let entry = labels
                    .entry(name.to_string())
                    .or_insert_with(|| LabelStats {
                        name: name.to_string(),
                        ..Default::default()
                    });
                entry.node_count = entry.node_count.saturating_add(count);
            }
        }
        _ => {
            // Legacy single-label SSTs (and the brief pre-release window where
            // id-primary SSTs shipped a label index without `per_label_counts`)
            // land here. For a named scope this is exact; for an empty scope the
            // count is parked under "" until the next flush/compaction rewrites
            // the descriptor with `per_label_counts`. Only matters for in-flight
            // dev data on this branch — no released manifest has an empty scope.
            let label = &sst.scope;
            let entry = labels.entry(label.clone()).or_insert_with(|| LabelStats {
                name: label.clone(),
                ..Default::default()
            });
            entry.node_count = entry.node_count.saturating_add(live);
        }
    }

    // Per-(label, property) stats for id-primary node SSTs (RFC 025). The SST
    // spans many labels with properties in one `__overflow_json` column, so the
    // typed-column `property_stats` below is empty; these come from a per-label
    // sidecar computed at flush/compaction. Resolve each label id via the
    // dictionary and fold min/max/null + the HLL sketch into that label's
    // PropStats. `non_null_count` backfills from `node_count - null_count`.
    for s in &sst.per_label_property_stats {
        let Some(name) = label_dict.name(LabelId(s.label_id)) else {
            continue;
        };
        let entry = labels
            .entry(name.to_string())
            .or_insert_with(|| LabelStats {
                name: name.to_string(),
                ..Default::default()
            });
        let prop = entry.properties.entry(s.property.clone()).or_default();
        prop.null_count = prop.null_count.saturating_add(s.null_count);
        prop.min = merge_min(prop.min.take(), s.min.clone());
        prop.max = merge_max(prop.max.take(), s.max.clone());
        absorb_hll_sketch(
            hll_merge,
            name.to_string(),
            s.property.clone(),
            s.ndv_estimate.as_ref(),
        );
    }

    // Legacy typed-column SSTs keep their per-column `property_stats` keyed by
    // `scope` (a single label). Under id-primary `property_stats` is empty and
    // this loop is a no-op (the RFC-025 sidecar above carries the stats).
    if !sst.property_stats.is_empty() {
        let label = &sst.scope;
        let entry = labels.entry(label.clone()).or_insert_with(|| LabelStats {
            name: label.clone(),
            ..Default::default()
        });
        for col in &sst.property_stats {
            let logical_name = strip_prop_prefix(&col.name).to_string();
            // Some columns are emitted under the schema-declared name; the
            // declared LabelDef is authoritative for whether the property
            // is known. We still ingest unknown columns to keep the
            // optimizer robust against schemaless ingest.
            let _declared = schema.label(label).map(|l| {
                l.properties
                    .iter()
                    .any(|p| p.name == logical_name || col.name == format!("prop_{}", p.name))
            });

            let prop = entry.properties.entry(logical_name.clone()).or_default();
            prop.null_count = prop.null_count.saturating_add(col.null_count);
            prop.min = merge_min(prop.min.take(), col.min.clone());
            prop.max = merge_max(prop.max.take(), col.max.clone());

            absorb_hll_sketch(
                hll_merge,
                label.clone(),
                logical_name,
                col.ndv_estimate.as_ref(),
            );
        }
    }
}

/// Track the cross-SST merge state for a single (label, prop) pair.
enum HllMerge {
    /// Every SST seen so far for this column carried a sketch — we
    /// have a running merged Hll.
    Complete(Hll),
    /// At least one SST omitted the sketch — final ndv will be `None`.
    Incomplete,
}

fn absorb_hll_sketch(
    state: &mut BTreeMap<(String, String), HllMerge>,
    label: String,
    prop: String,
    sketch_bytes: Option<&HllSketchBytes>,
) {
    let key = (label, prop);
    match (state.get_mut(&key), sketch_bytes) {
        // First time we see this column, with a sketch: try to seed it.
        (None, Some(bytes)) => {
            match Hll::from_bytes(bytes.as_bytes()) {
                Ok(hll) => {
                    state.insert(key, HllMerge::Complete(hll));
                }
                Err(_) => {
                    // Corrupt or unsupported sketch: degrade safely.
                    state.insert(key, HllMerge::Incomplete);
                }
            }
        }
        // First time we see this column, no sketch present.
        (None, None) => {
            state.insert(key, HllMerge::Incomplete);
        }
        // Already tracking; promote to Incomplete if we cross any
        // sketch-less SST or hit a merge error.
        (Some(HllMerge::Complete(running)), Some(bytes)) => {
            match Hll::from_bytes(bytes.as_bytes()) {
                Ok(other) => {
                    if running.merge(&other).is_err() {
                        state.insert(key, HllMerge::Incomplete);
                    }
                }
                Err(_) => {
                    state.insert(key, HllMerge::Incomplete);
                }
            }
        }
        (Some(HllMerge::Complete(_)), None) => {
            state.insert(key, HllMerge::Incomplete);
        }
        (Some(HllMerge::Incomplete), _) => {
            // Already incomplete; nothing more to do.
        }
    }
}

fn merge_edge_fwd_sst(
    sst: &SstDescriptor,
    schema: &Schema,
    edge_types: &mut BTreeMap<String, EdgeTypeStats>,
    sum_degree: &mut BTreeMap<String, u64>,
    key_count: &mut BTreeMap<String, u64>,
    total_edges: &mut u64,
) {
    let et_name = &sst.scope;
    let (tombstones, hist_sum, hist_max, hist_keys) = match &sst.kind_specific {
        KindSpecificStats::Edges {
            tombstone_count,
            degree_histogram,
            key_count: kc,
            ..
        } => (
            *tombstone_count,
            degree_histogram.sum_degree,
            degree_histogram.max_degree,
            *kc,
        ),
        _ => (0, 0, 0, 0),
    };
    let live = sst.row_count.saturating_sub(tombstones);
    let entry = edge_types
        .entry(et_name.clone())
        .or_insert_with(|| EdgeTypeStats {
            name: et_name.clone(),
            ..Default::default()
        });
    entry.edge_count = entry.edge_count.saturating_add(live);
    if let Some(et_def) = schema.edge_type(et_name) {
        entry.src_label = Some(et_def.src_label.clone());
        entry.dst_label = Some(et_def.dst_label.clone());
    }
    if hist_max > entry.max_out_degree {
        entry.max_out_degree = hist_max;
    }
    *sum_degree.entry(et_name.clone()).or_insert(0) =
        sum_degree.get(et_name).copied().unwrap_or(0) + hist_sum;
    *key_count.entry(et_name.clone()).or_insert(0) =
        key_count.get(et_name).copied().unwrap_or(0) + hist_keys;
    *total_edges = total_edges.saturating_add(live);
}

fn merge_edge_inv_sst(
    sst: &SstDescriptor,
    edge_types: &mut BTreeMap<String, EdgeTypeStats>,
    sum_degree: &mut BTreeMap<String, u64>,
    key_count: &mut BTreeMap<String, u64>,
) {
    let et_name = &sst.scope;
    let (hist_sum, hist_max, hist_keys) = match &sst.kind_specific {
        KindSpecificStats::Edges {
            degree_histogram,
            key_count: kc,
            ..
        } => (
            degree_histogram.sum_degree,
            degree_histogram.max_degree,
            *kc,
        ),
        _ => (0, 0, 0),
    };
    let entry = edge_types
        .entry(et_name.clone())
        .or_insert_with(|| EdgeTypeStats {
            name: et_name.clone(),
            ..Default::default()
        });
    if hist_max > entry.max_in_degree {
        entry.max_in_degree = hist_max;
    }
    *sum_degree.entry(et_name.clone()).or_insert(0) =
        sum_degree.get(et_name).copied().unwrap_or(0) + hist_sum;
    *key_count.entry(et_name.clone()).or_insert(0) =
        key_count.get(et_name).copied().unwrap_or(0) + hist_keys;
}

/// Strip the `prop_` prefix that the SST writer adds to declared
/// property columns. Columns without the prefix are returned as-is.
fn strip_prop_prefix(name: &str) -> &str {
    name.strip_prefix("prop_").unwrap_or(name)
}

fn merge_min(a: Option<StatScalar>, b: Option<StatScalar>) -> Option<StatScalar> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(min_scalar(x, y)),
    }
}

fn merge_max(a: Option<StatScalar>, b: Option<StatScalar>) -> Option<StatScalar> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some(max_scalar(x, y)),
    }
}

fn min_scalar(a: StatScalar, b: StatScalar) -> StatScalar {
    if scalar_lt(&a, &b) {
        a
    } else {
        b
    }
}

fn max_scalar(a: StatScalar, b: StatScalar) -> StatScalar {
    if scalar_lt(&a, &b) {
        b
    } else {
        a
    }
}

/// Total order on [`StatScalar`] within the same type variant.
///
/// Cross-type comparisons return `false` (treated as equal) — those
/// indicate schema drift that the optimizer falls back through.
fn scalar_lt(a: &StatScalar, b: &StatScalar) -> bool {
    match (a, b) {
        (StatScalar::Bool(x), StatScalar::Bool(y)) => x < y,
        (StatScalar::Int32(x), StatScalar::Int32(y)) => x < y,
        (StatScalar::Int64(x), StatScalar::Int64(y)) => x < y,
        (StatScalar::Float32(x), StatScalar::Float32(y)) => x < y,
        (StatScalar::Float64(x), StatScalar::Float64(y)) => x < y,
        (StatScalar::Utf8(x), StatScalar::Utf8(y)) => x < y,
        (StatScalar::LargeUtf8(x), StatScalar::LargeUtf8(y)) => x < y,
        (StatScalar::Binary(x), StatScalar::Binary(y)) => x < y,
        (StatScalar::Date32(x), StatScalar::Date32(y)) => x < y,
        (StatScalar::TimestampMicrosUtc(x), StatScalar::TimestampMicrosUtc(y)) => x < y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use namidb_core::{DataType, EdgeTypeDef, LabelDef, PropertyDef, SchemaBuilder};
    use namidb_storage::manifest::SstLevel;
    use namidb_storage::sst::stats::{DegreeHistogram, PropertyColumnStats};
    use uuid::Uuid;

    fn label_def(name: &str, props: &[(&str, DataType)]) -> LabelDef {
        LabelDef {
            name: name.to_string(),
            properties: props
                .iter()
                .map(|(n, t)| PropertyDef::new(*n, t.clone(), true).unwrap())
                .collect(),
        }
    }

    fn node_sst(
        label: &str,
        rows: u64,
        tombs: u64,
        props: Vec<PropertyColumnStats>,
    ) -> SstDescriptor {
        SstDescriptor {
            id: Uuid::now_v7(),
            kind: SstKind::Nodes,
            scope: label.into(),
            level: SstLevel::L0,
            path: format!("sst/{}-nodes.parquet", label),
            size_bytes: 1024,
            row_count: rows,
            created_at: Utc::now(),
            min_key: [0; 16],
            max_key: [0xFF; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: props,
            kind_specific: KindSpecificStats::Nodes {
                tombstone_count: tombs,
            },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            per_label_property_stats: Vec::new(),
        }
    }

    fn edge_sst(
        kind: SstKind,
        et: &str,
        rows: u64,
        tombs: u64,
        hist: DegreeHistogram,
        keys: u64,
    ) -> SstDescriptor {
        SstDescriptor {
            id: Uuid::now_v7(),
            kind,
            scope: et.into(),
            level: SstLevel::L0,
            path: format!("sst/{}-edges.csr", et),
            size_bytes: 1024,
            row_count: rows,
            created_at: Utc::now(),
            min_key: [0; 16],
            max_key: [0xFF; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: vec![],
            kind_specific: KindSpecificStats::Edges {
                key_count: keys,
                tombstone_count: tombs,
                degree_histogram: Box::new(hist),
            },
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            per_label_property_stats: Vec::new(),
        }
    }

    #[test]
    fn empty_catalog_has_no_labels_no_edges() {
        let cat = StatsCatalog::empty();
        assert_eq!(cat.total_nodes(), 0);
        assert_eq!(cat.total_edges(), 0);
        assert!(cat.label("Person").is_none());
        assert!(cat.edge_type("KNOWS").is_none());
    }

    #[test]
    fn from_manifest_with_just_schema_seeds_known_names() {
        let schema = SchemaBuilder::new()
            .label(label_def("Person", &[("name", DataType::Utf8)]))
            .unwrap()
            .edge_type(EdgeTypeDef {
                name: "KNOWS".into(),
                src_label: "Person".into(),
                dst_label: "Person".into(),
                properties: vec![],
            })
            .unwrap()
            .build();
        let m = Manifest {
            version: 0,
            epoch: namidb_storage::Epoch::ZERO,
            writer_id: Uuid::now_v7(),
            created_at: Utc::now(),
            schema,
            ssts: vec![],
            wal_segments: vec![],
            label_dict: Default::default(),
            vector_indexes: Vec::new(),
        };
        let cat = StatsCatalog::from_manifest(&m);
        let p = cat.label("Person").expect("Person seeded");
        assert_eq!(p.node_count, 0);
        let k = cat.edge_type("KNOWS").expect("KNOWS seeded");
        assert_eq!(k.edge_count, 0);
        assert_eq!(k.src_label.as_deref(), Some("Person"));
        assert_eq!(k.dst_label.as_deref(), Some("Person"));
    }

    #[test]
    fn node_sst_aggregates_into_label_stats() {
        let schema = SchemaBuilder::new()
            .label(label_def("Person", &[("age", DataType::Int32)]))
            .unwrap()
            .build();
        let prop = PropertyColumnStats {
            name: "prop_age".into(),
            null_count: 2,
            min: Some(StatScalar::Int32(18)),
            max: Some(StatScalar::Int32(90)),
            ndv_estimate: None,
        };
        let sst1 = node_sst("Person", 100, 5, vec![prop.clone()]);
        let prop2 = PropertyColumnStats {
            name: "prop_age".into(),
            null_count: 3,
            min: Some(StatScalar::Int32(10)),
            max: Some(StatScalar::Int32(85)),
            ndv_estimate: None,
        };
        let sst2 = node_sst("Person", 200, 0, vec![prop2]);

        let m = Manifest {
            version: 0,
            epoch: namidb_storage::Epoch::ZERO,
            writer_id: Uuid::now_v7(),
            created_at: Utc::now(),
            schema,
            ssts: vec![sst1, sst2],
            wal_segments: vec![],
            label_dict: Default::default(),
            vector_indexes: Vec::new(),
        };
        let cat = StatsCatalog::from_manifest(&m);
        let p = cat.label("Person").unwrap();
        // 100 - 5 + 200 = 295 live nodes.
        assert_eq!(p.node_count, 295);
        let age = p.properties.get("age").expect("age folded");
        // null_count: 2 + 3 = 5.
        assert_eq!(age.null_count, 5);
        // non_null = node_count - null_count = 290.
        assert_eq!(age.non_null_count, 290);
        // min/max merged.
        assert_eq!(age.min, Some(StatScalar::Int32(10)));
        assert_eq!(age.max, Some(StatScalar::Int32(90)));
        assert_eq!(cat.total_nodes(), 295);
    }

    #[test]
    fn edge_fwd_and_inv_aggregate_average_degrees() {
        let schema = SchemaBuilder::new()
            .label(label_def("P", &[]))
            .unwrap()
            .edge_type(EdgeTypeDef {
                name: "KNOWS".into(),
                src_label: "P".into(),
                dst_label: "P".into(),
                properties: vec![],
            })
            .unwrap()
            .build();
        let mut hist_fwd = DegreeHistogram::empty();
        // 4 keys: degrees 1, 2, 3, 4. Sum = 10. avg = 2.5.
        for d in [1u64, 2, 3, 4] {
            hist_fwd.observe(d);
        }
        let mut hist_inv = DegreeHistogram::empty();
        // 5 keys: degrees 1, 1, 2, 2, 4. Sum = 10. avg = 2.0.
        for d in [1u64, 1, 2, 2, 4] {
            hist_inv.observe(d);
        }
        let max_fwd = hist_fwd.max_degree;
        let max_inv = hist_inv.max_degree;
        let fwd = edge_sst(SstKind::EdgesFwd, "KNOWS", 10, 0, hist_fwd, 4);
        let inv = edge_sst(SstKind::EdgesInv, "KNOWS", 10, 0, hist_inv, 5);

        let m = Manifest {
            version: 0,
            epoch: namidb_storage::Epoch::ZERO,
            writer_id: Uuid::now_v7(),
            created_at: Utc::now(),
            schema,
            ssts: vec![fwd, inv],
            wal_segments: vec![],
            label_dict: Default::default(),
            vector_indexes: Vec::new(),
        };
        let cat = StatsCatalog::from_manifest(&m);
        let k = cat.edge_type("KNOWS").unwrap();
        assert_eq!(k.edge_count, 10);
        assert!((k.avg_out_degree - 2.5).abs() < f64::EPSILON);
        assert!((k.avg_in_degree - 2.0).abs() < f64::EPSILON);
        assert_eq!(k.max_out_degree, max_fwd);
        assert_eq!(k.max_in_degree, max_inv);
    }

    #[test]
    fn node_sst_without_schema_still_ingested() {
        // Schema-less ingest is the default in namidb; the catalog
        // must still capture the row counts of unknown labels.
        let prop = PropertyColumnStats {
            name: "prop_score".into(),
            null_count: 0,
            min: Some(StatScalar::Float64(1.0)),
            max: Some(StatScalar::Float64(99.5)),
            ndv_estimate: None,
        };
        let sst = node_sst("Unknown", 50, 5, vec![prop]);
        let m = Manifest {
            version: 0,
            epoch: namidb_storage::Epoch::ZERO,
            writer_id: Uuid::now_v7(),
            created_at: Utc::now(),
            schema: Schema::empty(),
            ssts: vec![sst],
            wal_segments: vec![],
            label_dict: Default::default(),
            vector_indexes: Vec::new(),
        };
        let cat = StatsCatalog::from_manifest(&m);
        let u = cat.label("Unknown").unwrap();
        assert_eq!(u.node_count, 45); // 50 - 5
        let score = u.properties.get("score").unwrap();
        assert_eq!(score.min, Some(StatScalar::Float64(1.0)));
        assert_eq!(score.max, Some(StatScalar::Float64(99.5)));
    }

    #[test]
    fn label_and_edge_iterators_include_seeded_names() {
        let schema = SchemaBuilder::new()
            .label(label_def("Person", &[]))
            .unwrap()
            .label(label_def("Message", &[]))
            .unwrap()
            .build();
        let m = Manifest {
            version: 0,
            epoch: namidb_storage::Epoch::ZERO,
            writer_id: Uuid::now_v7(),
            created_at: Utc::now(),
            schema,
            ssts: vec![],
            wal_segments: vec![],
            label_dict: Default::default(),
            vector_indexes: Vec::new(),
        };
        let cat = StatsCatalog::from_manifest(&m);
        let names: Vec<_> = cat.label_names().collect();
        assert!(names.contains(&"Person"));
        assert!(names.contains(&"Message"));
        assert_eq!(cat.edge_type_names().count(), 0);
    }

    #[test]
    fn merge_min_max_picks_smallest_and_largest() {
        let a = Some(StatScalar::Int64(5));
        let b = Some(StatScalar::Int64(2));
        assert_eq!(merge_min(a.clone(), b.clone()), Some(StatScalar::Int64(2)));
        assert_eq!(merge_max(a, b), Some(StatScalar::Int64(5)));
        assert_eq!(
            merge_min(None, Some(StatScalar::Int64(42))),
            Some(StatScalar::Int64(42))
        );
        assert_eq!(
            merge_max(Some(StatScalar::Int64(7)), None),
            Some(StatScalar::Int64(7))
        );
    }
}
