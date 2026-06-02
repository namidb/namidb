# RFC 025: Per-label property statistics under id-primary

**Status:** accepted
**Author(s):** Matías Fonseca <matias.fonseca@fonlescompany.com>
**Created:** 2026-06-02
**Updated:** 2026-06-02
**Implements:** Phase 1 (per-label stats sidecar) in progress; Phase 2 (typed union columns) deferred to a follow-up RFC
**Supersedes:** (none)

## Summary

The id-primary node layout (PRs #65, #66) stores every node property in a
single `__overflow_json` column and spans all labels in one SST. As a side
effect, per-(label, property) statistics (min/max/null/ndv) no longer exist, so
the cost model falls back to constants for every property predicate. This RFC
restores per-label property statistics via a manifest-side sidecar computed at
flush and compaction, without changing the physical column layout. Parquet
column pushdown is explicitly out of scope here and left to a future
typed-column layout (Phase 2).

## Motivation

Before id-primary, node SSTs were partitioned by label and carried typed
`prop_<name>` columns, so `compute_property_stats` (sst/nodes.rs) read the
Parquet footer per column and the manifest carried `PropertyColumnStats` keyed
by `(scope = label, prop)`. The optimizer keyed `StatsCatalog.labels[L]
.properties[prop] -> PropStats` and used real min/max/ndv.

Under id-primary the writer builds the SST with an empty `LabelDef`, so
`compute_property_stats` iterates an empty property list and emits nothing.
`StatsCatalog::from_manifest` therefore leaves every `PropStats` at its default
(`min/max/ndv = None`, `null_count = 0`). The consequences in
`cost/selectivity.rs`:

- `a.prop = literal` falls back to `FALLBACK_EQ = 0.10` (no ndv).
- `a.prop < / > literal` falls back to `FALLBACK_RANGE = 0.33` (no min/max).
- `IS NULL` falls back to `FALLBACK_IS_NULL = 0.05` (no null_count).

These feed cardinality, which drives join-side selection (hash build vs probe),
join reordering, and the AGM bound. Wrong-by-default selectivity means the
optimizer picks plans blind to data skew. Results stay correct, but plan
quality regresses versus the pre-id-primary engine. Cost of doing nothing: the
cost model is permanently estimate-blind for properties, and the relaxed
`cost_smoke` assertions (documented as deferred in #65/#66) never tighten back.

This is a real regression relative to the released label-partitioned engine,
not a new feature. Restoring it is the point of this RFC.

## Design

### Phase 1 (this RFC): per-(label, property) statistics sidecar

Keep `__overflow_json` as the property store. Compute per-(label, property)
statistics at flush and compaction by walking the rows grouped by their label
set, and carry them in the manifest descriptor (not a separate body object) so
the cost model reads them with the manifest fetch it already does.

**Where the stats come from.** The flush path already walks every row once to
build the label-index sidecar (`prepare_label_index_sidecar`, flush.rs): it
decodes each `NodeWriteRecord`, including `labels: Vec<LabelId>` and the
property map. In the same pass, for each row and each of its labels `L`, fold
each property value into an accumulator keyed by `(LabelId, prop_name)`:

- `min` / `max`: ordered merge over `StatScalar` (reuse `merge_min`/`merge_max`
  from cost/stats.rs or the writer-side equivalents).
- `null_count`: increment when the property is absent / JSON null on a row that
  carries `L`.
- `ndv`: one `Hll` sketch per `(LabelId, prop_name)`, hashing the non-null
  value, identical to the pre-id-primary `property_hlls` path but keyed by
  label too.

A node with `K` labels contributes to `K` accumulators. This matches the
multi-label semantics already used for per-label `node_count` in #65: a property
on a `:Person:Admin` node counts under both `Person` and `Admin`.

**Where the stats live.** Add an optional descriptor on `SstDescriptor`:

```rust
pub struct PerLabelPropertyStats {
    pub label_id: u32,
    pub property: String,            // logical name (no prop_ prefix)
    pub null_count: u64,
    pub non_null_count: u64,
    pub min: Option<StatScalar>,
    pub max: Option<StatScalar>,
    pub ndv_estimate: Option<HllSketchBytes>, // merged at read time
}

// on SstDescriptor:
#[serde(default)]
pub per_label_property_stats: Vec<PerLabelPropertyStats>,
```

`LabelId` is resolved to a name through the manifest's `label_dict`, exactly as
the per-label counts are. `#[serde(default)]` keeps older manifests loading
unchanged (they get an empty vec and fall back to today's behaviour). Estimated
size: an SST with `~10` labels times `~5` properties is `~50` entries, a few
hundred bytes of manifest JSON, well within budget.

**Where the cost model reads it.** In `cost/stats.rs merge_node_sst`, after the
per-label `node_count` is resolved, fold `per_label_property_stats` into
`labels[name].properties[prop]`: sum `null_count`, merge min/max, and merge the
HLL sketches (the existing `HllMerge` machinery already propagates "Incomplete"
when one SST omits a sketch). No selectivity/cardinality changes are needed;
they already consume `PropStats` and will simply see real values instead of
`None`.

**Compaction.** `compact.rs put_node_sst_l1` re-emits the sidecar from the
reconciled `merged_rows`, mirroring how it already re-emits the label index
(#65). Without this, per-label property stats reset to empty after the first
L0->L1 merge, the same failure mode the label-index rebuild fixed.

**No Parquet pushdown.** Properties still live in one JSON column, so
`eval_row_group` cannot prune by property value and projection pushdown cannot
elide columns. Predicate evaluation stays a per-row 3VL check after the scan,
as today. Phase 1 buys cost-model accuracy, not IO.

### Phase 2 (future, separate RFC): typed union columns

If Parquet predicate/column pushdown (RFC-013/015) becomes a priority, emit
typed `prop_<name>` columns equal to the union of all properties declared
across the namespace's labels (a node leaves the columns it lacks NULL;
`__overflow_json` keeps only undeclared keys). Per-column Parquet footer stats
return, and pushdown works again; per-label stats are derived by attributing
each column's stats to the labels that declare that property. This is decoupled
from Phase 1: once the cost-model foundation exists, the pushdown RFC can focus
purely on IO. It carries sparse-column storage/decode overhead and schema
coupling, which is why it is not Phase 1.

## Alternatives considered

Three approaches were evaluated (multi-agent design panel, 2026-06-02):

1. **Per-label stats sidecar (recommended, Phase 1).** Restores cost-model
   stats with no physical rewrite, low risk, reuses the label-index pattern.
   Does not restore Parquet pushdown.

2. **Union typed columns.** Restores both stats AND Parquet pushdown, but is
   roughly 3x the effort, adds sparse-column overhead, and couples the SST body
   to the namespace schema (union must be recomputed as labels/properties
   evolve). Better as Phase 2 when pushdown is the goal.

3. **Hybrid: typed columns only for `indexed`/`unique` (and a hot list).**
   Lower write amplification than the full union and restores pushdown for the
   promoted properties, but leaves ad-hoc filters on unpromoted properties on
   the fallback, so it is an incomplete stats solution and adds a schema
   annotation surface. Reasonable middle ground but not the cleanest first step.

The recommendation is to ship Phase 1 (sidecar) now for the cardinality
regression and revisit 2/3 as a separate pushdown RFC.

## Drawbacks

- Adds a per-SST manifest payload (bounded, but it grows with labels x
  properties). Pathological many-label schemas could bloat the manifest;
  mitigate with a cap or by omitting properties with no observed values.
- Flush/compaction pay an extra per-row property fold (JSON already decoded for
  the label-index pass, so the marginal cost is the accumulation, not parsing).
- Per-label attribution counts a multi-label node's property under each of its
  labels. This is the intended semantics but means the per-label stats are not
  mutually exclusive (consistent with per-label `node_count`).
- Stats are approximate across un-compacted SSTs and over schema evolution
  (a property added after an SST was written has no stats in that SST and falls
  back). Same bounded-error contract the cost model already documents.

## Open questions

1. **Compaction propagation.** Confirmed required (Phase 1 includes the L1
   re-emit); without it, stats vanish after recompaction. Documented as part of
   the design, not left open.
2. **Schema evolution.** A property added to a label after an SST was written
   has `None` stats in that SST and falls back to the constant. Acceptable for
   v0; a later pass could synthesize fallback entries from the schema.
3. **Manifest version bump.** The new field is `#[serde(default)]`, so older
   manifests load unchanged and no format-version bump is required (same
   decision as the `per_label_counts` field in #65). Confirm before merge.
4. **HLL key cardinality.** One sketch per `(label, property)` per SST. For
   wide schemas this is many small sketches; confirm the manifest size stays
   acceptable, or store sketches in a body sidecar (like bloom) if they grow.

## References

- RFC 002: SST format (the typed `prop_<name>` column layout this restores
  statistics for).
- RFC 010: cost-based optimizer (the `PropStats` consumers).
- RFC 013 / RFC 015: Parquet predicate pushdown / projection pushdown (the
  Phase 2 beneficiaries).
- PRs #65, #66: id-primary multi-label core and query layer, which deferred
  per-label property statistics to "a future typed-column layout" -- this RFC.
