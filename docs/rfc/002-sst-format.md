# RFC 002: SST Format — Property Columnar + CSR Adjacency

**Status:** draft
**Author(s):** NamiDB founding team
**Implements:** (links to PRs land here)
**Supersedes:** —
**Depends on:** [RFC-001](./001-storage-engine.md) (manifest CAS, namespace layout, write/read paths)

## Summary

This RFC defines the **on-disk format of the SSTs** that the NamiDB
LSM emits when it flushes a memtable. There are two physical kinds of
SST in v1:

1. **Node SSTs** — Apache **Parquet** files, one per `(label, level)`
 bucket produced by a flush. They hold the property columns of nodes
 plus a tombstone column, an `lsn` column, the `node_id` key column,
 and a mandatory `__overflow_json: Utf8` column that captures any
 property the schema did not declare at flush time. They use Parquet's
 standard page index + Zstd compression + dictionary encoding.
2. **Edge SSTs** — a NamiDB-native **CSR binary** format, one per
 `(edge_type, level)` bucket. Each flush emits two physical files per
 bucket: a **forward** SST (sorted by `src_id`) and an **inverse**
 partner SST (sorted by `dst_id`). Both share the same wire format,
 differentiated by a single header flag. They hold the adjacency for
 a fixed `(src_label, edge_type, dst_label)` triple, with bitpacked
 offsets, split-encoded neighbour lists, a fence-pointer index for
 large SSTs, and parallel Zstd-compressed property streams. They are
 designed for `O(deg(v))` neighbour expansion in **two ranged GETs
 warm / four to six cold** against object storage (counts derived in
 §3.4).

Following the resolution of the RFC-001 open question on bloom filters
and after a sizing review (revision 2):

- **Property stats, degree histograms, and key ranges** are embedded in
 the manifest's `SstDescriptor` — they fit in JSON and gate scans
 without extra GETs.
- **Bloom filters** live as **side-car files** (`<sst_id>.bloom`,
 raw binary, fetched on first probe and cached by foyer). They are
 too large (≈1.25 MiB for 1 M keys at 10 bits/key) to inline in a JSON
 manifest at production scale.

The read path therefore needs **two GETs** to gate by min/max + property
stats and **at most one extra GET per candidate SST** when a bloom
probe is needed (and that GET is foyer-cacheable across queries).

This RFC defines:

- Path conventions for SST objects.
- Schema-to-Parquet mapping for node SSTs.
- Byte-level wire format for edge SSTs.
- The extended `SstDescriptor` struct, including the embedded statistics
 and the bloom side-car pointer.
- Forward-compatibility rules that bind reader and writer.
- The read-path access patterns and the GETs they imply per neighbour
 expansion.

It does **not** define the flush orchestration, recovery / WAL replay,
or the read-side snapshot merge. Those are the subjects of follow-up
RFCs (RFC-003 flush + recovery, RFC-004 read snapshot).

## Motivation

After RFC-001 the storage engine can take writes (WAL + memtable) and
commit linearisable manifest versions. It still cannot **durably
materialise data into queryable columnar files**. Until SSTs exist:

- The memtable is the only home of accepted writes; a flush of any
 meaningful size cannot complete.
- The manifest's `ssts: []` is permanently empty, so the read path has
 no cold tier to fall back on.
- We cannot bench the **§14.1 thresholds** (cold <500 ms p50, warm
 <10 ms p50, ingest ≥10 k nodes/s) because there is nothing to read
.

We need a format that satisfies six non-negotiables:

1. **Immutable, write-once.** Required by manifest CAS — an SST that is
 referenced by manifest *v* must not change underneath a reader of *v*.
2. **Random-access ranged reads.** Object storage charges per GET and
 per byte; we want to fetch only the column slices we need.
3. **CSR-shaped edges, both directions.** Multi-hop traversals must do
 `O(deg(v))` work per expansion regardless of whether the
 expansion is by `src` or by `dst`. Single-direction CSR forces
 linear scans for the missing direction, which kills the
 cold-query budget.
4. **Stats good enough for the optimizer.** Without min/max key,
 bloom filters and degree histograms the read path turns every "look
 up a neighbour" into a fan-out across every SST in scope.
5. **Sane wire stability story.** v1 of the file will outlive v0.1.0
 of the binary. The format needs a version byte and explicit
 forward-compat rules.
6. **No data loss on schema drift.** Open-schema ingest at the SDK is
 the normal mode of operation for GraphRAG / agent-memory; properties
 the schema does not yet declare must round-trip through the SST
 intact, never be silently dropped.

Parquet covers (1), (2), (4), (5), and (6) out of the box for nodes.
For edges no off-the-shelf format gives us (3) without paying for
orthogonal machinery we do not need (e.g. Parquet repetition levels for
a CSR-shaped list-of-list adds metadata overhead and an indirection on
every neighbour lookup). Hence: Parquet for nodes, custom CSR for edges.

## Design

### 1. Naming conventions and paths

A single flush emits a set of SSTs and a parallel set of bloom side-cars.
Each SST is identified by a UUIDv7; its on-disk path is determined by
the namespace, level, kind, and scope:

```text
<bucket>/<namespace>/sst/level<L>/<uuidv7>-<kind>-<scope>.<ext>
<bucket>/<namespace>/sst/level<L>/<uuidv7>-<kind>-<scope>.bloom

where:
 <L> ∈ {0, 1, 2, …}
 <kind> ∈ {"nodes", "edges-fwd", "edges-inv"}
 <scope> is the label name (nodes) or edge type name (edges-fwd/-inv)
 <ext> is "parquet" for nodes, "csr" for edges
 <uuidv7> is the writer's `Uuid::now_v7()` rendered hex (no dashes)
```

Examples:

```text
acme/sst/level0/01959a3f7b...-nodes-Person.parquet
acme/sst/level0/01959a3f7b...-nodes-Person.bloom
acme/sst/level0/01959a3f7c...-edges-fwd-KNOWS.csr
acme/sst/level0/01959a3f7c...-edges-fwd-KNOWS.bloom
acme/sst/level0/01959a3f7d...-edges-inv-KNOWS.csr
acme/sst/level0/01959a3f7d...-edges-inv-KNOWS.bloom
```

The UUIDv7 is monotonically time-ordered, so a `list` on
`sst/level<L>/` yields candidate SSTs in creation order — which lets
the read-side merger apply "newest wins per key" without first sorting
by some metadata field. Forward and inverse partners produced by the
same flush carry **distinct UUIDv7s**: this preserves the rule that one
object_store path identifies exactly one immutable artefact, and lets
compaction retire one partner without the other when necessary.

`namidb-storage::paths::NamespacePaths` will grow two helpers:

```rust
impl NamespacePaths {
 pub fn sst_file(
 &self,
 level: u32,
 id: Uuid,
 kind: SstKind,
 scope: &str,
 ) -> Path { /* …-<kind>-<scope>.<ext> */ }

 pub fn sst_bloom_file(
 &self,
 level: u32,
 id: Uuid,
 kind: SstKind,
 scope: &str,
 ) -> Path { /* …-<kind>-<scope>.bloom */ }
}
```

Scope strings appear in object keys, so the writer **must** validate
them against the same DNS-safe ruleset that `NamespaceId` already
enforces. Schema declarations gate this upstream, but the writer
asserts it again to defend against bypasses.

### 2. Node SST — Parquet layout

#### 2.1 Logical schema

For each node label `L` with declared properties `p_1: T_1, …, p_k: T_k`
(see [`namidb-core::schema::LabelDef`](../../crates/namidb-core/src/schema.rs)),
a node SST has the following Arrow schema:

| Column | Arrow type | Nullable | Notes |
|-------------------|--------------------------|----------|----------------------------------------------------------------------------------------------------|
| `node_id` | `FixedSizeBinary(16)` | no | UUIDv7 of the node, big-endian. |
| `tombstone` | `Boolean` | no | `true` = node deleted as of `lsn`. |
| `lsn` | `UInt64` | no | LSN at which this row was applied. |
| `prop_<p_i>` | `<T_i>.to_arrow()` | per def | One column per `PropertyDef` declared in the schema at write time. |
| `__overflow_json` | `Utf8` | **yes** | Mandatory column. Stores undeclared properties as a JSON object string; `null` when there are none. |
| `__schema_version`| `UInt64` | no | Snapshot of the manifest's schema version at flush time. Lets the reader pin its decode rules. |

The `__overflow_json` column is **always present** in the Parquet schema
(even when every row is null), so a reader can rely on it being there
unconditionally without consulting the manifest first. This closes the
revision-1 open question about open-schema ingest: undeclared properties
flow through ingest → memtable → SST → reader without any silent drop,
and the SDK layer (Python / TS) reconstructs them on read into the
caller's native map type.

`__schema_version` is the manifest version the writer used to map
property names → columns. Two SSTs written under different schema
versions can co-exist; the reader chooses how to reconcile them
(typically: newer schema_version wins; older SST's columns are mapped
back to their names via the manifest of the version that produced
them — handled by RFC-004).

**Reserved column names.** Every declared property `p` is materialised
as the Parquet column **`prop_<p>`** (i.e. the `prop_` prefix is part
of the on-disk column name, not an editorial shorthand). Names that
would collide with the engine-managed columns — `node_id`, `tombstone`,
`lsn`, `__overflow_json`, `__schema_version` — are reserved. The
writer **rejects** any `PropertyDef` whose `name`:
- starts with the prefix `prop_` (would double-prefix on disk),
- starts with the prefix `__` (engine-private namespace), or
- equals one of `{node_id, tombstone, lsn}`.

Enforcement happens at schema-declaration time in
`namidb-core::schema::PropertyDef::new` *and* is asserted again at
flush time with `Error::SchemaConflict`. A reader that observes a
column whose name violates the namespace rules treats the SST as
corrupted.

#### 2.2 Sort order

Rows inside a node SST are sorted by **`(node_id ascending)`**. The
memtable already gives us this order — `MemKey::Node { label, id }`
sorts lexicographically by `id` inside a single label scope. The
Parquet writer asserts this invariant; out-of-order rows are a writer
bug and abort the flush.

This sort order matters because:

- The read path's merge needs an ordered stream per SST to do a k-way
 merge of `(SST_0, …, SST_n, frozen_memtable, live_memtable)` without
 buffering everything in memory.
- The Parquet page index gives O(log n) lookup for a target `node_id`
 using just min/max of each page, which is the basis of the warm
 point-lookup path.

Within an SST a `node_id` appears at most once: the memtable
already collapses repeated upserts of the same key.

#### 2.3 Encodings and compression

| Setting | Value | Rationale |
|--------------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------|
| Compression | `Zstd` level `6` | Sweet spot per Parquet benchmarks: ~25–35 % ratio on string-heavy graph data, ~250 MB/s decode. |
| Dictionary encoding | enabled (per column) | Mandatory for low-cardinality `Utf8` (e.g. country, status). `parquet-rs` falls back automatically when the dictionary stops paying. |
| Row group size | **128 K rows** (target) | Page-index granularity per §7.5 of the plan. **Assumed average row size 256 B → 32 MiB row groups warm.** Rotation algorithm: writer closes a row group as soon as **either** 128 K rows have been buffered **or** the in-memory uncompressed byte size of the buffered rows would exceed 256 MiB (whichever fires first). Short flushes that would produce a row group smaller than 16 MiB are merged with the previous group of the same SST when possible (policy in RFC-003). |
| Data page size | `1 MiB` | Largest page object_store will fetch with a single ranged GET. |
| Write batch size | `8192` rows | Standard Arrow batch sentinel. |
| Statistics | column min/max **on** | Per-page and per-row-group; used by reader for predicate pushdown. |
| Page index | **on** | Required by `parquet-rs 55` to enable per-page min/max-driven row-group pruning. |
| Bloom (Parquet built-in) | **disabled** | We bring our own side-car bloom over `node_id`. Parquet's bloom adds bytes for no win here. |
| Page checksums | enabled | Cheap, defends against torn S3 reads. |

These are defaults; the writer accepts a `NodeSstWriterOptions` to
override compression level (e.g. `Zstd-18` for cold archives), so a
future compaction worker can re-compress L≥2 SSTs without changing the
file format.

**NaN / `±Inf` handling.** `f32` / `f64` columns may contain
`NaN`, `+Inf`, and `-Inf` freely — the raw bytes always land in the
page data, the row is **never rejected** by the writer. The contract
is only at the stats layer: a column whose page contains any `NaN`
or `±Inf` produces `min = None` and `max = None` in
`PropertyColumnStats`. Predicate pushdown gracefully falls back to
per-row evaluation when min/max is unavailable. The page's `null_count`
remains accurate.

#### 2.4 Tombstones

Deletes are stored as rows with `tombstone = true` and **null** property
columns. The reader treats a tombstone as "this `node_id` does not
exist at `>= lsn`" — overriding any older SST that contains the same
`node_id`. Tombstones live until they are absorbed by a compaction that
proves no older SST or WAL segment still references the key (full-tree
compaction; will be defined in RFC-005 compaction policy).

We **do not** use Parquet's "delete files" (Iceberg-style positional
deletes). They require a separate index and a second manifest hop; we
get the same semantics by treating `tombstone` as a regular column.

**Nullable invariant.** Tombstone rows carry `null` in every declared
property column. To make that representable regardless of the
`PropertyDef.nullable` flag, the SST-level Arrow field for every
declared property is **always nullable** in the Parquet schema, even
when the schema declares `nullable = false`. Non-null contracts on
declared properties are an **ingest-time** invariant (enforced before
a row reaches the memtable), not an SST-time invariant. The writer
implementation rejects a non-tombstone row with a null value in a
`nullable = false` column at the ingest API boundary; once the row is
in the SST, only the `tombstone` flag determines whether the column is
"validly null".

#### 2.5 Footer + statistics extraction

When the writer closes a Parquet file it emits the standard Parquet
footer plus a sidecar `NodeSstStats` struct that goes into the
manifest's `SstDescriptor` (see §4 below):

```rust
pub struct NodeSstStats {
 pub row_count: u64,
 pub tombstone_count: u64,
 pub min_node_id: [u8; 16],
 pub max_node_id: [u8; 16],
 pub min_lsn: u64,
 pub max_lsn: u64,
 pub property_stats: Vec<PropertyColumnStats>,
 pub schema_version_min: u64,
 pub schema_version_max: u64,
}
```

`PropertyColumnStats` carries `null_count`, `min`, `max` for ordered
types, and `ndv_estimate` (HyperLogLog++ sketch, 1 KiB) for cardinality
hints. These are read from the Parquet footer column stats — no extra
scan required.

The bloom filter is **not** part of this struct (it goes to the side-car
file; see §4.2).

### 3. Edge SST — CSR binary format

#### 3.1 Why custom

A property graph SST for edges has structure that Parquet does not
natively express well:

- The natural unit of read is "give me all neighbours of node `s`
 reached by `edge_type`", which is a **variable-length list** keyed
 by `s`.
- We want **O(1)** access from `src_id` to the offset of its neighbour
 list; Parquet's repetition-level / definition-level list encoding
 gives us O(row_group) at best.
- Edge property values are co-located with the neighbour they describe.
 Co-locating them in independent Parquet files would lose the joint
 ordering invariant we rely on to make a single ranged GET hot.

A custom format also lets us implement two specific NamiDB
optimisations (open from PDF gap #9 / RFC-001 §"contribución propia"):

- **Power-law-aware encoding.** High-degree source nodes get a separate
 block layout (large delta, dense bitmap of neighbours when degree
 exceeds a threshold). Long-tail sources use the default
 split-top64/bottom64 encoding.
- **Edge-direction packing.** The same edge data is stored twice per
 edge type per flush: a **forward** SST sorted by `src_id`, and an
 **inverse** SST sorted by `dst_id`. Both use the same wire format,
 distinguished by `flags.INVERSE_PARTNER`. The inverse SST stores the
 *pair* `(dst, src)` in its neighbour positions, so reading "all
 in-edges of `v`" is the same code path as reading "all out-edges
 of `v`".

The choice of "always write inverse partner at flush time" (vs "only at
compaction") trades **2× write amplification** for **bounded
in-edge-query latency on freshly-flushed data**. Without inverse
partners, a `MATCH (n)-[:KNOWS]->(:Person {name: 'Bob'})` query that
hits L0 SSTs has to scan every neighbour list — for a 10 M-edge graph
that is single-digit seconds, blowing the budget. Bench data
during may later motivate making inverse generation optional
per edge type, but the default in v1 is "always".

#### 3.2 File layout

All multi-byte integers are **little-endian**. All offsets are absolute
byte offsets from the start of the file unless stated otherwise.

```text
┌─────────────────────────────────────┐ offset 0
│ File header (64 bytes, FROZEN) │
├─────────────────────────────────────┤
│ Section: key_ids │ kind = 0x0001
│ sorted UUIDv7s (16 B each) │
│ "src_ids" in fwd / "dst_ids" inv │
├─────────────────────────────────────┤
│ Section: offsets │ kind = 0x0002
│ one entry per key_id + 1 sentinel│
│ bitpacked u24/u32/u40/u48 │
├─────────────────────────────────────┤
│ Section: partners │ kind = 0x0003
│ split or dense per-group blocks │
│ "neighbours (dst)" in fwd │
│ "neighbours (src)" in inv │
├─────────────────────────────────────┤
│ Section: per_edge_lsn │ kind = 0x0004
│ u64 LE, one per edge in order │
├─────────────────────────────────────┤
│ Section: per_edge_tombstones (opt) │ kind = 0x0005
│ bitmap, 1 bit per edge │
│ (omitted when HAS_TOMBSTONES = 0)│
├─────────────────────────────────────┤
│ Section: fence_index (opt) │ kind = 0x0006
│ sparse index over key_ids │
│ (present when key_count > 65 536)│
├─────────────────────────────────────┤
│ Section: property_stream × N (opt) │ kind = 0x0100 (each)
│ one section per declared prop + │
│ `__overflow_json` if any present │
│ Zstd-compressed Arrow IPC chunk │
├─────────────────────────────────────┤
│ Footer (variable size) │
│ section table + body fields + │
│ 20-byte trailer (xxhash + len + │
│ magic) │
└─────────────────────────────────────┘ offset = file_size
```

Section **order on disk is not normative** — the footer's section
table determines the true byte ranges. The diagram above shows the
typical layout the writer produces in v1.0.

Each section is **independently addressable**: the footer carries a
`Section` table mapping `SectionKind → (offset, length, xxhash3, codec)`.
A reader that needs only `(key_ids, offsets, partners)` for a label scan
fetches exactly those three ranged GETs and ignores the property streams.

##### 3.2.1 File header (64 bytes, frozen)

The 64-byte header is **frozen for the lifetime of `format_major`**.
Adding any field here requires a major bump. Forward-compatible
extension happens only through new footer sections (see §5.2).

```text
offset size field value
─────── ──── ──────────────────────────── ────────────────────────────
 0 8 magic b"TGEDGE\0\0"
 8 1 format_major u8, current = 1
 9 1 format_minor u8, current = 0
10 2 header_size u16 = 64 (sanity check)
12 4 flags u32 bitfield (see below)
16 16 edge_type_id blake3(edge_type)[..16]
32 16 src_label_id blake3(src_label)[..16]
48 16 dst_label_id blake3(dst_label)[..16]
```

Flags:

| Bit | Name | Meaning |
|------|---------------------|------------------------------------------------------|
| 0 | `HAS_PROPERTIES` | At least one property column present in §6+. |
| 1 | `HAS_TOMBSTONES` | At least one tombstone bit is `1` (cheap shortcut). |
| 2 | `SKEW_BUCKETS` | Section 3 contains at least one skew-bucket group. |
| 3 | `INVERSE_PARTNER` | This file is the inverse-direction CSR; §1 path uses `edges-inv-`. |
| 4-31 | reserved | Must be zero in v1.0; reserved bits a v1.x reader does not recognise abort the read. |

A reader **must** reject the file if `format_major > 1` or `header_size
≠ 64`. It must treat `format_minor > 0` as forward-compatible per the
rules in §5.2. It must reject any non-zero reserved bit it does not
understand.

##### 3.2.2 Section 1: key_ids

`key_ids` is the strictly increasing, deduplicated array of key
`node_id`s present in this SST. **Semantics depend on `flags.INVERSE_PARTNER`:**

- `INVERSE_PARTNER = 0` (forward): keys are `src_id`s; partners in §3.2.4
 are `dst_id`s. Reads "out-edges of `s`".
- `INVERSE_PARTNER = 1` (inverse): keys are `dst_id`s; partners are
 `src_id`s. Reads "in-edges of `d`".

Fixed-size 16-byte UUIDv7 records. The writer asserts: every `key_id`
must be `>` the previous one (`Ord` on `[u8; 16]`), no duplicates.

This array is the binary-searchable handle into the offsets / partners
structure. Length is `key_count`; section size is `16 * key_count`.

##### 3.2.3 Section 2: offsets

`offsets[i]` is the byte offset (inside Section 3, relative to the
start of Section 3) at which the partner group of `key_ids[i]` begins.
`offsets[key_count]` is a sentinel == size of Section 3.

Encoding is bitpacked with a width chosen by the writer at close time
based on the maximum offset value:

| Section 3 size | Bits per offset | Format |
|---------------------|-----------------|------------|
| < 2²⁴ B (16 MiB) | 24 | 3-byte LE |
| < 2³² B (4 GiB) | 32 | `u32` LE |
| < 2⁴⁰ B (1 TiB) | 40 | 5-byte LE |
| < 2⁴⁸ B (256 TiB) | 48 | 6-byte LE |

A fixed-width layout is preferable to varint here because the read path
needs random access (`offsets[i]`) without scanning. The chosen width is
recorded in the footer (`offsets_bits` field).

##### 3.2.4 Section 3: partners (neighbours / sources)

Each per-key group is laid out as one of two block kinds. The writer
picks the kind per group based on degree.

Every block opens with a 1-byte tag and a varint `deg`. v1 defines two
tags; future v1.x readers may support more.

```text
┌──────────────┬─────────┬────────────────────────────────────────────┐
│ deg: varint │ tag: u8 │ payload │
└──────────────┴─────────┴────────────────────────────────────────────┘
 tag = 0x01 → split block (split-top64/bottom64 encoding)
 tag = 0x10 → dense block (raw 16-byte partners)
 tag = others → reject as Error::Corrupted in v1.0
```

The writer picks the block kind per key group based on the rule below;
the picked kind is independent of `flags.SKEW_BUCKETS`, which is set in
the header whenever **any** group of the file emitted a dense block
(`tag = 0x10`).

###### Selection rule

For a group of degree `d` with partners sorted ascending by the full
128-bit id, the writer computes the encoded byte cost of the split
block (deterministically, using the encoding below). It then emits:

```
let split_cost = …; // see "Split block — encoding" below
let dense_cost = 16 * d; // always

if d > skew_threshold || split_cost >= dense_cost {
 emit dense block (tag = 0x10)
} else {
 emit split block (tag = 0x01)
}
```

`skew_threshold` is bench-driven; the v1 default is `max(1024, 4 *
sqrt(key_count))`. The `split_cost >= dense_cost` clause is the
"always-correct fallback": for pathological partner distributions
(spanning the full `u64` range with near-uniform deltas) the split
encoding can balloon to 18 B per partner, so the dense block bounds
the worst case at 16 B per partner regardless.

###### Split block — encoding (`tag = 0x01`)

UUIDv7 splits cleanly into a top 64 bits (ms timestamp + 4-bit version
+ 12-bit sub-ms entropy) that is nearly monotonic over time, and a
bottom 64 bits that is uniformly random. We exploit the top half for
compression and write the bottom half raw.

Payload, partners sorted ascending by the full 128-bit id:

```text
top64[0]: varint // absolute top64 of partner[0]
bot64[0]: u64 LE // raw bottom64 of partner[0]
top64_delta[j]: varint // = top64[j] - top64[j-1] (j ∈ 1..deg)
bot64[j]: u64 LE // raw bottom64 of partner[j]
```

Encoded cost in bytes:
`split_cost = len_varint(top64[0]) + 8 + Σ_{j=1..deg-1} (len_varint(top64_delta[j]) + 8)`.

Typical cost (partners clustered within seconds of each other, so
`top64_delta <= 127`): **9 B per partner**. Cost when partners span
months but were created in the same year: **13–14 B per partner**.
Absolute worst case (artificial, e.g. `u64::MAX` deltas): **18 B per
partner** — the writer detects this via the selection rule above and
emits a dense block instead.

`top64_delta[j]` may legally be `0` (two partners created in the same
ms with the same 12-bit sub-ms entropy). The bottom-64 ordering breaks
the tie; the writer asserts strictly increasing 128-bit partner id, so
two partners with both halves equal are a writer bug.

###### Dense block — raw partners (`tag = 0x10`)

```text
┌──────────────┬─────────┬────────────────────────────────────────────┐
│ deg: varint │ tag: u8 │ partners: [u8; 16 * deg] │
└──────────────┴─────────┴────────────────────────────────────────────┘
```

Always-correct fallback. Used for super-nodes (`deg > skew_threshold`)
and for any group where the split encoding would not be smaller. Future
v1.x readers may support `tag = 0x11..` (e.g. Roaring on
`hash(partner) mod 2³²`) and a writer that emits them; v1.0 readers
reject any tag not in `{0x01, 0x10}` with `Error::Corrupted`.

##### 3.2.5 Section 4: per-edge LSN

For every edge, in the same order as the partner enumeration of
Section 3, one `u64` LE with the LSN at which that edge was applied.
Length is `edge_count * 8`.

Used for:

- Conflict resolution at read time when the same `(key, partner)` pair
 shows up in older and newer SSTs (the newer LSN wins).
- Compaction merge to filter shadowed edges.

##### 3.2.6 Section 5: per-edge tombstone bitmap

`ceil(edge_count / 8)` bytes; bit *j* is the tombstone flag of edge
*j* in the partner enumeration order of Section 3. A tombstone edge
keeps its position in the partner array; the reader filters it out
unless explicitly asked for history (branching / replay).

**Section-omission rule.** If no edge in the SST is tombstoned the
writer **omits** this section: it sets `flags.HAS_TOMBSTONES = 0` in
the header, and the footer's section table contains no entry of kind
`per_edge_tombstones`. A v1.X reader that finds `HAS_TOMBSTONES = 0`
must treat every edge as non-tombstoned without looking for the section;
a reader that finds `HAS_TOMBSTONES = 1` but no entry in the section
table treats the SST as corrupted.

**Forward / inverse consistency invariant.** When an edge `(s, d, lsn)`
is tombstoned in the writer's frozen memtable, its corresponding entry
in **both** the forward partner (key = `s`, partner = `d`) and the
inverse partner (key = `d`, partner = `s`) is tombstoned at the same
LSN. The writer enforces this by reading the tombstone bit from a
single canonical source (the frozen memtable's `MemOp::Tombstone`)
during the construction of each partner, never from independent
computations over the two transpositions. Tests #5 and #6 (§7) lock
this invariant down.

##### 3.2.7 Sections 6..N: property streams

One section per declared property `q` on this edge type. Each section
holds a Zstd-compressed Arrow IPC chunk with a single column whose
row *j* corresponds to edge *j* in the partner enumeration order.

We choose Arrow IPC (not Parquet) for property streams because:

- Each section already lives inside the CSR file's footer table, so we
 do not need Parquet's column metadata.
- Arrow IPC's record batch layout maps 1:1 to a column; zero-copy
 decode with `arrow-ipc::reader::StreamReader`.
- Reusing Arrow primitives means a property column for an edge looks
 identical to a property column for a node — same `DataType ↔
 ArrowDataType` mapping as in `namidb-core::schema`.

Schema-undeclared properties on edges land in a single
`__overflow_json` property stream with `name = "__overflow_json"`.
Unlike node SSTs (where the `__overflow_json` *column* is always
present in the Parquet schema, possibly all-null), the edge SST
`__overflow_json` **section** is **only emitted when at least one edge
has overflow data**. When no overflow is present, the writer omits the
section and `HAS_PROPERTIES` reflects only the declared properties
(or is `0` if there are none). A reader that needs overflow data and
finds no `__overflow_json` section reads every edge's overflow as
`null`.

##### 3.2.8 Footer

The footer is the last bytes of the file. It has a fixed-length
**trailer** (always 20 bytes at the very end) and a variable-length
**body** that precedes the trailer.

```text
┌──────────────────────────────────────────────────┐ ← footer body start
│ Section table: section_count × SectionEntry │
│ SectionEntry { │
│ kind: u16, // discriminator │
│ offset: u64, // from file byte 0 │
│ length: u64, // bytes │
│ codec: u8, // 0=none, 1=zstd │
│ reserved: u8, │
│ xxhash3_64: u64, // over the on-disk │
│ // bytes of the │
│ // section as stored │
│ name_len: u8, │
│ name: [u8; name_len], // utf8 │
│ } │
├──────────────────────────────────────────────────┤
│ section_count: u32 │
│ key_count: u64 │
│ edge_count: u64 │
│ offsets_bits: u8 // 24 / 32 / 40 / 48 │
│ min_key_id: [u8; 16] │
│ max_key_id: [u8; 16] │
│ min_lsn: u64 │
│ max_lsn: u64 │
│ schema_version_min: u64 │
│ schema_version_max: u64 │
├──────────────────────────────────────────────────┤ ← trailer start
│ footer_xxhash3_64: u64 (covers footer body) │
│ footer_len: u32 (body + trailer length) │
│ magic: 8 bytes b"TGEDGE\xFE\xEF" │
└──────────────────────────────────────────────────┘ ← end of file
```

Precise definitions:

- **`footer_xxhash3_64`** is computed over the **footer body** only:
 from the first byte of the section table up to and including
 `schema_version_max` (i.e. all bytes between the body-start marker
 and the trailer-start marker above). It does *not* cover any byte
 of the trailer itself.
- **`footer_len`** is the total byte length of footer body + trailer
 (i.e. the offset from the trailer's last byte to the body's first
 byte, inclusive). Equivalently: `footer_len = file_size -
 body_start`. A reader uses this to find the body start once it has
 the trailer.

Section `kind` discriminators (u16):

| Value | Kind | Notes |
|----------|----------------------------|-----------------------------------------------------------------------|
| 0x0001 | key_ids | Mandatory. |
| 0x0002 | offsets | Mandatory. |
| 0x0003 | partners | Mandatory. |
| 0x0004 | per_edge_lsn | Mandatory. |
| 0x0005 | per_edge_tombstones | Optional (see §3.2.6). |
| 0x0006 | fence_index | Optional; required when `key_count > 65 536`. See §3.2.9. |
| 0x0100 | property_stream | Optional; **one entry per property**, distinguished by `name`. Reserved names: `__overflow_json` for schema-undeclared props. |
| Others | reserved | A v1.0 reader skips unknown kinds outside the reserved ranges (forward-compat per §5.2). |

All property streams share the same `kind = 0x0100`; the `name` field
discriminates them. The writer rejects any property declaration whose
`name` collides with a reserved column name (see §2.1) at SST creation
time — `__overflow_json` is the only legal entry beginning with `__`.

A reader locates the footer by:

1. Ranged GET for the last 4 KiB of the object (covers any footer up to
 ~4 KiB; for SSTs with few sections this is enough).
2. Read the trailing 8-byte magic at the end of the response. If
 absent, expand to the last 64 KiB and retry. If still absent the
 file is corrupt.
3. From the trailer read `footer_len`. If the prefetched window is too
 small, issue a second ranged GET for
 `[file_size - footer_len, file_size)`.
4. Verify `footer_xxhash3_64` against the body bytes. Mismatch →
 `Error::Corrupted`.
5. Validate that every `SectionEntry`'s `[offset, offset + length)`
 range lies strictly within `[64, file_size - footer_len)`. Any
 overflow → `Error::Corrupted`.

The section table is sorted ascending by `offset`. A reader can
linear-scan by `kind` when looking up a specific section.

##### 3.2.9 Fence-pointer index (optional)

The naive lookup of "find `s` in `key_ids`" requires either fetching
the entire `key_ids` section (16 B × `key_count`) or doing a remote
binary search (≈`log2(key_count)` ranged GETs over 16-byte windows).
For `key_count = 1 M` the first option costs a 16 MiB cold GET; the
second costs ~20 round-trips. Neither is acceptable for the §14.1
cold-query budget when SSTs grow past a few hundred thousand keys.

The fence-pointer index solves this with a sparse local index over
`key_ids`. The writer emits one fence entry **every `fence_stride`
keys** (default `fence_stride = 256`). Each entry stores the key value
and the byte offset of that key within the `key_ids` section.

```text
┌──────────────────────────────────────────────────┐
│ fence_stride: u32 (e.g. 256) │
│ entry_count: u32 (= ceil(key_count / stride)) │
│ entries: [ FenceEntry ; entry_count ] │
│ FenceEntry { │
│ key: [u8; 16], // = key_ids[i * fence_stride] │
│ key_ids_offset: u64, // = i * fence_stride * 16 │
│ // (relative to byte 0 │
│ // of section key_ids) │
│ } │
└──────────────────────────────────────────────────┘
```

Total size: `4 + 4 + entry_count * 24` bytes. For 1 M keys with stride
256 → 3 906 entries → ≈94 KiB — cacheable by foyer on first probe.

**Writer rule.** A fence index is **emitted** when `key_count >
65 536`. Below this threshold the entire `key_ids` section is small
enough (≤ 1 MiB) to fetch and binary-search in memory cheaply.

**Reader algorithm for "find offset of key `k` inside `key_ids`":**

```text
if footer has no fence_index section:
 fetch the full key_ids section (≤ 1 MiB by construction)
 binary search in memory
else:
 fetch the fence_index section once (cached)
 binary search the fence entries to find the bracket
 [fence[i].key, fence[i+1].key) containing k
 issue one ranged GET for
 key_ids[fence[i].key_ids_offset .. fence[i+1].key_ids_offset]
 binary search that window in memory
```

Total cold cost: **2 GETs** (fence + key_ids window) regardless of
`key_count`. Warm cost: 1 GET (the window; fence is cached). The
fence index is a v1.0 optional artefact: an older reader that ignores
the section still works correctly via the naive path.

#### 3.3 Statistics extraction

When the writer closes either partner of an edge SST it emits:

```rust
pub enum EdgeDirection {
 /// Keys are src_id; partners are dst_id. File path uses `edges-fwd-`.
 Forward,
 /// Keys are dst_id; partners are src_id. File path uses `edges-inv-`.
 Inverse,
}

pub struct EdgeSstStats {
 pub direction: EdgeDirection,
 pub key_count: u64,
 pub edge_count: u64,
 pub tombstone_count: u64,
 pub min_key_id: [u8; 16],
 pub max_key_id: [u8; 16],
 pub min_lsn: u64,
 pub max_lsn: u64,
 pub degree_histogram: DegreeHistogram,
 pub property_stats: Vec<PropertyColumnStats>,
 pub schema_version_min: u64,
 pub schema_version_max: u64,
}

pub struct DegreeHistogram {
 /// 64 log2-spaced buckets:
 /// counts[i] = #keys with deg in [2^i, 2^(i+1))
 pub counts: [u32; 64],
 pub max_degree: u64,
 pub sum_degree: u64,
}
```

For a **forward** partner, `degree_histogram` describes out-degree.
For an **inverse** partner, it describes in-degree. The cost-based
optimizer reads the histogram of the partner it is about to traverse.

The bloom filter is **not** part of this struct (side-car; see §4.2).

#### 3.4 Read access patterns and ranged GETs

This section quantifies the GET count for the common access patterns of
v1, to make the columnar layout's cost explicit.

Notation:
- `D` = direct-cached descriptor reads (`current.json` + manifest body),
 amortised across queries.
- `B` = bloom side-car GET (1 GET per SST when min/max does not already
 exclude). Cached by foyer; second visit free.
- `F` = SST footer GET (last 4 KiB; cached per SST).
- `Khdr` = SST header GET (first 64 B + the section table prefix; can
 be coalesced with `F` in one ranged GET for SSTs ≤ ~16 MiB).

**Pattern A — point lookup `node_id = v`.**

```
D + (per candidate SST) [B + F + ranged GET into the matching page]
```

Cold per SST: ~3 GETs. With foyer warm, `B + F` are free; only the
page GET remains (~1 GET).

**Pattern B — out-edge expansion of a known src `s` (forward SST).**

The reader resolves `s → index_in_key_ids → offset_in_partners → range
of partners` using the fence index (§3.2.9) when present, or the full
`key_ids` section otherwise.

For SSTs **without** a fence index (`key_count ≤ 65 536`, so `key_ids
≤ 1 MiB`):

```
D + B + F + GET key_ids (≤ 1 MiB)
 + GET offsets[i..i+1]
 + GET partners[off..off+len]
```

Cold per SST: **5 GETs**; the `key_ids` and `offsets` ranges coalesce
into a single ranged GET when `key_count * 16 + offsets_bytes ≤ 1
MiB` (true for L0 SSTs after a single flush). Warm: `B + F + key_ids +
offsets` are foyer-cached; only the `partners` GET remains
(**1 GET warm**).

For SSTs **with** a fence index (`key_count > 65 536`):

```
D + B + F + GET fence_index (~100 KiB)
 + GET key_ids window (≤ fence_stride * 16 ≈ 4 KiB by default)
 + GET offsets[i..i+1]
 + GET partners[off..off+len]
```

Cold per SST: **6 GETs**, all independent and parallelisable. Warm: 1
GET (partners).

**Pattern C — in-edge expansion of a known dst `d` (inverse SST).**

Identical to Pattern B, just with the inverse partner SST.

**Pattern D — edge expansion with property predicate
(e.g. `where edge.since > date`).**

Pattern B + 1 additional GET on the property stream's range
corresponding to the partners we touched. Cold per SST: ~6 GETs;
property stream GET coalesces with `partners` when both ranges lie in
the same MiB window.

The "concurrent ranged GETs" feature of `object_store::aws` lets us
fire patterns B/D's GETs in parallel; cold p50 wall time is bounded by
the slowest GET, not their sum. With `S3 Express One Zone` the per-GET
floor drops from ~30 ms to ~5 ms — directly on the §14.1 budget.

### 4. Embedded statistics + bloom side-car in the manifest

#### 4.1 Extended `SstDescriptor`

This RFC promotes `SstDescriptor` from the minimal version in RFC-001
to the form below. Everything in this struct is JSON-cheap (a few
hundred bytes per SST excluding `property_stats`, which scales with
column count). For 100 K SSTs the manifest stays under ~10 MiB, the
budget at which we switch JSON → Arrow IPC (recorded as an Open Question
in RFC-001).

```rust
pub struct SstDescriptor {
 // ── identity ──
 pub id: Uuid,
 pub kind: SstKind, // Nodes | EdgesFwd | EdgesInv
 pub scope: String, // label or edge_type
 pub level: SstLevel,
 pub path: String, // relative to namespace

 // ── physical ──
 pub size_bytes: u64,
 pub row_count: u64, // node rows or edge rows
 pub created_at: DateTime<Utc>,

 // ── key range (raw bytes; serialised as base64 in JSON) ──
 pub min_key: [u8; 16], // node_id (Nodes) or key_id (Edges)
 pub max_key: [u8; 16],
 pub min_lsn: u64,
 pub max_lsn: u64,
 pub schema_version_min: u64,
 pub schema_version_max: u64,

 // ── stats embedded ──
 pub property_stats: Vec<PropertyColumnStats>,
 pub kind_specific: KindSpecificStats,

 // ── bloom side-car pointer (None when the SST is small enough
 // that scanning is cheaper than probing; see §4.2) ──
 pub bloom: Option<BloomDescriptor>,
}

pub enum SstKind {
 Nodes,
 EdgesFwd,
 EdgesInv,
 // Vectors lands in RFC-007; reserved here so reader code can match
 // exhaustively against the v1 set.
}

pub enum KindSpecificStats {
 Nodes { tombstone_count: u64 },
 Edges {
 // key_count == row_count for nodes; for edges key_count is
 // distinct src/dst count (depending on direction).
 key_count: u64,
 tombstone_count: u64,
 degree_histogram: DegreeHistogram,
 },
}
```

JSON serialisation: `min_key` / `max_key` are 16-byte arrays serialised
as **base64** (`base64::STANDARD`). All other fields use their natural
JSON encoding.

#### 4.2 BloomDescriptor (side-car pointer)

The bloom filter for an SST lives in its own object next to the SST
body. The manifest only carries a pointer to it plus the parameters
needed to probe it without re-reading the body:

```rust
pub struct BloomDescriptor {
 pub path: String, // object_store path of the side-car
 pub size_bytes: u32, // total side-car file size (header + blocks + trailer)
 pub key_count: u64, // number of keys inserted into the filter
 pub bits_per_key: u8, // default 10 → ~1 % FPR
 pub block_count: u32, // 256-bit (32-byte) blocks
 pub xxhash3_64: u64, // checksum over the side-car body (per §4.2 wire spec)
}
```

We use the **split-block bloom filter (SBBF)** construction Parquet
adopted — a single 64-bit hash per key drives a deterministic 8-bit
mask inside one 256-bit block. There is no separate `k_hashes`
parameter (the "k" is fixed at 8 by construction). The hash function
is **xxHash3-64** (same library, same seed = 0 as elsewhere in this
RFC). Block selection from a hash `h`:

```
let block_index = ((h >> 32) * block_count as u64) >> 32;
```

The 8-bit mask inside the chosen block is the standard SBBF mask (see
Putze et al., 2010; identical to Parquet's `bloom_filter_algorithm =
SPLIT_BLOCK`). Implementations crib the constants from `parquet-rs
55::bloom_filter`.

The total side-car size is exactly `28 (header) + 32 * block_count + 8
(trailer xxhash)`. For 10 bits / key the writer rounds up
`block_count = ceil(key_count * bits_per_key / 256)`; e.g. 1 M keys ⇒
`block_count = 39 063` ⇒ side-car = 1 250 052 bytes ≈ 1.19 MiB.

##### Side-car wire format

```text
┌──────────────────────────────────────┐ offset 0
│ magic: 8 bytes b"TGBLOOM\0" │
│ format_major: u8 = 1 │
│ format_minor: u8 = 0 │
│ reserved: u16 = 0 │
│ bits_per_key: u8 │
│ reserved2: u8 = 0 │ // was k_hashes pre-rev3; kept
│ │ // for alignment, value MUST be 0
│ reserved3: u16 = 0 │
│ block_count: u32 │
│ key_count: u64 │
├──────────────────────────────────────┤
│ blocks: [SbbfBlock; block_count] │
│ SbbfBlock = [u8; 32] │
├──────────────────────────────────────┤
│ xxhash3_64 over the entire file │
│ minus the trailing 8 bytes: │
│ trailing: u64 LE │
└──────────────────────────────────────┘
```

Split-block bloom filters (SBBF) at 10 bits/key give ~1 % FPR — the
same parameters Parquet uses internally. For 1 M keys the side-car is
≈1.25 MiB; for a typical SST of 100 K–200 K keys it is
≈125–250 KiB. **A reader probes the bloom by**:

1. (Optional) `min_key`/`max_key` overlap test — manifest-only, no GET.
 If no overlap, skip the SST.
2. Resolve `bloom.path` to an absolute object_store path.
3. Issue one ranged GET for the side-car body; foyer caches it after
 the first probe per process.
4. Verify `xxhash3_64`. Run k hashes against the appropriate
 `SbbfBlock`. If absent, skip the SST.

The bloom over `node_id` (for node SSTs) and over `key_id` (for edge
SSTs of either direction) is therefore the gate between "manifest says
maybe" and "let's pay for the SST body GET".

For very small SSTs (`size_bytes < 256 KiB`), the writer **omits the
bloom side-car** entirely — `SstDescriptor.bloom` is set to `None`
and no `.bloom` object exists on object storage. A 200-key SST is
faster to scan than to probe. Readers seeing `bloom = None` skip the
bloom step (and skip the corresponding ranged GET) but still respect
the manifest's `min_key`/`max_key` overlap test.

#### 4.3 PropertyColumnStats

```rust
pub struct PropertyColumnStats {
 pub name: String,
 pub null_count: u64,
 pub min: Option<StatScalar>,
 pub max: Option<StatScalar>,
 pub ndv_estimate: Option<HllSketchBytes>, // 1 KiB HLL++; None for vectors/json
}

pub enum StatScalar {
 Bool(bool),
 Int32(i32),
 Int64(i64),
 Float32(f32), // NaN / Inf are stat-disqualifying; field is None
 Float64(f64), // idem
 Utf8(String),
 Binary(Bytes),
 Date32(i32),
 TimestampMicrosUtc(i64),
}
```

Vector columns (`FloatVector { dim }`) and `Json` columns produce no
`min`/`max` (they remain `None`) but still contribute a `null_count`.
The `__overflow_json` column always produces `min`/`max = None`,
`null_count` only — its `ndv_estimate` is also `None` (HLL over JSON
documents has no operational use here).

### 5. Wire compatibility

#### 5.1 Node SSTs

Parquet itself carries its own version + magic (`PAR1`) and is
forward-compatible across `parquet-rs` minor versions. We pin
`parquet = "55"` workspace-wide. Reading SSTs written by future
NamiDB builds works as long as we do not introduce new logical
column conventions; if we do, we will bump a `node_sst_format` field
in `SstDescriptor.kind_specific` so old readers can refuse.

`__overflow_json` is required by **all** v1 node SSTs (even when every
row is null). A reader that loads an SST missing this column refuses
with `Error::Corrupted { detail: "node SST missing __overflow_json" }`.

#### 5.2 Edge SSTs

Edge SSTs are **owned** by NamiDB. The compatibility contract is:

| Condition observed by reader v1.X (X ≥ 0) | Action |
|------------------------------------------------------------|-------------------------------------------------------------|
| `format_major > 1` | Refuse: `Error::Corrupted`. |
| `format_major < 1` | Refuse: `Error::Corrupted` (no v0 exists). |
| `format_major = 1`, `header_size ≠ 64` | Refuse: `Error::Corrupted`. |
| `format_major = 1`, `format_minor ≤ X` | Read normally. |
| `format_major = 1`, `format_minor > X` | Read normally; skip any footer section whose `kind` is not in this reader's v1.X table; refuse if any section table entry crosses the file end. |
| Unknown reserved bit in `flags` | Refuse: an unknown flag implies an unknown invariant. |
| Unknown `partners` block tag | Refuse: `Error::Corrupted`. |
| Unknown footer section `kind` outside the reserved ranges | Skip the section (forward-compat). |

A writer **must not** introduce a breaking change to the 64-byte header
or to any existing section's internal layout without bumping
`format_major`. Adding a new footer section kind is a `format_minor`
bump only. Removing a footer section kind is a major bump.

#### 5.3 Side-cars

The bloom side-car follows the same major / minor convention. v1.0
readers refuse any bloom side-car with `format_major > 1`.

### 6. Implementation plan (Rust crate layout)

Inside `crates/namidb-storage/src/sst/`:

```text
sst/
├── mod.rs # re-exports + common types (SstId, BloomDescriptor, …)
├── stats.rs # PropertyColumnStats, DegreeHistogram, HLL sketch
├── bloom.rs # SBBF build + probe; side-car wire format
├── nodes.rs # NodeSstWriter, NodeSstReader (Parquet)
└── edges/
 ├── mod.rs # public API: EdgeSstWriter, EdgeSstReader, EdgeDirection
 ├── header.rs # 64-byte header struct + serde
 ├── footer.rs # section table + xxhash3 + magic (per §3.2.8)
 ├── writer.rs # streaming writer (forward + inverse in one pass)
 ├── reader.rs # ranged-GET reader; section cache; bloom integration
 ├── encoding.rs # bitpacked offsets, split-top64/bottom64 neighbours,
 │ # selection rule (split vs dense)
 ├── fence_index.rs # writer + reader for the optional fence-pointer index
 └── inverse.rs # in-memory transpose of a FrozenMemtable edge bucket
```

New types lift into `namidb-storage::lib.rs` as part of the public
crate API for downstream crates (query engine). The
`manifest` module is updated in lockstep to carry the extended
`SstDescriptor`.

Two new workspace dependencies are required:

- `xxhash-rust` — feature `xxh3`, used for all SST + bloom checksums.
- `base64` — used by the `min_key` / `max_key` JSON encoding in the
 manifest.

### 7. Test plan

The following tests land alongside this RFC's implementation. Test
budget: bring the workspace from 36 → ≥ 70 passing tests.

1. **Round-trip property nodes.** Build a memtable of `Person` rows,
 freeze, write Parquet SST to `object_store::memory::InMemory`,
 read back, assert byte-for-byte equality of property values.
2. **Overflow round-trip.** Write a node with one declared and one
 undeclared property; assert the undeclared one round-trips through
 `__overflow_json` losslessly.
3. **Tombstone semantics.** Insert + delete + insert at increasing
 LSNs; read back; assert the latest LSN wins and that
 deleted-then-reinserted nodes are present.
4. **Edge CSR round-trip (forward).** Build a graph with 100 K edges
 across 10 K sources, write forward CSR, read back, assert neighbour
 lists equal.
5. **Edge CSR inverse partner.** Same graph; assert the inverse SST,
 when probed by `dst`, returns each src that originally pointed to
 that dst, in sorted order.
6. **Inverse partner == transposed forward.** Build a graph; write
 both partners; for every edge, assert it is present in both.
7. **Edge skew bucket.** Construct one super-node with degree 5 000,
 the rest degree ≤ 4; assert the writer emitted `tag = 0x10` for
 that group and `tag = 0x01` otherwise; reader returns the full list.
8. **Split-encoded compression win.** Generate 1 000 partners with all
 their top64 equal (same ms); assert the encoded size is
 `< 9 * 1000 + small_overhead` (vs 16 KiB raw).
8a. **Split-to-dense fallback.** Construct a 100-partner group whose
 partners are spaced so that every `top64_delta` would require ≥ 9
 varint bytes; assert the writer emitted `tag = 0x10` (dense) for
 that group and that the bytes-on-disk for the group are exactly
 `1 (varint deg) + 1 (tag) + 100 * 16`.
8b. **Reserved column name rejected.** Build a `SchemaBuilder` with a
 `PropertyDef { name: "tombstone", … }`; assert
 `Error::SchemaConflict`.
8c. **Fence-pointer index round-trip.** Build an edge SST with
 `key_count = 200 000` (above the fence threshold); assert that
 the footer contains a `fence_index` section, that the reader
 cold-path issues exactly 2 GETs for a `src` lookup (fence + window),
 and that the result matches a naive linear scan.
8d. **Fence-pointer index absent below threshold.** Build an edge SST
 with `key_count = 1 000`; assert no `fence_index` section in the
 footer and that the reader takes the "fetch full key_ids" branch.
8e. **Tombstone consistency fwd ↔ inv.** Flush a memtable with one
 tombstoned edge `(s, d, lsn)`; assert that the forward partner has
 `tombstone_bit[j] = 1` at the position corresponding to
 `(s → d)` and the inverse partner has `tombstone_bit[k] = 1` at
 the position corresponding to `(d → s)`, with both LSNs equal.
9. **Random-access edge lookup.** Open the reader, query `src=X`,
 assert only the expected ranged GETs hit the store (use the
 `object_store::memory::InMemory` plus a counting wrapper). Validate
 pattern B's GET count.
10. **Stats correctness.** After writing an SST, the returned stats
 match those independently computed from the source data
 (`row_count`, `min/max`, `tombstone_count`, `degree_histogram`).
11. **Bloom correctness.** Bloom contains every inserted key (FPR
 check on a held-out set is ≤ 2 × theoretical).
12. **Bloom side-car wire.** Write a bloom side-car, corrupt one byte,
 assert `Error::Corrupted` at probe time.
13. **Small SST omits bloom.** Write an SST with `size_bytes < 256 KiB`;
 assert `bloom.path == ""` in the descriptor and the reader uses
 the in-body scan path.
14. **Footer corruption.** Truncate the last 16 bytes of an edge SST,
 assert `Error::Corrupted`.
15. **xxHash3 mismatch detected.** Flip one byte inside a section's
 body, assert the section's checksum verification fails when read.
16. **Forward-compat skip.** Write an SST with a synthetic
 `section_kind = 0x0FFF` of payload `"ignored"`; the v1 reader must
 ignore it and still return correct data.
17. **Major mismatch refused.** Manually flip `format_major = 2` in
 the header, assert the reader returns `Error::Corrupted`.
18. **Header size mismatch refused.** Flip `header_size = 80`, assert
 the reader refuses.
19. **Unknown reserved flag refused.** Set flag bit 5, assert
 `Error::Corrupted`.
20. **LocalStack integration.** A single end-to-end test (`#[ignore]`)
 that writes both a node SST and a forward+inverse edge pair through
 `object_store::aws` against LocalStack and reads them back,
 including pattern B GET-count assertions against the LocalStack
 request log.

## Alternatives considered

### A. Parquet for edges too

Use Parquet `list<struct<dst, props>>` keyed by `src_id`. Rejected:

- A list-shaped column needs Parquet repetition levels, which add ~2
 bytes per edge of metadata and a definition-level mask that has to
 be walked on read.
- Random access to "neighbours of src = X" still costs O(row_group),
 not O(1), because Parquet has no random access into a list cell.
- We lose the ability to encode the skew optimisation cleanly (would
 require a sibling sparse column).

### B. Lance v2 for edges

Lance v2 is excellent for vectors and blobs but is not optimised for
the adjacency-list shape. Its strengths (zero-copy random access into
blob columns; Vamana / IVF integration) do not map onto CSR. We will
use Lance for the **vector** SST kind in RFC-007.

### C. SlateDB's SST verbatim

SlateDB is a KV store. Its SST format is well-tuned for `(key, value)`
pairs but does not carry the columnar invariants we need for property
columns or for CSR offsets. Reusing it would force every read to do a
key-decode pass that we get for free in Parquet.

### D. Iceberg manifest + Parquet data files

We considered structuring SSTs as an Iceberg table. Rejected for v1
because:

- Iceberg's manifest layout is heavier than ours and adds a level of
 indirection irrelevant to a single-writer LSM.
- Iceberg's snapshot semantics overlap with ours but with different
 retention semantics; we would not be able to express "branch with
 fork retention" without subverting Iceberg's vacuum.
- We will revisit in **RFC-014 Iceberg integration** as an *export*
 surface — write an Iceberg view *of* the SSTs, not store them as one.

### E. Embedded bloom (revision-1 plan)

Originally we planned to inline the bloom inside the manifest
`SstDescriptor`. Rejected during revision 2: a 1.25 MiB raw bloom
becomes 1.65 MiB base64, and a 100 K-SST namespace would produce a
~165 GB manifest. Side-car keeps the manifest under the JSON budget
while still allowing the bloom to be fetched lazily and cached by
foyer.

### F. Single-direction edge SSTs

Skipping the inverse partner halves write amplification at flush time.
Rejected for v1: a `MATCH (n)-[:KNOWS]->(:Person {name: 'Bob'})` query
against L0 SSTs degenerates to `O(|E|)` neighbour scans, which destroys
the §14.1 cold-query budget. Single-direction may be reintroduced as a
per-edge-type override (e.g. for write-heavy log-shaped edges) once we
have bench data.

### G. Per-section CRC32 (revision-1 plan)

Originally CRC32 IEEE, matching the WAL. Rejected during revision 2 in
favour of xxHash3-64 for SSTs only. Rationale: S3 already provides
strong integrity (HTTP MD5 / CRC32C) end-to-end, so SST checksums exist
to defend against client-side / memory-side corruption. xxHash3 is
~3-5 × faster than CRC32 IEEE at the same defence quality for the
fail-modes that matter at this layer. WAL keeps CRC32 because its
fail-modes include torn 4 KiB writes where CRC's burst-error guarantees
are useful.

## Drawbacks

1. **Bloom side-car costs a GET.** A query that does not have
 min/max pruning available pays one extra ranged GET per candidate
 SST on the cold path. Mitigation: foyer caches every bloom side-car
 after first touch (typical size 125 KiB–1.5 MiB), and the
 "small-SST omit-bloom" rule means the cost only applies to SSTs
 large enough to benefit anyway. Bench-targeted.

2. **Inverse partner doubles write amp on the edges path.** Mitigation:
 per-edge-type override is a v1.1 follow-up. The expectation is that
 most graph workloads are write-once-read-many, so the asymmetry
 between flush and query cost is acceptable.

3. **Custom edge format is more code to maintain.** We are taking on
 a wire format that we now own forever. Mitigations: a small
 `format_major / minor` invariant, exhaustive round-trip tests, and
 a `namidb-storage` CLI subcommand (`inspect-sst <path>`) to
 dump the header / footer for ops debugging (lands with the writer).

4. **Parquet's per-row-group footer overhead** dominates for very
 small node SSTs (< 10 K rows). Mitigation: the writer aggregates
 short flushes to ≥ 128 K rows when possible — see flush path
 RFC-003 for the policy.

5. **Skew block ships only `tag = 0x10` dense.** Roaring integration
 (`tag = 0x11`) lands once bench data justifies it; until
 then a super-node with 1 M out-edges uses 16 MiB of dense storage
 per SST. Acceptable for prototype.

6. **`f32` / `f64` stats skip NaN / Inf.** This matches Parquet's
 strict stats but means we silently drop min / max when a column
 contains them. Predicate pushdown gracefully falls back to per-row
 evaluation. Tracked.

7. **Manifest growth is bounded but not constant.** With 100 K SSTs
 the manifest is still ~10 MiB (dominated by `property_stats`,
 `degree_histogram`, key ranges). The JSON → Arrow IPC switch lands
 when bench data warrants it; until then we set a hard 10 MiB cap
 and the writer fails the commit if a new manifest would exceed it
 (with a clear error pointing at the migration RFC).

## Open questions

1. **Bloom probe in pattern A vs page-index probe.** For point lookups
 on node SSTs the Parquet page index already gives row-group
 pruning at min/max granularity. The bloom helps when min/max
 intervals overlap. Bench may show that the bloom is unnecessary
 for node SSTs and only edge SSTs need it. Leaving the bloom on
 for both kinds in v1 keeps the read path uniform.

2. **HLL sketch byte budget.** 1 KiB per column per SST is the current
 default. Could be lowered to 256 bytes (less accuracy) if manifest
 growth becomes a bottleneck before the IPC migration. Bench-driven.

3. **`tag = 0x11` (Roaring) timing.** Promote when a workload has a
 super-node with degree ≥ 1 M *and* benches show ≥ 2 × savings.

4. **Edge property layout: per-edge vs per-key chunks.** Today property
 stream row *j* maps to edge *j* in partner enumeration order. An
 alternative is to chunk by key group so that all properties of a
 single src's out-edges are contiguous. The current choice maximises
 columnar scan efficiency for `WHERE edge.prop ...` predicates; the
 alternative would maximise per-key locality. Defer until query
 engine benches.

5. **JSON → Arrow IPC manifest threshold.** Currently set at 10 MiB.
 This RFC's structures keep manifests well under that for 100 K SSTs;
 the threshold will be re-evaluated when a namespace approaches it.
 Tracked as RFC-003 follow-up.

6. **Per-edge-type inverse opt-out.** Defer to bench data; a
 "log-shaped" edge type (e.g. immutable events) might never need
 in-edge expansion, and could opt out of inverse partner generation
 at schema declaration time.

## References

- Parquet specification, https://github.com/apache/parquet-format.
- Lemire, Boytsov, **Decoding billions of integers per second through
 vectorisation** (Software: Practice & Experience, 2015). Varint /
 bitpacking implementation reference.
- Chambi et al., **Better bitmap performance with Roaring bitmaps**
 (SP&E, 2016). Section 3.2.4 skew layout.
- Putze, Sanders, Singler, **Cache-, Hash- and Space-Efficient Bloom
 Filters** (J. Exp. Algorithmics, 2010). SBBF foundations.
- Heule, Nunkesser, Hall, **HyperLogLog in practice** (EDBT 2013).
- Y. Collet, **xxHash3** specification, https://github.com/Cyan4973/xxHash.
- Jin et al., **Kùzu** (CIDR 2023). Property graph + CSR + factorised
 representation in a single binary; reference architecture.
- Hu et al., **EmptyHeaded** (SIGMOD 2017). WCOJ over factorised
 intermediate results — relevant for how SST stats inform planning.
- **DuckDB DataChunk + Parquet integration**, https://duckdb.org/docs.
 Reference for column-store integration over Parquet.
- **turbopuffer architecture**, https://turbopuffer.com/docs/architecture.
 Embedded-stats-in-manifest pattern + bloom side-car pattern.
- **SlateDB SST format**, https://slatedb.io. We diverge by being
 column-oriented and CSR-aware.
- **Apache Iceberg manifest spec**, https://iceberg.apache.org/spec/.
 Reference for the design we did *not* adopt as the primary layout,
 but will use as an *export* surface in RFC-014.
- **UUIDv7 specification**, RFC 9562. Layout that underlies the
 split-top64 / bottom64 encoding in §3.2.4.
