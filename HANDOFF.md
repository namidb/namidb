# Handoff: Roadmap Progress

**Session:** 2026-06-18/19 (session 836fd4ad)
**Branch:** `feat/s3b-versioned-pointer`
**Method:** ultracode — 94-agent orchestration (68-agent adversarial review + 26-agent research/design) driving implementation.

## Test status (all green)

| Crate | Tests |
|-------|-------|
| namidb-storage | 310 (304 + 5 vector build + 1 DDL register, behind `vector-index`) |
| namidb-query | 462 default (+ 4 CALL parse, + 3 CALL e2e) → 471 with `vector-index` (adds 4 vector rewrite + 5 DDL parse + 1) |
| namidb-ann | 21 (new) |
| namidb-server | 32 (+ integration suites, +1 DDL HTTP behind `vector-index`) |
| namidb-mcp | 20 |
| namidb-graph | 10 |

## Item 12 — DiskANN/Vamana ANN (shipped this session, 9/9)

All 9 steps done, behind Cargo feature `vector-index` (default off — existing
namespaces stay byte-identical). Commits `9c85c22` → `<this commit>`:

1. **namidb-ann crate** (`39d3210`): Vamana graph build (α-robust prune,
   medoid entry, Auto/BruteForce/Random init) + greedy beam search
   (candidate min-heap, result max-heap, visited bitset, converged
   termination). `VectorSpace` trait with `F32CosineSpace` (recall-golden)
   + `Int8Space` (shipped path, cosine on int8 is scale-invariant). Recall
   validated: f32 ≥0.90, int8 ≥0.80 on clustered unit vectors. 21 tests.
2. **Storage scaffolding** (`ba6d324`): `SstKind::VectorGraph` +
   `KindSpecificStats::VectorGraph`, `Manifest.vector_indexes` +
   `VectorIndexDescriptor`/`VectorMetric` (with the CRITICAL
   `next_version()` clone-forward), `compact.rs` VectorGraph bucketing.
3. **Query scaffolding** (`ba6d324`): `LogicalPlan::VectorSearch` +
   `VectorDistance`, all ~17 forced match arms, `flat_vector_search` fallback
   (scan + project-embedding-only + score + top-k) on both flat & factor
   paths, `exec/expr::vector_score` shared helper.
4. **Build hook + reader** (`8e7e787`): compaction rebuilds Vamana indexes
   from GC'd merged node rows (id-primary label filtering via `label_dict`;
   rebuild-not-merge). `sst/vector.rs` self-contained `.vg` body
   (magic + bincode) + `VectorGraphIndex::decode/search` (full-precision
   rerank). End-to-end compaction test: 160 clustered docs → searchable
   index, ≥8/10 hits from the queried cluster.
5. **Optimizer rewrite** (`b74cc3d`): `optimize/vector_search.rs` collapses
   the flat KNN shape `TopN(Project([Filter]NodeScan))` into `VectorSearch`
   when a backing index exists (conservative: SKIP/non-DESC/multi-key/
   DISTINCT/metric-mismatch/no-index ⇒ unchanged). `StatsCatalog.vector_indexes`
   + `vector_index_for(...)` lookup. 4 unit tests.
6. **Executor dispatch** (`b74cc3d`): `Snapshot::vector_search` (unions
   in-scope `.vg` SSTs, re-ranks) + `try_index_search` serves the
   `VectorSearch` arm from the index, falling back to flat when none applies.
7. **`CREATE VECTOR INDEX` DDL** (`<this commit>`): the missing queryability
   layer. `WriterSession::register_vector_index(desc)` — a metadata-only
   manifest commit (mirrors `attach_ssts`: `next_version` → push →
   `manifest_store.commit` → `refresh_published`; no WAL, no memtable rows),
   rejecting duplicate name / duplicate `(label, property, metric)`. The
   parser gains `Clause::CreateVectorIndex` (always-present variant per C5)
   + `parse_create_vector_index` for `CREATE VECTOR INDEX <name> ON
   :Label(property) METRIC <m> DIMENSION <n> [WITH {r, l_build, alpha}]`
   (soft-keywords VECTOR/INDEX/METRIC/DIMENSION; reserved `WITH`/`ON`). The
   DDL never lowers to a `LogicalPlan` — `Query::as_create_vector_index()`
   lets the server **intercept it pre-plan** (like Bolt's `try_introspect`),
   so it needs no third dispatch branch nor a new `LogicalPlan` variant
   (avoids the C5 tax of ~12 exhaustive matches). Wired in all four
   read/write chokepoints: `run_cypher` + `run_cypher_multi` (HTTP),
   `run_query` (Bolt auto-commit, `StatementType::Schema`), and
   `run_query_in_tx` (Bolt, rejects DDL — it commits immediately and can't
   roll back). Feature-gated: feature off → DDL reaches the lowerer and is
   rejected (HTTP 400 / Bolt NotSupported); feature on → registers + the
   compaction hook builds + the optimizer accelerates. Read-only tokens are
   forbidden (DDL mutates durable schema). HTTP test: success (200) +
   duplicate (400) + read-only (403).

   Design note: the DDL is out-of-band, NOT a query, so it deliberately
   skips `contains_write()` / the writer's `execute_write` row path (which
   would stage an empty memtable batch). The server builds the
   `VectorIndexDescriptor` (Vamana defaults R=64/L=128/α=1.2, overridable
   via `WITH`) and republishes the snapshot; `catalog_for` rebuilds on the
   version bump so the next query's optimizer sees the new index.

## Completed earlier this session

### Adversarial-review fixes (68 agents → 41 confirmed bugs)
**Commit `b347038`** + follow-ups. The review found 4 critical / 15 high bugs; the
criticals and most highs are fixed:

- **CRITICAL s3b data loss on EC-LIST stores** — once the janitor reclaims `p0`
  below the horizon, a stale empty LIST made the pointer family look empty,
  `load_current` fell through to a `current.json` post-RFC commits never wrote,
  and `WriterSession::open` **re-bootstrapped a live namespace** (silent data
  loss on exactly the EC stores RFC-029 targets). **Fix:** publish an advisory
  `current.json` (plain PUT, universally supported) on every commit/bootstrap;
  `max_pointer_version` recovers the version via that advisory when LIST is
  empty AND p0 is gone. +regression test.
- **CRITICAL multi-tenant idle eviction never fired** — `last_access` stored
  `Instant::now().elapsed()` (always ~0). Fixed to anchor-relative seconds.
- **CRITICAL `max_namespaces=0` (documented "unlimited") failed the first
  open.** Fixed: 0 = no cap.
- **HIGH RRF off-by-one** — denominator had a spurious `+1` vs standard
  `1/(k+rank)`. Fixed.
- **HIGH hybrid channels sequential** → now `tokio::join`'d concurrently.
- **HIGH lexical channel no ORDER BY** → ranks were non-deterministic. Fixed.
- **HIGH PageRank mass conservation** — weighted edges normalized by count not
  weight-sum → scores diverged from 1.0. Fixed (normalize by Σw) + test.
- **HIGH graph_algorithm included placeholder stubs** → filtered.

### Multi-tenant (item 14) — production-hardened
- `07271b9` per-namespace maintenance tasks (flush/compaction/orphan-sweep) —
  closed the data-loss-on-crash / unbounded-read-amplification gap.
- `9a76870` X-NamiDB-Namespace header + default-namespace fallback (were
  documented but unimplemented) + namespace-isolation test.
- `7fc9500` per-namespace catalog cache (was rebuilt every multi-tenant query).

### Item 11 — typed "unsupported" errors (SHIPPED, `029fb84`)
EvalErrorKind{Generic,Unsupported} → ExecError::is_unsupported() → HTTP 400
`code:"unsupported"` + Bolt BackendError::Unsupported. +test.

### Item 16 — backup/restore maturity (SHIPPED safe subset, `fbc44cf`)
No-clobber is now a real `PutMode::Create` CAS (closes the TOCTOU); `--verify`
re-heads every referenced object; advisory pointer published on restore.

### Item 13 — hybrid search (Layer A SHIPPED, `362bdac` + fixes)
RRF fusion of lexical + semantic channels, concurrent, configurable weights.

### Item 08 — graph algorithms (MCP wedge SHIPPED, `45a15c7` + fixes)
WCC (union-find) + PageRank (power iteration, mass-conserving) in namidb-graph;
`graph_algorithm` MCP tool.

### Earlier this branch
- s3b versioned pointer RFC-029 (`e5d48a7`); multi-tenant foundation (`5415123`).

## Remaining work (with verified design plans)

The 26-agent design workflow produced file:line plans for the remaining items.
They are L-effort and need their own focused sessions. **Designs live in the
workflow output** (`/tmp/.../tasks/wu89feu6l.output`, run ID
`wf_e869ca77-7c5`); key decisions:

### Item 12 — ANN HNSW (L) — design winner: **DiskANN/Vamana** (judge score 97)
Beat HNSW-on-CSR (66) and IVF-Flat-PQ (65). Synthesized plan: object-store-native
Vamana graph as a new `SstKind::VectorGraph`, int8 short-vectors in RAM reusing
`quantize.rs` (asymmetric int8 scorer added there FIRST — shared by scorer +
bench), FreshDiskANN-style L0 delta covered by the existing flat-scan, full-
precision rerank, built during the compaction sweep (no new background task),
`CREATE VECTOR INDEX` DDL, optimizer rewrite to `LogicalPlan::VectorSearch`
falling through to the flat path when no descriptor, gated behind Cargo feature
`vector-index` + env `NAMIDB_VECTOR_INDEX` (default off), recall@k validated via
`namidb-bench/vector_recall`. **Step 1 (load-bearing, unblocks all): add
`dot_i8_asymmetric`/`norm_i8` to `quantize.rs` and rewire the bench's private
copy at `vector_recall.rs:96-100`.**

### Item 15 — auth/RBAC (L) — ship-with-fixes
Wave A OIDC/JWT (jsonwebtoken + JWKS, pinned-alg, group-claim→role, thread a
Principal on HTTP+Bolt) + Wave B AuthzHook PDP (OPA/Cedar). **Design holes to
address:** Role is `Copy` (7 by-value sites) — wrapping in `Arc<str>` loses Copy,
needs `.clone()` at pass-sites; Bolt `Authenticator` trait has no Principal
return channel; `serve()`/`make_policy` signatures change; reqwest must be a
feature-gated dep (not unconditional); default fail-closed for PDP. Gate behind
feature `jwt`.

### Item 08 PR1 — CALL/YIELD (SHIPPED this session)
`CALL <ns>.<name>() [YIELD col [AS a], …]` → `LogicalPlan::CallProcedure`
source leaf → executor runs `algo.wcc` / `algo.pagerank` over the full
snapshot. **Working end-to-end**: a 2-pairs-plus-isolate graph yields 3 WCC
components (isolate kept) and PageRank scores summing to ~1.0; unknown
procedures surface as a typed `unsupported` error. CALL flows through the
server's existing read path (`contains_write()` is false for it), so no
server changes were needed. +3 e2e tests (`tests/exec_call.rs`), +4 parse tests.

What shipped: `Clause::Call`/`CallClause`/`YieldItem` AST (CALL/YIELD soft
keywords — non-reserved, like EXPLAIN); `parse_call_clause` (qualified names,
optional parens/args, optional YIELD); `LogicalPlan::CallProcedure` leaf +
the C5 arms (cardinality, explain, walker flat+factor, writer delegate,
collect_produced, collect_plan_referenced_variables, the optimizer leaf
groups); the `flat_call_procedure` executor + `snapshot_to_algo_graph` bridge
(nodes via `observed_labels`+`scan_label` so isolates are kept; edges via
`observed_edge_types`+`scan_edge_type` so weights are carried).

The four prior "blocking holes", resolved against code (3-agent explore):

- **cfg/exhaustive-match** → CALL is **in-band, always-on** (not feature-gated,
  not out-of-band): `Clause::Call` + `LogicalPlan::CallProcedure` are
  always-present variants; the lower arm produces the leaf, the executor runs
  it. Same shape as `VectorSearch`. Graph algos are core, not experimental, so
  no Cargo gate.
- **isolates dropped** → the kernel handles isolates correctly
  (`weakly_connected_components` enumerates `graph.nodes()`); the fix is the
  *caller* must `add_node` for every node, not only `add_edge`. The
  Snapshot→Graph bridge does `observed_labels()` → `scan_label(l)` →
  `add_node(id)` per node (covers isolates), then `observed_edge_types()` →
  `scan_edge_type(t)` → `add_edge(src,dst,weight)`.
- **`neighbours_of` doesn't exist** → confirmed. The bridge uses
  `scan_edge_type` (full properties incl. weights, unlike CSR `out_edges`).
- **soft-keyword** → CALL/YIELD stay non-reserved (`Token::Ident`, matched
  case-insensitively like EXPLAIN/VECTOR). Add a `Some(Token::Ident(n)) if n
  eq "CALL"` arm in `parse_clause` before the fall-through error.

**API ground truth** (all `namidb-graph/src/algo.rs`):
- `Graph { add_node, add_edge(src,dst,Option<f64>), nodes, out_edges }`
- `weakly_connected_components(&Graph) -> Components { assignment: HashMap<NodeId,usize>, count }`
- `pagerank(&Graph, &PageRankOptions) -> PageRank { scores: HashMap<NodeId,f64>, iterations, converged }`
- `PageRankOptions { damping:0.85, max_iterations:100, tolerance:1e-6 }` (`Default`)
- Both kernels are **sync `fn`** taking `&Graph` — the caller builds the Graph.
  No existing Snapshot→Graph bridge (namidb-graph lists namidb-storage as a dep
  but doesn't use it); **write `Graph::from_snapshot(&Snapshot<'_>)` async
  helper in namidb-graph** (nodes via scan_label, edges via scan_edge_type).

**Executor integration** (`exec/walker.rs`): model on `VectorSearch`/`EdgeTypeCount`
— a source leaf returns `Vec<Row>` directly, each `Row::new().with(alias,
RuntimeValue)`. Column names = binding keys (the server derives columns from
`rows.first().bindings.keys()`, `lib.rs:1649`). Has `snapshot` + `params` in
scope. C5 tax: add the arm in walker flat (`:227`) + factor (`:2075`, wrap via
`FactorRowSet::from_flat`), `execute_capped` (`:821`), writer delegate group
(`:482`), `cardinality` (`:65`), `explain` (`:356`, no catch-all — required),
`collect_plan_referenced_variables` (`:3657`), + the optimizer leaf groups
(same set VectorSearch touched).

**MVP scope:** `CALL <ns>.<name>([args]) [YIELD col [AS a], …]`; CALL must be
the leading source clause; dispatch `algo.wcc` (YIELD `node_id, component`) and
`algo.pagerank` (YIELD `node_id, score`; optional map arg for
`{damping,max_iterations,tolerance}`). Deadline: the kernels are sync so the
per-operator `check_deadline` (`walker.rs:219`) fires only around the call —
documented limitation (a runaway kernel isn't interruptible mid-iteration;
follow-up: thread `cancel::deadline_exceeded()` into the kernel loops).

### Deferred bug follow-ups (lower severity, documented)
- s3b forward-probe `MAX_PROBE` gap-safety after GC (commented as best-effort;
  under sustained >8192-version write lag a stale pointer can be served).
- s3b bootstrap crash-atomicity (crash between v0.json and p0.json wedges).
- multi-tenant auth is global, not per-namespace (any valid token reaches any
  namespace) — needs per-namespace token scoping (Wave B of item 15).
- hybrid-search bm25 builtin (Layer B) for real lexical relevance.

## Commands

```bash
cargo test --workspace          # all green
cargo clippy --workspace -- -D warnings
git log --oneline -12           # this session's commits
```

---
Generated: 2026-06-19 (post-ultracode: 5 roadmap items shipped + 41-bug review fixed).
