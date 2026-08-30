# 25 TB readiness: functional coverage plan

Produced from a five-dimension audit of the functional surface (vector, full-text,
graph, query core, operational envelope) against the test corpus at branch
`harden/object-native-2.1.0`. A surface item counts as covered only when a test
asserts its semantics on the route that will actually serve production traffic.

## Area summaries

### FULL-TEXT SEARCH (search.bm25, search.hybrid sparse leg, bm25() scalar, FULLTEXT DDL, FT4 native / l

*20 surface items, 16 gaps, 2 blockers.*

Full-text search in NamiDB 2.1.0 has genuinely deep coverage at three layers: shared primitives (tokenizer incl. adversarial Unicode differential, parse_query phrase/prefix/CJK, BM25 math), the legacy .ft format (full differential vs flat scan, corruption fail-closed, range-reader parity), and the FT4 segment format (signed delta stats, tombstones, block-max exactness, external bounded-memory builds). Query-level tests are strong where it matters most: index-vs-flat rank+score parity for phrases/prefixes with the route asserted at the storage boundary, canonical property order across route flips, the label-scoped memtable-dirty gate (unrelated-label writes keep the index; relabel of an indexed doc forces flat), filter-before-top-k, and a thorough 5-test suite for the walker-level filtered over-fetch/widening loop with the candidate cap. The reconciled global-stats (delta_df) design is proven bit-identical across three FT4 segments and through physical delta-run merges. DDL create/drop/recreate is covered over HTTP and embedded py. The critical holes for a 25TB load: (1) the OTHER refill loop — the native coordinator's bounded segment_limit widening in search_lsm_read — is never driven past its first iteration, exactly the path that dominates an update-heavy corpus where reconciliation suppresses stale winners (silent short top-k risk); (2) no end-to-end Cypher DELETE → flush → native-route search test proving corpus-stat shrinkage; (3) empty/edge query inputs, k=0/unbounded, and >64-term prefix expansion displacement are untested at the procedure level; (4) a commit-documented known residual (lost barrier + downgrade marker suppresses rebuild → index unrecoverable) has no pinning test; (5) no concurrency test runs bm25 against live flush/compaction; and (6) the full-stack py/HTTP tests deliberately accept either route, so total loss of native serving (every query O(corpus)) would pass CI. Hybrid fusion math is unit-solid but its tuning knobs (rrf_k, k_dense/k_sparse, fusion-name errors) and any non-embedded transport (HTTP/Bolt) are untested.

### VECTOR SEARCH — full functional surface: CALL search.vector / db.index.vector.queryNodes / search.hy

*13 surface items, 14 gaps, 3 blockers.*

Vector search coverage on harden/object-native-2.1.0 is strong on the V5-base index route and the flat fallback: the KNN rewrite (all lowered shapes, guards, and plan-shape reachability assertions), filter semantics (native eq/IN postings, residual widening, exhaustion/short-page proofs with flat-scan canaries), all three metrics via Cypher, int8 exact-rescore, zero-norm both directions, write-time dim enforcement, and full DDL lifecycle are all asserted with real semantics (1436-line exec_vector_knn.rs plus format-level suites for v4/v5/v6 and the Search-LSM coordinator/invariant matrix). The material holes for a 25 TB production load are concentrated in three blockers — NaN/non-finite inputs (silently downgrade to an O(corpus) flat scan with NaN ordering, and an inconsistency between vector_fresh_delta and the flush classifier), relabels never tested end-to-end on the vector route, and zero reader-vs-writer concurrency tests (flush/compaction/janitor during a query) — plus high-severity route gaps: the ActiveSegments (VG6 delta) route is never asserted at the query layer (the exact reachability trap previously documented for this codebase), the capped-sidecar filtered fallback chain is untested at every layer, unknown procedure-map keys (including a typoed filter) are silently ignored, and queryNodes/search.vector metric variants beyond cosine are untested on the procedure forms. The hybrid dense leg has only ever run without an index. Everything else outstanding is medium/low (k=0 edges, query-dim-mismatch error, arg-error messages, legacy migration-window route at exec level, Vector8 params).

### GRAPH — MATCH patterns, writes (CREATE/MERGE/SET/REMOVE/DELETE), edge properties, parallel edges, su

*23 surface items, 14 gaps, 4 blockers.*

Graph surface enumerated from parser/AST (ast.rs), logical operators (logical.rs: NodeScan/NodeById/NodeByPropertyValue/Expand/Merge/Create/Set/Remove/Delete/Foreach/MultiwayJoin/EdgeTypeCount/CallProcedure), the algo.* dispatch (walker.rs:1712-1975), and storage APIs (read.rs, ingest.rs, adjacency.rs, sst/edges/*). Coverage is genuinely strong in three places: the write path (exec_writes.rs' 100+ tests pin MERGE/SET/REMOVE/DELETE semantics including correlated batches, duplicate-key batches, RYOW, rollback and auto-commit transactionality), storage edge primitives (point lookups across memtable/staged/paged/legacy routes with bytes-envelope and corruption fail-closed tests), and algo.* (10 procedures with kernel unit suites, projections, induced-subgraph rules and error paths). The dominant structural weakness is route asymmetry: every rich traversal semantic — var-length, shortest/allShortestPaths, alternation, WCOJ, path bindings — is exec-tested only against the unflushed memtable route, while the paged-SST route that will serve essentially all 25 TB of data is exercised only through low-level primitives; additionally the default partner-list route still hydrates full edge SST bodies for properties with no envelope test, and count/scan_edge_type are O(corpus)-resident. Correctness-envelope holes: bare DELETE of a node with edges silently creates dangling edges (and pushdown-vs-expand count divergence), same-type parallel-edge CREATE is a silent last-write-wins upsert (the audit's open question — resolved in the storage model, unpinned by any test), var-length rel aliases bind the last hop instead of a list, and edge-bucket tombstone GC at the deepest merge has no test despite the node-side invariant being well covered.

### QUERY CORE & FILTERS (parser/plan/optimize/exec in crates/namidb-query, storage read/scan API, serve

*33 surface items, 18 gaps, 2 blockers.*

Query-core coverage is strong exactly where the 2.x hardening work happened and thin where nobody has looked since v0. Excellent, route-aware coverage (plan-asserts-the-index + memtable/SST/staged/flat routes all exercised): WHERE equality-to-index routing incl. the AND-conjunct/residual selection, ordered-prefix index-backed ORDER BY with memtable-delta union, SKIP/LIMIT pushdown incl. param bounds, DISTINCT-before-LIMIT, count(*) O(1) fast paths with authority gating, write atomicity/rollback/auto-commit transactionality, WAL replay/crash/fencing, deadlines and row/memory caps, and Bolt streaming (PULL n/has_more/DISCARD-all/RESET, BEGIN/COMMIT/ROLLBACK). The two 25TB blockers: (1) non-DETACH DELETE of a connected node silently orphans edges with zero tests of subsequent read semantics on any route (Neo4j errors here); (2) the aggregate family beyond count — avg/min/max/collect/sum(DISTINCT) — has essentially no semantic tests (empty groups, null-skipping, mixed types, i64 sum overflow), with LDBC fixtures that use them running as execute-only smokes. High-severity holes behind those: STARTS WITH/ENDS WITH/CONTAINS never evaluated in any test, the simple CASE form untested, top-level UNION lacking both an e2e and any column-name validation, the keyset pagination API never executed against data (and unwired), and the HTTP JSON parameter route untested. Most remaining gaps are variant/error-path level (UNWIND edge inputs, WITH SKIP/LIMIT, ~10 untested scalar functions, IN/IS NULL pushdown execution parity, HTTP-vs-Bolt encoding parity, EXPLAIN-over-HTTP).

### Operational envelope for 25 TB (scale fixtures, backup/restore, janitor, upgrade/downgrade, crash re

*9 surface items, 13 gaps, 4 blockers.*

The operational envelope is the weakest layer of an otherwise well-tested engine. Solid: writer fencing (unit-complete incl. CAS loss, poison, retirement), generic crash/interrupted-commit recovery, janitor orphan/pin/WAL sweeps, backup unit tests covering every legacy sidecar including .npp, a genuinely semantic object-native CI gate (256 vectors, oracle-anchored, anti-tautology jq contract), and a thorough memory-governor state machine. The blockers for a 25 TB load: (1) nothing end-to-end runs above 256 rows — the one #[ignore] soak is builder-only and no CI/nightly executes it, while the project's own search-lsm.md mandates a long soak and 10M benchmark that don't exist; (2) backup/restore has never been tested over an Active Search-LSM generation, so restore-degrades-to-flat-scan would ship unnoticed; (3) the documented crash matrix for search publishes (fail after segment/barrier/compaction-output PUT) does not exist; (4) the known 22d392c residual — text base with lost barrier stranded between an adoption that can't succeed and a marker that suppresses rebuild — is unfixed and untested. High-severity: no frozen 2.0.6 wire decoder test (downgrade is simulated by struct mutation), no text-kind adoption twin, janitor never tested against active search state, memory ceiling proven only with synthetic RSS, and every S3/R2 behavior (multipart, ranged reads, retry, TLS) tested only against InMemory — the lone LocalStack test covers manifest CAS and runs nowhere. Concrete fixes: a nightly --ignored workflow + 1M-row search soak, a search-generation backup round-trip test, a FaultStore crash matrix, manifest_rollback_206.rs, the text wrap-failure->rebuild fallback with its test, a LocalStack CI job with a full ingest->search->backup cycle, gate growth to 2048 vectors, and a pre-load manual runbook (run-bench-r2.sh + LoadR2 SF1 + container memory-limit smoke) against the customer's real bucket.

## Prioritized work plan

### 1. [DONE] Every rich traversal semantic (var-length min..max, relationship-uniqueness, shortestPath/allShortestPaths, alternation, OPTIONAL var-length, WCOJ/multiway-join, back-reference expands) is exec-tested only on the unflushed memtable route; the paged-SST edge route that will serve essentially all 25 TB is exercised only via storage primitives. [Report 3; broadest single risk — the serving route for the entire dataset is unproven at the operator level.]

Add a route-matrix helper (e.g. fn assert_on_all_routes(build, asserts)) usable from crates/namidb-query/tests/exec_match_expand.rs, exec_shortest_path.rs, exec_alternation.rs, exec_multiway_join*.rs. Fixture: reuse build_friend_graph / exec_shortest_path::build_graph / the alternation and multiway builders, then materialize four states: (a) memtable-only (current baseline), (b) flush(schema) → pure SST, (c) flush then re-apply an upsert + a tombstone so a memtable delta overlays the SST, (d) staged-flush overlay. Assertions: for each existing key test, the full row set (including path bindings and lengths for shortestPath/allShortestPaths, dedup behavior for relationship-uniqueness, [:A|:B] alternation rows, OPTIONAL var-length null rows, multiway-join row multiset) is identical across all four states. Any divergence is a serving-route bug.

**Resolution (2026-08-15):** `crates/namidb-query/tests/exec_route_matrix.rs` runs twelve traversal shapes (directed/inverse/undirected expand, var-length directed 1..3 and undirected exact-2, path-binding lengths, back-reference 2-cycle, WCOJ triangle, [:KNOWS|:LIKES] alternation, OPTIONAL var-length with asserted null rows, shortestPath and allShortestPaths hop counts) against four physical states of one logical graph — memtable-only, pure SST, SST + same-batch tombstone/re-upsert overlay, staged half-flush — and requires identical canonical row multisets, with a non-vacuity guard on every baseline. All twelve were already route-invariant when the matrix landed (consistent with the item-14 ranged-hydration fix landing first).

### 2. [blocker-for-25TB] Bare (non-DETACH) DELETE of a node with incident edges silently tombstones the node and commits dangling edges (writer.rs apply_delete has no incident-edge check; Neo4j errors). Also breaks route parity: EdgeTypeCount pushdown counts dangling edges while NodeScan+Expand drops them via target confirmation. [Reports 3+4, identical finding — deduplicated.]

First decide the contract (recommend: typed error, Neo4j-compatible). Test in crates/namidb-query/tests/exec_writes.rs: fixture CREATE (a:L {k:1})-[:R]->(b:L {k:2}); run MATCH (a:L {k:1}) DELETE a. Assert either the typed 'node still has relationships' ExecError, or — if dangling-by-design — pin it: (i) MATCH (x)-[r:R]->(y) RETURN count(r) returns the same value on the EdgeTypeCount-pushdown plan and the raw NodeScan+Expand plan (extend exec_match_expand.rs::global_edge_type_count_pushdown_matches_nodescan_path with a node-tombstone-with-surviving-edges case), (ii) expand from b inbound yields zero rows, (iii) repeat both assertions after flush + compaction so the SST route is pinned too.

### 3. [blocker-for-25TB] Zero tests run any search (vector or text) concurrently with flush/compaction/janitor — the steady-state operating mode at 25 TB. Snapshot pinning + pin leases are relied on by argument only; the vector widening loop re-reads index objects across await points while compaction CAS + sweep can replace/delete generation objects mid-query. [Merges Report 1 blocker (vector) + Report 2 high (bm25 concurrency) into one framework.]

New crates/namidb-storage/tests/search_concurrency.rs. Fixture: one label with both a vector index and a text index, ~2k docs. Writer task loops {upsert batch with changed embeddings/text, tombstone a few, flush, compact_l0, run janitor sweep_orphans}. Reader tasks (4+) each: acquire a Snapshot, compute the exact answer from that snapshot (brute-force top-k for vector; flat bm25 scan for text), then run search.vector / Cypher KNN / search.bm25 and assert exact equality and that no query errors with NotFound/Unavailable-as-error. Add two orchestrated interleavings with explicit sync points: (a) janitor sweep between snapshot acquisition and the search call; (b) search compaction retires the delta generation mid-query while the reader holds a pin lease — old segments must still be readable until release.

### 4. [blocker-for-25TB] Native FT4 multi-segment refill loop (search_lsm_read.rs:1671-1735) never driven past its first iteration — the winner-reconciliation-suppresses-stale-candidates case that dominates an update-heavy 25 TB corpus. Both exits (widen x4 to segment_ceiling; Unavailable at ceiling) and the silent-short-top-k failure mode unproven. [Report 2.]

Inline tests in crates/namidb-storage/src/search_lsm_read.rs: extend text_fixture with a delta segment holding ~40 docs matching a term where the top TEXT_SEGMENT_OVERFETCH*k candidates are superseded by a later segment (higher-LSN updates that removed the term); query with small k. Assert (a) the returned ids are exactly the true survivors — provable only if a second widening round ran; (b) with segment_ceiling forced low (env/test hook), the coordinator returns Unavailable so the Snapshot layer falls back to flat, never a silently short page; (c) score parity vs a flat scan of the reconciled corpus.

### 5. [blocker-for-25TB] No end-to-end Cypher DELETE → flush → native FT4 route test: nothing proves a deleted doc is gone AND scores reflect shrunken corpus stats (delta_df -1) on the native route. Tombstone suppression is covered only in hand-built storage fixtures. [Report 2.]

crates/namidb-query/tests/exec_call.rs: register FULLTEXT index, write ~20 docs, flush+compact until the store's text_search returns Some (route asserted at the storage boundary, not inferred from results). Then: (1) DELETE one doc via Cypher pre-flush — assert the dirty overlay excludes it from search.bm25; (2) flush again — assert the native route (text_search still Some) excludes it and every surviving doc's bm25 score equals the flat scan of the surviving corpus to 1e-12 (corpus-stat shrinkage, not just absence).

### 6. [blocker-for-25TB] Relabel is untested end-to-end on the vector route: no test that REMOVE n:Doc suppresses a stale hit the persisted .vg still serves (memtable overlay), nor that a flushed VG6 delta does the same, nor that re-adding the label restores it. Only isolated unit arms exist. [Report 1.]

crates/namidb-query/tests/exec_vector_knn.rs (indexed mod): build index over :Doc, identify the top hit for a probe query. (a) MATCH … REMOVE d:Doc via the Cypher writer; assert it vanishes from both Cypher KNN and search.vector while the .vg demonstrably still contains it (memtable-dirty overlay route). (b) Flush the relabel; re-assert on the VG6-delta route. (c) SET d:Doc again; assert it returns. (d) Multilabel variant: node keeps another label — assert suppression is per-index-label, not per-node.

### 7. [blocker-for-25TB] NaN/non-finite vectors untested end-to-end: a NaN query on the V5 route raises Error::Invariant which optional_accelerator_fallback silently converts to an O(corpus) flat scan with NaN total_cmp ordering; a stored non-finite embedding ranks FIRST on the flat route (sort key -NaN); vector_fresh_delta merges non-finite fresh embeddings while the flush classifier (search_lsm_flush.rs:508) suppresses them — an internal inconsistency. [Report 1.]

crates/namidb-query/tests/exec_vector_knn.rs (indexed mod): (1) query with one NaN component against a built index — assert a defined outcome (recommend: typed error) identical on indexed and flat routes, AND assert the route taken so the silent O(corpus) downgrade is pinned; (2) store a NaN-component embedding via WriterSession, assert KNN drops it on both routes (mirror the existing zero-vector pair index_stored_zero_vector_absent_matches_flat / index_zero_query_returns_empty_matches_flat); (3) storage unit: vector_fresh_delta returns None (suppress) for non-finite or wrong-dim fresh embeddings, matching the flush classifier.

### 8. [MOSTLY DONE — see #35 for the open remainder] Nothing end-to-end runs above ~256 rows on any route; the 100k soak is builder-only, executed by no CI/nightly; the doc-mandated long update soak and 10M benchmark do not exist; no CI job even compiles namidb-bench. [Report 5; confirmed by the 2026-07-28 audit memory.]

(a) New .github/workflows/nightly.yml (schedule:) running cargo test --workspace -- --ignored so soak_100k_rows_1024d_is_disk_spooled actually executes, plus a cargo build -p namidb-bench step. (b) New #[ignore] soak in crates/namidb-storage/tests/: ingest >=1M nodes with vector+text indexes through repeated flush+compact cycles (forcing multi-SST, multi-level, multi-delta-segment state); assert native search parity vs flat scan on sampled queries, bounded RSS (test hook), bounded segment count. (c) Grow the object-native CI gate to --vectors 2048 --dim 64 --page-rows 32 and tighten --max-cold-bytes-ratio below 1.0 per the workflow's own calibration note. (d) Document a pre-load manual runbook: run-bench-r2.sh full + LoadR2 LDBC SF1 against the customer's real bucket.

**Resolution (2026-08-15):** (a) `.github/workflows/nightly.yml` runs the ignored builder soaks and the new lifecycle soak on a schedule. (b) `crates/namidb-storage/tests/soak_search_lsm.rs`: `NAMIDB_SOAK_ROWS` documents (default 1M) through 20 flushes with 1%/0.5% update/delete churn and periodic compactions; structural oracles (unique token per doc, self-embedding top-1), native-route serving asserted, physical fan-out bound, linux VmHWM ceiling. Validated locally at 20k (20s), 100k (41s) and the full 1M (288s, peak RSS 3.03 GiB). (d) `docs/testing/preload-runbook.md`. (c) was attempted and produced engine findings instead of a bigger CI gate — recorded as #35; the PR gate stays at 256 vectors with the 2.0 cold-ratio ceiling.

### 9. [blocker-for-25TB] Backup/restore never tested over an Active Search-LSM generation: no proof the .slb barrier, VG6/FT4 base, delta segments and search_lsm manifest survive copy_namespace_snapshot, nor that the RESTORED namespace serves the native route rather than silently degrading to flat scan forever. [Report 5.]

crates/namidb-storage/src/backup.rs test snapshot_round_trips_active_search_generations: build vector+text corpora to Active generations (reuse compact.rs fixtures), copy_namespace_snapshot(verify=true). On the destination assert: validate_search_barrier passes; select_search_read_plan returns the native route (assert the SELECTION, not just equal results — the index-reachability lesson); delta segments and coverage metadata intact; queries match source. Add a verify_snapshot negative case that detects a deliberately deleted barrier object.

### 10. [blocker-for-25TB] No crash matrix for search publishes: fault injection after search-segment PUT, barrier PUT, manifest body, and search-compaction output PUT (mandated by docs/architecture/search-lsm.md) does not exist; 'no visible half-generation' between barrier PUT and pointer CAS is unproven. [Report 5.]

Promote the FaultStore from ingest.rs tests into crates/namidb-storage/src/test_support.rs with fail_next_put_containing(substr). Matrix tests in ingest.rs/compact.rs, one per fault point {delta segment object, .slb barrier, manifest body, search-compaction output}: after the injected failure assert (1) a fresh reader's plan selection either serves the old generation or falls back to flat — never a half-generation or an error; (2) retrying the publish / reopening the store is idempotent and completes; (3) the janitor orphan sweep reclaims the partial objects; (4) post-recovery search results equal flat scan.

### 11. [DONE] Known residual from commit 22d392c, unfixed and untested: a text base with a lost .slb barrier cannot be adopted (probe accepts only NAMIFT03) while the minted downgrade-interop marker suppresses rebuild — text search permanently stranded on flat scan with no signal. Requires an engineering fix (wrap-failure → rebuild fallback) plus a pinning test. [Reports 2+5, same finding — deduplicated.]

Implement the fallback first. Then crates/namidb-storage/src/compact.rs test text_base_with_lost_barrier_falls_back_to_rebuild: build an FT4 Active generation + interop marker, delete the .slb barrier and wipe search_lsm state, run the maintenance/compaction pass. Assert: a rebuild is scheduled despite the marker; a fresh Active generation eventually serves via the native FT4 route (assert selection); queries are flat-correct throughout the window (never an error, never silently stale). Until the fix lands, a pinning test asserting today's flat-correct-but-stranded behavior so the state is at least visible.

**Resolution:** the stall had a second cause the plan did not know about: BasePrefix consolidations minted their interop markers with the NATIVE `text_lsm_catalog_signature`, which neither `adoption_catalog_signatures` nor `catalog_build_states` accepts — so for text the marker never certified anything (no adoption, no suppression consistency). Fixed by minting with the catalog-derived signatures the 2.0.6 paths compare against. On top of that, the unadoptable-magic fallback landed: when the adoption probe deterministically rejects a body (unsupported magic, unsafe wrap), the disproven marker is dropped at install (`unadoptable_search_markers` on `PreparedCompaction`; both install early-outs now treat a marker drop as a real change), which un-suppresses the full rebuild on the next pass. Transient read errors keep the marker and retry. Pinned by `compact.rs::text_base_with_lost_barrier_falls_back_to_rebuild` (FT4 base + minted marker, barrier lost, state wiped → pass 1 drops the marker, pass 2 rebuilds and `text_search` serves natively).

### 12. [blocker-for-25TB] Same-type parallel-edge semantics resolved in the storage model as (edge_type,src,dst) last-write-wins upsert but pinned by no test or doc: a second CREATE silently overwrites properties yet still increments edges_created. A 25 TB loader assuming multigraph semantics would corrupt silently. [Report 3.]

crates/namidb-query/tests/exec_writes.rs: CREATE the same (type,src,dst) twice with different property maps, once within a single statement and once across two statements/transactions. Assert: exactly one edge survives on MATCH (both pre- and post-flush); the LAST write's property map wins; decide and pin the edges_created counter contract (currently 2 for 1 surviving edge — either fix to 1 or document and assert 2). Add a doc note in the CREATE section stating the upsert contract.

### 13. [blocker-for-25TB] Aggregates avg/min/max/collect/sum(DISTINCT) have zero semantic tests at the Cypher level: empty groups (avg→NULL), null-skipping, mixed-type min/max ordering, DISTINCT dedup inside aggregates, i64 sum overflow (unchecked += in walker.rs ~5940). LDBC fixtures using them run execute-only. [Report 4.]

New crates/namidb-query/tests/exec_aggregates.rs. Fixture: one label with rows covering {ints, floats, nulls, duplicates}, one empty label. Grouped and global forms of each of count/sum/avg/min/max/collect, with and without DISTINCT. Assert exact RuntimeValue results: avg over empty group → Null; min/max over all-null → Null; global aggregate over zero rows returns one row (count=0, sum=0-or-Null — decide and pin); nulls skipped inside groups; mixed int/float min/max ordering; collect(DISTINCT) dedup with order contract; sum at i64::MAX + 1 → pinned behavior (error or promotion, not silent wrap). Run the matrix pre- and post-flush.

### 14. [DONE] Traversal partner-list route (edge_lookup_via_sst — the production default) hydrates the FULL edge SST body via a whole-body GET on every cold lookup even in Topology mode (read.rs:6041-6047): an O(edge-SST) read per hop at 25 TB. Both a test gap and an engineering gap. [Report 3.]

crates/namidb-storage/src/read.rs tests: build a multi-MB edge SST (many edge types / large property payloads), wrap the ObjectStore with the byte-counting instrumentation used by exact_partner_reads_only_the_winning_metadata_row, call out_edges/in_edges in Topology and SparseIdentity modes. Assert bytes fetched << body size (e.g. <5%). This test FAILS today — land it #[ignore]-annotated with the tracking issue, fix the ranged-read hydration, then de-ignore. Companion assertion for count_edge_type (see rank 20).

**Resolution (2026-08-15):** `edge_lookup_via_sst` and the exact point lookup now hydrate properties through `PagedEdgeReader::read_property_rows` over exactly the key's row range (legacy Arrow sections keep the eager fallback inside that call); the dead whole-body `fetch_edge_reader`/`fetch_edge_streams` pair was removed from those paths. Pinned by `tests/edge_partner_bytes.rs`: a 16 MB incompressible-property edge SST where a cold out_edges/in_edges lookup must stay under 5% of the body (pre-fix it read 16.3 MB — the whole body and change), plus property-parity, warm<cold, and both directions. Cold cost is dominated by 64 KiB verification pages and index reads (~500 KB fixed), a sliver at production SST sizes. The corrupt-sidecar cache test was re-anchored: the eager whole-body caches must now stay untouched.

### 15. [DONE] The ActiveSegments route (V5 base + VG6 delta, coordinator search_lsm_read.rs:983) is never ASSERTED at the query layer — exec tests would pass identically on fallback (the documented reachability trap); and the two feature-gated compaction tests still assert the obsolete single-body model. [Report 1.]

**Resolution (2026-08-15):** the two compaction tests were re-anchored to the incremental model in an earlier round (force-base env guard). The query-layer witness now exists: `exec_vector_knn.rs::active_route_witness` counts `.slb` barrier HEADs through a store probe — a clean snapshot must pin (zero pins = flat fallback answered = the parity proved nothing), and the dirty-memtable half asserts freshness through a document the persisted index has never seen (the coordinator legitimately pins while deciding, so the pin count is not the dirty-route witness). Text/hybrid had equivalent pins from the earlier hybrid work.

crates/namidb-query/tests/exec_vector_knn.rs: after build_index + a post-compaction flush producing a VG6 delta, assert select_search_read_plan(...) is ActiveSegments (or prove the VG6 segment object is read via a GET-counting ObjectStore probe like exec_hybrid_search's TextGetProbe). On that pinned state assert update/delete/filter/k>live-count semantics through both Cypher KNN and search.vector. Separately: rewrite compaction_builds_a_searchable_vector_graph (and its twin) to the multi-segment model using NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION under the env mutex.

### 16. [DONE] Unknown keys in search.vector/search.hybrid single-map args are silently ignored (proc_single_map, walker.rs:2968): a typoed 'filtre'/'Filter' returns UNFILTERED results — a data-exposure/correctness hazard, not a convenience. [Report 1.]

**Resolution (2026-08-15):** `proc_single_map` takes the procedure's allowed-key list and rejects any unknown key with an error naming it and the allowed set; wired for search.vector (7 keys) and search.hybrid (15 keys). Pinned by `exec_call.rs::unknown_procedure_map_keys_error_instead_of_running_unfiltered`.

Implement key validation, then in crates/namidb-query/tests/exec_hybrid_search.rs assert CALL search.vector({label…, query…, filtre:{…}}) errors with an unknown-option message listing valid keys; same for search.hybrid, search.bm25, and the queryNodes 4th map.

### 17. [DONE] Keyset pagination (pagination.rs v2) has zero consumers and zero executions against data — the exact deep-pagination workload 25 TB will hit; duplicate-free/gap-free pages across flushes, deletes, and stale-cursor rejection all unproven. [Report 4.]

**Resolution (2026-08-15):** `exec_pagination.rs` executes both cursor generations against data: v1 skip pages (10/10/5, gap-free, terminating), v2 keyset pages surviving a flush plus one already-served and one not-yet-served delete (no duplicates, the unseen delete vanishes, everything else exactly once), and the cursor wire contract (hash round-trip, doctored blobs rejected). The module remains without an in-tree server consumer — wiring it into the HTTP surface stays future product work, but its contracts are now pinned.

New crates/namidb-query/tests/exec_pagination.rs: seed ~50 nodes; page via paginate_plan_keyset/next_cursor_keyset to exhaustion; assert concatenation equals the full ordered scan; repeat with a flush mid-pagination and a row deleted between pages (no dup, no gap besides the deleted row); assert plan-hash mismatch rejects a stale cursor.

### 18. [DONE] No exec-level traversal ever crosses a flushed high-degree supernode: skew buckets and dense-block lookups are unit-tested but MATCH/Expand/var-length never touch a hub through the snapshot API. [Report 3.]

**Resolution (2026-08-15):** `exec_supernode.rs`: a flushed 2500-out/401-in hub crossed by single-hop out, inverse single-hop, directed `*2..2` through the hub, and count(r) pushdown at type scale — all exact.

crates/namidb-query/tests/exec_match_expand.rs: flush a 10k+ out-degree hub with some partners tombstoned and some memtable-fresh; assert MATCH (hub)-[:R]->(x) RETURN count(x), a 2-hop var-length through the hub, and an aggregate each match computed expectations.

### 19. [MOSTLY DONE — scan_edge_type residual] count_edge_type/scan_edge_type materialize a BTreeMap over every (src,dst) of the type (read.rs:5548-5692) — O(edges) RAM behind count(r) pushdown and every algo.* graph build; at 25 TB a single CALL algo.wcc() is O(corpus)-resident. [Report 3.]

**Resolution (2026-08-15), count half:** `count_edge_type` now takes a metadata fast path whenever the type has at most one forward SST (the compacted steady state): live = `row_count - tombstone_count` from the manifest stats plus an O(memtable) delta resolved by paged point lookups (every memtable entry outranks flushed rows by the LSM flush cut). Zero body bytes with an empty memtable. Multi-SST trees keep the exact merge fallback. Pinned by `count_edge_type_fast_path_matches_the_multi_sst_merge` (same logical graph built 1-flush and 2-flush; all four delta classes; count == merged scan on both). **Residual:** `scan_edge_type` still materialises the full map — it feeds `algo.*` graph builds, which need a streaming adjacency source design, tracked with the CSR/optimization backlog.

crates/namidb-storage/src/read.rs envelope test: count_edge_type over multi-SST disjoint edge streams asserting a peak-resident bound (no per-pair map when SSTs are disjoint); plus a documented size gate for algo.* graph builds beyond a threshold. Like rank 14, expect an engineering fix before the test passes.

### 20. [DONE] Edge-bucket tombstone GC at the deepest merge (merge_edge_sources, compact.rs:3602) untested — every preserved-then-GC'd test is node-only; forward/inverse SST consistency after edge-tombstone drop unasserted. [Report 3.]

**Resolution (2026-08-15):** `edge_tombstone_gc_at_the_deepest_merge_drops_rows_in_both_directions` (authoritative compaction physically drops tombstones from both directions, forward/inverse row sets mirror, stats agree, reads and count confirm) plus `non_authoritative_edge_merge_preserves_tombstones` (the `gc_tombstones` flag keeps the shadow when a deeper level exists).

crates/namidb-storage/src/compact.rs edge mirror of tombstone_above_a_deeper_level_is_preserved_then_gcd_at_the_deepest: edge upsert at L2, tombstone at L0; shallow merge asserts tombstone_count>0 in BOTH fwd and inv SSTs; deepest merge asserts 0 and out_edges/in_edges both empty.

### 21. [DONE] Var-length pattern with a relationship alias ([r:KNOWS*1..2]) silently binds only the LAST hop instead of a list (walker.rs:1546) — wrong-shape results for a query form production users will write. [Report 3.]

Either reject in plan/lower.rs (like shortestPath validation) with an exec_match_expand.rs test asserting the error, or implement list binding and assert MATCH (a)-[rs:KNOWS*1..2]->(b) RETURN rs yields per-path relationship lists. Rejection is the fast safe fix.

**Resolution (2026-08-15):** implemented the correct list binding instead: a starred alias accumulates the path's relationships per step and binds `List` on emission (empty list on the `*0..n` hop-0 row); the unstarred fixed form keeps its scalar. The factor expand route delegates starred-alias patterns to the flat executor (same pattern as path bindings) since its arena slots would resolve to the last hop. Pinned by `exec_match_expand.rs::var_length_alias_binds_the_relationship_list`.

### 22. [DONE] Prefix expansion beyond the 64-term cap untested on both flat (displaces_last lexicographic loop) and native (global expansion under-fill/overflow branches) — silently wrong ranking for every wildcard query over a 25 TB vocabulary. [Report 2.]

**Resolution (2026-08-15):** `exec_call.rs::prefix_expansion_beyond_the_cap_is_identical_on_every_route`: 100 distinct terms under one prefix expand to exactly the 64 lexicographically first on the flat overlay route, the single-base native route, AND the multi-segment native route (global reconciliation); a delta whose terms sort after the cap does not disturb the set, an unrelated-prefix delta neither, and an under-cap prefix returns every match. Passed first run.

crates/namidb-query/tests/exec_call.rs parity test: >64 distinct terms sharing one prefix split across two FT4 segments with overlapping vocabularies; assert index/flat rank+score parity and identical lexicographically-first-64 selection; storage-level assert the overlap-underfill branch returns None (flat authority) rather than an under-filled expansion.

### 23. [DONE] Empty/edge search inputs untested at the procedure layer across text and vector: query ''/whitespace/'*', k=0, k omitted (unbounded incl. SearchResultLimitExceeded guard), empty/unknown label, LIMIT 0 on KNN. [Merges Report 2 high + Report 1 medium k-edges.]

**Resolution (2026-08-15):** `exec_search_input_edges.rs` pins empty/whitespace bm25 queries (zero rows, no error), k:0 on both procedures, omitted-k bounded default, LIMIT 0, unknown label (clean empty/clean error, never a panic). All passed first run — fixation, not fixes.

crates/namidb-query/tests/exec_call.rs + exec_vector_knn.rs table-driven: each edge input asserted for zero-rows-no-error (or the typed limit error with env-shrunk cap) on BOTH a memtable-only corpus and an indexed one; LIMIT 0 KNN asserts VectorSearch still in the plan.

### 24. [DEPRIORITIZED — legacy-only path, not on the 25TB load] Capped-sidecar filtered fallback chain (native Unsupported → resolve_capped_filter_eligibility → try_vector_search_filtered, legacy-only) untested at every layer including the ID-candidate cap boundary and the empty-sidecar early return — the serving path for filtered KNN over migrated legacy v3 .vg bodies. [Report 1.]

**Rationale (2026-08-15):** this chain serves ONLY deployments carrying migrated legacy v3 `.vg` bodies. The 25 TB engagement loads fresh data on 2.1.0 (native V5 bodies throughout), so the chain is unreachable there; fabricating v3 fixture bytes needs format archaeology with no writer left in-tree. Keep for a maintenance window driven by an actual v3-carrying deployment.

crates/namidb-query/tests/exec_vector_knn.rs: build a legacy v3 .vg manifest (reuse the wire fixture from sst/vector.rs::real_v3_wire_body_decodes_and_leaves_filter_residual) plus equality sidecar; assert (a) filtered search.vector equals exact filtered top-k, (b) empty sidecar set → [] without touching the .vg, (c) NAMIDB_VECTOR_FILTER_ID_CANDIDATE_CAP=1 degrades to exact flat fallback with identical results.

### 25. [DONE] Non-cosine metrics untested on procedure forms: queryNodes never run against dot/euclidean descriptors; search.vector metric:'dot'|'euclidean'|'l2' options and unknown-metric error untested (Cypher forms are covered). [Report 1.]

**Resolution (2026-08-15):** same file: dot ranks by inner product (magnitude wins), euclidean/l2 by distance, unknown metric errors naming the contract, and db.index.vector.queryNodes serves a Dot-metric descriptor correctly.

crates/namidb-query/tests/exec_vector_knn.rs: build_index_metric with Dot and Euclidean; queryNodes matches brute force with correct orientation; search.vector metric:'euclidean' ranks ascending-distance; 'l2' alias accepted; 'chebyshev' errors with documented message.

### 26. [DONE] Expression evaluator dark corners: STARTS WITH/ENDS WITH/CONTAINS never evaluated in any test (matching, case sensitivity, null/non-string → null, residual-above-index role); simple/scrutinee CASE form has no parse or eval test; missing-ELSE → NULL untested. [Report 4, two gaps merged.]

**Resolution (2026-08-15):** `exec_expression_corners.rs` pins case-sensitive STARTS WITH/ENDS WITH/CONTAINS, NULL and non-string operands NULL-filtering, both CASE forms and missing-ELSE → NULL.

crates/namidb-query/src/exec/expr.rs unit blocks: the three string operators incl. NULL and integer operands; CASE 2 WHEN 1..WHEN 2..END branch selection; no-match-no-ELSE → Null; CASE NULL WHEN NULL pinned. One e2e in exec_match_expand.rs: WHERE p.name STARTS WITH 'A' as residual above an indexed equality conjunct over flushed data.

### 27. [DONE] Top-level UNION/UNION ALL: no read e2e test and no column-name compatibility validation across branches (mismatched branches silently yield mixed-shape rows). [Reports 3+4, deduplicated.]

**Resolution (2026-08-15):** the lowering now derives each branch's output column names (aliases or canonical expression names; CALL-with-YIELD binding names) and rejects any branch whose names or order differ from the head, naming both sets in the error. e2e UNION dedupe vs UNION ALL duplicates pinned in `exec_expression_corners.rs`.

exec_match_expand.rs: top-level MATCH..RETURN x UNION MATCH..RETURN x dedup + UNION ALL multiplicity, result-asserted on memtable and flushed states; plan/lower.rs test that RETURN a AS x UNION RETURN b AS y is rejected (after adding validation).

### 28. [DONE] HTTP JSON parameter route untested: params_from_json/json_to_runtime has no unit tests and no HTTP test posts a non-empty params map (nested list/map, u64-beyond-i64, float round-trip). [Report 4.]

**Resolution (2026-08-15):** found and fixed a real hazard — a u64 beyond i64::MAX silently degraded to a lossy float; it now rejects with a named error and HTTP 400. Unit tests pin nested list/map conversion, i64::MAX/MIN, bit-exact 0.1 round-trip and the out-of-range rejection; an HTTP e2e posts a non-empty params map (bool, list, map, float) through /v0/cypher and asserts the JSON round-trip.

crates/namidb-server/src/lib.rs inline: post_cypher with {"params": {"n":1, "tags":[..], "props":{..}, "big": 18446744073709551615}} asserting bound results and a typed 400 for the unrepresentable number.

### 29. [MOSTLY DONE — old-binary backup open stays manual] No frozen 2.0.6 wire-decoder rollback test (downgrade simulated by struct mutation only), and no test that a backup taken on 2.1.0 opens on an older binary. [Report 5, two related gaps merged.]

**Resolution (2026-08-15):** `tests/frozen_wire_206.rs` freezes the ACTUAL fa126e1 (v2.0.6) wire contract as golden constants — required manifest fields, required SstDescriptor core fields, decodable SstKind variants — and holds a real 2.1.0 manifest (active search generation included) against them, then prunes the JSON to exactly the 2.0.6 field set (a downgraded writer's rewrite) and reloads it with today's decoder, interop markers surviving. Changing a golden list is now a visible wire-break decision. Opening a 2.1.0 backup with a real old binary remains a manual pre-release step (needs the released 2.0.6 artifact).

New crates/namidb-storage/tests/manifest_rollback_206.rs modeled on manifest_rollback_204.rs: freeze the exact 2.0.6 manifest DTO; decode a 2.1.0 manifest carrying an Active generation; reserialize (dropping unknown fields); feed the round-trip to search_lsm_adoption_needed/compact_leveled asserting metadata-only re-adoption; plus decode a post-backup destination manifest (accelerators dropped) with the frozen DTO asserting it loads with self-contained WAL closure.

### 30. [DONE] Vector-kind downgrade adoption cycle has no text twin: barrier recovery and marker-gated adoption after a state wipe proven only for SearchLsmKind::Vector. [Report 5.]

**Resolution (2026-08-15):** `text_preserved_barrier_readopts_after_state_wipe`: a state wipe with the checksummed `.slb` preserved re-adopts the text generation metadata-only (0 SSTs removed/written, barrier reused byte-for-byte, `text_search` serves natively after), and a DDL-stale marker refuses adoption. The lost-barrier half was already pinned by `text_base_with_lost_barrier_falls_back_to_rebuild` (item 11); note the item-11 marker-signature fix is what makes text adoption possible at all.

crates/namidb-storage/src/compact.rs: extend compaction_builds_a_searchable_text_index with the vector test's three-phase downgrade block (state wiped → .slb metadata-only recovery; stale catalog marker → rebuild; adopted state serves via native FT4 route).

### 31. [DONE] Janitor never tested against active search state: sweep preserving barrier/base/delta objects of an Active generation, and sweep racing a pinned native read over a just-retired generation. [Report 5; the race half overlaps rank 3's framework — keep the unit-level pair here.]

**Resolution (2026-08-15):** `sweep_preserves_active_search_generations_until_the_horizon_passes`: the Active generation's body+barrier survive any sweep; after a consolidation retires them, a sweep pinned at the old version preserves the retired objects and a reader loaded at that version still serves the old corpus; past the horizon they are reclaimed while the new generation serves. The dynamic race half was already covered by `tests/search_concurrency.rs` (janitor fed by the readers' floor during live maintenance).

crates/namidb-storage/src/janitor.rs: (1) sweep_finds_no_orphans_with_active_search_generation — Active vector+text generation + one planted true orphan; sweep deletes only the orphan; (2) sweep_keeps_retired_generation_under_pin_lease — retire a delta generation via search compaction while a RetentionPin at the old version is held; old segments survive until release.

### 32. [MOSTLY DONE — container `auto` mode stays with the runbook] NAMIDB_MEMORY_MAX_BYTES proven only with synthetic RSS; no real-ceiling server test (503/Bolt-transient rejection, reclaim, recovery) and 'auto' cgroup mode never validated in a container. [Report 5.]

**Resolution (2026-08-15):** `real_rss_ceiling_rejects_queries_and_recovers` runs the ceiling against the REAL resident-set sample: a 64 KiB ceiling (below any real process RSS) rejects /v0/cypher with 503 and counts the rejection; a sane ceiling serves. Container `auto` cgroup validation stays in the pre-load runbook §3 where a real memory-limited container exists.

crates/namidb-server/tests/memory_ceiling.rs: AppState with a tiny explicit ceiling + a test hook feeding real RSS; drive concurrent Cypher to rejection; assert 503 body then recovery. Docker workflow smoke step: container under --memory with NAMIDB_MEMORY_MAX_BYTES=auto asserts startup succeeds (and fails without a limit).

### 33. [MOSTLY DONE — real-provider runbook remains] All S3/R2 behaviors (multipart of spooled artifacts, ranged GETs, CAS beyond manifest bootstrap, retry/backoff, TLS) tested only against InMemory; the lone LocalStack test covers manifest CAS and runs nowhere. [Report 5.]

**Resolution (2026-08-15):** `s3_integration.rs` gained `full_cycle_ingest_search_backup_against_s3` — ingest with vector+text indexes, flush, compact, reader-node native serving of both routes, a VERIFIED backup to a sibling prefix that serves too, and an orphan sweep, all against LocalStack S3 with real conditional writes. Validated locally against a live LocalStack (2/2 green) and wired into `nightly.yml` as the `s3-localstack` job with a health-checked service container. Real-provider behaviors (R2 WAN latency, TLS, retry) remain covered by the pre-load runbook §1-2.

Extend tests/s3_integration.rs with an #[ignore] full cycle (ingest → flush with an object exceeding one multipart part → compact building .vg/.ft → native search via ranged reads → backup to second prefix → restore + re-query); add a CI job starting tests/docker-compose.s3.yml and running cargo test --test s3_integration -- --ignored. TLS/WAN stays in the pre-load runbook against the customer's real bucket.

### 34. [DONE] Full-stack route-assertion gap: the only tests through the REAL server/py pipeline deliberately accept either route, so total permanent loss of native serving (every query O(corpus)) passes CI. [Report 2; the systemic version of the reachability trap.]

**Resolution (2026-08-15):** new `namidb_storage::route_telemetry` — process-wide monotonic counters recorded at the storage decision points (`text_search`/`try_vector_search_with_point_count` wrappers: served natively vs declined toward the fallback), exported by the server as `namidb_search_route_total{kind,route}` on `/v0/metrics`. A deployment whose native serving silently died now shows a flatlined native counter beside a climbing fallback counter. Pinned by a unit delta test, a storage e2e (native serve moves native, freshness decline moves fallback) and a server test asserting all four series render. Also fixed a latent env-race: `prepared_compaction_round_trip_matches_compact_l0` observed the policy env without the guard.

Expose a route observable (metrics counter for native text/vector serves — also fixes part of the metrics blind spot) and assert it in crates/namidb-py/tests/test_cypher.py::test_compaction_materializes_vector_and_text_indexes (and a server twin) after flush+compact with no pending delta, alongside the existing result checks.

## Dimensions not yet audited

- Server auth/authz surface end-to-end: crates/namidb-server/src has auth.rs, authz.rs, jwt.rs, pdp.rs, tls.rs with only small unit-test counts (2-9 each), and NO report audited whether HTTP and Bolt endpoints actually enforce authn/authz/TLS end-to-end (unauthenticated request rejection, expired/forged JWT, pdp decision wiring per procedure, Bolt auth handshake). For a single company's 25 TB production dataset this is the perimeter, and it was entirely outside all five reports' scope.
- namidb-mcp crate (MCP server binary) — a whole query-entry transport with zero audit: no report enumerated its tool surface, its parameter conversion, its error mapping, or whether it has any tests at all (crates/namidb-mcp has no tests/ dir).
- namidb-markdown crate (parse.rs/embed.rs/load.rs/remote.rs) — likely the document ingestion/embedding pipeline feeding the very vector+text indexes audited, including a remote.rs fetch path (SSRF/content-trust surface); completely unaudited, and if the 25 TB load flows through it, its chunking/embedding/id-assignment semantics are upstream of every search-correctness result.
- Cache coherence — AUDITED 2026-08-16: the design defense is content-addressed keys (compactions write new UUID paths; range-cache keys carry the backend generation plus a per-store-instance token). Pinned by `cache_invalidation.rs`: six warm read/update/flush/compact cycles with every cache tier at its default — the freshest committed revision wins every read from brand-new reader snapshots, never a superseded row.
- Server maintenance scheduler (crates/namidb-server/src/maintenance.rs — ZERO inline tests): the loop that decides WHEN flush/compaction/search-compaction/janitor run under production load; starvation, overlapping-run exclusion, and backoff-after-failure are unaudited, distinct from the janitor internals report 5 covered.
- Metrics/observability correctness (crates/namidb-server/src/metrics.rs, 7 inline tests): several proposed fixes rely on metrics counters as route observables, yet no report audited metric semantics (counter attribution, /metrics endpoint shape) — the instruments operators will trust during the 25 TB load are themselves unverified.
- Planner statistics lifecycle — AUDITED 2026-08-16: `StatsCatalog::from_manifest` derives everything from the committed manifest, so callers refresh by rebuilding from the current manifest. Pinned by `stats_lifecycle.rs`: memtable-only writes are invisible (documented zero shape), one flush+compaction later per-label and per-edge-type counts are exact (last-write-wins collapse included), and deletions track through the next maintenance cycle.
- Bulk-load operational contract: restartability and idempotency of a multi-day ingest (resume after crash mid-batch, duplicate handling on retried batches, WAL rotation and flush backpressure/write-stall behavior under sustained ingest pressure) — reports covered crash recovery and fencing generically but never the loader-facing semantics the actual 25 TB load will exercise first.
- Concurrent multi-session write semantics in-process — AUDITED 2026-08-16: the contract is mutex serialization on the shared WriterSession (with a lock timeout answering 503/Bolt-transient). Pinned by `concurrent_http_writes_serialize_and_both_commit`: eight simultaneous HTTP writes all succeed and every effect is visible exactly once.
- DDL racing queries and maintenance — AUDITED 2026-08-16, real bug found and fixed: the first compaction after DROP+CREATE INDEX aborted with "Building Search-LSM has partial coverage" (install validated the coverage of states its own rebuild replaces), wedging maintenance after every index recreation. Fixed by exempting `replaced_search_lsm` states from the node-rewrite coverage rebase. Pinned by `ddl_recreate_min.rs` (sequential 4-cycle storm) and `search_ddl_races.rs` (DDL storm racing live reader snapshots: never an error, native serves match same-snapshot oracles, the recreated index serves the full corpus after).
- Property type-system matrix — AUDITED 2026-08-16, no defects: `exec_type_matrix.rs` pins WHERE-equality per kind (i64::MAX, negative floats, bools, multibyte-UTF-8 indexed strings, dates/timestamps, nested list/map presence), byte-exact nested/binary round-trips through the flushed route, and NULL-stable ORDER BY, identical on memtable and SST routes.
- Local disk exhaustion (ENOSPC) during spooled builds: the disk-spooled index builders and multipart staging write to local scratch; fault injection covered object-store PUT failures but no report considered scratch-disk-full mid-build/mid-flush — highly plausible during a 25 TB load.

### 35. [INVESTIGATED — no engine bug; reduced to a calibration task] Serving-ANN recall at a 64-page synthetic corpus needs nprobe ~24-32/64 for 0.95, so the gate cannot yet prove byte-pruning below 1.0 at this scale.

Hard data (all at --vectors 2048 --documents 2048 --dim 64 --page-rows 32 --branch-factor 4 --seed 42, k=5, 5 queries):
- Pure `VectorV5Reader` on an equivalent clustered corpus (16 clusters, spread 0.08), in-memory build: recall@5 = 0.93 at nprobe 4, 1.00 at nprobe 8 (probing ~12% of pages). External builder (2 MiB budget, full quantile sample): 0.91 / 1.00. The engine's pruning is healthy.
- The gate's base vector tracks report 0.56-0.92 across cosine/dot/l2 at nprobe 4-8, moving with spread and centroid layout — partially the synthetic generator's overlapping random centroids (pairwise cosine up to ~0.4 at dim 64), but never reaching the pure-reader numbers; unexplained residual.
- `search_lsm.vector_serving_ann` recall is pinned at 0.72 unfiltered / 0.68-0.80 filtered and does NOT move when nprobe goes 4->8, when spread goes 0.35->0.08, or when centroids are orthogonalized; only probing every page (nprobe=max-nprobe=64) reaches 1.0. The serving track passes `config.nprobe/max_nprobe` into `V5BaseQueryMode::ServingAnn`, so the invariance is NOT explained by the ef-derived `vector_v5_search_options` path alone.
- Orthogonalizing `make_centroids` (Gram-Schmidt) was tried and REVERTED: it broke the 256-vector PR gate (dot 0.933) without moving the serving number.

Investigate: instrument `coordinated_vector_query`/the bench coordinator to dump, per lost truth id, whether it lives in the V5 base or a VG6 delta, which page holds it, and whether that page was probed. Prime suspects: the mutation model (`updated_vector` shifts +-0.35 off-cluster, placing updated rows far from their build-time page), truth/oracle versioning in `exact_*` vs what the coordinator fuses, and the widening criterion (candidate-count satisfaction never widens past the initial nprobe). Exit criteria: either a real engine/coordinator fix with the gate green at 2048 vectors, nprobe 8/32, min-recall 0.95, max-cold-ratio < 1.0 — or a proven harness bug fixed and the same green gate. Then grow the PR gate per the original 8(c).

**Findings (2026-08-15, second pass):** the "pinned at 0.72" invariance was an artifact of --queries 5 (25 samples, 0.04 granularity). At --queries 25: recall scales smoothly with nprobe (base cosine 0.896@8 -> 0.92@16; serving 0.80@8 -> 0.92@16, base == serving), and the coordinator's EXACT shadow reports node_ids_exact=true against the oracle — no engine bug, no harness bug. The residual difficulty is the synthetic generator's geometry: even Gram-Schmidt-orthogonalized centroids at spread 0.08 leave boundary queries whose true top-5 spans enough pages that 0.95 recall needs ~24-32 of 64 probes, at which point cold bytes exceed the corpus (nav+metadata dominance at 8 KiB exact pages). A separable-dataset flag was prototyped and reverted (broke the 256-vector PR gate's dot track and still did not reach 0.95@8).

**Remaining task (not a blocker):** validate byte-pruning on a corpus where it is geometrically provable — either a nightly gate at 100k+ vectors with page-rows 128+ and a measured threshold, or against the real dataset via the pre-load runbook (§2 LoadR2). The PR gate deliberately stays at 256 vectors / ratio 2.0 as a semantic contract, not a pruning proof.

## Optimization tracker status (2026-08-16)

Of the seven verified 10M-scale blockers from the 2026-07-28 audit:
1. O(corpus) BasePrefix rebuild RAM — CLOSED (streaming producer/consumer, `peak_resident_input_bytes` asserted).
2. Sequential VG6 delta scans per query — CLOSED (concurrent per-round segment waves, byte-identical results).
3. Block-Max/WAND reachable only via a single clean base — OPEN, post-release: absence costs text pruning with >1 segment, never correctness; needs a global-WAND-with-dirty-ids design.
4. FT4 dictionary block double fetch — CLOSED (bounded per-reader decoded-block cache, zero-new-reads pinned).
5. No process-wide caches on the search path — SUBSTANTIALLY MITIGATED: the RAM page cache now defaults ON with per-store-instance keys, so hot index/nav/posting pages are shared process-wide; per-snapshot decoded readers remain (CPU-light over cached pages) and per-query barrier HEADs are deliberate validation.
6. Legacy (format_minor 0) edge SSTs read full-body — BY DESIGN (compat); ranged property hydration covers modern SSTs.
7. Object-native gate — CLOSED earlier (running in CI with measured thresholds).

### 36. [DONE — planner, lands in 2.1.1] Pattern anchor is not inverted toward the selective endpoint.

**Resolution (2026-08-16):** new fixpoint pass
`optimize/anchor_inversion.rs`, run right after `unique_lookup`. When a
single-hop pattern is anchored on a bare `NodeScan` but the *target* node
carries an indexed equality conjunct (same `extract_indexed_conjunct`
machinery unique_lookup uses), the pass re-anchors: `NodeByPropertyValue`
on the target, the `Expand` direction flipped (`->` ↔ `<-`, `--` stays),
aliases/labels swapped to preserve every binding, residual predicates
re-attached above. Unique targets always invert; non-unique indexed
targets only when catalog stats say the target label is no larger than
the source (posting fanout could otherwise lose). Semantics-bearing
shapes are refused: var-length, OPTIONAL, back-references, shortestPath,
and walker-materialized trails (`path_binding`); single-hop `q = ...`
paths still invert because their trail is assembled statically from the
bindings by a Project. Pinned by six unit tests on lowered plans plus
`tests/exec_anchor_inversion.rs`: both spellings return identical
bindings on memtable and flushed routes, the slow spelling's optimized
plan contains the cid anchor and no Person scan, a post-flush memtable
delta stays visible through the inverted anchor, and the optimized path
shape equals the unoptimized reference. Cold all-memtable namespaces
(no manifest stats yet) keep the un-inverted plan by design.

Found 2026-08-16 during the live 200k-node S3 validation. With a UNIQUE
constraint on `Company.cid`, `MATCH (c:Company {cid: 0})<-[w:WORKS_AT]-(p:Person)
WHERE w.since = 0` answers instantly (index lookup + inverse expand, O(degree)),
but the semantically identical `MATCH (p:Person)-[w:WORKS_AT]->(c:Company
{cid: 0}) WHERE w.since = 0` scans all 200k Person nodes and expands forward
(~18s warm, times out cold at the default 30s query timeout). The optimizer
should consider anchoring at the most selective pattern endpoint regardless of
the direction the user wrote. Reproduction: any big label -> tiny
unique-anchored endpoint written left-to-right.

### 37. [DONE — storage, blocker-at-scale, fix on fix/flush-posting-fsync-storm] First large flush fsyncs once per distinct indexed-property value, holding the writer lock for hours.

Found 2026-08-16 during the 5M-record live S3 validation (LocalStack,
released 2.1.1 and 2.1.0 binaries — NOT a 2.1.1 regression). Loading 5M
records (2.475M Person + 50k Company with a UNIQUE string `cid` + 2.475M
WORKS_AT) stalled permanently at ~265k rows: the first flush past the
memtable threshold acquired the writer lock and crawled at ~128 KB/s to
anonymous O_TMPFILE scratch files, so every write got "writer is busy"
(503) for what would have been hours. Root cause:
`finish_external_posting` (flush.rs) called `sync_data()` once per
DISTINCT VALUE of every posting-indexed property — with string-heavy
properties (names are per-row distinct) that is O(rows) ext4 journal
commits at ~3 ms each. The fsync bought nothing: the scratch is a deleted
anonymous file (no durability role), its bytes are copied into the
paged-index value spool immediately while page-cache resident, that copy
is covered end-to-end by the crc32 stored in the index leaf (fail-stop on
read), and ENOSPC still surfaces at the value region's own amortized
`sync_data` when the builder finishes. The other nine `sync_data` sites
in the flush/SST pipeline are per-run or per-builder (amortized over MBs)
and remain untouched.

**Resolution (2026-08-16):** removed the per-value `sync_data`. With the
fix the same 5M load completes in 210 s (~24k rows/s sustained; flushes
now finish in ~20 s, visible only as brief throughput dips), RSS peaks at
1.50 GB under the 3 GB ceiling, and the item-36 validation passes 13/13
on the flushed data through both the patched build and the released 2.1.1
binary. Also re-verified during diagnosis: WAL recovery after `kill -9`
restored 580k acknowledged records exactly. Remaining related exposure
(untested, future work): writes are unavailable for the full duration of
any flush once the memtable passes the stall threshold — acceptable at
~20 s per flush, but a soak asserting an upper bound on write-outage
windows during sustained load would pin it.

### 38. [DONE — product, lands in 2.1.3] CREATE INDEX on already-loaded data has no effect until a flush/compaction rewrites the SSTs — there is no backfill or REINDEX.

Reported from field validation 2026-08-17 (twin-namespace experiment:
index-before-load = sub-millisecond lookups; index-after-load = ~850 ms,
identical to no index). Confirmed in code: property-posting sidecars are
only written by `finish_external_posting` during flush and by compaction
rewrites; `CREATE INDEX`/`CREATE CONSTRAINT` only stamps the schema. On a
loaded, quiet namespace the new index never materializes. Options, in
increasing effort: (a) document "indexes before bulk load" as a hard rule
(done — but it will burn users), (b) have property DDL schedule a
compaction of SSTs whose generation predates the DDL (the search-LSM
already does exactly this via signature retirement — PR #133 — the graph
side needs its twin), (c) an explicit `db.index.build()` procedure.
Recommend (b): the machinery and the precedent both exist.

**Resolution (2026-08-17):** option (b), and it turned out the heavy half
already existed: since 2.0.5 the compaction planner treats a node SST
whose descriptors lack a posting sidecar for a currently-indexed
Utf8/Bool property as needing migration (`node_descriptor_needs_migration`,
compact.rs) and will rewrite even a single fully-compacted L1 SST
(`plan_node_bucket`) — the schema commit alone flips `needs_compaction()`
to true. What was missing was the trigger: materialization waited for the
periodic compaction tick (default 5 min) and never happened with periodic
compaction disabled. Now every successful property DDL — HTTP
CREATE CONSTRAINT / CREATE INDEX (single- and multi-tenant) and the Bolt
twin — requests one compaction pass with a new `CompactionTrigger::Ddl`
(visible as `namidb_compactions_total{trigger="ddl"}`); the scheduler's
single-flight admission coalesces storms and an IF NOT EXISTS no-op
re-request answers Noop from metadata. Pinned by
`namidb-server/tests/ddl_index_backfill.rs`: with every periodic loop
disabled, 300 flushed rows + CREATE CONSTRAINT + CREATE INDEX →
both posting sidecars appear on every node SST (observed via a second
store handle on the same file:// prefix, in under 2 s), the lookup
answers through the query surface, and the ddl-trigger counter moves.
Still true and now recorded here: postings only exist for
Utf8/LargeUtf8/Bool properties (numeric equality stays scan-side — the
type filter in `union_indexed_props`/`EqualitySidecarCollector`), and in
the window between DDL and the pass finishing, one uncovered SST in
scope silently disables the index for the whole lookup (no metric) —
that observability gap is item 39's territory.

### 39. [DONE — observability, lands in 2.1.3] EXPLAIN shows NodeScan + Filter even when the posting index accelerates the scan at runtime.

The pushed-predicate/posting acceleration happens INSIDE the scan
operator, so the logical plan is truthful about shape but silent about
the physical route. Users from Neo4j expect `NodeIndexSeek` to confirm an
index works and will (reasonably) conclude ours don't; today the only
witness is `elapsed_ms`, plus the plan-level `NodeByPropertyValue` when
the optimizer rewrite fires. Fix direction: annotate EXPLAIN output with
the physical access path chosen at execution (or at minimum a
`route: native|scan` counter per operator in EXPLAIN VERBOSE), mirroring
the `namidb_search_route_total{route}` pattern that already exists for
search.

**Resolution (2026-08-17):** the recon found the reality was worse than
reported: the SERVER never handled EXPLAIN at all — `EXPLAIN <query>`
over HTTP or Bolt silently EXECUTED the query (an `EXPLAIN CREATE` wrote
data), and the only real consumer, `namidb explain` (CLI), renders with
an empty StatsCatalog so it can never show `NodeByPropertyValue`. Now:
(1) both surfaces serve `EXPLAIN [RAW] [VERBOSE]` — the optimized plan
against the real manifest catalog, one row per line, nothing executes;
(2) a `# route:` footer states the physical access path per index
lookup, computed from actual sidecar coverage
(`Snapshot::property_index_coverage`): `index (unique|posting lookup;
posting sidecars N/N SSTs)`, `memtable (no SSTs in scope)`,
`SCAN FALLBACK (sidecars K/N SSTs)`, or the numeric caveat
(`numeric equality is not posting-indexed`); (3) the silent
index-to-scan demotions in the storage lookup routes (pre-sidecar SST in
scope, unreadable sidecar — across the unique, string-multi, batch, and
equality-inner paths) now count into
`namidb_property_lookup_route_total{route="native"|"fallback"}` on
/v0/metrics, the same alarm shape as `namidb_search_route_total`; and
(4) the stale AST doc promising executor-side EXPLAIN handling was
corrected. Pinned by `namidb-server/tests/explain_surface.rs`: EXPLAIN
of a write creates nothing, the optimized plan shows
`NodeByPropertyValue`, the footer transitions memtable→index across a
flush, numeric equality carries the caveat, VERBOSE adds estimates, and
a real indexed lookup moves the native route counter on /v0/metrics.

### 40. [DONE — resilience, lands in 2.1.3] Disk-full during flush permanently wedges the namespace: writer lock never released, reads queue behind the guard, no error, no timeout — only a restart recovers.

Reported reproduced twice in field validation with a full spool disk.
The flush error path must (1) release the writer lock on ANY failure,
(2) fail the namespace into a degraded read-only state that keeps serving
the last committed snapshot instead of queueing reads forever, and
(3) surface a typed error on writes rather than infinite "writer is
busy". Reads blocking behind a failed flush is the worst part — losing
reads to a WRITE-side disk problem inverts the durability story. Needs a
FaultStore/ENOSPC injection test (tmpfs with a size cap) pinning all
three behaviors. This is the most serious resilience finding of the
round; the production mitigation until fixed is disk alerting
(node_exporter) plus the bulk deadline's clean 408.

**Resolution (2026-08-17):** the wedge was a four-link chain, each link now
cut. (1) The flush's local spool I/O errors were wrapped into
`Error::invariant`, hiding their nature — they now stay `Error::Io` and a
new `Error::is_local_persistence()` classifies them. (2) On such a failure
the periodic flush loops (single- and multi-tenant) mark the namespace
persistence-degraded in `WriterHealth` and back off retries 30 s
(`FLUSH_FAILURE_RETRY_BACKOFF`) instead of immediately re-running the doomed
O(corpus) build under the writer mutex; the first successful flush clears
the state. (3) While degraded, every write intake point — HTTP write, all
five schema-DDL handlers, all six Bolt write sites, Bolt BEGIN — rejects
with a typed 507 `namespace degraded: …` (Bolt: a non-retriable failure
carrying the reason) BEFORE queueing on the writer mutex, and reads are
never gated; `/v0/health` carries the reason. (4) The starvation vectors
that let the outage swallow reads are bounded: `/v0/admin/flush` now bounds
both its process-wide permit wait and its writer-lock wait (30 s → 503)
instead of pinning global HTTP concurrency slots on a route excluded from
every timeout, and the process-wide `FLUSH_BUILD_GATE` acquire — which a
detached, unabortable spool build could trap forever — is bounded at 10 min
(`NAMIDB_FLUSH_BUILD_GATE_TIMEOUT_SECS`). Pinned by
`namidb-storage/tests/flush_spool_failure.rs` (failed flush: Io-classified,
session not poisoned, memtable restored byte-exact, later commits accepted,
retry lossless) and `namidb-server/tests/disk_full_degraded.rs` (real HTTP:
flush on broken spool → 507 + degraded; writes 507 typed, reads 200,
health 503 with reason; after the disk clears one flush retry restores
writes with nothing lost — no restart). Both are single-test integration
binaries because they mutate the process-global `NAMIDB_SPOOL_DIR`.
Remaining (recorded, not blocking): each backoff-spaced retry still holds
the writer mutex for its build duration; the multi-tenant registry eviction
still waits unbounded on the evicted writer while holding the sessions
lock (short now that the mutex frees); moving the flush build off-lock like
compaction's prepare stays future work.

### 41. [CONTAINED — performance; gate lands in 2.1.4, deep fixes remain] Concurrent large scans collapse aggregate throughput (~660 rps mixed ceiling; pathological control: ~2 rps aggregate, 8 GB RSS on parallel 1M-row scans without index).

Suspected shared-cache thrash (unprofiled). The per-query protections
(row cap, deadline, breaker) contain the blast radius, but the engine
does not degrade gracefully on its own: one scan-heavy tenant can starve
the box. Also note 8 GB RSS implies the run was outside the memory
governor's admission (or the governor does not bound scan working sets —
worth checking which). Follow-ups: profile the pathological control
(cache hit rates, eviction churn), consider scan admission (limit
concurrent full scans per namespace), and per-tenant cache partitioning
or scan-resistant eviction (e.g., segmented LRU so one scan cannot flush
the point-lookup working set).

**Containment (2026-08-17) + mechanism map.** Recon established the full
chain: (1) a `NodeScan` materializes its whole label THREE times (the
LWW-reconciliation `BTreeMap` over potentially the whole store, the
`Vec<NodeView>`, then the walker's `Vec<Row>`) — nothing streams on the
executor path even though `visit_label_with_projection` exists; (2) the
memory governor's admission is wire-byte-sized (a 50-byte scan query
reserves ~65 KiB against a multi-GB execution) and `admit_query` samples
RSS point-in-time with zero headroom, so N scans admit low and balloon —
that is the 8 GB; (3) every body a scan touches was inserted into the
shared S3-FIFO body tier whose EFFECTIVE budget is ~86 MiB after
proportional scaling, and N concurrent scans re-touching keys promote
them past the probationary queue, flushing the point-lookup working set;
(4) at 90% RSS the watchdog's only lever is `clear_shared_caches()` —
nuking ALL tiers repeatedly while the scans keep allocating. Shipped
now: a process-wide **scan admission gate** — plans containing a
`NodeScan` acquire one of `NAMIDB_MAX_CONCURRENT_SCANS` permits
(default 4, `0` disables) before execution, on HTTP (both tenants) and
Bolt (auto-commit and in-tx); index lookups and expand chains pass free;
waiters queue fairly inside the request-timeout layer. Worst-case scan
memory drops from `1024 x largest-label` to `permits x largest-label`
and indexed traffic keeps its cache working set. Pinned by
`namidb-server/tests/scan_gate.rs` (8 concurrent full scans through one
permit all answer exactly, non-scan traffic flows meanwhile). Remaining
(recorded, sized): streaming `NodeScan` execution (the deep fix for
per-scan RSS), stats-based per-scan memory reservation through the
existing `reserve_query_headroom` CAS ledger (`LabelStats.node_count` x
estimated row bytes), scan-resistant insertion across the body/decoded/
page tiers (a first bypass attempt showed sweeps traverse the paged
sidecar and range-cache routes too — a per-route audit must precede any
policy change), and reclaim hysteresis so the watchdog stops
re-clearing caches it just cleared.

### 42. [AUDITED — product surface, backlog corrected] Index surface limits observed in the field: single-property indexes only (no composites), DDL must be a single statement, and vector/full-text are compile-time features the cloud build does not ship.

Recording as product backlog rather than defects. Composite indexes and
multi-statement DDL scripts are roadmap-sized; the cloud build's missing
`vector-index`/text features is a packaging decision to revisit before
any customer needs search on the managed tier.

**Audit (2026-08-17)** — two of the three reports needed correction:
- Composite UNIQUE CONSTRAINTS already work end-to-end (parser accepts
  `(n.a, n.b)`, `create_unique_constraint_named` validates existing data
  with a tuple scan and enforces on write via the transactional unique
  index's tuple `encode_probe_key`) — they are enforcement-only, with no
  read acceleration. What is actually missing: composite `CREATE INDEX`
  (the parser accepts exactly one property; `parse_create_index` errors
  on a comma) and a persisted tuple posting sidecar
  (`EqualityKeyEncoding::TupleV1` + flush/compaction harvesters +
  read-side tuple probe + planner conjunct-cover detection). Sized at
  ~3-4 weeks; roadmap.
  **Follow-up shipped (2026-08-28, unreleased):** composite `CREATE
  INDEX` is end-to-end. Schema DDL (`Schema::indexes` / `IndexDef`,
  every surface incl. `SHOW INDEXES` and the CLI), TupleV1 posting
  sidecars harvested at flush and re-emitted by compaction (whose
  migration predicate makes DDL-triggered sweeps backfill pre-existing
  SSTs), `Snapshot::indexed_node_ids_by_property_tuple` with the
  freshness/authoritative-or-decline contract, and the
  `NodeByPropertyTuple` planner rewrite (declaration-order
  canonicalization, unique-conjunct priority, residual filters, exact
  scan fallback). One deliberate semantic upgrade over the single-
  property route: TupleV1 canonicalizes numeric members to f64 keys and
  confirms with Cypher-coercing equality, so `30 = 30.0` matches
  through the index — the executor's `is_equal` coerces, and typed keys
  would have broken index/scan parity. Route observability:
  `tuple_native`/`tuple_fallback` counters +
  `namidb_tuple_lookup_route_total` on `/metrics`.
- The OFFICIAL artifacts all ship `vector-index,text-index`: release
  binaries (release-binaries.yml `features:` line), the Docker image
  (Dockerfile bakes them), and the wheels (pyproject maturin features).
  The field report's "cloud build lacks them" matches a FROM-SOURCE
  build: the README's `cargo install --path crates/namidb-server` had no
  `--features` flag, producing a server that 400s on search DDL — the
  README now includes the flags.
- Multi-statement DDL confirmed absent: the parser is one-statement
  (`expect_eof`; a trailing `;` is tolerated, a second statement errors)
  and there is no batch endpoint. Cheapest useful step is client-side
  splitting in namidb-cli (~hours); server-side scripts (sequential,
  stop-on-first-error, per-statement writer lock) are ~2-4 days.
  **Follow-up shipped:** `namidb run` splits `;`-separated scripts
  (string/backtick/comment-aware state machine), runs them sequentially
  on one session with stop-on-first-error, and intercepts
  CREATE CONSTRAINT / CREATE INDEX as schema commands — the CLI could
  not run DDL at all before. A server-side script endpoint stays
  unplanned: HTTP callers can split client-side with the same rules,
  and per-statement requests keep the error surface per statement.

## Second field report (2026-08-27) — contactability platform, 24.6k-node graph, Bolt via official Neo4j driver

Twelve engine limits reproduced against the running 2.1.4 instance (NDB-01..12), four
self-retracted complaints (RET-01..04). Recon corrected the mechanism of four before any
fix landed. Items 43–54, one per NDB finding.

### 43. [DONE — fail-fast in 2.2.0; full Bolt multi-tenant in 2.3.0] NDB-01: `--multi-tenant` silently never starts Bolt.

Confirmed exactly as reported: the multi-tenant serve path `return serve_http(...)`
(lib.rs) executes before the only `config.bolt_listen` consultation, no warning. This is
architectural (bolt::serve takes a single-namespace AppState; RFC-022 lists
multidatabase as a non-goal). Fix: fail-fast at boot when both flags are set, in the
validation block of `run_with_memory_max_bytes` (covers binary + embedded callers) +
HTTP-only caveats at every doc site that lacked them (README flags table and Bolt
paragraph, server README auth/Bolt sections, ARCHITECTURE multi-tenancy — the dedicated
multi-tenancy.md guide already documented it). Full Bolt-in-multitenant is sized
roadmap: plumb the Bolt 5 `db` field of RUN/BEGIN `extra` through the namidb-bolt
Backend trait (currently discarded), per-namespace WriterSession from the registry, tx
pinned to one namespace, `principal_for_in` at first RUN/BEGIN — roughly 3-5 days.

**2.3.0 addendum — full Bolt multi-tenant shipped.** Route resolution went where
drivers actually send the database — the `db` field of RUN/BEGIN extra, not HELLO —
plumbed through the namidb-bolt Backend trait as defaulted `*_on` methods (zero churn
for embedders/test backends). The server side is a wrapper (MultiTenantBackend) that
resolves the namespace per statement (scope re-check via principal_for_in, registry
get_or_open for the CURRENT incarnation, AppState::for_namespace Arc-view) and
delegates to an untouched single-namespace ServerBackend; an explicit tx pins one
delegate from BEGIN to COMMIT/ROLLBACK. The 2.2.0 startup bail is gone. Validated
end-to-end with the official Python driver: session(database=...) isolation, pinned
tx, and Security.Forbidden outside a token's scope.

### 44. [DONE — lands in 2.2.0] NDB-02: no token → server boots open, warn-only.

Confirmed: `(None, None) => AuthConfig::open()` + `Principal::anonymous_rw()` on every
HTTP request and `AuthPolicy::Open` on Bolt. Fix: boot REFUSES to start with no auth
source unless `--no-auth` / `NAMIDB_NO_AUTH=1` is passed explicitly (check sits at the
`is_open()` point so JWT-only configs keep booting). Breaking change, release-noted.

### 45. [DONE — taxonomy in 2.2.0; wire GQLSTATUS in 2.3.0] NDB-03: error taxonomy collapse.

Recon corrected two details: `50N42` is a driver-side polyfill (Bolt ≤5.4 carries no
GQLSTATUS; the string appears nowhere in the repo), and HTTP already distinguished
timeout (504) from row cap (413). The real defect was Bolt-only: `map_exec_err`
substring-bucketed every non-constraint error, landing Timeout and RowCap in
`Statement.ArgumentError`. Fix: typed `BackendError::Timeout` / `ResourceLimit`
(ClientError on purpose — drivers auto-retry TransientError.* only, and a deterministic
budget re-run fails identically); deterministic search caps moved out of the
auto-retried TransientError bucket; HTTP bodies now carry `neo4j_code` + `gql_status`
per family (table in the server README); eval errors are 400 not 500. Bolt ≥5.7
GQLSTATUS-on-the-wire: roadmap (~2-3 days: negotiate 5.7, extend FAILURE metadata).

**2.3.0 addendum:** Bolt 5.7 negotiated; on >= 5.7 sessions every FAILURE carries
gql_status/description/diagnostic_record AND neo4j_code — 5.7 renamed the code key,
and the official Python driver reads only the new name (found live-testing: without
it the driver fell back to UnknownError). All failures funnel through write_failure,
so one version check upgraded every site; TELEMETRY (5.5) was already handled.
Live-validated with the official driver: timeout => TransactionTimedOut + gql 57014
(previously the 50N42 polyfill), non-retriable.

### 46. [DONE — BFS in 2.2.0; ORDER BY sandwich + in-loop budgets in 2.2.1] NDB-04: var-length traversal enumerates all paths under DISTINCT+LIMIT.

Confirmed end to end: only shortest-mode had visited pruning; `execute_capped`'s
whitelist excluded `Project{distinct:true}`, so DISTINCT+LIMIT got zero pushdown; the
cap was honoured only at seed boundaries. Fix (executor-side shape detection, no plan
field): `try_execute_endpoint_distinct_project` intercepts `Project{distinct:true}`
directly over a var-length Expand when every projected expression reads ONLY the
endpoint binding — then the deg^hop trail enumeration collapses to a visited-set BFS
(each node expanded once per seed, emitted once globally across seeds), and the LIMIT
budget crosses the distinct projection exactly (globally-deduped emission makes a
capped prefix valid). Intercepted in both the flat and factor executors + the capped
path. Regression proof: dense-layered 30^5-walk graph completes; parity vs the
per-seed route asserted on memtable + flushed routes, including the seed-reachable-
through-a-cycle endpoint. Not covered (falls back to the exhaustive route, correct but
slow): a Filter between Expand and DISTINCT, projections referencing the seed or rel
list, `min >= 2`.

**2.2.1 addendum (found smoke-testing the released 2.2.0 binaries).** The bare-ORDER-BY
lowering places `TopN{keys}` BETWEEN the distinct projection and the Expand, so
`RETURN DISTINCT b.x ORDER BY b.x` missed the BFS eligibility and hung on the released
binary — invisible to the 2.2.0 tests because they executed `lower()` output while the
server executes `optimize()` output (the same class of trap as the index-reachability
rule: test the pipeline the server runs). Three fixes in 2.2.1: (a) eligibility looks
through the no-skip/no-limit `TopN{keys}` sandwich when the keys also read only the
endpoint, re-applying the sort to the BFS output; (b) `dedup_rows` is now
order-preserving first-occurrence — the old sort-by-fingerprint dedup silently
re-ordered its input ("I10;" < "I2;"), so `RETURN DISTINCT x ORDER BY x` returned
fingerprint order, a correctness bug PREDATING this work; (c) the deadline and row cap
now fire INSIDE a single seed's expansion (per-hop + every 4096 edges in both the flat
and factor executors) — before, one seed's deg^hop enumeration ran unbounded past the
30 s budget because both guards only probed at seed boundaries. The
exec_distinct_endpoints suite now runs every plan through `optimize()`.

### 47. [DONE — 2.2.0 gaps + 2.3.0 seed-grouping and trail fix] NDB-09: shortestPath blows the 1M row cap.

The claimed mechanism ("expands then trims") is REFUTED on 2.1.4: shortestPath lowers
onto the Expand with ShortestMode and the executor early-stops (BFS visited pruning
shipped in 666ef82, ancestor of v2.1.4; regression test covers the 40^5 shape). What
CAN reproduce the symptom: all-pairs seeding (`MATCH (a:L), (b:L)` → CrossProduct
row-cap at 24k² pairs — correct behaviour, the query requests 576M rows),
allShortestPaths' combinatorial same-length output, `*2..N` (pruning was disabled for
min ≥ 2 → exhaustive frontier → hang), and the silently-accepted unbound endpoint
(First mode returned ONE arbitrary endpoint per seed — an openCypher divergence that
pushed users to the plain var-length rewrite that genuinely blows the cap). Fixed in
2.2.0: (a) `(node, hop)` frontier dedup for First-mode `min >= 2` (placed after the
trail check so a rejected walk cannot consume the slot; documented theoretical corner:
the kept walk could be trail-blocked where a dropped one was not — accepted for "some
shortest path" output vs the previous effective hang); (b) unbound endpoints now error
at lowering with the previously-bound rule spelled out. Deferred (sized): seed-grouping
(one BFS per distinct source serving all its bound targets, 1-2 days) and bidirectional
meet-in-the-middle for single pairs.

**2.3.0 addendum:** seed-grouping shipped — consecutive seed rows sharing a source
(the CrossProduct's row-major pairs) run ONE multi-target BFS
(execute_expand_shortest_grouped): per-target first-reached-level bookkeeping
preserves First (one row per pair) and All (every same-level arrival) semantics,
row order preserved, engaged only when uncapped so LIMIT-pushdown prefix semantics
stay untouched. Implementing it surfaced ANOTHER pre-existing trail bug: nodes(p)
filled every intermediate hop with the pre-bound endpoint value (a 2-hop path
returned ["n0","n2","n2"]) — fixed in both the grouped and per-seed paths by looking
up the node actually reached. Bidirectional meet-in-the-middle shipped next:
single bound pairs in First mode expand the smaller frontier from either end
(reverse side flips the pattern direction), with first-arrival parent maps for
reconstruction, the classical df+db >= best stopping criterion for exactness, and
a route-telemetry counter so tests assert the route actually engages (parity alone
would pass on the unidirectional route too).

### 48. [DONE — lands in 2.2.0] NDB-05: reduce() does not parse.

Confirmed: no production existed; `acc = 1.0` parsed as Eq inside generic call args and
died at `|`. Fix: soft-keyword production (lookahead `reduce(<ident> =` keeps
`reduce` usable as a variable/function name), fold evaluation mirroring
eval_list_comprehension, arms added at every exhaustive ExpressionKind match
(display round-trip, optimizer alias collection, projection pushdown, lower walkers,
factor sink variable collection — compiler-enforced, no wildcard traps found).

### 49. [DONE — lands in 2.2.0] NDB-10: no statistical aggregates.

Confirmed: closed 6-entry registry in try_aggregate. Added stdev/stdevp (two-pass,
n-1 / n denominators, single value → 0.0, empty → NULL, numeric-only typed error) and
percentileCont/percentileDisc (two-arg, p validated in [0,1], linear interpolation vs
nearest-rank keeping input type), collect-then-compute like Collect. NULL-skipping,
DISTINCT, grouping asserted on memtable + flushed routes.

### 50. [DONE — lands in 2.2.0] NDB-11: `'texto' + {a:1}` leaks Debug repr.

Confirmed and broader than reported: every non-scalar (map, node, rel, list, bytes,
vector, path) stringified through `format!("{:?}")` when added to a string, and
toString() shared the helper. Fix: scalar-only rendering (ints, floats, bools, strings,
temporals as ISO-8601); string + structural value is now the `cannot apply + between
STRING and MAP` type error; list `+` gains real Cypher semantics (list+list concat,
list+element append, element+list prepend — previously `'a' + [1]` produced garbage
text); toString(non-scalar) errors. Behavior change, release-noted.

### 51. [PARTIALLY ADDRESSED — auto-commit group commit in 2.3.0; explicit-tx lock unchanged by design] NDB-06: explicit transactions hold the writer lock through think-time.

Accurate by design today (single writer per namespace; BEGIN holds it to COMMIT,
NAMIDB_BOLT_MAX_TX_LIFETIME caps it). RFC-034's own resolution for explicit
transactions stands: they remain a serialized, think-time-bounded path on purpose —
group commit targets the auto-commit paths (shipped in 2.3.0, see item 52). The
per-tenant mitigation stays "keep transactions short / make the graph rebuildable",
which the field team had already adopted.

### 52. [DONE — group commit lands in 2.3.0, opt-in window] NDB-07: serialized commit floor ~309 nodes/s at batch 1.

Accurate: ~3 ms/commit = two object-store round trips, so throughput scales only with
batch size (their own table shows 63.9k nodes/s at batch 5000). Not a defect of the
ingest path (which batches by design); interactive single-row writes need group commit.

**2.3.0 addendum — group commit shipped (RFC-034 "many stagers, one committer").**
Storage grew request scopes over the staged batch (begin/commit/rollback_staged_request:
mark-based truncation, RYOW-overlay rebuild by replay, and a request-scoped unique-index
undo layer whose merge preserves first-touch pre-batch values — the subtle
A-moves-tuple-then-B-touches-it case is regression-tested). The server runs one
committer per namespace: requests stage under the writer lock, register a waiter keyed
by their last staged LSN while still holding the lock (no stage/commit race), and ACK
only after the merged commit is durable and the snapshot republished. Wired into the
single-tenant HTTP and Bolt auto-commit paths; the multi-tenant follow-up shipped next:
the registry spawns one committer per namespace (cancelled on eviction), the window
reaches MaintenanceConfig, and both the multi-tenant HTTP arm and the Bolt
for_namespace views group. Default window 0s = inline commits, bit-for-bit
today's behaviour — the knob is the rollout kill-switch RFC-034 asked for. Tests follow
the RFC's plan: commit coalescing proven by manifest-version counting, concurrent-MERGE
isolation, solo rollback of a failed statement, Bolt-path ACK+RYOW. Fault-injected CAS
failure (shared fate) remains a test-harness follow-up; the error path reuses the
inline path's discard+recover machinery.

**Follow-up addendum — shared-fate test built, and it found a PRE-EXISTING
durability bug.** Injecting a pointer-CAS transport failure showed NACKed rows
RESURRECTING: the orphan WAL+body dangled, and repair_stalled_commit (correct for
crashes) later published them. Fixed in storage: resolve_failed_pointer_cas —
adopt-if-landed (version + fence writer_id + segment xxh3 ownership proof),
delete-orphans-if-definitively-absent (the NACK stays true), poison-with-honest-
indeterminate-message only when the store keeps failing. The inline commit path
had the same hazard since forever; the group-commit test is what surfaced it.

### 53. [DONE — lands in 2.2.0] NDB-08: Docker image crash-loops on a named volume at /var/lib/namidb.

Confirmed, with a nuance: /var/lib/namidb is not a coded default (--store is required)
but it IS the canonical example path in --help, storage error text, and every file://
doc example. The image now pre-creates it owned by uid 65532 so fresh named volumes
inherit ownership (copy-on-first-use); docs note the one-time `chown -R 65532:65532`
for bind mounts and volumes created by older images. docker-compose.yml was never
affected (S3/MinIO store).

### 54. [DONE — docs in 2.2.0; endpoint lands in 2.3.0] NDB-12: backup/restore CLI-only.

"No admin HTTP endpoint" confirmed. But the implied offline-only limitation is wrong
for backup: `copy_namespace_snapshot` is live-safe by design (durable retention-pin
lease honoured by the janitor; the residual race fails loudly with NotFound instead of
truncating; committed-but-unflushed WAL captured). The CLI help said "run against a
quiescent source" — stale, now corrected. Restore genuinely requires an offline
destination (no fencing). The POST /v0/admin/backup endpoint is deferred deliberately:
a client-supplied destination URI is an SSRF/exfiltration vector using the server's
ambient cloud credentials, so it needs an operator-configured target allowlist
(NAMIDB_BACKUP_TARGET_URI prefix) plus the admin-flush single-flight/bounded-wait
pattern — ~1 day, sized, not rushed into this release.

**2.3.0 addendum:** POST /v0/admin/backup shipped with the design above: disabled
(403) unless --backup-target-uri / NAMIDB_BACKUP_TARGET_URI allowlists destinations
(boundary-aware prefix match, so `.../namidb-evil` cannot ride `.../namidb`);
read-write role gate; single-flight with the 30 s bounded wait; Precondition→409,
local-persistence→507; live round-trip + force/verify covered by in-lib tests.
Restore stays CLI-only on purpose (offline destination required; the serving
process IS the writer). The multi-tenant twin shipped next:
/:namespace/v0/admin/backup with the same allowlist, process-wide single-flight,
and cold-namespace-safe source resolution (the copy reads committed state, so
no writer opens).

### Retractions (RET-01..04) — recorded for the record

RET-02 matters to us: their docs claimed NamiDB needs If-Match and discarded Garage on
that basis. Our own crates/namidb-storage README had the SAME stale claim ("If-Match,
If-None-Match, ETag") — now corrected everywhere: the commit path needs exactly ONE
conditional-write primitive, PUT-if-absent (`If-None-Match: *`, RFC-029); README gained
an object-store requirements note (an "S3-compatible" that ignores the precondition
cannot host NamiDB safely). RET-01/03/04 need no engine action.

## Third field report (2026-08-29) — four minor findings + one unreproduced

Recon-before-fix, as with the first two cycles (and again worth it: two of
five mechanisms as reported were wrong, one "unreproducible" turned out to
be a confirmed design gap).

### 55. [CORRECTED — fixed in 2.5.0] Bolt message cap "small, shape-dependent, and moves between runs"

Real failure, wrong mechanism. Not the boot-logged budget
(message_memory_budget_bytes is a different guard with a different error):
the per-message DecodeBudget allowed 64 KiB + 8x the message's own wire
bytes of estimated decoded heap. Tiny one-key row maps amplify at ~13x
heap-to-wire (128 B/map entry + 32 B/list slot + payload on ~13 wire
bytes), so the base absorbed the deficit for exactly ~889 rows; 4-field
rows (~6x, not "<20/row" — they cost ~650 est. bytes but ride under the
8x ratio) passed at any count. "Moves between runs" = the limit tracks
each message's own size (161200 = 65536 + 8x11958; 111936 = 65536 +
8x5800) — deterministic, no live state. Fix: base -> 2 MiB as a
documented client contract (any message whose estimated decoded heap fits
2 MiB decodes regardless of shape; ~11k tiny rows), shared semaphore base
in lockstep, and the rejection now names estimated bytes / limit / wire
size / formula. Amplification stays rejected; require_minimum_wire
already forces real bytes per declared entry.

### 56. [CONFIRMED — shipped in 2.5.0] No way to list or cancel an in-flight query

Tracking today is one AtomicI64 gauge; the slow-query log is post-hoc.
The useful discovery: cooperative cancellation PLUMBING already exists
end-to-end — a task-local deadline polled at ~75 executor/storage
chokepoints (writes included) plus Bolt's RunCancellation — it is just
not operator-reachable. Fix: widen the cancel task-local to
{deadline, flag}, a guard-based QueryRegistry at the four existing
in-flight chokepoints, GET /v0/admin/queries + POST
/v0/admin/queries/:id/cancel (auth + write-role gated; statements are
operator-visible), cancellation surfacing as an executor error so the
existing discard/recovery path handles cleanup.

### 57. [CORRECTED — fixed in 2.5.0] --multi-tenant rejects 1; ns "named in two sites"

Env bool parse confirmed exactly (clap bare-`action` SetTrue =>
BoolValueParser on env values; --no-auth and --sweep-delete shared it,
and our own docs instructed NAMIDB_NO_AUTH=1). But the ns half was
WORSE than reported, not a duplicate name: in multi-tenant mode ?ns=
was parsed, validated, and DISCARDED — an operator writing ?ns=main
silently got fallback namespace "default" (the flag, not the URI, was
always the source). Fixed: BoolishValueParser on the three flags
(1/0/yes/no/on/off; junk stays a hard error so nothing coerces
--no-auth), parse_store() makes ?ns= optional in multi-tenant boot with
a warning when a dead one is present.

### 58. [CONFIRMED — shipped in 2.5.0] Auth tokens file has no hot reload

Loaded once at boot, frozen in an Arc cloned by HTTP, Bolt, and shared
state — so the swap must be interior to AuthConfig (an Arc replacement
would miss the clones). Fix follows the in-repo JWKS precedent
(jwt.rs spawn_refresh): content-compare poll task, atomic swap of the
token set only, fail-to-last-good on malformed files (a reload can
never widen access or flip auth open). To document: revocation does not
kill live single-tenant Bolt sessions (LOGON-cached principal);
multi-tenant Bolt re-resolves per statement.

### 59. [CONFIRMED — shipped in 2.5.0, one hazard deferred] 24.5-minute write with status=ok under 30s timeouts (reporter could not reproduce)

Confirmed as a design gap without needing the repro: NEITHER timeout
ever covered the durability tail. --writer-lock-timeout bounds only the
writer-mutex wait; --write-timeout becomes a deadline armed around plan
STAGING only — writer.commit_batch() (WAL PUT + manifest body PUT +
pointer CAS + orphan-WAL re-seq + CAS resolution) runs after the
deadline scope on every foreground path, riding object_store client
defaults (S3: up to 180s retry budget x10 per op; file://: no timeout
at all). HTTP cannot log the signature (120s TimeoutLayer drops the
handler); Bolt has no outer bound — lock <=30s + staging <=30s + commit
unbounded, elapsed includes it: the reporter's write was a Bolt write
through a store stall that eventually succeeded. The 24-min-lock-wait
theory is refuted (lock wait is capped; maintenance convoys produce
503s, not slow oks). Fix plan: probe cancellation at commit_batch's
determinate boundaries (pre-PUT, pre-retry — never between WAL PUT and
CAS), move commit inside the deadline scope (HTTP) and wrap it (Bolt),
explicit store client timeouts, bound the group-commit ack wait with an
honest indeterminate error, and slow-log lock/staging/commit
sub-durations. Adjacent hazard recorded: the HTTP TimeoutLayer can
cancel a commit mid-durability at 120s — to bound, not drop.
**Follow-up shipped (2.5.0):** all of the plan above except the
TimeoutLayer hazard (deferred, recorded here): determinate-boundary
probes in commit_batch (pre-PUT and pre-orphan-retry only; the
PUT-to-CAS zone stays uninterruptible), deadline scope over the HTTP
commit and re-armed around both Bolt commits, bounded group-ACK wait
with the outcome-unknown error, >10s durability-tail warn, and opt-in
store client bounds via NAMIDB_STORE_{REQUEST_TIMEOUT,RETRY_TIMEOUT,
MAX_RETRIES}. The admin cancel flag (item 56) rides the same probes, so
a stuck-but-probing commit is also operator-killable pre-PUT.
