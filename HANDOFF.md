# Handoff: Roadmap Progress

**Session:** 2026-06-18/19 (session 836fd4ad)
**Branch:** `feat/s3b-versioned-pointer`
**Method:** ultracode — 94-agent orchestration (68-agent adversarial review + 26-agent research/design) driving implementation.

## Test status (all green)

| Crate | Tests |
|-------|-------|
| namidb-storage | 304 |
| namidb-server | 31 (+ integration suites) |
| namidb-mcp | 20 |
| namidb-graph | 10 |

## Completed this session

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

### Item 08 PR1 — CALL/YIELD (L) — **needs rework (design verdict)**
Lexer/AST/grammar → CallProcedure logical node → ProcRegistry source operator
exposing `algo.wcc`/`algo.pagerank`. **Blocking holes:** a cfg strategy that
keeps `Clause::Call` always-present but gates the lower arm **will not compile**
(exhaustive match, no wildcard); P0 query-timeout is over-claimed (the kernels
are synchronous CPU loops with no `.await`, so the deadline only fires after
return); isolates dropped (build only calls `add_edge`, never enumerates nodes);
`neighbours_of` does not exist on Snapshot (real API is `out_edges`); soft-keyword
(Call/Yield) not reserved to avoid breaking `CALL` as an identifier. Use the
EXPLAIN soft-keyword precedent.

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
