# Handoff: Roadmap Progress

**Session:** 2025-01-19 (session 836fd4ad)
**Branch:** `feat/s3b-versioned-pointer`
**Tests:** 302 passing in namidb-storage, 47 passing in namidb-server, 20 passing in namidb-mcp, 8 passing in namidb-graph

## Completed Work

### s3b - Versioned Pointer (RFC-029) ✅ SHIPPED
**Commit:** e5d48a7 "fix(storage): address adversarial review findings in RFC-029"

All 5 bugs from adversarial review fixed:
- ✅ Empty LIST misclassification (HEAD p0 before falling back)
- ✅ Overwrite restore leaves stale current.json
- ✅ Forward probe test coverage added
- ✅ MAX_PROBE increased 256→8192
- ✅ Forward probe gap-safety documented after GC

**Files:** manifest.rs, backup.rs, paths.rs, janitor.rs, local.rs, server/lib.rs
**Docs:** docs/rfc/029-versioned-pointer.md (new), docs/rfc/001-storage-engine.md (amended)

### Item 14 - Multi-tenant (COMPLETE) ✅ SHIPPED
**Commits:**
- 5415123 "feat(server): foundation for multi-tenant namespace registry"
- LATEST "feat(server): multi-tenant router wiring"

**Completed:**
- ✅ `registry.rs` with `NamespaceRegistry` and `NamespaceState`
- ✅ `shared.rs` with `SharedAppState` (process-wide state)
- ✅ Lazy WriterSession creation per namespace
- ✅ Idle eviction with timeout + cap
- ✅ Config/CLI flags for multi-tenant mode
- ✅ Multi-tenant router with namespace extraction from path (`/:namespace/v0/...`)
- ✅ Multi-tenant handlers (cypher_multi, health_multi, admin_flush_multi)
- ✅ Auth middleware for multi-tenant mode
- ✅ All 47 tests passing

**Usage:**
```bash
# Single-tenant mode (backward compatible)
namidb-server --store memory://mydb

# Multi-tenant mode
namidb-server --multi-tenant --store memory://
# Requests: /:namespace/v0/cypher, /:namespace/v0/health, etc.
```

**Deferred for next wave:**
- Per-namespace flush/compaction/orphan-sweep background tasks
- Multi-tenant routing tests (beyond basic unit tests)

### Item 13 - Hybrid Search (Layer A COMPLETE) ✅ SHIPPED
**Commit:** LATEST "feat(mcp): hybrid search with RRF fusion"

**Completed (Layer A - MCP-only, no engine change):**
- ✅ `hybrid_search` MCP tool combining lexical + semantic channels
- ✅ RRF (Reciprocal Rank Fusion) with formula: `weight / (k + rank)`, k=60
- ✅ Lexical channel: substring search in title/body with LIMIT 3*k
- ✅ Semantic channel: cosine KNN with LIMIT 3*k
- ✅ Configurable weights: `lexical_weight` and `semantic_weight` (default 1.0)
- ✅ Optional `where` pre-filter for metadata constraints
- ✅ All 17 tests passing

**Usage:**
```json
{
  "name": "hybrid_search",
  "arguments": {
    "query": "graph database",
    "k": 10,
    "lexical_weight": 1.0,
    "semantic_weight": 1.0,
    "where": "n.path STARTS WITH 'work/'"
  }
}
```

**Deferred (Layer B):**
- `bm25(text, query)` builtin in expr.rs
- Tantivy-backed full-text index
- `CREATE FULLTEXT INDEX` / `fulltext_search` tool

### Item 08 - Graph Algorithms (MCP wedge COMPLETE) ✅ SHIPPED
**Commit:** LATEST "feat(graph,mcp): WCC + PageRank graph algorithms"

**Completed (PR2 - MCP-only kernel, per audit recommendation):**
- ✅ `namidb-graph/src/algo.rs`: `Graph` in-memory adjacency structure
- ✅ WCC (Weakly Connected Components) via union-find (path halving + union by rank)
- ✅ PageRank via power iteration (dangling-node mass redistribution, L1 convergence)
- ✅ 8 unit tests in namidb-graph (all passing)
- ✅ MCP `graph_algorithm` tool: builds subgraph from Cypher, runs kernel, joins results
- ✅ 3 end-to-end tests in namidb-mcp (WCC, PageRank, error handling)
- ✅ Optional `edge_types` allowlist, configurable `damping`/`max_iterations`

**Usage:**
```json
{
  "name": "graph_algorithm",
  "arguments": {
    "algorithm": "wcc",
    "label": "Note",
    "edge_types": ["LINKS_TO", "EMBEDS"],
    "k": 10
  }
}
```

**Deferred (PR1 - CALL/YIELD Cypher surface):**
- CALL/YIELD clause in Cypher parser
- `CallProcedure` logical node
- `ProcRegistry` for named procedures

### "Now" Wave Status - ALL DONE ✅
- 02-set-plus-map ✅ Already implemented (`apply_set_map` exists, tests pass)
- 03-bulk-ingest ✅ Already implemented (`load_edges` exists)
- 09-uniqueness ✅ Already fixed (Python calls `enforce_node_unique_constraints`)
- 01-unwind-bind ✅ Already fixed (tests exist)
- 05-varlen-optional ✅ Already fixed (parser guard removed)
- 06-where-label ✅ Already implemented (`parse_postfix` handles `:`)
- 07-datetime-noarg ✅ Already implemented (`datetime()` with no args)
- 10-double-identity ✅ Already fixed (`_id` vs `id` separation)

**All "now" wave items from the audit are complete!**

## Next Wave Priority (from roadmap)

| Item | Severity | Effort | Status |
|------|----------|--------|--------|
| s3b-versioned-pointer | high | M | ✅ DONE |
| 14-single-writer | high | L | ✅ DONE |
| 13-hybrid-search | medium | M | ✅ Layer A DONE |
| 08-call-show-algos | medium | L | ✅ MCP wedge DONE |
| 15-auth-rbac | medium | L | Pending |
| 16-maturity | medium | M | Pending |
| 11-generic-500 | medium | S | Pending |
| 12-flat-vector (ANN) | high | L | Later wave |

## Remaining Work

### Item 13: Hybrid Search (medium/M)
- RRF (Reciprocal Rank Fusion) tool for combining multiple vector search results
- BM25 builtin for full-text search scores
- Integration with existing vector search capabilities

### Item 08: CALL/YIELD + Graph Algorithms (medium/L)
- CALL subsystem for invoking procedures
- YIELD for returning result sets
- Graph algorithms: WCC (Weakly Connected Components), PageRank

### Item 15: Auth/RBAC (medium/L)
- OIDC/JWT authentication
- Role-Based Access Control (RBAC)
- Policy Decision Point (PDP) hook

### Item 16: Maturity (medium/M)
- Backup CAS (Content-Addressable Storage)
- Restore fence

### Item 11: Generic 500 Errors (medium/S)
- Typed errors for "Unsupported kind" failures

## Commands for Next Agent

```bash
# Verify current state
git status
git log --oneline -5

# After item 14 complete, move to next priority:
# Item 13: hybrid-search (medium/M) - RRF tool + bm25 builtin
# Item 08: call-show-algos (medium/L) - CALL/YIELD subsystem
# Item 15: auth-rbac (medium/L) - OIDC/JWT + PDP hook
# Item 16: maturity (medium/M) - backup CAS + restore fence
# Item 11: generic-500 (medium/S) - typed errors

# Run tests
cargo test -p namidb-storage
cargo test -p namidb-server

# Run clippy
cargo clippy -p namidb-server -- -D warnings
```

---
Generated: 2025-01-19 (multi-tenant complete)
