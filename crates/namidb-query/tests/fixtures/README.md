# LDBC SNB Interactive Complex Q1–Q14 fixtures

Queries adapted from the **LDBC Social Network Benchmark — Interactive
Workload**, Complex Read queries, version v0.4. Each `ic_*.cypher` file holds
one canonical query.

These are used by `tests/parser_ldbc_snb_interactive.rs` to validate that
the v0 parser (RFC-004) accepts the 12 in-scope queries and explicitly
rejects (or marks ignored) the two out-of-scope ones that require
`shortestPath` / `allShortestPaths`.

The fixtures are **parser-level only** — semantics, execution and
performance are gated by future milestones.

References:
- LDBC SNB Specification v0.4 — https://ldbcouncil.org/ldbc_snb_docs/
- Erling et al., *The LDBC Social Network Benchmark*, SIGMOD 2015.
