# Pre-load runbook: validating the engine against the real bucket

Run this checklist against the customer's actual object-store account before
starting a large production load (the 25 TB engagement or anything of that
order). CI and the nightly soaks prove the engine on `InMemory` stores and
GitHub runners; this runbook is the only step that exercises the real WAN,
the real credential set, and the real memory ceiling of the serving
container. Nothing here is destructive: every step targets its own namespace
under the configured `root_prefix` and cleans up via the janitor.

## 0. Prerequisites

- A bucket (R2 or S3) reserved for validation, NOT the production data
  bucket. The load itself is idempotent, but keep blast radii separate.
- Credentials with read/write/list on that bucket only.
- A machine (or container) shaped like the production serving node — same
  memory limit, same core count.
- A release build of the workspace: `cargo build --release -p namidb-bench
  --features object-native` and the server image from
  `crates/namidb-server/Dockerfile`.

## 1. WAN storage benchmarks (`tests/run-bench-r2.sh`)

```sh
R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… R2_ACCOUNT_ID=… \
R2_BUCKET=namidb-validate tests/run-bench-r2.sh smoke
```

`smoke` validates conditional-write (CAS) semantics against the provider and
measures cold/warm read latency over the WAN. Then run the full matrix once:

```sh
… tests/run-bench-r2.sh full
```

which adds `ingest_throughput`, `parquet_ingest`, `recovery_replay` and
`concurrent_mix`. Record the numbers next to the engagement notes; regressions
against the previous engagement's numbers are a stop signal, not a curiosity.

For plain S3 the script's R2-specific endpoint wiring does not apply — export
`AWS_*` credentials and the standard `object_store` S3 env instead, and run
the same bench binaries directly.

## 2. Real-dataset load (`namidb-bench load-r2`, LDBC SNB SF1)

```sh
target/release/namidb-bench load-r2 \
  --scale 1 --bucket namidb-validate \
  --namespace bench-snb-sf1 --root-prefix tenants
```

SF1 materialises a genuine multi-million-edge social graph through the full
flush/compaction path against the remote store. The loader upserts, so a
re-run converges to the same state at a higher manifest version — verify that
by running it twice and comparing the reported `manifest_version` increments
and identical row counts.

## 3. Serving-node memory ceiling smoke

Start the server image with the production memory limit against the loaded
namespace, e.g.:

```sh
docker run --memory 4g --cpus 2 -e … namidb-server:candidate
```

and replay a mixed query set (KNN + BM25 + 2-hop traversals, in parallel at
the expected concurrency). The container must not be OOM-killed and the
`/metrics` memory-governor gauges must stay under the ceiling. A kill here
means the governor budgets need retuning BEFORE the load, not after.

## 4. Nightly soaks are green

Confirm the latest `nightly` workflow run passed, and that the
`search-lsm-soak` job actually executed the million-row lifecycle (its log
prints the row count). A `workflow_dispatch` run with `soak_rows` raised to
`5000000` before a very large engagement is cheap insurance.

## 5. Sign-off

All four sections green, numbers recorded → the engine is cleared for the
production load. Any red: stop, file the finding in
`docs/testing/25tb-readiness.md`, fix, and restart the checklist.
