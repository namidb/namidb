#!/usr/bin/env bash
#
# Run the storage benchmarks against Cloudflare R2 (real WAN, not
# LocalStack). The bench harness uses `object_store`'s S3 client, which
# R2 implements; the only differences from the LocalStack setup are
# the endpoint URL, the region literal ("auto"), and the absence of
# `AWS_ALLOW_HTTP` (R2 is HTTPS).
#
# Required env vars (the script will refuse to run without them):
#
#   R2_ACCESS_KEY_ID       — R2 API token Access Key ID
#   R2_SECRET_ACCESS_KEY   — R2 API token Secret Access Key
#   R2_ACCOUNT_ID          — Cloudflare account id (the prefix of the
#                            r2.cloudflarestorage.com endpoint)
#   R2_BUCKET              — bucket to use (must already exist)
#
# Optional:
#
#   BENCH_NODES, BENCH_BATCH_COMMIT, BENCH_FIXTURE_NODES,
#   BENCH_DURATION_SECS  — usual bench knobs, defaults from each bench.
#
# Usage:
#
#   R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… R2_ACCOUNT_ID=… \
#   R2_BUCKET=namidb-bench \
#       tests/run-bench-r2.sh [smoke|full]
#
# `smoke` (default): one short run of read_latency 1M to validate
# conditional-write semantics and measure cold/warm against the WAN.
# `full`: smoke + ingest_throughput + parquet_ingest + recovery_replay
# + concurrent_mix.

set -euo pipefail

MODE="${1:-smoke}"

require() {
    local var="$1"
    if [[ -z "${!var:-}" ]]; then
        echo "error: $var is required" >&2
        exit 1
    fi
}

require R2_ACCESS_KEY_ID
require R2_SECRET_ACCESS_KEY
require R2_ACCOUNT_ID
require R2_BUCKET

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_ENDPOINT_URL="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_REGION="auto"
export NAMIDB_TEST_BUCKET="$R2_BUCKET"
# DO NOT set AWS_ALLOW_HTTP — R2 is HTTPS only.

export BENCH_STORE="s3"

echo "=================================================================="
echo " R2 bench wrapper"
echo "=================================================================="
echo " endpoint : $AWS_ENDPOINT_URL"
echo " bucket   : $R2_BUCKET"
echo " mode     : $MODE"
echo "=================================================================="

run_bench() {
    local name="$1"
    shift
    echo
    echo "--- running $name $* ---"
    cargo bench -p namidb-storage --bench "$name" "$@"
}

case "$MODE" in
smoke)
    # 1M nodes — validates conditional-write CAS + gives one cold/warm
    # data point. ~3-5 min depending on RTT.
    BENCH_NODES="${BENCH_NODES:-1000000}" \
    BENCH_BATCH="${BENCH_BATCH:-1000000}" \
    BENCH_CACHE_MB="${BENCH_CACHE_MB:-64}" \
        run_bench read_latency
    ;;
full)
    # Same as smoke first, then the four storage harnesses at 1M nodes /
    # 100K-node mix fixture. Allow up to ~30 min wall time.
    BENCH_NODES="${BENCH_NODES:-1000000}" \
    BENCH_BATCH="${BENCH_BATCH:-1000000}" \
    BENCH_CACHE_MB="${BENCH_CACHE_MB:-64}" \
        run_bench read_latency

    BENCH_NODES="${BENCH_NODES:-1000000}" \
    BENCH_BATCH_COMMIT="${BENCH_BATCH_COMMIT:-10000}" \
        run_bench ingest_throughput

    BENCH_NODES="${BENCH_NODES:-1000000}" \
    BENCH_BATCH_COMMIT="${BENCH_BATCH_COMMIT:-10000}" \
        run_bench parquet_ingest

    BENCH_SEGMENTS="${BENCH_SEGMENTS:-10}" \
    BENCH_RECORDS_PER_SEGMENT="${BENCH_RECORDS_PER_SEGMENT:-10000}" \
        run_bench recovery_replay

    BENCH_FIXTURE_NODES="${BENCH_FIXTURE_NODES:-100000}" \
    BENCH_DURATION_SECS="${BENCH_DURATION_SECS:-15}" \
    BENCH_BATCH_COMMIT="${BENCH_BATCH_COMMIT:-10000}" \
        run_bench concurrent_mix
    ;;
*)
    echo "error: unknown mode '$MODE' (want 'smoke' or 'full')" >&2
    exit 1
    ;;
esac

echo
echo "=================================================================="
echo " R2 bench wrapper done"
echo "=================================================================="
