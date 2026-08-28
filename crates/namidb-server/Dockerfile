# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Pin the multi-architecture manifest lists, not an architecture-specific child:
# BuildKit still selects linux/amd64 or linux/arm64 while every release rebuild
# consumes byte-identical base manifests.
# ─── builder ───────────────────────────────────────────────────────────
FROM rust:1.85.1-slim-bookworm@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*

# Copy the workspace manifest + every crate's Cargo.toml first so the
# dependency graph caches independently of source changes.
# Do not copy the workspace `rust-toolchain.toml` here: it intentionally tracks
# local `stable`, which would make rustup replace this pinned release compiler
# during every image build.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build WITH vector + text search so `CREATE VECTOR INDEX` / `CREATE FULLTEXT
# INDEX` and index-backed KNN/BM25 work in the image out of the box (matching the
# prebuilt `namidb-server` release archive). Omit the features for a smaller image.
RUN cargo build --locked --release -p namidb-server --features vector-index,text-index \
    && strip target/release/namidb-server

# ─── runtime ───────────────────────────────────────────────────────────
FROM debian:bookworm-slim@sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818

ARG NAMIDB_VERSION=dev
ARG NAMIDB_REVISION=unknown

LABEL org.opencontainers.image.title="NamiDB Server" \
    org.opencontainers.image.description="Cloud-native graph database server on object storage" \
    org.opencontainers.image.url="https://namidb.com" \
    org.opencontainers.image.source="https://github.com/namidb/namidb" \
    org.opencontainers.image.documentation="https://github.com/namidb/namidb/tree/main/crates/namidb-server" \
    org.opencontainers.image.vendor="NamiDB, Inc." \
    org.opencontainers.image.licenses="BUSL-1.1" \
    org.opencontainers.image.version="${NAMIDB_VERSION}" \
    org.opencontainers.image.revision="${NAMIDB_REVISION}"

# /var/lib/namidb has no in-image consumer: it is the canonical `file://`
# store path from the docs, pre-created and owned by uid 65532 so a fresh
# named volume mounted there inherits that ownership (Docker copy-on-first-use)
# instead of being created root-owned and crash-looping the non-root server.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 65532 --user-group --no-create-home namidb \
    && install -d -o namidb -g namidb -m 0700 /var/tmp/namidb-spool \
    && install -d -o namidb -g namidb -m 0700 /var/cache/namidb-ranges \
    && install -d -o namidb -g namidb -m 0700 /var/lib/namidb

COPY --from=builder /build/target/release/namidb-server /usr/local/bin/namidb-server
COPY --chmod=0555 crates/namidb-server/docker-healthcheck.sh /usr/local/bin/namidb-healthcheck
COPY LICENSE /usr/share/licenses/namidb/LICENSE

USER namidb:namidb
EXPOSE 8080

# Bound glibc's per-thread allocation arenas. Large FTS/vector decodes and
# compactions otherwise leave many 64/128 MiB arenas resident after their
# temporary working set is dropped, obscuring the engine cache budget and
# increasing OOM risk during long-running bulk loads.
ENV NAMIDB_LISTEN=0.0.0.0:8080 \
    NAMIDB_SPOOL_DIR=/var/tmp/namidb-spool \
    NAMIDB_INDEX_BUILD_SPOOL_DIR=/var/tmp/namidb-spool \
    NAMIDB_SEARCH_WORKSPACE_MAX_BYTES=268435456 \
    NAMIDB_SEARCH_MAX_RESULT_BYTES=67108864 \
    NAMIDB_BM25_MAX_DOCUMENT_BYTES=1048576 \
    NAMIDB_EDGE_DECODE_MAX_BYTES=67108864 \
    RUST_LOG=info \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=131072

# Liveness: /v0/livez takes no lock and reads no namespace state, so a busy
# engine (a long write or compaction holding the writer lock) still answers.
# The helper follows NAMIDB_LISTEN and switches to local, certificate-agnostic
# HTTPS when NAMIDB_TLS_CERT / NAMIDB_TLS_KEY enable TLS.
HEALTHCHECK --interval=15s --timeout=4s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/namidb-healthcheck"]

ENTRYPOINT ["/usr/local/bin/namidb-server"]
