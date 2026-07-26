# syntax=docker/dockerfile:1.7

# ─── builder ───────────────────────────────────────────────────────────
FROM rust:1.85.1-slim-bookworm AS builder

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
FROM debian:bookworm-slim

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

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 65532 --user-group --no-create-home namidb

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
