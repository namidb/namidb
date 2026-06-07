# syntax=docker/dockerfile:1.7

# ─── builder ───────────────────────────────────────────────────────────
FROM rust:1.85-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*

# Copy the workspace manifest + every crate's Cargo.toml first so the
# dependency graph caches independently of source changes.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates

RUN cargo build --release -p namidb-server \
    && strip target/release/namidb-server

# ─── runtime ───────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 65532 --user-group --no-create-home namidb

COPY --from=builder /build/target/release/namidb-server /usr/local/bin/namidb-server

USER namidb:namidb
EXPOSE 8080

ENV NAMIDB_LISTEN=0.0.0.0:8080 \
    RUST_LOG=info

# Liveness: /v0/livez takes no lock and reads no namespace state, so a busy
# engine (a long write or compaction holding the writer lock) still answers.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/v0/livez || exit 1

ENTRYPOINT ["/usr/local/bin/namidb-server"]
