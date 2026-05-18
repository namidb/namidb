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
        ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 65532 --user-group --no-create-home namidb

COPY --from=builder /build/target/release/namidb-server /usr/local/bin/namidb-server

USER namidb:namidb
EXPOSE 8080

ENV NAMIDB_LISTEN=0.0.0.0:8080 \
    RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/namidb-server"]
