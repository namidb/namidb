# syntax=docker/dockerfile:1

# ---- build ----
FROM rust:1.85-slim AS build
WORKDIR /src
COPY . .
RUN cargo build --release --bin namidb-server

# ---- runtime ----
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/namidb-server /usr/local/bin/namidb-server

# REST API on 8080. Set NAMIDB_BOLT_LISTEN=0.0.0.0:7687 to also expose Bolt.
# NAMIDB_STORE is deployment-specific (e.g. s3://bucket/prefix or a path) and
# must be supplied at run time.
EXPOSE 8080
ENV NAMIDB_LISTEN=0.0.0.0:8080

# Liveness: /v0/livez takes no lock and reads no namespace state, so a busy
# engine (a long write or compaction holding the writer lock) still answers,
# and the server gets a SIGTERM-clean drain on `docker stop`.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8080/v0/livez || exit 1

ENTRYPOINT ["namidb-server"]
