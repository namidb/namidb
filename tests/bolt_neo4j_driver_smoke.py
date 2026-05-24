#!/usr/bin/env python3
"""Smoke test: connect the official `neo4j` Python driver to namidb-server.

Run flow:

    cargo build --release -p namidb-server
    NAMIDB_AUTH_TOKEN=test target/release/namidb-server \\
        --store memory://bolt-smoke \\
        --listen 127.0.0.1:8080 \\
        --bolt-listen 127.0.0.1:7687 &
    pip install neo4j
    python3 tests/bolt_neo4j_driver_smoke.py

Exit code 0 on success, non-zero on failure (with traceback). The
script does not arrange the server lifecycle on its own; that's done
by the developer or by an integration runner that wraps it. Keeping
the script lifecycle-free lets it double as a manual probe.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print(
            "neo4j driver not installed. Run `pip install neo4j` first.",
            file=sys.stderr,
        )
        return 2

    uri = os.environ.get("NAMIDB_BOLT_URI", "bolt://127.0.0.1:7687")
    token = os.environ.get("NAMIDB_AUTH_TOKEN", "test")

    print(f"connecting to {uri} with token={token!r}")
    driver = GraphDatabase.driver(uri, auth=("namidb", token))
    try:
        driver.verify_connectivity()
    except Exception as e:  # noqa: BLE001
        print(f"verify_connectivity failed: {e}", file=sys.stderr)
        return 1

    with driver.session() as s:
        s.run(
            "CREATE (a:Person {name: $name, age: $age})",
            name="Alice",
            age=30,
        ).consume()
        result = s.run(
            "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
        )
        rows = [dict(r) for r in result]
        assert len(rows) == 1, f"expected 1 row, got {rows!r}"
        assert rows[0]["name"] == "Alice", rows
        assert rows[0]["age"] == 30, rows
        print(f"ok: {rows[0]!r}")

    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
