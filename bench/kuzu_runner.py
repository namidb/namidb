"""LDBC-shaped bench harness for Kuzu, paired with `namidb-bench`.

Reads the same CSV dataset NamiDB loads, ingests it into Kuzu via
`COPY <Label> FROM '<path>.csv' (DELIM='|')`, then times the same four
LDBC SNB Complex Read queries. Output JSON is shape-compatible with
`namidb-bench run` so callers can diff side-by-side.

Pre-requisitos:
    pip install kuzu

Uso:
    # 1. Generar el dataset con el harness Rust:
    cargo run --release -p namidb-bench -- generate \\
        --scale 0.1 --seed 42 --out /tmp/snb-0.1

    # 2. Correr el bench NamiDB:
    cargo run --release -p namidb-bench -- run \\
        --scale 0.1 --dataset-dir /tmp/snb-0.1 --warm-runs 50 > tg.json

    # 3. Correr el bench Kuzu sobre el mismo dataset:
    python3 bench/kuzu_runner.py --dataset-dir /tmp/snb-0.1 \\
        --warm-runs 50 --param-count 3 > kuzu.json

    # 4. Comparar lado a lado:
    python3 bench/compare.py tg.json kuzu.json   # (helper opcional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

try:
    import kuzu
except ImportError:
    sys.stderr.write(
        "ERROR: kuzu not installed. Run: pip install kuzu\n"
        "       (Or pin a version: pip install 'kuzu>=0.6')\n"
    )
    sys.exit(2)


# ── Schema mirroring namidb_bench::loader::schema ────────────────────

DDL_STATEMENTS = [
    """CREATE NODE TABLE Person(
        id STRING,
        firstName STRING,
        lastName STRING,
        age INT64,
        creationDate INT64,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE Post(
        id STRING,
        content STRING,
        creationDate INT64,
        length INT64,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE Comment(
        id STRING,
        content STRING,
        creationDate INT64,
        length INT64,
        PRIMARY KEY (id)
    )""",
    """CREATE REL TABLE KNOWS(
        FROM Person TO Person,
        since INT64
    )""",
    # HAS_CREATOR, LIKES, REPLY_OF in the LDBC-shaped dataset are
    # polymorphic — a single CSV file mixes (Post|Comment)→Person etc.
    # Kuzu 0.11+ requires a multi-FROM/TO REL TABLE plus one COPY per
    # (FROM,TO) pair (see `split_rel_csv` below).
    """CREATE REL TABLE HAS_CREATOR(
        FROM Post TO Person,
        FROM Comment TO Person
    )""",
    """CREATE REL TABLE LIKES(
        FROM Person TO Post,
        FROM Person TO Comment,
        creationDate INT64
    )""",
    """CREATE REL TABLE REPLY_OF(
        FROM Comment TO Post,
        FROM Comment TO Comment
    )""",
]


COPY_NODE_STATEMENTS = [
    ("Person", "persons.csv"),
    ("Post", "posts.csv"),
    ("Comment", "comments.csv"),
]

# Single-pair rels can be COPYed directly without splitting.
SIMPLE_COPY_REL_STATEMENTS = [
    ("KNOWS", "knows.csv", "Person", "Person"),
]

# Multi-pair rels: each entry is (rel_table, source_csv, split_fn) where
# split_fn returns a list of (from_label, to_label, rows) tuples.
# Rows are the raw CSV rows (excluding header) routed to that pair.
MULTI_REL_FILES = [
    # (rel_table, source_csv)
    ("HAS_CREATOR", "has_creator.csv"),  # src label varies (Post|Comment), dst always Person
    ("LIKES", "likes.csv"),               # src always Person, dst varies (Post|Comment)
    ("REPLY_OF", "reply_of.csv"),         # src always Comment, dst varies (Post|Comment)
]


# Hex prefix → node label (matches dataset.rs encode_id prefixes).
PREFIX_TO_LABEL = {
    "50": "Person",   # 'P'
    "4f": "Post",     # 'O'
    "43": "Comment",  # 'C'
}


def _label_of(hex_id: str) -> str:
    prefix = hex_id[:2].lower()
    label = PREFIX_TO_LABEL.get(prefix)
    if label is None:
        raise ValueError(f"unknown id prefix in {hex_id!r}")
    return label


def split_rel_csv(rel_table: str, src_path: Path, out_dir: Path) -> list[tuple[str, str, Path]]:
    """Split a mixed rel CSV into one file per (from_label, to_label).

    Returns a list of (from_label, to_label, path) tuples; each path
    contains the header + the subset of rows whose endpoints match.
    """
    with src_path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        rows_by_pair: dict[tuple[str, str], list[str]] = {}
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("|")
            src_label = _label_of(cols[0])
            dst_label = _label_of(cols[1])
            rows_by_pair.setdefault((src_label, dst_label), []).append(line)

    out: list[tuple[str, str, Path]] = []
    for (fr, to), rows in rows_by_pair.items():
        path = out_dir / f"{rel_table.lower()}__{fr}__{to}.csv"
        with path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(r + "\n")
        out.append((fr, to, path))
    return out


# ── Four LDBC SNB Complex Read queries (parameterised) ───────────────────


def cypher_for(query: str, person_id: str) -> str:
    if query == "ic02":
        return (
            f"MATCH (p:Person {{id: '{person_id}'}})-[:KNOWS]->(friend:Person)"
            "<-[:HAS_CREATOR]-(message:Post) "
            "RETURN friend.firstName AS personFirstName, friend.lastName AS personLastName, "
            "       message.content AS messageContent, message.creationDate AS messageCreationDate "
            "ORDER BY messageCreationDate DESC LIMIT 20"
        )
    if query == "ic07":
        return (
            f"MATCH (p:Person {{id: '{person_id}'}})<-[:HAS_CREATOR]-(message:Post)"
            "<-[liker:LIKES]-(fan:Person) "
            "RETURN fan.firstName AS personFirstName, fan.lastName AS personLastName, "
            "       liker.creationDate AS likeCreationDate, message.content AS messageContent "
            "ORDER BY likeCreationDate DESC LIMIT 20"
        )
    if query == "ic08":
        return (
            f"MATCH (p:Person {{id: '{person_id}'}})<-[:HAS_CREATOR]-(post:Post)"
            "<-[:REPLY_OF]-(reply:Comment) "
            "RETURN reply.content AS replyContent, reply.creationDate AS replyDate, "
            "       post.content AS postContent "
            "ORDER BY replyDate DESC LIMIT 20"
        )
    if query == "ic09":
        return (
            f"MATCH (p:Person {{id: '{person_id}'}})-[:KNOWS]->(friend:Person)"
            "-[:KNOWS]->(fof:Person)<-[:HAS_CREATOR]-(msg:Post) "
            "RETURN fof.firstName AS personFirstName, fof.lastName AS personLastName, "
            "       msg.content AS messageContent, msg.creationDate AS messageCreationDate "
            "ORDER BY messageCreationDate DESC LIMIT 20"
        )
    raise ValueError(f"unknown query: {query}")


# ── Bench primitives ────────────────────────────────────────────────────


def percentile(samples: list[int], p: float) -> int:
    if not samples:
        return 0
    s = sorted(samples)
    idx = round((len(s) - 1) * p)
    return s[min(idx, len(s) - 1)]


def run_one(conn, query: str, person_id: str, warm_runs: int) -> dict:
    cypher = cypher_for(query, person_id)
    # Cold (we re-establish the conn? In Kuzu the prepared statement cache
    # is per-connection; new connection = closer to cold). But re-creating
    # the connection over and over is heavy on Kuzu in-process. Compromise:
    # cold = first execution after schema loaded, no re-conn.
    cold_start = time.perf_counter()
    result = conn.execute(cypher)
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    cold_us = int((time.perf_counter() - cold_start) * 1_000_000)

    samples_us: list[int] = []
    for _ in range(warm_runs):
        t0 = time.perf_counter()
        r = conn.execute(cypher)
        while r.has_next():
            r.get_next()
        samples_us.append(int((time.perf_counter() - t0) * 1_000_000))

    return {
        "backend": "kuzu",
        "query": query,
        "param": person_id,
        "rows": len(rows),
        "cold_us": cold_us,
        "warm_p50_us": percentile(samples_us, 0.50),
        "warm_p95_us": percentile(samples_us, 0.95),
        "warm_p99_us": percentile(samples_us, 0.99),
        "warm_runs": warm_runs,
    }


# ── Loader ──────────────────────────────────────────────────────────────


def load_kuzu(dataset_dir: Path, kuzu_db_path: Path) -> tuple[kuzu.Database, kuzu.Connection]:
    if kuzu_db_path.exists():
        # Wipe stale state so the run is reproducible. Kuzu 0.11+ stores
        # the database as a single file, but older runs may have left a
        # directory behind, so handle both cases.
        import shutil

        if kuzu_db_path.is_dir():
            shutil.rmtree(kuzu_db_path)
        else:
            kuzu_db_path.unlink()
    # Ensure parent exists; Kuzu will create the database file itself.
    kuzu_db_path.parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(kuzu_db_path))
    conn = kuzu.Connection(db)

    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)

    # COPY node tables.
    for table, csv_name in COPY_NODE_STATEMENTS:
        path = dataset_dir / csv_name
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        conn.execute(
            f"COPY {table} FROM '{path}' (HEADER=true, DELIM='|')"
        )

    # COPY single-pair rel tables.
    for table, csv_name, _fr, _to in SIMPLE_COPY_REL_STATEMENTS:
        path = dataset_dir / csv_name
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        conn.execute(
            f"COPY {table} FROM '{path}' (HEADER=true, DELIM='|')"
        )

    # COPY multi-pair rel tables: split the source CSV by (FROM,TO),
    # write the partition to a scratch dir, then COPY each with the
    # explicit (FROM='<L1>', TO='<L2>') pragma Kuzu requires.
    split_dir = kuzu_db_path.parent / f"{kuzu_db_path.name}__splits"
    if split_dir.exists():
        import shutil
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for table, csv_name in MULTI_REL_FILES:
        path = dataset_dir / csv_name
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        for fr, to, part_path in split_rel_csv(table, path, split_dir):
            conn.execute(
                f"COPY {table} FROM '{part_path}' "
                f"(HEADER=true, DELIM='|', FROM='{fr}', TO='{to}')"
            )

    return db, conn


# ── Param picker (mirror of namidb-bench main.rs) ────────────────────


def make_person_id_hex(i: int) -> str:
    """Match namidb_bench::main::make_person_id_hex (prefix=b'P')."""
    out = bytearray(16)
    out[0] = ord("P")
    i_bytes = i.to_bytes(16, byteorder="big", signed=False)
    out[1:] = i_bytes[1:]
    return out.hex()


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="Directory of CSVs emitted by `namidb-bench generate`.",
    )
    p.add_argument(
        "--kuzu-db", default=Path("/tmp/namidb-bench-kuzu-db"), type=Path,
        help="Where to materialise the Kuzu database (will be wiped).",
    )
    p.add_argument("--warm-runs", type=int, default=50)
    p.add_argument("--param-count", type=int, default=3)
    p.add_argument(
        "--only", action="append", default=None,
        help="Limit to specific queries (ic02/ic07/ic08/ic09). Repeat for multiple.",
    )
    args = p.parse_args()

    queries = args.only or ["ic02", "ic07", "ic08", "ic09"]

    print(f"loading dataset {args.dataset_dir} into kuzu @ {args.kuzu_db}", file=sys.stderr)
    db, conn = load_kuzu(args.dataset_dir, args.kuzu_db)

    # Count persons to size the param picker the same way the Rust bench does.
    n_persons = int(
        conn.execute("MATCH (p:Person) RETURN count(p) AS c").get_next()[0]
    )
    stride = max(1, n_persons // max(1, args.param_count))
    params = [make_person_id_hex(i * stride) for i in range(args.param_count)]

    results = []
    for q in queries:
        for param in params:
            r = run_one(conn, q, param, args.warm_runs)
            print(
                f"  {r['query']} param={r['param'][:8]} rows={r['rows']} "
                f"cold={r['cold_us']}µs warm_p50={r['warm_p50_us']}µs",
                file=sys.stderr,
            )
            results.append(r)

    out = {
        "backend": "kuzu",
        "results": results,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
