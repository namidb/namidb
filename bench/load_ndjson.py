#!/usr/bin/env python3
"""Load a `namidb-bench graph-shape-generate --ndjson-dir` graph through a
NamiDB Python client.

Runs against whichever `namidb` package the interpreter imports, so the same
graph can be written by an older release (for example a 2.x wheel in its own
virtualenv) and then read by the current engine, to compare on-disk layouts:

    python -m venv /tmp/v2 && /tmp/v2/bin/pip install 'namidb==2.6.19'
    namidb-bench graph-shape-generate --ndjson-dir /tmp/shape-json
    /tmp/v2/bin/python bench/load_ndjson.py /tmp/shape-json 'file:///tmp/shape-v2?ns=shape'

Nodes go in through `merge_nodes`, edges through `merge_edges`, committed and
flushed every `--batch` rows; the client's own flush-triggered compaction runs
as it would for an application, and a final `compact()` loop leaves the store
fully compacted.
"""

import argparse
import json
import os
import sys
import time

import namidb

LABELS = ["Device", "Call", "Message", "Contact", "Location"]


def batches(path, size):
    batch = []
    with open(path) as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= size:
                yield batch
                batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ndjson_dir")
    ap.add_argument("uri")
    ap.add_argument("--batch", type=int, default=100_000)
    args = ap.parse_args()

    print(f"namidb {getattr(namidb, '__version__', '?')}", file=sys.stderr)
    client = namidb.Client(args.uri)
    started = time.time()
    total = 0
    for label in LABELS:
        path = os.path.join(args.ndjson_dir, f"nodes_{label}.ndjson")
        for rows in batches(path, args.batch):
            client.merge_nodes(label, rows)
            client.commit()
            client.flush()
            total += len(rows)
            print(f"{label}: {total} nodes ({time.time() - started:.0f}s)", file=sys.stderr)
    edges = 0
    for rows in batches(os.path.join(args.ndjson_dir, "edges_ON_DEVICE.ndjson"), args.batch):
        client.merge_edges("ON_DEVICE", rows)
        client.commit()
        client.flush()
        edges += len(rows)
        print(f"ON_DEVICE: {edges} edges ({time.time() - started:.0f}s)", file=sys.stderr)
    for _ in range(64):
        report = client.compact()
        applied = report["applied"] if isinstance(report, dict) else getattr(report, "applied", False)
        if not applied:
            break
    print(f"done in {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
