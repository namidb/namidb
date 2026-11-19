#!/usr/bin/env python3
"""Mixed read workload at increasing concurrency against a running server.

Eighteen query shapes that a search + graph application sends to one
namespace: BM25 top-k, vector top-k, hybrid, search-then-filter, anchored
traversals of one to three hops, and GraphRAG chains (search -> a few
nodes -> their hub -> the hub's neighbours). Every level runs the same
uniform mix with `C` closed-loop clients and reports throughput, latency
percentiles, errors by status/code, whether every refusal carried
`Retry-After`, and the server's peak resident set.

The schema is a parameter, not an assumption. The defaults describe the
synthetic corpus `--load` writes (hubs `:Doc`, leaves `:Section`); pass
`--schema file.json` to run the same shapes against a dataset with other
names (every key of `DEFAULT_SCHEMA` may be overridden).

Single-stream baseline: on a corpus larger than RAM the caches keep
filling for minutes, so a C=1 level measured first is slower than a warm
one and inflates every "x single" ratio. The run therefore first warms up
until throughput is steady (`--steady-warmup`, at `--warm-concurrency`), and
the default levels measure C=1 both first and last; the ratios and the
latency budget use the faster (warmer) C=1.

Usage:
    mixed_concurrency.py --url http://127.0.0.1:8080/<namespace> \
        --levels 1,4,8,16,32,64,1 --duration 60 --rss-pid <server pid>
    mixed_concurrency.py --url ... --load --hubs 28000 --leaves 378000

The script only needs the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

DEFAULT_SCHEMA = {
    # Hub nodes (few, high in-degree) and their key.
    "hub_label": "Doc",
    "hub_key": "id",
    "hub_key_format": "d{}",
    "hub_count": 28000,
    # Leaf nodes (many, carry text and a vector) and their key.
    "leaf_label": "Section",
    "leaf_key": "id",
    "leaf_key_format": "s{}",
    "leaf_count": 378000,
    "text_property": "body",
    "title_property": "title",
    "vector_property": "embedding",
    "vector_dim": 1024,
    # Leaf property holding its hub's key (a foreign key).
    "fk_property": "doc_id",
    # Relationship types: hub -> leaf, hub -> hub, leaf -> leaf.
    "contains_rel": "HAS_SECTION",
    "amends_rel": "AMENDS",
    "version_rel": "HAS_VERSION",
    # Hubs with a very large fan-out (optional; random hubs otherwise).
    "hot_hubs": [],
    # Query terms: term_format.format(rank) for a rank in [term_min, term_max).
    "term_format": "t{}",
    "term_min": 20,
    "term_max": 3000,
}


def shapes(s: dict) -> list[tuple[str, str]]:
    """The eighteen query shapes as (name, Cypher) pairs."""
    H, L = s["hub_label"], s["leaf_label"]
    hk, lk = s["hub_key"], s["leaf_key"]
    T, TI, V, FK = s["text_property"], s["title_property"], s["vector_property"], s["fk_property"]
    C, A, VR = s["contains_rel"], s["amends_rel"], s["version_rel"]
    bm25 = f"CALL search.bm25({{label: '{L}', text_property: '{T}', query: $q, k: __K__}}) YIELD node, score "
    vec = f"CALL search.vector({{label: '{L}', property: '{V}', query: $v, k: __K__}}) YIELD node, score "
    hyb = (
        f"CALL search.hybrid({{label: '{L}', query_text: $q, text_property: '{T}', "
        f"query_vector: $v, vector_property: '{V}', k: __K__}}) YIELD node, score "
    )
    return [
        ("bm25_top10", bm25.replace("__K__", "10") + f"RETURN node.{lk} AS id, score"),
        ("bm25_top50", bm25.replace("__K__", "50") + f"RETURN node.{lk} AS id, node.{TI} AS title, score"),
        (
            "bm25_history",
            bm25.replace("__K__", "50")
            + f"OPTIONAL MATCH (node)-[:{VR}]->(v:{L}) "
            f"RETURN node.{lk} AS id, collect(v.{lk}) AS versions, score LIMIT 20",
        ),
        ("vector_k100", vec.replace("__K__", "100") + f"RETURN node.{lk} AS id, score"),
        (
            "vector_filter",
            f"CALL search.vector({{label: '{L}', property: '{V}', query: $v, k: 10, "
            f"filter: {{{FK}: $hubs}}}}) YIELD node, score RETURN node.{lk} AS id, score",
        ),
        ("hybrid_top10", hyb.replace("__K__", "10") + f"RETURN node.{lk} AS id, node.{TI} AS title, score"),
        (
            "bm25_300_filter",
            bm25.replace("__K__", "300")
            + f"WITH node, score WHERE node.{FK} IN $hubs OR node.{TI} CONTAINS $t "
            f"RETURN node.{lk} AS id, score ORDER BY score DESC LIMIT 20",
        ),
        (
            "vector_200_filter",
            vec.replace("__K__", "200")
            + f"WITH node, score WHERE node.{FK} IN $hubs OR node.{TI} CONTAINS $t "
            f"RETURN node.{lk} AS id, score ORDER BY score DESC LIMIT 20",
        ),
        ("point_leaf", f"MATCH (n:{L} {{{lk}: $leaf}}) RETURN n.{lk} AS id, n.{TI} AS title, n.{FK} AS hub"),
        ("hop1_out", f"MATCH (h:{H} {{{hk}: $hub}})-[:{C}]->(n:{L}) RETURN n.{lk} AS id LIMIT 100"),
        ("hop1_in", f"MATCH (h:{H} {{{hk}: $hub}})<-[:{A}]-(r:{H}) RETURN r.{hk} AS id LIMIT 100"),
        (
            "hop2",
            f"MATCH (h:{H} {{{hk}: $hub}})-[:{A}]-(m:{H})-[:{C}]->(n:{L}) "
            f"RETURN m.{hk} AS hub, count(n) AS sections",
        ),
        (
            "hop3",
            f"MATCH (h:{H} {{{hk}: $hub}})-[:{A}]-(m:{H})-[:{A}]-(k:{H})-[:{C}]->(n:{L}) "
            f"RETURN count(DISTINCT k) AS hubs, count(n) AS sections",
        ),
        (
            "siblings",
            f"MATCH (n:{L} {{{lk}: $leaf}})<-[:{C}]-(h:{H})-[:{C}]->(s:{L}) "
            f"RETURN h.{hk} AS hub, s.{lk} AS id LIMIT 50",
        ),
        ("hot_hub_count", f"MATCH (h:{H} {{{hk}: $hot}})-[r]->() RETURN type(r) AS t, count(*) AS n"),
        (
            "rag_bm25",
            bm25.replace("__K__", "20")
            + f"MATCH (node)<-[:{C}]-(h:{H}) WITH DISTINCT h LIMIT 20 "
            f"OPTIONAL MATCH (h)-[:{A}]-(r:{H}) RETURN h.{hk} AS hub, collect(r.{hk})[..10] AS related",
        ),
        (
            "rag_vector",
            vec.replace("__K__", "20")
            + f"MATCH (node)<-[:{C}]-(h:{H}) WITH DISTINCT h LIMIT 20 "
            f"MATCH (h)-[:{C}]->(s:{L}) RETURN h.{hk} AS hub, count(s) AS sections",
        ),
        (
            "rag_hybrid",
            hyb.replace("__K__", "20")
            + f"MATCH (node)<-[:{C}]-(h:{H}) WITH DISTINCT h LIMIT 20 "
            f"OPTIONAL MATCH (h)<-[:{A}]-(r:{H}) OPTIONAL MATCH (h)-[:{C}]->(s:{L}) "
            f"RETURN h.{hk} AS hub, count(DISTINCT r) AS amended_by, count(DISTINCT s) AS sections",
        ),
    ]


class Params:
    """Deterministic per-request parameters (vectors drawn from a fixed pool)."""

    def __init__(self, s: dict, seed: int):
        self.s = s
        rng = random.Random(seed)
        self.vectors = [unit_vector(rng, s["vector_dim"]) for _ in range(64)]

    def make(self, rng: random.Random) -> dict:
        s = self.s
        term = lambda: s["term_format"].format(rng.randrange(s["term_min"], s["term_max"]))
        hub = lambda: s["hub_key_format"].format(rng.randrange(s["hub_count"]))
        hot = s["hot_hubs"]
        return {
            "q": f"{term()} {term()}",
            "t": term(),
            "v": rng.choice(self.vectors),
            "hub": hub(),
            "hubs": [hub() for _ in range(200)],
            "leaf": s["leaf_key_format"].format(rng.randrange(s["leaf_count"])),
            "hot": rng.choice(hot) if hot else hub(),
        }


def unit_vector(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [round(x / norm, 6) for x in v]


def used_params(query: str, params: dict) -> dict:
    return {k: v for k, v in params.items() if f"${k}" in query}


def post(url: str, query: str, params: dict, timeout: float):
    body = json.dumps({"query": query, "params": params}).encode()
    req = urllib.request.Request(url, body, {"content-type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status, None, None, time.perf_counter() - started
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            code = json.loads(raw).get("code")
        except Exception:
            code = None
        return e.code, code, e.headers.get("Retry-After"), time.perf_counter() - started
    except Exception as e:  # connection reset, client timeout
        return 0, type(e).__name__, None, time.perf_counter() - started


def rss_bytes(pid: int | None) -> int:
    if not pid:
        return 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, math.ceil(p * len(sorted_values)) - 1))
    return sorted_values[idx]


def run_level(args, url, qs, params, level: int) -> dict:
    stop_at = [0.0]
    records: list[tuple[str, int, str | None, str | None, float, float]] = []
    lock = threading.Lock()
    measuring = threading.Event()
    peak_rss = [0]

    def sampler():
        while not stop_at[0] or time.monotonic() < stop_at[0] + 1:
            if measuring.is_set():
                peak_rss[0] = max(peak_rss[0], rss_bytes(args.rss_pid))
            time.sleep(0.2)
            if stop_at[0] and time.monotonic() > stop_at[0]:
                break

    def worker(index: int):
        rng = random.Random(args.seed * 1000 + level * 100 + index)
        i = index
        while True:
            now = time.monotonic()
            if stop_at[0] and now >= stop_at[0]:
                return
            name, query = qs[i % len(qs)]
            i += 1
            status, code, retry, elapsed = post(url, query, used_params(query, params.make(rng)), args.timeout)
            if measuring.is_set():
                with lock:
                    records.append((name, status, code, retry, elapsed, time.monotonic()))

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(level)]
    sample = threading.Thread(target=sampler, daemon=True)
    for t in threads:
        t.start()
    sample.start()
    time.sleep(args.warmup)
    measuring.set()
    measure_start = time.monotonic()
    stop_at[0] = measure_start + args.duration
    for t in threads:
        t.join()
    wall = time.monotonic() - measure_start
    sample.join(timeout=2)

    ok = [r for r in records if r[1] == 200]
    lat = sorted(r[4] for r in ok)
    errors = Counter(f"{r[1]}:{r[2]}" for r in records if r[1] != 200)
    refusals = [r for r in records if r[1] in (429, 503)]
    per_shape = defaultdict(list)
    for r in ok:
        per_shape[r[0]].append(r[4])
    return {
        "concurrency": level,
        "wall_s": round(wall, 2),
        "requests": len(records),
        "ok": len(ok),
        "errors": dict(errors),
        "error_rate": round((len(records) - len(ok)) / max(1, len(records)), 4),
        "refusals_with_retry_after": sum(1 for r in refusals if r[3]),
        "refusals": len(refusals),
        "throughput_qps": round(len(ok) / wall, 3) if wall else 0.0,
        "p50_ms": round(percentile(lat, 0.50) * 1000, 1),
        "p95_ms": round(percentile(lat, 0.95) * 1000, 1),
        "p99_ms": round(percentile(lat, 0.99) * 1000, 1),
        "max_ms": round((lat[-1] if lat else float("nan")) * 1000, 1),
        "peak_rss_bytes": peak_rss[0],
        "per_shape_p50_ms": {
            k: round(percentile(sorted(v), 0.5) * 1000, 1) for k, v in sorted(per_shape.items())
        },
    }


def load_synthetic(args, s: dict):
    """Write a synthetic corpus with the shape the workload expects.

    Hubs get a power-law in-degree on `amends_rel` (Zipf, capped near
    `--max-in-degree`), leaves get Spanish-like accented text drawn from a
    Zipf vocabulary, a 1024-dim clustered unit vector, a foreign key to
    their hub, and a few `version_rel` edges. Indexes are created first.
    """
    url = args.url.rstrip("/") + "/v0/cypher"
    rng = random.Random(args.seed)
    H, L = s["hub_label"], s["leaf_label"]

    def commit(q, p=None):
        for attempt in range(1, 400):
            status, code, _, _ = post(url, q, p or {}, 900)
            if status == 200:
                return
            time.sleep(min(5.0, 0.2 * attempt))
        raise SystemExit(f"gave up on {q[:80]}: {status} {code}")

    for q in [
        f"CREATE INDEX IF NOT EXISTS FOR (n:{H}) ON (n.{s['hub_key']})",
        f"CREATE INDEX IF NOT EXISTS FOR (n:{L}) ON (n.{s['leaf_key']})",
        f"CREATE INDEX IF NOT EXISTS FOR (n:{L}) ON (n.{s['fk_property']})",
        f"CREATE FULLTEXT INDEX {L.lower()}_ft IF NOT EXISTS FOR (n:{L}) ON EACH [n.{s['text_property']}]",
        f"CREATE VECTOR INDEX {L.lower()}_vec IF NOT EXISTS ON :{L}({s['vector_property']}) "
        f"METRIC cosine DIMENSION {s['vector_dim']}",
    ]:
        commit(q)
    hubs, leaves = args.hubs, args.leaves
    hk = s["hub_key_format"]
    for start in range(0, hubs, 2000):
        rows = [{s["hub_key"]: hk.format(i), "title": f"documento {i}"} for i in range(start, min(hubs, start + 2000))]
        commit(f"UNWIND $rows AS r CREATE (n:{H}) SET n = r", {"rows": rows})
    syll = ["ción", "mente", "dad", "ería", "ánimo", "jubil", "pensión", "regla", "artículo", "código",
            "tribunal", "público", "órgano", "derecho", "reforma", "ñandú", "acción", "título", "régimen", "sí"]
    vocab = [syll[i % 20] + syll[(i // 20) % 20] + (str(i // 400) if i >= 400 else "") for i in range(20000)]
    weights = [1.0 / (r + 1) ** 1.07 for r in range(len(vocab))]
    centers = [unit_vector(rng, s["vector_dim"]) for _ in range(64)]
    lk = s["leaf_key_format"]
    batch = 300
    for start in range(0, leaves, batch):
        rows = []
        for i in range(start, min(leaves, start + batch)):
            c = centers[rng.randrange(len(centers))]
            v = [x + rng.gauss(0, 0.6 / math.sqrt(len(c))) for x in c]
            n = math.sqrt(sum(x * x for x in v))
            rows.append({
                s["leaf_key"]: lk.format(i),
                s["fk_property"]: hk.format(i % hubs),
                s["title_property"]: " ".join(rng.choices(vocab, weights, k=8)),
                s["text_property"]: " ".join(rng.choices(vocab, weights, k=240)),
                s["vector_property"]: [round(x / n, 6) for x in v],
            })
        commit(f"UNWIND $rows AS r CREATE (n:{L}) SET n = r", {"rows": rows})
        if start % 30000 == 0:
            print("leaves", start, time.strftime("%H:%M:%S"), flush=True)

    def edges(rel, sl, sk, dl, dk, pairs):
        q = f"UNWIND $rows AS r MATCH (a:{sl} {{{sk}: r.s}}) MATCH (b:{dl} {{{dk}: r.d}}) CREATE (a)-[:{rel}]->(b)"
        for i in range(0, len(pairs), 2000):
            commit(q, {"rows": [{"s": a, "d": b} for a, b in pairs[i:i + 2000]]})
        print(rel, len(pairs), time.strftime("%H:%M:%S"), flush=True)

    edges(s["contains_rel"], H, s["hub_key"], L, s["leaf_key"], [(hk.format(i % hubs), lk.format(i)) for i in range(leaves)])
    # Power-law in-degree: target rank r receives ~max_in / r^0.9 amendments.
    pairs = []
    for r in range(hubs):
        for _ in range(int(args.max_in_degree / (r + 1) ** 0.9)):
            pairs.append((hk.format(rng.randrange(hubs)), hk.format(r)))
    edges(s["amends_rel"], H, s["hub_key"], H, s["hub_key"], pairs)
    edges(s["version_rel"], L, s["leaf_key"], L, s["leaf_key"],
          [(lk.format(rng.randrange(leaves)), lk.format(rng.randrange(leaves))) for _ in range(leaves // 8)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="base URL of the namespace, e.g. http://127.0.0.1:8080/ns")
    ap.add_argument("--schema", help="JSON file overriding DEFAULT_SCHEMA keys")
    ap.add_argument("--levels", default="1,4,8,16,32,64,1",
                    help="concurrency levels in order; C=1 may repeat (the warmest is the baseline)")
    ap.add_argument("--steady-warmup", type=float, default=300.0,
                    help="max seconds of unmeasured warm-up before the first level (0 = none)")
    ap.add_argument("--warm-concurrency", type=int, default=4)
    ap.add_argument("--steady-window", type=float, default=20.0,
                    help="warm-up window; steady when two consecutive windows differ by under 5%%")
    ap.add_argument("--duration", type=float, default=60.0, help="measured seconds per level")
    ap.add_argument("--warmup", type=float, default=5.0, help="unmeasured seconds per level")
    ap.add_argument("--timeout", type=float, default=120.0, help="client timeout per request")
    ap.add_argument("--rss-pid", type=int, help="server pid whose VmRSS is sampled")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", help="comma list of shape names to run")
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument("--label", default="", help="free-form label stored in the report")
    ap.add_argument("--load", action="store_true", help="write the synthetic corpus instead of measuring")
    ap.add_argument("--hubs", type=int, default=28000)
    ap.add_argument("--leaves", type=int, default=378000)
    ap.add_argument("--max-in-degree", type=int, default=250)
    args = ap.parse_args()

    schema = dict(DEFAULT_SCHEMA)
    if args.schema:
        with open(args.schema) as f:
            schema.update(json.load(f))
    if args.load:
        load_synthetic(args, schema)
        return
    url = args.url.rstrip("/") + "/v0/cypher"
    qs = shapes(schema)
    if args.only:
        wanted = set(args.only.split(","))
        qs = [q for q in qs if q[0] in wanted]
    params = Params(schema, args.seed)
    report = {"label": args.label, "levels": [], "warmup": []}
    if args.steady_warmup > 0:
        warm_args = argparse.Namespace(**{**vars(args), "warmup": 0.0, "duration": args.steady_window})
        started = time.monotonic()
        previous = None
        while time.monotonic() - started < args.steady_warmup:
            window = run_level(warm_args, url, qs, params, args.warm_concurrency)
            qps = window["throughput_qps"]
            report["warmup"].append(qps)
            print(json.dumps({"warmup_qps": qps, "elapsed_s": round(time.monotonic() - started)}), flush=True)
            if previous and qps > 0 and abs(qps - previous) / max(qps, previous) < 0.05:
                break
            previous = qps
    for level in [int(x) for x in args.levels.split(",")]:
        result = run_level(args, url, qs, params, level)
        report["levels"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_shape_p50_ms"}), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
    singles = [r for r in report["levels"] if r["concurrency"] == 1] or report["levels"][:1]
    base = max(singles, key=lambda r: r["throughput_qps"])
    report["baseline"] = {"throughput_qps": base["throughput_qps"], "p50_ms": base["p50_ms"],
                          "single_runs_qps": [r["throughput_qps"] for r in singles]}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print()
    print("baseline (warmest C=1): %.2f q/s, p50 %.0f ms; all C=1 runs: %s"
          % (base["throughput_qps"], base["p50_ms"], report["baseline"]["single_runs_qps"]))
    print(f"| C | ok | err% | q/s | x single | p50 ms | p95 ms | p99 ms | peak RSS MiB | 503/429 w/ Retry-After |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in report["levels"]:
        speedup = r["throughput_qps"] / base["throughput_qps"] if base["throughput_qps"] else float("nan")
        print(
            f"| {r['concurrency']} | {r['ok']} | {100 * r['error_rate']:.1f} | {r['throughput_qps']:.2f} | "
            f"{speedup:.2f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | "
            f"{r['peak_rss_bytes'] / 2**20:.0f} | {r['refusals_with_retry_after']}/{r['refusals']} |"
        )
    cpus = os.environ.get("BENCH_CPUS")
    if cpus:
        cpus = float(cpus)
        print()
        for r in report["levels"]:
            c = r["concurrency"]
            if c <= cpus:
                continue
            budget = (c / cpus) * base["p50_ms"] * 1.3
            print(f"C={c}: p50 {r['p50_ms']:.0f} ms vs budget (C/CPUs) x p50(C=1) x 1.3 = {budget:.0f} ms")
    if base.get("per_shape_p50_ms"):
        print()
        print("single-stream p50 per shape (ms):", json.dumps(base["per_shape_p50_ms"]))


if __name__ == "__main__":
    sys.exit(main())
