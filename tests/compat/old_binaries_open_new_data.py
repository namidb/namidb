"""Open data written by a new server with older releases, then roll forward.

Driven by old_binaries_open_new_data.sh. Phases, on a fresh local store:

1. The oldest release given creates a full-text index (the pre-folding
   analyzer), a vector index and documents with accented text.
2. The new server opens it: the legacy index keeps answering with its own
   analyzer, is rebuilt in the background with accent folding, and new
   indexes are created with the default and with `analyzer: 'exact'`.
3. Every old release opens a copy of that store: searches return the results
   of its own analyzer, writes are accepted, flushed, compacted, and survive a
   restart.
4. The new server opens each store the old release left behind and serves
   accent-insensitive results again once its background rebuild lands.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("NAMIDB_COMPAT_PORT", "18831"))
NS = "compat"
BODIES = [
    "La pensión jubilar se calcula sobre el último sueldo",
    "Ação judicial sobre a indenização",
    "Le cœur du problème est la procédure",
    "Straße und Brücke",
    "plain ascii text about pension rules",
]
ROUNDS = 8


class Server:
    def __init__(self, binary, store, log):
        self.binary, self.store, self.log = binary, store, log
        self.proc = None

    def __enter__(self):
        env = dict(os.environ, NAMIDB_NO_AUTH="1", RUST_LOG="warn")
        self.proc = subprocess.Popen(
            [
                self.binary,
                "--store", f"file://{self.store}?ns={NS}",
                "--listen", f"127.0.0.1:{PORT}",
                "--flush-interval", "1s",
                "--compaction-interval", "2s",
            ],
            stdout=open(self.log, "ab"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.binary} exited; see {self.log}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v0/livez", timeout=2)
                return self
            except Exception:
                time.sleep(0.3)
        raise RuntimeError(f"{self.binary} did not come up; see {self.log}")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def post(self, path, body=None):
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}{path}", data, {"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{path}: {error.code} {error.read()[:500]!r}") from None

    def cypher(self, query, **params):
        return self.post("/v0/cypher", {"query": query, "params": params}).get("rows", [])

    def flush(self):
        self.post("/v0/admin/flush")

    def compact(self):
        self.post("/v0/admin/compact")

    def hits(self, label, query):
        rows = self.cypher(
            f"CALL search.bm25({{label: '{label}', text_property: 'body', query: $q, k: 1000}}) "
            "YIELD node, score RETURN count(node) AS n",
            q=query,
        )
        return rows[0]["n"]

    def nearest(self):
        rows = self.cypher(
            "CALL db.index.vector.queryNodes('doc_vec', 3, $v) YIELD node, score "
            "RETURN node.body AS body",
            v=[1.0, 0.0, 0.0, 0.0],
        )
        return len(rows)


def manifest(store):
    files = sorted(glob.glob(os.path.join(store, NS, "manifest", "v*.json")))
    with open(files[-1]) as handle:
        return json.load(handle)


def analyzers(store):
    return {index["name"]: index.get("analyzer") for index in manifest(store)["text_indexes"]}


def wait_for(predicate, what, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}")


def expect(actual, expected, what):
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected}, got {actual}")
    print(f"  ok  {what} = {actual}", flush=True)


def seed(server):
    server.cypher("CREATE FULLTEXT INDEX doc_ft IF NOT EXISTS FOR (n:Doc) ON EACH [n.body]")
    server.cypher(
        "CREATE VECTOR INDEX doc_vec IF NOT EXISTS FOR (n:Doc) ON (n.emb) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 4, `vector.similarity_function`: 'cosine'}}"
    )
    rows = []
    for round_ in range(ROUNDS):
        for i, body in enumerate(BODIES):
            rows.append({"body": f"{body} r{round_}", "emb": [1.0, float(i), float(round_), 0.5]})
    server.cypher("UNWIND $rows AS r CREATE (d:Doc) SET d.body = r.body, d.emb = r.emb", rows=rows)
    server.flush()
    server.compact()


# The first release whose full-text analyzer folds accents by default. It
# reads the analyzer an index records, so it serves a folded index (one the
# new server rebuilt) accent-insensitively; earlier releases serve it with
# their lowercase-only analyzer.
FOLDING_SINCE = (3, 5, 2)


def release(tag):
    """`v3.5.2` -> (3, 5, 2)."""
    return tuple(int(part) for part in tag.lstrip("v").split(".")[:3])


def folds_accents(tag):
    return release(tag) >= FOLDING_SINCE


def check_old_semantics(server, tag, extra_docs=0):
    if folds_accents(tag):
        # 'pensión' and 'pension' fold to one term: both groups of documents.
        expect(server.hits("Doc", "pensión"), 2 * ROUNDS, f"{tag} Doc 'pensión' (folding analyzer)")
        expect(server.hits("Doc", "pension"), 2 * ROUNDS, f"{tag} Doc 'pension' (folding analyzer)")
    else:
        expect(server.hits("Doc", "pensión"), ROUNDS, f"{tag} Doc 'pensión' (lowercase-only analyzer)")
        expect(server.hits("Doc", "pension"), ROUNDS, f"{tag} Doc 'pension'")
    expect(server.hits("Note", "jubilación"), 1, f"{tag} Note 'jubilación'")
    expect(server.hits("Memo", "Educación"), 1, f"{tag} Memo 'Educación'")
    expect(server.nearest(), 3, f"{tag} vector top-3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", action="append", default=[], help="tag=binary")
    args = parser.parse_args()
    olds = [tuple(item.split("=", 1)) for item in args.old]
    if not olds:
        sys.exit("give at least one --old tag=binary")
    work = tempfile.mkdtemp(prefix="namidb-compat-")
    log = os.path.join(work, "server.log")
    print(f"work dir {work}", flush=True)
    base = os.path.join(work, "base")
    os.makedirs(base)

    oldest_tag, oldest = olds[0]
    if folds_accents(oldest_tag):
        sys.exit(
            f"the first --old ({oldest_tag}) seeds the legacy index and must predate "
            f"accent folding (v{'.'.join(map(str, FOLDING_SINCE))}); give the releases oldest first"
        )
    print(f"[1] {oldest_tag} seeds a legacy full-text index", flush=True)
    with Server(oldest, base, log) as server:
        seed(server)
        expect(server.hits("Doc", "pensión"), ROUNDS, f"{oldest_tag} Doc 'pensión'")
    expect(analyzers(base).get("doc_ft"), None, "seeded index analyzer field")

    print("[2] new server opens it and rebuilds the legacy index", flush=True)
    with Server(args.new, base, log) as server:
        # Before the rebuild lands the legacy index answers with its own
        # analyzer; after it, accent-insensitively. Either way no error.
        server.hits("Doc", "pension")
        wait_for(lambda: analyzers(base).get("doc_ft") == "folded", "background re-analysis")
        expect(server.hits("Doc", "PENSION"), 2 * ROUNDS, "new Doc 'PENSION' (folded)")
        expect(server.hits("Doc", "acao"), ROUNDS, "new Doc 'acao'")
        server.cypher("CREATE FULLTEXT INDEX note_ft IF NOT EXISTS FOR (n:Note) ON EACH [n.body]")
        server.cypher(
            "CREATE FULLTEXT INDEX memo_ft IF NOT EXISTS FOR (n:Memo) ON EACH [n.body] "
            "OPTIONS {analyzer: 'exact'}"
        )
        server.cypher("CREATE (:Note {body: 'Jubilación anticipada'}), (:Memo {body: 'Educación pública'})")
        server.flush()
        server.compact()
        expect(server.hits("Note", "jubilacion"), 1, "new Note 'jubilacion' (folded)")
        expect(server.hits("Memo", "educacion"), 0, "new Memo 'educacion' (exact opt-out)")
        expect(server.hits("Memo", "educación"), 1, "new Memo 'educación' (exact opt-out)")
        expect(server.nearest(), 3, "new vector top-3")
    written = analyzers(base)
    expect(written, {"doc_ft": "folded", "note_ft": "folded", "memo_ft": "exact"}, "new analyzers")

    failures = []
    for tag, binary in olds:
        print(f"[3] {tag} opens the new data", flush=True)
        store = os.path.join(work, f"store-{tag}")
        shutil.copytree(base, store)
        try:
            with Server(binary, store, log) as server:
                check_old_semantics(server, tag)
                server.cypher("CREATE (:Doc {body: 'nueva resolución', emb: [0.0, 1.0, 0.0, 0.0]})")
                server.flush()
                expect(server.hits("Doc", "resolución"), 1, f"{tag} write then search")
                server.compact()
                expect(server.hits("Doc", "resolución"), 1, f"{tag} after compaction")
            with Server(binary, store, log) as server:
                check_old_semantics(server, tag)
                expect(server.hits("Doc", "resolución"), 1, f"{tag} after restart")
            print(f"[4] new server rolls forward from {tag}", flush=True)
            with Server(args.new, store, log) as server:
                wait_for(
                    lambda: all(a in ("folded", "exact") for a in analyzers(store).values()),
                    "re-analysis after roll-forward",
                )
                expect(server.hits("Doc", "resolucion"), 1, f"roll-forward from {tag}: 'resolucion'")
                expect(server.hits("Doc", "pension"), 2 * ROUNDS, f"roll-forward from {tag}: 'pension'")
                expect(server.nearest(), 3, f"roll-forward from {tag}: vector top-3")
            print(f"PASS {tag}", flush=True)
        except Exception as error:  # noqa: BLE001 - report every release
            failures.append(f"{tag}: {error}")
            print(f"FAIL {tag}: {error}", flush=True)
    if failures:
        print(f"server log: {log}")
        sys.exit("\n".join(failures))
    shutil.rmtree(work)
    print("all releases open the new data", flush=True)


if __name__ == "__main__":
    main()
