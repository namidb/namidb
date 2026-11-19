#!/usr/bin/env bash
# Rollback compatibility: data written by a new server must open, read back
# identically, and accept writes under older release binaries, and the new
# binary must open what the old ones wrote afterwards.
#
# usage: rollback_compat.sh <new server binary> <old server binary>...
#
# Every server runs on a private file:// store bound to 127.0.0.1. The data
# exercises every persisted structure a read touches: nodes and
# relationships flushed to SSTs, a property index, a full-text index, a
# vector index, and a WAL tail that is still unflushed at shutdown.
set -euo pipefail
new=$1; shift
[ $# -gt 0 ] || { echo "usage: $0 <new binary> <old binary>..."; exit 2; }
work=$(mktemp -d)
mkdir -p "$work/spool"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$work"' EXIT
store="file://$work/store?ns=compat"
port=$((20000 + RANDOM % 20000))
url=http://127.0.0.1:$port

serve() { # serve <binary> <log>
  NAMIDB_NO_AUTH=1 NAMIDB_SPOOL_DIR=$work/spool NAMIDB_INDEX_BUILD_SPOOL_DIR=$work/spool \
    "$1" --store "$store" --listen 127.0.0.1:$port >"$2" 2>&1 &
  pid=$!
  for _ in $(seq 1 200); do curl -sf -o /dev/null $url/v0/livez && return 0; sleep 0.1; done
  echo "server $1 did not start"; tail -20 "$2"; exit 1
}
stop() { kill -INT $pid; wait $pid || true; }
q() { # q <cypher> [params-json]
  local body
  body=$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1], "params": json.loads(sys.argv[2])}))' "$1" "${2:-{\}}")
  local resp
  resp=$(curl -s --fail-with-body -H 'content-type: application/json' -d "$body" $url/v0/cypher) \
    || { echo "query failed: $1: $resp" >&2; return 1; }
  echo "$resp"
}
post() { # post <admin path>
  local resp
  resp=$(curl -s --fail-with-body -X POST "$url$1") || { echo "POST $1 failed: $resp" >&2; return 1; }
}
vec() { python3 -c 'import json,math,sys; i=int(sys.argv[1]); v=[math.sin(i*7+j) for j in range(16)]; n=math.sqrt(sum(x*x for x in v)); print(json.dumps([round(x/n,6) for x in v]))' "$1"; }
reads() { # the read set compared across binaries, one JSON document per line
  q "MATCH (d:Doc) RETURN count(d) AS docs"
  q "MATCH (s:Section) RETURN count(s) AS sections"
  q "MATCH (d:Doc {id: 'd3'})-[:HAS_SECTION]->(s:Section) RETURN s.id AS id ORDER BY id"
  q "MATCH (s:Section {id: 's42'}) RETURN s.body AS body, s.doc_id AS doc"
  q "CALL search.bm25({label: 'Section', text_property: 'body', query: 'jubilación régimen', k: 5}) YIELD node, score RETURN node.id AS id ORDER BY score DESC, id"
  q "CALL search.vector({label: 'Section', property: 'embedding', query: \$v, k: 5}) YIELD node RETURN node.id AS id ORDER BY id" "{\"v\": $(vec 7)}"
  q "MATCH (a:Doc)-[:AMENDS]->(b:Doc) RETURN count(*) AS amends"
}

echo "== new binary: $("$new" --version)"
serve "$new" $work/new.log
q "CREATE INDEX IF NOT EXISTS FOR (n:Section) ON (n.id)" >/dev/null
q "CREATE INDEX IF NOT EXISTS FOR (n:Doc) ON (n.id)" >/dev/null
q "CREATE FULLTEXT INDEX section_ft IF NOT EXISTS FOR (n:Section) ON EACH [n.body]" >/dev/null
q "CREATE VECTOR INDEX section_vec IF NOT EXISTS ON :Section(embedding) METRIC cosine DIMENSION 16" >/dev/null
q "UNWIND range(0, 19) AS i CREATE (:Doc {id: 'd' + toString(i)})" >/dev/null
rows=$(python3 -c '
import json, math
words = ["jubilación", "régimen", "pensión", "artículo", "código", "reforma", "tribunal", "derecho"]
out = []
for i in range(200):
    v = [math.sin(i * 7 + j) for j in range(16)]; n = math.sqrt(sum(x * x for x in v))
    out.append({"id": f"s{i}", "doc_id": f"d{i % 20}", "body": " ".join(words[(i * k) % 8] for k in range(1, 6)),
                "embedding": [round(x / n, 6) for x in v]})
print(json.dumps({"rows": out}))')
q "UNWIND \$rows AS r CREATE (s:Section) SET s = r" "$rows" >/dev/null
q "UNWIND range(0, 199) AS i MATCH (d:Doc {id: 'd' + toString(i % 20)}), (s:Section {id: 's' + toString(i)}) CREATE (d)-[:HAS_SECTION]->(s)" >/dev/null
q "UNWIND range(0, 18) AS i MATCH (a:Doc {id: 'd' + toString(i)}), (b:Doc {id: 'd' + toString(i + 1)}) CREATE (a)-[:AMENDS]->(b)" >/dev/null
post /v0/admin/flush
curl -sf -X POST "$url/v0/admin/compact" >/dev/null 2>&1 || true
# A WAL tail that is not flushed when the server stops.
q "CREATE (:Section {id: 's-tail', doc_id: 'd0', body: 'reforma tardía'})" >/dev/null
reads >$work/expected.txt
echo "read set written by the new binary:"; cut -c1-160 $work/expected.txt
stop

status=0
for old in "$@"; do
  echo "== old binary: $("$old" --version)"
  serve "$old" $work/old.log
  reads >$work/got.txt || { echo "FAIL: reads errored under $old"; tail -20 $work/old.log; status=1; stop; continue; }
  if diff -u $work/expected.txt $work/got.txt; then echo "reads identical"; else echo "FAIL: reads differ"; status=1; fi
  q "CREATE (:Section {id: 's-old', doc_id: 'd1', body: 'escrito por la versión anterior'})" >/dev/null \
    && echo "write accepted" || { echo "FAIL: write rejected"; status=1; }
  q "MATCH (s:Section {id: 's-old'}) DETACH DELETE s" >/dev/null
  curl -sf -X POST $url/v0/admin/flush >/dev/null || true
  stop
  # Roll forward again: the new binary opens what the old one wrote.
  serve "$new" $work/new2.log
  reads >$work/again.txt
  if diff -u $work/expected.txt $work/again.txt; then echo "roll-forward identical"; else echo "FAIL: roll-forward differs"; status=1; fi
  stop
done
[ $status -eq 0 ] && echo "PASS" || echo "FAILED"
exit $status
