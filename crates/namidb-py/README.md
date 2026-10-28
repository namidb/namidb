# namidb (Python bindings)

Python wrapper around the NamiDB storage and query engine. The heavy
lifting is Rust, bridged with [`pyo3`](https://pyo3.rs/) and built with
[`maturin`](https://www.maturin.rs/).

## Install

```bash
pip install namidb              # released wheel, Python >= 3.9, abi3
pip install 'namidb[pandas]'    # + DataFrame interop
pip install 'namidb[polars]'    # + polars interop
```

Wheels go out for Linux (x86_64, aarch64), macOS (arm64, x86_64), and
Windows (x86_64) from the `python-wheels.yml` workflow on every `py-v*` tag.
`pyarrow >= 14` is a hard transitive dependency. Any other platform falls
back to the sdist, which needs a Rust toolchain to install.

## Build from source

```bash
pip install maturin
cd crates/namidb-py
maturin develop --release --extras test
```

Once `maturin develop` finishes, `namidb` imports from any Python >= 3.9
environment:

```python
import uuid
import namidb

client = namidb.Client("memory://acme")

alice = str(uuid.uuid7())
bob = str(uuid.uuid7())

client.upsert_node("Person", alice, {"name": "Alice", "age": 30})
client.upsert_node("Person", bob, {"name": "Bob"})
client.upsert_edge("KNOWS", alice, bob, {"since": 2020})
client.commit()
client.flush()

print(client.lookup_node("Person", alice))
# {'id': '...', 'label': 'Person', 'lsn': 1, 'schema_version': 0,
#  'properties': {'name': 'Alice', 'age': 30}}

print(client.scan_label("Person"))
print(client.out_edges("KNOWS", alice))
print(client.cache_stats())
```

## Cypher queries

`Client.cypher(query, params=None)` runs a Cypher query against the
current namespace and hands back a `QueryResult`:

```python
client.cypher("CREATE (a:Person {name: 'Alice', age: 30})")
client.cypher("CREATE (a:Person {name: 'Bob',   age: 25})")
client.commit()

result = client.cypher(
    "MATCH (p:Person) WHERE p.age > $min RETURN p.name AS name, p.age AS age",
    params={"min": 26},
)

print(result.columns)   # ['name', 'age']
print(len(result))      # 1
print(result.first())   # {'name': 'Alice', 'age': 30}
for row in result.rows():
    print(row)          # {'name': 'Alice', 'age': 30}
```

One thing to keep straight: Cypher writes (CREATE / SET / DELETE / MERGE
/ REMOVE) are durably committed (WAL append plus manifest CAS) *before*
`cypher()` returns, because the executor calls `commit_batch()` at the
end of every write plan. You still want to call `client.flush()` now and
then to push the memtable into L0 SSTs. That's different from the
`upsert_node` / `upsert_edge` / `tombstone_*` API, which stages mutations
and waits for an explicit `client.commit()`.

### Timeouts and row limits

`cypher` and `acypher` take two keyword-only limits:

```python
from namidb import QueryRowCapError, QueryTimeoutError

try:
    client.cypher("MATCH (n)-[]->(m) RETURN n, m", timeout_ms=5_000, max_rows=100_000)
except QueryTimeoutError:   # a TimeoutError subclass
    ...
except QueryRowCapError:    # a RuntimeError subclass
    ...
```

`timeout_ms` counts from the call, including time a write spends waiting
for the client's lock behind another write, or a read-only client's read
waits behind a `refresh()`. A write that times out is thrown away, not
committed.
`max_rows` fails a read once any step of the plan produces more rows than
that. Writes ignore it.

### Threads

Every blocking call releases the GIL while it waits on storage, so other
Python threads keep running. Reads (`cypher` reads, `lookup_node`,
`scan_*`, edge lookups) run on the latest committed snapshot without
waiting for the writer lock, so several threads can query one `Client` at
the same time, a long read does not hold up writes, and a long write (a
bulk load, say) does not hold up reads. A read sees each write once the
call that made it returns. Writes still go one at a time. A `sweep()`,
explicit or after a compaction, keeps every file a read still running in
another thread needs.

## Reading from another process

A namespace has one writer. Opening a plain `Client` claims the namespace
and fences whatever process was writing before, so a second process that
only wants to read should open it read-only:

```python
reader = namidb.Client("s3://bucket/data?ns=prod", read_only=True)
reader.cypher("MATCH (p:Person) RETURN count(p) AS n")

# Later, pick up whatever the writer committed since:
if reader.refresh():
    print("new data")
```

A read-only client never claims the writer epoch. It sees every commit up
to the moment it was opened (or last refreshed), including commits the
writer has not flushed yet, and stays on that version until `refresh()`,
or for at most an hour: a read on an older version refreshes first.
Every mutating call (`upsert_*`, `tombstone_*`, `merge_*`, `put_nodes`,
`load_*`, `commit`, `flush`, `compact`, `sweep`, write or DDL Cypher)
raises `PermissionError`. The namespace must already exist. A reader never
compacts or sweeps, so `spool_dir`, `auto_sweep_min_age_secs` and
`compaction_l0_trigger` do nothing there.

It keeps a small retention lease under `manifest/pins/` so the writer's
cleanup does not delete files it still reads. The client renews the lease
in the background, so a read that runs for a long time keeps its files.
While the lease lives, the writer cannot reclaim anything from that
version on, so once the version is an hour old and no read is using it the
lease is left to expire, and the next read refreshes first. A read also
refreshes first when the lease may have expired anyway (the process was
suspended, say). `close()`, or leaving a
`with namidb.Client(uri, read_only=True) as reader:` block, drops the
lease right away; otherwise it expires about 15 minutes after it was last
renewed, or after the process dies. Credentials that cannot write under
`manifest/pins/` get no lease, and then only the writer's sweep age
margin protects the reader's files. `refresh()` replays the writer's
unflushed WAL, so it costs more the longer the writer goes without a
flush.

## Vector and full-text indexes

Published wheels include the same persistent Vamana ANN and BM25 index
features as the server image. Their DDL works directly through the embedded
client:

```python
client.cypher(
    "CREATE VECTOR INDEX doc_emb ON :Doc(embedding) "
    "METRIC cosine DIMENSION 384"
)
client.cypher("CREATE FULLTEXT INDEX doc_text ON :Doc(title, body)")
client.cypher(
    "CREATE INDEX IF NOT EXISTS FOR (d:Doc) ON (d.tenant_id)"
)
client.cypher(
    "CREATE INDEX IF NOT EXISTS FOR (d:Doc) ON (d.vigente)"
)
```

Index bodies are built by an authoritative compaction. `compact()` prepares
the merge and the `.vg` / `.ft` bodies without holding the foreground writer
mutex, then briefly re-acquires it to install the new manifest:

```python
client.flush()
report = client.compact()
print(report["applied"], report["l0_before"], report["l0_after"])
```

`await client.acompact()` is the coroutine equivalent. Until a body is
materialized, and whenever freshness cannot be proven, vector and BM25
queries automatically use their exact flat fallback.

`search.vector` can apply metadata eligibility before the returned `k`, so a
selective slice does not get truncated out of an unfiltered top-k:

```python
hits = client.cypher(
    """
    CALL search.vector({
      label: 'Doc', property: 'embedding', query: $q, k: 10,
      filter: { tenant_id: $tenant, vigente: true }
    })
    YIELD node, score
    RETURN node, score
    """,
    params={"q": [0.1] * 384, "tenant": "laboral"},
).rows()
```

A scalar filter value means equality, a list means `IN`, and different keys
AND-combine. Indexed String/Boolean equality and `IN` clauses use native
vector-ordinal postings; unsupported or range clauses retain the exact
widening/flat-scan fallback. The index changes the cost, not the result.

## Async API

The same surface is available as a coroutine through `Client.acypher`,
for `asyncio` / `FastAPI` / `aiohttp`:

```python
import asyncio
import namidb


async def main() -> None:
    client = namidb.Client("memory://acme")
    await client.acypher("CREATE (p:Person {name: 'Alice'})")
    client.commit()
    result = await client.acypher(
        "MATCH (p:Person {name: $name}) RETURN p.name AS name",
        params={"name": "Alice"},
    )
    print(result.rows())


asyncio.run(main())
```

`acypher` rides the `pyo3-async-runtimes` tokio bridge. Every call runs
on the same multi-threaded tokio runtime that backs the synchronous API,
so mixing the two from one `Client` is fine.

## Type mapping (Cypher and Python)

Both `cypher` parameters and `QueryResult.rows()` follow the same
mapping:

| Cypher `RuntimeValue` | Python type             |
|-----------------------|--------------------------|
| `Null`                | `None`                   |
| `Bool`                | `bool`                   |
| `Integer`             | `int`                    |
| `Float`               | `float`                  |
| `String`              | `str`                    |
| `Bytes`               | `bytes`                  |
| `Vector(Vec<f32>)`    | `list[float]`            |
| `List`                | `list`                   |
| `Map`                 | `dict[str, ...]`         |
| `Date`                | `datetime.date`          |
| `DateTime` (UTC, microseconds) | `datetime.datetime` UTC  |
| `Node(NodeValue)`     | `{"_kind": "node", "id", "label", "properties"}` |
| `Rel(RelValue)`       | `{"_kind": "rel", "edge_type", "src", "dst", "properties"}` |
| `Path`                | `list[Node\|Rel]` alternating |

`bool` is checked before `int` on purpose, so that Python `True` /
`False` don't quietly round-trip as `Integer(1)` / `Integer(0)`.

## Bulk inserts

`Client.put_nodes`, `Client.merge_nodes` and `Client.merge_edges` batch
a lot of writes behind a single tokio-runtime plus mutex round-trip.
They're the right ingestion path once you have thousands of rows, since
Cypher `CREATE` parses, plans and executes once per call:

```python
import uuid
import namidb

client = namidb.Client("memory://acme")

# First load: each row needs an "_id" UUID string + arbitrary properties.
client.put_nodes(
    "Person",
    [{"_id": str(uuid.uuid4()), "name": f"p{i}", "age": 20 + i} for i in range(10_000)],
)
# Edges: each row needs "src" + "dst" UUIDs.
client.merge_edges(
    "KNOWS",
    [
        {"src": "uuid-a", "dst": "uuid-b", "since": 2020},
        {"src": "uuid-b", "dst": "uuid-c", "since": 2021},
    ],
)
client.commit()        # WAL + manifest CAS
client.flush()         # memtable -> L0 SSTs
```

`put_nodes` replaces each node: the row is its full property set and
`label` its only label. Nothing is read first, so it is the fast path for
first loads. `merge_nodes` reads each node and merges the row into it:
keys the row doesn't mention are kept, a `None` value removes the key,
and `label` is added to the node's existing labels. Rows repeating an id
in one call merge in order.

`_id` is the internal node id, the same thing `n._id` returns in Cypher.
It is not stored as a property, so a business key called `id` can sit
next to it. Rows keyed by `id` alone (the older form) still work: `id`
becomes the node id and is also kept as an `id` property, so
`WHERE n.id = $id` finds the node.

`put_nodes` / `merge_nodes` / `merge_edges` stage into the current batch
(same lifecycle as `upsert_*`), so call `client.commit()` to make the
mutations durable.

## Compaction and disk space

There is no background maintenance in embedded mode, so the client does
it inline:

- `flush()` runs a compaction pass once any bucket holds
  `compaction_l0_trigger` L0 SSTs (default 8). Each extra L0 file is one
  more file every read has to visit. Pass `compaction_l0_trigger=0` to
  turn it off and call `compact()` yourself.
- A compaction that applies is followed by a sweep that deletes what it
  retired: the merged-away L0 files, WAL segments that flushes already
  folded into SSTs, old manifest versions. Only objects written more than
  `auto_sweep_min_age_secs` ago (default 60) go. That age counts from when
  the file was written, not from when it stopped being used, so it is no
  grace period for a reader in another process: a server, the CLI or a
  read-only client that loaded the manifest before the pass can find its
  files gone. If another process reads the graph while this client
  writes, pass `None` to turn the auto sweep off and call `sweep()` when
  the readers are done.
- `client.sweep(min_age_secs=60, dry_run=False)` runs the same sweep on
  demand and returns counts of what it removed (or would remove). The
  auto sweep runs right after the pass, so that pass's own leftovers are
  usually still too young. Call `sweep()` once more at the end of a bulk
  load, or `sweep(min_age_secs=0)` if no other process reads the graph.
- If the compaction a `flush()` triggered fails, the flush still counts
  (it is already durable) and you get a `RuntimeWarning` instead of an
  exception.

Compaction writes scratch files while it merges. For a `file://` URI they
go under `<root>/.namidb-spool`, next to the graph, unless you pass
`spool_dir=` or set `NAMIDB_SPOOL_DIR`. Other backends use
`NAMIDB_SPOOL_DIR`, else `/var/tmp`. Plan for roughly three times the
compacted bytes on that volume while a big pass runs. The spool dir is
per process, so the last client opened decides it for every client in
the process, and a spool dir that has been deleted falls back to the
default.

```python
client = namidb.Client(
    "file:///data/graph?ns=case",
    spool_dir="/data/scratch",
    compaction_l0_trigger=8,
    auto_sweep_min_age_secs=60,
)
```

## Arrow / pandas / polars output

`pyarrow >= 14` is a hard dependency. Every `QueryResult` can
materialise as a `pyarrow.Table`, and the pandas / polars conversions
just delegate to that.

```python
result = client.cypher(
    "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY p.age DESC"
)

table = result.to_arrow()              # pyarrow.Table
df = result.to_pandas()                # pandas.DataFrame  (needs pandas)
pl_df = result.to_polars()             # polars.DataFrame  (needs polars)
```

Column order follows the `RETURN` projection from the parsed plan, not
the runtime row's `BTreeMap` ordering, so `RETURN p.name AS name, p.age
AS age` always gives you columns `["name", "age"]` even when nothing
matches.

pandas and polars are optional extras:

```bash
pip install 'namidb[pandas]'
pip install 'namidb[polars]'
```

Calling `to_polars()` without the polars extra raises a clear
`ImportError` that points at the install command.

For label-wide scans you can skip the Cypher round-trip:

```python
table = client.scan_label_arrow("Person")
# Columns: id, label, lsn, schema_version, then the union of property
# keys across the scanned views (missing keys filled with None).
```

On a large label, pass `limit=` to `scan_label`, `scan_label_arrow` or
`scan_edge_type` to look at a sample, or walk the scan with
`iter_label(label)` / `iter_edge_type(edge_type)`. The iterators build one
dict per step instead of a list of every row, so Python memory stays flat.
The scan itself is still read in full before the first row comes back.

## Storage backends

| URI scheme | Backend | Status |
|---|---|---|
| `memory://<ns>` | `object_store::memory::InMemory` | Stable. Ephemeral, single-process. |
| `file:///abs/dir?ns=<ns>` (or `file://./rel?ns=<ns>`) | NamiDB `LocalFileObjectStore` over `LocalFileSystem`; Create-only pointer CAS uses `O_CREAT\|O_EXCL` | Stable. |
| `s3://<bucket>[/<prefix>]?ns=<ns>...` | `object_store::aws::AmazonS3` | Stable. AWS S3, Cloudflare R2, MinIO, Tigris, LocalStack, any S3-compatible service. |
| `gs://<bucket>[/<prefix>]?ns=<ns>` | `object_store::gcp::GoogleCloudStorage` | Stable. Auth via `GOOGLE_APPLICATION_CREDENTIALS` or `?service_account=...`. |
| `az://<account>/<container>[/<prefix>]?ns=<ns>` | `object_store::azure::MicrosoftAzure` | Stable. Auth via `AZURE_STORAGE_*` env vars; `?use_emulator=true` for Azurite. |

### Local filesystem

For development, single-machine deployments, and CI fixtures. Full
manifest CAS uses the same Create-only versioned pointer family as the cloud
backends, with local `O_CREAT|O_EXCL` providing PUT-if-absent. It passes the
same concurrency test suite as `s3://`.

```python
import namidb

client = namidb.Client("file:///var/lib/namidb?ns=prod")
# or relative
client = namidb.Client("file://./data?ns=dev")
```

### AWS S3

```python
import namidb

client = namidb.Client(
    "s3://my-bucket/data?ns=prod"
    "&region=us-west-2"
)
```

Credentials come from the standard AWS environment variables
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_DEFAULT_REGION`). A query-string `region=...` overrides the env.

### Cloudflare R2

```python
import namidb

client = namidb.Client(
    "s3://my-bucket?ns=prod"
    "&endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
    "&region=auto"
)
```

`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` should hold the R2 API
token credentials.

### Google Cloud Storage

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs-key.json"

client = namidb.Client("gs://my-bucket/data?ns=prod")
```

### Azure Blob Storage

```python
import os
os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "myacct"
os.environ["AZURE_STORAGE_ACCESS_KEY"]   = "..."

client = namidb.Client("az://myacct/mycontainer?ns=prod")
```

### LocalStack (local persistent storage)

```bash
docker run -p 4566:4566 -e SERVICES=s3 localstack/localstack
aws --endpoint-url=http://localhost:4566 s3 mb s3://namidb-dev
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
```

```python
client = namidb.Client(
    "s3://namidb-dev?ns=local"
    "&endpoint=http://localhost:4566"
    "&allow_http=true"
    "&region=us-east-1"
)
```

You need `allow_http=true` because LocalStack doesn't serve TLS by
default.

## Scope

- Five storage backends: `memory://`, `file://`, `s3://`, `gs://`,
  `az://`. All four non-memory backends share the same Create-only
  versioned manifest-pointer protocol: each `p<N>.json` is written once
  with PUT-if-absent (`If-None-Match: *` on object stores,
  `O_CREAT|O_EXCL` locally), the highest `N` is current, and the same
  single-writer-per-namespace epoch fencing applies.
- A synchronous Python API plus async coroutine APIs (`acypher`,
  `acompact`).
  Under the hood every call drives a tokio runtime owned by the
  `Client` with the GIL released; the first call per process pays the
  bootstrap cost.
- One writer per namespace, any number of `read_only=True` readers in
  other processes.
- Persistent Vamana vector and BM25 full-text indexes, including their
  Cypher DDL and off-writer-lock compaction lifecycle.
- The same SST plus bloom cache the Rust read path uses
  ([`SstCache`](../namidb-storage/src/cache.rs)) is exposed through
  `client.cache_stats()`, so application dashboards can graph the hit
  rate.
- Cypher coverage matches the Rust engine: LDBC SNB Interactive IC01
  through IC12, with factorized execution toggled by
  `NAMIDB_FACTORIZE=1`. See the project [`README`](../../README.md) for
  the engine's surface and the RFCs in [`docs/rfc/`](../../docs/rfc/)
  for the design details.

## Running the integration test (optional)

The pytest suite under `tests/` ships a LocalStack round-trip test that
is `@pytest.mark.skipif`-guarded on the `NAMIDB_TEST_LOCALSTACK_BUCKET`
env var. To turn it on:

```bash
docker run -p 4566:4566 -e SERVICES=s3 localstack/localstack &
aws --endpoint-url=http://localhost:4566 s3 mb s3://namidb-it
export NAMIDB_TEST_LOCALSTACK_BUCKET=namidb-it
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
.venv/bin/pytest tests/test_uri.py::test_s3_localstack_round_trip -v
```

## Releasing to PyPI

Run every command below from the repository root.

1. Set the same `X.Y.Z` in the root `Cargo.toml`
   (`workspace.package.version` and every internal workspace dependency pin)
   and in `crates/namidb-py/pyproject.toml`. Every crate, including
   `namidb-py`, inherits its Rust version from the root workspace manifest.
2. Refresh `Cargo.lock` and update `CHANGELOG.md`.
3. Validate the complete release metadata before creating an immutable tag:
   ```bash
   VERSION=X.Y.Z
   python scripts/check-release-metadata.py \
     --tag "v$VERSION" --tag-kind engine
   python scripts/check-release-metadata.py \
     --tag "py-v$VERSION" --tag-kind python
   ```
4. Commit and push the release commit and wait for `ci` and `python-wheels` to
   pass on `main`. Before creating an immutable tag, manually dispatch
   `release-binaries.yml` (full OS/architecture release dry-run) and
   `docker-image.yml` with an empty `version` (native amd64/arm64 build-check)
   on that same commit, and wait for both to pass.
5. Create both annotated tags on that exact commit:
   ```bash
   git tag -a "v$VERSION" -m "NamiDB $VERSION"
   git tag -a "py-v$VERSION" -m "namidb Python $VERSION"
   test "$(git rev-parse "v$VERSION^{commit}")" = \
        "$(git rev-parse "py-v$VERSION^{commit}")"
   git push --atomic origin "v$VERSION" "py-v$VERSION"
   ```
6. `python-wheels.yml` builds 5 wheels (Linux x86_64/aarch64, macOS
   arm64/x86_64, Windows x86_64) plus the sdist, smoke-tests one wheel on
   Python 3.9 and 3.13, then publishes to PyPI via OIDC trusted
   publishing (set up once per account at
   https://pypi.org/manage/account/publishing/).

The `v*` tag creates the GitHub Release, prebuilt binaries, and container
images. The `py-v*` tag publishes the Python distribution to PyPI. The
release workflows reject either tag unless it exactly matches the Cargo,
lockfile, Python version, dated changelog entry, and bundled license.
