# namidb (Python bindings)

Python wrapper for the NamiDB storage + query engine. Backed by Rust
via [`pyo3`](https://pyo3.rs/) and built with
[`maturin`](https://www.maturin.rs/).

## Install

```bash
pip install namidb              # released wheel — Python ≥ 3.9, abi3
pip install 'namidb[pandas]'    # + DataFrame interop
pip install 'namidb[polars]'    # + polars interop
```

Wheels are published for Linux (x86_64, aarch64), macOS (arm64), and
Windows (x86_64) via the `python-wheels.yml` workflow on every `py-v*`
tag — `pyarrow >= 14` is a hard transitive dependency. Intel macOS
users fall back to the sdist (slower install, same runtime behaviour).

## Build from source

```bash
pip install maturin
cd crates/namidb-py
maturin develop --release --extras test
```

Once `maturin develop` finishes, the `namidb` module is importable
from any Python ≥ 3.9 environment:

```python
import uuid
import namidb as tg

client = tg.Client("memory://acme")

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
current namespace and returns a `QueryResult`:

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

Cypher writes (CREATE / SET / DELETE / MERGE / REMOVE) are durably
committed (WAL append + manifest CAS) **before `cypher()` returns** —
the executor calls `commit_batch()` internally at the end of every
write plan. Call `client.flush()` periodically to push the memtable
into L0 SSTs. This is *different* from the `upsert_node` /
`upsert_edge` / `tombstone_*` API, which stages mutations and
requires an explicit `client.commit()`.

## Async API

The same surface is available as a Python coroutine via
`Client.acypher` for `asyncio` / `FastAPI` / `aiohttp` integration:

```python
import asyncio
import namidb as tg


async def main() -> None:
    client = tg.Client("memory://acme")
    await client.acypher("CREATE (p:Person {name: 'Alice'})")
    client.commit()
    result = await client.acypher(
        "MATCH (p:Person {name: $name}) RETURN p.name AS name",
        params={"name": "Alice"},
    )
    print(result.rows())


asyncio.run(main())
```

`acypher` is driven by the `pyo3-async-runtimes` tokio bridge — every
call runs on the same multi-threaded tokio runtime that backs the
synchronous API, so mixing the two from the same `Client` is safe.

## Type mapping (Cypher ↔ Python)

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
| `DateTime` (UTC µs)   | `datetime.datetime` UTC  |
| `Node(NodeValue)`     | `{"_kind": "node", "id", "label", "properties"}` |
| `Rel(RelValue)`       | `{"_kind": "rel", "edge_type", "src", "dst", "properties"}` |
| `Path`                | `list[Node\|Rel]` alternating |

`bool` is intentionally checked before `int` so that Python `True` /
`False` do not silently round-trip as `Integer(1)` / `Integer(0)`.

## Bulk inserts

`Client.merge_nodes` and `Client.merge_edges` batch many writes under
a single tokio-runtime + mutex round-trip. They are the right
ingestion path when you have thousands of rows (Cypher `CREATE`
parses + plans + executes per call):

```python
import uuid
import namidb as tg

client = tg.Client("memory://acme")

# Bulk insert: each row needs an "id" UUID string + arbitrary properties.
client.merge_nodes(
    "Person",
    [{"id": str(uuid.uuid4()), "name": f"p{i}", "age": 20 + i} for i in range(10_000)],
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

`merge_nodes` / `merge_edges` stage into the current batch (same
lifecycle as `upsert_*`) — call `client.commit()` to make the
mutations durable.

## Arrow / pandas / polars output

`pyarrow >= 14` ships as a hard dependency. Every `QueryResult` can
materialise as a `pyarrow.Table`; pandas / polars conversions
delegate to it.

```python
result = client.cypher(
    "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY p.age DESC"
)

table = result.to_arrow()              # pyarrow.Table
df = result.to_pandas()                # pandas.DataFrame  (needs pandas)
pl_df = result.to_polars()             # polars.DataFrame  (needs polars)
```

Column order follows the `RETURN` projection from the parsed plan
(not the runtime row's `BTreeMap` ordering), so `RETURN p.name AS
name, p.age AS age` always yields columns `["name", "age"]` even
when zero rows match.

Pandas and Polars are *optional* extras:

```bash
pip install 'namidb[pandas]'
pip install 'namidb[polars]'
```

Calling `to_polars()` without the polars extra raises a clear
`ImportError` pointing at the install command.

For label-wide scans you can skip the Cypher round-trip entirely:

```python
table = client.scan_label_arrow("Person")
# Columns: id, label, lsn, schema_version, then the union of property
# keys across the scanned views (missing keys filled with None).
```

## Storage backends

| URI scheme | Backend | Status |
|---|---|---|
| `memory://<ns>` | `object_store::memory::InMemory` | Stable. Ephemeral, single-process. |
| `file:///abs/dir?ns=<ns>` (or `file://./rel?ns=<ns>`) | NamiDB `LocalFileObjectStore` (wraps `LocalFileSystem` and adds manifest CAS via `flock` + atomic rename) | Stable. |
| `s3://<bucket>[/<prefix>]?ns=<ns>...` | `object_store::aws::AmazonS3` | Stable. AWS S3, Cloudflare R2, MinIO, Tigris, LocalStack — any S3-compatible service. |
| `gs://<bucket>[/<prefix>]?ns=<ns>` | `object_store::gcp::GoogleCloudStorage` | Stable. Auth via `GOOGLE_APPLICATION_CREDENTIALS` or `?service_account=…`. |
| `az://<account>/<container>[/<prefix>]?ns=<ns>` | `object_store::azure::MicrosoftAzure` | Stable. Auth via `AZURE_STORAGE_*` env vars; `?use_emulator=true` for Azurite. |

### Local filesystem

For development, single-machine deployments, and CI fixtures. Full
manifest CAS via per-namespace `flock` + atomic rename — passes the
same concurrency test suite as `s3://`.

```python
import namidb as tg

client = tg.Client("file:///var/lib/namidb?ns=prod")
# or relative
client = tg.Client("file://./data?ns=dev")
```

### AWS S3

```python
import namidb as tg

client = tg.Client(
    "s3://my-bucket/data?ns=prod"
    "&region=us-west-2"
)
```

Credentials are read from the standard AWS environment variables
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_DEFAULT_REGION`). Query-string `region=...` overrides the env.

### Cloudflare R2

```python
import namidb as tg

client = tg.Client(
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

client = tg.Client("gs://my-bucket/data?ns=prod")
```

### Azure Blob Storage

```python
import os
os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "myacct"
os.environ["AZURE_STORAGE_ACCESS_KEY"]   = "..."

client = tg.Client("az://myacct/mycontainer?ns=prod")
```

### LocalStack (local persistent storage)

```bash
docker run -p 4566:4566 -e SERVICES=s3 localstack/localstack
aws --endpoint-url=http://localhost:4566 s3 mb s3://namidb-dev
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
```

```python
client = tg.Client(
    "s3://namidb-dev?ns=local"
    "&endpoint=http://localhost:4566"
    "&allow_http=true"
    "&region=us-east-1"
)
```

The `allow_http=true` flag is required because LocalStack does not
serve TLS by default.

## Scope (v0)

- Six storage backends: `memory://`, `file://`, `s3://`, `gs://`,
  `az://`. All five non-memory backends share the same manifest CAS
  protocol (`If-Match` on object stores, `flock` + atomic rename on
  the filesystem) and the same single-writer-per-namespace epoch
  fencing.
- Synchronous Python API + async coroutine API (`acypher`). Under the
  hood every call drives a tokio runtime owned by the `Client`; the
  first call you make per process pays the bootstrap cost.
- The same SST + bloom cache used by the Rust read path
  ([`SstCache`](../namidb-storage/src/cache.rs)) is exposed via
  `client.cache_stats()` so application-level dashboards can graph
  hit rate.
- Cypher coverage matches the Rust engine: LDBC SNB Interactive IC01
  through IC12, factorized execution toggleable via
  `NAMIDB_FACTORIZE=1`. See the project [`README`](../../README.md)
  for the engine's surface and the RFCs in [`docs/rfc/`](../../docs/rfc/)
  for design details.

## Running the integration test (optional)

The pytest suite under `tests/` ships a LocalStack round-trip test
that is `@pytest.mark.skipif`-guarded on the
`NAMIDB_TEST_LOCALSTACK_BUCKET` env var. To enable it:

```bash
docker run -p 4566:4566 -e SERVICES=s3 localstack/localstack &
aws --endpoint-url=http://localhost:4566 s3 mb s3://namidb-it
export NAMIDB_TEST_LOCALSTACK_BUCKET=namidb-it
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
.venv/bin/pytest tests/test_uri.py::test_s3_localstack_round_trip -v
```

## Releasing to PyPI

1. Bump `version` in `crates/namidb-py/pyproject.toml` and
   `crates/namidb-py/Cargo.toml` (they must match).
2. Update `CHANGELOG.md` (or this README's release notes section).
3. Commit, then tag and push:
   ```bash
   git tag py-v0.1.0
   git push origin py-v0.1.0
   ```
4. `python-wheels.yml` builds 4 wheels (Linux x86_64/aarch64, macOS
   arm64, Windows x86_64) + sdist, smoke-tests one wheel on Python
   3.9 and 3.13, then publishes to PyPI via OIDC trusted publishing
   (configured once per account at
   https://pypi.org/manage/account/publishing/).

`py-v*` tag prefix keeps Python releases separate from any future
`v*` tags that mark engine / crate releases.
