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

Wheels go out for Linux (x86_64, aarch64), macOS (arm64), and Windows
(x86_64) from the `python-wheels.yml` workflow on every `py-v*` tag.
`pyarrow >= 14` is a hard transitive dependency. Intel macOS falls back
to the sdist, which is slower to install but behaves the same at runtime.

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

`Client.merge_nodes` and `Client.merge_edges` batch a lot of writes
behind a single tokio-runtime plus mutex round-trip. They're the right
ingestion path once you have thousands of rows, since Cypher `CREATE`
parses, plans and executes once per call:

```python
import uuid
import namidb

client = namidb.Client("memory://acme")

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
lifecycle as `upsert_*`), so call `client.commit()` to make the
mutations durable.

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

## Storage backends

| URI scheme | Backend | Status |
|---|---|---|
| `memory://<ns>` | `object_store::memory::InMemory` | Stable. Ephemeral, single-process. |
| `file:///abs/dir?ns=<ns>` (or `file://./rel?ns=<ns>`) | NamiDB `LocalFileObjectStore` (wraps `LocalFileSystem` and adds manifest CAS via `flock` + atomic rename) | Stable. |
| `s3://<bucket>[/<prefix>]?ns=<ns>...` | `object_store::aws::AmazonS3` | Stable. AWS S3, Cloudflare R2, MinIO, Tigris, LocalStack, any S3-compatible service. |
| `gs://<bucket>[/<prefix>]?ns=<ns>` | `object_store::gcp::GoogleCloudStorage` | Stable. Auth via `GOOGLE_APPLICATION_CREDENTIALS` or `?service_account=...`. |
| `az://<account>/<container>[/<prefix>]?ns=<ns>` | `object_store::azure::MicrosoftAzure` | Stable. Auth via `AZURE_STORAGE_*` env vars; `?use_emulator=true` for Azurite. |

### Local filesystem

For development, single-machine deployments, and CI fixtures. Full
manifest CAS via per-namespace `flock` plus atomic rename, and it passes
the same concurrency test suite as `s3://`.

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

## Scope (v0)

- Six storage backends: `memory://`, `file://`, `s3://`, `gs://`,
  `az://`. All five non-memory backends share the same manifest CAS
  protocol (`If-Match` on object stores, `flock` plus atomic rename on
  the filesystem) and the same single-writer-per-namespace epoch
  fencing.
- A synchronous Python API plus an async coroutine API (`acypher`).
  Under the hood every call drives a tokio runtime owned by the
  `Client`; the first call per process pays the bootstrap cost.
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
4. Commit and push the release commit, wait for `ci` and `python-wheels` to
   pass on `main`, then create both annotated tags on that exact commit:
   ```bash
   git tag -a "v$VERSION" -m "NamiDB $VERSION"
   git tag -a "py-v$VERSION" -m "namidb Python $VERSION"
   test "$(git rev-parse "v$VERSION^{commit}")" = \
        "$(git rev-parse "py-v$VERSION^{commit}")"
   git push --atomic origin "v$VERSION" "py-v$VERSION"
   ```
5. `python-wheels.yml` builds 4 wheels (Linux x86_64/aarch64, macOS
   arm64, Windows x86_64) plus the sdist, smoke-tests one wheel on
   Python 3.9 and 3.13, then publishes to PyPI via OIDC trusted
   publishing (set up once per account at
   https://pypi.org/manage/account/publishing/).

The `v*` tag creates the GitHub Release, prebuilt binaries, and container
images. The `py-v*` tag publishes the Python distribution to PyPI. The
release workflows reject either tag unless it exactly matches the Cargo,
lockfile, Python version, dated changelog entry, and bundled license.
