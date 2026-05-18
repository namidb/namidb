"""S19.C — URI scheme parsing + S3 backend (LocalStack integration).

Most tests exercise the URI parsing surface (which fails before the
client actually bootstraps storage). The opt-in LocalStack test at
the bottom validates the full s3:// round-trip — set
`NAMIDB_TEST_LOCALSTACK_BUCKET=<pre-created-bucket>` to enable it.
"""

from __future__ import annotations

import os
import uuid

import pytest

import namidb as tg


# ── memory:// regression (still works after refactor) ──────────────────


def test_memory_uri_still_works() -> None:
    client = tg.Client(f"memory://{uuid.uuid4().hex[:8]}")
    client.cypher("CREATE (p:Person {name: 'Alice'})")
    result = client.cypher("MATCH (p:Person) RETURN p.name AS name")
    assert result.first() == {"name": "Alice"}


def test_memory_uri_missing_namespace_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        tg.Client("memory://")
    assert "namespace" in str(exc_info.value).lower()


# ── file:// — intentionally unsupported in v0 ──────────────────────────


def test_file_uri_raises_with_helpful_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        tg.Client("file:///tmp/some-path?ns=acme")
    msg = str(exc_info.value)
    assert "file://" in msg
    assert "PutMode::Update" in msg
    # Suggest the workarounds.
    assert "memory://" in msg
    assert "s3://" in msg


# ── gs:// / az:// — planned ────────────────────────────────────────────


def test_gs_uri_raises_helpful_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        tg.Client("gs://my-bucket?ns=acme")
    assert "gs" in str(exc_info.value).lower()


def test_az_uri_raises_helpful_message() -> None:
    with pytest.raises(ValueError) as exc_info:
        tg.Client("az://my-account/my-container?ns=acme")
    assert "az" in str(exc_info.value).lower()


def test_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        tg.Client("ftp://wat?ns=acme")
    assert "unsupported URI scheme" in str(exc_info.value)


# ── s3:// — URI shape validation (no live connection needed) ───────────


def test_s3_uri_missing_bucket_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        tg.Client("s3://?ns=acme")
    assert "bucket" in str(exc_info.value).lower()


def test_s3_uri_missing_namespace_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        tg.Client("s3://my-bucket")
    assert "ns" in str(exc_info.value).lower()


def test_s3_uri_invalid_namespace_raises() -> None:
    # Namespace must be DNS-safe — capital letters / special chars fail.
    with pytest.raises(ValueError) as exc_info:
        tg.Client("s3://my-bucket?ns=INVALID_NAMESPACE_WITH_CAPS")
    assert "namespace" in str(exc_info.value).lower()


# ── s3:// — LocalStack integration (opt-in via env var) ────────────────


@pytest.mark.skipif(
    not os.environ.get("NAMIDB_TEST_LOCALSTACK_BUCKET"),
    reason="opt-in LocalStack test; set NAMIDB_TEST_LOCALSTACK_BUCKET=<bucket>",
)
def test_s3_localstack_round_trip() -> None:
    """End-to-end against LocalStack:

    1. CREATE a node.
    2. Flush memtable to L0 SSTs (forces a real S3 round-trip).
    3. Drop the Client (closes the writer + frees the memtable).
    4. Re-open the same URI from a fresh Client.
    5. MATCH and verify the node survived.

    Requires:
    - LocalStack running on http://localhost:4566 (override with
      NAMIDB_TEST_LOCALSTACK_ENDPOINT).
    - A pre-created bucket named via
      `NAMIDB_TEST_LOCALSTACK_BUCKET`.
    - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` set in env
      (`test` / `test` works for LocalStack).
    """
    bucket = os.environ["NAMIDB_TEST_LOCALSTACK_BUCKET"]
    endpoint = os.environ.get(
        "NAMIDB_TEST_LOCALSTACK_ENDPOINT", "http://localhost:4566"
    )
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    ns = f"ns-{uuid.uuid4().hex[:8]}"
    uri = (
        f"s3://{bucket}?ns={ns}"
        f"&endpoint={endpoint}"
        f"&region={region}"
        f"&allow_http=true"
    )

    client = tg.Client(uri)
    client.cypher("CREATE (p:Person {name: 'Alice', age: 30})")
    client.flush()
    del client

    client2 = tg.Client(uri)
    result = client2.cypher("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
    assert result.first() == {"name": "Alice", "age": 30}
