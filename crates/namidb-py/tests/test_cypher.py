"""S19.A — Cypher execution surface tests.

These tests cover Client.cypher (sync) + Client.acypher (async via the
pyo3-async-runtimes tokio bridge), QueryResult shape, parameter type
conversions, and the stubbed Arrow / pandas / polars output paths.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

import namidb as tg


# ── basic cypher round-trip ────────────────────────────────────────────


def test_simple_create_then_match(client: tg.Client) -> None:
    client.cypher("CREATE (a:Person {name: 'Alice', age: 30})")
    client.commit()
    result = client.cypher("MATCH (p:Person) RETURN p.name AS name, p.age AS age")
    assert len(result) == 1
    assert sorted(result.columns) == ["age", "name"]
    row = result.first()
    assert row == {"name": "Alice", "age": 30}


def test_match_returns_empty(client: tg.Client) -> None:
    result = client.cypher("MATCH (p:NoSuchLabel) RETURN p.name AS name")
    assert len(result) == 0
    # Columns come from the plan's Project items, so the schema is known
    # even when zero rows match.
    assert result.columns == ["name"]
    assert result.first() is None
    assert result.rows() == []


def test_count_aggregation(people_client: tg.Client) -> None:
    result = people_client.cypher("MATCH (p:Person) RETURN count(*) AS n")
    assert len(result) == 1
    assert result.first() == {"n": 2}


def test_query_result_repr(people_client: tg.Client) -> None:
    result = people_client.cypher("MATCH (p:Person) RETURN p.name AS name")
    assert "rows=2" in repr(result)
    assert "name" in repr(result)


# ── parameters ─────────────────────────────────────────────────────────


def test_int_param_filter(people_client: tg.Client) -> None:
    result = people_client.cypher(
        "MATCH (p:Person) WHERE p.age > $min RETURN p.name AS name",
        params={"min": 26},
    )
    assert len(result) == 1
    assert result.first() == {"name": "Alice"}


def test_str_param(people_client: tg.Client) -> None:
    result = people_client.cypher(
        "MATCH (p:Person {name: $name}) RETURN p.age AS age",
        params={"name": "Bob"},
    )
    assert result.first() == {"age": 25}


def test_bool_param_distinct_from_int(client: tg.Client) -> None:
    """Sanity: True/False round-trip as Bool, not as Integer(1)/Integer(0)."""
    # Cypher boolean literal comparison
    result = client.cypher("RETURN $flag AS x", params={"flag": True})
    assert result.first() == {"x": True}
    result = client.cypher("RETURN $flag AS x", params={"flag": False})
    assert result.first() == {"x": False}


def test_none_param(client: tg.Client) -> None:
    result = client.cypher("RETURN $v AS x", params={"v": None})
    assert result.first() == {"x": None}


def test_float_param(client: tg.Client) -> None:
    result = client.cypher("RETURN $v AS x", params={"v": 3.14})
    row = result.first()
    assert row is not None
    assert row["x"] == pytest.approx(3.14)


def test_bytes_param(client: tg.Client) -> None:
    result = client.cypher("RETURN $v AS x", params={"v": b"\x00\x01\x02"})
    assert result.first() == {"x": b"\x00\x01\x02"}


def test_list_param(client: tg.Client) -> None:
    result = client.cypher("RETURN $v AS x", params={"v": [1, 2, 3]})
    assert result.first() == {"x": [1, 2, 3]}


def test_dict_param(client: tg.Client) -> None:
    result = client.cypher("RETURN $v AS x", params={"v": {"a": 1, "b": "two"}})
    assert result.first() == {"x": {"a": 1, "b": "two"}}


def test_datetime_param_roundtrips_to_utc(client: tg.Client) -> None:
    when = dt.datetime(2026, 5, 18, 12, 34, 56, tzinfo=dt.timezone.utc)
    result = client.cypher("RETURN $when AS x", params={"when": when})
    row = result.first()
    assert row is not None
    assert isinstance(row["x"], dt.datetime)
    # Microsecond precision preserved (DateTime is stored as i64 µs).
    assert row["x"] == when


def test_date_param_roundtrips(client: tg.Client) -> None:
    today = dt.date(2026, 5, 18)
    result = client.cypher("RETURN $d AS x", params={"d": today})
    row = result.first()
    assert row is not None
    assert isinstance(row["x"], dt.date)
    assert row["x"] == today


# ── error mapping ──────────────────────────────────────────────────────


def test_parse_error_is_value_error(client: tg.Client) -> None:
    with pytest.raises(ValueError) as exc_info:
        client.cypher("SELECT * FROM nope")
    assert "parse error" in str(exc_info.value).lower()


def test_lower_error_is_value_error(client: tg.Client) -> None:
    # Reference to undefined variable — caught at lowering.
    with pytest.raises(ValueError):
        client.cypher("RETURN nonexistent.name")


# ── async (acypher) ────────────────────────────────────────────────────


def test_acypher_simple_match() -> None:
    """Async sibling resolves to the same QueryResult shape."""

    async def run() -> dict:
        client = tg.Client("memory://async-simple")
        await client.acypher("CREATE (a:Person {name: 'Alice', age: 30})")
        client.commit()
        result = await client.acypher(
            "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
        )
        assert len(result) == 1
        return result.first()

    row = asyncio.run(run())
    assert row == {"name": "Alice", "age": 30}


def test_acypher_with_params() -> None:
    async def run() -> dict | None:
        client = tg.Client("memory://async-params")
        await client.acypher(
            "CREATE (a:Person {name: 'Alice', age: 30}), (b:Person {name: 'Bob', age: 25})"
        )
        client.commit()
        result = await client.acypher(
            "MATCH (p:Person) WHERE p.age >= $min RETURN p.name AS name",
            params={"min": 30},
        )
        return result.first()

    row = asyncio.run(run())
    assert row == {"name": "Alice"}


def test_acypher_parse_error_propagates_as_value_error() -> None:
    async def run() -> None:
        client = tg.Client("memory://async-err")
        await client.acypher("SELECT * FROM nope")

    with pytest.raises(ValueError):
        asyncio.run(run())


# ── write semantics — cypher writes auto-commit (execute_write batches) ─


def test_cypher_write_durable_without_explicit_commit(client: tg.Client) -> None:
    """execute_write calls commit_batch() internally; no client.commit()
    is needed between Cypher write and Cypher read on the same client."""
    client.cypher("CREATE (a:Person {name: 'Alice'})")
    # Intentionally NO client.commit() — the write is already durable.
    result = client.cypher("MATCH (p:Person) RETURN p.name AS name")
    assert result.first() == {"name": "Alice"}


def test_create_with_relationship(client: tg.Client) -> None:
    client.cypher(
        "CREATE (a:Person {name: 'Ada'})-[r:KNOWS {weight: 5}]->(b:Person {name: 'Lin'})"
    )
    result = client.cypher(
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) "
        "RETURN a.name AS src, b.name AS dst, r.weight AS w"
    )
    assert len(result) == 1
    assert result.first() == {"src": "Ada", "dst": "Lin", "w": 5}


# ── schema DDL / introspection on the embedded client ──────────────────


def test_embedded_ddl_constraint_index_and_show(client: tg.Client) -> None:
    # Schema DDL runs directly on the embedded client — no Bolt/HTTP round-trip.
    client.cypher(
        "CREATE CONSTRAINT cfg_uq FOR (n:Cfg) REQUIRE (n.tenant, n.name) IS UNIQUE"
    )
    client.cypher("CREATE INDEX FOR (n:Doc) ON (n.slug)")
    # Re-running with IF NOT EXISTS is a no-op (must not raise).
    client.cypher(
        "CREATE CONSTRAINT cfg_uq IF NOT EXISTS "
        "FOR (n:Cfg) REQUIRE (n.tenant, n.name) IS UNIQUE"
    )

    cons = {row["name"]: row for row in client.cypher("SHOW CONSTRAINTS").rows()}
    assert "cfg_uq" in cons
    assert cons["cfg_uq"]["properties"] == ["tenant", "name"]
    assert cons["cfg_uq"]["type"] == "UNIQUENESS"
    assert cons["cfg_uq"]["labelsOrTypes"] == ["Cfg"]

    idx_labels = [row["labelsOrTypes"][0] for row in client.cypher("SHOW INDEXES").rows()]
    assert "Doc" in idx_labels


def test_embedded_vector_and_fulltext_ddl_lifecycle(client: tg.Client) -> None:
    vector_ddl = (
        "CREATE VECTOR INDEX doc_emb ON :Doc(embedding) "
        "METRIC cosine DIMENSION 3"
    )
    fulltext_ddl = "CREATE FULLTEXT INDEX doc_text ON :Doc(body, title)"

    assert client.cypher(vector_ddl).rows() == []
    assert client.cypher(fulltext_ddl).rows() == []

    indexes = {row["name"]: row for row in client.cypher("SHOW INDEXES").rows()}
    assert indexes["doc_emb"]["type"] == "VECTOR"
    assert indexes["doc_emb"]["labelsOrTypes"] == ["Doc"]
    assert indexes["doc_emb"]["properties"] == ["embedding"]
    assert indexes["doc_text"]["type"] == "FULLTEXT"
    assert indexes["doc_text"]["labelsOrTypes"] == ["Doc"]
    assert indexes["doc_text"]["properties"] == ["body", "title"]

    # Duplicate name/target is an error unless explicitly idempotent.
    with pytest.raises(RuntimeError, match="already exists"):
        client.cypher(vector_ddl)
    with pytest.raises(RuntimeError, match="already exists"):
        client.cypher(fulltext_ddl)
    client.cypher(
        "CREATE VECTOR INDEX doc_emb IF NOT EXISTS ON :Doc(embedding) "
        "METRIC cosine DIMENSION 3"
    )
    client.cypher(
        "CREATE FULLTEXT INDEX doc_text IF NOT EXISTS ON :Doc(body, title)"
    )

    client.cypher("DROP VECTOR INDEX doc_emb")
    client.cypher("DROP FULLTEXT INDEX doc_text")
    assert {
        row["name"] for row in client.cypher("SHOW INDEXES").rows()
    }.isdisjoint({"doc_emb", "doc_text"})

    with pytest.raises(RuntimeError, match="no vector index"):
        client.cypher("DROP VECTOR INDEX doc_emb")
    with pytest.raises(RuntimeError, match="no text index"):
        client.cypher("DROP INDEX doc_text")
    client.cypher("DROP VECTOR INDEX doc_emb IF EXISTS")
    client.cypher("DROP INDEX doc_text IF EXISTS")

    # Both full-text DROP spellings free the descriptor slot.
    client.cypher(fulltext_ddl)
    client.cypher("DROP INDEX doc_text")
    client.cypher(fulltext_ddl)
    client.cypher("DROP FULLTEXT INDEX doc_text")


def test_vector_ddl_rejects_invalid_metric_and_dimension(client: tg.Client) -> None:
    # Parser-level validation: int8 navigation is cosine-only.
    with pytest.raises(ValueError, match="int8 quantization requires METRIC cosine"):
        client.cypher(
            "CREATE VECTOR INDEX bad_int8 ON :Doc(embedding) "
            "METRIC dot DIMENSION 3 WITH {quantization: int8}"
        )

    client.cypher(
        "CREATE VECTOR INDEX doc_emb ON :Doc(embedding) "
        "METRIC cosine DIMENSION 3"
    )
    # A Python numeric list is accepted as a vector parameter and coerced by
    # the indexed write path, but the declared dimension remains mandatory.
    with pytest.raises(RuntimeError, match=r"dim 2.*declares 3"):
        client.cypher(
            "CREATE (:Doc {title: 'bad', embedding: $embedding})",
            params={"embedding": [1.0, 0.0]},
        )
    client.cypher(
        "CREATE (:Doc {title: 'ok', embedding: $embedding})",
        params={"embedding": [1.0, 0.0, 0.0]},
    )
    result = client.cypher(
        "CALL search.vector({label: 'Doc', property: 'embedding', "
        "query: $query, k: 1}) "
        "YIELD node, score RETURN node.title AS title, score",
        params={"query": [1.0, 0.0, 0.0]},
    )
    assert result.first()["title"] == "ok"


def test_compaction_materializes_vector_and_text_indexes(client: tg.Client) -> None:
    client.cypher(
        "CREATE VECTOR INDEX doc_emb ON :Doc(embedding) "
        "METRIC cosine DIMENSION 3"
    )
    client.cypher("CREATE FULLTEXT INDEX doc_text ON :Doc(body)")

    # Two flushes create two node L0 inputs, which makes the next authoritative
    # compaction build one Nodes L1 plus the `.vg` and `.ft` bodies.
    client.cypher(
        "CREATE (:Doc {title: 'alpha', body: 'fox jumps quickly', "
        "embedding: vector([1.0, 0.0, 0.0])}), "
        "(:Doc {title: 'beta', body: 'database storage engine', "
        "embedding: vector([0.0, 1.0, 0.0])})"
    )
    client.flush()
    client.cypher(
        "CREATE (:Doc {title: 'gamma', body: 'fox graph database', "
        "embedding: vector([0.8, 0.2, 0.0])})"
    )
    client.flush()

    vector_query = (
        "CALL search.vector({label: 'Doc', property: 'embedding', "
        "query: $query, k: 2}) "
        "YIELD node, score RETURN node.title AS title, score"
    )
    bm25_query = (
        "CALL search.bm25({label: 'Doc', text_property: 'body', "
        "query: 'fox', k: 10}) "
        "YIELD node, score RETURN node.title AS title, score"
    )

    # With descriptors but no immutable index body yet, both procedures use
    # their exact flat fallback.
    vector_before = [
        row["title"]
        for row in client.cypher(
            vector_query, params={"query": [1.0, 0.0, 0.0]}
        ).rows()
    ]
    bm25_before = {
        row["title"] for row in client.cypher(bm25_query).rows()
    }

    report = client.compact()
    assert report["applied"] is True
    assert report["manifest_version_after"] > report["manifest_version_before"]
    assert report["l0_before"] >= 2
    assert report["l0_after"] < report["l0_before"]
    assert report["source_ssts_removed"] >= 2
    # Nodes L1 + one VectorGraph + one TextIndex.
    assert report["new_ssts_written"] >= 3

    # The now-authoritative bodies are consumed by the feature-backed paths and
    # remain result-equivalent to the flat baseline.
    vector_after = [
        row["title"]
        for row in client.cypher(
            vector_query, params={"query": [1.0, 0.0, 0.0]}
        ).rows()
    ]
    bm25_after = {
        row["title"] for row in client.cypher(bm25_query).rows()
    }
    assert vector_before == vector_after == ["alpha", "gamma"]
    assert bm25_before == bm25_after == {"alpha", "gamma"}

    # A second pass over an already-compacted namespace is a structured no-op.
    noop = client.compact()
    assert noop["applied"] is False
    assert noop["source_ssts_removed"] == 0
    assert noop["new_ssts_written"] == 0


def test_acompact_returns_the_same_report_shape(client: tg.Client) -> None:
    async def run() -> dict:
        return await client.acompact()

    report = asyncio.run(run())
    assert report == {
        "applied": False,
        "manifest_version_before": report["manifest_version_before"],
        "manifest_version_after": report["manifest_version_after"],
        "l0_before": 0,
        "l0_after": 0,
        "source_ssts_removed": 0,
        "new_ssts_written": 0,
        "bloom_sidecars_written": 0,
    }
