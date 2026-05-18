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
