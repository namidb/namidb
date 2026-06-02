"""S19.B — Bulk APIs (merge_nodes / merge_edges) + Arrow output."""

from __future__ import annotations

import datetime as dt
import uuid

import pyarrow as pa
import pytest

import namidb as tg


# ── merge_nodes / merge_edges bulk insert ──────────────────────────────


def test_merge_nodes_basic(client: tg.Client) -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    n = client.merge_nodes(
        "Person",
        [
            {"id": ids[0], "name": "Alice", "age": 30},
            {"id": ids[1], "name": "Bob", "age": 25},
            {"id": ids[2], "name": "Carol", "age": 42},
        ],
    )
    assert n == 3
    client.commit()
    result = client.cypher("MATCH (p:Person) RETURN count(*) AS n")
    assert result.first() == {"n": 3}


def test_merge_nodes_missing_id_raises(client: tg.Client) -> None:
    with pytest.raises(ValueError) as exc_info:
        client.merge_nodes("Person", [{"name": "no id"}])
    assert "'id'" in str(exc_info.value)


def test_merge_nodes_id_must_be_string(client: tg.Client) -> None:
    with pytest.raises(ValueError):
        client.merge_nodes("Person", [{"id": 12345, "name": "wrong"}])


def test_merge_nodes_empty_list_is_noop(client: tg.Client) -> None:
    assert client.merge_nodes("Person", []) == 0
    client.commit()
    result = client.cypher("MATCH (p:Person) RETURN count(*) AS n")
    assert result.first() == {"n": 0}


def test_merge_nodes_then_lookup(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    client.merge_nodes("Person", [{"id": node_id, "name": "Alice", "age": 30}])
    client.commit()
    looked_up = client.lookup_node("Person", node_id)
    assert looked_up is not None
    assert looked_up["properties"] == {"name": "Alice", "age": 30}


def test_merge_edges_basic(client: tg.Client) -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    client.merge_nodes(
        "Person",
        [{"id": ids[0], "name": "A"}, {"id": ids[1], "name": "B"}, {"id": ids[2], "name": "C"}],
    )
    n = client.merge_edges(
        "KNOWS",
        [
            {"src": ids[0], "dst": ids[1], "since": 2020},
            {"src": ids[1], "dst": ids[2], "since": 2021},
        ],
    )
    assert n == 2
    client.commit()
    out = client.out_edges("KNOWS", ids[0])
    assert len(out) == 1
    assert out[0]["dst"] == ids[1]
    assert out[0]["properties"] == {"since": 2020}


def test_merge_edges_missing_src_or_dst_raises(client: tg.Client) -> None:
    with pytest.raises(ValueError):
        client.merge_edges("KNOWS", [{"dst": str(uuid.uuid4())}])
    with pytest.raises(ValueError):
        client.merge_edges("KNOWS", [{"src": str(uuid.uuid4())}])


def test_bulk_then_cypher_scan(client: tg.Client) -> None:
    ids = [str(uuid.uuid4()) for _ in range(10)]
    client.merge_nodes(
        "Person",
        [{"id": ids[i], "name": f"p{i}", "age": 20 + i} for i in range(10)],
    )
    client.commit()
    result = client.cypher(
        "MATCH (p:Person) WHERE p.age >= $min RETURN count(*) AS n",
        params={"min": 25},
    )
    assert result.first() == {"n": 5}


# ── QueryResult.to_arrow ───────────────────────────────────────────────


def test_to_arrow_primitive_columns(people_client: tg.Client) -> None:
    result = people_client.cypher(
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
    )
    table = result.to_arrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert set(table.column_names) == {"name", "age"}
    # Verify dtypes inferred correctly.
    assert table.schema.field("name").type == pa.string()
    assert table.schema.field("age").type == pa.int64()


def test_to_arrow_empty_result(client: tg.Client) -> None:
    result = client.cypher("MATCH (p:NoSuchLabel) RETURN p.name AS name")
    table = result.to_arrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 0
    # Schema is known from the plan even with zero rows — pyarrow
    # infers `null` type when the column is empty.
    assert table.column_names == ["name"]


def test_to_arrow_int_string_mix_columns(people_client: tg.Client) -> None:
    result = people_client.cypher(
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY p.age ASC"
    )
    table = result.to_arrow()
    rows = table.to_pylist()
    assert rows == [
        {"name": "Bob", "age": 25},
        {"name": "Alice", "age": 30},
    ]


def test_to_arrow_datetime_column(client: tg.Client) -> None:
    when = dt.datetime(2026, 5, 18, 12, 34, 56, tzinfo=dt.timezone.utc)
    result = client.cypher("RETURN $when AS x", params={"when": when})
    table = result.to_arrow()
    field = table.schema.field("x")
    # pyarrow infers timestamp[us, UTC] for tz-aware datetimes.
    assert pa.types.is_timestamp(field.type)
    assert table.column("x").to_pylist() == [when]


# ── QueryResult.to_pandas ──────────────────────────────────────────────


def test_to_pandas_basic(people_client: tg.Client) -> None:
    pd = pytest.importorskip("pandas")
    result = people_client.cypher(
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY p.age ASC"
    )
    df = result.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["name", "age"]
    assert df.iloc[0]["name"] == "Bob"
    assert df.iloc[1]["age"] == 30


# ── QueryResult.to_polars ──────────────────────────────────────────────


def test_to_polars_basic(people_client: tg.Client) -> None:
    pl = pytest.importorskip("polars")
    result = people_client.cypher(
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY p.age ASC"
    )
    df = result.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["name", "age"]
    rows = df.to_dicts()
    assert rows[0]["name"] == "Bob"
    assert rows[1]["age"] == 30


# ── Client.scan_label_arrow ────────────────────────────────────────────


def test_scan_label_arrow_basic(client: tg.Client) -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    client.merge_nodes(
        "Person",
        [
            {"id": ids[0], "name": "Alice", "age": 30},
            {"id": ids[1], "name": "Bob"},  # no age
            {"id": ids[2], "name": "Carol", "age": 42, "city": "NYC"},
        ],
    )
    client.commit()
    table = client.scan_label_arrow("Person")
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    # Metadata columns first, then properties in alpha order from BTreeSet.
    # `labels` (full set) sits next to the representative `label`.
    assert table.column_names[:5] == ["id", "label", "labels", "lsn", "schema_version"]
    assert set(table.column_names[5:]) == {"age", "city", "name"}
    # Bob is missing age + city — should land as null.
    rows_by_name = {r["name"]: r for r in table.to_pylist()}
    assert rows_by_name["Bob"]["age"] is None
    assert rows_by_name["Bob"]["city"] is None
    assert rows_by_name["Carol"]["city"] == "NYC"


def test_scan_label_arrow_empty(client: tg.Client) -> None:
    table = client.scan_label_arrow("NoSuchLabel")
    assert isinstance(table, pa.Table)
    assert table.num_rows == 0
    # Metadata columns are always present even for empty scans.
    assert table.column_names == ["id", "label", "labels", "lsn", "schema_version"]
