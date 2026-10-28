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
    assert "'_id'" in str(exc_info.value)


def test_merge_nodes_preserves_unmentioned_props(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    client.merge_nodes("Person", [{"_id": node_id, "a": 1, "b": 2}])
    client.commit()
    client.merge_nodes("Person", [{"_id": node_id, "b": 3}])
    client.commit()
    assert client.lookup_node("Person", node_id)["properties"] == {"a": 1, "b": 3}


def test_merge_nodes_none_removes_key(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    client.merge_nodes("Person", [{"_id": node_id, "a": 1, "b": 2}])
    client.merge_nodes("Person", [{"_id": node_id, "b": None}])
    client.commit()
    assert client.lookup_node("Person", node_id)["properties"] == {"a": 1}


def test_merge_nodes_keeps_extra_labels(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    client.upsert_node_with_labels(["A", "B"], node_id, {"x": 1})
    client.commit()
    client.merge_nodes("A", [{"_id": node_id, "y": 2}])
    client.commit()
    node = client.lookup_node("A", node_id)
    assert node["labels"] == ["A", "B"]
    assert node["properties"] == {"x": 1, "y": 2}


def test_merge_nodes_same_id_twice_in_batch(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    n = client.merge_nodes(
        "Person",
        [{"_id": node_id, "a": 1}, {"_id": node_id, "b": 2}, {"_id": node_id, "a": 3}],
    )
    assert n == 3
    client.commit()
    assert client.lookup_node("Person", node_id)["properties"] == {"a": 3, "b": 2}


def test_put_nodes_replaces(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    client.upsert_node_with_labels(["A", "B"], node_id, {"a": 1, "b": 2})
    client.commit()
    assert client.put_nodes("A", [{"_id": node_id, "b": 3}]) == 1
    client.commit()
    node = client.lookup_node("A", node_id)
    assert node["labels"] == ["A"]
    assert node["properties"] == {"b": 3}


def test_merge_nodes_reupsert_same_unique_value_ok(client: tg.Client) -> None:
    client.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS UNIQUE")
    node_id = str(uuid.uuid4())
    client.merge_nodes("Person", [{"_id": node_id, "email": "a@x.io"}])
    client.commit()
    # The node's own value is not a conflict, for merge, put or a repeat in
    # one batch.
    client.merge_nodes("Person", [{"_id": node_id, "email": "a@x.io", "n": 1}])
    client.put_nodes(
        "Person",
        [{"_id": node_id, "email": "a@x.io"}, {"_id": node_id, "email": "a@x.io"}],
    )
    client.commit()
    with pytest.raises(ValueError, match="unique"):
        client.merge_nodes("Person", [{"_id": str(uuid.uuid4()), "email": "a@x.io"}])


def test_merge_nodes_id_visible_as_property(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    client.merge_nodes("Person", [{"id": node_id, "name": "Alice"}])
    client.commit()
    row = client.cypher(
        "MATCH (p:Person) WHERE p.id = $id RETURN p.id AS id, p._id AS iid",
        params={"id": node_id},
    ).first()
    assert row is not None
    assert row["id"] == node_id
    assert str(row["iid"]) == node_id


def test_merge_nodes_underscore_id_keeps_business_id(client: tg.Client) -> None:
    node_id = str(uuid.uuid4())
    client.merge_nodes("Person", [{"_id": node_id, "id": "case-42"}])
    client.commit()
    assert client.lookup_node("Person", node_id)["properties"] == {"id": "case-42"}
    row = client.cypher("MATCH (p:Person) RETURN p.id AS id").first()
    assert row == {"id": "case-42"}


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
    # A legacy `id` key is the node id and also stays a property.
    assert looked_up["properties"] == {"id": node_id, "name": "Alice", "age": 30}


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
            {"_id": ids[0], "name": "Alice", "age": 30},
            {"_id": ids[1], "name": "Bob"},  # no age
            {"_id": ids[2], "name": "Carol", "age": 42, "city": "NYC"},
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
