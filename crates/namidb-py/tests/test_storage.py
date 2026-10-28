"""S19.D — direct coverage of the storage CRUD API.

These exercise `upsert_node` / `upsert_edge` / `tombstone_*` /
`commit` / `flush` / `lookup_node` / `scan_*` / `cache_stats` /
`namespace_prefix` / `store_repr` independently of the Cypher
surface (which covers the same primitives indirectly).
"""

from __future__ import annotations

import uuid

import pytest

import namidb as tg


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ── upsert_node / lookup_node ──────────────────────────────────────────


def test_upsert_then_lookup_roundtrip(client: tg.Client) -> None:
    node_id = _new_uuid()
    client.upsert_node("Person", node_id, {"name": "Alice", "age": 30})
    lsn = client.commit()
    assert lsn is not None and lsn >= 1
    view = client.lookup_node("Person", node_id)
    assert view is not None
    assert view["id"] == node_id
    assert view["label"] == "Person"
    assert view["lsn"] == lsn
    assert view["properties"] == {"name": "Alice", "age": 30}


def test_lookup_returns_none_for_missing(client: tg.Client) -> None:
    assert client.lookup_node("Person", _new_uuid()) is None


def test_upsert_invalid_uuid_raises(client: tg.Client) -> None:
    with pytest.raises(ValueError):
        client.upsert_node("Person", "not-a-uuid", {})


def test_upsert_uses_last_write_wins(client: tg.Client) -> None:
    node_id = _new_uuid()
    client.upsert_node("Person", node_id, {"age": 30})
    client.commit()
    client.upsert_node("Person", node_id, {"age": 31})
    client.commit()
    view = client.lookup_node("Person", node_id)
    assert view is not None
    assert view["properties"]["age"] == 31


# ── multi-label nodes ──────────────────────────────────────────────────


def test_upsert_node_with_labels_roundtrip(client: tg.Client) -> None:
    node_id = _new_uuid()
    client.upsert_node_with_labels(["Person", "Admin"], node_id, {"name": "Ada"})
    client.commit()
    # lookup_node returns the representative `label` plus the full `labels` set.
    view = client.lookup_node("Person", node_id)
    assert view is not None
    assert view["id"] == node_id
    assert set(view["labels"]) == {"Person", "Admin"}
    assert view["label"] in {"Person", "Admin"}
    # The node surfaces under each of its labels individually.
    assert client.lookup_node("Admin", node_id) is not None
    assert {n["id"] for n in client.scan_label("Person")} == {node_id}
    assert {n["id"] for n in client.scan_label("Admin")} == {node_id}


def test_scan_label_arrow_carries_labels_column(client: tg.Client) -> None:
    node_id = _new_uuid()
    client.upsert_node_with_labels(["Person", "Admin"], node_id, {"name": "Ada"})
    client.commit()
    table = client.scan_label_arrow("Person")
    assert "labels" in table.column_names  # full set
    assert "label" in table.column_names  # representative, back-compat
    labels_col = table.column("labels").to_pylist()
    assert len(labels_col) == 1
    assert set(labels_col[0]) == {"Person", "Admin"}


# ── tombstone_node ─────────────────────────────────────────────────────


def test_tombstone_node_hides_from_lookup(client: tg.Client) -> None:
    node_id = _new_uuid()
    client.upsert_node("Person", node_id, {"name": "Alice"})
    client.commit()
    assert client.lookup_node("Person", node_id) is not None
    client.tombstone_node("Person", node_id)
    client.commit()
    assert client.lookup_node("Person", node_id) is None


# ── upsert_edge / out_edges / in_edges ─────────────────────────────────


def test_upsert_edge_roundtrip(client: tg.Client) -> None:
    a, b = _new_uuid(), _new_uuid()
    client.upsert_node("Person", a, {"name": "A"})
    client.upsert_node("Person", b, {"name": "B"})
    client.upsert_edge("KNOWS", a, b, {"since": 2020})
    client.commit()
    out = client.out_edges("KNOWS", a)
    assert len(out) == 1
    assert out[0]["src"] == a
    assert out[0]["dst"] == b
    assert out[0]["properties"] == {"since": 2020}
    inn = client.in_edges("KNOWS", b)
    assert len(inn) == 1
    assert inn[0]["src"] == a


def test_tombstone_edge_hides_from_scan(client: tg.Client) -> None:
    a, b = _new_uuid(), _new_uuid()
    client.upsert_edge("KNOWS", a, b, {})
    client.commit()
    assert len(client.out_edges("KNOWS", a)) == 1
    client.tombstone_edge("KNOWS", a, b)
    client.commit()
    assert client.out_edges("KNOWS", a) == []


# ── commit semantics ───────────────────────────────────────────────────


def test_commit_empty_batch_returns_none(client: tg.Client) -> None:
    assert client.commit() is None
    # And again — still empty, still None.
    assert client.commit() is None


def test_commit_returns_increasing_lsn(client: tg.Client) -> None:
    client.upsert_node("Person", _new_uuid(), {})
    a = client.commit()
    client.upsert_node("Person", _new_uuid(), {})
    b = client.commit()
    assert a is not None and b is not None
    assert b > a


# ── flush + cache_stats ────────────────────────────────────────────────


def test_flush_then_scan(client: tg.Client) -> None:
    for _ in range(5):
        client.upsert_node("Person", _new_uuid(), {})
    client.commit()
    client.flush()
    assert len(client.scan_label("Person")) == 5


def test_cache_stats_shape(client: tg.Client) -> None:
    hits, misses, inserts, usage = client.cache_stats()
    assert all(isinstance(x, int) for x in (hits, misses, inserts, usage))
    assert usage >= 0


# ── scan_label / scan_edge_type ────────────────────────────────────────


def test_scan_label_returns_every_node(client: tg.Client) -> None:
    ids = [_new_uuid() for _ in range(7)]
    for nid in ids:
        client.upsert_node("Person", nid, {})
    client.commit()
    seen = {n["id"] for n in client.scan_label("Person")}
    assert seen == set(ids)


def test_scan_edge_type_returns_every_edge(client: tg.Client) -> None:
    pairs = [(_new_uuid(), _new_uuid()) for _ in range(3)]
    for a, b in pairs:
        client.upsert_edge("KNOWS", a, b, {})
    client.commit()
    seen = {(e["src"], e["dst"]) for e in client.scan_edge_type("KNOWS")}
    assert seen == set(pairs)


def test_scan_limit_caps_the_result(client: tg.Client) -> None:
    for _ in range(10):
        client.upsert_node("Person", _new_uuid(), {})
        client.upsert_edge("KNOWS", _new_uuid(), _new_uuid(), {})
    client.commit()
    assert len(client.scan_label("Person", limit=3)) == 3
    assert len(client.scan_label("Person", limit=100)) == 10
    assert client.scan_label_arrow("Person", limit=4).num_rows == 4
    assert len(client.scan_edge_type("KNOWS", limit=3)) == 3
    assert client.scan_label("Person", limit=0) == []


def test_iter_label_and_edge_type_yield_the_scan(client: tg.Client) -> None:
    ids = {_new_uuid() for _ in range(5)}
    for nid in ids:
        client.upsert_node("Person", nid, {})
    client.upsert_edge("KNOWS", *sorted(ids)[:2], {})
    client.commit()

    it = client.iter_label("Person")
    assert iter(it) is it
    first = next(it)
    assert first["label"] == "Person"
    assert {first["id"]} | {n["id"] for n in it} == ids
    with pytest.raises(StopIteration):
        next(it)

    assert len(list(client.iter_label("Person", limit=2))) == 2
    edges = list(client.iter_edge_type("KNOWS"))
    assert [(e["src"], e["dst"]) for e in edges] == [tuple(sorted(ids)[:2])]


# ── read_only clients ──────────────────────────────────────────────────


def _file_uri(tmp_path) -> str:
    return f"file://{tmp_path}?ns=ns{uuid.uuid4().hex[:8]}"


def test_read_only_client_does_not_fence_the_writer(tmp_path) -> None:
    uri = _file_uri(tmp_path)
    writer = tg.Client(uri)
    alice = _new_uuid()
    writer.upsert_node("Person", alice, {"name": "Alice"})
    writer.commit()

    # The commit is only in the WAL; the reader replays it.
    reader = tg.Client(uri, read_only=True)
    assert reader.lookup_node("Person", alice)["properties"] == {"name": "Alice"}

    # Opening the reader did not take the namespace from the writer.
    bob = _new_uuid()
    writer.upsert_node("Person", bob, {"name": "Bob"})
    assert writer.commit() is not None
    writer.flush()


def test_read_only_client_rejects_writes(tmp_path) -> None:
    uri = _file_uri(tmp_path)
    tg.Client(uri).commit()
    reader = tg.Client(uri, read_only=True)
    node_id = _new_uuid()
    for call in (
        lambda: reader.upsert_node("Person", node_id, {}),
        lambda: reader.upsert_edge("KNOWS", node_id, node_id, {}),
        lambda: reader.tombstone_node("Person", node_id),
        lambda: reader.merge_nodes("Person", [{"id": node_id}]),
        reader.commit,
        reader.flush,
        reader.compact,
        lambda: reader.cypher("CREATE (:Person {name: 'x'})"),
        lambda: reader.cypher("CREATE INDEX FOR (p:Person) ON (p.name)"),
    ):
        with pytest.raises(PermissionError):
            call()
    # Reads still work.
    assert reader.cypher("MATCH (p:Person) RETURN count(p) AS n").first() == {"n": 0}


def test_read_only_refresh_sees_new_commits(tmp_path) -> None:
    uri = _file_uri(tmp_path)
    writer = tg.Client(uri)
    writer.commit()
    reader = tg.Client(uri, read_only=True)
    assert reader.refresh() is False

    writer.cypher("CREATE (:Person {name: 'Alice'})")
    query = "MATCH (p:Person) RETURN p.name AS name"
    assert len(reader.cypher(query)) == 0
    assert reader.refresh() is True
    assert reader.cypher(query).first() == {"name": "Alice"}
    assert writer.refresh() is False


def test_read_only_requires_an_existing_namespace(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        tg.Client(_file_uri(tmp_path), read_only=True)


# ── property type round-trip (storage Value coverage) ──────────────────


def test_property_types_roundtrip(client: tg.Client) -> None:
    node_id = _new_uuid()
    client.upsert_node(
        "Doc",
        node_id,
        {
            "name": "report",
            "page_count": 42,
            "ratio": 0.75,
            "published": True,
            "blob": b"\x00\x01\x02",
            "embedding": [0.1, 0.2, 0.3],  # float32 vector
            "missing": None,
        },
    )
    client.commit()
    view = client.lookup_node("Doc", node_id)
    assert view is not None
    props = view["properties"]
    assert props["name"] == "report"
    assert props["page_count"] == 42
    assert props["ratio"] == pytest.approx(0.75)
    assert props["published"] is True
    assert props["blob"] == b"\x00\x01\x02"
    assert props["embedding"] == pytest.approx([0.1, 0.2, 0.3])
    assert props["missing"] is None


# ── diagnostics ────────────────────────────────────────────────────────


def test_namespace_prefix_is_string(client: tg.Client) -> None:
    prefix = client.namespace_prefix()
    assert isinstance(prefix, str)
    assert len(prefix) > 0


def test_store_repr_is_string(client: tg.Client) -> None:
    rep = client.store_repr()
    assert isinstance(rep, str)


# ── version surface ────────────────────────────────────────────────────


def test_version_attr_exists() -> None:
    assert isinstance(tg.__version__, str)
    assert len(tg.__version__) > 0
