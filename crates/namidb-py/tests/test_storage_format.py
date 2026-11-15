"""Storage format stamps, opening older and newer namespaces, and the
explicit node-layout migration.

`fixtures/store-2.6.19` was written by the published `namidb==2.6.19`
wheel: 82 `:Doc`/`:Device` nodes with 8-float embeddings in two node SSTs
(a compacted L1 and an L0), `ON_DEVICE` edges, every object no manifest
references removed. Tests work on a copy.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import namidb

FIXTURE = Path(__file__).parent / "fixtures" / "store-2.6.19"
NS = "legacy"

DOCS = "MATCH (d:Doc) RETURN d.name AS name, d.n AS n, d.embedding AS e ORDER BY n"
EDGES = "MATCH (d:Doc)-[r:ON_DEVICE]->(v:Device) RETURN d.n AS n, v.name AS v, r.w AS w ORDER BY n"


@pytest.fixture
def legacy_uri(tmp_path: Path) -> str:
    root = tmp_path / "store"
    shutil.copytree(FIXTURE, root)
    return f"file://{root}?ns={NS}"


def manifest_dir(uri: str) -> Path:
    root = uri[len("file://") :].split("?")[0]
    return Path(root) / NS / "manifest"


def current_manifest(uri: str) -> dict:
    d = manifest_dir(uri)
    pointer = json.loads((d / "current.json").read_text())
    root = d.parent.parent
    return json.loads((root / pointer["manifest_path"]).read_text())


def rows(client: namidb.Client, query: str) -> list[tuple]:
    return [tuple(r.values()) for r in client.cypher(query).rows()]


def expected_docs() -> list[tuple]:
    return [
        (f"doc{i}", i, [float(i * 8 + j) / 4 for j in range(8)]) for i in range(80)
    ]


def test_module_exposes_the_format_api():
    assert namidb.STORAGE_FORMAT_VERSION == 2
    assert issubclass(namidb.StorageFormatError, RuntimeError)


def test_opening_a_2x_namespace_for_writing_needs_opt_in(legacy_uri: str):
    before = sorted(os.listdir(manifest_dir(legacy_uri) / "pointer"))
    with pytest.raises(namidb.StorageFormatError, match="upgrade=True"):
        namidb.Client(legacy_uri)
    # Refused before claiming anything: no new manifest version.
    assert sorted(os.listdir(manifest_dir(legacy_uri) / "pointer")) == before
    assert "storage_format" not in current_manifest(legacy_uri)


def test_a_2x_namespace_reads_read_only_without_changes(legacy_uri: str):
    before = sorted(os.listdir(manifest_dir(legacy_uri) / "pointer"))
    with namidb.Client(legacy_uri, read_only=True) as reader:
        assert rows(reader, DOCS) == expected_docs()
        assert len(rows(reader, EDGES)) == 80
        fmt = reader.storage_format()
    assert fmt["stamp"] is None
    assert fmt["effective_version"] == 1
    assert fmt["legacy_node_ssts"] == 2
    assert fmt["needs_migration"] is True
    assert sorted(os.listdir(manifest_dir(legacy_uri) / "pointer")) == before


def test_upgrade_stamps_without_rewriting(legacy_uri: str):
    ssts_before = current_manifest(legacy_uri)["ssts"]
    client = namidb.Client(legacy_uri, upgrade=True, compaction_l0_trigger=0)
    stamp = current_manifest(legacy_uri)["storage_format"]
    assert stamp["version"] == 2 and stamp["min_reader_version"] == 2
    assert stamp["written_by"] == f"namidb {namidb.__version__}"
    assert current_manifest(legacy_uri)["ssts"] == ssts_before
    assert client.storage_format()["legacy_node_ssts"] == 2
    del client
    # The stamp records the upgrade: later opens need no flag.
    namidb.Client(legacy_uri)


def test_migrate_rewrites_every_2x_sst_and_keeps_the_data(legacy_uri: str):
    client = namidb.Client(legacy_uri, upgrade=True, auto_sweep_min_age_secs=None)
    docs, edges = rows(client, DOCS), rows(client, EDGES)
    dry = client.migrate(dry_run=True)
    assert dry["legacy_node_ssts"] == 2 and dry["migrated"] == 0
    assert client.storage_format()["legacy_node_ssts"] == 2

    seen = []
    report = client.migrate(progress=seen.append)
    assert report["migrated"] == 2 and report["remaining"] == 0
    assert [s["done"] for s in seen] == [1, 2]
    assert sum(s["rows"] for s in seen) == report["legacy_node_rows"] == 82
    assert rows(client, DOCS) == docs == expected_docs()
    assert rows(client, EDGES) == edges
    fmt = client.storage_format()
    assert fmt["legacy_node_ssts"] == 0 and fmt["needs_migration"] is False
    # Same levels as before: nothing was merged.
    levels = sorted(s["level"] for s in current_manifest(legacy_uri)["ssts"] if s["kind"] == "Nodes")
    assert levels == [0, 1]
    assert client.migrate()["legacy_node_ssts"] == 0

    del client
    with namidb.Client(legacy_uri, read_only=True) as reader:
        assert rows(reader, DOCS) == docs


def test_an_interrupted_migration_resumes(legacy_uri: str):
    client = namidb.Client(legacy_uri, upgrade=True, auto_sweep_min_age_secs=None)
    docs = rows(client, DOCS)

    def stop_after_first(step):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        client.migrate(progress=stop_after_first)
    # The first SST was committed before the callback ran; the second was not.
    assert client.storage_format()["legacy_node_ssts"] == 1
    assert rows(client, DOCS) == docs

    del client
    client = namidb.Client(legacy_uri)
    report = client.migrate()
    assert report["legacy_node_ssts"] == 1 and report["remaining"] == 0
    assert rows(client, DOCS) == docs


def test_writes_after_migration_and_compaction_keep_reading(legacy_uri: str):
    client = namidb.Client(legacy_uri, upgrade=True)
    client.migrate()
    client.cypher("CREATE (:Doc {name: 'new', n: 1000, embedding: [1.0, 2.0]})")
    client.commit()
    client.flush()
    client.compact()
    assert rows(client, "MATCH (d:Doc) RETURN count(d) AS c") == [(81,)]
    assert rows(client, "MATCH (d:Doc {n: 5}) RETURN d.embedding AS e") == [
        ([float(5 * 8 + j) / 4 for j in range(8)],)
    ]


def test_migrate_is_refused_on_a_read_only_client(legacy_uri: str):
    with namidb.Client(legacy_uri, read_only=True) as reader:
        with pytest.raises(PermissionError):
            reader.migrate()


def _publish_newer_stamp(uri: str, min_reader: int, min_writer: int) -> None:
    """Commit one more manifest version by hand, as a newer release would."""
    d = manifest_dir(uri)
    manifest = current_manifest(uri)
    manifest["version"] += 1
    manifest["storage_format"] = {
        "version": 9,
        "min_reader_version": min_reader,
        "min_writer_version": min_writer,
        "written_by": "namidb 9.0.0",
    }
    name = "%016x.json" % manifest["version"]
    (d / f"v{name}").write_text(json.dumps(manifest))
    pointer = {
        "version": manifest["version"],
        "epoch": manifest["epoch"],
        "manifest_path": f"{NS}/manifest/v{name}",
    }
    (d / "pointer" / f"p{name}").write_text(json.dumps(pointer))
    (d / "current.json").write_text(json.dumps(pointer))


def test_a_namespace_from_a_newer_format_is_refused(legacy_uri: str):
    namidb.Client(legacy_uri, upgrade=True)
    _publish_newer_stamp(legacy_uri, min_reader=9, min_writer=9)
    before = sorted(os.listdir(manifest_dir(legacy_uri) / "pointer"))
    with pytest.raises(namidb.StorageFormatError, match="namidb 9.0.0"):
        namidb.Client(legacy_uri)
    with pytest.raises(namidb.StorageFormatError, match="upgrade NamiDB"):
        namidb.Client(legacy_uri, read_only=True)
    with pytest.raises(namidb.StorageFormatError):
        namidb.Client(legacy_uri, upgrade=True)
    assert sorted(os.listdir(manifest_dir(legacy_uri) / "pointer")) == before
    assert current_manifest(legacy_uri)["storage_format"]["written_by"] == "namidb 9.0.0"


def test_a_newer_writer_format_still_reads(legacy_uri: str):
    namidb.Client(legacy_uri, upgrade=True)
    _publish_newer_stamp(legacy_uri, min_reader=2, min_writer=9)
    with pytest.raises(namidb.StorageFormatError, match="write"):
        namidb.Client(legacy_uri)
    with namidb.Client(legacy_uri, read_only=True) as reader:
        assert rows(reader, DOCS) == expected_docs()


def test_a_fresh_namespace_is_stamped(tmp_path: Path):
    uri = f"file://{tmp_path}?ns=fresh"
    client = namidb.Client(uri)
    client.cypher("CREATE (:A {x: 1})")
    client.commit()
    fmt = client.storage_format()
    assert fmt["stamp"]["version"] == namidb.STORAGE_FORMAT_VERSION
    assert fmt["effective_version"] == 2
    assert fmt["needs_migration"] is False


# `fixtures/store-2.6.19-wal` holds five `:Doc` nodes that `namidb==2.6.19`
# committed but never flushed: the data lives only in a version-1 WAL segment.
WAL_FIXTURE = Path(__file__).parent / "fixtures" / "store-2.6.19-wal"


def test_a_2x_namespace_with_only_wal_data_needs_opt_in(tmp_path: Path):
    root = tmp_path / "store"
    shutil.copytree(WAL_FIXTURE, root)
    uri = f"file://{root}?ns=walonly"
    with pytest.raises(namidb.StorageFormatError, match="upgrade=True"):
        namidb.Client(uri)
    with namidb.Client(uri, read_only=True) as reader:
        assert rows(reader, "MATCH (d:Doc) RETURN count(d) AS c") == [(5,)]
        fmt = reader.storage_format()
    assert fmt["stamp"] is None
    assert fmt["effective_version"] == 1
    assert fmt["legacy_wal_segments"] == 1
    client = namidb.Client(uri, upgrade=True)
    assert rows(client, "MATCH (d:Doc {n: 3}) RETURN d.embedding AS e") == [([3.0, 0.5],)]
