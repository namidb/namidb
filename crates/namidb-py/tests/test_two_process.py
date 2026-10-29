"""A loader and a read_only reader in two processes on one file:// graph.

This is how the forensic pipeline runs: one process writes, flushes and
lets the client compact and sweep on its own, while another process
queries the same directory. The reader must not fence the writer, must
not be blocked by it, must see new commits after `refresh()`, and the
writer's sweeps must leave the reader's files alone.
"""

from __future__ import annotations

import multiprocessing
import uuid

import namidb as tg

COUNT = "MATCH (p:Person) RETURN count(p) AS n"


def _people(count: int) -> list[dict]:
    return [{"_id": str(uuid.uuid4()), "n": i, "pad": "x" * 100} for i in range(count)]


def _reader(uri: str, conn) -> None:
    try:
        reader = tg.Client(uri, read_only=True, cache_bytes=0)
        it = reader.iter_label("Person")
        ids = [next(it)["id"]]
        conn.send(("open", reader.cypher(COUNT).first()["n"]))
        assert conn.recv() == "go"
        # The writer has loaded more, compacted and swept since. The
        # iterator and plain reads stay on the version this client loaded.
        ids += [n["id"] for n in it]
        stale = reader.cypher(COUNT).first()["n"]
        advanced = reader.refresh()
        fresh = reader.cypher(COUNT, timeout_ms=60_000).first()["n"]
        fresh_iter = sum(1 for _ in reader.iter_label("Person"))
        reader.close()
        conn.send(("done", len(ids), len(set(ids)), stale, advanced, fresh, fresh_iter))
    except BaseException as e:  # noqa: BLE001
        conn.send(("error", repr(e)))


def _recv(conn, timeout: float = 120):
    assert conn.poll(timeout), "the reader process went quiet"
    message = conn.recv()
    assert message[0] != "error", message[1]
    return message


def test_reader_process_survives_writer_compaction_and_sweep(tmp_path) -> None:
    uri = f"file://{tmp_path}?ns=forensic_case"
    # Auto compaction every second flush, and an auto sweep with no grace
    # period: only the reader's retention lease keeps its files.
    writer = tg.Client(uri, auto_sweep_min_age_secs=0, compaction_l0_trigger=2)
    for _ in range(3):
        writer.put_nodes("Person", _people(800))
        writer.commit()
        writer.flush()

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_reader, args=(uri, child))
    proc.start()
    try:
        _, before = _recv(parent)
        assert before == 2400

        # The reader did not take the namespace from the writer.
        for _ in range(4):
            writer.put_nodes("Person", _people(800))
            assert writer.commit() is not None
            writer.flush()
        writer.compact()
        report = writer.sweep(min_age_secs=0)
        assert report["pins_honoured"] >= 1
        assert report["aborted_by_pin"] is False

        parent.send("go")
        _, seen, distinct, stale, advanced, fresh, fresh_iter = _recv(parent)
        assert (seen, distinct) == (2400, 2400)
        assert stale == 2400
        assert advanced is True
        assert fresh == fresh_iter == 5600
    finally:
        proc.join(60)
        if proc.is_alive():
            proc.kill()
    assert proc.exitcode == 0

    # With the reader gone, the files it held are reclaimed.
    assert writer.sweep(min_age_secs=0)["orphans_deleted"] > 0
    assert len(writer.scan_label("Person")) == 5600
    assert not list(tmp_path.rglob("pins/*.json"))
    assert (tmp_path / ".namidb-spool").is_dir()
