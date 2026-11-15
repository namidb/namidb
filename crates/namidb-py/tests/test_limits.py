"""Per-query memory limits and cancellation for the embedded client.

A query that needs more memory than its limit raises `QueryMemoryError`
instead of growing until the machine runs out; a running query can be
stopped from another thread (`CancelToken`), by cancelling its asyncio task,
or with Ctrl-C, and it stops promptly instead of running to completion.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import uuid

import pytest

import namidb as tg

MIB = 1024 * 1024
# Per-query budget the heavy shapes run under (the graph is ~3 MiB).
LIMIT = MIB

# CPU-bound for minutes while holding little memory: only a cancel stops it.
ENDLESS = (
    "UNWIND range(1, 1000000) AS i "
    "RETURN sum(reduce(s = 0, x IN range(1, 5000) | s + x)) AS t"
)


def fresh_client(**kwargs) -> tg.Client:
    return tg.Client(f"memory://limits-{uuid.uuid4().hex[:8]}", **kwargs)


@pytest.fixture(scope="module")
def graph() -> tg.Client:
    """2,000 :Item nodes with a 1 KiB payload, each linked to the next
    three (a dense chain for variable-length paths) and to one of four
    :Hub nodes."""
    client = fresh_client()
    hubs = [str(uuid.uuid4()) for _ in range(4)]
    items = [str(uuid.uuid4()) for _ in range(2000)]
    client.put_nodes("Hub", [{"_id": h, "h": i} for i, h in enumerate(hubs)])
    client.put_nodes(
        "Item", [{"_id": n, "k": i, "blob": "abcdefgh" * 128} for i, n in enumerate(items)]
    )
    client.merge_edges(
        "LINK",
        [
            {"src": n, "dst": items[(i + step) % len(items)]}
            for i, n in enumerate(items)
            for step in (1, 2, 3)
        ],
    )
    client.merge_edges(
        "ON_HUB", [{"src": n, "dst": hubs[i % 4]} for i, n in enumerate(items)]
    )
    client.commit()
    client.flush()
    return client


# ── memory limits ──────────────────────────────────────────────────────

HEAVY_SHAPES = {
    "cartesian": "MATCH (a:Item), (b:Item) RETURN a.k, b.k",
    "expand": "MATCH (h:Hub)<-[:ON_HUB]-(i:Item) RETURN h, i",
    "var_length": "MATCH (a:Item)-[:LINK*]->(b:Item) RETURN a.k, b.k",
    "collect": "MATCH (a:Item), (b:Item) RETURN collect(a.blob + b.blob) AS all",
    "order_by": "MATCH (a:Item), (b:Item) RETURN a.blob + b.blob AS s ORDER BY s DESC",
    "distinct": "MATCH (a:Item), (b:Item) RETURN DISTINCT a.blob + b.blob AS s",
    "aggregation": "MATCH (a:Item), (b:Item) RETURN a.k * 10000 + b.k AS g, count(*) AS c",
    "unwind": "UNWIND range(1, 100000000) AS i RETURN i",
    "list": "RETURN size(range(1, 100000000)) AS n",
    "scan": "MATCH (n) RETURN n",
}


@pytest.mark.parametrize("name", sorted(HEAVY_SHAPES))
def test_every_heavy_operator_stops_at_the_memory_limit(graph: tg.Client, name: str) -> None:
    started = time.monotonic()
    with pytest.raises(tg.QueryMemoryError) as exc_info:
        graph.cypher(HEAVY_SHAPES[name], max_memory_bytes=LIMIT)
    elapsed = time.monotonic() - started
    error = exc_info.value
    assert isinstance(error, MemoryError)
    assert error.scope == "query"
    assert error.limit_bytes == LIMIT
    assert error.required_bytes > error.limit_bytes
    assert "max_memory_bytes" in str(error)
    assert elapsed < 20, f"{name} took {elapsed:.1f}s to hit the limit"


def test_queries_that_fit_still_answer(graph: tg.Client) -> None:
    rows = graph.cypher(
        "MATCH (n:Item) WHERE n.k < 10 RETURN n.k AS k", max_memory_bytes=LIMIT
    ).rows()
    assert sorted(row["k"] for row in rows) == list(range(10))
    assert graph.cypher(
        "MATCH (n:Item) RETURN count(n) AS c", max_memory_bytes=LIMIT
    ).first() == {"c": 2000}


def test_client_default_and_per_call_override() -> None:
    client = fresh_client(query_memory_max_bytes=4 * MIB, memory_max_bytes=0)
    assert client.query_memory_max_bytes == 4 * MIB
    assert client.memory_max_bytes == 0
    with pytest.raises(tg.QueryMemoryError):
        client.cypher("UNWIND range(1, 10000000) AS i RETURN collect(i) AS xs")
    # `0` lifts the per-query budget for one call.
    result = client.cypher(
        "UNWIND range(1, 300000) AS i RETURN size(collect(i)) AS n", max_memory_bytes=0
    )
    assert result.first() == {"n": 300000}


def test_limits_default_to_a_share_of_the_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NAMIDB_QUERY_MEMORY_MAX_BYTES", raising=False)
    monkeypatch.delenv("NAMIDB_MEMORY_MAX_BYTES", raising=False)
    client = fresh_client()
    if sys.platform.startswith("linux"):
        # 25% / 80% of the memory available to the process.
        assert 0 < client.query_memory_max_bytes < client.memory_max_bytes


def test_limits_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAMIDB_QUERY_MEMORY_MAX_BYTES", str(8 * MIB))
    monkeypatch.setenv("NAMIDB_MEMORY_MAX_BYTES", "0")
    client = fresh_client()
    assert client.query_memory_max_bytes == 8 * MIB
    assert client.memory_max_bytes == 0
    monkeypatch.setenv("NAMIDB_QUERY_MEMORY_MAX_BYTES", "lots")
    with pytest.raises(ValueError, match="NAMIDB_QUERY_MEMORY_MAX_BYTES"):
        fresh_client()


def test_process_memory_stays_bounded(tmp_path) -> None:
    """The real resident set, not only the estimate, stays near the limit."""
    script = tmp_path / "bounded.py"
    script.write_text(
        textwrap.dedent(
            """
            import resource, sys, uuid
            import namidb
            c = namidb.Client(f"memory://rss-{uuid.uuid4().hex[:8]}",
                              query_memory_max_bytes=128 * 1024 * 1024)
            base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            for q in [
                "UNWIND range(1, 300000000) AS i RETURN collect(i) AS xs",
                "UNWIND range(1, 300000000) AS i RETURN i",
                "UNWIND range(1, 20000) AS a UNWIND range(1, 20000) AS b RETURN a, b",
                "UNWIND range(1, 3000000) AS i RETURN i ORDER BY i DESC",
            ]:
                try:
                    c.cypher(q)
                    print("completed", q)
                except namidb.QueryMemoryError:
                    pass
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print((peak - base) // 1024)
            """
        )
    )
    out = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert "completed" not in out.stdout, out.stdout
    growth_mib = int(out.stdout.strip().splitlines()[-1])
    # Before the limit existed the first query alone grew past 6 GiB.
    assert growth_mib < 512, f"resident set grew by {growth_mib} MiB"


# ── cancellation ───────────────────────────────────────────────────────


def cpu_seconds_over(window: float) -> float:
    """CPU time all threads of this process use over `window` seconds."""
    before = time.process_time()
    time.sleep(window)
    return time.process_time() - before


def test_cancel_token_stops_a_running_query_from_another_thread() -> None:
    client = fresh_client()
    token = tg.CancelToken()
    assert not token.cancelled
    fired = []

    def cancel_soon() -> None:
        time.sleep(0.5)
        fired.append(time.monotonic())
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    with pytest.raises(tg.QueryCancelledError):
        client.cypher(ENDLESS, cancel=token)
    latency = time.monotonic() - fired[0]
    assert token.cancelled
    assert latency < 1.0, f"cancel took {latency:.2f}s"
    # The work stopped, not only the call.
    assert cpu_seconds_over(0.5) < 0.25
    assert client.cypher("RETURN 1 AS x").first() == {"x": 1}


def test_a_cancelled_token_stops_the_query_before_it_starts() -> None:
    client = fresh_client()
    token = tg.CancelToken()
    token.cancel()
    started = time.monotonic()
    with pytest.raises(tg.QueryCancelledError):
        client.cypher(ENDLESS, cancel=token)
    assert time.monotonic() - started < 1.0
    # A cancel is not a timeout.
    assert not issubclass(tg.QueryCancelledError, TimeoutError)


async def test_cancelling_the_asyncio_task_stops_the_query() -> None:
    client = fresh_client()
    task = asyncio.ensure_future(client.acypher(ENDLESS))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.2)
    assert await asyncio.to_thread(cpu_seconds_over, 0.5) < 0.25
    assert (await client.acypher("RETURN 1 AS x")).first() == {"x": 1}


async def test_asyncio_wait_for_timeout_stops_the_query() -> None:
    client = fresh_client()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(client.acypher(ENDLESS), timeout=0.5)
    await asyncio.sleep(0.2)
    assert await asyncio.to_thread(cpu_seconds_over, 0.5) < 0.25


async def test_cancel_token_works_with_acypher() -> None:
    client = fresh_client()
    token = tg.CancelToken()
    loop = asyncio.get_running_loop()
    loop.call_later(0.3, token.cancel)
    with pytest.raises(tg.QueryCancelledError):
        await client.acypher(ENDLESS, cancel=token)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_ctrl_c_stops_the_running_query(tmp_path) -> None:
    script = tmp_path / "ctrl_c.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys, time, uuid
            import namidb
            c = namidb.Client(f"memory://sigint-{{uuid.uuid4().hex[:8]}}")
            print("ready", flush=True)
            try:
                c.cypher({ENDLESS!r})
                print("completed", flush=True)
            except KeyboardInterrupt:
                print("interrupted", flush=True)
                before = time.process_time()
                time.sleep(0.5)
                print(f"cpu {{time.process_time() - before:.3f}}", flush=True)
            """
        )
    )
    child = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        time.sleep(1.0)
        sent = time.monotonic()
        child.send_signal(signal.SIGINT)
        assert child.stdout.readline().strip() == "interrupted"
        latency = time.monotonic() - sent
        cpu = float(child.stdout.readline().split()[1])
        child.wait(timeout=30)
    finally:
        child.kill()
    assert latency < 1.0, f"Ctrl-C took {latency:.2f}s"
    assert cpu < 0.25, f"the query kept running after Ctrl-C ({cpu:.2f}s CPU)"


def test_a_long_query_does_not_hold_up_other_threads() -> None:
    client = fresh_client()
    token = tg.CancelToken()
    worker = threading.Thread(
        target=lambda: pytest.raises(tg.QueryCancelledError, client.cypher, ENDLESS, cancel=token),
        daemon=True,
    )
    worker.start()
    time.sleep(0.3)
    started = time.monotonic()
    assert client.cypher("RETURN 1 AS x").first() == {"x": 1}
    assert time.monotonic() - started < 1.0
    token.cancel()
    worker.join(timeout=5)
    assert not worker.is_alive()
