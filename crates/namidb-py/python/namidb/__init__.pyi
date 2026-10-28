"""Type stubs for the `namidb` package.

The runtime objects live in the Rust extension at ``namidb._lib``;
``__init__.py`` re-exports them. These stubs describe the public
surface so IDEs / Pylance / mypy can type-check user code.
"""

from __future__ import annotations

from typing import Any, Coroutine, Optional, TypedDict

__version__: str

# `pyarrow.Table` / `pandas.DataFrame` / `polars.DataFrame` are not
# imported at type-stub time to keep `pandas` / `polars` strictly
# optional. Users that care about precise types can `cast(...)` on
# their side.
_Table = Any
_DataFrame = Any


class _CompactionReport(TypedDict):
    applied: bool
    manifest_version_before: int
    manifest_version_after: int
    l0_before: int
    l0_after: int
    source_ssts_removed: int
    new_ssts_written: int
    bloom_sidecars_written: int


class _SweepReport(TypedDict):
    orphans_found: int
    orphans_deleted: int
    bytes_freed: int
    skipped_too_young: int
    wal_segments_reclaimed: int
    wal_bytes_freed: int
    manifest_snapshots_reclaimed: int
    manifest_bytes_freed: int


class QueryResult:
    """Result of a Cypher query.

    Wraps the Rust-side `Vec<namidb_query::Row>`. Column order
    follows the parsed `RETURN` projection (not the runtime
    `BTreeMap` ordering).
    """

    @property
    def columns(self) -> list[str]:
        """Column names in the order declared by the `RETURN`
        projection. Available even when zero rows match."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def rows(self) -> list[dict[str, Any]]:
        """Materialise every row as a `dict[str, Any]` keyed by
        column name. Values follow the Cypher-to-Python mapping
        described in the package README."""

    def first(self) -> Optional[dict[str, Any]]:
        """First row as `dict[str, Any]`, or `None` if the result
        is empty."""

    def to_arrow(self) -> _Table:
        """`pyarrow.Table` view of the result. `pyarrow >= 14` is a
        hard dependency of `namidb`."""

    def to_pandas(self) -> _DataFrame:
        """`pandas.DataFrame` view. Requires the optional `pandas`
        extra (`pip install 'namidb[pandas]'`)."""

    def to_polars(self) -> _DataFrame:
        """`polars.DataFrame` view. Requires the optional `polars`
        extra (`pip install 'namidb[polars]'`)."""


class Client:
    """A namespace handle: object store + writer session + a tokio
    runtime that drives every async storage call from synchronous
    Python.

    Open with::

        client = namidb.Client("memory://<ns>")
        client = namidb.Client(
            "s3://my-bucket/data?ns=prod"
            "&region=us-west-2"
        )
    """

    def __init__(
        self,
        uri: str,
        cache_bytes: Optional[int] = ...,
        spool_dir: Optional[str] = None,
        auto_sweep_min_age_secs: Optional[int] = 60,
        compaction_l0_trigger: int = 8,
    ) -> None:
        """`spool_dir` holds compaction scratch files; it defaults to
        `NAMIDB_SPOOL_DIR`, then `<root>/.namidb-spool` for `file://`
        URIs. An applied compaction sweeps retired objects older than
        `auto_sweep_min_age_secs` (`None` disables). `flush()` compacts
        once a bucket holds `compaction_l0_trigger` L0 SSTs (`0`
        disables)."""

    # ── Storage CRUD (stages into the current batch) ────────────────

    def upsert_node(
        self,
        label: str,
        id: str,
        properties: dict[str, Any],
    ) -> None:
        """Stage a single node upsert. `id` must be a UUID string."""

    def upsert_node_with_labels(
        self,
        labels: list[str],
        id: str,
        properties: dict[str, Any],
    ) -> None:
        """Stage a multi-label node upsert. `labels` is the full label
        set; the node is keyed by `id` alone, so a later upsert with a
        different set replaces it (last-write-wins). `id` must be a UUID
        string."""

    def tombstone_node(self, label: str, id: str) -> None:
        """Stage a node tombstone."""

    def upsert_edge(
        self,
        edge_type: str,
        src: str,
        dst: str,
        properties: dict[str, Any],
    ) -> None:
        """Stage a single edge upsert."""

    def tombstone_edge(self, edge_type: str, src: str, dst: str) -> None:
        """Stage an edge tombstone."""

    def commit(self) -> Optional[int]:
        """Durably commit the current batch (WAL + manifest CAS).
        Returns the last LSN persisted, or `None` if the batch was
        empty."""

    def flush(self) -> None:
        """Flush the memtable to L0 SSTs, then run a compaction pass if
        a bucket reached `compaction_l0_trigger` L0 SSTs. A failed pass
        only emits a `RuntimeWarning`: the flush itself is durable."""

    def compact(self) -> _CompactionReport:
        """Run one leveled compaction pass. Expensive preparation runs
        without the writer mutex; the returned report describes whether
        the pass applied and which immutable SSTs it replaced. An
        applied pass is followed by a sweep unless
        `auto_sweep_min_age_secs` is `None`."""

    def acompact(self) -> Coroutine[Any, Any, _CompactionReport]:
        """Async sibling of [`Client.compact`], returning the same report."""

    def sweep(self, min_age_secs: int = 60, dry_run: bool = False) -> _SweepReport:
        """Delete objects no manifest version references any more and
        that are at least `min_age_secs` old: SSTs a compaction merged
        away, WAL segments a flush retired, superseded manifests. With
        `dry_run=True` nothing is deleted. The auto sweep after a
        compaction skips that pass's own young leftovers, so call this
        once more at the end of a bulk load."""

    # ── Bulk APIs (batched under one runtime + lock hop) ────────────

    def merge_nodes(
        self,
        label: str,
        rows: list[dict[str, Any]],
    ) -> int:
        """Stage many node upserts that merge into the stored nodes.
        Each `row` is a dict with an `"_id"` (UUID string, the internal
        node id) and arbitrary properties. A row keyed by the legacy
        `"id"` uses it as the node id and also keeps it as an `id`
        property.

        Unmentioned keys are kept, a `None` value removes the key, and
        `label` is added to the node's labels. Each node is read before
        it is written; use [`Client.put_nodes`] for first loads.

        Returns the count of staged rows. Like
        [`Client.upsert_node`], the mutations require a subsequent
        [`Client.commit`] to become durable."""

    def put_nodes(
        self,
        label: str,
        rows: list[dict[str, Any]],
    ) -> int:
        """Stage many node upserts that replace the stored nodes. Rows
        take the same shape as [`Client.merge_nodes`], but each row is
        the node's full property set and `label` its only label. No
        read happens first.

        Returns the count of staged rows. Requires a subsequent
        [`Client.commit`]."""

    def merge_edges(
        self,
        edge_type: str,
        rows: list[dict[str, Any]],
    ) -> int:
        """Stage many edge upserts. Each `row` has `"src"` + `"dst"`
        (UUID strings) and arbitrary properties.

        Returns the count of staged rows. Requires a subsequent
        [`Client.commit`]."""

    # ── Reads ──────────────────────────────────────────────────────

    def lookup_node(
        self,
        label: str,
        id: str,
    ) -> Optional[dict[str, Any]]:
        """Snapshot-isolated lookup. Returns `{"id", "label", "labels",
        "lsn", "schema_version", "properties"}` or `None`. `label` is the
        representative (first) label; `labels` is the full set."""

    def out_edges(
        self,
        edge_type: str,
        src: str,
    ) -> list[dict[str, Any]]:
        """All outgoing edges of `(edge_type, src)` from the
        current snapshot."""

    def in_edges(
        self,
        edge_type: str,
        dst: str,
    ) -> list[dict[str, Any]]:
        """All incoming edges of `(edge_type, dst)`."""

    def scan_label(self, label: str) -> list[dict[str, Any]]:
        """Every node under `label` (memtable + SSTs,
        last-write-wins)."""

    def scan_edge_type(self, edge_type: str) -> list[dict[str, Any]]:
        """Every edge of `edge_type` (memtable + SSTs,
        last-write-wins)."""

    def scan_label_arrow(self, label: str) -> _Table:
        """Like [`Client.scan_label`] but returns a `pyarrow.Table`
        directly. Columns: `id`, `label` (representative), `labels`
        (full set, list<string>), `lsn`, `schema_version`, then the
        union of property keys (missing keys are null). A property named
        like one of those columns is renamed `property.<key>`, or dropped
        when it is an `id` that only repeats the node id."""

    # ── Cypher ─────────────────────────────────────────────────────

    def cypher(
        self,
        query: str,
        params: Optional[dict[str, Any]] = ...,
    ) -> QueryResult:
        """Run a Cypher query synchronously.

        Write plans (CREATE / SET / DELETE / MERGE / REMOVE) are
        durably committed before this returns — `commit_batch()`
        runs inside `execute_write`. Call `flush()` periodically to
        push the memtable into L0 SSTs."""

    def acypher(
        self,
        query: str,
        params: Optional[dict[str, Any]] = ...,
    ) -> Coroutine[Any, Any, QueryResult]:
        """Async sibling of [`Client.cypher`]. Returns a Python
        coroutine. Backed by the `pyo3-async-runtimes` tokio
        bridge; mixing sync + async calls on the same Client is
        safe — both share the same tokio runtime."""

    # ── Diagnostics ────────────────────────────────────────────────

    def cache_stats(self) -> tuple[int, int, int, int]:
        """`(hits, misses, inserts, usage_bytes)` of the SST + bloom
        cache backing this Client."""

    def namespace_prefix(self) -> str:
        """The prefix this Client uses inside the object store
        (e.g. `tenants/acme/`)."""

    def store_repr(self) -> str:
        """Debug representation of the underlying object store
        backend."""
