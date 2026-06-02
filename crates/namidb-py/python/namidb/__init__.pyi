"""Type stubs for the `namidb` package.

The runtime objects live in the Rust extension at ``namidb._lib``;
``__init__.py`` re-exports them. These stubs describe the public
surface so IDEs / Pylance / mypy can type-check user code.
"""

from __future__ import annotations

from typing import Any, Coroutine, Optional

__version__: str

# `pyarrow.Table` / `pandas.DataFrame` / `polars.DataFrame` are not
# imported at type-stub time to keep `pandas` / `polars` strictly
# optional. Users that care about precise types can `cast(...)` on
# their side.
_Table = Any
_DataFrame = Any

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
    ) -> None: ...

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
        """Flush the memtable to L0 SSTs."""

    # ── Bulk APIs (batched under one runtime + lock hop) ────────────

    def merge_nodes(
        self,
        label: str,
        rows: list[dict[str, Any]],
    ) -> int:
        """Stage many node upserts in one call. Each `row` is a
        dict with an `"id"` (UUID string) and arbitrary properties.

        Returns the count of staged rows. Like
        [`Client.upsert_node`], the mutations require a subsequent
        [`Client.commit`] to become durable."""

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
        union of property keys (missing keys -> null)."""

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
