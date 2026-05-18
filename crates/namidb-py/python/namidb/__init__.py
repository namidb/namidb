"""NamiDB — cloud-native graph database on object storage.

This module is a thin Python wrapper around the Rust extension at
``namidb._lib``. The classes themselves are implemented in Rust via
pyo3; this file exists so the package ships PEP 561 type stubs
alongside the extension.
"""

from __future__ import annotations

from ._lib import Client, QueryResult, __version__

__all__ = ["Client", "QueryResult", "__version__"]
