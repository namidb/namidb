"""Shared pytest fixtures for the namidb Python bindings."""

from __future__ import annotations

import uuid

import pytest

import namidb as tg


@pytest.fixture
def client() -> tg.Client:
    """A fresh in-memory namespace per test."""
    ns = f"test-{uuid.uuid4().hex[:8]}"
    return tg.Client(f"memory://{ns}")


@pytest.fixture
def people_client(client: tg.Client) -> tg.Client:
    """Client with two Person nodes (Alice 30, Bob 25) and a KNOWS edge."""
    client.cypher(
        "CREATE (a:Person {name: 'Alice', age: 30}), "
        "       (b:Person {name: 'Bob',   age: 25}), "
        "       (a)-[:KNOWS {since: 2020}]->(b)"
    )
    client.commit()
    return client
