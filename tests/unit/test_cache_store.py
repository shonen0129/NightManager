"""Tests for the SQLite transactional cache store."""

from __future__ import annotations

import pandas as pd
import pytest

from leadlag.data.cache_store import SqliteCacheStore


@pytest.fixture
def store(tmp_path):
    return SqliteCacheStore(tmp_path / "cache.sqlite")


def test_set_get_roundtrip(store):
    store.set("x", {"a": [1, 2, 3]})
    assert store.get("x") == {"a": [1, 2, 3]}


def test_get_missing_returns_default(store):
    assert store.get("missing") is None
    assert store.get("missing", default=42) == 42


def test_delete_and_keys(store):
    store.set("a", 1)
    store.set("b", 2)
    assert set(store.keys()) == {"a", "b"}
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert store.keys() == ["b"]


def test_clear(store):
    store.set("a", 1)
    store.clear()
    assert store.get("a") is None


def test_pandas_roundtrip(store):
    df = pd.DataFrame({"x": [1, 2, 3]})
    store.set("df", df)
    loaded = store.get("df")
    pd.testing.assert_frame_equal(loaded, df)


def test_meta(store):
    store.set("a", 1)
    meta = store.meta("a")
    assert meta is not None
    assert "updated_at" in meta
