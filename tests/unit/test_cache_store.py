"""Tests for the SQLite transactional cache store."""

from __future__ import annotations

import numpy as np
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


def test_ndarray_roundtrip(store):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    store.set("arr", arr)
    loaded = store.get("arr")
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == arr.shape
    assert loaded.dtype == arr.dtype
    np.testing.assert_allclose(loaded, arr)


def test_bytes_roundtrip(store):
    data = b"hello world"
    store.set("data", data)
    loaded = store.get("data")
    assert isinstance(loaded, bytes)
    assert loaded == data


def test_series_with_datetime_index_roundtrip(store):
    dates = pd.date_range("2020-01-06", periods=5, freq="B")
    s = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01], index=dates, name="ret")
    store.set("s", s)
    loaded = store.get("s")
    assert isinstance(loaded, pd.Series)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.name == s.name
    pd.testing.assert_series_equal(loaded, s, check_names=False)


def test_nested_dict_with_ndarray_roundtrip(store):
    payload = {"mu": np.array([0.1, 0.2, 0.3]), "meta": {"n_j": 17}}
    store.set("payload", payload)
    loaded = store.get("payload")
    assert isinstance(loaded["mu"], np.ndarray)
    np.testing.assert_allclose(loaded["mu"], payload["mu"])
    assert loaded["meta"] == payload["meta"]
