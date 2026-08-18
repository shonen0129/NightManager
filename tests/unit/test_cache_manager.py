"""Unit tests for the unified CacheManager / _NamespaceCache."""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.config.schemas import BLPXConfig
from leadlag.utils.cache_manager import CacheManager


def _cm(cfg: BLPXConfig, maxsize: int = 128) -> CacheManager:
    return CacheManager(CacheManager.config_hash_from_pydantic(cfg), maxsize=maxsize)


@pytest.mark.unit
class TestNamespaceCache:
    """Test LRU semantics and config isolation of the namespace cache."""

    def test_namespace_is_isolated(self) -> None:
        manager = _cm(BLPXConfig())
        ns1 = manager.namespace("foo")
        ns2 = manager.namespace("bar")

        ns1["k"] = 1
        ns2["k"] = 2

        assert ns1["k"] == 1
        assert ns2["k"] == 2

    def test_config_hash_isolation(self) -> None:
        cfg_a = BLPXConfig(param_set="a")
        cfg_b = BLPXConfig(param_set="b")

        manager_a = _cm(cfg_a)
        manager_b = _cm(cfg_b)

        ns_a = manager_a.namespace("test")
        ns_b = manager_b.namespace("test")

        ns_a["k"] = "a"
        ns_b["k"] = "b"

        # Same namespace name, different config hash -> isolated.
        assert ns_a["k"] == "a"
        assert ns_b["k"] == "b"

    def test_lru_eviction(self) -> None:
        manager = _cm(BLPXConfig(), maxsize=3)
        ns = manager.namespace("lru")

        ns["a"] = 1
        ns["b"] = 2
        ns["c"] = 3

        # Access 'a' to move it to the end.
        _ = ns["a"]
        ns["d"] = 4

        # 'b' should have been evicted.
        assert "b" not in ns
        assert ns["a"] == 1
        assert ns["c"] == 3
        assert ns["d"] == 4

    def test_clear_removes_all_entries(self) -> None:
        manager = _cm(BLPXConfig())
        ns1 = manager.namespace("one")
        ns2 = manager.namespace("two")

        ns1["k"] = 1
        ns2["k"] = 2

        manager.clear()

        assert "k" not in ns1
        assert "k" not in ns2

    def test_reinsert_moves_to_end(self) -> None:
        manager = _cm(BLPXConfig(), maxsize=2)
        ns = manager.namespace("reinsert")

        ns["a"] = 1
        ns["b"] = 2
        ns["a"] = 3  # reinsert
        ns["c"] = 4

        assert "b" not in ns
        assert ns["a"] == 3
        assert ns["c"] == 4

    def test_values_are_stored_correctly(self) -> None:
        manager = _cm(BLPXConfig())
        ns = manager.namespace("arrays")

        arr = np.array([1.0, 2.0, 3.0])
        ns["arr"] = arr

        assert np.array_equal(ns["arr"], arr)
