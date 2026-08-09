"""Tests for immutable config helpers."""

from __future__ import annotations

import pytest

from leadlag.config import (
    ConfigMutationError,
    FrozenConfigDict,
    StrategyConfig,
    freeze_config_dict,
    safe_config_copy,
)


def test_safe_config_copy_dict_prevents_cross_contamination():
    base = {"a": {"b": 1}}
    copy1 = safe_config_copy(base)
    copy2 = safe_config_copy(base)

    copy1["a"]["b"] = 2

    assert copy1["a"]["b"] == 2
    assert copy2["a"]["b"] == 1
    assert base["a"]["b"] == 1


def test_safe_config_copy_pydantic():
    cfg = StrategyConfig(k=4, q=0.4)
    copy1 = safe_config_copy(cfg)
    assert copy1 == cfg
    assert copy1 is not cfg


def test_frozen_config_dict_blocks_writes():
    frozen = freeze_config_dict({"a": {"b": 1}})
    assert frozen["a"]["b"] == 1
    with pytest.raises(ConfigMutationError):
        frozen["a"]["b"] = 2
    with pytest.raises(ConfigMutationError):
        del frozen["a"]


def test_frozen_config_dict_nested_isolation():
    frozen = freeze_config_dict({"a": {"b": 1}})
    inner = frozen["a"]
    assert isinstance(inner, FrozenConfigDict)
    assert inner["b"] == 1


def test_frozen_config_dict_to_dict_is_mutable_deep_copy():
    frozen = freeze_config_dict({"a": {"b": 1}})
    mutable = frozen.to_dict()
    mutable["a"]["b"] = 999
    # Original frozen view is unchanged
    assert frozen["a"]["b"] == 1


def test_shallow_copy_still_fails_in_experiment_scenario():
    # Reproduce the exact AGENTS.md footgun.
    base = {"blpx": {"robust_pca": False}}
    cfg1 = base.copy()
    cfg2 = base.copy()
    cfg1["blpx"]["robust_pca"] = True
    # shallow copy shares the nested dict — classic bug.
    assert cfg2["blpx"]["robust_pca"] is True

    # safe_config_copy fixes it (start from a fresh, unmutated base).
    base = {"blpx": {"robust_pca": False}}
    cfg1 = safe_config_copy(base)
    cfg2 = safe_config_copy(base)
    cfg1["blpx"]["robust_pca"] = True
    assert cfg2["blpx"]["robust_pca"] is False
