"""Tests for strict config validation."""

from __future__ import annotations

import pytest

from leadlag.execution.config import (
    UnknownConfigKeyError,
    build_app_config_from_dict,
    load_config_from_yaml,
)


def test_strict_rejects_unknown_top_level_key() -> None:
    with pytest.raises(UnknownConfigKeyError):
        build_app_config_from_dict({"unknown_typo_section": {}}, strict=True)


def test_strict_accepts_production_yaml() -> None:
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True)
    assert cfg is not None
    assert cfg.strategy is not None
    assert cfg.risk is not None
