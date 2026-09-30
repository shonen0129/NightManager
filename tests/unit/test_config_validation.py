"""Tests for strict config validation."""

from __future__ import annotations

import pytest

from leadlag.execution.config import (
    UnknownConfigKeyError,
    build_app_config_from_dict,
    load_config_from_yaml,
)


def test_tachibana_default_endpoint_uses_current_api_version(monkeypatch) -> None:
    monkeypatch.delenv("TACHIBANA_API_URL", raising=False)

    cfg = build_app_config_from_dict({})

    assert cfg.tachibana.api_url == "https://kabuka.e-shiten.jp/e_api_v4r10"


def test_tachibana_endpoint_can_be_overridden_for_demo(monkeypatch) -> None:
    monkeypatch.setenv("TACHIBANA_API_URL", "https://demo-kabuka.e-shiten.jp/e_api_v4r10")

    cfg = build_app_config_from_dict({})

    assert cfg.tachibana.api_url == "https://demo-kabuka.e-shiten.jp/e_api_v4r10"


def test_strict_rejects_unknown_top_level_key() -> None:
    with pytest.raises(UnknownConfigKeyError):
        build_app_config_from_dict({"unknown_typo_section": {}}, strict=True)


@pytest.mark.parametrize(
    "section,values",
    [
        ("risk", {"var_stpo": 0.005}),
        ("costs", {"slippage_bps_per_sdie": 999}),
        ("ml_order_overlay", {"enabeld": True}),
        ("blpx", {"rho_typo": 0.5}),
        ("model", {"var_stpo": 0.005}),
    ],
)
def test_strict_rejects_unknown_nested_fields(section: str, values: dict[str, object]) -> None:
    with pytest.raises(UnknownConfigKeyError, match=section):
        build_app_config_from_dict({section: values}, strict=True)


def test_strict_accepts_production_yaml() -> None:
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True)
    assert cfg is not None
    assert cfg.strategy is not None
    assert cfg.risk is not None


def test_explicit_missing_config_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file"):
        load_config_from_yaml(tmp_path / "missing.yaml", strict=True)
