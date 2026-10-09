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


def test_production_side_leverage_is_reduced_without_loosening_risk_stops() -> None:
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True)

    assert cfg.v2.costs.side_leverage == pytest.approx(1.30)
    assert cfg.risk.var_stop == pytest.approx(0.03)
    assert cfg.risk.es_stop == pytest.approx(0.04)
    assert cfg.risk.max_gross_exposure == pytest.approx(3.0)


def test_production_config_selects_v2_only_and_research_config_selects_overlay_candidate() -> None:
    production = load_config_from_yaml("configs/production/production.yaml", strict=True)
    research = load_config_from_yaml(
        "configs/research/ml_overlay_forward_shadow_20261009.yaml", strict=True
    )

    assert production.v2.ml_overlay_enabled is False
    assert production.v2.ml_overlay_model_dir == ""
    assert research.v2.ml_overlay_enabled is True
    assert research.v2.ml_overlay_model_dir == "models/ml_order_overlay/production_20260923"


def test_explicit_missing_config_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file"):
        load_config_from_yaml(tmp_path / "missing.yaml", strict=True)
