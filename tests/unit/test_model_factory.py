"""Tests for the shared V2 model-construction boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.config.paths import project_root
from leadlag.config.schemas import AppConfig
from leadlag.runner.model_factory import (
    build_blpx_model,
    build_v2_model_bundle,
    model_config_fingerprint,
    resolve_overlay_settings,
)


def _config(*, overlay_enabled: bool = False, overlay_dir: str = "") -> AppConfig:
    base = AppConfig()
    run_cfg = base.v2.model_copy(
        update={
            "ml_overlay_enabled": overlay_enabled,
            "ml_overlay_model_dir": overlay_dir,
        }
    )
    return base.model_copy(update={"v2": run_cfg})


def test_bundle_uses_one_run_config_and_shared_blpx_instance() -> None:
    app_config = _config()

    bundle = build_v2_model_bundle(app_config)

    assert bundle.run_config is app_config.v2
    assert bundle.decision_model.run_config is app_config.v2
    assert bundle.decision_model._blpx_model is bundle.blpx_model
    assert bundle.overlay_model is None
    assert bundle.overlay_enabled is False


def test_relative_overlay_path_is_resolved_from_project_root() -> None:
    app_config = _config(overlay_enabled=False, overlay_dir="models/example")

    enabled, path = resolve_overlay_settings(app_config)

    assert enabled is False
    assert path == project_root() / Path("models/example")


def test_explicit_overlay_override_is_loaded_even_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app_config = _config(overlay_enabled=False)
    sentinel = object()
    loaded: list[Path] = []

    def fake_load(path: Path) -> object:
        loaded.append(path)
        return sentinel

    monkeypatch.setattr("leadlag.runner.model_factory.load_overlay_model", fake_load)

    bundle = build_v2_model_bundle(app_config, overlay_model_dir=tmp_path / "overlay")

    assert loaded == [tmp_path / "overlay"]
    assert bundle.overlay_model is sentinel
    assert bundle.overlay_enabled is False


def test_factory_rejects_raw_config_dict() -> None:
    with pytest.raises(TypeError, match="validated AppConfig"):
        build_v2_model_bundle({})  # type: ignore[arg-type]


def test_blpx_factory_builds_without_loading_configured_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    app_config = _config(overlay_enabled=True, overlay_dir="models/missing")
    monkeypatch.setattr(
        "leadlag.runner.model_factory.load_overlay_model",
        lambda _path: pytest.fail("BLPX-only construction must not load an overlay"),
    )

    model = build_blpx_model(app_config)

    assert model.cfg is app_config.v2.blpx


def test_model_config_fingerprint_is_stable_and_excludes_broker_config() -> None:
    app_config = _config()
    changed = app_config.model_copy(update={"broker_provider": "tachibana"})

    assert model_config_fingerprint(app_config) == model_config_fingerprint(changed)
    assert len(model_config_fingerprint(app_config)) == 64
