"""Canonical construction of the V2 model components.

This module contains only dependency wiring.  It is shared by the live
``ProductionRunner`` and the historical backtest so that both paths construct
the same BLPX model, decision model, and (when enabled) overlay artifact.

The factory deliberately does not import execution, broker, reporting, or CLI
modules.  Callers own lifecycle concerns such as cache clearing for parallel
workers and the decision inputs supplied to the model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from leadlag.config.paths import project_root
from leadlag.config.schemas import AppConfig, ProductionV2RunConfig
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.ml_order_overlay import MLOrderOverlayModel, load_overlay_model
from leadlag.models.production_v2 import ProductionV2Model

__all__ = [
    "V2ModelBundle",
    "build_blpx_model",
    "build_v2_model_bundle",
    "model_config_fingerprint",
    "resolve_overlay_settings",
]


@dataclass(frozen=True)
class V2ModelBundle:
    """The model components selected for one run.

    The objects themselves are intentionally not copied.  A bundle owns the
    relationship between them: ``decision_model`` receives the same
    ``blpx_model`` and ``overlay_model`` that the caller records in its run
    manifest.  Callers must not share a bundle between concurrent workers.
    """

    run_config: ProductionV2RunConfig
    blpx_model: ProductionBLPXModel
    decision_model: ProductionV2Model
    overlay_model: MLOrderOverlayModel | None
    overlay_enabled: bool
    overlay_path: Path | None


def _canonical_overlay_values(app_config: AppConfig) -> tuple[bool, str | Path | None]:
    """Read overlay settings from the canonical V2 config.

    The nested ``AppConfig.ml_order_overlay`` section is retained for older
    configuration producers, but is not consulted here.  YAML compatibility is
    resolved before ``AppConfig`` construction; the factory therefore has one
    source of truth and does not repeat legacy key precedence.
    """

    run_cfg = app_config.v2
    return bool(run_cfg.ml_overlay_enabled), run_cfg.ml_overlay_model_dir or None


def resolve_overlay_settings(
    app_config: AppConfig,
    *,
    overlay_model_dir: str | Path | None = None,
) -> tuple[bool, Path | None]:
    """Resolve the enabled flag and artifact path for one model bundle.

    An explicit ``overlay_model_dir`` is an intentional caller override and is
    resolved even when the config flag is disabled.  The returned enabled flag
    remains the resolved config flag so research callers can load an artifact
    for an explicit comparison without silently changing production behavior.
    """

    enabled, configured_path = _canonical_overlay_values(app_config)
    selected = overlay_model_dir if overlay_model_dir is not None else configured_path
    if selected is None:
        return enabled, None

    path = Path(selected)
    if not path.is_absolute():
        path = project_root() / path
    return enabled, path


def model_config_fingerprint(app_config: AppConfig) -> str:
    """Return a stable digest of the effective, non-secret V2 model config."""

    if not isinstance(app_config, AppConfig):
        raise TypeError(
            "model_config_fingerprint requires a validated AppConfig; "
            "normalize raw YAML at the config boundary first"
        )
    payload = app_config.v2.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_v2_model_bundle(
    app_config: AppConfig,
    *,
    overlay_model: MLOrderOverlayModel | None = None,
    overlay_model_dir: str | Path | None = None,
    clear_blpx_cache: bool = False,
) -> V2ModelBundle:
    """Construct the canonical BLPX/V2/overlay components for one run.

    ``overlay_model`` takes precedence over loading from disk.  When no model
    object is supplied, a configured artifact is loaded only if the V2 overlay
    flag is enabled, or if an explicit directory override was provided.  This
    matches the existing live/backtest behavior while keeping the selection in
    one place.
    """

    if not isinstance(app_config, AppConfig):
        raise TypeError(
            "build_v2_model_bundle requires a validated AppConfig; "
            "normalize raw YAML at the config boundary first"
        )

    run_cfg = app_config.v2
    overlay_enabled, resolved_overlay_path = resolve_overlay_settings(
        app_config,
        overlay_model_dir=overlay_model_dir,
    )

    selected_overlay = overlay_model
    explicit_path = overlay_model_dir is not None
    should_load = (
        selected_overlay is None
        and resolved_overlay_path is not None
        and (overlay_enabled or explicit_path)
    )
    if should_load:
        assert resolved_overlay_path is not None
        selected_overlay = load_overlay_model(resolved_overlay_path)

    blpx_model = build_blpx_model(app_config, clear_cache=clear_blpx_cache)
    decision_model = ProductionV2Model(
        run_cfg,
        blpx_model=blpx_model,
        overlay_model=selected_overlay,
    )
    return V2ModelBundle(
        run_config=run_cfg,
        blpx_model=blpx_model,
        decision_model=decision_model,
        overlay_model=selected_overlay,
        overlay_enabled=overlay_enabled,
        overlay_path=resolved_overlay_path,
    )


def build_blpx_model(
    app_config: AppConfig,
    *,
    clear_cache: bool = False,
) -> ProductionBLPXModel:
    """Construct the canonical BLPX model without loading an overlay.

    Gap-distribution generation and other model-only tools need the BLPX
    signal/prior calculations but do not apply the ML order overlay.  Keeping
    this small factory beside ``build_v2_model_bundle`` makes that boundary
    explicit and prevents those tools from reimplementing config-to-model
    construction or accidentally loading the production artifact.
    """

    if not isinstance(app_config, AppConfig):
        raise TypeError(
            "build_blpx_model requires a validated AppConfig; "
            "normalize raw YAML at the config boundary first"
        )
    model = ProductionBLPXModel(app_config.v2.blpx)
    if clear_cache:
        model.clear_caches()
    return model
