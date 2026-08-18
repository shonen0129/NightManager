"""Execution configuration: loads and validates configuration from YAML and environment using Pydantic."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from leadlag.config.paths import live, results
from leadlag.config.schemas import (
    AppConfig,
    BLPXConfig,
    CostConfig,
    KabuApiConfig,
    MLOrderOverlayConfig,
    ProductionV2RunConfig,
    RiskConfig,
    TachibanaApiConfig,
)
from leadlag.config.schemas import (
    StrategyConfig as StrategyConfig,
)

logger = logging.getLogger(__name__)


_BASE_KEY = "__base__"


def _resolve_config_path(path: str, relative_to: Path) -> Path:
    """Resolve an include path relative to the containing YAML file."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (relative_to.parent / p).resolve()


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge override into base. Dicts are merged; other values override."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = _deep_merge(merged.get(k), v) if k in merged else v
        return merged
    return override


def _load_yaml_with_base(
    yaml_path: str | Path,
    _seen: set[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML file, recursively merging any ``__base__`` includes."""
    _seen = _seen or set()
    yaml_path = Path(yaml_path).resolve()
    key = str(yaml_path)
    if key in _seen:
        raise ValueError(f"Circular __base__ reference detected: {yaml_path}")
    _seen.add(key)

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    base_path = data.pop(_BASE_KEY, None)
    if base_path:
        base_file = _resolve_config_path(str(base_path), yaml_path)
        base_data = _load_yaml_with_base(base_file, _seen)
        data = _deep_merge(base_data, data)

    return data


class UnknownConfigKeyError(ValueError):
    """Raised when a YAML config contains an unrecognized top-level key."""


# Known top-level YAML sections. Unknown sections trigger UnknownConfigKeyError
# in strict mode to catch typos early.
# Nested section names are kept for backward compatibility; flat V2 keys are
# derived from the Pydantic schemas below.
_ALLOWED_TOP_LEVEL_KEYS = frozenset({
    "model",
    "signal_components",
    "final_signal",
    "ranking",
    "gross_scaling",
    "gap_distribution",
    "portfolio",
    "risk",
    "costs",
    "blpx",
    "residualization",
    "features",
    "multi_horizon_blend",
    "cs_feature_overlay",
    "fallback",
    "audit",
    "ml_order_overlay",
    "execution",
    "output",
    "broker",
    "start_date",
})


def _allowed_top_level_keys() -> frozenset[str]:
    """Return the full set of recognized top-level YAML keys."""
    v2_keys = set(ProductionV2RunConfig.model_fields)
    blpx_flat = {f"blpx_{k}" for k in BLPXConfig.model_fields}
    costs_flat = set(CostConfig.model_fields)
    return _ALLOWED_TOP_LEVEL_KEYS | frozenset(v2_keys | blpx_flat | costs_flat)


# Load .env files from typical locations
_env_paths = [
    Path(__file__).parent.parent.parent.parent / ".env",  # Project root
    Path(__file__).parent.parent.parent / ".env",  # src/
    Path(__file__).parent / ".env",  # execution/
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)


def _normalize_kabu_api_url(api_url: str) -> str:
    """Ensure the API base URL includes /kabusapi."""
    if not api_url:
        return api_url
    normalized = api_url.rstrip("/")
    if normalized.endswith("/kabusapi"):
        return normalized
    return f"{normalized}/kabusapi"


def _map_risk_section(risk_data: dict) -> dict:
    """YAML の risk セクションを RiskConfig キーにマッピングする（単一正本）.

    strategy_kwargs と risk_kwargs の両方に同じマッピングが必要なケースで
    このヘルパーを使うことで、変更箇所を 1 か所に集約する。
    """
    defaults = RiskConfig().model_dump()
    return {k: risk_data.get(k, defaults[k]) for k in defaults}


def build_app_config_from_dict(yaml_data: dict[str, Any], strict: bool = False) -> AppConfig:
    """Build a validated ``AppConfig`` from a raw YAML-style dict + env.

    This is the single Pydantic construction point used by
    ``load_config_from_yaml`` and by callers that already have a parsed
    config dict.

    Args:
        yaml_data: Parsed YAML dict.
        strict: If True, reject top-level keys not in the allowlist.
            This catches typos in user-facing config files.
    """
    if strict:
        unknown = set(yaml_data.keys()) - _allowed_top_level_keys()
        if unknown:
            raise UnknownConfigKeyError(
                f"Unknown top-level config keys: {sorted(unknown)}. "
                f"If these are intentional, run with strict=False."
            )

    # Extract AppConfig-level sections. V2 runtime parameters are normalized
    # through ``parse_run_config`` below, so legacy nested sections are only
    # used for fields that live exclusively in ``StrategyConfig``.
    model_data = yaml_data.get("model", {})
    portfolio_data = yaml_data.get("portfolio", {})
    costs_data = yaml_data.get("costs", {})
    res_data = yaml_data.get("residualization", {})
    risk_data = yaml_data.get("risk", {})
    output_data = yaml_data.get("output", {})

    # Risk parameters — single source via helper (eliminates duplication)
    risk_kwargs = _map_risk_section(risk_data)

    # V2 config is the single source of truth for production parameters.
    # Build it before StrategyConfig so we can fall back to its values.
    from leadlag.models.production_v2 import parse_run_config

    v2_cfg = parse_run_config(yaml_data)

    # Build StrategyConfig from the canonical V2 config. Legacy nested values
    # still take precedence; this centralizes the duplicated mapping logic.
    side_leverage_raw = (
        yaml_data.get("execution", {}).get("side_leverage")
        if "execution" in yaml_data and "side_leverage" in (yaml_data.get("execution") or {})
        else None
    )

    strategy_cfg = StrategyConfig.from_v2(
        v2_cfg,
        model_data=model_data,
        portfolio_data=portfolio_data,
        residualization_data=res_data,
        costs_data=costs_data,
        risk_kwargs=risk_kwargs,
        start_date=yaml_data.get("start_date", "2015-01-05"),
        side_leverage=float(side_leverage_raw) if side_leverage_raw is not None else None,
        env_slippage_bps=float(os.environ["STRATEGY_SLIPPAGE_BPS"]) if "STRATEGY_SLIPPAGE_BPS" in os.environ else None,
    )


    broker_provider = os.environ.get("BROKER_PROVIDER", "kabu").lower().strip()

    # Load Kabu config
    kabu_url = _normalize_kabu_api_url(os.environ.get("KABU_API_URL", "http://localhost:18080"))
    kabu_token = os.environ.get("KABU_API_TOKEN", "")
    kabu_password = os.environ.get("KABU_API_PASSWORD", "")
    kabu_timeout = int(os.environ.get("KABU_REQUEST_TIMEOUT", "10"))
    kabu_margin = int(os.environ.get("KABU_MARGIN_TRADE_TYPE", "3"))
    kabu_account = int(os.environ.get("KABU_ACCOUNT_TYPE", "4"))

    if broker_provider == "kabu":
        if not kabu_url:
            raise ValueError("Kabu API URL is required (set KABU_API_URL env)")
        if not kabu_url.startswith(("http://", "https://")):
            raise ValueError(f"Kabu API URL must start with http:// or https://, got: {kabu_url}")
        if not kabu_token and not kabu_password:
            logger.warning(
                "Neither KABU_API_TOKEN nor KABU_API_PASSWORD is set. Token will need to be provided."
            )
        if kabu_margin not in (1, 2, 3):
            raise ValueError(f"Invalid margin trade type: {kabu_margin}. Supported: 1, 2, 3")
        if kabu_account not in (2, 4, 12):
            raise ValueError(f"Invalid account type: {kabu_account}. Supported: 2, 4, 12")

    kabu_cfg = KabuApiConfig(
        api_url=kabu_url,
        api_token=kabu_token,
        api_password=kabu_password,
        request_timeout=kabu_timeout,
        margin_trade_type=kabu_margin,
        account_type=kabu_account,
    )

    # Load Tachibana config
    tachi_url = os.environ.get("TACHIBANA_API_URL", "https://kabuka.e-shiten.jp/e_api_v4r9")
    tachi_auth_id = os.environ.get("TACHIBANA_AUTH_ID", "")
    tachi_priv_key = os.environ.get("TACHIBANA_PRIVATE_KEY_PATH", "")
    tachi_sec_pw = os.environ.get("TACHIBANA_SECOND_PASSWORD", "")
    tachi_timeout = int(os.environ.get("TACHIBANA_REQUEST_TIMEOUT", "10"))
    tachi_margin = int(os.environ.get("TACHIBANA_MARGIN_TRADE_TYPE", "3"))
    tachi_account = int(os.environ.get("TACHIBANA_ACCOUNT_TYPE", "4"))

    if broker_provider == "tachibana":
        if not tachi_url:
            raise ValueError("Tachibana API URL is required (set TACHIBANA_API_URL env)")
        if not tachi_url.startswith(("http://", "https://")):
            raise ValueError(f"Tachibana API URL must start with http:// or https://, got: {tachi_url}")
        if not tachi_auth_id:
            raise ValueError("Tachibana Auth ID is required (set TACHIBANA_AUTH_ID env)")
        if tachi_margin not in (1, 2, 3):
            raise ValueError(f"Invalid margin trade type: {tachi_margin}. Supported: 1, 2, 3")
        if tachi_account not in (2, 4, 12):
            raise ValueError(f"Invalid account type: {tachi_account}. Supported: 2, 4, 12")

    tachi_cfg = TachibanaApiConfig(
        api_url=tachi_url,
        auth_id=tachi_auth_id,
        private_key_path=tachi_priv_key,
        second_password=tachi_sec_pw,
        request_timeout=tachi_timeout,
        margin_trade_type=tachi_margin,
        account_type=tachi_account,
    )

    risk_cfg = RiskConfig(**risk_kwargs)

    ml_overlay_data = yaml_data.get("ml_order_overlay", {})
    ml_overlay_cfg = MLOrderOverlayConfig(
        enabled=ml_overlay_data.get("enabled") if "enabled" in ml_overlay_data else v2_cfg.ml_overlay_enabled,
        model_dir=ml_overlay_data.get("model_dir") if "model_dir" in ml_overlay_data else v2_cfg.ml_overlay_model_dir,
        use_ticker=ml_overlay_data.get("use_ticker") if "use_ticker" in ml_overlay_data else v2_cfg.ml_overlay_use_ticker,
        use_classification=ml_overlay_data.get("use_classification") if "use_classification" in ml_overlay_data else v2_cfg.ml_overlay_use_classification,
        per_ticker_interactions=ml_overlay_data.get("per_ticker_interactions") if "per_ticker_interactions" in ml_overlay_data else v2_cfg.ml_overlay_per_ticker_interactions,
    )

    gap_dir = (yaml_data.get("gap_distribution") or {}).get("dir")
    if gap_dir is None:
        gap_dir = str(v2_cfg.gap_input_dir or "")

    return AppConfig(
        strategy=strategy_cfg,
        risk=risk_cfg,
        v2=v2_cfg,
        kabu=kabu_cfg,
        tachibana=tachi_cfg,
        ml_order_overlay=ml_overlay_cfg,
        broker_provider=broker_provider,
        output_base_dir=output_data.get("base_dir", str(results("sector_relative_ensemble"))),
        output_live_dir=output_data.get("live_dir", str(live("sector_relative_ensemble"))),
        run_audit=output_data.get("run_audit", True),
        gap_distribution_dir=gap_dir,
    )


def load_config_from_yaml(
    yaml_path: str | Path | None = None,
    strict: bool = False,
) -> AppConfig:
    """Load config from YAML, merge with env variables, and validate via Pydantic.

    Args:
        yaml_path: Path to the configuration YAML file.
                   Defaults to project_root/configs/production/production.yaml if exists.
        strict: If True, reject unrecognized top-level YAML keys.
    """
    if yaml_path is None:
        default_yaml = Path(__file__).parent.parent.parent.parent / "configs" / "production" / "production.yaml"
        if default_yaml.exists():
            yaml_path = default_yaml

    yaml_data: dict[str, Any] = {}
    if yaml_path and Path(yaml_path).exists():
        logger.info("Loading configuration from %s", yaml_path)
        yaml_data = _load_yaml_with_base(yaml_path)
    else:
        logger.info("No configuration YAML found, using default settings")

    return build_app_config_from_dict(yaml_data, strict=strict)
