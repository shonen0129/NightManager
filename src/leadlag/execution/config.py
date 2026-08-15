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
    return {
        "var_confidence": risk_data.get("var_confidence", 0.99),
        "var_window": risk_data.get("var_window", 250),
        "var_method": risk_data.get("var_method", "historical"),
        "var_warning": risk_data.get("var_warning", 0.02),
        "var_stop": risk_data.get("var_stop", 0.03),
        "es_warning": risk_data.get("es_warning", 0.025),
        "es_stop": risk_data.get("es_stop", 0.04),
        "daily_loss_warning": risk_data.get("daily_loss_warning", 0.015),
        "daily_loss_stop": risk_data.get("daily_loss_stop", 0.025),
        "monthly_loss_stop": risk_data.get("monthly_loss_stop", 0.05),
        "max_net_exposure": risk_data.get("max_net_exposure", 0.05),
        "max_gross_exposure": risk_data.get("max_gross_exposure", 2.0),
    }


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
    blpx = v2_cfg.blpx
    costs = v2_cfg.costs

    # Map StrategyConfig fields (legacy nested values take precedence over V2)
    strategy_kwargs = {
        "model_name": model_data.get("name", "sector_relative_ensemble"),
        "k": model_data.get("k") if "k" in model_data else blpx.k,
        "lambda_reg": model_data.get("lambda_reg") if "lambda_reg" in model_data else blpx.lambda_reg,
        "q": portfolio_data.get("long_short_frac") if "long_short_frac" in portfolio_data else blpx.q,
        "weight_mode": portfolio_data.get("weight_mode") if "weight_mode" in portfolio_data else blpx.weight_mode,
        "dispersion_filter": portfolio_data.get("dispersion_filter", False),
        "dispersion_metric": portfolio_data.get("dispersion_metric", "long_short_mean_gap"),
        "v3_mode": portfolio_data.get("v3_mode", "static"),
        "ewma_half_life": model_data.get("ewma_half_life") if "ewma_half_life" in model_data else blpx.ewma_halflife,
        "lambda_lw": model_data.get("lambda_lw") if "lambda_lw" in model_data else blpx.lambda_lw,
        "lw_target": model_data.get("lw_target") if "lw_target" in model_data else blpx.lw_target,
        "corr_window": model_data.get("corr_window") if "corr_window" in model_data else blpx.corr_window,
        "include_v4_prior": model_data.get("include_v4_prior") if "include_v4_prior" in model_data else blpx.include_v4_prior,
        "signal_mode": portfolio_data.get("signal_mode", "gap_residual"),
        "gap_open_coef": portfolio_data.get("gap_open_coef") if "gap_open_coef" in portfolio_data else blpx.gap_open_coef,
        "topix_beta_coef": res_data.get("topix_beta_coef") if "topix_beta_coef" in res_data else blpx.topix_beta_coef,
        "beta_window": res_data.get("beta_window") if "beta_window" in res_data else blpx.beta_window,
        "beta_ewma_halflife": res_data.get("beta_ewma_halflife"),
        "beta_shrinkage": res_data.get("shrinkage") if "shrinkage" in res_data else v2_cfg.residualization_beta_shrinkage,
        "beta_winsor_sigma": res_data.get("winsor_sigma") if "winsor_sigma" in res_data else v2_cfg.residualization_beta_winsor_sigma,
        "gamma": portfolio_data.get("gamma", 0.5),
        "slippage_bps": costs_data.get("slippage_bps_per_side") if "slippage_bps_per_side" in costs_data else costs.slippage_bps_per_side,
        "vol_adjusted_target": portfolio_data.get("vol_adjusted_target") if "vol_adjusted_target" in portfolio_data else blpx.vol_adjusted_target,
        "min_raw_weight": portfolio_data.get("min_raw_weight") if "min_raw_weight" in portfolio_data else blpx.min_raw_weight,
        "overnight_alpha_long": costs_data.get("overnight_alpha_long") if "overnight_alpha_long" in costs_data else costs.overnight_alpha_long,
        "overnight_alpha_short": costs_data.get("overnight_alpha_short") if "overnight_alpha_short" in costs_data else costs.overnight_alpha_short,
        "buy_interest_annual": costs_data.get("buy_interest_annual") if "buy_interest_annual" in costs_data else costs.buy_interest_annual,
        "borrow_fee_annual": costs_data.get("borrow_fee_annual") if "borrow_fee_annual" in costs_data else costs.borrow_fee_annual,
        "reverse_fee_bps": costs_data.get("reverse_fee_bps") if "reverse_fee_bps" in costs_data else costs.reverse_fee_bps,
        "side_leverage": float(
            yaml_data.get("execution", {}).get("side_leverage") if "execution" in yaml_data and "side_leverage" in (yaml_data.get("execution") or {}) else costs.side_leverage
        ),
        "start_date": yaml_data.get("start_date", "2015-01-05"),
        # Copula / min-variance (canonical values live in V2 BLPX sub-model)
        "copula_enabled": model_data.get("copula_enabled") if "copula_enabled" in model_data else blpx.copula_enabled,
        "copula_blend_weight": model_data.get("copula_blend_weight") if "copula_blend_weight" in model_data else blpx.copula_blend_weight,
        "copula_dynamic_blend": model_data.get("copula_dynamic_blend") if "copula_dynamic_blend" in model_data else blpx.copula_dynamic_blend,
        "copula_stress_threshold": model_data.get("copula_stress_threshold") if "copula_stress_threshold" in model_data else blpx.copula_stress_threshold,
        "copula_nu_init": model_data.get("copula_nu_init") if "copula_nu_init" in model_data else blpx.copula_nu_init,
        "copula_marginal_method": model_data.get("copula_marginal_method", "empirical"),
        "minvar_enabled": portfolio_data.get("minvar_enabled") if "minvar_enabled" in portfolio_data else blpx.minvar_enabled,
        "minvar_alpha": portfolio_data.get("minvar_alpha") if "minvar_alpha" in portfolio_data else blpx.minvar_alpha,
        # Include risk thresholds for backward compat with production runners
        # that pass a single StrategyConfig to both strategy and risk layers.
        **risk_kwargs,
    }

    # Override strategy from env if present
    if "STRATEGY_SLIPPAGE_BPS" in os.environ:
        strategy_kwargs["slippage_bps"] = float(os.environ["STRATEGY_SLIPPAGE_BPS"])


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

    strategy_cfg = StrategyConfig(**strategy_kwargs)
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
        with open(yaml_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        logger.info("No configuration YAML found, using default settings")

    return build_app_config_from_dict(yaml_data, strict=strict)
