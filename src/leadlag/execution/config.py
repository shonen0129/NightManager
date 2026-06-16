"""Execution configuration: loads and validates configuration from YAML and environment using Pydantic."""

from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Any
import yaml
from dotenv import load_dotenv

from leadlag.config import AppConfig, KabuApiConfig, RiskConfig, StrategyConfig

logger = logging.getLogger(__name__)

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


def load_config_from_yaml(yaml_path: str | Path | None = None) -> AppConfig:
    """Load config from YAML, merge with env variables, and validate via Pydantic.

    Args:
        yaml_path: Path to the configuration YAML file.
                   Defaults to project_root/configs/production.yaml if exists.
    """
    if yaml_path is None:
        default_yaml = Path(__file__).parent.parent.parent.parent / "configs" / "production.yaml"
        if default_yaml.exists():
            yaml_path = default_yaml

    yaml_data: dict[str, Any] = {}
    if yaml_path and Path(yaml_path).exists():
        logger.info("Loading configuration from %s", yaml_path)
        with open(yaml_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        logger.info("No configuration YAML found, using default settings")

    # Extract sections
    model_data = yaml_data.get("model", {})
    portfolio_data = yaml_data.get("portfolio", {})
    costs_data = yaml_data.get("costs", {})
    res_data = yaml_data.get("residualization", {})
    risk_data = yaml_data.get("risk", {})
    output_data = yaml_data.get("output", {})

    # Risk parameters — single source via helper (eliminates duplication)
    risk_kwargs = _map_risk_section(risk_data)

    # Map StrategyConfig fields (strategy / signal / portfolio params only)
    strategy_kwargs = {
        "model_name": model_data.get("name", "sector_relative_ensemble"),
        "k": model_data.get("k", 6),
        "lambda_reg": model_data.get("lambda_reg", 0.75),
        "q": portfolio_data.get("long_short_frac", 0.3),
        "weight_mode": portfolio_data.get("weight_mode", "signal"),
        "dispersion_filter": portfolio_data.get("dispersion_filter", False),
        "dispersion_metric": portfolio_data.get("dispersion_metric", "long_short_mean_gap"),
        "v3_mode": portfolio_data.get("v3_mode", "static"),
        "ewma_half_life": portfolio_data.get("ewma_half_life", 45),
        "lambda_lw": portfolio_data.get("lambda_lw", 0.5),
        "lw_target": portfolio_data.get("lw_target", "equicorrelation"),
        "corr_window": portfolio_data.get("corr_window", 60),
        "include_v4_prior": portfolio_data.get("include_v4_prior", True),
        "signal_mode": portfolio_data.get("signal_mode", "gap_residual"),
        "gap_open_coef": portfolio_data.get("gap_open_coef", 0.70),
        "topix_beta_coef": res_data.get("topix_beta_coef", 1.20),
        "beta_window": res_data.get("beta_window", 60),
        "gamma": portfolio_data.get("gamma", 0.5),
        "slippage_bps": costs_data.get("slippage_bps_per_side", 5.0),
        "vol_adjusted_target": portfolio_data.get("vol_adjusted_target", True),
        "start_date": yaml_data.get("start_date", "2015-01-01"),
        # Include risk thresholds for backward compat with production runners
        # that pass a single StrategyConfig to both strategy and risk layers.
        **risk_kwargs,
    }

    # Override strategy from env if present
    if "STRATEGY_SLIPPAGE_BPS" in os.environ:
        strategy_kwargs["slippage_bps"] = float(os.environ["STRATEGY_SLIPPAGE_BPS"])


    kabu_url = _normalize_kabu_api_url(os.environ.get("KABU_API_URL", "http://localhost:18080"))
    kabu_token = os.environ.get("KABU_API_TOKEN", "")
    kabu_password = os.environ.get("KABU_API_PASSWORD", "")
    kabu_timeout = int(os.environ.get("KABU_REQUEST_TIMEOUT", "10"))
    kabu_margin = int(os.environ.get("KABU_MARGIN_TRADE_TYPE", "3"))
    kabu_account = int(os.environ.get("KABU_ACCOUNT_TYPE", "4"))

    # Basic validations (can also be done via Pydantic validators, but let's keep inline for exact logic match)
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

    strategy_cfg = StrategyConfig(**strategy_kwargs)
    risk_cfg = RiskConfig(**risk_kwargs)

    return AppConfig(
        strategy=strategy_cfg,
        risk=risk_cfg,
        kabu=kabu_cfg,
        output_base_dir=output_data.get("base_dir", "results/sector_relative_ensemble"),
        output_live_dir=output_data.get("live_dir", "live/sector_relative_ensemble"),
        run_audit=output_data.get("run_audit", True),
    )
