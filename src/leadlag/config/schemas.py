"""Pydantic schemas for validated configuration variables."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """Strategy parameters validation schema."""
    model_config = {"frozen": True}

    model_name: str = Field(default="sector_relative_ensemble")
    k: int = Field(default=6, ge=1)
    lambda_reg: float = Field(default=0.75, ge=0.0, le=1.0)
    q: float = Field(default=0.3, ge=0.0, le=1.0)
    weight_mode: str = Field(default="signal")
    dispersion_filter: bool = Field(default=False)
    dispersion_metric: str = Field(default="long_short_mean_gap")
    v3_mode: str = Field(default="static")
    ewma_half_life: int = Field(default=45, ge=1)
    lambda_lw: float = Field(default=0.5, ge=0.0, le=1.0)
    lw_target: str = Field(default="equicorrelation")
    corr_window: int = Field(default=60, ge=1)
    include_v4_prior: bool = Field(default=True)
    signal_mode: str = Field(default="gap_residual")
    gap_open_coef: float = Field(default=0.70)
    topix_beta_coef: float = Field(default=0.6)
    beta_window: int = Field(default=60, ge=1)
    gamma: float = Field(default=0.5)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    vol_adjusted_target: bool = Field(default=True)

    # Risk parameters & start date for production runners
    start_date: str = Field(default="2015-01-01")
    var_confidence: float = Field(default=0.99, ge=0.0, le=1.0)
    var_window: int = Field(default=250, ge=1)
    var_warning: float = Field(default=0.02, ge=0.0, le=1.0)
    var_stop: float = Field(default=0.03, ge=0.0, le=1.0)
    es_warning: float = Field(default=0.025, ge=0.0, le=1.0)
    es_stop: float = Field(default=0.04, ge=0.0, le=1.0)
    daily_loss_warning: float = Field(default=0.015, ge=0.0, le=1.0)
    daily_loss_stop: float = Field(default=0.025, ge=0.0, le=1.0)
    monthly_loss_stop: float = Field(default=0.05, ge=0.0, le=1.0)
    max_net_exposure: float = Field(default=0.05, ge=0.0, le=1.0)
    max_gross_exposure: float = Field(default=2.0, ge=0.0)


class RiskConfig(BaseModel):
    """Risk management parameters validation schema."""
    model_config = {"frozen": True}

    var_confidence: float = Field(default=0.99, ge=0.0, le=1.0)
    var_window: int = Field(default=250, ge=1)
    var_warning: float = Field(default=0.02, ge=0.0, le=1.0)
    var_stop: float = Field(default=0.03, ge=0.0, le=1.0)
    es_warning: float = Field(default=0.025, ge=0.0, le=1.0)
    es_stop: float = Field(default=0.04, ge=0.0, le=1.0)
    daily_loss_warning: float = Field(default=0.015, ge=0.0, le=1.0)
    daily_loss_stop: float = Field(default=0.025, ge=0.0, le=1.0)
    monthly_loss_stop: float = Field(default=0.05, ge=0.0, le=1.0)
    max_net_exposure: float = Field(default=0.05, ge=0.0, le=1.0)
    max_gross_exposure: float = Field(default=2.0, ge=0.0)


class KabuApiConfig(BaseModel):
    """kabuステーション API configuration."""
    model_config = {"frozen": True}

    api_url: str = Field(default="http://localhost:18080/kabusapi")
    api_token: str = Field(default="")
    api_password: str = Field(default="")
    request_timeout: int = Field(default=10, ge=1)
    margin_trade_type: int = Field(default=3)
    account_type: int = Field(default=4)


class AppConfig(BaseModel):
    """Full application configuration."""
    model_config = {"frozen": True}

    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    kabu: KabuApiConfig = Field(default_factory=KabuApiConfig)
    output_base_dir: str = Field(default="results/sector_relative_ensemble")
    output_live_dir: str = Field(default="live/sector_relative_ensemble")
    run_audit: bool = Field(default=True)
