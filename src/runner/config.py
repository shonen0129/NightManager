"""runner/config.py — ProductionConfig dataclass.

Single definition of strategy + risk parameters used across all run modes.
Importing from here avoids circular dependencies between runner modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import PRODUCTION_DEFAULTS


@dataclass(frozen=True)
class ProductionConfig:
    """Immutable configuration for a production run.

    Strategy hyperparameters come from ``config.PRODUCTION_DEFAULTS``.
    Risk thresholds are hard-coded here (override via subclass if needed).
    """

    # Strategy parameters
    start_date: str = PRODUCTION_DEFAULTS["start_date"]
    k: int = PRODUCTION_DEFAULTS["k"]
    lambda_reg: float = PRODUCTION_DEFAULTS["lambda_reg"]
    q: float = PRODUCTION_DEFAULTS["q"]
    weight_mode: str = PRODUCTION_DEFAULTS["weight_mode"]
    dispersion_filter: bool = PRODUCTION_DEFAULTS["dispersion_filter"]
    dispersion_metric: str = PRODUCTION_DEFAULTS["dispersion_metric"]
    v3_mode: str = PRODUCTION_DEFAULTS["v3_mode"]
    ewma_half_life: int = PRODUCTION_DEFAULTS["ewma_half_life"]
    lambda_lw: float = PRODUCTION_DEFAULTS["lambda_lw"]
    lw_target: str = PRODUCTION_DEFAULTS["lw_target"]
    corr_window: int = PRODUCTION_DEFAULTS["corr_window"]
    include_v4_prior: bool = PRODUCTION_DEFAULTS["include_v4_prior"]
    signal_mode: str = PRODUCTION_DEFAULTS["signal_mode"]
    gap_open_coef: float = PRODUCTION_DEFAULTS["gap_open_coef"]
    topix_beta_coef: float = PRODUCTION_DEFAULTS["topix_beta_coef"]
    beta_window: int = PRODUCTION_DEFAULTS["beta_window"]
    gamma: float = PRODUCTION_DEFAULTS.get("gamma", 0.5)
    slippage_bps: float = PRODUCTION_DEFAULTS.get("slippage_bps", 5.0)
    vol_adjusted_target: bool = PRODUCTION_DEFAULTS["vol_adjusted_target"]

    # Risk thresholds
    var_confidence: float = 0.99
    var_window: int = 250
    var_warning: float = 0.02
    var_stop: float = 0.03
    es_warning: float = 0.025
    es_stop: float = 0.04
    daily_loss_warning: float = 0.015
    daily_loss_stop: float = 0.025
    monthly_loss_stop: float = 0.05
    max_net_exposure: float = 0.05
    max_gross_exposure: float = 3.0
