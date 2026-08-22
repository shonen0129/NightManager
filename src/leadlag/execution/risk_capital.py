"""Risk and capital allocation helpers.

This module was split from ``execution/helpers.py`` as part of P1-B1 to
isolate risk configuration, risk checks, gross-exposure adjustment, and
capital allocation from broker, pricing, output, and post-decision flow.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from leadlag.core import allocator as domain_allocator
from leadlag.core.portfolio import adjust_gross_exposure, classify_actions
from leadlag.core.risk import evaluate_risk_checks
from leadlag.core.types import RiskConfig
from leadlag.execution.config import StrategyConfig as ProductionConfig

logger = logging.getLogger(__name__)


def build_risk_config(config: ProductionConfig) -> RiskConfig:
    """Build a ``RiskConfig`` from production strategy configuration."""
    return RiskConfig(
        var_confidence=config.var_confidence,
        var_window=config.var_window,
        var_method=config.var_method,
        var_warning=config.var_warning,
        var_stop=config.var_stop,
        es_warning=config.es_warning,
        es_stop=config.es_stop,
        daily_loss_warning=config.daily_loss_warning,
        daily_loss_stop=config.daily_loss_stop,
        monthly_loss_stop=config.monthly_loss_stop,
        max_net_exposure=config.max_net_exposure,
        max_gross_exposure=config.max_gross_exposure,
    )


def run_risk_checks(
    decision: dict,
    total_buy_allocated: float,
    total_sell_allocated: float,
    max_capital: float,
    hist_daily_returns: pd.Series,
    config: ProductionConfig,
) -> dict:
    """Run risk checks against the current decision and return a report dict."""
    weights = np.asarray(decision["weight"], dtype=float)
    risk_config = build_risk_config(config)
    report = evaluate_risk_checks(
        weights=weights,
        total_buy_allocated=total_buy_allocated,
        total_sell_allocated=total_sell_allocated,
        max_capital=max_capital,
        hist_daily_returns=hist_daily_returns,
        config=risk_config,
    )
    return {
        "target_net_exposure": report.target_net_exposure,
        "target_gross_exposure": report.target_gross_exposure,
        "allocated_net_ratio": report.allocated_net_ratio,
        "allocated_gross_ratio": report.allocated_gross_ratio,
        "var_es": {
            "available": report.var_es.available,
            "samples": report.var_es.samples,
            "window": report.var_es.window,
            "var_loss": report.var_es.var_loss,
            "es_loss": report.var_es.es_loss,
        },
        "warning_breaches": report.warning_breaches,
        "stop_breaches": report.stop_breaches,
        "is_blocked": report.is_blocked,
    }


def auto_adjust_gross_exposure(decision: dict, config: ProductionConfig) -> dict:
    """Scale weights down if gross exposure exceeds the configured limit.

    ``config.max_gross_exposure`` is interpreted in raw-weight units (before
    ``side_leverage`` is applied to notional). This matches the unit convention
    used by the portfolio construction stage and the existing test suite. The
    caller must ensure that the configured limit is expressed in the same units
    as the weights; for example, with ``side_leverage=1.5`` and a desired
    notional gross of 3.0, ``max_gross_exposure`` should be ``2.0`` (not 3.0).

    Returns a new decision dict with gross exposure metadata and, if needed,
    scaled weights/actions.
    """
    weights = np.asarray(decision["weight"], dtype=float)
    result = adjust_gross_exposure(weights, config.max_gross_exposure)

    adjusted = dict(decision)
    adjusted["gross_before"] = result.gross_before
    adjusted["gross_limit"] = result.gross_limit
    adjusted["gross_adjusted"] = result.was_adjusted
    adjusted["gross_adjustment_factor"] = result.adjustment_factor
    adjusted["gross_after"] = result.gross_after

    if result.was_adjusted:
        scaled = weights * result.adjustment_factor
        adjusted["weight"] = scaled
        adjusted["action"] = classify_actions(scaled)

    return adjusted


def allocate_capital(
    decision: dict,
    manual_opens: dict,
    max_capital: float,
    max_net_exposure: float | None = None,
    side_leverage: float = domain_allocator.DEFAULT_SIDE_LEVERAGE,
) -> dict:
    """Convert decision weights to share quantities using open prices and capital."""
    tickers = decision["tickers"]
    weights = np.asarray(decision["weight"], dtype=float)
    allocation = domain_allocator.allocate_capital(
        weights=weights,
        tickers=tickers,
        open_prices=manual_opens,
        max_capital=float(max_capital),
        max_net_exposure=max_net_exposure,
        side_leverage=side_leverage,
    )
    return {
        "qty": allocation.quantities.astype(int),
        "allocated": allocation.allocated_amounts,
        "buy_budget": float(allocation.buy_budget),
        "sell_budget": float(allocation.sell_budget),
        "gross_budget": float(allocation.gross_budget),
    }
