"""runner/helpers.py — shared utility functions for all runner modules.

Contains pure utility functions that are used by decision.py, fast.py,
close.py, and backtest.py to avoid code duplication.

Nothing here starts a run; these are building blocks only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from broker.base import BrokerClient
from broker.factory import create_broker_from_args
from config import KABU_API_CONFIG, get_validated_kabu_config
from data_loader import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from domain.models.types import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    RiskConfig,
)
from domain.portfolio import allocator as domain_allocator
from domain.portfolio.optimizer import adjust_gross_exposure, classify_actions
from domain.risk.metrics import evaluate_risk_checks
from results_format import create_results_output_dir
from runner.config import ProductionConfig
from services.cache_service import get_hist_returns_for_risk as _get_hist_returns_for_risk
from services.formatting import (
    log_decision_summary as _log_decision_summary,
    print_risk_report as _print_risk_report,
    print_text_orders as _print_text_orders,
)
from strategy import LeadLagStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def build_output_dir(
    output_root: str,
    run_tag: Optional[str],
    run_name: str,
) -> str:
    return create_results_output_dir(
        run_name=run_name,
        output_root=output_root,
        run_tag=run_tag,
        manifest_extra={"entry_point": "production.py"},
    )


# ---------------------------------------------------------------------------
# Broker client
# ---------------------------------------------------------------------------


def build_api_client(
    api_url: Optional[str],
    api_token: Optional[str],
    api_dry_run: bool = False,
) -> BrokerClient:
    """Build and validate a BrokerClient.

    Delegates to ``broker.factory.create_broker_from_args``.
    When ``api_dry_run=True``, returns a ``DryRunBrokerClient`` (no network).
    """
    final_api_url = api_url if api_url else KABU_API_CONFIG.get("api_url")
    final_api_token = api_token if api_token else KABU_API_CONFIG.get("api_token")
    request_timeout = KABU_API_CONFIG.get("request_timeout", 10)

    if not final_api_url:
        try:
            validated = get_validated_kabu_config()
            final_api_url = validated["api_url"]
        except ValueError as e:
            logger.error("Configuration validation failed: %s", e)
            raise

    api_password = os.environ.get("KABU_API_PASSWORD", "")
    margin_trade_type = KABU_API_CONFIG.get("margin_trade_type", 3)
    account_type = KABU_API_CONFIG.get("account_type", 4)

    client = create_broker_from_args(
        api_url=final_api_url,
        api_token=final_api_token or None,
        api_password=api_password or None,
        dry_run=api_dry_run,
        margin_trade_type=margin_trade_type,
        account_type=account_type,
        request_timeout=request_timeout,
    )

    logger.info("[API] Checking API connectivity...")
    if not client.health_check():
        if api_dry_run:
            logger.warning("[API] Health check failed, continuing in dry-run mode...")
        else:
            raise RuntimeError(
                "Failed to connect to kabuステーション API. "
                "Verify API URL and token are correct."
            )
    else:
        logger.info("[API] Connection successful")

    return client


def resolve_wallet_capital(api_client: BrokerClient) -> float:
    wallet = api_client.get_wallet()
    cash_available = float(wallet.cash_available)
    logger.info(
        "[CAPITAL] Using cash wallet balance for sizing: %s JPY",
        f"{cash_available:,.0f}",
    )
    return cash_available


# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------


def build_strategy(config: ProductionConfig, df_exec: pd.DataFrame) -> LeadLagStrategy:
    return LeadLagStrategy(
        df_exec=df_exec,
        K=config.k,
        lambda_reg=config.lambda_reg,
        q=config.q,
        weight_mode=config.weight_mode,
        dispersion_filter=config.dispersion_filter,
        dispersion_metric=config.dispersion_metric,
        v3_mode=config.v3_mode,
        ewma_half_life=config.ewma_half_life,
        lambda_lw=config.lambda_lw,
        lw_target=config.lw_target,
        corr_window=config.corr_window,
        include_v4_prior=config.include_v4_prior,
        signal_mode=config.signal_mode,
        gap_open_coef=config.gap_open_coef,
        topix_beta_coef=config.topix_beta_coef,
        beta_window=config.beta_window,
        slippage_bps=config.slippage_bps,
    )


# ---------------------------------------------------------------------------
# Risk checks
# ---------------------------------------------------------------------------


def build_risk_config(config: ProductionConfig) -> RiskConfig:
    return RiskConfig(
        var_confidence=config.var_confidence,
        var_window=config.var_window,
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


# ---------------------------------------------------------------------------
# Gross exposure auto-adjustment
# ---------------------------------------------------------------------------


def auto_adjust_gross_exposure(decision: dict, config: ProductionConfig) -> dict:
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


# ---------------------------------------------------------------------------
# Capital allocation
# ---------------------------------------------------------------------------


def allocate_capital(
    decision: dict,
    manual_opens: dict,
    max_capital: float,
    max_net_exposure: float | None = None,
) -> dict:
    tickers = decision["tickers"]
    weights = np.asarray(decision["weight"], dtype=float)
    allocation = domain_allocator.allocate_capital(
        weights=weights,
        tickers=tickers,
        open_prices=manual_opens,
        max_capital=float(max_capital),
        max_net_exposure=max_net_exposure,
    )
    return {
        "qty": allocation.quantities.astype(int),
        "allocated": allocation.allocated_amounts,
        "buy_budget": float(allocation.buy_budget),
        "sell_budget": float(allocation.sell_budget),
        "gross_budget": float(allocation.gross_budget),
    }


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------


def save_decision_output(
    decision_df: pd.DataFrame, output_dir: str, trade_date: pd.Timestamp
) -> str:
    out_path = os.path.join(output_dir, f"decision_{trade_date.strftime('%Y%m%d')}.csv")
    decision_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ---------------------------------------------------------------------------
# Order submission via broker
# ---------------------------------------------------------------------------


def submit_orders_via_api(
    decision_df: pd.DataFrame,
    api_client: BrokerClient,
    output_dir: str,
    dry_run: bool = False,
) -> dict:
    """Submit trade orders to the broker API.

    Args:
        decision_df: Trade decision DataFrame (ticker, action, quantity, …)
        api_client: BrokerClient instance
        output_dir: Directory to write api_execution_log.json
        dry_run: If True, simulate without submitting

    Returns:
        Summary dict (buy_results, sell_results, submitted_orders_count, …)
    """
    active_orders = decision_df[decision_df["action"].isin(["BUY", "SELL"])].copy()
    qty = pd.to_numeric(active_orders["quantity"], errors="coerce").fillna(0)
    valid_orders = active_orders[qty > 0].copy()

    skipped_count = len(active_orders) - len(valid_orders)
    if skipped_count > 0:
        logger.info("Skipping orders with quantity<=0: %d", skipped_count)

    buy_count = int((valid_orders["action"] == "BUY").sum())
    sell_count = int((valid_orders["action"] == "SELL").sum())

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "buy_orders_count": buy_count,
        "sell_orders_count": sell_count,
        "buy_results": [],
        "sell_results": [],
    }

    order_requests = []
    for _, row in valid_orders.iterrows():
        limit_price = row.get("limit_price", None)
        if limit_price is not None and pd.notna(limit_price):
            order_requests.append(
                OrderRequest(
                    ticker=str(row["ticker"]),
                    side=OrderSide(str(row["action"])),
                    quantity=int(row["quantity"]),
                    order_type=OrderType.LIMIT,
                    limit_price=float(limit_price),
                )
            )
        else:
            order_requests.append(
                OrderRequest(
                    ticker=str(row["ticker"]),
                    side=OrderSide(str(row["action"])),
                    quantity=int(row["quantity"]),
                    order_type=OrderType.MARKET,
                )
            )
    expected_orders_count = len(order_requests)
    summary["expected_orders_count"] = expected_orders_count

    if dry_run:
        logger.info("[DRY RUN MODE] Simulating order submission (no actual orders sent)...")
        for req in order_requests:
            clean = req.ticker.replace(".T", "")
            simulated = {
                "order_id": f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{clean}",
                "status": "SIMULATED",
                "ticker": req.ticker,
                "side": req.side.value,
                "quantity": req.quantity,
            }
            if req.order_type == OrderType.LIMIT:
                simulated["order_type"] = "LO"
                simulated["limit_price"] = req.limit_price
                lp_val = req.limit_price
                lp_str = f"{lp_val:.1f}" if lp_val % 1 != 0 else f"{int(lp_val)}"
                logger.info(
                    "  [SIMULATED %s] %s: %d shares @ LIMIT %s JPY",
                    req.side.value, req.ticker, req.quantity, lp_str,
                )
            else:
                simulated["order_type"] = "MO"
                logger.info(
                    "  [SIMULATED %s] %s: %d shares",
                    req.side.value, req.ticker, req.quantity,
                )
            if req.side.value == "BUY":
                summary["buy_results"].append(simulated)
            else:
                summary["sell_results"].append(simulated)
    else:
        logger.info("[LIVE MODE] Submitting orders to broker API...")
        if order_requests:
            results = api_client.submit_orders_batch(order_requests, delay_ms=250)
            for result in results:
                side = result.side.value
                logger.info(
                    "  [%s SUBMITTED] %s: %d shares (Order ID: %s)",
                    side, result.ticker, result.quantity, result.order_id,
                )
                result_dict = {
                    "order_id": result.order_id,
                    "status": result.status.value,
                    "ticker": result.ticker,
                    "side": side,
                    "quantity": result.quantity,
                    "message": result.message,
                }
                if side == "BUY":
                    summary["buy_results"].append(result_dict)
                elif side == "SELL":
                    summary["sell_results"].append(result_dict)

    submitted_orders_count = len(summary["buy_results"]) + len(summary["sell_results"])
    summary["submitted_orders_count"] = submitted_orders_count
    summary["failed_orders_count"] = max(0, expected_orders_count - submitted_orders_count)

    log_path = os.path.join(output_dir, "api_execution_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("API execution log saved: %s", log_path)

    if not dry_run and summary["failed_orders_count"] > 0:
        raise RuntimeError(
            "Order submission incomplete: "
            f"submitted={submitted_orders_count}/expected={expected_orders_count}. "
            f"See {log_path} for details."
        )

    return summary


# ---------------------------------------------------------------------------
# Summary files (backtest mode)
# ---------------------------------------------------------------------------


def save_summary_files(
    results: pd.DataFrame,
    metrics: dict,
    config: ProductionConfig,
    output_dir: str,
) -> None:
    results_path = os.path.join(output_dir, "daily_results.csv")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    summary_path = os.path.join(output_dir, "run_summary.json")

    results.to_csv(results_path, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    wealth = (1.0 + results["daily_return"]).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "samples": int(len(results)),
        "first_trade_date": str(results.index.min().date()),
        "last_trade_date": str(results.index.max().date()),
        "final_wealth": float(wealth.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "output_files": {
            "daily_results": results_path,
            "metrics": metrics_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Hist returns for risk (convenience re-export)
# ---------------------------------------------------------------------------

get_hist_returns_for_risk = _get_hist_returns_for_risk
log_decision_summary = _log_decision_summary
print_risk_report = _print_risk_report
print_text_orders = _print_text_orders
