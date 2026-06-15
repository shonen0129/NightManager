"""runner/helpers.py — shared utility functions for all runner modules.

Contains pure utility functions that are used by decision.py, fast.py,
close.py, and backtest.py to avoid code duplication.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS

from leadlag.broker.base import BrokerClient
from leadlag.broker.factory import create_broker_from_args
from leadlag.core import allocator as domain_allocator
from leadlag.core.portfolio import adjust_gross_exposure, classify_actions
from leadlag.core.risk import evaluate_risk_checks
from leadlag.core.types import (
    OrderRequest,
    OrderSide,
    OrderType,
    RiskConfig,
)
from leadlag.data.cache import get_hist_returns_for_risk as _get_hist_returns_for_risk
from leadlag.execution.config import StrategyConfig as ProductionConfig
from leadlag.execution.config import load_config_from_yaml
from leadlag.reporting.formatter import (
    log_decision_summary as _log_decision_summary,
)
from leadlag.reporting.formatter import (
    print_risk_report as _print_risk_report,
)
from leadlag.reporting.formatter import (
    print_text_orders as _print_text_orders,
)
from leadlag.reporting.results_format import create_results_output_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def build_output_dir(
    output_root: str,
    run_tag: str | None,
    run_name: str,
) -> str:
    return create_results_output_dir(
        run_name=run_name,
        output_root=output_root,
        run_tag=run_tag,
        manifest_extra={"entry_point": "cli.py"},
    )


# ---------------------------------------------------------------------------
# Broker client
# ---------------------------------------------------------------------------


def build_api_client(
    api_url: str | None,
    api_token: str | None,
    api_dry_run: bool = False,
) -> BrokerClient:
    """Build and validate a BrokerClient.

    Delegates to ``broker.factory.create_broker_from_args``.
    """
    app_cfg = load_config_from_yaml()
    final_api_url = api_url if api_url else app_cfg.kabu.api_url
    final_api_token = api_token if api_token else app_cfg.kabu.api_token
    request_timeout = app_cfg.kabu.request_timeout

    api_password = app_cfg.kabu.api_password or os.environ.get("KABU_API_PASSWORD", "")
    margin_trade_type = app_cfg.kabu.margin_trade_type
    account_type = app_cfg.kabu.account_type

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
                "Failed to connect to kabuステーション API. Verify API URL and token are correct."
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


def build_strategy(config: ProductionConfig, df_exec: pd.DataFrame):
    """Factory function for compat wrap."""
    # SRE model can be built directly
    from leadlag.models.sre import SectorRelativeEnsembleModel

    return SectorRelativeEnsembleModel(config)


# ---------------------------------------------------------------------------
# Risk checks
# ---------------------------------------------------------------------------


def build_risk_config(config: ProductionConfig) -> RiskConfig:
    # Use config loader's default risk configurations matching ProductionConfig
    app_cfg = load_config_from_yaml()
    return RiskConfig(
        var_confidence=app_cfg.risk.var_confidence,
        var_window=app_cfg.risk.var_window,
        var_warning=app_cfg.risk.var_warning,
        var_stop=app_cfg.risk.var_stop,
        es_warning=app_cfg.risk.es_warning,
        es_stop=app_cfg.risk.es_stop,
        daily_loss_warning=app_cfg.risk.daily_loss_warning,
        daily_loss_stop=app_cfg.risk.daily_loss_stop,
        monthly_loss_stop=app_cfg.risk.monthly_loss_stop,
        max_net_exposure=app_cfg.risk.max_net_exposure,
        max_gross_exposure=app_cfg.risk.max_gross_exposure,
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
    app_cfg = load_config_from_yaml()
    result = adjust_gross_exposure(weights, app_cfg.risk.max_gross_exposure)

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
    """Submit trade orders to the broker API."""
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

    order_requests = [
        OrderRequest(
            ticker=str(row["ticker"]),
            side=OrderSide(str(row["action"])),
            quantity=int(row["quantity"]),
            order_type=OrderType.MARKET,
        )
        for _, row in valid_orders.iterrows()
    ]
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
            logger.info(
                "  [SIMULATED %s] %s: %d shares",
                req.side.value,
                req.ticker,
                req.quantity,
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
                    side,
                    result.ticker,
                    result.quantity,
                    result.order_id,
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
    if hasattr(config, "model_dump"):
        cfg_dict = config.model_dump()
    else:
        cfg_dict = asdict(config)

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "config": cfg_dict,
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


def execute_post_decision_flow(
    decision: dict,
    config: ProductionConfig,
    manual_opens: dict,
    max_capital: float,
    hist_returns: pd.Series,
    output_dir: str,
    api_client: BrokerClient | None = None,
    api_dry_run: bool = False,
    text_output: bool = False,
) -> str:
    """Execute post-decision flow (gross adjustment, risk check, capital allocation, order submission, and output writing).

    Returns:
        Path to the decision output CSV.
    """
    decision = auto_adjust_gross_exposure(decision, config)

    # Map sides to BUY, SELL, HOLD for backward compatibility with runner code
    actions_mapped = []
    for side in decision["action"]:
        if side in ("LONG", "BUY"):
            actions_mapped.append("BUY")
        elif side in ("SHORT", "SELL"):
            actions_mapped.append("SELL")
        else:
            actions_mapped.append("HOLD")
    decision["action"] = actions_mapped

    capital_alloc = allocate_capital(
        decision,
        manual_opens,
        max_capital,
        max_net_exposure=config.max_net_exposure,
    )

    decision_df = pd.DataFrame(
        {
            "ticker": decision["tickers"],
            "open_price": [manual_opens[tk] for tk in decision["tickers"]],
            "signal": decision["signal"],
            "weight": decision["weight"],
            "action": decision["action"],
            "etf_amount": capital_alloc["allocated"],
            "quantity": capital_alloc["qty"],
        }
    )

    _log_decision_summary(decision_df, decision)
    if decision.get("gross_adjusted", False):
        logger.info(
            "Gross auto-adjust applied: before=%.6f, after=%.6f, factor=%.6f",
            decision["gross_before"],
            decision["gross_after"],
            decision["gross_adjustment_factor"],
        )
    logger.info(
        "Equity capital (used for sizing): %s JPY "
        "(margin assumed: long+short notionals can exceed equity)",
        f"{max_capital:,.0f}",
    )

    buy_mask = decision_df["action"] == "BUY"
    sell_mask = decision_df["action"] == "SELL"
    total_buy_allocated = float(decision_df.loc[buy_mask, "etf_amount"].sum())
    total_sell_allocated = float(decision_df.loc[sell_mask, "etf_amount"].sum())
    total_gross_allocated = total_buy_allocated + total_sell_allocated
    total_net_allocated = total_buy_allocated - total_sell_allocated
    gross_budget = float(
        capital_alloc.get(
            "gross_budget", capital_alloc["buy_budget"] + capital_alloc["sell_budget"]
        )
    )

    logger.info("Target BUY budget: %s JPY", f"{capital_alloc['buy_budget']:,.0f}")
    logger.info("Target SELL budget: %s JPY", f"{capital_alloc['sell_budget']:,.0f}")
    logger.info("Allocated BUY notional: %s JPY", f"{total_buy_allocated:,.0f}")
    logger.info("Allocated SELL notional: %s JPY", f"{total_sell_allocated:,.0f}")
    logger.info("Allocated gross notional: %s JPY", f"{total_gross_allocated:,.0f}")
    logger.info("Allocated net notional: %s JPY", f"{total_net_allocated:,.0f}")
    logger.info("Unallocated gross budget: %s JPY", f"{gross_budget - total_gross_allocated:,.0f}")

    risk_report = run_risk_checks(
        decision=decision,
        total_buy_allocated=total_buy_allocated,
        total_sell_allocated=total_sell_allocated,
        max_capital=max_capital,
        hist_daily_returns=hist_returns,
        config=config,
    )
    _print_risk_report(risk_report)
    if risk_report["is_blocked"]:
        raise RuntimeError(
            "Risk stop threshold breached; order submission blocked. See [RISK-STOP] logs above."
        )

    logger.info("[4/4] Writing decision artifact...")
    out_path = save_decision_output(decision_df, output_dir, decision["trade_date"])
    logger.info("Decision saved: %s", out_path)

    if text_output:
        _print_text_orders(decision_df)

    if api_client is not None:
        submit_orders_via_api(
            decision_df=decision_df,
            api_client=api_client,
            output_dir=output_dir,
            dry_run=api_dry_run,
        )

    return out_path


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

get_hist_returns_for_risk = _get_hist_returns_for_risk
log_decision_summary = _log_decision_summary
print_risk_report = _print_risk_report
print_text_orders = _print_text_orders
