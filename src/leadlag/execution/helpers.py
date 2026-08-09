"""runner/helpers.py — shared utility functions for all runner modules.

Contains pure utility functions that are used by decision.py, fast.py,
close.py, and backtest.py to avoid code duplication.
"""

from __future__ import annotations

__all__ = [
    "build_output_dir",
    "build_api_client",
    "fetch_current_positions",
    "resolve_wallet_capital",
    "build_risk_config",
    "run_risk_checks",
    "auto_adjust_gross_exposure",
    "allocate_capital",
    "save_decision_output",
    "split_large_orders",
    "submit_orders_via_api",
    "save_summary_files",
    "execute_post_decision_flow",
    "resolve_daily_open_prices",
    "fetch_fill_prices",
    "save_position_snapshot",
    "save_wallet_snapshot",
    "save_daily_journal",
    "get_hist_returns_for_risk",
    "log_decision_summary",
    "print_risk_report",
    "print_text_orders",
]

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.core import allocator as domain_allocator
from leadlag.core.portfolio import adjust_gross_exposure, classify_actions
from leadlag.core.risk import evaluate_risk_checks
from leadlag.core.types import (
    RiskConfig,
)
from leadlag.data.cache import get_hist_returns_for_risk as _get_hist_returns_for_risk
from leadlag.execution.broker_ops import (
    build_api_client,
    fetch_current_positions,
    resolve_wallet_capital,
    split_large_orders,
    submit_orders_via_api,
)
from leadlag.execution.config import StrategyConfig as ProductionConfig
from leadlag.execution.pricing import fetch_fill_prices, resolve_daily_open_prices
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
# Risk checks
# ---------------------------------------------------------------------------


def build_risk_config(config: ProductionConfig) -> RiskConfig:
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
    side_leverage: float = domain_allocator.DEFAULT_SIDE_LEVERAGE,
) -> dict:
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


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------


def save_decision_output(
    decision_df: pd.DataFrame, output_dir: str, trade_date: pd.Timestamp
) -> str:
    out_path = os.path.join(output_dir, f"decision_{trade_date.strftime('%Y%m%d')}.csv")
    decision_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path



def save_summary_files(
    results: pd.DataFrame,
    metrics: dict,
    config: ProductionConfig | dict[str, Any] | Any,
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
    elif is_dataclass(config):
        cfg_dict = asdict(config)
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = dict(config)

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


def _prepare_decision_df(
    decision: dict,
    config: ProductionConfig,
    manual_opens: dict,
    max_capital: float,
) -> tuple[pd.DataFrame, dict]:
    """Apply gross exposure adjustment, map actions, allocate, and build decision_df.

    Returns:
        (decision_df, capital_alloc)
    """
    adjusted = auto_adjust_gross_exposure(decision, config)
    # Preserve the adjusted state in the same decision dict so downstream helpers see it.
    decision.update(adjusted)

    actions_mapped: list[str] = []
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
        side_leverage=getattr(config, "side_leverage", domain_allocator.DEFAULT_SIDE_LEVERAGE),
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

    return decision_df, capital_alloc


def _log_decision_allocations(
    decision_df: pd.DataFrame,
    capital_alloc: dict,
    max_capital: float,
    decision: dict,
) -> None:
    """Log the decision summary, gross adjustment, and target/allocated budgets."""
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


def _run_risk_check_and_print(
    decision: dict,
    decision_df: pd.DataFrame,
    max_capital: float,
    hist_returns: pd.Series,
    config: ProductionConfig,
) -> dict:
    """Compute allocated totals, run risk checks, print the report, and return it."""
    buy_mask = decision_df["action"] == "BUY"
    sell_mask = decision_df["action"] == "SELL"
    total_buy_allocated = float(decision_df.loc[buy_mask, "etf_amount"].sum())
    total_sell_allocated = float(decision_df.loc[sell_mask, "etf_amount"].sum())

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
    return risk_report


def _write_decision_output_and_submit(
    decision_df: pd.DataFrame,
    decision: dict,
    output_dir: str,
    text_output: bool,
    api_client: BrokerClient | None,
    current_positions: dict[str, int] | None,
) -> str:
    """Write decision CSV, optionally print text, submit orders, and save journal.

    Returns:
        Path to the decision output CSV.
    """
    logger.info("[4/4] Writing decision artifact...")
    out_path = save_decision_output(decision_df, output_dir, decision["trade_date"])
    logger.info("Decision saved: %s", out_path)

    if text_output:
        _print_text_orders(decision_df)

    if api_client is not None:
        order_summary = submit_orders_via_api(
            decision_df=decision_df,
            api_client=api_client,
            output_dir=output_dir,
            current_positions=current_positions,
        )

        # --- Trade journal: collect post-execution data for model improvement ---
        api_log_path = os.path.join(output_dir, "api_execution_log.json")

        # Fetch fill prices (約定価格) for slippage analysis
        from leadlag.broker.dry_run import DryRunBrokerClient

        if not isinstance(api_client, DryRunBrokerClient) and order_summary:
            all_results = order_summary.get("buy_results", []) + order_summary.get("sell_results", [])
            if all_results:
                fetch_fill_prices(api_client, all_results)
                # Re-save the enriched api_execution_log with fill data
                with open(api_log_path, "w", encoding="utf-8") as f:
                    json.dump(order_summary, f, ensure_ascii=False, indent=2)
                logger.info("[JOURNAL] Fill prices enriched in api_execution_log.json")

        # Save position snapshot (建単価・評価単価・評価損益)
        pos_snapshot_path = save_position_snapshot(api_client, output_dir, label="decision")

        # Save wallet snapshot (維持率・受入保証金)
        wallet_snapshot_path = save_wallet_snapshot(api_client, output_dir, label="decision")

        # Save daily journal index
        save_daily_journal(
            output_dir=output_dir,
            decision_csv_path=out_path,
            api_execution_log_path=api_log_path,
            position_snapshot_path=pos_snapshot_path,
            wallet_snapshot_path=wallet_snapshot_path,
        )

    return out_path


def execute_post_decision_flow(
    decision: dict,
    config: ProductionConfig,
    manual_opens: dict,
    max_capital: float,
    hist_returns: pd.Series,
    output_dir: str,
    api_client: BrokerClient | None = None,
    text_output: bool = False,
    current_positions: dict[str, int] | None = None,
) -> str:
    """Execute post-decision flow (gross adjustment, risk check, capital allocation, order submission, and output writing).

    The dry-run vs live behaviour is determined by the ``api_client`` type:
    pass a ``DryRunBrokerClient`` for simulated execution, or a
    ``KabuBrokerClient`` for live trading.

    Returns:
        Path to the decision output CSV.
    """
    decision_df, capital_alloc = _prepare_decision_df(
        decision, config, manual_opens, max_capital
    )

    _log_decision_allocations(decision_df, capital_alloc, max_capital, decision)

    _run_risk_check_and_print(decision, decision_df, max_capital, hist_returns, config)

    out_path = _write_decision_output_and_submit(
        decision_df=decision_df,
        decision=decision,
        output_dir=output_dir,
        text_output=text_output,
        api_client=api_client,
        current_positions=current_positions,
    )

    return out_path

# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

get_hist_returns_for_risk = _get_hist_returns_for_risk
log_decision_summary = _log_decision_summary
print_risk_report = _print_risk_report
print_text_orders = _print_text_orders


# ---------------------------------------------------------------------------
# Trade journal — daily data collection for model improvement
# ---------------------------------------------------------------------------


def save_position_snapshot(
    api_client: BrokerClient,
    output_dir: str,
    *,
    label: str = "decision",
    date_str: str | None = None,
) -> str | None:
    """Save current position snapshot with entry/evaluation prices.

    Saves a JSON file with per-position details including:
      - ticker, side, quantity, entry_price (建単価)
      - evaluation_price (評価単価), unrealized_pnl (評価損益)
      - margin costs (順日歩, 逆日歩, 貸株料)

    Args:
        api_client: BrokerClient instance
        output_dir: Directory to save the snapshot file
        label: Label for the filename (e.g. 'decision', 'close')
        date_str: Optional date string (YYYYMMDD). Defaults to today.

    Returns:
        Path to the saved file, or None if no positions or error.
    """
    try:
        positions = api_client.get_positions()
    except Exception as e:
        logger.warning("Failed to fetch positions for snapshot: %s", e)
        return None

    if not positions:
        logger.info("[JOURNAL] No open positions for snapshot.")
        return None

    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "positions": [],
    }

    for pos in positions:
        extra = pos.extra or {}
        entry_price = pos.price
        eval_price = float(extra.get("sOrderHyoukaTanka", 0) or 0)
        unrealized_pnl = float(extra.get("sOrderGaisanHyoukaSoneki", 0) or 0)
        unrealized_pnl_pct = float(extra.get("sOrderGaisanHyoukaSonekiRitu", 0) or 0)

        snapshot["positions"].append({
            "ticker": pos.ticker,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": entry_price,
            "evaluation_price": eval_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "execution_id": pos.execution_id,
            "margin_trade_type": pos.margin_trade_type,
            "account_type": pos.account_type,
            "tategyoku_day": extra.get("sOrderTategyokuDay"),
            "tategyoku_kizitu_day": extra.get("sOrderTategyokuKizituDay"),
            "tategyoku_daikin": float(extra.get("sOrderTategyokuDaikin", 0) or 0),
            "tate_tesuryou": float(extra.get("sOrderTateTesuryou", 0) or 0),
            "jun_hibu": float(extra.get("sOrderZyunHibu", 0) or 0),
            "gyaku_hibu": float(extra.get("sOrderGyakuhibu", 0) or 0),
            "kasikaburyou": float(extra.get("sOrderKasikaburyou", 0) or 0),
            "hensai_kanou_suryou": extra.get("sOrderHensaiKanouSuryou"),
        })

    snapshot["position_count"] = len(snapshot["positions"])
    snapshot["total_unrealized_pnl"] = sum(p["unrealized_pnl"] for p in snapshot["positions"])

    filename_date = date_str or datetime.now().strftime('%Y%m%d')
    filename = f"positions_{label}_{filename_date}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    logger.info("[JOURNAL] Position snapshot saved: %s (%d positions, P&L=%s)",
                filepath, snapshot["position_count"],
                f"{snapshot['total_unrealized_pnl']:,.0f}")
    return filepath


def save_wallet_snapshot(
    api_client: BrokerClient,
    output_dir: str,
    *,
    label: str = "decision",
    date_str: str | None = None,
) -> str | None:
    """Save wallet/balance snapshot with margin details.

    Saves cash_available, margin_available, 受入保証金, 維持率, 追証フラグ.

    Args:
        api_client: BrokerClient instance
        output_dir: Directory to save the snapshot file
        label: Label for the filename
        date_str: Optional date string (YYYYMMDD). Defaults to today.

    Returns:
        Path to the saved file, or None on error.
    """
    try:
        wallet = api_client.get_wallet()
    except Exception as e:
        logger.warning("Failed to fetch wallet for snapshot: %s", e)
        return None

    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "cash_available": wallet.cash_available,
        "margin_available": wallet.margin_available,
        "ukeire_hosyoukin": wallet.extra.get("ukeire_hosyoukin"),
        "hosyoukin_yoryoku": wallet.extra.get("hosyoukin_yoryoku"),
        "hosyoukin_ritu": wallet.extra.get("hosyoukin_ritu"),
        "sHosyouKinritu": wallet.extra.get("sHosyouKinritu"),
        "sOisyouHasseiFlg": wallet.extra.get("sOisyouHasseiFlg"),
        "sTatekaekinHasseiFlg": wallet.extra.get("sTatekaekinHasseiFlg"),
    }

    filename_date = date_str or datetime.now().strftime('%Y%m%d')
    filename = f"wallet_{label}_{filename_date}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    logger.info("[JOURNAL] Wallet snapshot saved: %s (margin=%s JPY, 維持率=%s%%)",
                filepath,
                f"{wallet.margin_available:,.0f}",
                snapshot.get("hosyoukin_ritu", "N/A"))
    return filepath


def save_daily_journal(
    output_dir: str,
    decision_csv_path: str | None = None,
    api_execution_log_path: str | None = None,
    position_snapshot_path: str | None = None,
    wallet_snapshot_path: str | None = None,
    close_execution_log_path: str | None = None,
) -> str:
    """Save a daily journal index file that links all collected data.

    Creates a single JSON file per day that references all collected
    artifacts (decision, fills, positions, wallet, close) for easy
    retrospective analysis.

    Args:
        output_dir: Directory for the journal file
        decision_csv_path: Path to decision CSV
        api_execution_log_path: Path to API execution log JSON
        position_snapshot_path: Path to position snapshot JSON
        wallet_snapshot_path: Path to wallet snapshot JSON
        close_execution_log_path: Path to close execution log JSON

    Returns:
        Path to the journal index file.
    """
    journal_dir = os.path.join(os.path.dirname(output_dir), "trade_journal")
    os.makedirs(journal_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    journal = {
        "date": date_str,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {},
    }

    for label, path in [
        ("decision_csv", decision_csv_path),
        ("api_execution_log", api_execution_log_path),
        ("position_snapshot", position_snapshot_path),
        ("wallet_snapshot", wallet_snapshot_path),
        ("close_execution_log", close_execution_log_path),
    ]:
        if path and os.path.exists(path):
            journal["artifacts"][label] = path

    journal_path = os.path.join(journal_dir, f"journal_{date_str}.json")
    with open(journal_path, "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)
    logger.info("[JOURNAL] Daily journal saved: %s", journal_path)
    return journal_path

