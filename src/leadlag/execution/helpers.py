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

from leadlag.broker.base import BrokerClient
from leadlag.broker.factory import create_broker_from_args
from leadlag.core import allocator as domain_allocator
from leadlag.core.portfolio import adjust_gross_exposure, classify_actions
from leadlag.core.risk import evaluate_risk_checks
from leadlag.core.types import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    RiskConfig,
)
from leadlag.data.cache import get_hist_returns_for_risk as _get_hist_returns_for_risk
from leadlag.execution.config import StrategyConfig as ProductionConfig
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.sre import SectorRelativeEnsembleModel
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
    provider = app_cfg.broker_provider

    if provider == "tachibana" and not api_dry_run:
        tachi = app_cfg.tachibana
        final_api_url = api_url if api_url else tachi.api_url
        final_api_token = api_token if api_token else tachi.auth_id
        api_password = tachi.second_password
        margin_trade_type = tachi.margin_trade_type
        account_type = tachi.account_type
        request_timeout = tachi.request_timeout
        extra = {"private_key_path": tachi.private_key_path}
    else:
        kabu = app_cfg.kabu
        final_api_url = api_url if api_url else kabu.api_url
        final_api_token = api_token if api_token else kabu.api_token
        api_password = kabu.api_password or os.environ.get("KABU_API_PASSWORD", "")
        margin_trade_type = kabu.margin_trade_type
        account_type = kabu.account_type
        request_timeout = kabu.request_timeout
        extra = {}

    client = create_broker_from_args(
        api_url=final_api_url,
        api_token=final_api_token or None,
        api_password=api_password or None,
        dry_run=api_dry_run,
        margin_trade_type=margin_trade_type,
        account_type=account_type,
        request_timeout=request_timeout,
        extra=extra,
    )

    logger.info("[API] Checking API connectivity (provider=%s)...", provider)
    if not client.health_check():
        if api_dry_run:
            logger.warning("[API] Health check failed, continuing in dry-run mode...")
        else:
            raise RuntimeError(
                f"Failed to connect to broker API (provider={provider}). "
                "Verify API URL, token, and credentials are correct."
            )
    else:
        logger.info("[API] Connection successful")

    return client


def fetch_current_positions(api_client: BrokerClient) -> dict[str, int]:
    """Fetch current open positions and return as a signed-quantity dict.

    Returns:
        Dict mapping ticker → signed quantity (positive=long, negative=short).
    """
    positions = api_client.get_positions()
    current: dict[str, int] = {}
    for pos in positions:
        if pos.quantity <= 0:
            continue
        signed_qty = pos.quantity if pos.side == "BUY" else -pos.quantity
        current[pos.ticker] = current.get(pos.ticker, 0) + signed_qty
    logger.info("[POSITIONS] Current holdings: %s", current or "(none)")
    return current


def resolve_wallet_capital(api_client: BrokerClient) -> float:
    wallet = api_client.get_wallet()
    # Prefer 受入保証金 (deposited margin = equity base) for margin trading.
    # This value is stable regardless of overnight positions, unlike
    # cash_available (現物買付可能額) or margin_available (信用新規建可能額)
    # which are reduced by existing positions.
    ukeire = wallet.extra.get("ukeire_hosyoukin")
    if ukeire is not None and ukeire > 0:
        logger.info(
            "[CAPITAL] Using 受入保証金 (deposited margin) for sizing: %s JPY",
            f"{ukeire:,.0f}",
        )
        return float(ukeire)
    # Fallback for brokers without 受入保証金 (e.g. kabu)
    cash_available = float(wallet.cash_available)
    logger.info(
        "[CAPITAL] Using cash wallet balance for sizing: %s JPY",
        f"{cash_available:,.0f}",
    )
    return cash_available


# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------


def build_strategy(
    config: ProductionConfig,
    df_exec: pd.DataFrame | None = None,
) -> SectorRelativeEnsembleModel:
    """Factory function for compat wrap.

    df_exec is unused but kept for backward compatibility with runner scripts.
    """
    return SectorRelativeEnsembleModel(config)


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
    current_positions: dict[str, int] | None = None,
) -> dict:
    """Submit trade orders to the broker API, accounting for existing positions.

    When ``current_positions`` is provided, only the delta between target and
    current quantities is submitted.  This avoids re-ordering the full target
    when positions are already held from overnight carry-over.

    The dry-run vs live distinction is handled entirely by the BrokerClient
    implementation: ``DryRunBrokerClient`` simulates orders without sending
    them, while ``KabuBrokerClient`` submits to the real API.  This function
    does not need to know which variant is being used.
    """
    is_dry_run = type(api_client).__name__ == "DryRunBrokerClient"
    current = current_positions or {}

    # Compute delta quantities (target - current), then split into close vs new orders.
    # Close orders (返済) reduce existing positions; new orders (新規) open or increase positions.
    # When a delta crosses zero (e.g. LONG→SHORT), the portion that closes the existing
    # position is a close order and the remainder is a new order.
    close_orders: list[tuple[str, OrderSide, int]] = []
    new_orders: list[tuple[str, OrderSide, int]] = []
    for _, row in decision_df.iterrows():
        ticker = str(row["ticker"])
        target_qty = int(row["quantity"])
        # Target side: BUY → positive, SELL → negative
        if row["action"] == "BUY":
            target_signed = target_qty
        elif row["action"] == "SELL":
            target_signed = -target_qty
        else:
            target_signed = 0

        current_signed = current.get(ticker, 0)
        delta = target_signed - current_signed

        if delta == 0:
            continue

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        abs_delta = abs(delta)

        # Determine how much of the delta closes the existing position
        if current_signed > 0 and delta < 0:
            # Reducing LONG: close up to current_signed, rest is new SHORT
            close_qty = min(abs_delta, current_signed)
            new_qty = abs_delta - close_qty
        elif current_signed < 0 and delta > 0:
            # Reducing SHORT: close up to abs(current_signed), rest is new LONG
            close_qty = min(abs_delta, abs(current_signed))
            new_qty = abs_delta - close_qty
        else:
            # No existing position, or delta in same direction as current → all new
            close_qty = 0
            new_qty = abs_delta

        if close_qty > 0:
            close_orders.append((ticker, side, close_qty))
        if new_qty > 0:
            new_orders.append((ticker, side, new_qty))

    # Log position reconciliation
    if current:
        logger.info("[DELTA] Reconciling against %d existing position(s):", len(current))
        for _, row in decision_df.iterrows():
            ticker = str(row["ticker"])
            target_qty = int(row["quantity"])
            cur = current.get(ticker, 0)
            if row["action"] == "BUY":
                target_signed = target_qty
            elif row["action"] == "SELL":
                target_signed = -target_qty
            else:
                target_signed = 0
            delta = target_signed - cur
            if delta != 0:
                logger.info("  %s: target=%d, current=%d, delta=%d", ticker, target_signed, cur, delta)
    else:
        logger.info("[DELTA] No existing positions; submitting full target quantities")

    buy_count = sum(1 for _, side, _ in new_orders if side == OrderSide.BUY)
    sell_count = sum(1 for _, side, _ in new_orders if side == OrderSide.SELL)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dry_run": is_dry_run,
        "buy_orders_count": buy_count,
        "sell_orders_count": sell_count,
        "current_positions": current,
        "buy_results": [],
        "sell_results": [],
        "close_results": [],
    }

    # --- Phase 1: Submit close orders (返済) first to free margin ---
    close_order_requests = [
        OrderRequest(
            ticker=ticker,
            side=side,
            quantity=qty,
            order_type=OrderType.MARKET,
        )
        for ticker, side, qty in close_orders
    ]
    close_results_all: list[OrderResult] = []
    if close_order_requests:
        logger.info("[CLOSE PHASE] Submitting %d close (返済) orders first...", len(close_order_requests))
        close_results_all = api_client.submit_orders_batch(
            close_order_requests, delay_ms=250, is_close=True,
        )
        for result in close_results_all:
            logger.info(
                "  [CLOSE] %s: %d shares (Order ID: %s, Status: %s)",
                result.ticker,
                result.quantity,
                result.order_id,
                result.status.value,
            )
            summary["close_results"].append({
                "order_id": result.order_id,
                "status": result.status.value,
                "ticker": result.ticker,
                "side": result.side.value,
                "quantity": result.quantity,
                "message": result.message,
            })

    # --- Phase 2: Submit new orders (新規) after close orders ---
    new_order_requests = [
        OrderRequest(
            ticker=ticker,
            side=side,
            quantity=qty,
            order_type=OrderType.MARKET,
        )
        for ticker, side, qty in new_orders
    ]
    expected_orders_count = len(close_order_requests) + len(new_order_requests)
    summary["expected_orders_count"] = expected_orders_count

    if new_order_requests:
        logger.info("[NEW PHASE] Submitting %d new (新規) orders...", len(new_order_requests))
        results = api_client.submit_orders_batch(new_order_requests, delay_ms=250)
        for result in results:
            side = result.side.value
            logger.info(
                "  [%s] %s: %d shares (Order ID: %s, Status: %s)",
                side,
                result.ticker,
                result.quantity,
                result.order_id,
                result.status.value,
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

    submitted_orders_count = len(summary["buy_results"]) + len(summary["sell_results"]) + len(summary["close_results"])
    summary["submitted_orders_count"] = submitted_orders_count
    summary["failed_orders_count"] = max(0, expected_orders_count - submitted_orders_count)

    log_path = os.path.join(output_dir, "api_execution_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("API execution log saved: %s", log_path)

    if not is_dry_run and summary["failed_orders_count"] > 0:
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


def resolve_daily_open_prices(
    api_client: BrokerClient | None,
    config: ProductionConfig,
    opens_csv: str | None,
    use_google_opens: bool,
) -> tuple[dict[str, float], float | None]:
    """Fetch JP open prices with API -> Google -> CSV fallback mechanism.

    Used by both decision.py and fast.py.
    """
    from leadlag.data.market_data import (
        fetch_opens_from_google as _fetch_opens_from_google,
    )
    from leadlag.data.market_data import (
        load_opens_from_csv as _load_opens_from_csv,
    )
    from leadlag.data.market_data import (
        validate_manual_opens as _validate_manual_opens,
    )
    from leadlag.data.market_data import (
        validate_topix_open as _validate_topix_open,
    )
    from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER

    tickers_for_opens = JP_TICKERS
    if config.signal_mode == "gap_residual":
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]

    if api_client is not None:
        logger.info("Fetching JP opens from broker API...")
        manual_opens = api_client.fetch_open_prices(tickers_for_opens, allow_missing=True)
        missing = [tk for tk in tickers_for_opens if tk not in manual_opens]
        if missing:
            logger.warning(
                "Falling back to Google Finance for %d ticker(s): %s",
                len(missing),
                ", ".join(missing),
            )
            google_fetched = _fetch_opens_from_google(tickers=missing, allow_missing=True)
            manual_opens.update(google_fetched)
            missing_jp = [tk for tk in JP_TICKERS if tk not in manual_opens]
            if missing_jp:
                raise ValueError(
                    "Missing open prices after API + Google fallback: " + ", ".join(missing_jp)
                )
        logger.info("Resolved open prices for %d tickers", len(manual_opens))
    elif use_google_opens:
        logger.info("Fetching JP current real-time prices from Google Finance...")
        manual_opens = _fetch_opens_from_google(tickers=tickers_for_opens)
    elif opens_csv is not None:
        logger.info("Loading JP opens from CSV...")
        manual_opens = _load_opens_from_csv(opens_csv)
    else:
        raise ValueError(
            "--jp-opens-csv or --google-opens is required when API is not enabled. "
            "Either provide a CSV file, use --google-opens, or use --api-enable."
        )

    _validate_manual_opens(manual_opens)
    topix_open = None
    if config.signal_mode == "gap_residual":
        topix_open = _validate_topix_open(manual_opens)

    return manual_opens, topix_open


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


def fetch_fill_prices(
    api_client: BrokerClient,
    order_results: list[dict],
    *,
    wait_seconds: float = 3.0,
) -> list[dict]:
    """Fetch fill prices for submitted orders via CLMOrderListDetail.

    For each order in order_results (containing 'order_id'), queries the
    broker for fill details and enriches the dict with:
      - fill_price: 約定単価 (float or None)
      - fill_quantity: 約定株数 (int or None)
      - fill_status: 約定ステータス (str)
      - fill_detail: raw API response (dict)

    Args:
        api_client: BrokerClient with get_order_detail support
        order_results: List of order result dicts from submit_orders_via_api
        wait_seconds: Delay before fetching fills (allows exchange processing)

    Returns:
        Enriched order_results with fill information.
    """
    import time

    from leadlag.broker.tachibana.client import TachibanaBrokerClient

    if not isinstance(api_client, TachibanaBrokerClient):
        logger.debug("Fill price fetch not supported for this broker type, skipping.")
        return order_results

    if not order_results:
        return order_results

    eigyou_day = datetime.now().strftime("%Y%m%d")
    logger.info("[HEARTBEAT] Waiting %.1f seconds before fetching fill prices", wait_seconds)
    time.sleep(wait_seconds)

    for result in order_results:
        order_id = result.get("order_id", "")
        if not order_id or result.get("status") != "SUBMITTED":
            result["fill_price"] = None
            result["fill_quantity"] = None
            result["fill_status"] = "NOT_SUBMITTED"
            continue

        try:
            detail = api_client.get_order_detail(order_id, eigyou_day)
            fill_price_str = detail.get("sYakuzyouPrice", "0.0000")
            fill_qty_str = detail.get("sYakuzyouSuryou", "0")
            fill_status = detail.get("sOrderStatus", "")

            fill_price = float(fill_price_str) if fill_price_str and fill_price_str != "0.0000" else None
            fill_quantity = int(fill_qty_str) if fill_qty_str else None

            result["fill_price"] = fill_price
            result["fill_quantity"] = fill_quantity
            result["fill_status"] = fill_status
            result["fill_detail"] = {
                "sYakuzyouPrice": fill_price_str,
                "sYakuzyouSuryou": fill_qty_str,
                "sOrderStatus": fill_status,
                "sOrderYakuzyouStatus": detail.get("sOrderYakuzyouStatus"),
                "sBaiBaiDaikin": detail.get("sBaiBaiDaikin"),
                "sBaiBaiTesuryo": detail.get("sBaiBaiTesuryo"),
                "aYakuzyouSikkouList": detail.get("aYakuzyouSikkouList"),
                "aKessaiOrderTategyokuList": detail.get("aKessaiOrderTategyokuList"),
            }

            logger.info(
                "  [FILL] %s: %d shares @ %s (Order ID: %s, Status: %s)",
                result.get("ticker"),
                fill_quantity,
                fill_price,
                order_id,
                fill_status,
            )
        except Exception as e:
            logger.warning("Failed to fetch fill detail for order %s: %s", order_id, e)
            result["fill_price"] = None
            result["fill_quantity"] = None
            result["fill_status"] = "FETCH_ERROR"

    return order_results


def save_position_snapshot(
    api_client: BrokerClient,
    output_dir: str,
    *,
    label: str = "decision",
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

    filename = f"positions_{label}_{datetime.now().strftime('%Y%m%d')}.json"
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
) -> str | None:
    """Save wallet/balance snapshot with margin details.

    Saves cash_available, margin_available, 受入保証金, 維持率, 追証フラグ.

    Args:
        api_client: BrokerClient instance
        output_dir: Directory to save the snapshot file
        label: Label for the filename

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

    filename = f"wallet_{label}_{datetime.now().strftime('%Y%m%d')}.json"
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

