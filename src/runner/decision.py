"""runner/decision.py — standard decision runner.

Provides ``run_decision()`` which:
1. Fetches JP open prices (API → Google → CSV)
2. Loads / downloads df_exec (fast path: npz cache; slow path: yfinance)
3. Runs LeadLagStrategy.generate_trade_decision()
4. Applies gross-exposure adjustment and VaR/ES risk checks
5. Allocates capital and optionally submits orders via BrokerClient

This is the ``--mode decision`` (non-fast) execution path.
"""

from __future__ import annotations

import logging
import time as time_module
from typing import Optional

import numpy as np
import pandas as pd

from broker.base import BrokerClient
from data_loader import (
    JP_TICKERS,
    TOPIX_TICKER,
    download_data,
    is_decision_cache_valid,
    load_decision_cache,
    load_jp_close_from_cache,
    preprocess_data,
    save_decision_cache,
)
from runner.config import ProductionConfig
from runner.helpers import (
    allocate_capital,
    auto_adjust_gross_exposure,
    build_api_client,
    build_output_dir,
    build_strategy,
    get_hist_returns_for_risk,
    log_decision_summary,
    print_risk_report,
    print_text_orders,
    resolve_wallet_capital,
    run_risk_checks,
    save_decision_output,
    submit_orders_via_api,
)
from services.market_data import (
    compute_gap_from_jp_close as _compute_gap_from_jp_close,
    compute_gap_override as _compute_gap_override,
    compute_topix_night_override as _compute_topix_night_override,
    fetch_opens_from_google as _fetch_opens_from_google,
    load_opens_from_csv as _load_opens_from_csv,
    normalize_to_tokyo_date as _normalize_to_tokyo_date,
    validate_manual_opens as _validate_manual_opens,
    validate_topix_open as _validate_topix_open,
)

logger = logging.getLogger(__name__)


def run_decision(
    start_date: str,
    output_root: str,
    run_tag: Optional[str],
    trade_date: Optional[str],
    opens_csv: Optional[str],
    max_capital: float,
    api_enable: bool = False,
    api_url: Optional[str] = None,
    api_token: Optional[str] = None,
    api_dry_run: bool = False,
    use_google_opens: bool = False,
    text_output: bool = False,
    use_wallet_capital: bool = False,
) -> str:
    """Run the standard (non-fast) trade decision pipeline.

    Args:
        start_date: Historical start date for strategy fitting
        output_root: Root directory for run artifacts
        run_tag: Optional tag appended to the output directory name
        trade_date: Trade date override (defaults to today)
        opens_csv: Path to CSV with JP open prices (alternative to API/Google)
        max_capital: Equity capital budget in JPY
        api_enable: If True, connect to the broker API
        api_url: Broker API URL (or env default)
        api_token: Broker API token (or env default)
        api_dry_run: If True, simulate without actual order submission
        use_google_opens: If True, fetch JP opens from Google Finance
        text_output: If True, print order list in human-readable text
        use_wallet_capital: If True, override max_capital with wallet balance

    Returns:
        Path to the decision output CSV
    """
    config = ProductionConfig(start_date=start_date)
    output_dir = build_output_dir(output_root, run_tag, run_name="production_decision")

    # Build broker API client (if enabled)
    api_client: Optional[BrokerClient] = None
    if api_enable:
        api_client = build_api_client(api_url, api_token, api_dry_run)
        if use_wallet_capital:
            max_capital = resolve_wallet_capital(api_client)

    t_trade = (
        pd.to_datetime(trade_date).normalize()
        if trade_date is not None
        else pd.Timestamp.now().normalize()
    )

    # ---- [1/4] JP open prices ----
    if api_client is not None:
        logger.info("[1/4] Fetching JP opens from broker API...")
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]
        manual_opens = api_client.fetch_jp_open_prices(
            tickers_for_opens, allow_missing=True
        )
        missing = [tk for tk in tickers_for_opens if tk not in manual_opens]
        if missing:
            logger.warning(
                "[1/4] Falling back to Google Finance for %d ticker(s): %s",
                len(missing),
                ", ".join(missing),
            )
            google_opens = _fetch_opens_from_google(tickers=missing, allow_missing=True)
            manual_opens.update(google_opens)
            missing = [tk for tk in JP_TICKERS if tk not in manual_opens]
            if missing:
                raise ValueError(
                    "Missing open prices after API + Google fallback: "
                    + ", ".join(missing)
                )
        logger.info("  Resolved open prices for %d tickers", len(manual_opens))
    elif use_google_opens:
        logger.info("[1/4] Fetching JP current real-time prices from Google Finance...")
        tickers_for_opens = JP_TICKERS
        if config.signal_mode == "gap_residual":
            tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]
        manual_opens = _fetch_opens_from_google(tickers=tickers_for_opens)
    elif opens_csv is not None:
        logger.info("[1/4] Loading JP opens from CSV...")
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

    # ---- [2/4] Market data (fast path via decision cache, else yfinance) ----
    _t0 = time_module.perf_counter()
    if is_decision_cache_valid():
        logger.info("[2/4] Loading execution dataset from decision cache (fast path)...")
        df_exec = load_decision_cache()
        jp_close = load_jp_close_from_cache()
        jp_close.index = _normalize_to_tokyo_date(jp_close.index)
        gap_override = _compute_gap_from_jp_close(jp_close, t_trade, manual_opens)
        topix_night_override = None
        if topix_open is not None:
            topix_night_override = _compute_topix_night_override(
                jp_close, t_trade, topix_open
            )
    else:
        logger.info("[2/4] Downloading/loading market data (full path)...")
        data = download_data(beta_window=config.beta_window)
        df_exec = preprocess_data(data, beta_window=config.beta_window)
        try:
            save_decision_cache(df_exec)
        except Exception as e:
            logger.warning("Failed to save decision cache: %s", e)
        gap_override = _compute_gap_override(data, t_trade, manual_opens)
        jp_close = data["jp_close"].copy()
        jp_close.index = _normalize_to_tokyo_date(jp_close.index)
        topix_night_override = None
        if topix_open is not None:
            topix_night_override = _compute_topix_night_override(
                jp_close, t_trade, topix_open
            )

    _t1 = time_module.perf_counter()
    logger.info("  Data loading completed in %.3fs", _t1 - _t0)

    # Append synthetic row for today if trade_date is not yet in df_exec
    if t_trade not in df_exec.index:
        if len(df_exec) == 0:
            raise ValueError("df_exec is empty; cannot construct decision row")
        base_row = df_exec.iloc[-1].copy()
        base_row["sig_date"] = df_exec.iloc[-1]["sig_date"]
        if "topix_night_return" in df_exec.columns:
            base_row["topix_night_return"] = (
                topix_night_override
                if topix_night_override is not None
                else df_exec.iloc[-1]["topix_night_return"]
            )
        for col in [c for c in df_exec.columns if c.startswith("jp_beta_")]:
            base_row[col] = df_exec.iloc[-1][col]
        for k, tk in enumerate(JP_TICKERS):
            base_row[f"jp_gap_{tk}"] = gap_override[k]
            base_row[f"jp_oc_{tk}"] = 0.0  # unknown at decision time
        df_exec = pd.concat(
            [df_exec, pd.DataFrame([base_row], index=[t_trade])],
            axis=0,
        ).sort_index()

    # ---- [3/4] Strategy signal → risk → allocation ----
    logger.info("[3/4] Generating one-day trade decision...")
    strategy = build_strategy(config, df_exec)
    decision = strategy.generate_trade_decision(
        trade_date=t_trade,
        start_date=config.start_date,
        jp_gap_override=gap_override,
    )
    decision = auto_adjust_gross_exposure(decision, config)

    hist_returns = get_hist_returns_for_risk(
        strategy=strategy,
        config=config,
        output_root=output_root,
        trade_date=decision["trade_date"],
    )

    capital_alloc = allocate_capital(
        decision,
        manual_opens,
        max_capital,
        max_net_exposure=config.max_net_exposure,
    )

    # Compute limit prices based on yesterday's close and predicted returns
    limit_prices = []
    prev_dates = jp_close.index[jp_close.index < t_trade]
    prev_date = prev_dates[-1] if len(prev_dates) > 0 else jp_close.index[-1]
    sigma_s = float(decision["sigma_s"])
    gamma = float(config.gamma)
    for tk, sig, act in zip(decision["tickers"], decision["signal"], decision["action"]):
        prev_close = float(jp_close.loc[prev_date, tk])
        if act == "BUY":
            limit_price = prev_close * (1.0 + gamma * sig * sigma_s)
        elif act == "SELL":
            limit_price = prev_close * (1.0 - gamma * abs(sig) * sigma_s)
        else:
            limit_price = None

        if limit_price is not None:
            # Round limit price to appropriate TSE tick size
            if limit_price <= 1000.0:
                limit_price = round(limit_price, 1)
            elif limit_price <= 10000.0:
                limit_price = round(limit_price)
            elif limit_price <= 100000.0:
                limit_price = round(limit_price / 10.0) * 10
            else:
                limit_price = round(limit_price / 100.0) * 100
        limit_prices.append(limit_price)

    decision_df = pd.DataFrame(
        {
            "ticker": decision["tickers"],
            "open_price": [manual_opens[tk] for tk in decision["tickers"]],
            "signal": decision["signal"],
            "weight": decision["weight"],
            "action": decision["action"],
            "etf_amount": capital_alloc["allocated"],
            "quantity": capital_alloc["qty"],
            "limit_price": limit_prices,
        }
    )

    log_decision_summary(decision_df, decision)
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
        capital_alloc.get("gross_budget", capital_alloc["buy_budget"] + capital_alloc["sell_budget"])
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
    print_risk_report(risk_report)
    if risk_report["is_blocked"]:
        raise RuntimeError(
            "Risk stop threshold breached; order submission blocked. "
            "See [RISK-STOP] logs above."
        )

    # ---- [4/4] Write artifacts & submit orders ----
    logger.info("[4/4] Writing decision artifact...")
    out_path = save_decision_output(decision_df, output_dir, decision["trade_date"])
    logger.info("Decision saved: %s", out_path)

    if text_output:
        print_text_orders(decision_df)

    try:
        if api_client is not None:
            submit_orders_via_api(
                decision_df=decision_df,
                api_client=api_client,
                output_dir=output_dir,
                dry_run=api_dry_run,
            )
    finally:
        if api_client is not None:
            api_client.close()

    return out_path
