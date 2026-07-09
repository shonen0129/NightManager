"""V2 execution bridge: connect generate_v2_production_portfolio to broker order submission.

This module bridges the V2 portfolio generation pipeline (mu_over_sigma + RuleD)
with the existing broker execution layer (execute_post_decision_flow).

Flow:
  1. Load V2 config YAML
  2. Call generate_v2_production_portfolio() to get w_final, scores, etc.
  3. Fetch JP open prices via broker API
  4. Convert V2 weights into a decision dict compatible with execute_post_decision_flow
  5. Execute order submission via the existing infrastructure
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.helpers import (
    build_api_client,
    build_output_dir,
    execute_post_decision_flow,
    fetch_current_positions,
    resolve_daily_open_prices,
    resolve_wallet_capital,
)
from leadlag.models.production_v2 import generate_v2_production_portfolio
from leadlag.reporting.production_v2_writer import write_production_files

logger = logging.getLogger(__name__)


def run_v2_decision(
    config_path: str | Path,
    gap_input_dir: str | Path | None = None,
    live_dir: str | Path = "live/production_residual_blpx",
    trade_date: str | None = None,
    api_enable: bool = False,
    api_dry_run: bool = False,
    capital_from_wallet: bool = False,
    text_output: bool = False,
    output_root: str = "results",
    jp_opens_csv: str | None = None,
    google_opens: bool = False,
) -> str:
    """Run V2 production decision and submit orders via broker API.

    Args:
        config_path: Path to V2 production YAML config.
        gap_input_dir: Directory containing mu_gap/omega_gap .npy files.
        live_dir: Live output directory for V2 artifacts.
        trade_date: Trade date string (YYYY-MM-DD). Defaults to today.
        api_enable: If True, submit orders to broker API.
        api_dry_run: If True, simulate order submission.
        capital_from_wallet: If True, use wallet balance as max capital.
        text_output: If True, print text order summary.
        output_root: Root directory for decision output.
        jp_opens_csv: Path to JP opens CSV (fallback if API unavailable).
        google_opens: If True, use Google Sheets for JP opens.

    Returns:
        Path to the decision output CSV.
    """
    ROOT = Path(__file__).resolve().parents[3]

    # Resolve config
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    logger.info("Loading V2 config: %s", config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve trade date
    if trade_date is None:
        trade_date = pd.Timestamp.now().tz_localize(None).normalize().strftime("%Y-%m-%d")
    t_trade = pd.to_datetime(trade_date).normalize()
    logger.info("Trade date: %s", trade_date)

    # Resolve gap input dir
    gap_dir: Path | None = None
    if gap_input_dir is not None:
        gap_dir = Path(gap_input_dir)
        if not gap_dir.is_absolute():
            gap_dir = ROOT / gap_dir
        if not gap_dir.exists():
            logger.warning("Gap input dir not found: %s. Will use flat position.", gap_dir)
            gap_dir = None
    else:
        # Try default from config
        default_gap = cfg.get("gap_distribution", {}).get("dir", "")
        if default_gap:
            gap_dir = ROOT / default_gap
            if gap_dir.exists():
                logger.info("Using gap_distribution.dir from config: %s", gap_dir)
            else:
                logger.warning("Config gap dir not found: %s. Will use flat position.", gap_dir)
                gap_dir = None

    # Resolve live dir
    live_path = Path(live_dir)
    if not live_path.is_absolute():
        live_path = ROOT / live_dir

    # --- Step 1: Generate V2 portfolio ---
    logger.info("[1/4] Generating V2 production portfolio...")
    result = generate_v2_production_portfolio(
        trade_date=trade_date,
        gap_input_dir=gap_dir,
        cfg=cfg,
    )

    # Write V2 production files (latest_weights.csv, audit, etc.)
    write_production_files(trade_date, live_path, result, dry_run=False)

    fallback_used = result["fallback"]["gap_data_missing"]
    if fallback_used:
        logger.warning("[V2] Gap data missing. Flat position (w_final=0) returned.")
    else:
        logger.info(
            "[V2] Portfolio OK. Bin=%s, Mult=%.2f, Gross=%.4f, IR=%.4f",
            result["pit_binning"]["assigned_bin"],
            result["pit_binning"]["multiplier"],
            float(np.sum(np.abs(result["w_final"]))),
            result["summary"]["predicted_portfolio_ir"],
        )

    # --- Step 2: Fetch JP open prices ---
    api_client = None
    manual_opens: dict[str, float] = {}
    max_capital = 350000.0  # default fallback

    app_config = load_config_from_yaml(str(config_path))

    try:
        if api_enable:
            api_client = build_api_client(
                api_url=None,
                api_token=None,
                api_dry_run=api_dry_run,
            )

            logger.info("[2/4] Fetching JP opens...")
            manual_opens, topix_open = resolve_daily_open_prices(
                api_client=api_client,
                config=app_config.strategy,
                opens_csv=jp_opens_csv,
                use_google_opens=google_opens,
            )

            # Resolve capital
            if capital_from_wallet:
                max_capital = resolve_wallet_capital(api_client)
                logger.info("[CAPITAL] Using wallet balance: %s JPY", f"{max_capital:,.0f}")
        else:
            logger.info("[2/4] API disabled. Using dummy opens (1000 JPY for all tickers).")
            manual_opens = {tk: 1000.0 for tk in JP_TICKERS}
    except Exception as e:
        logger.error("[2/4] Failed to fetch opens: %s", e)
        if api_client is not None:
            api_client.close()
        raise

    # --- Step 3: Build decision dict for execute_post_decision_flow ---
    logger.info("[3/4] Building decision dict for execution...")
    w_final = result["w_final"]
    scores = result["scores"]

    # Map weights to actions
    action = np.where(
        w_final > 1e-8, "LONG",
        np.where(w_final < -1e-8, "SHORT", "HOLD"),
    )

    decision = {
        "trade_date": t_trade,
        "tickers": JP_TICKERS,
        "signal": scores,
        "weight": w_final,
        "raw_weight": w_final,
        "scale": 1.0,
        "action": action,
        "sigma_s": 0.0,
        "dispersion_indicator": 0.0,
        "gross_before": float(np.sum(np.abs(w_final))),
        "gross_after": float(np.sum(np.abs(w_final))),
        "gross_adjusted": False,
        "gross_adjustment_factor": 1.0,
    }

    # --- Step 4: Execute post-decision flow ---
    logger.info("[4/4] Executing post-decision flow (risk checks, order submission)...")

    # app_config already loaded above

    output_dir = build_output_dir(
        output_root,
        run_tag=None,
        run_name="production_decision_v2",
    )

    # Fetch current positions for delta-based orders
    current_positions = None
    if api_client is not None:
        try:
            current_positions = fetch_current_positions(api_client)
        except Exception as e:
            logger.warning("Failed to fetch current positions: %s. Will submit full target.", e)

    # Historical returns for VaR/ES
    from leadlag.data.cache import read_cache_with_lock as _read_cache

    cache_dir = os.path.join(output_root, ".cache")
    returns_cache = os.path.join(cache_dir, "daily_returns.csv")
    hist_returns = _read_cache(returns_cache, t_trade)
    if hist_returns is None:
        logger.info("No returns cache; rebuilding VaR/ES history from local cache...")
        from leadlag.data.cache import load_df_exec_from_local_cache as _load_df_exec
        from leadlag.execution.helpers import build_strategy
        from leadlag.execution.backtester import BacktestEngine
        from leadlag.data.cache import write_cache_with_lock as _write_cache

        df_exec = _load_df_exec()
        strategy = build_strategy(app_config.strategy, df_exec)
        out_res = BacktestEngine.run_backtest(strategy, df_exec, start_date=app_config.strategy.start_date)
        hist_results = pd.DataFrame(
            {"daily_return": out_res["daily_returns"]},
            index=out_res["daily_returns"].index,
        )
        _write_cache(returns_cache, hist_results)
        hist_returns = pd.Series(
            hist_results.loc[hist_results.index < t_trade, "daily_return"]
        )

    out_path = execute_post_decision_flow(
        decision=decision,
        config=app_config.strategy,
        manual_opens=manual_opens,
        max_capital=max_capital,
        hist_returns=hist_returns,
        output_dir=output_dir,
        api_client=api_client,
        text_output=text_output,
        current_positions=current_positions,
    )

    logger.info("V2 decision completed. Output: %s", out_path)

    if api_client is not None:
        api_client.close()

    return out_path
