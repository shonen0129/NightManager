"""V2 execution bridge: connect the unified ProductionRunner to broker order submission.

This module bridges the V2 production pipeline (Residual-BLPX-RA v2) with the
existing broker execution layer (``execute_post_decision_flow``).  It builds the
unified ``df_exec`` once and runs ``ProductionRunner`` to obtain weights,
then converts those weights into a decision dict compatible with the broker
submission flow.

Flow:
  1. Load V2 config YAML
  2. Build ``df_exec`` (historical data + today's placeholders)
  3. Fetch JP open prices via broker API / CSV
  4. Run ``ProductionRunner`` to obtain w_final, scores, etc.
  5. Write V2 production files
  6. Convert V2 weights into a decision dict
  7. Execute order submission via the existing infrastructure
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.broker.tachibana.session_cache import load_open_prices_cache
from leadlag.config.paths import live as live_path
from leadlag.config.paths import results as results_path
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.execution.backtest import _load_df_exec
from leadlag.execution.broker_ops import (
    build_api_client,
    fetch_current_positions,
    resolve_wallet_capital,
)
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.output_ops import build_output_dir
from leadlag.execution.post_decision import execute_post_decision_flow
from leadlag.execution.pricing import resolve_daily_open_prices
from leadlag.execution.var_history import get_hist_returns_for_risk as _get_hist_returns_for_risk
from leadlag.reporting.production_v2_writer import write_production_files
from leadlag.runner.production import ProductionRunner, RunnerInputs

logger = logging.getLogger(__name__)


def _resolve_gap_dir(
    gap_input_dir: str | Path | None,
    app_config,
    root: Path,
) -> Path | None:
    """Resolve the V2 gap input directory from CLI or config."""
    gap_dir: Path | None = None
    if gap_input_dir is not None:
        gap_dir = Path(gap_input_dir)
        if not gap_dir.is_absolute():
            gap_dir = root / gap_dir
    else:
        default_gap = app_config.gap_distribution_dir or app_config.v2.gap_input_dir
        if default_gap:
            gap_dir = Path(default_gap)
            if not gap_dir.is_absolute():
                gap_dir = root / gap_dir

    if gap_dir is not None and not gap_dir.exists():
        logger.warning("Gap input dir not found: %s. Will use flat position.", gap_dir)
        gap_dir = None
    return gap_dir


def _resolve_trade_date(
    trade_date: str | None,
    live_dir: Path,
) -> str:
    """Resolve the trade date, including the ``latest`` shortcut.

    - ``latest`` reads ``live_dir/latest_weights.csv`` and uses its ``trade_date``.
    - If the CSV is missing, ``latest`` falls back to today (or the previous
      trading day if today is a market holiday).
    - ``None`` defaults to today.
    """
    if trade_date == "latest":
        latest_file = live_dir / "latest_weights.csv"
        if latest_file.exists():
            try:
                df_tmp = pd.read_csv(latest_file)
                raw_date = df_tmp.iloc[0]["trade_date"]
                if pd.isna(raw_date):
                    raise ValueError("trade_date is NaN")
                resolved = pd.to_datetime(str(raw_date)).strftime("%Y-%m-%d")
                logger.info("Resolved latest trade date from %s: %s", latest_file, resolved)
                return resolved
            except Exception as e:
                logger.error("Failed to parse latest trade_date from %s: %s", latest_file, e)
                raise
        else:
            from leadlag.core.market_calendar import is_market_closed, previous_trading_day

            today = pd.Timestamp.now().date()
            if is_market_closed(today):
                resolved = previous_trading_day(today).strftime("%Y-%m-%d")
                logger.warning(
                    "latest_weights.csv not found and today is a non-trading day; using %s",
                    resolved,
                )
            else:
                resolved = today.strftime("%Y-%m-%d")
                logger.warning("latest_weights.csv not found; using today: %s", resolved)
            return resolved

    if trade_date is None:
        resolved = pd.Timestamp.now().tz_localize(None).normalize().strftime("%Y-%m-%d")
        logger.info("No trade date provided; using today: %s", resolved)
        return resolved

    return trade_date


def _resolve_current_prices(
    app_config,
    api_client,
    jp_opens_csv: str | None,
    google_opens: bool,
) -> dict[str, float]:
    """Return a mapping of all JP tickers to current open prices."""
    manual_opens, topix_open = resolve_daily_open_prices(
        api_client=api_client,
        config=app_config.strategy,
        opens_csv=jp_opens_csv,
        use_google_opens=google_opens,
    )
    current_prices = dict(manual_opens)
    if topix_open is not None:
        current_prices[TOPIX_TICKER] = float(topix_open)
    return current_prices


def run_v2_decision(
    config_path: str | Path,
    gap_input_dir: str | Path | None = None,
    live_dir: str | Path = live_path("production_residual_blpx"),
    trade_date: str | None = None,
    api_enable: bool = False,
    api_dry_run: bool = False,
    capital_from_wallet: bool = False,
    text_output: bool = False,
    output_root: str = str(results_path()),
    jp_opens_csv: str | None = None,
    google_opens: bool = False,
    max_capital: float | None = None,
    api_url: str | None = None,
    api_token: str | None = None,
    run_tag: str | None = None,
    dry_run: bool = False,
) -> str:
    """Run V2 production decision and optionally submit orders via broker API.

    Args:
        config_path: Path to V2 production YAML config.
        gap_input_dir: Directory containing mu_gap/omega_gap .npy files.
        live_dir: Live output directory for V2 artifacts.
        trade_date: Trade date string (YYYY-MM-DD or ``latest``). Defaults to today.
        api_enable: If True, submit orders to broker API.
        api_dry_run: If True, simulate order submission.
        capital_from_wallet: If True, use wallet balance as max capital.
        text_output: If True, print text order summary.
        output_root: Root directory for decision output.
        jp_opens_csv: Path to JP opens CSV (fallback if API unavailable).
        google_opens: If True, use Google Sheets for JP opens.
        max_capital: Default max capital in JPY. Defaults to 350,000.0.
        api_url: Optional broker API URL override.
        api_token: Optional broker API token override.
        run_tag: Optional run tag for output directory.
        dry_run: If True, calculate weights but do not write files or submit orders.

    Returns:
        Path to the decision output CSV, or a dry-run summary path.
    """
    ROOT = Path(__file__).resolve().parents[3]

    # Resolve config
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    logger.info("Loading V2 config: %s", config_path)
    app_config = load_config_from_yaml(str(config_path))

    # Resolve live dir before trade date, because ``latest`` reads from it.
    live_path = Path(live_dir)
    if not live_path.is_absolute():
        live_path = ROOT / live_dir

    # Resolve trade date
    trade_date = _resolve_trade_date(trade_date, live_path)
    t_trade = pd.to_datetime(trade_date).normalize()
    logger.info("Trade date: %s", trade_date)

    # Resolve gap input dir
    gap_dir = _resolve_gap_dir(gap_input_dir, app_config, ROOT)
    if gap_dir is not None:
        logger.info("Using gap input dir: %s", gap_dir)

    if max_capital is None:
        max_capital = 350000.0  # default fallback

    # --- Step 1: Build df_exec (historical data + today's placeholders) ---
    logger.info("[1/5] Loading/building df_exec...")
    df_exec = _load_df_exec(app_config, data_source="cache")

    # --- Step 2: Fetch JP open prices ---
    api_client = None
    current_prices: dict[str, float] = {}
    try:
        if api_enable:
            api_client = build_api_client(
                api_url=api_url,
                api_token=api_token,
                api_dry_run=api_dry_run,
            )

            trade_date_str = t_trade.strftime("%Y%m%d")
            cached_opens = (
                load_open_prices_cache(trade_date_str)
                if app_config.broker_provider == "tachibana"
                else None
            )
            if cached_opens is not None:
                cached_prices, topix_open = cached_opens
                current_prices = dict(cached_prices)
                if topix_open is not None:
                    current_prices[TOPIX_TICKER] = float(topix_open)
                missing = [tk for tk in JP_TICKERS if tk not in current_prices]
                if not missing:
                    logger.info("[2/5] Using cached JP opens from gap distribution step.")
                else:
                    logger.info("[2/5] Cached opens incomplete (%d missing); fetching from API...", len(missing))
                    current_prices = _resolve_current_prices(
                        app_config, api_client, jp_opens_csv, google_opens
                    )
            else:
                logger.info("[2/5] Fetching JP opens...")
                current_prices = _resolve_current_prices(
                    app_config, api_client, jp_opens_csv, google_opens
                )

            if capital_from_wallet:
                max_capital = resolve_wallet_capital(api_client)
                logger.info("[CAPITAL] Using wallet balance: %s JPY", f"{max_capital:,.0f}")
        else:
            logger.info("[2/5] API disabled. Using dummy opens (1000 JPY for all tickers).")
            current_prices = {tk: 1000.0 for tk in JP_TICKERS}
    except Exception as e:
        logger.error("[2/5] Failed to fetch opens: %s", e)
        if api_client is not None:
            api_client.close()
        raise

    # --- Step 3: Generate V2 portfolio ---
    logger.info("[3/5] Generating V2 production portfolio...")
    runner = ProductionRunner(app_config)
    inputs = RunnerInputs(
        trade_date=trade_date,
        df_exec=df_exec,
        gap_input_dir=gap_dir,
        current_prices=current_prices,
        use_file_cache=True,
    )
    result = runner.run(inputs)

    write_production_files(trade_date, live_path, result, dry_run=dry_run)

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

    out_path: str
    if not dry_run:
        # --- Step 4: Build decision dict for execute_post_decision_flow ---
        logger.info("[4/5] Building decision dict for execution...")
        w_final = result["w_final"]
        scores = result["scores"]

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

        # --- Step 5: Execute post-decision flow ---
        logger.info("[5/5] Executing post-decision flow (risk checks, order submission)...")

        output_dir = build_output_dir(
            output_root,
            run_tag=run_tag,
            run_name="production_decision_v2",
        )

        current_positions = None
        if api_client is not None:
            try:
                current_positions = fetch_current_positions(api_client)
            except Exception as e:
                logger.warning("Failed to fetch current positions: %s. Will submit full target.", e)

        hist_returns = _get_hist_returns_for_risk(
            strategy=None,
            config=app_config.strategy,
            output_root=output_root,
            trade_date=t_trade,
            config_path=config_path,
            gap_input_dir=gap_dir,
        )

        out_path = execute_post_decision_flow(
            decision=decision,
            config=app_config.strategy,
            manual_opens=current_prices,
            max_capital=max_capital,
            hist_returns=hist_returns,
            output_dir=output_dir,
            api_client=api_client,
            text_output=text_output,
            current_positions=current_positions,
        )

        logger.info("V2 decision completed. Output: %s", out_path)
    else:
        logger.info("[DRY-RUN] Skipping order submission and file writes.")
        out_path = str(live_path)

    if api_client is not None:
        api_client.close()

    return out_path
