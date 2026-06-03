"""runner/fast.py — fast decision mode (no yfinance).

Uses precomputed strategy cache + broker API for US returns + JP opens.
Skips the heavy computation (eigen decomposition, full df_exec build)
for sub-second decision latency.

Typical workflow::

    # First-run (cache miss): builds cache from local etf_data.pkl
    # Subsequent runs: loads precomputed cache in <50ms

    from runner.fast import run_fast_decision

    result_path = run_fast_decision(
        config=config,
        output_dir=output_dir,
        output_root=output_root,
        api_client=api_client,
        api_dry_run=False,
        trade_date=t_trade,
        max_capital=10_000_000,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time as time_module
from typing import Optional

import numpy as np
import pandas as pd

from broker.base import BrokerClient
from data_loader import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from data_loader import load_jp_close_from_cache
from runner.config import ProductionConfig
from runner.helpers import (
    allocate_capital,
    auto_adjust_gross_exposure,
    build_strategy,
    log_decision_summary,
    print_risk_report,
    print_text_orders,
    resolve_wallet_capital,
    run_risk_checks,
    save_decision_output,
    submit_orders_via_api,
)
from services.cache_service import (
    is_strategy_cache_valid as _is_cache_valid,
    load_df_exec_from_local_cache as _load_df_exec_from_local_cache,
    read_cache_with_lock as _read_cache_with_lock,
    write_cache_with_lock as _write_cache_with_lock,
    exclusive_lock as _exclusive_lock,
)
from services.market_data import (
    compute_gap_from_jp_close as _compute_gap_from_jp_close,
    compute_topix_night_override as _compute_topix_night_override,
    fetch_opens_from_google as _fetch_opens_from_google,
    load_opens_from_csv as _load_opens_from_csv,
    normalize_to_tokyo_date as _normalize_to_tokyo_date,
    validate_manual_opens as _validate_manual_opens,
    validate_topix_open as _validate_topix_open,
    validate_us_returns_map as _validate_us_returns_map,
)
from strategy import LeadLagStrategy, PrecomputedLeadLagStrategy

logger = logging.getLogger(__name__)

# US returns cache TTL: 3 hours (refreshed if older)
FAST_US_RETURNS_CACHE_MAX_AGE_SECONDS: int = 3 * 60 * 60


# ---------------------------------------------------------------------------
# Precomputed cache builder
# ---------------------------------------------------------------------------


def build_precomputed_cache(
    config: ProductionConfig,
    df_exec: pd.DataFrame,
    cache_path: str,
) -> str:
    """Build and save the precomputed strategy cache to ``cache_path``.

    Args:
        config: ProductionConfig instance (v3_mode must be "static")
        df_exec: Execution DataFrame from preprocess_data()
        cache_path: Target .npz file path

    Returns:
        Path to the written cache file
    """
    if config.v3_mode != "static":
        raise ValueError(
            "FAST MODE currently supports v3_mode='static' only. "
            f"Got v3_mode={config.v3_mode!r}."
        )

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    strategy = LeadLagStrategy(
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
    )
    strategy.save_precomputed_cache(cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# US returns cache (broker API → JSON)
# ---------------------------------------------------------------------------


def fetch_us_returns_from_api(
    api_client: BrokerClient,
    output_root: str,
) -> np.ndarray:
    """Fetch US ETF returns from the broker API with local JSON cache.

    Returns cached values if they are from today and within
    ``FAST_US_RETURNS_CACHE_MAX_AGE_SECONDS``.

    Args:
        api_client: BrokerClient instance
        output_root: Root directory for the ``.cache/`` subdirectory

    Returns:
        numpy array of US ETF returns in ``US_TICKERS`` order
    """
    cache_dir = os.path.join(output_root, ".cache")
    us_cache = os.path.join(cache_dir, "us_returns.json")
    lock_path = us_cache + ".lock"
    os.makedirs(cache_dir, exist_ok=True)

    with _exclusive_lock(lock_path):
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        now_epoch = time_module.time()

        if os.path.exists(us_cache):
            try:
                with open(us_cache, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                fetched_at_epoch = float(cached.get("fetched_at_epoch", 0.0))
                cache_age = now_epoch - fetched_at_epoch if fetched_at_epoch > 0 else None

                if (
                    cached.get("date", "") == today
                    and cache_age is not None
                    and cache_age <= FAST_US_RETURNS_CACHE_MAX_AGE_SECONDS
                ):
                    cached_returns = _validate_us_returns_map(cached.get("returns", {}))
                    logger.info(
                        "[FAST MODE] Using cached US returns (age=%.0fs)...", cache_age
                    )
                    return np.array([cached_returns[tk] for tk in US_TICKERS], dtype=float)

                if cached.get("date", "") == today:
                    logger.info(
                        "[FAST MODE] US return cache is stale; refetching (max_age=%ss)",
                        FAST_US_RETURNS_CACHE_MAX_AGE_SECONDS,
                    )
            except Exception as e:
                logger.warning("[FAST MODE] Ignoring invalid US return cache and refetching: %s", e)

        logger.info("[FAST MODE] Fetching US ETF returns from broker API...")
        fetched = api_client.fetch_us_etf_returns(US_TICKERS)
        normalized = _validate_us_returns_map(fetched)

        cache_data = {
            "date": today,
            "fetched_at_epoch": now_epoch,
            "returns": normalized,
        }
        tmp_path = f"{us_cache}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
            os.replace(tmp_path, us_cache)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return np.array([normalized[tk] for tk in US_TICKERS], dtype=float)


# ---------------------------------------------------------------------------
# JP opens fetcher for fast mode
# ---------------------------------------------------------------------------


def fetch_jp_opens_for_fast_mode(
    api_client: BrokerClient,
    config: ProductionConfig,
    jp_opens_csv: Optional[str],
    google_opens: bool,
) -> tuple[dict, Optional[float]]:
    """Fetch JP open prices for fast mode with API → Google → CSV fallback.

    Args:
        api_client: BrokerClient instance
        config: ProductionConfig instance (used for signal_mode check)
        jp_opens_csv: Optional path to CSV override
        google_opens: If True, use Google Finance instead of API

    Returns:
        Tuple of (manual_opens dict, topix_open float or None)
    """
    if jp_opens_csv is not None:
        manual_opens = _load_opens_from_csv(jp_opens_csv)
    elif google_opens:
        logger.info("  Fetching from Google Finance...")
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]
        manual_opens = _fetch_opens_from_google(tickers=tickers_for_opens)
    else:
        logger.info("  Fetching from kabu API...")
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]
        manual_opens = api_client.fetch_jp_open_prices(
            tickers_for_opens, allow_missing=True
        )
        missing = [tk for tk in tickers_for_opens if tk not in manual_opens]
        if missing:
            logger.warning(
                "  Falling back to Google Finance for %d ticker(s): %s",
                len(missing),
                ", ".join(missing),
            )
            google_fetched = _fetch_opens_from_google(tickers=missing, allow_missing=True)
            manual_opens.update(google_fetched)
            missing_jp = [tk for tk in JP_TICKERS if tk not in manual_opens]
            if missing_jp:
                raise ValueError(
                    "Missing open prices after API + Google fallback: "
                    + ", ".join(missing_jp)
                )

    _validate_manual_opens(manual_opens)
    topix_open = None
    if config.signal_mode == "gap_residual":
        topix_open = _validate_topix_open(manual_opens)

    return manual_opens, topix_open


# ---------------------------------------------------------------------------
# Core fast decision runner
# ---------------------------------------------------------------------------


def run_decision_fast(
    config: ProductionConfig,
    cache_path: str,
    trade_date: pd.Timestamp,
    manual_opens: dict,
    gap_override: np.ndarray,
    topix_night_override: Optional[float],
    us_returns_today: np.ndarray,
    max_capital: float,
    output_dir: str,
    output_root: str,
    api_client: Optional[BrokerClient] = None,
    api_dry_run: bool = False,
    text_output: bool = False,
) -> str:
    """Generate a trade decision using the precomputed strategy cache.

    Skips full df_exec rebuild and correlation matrix computation.
    Requires precomputed cache (built via build_precomputed_cache).

    Args:
        config: ProductionConfig instance
        cache_path: Path to precomputed strategy cache .npz
        trade_date: Today's trade date
        manual_opens: Dict of ticker → open price
        gap_override: Array of JP gap overrides
        topix_night_override: TOPIX overnight return (or None)
        us_returns_today: Array of today's US ETF returns
        max_capital: Equity capital for sizing
        output_dir: Directory for output artifacts
        output_root: Root directory for risk caches
        api_client: Optional BrokerClient for order submission
        api_dry_run: If True, simulate without submission
        text_output: If True, print orders in text format

    Returns:
        Path to the decision output CSV
    """
    logger.info("[FAST MODE] Loading precomputed strategy cache...")
    precomputed = PrecomputedLeadLagStrategy.load_precomputed_strategy(
        cache_path,
        gap_override=gap_override,
    )
    logger.info("[FAST MODE] Cache loaded successfully")

    logger.info("[FAST MODE] Generating trade decision (fast path)...")
    decision = precomputed.generate_trade_decision(
        trade_date=trade_date,
        jp_gap_override=gap_override,
        us_returns_override=us_returns_today,
        topix_night_override=topix_night_override,
    )
    decision = auto_adjust_gross_exposure(decision, config)

    # Historical returns for VaR/ES (file-locked, local caches only)
    cache_dir = os.path.join(output_root, ".cache")
    returns_cache = os.path.join(cache_dir, "daily_returns.csv")
    hist_returns = _read_cache_with_lock(returns_cache, decision["trade_date"])
    if hist_returns is None:
        logger.info("No returns cache; rebuilding VaR/ES history from local cache...")
        df_exec = _load_df_exec_from_local_cache()
        strategy = build_strategy(config, df_exec)
        hist_results = strategy.run_backtest(start_date=config.start_date)
        _write_cache_with_lock(returns_cache, hist_results)
        hist_returns = pd.Series(
            hist_results.loc[hist_results.index < decision["trade_date"], "daily_return"]
        )

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

    log_decision_summary(decision_df, decision)

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
    print_risk_report(risk_report)
    if risk_report["is_blocked"]:
        raise RuntimeError(
            "Risk stop threshold breached; order submission blocked. "
            "See [RISK-STOP] logs above."
        )

    logger.info("[4/4] Writing decision artifact...")
    out_path = save_decision_output(decision_df, output_dir, decision["trade_date"])
    logger.info("Decision saved: %s", out_path)

    if text_output:
        print_text_orders(decision_df)

    if api_client is not None:
        submit_orders_via_api(
            decision_df=decision_df,
            api_client=api_client,
            output_dir=output_dir,
            dry_run=api_dry_run,
        )

    return out_path
