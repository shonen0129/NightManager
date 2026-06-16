"""runner/decision.py — standard decision runner.

Provides ``run_decision()`` which:
1. Fetches JP open prices (API → Google → CSV)
2. Loads / downloads df_exec (fast path: npz cache; slow path: yfinance)
3. Runs strategy signals and weights
4. Applies gross-exposure adjustment and VaR/ES risk checks
5. Allocates capital and optionally submits orders via BrokerClient
"""

from __future__ import annotations

import logging
import time as time_module

import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.data.cache import (
    is_decision_cache_valid,
    load_decision_cache,
    load_jp_close_from_cache,
    save_decision_cache,
)
from leadlag.data.fetcher import download_data
from leadlag.data.market_data import (
    compute_gap_from_jp_close as _compute_gap_from_jp_close,
)
from leadlag.data.market_data import (
    compute_gap_override as _compute_gap_override,
)
from leadlag.data.market_data import (
    compute_topix_night_override as _compute_topix_night_override,
)
from leadlag.data.market_data import (
    fetch_opens_from_google as _fetch_opens_from_google,
)
from leadlag.data.market_data import (
    load_opens_from_csv as _load_opens_from_csv,
)
from leadlag.data.market_data import (
    normalize_to_tokyo_date as _normalize_to_tokyo_date,
)
from leadlag.data.market_data import (
    validate_manual_opens as _validate_manual_opens,
)
from leadlag.data.market_data import (
    validate_topix_open as _validate_topix_open,
)
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.execution.config import StrategyConfig as ProductionConfig
from leadlag.execution.helpers import (
    build_api_client,
    build_output_dir,
    build_strategy,
    get_hist_returns_for_risk,
    resolve_wallet_capital,
    execute_post_decision_flow,
)

logger = logging.getLogger(__name__)


import numpy as np

def generate_daily_decision_results(model, df_exec, trade_date, current_weights=None):
    if trade_date == "latest":
        i = len(df_exec) - 1
        trade_date = df_exec.index[i]
    else:
        trade_date = pd.to_datetime(trade_date)
        i = df_exec.index.get_loc(trade_date)
    sig_date = df_exec["sig_date"].values[i]
    
    pred = model.predict_signals(df_exec)
    
    p0_sig = pred["p0_signals"].iloc[i].values
    p3_sig = pred["p3_signals"].iloc[i].values
    s_ens = pred["signals"].iloc[i].values
    
    z0 = model.normalize_signals(p0_sig, model.normalization_method)
    z3 = model.normalize_signals(p3_sig, model.normalization_method)
    
    w = model.build_weights(s_ens)
    ranks = pd.Series(s_ens).rank(ascending=False).values.astype(int)
    
    side = []
    for weight in w:
        if weight > 1e-10:
            side.append("LONG")
        elif weight < -1e-10:
            side.append("SHORT")
        else:
            side.append("NEUTRAL")
            
    sig_records = []
    for j, tk in enumerate(JP_TICKERS):
        rec = {
            "signal_date": sig_date,
            "trade_date": trade_date.strftime("%Y-%m-%d"),
            "ticker": tk,
            "production_signal": float(p0_sig[j]),
            "residual_signal": float(p3_sig[j]),
            "production_z": float(z0[j]),
            "residual_z": float(z3[j]),
            "ensemble_signal": float(s_ens[j]),
            "rank": int(ranks[j]),
            "side": side[j],
        }
        if getattr(model, "us_res_enabled", False):
            p4_sig = pred["p4_signals"].iloc[i].values
            z4 = model.normalize_signals(p4_sig, model.normalization_method)
            rec["us_residual_signal"] = float(p4_sig[j])
            rec["us_residual_z"] = float(z4[j])
        sig_records.append(rec)
    latest_signal_df = pd.DataFrame(sig_records)
    
    gross_exp = float(np.sum(np.abs(w)))
    net_exp = float(np.sum(w))
    
    weight_records = []
    for j, tk in enumerate(JP_TICKERS):
        weight_records.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "ticker": tk,
                "ensemble_signal": float(s_ens[j]),
                "weight": float(w[j]),
                "side": side[j],
                "gross_exposure": gross_exp,
                "net_exposure": net_exp,
            }
        )
    latest_weights_df = pd.DataFrame(weight_records)
    
    order_records = []
    for j, tk in enumerate(JP_TICKERS):
        curr_w = float(current_weights.get(tk, 0.0)) if current_weights is not None else 0.0
        target_w = float(w[j])
        delta_w = target_w - curr_w
        
        note = ""
        if side[j] == "LONG":
            note = "Buy to target weight" if delta_w > 0 else "Reduce long weight"
        elif side[j] == "SHORT":
            note = "Sell to target weight" if delta_w < 0 else "Cover short weight"
        else:
            note = "Close position" if abs(curr_w) > 1e-10 else "No position"
            
        order_records.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "ticker": tk,
                "current_weight": curr_w,
                "target_weight": target_w,
                "delta_weight": delta_w,
                "side": side[j],
                "note": note,
            }
        )
    latest_orders_df = pd.DataFrame(order_records)
    
    return {
        "signal_df": latest_signal_df,
        "weights_df": latest_weights_df,
        "orders_df": latest_orders_df,
        "trade_date": trade_date,
        "sig_date": sig_date,
    }


def run_decision(
    start_date: str,
    output_root: str,
    run_tag: str | None,
    trade_date: str | None,
    opens_csv: str | None,
    max_capital: float,
    api_enable: bool = False,
    api_url: str | None = None,
    api_token: str | None = None,
    api_dry_run: bool = False,
    use_google_opens: bool = False,
    text_output: bool = False,
    use_wallet_capital: bool = False,
) -> str:
    """Run the standard (non-fast) trade decision pipeline.

    Returns:
        Path to the decision output CSV
    """
    config = ProductionConfig(start_date=start_date)
    output_dir = build_output_dir(output_root, run_tag, run_name="production_decision")

    # Build broker API client (if enabled)
    api_client: BrokerClient | None = None
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
        manual_opens = api_client.fetch_open_prices(tickers_for_opens, allow_missing=True)
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
                    "Missing open prices after API + Google fallback: " + ", ".join(missing)
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
            topix_night_override = _compute_topix_night_override(jp_close, t_trade, topix_open)
    else:
        logger.info("[2/4] Downloading/loading market data (full path)...")
        data = download_data(beta_window=config.beta_window)
        df_exec = preprocess_data(data, beta_window=config.beta_window)
        try:
            save_decision_cache(df_exec)
        except Exception as e:
            logger.warning("Failed to save decision cache: %s", e)
        gap_override = _compute_gap_override(data, t_trade, manual_opens)
        topix_night_override = None
        if topix_open is not None:
            jp_close = data["jp_close"].copy()
            jp_close.index = _normalize_to_tokyo_date(jp_close.index)
            topix_night_override = _compute_topix_night_override(jp_close, t_trade, topix_open)

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
    model = build_strategy(config, df_exec)

    sre_run_res = generate_daily_decision_results(
        model=model,
        df_exec=df_exec,
        trade_date=t_trade,
        current_weights=None,
    )

    # Convert daily results back to decision format
    sre_signals_df = sre_run_res["signal_df"].set_index("ticker")
    sre_weights_df = sre_run_res["weights_df"].set_index("ticker")

    decision = {
        "trade_date": t_trade,
        "tickers": JP_TICKERS,
        "signal": sre_signals_df.loc[JP_TICKERS, "ensemble_signal"].values,
        "weight": sre_weights_df.loc[JP_TICKERS, "weight"].values,
        "action": sre_weights_df.loc[JP_TICKERS, "side"].values,  # "LONG", "SHORT", "NEUTRAL"
    }

    # Re-instantiate the model wrapper to get backtest for returns history
    hist_returns = get_hist_returns_for_risk(
        strategy=model,
        config=config,
        output_root=output_root,
        trade_date=decision["trade_date"],
    )

    out_path = execute_post_decision_flow(
        decision=decision,
        config=config,
        manual_opens=manual_opens,
        max_capital=max_capital,
        hist_returns=hist_returns,
        output_dir=output_dir,
        api_client=api_client,
        text_output=text_output,
    )

    return out_path
