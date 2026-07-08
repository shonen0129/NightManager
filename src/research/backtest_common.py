"""Shared utilities for backtest & experiment scripts.

Consolidates logic previously duplicated across:
  - scripts/backtest/run_overnight_holding_backtest.py
  - scripts/backtest/run_overnight_robustness_analysis.py
  - scripts/backtest/run_selective_overnight_backtest.py
  - scripts/backtest/run_production_backtest.py
  - scripts/experiments/*.py (30+ experiment scripts)

Includes:
  - CostParams: single source of research cost constants
  - load_execution_data / run_baseline_backtest: data + baseline setup
  - run_backtest_with_costs: BacktestEngine.run_backtest with standard cost params
  - prepare_target_and_gap_returns: target/gap DataFrame alignment
  - simulate_overnight_holding: per-asset alpha-mask overnight simulation
  - extended_metrics / compute_backtest_metrics: metrics from daily returns or results dict
  - compute_rank_ic: daily Spearman rank IC timeseries + summary
  - load_config: YAML config loader
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import StrategyConfig
from leadlag.execution.helpers import build_strategy
from leadlag.models.sre import compute_jp_target_returns
from leadlag.reporting.metrics import calculate_metrics

TRADING_DAYS = 245


@dataclass(frozen=True)
class CostParams:
    """Research cost parameters (defaults match configs/production/production.yaml costs)."""

    slippage_bps: float = 5.0
    buy_interest_annual: float = 0.025
    borrow_fee_annual: float = 0.0115
    reverse_fee_bps: float = 2.0

    @property
    def slip(self) -> float:
        return self.slippage_bps / 10000.0

    @property
    def financing_daily(self) -> float:
        return self.buy_interest_annual / 365.0

    @property
    def borrow_daily(self) -> float:
        return self.borrow_fee_annual / 365.0

    @property
    def reverse_daily(self) -> float:
        return self.reverse_fee_bps / 10000.0


def load_execution_data(
    beta_window: int = 60,
    beta_ewma_halflife: float | None = None,
    beta_shrinkage: float = 0.0,
    beta_winsor_sigma: float | None = None,
) -> pd.DataFrame:
    """Download raw data and build the aligned execution DataFrame (df_exec)."""
    raw_data = download_data(beta_window=beta_window)
    return preprocess_data(
        raw_data,
        beta_window=beta_window,
        beta_ewma_halflife=beta_ewma_halflife,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )


def run_baseline_backtest(
    df_exec: pd.DataFrame,
    start_date: str = "2015-01-05",
    slippage_bps: float = 5.0,
    **backtest_kwargs,
) -> dict:
    """Build the standard production strategy and run a baseline backtest.

    Returns the BacktestEngine result dict (weights, signals, daily_returns, ...).
    """
    config = StrategyConfig(start_date=start_date, slippage_bps=slippage_bps)
    model = build_strategy(config, df_exec)
    return BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=slippage_bps,
        **backtest_kwargs,
    )


def prepare_target_and_gap_returns(
    df_exec: pd.DataFrame,
    sim_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (target_returns_df, gap_returns_df) aligned to sim_dates.

    target: 9:10-to-close returns. gap: overnight gap(t) = open(t)/close(t-1) - 1.
    """
    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    target_df = pd.DataFrame(y_jp_target, index=df_exec.index, columns=JP_TICKERS).loc[sim_dates]

    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    gap_df = df_exec[gap_cols].copy()
    gap_df.columns = JP_TICKERS
    gap_df = gap_df.loc[sim_dates]
    return target_df, gap_df


# alpha_mask_fn(i, date, w_t, r_target, signals_row) -> np.ndarray of per-asset alpha
AlphaMaskFn = Callable[[int, pd.Timestamp, np.ndarray, np.ndarray, "np.ndarray | None"], np.ndarray]


def simulate_overnight_holding(
    weights_df: pd.DataFrame,
    target_returns_df: pd.DataFrame,
    gap_returns_df: pd.DataFrame,
    alpha: float | AlphaMaskFn,
    costs: CostParams | None = None,
    signals_df: pd.DataFrame | None = None,
) -> dict:
    """Simulate overnight position carry-over with realistic costs.

    Return decomposition:
        intraday  = w_t . r_910toClose(t)
        overnight = alpha_mask . (w_t * r_gap(t+1))

    Cost model (per asset j):
        (1 - alpha_j) fraction: full round-trip slippage
        alpha_j fraction:       rebalance slippage only
        held long:  financing (buy interest)
        held short: borrow fee + reverse fee (逆日歩)

    Args:
        weights_df: Daily portfolio weights (dates x tickers).
        target_returns_df: 9:10-to-close returns (dates x tickers).
        gap_returns_df: Overnight gap returns (dates x tickers).
        alpha: Scalar holding fraction in [0, 1], or a callable returning a
            per-asset alpha mask for each date (selective holding strategies).
        costs: Cost parameters (defaults to CostParams()).
        signals_df: Optional signals passed through to the alpha mask callable.

    Returns:
        Dict of daily Series: returns, intraday, overnight, cost components,
        turnover, exposures, hold_counts, equity_curve, drawdown.
    """
    costs = costs or CostParams()
    dates = weights_df.index
    n_assets = weights_df.shape[1]
    n_days = len(dates)

    # Convert to numpy arrays for faster access
    weights_arr = weights_df.values
    target_arr = target_returns_df.values
    gap_arr = gap_returns_df.values if gap_returns_df is not None else np.zeros((n_days, n_assets))
    signals_arr = signals_df.values if signals_df is not None else None

    records: dict[str, list[float]] = {
        "intraday": [], "overnight": [], "slip": [], "financing": [],
        "borrow": [], "reverse": [], "turnover": [], "gross": [],
        "long": [], "short": [], "hold_count": [],
    }

    w_prev = np.zeros(n_assets)
    for i in range(n_days):
        w_t = weights_arr[i]
        r_target = target_arr[i]

        if callable(alpha):
            sig_row = signals_arr[i] if signals_arr is not None else None
            alpha_mask = np.asarray(alpha(i, dates[i], w_t, r_target, sig_row), dtype=float)
        else:
            alpha_mask = np.full(n_assets, float(alpha))

        intraday = float(np.sum(w_t * r_target))
        gross_exp = float(np.sum(np.abs(w_t)))
        long_exp = float(np.sum(np.maximum(w_t, 0.0)))
        short_exp = float(np.sum(np.maximum(-w_t, 0.0)))
        turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)

        overnight = 0.0
        if np.any(alpha_mask > 0) and i < n_days - 1:
            r_gap_next = gap_arr[i + 1]
            overnight = float(np.sum(alpha_mask * w_t * r_gap_next))

        slip_cost = costs.slip * (
            2.0 * float(np.sum((1.0 - alpha_mask) * np.abs(w_t)))
            + float(np.sum(alpha_mask * np.abs(w_t - w_prev)) / 2.0)
        )
        held_long = float(np.sum(alpha_mask * np.maximum(w_t, 0.0)))
        held_short = float(np.sum(alpha_mask * np.maximum(-w_t, 0.0)))
        fin_cost = held_long * costs.financing_daily
        borrow_cost = held_short * costs.borrow_daily
        reverse_cost = held_short * costs.reverse_daily

        records["intraday"].append(intraday)
        records["overnight"].append(overnight)
        records["slip"].append(slip_cost)
        records["financing"].append(fin_cost)
        records["borrow"].append(borrow_cost)
        records["reverse"].append(reverse_cost)
        records["turnover"].append(turnover)
        records["gross"].append(gross_exp)
        records["long"].append(long_exp)
        records["short"].append(short_exp)
        records["hold_count"].append(float(np.sum(alpha_mask > 0)))

        w_prev = w_t

    series = {k: pd.Series(v, index=dates) for k, v in records.items()}
    daily_costs = series["slip"] + series["financing"] + series["borrow"] + series["reverse"]
    daily_returns = series["intraday"] + series["overnight"] - daily_costs
    wealth = (1.0 + daily_returns).cumprod()
    drawdown = (wealth / wealth.cummax()) - 1.0

    return {
        "daily_returns": daily_returns,
        "daily_intraday": series["intraday"],
        "daily_overnight": series["overnight"],
        "daily_costs": daily_costs,
        "daily_slip": series["slip"],
        "daily_financing": series["financing"],
        "daily_borrow": series["borrow"],
        "daily_reverse": series["reverse"],
        "daily_turnover": series["turnover"],
        "daily_gross": series["gross"],
        "daily_long": series["long"],
        "daily_short": series["short"],
        "daily_hold_counts": series["hold_count"],
        "equity_curve": wealth,
        "drawdown": drawdown,
    }


def extended_metrics(daily_returns: pd.Series) -> dict:
    """calculate_metrics + Hit Rate / Avg Daily Return / Calmar."""
    m = calculate_metrics(daily_returns)
    dr = daily_returns.dropna()
    if len(dr) > 0:
        m["Hit Rate"] = float((dr > 0).mean())
        m["Avg Daily Return"] = float(dr.mean())
        mdd = m.get("MDD", 0)
        m["Calmar"] = float(m.get("AR", 0) / abs(mdd)) if mdd != 0 else np.nan
    return m


def yearly_metrics(returns: pd.Series, label: str, min_days: int = 20) -> pd.DataFrame:
    """Compute per-year metrics for a return series."""
    rows = []
    for year in sorted(returns.index.year.unique()):
        yr_ret = returns[returns.index.year == year]
        if len(yr_ret) < min_days:
            continue
        m = calculate_metrics(yr_ret)
        m["Year"] = year
        m["Label"] = label
        m["Days"] = len(yr_ret)
        m["HitRate"] = float((yr_ret > 0).mean())
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment-focused utilities (scripts/experiments/)
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path) -> dict:
    """Load a YAML config file and return a dict."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_cached_df_exec() -> pd.DataFrame:
    """Load df_exec from local cache (etf_data.pkl via preprocessor)."""
    return load_df_exec_from_local_cache()


def run_backtest_with_costs(
    model,
    df_exec: pd.DataFrame,
    start_date: str = "2015-01-05",
    slippage_bps: float = 5.0,
    overnight_alpha_long: float = 0.75,
    overnight_alpha_short: float = 0.5,
    buy_interest_annual: float = 0.025,
    borrow_fee_annual: float = 0.0115,
    reverse_fee_bps: float = 2.0,
    **extra_kwargs,
) -> dict:
    """Run BacktestEngine.run_backtest with standard production cost parameters.

    This is the single call-site for the cost-parameter defaults that were
    duplicated across 30+ experiment scripts.
    """
    return BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=slippage_bps,
        overnight_alpha_long=overnight_alpha_long,
        overnight_alpha_short=overnight_alpha_short,
        buy_interest_annual=buy_interest_annual,
        borrow_fee_annual=borrow_fee_annual,
        reverse_fee_bps=reverse_fee_bps,
        **extra_kwargs,
    )


def compute_backtest_metrics(
    results: dict,
    name: str | None = None,
    include_rank_ic: bool = False,
    signals_df: pd.DataFrame | None = None,
    y_target: np.ndarray | None = None,
    sim_dates: pd.DatetimeIndex | None = None,
    start_idx: int = 0,
) -> dict:
    """Compute standard metrics from a BacktestEngine results dict.

    Returns a flat dict with keys:
        AR_net, AR_gross, Vol_net, Sharpe_net, Sharpe_gross, Sharpe_monthly,
        MDD, Turnover, GrossExp, n_days, elapsed_s
        (+ Mean_Rank_IC, ICIR, IC_positive_rate if include_rank_ic=True)
    """
    dr = results["daily_returns"]
    dr_gross = results.get("daily_returns_gross", dr)

    ar = float(dr.mean() * TRADING_DAYS)
    ar_gross = float(dr_gross.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    vol_gross = float(dr_gross.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_gross = ar_gross / vol_gross if vol_gross > 0 else np.nan

    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(results.get("daily_turnover", pd.Series()).mean()) if "daily_turnover" in results else np.nan
    gross_exp = float(results.get("daily_gross_exps", pd.Series()).mean()) if "daily_gross_exps" in results else np.nan

    monthly = (1.0 + dr).groupby(dr.index.year * 12 + dr.index.month).prod() - 1.0
    monthly_sharpe = float((monthly.mean() / monthly.std(ddof=1)) * np.sqrt(12.0)) if len(monthly) > 1 else np.nan

    m: dict = {
        "AR_net": ar,
        "AR_gross": ar_gross,
        "Vol_net": vol,
        "Sharpe_net": sharpe,
        "Sharpe_gross": sharpe_gross,
        "Sharpe_monthly": monthly_sharpe,
        "MDD": mdd,
        "Turnover": turnover,
        "GrossExp": gross_exp,
        "n_days": len(dr),
    }
    if name is not None:
        m["name"] = name

    if include_rank_ic and signals_df is not None and y_target is not None and sim_dates is not None:
        ic_df = compute_rank_ic(signals_df, y_target, sim_dates, start_idx)
        m["Mean_Rank_IC"] = float(ic_df["ic"].mean()) if len(ic_df) > 0 else np.nan
        std_ic = float(ic_df["ic"].std(ddof=1)) if len(ic_df) > 1 else np.nan
        m["ICIR"] = (m["Mean_Rank_IC"] / std_ic * np.sqrt(252)) if (std_ic and std_ic > 1e-8) else np.nan
        m["IC_positive_rate"] = float((ic_df["ic"] > 0).mean()) if len(ic_df) > 0 else np.nan

    return m


def compute_rank_ic(
    signals_df: pd.DataFrame,
    y_target: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    start_idx: int,
) -> pd.DataFrame:
    """Compute daily Spearman rank IC between signals and target returns.

    Returns a DataFrame with columns ['date', 'ic'] indexed by date.
    """
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)
    ic_list, ic_dates = [], []
    for i in range(start_idx, len(sim_dates)):
        date = sim_dates[i]
        if date not in signals_df.index:
            continue
        sig_t = signals_df.loc[date].values
        y_t = y_df.loc[date].values
        valid = ~(np.isnan(sig_t) | np.isnan(y_t))
        if valid.sum() >= 3:
            rho, _ = stats.spearmanr(sig_t[valid], y_t[valid])
            if np.isfinite(rho):
                ic_list.append(float(rho))
                ic_dates.append(date)
    return pd.DataFrame({"date": ic_dates, "ic": ic_list}).set_index("date")


def run_variant_timed(
    name: str,
    model,
    df_exec: pd.DataFrame,
    start_date: str = "2015-01-05",
    slippage_bps: float = 5.0,
    **backtest_kwargs,
) -> tuple[dict, dict, float]:
    """Run a single backtest variant, returning (metrics, results, elapsed_seconds).

    Convenience wrapper: run_backtest_with_costs + compute_backtest_metrics + timing.
    """
    t0 = time.perf_counter()
    results = run_backtest_with_costs(
        model, df_exec, start_date=start_date, slippage_bps=slippage_bps, **backtest_kwargs
    )
    elapsed = time.perf_counter() - t0
    metrics = compute_backtest_metrics(results, name=name)
    metrics["elapsed_s"] = elapsed
    return metrics, results, elapsed
