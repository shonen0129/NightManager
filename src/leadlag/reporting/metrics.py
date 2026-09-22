"""Performance metrics computation and report generation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annual trading days
TRADING_DAYS_PER_YEAR = 245


@dataclass(frozen=True)
class MetricsSpec:
    """Explicit definition of the return series used for evaluation.

    Returns and daily cost components are decimal fractions of notional.  The
    spec keeps frequency, annualisation, and flat-day treatment together so a
    report cannot silently switch definitions because its index contains
    timestamps.
    """

    frequency: Literal["daily", "monthly"] = "daily"
    annualization_periods: int = TRADING_DAYS_PER_YEAR
    include_flat_days: bool = True
    return_unit: Literal["return_fraction"] = "return_fraction"
    cost_unit: Literal["return_fraction"] = "return_fraction"

    def __post_init__(self) -> None:
        if self.frequency not in {"daily", "monthly"}:
            raise ValueError("frequency must be 'daily' or 'monthly'")
        if int(self.annualization_periods) <= 0:
            raise ValueError("annualization_periods must be positive")
        if self.frequency == "monthly" and self.annualization_periods != 12:
            raise ValueError("monthly MetricsSpec requires annualization_periods=12")


def compute_drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Return drawdowns aligned to each return observation.

    The high-water mark starts at initial wealth 1.0, so a loss on the first
    observation is represented in the returned series as well as in MDD.
    """
    series = pd.Series(daily_returns).astype(float)
    wealth = (1.0 + series).cumprod()
    wealth_with_initial = np.concatenate(([1.0], wealth.to_numpy()))
    running_max = np.maximum.accumulate(wealth_with_initial)
    return pd.Series(
        wealth.to_numpy() / running_max[1:] - 1.0,
        index=series.index,
    )


def _extract_monthly_returns(daily_returns: pd.Series) -> pd.Series | None:
    """Aggregate daily returns into monthly returns when a datetime index is available."""
    series = pd.Series(daily_returns).dropna()
    if len(series) == 0:
        return None

    if isinstance(series.index, pd.DatetimeIndex):
        dt_index = series.index
    else:
        if isinstance(series.index, pd.RangeIndex):
            return None
        parsed = pd.to_datetime(series.index, errors="coerce")
        if parsed.isna().any():
            return None
        dt_index = pd.DatetimeIndex(parsed)

    series = series.astype(float)
    series.index = dt_index
    monthly = (1.0 + series).groupby(series.index.year * 12 + series.index.month).prod() - 1.0
    return monthly.astype(float)


def calculate_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    *,
    frequency: str = "daily",
    spec: MetricsSpec | None = None,
) -> dict[str, float]:
    """Calculate AR, RISK, R/R, MDD based on daily returns.

    The primary metric contract is daily observations and 245 trading days per
    year.  ``frequency="monthly"`` is available only for callers that
    explicitly request monthly aggregation; date-indexed input no longer
    silently changes the definition of Sharpe and annualized risk.
    """
    if spec is not None:
        if frequency != "daily" and frequency != spec.frequency:
            raise ValueError("frequency and spec.frequency disagree")
        effective_frequency: str = spec.frequency
        periods_per_year = spec.annualization_periods
        daily_series = pd.Series(daily_returns).dropna().astype(float)
        if not spec.include_flat_days:
            daily_series = daily_series.loc[~np.isclose(daily_series, 0.0)]
    else:
        effective_frequency = frequency
        periods_per_year = TRADING_DAYS_PER_YEAR
        daily_series = pd.Series(daily_returns).dropna().astype(float)
    t_daily = len(daily_series)
    if t_daily == 0:
        return {}

    if effective_frequency not in {"daily", "monthly"}:
        raise ValueError("frequency must be 'daily' or 'monthly'")
    monthly_returns = (
        _extract_monthly_returns(daily_series)
        if effective_frequency == "monthly"
        else None
    )

    if monthly_returns is not None and len(monthly_returns) > 0:
        t_months = len(monthly_returns)
        mu_m = float(np.mean(monthly_returns))
        total_monthly_wealth = float(np.prod(1.0 + monthly_returns))
        ar = (
            float(total_monthly_wealth ** (periods_per_year / t_months) - 1.0)
            if total_monthly_wealth > 0.0
            else float("nan")
        )

        if t_months > 1:
            risk = float(
                np.sqrt(periods_per_year / (t_months - 1) * np.sum((monthly_returns - mu_m) ** 2))
            )
            monthly_std = float(np.std(monthly_returns, ddof=1))
            monthly_rf = risk_free_rate / periods_per_year
            sharpe_ratio = (
                ((mu_m - monthly_rf) / monthly_std) * np.sqrt(periods_per_year)
                if monthly_std > 0
                else np.nan
            )
        else:
            risk = np.nan
            sharpe_ratio = np.nan

        rr_ratio = ar / risk if np.isfinite(risk) and risk > 0 else np.nan
    else:
        total_wealth = float(np.prod(1.0 + daily_series))
        ar = (
            float(total_wealth ** (periods_per_year / t_daily) - 1.0)
            if total_wealth > 0.0
            else float("nan")
        )
        risk = float(np.std(daily_series, ddof=1) * np.sqrt(periods_per_year))
        rr_ratio = ar / risk if risk > 0 else np.nan

        daily_rf = risk_free_rate / periods_per_year
        daily_std = float(np.std(daily_series, ddof=1))
        excess_return = float(np.mean(daily_series) - daily_rf)
        sharpe_ratio = (
            (excess_return / daily_std) * np.sqrt(periods_per_year)
            if daily_std > 0
            else np.nan
        )

    # Max Drawdown (MDD)
    # Include initial wealth in the high-water mark.  Otherwise a loss on the
    # first observation is invisible to maximum drawdown.
    W_t = (1.0 + daily_series).cumprod()
    drawdowns = compute_drawdown_series(daily_series)
    mdd = min(0.0, float(drawdowns.min()))

    return {
        "AR": ar,
        "RISK": risk,
        "R/R": rr_ratio,
        "Sharpe": sharpe_ratio,
        "MDD": mdd,
        "Total Return": W_t.iloc[-1] - 1.0,
    }


def generate_report(results_df: pd.DataFrame, output_dir: str) -> None:
    """Generate backtest performance reports and plots."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = calculate_metrics(results_df["daily_return"])

    print("=== Backtest Performance Metrics ===")
    for k, v in metrics.items():
        if k in ["AR", "RISK", "MDD", "Total Return"]:
            print(f"{k}: {v * 100:.2f}%")
        elif k == "Sharpe":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.2f}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    # Plot Cumulative Returns
    W_t = (1 + results_df["daily_return"]).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(W_t.index, W_t.values, label="Lead-Lag Strategy")
    plt.title("Cumulative Return (2015 - Present)")
    plt.ylabel("Cumulative Wealth (starting at 1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_return.png"))
    plt.close()

    # Plot Drawdowns
    drawdowns = compute_drawdown_series(results_df["daily_return"])
    plt.figure(figsize=(10, 6))
    plt.plot(drawdowns.index, drawdowns.values, color="red")
    plt.fill_between(drawdowns.index, drawdowns.values, 0, color="red", alpha=0.3)
    plt.title("Drawdown Profile")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drawdowns.png"))
    plt.close()

    logger.info(f"Report components saved to {output_dir}")
