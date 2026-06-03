import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

# 年間営業日数
TRADING_DAYS_PER_YEAR = 245


def _extract_monthly_returns(daily_returns: pd.Series) -> pd.Series | None:
    """Aggregate daily returns into monthly returns when a datetime index is available."""
    series = pd.Series(daily_returns).dropna()
    if len(series) == 0:
        return None

    if isinstance(series.index, pd.DatetimeIndex):
        dt_index = series.index
    else:
        # Range/Int index cannot be reliably mapped to calendar months.
        if isinstance(series.index, pd.RangeIndex):
            return None
        parsed = pd.to_datetime(series.index, errors="coerce")
        if parsed.isna().any():
            return None
        dt_index = pd.DatetimeIndex(parsed)

    series = series.astype(float)
    series.index = dt_index
    monthly = (1.0 + series).groupby(series.index.to_period("M")).prod() - 1.0
    return monthly.astype(float)


def calculate_metrics(daily_returns, risk_free_rate=0.0):
    """
    Calculates AR, RISK, R/R, MDD based on daily returns.
    Assume 245 trading days per year for annualized metrics.

    Args:
        daily_returns: Series or array of daily returns
        risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 0.0)

    Returns:
        Dict with performance metrics
    """
    daily_series = pd.Series(daily_returns).dropna().astype(float)
    t_daily = len(daily_series)
    if t_daily == 0:
        return {}

    monthly_returns = _extract_monthly_returns(daily_series)

    if monthly_returns is not None and len(monthly_returns) > 0:
        t_months = len(monthly_returns)
        mu_m = float(np.mean(monthly_returns))
        ar = float(np.sum(monthly_returns) * 12.0 / t_months)

        if t_months > 1:
            risk = float(
                np.sqrt(12.0 / (t_months - 1) * np.sum((monthly_returns - mu_m) ** 2))
            )
            monthly_std = float(np.std(monthly_returns, ddof=1))
            monthly_rf = risk_free_rate / 12.0
            sharpe_ratio = (
                ((mu_m - monthly_rf) / monthly_std) * np.sqrt(12.0)
                if monthly_std > 0
                else np.nan
            )
        else:
            risk = np.nan
            sharpe_ratio = np.nan

        rr_ratio = ar / risk if np.isfinite(risk) and risk > 0 else np.nan
    else:
        # Fallback for callers that pass arrays without date index.
        ar = float(np.sum(daily_series) * TRADING_DAYS_PER_YEAR / t_daily)
        risk = float(np.std(daily_series, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        rr_ratio = ar / risk if risk > 0 else np.nan

        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        daily_std = float(np.std(daily_series, ddof=1))
        excess_return = float(np.mean(daily_series) - daily_rf)
        sharpe_ratio = (
            (excess_return / daily_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if daily_std > 0
            else np.nan
        )

    # Max Drawdown (MDD)
    # W_t = product(1 + R_t)
    W_t = (1 + daily_series).cumprod()
    running_max = W_t.cummax()
    drawdowns = (W_t / running_max) - 1.0
    # MDD is the maximum loss from peak (always <= 0, or 0 if no drawdown)
    mdd = min(0.0, drawdowns.min())

    return {
        "AR": ar,
        "RISK": risk,
        "R/R": rr_ratio,
        "Sharpe": sharpe_ratio,
        "MDD": mdd,
        "Total Return": W_t.iloc[-1] - 1.0,
    }


def generate_report(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Calculate metrics
    metrics = calculate_metrics(results_df["daily_return"])

    # Print metrics
    print("=== Backtest Performance Metrics ===")
    for k, v in metrics.items():
        if k in ["AR", "RISK", "MDD", "Total Return"]:
            print(f"{k}: {v*100:.2f}%")
        elif k == "Sharpe":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.2f}")

    # Save metrics to CSV
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
    running_max = W_t.cummax()
    drawdowns = (W_t / running_max) - 1.0
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
