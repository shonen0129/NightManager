"""Fractional Differentiation for return series preprocessing.

Implements the López de Prado (2018) binomial expansion approach to fractional
differencing, which preserves long memory while achieving stationarity.

Reference:
    López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 5.
    Hosking, J. R. M. (1981). "Fractional Differencing", Biometrika.

Key functions:
    compute_weights(d, threshold) — binomial expansion coefficients
    fractional_diff(series, d, threshold, window) — single series transform
    fractional_diff_df(df, d, threshold, window) — DataFrame batch transform
    adf_test(series) — Augmented Dickey-Fuller test (pure numpy)
    hurst_exponent(series) — R/S analysis Hurst exponent
    find_optimal_d(series, d_range, threshold) — grid search for best d
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: binomial expansion weights
# ---------------------------------------------------------------------------


def compute_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute fractional differencing weights via binomial expansion.

    The weights are: w_k = (-1)^k * C(d, k) = product_{i=0}^{k-1} (d - i) / k!

    Weights are truncated when |w_k| falls below *threshold*.

    Args:
        d: Differencing order (0 < d < 1 for fractional).
        threshold: Weight cutoff threshold (weights below this are dropped).

    Returns:
        1-D numpy array of weights, ordered from w_0 (newest) to w_n (oldest).
    """
    weights = [1.0]
    k = 1
    while True:
        # Recursive: w_k = w_{k-1} * (k-1-d) / k
        # This encodes (-1)^k * C(d, k) = product_{i=0}^{k-1} (i-d) / k!
        w = weights[-1] * (k - 1 - d) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        # Safety cap to prevent infinite loop
        if k > 10000:
            logger.warning("compute_weights: hit 10000 cap for d=%.3f", d)
            break
    return np.array(weights)


# ---------------------------------------------------------------------------
# Core: fractional differencing transform
# ---------------------------------------------------------------------------


def fractional_diff(
    series: pd.Series,
    d: float = 0.5,
    threshold: float = 1e-5,
    window: int = 100,
) -> pd.Series:
    """Apply fractional differencing to a single time series.

    Uses the expanding-window binomial filter from López de Prado (2018).
    For each point t, the transformed value is:

        fd(t) = sum_{k=0}^{L-1} w_k * x(t-k)

    where L = min(len(weights), t+1, window).

    When d=1.0 this reduces to simple first differencing (x_t - x_{t-1}).
    When d=0.0 this returns the original series unchanged.

    Args:
        series: Input time series (pd.Series with numeric values).
        d: Differencing order (0.0 = no differencing, 1.0 = full differencing).
        threshold: Weight cutoff for binomial expansion.
        window: Maximum lookback for the filter (limits computational cost).

    Returns:
        Transformed series of same length, with NaN for initial warmup period.
    """
    if d == 0.0:
        return series.copy()
    if d == 1.0:
        return series.diff()

    values = series.values.astype(float)
    n = len(values)
    weights = compute_weights(d, threshold)
    width = len(weights)

    result = np.full(n, np.nan)

    for i in range(n):
        # Determine how many weights we can apply
        lookback = min(width, i + 1, window)
        w_slice = weights[:lookback]
        # Normalize weights to sum to 1 (optional but improves stability)
        # We skip normalization to match López de Prado's formulation
        x_slice = values[i - lookback + 1 : i + 1][::-1]
        result[i] = np.dot(w_slice, x_slice)

    return pd.Series(result, index=series.index, name=series.name)


def fractional_diff_df(
    df: pd.DataFrame,
    d: float = 0.5,
    threshold: float = 1e-5,
    window: int = 100,
) -> pd.DataFrame:
    """Apply fractional differencing to each column of a DataFrame.

    Args:
        df: Input DataFrame where each column is a time series.
        d: Differencing order.
        threshold: Weight cutoff.
        window: Maximum lookback.

    Returns:
        DataFrame with same shape, each column fractionally differenced.
    """
    result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        result[col] = fractional_diff(df[col], d=d, threshold=threshold, window=window)
    return result


# ---------------------------------------------------------------------------
# Stationarity test: Augmented Dickey-Fuller (pure numpy implementation)
# ---------------------------------------------------------------------------


def adf_test(series: pd.Series | np.ndarray, max_lags: int = 1) -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    Regresses Δy_t on y_{t-1} + lagged differences + constant.
    Returns the t-statistic and approximate p-value.

    This is a simplified implementation that does not require statsmodels.
    For critical values, we use the MacKinnon (1994) approximate values.

    Args:
        series: Input series (NaNs are dropped).
        max_lags: Number of lagged difference terms to include.

    Returns:
        Dict with keys: 'statistic', 'p_value', 'is_stationary'.
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 10:
        return {"statistic": np.nan, "p_value": 1.0, "is_stationary": False}

    # Check for constant series (no variance → trivially stationary)
    if np.std(y) < 1e-12:
        return {"statistic": -np.inf, "p_value": 0.0, "is_stationary": True}

    dy = np.diff(y)  # Δy_t
    y_lag = y[:-1]  # y_{t-1}

    # Build regression: Δy_t = α + β * y_{t-1} + Σ γ_i * Δy_{t-i} + ε
    X_cols = [np.ones(n - 1), y_lag]
    for lag in range(1, max_lags + 1):
        if lag < len(dy):
            lag_col = np.zeros(n - 1)
            lag_col[lag:] = dy[:-lag]
            X_cols.append(lag_col)
        else:
            X_cols.append(np.zeros(n - 1))

    X = np.column_stack(X_cols)
    y_dep = dy

    # OLS
    try:
        beta = np.linalg.lstsq(X, y_dep, rcond=None)[0]
        resid = y_dep - X @ beta
        n_obs = len(y_dep)
        k = X.shape[1]
        sigma2 = np.sum(resid**2) / (n_obs - k)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        t_stat = beta[1] / np.sqrt(cov[1, 1])
        if not np.isfinite(t_stat):
            return {"statistic": np.nan, "p_value": 1.0, "is_stationary": False}
    except Exception:
        return {"statistic": np.nan, "p_value": 1.0, "is_stationary": False}

    # MacKinnon (1994) approximate critical values for ADF (constant, no trend)
    # 1%: -3.43, 5%: -2.86, 10%: -2.57
    # Approximate p-value via interpolation
    if t_stat < -3.43:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.01 + (0.05 - 0.01) * (-3.43 - t_stat) / (-3.43 + 2.86)
        p_value = min(p_value, 0.05)
    elif t_stat < -2.57:
        p_value = 0.05 + (0.10 - 0.05) * (-2.86 - t_stat) / (-2.86 + 2.57)
        p_value = min(p_value, 0.10)
    else:
        p_value = 0.10 + max(0, (-2.57 - t_stat)) / 10.0
        p_value = min(max(p_value, 0.10), 1.0)

    return {
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "is_stationary": bool(t_stat < -2.86),
    }


# ---------------------------------------------------------------------------
# Long memory: Hurst exponent via R/S analysis
# ---------------------------------------------------------------------------


def hurst_exponent(series: pd.Series | np.ndarray, max_lag: int = 100) -> float:
    """Estimate Hurst exponent using Rescaled Range (R/S) analysis.

    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk
    H > 0.5: trending (persistent)

    Args:
        series: Input series (NaNs are dropped).
        max_lag: Maximum lag for R/S computation.

    Returns:
        Hurst exponent estimate (float).
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 20:
        return np.nan

    max_lag = min(max_lag, n // 2)
    # Use power-of-2-ish lags for more robust R/S estimation
    lags = []
    lag = 2
    while lag <= max_lag:
        lags.append(lag)
        lag = int(lag * 1.5)
    lags = np.array(lags)

    rs_values = []
    lag_values = []
    for lag in lags:
        n_sub = n // lag
        if n_sub < 2:
            continue
        rs_list = []
        for i in range(n_sub):
            segment = y[i * lag : (i + 1) * lag]
            if len(segment) < 2:
                continue
            mean_seg = np.mean(segment)
            dev = segment - mean_seg
            cumdev = np.cumsum(dev)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(segment, ddof=1)
            if S > 1e-12 and R > 1e-12:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            lag_values.append(lag)

    if len(lag_values) < 3:
        return np.nan

    log_lags = np.log(np.array(lag_values, dtype=float))
    log_rs = np.log(np.array(rs_values, dtype=float))

    slope, _intercept, _r_value, _p_value, _std_err = sp_stats.linregress(log_lags, log_rs)
    return float(slope)


# ---------------------------------------------------------------------------
# Optimal d search
# ---------------------------------------------------------------------------


def find_optimal_d(
    series: pd.Series,
    d_range: np.ndarray | None = None,
    threshold: float = 1e-5,
    window: int = 100,
    target_hurst: float = 0.45,
) -> dict:
    """Find optimal fractional differencing order d.

    Grid-searches over d values and selects the one that best balances
    stationarity (ADF p < 0.05) and long memory (Hurst close to target).

    Selection criteria:
    1. Filter for stationary series (ADF p < 0.05)
    2. Among stationary, pick the one with Hurst closest to target_hurst
    3. If none stationary, pick the one with lowest ADF p-value

    Args:
        series: Input time series.
        d_range: Array of d values to search (default: 0.1 to 1.0 step 0.1).
        threshold: Weight cutoff for binomial expansion.
        window: Maximum lookback.
        target_hurst: Target Hurst exponent (0.45 = mild persistence).

    Returns:
        Dict with: 'best_d', 'results' (list of per-d metrics), 'best_idx'.
    """
    if d_range is None:
        d_range = np.arange(0.1, 1.01, 0.1)

    results = []
    for d_val in d_range:
        fd = fractional_diff(series, d=float(d_val), threshold=threshold, window=window)
        fd_clean = fd.dropna()
        if len(fd_clean) < 20:
            results.append({
                "d": float(d_val),
                "adf_stat": np.nan,
                "adf_p": 1.0,
                "is_stationary": False,
                "hurst": np.nan,
            })
            continue
        adf = adf_test(fd_clean)
        h = hurst_exponent(fd_clean)
        results.append({
            "d": float(d_val),
            "adf_stat": adf["statistic"],
            "adf_p": adf["p_value"],
            "is_stationary": adf["is_stationary"],
            "hurst": h,
        })

    # Select best d
    stationary = [r for r in results if r["is_stationary"]]
    if stationary:
        best = min(stationary, key=lambda r: abs(r["hurst"] - target_hurst) if np.isfinite(r["hurst"]) else 999)
    else:
        best = min(results, key=lambda r: r["adf_p"])

    best_idx = next(i for i, r in enumerate(results) if r["d"] == best["d"])

    return {
        "best_d": best["d"],
        "results": results,
        "best_idx": best_idx,
    }


# ---------------------------------------------------------------------------
# Utility: apply fractional diff to df_exec US return columns
# ---------------------------------------------------------------------------


def apply_fractional_diff_to_df_exec(
    df_exec: pd.DataFrame,
    us_tickers: list[str],
    d: float = 0.5,
    threshold: float = 1e-5,
    window: int = 100,
) -> pd.DataFrame:
    """Apply fractional differencing to US return columns in df_exec.

    Transforms the `us_cc_{ticker}` columns using fractional differencing.
    Other columns (JP returns, gaps, betas) are left unchanged.

    Args:
        df_exec: Execution DataFrame.
        us_tickers: List of US ticker symbols.
        d: Differencing order.
        threshold: Weight cutoff.
        window: Maximum lookback.

    Returns:
        Modified copy of df_exec with fractionally differenced US returns.
    """
    df = df_exec.copy()
    us_cols = [f"us_cc_{tk}" for tk in us_tickers]
    us_df = df[us_cols]
    fd_df = fractional_diff_df(us_df, d=d, threshold=threshold, window=window)
    # Forward-fill the warmup NaNs with 0 to avoid losing data
    fd_df = fd_df.fillna(0.0)
    df[us_cols] = fd_df
    return df
