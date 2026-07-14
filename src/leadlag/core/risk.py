"""Risk management: VaR/ES computation and risk checks."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from leadlag.core.types import RiskConfig, RiskReport, VarEsResult

logger = logging.getLogger(__name__)


def compute_var_es(
    daily_returns: pd.Series,
    confidence: float = 0.99,
    window: int = 250,
    var_method: str = "historical",
) -> VarEsResult:
    """Compute one-day VaR/ES as positive loss ratios.

    Uses linear interpolation for VaR percentile calculation as per
    standard financial risk management practices (e.g., Basel guidelines).

    Args:
        daily_returns: Series of daily returns
        confidence: Confidence level (e.g., 0.99 for 99%)
        window: Lookback window in trading days
        var_method: "historical" for empirical quantile, "cornish_fisher"
            for Cornish-Fisher expansion (uses skewness & kurtosis to
            adjust the Gaussian quantile, more stable with small tail samples)

    Returns:
        VarEsResult with VaR/ES computation results
    """
    series = pd.Series(daily_returns).dropna()
    if len(series) < window:
        return VarEsResult(
            available=False,
            samples=int(len(series)),
            window=int(window),
            var_loss=np.nan,
            es_loss=np.nan,
            var_method=var_method,
        )

    sample = series.iloc[-window:].to_numpy(dtype=float)
    alpha = 1.0 - float(confidence)
    z_alpha = float(np.percentile(sample, alpha * 100.0, method="linear"))

    if var_method == "cornish_fisher":
        from scipy.stats import norm

        mu = float(np.mean(sample))
        sigma = float(np.std(sample, ddof=1))
        if sigma < 1e-12:
            sigma = 1e-12
        skew = float(np.mean(((sample - mu) / sigma) ** 3))
        kurt = float(np.mean(((sample - mu) / sigma) ** 4) - 3.0)

        z = norm.ppf(alpha)
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )
        q = mu + z_cf * sigma
    else:
        q = z_alpha

    # ES: average of all losses beyond VaR threshold
    tail = sample[sample <= q]
    es_raw = float(np.mean(tail)) if len(tail) > 0 else q

    return VarEsResult(
        available=True,
        samples=int(len(sample)),
        window=int(window),
        var_loss=max(0.0, -q),
        es_loss=max(0.0, -es_raw),
        var_quantile=float(q),
        tail_count=int(len(tail)),
        var_method=var_method,
    )


def evaluate_risk_checks(
    weights: np.ndarray,
    total_buy_allocated: float,
    total_sell_allocated: float,
    max_capital: float,
    hist_daily_returns: pd.Series,
    config: RiskConfig,
) -> RiskReport:
    """Evaluate policy risk checks and classify warning/stop breaches.

    Args:
        weights: Target weight array
        total_buy_allocated: Total JPY allocated to buy side
        total_sell_allocated: Total JPY allocated to sell side
        max_capital: Maximum available capital
        hist_daily_returns: Historical daily returns series
        config: Risk configuration

    Returns:
        RiskReport with risk check results
    """
    weights_arr = np.asarray(weights, dtype=float)
    target_net_exposure = float(np.sum(weights_arr))
    target_gross_exposure = float(np.sum(np.abs(weights_arr)))

    if float(max_capital) > 0:
        allocated_net_ratio = abs(total_buy_allocated - total_sell_allocated) / float(max_capital)
        allocated_gross_ratio = (total_buy_allocated + total_sell_allocated) / float(max_capital)
    else:
        allocated_net_ratio = 0.0
        allocated_gross_ratio = 0.0

    hist_series = pd.Series(hist_daily_returns).dropna().astype(float)

    var_es = compute_var_es(
        hist_daily_returns,
        confidence=config.var_confidence,
        window=config.var_window,
        var_method=getattr(config, "var_method", "historical"),
    )

    warning_breaches: list[str] = []
    stop_breaches: list[str] = []

    # Exposure checks
    if abs(target_net_exposure) > config.max_net_exposure:
        stop_breaches.append(
            f"target_net_exposure={target_net_exposure:.4f} exceeds ±{config.max_net_exposure:.4f}"
        )
    if target_gross_exposure > config.max_gross_exposure:
        stop_breaches.append(
            f"target_gross_exposure={target_gross_exposure:.4f} > {config.max_gross_exposure:.4f}"
        )

    if abs(allocated_net_ratio) > config.max_net_exposure:
        stop_breaches.append(
            f"allocated_net_ratio={allocated_net_ratio:.4f} exceeds ±{config.max_net_exposure:.4f}"
        )
    if allocated_gross_ratio > config.max_gross_exposure:
        stop_breaches.append(
            f"allocated_gross_ratio={allocated_gross_ratio:.4f} > {config.max_gross_exposure:.4f}"
        )

    # Daily loss checks
    if len(hist_series) > 0:
        latest_return = float(hist_series.iloc[-1])
        latest_loss = max(0.0, -latest_return)
        if latest_loss >= config.daily_loss_stop:
            stop_breaches.append(
                f"DailyLoss={latest_loss:.4%} >= stop {config.daily_loss_stop:.2%}"
            )
        elif latest_loss >= config.daily_loss_warning:
            warning_breaches.append(
                f"DailyLoss={latest_loss:.4%} >= warning {config.daily_loss_warning:.2%}"
            )

        # Monthly cumulative loss stop
        if isinstance(hist_series.index, pd.DatetimeIndex):
            dt_index = hist_series.index
        elif isinstance(hist_series.index, pd.RangeIndex):
            dt_index = None
        else:
            parsed = pd.to_datetime(hist_series.index, errors="coerce")
            dt_index = None if parsed.isna().any() else pd.DatetimeIndex(parsed)

        if dt_index is not None:
            series_dt = hist_series.copy()
            series_dt.index = dt_index
            month_period = series_dt.index[-1].to_period("M")
            month_slice = series_dt[series_dt.index.to_period("M") == month_period]
            month_return = float((1.0 + month_slice).prod() - 1.0)
            if month_return <= -config.monthly_loss_stop:
                stop_breaches.append(
                    f"MonthlyLoss={-month_return:.4%} >= stop {config.monthly_loss_stop:.2%}"
                )

    # VaR/ES checks
    if var_es.available:
        var_loss = float(var_es.var_loss)
        es_loss = float(var_es.es_loss)

        if var_loss >= config.var_stop:
            stop_breaches.append(f"VaR99={var_loss:.4%} >= stop {config.var_stop:.2%}")
        elif var_loss >= config.var_warning:
            warning_breaches.append(f"VaR99={var_loss:.4%} >= warning {config.var_warning:.2%}")

        if es_loss >= config.es_stop:
            stop_breaches.append(f"ES99={es_loss:.4%} >= stop {config.es_stop:.2%}")
        elif es_loss >= config.es_warning:
            warning_breaches.append(f"ES99={es_loss:.4%} >= warning {config.es_warning:.2%}")
    else:
        warning_breaches.append(
            f"VaR/ES skipped due to insufficient history ({var_es.samples}/{var_es.window})"
        )

    return RiskReport(
        target_net_exposure=target_net_exposure,
        target_gross_exposure=target_gross_exposure,
        allocated_net_ratio=float(allocated_net_ratio),
        allocated_gross_ratio=float(allocated_gross_ratio),
        var_es=var_es,
        warning_breaches=warning_breaches,
        stop_breaches=stop_breaches,
    )
