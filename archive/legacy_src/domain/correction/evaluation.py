"""domain.correction.evaluation – Performance evaluation and adoption gate.

This module provides:
- ``CostModel``               : Transaction-cost model (commission + spread + impact)
- ``compute_net_returns``     : Apply cost deductions to gross daily returns
- ``compute_signal_ic``       : Spearman IC between signal and realized returns
- ``deflated_sharpe_ratio``   : Bailey-de Prado DSR correction for multiple trials
- ``PerformanceMetrics``      : Container for all evaluation metrics
- ``evaluate_correction_adoption`` : Go/No-go gate comparing corrected vs linear

Adoption decision rule (strict)
---------------------------------
"ADOPT" is returned only when **all three** of the following hold in the OOS period:
  1. Net R/R (corrected) > Net R/R (linear)            [outperforms on risk-adj. basis]
  2. Net AR  (corrected) > Net AR  (linear)             [higher absolute return]
  3. DSR p-value (corrected) < significance_level       [survives multiple-testing correction]

In any other case the function returns "REJECT" with an explanation.
A backtest that looks good in-sample only does NOT qualify for adoption.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Transaction cost parameters (all expressed as rates per one-way trade).

    Attributes
    ----------
    commission_rate : float
        Broker commission per one-way trade (e.g. 0.0005 = 5 bps).
    spread_rate : float
        Half-spread cost per trade (e.g. 0.0003 = 3 bps).
    impact_rate : float
        Market-impact cost per trade (e.g. 0.0002 = 2 bps).
    """

    commission_rate: float = 0.0005
    spread_rate: float = 0.0003
    impact_rate: float = 0.0002

    @property
    def total_one_way(self) -> float:
        """Total one-way cost rate."""
        return self.commission_rate + self.spread_rate + self.impact_rate

    @property
    def round_trip(self) -> float:
        """Total round-trip cost rate (open + close)."""
        return 2.0 * self.total_one_way


def compute_net_returns(
    gross_daily: pd.Series,
    gross_exposure_series: Optional[pd.Series] = None,
    cost_model: Optional[CostModel] = None,
) -> pd.Series:
    """Deduct transaction costs from gross daily returns.

    The cost per day is ``round_trip_cost * gross_exposure / 2`` because
    we assume positions are opened and closed each day (intraday).
    If ``gross_exposure_series`` is not provided, a gross exposure of 2.0
    (fully invested long + short at 1× each) is assumed.

    Parameters
    ----------
    gross_daily : pd.Series
        Daily gross strategy returns.
    gross_exposure_series : pd.Series or None
        Daily gross exposure (|long| + |short| summed across sectors).
        Defaults to 2.0 every day if None.
    cost_model : CostModel or None
        Cost parameters.  Defaults to CostModel() if None.

    Returns
    -------
    net_daily : pd.Series
        Net returns after cost deduction.
    """
    if cost_model is None:
        cost_model = CostModel()

    if gross_exposure_series is None:
        gross_exposure = 2.0
    else:
        gross_exposure = gross_exposure_series.reindex(gross_daily.index).fillna(2.0)

    daily_cost = cost_model.round_trip * gross_exposure / 2.0
    return gross_daily - daily_cost


# ---------------------------------------------------------------------------
# Signal IC
# ---------------------------------------------------------------------------


def compute_signal_ic(
    signals: np.ndarray,
    returns: np.ndarray,
    method: Literal["spearman", "pearson"] = "spearman",
) -> float:
    """Compute daily cross-sectional IC and return the mean.

    Parameters
    ----------
    signals : np.ndarray, shape (T, N_J)
        Predicted signals for each day and sector.
    returns : np.ndarray, shape (T, N_J)
        Realized Open-to-Close returns.
    method : {"spearman", "pearson"}
        Correlation method.

    Returns
    -------
    float
        Mean IC across all days.
    """
    from scipy.stats import spearmanr, pearsonr

    T, N_J = signals.shape
    ic_values: list[float] = []

    for t in range(T):
        sig_t = signals[t]
        ret_t = returns[t]
        valid = np.isfinite(sig_t) & np.isfinite(ret_t)
        if valid.sum() < 3:
            continue
        if method == "spearman":
            corr, _ = spearmanr(sig_t[valid], ret_t[valid])
        else:
            corr, _ = pearsonr(sig_t[valid], ret_t[valid])
        if np.isfinite(corr):
            ic_values.append(float(corr))

    if len(ic_values) == 0:
        return float("nan")
    return float(np.mean(ic_values))


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & de Prado, 2014)
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    sharpe_hat: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> tuple[float, float]:
    """Compute the Deflated Sharpe Ratio and its p-value.

    Implements the correction proposed in:
    Bailey & de Prado (2014) "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality"

    Parameters
    ----------
    sharpe_hat : float
        Observed (annualized) Sharpe Ratio of the best strategy.
    n_trials : int
        Number of hyperparameter configurations tried (selection bias correction).
    n_obs : int
        Number of observations (days) in the evaluation period.
    skewness : float
        Return distribution skewness (0 = normal).
    kurtosis : float
        Return distribution kurtosis (3 = normal).

    Returns
    -------
    dsr : float
        Deflated Sharpe Ratio p-value (probability the strategy beats SR=0
        after correcting for the number of trials).
    sr_star : float
        Expected maximum SR under H0 (benchmark for comparison).
    """
    from scipy.stats import norm

    # Expected maximum SR under H0 (Eq. 8 in the paper)
    # Euler–Mascheroni constant ≈ 0.5772
    euler_mascheroni = 0.5772156649
    if n_trials <= 1:
        sr_star = 0.0
    else:
        e_max = (
            (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
            + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * math.e))
        )
        sr_star = e_max

    # Non-normality adjustment (Eq. 9)
    # Convert annualized SR to daily-level SR for the formula
    sr_daily = sharpe_hat / math.sqrt(245)
    sr_star_daily = sr_star / math.sqrt(245)

    dsr_num = (sr_daily - sr_star_daily) * math.sqrt(n_obs - 1)
    dsr_denom = math.sqrt(
        1.0 - skewness * sr_daily + (kurtosis - 1) / 4.0 * sr_daily ** 2
    )
    if dsr_denom <= 0:
        dsr_denom = 1e-8

    dsr_z = dsr_num / dsr_denom
    p_value = float(norm.cdf(dsr_z))

    return p_value, float(sr_star)


# ---------------------------------------------------------------------------
# Performance metrics container
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Container for all performance evaluation metrics."""

    label: str
    ar: float = float("nan")           # Annualized return (net)
    risk: float = float("nan")         # Annualized volatility
    rr: float = float("nan")           # Risk/Return ratio (= AR / RISK)
    mdd: float = float("nan")          # Maximum drawdown (always <= 0)
    sharpe: float = float("nan")       # Annualized Sharpe ratio (net, rf=0)
    ic_mean: float = float("nan")      # Mean cross-sectional Spearman IC
    dsr_pvalue: float = float("nan")   # DSR p-value (multiple-trial corrected)
    sr_star: float = float("nan")      # Expected max SR under H0
    n_obs: int = 0                     # Number of OOS days evaluated
    n_trials: int = 0                  # Hyperparameter configurations tried
    extra: dict = field(default_factory=dict)

    def summary_str(self) -> str:
        return (
            f"[{self.label}]\n"
            f"  AR={self.ar*100:.2f}%  RISK={self.risk*100:.2f}%  "
            f"R/R={self.rr:.2f}  MDD={self.mdd*100:.2f}%\n"
            f"  Sharpe={self.sharpe:.3f}  IC={self.ic_mean:.4f}  "
            f"DSR p={self.dsr_pvalue:.4f}  SR*={self.sr_star:.3f}  "
            f"n_obs={self.n_obs}  n_trials={self.n_trials}"
        )


def compute_performance_metrics(
    net_daily: pd.Series,
    signals: Optional[np.ndarray] = None,
    returns_panel: Optional[np.ndarray] = None,
    label: str = "strategy",
    n_trials: int = 1,
    trading_days_per_year: int = 245,
) -> PerformanceMetrics:
    """Compute all performance metrics from a net daily return series.

    Parameters
    ----------
    net_daily : pd.Series
        Net daily strategy returns (after cost deduction).
    signals : np.ndarray or None, shape (T, N_J)
        Panel of predicted signals (for IC calculation).
    returns_panel : np.ndarray or None, shape (T, N_J)
        Panel of realized OC returns (for IC calculation).
    label : str
        Human-readable label for this strategy variant.
    n_trials : int
        Number of hyperparameter configurations explored (for DSR).
    trading_days_per_year : int
        Annualization factor.

    Returns
    -------
    PerformanceMetrics
    """
    series = pd.Series(net_daily).dropna().astype(float)
    n_obs = len(series)
    m = PerformanceMetrics(label=label, n_obs=n_obs, n_trials=n_trials)

    if n_obs == 0:
        return m

    # Monthly aggregation for AR/RISK following the strategy's own convention
    if isinstance(series.index, pd.DatetimeIndex):
        monthly = (1.0 + series).groupby(series.index.to_period("M")).prod() - 1.0
        t_m = len(monthly)
        if t_m > 1:
            mu_m = float(monthly.mean())
            m.ar = float(monthly.sum() * 12.0 / t_m)
            m.risk = float(np.sqrt(12.0 / (t_m - 1) * np.sum((monthly - mu_m) ** 2)))
            m.rr = m.ar / m.risk if m.risk > 0 else float("nan")
            m.sharpe = (
                (mu_m / float(monthly.std(ddof=1))) * np.sqrt(12.0)
                if float(monthly.std(ddof=1)) > 0
                else float("nan")
            )
    else:
        # Fallback: daily annualization
        mu_d = float(series.mean())
        std_d = float(series.std(ddof=1))
        m.ar = mu_d * trading_days_per_year
        m.risk = std_d * math.sqrt(trading_days_per_year)
        m.rr = m.ar / m.risk if m.risk > 0 else float("nan")
        m.sharpe = mu_d / std_d * math.sqrt(trading_days_per_year) if std_d > 0 else float("nan")

    # MDD
    wealth = (1.0 + series).cumprod()
    m.mdd = float(min(0.0, (wealth / wealth.cummax() - 1.0).min()))

    # IC
    if signals is not None and returns_panel is not None:
        m.ic_mean = compute_signal_ic(signals, returns_panel)

    # DSR
    if np.isfinite(m.sharpe):
        m.dsr_pvalue, m.sr_star = deflated_sharpe_ratio(
            m.sharpe, n_trials=n_trials, n_obs=n_obs
        )

    return m


# ---------------------------------------------------------------------------
# Adoption decision gate
# ---------------------------------------------------------------------------


@dataclass
class AdoptionDecision:
    """Result of the correction adoption evaluation."""

    decision: Literal["ADOPT", "REJECT"]
    reason: str
    linear_metrics: PerformanceMetrics
    corrected_metrics: PerformanceMetrics

    def __str__(self) -> str:
        sep = "=" * 70
        return (
            f"\n{sep}\n"
            f"ADOPTION DECISION: {self.decision}\n"
            f"Reason: {self.reason}\n"
            f"{sep}\n"
            f"{self.linear_metrics.summary_str()}\n"
            f"{self.corrected_metrics.summary_str()}\n"
            f"{sep}"
        )


def evaluate_correction_adoption(
    linear_metrics: PerformanceMetrics,
    corrected_metrics: PerformanceMetrics,
    significance_level: float = 0.05,
) -> AdoptionDecision:
    """Determine whether the nonlinear correction layer should be adopted.

    The correction is adopted only if **all three** conditions hold:
      1. OOS Net R/R (corrected) > OOS Net R/R (linear)
      2. OOS Net AR  (corrected) > OOS Net AR  (linear)
      3. DSR p-value (corrected) < significance_level

    Parameters
    ----------
    linear_metrics : PerformanceMetrics
        Out-of-sample metrics for the linear-only baseline.
    corrected_metrics : PerformanceMetrics
        Out-of-sample metrics for the linear + correction strategy.
    significance_level : float
        Maximum acceptable DSR p-value (default: 0.05).

    Returns
    -------
    AdoptionDecision
        Contains decision ("ADOPT" or "REJECT") and a human-readable reason.
    """
    failures: list[str] = []

    # Condition 1: R/R comparison
    rr_lin = linear_metrics.rr
    rr_cor = corrected_metrics.rr
    if not (np.isfinite(rr_cor) and np.isfinite(rr_lin) and rr_cor > rr_lin):
        failures.append(
            f"Net R/R not improved: corrected={rr_cor:.3f} vs linear={rr_lin:.3f}"
        )

    # Condition 2: AR comparison
    ar_lin = linear_metrics.ar
    ar_cor = corrected_metrics.ar
    if not (np.isfinite(ar_cor) and np.isfinite(ar_lin) and ar_cor > ar_lin):
        failures.append(
            f"Net AR not improved: corrected={ar_cor*100:.2f}% vs linear={ar_lin*100:.2f}%"
        )

    # Condition 3: DSR significance
    dsr_p = corrected_metrics.dsr_pvalue
    if not (np.isfinite(dsr_p) and dsr_p < significance_level):
        failures.append(
            f"DSR p-value not significant: {dsr_p:.4f} >= {significance_level} "
            f"(n_trials={corrected_metrics.n_trials})"
        )

    if failures:
        reason = (
            "REJECT – the nonlinear correction does NOT outperform the linear benchmark "
            "on an OOS, net-of-cost, multiple-testing-corrected basis.\n"
            "Specific failures:\n" + "\n".join(f"  • {f}" for f in failures)
        )
        return AdoptionDecision(
            decision="REJECT",
            reason=reason,
            linear_metrics=linear_metrics,
            corrected_metrics=corrected_metrics,
        )

    reason = (
        f"ADOPT – corrected strategy outperforms on all three OOS gates:\n"
        f"  • Net R/R: {rr_cor:.3f} > {rr_lin:.3f}\n"
        f"  • Net AR:  {ar_cor*100:.2f}% > {ar_lin*100:.2f}%\n"
        f"  • DSR p:   {dsr_p:.4f} < {significance_level} (n_trials={corrected_metrics.n_trials})"
    )
    return AdoptionDecision(
        decision="ADOPT",
        reason=reason,
        linear_metrics=linear_metrics,
        corrected_metrics=corrected_metrics,
    )
