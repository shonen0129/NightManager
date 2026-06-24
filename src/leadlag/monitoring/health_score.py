"""Model Health Score — composite monitoring metric for strategy health.

Combines multiple diagnostic signals into a single 0-100 score:

Components (weighted):
  - IC decay (30%): Recent IC vs trailing IC. Score drops when predictive
    power degrades.
  - Turnover (20%): Penalizes excessive day-over-day weight changes that
    erode returns via transaction costs.
  - Gross exposure deviation (15%): Penalizes deviation from target gross.
  - Fallback rate (20%): Penalizes frequent v1/audit fallbacks.
  - Signal drift (15%): Penalizes large shifts in signal distribution
    statistics (mean/std) relative to trailing window.

Usage::

    from leadlag.monitoring.health_score import HealthScoreCalculator

    calc = HealthScoreCalculator()
    score = calc.compute(
        signals_df=signals,
        weights_df=weights,
        daily_returns=daily_rets,
        fallback_flags=fallback_series,
        target_gross=2.0,
    )
    print(f"Health Score: {score.score:.1f} ({score.grade})")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HealthGrade(str, Enum):
    EXCELLENT = "A"
    GOOD = "B"
    FAIR = "C"
    POOR = "D"
    CRITICAL = "F"


@dataclass
class ComponentScore:
    """Individual component score (0-100)."""
    name: str
    score: float
    weight: float
    detail: str = ""

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class HealthScore:
    """Composite health score result."""
    score: float
    grade: HealthGrade
    components: list[ComponentScore] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.score >= 70.0

    @property
    def is_critical(self) -> bool:
        return self.score < 40.0

    def summary(self) -> str:
        lines = [f"Health Score: {self.score:.1f} ({self.grade.value})"]
        for c in self.components:
            lines.append(f"  {c.name}: {c.score:.1f}/100 (w={c.weight:.0%}) — {c.detail}")
        return "\n".join(lines)


class HealthScoreCalculator:
    """Calculates composite model health score from strategy outputs.

    All component scores are normalized to 0-100 where 100 is best.
    """

    def __init__(
        self,
        ic_weight: float = 0.375,
        turnover_weight: float = 0.0,
        gross_dev_weight: float = 0.1875,
        fallback_weight: float = 0.25,
        signal_drift_weight: float = 0.1875,
        trailing_window: int = 60,
        recent_window: int = 20,
        target_turnover: float = 0.15,
        max_turnover: float = 0.50,
    ) -> None:
        self.weights = {
            "ic_decay": ic_weight,
            "turnover": turnover_weight,
            "gross_deviation": gross_dev_weight,
            "fallback_rate": fallback_weight,
            "signal_drift": signal_drift_weight,
        }
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.trailing_window = trailing_window
        self.recent_window = recent_window
        self.target_turnover = target_turnover
        self.max_turnover = max_turnover

    def compute(
        self,
        signals_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        daily_returns: np.ndarray | pd.Series | None = None,
        fallback_flags: np.ndarray | pd.Series | None = None,
        target_gross: float = 2.0,
        realized_returns: np.ndarray | pd.Series | None = None,
    ) -> HealthScore:
        """Compute composite health score.

        Args:
            signals_df: DataFrame of signals (T × N), index = dates.
            weights_df: DataFrame of weights (T × N), index = dates.
            daily_returns: Array of daily net returns (length T).
            fallback_flags: Boolean array (length T), True if fallback used.
            target_gross: Target gross exposure.
            realized_returns: Array of realized target returns for IC calc.
                If None, uses daily_returns as proxy.

        Returns:
            HealthScore with component breakdown.
        """
        components: list[ComponentScore] = []

        # 1. IC Decay
        ic_score = self._score_ic_decay(
            signals_df, realized_returns or daily_returns
        )
        components.append(ic_score)

        # 2. Turnover
        to_score = self._score_turnover(weights_df)
        components.append(to_score)

        # 3. Gross Exposure Deviation
        gross_score = self._score_gross_deviation(weights_df, target_gross)
        components.append(gross_score)

        # 4. Fallback Rate
        fb_score = self._score_fallback_rate(fallback_flags, len(weights_df))
        components.append(fb_score)

        # 5. Signal Drift
        drift_score = self._score_signal_drift(signals_df)
        components.append(drift_score)

        active_components = [c for c in components if c.weight > 0]
        total_weight = sum(c.weight for c in active_components)
        if total_weight > 0:
            total = sum(c.weighted_score for c in active_components) / total_weight
        else:
            total = 50.0
        grade = self._grade_from_score(total)

        return HealthScore(score=total, grade=grade, components=components)

    def _grade_from_score(self, score: float) -> HealthGrade:
        if score >= 85:
            return HealthGrade.EXCELLENT
        elif score >= 70:
            return HealthGrade.GOOD
        elif score >= 55:
            return HealthGrade.FAIR
        elif score >= 40:
            return HealthGrade.POOR
        else:
            return HealthGrade.CRITICAL

    def _score_ic_decay(
        self,
        signals_df: pd.DataFrame,
        returns: np.ndarray | pd.Series | None,
    ) -> ComponentScore:
        """Score IC decay: recent IC vs trailing IC."""
        if returns is None or len(returns) == 0:
            return ComponentScore("ic_decay", 50.0, self.weights["ic_decay"], "No returns data")

        returns_arr = np.asarray(returns).flatten()
        T = min(len(signals_df), len(returns_arr))
        sig_flat = signals_df.iloc[:T].mean(axis=1).values
        ret_flat = returns_arr[:T]

        recent_n = min(self.recent_window, T // 2)
        trailing_n = min(self.trailing_window, T - recent_n)

        if recent_n < 5 or trailing_n < 10:
            return ComponentScore("ic_decay", 50.0, self.weights["ic_decay"], "Insufficient data")

        # Recent IC (rank correlation)
        recent_sig = sig_flat[T - recent_n:]
        recent_ret = ret_flat[T - recent_n:]
        recent_ic = self._rank_corr(recent_sig, recent_ret)

        # Trailing IC
        trail_sig = sig_flat[T - recent_n - trailing_n: T - recent_n]
        trail_ret = ret_flat[T - recent_n - trailing_n: T - recent_n]
        trailing_ic = self._rank_corr(trail_sig, trail_ret)

        if trailing_ic <= 0:
            score = 50.0
            detail = f"Trailing IC={trailing_ic:.3f} (non-positive baseline)"
        else:
            ratio = recent_ic / trailing_ic
            # ratio=1.0 → 100, ratio=0.5 → 50, ratio=0.0 → 0
            score = max(0.0, min(100.0, ratio * 100.0))
            detail = f"Recent IC={recent_ic:.3f}, Trailing IC={trailing_ic:.3f}, ratio={ratio:.2f}"

        return ComponentScore("ic_decay", score, self.weights["ic_decay"], detail)

    def _score_turnover(self, weights_df: pd.DataFrame) -> ComponentScore:
        """Score turnover: lower is better."""
        if len(weights_df) < 2:
            return ComponentScore("turnover", 50.0, self.weights["turnover"], "Insufficient data")

        w = weights_df.values
        gross = np.sum(np.abs(w), axis=1)
        gross_safe = np.where(gross > 1e-10, gross, 1.0)
        daily_turnover = np.sum(np.abs(np.diff(w, axis=0)), axis=1) / gross_safe[:-1]
        avg_turnover = float(np.mean(daily_turnover))

        if avg_turnover <= self.target_turnover:
            score = 100.0
        elif avg_turnover >= self.max_turnover:
            score = 0.0
        else:
            # Linear interpolation between target and max
            score = 100.0 * (1.0 - (avg_turnover - self.target_turnover) /
                             (self.max_turnover - self.target_turnover))

        detail = f"Avg turnover={avg_turnover:.4f} (target={self.target_turnover}, max={self.max_turnover})"
        return ComponentScore("turnover", score, self.weights["turnover"], detail)

    def _score_gross_deviation(
        self, weights_df: pd.DataFrame, target_gross: float
    ) -> ComponentScore:
        """Score gross exposure deviation from target."""
        if len(weights_df) == 0:
            return ComponentScore("gross_deviation", 50.0, self.weights["gross_deviation"], "No data")

        gross = np.sum(np.abs(weights_df.values), axis=1)
        avg_gross = float(np.mean(gross))
        deviation = abs(avg_gross - target_gross)
        rel_dev = deviation / target_gross if target_gross > 0 else float('inf')

        # rel_dev=0 → 100, rel_dev=0.5 → 0
        score = max(0.0, min(100.0, 100.0 * (1.0 - 2.0 * rel_dev)))
        detail = f"Avg gross={avg_gross:.3f}, target={target_gross:.3f}, rel_dev={rel_dev:.2%}"
        return ComponentScore("gross_deviation", score, self.weights["gross_deviation"], detail)

    def _score_fallback_rate(
        self,
        fallback_flags: np.ndarray | pd.Series | None,
        total_days: int,
    ) -> ComponentScore:
        """Score fallback rate: lower is better."""
        if fallback_flags is None or total_days == 0:
            return ComponentScore("fallback_rate", 100.0, self.weights["fallback_rate"], "No fallback data")

        flags = np.asarray(fallback_flags).flatten()
        n_fallback = int(np.sum(flags))
        rate = n_fallback / total_days

        # rate=0 → 100, rate=0.5 → 0
        score = max(0.0, min(100.0, 100.0 * (1.0 - 2.0 * rate)))
        detail = f"Fallback rate={rate:.2%} ({n_fallback}/{total_days} days)"
        return ComponentScore("fallback_rate", score, self.weights["fallback_rate"], detail)

    def _score_signal_drift(self, signals_df: pd.DataFrame) -> ComponentScore:
        """Score signal distribution drift: recent vs trailing statistics."""
        T = len(signals_df)
        if T < self.recent_window + 10:
            return ComponentScore("signal_drift", 50.0, self.weights["signal_drift"], "Insufficient data")

        recent = signals_df.iloc[T - self.recent_window:]
        trailing = signals_df.iloc[max(0, T - self.recent_window - self.trailing_window):T - self.recent_window]

        recent_mean = float(recent.values.mean())
        trailing_mean = float(trailing.values.mean())
        recent_std = float(recent.values.std())
        trailing_std = float(trailing.values.std())

        # Mean drift (relative to trailing std)
        mean_drift = abs(recent_mean - trailing_mean) / (trailing_std + 1e-10)
        # Std ratio
        std_ratio = recent_std / (trailing_std + 1e-10)

        # Penalize large mean drift and std changes
        mean_score = max(0.0, 100.0 - mean_drift * 50.0)
        std_score = max(0.0, 100.0 - abs(std_ratio - 1.0) * 100.0)
        score = 0.5 * mean_score + 0.5 * std_score

        detail = (f"Mean drift={mean_drift:.2f}σ, Std ratio={std_ratio:.2f} "
                  f"(recent μ={recent_mean:.5f}, σ={recent_std:.5f})")
        return ComponentScore("signal_drift", score, self.weights["signal_drift"], detail)

    @staticmethod
    def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Spearman rank correlation between two arrays."""
        if len(a) < 3:
            return 0.0
        a_rank = pd.Series(a).rank().values
        b_rank = pd.Series(b).rank().values
        a_centered = a_rank - a_rank.mean()
        b_centered = b_rank - b_rank.mean()
        denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
        if denom < 1e-10:
            return 0.0
        return float(np.sum(a_centered * b_centered) / denom)
