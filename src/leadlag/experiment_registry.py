"""Experiment registry for tracking trials, outcomes, and adoption decisions.

The registry turns the ad-hoc management of backtest experiments into a
machine-readable audit trail. Every record stores the hypothesis, parameters,
metrics, decision, and a deflated Sharpe ratio that accounts for the number
of independent trials tried before selecting the reported result.

References
----------
- Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality", JPM 2014.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Decision(StrEnum):
    PENDING = "pending"
    REJECTED = "rejected"
    ADOPTED = "adopted"


class ExperimentRecord:
    """One experiment trial with enough metadata for reproducibility and audit."""

    __slots__ = (
        "name",
        "hypothesis",
        "start_time",
        "end_time",
        "parameters",
        "metrics",
        "decision",
        "report_path",
        "related_records",
    )

    def __init__(
        self,
        name: str,
        hypothesis: str,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        parameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        decision: Decision = Decision.PENDING,
        report_path: str | None = None,
        related_records: list[str] | None = None,
    ) -> None:
        self.name = name
        self.hypothesis = hypothesis
        self.start_time = start_time or _utc_now()
        self.end_time = end_time
        self.parameters = dict(parameters) if parameters is not None else {}
        self.metrics = dict(metrics) if metrics is not None else {}
        self.decision = Decision(decision)
        self.report_path = report_path
        self.related_records = list(related_records) if related_records is not None else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "decision": self.decision.value,
            "report_path": self.report_path,
            "related_records": self.related_records,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentRecord:
        return cls(
            name=raw["name"],
            hypothesis=raw["hypothesis"],
            start_time=datetime.fromisoformat(raw["start_time"]),
            end_time=datetime.fromisoformat(raw["end_time"]) if raw.get("end_time") else None,
            parameters=raw.get("parameters", {}),
            metrics=raw.get("metrics", {}),
            decision=Decision(raw.get("decision", "pending")),
            report_path=raw.get("report_path"),
            related_records=raw.get("related_records", []),
        )

    def deflated_sharpe(self) -> float | None:
        """Return the Deflated Sharpe Ratio (DSR) or None if inputs are missing."""
        return compute_deflated_sharpe(self.metrics)


def _moments(returns: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, skewness, excess_kurtosis) of a return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        raise ValueError("At least 4 finite returns are required for moments")
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    if sigma < 1e-16:
        raise ValueError("Return series has near-zero variance")
    skew = np.mean((r - mu) ** 3) / (sigma ** 3)
    kurt = np.mean((r - mu) ** 4) / (sigma ** 4)
    return mu, skew, kurt


def compute_deflated_sharpe(metrics: dict[str, Any]) -> float | None:
    """Compute the Deflated Sharpe Ratio from a metrics dictionary.

    Required fields in ``metrics``:
      - ``net_sharpe`` (float): the selected annualized Sharpe ratio (SR*).
      - ``trials`` (int): number of independent trials performed before
        selecting this result (N).
      - ``n_observations`` (int): track-record length T.

    Optional fields:
      - ``returns`` (list[float]): strategy return series for skew/kurtosis.
      - ``trial_sharpes`` (list[float]): Sharpe ratios of all N trials for
        estimating cross-trial variance V.
      - ``trial_sharpe_variance`` (float): explicit V estimate.

    Returns None if the required fields are absent or if the denominator is
    non-positive (e.g. extreme Sharpe/kurtosis).
    """
    sr = metrics.get("net_sharpe")
    n = metrics.get("trials")
    t = metrics.get("n_observations")
    if sr is None or n is None or t is None:
        return None
    sr = float(sr)
    n = int(n)
    t = int(t)
    if n <= 0 or t <= 1:
        return None

    returns = metrics.get("returns")
    if returns is not None:
        try:
            _, skew, kurt = _moments(np.asarray(returns, dtype=float))
        except ValueError as exc:
            logger.warning("Could not compute return moments for DSR: %s", exc)
            skew, kurt = 0.0, 3.0
    else:
        skew, kurt = 0.0, 3.0

    # Variance of the Sharpe estimate, accounting for non-normality.
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    if denom <= 0:
        return None
    sr_std = np.sqrt(denom / (t - 1))
    if sr_std <= 1e-16:
        return None

    # Cross-trial variance of Sharpe estimates (V[SR_n]).
    trial_sharpes = metrics.get("trial_sharpes")
    explicit_v = metrics.get("trial_sharpe_variance")
    if explicit_v is not None:
        var = float(explicit_v)
    elif trial_sharpes is not None and len(trial_sharpes) >= 2:
        var = float(np.var(trial_sharpes, ddof=1))
    else:
        # Fallback: variance of a single Sharpe under the null.
        var = 1.0 / (t - 1)
    if var <= 0:
        return None

    # Expected maximum Sharpe under the null after N trials.
    # For N=1 there is no selection bias, so SR_0 = 0 (DSR reduces to PSR).
    if n == 1:
        sr_0 = 0.0
    else:
        gamma = 0.5772156649015329  # Euler-Mascheroni constant
        z_n = sps.norm.ppf(1.0 - 1.0 / n)
        z_ne = sps.norm.ppf(1.0 - 1.0 / (n * np.e))
        sr_0 = np.sqrt(var) * ((1.0 - gamma) * z_n + gamma * z_ne)

    dsr = sps.norm.cdf((sr - sr_0) / sr_std)
    return float(dsr)


class ExperimentRegistry:
    """Append-only JSONL registry for experiment records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: ExperimentRecord) -> ExperimentRecord:
        """Append a record to the registry."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def __iter__(self) -> Iterator[ExperimentRecord]:
        """Iterate all records in file order."""
        if not self.path.exists():
            return iter([])

        def _records() -> Iterator[ExperimentRecord]:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    yield ExperimentRecord.from_dict(raw)

        return _records()

    def iter_records(
        self,
        *,
        name: str | None = None,
        decision: Decision | None = None,
    ) -> Iterator[ExperimentRecord]:
        """Filtered iteration."""
        for rec in self:
            if name is not None and rec.name != name:
                continue
            if decision is not None and rec.decision != decision:
                continue
            yield rec

    def count_trials(self, since: datetime | None = None) -> int:
        """Count records since a given UTC time."""
        if since is None:
            return sum(1 for _ in self)
        return sum(1 for rec in self if rec.start_time >= since)

    def decisions(self) -> list[dict[str, Any]]:
        """Return a summary of adopted/rejected/pending records."""
        out: list[dict[str, Any]] = []
        for rec in self:
            dsr = rec.deflated_sharpe()
            out.append(
                {
                    "name": rec.name,
                    "hypothesis": rec.hypothesis,
                    "decision": rec.decision.value,
                    "net_sharpe": rec.metrics.get("net_sharpe"),
                    "trials": rec.metrics.get("trials"),
                    "deflated_sharpe": dsr,
                    "report_path": rec.report_path,
                    "start_time": rec.start_time.isoformat(),
                }
            )
        return out

    def find_related(
        self, names: Iterable[str], include_rejected: bool = True
    ) -> list[ExperimentRecord]:
        """Find all records that are related to the given names."""
        name_set = set(names)
        out: list[ExperimentRecord] = []
        for rec in self:
            if rec.name in name_set or any(r in name_set for r in rec.related_records):
                if not include_rejected and rec.decision == Decision.REJECTED:
                    continue
                out.append(rec)
        return out
