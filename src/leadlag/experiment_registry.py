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

import hashlib
import json
import logging
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        "record_id",
        "study_id",
        "metric_schema_version",
        "correction_of",
        "supersedes",
        "trial_id",
        "trial_status",
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
        record_id: str | None = None,
        study_id: str | None = None,
        metric_schema_version: str | None = None,
        correction_of: str | None = None,
        supersedes: list[str] | None = None,
        trial_id: str | None = None,
        trial_status: str = "completed",
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
        self.record_id = str(record_id or uuid4().hex)
        self.study_id = None if study_id is None else str(study_id)
        self.metric_schema_version = metric_schema_version or self.metrics.get(
            "metric_schema_version"
        )
        self.correction_of = correction_of
        self.supersedes = list(supersedes) if supersedes is not None else []
        self.trial_id = trial_id
        if trial_status not in {"completed", "aborted", "failed"}:
            raise ValueError("Unknown trial_status")
        self.trial_status = trial_status

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
            "record_id": self.record_id,
            "study_id": self.study_id,
            "metric_schema_version": self.metric_schema_version,
            "correction_of": self.correction_of,
            "supersedes": self.supersedes,
            "trial_id": self.trial_id,
            "trial_status": self.trial_status,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentRecord:
        legacy_id = (
            "legacy:"
            + hashlib.sha256(
                json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()
        )
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
            record_id=raw.get("record_id") or legacy_id,
            study_id=raw.get("study_id"),
            metric_schema_version=raw.get("metric_schema_version"),
            correction_of=raw.get("correction_of"),
            supersedes=raw.get("supersedes", []),
            trial_id=raw.get("trial_id"),
            trial_status=raw.get("trial_status", "completed"),
        )

    def deflated_sharpe(self) -> float | None:
        """Return the Deflated Sharpe Ratio (DSR) or None if inputs are missing."""
        if self.trial_status != "completed":
            return None
        if self.metric_schema_version != self.metrics.get("metric_schema_version"):
            return None
        return compute_deflated_sharpe(self.metrics)


def _moments(returns: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, skewness, Pearson kurtosis) of a return series."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all() or len(r) < 4:
        raise ValueError("At least 4 finite returns are required for moments")
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    if sigma < 1e-16:
        raise ValueError("Return series has near-zero variance")
    skew = np.mean((r - mu) ** 3) / (sigma**3)
    kurt = np.mean((r - mu) ** 4) / (sigma**4)
    return mu, skew, kurt


def compute_deflated_sharpe(metrics: dict[str, Any]) -> float | None:
    """Compute the Deflated Sharpe Ratio from a metrics dictionary.

    Requires explicit daily-v1 schema, valid status, annual/daily Sharpe frequency,
    positive integer annualization, complete finite evaluation returns and matching T.
    For N > 1, supply all N trial Sharpes (sample variance, ddof=1), or a
    nonnegative variance with basis cross_trial_ddof1/external_estimate. If both
    are supplied they must agree after frequency conversion. N=1 reduces to PSR.
    Unknown/lower-bound search histories cannot produce a validated DSR.
    Missing or inconsistent inputs return None; returns are never filtered.
    """
    sr = metrics.get("net_sharpe")
    n = metrics.get("trials")
    t = metrics.get("n_observations")
    if sr is None or n is None or t is None or metrics.get("metric_status") != "valid":
        return None
    if metrics.get("metric_schema_version") != "daily-v1":
        return None
    if metrics.get("trial_count_status", "complete") != "complete":
        return None
    if metrics.get("include_flat_days", True) is not True:
        return None
    # Returns and ``n_observations`` are daily by contract.  Registry records
    # store the selected Sharpe annualized; convert it before applying the
    # finite-sample/non-normality correction.
    sharpe_frequency = str(metrics.get("net_sharpe_frequency")).lower()
    if sharpe_frequency not in {"annual", "annualized", "yearly", "daily"}:
        return None
    try:
        annual_factor = float(metrics["trading_days_per_year"])
        if isinstance(sr, bool):
            return None
        sr = float(sr)
        if not np.isfinite([annual_factor, sr, float(n), float(t)]).all():
            return None
        if (
            annual_factor <= 0
            or annual_factor != int(annual_factor)
            or isinstance(metrics.get("trading_days_per_year"), bool)
            or isinstance(n, bool)
            or isinstance(t, bool)
        ):
            return None
        if float(n) != int(n) or float(t) != int(t):
            return None
        n, t = int(n), int(t)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if sharpe_frequency in {"annual", "annualized", "yearly"}:
        sr /= np.sqrt(annual_factor)
    if n <= 0 or t <= 1:
        return None

    returns = metrics.get("returns")
    if returns is not None:
        try:
            return_arr = np.asarray(returns, dtype=float)
            if return_arr.ndim != 1 or len(return_arr) != t:
                return None
            _, skew, kurt = _moments(return_arr)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None

    # Variance of the Sharpe estimate, accounting for non-normality.
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)
    if not np.isfinite(denom) or denom <= 0:
        return None
    sr_std = np.sqrt(denom / (t - 1))
    if sr_std <= 1e-16:
        return None

    # Cross-trial variance of Sharpe estimates (V[SR_n]).
    trial_sharpes = metrics.get("trial_sharpes")
    explicit_v = metrics.get("trial_sharpe_variance")
    variance_frequency = str(
        metrics.get("trial_sharpe_variance_frequency", sharpe_frequency)
    ).lower()
    if variance_frequency not in {"annual", "annualized", "yearly", "daily"}:
        return None
    trial_frequency = str(metrics.get("trial_sharpes_frequency", sharpe_frequency)).lower()
    if trial_frequency not in {"annual", "annualized", "yearly", "daily"}:
        return None
    sample_var = None
    if trial_sharpes is not None:
        try:
            trial_arr = np.asarray(trial_sharpes, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return None
        if trial_arr.ndim != 1 or len(trial_arr) != n or not np.isfinite(trial_arr).all():
            return None
        if trial_frequency != "daily":
            trial_arr = trial_arr / np.sqrt(annual_factor)
        sample_var = float(np.var(trial_arr, ddof=1)) if n > 1 else 0.0
    if explicit_v is not None:
        if metrics.get("trial_sharpe_variance_basis") not in {
            "cross_trial_ddof1",
            "external_estimate",
        }:
            return None
        try:
            var = float(explicit_v)
        except (TypeError, ValueError, OverflowError):
            return None
        if isinstance(explicit_v, bool):
            return None
        if variance_frequency != "daily":
            var /= annual_factor
        if sample_var is not None and not np.isclose(var, sample_var, rtol=1e-10, atol=1e-14):
            return None
    elif sample_var is not None:
        var = sample_var
    elif n == 1:
        var = 0.0
    else:
        # A null single-trial variance is not an observed cross-trial estimate.
        return None
    if not np.isfinite(var) or var < 0:
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
    return float(dsr) if np.isfinite(dsr) else None


class ExperimentRegistry:
    """Append-only JSONL registry for experiment records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: ExperimentRecord) -> ExperimentRecord:
        """Append a record to the registry."""
        if entry.trial_id is not None:
            started = self.get_trial(entry.trial_id)
            if started["study_id"] != entry.study_id or started["candidate_id"] != entry.name:
                raise ValueError("Trial identity does not match its start event")
            if entry.correction_of is None and any(rec.trial_id == entry.trial_id for rec in self):
                raise ValueError("Trial already has an outcome")
            if entry.parameters != started["parameters"]:
                raise ValueError("Trial parameters differ from the frozen start event")
        self._append(entry.to_dict())
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
                    if "event_type" not in raw:
                        yield ExperimentRecord.from_dict(raw)

        return _records()

    def _append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def study_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            raw
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip() and "event_type" in (raw := json.loads(line))
        ]

    def register_study(
        self,
        study_id: str,
        *,
        hypothesis: str,
        candidates: list[str],
        protocol: dict[str, Any],
        history_complete: bool = False,
        historical_trials_lower_bound: int = 0,
    ) -> dict[str, Any]:
        """Freeze a family before any candidate starts; never infer complete history."""
        if not study_id.strip() or not hypothesis.strip() or not candidates:
            raise ValueError("Study ID, hypothesis and candidates are required")
        if any(not candidate.strip() for candidate in candidates) or len(set(candidates)) != len(
            candidates
        ):
            raise ValueError("Candidate IDs must be nonempty and unique")
        if any(event["study_id"] == study_id for event in self.study_events()) or self.count_trials(
            study_id=study_id
        ):
            raise ValueError("Study already exists; use a new study ID for a new protocol")
        required = {
            "target_schema",
            "cost_schema",
            "is_period",
            "oos_period",
            "purge",
            "selection_rule",
            "metrics_spec",
        }
        if required - protocol.keys() or any(not protocol[key] for key in required):
            raise ValueError(
                "Study protocol requires target/cost, IS/OOS, purge, selection rule and metrics spec"
            )
        if not isinstance(history_complete, bool):
            raise ValueError("history_complete must be boolean")
        spec = protocol["metrics_spec"]
        if (
            not isinstance(spec, dict)
            or spec.get("frequency") != "daily"
            or spec.get("include_flat_days") is not True
        ):
            raise ValueError("Study metrics require daily frequency and all flat days")
        annual = spec.get("annualization_periods")
        if isinstance(annual, bool) or not isinstance(annual, int) or annual <= 0:
            raise ValueError("Study annualization must be a positive integer")
        for key in ("target_schema", "cost_schema", "selection_rule"):
            if not isinstance(protocol[key], str) or not protocol[key].strip():
                raise ValueError(f"{key} must be a nonempty schema/rule")
        periods = []
        for key in ("is_period", "oos_period"):
            period = protocol[key]
            if not isinstance(period, dict):
                raise ValueError("IS/OOS must provide start/end dates")
            try:
                start, end = (datetime.fromisoformat(period[k]) for k in ("start", "end"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("IS/OOS must provide ISO start/end dates") from exc
            if start > end:
                raise ValueError("Period starts after its end")
            periods.append((start, end))
        if periods[0][1] >= periods[1][0]:
            raise ValueError("IS must end before OOS begins")
        purge = protocol["purge"]
        if not isinstance(purge, dict) or not purge.get("rationale"):
            raise ValueError("Purge requires a rationale and session counts")
        for key in ("sessions", "embargo_sessions"):
            value = purge.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Purge/embargo session counts must be nonnegative integers")
        if (
            isinstance(historical_trials_lower_bound, bool)
            or not isinstance(historical_trials_lower_bound, int)
            or historical_trials_lower_bound < 0
        ):
            raise ValueError("Historical trial lower bound must be a nonnegative integer")
        event = {
            "event_type": "study_registered",
            "study_id": study_id,
            "timestamp": _utc_now().isoformat(),
            "hypothesis": hypothesis,
            "candidates": candidates,
            "protocol": protocol,
            "protocol_hash": hashlib.sha256(
                json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "history_complete": history_complete,
            "historical_trials_lower_bound": historical_trials_lower_bound,
        }
        self._append(event)
        return event

    def get_study(self, study_id: str) -> dict[str, Any]:
        for event in self.study_events():
            if event["event_type"] == "study_registered" and event["study_id"] == study_id:
                return event
        raise KeyError(f"Study must be registered before execution: {study_id}")

    def start_trial(
        self,
        study_id: str,
        candidate_id: str,
        *,
        parameters: dict[str, Any],
        code_hash: str,
        data_hash: str,
    ) -> str:
        """Persist frozen provenance before evaluation, including trials that fail."""
        study = self.get_study(study_id)
        if candidate_id not in study["candidates"]:
            raise ValueError("Candidate is outside the preregistered study")
        if any(
            e["event_type"] == "study_selection" and e["study_id"] == study_id
            for e in self.study_events()
        ):
            raise ValueError("Study is already selected; register a new study")
        for value in (code_hash, data_hash):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Code/data provenance must be SHA-256 hashes")
        trial_id = uuid4().hex
        config_hash = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self._append(
            {
                "event_type": "trial_started",
                "study_id": study_id,
                "timestamp": _utc_now().isoformat(),
                "trial_id": trial_id,
                "candidate_id": candidate_id,
                "parameters": parameters,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "data_hash": data_hash,
                "protocol_hash": study["protocol_hash"],
            }
        )
        return trial_id

    def get_trial(self, trial_id: str) -> dict[str, Any]:
        for event in self.study_events():
            if event.get("trial_id") == trial_id and event["event_type"] == "trial_started":
                return event
        raise KeyError(f"Unknown started trial: {trial_id}")

    def select_study(
        self, study_id: str, *, selected_trial_id: str | None, reason: str
    ) -> dict[str, Any]:
        """Snapshot all attempted and unattempted candidates at selection time."""
        study = self.get_study(study_id)
        events = [e for e in self.study_events() if e["study_id"] == study_id]
        if any(e["event_type"] == "study_selection" for e in events):
            raise ValueError("Study already has a selection")
        outcomes = {
            rec.trial_id: rec for rec in self.iter_current_records() if rec.study_id == study_id
        }
        trials = []
        for event in events:
            if event["event_type"] != "trial_started":
                continue
            rec = outcomes.get(event["trial_id"])
            trials.append(
                {
                    "trial_id": event["trial_id"],
                    "candidate_id": event["candidate_id"],
                    "status": rec.trial_status if rec else "running",
                    "record_id": rec.record_id if rec else None,
                    "decision": rec.decision.value if rec else "pending",
                }
            )
        if not reason.strip() or any(t["status"] == "running" for t in trials):
            raise ValueError("Selection requires a reason and outcomes for every started trial")
        if selected_trial_id is not None:
            selected = outcomes.get(selected_trial_id)
            if selected is None or selected.trial_status != "completed":
                raise ValueError("Selected trial must be completed within the study")
        selected_dsr = None
        if (
            selected_trial_id is not None
            and study["history_complete"]
            and outcomes[selected_trial_id].metric_schema_version
            == outcomes[selected_trial_id].metrics.get("metric_schema_version")
        ):
            selected_metrics = dict(outcomes[selected_trial_id].metrics)
            selected_metrics["trials"] = len(trials) + study["historical_trials_lower_bound"]
            selected_metrics["trial_count_status"] = "complete"
            # Use the explicitly supplied full-trial variance, never infer it from successful runs only.
            selected_dsr = compute_deflated_sharpe(selected_metrics)
        event = {
            "event_type": "study_selection",
            "study_id": study_id,
            "timestamp": _utc_now().isoformat(),
            "selected_trial_id": selected_trial_id,
            "reason": reason,
            "trials": trials,
            "selected_deflated_sharpe": selected_dsr,
            "unattempted_candidates": sorted(
                set(study["candidates"]) - {t["candidate_id"] for t in trials}
            ),
            "trials_lower_bound": len(trials) + study["historical_trials_lower_bound"],
            "trial_count_status": "complete" if study["history_complete"] else "lower_bound",
        }
        self._append(event)
        return event

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

    def record_correction(
        self,
        original_record_id: str,
        corrected: ExperimentRecord,
    ) -> ExperimentRecord:
        """Append a correction while retaining the original record unchanged."""
        original = next((rec for rec in self if rec.record_id == original_record_id), None)
        if original is None:
            raise KeyError(f"Unknown experiment record_id: {original_record_id}")
        if corrected.record_id == original_record_id:
            raise ValueError("A correction must have a new record_id")
        payload = corrected.to_dict()
        payload["correction_of"] = original_record_id
        if original.trial_id is not None:
            payload["trial_id"] = original.trial_id
            payload["trial_status"] = original.trial_status
        payload["supersedes"] = sorted(set(corrected.supersedes) | {original_record_id})
        payload["related_records"] = sorted(
            set(payload.get("related_records", [])) | {original_record_id}
        )
        correction = ExperimentRecord.from_dict(payload)
        return self.record(correction)

    def iter_current_records(self) -> Iterator[ExperimentRecord]:
        """Iterate the latest non-superseded view without deleting history."""
        records = list(self)
        superseded = {
            str(record_id)
            for rec in records
            for record_id in (rec.supersedes + ([rec.correction_of] if rec.correction_of else []))
        }
        return iter(rec for rec in records if rec.record_id not in superseded)

    def count_trials(
        self,
        since: datetime | None = None,
        *,
        study_id: str | None = None,
    ) -> int:
        """Count original trial records, optionally within an explicit study."""
        corrected_study_ids: dict[str, str] = {}
        if study_id is not None:
            all_records = {rec.record_id: rec for rec in self}
            for correction in self.iter_current_records():
                if correction.study_id is None:
                    continue
                ancestors = list(correction.supersedes)
                if correction.correction_of is not None:
                    ancestors.append(correction.correction_of)
                seen: set[str] = set()
                while ancestors:
                    record_id = ancestors.pop()
                    if record_id in seen:
                        continue
                    seen.add(record_id)
                    corrected_study_ids[record_id] = correction.study_id
                    previous = all_records.get(record_id)
                    if previous is not None:
                        ancestors.extend(previous.supersedes)
                        if previous.correction_of is not None:
                            ancestors.append(previous.correction_of)
        starts = [e for e in self.study_events() if e["event_type"] == "trial_started"]
        started_ids = {e["trial_id"] for e in starts}
        start_count = sum(
            1
            for e in starts
            if (study_id is None or e["study_id"] == study_id)
            and (since is None or datetime.fromisoformat(e["timestamp"]) >= since)
        )
        return start_count + sum(
            1
            for rec in self
            if rec.correction_of is None
            and rec.trial_id not in started_ids
            and (since is None or rec.start_time >= since)
            and (
                study_id is None
                or rec.study_id == study_id
                or corrected_study_ids.get(rec.record_id) == study_id
            )
        )

    def decisions(self) -> list[dict[str, Any]]:
        """Return a summary of adopted/rejected/pending records."""
        out: list[dict[str, Any]] = []
        for rec in self.iter_current_records():
            dsr = rec.deflated_sharpe()
            out.append(
                {
                    "record_id": rec.record_id,
                    "study_id": rec.study_id,
                    "trial_id": rec.trial_id,
                    "trial_status": rec.trial_status,
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
