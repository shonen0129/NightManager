"""Append-only paired ML-overlay shadow decisions for prospective evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from leadlag.config.paths import project_root
from leadlag.domain.inputs import DecisionInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.runner.production import ProductionRunner

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "ml-overlay-paired-shadow-v1"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else _json_value(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _result_payload(
    result: PortfolioDecision,
    ticker_order: tuple[str, ...],
) -> dict[str, Any]:
    costs = result.costs
    return cast(dict[str, Any], _json_value({
        "weights": {ticker: float(weight) for ticker, weight in zip(ticker_order, result.w_final, strict=True)},
        "scores": [float(value) for value in result.scores],
        "scores_overlay": (
            None if result.scores_overlay is None
            else [float(value) for value in result.scores_overlay]
        ),
        "effective_distribution_sha256": {
            "mu_gap": _array_hash(result.mu_gap),
            "omega_gap": _array_hash(result.Omega_gap),
        },
        "gross_exposure": float(np.sum(np.abs(result.w_final))),
        "net_exposure": float(np.sum(result.w_final)),
        "modeled_costs_return_fraction": (
            None if costs is None else {
                "slippage": float(costs.slippage),
                "financing": float(costs.financing),
                "borrow": float(costs.borrow),
                "reverse": float(costs.reverse),
                "total": float(costs.total),
            }
        ),
        "fallback": _json_value(result.fallback),
        "pit_binning": _json_value(result.pit_binning),
        "leakage_status": _json_value(result.leakage.get("status")),
        "numerical_status": _json_value(result.numerical.get("status")),
        "alerts": _json_value(result.alerts),
    }))


def _read_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed shadow JSONL at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Shadow JSONL row at {path}:{line_number} is not an object")
            records.append(item)
    return records


def append_ml_overlay_shadow(
    *,
    app_config: Any,
    decision_inputs: DecisionInputs,
    ml_enabled_result: PortfolioDecision,
    output_dir: str | Path,
    overlay_metadata: dict[str, Any] | None = None,
    capital_jpy: float | None = None,
    raise_on_baseline_failure: bool = False,
) -> Path:
    """Run the same production model with ML disabled and append the pair.

    This function never imports or invokes a broker. The ML-enabled result is
    the already-computed production result; only the disabled counterfactual
    is run here. Baseline calculation failures are recorded as incomplete
    rows; persistence failures are raised to the caller, which should keep
    shadow failure separate from the live decision path.
    """
    if not bool(app_config.v2.ml_overlay_enabled):
        raise ValueError("paired ML shadow requires the production ML overlay to be enabled")

    disabled_v2 = app_config.v2.model_copy(deep=True, update={"ml_overlay_enabled": False})
    disabled_config = app_config.model_copy(deep=True, update={"v2": disabled_v2})
    baseline_error: str | None = None
    try:
        baseline_result = ProductionRunner(disabled_config).run(decision_inputs)
    except Exception as exc:  # Keep baseline failures visible without affecting production.
        baseline_result = None
        baseline_error = f"{type(exc).__name__}: {exc}"
        logger.exception("ML-disabled paired shadow calculation failed")

    v2_config = app_config.v2.model_dump(mode="json")
    baseline_v2_config = disabled_v2.model_dump(mode="json")
    shared_config = {
        "risk": _model_dump(getattr(app_config, "risk", None)),
        "strategy": _model_dump(getattr(app_config, "strategy", None)),
        "broker_provider": getattr(app_config, "broker_provider", None),
    }
    cost_config = getattr(app_config.v2, "costs", None)
    side_leverage = getattr(cost_config, "side_leverage", None)
    input_version = decision_inputs.version
    ticker_order = tuple(decision_inputs.known.ticker_order)
    record: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "trade_date": decision_inputs.known.trade_date.strftime("%Y-%m-%d"),
        "as_of": decision_inputs.known.as_of.isoformat(),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "decision_source": decision_inputs.known.source,
        "input_version": {
            "schema_version": input_version.schema_version,
            "digest": input_version.digest,
            "known_fingerprint": input_version.known_fingerprint,
            "historical_fingerprint": input_version.historical_fingerprint,
        },
        "quote_snapshot_id": getattr(decision_inputs.known, "quote_snapshot_id", None),
        "config_fingerprints": {
            "ml_enabled": _canonical_hash({**shared_config, "v2": v2_config}),
            "ml_disabled": _canonical_hash({**shared_config, "v2": baseline_v2_config}),
        },
        "overlay_artifact": _json_value(overlay_metadata or {}),
        "sizing": {
            "capital_jpy": None if capital_jpy is None else float(capital_jpy),
            "side_leverage": None if side_leverage is None else float(side_leverage),
        },
        "input_prices": {
            "current": _json_value(dict(decision_inputs.known.current_prices)),
            "previous_close": _json_value(dict(decision_inputs.known.prev_closes)),
            "price_sources": _json_value(dict(decision_inputs.known.price_sources)),
            "price_observed_at": _json_value(
                dict(getattr(decision_inputs.known, "price_observed_at", {}))
            ),
        },
        "variants": {
            "ml_enabled": _result_payload(ml_enabled_result, ticker_order),
            "ml_disabled": (
                None if baseline_result is None
                else _result_payload(baseline_result, ticker_order)
            ),
        },
        "status": "baseline_failed" if baseline_error is not None else "complete",
        "baseline_error": baseline_error,
        "cost_evidence": {
            "status": "modeled_only_pending_real_execution_reconciliation",
            "note": "Decision cost fields are model estimates, not observed fill costs.",
        },
    }
    record["record_fingerprint"] = _canonical_hash(
        {key: value for key, value in record.items() if key != "recorded_at_utc"}
    )

    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = project_root() / directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "daily.jsonl"
    lock_path = directory / ".daily.jsonl.lock"
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = _read_existing_records(path)
        duplicate = next(
            (
                item for item in existing
                if item.get("record_fingerprint") == record["record_fingerprint"]
            ),
            None,
        )
        if duplicate is not None:
            logger.info("Paired ML shadow already recorded for %s", record["trade_date"])
            if raise_on_baseline_failure and duplicate.get("status") != "complete":
                raise RuntimeError("ML-disabled counterfactual failed; incomplete shadow row is recorded")
            return path
        record["attempt"] = 1 + sum(
            item.get("trade_date") == record["trade_date"] for item in existing
        )
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    if raise_on_baseline_failure and baseline_error is not None:
        raise RuntimeError("ML-disabled counterfactual failed; incomplete shadow row is recorded")
    return path


__all__ = ["append_ml_overlay_shadow"]
