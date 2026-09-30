"""Join paired shadow decisions to verified 09:10-to-close outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from leadlag.execution.account_risk import REQUIRED_PNL_BASIS
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
from leadlag.reporting.ml_overlay_forward_schema import OUTCOME_SCHEMA


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _key(record: Mapping[str, Any]) -> tuple[str, str]:
    trade_date = str(record.get("trade_date") or "")
    input_version = record.get("input_version")
    if not isinstance(input_version, Mapping):
        raise ValueError("shadow record lacks input_version")
    digest = str(input_version.get("digest") or "")
    if not trade_date or not digest:
        raise ValueError("shadow record requires trade_date and input_version.digest")
    return trade_date, digest


def _variant_return(variant: Mapping[str, Any], targets: Mapping[str, float]) -> dict[str, float]:
    weights = variant.get("weights")
    costs = variant.get("modeled_costs_return_fraction")
    if not isinstance(weights, Mapping) or not isinstance(costs, Mapping):
        raise ValueError("shadow variant requires weights and modeled costs")
    if set(weights) != set(targets):
        raise ValueError("shadow weights and realized target ticker sets differ")
    gross = sum(_finite_number(weights[ticker], f"weight[{ticker}]") * targets[ticker]
                for ticker in targets)
    modeled_cost = _finite_number(costs.get("total"), "modeled_costs_return_fraction.total")
    return {"gross_return": gross, "modeled_net_return": gross - modeled_cost}


def _actual_candidate_return(
    outcome: Mapping[str, Any],
    *,
    capital_jpy: float | None,
    input_digest: str,
) -> tuple[float | None, str]:
    execution = outcome.get("execution_reconciliation")
    if not isinstance(execution, Mapping):
        return None, "missing_execution_reconciliation"
    required = (
        "fill_reconciled",
        "positions_reconciled",
        "cash_reconciled",
        "fees_complete",
    )
    if any(execution.get(field) is not True for field in required):
        return None, "execution_or_fee_reconciliation_incomplete"
    if execution.get("pnl_basis") != REQUIRED_PNL_BASIS:
        return None, "unsupported_actual_pnl_basis"
    if execution.get("status") != "complete":
        return None, "execution_reconciliation_not_complete"
    if execution.get("input_version_digest") != input_digest:
        return None, "execution_input_digest_mismatch"
    try:
        capital = _finite_number(capital_jpy, "capital_jpy")
    except ValueError:
        return None, "capital_missing_or_non_positive"
    if capital <= 0.0:
        return None, "capital_missing_or_non_positive"
    try:
        pnl = _finite_number(
            execution.get("candidate_actual_net_pnl_jpy"),
            "candidate_actual_net_pnl_jpy",
        )
    except ValueError:
        return None, "actual_net_pnl_missing_or_invalid"
    return pnl / capital, "complete"


def _metrics(values: Sequence[float], annualization_periods: int) -> dict[str, Any]:
    if not values:
        return {"status": "insufficient_complete_pairs", "n_observations": 0}
    series = pd.Series(values, dtype=float)
    metrics = calculate_metrics(
        series,
        spec=MetricsSpec(annualization_periods=annualization_periods, include_flat_days=True),
    )
    return {
        "status": "complete",
        "n_observations": len(values),
        "annualization_periods": annualization_periods,
        "net_sharpe": (
            float(metrics["Sharpe"]) if math.isfinite(float(metrics["Sharpe"])) else None
        ),
        "max_drawdown": (
            float(metrics["MDD"]) if math.isfinite(float(metrics["MDD"])) else None
        ),
        "total_return": (
            float(metrics["Total Return"])
            if math.isfinite(float(metrics["Total Return"])) else None
        ),
        "mean_daily_return": float(series.mean()),
    }


def evaluate_ml_overlay_forward(
    shadow_records: Sequence[Mapping[str, Any]],
    outcome_records: Sequence[Mapping[str, Any]],
    *,
    annualization_periods: int = 252,
    expected_trade_dates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score matched on/off decisions and retain every unmatched shadow date.

    Outcomes must supply realized returns computed from the captured 09:10
    quote midpoint and the official close. Actual P&L is reported separately
    and only when fills, inventory, cash, and fees are fully reconciled.
    Counterfactuals remain weight-based modeled results; this function never
    labels them as fills or same-lot executions.
    """
    if annualization_periods <= 1:
        raise ValueError("annualization_periods must be greater than one")
    outcome_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for outcome in outcome_records:
        if outcome.get("schema_version") != OUTCOME_SCHEMA:
            raise ValueError("unsupported ML-overlay forward outcome schema")
        outcome_key = _key(outcome)
        if outcome_key in outcome_by_key:
            raise ValueError(f"duplicate forward outcome for {outcome_key}")
        outcome_by_key[outcome_key] = outcome

    grouped_shadows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for shadow in shadow_records:
        grouped_shadows.setdefault(_key(shadow), []).append(shadow)
    digests_by_date: dict[str, set[str]] = {}
    for trade_date, input_digest in grouped_shadows:
        digests_by_date.setdefault(trade_date, set()).add(input_digest)
    selected_shadows: list[Mapping[str, Any]] = []
    daily: list[dict[str, Any]] = []
    ambiguous_dates = {
        trade_date for trade_date, digests in digests_by_date.items() if len(digests) > 1
    }
    for trade_date in sorted(ambiguous_dates):
        daily.append(
            {
                "trade_date": trade_date,
                "status": "multiple_shadow_input_versions",
                "input_digests": sorted(digests_by_date[trade_date]),
            }
        )
    for shadow_key, group in grouped_shadows.items():
        if shadow_key[0] in ambiguous_dates:
            continue
        fingerprints = {
            str(item.get("record_fingerprint") or _digest(item)) for item in group
        }
        if len(fingerprints) > 1:
            daily.append(
                {
                    "trade_date": shadow_key[0],
                    "input_digest": shadow_key[1],
                    "status": "shadow_decision_ambiguous",
                    "attempt_count": len(group),
                    "record_fingerprints": sorted(fingerprints),
                }
            )
            continue
        selected_shadows.append(group[0])
    enabled_returns: list[float] = []
    disabled_returns: list[float] = []
    deltas: list[float] = []
    actual_returns: list[float] = []
    actual_return_dates: list[str] = []
    for shadow in selected_shadows:
        trade_date, input_digest = _key(shadow)
        record: dict[str, Any] = {
            "trade_date": trade_date,
            "input_digest": input_digest,
            "quote_snapshot_id": shadow.get("quote_snapshot_id"),
            "shadow_status": shadow.get("status"),
        }
        matched_outcome = outcome_by_key.get((trade_date, input_digest))
        if matched_outcome is None:
            record["status"] = "outcome_missing"
            daily.append(record)
            continue
        if matched_outcome.get("target_basis") != "frozen_0910_quote_mid_to_official_close":
            record["status"] = "target_basis_unverified"
            daily.append(record)
            continue
        if matched_outcome.get("quote_snapshot_id") != shadow.get("quote_snapshot_id"):
            record["status"] = "quote_snapshot_mismatch"
            daily.append(record)
            continue
        if matched_outcome.get("label_status") != "complete":
            record["status"] = str(matched_outcome.get("label_status") or "label_incomplete")
            daily.append(record)
            continue

        raw_targets = matched_outcome.get("target_returns")
        variants = shadow.get("variants")
        if not isinstance(raw_targets, Mapping) or not isinstance(variants, Mapping):
            record["status"] = "invalid_target_or_variant_payload"
            daily.append(record)
            continue
        targets = {
            str(ticker): _finite_number(value, f"target_returns[{ticker}]")
            for ticker, value in raw_targets.items()
        }
        enabled = variants.get("ml_enabled")
        disabled = variants.get("ml_disabled")
        if not isinstance(enabled, Mapping) or not isinstance(disabled, Mapping):
            record["status"] = "paired_shadow_incomplete"
            daily.append(record)
            continue
        try:
            enabled_score = _variant_return(enabled, targets)
            disabled_score = _variant_return(disabled, targets)
        except ValueError as exc:
            record["status"] = "invalid_pair: " + str(exc)
            daily.append(record)
            continue

        enabled_returns.append(enabled_score["modeled_net_return"])
        disabled_returns.append(disabled_score["modeled_net_return"])
        deltas.append(enabled_score["modeled_net_return"] - disabled_score["modeled_net_return"])
        capital = shadow.get("sizing", {}).get("capital_jpy") if isinstance(
            shadow.get("sizing"), Mapping
        ) else None
        actual_return, actual_status = _actual_candidate_return(
            matched_outcome,
            capital_jpy=capital,
            input_digest=input_digest,
        )
        if actual_return is not None:
            actual_returns.append(actual_return)
            actual_return_dates.append(trade_date)
        record.update(
            {
                "status": "paired_complete",
                "target_source": matched_outcome.get("target_source"),
                "ml_enabled_gross_return": enabled_score["gross_return"],
                "ml_enabled_modeled_net_return": enabled_score["modeled_net_return"],
                "ml_disabled_gross_return": disabled_score["gross_return"],
                "ml_disabled_modeled_net_return": disabled_score["modeled_net_return"],
                "paired_modeled_net_delta": (
                    enabled_score["modeled_net_return"] - disabled_score["modeled_net_return"]
                ),
                "candidate_actual_net_return": actual_return,
                "candidate_actual_status": actual_status,
                "baseline_evidence": "modeled_weights_same_capital; no_lot_or_fill_claim",
            }
        )
        daily.append(record)

    daily.sort(key=lambda item: (item["trade_date"], item.get("input_digest", "")))
    expected_dates = sorted(set(str(value) for value in (expected_trade_dates or [])))
    seen_dates = {str(item.get("trade_date") or "") for item in daily}
    for missing_date in expected_dates:
        if missing_date not in seen_dates:
            daily.append({"trade_date": missing_date, "status": "decision_missing"})
    daily.sort(key=lambda item: (item["trade_date"], item.get("input_digest", "")))
    complete = sum(item.get("status") == "paired_complete" for item in daily)
    return {
        "schema_version": "ml-overlay-forward-evaluation-v1",
        "target_basis": "frozen_0910_quote_mid_to_official_close",
        "annualization_periods": annualization_periods,
        "shadow_record_count": len(shadow_records),
        "expected_trade_date_count": len(expected_dates),
        "decision_missing_count": sum(item.get("status") == "decision_missing" for item in daily),
        "ambiguous_shadow_date_count": sum(
            item.get("status") in {
                "shadow_decision_ambiguous",
                "multiple_shadow_input_versions",
            }
            for item in daily
        ),
        "paired_complete_count": complete,
        "outcome_missing_count": sum(item.get("status") == "outcome_missing" for item in daily),
        "actual_fill_pnl_complete_count": len(actual_returns),
        "modeled_counterfactuals_are_fills": False,
        "paired_metrics": {
            "ml_enabled": _metrics(enabled_returns, annualization_periods),
            "ml_disabled": _metrics(disabled_returns, annualization_periods),
            "mean_paired_net_delta": float(np.mean(deltas)) if deltas else None,
            "paired_delta_count": len(deltas),
        },
        "candidate_actual_fill_metrics": {
            **_metrics(actual_returns, annualization_periods),
            "trade_dates": actual_return_dates,
            "source_status": "fully_reconciled_only",
        },
        "daily": daily,
        "evaluation_fingerprint": _digest(daily),
    }


__all__ = ["OUTCOME_SCHEMA", "evaluate_ml_overlay_forward"]
