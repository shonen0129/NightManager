"""Operational provenance written beside live decision and close artifacts.

The manifest is deliberately a reporting-boundary object.  It records which
resolved inputs and model were used, why a fallback or overlay decision was
made, and what execution evidence was collected.  It is not used as an
authorization to submit or retry orders.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.paths import project_root
from leadlag.domain.inputs import DecisionInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.reporting.results_format import update_run_manifest

RUNTIME_MANIFEST_VERSION = "v1"


def _json_value(value: Any) -> Any:
    """Convert common runtime objects to stable JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _json_value(value.value)
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_provenance(root: Path) -> dict[str, Any]:
    """Return local revision evidence without contacting a remote."""
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--no-ext-diff", "--binary")
    staged = run("diff", "--cached", "--no-ext-diff", "--binary")
    dirty_material = "\n".join(item or "" for item in (status, diff, staged))
    return {
        "code_revision": revision,
        "dirty": bool(status),
        "dirty_diff_hash": hashlib.sha256(dirty_material.encode("utf-8")).hexdigest(),
    }


def _max_timestamp(values: list[Any]) -> str | None:
    parsed: list[pd.Timestamp] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("Asia/Tokyo").tz_localize(None)
            if not pd.isna(timestamp):
                parsed.append(timestamp)
        except (TypeError, ValueError, OverflowError):
            continue
    return max(parsed).isoformat() if parsed else None


def _observed_at(inputs: DecisionInputs) -> dict[str, str]:
    observed: dict[str, str] = dict(inputs.known.observed_at)
    observed.update(
        {
            f"price_observed_at[{ticker}]": timestamp
            for ticker, timestamp in inputs.known.price_observed_at.items()
        }
    )
    observed.update(dict(inputs.historical.observed_at))
    return observed


def _order_quantities(
    results: list[Mapping[str, Any]],
    quantity_keys: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """Aggregate requested or filled quantities without losing order side."""
    quantities: dict[str, dict[str, int]] = defaultdict(lambda: {"BUY": 0, "SELL": 0})
    for item in results:
        if not isinstance(item, Mapping):
            continue
        ticker = str(item.get("ticker") or "")
        side = str(item.get("side") or "").upper()
        if not ticker or side not in {"BUY", "SELL"}:
            continue
        value = next((item.get(key) for key in quantity_keys if item.get(key) is not None), None)
        if value is None and str(item.get("status") or "").upper() in {"FILLED", "SIMULATED"}:
            value = item.get("quantity", 0)
        try:
            quantities[ticker][side] += int(value or 0)
        except (TypeError, ValueError):
            continue
    return {ticker: dict(side_values) for ticker, side_values in quantities.items()}


def _position_exposure(path: str | Path | None) -> dict[str, Any]:
    """Summarize the signed quantity in a saved position snapshot."""
    if path is None:
        return {"snapshot_readable": False}
    try:
        with Path(path).open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
        positions = snapshot.get("positions", []) if isinstance(snapshot, Mapping) else []
        signed: dict[str, int] = defaultdict(int)
        for item in positions:
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("ticker") or "")
            if not ticker:
                continue
            try:
                quantity = int(item.get("quantity") or 0)
            except (TypeError, ValueError):
                continue
            side = str(item.get("side") or "").upper()
            signed[ticker] += -quantity if side in {"SELL", "SHORT", "売", "売建"} else quantity
        signed_values = dict(signed)
        return {
            "snapshot_readable": True,
            "signed_quantity_by_ticker": signed_values,
            "net_quantity": sum(signed_values.values()),
            "gross_quantity": sum(abs(value) for value in signed_values.values()),
            "position_count": len(positions),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"snapshot_readable": False, "snapshot_read_error": str(exc)}


def build_decision_manifest(
    *,
    app_config: Any,
    inputs: DecisionInputs,
    result: PortfolioDecision,
    config_path: str | Path,
    gap_input_dir: str | Path | None,
    model: Any,
) -> dict[str, Any]:
    """Build reproducibility and failure-reason evidence for one decision."""
    observed = _observed_at(inputs)
    overlay = getattr(model, "_overlay_model", None)
    overlay_metadata = getattr(overlay, "metadata", {}) or {}
    diagnostics = result.diagnostics or {}
    distribution = diagnostics.get("distribution_resolution", {})
    provenance = diagnostics.get("distribution_provenance", {})
    gap_metadata = provenance if isinstance(provenance, Mapping) else {}
    return {
        "operational_manifest_version": RUNTIME_MANIFEST_VERSION,
        "manifest_created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trade_date": inputs.trade_date.strftime("%Y-%m-%d"),
        "as_of": inputs.known.as_of.isoformat(),
        "max_available_at": _max_timestamp(list(observed.values())),
        "ticker_order": list(inputs.known.ticker_order),
        "input_version": {
            "schema_version": inputs.version.schema_version,
            "digest": inputs.version.digest,
            "known_fingerprint": inputs.version.known_fingerprint,
            "historical_fingerprint": inputs.version.historical_fingerprint,
            "source": inputs.version.source,
        },
        "observed_at": observed,
        "quote_snapshot": {
            "snapshot_id": inputs.known.quote_snapshot_id,
            "price_sources": _json_value(dict(inputs.known.price_sources)),
            "price_observed_at": _json_value(dict(inputs.known.price_observed_at)),
        },
        "code": _git_provenance(project_root()),
        "config": {
            "path": str(config_path),
            "resolved_hash": _digest(app_config.model_dump(mode="json")),
            "model_config_hash": _digest(app_config.v2.model_dump(mode="json")),
        },
        "model": {
            "production_version": getattr(result.run_config, "version", None),
            "gap_input_dir": str(gap_input_dir) if gap_input_dir is not None else None,
            "overlay_enabled": bool(getattr(model, "_overlay_model", None) is not None),
            "overlay_model_dir": str(getattr(result.run_config, "ml_overlay_model_dir", "") or ""),
            "overlay_artifact_version": overlay_metadata.get("artifact_version"),
            "overlay_model_sha256": overlay_metadata.get("model_sha256"),
            "overlay_training_data_hash": overlay_metadata.get("data_hash"),
        },
        "gap": {
            "source": distribution.get("source") or provenance.get("source"),
            "status": distribution.get("status") or provenance.get("status"),
            "reason": distribution.get("reason"),
            "version": (
                distribution.get("version")
                or gap_metadata.get("bundle_version")
                or gap_metadata.get("version")
                or gap_metadata.get("schema_version")
            ),
            "metadata": _json_value(result.diagnostics.get("distribution_provenance"))
            if isinstance(result.diagnostics, Mapping)
            else None,
        },
            "decision": {
                "fallback": _json_value(result.fallback),
                "fallback_reasons": sorted(str(key) for key, value in result.fallback.items() if value),
                "pit_binning": _json_value(result.pit_binning),
            "audits": {
                "leakage": _json_value(result.leakage),
                "numerical": _json_value(result.numerical),
            },
            "alerts": _json_value(result.alerts),
            "diagnostics": _json_value(result.diagnostics),
            "summary": _json_value(result.summary),
            "model_net_exposure": float(np.sum(result.w_final)),
                "model_gross_exposure": float(np.sum(np.abs(result.w_final))),
                "weights": {
                    ticker: float(weight)
                    for ticker, weight in zip(inputs.known.ticker_order, result.w_final, strict=True)
                },
                "arrays": {
                    "scores": _json_value(result.scores_overlay if result.scores_overlay is not None else result.scores),
                    "scores_base": _json_value(result.scores),
                    "mu_gap": _json_value(result.mu_gap),
                    "sigma_gap": _json_value(result.sigma_gap),
                    "Omega_gap": _json_value(result.Omega_gap),
                },
            },
        }


def update_decision_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> str:
    """Merge decision provenance into the standard run manifest."""
    return update_run_manifest(str(output_dir), {"runtime": _json_value(manifest)})


def update_execution_manifest(
    output_dir: str | Path,
    *,
    phase: str,
    summary: Mapping[str, Any],
    position_snapshot_path: str | Path | None = None,
    wallet_snapshot_path: str | Path | None = None,
) -> str:
    """Persist execution/reconciliation evidence without changing retry rules."""
    report = summary.get("execution_report")
    results = (
        list(summary.get("buy_results") or [])
        + list(summary.get("sell_results") or [])
        + list(summary.get("close_results") or [])
    )
    incomplete_report = isinstance(report, Mapping) and bool(report.get("incomplete"))
    requested_quantities = _order_quantities(results, ("quantity", "requested_quantity"))
    filled_quantities = _order_quantities(
        results,
        ("filled_quantity", "fill_quantity", "executed_quantity"),
    )
    update = {
        "execution": {
            "phase": phase,
            "run_id": summary.get("run_id"),
            "status": (
                "incomplete"
                if summary.get("close_incomplete")
                or summary.get("reconciliation_errors")
                or incomplete_report
                else "recorded"
            ),
            "execution_report": _json_value(report),
            "reconciliation_errors": _json_value(summary.get("reconciliation_errors", [])),
            "order_ids": [str(item.get("order_id")) for item in results if item.get("order_id")],
            "position_snapshot": str(position_snapshot_path) if position_snapshot_path else None,
            "wallet_snapshot": str(wallet_snapshot_path) if wallet_snapshot_path else None,
            "post_execution_exposure": {
                "position_snapshot_available": position_snapshot_path is not None,
                "observed_order_count": len(results),
                "filled_order_count": sum(
                    1 for item in results
                    if isinstance(item, Mapping) and item.get("status") in {"FILLED", "SIMULATED"}
                ),
                "pending_order_count": sum(
                    1 for item in results
                    if isinstance(item, Mapping) and item.get("status") in {"SUBMITTED", "PARTIALLY_FILLED"}
                ),
                "failed_order_count": sum(
                    1 for item in results
                    if isinstance(item, Mapping) and item.get("status") in {"FAILED", "CANCELLED", "SKIPPED"}
                ),
                "requested_quantity_by_ticker": requested_quantities,
                "filled_quantity_by_ticker": filled_quantities,
                **_position_exposure(position_snapshot_path),
            },
        }
    }
    return update_run_manifest(str(output_dir), update)


__all__ = [
    "build_decision_manifest",
    "update_decision_manifest",
    "update_execution_manifest",
]
