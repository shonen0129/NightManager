"""Evidence-only historical search index and append-only DSR review.

Reports are documents, not independent trials. Index entries never manufacture
ExperimentRecords; family totals are conservative lower bounds, not sums of
possibly overlapping reports. Past provenance that is unavailable stays unknown.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from leadlag.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)

# Named candidates and reported counts are transcribed from these exact reports.
# The 25 older sensitivity attempts have no individual provenance here.
EVIDENCE = {
    "reports/20260924_sensitivity_pipeline_audit_long/report.md": {
        "family": "historical:sensitivity",
        "reported_trials_lower_bound": 35,
        "candidate_ids": [
            "all_w3_w6_zero",
            "drop_w3",
            "drop_w4",
            "drop_w5",
            "drop_w6",
            "user_jpx33_pitcap_annual",
            "legacy_jpx33_pitcap_annual",
            "rate_w6_us_only",
            "rate_w6_jp_only",
            "rate_w6_both",
        ],
        "limitation": "10 named candidates; report says 25 older related trials; periods/targets differ. Corrections are not new trials.",
    },
    "reports/20260923_profitability_order_11/report.md": {
        "family": "historical:ml-order",
        "reported_trials_lower_bound": 6,
        "candidate_ids": [
            f"{cost}:{variant}"
            for cost in ("10bps", "12.5bps", "15bps")
            for variant in ("baseline", "gated")
        ],
        "limitation": "Six same-run cost/variant cases; report explicitly says earlier unregistered searches and viewed OOS are unknown.",
    },
    "reports/ml_order_decision/phase2_5_walkforward/phase2_5_walkforward_report.md": {
        "family": "historical:ml-order",
        "reported_trials_lower_bound": 12,
        "candidate_ids": [],
        "limitation": "Reported DSR uses 12 trials; individual attempted/aborted candidates and historical hashes are unknown.",
    },
}


def _family(path: str) -> str:
    if "sensitivity" in path or "jpx33" in path:
        return "historical:sensitivity"
    if any(word in path for word in ("ml_order", "ml_overlay", "lgbm", "profitability_order")):
        return "historical:ml-order"
    return "historical:unclassified"


def build_report_index(root: Path) -> dict[str, Any]:
    """Index every historical Markdown report with source hashes and unknowns."""
    entries = []
    for path in sorted((root / "reports").rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("reports/20261008_study_governance/"):
            continue
        text = path.read_text(encoding="utf-8")
        evidence = EVIDENCE.get(relative, {})
        entries.append(
            {
                "report_path": relative,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "title": next(
                    (line.lstrip("# ") for line in text.splitlines() if line.startswith("# ")),
                    path.stem,
                ),
                "study_id": evidence.get("family", _family(relative)),
                "registration_status": "retrospective_index",
                "trial_count_status": "lower_bound",
                "reported_trials_lower_bound": evidence.get("reported_trials_lower_bound", 0),
                "candidate_ids": evidence.get("candidate_ids", []),
                "unknown": [
                    "complete_trial_count",
                    "aborted_unreported_trials",
                    "selection_timestamp",
                    "historical_code_hash",
                    "historical_config_hash",
                    "historical_data_hash",
                    "target_cost_schema",
                    "is_oos_purge",
                ],
                "limitation": evidence.get(
                    "limitation",
                    "Report indexed; candidate identity/overlap and historical run provenance unreviewed.",
                ),
            }
        )
    families = []
    for family in sorted({entry["study_id"] for entry in entries}):
        members = [e for e in entries if e["study_id"] == family]
        families.append(
            {
                "study_id": family,
                "trial_count_status": "lower_bound",
                "trials_lower_bound": max(e["reported_trials_lower_bound"] for e in members),
                "complete_trial_count": "unknown",
                "report_count": len(members),
                "aggregation": "max reported lower bound; overlapping reports are not added",
            }
        )
    return {"schema": "historical-study-index-v1", "families": families, "reports": entries}


def review_registry(registry: ExperimentRegistry, *, write: bool = False) -> dict[str, Any]:
    """Identify individual stored DSRs; optionally append idempotent corrections.

    A valid numerical result without complete search provenance is unverified,
    not declared mathematically wrong. Missing registry files are never empty
    successful audits. The current view avoids reviewing an earlier correction twice.
    """
    if not registry.path.exists():
        return {"registry_status": "unavailable", "reviewed": 0, "corrections": [], "appended": 0}
    source_hash = hashlib.sha256(registry.path.read_bytes()).hexdigest()
    corrections = []
    for rec in registry.iter_current_records():
        stored = rec.metrics.get("deflated_sharpe")
        if stored is None:
            continue
        metrics = dict(rec.metrics)
        reasons = []
        numerical = compute_deflated_sharpe(metrics)
        if numerical is None:
            reasons.append("invalid_or_missing_dsr_inputs")
        else:
            try:
                differs = not np.isfinite(float(stored)) or not np.isclose(
                    float(stored), numerical, rtol=1e-10, atol=1e-14
                )
            except (TypeError, ValueError, OverflowError):
                differs = True
            if differs:
                reasons.append("stored_value_differs")
        complete = False
        if rec.study_id and rec.trial_id:
            try:
                study = registry.get_study(rec.study_id)
                started = registry.get_trial(rec.trial_id)
                expected_count = (
                    registry.count_trials(study_id=rec.study_id)
                    + study["historical_trials_lower_bound"]
                )
                complete = (
                    study["history_complete"]
                    and started["study_id"] == rec.study_id
                    and started["candidate_id"] == rec.name
                    and metrics.get("trials") == expected_count
                )
            except KeyError:
                pass
        if not complete:
            reasons.append("unverified_search_history")
            metrics["trial_count_status"] = "lower_bound"
        if rec.metric_schema_version != metrics.get("metric_schema_version"):
            reasons.append("record_metric_schema_disagrees")
        if not reasons:
            continue
        updated = numerical if complete and reasons == ["stored_value_differs"] else None
        metrics["deflated_sharpe"] = updated
        metrics["dsr_review"] = {
            "status": "recalculated" if updated is not None else "unverified_or_invalid",
            "previous_value": stored,
            "reasons": reasons,
            "source_record_id": rec.record_id,
        }
        payload = rec.to_dict()
        payload["metrics"] = metrics
        payload["record_id"] = (
            "dsr-review:"
            + hashlib.sha256(
                json.dumps(
                    {"source": rec.record_id, "review": metrics["dsr_review"]}, sort_keys=True
                ).encode()
            ).hexdigest()
        )
        corrected = ExperimentRecord.from_dict(payload)
        corrections.append(
            {
                "record_id": rec.record_id,
                "name": rec.name,
                "reasons": reasons,
                "previous_value": stored,
                "recalculated_value": updated,
            }
        )
        if write:
            registry.record_correction(rec.record_id, corrected)
    return {
        "registry_status": "reviewed",
        "source_sha256": source_hash,
        "reviewed": len(list(registry.iter_current_records())),
        "corrections": corrections,
        "appended": len(corrections) if write else 0,
    }
