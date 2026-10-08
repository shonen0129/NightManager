"""Publish the cutoff-safe overlay and its explicit VaR history schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.config import load_config_from_yaml
from leadlag.models.ml_overlay_artifact import load_overlay_model
from leadlag.runner.production import ProductionRunner

SOURCE = ROOT / "var/results/20260927_ml_overlay_var_history_2025/artifact"
PRODUCTION = ROOT / "models/ml_order_overlay/production_20260923"
REPORT = ROOT / "reports/20260927_ml_overlay_var_history"
NEW_VERSION = "20260926T192555935698Z-ee306a32f3ec"
PREVIOUS_VALID_VERSION = "20260922T011723828589Z-7e1a81ab2617"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def publish() -> None:
    candidate = load_overlay_model(SOURCE)
    if candidate.metadata.get("artifact_version") != NEW_VERSION:
        raise ValueError("Candidate CURRENT does not match the reviewed cutoff artifact")
    if candidate.metadata.get("train_end") != "2025-12-30":
        raise ValueError("Candidate training boundary must remain 2025-12-30")
    history_report = json.loads((REPORT / "history.json").read_text(encoding="utf-8"))
    history_return_path = ROOT / "var/results/20260927_ml_overlay_var_history_2025/history/versioned_var_returns.pkl"
    if history_report.get("status") != "created_and_audited":
        raise ValueError("The versioned return history has not passed its audit gate")
    if history_report.get("history_end") != "2026-09-25" or history_report.get("var_window") != 250:
        raise ValueError("The reviewed VaR history does not cover the expected latest window")
    if int(history_report.get("fallback_days", -1)) != 0:
        raise ValueError("VaR history contains fallback days")
    retrospective_post_cutoff_days = history_report.get("new_artifact_dates")
    if retrospective_post_cutoff_days is None:
        retrospective_post_cutoff_days = history_report.get(
            "cutoff_2025_artifact", {}
        ).get("applied_days")
    if retrospective_post_cutoff_days is None:
        raise ValueError("History report does not include the rebuilt artifact's applied-day count")
    if not history_return_path.is_file():
        raise FileNotFoundError(history_return_path)

    source_version_dir = SOURCE / "versions" / NEW_VERSION
    production_version_dir = PRODUCTION / "versions" / NEW_VERSION
    if production_version_dir.exists():
        if (
            _sha256(source_version_dir / "model.pkl") != _sha256(production_version_dir / "model.pkl")
            or _sha256(source_version_dir / "metadata.json") != _sha256(production_version_dir / "metadata.json")
        ):
            raise ValueError("Production version directory exists with different artifact bytes")
    else:
        shutil.copytree(source_version_dir, production_version_dir)

    # The two-model schedule is point-in-time strict. The old artifact covers
    # only dates after its own 2024-12-20 cutoff; the rebuilt one starts after
    # its 2025-12-30 label cutoff and is open-ended for future dates.
    history_manifest = {
        "schema_version": 1,
        "purpose": "train-end-safe overlay selection for VaR historical replay",
        "history_through": history_report["history_end"],
        "segments": [
            {
                "start_date": "2024-12-21",
                "end_date": "2025-12-30",
                "version": PREVIOUS_VALID_VERSION,
            },
            {
                "start_date": "2025-12-31",
                "end_date": None,
                "version": NEW_VERSION,
            },
        ],
    }
    manifest_path = PRODUCTION / "HISTORY.json"
    manifest_created = False
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != history_manifest:
            raise ValueError(f"Refusing to replace a different existing history schedule: {manifest_path}")
    else:
        _atomic_text(manifest_path, json.dumps(history_manifest, indent=2, ensure_ascii=False) + "\n")
        manifest_created = True

    current_path = PRODUCTION / "CURRENT"
    current_pointer_value = current_path.read_text(encoding="utf-8").strip()
    prior_promotion_path = PRODUCTION / "promotions/20260927_operator_directed_promotion.json"
    if not prior_promotion_path.is_file():
        raise FileNotFoundError(prior_promotion_path)
    prior_promotion = json.loads(prior_promotion_path.read_text(encoding="utf-8"))
    active_before = (
        str(prior_promotion["active_version_after"])
        if current_pointer_value == NEW_VERSION
        else current_pointer_value
    )
    pointer_changed = current_pointer_value != NEW_VERSION
    if pointer_changed:
        _atomic_text(current_path, NEW_VERSION + "\n")
    try:
        active = load_overlay_model(PRODUCTION)
        if active.metadata.get("artifact_version") != NEW_VERSION or active.metadata.get("model_sha256") != candidate.metadata.get("model_sha256"):
            raise ValueError("Production loader did not load the exact published artifact")
        app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
        runner = ProductionRunner(app)
        if not runner._overlay_enabled or runner.model._overlay_model.metadata.get("artifact_version") != NEW_VERSION:
            raise ValueError("Resolved ProductionRunner did not load the rebuilt overlay")
    except Exception:
        if pointer_changed:
            _atomic_text(current_path, active_before + "\n")
        if manifest_created:
            manifest_path.unlink(missing_ok=True)
        raise

    now_jst = datetime.now(UTC).astimezone(ZoneInfo("Asia/Tokyo"))
    promotion = {
        "decision": "published_by_user-directed_cutoff_rebuild_for_var_history",
        "promotion_date_jst": now_jst.isoformat(),
        "approval": "User instructed: recreate the new artifact using data through end-2025, then create history; prior session explicitly authorized production activation.",
        "production_root": str(PRODUCTION.relative_to(ROOT)),
        "source_artifact_root": str(SOURCE.relative_to(ROOT)),
        "active_version_before": active_before,
        "active_version_after": NEW_VERSION,
        "rollback_version": active_before,
        "artifact_train_end": candidate.metadata["train_end"],
        "artifact_model_sha256": candidate.metadata["model_sha256"],
        "history_manifest": str(manifest_path.relative_to(ROOT)),
        "history_manifest_sha256": _sha256(manifest_path),
        "history_report": str((REPORT / "history.json").relative_to(ROOT)),
        "history_return_artifact": str(history_return_path.relative_to(ROOT)),
        "history_days": history_report["history_days"],
        "history_end": history_report["history_end"],
        "history_fallback_days": history_report["fallback_days"],
        "history_numerical_audit": history_report["numerical_audit"],
        "history_leakage_audit": history_report["leakage_audit"],
        "original_promotion_gate_passed": False,
        "operator_override": True,
        "gate_status": {
            "prospective_forward_labels_since_rebuild": 0,
            "retrospective_post_cutoff_days": retrospective_post_cutoff_days,
            "retrospective_results_are_independent_oos": False,
            "historical_provider_available_at_proven": False,
            "numerical_promotion_criteria_pass": False,
            "current_250d_var_breach": history_report["var_es"]["var_loss"] >= app.risk.var_stop,
            "current_250d_es_breach": history_report["var_es"]["es_loss"] >= app.risk.es_stop,
        },
        "production_runner_verified": True,
        "broker_adapter_created": False,
        "production_orders_sent": 0,
    }
    promotion_path = PRODUCTION / "promotions/20260927_var_history_cutoff_rebuild.json"
    _atomic_text(promotion_path, json.dumps(promotion, indent=2, ensure_ascii=False, default=str) + "\n")

    # Keep the existing promotion ledger's history, updating only its active
    # version summary to the newly selected immutable artifact.
    summary_path = PRODUCTION / "PROMOTION.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "decision": promotion["decision"],
            "promotion_date_jst": promotion["promotion_date_jst"],
            "approval": promotion["approval"],
            "active_version_before": active_before,
            "active_version_after": NEW_VERSION,
            "rollback_version": active_before,
            "model_sha256": candidate.metadata["model_sha256"],
            "source_artifact_root": str(SOURCE.relative_to(ROOT)),
            "published_root": str(PRODUCTION.relative_to(ROOT)),
            "operator_override": True,
            "original_promotion_gate_passed": False,
            "numerical_promotion_criteria_pass": False,
            "gate_status": promotion["gate_status"],
            "promotion_record": str(promotion_path.relative_to(ROOT)),
            "history_manifest": str(manifest_path.relative_to(ROOT)),
            "history_end": history_report["history_end"],
            "history_days": history_report["history_days"],
            "risk_stop_after_history": promotion["gate_status"],
        }
    )
    _atomic_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    _atomic_text(
        REPORT / "publication.json",
        json.dumps(promotion, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    print(json.dumps(promotion, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["publish"])
    parser.parse_args()
    publish()
