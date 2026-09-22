"""Collect local, read-only evidence for the remaining structural gates."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.state_store import ExecutionStateStore


def command_output(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
    }


def main() -> None:
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    state = ExecutionStateStore(ROOT / "var/live/pipeline_data/execution/execution_state.sqlite")
    artifact_root = ROOT / "var/results/20260920_structural_completion/ml_overlay_retrained"
    current = (artifact_root / "CURRENT").read_text(encoding="utf-8").strip()
    metadata = json.loads(
        (artifact_root / "versions" / current / "metadata.json").read_text(encoding="utf-8")
    )
    gap_root = ROOT / "reports/20260922_structural_completion/gap_regeneration"
    metadata_files = sorted(gap_root.glob("*/matrices/gap_metadata*.json"))
    gap_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
    required_horizons = {1, 3, 5}
    gap_checks = {
        "files": len(gap_metadata),
        "horizons": sorted({int(item["horizon"]) for item in gap_metadata}),
        "all_have_observed_at": all("observed_at" in item for item in gap_metadata),
        "all_have_label_availability": all("label_available_at" in item for item in gap_metadata),
        "all_have_calculation_as_of": all("calculation_as_of" in item for item in gap_metadata),
        "all_have_history_fingerprint": all(
            bool(item.get("historical_inputs_fingerprint")) for item in gap_metadata
        ),
    }
    shadow_scorecard = pd.read_csv(
        ROOT / "var/results/shadow_monitor_v2_production_overlay/monitoring_scorecard.csv"
    ).to_dict(orient="records")
    services = [
        "com.leadlag.update-market-data",
        "com.leadlag.distribution-diagnostics",
        "com.leadlag.decision",
        "com.leadlag.close",
        "com.leadlag.pnl_report",
    ]
    scheduler = [command_output(["launchctl", "print", f"gui/501/{service}"]) for service in services]
    scheduler_checks = []
    for item in scheduler:
        stdout = str(item["stdout"])
        scheduler_checks.append(
            {
                "service": str(item["command"][-1]).rsplit("/", 1)[-1],
                "registered": item["returncode"] == 0,
                "workspace_path_current": f"working directory = {ROOT}" in stdout,
                "last_exit_code": (
                    int(match.group(1))
                    if (match := re.search(r"last exit code = ([-0-9]+)", stdout))
                    else None
                ),
            }
        )
    result = {
        "collected_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "resolved_config": {
            "ml_overlay_enabled": cfg.v2.ml_overlay_enabled,
            "mh_horizons": list(cfg.v2.mh_horizons),
            "fallback_on_audit_failure": cfg.v2.fallback_on_audit_failure,
        },
        "gap_regeneration": gap_checks,
        "gap_regeneration_complete": (
            set(gap_checks["horizons"]) == required_horizons
            and gap_checks["all_have_observed_at"]
            and gap_checks["all_have_label_availability"]
            and gap_checks["all_have_calculation_as_of"]
            and gap_checks["all_have_history_fingerprint"]
        ),
        "artifact": {
            "current": current,
            "metadata_status": metadata.get("metadata_status"),
            "train_start": metadata.get("train_start"),
            "train_end": metadata.get("train_end"),
            "data_hash": metadata.get("data_hash"),
            "config_hash": metadata.get("config_hash"),
            "oos_evidence_in_metadata": any(
                key in metadata for key in ("oos_report", "oos_report_hash", "walk_forward_oos")
            ),
        },
        "existing_shadow_oos": shadow_scorecard,
        "execution_state": {
            "recovery_candidates": len(state.list_recovery_candidates()),
        },
        "scheduler": scheduler,
        "scheduler_checks": scheduler_checks,
        "hosted_ci": {
            "workflow": ".github/workflows/ci.yml",
            "gh_available": subprocess.run(["sh", "-c", "command -v gh"], capture_output=True).returncode == 0,
            "run_url": None,
        },
    }
    output = Path(__file__).parent / "acceptance_local.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({
        "gap_regeneration_complete": result["gap_regeneration_complete"],
        "artifact_train_start": metadata.get("train_start"),
        "recovery_candidates": result["execution_state"]["recovery_candidates"],
        "scheduler_services": sum(item["returncode"] == 0 for item in scheduler),
        "scheduler_total": len(scheduler),
        "hosted_ci_run_url": None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
