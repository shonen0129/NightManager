"""Verify full decision-field parity without rewriting acceptance artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "reports/20260922_production_acceptance"
WORK = ROOT / "var/results/20260922_production_acceptance"
INPUT = WORK / "inputs"
GAP = INPUT / "gap"
OUTPUT = ROOT / "reports/20260923_profitability_order_1/field_parity.json"

sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.pit_lake import PITDataLake  # noqa: E402
from leadlag.domain.inputs import DecisionInputs  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.models.ml_overlay_artifact import load_overlay_model  # noqa: E402
from leadlag.runner.production import ProductionRunner  # noqa: E402
from research.scripts.experiments.structural_artifact_acceptance import (  # noqa: E402
    resources,
)
from research.scripts.experiments.structural_artifact_evaluate import (  # noqa: E402
    apply_candidate,
    load,
)


def max_error(left: Any, right: Any) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def main() -> int:
    frame, history, snapshot_v2 = resources()
    current_app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    normalized_current = current_app.v2.model_copy(
        deep=True,
        update={"ml_overlay_model_dir": snapshot_v2.ml_overlay_model_dir},
    )
    if normalized_current != snapshot_v2:
        raise AssertionError("Current V2 config differs from the fixed acceptance config beyond artifact pointer")

    model_dir = WORK / "artifacts" / "evaluation_2025"
    candidate_v2 = snapshot_v2.model_copy(
        deep=True,
        update={"ml_overlay_enabled": True, "ml_overlay_model_dir": str(model_dir)},
    )
    candidate_app = current_app.model_copy(deep=True, update={"v2": candidate_v2})
    model = load_overlay_model(model_dir)
    runner = ProductionRunner(candidate_app)

    chunks = [load(WORK / "collected" / f"chunk_{index}.pkl") for index in range(4)]
    base = {key: value for chunk in chunks for key, value in chunk["decisions"].items()}
    parity_payload = json.loads((ACCEPTANCE / "artifact_parity.json").read_text(encoding="utf-8"))
    dates = [pd.Timestamp(row["date"]) for row in parity_payload]
    candidate = apply_candidate(base, dates, frame, history, model)
    lake = PITDataLake(frame)
    rows: list[dict[str, Any]] = []
    for date in dates:
        as_of = date + pd.Timedelta(hours=9, minutes=10)
        snapshot = lake.get_execution_snapshot(as_of, history.open_910_returns)
        known = snapshot.to_known_inputs(
            sig_date=frame.loc[date, "sig_date"],
            source="fixed_acceptance_field_parity",
        )
        actual = runner.run(
            DecisionInputs(
                known=known,
                historical=history,
                gap_input_dir=GAP,
                use_file_cache=True,
            )
        )
        expected = candidate[date]
        actual_scores = actual.scores_overlay if actual.scores_overlay is not None else actual.scores
        expected_scores = expected.scores_overlay if expected.scores_overlay is not None else expected.scores
        row = {
            "date": str(date.date()),
            "max_weight_error": max_error(actual.w_final, expected.w_final),
            "max_score_error": max_error(actual_scores, expected_scores),
            "max_mu_error": max_error(actual.mu_gap, expected.mu_gap),
            "max_sigma_error": max_error(actual.sigma_gap, expected.sigma_gap),
            "max_omega_error": max_error(actual.Omega_gap, expected.Omega_gap),
            "pit_equal": json.dumps(actual.pit_binning, sort_keys=True, default=str)
            == json.dumps(expected.pit_binning, sort_keys=True, default=str),
        }
        rows.append(row)

    checks = {
        "weights": all(row["max_weight_error"] <= 1e-10 for row in rows),
        "scores": all(row["max_score_error"] <= 1e-10 for row in rows),
        "mu": all(row["max_mu_error"] <= 1e-10 for row in rows),
        "sigma": all(row["max_sigma_error"] <= 1e-10 for row in rows),
        "omega": all(row["max_omega_error"] <= 1e-10 for row in rows),
        "pit": all(row["pit_equal"] for row in rows),
    }
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model_dir": str(model_dir.relative_to(ROOT)),
        "dates": rows,
        "checks": checks,
        "note": "Read-only replay; acceptance artifact files were not rewritten.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
