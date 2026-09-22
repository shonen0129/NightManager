"""Capture the pure gap-calculation outputs used by the research command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/research"))
import compute_gap_adjusted_distribution as gap_script


def main() -> None:
    original = gap_script.compute_gap_distribution
    original_attach = gap_script.attach_topix_trade_returns
    calls: list[dict[str, object]] = []

    def capture_attached_frame(*args, **kwargs):  # type: ignore[no-untyped-def]
        frame = original_attach(*args, **kwargs)
        frame.to_pickle("var/results/20260920_structural_completion/r3_capture_gap/exact_df_exec.pkl")
        return frame

    def wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = original(*args, **kwargs)
        calls.append({
            "mu_raw_first": float(result.mu_raw[0]),
            "mu_gap_first": float(result.mu_gap[0]),
            "omega_diag_first": float(result.omega_gap[0, 0]),
            "gap_open_coef": float(kwargs["gap_open_coef"]),
            "topix_beta_coef": float(kwargs["topix_beta_coef"]),
            "gap_first": float(kwargs["gap_override"][0]),
            "beta_first": float(kwargs["betas_t"][0]),
            "topix_night": float(kwargs["topix_night_t"]),
            "mu_raw": np.asarray(result.mu_raw, dtype=float).tolist(),
            "mu_gap": np.asarray(result.mu_gap, dtype=float).tolist(),
            "gap_override": np.asarray(kwargs["gap_override"], dtype=float).tolist(),
            "betas_t": np.asarray(kwargs["betas_t"], dtype=float).tolist(),
        })
        return result

    gap_script.compute_gap_distribution = wrapped
    gap_script.attach_topix_trade_returns = capture_attached_frame
    sys.argv = [
        "compute_gap_adjusted_distribution.py",
        "--distribution-input-dir", "var/results/distribution_diagnostics/20260708_184812",
        "--validation-input-dir", "var/results/distribution_validation/20260614_235836",
        "--vol-state-panel", "var/results/vol_state_diagnostics/20260614_115821/state_panel.csv",
        "--results-dir", "var/live/pipeline_data/diagnostics_weights",
        "--output-dir", "var/results/20260920_structural_completion/r3_capture_gap",
        "--start", "2026-08-14",
        "--end", "2026-08-17",
        "--save-to-gap-store", "false",
        "--save-daily-matrices", "true",
        "--save-multi-horizon", "true",
        "--save-rank-reversal", "false",
        "--n-jobs", "1",
    ]
    code = gap_script.main()
    out = Path("reports/20260920_structural_completion/evidence/r3_captured_gap_calls.json")
    out.write_text(json.dumps({"return_code": code, "calls": calls}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"return_code": code, "calls": calls}, indent=2))


if __name__ == "__main__":
    main()
