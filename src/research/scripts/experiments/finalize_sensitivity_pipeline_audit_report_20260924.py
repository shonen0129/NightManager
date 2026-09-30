#!/usr/bin/env python3
"""Regenerate the sensitivity audit report from its frozen result summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research/scripts/experiments"))

import experiment_sensitivity_pipeline_audit_20260924 as audit  # noqa: E402

RESULTS_DIR = ROOT / "var/results/20260924_sensitivity_pipeline_audit"


def _display(value: float, digits: int = 4) -> float | str:
    return round(float(value), digits) if pd.notna(value) else "n/a"


def main() -> None:
    summary_path = RESULTS_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # The diagnostic is deliberately rerun after the experiment so this report
    # includes its full PIT and mapping quality-gate reasons.
    summary["direct33_quality"] = audit._direct33_data_quality()
    current = summary["variants"]["current"]
    rows = []
    for name, metric in summary["variants"].items():
        ci = metric.get("paired_net_return_delta_ci95")
        ci_display = "—" if not ci else f"[{ci[0] * 10000:+.3f}, {ci[1] * 10000:+.3f}]"
        baseline = name == "current"
        rows.append({
            "variant": name,
            "mean_abs_mu_delta_bps": _display(metric.get("mu_effect_vs_current", {}).get("mean_abs_delta_bps", 0.0 if baseline else float("nan"))),
            "mean_l1_weight_delta": _display(metric.get("mean_l1_weight_delta", 0.0), 5),
            "net_sharpe": _display(metric["net_sharpe"]),
            "net_sharpe_delta": _display(metric["net_sharpe"] - current["net_sharpe"]),
            "gross_sharpe": _display(metric["gross_sharpe"]),
            "gross_sharpe_delta": _display(metric["gross_sharpe"] - current["gross_sharpe"]),
            "max_drawdown": f"{metric['max_drawdown'] * 100:.2f}%",
            "mean_daily_turnover_one_way": _display(metric["mean_daily_turnover_one_way"], 5),
            "turnover_change": "—" if baseline else f"{metric['turnover_change_vs_current_fraction'] * 100:+.2f}%",
            "net_daily_delta_ci95_bps": ci_display,
            "fallback_rate": f"{metric['fallback_rate'] * 100:.2f}%",
            "rank_ic_dates": int(metric.get("rank_ic", {}).get("paired_dates", 0)),
            "dates_all17": int(metric.get("rank_ic", {}).get("dates_all_17", 0)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "variant_comparison.csv", index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=audit.v2_helpers._json_default),
        encoding="utf-8",
    )
    audit._write_report(summary, table)


if __name__ == "__main__":
    main()
