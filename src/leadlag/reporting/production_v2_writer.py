"""Production v2 file output writer.

Writes all daily production artifacts to the *live* directory:

  - latest_weights.csv           — primary production weights
  - v1_baseline_weights.csv      — v1 Residual-BLPX weights (fallback / comparison)
  - production_scores.csv        — mu_over_sigma scores for all 17 tickers
  - production_summary.csv       — one-row summary statistics
  - pit_binning.json             — PIT binning audit details
  - leakage_audit.json           — lookahead leakage audit
  - numerical_audit.json         — numerical boundary audit
  - production_audit.json        — aggregated audit (all_passed flag)
  - daily_production_report.md   — human-readable daily report
  - run_config.json              — execution metadata
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.models.production_v2 import VERSION

logger = logging.getLogger(__name__)


def write_production_files(
    trade_date: str,
    live_dir: Path,
    result: dict,
    dry_run: bool = False,
) -> None:
    """Write all production output files to *live_dir*.

    Args:
        trade_date: Trade execution date (e.g. '2026-06-16').
        live_dir: Target directory.  Created if absent.
        result: Return value of ``generate_v2_production_portfolio()``.
        dry_run: When True, log a summary to stdout but do not write files.
    """
    if dry_run:
        logger.info("[DRY-RUN] Would write files to: %s", live_dir)
        _print_dry_run_summary(trade_date, result)
        return

    live_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = result["run_config"]
    cost_bps_per_gross = run_cfg.cost_bps_per_gross
    w_final = result["w_final"]
    w_v1 = result["w_v1"]
    scores = result["scores"]
    mu_gap = result["mu_gap"]
    sigma_gap = result["sigma_gap"]
    pit = result["pit_binning"]

    # 1. latest_weights.csv
    rows = []
    for j, tk in enumerate(JP_TICKERS):
        side = "LONG" if w_final[j] > 1e-8 else ("SHORT" if w_final[j] < -1e-8 else "NEUTRAL")
        rows.append({
            "trade_date": trade_date,
            "ticker": tk,
            "weight": float(w_final[j]),
            "side": side,
            "score": float(scores[j]),
            "mu_gap": float(mu_gap[j]),
            "sigma_gap": float(sigma_gap[j]),
            "ensemble_signal": float(scores[j]),  # backward compat with shadow runner
            "gross_multiplier": float(pit["multiplier"]),
            "pit_bin": pit["assigned_bin"],
            "version": VERSION,
            "fallback_flag": int(result["fallback"]["v1_fallback_used"]),
        })
    pd.DataFrame(rows).to_csv(live_dir / "latest_weights.csv", index=False)
    logger.info("Written: latest_weights.csv  (gross=%.4f)", float(np.sum(np.abs(w_final))))

    # 2. v1_baseline_weights.csv - REMOVED: circular reference issue
    # V1 fallback is no longer supported; gap data missing results in flat position (w_final=0)

    # 3. production_scores.csv
    score_rows = [
        {
            "trade_date": trade_date,
            "ticker": JP_TICKERS[j],
            "mu_gap": float(mu_gap[j]),
            "sigma_gap": float(sigma_gap[j]),
            "mu_over_sigma_score": float(scores[j]),
        }
        for j in range(len(JP_TICKERS))
    ]
    pd.DataFrame(score_rows).to_csv(live_dir / "production_scores.csv", index=False)
    logger.info("Written: production_scores.csv")

    # 4. production_summary.csv
    pd.DataFrame([result["summary"]]).to_csv(live_dir / "production_summary.csv", index=False)
    logger.info("Written: production_summary.csv")

    # 5. JSON audit files
    with open(live_dir / "pit_binning.json", "w") as f:
        json.dump(result["pit_binning"], f, indent=4, default=str)

    with open(live_dir / "leakage_audit.json", "w") as f:
        json.dump(result["leakage"], f, indent=4)

    with open(live_dir / "numerical_audit.json", "w") as f:
        json.dump(result["numerical"], f, indent=4)

    all_passed = (
        result["leakage"]["status"] == "PASSED"
        and result["numerical"]["status"] == "PASSED"
    )
    production_audit = {
        "trade_date": trade_date,
        "version": VERSION,
        "all_passed": all_passed,
        "leakage_status": result["leakage"]["status"],
        "numerical_status": result["numerical"]["status"],
        "fallback_triggered": result["fallback"]["v1_fallback_used"],
        "alerts": result["alerts"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(live_dir / "production_audit.json", "w") as f:
        json.dump(production_audit, f, indent=4)
    logger.info("Written: production_audit.json  (all_passed=%s)", all_passed)

    # 6. run_config.json
    run_config = {
        "trade_date": trade_date,
        "version": VERSION,
        "candidate": "primary_ruleD",
        "ranking_mode": "mu_over_sigma",
        "sizing_mode": "baseline_style",
        "gross_scaling_rule": "RuleD",
        "post_open_requirement": "Tokyo 9:10 POST_OPEN",
        "slippage_bps_per_side": 5.0,
        "cost_bps_per_gross": cost_bps_per_gross,
        "timestamp": datetime.now().isoformat(),
    }
    with open(live_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    # 7. daily_production_report.md
    _write_daily_report(trade_date, live_dir, result)
    logger.info("Written: daily_production_report.md")


def _print_dry_run_summary(trade_date: str, result: dict) -> None:
    """Log a dry-run summary to stdout."""
    pit = result["pit_binning"]
    s = result["summary"]
    logger.info("=== DRY-RUN SUMMARY: %s ===", trade_date)
    logger.info("  Candidate     : primary_ruleD (v2)")
    logger.info("  PIT Bin       : %s (mult=%.2f)", pit["assigned_bin"], pit["multiplier"])
    logger.info("  Target Gross  : %.4f", s["target_gross"])
    logger.info("  Target Net    : %.6f", s["target_net"])
    logger.info("  Ex-Ante IR    : %.4f", s["predicted_portfolio_ir"])
    logger.info("  Fallback      : %s", result["fallback"]["v1_fallback_used"])
    logger.info("  Leakage Audit : %s", result["leakage"]["status"])
    logger.info("  Numerical Audit: %s", result["numerical"]["status"])
    logger.info("  Alerts        : %s", result["alerts"])

    w = result["w_final"]
    long_tks = [JP_TICKERS[i] for i in range(len(JP_TICKERS)) if w[i] > 1e-8]
    short_tks = [JP_TICKERS[i] for i in range(len(JP_TICKERS)) if w[i] < -1e-8]
    logger.info("  Longs  (%d): %s", len(long_tks), long_tks)
    logger.info("  Shorts (%d): %s", len(short_tks), short_tks)
    logger.info("=========================")


def _write_daily_report(trade_date: str, live_dir: Path, result: dict) -> None:
    """Write the human-readable daily production report in Markdown."""
    pit = result["pit_binning"]
    s = result["summary"]
    w = result["w_final"]
    fb = result["fallback"]

    long_weights = [(JP_TICKERS[i], w[i]) for i in range(len(JP_TICKERS)) if w[i] > 1e-8]
    short_weights = [(JP_TICKERS[i], w[i]) for i in range(len(JP_TICKERS)) if w[i] < -1e-8]
    long_weights.sort(key=lambda x: -x[1])
    short_weights.sort(key=lambda x: x[1])

    fallback_note = ""
    if fb["v1_fallback_used"]:
        fallback_note = (
            "\n\n> [!WARNING]\n> **フォールバック発動**: gap data 未利用。"
            "v1 Residual-BLPX ウェイトを使用しています。\n"
        )

    alert_text = ""
    if result["alerts"]:
        alert_text = "\n## Alerts\n" + "\n".join(f"- {a}" for a in result["alerts"]) + "\n"

    def _fmt_thresh(v: float) -> str:
        return f"{v:.4f}" if (isinstance(v, float) and v == v) else "N/A"

    thresh_lo = _fmt_thresh(pit["threshold_low"])
    thresh_hi = _fmt_thresh(pit["threshold_high"])

    rep = f"""# Production Daily Report — {trade_date}

**Version**: `{VERSION}` | **Candidate**: `primary_ruleD`
**Ranking**: `mu_over_sigma` | **Sizing**: `baseline_style` | **Gross**: `RuleD`
**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")}
{fallback_note}
## 1. Portfolio Summary

| Metric | Value |
|---|---|
| Target Gross | {s['target_gross']:.4f} |
| Target Net | {s['target_net']:.6f} |
| Gross Multiplier (RuleD) | {s['gross_multiplier']:.2f} |
| PIT Bin | **{pit['assigned_bin']}** |
| Predicted Portfolio IR | {s['predicted_portfolio_ir']:.4f} |
| Expected Cost (bps) | {s['expected_cost_bps']:.1f} |
| Longs | {s['long_count']} |
| Shorts | {s['short_count']} |
| Fallback Triggered | {"YES ⚠️" if fb['v1_fallback_used'] else "No"} |

## 2. RuleD Dynamic Gross Binning

| Item | Value |
|---|---|
| Current Ex-Ante IR | {pit.get('current_ir', 0.0):.4f} |
| Assigned Bin | **{pit['assigned_bin']}** |
| Threshold Low (33rd pct) | {thresh_lo} |
| Threshold High (67th pct) | {thresh_hi} |
| Multiplier | {pit['multiplier']:.2f} |
| History Days | {pit['history_count']} |

## 3. Selected Positions

**Longs:**
| Ticker | Weight |
|---|---:|
"""
    for tk, wt in long_weights:
        rep += f"| {tk} | {wt:.4f} |\n"

    rep += "\n**Shorts:**\n| Ticker | Weight |\n|---|---:|\n"
    for tk, wt in short_weights:
        rep += f"| {tk} | {wt:.4f} |\n"

    rep += f"""
## 4. Safety Audit Status

| Audit | Status |
|---|---|
| Leakage Audit | **{result['leakage']['status']}** |
| Numerical Audit | **{result['numerical']['status']}** |
{alert_text}
---
*This file is generated automatically by `run_daily_production_v2.py`.
No trades are placed by this script; it writes weight targets only.*
"""

    with open(live_dir / "daily_production_report.md", "w", encoding="utf-8") as f:
        f.write(rep)
