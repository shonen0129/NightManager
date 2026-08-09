#!/usr/bin/env python3
"""本番 V2 ロジックを使った shadow run を生成する。

`tools/validation/run_daily_residual_blpx_shadow.py` ではなく、
`src/leadlag/models/production_v2.py::generate_v2_production_portfolio_with_overlay`
を用いて各日のポートフォリオを構築し、shadow_runs/ 配下に出力する。

使い方:
    python3 scripts/experiments/build_v2_production_shadow_run.py \
        --config configs/production/production.yaml \
        --gap-input-dir live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
        --shadow-root shadow_runs/v2_production_20200106_20260729 \
        --start 2020-01-06 --end 2026-07-29
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.ml_order_overlay import (
    generate_v2_production_portfolio_with_overlay,
    load_overlay_model,
)
from leadlag.models.production_v2 import generate_v2_production_portfolio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BuildV2ProductionShadowRun")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build V2 production shadow run")
    p.add_argument("--config", default="configs/production/production.yaml")
    p.add_argument("--gap-input-dir", default="live/pipeline_data/gap_adjusted_distribution/20260731_024303")
    p.add_argument("--shadow-root", default="shadow_runs/v2_production_20200106_20260729")
    p.add_argument("--start", default="2020-01-06")
    p.add_argument("--end", default="2026-07-29")
    p.add_argument("--overlay", default="true", choices=["true", "false"],
                   help="Apply ML order overlay if enabled in config (default: true)")
    p.add_argument("--clean", default="false", choices=["true", "false"],
                   help="Remove existing shadow root before running")
    p.add_argument("--max-pit-history", type=int, default=0,
                   help="Limit PIT IR history to the latest N rows (0=unlimited, default). "
                        "This simulates live latest diagnostics history.")
    return p.parse_args()


def _rank_in_selected(score: float, selected_scores: np.ndarray) -> int:
    """Rank within selected tickers (1-based); 0 if not selected."""
    if len(selected_scores) == 0:
        return 0
    sorted_idx = np.argsort(selected_scores)
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, len(sorted_idx) + 1)
    pos = np.where(selected_scores == score)[0]
    if len(pos) == 0:
        return 0
    return int(ranks[pos[0]])


def write_daily_files(
    trade_date: str,
    output_dir: Path,
    result: dict,
) -> None:
    """Write shadow-run files compatible with monitor_residual_blpx_shadow_performance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.to_datetime(trade_date)

    w_final = result["w_final"]
    scores = result["scores"]
    mu_gap = result["mu_gap"]
    sigma_gap = result["sigma_gap"]
    Omega_gap = result["Omega_gap"]
    pit = result["pit_binning"]
    mult = float(pit["multiplier"])
    run_cfg = result["run_config"]
    cost_bps_per_gross = float(run_cfg.cost_bps_per_gross)

    len(JP_TICKERS)
    long_mask = w_final > 1e-8
    short_mask = w_final < -1e-8
    selected_mask = long_mask | short_mask
    selected_idx = np.where(selected_mask)[0]
    selected_scores = scores[selected_idx]

    target_gross = float(np.sum(np.abs(w_final)))
    target_net = float(np.sum(w_final))

    # 1. shadow_portfolios.csv
    port_records = []
    for j, tk in enumerate(JP_TICKERS):
        side = "LONG" if long_mask[j] else ("SHORT" if short_mask[j] else "NEUTRAL")
        rank = _rank_in_selected(scores[j], selected_scores) if selected_mask[j] else 0
        w_pre = w_final[j] / mult if mult != 0 else w_final[j]
        port_records.append({
            "signal_date": trade_date,
            "trade_date": trade_date,
            "candidate": "primary_ruleD",
            "ticker": tk,
            "side": side,
            "rank": rank,
            "score": float(scores[j]),
            "mu_gap": float(mu_gap[j]),
            "sigma_gap": float(sigma_gap[j]),
            "weight_pre_gross": float(w_pre),
            "gross_multiplier": mult,
            "weight_final": float(w_final[j]),
            "target_gross": target_gross,
            "target_net": target_net,
            "expected_cost_bps": abs(float(w_final[j])) * cost_bps_per_gross,
            "predicted_asset_var": float(Omega_gap[j, j]),
            "selected_flag": int(selected_mask[j]),
            "fallback_flag": int(pit.get("fallback_flag", False)),
            "timestamp_category": "POST_OPEN",
        })
    pd.DataFrame(port_records).to_csv(output_dir / "shadow_portfolios.csv", index=False)

    # 2. shadow_candidate_summary.csv
    summary = dict(result.get("summary", {}))
    summary["candidate"] = "primary_ruleD"
    summary["trade_date"] = trade_date
    summary["signal_date"] = trade_date
    pd.DataFrame([summary]).to_csv(output_dir / "shadow_candidate_summary.csv", index=False)

    # 3. shadow_scores.csv
    score_records = []
    for j, tk in enumerate(JP_TICKERS):
        score_records.append({
            "trade_date": trade_date,
            "ticker": tk,
            "mu_gap": float(mu_gap[j]),
            "sigma_gap": float(sigma_gap[j]),
            "mu_over_sigma_score": float(scores[j]),
        })
    pd.DataFrame(score_records).to_csv(output_dir / "shadow_scores.csv", index=False)

    # 4. shadow_risk_estimates.csv
    df_cov = pd.DataFrame(Omega_gap, index=JP_TICKERS, columns=JP_TICKERS)
    df_cov.to_csv(output_dir / "shadow_risk_estimates.csv")

    # 5. shadow_orders_preview.csv
    order_records = []
    for j, tk in enumerate(JP_TICKERS):
        side = "LONG" if long_mask[j] else ("SHORT" if short_mask[j] else "NEUTRAL")
        order_records.append({
            "trade_date": trade_date,
            "candidate": "primary_ruleD",
            "ticker": tk,
            "current_weight": 0.0,
            "target_weight": float(w_final[j]),
            "delta_weight": float(w_final[j]),
            "side": side,
            "note": "V2 production shadow target",
        })
    pd.DataFrame(order_records).to_csv(output_dir / "shadow_orders_preview.csv", index=False)

    # 6. audits and config
    with open(output_dir / "pit_binning_audit.json", "w") as f:
        json.dump(pit, f, indent=2, default=str)

    with open(output_dir / "leakage_audit.json", "w") as f:
        json.dump(result["leakage"], f, indent=2, default=str)

    with open(output_dir / "numerical_audit.json", "w") as f:
        json.dump(result["numerical"], f, indent=2, default=str)

    data_avail = {
        "trade_date": trade_date,
        "mu_gap_available": True,
        "Omega_gap_available": True,
        "fallback_triggered": bool(result["fallback"].get("gap_data_missing", False)),
        "alerts": result.get("alerts", []),
    }
    with open(output_dir / "data_availability.json", "w") as f:
        json.dump(data_avail, f, indent=2, default=str)

    run_config = {
        "trade_date": trade_date,
        "version": "production_residual_blpx_v2",
        "candidate": "primary_ruleD",
        "ranking_mode": "mu_over_sigma",
        "sizing_mode": "baseline_style" if not run_cfg.minvar_enabled else f"minvar_alpha_{run_cfg.minvar_alpha}",
        "gross_scaling_rule": "RuleD",
        "post_open_requirement": "Tokyo 9:10 POST_OPEN",
        "slippage_bps_per_side": 5.0,
        "cost_bps_per_gross": cost_bps_per_gross,
        "target_gross": target_gross,
        "target_net": target_net,
        "pit_bin": pit.get("assigned_bin", "Medium"),
        "pit_multiplier": mult,
        "pit_history_count": pit.get("history_count", 0),
        "overlay_applied": int(summary.get("overlay_applied", 0)),
        "p_trade_mean": summary.get("p_trade_mean", None),
        "p_trade_std": summary.get("p_trade_std", None),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)


def build_shadow_run(args: argparse.Namespace) -> int:
    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    gap_input_dir = ROOT / args.gap_input_dir
    if not gap_input_dir.exists():
        logger.error("Gap input dir not found: %s", gap_input_dir)
        return 1

    shadow_root = ROOT / args.shadow_root
    if args.clean == "true" and shadow_root.exists():
        shutil.rmtree(shadow_root)
    shadow_root.mkdir(parents=True, exist_ok=True)

    # Resolve trade dates from the long-form panel
    long_csv = gap_input_dir / "gap_adjusted_distribution_long.csv"
    if not long_csv.exists():
        logger.error("Long panel not found: %s", long_csv)
        return 1
    df_long = pd.read_csv(long_csv)
    df_long["trade_date"] = pd.to_datetime(df_long["trade_date"]).dt.strftime("%Y-%m-%d")
    all_dates = sorted(df_long["trade_date"].unique())
    dates = [d for d in all_dates if args.start <= d <= args.end]
    logger.info("Resolved %d trade dates between %s and %s", len(dates), args.start, args.end)

    # Load overlay model and df_exec if enabled
    overlay_model = None
    df_exec = None
    overlay_cfg = cfg.get("ml_order_overlay", {})
    if args.overlay == "true" and overlay_cfg.get("enabled", False):
        model_dir = overlay_cfg.get("model_dir", "models/ml_order_overlay/phase2_8")
        model_path = ROOT / model_dir
        if model_path.exists():
            try:
                overlay_model = load_overlay_model(model_path)
                df_exec = load_df_exec_from_local_cache()
                logger.info("ML overlay loaded from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load overlay model or df_exec: %s", e)
        else:
            logger.warning("Overlay model dir not found: %s", model_path)

    # Optimize _derive_signal_date to avoid per-call directory glob
    # Precompute signal_date from the long panel (sig_date column)
    import leadlag.models.production_v2 as pv2
    sig_map = (
        df_long.groupby("trade_date")["signal_date"]
        .first()
        .to_dict()
    )
    original_derive_signal_date = pv2._derive_signal_date

    def fast_derive_signal_date(gap_input_dir: Path | None, trade_date: str) -> str:
        if trade_date in sig_map:
            return pd.to_datetime(sig_map[trade_date]).strftime("%Y-%m-%d")
        return original_derive_signal_date(gap_input_dir, trade_date)

    pv2._derive_signal_date = fast_derive_signal_date

    # Optionally limit PIT history to simulate live latest diagnostics
    if args.max_pit_history > 0:
        original_load_pit_ir_history = pv2.load_pit_ir_history

        def limited_load_pit_ir_history(gap_input_dir: Path, trade_date: str):
            history_ir, alerts, history_trade_dates = original_load_pit_ir_history(gap_input_dir, trade_date)
            if len(history_ir) > args.max_pit_history:
                history_ir = history_ir[-args.max_pit_history:]
                history_trade_dates = history_trade_dates[-args.max_pit_history:]
                alerts.append(f"PIT history truncated to {args.max_pit_history} rows for live alignment")
            return history_ir, alerts, history_trade_dates

        pv2.load_pit_ir_history = limited_load_pit_ir_history

    # Build shadow portfolios day by day
    for i, trade_date in enumerate(dates, 1):
        try:
            if overlay_model is not None and df_exec is not None:
                result = generate_v2_production_portfolio_with_overlay(
                    trade_date=trade_date,
                    gap_input_dir=gap_input_dir,
                    cfg=cfg,
                    df_exec=df_exec,
                    overlay_model=overlay_model,
                )
            else:
                result = generate_v2_production_portfolio(
                    trade_date=trade_date,
                    gap_input_dir=gap_input_dir,
                    cfg=cfg,
                )

            out_dir = shadow_root / trade_date.replace("-", "")
            write_daily_files(trade_date, out_dir, result)

            if i % 100 == 0 or i == len(dates):
                logger.info("Progress: %d/%d days written", i, len(dates))
        except Exception as e:
            logger.error("[%s] Failed to build shadow portfolio: %s", trade_date, e)
            # Write a minimal fallback record so the run continues
            out_dir = shadow_root / trade_date.replace("-", "")
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "build_error.json", "w") as f:
                json.dump({"trade_date": trade_date, "error": str(e)}, f, indent=2)

    logger.info("V2 production shadow run written to %s", shadow_root)
    return 0


if __name__ == "__main__":
    sys.exit(build_shadow_run(parse_args()))
