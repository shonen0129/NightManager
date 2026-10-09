"""Reprice frozen research weights under inventory-v3 without broker or data refresh."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.pnl import simulate_daily_pnl
from leadlag.core.risk import compute_var_es
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtest import _save_detailed_backtest_results
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint

OLD_COMMIT = "b5b901e6d467fb000263f9178722e3954d652720"
ON = ROOT / "reports/20260923_profitability_order_5/v2_current_overlay_oos"
OFF = ROOT / "reports/20260923_profitability_order_6/ml_off"
OUTPUT = ROOT / "reports/20261008_issue33_inventory/revaluation"


def _metrics(pnl: dict[str, Any], dates: pd.DatetimeIndex, risk: Any) -> dict[str, Any]:
    net = pd.Series(pnl["net_returns"], index=dates)
    gross = pd.Series(pnl["gross_returns"], index=dates)
    if not np.isfinite(net).all() or not np.isfinite(gross).all():
        raise ValueError("Cannot evaluate incomplete active price intervals")
    metrics = calculate_metrics(net, spec=MetricsSpec(annualization_periods=252))
    gross_metrics = calculate_metrics(gross, spec=MetricsSpec(annualization_periods=252))
    tail = compute_var_es(
        net, confidence=risk.var_confidence, window=risk.var_window, var_method=risk.var_method
    )
    return {
        "days": len(dates),
        "net_sharpe_annual_252": float(metrics["Sharpe"]),
        "gross_sharpe_annual_252": float(gross_metrics["Sharpe"]),
        "net_compounded_return": float((1 + net).prod() - 1),
        "max_drawdown": float(metrics["MDD"]),
        "mean_turnover": float(np.mean(pnl["turnover"])),
        "mean_execution_volume": float(np.mean(pnl["execution_volume"]))
        if "execution_volume" in pnl
        else None,
        "mean_target_weight_turnover": float(np.mean(pnl["target_weight_turnover"]))
        if "target_weight_turnover" in pnl
        else float(np.mean(pnl["turnover"])),
        "cost_sums_return_fraction": {
            key: float(np.sum(pnl[key]))
            for key in ("slip_costs", "financing_costs", "borrow_costs", "reverse_costs", "costs")
        },
        "carry_gap_sum": float(np.sum(pnl["carry_gap_returns"]))
        if "carry_gap_returns" in pnl
        else None,
        "carry_morning_sum": float(np.sum(pnl["carry_open_910_returns"]))
        if "carry_open_910_returns" in pnl
        else None,
        "var_available": tail.available,
        "var_samples": tail.samples,
        "var99": float(tail.var_loss) if tail.available else None,
        "es99": float(tail.es_loss) if tail.available else None,
        "es_tail_count": tail.tail_count,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_csv(args.df_exec, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    # Keep the absence of measured 09:10 evidence explicit. No implicit cache
    # adapter or network fallback is allowed in this retrospective repricing.
    morning = (
        pd.read_csv(args.open_910, index_col=0, parse_dates=True).reindex(columns=JP_TICKERS)
        if args.open_910
        else pd.DataFrame(np.nan, index=frame.index, columns=JP_TICKERS)
    )
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    costs = app.v2.costs
    common = dict(
        slip=float(costs.slippage_bps_per_side) / 10000,
        financing_daily=float(costs.buy_interest_annual) / 365,
        borrow_daily=float(costs.borrow_fee_annual) / 365,
        reverse_daily=float(costs.reverse_fee_bps) / 10000,
        alpha_long=float(costs.overnight_alpha_long),
        alpha_short=float(costs.overnight_alpha_short),
    )
    source_weights = {
        "baseline_ml_on": pd.read_csv(ON / "daily_weights.csv", index_col=0, parse_dates=True),
        "ml_off": pd.read_csv(OFF / "daily_weights.csv", index_col=0, parse_dates=True),
    }
    dates = pd.DatetimeIndex(source_weights["baseline_ml_on"].index)
    if not dates.equals(source_weights["ml_off"].index) or dates.min() < pd.Timestamp("2015-01-05"):
        raise ValueError("Paired weights must have the same post-prior dates")
    if not dates.isin(frame.index).all():
        raise ValueError("Explicit price snapshot must cover every requested day")
    if "is_provisional" in frame and frame.loc[dates, "is_provisional"].astype(bool).any():
        raise ValueError("Cannot price provisional close rows")
    target, gaps, moves = BacktestEngine._compute_price_intervals(
        frame, frame.index, dates, morning
    )
    args.output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "status": "FROZEN_WEIGHT_REPRICING_NOT_MODEL_RERUN_OR_ADOPTION",
        "old_commit": OLD_COMMIT,
        "new_contract": "inventory-v3",
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "df_exec": str(args.df_exec),
        "df_exec_sha256": hashlib.sha256(args.df_exec.read_bytes()).hexdigest(),
        "frame_fingerprint": dataframe_fingerprint(frame),
        "open_910_input": str(args.open_910) if args.open_910 else None,
        "measured_morning_cells": int(
            morning.reindex(index=dates, columns=JP_TICKERS).notna().sum().sum()
        ),
        "entry_proxy": "09:10_when_measured_else_open",
        "resolved_costs": costs.model_dump(mode="json"),
        "resolved_risk": app.risk.model_dump(mode="json"),
        "comparisons": {},
        "limitations": [
            "Reuses frozen model weights; no decision, numerical or leakage audit rerun is claimed.",
            "Bundled prices are a different snapshot from the missing historical run-owned inputs; old-vs-new below uses identical supplied prices and costs.",
            "Without an open_910 input, all entry marks use open proxies; actual morning contribution is unmeasured, not evidence of zero morning movement.",
            "Historical 269-day source pickles and prices through 2026-09-25 are missing; that exact VaR/ES and versioned ML paired comparison remains pending.",
            "Not observed account PnL: five-minute midpoint optimism, actual fills and all broker fee reconciliation remain unverified.",
            "Existing cost/leverage choices are repriced, not a new parameter search or OOS adoption decision.",
        ],
    }
    daily = pd.DataFrame(index=dates)
    old_source = subprocess.run(
        ["git", "show", f"{OLD_COMMIT}:src/leadlag/core/pnl.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="issue33-old-accounting-") as tmp:
        old_path = Path(tmp) / "old_pnl.py"
        old_path.write_text(old_source)
        spec = importlib.util.spec_from_file_location("issue33_old_pnl", old_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for label, weights_frame in source_weights.items():
            if list(weights_frame.columns) != JP_TICKERS:
                raise ValueError("Canonical ticker order required")
            weights = weights_frame.to_numpy(dtype=float)
            model_gross = np.abs(weights).sum(axis=1)
            model_net = weights.sum(axis=1)
            if model_gross.max() > 2 + 1e-6 or np.abs(model_net).max() > 0.05 + 1e-6:
                raise ValueError("Stored weights violate model constraints")
            source_dir = ON if label == "baseline_ml_on" else OFF
            flags = (
                pd.read_csv(source_dir / "daily_fallback.csv", index_col=0, parse_dates=True)
                .iloc[:, 0]
                .astype(bool)
            )
            if not flags.index.equals(dates):
                raise ValueError("Fallback dates do not align with frozen weights")
            for leverage in (1.5, 1.3):
                key = f"{label}_side_{leverage}"
                inputs = dict(
                    weights=weights,
                    target_returns=target,
                    gap_returns=gaps,
                    sim_dates=dates,
                    side_leverage=leverage,
                    **common,
                )
                old = module.simulate_daily_pnl(**inputs)
                new = simulate_daily_pnl(**inputs, open_910_returns=moves)
                no_carry_inputs = dict(inputs, alpha_long=0.0, alpha_short=0.0)
                no_carry = simulate_daily_pnl(**no_carry_inputs, open_910_returns=moves)
                np.testing.assert_allclose(
                    np.asarray(new["gross_returns"]) - new["costs"], new["net_returns"], atol=1e-14
                )
                np.testing.assert_allclose(
                    np.cumprod(1 + np.asarray(new["net_returns"])), new["equity"], rtol=1e-12
                )
                summary["comparisons"][key] = {
                    "old_inventory_v1": _metrics(old, dates, app.risk),
                    "inventory_v3": _metrics(new, dates, app.risk),
                    "no_carry_inventory_v3": _metrics(no_carry, dates, app.risk),
                    "model_gross_max": float(model_gross.max()),
                    "model_abs_net_max": float(np.abs(model_net).max()),
                    "effective_gross_max": float(model_gross.max() * leverage),
                    "effective_abs_net_max": float(np.abs(model_net).max() * leverage),
                    "within_resolved_risk_gross_cap": bool(
                        model_gross.max() * leverage <= app.risk.max_gross_exposure + 1e-6
                    ),
                    "saved_fallback_rate": float(flags.mean()),
                    "terminal_inventory": new["terminal_inventory"],
                    "max_abs_daily_net_delta_same_inputs": float(
                        np.max(np.abs(np.asarray(new["net_returns"]) - old["net_returns"]))
                    ),
                }
                daily[f"{key}_old"] = old["net_returns"]
                daily[f"{key}_new"] = new["net_returns"]
                daily[f"{key}_delta"] = daily[f"{key}_new"] - daily[f"{key}_old"]
                daily[f"{key}_no_carry"] = no_carry["net_returns"]
                result = BacktestEngine._assemble_v2_results(
                    new,
                    weights_frame,
                    flags.to_numpy(),
                    [],
                    dates,
                    common["alpha_long"],
                    common["alpha_short"],
                    leverage,
                )
                _save_detailed_backtest_results(result, args.output / key)
            saved_net = pd.read_csv(
                source_dir / "daily_net_returns.csv", index_col=0, parse_dates=True
            ).iloc[:, 0]
            daily[f"{label}_saved_historical_net"] = saved_net
    daily["paired_new_on_minus_off_side_1.3"] = (
        daily["baseline_ml_on_side_1.3_new"] - daily["ml_off_side_1.3_new"]
    )
    summary["paired_1.3_mean_daily_net_delta"] = float(
        daily["paired_new_on_minus_off_side_1.3"].mean()
    )
    daily.to_csv(args.output / "daily_comparison.csv", index_label="trade_date")
    record = ExperimentRecord(
        name="issue33_inventory_accounting_repricing",
        hypothesis="Inventory-v3 completes receiving-day carry, terminal cash and execution flows for frozen historical weights.",
        parameters={
            "old_commit": OLD_COMMIT,
            "frame_sha256": summary["df_exec_sha256"],
            "costs": summary["resolved_costs"],
        },
        metrics={"comparisons": summary["comparisons"], "new_contract": "inventory-v3"},
        decision=Decision.PENDING,
        report_path="reports/20261008_issue33_inventory/report.md",
        study_id="issue33-inventory-repricing-20261008",
    )
    registry_path = ROOT / "var/experiments/registry.jsonl"
    ExperimentRegistry(registry_path).record(record)
    summary["registry_path"] = str(registry_path.relative_to(ROOT))
    summary["registry_record_id"] = record.record_id
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(
        json.dumps(
            {"output": str(args.output), "comparisons": summary["comparisons"]},
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--df-exec", type=Path, default=ROOT / "tests/regression/baselines/df_exec_20260814.csv.gz"
    )
    parser.add_argument(
        "--open-910",
        type=Path,
        help="Optional immutable date×canonical-ticker measured open→09:10 returns",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
