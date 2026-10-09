"""Evaluate a predeclared turnover-aware execution transform.

This experiment operates on the fixed, already accepted production-artifact
weight path.  It does not alter the production configuration or recompute a
signal.  The only research hypothesis is that limiting the raw-weight L1
change between consecutive targets can preserve alpha while reducing
turnover-driven costs.

Selection is performed on 2020--2024.  The 2025--2026-08-14 period is held
out and is never used to select the cap.  All evaluation dates are retained,
including dates where the source result is flat (if any).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.pnl import simulate_daily_pnl
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import (
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics, compute_drawdown_series

SOURCE_RESULT = ROOT / "var/results/20260922_production_acceptance/evaluation/aggregate.pkl"
REPORT_DIR = ROOT / "reports/20260923_turnover_aware_no_trade"
REPORT_PATH = REPORT_DIR / "report.md"
RESULTS_PATH = REPORT_DIR / "results.json"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"

# Predeclared central value 1.5 with a symmetric local sensitivity check.
CAPS = (1.0, 1.5, 2.0)
SELECTION_START = pd.Timestamp("2020-01-06")
SELECTION_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END = pd.Timestamp("2026-08-14")
ANNUAL_FACTOR = 245
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SOURCE_RESULT.exists():
        raise FileNotFoundError(f"fixed acceptance result is missing: {SOURCE_RESULT}")
    with SOURCE_RESULT.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "candidate" not in payload:
        raise ValueError("fixed acceptance result has no candidate result")
    candidate = payload["candidate"]
    weights = candidate.get("weights")
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("candidate weights must be a DataFrame")
    if list(weights.columns) != list(JP_TICKERS):
        raise ValueError("candidate ticker order does not match JP_TICKERS")
    if not isinstance(weights.index, pd.DatetimeIndex):
        weights = weights.copy()
        weights.index = pd.to_datetime(weights.index)
    weights = weights.astype(float)
    stored_returns = candidate["daily_returns"].astype(float)
    if not stored_returns.index.equals(weights.index):
        raise ValueError("candidate returns and weights have different date indexes")
    return weights, stored_returns


def _apply_turnover_cap(targets: pd.DataFrame, max_l1_change: float) -> pd.DataFrame:
    """Move toward each target by at most a raw-weight L1 amount per day."""
    if max_l1_change <= 0.0 or not np.isfinite(max_l1_change):
        raise ValueError("max_l1_change must be finite and positive")
    target_arr = targets.to_numpy(dtype=float, copy=True)
    output = np.zeros_like(target_arr)
    previous = np.zeros(target_arr.shape[1], dtype=float)
    for row, target in enumerate(target_arr):
        delta = target - previous
        l1_change = float(np.abs(delta).sum())
        fraction = 1.0 if l1_change <= max_l1_change else max_l1_change / l1_change
        current = previous + fraction * delta
        # Both endpoints are market-neutral.  This check catches accidental
        # future edits to the transform instead of silently repairing them.
        if abs(float(current.sum())) > 1e-10:
            raise AssertionError("turnover transform broke market neutrality")
        if float(np.abs(current).sum()) > 2.0 + 1e-10:
            raise AssertionError("turnover transform broke model gross limit")
        output[row] = current
        previous = current
    return pd.DataFrame(output, index=targets.index, columns=targets.columns)


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    net = pd.Series(result["daily_returns"], dtype=float)
    gross = pd.Series(result["daily_returns_gross"], dtype=float)
    costs = pd.Series(result["daily_costs"], dtype=float)
    components = sum(
        pd.Series(result[f"daily_{part}_costs"], dtype=float)
        for part in ("slip", "financing", "borrow", "reverse")
    )
    if not np.isfinite(np.concatenate([net.to_numpy(), gross.to_numpy(), costs.to_numpy()])).all():
        raise ValueError("non-finite daily P&L")
    np.testing.assert_allclose(gross - costs, net, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(components, costs, atol=1e-14, rtol=0.0)
    spec = MetricsSpec(frequency="daily", annualization_periods=ANNUAL_FACTOR, include_flat_days=True)
    net_m = calculate_metrics(net, spec=spec)
    gross_m = calculate_metrics(gross, spec=spec)
    drawdown = compute_drawdown_series(net)
    weights = result["weights"]
    return {
        "days": int(len(net)),
        "start": str(net.index[0].date()),
        "end": str(net.index[-1].date()),
        "net_sharpe": float(net_m["Sharpe"]),
        "gross_sharpe": float(gross_m["Sharpe"]),
        "annualized_net_return": float(net_m["AR"]),
        "max_drawdown": float(drawdown.min()),
        "mean_turnover": float(pd.Series(result["daily_turnover"], dtype=float).mean()),
        "mean_gross_exposure": float(pd.Series(result["daily_gross_exps"], dtype=float).mean()),
        "net_cumulative_return": float(np.prod(1.0 + net.to_numpy()) - 1.0),
        "cost_period_sum_return_fraction": {
            part: float(pd.Series(result[f"daily_{part}_costs"], dtype=float).sum())
            for part in ("slip", "financing", "borrow", "reverse")
        },
        "total_cost_period_sum_return_fraction": float(costs.sum()),
        "max_abs_model_net": float(np.abs(weights.sum(axis=1)).max()),
        "max_model_gross": float(np.abs(weights).sum(axis=1).max()),
        "flat_days": int((np.abs(weights).sum(axis=1) < 1e-12).sum()),
    }


def _simulate(weights: pd.DataFrame, df_exec: pd.DataFrame, app_config: Any) -> dict[str, Any]:
    dates = pd.DatetimeIndex(weights.index)
    if not dates.isin(df_exec.index).all():
        missing = dates[~dates.isin(df_exec.index)]
        raise ValueError(f"source weights contain dates missing from df_exec: {missing[:3].tolist()}")
    full_dates = pd.DatetimeIndex(df_exec.index)
    open_910 = build_open_910_returns(df_exec, JP_TICKERS)
    targets, gaps, morning_returns = BacktestEngine._compute_price_intervals(
        df_exec, full_dates, dates, open_910_returns=open_910
    )
    params = BacktestEngine._resolve_v2_backtest_cost_params(app_config, *([None] * 7))
    pnl = simulate_daily_pnl(
        weights=weights.to_numpy(dtype=float),
        target_returns=targets,
        open_910_returns=morning_returns,
        gap_returns=gaps,
        sim_dates=dates,
        slip=float(params["slip_bps"]) / 10000.0,
        financing_daily=float(params["fin_annual"]) / 365.0,
        borrow_daily=float(params["borrow_annual"]) / 365.0,
        reverse_daily=float(params["rev_bps"]) / 10000.0,
        alpha_long=float(params["alpha_long"]),
        alpha_short=float(params["alpha_short"]),
        side_leverage=float(params["side_leverage"]),
    )
    return BacktestEngine._assemble_v2_results(
        pnl,
        weights,
        np.zeros(len(dates), dtype=bool),
        [{"trade_date": str(date.date())} for date in dates],
        dates,
        float(params["alpha_long"]),
        float(params["alpha_short"]),
        float(params["side_leverage"]),
    )


def _slice_result(result: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    mask = (result["weights"].index >= start) & (result["weights"].index <= end)
    sliced: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, (pd.Series, pd.DataFrame)) and len(value) == len(result["weights"]):
            sliced[key] = value.loc[mask]
        elif key == "v2_summaries":
            sliced[key] = [item for item, include in zip(value, mask) if include]
        else:
            sliced[key] = value
    return sliced


def _bootstrap_delta(base: pd.Series, candidate: pd.Series) -> dict[str, Any]:
    delta = (candidate - base).to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = int(np.ceil(len(delta) / BOOTSTRAP_BLOCK))
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, len(delta), blocks)
        indices = np.concatenate([(start + np.arange(BOOTSTRAP_BLOCK)) % len(delta) for start in starts])[: len(delta)]
        samples.append(float(delta[indices].mean()))
    return {
        "mean_daily_net_delta": float(delta.mean()),
        "block_days": BOOTSTRAP_BLOCK,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_daily_net_delta_ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _write_report(output: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    selected = output["variants"][output["selected_variant"]]
    lines = [
        "# Turnover-aware no-trade 研究結果",
        "",
        f"実行日: {output['run_at']}。本番設定・本番artifactは変更していない。",
        "",
        "## 仮説と設計",
        "",
        "現行の accepted production artifact の target weight を信号として固定し、"
        "前日実効weightから当日targetへの raw-weight L1 変更量を上限化すると、alphaを大きく失わずに "
        "slippage と turnover を減らせるかを検証した。候補は事前固定した cap=1.0/1.5/2.0。",
        "選択期間は2020-01-06〜2024-12-31、holdoutは2025-01-01〜2026-08-14。全営業日を含み、"
        "flat日を除外していない。",
        "",
        "## 集計結果（全期間）",
        "",
        "| variant | net Sharpe | gross Sharpe | max DD | mean turnover | total cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["baseline", *[f"cap_{cap:.1f}" for cap in CAPS]]:
        item = output["variants"][name]
        lines.append(
            f"| {name} | {item['net_sharpe']:.4f} | {item['gross_sharpe']:.4f} | "
            f"{item['max_drawdown']:.2%} | {item['mean_turnover']:.4f} | "
            f"{item['total_cost_period_sum_return_fraction']:.6f} |"
        )
    lines.extend([
        "",
        "## 期間別比較",
        "",
        "| period | baseline SR | selected SR | ΔSR | baseline DD | selected DD | Δ turnover |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for period, fold in output["folds"].items():
        if period == "holdout":
            continue
        b = fold["baseline"]
        c = fold[output["selected_variant"]]
        lines.append(
            f"| {period} | {b['net_sharpe']:.4f} | {c['net_sharpe']:.4f} | "
            f"{c['net_sharpe'] - b['net_sharpe']:+.4f} | {b['max_drawdown']:.2%} | "
            f"{c['max_drawdown']:.2%} | {c['mean_turnover'] - b['mean_turnover']:+.4f} |"
        )
    lines.extend([
        "",
        "### Holdout sensitivity (2025-01-01〜2026-08-14)",
        "",
        "| variant | net Sharpe | max DD | mean turnover | mean daily Δ vs baseline | bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    holdout_fold = output["folds"]["holdout"]
    for name in ["baseline", *[f"cap_{cap:.1f}" for cap in CAPS]]:
        item = holdout_fold[name]
        bootstrap = holdout_fold["paired_block_bootstrap"][name]
        ci = bootstrap["mean_daily_net_delta_ci95"]
        lines.append(
            f"| {name} | {item['net_sharpe']:.4f} | {item['max_drawdown']:.2%} | "
            f"{item['mean_turnover']:.4f} | {bootstrap['mean_daily_net_delta']:+.8f} | "
            f"[{ci[0]:+.8f}, {ci[1]:+.8f}] |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
        f"選択候補: `{output['selected_variant']}`。選択期間の事前ゲートは `{output['selection_pass']}`、"
        f"未使用 holdout の確認は `{output['holdout_pass']}`、総合判定は **{output['decision']}**。",
        "",
        "採用条件は、選択期間で baseline 以上の net Sharpe・max DD と turnover低下を満たし、"
        "holdoutでも net Sharpe と max DD を悪化させないこと。bootstrap区間は平均日次差の不確実性であり、"
        "Sharpe改善の有意性を意味しない。DSRは候補4試行（baselineを含む）の選択期間Sharpeに対する補正値であり、"
        "過去の未登録試行を完全に数えたものではない。",
        "",
        "### コスト内訳（selected / 全期間）",
        "",
    ])
    for part, amount in selected["cost_period_sum_return_fraction"].items():
        lines.append(f"- {part}: {amount:.8f}")
    lines.extend([
        "",
        f"再現結果: `{RESULTS_PATH.relative_to(ROOT)}`。実験registryにも結果を追記した。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    weights, stored_returns = _load_source()
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        raise RuntimeError("df_exec cache is unavailable")
    app_config = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)

    base_result = _simulate(weights, df_exec, app_config)
    base_diff = float(np.max(np.abs(base_result["daily_returns"].to_numpy() - stored_returns.to_numpy())))
    if base_diff > 1e-10:
        raise RuntimeError(
            f"current canonical input does not reproduce the accepted baseline: max daily return diff={base_diff:.3e}"
        )

    results: dict[str, dict[str, Any]] = {"baseline": base_result}
    for cap in CAPS:
        candidate_weights = _apply_turnover_cap(weights, cap)
        results[f"cap_{cap:.1f}"] = _simulate(candidate_weights, df_exec, app_config)

    selection = {
        "baseline": _metrics(_slice_result(results["baseline"], SELECTION_START, SELECTION_END)),
    }
    for cap in CAPS:
        name = f"cap_{cap:.1f}"
        selection[name] = _metrics(_slice_result(results[name], SELECTION_START, SELECTION_END))
    eligible = [
        name for name in selection if name != "baseline"
        and selection[name]["net_sharpe"] >= selection["baseline"]["net_sharpe"]
        and selection[name]["max_drawdown"] >= selection["baseline"]["max_drawdown"]
        and selection[name]["mean_turnover"] < selection["baseline"]["mean_turnover"]
    ]
    selected_variant = max(eligible, key=lambda name: selection[name]["net_sharpe"]) if eligible else "baseline"
    trial_sharpes = [selection[name]["net_sharpe"] for name in selection]
    dsr = compute_deflated_sharpe({
        "net_sharpe": selection[selected_variant]["net_sharpe"],
        "net_sharpe_frequency": "annual",
        "trading_days_per_year": ANNUAL_FACTOR,
        "trials": len(trial_sharpes),
        "n_observations": selection[selected_variant]["days"],
        "returns": _slice_result(results[selected_variant], SELECTION_START, SELECTION_END)["daily_returns"].tolist(),
        "trial_sharpes": trial_sharpes,
    })

    holdout_slices = {
        name: _slice_result(result, HOLDOUT_START, HOLDOUT_END)
        for name, result in results.items()
    }
    holdout_base = holdout_slices["baseline"]
    holdout_selected = holdout_slices[selected_variant]
    holdout_metrics = {name: _metrics(item) for name, item in holdout_slices.items()}
    holdout_bootstraps = {
        name: _bootstrap_delta(
            holdout_slices["baseline"]["daily_returns"], holdout_slices[name]["daily_returns"]
        )
        for name in results
    }
    selection_pass = bool(selected_variant != "baseline")
    holdout_pass = bool(
        holdout_selected["daily_returns"].mean() >= holdout_base["daily_returns"].mean()
        and holdout_selected["drawdown"].min() >= holdout_base["drawdown"].min()
    )
    decision = "ADOPTED" if selection_pass and holdout_pass else "REJECTED"

    output: dict[str, Any] = {
        "run_at": datetime.now(UTC).isoformat(),
        "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
        "source_result_sha256": _digest_file(SOURCE_RESULT),
        "df_exec_end": str(df_exec.index.max().date()),
        "evaluation_start": str(weights.index.min().date()),
        "evaluation_end": str(weights.index.max().date()),
        "candidate_caps_raw_weight_l1": list(CAPS),
        "annual_factor": ANNUAL_FACTOR,
        "all_dates_included": True,
        "base_reproduction_max_daily_return_error": base_diff,
        "selected_variant": selected_variant,
        "selection_pass": selection_pass,
        "holdout_pass": holdout_pass,
        "decision": decision,
        "selection_dsr": dsr,
        "trial_count": len(trial_sharpes),
        "trial_sharpes_selection": trial_sharpes,
        "variants": {name: _metrics(result) for name, result in results.items()},
        "folds": {
            "selection_2020_2024": selection,
            "holdout": {
                **holdout_metrics,
                "paired_block_bootstrap": holdout_bootstraps,
            },
        },
    }
    _write_report(output)

    record = ExperimentRecord(
        "20260923_turnover_aware_no_trade",
        "A predeclared raw-weight turnover cap preserves net alpha while reducing cost and turnover.",
        end_time=datetime.now(UTC),
        parameters={
            "source_result": str(SOURCE_RESULT.relative_to(ROOT)),
            "source_result_sha256": _digest_file(SOURCE_RESULT),
            "caps_raw_weight_l1": list(CAPS),
            "selection_period": [str(SELECTION_START.date()), str(SELECTION_END.date())],
            "holdout_period": [str(HOLDOUT_START.date()), str(HOLDOUT_END.date())],
            "bootstrap": {"block_days": BOOTSTRAP_BLOCK, "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        },
        metrics={
            "net_sharpe": output["folds"]["holdout"][selected_variant]["net_sharpe"],
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": ANNUAL_FACTOR,
            "n_observations": output["folds"]["holdout"][selected_variant]["days"],
            "returns": _slice_result(results[selected_variant], HOLDOUT_START, HOLDOUT_END)["daily_returns"].tolist(),
            "trial_sharpes": trial_sharpes,
            "trials": len(trial_sharpes),
            "selection_dsr": dsr,
            "holdout_pass": holdout_pass,
            "selection_pass": selection_pass,
            "base_reproduction_max_daily_return_error": base_diff,
        },
        decision=Decision.ADOPTED if decision == "ADOPTED" else Decision.REJECTED,
        report_path=str(REPORT_PATH.relative_to(ROOT)),
    )
    ExperimentRegistry(REGISTRY_PATH).record(record)
    print(json.dumps({
        "decision": decision,
        "selected_variant": selected_variant,
        "selection_pass": selection_pass,
        "holdout_pass": holdout_pass,
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
