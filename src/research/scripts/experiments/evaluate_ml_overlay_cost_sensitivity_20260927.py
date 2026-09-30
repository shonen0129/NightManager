#!/usr/bin/env python3
"""Reprice the fixed ML-on/off OOS weights under predetermined slip stresses.

This is a cost-model sensitivity analysis over existing daily V2 outputs. It
does not re-predict, retrain, execute orders, or treat modeled costs as fills.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry

ON_DIR = ROOT / "reports/20260923_profitability_order_5/v2_current_overlay_oos"
OFF_DIR = ROOT / "reports/20260923_profitability_order_6/ml_off"
OUTPUT_DIR = ROOT / "reports/20260927_ml_overlay_cost_sensitivity"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
SLIPPAGE_STRESSES_BPS = (5.0, 10.0, 20.0)
BASELINE_SLIPPAGE_BPS = 5.0
SIDE_LEVERAGE = 1.5
ALPHA_LONG = 0.75
ALPHA_SHORT = 0.50
BLOCK_DAYS = 20
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20260924
ANNUALIZATION_DAYS = 252.0


def _load_series(directory: Path, filename: str, column: str) -> pd.Series:
    frame = pd.read_csv(directory / filename, index_col=0, parse_dates=True)
    if column not in frame.columns:
        raise ValueError(f"Missing {column!r} in {directory / filename}")
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    values.index = pd.DatetimeIndex(values.index).normalize()
    if values.index.has_duplicates or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Invalid or duplicate daily values in {directory / filename}")
    return values


def _load_weights(directory: Path) -> pd.DataFrame:
    values = pd.read_csv(directory / "daily_weights.csv", index_col=0, parse_dates=True)
    values.index = pd.DatetimeIndex(values.index).normalize()
    values = values.apply(pd.to_numeric, errors="raise").astype(float)
    if values.index.has_duplicates or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Invalid or duplicate weights in {directory / 'daily_weights.csv'}")
    return values


def _slippage_cost_per_bp(weights: pd.DataFrame) -> pd.Series:
    """Mirror simulate_daily_pnl's open/close turnover slippage basis."""
    if weights.empty:
        raise ValueError("Daily weights are empty")
    previous_held = np.zeros(weights.shape[1], dtype=float)
    costs: list[float] = []
    for row in weights.to_numpy(dtype=float):
        carry = np.where(row > 0.0, ALPHA_LONG, np.where(row < 0.0, ALPHA_SHORT, 0.0)) * row
        opening_trade = float(np.abs(row - previous_held).sum())
        closing_trade = float(np.abs(row - carry).sum())
        costs.append(SIDE_LEVERAGE * (opening_trade + closing_trade) / 10_000.0)
        previous_held = carry
    return pd.Series(costs, index=weights.index, dtype=float)


def _assert_same_index(named: dict[str, pd.Series | pd.DataFrame]) -> pd.DatetimeIndex:
    index = next(iter(named.values())).index
    for name, values in named.items():
        if not values.index.equals(index):
            raise ValueError(f"Date index mismatch: {name}")
    return pd.DatetimeIndex(index)


def _metrics(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    stdev = float(array.std(ddof=1))
    wealth = np.cumprod(1.0 + array)
    peaks = np.maximum.accumulate(wealth)
    return {
        "n": len(array),
        "mean_daily": mean,
        "annualized_return": mean * ANNUALIZATION_DAYS,
        "annualized_volatility": stdev * np.sqrt(ANNUALIZATION_DAYS),
        "net_sharpe": mean / stdev * np.sqrt(ANNUALIZATION_DAYS) if stdev > 1e-12 else 0.0,
        "sum_return": float(array.sum()),
        "max_drawdown": float(np.min(wealth / peaks - 1.0)),
    }


def _bootstrap(delta: pd.Series) -> dict[str, Any]:
    values = delta.to_numpy(dtype=float)
    if len(values) < BLOCK_DAYS:
        raise ValueError(f"Need at least {BLOCK_DAYS} days for block bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks_per_sample = int(np.ceil(len(values) / BLOCK_DAYS))
    eligible_starts = len(values) - BLOCK_DAYS + 1
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for sample in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, eligible_starts, blocks_per_sample)
        indices = np.concatenate(
            [np.arange(start, start + BLOCK_DAYS) for start in starts]
        )[: len(values)]
        means[sample] = float(values[indices].mean())
    return {
        "block_days": BLOCK_DAYS,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "ci95_mean_daily_delta": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _run() -> dict[str, Any]:
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    configured_slippage = float(app.v2.costs.slippage_bps_per_side)
    if not np.isclose(configured_slippage, BASELINE_SLIPPAGE_BPS):
        raise ValueError(
            "The stored OOS outputs use a 5 bps baseline; resolved production slippage "
            f"is {configured_slippage} bps. Refusing to reprice mismatched inputs."
        )
    if not np.isclose(float(app.strategy.side_leverage), SIDE_LEVERAGE):
        raise ValueError("Resolved side leverage differs from the stored OOS assumptions")
    if not np.isclose(float(app.v2.costs.overnight_alpha_long), ALPHA_LONG):
        raise ValueError("Resolved long carry differs from the stored OOS assumptions")
    if not np.isclose(float(app.v2.costs.overnight_alpha_short), ALPHA_SHORT):
        raise ValueError("Resolved short carry differs from the stored OOS assumptions")

    on = {
        "net": _load_series(ON_DIR, "daily_net_returns.csv", "net_return"),
        "gross": _load_series(ON_DIR, "daily_gross_returns.csv", "gross_return"),
        "cost": _load_series(ON_DIR, "daily_costs_total.csv", "cost"),
        "slippage": _load_series(ON_DIR, "daily_costs_slip.csv", "slip_cost"),
        "turnover": _load_series(ON_DIR, "daily_turnover.csv", "turnover"),
        "weights": _load_weights(ON_DIR),
    }
    off = {
        "net": _load_series(OFF_DIR, "daily_net_returns.csv", "net_return"),
        "gross": _load_series(OFF_DIR, "daily_gross_returns.csv", "gross_return"),
        "cost": _load_series(OFF_DIR, "daily_costs_total.csv", "cost"),
        "turnover": _load_series(OFF_DIR, "daily_turnover.csv", "turnover"),
        "weights": _load_weights(OFF_DIR),
    }
    dates = _assert_same_index(
        {
            "on_" + key: value for key, value in on.items()
        }
        | {"off_" + key: value for key, value in off.items()}
    )
    if len(dates) != 368:
        raise ValueError(f"Expected the fixed 368-day Stage 7 OOS, found {len(dates)}")

    on_slip_per_bp = _slippage_cost_per_bp(on["weights"])
    off_slip_per_bp = _slippage_cost_per_bp(off["weights"])
    on_slip5 = on_slip_per_bp * BASELINE_SLIPPAGE_BPS
    off_slip5 = off_slip_per_bp * BASELINE_SLIPPAGE_BPS
    if not np.allclose(on_slip5, on["slippage"], atol=1e-12, rtol=1e-10):
        raise ValueError("Reconstructed ML-on slippage differs from stored daily slippage")
    if not np.allclose(on["gross"] - on["cost"], on["net"], atol=1e-12, rtol=1e-10):
        raise ValueError("ML-on daily gross/cost/net identity failed")
    if not np.allclose(off["gross"] - off["cost"], off["net"], atol=1e-12, rtol=1e-10):
        raise ValueError("ML-off daily gross/cost/net identity failed")
    on_non_slip = on["cost"] - on_slip5
    off_non_slip = off["cost"] - off_slip5
    if (off_slip_per_bp < 0.0).any() or (off_non_slip < -1e-10).any():
        raise ValueError("Reconstructed ML-off cost components are invalid")
    daily: dict[str, Any] = {"trade_date": dates.strftime("%Y-%m-%d")}
    stress_results: dict[str, Any] = {}
    all_bootstraps: dict[str, Any] = {}
    for bps in SLIPPAGE_STRESSES_BPS:
        on_net = on["gross"] - on_non_slip - on_slip_per_bp * bps
        off_net = off["gross"] - off_non_slip - off_slip_per_bp * bps
        delta = on_net - off_net
        tag = f"{int(bps)}bps"
        daily[f"ml_on_net_{tag}"] = on_net.to_numpy()
        daily[f"ml_off_net_{tag}"] = off_net.to_numpy()
        daily[f"delta_net_{tag}"] = delta.to_numpy()
        daily[f"ml_on_slippage_{tag}"] = (on_slip_per_bp * bps).to_numpy()
        daily[f"ml_off_slippage_{tag}"] = (off_slip_per_bp * bps).to_numpy()
        stress_results[tag] = {
            "ml_on": {**_metrics(on_net), "total_cost": float((on_non_slip + on_slip_per_bp * bps).sum())},
            "ml_off": {**_metrics(off_net), "total_cost": float((off_non_slip + off_slip_per_bp * bps).sum())},
            "paired_delta": {
                **_metrics(delta),
                "mean_daily_delta_ci95": _bootstrap(delta)["ci95_mean_daily_delta"],
                "gross_delta_sum": float((on["gross"] - off["gross"]).sum()),
                "cost_delta_sum": float(
                    ((on_non_slip + on_slip_per_bp * bps) - (off_non_slip + off_slip_per_bp * bps)).sum()
                ),
                "turnover_delta_mean": float((on["turnover"] - off["turnover"]).mean()),
            },
        }
        all_bootstraps[tag] = stress_results[tag]["paired_delta"]["mean_daily_delta_ci95"]

    bps5_delta = float(stress_results["5bps"]["paired_delta"]["sum_return"])
    incremental_slippage_per_bp = float((on_slip_per_bp - off_slip_per_bp).sum())
    break_even_bps = (
        BASELINE_SLIPPAGE_BPS + bps5_delta / incremental_slippage_per_bp
        if incremental_slippage_per_bp > 0.0
        else None
    )
    summary = {
        "status": "MODELED_COST_SENSITIVITY_PENDING_REAL_EXECUTION_VALIDATION",
        "period": {"start": str(dates.min().date()), "end": str(dates.max().date()), "days": len(dates)},
        "comparison": "fixed ML-on and ML-off Stage 5/6 daily outputs; same dates and gross-return basis",
        "inputs": {
            "baseline_slippage_bps_per_side": BASELINE_SLIPPAGE_BPS,
            "stresses_bps_per_side": list(SLIPPAGE_STRESSES_BPS),
            "side_leverage": SIDE_LEVERAGE,
            "overnight_alpha_long": ALPHA_LONG,
            "overnight_alpha_short": ALPHA_SHORT,
            "non_slippage_costs_held_at_existing_modeled_levels": True,
            "realized_fills_used": False,
        },
        "bootstrap": {
            "method": "non-circular moving-block bootstrap of paired daily net delta",
            "block_days": BLOCK_DAYS,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "stress_results": stress_results,
        "break_even_common_slippage_bps_per_side": break_even_bps,
        "identity_checks": {
            "ml_on_reconstructed_5bps_slippage_max_abs_error": float(np.max(np.abs(on_slip5 - on["slippage"]))),
            "ml_on_gross_minus_cost_max_abs_error": float(np.max(np.abs(on["gross"] - on["cost"] - on["net"]))),
            "ml_off_gross_minus_cost_max_abs_error": float(np.max(np.abs(off["gross"] - off["cost"] - off["net"]))),
        },
        "limitations": [
            "All stress rows use modeled returns and costs, not broker fills.",
            "Financing, borrow, and reverse costs stay at the stored model levels.",
            "The result reprices fixed weights; it does not re-optimize orders or model lot rounding, fill probability, or market impact.",
            "The 368-day period is previously examined and is not a new prospective holdout.",
        ],
    }
    return summary, pd.DataFrame(daily)


def _write_report(summary: dict[str, Any]) -> None:
    rows = []
    for tag, result in summary["stress_results"].items():
        on = result["ml_on"]
        off = result["ml_off"]
        delta = result["paired_delta"]
        ci = delta["mean_daily_delta_ci95"]
        rows.append(
            "| {bps} | {on_sum:.4f} | {off_sum:.4f} | {delta_sum:+.4f} | "
            "{on_sharpe:.3f} | {off_sharpe:.3f} | {ci_lo:+.6f} .. {ci_hi:+.6f} | "
            "{on_cost:.4f} | {off_cost:.4f} |".format(
                bps=tag,
                on_sum=on["sum_return"],
                off_sum=off["sum_return"],
                delta_sum=delta["sum_return"],
                on_sharpe=on["net_sharpe"],
                off_sharpe=off["net_sharpe"],
                ci_lo=ci[0],
                ci_hi=ci[1],
                on_cost=on["total_cost"],
                off_cost=off["total_cost"],
            )
        )
    report = [
        "# ML overlay cost sensitivity",
        "",
        "判定: **PENDING_REAL_EXECUTION_COSTS**",
        "",
        f"- 固定OOS期間: {summary['period']['start']}〜{summary['period']['end']} ({summary['period']['days']}日)",
        "- 比較: Stage 5のML有効とStage 6のML無効。ウェイト・gross収益は固定し、slippageだけ片道5/10/20bpsへ再価格付け。",
        "- financing / borrow / reverse は既存モデル値を据え置く。実約定・非約定・ロット丸め・impactは含まない。",
        f"- bootstrap: paired日次net差、20日non-circular block、{BOOTSTRAP_SAMPLES:,}回、seed={BOOTSTRAP_SEED}。",
        "",
        "| 片道slippage | ML on net合計 | ML off net合計 | 差 (on−off) | ML on Sharpe | ML off Sharpe | 差の平均95%区間 | ML on cost | ML off cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"同一slippage前提のML損益分岐点: **{summary['break_even_common_slippage_bps_per_side']:.3f} bps/片道**" if summary["break_even_common_slippage_bps_per_side"] is not None else "同一slippageを上げてもML側の相対コスト増は生じず、単一の損益分岐bpsはありません。",
        "",
        "## 判定",
        "",
        "この感度分析では価格・費用モデルの仮定を変えた影響を確認できるが、broker約定費用込みのML価値はまだ判定できない。既知368日を新しいforward OOSとは扱わない。現行のML有効版は実費優位が確認されるまで研究上pendingとする。",
        "",
        "## 検証",
        "",
        f"- 5bpsの再構成slippageと保存系列の最大差: {summary['identity_checks']['ml_on_reconstructed_5bps_slippage_max_abs_error']:.3e}",
        f"- ML on gross−cost−netの最大差: {summary['identity_checks']['ml_on_gross_minus_cost_max_abs_error']:.3e}",
        f"- ML off gross−cost−netの最大差: {summary['identity_checks']['ml_off_gross_minus_cost_max_abs_error']:.3e}",
        "- 実約定との対応: なし。機能の実費込み採用判定は保留。",
        "",
        "日次結果: `daily_cost_sensitivity.csv`; machine-readable summary: `summary.json`。",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(REGISTRY_PATH)
    name = "ml_overlay_cost_sensitivity_20260927"
    run_number = sum(1 for _ in registry.iter_records(name=name)) + 1
    started_at = datetime.now(timezone.utc)
    summary, daily = _run()
    daily.to_csv(OUTPUT_DIR / "daily_cost_sensitivity.csv", index=False)
    summary["registry_record"] = {
        "name": name,
        "decision": Decision.PENDING.value,
        "run_number": run_number,
        "deflated_sharpe": None,
        "note": "Predeclared cost stresses; not independent strategy candidates; DSR not used for adoption.",
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(summary)
    record = ExperimentRecord(
        name=name,
        hypothesis="Check whether the existing modeled ML-on incremental net return survives predetermined one-way slippage stress levels.",
        start_time=started_at,
        end_time=datetime.now(timezone.utc),
        parameters={
            "source_on": str(ON_DIR.relative_to(ROOT)),
            "source_off": str(OFF_DIR.relative_to(ROOT)),
            "slippage_stresses_bps_per_side": list(SLIPPAGE_STRESSES_BPS),
            "non_slippage_costs": "held at stored Stage 5/6 model values",
            "period": summary["period"],
            "block_bootstrap": summary["bootstrap"],
        },
        metrics={
            "paired_delta_by_slippage": {
                key: value["paired_delta"] for key, value in summary["stress_results"].items()
            },
            "realized_fill_costs_used": False,
            "break_even_common_slippage_bps_per_side": summary["break_even_common_slippage_bps_per_side"],
            "run_number": run_number,
            "dsr_not_calculated": "Predetermined cost stress only; no candidate parameter was selected.",
        },
        decision=Decision.PENDING,
        report_path="reports/20260927_ml_overlay_cost_sensitivity/report.md",
        related_records=["stage7_ml_trade_value_20260923"],
    )
    registry.record(record)


if __name__ == "__main__":
    main()
