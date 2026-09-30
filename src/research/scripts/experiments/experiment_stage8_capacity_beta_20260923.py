#!/usr/bin/env python3
"""Stage 8: diagnose concentration, beta, and modeled capacity limits.

The calculation is intentionally split into (a) deterministic model-weight
constraints and (b) execution-dependent capacity evidence.  It never treats
close prices or five-minute bars as executable 09:10 quotes and never changes
production configuration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.experiment_registry import Decision
from research.experiment_utils import record_simple_experiment

OUTPUT_DIR = ROOT / "reports" / "20260923_profitability_order_8"
BACKTEST_DIR = ROOT / "reports" / "20260923_profitability_order_5" / "v2_current_overlay_oos"
MICROSTRUCTURE_PATH = ROOT / "reports" / "20260923_profitability_order_2" / "microstructure.json"
ANNUALIZATION_DAYS = 252.0
SIDE_LEVERAGE = 1.5
MODEL_GROSS_LIMIT = 2.0
MODEL_NET_LIMIT = 0.05
BASELINE_SLIPPAGE_BPS = 5.0
SLIPPAGE_STRESS_BPS = (5.0, 10.0, 20.0)
ILLUSTRATIVE_AUM_JPY = (10_000_000.0, 100_000_000.0, 1_000_000_000.0, 3_000_000_000.0)


def _series(filename: str, column: str) -> pd.Series:
    frame = pd.read_csv(BACKTEST_DIR / filename, index_col=0, parse_dates=True)
    values = pd.to_numeric(frame[column], errors="raise")
    values.index = pd.DatetimeIndex(values.index).normalize()
    return values.astype(float)


def _weights() -> pd.DataFrame:
    frame = pd.read_csv(BACKTEST_DIR / "daily_weights.csv", index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    missing = [ticker for ticker in JP_TICKERS if ticker not in frame.columns]
    if missing:
        raise ValueError(f"weights missing JP tickers: {missing}")
    return frame[JP_TICKERS].astype(float)


def _stats(values: pd.Series) -> dict[str, float | int | None]:
    values = values.astype(float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "n": int(len(values)),
        "mean_daily": float(values.mean()),
        "annualized_sharpe": float(values.mean() / std * np.sqrt(ANNUALIZATION_DAYS))
        if std > 1e-12
        else None,
        "sum_return": float(values.sum()),
        "max_drawdown": float(drawdown.min()),
    }


def _load_microstructure() -> dict[str, object]:
    return json.loads(MICROSTRUCTURE_PATH.read_text(encoding="utf-8"))


def _build() -> tuple[dict[str, object], pd.DataFrame]:
    weights = _weights()
    net = _series("daily_net_returns.csv", "net_return")
    gross = _series("daily_gross_returns.csv", "gross_return")
    costs = _series("daily_costs_total.csv", "cost")
    slip = _series("daily_costs_slip.csv", "slip_cost")
    turnover = _series("daily_turnover.csv", "turnover")
    for series in (net, gross, costs, slip, turnover):
        if not series.index.equals(weights.index):
            raise ValueError("backtest series and weights are not aligned")
    np.testing.assert_allclose(gross - costs, net, atol=1e-14, rtol=0.0)

    abs_weights = weights.abs()
    gross_exposure = abs_weights.sum(axis=1)
    net_exposure = weights.sum(axis=1)
    sorted_abs = np.sort(abs_weights.to_numpy(dtype=float), axis=1)[:, ::-1]
    gross_safe = gross_exposure.replace(0.0, np.nan)
    concentration = pd.DataFrame(
        {
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "max_single_weight": abs_weights.max(axis=1),
            "top1_abs_share": pd.Series(sorted_abs[:, 0], index=weights.index) / gross_safe,
            "top3_abs_share": pd.Series(sorted_abs[:, :3].sum(axis=1), index=weights.index) / gross_safe,
            "hhi_abs_weights": (abs_weights.pow(2).sum(axis=1) / gross_safe.pow(2)),
            "long_count": (weights > 0.0).sum(axis=1),
            "short_count": (weights < 0.0).sum(axis=1),
            "weighted_gap_beta": np.nan,
        },
        index=weights.index,
    )

    df_exec = load_df_exec_from_local_cache()
    beta_columns = [f"jp_beta_{ticker}" for ticker in JP_TICKERS]
    beta_available = all(column in df_exec.columns for column in beta_columns)
    if beta_available:
        beta = df_exec[beta_columns].reindex(weights.index).astype(float)
        beta.columns = JP_TICKERS
        weighted_beta = (weights * beta).sum(axis=1, min_count=len(JP_TICKERS))
        concentration["weighted_gap_beta"] = weighted_beta
    else:
        weighted_beta = pd.Series(np.nan, index=weights.index)

    stress_rows: list[dict[str, float | int | None]] = []
    other_cost = costs - slip
    for slippage_bps in SLIPPAGE_STRESS_BPS:
        stressed_cost = other_cost + slip * (slippage_bps / BASELINE_SLIPPAGE_BPS)
        stressed_net = gross - stressed_cost
        row = {
            "slippage_bps_per_side": slippage_bps,
            "additional_cost_sum": float((stressed_cost - costs).sum()),
            **_stats(stressed_net),
        }
        stress_rows.append(row)

    aum_rows: list[dict[str, float | None]] = []
    for aum in ILLUSTRATIVE_AUM_JPY:
        gross_notional = aum * gross_exposure
        aum_rows.append(
            {
                "aum_jpy": aum,
                "mean_model_gross_notional_jpy": float(gross_notional.mean()),
                "p95_model_gross_notional_jpy": float(gross_notional.quantile(0.95)),
                "max_model_gross_notional_jpy": float(gross_notional.max()),
                "p95_largest_single_name_notional_jpy": float(
                    (aum * abs_weights.max(axis=1)).quantile(0.95)
                ),
            }
        )

    micro = _load_microstructure()
    live_capture = micro["live_capture"]
    fill_evidence = micro["fill_evidence"]
    execution_inputs = {
        "true_0910_quotes": bool(micro["completion_rules"]["true_timestamped_0910_quote"]),
        "all_17_lob_snapshots": bool(micro["completion_rules"]["all_17_lob_snapshots"]),
        "fill_and_fee_detail": bool(micro["completion_rules"]["broker_fill_and_fee_detail"]),
        "observed_spread_count": live_capture["spread_bps_summary"]["count"],
        "observed_spread_is_oos_executable": False,
        "official_quantity_reconciled": fill_evidence["official_quantity_for_local_keys"] > 0,
        "runtime_quantity_gap": fill_evidence["runtime_quantity_gap"],
        "borrow_by_ticker_available": False,
        "lot_size_and_fill_price_available_for_oos": False,
    }
    concentration_frame = concentration.copy()
    concentration_frame["net_return"] = net
    concentration_frame["gross_return"] = gross
    concentration_frame["cost"] = costs
    concentration_frame["turnover"] = turnover
    concentration_frame.to_csv(OUTPUT_DIR / "daily_capacity_beta.csv", index_label="trade_date")

    model_constraints = {
        "max_abs_model_net": float(net_exposure.abs().max()),
        "max_model_gross": float(gross_exposure.max()),
        "max_effective_abs_net": float(net_exposure.abs().max() * SIDE_LEVERAGE),
        "max_effective_gross": float(gross_exposure.max() * SIDE_LEVERAGE),
        "model_net_limit": MODEL_NET_LIMIT,
        "model_gross_limit": MODEL_GROSS_LIMIT,
        "model_constraints_pass": bool(
            net_exposure.abs().max() <= MODEL_NET_LIMIT + 1e-12
            and gross_exposure.max() <= MODEL_GROSS_LIMIT + 1e-12
        ),
        "effective_exposure_is_reported_separately": True,
    }
    summary: dict[str, object] = {
        "stage": 8,
        "status": "COMPLETE_MODEL_DIAGNOSTIC_PENDING_EXECUTION_CAPACITY",
        "decision": "PENDING_NOT_ADOPTED",
        "period": {
            "start": str(weights.index.min().date()),
            "end": str(weights.index.max().date()),
            "days": int(len(weights)),
        },
        "model_constraints": model_constraints,
        "concentration": {
            "mean_max_single_weight": float(concentration["max_single_weight"].mean()),
            "p95_max_single_weight": float(concentration["max_single_weight"].quantile(0.95)),
            "max_max_single_weight": float(concentration["max_single_weight"].max()),
            "mean_top1_abs_share": float(concentration["top1_abs_share"].mean()),
            "p95_top3_abs_share": float(concentration["top3_abs_share"].quantile(0.95)),
            "max_hhi_abs_weights": float(concentration["hhi_abs_weights"].max()),
            "mean_long_count": float(concentration["long_count"].mean()),
            "mean_short_count": float(concentration["short_count"].mean()),
        },
        "gap_beta_proxy": {
            "available": beta_available,
            "mean_weighted_gap_beta": float(weighted_beta.mean()) if beta_available else None,
            "p95_abs_weighted_gap_beta": float(weighted_beta.abs().quantile(0.95))
            if beta_available
            else None,
            "max_abs_weighted_gap_beta": float(weighted_beta.abs().max()) if beta_available else None,
            "definition": "sum(model_weight * strictly historical jp_beta); not an executed/fill beta",
        },
        "slippage_stress": stress_rows,
        "illustrative_aum_notional": aum_rows,
        "execution_capacity_inputs": execution_inputs,
        "production_changed": False,
        "limitations": [
            "No historical timestamped 09:10 quote/5-level executable book exists for this OOS period.",
            "No OOS order quantity, fill rate, lot rounding, or order-level spread/impact is available.",
            "AUM rows are model notional scenarios, not capacity or fillability estimates.",
            "Slippage stress scales the backtest slip component and is not a substitute for observed spread/depth.",
            "The beta result is a model gap-beta proxy and does not prove realized post-fill beta neutrality.",
        ],
    }
    return summary, concentration_frame


def _report(summary: dict[str, object]) -> str:
    constraints = summary["model_constraints"]
    concentration = summary["concentration"]
    beta = summary["gap_beta_proxy"]
    execution = summary["execution_capacity_inputs"]
    lines = [
        "# 収益改善順序8：容量・集中・β制約",
        "",
        "発注・取消・再送を行わない、Stage 5の同一OOS weightsを使ったモデル側診断です。",
        "",
        f"判定: **{summary['decision']}**",
        "",
        "## 固定条件",
        "",
        f"- 期間: {summary['period']['start']} -> {summary['period']['end']} ({summary['period']['days']}日)",
        "- model gross上限2.0、model net±0.05、side leverage 1.5を分けて集計",
        "- 5/10/20bps片道は事前固定のストレス表示であり、採用パラメータ探索ではない",
        "",
        "## 制約と集中",
        "",
        f"- max model net: {constraints['max_abs_model_net']:.3e}、max model gross: {constraints['max_model_gross']:.6f}",
        f"- max effective net: {constraints['max_effective_abs_net']:.3e}、max effective gross: {constraints['max_effective_gross']:.6f}",
        f"- model制約（net/gross）: {'PASS' if constraints['model_constraints_pass'] else 'FAIL'}",
        f"- 単一銘柄weight abs 平均 / p95 / 最大: {concentration['mean_max_single_weight']:.6f} / {concentration['p95_max_single_weight']:.6f} / {concentration['max_max_single_weight']:.6f}",
        f"- top1 abs share平均: {concentration['mean_top1_abs_share']:.3%}、top3 abs share p95: {concentration['p95_top3_abs_share']:.3%}",
        f"- abs-weight HHI最大: {concentration['max_hhi_abs_weights']:.6f}、平均long/short銘柄数: {concentration['mean_long_count']:.2f} / {concentration['mean_short_count']:.2f}",
        "",
        "## β診断",
        "",
        f"- historical jp_betaによるweighted gap-beta proxy: {'available' if beta['available'] else 'unavailable'}",
        f"- weighted gap-beta 平均 / abs p95 / abs最大: {beta['mean_weighted_gap_beta'] if beta['available'] else 'NA'} / {beta['p95_abs_weighted_gap_beta'] if beta['available'] else 'NA'} / {beta['max_abs_weighted_gap_beta'] if beta['available'] else 'NA'}",
        "- これは `sum(model_weight * jp_beta)` であり、約定後の実現βやTOPIXへの日中βを証明しない。",
        "",
        "## スリッページ・資金規模の表示",
        "",
        "| 片道slippage | 追加費用合計 | net合計 | net Sharpe | max DD |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["slippage_stress"]:
        lines.append(
            f"| {row['slippage_bps_per_side']:.1f}bps | {row['additional_cost_sum']:.6f} | {row['sum_return']:.6f} | {row['annualized_sharpe']:.6f} | {row['max_drawdown']:.6f} |"
        )
    lines.extend(
        [
            "",
            "AUM表はweightから得た理論notionalで、ADV・板厚・約定率を含まないためcapacityとは呼ばない。詳細は `stage8_summary.json` と `daily_capacity_beta.csv` を参照。",
            "",
            "## 実行可能性の未充足",
            "",
            f"- true 09:10 quote: {execution['true_0910_quotes']}、17銘柄5段板: {execution['all_17_lob_snapshots']}",
            f"- 約定・手数料明細: {execution['fill_and_fee_detail']}、runtime数量gap: {execution['runtime_quantity_gap']}",
            f"- OOSのlot/約定価格: {execution['lot_size_and_fill_price_available_for_oos']}、銘柄別borrow: {execution['borrow_by_ticker_available']}",
            "- よって、資金規模別の実容量、spread/impact、fill rate、貸株制約を判定できない。",
            "",
            "## 判定と次の依存",
            "",
            "- **COMPLETE_MODEL_DIAGNOSTIC_PENDING_EXECUTION_CAPACITY / PENDING_NOT_ADOPTED**",
            "- モデルweight制約・集中・beta proxy・固定stressは診断したが、順序8の経済的完了条件（板厚、約定率、口数、実約定後exposure）は未充足。",
            "- 順序9は、実行可能baselineと容量制約を通過した独立OOSがないため、新特徴量/universeの追加実験へ進めない。",
            "",
            "## 参照",
            "",
            "- `stage8_summary.json`",
            "- `daily_capacity_beta.csv`",
            "- `reports/20260923_profitability_order_2/microstructure.json`",
            "- `reports/20260923_profitability_order_5/report.md`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, _ = _build()
    record = record_simple_experiment(
        name="stage8_capacity_beta_20260923",
        hypothesis="The current OOS portfolio remains executable at fixed AUM sizes while respecting concentration, beta, spread, depth, and fill constraints.",
        parameters={
            "aum_scenarios_jpy": list(ILLUSTRATIVE_AUM_JPY),
            "slippage_stress_bps_per_side": list(SLIPPAGE_STRESS_BPS),
            "side_leverage": SIDE_LEVERAGE,
        },
        metrics={
            "stage": 8,
            "n_observations": summary["period"]["days"],
            "model_constraints_pass": summary["model_constraints"]["model_constraints_pass"],
            "max_model_gross": summary["model_constraints"]["max_model_gross"],
            "max_abs_weighted_gap_beta": summary["gap_beta_proxy"]["max_abs_weighted_gap_beta"],
            "true_0910_quotes": summary["execution_capacity_inputs"]["true_0910_quotes"],
            "fill_and_fee_detail": summary["execution_capacity_inputs"]["fill_and_fee_detail"],
        },
        decision=Decision.PENDING,
        report_path="reports/20260923_profitability_order_8/report.md",
    )
    summary["registry_record"] = {
        "name": record.name,
        "decision": record.decision.value,
        "trials": record.metrics.get("trials"),
        "deflated_sharpe": record.metrics.get("deflated_sharpe"),
    }
    (OUTPUT_DIR / "stage8_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "report.md").write_text(_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
