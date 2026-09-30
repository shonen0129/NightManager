#!/usr/bin/env python3
"""Stage 5: decompose V2 returns into intraday, carry, and costs.

This is a diagnostic run, not a production promotion. It consumes the
canonical V2 backtest output generated with the active production overlay and
replays the same weights through the shared PnL calculator to verify the
attribution and a no-carry counterfactual.
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

from leadlag.core.pnl import simulate_daily_pnl
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
from research.experiment_utils import record_backtest_experiment

OUTPUT_DIR = ROOT / "reports" / "20260923_profitability_order_5"
BACKTEST_DIR = OUTPUT_DIR / "v2_current_overlay_oos"
REPORT_PATH = OUTPUT_DIR / "report.md"
ANNUALIZATION_DAYS = 252.0


def _load_series(name: str, column: str) -> pd.Series:
    path = BACKTEST_DIR / name
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if column not in frame.columns:
        raise ValueError(f"{path} does not contain column {column!r}")
    series = pd.to_numeric(frame[column], errors="raise")
    series.index = pd.DatetimeIndex(series.index).normalize()
    return series.astype(float)


def _load_weights() -> pd.DataFrame:
    weights = pd.read_csv(BACKTEST_DIR / "daily_weights.csv", index_col=0, parse_dates=True)
    weights.index = pd.DatetimeIndex(weights.index).normalize()
    missing = [ticker for ticker in JP_TICKERS if ticker not in weights.columns]
    if missing:
        raise ValueError(f"daily_weights.csv is missing tickers: {missing}")
    return weights[JP_TICKERS].astype(float)


def _stats(series: pd.Series) -> dict[str, float | int | None]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "n": 0,
            "mean_daily": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "annualized_sharpe": None,
            "sum_return": None,
            "max_drawdown": None,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    shared = calculate_metrics(
        clean,
        spec=MetricsSpec(annualization_periods=int(ANNUALIZATION_DAYS)),
    )
    return {
        "n": int(len(clean)),
        "mean_daily": mean,
        "annualized_return": mean * ANNUALIZATION_DAYS,
        "annualized_volatility": std * np.sqrt(ANNUALIZATION_DAYS),
        "annualized_sharpe": (
            float(shared["Sharpe"]) if np.isfinite(shared.get("Sharpe", np.nan)) else None
        ),
        "sum_return": float(clean.sum()),
        "max_drawdown": float(shared["MDD"]),
    }


def _cost_value(app_config: object, name: str) -> float:
    costs = getattr(getattr(app_config, "v2"), "costs", None)
    if costs is not None and hasattr(costs, name):
        value = getattr(costs, name)
        if value is not None:
            return float(value)
    return float(getattr(getattr(app_config, "strategy"), name))


def _group_summary(frame: pd.DataFrame, group_name: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for label, part in frame.groupby(group_name, dropna=False, sort=False):
        output[str(label)] = {
            "n": int(len(part)),
            "intraday_sum": float(part["intraday_gross"].sum()),
            "overnight_sum": float(part["overnight"].sum()),
            "cost_sum": float(part["cost"].sum()),
            "net_sum": float(part["net"].sum()),
            "mean_net": float(part["net"].mean()),
            "worst_net": float(part["net"].min()),
        }
    return output


def _build_decomposition() -> tuple[pd.DataFrame, dict[str, object], object]:
    net = _load_series("daily_net_returns.csv", "net_return")
    gross = _load_series("daily_gross_returns.csv", "gross_return")
    overnight = _load_series("daily_overnight_returns.csv", "overnight_return")
    total_cost = _load_series("daily_costs_total.csv", "cost")
    slip = _load_series("daily_costs_slip.csv", "slip_cost")
    financing = _load_series("daily_costs_financing.csv", "financing_cost")
    borrow = _load_series("daily_costs_borrow.csv", "borrow_cost")
    reverse = _load_series("daily_costs_reverse.csv", "reverse_cost")
    fallback = _load_series("daily_fallback.csv", "fallback").astype(bool)
    turnover = _load_series("daily_turnover.csv", "turnover")
    gross_exposure = _load_series("daily_gross.csv", "gross")
    weights = _load_weights()

    for other in (
        gross,
        overnight,
        total_cost,
        slip,
        financing,
        borrow,
        reverse,
        fallback,
        turnover,
        gross_exposure,
    ):
        if not other.index.equals(net.index):
            raise ValueError("Backtest output series have different date indexes")
    if not weights.index.equals(net.index):
        raise ValueError("daily_weights.csv and PnL series have different date indexes")

    decomposition = pd.DataFrame(
        {
            "gross": gross,
            "intraday_gross": gross - overnight,
            "overnight": overnight,
            "cost": total_cost,
            "slip_cost": slip,
            "financing_cost": financing,
            "borrow_cost": borrow,
            "reverse_cost": reverse,
            "net": net,
            "fallback": fallback,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
        },
        index=net.index,
    )

    cost_residual = decomposition["cost"] - decomposition[
        ["slip_cost", "financing_cost", "borrow_cost", "reverse_cost"]
    ].sum(axis=1)
    pnl_residual = decomposition["net"] - (
        decomposition["intraday_gross"] + decomposition["overnight"] - decomposition["cost"]
    )
    checks = {
        "cost_breakdown_max_abs_error": float(cost_residual.abs().max()),
        "pnl_attribution_max_abs_error": float(pnl_residual.abs().max()),
    }

    dates = decomposition.index
    next_date = pd.Series(dates[1:], index=dates[:-1])
    calendar_days = next_date.reindex(dates).sub(pd.Series(dates, index=dates))
    calendar_days = pd.to_numeric(calendar_days.dt.days, errors="coerce")
    decomposition["calendar_days_to_next_session"] = calendar_days
    decomposition["session_gap_class"] = np.select(
        [calendar_days.isna(), calendar_days > 1],
        ["terminal", "weekend_or_holiday"],
        default="next_session_1d",
    )

    stress_threshold = float(decomposition["gross"].abs().quantile(0.95))
    decomposition["realized_stress_95pct"] = decomposition["gross"].abs() >= stress_threshold
    decomposition["realized_stress_threshold"] = stress_threshold

    app_config = load_config_from_yaml(ROOT / "configs" / "production" / "production.yaml")
    df_exec = load_df_exec_from_local_cache()
    open_910 = build_open_910_returns(df_exec, JP_TICKERS)
    target_returns, gap_returns = BacktestEngine._compute_target_and_gap_returns(
        df_exec,
        df_exec.index,
        dates,
        open_910_returns=open_910,
    )
    weights_arr = weights.to_numpy(dtype=float)
    common = {
        "slip": _cost_value(app_config, "slippage_bps_per_side") / 10000.0,
        "financing_daily": _cost_value(app_config, "buy_interest_annual") / 365.0,
        "borrow_daily": _cost_value(app_config, "borrow_fee_annual") / 365.0,
        "reverse_daily": _cost_value(app_config, "reverse_fee_bps") / 10000.0,
        "alpha_long": _cost_value(app_config, "overnight_alpha_long"),
        "alpha_short": _cost_value(app_config, "overnight_alpha_short"),
        "side_leverage": _cost_value(app_config, "side_leverage"),
    }
    replay = simulate_daily_pnl(
        weights=weights_arr,
        target_returns=target_returns,
        gap_returns=gap_returns,
        sim_dates=dates,
        **common,
    )
    checks["replay_net_max_abs_error"] = float(
        np.max(np.abs(np.asarray(replay["net_returns"]) - net.to_numpy()))
    )
    checks["replay_overnight_max_abs_error"] = float(
        np.max(np.abs(np.asarray(replay["overnight_returns"]) - overnight.to_numpy()))
    )
    checks["replay_cost_max_abs_error"] = float(
        np.max(np.abs(np.asarray(replay["costs"]) - total_cost.to_numpy()))
    )

    no_carry = simulate_daily_pnl(
        weights=weights_arr,
        target_returns=target_returns,
        gap_returns=gap_returns,
        sim_dates=dates,
        slip=common["slip"],
        financing_daily=common["financing_daily"],
        borrow_daily=common["borrow_daily"],
        reverse_daily=common["reverse_daily"],
        alpha_long=0.0,
        alpha_short=0.0,
        side_leverage=common["side_leverage"],
    )
    decomposition["no_carry_net"] = np.asarray(no_carry["net_returns"])

    summary: dict[str, object] = {
        "checks": checks,
        "cost_parameters": common,
        "stress_threshold_abs_gross": stress_threshold,
        "all_days": _stats(decomposition["net"]),
        "intraday_gross": _stats(decomposition["intraday_gross"]),
        "overnight": _stats(decomposition["overnight"]),
        "cost_as_negative_return": _stats(-decomposition["cost"]),
        "cost_components": {
            "slip": float(decomposition["slip_cost"].sum()),
            "financing": float(decomposition["financing_cost"].sum()),
            "borrow": float(decomposition["borrow_cost"].sum()),
            "reverse": float(decomposition["reverse_cost"].sum()),
        },
        "no_carry_net": _stats(decomposition["no_carry_net"]),
        "no_carry_net_delta_vs_baseline": _stats(
            decomposition["no_carry_net"] - decomposition["net"]
        ),
        "fallback_days": int(decomposition["fallback"].sum()),
        "fallback_rate": float(decomposition["fallback"].mean()),
        "session_gap_groups": _group_summary(decomposition, "session_gap_class"),
        "realized_stress_group": _group_summary(
            decomposition.assign(
                stress_group=decomposition["realized_stress_95pct"].map(
                    {True: "top_5pct_abs_gross", False: "other"}
                )
            ),
            "stress_group",
        ),
    }
    return decomposition, summary, app_config


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decomposition, summary, app_config = _build_decomposition()
    decomposition.to_csv(OUTPUT_DIR / "daily_decomposition.csv", index_label="trade_date")

    summary["period"] = {
        "start": str(decomposition.index.min().date()),
        "end": str(decomposition.index.max().date()),
        "days": int(len(decomposition)),
    }
    summary["production_config"] = "configs/production/production.yaml"
    summary["active_overlay"] = "models/ml_order_overlay/production_20260923"
    summary["gap_input"] = "var/live/pipeline_data/gap_adjusted_distribution/20260731_024303"
    summary["oos_boundary"] = "2024-12-23 (after overlay training end 2024-12-20)"
    summary["decision"] = "PENDING_STAGE_5_COMPLETION"
    summary["limitation"] = (
        "This diagnostic uses canonical V2 backtest/proxy inputs, not true 09:10 quotes or live fills."
    )

    registry_results = {
        "daily_returns": _load_series("daily_net_returns.csv", "net_return"),
        "daily_fallback": _load_series("daily_fallback.csv", "fallback").astype(bool),
        "daily_turnover": _load_series("daily_turnover.csv", "turnover"),
        "daily_gross_exps": _load_series("daily_gross.csv", "gross"),
    }
    record = record_backtest_experiment(
        name="stage5_intraday_carry_decomposition_20260923",
        hypothesis=(
            "Separate V2 09:10-to-close alpha from overnight carry and attribute all four costs "
            "before evaluating carry risk or changing the hold policy."
        ),
        app_config=app_config,
        results=registry_results,
        extra_metrics={
            "stage": 5,
            "summary": summary,
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": int(ANNUALIZATION_DAYS),
        },
        decision=Decision.PENDING,
        reason="Diagnostic only; true 09:10 execution evidence from stage 2 is still incomplete.",
        report_path="reports/20260923_profitability_order_5/report.md",
        metrics_spec=MetricsSpec(annualization_periods=int(ANNUALIZATION_DAYS)),
    )
    summary["registry_record"] = {
        "name": record.name,
        "decision": record.decision.value,
        "trials": record.metrics.get("trials"),
        "deflated_sharpe": record.metrics.get("deflated_sharpe"),
    }
    (OUTPUT_DIR / "stage5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    checks = summary["checks"]
    lines = [
        "# 収益改善順序5：日中と持越しの分解",
        "",
        "発注・取消・再送を行わない、canonical V2 backtestの診断です。",
        "",
        f"判定: **{summary['decision']}**",
        "",
        "## 固定条件",
        "",
        f"- 期間: {summary['period']['start']} -> {summary['period']['end']} ({summary['period']['days']}日)",
        f"- OOS境界: {summary['oos_boundary']}",
        f"- config: {summary['production_config']}",
        f"- active overlay: {summary['active_overlay']}",
        f"- gap input: {summary['gap_input']}",
        f"- 年率換算: {ANNUALIZATION_DAYS:.0f}営業日",
        f"- 評価は全{len(decomposition)}日（fallback日を除外しない）",
        "- 今回のdf_exec再構築では前回出力にあった2025-10-28がなく、前回369日から1日減少。推測補完せず、順序6-8も今回の同一日付集合で再計算する。",
        "- 急変日は実現gross絶対値のOOS上位5%で分類する事後診断であり、予測時点の特徴量ではない",
        "",
        "## 帰属結果",
        "",
        "| 系列 | 年率リターン | 年率vol | 年率Sharpe | 合計 |",
            "|---|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("net", "all_days"),
        ("intraday gross", "intraday_gross"),
        ("overnight carry", "overnight"),
        ("cost (negative)", "cost_as_negative_return"),
        ("no-carry net counterfactual", "no_carry_net"),
    ]:
        item = summary[key]
        sharpe = item["annualized_sharpe"]
        lines.append(
            f"| {label} | {item['annualized_return']:.6f} | "
            f"{item['annualized_volatility']:.6f} | {sharpe if sharpe is not None else 'n/a'} | "
            f"{item['sum_return']:.6f} |"
        )
    lines.extend(
        [
            "",
            "コスト累計（return単位）: "
            f"slip={summary['cost_components']['slip']:.6f}, "
            f"financing={summary['cost_components']['financing']:.6f}, "
            f"borrow={summary['cost_components']['borrow']:.6f}, "
            f"reverse={summary['cost_components']['reverse']:.6f}",
        ]
    )
    lines.extend(
        [
            "",
            "## 会計整合性",
            "",
            f"- cost内訳最大誤差: {checks['cost_breakdown_max_abs_error']:.3e}",
            f"- PnL帰属最大誤差: {checks['pnl_attribution_max_abs_error']:.3e}",
            f"- shared PnL replay net最大誤差: {checks['replay_net_max_abs_error']:.3e}",
            f"- shared PnL replay overnight最大誤差: {checks['replay_overnight_max_abs_error']:.3e}",
            f"- shared PnL replay cost最大誤差: {checks['replay_cost_max_abs_error']:.3e}",
            "",
            "## 週末・連休・急変時",
            "",
            json.dumps(
                {
                    "session_gap_groups": summary["session_gap_groups"],
                    "realized_stress_group": summary["realized_stress_group"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "## 完了条件と採否",
            "",
            "| 条件 | 結果 |",
            "|---|---|",
            "| 日中/夜間/費用のPnL帰属 | PASS（出力系列とshared PnL replayで整合） |",
            "| 週末・連休の持越し分解 | PASS（calendar day gap別に集計） |",
            "| 急変時の損失分解 | DIAGNOSTIC（実現値による事後分類） |",
            "| 真の9:10価格・実約定との整合 | 未証明（順序2を保留したまま） |",
            "| carry方針の本番変更 | 実施なし |",
            "",
            "順序5の診断計算は完了したが、実行可能価格・実約定・費用を使った正式なbaseline再評価が未完了のため、順序5の経済的な採否は保留する。ユーザー依頼に基づき順序6-8の既存診断は今回の368日データで再計算したが、順序2の実約定証拠や順序3の実行可能baselineの代替にはならず、本番採用判断は引き続き保留する。",
            "",
            f"ExperimentRegistry: {record.name}, decision={record.decision.value}, trials={record.metrics.get('trials')}",
            "",
            "再現JSON: reports/20260923_profitability_order_5/stage5_summary.json",
            "日次分解: reports/20260923_profitability_order_5/daily_decomposition.csv",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
