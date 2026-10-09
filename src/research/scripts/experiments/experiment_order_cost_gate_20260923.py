#!/usr/bin/env python3
"""First-pass V2 test of a forecast-edge-vs-order-cost rebalance gate.

The test reruns the canonical 2024-12-23--2026-07-29 OOS path to capture each
decision's point-in-time ``mu_gap`` forecast. It then moves each day's prior
selected model weight only toward that day's canonical target, and only for
name-level orders whose forecast incremental return exceeds an estimated
execution-plus-carry cost. A small LP chooses partial order sizes while
preserving model net exposure and gross limits. No live orders or production
configuration are changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

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
from leadlag.experiment_registry import (
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics, compute_drawdown_series

OUTPUT_DIR = ROOT / "reports" / "20260923_profitability_order_10"
REPORT_PATH = OUTPUT_DIR / "report.md"
RESULTS_PATH = OUTPUT_DIR / "results.json"
WEIGHTS_PATH = OUTPUT_DIR / "daily_gate_weights.csv"
DECISIONS_PATH = OUTPUT_DIR / "daily_gate_decisions.csv"
REGISTRY_PATH = ROOT / "var" / "experiments" / "registry.jsonl"
BASELINE_DIR = ROOT / "reports" / "20260923_profitability_order_5" / "v2_current_overlay_oos"
GAP_DIR = ROOT / "var" / "live" / "pipeline_data" / "gap_adjusted_distribution" / "20260731_024303"
CONFIG_PATH = ROOT / "configs" / "production" / "production.yaml"
OVERLAY_DIR = ROOT / "models" / "ml_order_overlay" / "production_20260923"
START_DATE = "2024-12-23"
END_DATE = "2026-07-29"
ANNUALIZATION_DAYS = 252
BLOCK_DAYS = 20
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42
RUN_COMMAND = (
    ".venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 "
    ".venv/bin/python src/research/scripts/experiments/experiment_order_cost_gate_20260923.py "
    "> /tmp/20260923_order_cost_gate.log 2>&1"
)

# The spread threshold is interpreted as a full quoted spread width. Crossing
# half of that spread on entry/exit gives 12.5 bps per side at the central
# 0.25% scenario. 10/15 bps are fixed +/-20% sensitivities, not fitted values.
COST_SCENARIOS_BPS_PER_SIDE = (10.0, 12.5, 15.0)
CENTRAL_COST_BPS_PER_SIDE = 12.5
MAX_GROSS = 2.0
WEIGHT_EPS = 1e-12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_baseline() -> tuple[pd.DataFrame, pd.Series]:
    weights = pd.read_csv(BASELINE_DIR / "daily_weights.csv", index_col=0, parse_dates=True)
    weights.index = pd.DatetimeIndex(weights.index).normalize()
    missing = [ticker for ticker in JP_TICKERS if ticker not in weights.columns]
    if missing:
        raise ValueError(f"baseline weights are missing tickers: {missing}")
    weights = weights.loc[:, JP_TICKERS].astype(float)
    returns_frame = pd.read_csv(
        BASELINE_DIR / "daily_net_returns.csv", index_col=0, parse_dates=True
    )
    returns_frame.index = pd.DatetimeIndex(returns_frame.index).normalize()
    baseline_returns = returns_frame["net_return"].astype(float)
    if not baseline_returns.index.equals(weights.index):
        raise ValueError("baseline weight and return dates differ")
    return weights, baseline_returns


def _capture_transform(captured: dict[str, dict[str, np.ndarray]]) -> Any:
    def capture(date_str: str, decision: Any) -> Any:
        captured[date_str] = {
            "mu": np.asarray(decision.mu_gap, dtype=float).copy(),
            "target": np.asarray(decision.w_final, dtype=float).copy(),
        }
        return decision

    return capture


def _run_canonical_and_capture() -> tuple[
    dict[str, Any], dict[str, dict[str, np.ndarray]], Any, pd.DataFrame
]:
    baseline_weights, baseline_returns = _load_baseline()
    app_config = load_config_from_yaml(CONFIG_PATH, strict=True)
    df_exec = load_df_exec_from_local_cache()
    if df_exec is None or df_exec.empty:
        raise RuntimeError("df_exec cache is unavailable")

    captured: dict[str, dict[str, np.ndarray]] = {}
    result = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=GAP_DIR,
        df_exec=df_exec,
        start_date=START_DATE,
        end_date=END_DATE,
        slippage_bps=5.0,
        n_jobs=1,
        overlay_model_dir=OVERLAY_DIR,
        decision_transform=_capture_transform(captured),
    )
    result_weights = result["weights"].copy()
    result_weights.index = pd.DatetimeIndex(result_weights.index).normalize()
    if not result_weights.index.equals(baseline_weights.index):
        raise ValueError(
            f"canonical rerun date mismatch: {len(result_weights)} vs {len(baseline_weights)}"
        )
    weight_error = float(
        np.max(np.abs(result_weights[JP_TICKERS].to_numpy() - baseline_weights.to_numpy()))
    )
    if weight_error > 1e-8:
        raise RuntimeError(f"canonical target weights did not reproduce: max error={weight_error:.3e}")
    net = result["daily_returns"].astype(float).copy()
    net.index = pd.DatetimeIndex(net.index).normalize()
    if not net.index.equals(baseline_returns.index):
        raise ValueError("canonical rerun return dates differ from baseline")
    return_error = float(np.max(np.abs(net.to_numpy() - baseline_returns.to_numpy())))
    if return_error > 1e-10:
        raise RuntimeError(f"5 bps baseline did not reproduce: max return error={return_error:.3e}")

    expected_dates = {
        date.strftime("%Y-%m-%d")
        for date, fallback in zip(result["weights"].index, result["daily_fallback"].to_numpy())
        if not bool(fallback)
    }
    missing_mu = expected_dates - set(captured)
    if missing_mu:
        raise RuntimeError(f"missing point-in-time mu_gap forecasts for {len(missing_mu)} dates")
    for i, date in enumerate(result["weights"].index):
        date_str = date.strftime("%Y-%m-%d")
        if date_str not in captured:
            continue
        mu = captured[date_str]["mu"]
        if mu.shape != (len(JP_TICKERS),) or not np.isfinite(mu).all():
            raise ValueError(f"invalid mu_gap forecast on {date_str}")
        summary = result["v2_summaries"][i]
        if summary and "predicted_portfolio_mean" in summary:
            predicted = float(np.dot(result["weights"].iloc[i].to_numpy(), mu))
            if abs(predicted - float(summary["predicted_portfolio_mean"])) > 1e-8:
                raise RuntimeError(f"mu_gap does not reproduce summary forecast on {date_str}")

    result["canonical_weight_reproduction_max_error"] = weight_error
    result["canonical_return_reproduction_max_error"] = return_error
    return result, captured, app_config, df_exec


def _cost_params(app_config: Any, bps_per_side: float) -> dict[str, float]:
    return BacktestEngine._resolve_v2_backtest_cost_params(
        app_config,
        bps_per_side,
        *([None] * 6),
    )


def _order_cost_gate(
    targets: pd.DataFrame,
    forecasts: dict[str, dict[str, np.ndarray]],
    fallback: pd.Series,
    *,
    bps_per_side: float,
    cost_params: dict[str, float],
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select only individually profitable target moves under neutral/gross constraints."""
    target_values = targets.loc[:, JP_TICKERS].to_numpy(dtype=float)
    output = np.zeros_like(target_values)
    rows: list[dict[str, Any]] = []
    previous = np.zeros(len(JP_TICKERS), dtype=float)
    side_leverage = float(cost_params["side_leverage"])
    long_alpha = float(cost_params["alpha_long"])
    short_alpha = float(cost_params["alpha_short"])
    fin_daily = float(cost_params["fin_annual"]) / 365.0
    borrow_daily = float(cost_params["borrow_annual"]) / 365.0
    reverse_daily = float(cost_params["rev_bps"]) / 10000.0
    slip = float(bps_per_side) / 10000.0

    for row_index, date in enumerate(dates):
        target = target_values[row_index]
        date_str = date.strftime("%Y-%m-%d")
        if bool(fallback.iloc[row_index]):
            # Keep the production V2 fallback contract: an audit/data fallback is flat.
            chosen = np.zeros_like(previous)
            eligible = np.zeros(len(JP_TICKERS), dtype=bool)
            fractions = np.zeros(len(JP_TICKERS), dtype=float)
            mu = np.zeros(len(JP_TICKERS), dtype=float)
            edge = np.zeros(len(JP_TICKERS), dtype=float)
            estimated_cost = np.zeros(len(JP_TICKERS), dtype=float)
            status = "fallback_flat"
        else:
            if date_str not in forecasts:
                raise ValueError(f"no captured point-in-time forecast for {date_str}")
            mu = forecasts[date_str]["mu"]
            delta = target - previous
            edge = side_leverage * delta * mu
            target_alpha = np.where(
                target > WEIGHT_EPS,
                long_alpha,
                np.where(target < -WEIGHT_EPS, short_alpha, np.where(previous >= 0, long_alpha, short_alpha)),
            )
            # Estimated order cost includes one opening fill plus the modeled
            # intraday close fraction. Carry is the incremental next-session
            # cost versus holding the previous model-weight proxy.
            trading_sides = np.where(np.abs(target) > WEIGHT_EPS, 2.0 - target_alpha, 1.0)
            next_date = dates[row_index + 1] if row_index + 1 < len(dates) else date + pd.Timedelta(days=1)
            calendar_days = max(1, int((next_date - date).days))
            target_carry_rate = np.where(
                target > WEIGHT_EPS,
                long_alpha * fin_daily * calendar_days,
                np.where(
                    target < -WEIGHT_EPS,
                    short_alpha * (borrow_daily + reverse_daily) * calendar_days,
                    0.0,
                ),
            )
            previous_carry_rate = np.where(
                previous > WEIGHT_EPS,
                long_alpha * fin_daily * calendar_days,
                np.where(
                    previous < -WEIGHT_EPS,
                    short_alpha * (borrow_daily + reverse_daily) * calendar_days,
                    0.0,
                ),
            )
            estimated_cost = side_leverage * (
                slip * trading_sides * np.abs(delta)
                + target_carry_rate * np.abs(target)
                - previous_carry_rate * np.abs(previous)
            )
            eligible = (np.abs(delta) > WEIGHT_EPS) & (edge > estimated_cost)
            margin = np.where(eligible, edge - estimated_cost, 0.0)
            n = len(JP_TICKERS)
            # Decision variables are [fraction of each target move, absolute
            # resulting model weight]. This is a small LP with a unique set of
            # economic constraints even where the solver has tied optima.
            objective = np.zeros(2 * n, dtype=float)
            objective[:n] = -margin
            a_ub: list[np.ndarray] = []
            b_ub: list[float] = []
            for i in range(n):
                row_pos = np.zeros(2 * n, dtype=float)
                row_pos[i] = delta[i]
                row_pos[n + i] = -1.0
                a_ub.append(row_pos)
                b_ub.append(-previous[i])
                row_neg = np.zeros(2 * n, dtype=float)
                row_neg[i] = -delta[i]
                row_neg[n + i] = -1.0
                a_ub.append(row_neg)
                b_ub.append(previous[i])
            gross_row = np.zeros(2 * n, dtype=float)
            gross_row[n:] = 1.0
            a_ub.append(gross_row)
            b_ub.append(MAX_GROSS)
            a_eq = np.zeros((1, 2 * n), dtype=float)
            a_eq[0, :n] = delta
            b_eq = np.array([-float(previous.sum())], dtype=float)
            bounds = [
                (0.0, 1.0 if bool(eligible[i]) else 0.0) for i in range(n)
            ] + [(0.0, None) for _ in range(n)]
            lp = linprog(
                objective,
                A_ub=np.asarray(a_ub),
                b_ub=np.asarray(b_ub),
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if not lp.success:
                raise RuntimeError(f"order gate LP failed on {date_str}: {lp.message}")
            fractions = np.clip(lp.x[:n], 0.0, 1.0)
            chosen = previous + fractions * delta
            # Remove solver-scale numerical noise only; do not repair exposure.
            chosen[np.abs(chosen) < 1e-13] = 0.0
            if abs(float(chosen.sum())) > 1e-7:
                raise AssertionError(f"gate broke market neutrality on {date_str}: {chosen.sum():.3e}")
            if float(np.abs(chosen).sum()) > MAX_GROSS + 1e-7:
                raise AssertionError(f"gate broke gross constraint on {date_str}")
            status = "gated"

        actual_delta = chosen - previous
        rows.append(
            {
                "trade_date": date,
                "eligible_order_count": int(eligible.sum()),
                "executed_order_count": int((np.abs(actual_delta) > WEIGHT_EPS).sum()),
                "target_change_l1": float(np.abs(target - previous).sum()),
                "executed_change_l1": float(np.abs(actual_delta).sum()),
                "predicted_incremental_return": float(np.dot(actual_delta, mu) * side_leverage),
                "estimated_order_cost": float(estimated_cost[eligible].sum()) if not bool(fallback.iloc[row_index]) else 0.0,
                "mean_eligible_margin": float((edge[eligible] - estimated_cost[eligible]).mean()) if eligible.any() else 0.0,
                "gate_status": status,
                "model_net": float(chosen.sum()),
                "model_gross": float(np.abs(chosen).sum()),
            }
        )
        output[row_index] = chosen
        previous = chosen

    weights = pd.DataFrame(output, index=dates, columns=JP_TICKERS)
    return weights, pd.DataFrame(rows).set_index("trade_date")


def _simulate(
    weights: pd.DataFrame,
    df_exec: pd.DataFrame,
    app_config: Any,
    bps_per_side: float,
    fallback_flags: pd.Series | None = None,
) -> dict[str, Any]:
    dates = pd.DatetimeIndex(weights.index)
    open_910 = build_open_910_returns(df_exec, JP_TICKERS)
    targets, gaps, morning_returns = BacktestEngine._compute_price_intervals(
        df_exec,
        pd.DatetimeIndex(df_exec.index),
        dates,
        open_910_returns=open_910,
    )
    params = _cost_params(app_config, bps_per_side)
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
    fallback_array = (
        np.zeros(len(dates), dtype=bool)
        if fallback_flags is None
        else fallback_flags.reindex(dates).to_numpy(dtype=bool)
    )
    return BacktestEngine._assemble_v2_results(
        pnl,
        weights,
        fallback_array,
        [{"trade_date": str(date.date())} for date in dates],
        dates,
        float(params["alpha_long"]),
        float(params["alpha_short"]),
        float(params["side_leverage"]),
    )


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    net = result["daily_returns"].astype(float)
    gross = result["daily_returns_gross"].astype(float)
    costs = result["daily_costs"].astype(float)
    components = sum(
        result[f"daily_{part}_costs"].astype(float)
        for part in ("slip", "financing", "borrow", "reverse")
    )
    np.testing.assert_allclose(gross - costs, net, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(components, costs, atol=1e-14, rtol=0.0)
    spec = MetricsSpec(
        frequency="daily", annualization_periods=ANNUALIZATION_DAYS, include_flat_days=True
    )
    net_stats = calculate_metrics(net, spec=spec)
    gross_stats = calculate_metrics(gross, spec=spec)
    weights = result["weights"].astype(float)
    side_leverage = float(result["side_leverage"])
    model_gross_max = float(np.abs(weights).sum(axis=1).max())
    return {
        "days": int(len(net)),
        "start": str(net.index[0].date()),
        "end": str(net.index[-1].date()),
        "net_sharpe": float(net_stats["Sharpe"]),
        "gross_sharpe": float(gross_stats["Sharpe"]),
        "annualized_net_return": float(net_stats["AR"]),
        "net_sum_simple": float(net.sum()),
        "gross_sum_simple": float(gross.sum()),
        "max_drawdown": float(compute_drawdown_series(net).min()),
        "mean_turnover": float(result["daily_turnover"].mean()),
        "mean_model_gross": float(result["daily_gross_exps"].mean()),
        "model_net_abs_max": float(np.abs(weights.sum(axis=1)).max()),
        "model_gross_max": model_gross_max,
        "side_leverage": side_leverage,
        "mean_effective_gross": float(result["daily_gross_exps"].mean()) * side_leverage,
        "max_effective_gross": model_gross_max * side_leverage,
        "cost_components": {
            part: float(result[f"daily_{part}_costs"].astype(float).sum())
            for part in ("slip", "financing", "borrow", "reverse")
        },
        "total_cost_sum": float(costs.sum()),
        "fallback_days": int(result["daily_fallback"].sum()),
        "returns": net.tolist(),
    }


def _paired_bootstrap_delta(base: pd.Series, candidate: pd.Series) -> dict[str, Any]:
    delta = (candidate.astype(float) - base.astype(float)).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_blocks = int(np.ceil(len(delta) / BLOCK_DAYS))
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for sample_idx in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, len(delta), n_blocks)
        indices = np.concatenate(
            [(start + np.arange(BLOCK_DAYS)) % len(delta) for start in starts]
        )[: len(delta)]
        samples[sample_idx] = float(delta[indices].mean())
    return {
        "mean_daily_net_delta": float(delta.mean()),
        "sum_daily_net_delta": float(delta.sum()),
        "block_days": BLOCK_DAYS,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_daily_delta_ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_report(output: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(_jsonable(output), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 収益改善順序10：期待利益対注文コストの一次検証",
        "",
        f"実行日時: {output['run_at']}。発注・本番設定・モデルartifactの変更は行っていない。",
        "",
        "## 仮説と固定設計",
        "",
        "V2の9:10時点予測 `mu_gap` を使い、前日の選択ウェイトから当日のcanonical目標ウェイトへ動かす各銘柄の予測増分利益が、推定注文コストを上回る場合だけその方向の注文を許す。注文サイズは0〜目標差分の範囲で選び、市場中立・model gross≤2.0を保つ。",
        f"- 評価期間: {output['period']['start']}〜{output['period']['end']} ({output['period']['days']}営業日)。既存stage5と同じ既知OOSであり、未使用の独立holdoutではない。",
        f"- 対象銘柄: TOPIX-17。設定: `{output['config']}`。gap入力: `{output['gap_input']}`。ML overlay: `{output['overlay_dir']}`。",
        f"- 実行コマンド: `{output['run_command']}`。中心scenarioの解決済み設定: slippage={output['resolved_cost_parameters']['slip_bps']:.1f}bps/side, financing={output['resolved_cost_parameters']['fin_annual']:.2%}/年, borrow={output['resolved_cost_parameters']['borrow_annual']:.2%}/年, reverse={output['resolved_cost_parameters']['rev_bps']:.1f}bps/日, overnight alpha long/short={output['resolved_cost_parameters']['alpha_long']:.2f}/{output['resolved_cost_parameters']['alpha_short']:.2f}, side leverage={output['resolved_cost_parameters']['side_leverage']:.2f}。",
        "- 予測利益: `side_leverage × Δweight_i × mu_gap_i`。実現リターンはゲートに使わない。",
        "- 注文費用代理: 0.25%をfull spread幅と見て12.5bps/sideを中心に設定。日中取引のentryとclose部分、増分carryをコスト比較に含む。±20%感度は10/15bps/side。",
        "- 約定損益は各費用シナリオと同じ片道slippageを共有PnL計算へ渡し、financing/borrow/reverseも同じ解決済み設定で計上。",
        "- 部分約定、板厚、価格インパクト、ticker別spread、lot丸めは未モデル化。prev weightは実在庫ではない。",
        "",
        "## 正本baselineの再現",
        "",
        f"- canonical weights最大差: {output['baseline_reproduction']['max_weight_error']:.3e}。",
        f"- 5bps/side netリターン最大差: {output['baseline_reproduction']['max_daily_net_error']:.3e}。",
        "- 予測は各日のV2 `mu_gap` から回収し、全日についてbaselineの予測ポートフォリオ平均と照合した。",
        "",
        "## 結果（同じ費用前提のbaselineとの比較）",
        "",
        "| gate cost (bps/side) | variant | net Sharpe | gross Sharpe | max DD | mean turnover | cost sum | mean net Δ/day | block-bootstrap 95% CI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for bps in COST_SCENARIOS_BPS_PER_SIDE:
        key = f"{bps:.1f}"
        base = output["scenarios"][key]["baseline"]
        candidate = output["scenarios"][key]["gated"]
        bootstrap = output["scenarios"][key]["paired_bootstrap"]
        ci = bootstrap["mean_daily_delta_ci95"]
        lines.append(
            f"| {bps:.1f} | baseline | {base['net_sharpe']:.4f} | {base['gross_sharpe']:.4f} | "
            f"{base['max_drawdown']:.2%} | {base['mean_turnover']:.4f} | {base['total_cost_sum']:.6f} | — | — |"
        )
        lines.append(
            f"| {bps:.1f} | gated | {candidate['net_sharpe']:.4f} | {candidate['gross_sharpe']:.4f} | "
            f"{candidate['max_drawdown']:.2%} | {candidate['mean_turnover']:.4f} | {candidate['total_cost_sum']:.6f} | "
            f"{bootstrap['mean_daily_net_delta']:+.7f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] |"
        )
    central = output["scenarios"][f"{CENTRAL_COST_BPS_PER_SIDE:.1f}"]
    gate_stats = central["gate_diagnostics"]
    lines.extend(
        [
            "",
            f"中心シナリオの注文診断: 閾値を超えた銘柄注文 {gate_stats['eligible_order_count']}件、実際に動かした銘柄注文 {gate_stats['executed_order_count']}件、変更があった日 {gate_stats['active_days']} / {output['period']['days']}日。モデル制約の最大違反量は net {gate_stats['max_abs_model_net']:.3e}, gross超過 {gate_stats['max_gross_excess']:.3e}。side leverage={central['gated']['side_leverage']:.2f}適用後の平均実効grossはbaseline {central['baseline']['mean_effective_gross']:.3f}、gated {central['gated']['mean_effective_gross']:.3f}、最大 {central['gated']['max_effective_gross']:.2f}。",
            "",
            "## コスト内訳（中心シナリオ）",
            "",
            "| 系列 | baseline | gated | 差分 |",
            "|---|---:|---:|---:|",
        ]
    )
    for part in ("slip", "financing", "borrow", "reverse"):
        base_cost = central["baseline"]["cost_components"][part]
        gate_cost = central["gated"]["cost_components"][part]
        lines.append(f"| {part} | {base_cost:.6f} | {gate_cost:.6f} | {gate_cost-base_cost:+.6f} |")
    dsr = output["dsr"]
    lines.extend(
        [
            "",
            "## 統計評価と判定",
            "",
            "- 20営業日paired block bootstrap: 1,000回、seed=42。区間は日次平均net差の不確実性で、Sharpe差の有意性を意味しない。",
            f"- 名目DSR (中心variant; 同一検証内の3コスト水準×baseline/gatedの6候補): {dsr['value'] if dsr['value'] is not None else '算出不能'}。既存の関連no-trade試行とこのOOSの事前閲覧分を完全に網羅した総試行数ではないため、採否を確定する証拠としては使わない。",
            f"- 事前ゲート: 中心12.5bpsでnet Sharpeが同費用baseline以上、max DDが悪化せず、turnoverとcost sumが減ること。結果: `{output['predeclared_gate_pass']}`。",
            f"- 研究判定: **{output['decision']}**。{output['decision_reason']}",
            "- 0.25% spread換算は約定実績ではない。現行データでは真の9:10板・fill価格が不足しているため、仮に数値ゲートを通過しても本番採用は保留。",
            "",
            "## 再現成果物",
            "",
            f"- script: `{output['script_path']}`",
            f"- results: `{RESULTS_PATH.relative_to(ROOT)}`",
            f"- candidate weights: `{WEIGHTS_PATH.relative_to(ROOT)}`",
            f"- per-day gate diagnostics: `{DECISIONS_PATH.relative_to(ROOT)}`",
            "- registry: `var/experiments/registry.jsonl`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    baseline_weights, baseline_returns = _load_baseline()
    run_result, forecasts, app_config, df_exec = _run_canonical_and_capture()
    dates = pd.DatetimeIndex(run_result["weights"].index).normalize()
    run_weights = run_result["weights"].copy()
    run_weights.index = dates
    fallback = run_result["daily_fallback"].copy()
    fallback.index = dates

    scenarios: dict[str, Any] = {}
    all_decisions: list[pd.DataFrame] = []
    all_gate_weights: dict[str, pd.DataFrame] = {}
    for bps in COST_SCENARIOS_BPS_PER_SIDE:
        params = _cost_params(app_config, bps)
        gate_weights, diagnostics = _order_cost_gate(
            run_weights,
            forecasts,
            fallback,
            bps_per_side=bps,
            cost_params=params,
            dates=dates,
        )
        gate_weights.columns = pd.MultiIndex.from_product([[f"bps_{bps:.1f}"], JP_TICKERS])
        all_gate_weights.update({(f"bps_{bps:.1f}", ticker): gate_weights[(f"bps_{bps:.1f}", ticker)] for ticker in JP_TICKERS})
        diagnostics = diagnostics.copy()
        diagnostics["cost_bps_per_side"] = bps
        diagnostics["model_net"] = gate_weights.sum(axis=1).to_numpy()
        diagnostics["model_gross"] = gate_weights.abs().sum(axis=1).to_numpy()
        all_decisions.append(diagnostics.assign(cost_bps_per_side=bps))

        baseline_result = _simulate(run_weights, df_exec, app_config, bps, fallback)
        gated_result = _simulate(
            gate_weights.set_axis(JP_TICKERS, axis=1), df_exec, app_config, bps, fallback
        )
        base_metrics = _metrics(baseline_result)
        gate_metrics = _metrics(gated_result)
        paired = _paired_bootstrap_delta(
            baseline_result["daily_returns"], gated_result["daily_returns"]
        )
        scenarios[f"{bps:.1f}"] = {
            "baseline": base_metrics,
            "gated": gate_metrics,
            "paired_bootstrap": paired,
            "gate_diagnostics": {
                "eligible_order_count": int(diagnostics["eligible_order_count"].sum()),
                "executed_order_count": int(diagnostics["executed_order_count"].sum()),
                "active_days": int((diagnostics["executed_order_count"] > 0).sum()),
                "max_abs_model_net": float(diagnostics["model_net"].abs().max()),
                "max_gross_excess": float(max(0.0, diagnostics["model_gross"].max() - MAX_GROSS)),
                "total_target_change_l1": float(diagnostics["target_change_l1"].sum()),
                "total_executed_change_l1": float(diagnostics["executed_change_l1"].sum()),
                "predicted_incremental_return_sum": float(diagnostics["predicted_incremental_return"].sum()),
                "estimated_order_cost_sum": float(diagnostics["estimated_order_cost"].sum()),
            },
        }

    # Export a clear, flat column schema for the candidate weight paths.
    candidate_frames = []
    for bps in COST_SCENARIOS_BPS_PER_SIDE:
        frame = pd.DataFrame(
            np.column_stack(
                [all_gate_weights[(f"bps_{bps:.1f}", ticker)].to_numpy() for ticker in JP_TICKERS]
            ),
            index=dates,
            columns=[f"{ticker}_bps_{bps:.1f}" for ticker in JP_TICKERS],
        )
        candidate_frames.append(frame)
    candidate_weights_export = pd.concat(candidate_frames, axis=1)
    decisions_export = pd.concat(all_decisions).sort_index()

    central_key = f"{CENTRAL_COST_BPS_PER_SIDE:.1f}"
    central = scenarios[central_key]
    central_base = central["baseline"]
    central_gate = central["gated"]
    predeclared_gate_pass = bool(
        central_gate["net_sharpe"] >= central_base["net_sharpe"]
        and central_gate["max_drawdown"] >= central_base["max_drawdown"]
        and central_gate["mean_turnover"] < central_base["mean_turnover"]
        and central_gate["total_cost_sum"] < central_base["total_cost_sum"]
    )
    trial_sharpes = [
        float(scenarios[f"{bps:.1f}"][variant]["net_sharpe"])
        for bps in COST_SCENARIOS_BPS_PER_SIDE
        for variant in ("baseline", "gated")
    ]
    dsr_value = compute_deflated_sharpe(
        {
            "net_sharpe": central_gate["net_sharpe"],
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": ANNUALIZATION_DAYS,
            "trials": len(trial_sharpes),
            "n_observations": central_gate["days"],
            "returns": central_gate["returns"],
            "trial_sharpes": trial_sharpes,
        }
    )
    if predeclared_gate_pass:
        decision = "pending"
        reason = "数値ゲートは通過したが、既知OOSと代理コストだけなので実約定・独立OOSによる確認待ち。"
    else:
        decision = "rejected"
        reason = "中心シナリオが事前のSharpe/DD/turnover/cost条件を満たさなかった。"

    baseline_original = baseline_returns
    five_bps_original_result = _simulate(run_weights, df_exec, app_config, 5.0, fallback)
    five_bps_original = five_bps_original_result["daily_returns"].astype(float)
    five_bps_original.index = pd.DatetimeIndex(five_bps_original.index).normalize()
    return_error = float(np.max(np.abs(five_bps_original.to_numpy() - baseline_original.to_numpy())))
    output: dict[str, Any] = {
        "run_at": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).relative_to(ROOT)),
        "run_command": RUN_COMMAND,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "gap_input": str(GAP_DIR.relative_to(ROOT)),
        "overlay_dir": str(OVERLAY_DIR.relative_to(ROOT)),
        "source_weights_sha256": _sha256(BASELINE_DIR / "daily_weights.csv"),
        "source_returns_sha256": _sha256(BASELINE_DIR / "daily_net_returns.csv"),
        "period": {
            "start": str(dates[0].date()),
            "end": str(dates[-1].date()),
            "days": int(len(dates)),
            "is_fresh_holdout": False,
        },
        "baseline_reproduction": {
            "max_weight_error": float(run_result["canonical_weight_reproduction_max_error"]),
            "max_daily_net_error": float(run_result["canonical_return_reproduction_max_error"]),
            "baseline_5bps_source_return_error": return_error,
        },
        "scenario_cost_bps_per_side": list(COST_SCENARIOS_BPS_PER_SIDE),
        "central_cost_bps_per_side": CENTRAL_COST_BPS_PER_SIDE,
        "resolved_cost_parameters": _cost_params(app_config, CENTRAL_COST_BPS_PER_SIDE),
        "cost_interpretation": "0.25% full spread -> 12.5 bps per crossing side from mid; sensitivity 10/15 bps per side",
        "order_cost_gate": {
            "expected_edge": "side_leverage * delta_weight * mu_gap_i",
            "estimated_cost": "side_leverage * [one_way_bps * abs(delta_weight) * (1 + expected_intraday_close_fraction) + incremental overnight carry]",
            "eligible_rule": "expected_edge > estimated_cost, strictly",
            "weight_path": "fractional 0..1 move from previous selected model weight toward canonical target",
            "constraints": {"model_net_abs_max": 0.05, "model_gross_max": MAX_GROSS, "solver": "scipy.optimize.linprog(method='highs')"},
            "per_ticker_spread_data_available": False,
            "fill_model": "full fills at the modeled spread; no partial fill/queue/impact/lot rounding",
        },
        "annualization_days": ANNUALIZATION_DAYS,
        "bootstrap": {"block_days": BLOCK_DAYS, "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "scenarios": scenarios,
        "trial_sharpes_for_nominal_dsr": trial_sharpes,
        "dsr": {"value": dsr_value, "trials": len(trial_sharpes), "scope": "same-run 3 cost sensitivities x baseline/gate"},
        "predeclared_gate_pass": predeclared_gate_pass,
        "decision": decision,
        "decision_reason": reason,
        "production_changed": False,
        "limitations": [
            "The evaluation dates are the already reviewed 368-day Stage 5 period, not a fresh independent holdout.",
            "Current weights are previous model weights, not broker-confirmed inventory.",
            "0.25% spread is a market-making standard scenario, not an observed fill price or guaranteed spread for our orders.",
            "All forecasts are point-in-time mu_gap predictions, but the forecast benefit includes only 09:10-to-close expected returns; future overnight gap returns are not forecast by this gate.",
            "Equal spread cost across TOPIX-17 ignores ticker-specific tick sizes, depth, order size impact, fill probability, and lot rounding.",
            "Broker commissions and other explicit ticket fees are not included because order-linked historical fees are unavailable.",
            "DSR only covers the six variants compared here and does not fully count all historical/unregistered research trials; it is not adoption-grade.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_weights_export.to_csv(WEIGHTS_PATH, index_label="trade_date")
    decisions_export.to_csv(DECISIONS_PATH, index_label="trade_date")
    _write_report(output)

    registry = ExperimentRegistry(REGISTRY_PATH)
    known_trials = registry.count_trials()
    record = ExperimentRecord(
        "20260923_order_cost_gate_primary",
        "Only trade the profitable fraction of each move toward the daily target when point-in-time expected incremental return exceeds estimated order cost, while preserving market neutrality and gross limits.",
        end_time=datetime.now(UTC),
        parameters={
            "source_period": [START_DATE, END_DATE],
            "source_weights_sha256": output["source_weights_sha256"],
            "bps_per_side_sensitivities": list(COST_SCENARIOS_BPS_PER_SIDE),
            "central_bps_per_side": CENTRAL_COST_BPS_PER_SIDE,
            "bootstrap": output["bootstrap"],
            "net_neutrality_max": 0.05,
            "gross_max": MAX_GROSS,
            "historical_registry_records_before_run": known_trials,
        },
        metrics={
            "net_sharpe": central_gate["net_sharpe"],
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": ANNUALIZATION_DAYS,
            "n_observations": central_gate["days"],
            "returns": central_gate["returns"],
            "trial_sharpes": trial_sharpes,
            "trials": len(trial_sharpes),
            "nominal_dsr": dsr_value,
            "predeclared_gate_pass": predeclared_gate_pass,
            "max_daily_net_reproduction_error": output["baseline_reproduction"]["max_daily_net_error"],
            "paired_mean_daily_net_delta": central["paired_bootstrap"]["mean_daily_net_delta"],
            "paired_ci95_mean_daily_net_delta": central["paired_bootstrap"]["mean_daily_delta_ci95"],
            "fallback_days": central_gate["fallback_days"],
            "is_fresh_holdout": False,
        },
        decision=Decision.PENDING if decision == "pending" else Decision.REJECTED,
        report_path=str(REPORT_PATH.relative_to(ROOT)),
    )
    registry.record(record)

    if decision == "rejected":
        graveyard = ROOT / "docs" / "experiment_graveyard.md"
        with graveyard.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n- **Per-order expected-edge cost gate 不採用（2026-09-23）**: "
                f"一次代理検証の中心12.5bps/sideで事前ゲートを満たさず（net Sharpe "
                f"{central_gate['net_sharpe']:.4f} vs same-cost baseline {central_base['net_sharpe']:.4f}, "
                f"max DD {central_gate['max_drawdown']:.2%} vs {central_base['max_drawdown']:.2%}, "
                f"turnover {central_gate['mean_turnover']:.4f} vs {central_base['mean_turnover']:.4f}）。"
                f"固定full-spread代理であり実約定費用検証ではない。詳細は `{REPORT_PATH.relative_to(ROOT)}`。\n"
            )


if __name__ == "__main__":
    main()
