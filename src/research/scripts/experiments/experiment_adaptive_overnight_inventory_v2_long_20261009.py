#!/usr/bin/env python3
"""Long historical V2-only replay of adaptive overnight inventory sizing."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision
from leadlag.reporting.metrics import MetricsSpec
from research.experiment_utils import record_backtest_experiment
from research.overnight_inventory import adaptive_carry_masks, realized_inventory_reuse
from research.scripts.experiments.experiment_adaptive_overnight_inventory_20261008 import (
    _calendar_days,
    _fixed_masks,
    _gross_stats,
    _paired_block_bootstrap,
    _policy_config,
    _read_yaml,
    _replay,
    _safe_json,
    _segment_rows,
    _stats,
    _summary_row,
    _weighted_alpha_mean,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "configs/research/adaptive_overnight_inventory_v2_long_20261009.yaml"
PRODUCTION_CONFIG_PATH = ROOT / "configs/production/production.yaml"
OUTPUT_DIR = ROOT / "reports/20261009_adaptive_overnight_inventory_v2_long"
REPORT_PATH = OUTPUT_DIR / "report.md"
STUDY_ID = "adaptive-overnight-inventory-v2-long-2026-10-09"
ANNUALIZATION = 245


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gap_store_coverage(path: Path) -> dict[str, Any]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT COUNT(DISTINCT trade_date), MIN(trade_date), MAX(trade_date) "
            "FROM gap_matrices WHERE matrix_type='mu' AND horizon=-1"
        ).fetchone()
    if row is None or not row[0]:
        return {"dates": 0, "start": None, "end": None}
    return {"dates": int(row[0]), "start": str(row[1]), "end": str(row[2])}


def _period_summary(
    *,
    policy: str,
    replay: dict[str, Any],
    weights: np.ndarray,
    alpha_masks: np.ndarray,
    dates: pd.DatetimeIndex,
    mask: np.ndarray,
    side_leverage: float,
) -> dict[str, Any]:
    daily = replay["daily"].loc[mask]
    selected_weights = weights[mask]
    selected_alpha = alpha_masks[mask]
    carried = selected_weights * selected_alpha
    row: dict[str, Any] = {
        "policy": policy,
        "start": str(dates[mask][0].date()),
        "end": str(dates[mask][-1].date()),
        **_stats(daily["net"]),
        **_gross_stats(daily["gross"]),
        "slippage_cost": float(daily["slippage"].sum()),
        "financing_cost": float(daily["financing"].sum()),
        "borrow_cost": float(daily["borrow"].sum()),
        "reverse_cost": float(daily["reverse"].sum()),
        "carry_cost": float(daily[["financing", "borrow", "reverse"]].to_numpy().sum()),
        "mean_daily_turnover": float(daily["turnover"].mean()),
        "fallback_rate": float(daily["fallback"].mean()),
        "mean_carry_alpha_long": _weighted_alpha_mean(
            selected_alpha, selected_weights, positive=True
        ),
        "mean_carry_alpha_short": _weighted_alpha_mean(
            selected_alpha, selected_weights, positive=False
        ),
        "mean_carry_net_exposure": float((carried.sum(axis=1) * side_leverage).mean()),
        "max_abs_carry_net_exposure": float(
            np.abs(carried.sum(axis=1) * side_leverage).max()
        ),
        "mean_carry_gross_exposure": float(
            (np.abs(carried).sum(axis=1) * side_leverage).mean()
        ),
        **realized_inventory_reuse(weights=selected_weights, alpha_masks=selected_alpha),
    }
    return row


def _period_masks(dates: pd.DatetimeIndex, eligible_start: int) -> dict[str, np.ndarray]:
    eligible = np.arange(len(dates)) >= eligible_start
    masks: dict[str, np.ndarray] = {"full_oos": eligible}
    years = dates.year
    for year in sorted(set(years[eligible])):
        masks[str(year)] = eligible & (years == year)
    return masks


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    research_config = _read_yaml(CONFIG_PATH)
    app_config = load_config_from_yaml(PRODUCTION_CONFIG_PATH)
    costs = app_config.v2.costs
    alpha_long = float(costs.overnight_alpha_long)
    alpha_short = float(costs.overnight_alpha_short)
    if (alpha_long, alpha_short) != (0.75, 0.50):
        raise ValueError(f"Expected fixed baseline 0.75/0.50, got {(alpha_long, alpha_short)}")
    if app_config.v2.ml_overlay_enabled or app_config.v2.ml_overlay_model_dir:
        raise ValueError("This follow-up requires the resolved V2-only production configuration")

    bt = research_config["backtest"]
    start_date = str(bt["start_date"])
    end_date = str(bt["end_date"])
    df_exec_full = load_df_exec_from_local_cache(max_stale_bdays=None)
    df_exec = df_exec_full.loc[df_exec_full.index <= pd.Timestamp(end_date)].copy()
    if pd.Timestamp(df_exec.index.max()).normalize() != pd.Timestamp(end_date):
        raise ValueError(f"Execution cache does not contain the configured end date {end_date}")

    gap_store = ROOT / app_config.v2.gap_input_dir
    if not gap_store.exists():
        raise FileNotFoundError(f"Configured gap store does not exist: {gap_store}")
    logger.info(
        "Starting long V2-only generation: %s..%s; input rows %d..%d; overlay disabled",
        start_date,
        end_date,
        len(df_exec),
        len(df_exec_full),
    )
    v2_results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_store,
        df_exec=df_exec,
        start_date=start_date,
        end_date=end_date,
        n_jobs=1,
    )
    weights_frame = v2_results["weights"].reindex(columns=JP_TICKERS).astype(float)
    dates = pd.DatetimeIndex(weights_frame.index).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("V2 weights have duplicated or unordered dates")
    weights = weights_frame.to_numpy(dtype=float)
    fallback = v2_results["daily_fallback"].reindex(dates).astype(bool)
    if not fallback.index.equals(dates):
        raise ValueError("V2 fallback flags are not aligned to weights")
    if not np.isfinite(weights).all():
        raise ValueError("V2 weights contain non-finite values")

    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(
        df_exec, start_date, end_date, 0
    )
    expected_dates = pd.DatetimeIndex(sim_dates[start_idx : end_idx + 1]).normalize()
    if not dates.equals(expected_dates):
        raise ValueError("V2 weight dates do not match the requested execution-date slice")
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    target_returns, gap_returns, morning_returns = BacktestEngine._compute_price_intervals(
        df_exec, sim_dates, pd.DatetimeIndex(sim_dates[start_idx : end_idx + 1]), open_910_returns
    )
    gap_returns = (1.0 + gap_returns) * (1.0 + morning_returns) - 1.0
    calendar_days = _calendar_days(dates)

    slip = float(costs.slippage_bps_per_side) / 10_000.0
    financing_annual = float(costs.buy_interest_annual)
    borrow_annual = float(costs.borrow_fee_annual)
    reverse_fee_bps = float(costs.reverse_fee_bps)
    side_leverage = float(costs.side_leverage)
    baseline_masks = _fixed_masks(weights, alpha_long, alpha_short)
    main_config = _policy_config(research_config)
    adaptive_masks, diagnostics = adaptive_carry_masks(
        weights=weights,
        gap_returns=gap_returns,
        sim_dates=dates,
        slip=slip,
        financing_annual=financing_annual,
        borrow_annual=borrow_annual,
        reverse_fee_bps=reverse_fee_bps,
        config=main_config,
    )
    warmup = main_config.lookback_sessions + 1
    if len(dates) <= warmup + 500:
        raise ValueError(f"Too few dates for long evaluation after {warmup} warm-up sessions")
    oos_start_index = warmup
    oos_dates = dates[oos_start_index:]

    policy_masks: dict[str, np.ndarray] = {
        "fixed_alpha_0.75_0.50": baseline_masks,
        "adaptive_252": adaptive_masks,
    }
    model = research_config["model"]
    for multiplier in model["sensitivity"]["lookback_multipliers"]:
        lookback = int(round(main_config.lookback_sessions * float(multiplier)))
        policy_masks[f"adaptive_lookback_{lookback}"] = adaptive_carry_masks(
            weights=weights,
            gap_returns=gap_returns,
            sim_dates=dates,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=borrow_annual,
            reverse_fee_bps=reverse_fee_bps,
            config=_policy_config(research_config, lookback=lookback),
        )[0]
    for multiplier in model["sensitivity"]["reversal_exit_cost_multipliers"]:
        value = float(multiplier)
        policy_masks[f"adaptive_reversal_cost_{value:.1f}x"] = adaptive_carry_masks(
            weights=weights,
            gap_returns=gap_returns,
            sim_dates=dates,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=borrow_annual,
            reverse_fee_bps=reverse_fee_bps,
            config=_policy_config(research_config, reversal_multiplier=value),
        )[0]

    replay_by_policy: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    daily_outputs: list[pd.DataFrame] = []
    for name, masks in policy_masks.items():
        replay = _replay(
            weights=weights,
            target_returns=target_returns,
            gap_returns=gap_returns,
            dates=dates,
            alpha_masks=masks,
            calendar_days=calendar_days,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=borrow_annual,
            reverse_fee_bps=reverse_fee_bps,
            side_leverage=side_leverage,
            fallback=fallback,
            policy_name=name,
        )
        replay_by_policy[name] = replay
        daily_outputs.append(replay["daily"])
        row = _summary_row(
            name=name,
            replay=replay,
            alpha_masks=masks,
            weights=weights,
            dates=dates,
            oos_start_index=oos_start_index,
        )
        row["mean_carry_net_exposure"] = float(
            (weights[oos_start_index:] * masks[oos_start_index:]).sum(axis=1).mean()
            * side_leverage
        )
        row["max_abs_carry_net_exposure"] = float(
            np.abs(
                (weights[oos_start_index:] * masks[oos_start_index:]).sum(axis=1)
                * side_leverage
            ).max()
        )
        row["mean_carry_gross_exposure"] = float(
            (np.abs(weights[oos_start_index:] * masks[oos_start_index:]).sum(axis=1).mean())
            * side_leverage
        )
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)

    stress_rows: list[dict[str, Any]] = []
    for stress_borrow in research_config["cost_stress"]["high_short_borrow_annual"]:
        stress_borrow = float(stress_borrow)
        stress_masks = adaptive_carry_masks(
            weights=weights,
            gap_returns=gap_returns,
            sim_dates=dates,
            slip=slip,
            financing_annual=financing_annual,
            borrow_annual=stress_borrow,
            reverse_fee_bps=reverse_fee_bps,
            config=main_config,
        )[0]
        for name, masks in (
            ("fixed_alpha_0.75_0.50", baseline_masks),
            ("adaptive_252", stress_masks),
        ):
            replay = _replay(
                weights=weights,
                target_returns=target_returns,
                gap_returns=gap_returns,
                dates=dates,
                alpha_masks=masks,
                calendar_days=calendar_days,
                slip=slip,
                financing_annual=financing_annual,
                borrow_annual=stress_borrow,
                reverse_fee_bps=reverse_fee_bps,
                side_leverage=side_leverage,
                fallback=fallback,
                policy_name=f"{name}_borrow_{stress_borrow:.0%}",
            )
            row = _summary_row(
                name=name,
                replay=replay,
                alpha_masks=masks,
                weights=weights,
                dates=dates,
                oos_start_index=oos_start_index,
            )
            row["short_borrow_annual"] = stress_borrow
            stress_rows.append(row)
            daily_outputs.append(replay["daily"])

    segment_rows: list[dict[str, Any]] = []
    for name in ("fixed_alpha_0.75_0.50", "adaptive_252"):
        segment_rows.extend(
            {"policy": name, **row}
            for row in _segment_rows(
                replay=replay_by_policy[name],
                weights=weights,
                alpha_masks=policy_masks[name],
                target_returns=target_returns,
                gap_returns=gap_returns,
                dates=dates,
                calendar_days=calendar_days,
                oos_start_index=oos_start_index,
                slip=slip,
                financing_annual=financing_annual,
                borrow_annual=borrow_annual,
                reverse_fee_bps=reverse_fee_bps,
                side_leverage=side_leverage,
            )
        )
    segment_frame = pd.DataFrame(segment_rows)

    period_rows: list[dict[str, Any]] = []
    period_masks = _period_masks(dates, oos_start_index)
    for period, mask in period_masks.items():
        for name in ("fixed_alpha_0.75_0.50", "adaptive_252"):
            row = _period_summary(
                policy=name,
                replay=replay_by_policy[name],
                weights=weights,
                alpha_masks=policy_masks[name],
                dates=dates,
                mask=mask,
                side_leverage=side_leverage,
            )
            row["period"] = period
            period_rows.append(row)
    period_frame = pd.DataFrame(period_rows)

    baseline_daily = replay_by_policy["fixed_alpha_0.75_0.50"]["daily"].iloc[oos_start_index:]
    adaptive_daily = replay_by_policy["adaptive_252"]["daily"].iloc[oos_start_index:]
    bootstrap_cfg = bt["block_bootstrap"]
    bootstrap = _paired_block_bootstrap(
        adaptive_daily["net"].to_numpy() - baseline_daily["net"].to_numpy(),
        block_sessions=int(bootstrap_cfg["block_sessions"]),
        resamples=int(bootstrap_cfg["resamples"]),
        seed=int(bootstrap_cfg["seed"]),
    )

    audit_statuses = Counter(
        f"{summary.get('audit_status', {}).get('numerical')}|"
        f"{summary.get('audit_status', {}).get('leakage')}"
        for summary in v2_results["v2_summaries"]
        if isinstance(summary, dict) and "audit_status" in summary
    )
    gap_coverage = _read_gap_store_coverage(gap_store)
    macro_path = ROOT / "var/market_data/macro_prices_verified.pkl"
    macro_provenance_path = ROOT / "var/market_data/macro_prices_verified.provenance.json"
    macro_frame = pd.read_pickle(macro_path)
    macro_provenance = json.loads(macro_provenance_path.read_text(encoding="utf-8"))

    policy_summary_path = OUTPUT_DIR / "policy_summary.csv"
    summary_frame.to_csv(policy_summary_path, index=False, float_format="%.12g")
    period_frame.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False, float_format="%.12g")
    segment_frame.to_csv(OUTPUT_DIR / "calendar_and_reversal_segments.csv", index=False, float_format="%.12g")
    pd.DataFrame(stress_rows).to_csv(OUTPUT_DIR / "borrow_stress_summary.csv", index=False, float_format="%.12g")
    weights_frame.to_csv(OUTPUT_DIR / "daily_weights.csv", float_format="%.12g")
    pd.concat(daily_outputs).to_csv(OUTPUT_DIR / "daily_policy_replay.csv", float_format="%.12g")
    diagnostics = diagnostics.assign(
        ticker=diagnostics["ticker_index"].map({i: ticker for i, ticker in enumerate(JP_TICKERS)})
    )
    diagnostics.to_csv(OUTPUT_DIR / "ticker_carry_diagnostics.csv", index=False, float_format="%.12g")

    trial_sharpes = summary_frame["net_sharpe"].astype(float).to_numpy()
    trial_variance = float(np.var(trial_sharpes, ddof=1))
    estimated_trials = 15  # 9 previously counted candidate variants + 6 current policies.
    fixed_row = summary_frame.loc[summary_frame["policy"] == "fixed_alpha_0.75_0.50"].iloc[0]
    adaptive_row = summary_frame.loc[summary_frame["policy"] == "adaptive_252"].iloc[0]
    payload = {
        "created_date": "2026-10-09",
        "study_id": STUDY_ID,
        "hypothesis": "A frozen PIT ticker-specific overnight carry policy improves long-history V2-only net results over fixed alpha.",
        "production_changed": False,
        "production_config": str(PRODUCTION_CONFIG_PATH.relative_to(ROOT)),
        "production_config_sha256": _sha256(PRODUCTION_CONFIG_PATH),
        "research_config": str(CONFIG_PATH.relative_to(ROOT)),
        "research_config_sha256": _sha256(CONFIG_PATH),
        "source_data_start": str(df_exec.index.min().date()),
        "source_data_end": str(df_exec.index.max().date()),
        "excluded_newer_source_data_end": str(df_exec_full.index.max().date()),
        "backtest_start": str(dates[0].date()),
        "backtest_end": str(dates[-1].date()),
        "oos_start": str(oos_dates[0].date()),
        "oos_end": str(oos_dates[-1].date()),
        "n_backtest_sessions": int(len(dates)),
        "n_oos_sessions": int(len(oos_dates)),
        "initial_inventory": "flat at first backtest entry",
        "terminal_policy": "liquidate remaining inventory at final close; final carry alpha is zero",
        "inventory_state": "continuous across all dates; no resets",
        "execution_volume_contract": "side_leverage * (opening inventory flow + closing inventory flow); turnover = volume / 2",
        "side_leverage": side_leverage,
        "slippage_bps_one_way": slip * 10000,
        "buy_interest_annual": financing_annual,
        "borrow_fee_annual_baseline": borrow_annual,
        "reverse_fee_bps_per_calendar_day": reverse_fee_bps,
        "baseline_alpha_long": alpha_long,
        "baseline_alpha_short": alpha_short,
        "ml_overlay_enabled": bool(app_config.v2.ml_overlay_enabled),
        "gap_store_matrices": gap_coverage["dates"],
        "gap_store_trade_date_range": [gap_coverage["start"], gap_coverage["end"]],
        "fallback_rate_oos": float(fallback.iloc[oos_start_index:].mean()),
        "v2_audit_status_counts": dict(audit_statuses),
        "macro_snapshot_path": str(macro_path.relative_to(ROOT)),
        "macro_snapshot_sha256": _sha256(macro_path),
        "macro_snapshot_date_range": [str(macro_frame.index.min().date()), str(macro_frame.index.max().date())],
        "macro_snapshot_provenance": macro_provenance,
        "summary": summary_frame.to_dict(orient="records"),
        "yearly_summary": period_frame.to_dict(orient="records"),
        "borrow_stress": stress_rows,
        "calendar_and_reversal_segments": segment_rows,
        "paired_block_bootstrap": bootstrap,
        "trial_count_estimate": estimated_trials,
        "trial_count_basis": "Previous approximate count of 9 plus 6 current policies; historical trial series are incomplete.",
        "dsr_trial_sharpes_current_window": trial_sharpes.tolist(),
        "dsr_trial_sharpe_variance_annual_current_window": trial_variance,
        "decision": "PENDING",
        "decision_reason": "The extended historical result is not a prospective untouched holdout; actual fills and ticker-specific borrow/reverse costs remain unavailable, and historical macro-provider availability is unproven.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(_safe_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = [
        "# 銘柄別オーバーナイト在庫 sizing：V2単独の長期比較\n\n",
        "## 仮説・固定条件\n\n",
        "2026-10-08時点で固定した252営業日adaptive carry policyを、V2単独の長期ウェイト列でfixed alphaと比較する。期間を広げるだけの再検証とし、policy選択・閾値調整は行わない。\n\n",
        f"- 生成・評価対象: {dates[0].date()}–{dates[-1].date()}（{len(dates)}営業日）。評価: {oos_dates[0].date()}–{oos_dates[-1].date()}（{len(oos_dates)}営業日、252遷移warm-up）。\n",
        f"- V2単独: resolved production `ml_overlay_enabled={app_config.v2.ml_overlay_enabled}`、overlay model dir empty。V2 audit status counts={dict(audit_statuses)}。\n",
        f"- df_exec source={df_exec.index.min().date()}–{df_exec.index.max().date()}。最新market row {df_exec_full.index.max().date()} はmacro snapshot期限に合わせて除外。\n",
        f"- macro snapshot={macro_frame.index.min().date()}–{macro_frame.index.max().date()}、source SHA256={_sha256(macro_path)}。provenanceは `historical_provider_available_at_proven={macro_provenance.get('historical_provider_available_at_proven')}`。各決定は当該日closeより前のmacro行だけを使う。\n",
        f"- gap storeは{gap_coverage['dates']}日（{gap_coverage['start']}–{gap_coverage['end']}）。その他の期間はresolved V2 on-demand設定で生成。評価fallback率={payload['fallback_rate_oos']:.2%}。\n",
        f"- fixed carry alpha={alpha_long:.2f}/{alpha_short:.2f}、side leverage={side_leverage:.2f}、slippage={slip*10000:.1f}bp/side、financing={financing_annual:.2%}/年、borrow={borrow_annual:.2%}/年、reverse={reverse_fee_bps:.1f}bp/暦日。\n",
        "- inventoryはflat start・連続状態・最終引け全清算。execution volumeはopening/closing inventory flow合計、turnoverはその1/2。gross/net PnL、MDD、Sharpeは全評価日を含む。\n\n",
        "## 全期間比較\n\n",
        "PnL列は日次return fractionの合計/複利値。\n\n",
        "| Policy | n | Net Sharpe | MDD | Gross sum / comp. | Net sum / comp. | Slip | Carry | Turnover/day | Reuse | Fallback | Carry net exp. | Carry gross exp. |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in (fixed_row, adaptive_row):
        report.append(
            f"| {row['policy']} | {int(row['n'])} | {row['net_sharpe']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row['gross_pnl_sum']:.4f} / {row['gross_pnl_compounded']:.2%} | "
            f"{row['net_pnl_sum']:.4f} / {row['net_pnl_compounded']:.2%} | {row['slippage_cost']:.4f} | "
            f"{row['carry_cost']:.4f} | {row['mean_daily_turnover']:.4f} | {row['reuse_ratio']:.2%} | "
            f"{row['fallback_rate']:.2%} | {row['mean_carry_net_exposure']:.3f} | "
            f"{row['mean_carry_gross_exposure']:.3f} |\n"
        )
    report.extend(
        [
            "\n### 年別評価（2026年はYTD）\n\n",
            "| 年/期間 | Policy | n | Net Sharpe | MDD | Net sum | Slip | Carry | Turnover/day | Reuse | Carry net exp. |\n",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    yearly_rows = period_frame.loc[
        period_frame["policy"].isin(["fixed_alpha_0.75_0.50", "adaptive_252"])
        & (period_frame["period"] != "full_oos")
    ]
    for _, row in yearly_rows.iterrows():
        report.append(
            f"| {row['period']} | {row['policy']} | {int(row['n'])} | {row['net_sharpe']:.3f} | "
            f"{row['max_drawdown']:.2%} | {row['net_pnl_sum']:.4f} | {row['slippage_cost']:.4f} | "
            f"{row['carry_cost']:.4f} | {row['mean_daily_turnover']:.4f} | {row['reuse_ratio']:.2%} | "
            f"{row['mean_carry_net_exposure']:.3f} |\n"
        )
    report.extend(
        [
            "\n### 固定候補の感度確認\n\n",
            "| Policy | Net Sharpe | MDD | Net sum | Slip | Carry | Reuse | Long/short alpha |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in summary_rows:
        report.append(
            f"| {row['policy']} | {row['net_sharpe']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row['net_pnl_sum']:.4f} | {row['slippage_cost']:.4f} | {row['carry_cost']:.4f} | "
            f"{row['reuse_ratio']:.2%} | {row['mean_carry_alpha_long']:.3f}/{row['mean_carry_alpha_short']:.3f} |\n"
        )
    report.extend(
        [
            "\n### 連休・翌朝反転\n\n",
            "| Policy | Segment | n | Net sum | Mean net | Overnight | Slip | Next-open slip delta vs flat | Carry | Reuse |\n",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in segment_rows:
        report.append(
            f"| {row['policy']} | {row['segment']} | {row['n']} | {row['net_pnl_sum']:.4f} | "
            f"{row['mean_net']:.5f} | {row['overnight_pnl_sum']:.5f} | {row['slippage_cost']:.5f} | "
            f"{row['next_open_slippage_delta_vs_flat']:.5f} | {row['carry_cost']:.5f} | {row['reuse_ratio']:.2%} |\n"
        )
    report.extend(
        [
            "\n### Borrow stressと不確実性\n\n",
            "Borrow stressはuniformな仮定であり、tickerごとの実費ではない。\n\n",
            "| Borrow annual | Policy | Net Sharpe | MDD | Net sum | Borrow | Reverse | Mean short alpha |\n",
            "|---:|---|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in stress_rows:
        report.append(
            f"| {row['short_borrow_annual']:.0%} | {row['policy']} | {row['net_sharpe']:.3f} | "
            f"{row['max_drawdown']:.2%} | {row['net_pnl_sum']:.4f} | {row['borrow_cost']:.5f} | "
            f"{row['reverse_cost']:.5f} | {row['mean_carry_alpha_short']:.3f} |\n"
        )
    report.extend(
        [
            f"\nAdaptive minus fixed paired daily net delta={bootstrap['mean_daily_net_delta']:.7f}; "
            f"20-session circular block bootstrap CI95={bootstrap['ci95_mean_daily_net_delta']} "
            f"({bootstrap['resamples']} resamples, seed {bootstrap['seed']}).\n\n",
            f"DSR trial estimate={estimated_trials}; current same-period candidate Sharpe variance={trial_variance:.6f}. "
            "This remains approximate because earlier candidate return series are not available on the same dates and the complete historical trial family is unknown.\n\n",
            "## 判定と限界\n\n",
            "判定: **PENDING**。この長期再生は約10年の相場環境と多数の通常日・連休・反転遷移を含むため、159日評価の期間依存性を確認する材料になる。ただし、現時点ですでに観測可能な歴史データを使うため、将来の未観測データによるprospective OOSではない。約定価格、銘柄別借株・逆日歩、share/cash drift台帳は未取得。macro snapshotのhistorical availability provenanceも未証明。したがって長期化だけで本番採用条件は満たさない。\n\n",
            "再現コマンド:\n\n",
            "```sh\n",
            "timeout -k 10s 8100s env MPLCONFIGDIR=/private/tmp/leadlag-mpl-cache .venv/bin/python src/research/scripts/experiments/experiment_adaptive_overnight_inventory_v2_long_20261009.py 2>&1 | tee reports/20261009_adaptive_overnight_inventory_v2_long/run.log\n",
            "```\n\n",
            "Artifacts: `metrics.json`, `policy_summary.csv`, `yearly_summary.csv`, `borrow_stress_summary.csv`, `calendar_and_reversal_segments.csv`, `ticker_carry_diagnostics.csv`, `daily_weights.csv`, `daily_policy_replay.csv`.\n",
        ]
    )
    REPORT_PATH.write_text("".join(report), encoding="utf-8")

    adaptive_daily = replay_by_policy["adaptive_252"]["daily"].iloc[oos_start_index:]
    registry_record = record_backtest_experiment(
        name=Path(__file__).stem,
        hypothesis="A frozen PIT ticker-specific overnight carry sizing policy improves long-history V2-only net results over fixed alpha.",
        app_config=app_config,
        results={
            "daily_returns": adaptive_daily["net"],
            "daily_fallback": adaptive_daily["fallback"],
            "daily_turnover": adaptive_daily["turnover"],
            "daily_gross_exps": adaptive_daily["gross_exposure"],
        },
        extra_metrics={
            "study_id": STUDY_ID,
            "trials": estimated_trials,
            "trial_sharpe_variance": trial_variance,
            "trial_sharpe_variance_frequency": "annual",
            "trial_sharpes": trial_sharpes.tolist(),
            "trading_days_per_year": ANNUALIZATION,
            "include_flat_days": True,
            "oos_start": str(oos_dates[0].date()),
            "oos_end": str(oos_dates[-1].date()),
            "baseline_policy": "fixed alpha long=0.75 short=0.50",
            "adaptive_policy": main_config.__dict__,
            "paired_block_bootstrap": bootstrap,
            "production_changed": False,
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
        },
        decision=Decision.PENDING,
        reason="Long historical V2-only replay broadens the evidence base but is not a prospective holdout and lacks verified ticker-level fills/borrow costs.",
        report_path=REPORT_PATH.relative_to(ROOT),
        registry_path=ROOT / "var/experiments/registry.jsonl",
        metrics_spec=MetricsSpec(annualization_periods=ANNUALIZATION, include_flat_days=True),
        study_id=STUDY_ID,
    )
    report.append(
        f"\nRegistry: record_id=`{registry_record.record_id}`, decision=`{registry_record.decision.value}`, "
        f"DSR={registry_record.deflated_sharpe()!r}, study_id=`{STUDY_ID}`.\n"
    )
    REPORT_PATH.write_text("".join(report), encoding="utf-8")
    payload["registry_record_id"] = registry_record.record_id
    payload["deflated_sharpe"] = registry_record.deflated_sharpe()
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(_safe_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Completed long V2-only carry replay: OOS %s..%s (%d sessions); fixed Sharpe %.3f, adaptive %.3f; report=%s",
        oos_dates[0].date(),
        oos_dates[-1].date(),
        len(oos_dates),
        fixed_row["net_sharpe"],
        adaptive_row["net_sharpe"],
        REPORT_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
