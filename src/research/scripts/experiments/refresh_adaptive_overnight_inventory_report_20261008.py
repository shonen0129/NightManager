#!/usr/bin/env python3
"""Rebuild ticker-transition attribution from the saved 2026-10-08 replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_adaptive_overnight_inventory_20261008 import (  # noqa: E402
    OUTPUT_DIR,
    _calendar_days,
    _fixed_masks,
    _segment_rows,
)

from leadlag.data.intraday_inputs import build_open_910_returns  # noqa: E402
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.tickers import JP_TICKERS  # noqa: E402
from leadlag.execution.backtester import BacktestEngine  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from research.overnight_inventory import (  # noqa: E402
    AdaptiveCarryConfig,
    adaptive_carry_masks,
    attribute_inventory_transitions,
)


def _format_segment_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Policy | Segment | Unit | n | Gross sum | Net sum | Mean net / unit | Overnight | Actual slip | Next-open slip delta vs flat | Carry cost | Reuse ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['segment']} | {row['unit']} | {row['n']} | "
            f"{row['gross_pnl_sum']:.4f} | {row['net_pnl_sum']:.4f} | {row['mean_net']:.5f} | "
            f"{row['overnight_pnl_sum']:.5f} | {row['slippage_cost']:.5f} | "
            f"{row['next_open_slippage_delta_vs_flat']:.5f} | "
            f"{row['carry_cost']:.5f} | {row['reuse_ratio']:.2%} |"
        )
    return "\n".join(lines)


def main() -> int:
    metrics_path = OUTPUT_DIR / "metrics.json"
    report_path = OUTPUT_DIR / "report.md"
    log_path = OUTPUT_DIR / "run.log"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    expected_rows = int(metrics["n_backtest_sessions"])
    adr_skip_count = log_text.count(
        "ADR features were not supplied in the decision snapshot; skipping overlay"
    )
    macro_empty_count = log_text.count("Macro: data insufficient (0 rows).")
    if adr_skip_count != expected_rows or macro_empty_count != expected_rows:
        raise ValueError(
            "Run-log availability evidence does not match the saved backtest length: "
            f"ADR skips={adr_skip_count}, macro-empty={macro_empty_count}, rows={expected_rows}"
        )

    production_config = load_config_from_yaml(ROOT / "configs" / "production" / "production.yaml")
    costs = production_config.v2.costs
    if (float(costs.overnight_alpha_long), float(costs.overnight_alpha_short)) != (
        float(metrics["baseline_alpha_long"]),
        float(metrics["baseline_alpha_short"]),
    ):
        raise ValueError("Resolved baseline alpha differs from the saved experiment")
    if float(costs.side_leverage) != float(metrics["side_leverage"]):
        raise ValueError("Resolved side leverage differs from the saved experiment")

    weights_frame = pd.read_csv(OUTPUT_DIR / "daily_weights.csv", parse_dates=["trade_date"])
    weights_frame = weights_frame.set_index("trade_date").reindex(columns=JP_TICKERS)
    dates = pd.DatetimeIndex(weights_frame.index).normalize()
    weights = weights_frame.to_numpy(dtype=float)
    df_exec = load_df_exec_from_local_cache(max_stale_bdays=None)
    sim_dates, start_idx, end_idx = BacktestEngine._resolve_sim_dates(
        df_exec,
        str(metrics["backtest_start"]),
        "latest" if metrics["backtest_end"] == str(pd.Timestamp(df_exec.index.max()).date()) else metrics["backtest_end"],
        0,
    )
    expected_dates = pd.DatetimeIndex(sim_dates[start_idx : end_idx + 1]).normalize()
    if not dates.equals(expected_dates):
        raise ValueError("Saved weights no longer align with the local execution-data dates")
    target_returns, gap_returns = BacktestEngine._compute_target_and_gap_returns(
        df_exec,
        sim_dates,
        pd.DatetimeIndex(sim_dates[start_idx : end_idx + 1]),
        build_open_910_returns(df_exec, JP_TICKERS),
    )
    calendar_days = _calendar_days(dates)
    slip = float(costs.slippage_bps_per_side) / 10_000.0
    financing_annual = float(costs.buy_interest_annual)
    borrow_annual = float(costs.borrow_fee_annual)
    reverse_fee_bps = float(costs.reverse_fee_bps)
    side_leverage = float(costs.side_leverage)
    baseline_masks = _fixed_masks(
        weights, float(metrics["baseline_alpha_long"]), float(metrics["baseline_alpha_short"])
    )
    adaptive_masks, _ = adaptive_carry_masks(
        weights=weights,
        gap_returns=gap_returns,
        sim_dates=dates,
        slip=slip,
        financing_annual=financing_annual,
        borrow_annual=borrow_annual,
        reverse_fee_bps=reverse_fee_bps,
        config=AdaptiveCarryConfig(**metrics["adaptive_config"]),
    )
    policy_masks = {
        "fixed_alpha_0.75_0.50": baseline_masks,
        "adaptive_252": adaptive_masks,
    }
    oos_start_index = int(np.flatnonzero(dates == pd.Timestamp(metrics["oos_start"]))[0])
    daily_frame = pd.read_csv(OUTPUT_DIR / "daily_policy_replay.csv", parse_dates=["trade_date"])
    segment_rows: list[dict[str, Any]] = []
    transition_frames: list[pd.DataFrame] = []
    for policy, masks in policy_masks.items():
        daily = daily_frame.loc[daily_frame["policy"] == policy].set_index("trade_date")
        daily = daily.reindex(dates)
        if daily[["gross", "net", "slippage", "financing", "borrow", "reverse", "overnight"]].isna().any().any():
            raise ValueError(f"Saved daily replay does not fully cover {policy}")
        replay = {"daily": daily}
        segment_rows.extend(
            {"policy": policy, **row}
            for row in _segment_rows(
                replay=replay,
                weights=weights,
                alpha_masks=masks,
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
        transitions = attribute_inventory_transitions(
            weights=weights,
            target_returns=target_returns,
            gap_returns=gap_returns,
            alpha_masks=masks,
            sim_dates=dates,
            calendar_days=calendar_days,
            slip=slip,
            financing_daily=financing_annual / 365.0,
            borrow_daily=borrow_annual / 365.0,
            reverse_daily=reverse_fee_bps / 10_000.0,
            side_leverage=side_leverage,
        )
        transitions = transitions.loc[transitions["date"] >= pd.Timestamp(metrics["oos_start"])].copy()
        ledger_by_date = transitions.groupby("date").sum(numeric_only=True).reindex(daily.index)
        reconciliation = {
            "gross_pnl": "gross",
            "net_pnl": "net",
            "total_cost": "cost",
            "slippage_cost": "slippage",
            "financing_cost": "financing",
            "borrow_cost": "borrow",
            "reverse_cost": "reverse",
            "overnight_pnl": "overnight",
        }
        evaluation_dates = dates[oos_start_index:]
        for ledger_column, daily_column in reconciliation.items():
            if not np.allclose(
                ledger_by_date.loc[evaluation_dates, ledger_column].to_numpy(dtype=float),
                daily.loc[evaluation_dates, daily_column].to_numpy(dtype=float),
                atol=1e-10,
                rtol=0.0,
            ):
                raise ValueError(f"Ticker/day {ledger_column} does not reconcile to daily {daily_column}")
        transitions["ticker"] = transitions["ticker_index"].map(
            {index: ticker for index, ticker in enumerate(JP_TICKERS)}
        )
        transitions["policy"] = policy
        transition_frames.append(transitions)

    segment_frame = pd.DataFrame(segment_rows)
    segment_frame.to_csv(
        OUTPUT_DIR / "calendar_and_reversal_segments.csv", index=False, float_format="%.12g"
    )
    pd.concat(transition_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "ticker_transition_attribution.csv", index=False, float_format="%.12g"
    )
    metrics["calendar_and_reversal_segments"] = segment_frame.to_dict(orient="records")
    metrics["overlay_status_counts"] = {"skipped": adr_skip_count}
    metrics["overlay_skip_reason_counts"] = {"adr_not_supplied": adr_skip_count}
    metrics["overlay_applied_rows"] = 0
    metrics["macro_feature_status"] = {
        "run_owned_price_load_failures": log_text.count("Failed to load run-owned macro prices:"),
        "dates_with_insufficient_rows": macro_empty_count,
    }
    for segment in metrics["overlay_schedule"]:
        rows = int(segment["n_weight_rows"])
        segment["overlay_status_counts"] = {"skipped": rows}
        segment["overlay_skip_reason_counts"] = {"adr_not_supplied": rows}
    metrics["trial_count_estimate"] = 9
    metrics["trial_count_basis"] = {
        "previous_fixed_alpha_variants_found_in_existing_result_files": 3,
        "current_window_variants_including_baseline": 6,
        "repeated_baseline_counted_twice_conservatively": True,
        "historical_same_window_sharpe_series_available": False,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "日次の同じ本番ウェイトを両方式に使い、carry配分だけを比較した。",
        "日次の同じV2ウェイトを両方式に使い、carry配分だけを比較した。ML overlayは入力不足で全行skipとなり、生成weightはoverlay適用前のV2値。",
    )
    report_lines = report.splitlines()
    refreshed_lines: list[str] = []
    for line in report_lines:
        if line.startswith("- Overlay適用状態:") or line.startswith("- Macro入力:"):
            continue
        refreshed_lines.append(line)
        if line.startswith("- Overlay:"):
            refreshed_lines.append(
                f"- Overlay適用状態: skipped {adr_skip_count}/{expected_rows}、理由 `adr_not_supplied`。本比較の全weightはML overlay適用前。"
            )
            refreshed_lines.append(
                f"- Macro入力: run-owned価格の読込に失敗し、{macro_empty_count}/{expected_rows}日でmacro data insufficient (0 rows)。macro調整は適用されていない。"
            )
    report = "\n".join(refreshed_lines) + "\n"
    report = report.replace(
        "区分は次営業日の実現date gap・実現weightを使った事後診断で、carry decisionには使っていない。日次PnLはoutgoing trade dateへ帰属し、overnight markを含む。\n\n",
        "",
    )
    heading = "### 連休・翌朝反転の事後区分"
    heading_index = report.index(heading)
    table_start = report.index("| Policy | Segment |", heading_index)
    table_end = report.index("\n\n### 比較の不確実性と判定", table_start)
    intro = (
        "区分は次営業日の実現date gap・実現weightを使った事後診断で、carry decisionには使っていない。"
        "calendar区分はportfolio session単位、same/reversal/flat区分は銘柄×session単位の加算可能な台帳。"
        "日次PnLはoutgoing trade dateへ帰属し、overnight markを含む。`Next-open slip delta vs flat` は次営業日寄りの実slippageを、carry在庫なしで同じtargetを売買した反実仮想と比べた増減費用（負値は節約）。\n\n"
    )
    table_end_heading = report.index("\n\n### 比較の不確実性と判定", table_end)
    report = (
        report[: heading_index + len(heading)]
        + "\n\n"
        + intro
        + _format_segment_table(segment_rows)
        + report[table_end_heading:]
    )
    report = report.replace(
        "`calendar_and_reversal_segments.csv`, `ticker_carry_diagnostics.csv`",
        "`calendar_and_reversal_segments.csv`, `ticker_transition_attribution.csv`, `ticker_carry_diagnostics.csv`",
    )
    report = report.replace(
        "再現: `timeout -k 10s 3600s .venv/bin/python src/research/scripts/experiments/experiment_adaptive_overnight_inventory_20261008.py`\n\n",
        "再現コマンド:\n\n```sh\ntimeout -k 10s 3600s .venv/bin/python src/research/scripts/experiments/experiment_adaptive_overnight_inventory_20261008.py 2>&1 | tee reports/20261008_adaptive_overnight_inventory/run.log\ntimeout -k 10s 180s .venv/bin/python src/research/scripts/experiments/refresh_adaptive_overnight_inventory_report_20261008.py\n```\n\n",
    )
    report_path.write_text(report, encoding="utf-8")
    print(
        f"Updated ticker-transition attribution: {len(segment_rows)} groups, "
        f"ADR skips={adr_skip_count}, macro-empty={macro_empty_count}; report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
