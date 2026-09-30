#!/usr/bin/env python3
"""Long-period paired V2 comparison for the 2026-09-24 sensitivity audit.

Historical results are retrospective diagnostics. 09:10 returns use the local
5-minute proxy where present and the existing open-to-close fallback elsewhere.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research/scripts/experiments"))

import experiment_sensitivity_pipeline_audit_20260924 as audit  # noqa: E402
import experiment_sensitivity_rate_prior_v2_20260924 as v2_helpers  # noqa: E402

from leadlag.config.frozen import safe_config_copy  # noqa: E402
from leadlag.data.intraday_inputs import compute_jp_target_returns  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, N_JP, US_TICKERS  # noqa: E402
from leadlag.domain.inputs import HistoricalInputs  # noqa: E402
from leadlag.execution.backtester import BacktestEngine  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import (  # noqa: E402
    Decision,
    ExperimentRecord,
    ExperimentRegistry,
    compute_deflated_sharpe,
)
from leadlag.models.v2.distribution_source import OnDemandDistributionSource  # noqa: E402
from leadlag.runner.model_factory import (  # noqa: E402
    model_config_fingerprint,
    resolve_overlay_settings,
)
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

EXPERIMENT_ID = "20260924_sensitivity_pipeline_audit_long"
START_DATE = "2015-01-05"
END_DATE = "2026-09-18"
REPORT_DIR = ROOT / "reports/20260924_sensitivity_pipeline_audit_long"
RESULTS_DIR = ROOT / "var/results/20260924_sensitivity_pipeline_audit_long"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
TRADING_DAYS = 245
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260924
LOG = logging.getLogger(EXPERIMENT_ID)


def _load_long_candidates() -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, Any]],
]:
    score_cfg = audit._read_yaml(audit.CONFIG_PATH)
    prior_cfg = audit._read_yaml(audit.PRIOR_CONFIG_PATH)
    rate_cfg = audit._read_yaml(audit.RATE_CONFIG_PATH)
    caps = pd.read_csv(audit.CAP_PATH)
    baseline = audit._current_copy()
    variants: dict[str, dict[str, dict[str, float]]] = {"current": copy.deepcopy(baseline)}
    variants["all_w3_w6_zero"] = audit._zero_channel(baseline, None)
    for factor in ("w3", "w4", "w5", "w6"):
        variants[f"drop_{factor}"] = audit._zero_channel(baseline, factor)

    dynamic: dict[str, dict[str, dict[str, float]]] = {}
    cap_years = sorted(int(year) for year in caps["snapshot_year"].unique())
    expected_years = set(range(2014, 2026))
    if set(cap_years) != expected_years:
        raise ValueError(f"Expected complete JPX cap snapshots 2014–2025; got {cap_years}")

    for source_name, source_cfg in (("user", score_cfg), ("legacy", prior_cfg)):
        annual: dict[str, dict[str, dict[str, float]]] = {}
        for trade_year in range(2015, 2027):
            cap_year = trade_year - 1
            if cap_year not in expected_years:
                continue
            labels = audit._labels_from_industry_table(source_cfg, prior_cfg, caps, cap_year)
            annual[str(trade_year)] = labels
        name = f"{source_name}_jpx33_pitcap_annual"
        dynamic[name] = annual
        variants[name] = copy.deepcopy(baseline)

    rate_cfg_map = {str(k): float(v) for k, v in rate_cfg["w6_candidate"].items()}
    all_tickers = (*US_TICKERS, *JP_TICKERS)
    if set(rate_cfg_map) != set(all_tickers):
        raise ValueError("Rate-prior candidate does not cover exactly the 32 V2 tickers")
    for name, update_us, update_jp in (
        ("rate_w6_us_only", True, False),
        ("rate_w6_jp_only", False, True),
        ("rate_w6_both", True, True),
    ):
        labels = copy.deepcopy(baseline)
        for ticker in all_tickers:
            if (ticker in US_TICKERS and update_us) or (ticker in JP_TICKERS and update_jp):
                labels[ticker]["w6"] = rate_cfg_map[ticker]
        variants[name] = labels

    return baseline, variants, dynamic


def _bind_pit_schedule(
    eval_dates: pd.DatetimeIndex,
    dynamic: dict[str, dict[str, dict[str, dict[str, float]]]],
    baseline: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], dict[str, Any]]:
    schedules: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    first_sessions: dict[int, pd.Timestamp] = {}
    for date in eval_dates:
        first_sessions.setdefault(int(date.year), pd.Timestamp(date))
    first_dates = set(first_sessions.values())
    for variant_name, annual_labels in dynamic.items():
        schedule: dict[str, dict[str, dict[str, float]]] = {}
        for date in eval_dates:
            date = pd.Timestamp(date)
            if date in first_dates:
                # JPX monthly stats are released after the first business-day
                # 09:10 decision; keep the previous baseline for that session.
                schedule[date.strftime("%Y-%m-%d")] = copy.deepcopy(baseline)
                continue
            labels = annual_labels.get(str(date.year))
            if labels is None:
                raise ValueError(f"No PIT annual labels for {variant_name} on {date.date()}")
            schedule[date.strftime("%Y-%m-%d")] = copy.deepcopy(baseline)
            for ticker in JP_TICKERS:
                schedule[date.strftime("%Y-%m-%d")][ticker].update(labels[ticker])
        schedules[variant_name] = schedule
    metadata = {
        "first_japanese_session_per_year_uses_current_labels": {
            str(year): date.strftime("%Y-%m-%d") for year, date in sorted(first_sessions.items())
        },
        "annual_cap_rule": "previous-calendar-year JPX year-end industry total market cap; candidate labels start on the second evaluation session each year",
        "cap_is_float_adjusted_etf_weight": False,
    }
    return schedules, metadata


def _validate_shared_historical(historical: HistoricalInputs, frame: pd.DataFrame) -> HistoricalInputs:
    """Keep the shared historical snapshot intact for the normal V2 path."""
    # The model/backtest uses the same local 09:10 panel and per-cell fallback
    # as the short audit. This helper only validates that the supplied frame is
    # exactly the immutable HistoricalInputs snapshot used by every variant.
    if dataframe_fingerprint(historical.to_frame()) != dataframe_fingerprint(frame):
        raise ValueError("HistoricalInputs frame differs from the shared evaluation frame")
    return historical


def _run_preflight(
    app_config: Any,
    frame: pd.DataFrame,
    eval_date: pd.Timestamp,
    historical: HistoricalInputs,
    input_manifest: dict[str, Any],
    transform_cache: dict[str, Any],
) -> dict[str, Any]:
    old_labels = audit._install_labels(audit._current_copy())
    counters, originals = v2_helpers._install_measurement_hooks(transform_cache, "long_preflight_current")
    try:
        result = BacktestEngine.run_v2_backtest(
            cfg=app_config,
            gap_input_dir=input_manifest["gap_store_path"],
            df_exec=frame,
            start_date=eval_date.strftime("%Y-%m-%d"),
            end_date=eval_date.strftime("%Y-%m-%d"),
            n_jobs=1,
            historical_inputs=historical,
        )
    finally:
        v2_helpers._restore_measurement_hooks(originals)
        audit._install_labels(old_labels)
    if counters.get("decide_exception") or counters["decisions"] != 1:
        raise RuntimeError(f"Long-period preflight did not produce one valid decision: {counters}")
    if bool(result["daily_fallback"].iloc[0]):
        raise RuntimeError("Long-period baseline preflight used the flat fallback")
    return {
        "date": eval_date.strftime("%Y-%m-%d"),
        "decisions": int(counters["decisions"]),
        "leakage_status_counts": dict(counters["leakage_status"]),
        "numerical_status_counts": dict(counters["numerical_status"]),
        "on_demand_status_counts": {
            f"h{h}:{status}": int(count)
            for (h, status), count in counters["on_demand_by_horizon_status"].items()
        },
        "fallback": bool(result["daily_fallback"].iloc[0]),
        "net_return": float(result["daily_returns"].iloc[0]),
        "transform_equivalence_checks": int(transform_cache["equivalence_checks"]),
    }


def _moving_block_ci(base: np.ndarray, candidate: np.ndarray, seed: int) -> dict[str, list[float]]:
    return v2_helpers._moving_block_ci(
        base,
        candidate,
        block_length=BOOTSTRAP_BLOCK,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=seed,
    )


def _write_report(summary: dict[str, Any], table: pd.DataFrame) -> None:
    baseline = summary["variants"]["current"]
    rows = [
        "# 感応度監査の長期V2比較",
        "",
        "## 判定",
        "",
        "**現行感応度は暫定ベースラインとして維持する。** 過去期間はすでに閲覧済みであり、この比較は未使用OOSではない。結果を見て感応度を採択する用途には使わない。",
        "",
        "## 対象と制約",
        "",
        f"- 期間: {summary['evaluation']['start_date']}〜{summary['evaluation']['end_date']}、{summary['evaluation']['simulation_dates']}営業日。評価系列は全変種で日付を一致させた。",
        f"- 09:10 proxyは任意銘柄で{summary['evaluation']['open_910_dates_any']}日、全17ETF同時に揃うのは{summary['evaluation']['open_910_dates_all17']}日。価格proxyがある銘柄・日は09:10 proxyを使い、欠けたセルは既存経路の始値〜大引けtarget fallbackを使う。従って本結果を実約定可能な9:10戦略の成績とは読まない。",
        "- 全候補で同じV2・BLPX・リスク・RuleD・コスト設定を使用。μ/Ω cacheは使わず毎日on-demandで再計算した。過去用の学習済みML overlayがないため、短期監査と同様に比較全体でML overlayを無効化。",
        "- 過去データを使った回顧診断で、正式な採否は仕様凍結後のforward shadowで行う。バックテストcostは推定値であり、約定実績のコストではない。",
        "- 33業種案は前年JPX年末の業種総時価総額を親TOPIX-17内で集約。毎年の第1評価セッションでは9:10時点で当年初公表データを使えないため現行ラベルを維持し、第2セッションから前年末capを適用した。float-adjusted ETF holdingsではない。",
        "- 比較対象は現行、w3〜w6全ゼロ、各チャネル除外、33業種提案/従来案のPIT集約、w6 US-only/JP-only/both。ランダム置換と探索で最大差だった単セル案は、追加探索を避けるため長期損益の比較対象から外し、前回監査の構造診断に留めた。",
        "",
        "## コスト後成績",
        "",
        "| 変種 | Net Sharpe | Δ Sharpe | Gross Sharpe | Max DD | Net日次差 CI 95% (bp/日) | Sharpe差 CI 95% | turnover変化 | fallback率 | Δμ平均絶対値 (bp) | 平均最終weight L1差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        rows.append(
            f"| {row['variant']} | {row['net_sharpe']:.4f} | {row['net_sharpe_delta']:+.4f} | "
            f"{row['gross_sharpe']:.4f} | {row['max_drawdown']:.2%} | "
            f"[{row['net_daily_delta_ci95_bps'][0]:+.3f}, {row['net_daily_delta_ci95_bps'][1]:+.3f}] | "
            f"[{row['sharpe_delta_ci95'][0]:+.4f}, {row['sharpe_delta_ci95'][1]:+.4f}] | "
            f"{row['turnover_change']:+.2%} | {row['fallback_rate']:.2%} | "
            f"{row['mean_abs_mu_delta_bps']:.4f} | {row['mean_l1_weight_delta']:.5f} |"
        )
    rows.extend([
        "",
        "指標は年245営業日換算。CIは日付を揃えた20営業日non-circular moving-block bootstrap 5,000回。差CIが0をまたぐ場合は、差を識別できたとは扱わない。最大DD・turnover・cost内訳も全営業日を含む。",
        "",
        "## 結果の読み方",
        "",
        f"- 現行Net Sharpeは{baseline['net_sharpe']:.4f}、Gross Sharpeは{baseline['gross_sharpe']:.4f}、平均日次costは{baseline['mean_daily_cost_bps']:.3f}bp。コスト後の符号だけから『コスト負け』とは判定できず、Gross/Netとコストを合わせて見る。",
        "- 感応度候補の成績差は、事前登録したforward基準（Net Sharpe +0.05以上、paired日次PnL差CI下限>0、DD・turnover・fallback・exposure制約）を満たしても、既知の過去期間だけで本番採用しない。",
        "- 収益規模が大きく見える場合は、長期複利とside leverageによる影響が含まれる。単年別成績と日次分布を併せて参照する。",
        "",
        "### 年次成績と費用",
        "",
        "`annual_metrics.csv` に年ごとのNet return、Sharpe、DD、turnover、fallbackを保存。各変種の日次コスト内訳とV2監査カウンタは `daily_*.csv` と `summary.json` に保存した。",
        "",
        "## 再現情報",
        "",
        f"- Script: `{Path(__file__).relative_to(ROOT)}`。df_exec SHA-256: `{summary['df_exec_fingerprint']}`。実効V2 config fingerprint: `{summary['effective_config_fingerprint']}`。",
        f"- Registry開始件数 {summary['registry_records_before']}、関連する既知感応度試行 {summary['relevant_prior_trials']}、今回の候補 {summary['candidate_trial_count']}、DSR診断の総試行数 {summary['dsr_trial_count']}。",
        f"- 実行秒数: `{json.dumps(summary['elapsed_seconds'], ensure_ascii=False)}`。全成果物は `{RESULTS_DIR.relative_to(ROOT)}/` に保存。",
        "- production設定、ticker labels、ライブcache/storeは変更していない。",
        "",
    ])
    (REPORT_DIR / "report.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)

    baseline_labels, variants, dynamic_labels = _load_long_candidates()
    production_config = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    production_overlay_enabled, production_overlay_path = resolve_overlay_settings(production_config)
    app_config = safe_config_copy(production_config)
    if production_overlay_enabled:
        app_config = app_config.model_copy(update={
            "v2": app_config.v2.model_copy(update={"ml_overlay_enabled": False}),
            "ml_order_overlay": app_config.ml_order_overlay.model_copy(update={"enabled": False}),
        })

    frame, eval_dates, historical_raw, input_manifest = v2_helpers._load_owned_inputs(
        app_config, START_DATE, END_DATE
    )
    historical = _validate_shared_historical(historical_raw, frame)
    dynamic_schedule, pit_metadata = _bind_pit_schedule(eval_dates, dynamic_labels, baseline_labels)
    input_manifest["production_effective_config_fingerprint"] = model_config_fingerprint(production_config)
    input_manifest["production_overlay_enabled"] = bool(production_overlay_enabled)
    input_manifest["production_overlay_path"] = None if production_overlay_path is None else str(production_overlay_path)
    input_manifest["research_overlay_enabled"] = False
    input_manifest["research_overlay_disabled_reason"] = (
        "Historical fold-specific ML overlay artifacts are unavailable; disabled for all paired variants."
        if production_overlay_enabled else None
    )
    transforms = v2_helpers._build_transform_cache(frame, app_config, eval_dates)
    input_manifest["causal_transform_equivalence_checks"] = int(transforms["equivalence_checks"])
    input_manifest["causal_transform_horizons"] = list(transforms["horizons"])
    input_manifest["fractional_difference_parameters"] = transforms["frac_parameters"]
    if len(eval_dates) < 2000:
        raise ValueError(f"Expected a multiyear comparison, received only {len(eval_dates)} sessions")

    preflight = _run_preflight(
        app_config, frame, eval_dates[0], historical, input_manifest, transforms
    )
    (RESULTS_DIR / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2, default=v2_helpers._json_default),
        encoding="utf-8",
    )
    LOG.warning(
        "Long sensitivity comparison starts: %d dates (%s to %s), 09:10 all-17 dates=%d, variants=%d",
        len(eval_dates), eval_dates.min().date(), eval_dates.max().date(),
        input_manifest["open_910_dates_complete_for_17_etfs"], len(variants),
    )

    daily_by_variant: dict[str, pd.DataFrame] = {}
    mu_by_variant: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    weights_by_variant: dict[str, np.ndarray] = {}
    metric_by_variant: dict[str, dict[str, Any]] = {}
    label_by_variant: dict[str, dict[str, dict[str, float]]] = {}
    elapsed: dict[str, float] = {}
    for variant_name, static_labels in variants.items():
        LOG.warning("Run started: %s", variant_name)
        previous_labels = audit._install_labels(static_labels)
        counters, originals = v2_helpers._install_measurement_hooks(transforms, variant_name)
        captured_mu: dict[tuple[str, int], np.ndarray] = {}
        hooked_resolve = OnDemandDistributionSource.resolve

        def capture_resolve(
            self: Any,
            trade_date: str,
            df_exec: Any,
            current_prices: Any,
            *,
            horizon: int = 1,
            **kwargs: Any,
        ) -> Any:
            date_key = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
            if variant_name in dynamic_schedule:
                dated_labels = dynamic_schedule[variant_name].get(date_key)
                if dated_labels is None:
                    raise ValueError(f"Missing point-in-time label schedule for {variant_name} at {date_key}")
                audit._install_labels(dated_labels)
            result = hooked_resolve(
                self, trade_date, df_exec, current_prices, horizon=horizon, **kwargs
            )
            if result.mu_gap is not None:
                captured_mu[(date_key, int(horizon))] = np.asarray(result.mu_gap, dtype=float).copy()
            return result

        OnDemandDistributionSource.resolve = capture_resolve  # type: ignore[method-assign]
        run_started = time.monotonic()
        try:
            result = BacktestEngine.run_v2_backtest(
                cfg=app_config,
                gap_input_dir=input_manifest["gap_store_path"],
                df_exec=frame,
                start_date=START_DATE,
                end_date=END_DATE,
                n_jobs=1,
                historical_inputs=historical,
            )
        finally:
            elapsed[variant_name] = time.monotonic() - run_started
            v2_helpers._restore_measurement_hooks(originals)
            audit._install_labels(previous_labels)
        if len(result["daily_returns"]) != len(eval_dates):
            raise ValueError(
                f"{variant_name} returned {len(result['daily_returns'])} days, expected {len(eval_dates)}"
            )
        metric, daily = v2_helpers._metrics(result)
        metric["decisions_counted"] = int(counters["decisions"])
        metric["leakage_status_counts"] = dict(counters["leakage_status"])
        metric["numerical_status_counts"] = dict(counters["numerical_status"])
        metric["on_demand_by_horizon_status"] = {
            f"h{h}:{status}": int(count)
            for (h, status), count in counters["on_demand_by_horizon_status"].items()
        }
        metric["elapsed_seconds"] = elapsed[variant_name]
        if counters["decisions"] != len(eval_dates):
            raise ValueError(f"Decision count mismatch in {variant_name}: {counters['decisions']}")
        daily_by_variant[variant_name] = daily
        mu_by_variant[variant_name] = captured_mu
        weights_by_variant[variant_name] = np.asarray(result["weights"], dtype=float)
        metric_by_variant[variant_name] = metric
        label_by_variant[variant_name] = static_labels
        pd.DataFrame(weights_by_variant[variant_name], index=result["daily_returns"].index, columns=JP_TICKERS).to_csv(
            RESULTS_DIR / f"weights_{variant_name}.csv", index_label="trade_date"
        )
        daily.to_csv(RESULTS_DIR / f"daily_{variant_name}.csv", index_label="trade_date")
        pd.DataFrame([
            {"trade_date": date, "horizon": horizon, **{
                f"mu_{ticker}": float(value) for ticker, value in zip(JP_TICKERS, mu, strict=True)
            }}
            for (date, horizon), mu in sorted(captured_mu.items()) if mu.shape == (N_JP,)
        ]).to_csv(RESULTS_DIR / f"mu_{variant_name}.csv", index=False)
        LOG.warning(
            "Run complete: %s elapsed=%.1fs net_sharpe=%.4f decisions=%d fallbacks=%d",
            variant_name, elapsed[variant_name], metric["net_sharpe"], counters["decisions"],
            metric["fallback_days"],
        )

    baseline_daily = daily_by_variant["current"]
    baseline_net = baseline_daily["net_return"].to_numpy(dtype=float)
    baseline_mu = mu_by_variant["current"]
    open_to_910 = historical.open_910_returns.loc[eval_dates].copy()
    open_to_910.columns = list(JP_TICKERS)
    target_values = compute_jp_target_returns(
        frame, JP_TICKERS, open_910_returns=historical.open_910_returns
    )
    target_panel = pd.DataFrame(target_values, index=frame.index, columns=JP_TICKERS).loc[eval_dates].copy()
    prior_records = list(ExperimentRegistry(REGISTRY_PATH))
    relevant_prior = [record for record in prior_records if any(tag in record.name.lower() for tag in ("sensitiv", "jpx33"))]
    candidate_names = [name for name in variants if name != "current"]
    trial_count = len(relevant_prior) + len(candidate_names)
    finite_sharpes = [metric_by_variant[name]["net_sharpe"] for name in candidate_names]
    summary_variants: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []

    for name, metric in metric_by_variant.items():
        daily = daily_by_variant[name]
        if name == "current":
            metric["mu_effect_vs_current"] = {"matched_mu_cells": len(baseline_mu), "mean_abs_delta_bps": 0.0, "corr_with_current_mu": 1.0}
            baseline_h1 = {date: mu for (date, horizon), mu in baseline_mu.items() if horizon == 1}
            metric["rank_ic"] = audit._rank_metrics(
                baseline_h1, target_panel
            )
            metric["rank_ic_open_to_0910_diagnostic"] = audit._rank_metrics(
                baseline_h1, open_to_910
            )
            metric["paired_net_return_delta_ci95"] = [0.0, 0.0]
            metric["paired_net_sharpe_delta_ci95"] = [0.0, 0.0]
            metric["mean_daily_net_delta_bps"] = 0.0
            metric["mean_l1_weight_delta"] = 0.0
            metric["turnover_change_vs_current_fraction"] = 0.0
            metric["drawdown_worsening_pp_vs_current"] = 0.0
            metric["dsr"] = None
            metric["dsr_trial_count"] = trial_count
            summary_variants[name] = metric
            table_rows.append({
                "variant": name,
                "net_sharpe": metric["net_sharpe"],
                "net_sharpe_delta": 0.0,
                "gross_sharpe": metric["gross_sharpe"],
                "max_drawdown": metric["max_drawdown"],
                "net_daily_delta_ci95_bps": [0.0, 0.0],
                "sharpe_delta_ci95": [0.0, 0.0],
                "turnover_change": 0.0,
                "fallback_rate": metric["fallback_rate"],
                "mean_abs_mu_delta_bps": 0.0,
                "mean_l1_weight_delta": 0.0,
            })
            continue

        candidate_net = daily["net_return"].to_numpy(dtype=float)
        ci = _moving_block_ci(baseline_net, candidate_net, BOOTSTRAP_SEED + len(summary_variants))
        mu_effect = audit._mu_effect(baseline_mu, mu_by_variant[name])
        l1_delta = np.abs(weights_by_variant[name] - weights_by_variant["current"]).sum(axis=1)
        turn_change = metric["total_turnover_one_way"] / max(metric_by_variant["current"]["total_turnover_one_way"], 1e-12) - 1.0
        metric["mu_effect_vs_current"] = mu_effect
        candidate_h1 = {date: mu for (date, horizon), mu in mu_by_variant[name].items() if horizon == 1}
        metric["rank_ic"] = audit._rank_metrics(candidate_h1, target_panel)
        metric["rank_ic_open_to_0910_diagnostic"] = audit._rank_metrics(candidate_h1, open_to_910)
        metric["paired_net_return_delta_ci95"] = ci["mean_daily_net_return_delta"]
        metric["paired_net_sharpe_delta_ci95"] = ci["annualized_net_sharpe_delta"]
        metric["mean_daily_net_delta_bps"] = float((candidate_net - baseline_net).mean() * 10000.0)
        metric["mean_l1_weight_delta"] = float(l1_delta.mean())
        metric["dates_with_changed_final_weights"] = int((l1_delta > 1e-10).sum())
        metric["turnover_change_vs_current_fraction"] = float(turn_change)
        metric["drawdown_worsening_pp_vs_current"] = float(
            (metric_by_variant["current"]["max_drawdown"] - metric["max_drawdown"]) * 100.0
        )
        metric["model_constraints_pass"] = bool(
            metric["max_model_gross"] <= 2.0 + 1e-8 and metric["max_abs_model_net"] <= 0.05 + 1e-8
        )
        metric["dsr"] = compute_deflated_sharpe({
            "net_sharpe": metric["net_sharpe"],
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": TRADING_DAYS,
            "trials": max(trial_count, 1),
            "n_observations": metric["n_observations"],
            "returns": candidate_net.tolist(),
            "trial_sharpes": finite_sharpes,
        })
        metric["dsr_trial_count"] = trial_count
        summary_variants[name] = metric
        table_rows.append({
            "variant": name,
            "net_sharpe": float(metric["net_sharpe"]),
            "net_sharpe_delta": float(metric["net_sharpe"] - metric_by_variant["current"]["net_sharpe"]),
            "gross_sharpe": float(metric["gross_sharpe"]),
            "max_drawdown": float(metric["max_drawdown"]),
            "net_daily_delta_ci95_bps": [float(x) * 10000.0 for x in ci["mean_daily_net_return_delta"]],
            "sharpe_delta_ci95": ci["annualized_net_sharpe_delta"],
            "turnover_change": float(turn_change),
            "fallback_rate": float(metric["fallback_rate"]),
            "mean_abs_mu_delta_bps": float(mu_effect["mean_abs_delta_bps"] or 0.0),
            "mean_l1_weight_delta": float(l1_delta.mean()),
        })

    table = pd.DataFrame(table_rows)
    table.to_csv(RESULTS_DIR / "variant_comparison.csv", index=False)
    annual_rows = []
    for variant_name, daily in daily_by_variant.items():
        for year, group in daily.groupby("year", sort=True):
            values = group["net_return"].to_numpy(dtype=float)
            annual_rows.append({
                "variant": variant_name,
                "year": int(year),
                "dates": int(len(group)),
                "net_compounded_return": float(np.prod(1.0 + values) - 1.0),
                "net_sharpe": v2_helpers._safe_sharpe(values),
                "max_drawdown": float(group["net_return"].add(1.0).cumprod().div(group["net_return"].add(1.0).cumprod().cummax()).sub(1.0).min()),
                "mean_daily_turnover_one_way": float(group["turnover_one_way"].mean()),
                "fallback_rate": float(group["fallback"].mean()),
            })
    annual_table = pd.DataFrame(annual_rows)
    annual_table.to_csv(RESULTS_DIR / "annual_metrics.csv", index=False)

    cap_source_manifest = json.loads(audit.CAP_SOURCE_PATH.read_text(encoding="utf-8"))
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "evaluation": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "simulation_dates": int(len(eval_dates)),
            "open_910_dates_any": int(input_manifest["open_910_dates_with_any_finite_price"]),
            "open_910_dates_all17": int(input_manifest["open_910_dates_complete_for_17_etfs"]),
            "target": "09:10-to-close using local 5-minute proxy per observed cell; otherwise existing open-to-close fallback",
            "actual_executable_quotes": False,
            "prior_period_seen": True,
            "annualization_trading_days": TRADING_DAYS,
            "bootstrap": {"block_length": BOOTSTRAP_BLOCK, "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        },
        "preflight": preflight,
        "input_manifest": input_manifest,
        "pit_jpx33_metadata": pit_metadata,
        "variants": summary_variants,
        "df_exec_fingerprint": dataframe_fingerprint(frame),
        "effective_config_fingerprint": model_config_fingerprint(app_config),
        "research_config_sha256": hashlib.sha256(
            audit.CONFIG_PATH.read_bytes() + audit.PRIOR_CONFIG_PATH.read_bytes()
            + audit.CAP_PATH.read_bytes() + audit.RATE_CONFIG_PATH.read_bytes()
        ).hexdigest(),
        "cap_source_manifest": cap_source_manifest,
        "registry_records_before": len(prior_records),
        "relevant_prior_trials": len(relevant_prior),
        "candidate_trial_count": len(candidate_names),
        "dsr_trial_count": trial_count,
        "elapsed_seconds": elapsed,
        "production_modified": False,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=v2_helpers._json_default),
        encoding="utf-8",
    )
    (RESULTS_DIR / "variant_labels_static.csv").write_text(
        pd.DataFrame([
            {"variant": name, "ticker": ticker, **factors}
            for name, labels in label_by_variant.items()
            if name not in dynamic_schedule
            for ticker, factors in labels.items()
        ]).to_csv(index=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / "variant_labels_by_trade_year.csv").write_text(
        pd.DataFrame([
            {"variant": name, "trade_year": int(year), "ticker": ticker, **factors}
            for name, years in dynamic_labels.items()
            for year, labels in years.items()
            for ticker, factors in labels.items()
        ]).to_csv(index=False),
        encoding="utf-8",
    )
    _write_report(summary, table)

    registry = ExperimentRegistry(REGISTRY_PATH)
    finished_at = datetime.now(UTC)
    related = [record.name for record in relevant_prior]
    for name in candidate_names:
        registry.record(ExperimentRecord(
            name=f"{EXPERIMENT_ID}_{name}",
            hypothesis="The long-period sensitivity audit checks whether recent w3-w6 ablations, annual PIT JPX33 aggregation, or separated US/JP w6 proposals change V2 cost-adjusted returns versus the current prior.",
            start_time=started_at,
            end_time=finished_at,
            parameters={
                "variant": name,
                "candidate_labels_static": label_by_variant[name] if name not in dynamic_schedule else None,
                "candidate_labels_by_trade_year": dynamic_labels.get(name),
                "label_schedule_artifact": (
                    str((RESULTS_DIR / "variant_labels_by_trade_year.csv").relative_to(ROOT))
                    if name in dynamic_schedule else None
                ),
                "target_window": [START_DATE, END_DATE],
                "target": summary["evaluation"]["target"],
                "cache_policy": "bypass mu/Omega cache; on-demand BLPX for each date",
                "annual_pit_schedule": pit_metadata if name in dynamic_schedule else None,
                "research_effective_config_fingerprint": model_config_fingerprint(app_config),
                "production_effective_config_fingerprint": model_config_fingerprint(production_config),
                "df_exec_fingerprint": summary["df_exec_fingerprint"],
                "related_prior_trial_count": len(relevant_prior),
                "all_prior_trial_ids": related,
                "adoption": "retrospective diagnostics only; not untouched OOS or executable quotes",
            },
            metrics=summary_variants[name],
            decision=Decision.PENDING,
            report_path=str((REPORT_DIR / "report.md").relative_to(ROOT)),
            related_records=related,
        ))
    print(f"report={REPORT_DIR / 'report.md'}", flush=True)
    print(f"results={RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
