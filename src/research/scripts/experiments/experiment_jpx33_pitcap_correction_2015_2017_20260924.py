#!/usr/bin/env python3
"""Measure the 2015-2017 forecast effect of replacing the future 2017 cap map."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research/scripts/experiments"))

import experiment_sensitivity_pipeline_audit_20260924 as audit  # noqa: E402
from leadlag.core.gap_adjustment import build_raw_distribution  # noqa: E402
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, N_JP, N_US, SENSITIVITY_LABELS  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry  # noqa: E402
from leadlag.models.blpx.model import ProductionBLPXModel  # noqa: E402

EXPERIMENT_ID = "20260924_jpx33_pitcap_correction_2015_2017"
REPORT_DIR = ROOT / "reports/20260924_jpx33_pitcap_correction_2015_2017"
RESULTS_DIR = ROOT / "var/results/20260924_jpx33_pitcap_correction_2015_2017"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260924


def _block_ci(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < BOOTSTRAP_BLOCK:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_blocks = int(np.ceil(len(values) / BOOTSTRAP_BLOCK))
    max_start = len(values) - BOOTSTRAP_BLOCK
    estimates = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        block_sample = np.concatenate([values[start : start + BOOTSTRAP_BLOCK] for start in starts])
        estimates[i] = float(np.mean(block_sample[: len(values)]))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _daily_metrics(prediction: np.ndarray, target: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for i, date in enumerate(dates):
        mask = np.isfinite(prediction[i]) & np.isfinite(target[i])
        if mask.sum() < 10:
            continue
        x = prediction[i, mask]
        y = target[i, mask]
        ic = 0.0 if np.ptp(x) <= 1e-14 or np.ptp(y) <= 1e-14 else float(spearmanr(x, y).statistic)
        error = x - y
        rows.append({
            "trade_date": date,
            "year": int(date.year),
            "observed_etfs": int(mask.sum()),
            "rank_ic": ic,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
        })
    return pd.DataFrame(rows)


def _metric_summary(daily: pd.DataFrame) -> dict[str, Any]:
    return {
        "paired_dates": int(len(daily)),
        "mean_observed_etfs": float(daily["observed_etfs"].mean()),
        "mean_rank_ic": float(daily["rank_ic"].mean()),
        "mean_mae_bps": float(daily["mae"].mean() * 10000.0),
        "mean_rmse_bps": float(daily["rmse"].mean() * 10000.0),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    frame = load_df_exec_from_local_cache(max_stale_bdays=None)
    frame = frame.loc[(frame.index >= "2010-01-01") & (frame.index <= "2017-12-29")].copy()
    baseline_rows = int(((frame.index >= "2010-01-01") & (frame.index < "2015-01-01")).sum())
    # The first exchange session is normally 2010-01-04, so checking equality
    # with the calendar date incorrectly rejects a complete fixed baseline.
    if frame.empty or frame.index.min() > pd.Timestamp("2010-01-31") or baseline_rows < 1000:
        raise ValueError("The fixed 2010-2014 prior history is incomplete")
    if "is_provisional" in frame.columns:
        frame = frame.loc[~frame["is_provisional"].fillna(False).astype(bool)].copy()
    raw_columns = [f"jp_oc_{ticker}" for ticker in JP_TICKERS]
    raw_targets = frame[raw_columns].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(frame.index[(frame.index >= "2015-01-01") & (frame.index <= "2017-12-29")])
    first_sessions = set()
    for year in (2015, 2016, 2017):
        in_year = dates[dates.year == year]
        if len(in_year):
            first_sessions.add(pd.Timestamp(in_year[0]))
    dates = pd.DatetimeIndex([date for date in dates if date not in first_sessions])
    date_positions = np.searchsorted(frame.index.values, dates.values)
    full_y = raw_targets[np.isin(frame.index, dates)]
    if len(full_y) != len(dates) or not np.isfinite(full_y).all():
        raise ValueError("2015-2017 evaluation requires complete open-to-close targets")

    model = ProductionBLPXModel(app.v2.blpx)
    model.clear_caches()
    common = model._prepare_common_inputs(
        frame,
        horizon=1,
        y_jp_target=frame[raw_columns].to_numpy(dtype=float),
        open_910_returns=None,
        allow_implicit_io=False,
    )
    all_returns = np.asarray(common["jp_res_returns_p3"], dtype=float)
    targets = all_returns[date_positions, N_US : N_US + N_JP]
    c_full = np.asarray(common["c_full_p3"], dtype=float)
    current_labels = {ticker: {k: float(v) for k, v in row.items()} for ticker, row in SENSITIVITY_LABELS.items()}
    base_v0 = np.asarray(common["v0_static"], dtype=float)
    prior_cfg = audit._read_yaml(audit.PRIOR_CONFIG_PATH)
    caps = pd.read_csv(audit.CAP_PATH)
    fixed_labels = audit._labels_from_industry_table(prior_cfg, prior_cfg, caps, 2017)
    fixed_v0 = audit._v0_for({
        **current_labels,
        **{ticker: {**current_labels[ticker], **fixed_labels[ticker]} for ticker in JP_TICKERS},
    })
    pit_v0_by_year: dict[int, np.ndarray] = {}
    pit_labels_by_year: dict[int, dict[str, dict[str, float]]] = {}
    for year, snapshot_year in ((2015, 2014), (2016, 2015), (2017, 2016)):
        pit_labels = audit._labels_from_industry_table(prior_cfg, prior_cfg, caps, snapshot_year)
        pit_labels_by_year[year] = pit_labels
        merged = copy.deepcopy(current_labels)
        for ticker in JP_TICKERS:
            merged[ticker].update(pit_labels[ticker])
        pit_v0_by_year[year] = audit._v0_for(merged)
    if not np.allclose(base_v0, audit._v0_for(current_labels), rtol=0.0, atol=1e-14):
        raise ValueError("Current production static prior differs from the ticker-registry build")

    predictions = {name: np.full((len(dates), N_JP), np.nan) for name in ("current_17", "legacy_33_fixed_2017", "legacy_33_pit_cap")}
    for local_i, (date, current_index) in enumerate(zip(dates, date_positions, strict=True)):
        for name, v0 in (
            ("current_17", base_v0),
            ("legacy_33_fixed_2017", fixed_v0),
            ("legacy_33_pit_cap", pit_v0_by_year[int(date.year)]),
        ):
            result = model.compute_blp_signal(
                all_returns=all_returns,
                current_index=int(current_index),
                gap_override=None,
                betas_t=None,
                topix_night_t=None,
                v0_static=v0,
                c_full=c_full,
                is_residual=True,
                return_matrices=True,
            )
            mu, _ = build_raw_distribution(result, vol_adjusted_target=bool(model.vol_adjusted_target))
            mu = np.asarray(mu, dtype=float)
            if mu.shape != (N_JP,) or not np.isfinite(mu).all():
                raise ValueError(f"Invalid forecast {name} on {date.date()}")
            predictions[name][local_i] = mu
        if (local_i + 1) % 100 == 0 or local_i + 1 == len(dates):
            print(f"forecast rows {local_i + 1}/{len(dates)}", flush=True)

    daily_by_variant = {name: _daily_metrics(pred, targets, dates) for name, pred in predictions.items()}
    for name, daily in daily_by_variant.items():
        daily.to_csv(RESULTS_DIR / f"daily_{name}.csv", index=False)
    prediction_rows = []
    for i, date in enumerate(dates):
        for j, ticker in enumerate(JP_TICKERS):
            prediction_rows.append({
                "trade_date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "residual_target": float(targets[i, j]),
                **{f"mu_{name}": float(pred[i, j]) for name, pred in predictions.items()},
            })
    pd.DataFrame(prediction_rows).to_csv(RESULTS_DIR / "daily_predictions.csv", index=False)
    label_rows = []
    for year, labels in pit_labels_by_year.items():
        for ticker in JP_TICKERS:
            label_rows.append({"evaluation_year": year, "cap_snapshot_year": year - 1, "ticker": ticker, **labels[ticker]})
    pd.DataFrame(label_rows).to_csv(RESULTS_DIR / "annual_pit_aggregated_labels.csv", index=False)

    comparisons = {}
    pairs = (
        ("fixed_2017_vs_pit", "legacy_33_fixed_2017", "legacy_33_pit_cap"),
        ("current_vs_fixed_2017", "current_17", "legacy_33_fixed_2017"),
        ("current_vs_pit", "current_17", "legacy_33_pit_cap"),
    )
    for label, left, right in pairs:
        a = daily_by_variant[left].reset_index(drop=True)
        b = daily_by_variant[right].reset_index(drop=True)
        if not np.array_equal(a["trade_date"].to_numpy(), b["trade_date"].to_numpy()):
            raise ValueError(f"Paired date mismatch for {label}")
        comparisons[label] = {
            "rank_ic_delta_mean": float((b["rank_ic"] - a["rank_ic"]).mean()),
            "rank_ic_delta_ci95": _block_ci((b["rank_ic"] - a["rank_ic"]).to_numpy()),
            "mae_delta_bps_mean": float((b["mae"] - a["mae"]).mean() * 10000.0),
            "mae_delta_ci95_bps": [value * 10000.0 for value in _block_ci((b["mae"] - a["mae"]).to_numpy())],
            "paired_dates": int(len(a)),
        }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "evaluation": {
            "start_date": dates.min().strftime("%Y-%m-%d"),
            "end_date": dates.max().strftime("%Y-%m-%d"),
            "first_jp_sessions_excluded": [x.strftime("%Y-%m-%d") for x in sorted(first_sessions)],
            "dates": int(len(dates)),
            "target": "open-to-close residual return; not 09:10 target",
            "scope": "retrospective correction diagnostic; not unused OOS",
        },
        "variants": {name: _metric_summary(daily) for name, daily in daily_by_variant.items()},
        "comparisons": comparisons,
        "cap_snapshot_year_by_evaluation_year": {"2015": 2014, "2016": 2015, "2017": 2016},
        "cap_table_sha256": hashlib.sha256(audit.CAP_PATH.read_bytes()).hexdigest(),
        "df_exec_sha256": audit.dataframe_fingerprint(frame),
        "model_param_set": model.param_set,
        "V0_frobenius_fixed2017_vs_annual_pit_by_year": {
            str(year): float(np.linalg.norm(fixed_v0 - pit_v0_by_year[year], ord="fro"))
            for year in pit_v0_by_year
        },
        "production_modified": False,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [
        "# 2015〜2017 JPX33時価総額ウェイトのPIT訂正影響",
        "",
        "## 判定",
        "",
        "**2017年末ウェイトを2015〜2017年へ遡及する比較は時点整合性を満たさない。** 前年12月のJPX業種時価総額を使い、各年最初の日本取引日を除外して翌日から有効とした修正版を、同じ3年期間で比較した。成績は過去を既に確認した回顧診断で、本番採用や未使用OOSの根拠にはしない。",
        "",
        f"- 対象日は{summary['evaluation']['start_date']}〜{summary['evaluation']['end_date']}の{summary['evaluation']['dates']}日。2015〜2017年の各年最初の取引日 {', '.join(summary['evaluation']['first_jp_sessions_excluded'])} は、JPXが第1営業日13時以降に前年末資料を公表するため比較から除いた。",
        "- ターゲットはTOPIX残差化した始値→大引けリターン。各日のBLPX入力窓は当日を除外し、c_fullは2010〜2014固定。9:10→大引け成績ではない。",
        "- fixed_2017 は従前の33業種スコアを2017年12月市場時価総額で17ETF priorに写像。annual PITは2015に2014、2016に2015、2017に2016の年末業種時価総額を使用した。",
        "",
        "## 予測指標",
        "",
        "| 変種 | paired日数 | mean Rank IC | mean MAE (bp) | mean RMSE (bp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in summary["variants"].items():
        rows.append(f"| {name} | {metric['paired_dates']} | {metric['mean_rank_ic']:+.4f} | {metric['mean_mae_bps']:.2f} | {metric['mean_rmse_bps']:.2f} |")
    rows.extend([
        "",
        "### 2017固定ウェイトと年次PITウェイトの差",
        "",
        f"| 比較 | ΔRank IC 95% block CI | ΔMAE bp 95% block CI | ΔRank IC平均 | ΔMAE平均 (bp) | 日数 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in comparisons.items():
        ci_ic = item["rank_ic_delta_ci95"]
        ci_mae = item["mae_delta_ci95_bps"]
        rows.append(f"| {name} | [{ci_ic[0]:+.4f}, {ci_ic[1]:+.4f}] | [{ci_mae[0]:+.2f}, {ci_mae[1]:+.2f}] | {item['rank_ic_delta_mean']:+.4f} | {item['mae_delta_bps_mean']:+.2f} | {item['paired_dates']} |")
    rows.extend([
        "",
        "CIは20営業日non-circular moving-block bootstrap、5,000回、seed 20260924。Δは右辺variant−左辺variant。正のMAE差は誤差悪化を表す。",
        "",
        "## 時点整合・解釈",
        "",
        "PIT年次更新は固定2017表より情報整合的だが、JPXの年末業種時価総額は浮動株調整ウェイトではない。2015〜2021はFirst Section、2022年以降はPrimeへ公表対象市場が変わる。また当研究の比較区間は既知データであり、修正後でも将来OOSではない。",
        "この検証は固定2017ウェイトが作る入力priorの差を実際の3年予測指標で測った。ここで示すのは同じ実績期間内での訂正影響であり、訂正後モデルが一般に優れるという結論ではない。",
        "",
        "## 再現情報",
        "",
        f"- スクリプト: `{Path(__file__).relative_to(ROOT)}`。prior mapping: `{audit.PRIOR_CONFIG_PATH.relative_to(ROOT)}`。年次時価総額: `{audit.CAP_PATH.relative_to(ROOT)}` と `{audit.CAP_SOURCE_PATH.relative_to(ROOT)}`。",
        f"- df_exec SHA-256: `{summary['df_exec_sha256']}`。BLPX parameter set: `{model.param_set}`。本番設定・本番コードは変更なし。",
        "",
    ])
    report = REPORT_DIR / "report.md"
    report.write_text("\n".join(rows), encoding="utf-8")

    registry = ExperimentRegistry(REGISTRY_PATH)
    related = [r.name for r in registry if "jpx33" in r.name.lower() or "sensitivity" in r.name.lower()]
    for name, metric in summary["variants"].items():
        if name == "current_17":
            continue
        registry.record(ExperimentRecord(
            name=f"{EXPERIMENT_ID}_{name}",
            hypothesis="Replacing the fixed 2017 JPX industry market-cap map with the last published prior-year snapshot changes the 2015-2017 17-ETF residual forecast, and the effect should be measured rather than assumed.",
            start_time=started,
            end_time=datetime.now(UTC),
            parameters={
                "variant": name,
                "evaluation_years": [2015, 2016, 2017],
                "cap_snapshots_by_year": {"2015": 2014, "2016": 2015, "2017": 2016} if name == "legacy_33_pit_cap" else {"2015": 2017, "2016": 2017, "2017": 2017},
                "target": "open-to-close residual return",
                "cap_source": str(audit.CAP_PATH.relative_to(ROOT)),
                "df_exec_sha256": summary["df_exec_sha256"],
                "fixed_c_full_period": "2010-2014",
            },
            metrics={**metric, **comparisons["fixed_2017_vs_pit"]},
            decision=Decision.PENDING,
            report_path=str(report.relative_to(ROOT)),
            related_records=related,
        ))
    print(f"report={report}", flush=True)
    print(f"results={RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
