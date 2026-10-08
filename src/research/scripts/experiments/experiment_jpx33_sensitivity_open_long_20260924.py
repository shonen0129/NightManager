#!/usr/bin/env python3
"""Long-period open-to-close comparison for the frozen JPX-33 sensitivity prior."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
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

import experiment_jpx33_sensitivity_prior_20260924 as prior  # noqa: E402

from leadlag.core.gap_adjustment import build_raw_distribution  # noqa: E402
from leadlag.data.market_data_cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, N_JP, N_US, SENSITIVITY_LABELS  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry  # noqa: E402
from leadlag.models.blpx.model import ProductionBLPXModel  # noqa: E402
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

CONFIG_PATH = ROOT / "configs/research/jpx33_sensitivity_open_long_2015_2026.yaml"
INDUSTRY_CONFIG_PATH = ROOT / "configs/research/jpx33_sensitivity_prior_2017.yaml"
REPORT_DIR = ROOT / "reports/20260924_jpx33_sensitivity_open_long"
RESULTS_DIR = ROOT / "var/results/20260924_jpx33_sensitivity_open_long"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
LOG = logging.getLogger("jpx33_sensitivity_open_long")


def _read_config() -> tuple[dict[str, Any], dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    industry_config = prior._read_config()
    if not isinstance(config, dict) or not isinstance(config.get("evaluation"), dict):
        raise TypeError("Long-period experiment configuration must be a YAML mapping")
    if config.get("base_industry_config") != str(INDUSTRY_CONFIG_PATH.relative_to(ROOT)):
        raise ValueError("Long-period experiment must point to the frozen JPX-33 config")
    return config, industry_config


def _annual_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["year"] = pd.to_datetime(work["trade_date"]).dt.year
    annual = (
        work.groupby(["variant", "year"], sort=True)
        .agg(
            paired_dates=("trade_date", "count"),
            mean_observed_etfs=("observed_etfs", "mean"),
            baseline_mean_daily_rank_ic=("baseline_rank_ic", "mean"),
            candidate_mean_daily_rank_ic=("candidate_rank_ic", "mean"),
            rank_ic_difference=("rank_ic_difference", "mean"),
            baseline_mean_daily_mae=("baseline_mae", "mean"),
            candidate_mean_daily_mae=("candidate_mae", "mean"),
            mae_difference=("mae_difference", "mean"),
            baseline_mean_daily_rmse=("baseline_rmse", "mean"),
            candidate_mean_daily_rmse=("candidate_rmse", "mean"),
            rmse_difference=("rmse_difference", "mean"),
        )
        .reset_index()
    )
    return annual


def _safe_rank_ic(prediction: np.ndarray, target: np.ndarray) -> tuple[float, bool]:
    """Treat a rankless cross-section as zero information and keep the date."""
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    rankless = (
        np.ptp(prediction) <= 1e-14
        or np.ptp(target) <= 1e-14
    )
    if rankless:
        return 0.0, True
    value = spearmanr(prediction, target).statistic
    if not np.isfinite(value):
        return 0.0, True
    return float(value), False


def _variant_metrics(
    baseline: np.ndarray,
    variant: np.ndarray,
    targets: np.ndarray,
    observations: np.ndarray,
    dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    evaluation = config["evaluation"]
    min_assets = int(evaluation["min_observed_etfs_per_date"])
    rows: list[dict[str, Any]] = []
    for row_index, date in enumerate(dates):
        observed = observations[row_index]
        if int(observed.sum()) < min_assets:
            continue
        base = baseline[row_index]
        candidate = variant[row_index]
        actual = targets[row_index]
        mask = observed & np.isfinite(base) & np.isfinite(candidate) & np.isfinite(actual)
        if int(mask.sum()) < min_assets:
            continue
        base_ic, base_rankless = _safe_rank_ic(base[mask], actual[mask])
        candidate_ic, candidate_rankless = _safe_rank_ic(candidate[mask], actual[mask])
        base_error = base[mask] - actual[mask]
        candidate_error = candidate[mask] - actual[mask]
        rows.append(
            {
                "trade_date": date.strftime("%Y-%m-%d"),
                "observed_etfs": int(mask.sum()),
                "baseline_rank_ic": base_ic,
                "candidate_rank_ic": candidate_ic,
                "rank_ic_difference": candidate_ic - base_ic,
                "baseline_rankless": base_rankless,
                "candidate_rankless": candidate_rankless,
                "baseline_mae": float(np.mean(np.abs(base_error))),
                "candidate_mae": float(np.mean(np.abs(candidate_error))),
                "mae_difference": float(np.mean(np.abs(candidate_error)) - np.mean(np.abs(base_error))),
                "baseline_rmse": float(np.sqrt(np.mean(base_error**2))),
                "candidate_rmse": float(np.sqrt(np.mean(candidate_error**2))),
                "rmse_difference": float(np.sqrt(np.mean(candidate_error**2)) - np.sqrt(np.mean(base_error**2))),
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No paired evaluation dates passed the observation mask")
    bootstrap = evaluation["bootstrap"]
    rank_ci = prior._moving_block_ci(
        daily["rank_ic_difference"].to_numpy(dtype=float),
        block_length=int(bootstrap["block_length_trading_dates"]),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    mae_ci = prior._moving_block_ci(
        daily["mae_difference"].to_numpy(dtype=float),
        block_length=int(bootstrap["block_length_trading_dates"]),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]) + 1,
    )
    rank_delta = daily["rank_ic_difference"].to_numpy(dtype=float)
    mae_delta = daily["mae_difference"].to_numpy(dtype=float)
    metrics = {
        "paired_dates": int(len(daily)),
        "mean_observed_etfs": float(daily["observed_etfs"].mean()),
        "min_observed_etfs": int(daily["observed_etfs"].min()),
        "max_observed_etfs": int(daily["observed_etfs"].max()),
        "baseline_mean_daily_rank_ic": float(daily["baseline_rank_ic"].mean()),
        "candidate_mean_daily_rank_ic": float(daily["candidate_rank_ic"].mean()),
        "rank_ic_difference": float(rank_delta.mean()),
        "rank_ic_difference_ci95": list(rank_ci),
        "baseline_mean_daily_mae": float(daily["baseline_mae"].mean()),
        "candidate_mean_daily_mae": float(daily["candidate_mae"].mean()),
        "mae_difference": float(mae_delta.mean()),
        "mae_difference_ci95": list(mae_ci),
        "baseline_mean_daily_rmse": float(daily["baseline_rmse"].mean()),
        "candidate_mean_daily_rmse": float(daily["candidate_rmse"].mean()),
        "rmse_difference": float(daily["rmse_difference"].mean()),
        "baseline_rankless_dates": int(daily["baseline_rankless"].sum()),
        "candidate_rankless_dates": int(daily["candidate_rankless"].sum()),
    }
    return metrics, daily


def _markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in labels) + "|",
    ]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    config: dict[str, Any],
    industry_config: dict[str, Any],
    coverage: dict[str, Any],
    base_metrics: dict[str, Any],
    variant_metrics: dict[str, dict[str, Any]],
    annual: pd.DataFrame,
    decision: Decision,
    input_fingerprint: str,
    config_fingerprint: str,
    records_before: int,
    registry_record_name: str,
) -> None:
    eval_cfg = config["evaluation"]
    primary = variant_metrics["jpx33_primary"]
    low, high = primary["rank_ic_difference_ci95"]
    decision_text = {
        Decision.ADOPTED: "研究候補として採択。本番反映はしていない。",
        Decision.REJECTED: "不採用。",
        Decision.PENDING: "保留。採択基準を満たす十分な証拠は得られなかった。",
    }[decision]

    annual_primary = annual.loc[annual["variant"] == "jpx33_primary"].copy()
    annual_display = annual_primary.copy()
    for column in (
        "baseline_mean_daily_rank_ic",
        "candidate_mean_daily_rank_ic",
        "rank_ic_difference",
    ):
        annual_display[column] = annual_display[column].map(lambda value: f"{value:+.4f}")
    for column in ("mae_difference", "rmse_difference"):
        annual_display[column] = annual_display[column].map(lambda value: f"{value * 10000:+.2f}")
    annual_table = _markdown_table(
        annual_display,
        [
            "year",
            "paired_dates",
            "mean_observed_etfs",
            "baseline_mean_daily_rank_ic",
            "candidate_mean_daily_rank_ic",
            "rank_ic_difference",
            "mae_difference",
            "rmse_difference",
        ],
        ["年", "日数", "平均ETF数", "現行Rank IC", "33業種Rank IC", "差", "MAE差 (bps)", "RMSE差 (bps)"],
    )

    robustness_rows = []
    for name, label in (
        ("jpx33_toward_zero", "33業種ラベルを1段中心側へ"),
        ("jpx33_away_from_zero", "33業種ラベルを1段外側へ"),
    ):
        metrics = variant_metrics[name]
        ci_low, ci_high = metrics["rank_ic_difference_ci95"]
        robustness_rows.append(
            {
                "variant": label,
                "candidate_rank_ic": metrics["candidate_mean_daily_rank_ic"],
                "rank_ic_difference": metrics["rank_ic_difference"],
                "ci": f"[{ci_low:+.4f}, {ci_high:+.4f}]",
                "mae_difference": metrics["mae_difference"],
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    robustness["candidate_rank_ic"] = robustness["candidate_rank_ic"].map(lambda value: f"{value:+.4f}")
    robustness["rank_ic_difference"] = robustness["rank_ic_difference"].map(lambda value: f"{value:+.4f}")
    robustness["mae_difference"] = robustness["mae_difference"].map(lambda value: f"{value * 10000:+.2f}")
    robustness_table = _markdown_table(
        robustness,
        ["variant", "candidate_rank_ic", "rank_ic_difference", "ci", "mae_difference"],
        ["摂動", "平均Rank IC", "IC差", "差95%ブロックCI", "MAE差 (bps)"],
    )
    annual_positive = int((annual_primary["rank_ic_difference"] > 0.0).sum())
    annual_count = int(len(annual_primary))

    report = f"""# JPX33感応度 prior の始値→大引け長期比較

## 判定

**{decision_text}**

事前固定した採択条件は、paired日数100日以上、平均日次Rank IC差が+0.02以上、20営業日 moving-block bootstrap 95%区間の下限が0より大きいこと、平均日次MAEが悪化しないこと、±1段の摂動が極端に逆方向にならないこと。長期期間全体の基準指標はTOPIX残差リターンに対する日次クロスセクションRank IC。

## 比較内容

- baseline: `src/leadlag/data/tickers.py` の現行w3–w6感応度ラベル。
- candidate: 固定済みのJPX 33業種ラベルを2017-12-29の業種時価総額でTOPIX-17の17ETFへ集約し、現行の17次元BLPX静的priorへ入れる。
- 33業種から17業種への対応は[JPXの業種分類資料]({industry_config['provenance']['taxonomy_source']})、時価総額は[JPXの2017年12月末資料]({industry_config['provenance']['market_cap_source']})に基づく。JPX公式分類は33業種。
- 比較するモデル、米国15ETF感応度、BLPX設定、対象ETFは同じ。変えるのは日本側の静的感応度priorだけ。モデルとターゲットは引き続き17ETF次元。
- この比較は「33業種を別々に予測してから17へ戻す」実験ではなく、33業種の感応度情報を使って17次元モデルのpriorを作る実験。

## 期間とデータ

- 評価期間: {coverage['start_date']}〜{coverage['end_date']}（設定した上限は{eval_cfg['end_date']}）。2015年から2026年9月18日までの確認済み行を使用。
- 対象営業日: {coverage['all_trade_dates']}日、paired評価日: {coverage['eligible_dates']}日。1日あたり観測ETFは最少{coverage['min_observed_etfs']}、平均{coverage['mean_observed_etfs']:.2f}、最多{coverage['max_observed_etfs']}。17ETF全て揃った日は{coverage['dates_with_all_17']}日。
- 実測ターゲット: `jp_oc_*` の始値→大引けリターン。予測・主要評価ターゲットはTOPIXに対する残差リターンで、各日のベータは過去60行から推定。当日ターゲットを評価値として使うだけで、当日の学習窓には含めない。
- BLPXの学習窓は[t−window, t)で評価日tの行を除外。基準相関c_fullは2010–2014固定。終端の暫定行は除外し、後日の行も入力フレームから除いた。
- 生の始値→大引けリターン自体は予測モデルの教師ターゲットにしているが、誤差・Rank ICの正式比較は残差化後のターゲット上で行う。したがって、この数値をTOPIX込みの価格リターン予測精度として解釈しない。

## 長期集計

| 指標 | 現行ラベル | 33業種集約 | 差 |
|---|---:|---:|---:|
| paired日数 | {base_metrics['paired_dates']} | {primary['paired_dates']} | 0 |
| 平均日次Rank IC | {base_metrics['baseline_mean_daily_rank_ic']:+.4f} | {primary['candidate_mean_daily_rank_ic']:+.4f} | {primary['rank_ic_difference']:+.4f} |
| Rank IC差 95% moving-block CI | — | — | [{low:+.4f}, {high:+.4f}] |
| 平均日次MAE | {base_metrics['baseline_mean_daily_mae'] * 10000:.2f} bps | {primary['candidate_mean_daily_mae'] * 10000:.2f} bps | {primary['mae_difference'] * 10000:+.2f} bps |
| 平均日次RMSE | {base_metrics['baseline_mean_daily_rmse'] * 10000:.2f} bps | {primary['candidate_mean_daily_rmse'] * 10000:.2f} bps | {primary['rmse_difference'] * 10000:+.2f} bps |

年別のRank IC差が正の年は{annual_positive}/{annual_count}年。2026年は9月18日までの途中経過。

{annual_table}

### 感度診断

{robustness_table}

区間推定は対応する日次Rank IC差に対するnon-circular moving-block bootstrap。ブロック長{eval_cfg['bootstrap']['block_length_trading_dates']}営業日、再標本化{eval_cfg['bootstrap']['resamples']}回、seed {eval_cfg['bootstrap']['seed']}。MAE差のCIもsummaryと日次成果物に記録した。

予測の横断面に順位差がなくSpearman相関が定義できない日は、評価日から除外せずRank IC=0として計上した。主候補でbaselineが順位なしの日は{primary['baseline_rankless_dates']}日、候補が順位なしの日は{primary['candidate_rankless_dates']}日。

## 解釈上の範囲

- 評価は予測精度の比較で、売買損益・執行コスト・市場中立ポートフォリオのSharpeやDDは計算していない。Sharpeの候補選択を行っていないためDSRは適用対象外。
- 感応度値は固定した専門判断ベースで、期間内リターンに当てはめて最適化していない。ただし、長期の過去比較であり、新しい完全未使用OOSを意味しない。
- 先行する短期9:10→大引け比較 `20260924_jpx33_sensitivity_prior` と評価ホライズンが違うため、その結果と直接混ぜず、ここでは始値→大引けだけで比較した。
- 訂正実行前のregistryは{records_before}件。初回の同名recordは標準化尺度の出力ミスを含むため成績判定には使わず、監査履歴として残した。訂正版は別名で追記した。先行短期比較は関連する1試行として扱う。未登録の過去試行が残る可能性はある。

## 再現情報

- 設定: `configs/research/jpx33_sensitivity_open_long_2015_2026.yaml` と `configs/research/jpx33_sensitivity_prior_2017.yaml`
- 実験コード: `src/research/scripts/experiments/experiment_jpx33_sensitivity_open_long_20260924.py`
- 実行環境: `.venv/bin/python`、外側の停止期限は15分
- 入力df_exec fingerprint: `{input_fingerprint}`
- 両研究設定を連結したSHA-256: `{config_fingerprint}`
- 成果物: `var/results/20260924_jpx33_sensitivity_open_long/`
- registry: `var/experiments/registry.jsonl`
- registry記録名: `{registry_record_name}`。初回の同名record（`20260924_jpx33_sensitivity_open_long`）は標準化予測値をリターンと誤認していたため、監査履歴として残し、この訂正recordを正式成績として追記した。
- 本番設定・本番コードは変更していない。
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = datetime.now(UTC)
    config, industry_config = _read_config()
    evaluation = config["evaluation"]
    start = pd.Timestamp(evaluation["start_date"])
    end = pd.Timestamp(evaluation["end_date"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_df_exec_from_local_cache()
    frame = frame.loc[frame.index <= end].copy()
    if "is_provisional" in frame.columns:
        provisional = frame["is_provisional"].fillna(False).astype(bool)
        frame = frame.loc[~provisional].copy()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("df_exec dates must be unique and increasing")
    if frame.empty or frame.index[0] > pd.Timestamp("2010-01-01"):
        raise ValueError("Input frame must contain the fixed 2010-2014 prior history")

    raw_columns = [f"jp_oc_{ticker}" for ticker in JP_TICKERS]
    missing = sorted(set(raw_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing open-to-close target columns: {missing}")
    raw_targets_frame = frame[raw_columns].astype(float)
    eval_dates_all = frame.index[(frame.index >= start) & (frame.index <= end)]
    if len(eval_dates_all) == 0:
        raise ValueError("No rows are available in the configured evaluation period")
    raw_target_eval = raw_targets_frame.loc[eval_dates_all].to_numpy(dtype=float)
    observations_all = np.isfinite(raw_target_eval)
    if evaluation.get("only_complete_confirmed_rows", False) and not observations_all.all():
        raise ValueError("Configured confirmed-row evaluation requires all 17 open-to-close targets")
    min_assets = int(evaluation["min_observed_etfs_per_date"])
    eligible_rows = observations_all.sum(axis=1) >= min_assets
    eval_dates = pd.DatetimeIndex(eval_dates_all[eligible_rows], name=frame.index.name)
    observations = observations_all[eligible_rows]
    raw_targets = raw_target_eval[eligible_rows]
    if len(eval_dates) == 0:
        raise RuntimeError("No open-to-close dates pass the configured observation threshold")

    app_config = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    model = ProductionBLPXModel(app_config.v2.blpx)
    model.clear_caches()
    common = model._prepare_common_inputs(
        frame,
        horizon=1,
        y_jp_target=raw_targets_frame.to_numpy(dtype=float),
        open_910_returns=None,
        allow_implicit_io=False,
    )
    baseline_prior = np.asarray(common["v0_static"], dtype=float)
    baseline_labels = copy.deepcopy(SENSITIVITY_LABELS)
    labels_primary = prior.aggregate_industry_labels(industry_config)
    labels_toward_zero = prior.aggregate_industry_labels(industry_config, shift="toward_zero")
    labels_away_zero = prior.aggregate_industry_labels(industry_config, shift="away_from_zero")
    priors = {
        "baseline": baseline_prior,
        "jpx33_primary": prior.build_static_prior(labels_primary),
        "jpx33_toward_zero": prior.build_static_prior(labels_toward_zero),
        "jpx33_away_from_zero": prior.build_static_prior(labels_away_zero),
    }
    if baseline_prior.shape != (N_US + N_JP, 6):
        raise ValueError(f"Unexpected static prior shape: {baseline_prior.shape}")
    history_start = int(np.searchsorted(frame.index.values, np.datetime64("2010-01-01")))
    history_end = int(np.searchsorted(frame.index.values, np.datetime64("2015-01-01")))
    if history_end - history_start < 126:
        raise ValueError("Fixed 2010-2014 baseline history is incomplete")

    all_returns = np.asarray(common["jp_res_returns_p3"], dtype=float)
    residual_targets = all_returns[:, N_US:][
        np.searchsorted(frame.index.values, eval_dates.values)
    ].copy()
    target_observations = np.isfinite(residual_targets) & observations
    if not np.array_equal(target_observations, observations):
        observations = target_observations
        eligible_rows = observations.sum(axis=1) >= min_assets
        eval_dates = eval_dates[eligible_rows]
        raw_targets = raw_targets[eligible_rows]
        residual_targets = residual_targets[eligible_rows]
        observations = observations[eligible_rows]
    if len(eval_dates) == 0:
        raise ValueError("No dates have enough finite residual target values")

    date_indices = np.searchsorted(frame.index.values, eval_dates.values)
    predictions = {
        name: np.full((len(eval_dates), N_JP), np.nan, dtype=float)
        for name in priors
    }
    c_full = np.asarray(common["c_full_p3"], dtype=float)
    for date_i, current_index in enumerate(date_indices):
        for name, static_prior in priors.items():
            result = model.compute_blp_signal(
                all_returns=all_returns,
                current_index=int(current_index),
                gap_override=None,
                betas_t=None,
                topix_night_t=None,
                v0_static=static_prior,
                c_full=c_full,
                is_residual=True,
                return_matrices=True,
            )
            # Without a gap input, ``signal`` remains standardized z. Restore
            # the raw residual-return mean before comparing it with returns.
            prediction, _ = build_raw_distribution(
                result,
                vol_adjusted_target=bool(model.vol_adjusted_target),
            )
            prediction = np.asarray(prediction, dtype=float)
            if prediction.shape != (N_JP,) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid prediction for {eval_dates[date_i]} ({name})")
            predictions[name][date_i] = prediction
        if (date_i + 1) % 100 == 0 or date_i + 1 == len(eval_dates):
            LOG.info("Scored %d/%d evaluation dates", date_i + 1, len(eval_dates))

    residual_targets[~observations] = np.nan
    metrics_config = {
        "evaluation": {
            **evaluation,
            "bootstrap": evaluation["bootstrap"],
        }
    }
    base_metrics, _ = _variant_metrics(
        predictions["baseline"], predictions["baseline"], residual_targets,
        observations, eval_dates, metrics_config,
    )
    variant_metrics: dict[str, dict[str, Any]] = {}
    daily_frames = []
    for name in ("jpx33_primary", "jpx33_toward_zero", "jpx33_away_from_zero"):
        metrics, daily = _variant_metrics(
            predictions["baseline"], predictions[name], residual_targets,
            observations, eval_dates, metrics_config,
        )
        variant_metrics[name] = metrics
        daily.insert(0, "variant", name)
        daily_frames.append(daily)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    annual = _annual_metrics(daily_all)
    decision = prior._decision(
        variant_metrics["jpx33_primary"],
        {
            "toward_zero": variant_metrics["jpx33_toward_zero"],
            "away_from_zero": variant_metrics["jpx33_away_from_zero"],
        },
        metrics_config,
    )

    prediction_rows = []
    for day_i, date in enumerate(eval_dates):
        for asset_i, ticker in enumerate(JP_TICKERS):
            prediction_rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "open_to_close_return": float(raw_targets[day_i, asset_i])
                    if observations[day_i, asset_i] else np.nan,
                    "topix_residual_target": float(residual_targets[day_i, asset_i])
                    if observations[day_i, asset_i] else np.nan,
                    "observed": bool(observations[day_i, asset_i]),
                    "baseline_prediction": float(predictions["baseline"][day_i, asset_i]),
                    "jpx33_primary_prediction": float(predictions["jpx33_primary"][day_i, asset_i]),
                    "jpx33_toward_zero_prediction": float(predictions["jpx33_toward_zero"][day_i, asset_i]),
                    "jpx33_away_from_zero_prediction": float(predictions["jpx33_away_from_zero"][day_i, asset_i]),
                }
            )
    pd.DataFrame(prediction_rows).to_csv(RESULTS_DIR / "daily_predictions.csv", index=False)
    daily_all.to_csv(RESULTS_DIR / "daily_metrics.csv", index=False)
    annual.to_csv(RESULTS_DIR / "annual_metrics.csv", index=False)

    label_rows = []
    for ticker in JP_TICKERS:
        row = {"ticker": ticker}
        for factor in ("w3", "w4", "w5", "w6"):
            row[f"baseline_{factor}"] = float(baseline_labels[ticker][factor])
            row[f"jpx33_{factor}"] = float(labels_primary[ticker][factor])
            row[f"toward_zero_{factor}"] = float(labels_toward_zero[ticker][factor])
            row[f"away_from_zero_{factor}"] = float(labels_away_zero[ticker][factor])
        label_rows.append(row)
    pd.DataFrame(label_rows).to_csv(RESULTS_DIR / "aggregated_labels.csv", index=False)

    all_observed_counts = observations.sum(axis=1)
    coverage = {
        "start_date": eval_dates[0].strftime("%Y-%m-%d"),
        "end_date": eval_dates[-1].strftime("%Y-%m-%d"),
        "all_trade_dates": int(len(eval_dates_all)),
        "eligible_dates": int(len(eval_dates)),
        "dates_with_all_17": int((all_observed_counts == N_JP).sum()),
        "min_observed_etfs": int(all_observed_counts.min()),
        "mean_observed_etfs": float(all_observed_counts.mean()),
        "max_observed_etfs": int(all_observed_counts.max()),
        "latest_confirmed_input_date": frame.index[-1].strftime("%Y-%m-%d"),
    }
    input_fingerprint = dataframe_fingerprint(frame)
    config_fingerprint = hashlib.sha256(
        CONFIG_PATH.read_bytes() + b"\0" + INDUSTRY_CONFIG_PATH.read_bytes()
    ).hexdigest()
    registry = ExperimentRegistry(REGISTRY_PATH)
    existing_records = list(registry)
    records_before = len(existing_records)
    existing_names = {record.name for record in existing_records}
    registry_record_name = config["experiment_id"]
    if registry_record_name in existing_names:
        suffix = 1
        while f"{registry_record_name}_correction_{suffix}" in existing_names:
            suffix += 1
        registry_record_name = f"{registry_record_name}_correction_{suffix}"
    _write_report(
        config=config,
        industry_config=industry_config,
        coverage=coverage,
        base_metrics=base_metrics,
        variant_metrics=variant_metrics,
        annual=annual,
        decision=decision,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        records_before=records_before,
        registry_record_name=registry_record_name,
    )

    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision.value,
        "coverage": coverage,
        "baseline": base_metrics,
        "variants": variant_metrics,
        "annual_metrics": annual.to_dict(orient="records"),
        "bootstrap": evaluation["bootstrap"],
        "input_df_exec_sha256": input_fingerprint,
        "research_configs_sha256": config_fingerprint,
        "production_blpx_param_set": model.param_set,
        "historical_registry_records_before": records_before,
        "registry_record_name": registry_record_name,
        "related_records": ["20260924_jpx33_sensitivity_prior"],
        "production_modified": False,
        "report_path": str((REPORT_DIR / "report.md").relative_to(ROOT)),
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    registry.record(
        ExperimentRecord(
            name=registry_record_name,
            hypothesis=(
                "A fixed 2017 market-cap-weighted JPX-33 sensitivity prior aggregated "
                "to TOPIX-17 improves long-period open-to-close residual forecast accuracy."
            ),
            start_time=started,
            end_time=datetime.now(UTC),
            parameters={
                "baseline": "current SENSITIVITY_LABELS in ticker registry",
                "candidate": "frozen JPX33 grid labels weighted to TOPIX-17",
                "market_cap_as_of": industry_config["provenance"]["market_cap_as_of"],
                "evaluation": evaluation,
                "target_dimension": N_JP,
                "evaluation_dates": coverage["eligible_dates"],
                "target": "jp_oc, residualized on TOPIX with strictly historical rolling beta",
                "related_prior_record": "20260924_jpx33_sensitivity_prior",
                "production_modified": False,
                "data_fingerprint": input_fingerprint,
                "config_sha256": config_fingerprint,
            },
            metrics={
                "baseline_mean_daily_rank_ic": base_metrics["baseline_mean_daily_rank_ic"],
                "candidate_mean_daily_rank_ic": variant_metrics["jpx33_primary"]["candidate_mean_daily_rank_ic"],
                "rank_ic_difference": variant_metrics["jpx33_primary"]["rank_ic_difference"],
                "rank_ic_difference_ci95": variant_metrics["jpx33_primary"]["rank_ic_difference_ci95"],
                "baseline_mean_daily_mae": base_metrics["baseline_mean_daily_mae"],
                "candidate_mean_daily_mae": variant_metrics["jpx33_primary"]["candidate_mean_daily_mae"],
                "mae_difference": variant_metrics["jpx33_primary"]["mae_difference"],
                "paired_dates": variant_metrics["jpx33_primary"]["paired_dates"],
                "annual_positive_rank_ic_difference_years": int(
                    (annual.loc[annual["variant"] == "jpx33_primary", "rank_ic_difference"] > 0).sum()
                ),
                "annual_years": int(annual.loc[annual["variant"] == "jpx33_primary", "year"].nunique()),
                "trial_count_in_current_experiment": 1,
                "dsr": None,
                "dsr_reason": "Forecast accuracy comparison; no portfolio Sharpe was selected. Corrects the prior record's standardized-versus-return-scale output mistake.",
            },
            decision=decision,
            report_path=str((REPORT_DIR / "report.md").relative_to(ROOT)),
            related_records=["20260924_jpx33_sensitivity_prior"],
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
