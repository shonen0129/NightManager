#!/usr/bin/env python3
"""Compare a predeclared JPX-33 sensitivity prior with current TOPIX-17 labels.

The model stays at 17 JP ETF targets. The 33 industry scores are aggregated
into 17 prior labels using a fixed 2017-12-29 market-cap snapshot. Evaluation
uses only dates and ETF targets with an observed local 09:10 bar.
"""
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

from leadlag.core.correlation import build_v3_static  # noqa: E402
from leadlag.data.intraday_inputs import (  # noqa: E402
    build_open_910_returns,
    compute_jp_target_returns,
)
from leadlag.data.market_data_cache import (  # noqa: E402
    load_df_exec_from_local_cache,
    load_intraday_cache,
)
from leadlag.data.pit_lake import PITDataLake  # noqa: E402
from leadlag.data.tickers import JP_TICKERS, N_JP, N_US, SENSITIVITY_LABELS  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry  # noqa: E402
from leadlag.models.blpx.model import ProductionBLPXModel  # noqa: E402
from leadlag.pipeline.gap_distribution import (  # noqa: E402
    compute_gap_distribution,
    select_gap_coefficients,
)
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint  # noqa: E402

CONFIG_PATH = ROOT / "configs/research/jpx33_sensitivity_prior_2017.yaml"
REPORT_DIR = ROOT / "reports/20260924_jpx33_sensitivity_prior"
RESULTS_DIR = ROOT / "var/results/20260924_jpx33_sensitivity_prior"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
LOG = logging.getLogger("jpx33_sensitivity_prior")


def _read_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Experiment configuration must be a YAML mapping")
    industry_table = raw.get("industries", {})
    if isinstance(industry_table, dict):
        columns = list(industry_table.get("columns", []))
        rows = list(industry_table.get("rows", []))
        if not columns or any(len(row) != len(columns) for row in rows):
            raise ValueError("Industry rows must match the declared column order")
        raw["industries"] = [dict(zip(columns, row, strict=True)) for row in rows]
    if not isinstance(raw.get("industries"), list):
        raise TypeError("Experiment industries must be a list or a columns/rows table")
    return raw


def _shift_score(value: float, grid: list[float], direction: str) -> float:
    index = grid.index(float(value))
    if direction == "toward_zero":
        if value > 0:
            index = max(0, index - 1)
        elif value < 0:
            index = min(len(grid) - 1, index + 1)
    elif direction == "away_from_zero":
        if value > 0:
            index = min(len(grid) - 1, index + 1)
        elif value < 0:
            index = max(0, index - 1)
    else:
        raise ValueError(f"Unknown sensitivity shift: {direction}")
    return float(grid[index])


def aggregate_industry_labels(
    config: dict[str, Any],
    *,
    shift: str | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate fixed 33-class scores into 17 ETF scores."""
    grid = [float(value) for value in config["sensitivity_grid"]["values"]]
    industries = config["industries"]
    expected_tickers = set(config["sensitivity_grid"]["jp_tickers"].values())
    if len(industries) != 33 or len({item["name"] for item in industries}) != 33:
        raise ValueError("Expected exactly 33 unique official industry classes")

    sums: dict[str, dict[str, float]] = {}
    caps: dict[str, float] = {}
    for item in industries:
        parent = str(item["topix17"])
        cap = float(item["market_cap"])
        if cap <= 0:
            raise ValueError(f"Invalid market cap for {item['name']}")
        sums.setdefault(parent, {factor: 0.0 for factor in ("w3", "w4", "w5", "w6")})
        caps[parent] = caps.get(parent, 0.0) + cap
        for factor in ("w3", "w4", "w5", "w6"):
            score = float(item[factor])
            if score not in grid:
                raise ValueError(f"{item['name']} {factor} score is outside the frozen grid")
            if shift is not None:
                score = _shift_score(score, grid, shift)
            sums[parent][factor] += cap * score

    expected_parents = set(config["sensitivity_grid"]["jp_tickers"])
    if set(sums) != expected_parents:
        raise ValueError(
            f"Industry map parents differ from the TOPIX-17 registry: "
            f"missing={sorted(expected_parents - set(sums))}, "
            f"extra={sorted(set(sums) - expected_parents)}"
        )
    ticker_by_parent = config["sensitivity_grid"]["jp_tickers"]
    if set(ticker_by_parent.values()) != expected_tickers or expected_tickers != set(JP_TICKERS):
        raise ValueError("TOPIX-17 ticker mapping differs from leadlag.data.tickers")

    result: dict[str, dict[str, float]] = {}
    for parent, parent_ticker in ticker_by_parent.items():
        total_cap = caps[parent]
        result[str(parent_ticker)] = {
            factor: sums[parent][factor] / total_cap
            for factor in ("w3", "w4", "w5", "w6")
        }
    return result


def build_static_prior(labels: dict[str, dict[str, float]]) -> np.ndarray:
    """Build V0 while restoring the shared ticker registry even on failure."""
    saved = copy.deepcopy(SENSITIVITY_LABELS)
    try:
        for ticker in JP_TICKERS:
            SENSITIVITY_LABELS[ticker] = {
                **SENSITIVITY_LABELS[ticker],
                **labels[ticker],
            }
        return build_v3_static(N_US, N_JP, include_v4=True)
    finally:
        SENSITIVITY_LABELS.clear()
        SENSITIVITY_LABELS.update(saved)


def _rank_ic(prediction: np.ndarray, target: np.ndarray, observed: np.ndarray) -> float:
    mask = observed & np.isfinite(prediction) & np.isfinite(target)
    if int(mask.sum()) < 10:
        raise ValueError("A scored date has fewer than 10 observed ETF targets")
    value = spearmanr(prediction[mask], target[mask]).statistic
    if not np.isfinite(value):
        raise ValueError("Daily cross-sectional rank IC is not finite")
    return float(value)


def _moving_block_ci(
    paired_values: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(paired_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < block_length:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    max_start = len(values) - block_length
    estimates = np.empty(resamples, dtype=float)
    blocks_needed = int(np.ceil(len(values) / block_length))
    for index in range(resamples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sample = np.concatenate(
            [values[start : start + block_length] for start in starts]
        )[: len(values)]
        estimates[index] = float(np.mean(sample))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


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
    paired_rows: list[dict[str, Any]] = []
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
        base_ic = _rank_ic(base, actual, mask)
        candidate_ic = _rank_ic(candidate, actual, mask)
        base_error = base[mask] - actual[mask]
        candidate_error = candidate[mask] - actual[mask]
        paired_rows.append(
            {
                "trade_date": date.strftime("%Y-%m-%d"),
                "observed_etfs": int(mask.sum()),
                "baseline_rank_ic": base_ic,
                "candidate_rank_ic": candidate_ic,
                "rank_ic_difference": candidate_ic - base_ic,
                "baseline_mae": float(np.mean(np.abs(base_error))),
                "candidate_mae": float(np.mean(np.abs(candidate_error))),
                "mae_difference": float(
                    np.mean(np.abs(candidate_error)) - np.mean(np.abs(base_error))
                ),
                "baseline_rmse": float(np.sqrt(np.mean(base_error**2))),
                "candidate_rmse": float(np.sqrt(np.mean(candidate_error**2))),
                "rmse_difference": float(
                    np.sqrt(np.mean(candidate_error**2)) - np.sqrt(np.mean(base_error**2))
                ),
            }
        )
    daily = pd.DataFrame(paired_rows)
    if daily.empty:
        raise ValueError("No paired evaluation dates passed the observation mask")

    bootstrap = evaluation["bootstrap"]
    rank_delta = daily["rank_ic_difference"].to_numpy(dtype=float)
    mae_delta = daily["mae_difference"].to_numpy(dtype=float)
    rank_ci = _moving_block_ci(
        rank_delta,
        block_length=int(bootstrap["block_length_trading_dates"]),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    mae_ci = _moving_block_ci(
        mae_delta,
        block_length=int(bootstrap["block_length_trading_dates"]),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]) + 1,
    )
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
    }
    return metrics, daily


def _decision(
    primary: dict[str, Any],
    robustness: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> Decision:
    criteria = config["evaluation"]["acceptance"]
    if primary["paired_dates"] < int(criteria["min_paired_dates"]):
        return Decision.PENDING
    robust_deltas = [item["rank_ic_difference"] for item in robustness.values()]
    if (
        primary["rank_ic_difference"] >= float(criteria["min_rank_ic_improvement"])
        and primary["rank_ic_difference_ci95"][0]
        > float(criteria["rank_ic_difference_ci_lower_gt"])
        and primary["mae_difference"] <= float(criteria["mean_absolute_error_difference_lte"])
        and all(value >= -0.01 for value in robust_deltas)
    ):
        return Decision.ADOPTED
    if (
        primary["rank_ic_difference"] <= 0.0
        or primary["rank_ic_difference_ci95"][1] <= 0.0
    ):
        return Decision.REJECTED
    return Decision.PENDING


def _markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    header = "| " + " | ".join(labels) + " |"
    separator = "|" + "|".join("---" for _ in labels) + "|"
    lines = [header, separator]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _write_report(
    config: dict[str, Any],
    baseline_labels: dict[str, dict[str, float]],
    primary_labels: dict[str, dict[str, float]],
    baseline_metrics: dict[str, Any],
    primary_metrics: dict[str, Any],
    robustness_metrics: dict[str, dict[str, Any]],
    decision: Decision,
    coverage: dict[str, Any],
    input_fingerprint: str,
    config_fingerprint: str,
) -> None:
    label_rows = []
    for ticker in JP_TICKERS:
        label_rows.append(
            {
                "ticker": ticker,
                "baseline_w3": baseline_labels[ticker]["w3"],
                "candidate_w3": primary_labels[ticker]["w3"],
                "baseline_w4": baseline_labels[ticker]["w4"],
                "candidate_w4": primary_labels[ticker]["w4"],
                "baseline_w5": baseline_labels[ticker]["w5"],
                "candidate_w5": primary_labels[ticker]["w5"],
                "baseline_w6": baseline_labels[ticker]["w6"],
                "candidate_w6": primary_labels[ticker]["w6"],
            }
        )
    label_frame = pd.DataFrame(label_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    label_frame.to_csv(RESULTS_DIR / "aggregated_labels.csv", index=False)

    label_display_frame = label_frame.copy()
    for column in label_display_frame.columns:
        if column != "ticker":
            label_display_frame[column] = label_display_frame[column].round(3)
    label_table = _markdown_table(
        label_display_frame,
        [
            "ticker",
            "baseline_w3",
            "candidate_w3",
            "baseline_w4",
            "candidate_w4",
            "baseline_w5",
            "candidate_w5",
            "baseline_w6",
            "candidate_w6",
        ],
        [
            "ETF",
            "現行w3",
            "33業種w3",
            "現行w4",
            "33業種w4",
            "現行w5",
            "33業種w5",
            "現行w6",
            "33業種w6",
        ],
    )
    ci_low, ci_high = primary_metrics["rank_ic_difference_ci95"]
    decision_text = {
        Decision.ADOPTED: "研究候補として採択。本番反映はしていない。",
        Decision.REJECTED: "不採用。",
        Decision.PENDING: "保留。観測期間と統計的確度では採否を決めない。",
    }[decision]
    robust_lines = []
    for name, metrics in robustness_metrics.items():
        robust_lines.append(
            f"| {name} | {metrics['candidate_mean_daily_rank_ic']:.4f} | "
            f"{metrics['rank_ic_difference']:+.4f} | "
            f"[{metrics['rank_ic_difference_ci95'][0]:+.4f}, "
            f"{metrics['rank_ic_difference_ci95'][1]:+.4f}] | "
            f"{metrics['mae_difference']:+.6f} |"
        )
    robust_table = "\n".join(
        [
            "| variant | 平均日次Rank IC | IC差 | IC差95%ブロックCI | MAE差 |",
            "|---|---:|---:|---:|---:|",
            *robust_lines,
        ]
    )
    report = f"""# JPX33感応度 prior → TOPIX-17予測 実験

## 判定

**{decision_text}**

事前に固定した採択基準は、paired日数100日以上、平均日次Rank IC差が+0.02以上、その20営業日 moving-block bootstrap 95%区間の下限が0より大きいこと、平均日次MAEが悪化しないこと、さらに±1段の感度診断が極端に逆方向にならないこと。判定はこの基準に沿う。

## 仮説と比較

- 仮説: JPX33業種別に置いた景気循環・円安・エネルギー・金利/物価の感応度を、2017年末の業種時価総額でTOPIX-17へ畳み込むと、現行17ETFラベルより翌営業日の相対予測順位が改善する。
- baseline: src/leadlag/data/tickers.py の現行w3–w6ラベル。
- candidate: YAMLに事前固定した33業種の7段階ラベルを、2017-12-29時価総額で17ETFへ加重平均。集約後は再量子化しない。米国15 ETFラベルと他のBLPX設定は現行値のまま。
- 33→17の公式対応は[JPXの業種分類資料](https://www.jpx.co.jp/markets/indices/factsheets/files/000_fac2_sector.pdf)、ウェイトは[JPXの2017年12月末業種別時価総額](https://www.jpx.co.jp/markets/statistics-equities/misc/nlsgeu000002vgd0-att/201712.pdf)から固定。公式分類が33業種のため、この実験は33分類で実施した。
- 予測次元・実測ターゲット: ともにTOPIX-17の17次元。
- 評価対象: 1日 horizon の gap-adjusted 予測平均 mu_gap と、同日の実測9:10→大引けリターン。評価は予測精度だけで、ポートフォリオ損益は評価していない。

## データとPIT

- 期間: {coverage['start_date']}〜{coverage['end_date']}。ローカル5分足キャッシュで9:10観測が1 ETFあたり10銘柄以上ある {coverage['eligible_dates']} 日を使用。各日の実測値は9:10足を実際に観測したETFだけに限定し、同じマスクをbaseline/candidateへ適用。
- 実測9:10カバー率: 評価期間の {coverage['all_trade_dates']} 営業日のうち、何らかの9:10観測がある日は {coverage['dates_with_any_910']} 日。対象日のETF数は最少 {coverage['min_observed_etfs']}、平均 {coverage['mean_observed_etfs']:.2f}、最多 {coverage['max_observed_etfs']}。全17 ETFが同時に観測できた日は {coverage['dates_with_all_17']} 日。
- 9:10足のないターゲットセルはopen-to-closeで補完される計算経路だが、今回の精度指標から除外した。
- 予測日に使った履歴は当日行より前。BLPXの学習窓は[t−window, t)で当日ターゲットを除外。TOPIX残差化ベータは過去60行で推定し、当日データを学習に含めない。基準相関c_fullは既存仕様の2010–2014固定期間。データフレームを評価終了日 {coverage['end_date']} で打ち切り、後日の暫定行を使っていない。
- 2010–2014の固定基準相関・過去ラベルは、5分足がない期間では既存のopen-to-close fallbackを含む。これはproductionの現行履歴仕様を維持した制約。

## 結果

| 指標 | 現行ラベル | 33業種集約 | 差 |
|---|---:|---:|---:|
| paired日数 | {baseline_metrics['paired_dates']} | {primary_metrics['paired_dates']} | 0 |
| 平均日次クロスセクションRank IC | {baseline_metrics['baseline_mean_daily_rank_ic']:.4f} | {primary_metrics['candidate_mean_daily_rank_ic']:.4f} | {primary_metrics['rank_ic_difference']:+.4f} |
| Rank IC差 95% moving-block CI | — | — | [{ci_low:+.4f}, {ci_high:+.4f}] |
| 平均日次MAE | {baseline_metrics['baseline_mean_daily_mae'] * 10000:.2f} bps | {primary_metrics['candidate_mean_daily_mae'] * 10000:.2f} bps | {primary_metrics['mae_difference'] * 10000:+.4f} bps |
| 平均日次RMSE | {baseline_metrics['baseline_mean_daily_rmse'] * 10000:.2f} bps | {primary_metrics['candidate_mean_daily_rmse'] * 10000:.2f} bps | {primary_metrics['rmse_difference'] * 10000:+.4f} bps |

この短い区間では両方の平均Rank ICが負で、候補は現行をわずかにゼロ方向へ動かしただけ。MAE/RMSEの差も実務上ほぼゼロで、予測力が改善したとは結論できない。

感度診断（各33業種ラベルを各w列ごとに7段階グリッドで1段だけ中心側/外側へずらし、同じ事前固定ウェイトで再集約）:

{robust_table}

統計区間は日次の対応差に対するnon-circular moving-block bootstrap。ブロック長20営業日、再標本化{config['evaluation']['bootstrap']['resamples']}回、seed {config['evaluation']['bootstrap']['seed']}。実施前に既存registry 34件と関連レポートを確認し、この感応度priorと同じ実験名の登録はなかった。これ以外に未登録の過去試行が残る可能性はある。独立パラメータ探索は行っていない。Sharpe/リターン採択を行っていないためDSRは適用対象外。

## 33業種→17ETFラベル

各4因子の値は「景気循環」「円安恩恵」「エネルギー価格上昇恩恵」「金利/物価上昇恩恵」の仮説ラベル。銘柄リターンから推定・最適化していない専門判断ベースであり、今回のOOS期間を見て調整していない。

{label_table}

## 解釈と制約

- 「業種を細かく見る」効果だけを切り分けるため、33業種リターンを独立予測してから17ETFへ戻す実験ではなく、33業種情報を使って17次元モデルの静的priorを組み替える実験。
- 2017年末の時価総額は評価期間より前に固定したため将来構成を参照しない。一方で2026年の業種構成に対して古く、ウェイトの陳腐化がある。
- 9:10足キャッシュが2026-03-03〜2026-08-06の一部ETFに限られるため、評価は約5か月に留まる。ETF観測数も日ごとに10〜17で異なる。結果が良くても本番採用には不足で、より長い実9:10データでの未使用OOS再検証が必要。
- 既存の79サブセクターを独立次元で予測して17へ戻す過去実験とは、次元拡張ではなくpriorラベル集約を試す別仮説。過去の次元拡張結果を本実験の結論と混同しない。

## 再現情報

- 設定: configs/research/jpx33_sensitivity_prior_2017.yaml
- 実験コード: src/research/scripts/experiments/experiment_jpx33_sensitivity_prior_20260924.py
- 実行環境: .venv/bin/python
- 入力df_exec fingerprint: {input_fingerprint}
- 研究設定SHA-256: {config_fingerprint}
- 出力: var/results/20260924_jpx33_sensitivity_prior/
- レジストリ: var/experiments/registry.jsonl
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    started = datetime.now(UTC)
    config = _read_config()
    eval_cfg = config["evaluation"]
    start_date = str(eval_cfg["start_date"])
    end_date = str(eval_cfg["end_date"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_df_exec_from_local_cache()
    frame = frame.loc[frame.index <= pd.Timestamp(end_date)].copy()
    if "is_provisional" in frame.columns:
        provisional = frame["is_provisional"].fillna(False).astype(bool)
        frame = frame.loc[~provisional].copy()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("df_exec dates must be unique and increasing")

    bars = load_intraday_cache("5m")
    if bars is None or bars.empty:
        raise RuntimeError("A local 5-minute cache is required for observed 09:10 labels")
    open_910 = build_open_910_returns(frame, JP_TICKERS, df_5m=bars)
    target = compute_jp_target_returns(
        frame,
        JP_TICKERS,
        horizon=1,
        open_910_returns=open_910,
    )
    target_frame = pd.DataFrame(target, index=frame.index, columns=JP_TICKERS)
    eval_dates_all = frame.index[
        (frame.index >= pd.Timestamp(start_date))
        & (frame.index <= pd.Timestamp(end_date))
    ]
    obs_frame = open_910.loc[eval_dates_all, JP_TICKERS].notna()
    minimum = int(eval_cfg["min_observed_etfs_per_date"])
    eval_dates = pd.DatetimeIndex(
        obs_frame.index[obs_frame.sum(axis=1) >= minimum],
        name=frame.index.name,
    )
    if len(eval_dates) == 0:
        raise RuntimeError("No evaluation dates have enough observed 09:10 targets")
    observations = obs_frame.loc[eval_dates].to_numpy(dtype=bool)
    targets = target_frame.loc[eval_dates].to_numpy(dtype=float, copy=True)
    targets[~observations] = np.nan

    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    model = ProductionBLPXModel(app.v2.blpx)
    model.clear_caches()
    common = model._prepare_common_inputs(
        frame,
        horizon=1,
        y_jp_target=target_frame.reindex(frame.index).to_numpy(dtype=float),
        open_910_returns=open_910,
        allow_implicit_io=False,
    )
    baseline_prior = np.asarray(common["v0_static"], dtype=float)
    baseline_labels = copy.deepcopy(SENSITIVITY_LABELS)
    candidate_labels = aggregate_industry_labels(config)
    lower_labels = aggregate_industry_labels(config, shift="toward_zero")
    upper_labels = aggregate_industry_labels(config, shift="away_from_zero")
    priors = {
        "baseline": baseline_prior,
        "jpx33_primary": build_static_prior(candidate_labels),
        "jpx33_toward_zero": build_static_prior(lower_labels),
        "jpx33_away_from_zero": build_static_prior(upper_labels),
    }

    if common["v0_static"].shape != (N_US + N_JP, 6):
        raise ValueError(f"Unexpected prior shape {common['v0_static'].shape}")
    if (
        frame.index[0] > pd.Timestamp("2010-01-01")
        or np.asarray(common["all_returns_raw"]).shape[0] != len(frame)
    ):
        raise ValueError("Common inputs do not cover the required fixed historical frame")
    history_start = int(np.searchsorted(frame.index.values, np.datetime64("2010-01-01")))
    history_end = int(np.searchsorted(frame.index.values, np.datetime64("2015-01-01")))
    if history_end - history_start < 126:
        raise ValueError("Fixed 2010-2014 baseline history is incomplete")

    lake = PITDataLake(frame)
    indices = {date: int(frame.index.get_loc(date)) for date in eval_dates}
    snapshots = {}
    for date in eval_dates:
        snapshots[date] = lake.get_execution_snapshot(
            date + pd.Timedelta(hours=9, minutes=10),
            open_910,
        )

    predictions = {name: np.full((len(eval_dates), N_JP), np.nan) for name in priors}
    all_returns = np.asarray(common["jp_res_returns_p3"], dtype=float)
    c_full = np.asarray(common["c_full_p3"], dtype=float)
    for date_i, date in enumerate(eval_dates):
        current_index = indices[date]
        snapshot = snapshots[date]
        for variant_name, prior in priors.items():
            result = model.compute_blp_signal(
                all_returns=all_returns,
                current_index=current_index,
                gap_override=np.asarray(snapshot.jp_gap_returns, dtype=float),
                betas_t=np.asarray(snapshot.jp_betas, dtype=float),
                topix_night_t=float(snapshot.topix_night_return),
                v0_static=prior,
                c_full=c_full,
                is_residual=True,
                return_matrices=True,
            )
            gap_open_coef, topix_beta_coef = select_gap_coefficients(model, result)
            distribution = compute_gap_distribution(
                result,
                gap_override=np.asarray(snapshot.jp_gap_returns, dtype=float),
                betas_t=np.asarray(snapshot.jp_betas, dtype=float),
                topix_night_t=float(snapshot.topix_night_return),
                vol_adjusted_target=bool(model.vol_adjusted_target),
                gap_open_coef=gap_open_coef,
                topix_beta_coef=topix_beta_coef,
            )
            prediction = np.asarray(distribution.mu_gap, dtype=float)
            if prediction.shape != (N_JP,) or not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite or wrong-sized prediction for {date} ({variant_name})")
            predictions[variant_name][date_i] = prediction
        if (date_i + 1) % 20 == 0 or date_i + 1 == len(eval_dates):
            LOG.info("Scored %d/%d dates", date_i + 1, len(eval_dates))

    base_metrics, _ = _variant_metrics(
        predictions["baseline"],
        predictions["baseline"],
        targets,
        observations,
        eval_dates,
        config,
    )
    variant_metrics: dict[str, dict[str, Any]] = {}
    daily_frames: list[pd.DataFrame] = []
    for name in ("jpx33_primary", "jpx33_toward_zero", "jpx33_away_from_zero"):
        metrics, daily = _variant_metrics(
            predictions["baseline"],
            predictions[name],
            targets,
            observations,
            eval_dates,
            config,
        )
        variant_metrics[name] = metrics
        daily.insert(0, "variant", name)
        daily_frames.append(daily)
    primary_metrics = variant_metrics["jpx33_primary"]
    robustness_metrics = {
        "one grid step toward zero": variant_metrics["jpx33_toward_zero"],
        "one grid step away from zero": variant_metrics["jpx33_away_from_zero"],
    }
    decision = _decision(primary_metrics, robustness_metrics, config)

    prediction_rows = []
    for row_index, date in enumerate(eval_dates):
        for ticker_index, ticker in enumerate(JP_TICKERS):
            observed = bool(observations[row_index, ticker_index])
            prediction_rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "target_0910_observed": observed,
                    "target_return": float(targets[row_index, ticker_index])
                    if observed
                    else np.nan,
                    "baseline_mu_gap": float(predictions["baseline"][row_index, ticker_index]),
                    "jpx33_primary_mu_gap": float(
                        predictions["jpx33_primary"][row_index, ticker_index]
                    ),
                    "jpx33_toward_zero_mu_gap": float(
                        predictions["jpx33_toward_zero"][row_index, ticker_index]
                    ),
                    "jpx33_away_from_zero_mu_gap": float(
                        predictions["jpx33_away_from_zero"][row_index, ticker_index]
                    ),
                }
            )
    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame.to_csv(RESULTS_DIR / "daily_predictions.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "daily_metrics.csv", index=False
    )

    label_rows = []
    for ticker in JP_TICKERS:
        row = {"ticker": ticker}
        for factor in ("w3", "w4", "w5", "w6"):
            row[f"baseline_{factor}"] = float(baseline_labels[ticker][factor])
            row[f"jpx33_{factor}"] = float(candidate_labels[ticker][factor])
            row[f"toward_zero_{factor}"] = float(lower_labels[ticker][factor])
            row[f"away_from_zero_{factor}"] = float(upper_labels[ticker][factor])
        label_rows.append(row)
    pd.DataFrame(label_rows).to_csv(RESULTS_DIR / "aggregated_labels.csv", index=False)

    coverage = {
        "start_date": eval_dates[0].strftime("%Y-%m-%d"),
        "end_date": eval_dates[-1].strftime("%Y-%m-%d"),
        "eligible_dates": int(len(eval_dates)),
        "all_trade_dates": int(len(eval_dates_all)),
        "dates_with_any_910": int(obs_frame.any(axis=1).sum()),
        "dates_with_all_17": int((obs_frame.sum(axis=1) == N_JP).sum()),
        "min_observed_etfs": int(observations.sum(axis=1).min()),
        "mean_observed_etfs": float(observations.sum(axis=1).mean()),
        "max_observed_etfs": int(observations.sum(axis=1).max()),
        "local_5m_date_start": str(pd.Timestamp(bars.index.min()).date()),
        "local_5m_date_end": str(pd.Timestamp(bars.index.max()).date()),
    }
    input_fingerprint = dataframe_fingerprint(frame)
    config_fingerprint = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    _write_report(
        config,
        baseline_labels,
        candidate_labels,
        base_metrics,
        primary_metrics,
        robustness_metrics,
        decision,
        coverage,
        input_fingerprint,
        config_fingerprint,
    )

    summary = {
        "experiment_id": config["experiment_id"],
        "decision": decision.value,
        "coverage": coverage,
        "baseline": base_metrics,
        "variants": variant_metrics,
        "input_df_exec_sha256": input_fingerprint,
        "research_config_sha256": config_fingerprint,
        "production_blpx_param_set": model.param_set,
        "historical_registry_records_before": len(list(ExperimentRegistry(REGISTRY_PATH))),
        "report_path": str((REPORT_DIR / "report.md").relative_to(ROOT)),
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    registry = ExperimentRegistry(REGISTRY_PATH)
    registry.record(
        ExperimentRecord(
            name=config["experiment_id"],
            hypothesis=(
                "2017 market-cap-weighted JPX-33 sensitivity labels, aggregated to "
                "TOPIX-17, improve paired one-day 09:10-to-close forecast accuracy."
            ),
            start_time=started,
            end_time=datetime.now(UTC),
            parameters={
                "baseline": "current SENSITIVITY_LABELS in ticker registry",
                "candidate": "JPX33 grid labels market-cap-weighted to TOPIX-17",
                "market_cap_as_of": str(config["provenance"]["market_cap_as_of"]),
                "evaluation": config["evaluation"],
                "target_dimension": 17,
                "evaluation_dates": coverage["eligible_dates"],
                "sensitivity_variants": [
                    "one grid step toward zero",
                    "one grid step away from zero",
                ],
                "production_modified": False,
                "data_fingerprint": input_fingerprint,
                "config_sha256": config_fingerprint,
            },
            metrics={
                "baseline_mean_daily_rank_ic": base_metrics["baseline_mean_daily_rank_ic"],
                "candidate_mean_daily_rank_ic": primary_metrics["candidate_mean_daily_rank_ic"],
                "rank_ic_difference": primary_metrics["rank_ic_difference"],
                "rank_ic_difference_ci95": primary_metrics["rank_ic_difference_ci95"],
                "baseline_mean_daily_mae": base_metrics["baseline_mean_daily_mae"],
                "candidate_mean_daily_mae": primary_metrics["candidate_mean_daily_mae"],
                "mae_difference": primary_metrics["mae_difference"],
                "paired_dates": primary_metrics["paired_dates"],
                "trial_count_in_current_experiment": 1,
                "dsr": None,
                "dsr_reason": "Forecast-only comparison; no portfolio Sharpe was selected.",
            },
            decision=decision,
            report_path=str((REPORT_DIR / "report.md").relative_to(ROOT)),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
