#!/usr/bin/env python3
"""Historical walk-forward diagnostic for a BLPX-residual ML target.

This script reuses the audited 2026-09-27 collection and evaluates three
predeclared residual scales over 2020-2024. It never changes production
configuration or production artifacts. The periods are previously viewed and
cannot qualify as fresh OOS evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry
from leadlag.models.ml_order_overlay import DEFAULT_LGBM_KWARGS, ROUND_TRIP_COST
from leadlag.models.ml_overlay_artifact import save_overlay_model
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from research.experiments.ml_overlay_training import _train_overlay_lgbm
from research.scripts.experiments import retrain_ml_overlay_20260927 as retrain
from research.scripts.experiments import structural_artifact_evaluate as evaluator

REPORT = ROOT / "reports/20260929_ml_overlay_blpx_residual"
WORK = ROOT / "var/results/20260929_ml_overlay_blpx_residual"
YEARS = tuple(range(2020, 2025))
RESIDUAL_SCALES = (0.8, 1.0, 1.2)
STUDY_TRIALS = 1 + len(RESIDUAL_SCALES)  # raw target plus residual variants
PURGE_TRADING_DAYS = 5
ANNUALIZATION_DAYS = 245.0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _annual_sharpe(values: np.ndarray) -> float:
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values) / std * np.sqrt(ANNUALIZATION_DAYS)) if std > 1e-16 else 0.0


def _prepare_evaluator() -> tuple[pd.DataFrame, HistoricalInputs, Any, pd.DataFrame, dict]:
    evaluator.ROOT = ROOT
    evaluator.REPORT = REPORT
    evaluator.WORK = WORK / "walkforward"
    evaluator.GAP = retrain.GAP
    REPORT.mkdir(parents=True, exist_ok=True)
    source_provenance_path = retrain.REPORT / "input_provenance.json"
    source_reconciliation_path = retrain.REPORT / "reconciliation.json"
    _write_json(
        REPORT / "input_provenance.json",
        {
            "source_report": str(retrain.REPORT.relative_to(ROOT)),
            "source_input_provenance_sha256": _sha256(source_provenance_path.read_bytes()),
            "source_reconciliation_sha256": _sha256(source_reconciliation_path.read_bytes()),
            "historical_provider_available_at_proven": False,
            "note": "Inputs are reused from the audited 2026-09-27 collection; this is provenance linkage, not new availability-time evidence.",
        },
    )

    frame = pd.read_pickle(retrain.INPUT / "df_exec_corrected.pkl")
    history = HistoricalInputs(frame, **retrain._load(retrain.INPUT / "history_kwargs.pkl"))
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    training, base_decisions = retrain._load_collection()

    if training.duplicated(["trade_date", "ticker"]).any():
        raise ValueError("Cached training collection has duplicate date/ticker rows")
    required = {"trade_date", "ticker", "score", "mu_gap", "target"}
    missing = sorted(required - set(training.columns))
    if missing:
        raise ValueError(f"Cached training collection is missing columns: {missing}")
    numeric = training[["score", "mu_gap", "target"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Cached target or BLPX forecast contains non-finite values")
    return frame, history, app, training, base_decisions


def _transform_residual_target(rows: pd.DataFrame, scale: float) -> pd.DataFrame:
    transformed = rows.copy(deep=True)
    score = transformed["score"].to_numpy(dtype=float)
    side = np.where(score > 0.0, 1.0, -1.0)
    directional_baseline = side * transformed["mu_gap"].to_numpy(dtype=float)
    # Collected targets are directional realized return less a flat 10 bps.
    # Restore that fixed charge and learn the return surprise beyond BLPX;
    # trading costs depend on orders and are handled outside this allocator.
    transformed["target"] = (
        transformed["target"].to_numpy(dtype=float)
        + ROUND_TRIP_COST
        - scale * directional_baseline
    )
    if not np.isfinite(transformed["target"].to_numpy(dtype=float)).all():
        raise ValueError("Residual target contains non-finite values")
    return transformed


def _make_fold_model(
    training: pd.DataFrame,
    frame: pd.DataFrame,
    run_cfg: ProductionV2RunConfig,
    year: int,
    scale: float,
    source_hash: str,
    plan_hash: str,
) -> Any:
    prior_dates = frame.index[
        (frame.index >= "2015-01-05") & (frame.index < f"{year}-01-01")
    ]
    if len(prior_dates) <= PURGE_TRADING_DAYS:
        raise ValueError(f"Not enough prior dates for 5-day purge before {year}")
    cutoff = prior_dates[-(PURGE_TRADING_DAYS + 1)]
    rows = training.loc[pd.to_datetime(training["trade_date"]) <= cutoff].copy()
    if rows.empty:
        raise ValueError(f"No training rows before {cutoff.date()}")
    rows = _transform_residual_target(rows, scale)
    model = _train_overlay_lgbm(
        rows,
        lgbm_kwargs=DEFAULT_LGBM_KWARGS,
        use_ticker=True,
        use_classification=False,
        per_ticker_interactions=True,
        p_trade_scale=1.0,
    )
    fold_dir = WORK / "artifacts" / f"scale_{scale:.1f}" / f"evaluation_{year}"
    provenance = {
        "metadata_status": "verified",
        "train_start": "2015-01-05",
        "train_end": str(cutoff.date()),
        "label_asof_end": str(pd.to_datetime(rows["trade_date"]).max().date()),
        "data_hash": dataframe_fingerprint(rows.reset_index(drop=True)),
        "config_hash": hashlib.sha256(
            json.dumps(run_cfg.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "source_code_hash": source_hash,
        "evaluation_plan_sha256": plan_hash,
        "training_rows": int(len(rows)),
        "training_dates": int(rows["trade_date"].nunique()),
        "purged_trading_days": PURGE_TRADING_DAYS,
        "target_type": "blpx_residual",
        "baseline_residual_scale": scale,
        "target_round_trip_cost_bps": 0.0,
        "restored_source_round_trip_cost_bps": 10.0,
        "historical_provider_available_at_proven": False,
        "availability_evidence": "historical provider retrieval timestamps unavailable",
        "candidate_parameterizations": STUDY_TRIALS,
        "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
        "seed": 42,
    }
    save_overlay_model(model, fold_dir, training_metadata=provenance)
    return model


def _evaluate() -> dict[str, Any]:
    logging.basicConfig(level=logging.ERROR)
    study_start = datetime.now(UTC)
    frame, history, app, training, base = _prepare_evaluator()
    code_hash = _sha256(Path(__file__).read_bytes())
    plan_hash = _sha256((REPORT / "evaluation_plan.md").read_bytes())
    results: dict[str, list[dict[str, Any]]] = {str(scale): [] for scale in RESIDUAL_SCALES}
    daily: list[pd.DataFrame] = []
    trial_sharpes: dict[str, float] = {}
    ml_off_daily: list[pd.Series] = []
    raw_target_daily: list[pd.Series] = []

    for year in YEARS:
        dates = frame.index[(frame.index >= f"{year}-01-01") & (frame.index <= f"{year}-12-31")]
        baseline_result, baseline_summary, costs = evaluator.simulate(
            frame, history, app, base, dates
        )
        ml_off_returns = baseline_result["daily_returns"].astype(float)
        ml_off_daily.append(pd.Series(ml_off_returns.to_numpy(), index=dates))
        raw_model, _raw_directory, _raw_provenance = evaluator.train_candidate(
            training, frame, app.v2, year, code_hash
        )
        raw_decisions = evaluator.apply_candidate(base, dates, frame, history, raw_model)
        raw_result, raw_summary, _ = evaluator.simulate(
            frame, history, app, raw_decisions, dates
        )
        raw_returns = raw_result["daily_returns"].astype(float)
        raw_target_daily.append(pd.Series(raw_returns.to_numpy(), index=dates))
        for scale in RESIDUAL_SCALES:
            model = _make_fold_model(
                training, frame, app.v2, year, scale, code_hash, plan_hash
            )
            candidate = evaluator.apply_candidate(base, dates, frame, history, model)
            candidate_result, candidate_summary, _ = evaluator.simulate(
                frame, history, app, candidate, dates
            )
            candidate_returns = candidate_result["daily_returns"].astype(float)
            results[str(scale)].append(
                {
                    "year": year,
                    "train_end": model.metadata["train_end"],
                    "train_rows": int(model.metadata["training_rows"]),
                    "ml_off_reference": baseline_summary,
                    "raw_target_baseline": raw_summary,
                    "candidate": candidate_summary,
                    "paired_bootstrap": evaluator.block_interval(
                        (candidate_returns - raw_returns).to_numpy()
                    ),
                    "audit_status_counts": {
                        "numerical": candidate_summary["numerical_audit_counts"],
                        "leakage": candidate_summary["leakage_audit_counts"],
                    },
                }
            )
            daily.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "year": year,
                        "ml_off_net": ml_off_returns.to_numpy(),
                        "raw_target_net": raw_returns.to_numpy(),
                        "candidate_net": candidate_returns.to_numpy(),
                        "paired_net_delta_vs_raw_target": candidate_returns.to_numpy()
                        - raw_returns.to_numpy(),
                        "paired_net_delta_vs_ml_off": candidate_returns.to_numpy()
                        - ml_off_returns.to_numpy(),
                        "residual_scale": scale,
                    }
                )
            )
            print(
                json.dumps(
                    {
                        "year": year,
                        "residual_scale": scale,
                        "ml_off_sharpe": baseline_summary["net_sharpe"],
                        "raw_target_sharpe": raw_summary["net_sharpe"],
                        "candidate_sharpe": candidate_summary["net_sharpe"],
                        "candidate_max_drawdown": candidate_summary["max_drawdown"],
                    }
                ),
                flush=True,
            )

    ml_off = pd.concat(ml_off_daily).sort_index()
    raw_target = pd.concat(raw_target_daily).sort_index()
    trial_sharpes["raw_target"] = _annual_sharpe(raw_target.to_numpy(dtype=float))
    summary_variants: dict[str, Any] = {}
    for scale in RESIDUAL_SCALES:
        selected = pd.concat(
            [entry for entry in daily if float(entry["residual_scale"].iloc[0]) == scale],
            ignore_index=True,
        ).sort_values("date")
        values = selected["candidate_net"].to_numpy(dtype=float)
        paired = selected["paired_net_delta_vs_raw_target"].to_numpy(dtype=float)
        wealth = np.cumprod(1.0 + values)
        drawdown = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1.0
        sharpe = _annual_sharpe(values)
        trial_sharpes[f"blpx_residual_{scale:.1f}"] = sharpe
        bootstrap = evaluator.block_interval(paired)
        summary_variants[f"{scale:.1f}"] = {
            "days": int(len(selected)),
            "net_sharpe": sharpe,
            "raw_target_net_sharpe": trial_sharpes["raw_target"],
            "ml_off_net_sharpe": _annual_sharpe(ml_off.to_numpy(dtype=float)),
            "net_cumulative_return": float(np.prod(1.0 + values) - 1.0),
            "raw_target_net_cumulative_return": float(np.prod(1.0 + raw_target.to_numpy()) - 1.0),
            "ml_off_net_cumulative_return": float(np.prod(1.0 + ml_off.to_numpy()) - 1.0),
            "max_drawdown": float(np.min(drawdown)),
            "mean_turnover": float(np.mean([x["candidate"]["daily_turnover_mean_raw_weight_units"] for x in results[f"{scale:.1f}"]])),
            "paired_mean_daily_net_delta": float(np.mean(paired)),
            "paired_20_day_bootstrap": bootstrap,
            "folds": results[f"{scale:.1f}"],
        }

    best_key = max(summary_variants, key=lambda key: summary_variants[key]["net_sharpe"])
    best_values = pd.concat(
        [entry for entry in daily if float(entry["residual_scale"].iloc[0]) == float(best_key)],
        ignore_index=True,
    ).sort_values("date")["candidate_net"].to_numpy(dtype=float)
    from leadlag.experiment_registry import compute_deflated_sharpe

    nominal_dsr = compute_deflated_sharpe(
        {
            "net_sharpe": summary_variants[best_key]["net_sharpe"],
            "trials": STUDY_TRIALS,
            "n_observations": int(len(best_values)),
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": int(ANNUALIZATION_DAYS),
            "returns": best_values.tolist(),
            "trial_sharpes": list(trial_sharpes.values()),
        }
    )
    historical_reject = all(
        item["net_sharpe"] < trial_sharpes["raw_target"]
        and item["paired_20_day_bootstrap"]["mean_daily_difference_ci95"][1] < 0.0
        for item in summary_variants.values()
    )
    summary = {
        "study": "ml_overlay_blpx_residual_target",
        "status": (
            "historical_candidate_rejected_forward_gate_pending"
            if historical_reject
            else "historical_diagnostic_pending_forward_oos"
        ),
        "candidate_decision": "rejected" if historical_reject else "pending",
        "candidate_decision_basis": (
            "残差target 3候補すべてでraw targetよりnet Sharpeが低く、"
            "paired 95%平均net差区間の上限も0未満"
            if historical_reject
            else "retrospective diagnostic is inconclusive; forward gate remains pending"
        ),
        "period": {"start": "2020-01-01", "end": "2024-12-31", "years": list(YEARS)},
        "annualization_days": int(ANNUALIZATION_DAYS),
        "purge_trading_days": PURGE_TRADING_DAYS,
        "historically_used_periods": True,
        "independent_oos": False,
        "production_changed": False,
        "active_artifact_changed": False,
        "candidate_parameterizations_including_raw_target": STUDY_TRIALS,
        "evaluation_plan_sha256": plan_hash,
        "source_code_sha256": code_hash,
        "residual_scales": list(RESIDUAL_SCALES),
        "trial_sharpes": trial_sharpes,
        "best_residual_scale_by_net_sharpe": best_key,
        "nominal_within_study_dsr": nominal_dsr,
        "dsr_limit": "does not account for prior related ML experiments or repeated viewing of these years",
        "historical_provider_available_at_proven": False,
        "forward_labels_after_2025_12_30": 0,
        "variants": summary_variants,
        "limitations": [
            "The five calendar-year folds are retrospective diagnostics over periods already inspected.",
            "Historical provider available_at timestamps remain unproven.",
            "The 09:10 historical input is not a frozen observed quote for every date.",
            "Costs are model estimates; actual inventory, order-level fills, and complete fee reconciliation are unavailable.",
            "This target improves the relative allocator forecast only; it is not a trade/no-trade gate.",
        ],
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    _write_json(REPORT / "summary.json", summary)
    _write_json(WORK / "summary.json", summary)
    daily_frame = pd.concat(daily, ignore_index=True).sort_values(["residual_scale", "date"])
    daily_frame.to_csv(REPORT / "daily_paired.csv", index=False)
    _write_json(REPORT / "trial_sharpes.json", trial_sharpes)

    registry = ExperimentRegistry(ROOT / "var/experiments/registry.jsonl")
    records = []
    for key, metrics in summary_variants.items():
        record = ExperimentRecord(
            name="ml_overlay_blpx_residual_target",
            hypothesis=(
                "Predict score-directional return residuals relative to the same-day "
                "BLPX expected return and use them only for within-side allocation."
            ),
            parameters={
                "target_type": "blpx_residual",
                "baseline_residual_scale": float(key),
                "training_period": "2015-01-05 through fold cutoff",
                "evaluation_years": list(YEARS),
                "purge_trading_days": PURGE_TRADING_DAYS,
                "candidate_parameterizations_including_raw_target": STUDY_TRIALS,
                "lgbm_kwargs": DEFAULT_LGBM_KWARGS,
                "report": "reports/20260929_ml_overlay_blpx_residual/report.md",
            },
            metrics={
                "net_sharpe": metrics["net_sharpe"],
                "net_sharpe_frequency": "annual",
                "trading_days_per_year": int(ANNUALIZATION_DAYS),
                "n_observations": metrics["days"],
                "trials": STUDY_TRIALS,
                "trial_sharpes": list(trial_sharpes.values()),
                "returns": daily_frame.loc[
                    daily_frame["residual_scale"] == float(key), "candidate_net"
                ].tolist(),
                "paired_mean_daily_net_delta": metrics["paired_mean_daily_net_delta"],
                "max_drawdown": metrics["max_drawdown"],
                "nominal_within_study_dsr": nominal_dsr,
                "independent_oos": False,
                "historical_provider_available_at_proven": False,
            },
            start_time=study_start,
            end_time=datetime.now(UTC),
            decision=(Decision.REJECTED if historical_reject else Decision.PENDING),
            report_path="reports/20260929_ml_overlay_blpx_residual/report.md",
        )
        registry.record(record)
        records.append(record.record_id)
    summary["registry_record_ids"] = records
    _write_json(REPORT / "summary.json", summary)
    return summary


def _finalize_existing() -> dict[str, Any]:
    summary_path = REPORT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_sharpe = float(summary["trial_sharpes"]["raw_target"])
    historical_reject = all(
        float(item["net_sharpe"]) < raw_sharpe
        and float(item["paired_20_day_bootstrap"]["mean_daily_difference_ci95"][1]) < 0.0
        for item in summary["variants"].values()
    )
    if not historical_reject:
        raise ValueError("Stored results do not meet the historical diagnostic rejection rule")
    basis = (
        "残差target 3候補すべてでraw targetよりnet Sharpeが低く、"
        "paired 95%平均net差区間の上限も0未満"
    )
    summary["status"] = "historical_candidate_rejected_forward_gate_pending"
    summary["candidate_decision"] = "rejected"
    summary["candidate_decision_basis"] = basis

    registry = ExperimentRegistry(ROOT / "var/experiments/registry.jsonl")
    existing = list(registry)
    ids: list[str] = []
    now = datetime.now(UTC)
    for old_id in summary.get("registry_record_ids", []):
        old = next((record for record in existing if record.record_id == old_id), None)
        if old is None:
            raise ValueError(f"Original pending experiment record is missing: {old_id}")
        prior_correction = next(
            (
                record
                for record in existing
                if old_id in record.supersedes
                and record.name == old.name
                and record.parameters.get("baseline_residual_scale")
                == old.parameters.get("baseline_residual_scale")
            ),
            None,
        )
        if prior_correction is not None:
            ids.append(prior_correction.record_id)
            continue
        corrected_metrics = dict(old.metrics)
        corrected_metrics.update(
            {
                "candidate_decision": "rejected",
                "candidate_decision_basis": basis,
                "forward_oos_pending": True,
            }
        )
        corrected = ExperimentRecord(
            name=old.name,
            hypothesis=old.hypothesis,
            start_time=now,
            end_time=now,
            parameters={
                **old.parameters,
                "correction_reason": "study-level historical diagnostic decision",
            },
            metrics=corrected_metrics,
            decision=Decision.REJECTED,
            report_path=old.report_path,
            related_records=[old_id],
            correction_of=old_id,
            supersedes=[old_id],
        )
        registry.record(corrected)
        ids.append(corrected.record_id)
    summary["registry_correction_record_ids"] = ids
    _write_json(REPORT / "summary.json", summary)
    _write_json(WORK / "summary.json", summary)
    return summary


def _render_report(summary: dict[str, Any]) -> str:
    rows = [
        "| BLPX residual scale | Net Sharpe | Raw-target ML Sharpe | ML-off Sharpe | Compounded net | Raw-target compounded net | Max DD | Mean daily paired net difference vs raw target | Paired 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, item in summary["variants"].items():
        ci = item["paired_20_day_bootstrap"]["mean_daily_difference_ci95"]
        rows.append(
            f"| {scale} | {item['net_sharpe']:.3f} | {item['raw_target_net_sharpe']:.3f} | "
            f"{item['ml_off_net_sharpe']:.3f} | {item['net_cumulative_return']:.2%} | "
            f"{item['raw_target_net_cumulative_return']:.2%} | "
            f"{item['max_drawdown']:.2%} | {item['paired_mean_daily_net_delta']:.6%} | "
            f"[{ci[0]:.6%}, {ci[1]:.6%}] |"
        )
    return "\n".join(
        [
            "# BLPX residual target ML overlay diagnostic",
            "",
            f"判定: **残差target候補は既知期間診断で{('不採用' if summary['candidate_decision'] == 'rejected' else '保留')}。前向き採否gateは未判定で、本番artifact/configは変更していない。**",
            "",
            "## 仮説と教師target",
            "",
            "MLに既存BLPX予測そのものを再学習させず、既存の方向付き予測から残る誤差を学ばせる。日t、銘柄jの教師は `sign(score[t,j]) × realized_return[t,j] − scale × sign(score[t,j]) × mu_gap[t,j]`。既存学習labelの固定10bpsは復元し、これは配分予測専用で注文費用ゲートとは分離する。主比較は同じfoldで再学習した従来raw-target ML、ML無効BLPXは参照値。scaleは0.8/1.0/1.2の±20%感度。",
            "",
            "## 固定した評価",
            "",
            "- 既存の監査済み2026-09-27学習行を再利用。年ごとに2020〜2024のexpanding walk-forwardを実施し、評価年直前5取引日をpurge。特徴量・LightGBM設定・コスト計算は固定。",
            "- 年・銘柄行は過去に閲覧済みであり、独立fresh OOSではない。20日block bootstrapは既存evaluatorと同じ1,000回・seed 42。DSRは当study内のraw-target ML+3候補だけを使う名目値で、ML無効BLPXは参照であり、過去の関連ML試行を完全には補正しない。",
            "- 記録先: `reports/20260929_ml_overlay_blpx_residual/daily_paired.csv` と `var/experiments/registry.jsonl`。",
            "",
            "## 結果",
            "",
            *rows,
            "",
            f"名目within-study DSR: {summary['nominal_within_study_dsr']!r}。これはSharpeの帰無仮説に対する名目値でraw targetより優れる確率ではない。試行分散と探索履歴が不足するため、採用判断用DSRではない。",
            "",
            f"歴史診断の判定根拠: {summary['candidate_decision_basis']}。",
            "",
            "## 判定と制約",
            "",
            "この実装は相対配分向けの教師targetを追加したもので、MLに取引確率や総gross制御を持たせない。歴史provider `available_at`は未証明、09:10の凍結quote・完全な実約定費用・実口座建玉も揃っていないため、結果が良くても本番候補の有効性は未確定。現在の2025-12-30 cutoff artifact以降の完全な前向き250日paired labelsは0日であり、本targetのproduction training/artifact publicationも行っていない。",
            "",
            "次段階は、同一artifact・同一入力の前向きpaired記録を蓄積し、必要な口座・注文cost正本が揃った後にincremental trade-value gateを別実験する。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Apply the study-level decision and append superseding registry corrections without retraining.",
    )
    args = parser.parse_args()
    if args.finalize_existing:
        summary = _finalize_existing()
    else:
        summary = _evaluate()
    (REPORT / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "candidate_decision": summary["candidate_decision"],
                "best_residual_scale_by_net_sharpe": summary[
                    "best_residual_scale_by_net_sharpe"
                ],
                "nominal_within_study_dsr": summary["nominal_within_study_dsr"],
                "production_changed": summary["production_changed"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
