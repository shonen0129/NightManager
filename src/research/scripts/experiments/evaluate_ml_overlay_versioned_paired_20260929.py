#!/usr/bin/env python3
"""Run a reproducible 269-day, version-aware ML-on/off paired replay.

This is a retrospective modeled comparison, not the preregistered prospective
250-day production gate. Each date uses the train-end-safe artifact assigned
to it by the production HISTORY manifest. The ML-off control changes only the
overlay-enabled flag and receives the same historical input snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.frozen import safe_config_copy
from leadlag.data import macro as macro_data
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution import var_inputs
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.var_history import _build_var_historical_inputs, _load_overlay_version
from leadlag.experiment_registry import Decision, ExperimentRecord, ExperimentRegistry
from leadlag.models.ml_overlay_artifact import load_overlay_model
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint

PRODUCTION_ROOT = ROOT / "models/ml_order_overlay/production_20260923"
PRODUCTION_CONFIG = ROOT / "configs/production/production.yaml"
SOURCE_HISTORY = ROOT / "reports/20260927_ml_overlay_var_history/history.json"
SOURCE_RETURNS = ROOT / "var/results/20260927_ml_overlay_var_history_2025/history/versioned_var_returns.pkl"
MACRO_PROVENANCE = ROOT / "var/market_data/macro_prices_verified.provenance.json"
OUTPUT_ROOT = ROOT / "var/results/20260929_ml_overlay_paired_269"
REPORT_ROOT = ROOT / "reports/20260929_ml_overlay_paired_269"
REGISTRY_PATH = ROOT / "var/experiments/registry.jsonl"
STUDY_ID = "ml-overlay-versioned-paired-replay-2026-09-29"
OLD_VERSION = "20260922T011723828589Z-7e1a81ab2617"
CURRENT_VERSION = "20260926T192555935698Z-ee306a32f3ec"
EXPECTED_DAYS = 269
BOOTSTRAP_BLOCK_DAYS = 20
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20260924
ANNUALIZATION_DAYS = 252.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_metadata(version: str) -> dict[str, Any]:
    path = PRODUCTION_ROOT / "versions" / version / "metadata.json"
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if value.get("artifact_version") != version:
        raise ValueError(f"Artifact version mismatch in {path}")
    return value


def _metric_block(returns: pd.Series, weights: pd.DataFrame, result: dict[str, Any]) -> dict[str, Any]:
    values = returns.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Return series must contain at least two finite observations")
    std = float(np.std(values, ddof=1))
    equity = pd.Series(np.concatenate(([1.0], np.cumprod(1.0 + values))))
    drawdown = equity / equity.cummax() - 1.0
    return {
        "observations": len(values),
        "cumulative_net_return": float(equity.iloc[-1] - 1.0),
        "mean_daily_net_return": float(np.mean(values)),
        "annualized_net_sharpe": (
            float(np.mean(values) / std * np.sqrt(ANNUALIZATION_DAYS)) if std > 1e-12 else None
        ),
        "max_drawdown": float(drawdown.min()),
        "gross_return_sum": float(result["daily_returns_gross"].sum()),
        "net_return_sum": float(returns.sum()),
        "cost_sums": {
            key.removeprefix("daily_"): float(result[key].sum())
            for key in (
                "daily_slip_costs",
                "daily_financing_costs",
                "daily_borrow_costs",
                "daily_reverse_costs",
                "daily_costs",
            )
        },
        "average_daily_turnover": float(result["daily_turnover"].mean()),
        "total_daily_turnover": float(result["daily_turnover"].sum()),
        "fallback_days": int(result["daily_fallback"].sum()),
        "fallback_rate": float(result["daily_fallback"].mean()),
        "max_model_gross": float(weights.abs().sum(axis=1).max()),
        "max_abs_model_net": float(weights.sum(axis=1).abs().max()),
        "side_leverage": float(result["side_leverage"]),
    }


def _slice_result(result: dict[str, Any], dates: pd.DatetimeIndex) -> dict[str, Any]:
    return {
        key: value.loc[dates] if isinstance(value, (pd.Series, pd.DataFrame)) else value
        for key, value in result.items()
    }


def _block_bootstrap_mean_ci(delta: pd.Series) -> list[float]:
    values = delta.to_numpy(dtype=float)
    if len(values) < BOOTSTRAP_BLOCK_DAYS:
        raise ValueError("Not enough observations for the paired block bootstrap")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks_per_sample = int(np.ceil(len(values) / BOOTSTRAP_BLOCK_DAYS))
    eligible_starts = len(values) - BOOTSTRAP_BLOCK_DAYS + 1
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for sample in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, eligible_starts, blocks_per_sample)
        indices = np.concatenate(
            [np.arange(start, start + BOOTSTRAP_BLOCK_DAYS) for start in starts]
        )[: len(values)]
        means[sample] = float(values[indices].mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _audit_rows(result: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries = result["v2_summaries"]
    weights = result["weights"]
    for date, summary, fallback, weight in zip(
        result["daily_returns"].index,
        summaries,
        result["daily_fallback"].to_numpy(dtype=bool),
        weights.to_numpy(dtype=float),
    ):
        audit = summary.get("audit_status", {})
        rows.append(
            {
                "trade_date": date.strftime("%Y-%m-%d"),
                "run": label,
                "numerical": audit.get("numerical"),
                "leakage": audit.get("leakage"),
                "fallback": bool(fallback or audit.get("fallback", False)),
                "overlay_applied": bool(summary.get("overlay_applied", False)),
                "model_gross": float(np.abs(weight).sum()),
                "model_net": float(weight.sum()),
            }
        )
    return rows


def _run_chunk(
    app: Any,
    frame: pd.DataFrame,
    gap_snapshot: Path,
    history_inputs: Any,
    dates: pd.DatetimeIndex,
    *,
    overlay_model: Any | None,
) -> dict[str, Any]:
    result = BacktestEngine.run_v2_backtest(
        cfg=app,
        gap_input_dir=gap_snapshot,
        df_exec=frame,
        start_date=str(dates.min().date()),
        end_date=str(dates.max().date()),
        n_jobs=4,
        overlay_model=overlay_model,
        historical_inputs=history_inputs,
    )
    if not result["daily_returns"].index.equals(dates):
        raise ValueError("Backtest returned a different trade-date index")
    if result["daily_fallback"].isna().any() or result["daily_returns"].isna().any():
        raise ValueError("Backtest returned missing fallback or return values")
    if not np.isfinite(result["daily_returns"].to_numpy(dtype=float)).all():
        raise ValueError("Backtest returned non-finite net returns")
    if not np.allclose(
        result["daily_returns_gross"] - result["daily_costs"],
        result["daily_returns"],
        atol=1e-10,
        rtol=1e-9,
    ):
        raise ValueError("Daily gross - costs = net identity failed")
    weights = result["weights"]
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Backtest returned non-finite weights")
    if float(weights.abs().sum(axis=1).max()) > 2.0 + 1e-6:
        raise ValueError("Model gross exposure exceeded 2.0")
    if float(weights.sum(axis=1).abs().max()) > 0.05 + 1e-6:
        raise ValueError("Model net exposure exceeded +/-0.05")
    return result


def _write_report(summary: dict[str, Any]) -> None:
    all_pair = summary["paired"]["all_269_days"]
    old_pair = summary["paired"]["prior_artifact_99_days"]
    new_pair = summary["paired"]["cutoff_artifact_170_days"]
    on = all_pair["ml_on"]
    off = all_pair["ml_off"]
    delta = all_pair["net_delta_on_minus_off"]
    report = f"""# Version-aware ML overlay 269日 paired replay

実施日: 2026-09-29
判定: **回顧的なモデル費用ベース比較。前向き250日gateの判定には使用しない**

## 比較設計

- 期間: {summary['period']['start']}〜{summary['period']['end']}、{summary['period']['days']}営業日。全日を同一日付pairedで比較。
- ML有効: 2025-12-30まではartifact `{summary['artifacts']['prior']['version']}`（train end {summary['artifacts']['prior']['train_end']}）、2025-12-31以降はartifact `{summary['artifacts']['cutoff']['version']}`（train end {summary['artifacts']['cutoff']['train_end']}）。各予測日のtrain endより後のartifactだけを使用。
- ML無効: 同一 `DecisionInputs`、gap snapshot、cost/configから `ml_overlay_enabled` だけをfalseにした対照。
- 片道slippage {summary['cost_config']['slippage_bps_per_side']}bps、financing / borrow / reverseを含む本番モデル費用。実約定・口座PnLではない。
- 年率Sharpeは252日換算。95%区間は日次net差の20日non-circular moving-block bootstrap、{BOOTSTRAP_SAMPLES:,}回、seed {BOOTSTRAP_SEED}。

## 全269日

| 指標 | ML有効 | ML無効 | 差 (有効−無効) |
|---|---:|---:|---:|
| 複利net return | {on['cumulative_net_return']:.4%} | {off['cumulative_net_return']:.4%} | {(on['cumulative_net_return'] - off['cumulative_net_return']):+.4%} |
| 年率net Sharpe | {on['annualized_net_sharpe']:.3f} | {off['annualized_net_sharpe']:.3f} | {(on['annualized_net_sharpe'] - off['annualized_net_sharpe']):+.3f} |
| 最大DD | {on['max_drawdown']:.4%} | {off['max_drawdown']:.4%} | {(on['max_drawdown'] - off['max_drawdown']):+.4%} |
| 平均日次turnover | {on['average_daily_turnover']:.4f} | {off['average_daily_turnover']:.4f} | {(on['average_daily_turnover'] - off['average_daily_turnover']):+.4f} |
| fallback日 | {on['fallback_days']} | {off['fallback_days']} | {on['fallback_days'] - off['fallback_days']} |
| gross cost合計 | {on['cost_sums']['costs']:.4%} | {off['cost_sums']['costs']:.4%} | {(on['cost_sums']['costs'] - off['cost_sums']['costs']):+.4%} |
| 日次net差の平均 | — | — | {delta['mean_daily']:+.6%} |
| 日次net差の95% block-bootstrap区間 | — | — | [{delta['mean_daily_ci95'][0]:+.6%}, {delta['mean_daily_ci95'][1]:+.6%}] |

### artifact区間別

| 区間 | 日数 | ML有効複利net | ML無効複利net | 差 | 平均日次net差 |
|---|---:|---:|---:|---:|---:|
| 旧artifact | {old_pair['observations']} | {old_pair['ml_on']['cumulative_net_return']:.4%} | {old_pair['ml_off']['cumulative_net_return']:.4%} | {(old_pair['ml_on']['cumulative_net_return'] - old_pair['ml_off']['cumulative_net_return']):+.4%} | {old_pair['net_delta_on_minus_off']['mean_daily']:+.6%} |
| 2025年末cutoff artifact | {new_pair['observations']} | {new_pair['ml_on']['cumulative_net_return']:.4%} | {new_pair['ml_off']['cumulative_net_return']:.4%} | {(new_pair['ml_on']['cumulative_net_return'] - new_pair['ml_off']['cumulative_net_return']):+.4%} | {new_pair['net_delta_on_minus_off']['mean_daily']:+.6%} |

## 監査と限界

- paired日数: {summary['period']['days']}。同日index、入力dataframe fingerprint、gap snapshot fingerprintをML有効/無効で共有。
- ML有効: 数値監査 {summary['audits']['ml_on_numerical_passed']}/{summary['period']['days']}、リーク監査 {summary['audits']['ml_on_leakage_passed']}/{summary['period']['days']}。ML無効: 数値監査 {summary['audits']['ml_off_numerical_passed']}/{summary['period']['days']}、リーク監査 {summary['audits']['ml_off_leakage_passed']}/{summary['period']['days']}。
- overlay適用 {summary['overlay']['applied_days']}日、ADR特徴不足等でskip {summary['overlay']['skipped_days']}日。両側ともfallback {on['fallback_days']} / {off['fallback_days']}日。モデルweight制約は全日に適合。
- 今回のcutoff済みgap snapshot fingerprintは前回VaR再生記録の値と一致しない（今回 `{summary['input_identity']['gap_input_fingerprint']}`、前回 `{summary['input_identity']['previous_var_history_gap_fingerprint']}`）。本比較のML有効/無効は今回の同一snapshotを共有する。ML有効再生と前回保存returnの最大日次差は {summary['reproduction_check']['max_abs_net_return_diff_vs_prior_var_replay']:.6g}。
- macro providerの過去時点 `available_at` は未証明。09:10 midpointは凍結実quoteではなく既存df_execのhistorical inputから作ったproxyで、約定・板・feeの完全照合もない。
- 期間・入力・モデル版を見た後の回顧replayであり、新規の独立OOSでも、事前登録した現行artifact固定の前向き250日gateでもない。研究判定は **PENDING**。本番設定・CURRENT pointerは変更していない。

## 再現

`{summary['command']}`

成果物: `var/results/20260929_ml_overlay_paired_269/summary.json`、`daily_paired.csv`、`audit_rows.json`、`paired_returns.pkl`。
"""
    REPORT_ROOT.mkdir(parents=True, exist_ok=False)
    (REPORT_ROOT / "report.md").write_text(report, encoding="utf-8")


def run() -> dict[str, Any]:
    if OUTPUT_ROOT.exists() or REPORT_ROOT.exists():
        raise FileExistsError("Refusing to overwrite existing paired replay outputs")
    existing = list(ExperimentRegistry(REGISTRY_PATH))
    if any(record.study_id == STUDY_ID for record in existing):
        raise ValueError(f"Study already exists in experiment registry: {STUDY_ID}")

    source_history = json.loads(SOURCE_HISTORY.read_text(encoding="utf-8"))
    manifest = json.loads((PRODUCTION_ROOT / "HISTORY.json").read_text(encoding="utf-8"))
    if manifest.get("history_through") != source_history.get("history_end"):
        raise ValueError("Production HISTORY end differs from the audited VaR history")
    current_pointer = (PRODUCTION_ROOT / "CURRENT").read_text(encoding="utf-8").strip()
    if current_pointer != CURRENT_VERSION:
        raise ValueError("Production CURRENT changed since the paired plan was fixed")

    app = load_config_from_yaml(PRODUCTION_CONFIG, strict=True)
    if not app.v2.ml_overlay_enabled:
        raise ValueError("Resolved production config no longer enables the ML overlay")
    app_off = safe_config_copy(app)
    app_off = app_off.model_copy(
        update={"v2": app_off.v2.model_copy(update={"ml_overlay_enabled": False})}
    )
    on_cfg = app.v2.model_dump(mode="json")
    expected_off_cfg = copy.deepcopy(on_cfg)
    expected_off_cfg["ml_overlay_enabled"] = False
    if app_off.v2.model_dump(mode="json") != expected_off_cfg:
        raise ValueError("ML-off control changed more than ml_overlay_enabled")

    frame = load_df_exec_from_local_cache(max_stale_bdays=None)
    last_date = pd.Timestamp(source_history["history_end"])
    frame = frame.loc[frame.index <= last_date].copy()
    frame_hash = dataframe_fingerprint(frame)
    if frame_hash != source_history["df_exec_hash_in_run_owned_snapshot"]:
        raise ValueError("Local df_exec no longer matches the audited historical input snapshot")
    available = pd.DatetimeIndex(frame.index)
    sim_dates = available[-EXPECTED_DAYS:]
    if (
        len(sim_dates) != EXPECTED_DAYS
        or str(sim_dates.min().date()) != source_history["history_start"]
        or str(sim_dates.max().date()) != source_history["history_end"]
    ):
        raise ValueError("Local df_exec does not reproduce the 269-day audited date range")

    prior_meta = _load_metadata(OLD_VERSION)
    cutoff_meta = _load_metadata(CURRENT_VERSION)
    if prior_meta["train_end"] != "2024-12-20" or cutoff_meta["train_end"] != "2025-12-30":
        raise ValueError("Unexpected artifact train-end metadata")
    # Use the production manifest itself as the date-to-artifact mapping.
    dates_by_version: dict[str, list[pd.Timestamp]] = {OLD_VERSION: [], CURRENT_VERSION: []}
    for date in sim_dates:
        matching = [
            item
            for item in manifest["segments"]
            if pd.Timestamp(item["start_date"]) <= date
            and (item.get("end_date") is None or date <= pd.Timestamp(item["end_date"]))
        ]
        if len(matching) != 1:
            raise ValueError(f"HISTORY manifest maps {date.date()} to {len(matching)} artifacts")
        version = str(matching[0]["version"])
        if version not in dates_by_version:
            raise ValueError(f"Unexpected artifact version in HISTORY: {version}")
        if date <= pd.Timestamp(_load_metadata(version)["train_end"]):
            raise ValueError(f"In-sample artifact assignment on {date.date()}")
        dates_by_version[version].append(date)
    if len(dates_by_version[OLD_VERSION]) != 99 or len(dates_by_version[CURRENT_VERSION]) != 170:
        raise ValueError("HISTORY manifest no longer yields the planned 99/170 version split")
    gap_source = Path(app.gap_distribution_dir)
    if not gap_source.is_absolute():
        gap_source = ROOT / gap_source
    gap_snapshot, gap_hash, gap_owner = var_inputs._snapshot_gap_input(
        gap_source,
        max_trade_date=last_date,
    )
    if gap_owner is None or gap_snapshot is None:
        raise RuntimeError("Could not create an immutable run-owned gap snapshot")

    original_macro_loader = macro_data.load_macro_prices
    macro_frame_hash: str | None = None

    def _offline_macro_loader(
        start: str | None = None,
        end: str | None = None,
        period: str = "10y",
        timeout: float = 30.0,
        cache: Any | None = None,
    ) -> pd.DataFrame:
        del period, timeout, cache
        frame_value = macro_data._load_persisted_macro_prices(start, end)
        if frame_value is None:
            raise RuntimeError("Persisted macro snapshot does not cover the replay period")
        return frame_value

    macro_data.load_macro_prices = _offline_macro_loader
    try:
        history_inputs = _build_var_historical_inputs(
            frame,
            app,
            gap_snapshot,
            sim_dates,
            overlay_enabled=True,
        )
        macro_frame = history_inputs.macro_prices
        macro_frame_hash = (
            dataframe_fingerprint(macro_frame) if macro_frame is not None else None
        )
        if macro_frame is None:
            raise ValueError("Production config requires macro features, but offline snapshot is missing")

        prior_model = _load_overlay_version(PRODUCTION_ROOT, OLD_VERSION)
        cutoff_model = load_overlay_model(PRODUCTION_ROOT)
        if prior_model.metadata["artifact_version"] != OLD_VERSION:
            raise ValueError("Loaded prior artifact does not match production HISTORY")
        if cutoff_model.metadata["artifact_version"] != CURRENT_VERSION:
            raise ValueError("Loaded current artifact does not match production CURRENT")

        on_parts: list[dict[str, Any]] = []
        on_audits: list[dict[str, Any]] = []
        model_by_version = {OLD_VERSION: prior_model, CURRENT_VERSION: cutoff_model}
        for version in (OLD_VERSION, CURRENT_VERSION):
            dates = pd.DatetimeIndex(dates_by_version[version])
            part = _run_chunk(
                app,
                frame,
                gap_snapshot,
                history_inputs,
                dates,
                overlay_model=model_by_version[version],
            )
            on_parts.append(part)
            on_audits.extend(_audit_rows(part, f"ml_on:{version}"))

        off_result = _run_chunk(
            app_off,
            frame,
            gap_snapshot,
            history_inputs,
            sim_dates,
            overlay_model=None,
        )
        off_audits = _audit_rows(off_result, "ml_off")
    finally:
        macro_data.load_macro_prices = original_macro_loader
        gap_owner.cleanup()

    on_result: dict[str, Any] = {}
    for key in (
        "daily_returns",
        "daily_returns_gross",
        "daily_costs",
        "daily_slip_costs",
        "daily_financing_costs",
        "daily_borrow_costs",
        "daily_reverse_costs",
        "daily_turnover",
        "daily_fallback",
        "weights",
    ):
        on_result[key] = pd.concat([part[key] for part in on_parts]).sort_index()
    on_result["side_leverage"] = on_parts[-1]["side_leverage"]
    on_audit_frame = pd.DataFrame(on_audits).sort_values("trade_date")
    off_audit_frame = pd.DataFrame(off_audits).sort_values("trade_date")
    if len(on_result["daily_returns"]) != EXPECTED_DAYS:
        raise ValueError("ML-on did not produce all 269 planned days")
    if not on_result["daily_returns"].index.equals(off_result["daily_returns"].index):
        raise ValueError("ML-on and ML-off dates do not pair exactly")
    if on_audit_frame[["numerical", "leakage"]].isna().any().any():
        raise ValueError("ML-on audit status missing on one or more paired dates")
    if off_audit_frame[["numerical", "leakage"]].isna().any().any():
        raise ValueError("ML-off audit status missing on one or more paired dates")
    if not (on_audit_frame["numerical"] == "PASSED").all() or not (
        on_audit_frame["leakage"] == "PASSED"
    ).all():
        raise ValueError("ML-on numerical or leakage audit failed")
    if not (off_audit_frame["numerical"] == "PASSED").all() or not (
        off_audit_frame["leakage"] == "PASSED"
    ).all():
        raise ValueError("ML-off numerical or leakage audit failed")

    previous = pickle.loads(SOURCE_RETURNS.read_bytes())
    source_series = previous["daily_returns"].sort_index()
    replay_series = on_result["daily_returns"].sort_index()
    if not source_series.index.equals(replay_series.index):
        raise ValueError("ML-on replay date range differs from the previous VaR history")
    prior_history_max_abs_diff = float(
        np.max(
            np.abs(
                source_series.to_numpy(dtype=float) - replay_series.to_numpy(dtype=float)
            )
        )
    )

    delta = on_result["daily_returns"] - off_result["daily_returns"]
    segments = {
        "prior_artifact_99_days": pd.DatetimeIndex(dates_by_version[OLD_VERSION]),
        "cutoff_artifact_170_days": pd.DatetimeIndex(dates_by_version[CURRENT_VERSION]),
    }

    def pair_metrics(dates: pd.DatetimeIndex, with_bootstrap: bool) -> dict[str, Any]:
        on_slice = on_result["daily_returns"].loc[dates]
        off_slice = off_result["daily_returns"].loc[dates]
        delta_slice = on_slice - off_slice
        result: dict[str, Any] = {
            "observations": len(dates),
            "ml_on": _metric_block(
                on_slice,
                on_result["weights"].loc[dates],
                _slice_result(on_result, dates),
            ),
            "ml_off": _metric_block(
                off_slice,
                off_result["weights"].loc[dates],
                _slice_result(off_result, dates),
            ),
            "net_delta_on_minus_off": {
                "sum_daily_differences": float(delta_slice.sum()),
                "mean_daily": float(delta_slice.mean()),
                "mean_daily_ci95": _block_bootstrap_mean_ci(delta_slice) if with_bootstrap else None,
                "positive_delta_days": int((delta_slice > 0.0).sum()),
                "negative_delta_days": int((delta_slice < 0.0).sum()),
                "zero_delta_days": int((delta_slice == 0.0).sum()),
            },
        }
        return result

    all_pair = pair_metrics(sim_dates, with_bootstrap=True)
    old_pair = pair_metrics(segments["prior_artifact_99_days"], with_bootstrap=False)
    new_pair = pair_metrics(segments["cutoff_artifact_170_days"], with_bootstrap=False)
    overlay_applied_days = int(on_audit_frame["overlay_applied"].sum())
    audits = {
        "ml_on_numerical_passed": int((on_audit_frame["numerical"] == "PASSED").sum()),
        "ml_on_leakage_passed": int((on_audit_frame["leakage"] == "PASSED").sum()),
        "ml_off_numerical_passed": int((off_audit_frame["numerical"] == "PASSED").sum()),
        "ml_off_leakage_passed": int((off_audit_frame["leakage"] == "PASSED").sum()),
    }
    summary = {
        "status": "RETROSPECTIVE_VERSIONED_PAIRED_REPLAY_PENDING_PIT_PROVIDER_AND_EXECUTION_VALIDATION",
        "study_id": STUDY_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "period": {
            "start": str(sim_dates.min().date()),
            "end": str(sim_dates.max().date()),
            "days": len(sim_dates),
            "old_artifact_days": len(segments["prior_artifact_99_days"]),
            "cutoff_artifact_days": len(segments["cutoff_artifact_170_days"]),
        },
        "artifacts": {
            "prior": {
                "version": OLD_VERSION,
                "train_end": prior_meta["train_end"],
                "model_sha256": prior_meta["model_sha256"],
            },
            "cutoff": {
                "version": CURRENT_VERSION,
                "train_end": cutoff_meta["train_end"],
                "model_sha256": cutoff_meta["model_sha256"],
            },
        },
        "input_identity": {
            "df_exec_fingerprint": frame_hash,
            "gap_input_fingerprint": gap_hash,
            "previous_var_history_gap_fingerprint": source_history["gap_input_fingerprint"],
            "matches_previous_var_history_gap_snapshot": (
                gap_hash == source_history["gap_input_fingerprint"]
            ),
            "macro_prices_fingerprint": macro_frame_hash,
            "macro_snapshot_provenance_sha256": _sha256(MACRO_PROVENANCE),
            "historical_provider_available_at_proven": False,
            "same_historical_inputs_used_for_both_sides": True,
        },
        "cost_config": {
            "slippage_bps_per_side": float(app.v2.costs.slippage_bps_per_side),
            "buy_interest_annual": float(app.v2.costs.buy_interest_annual),
            "borrow_fee_annual": float(app.v2.costs.borrow_fee_annual),
            "reverse_fee_bps": float(app.v2.costs.reverse_fee_bps),
            "overnight_alpha_long": float(app.v2.costs.overnight_alpha_long),
            "overnight_alpha_short": float(app.v2.costs.overnight_alpha_short),
            "side_leverage": float(app.v2.costs.side_leverage),
        },
        "config_fingerprints": {
            "ml_on": hashlib.sha256(json.dumps(on_cfg, sort_keys=True).encode()).hexdigest(),
            "ml_off": hashlib.sha256(
                json.dumps(app_off.v2.model_dump(mode="json"), sort_keys=True).encode()
            ).hexdigest(),
            "difference_only_ml_overlay_enabled": True,
        },
        "paired": {
            "all_269_days": all_pair,
            "prior_artifact_99_days": old_pair,
            "cutoff_artifact_170_days": new_pair,
        },
        "overlay": {
            "applied_days": overlay_applied_days,
            "skipped_days": EXPECTED_DAYS - overlay_applied_days,
            "skipped_dates": on_audit_frame.loc[~on_audit_frame["overlay_applied"], "trade_date"].tolist(),
        },
        "audits": audits,
        "reproduction_check": {
            "max_abs_net_return_diff_vs_prior_var_replay": prior_history_max_abs_diff,
            "prior_replay_source": str(SOURCE_RETURNS.relative_to(ROOT)),
        },
        "model_constraints": {
            "ml_on_max_gross": float(on_result["weights"].abs().sum(axis=1).max()),
            "ml_on_max_abs_net": float(on_result["weights"].sum(axis=1).abs().max()),
            "ml_off_max_gross": float(off_result["weights"].abs().sum(axis=1).max()),
            "ml_off_max_abs_net": float(off_result["weights"].sum(axis=1).abs().max()),
            "effective_side_leverage": float(on_result["side_leverage"]),
        },
        "limitations": [
            "Retrospective replay on already examined data; not an independent or prospective holdout.",
            "Version-aware two-artifact sequence; not 269 days of one fixed current artifact.",
            "Historical provider available_at lineage is unproven.",
            "09:10 inputs and cost model are historical proxies; no complete fills, board, fee, cash, or inventory reconciliation.",
            "Results are not an adoption decision and do not satisfy the preregistered prospective 250-day gate.",
        ],
        "command": ".venv/bin/python3 src/research/scripts/experiments/evaluate_ml_overlay_versioned_paired_20260929.py",
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    daily = pd.DataFrame(
        {
            "artifact_version": [
                OLD_VERSION if date <= pd.Timestamp("2025-12-30") else CURRENT_VERSION
                for date in sim_dates
            ],
            "ml_on_net": on_result["daily_returns"],
            "ml_off_net": off_result["daily_returns"],
            "delta_net_on_minus_off": delta,
            "ml_on_gross": on_result["daily_returns_gross"],
            "ml_off_gross": off_result["daily_returns_gross"],
            "ml_on_cost": on_result["daily_costs"],
            "ml_off_cost": off_result["daily_costs"],
            "ml_on_turnover": on_result["daily_turnover"],
            "ml_off_turnover": off_result["daily_turnover"],
            "ml_on_fallback": on_result["daily_fallback"],
            "ml_off_fallback": off_result["daily_fallback"],
            "overlay_applied": on_audit_frame.set_index("trade_date")["overlay_applied"].reindex(
                sim_dates.strftime("%Y-%m-%d")
            ).to_numpy(),
        },
        index=sim_dates,
    )
    daily.index.name = "trade_date"
    daily.to_csv(OUTPUT_ROOT / "daily_paired.csv", float_format="%.12g")
    with (OUTPUT_ROOT / "paired_returns.pkl").open("wb") as handle:
        pickle.dump(
            {
                "ml_on": on_result["daily_returns"],
                "ml_off": off_result["daily_returns"],
                "delta": delta,
                "weights_on": on_result["weights"],
                "weights_off": off_result["weights"],
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    all_audits = pd.concat([on_audit_frame, off_audit_frame], ignore_index=True)
    (OUTPUT_ROOT / "audit_rows.json").write_text(
        all_audits.to_json(orient="records", indent=2) + "\n", encoding="utf-8"
    )
    summary["artifacts_output"] = {
        "daily_paired_csv": str((OUTPUT_ROOT / "daily_paired.csv").relative_to(ROOT)),
        "paired_returns_pickle": str((OUTPUT_ROOT / "paired_returns.pkl").relative_to(ROOT)),
        "audit_rows_json": str((OUTPUT_ROOT / "audit_rows.json").relative_to(ROOT)),
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    _write_report(summary)

    registry_period = cast(dict[str, Any], summary["period"])
    registry_cost_config = cast(dict[str, Any], summary["cost_config"])
    registry_record = ExperimentRecord(
        name="ml_overlay_versioned_paired_replay_269d",
        hypothesis=(
            "A train-end-safe, version-aware historical replay produces a paired modeled net-return "
            "comparison for ML overlay on vs off across the available 269-date history."
        ),
        parameters={
            "study_id": STUDY_ID,
            "date_start": registry_period["start"],
            "date_end": registry_period["end"],
            "days": EXPECTED_DAYS,
            "prior_artifact": OLD_VERSION,
            "cutoff_artifact": CURRENT_VERSION,
            "costs": registry_cost_config,
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        metrics={
            "metric_schema_version": "ml-overlay-versioned-paired-v1",
            "n_observations": EXPECTED_DAYS,
            "net_sharpe": all_pair["ml_on"]["annualized_net_sharpe"],
            "net_sharpe_frequency": "annual",
            "trading_days_per_year": ANNUALIZATION_DAYS,
            "net_return_compounded": all_pair["ml_on"]["cumulative_net_return"],
            "ml_off_net_return_compounded": all_pair["ml_off"]["cumulative_net_return"],
            "paired_mean_daily_net_delta": all_pair["net_delta_on_minus_off"]["mean_daily"],
            "paired_mean_daily_net_delta_ci95": all_pair["net_delta_on_minus_off"]["mean_daily_ci95"],
            "historical_provider_available_at_proven": False,
            "prospective_gate_eligible": False,
        },
        decision=Decision.PENDING,
        report_path="reports/20260929_ml_overlay_paired_269/report.md",
        study_id=STUDY_ID,
        metric_schema_version="ml-overlay-versioned-paired-v1",
    )
    ExperimentRegistry(REGISTRY_PATH).record(registry_record)
    summary["experiment_registry_record_id"] = registry_record.record_id
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return summary


if __name__ == "__main__":
    run()
