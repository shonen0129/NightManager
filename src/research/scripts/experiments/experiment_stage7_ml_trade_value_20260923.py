#!/usr/bin/env python3
"""Stage 7: evaluate ML overlay value after modeled costs.

This stage consumes the fixed Stage 5 baseline and Stage 6 ML-off paired
outputs.  It does not retrain or publish an artifact.  The purpose is to
separate the overlay's gross contribution, modeled costs, and net contribution
while checking whether observed fill fees/inventory and calibration evidence
are actually available for an adoption decision.
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

from leadlag.experiment_registry import Decision
from research.experiments.ml_overlay_training import ROUND_TRIP_COST, SLIPPAGE_BPS_PER_SIDE
from research.experiment_utils import record_simple_experiment

OUTPUT_DIR = ROOT / "reports" / "20260923_profitability_order_7"
ON_DIR = ROOT / "reports" / "20260923_profitability_order_5" / "v2_current_overlay_oos"
OFF_DIR = ROOT / "reports" / "20260923_profitability_order_6" / "ml_off"
METADATA_PATH = (
    ROOT
    / "models"
    / "ml_order_overlay"
    / "production_20260923"
    / "versions"
    / "20260922T011723828589Z-7e1a81ab2617"
    / "metadata.json"
)
FILL_STATUS_PATH = ROOT / "reports" / "20260922_production_acceptance" / "historical_fill_status.json"
RECONCILIATION_PATH = (
    ROOT / "reports" / "20260922_production_acceptance" / "remaining_condition_evidence.json"
)
BLOCK_DAYS = 20
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42
ANNUALIZATION_DAYS = 252.0
WEIGHT_CHANGE_EPS = 1e-12


def _series(directory: Path, filename: str, column: str) -> pd.Series:
    frame = pd.read_csv(directory / filename, index_col=0, parse_dates=True)
    if column not in frame.columns:
        raise ValueError(f"{directory / filename} does not contain {column!r}")
    values = pd.to_numeric(frame[column], errors="raise")
    values.index = pd.DatetimeIndex(values.index).normalize()
    return values.astype(float)


def _weights(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "daily_weights.csv", index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    return frame.astype(float)


def _load_variant(directory: Path) -> dict[str, object]:
    return {
        "net": _series(directory, "daily_net_returns.csv", "net_return"),
        "gross": _series(directory, "daily_gross_returns.csv", "gross_return"),
        "cost": _series(directory, "daily_costs_total.csv", "cost"),
        "turnover": _series(directory, "daily_turnover.csv", "turnover"),
        "gross_exposure": _series(directory, "daily_gross.csv", "gross"),
        "weights": _weights(directory),
    }


def _assert_aligned(on: dict[str, object], off: dict[str, object]) -> pd.DatetimeIndex:
    on_net = on["net"]
    off_net = off["net"]
    if not isinstance(on_net, pd.Series) or not isinstance(off_net, pd.Series):
        raise TypeError("net series must be pandas Series")
    dates = on_net.index
    for variant in (on, off):
        for name, value in variant.items():
            if isinstance(value, (pd.Series, pd.DataFrame)) and not value.index.equals(dates):
                raise ValueError(f"{name} is not aligned to the canonical date index")
    if not dates.equals(off_net.index):
        raise ValueError("ML-on and ML-off dates differ")
    return dates


def _stats(values: pd.Series) -> dict[str, float | int | None]:
    values = values.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "n": int(len(values)),
        "mean_daily": mean,
        "annualized_return": mean * ANNUALIZATION_DAYS,
        "annualized_volatility": std * np.sqrt(ANNUALIZATION_DAYS),
        "annualized_sharpe": mean / std * np.sqrt(ANNUALIZATION_DAYS) if std > 1e-12 else None,
        "sum_return": float(values.sum()),
        "max_drawdown": float(drawdown.min()),
    }


def _bootstrap(delta: pd.Series) -> dict[str, object]:
    values = delta.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_blocks = int(np.ceil(len(values) / BLOCK_DAYS))
    means: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, len(values), n_blocks)
        indices = np.concatenate(
            [(start + np.arange(BLOCK_DAYS)) % len(values) for start in starts]
        )[: len(values)]
        means.append(float(np.mean(values[indices])))
    return {
        "block_days": BLOCK_DAYS,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_daily_delta": float(np.mean(values)),
        "sum_delta": float(np.sum(values)),
        "ci95_mean_daily_delta": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution_evidence() -> dict[str, object]:
    fill = _load_json(FILL_STATUS_PATH)
    reconciliation = _load_json(RECONCILIATION_PATH)
    official = reconciliation["broker_reconciliation"]["official_transaction_csv"]
    return {
        "observed_provider_available_at": False,
        "historical_fill_detail_requery_success": int(fill.get("fresh_fill_details", 0)) > 0,
        "historical_fill_detail_requery_success_count": int(fill.get("fresh_fill_details", 0)),
        "broker_requery_failed_count": int(fill.get("query_failed_records", 0)),
        "official_quantity_reconciled": int(official.get("unreconciled_local_group_count", 1)) == 0,
        "official_fee_components_present": bool(official.get("fee_components_present", False)),
        "official_transaction_source": official.get("source"),
        "actual_9_10_fill_costs_available": False,
        "actual_inventory_path_available": False,
    }


def _build_summary() -> tuple[dict[str, object], pd.DataFrame]:
    on = _load_variant(ON_DIR)
    off = _load_variant(OFF_DIR)
    dates = _assert_aligned(on, off)

    on_net = on["net"]
    off_net = off["net"]
    on_gross = on["gross"]
    off_gross = off["gross"]
    on_cost = on["cost"]
    off_cost = off["cost"]
    on_turnover = on["turnover"]
    off_turnover = off["turnover"]
    on_weights = on["weights"]
    off_weights = off["weights"]
    if not all(
        isinstance(value, pd.Series)
        for value in (on_net, off_net, on_gross, off_gross, on_cost, off_cost, on_turnover, off_turnover)
    ):
        raise TypeError("variant series have unexpected types")
    if not isinstance(on_weights, pd.DataFrame) or not isinstance(off_weights, pd.DataFrame):
        raise TypeError("variant weights have unexpected types")

    delta_net = on_net - off_net
    delta_gross = on_gross - off_gross
    delta_cost = on_cost - off_cost
    delta_turnover = on_turnover - off_turnover
    paired = pd.DataFrame(
        {
            "ml_on_net": on_net,
            "ml_off_net": off_net,
            "ml_on_gross": on_gross,
            "ml_off_gross": off_gross,
            "ml_on_cost": on_cost,
            "ml_off_cost": off_cost,
            "ml_on_turnover": on_turnover,
            "ml_off_turnover": off_turnover,
            "delta_net": delta_net,
            "delta_gross": delta_gross,
            "delta_cost": delta_cost,
            "delta_turnover": delta_turnover,
        },
        index=dates,
    )
    identity_error = float(np.max(np.abs(delta_net - (delta_gross - delta_cost))))

    weight_delta = on_weights - off_weights
    weight_l1 = weight_delta.abs().sum(axis=1)
    weight_max = weight_delta.abs().max(axis=1)
    changed = weight_max > WEIGHT_CHANGE_EPS
    on_gross_weight = on_weights.abs().sum(axis=1)
    off_gross_weight = off_weights.abs().sum(axis=1)
    on_net_weight = on_weights.sum(axis=1).abs()
    off_net_weight = off_weights.sum(axis=1).abs()
    paired["weight_l1_delta"] = weight_l1
    paired["weight_max_delta"] = weight_max

    metadata = _load_json(METADATA_PATH)
    execution = _execution_evidence()
    target_cost_contract = {
        "target_type": metadata.get("target_type"),
        "training_code_round_trip_cost": ROUND_TRIP_COST,
        "training_code_slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "target_includes_fixed_round_trip_cost": metadata.get("target_type") == "raw",
        "observed_fill_costs_used_in_target": False,
    }
    summary: dict[str, object] = {
        "stage": 7,
        "status": "PENDING_COST_CALIBRATION_AND_EXECUTION_EVIDENCE",
        "period": {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "days": int(len(dates)),
        },
        "annualization_days": ANNUALIZATION_DAYS,
        "comparison": {
            "ml_on": {
                "net": _stats(on_net),
                "gross": _stats(on_gross),
                "cost": _stats(-on_cost),
                "avg_turnover": float(on_turnover.mean()),
                "avg_gross_exposure": float(on["gross_exposure"].mean()),
            },
            "ml_off": {
                "net": _stats(off_net),
                "gross": _stats(off_gross),
                "cost": _stats(-off_cost),
                "avg_turnover": float(off_turnover.mean()),
                "avg_gross_exposure": float(off["gross_exposure"].mean()),
            },
            "paired_delta": {
                "net": _stats(delta_net),
                "gross": _stats(delta_gross),
                "cost": _stats(-delta_cost),
                "turnover": _stats(delta_turnover),
                "block_bootstrap": _bootstrap(delta_net),
                "net_identity_max_abs_error": identity_error,
            },
        },
        "inventory_proxy": {
            "actual_fills_available": execution["actual_inventory_path_available"],
            "weight_change_days": int(changed.sum()),
            "weight_change_rate": float(changed.mean()),
            "mean_l1_weight_delta": float(weight_l1.mean()),
            "max_l1_weight_delta": float(weight_l1.max()),
            "mean_max_single_ticker_delta": float(weight_max.mean()),
            "max_model_net_ml_on": float(on_net_weight.max()),
            "max_model_net_ml_off": float(off_net_weight.max()),
            "max_model_gross_ml_on": float(on_gross_weight.max()),
            "max_model_gross_ml_off": float(off_gross_weight.max()),
            "note": "Weights are a backtest proxy; they are not broker inventory or fill quantities.",
        },
        "target_cost_contract": target_cost_contract,
        "artifact": {
            "metadata_path": str(METADATA_PATH.relative_to(ROOT)),
            "artifact_version": metadata.get("artifact_version"),
            "target_type": metadata.get("target_type"),
            "train_start": metadata.get("train_start"),
            "train_end": metadata.get("train_end"),
            "label_asof_end": metadata.get("label_asof_end"),
            "historical_provider_available_at_proven": metadata.get(
                "historical_provider_available_at_proven"
            ),
            "metadata_status": metadata.get("metadata_status"),
        },
        "execution_evidence": execution,
        "production_changed": False,
        "decision": "PENDING_NOT_ADOPTED",
        "limitations": [
            "The comparison uses modeled V2 costs, not broker fee components or observed 09:10 fills.",
            "The artifact target subtracts a fixed round-trip cost; observed fill costs are not in the target.",
            "No probability calibration curve or realized fill-cost calibration sample is stored for this artifact.",
            "Weights provide a position proxy only; broker inventory and order-level quantities are unavailable for this OOS period.",
            "No new ML target, artifact, or production setting is promoted at stage 7.",
        ],
    }
    return summary, paired


def _report(summary: dict[str, object]) -> str:
    comparison = summary["comparison"]
    on = comparison["ml_on"]
    off = comparison["ml_off"]
    delta = comparison["paired_delta"]
    bootstrap = delta["block_bootstrap"]
    execution = summary["execution_evidence"]
    artifact = summary["artifact"]
    proxy = summary["inventory_proxy"]
    return "\n".join(
        [
            "# 収益改善順序7：コスト後のML取引価値",
            "",
            "発注・取消・再送を行わない、Stage 5のML有効とStage 6のML無効を同一OOSで比較する診断です。",
            "",
            f"判定: **{summary['decision']}**",
            "",
            "## 固定条件",
            "",
            f"- 期間: {summary['period']['start']} -> {summary['period']['end']} ({summary['period']['days']}日)",
            f"- 年率換算: {summary['annualization_days']:.0f}営業日",
            "- 同一のV2入力・費用モデル・全営業日で比較",
            "- ML有効: `reports/20260923_profitability_order_5/v2_current_overlay_oos`",
            "- ML無効: `reports/20260923_profitability_order_6/ml_off`",
            "",
            "## 取引価値の分解",
            "",
            "| 系列 | ML有効 | ML無効 | ML有効−無効 |",
            "|---|---:|---:|---:|",
            f"| net Sharpe | {on['net']['annualized_sharpe']:.6f} | {off['net']['annualized_sharpe']:.6f} | {on['net']['annualized_sharpe'] - off['net']['annualized_sharpe']:.6f} |",
            f"| net 合計 | {on['net']['sum_return']:.6f} | {off['net']['sum_return']:.6f} | {delta['net']['sum_return']:.6f} |",
            f"| gross 合計 | {on['gross']['sum_return']:.6f} | {off['gross']['sum_return']:.6f} | {delta['gross']['sum_return']:.6f} |",
            f"| cost 合計（負値表示） | {on['cost']['sum_return']:.6f} | {off['cost']['sum_return']:.6f} | {delta['cost']['sum_return']:.6f} |",
            f"| 平均turnover | {on['avg_turnover']:.6f} | {off['avg_turnover']:.6f} | {on['avg_turnover'] - off['avg_turnover']:.6f} |",
            "",
            f"- 日次net差分の20日block bootstrap 95%区間: [{bootstrap['ci95_mean_daily_delta'][0]:.9f}, {bootstrap['ci95_mean_daily_delta'][1]:.9f}]",
            f"- 日次net差分の平均: {bootstrap['mean_daily_delta']:.9f}、合計: {bootstrap['sum_delta']:.9f}",
            f"- `delta_net = delta_gross - delta_cost` 最大誤差: {delta['net_identity_max_abs_error']:.3e}",
            "- ML有効は総収益を増やす一方、同じOOSではML無効よりnet Sharpeがわずかに低く、費用後のリスク調整改善は証明されていない。",
            "",
            "## 在庫・約定の証拠",
            "",
            f"- 重みが変わった日: {proxy['weight_change_days']} / {summary['period']['days']}日 ({proxy['weight_change_rate']:.3%})",
            f"- 平均weight L1差分: {proxy['mean_l1_weight_delta']:.6f}、最大: {proxy['max_l1_weight_delta']:.6f}",
            "- 上記はbacktestのweight proxyであり、brokerの実在庫・注文数量・約定価格ではない。",
            f"- broker再照会成功: {execution['historical_fill_detail_requery_success_count']}件、失敗: {execution['broker_requery_failed_count']}件",
            f"- 公式CSVのfee components: {'あり' if execution['official_fee_components_present'] else 'なし'}",
            "- したがって、実約定費用・在庫を反映したML targetの校正ゲートは未充足。",
            "",
            "## ML target / artifact",
            "",
            f"- artifact: `{artifact['artifact_version']}`、target_type: `{artifact['target_type']}`、metadata: `{artifact['metadata_status']}`",
            f"- 学習期間: {artifact['train_start']} -> {artifact['train_end']}、label_asof_end: {artifact['label_asof_end']}",
            f"- 学習コード上の固定round-trip cost: {summary['target_cost_contract']['training_code_round_trip_cost']:.6f} ({summary['target_cost_contract']['training_code_slippage_bps_per_side']:.1f}bps/side)",
            "- raw targetは固定round-trip costを控除するが、観測されたbroker fee/fill costをtargetに接続していない。",
            "- p_tradeの校正曲線と実約定費用の校正サンプルはartifact/保存成果物から確認できない。",
            "",
            "## 判定と次の依存",
            "",
            "- **PENDING_NOT_ADOPTED**: 現行MLを削除・再学習・本番昇格する判断はしない。",
            "- Stage 7の数値分解は完了したが、実費・実在庫・校正証拠不足のため、収益改善順序7を経済的採用完了とは扱わない。",
            "- Stage 8（容量・集中・beta）とStage 9（新規特徴量・universe）は、Stage 2の実行証拠およびStage 3の実行可能baselineが未完了のため保留する。",
            "",
            "## 参照",
            "",
            "- `stage7_summary.json`",
            "- `daily_ml_value.csv`",
            "- `reports/20260923_profitability_order_5/report.md`",
            "- `reports/20260923_profitability_order_6/report.md`",
            "- `models/ml_order_overlay/production_20260923/versions/20260922T011723828589Z-7e1a81ab2617/metadata.json`",
            "",
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, paired = _build_summary()
    paired.to_csv(OUTPUT_DIR / "daily_ml_value.csv", index_label="trade_date")
    record = record_simple_experiment(
        name="stage7_ml_trade_value_20260923",
        hypothesis="After the same modeled costs, the current ML overlay adds robust risk-adjusted trade value and is supported by observed execution-cost calibration.",
        parameters={
            "on_dir": str(ON_DIR.relative_to(ROOT)),
            "off_dir": str(OFF_DIR.relative_to(ROOT)),
            "metadata_path": str(METADATA_PATH.relative_to(ROOT)),
            "block_days": BLOCK_DAYS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        metrics={
            "stage": 7,
            "n_observations": summary["period"]["days"],
            "paired_mean_daily_net_delta": summary["comparison"]["paired_delta"]["block_bootstrap"]["mean_daily_delta"],
            "paired_sum_net_delta": summary["comparison"]["paired_delta"]["block_bootstrap"]["sum_delta"],
            "paired_ci95_mean_daily_net_delta": summary["comparison"]["paired_delta"]["block_bootstrap"]["ci95_mean_daily_delta"],
            "net_identity_max_abs_error": summary["comparison"]["paired_delta"]["net_identity_max_abs_error"],
            "actual_fill_costs_available": summary["execution_evidence"]["actual_9_10_fill_costs_available"],
            "actual_inventory_available": summary["execution_evidence"]["actual_inventory_path_available"],
        },
        decision=Decision.PENDING,
        report_path="reports/20260923_profitability_order_7/report.md",
    )
    summary["registry_record"] = {
        "name": record.name,
        "decision": record.decision.value,
        "trials": record.metrics.get("trials"),
        "deflated_sharpe": record.metrics.get("deflated_sharpe"),
    }
    (OUTPUT_DIR / "stage7_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "report.md").write_text(_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
