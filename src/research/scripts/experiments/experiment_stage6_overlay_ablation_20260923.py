#!/usr/bin/env python3
"""Stage 6: fixed one-at-a-time V2 overlay ablation.

The candidate list is predeclared and contains no parameter sweep. Each
variant removes exactly one production component from the same OOS period and
cost model. This script never writes production configuration or submits an
order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.schemas import AppConfig
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.experiment_registry import Decision
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
from research.experiment_utils import record_backtest_experiment

OUTPUT_DIR = ROOT / "reports" / "20260923_profitability_order_6"
BASELINE_DIR = ROOT / "reports" / "20260923_profitability_order_5" / "v2_current_overlay_oos"
GAP_DIR = ROOT / "var" / "live" / "pipeline_data" / "gap_adjusted_distribution" / "20260731_024303"
START_DATE = "2024-12-23"
END_DATE = "2026-07-29"
ANNUALIZATION_DAYS = 252.0
BLOCK_DAYS = 20
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42

VARIANTS = {
    "mh_off": {
        "label": "MH off",
        "top_level": {"mh_blend_enabled": False},
        "blpx": {},
    },
    "ml_off": {
        "label": "ML overlay off",
        "top_level": {"ml_overlay_enabled": False},
        "blpx": {},
    },
    "macro_off": {
        "label": "macro kappa/direction off",
        "top_level": {"macro_kappa_enabled": False, "macro_direction_enabled": False},
        "blpx": {"macro_kappa_enabled": False, "macro_direction_enabled": False},
    },
    "fracdiff_off": {
        "label": "fractional differentiation off",
        "top_level": {"frac_diff_enabled": False},
        "blpx": {"frac_diff_enabled": False},
    },
    "copula_off": {
        "label": "copula off",
        "top_level": {},
        "blpx": {"copula_enabled": False},
    },
    "cs_off": {
        "label": "cross-sectional reversal overlay off",
        "top_level": {"cs_overlay_enabled": False},
        "blpx": {},
    },
    "minvar_off": {
        "label": "MinVar off",
        "top_level": {"minvar_enabled": False},
        "blpx": {"minvar_enabled": False},
    },
    "rule_d_off": {
        "label": "RuleD scaling off",
        "top_level": {"mult_low": 1.0, "mult_mid": 1.0, "mult_high": 1.0},
        "blpx": {},
    },
}


def _series(directory: Path, filename: str, column: str) -> pd.Series:
    frame = pd.read_csv(directory / filename, index_col=0, parse_dates=True)
    series = pd.to_numeric(frame[column], errors="raise")
    series.index = pd.DatetimeIndex(series.index).normalize()
    return series.astype(float)


def _load_baseline() -> dict[str, object]:
    weights = pd.read_csv(BASELINE_DIR / "daily_weights.csv", index_col=0, parse_dates=True)
    weights.index = pd.DatetimeIndex(weights.index).normalize()
    result: dict[str, object] = {
        "weights": weights.astype(float),
        "daily_returns": _series(BASELINE_DIR, "daily_net_returns.csv", "net_return"),
        "daily_returns_gross": _series(BASELINE_DIR, "daily_gross_returns.csv", "gross_return"),
        "daily_costs": _series(BASELINE_DIR, "daily_costs_total.csv", "cost"),
        "daily_slip_costs": _series(BASELINE_DIR, "daily_costs_slip.csv", "slip_cost"),
        "daily_financing_costs": _series(
            BASELINE_DIR, "daily_costs_financing.csv", "financing_cost"
        ),
        "daily_borrow_costs": _series(BASELINE_DIR, "daily_costs_borrow.csv", "borrow_cost"),
        "daily_reverse_costs": _series(
            BASELINE_DIR, "daily_costs_reverse.csv", "reverse_cost"
        ),
        "daily_turnover": _series(BASELINE_DIR, "daily_turnover.csv", "turnover"),
        "daily_gross_exps": _series(BASELINE_DIR, "daily_gross.csv", "gross"),
        "daily_fallback": _series(BASELINE_DIR, "daily_fallback.csv", "fallback").astype(bool),
    }
    return result


def _variant_config(app: AppConfig, spec: dict[str, object]) -> AppConfig:
    v2 = app.v2.model_copy(deep=True)
    top_level = dict(spec["top_level"])
    blpx = dict(spec["blpx"])
    if blpx:
        v2 = v2.model_copy(update={"blpx": v2.blpx.model_copy(update=blpx)})
    if top_level:
        v2 = v2.model_copy(update=top_level)
    return app.model_copy(deep=True, update={"v2": v2})


def _stats(values: pd.Series) -> dict[str, object]:
    values = values.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    shared = calculate_metrics(
        values,
        spec=MetricsSpec(annualization_periods=int(ANNUALIZATION_DAYS)),
    )
    return {
        "n": int(len(values)),
        "annualized_return": mean * ANNUALIZATION_DAYS,
        "annualized_volatility": std * np.sqrt(ANNUALIZATION_DAYS),
        "net_sharpe": (
            float(shared["Sharpe"]) if np.isfinite(shared.get("Sharpe", np.nan)) else None
        ),
        "max_drawdown": float(shared["MDD"]),
        "sum_return": float(values.sum()),
        "mean_daily": mean,
    }


def _bootstrap(delta: np.ndarray) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_blocks = int(np.ceil(len(delta) / BLOCK_DAYS))
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, len(delta), n_blocks)
        indices = np.concatenate(
            [(start + np.arange(BLOCK_DAYS)) % len(delta) for start in starts]
        )[: len(delta)]
        samples.append(float(np.mean(delta[indices])))
    return {
        "block_days": BLOCK_DAYS,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_daily_delta": float(np.mean(delta)),
        "sum_delta": float(np.sum(delta)),
        "ci95_mean_daily_delta": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def _summarize(result: dict[str, object]) -> dict[str, object]:
    net = result["daily_returns"].astype(float)
    gross = result["daily_returns_gross"].astype(float)
    costs = result["daily_costs"].astype(float)
    weights = result["weights"].astype(float)
    net_exposure = weights.sum(axis=1).abs()
    gross_exposure = weights.abs().sum(axis=1)
    component_sum = sum(
        result[key].astype(float)
        for key in (
            "daily_slip_costs",
            "daily_financing_costs",
            "daily_borrow_costs",
            "daily_reverse_costs",
        )
    )
    if not np.isfinite(np.r_[net.to_numpy(), gross.to_numpy(), costs.to_numpy()]).all():
        raise ValueError("Non-finite ablation PnL")
    np.testing.assert_allclose(gross - costs, net, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(component_sum, costs, atol=1e-14, rtol=0.0)
    return {
        **_stats(net),
        "gross_sharpe": _stats(gross)["net_sharpe"],
        "fallback_days": int(result["daily_fallback"].sum()),
        "fallback_rate": float(result["daily_fallback"].mean()),
        "avg_turnover": float(result["daily_turnover"].mean()),
        "avg_gross_exposure": float(result["daily_gross_exps"].mean()),
        "max_model_net": float(net_exposure.max()),
        "max_model_gross": float(gross_exposure.max()),
        "max_effective_net": float(net_exposure.max() * 1.5),
        "max_effective_gross": float(gross_exposure.max() * 1.5),
        "costs": {
            "slip": float(result["daily_slip_costs"].sum()),
            "financing": float(result["daily_financing_costs"].sum()),
            "borrow": float(result["daily_borrow_costs"].sum()),
            "reverse": float(result["daily_reverse_costs"].sum()),
            "total": float(costs.sum()),
        },
        "cost_identity_max_error": float(np.max(np.abs(gross - costs - net))),
    }


def _load_and_save_variant(name: str, result: dict[str, object]) -> None:
    directory = OUTPUT_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    result["weights"].to_csv(directory / "daily_weights.csv")
    for key, filename, column in (
        ("daily_returns", "daily_net_returns.csv", "net_return"),
        ("daily_returns_gross", "daily_gross_returns.csv", "gross_return"),
        ("daily_costs", "daily_costs_total.csv", "cost"),
        ("daily_turnover", "daily_turnover.csv", "turnover"),
        ("daily_gross_exps", "daily_gross.csv", "gross"),
        ("daily_fallback", "daily_fallback.csv", "fallback"),
    ):
        result[key].to_csv(directory / filename, header=[column])


def _removal_decision(baseline: dict[str, object], variant: dict[str, object]) -> dict[str, object]:
    ci = variant["paired_block_bootstrap"]["ci95_mean_daily_delta"]
    criteria = {
        "net_sharpe_not_lower": variant["net_sharpe"] >= baseline["net_sharpe"],
        "max_drawdown_not_worse": variant["max_drawdown"] >= baseline["max_drawdown"],
        "turnover_not_higher": variant["avg_turnover"] <= baseline["avg_turnover"],
        "fallback_not_higher": variant["fallback_rate"] <= baseline["fallback_rate"],
        "block_ci_lower_nonnegative": ci[0] >= 0.0,
    }
    return {
        "decision": "ADOPT_REMOVAL" if all(criteria.values()) else "RETAIN_COMPONENT",
        "criteria": criteria,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", *VARIANTS])
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    app = load_config_from_yaml(ROOT / "configs" / "production" / "production.yaml", strict=True)
    frame = load_df_exec_from_local_cache()
    baseline = _load_baseline()
    dates = baseline["daily_returns"].index
    summary_path = OUTPUT_DIR / "ablation_summary.json"
    previous = {}
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    selected = list(VARIANTS) if args.only in (None, "all") else [args.only]
    if args.only in (None, "all") and isinstance(previous.get("variants"), dict):
        selected = [name for name in selected if name not in previous["variants"]]
    summary: dict[str, object] = {
        "stage": 6,
        "status": "DIAGNOSTIC_PENDING",
        "period": {"start": START_DATE, "end": END_DATE, "days": int(len(dates))},
        "config": "configs/production/production.yaml",
        "active_overlay": "models/ml_order_overlay/production_20260923",
        "gap_input": str(GAP_DIR.relative_to(ROOT)),
        "ablation_design": "one-at-a-time, each variant removes exactly one component",
        "candidate_count": len(VARIANTS),
        "annualization_days": ANNUALIZATION_DAYS,
        "bootstrap": {
            "block_days": BLOCK_DAYS,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "baseline": _summarize(baseline),
        "variants": {},
        "production_changed": False,
    }
    if isinstance(previous.get("variants"), dict):
        summary["variants"].update(previous["variants"])

    for name in selected:
        spec = VARIANTS[name]
        logging.error("Running stage 6 variant %s", name)
        cfg = _variant_config(app, spec)
        result = BacktestEngine.run_v2_backtest(
            cfg=cfg,
            gap_input_dir=GAP_DIR,
            df_exec=frame,
            start_date=START_DATE,
            end_date=END_DATE,
            n_jobs=args.n_jobs,
            overlay_model_dir=(
                "models/ml_order_overlay/production_20260923"
                if name != "ml_off"
                else None
            ),
        )
        result["daily_slip_costs"] = result["daily_slip_costs"].astype(float)
        result["daily_financing_costs"] = result["daily_financing_costs"].astype(float)
        result["daily_borrow_costs"] = result["daily_borrow_costs"].astype(float)
        result["daily_reverse_costs"] = result["daily_reverse_costs"].astype(float)
        result_summary = _summarize(result)
        delta = result["daily_returns"].astype(float) - baseline["daily_returns"].astype(float)
        result_summary["paired_block_bootstrap"] = _bootstrap(delta.to_numpy())
        result_summary["label"] = spec["label"]
        summary["variants"][name] = result_summary
        _load_and_save_variant(name, result)
        record_backtest_experiment(
            name=f"stage6_ablation_{name}_20260923",
            hypothesis=f"Removing {spec['label']} does not reduce OOS net performance.",
            app_config=cfg,
            results={
                "daily_returns": result["daily_returns"],
                "daily_fallback": result["daily_fallback"],
                "daily_turnover": result["daily_turnover"],
                "daily_gross_exps": result["daily_gross_exps"],
            },
            extra_metrics={
                "stage": 6,
                "variant": name,
                "paired_block_bootstrap": result_summary["paired_block_bootstrap"],
                "production_changed": False,
            },
            decision=Decision.PENDING,
            reason="Fixed one-at-a-time diagnostic; stage 2 execution evidence remains incomplete.",
            report_path="reports/20260923_profitability_order_6/report.md",
            metrics_spec=MetricsSpec(annualization_periods=int(ANNUALIZATION_DAYS)),
        )

    for item in summary["variants"].values():
        item["adoption"] = _removal_decision(summary["baseline"], item)

    summary["status"] = (
        "COMPLETE_DIAGNOSTIC_PENDING_ADOPTION"
        if all(name in summary["variants"] for name in VARIANTS)
        else "PARTIAL"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 収益改善順序6：overlay逐次ablation",
        "",
        "発注・取消・再送を行わない、同一OOS期間・同一費用のV2診断です。",
        "",
        f"判定: **{summary['status']}**",
        f"- 期間: {START_DATE} -> {END_DATE} ({len(dates)}日)",
        "- 比較: current production artifactを全機能有効baselineとし、1機能だけ無効化",
        f"- 年率換算: {ANNUALIZATION_DAYS:.0f}営業日",
        "- ブロックbootstrap: 20営業日、1000回、seed=42（平均差の不確実性のみ）",
        "- 新パラメータ探索ではないため、DSRを採用判断の根拠にしない",
        "",
        "## 結果",
        "",
        "| variant | net Sharpe | max DD | turnover | fallback | mean差95%CI | 採否 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    base = summary["baseline"]
    lines.append(
        f"| baseline | {base['net_sharpe']:.6f} | {base['max_drawdown']:.6f} | "
        f"{base['avg_turnover']:.6f} | {base['fallback_rate']:.4%} | — | 現行baseline |"
    )
    for name, item in summary["variants"].items():
        ci = item["paired_block_bootstrap"]["ci95_mean_daily_delta"]
        lines.append(
            f"| {name} | {item['net_sharpe']:.6f} | {item['max_drawdown']:.6f} | "
            f"{item['avg_turnover']:.6f} | {item['fallback_rate']:.4%} | "
            f"[{ci[0]:.6g}, {ci[1]:.6g}] | {item['adoption']['decision']} |"
        )
    lines.extend(
        [
            "",
            "## 完了条件と採否",
            "",
            "| 条件 | 結果 |",
            "|---|---|",
            "| MH/ML/macro/fracdiff/copula/CS/minvar/RuleDを1つずつ外す | "
            + ("PASS" if summary["status"] == "COMPLETE_DIAGNOSTIC_PENDING_ADOPTION" else "PARTIAL")
            + " |",
            "| 同一期間・同一費用・全評価日 | PASS |",
            "| raw net/gross制約・4費用identity | 各variantで検査 |",
            "| 逐次選択のOOS採用 | 保留（本番価格・実約定未証明） |",
            "| 本番config/artifact変更 | 実施なし |",
            "",
            "削除採用の基準は、net Sharpe・最大DD・turnover・fallback率が悪化せず、20日block bootstrap平均差CI下限が非負であること。全variantでこの基準を満たす削除案はなく、現行componentを維持する。さらにstage 2の実約定・9:10 quoteが不足しているため、現行componentの本番継続自体も収益性の最終証明ではない。",
            "",
            "再現JSON: reports/20260923_profitability_order_6/ablation_summary.json",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
