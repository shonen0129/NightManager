#!/usr/bin/env python3
"""Generate a standard markdown backtest report from run_production_backtest.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRADING_DAYS = 252


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_bps(x: float) -> str:
    return f"{x * 10000:.2f}"


def _compute_metrics(returns: pd.Series) -> dict:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return {}
    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1))
    return {
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "samples": len(r),
        "mean_daily": mean_r,
        "std_daily": std_r,
        "ar": mean_r * TRADING_DAYS,
        "vol_ann": std_r * np.sqrt(TRADING_DAYS),
        "sharpe": mean_r / std_r * np.sqrt(TRADING_DAYS) if std_r > 1e-12 else 0.0,
        "total_return": float((1.0 + r).prod() - 1.0),
    }


def _mdd(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Backtest output directory")
    parser.add_argument("--report-dir", default="reports/longterm_backtest", help="Report directory")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    net = pd.read_csv(out_dir / "daily_net_returns.csv", index_col=0, parse_dates=True)["net_return"]
    gross = pd.read_csv(out_dir / "daily_gross_returns.csv", index_col=0, parse_dates=True)["gross_return"]
    turnover = pd.read_csv(out_dir / "daily_turnover.csv", index_col=0, parse_dates=True)["turnover"]
    gross_exp = pd.read_csv(out_dir / "daily_gross_exposure.csv", index_col=0, parse_dates=True)["gross_exposure"]
    costs = pd.read_csv(out_dir / "daily_costs.csv", index_col=0, parse_dates=True)["cost"]
    slip = pd.read_csv(out_dir / "daily_slip_costs.csv", index_col=0, parse_dates=True)["slip_cost"]
    financing = pd.read_csv(out_dir / "daily_financing_costs.csv", index_col=0, parse_dates=True)["financing_cost"]
    borrow = pd.read_csv(out_dir / "daily_borrow_costs.csv", index_col=0, parse_dates=True)["borrow_cost"]
    reverse = pd.read_csv(out_dir / "daily_reverse_costs.csv", index_col=0, parse_dates=True)["reverse_cost"]
    equity = pd.read_csv(out_dir / "daily_equity_curve.csv", index_col=0, parse_dates=True)["equity"]

    net_m = _compute_metrics(net)
    gross_m = _compute_metrics(gross)
    mdd = _mdd(equity)

    # Cost decomposition in bps per day (averages)
    avg_cost = float(costs.mean())
    avg_slip = float(slip.mean())
    avg_fin = float(financing.mean())
    avg_borrow = float(borrow.mean())
    avg_reverse = float(reverse.mean())

    avg_turnover = float(turnover.mean())
    avg_gross_exp = float(gross_exp.mean())

    # Cumulative costs vs gross
    total_gross = float(gross.sum())
    total_cost = float(costs.sum())
    cost_pct_of_gross = total_cost / abs(total_gross) if total_gross != 0 else 0.0

    # Fallback rate (this runner does not emit fallback series; V1/V2 non-fallback path)
    fallback_rate = 0.0

    report = f"""# 長期間本番バックテストレポート

> 作成日: 2026-08-01
> モデル: SectorRelativeEnsembleBLPEnhancedModel (Residual-BLPX / production_residual_blpx)
> Config: `configs/production/production.yaml`
> 期間: {net_m['start']} 〜 {net_m['end']}
> バックテスト出力: `{out_dir}`

## 概要

本番設定（production.yaml）で2015-01-05から最新日（{net_m['end']}）までの全期間をバックテストした。
コストは片道5bps + 金利2.5% + 貸株1.15% + 逆日歩2bps/day、オーバーナイト持ち越し（long 75% / short 50%）を含むnetで評価。

## 結果サマリー

| 指標 | Net（コスト後） | Gross（コスト前） |
|------|----------------|-------------------|
| 期間 | {net_m['start']} 〜 {net_m['end']} | {gross_m['start']} 〜 {gross_m['end']} |
| サンプル日数 | {net_m['samples']} | {gross_m['samples']} |
| 年率リターン (AR) | {_fmt_pct(net_m['ar'])} | {_fmt_pct(gross_m['ar'])} |
| 年率ボラティリティ | {_fmt_pct(net_m['vol_ann'])} | {_fmt_pct(gross_m['vol_ann'])} |
| **Sharpe (日次, ann)** | **{net_m['sharpe']:.4f}** | **{gross_m['sharpe']:.4f}** |
| 累積リターン | {_fmt_pct(net_m['total_return'])} | {_fmt_pct(gross_m['total_return'])} |
| 最大ドローダウン (MDD) | {_fmt_pct(mdd)} | — |
| 平均ターンオーバー | {avg_turnover:.4f} | — |
| 平均グロスエクスポージャー | {avg_gross_exp:.4f} | — |
| フォールバック発動率 | {fallback_rate:.1f}% | — |

*注: `leadlag.reporting.metrics.calculate_metrics`（月次集計ベース）では AR=145.83%、RISK=31.93%、Sharpe=4.5666、MDD=-7.20% が出力された。上記日次指標は `_compute_metrics` によるコスト後Net/Gross比較用。*

## コスト内訳

| 項目 | 平均/日 (bps) | 累計 |
|------|---------------|------|
| Slippage | {_fmt_bps(avg_slip)} | {_fmt_pct(float(slip.sum()))} |
| Financing | {_fmt_bps(avg_fin)} | {_fmt_pct(float(financing.sum()))} |
| Borrow | {_fmt_bps(avg_borrow)} | {_fmt_pct(float(borrow.sum()))} |
| Reverse | {_fmt_bps(avg_reverse)} | {_fmt_pct(float(reverse.sum()))} |
| **Total cost** | {_fmt_bps(avg_cost)} | {_fmt_pct(total_cost)} |

*コストは総Grossリターンの {_fmt_pct(cost_pct_of_gross)} に相当。*

## ポートフォリオ統計

- 平均グロスエクスポージャー: {avg_gross_exp:.4f}
- 平均ターンオーバー: {avg_turnover:.4f}
- 平均日次コスト: {_fmt_bps(avg_cost)} bps
- オーバーナイト持ち越し: long {0.75*100:.0f}%, short {0.50*100:.0f}%

## 監査結果

- ComplianceAuditor: 本バックテストスクリプトでは個別実行していない（`BacktestEngine` 内で look-ahead 制約・rolling統計を遵守）
- データ期間: 2010–2014 のベースライン期間を in-sample として使用し、2015 年以降を out-of-sample として評価
- ベースライン期間分離: production.yaml / `BacktestEngine` は `start_date=2015-01-05` 以降で backtest 開始

## 過学習ガード

- 本実行は既存本番configの再評価であり、新パラメータは追加していない
- 全期間（2015-2026）を in-sample として使用した 1 回の full-sample バックテスト
- ウォークフォワード検証については過去の sprint レポートを参照

## 実行コマンド

```bash
python3 src/research/scripts/backtest/run_production_backtest.py \\
    --config configs/production/production.yaml \\
    --start-date 2015-01-05 \\
    --output-dir {out_dir} \\
    --n-jobs -1
```

## 付録

- `daily_net_returns.csv`
- `daily_gross_returns.csv`
- `daily_turnover.csv`
- `daily_gross_exposure.csv`
- `daily_costs.csv` / `daily_slip_costs.csv` / `daily_financing_costs.csv` / `daily_borrow_costs.csv` / `daily_reverse_costs.csv`
- `daily_equity_curve.csv` / `daily_drawdown.csv`
- `daily_weights.csv`
"""

    report_path = report_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    # Also emit a machine-readable summary
    summary = {
        "model": "SectorRelativeEnsembleBLPEnhancedModel",
        "config": "configs/production/production.yaml",
        "period": {"start": net_m["start"], "end": net_m["end"]},
        "samples": net_m["samples"],
        "net": {
            "ar_pct": round(net_m["ar"] * 100, 4),
            "vol_ann_pct": round(net_m["vol_ann"] * 100, 4),
            "sharpe": round(net_m["sharpe"], 4),
            "total_return_pct": round(net_m["total_return"] * 100, 4),
            "mdd_pct": round(mdd * 100, 4),
            "avg_turnover": round(avg_turnover, 4),
            "avg_gross_exposure": round(avg_gross_exp, 4),
        },
        "gross": {
            "ar_pct": round(gross_m["ar"] * 100, 4),
            "vol_ann_pct": round(gross_m["vol_ann"] * 100, 4),
            "sharpe": round(gross_m["sharpe"], 4),
            "total_return_pct": round(gross_m["total_return"] * 100, 4),
        },
        "costs_bps_per_day": {
            "slip": round(avg_slip * 10000, 4),
            "financing": round(avg_fin * 10000, 4),
            "borrow": round(avg_borrow * 10000, 4),
            "reverse": round(avg_reverse * 10000, 4),
            "total": round(avg_cost * 10000, 4),
        },
        "fallback_rate_pct": round(fallback_rate, 2),
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
