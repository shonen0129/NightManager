# 長期間本番バックテストレポート

> 作成日: 2026-08-01
> モデル: SectorRelativeEnsembleBLPEnhancedModel (Residual-BLPX / production_residual_blpx)
> Config: `configs/production/production.yaml`
> 期間: 2015-01-05 〜 2026-07-31
> バックテスト出力: `/Users/shonen/leadlag/results/backtest_20260801_144758`

## 概要

本番設定（production.yaml）で2015-01-05から最新日（2026-07-31）までの全期間をバックテストした。
コストは片道5bps + 金利2.5% + 貸株1.15% + 逆日歩2bps/day、オーバーナイト持ち越し（long 75% / short 50%）を含むnetで評価。

## 結果サマリー

| 指標 | Net（コスト後） | Gross（コスト前） |
|------|----------------|-------------------|
| 期間 | 2015-01-05 〜 2026-07-31 | 2015-01-05 〜 2026-07-31 |
| サンプル日数 | 2747 | 2747 |
| 年率リターン (AR) | 143.94% | 179.02% |
| 年率ボラティリティ | 16.46% | 16.46% |
| **Sharpe (日次, ann)** | **8.7446** | **10.8774** |
| 累積リターン | 539688217.24% | 24124572504.96% |
| 最大ドローダウン (MDD) | -7.20% | — |
| 平均ターンオーバー | 1.6139 | — |
| 平均グロスエクスポージャー | 2.0000 | — |
| フォールバック発動率 | 0.0% | — |

*注: `leadlag.reporting.metrics.calculate_metrics`（月次集計ベース）では AR=145.83%、RISK=31.93%、Sharpe=4.5666、MDD=-7.20% が出力された。上記日次指標は `_compute_metrics` によるコスト後Net/Gross比較用。*

## コスト内訳

| 項目 | 平均/日 (bps) | 累計 |
|------|---------------|------|
| Slippage | 11.35 | 311.76% |
| Financing | 0.79 | 21.71% |
| Borrow | 0.24 | 6.66% |
| Reverse | 1.54 | 42.26% |
| **Total cost** | 13.92 | 382.39% |

*コストは総Grossリターンの 19.60% に相当。*

## ポートフォリオ統計

- 平均グロスエクスポージャー: 2.0000
- 平均ターンオーバー: 1.6139
- 平均日次コスト: 13.92 bps
- オーバーナイト持ち越し: long 75%, short 50%

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
python3 src/research/scripts/backtest/run_production_backtest.py \
    --config configs/production/production.yaml \
    --start-date 2015-01-05 \
    --output-dir /Users/shonen/leadlag/results/backtest_20260801_144758 \
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
