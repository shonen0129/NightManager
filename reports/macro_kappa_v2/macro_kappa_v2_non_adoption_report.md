# Macro Kappa V2 統合実験 不採用レポート

## 概要

マクロ Factor-Kappa（`compute_sigma_yy_inflation` による `Omega_gap` 膨張）を v2 production pipeline に統合し、複数の kappa 設定でバックテストを実施した。結果、改善は限定的かつパラメータに敏感であったため、**本レポート作成時点では不採用**とする。

## 目的

`reports/sprint_macro_direction/macro_direction_adjustment_report.md` で「Kappa only」が `Sharpe 4.41 → 4.50（+1.8%）` と軽微な改善を示したことを受け、同じ Omega_gap 膨張ロジックを v2 production pipeline（`production_v2.py`）で再検証し、採用可否を判断する。

## 実装内容

### 1. 実験スクリプト

`scripts/experiments/experiment_macro_kappa_v2.py` を新規作成。

- `configs/production/production.yaml` を読み込み
- `portfolio.macro_kappa_enabled` と `macro_kappas` を動的に変更
- `BacktestEngine.run_v2_backtest` を使用して gap-adjusted distribution matrices 上でバックテスト
- マクロ価格は `download_macro_prices` を事前に 1 回ダウンロードし、当日以前のデータのみ返すようにキャッシュパッチ（ルックアヘッド防止）

### 2. 比較バリアント

- **Baseline**: `macro_kappa_enabled=False`
- **Kappa [3.0, 0.5, 0.5]**: デフォルト値
- **Kappa [1.5, 0.5, 0.5]**: USDJPY 係数半減
- **Kappa [6.0, 1.0, 1.0]**: 全係数強化
- **Kappa [3.0, 0.0, 0.0]**: USDJPY のみ
- **Kappa [1.0, 0.3, 0.3]**: 小さめ設定
- **Kappa [0.5, 0.2, 0.2]**: より小さめ設定
- **Kappa [0.3, 0.1, 0.1]**: 最小設定

## バックテスト条件

- **期間**: 2020-01-06 ～ 2026-06-10（1,512 営業日）
- **データ**: gap-adjusted distribution matrices（`results/gap_adjusted_distribution/20260615_004202`）
- **コストモデル**: 片道 5bps slippage、overnight alpha=0（日次全額決済）
- **macro surprise halflife**: mean=20d, vol=60d
- **マクロ因子**: USDJPY, CLF, TNX
- **ルックアヘッド防止**: キャッシュされたマクロ価格を各バックテスト日の `end` 日付でスライス

## 結果

### サマリー指標

| Variant | Sharpe (net) | AR (net) | MDD | Turnover | GrossExp | Fallback |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (macro kappa OFF) | 3.98 | 114.51% | -7.10% | 1.401 | 1.808 | 3.70% |
| [3.0, 0.5, 0.5] | 3.98 | 108.64% | -6.88% | 1.301 | 1.677 | 3.70% |
| [1.5, 0.5, 0.5] | **4.05** | 111.46% | -6.86% | 1.336 | 1.721 | 3.70% |
| [6.0, 1.0, 1.0] | 4.01 | 105.18% | -6.63% | 1.263 | 1.628 | 3.70% |
| [3.0, 0.0, 0.0] | 3.97 | 108.46% | -6.78% | 1.309 | 1.687 | 3.70% |
| [1.0, 0.3, 0.3] | 4.02 | 112.93% | -6.94% | 1.354 | 1.744 | 3.70% |
| [0.5, 0.2, 0.2] | **4.05** | 113.73% | -7.20% | 1.373 | 1.768 | 3.70% |
| [0.3, 0.1, 0.1] | 4.01 | 114.18% | -7.25% | 1.384 | 1.783 | 3.70% |

### 主要指標の相対比較

| Variant | Sharpe vs Baseline | AR vs Baseline | MDD vs Baseline |
|---|---:|---:|---:|
| [3.0, 0.5, 0.5] | 0.00 | -5.87pp | +0.22pp |
| [1.5, 0.5, 0.5] | +0.07 | -3.05pp | +0.24pp |
| [6.0, 1.0, 1.0] | +0.03 | -9.33pp | +0.47pp |
| [3.0, 0.0, 0.0] | -0.01 | -6.05pp | +0.32pp |
| [1.0, 0.3, 0.3] | +0.04 | -1.58pp | +0.16pp |
| [0.5, 0.2, 0.2] | +0.07 | -0.78pp | -0.10pp |
| [0.3, 0.1, 0.1] | +0.03 | -0.33pp | -0.15pp |

## 分析

### 限定的な Sharpe 改善

- 最良設定 `[1.5, 0.5, 0.5]` と `[0.5, 0.2, 0.2]` で `Sharpe = 4.05`、baseline `3.98` から **+1.8% 相対改善**
- しかし他の 4 つの設定では baseline 以下または同等
- 改善幅は 0.07 Sharpe ポイントと限定的

### AR とのトレードオフ

- kappa を強めるほど `Turnover` と `GrossExp`、MDD は改善する傾向
- 一方で `AR` は継続的に低下
- `[6.0, 1.0, 1.0]` では AR が -9.33% と大幅低下
- 最小設定 `[0.3, 0.1, 0.1]` では AR ほぼ維持（-0.33pp）だが Sharpe 改善もわずか（+0.03）

### パラメータ感性

- kappa の大小で結果が非単調に変動
- 最適 kappa の特定が困難であり、過学習リスクが高い
- USDJPY のみ（CLF/TNX=0）でも改善せず、複数因子の寄与が不透明

### ルックアヘッド防止の確認

- 初回実行ではマクロ価格のキャッシュパッチが `end` 日付を無視し、未来データを使用するバグが発生
- 修正後に再実行。サプライズの日次変動が確認でき、結果に未来情報が含まれていないことを確認

## 結論

| Variant | 判定 | 理由 |
|---|---|---|
| Baseline | **継続採用** | 最も安定的で高い AR |
| [1.5, 0.5, 0.5] / [0.5, 0.2, 0.2] | **不採用** | Sharpe +0.07 のみ。AR 低下、パラメータ感度大 |
| その他 kappa 設定 | **不採用** | baseline 以下または同等 |

**Macro Kappa の v2 production pipeline 有効化は現時点では不採用とする。**

理由:
1. 最大でも 0.07 ポイントの Sharpe 改善に留まり、経済的効果が小さい
2. 改善は特定の kappa 値に依存し、パラメータロバスト性に欠ける
3. kappa ON 全般で AR が低下しており、alpha 生成を損なう
4. ウォークフォワード検証と PBO 評価が未実施

## 注意事項

- 本バックテストは overnight alpha=0（日次全額決済）で実施。本番の `alpha_long=0.75, alpha_short=0.5` では金利・貸株・逆日歩コストが追加される
- レポート作成時点ではウォークフォワード検証未実施。過学習リスクを完全に排除できていない
- マクロ感応度行列 `MACRO_SENS_MATRIX` が固定値である点も改善余地として残存
- 実験スクリプトと CSV 結果は `scripts/experiments/experiment_macro_kappa_v2.py` および `artifacts/macro_kappa_v2/macro_kappa_v2_results.csv` に保存
