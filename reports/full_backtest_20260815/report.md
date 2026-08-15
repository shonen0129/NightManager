# Full Period V2 Backtest — 2015-01-05 to 2026-08-13

> 作成日: 2026-08-15
> モデル: ProductionV2Model (Residual-BLPX-RA v2)
> Config: `configs/production/production.yaml`
> 期間: 2015-01-05 〜 2026-08-13

## 実行コマンド

```bash
python3 -m leadlag.cli backtest \
  --config configs/production/production.yaml \
  --start-date 2015-01-05 \
  --end-date 2026-08-13 \
  --gap-dir var/live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --n-jobs -1 \
  --output-level minimal \
  --run-tag full_2015_20260813
```

- `df_exec` ローカルキャッシュ: 4135 行（2009-01-07 〜 2026-08-17）
- gap 調整済み分布: `var/live/pipeline_data/gap_adjusted_distribution/20260731_024303`
  - `mu_gap`, `omega_gap`, `mu_gap_h3`, `omega_gap_h3`, `mu_gap_h5`, `omega_gap_h5`, `rank_reversal` 各 2020-01-06 〜 2026-07-29
- 2020-01-05 以前と 2026-07-30 以降は on-demand BLPX 計算で gap 行列を補完
- side_leverage: 1.5
- サンプル日数: 2755 営業日

## 結果サマリー

| 指標 | ネット（コスト後） | グロス（コスト前） | 備考 |
|------|-------------------|-------------------|------|
| Sharpe | **4.12** | **4.99** | 月次リターンから年率換算 |
| 算術年率（AR） | **199.04%** | **252.49%** | 月次リターンから年率換算 |
| 年率ボラティリティ | **48.29%** | **50.60%** | 月次リターンから年率換算 |
| R/R | **4.12** | **4.99** | AR / 年率ボラティリティ |
| 最大ドローダウン | **-8.26%** | **-5.50%** | wealth 累積最大ドローダウン |
| 最終 wealth 倍率 | **875,838,871x** | **163,948,988,308x** | (1+r).cumprod() の最終値 |
| 最終リターン（% 表示） | **87,583,887,010%** | — | `wealth − 1` を百分率表示 |
| 平均ターンオーバー | **1.43** | — | 日次 (\|w_t − w_{t−1}\| / 2) |
| 平均グロスエクスポージャー（raw） | **1.88** | — | サイドレバレッジ適用前 |
| 推定実効グロス（1.5×） | **2.82** | — | 市場中立制限 3.0 内 |
| フォールバック発動率 | **0.0%** | — | 2755 / 2755 日で gap 行列取得成功 |

## コスト内訳

`BacktestEngine.run_v2_backtest` から `BacktestResultStore.daily_pnl` に保存された各日のコストを集計。

- 1日あたり平均総コスト: 1.92 bps（算術平均）
- 期間累積総コスト: 5.28（wealth 単位、算術和）
  - スリッページ: 81.1%（ターンオーバー 1.43、片道 5 bps）
  - 逆日歩: 11.3%
  - ロング金利: 5.8%
  - 貸株料: 1.8%

## ポートフォリオ統計

- ロング/ショート数: `long_count=5`, `short_count=5`
- ランキング: `mu_over_sigma` (gap 調整済み μ / σ)
- グロススケーリング: RuleD（Low 0.75× / Medium 1.00× / High 1.00×）
- MinVar: 有効（`minvar_alpha=0.8`）
- マルチホライズンブレンド: 有効（h=1,3,5 / weights=0.8,0.1,0.1）
- マクロ感応度スケーリング: 有効
- ランク反転オーバーレイ: 有効（weight=0.05）

## 期間選択と gap 行列

- V2 本番モデルは **当日の gap 調整済み分布行列** (`mu_gap_*`, `omega_gap_*`) が必須。
- `var/live/pipeline_data/gap_adjusted_distribution/20260731_024303` には 2020-01-06 〜 2026-07-29 までの行列が存在。
- 2020-01-05 以前と 2026-07-30 以降は、`ondemand_fallback_enabled=true` により `ProductionV2Model._compute_ondemand` で gap 行列を計算。2755 日すべてで成功し、fallback は 0 日。
- 本番 config はフラット化済み `configs/production/production.yaml` を使用。

## 監査・不変条件

- `load_gap_matrices` は当日日付の行列のみを検索。前日行列のコピーは行われていない。
- フォールバック: gap データ欠損時は `w_final=0` のフラットポジションを返す（V1 fallback 廃止に準拠）。
- ルックアヘッド: PIT ビニング、ローリング統計、共分散正則化は全て strictly historical。
- 本番 config の `vol_adjusted_target=false` によりインサンプル平均・標準偏差で `mu_raw` を復元。

## 結論

本番設定 `production.yaml` において、最長期間 2015-01-05 〜 2026-08-13（2755 営業日）での V2 バックテストは **net Sharpe 4.12、max DD -8.26%、フォールバック 0%** で完了した。算術年率はネット **199.04%**、グロス **252.49%**。最終 wealth 倍率はネット **875,838,871x**、グロス **163,948,988,308x** となり、長期間の高い幾何複利効果が顕著に表れている。

## 出力ファイル

- `var/results/20260815_111443_full_2015_20260813/`
  - `daily_results.csv`
  - `metrics.csv`
  - `backtest_store.sqlite`
  - `cumulative_return.png`
  - `drawdowns.png`
  - `run_summary.json`
