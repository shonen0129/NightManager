# V2 本番設定ロングバックテスト（最長期間）

> 作成日: 2026-08-06
> モデル: ProductionV2Model (Residual-BLPX-RA v2)
> Config: `configs/production/production.yaml`
> 期間: 2020-01-06 〜 2026-07-29

## 実行コマンド

```bash
python3 scripts/run_v2_backtest.py \
  --config configs/production/production.yaml \
  --gap-dir live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --start-date 2020-01-06 \
  --end-date 2026-07-29 \
  --output-dir results/v2_backtest_20200106_20260729_live \
  --n-jobs -1
```

- `df_exec` ローカルキャッシュ: 4165 行（2009-01-07 〜 2026-08-05）
- gap 調整済み分布: `live/pipeline_data/gap_adjusted_distribution/20260731_024303`
  - `mu_gap`, `omega_gap`, `mu_gap_h3`, `omega_gap_h3`, `mu_gap_h5`, `omega_gap_h5`, `rank_reversal` 各 1544 日分
- side_leverage: 1.5（実行者の設定に合わせたデフォルト値）

## 結果サマリー

| 指標 | ネット（コスト後） | グロス（コスト前） | 備考 |
|------|-------------------|-------------------|------|
| Sharpe | **7.66** | **9.72** | 年率換算、フォールバック日除外 |
| 累積リターン（幾何） | **3,175,356%** | **55,850,068%** | (1+r).cumprod() − 1 |
| 幾何年率（CAGR） | **385.58%** | **651.77%** | 累積リターンから逆算 |
| 算術年率（AR） | **172.22%** | **219.39%** | mean × 252 |
| 年率ボラティリティ | **22.48%** | **22.56%** | std × √252 |
| 最大ドローダウン | **-6.96%** | — | wealth 累積最大ドローダウン |
| 平均ターンオーバー | **1.36** | — | 日次 (\|w_t − w_{t−1}\| / 2) |
| 平均グロスエクスポージャー（raw） | **1.86** | — | サイドレバレッジ適用前 |
| 推定実効グロス（1.5×） | **2.79** | — | 市場中立制限 3.0 内 |
| フォールバック発動率 | **0.0%** | — | 1544 / 1544 日で gap 行列取得成功 |
| 平均コスト（1日あたり） | **18.72 bps** | — | トータルコスト |

## コスト内訳について

`scripts/run_v2_backtest.py` は合計コスト `daily_costs` のみを保存する。内訳（slippage / financing / borrow / reverse）を得るには `BacktestEngine.run_v2_backtest` の返り値を直接利用する必要がある。合計コストは `gross_return - net_return` と一致し、ターンオーバー 1.36、スリッページ 5 bps/片道、overnight holding (long α=0.75, short α=0.5)、金利・貸株・逆日歩を含んでいる。

## ポートフォリオ統計

- ロング/ショート数: config `long_count=5`, `short_count=5`
- ランキング: `mu_over_sigma` (gap 調整済み μ / σ)
- グロススケーリング: RuleD（Low 0.75× / Medium 1.00× / High 1.00×）
- MinVar: 有効（α=0.8）
- マルチホライズンブレンド: 有効（h=1,3,5 / weights=0.8,0.1,0.1）
- ランク反転オーバーレイ: 有効（weight=0.05）

## 期間選択の理由

- V2 本番モデルは **当日の gap 調整済み分布行列** (`mu_gap_*`, `omega_gap_*`) が必須。
- `results/gap_adjusted_distribution/latest`（20260708_184828）には 2025-10 〜 2026-01 の一部 gap 行列が欠損しており、フォールバック 56 日（3.7%）が発生した。
- より新しく完全な `live/pipeline_data/gap_adjusted_distribution/20260731_024303` を使用することで、**2020-01-06 〜 2026-07-29 の 1544 営業日、0 フォールバック** でバックテストできた。
- 2020-01-06 より前の期間については gap 行列が存在しないため、V2 本番設定では 2020-01-06 以降が現時点で最長期間となる。

## 監査・不変条件

- `load_gap_matrices` は当日日付の行列のみを検索。前日行列のコピーは行われていない。
- フォールバック: gap データ欠損時は `w_final=0` のフラットポジションを返す（V1 fallback 廃止に準拠）。
- ルックアヘッド: PIT ビニング、ローリング統計、共分散正則化は全て strictly historical。

## 結論

本番設定 `production.yaml` において、現時点で利用可能な最長期間（2020-01-06 〜 2026-07-29、1544 日）での V2 バックテストは **net Sharpe 7.66、max DD -6.96%、フォールバック 0%** で完了した。幾何累積リターンは **3,175,356%**、CAGR は **385.58%** となる。2026-07-30 以降のデータについては最新の gap 調整済み分布行列を再計算・配置することで期間を延長可能。
