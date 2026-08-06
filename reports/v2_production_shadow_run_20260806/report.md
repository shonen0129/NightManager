# 本番 V2 ロジック shadow run 生成レポート

> 作成日: 2026-08-06
> 生成スクリプト: `scripts/experiments/build_v2_production_shadow_run.py`

## 概要

既存のシャドウラン `shadow_runs/p8p3_blpx` は `tools/validation/run_daily_residual_blpx_shadow.py` という独立した簡易実装で作成されており、本番 `ProductionV2Model` と**ロジックが異なる**。そのため、本番に合わせた shadow run を新規に作成した。

本レポートでは、本番 V2 ロジックを使って生成した shadow run の結果を既存 `p8p3_blpx` および V2 バックテストと比較する。

## 作成した shadow run

### スクリプト

- `scripts/experiments/build_v2_production_shadow_run.py`
  - 各日付で `src/leadlag/models/production_v2.py::generate_v2_production_portfolio`（overlay あり版は `generate_v2_production_portfolio_with_overlay`）を呼び出す。
  - `configs/production/production.yaml` をそのまま使用。
  - `gap_input_dir` にはフル期間を持つ `live/pipeline_data/gap_adjusted_distribution/20260731_024303` を使用。
  - 出力は `shadow_runs/v2_production_20200106_20260729`（overlay なし）および `shadow_runs/v2_production_20200106_20260729_overlay`（overlay あり）。

### 本番と同じ経路で生成

本 shadow run は以下の本番機能をすべて有効化:

- `mu_over_sigma` ランキング
- MinVar 最適化（alpha=0.8）
- RuleD 動的グロス（PIT 三分位: Low 0.75 / Mid 1.0 / High 1.0）
- マルチホライズンブレンド（h=1,3,5, weights=0.8,0.1,0.1）
- ランク反転オーバーレイ（weight=0.05）
- ML order overlay（`models/ml_order_overlay/phase2_8`）
- 本番 config のすべてのパラメータ

### 実行コマンド

```bash
# overlay あり本番 V2 shadow run
python3 scripts/experiments/build_v2_production_shadow_run.py \
  --start 2020-01-06 --end 2026-07-29 \
  --overlay true \
  --shadow-root shadow_runs/v2_production_20200106_20260729_overlay \
  --clean true

# 上記の shadow run を monitor
python3 tools/validation/monitor_residual_blpx_shadow_performance.py \
  --shadow-root shadow_runs/v2_production_20200106_20260729_overlay \
  --gap-input-dir live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --output-dir results/shadow_monitor_v2_production_overlay
```

## パフォーマンス比較

| 指標 | V2 backtest (2020-01-06〜2026-07-29) | V2 shadow overlay あり | V2 shadow overlay なし | 既存 p8p3_blpx (2020-01-06〜2026-06-12) |
|------|--------------------------------------|------------------------|------------------------|------------------------------------------|
| 期間日数 | 1544 | 1544 | 1544 | 1458 |
| 年率リターン | **172.22%** | **97.35%** | **93.36%** | **84.33%** |
| 年率ボラティリティ | 22.48% | 13.56% | 13.16% | 13.65% |
| Net Sharpe | **7.66** | **7.18** | **7.09** | **6.18** |
| Sortino | — | 16.06 | 16.17 | 12.97 |
| Max DD | -6.96% | -5.17% | -4.90% | -5.44% |
| Calmar | — | 18.81 | 19.06 | 15.49 |
| Avg Turnover | 1.36 | 2.74 | 2.72 | 2.90 |
| Hit Rate | — | 67.9% | 68.2% | 63.9% |

### ポイント

1. **既存 p8p3_blpx との差**
   - `p8p3_blpx` は MinVar、マルチホライズン、ランク反転、ML overlay を含まない簡易実装。
   - 本番 V2 shadow（overlay あり）の方が **Sharpe +1.00（7.18 vs 6.18）**、**AR +13pp（97.35% vs 84.33%）**、**Max DD も浅い（-5.17% vs -5.44%）**。
   - ターンバーはほぼ同水準（2.74 vs 2.90）。

2. **V2 backtest との差**
   - V2 バックテストは `side_leverage=1.5`、overnight holding、slippage/financing/borrow/reverse を含むため、**AR/ボラがシャドウの約 1.8 倍**になる。
   - shadow monitor はコストを「10 bps per unit gross」で一元的に引き、side leverage も overnight もモデル化していない。
   - side leverage 1.5x を shadow raw return に掛けると、overlay ありで約 **146% AR**、overlay なしで約 **140% AR** と推定される。依然として backtest 172% より低いが、その差は backtest の overnight alpha、slippage コスト構造、`df_exec` 由来のターゲットリターンの違いに起因。

3. **ML overlay の効果**
   - overlay あり: AR 97.35%, Sharpe 7.18, MDD -5.17%
   - overlay なし: AR 93.36%, Sharpe 7.09, MDD -4.90%
   - リターンは +4pp、Sharpe は微増、MDD はやや大きくなる。本番では overlay 有効なので、**overlay ありを本番基準とする**。

## shadow ディレクトリ構成

```text
shadow_runs/v2_production_20200106_20260729_overlay/
├── 20200106/
│   ├── shadow_portfolios.csv
│   ├── shadow_candidate_summary.csv
│   ├── shadow_scores.csv
│   ├── shadow_risk_estimates.csv
│   ├── shadow_orders_preview.csv
│   ├── run_config.json
│   ├── pit_binning_audit.json
│   ├── data_availability.json
│   ├── leakage_audit.json
│   └── numerical_audit.json
├── 20200107/
│   └── ...
└── ...
```

`shadow_portfolios.csv` は `tools/validation/monitor_residual_blpx_shadow_performance.py` と互換性がある形式で出力。

## 今後の利用方法

本番 V2 shadow run を継続的にメンテナンスするには:

1. `build_v2_production_shadow_run.py` を定期実行し、最新の gap 調整済み分布を使用する。
2. monitor を `live/pipeline_data/gap_adjusted_distribution/latest` ではなく、フル期間を蓄積した gap ディレクトリで実行する（PIT 履歴不足を防ぐ）。
3. 本番 V2 shadow と live 本番の weights を日次で diff し、差分を監視する。

## 注意点

- shadow monitor は side_leverage（1.5x）、overnight holding alpha、financing/borrow/reverse コストを含まない。したがって **本番口座の実現 P&L ではなく、「本番ロジックの生のポートフォリオリターン」を評価する指標** として扱う。
- V2 backtest の最終 wealth 31,754x は compounding 下での理論値であり、実際の口座は証拠金・整数株制約で同一 compounding は不可能。
