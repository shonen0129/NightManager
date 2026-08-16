# V2 信号経由凸最適化バックテスト実験

**Period**: 2020-01-06 to 2026-07-29

**Trading days**: 1544

**V2 fallback days**: 0

**Convex optimization failures (exception)**: 0

**Convex SLSQP non-convergence days**: 293

**Hypothesis**: V2 同等の信号・ウェイト処理を経た上で凸最適化を適用すると、
Net Sharpe や MDD を改善できるか検証する。

## Methodology

- V2 Baseline: `ProductionV2Model.decide()` をそのまま使用
- Convex (V2 prior): V2 の `mu_gap` / `Omega_gap` / RuleD グロス乗数を使用し、
  `w_prev` には前営業日の V2 `w_final` を指定
- Convex (convex prior): 同じ V2 入力に対し `w_prev` を前日の凸最適化結果とする
- Convex hyperparameters: lambda_risk=5.0, cost_bps=5.0, turnover_penalty=0.0001, max_single_weight=0.25

## Results

| Metric | V2 Baseline | Convex (V2 prior) | Convex (convex prior) |
|---|---|---|---|
| Net Sharpe | 4.8162 | 2.4401 | 2.5004 |
| Gross Sharpe | 5.9320 | 3.6189 | 3.7742 |
| Annualized Net Return | 165.29% | 74.98% | 72.47% |
| Annualized Net Vol | 34.32% | 30.73% | 28.98% |
| Return Risk | 4.8162 | 2.4401 | 2.5004 |
| Total Net Return | 2015999.26% | 9023.84% | 7932.49% |
| Max Drawdown | -6.61% | -28.86% | -29.26% |
| Avg Daily Turnover | 1.2891 | 1.2214 | 1.1842 |
| Avg Gross Exposure | 1.7850 | 1.6090 | 1.6434 |
| Total Slippage Cost | 223.05% | 206.13% | 207.73% |
| Total Financing Cost | 16.50% | 14.84% | 15.20% |
| Total Borrow Cost | 5.06% | 4.55% | 4.66% |
| Total Reverse Fee | 32.12% | 28.89% | 29.58% |

## Caveats

- これは in-sample バックテストであり、ウォークフォワード検証やシャドー運用は別途必要。
- 凸最適化は V2 生成後の `mu_gap` / `Omega_gap` を入力に用いているが、
  V2 の multi-horizon blend・rank overlay・minvar は `scores` / ウェイトとして扱われ、
  これらを凸最適化に直接組み込むためには追加設計が必要。
- 2020-01-06 以前は gap ファイルがないため on-demand 計算になり、計算時間が増加する。
