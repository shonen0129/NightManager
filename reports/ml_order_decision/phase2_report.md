# Phase 2 Report: ML Order Decision Overlay (Per-Ticker Gap LightGBM)

**Test period:** 2023-01-04 → 2024-12-30
**Trading days:** 474
**Output directory:** `reports/ml_order_decision/phase2_results`

## 1. Overall performance (net of costs)

| Metric | Baseline | Overlay | Diff |
|--------|----------|---------|------|
| Sharpe | 5.127 | 5.159 | +0.032 |
| Annual Return | 0.971 | 0.988 | +0.017 |
| Volatility | 0.189 | 0.191 | +0.002 |
| Max DD | -0.084 | -0.084 | -0.000 |
| Skewness | 0.108 | 0.153 | +0.045 |
| Excess Kurt | 1.458 | 1.433 | -0.026 |
| Mean Turnover | 1.530 | 1.536 | +0.007 |
| Mean Gross Exp | 2.000 | 2.000 | +0.000 |

## 2. Statistical test

- Mean daily return difference (overlay - baseline): **0.000068**
- Paired t-statistic: **2.082**
- p-value: **0.0379**

## 3. Yearly breakdown

| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD |
|------|-------------|----------------|---------|------------|----------|-------------|
| 2023 | 3.674 | 3.628 | 0.594 | 0.596 | -0.049 | -0.053 |
| 2024 | 6.394 | 6.498 | 1.351 | 1.382 | -0.084 | -0.084 |

## 4. Top LightGBM feature importance

| Feature | Importance |
|---------|-------------|
| ticker | 84.000000 |
| topix_night | 64.000000 |
| mu_gap | 52.000000 |
| score_x_gap_idio | 51.000000 |
| market_vol_20d | 40.000000 |
| gap_idio | 39.000000 |
| score_x_gap | 31.000000 |
| abs_gap | 25.000000 |
| sigma_gap | 23.000000 |
| gap | 23.000000 |
| abs_score | 15.000000 |
| score | 10.000000 |

## 5. Notes and limitations

- The overlay applies ``p_trade = sigmoid(contribution_hat / target_std)`` to rescale the raw ``mu_gap / sigma_gap`` scores before V2 weight construction.
- RuleD multiplier is taken from the baseline V2 run (PIT history is not recomputed for the overlay because the available gap distribution output lacks a diagnostics CSV; multiplier is 1.0 in this run).
- Cost, financing, borrow, and reverse-fee calculations are exactly those used by ``BacktestEngine.run_v2_backtest`` because the overlay is injected by monkey-patching ``generate_v2_production_portfolio``.
- Training target is ``side * realized_9:10_close - round_trip_cost`` per ticker, where ``realized`` comes from ``compute_jp_target_returns``.

