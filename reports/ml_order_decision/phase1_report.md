# Phase 1 Report: ML Order Decision Overlay (Per-Ticker Gap Ridge)

**Test period:** 2023-01-04 → 2024-12-30
**Trading days:** 474
**Output directory:** `reports/ml_order_decision/phase1_results`

## 1. Overall performance (net of costs)

| Metric | Baseline | Overlay | Diff |
|--------|----------|---------|------|
| Sharpe | 5.127 | 5.161 | +0.034 |
| Annual Return | 0.971 | 0.975 | +0.004 |
| Volatility | 0.189 | 0.189 | -0.001 |
| Max DD | -0.084 | -0.077 | +0.007 |
| Skewness | 0.108 | 0.099 | -0.009 |
| Excess Kurt | 1.458 | 1.137 | -0.321 |
| Mean Turnover | 1.530 | 1.535 | +0.005 |
| Mean Gross Exp | 2.000 | 2.000 | +0.000 |

## 2. Statistical test

- Mean daily return difference (overlay - baseline): **0.000015**
- Paired t-statistic: **0.351**
- p-value: **0.7258**

## 3. Yearly breakdown

| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD |
|------|-------------|----------------|---------|------------|----------|-------------|
| 2023 | 3.674 | 3.649 | 0.594 | 0.598 | -0.049 | -0.053 |
| 2024 | 6.394 | 6.493 | 1.351 | 1.355 | -0.084 | -0.077 |

## 4. Top Ridge coefficients

| Feature | Coefficient |
|---------|-------------|
| score_x_gap | 0.001679 |
| score_x_gap_idio | -0.001357 |
| abs_score | 0.001189 |
| gap_idio | 0.000819 |
| ticker_1629.T | -0.000792 |
| ticker_1625.T | 0.000677 |
| market_vol_20d | 0.000635 |
| ticker_1624.T | -0.000628 |
| ticker_1617.T | 0.000616 |
| gap | -0.000561 |
| ticker_1618.T | -0.000471 |
| ticker_1620.T | 0.000437 |
| ticker_1621.T | 0.000415 |
| ticker_1631.T | 0.000370 |
| abs_gap | 0.000315 |

## 5. Notes and limitations

- The overlay applies ``p_trade = sigmoid(contribution_hat / target_std)`` to rescale the raw ``mu_gap / sigma_gap`` scores before V2 weight construction.
- RuleD multiplier is taken from the baseline V2 run (PIT history is not recomputed for the overlay because the available gap distribution output lacks a diagnostics CSV; multiplier is 1.0 in this run).
- Cost, financing, borrow, and reverse-fee calculations are exactly those used by ``BacktestEngine.run_v2_backtest`` because the overlay is injected by monkey-patching ``generate_v2_production_portfolio``.
- Training target is ``side * realized_9:10_close - round_trip_cost`` per ticker, where ``realized`` comes from ``compute_jp_target_returns``.

