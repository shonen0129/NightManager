# Adaptive TOPIX-Beta Window Backtest Report

**Date:** 2026-07-29 01:46  
**Config:** `configs/production/production.yaml`  
**Adaptive windows:** [60, 126, 252]  
**Chow threshold:** 6.0  
**Runtime:** 5.3 minutes

## Methodology

- **Baseline:** fixed 60-day TOPIX-beta residualisation window.
- **Adaptive:** for each sector and each date, select the longest window among
  the candidate list whose Chow F-statistic (midpoint split) is below the
  threshold.  If no window is stable, the shortest window with enough data is
  used.  The test is run on historical data only (rows strictly before t).

## Results

| Metric | Baseline (fixed 60) | Adaptive | Delta |
|---|---|---|---|
| Net Sharpe | 4.5986 | 3.6353 | -20.95% |
| Ann. Return | 145.86% | 123.29% | -22.56% |
| Ann. Risk | 31.72% | 33.92% | +2.20% |
| Max Drawdown | -5.97% | -11.56% | -5.58% |
| Total Return | 543123077.78% | 48020020.45% | compound metric; delta not meaningful |
| Avg Turnover | 1.6264 | 1.6094 | -1.71% |
| Avg Gross Exp | 2.0000 | 2.0000 | +0.00% |
| Avg Total Cost (bps/day) | 13.95 | 13.88 | -0.00% |

## Cost Breakdown (bps/day)

| Component | Baseline | Adaptive |
|---|---|---|
| Slippage | 11.38 | 11.31 |
| Financing | 0.79 | 0.79 |
| Borrow | 0.24 | 0.24 |
| Reverse | 1.54 | 1.54 |

## Interpretation

Compare net Sharpe, maximum drawdown, and turnover.  A material improvement
in Sharpe with stable or lower turnover supports the adaptive window idea.
If the delta is small or negative, the added complexity is not justified by
this in-sample backtest and further tuning (e.g. threshold, window set) or
walk-forward validation is required before adoption.

## Artifacts

- `/Users/shonen/日米ラグ/results/adaptive_beta_window/baseline`
- `/Users/shonen/日米ラグ/results/adaptive_beta_window/adaptive`
