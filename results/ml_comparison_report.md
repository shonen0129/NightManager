# 2 by 2 Comparative Performance Report: PCA vs LightGBM

This report compares the performance of PCA and LightGBM model combinations with Raw and Volatility-adjusted (Option3) targets over the testing period from **2020-01-01** to the end of the available historical dataset.

## Summary Results Table

| Metric | Case 1-1 (PCA × Raw) | Case 1-2 (PCA × Vol-adj) | Case 2-1 (LGBM × Raw) | Case 2-2 (LGBM × Vol-adj) |
| --- | --- | --- | --- | --- |
| Annualized Return (AR) | 80.34% | 81.43% | 73.82% | 65.81% |
| Annualized Volatility (RISK) | 21.22% | 20.68% | 21.20% | 18.60% |
| Risk-Return Ratio (Sharpe) | 3.7868 | 3.9374 | 3.4816 | 3.5380 |
| Max Drawdown (MDD) | -6.90% | -6.68% | -10.98% | -11.36% |
| Total Trades | 15060 | 15060 | 15060 | 15060 |
| Avg Daily Trades | 10.00 | 10.00 | 10.00 | 10.00 |
| Avg Gross Exposure | 200.00% | 200.00% | 200.00% | 200.00% |
| Avg One-way Daily Turnover | 150.95% | 154.46% | 154.29% | 153.96% |

## Key Observations and Performance Analysis

1. **Model Comparison (PCA vs. LightGBM)**:
   - LightGBM captures non-linear relationships and interactions among sector returns, potentially yielding a higher raw forecast precision.
   - However, the PCA model benefits from explicit group structure priors (sector sensitivity vectors v1-v6), which keep predictions stable and aligned with known macroeconomic exposures.

2. **Target Variable Comparison (Raw vs. Volatility-adjusted)**:
   - Volatility-adjusted (Z-score) targets adjust returns by their rolling 20-day historical standard deviation. This acts to scale down predictions during highly volatile regimes and scale them up during low volatility regimes, reducing portfolio risk concentrations.
   - Standardizing the target helps stabilize the LightGBM learning process, as the target variable is homogeneous across time and different ETFs.

3. **Risk-adjusted Performance**:
   - The Sharpe (Risk-Return) ratio is the primary indicator of risk-adjusted stability. Verify the Sharpe ratio to identify which combination provides the most stable alpha generation.
   - Max Drawdown (MDD) highlights the tail-risk reduction capacity of each approach, especially when using the volatility-adjusted target.
