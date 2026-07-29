# Phase 2.5 Walk-Forward Report (EMA + stronger regularization)
**Config**: EMA span=0.0, n_estimators=100, num_leaves=20, reg_lambda=1.0, per_ticker_interactions=True

## Pooled OOS performance
- Periods: 2022, 2023, 2024
- Total OOS days: 709
- Baseline Sharpe: 5.9664
- Overlay Sharpe: 5.9700
- Mean daily return difference: 0.000125
- Walk-forward win rate: 33.3% (1/3)
- DSR (12 trials): 1.0000
- Mean turnover: base=1.5374, overlay=1.5466

## Per-fold metrics
| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |
|------|-------------|----------------|---------|------------|----------|-------------|---------|
| 2022 | 7.7706 | 7.5439 | 1.4103 | 1.4412 | -0.0412 | -0.0421 | 0.4069 |
| 2023 | 3.6740 | 3.6247 | 0.5939 | 0.6056 | -0.0492 | -0.0579 | 0.5798 |
| 2024 | 6.3941 | 6.6081 | 1.3513 | 1.4005 | -0.0839 | -0.0869 | 0.0262 |

## Sensitivity (2024 fold, ±20% hyperparameters)
| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |
|---------|----------------|------------|-------------|----------|
| minus20 | 6.4638 | 1.3614 | 0.2106 | 1.5359 |
| plus20 | 6.5512 | 1.3843 | 0.2113 | 1.5410 |

- Sensitivity Sharpe range: 1.3%

## Verdict against adoption criteria
1. Pooled net Sharpe > baseline by >0.01: FAIL (+0.0037)
2. Walk-forward win rate >= 8/12: FAIL (33.3%)
3. DSR >= 0.95: PASS (1.0000)
4. ±20% parameter perturbation Sharpe change < 20%: PASS (1.3%)
5. Turnover not significantly increased: PASS
