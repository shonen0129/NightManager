# Phase 2.5 Walk-Forward Report (EMA + stronger regularization)
**Config**: EMA span=3.0, n_estimators=100, num_leaves=20, reg_lambda=1.0

## Pooled OOS performance
- Periods: 2022, 2023, 2024
- Total OOS days: 709
- Baseline Sharpe: 5.9664
- Overlay Sharpe: 5.9500
- Mean daily return difference: 0.000034
- Walk-forward win rate: 33.3% (1/3)
- DSR (12 trials): 1.0000
- Mean turnover: base=1.5374, overlay=1.5387

## Per-fold metrics
| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |
|------|-------------|----------------|---------|------------|----------|-------------|---------|
| 2022 | 7.7706 | 7.6931 | 1.4103 | 1.4130 | -0.0412 | -0.0421 | 0.8390 |
| 2023 | 3.6740 | 3.7229 | 0.5939 | 0.6082 | -0.0492 | -0.0512 | 0.0839 |
| 2024 | 6.3941 | 6.3708 | 1.3513 | 1.3593 | -0.0839 | -0.0849 | 0.4737 |

## Sensitivity (2024 fold, ±20% hyperparameters)
| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |
|---------|----------------|------------|-------------|----------|
| minus20 | 6.3549 | 1.3542 | 0.2131 | 1.5301 |
| plus20 | 6.3630 | 1.3575 | 0.2133 | 1.5301 |

- Sensitivity Sharpe range: 0.1%

## Verdict against adoption criteria
1. Pooled net Sharpe > baseline by >0.01: FAIL (-0.0164)
2. Walk-forward win rate >= 8/12: FAIL (33.3%)
3. DSR >= 0.95: PASS (1.0000)
4. ±20% parameter perturbation Sharpe change < 20%: PASS (0.1%)
5. Turnover not significantly increased: PASS
