# Phase 2.5 Walk-Forward Report (EMA + stronger regularization)
**Config**: EMA span=3.0, n_estimators=100, num_leaves=20, reg_lambda=1.0

## Pooled OOS performance
- Periods: 2022, 2023, 2024
- Total OOS days: 709
- Baseline Sharpe: 5.9664
- Overlay Sharpe: 5.8939
- Mean daily return difference: 0.000037
- Walk-forward win rate: 66.7% (2/3)
- DSR (12 trials): 1.0000
- Mean turnover: base=1.5374, overlay=1.5414

## Per-fold metrics
| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |
|------|-------------|----------------|---------|------------|----------|-------------|---------|
| 2022 | 7.7706 | 7.4279 | 1.4103 | 1.4036 | -0.0412 | -0.0420 | 0.8611 |
| 2023 | 3.6740 | 3.7001 | 0.5939 | 0.6116 | -0.0492 | -0.0517 | 0.2985 |
| 2024 | 6.3941 | 6.4357 | 1.3513 | 1.3675 | -0.0839 | -0.0869 | 0.4064 |

## Sensitivity (2024 fold, ±20% hyperparameters)
| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |
|---------|----------------|------------|-------------|----------|
| minus20 | 6.3767 | 1.3467 | 0.2112 | 1.5325 |
| plus20 | 6.4489 | 1.3722 | 0.2128 | 1.5351 |

- Sensitivity Sharpe range: 1.1%

## Verdict against adoption criteria
1. Pooled net Sharpe > baseline by >0.01: FAIL (-0.0725)
2. Walk-forward win rate >= 8/12: PASS (66.7%)
3. DSR >= 0.95: PASS (1.0000)
4. ±20% parameter perturbation Sharpe change < 20%: PASS (1.1%)
5. Turnover not significantly increased: PASS
