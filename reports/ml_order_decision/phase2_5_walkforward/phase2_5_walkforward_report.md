# Phase 2.5 Walk-Forward Report (EMA + stronger regularization)
**Config**: EMA span=3.0, n_estimators=100, num_leaves=20, reg_lambda=1.0

## Pooled OOS performance
- Periods: 2022, 2023, 2024
- Total OOS days: 709
- Baseline Sharpe: 5.9664
- Overlay Sharpe: 5.9510
- Mean daily return difference: 0.000066
- Walk-forward win rate: 33.3% (1/3)
- DSR (12 trials): 1.0000
- Mean turnover: base=1.5374, overlay=1.5402

## Per-fold metrics
| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |
|------|-------------|----------------|---------|------------|----------|-------------|---------|
| 2022 | 7.7706 | 7.6959 | 1.4103 | 1.4362 | -0.0412 | -0.0450 | 0.2484 |
| 2023 | 3.6740 | 3.7097 | 0.5939 | 0.6073 | -0.0492 | -0.0507 | 0.0995 |
| 2024 | 6.3941 | 6.3625 | 1.3513 | 1.3604 | -0.0839 | -0.0831 | 0.4714 |

## Sensitivity (2024 fold, ±20% hyperparameters)
| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |
|---------|----------------|------------|-------------|----------|
| minus20 | 6.3832 | 1.3601 | 0.2131 | 1.5293 |
| plus20 | 6.3838 | 1.3644 | 0.2137 | 1.5317 |

- Sensitivity Sharpe range: 0.0%

## Verdict against adoption criteria
1. Pooled net Sharpe > baseline by >0.01: FAIL (-0.0154)
2. Walk-forward win rate >= 8/12: FAIL (33.3%)
3. DSR >= 0.95: PASS (1.0000)
4. ±20% parameter perturbation Sharpe change < 20%: PASS (0.0%)
5. Turnover not significantly increased: PASS
