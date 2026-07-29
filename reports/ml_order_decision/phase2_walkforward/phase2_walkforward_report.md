# Phase 2 Walk-Forward Validation + DSR (LightGBM Overlay)
## Pooled OOS performance
- Periods: 2022, 2023, 2024
- Total OOS days: 709
- Baseline Sharpe (pooled): 5.9664
- Overlay Sharpe (pooled): 5.9916
- Overlay AR: 1.1491
- Overlay Vol: 0.1918
- Overlay MDD: -0.0856
- Mean daily return difference (O - B): 0.000133
- Walk-forward win rate: 33.3% (1/3)
- DSR (12 trials): 1.0000
- Mean turnover: base=1.5374, overlay=1.5466

## Per-fold metrics
| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |
|------|-------------|----------------|---------|------------|----------|-------------|---------|
| 2022 | 7.7706 | 7.7101 | 1.4103 | 1.4576 | -0.0412 | -0.0462 | 0.0433 |
| 2023 | 3.6740 | 3.6281 | 0.5939 | 0.5963 | -0.0492 | -0.0530 | 0.7699 |
| 2024 | 6.3941 | 6.5194 | 1.3513 | 1.3995 | -0.0839 | -0.0856 | 0.0089 |

## Sensitivity (2024 fold, ±20% hyperparameters)
| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |
|---------|----------------|------------|-------------|----------|
| minus20 | 6.4306 | 1.3646 | 0.2122 | 1.5402 |
| plus20 | 6.5716 | 1.3991 | 0.2129 | 1.5393 |

- Sensitivity Sharpe range: 2.1%

## Verdict against adoption criteria
1. Pooled net Sharpe > baseline by >0.5 SE (≈0.01): PASS (+0.0252)
2. Walk-forward win rate >= 8/12: FAIL (33.3%)
3. DSR >= 0.95: PASS (1.0000)
4. ±20% parameter perturbation Sharpe change < 20%: PASS (2.1%)
5. Turnover not significantly increased: PASS
