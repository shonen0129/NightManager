# Phase 2.5 Walk-Forward Report (EMA + stronger regularization)
**Config**: EMA span=0.0, n_estimators=100, num_leaves=20, reg_lambda=1.0, per_ticker_interactions=True

## Pooled OOS performance
- Periods: 2022, 2023, 2024
- Total OOS days: 709
- Baseline Sharpe: 5.9664
- Overlay Sharpe: 5.9881
- Mean daily return difference: 0.000087
- Walk-forward win rate: 66.7% (2/3)
- DSR (12 trials): 1.0000
- Mean turnover: base=1.5374, overlay=1.5447

## Per-fold metrics
| Year | Base Sharpe | Overlay Sharpe | Base AR | Overlay AR | Base MDD | Overlay MDD | p-value |
|------|-------------|----------------|---------|------------|----------|-------------|---------|
| 2022 | 7.7706 | 7.7762 | 1.4103 | 1.4346 | -0.0412 | -0.0420 | 0.1547 |
| 2023 | 3.6740 | 3.6729 | 0.5939 | 0.6051 | -0.0492 | -0.0554 | 0.2562 |
| 2024 | 6.3941 | 6.4503 | 1.3513 | 1.3796 | -0.0839 | -0.0828 | 0.0315 |

## Sensitivity (2024 fold, ±20% hyperparameters)
| Variant | Overlay Sharpe | Overlay AR | Overlay Vol | Turnover |
|---------|----------------|------------|-------------|----------|
| minus20 | 6.4158 | 1.3750 | 0.2143 | 1.5370 |
| plus20 | 6.5061 | 1.3938 | 0.2142 | 1.5368 |

- Sensitivity Sharpe range: 1.4%

## Verdict against adoption criteria
1. Pooled net Sharpe > baseline by >0.01: PASS (+0.0217)
2. Walk-forward win rate >= 8/12: PASS (66.7%)
3. DSR >= 0.95: PASS (1.0000)
4. ±20% parameter perturbation Sharpe change < 20%: PASS (1.4%)
5. Turnover not significantly increased: PASS
