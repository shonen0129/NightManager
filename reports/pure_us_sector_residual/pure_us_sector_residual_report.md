# Pure US Sector Component Signal — Backtest Report

- Config: `/Users/shonen/leadlag/configs/production/production.yaml`
- Backtest period: 2015-01-05 → latest
- US residualization: beta_window=60, gamma=1.0
- Fractional differencing: d=0.1
- Benchmark proxy: equal-weight average of the 15 US sector/style ETFs

## Overall Metrics

| Metric | Baseline (Residual-BLPX, raw US + FD) | Pure US Sector Residual + FD |
|---|---|---|
| AR_net | 1.3797 | 1.3481 |
| AR_gross | 1.7221 | 1.6904 |
| Vol_net | 0.1599 | 0.1610 |
| Sharpe_net | 8.6285 | 8.3732 |
| Sharpe_gross | 10.7716 | 10.5025 |
| Sharpe_monthly | 4.6389 | 4.6162 |
| MDD | -0.0597 | -0.0598 |
| Turnover | 1.6262 | 1.6206 |
| GrossExp | 2.0000 | 2.0000 |
| Mean_Rank_IC | 0.2263 | 0.2197 |
| ICIR | 12.5966 | 12.3105 |
| IC_positive_rate | 0.7819 | 0.7764 |

- Baseline elapsed: 111.6s
- Pure elapsed: 102.0s

## Year-by-Year Net Sharpe

| Year | Baseline | Pure | Δ |
|---|---|---|---|
| 2015 | 14.1584 | 12.9124 | -1.2461 |
| 2016 | 16.5527 | 15.7594 | -0.7933 |
| 2017 | 10.1524 | 10.2036 | +0.0512 |
| 2018 | 6.0504 | 5.8552 | -0.1952 |
| 2019 | 7.4301 | 7.6871 | +0.2571 |
| 2020 | 9.4590 | 9.1194 | -0.3396 |
| 2021 | 6.6007 | 5.8665 | -0.7342 |
| 2022 | 8.3483 | 8.5263 | +0.1780 |
| 2023 | 4.4793 | 4.8431 | +0.3639 |
| 2024 | 7.1053 | 7.1307 | +0.0254 |
| 2025 | 7.6976 | 7.0072 | -0.6904 |
| 2026 | 8.2957 | 8.5115 | +0.2159 |

## Year-by-Year AR (net, annualized)

| Year | Baseline | Pure | Δ |
|---|---|---|---|
| 2015 | 219.14% | 201.62% | -17.52% |
| 2016 | 305.42% | 289.67% | -15.75% |
| 2017 | 109.22% | 109.60% | +0.39% |
| 2018 | 75.09% | 76.00% | +0.91% |
| 2019 | 96.76% | 99.76% | +3.00% |
| 2020 | 171.04% | 172.59% | +1.56% |
| 2021 | 104.34% | 96.48% | -7.86% |
| 2022 | 115.67% | 114.09% | -1.58% |
| 2023 | 55.81% | 59.67% | +3.85% |
| 2024 | 113.02% | 113.24% | +0.22% |
| 2025 | 129.80% | 124.92% | -4.88% |
| 2026 | 186.05% | 187.60% | +1.55% |

## Interpretation & Next Steps

- The **Pure US Sector Component** removes the common US market factor before
  fractional differencing and BLPX projection.
- A positive Δ in Sharpe/AR suggests the macro purification adds predictive
  information for next-day Japanese sector returns.
- If Δ is negative or unstable year-to-year, the macro component may already be
  captured by the gap/TOPIX residualization and the extra orthogonalization
  destroys signal.