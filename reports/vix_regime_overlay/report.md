# VIX Regime Overlay Experiment Report

- Date: 2026-08-01 23:58
- Hypothesis: US VIX high -> US-led (maintain gross); JP VIX high while US VIX low -> Japan-led (reduce gross)
- Model: SectorRelativeEnsembleBLPEnhancedModel (Residual-BLPX, V1-equivalent)
- Config: configs/production/production.yaml
- Period: 2018-04-01 ~ 2024-12-31
- VIX data: ^VIX (US), ^NKVI.OS (Nikkei VI / Japan)
- Method: 60-day rolling z-score on log VIX; spread = JP_z - US_z

## Aggregate Results

| name | AR | RISK | Sharpe | MDD | Total Return | mean_gross_exp | median_gross_exp | mean_turnover | mean_cost | n_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 104.69% | 22.58% | 4.6357 | -7.25% | 825.2566 | 2.0 | 2.0 | 1.6071560994292533 | 0.0013906629125057 | 1603 |
| discrete_jp_led_w60 | 83.63% | 18.78% | 4.4528 | -6.50% | 225.6828 | 1.6805988771054272 | 2.0 | 1.3570460508541704 | 0.0011706126741175 | 1603 |
| discrete_global_stress_w60 | 86.87% | 19.25% | 4.5135 | -6.78% | 276.4020 | 1.7504678727386151 | 2.0 | 1.4105124593951064 | 0.0012192677405772 | 1603 |
| continuous_spread_w60 | 95.03% | 21.14% | 4.4943 | -6.19% | 455.5513 | 1.8398156749920145 | 1.9935857243334636 | 1.4806971326296907 | 0.0012790642372826 | 1603 |

## Walk-Forward Year-by-Year Sharpe

| year | baseline | continuous_spread_w60 | discrete_global_stress_w60 | discrete_jp_led_w60 |
| --- | --- | --- | --- | --- |
| 2018 | 4.88 | 4.68 | 4.81 | 4.41 |
| 2019 | 5.31 | 5.31 | 5.72 | 5.70 |
| 2020 | 6.67 | 6.30 | 6.16 | 5.94 |
| 2021 | 3.83 | 3.64 | 3.73 | 3.76 |
| 2022 | 8.49 | 8.86 | 7.40 | 7.92 |
| 2023 | 3.52 | 3.15 | 3.69 | 3.44 |
| 2024 | 4.89 | 5.09 | 4.79 | 4.68 |

## Paired t-test vs Baseline (daily net returns)

| variant | n_days | mean diff (bps) | t-stat | p-value | win rate |
| --- | --- | --- | --- | --- | --- |
| discrete_jp_led_w60 | 1603 | -8.24 | -10.044 | 0.0000 | 17.3% |
| discrete_global_stress_w60 | 1603 | -6.95 | -9.190 | 0.0000 | 12.7% |
| continuous_spread_w60 | 1603 | -3.78 | -10.337 | 0.0000 | 27.0% |

## Conclusion

**No overlay improved net Sharpe.** Baseline Sharpe = 4.64. All VIX regime overlays reduced both annualized return and Sharpe, while marginally improving MDD. The hypothesis that Japan-led VIX spikes are a reliable signal to reduce gross was not supported in this specification.

## Notes / Next Steps

- Overlay tested on V1 BLPX model, not the production V2 (Residual-BLPX-RA v2).
- Thresholds (spread_z > 0.8, high_vix_z > 0.5, multiplier 0.5) are ad-hoc; a parameter sweep with Deflated Sharpe is recommended before drawing final conclusions.
- Cost breakdown and gross/net decomposition were not saved in this run; re-run `experiment_vix_regime_overlay.py` with cost CSV flags for a full audit.
