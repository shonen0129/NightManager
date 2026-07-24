# Meta-Labeling Walkforward Validation Report

## Overview
- **Method**: López de Prado (2018) Ch.3 Meta-Labeling
- **Primary model**: Residual-BLPX (production config)
- **Meta-model**: Logistic Regression on ex-ante features
- **Gating**: Gross exposure scaled by P(success)

## Baseline vs Meta-Labeled (Pooled 2018-2026, OOS)

| Metric | Baseline | Meta-Labeled | Delta |
|--------|----------|-------------|-------|
| Sharpe (net) | 7.1873 | 7.2376 | +0.0503 |
| Annual Return | 112.54% | 112.92% | +0.37% |
| Volatility | 15.66% | 15.60% | |
| Max DD | -5.97% | -5.97% | |
| N days | 2023 | 2023 | |

## Meta-Labeling Statistics
- Avg gross multiplier: 0.998
- Days fully gated (mult=0): 1 (0.0%)
- Days full exposure (mult=1): 2016 (99.7%)
- Brier score: 0.2283 (0.25=random, 0=perfect)

## Calibration Curve

| Bin Center | Observed Freq | Count |
|------------|---------------|-------|
| 0.05 | nan | 0 |
| 0.15 | nan | 0 |
| 0.25 | nan | 0 |
| 0.35 | 1.000 | 1 |
| 0.45 | 0.500 | 2 |
| 0.55 | 0.747 | 1409 |
| 0.65 | 0.782 | 55 |
| 0.75 | 0.745 | 204 |
| 0.85 | 0.789 | 332 |
| 0.95 | 0.800 | 20 |

## Walkforward Per-Period (Meta-Labeled)

 Sharpe_net   AR_net  Vol_net       MDD  n_days period
   6.589840 0.781077 0.118527 -0.059684     251   2018
   7.430067 0.967607 0.130229 -0.029336     233   2019
   9.459033 1.710362 0.180818 -0.031639     234   2020
   6.600712 1.043427 0.158078 -0.055413     237   2021
   8.348298 1.156740 0.138560 -0.030360     235   2022
   4.479271 0.558139 0.124605 -0.055450     238   2023
   7.105267 1.130161 0.159060 -0.058334     236   2024
   7.697615 1.298033 0.168628 -0.033508     231   2025
   8.455649 1.906998 0.225529 -0.054902     128   2026

- Mean Sharpe: 7.3518
- Std Sharpe: 1.4245
- Positive periods: 9/9

## Deflated Sharpe Ratio
- DSR = 1.0000 (threshold: 0.95)
- N_trials = 60, Trials Sharpe Std = 0.5
- Skewness: 1.1104, Excess Kurtosis: 5.6578
- **PASS** (DSR ≥ 0.95)

## Sensitivity Analysis (±20% prob thresholds)

 Sharpe_net   AR_net  Vol_net       MDD  n_days variant
   7.188166 1.125560 0.156585 -0.059684    2023 minus20
   6.817220 0.735195 0.107844 -0.059684    2023    base
   3.779158 0.336429 0.089022 -0.059684    2023  plus20

- Sensitivity range: 57.5%

## Verdict
**NOT ADOPTED**: Sharpe +0.0503 (marginal). Meta-model gates only 0.0% of days; Brier 0.2283 barely better than random (0.25). Calibration poor (predicted ~55% vs actual ~75%). RuleD PIT binning remains the primary gating mechanism.
⚠️ Parameter fragility: ±20% perturbation changes Sharpe by 57.5%
