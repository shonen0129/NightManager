# A7: Walkforward Validation + Deflated Sharpe Ratio Report

## Configuration

- Training start: 2015-01-05

- Test periods: 2018–2026 (9 periods)

- Purge: 61 days, Embargo: 5 days

- N_trials for DSR: 55


## Per-Period Metrics

 Sharpe_net   AR_net  Vol_net       MDD  n_days period      start        end  elapsed_s
   6.148684 0.763967 0.124249 -0.063116     251   2018 2018-01-01 2018-12-31 114.561529
   7.392622 0.963063 0.130274 -0.029959     234   2019 2019-01-01 2019-12-31  92.734331
   9.402235 1.692007 0.179958 -0.032007     235   2020 2020-01-01 2020-12-31  81.156556
   6.694208 1.067261 0.159430 -0.055268     238   2021 2021-01-01 2021-12-31  55.749770
   8.642846 1.180698 0.136610 -0.029556     236   2022 2022-01-01 2022-12-31  55.531722
   4.696679 0.590044 0.125630 -0.055588     239   2023 2023-01-01 2023-12-31   0.859883
   7.257950 1.149939 0.158439 -0.056593     237   2024 2024-01-01 2024-12-31   0.677999
   7.681836 1.308981 0.170400 -0.033071     232   2025 2025-01-01 2025-12-31   0.508252
   8.766934 1.964597 0.224092 -0.053032     122   2026 2026-01-01 2026-07-13   0.337780


## Pooled Statistics

- Sharpe (net): 7.2856

- Annual Return: 113.85%

- Volatility: 15.63%

- Max DD: -6.31%

- N days: 2024

- Skewness: 1.0112

- Excess Kurtosis: 5.1142


## Deflated Sharpe Ratio

- DSR = 1.0000 (threshold: 0.95)

- N_trials = 55, Trials Sharpe Std = 0.5

- **PASS** (DSR ≥ 0.95)


## Per-Period Sharpe Stability

- Mean: 7.4093

- Std: 1.4531

- Range: [4.6967, 9.4022]

- Positive periods: 9/9

- Negative periods: 0/9


## Sensitivity Analysis (±20% all params)

 Sharpe_net   AR_net  Vol_net       MDD  n_days variant  multiplier
   8.715384 1.390612 0.159558 -0.063747    2734 minus20         0.8
   8.719114 1.391471 0.159589 -0.063116    2734    base         1.0
   8.718326 1.390084 0.159444 -0.061899    2734  plus20         1.2


## Verdict

**ROBUST**: DSR=1.0000≥0.95 after 55 trials correction

Strategy performance is statistically significant even after multiple testing bias correction.
All 9 walkforward periods are positive (0 negative periods).
Sensitivity range: 0.04% of mean Sharpe (extremely stable).
