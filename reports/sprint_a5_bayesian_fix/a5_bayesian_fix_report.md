# A5: BayesianBLPX Kalman Fix Report

## Data: 4149 rows, start=2015-01-05


## Fixes Applied

1. **R3 leak fix**: `y_jp_target[i]` → `y_jp_target[i-1]` (1-day lag)

2. **Q estimation**: non-overlapping subsampling (every blp_window steps)

3. **R estimation**: bias-corrected theoretical value `(Var(ΔB̂) - Q) * window / 2`


## Metrics Comparison

 Sharpe_net   AR_net  Vol_net       MDD  n_days                 variant
   8.719114 1.391471 0.159589 -0.063116    2734   baseline_blp_enhanced
   8.496310 1.424817 0.167698 -0.073046    2734     bayesian_ic_postfix
   8.498182 1.425780 0.167775 -0.073018    2734 bayesian_kalman_postfix
   6.687932 1.193034 0.178386 -0.115836    2734 bayesian_cs_var_postfix


## Verdict

- Bayesian[ic]: Sharpe 8.4963 vs baseline 8.7191 (-2.6%)

- Bayesian[kalman]: Sharpe 8.4982 vs baseline 8.7191 (-2.5%)

- Bayesian[cs_var]: Sharpe 6.6879 vs baseline 8.7191 (-23.3%)


No Bayesian mode improves over baseline BLPEnhanced.

Recommend: deprecate BayesianBLPX (kalman/cs_var/ic modes) to reduce code complexity.

Note: R3 leak fix is still correct regardless of performance impact.
