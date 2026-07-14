# A1: Adaptive Gap Coefficient Report

## Data: 4149 rows, start=2015-01-05, β_rev_oc window=252


## β_rev_oc Statistics

Mean: -0.2319 (expect ~-0.23)

Std:  0.2039

Theory: c_t = 1 + β_rev_oc ≈ 0.7681 (vs current 0.70)


## Metrics Comparison

 Sharpe_net   AR_net  Vol_net       MDD  n_days             variant
   8.719114 1.391471 0.159589 -0.063116    2734 baseline_fixed_0.70
   8.703077 1.387911 0.159474 -0.063342    2734 adaptive_lambda0.25
   8.704095 1.385834 0.159216 -0.063512    2734  adaptive_lambda0.5
   8.684805 1.381332 0.159052 -0.063698    2734 adaptive_lambda0.75


## Verdict

Adaptive c_t does NOT improve Sharpe (best: -0.2%).

Theory: 1+β_rev_oc ≈ 0.77 is close to current 0.70.

The small gap (0.07) is within estimation noise → no benefit from adaptation.

Recommend: keep fixed gap_open_coef=0.70.
