# A4: Ensemble IC Optimization Report

## Data: 4149 rows, start=2015-01-05, IC window=504


## Mean Daily IC per Component

raw_pca          0.217830
residual_pca     0.217559
raw_blpx         0.226966
residual_blpx    0.227514


## Metrics Comparison

 Sharpe_net   AR_net  Vol_net       MDD  n_days                       variant
   8.614776 1.449151 0.168217 -0.069478    2734         baseline_static_equal
   8.643059 1.451449 0.167932 -0.069286    2734          ic_optimal_delta0.25
   8.685078 1.457476 0.167814 -0.068241    2734           ic_optimal_delta0.5
   8.719114 1.391471 0.159589 -0.063116    2734 production_residual_blpx_only


## Verdict

IC-optimal weights do NOT improve Sharpe (best delta: +0.8%).

Recommend: keep static weights. Meta-learning deprecate candidate.


Note: Production (residual_blpx only) Sharpe=8.7191 is the real baseline.

This experiment uses equal-weight 4-component as comparison baseline.
