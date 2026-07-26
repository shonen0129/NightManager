# Gârleanu & Pedersen (2013) Transaction-Cost-Aware MinVar Validation

**Date**: 2026-07-26  
**Experiment**: `scripts/experiments/validate_gp_transaction_cost.py`  
**Reference**: Gârleanu, N. and Pedersen, L. H. (2013). "Dynamic Trading with Predictable Returns and Transaction Costs." *The Journal of Finance*, 68(6), 2309-2340.

## 1. Hypothesis

Add a quadratic penalty on weight changes to the existing closed-form MinVar
optimizer so that "small signal changes do not trigger full rebalancing".  In a
one-period approximation the objective becomes:

```
min  alpha * w' Sigma w
     + (1-alpha) * ||w - w_signal||^2
     + lambda_tc * (w - w_prev)' Lambda (w - w_prev)
```

with closed-form solution `w = (alpha*Sigma + (1-alpha)*I + lambda_tc*Lambda)^-1
* ((1-alpha)*w_signal + lambda_tc*Lambda*w_prev)`.

We expected turnover to fall and net Sharpe to improve, especially if current
slippage/financing costs are materially eroding returns.

## 2. Methods

- **Model**: Residual-BLPX v1 (`configs/production/production_residual_blpx.yaml`)
- **Weight builder**: `experiments/transaction_cost_minvar.build_weights_minvar_gp`
- **Baseline**: `lambda_tc = 0.0` (identical to `build_weights_minvar`)
- **G&P variants**: `lambda_tc ∈ {0.5, 1.0, 2.0, 5.0}`
- **Cost matrix**: isotropic, `Lambda = I`
- **Overnight carry**: `alpha_long = 0.75`, `alpha_short = 0.50` (same as production config)
- **Backtest engine**: `BacktestEngine.run_backtest` with `n_jobs=1` (stateful `w_prev`)
- **Costs**: 5 bps one-way slippage, 2.5% buy interest, 1.15% borrow fee, 2 bps/day reverse fee
- **Period**: 2015-01-05 to 2026-07-24 (2,742 trading days)

## 3. Results

| name            | AR_net | AR_gross | Vol_net | Sharpe_net | Sharpe_gross | MDD     | Turnover | n_days |
|-----------------|-------:|---------:|--------:|-----------:|-------------:|--------:|---------:|-------:|
| baseline        | 1.3979 | 1.7398   | 0.1628  | **8.5864** | 10.6872      | -5.97%  | 1.6272   | 2742   |
| lambda_tc_0.50  | 1.2183 | 1.5487   | 0.1606  | 7.5876     | 9.6494       | -5.91%  | 1.4783   | 2742   |
| lambda_tc_1.00  | 1.1678 | 1.4978   | 0.1690  | 6.9089     | 8.8654       | -7.91%  | 1.4717   | 2742   |
| lambda_tc_2.00  | 1.1344 | 1.4648   | 0.1796  | 6.3164     | 8.1592       | -9.25%  | 1.4743   | 2742   |
| lambda_tc_5.00  | 1.1140 | 1.4450   | 0.1912  | 5.8262     | 7.5606       | -9.98%  | 1.4810   | 2742   |

*All annualized except MDD (decimal) and Turnover (average daily gross turnover).*

## 4. Key Findings

1. **No Sharpe improvement at any tested `lambda_tc`**.  The baseline (standard
   MinVar, `alpha=0.8`) dominates every G&P-penalized variant.
2. **Turnover falls only modestly** (1.63 → ~1.47-1.48).  Overnight holding
   (`alpha_long=0.75`, `alpha_short=0.5`) already carries a large fraction of
   the book, so the additional quadratic penalty has limited room to reduce
   trading.
3. **Risk-adjusted return falls faster than turnover**.  As `lambda_tc` rises,
   weights become sticky and lag signal changes.  Net volatility rises and MDD
   worsens, suggesting the penalty causes delayed exits from deteriorating
   positions.
4. **Gross return also declines**, confirming that the lost alpha is not simply
   a rebalancing-cost trade-off but a genuine signal degradation.

## 5. Verdict

**Not adopted.**

Under the current cost assumptions (5 bps one-way slippage, low financing /
borrow / reverse fees) and with the existing overnight-holding and MinVar
(`alpha=0.8`) smoothing, the Gârleanu & Pedersen quadratic transaction-cost
penalty **does not improve net Sharpe**.  Even a mild penalty (`lambda_tc=0.5`)
reduces Sharpe from 8.59 to 7.59.

## 6. Caveats and Next Steps

- **v1 proxy**: This experiment was run on the Residual-BLPX v1 model because
  full v2 gap-adjusted distribution matrices are not available for the entire
  backtest window.  Production v2 uses the same `build_weights_minvar` core, so
  the qualitative conclusion is likely transferable, but a v2-specific test
  should be run before a final decision.
- **Isotropic Lambda**: Only `Lambda = I` was tested.  A diagonal cost matrix
  scaled by estimated bid-ask spread or ADV could in principle give different
  results, but the observed pattern (sticky weights eroding alpha) is
  structural.
- **Linear vs quadratic costs**: Sprint 2 (`reports/sprint2_cost_aware_aum1m/`)
  previously found that a *linear* transaction-cost-aware MVO improved
  performance in an AUM 1M / high-spread scenario.  The quadratic G&P penalty
  evaluated here is a different, smoother formulation and behaves differently.
- **Cost regime**: If future slippage or borrow costs rise materially, the
  trade-off may flip.  Re-test with slippage ≥ 20 bps if cost assumptions change.

## 7. Artifacts

- Implementation: `src/experiments/transaction_cost_minvar.py`
- Validation script: `scripts/experiments/validate_gp_transaction_cost.py`
- Numerical results: `reports/transaction_cost_minvar/metrics_summary.csv`
- Per-variant weights & returns: `reports/transaction_cost_minvar/lambda_tc_*/`
