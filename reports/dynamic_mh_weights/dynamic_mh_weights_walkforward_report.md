# Dynamic Multi-Horizon Weighting: Walk-Forward Validation Report

**Date**: 2026-07-24
**Experiment**: Discounted-MSFE combination weights for multi-horizon signal blending
**References**: Rapach, Strauss & Zhou (2010, RFS); Timmermann (2006, Handbook of Economic Forecasting); Bailey & López de Prado (2014, DSR)

---

## 1. Hypothesis

Replace fixed multi-horizon blend weights `[0.8, 0.1, 0.1]` (h=1, 3, 5) with dynamic weights based on discounted mean squared forecast errors (MSFE). The academic literature predicts that combination forecasts using inverse-MSFE weights can outperform fixed weights out-of-sample, with a single discount rate parameter controlling the recency bias.

**Weight formula**:
```
MSFE_{h,t} = Σ_{s<t} δ^{t-s} · e_{h,s}²
w_{h,t}    = (1/MSFE_{h,t}) / Σ_h (1/MSFE_{h,t})
```

## 2. Experimental Setup

- **Backtest period**: 2020-01-06 to 2026-07-14 (1539 trading days)
- **Gap matrices**: `live/pipeline_data/gap_adjusted_distribution/20260714_085119/` (h1, h3, h5 all available)
- **Model**: ProductionV2 (Residual-BLPX-RA v2) with MinVar weights, RuleD dynamic gross, rank reversal overlay
- **Costs**: net (slippage 5bps/side + financing + borrow + reverse)
- **Discount rates tested**: δ = {0.90, 0.95, 0.99, 1.00}
- **Baseline**: fixed weights [0.8, 0.1, 0.1]
- **Walk-forward windows**: 7 yearly non-overlapping windows (2020–2026)
- **Min history**: 60 days (equal weights before warmup)
- **Trials for DSR**: 5 (4 dynamic + 1 fixed baseline)

## 3. Full-Period Results

| Config | Sharpe (net) | AR (net) | Vol | Max DD | Turnover |
|--------|-------------|----------|-----|--------|----------|
| **fixed_0.8_0.1_0.1** | **7.496** | **153.7%** | 20.5% | -7.47% | 1.298 |
| dynamic_delta0.90 | 7.157 | 134.8% | 18.8% | -7.91% | 0.969 |
| dynamic_delta0.95 | 7.130 | 134.4% | 18.8% | -8.30% | 0.959 |
| dynamic_delta0.99 | 7.169 | 134.3% | 18.7% | -7.03% | 0.961 |
| dynamic_delta1.00 | 7.241 | 137.7% | 19.0% | -7.07% | 1.001 |

**Key observations**:
- Fixed weights baseline has the highest Sharpe (7.496) and annual return (153.7%)
- All dynamic variants have lower Sharpe (7.13–7.24, **-3.4% to -4.9%**)
- Dynamic weights reduce turnover by ~25% (0.96–1.00 vs 1.30)
- Dynamic weights reduce volatility (18.7–19.0% vs 20.5%)
- Max DD is comparable across all configs (-7.0% to -8.3%)

## 4. Walk-Forward Yearly Sharpe

| Year | fixed | δ=0.90 | δ=0.95 | δ=0.99 | δ=1.00 |
|------|-------|--------|--------|--------|--------|
| 2020 | 9.14 | 7.71 | 7.80 | 7.90 | 7.79 |
| 2021 | 7.07 | **7.56** | **7.54** | **7.59** | **7.43** |
| 2022 | 8.96 | 8.57 | 8.58 | 8.37 | 8.31 |
| 2023 | 5.10 | **5.19** | **5.20** | **5.23** | **5.38** |
| 2024 | 7.16 | 6.68 | 6.75 | 6.90 | 7.05 |
| 2025 | 7.69 | 7.13 | 7.01 | 7.17 | 7.54 |
| 2026 | 10.37 | **10.59** | 10.18 | 9.78 | **10.23** |

### Consistency Analysis (vs fixed baseline)

| Config | Windows positive | Mean Δ Sharpe |
|--------|-----------------|---------------|
| dynamic_delta0.90 | 3/7 | -0.297 |
| dynamic_delta0.95 | 2/7 | -0.349 |
| dynamic_delta0.99 | 2/7 | -0.366 |
| dynamic_delta1.00 | 2/7 | -0.252 |

**No dynamic variant achieves majority positive windows.** The dynamic scheme outperforms in 2021 and 2023 (low-volatility years where diversification across horizons helps) but underperforms in 2020, 2022, 2024, and 2025.

## 5. Sensitivity Analysis

| Discount rate | Sharpe | AR | Max DD | Turnover |
|--------------|--------|-----|--------|----------|
| 0.90 | 7.157 | 134.8% | -7.91% | 0.969 |
| 0.95 | 7.130 | 134.4% | -8.30% | 0.959 |
| 0.99 | 7.169 | 134.3% | -7.03% | 0.961 |
| 1.00 | 7.241 | 137.7% | -7.07% | 1.001 |
| (fixed) | 7.496 | 153.7% | -7.47% | 1.298 |

Sensitivity is low: Sharpe ranges from 7.13 to 7.24 across discount rates (±0.05 around mean). The δ=1.00 (equal weight on all history, no decay) is marginally best among dynamic variants, suggesting that recency bias does not help.

## 6. Deflated Sharpe Ratio

| Config | Sharpe | DSR |
|--------|--------|-----|
| fixed_0.8_0.1_0.1 (best) | 7.496 | 1.000 |
| Baseline DSR | 7.496 | 1.000 |

With n_trials=5 and n_obs=1539, the DSR correction is minimal. Both best and baseline DSR = 1.000, indicating the observed Sharpe is statistically significant even after trial count adjustment.

## 7. Leakage Audit

| Check | Status |
|-------|--------|
| forecast_errors_finite | FAIL* |
| msfe_no_future_leakage | **PASS** |
| weights_finite | **PASS** |
| weights_sum_to_one | **PASS** |
| weights_in_range | **PASS** |
| equal_weights_warmup | **PASS** |
| v2_leakage_audit_passes | **PASS** |
| **overall_status** | **PASS** |

*`forecast_errors_finite` fails because some dates have NaN forecast errors due to missing gap matrices for h3/h5 on certain dates. This is a data availability issue, not a leakage concern. The critical leakage check (`msfe_no_future_leakage`) passes: perturbing future errors does not change past MSFE values.

## 8. Dynamic Weight Behavior

The dynamic weights shift significantly from the fixed baseline:
- **h1 weight**: ranges 0.20–0.58 (vs fixed 0.80) — substantially lower
- **h3 weight**: ranges 0.22–0.46 (vs fixed 0.10) — substantially higher
- **h5 weight**: ranges 0.15–0.34 (vs fixed 0.10) — higher

The MSFE-based scheme equalizes weights toward 1/3 each, as h3 and h5 signals have comparable (or lower) forecast errors to h1. This diversification reduces turnover but also reduces the alpha contribution from the h1 signal, which appears to be the strongest predictor.

## 9. Conclusion

**The discounted-MSFE dynamic weighting scheme does NOT improve upon fixed weights.**

- Sharpe degradation: -3.4% to -4.9% across all discount rates
- Walk-forward consistency: only 2–3/7 windows positive (vs 4–5/7 negative)
- Turnover reduction: ~25% (beneficial but insufficient to offset return loss)
- Sensitivity: low (±0.05 Sharpe across δ), but all below baseline
- No leakage detected (strictly historical error computation verified)

**Recommendation**: Do not adopt dynamic MH weights for production. The fixed [0.8, 0.1, 0.1] weights, which heavily favor the h=1 signal, appear to be well-calibrated. The h=1 gap-adjusted signal carries the strongest predictive content, and the MSFE scheme's equalization toward 1/3 dilutes this edge.

**Possible follow-up**: The turnover reduction is notable (0.96 vs 1.30). A shrinkage variant `(1-λ)·w_MSFE + λ·w_fixed` with high λ (e.g. 0.8) could preserve most of the h1 concentration while gaining minor adaptivity. However, this adds a second parameter and the potential improvement is marginal given the Sharpe gap.

## 10. Output Files

- `outputs/experiments/dynamic_mh_weights/full_metrics.csv` — full-period metrics
- `outputs/experiments/dynamic_mh_weights/walkforward_yearly_results.csv` — yearly window results
- `outputs/experiments/dynamic_mh_weights/sensitivity_analysis.csv` — discount rate sensitivity
- `outputs/experiments/dynamic_mh_weights/dynamic_weights_delta099.csv` — weight timeseries
- `outputs/experiments/dynamic_mh_weights/equity_curves.png` — equity curve comparison
- `outputs/experiments/dynamic_mh_weights/yearly_sharpe_by_config.png` — yearly Sharpe plot
- `outputs/experiments/dynamic_mh_weights/yearly_sharpe_delta.png` — yearly Sharpe delta vs baseline
- `outputs/experiments/dynamic_mh_weights/dynamic_weights_timeseries.png` — weight evolution plot
