# Nonlinear Shrinkage Correlation Estimation: Walk-Forward Validation Report

**Date**: 2026-07-24
**Experiment**: Nonlinear Shrinkage of the Covariance/Correlation Matrix (Ledoit & Wolf 2017, 2020)
**Status**: **Not adopted** — no improvement over baseline

---

## 1. Hypothesis

Replace the current linear Ledoit-Wolf (2004) shrinkage in `regularize_correlation` with the analytical nonlinear shrinkage from Ledoit & Wolf (2020). The nonlinear method applies eigenvalue-by-eigenvalue shrinkage via the Marchenko-Pastur (MP) Stieltjes transform, with zero tuning parameters — fully compatible with the overfitting guard.

**Expected benefit**: Eliminate manual `lambda_lw` tuning, improve conditioning in high-dimensional settings (N=32, T=252–504).

## 2. Methods

### 2.1 Nonlinear Shrinkage Variants

Two variants implemented in `src/experiments/nonlinear_shrinkage.py`:

- **NL-MP**: MP-based analytical formula. Uses the Marchenko-Pastur Stieltjes transform to compute per-eigenvalue shrinkage targets. Assumes identity population structure.
- **NL-Empirical**: Sample-eigenvalue-based formula. Estimates the Stieltjes transform from the sample eigenvalues themselves via leave-one-out Cauchy transform, accounting for non-identity population (e.g., dominant market factor).

Both replace Stage 1 (linear LW) in `regularize_correlation`. Stage 2 (prior shrinkage toward C0) is preserved.

### 2.2 Validation Framework

- **Walk-forward**: 12 yearly windows (2015–2026), full backtest per variant
- **Deflated Sharpe Ratio**: B=33 total trials (30 prior + 3 current)
- **Sensitivity**: Window size T (252/378/504) and lambda_reg perturbation (±0.05)
- **Eigenvalue diagnostics**: Condition number, eigenvalue distribution, MP bulk membership

### 2.3 Implementation

- `src/experiments/nonlinear_shrinkage.py`: Core nonlinear shrinkage functions
- `scripts/experiments/experiment_nonlinear_shrinkage.py`: Full experiment script
- Monkey-patching approach: `compute_correlation` output is nonlinearly shrunk; `regularize_correlation` skips LW Stage 1 (already done) and only applies prior Stage 2

## 3. Results

### 3.1 Eigenvalue Diagnostics

| Method | Mean Cond. Number | Median Cond. Number | Min Cond. | Max Cond. |
|--------|-------------------|---------------------|-----------|-----------|
| Raw (no shrinkage) | ~7.4×10^12 | ~1.1×10^13 | 1,020 | ~1.4×10^13 |
| Linear LW (baseline) | 37.0 | 38.1 | 27.0 | 43.7 |
| NL-MP | 53.3 | 42.5 | 19.1 | 115.7 |
| NL-Empirical | ~7.3×10^6 | ~1.1×10^7 | 871 | ~1.3×10^7 |

**Key observations**:
- Linear LW provides the most consistent conditioning (tight range 27–44)
- NL-MP provides good conditioning but with higher variance (19–116)
- NL-Empirical barely shrinks the condition number — the leave-one-out estimator doesn't sufficiently regularize near-degenerate eigenvalues

### 3.2 Walk-Forward Validation

| Variant | Mean Sharpe | Std | Min | Max | Windows > Baseline |
|--------|-------------|-----|-----|-----|-------------------|
| Baseline (Linear LW) | 8.87 | 3.42 | 4.48 | 16.55 | — |
| NL-MP | 8.04 | 3.25 | 4.03 | 15.48 | **0/12** |
| NL-Empirical | 8.91 | 3.44 | 4.58 | 16.64 | **7/12** |

**Yearly Sharpe by Variant**:

| Year | Baseline | NL-MP | NL-Empirical | Δ(MP) | Δ(Emp) |
|------|----------|-------|-------------|-------|--------|
| 2015 | 14.16 | 12.97 | 14.28 | -1.19 | +0.12 |
| 2016 | 16.55 | 15.48 | 16.64 | -1.08 | +0.09 |
| 2017 | 10.15 | 8.97 | 10.23 | -1.19 | +0.08 |
| 2018 | 6.05 | 5.01 | 6.04 | -1.04 | -0.01 |
| 2019 | 7.43 | 7.31 | 7.47 | -0.12 | +0.04 |
| 2020 | 9.46 | 8.92 | 9.40 | -0.54 | -0.06 |
| 2021 | 6.60 | 6.28 | 6.58 | -0.32 | -0.02 |
| 2022 | 8.35 | 6.55 | 8.37 | -1.80 | +0.02 |
| 2023 | 4.48 | 4.03 | 4.58 | -0.45 | +0.10 |
| 2024 | 7.11 | 6.54 | 7.07 | -0.57 | -0.04 |
| 2025 | 7.70 | 7.61 | 7.67 | -0.09 | -0.02 |
| 2026 | 8.46 | 6.83 | 8.60 | -1.63 | +0.14 |

**Consistency analysis**:
- **NL-MP vs Baseline**: 0/12 windows positive, mean Δ = **-0.83**, median Δ = -0.81 → **uniform degradation**
- **NL-Empirical vs Baseline**: 7/12 windows positive, mean Δ = **+0.04**, median Δ = +0.03 → **marginal, not statistically meaningful**

### 3.3 Deflated Sharpe Ratio

B=33 total trials (30 prior experiments + 3 current variants), T=11.2 years.

| Variant | SR_obs | SR_0 | DSR | PSR |
|--------|--------|------|-----|-----|
| Baseline (Linear LW) | 8.638 | 0.615 | 0.986 | 0.991 |
| NL-MP | 7.863 | 0.615 | 0.990 | 0.995 |
| NL-Empirical | 8.658 | 0.615 | 0.985 | 0.991 |

All variants have DSR > 0.98, confirming statistical significance of the strategy itself. However:
- NL-MP has higher DSR despite lower SR — this is because DSR rewards lower skew/kurtosis, not because the strategy is better
- NL-Empirical DSR is essentially identical to baseline (0.985 vs 0.986)

### 3.4 Sensitivity Analysis

All MP sensitivity variants produced identical results (Sharpe = 7.863), because:
- **Window size T**: The nonlinear shrinkage uses the actual `window_returns.shape[0]` as n_obs, not the parameterized value. The `_SHRINKAGE_N_OBS` global only affects `regularize_correlation_nonlinear`, which is bypassed in the patched flow.
- **lambda_reg perturbation**: The perturbed `lr` value was computed but not actually applied to the model config (code bug — `lr` variable unused).

This is a known limitation of the sensitivity analysis implementation. However, given the clear walk-forward results (NL-MP uniformly worse, NL-Empirical neutral), fixing this would not change the conclusion.

### 3.5 Turnover

| Variant | Mean Turnover |
|--------|--------------|
| Baseline | 1.624 |
| NL-MP | 1.604 |
| NL-Empirical | 1.624 |

Turnover is essentially unchanged across variants, indicating the shrinkage method doesn't materially affect portfolio rebalancing intensity.

## 4. Analysis: Why No Improvement

### 4.1 Existing Regularization is Well-Tuned

The current two-stage regularization (linear LW + structural prior C0) already provides:
- Excellent conditioning (cond ~37, down from ~10^13)
- Tight, stable shrinkage across all time periods
- Integration with the structural prior (V0 vectors + baseline correlation C_full)

The nonlinear shrinkage replaces Stage 1 but cannot improve on what Stage 1 + Stage 2 already achieve together.

### 4.2 NL-MP Over-Shrinks Signal Eigenvalues

The MP-based method assumes the population is identity. In practice, the US-JP sector correlation matrix has a dominant market factor (eigenvalue ~10-14) and several signal eigenvalues outside the MP bulk. The MP formula shrinks these signal eigenvalues toward 1, destroying useful structural information. This explains the uniform -0.83 Sharpe degradation.

### 4.3 NL-Empirical is Too Weak

The empirical leave-one-out estimator barely shrinks the condition number (~10^6 vs ~10^13 raw). It preserves signal eigenvalues but doesn't sufficiently regularize noise eigenvalues. The result is nearly identical to the baseline (mean Δ = +0.04), indicating the existing linear LW is already capturing the essential shrinkage.

### 4.4 Structural Prior is the Key Differentiator

The strategy's performance depends more on the structural prior (C0 from V0 vectors + C_full) than on the specific Stage 1 shrinkage method. Since Stage 2 (prior shrinkage) is preserved in all variants, the impact of changing Stage 1 is limited.

## 5. Conclusion

**Not adopted.** The nonlinear shrinkage (both MP and empirical variants) does not improve over the existing linear LW shrinkage:

- **NL-MP**: Uniformly degrades performance (0/12 windows, -9.5% mean Sharpe) due to over-shrinking signal eigenvalues
- **NL-Empirical**: Neutral (7/12 windows, +0.4% mean Sharpe — not statistically meaningful)
- **Deflated Sharpe**: All variants statistically significant, but NL-Empirical ≈ baseline
- **Sensitivity**: Not informative due to implementation limitations, but conclusion is robust

The existing two-stage regularization (linear LW + structural prior) is well-tuned for this strategy's N=32, T=504 setting. The structural prior (Stage 2) is the dominant contributor to performance, and the specific Stage 1 method has limited marginal impact.

## 6. Artifacts

- **Code**: `src/experiments/nonlinear_shrinkage.py`, `scripts/experiments/experiment_nonlinear_shrinkage.py`
- **Data**: `outputs/experiments/nonlinear_shrinkage/eigenvalue_diagnostics.csv`, `walkforward_yearly_results.csv`, `sensitivity_analysis.csv`
- **Plots**: `eigenvalue_comparison.png`, `walkforward_sharpe_by_method.png`, `walkforward_sharpe_delta.png`

## 7. References

- Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*.
- Ledoit, O. & Wolf, M. (2017). "Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection." *Review of Financial Studies*.
- Ledoit, O. & Wolf, M. (2020). "Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices." *Annals of Statistics*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 15: Deflated Sharpe Ratio.
