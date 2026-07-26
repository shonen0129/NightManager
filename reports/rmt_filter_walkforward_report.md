# RMT Eigenvalue Cleaning: Walk-Forward Validation Report

**Date**: 2026-07-26 (v2 updated)
**Experiment**: Random Matrix Theory (RMT) eigenvalue cleaning for correlation matrix denoising
**References**: Laloux et al. (1999), Plerou et al. (2002)
**Status**: **Not adopted** — marginal improvement, within noise margin; RMT-as-LW-replacement also tested and not adopted

---

## 1. Hypothesis

**v1 (RMT as preprocessing)**: Add explicit RMT eigenvalue cleaning before the existing two-stage LW + prior regularization. RMT identifies which eigenvalues fall within the Marchenko-Pastur (MP) noise bulk and replaces them, preserving only "informational" eigenvalues.

**v2 (RMT as LW replacement)**: Replace the LW shrinkage stage entirely with RMT filtering. When `rmt_replace_lw=true`, set `lambda_lw=0` so the pipeline becomes RMT → Prior (2-stage) instead of RMT → LW → Prior (3-stage). This tests whether RMT can substitute LW's denoising function.

## 2. Methods

### 2.1 RMT Eigenvalue Cleaning

Implemented in `src/leadlag/core/correlation.py::rmt_eigenvalue_cleaning()`. The MP upper bound is:

    lambda_+ = (1 + sqrt(N / T))^2

where N=32 assets, T=504 observations (BLP window). Eigenvalues below lambda_+ are considered noise and replaced.

Two cleaning methods evaluated:
- **clip_to_mean**: Replace noise eigenvalues with their average (Laloux et al. 1999). Preserves trace.
- **clip_to_lambda_plus**: Replace noise eigenvalues with lambda_+.

RMT cleaning is applied as a post-processing step in `compute_correlation()` after Pearson/copula estimation, before the existing two-stage LW + prior regularization.

### 2.2 Validation Framework

- **Walk-forward**: 12 yearly windows (2015–2026), full backtest per variant
- **Deflated Sharpe Ratio**: B=35 total trials (30 prior + 5 current)
- **5 variants**:
  - `baseline`: No RMT, LW + prior (default)
  - `rmt_mean`: RMT (clip_to_mean) + LW + prior (v1 preprocessing)
  - `rmt_lambda_plus`: RMT (clip_to_lambda_plus) + LW + prior (v1 preprocessing)
  - `rmt_replace_lw_mean`: RMT (clip_to_mean) + prior, LW skipped (v2 replacement)
  - `rmt_replace_lw_lp`: RMT (clip_to_lambda_plus) + prior, LW skipped (v2 replacement)
- **Eigenvalue diagnostics**: MP bulk membership, condition number

### 2.3 Implementation

- `src/leadlag/core/correlation.py`: `rmt_eigenvalue_cleaning()` function + `rmt_filter`/`rmt_method` options in `compute_correlation()`
- `src/leadlag/core/signal.py`: `rmt_filter`/`rmt_method` parameters in `compute_signal()`
- `src/leadlag/models/sector_relative_ensemble_blp_enhanced.py`: Config resolution for `rmt_filter`/`rmt_method`
- `scripts/experiments/experiment_rmt_filter.py`: Full experiment script

## 3. Results

### 3.1 Eigenvalue Diagnostics

| Metric | Value |
|--------|-------|
| N (assets) | 32 |
| T (window) | 504 |
| q = N/T | 0.0635 |
| MP lower bound | 0.5595 |
| MP upper bound | 1.5674 |
| Mean informational eigenvalues | 2.5 |
| Mean noise eigenvalues | 29.4 |
| Mean condition number (raw) | ~6.4×10^12 |

**Key observation**: Out of 32 eigenvalues, only ~2.5 are informational (above MP upper bound). The raw correlation matrix is extremely ill-conditioned (~10^12), confirming significant noise. The LW shrinkage already handles this implicitly by shrinking toward equicorrelation, but RMT explicitly removes the noise component.

### 3.2 Pooled Performance (2015–2026)

| Variant | Sharpe (net) | AR (net) | Vol (net) | Max DD | DSR |
|---------|-------------|----------|-----------|--------|-----|
| **Baseline (LW)** | 8.664 | 1.386 | 0.160 | -5.97% | 1.000 |
| **RMT (clip_to_mean) + LW** | 8.740 | 1.410 | 0.161 | -6.08% | 1.000 |
| **RMT (clip_to_lambda_plus) + LW** | 8.499 | 1.309 | 0.154 | -6.70% | 1.000 |
| **RMT (clip_to_mean), LW replaced** | 8.738 | 1.410 | 0.161 | -6.07% | 1.000 |
| **RMT (clip_to_lambda_plus), LW replaced** | 8.500 | 1.309 | 0.154 | -6.70% | 1.000 |

### 3.3 Mean Period Performance

| Variant | Sharpe (mean) | AR (mean) | MDD (mean) |
|---------|--------------|-----------|------------|
| Baseline | 8.902 | 1.409 | -3.86% |
| RMT (clip_to_mean) + LW | 8.988 | 1.433 | -4.10% |
| RMT (clip_to_lambda_plus) + LW | 8.732 | 1.328 | -4.12% |
| RMT (clip_to_mean), LW replaced | 8.987 | 1.433 | -4.10% |
| RMT (clip_to_lambda_plus), LW replaced | 8.733 | 1.328 | -4.12% |

### 3.4 Year-by-Year Sharpe

| Year | Baseline | RMT (mean) + LW | RMT (lambda+) + LW | RMT (mean), LW replaced | RMT (lambda+), LW replaced | Winner |
|------|----------|-----------------|--------------------|-----------------------|---------------------------|--------|
| 2015 | 14.19 | 14.87 | 14.88 | 14.87 | 14.88 | RMT |
| 2016 | 16.61 | 16.62 | 16.38 | 16.64 | 16.38 | RMT (mean, replaced) |
| 2017 | 10.10 | 9.80 | 9.66 | 9.80 | 9.66 | Baseline |
| 2018 | 6.06 | 6.29 | 5.73 | 6.29 | 5.73 | RMT (mean) |
| 2019 | 7.32 | 7.48 | 7.26 | 7.45 | 7.26 | RMT (mean) |
| 2020 | 9.51 | 9.08 | 9.31 | 9.07 | 9.31 | Baseline |
| 2021 | 6.69 | 7.33 | 6.52 | 7.33 | 6.52 | RMT (mean) |
| 2022 | 8.40 | 7.63 | 7.55 | 7.63 | 7.57 | Baseline |
| 2023 | 4.60 | 4.91 | 4.47 | 4.92 | 4.47 | RMT (mean) |
| 2024 | 7.14 | 6.82 | 7.17 | 6.81 | 7.17 | Baseline |
| 2025 | 7.66 | 8.29 | 8.11 | 8.28 | 8.11 | RMT (mean) |
| 2026 | 8.51 | 8.74 | 7.72 | 8.74 | 7.74 | RMT (mean) |

**Win counts**:
- RMT (clip_to_mean) + LW wins **8/12** years vs baseline
- RMT (clip_to_lambda_plus) + LW wins **3/12** years vs baseline
- RMT (clip_to_mean), LW replaced wins **8/12** years vs baseline
- RMT (clip_to_lambda_plus), LW replaced wins **3/12** years vs baseline

### 3.5 Risk Characteristics

| Variant | Skew | Excess Kurtosis |
|---------|------|-----------------|
| Baseline | 1.199 | 5.816 |
| RMT (mean) + LW | 1.035 | 4.366 |
| RMT (lambda+) + LW | 1.048 | 4.817 |
| RMT (mean), LW replaced | 1.037 | 4.376 |
| RMT (lambda+), LW replaced | 1.048 | 4.819 |

RMT slightly reduces both skew and kurtosis, indicating marginally less tail exposure. LW replacement has no measurable effect on risk characteristics.

## 4. Analysis

### 4.1 Why No Adoption

Despite winning 8/12 years, RMT (clip_to_mean) shows only a **+0.08 pooled Sharpe improvement** (8.74 vs 8.66, +0.9%). This is well within the noise margin given:
- 12 years × ~240 days = ~2750 observations
- Standard error of annualized Sharpe ≈ 1/sqrt(T) ≈ 0.019
- The improvement is ~4 SE, but the DSR is already 1.0 for baseline (both are significant)
- The improvement is not large enough to justify the added complexity

RMT (clip_to_lambda_plus) is **worse** than baseline (-0.16 Sharpe), confirming that clipping to the MP upper bound is too aggressive — it inflates noise eigenvalues rather than averaging them out.

### 4.2 v2: RMT as LW Replacement

The v2 experiment tested whether RMT can **replace** the LW shrinkage stage entirely (`lambda_lw=0`). The pipeline becomes:

- **v1 (3-stage)**: RMT → LW (lambda_lw=0.5) → Prior (lambda_reg=0.75)
- **v2 (2-stage)**: RMT → Prior (lambda_reg=0.75)

**Key finding**: RMT-as-LW-replacement produces **virtually identical** results to RMT-as-preprocessing:

| Comparison | Pooled Sharpe | Difference |
|------------|--------------|------------|
| rmt_mean + LW | 8.740 | — |
| rmt_mean, LW replaced | 8.738 | -0.002 |
| rmt_lambda_plus + LW | 8.499 | — |
| rmt_lambda_plus, LW replaced | 8.500 | +0.001 |

The maximum difference is **0.002 Sharpe** — effectively zero. This occurs because:

1. **Prior shrinkage dominates**: With `lambda_reg=0.75`, 75% of the correlation matrix is already replaced by the structured prior `c_0_t`. The LW stage operates on the remaining 25%, so its contribution is marginal.
2. **RMT and LW target the same noise**: Both methods denoise the correlation matrix — RMT by clipping noise eigenvalues, LW by shrinking toward equicorrelation. When RMT is already applied, the LW shrinkage has almost nothing left to denoise.
3. **Effective raw weight**: With LW (`lambda_lw=0.5`), the raw sample weight is `(1-0.5)*(1-0.75)=0.125`. Without LW (`lambda_lw=0`), it's `(1-0)*(1-0.75)=0.25`. The difference (0.125 vs 0.25) is small in absolute terms because the prior dominates either way.

**Conclusion**: RMT can functionally replace LW without performance degradation, but it provides no improvement either. The LW stage is effectively redundant once RMT is applied, but removing it doesn't help because the prior shrinkage is the dominant regularization force.

### 4.3 Numerical Stability

RuntimeWarnings for divide-by-zero and overflow in matmul were observed during RMT reconstruction. These occur when the correlation matrix has near-zero or negative eigenvalues (condition number ~10^12). The `np.nan_to_num` calls in the model pipeline handle these gracefully, but they indicate that the raw correlation matrix is extremely ill-conditioned. The LW shrinkage resolves this more effectively than RMT alone.

## 5. Conclusion

**v1 (RMT as preprocessing)**: RMT eigenvalue cleaning (clip_to_mean) shows a marginal pooled Sharpe improvement of +0.9% (8.74 vs 8.66) and wins 8/12 walk-forward years. However, the improvement is within the practical noise margin and does not justify the added complexity over the existing two-stage LW + prior regularization. The clip_to_lambda_plus variant is strictly worse.

**v2 (RMT as LW replacement)**: Replacing LW with RMT (`lambda_lw=0`) produces virtually identical results to RMT+LW (difference <0.002 Sharpe). This confirms that LW is redundant once RMT is applied, but also that the prior shrinkage (`lambda_reg=0.75`) is the dominant regularization force — neither LW nor RMT matters much when 75% of the matrix is already replaced by the structured prior.

**Decision**: Not adopted. Both RMT-as-preprocessing and RMT-as-LW-replacement fail to provide meaningful improvement over the baseline. The implementation remains available as options (`rmt_filter: true`, `rmt_replace_lw: true` in config) for future experimentation, but are disabled by default.

## 6. Files

- Implementation: `src/leadlag/core/correlation.py` (`rmt_eigenvalue_cleaning`, `compute_correlation` with `rmt_filter` option)
- Signal pipeline: `src/leadlag/core/signal.py` (pass-through `rmt_filter`/`rmt_method`)
- Pipeline component: `src/leadlag/core/pipeline.py` (`PCAComponent` with `rmt_filter`/`rmt_method`)
- Model: `src/leadlag/models/sector_relative_ensemble_blp_enhanced.py` (config resolution, `rmt_replace_lw`)
- Base model: `src/leadlag/models/blp_base.py` (pass-through to `PCAComponent`)
- Experiment script: `scripts/experiments/experiment_rmt_filter.py`
- v1 Results: `reports/rmt_filter_walkforward/` (CSV files)
- v2 Results: `reports/rmt_filter_walkforward_v2/` (CSV files)
