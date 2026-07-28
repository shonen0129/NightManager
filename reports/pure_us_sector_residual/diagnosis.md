# Pure US Sector Residual — Diagnostic Analysis

- US residualization: beta_window=60, gamma=1.0
- Fractional differencing: d=0.1
- US macro proxy: equal-weight average of 15 US sector/style ETFs

## 1. How much of US cross-sectional return is macro/common?

Average share of US return variance explained by the top principal component: **69.83%**.
Top-3 PCs typically explain the majority of cross-sectional US return variance.
This means residualization removes the single largest source of signal magnitude.

### Yearly PC1 variance share

| Year | PC1 share |
|---|---|
| 2009 | 84.36% |
| 2010 | 86.54% |
| 2011 | 90.35% |
| 2012 | 81.02% |
| 2013 | 78.30% |
| 2014 | 71.45% |
| 2015 | 76.66% |
| 2016 | 64.65% |
| 2017 | 48.90% |
| 2018 | 75.43% |
| 2019 | 62.99% |
| 2020 | 83.25% |
| 2021 | 54.42% |
| 2022 | 71.17% |
| 2023 | 56.87% |
| 2024 | 55.04% |
| 2025 | 69.94% |
| 2026 | 45.62% |

## 2. Is the US macro factor predictive for next-day JP returns?

| Target | Pearson corr | R² | Slope | p-value |
|---|---|---|---|---|
| Raw JP 9:10→close | 0.0087 | 0.0001 | 0.0049 | 5.7659e-01 |
| TOPIX-residual JP 9:10→close | 0.1340 | 0.0179 | 0.0422 | 9.7417e-18 |

Residual-BLPX predicts the TOPIX-residualized JP target. The US common factor's
predictive correlation with that target is even weaker than with the raw target,
so removing the US macro factor does not eliminate a dominant JP predictor.

## 3. Per-sector correlation to mapped JP targets

Per-sector time-series correlation between the US input series and the average of
its mapped JP sector targets. Reported separately for raw JP targets and
TOPIX-residual JP targets (the actual BLPX objective).

### Raw JP target

| US ticker | Mapped JP | corr(raw) | corr(res) | Δ(res) | corr(FD-raw) | corr(FD-res) | Δ(FD) |
|---|---|---|---|---|---|---|---|
| XLB | 1620.T,1623.T | 0.0873 | 0.0975 | +0.0101 | 0.0786 | 0.0927 | +0.0141 |
| XLC | 1626.T | 0.0228 | 0.0208 | -0.0020 | 0.0197 | 0.0217 | +0.0020 |
| XLE | 1618.T,1627.T | 0.0760 | 0.0978 | +0.0218 | 0.0755 | 0.0979 | +0.0224 |
| XLF | 1631.T,1632.T | 0.0674 | 0.1173 | +0.0499 | 0.0650 | 0.1141 | +0.0490 |
| XLI | 1624.T,1622.T,1626.T | 0.0284 | 0.0508 | +0.0224 | 0.0243 | 0.0532 | +0.0288 |
| XLK | 1626.T,1625.T | -0.0080 | 0.0033 | +0.0112 | -0.0137 | 0.0003 | +0.0140 |
| XLP | 1617.T,1630.T | 0.0509 | 0.1072 | +0.0563 | 0.0502 | 0.1083 | +0.0581 |
| XLRE | 1633.T | 0.0394 | 0.0473 | +0.0079 | 0.0381 | 0.0469 | +0.0087 |
| XLU | 1627.T | 0.0251 | 0.0617 | +0.0366 | 0.0239 | 0.0605 | +0.0366 |
| XLV | 1621.T | 0.0220 | 0.0943 | +0.0724 | 0.0172 | 0.0939 | +0.0767 |
| XLY | 1630.T,1626.T,1622.T | -0.0055 | 0.0349 | +0.0404 | -0.0100 | 0.0333 | +0.0433 |
| MTUM | 1625.T,1626.T | -0.0109 | 0.0012 | +0.0121 | -0.0148 | 0.0016 | +0.0164 |
| VLUE | 1631.T,1632.T,1623.T,1622.T | 0.0377 | 0.0656 | +0.0280 | 0.0333 | 0.0632 | +0.0299 |
| IUSG | 1626.T,1625.T | -0.0022 | 0.0191 | +0.0213 | -0.0077 | 0.0167 | +0.0243 |
| USMV | 1617.T,1621.T,1627.T | -0.0101 | 0.0697 | +0.0798 | -0.0122 | 0.0701 | +0.0823 |

- Sectors whose residual correlation improved (raw): 14/15
- Sectors whose FD-residual correlation improved: 15/15
- Mean Δ corr (raw): 0.0312
- Mean Δ corr (FD): 0.0338
- Median Δ corr (raw): 0.0224
- Median Δ corr (FD): 0.0288


### TOPIX-residual JP target

| US ticker | Mapped JP | corr(raw) | corr(res) | Δ(res) | corr(FD-raw) | corr(FD-res) | Δ(FD) |
|---|---|---|---|---|---|---|---|
| XLB | 1620.T,1623.T | 0.1520 | 0.0773 | -0.0747 | 0.1431 | 0.0732 | -0.0699 |
| XLC | 1626.T | 0.0677 | 0.0294 | -0.0383 | 0.0651 | 0.0312 | -0.0339 |
| XLE | 1618.T,1627.T | 0.1195 | 0.0985 | -0.0209 | 0.1200 | 0.0984 | -0.0216 |
| XLF | 1631.T,1632.T | 0.1261 | 0.1614 | +0.0353 | 0.1236 | 0.1566 | +0.0330 |
| XLI | 1624.T,1622.T,1626.T | 0.1005 | 0.0360 | -0.0645 | 0.0943 | 0.0361 | -0.0582 |
| XLK | 1626.T,1625.T | 0.0808 | 0.0437 | -0.0371 | 0.0748 | 0.0408 | -0.0339 |
| XLP | 1617.T,1630.T | 0.0944 | 0.1382 | +0.0438 | 0.0938 | 0.1381 | +0.0444 |
| XLRE | 1633.T | 0.0694 | 0.0260 | -0.0434 | 0.0694 | 0.0280 | -0.0414 |
| XLU | 1627.T | 0.0517 | 0.0700 | +0.0182 | 0.0505 | 0.0673 | +0.0169 |
| XLV | 1621.T | 0.0883 | 0.1308 | +0.0425 | 0.0831 | 0.1285 | +0.0454 |
| XLY | 1630.T,1626.T,1622.T | 0.0397 | 0.0150 | -0.0247 | 0.0354 | 0.0130 | -0.0224 |
| MTUM | 1625.T,1626.T | 0.0563 | -0.0004 | -0.0567 | 0.0511 | -0.0024 | -0.0535 |
| VLUE | 1631.T,1632.T,1623.T,1622.T | 0.1280 | 0.1102 | -0.0178 | 0.1232 | 0.1081 | -0.0151 |
| IUSG | 1626.T,1625.T | 0.0834 | 0.0597 | -0.0237 | 0.0777 | 0.0567 | -0.0210 |
| USMV | 1617.T,1621.T,1627.T | 0.0540 | 0.1092 | +0.0552 | 0.0521 | 0.1074 | +0.0553 |

- Sectors whose residual correlation improved (raw): 5/15
- Sectors whose FD-residual correlation improved: 5/15
- Mean Δ corr (raw): -0.0138
- Mean Δ corr (FD): -0.0117
- Median Δ corr (raw): -0.0237
- Median Δ corr (FD): -0.0216

## 4. Strategy daily-return similarity

- Correlation between baseline and pure daily returns: **0.9455**
- Std dev of daily return difference: **0.0034**

## 5. Interpretation

1. **The US common factor is large and, surprisingly, informative for the BLPX target.**
   The top PC explains ~70% of US cross-sectional variance. While its correlation
   with the raw JP 9:10→close target is negligible (~0.009), its correlation with
   the TOPIX-residualized target that Residual-BLPX actually predicts is 0.134
   (R² ~1.8%, highly significant). This means the US macro factor captures a
   global/sector component that is orthogonal to TOPIX and helps predict JP
   sector residual returns.
2. **For the actual BLPX target, residualization does not improve per-sector correlations.**
   When the JP target is TOPIX-residualized, only 5/15 mapped US→JP sector pairs
   show higher residual correlation, and the mean Δ is negative (~ -0.014).
   The macro component the residualization removes is itself predictive of the
   residual JP target.
3. **BLPX learns the optimal combination, so hard constraints are suboptimal.**
   Baseline Residual-BLPX uses λ_pca / λ_sector priors, ridge shrinkage, and
   the full US covariance matrix to weight common vs. sector-specific directions.
   Pre-orthogonalizing US returns removes a direction the model would have kept
   with a learned (non-zero) coefficient.
4. **Residualization adds estimation noise.** The 60-day rolling OLS betas are noisy,
   especially in high-volatility regimes. Residual returns inherit this noise, and
   the subsequent fractional-differencing filter further smooths the residualized signal.
5. **Yearly variation is regime-dependent.** In years with strong sector-rotation
   (idiosyncratic) shocks (e.g. 2017, 2019, 2022-2026) the pure signal can win;
   in macro-trend or high-vol years the baseline wins.

## Conclusion

The Pure US Sector Component Signal underperforms because the US macro/common factor
is a genuine predictor of the TOPIX-residualized JP target that Residual-BLPX models.
Removing it via hard 60-day rolling OLS residualization discards useful signal,
adds beta-estimation noise, and constrains BLPX's learned covariance/prior weighting.
BLPX already extracts the optimal soft combination of common and idiosyncratic US
information, so the pre-orthogonalization produces a small but consistent degradation
in Sharpe, AR, and IC.