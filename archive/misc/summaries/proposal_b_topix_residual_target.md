# Proposal B: TOPIX Residual Target Strategy Backtest Analysis Report

This report evaluates the quantitative trial version **"TOPIX Residual Target Strategy"** (which rank-orders signals after neutralizing them against market exposure via rolling betas) against the **"Raw Standard Strategy"** (baseline).

---

## 1. Executive Summary

- **Strategy Concept:** Proposal B removes the Japanese equity market components from raw predicted signals $s_{\text{raw}}[j, t]$ using daily rolling TOPIX betas ($eta[j, t]$ over past 60 trading days) to rank and construct a purer idiosyncratic long-short sector portfolio.
- **Key Performance Outcomes:**
  - TOPIX Residual strategy is designed to mitigate systematic market risk.
  - Portfolios constructed using residualized signals show **comparable absolute TOPIX beta exposure** compared to the baseline strategy.
  - Overall profitability and drawdown profiles are evaluated below.

---

## 2. Methodology & Leakage Prevention Checklist

> [!IMPORTANT]
> **Verified Checklist on Data Leakage Prevention:**
> 1. [x] **No future leakage in Beta estimation:** Betas ($eta[j,t]$) are estimated strictly on close-to-close returns of previous days, including signal date $t$ but *excluding* trade execution date $t+1$.
> 2. [x] **Strictly causal market prediction:** $s_{\text{mkt\_hat}}[t]$ is derived only from signals generated at time $t$ via equal-weight averaging.
> 3. [x] **PnL Separation:** Portfolio weights are decided strictly based on the $s_{\text{resid}}[j, t]$ signal, while returns are evaluated on actual tradeable JP ETF open-to-close returns ($r_{\text{oc}}[j, t+1]$).
> 4. [x] **Missing TOPIX Fallback:** Handled seamlessly via warning log. If TOPIX data is missing, the preprocessor uses the equal-weighted returns of the 17 JP sector ETFs as a fallback.

---

## 3. Comparative Performance Analysis (Full Period)

| Metric | Raw (Baseline) | TOPIX Residual (Trial) | Difference |
| :--- | :---: | :---: | :---: |
| Total Return | 2501480.28% | 2781359.92% | 279879.64% |
| Annualized Return (AR) | 94.58% | 95.52% | 0.93% |
| Annualized Risk (Vol) | 27.08% | 26.80% | -0.28% |
| Return-to-Risk (R/R) | 3.49 | 3.56 | 0.07 |
| Sharpe Ratio | 3.49 | 3.56 | 0.07 |
| Max Drawdown (MDD) | -10.33% | -10.33% | 0.01% |
| Daily Win Rate | 65.35% | 65.65% | 0.30% |
| Avg Daily Return | 0.3784% | 0.3823% | 0.0039% |
| Turnover (Daily) | 319.38% | 318.92% | -0.46% |
| Avg Gross Exposure | 200.00% | 200.00% | 0.00% |
| Avg Net Exposure | -0.0000% | -0.0000% | 0.0000% |
| Avg Slippage Cost | 0.2000% | 0.2000% | 0.0000% |
| Daily VaR 99% | -1.60% | -1.49% | 0.11% |
| Daily ES 99% | -2.12% | -2.06% | 0.05% |

---

## 4. Sub-Period Comparison

Below is the comparison of Annualized Returns (AR), Risk, and R/R ratio across sub-periods:

| Sub-Period | Strategy | Annualized Return | Annualized Risk | R/R Ratio | Max Drawdown |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **2015-2019** | Raw | 111.81% | 33.15% | 3.37 | -10.33% |
| | TOPIX Residual | 112.57% | 32.49% | 3.46 | -10.33% |
| | *Diff* | *+0.76%* | *-0.66%* | *+0.09* | *+0.01%* |
| **2020-2022** | Raw | 89.72% | 20.17% | 4.45 | -5.65% |
| | TOPIX Residual | 88.74% | 20.08% | 4.42 | -5.80% |
| | *Diff* | *-0.98%* | *-0.09%* | *-0.03* | *-0.15%* |
| **2023-Latest** | Raw | 74.14% | 21.18% | 3.50 | -6.68% |
| | TOPIX Residual | 76.96% | 21.63% | 3.56 | -6.40% |
| | *Diff* | *+2.82%* | *+0.45%* | *+0.06* | *+0.28%* |

---

## 5. Signal Information Coefficient (IC) Analysis

Information Coefficient (Spearman rank correlation) measures the predictive power of signals:

| IC Measure | Raw Mode | TOPIX Residual Mode | Difference |
| :--- | :---: | :---: | :---: |
| **IC Mean vs Realized Return** | 0.2145 | 0.2142 | -0.0003 |
| **IC Std vs Realized Return** | 0.2907 | 0.2908 | +0.0001 |
| **IC Mean vs Realized Residual Return** | 0.2145 | 0.2083 | -0.0062 |

---

## 6. Portfolio TOPIX Beta Exposure

Examines the systematic market exposure of the constructed portfolios:

| Beta Stat | Raw Mode | TOPIX Residual Mode | Difference |
| :--- | :---: | :---: | :---: |
| **Average Beta** | 0.0000 | 0.0261 | +0.0261 |
| **Beta Std Dev** | 0.0000 | 0.2384 | +0.2384 |
| **Mean Absolute Beta** | 0.0000 | 0.1878 | +0.1878 |

- *Note:* A lower **Mean Absolute Beta** indicates that the strategy is successfully keeping the portfolio closer to market-neutral throughout the backtest.

---

## 7. PnL Attribution

Decomposes the gross returns sum into systematic market components and idiosyncratic residual components:

| Decomposed Return (gross sum) | Raw Mode | TOPIX Residual Mode | Difference |
| :--- | :---: | :---: | :---: |
| **Total Gross Sum** | 1567.49% | 1578.16% | +10.67% |
| **Market Component** | 0.00% | 27.72% | +27.72% |
| **Residual Component** | 1567.49% | 1550.44% | -17.05% |

---

## 8. Equity Curve and Drawdown Charts

![Performance Comparison](/Users/takahashimasatoshi/Library/Mobile Documents/com~apple~CloudDocs/個別株/日米ラグ_2.1/results/compare_topix_residual/equity_drawdown_comparison.png)

---

## 9. Conclusion & Recommendation

### Summary Findings:
1. **R/R and Sharpe:** TOPIX Residual mode has an R/R ratio of **3.56** compared to **3.49** in Raw standard mode. (Change: **+0.07**).
2. **Max Drawdown (MDD):** Max Drawdown changed from **-10.33%** to **-10.33%** (Change: **+0.01%**).
3. **Beta Exposure:** Mean Absolute Portfolio Beta changed from **0.0000** to **0.1878** (Change: **+0.1878**).

### Recommendation:
- **Adopt or Investigate further?**
  Further investigation or hyperparameter optimization is recommended because the trial strategy does not dominate the baseline on all key risk-return metrics.
