# Macro Sensitivity Matrix Comparison Report

**Config**: `configs/production/production.yaml`
**Start date**: 2015-01-05
**Date**: 2026-07-07

## 1. Matrix Difference Statistics

- Mean absolute difference: 0.2608
- Max absolute difference: 1.0000
- Number of changed entries (>0.01): 34/51
- Correlation between matrices: 0.5427

## 2. Backtest Results

| Config | Sharpe | AR | RISK | MDD | Turnover | Scale Mean | Scale Std | Scale Max | Scale P95 |
|--------|--------|----|------|-----|----------|------------|-----------|-----------|----------|
| baseline | 4.4508 | 139.20% | 31.28% | -7.04% | nan | 2.3429 | 2.1726 | 140.9089 | 4.8688 |
| direction_A | 3.3307 | 67.92% | 20.39% | -16.97% | nan | 2.3429 | 2.1726 | 140.9089 | 4.8688 |
| sigma_yy_E | 4.5633 | 148.19% | 32.48% | -7.51% | nan | 2.3429 | 2.1726 | 140.9089 | 4.8688 |
| dir_A+sigE | 3.4685 | 75.53% | 21.78% | -16.68% | nan | 2.3429 | 2.1726 | 140.9089 | 4.8688 |
| disabled | 4.4016 | 140.81% | 31.99% | -7.19% | nan | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

## 3. Daily Return Correlations

| | baseline | direction_A | sigma_yy_E | dir_A+sigE | disabled |
|---|--------|--------|--------|--------|--------|
| baseline | 1.0000 | 0.4781 | 0.9881 | 0.4752 | 0.9853 |
| direction_A | 0.4781 | 1.0000 | 0.4797 | 0.9914 | 0.4640 |
| sigma_yy_E | 0.9881 | 0.4797 | 1.0000 | 0.4860 | 0.9610 |
| dir_A+sigE | 0.4752 | 0.9914 | 0.4860 | 1.0000 | 0.4552 |
| disabled | 0.9853 | 0.4640 | 0.9610 | 0.4552 | 1.0000 |

## 4. Key Findings

- baseline Sharpe: 4.4508
- direction_A Sharpe: 3.3307
- sigma_yy_E Sharpe: 4.5633
- dir_A+sigE Sharpe: 3.4685
- disabled Sharpe: 4.4016
- Macro confidence effect (baseline - disabled): +0.0492
