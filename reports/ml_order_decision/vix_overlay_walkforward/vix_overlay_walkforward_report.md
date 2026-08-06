# LightGBM Overlay with US/JP VIX Features: Walk-Forward Report

**Config**: n_estimators=100, num_leaves=20, max_depth=3, min_child_samples=300, reg_alpha=0.5, reg_lambda=1.0
- per_ticker_interactions=True
- VIX features: 60-day log z-score (US), 60-day log z-score (JP), 60-day z-score of JP-US spread
- VIX added as: level + `× score` + `× gap` + `× score×gap` + `× score×gap_idio`

## Pooled OOS Performance
- Periods: 2022, 2023, 2024
- Total OOS days: 709
- Baseline V2 Sharpe: 6.9930
- No-VIX Overlay Sharpe: 7.0176 (p=0.0000, DSR=1.0000)
- VIX Overlay Sharpe: 7.0241 (p=0.0000, DSR=1.0000)
- Mean daily diff (VIX vs V2 base): 0.000117 (p=0.0000)
- Mean daily diff (No-VIX vs V2 base): 0.000127 (p=0.0000)
- Mean daily diff (VIX vs No-VIX): -0.000010 (p=0.4106)
- WF win rate vs V2 base: No-VIX 100%, VIX 67%
- WF win rate vs No-VIX: 67% (2/3)

## Per-Fold Metrics

| Year | Base Sharpe | No-VIX Sharpe | VIX Sharpe | No-VIX p | VIX p | VIX vs No-VIX p |
|------|-------------|---------------|------------|----------|-------|-----------------|
| 2022 | 8.3980 | 8.4192 | 8.4441 | 0.0012 | 0.0003 | 0.8429 |
| 2023 | 5.2065 | 5.2100 | 5.1958 | 0.0150 | 0.0397 | 0.6607 |
| 2024 | 7.3401 | 7.3846 | 7.3929 | 0.0003 | 0.0021 | 0.4177 |

## Per-Fold Additional Metrics

| Year | Base AR | No-VIX AR | VIX AR | Base Vol | No-VIX Vol | VIX Vol | Base MDD | No-VIX MDD | VIX MDD |
|------|---------|-----------|--------|----------|------------|---------|----------|------------|---------|
| 2022 | 1.5126 | 1.5484 | 1.5475 | 0.1801 | 0.1839 | 0.1833 | -0.0350 | -0.0359 | -0.0369 |
| 2023 | 0.7745 | 0.7950 | 0.7924 | 0.1488 | 0.1526 | 0.1525 | -0.0414 | -0.0447 | -0.0454 |
| 2024 | 1.5093 | 1.5462 | 1.5425 | 0.2056 | 0.2094 | 0.2086 | -0.0696 | -0.0704 | -0.0701 |

## VIX Overlay Top Features (2024 fold)

| feature | importance |
|---------|------------|
| abs_score | 55.00 |
| ticker | 53.00 |
| gap_idio | 35.00 |
| vix_spread_z_x_score_x_gap | 33.00 |
| vix_spread_z_x_score | 33.00 |
| topix_night | 33.00 |
| jp_vix_z | 30.00 |
| jp_vix_z_x_score_x_gap_idio | 20.00 |
| vix_spread_z_x_gap | 17.00 |
| market_vol_20d | 15.00 |
| abs_gap | 14.00 |
| us_vix_z | 13.00 |
| us_vix_z_x_score_x_gap_idio | 9.00 |
| us_vix_z_x_score | 9.00 |
| vix_spread_z_x_score_x_gap_idio | 7.00 |
| jp_vix_z_x_score_x_gap | 7.00 |
| jp_vix_z_x_score | 5.00 |
| us_vix_z_x_score_x_gap | 5.00 |
| sigma_gap | 5.00 |
| vix_spread_z | 3.00 |


## Verdict against Adoption Criteria
1. VIX overlay Sharpe > no-VIX overlay by >0.01: FAIL (+0.0065)
2. VIX overlay beats no-VIX in >=2/3 folds: PASS (67%)
3. DSR >= 0.95: PASS (1.0000)
4. Pooled daily return diff (VIX vs no-VIX) p < 0.10: FAIL (p=0.4106)