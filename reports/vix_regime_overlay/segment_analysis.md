# VIX Regime Segment Analysis: Baseline Large Return/Loss Days

- Period: 2018-04-01 ~ 2024-12-31
- VIX z-score: 60-day rolling on log VIX
- Regime threshold: z > 0.5 = high
- Tail definition: top/bottom 10% of daily net returns

## Baseline Returns by VIX Regime

| regime | n_days | pct_days | mean_ret_bps | std_ret_bps | sharpe | total_ret_pct | median_ret_bps | max_ret_bps | min_ret_bps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| US_high_JP_low | 139 | 8.704 | 30.428 | 91.937 | 5.254 | 51.669 | 27.748 | 318.703 | -191.083 |
| US_low_JP_high | 143 | 8.954 | 46.993 | 83.842 | 8.897 | 94.544 | 39.515 | 304.118 | -130.044 |
| both_high | 319 | 19.975 | 57.455 | 116.646 | 7.819 | 508.812 | 50.304 | 744.427 | -348.049 |
| both_low | 977 | 61.177 | 38.597 | 85.980 | 7.126 | 4059.658 | 31.666 | 493.783 | -332.651 |

## Tail Day Characteristics (top/bottom 10%)

| tail | n_days | mean_ret_bps | mean_us_vix_z | mean_jp_vix_z | mean_spread_z | pct_US_high_JP_low | pct_US_low_JP_high | pct_both_high | pct_both_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top | 160 | 230.228 | 0.263 | 0.412 | 0.150 | 8.750 | 9.375 | 31.875 | 50.000 |
| bottom | 160 | -100.991 | 0.044 | -0.057 | -0.101 | 13.125 | 7.500 | 23.125 | 55.000 |

## Overlay Impact by Regime (overlay - baseline, bps)

### discrete_jp_led_w60

| regime | n_days | base_mean_bps | overlay_mean_bps | mean_diff_bps | win_rate_pct | tail_bottom10_avg_diff_bps | tail_top10_avg_diff_bps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US_high_JP_low | 139 | 30.428 | 30.512 | 0.084 | 13.669 | 0.000 | 0.722 |
| US_low_JP_high | 143 | 46.993 | 35.330 | -11.662 | 37.762 | -85.111 | 31.411 |
| both_high | 319 | 57.455 | 28.576 | -28.879 | 31.034 | -143.675 | 57.348 |
| both_low | 977 | 38.597 | 36.326 | -2.271 | 10.747 | -33.369 | 10.724 |

### discrete_global_stress_w60

| regime | n_days | base_mean_bps | overlay_mean_bps | mean_diff_bps | win_rate_pct | tail_bottom10_avg_diff_bps | tail_top10_avg_diff_bps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US_high_JP_low | 139 | 30.428 | 30.512 | 0.084 | 14.388 | 0.000 | 0.722 |
| US_low_JP_high | 143 | 46.993 | 34.502 | -12.490 | 37.063 | -85.066 | 31.231 |
| both_high | 319 | 57.455 | 28.573 | -28.882 | 31.034 | -143.675 | 57.348 |
| both_low | 977 | 38.597 | 38.604 | 0.007 | 3.173 | -0.126 | 0.200 |

### continuous_spread_w60

| regime | n_days | base_mean_bps | overlay_mean_bps | mean_diff_bps | win_rate_pct | tail_bottom10_avg_diff_bps | tail_top10_avg_diff_bps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US_high_JP_low | 139 | 30.428 | 30.443 | 0.015 | 12.950 | -0.001 | 0.142 |
| US_low_JP_high | 143 | 46.993 | 33.217 | -13.776 | 27.972 | -77.416 | 30.448 |
| both_high | 319 | 57.455 | 53.509 | -3.946 | 30.094 | -34.787 | 8.628 |
| both_low | 977 | 38.597 | 35.685 | -2.912 | 28.454 | -28.665 | 10.331 |

## Interpretation

- **Top 10% return days** (160 days) have mean US VIX z = 0.26, JP VIX z = 0.41.
- **Bottom 10% return days** (160 days) have mean US VIX z = 0.04, JP VIX z = -0.06.
- **JP VIX is actually higher on large *gain* days than on large *loss* days.** This is the opposite of the JP-led-shock hypothesis and explains why cutting gross in JP-high days hurt alpha.
- **Best regime for baseline**: both_low (total return 4059.7%, Sharpe 7.13).
- **Worst regime for baseline**: US_high_JP_low (total return 51.7%, Sharpe 5.25).

## Files

- `results/vix_regime_overlay/baseline_regime_stats.csv`
- `results/vix_regime_overlay/baseline_tail_analysis.csv`
- `results/vix_regime_overlay/overlay_impact_by_regime.csv`
