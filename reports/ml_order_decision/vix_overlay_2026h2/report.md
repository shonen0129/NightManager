# LGBM VIX Overlay on 2026-07 (H2 2026 to date)

**Config**: n_estimators=100, num_leaves=20, max_depth=3, reg_alpha=0.5, reg_lambda=1.0, per_ticker_interactions=True
- Train: 2020-01-06 ~ 2024-12-31
- Test:  2026-07-01 ~ 2026-07-31
- VIX features: 60-day log z-score (US), 60-day log z-score (JP), JP-US spread z-score

## 2026-07 Pooled Performance
- Total OOS days: 21
- Baseline V2 Sharpe: -0.8662
- No-VIX Overlay Sharpe: -0.9198 (p=0.3069)
- VIX Overlay Sharpe: -0.8993 (p=0.5318)
- VIX vs No-VIX: ΔSharpe=+0.0206, p=0.5202
- Mean daily diff (VIX vs no-VIX): 0.000030

## Last 7 Trading Days
- Period: 2026-07-23 ~ 2026-07-31

| Model | 7-day total | 7-day MDD |
|-------|-------------|-----------|
| baseline | -7.4037% | -10.6614% |
| no_vix | -7.6395% | -10.8842% |
| vix | -7.6418% | -10.8864% |

- VIX vs no-VIX 7-day total diff: -0.0023%
- VIX vs no-VIX 7-day MDD diff: -0.0022%