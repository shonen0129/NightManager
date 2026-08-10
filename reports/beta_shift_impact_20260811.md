# Preprocessor beta/winsorize shift impact report

Date: 2026-08-11 15:15

## Direct df_exec comparison

- Common days: 4168
- Beta columns compared: 17
- Max absolute beta difference (all): 3.878917e+00
- Max absolute beta difference (>=120d warm-up): 3.878917e+00
- Mean absolute beta difference (all): 2.175791e-02
- Mean absolute beta difference (>=120d warm-up): 2.169756e-02
- q50 / q95 / q99 warm-up abs diff: 8.027771e-03 / 7.178201e-02 / 2.025329e-01
- Pearson correlation of beta matrices (warm-up): 0.985263

### Per-ticker max absolute beta difference (warm-up)

| Ticker | Max abs diff |
| --- | --- |
| jp_beta_1617.T | 2.861765e+00 |
| jp_beta_1618.T | 1.859934e+00 |
| jp_beta_1619.T | 3.158295e+00 |
| jp_beta_1620.T | 2.219429e+00 |
| jp_beta_1621.T | 2.954633e+00 |
| jp_beta_1622.T | 3.570754e+00 |
| jp_beta_1623.T | 3.878917e+00 |
| jp_beta_1624.T | 1.309380e+00 |
| jp_beta_1625.T | 1.674809e+00 |
| jp_beta_1626.T | 2.513270e+00 |
| jp_beta_1627.T | 2.236569e+00 |
| jp_beta_1628.T | 7.433179e-01 |
| jp_beta_1629.T | 2.124947e+00 |
| jp_beta_1630.T | 8.890892e-01 |
| jp_beta_1631.T | 1.219060e+00 |
| jp_beta_1632.T | 1.333397e+00 |
| jp_beta_1633.T | 3.802859e+00 |

## Signal-level impact (gap_idio / gap_filt)

These are the quantities actually used in `core/signal.py` and the ML overlay.

| Metric | gap_idio | gap_filt |
| --- | --- | --- |
| mean abs diff (warm) | 7.279423e-04 | 4.367654e-04 |
| max abs diff (warm) | 1.372952e+00 | 8.237712e-01 |
| q50 | 3.164089e-05 | 1.898454e-05 |
| q95 | 7.714933e-04 | 4.628960e-04 |
| q99 | 3.563046e-03 | 2.137828e-03 |

## V2 Backtest comparison

| Metric | Old (no shift) | New (shift) |
| --- | --- | --- |
| net Sharpe | 4.736 | 4.736 |
| max DD | -0.069 | -0.069 |
| mean turnover | 0.000 | 0.000 |
| fallback rate | 0.000 | 0.000 |

## Recommendation

The `gap_idio` feature shift exceeds the ML overlay tolerance. Retrain `models/ml_order_overlay/phase2_8` with the new preprocessor before enabling the overlay.

