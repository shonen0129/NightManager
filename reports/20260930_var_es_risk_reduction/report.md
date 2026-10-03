# VaR/ES risk reduction through lower side leverage

## Decision

Selected existing side_leverage setting: **1.30**. The fixed 250-day historical sample clears both unchanged risk stops, while the model-space weights and audit outcomes are reused unchanged. The effective maximum gross falls from 3.000 to 2.600; the inherited risk gross cap remains 3.000.

The selection rule was fixed before the replay: choose the highest tested leverage strictly below both existing stop thresholds and within the existing effective-gross cap. This leaves the risk stop enabled; it does not change VaR/ES limits.

## Same-input comparison

| side_leverage | VaR99 | ES99 | compounded net return | net Sharpe | max drawdown | modeled total cost |
|---:|---:|---:|---:|---:|---:|---:|
| 1.50 | 3.053% | 4.591% | 218.758% | 4.307 | -33.401% | 67.604% |
| 1.30 | 2.646% | 3.979% | 174.307% | 4.307 | -29.556% | 58.590% |
| 1.25 | 2.544% | 3.826% | 164.143% | 4.307 | -28.567% | 56.336% |

Cost values above are simple sums of daily modeled return fractions, not compounded returns or observed broker fees.

| Modeled cost component | Sum of daily return fractions at 1.30 |
|---|---:|
| Slippage | 50.437% |
| Financing | 2.506% |
| Borrow | 0.768% |
| Reverse fee | 4.878% |
| Total | 58.590% |

The artifact-segment baseline PnL replay differs from the saved audited daily returns by at most 0. The saved history spans 2 immutable overlay artifact segments, so each segment was replayed separately across the existing state-reset boundary. The PnL implementation scales market returns, slippage, financing, borrow, and reverse costs by the same leverage factor; candidate outputs were checked against this linear identity.

## Attribution

The 269-day sample is 2025-07-29 through 2026-09-25. Overlay was applied on 244 days and skipped on 25 days; fallback days are 0. Gross contribution by ticker and long/short side, PIT multiplier band, realized gap-impact quartile, and all cost categories are in the CSV and JSON artifacts beside this report.

The worst baseline day was 2026-07-30 at -6.249%; overnight PnL was -4.318%, including -3.513% from shorts. The two largest ticker contributions on that day were 1625.T (short, combined gross -2.676%) and 1623.T (short, combined gross -1.303%). These are ex-post attribution, not decision-time signals.

PIT labels are inferred from saved gross weights and are reported as 0.75 multiplier versus full multiplier. Realized gap-impact quartiles are ex-post attribution buckets. The report does not use these realized labels to change daily decisions.

## Provenance and limits

- df_exec fingerprint: `6b81e5ae4abd09578365618caf218e6375b863a9f6ecd18d7858c83b988166a5`
- gap input fingerprint: `c99ca4e8d8e0ad75`
- current 250-day risk window ends: 2026-09-25; ES tail count: 3
- resolved production side_leverage at replay: 1.30; fixed historical replay baseline: 1.50
- risk stops remain VaR99 3.00% and ES99 4.00%; neither threshold nor fail-closed handling changed.
- the saved macro snapshot ends 2026-09-25 and provider point-in-time availability is not proven; no newer market data were claimed.
- modeled fees are not actual account PnL or a complete fill-cost reconciliation.
- this historical result does not establish that the current live risk window passes or replace the daily fail-closed risk check.

Reproduction: `.venv/bin/python src/research/scripts/experiments/evaluate_var_es_side_leverage_20260930.py`.
