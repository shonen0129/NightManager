# ADR-0007: Sensitivity Labels From the Ticker Registry

- Date: 2026-08-09
- Status: accepted
- Deciders: Devin

## Context

`core/correlation.py` had 32-dim hard-coded arrays for sensitivity labels `w3`
to `w6`. `AGENTS.md` invariant #5 explicitly warns that any universe change in
`ticker.py` must also update the hard-coded arrays in `correlation.py`.

## Decision

Move the per-ticker sensitivity values to `SENSITIVITY_LABELS` in
`leadlag/data/tickers.py`. `get_static_sensitivity_labels()` now builds the four
`(N_TOTAL,)` arrays by iterating `US_TICKERS + JP_TICKERS` and looking up each
label in the registry.

## Consequences

- Universe changes require updates in only one file: `tickers.py`.
- A regression test pins the generated arrays against the legacy hard-coded
  values to prevent silent drift.
- The domain knowledge is preserved; it is simply expressed per-ticker.
