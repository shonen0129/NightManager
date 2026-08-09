# Phase 23: V1 Backtest Deprecation

- CLI `backtest` unified on `BacktestEngine.run_v2_backtest`.
- Legacy `run_backtest` and V1 SRE model moved to `archive-2026-08` / `research.backtest_v1`.
- Flat position (w_final=0) on missing gap data; no V1 fallback.
