# Historical July 2026 comparison (Issue #74)

The original `scripts/compare_bt_vs_actual.py` is retained here as **historical source evidence**, not a supported analysis command. It compared a legacy V1-style `SectorRelativeEnsembleBLPEnhancedModel` backtest with a July 2026 ProductionV2 live wallet proxy. No current V2 performance conclusion should be inferred from its hard-coded output. Historical output such as “BT +5.20% vs Actual -3.64%” is a period-specific, non-like-for-like claim, not a verified measurement for the present strategy.

Original sources (commit `4c370755`):
- Close snapshots: `var/results/202607*_production_close_positions/wallet_close_*.json`.
- Decision snapshots: `var/results/202607*_production_decision_v2/wallet_decision_*.json`.
- Position snapshots: `var/results/202607*_production_close_positions/positions_close_*.json`.
- Legacy V1 return CSV: `var/legacy_results_src/production_backtest/daily_net_returns.csv` (not a canonical V2 backtest).
- Original workstation prefix: `/Users/shonen/leadlag/`. These inputs were **not** bundled in this archive; see original commit for code and history. No evidence shows either old script was a daily/CI entry point.

The historical script is preserved for provenance only: it executes reads at import time and depends on the author's old absolute paths. Do **not** execute/import it as a generic CLI. Its printed root-cause ranking and model assumptions are historical hypotheses, not current independently validated findings. The live `ukeire_hosyoukin` wallet field is a margin/cash proxy rather than realized net account P&L; gaps, transfers, valuation intervals and missing sessions prevent equivalence to true daily returns.

For exploratory statistics on **available current snapshots**, use:
```bash
uv run --locked python tools/research/calculate_live_sharpe.py --start-date 2026-07-01 --end-date 2026-07-31
uv run --locked python tools/research/calculate_live_sharpe.py --results-dir /absolute/path/to/var/results --start-date 2026-07-01 --end-date 2026-07-31
```
This new command is offline and uses `leadlag.config.paths.results()` by default. It is not a ledger reconciliation or a valid production Sharpe estimate. Formal ledger / earnings reconciliation remains outside this issue (#25/#37).
