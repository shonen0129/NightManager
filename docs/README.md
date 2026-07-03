# Sector Relative Ensemble (PCA-Ensemble) Strategy

This repository contains the codebase and tools for the **Sector Relative Ensemble (PCA-Ensemble)** strategy, which is the official production trading model of the Lead-Lag Market-Neutral Fund.

PCA-Ensemble was previously referred to as the `Raw-PCA/Residual-PCA signal-level 50/50 ensemble`. The naming with experimental numbers (Raw-PCA, Residual-PCA, P5, P6) is deprecated.

## Model Overview
PCA-Ensemble combines the standard Production signal (Raw-PCA) with a TOPIX-residualized Production target signal (Residual-PCA) at the signal level. It uses a cross-sectional Z-score normalization for both components before taking a 50/50 average. Portfolio weights are built using the canonical signal-weighted allocator.

> [!WARNING]
> Uniform/equal-weighting is strictly forbidden in PCA-Ensemble. The model must always use `signals.build_weights(..., weight_mode="signal")` to maintain high risk-adjusted Sharpe profiles.

## Data Requirements
Execution requires daily US sector ETF returns and Japanese TOPIX-17 sector ETF prices, as well as TOPIX index price series for OLS residualization. These can be fetched automatically via yfinance or kabuステーション API.

## Configuration File
The canonical production configuration is stored in:
`configs/production/production.yaml`

Do not create dedicated model configurations like `configs/sector_relative_ensemble.yaml`. Maintain PCA-Ensemble parameters directly inside `configs/production/production.yaml`.

## Commands

### Historical Backtesting
To run a historical backtest of the PCA-Ensemble model under a 5 bps slippage assumption:
```bash
python tools/research/backtest_sector_relative_ensemble.py \
    --config configs/production/production.yaml \
    --slippage-bps 5 \
    --output-dir results/sector_relative_ensemble/
```

### Daily Dry Run / Execution
To run PCA-Ensemble for daily signal and order generation:
```bash
python tools/research/run_daily_sector_relative_ensemble.py \
    --config configs/production/production.yaml \
    --signal-date latest \
    --output-dir live/sector_relative_ensemble/ \
    --dry-run
```

## Output Files

### Backtest Outputs (`results/sector_relative_ensemble/`)
- `metrics_summary_train.csv`, `metrics_summary_oos.csv`, `metrics_summary_full.csv`
- `daily_gross_returns.csv`, `daily_costs.csv`, `daily_net_returns.csv`, `daily_equity_curve.csv`, `daily_drawdown.csv`, `daily_turnover.csv`
- `signals.csv`, `normalized_signals.csv`, `weights.csv`, `positions.csv`
- `component_signal_correlation.csv`, `component_rank_correlation.csv`, `component_signal_agreement.csv`, `top_drawdown_periods.csv`, `slippage_sensitivity.csv`
- `equity_curve.png`, `drawdown.png`, `rolling_sharpe.png`, `rolling_ic.png`, `turnover.png`, `signal_heatmap.png`
- `final_report.md`

### Daily Live Outputs (`live/sector_relative_ensemble/`)
- `latest_signal.csv`
- `latest_weights.csv`
- `latest_orders.csv`
- `latest_audit.json`
- `run_log.txt`

## Safety Audits
The backtest automatically triggers a set of safety checks written to `results/sector_relative_ensemble/audit/`:
- `baseline_definition_audit.csv`
- `signal_weighting_audit.csv`
- `date_alignment_audit.csv`
- `residualization_leakage_audit.csv`
- `cost_consistency_audit.csv`
- `weight_constraint_audit.csv`
- `ticker_order_audit.csv`
- `config_audit.csv`

## Deprecated Experiments
All deprecated experimental scripts (P2, P5, P6, Gap Shrinkage, Low-Rank models, risk overlays, etc.) have been archived under `archive/experiments/`. For details on why these were not adopted, see:
[deprecated_experiments.md](deprecated_experiments.md)
