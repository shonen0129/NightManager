#!/usr/bin/env python3
"""Run 98 subsector BLPX signal with reviewed sensitivity labels and evaluate IC."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

# Canonical 17 TOPIX-17 ETF tickers
_ORIGINAL_JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T",
]

# Load 98 subsector configuration
with open(ROOT / "configs" / "research" / "subsector_98_config.yaml", "r", encoding="utf-8") as f:
    subsector_config = yaml.safe_load(f)

SUBSECTOR_NAMES = [s["name"] for s in subsector_config["subsectors"]]
SUBSECTOR_LABELS = {s["name"]: s["sensitivity_labels"] for s in subsector_config["subsectors"]}

print(f"Subsector count: {len(SUBSECTOR_NAMES)}")

# Load aggregation matrix
with open(ROOT / "configs" / "research" / "subsector_98_aggregation.yaml", "r", encoding="utf-8") as f:
    agg_config = yaml.safe_load(f)

A = np.array(agg_config["aggregation_matrix"])  # Shape: (17, 99)
print(f"Aggregation matrix shape: {A.shape}")

import leadlag.data.tickers as _tickers
import leadlag.data.preprocessor as _preprocessor
import leadlag.core.pipeline as _pipeline
import leadlag.core.correlation as _correlation
import leadlag.models.blpx.prior_builder as _prior_builder
import leadlag.models.blpx.model_predict as _model_predict


def _set_jp_universe(names: list[str], is_subsector: bool = False) -> None:
    """Switch the in-process JP ticker list used by BLPX helpers."""
    if not is_subsector:
        importlib.reload(_tickers)
    _tickers.JP_TICKERS = names
    _preprocessor.JP_TICKERS = names
    _pipeline.JP_TICKERS = names
    _prior_builder.JP_TICKERS = names
    _model_predict.JP_TICKERS = names
    if is_subsector:
        # Load 98 subsector sensitivity labels
        _original_labels = _tickers.SENSITIVITY_LABELS.copy()
        for ticker in _ORIGINAL_JP_TICKERS:
            if ticker in _tickers.SENSITIVITY_LABELS:
                del _tickers.SENSITIVITY_LABELS[ticker]
        for subsector in SUBSECTOR_NAMES:
            if subsector in SUBSECTOR_LABELS:
                _tickers.SENSITIVITY_LABELS[subsector] = SUBSECTOR_LABELS[subsector]
        return _original_labels
    return None


def _restore_jp_universe(original_labels: dict[str, dict[str, float]]) -> None:
    """Restore original JP universe and sensitivity labels."""
    importlib.reload(_tickers)
    _tickers.SENSITIVITY_LABELS = original_labels
    _tickers.JP_TICKERS = _ORIGINAL_JP_TICKERS
    _preprocessor.JP_TICKERS = _ORIGINAL_JP_TICKERS
    _pipeline.JP_TICKERS = _ORIGINAL_JP_TICKERS
    _prior_builder.JP_TICKERS = _ORIGINAL_JP_TICKERS
    _model_predict.JP_TICKERS = _ORIGINAL_JP_TICKERS


# Build 98 subsector panel from individual tickers
print("Building 98 subsector panel...")
_all_returns = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_vw.parquet")

# Create subsector returns by averaging individual ticker returns
subsector_returns = {}
for sub in subsector_config["subsectors"]:
    sub_tickers = [t["ticker"] for t in sub["tickers"]]
    # Check if tickers exist in the returns data
    available_tickers = [t for t in sub_tickers if f"jp_cc_{t}" in _all_returns.columns]
    if available_tickers:
        sub_cols = [f"jp_cc_{t}" for t in available_tickers]
        subsector_returns[sub["name"]] = _all_returns[sub_cols].mean(axis=1)
    else:
        print(f"Warning: No tickers found for subsector {sub['name']}")

df_subsector = pd.DataFrame(subsector_returns)
print(f"df_subsector shape: {df_subsector.shape}")

# Replace the original subsector panel with our 98 subsector version
df_subsector.to_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_98_oc_vw.parquet")

# Switch to subsector universe
original_labels = _set_jp_universe(SUBSECTOR_NAMES, is_subsector=True)

try:
    # Run subsector BLPX
    print("Running 98 subsector BLPX...")
    from leadlag.core.pipeline import run_blpx_pipeline

    result_sub = run_blpx_pipeline(
        start_date="2015-01-05",
        param_set="subsector_98_experiment",
        raw_pca_weight=0.0,
        residual_pca_weight=0.0,
        raw_blpx_weight=0.0,
        residual_blpx_weight=1.0,
        minvar_enabled=False,
        macro_confidence_enabled=False,
        macro_kappa_enabled=False,
        macro_direction_enabled=False,
        copula_enabled=False,
    )

    signal_sub = result_sub["weights"]
    print(f"signal_sub shape: {signal_sub.shape}")

    # Aggregate to 17 ETF space
    signal_17_from_sub = signal_sub @ A.T
    print(f"signal_17_from_sub shape: {signal_17_from_sub.shape}")

finally:
    # Restore original universe
    _restore_jp_universe(original_labels)

# Run direct 17-dim baseline
print("Running direct 17-dim BLPX baseline...")
result_17 = run_blpx_pipeline(
    start_date="2015-01-05",
    param_set="direct_17_dim",
    raw_pca_weight=0.0,
    residual_pca_weight=0.0,
    raw_blpx_weight=0.0,
    residual_blpx_weight=1.0,
    minvar_enabled=False,
    macro_confidence_enabled=False,
    macro_kappa_enabled=False,
    macro_direction_enabled=False,
    copula_enabled=False,
)

signal_17_direct = result_17["weights"]
print(f"signal_17_direct shape: {signal_17_direct.shape}")

# Align dates
common_dates = signal_17_from_sub.index.intersection(signal_17_direct.index)
signal_17_from_sub = signal_17_from_sub.loc[common_dates]
signal_17_direct = signal_17_direct.loc[common_dates]

# Load returns for IC calculation
_all_returns = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_vw.parquet")
jp_returns_17 = _all_returns[[c for c in _all_returns.columns if c.startswith("jp_cc_")]]
jp_returns_17 = jp_returns_17.loc[common_dates]

# Compute per-ETF rank IC
subsector_rank_ic = []
for i, ticker in enumerate(_ORIGINAL_JP_TICKERS):
    signal_rank = signal_17_from_sub.iloc[:, i].rank(pct=True)
    returns_rank = jp_returns_17.iloc[:, i].rank(pct=True)
    ic = signal_rank.corr(returns_rank)
    if not np.isnan(ic):
        subsector_rank_ic.append(ic)

direct_rank_ic = []
for i, ticker in enumerate(_ORIGINAL_JP_TICKERS):
    signal_rank = signal_17_direct.iloc[:, i].rank(pct=True)
    returns_rank = jp_returns_17.iloc[:, i].rank(pct=True)
    ic = signal_rank.corr(returns_rank)
    if not np.isnan(ic):
        direct_rank_ic.append(ic)

# Compute daily cross-sectional IC
subsector_daily_ic = []
for date in signal_17_from_sub.index:
    daily_ic = signal_17_from_sub.loc[date].corr(jp_returns_17.loc[date])
    if not np.isnan(daily_ic):
        subsector_daily_ic.append(daily_ic)

# Results
results = {
    "subsector_98_aggregated": {
        "mean_rank_ic": float(np.mean(subsector_rank_ic)) if subsector_rank_ic else np.nan,
        "median_rank_ic": float(np.median(subsector_rank_ic)) if subsector_rank_ic else np.nan,
        "mean_daily_ic": float(np.mean(subsector_daily_ic)) if subsector_daily_ic else np.nan,
    },
    "direct_17_dim": {
        "mean_rank_ic": float(np.mean(direct_rank_ic)) if direct_rank_ic else np.nan,
        "median_rank_ic": float(np.median(direct_rank_ic)) if direct_rank_ic else np.nan,
    },
    "summary": {
        "mean_ic_aggregated": float(np.mean(subsector_rank_ic)) if subsector_rank_ic else np.nan,
        "median_ic_aggregated": float(np.median(subsector_rank_ic)) if subsector_rank_ic else np.nan,
        "mean_daily_ic_aggregated": float(np.mean(subsector_daily_ic)) if subsector_daily_ic else np.nan,
        "mean_ic_direct_17": float(np.mean(direct_rank_ic)) if direct_rank_ic else np.nan,
        "median_ic_direct_17": float(np.median(direct_rank_ic)) if direct_rank_ic else np.nan,
    }
}

print("\nSummary:")
print(json.dumps(results["summary"], indent=2))

# Save results
output_dir = ROOT / "reports" / "subsector_refinement" / "phase2_blpx"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "subsector_98_ic_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_dir / 'subsector_98_ic_results.json'}")
