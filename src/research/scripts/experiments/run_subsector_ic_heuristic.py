#!/usr/bin/env python3
"""Run subsector BLPX signal with heuristic sensitivity labels and evaluate IC.

Uses heuristic sensitivity labels instead of dominant ETF inheritance.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

# Canonical 17 TOPIX-17 ETF tickers
_ORIGINAL_JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T",
]

_subsector_oc = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_vw.parquet")
SUBSECTOR_NAMES = [str(c) for c in _subsector_oc.columns]

import leadlag.data.tickers as _tickers
import leadlag.data.preprocessor as _preprocessor
import leadlag.data.intraday_inputs as _intraday_inputs
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
        # Load heuristic sensitivity labels
        with open(ROOT / "configs" / "research" / "subsector_sensitivity_labels_heuristic.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            heuristic_labels = config["sensitivity_labels"]

        # Apply heuristic labels to subsectors
        for k, sub in enumerate(SUBSECTOR_NAMES):
            if sub in heuristic_labels:
                _tickers.SENSITIVITY_LABELS[sub] = heuristic_labels[sub]
            else:
                # Fallback to dominant ETF if not in heuristic mapping
                etf_idx = int(np.argmax(A[:, k]))
                etf = ETF_ROWS[etf_idx]
                if etf in _tickers.SENSITIVITY_LABELS:
                    _tickers.SENSITIVITY_LABELS[sub] = dict(_tickers.SENSITIVITY_LABELS[etf])

        # Remove stale 17-ETF JP sensitivity entries
        for tk in _ORIGINAL_JP_TICKERS:
            _tickers.SENSITIVITY_LABELS.pop(tk, None)

    importlib.reload(_preprocessor)
    importlib.reload(_correlation)
    importlib.reload(_pipeline)
    importlib.reload(_prior_builder)


def _prepare_common_inputs_sub(
    self: ProductionBLPXModel,
    df_exec: pd.DataFrame,
    *,
    horizon: int = 1,
    p_910_df: pd.DataFrame | None = None,
    y_jp_target: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build CommonInputs for the 79-dimensional subsector universe."""
    from leadlag.core.pipeline import build_common_inputs

    cache_key = (
        df_exec.index[0],
        df_exec.index[-1],
        df_exec.shape,
        hash(tuple(df_exec.columns)),
        horizon,
    )
    common_inputs_cache = self._cache_manager.namespace("common_inputs")
    if cache_key in common_inputs_cache:
        return common_inputs_cache[cache_key]

    if y_jp_target is None:
        y_jp_target = _intraday_inputs.compute_jp_target_returns(
            df_exec, SUBSECTOR_NAMES, horizon=horizon, p_910_df=p_910_df
        )

    inputs = build_common_inputs(
        df_exec,
        y_jp_target,
        n_u=self.n_u,
        n_j=self.n_j,
        ewma_half_life=self.ewma_halflife,
        beta_window=self.beta_window,
        include_v4_prior=self.include_v4_prior,
        us_res_enabled=self.us_res_enabled,
        us_res_gamma=self.us_res_gamma,
        us_res_beta_window=self.us_res_beta_window,
        frac_diff_enabled=self.frac_diff_enabled,
        frac_diff_d=self.frac_diff_d,
        frac_diff_threshold=self.frac_diff_threshold,
        frac_diff_window=self.frac_diff_window,
        frac_diff_normalize=self.frac_diff_normalize,
    )
    out = inputs.to_dict()
    out["y_jp_target"] = y_jp_target
    common_inputs_cache[cache_key] = out
    return out


_ORIGINAL_SECTOR_MAP_FOR_RESTORE = dict(_prior_builder._SECTOR_MAPPING_STRUCTURE)

with open(ROOT / "configs" / "research" / "subsector_aggregation_vw.yaml", "r", encoding="utf-8") as f:
    _agg = yaml.safe_load(f)
A = np.array(_agg["aggregation_matrix_A"]["data"])
ETF_ROWS = _agg["aggregation_matrix_A"]["rows"]

_ORIGINAL_SECTOR_MAP = {
    "XLB": ["1620.T", "1623.T"],
    "XLC": ["1626.T"],
    "XLE": ["1618.T", "1627.T"],
    "XLF": ["1631.T", "1632.T"],
    "XLI": ["1624.T", "1622.T", "1626.T"],
    "XLK": ["1626.T", "1625.T"],
    "XLP": ["1617.T", "1630.T"],
    "XLRE": ["1633.T"],
    "XLU": ["1627.T"],
    "XLV": ["1621.T"],
    "XLY": ["1630.T", "1626.T", "1622.T"],
    "MTUM": ["1625.T", "1626.T"],
    "VLUE": ["1631.T", "1632.T", "1623.T", "1622.T"],
    "IUSG": ["1626.T", "1625.T"],
    "USMV": ["1617.T", "1621.T", "1627.T"],
}

_SUBSECTOR_SECTOR_MAP: dict[str, list[str]] = {}
for us_tk, etfs in _ORIGINAL_SECTOR_MAP.items():
    subs = []
    for etf in etfs:
        if etf in ETF_ROWS:
            etf_idx = ETF_ROWS.index(etf)
            for k, sub in enumerate(SUBSECTOR_NAMES):
                if A[etf_idx, k] > 0:
                    subs.append(sub)
    _SUBSECTOR_SECTOR_MAP[us_tk] = subs

from leadlag.config.schemas import BLPXConfig
from leadlag.models.blpx.model import ProductionBLPXModel

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, US_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank IC for two aligned 1-D arrays (NaNs dropped)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 10:
        return np.nan
    return float(df["x"].corr(df["y"], method="spearman"))


def _compute_ics(signal_df: pd.DataFrame, oc_df: pd.DataFrame) -> tuple[list[float], float, float]:
    ics = []
    for tk in signal_df.columns:
        ic = _rank_ic(signal_df[tk].values, oc_df[tk].values)
        ics.append(ic)
    arr = np.array([i for i in ics if not np.isnan(i)])
    return ics, float(np.nanmean(arr)), float(np.nanmedian(arr))


def main() -> int:
    df_exec_sub = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "df_exec_subsector_vw.parquet")
    logger.info("df_exec_sub shape: %s", df_exec_sub.shape)

    _set_jp_universe(_ORIGINAL_JP_TICKERS)
    _prior_builder._SECTOR_MAPPING_STRUCTURE = _ORIGINAL_SECTOR_MAP_FOR_RESTORE
    ProductionBLPXModel._SECTOR_MAPPING_STRUCTURE = _ORIGINAL_SECTOR_MAP_FOR_RESTORE

    data = download_data(start_date="2009-01-01", end_date="2026-12-31", force=False)
    df_exec_17 = preprocess_data(data)

    _set_jp_universe(SUBSECTOR_NAMES, is_subsector=True)
    _prior_builder._SECTOR_MAPPING_STRUCTURE = _SUBSECTOR_SECTOR_MAP
    ProductionBLPXModel._SECTOR_MAPPING_STRUCTURE = _SUBSECTOR_SECTOR_MAP
    logger.info("US tickers: %d, JP tickers: %d", len(US_TICKERS), len(JP_TICKERS))

    cfg = BLPXConfig(
        n_u=len(US_TICKERS),
        n_j=len(SUBSECTOR_NAMES),
        param_set="subsector_experiment_heuristic",
        model_name="SubsectorBLPXModel",
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
    model = ProductionBLPXModel(cfg)
    model._prepare_common_inputs = lambda df_exec, **kwargs: _prepare_common_inputs_sub(model, df_exec, **kwargs)
    logger.info("Subsector model n_u=%d n_j=%d", model.n_u, model.n_j)

    pkg_sub = model.predict_signals(df_exec_sub, n_jobs=1)
    signal_sub = pkg_sub.residual_blpx_signals.values
    if signal_sub is None:
        logger.error("No residual_blpx signal in package")
        return 1
    logger.info("signal_sub shape: %s", signal_sub.shape)

    signal_17_from_sub = signal_sub @ A.T
    signal_17_df = pd.DataFrame(signal_17_from_sub, index=df_exec_sub.index, columns=ETF_ROWS)

    oc_17 = df_exec_17[[f"jp_oc_{tk}" for tk in ETF_ROWS]].rename(columns=lambda c: c.replace("jp_oc_", ""))
    oc_17 = oc_17.reindex(signal_17_df.index).dropna(how="all")
    signal_17_df = signal_17_df.loc[oc_17.index]

    ics, mean_ic, median_ic = _compute_ics(signal_17_df, oc_17)
    mean_daily_ic = _rank_ic(signal_17_df.mean(axis=1).values, oc_17.mean(axis=1).values)

    _set_jp_universe(_ORIGINAL_JP_TICKERS)
    _prior_builder._SECTOR_MAPPING_STRUCTURE = _ORIGINAL_SECTOR_MAP_FOR_RESTORE
    ProductionBLPXModel._SECTOR_MAPPING_STRUCTURE = _ORIGINAL_SECTOR_MAP_FOR_RESTORE

    cfg_17 = BLPXConfig(
        n_u=len(US_TICKERS),
        n_j=17,
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
    model_17 = ProductionBLPXModel(cfg_17)
    pkg_17 = model_17.predict_signals(df_exec_17, n_jobs=1)
    signal_17_direct = pkg_17.residual_blpx_signals.values
    signal_17_direct_df = pd.DataFrame(signal_17_direct, index=df_exec_17.index, columns=ETF_ROWS)
    signal_17_direct_df = signal_17_direct_df.reindex(signal_17_df.index)

    ics_direct, mean_ic_direct, median_ic_direct = _compute_ics(signal_17_direct_df, oc_17)

    summary = {
        "mean_ic_aggregated": mean_ic,
        "median_ic_aggregated": median_ic,
        "mean_daily_ic_aggregated": mean_daily_ic,
        "mean_ic_direct_17": mean_ic_direct,
        "median_ic_direct_17": median_ic_direct,
    }

    out_dir = ROOT / "reports" / "subsector_refinement" / "phase2_blpx"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "subsector_ic_summary_heuristic.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# Phase 2: Subsector BLPX IC Evaluation (Heuristic Sensitivity Labels)",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"**Subsector df_exec**: {df_exec_sub.shape}",
        f"**Aggregation matrix A**: {A.shape}",
        "",
        "## IC vs realized open-to-close returns (17 TOPIX ETFs)",
        f"- Aggregated subsector BLPX mean IC (per-ETF): {mean_ic:.4f}",
        f"- Aggregated subsector BLPX median IC (per-ETF): {median_ic:.4f}",
        f"- Aggregated subsector BLPX mean daily IC (cross-sectional): {mean_daily_ic:.4f}",
        f"- Direct 17-dim BLPX mean IC (per-ETF): {mean_ic_direct:.4f}",
        f"- Direct 17-dim BLPX median IC (per-ETF): {median_ic_direct:.4f}",
        "",
        "## Per-ETF rank IC",
        "| ETF | Aggregated Subsector (Heuristic) | Direct 17-dim |",
        "|---|---|---|",
    ]
    for i, tk in enumerate(ETF_ROWS):
        report_lines.append(f"| {tk} | {ics[i]:.4f} | {ics_direct[i]:.4f} |")

    (out_dir / "subsector_ic_report_heuristic.md").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
