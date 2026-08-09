#!/usr/bin/env python3
"""Diagnose which BLP signal variables are non-finite on fallback dates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

CFG_PATH = ROOT / "configs/production/production.yaml"

def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    df_exec = load_df_exec_from_local_cache()
    inputs = model._prepare_common_inputs(df_exec)

    fallback_dates = [
        "2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31",
        "2025-11-18", "2025-12-01", "2026-01-05", "2026-01-26",
    ]
    for d in fallback_dates:
        dt = pd.Timestamp(d)
        i = df_exec.index.get_indexer([dt])[0]
        if i == -1:
            print(f"{d}: not in df_exec")
            continue
        gap_override = np.nan_to_num(inputs["jp_gap"][i], nan=0.0)
        betas_t = np.asarray(inputs["jp_beta"][i], dtype=float)
        topix_night_t = float(inputs["topix_night"][i])
        try:
            res = model.compute_blp_signal(
                inputs["jp_res_returns_p3"],
                i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
                rolling_std=None,
                v0_static=inputs["v0_static"],
                c_full=inputs["c_full_p3"],
                is_residual=True,
                return_matrices=True,
            )
        except Exception as e:
            print(f"{d}: EXCEPTION {e}")
            continue
        parts = []
        for k in ["mu_Y", "sigma_Y", "sigma_Y_denorm", "z_hat_j_t1", "signal"]:
            arr = res.get(k)
            if arr is None:
                continue
            finite = bool(np.isfinite(arr).all())
            parts.append(
                f"{k}=(finite={finite}, nan={np.isnan(arr).sum()}, "
                f"inf={np.isinf(arr).sum()}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g})"
            )
        print(f"{d}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
