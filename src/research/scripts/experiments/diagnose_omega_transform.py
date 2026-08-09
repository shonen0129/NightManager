#!/usr/bin/env python3
"""Trace which transformed variable becomes non-finite in Step 2."""
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


def _load_omega_struct_for_sig(sig_dt: pd.Timestamp, dist_dir: Path, dt: pd.Timestamp):
    """Replicate Step 2 omega_struct fallback logic."""
    matrices_dir = dist_dir / "matrices"
    sig_dt_str = sig_dt.strftime("%Y%m%d")
    dt_str = dt.strftime("%Y%m%d")
    f = matrices_dir / f"omega_struct_{sig_dt_str}.npy"
    if not f.exists():
        import re

        fallback_files = sorted(
            p for p in matrices_dir.glob("omega_struct_*.npy")
            if re.fullmatch(r"omega_struct_(\d{8})", p.stem)
        )
        fallback_files = [p for p in fallback_files if p.stem.split("_")[-1] <= sig_dt_str]
        if fallback_files:
            f = fallback_files[-1]
        else:
            f = matrices_dir / f"omega_struct_{dt_str}.npy"
    return np.load(f), f.name


def main():
    cfg = yaml.safe_load((ROOT / "configs/production/production.yaml").read_text())
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    df_exec = load_df_exec_from_local_cache()
    inputs = model._prepare_common_inputs(df_exec)
    c = model.gap_open_coef
    b = model.topix_beta_coef

    dist_dir = ROOT / "var/outputs/long_period/omega_struct"
    fallback_dates = [
        "2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31",
        "2025-11-18", "2025-12-01", "2025-12-30", "2026-01-05", "2026-01-26",
    ]

    for d in fallback_dates:
        dt = pd.Timestamp(d)
        i = df_exec.index.get_indexer([dt])[0]
        if i == -1:
            print(f"{d}: not in df_exec")
            continue
        sig = pd.to_datetime(df_exec["sig_date"].values[i]).normalize()
        gap_override = np.nan_to_num(inputs["jp_gap"][i], nan=0.0)
        betas_t = np.asarray(inputs["jp_beta"][i], dtype=float)
        topix_night_t = float(inputs["topix_night"][i])

        res = model.compute_blp_signal(
            inputs["jp_res_returns_p3"], i,
            gap_override=gap_override, betas_t=betas_t, topix_night_t=topix_night_t,
            rolling_std=None, v0_static=inputs["v0_static"], c_full=inputs["c_full_p3"],
            is_residual=True, return_matrices=True,
        )

        mu_Y = res["mu_Y"]
        sigma_Y = res["sigma_Y"]
        z_hat_j = res["z_hat_j_t1"]
        sigma_Y_denorm = res["sigma_Y_denorm"]
        mu_raw = mu_Y + sigma_Y * z_hat_j

        Omega_struct, omega_name = _load_omega_struct_for_sig(sig, dist_dir, dt)
        Omega_raw = np.diag(sigma_Y_denorm) @ Omega_struct @ np.diag(sigma_Y_denorm)

        gap_syst = betas_t * topix_night_t
        gap_idio = gap_override - gap_syst
        gap_filt = c * gap_idio + (c - b) * gap_syst
        denominator = 1.0 + gap_filt
        denominator_floored = np.maximum(denominator, 0.1)
        D_gap = np.diag(1.0 / denominator_floored)

        mu_gap = (1.0 + mu_raw) / denominator_floored - 1.0
        Omega_gap = D_gap @ Omega_raw @ D_gap

        checks = {
            "mu_Y": mu_Y,
            "sigma_Y": sigma_Y,
            "sigma_Y_denorm": sigma_Y_denorm,
            "z_hat_j_t1": z_hat_j,
            "mu_raw": mu_raw,
            "Omega_struct_loaded": Omega_struct,
            "Omega_raw": Omega_raw,
            "denominator": denominator,
            "denominator_floored": denominator_floored,
            "mu_gap": mu_gap,
            "Omega_gap": Omega_gap,
        }

        print(f"{d} (sig={sig.date()}, omega={omega_name}):")
        for name, arr in checks.items():
            finite = bool(np.isfinite(arr).all())
            print(
                f"  {name}: finite={finite}, nan={np.isnan(arr).sum()}, inf={np.isinf(arr).sum()}, "
                f"min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}"
            )


if __name__ == "__main__":
    main()
