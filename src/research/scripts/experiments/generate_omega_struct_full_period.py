#!/usr/bin/env python3
"""Generate Step 1 omega_struct matrices for the full history without re-running diagnostics.

Uses the local execution DataFrame cache and Residual-BLPX compute_blp_signal output
to build the standardized prediction-error covariance Omega_struct for every date.
The resulting matrices can be consumed by compute_gap_adjusted_distribution.py as a
custom distribution-input-dir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-history omega_struct matrices for Step 1 fallback."
    )
    parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--start",
        default="2015-01-05",
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="latest",
        help="End date (YYYY-MM-DD) or 'latest'.",
    )
    parser.add_argument(
        "--output-dir",
        default="var/outputs/long_period/omega_struct",
        help="Output directory for matrices.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers. 1 = sequential.",
    )
    return parser.parse_args()


def _omega_from_blp_res(res: dict) -> np.ndarray:
    """Compute standardized Omega_struct from compute_blp_signal matrix outputs."""
    Sigma_XX = res["Sigma_XX"]
    Sigma_YX = res["Sigma_YX"]
    Sigma_YY = res["Sigma_YY"]
    B_struct = res["B_struct"]
    Sigma_XY = Sigma_YX.T

    Omega_struct = (
        Sigma_YY
        - B_struct @ Sigma_XY
        - Sigma_YX @ B_struct.T
        + B_struct @ Sigma_XX @ B_struct.T
    )
    Omega_struct = 0.5 * (Omega_struct + Omega_struct.T)
    return Omega_struct


def _process_date(dt: pd.Timestamp, df_exec: pd.DataFrame, model, inputs: dict) -> tuple[str, np.ndarray] | None:
    i = df_exec.index.get_indexer([dt])[0]
    if i == -1:
        return None
    if i < model.corr_window:
        return None

    df_exec["sig_date"].values[i]
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
        logger.warning("Error computing BLP signal on %s: %s", dt.strftime("%Y-%m-%d"), e)
        return None

    Omega_struct = _omega_from_blp_res(res)
    if not np.isfinite(Omega_struct).all():
        logger.warning("Non-finite values in Omega_struct on %s", dt.strftime("%Y-%m-%d"))
        return None

    dt_str = dt.strftime("%Y%m%d")
    return dt_str, Omega_struct


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = out_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from local cache...")
    df_exec = load_df_exec_from_local_cache()

    end_date = df_exec.index[-1] if args.end == "latest" else pd.Timestamp(args.end)
    sim_dates = df_exec.index[(df_exec.index >= args.start) & (df_exec.index <= end_date)]
    logger.info("Target window: %s to %s (%d dates)", sim_dates[0], sim_dates[-1], len(sim_dates))

    with open(ROOT / args.config) as f:
        import yaml

        cfg = yaml.safe_load(f)

    logger.info("Instantiating Residual-BLPX model...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    inputs = model._prepare_common_inputs(df_exec)

    if args.n_jobs == 1:
        for k, dt in enumerate(sim_dates, 1):
            result = _process_date(dt, df_exec, model, inputs)
            if result is None:
                continue
            dt_str, Omega_struct = result
            np.save(matrices_dir / f"omega_struct_{dt_str}.npy", Omega_struct)
            if k % 100 == 0:
                logger.info("Generated %d/%d matrices", k, len(sim_dates))
    else:
        logger.info("Running parallel n_jobs=%d", args.n_jobs)
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(_process_date)(dt, df_exec, model, inputs) for dt in sim_dates
        )
        for result in results:
            if result is None:
                continue
            dt_str, Omega_struct = result
            np.save(matrices_dir / f"omega_struct_{dt_str}.npy", Omega_struct)

    logger.info("Done. Matrices written to %s", matrices_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
