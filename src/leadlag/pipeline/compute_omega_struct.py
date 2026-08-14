#!/usr/bin/env python
"""Compute Step 1 structured prediction covariance (Omega_struct) per trade date.

This is a minimal, file-on-demand replacement for the Step 1 part of
``tools/research/compute_structured_prediction_covariance.py``.  For each
trade date it:

1. Loads the execution DataFrame ``df_exec``.
2. Builds the residual-BLPX model.
3. Calls ``compute_blp_signal(..., return_matrices=True)``.
4. Reconstructs the standardized JP prediction-error correlation matrix
   ``Omega_struct`` from the returned covariance blocks.
5. Saves ``matrices/omega_struct_YYYYMMDD.npy``.

No plots, no per-date CSVs, no portfolio-level diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Resolve project root. This file lives at src/leadlag/pipeline/compute_omega_struct.py,
# so three parents up is the repository root.
ROOT = Path(__file__).resolve().parents[3]

# Ensure src/ is importable when the script is invoked directly.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.gap_adjustment import _omega_from_blp_res  # noqa: E402
from leadlag.data.fetcher import download_data  # noqa: E402
from leadlag.data.preprocessor import preprocess_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("compute_omega_struct")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Omega_struct for each trade date (Step 1)."
    )
    parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to YAML config file (default: configs/production/production.yaml).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to the first date in df_exec.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (YYYY-MM-DD). Defaults to the last date in df_exec.",
    )
    parser.add_argument(
        "--df-exec",
        default=None,
        help="Optional path to a pickled/csv/parquet df_exec cache.",
    )
    parser.add_argument(
        "--output-dir",
        default="var/live/pipeline_data/distribution_diagnostics/latest",
        help="Output directory for matrices and diagnostics.",
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Write a small JSON diagnostics file for each date.",
    )
    parser.add_argument(
        "--beta-window",
        type=int,
        default=None,
        help="Rolling window for preprocessor betas. "
             "Defaults to config residualization/blpx beta_window or 60.",
    )
    return parser.parse_args()


def _resolve_path(path: str | Path | None) -> Path:
    """Return an absolute Path resolved against the project root if needed."""
    if path is None:
        return ROOT
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _load_config(config_path: Path) -> Any:
    """Load YAML and build an AppConfig.

    The canonical entry point is ``leadlag.execution.config.load_config_from_yaml``,
    which validates V2 sections and produces a flat ``ProductionV2RunConfig``.
    """
    from leadlag.execution.config import load_config_from_yaml

    app_config = load_config_from_yaml(str(config_path))
    logger.info("Loaded AppConfig via load_config_from_yaml")
    return app_config


def _resolve_residualization_params(
    app_config: Any, args: argparse.Namespace
) -> dict[str, Any]:
    """Return preprocessor residualization parameters from the validated config."""
    v2 = app_config.v2
    params: dict[str, Any] = {
        "beta_window": int(
            args.beta_window
            if args.beta_window is not None
            else getattr(v2, "residualization_beta_window", 60)
        ),
        "beta_shrinkage": float(getattr(v2, "residualization_beta_shrinkage", 0.05)),
    }
    winsor = getattr(v2, "residualization_beta_winsor_sigma", 3.0)
    if winsor is not None:
        params["beta_winsor_sigma"] = float(winsor)
    ewma = getattr(app_config, "strategy", None)
    if ewma is not None:
        ewma_hl = getattr(ewma, "beta_ewma_halflife", None)
        if ewma_hl is not None:
            params["beta_ewma_halflife"] = float(ewma_hl)
    return params


def _load_df_exec_from_cache(cache_path: Path) -> pd.DataFrame:
    """Load a df_exec cache from pickle/csv/parquet."""
    suffix = cache_path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        df_exec = pd.read_pickle(cache_path)
    elif suffix == ".parquet":
        df_exec = pd.read_parquet(cache_path)
    elif suffix == ".csv":
        df_exec = pd.read_csv(cache_path, index_col=0)
    else:
        raise ValueError(
            f"Unsupported df_exec cache format: {suffix}. "
            "Use .pkl, .parquet, or .csv."
        )

    if not isinstance(df_exec, pd.DataFrame):
        raise TypeError(f"Loaded df_exec is not a DataFrame: {type(df_exec)}")

    # Normalize the index to tz-naive dates for consistent lookups.
    df_exec.index = pd.to_datetime(df_exec.index, errors="coerce").tz_localize(
        None
    ).normalize()
    df_exec = df_exec.sort_index()
    return df_exec


def _build_df_exec(app_config: Any, args: argparse.Namespace) -> pd.DataFrame:
    """Download raw data and build df_exec through ``preprocess_data``."""
    params = _resolve_residualization_params(app_config, args)
    logger.info("Downloading market data (residualization=%s)...", params)
    raw_data = download_data(beta_window=params["beta_window"])
    logger.info("Preprocessing market data...")
    df_exec = preprocess_data(raw_data, **params)
    if df_exec is None or df_exec.empty:
        raise RuntimeError("preprocess_data returned an empty df_exec")
    return df_exec


def _resolve_df_exec(args: argparse.Namespace, app_config: Any) -> pd.DataFrame:
    """Load df_exec from --df-exec or build it from scratch."""
    if args.df_exec:
        cache_path = _resolve_path(args.df_exec)
        if not cache_path.exists():
            raise FileNotFoundError(f"df_exec cache not found: {cache_path}")
        logger.info("Loading df_exec from %s", cache_path)
        return _load_df_exec_from_cache(cache_path)

    return _build_df_exec(app_config, args)


def _build_blpx_model(app_config: Any) -> Any:
    """Build the residual-BLPX model from the validated V2 config."""
    from leadlag.models.blpx import ProductionBLPXModel

    return ProductionBLPXModel(app_config.v2.model_dump())


def _select_trade_dates(
    df_exec: pd.DataFrame, start: str | None, end: str | None
) -> pd.DatetimeIndex:
    """Return the df_exec dates within the requested [start, end] range."""
    sim_dates = df_exec.index
    mask = np.ones(len(sim_dates), dtype=bool)
    if start:
        mask &= sim_dates >= pd.Timestamp(start)
    if end:
        mask &= sim_dates <= pd.Timestamp(end)
    return sim_dates[mask]


def _resolve_current_index(df_exec: pd.DataFrame, trade_date: pd.Timestamp) -> int:
    """Return the integer position of ``trade_date`` in ``df_exec``."""
    loc = df_exec.index.get_loc(trade_date)
    if isinstance(loc, int):
        return loc
    if isinstance(loc, slice):
        start = loc.start
        if start is None:
            start = 0
        return int(start)
    # Mask or array-like
    positions = np.where(np.asarray(loc))[0]
    if len(positions) == 0:
        raise KeyError(trade_date)
    return int(positions[0])


def _build_omega_struct(blpx_result: dict[str, Any]) -> np.ndarray:
    """Reconstruct standardized ``Omega_struct`` from BLPX matrix outputs."""
    return _omega_from_blp_res(blpx_result)


def _save_date_diagnostics(
    diag_path: Path,
    trade_date: pd.Timestamp,
    Omega_struct: np.ndarray,
    blpx_result: dict[str, Any],
) -> None:
    """Write a small per-date diagnostics JSON."""
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    eigvals = np.linalg.eigh(Omega_struct)[0]
    diag = np.diag(Omega_struct)
    record = {
        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "trade_date_ymd": trade_date.strftime("%Y%m%d"),
        "shape": list(Omega_struct.shape),
        "min_eigenvalue": float(np.min(eigvals)),
        "max_eigenvalue": float(np.max(eigvals)),
        "negative_eigen_count": int(np.sum(eigvals < 0)),
        "trace": float(np.trace(Omega_struct)),
        "frob_norm": float(np.linalg.norm(Omega_struct, "fro")),
        "min_diag": float(np.min(diag)),
        "max_diag": float(np.max(diag)),
        "cond_num": float(blpx_result.get("cond_num", 0.0)),
        "num_training_samples": int(blpx_result.get("num_training_samples", 0)),
    }
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    config_path = _resolve_path(args.config)
    output_dir = _resolve_path(args.output_dir)
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    logger.info("Loading config from %s", config_path)
    app_config = _load_config(config_path)

    logger.info("Building BLPX model...")
    blpx_model = _build_blpx_model(app_config)

    logger.info("Resolving df_exec...")
    df_exec = _resolve_df_exec(args, app_config)

    logger.info("Preparing common inputs (this may take a while)...")
    inputs = blpx_model._prepare_common_inputs(df_exec, horizon=1)

    trade_dates = _select_trade_dates(df_exec, args.start, args.end)
    if len(trade_dates) == 0:
        logger.warning("No trade dates selected in the requested range.")
        return 0

    logger.info(
        "Computing Omega_struct for %d dates (%s to %s)",
        len(trade_dates),
        trade_dates[0].strftime("%Y-%m-%d"),
        trade_dates[-1].strftime("%Y-%m-%d"),
    )

    # Skip the first rows until the model's correlation window is satisfied.
    # This mirrors the check in tools/research/compute_structured_prediction_covariance.py.
    min_window = getattr(blpx_model, "corr_window", 60)
    diagnostics: list[dict[str, Any]] = []

    for trade_date in trade_dates:
        date_str = trade_date.strftime("%Y%m%d")
        try:
            current_index = _resolve_current_index(df_exec, trade_date)

            if current_index < min_window:
                logger.debug(
                    "Skipping %s: current_index=%d < blp_window=%d",
                    trade_date.strftime("%Y-%m-%d"),
                    current_index,
                    min_window,
                )
                diagnostics.append(
                    {
                        "date": date_str,
                        "status": "skipped",
                        "reason": f"current_index {current_index} < blp_window {min_window}",
                    }
                )
                continue

            blpx_result = blpx_model.compute_blp_signal(
                all_returns=inputs["jp_res_returns_p3"],
                current_index=current_index,
                v0_static=inputs["v0_static"],
                c_full=inputs["c_full_p3"],
                is_residual=True,
                return_matrices=True,
            )

            Omega_struct = _build_omega_struct(blpx_result)

            out_path = matrices_dir / f"omega_struct_{date_str}.npy"
            np.save(out_path, Omega_struct)
            logger.info("Saved %s", out_path)

            if args.save_diagnostics:
                _save_date_diagnostics(
                    output_dir / "diagnostics" / f"omega_struct_{date_str}.json",
                    trade_date,
                    Omega_struct,
                    blpx_result,
                )

            diagnostics.append(
                {
                    "date": date_str,
                    "status": "ok",
                    "shape": list(Omega_struct.shape),
                }
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to compute Omega_struct for %s: %s",
                trade_date.strftime("%Y-%m-%d"),
                exc,
                exc_info=False,
            )
            diagnostics.append(
                {
                    "date": date_str,
                    "status": "error",
                    "error": str(exc),
                }
            )

    summary = {
        "script": "compute_omega_struct",
        "config": str(config_path),
        "output_dir": str(output_dir),
        "start": args.start,
        "end": args.end,
        "df_exec_path": args.df_exec,
        "total_dates": len(trade_dates),
        "computed": len([d for d in diagnostics if d["status"] == "ok"]),
        "skipped": len([d for d in diagnostics if d["status"] == "skipped"]),
        "errors": len([d for d in diagnostics if d["status"] == "error"]),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "diagnostics": diagnostics,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(
        "Done. Summary: %d computed, %d skipped, %d errors. Wrote %s",
        summary["computed"],
        summary["skipped"],
        summary["errors"],
        summary_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
