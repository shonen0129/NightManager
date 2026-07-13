"""Compare sum vs cumprod multi-horizon cumulative return methods.

Generates h=3 and h=5 gap matrices using both methods, then runs
V2 production portfolio backtests with each set to compare Sharpe,
max DD, and turnover.

Usage:
    python3 scripts/experiments/compare_cumulative_method.py
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "production"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.fetcher import download_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
    _BLP_CORR_CACHE,
    _RAW_PCA_CACHE,
    _RESIDUAL_PCA_CACHE,
)
from leadlag.models.production_v2 import generate_v2_production_portfolio
from leadlag.models.sre import compute_jp_target_returns
from compute_gap_adjusted_distribution import compute_cumulative_returns

GAP_DIR = ROOT / "live/pipeline_data/gap_adjusted_distribution/latest"
DIST_DIR = ROOT / "live/pipeline_data/distribution_diagnostics/20260712_230821"
CONFIG_PATH = ROOT / "configs/production/production.yaml"
START_DATE = "2020-01-06"
END_DATE = "2026-07-10"
MH_HORIZONS = [3, 5]


def load_data_and_model():
    """Load df_exec, config, and set up the base model."""
    raw_data = download_data(beta_window=60)
    df_exec = preprocess_data(raw_data, beta_window=60)

    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    inputs = model._prepare_common_inputs(df_exec)

    return df_exec, cfg, model, inputs


def generate_mh_matrices(df_exec, cfg, model, inputs, method: str, out_dir: Path):
    """Generate h=3 and h=5 gap matrices using the specified cumulative method."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrices").mkdir(exist_ok=True)

    # Symlink h=1 matrices and rank reversal files from the existing gap dir
    src_matrices = GAP_DIR / "matrices"
    for f in src_matrices.iterdir():
        if f.name.startswith("mu_gap_") and "h" not in f.name.split("_")[2]:
            dest = out_dir / "matrices" / f.name
            if not dest.exists():
                dest.symlink_to(f.resolve())
        elif f.name.startswith("omega_gap_") and "h" not in f.name.split("_")[2]:
            dest = out_dir / "matrices" / f.name
            if not dest.exists():
                dest.symlink_to(f.resolve())
        elif f.name.startswith("rank_reversal_"):
            dest = out_dir / "matrices" / f.name
            if not dest.exists():
                dest.symlink_to(f.resolve())

    # Set up multi-horizon models
    mh_models = {}
    mh_inputs = {}
    for h in MH_HORIZONS:
        logger.info(f"[{method}] Setting up multi-horizon model for h={h}...")
        df_exec_h = compute_cumulative_returns(df_exec, h, method=method)
        _BLP_CORR_CACHE.clear()
        _RAW_PCA_CACHE.clear()
        _RESIDUAL_PCA_CACHE.clear()
        model_h = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        inputs_h = model_h._prepare_common_inputs(df_exec_h)
        mh_models[h] = model_h
        mh_inputs[h] = inputs_h
        logger.info(f"  h={h} model ready (corr_window={model_h.corr_window})")

    # Common parameters from base model
    c = model.gap_open_coef
    b = model.topix_beta_coef

    # Process each date
    sim_dates = df_exec.index[(df_exec.index >= START_DATE) & (df_exec.index <= END_DATE)]
    n_processed = 0
    n_skipped = 0

    for dt in sim_dates:
        i = df_exec.index.get_indexer([dt])[0]
        if i < model.corr_window:
            n_skipped += 1
            continue

        dt_str = dt.strftime("%Y%m%d")
        date_str = dt.strftime("%Y-%m-%d")

        # Load Omega_struct from Step 1
        omega_struct_file = DIST_DIR / "matrices" / f"omega_struct_{dt_str}.npy"
        if not omega_struct_file.exists():
            n_skipped += 1
            continue

        Omega_struct = np.load(omega_struct_file)
        if not np.isfinite(Omega_struct).all():
            n_skipped += 1
            continue

        for h in MH_HORIZONS:
            try:
                model_h = mh_models[h]
                inputs_h = mh_inputs[h]
                gap_h = inputs_h["jp_gap"]
                beta_h = inputs_h["jp_beta"]
                topix_night_h = inputs_h["topix_night"]
                jp_res_h = inputs_h["jp_res_returns_p3"]
                c_full_h = inputs_h["c_full_p3"]
                v0_h = inputs_h["v0_static"]

                gap_override_h = np.nan_to_num(gap_h[i], nan=0.0) if gap_h is not None else np.zeros(model_h.n_j)
                betas_t_h = np.asarray(beta_h[i], dtype=float) if beta_h is not None else np.zeros(model_h.n_j)
                topix_night_t_h = float(topix_night_h[i]) if topix_night_h is not None else 0.0

                res_h = model_h.compute_blp_signal(
                    jp_res_h, i,
                    gap_override=gap_override_h,
                    betas_t=betas_t_h,
                    topix_night_t=topix_night_t_h,
                    rolling_std=None,
                    v0_static=v0_h,
                    c_full=c_full_h,
                    is_residual=True,
                    return_matrices=True,
                )

                z_hat_h = res_h["z_hat_j_t1"]
                sigma_Y_denorm_h = res_h["sigma_Y_denorm"]
                mu_Y_h = res_h["mu_Y"]

                if model_h.vol_adjusted_target:
                    mu_raw_h = z_hat_h * sigma_Y_denorm_h
                else:
                    mu_raw_h = mu_Y_h + res_h["sigma_Y"] * z_hat_h

                Omega_raw_h = np.diag(sigma_Y_denorm_h) @ Omega_struct @ np.diag(sigma_Y_denorm_h)

                gap_syst_h = betas_t_h * topix_night_t_h
                gap_idio_h = gap_override_h - gap_syst_h
                gap_filt_h = c * gap_idio_h + (c - b) * gap_syst_h
                denom_h = np.maximum(1.0 + gap_filt_h, 0.1)
                D_gap_h = np.diag(1.0 / denom_h)
                mu_gap_h = (1.0 + mu_raw_h) / denom_h - 1.0
                Omega_gap_h = D_gap_h @ Omega_raw_h @ D_gap_h
                Omega_gap_h = 0.5 * (Omega_gap_h + Omega_gap_h.T)

                np.save(out_dir / "matrices" / f"mu_gap_h{h}_{dt_str}.npy", mu_gap_h)
                np.save(out_dir / "matrices" / f"omega_gap_h{h}_{dt_str}.npy", Omega_gap_h)
            except Exception as e:
                logger.warning(f"[{method}] h={h} failed on {date_str}: {e}")

        n_processed += 1
        if n_processed % 200 == 0:
            logger.info(f"[{method}] Processed {n_processed} dates (skipped {n_skipped})")

    logger.info(f"[{method}] Done: {n_processed} processed, {n_skipped} skipped")
    return out_dir


def run_v2_backtest(df_exec, cfg, gap_dir: Path, y_jp_target: np.ndarray, label: str) -> dict:
    """Run V2 production portfolio backtest with the given gap directory."""
    logger.info(f"[{label}] Starting V2 backtest...")

    sim_dates = df_exec.index[(df_exec.index >= START_DATE) & (df_exec.index <= END_DATE)]

    daily_returns = []
    daily_gross = []
    daily_turnover = []
    w_prev = np.zeros(len(JP_TICKERS))
    n_fallback = 0
    n_success = 0

    for dt in sim_dates:
        date_str = dt.strftime("%Y-%m-%d")
        i = df_exec.index.get_indexer([dt])[0]

        try:
            result = generate_v2_production_portfolio(date_str, gap_dir, cfg)
        except Exception as e:
            logger.warning(f"[{label}] Failed on {date_str}: {e}")
            daily_returns.append(0.0)
            daily_gross.append(0.0)
            daily_turnover.append(0.0)
            continue

        w = result["w_final"]
        if result["fallback"]["gap_data_missing"]:
            n_fallback += 1
            daily_returns.append(0.0)
            daily_gross.append(0.0)
            daily_turnover.append(0.0)
            w_prev = w.copy()
            continue

        n_success += 1
        gross = float(np.sum(np.abs(w)))
        net = float(np.sum(w))
        turnover = float(np.sum(np.abs(w - w_prev)))
        ret = float(np.dot(w, y_jp_target[i]))

        daily_returns.append(ret)
        daily_gross.append(gross)
        daily_turnover.append(turnover)
        w_prev = w.copy()

    returns = np.array(daily_returns)
    gross_arr = np.array(daily_gross)
    turnover_arr = np.array(daily_turnover)

    # Compute metrics
    n_days = len(returns)
    total_return = float(np.sum(returns))
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0

    # Max drawdown
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd = float(np.min(drawdowns))

    avg_turnover = float(np.mean(turnover_arr))
    avg_gross = float(np.mean(gross_arr))

    logger.info(
        f"[{label}] Done: {n_success} success, {n_fallback} fallback, "
        f"Sharpe={sharpe:.4f}, MaxDD={max_dd:.4f}, Turnover={avg_turnover:.4f}"
    )

    return {
        "label": label,
        "n_days": n_days,
        "n_success": n_success,
        "n_fallback": n_fallback,
        "total_return": total_return,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_turnover": avg_turnover,
        "avg_gross": avg_gross,
        "daily_returns": returns,
        "daily_turnover": turnover_arr,
    }


def main():
    logger.info("Loading data and setting up base model...")
    df_exec, cfg, model, inputs = load_data_and_model()
    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    # Create temp directories for both methods
    tmp_base = Path(tempfile.mkdtemp(prefix="cumulative_compare_"))
    sum_dir = tmp_base / "sum"
    cumprod_dir = tmp_base / "cumprod"

    try:
        # Generate h>1 matrices for both methods
        generate_mh_matrices(df_exec, cfg, model, inputs, "sum", sum_dir)
        generate_mh_matrices(df_exec, cfg, model, inputs, "cumprod", cumprod_dir)

        # Run V2 backtests with both
        results_sum = run_v2_backtest(df_exec, cfg, sum_dir, y_jp_target, "sum")
        results_cumprod = run_v2_backtest(df_exec, cfg, cumprod_dir, y_jp_target, "cumprod")

        # Compare
        print("\n" + "=" * 80)
        print("COMPARISON: sum vs cumprod multi-horizon cumulative returns")
        print("=" * 80)
        print(f"{'Metric':<25} {'sum':>15} {'cumprod':>15} {'diff':>15}")
        print("-" * 80)
        for key in ["n_success", "n_fallback", "total_return", "mean_return", "std_return", "sharpe", "max_dd", "avg_turnover", "avg_gross"]:
            v_sum = results_sum[key]
            v_cum = results_cumprod[key]
            if isinstance(v_sum, float):
                print(f"{key:<25} {v_sum:>15.6f} {v_cum:>15.6f} {v_cum - v_sum:>15.6f}")
            else:
                print(f"{key:<25} {v_sum:>15} {v_cum:>15} {v_cum - v_sum:>15}")

        # Daily return correlation
        r_sum = results_sum["daily_returns"]
        r_cum = results_cumprod["daily_returns"]
        corr = float(np.corrcoef(r_sum, r_cum)[0, 1])
        print(f"\n{'daily_return_corr':<25} {corr:>15.6f}")

        # Max absolute difference in daily returns
        max_diff = float(np.max(np.abs(r_cum - r_sum)))
        print(f"{'max_abs_daily_ret_diff':<25} {max_diff:>15.8f}")

        print("=" * 80)

    finally:
        logger.info(f"Cleaning up temp directory: {tmp_base}")
        shutil.rmtree(tmp_base, ignore_errors=True)


if __name__ == "__main__":
    main()
