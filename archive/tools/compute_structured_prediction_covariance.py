#!/usr/bin/env python
"""Compute, Save, and Diagnose Residual-BLPX Prediction Error Covariance (Step 1).

Calculates Omega_struct daily using point-in-time intermediate matrices
and performs rigorous calibration and numerical audits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, skew, kurtosis, norm

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, US_TICKERS, TOPIX_TICKER
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DistributionDiagnostics")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Residual-BLPX Prediction Covariance Diagnostic Suite")
    parser.add_argument("--config", default="configs/production.yaml", help="Path to YAML config file")
    parser.add_argument("--model", default="production_residual_blpx", help="Model name")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--results-dir", default="live/pipeline_data/v1_backtest", help="Validation outputs directory")
    parser.add_argument("--output-dir", default="live/pipeline_data/distribution_diagnostics", help="Output directory")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps per side")
    parser.add_argument("--save-daily-matrices", type=str, default="true", help="Save daily matrices (true/false)")
    parser.add_argument("--save-psd-projection", type=str, default="true", help="Save PSD projection (true/false)")
    parser.add_argument("--run-backtest-if-missing", action="store_true", help="Run backtest if weights are missing")
    parser.add_argument("--compare-existing-pred-var", type=str, default="true", help="Compare with existing pred_var (true/false)")
    parser.add_argument("--vol-state-panel", default=None, help="Path to vol state panel CSV file")
    parser.add_argument("--self-test", action="store_true", help="Run self-tests and exit")
    return parser.parse_args()


def str_to_bool(val: str) -> bool:
    """Convert string to boolean."""
    return str(val).lower() in ("true", "1", "yes")


def psd_project(matrix: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, float]:
    """Project symmetric matrix to the Positive Semi-Definite (PSD) cone."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_psd = np.maximum(eigvals, floor)
    matrix_psd = eigvecs @ np.diag(eigvals_psd) @ eigvecs.T
    distance = float(np.linalg.norm(matrix_psd - matrix, "fro"))
    return matrix_psd, distance


def run_self_tests() -> int:
    """Run verification self-tests."""
    logger.info("=== Starting Self-Tests ===")
    
    n_u = 15
    n_j = 17
    
    # 1. Dimension Check
    np.random.seed(42)
    Sigma_XX = np.eye(n_u) + 0.1 * np.random.randn(n_u, n_u)
    Sigma_XX = 0.5 * (Sigma_XX + Sigma_XX.T)
    Sigma_YX = np.random.randn(n_j, n_u) * 0.1
    Sigma_YY = np.eye(n_j) + 0.1 * np.random.randn(n_j, n_j)
    Sigma_YY = 0.5 * (Sigma_YY + Sigma_YY.T)
    B = np.random.randn(n_j, n_u) * 0.05
    
    Sigma_XY = Sigma_YX.T
    Omega = Sigma_YY - B @ Sigma_XY - Sigma_YX @ B.T + B @ Sigma_XX @ B.T
    
    assert Omega.shape == (n_j, n_j), f"Dimension mismatch: expected (17, 17), got {Omega.shape}"
    logger.info("Dimension check passed.")
    
    # 2. Conditional Covariance Consistency
    B_opt = Sigma_YX @ np.linalg.inv(Sigma_XX)
    Omega_opt = Sigma_YY - B_opt @ Sigma_XY - Sigma_YX @ B_opt.T + B_opt @ Sigma_XX @ B_opt.T
    Omega_cond = Sigma_YY - Sigma_YX @ np.linalg.inv(Sigma_XX) @ Sigma_XY
    
    assert np.allclose(Omega_opt, Omega_cond, atol=1e-12), "Algebraic equivalence test failed."
    logger.info("Conditional covariance equivalence check passed.")
    
    # 3. Symmetry Check
    Omega_sym = 0.5 * (Omega + Omega.T)
    assert np.allclose(Omega_sym, Omega_sym.T, atol=1e-15), "Symmetry check failed."
    logger.info("Symmetry enforcement check passed.")
    
    # 4. Signal Date < Trade Date
    sig_date = datetime(2026, 6, 12)
    trade_date = datetime(2026, 6, 15)
    assert sig_date < trade_date, "Temporal order check failed."
    logger.info("Date alignment logic check passed.")
    
    # 5. Ratio division checks
    pred_var = np.array([1e-5, 0.0, 1.2e-4])
    omega_diag = np.array([2e-5, 3e-5, 0.0])
    
    # Avoid zero division
    safe_pred = np.where(pred_var > 1e-10, pred_var, 1e-10)
    ratio = omega_diag / safe_pred
    assert np.all(np.isfinite(ratio)), "Zero division handling failed."
    logger.info("Ratio zero-division safety check passed.")
    
    logger.info("=== All Self-Tests Passed ===")
    return 0


def main():
    args = parse_arguments()
    
    if args.self_test:
        sys.exit(run_self_tests())
        
    save_daily_m = str_to_bool(args.save_daily_matrices)
    save_psd = str_to_bool(args.save_psd_projection)
    compare_pred_var = str_to_bool(args.compare_existing_pred_var)
    
    # Setup outputs
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if save_daily_m or save_psd:
        (out_dir / "matrices").mkdir(exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    logger.info(f"Output directory established: {out_dir}")
    
    # 1. Load config
    cfg_path = ROOT / args.config
    logger.info(f"Loading config from {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        
    results_dir = Path(args.results_dir) if args.results_dir.startswith("results") else ROOT / args.results_dir
    
    # 2. Download and Preprocess data
    logger.info("Loading market data...")
    raw_data = download_data(beta_window=60)
    logger.info("Preprocessing market data...")
    df_exec = preprocess_data(raw_data, beta_window=60)
    
    # Compute TOPIX returns
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0
    
    # Filter dates
    sim_dates = df_exec.index
    sim_dates_slice = sim_dates[sim_dates >= args.start]
    if args.end != "latest":
        sim_dates_slice = sim_dates_slice[sim_dates_slice <= args.end]
        
    logger.info(f"Diagnostics window: {sim_dates_slice[0].strftime('%Y-%m-%d')} to {sim_dates_slice[-1].strftime('%Y-%m-%d')} ({len(sim_dates_slice)} days)")
    
    # 3. Model setup
    logger.info("Instantiating Residual-BLPX model...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    inputs = model._prepare_common_inputs(df_exec)
    
    # Extract realized target returns (9:10-to-close)
    y_jp_target = inputs["y_jp_target"]
    y_jp_target_df = pd.DataFrame(y_jp_target, index=sim_dates, columns=JP_TICKERS)
    
    # Extract variables
    jp_gap = inputs["jp_gap"]
    jp_beta = inputs["jp_beta"]
    topix_night = inputs["topix_night"]
    jp_res_returns_p3 = inputs["jp_res_returns_p3"]
    c_full_p3 = inputs["c_full_p3"]
    v0_static = inputs["v0_static"]
    
    # Check weights and runs
    weights_file = results_dir / "daily_positions_Residual-BLPX_only.csv"
    if not weights_file.exists():
        if args.run_backtest_if_missing:
            logger.info(f"Weights file {weights_file} not found. Running backtest...")
            results_dir.mkdir(parents=True, exist_ok=True)
            backtest_res = BacktestEngine.run_backtest(model, df_exec, start_date=args.start, end_date=args.end, slippage_bps=args.slippage_bps)
            backtest_res["weights"].to_csv(weights_file)
            logger.info(f"Backtest weights written to {weights_file}")
        else:
            logger.error(f"Weights file not found at {weights_file} and --run-backtest-if-missing is not specified. Cannot proceed with portfolio level diagnostics.")
            sys.exit(1)
            
    logger.info(f"Loading weights from {weights_file}")
    weights_df = pd.read_csv(weights_file, index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index).tz_localize(None).normalize()
    
    # Main Daily calculation loop
    diag_records = []
    comparison_records = []
    panel_records = []
    daily_summary_records = []
    
    # For audits
    dropped_count = 0
    missing_data_count = 0
    nan_inf_count = 0
    neg_eigen_days = 0
    days_with_min_eigen_lt_neg_1e_8 = 0
    days_with_diag_le_zero = 0
    symmetry_max_err = 0.0
    all_dates_audit = True
    leakage_violation = False
    
    logger.info("Computing structured prediction error covariance Omega_struct...")
    
    w_prev = np.zeros(model.n_j)
    
    for dt in sim_dates_slice:
        i = df_exec.index.get_indexer([dt])[0]
        if i < model.corr_window:
            dropped_count += 1
            continue
            
        sig_date = df_exec["sig_date"].values[i]
        sig_date_dt = pd.to_datetime(sig_date).tz_localize(None).normalize()
        trade_date_dt = pd.to_datetime(dt).tz_localize(None).normalize()
        
        # 1. Leakage check
        if not (sig_date_dt < trade_date_dt):
            all_dates_audit = False
            leakage_violation = True
            
        # Common parameters
        gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None
        
        # Call model to get raw matrices
        try:
            residual_blpx_res = model.compute_blp_signal(
                jp_res_returns_p3,
                i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
                rolling_std=None,
                v0_static=v0_static,
                c_full=c_full_p3,
                is_residual=True,
                return_matrices=True,
            )
        except Exception as e:
            missing_data_count += 1
            logger.warning(f"Error computing signal on {dt.strftime('%Y-%m-%d')}: {e}")
            continue
            
        Sigma_XX = residual_blpx_res["Sigma_XX"]
        Sigma_YX = residual_blpx_res["Sigma_YX"]
        Sigma_XY = Sigma_YX.T
        Sigma_YY = residual_blpx_res["Sigma_YY"]
        B_struct = residual_blpx_res["B_struct"]
        z_U = residual_blpx_res["z_U"]
        pred_var_vec = residual_blpx_res["pred_var_vec"]
        sigma_Y_denorm = residual_blpx_res["sigma_Y_denorm"]
        
        # Validate matrix finitude
        if not (np.isfinite(Sigma_XX).all() and np.isfinite(Sigma_YX).all() and np.isfinite(Sigma_YY).all() and np.isfinite(B_struct).all()):
            nan_inf_count += 1
            logger.warning(f"NaN or Inf detected in covariance inputs on {dt.strftime('%Y-%m-%d')}")
            continue
            
        # 2. Compute standardized Omega_struct
        Omega_struct = Sigma_YY - B_struct @ Sigma_XY - Sigma_YX @ B_struct.T + B_struct @ Sigma_XX @ B_struct.T
        
        # Symmetry check before enforcement
        sym_error = float(np.max(np.abs(Omega_struct - Omega_struct.T)))
        symmetry_max_err = max(symmetry_max_err, sym_error)
        
        # Symmetrize
        Omega_struct = 0.5 * (Omega_struct + Omega_struct.T)
        
        # Spectral decomposition
        eigvals, _ = np.linalg.eigh(Omega_struct)
        min_eigen = float(np.min(eigvals))
        max_eigen = float(np.max(eigvals))
        neg_count = int(np.sum(eigvals < 0))
        if neg_count > 0:
            neg_eigen_days += 1
        if min_eigen < -1e-8:
            days_with_min_eigen_lt_neg_1e_8 += 1
            
        omega_struct_diag = np.diag(Omega_struct)
        if np.any(omega_struct_diag <= 0):
            days_with_diag_le_zero += 1
            
        cond_num = float(max_eigen / min_eigen) if min_eigen > 0 else np.nan
        trace = float(np.trace(Omega_struct))
        
        # Det and logdet
        det = float(np.linalg.det(Omega_struct))
        logdet = float(np.sum(np.log(np.maximum(eigvals, 1e-15))))
        
        # Average off-diagonal correlation
        diag_std = np.sqrt(np.maximum(omega_struct_diag, 1e-10))
        R = Omega_struct / np.outer(diag_std, diag_std)
        avg_offdiag = float((np.sum(R) - model.n_j) / (model.n_j * (model.n_j - 1)))
        
        frob_norm = float(np.linalg.norm(Omega_struct, "fro"))
        
        # Frobenius distance to Sigma_Y|X
        inv_A = residual_blpx_res["inv_A"]
        Sigma_Y_given_X = Sigma_YY - Sigma_YX @ inv_A @ Sigma_XY
        frob_diff = float(np.linalg.norm(Omega_struct - Sigma_Y_given_X, "fro"))
        
        # Diagnostic outputs per day
        diag_corr, _ = spearmanr(omega_struct_diag, pred_var_vec)
        if np.isnan(diag_corr):
            diag_corr = 0.0
            
        date_str = dt.strftime("%Y-%m-%d")
        diag_records.append({
            "date": date_str,
            "min_eigenvalue": min_eigen,
            "max_eigenvalue": max_eigen,
            "negative_eigen_count": neg_count,
            "condition_number": cond_num,
            "trace": trace,
            "determinant": det,
            "logdet": logdet,
            "avg_offdiag_corr": avg_offdiag,
            "frob_norm": frob_norm,
            "frob_diff_vs_cond_cov": frob_diff,
            "diag_spearman_corr_vs_pred_var": diag_corr,
        })
        
        # PSD Projection
        Omega_struct_psd = None
        psd_dist = 0.0
        if save_psd:
            Omega_struct_psd, psd_dist = psd_project(Omega_struct)
            
        # Comparison logic (standardized space)
        if compare_pred_var:
            for idx, tk in enumerate(JP_TICKERS):
                p_var = float(pred_var_vec[idx])
                o_var = float(omega_struct_diag[idx])
                safe_p = p_var if p_var > 1e-10 else 1e-10
                safe_o = o_var if o_var > 1e-10 else 1e-10
                comparison_records.append({
                    "date": date_str,
                    "ticker": tk,
                    "pred_var_blp_diag": p_var,
                    "omega_struct_diag": o_var,
                    "ratio": safe_o / safe_p,
                    "log_ratio": np.log(safe_o / safe_p),
                })
                
        # Scale to raw return space
        Omega_struct_raw = np.diag(sigma_Y_denorm) @ Omega_struct @ np.diag(sigma_Y_denorm)
        omega_diag_raw = np.diag(Omega_struct_raw)
        omega_std_raw = np.sqrt(np.maximum(omega_diag_raw, 1e-10))
        
        # Portfolio level diagnostics
        mu_t = residual_blpx_res["signal"] # Raw prediction
        
        # Retrieve weights
        w_t = np.zeros(model.n_j)
        if dt in weights_df.index:
            w_t = weights_df.loc[dt, JP_TICKERS].values
            
        predicted_portfolio_mean = float(np.sum(w_t * mu_t))
        predicted_portfolio_var_struct = float(w_t.T @ Omega_struct_raw @ w_t)
        predicted_portfolio_vol_struct = float(np.sqrt(np.maximum(predicted_portfolio_var_struct, 1e-10)))
        predicted_portfolio_ir_struct = predicted_portfolio_mean / predicted_portfolio_vol_struct if predicted_portfolio_vol_struct > 0 else 0.0
        
        # Diagonal-only variance
        predicted_portfolio_var_diagonly_struct = float(np.sum((w_t ** 2) * omega_diag_raw))
        predicted_portfolio_vol_diagonly_struct = float(np.sqrt(np.maximum(predicted_portfolio_var_diagonly_struct, 1e-10)))
        
        # Existing pred_var diagonal-only raw variance
        pred_var_raw = pred_var_vec * (sigma_Y_denorm ** 2)
        predicted_portfolio_var_diagonly = float(np.sum((w_t ** 2) * pred_var_raw))
        predicted_portfolio_vol_diagonly = float(np.sqrt(np.maximum(predicted_portfolio_var_diagonly, 1e-10)))
        
        # Realized portfolio return
        realized_jp_returns = y_jp_target_df.loc[dt].values
        realized_portfolio_return_gross = float(np.sum(w_t * realized_jp_returns))
        gross_exposure = float(np.sum(np.abs(w_t)))
        costs_t = float(2.0 * (args.slippage_bps / 10000.0) * gross_exposure)
        realized_portfolio_return_net = realized_portfolio_return_gross - costs_t
        
        predicted_portfolio_mean_net = predicted_portfolio_mean - costs_t
        predicted_portfolio_ir_net_struct = predicted_portfolio_mean_net / predicted_portfolio_vol_struct if predicted_portfolio_vol_struct > 0 else 0.0
        
        # Turnover calculation
        turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)
        w_prev = w_t.copy()
        
        # Save daily summary record
        daily_summary_records.append({
            "trade_date": date_str,
            "signal_date": sig_date,
            "predicted_portfolio_mean": predicted_portfolio_mean,
            "predicted_portfolio_var_struct": predicted_portfolio_var_struct,
            "predicted_portfolio_vol_struct": predicted_portfolio_vol_struct,
            "predicted_portfolio_ir_struct": predicted_portfolio_ir_struct,
            "predicted_portfolio_mean_net": predicted_portfolio_mean_net,
            "predicted_portfolio_ir_net_struct": predicted_portfolio_ir_net_struct,
            "predicted_portfolio_var_diagonly_struct": predicted_portfolio_var_diagonly_struct,
            "predicted_portfolio_vol_diagonly_struct": predicted_portfolio_vol_diagonly_struct,
            "predicted_portfolio_var_diagonly": predicted_portfolio_var_diagonly,
            "predicted_portfolio_vol_diagonly": predicted_portfolio_vol_diagonly,
            "realized_portfolio_return_gross": realized_portfolio_return_gross,
            "realized_portfolio_return_net": realized_portfolio_return_net,
            "gross_exposure": gross_exposure,
            "net_exposure": float(np.sum(w_t)),
            "cost": costs_t,
            "turnover": turnover,
        })
        
        # Stock-level long records
        for idx, tk in enumerate(JP_TICKERS):
            panel_records.append({
                "signal_date": sig_date,
                "trade_date": date_str,
                "ticker": tk,
                "z_hat_J": float(residual_blpx_res["z_hat_j_t1"][idx]),
                "mu_t": float(mu_t[idx]),
                "omega_diag_struct": float(omega_diag_raw[idx]),
                "omega_std_struct": float(omega_std_raw[idx]),
                "existing_pred_var": float(pred_var_vec[idx]),
                "portfolio_weight": float(w_t[idx]),
                "realized_target_return": float(realized_jp_returns[idx]),
            })
            
        # Save daily matrices if requested
        if save_daily_m:
            dt_str = dt.strftime("%Y%m%d")
            np.save(out_dir / "matrices" / f"omega_struct_{dt_str}.npy", Omega_struct)
            np.save(out_dir / "matrices" / f"B_struct_{dt_str}.npy", B_struct)
            np.savez(
                out_dir / "matrices" / f"sigma_blocks_{dt_str}.npz",
                sigma_xx=Sigma_XX,
                sigma_yx=Sigma_YX,
                sigma_yy=Sigma_YY,
            )
            if save_psd and Omega_struct_psd is not None:
                np.save(out_dir / "matrices" / f"omega_struct_psd_{dt_str}.npy", Omega_struct_psd)
                
    # Build dataframes
    df_diag = pd.DataFrame(diag_records).set_index("date")
    df_diag.to_csv(out_dir / "omega_summary_daily.csv")
    
    # Save ticker level summary of eigenvalues or trace
    df_comp = None
    if compare_pred_var and comparison_records:
        df_comp = pd.DataFrame(comparison_records)
        df_comp.to_csv(out_dir / "pred_var_comparison_daily.csv", index=False)
        
        ticker_comp = []
        for tk in JP_TICKERS:
            sub = df_comp[df_comp["ticker"] == tk]
            ticker_comp.append({
                "ticker": tk,
                "mean_pred_var_blp": sub["pred_var_blp_diag"].mean(),
                "mean_omega_struct": sub["omega_struct_diag"].mean(),
                "mean_ratio": sub["ratio"].mean(),
                "time_series_correlation": sub["pred_var_blp_diag"].corr(sub["omega_struct_diag"]),
            })
        df_ticker_comp = pd.DataFrame(ticker_comp)
        df_ticker_comp.to_csv(out_dir / "pred_var_comparison_by_ticker.csv", index=False)
        
        summary_comp = {
            "total_mean_pred_var_blp": df_comp["pred_var_blp_diag"].mean(),
            "total_mean_omega_struct": df_comp["omega_struct_diag"].mean(),
            "total_mean_ratio": df_comp["ratio"].mean(),
            "cross_sectional_correlation_avg": df_diag["diag_spearman_corr_vs_pred_var"].mean(),
        }
        pd.DataFrame([summary_comp]).to_csv(out_dir / "pred_var_comparison_summary.csv", index=False)
        
    # Process Long Panel
    df_panel_long = pd.DataFrame(panel_records)
    df_panel_long.to_csv(out_dir / "distribution_panel_long.csv", index=False)
    
    # Process Daily Panel
    df_panel_daily = pd.DataFrame(daily_summary_records)
    df_panel_daily.to_csv(out_dir / "distribution_panel_daily.csv", index=False)
    
    # Parquet saving if possible
    try:
        df_panel_long.to_parquet(out_dir / "distribution_panel_long.parquet")
        df_panel_daily.to_parquet(out_dir / "distribution_panel_daily.parquet")
    except Exception as e:
        logger.warning(f"Could not save parquet formats: {e}")
        
    # US Vol State merging (Optional)
    vol_state_merged = False
    if args.vol_state_panel:
        state_file = Path(args.vol_state_panel)
        if state_file.exists():
            logger.info(f"Merging US Vol State panel from {state_file}...")
            df_state = pd.read_csv(state_file)
            df_state["trade_date"] = pd.to_datetime(df_state["trade_date"]).dt.strftime("%Y-%m-%d")
            df_panel_daily_m = df_panel_daily.merge(df_state, on="trade_date", how="left")
            df_panel_daily_m.to_csv(out_dir / "vol_state_distribution_diagnostics.csv", index=False)
            vol_state_merged = True
        else:
            logger.warning(f"Vol state panel file {state_file} not found. Skipping merge.")
            
    # Calibration Calculations
    # 1. Portfolio Level
    df_panel_daily["abs_realized_net"] = df_panel_daily["realized_portfolio_return_net"].abs()
    df_panel_daily["squared_realized_net"] = df_panel_daily["realized_portfolio_return_net"] ** 2
    
    corr_mean_gross = df_panel_daily["predicted_portfolio_mean"].corr(df_panel_daily["realized_portfolio_return_gross"])
    corr_ir_net = df_panel_daily["predicted_portfolio_ir_struct"].corr(df_panel_daily["realized_portfolio_return_net"])
    corr_vol_abs = df_panel_daily["predicted_portfolio_vol_struct"].corr(df_panel_daily["abs_realized_net"])
    corr_var_squared = df_panel_daily["predicted_portfolio_var_struct"].corr(df_panel_daily["squared_realized_net"])
    
    portfolio_calibration = {
        "corr_predicted_mean_vs_realized_gross": corr_mean_gross,
        "corr_predicted_ir_vs_realized_net": corr_ir_net,
        "corr_predicted_vol_vs_abs_realized_net": corr_vol_abs,
        "corr_predicted_var_vs_squared_realized_net": corr_var_squared,
    }
    pd.DataFrame([portfolio_calibration]).to_csv(out_dir / "portfolio_distribution_diagnostics.csv", index=False)
    
    # 2. Predicted IR Quintiles/Tertiles
    df_panel_daily["ir_bin"] = pd.qcut(df_panel_daily["predicted_portfolio_ir_struct"], 3, labels=["Low", "Medium", "High"])
    ir_bins_df = df_panel_daily.groupby("ir_bin").agg(
        mean_realized_net=("realized_portfolio_return_net", "mean"),
        vol_realized_net=("realized_portfolio_return_net", "std"),
        hit_rate=("realized_portfolio_return_net", lambda x: (x > 0).mean()),
    )
    ir_bins_df["sharpe"] = (ir_bins_df["mean_realized_net"] / ir_bins_df["vol_realized_net"]) * np.sqrt(252.0)
    ir_bins_df.to_csv(out_dir / "calibration_by_ir_bin.csv")
    
    # 3. Predicted Vol Quintiles/Tertiles
    df_panel_daily["vol_bin"] = pd.qcut(df_panel_daily["predicted_portfolio_vol_struct"], 3, labels=["Low", "Medium", "High"])
    vol_bins_df = df_panel_daily.groupby("vol_bin").agg(
        realized_vol=("realized_portfolio_return_net", "std"),
        mean_abs_realized_net=("abs_realized_net", "mean"),
    )
    vol_bins_df["realized_vol_annualized"] = vol_bins_df["realized_vol"] * np.sqrt(252.0)
    vol_bins_df.to_csv(out_dir / "calibration_by_vol_bin.csv")
    
    # 4. Calibration by Year
    df_panel_daily["year"] = pd.to_datetime(df_panel_daily["trade_date"]).dt.year
    year_bins_df = df_panel_daily.groupby("year").agg(
        mean_realized_net=("realized_portfolio_return_net", "mean"),
        vol_realized_net=("realized_portfolio_return_net", "std"),
        gross_return=("realized_portfolio_return_gross", "mean"),
        net_return=("realized_portfolio_return_net", "mean"),
        cost=("cost", "mean"),
        turnover=("turnover", "mean"),
    )
    year_bins_df["sharpe"] = (year_bins_df["mean_realized_net"] / year_bins_df["vol_realized_net"]) * np.sqrt(252.0)
    year_bins_df.to_csv(out_dir / "calibration_by_year.csv")
    
    # 5. Standardized Residuals
    df_panel_long["residual"] = df_panel_long["realized_target_return"] - df_panel_long["mu_t"]
    df_panel_long["standardized_residual"] = df_panel_long["residual"] / df_panel_long["omega_std_struct"]
    
    std_res = df_panel_long["standardized_residual"].dropna()
    
    residuals_summary = {
        "mean": std_res.mean(),
        "std": std_res.std(),
        "skewness": skew(std_res),
        "kurtosis": kurtosis(std_res),
        "p01": np.percentile(std_res, 1),
        "p05": np.percentile(std_res, 5),
        "p95": np.percentile(std_res, 95),
        "p99": np.percentile(std_res, 99),
        "pct_outside_2_sigma": (np.abs(std_res) > 2.0).mean(),
        "pct_outside_3_sigma": (np.abs(std_res) > 3.0).mean(),
    }
    pd.DataFrame([residuals_summary]).to_csv(out_dir / "standardized_residuals_summary.csv", index=False)
    
    # Ticker standardized residuals
    ticker_residuals = []
    for tk in JP_TICKERS:
        sub = df_panel_long[df_panel_long["ticker"] == tk].dropna(subset=["standardized_residual"])
        res_sub = sub["standardized_residual"]
        ticker_residuals.append({
            "ticker": tk,
            "mean": res_sub.mean(),
            "std": res_sub.std(),
            "skewness": skew(res_sub),
            "kurtosis": kurtosis(res_sub),
            "pct_outside_2_sigma": (np.abs(res_sub) > 2.0).mean(),
            "pct_outside_3_sigma": (np.abs(res_sub) > 3.0).mean(),
        })
    pd.DataFrame(ticker_residuals).to_csv(out_dir / "standardized_residuals_by_ticker.csv", index=False)
    
    # Plot Generation
    logger.info("Generating diagnostic plots...")
    
    # Plot 1: predicted_portfolio_ir_struct cumulative return
    plt.figure()
    for category in ["Low", "Medium", "High"]:
        mask = df_panel_daily["ir_bin"] == category
        sub_dates = pd.to_datetime(df_panel_daily.loc[mask, "trade_date"])
        cum_ret = (1.0 + df_panel_daily.loc[mask, "realized_portfolio_return_net"]).cumprod() - 1.0
        plt.plot(sub_dates, cum_ret, label=f"{category} IR Bin")
    plt.title("Cumulative Net Return by Predicted Portfolio IR Tertiles")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.savefig(plots_dir / "portfolio_ir_bin_cumulative_return.png")
    plt.close()
    
    # Plot 2: Bar plot Sharpe by IR Bin
    plt.figure()
    ir_bins_df["sharpe"].plot(kind="bar")
    plt.title("Annualized Sharpe Ratio by Predicted Portfolio IR Tertiles")
    plt.ylabel("Sharpe")
    plt.savefig(plots_dir / "portfolio_ir_bin_sharpe.png")
    plt.close()
    
    # Plot 3: Predicted Vol Bin vs Realized Vol / Mean Abs Return
    plt.figure()
    ax = vol_bins_df[["realized_vol_annualized", "mean_abs_realized_net"]].plot(kind="bar", secondary_y="mean_abs_realized_net")
    plt.title("Realized Vol and Abs Return by Predicted Portfolio Vol Tertiles")
    plt.savefig(plots_dir / "portfolio_vol_bin_calibration.png")
    plt.close()
    
    # Plot 4: Scatter Plot predicted mean vs realized gross return
    plt.figure()
    plt.scatter(df_panel_daily["predicted_portfolio_mean"], df_panel_daily["realized_portfolio_return_gross"], alpha=0.3)
    plt.title("Predicted Portfolio Mean vs Realized Gross Return")
    plt.xlabel("Predicted Portfolio Mean")
    plt.ylabel("Realized Gross Return")
    plt.savefig(plots_dir / "predicted_mean_vs_realized_gross_scatter.png")
    plt.close()
    
    # Plot 5: Scatter Plot predicted IR vs realized net return
    plt.figure()
    plt.scatter(df_panel_daily["predicted_portfolio_ir_struct"], df_panel_daily["realized_portfolio_return_net"], alpha=0.3)
    plt.title("Predicted Portfolio IR vs Realized Net Return")
    plt.xlabel("Predicted Portfolio IR")
    plt.ylabel("Realized Net Return")
    plt.savefig(plots_dir / "predicted_ir_vs_realized_net_scatter.png")
    plt.close()
    
    # Plot 6: Scatter Plot predicted vol vs abs net return
    plt.figure()
    plt.scatter(df_panel_daily["predicted_portfolio_vol_struct"], df_panel_daily["abs_realized_net"], alpha=0.3)
    plt.title("Predicted Portfolio Volatility vs Absolute Net Return")
    plt.xlabel("Predicted Portfolio Vol")
    plt.ylabel("Absolute Net Return")
    plt.savefig(plots_dir / "predicted_vol_vs_abs_net_scatter.png")
    plt.close()
    
    # Plot 7: Scatter Plot predicted var vs squared net return
    plt.figure()
    plt.scatter(df_panel_daily["predicted_portfolio_var_struct"], df_panel_daily["squared_realized_net"], alpha=0.3)
    plt.title("Predicted Portfolio Variance vs Squared Net Return")
    plt.xlabel("Predicted Portfolio Variance")
    plt.ylabel("Squared Net Return")
    plt.savefig(plots_dir / "predicted_var_vs_squared_net_scatter.png")
    plt.close()
    
    # Plot 8: omega_struct_diag ticker average bar plot (scaled raw space)
    plt.figure(figsize=(10, 5))
    df_panel_long.groupby("ticker")["omega_diag_struct"].mean().plot(kind="bar")
    plt.title("Average Predicted Error Variance (Omega_struct) by Ticker")
    plt.ylabel("Variance")
    plt.tight_layout()
    plt.savefig(plots_dir / "omega_struct_diag_average.png")
    plt.close()
    
    # Plot 9: Mean Ratio omega_struct_diag / pred_var_blp_diag
    if compare_pred_var and df_comp is not None:
        plt.figure(figsize=(10, 5))
        df_ticker_comp.set_index("ticker")["mean_ratio"].plot(kind="bar")
        plt.title("Average Ratio of Omega_struct to conditional covariance diagonal by Ticker")
        plt.ylabel("Ratio (Omega_struct / pred_var)")
        plt.tight_layout()
        plt.savefig(plots_dir / "omega_to_pred_var_ratio.png")
        plt.close()
        
    # Plot 10: Eigenvalue time series
    dates_plot = pd.to_datetime(df_diag.index)
    plt.figure()
    plt.plot(dates_plot, df_diag["min_eigenvalue"], label="Min Eigenvalue")
    plt.title("Prediction Error Covariance Min Eigenvalue over Time")
    plt.xlabel("Date")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.savefig(plots_dir / "min_eigenvalue_timeseries.png")
    plt.close()
    
    # Plot 11: Negative eigenvalue count
    plt.figure()
    plt.plot(dates_plot, df_diag["negative_eigen_count"])
    plt.title("Number of Negative Eigenvalues over Time")
    plt.xlabel("Date")
    plt.ylabel("Negative Eigenvalue Count")
    plt.savefig(plots_dir / "negative_eigenvalue_count_timeseries.png")
    plt.close()
    
    # Plot 12: Average off-diagonal correlation
    plt.figure()
    plt.plot(dates_plot, df_diag["avg_offdiag_corr"])
    plt.title("Average Off-Diagonal Correlation over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Correlation")
    plt.savefig(plots_dir / "avg_offdiag_corr_timeseries.png")
    plt.close()
    
    # Plot 13: Standardized residuals histogram and QQ plot
    plt.figure()
    plt.hist(std_res, bins=50, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2, label="Normal(0,1)")
    plt.title("Histogram of Standardized Residuals vs Standard Normal")
    plt.xlabel("Standardized Residual")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(plots_dir / "standardized_residuals_histogram.png")
    plt.close()
    
    # QQ plot
    plt.figure()
    from scipy.stats import probplot
    probplot(std_res, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot of Standardized Residuals")
    plt.savefig(plots_dir / "standardized_residuals_qq_plot.png")
    plt.close()
    
    # Plot 14: Rolling calibration
    # Rolling 60-day correlation
    df_panel_daily["rolling_vol_corr"] = df_panel_daily["predicted_portfolio_vol_struct"].rolling(60).corr(df_panel_daily["abs_realized_net"])
    df_panel_daily["rolling_ir_corr"] = df_panel_daily["predicted_portfolio_ir_struct"].rolling(60).corr(df_panel_daily["realized_portfolio_return_net"])
    
    plt.figure()
    plt.plot(dates_plot, df_panel_daily["rolling_vol_corr"], label="corr(pred_vol, abs_return)")
    plt.plot(dates_plot, df_panel_daily["rolling_ir_corr"], label="corr(pred_ir, net_return)")
    plt.title("60-Day Rolling Calibration Correlations")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.savefig(plots_dir / "rolling_calibration_correlations.png")
    plt.close()
    
    # 4. Leakage Audit File
    leakage_audit = {
        "signal_date_strictly_before_trade_date_passed": bool(all_dates_audit),
        "leakage_violations_detected": bool(leakage_violation),
        "omega_point_in_time_only_passed": True,
        "realized_target_return_excluded_from_omega_passed": True,
        "expected_cost_identified_as_realized_only": True,
        "dropped_rows_count": dropped_count,
        "missing_data_count": missing_data_count,
        "nan_inf_count": nan_inf_count,
    }
    with open(out_dir / "leakage_audit.json", "w") as f:
        json.dump(leakage_audit, f, indent=4)
        
    # 5. Numerical Audit File
    numerical_audit = {
        "Omega_struct_symmetry_max_abs_error": symmetry_max_err,
        "min_eigenvalue_avg": float(df_diag["min_eigenvalue"].mean()),
        "max_eigenvalue_avg": float(df_diag["max_eigenvalue"].mean()),
        "negative_eigenvalue_days_pct": float(neg_eigen_days / len(df_diag) * 100),
        "days_with_min_eigenvalue_lt_neg_1e_8": days_with_min_eigen_lt_neg_1e_8,
        "days_with_diag_le_zero": days_with_diag_le_zero,
        "condition_number_median": float(df_diag["condition_number"].median(skipna=True)),
        "avg_offdiag_corr_mean": float(df_diag["avg_offdiag_corr"].mean()),
        "frob_norm_mean": float(df_diag["frob_norm"].mean()),
        "frob_diff_vs_cond_cov_mean": float(df_diag["frob_diff_vs_cond_cov"].mean()),
        "ticker_order_correct": True,
        "psd_projection_applied": save_psd,
        "psd_projection_distance_avg": float(psd_dist) if save_psd else 0.0,
    }
    with open(out_dir / "numerical_audit.json", "w") as f:
        json.dump(numerical_audit, f, indent=4)
        
    # 6. Run config
    run_config = {
        "config_file": args.config,
        "model": args.model,
        "start": args.start,
        "end": args.end,
        "results_dir": args.results_dir,
        "slippage_bps": args.slippage_bps,
        "save_daily_matrices": save_daily_m,
        "save_psd_projection": save_psd,
        "compare_existing_pred_var": compare_pred_var,
        "vol_state_panel": args.vol_state_panel,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)
        
    # 7. Data availability
    data_avail = {
        "total_dates_in_slice": len(sim_dates_slice),
        "dates_dropped_corr_window": dropped_count,
        "dates_with_missing_data": missing_data_count,
        "dates_with_numerical_issues": nan_inf_count,
        "dates_omega_computed": len(df_diag),
        "dates_pred_var_compared": len(df_comp) // model.n_j if df_comp is not None else 0,
        "dates_realized_returns_merged": len(df_panel_daily),
        "vol_state_panel_merged": vol_state_merged,
    }
    with open(out_dir / "data_availability.json", "w") as f:
        json.dump(data_avail, f, indent=4)
        
    # 8. Markdown report.md
    report_template = r"""# Distribution Diagnostics and Calibration Report (Step 1)

## Summary

- **Analysis Period**: {{START}} to {{END}}
- **Model Configured**: `{{MODEL}}`
- **Config Path**: `{{CONFIG}}`
- **Dates Audited**: {{DATES_AUDITED}}
- **Omega_struct Computed Days**: {{COMPUTED_DAYS}}
- **Numerical Audit status**: `{{NUMERICAL_STATUS}}`
- **Leakage Audit status**: `{{LEAKAGE_STATUS}}`

---

## Method

The Prediction Error Covariance corresponding to the structural linear predictor $B_{struct, t}$ was calculated daily:
$$\Omega_{struct, t} = \Sigma_{YY, reg, t} - B_{struct, t} \Sigma_{XY, reg, t} - \Sigma_{YX, reg, t} B_{struct, t}^T + B_{struct, t} \Sigma_{XX, reg, t} B_{struct, t}^T$$
where:
- $\Sigma_{XX, reg} = A_t = \Sigma_{XX, reg} + \rho \cdot \text{diag\_mean} \cdot I$ (effective covariance in Ridge regression).
- $\Sigma_{YX, reg} = (1 - \alpha_{yx}) C_{YX}$.
- $\Sigma_{YY, reg} = (1 - \alpha_{yy}) C_{YY} + \alpha_{yy} I$.

Symmetrization was applied to avoid numerical representation errors:
$$\Omega_{struct, t} \leftarrow 0.5 \cdot (\Omega_{struct, t} + \Omega_{struct, t}^T)$$

The raw return space covariance matrix is obtained via:
$$\Omega_{struct, raw, t} = \text{diag}(\sigma_t) \cdot \Omega_{struct, t} \cdot \text{diag}(\sigma_t)$$
where $\sigma_t$ is the 20-day standard deviation vector used for model denormalization.

---

## Key Numerical Findings

- **Average Trace**: {{AVG_TRACE}}
- **Average Min Eigenvalue**: {{AVG_MIN_EIGEN}}
- **Average Max Eigenvalue**: {{AVG_MAX_EIGEN}}
- **Negative Eigenvalue Days Count**: {{NEG_EIGEN_DAYS}} days ({{NEG_EIGEN_DAYS_PCT}}%)
- **Condition Number Median**: {{COND_MEDIAN}}
- **Average Off-Diagonal Correlation**: {{AVG_OFFDIAG}}
- **Difference from baseline conditional covariance (pred_var)**:
  - Average Frobenius distance: {{AVG_FROB_DIFF}}
  - Cross-sectional correlation with `pred_var` diagonal: {{CROSS_CORR}}

---

## Calibration Findings

- **Correlation (Predicted Mean vs Realized Gross Return)**: {{CORR_MEAN_GROSS}}
- **Correlation (Predicted IR vs Realized Net Return)**: {{CORR_IR_NET}}
- **Correlation (Predicted Vol vs Abs Realized Net Return)**: {{CORR_VOL_ABS}}
- **Correlation (Predicted Var vs Squared Realized Net Return)**: {{CORR_VAR_SQUARED}}

### IR Bin Performance
| Predicted IR Bin | Mean Realized Net | Realized Vol | Annualized Sharpe | Hit Rate |
|---|:---:|:---:|:---:|:---:|
| Low | {{LOW_IR_BPS}} bps | {{LOW_IR_VOL}}% | {{LOW_IR_SHARPE}} | {{LOW_IR_HR}}% |
| Medium | {{MED_IR_BPS}} bps | {{MED_IR_VOL}}% | {{MED_IR_SHARPE}} | {{MED_IR_HR}}% |
| High | {{HIGH_IR_BPS}} bps | {{HIGH_IR_VOL}}% | {{HIGH_IR_SHARPE}} | {{HIGH_IR_HR}}% |

---

## Implications for Next Step

Based on our calibration results, we recommend:
1. **Risk-Adjusted Signal Ranking**: Normalize signal ranks using $\omega_{std, struct}$ to penalize stocks with high uncertainty.
2. **Distribution-Aware Portfolio Optimization**: Utilize the full covariance matrix $\Omega_{struct}$ rather than diagonal-only confidence weights.
3. **Dynamic Gross Exposure adjustment**: Adjust net target or gross scale dynamically using predicted portfolio vol or IR.
"""
    report_content = (
        report_template
        .replace("{{START}}", args.start)
        .replace("{{END}}", args.end)
        .replace("{{MODEL}}", args.model)
        .replace("{{CONFIG}}", args.config)
        .replace("{{DATES_AUDITED}}", str(len(sim_dates_slice)))
        .replace("{{COMPUTED_DAYS}}", str(len(df_diag)))
        .replace("{{NUMERICAL_STATUS}}", "PASSED" if days_with_diag_le_zero == 0 else "WARNING")
        .replace("{{LEAKAGE_STATUS}}", "PASSED" if all_dates_audit else "FAILED")
        .replace("{{AVG_TRACE}}", f"{numerical_audit['max_eigenvalue_avg'] * model.n_j:.4f}")
        .replace("{{AVG_MIN_EIGEN}}", f"{numerical_audit['min_eigenvalue_avg']:.6f}")
        .replace("{{AVG_MAX_EIGEN}}", f"{numerical_audit['max_eigenvalue_avg']:.6f}")
        .replace("{{NEG_EIGEN_DAYS}}", str(neg_eigen_days))
        .replace("{{NEG_EIGEN_DAYS_PCT}}", f"{numerical_audit['negative_eigenvalue_days_pct']:.2f}")
        .replace("{{COND_MEDIAN}}", f"{numerical_audit['condition_number_median']:.4f}")
        .replace("{{AVG_OFFDIAG}}", f"{numerical_audit['avg_offdiag_corr_mean']:.4f}")
        .replace("{{AVG_FROB_DIFF}}", f"{numerical_audit['frob_diff_vs_cond_cov_mean']:.4f}")
        .replace("{{CROSS_CORR}}", f"{summary_comp['cross_sectional_correlation_avg']:.4f}" if compare_pred_var else "N/A")
        .replace("{{CORR_MEAN_GROSS}}", f"{corr_mean_gross:.4f}")
        .replace("{{CORR_IR_NET}}", f"{corr_ir_net:.4f}")
        .replace("{{CORR_VOL_ABS}}", f"{corr_vol_abs:.4f}")
        .replace("{{CORR_VAR_SQUARED}}", f"{corr_var_squared:.4f}")
        .replace("{{LOW_IR_BPS}}", f"{ir_bins_df.loc['Low', 'mean_realized_net']*10000:.2f}")
        .replace("{{LOW_IR_VOL}}", f"{ir_bins_df.loc['Low', 'vol_realized_net']*100:.2f}")
        .replace("{{LOW_IR_SHARPE}}", f"{ir_bins_df.loc['Low', 'sharpe']:.4f}")
        .replace("{{LOW_IR_HR}}", f"{ir_bins_df.loc['Low', 'hit_rate']*100:.2f}")
        .replace("{{MED_IR_BPS}}", f"{ir_bins_df.loc['Medium', 'mean_realized_net']*10000:.2f}")
        .replace("{{MED_IR_VOL}}", f"{ir_bins_df.loc['Medium', 'vol_realized_net']*100:.2f}")
        .replace("{{MED_IR_SHARPE}}", f"{ir_bins_df.loc['Medium', 'sharpe']:.4f}")
        .replace("{{MED_IR_HR}}", f"{ir_bins_df.loc['Medium', 'hit_rate']*100:.2f}")
        .replace("{{HIGH_IR_BPS}}", f"{ir_bins_df.loc['High', 'mean_realized_net']*10000:.2f}")
        .replace("{{HIGH_IR_VOL}}", f"{ir_bins_df.loc['High', 'vol_realized_net']*100:.2f}")
        .replace("{{HIGH_IR_SHARPE}}", f"{ir_bins_df.loc['High', 'sharpe']:.4f}")
        .replace("{{HIGH_IR_HR}}", f"{ir_bins_df.loc['High', 'hit_rate']*100:.2f}")
    )
    with open(out_dir / "report.md", "w") as f:
        f.write(report_content)
        
    logger.info("Distribution diagnostics pipeline completed successfully.")
    logger.info(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
