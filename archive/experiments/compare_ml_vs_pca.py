"""Comparative backtesting script: PCA vs LightGBM models under raw and vol-adjusted targets.

Saves a summary comparison table to results/ml_comparison_report.md and prints to stdout.
"""

from __future__ import annotations

import logging
import os
import sys
import numpy as np
import pandas as pd

# Add src/ directory to path to ensure proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import STRATEGY_DEFAULTS
from data.ticker_registry import US_TICKERS, JP_TICKERS, TOPIX_TICKER, N_US_ASSETS, N_JP_ASSETS
from data.downloader import download_data
from data.preprocessor import preprocess_data
from domain.models.types import StrategyConfig
from domain.signals import lead_lag as signals
from domain.models.ml_predictor import MLRollingRunner, compute_jp_volatility
from performance import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_gap_residual_signal(
    r_hat_jp_cc: np.ndarray,
    gap_vec: np.ndarray,
    betas_vec: np.ndarray | None,
    topix_night_t: float | None,
    gap_open_coef: float = 0.70,
    topix_beta_coef: float = 0.60,
) -> np.ndarray:
    """Compute trading signal using the gap residual filtering method."""
    n_j = len(r_hat_jp_cc)
    use_topix = (
        betas_vec is not None
        and topix_night_t is not None
        and np.all(np.isfinite(betas_vec))
        and np.isfinite(topix_night_t)
    )

    if use_topix:
        gap_syst = betas_vec * topix_night_t
        gap_idio = gap_vec - gap_syst
        gap_filt = gap_open_coef * gap_idio + (gap_open_coef - topix_beta_coef) * gap_syst
        denom = np.maximum(1.0 + gap_filt, 0.1)
        signal = (1.0 + r_hat_jp_cc) / denom - 1.0
    else:
        signal = r_hat_jp_cc - gap_open_coef * gap_vec

    return signal


def run_simulation(
    df_exec: pd.DataFrame,
    predictions_cc: pd.DataFrame,
    config: StrategyConfig,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Run lead-lag strategy backtest with custom price predictions."""
    jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
    gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]

    n_j = len(jp_oc_cols)
    jp_oc = df_exec[jp_oc_cols].values
    jp_gap = df_exec[gap_cols].values
    jp_beta = df_exec[beta_cols].values if beta_cols else None
    topix_night = (
        df_exec["topix_night_return"].values
        if "topix_night_return" in df_exec.columns
        else None
    )

    start_idx = df_exec.index.searchsorted(pd.to_datetime(start_date))

    results = []
    dispersion_history = []

    # Pre-populate dispersion history
    for hist_i in range(max(0, start_idx - 60), start_idx):
        pred_cc = predictions_cc.values[hist_i]
        gap_vec = jp_gap[hist_i]
        betas_vec = jp_beta[hist_i] if jp_beta is not None else None
        topix_night_t = topix_night[hist_i] if topix_night is not None else None

        sig = compute_gap_residual_signal(
            pred_cc,
            gap_vec,
            betas_vec,
            topix_night_t,
            config.gap_open_coef,
            config.topix_beta_coef,
        )
        disp = signals.compute_dispersion_indicator(
            sig, config.q, n_j, config.dispersion_metric
        )
        dispersion_history.append(disp)

    # Simulation loop
    for i in range(start_idx, len(df_exec)):
        t_trade = df_exec.index[i]

        pred_cc = predictions_cc.values[i]
        gap_vec = jp_gap[i]
        betas_vec = jp_beta[i] if jp_beta is not None else None
        topix_night_t = topix_night[i] if topix_night is not None else None

        signal = compute_gap_residual_signal(
            pred_cc,
            gap_vec,
            betas_vec,
            topix_night_t,
            config.gap_open_coef,
            config.topix_beta_coef,
        )

        weights = signals.build_weights(signal, config.q, n_j, config.weight_mode)

        dispersion_ind = signals.compute_dispersion_indicator(
            signal, config.q, n_j, config.dispersion_metric
        )
        scale = signals.dispersion_scale(
            dispersion_ind, dispersion_history, config.dispersion_filter
        )
        dispersion_history.append(dispersion_ind)

        scaled_weights = weights * scale

        # Calculate returns
        r_oc_t1 = jp_oc[i]
        daily_return_gross = float(np.sum(scaled_weights * r_oc_t1))

        # Slippage: 5 bps per side, round trip = 2 * bps * gross_exposure
        gross_exposure = float(np.sum(np.abs(scaled_weights)))
        slippage_rate = getattr(config, "slippage_bps", 5.0) / 10000.0
        slippage_cost = 2.0 * slippage_rate * gross_exposure
        daily_return = daily_return_gross - slippage_cost

        results.append(
            {
                "trade_date": t_trade,
                "daily_return": daily_return,
                "daily_return_gross": daily_return_gross,
                "slippage_cost": slippage_cost,
                "gross_exposure": gross_exposure,
                "weights": scaled_weights,
                "active_count": int(np.sum(np.abs(scaled_weights) > 1e-12)),
            }
        )

    res_df = pd.DataFrame(results).set_index("trade_date")
    return res_df


def compute_pca_predictions(
    df_exec: pd.DataFrame, config: StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute raw and vol-adjusted PCA predictions for all dates."""
    logger.info("Computing baseline PCA predictions...")
    all_cc_cols = [
        c for c in df_exec.columns if c.startswith("us_cc_") or c.startswith("jp_cc_")
    ]
    all_returns = df_exec[all_cc_cols].values
    date_index = df_exec.index.values
    n_u = N_US_ASSETS
    n_j = N_JP_ASSETS

    c_full = signals.compute_baseline_correlation(
        all_returns,
        date_index,
        config.ewma_half_life,
    )

    v0_static = signals.build_v3_static(
        n_u,
        n_j,
        config.include_v4_prior,
    )
    base_vectors = signals.build_base_vectors(n_u, n_j)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    jp_gap = df_exec[gap_cols].values if len(gap_cols) == n_j else None
    beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]
    jp_beta = df_exec[beta_cols].values if len(beta_cols) == n_j else None
    topix_night = (
        df_exec["topix_night_return"].values
        if "topix_night_return" in df_exec.columns
        else None
    )

    pca_cc_raw = np.zeros((len(df_exec), n_j))
    pca_z_raw = np.zeros((len(df_exec), n_j))

    vols_df = compute_jp_volatility(df_exec, JP_TICKERS, vol_window=20)
    vols = vols_df.values

    for i in range(len(df_exec)):
        if i < config.corr_window:
            pca_cc_raw[i] = np.nan
            pca_z_raw[i] = np.nan
            continue

        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_result = signals.compute_signal(
            all_returns,
            i,
            n_u,
            config.corr_window,
            c_full,
            v0_static,
            v1,
            v2,
            config.k,
            config.lambda_reg,
            config.lambda_lw,
            config.lw_target,
            config.ewma_half_life,
            v3_dynamic=(config.v3_mode == "dynamic"),
            gap_override=None,  # Pass None to get model's raw price predictions
            gap_open_coef=config.gap_open_coef,
            topix_beta_coef=config.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
        )

        pca_cc_raw[i] = sig_result["r_hat_jp_cc"]
        pca_z_raw[i] = sig_result["z_hat_j_t1"]

    pca_cc_df = pd.DataFrame(
        pca_cc_raw, index=df_exec.index, columns=[f"pred_cc_{tk}" for tk in JP_TICKERS]
    )

    # Vol-adjusted PCA: z_hat * sigma_20
    pca_cc_vol_adj = pca_z_raw * vols
    pca_cc_vol_adj_df = pd.DataFrame(
        pca_cc_vol_adj, index=df_exec.index, columns=[f"pred_cc_{tk}" for tk in JP_TICKERS]
    )

    return pca_cc_df, pca_cc_vol_adj_df


def calculate_metrics_for_report(res_df: pd.DataFrame) -> dict:
    """Calculate rich performance statistics for comparison table."""
    m = calculate_metrics(res_df["daily_return"])

    total_trades = int(res_df["active_count"].sum())
    avg_daily_trades = float(res_df["active_count"].mean())
    avg_gross_exposure = float(res_df["gross_exposure"].mean())

    weights = np.stack(res_df["weights"].values)
    diffs = np.zeros(len(weights))
    diffs[0] = np.sum(np.abs(weights[0]))
    for i in range(1, len(weights)):
        diffs[i] = np.sum(np.abs(weights[i] - weights[i - 1]))

    # Average one-way daily turnover
    avg_one_way_turnover = float(np.mean(diffs) / 2.0)

    m["Total Trades"] = total_trades
    m["Avg Daily Trades"] = avg_daily_trades
    m["Avg Gross Exposure"] = avg_gross_exposure
    m["Avg One-way Turnover"] = avg_one_way_turnover

    return m


def main():
    logger.info("Downloading historical market data...")
    data = download_data()
    logger.info("Preprocessing market data...")
    df_exec = preprocess_data(data)

    # Set up config
    config = StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
        dispersion_metric=STRATEGY_DEFAULTS["dispersion_metric"],
        v3_mode=STRATEGY_DEFAULTS["v3_mode"],
        ewma_half_life=STRATEGY_DEFAULTS["ewma_half_life"],
        lambda_lw=STRATEGY_DEFAULTS["lambda_lw"],
        lw_target=STRATEGY_DEFAULTS["lw_target"],
        corr_window=STRATEGY_DEFAULTS["corr_window"],
        include_v4_prior=STRATEGY_DEFAULTS["include_v4_prior"],
        signal_mode=STRATEGY_DEFAULTS["signal_mode"],
        gap_open_coef=STRATEGY_DEFAULTS["gap_open_coef"],
        topix_beta_coef=STRATEGY_DEFAULTS["topix_beta_coef"],
        beta_window=STRATEGY_DEFAULTS["beta_window"],
        gamma=STRATEGY_DEFAULTS.get("gamma", 0.5),
        slippage_bps=STRATEGY_DEFAULTS.get("slippage_bps", 5.0),
    )

    test_start_date = "2020-01-01"

    # 1. Compute PCA baseline predictions
    pca_raw_preds, pca_vol_adj_preds = compute_pca_predictions(df_exec, config)

    # 2. Run LightGBM Rolling predictions
    logger.info("Running LightGBM Raw Target predictions...")
    runner_raw = MLRollingRunner(
        df_exec=df_exec,
        us_tickers=US_TICKERS,
        jp_tickers=JP_TICKERS,
        train_window=250,
        refit_interval=10,  # Refit every 10 days for performance
    )
    lgb_raw_preds = runner_raw.run_rolling_predictions(
        start_date=test_start_date, vol_adjusted_target=False
    )

    logger.info("Running LightGBM Vol-adjusted Target predictions...")
    runner_vol = MLRollingRunner(
        df_exec=df_exec,
        us_tickers=US_TICKERS,
        jp_tickers=JP_TICKERS,
        train_window=250,
        refit_interval=10,
    )
    lgb_vol_preds = runner_vol.run_rolling_predictions(
        start_date=test_start_date, vol_adjusted_target=True
    )

    # 3. Run backtests
    logger.info("Running Case 1-1 backtest...")
    res_11 = run_simulation(df_exec, pca_raw_preds, config, start_date=test_start_date)
    logger.info("Running Case 1-2 backtest...")
    res_12 = res_12 = run_simulation(
        df_exec, pca_vol_adj_preds, config, start_date=test_start_date
    )
    logger.info("Running Case 2-1 backtest...")
    res_21 = run_simulation(df_exec, lgb_raw_preds, config, start_date=test_start_date)
    logger.info("Running Case 2-2 backtest...")
    res_22 = run_simulation(df_exec, lgb_vol_preds, config, start_date=test_start_date)

    # 4. Calculate metrics
    m11 = calculate_metrics_for_report(res_11)
    m12 = calculate_metrics_for_report(res_12)
    m21 = calculate_metrics_for_report(res_21)
    m22 = calculate_metrics_for_report(res_22)

    # Build report dict
    report_data = {
        "Case 1-1 (PCA × Raw Target)": m11,
        "Case 1-2 (PCA × Vol-adjusted Target)": m12,
        "Case 2-1 (LightGBM × Raw Target)": m21,
        "Case 2-2 (LightGBM × Vol-adjusted Target)": m22,
    }

    # Generate markdown table
    header = "| Metric | Case 1-1 (PCA × Raw) | Case 1-2 (PCA × Vol-adj) | Case 2-1 (LGBM × Raw) | Case 2-2 (LGBM × Vol-adj) |"
    separator = "| --- | --- | --- | --- | --- |"
    rows = []

    metrics_keys = [
        ("Annualized Return (AR)", "AR", "{:.2f}%", 100.0),
        ("Annualized Volatility (RISK)", "RISK", "{:.2f}%", 100.0),
        ("Risk-Return Ratio (Sharpe)", "Sharpe", "{:.4f}", 1.0),
        ("Max Drawdown (MDD)", "MDD", "{:.2f}%", 100.0),
        ("Total Trades", "Total Trades", "{:.0f}", 1.0),
        ("Avg Daily Trades", "Avg Daily Trades", "{:.2f}", 1.0),
        ("Avg Gross Exposure", "Avg Gross Exposure", "{:.2f}%", 100.0),
        ("Avg One-way Daily Turnover", "Avg One-way Turnover", "{:.2f}%", 100.0),
    ]

    for label, key, fmt, scale in metrics_keys:
        row = f"| {label} "
        for case in report_data.keys():
            val = report_data[case].get(key, np.nan)
            if np.isnan(val):
                row += "| N/A "
            else:
                row += f"| {fmt.format(val * scale)} "
        row += "|"
        rows.append(row)

    table_md = "\n".join([header, separator] + rows)

    # Generate final report string
    report_text = f"""# 2 by 2 Comparative Performance Report: PCA vs LightGBM

This report compares the performance of PCA and LightGBM model combinations with Raw and Volatility-adjusted (Option3) targets over the testing period from **{test_start_date}** to the end of the available historical dataset.

## Summary Results Table

{table_md}

## Key Observations and Performance Analysis

1. **Model Comparison (PCA vs. LightGBM)**:
   - LightGBM captures non-linear relationships and interactions among sector returns, potentially yielding a higher raw forecast precision.
   - However, the PCA model benefits from explicit group structure priors (sector sensitivity vectors v1-v6), which keep predictions stable and aligned with known macroeconomic exposures.

2. **Target Variable Comparison (Raw vs. Volatility-adjusted)**:
   - Volatility-adjusted (Z-score) targets adjust returns by their rolling 20-day historical standard deviation. This acts to scale down predictions during highly volatile regimes and scale them up during low volatility regimes, reducing portfolio risk concentrations.
   - Standardizing the target helps stabilize the LightGBM learning process, as the target variable is homogeneous across time and different ETFs.

3. **Risk-adjusted Performance**:
   - The Sharpe (Risk-Return) ratio is the primary indicator of risk-adjusted stability. Verify the Sharpe ratio to identify which combination provides the most stable alpha generation.
   - Max Drawdown (MDD) highlights the tail-risk reduction capacity of each approach, especially when using the volatility-adjusted target.
"""

    # Output to stdout
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS SUMMARY")
    print("=" * 50)
    print(table_md)
    print("=" * 50)

    # Save to report file
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "ml_comparison_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Comparison report successfully saved to: %s", report_path)


if __name__ == "__main__":
    main()
