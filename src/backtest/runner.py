"""Backtest runner: executes the strategy over historical data."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from config import STRATEGY_DEFAULTS, N_US_ASSETS, N_JP_ASSETS
from data_loader import download_data, preprocess_data
from domain.models.types import StrategyConfig
from domain.signals import lead_lag as signals
from results_format import create_results_output_dir

logger = logging.getLogger(__name__)

# Numeric constants from original strategy
EPSILON_WEIGHT = 1e-12


def run_backtest_with_config(
    df_exec: pd.DataFrame,
    config: StrategyConfig,
    start_date: str = "2015-01-01",
    w6_override: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run backtest with the given strategy config.

    Args:
        df_exec: Execution DataFrame from preprocess_data
        config: StrategyConfig
        start_date: Backtest start date

    Returns:
        DataFrame with daily results (index=trade_date)
    """
    all_cc_cols = [
        c for c in df_exec.columns if c.startswith("us_cc_") or c.startswith("jp_cc_")
    ]
    jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
    jp_close_sig_cols = [c for c in df_exec.columns if c.startswith("jp_close_sig_")]
    jp_open_trade_cols = [c for c in df_exec.columns if c.startswith("jp_open_trade_")]

    all_returns = df_exec[all_cc_cols].values
    date_index = df_exec.index.values
    jp_oc = df_exec[jp_oc_cols].values if jp_oc_cols else None
    jp_close_sig = df_exec[jp_close_sig_cols].values if jp_close_sig_cols else None
    jp_open_trade = df_exec[jp_open_trade_cols].values if jp_open_trade_cols else None

    n_u = N_US_ASSETS
    n_j = N_JP_ASSETS

    # Pre-compute baseline correlation
    c_full = signals.compute_baseline_correlation(
        all_returns,
        date_index,
        config.ewma_half_life,
    )

    # Build V0 static
    v0_static = signals.build_v3_static(
        n_u,
        n_j,
        config.include_v4_prior,
        w6_override=w6_override,
    )
    base_vectors = signals.build_base_vectors(n_u, n_j)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    # Determine start index
    start_idx = max(
        df_exec.index.searchsorted(pd.to_datetime(start_date)),
        config.corr_window,
    )

    # Pre-extract gap data as numpy array to avoid per-iteration DataFrame access
    gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    jp_gap = df_exec[gap_cols].values if len(gap_cols) == n_j else None
    beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]
    jp_beta = df_exec[beta_cols].values if len(beta_cols) == n_j else None
    topix_night = (
        df_exec["topix_night_return"].values
        if "topix_night_return" in df_exec.columns
        else None
    )

    results = []
    dispersion_history = []

    def _compute_dispersion_at(index: int) -> float:
        gap_hist = None
        if config.signal_mode == "gap_residual":
            gap_hist = (
                np.nan_to_num(jp_gap[index], nan=0.0)
                if jp_gap is not None
                else np.zeros(n_j)
            )
        betas_hist = (
            np.asarray(jp_beta[index], dtype=float) if jp_beta is not None else None
        )
        topix_night_hist = (
            float(topix_night[index]) if topix_night is not None else None
        )
        sig_result_hist = signals.compute_signal(
            all_returns,
            index,
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
            gap_override=gap_hist,
            gap_open_coef=config.gap_open_coef,
            topix_beta_coef=config.topix_beta_coef,
            betas_t=betas_hist,
            topix_night_t=topix_night_hist,
            vol_adjusted_target=config.vol_adjusted_target,
        )

        signal_hist = np.asarray(sig_result_hist["signal"], dtype=float)

        return signals.compute_dispersion_indicator(
            signal_hist,
            config.q,
            n_j,
            config.dispersion_metric,
        )

    history_start = max(0, start_idx - 60)
    for hist_i in range(history_start, start_idx):
        dispersion_history.append(_compute_dispersion_at(hist_i))

    for i in range(start_idx, len(df_exec)):
        t_trade = df_exec.index[i]

        # Compute signal
        gap_t1 = None
        if config.signal_mode == "gap_residual":
            gap_t1 = (
                np.nan_to_num(jp_gap[i], nan=0.0)
                if jp_gap is not None
                else np.zeros(n_j)
            )
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
            gap_override=gap_t1,
            gap_open_coef=config.gap_open_coef,
            topix_beta_coef=config.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=config.vol_adjusted_target,
        )

        signal = np.asarray(sig_result["signal"], dtype=float)
        sigma_s = float(sig_result["sigma_s"])

        gap_stats = (
            np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(n_j)
        )

        # Compute dispersion indicator
        dispersion_ind = signals.compute_dispersion_indicator(
            signal,
            config.q,
            n_j,
            config.dispersion_metric,
        )

        # Build weights
        if (
            config.signal_mode == "gap_tolerant"
            and jp_close_sig is not None
            and jp_open_trade is not None
        ):
            base_weights = signals.build_weights(
                signal,
                config.q,
                n_j,
                config.weight_mode,
            )
            jp_close_t = np.nan_to_num(jp_close_sig[i], nan=1.0, copy=True)
            jp_open_t1 = np.nan_to_num(jp_open_trade[i], nan=1.0, copy=True)
            weights, long_exec, short_exec, executed = (
                signals.apply_gap_tolerant_filter(
                    signal,
                    sigma_s,
                    base_weights,
                    jp_close_t,
                    jp_open_t1,
                    config.gamma,
                    config.q,
                    n_j,
                    config.weight_mode,
                )
            )
        else:
            weights = signals.build_weights(
                signal,
                config.q,
                n_j,
                config.weight_mode,
            )

        # Apply dispersion scale
        scale = signals.dispersion_scale(
            dispersion_ind,
            dispersion_history,
            config.dispersion_filter,
        )
        dispersion_history.append(dispersion_ind)

        scaled_weights = weights * scale

        # Compute daily return (before slippage)
        r_oc_t1 = (
            np.nan_to_num(jp_oc[i], nan=0.0, copy=True)
            if jp_oc is not None
            else np.zeros(n_j)
        )
        daily_return_gross = float(np.sum(scaled_weights * r_oc_t1))
        long_ret = float(
            np.sum(scaled_weights[scaled_weights > 0] * r_oc_t1[scaled_weights > 0])
        )
        short_ret = float(
            np.sum(scaled_weights[scaled_weights < 0] * r_oc_t1[scaled_weights < 0])
        )

        # Slippage cost: round-trip = 2 × (bps/10000) × gross_exposure
        # TOPIX-17 ETFの厕付き成行注文におけるスプレッド+約定不利を模定。
        gross_exposure = float(np.sum(np.abs(scaled_weights)))
        slippage_rate = getattr(config, 'slippage_bps', 5.0) / 10000.0
        slippage_cost = 2.0 * slippage_rate * gross_exposure
        daily_return = daily_return_gross - slippage_cost

        results.append(
            {
                "trade_date": t_trade,
                "daily_return": daily_return,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "daily_return_gross": daily_return_gross,
                "slippage_cost": slippage_cost,
                "gross_exposure": gross_exposure,
                "sigma_s": sigma_s,
                "dispersion_indicator": dispersion_ind,
                "scale": scale,
                "signal_mode": config.signal_mode,
                "gap_open_coef": config.gap_open_coef,
                "dispersion_metric": config.dispersion_metric,
                "signal_mean": float(np.mean(signal)),
                "signal_std": float(np.std(signal)),
                "weight_concentration": float(np.sqrt(np.sum(scaled_weights**2))),
                "active_count": int(np.sum(np.abs(scaled_weights) > 1e-12)),
                "gap_mean": float(np.mean(gap_stats)),
                "gap_std": float(np.std(gap_stats)),
            }
        )

    return pd.DataFrame(results).set_index("trade_date")


def main():
    """Run backtest from CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Backtest runner")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    data = download_data(beta_window=STRATEGY_DEFAULTS["beta_window"])
    df_exec = preprocess_data(data, beta_window=STRATEGY_DEFAULTS["beta_window"])

    # Build config
    config = StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
        dispersion_metric=STRATEGY_DEFAULTS.get(
            "dispersion_metric", "long_short_mean_gap"
        ),
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
    )

    # Run backtest
    logger.info("Running backtest...")
    results = run_backtest_with_config(df_exec, config, args.start_date)

    # Save results
    output_dir = args.output_dir or create_results_output_dir(
        run_name="backtest_runner",
        manifest_extra={"entry_point": "backtest/runner.py"},
    )
    if args.output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results.to_csv(os.path.join(output_dir, "daily_results.csv"), encoding="utf-8-sig")

    # Calculate and save metrics
    from performance import calculate_metrics

    metrics = calculate_metrics(results["daily_return"])
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Print summary
    print("\n=== Backtest Results ===")
    for k, v in metrics.items():
        if k in ["AR", "RISK", "MDD", "Total Return"]:
            print(f"{k}: {v*100:.2f}%")
        elif k == "Sharpe":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v:.2f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
