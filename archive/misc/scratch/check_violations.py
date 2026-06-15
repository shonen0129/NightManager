import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
sys.path.insert(0, str(Path('.')))
import numpy as np
import pandas as pd
import logging

from config import STRATEGY_DEFAULTS, N_US_ASSETS, N_JP_ASSETS
from data.downloader import download_data
from data.preprocessor import preprocess_data
from data.ticker_registry import JP_TICKERS, US_TICKERS, TOPIX_TICKER
from domain.signals.lead_lag import (
    build_v3_static,
    build_base_vectors,
)
from domain.signals import lead_lag as signals
from domain.models.residual_lowrank import compute_rolling_ols_betas
from tools.backtest_p6_production_residual_ensemble import simulate_p6_fast, build_portfolio_weights, cs_normalize

# Setup data
std_data = download_data(beta_window=60)
df_exec = preprocess_data(std_data, beta_window=60)

topix_close = std_data["jp_close"][TOPIX_TICKER].copy()
topix_open = std_data["jp_open"][TOPIX_TICKER].copy()
topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
r_topix_oc = topix_close / topix_open - 1.0
df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values

for tk in JP_TICKERS:
    df_exec[f"jp_trade_cc_{tk}"] = (1.0 + df_exec[f"jp_gap_{tk}"]) * (1.0 + df_exec[f"jp_oc_{tk}"]) - 1.0
df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
jp_cc_cols = [f"jp_trade_cc_{tk}" for tk in JP_TICKERS]
y_jp_oc_df = df_exec[jp_oc_cols].rename(columns=lambda c: c.replace("jp_oc_", ""))
y_jp_cc_df = df_exec[jp_cc_cols].rename(columns=lambda c: c.replace("jp_trade_cc_", ""))
y_topix_oc_series = df_exec["topix_oc_return"]
y_topix_cc_series = df_exec["topix_cc_trade"]
all_returns_raw = df_exec[[c for c in df_exec.columns if c.startswith("us_cc_") or c.startswith("jp_cc_")]].values

config_base = {
    "train_window": 756,
}
start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-05")), config_base["train_window"] + 120)
sim_dates = df_exec.index[start_idx:]
T = len(df_exec)

v0_static = build_v3_static(15, 17, include_v4=True)
base_vectors = signals.build_base_vectors(15, 17)
v1, v2 = base_vectors["v1"], base_vectors["v2"]
c_full = signals.compute_baseline_correlation(all_returns_raw, df_exec.index.values, 45)

jp_gap = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values
jp_beta = df_exec[[f"jp_beta_{tk}" for tk in JP_TICKERS]].values
topix_night = df_exec["topix_night_return"].values

y_data_p3 = y_jp_cc_df[JP_TICKERS].values
x_data_p3 = y_topix_cc_series.values.reshape(-1, 1)
betas_jp_p3 = compute_rolling_ols_betas(y_data_p3, x_data_p3, 60)
y_residuals_p3 = y_data_p3 - betas_jp_p3[:, :, 0] * x_data_p3
y_residuals_p3_shifted = np.roll(y_residuals_p3, 1, axis=0)
y_residuals_p3_shifted[0] = 0.0
jp_res_returns_p3 = all_returns_raw.copy()
jp_res_returns_p3[:, 15:] = y_residuals_p3_shifted

market_percentiles = {0.95: np.zeros(T)}
etf_percentiles = {0.99: np.zeros((T, 17))}
topix_night_abs = np.abs(topix_night)
jp_gap_abs = np.abs(jp_gap)

for i in range(start_idx, T):
    hist_window_topix = topix_night_abs[i - 252 : i]
    hist_window_jp = jp_gap_abs[i - 252 : i]
    market_percentiles[0.95][i] = np.percentile(hist_window_topix, 95.0)
    for j in range(17):
        etf_percentiles[0.99][i, j] = np.percentile(hist_window_jp[:, j], 99.0)

# Raw signals
daily_signals = {
    "P0": np.zeros((T, 17)),
    "P3": np.zeros((T, 17))
}
for idx, date in enumerate(sim_dates):
    i = start_idx + idx
    gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0)
    betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
    topix_night_t = float(topix_night[i]) if topix_night is not None else None

    sig_res_p0 = signals.compute_signal(
        all_returns_raw, i, 15, 60, c_full, v0_static, v1, v2,
        6, 0.75, 0.5, "equicorrelation", 45, v3_dynamic=False,
        gap_override=gap_t1, gap_open_coef=0.70, topix_beta_coef=0.6,
        betas_t=betas_t, topix_night_t=topix_night_t, vol_adjusted_target=True
    )
    daily_signals["P0"][i] = sig_res_p0["signal"]

    sig_res_p3 = signals.compute_signal(
        jp_res_returns_p3, i, 15, 60, c_full, v0_static, v1, v2,
        6, 0.75, 0.5, "equicorrelation", 45, v3_dynamic=False,
        gap_override=gap_t1, gap_open_coef=0.70, topix_beta_coef=0.6,
        betas_t=betas_t, topix_night_t=topix_night_t, vol_adjusted_target=True
    )
    daily_signals["P3"][i] = sig_res_p3["signal"]

# Now audit all models
w_p0_list = []
w_p3_list = []
w_eq_list = []
sim_dates_idx = list(range(start_idx, T))
y_jp_oc_all = y_jp_oc_df.values

for idx, date in enumerate(sim_dates):
    i = start_idx + idx
    w_t0 = build_portfolio_weights(daily_signals["P0"][i])
    disp0 = signals.compute_dispersion_indicator(daily_signals["P0"][i], 0.3, 17, "long_short_mean_gap")
    disp0_hist = []
    for h in range(max(0, i - 60), i):
        if not np.isnan(daily_signals["P0"][h]).all():
            disp0_hist.append(signals.compute_dispersion_indicator(daily_signals["P0"][h], 0.3, 17, "long_short_mean_gap"))
    scale0 = signals.dispersion_scale(disp0, disp0_hist, False)
    w_p0_list.append(w_t0 * scale0)

    w_t3 = build_portfolio_weights(daily_signals["P3"][i])
    disp3 = signals.compute_dispersion_indicator(daily_signals["P3"][i], 0.3, 17, "long_short_mean_gap")
    disp3_hist = []
    for h in range(max(0, i - 60), i):
        if not np.isnan(daily_signals["P3"][h]).all():
            disp3_hist.append(signals.compute_dispersion_indicator(daily_signals["P3"][h], 0.3, 17, "long_short_mean_gap"))
    scale3 = signals.dispersion_scale(disp3, disp3_hist, False)
    w_p3_list.append(w_t3 * scale3)

    sig_comb = 0.5 * cs_normalize(daily_signals["P0"][i], "zscore") + 0.5 * cs_normalize(daily_signals["P3"][i], "zscore")
    w_t_eq = build_portfolio_weights(sig_comb)
    w_eq_list.append(w_t_eq)

daily_weights_out = {
    "P0": pd.DataFrame(w_p0_list, index=sim_dates, columns=JP_TICKERS),
    "P3": pd.DataFrame(w_p3_list, index=sim_dates, columns=JP_TICKERS),
    "ens_P0_P3_equal": pd.DataFrame(w_eq_list, index=sim_dates, columns=JP_TICKERS)
}

# Add overlay models
import json
with open("results/p6_production_residual_ensemble/p6_selected_params.json") as f:
    best_params = json.load(f)

def_vol_target = {"vol_target_enabled": True, "target_vol": 0.20, "vol_window": 20, "vol_scale_min": 0.5, "vol_scale_max": 1.25}
def_drawdown = {"drawdown_scaling_enabled": True, "dd_window_short": 20, "dd_threshold_short": -0.03, "dd_scale_short": 0.75, "dd_window_long": 60, "dd_threshold_long": -0.05, "dd_scale_long": 0.5}
def_market_gap = {"gap_market_filter": True, "market_gap_threshold": 0.95, "market_gap_scale": 0.75}
def_individual_gap = {"individual_gap_filter": True, "individual_gap_threshold": 0.99, "individual_gap_scale": 0.75}
def_ic = {"ic_filter_enabled": True, "ic_window": 60, "ic_threshold": 0.0, "ic_scale": 0.75}

overlay_variants = {
    "P6_base_50_50": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 1.0},
    "P6_agreement": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 0.5},
    "P6_gap_filter": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 1.0, **def_market_gap, **def_individual_gap},
    "P6_vol_target": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 1.0, **def_vol_target},
    "P6_drawdown_scaling": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 1.0, **def_drawdown},
    "P6_agree_gap": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 0.5, **def_market_gap, **def_individual_gap},
    "P6_agree_vol_dd": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 0.5, **def_vol_target, **def_drawdown},
    "P6_full": {"w0": 0.5, "norm_method": "zscore", "agreement_mode": "hard_penalty", "agreement_penalty": 0.5, **def_market_gap, **def_individual_gap, **def_vol_target, **def_drawdown, **def_ic},
    "P6_optimal": best_params
}

for name, cfg in overlay_variants.items():
    res = simulate_p6_fast(
        daily_signals["P0"], daily_signals["P3"], cfg,
        sim_dates_idx, df_exec, jp_gap, topix_night, y_jp_oc_all,
        {0.95: market_percentiles[0.95], 0.975: market_percentiles[0.95], 0.99: market_percentiles[0.95]}, # placeholder
        {0.975: etf_percentiles[0.99], 0.99: etf_percentiles[0.99]} # placeholder
    )
    daily_weights_out[name] = res["weights"]

for m, w_df in daily_weights_out.items():
    viols = 0
    max_net = 0
    max_gross = 0
    for idx in w_df.index:
        w = w_df.loc[idx].values
        net = np.sum(w)
        gross = np.sum(np.abs(w))
        max_net = max(max_net, abs(net))
        max_gross = max(max_gross, gross)
        if abs(net) > 1e-4 or gross > 2.0001:
            viols += 1
    print(f"Model: {m:<25} Violations: {viols:<5} Max Net: {max_net:.6f} Max Gross: {max_gross:.6f}")
