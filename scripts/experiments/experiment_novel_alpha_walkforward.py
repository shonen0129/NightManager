"""Walk-forward OOS validation for novel alpha sources.

Design:
  Train window: 504 days (2yr) — compute IC per (signal, sector) pair
  Test window:  63 days (1q)  — apply train-derived weights OOS
  Step:         63 days
  Purge:        1 day

For each test window:
  1. Compute IC of each alt signal vs each JP sector using train window only
  2. Select significant pairs (p < 0.05, |IC| > 0.02)
  3. Build combined signal: BLPX + blend_weight * IC * z(alt_signal)
  4. Run backtest on test window

Aggregate all test windows for final OOS Sharpe.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
    _BLP_CORR_CACHE,
    _RAW_PCA_RESIDUAL_PCA_CACHE,
)
from leadlag.core.signal import build_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_PARAMS = {
    "alpha_xx": 0.20, "alpha_yy": 0.50, "alpha_yx": 0.15,
    "lambda_pca": 0.10, "lambda_sector": 0.60, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 504,
    "blp_ewma_halflife": 120, "sector_eta": 0.5, "sector_gamma": 4.0,
}
SIGNAL_WEIGHTS = {
    "raw_pca": {"enabled": True, "weight": 0.2},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": True, "weight": 0.8},
    "residual_blpx": {"enabled": False, "weight": 0.0},
}

TRAIN_WINDOW = 504
TEST_WINDOW = 63
STEP = 63
PURGE = 1
BLEND_WEIGHTS = [0.0, 0.20, 0.30, 0.40]
MIN_IC = 0.02
P_THRESHOLD = 0.05


def compute_train_ic(alt_data, y_target, sim_dates, jp_tickers, train_start, train_end):
    """Compute IC for each (signal, sector) pair using only train window."""
    results = []
    for signal_name, signal_series in alt_data.items():
        sig_aligned = signal_series.reindex(sim_dates)

        for j, tk in enumerate(jp_tickers):
            y_j = y_target[:, j]
            pairs = []
            for i in range(train_start, train_end):
                sig_val = sig_aligned.iloc[i]
                y_val = y_j[i]
                if np.isfinite(sig_val) and np.isfinite(y_val):
                    pairs.append((sig_val, y_val))

            if len(pairs) < 50:
                continue

            arr = np.array(pairs)
            rho, pval = stats.spearmanr(arr[:, 0], arr[:, 1])

            if pval < P_THRESHOLD and abs(rho) >= MIN_IC:
                results.append((signal_name, tk, float(rho)))

    return results


def build_combined_for_window(blpx_signals, alt_data, sig_pairs,
                               sim_dates, jp_tickers, blend_weight,
                               test_start, test_end):
    """Build combined signal for a specific test window."""
    n_j = len(jp_tickers)
    combined = blpx_signals.reindex(sim_dates).fillna(0.0).copy()

    ticker_signals = {}
    for signal_name, ticker, ic in sig_pairs:
        ticker_signals.setdefault(ticker, []).append((signal_name, ic))

    for j, tk in enumerate(jp_tickers):
        if tk not in ticker_signals:
            continue
        for signal_name, ic in ticker_signals[tk]:
            if signal_name not in alt_data:
                continue
            alt_series = alt_data[signal_name].reindex(sim_dates).shift(1)
            rmean = alt_series.rolling(252, min_periods=60).mean()
            rstd = alt_series.rolling(252, min_periods=60).std()
            z_alt = (alt_series - rmean) / rstd.replace(0, np.nan)
            adjustment = blend_weight * ic * z_alt
            combined[tk] = combined[tk] + adjustment.reindex(sim_dates).fillna(0.0).values

    return combined


class WFModel:
    """Walk-forward model with pre-computed weights for a specific window."""
    def __init__(self, combined_signals, df_exec, win_start, win_end):
        self.combined_signals = combined_signals
        self.df_exec = df_exec
        self.win_start = win_start
        self.win_end = win_end
        self.n_j = len(JP_TICKERS)
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"
        self._wc = 0
        self._weights = None

    def _compute_weights(self):
        T = len(self.df_exec)
        sd = self.df_exec.index
        weights = np.zeros((T, self.n_j))
        for i in range(self.win_start, self.win_end):
            if i >= T:
                break
            sig_i = self.combined_signals.iloc[i].values
            if np.isfinite(sig_i).any():
                weights[i] = build_weights(sig_i, q=self.q, n_j=self.n_j,
                                           weight_mode="signal", enforce_sign=False)
        self._weights = weights

    def predict_signals(self, df_exec):
        if self._weights is None:
            self._compute_weights()
        si = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), self.corr_window)
        self._wc = si
        T = len(df_exec)
        sd = df_exec.index
        blpx = self.combined_signals.reindex(sd).fillna(0.0)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sd, columns=JP_TICKERS)
        y_oc = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", ""))
        return {"raw_pca_signals": empty, "residual_pca_signals": empty,
                "p4_signals": empty, "signals": blpx,
                "normalized_signals": blpx, "y_jp_oc_df": y_oc}

    def build_weights(self, signal, q=None):
        if self._weights is not None and self._wc < len(self._weights):
            w = self._weights[self._wc]
            self._wc += 1
            return w
        return np.zeros(self.n_j)


def run_wf_backtest(combined_signals, df_exec, y_target, win_start, win_end):
    """Run backtest on a specific window."""
    model = WFModel(combined_signals, df_exec, win_start, win_end)
    results = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-01",
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0, slippage_bps=5.0)
    return results


def main():
    output_dir = ROOT / "artifacts" / "novel_alpha"
    yaml_path = str(ROOT / "configs" / "production.yaml")

    logger.info("Loading data...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    T = len(sim_dates)
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)

    alt_data = pd.read_pickle(output_dir / "novel_data.pkl")

    # Get BLPX baseline signals
    logger.info("Computing BLPX baseline...")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("blpx", {}).update(BASE_PARAMS)
    cfg["signal_components"] = SIGNAL_WEIGHTS
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    blpx_signals = model_base.predict_signals(df_exec)["signals"]

    # Generate walk-forward windows
    windows = []
    wf_start = start_idx + TRAIN_WINDOW + PURGE
    while wf_start + TEST_WINDOW <= T:
        train_start = wf_start - TRAIN_WINDOW - PURGE
        train_end = wf_start - PURGE
        test_start = wf_start
        test_end = min(wf_start + TEST_WINDOW, T)
        windows.append((train_start, train_end, test_start, test_end))
        wf_start += STEP

    logger.info("Walk-forward windows: %d (train=%d, test=%d, step=%d)",
                len(windows), TRAIN_WINDOW, TEST_WINDOW, STEP)

    # For each blend_weight, run walk-forward
    all_results = {}

    for bw in BLEND_WEIGHTS:
        label = "baseline" if bw == 0.0 else f"bw{bw:.2f}"
        logger.info("=== Walk-forward: %s ===", label)

        # Collect daily returns from all test windows
        all_daily_returns = []
        all_turnover = []
        n_pairs_per_window = []

        for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            if bw == 0.0:
                combined = blpx_signals
                n_pairs = 0
            else:
                # Compute IC on train window
                sig_pairs = compute_train_ic(
                    alt_data, y_target, sim_dates, JP_TICKERS, tr_s, tr_e)
                n_pairs = len(sig_pairs)

                if n_pairs == 0:
                    combined = blpx_signals
                else:
                    combined = build_combined_for_window(
                        blpx_signals, alt_data, sig_pairs,
                        sim_dates, JP_TICKERS, bw, te_s, te_e)

            # Run backtest on test window
            results = run_wf_backtest(combined, df_exec, y_target, te_s, te_e)
            dr = results["daily_returns"]

            # Extract only test window returns
            test_dates = sim_dates[te_s:te_e]
            dr_window = dr.reindex(test_dates).dropna()
            if len(dr_window) > 0:
                all_daily_returns.append(dr_window)
                all_turnover.append(results["daily_turnover"].reindex(test_dates).dropna())

            n_pairs_per_window.append(n_pairs)

            if wi % 10 == 0:
                logger.info("  Window %d/%d: %s-%s, n_pairs=%d, dr.mean=%.4f",
                            wi+1, len(windows),
                            sim_dates[te_s].date(), sim_dates[te_e-1].date(),
                            n_pairs, dr_window.mean() if len(dr_window) > 0 else np.nan)

        # Aggregate OOS performance
        all_dr = pd.concat(all_daily_returns) if all_daily_returns else pd.Series(dtype=float)
        ar = float(all_dr.mean() * 245)
        vol = float(all_dr.std(ddof=1) * np.sqrt(245))
        sharpe = ar / vol if vol > 0 else np.nan
        wealth = (1.0 + all_dr).cumprod()
        mdd = float(((wealth / wealth.cummax()) - 1.0).min())
        turnover = float(pd.concat(all_turnover).mean()) if all_turnover else np.nan
        avg_pairs = float(np.mean(n_pairs_per_window)) if n_pairs_per_window else 0.0

        all_results[label] = {
            "Sharpe": sharpe, "AR": ar, "Vol": vol, "MDD": mdd,
            "Turnover": turnover, "avg_n_pairs": avg_pairs,
            "n_windows": len(windows), "n_days": len(all_dr),
        }
        logger.info("  %s OOS: Sharpe=%.4f AR=%.4f MDD=%.2f%% avg_pairs=%.0f",
                    label, sharpe, ar, mdd*100, avg_pairs)

    # Print results
    print("\n" + "=" * 100)
    print("WALK-FORWARD OOS RESULTS")
    print("=" * 100)

    base_s = all_results.get("baseline", {}).get("Sharpe", 0.0)

    print(f"\n{'Label':<15} {'Sharpe':<10} {'dSharpe':<10} {'AR':<10} {'Vol':<10} "
          f"{'MDD%':<8} {'Turnover':<10} {'avgPairs':<10} {'nDays':<8}")
    print("-" * 95)
    for label, r in all_results.items():
        ds = r["Sharpe"] - base_s if np.isfinite(r["Sharpe"]) else np.nan
        print(f"{label:<15} {r['Sharpe']:<10.4f} {ds:<+10.4f} {r['AR']:<10.4f} "
              f"{r['Vol']:<10.4f} {r['MDD']*100:<8.2f} {r['Turnover']:<10.2f} "
              f"{r['avg_n_pairs']:<10.0f} {r['n_days']:<8}")

    print(f"\nTrain={TRAIN_WINDOW}d, Test={TEST_WINDOW}d, Step={STEP}d, Purge={PURGE}d")
    print(f"Windows: {len(windows)}, Total OOS days: {all_results.get('baseline',{}).get('n_days',0)}")

    pd.DataFrame(all_results).T.to_csv(output_dir / "walkforward_results.csv")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
