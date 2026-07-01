"""Phase 3A: Walk-Forward Integrated Validation.

Validates Phase 2 winners across yearly time windows to check consistency.

Models compared:
  1. prod_baseline: Current production (residual_blpx only)
  2. phase2_baseline: Phase 2 base (raw_blpx 0.8 + raw_pca 0.2)
  3. phase2a_blend: Multi-horizon blend 80/10/10 (Phase 2A winner)
  4. phase2d_rankrev: Rank reversal w=0.05 (Phase 2D winner)
  5. phase2_combined: Phase 2A + Phase 2D combined

Walk-forward: Yearly windows from 2015 to 2025.
Per-window: Sharpe, IC, ICIR, MDD, AR, Turnover.
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BLPX_PARAMS = {
    "alpha_xx": 0.20, "alpha_yy": 0.50, "alpha_yx": 0.15,
    "lambda_pca": 0.10, "lambda_sector": 0.60, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 504,
    "blp_ewma_halflife": 120, "sector_eta": 0.5, "sector_gamma": 4.0,
}

PHASE2_WEIGHTS = {
    "raw_pca": {"enabled": True, "weight": 0.2},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": True, "weight": 0.8},
    "residual_blpx": {"enabled": False, "weight": 0.0},
}

PROD_WEIGHTS = {
    "raw_pca": {"enabled": False, "weight": 0.0},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": False, "weight": 0.0},
    "residual_blpx": {"enabled": True, "weight": 1.0},
}

MIX_BLEND = {"w1": 0.8, "w3": 0.1, "w5": 0.1}
RANK_REV_WEIGHT = 0.05


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        cfg.setdefault("blpx", {}).update(blpx_overrides)
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


def compute_cumulative_returns(df_exec, horizon):
    df_mod = df_exec.copy()
    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    jp_gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    for col in us_cols:
        df_mod[col] = df_exec[col].rolling(horizon).sum()
    for col in jp_oc_cols:
        df_mod[col] = df_exec[col].rolling(horizon).sum()
    for col in jp_gap_cols:
        df_mod[col] = df_exec[col].rolling(horizon).sum()
    for col in ["topix_night_return", "topix_oc_return", "topix_cc_trade"]:
        if col in df_exec.columns:
            df_mod[col] = df_exec[col].rolling(horizon).sum()
    return df_mod


def compute_cs_rank_reversal(df_exec, col_prefix="jp_oc"):
    cols = [f"{col_prefix}_{tk}" for tk in JP_TICKERS]
    df = df_exec[cols].copy()
    df.columns = JP_TICKERS
    ranks = df.shift(1).rank(axis=1)
    rank_change = ranks.diff()
    return -rank_change


def cross_sectional_zscore(df):
    centered = df.sub(df.median(axis=1), axis=0)
    std = centered.std(axis=1)
    std_safe = std.where(std > 1e-8, 1.0)
    return centered.div(std_safe, axis=0)


def compute_rank_ic(signals_df, y_target, sim_dates, start_idx, end_idx=None):
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)
    ic_list = []
    end = end_idx if end_idx is not None else len(sim_dates)
    for i in range(start_idx, end):
        date = sim_dates[i]
        if date not in signals_df.index:
            continue
        sig_t = signals_df.loc[date].values
        y_t = y_df.loc[date].values
        valid = ~(np.isnan(sig_t) | np.isnan(y_t))
        if valid.sum() >= 3:
            rho, _ = stats.spearmanr(sig_t[valid], y_t[valid])
            if np.isfinite(rho):
                ic_list.append(float(rho))
    if not ic_list:
        return np.nan, np.nan
    ic_arr = np.array(ic_list)
    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr, ddof=1))
    icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic > 1e-8 else np.nan
    return mean_ic, icir


def run_backtest_full(model, df_exec, y_target, slippage_bps=5.0,
                      alpha_long=0.75, alpha_short=0.5):
    results = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-01",
        overnight_alpha_long=alpha_long, overnight_alpha_short=alpha_short,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0, slippage_bps=slippage_bps,
    )
    return results


def compute_window_metrics(daily_returns, signals_df, y_target, sim_dates,
                           start_idx, end_idx):
    dr = daily_returns.iloc[start_idx:end_idx]
    if len(dr) == 0 or dr.std(ddof=1) < 1e-8:
        return {"Sharpe": np.nan, "AR": np.nan, "Vol": np.nan, "MDD": np.nan}
    ar = float(dr.mean() * 245)
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    mean_ic, icir = compute_rank_ic(signals_df, y_target, sim_dates, start_idx, end_idx)
    return {"Sharpe": sharpe, "AR": ar, "Vol": vol, "MDD": mdd,
            "IC": mean_ic, "ICIR": icir}


def get_yearly_windows(sim_dates, start_year=2015, end_year=2025):
    windows = []
    for year in range(start_year, end_year + 1):
        start = pd.to_datetime(f"{year}-01-01")
        end = pd.to_datetime(f"{year}-12-31")
        mask = (sim_dates >= start) & (sim_dates <= end)
        if mask.sum() > 0:
            windows.append({"year": year, "start": start, "end": end})
    return windows


class MultiHorizonBlendModel:
    def __init__(self, configs, horizons, weights, df_exec_raw):
        self.configs = configs
        self.horizons = horizons
        self.weights = weights
        self.df_exec_raw = df_exec_raw
        self.n_j = len(JP_TICKERS)
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"
        self._signals_cache = None

    def _prepare(self):
        if self._signals_cache is not None:
            return self._signals_cache
        all_signals = {}
        for h, cfg in zip(self.horizons, self.configs):
            if h == 1:
                df_mod = self.df_exec_raw
            else:
                df_mod = compute_cumulative_returns(self.df_exec_raw, h)
            _BLP_CORR_CACHE.clear()
            _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            pred = model.predict_signals(df_mod)
            all_signals[h] = pred["signals"]
        self._signals_cache = all_signals
        return all_signals

    def predict_signals(self, df_exec):
        all_signals = self._prepare()
        T = len(df_exec)
        sim_dates = df_exec.index
        combined = np.zeros((T, self.n_j))
        for h, w in zip(self.horizons, self.weights):
            sig_h = all_signals[h].reindex(sim_dates).fillna(0.0).values
            z_h = np.zeros_like(sig_h)
            for i in range(T):
                z_h[i] = self._normalize(sig_h[i])
            combined += w * z_h
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "raw_pca_signals": empty, "residual_pca_signals": empty,
            "p4_signals": empty,
            "signals": pd.DataFrame(combined, index=sim_dates, columns=JP_TICKERS),
            "normalized_signals": pd.DataFrame(combined, index=sim_dates, columns=JP_TICKERS),
            "y_jp_oc_df": y_jp_oc_df,
        }

    def _normalize(self, sig):
        centered = sig - np.median(sig)
        std = np.std(centered)
        return centered / (std if std > 1e-8 else 1.0)

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        q_val = q if q is not None else self.q
        return build_weights(signal=signal, q=q_val, n_j=self.n_j,
                             weight_mode=self.weight_mode, enforce_sign=False)


class CSFeatureBlendModel:
    def __init__(self, blpx_signals, feature_signals, weights, n_j):
        self.blpx_signals = blpx_signals
        self.feature_signals = feature_signals
        self.weights = weights
        self.n_j = n_j
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"

    def predict_signals(self, df_exec):
        T = len(df_exec)
        sim_dates = df_exec.index
        blpx = self.blpx_signals.reindex(sim_dates).fillna(0.0)
        z_blpx = cross_sectional_zscore(blpx)
        combined = z_blpx.copy()
        for fname, w in self.weights.items():
            feat = self.feature_signals[fname].reindex(sim_dates).fillna(0.0)
            z_feat = cross_sectional_zscore(feat)
            combined = combined + w * z_feat
        combined = combined.fillna(0.0)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "raw_pca_signals": empty, "residual_pca_signals": empty,
            "p4_signals": empty,
            "signals": combined,
            "normalized_signals": combined,
            "y_jp_oc_df": y_jp_oc_df,
        }

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        q_val = q if q is not None else self.q
        return build_weights(signal=signal, q=q_val, n_j=self.n_j,
                             weight_mode=self.weight_mode, enforce_sign=False)


class CombinedModel:
    """Phase 2A (multi-horizon) + Phase 2D (rank reversal) combined."""

    def __init__(self, configs, horizons, mh_weights, rank_rev_weight,
                 df_exec_raw, n_j):
        self.configs = configs
        self.horizons = horizons
        self.mh_weights = mh_weights
        self.rank_rev_weight = rank_rev_weight
        self.df_exec_raw = df_exec_raw
        self.n_j = n_j
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"
        self._mh_signals = None
        self._rank_rev = None

    def _prepare(self):
        if self._mh_signals is not None:
            return
        all_signals = {}
        for h, cfg in zip(self.horizons, self.configs):
            if h == 1:
                df_mod = self.df_exec_raw
            else:
                df_mod = compute_cumulative_returns(self.df_exec_raw, h)
            _BLP_CORR_CACHE.clear()
            _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            pred = model.predict_signals(df_mod)
            all_signals[h] = pred["signals"]

        T = len(self.df_exec_raw)
        sim_dates = self.df_exec_raw.index
        combined = np.zeros((T, self.n_j))
        for h, w in zip(self.horizons, self.mh_weights):
            sig_h = all_signals[h].reindex(sim_dates).fillna(0.0).values
            z_h = np.zeros_like(sig_h)
            for i in range(T):
                z_h[i] = self._normalize(sig_h[i])
            combined += w * z_h
        self._mh_signals = pd.DataFrame(combined, index=sim_dates, columns=JP_TICKERS)
        self._rank_rev = compute_cs_rank_reversal(self.df_exec_raw)

    def predict_signals(self, df_exec):
        self._prepare()
        T = len(df_exec)
        sim_dates = df_exec.index

        z_mh = cross_sectional_zscore(self._mh_signals.reindex(sim_dates).fillna(0.0))
        z_rr = cross_sectional_zscore(self._rank_rev.reindex(sim_dates).fillna(0.0))
        combined = z_mh + self.rank_rev_weight * z_rr
        combined = combined.fillna(0.0)

        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "raw_pca_signals": empty, "residual_pca_signals": empty,
            "p4_signals": empty,
            "signals": combined,
            "normalized_signals": combined,
            "y_jp_oc_df": y_jp_oc_df,
        }

    def _normalize(self, sig):
        centered = sig - np.median(sig)
        std = np.std(centered)
        return centered / (std if std > 1e-8 else 1.0)

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        q_val = q if q is not None else self.q
        return build_weights(signal=signal, q=q_val, n_j=self.n_j,
                             weight_mode=self.weight_mode, enforce_sign=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 3A: Walk-Forward Integrated Validation")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/phase3a_walkforward")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index

    start_idx_global = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    yearly_windows = get_yearly_windows(sim_dates)

    # Build configs
    cfg_phase2 = build_config(yaml_path, blpx_overrides=BLPX_PARAMS,
                              signal_components=PHASE2_WEIGHTS)
    cfg_prod = build_config(yaml_path, blpx_overrides=BLPX_PARAMS,
                            signal_components=PROD_WEIGHTS)

    models_to_test = {}

    # 1. Production baseline (residual_blpx only)
    logger.info("Preparing production baseline (residual_blpx)...")
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    models_to_test["prod_baseline"] = {
        "model": SectorRelativeEnsembleBLPEnhancedModel(cfg_prod),
        "label": "Production (residual_blpx)",
    }

    # 2. Phase 2 baseline (raw_blpx + raw_pca)
    logger.info("Preparing Phase 2 baseline (raw_blpx + raw_pca)...")
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    models_to_test["phase2_baseline"] = {
        "model": SectorRelativeEnsembleBLPEnhancedModel(cfg_phase2),
        "label": "Phase 2 baseline (raw_blpx 0.8 + raw_pca 0.2)",
    }

    # 3. Phase 2A: Multi-horizon blend
    logger.info("Preparing Phase 2A (multi-horizon blend 80/10/10)...")
    configs_mh = [cfg_phase2, cfg_phase2, cfg_phase2]
    horizons = [1, 3, 5]
    mh_weights = [MIX_BLEND["w1"], MIX_BLEND["w3"], MIX_BLEND["w5"]]
    models_to_test["phase2a_blend"] = {
        "model": MultiHorizonBlendModel(configs_mh, horizons, mh_weights, df_exec),
        "label": "Phase 2A blend (80/10/10)",
    }

    # 4. Phase 2D: Rank reversal
    logger.info("Preparing Phase 2D (rank reversal w=0.05)...")
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    base_model = SectorRelativeEnsembleBLPEnhancedModel(cfg_phase2)
    base_pred = base_model.predict_signals(df_exec)
    blpx_signals = base_pred["signals"]
    rank_rev = compute_cs_rank_reversal(df_exec)
    models_to_test["phase2d_rankrev"] = {
        "model": CSFeatureBlendModel(blpx_signals, {"rank_reversal": rank_rev},
                                     {"rank_reversal": RANK_REV_WEIGHT}, len(JP_TICKERS)),
        "label": "Phase 2D rank reversal (w=0.05)",
    }

    # 5. Combined: Phase 2A + Phase 2D
    logger.info("Preparing Combined (Phase 2A + Phase 2D)...")
    models_to_test["phase2_combined"] = {
        "model": CombinedModel(configs_mh, horizons, mh_weights, RANK_REV_WEIGHT,
                               df_exec, len(JP_TICKERS)),
        "label": "Combined (2A blend + 2D rank reversal)",
    }

    # Run backtests and collect per-window metrics
    all_window_results = []
    full_results = {}

    for model_key, model_info in models_to_test.items():
        label = model_info["label"]
        model = model_info["model"]
        logger.info("=== Backtest: %s ===", label)
        t0 = time.perf_counter()
        results = run_backtest_full(model, df_exec, y_target, slippage_bps=args.slippage_bps)
        elapsed = time.perf_counter() - t0

        dr = results["daily_returns"]
        ar = float(dr.mean() * 245)
        vol = float(dr.std(ddof=1) * np.sqrt(245))
        sharpe = ar / vol if vol > 0 else np.nan
        wealth = (1.0 + dr).cumprod()
        mdd = float(((wealth / wealth.cummax()) - 1.0).min())
        turnover = float(results["daily_turnover"].mean())
        mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx_global)

        full_results[model_key] = {
            "label": label, "Sharpe": sharpe, "AR": ar, "Vol": vol,
            "MDD": mdd, "IC": mean_ic, "ICIR": icir, "Turnover": turnover,
            "elapsed_s": elapsed,
        }
        logger.info("  Full: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                    sharpe, mean_ic, icir, mdd * 100, elapsed)

        # Per-window metrics
        for w in yearly_windows:
            w_start = sim_dates.searchsorted(w["start"])
            w_end = min(sim_dates.searchsorted(w["end"]) + 1, len(sim_dates))
            if w_start >= w_end or w_start < start_idx_global:
                continue
            w_metrics = compute_window_metrics(
                dr, results["signals"], y_target, sim_dates,
                max(w_start, start_idx_global), w_end
            )
            all_window_results.append({
                "model": model_key, "label": label, "year": w["year"],
                **w_metrics,
            })
            if np.isfinite(w_metrics["Sharpe"]):
                logger.info("  %d: Sharpe=%.4f IC=%.4f MDD=%.2f%%",
                            w["year"], w_metrics["Sharpe"],
                            w_metrics.get("IC", np.nan),
                            w_metrics["MDD"] * 100)

    # Save results
    window_df = pd.DataFrame(all_window_results)
    window_df.to_csv(output_dir / "window_results.csv", index=False)

    full_df = pd.DataFrame(full_results).T
    full_df.to_csv(output_dir / "full_results.csv")

    # Print summary
    print("\n" + "=" * 110)
    print("PHASE 3A — WALK-FORWARD INTEGRATED VALIDATION — RESULTS")
    print("=" * 110)

    print("\n--- Full Period (2015-2025) ---")
    print(f"{'Model':<45} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Turnover':<10}")
    print("-" * 91)
    for key, r in full_results.items():
        print(f"{r['label']:<45} {r['Sharpe']:<10.4f} {r['IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {r['Turnover']:<10.4f}")

    print("\n--- Yearly Sharpe by Model ---")
    pivot = window_df.pivot_table(values="Sharpe", index="year", columns="model", aggfunc="first")
    print(pivot.to_string(float_format="%.4f"))

    print("\n--- Yearly IC by Model ---")
    pivot_ic = window_df.pivot_table(values="IC", index="year", columns="model", aggfunc="first")
    print(pivot_ic.to_string(float_format="%.4f"))

    # Consistency analysis
    print("\n--- Consistency Analysis ---")
    baseline_key = "phase2_baseline"
    for cmp_key in ["phase2a_blend", "phase2d_rankrev", "phase2_combined"]:
        if cmp_key not in pivot.columns or baseline_key not in pivot.columns:
            continue
        diff = pivot[cmp_key] - pivot[baseline_key]
        wins = (diff > 0).sum()
        total = diff.notna().sum()
        print(f"  {cmp_key} vs {baseline_key}: {wins}/{total} windows positive, "
              f"mean delta={diff.mean():+.4f}, median={diff.median():+.4f}")

    # Best model selection
    print("\n--- Model Ranking (by full-period Sharpe) ---")
    ranked = sorted(full_results.items(), key=lambda x: x[1]["Sharpe"], reverse=True)
    for rank, (key, r) in enumerate(ranked, 1):
        print(f"  {rank}. {r['label']}: Sharpe={r['Sharpe']:.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
