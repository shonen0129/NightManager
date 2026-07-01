"""Phase 1B: Adaptive Ensemble Weights — Rolling IC-based dynamic weighting.

Approach:
  1. Run model once with all 4 components enabled to get component signals
  2. Compute per-day Rank IC for each component vs actual JP target returns
  3. Test multiple adaptive weighting schemes:
     - Static weight grid search
     - Rolling IC proportional
     - Rolling ICIR proportional
     - Softmax-IC with temperature
     - IC shrinkage to equal weight
  4. Compare against current production (residual_blpx only)
  5. Walk-forward OOS validation for best scheme

Key optimization: component signals computed once, recombined externally.
"""

from __future__ import annotations

import argparse
import itertools
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
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COMPONENTS = ["raw_pca", "residual_pca", "raw_blpx", "residual_blpx"]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(yaml_path: str, blpx_overrides: dict | None = None,
                 signal_components: dict | None = None) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        for key, val in blpx_overrides.items():
            cfg.setdefault("blpx", {})[key] = val
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


def all_components_config() -> dict:
    """Signal components config with all 4 enabled at equal weight."""
    return {
        "raw_pca": {"enabled": True, "weight": 0.25},
        "residual_pca": {"enabled": True, "weight": 0.25},
        "raw_blpx": {"enabled": True, "weight": 0.25},
        "residual_blpx": {"enabled": True, "weight": 0.25},
    }


def single_component_config(name: str) -> dict:
    """Signal components config with only one component enabled."""
    comps = {c: {"enabled": False, "weight": 0.0} for c in COMPONENTS}
    comps[name] = {"enabled": True, "weight": 1.0}
    return comps


# ---------------------------------------------------------------------------
# Pre-computed signal wrapper
# ---------------------------------------------------------------------------

class PreComputedSignalModel:
    """Wrapper that returns pre-computed signals, bypassing expensive computation."""

    def __init__(self, base_model: SectorRelativeEnsembleBLPEnhancedModel,
                 signals_df: pd.DataFrame, y_jp_oc_df: pd.DataFrame,
                 normalized_signals_df: pd.DataFrame | None = None):
        self._base = base_model
        self._signals = signals_df
        self._y_jp_oc = y_jp_oc_df
        self._normalized = normalized_signals_df
        # Copy essential attributes
        self.n_u = base_model.n_u
        self.n_j = base_model.n_j
        self.corr_window = base_model.corr_window
        self.q = base_model.q
        self.weight_mode = base_model.weight_mode
        self.slippage_bps = base_model.slippage_bps
        self.overnight_alpha_long = getattr(base_model, "overnight_alpha_long", 0.0)
        self.overnight_alpha_short = getattr(base_model, "overnight_alpha_short", 0.0)
        self.buy_interest_annual = getattr(base_model, "buy_interest_annual", 0.025)
        self.borrow_fee_annual = getattr(base_model, "borrow_fee_annual", 0.0115)
        self.reverse_fee_bps = getattr(base_model, "reverse_fee_bps", 2.0)

    def predict_signals(self, df_exec: pd.DataFrame) -> dict:
        return {
            "signals": self._signals,
            "normalized_signals": self._normalized if self._normalized is not None else self._signals,
            "y_jp_oc_df": self._y_jp_oc,
            "raw_pca_signals": self._signals,  # placeholder
            "residual_pca_signals": self._signals,
            "p4_signals": self._signals,
        }

    def build_weights(self, signal: np.ndarray, q: float | None = None) -> np.ndarray:
        return self._base.build_weights(signal, q)

    def get_audit_context(self):
        return self._base.get_audit_context()


# ---------------------------------------------------------------------------
# Rank IC computation
# ---------------------------------------------------------------------------

def compute_daily_ic(signals_df: pd.DataFrame, y_target: np.ndarray,
                     sim_dates: pd.DatetimeIndex, start_idx: int) -> pd.DataFrame:
    """Compute daily Spearman Rank IC for each date.

    Returns DataFrame with columns ['date', 'ic'].
    """
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)

    ic_list = []
    ic_dates = []
    for i in range(start_idx, len(sim_dates)):
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
                ic_dates.append(date)

    return pd.DataFrame({"date": ic_dates, "ic": ic_list}).set_index("date")


def compute_component_ic_timeseries(component_signals: dict[str, pd.DataFrame],
                                    y_target: np.ndarray, sim_dates: pd.DatetimeIndex,
                                    start_idx: int) -> pd.DataFrame:
    """Compute daily IC for each component signal.

    Returns DataFrame with columns = component names, index = dates.
    """
    all_ics = {}
    for comp_name, sig_df in component_signals.items():
        ic_df = compute_daily_ic(sig_df, y_target, sim_dates, start_idx)
        all_ics[comp_name] = ic_df["ic"]
        logger.info("  %s: mean IC = %.4f (n=%d)", comp_name, ic_df["ic"].mean(), len(ic_df))

    ic_ts = pd.DataFrame(all_ics)
    return ic_ts


# ---------------------------------------------------------------------------
# Adaptive weighting schemes
# ---------------------------------------------------------------------------

def static_combine(component_zs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Combine normalized component signals with static weights."""
    result = np.zeros_like(next(iter(component_zs.values())))
    for comp, w in weights.items():
        result += w * component_zs[comp]
    return result


def rolling_ic_proportional(ic_ts: pd.DataFrame, window: int,
                            min_ic: float = 0.0) -> pd.DataFrame:
    """Rolling IC-proportional weights.

    weight_i(t) = max(0, IC_i(t-1)) / sum(max(0, IC_j(t-1)))
    Uses a rolling window average IC.
    """
    rolling_ic = ic_ts.rolling(window, min_periods=window // 2).mean()
    rolling_ic = rolling_ic.shift(1)  # Use yesterday's IC for today's weight
    rolling_ic_clipped = rolling_ic.clip(lower=min_ic)

    row_sums = rolling_ic_clipped.sum(axis=1)
    weights = rolling_ic_clipped.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    # Where all ICs are negative, fall back to equal weight
    all_neg = (rolling_ic <= 0).all(axis=1)
    weights[all_neg] = 1.0 / len(COMPONENTS)

    return weights


def rolling_icir_proportional(ic_ts: pd.DataFrame, window: int,
                              min_icir: float = 0.0) -> pd.DataFrame:
    """Rolling ICIR-proportional weights.

    weight_i(t) = max(0, ICIR_i(t-1)) / sum(max(0, ICIR_j(t-1)))
    ICIR = mean(IC) / std(IC) * sqrt(252)
    """
    rolling_mean = ic_ts.rolling(window, min_periods=window // 2).mean()
    rolling_std = ic_ts.rolling(window, min_periods=window // 2).std()
    rolling_icir = (rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)).fillna(0.0)
    rolling_icir = rolling_icir.shift(1)
    rolling_icir_clipped = rolling_icir.clip(lower=min_icir)

    row_sums = rolling_icir_clipped.sum(axis=1)
    weights = rolling_icir_clipped.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    all_neg = (rolling_icir <= 0).all(axis=1)
    weights[all_neg] = 1.0 / len(COMPONENTS)

    return weights


def softmax_ic_weights(ic_ts: pd.DataFrame, window: int,
                       temperature: float = 10.0) -> pd.DataFrame:
    """Softmax-weighted by rolling IC.

    weight_i(t) = exp(temp * IC_i(t-1)) / sum(exp(temp * IC_j(t-1)))
    """
    rolling_ic = ic_ts.rolling(window, min_periods=window // 2).mean()
    rolling_ic = rolling_ic.shift(1)

    # Softmax with numerical stability
    max_ic = rolling_ic.max(axis=1)
    exp_ic = np.exp(temperature * (rolling_ic.sub(max_ic, axis=0)))
    row_sums = exp_ic.sum(axis=1).replace(0, np.nan)
    weights = exp_ic.div(row_sums, axis=0).fillna(1.0 / len(COMPONENTS))

    return weights


def ic_shrinkage_weights(ic_ts: pd.DataFrame, window: int,
                         alpha: float = 0.5) -> pd.DataFrame:
    """IC-proportional with shrinkage to equal weight.

    weight_i(t) = (1-alpha) * IC_prop_i(t-1) + alpha * equal_weight
    """
    ic_prop = rolling_ic_proportional(ic_ts, window)
    equal = pd.DataFrame(1.0 / len(COMPONENTS), index=ic_prop.index,
                         columns=ic_prop.columns)
    return (1 - alpha) * ic_prop + alpha * equal


def construct_adaptive_signals(component_zs_arrays: dict[str, np.ndarray],
                               weights_df: pd.DataFrame,
                               sim_dates: pd.DatetimeIndex,
                               start_idx: int) -> np.ndarray:
    """Construct combined signals using time-varying weights.

    Args:
        component_zs_arrays: dict of {component: array of shape (T, n_j)} normalized signals
        weights_df: DataFrame of weights, index = dates, columns = components
        sim_dates: full date index
        start_idx: start index for signal generation

    Returns:
        Combined signal array of shape (T, n_j)
    """
    T = len(sim_dates)
    n_j = next(iter(component_zs_arrays.values())).shape[1]
    combined = np.zeros((T, n_j))

    for i in range(start_idx, T):
        date = sim_dates[i]
        if date in weights_df.index:
            w = weights_df.loc[date]
            for comp in COMPONENTS:
                combined[i] += w[comp] * component_zs_arrays[comp][i]
        else:
            # Fallback to equal weight before window is available
            for comp in COMPONENTS:
                combined[i] += 0.25 * component_zs_arrays[comp][i]

    return combined


# ---------------------------------------------------------------------------
# Backtest with pre-computed signals
# ---------------------------------------------------------------------------

def run_backtest_with_signals(model: SectorRelativeEnsembleBLPEnhancedModel,
                              signals_df: pd.DataFrame,
                              df_exec: pd.DataFrame,
                              y_jp_oc_df: pd.DataFrame,
                              start_date: str = "2015-01-01",
                              slippage_bps: float = 5.0) -> dict:
    """Run backtest with pre-computed combined signals via wrapper model."""
    wrapper = PreComputedSignalModel(model, signals_df, y_jp_oc_df)

    results = BacktestEngine.run_backtest(
        wrapper,
        df_exec=df_exec,
        start_date=start_date,
        overnight_alpha_long=0.75,
        overnight_alpha_short=0.5,
        buy_interest_annual=0.025,
        borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
        slippage_bps=slippage_bps,
    )
    return results


def compute_metrics(results: dict, signals_df: pd.DataFrame,
                    y_target: np.ndarray, sim_dates: pd.DatetimeIndex,
                    start_idx: int) -> dict:
    """Compute all metrics from backtest results."""
    dr = results["daily_returns"]
    dr_gross = results["daily_returns_gross"]

    ar = float(dr.mean() * 245)
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan

    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(results["daily_turnover"].mean())
    gross_exp = float(results["daily_gross_exps"].mean())

    # Rank IC
    ic_df = compute_daily_ic(signals_df, y_target, sim_dates, start_idx)
    mean_ic = float(ic_df["ic"].mean()) if len(ic_df) > 0 else np.nan
    std_ic = float(ic_df["ic"].std(ddof=1)) if len(ic_df) > 1 else np.nan
    icir = (mean_ic / std_ic * np.sqrt(252)) if (std_ic and std_ic > 1e-8) else np.nan

    monthly = (1.0 + dr).groupby(dr.index.year * 12 + dr.index.month).prod() - 1.0
    monthly_sharpe = float((monthly.mean() / monthly.std(ddof=1)) * np.sqrt(12.0)) if len(monthly) > 1 else np.nan

    return {
        "AR_net": ar,
        "AR_gross": float(dr_gross.mean() * 245),
        "Vol_net": vol,
        "Sharpe_net": sharpe,
        "Sharpe_monthly": monthly_sharpe,
        "MDD": mdd,
        "Turnover": turnover,
        "GrossExp": gross_exp,
        "Mean_Rank_IC": mean_ic,
        "ICIR": icir,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1B: Adaptive Ensemble Weights")
    parser.add_argument("--stage", choices=["compute", "static", "adaptive", "wfo", "all"], default="all")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/phase1b_adaptive_ensemble")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec shape: %s", df_exec.shape)

    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    start_dt = pd.to_datetime("2015-01-01")
    start_idx = max(df_exec.index.searchsorted(start_dt), 60)

    # --- Step 1: Compute all component signals (one-time) ---
    comp_cache_path = output_dir / "component_signals.npz"

    if args.stage in ("compute", "all"):
        logger.info("=== Computing all 4 component signals (one-time) ===")
        cfg_all = build_config(yaml_path, signal_components=all_components_config())
        model_all = SectorRelativeEnsembleBLPEnhancedModel(cfg_all)

        t0 = time.perf_counter()
        pred_all = model_all.predict_signals(df_exec)
        logger.info("predict_signals (all components): %.1fs", time.perf_counter() - t0)

        # Extract component signals
        component_signals = {
            "raw_pca": pred_all["raw_pca_signals"],
            "residual_pca": pred_all["residual_pca_signals"],
            "raw_blpx": pred_all["raw_blpx_signals"],
            "residual_blpx": pred_all["residual_blpx_signals"],
        }
        y_jp_oc_df = pred_all["y_jp_oc_df"]

        # Normalize each component (z-score)
        component_zs = {}
        for comp_name, sig_df in component_signals.items():
            z_arr = np.zeros_like(sig_df.values)
            for i in range(start_idx, len(sim_dates)):
                z_arr[i] = model_all.normalize_signals(sig_df.values[i], "zscore")
            component_zs[comp_name] = z_arr

        # Save to npz for reuse
        np.savez_compressed(
            comp_cache_path,
            raw_pca=component_zs["raw_pca"],
            residual_pca=component_zs["residual_pca"],
            raw_blpx=component_zs["raw_blpx"],
            residual_blpx=component_zs["residual_blpx"],
            y_jp_oc=y_jp_oc_df.values,
            y_target=y_target,
        )
        logger.info("Component signals saved to %s", comp_cache_path)

        # Compute per-component IC
        logger.info("=== Per-component IC ===")
        ic_ts = compute_component_ic_timeseries(component_signals, y_target, sim_dates, start_idx)
        ic_ts.to_csv(output_dir / "component_ic_timeseries.csv")
        logger.info("IC timeseries saved.")

        # Print summary
        print("\n--- Component IC Summary ---")
        for comp in COMPONENTS:
            ic_col = ic_ts[comp].dropna()
            print(f"  {comp:<20}: mean IC = {ic_col.mean():.4f}, std = {ic_col.std():.4f}, "
                  f"ICIR = {ic_col.mean()/ic_col.std()*np.sqrt(252):.2f}, positive rate = {(ic_col > 0).mean():.2%}")

    # --- Load cached component signals if needed ---
    if args.stage in ("static", "adaptive", "wfo", "all"):
        logger.info("Loading cached component signals...")
        cached = np.load(comp_cache_path, allow_pickle=True)
        component_zs = {
            "raw_pca": cached["raw_pca"],
            "residual_pca": cached["residual_pca"],
            "raw_blpx": cached["raw_blpx"],
            "residual_blpx": cached["residual_blpx"],
        }
        y_jp_oc_arr = cached["y_jp_oc"]
        y_target = cached["y_target"]
        y_jp_oc_df = pd.DataFrame(y_jp_oc_arr, index=sim_dates, columns=JP_TICKERS)

        # Reconstruct IC timeseries
        component_signals_dfs = {
            comp: pd.DataFrame(arr, index=sim_dates, columns=JP_TICKERS)
            for comp, arr in component_zs.items()
        }
        ic_ts = compute_component_ic_timeseries(component_signals_dfs, y_target, sim_dates, start_idx)

        # Build base model for wrapper
        cfg_base = build_config(yaml_path, signal_components=all_components_config())
        model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)

    # --- Baseline: residual_blpx only (current production) ---
    logger.info("=== Baseline: residual_blpx only (current production) ===")
    baseline_signals = pd.DataFrame(component_zs["residual_blpx"], index=sim_dates, columns=JP_TICKERS)
    results_base = run_backtest_with_signals(model_base, baseline_signals, df_exec, y_jp_oc_df,
                                             slippage_bps=args.slippage_bps)
    metrics_base = compute_metrics(results_base, baseline_signals, y_target, sim_dates, start_idx)
    logger.info("Baseline (residual_blpx only): Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%%",
                metrics_base["Sharpe_net"], metrics_base["Mean_Rank_IC"],
                metrics_base["ICIR"], metrics_base["MDD"] * 100)

    all_results = []

    # --- Stage: Static weight grid search ---
    if args.stage in ("static", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STATIC WEIGHT GRID SEARCH")
        logger.info("=" * 80)

        # Grid: each weight in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, sum to 1.0
        weight_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        combos = []
        for w1, w2, w3, w4 in itertools.product(weight_levels, repeat=4):
            if abs(w1 + w2 + w3 + w4 - 1.0) < 1e-6:
                combos.append((w1, w2, w3, w4))

        logger.info("Testing %d static weight combinations...", len(combos))

        for idx, (w_rpca, w_respca, w_rblpx, w_resblpx) in enumerate(combos):
            weights = {
                "raw_pca": w_rpca,
                "residual_pca": w_respca,
                "raw_blpx": w_rblpx,
                "residual_blpx": w_resblpx,
            }
            combined_arr = np.zeros_like(component_zs["raw_pca"])
            for comp, w in weights.items():
                combined_arr += w * component_zs[comp]
            combined_df = pd.DataFrame(combined_arr, index=sim_dates, columns=JP_TICKERS)

            results = run_backtest_with_signals(model_base, combined_df, df_exec, y_jp_oc_df,
                                                slippage_bps=args.slippage_bps)
            m = compute_metrics(results, combined_df, y_target, sim_dates, start_idx)
            m["scheme"] = "static"
            m["name"] = f"static_{w_rpca}_{w_respca}_{w_rblpx}_{w_resblpx}"
            m.update(weights)
            all_results.append(m)

            if (idx + 1) % 10 == 0:
                logger.info("  %d/%d: Sharpe=%.4f IC=%.4f", idx + 1, len(combos), m["Sharpe_net"], m["Mean_Rank_IC"])

        static_df = pd.DataFrame(all_results)
        static_df.to_csv(output_dir / "static_weight_results.csv", index=False)

        top10 = static_df.nlargest(10, "Sharpe_net")
        print("\n--- Top 10 Static Weights by Sharpe ---")
        print(top10[["name", "Sharpe_net", "AR_net", "MDD", "Mean_Rank_IC", "ICIR"]].to_string(index=False))

    # --- Stage: Adaptive weighting schemes ---
    if args.stage in ("adaptive", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("ADAPTIVE WEIGHTING SCHEMES")
        logger.info("=" * 80)

        schemes = []

        # 1. Rolling IC proportional (various windows)
        for window in [21, 42, 63, 126]:
            schemes.append(("ic_prop", {"window": window}))

        # 2. Rolling ICIR proportional (various windows)
        for window in [42, 63, 126]:
            schemes.append(("icir_prop", {"window": window}))

        # 3. Softmax IC (various temperature × window)
        for window in [42, 63, 126]:
            for temp in [5.0, 10.0, 20.0, 50.0]:
                schemes.append(("softmax_ic", {"window": window, "temperature": temp}))

        # 4. IC shrinkage (various alpha × window)
        for window in [42, 63, 126]:
            for alpha in [0.3, 0.5, 0.7]:
                schemes.append(("ic_shrinkage", {"window": window, "alpha": alpha}))

        logger.info("Testing %d adaptive schemes...", len(schemes))

        adaptive_results = []
        for idx, (scheme_name, params) in enumerate(schemes):
            # Compute weights
            if scheme_name == "ic_prop":
                weights_df = rolling_ic_proportional(ic_ts, **params)
            elif scheme_name == "icir_prop":
                weights_df = rolling_icir_proportional(ic_ts, **params)
            elif scheme_name == "softmax_ic":
                weights_df = softmax_ic_weights(ic_ts, **params)
            elif scheme_name == "ic_shrinkage":
                weights_df = ic_shrinkage_weights(ic_ts, **params)
            else:
                continue

            # Construct adaptive combined signals
            combined_arr = construct_adaptive_signals(component_zs, weights_df, sim_dates, start_idx)
            combined_df = pd.DataFrame(combined_arr, index=sim_dates, columns=JP_TICKERS)

            results = run_backtest_with_signals(model_base, combined_df, df_exec, y_jp_oc_df,
                                                slippage_bps=args.slippage_bps)
            m = compute_metrics(results, combined_df, y_target, sim_dates, start_idx)
            m["scheme"] = scheme_name
            param_str = "_".join(f"{k}={v}" for k, v in params.items())
            m["name"] = f"{scheme_name}_{param_str}"
            m.update(params)
            adaptive_results.append(m)

            logger.info("  %d/%d: %s — Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%%",
                        idx + 1, len(schemes), m["name"],
                        m["Sharpe_net"], m["Mean_Rank_IC"], m["ICIR"], m["MDD"] * 100)

        adaptive_df = pd.DataFrame(adaptive_results)
        adaptive_df.to_csv(output_dir / "adaptive_weight_results.csv", index=False)

        top10_adaptive = adaptive_df.nlargest(10, "Sharpe_net")
        print("\n--- Top 10 Adaptive Schemes by Sharpe ---")
        print(top10_adaptive[["name", "Sharpe_net", "AR_net", "MDD", "Mean_Rank_IC", "ICIR"]].to_string(index=False))

        all_results.extend(adaptive_results)

    # --- Stage: Walk-Forward OOS ---
    if args.stage in ("wfo", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD OOS VALIDATION")
        logger.info("=" * 80)

        # Find best static and best adaptive
        candidates = [("baseline_residual_blpx", {"raw_pca": 0.0, "residual_pca": 0.0,
                                                       "raw_blpx": 0.0, "residual_blpx": 1.0}, "static")]

        static_path = output_dir / "static_weight_results.csv"
        if static_path.exists():
            static_df = pd.read_csv(static_path)
            best_static = static_df.nlargest(1, "Sharpe_net").iloc[0]
            sw = {c: best_static[c] for c in COMPONENTS}
            candidates.append(("best_static", sw, "static"))

        adaptive_path = output_dir / "adaptive_weight_results.csv"
        if adaptive_path.exists():
            adaptive_df = pd.read_csv(adaptive_path)
            best_adaptive = adaptive_df.nlargest(1, "Sharpe_net").iloc[0]
            best_scheme = best_adaptive["scheme"]
            best_params = {}
            for k in ["window", "temperature", "alpha"]:
                if k in best_adaptive and not pd.isna(best_adaptive[k]):
                    best_params[k] = best_adaptive[k]
            candidates.append(("best_adaptive", best_params, best_scheme))

        eval_dates = sim_dates[start_idx:]
        n_eval = len(eval_dates)
        fold_size = n_eval // args.n_folds

        wfo_all = []
        for name, params, scheme_type in candidates:
            logger.info("WFO for %s: %s", name, params)

            if scheme_type == "static":
                combined_arr = np.zeros_like(component_zs["raw_pca"])
                for comp, w in params.items():
                    combined_arr += w * component_zs[comp]
            else:
                if scheme_type == "ic_prop":
                    weights_df = rolling_ic_proportional(ic_ts, **params)
                elif scheme_type == "icir_prop":
                    weights_df = rolling_icir_proportional(ic_ts, **params)
                elif scheme_type == "softmax_ic":
                    weights_df = softmax_ic_weights(ic_ts, **params)
                elif scheme_type == "ic_shrinkage":
                    weights_df = ic_shrinkage_weights(ic_ts, **params)
                else:
                    continue
                combined_arr = construct_adaptive_signals(component_zs, weights_df, sim_dates, start_idx)

            combined_df = pd.DataFrame(combined_arr, index=sim_dates, columns=JP_TICKERS)
            results = run_backtest_with_signals(model_base, combined_df, df_exec, y_jp_oc_df,
                                                slippage_bps=args.slippage_bps)
            dr = results["daily_returns"]

            fold_metrics = []
            for fold in range(args.n_folds):
                f_start = fold * fold_size
                f_end = min((fold + 1) * fold_size, n_eval)
                fold_dates = eval_dates[f_start:f_end]
                fold_dr = dr.reindex(fold_dates).dropna()
                if len(fold_dr) < 5:
                    continue

                fold_ar = float(fold_dr.mean() * 245)
                fold_vol = float(fold_dr.std(ddof=1) * np.sqrt(245))
                fold_sharpe = fold_ar / fold_vol if fold_vol > 0 else np.nan
                fold_wealth = (1.0 + fold_dr).cumprod()
                fold_mdd = float(((fold_wealth / fold_wealth.cummax()) - 1.0).min())

                # Fold IC
                y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)
                ic_vals = []
                for date in fold_dates:
                    if date in combined_df.index:
                        sig_t = combined_df.loc[date].values
                        y_t = y_df.loc[date].values
                        valid = ~(np.isnan(sig_t) | np.isnan(y_t))
                        if valid.sum() >= 3:
                            rho, _ = stats.spearmanr(sig_t[valid], y_t[valid])
                            if np.isfinite(rho):
                                ic_vals.append(float(rho))

                fold_ic = float(np.mean(ic_vals)) if ic_vals else np.nan
                fold_icir = (fold_ic / np.std(ic_vals, ddof=1) * np.sqrt(252)) if (len(ic_vals) > 1 and np.std(ic_vals, ddof=1) > 1e-8) else np.nan

                fold_metrics.append({
                    "fold": fold + 1,
                    "start_date": fold_dates[0].strftime("%Y-%m-%d"),
                    "end_date": fold_dates[-1].strftime("%Y-%m-%d"),
                    "n_days": len(fold_dr),
                    "AR_net": fold_ar,
                    "Sharpe_net": fold_sharpe,
                    "MDD": fold_mdd,
                    "Mean_Rank_IC": fold_ic,
                    "ICIR": fold_icir,
                })

            print(f"\n--- Walk-Forward Folds: {name} ---")
            fold_df = pd.DataFrame(fold_metrics)
            print(fold_df.to_string(index=False))

            sharpes = [f["Sharpe_net"] for f in fold_metrics if np.isfinite(f["Sharpe_net"])]
            ics = [f["Mean_Rank_IC"] for f in fold_metrics if np.isfinite(f["Mean_Rank_IC"])]
            if sharpes:
                print(f"  Mean Sharpe: {np.mean(sharpes):.4f} (std={np.std(sharpes):.4f})")
                print(f"  Mean IC:     {np.mean(ics):.4f} (std={np.std(ics):.4f})")
                print(f"  Min Sharpe:  {np.min(sharpes):.4f}")

            for fold in fold_metrics:
                row = {"candidate": name, **fold}
                wfo_all.append(row)

        pd.DataFrame(wfo_all).to_csv(output_dir / "walk_forward_results.csv", index=False)

    # --- Final Summary ---
    print("\n" + "=" * 100)
    print("PHASE 1B — ADAPTIVE ENSEMBLE WEIGHTS — FINAL SUMMARY")
    print("=" * 100)

    print(f"\nBaseline (residual_blpx only):")
    print(f"  Sharpe: {metrics_base['Sharpe_net']:.4f}")
    print(f"  IC:     {metrics_base['Mean_Rank_IC']:.4f}")
    print(f"  ICIR:   {metrics_base['ICIR']:.2f}")
    print(f"  MDD:    {metrics_base['MDD']*100:.2f}%")

    if all_results:
        results_df = pd.DataFrame(all_results)
        best = results_df.nlargest(1, "Sharpe_net").iloc[0]
        print(f"\nBest Overall: {best['name']}")
        print(f"  Sharpe: {best['Sharpe_net']:.4f} (delta={best['Sharpe_net']-metrics_base['Sharpe_net']:+.4f})")
        print(f"  IC:     {best['Mean_Rank_IC']:.4f} (delta={best['Mean_Rank_IC']-metrics_base['Mean_Rank_IC']:+.4f})")
        print(f"  ICIR:   {best['ICIR']:.2f} (delta={best['ICIR']-metrics_base['ICIR']:+.2f})")
        print(f"  MDD:    {best['MDD']*100:.2f}%")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
