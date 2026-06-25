"""Sector Relative Ensemble Model class.

Implements the PCA-Ensemble model (signal-level ensemble of Raw-PCA, Residual-PCA, and P4 components with options for US residualization and residual space prior variant).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from leadlag.core import signal as signals
from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
    compute_correlation,
)
from leadlag.core.residualize import compute_rolling_ols_betas
from leadlag.data.preprocessor import compute_us_residualized_returns
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.base import BaseModel

logger = logging.getLogger(__name__)

_COMMON_INPUTS_CACHE = {}
_PRODUCTION_SIGNAL_CACHE = {}
_RESIDUAL_SIGNAL_CACHE = {}
_RESIDUAL_PRIOR_CACHE = {}


def compute_jp_target_returns(df_exec: pd.DataFrame, jp_tickers: list[str]) -> np.ndarray:
    """Compute 9:10-to-close returns for JP assets, with Open-to-Close as fallback."""
    jp_oc = df_exec[[f"jp_oc_{tk}" for tk in jp_tickers]].values
    y_jp_target = jp_oc.copy()

    from leadlag.data.cache import load_intraday_cache
    df_5m = load_intraday_cache("5m")
    if df_5m is not None and not df_5m.empty:
        dates_5m = pd.Series(df_5m.index.date).unique()
        r_open_910_dict = {}
        for dt in dates_5m:
            dt_ts = pd.Timestamp(dt)
            day_data = df_5m[df_5m.index.date == dt]

            idx_910 = pd.Timestamp(f"{dt} 09:10:00")
            row_910 = day_data.loc[idx_910] if idx_910 in day_data.index else None

            ticker_returns = {}
            for ticker in jp_tickers:
                p_910 = np.nan
                if row_910 is not None:
                    high = row_910.get(("High", ticker))
                    low = row_910.get(("Low", ticker))
                    close = row_910.get(("Close", ticker))
                    p_910 = (high + low) / 2 if (pd.notna(high) and pd.notna(low)) else close

                p_open_5m = np.nan
                for time_str in ["09:00:00", "09:05:00", "09:10:00"]:
                    idx_time = pd.Timestamp(f"{dt} {time_str}")
                    if idx_time in day_data.index:
                        row_time = day_data.loc[idx_time]
                        op = row_time.get(("Open", ticker))
                        cl = row_time.get(("Close", ticker))
                        val = op if pd.notna(op) else cl
                        if pd.notna(val):
                            p_open_5m = val
                            break

                ret_open_910 = 0.0
                if pd.notna(p_910) and pd.notna(p_open_5m) and p_open_5m > 0:
                    ret_open_910 = float(p_910 / p_open_5m - 1.0)
                ticker_returns[ticker] = ret_open_910
            r_open_910_dict[dt_ts] = ticker_returns

        for idx, date in enumerate(df_exec.index):
            date_ts = pd.Timestamp(date)
            if date_ts in r_open_910_dict:
                ticker_returns = r_open_910_dict[date_ts]
                for t_idx, ticker in enumerate(jp_tickers):
                    ret_oc = jp_oc[idx, t_idx]
                    ret_open_910 = ticker_returns.get(ticker, 0.0)
                    y_jp_target[idx, t_idx] = (1.0 + ret_oc) / (1.0 + ret_open_910) - 1.0
    return y_jp_target


class SectorRelativeEnsembleModel(BaseModel):
    """Sector Relative Ensemble (PCA-Ensemble) Model.

    Ensembles standard Production signal (Raw-PCA), TOPIX-residualized Production signal (Residual-PCA),
    and SPY/TOPIX-residualized signal (P4).
    """

    def __init__(self, config: dict | object):
        """Initialize SectorRelativeEnsembleModel.

        Args:
            config: Dict or object containing configuration options.
        """
        self.config = config
        self.n_u = len(US_TICKERS)
        self.n_j = len(JP_TICKERS)

        # Config resolution
        self.model_name = self._resolve_val("model_name", "sector_relative_ensemble")
        self.k = self._resolve_val("k", 6)
        self.lambda_reg = self._resolve_val("lambda_reg", 0.75)
        self.q = self._resolve_val("q", 0.3)
        self.weight_mode = self._resolve_val("weight_mode", "signal")
        self.ewma_half_life = self._resolve_val("ewma_half_life", 45)
        self.lambda_lw = self._resolve_val("lambda_lw", 0.5)
        self.lw_target = self._resolve_val("lw_target", "equicorrelation")
        self.corr_window = self._resolve_val("corr_window", 60)
        self.include_v4_prior = self._resolve_val("include_v4_prior", True)
        self.gap_open_coef = self._resolve_val("gap_open_coef", 0.70)
        self.topix_beta_coef = self._resolve_val("topix_beta_coef", 0.6)
        self.beta_window = self._resolve_val("beta_window", 60)
        self.vol_adjusted_target = self._resolve_val("vol_adjusted_target", True)

        # Resolve ensemble weights
        self.raw_pca_weight = float(self._resolve_val("raw_pca_weight", 0.5))
        self.residual_pca_weight = float(self._resolve_val("residual_pca_weight", 0.5))
        self.p4_weight = float(self._resolve_val("p4_weight", 0.0))

        # Resolve US residualization config
        self.us_res_enabled = self._resolve_nested("us_residualization.enabled", self.p4_weight > 0.0)
        self.us_res_benchmark = self._resolve_nested("us_residualization.benchmark", "SPY")
        self.us_res_beta_window = int(self._resolve_nested("us_residualization.beta_window", 60))
        self.us_res_beta_shift = int(self._resolve_nested("us_residualization.beta_shift", 1))
        self.us_res_gamma = float(self._resolve_nested("us_residualization.gamma", 0.5))

        # Resolve US Residual Prior config
        self.prior_variant = self._resolve_nested("prior.variant", None)
        self.normalization_method = self._resolve_val("normalization", "zscore")

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_val("slippage_bps", 5.0)
        if isinstance(config, dict):
            if "costs" in config and "slippage_bps_per_side" in config["costs"]:
                self.slippage_bps = float(config["costs"]["slippage_bps_per_side"])

        # Overnight holding parameters
        self.overnight_alpha_long = self._resolve_val("overnight_alpha_long", 0.0)
        self.overnight_alpha_short = self._resolve_val("overnight_alpha_short", 0.0)
        self.buy_interest_annual = self._resolve_val("buy_interest_annual", 0.0)
        self.borrow_fee_annual = self._resolve_val("borrow_fee_annual", 0.0)
        self.reverse_fee_bps = self._resolve_val("reverse_fee_bps", 0.0)

    def _resolve_val(self, key: str, default: any) -> any:
        """Resolve value from config object or dict."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        if isinstance(self.config, dict):
            if key in self.config:
                return self.config[key]
            for section in [
                "model",
                "ensemble",
                "portfolio",
                "costs",
                "residualization",
                "audit",
                "output",
                "prior",
            ]:
                if section in self.config and key in self.config[section]:
                    return self.config[section][key]
            # Translations
            if key == "model_name" and "name" in self.config.get("model", {}):
                return self.config["model"]["name"]
            if key == "k" and "k" in self.config.get("model", {}):
                return self.config["model"]["k"]
            if key == "q" and "long_short_frac" in self.config.get("portfolio", {}):
                return self.config["portfolio"]["long_short_frac"]
        return default

    def _resolve_nested(self, key: str, default: any) -> any:
        """Resolve dotted nested keys or falls back to _resolve_val."""
        parts = key.split(".")
        val = self._resolve_val(parts[-1], None)
        if val is not None:
            return val
        if isinstance(self.config, dict):
            curr = self.config
            for part in parts:
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return default
            return curr
        return default

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Prepare and compute arrays and target matrices common to backtesting and live run."""
        cache_key = (
            len(df_exec),
            self.ewma_half_life,
            self.beta_window,
            self.include_v4_prior,
            self.us_res_enabled,
            self.us_res_gamma,
            self.us_res_beta_window,
        )
        global _COMMON_INPUTS_CACHE
        if cache_key in _COMMON_INPUTS_CACHE:
            cached_val = _COMMON_INPUTS_CACHE[cache_key]
            self.v0_static_obj = cached_val["v0_static"]
            return cached_val.copy()

        sim_dates = df_exec.index

        # Target returns for JP on trade_date (D_t+1)
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)

        # Build all_returns_raw: US columns are us_cc_* (on D_t), JP columns are y_jp_target (on D_t+1)
        us_returns_raw = df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].values
        all_returns_raw = np.column_stack([us_returns_raw, y_jp_target])

        c_full = compute_baseline_correlation(
            all_returns_raw, sim_dates.values, self.ewma_half_life
        )
        v0_static = build_v3_static(self.n_u, self.n_j, self.include_v4_prior)
        self.v0_static_obj = v0_static
        base_vectors = build_base_vectors(self.n_u, self.n_j)
        v1, v2 = base_vectors["v1"], base_vectors["v2"]

        # Parse arrays
        jp_gap = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values
        jp_beta = (
            df_exec[[f"jp_beta_{tk}" for tk in JP_TICKERS]].values
            if any(c.startswith("jp_beta_") for c in df_exec.columns)
            else None
        )
        topix_night = (
            df_exec["topix_night_return"].values
            if "topix_night_return" in df_exec.columns
            else None
        )

        # Build trade date returns
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )

        topix_cc_trade = (
            df_exec["topix_cc_trade"].values
            if "topix_cc_trade" in df_exec.columns
            else df_exec["topix_night_return"].values + df_exec["topix_oc_return"].values
        )

        # Rolling OLS residualization for Residual-PCA (lookahead-safe) using target returns
        betas_jp_p3 = compute_rolling_ols_betas(
            y_jp_target, topix_cc_trade.reshape(-1, 1), self.beta_window
        )
        y_residuals_p3 = y_jp_target - betas_jp_p3[:, :, 0] * topix_cc_trade.reshape(-1, 1)

        # Replace JP columns with residuals for PCA (no shifting needed under new alignment)
        jp_res_returns_p3 = all_returns_raw.copy()
        jp_res_returns_p3[:, self.n_u :] = y_residuals_p3

        c_full_p3 = compute_baseline_correlation(
            jp_res_returns_p3, sim_dates.values, self.ewma_half_life
        )

        out = {
            "all_returns_raw": all_returns_raw,
            "c_full": c_full,
            "c_full_p3": c_full_p3,
            "v0_static": v0_static,
            "v1": v1,
            "v2": v2,
            "jp_gap": jp_gap,
            "jp_beta": jp_beta,
            "topix_night": topix_night,
            "y_jp_oc_df": y_jp_oc_df,
            "jp_res_returns_p3": jp_res_returns_p3,
        }

        # US Residualization P4 logic
        if self.us_res_enabled:
            spy_col = None
            for col in ["spy_cc", "SPY_cc", "SPY", "spy", "r_US_MKT"]:
                if col in df_exec.columns:
                    spy_col = col
                    break
            if spy_col is None:
                raise ValueError("SPY benchmark return column not found in df_exec")
            spy_returns = df_exec[spy_col].values

            us_returns_raw = all_returns_raw[:, : self.n_u]
            r_us_adj = compute_us_residualized_returns(
                us_returns_raw,
                spy_returns,
                beta_window=self.us_res_beta_window,
                gamma=self.us_res_gamma,
            )

            all_returns_p4 = jp_res_returns_p3.copy()
            all_returns_p4[:, : self.n_u] = r_us_adj

            out["all_returns_p4"] = all_returns_p4
            out["r_us_adj"] = r_us_adj
            out["spy_returns"] = spy_returns

        _COMMON_INPUTS_CACHE[cache_key] = out
        return out

    def _prepare_residual_prior(self, df_exec: pd.DataFrame, inputs: dict) -> dict:
        """Precompute the V0_resid and C0_resid matrices for the baseline period."""
        sim_dates = df_exec.index
        n = self.n_u + self.n_j

        # Slice baseline period: 2010-01-01 to 2014-12-31
        baseline_mask = (sim_dates >= pd.to_datetime("2010-01-01")) & (
            sim_dates <= pd.to_datetime("2014-12-31")
        )
        baseline_indices = np.where(baseline_mask)[0]
        if len(baseline_indices) == 0:
            # Fallback if baseline period not found
            baseline_indices = np.arange(min(len(df_exec), 1260))

        all_returns_p4 = inputs["all_returns_p4"]
        all_returns_raw = inputs["all_returns_raw"]

        baseline_returns_p4 = all_returns_p4[baseline_indices]
        baseline_returns_raw = all_returns_raw[baseline_indices]

        global _RESIDUAL_PRIOR_CACHE
        cache_key = (
            self.prior_variant,
            self.ewma_half_life,
            self.n_u,
            self.n_j,
            baseline_returns_p4.shape,
            hash(baseline_returns_p4.tobytes()),
            baseline_returns_raw.shape,
            hash(baseline_returns_raw.tobytes()),
        )
        if cache_key in _RESIDUAL_PRIOR_CACHE:
            cached = _RESIDUAL_PRIOR_CACHE[cache_key]
            return {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in cached.items()
            }

        # Prior vectors
        w3 = np.array([1.0, 0.3, 0.2, 0.8, 0.9, 0.7, -1.0, 0.4, -0.9, -0.8, 1.0, 0.0, 0.6, -0.2, -0.7, -0.9, 0.3, 0.6, 0.9, -0.9, 1.0, 1.0, 0.9, 0.8, -0.3, -1.0, -0.4, 0.7, -0.5, 0.8, 0.6, 0.5], dtype=float)
        w4 = np.array([0.4, 0.0, 0.1, 0.2, 0.7, 0.8, -0.5, -0.4, -0.7, -0.4, 0.6, 0.3, 0.1, 0.6, -0.3, -0.6, 0.2, 0.2, 0.5, -0.2, 1.0, 0.6, 0.8, 1.0, -0.2, -0.8, -0.4, 0.8, -0.7, 0.3, 0.0, -0.9], dtype=float)
        w5 = np.array([0.4, 0.0, 1.0, 0.0, 0.2, 0.0, -0.3, 0.0, -0.8, 0.0, -0.3, 0.0, 0.4, -0.1, -0.3, -0.3, 1.0, -0.1, 0.3, 0.0, -0.2, 0.2, 0.0, 0.0, 0.0, -0.9, -0.1, 0.7, -0.2, -0.2, 0.0, 0.0], dtype=float)
        w6 = np.array([0.8, -0.3, 1.0, 0.3, 0.3, -0.5, -0.2, 0.4, -0.7, -0.2, -0.4, -0.1, 0.5, -0.4, -0.3, -0.4, 1.0, 0.3, 0.7, -0.2, -0.1, 0.6, 0.2, -0.3, -0.3, -0.8, -0.3, 0.8, -0.5, 0.2, 0.1, 0.3], dtype=float)

        def _orthogonalize_and_normalize(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
            result = vector.astype(float, copy=True)
            for base in basis:
                result = result - (result @ base) * base
            norm = np.linalg.norm(result)
            if norm <= 1e-10:
                return np.zeros_like(result)
            return result / norm

        v1_new = np.ones(n) / np.sqrt(n)
        denom = np.sqrt(float(self.n_u * self.n_j * n))
        v2_new = np.zeros(n)
        v2_new[: self.n_u] = self.n_j / denom
        v2_new[self.n_u :] = -self.n_u / denom

        v1_v2_scale = None
        c0_source = "residualized"
        k_expected = 6

        if self.prior_variant == "raw_v1_to_v6":
            v3_new = _orthogonalize_and_normalize(w3, [v1_new, v2_new])
            v4_new = _orthogonalize_and_normalize(w4, [v1_new, v2_new, v3_new])
            v5_new = _orthogonalize_and_normalize(w5, [v1_new, v2_new, v3_new, v4_new])
            v6_new = _orthogonalize_and_normalize(w6, [v1_new, v2_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v2_new, v3_new, v4_new, v5_new, v6_new])
            c0_source = "raw_existing"
            k_expected = 6
        elif self.prior_variant == "resid_v2_removed":
            v3_new = _orthogonalize_and_normalize(w3, [v1_new])
            v4_new = _orthogonalize_and_normalize(w4, [v1_new, v3_new])
            v5_new = _orthogonalize_and_normalize(w5, [v1_new, v3_new, v4_new])
            v6_new = _orthogonalize_and_normalize(w6, [v1_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v3_new, v4_new, v5_new, v6_new])
            k_expected = 5
        elif self.prior_variant == "resid_v1_v2_removed":
            v3_new = w3 / np.linalg.norm(w3)
            v4_new = _orthogonalize_and_normalize(w4, [v3_new])
            v5_new = _orthogonalize_and_normalize(w5, [v3_new, v4_new])
            v6_new = _orthogonalize_and_normalize(w6, [v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v3_new, v4_new, v5_new, v6_new])
            k_expected = 4
        elif self.prior_variant == "resid_v1_v2_scaled_025":
            v3_new = _orthogonalize_and_normalize(w3, [v1_new, v2_new])
            v4_new = _orthogonalize_and_normalize(w4, [v1_new, v2_new, v3_new])
            v5_new = _orthogonalize_and_normalize(w5, [v1_new, v2_new, v3_new, v4_new])
            v6_new = _orthogonalize_and_normalize(w6, [v1_new, v2_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v2_new, v3_new, v4_new, v5_new, v6_new])
            v1_v2_scale = 0.25
            k_expected = 6
        elif self.prior_variant == "resid_v1_v2_scaled_050":
            v3_new = _orthogonalize_and_normalize(w3, [v1_new, v2_new])
            v4_new = _orthogonalize_and_normalize(w4, [v1_new, v2_new, v3_new])
            v5_new = _orthogonalize_and_normalize(w5, [v1_new, v2_new, v3_new, v4_new])
            v6_new = _orthogonalize_and_normalize(w6, [v1_new, v2_new, v3_new, v4_new, v5_new])
            V0_resid = np.column_stack([v1_new, v2_new, v3_new, v4_new, v5_new, v6_new])
            v1_v2_scale = 0.50
            k_expected = 6
        else:
            raise ValueError(f"Unknown prior_variant: {self.prior_variant}")

        # Compute C_full_resid
        if c0_source == "raw_existing":
            _, _, C_full_resid = compute_correlation(baseline_returns_raw, self.ewma_half_life)
        else:
            _, _, C_full_resid = compute_correlation(baseline_returns_p4, self.ewma_half_life)

        # Build C0_resid
        mat = V0_resid.T @ C_full_resid @ V0_resid
        d_vals = np.diag(mat)
        d_vals = np.maximum(d_vals, 1e-10)

        # Record original eigenvalues before scale for diagnostics
        d_vals_orig = d_vals.copy()

        if v1_v2_scale is not None:
            d_vals[0] *= v1_v2_scale
            d_vals[1] *= v1_v2_scale

        d0 = np.diag(d_vals)
        c0_raw = V0_resid @ d0 @ V0_resid.T

        delta = np.diag(c0_raw)
        delta = np.maximum(delta, 1e-10)
        delta_inv_sqrt = np.diag(1.0 / np.sqrt(delta))
        C0_resid = delta_inv_sqrt @ c0_raw @ delta_inv_sqrt
        np.fill_diagonal(C0_resid, 1.0)

        C0_resid = np.nan_to_num(C0_resid, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(C0_resid, 1.0)

        # Diagnostics info
        eigs = np.linalg.eigvalsh(C0_resid)
        cond_num = float(np.max(eigs) / np.maximum(np.min(eigs), 1e-12))

        result_dict = {
            "V0_resid": V0_resid,
            "C0_resid": C0_resid,
            "C_full_resid": C_full_resid,
            "d_vals_orig": d_vals_orig,
            "d_vals_scaled": d_vals,
            "min_eig": float(np.min(eigs)),
            "max_eig": float(np.max(eigs)),
            "cond_num": cond_num,
            "c0_source": c0_source,
            "k_expected": k_expected,
            "v1_new": v1_new,
            "v2_new": v2_new,
        }
        _RESIDUAL_PRIOR_CACHE[cache_key] = result_dict
        return {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in result_dict.items()
        }

    def compute_production_signal(
        self,
        df_exec: pd.DataFrame,
        i: int,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        all_returns: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the Raw-PCA (Production) signal at index i."""
        cache_key = (i, self.lambda_reg)
        global _PRODUCTION_SIGNAL_CACHE
        if cache_key in _PRODUCTION_SIGNAL_CACHE:
            return _PRODUCTION_SIGNAL_CACHE[cache_key].copy()
        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(self.n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_res = signals.compute_signal(
            all_returns=all_returns,
            current_index=i,
            n_u=self.n_u,
            corr_window=self.corr_window,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            k=self.k,
            lambda_reg=self.lambda_reg,
            lambda_lw=self.lambda_lw,
            lw_target=self.lw_target,
            ewma_half_life=self.ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_t1,
            gap_open_coef=self.gap_open_coef,
            topix_beta_coef=self.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=self.vol_adjusted_target,
        )
        sig = np.asarray(sig_res["signal"], dtype=float)
        _PRODUCTION_SIGNAL_CACHE[cache_key] = sig
        return sig

    def compute_residual_signal(
        self,
        df_exec: pd.DataFrame,
        jp_res_returns_p3: np.ndarray,
        i: int,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the Residual-PCA (JP Residual target) or P4 signal at index i."""
        is_p3 = hasattr(self, "v0_static_obj") and id(v0_static) == id(self.v0_static_obj)
        if is_p3:
            cache_key = ("Residual-PCA", i, self.lambda_reg, self.k)
        else:
            cache_key = (
                "P4",
                i,
                getattr(self, "prior_variant", None),
                getattr(self, "us_res_gamma", None),
                self.lambda_reg,
            )
        global _RESIDUAL_SIGNAL_CACHE
        if cache_key in _RESIDUAL_SIGNAL_CACHE:
            return _RESIDUAL_SIGNAL_CACHE[cache_key].copy()
        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(self.n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_res = signals.compute_signal(
            all_returns=jp_res_returns_p3,
            current_index=i,
            n_u=self.n_u,
            corr_window=self.corr_window,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            k=self.k,
            lambda_reg=self.lambda_reg,
            lambda_lw=self.lambda_lw,
            lw_target=self.lw_target,
            ewma_half_life=self.ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_t1,
            gap_open_coef=self.gap_open_coef,
            topix_beta_coef=self.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=self.vol_adjusted_target,
        )
        sig = np.asarray(sig_res["signal"], dtype=float)
        _RESIDUAL_SIGNAL_CACHE[cache_key] = sig
        return sig

    def normalize_signals(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Cross-sectionally normalize the signal values."""
        if method == "identity":
            return sig
        centered = sig - np.median(sig)
        if method == "zscore":
            std = np.std(centered)
            return centered / (std if std > 0 else 1.0)
        elif method == "rank_normalize":
            ranks = pd.Series(sig).rank(pct=True).values
            return (ranks - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def combine_signals(self, z0: np.ndarray, z3: np.ndarray) -> np.ndarray:
        """Combine Raw-PCA and Residual-PCA signals (signal-level 50/50 ensemble)."""
        return 0.5 * z0 + 0.5 * z3

    def build_weights(self, signal: np.ndarray, q: float | None = None) -> np.ndarray:
        """Construct portfolio weights from combined PCA-Ensemble signal."""
        q_val = q if q is not None else self.q
        return signals.build_weights(
            signal=signal,
            q=q_val,
            n_j=self.n_j,
            weight_mode=self.weight_mode,
            enforce_sign=False,
        )

    def predict_signals(self, df_exec: pd.DataFrame) -> dict[str, Any]:
        """Generate raw signals for all rows in df_exec from corr_window index onwards.

        Args:
            df_exec: Execution DataFrame containing historical data.

        Returns:
            Dict of DataFrames/arrays.
        """
        T = len(df_exec)
        sim_dates = df_exec.index

        inputs = self._prepare_common_inputs(df_exec)
        all_returns_raw = inputs["all_returns_raw"]
        c_full = inputs["c_full"]
        c_full_p3 = inputs["c_full_p3"]
        v0_static = inputs["v0_static"]
        v1 = inputs["v1"]
        v2 = inputs["v2"]
        jp_gap = inputs["jp_gap"]
        jp_beta = inputs["jp_beta"]
        topix_night = inputs["topix_night"]
        jp_res_returns_p3 = inputs["jp_res_returns_p3"]

        # Optional Residualized US/JP signal (P4)
        prior_info = None
        if self.us_res_enabled or self.p4_weight > 0.0:
            all_returns_p4 = inputs["all_returns_p4"]
            if self.prior_variant is not None:
                prior_info = self._prepare_residual_prior(df_exec, inputs)
                C0_resid = prior_info["C0_resid"]
                V0_resid = prior_info["V0_resid"]
                k_p4 = prior_info["k_expected"]

        # Setup arrays
        raw_pca_signals = np.zeros((T, self.n_j))
        residual_pca_signals = np.zeros((T, self.n_j))
        p4_signals = np.zeros((T, self.n_j))
        sre_signals = np.zeros((T, self.n_j))

        # Fill first corr_window rows with zeros, or run from corr_window
        start_idx = self.corr_window
        for i in range(start_idx, T):
            # Raw-PCA
            raw_pca_sig = self.compute_production_signal(
                df_exec, i, c_full, v0_static, v1, v2, all_returns_raw, jp_gap, jp_beta, topix_night
            )
            raw_pca_signals[i] = raw_pca_sig

            # Residual-PCA
            residual_pca_sig = self.compute_residual_signal(
                df_exec, jp_res_returns_p3, i, c_full_p3, v0_static, v1, v2, jp_gap, jp_beta, topix_night
            )
            residual_pca_signals[i] = residual_pca_sig

            # Normalization
            z0 = self.normalize_signals(raw_pca_sig, self.normalization_method)
            z3 = self.normalize_signals(residual_pca_sig, self.normalization_method)

            # P4
            if self.us_res_enabled or self.p4_weight > 0.0:
                if self.prior_variant is not None:
                    orig_build_c0 = signals.build_c0_from_v0
                    signals.build_c0_from_v0 = lambda v0, c: C0_resid
                    orig_k = self.k
                    self.k = k_p4
                    try:
                        p4_sig = self.compute_residual_signal(
                            df_exec,
                            all_returns_p4,
                            i,
                            c_full=C0_resid,
                            v0_static=V0_resid,
                            v1=V0_resid[:, 0],
                            v2=V0_resid[:, 1] if V0_resid.shape[1] > 1 else V0_resid[:, 0],
                            jp_gap=jp_gap,
                            jp_beta=jp_beta,
                            topix_night=topix_night,
                        )
                    finally:
                        signals.build_c0_from_v0 = orig_build_c0
                        self.k = orig_k
                else:
                    p4_sig = self.compute_residual_signal(
                        df_exec,
                        all_returns_p4,
                        i,
                        c_full,
                        v0_static,
                        v1,
                        v2,
                        jp_gap,
                        jp_beta,
                        topix_night,
                    )
                p4_signals[i] = p4_sig
                z4 = self.normalize_signals(p4_sig, self.normalization_method)
            else:
                z4 = np.zeros(self.n_j)

            # Ensemble
            s_ens = self.raw_pca_weight * z0 + self.residual_pca_weight * z3 + self.p4_weight * z4
            sre_signals[i] = s_ens

        # Create DataFrames
        raw_pca_df = pd.DataFrame(raw_pca_signals, index=sim_dates, columns=JP_TICKERS)
        residual_pca_df = pd.DataFrame(residual_pca_signals, index=sim_dates, columns=JP_TICKERS)
        p4_df = pd.DataFrame(p4_signals, index=sim_dates, columns=JP_TICKERS)
        sre_df = pd.DataFrame(sre_signals, index=sim_dates, columns=JP_TICKERS)

        sre_normalized_df = pd.DataFrame(index=sim_dates, columns=JP_TICKERS)
        for date in sim_dates:
            idx = df_exec.index.get_loc(date)
            sre_normalized_df.loc[date] = self.normalize_signals(
                sre_signals[idx], self.normalization_method
            )

        out_res = {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "signals": sre_df,
            "normalized_signals": sre_normalized_df,
            "y_jp_oc_df": inputs["y_jp_oc_df"],
        }
        if prior_info is not None:
            out_res["prior_info"] = prior_info

        return out_res
