"""LeadLagStrategy: high-level strategy interface.

Provides ``LeadLagStrategy`` (full backtest) and
``PrecomputedLeadLagStrategy`` (fast inference from cached eigen-data).
Both delegate signal computation and portfolio construction to the
modular ``domain.signals`` / ``domain.portfolio`` layers.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from config import STRATEGY_DEFAULTS, N_US_ASSETS, N_JP_ASSETS, N_TOTAL_ASSETS
from domain.signals import lead_lag as signals
from domain.models.types import StrategyConfig

logger = logging.getLogger(__name__)


class LeadLagStrategy:
    """Compatibility wrapper for the new modular signal/portfolio architecture.

    This class maintains the same interface as the original LeadLagStrategy
    but delegates to the new domain layer modules internally.
    """

    def __init__(
        self,
        df_exec,
        K=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
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
        gap_noise_std=STRATEGY_DEFAULTS.get("gap_noise_std", 0.0),
        gap_noise_seed=STRATEGY_DEFAULTS.get("gap_noise_seed", 42),
        dispersion_metric=STRATEGY_DEFAULTS.get(
            "dispersion_metric", "long_short_mean_gap"
        ),
        gamma=STRATEGY_DEFAULTS.get("gamma", 0.5),
        slippage_bps=STRATEGY_DEFAULTS.get("slippage_bps", 5.0),
        vol_adjusted_target=STRATEGY_DEFAULTS.get("vol_adjusted_target", True),
    ):
        self.df_exec = df_exec
        self.config = StrategyConfig(
            k=K,
            lambda_reg=lambda_reg,
            q=q,
            weight_mode=weight_mode,
            dispersion_filter=dispersion_filter,
            dispersion_metric=dispersion_metric,
            v3_mode=v3_mode,
            ewma_half_life=ewma_half_life,
            lambda_lw=lambda_lw,
            lw_target=lw_target,
            corr_window=corr_window,
            include_v4_prior=include_v4_prior,
            signal_mode=signal_mode,
            gap_open_coef=gap_open_coef,
            topix_beta_coef=topix_beta_coef,
            beta_window=beta_window,
            gamma=gamma,
            slippage_bps=float(slippage_bps),
            vol_adjusted_target=vol_adjusted_target,
        )
        self.gap_noise_std = float(gap_noise_std)
        self.gap_noise_seed = int(gap_noise_seed)

        # Column discovery
        self.us_cols = [c for c in df_exec.columns if c.startswith("us_cc_")]
        self.jp_cc_cols = [c for c in df_exec.columns if c.startswith("jp_cc_")]
        self.jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
        self.jp_gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
        self.jp_beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]
        self.topix_night_col = (
            "topix_night_return" if "topix_night_return" in df_exec.columns else None
        )
        self.all_cc_cols = self.us_cols + self.jp_cc_cols

        self.N_U = N_US_ASSETS
        self.N_J = N_JP_ASSETS
        self.N = N_TOTAL_ASSETS

        # Pre-compute static components
        self.C_full = self._compute_C_full()
        self.V_0 = signals.build_v3_static(
            self.N_U,
            self.N_J,
            self.config.include_v4_prior,
        )
        self.C_0 = signals.build_c0_from_v0(self.V_0, self.C_full)
        base_vectors = signals.build_base_vectors(self.N_U, self.N_J)
        self.v1 = base_vectors["v1"]
        self.v2 = base_vectors["v2"]

    def _compute_C_full(self):
        """Compute baseline correlation matrix from 2010-2014 data."""
        df_base = self.df_exec[
            (self.df_exec.index >= "2010-01-01") & (self.df_exec.index <= "2014-12-31")
        ]
        returns = df_base[self.all_cc_cols].values
        if returns.shape[0] == 0:
            raise ValueError("No rows found for C_full baseline period (2010-2014)")
        return signals.compute_correlation(
            returns,
            self.config.ewma_half_life,
        )[2]

    def run_backtest(self, start_date="2015-01-01"):
        """Run backtest and return results DataFrame."""
        from backtest.runner import run_backtest_with_config

        return run_backtest_with_config(self.df_exec, self.config, start_date)

    def generate_trade_decision(
        self,
        trade_date=None,
        start_date="2015-01-01",
        jp_gap_override=None,
    ):
        """Generate single-day trade decision."""
        all_cc_cols = [
            c
            for c in self.df_exec.columns
            if c.startswith("us_cc_") or c.startswith("jp_cc_")
        ]
        all_returns = self.df_exec[all_cc_cols].values
        date_index = self.df_exec.index.values

        cfg = self.config

        # Find index
        if trade_date is None:
            i = len(self.df_exec) - 1
            t_trade = self.df_exec.index[i]
        else:
            t_trade = pd.to_datetime(trade_date)
            try:
                i = self.df_exec.index.get_loc(t_trade)
            except KeyError:
                raise ValueError(f"trade_date not found in df_exec index: {trade_date}")

        # Compute dispersion history
        dispersion_history = []
        gap_cols = [c for c in self.df_exec.columns if c.startswith("jp_gap_")]
        for hist_i in range(max(0, i - 60), i):
            gap_hist = None
            if cfg.signal_mode == "gap_residual" and len(gap_cols) == self.N_J:
                gap_hist = np.nan_to_num(
                    self.df_exec.iloc[hist_i][gap_cols].values,
                    nan=0.0,
                    copy=True,
                ).astype(float, copy=False)
            betas_hist = None
            topix_night_hist = None
            if self.jp_beta_cols and len(self.jp_beta_cols) == self.N_J:
                betas_hist = np.asarray(
                    self.df_exec.iloc[hist_i][self.jp_beta_cols].values,
                    dtype=float,
                )
            if self.topix_night_col is not None:
                topix_night_hist = float(
                    self.df_exec.iloc[hist_i][self.topix_night_col]
                )
            sig = signals.compute_signal(
                all_returns,
                hist_i,
                self.N_U,
                cfg.corr_window,
                self.C_full,
                self.V_0,
                self.v1,
                self.v2,
                cfg.k,
                cfg.lambda_reg,
                cfg.lambda_lw,
                cfg.lw_target,
                cfg.ewma_half_life,
                v3_dynamic=(cfg.v3_mode == "dynamic"),
                gap_override=gap_hist,
                gap_open_coef=cfg.gap_open_coef,
                topix_beta_coef=cfg.topix_beta_coef,
                betas_t=betas_hist,
                topix_night_t=topix_night_hist,
                vol_adjusted_target=cfg.vol_adjusted_target,
            )
            disp = signals.compute_dispersion_indicator(
                sig["signal"],
                cfg.q,
                self.N_J,
                cfg.dispersion_metric,
            )
            dispersion_history.append(disp)

        # Compute current signal
        gap_arr = (
            np.asarray(jp_gap_override, dtype=float)
            if jp_gap_override is not None
            else None
        )
        if (
            gap_arr is None
            and cfg.signal_mode == "gap_residual"
            and len(gap_cols) == self.N_J
        ):
            gap_arr = np.nan_to_num(
                self.df_exec.iloc[i][gap_cols].values,
                nan=0.0,
                copy=True,
            ).astype(float, copy=False)
        betas_t = None
        topix_night_t = None
        if self.jp_beta_cols and len(self.jp_beta_cols) == self.N_J:
            betas_t = np.asarray(
                self.df_exec.iloc[i][self.jp_beta_cols].values,
                dtype=float,
            )
        if self.topix_night_col is not None:
            topix_night_t = float(self.df_exec.iloc[i][self.topix_night_col])
        sig_result = signals.compute_signal(
            all_returns,
            i,
            self.N_U,
            cfg.corr_window,
            self.C_full,
            self.V_0,
            self.v1,
            self.v2,
            cfg.k,
            cfg.lambda_reg,
            cfg.lambda_lw,
            cfg.lw_target,
            cfg.ewma_half_life,
            v3_dynamic=(cfg.v3_mode == "dynamic"),
            gap_override=gap_arr,
            gap_open_coef=cfg.gap_open_coef,
            topix_beta_coef=cfg.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=cfg.vol_adjusted_target,
        )

        # Build decision
        if cfg.signal_mode == "gap_tolerant":
            jp_close_sig_cols = [
                c for c in self.df_exec.columns if c.startswith("jp_close_sig_")
            ]
            jp_open_trade_cols = [
                c for c in self.df_exec.columns if c.startswith("jp_open_trade_")
            ]
            jp_close_sig = (
                self.df_exec[jp_close_sig_cols].values if jp_close_sig_cols else None
            )
            jp_open_trade = (
                self.df_exec[jp_open_trade_cols].values if jp_open_trade_cols else None
            )

            if jp_close_sig is not None and jp_open_trade is not None:
                base_weights = signals.build_weights(
                    sig_result["signal"],
                    cfg.q,
                    self.N_J,
                    cfg.weight_mode,
                )
                jp_close_t = np.nan_to_num(jp_close_sig[i], nan=1.0, copy=True)
                jp_open_t1 = np.nan_to_num(jp_open_trade[i], nan=1.0, copy=True)
                weights, _, _, _ = signals.apply_gap_tolerant_filter(
                    sig_result["signal"],
                    sig_result["sigma_s"],
                    base_weights,
                    jp_close_t,
                    jp_open_t1,
                    cfg.gamma,
                    cfg.q,
                    self.N_J,
                    cfg.weight_mode,
                )
            else:
                weights = signals.build_weights(
                    sig_result["signal"],
                    cfg.q,
                    self.N_J,
                    cfg.weight_mode,
                )
        else:
            weights = signals.build_weights(
                sig_result["signal"],
                cfg.q,
                self.N_J,
                cfg.weight_mode,
            )

        dispersion_ind = signals.compute_dispersion_indicator(
            sig_result["signal"],
            cfg.q,
            self.N_J,
            cfg.dispersion_metric,
        )
        scale = signals.dispersion_scale(
            dispersion_ind,
            dispersion_history,
            cfg.dispersion_filter,
        )
        scaled_weights = weights * scale

        action = np.where(
            scaled_weights > 1e-12,
            "BUY",
            np.where(scaled_weights < -1e-12, "SELL", "HOLD"),
        )

        jp_tickers = [c.replace("jp_oc_", "") for c in self.jp_oc_cols]
        return {
            "trade_date": t_trade,
            "tickers": jp_tickers,
            "signal": sig_result["signal"],
            "raw_weight": weights,
            "scale": float(scale),
            "weight": scaled_weights,
            "action": action,
            "sigma_s": float(sig_result["sigma_s"]),
            "dispersion_indicator": float(dispersion_ind),
            "dispersion_metric": cfg.dispersion_metric,
        }

    def save_precomputed_cache(self, cache_path: str) -> None:
        """Save precomputed strategy components to disk."""
        cache_data = {
            "C_full": self.C_full,
            "v3_mode": self.config.v3_mode,
        }

        if self.config.v3_mode == "static":
            cache_data["V_0"] = self.V_0
            cache_data["C_0"] = self.C_0

        cache_data["all_cc"] = self.df_exec[self.all_cc_cols].values
        cache_data["us_cc"] = self.df_exec[self.us_cols].values
        cache_data["jp_oc"] = self.df_exec[self.jp_oc_cols].values

        if self.jp_gap_cols and len(self.jp_gap_cols) == self.N_J:
            cache_data["jp_gap"] = self.df_exec[self.jp_gap_cols].values

        if self.jp_beta_cols and len(self.jp_beta_cols) == self.N_J:
            cache_data["jp_beta"] = self.df_exec[self.jp_beta_cols].values

        if self.topix_night_col is not None:
            cache_data["topix_night"] = self.df_exec[self.topix_night_col].values

        cache_data["trade_dates"] = self.df_exec.index.values
        cache_data["ewma_half_life"] = (
            float(self.config.ewma_half_life) if self.config.ewma_half_life else -1.0
        )
        cache_data["lambda_reg"] = self.config.lambda_reg
        cache_data["lambda_lw"] = self.config.lambda_lw
        cache_data["lw_target"] = self.config.lw_target
        cache_data["corr_window"] = self.config.corr_window
        cache_data["N_U"] = self.N_U
        cache_data["N_J"] = self.N_J
        cache_data["N"] = self.N
        cache_data["v1"] = self.v1
        cache_data["v2"] = self.v2
        cache_data["K"] = self.config.k
        cache_data["q"] = self.config.q
        cache_data["signal_mode"] = self.config.signal_mode
        cache_data["gap_open_coef"] = self.config.gap_open_coef
        cache_data["topix_beta_coef"] = self.config.topix_beta_coef
        cache_data["beta_window"] = self.config.beta_window
        cache_data["dispersion_metric"] = self.config.dispersion_metric
        cache_data["dispersion_filter"] = self.config.dispersion_filter
        cache_data["weight_mode"] = self.config.weight_mode
        cache_data["include_v4_prior"] = self.config.include_v4_prior
        cache_data["gamma"] = self.config.gamma
        cache_data["vol_adjusted_target"] = self.config.vol_adjusted_target

        os.makedirs(
            os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
            exist_ok=True,
        )
        np.savez_compressed(cache_path, **cache_data)
        logger.info(f"Precomputed cache saved to {cache_path}")

    @classmethod
    def load_precomputed_strategy(
        cls,
        cache_path: str,
        gap_override: Optional[np.ndarray] = None,
    ):
        """Load precomputed strategy from cache."""
        npz_data = np.load(cache_path, allow_pickle=True)
        cache_data = {key: npz_data[key] for key in npz_data.files}
        return PrecomputedLeadLagStrategy(
            cache_data=cache_data,
            gap_override=gap_override,
        )


class PrecomputedLeadLagStrategy:
    """Lightweight strategy using precomputed cache for fast decision-making."""

    def __init__(
        self,
        cache_data: dict,
        gap_override: Optional[np.ndarray] = None,
        ewma_half_life: Optional[float] = None,
        lambda_reg: Optional[float] = None,
        lambda_lw: Optional[float] = None,
        lw_target: Optional[str] = None,
        corr_window: Optional[int] = None,
        K: Optional[int] = None,
        q: Optional[float] = None,
        weight_mode: Optional[str] = None,
        dispersion_filter: Optional[bool] = None,
        dispersion_metric: Optional[str] = None,
        signal_mode: Optional[str] = None,
        gap_open_coef: Optional[float] = None,
        topix_beta_coef: Optional[float] = None,
        beta_window: Optional[int] = None,
        gamma: Optional[float] = None,
        vol_adjusted_target: Optional[bool] = None,
    ):
        def _resolve_value(key: str, override, default):
            if override is not None:
                return override
            if key in cache_data:
                raw = np.asarray(cache_data[key])
                if raw.shape == ():
                    return raw.item()
                return raw
            return default

        self.C_full = cache_data["C_full"]
        self.v3_mode = str(cache_data.get("v3_mode", "static"))

        if self.v3_mode != "static":
            raise ValueError(
                "PrecomputedLeadLagStrategy currently supports only " "v3_mode='static'"
            )
        if "V_0" not in cache_data or "C_0" not in cache_data:
            raise ValueError(
                "Precomputed cache is missing V_0/C_0. Rebuild cache first."
            )
        self.V_0 = cache_data["V_0"]
        self.C_0 = cache_data["C_0"]

        self.all_cc = cache_data["all_cc"]
        self.us_cc = cache_data["us_cc"]
        self.jp_oc = cache_data["jp_oc"]
        self.jp_gap = cache_data["jp_gap"] if "jp_gap" in cache_data else None
        self.jp_beta = cache_data["jp_beta"] if "jp_beta" in cache_data else None
        self.topix_night = (
            cache_data["topix_night"] if "topix_night" in cache_data else None
        )
        self.trade_dates = pd.DatetimeIndex(cache_data.get("trade_dates", []))

        ewma_value = _resolve_value(
            "ewma_half_life",
            ewma_half_life,
            STRATEGY_DEFAULTS["ewma_half_life"],
        )
        ewma_value = float(ewma_value)
        self.ewma_half_life = None if ewma_value < 0 else ewma_value

        self.lambda_reg = float(
            _resolve_value(
                "lambda_reg",
                lambda_reg,
                STRATEGY_DEFAULTS["lambda_reg"],
            )
        )
        self.lambda_lw = float(
            _resolve_value(
                "lambda_lw",
                lambda_lw,
                STRATEGY_DEFAULTS["lambda_lw"],
            )
        )
        self.lw_target = str(
            _resolve_value(
                "lw_target",
                lw_target,
                STRATEGY_DEFAULTS["lw_target"],
            )
        )
        self.corr_window = int(
            _resolve_value(
                "corr_window",
                corr_window,
                STRATEGY_DEFAULTS["corr_window"],
            )
        )
        self.N_U = int(cache_data.get("N_U", N_US_ASSETS))
        self.N_J = int(cache_data.get("N_J", N_JP_ASSETS))
        self.N = int(cache_data.get("N", N_TOTAL_ASSETS))
        self.v1 = cache_data["v1"]
        self.v2 = cache_data["v2"]
        self.K = int(_resolve_value("K", K, STRATEGY_DEFAULTS["K"]))
        self.q = float(_resolve_value("q", q, STRATEGY_DEFAULTS["q"]))
        self.weight_mode = str(
            _resolve_value(
                "weight_mode",
                weight_mode,
                STRATEGY_DEFAULTS["weight_mode"],
            )
        )
        self.dispersion_filter = bool(
            _resolve_value(
                "dispersion_filter",
                dispersion_filter,
                STRATEGY_DEFAULTS["dispersion_filter"],
            )
        )
        self.dispersion_metric = str(
            _resolve_value(
                "dispersion_metric",
                dispersion_metric,
                STRATEGY_DEFAULTS["dispersion_metric"],
            )
        )
        self.signal_mode = str(
            _resolve_value(
                "signal_mode",
                signal_mode,
                STRATEGY_DEFAULTS["signal_mode"],
            )
        )
        self.gamma = float(
            _resolve_value(
                "gamma",
                gamma,
                STRATEGY_DEFAULTS.get("gamma", 0.5),
            )
        )
        self.gap_open_coef = float(
            _resolve_value(
                "gap_open_coef",
                gap_open_coef,
                STRATEGY_DEFAULTS["gap_open_coef"],
            )
        )
        self.topix_beta_coef = float(
            _resolve_value(
                "topix_beta_coef",
                topix_beta_coef,
                STRATEGY_DEFAULTS.get("topix_beta_coef", 1.20),
            )
        )
        self.beta_window = int(
            _resolve_value(
                "beta_window",
                beta_window,
                STRATEGY_DEFAULTS.get("beta_window", 60),
            )
        )
        self.vol_adjusted_target = bool(
            _resolve_value(
                "vol_adjusted_target",
                vol_adjusted_target,
                STRATEGY_DEFAULTS.get("vol_adjusted_target", True),
            )
        )
        self.gap_override = gap_override
        self._signal_cache: dict[int, dict[str, np.ndarray | float]] = {}

    @classmethod
    def load_precomputed_strategy(
        cls,
        cache_path: str,
        gap_override: Optional[np.ndarray] = None,
    ) -> "PrecomputedLeadLagStrategy":
        """Load precomputed strategy from a compressed numpy cache."""
        npz_data = np.load(cache_path, allow_pickle=True)
        cache_data = {key: npz_data[key] for key in npz_data.files}
        return cls(cache_data=cache_data, gap_override=gap_override)

    def _resolve_trade_index(self, trade_date) -> tuple[int, pd.Timestamp]:
        """Resolve the model row index from trade_date for cache-based inference."""
        total_rows = len(self.all_cc)
        if total_rows == 0:
            raise ValueError("Precomputed cache has no rows")

        last_index = total_rows - 1

        if trade_date is None:
            if len(self.trade_dates) > last_index:
                resolved = pd.Timestamp(self.trade_dates[last_index]).normalize()
            else:
                resolved = pd.Timestamp.now().normalize()
            return last_index, resolved

        requested = pd.to_datetime(trade_date).normalize()
        if len(self.trade_dates) == 0:
            return last_index, requested

        normalized_dates = pd.DatetimeIndex(self.trade_dates).normalize()
        matches = np.where(normalized_dates == requested)[0]
        if len(matches) > 0:
            return int(matches[-1]), requested

        if requested >= normalized_dates.max():
            return last_index, requested
        if requested < normalized_dates.min():
            raise ValueError(
                "trade_date is older than the precomputed cache range: "
                f"{requested.date()} < {normalized_dates.min().date()}"
            )

        raise ValueError(
            "trade_date is not present in precomputed cache. "
            "Rebuild cache or use a supported trade_date."
        )

    def _compute_signal_at_index(
        self,
        current_index: int,
        jp_gap_override: np.ndarray | None = None,
        us_returns_override: np.ndarray | None = None,
        topix_night_override: float | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Compute one-step signal at a specific index using cache arrays."""
        if current_index <= 0 or current_index >= len(self.all_cc):
            raise ValueError(
                "current_index must be within [1, T-1], "
                f"got {current_index} for T={len(self.all_cc)}"
            )

        use_cache = (
            jp_gap_override is None
            and us_returns_override is None
            and topix_night_override is None
        )
        if use_cache and current_index in self._signal_cache:
            return self._signal_cache[current_index]

        all_returns = self.all_cc
        if us_returns_override is not None:
            us_vec = np.asarray(us_returns_override, dtype=float).reshape(-1)
            if us_vec.shape[0] != self.N_U:
                raise ValueError(
                    "us_returns_override length must be "
                    f"{self.N_U}, got {us_vec.shape[0]}"
                )
            if not np.all(np.isfinite(us_vec)):
                raise ValueError("us_returns_override contains non-finite values")

            all_returns = self.all_cc.copy()
            all_returns[current_index, : self.N_U] = us_vec

        gap_arr = None
        if self.signal_mode == "gap_residual":
            if jp_gap_override is not None:
                gap_arr = np.asarray(jp_gap_override, dtype=float).reshape(-1)
                if gap_arr.shape[0] != self.N_J:
                    raise ValueError(
                        "jp_gap_override length must be "
                        f"{self.N_J}, got {gap_arr.shape[0]}"
                    )
                if not np.all(np.isfinite(gap_arr)):
                    raise ValueError("jp_gap_override contains non-finite values")
            elif self.jp_gap is not None:
                gap_arr = np.nan_to_num(
                    self.jp_gap[current_index],
                    nan=0.0,
                    copy=True,
                ).astype(float, copy=False)

        betas_t = None
        if self.jp_beta is not None:
            betas_t = np.asarray(self.jp_beta[current_index], dtype=float)

        topix_night_t = None
        if self.topix_night is not None:
            topix_night_t = float(self.topix_night[current_index])
        if topix_night_override is not None:
            topix_night_t = float(topix_night_override)

        result = signals.compute_signal(
            all_returns,
            current_index,
            self.N_U,
            self.corr_window,
            self.C_full,
            self.V_0,
            self.v1,
            self.v2,
            self.K,
            self.lambda_reg,
            self.lambda_lw,
            self.lw_target,
            self.ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_arr,
            gap_open_coef=self.gap_open_coef,
            topix_beta_coef=self.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=self.vol_adjusted_target,
        )

        if use_cache:
            self._signal_cache[current_index] = result

        return result

    def _build_dispersion_history(self, current_index: int) -> list[float]:
        """Build rolling dispersion history aligned with standard decision mode."""
        if current_index <= 1:
            return []

        history: list[float] = []
        start = max(1, current_index - 60)
        for hist_i in range(start, current_index):
            sig_hist = self._compute_signal_at_index(hist_i)
            indicator = signals.compute_dispersion_indicator(
                np.asarray(sig_hist["signal"], dtype=float),
                self.q,
                self.N_J,
                self.dispersion_metric,
            )
            history.append(indicator)
        return history

    def generate_trade_decision(
        self,
        trade_date=None,
        start_date="2015-01-01",
        jp_gap_override=None,
        us_returns_override: np.ndarray | None = None,
        topix_night_override: float | None = None,
    ):
        """Generate trade decision from precomputed cache."""
        _ = start_date
        i, resolved_trade_date = self._resolve_trade_index(trade_date)
        effective_gap_override = (
            jp_gap_override if jp_gap_override is not None else self.gap_override
        )
        sig_result = self._compute_signal_at_index(
            i,
            jp_gap_override=effective_gap_override,
            us_returns_override=us_returns_override,
            topix_night_override=topix_night_override,
        )
        signal = np.asarray(sig_result["signal"], dtype=float)
        sigma_s = float(sig_result["sigma_s"])

        # Build weights
        weights = signals.build_weights(signal, self.q, self.N_J, self.weight_mode)
        dispersion_ind = signals.compute_dispersion_indicator(
            signal,
            self.q,
            self.N_J,
            self.dispersion_metric,
        )
        dispersion_history = self._build_dispersion_history(i)
        scale = signals.dispersion_scale(
            dispersion_ind,
            dispersion_history,
            self.dispersion_filter,
        )
        scaled_weights = weights * scale

        action = np.where(
            scaled_weights > 1e-12,
            "BUY",
            np.where(scaled_weights < -1e-12, "SELL", "HOLD"),
        )

        jp_tickers = [f"{t}.T" for t in range(1617, 1617 + self.N_J)]
        return {
            "trade_date": resolved_trade_date,
            "tickers": jp_tickers,
            "signal": signal,
            "raw_weight": weights,
            "scale": float(scale),
            "weight": scaled_weights,
            "action": action,
            "sigma_s": sigma_s,
            "dispersion_indicator": float(dispersion_ind),
            "dispersion_metric": self.dispersion_metric,
        }
