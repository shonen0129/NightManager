"""runner/fast.py — fast decision mode (no yfinance).

Uses precomputed strategy cache + broker API for US returns + JP opens.
Skips the heavy computation (eigen decomposition, full df_exec build)
for sub-second decision latency.
"""

from __future__ import annotations

import json
import logging
import os
import time as time_module

import numpy as np
import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.core import signal as signals
from leadlag.data.cache import (
    exclusive_lock as _exclusive_lock,
)
from leadlag.data.cache import (
    load_df_exec_from_local_cache as _load_df_exec_from_local_cache,
)
from leadlag.data.cache import (
    read_cache_with_lock as _read_cache_with_lock,
)
from leadlag.data.cache import (
    write_cache_with_lock as _write_cache_with_lock,
)
from leadlag.data.market_data import (
    fetch_opens_from_google as _fetch_opens_from_google,
)
from leadlag.data.market_data import (
    load_opens_from_csv as _load_opens_from_csv,
)
from leadlag.data.market_data import (
    validate_manual_opens as _validate_manual_opens,
)
from leadlag.data.market_data import (
    validate_topix_open as _validate_topix_open,
)
from leadlag.data.market_data import (
    validate_us_returns_map as _validate_us_returns_map,
)
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from leadlag.execution.config import StrategyConfig as ProductionConfig
from leadlag.execution.helpers import (
    build_strategy,
    execute_post_decision_flow,
)

logger = logging.getLogger(__name__)

# US returns cache TTL: 3 hours
FAST_US_RETURNS_CACHE_MAX_AGE_SECONDS: int = 3 * 60 * 60


class PrecomputedLeadLagStrategy:
    """Lightweight strategy using precomputed cache for fast decision-making."""

    def __init__(
        self,
        cache_data: dict,
        gap_override: np.ndarray | None = None,
        ewma_half_life: float | None = None,
        lambda_reg: float | None = None,
        lambda_lw: float | None = None,
        lw_target: str | None = None,
        corr_window: int | None = None,
        K: int | None = None,
        q: float | None = None,
        weight_mode: str | None = None,
        dispersion_filter: bool | None = None,
        dispersion_metric: str | None = None,
        signal_mode: str | None = None,
        gap_open_coef: float | None = None,
        topix_beta_coef: float | None = None,
        beta_window: int | None = None,
        gamma: float | None = None,
        vol_adjusted_target: bool | None = None,
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

        # defaults matching StrategyConfig
        default_cfg = StrategyConfigCompatibility()

        self.C_full = cache_data["C_full"]
        self.v3_mode = str(cache_data.get("v3_mode", "static"))

        if self.v3_mode != "static":
            raise ValueError("PrecomputedLeadLagStrategy currently supports only v3_mode='static'")
        if "V_0" not in cache_data or "C_0" not in cache_data:
            raise ValueError("Precomputed cache is missing V_0/C_0. Rebuild cache first.")
        self.V_0 = cache_data["V_0"]
        self.C_0 = cache_data["C_0"]

        self.all_cc = cache_data["all_cc"]
        self.us_cc = cache_data["us_cc"]
        self.jp_oc = cache_data["jp_oc"]
        self.jp_gap = cache_data["jp_gap"] if "jp_gap" in cache_data else None
        self.jp_beta = cache_data["jp_beta"] if "jp_beta" in cache_data else None
        self.topix_night = cache_data["topix_night"] if "topix_night" in cache_data else None
        self.trade_dates = pd.DatetimeIndex(cache_data.get("trade_dates", []))

        ewma_value = _resolve_value(
            "ewma_half_life",
            ewma_half_life,
            default_cfg.ewma_half_life,
        )
        ewma_value = float(ewma_value)
        self.ewma_half_life = None if ewma_value < 0 else ewma_value

        self.lambda_reg = float(_resolve_value("lambda_reg", lambda_reg, default_cfg.lambda_reg))
        self.lambda_lw = float(_resolve_value("lambda_lw", lambda_lw, default_cfg.lambda_lw))
        self.lw_target = str(_resolve_value("lw_target", lw_target, default_cfg.lw_target))
        self.corr_window = int(_resolve_value("corr_window", corr_window, default_cfg.corr_window))
        self.N_U = int(cache_data.get("N_U", len(US_TICKERS)))
        self.N_J = int(cache_data.get("N_J", len(JP_TICKERS)))
        self.N = int(cache_data.get("N", len(US_TICKERS) + len(JP_TICKERS)))
        self.v1 = cache_data["v1"]
        self.v2 = cache_data["v2"]
        self.K = int(_resolve_value("K", K, default_cfg.k))
        self.q = float(_resolve_value("q", q, default_cfg.q))
        self.weight_mode = str(_resolve_value("weight_mode", weight_mode, default_cfg.weight_mode))
        self.dispersion_filter = bool(
            _resolve_value("dispersion_filter", dispersion_filter, default_cfg.dispersion_filter)
        )
        self.dispersion_metric = str(
            _resolve_value("dispersion_metric", dispersion_metric, default_cfg.dispersion_metric)
        )
        self.signal_mode = str(_resolve_value("signal_mode", signal_mode, default_cfg.signal_mode))
        self.gamma = float(_resolve_value("gamma", gamma, default_cfg.gamma))
        self.gap_open_coef = float(
            _resolve_value("gap_open_coef", gap_open_coef, default_cfg.gap_open_coef)
        )
        self.topix_beta_coef = float(
            _resolve_value("topix_beta_coef", topix_beta_coef, default_cfg.topix_beta_coef)
        )
        self.beta_window = int(_resolve_value("beta_window", beta_window, default_cfg.beta_window))
        self.vol_adjusted_target = bool(
            _resolve_value(
                "vol_adjusted_target", vol_adjusted_target, default_cfg.vol_adjusted_target
            )
        )
        self.gap_override = gap_override
        self._signal_cache: dict[int, dict[str, np.ndarray | float]] = {}

    @classmethod
    def load_precomputed_strategy(
        cls,
        cache_path: str,
        gap_override: np.ndarray | None = None,
    ) -> PrecomputedLeadLagStrategy:
        """Load precomputed strategy from a compressed numpy cache."""
        npz_data = np.load(cache_path, allow_pickle=True)
        cache_data = {key: npz_data[key] for key in npz_data.files}
        return cls(cache_data=cache_data, gap_override=gap_override)

    def _resolve_trade_index(self, trade_date) -> tuple[int, pd.Timestamp]:
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
        if current_index <= 0 or current_index >= len(self.all_cc):
            raise ValueError(
                "current_index must be within [1, T-1], "
                f"got {current_index} for T={len(self.all_cc)}"
            )

        use_cache = (
            jp_gap_override is None and us_returns_override is None and topix_night_override is None
        )
        if use_cache and current_index in self._signal_cache:
            return self._signal_cache[current_index]

        all_returns = self.all_cc
        if us_returns_override is not None:
            us_vec = np.asarray(us_returns_override, dtype=float).reshape(-1)
            if us_vec.shape[0] != self.N_U:
                raise ValueError(
                    f"us_returns_override length must be {self.N_U}, got {us_vec.shape[0]}"
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
                        f"jp_gap_override length must be {self.N_J}, got {gap_arr.shape[0]}"
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

        return {
            "trade_date": resolved_trade_date,
            "tickers": JP_TICKERS,
            "signal": signal,
            "raw_weight": weights,
            "scale": float(scale),
            "weight": scaled_weights,
            "action": action,
            "sigma_s": sigma_s,
            "dispersion_indicator": float(dispersion_ind),
            "dispersion_metric": self.dispersion_metric,
        }


class StrategyConfigCompatibility:
    """Mock StrategyConfig compatibility properties."""

    k: int = 6
    lambda_reg: float = 0.75
    q: float = 0.3
    weight_mode: str = "signal"
    dispersion_filter: bool = False
    dispersion_metric: str = "long_short_mean_gap"
    v3_mode: str = "static"
    ewma_half_life: int = 45
    lambda_lw: float = 0.5
    lw_target: str = "equicorrelation"
    corr_window: int = 60
    include_v4_prior: bool = True
    signal_mode: str = "gap_residual"
    gap_open_coef: float = 0.70
    topix_beta_coef: float = 0.6
    beta_window: int = 60
    gamma: float = 0.5
    slippage_bps: float = 5.0
    vol_adjusted_target: bool = True


# ---------------------------------------------------------------------------
# Precomputed cache builder
# ---------------------------------------------------------------------------


def build_precomputed_cache(
    config: ProductionConfig,
    df_exec: pd.DataFrame,
    cache_path: str,
) -> str:
    """Build and save the precomputed strategy cache to ``cache_path``."""
    if config.v3_mode != "static":
        raise ValueError(
            f"FAST MODE currently supports v3_mode='static' only. Got v3_mode={config.v3_mode!r}."
        )

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # In SRE mode, SRE uses LeadLagStrategy static components inside P0 / P3
    # SRE model handles saving cache using nested strategies
    from leadlag.models.sre import SectorRelativeEnsembleModel

    SectorRelativeEnsembleModel(config)

    # Adapt to save static cache fields
    # We can reconstruct it matching LeadLagStrategy cache fields
    from leadlag.core.correlation import (
        build_base_vectors,
        build_c0_from_v0,
        build_v3_static,
        compute_baseline_correlation,
    )

    all_cc_vals = df_exec[
        [c for c in df_exec.columns if c.startswith("us_cc_") or c.startswith("jp_cc_")]
    ].values
    sim_dates = df_exec.index.values
    C_full = compute_baseline_correlation(all_cc_vals, sim_dates, config.ewma_half_life)
    V_0 = build_v3_static(len(US_TICKERS), len(JP_TICKERS), config.include_v4_prior)
    C_0 = build_c0_from_v0(V_0, C_full)
    base_vecs = build_base_vectors(len(US_TICKERS), len(JP_TICKERS))

    cache_data = {
        "C_full": C_full,
        "v3_mode": config.v3_mode,
        "V_0": V_0,
        "C_0": C_0,
        "all_cc": all_cc_vals,
        "us_cc": df_exec[[c for c in df_exec.columns if c.startswith("us_cc_")]].values,
        "jp_oc": df_exec[[c for c in df_exec.columns if c.startswith("jp_oc_")]].values,
        "jp_gap": df_exec[[c for c in df_exec.columns if c.startswith("jp_gap_")]].values,
        "trade_dates": df_exec.index.values,
        "ewma_half_life": float(config.ewma_half_life) if config.ewma_half_life else -1.0,
        "lambda_reg": config.lambda_reg,
        "lambda_lw": config.lambda_lw,
        "lw_target": config.lw_target,
        "corr_window": config.corr_window,
        "N_U": len(US_TICKERS),
        "N_J": len(JP_TICKERS),
        "N": len(US_TICKERS) + len(JP_TICKERS),
        "v1": base_vecs["v1"],
        "v2": base_vecs["v2"],
        "K": config.k,
        "q": config.q,
        "signal_mode": config.signal_mode,
        "gap_open_coef": config.gap_open_coef,
        "topix_beta_coef": config.topix_beta_coef,
        "beta_window": config.beta_window,
        "dispersion_metric": config.dispersion_metric,
        "dispersion_filter": config.dispersion_filter,
        "weight_mode": config.weight_mode,
        "include_v4_prior": config.include_v4_prior,
        "gamma": config.gamma,
        "vol_adjusted_target": config.vol_adjusted_target,
    }

    if any(c.startswith("jp_beta_") for c in df_exec.columns):
        cache_data["jp_beta"] = df_exec[
            [c for c in df_exec.columns if c.startswith("jp_beta_")]
        ].values
    if "topix_night_return" in df_exec.columns:
        cache_data["topix_night"] = df_exec["topix_night_return"].values

    np.savez_compressed(cache_path, **cache_data)
    logger.info(f"Precomputed cache saved to {cache_path}")
    return cache_path


def fetch_us_returns_from_api(
    api_client: BrokerClient,
    output_root: str,
) -> np.ndarray:
    """Fetch US ETF returns from the broker API with local JSON cache."""
    cache_dir = os.path.join(output_root, ".cache")
    us_cache = os.path.join(cache_dir, "us_returns.json")
    lock_path = us_cache + ".lock"
    os.makedirs(cache_dir, exist_ok=True)

    with _exclusive_lock(lock_path):
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        now_epoch = time_module.time()

        if os.path.exists(us_cache):
            try:
                with open(us_cache, encoding="utf-8") as f:
                    cached = json.load(f)
                fetched_at_epoch = float(cached.get("fetched_at_epoch", 0.0))
                cache_age = now_epoch - fetched_at_epoch if fetched_at_epoch > 0 else None

                if (
                    cached.get("date", "") == today
                    and cache_age is not None
                    and cache_age <= FAST_US_RETURNS_CACHE_MAX_AGE_SECONDS
                ):
                    cached_returns = _validate_us_returns_map(cached.get("returns", {}))
                    logger.info("[FAST MODE] Using cached US returns (age=%.0fs)...", cache_age)
                    return np.array([cached_returns[tk] for tk in US_TICKERS], dtype=float)

                if cached.get("date", "") == today:
                    logger.info(
                        "[FAST MODE] US return cache is stale; refetching (max_age=%ss)",
                        FAST_US_RETURNS_CACHE_MAX_AGE_SECONDS,
                    )
            except Exception as e:
                logger.warning("[FAST MODE] Ignoring invalid US return cache and refetching: %s", e)

        logger.info("[FAST MODE] Fetching US ETF returns from broker API...")
        fetched = api_client.fetch_us_etf_returns(US_TICKERS)
        normalized = _validate_us_returns_map(fetched)

        cache_data = {
            "date": today,
            "fetched_at_epoch": now_epoch,
            "returns": normalized,
        }
        tmp_path = f"{us_cache}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
            os.replace(tmp_path, us_cache)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return np.array([normalized[tk] for tk in US_TICKERS], dtype=float)


def fetch_jp_opens_for_fast_mode(
    api_client: BrokerClient,
    config: ProductionConfig,
    jp_opens_csv: str | None,
    google_opens: bool,
) -> tuple[dict, float | None]:
    """Fetch JP open prices for fast mode with API → Google → CSV fallback."""
    if jp_opens_csv is not None:
        manual_opens = _load_opens_from_csv(jp_opens_csv)
    elif google_opens:
        logger.info("  Fetching from Google Finance...")
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]
        manual_opens = _fetch_opens_from_google(tickers=tickers_for_opens)
    else:
        logger.info("  Fetching from kabu API...")
        tickers_for_opens = JP_TICKERS + [TOPIX_TICKER]
        manual_opens = api_client.fetch_open_prices(tickers_for_opens, allow_missing=True)
        missing = [tk for tk in tickers_for_opens if tk not in manual_opens]
        if missing:
            logger.warning(
                "  Falling back to Google Finance for %d ticker(s): %s",
                len(missing),
                ", ".join(missing),
            )
            google_fetched = _fetch_opens_from_google(tickers=missing, allow_missing=True)
            manual_opens.update(google_fetched)
            missing_jp = [tk for tk in JP_TICKERS if tk not in manual_opens]
            if missing_jp:
                raise ValueError(
                    "Missing open prices after API + Google fallback: " + ", ".join(missing_jp)
                )

    _validate_manual_opens(manual_opens)
    topix_open = None
    if config.signal_mode == "gap_residual":
        topix_open = _validate_topix_open(manual_opens)

    return manual_opens, topix_open


def run_decision_fast(
    config: ProductionConfig,
    cache_path: str,
    trade_date: pd.Timestamp,
    manual_opens: dict,
    gap_override: np.ndarray,
    topix_night_override: float | None,
    us_returns_today: np.ndarray,
    max_capital: float,
    output_dir: str,
    output_root: str,
    api_client: BrokerClient | None = None,
    api_dry_run: bool = False,
    text_output: bool = False,
) -> str:
    """Generate a trade decision using the precomputed strategy cache."""
    logger.info("[FAST MODE] Loading precomputed strategy cache...")
    precomputed = PrecomputedLeadLagStrategy.load_precomputed_strategy(
        cache_path,
        gap_override=gap_override,
    )
    logger.info("[FAST MODE] Cache loaded successfully")

    logger.info("[FAST MODE] Generating trade decision (fast path)...")
    decision = precomputed.generate_trade_decision(
        trade_date=trade_date,
        jp_gap_override=gap_override,
        us_returns_override=us_returns_today,
        topix_night_override=topix_night_override,
    )
    # Historical returns for VaR/ES (file-locked, local caches only)
    cache_dir = os.path.join(output_root, ".cache")
    returns_cache = os.path.join(cache_dir, "daily_returns.csv")
    hist_returns = _read_cache_with_lock(returns_cache, decision["trade_date"])
    if hist_returns is None:
        logger.info("No returns cache; rebuilding VaR/ES history from local cache...")
        from leadlag.execution.backtester import BacktestEngine
        df_exec = _load_df_exec_from_local_cache()
        strategy = build_strategy(config, df_exec)
        out_res = BacktestEngine.run_backtest(strategy, df_exec, start_date=config.start_date)
        hist_results = pd.DataFrame(
            {"daily_return": out_res["daily_returns"]}, index=out_res["daily_returns"].index
        )
        _write_cache_with_lock(returns_cache, hist_results)
        hist_returns = pd.Series(
            hist_results.loc[hist_results.index < decision["trade_date"], "daily_return"]
        )

    out_path = execute_post_decision_flow(
        decision=decision,
        config=config,
        manual_opens=manual_opens,
        max_capital=max_capital,
        hist_returns=hist_returns,
        output_dir=output_dir,
        api_client=api_client,
        api_dry_run=api_dry_run,
        text_output=text_output,
    )

    return out_path
