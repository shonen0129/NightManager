"""BacktestEngine — V2 production backtest engine.

The legacy V1 generic ``BaseModel`` backtest has been moved to
``research.backtest_v1.run_v1_backtest``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.config.schemas import AppConfig
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.preprocessor import compute_jp_target_returns
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import build_app_config_from_dict
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.ml_order_overlay import (
    MLOrderOverlayModel,
    load_overlay_model,
)
from leadlag.models.production_v2 import ProductionV2Model

logger = logging.getLogger(__name__)


class BacktestEngine:
    """V2 engine for executing production historical backtests."""

    @staticmethod
    def _resolve_sim_dates(
        df_exec: pd.DataFrame,
        start_date: str,
        end_date: str,
        min_start_idx: int,
    ) -> tuple[pd.DatetimeIndex, int, int]:
        """Resolve simulation start/end indices and the full date index."""
        T = len(df_exec)
        sim_dates = cast(pd.DatetimeIndex, df_exec.index)

        start_dt = pd.to_datetime(start_date)
        start_idx = max(int(sim_dates.searchsorted(start_dt)), min_start_idx)

        if end_date != "latest":
            end_dt = pd.to_datetime(end_date)
            end_idx = min(int(sim_dates.searchsorted(end_dt)), T - 1)
        else:
            end_idx = T - 1

        # Exclude provisional rows (today's close not yet available, r_oc=0.0)
        if "is_provisional" in df_exec.columns:
            while end_idx >= start_idx and bool(df_exec["is_provisional"].iloc[end_idx]):
                end_idx -= 1

        return sim_dates, start_idx, end_idx

    @staticmethod
    def _compute_target_and_gap_returns(
        df_exec: pd.DataFrame,
        sim_dates: pd.DatetimeIndex,
        sim_dates_slice: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 9:10-to-close target returns and overnight gap returns."""
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
        y_jp_target_df = pd.DataFrame(y_jp_target, index=sim_dates, columns=JP_TICKERS)
        y_jp_target_arr = y_jp_target_df.loc[sim_dates_slice].values

        # Overnight gap returns: gap(t) = open(t)/close(t-1) - 1
        gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
        if all(c in df_exec.columns for c in gap_cols):
            gap_returns_df = df_exec[gap_cols].copy()
            gap_returns_df.columns = JP_TICKERS
        else:
            gap_returns_df = pd.DataFrame(0.0, index=sim_dates, columns=JP_TICKERS)

        gap_returns_arr = gap_returns_df.loc[sim_dates_slice].values
        return y_jp_target_arr, gap_returns_arr

    @classmethod
    def _simulate_daily_pnl(
        cls,
        weights: np.ndarray,
        target_returns: np.ndarray,
        gap_returns: np.ndarray,
        sim_dates: pd.DatetimeIndex,
        slip: float,
        financing_daily: float,
        borrow_daily: float,
        reverse_daily: float,
        alpha_long: float,
        alpha_short: float,
        side_leverage: float = 1.0,
        oc_returns: np.ndarray | None = None,
    ) -> dict:
        """Simulate daily gross/net returns and costs for a weight matrix.

        This is the common cost model shared by V1
        (``research.backtest_v1.run_v1_backtest``) and V2
        (``run_v2_backtest``).  The only differences are the optional
        side-leverage multiplier and the optional open-to-close auxiliary
        series.

        Args:
            weights: (n_sim_days, n_j) array of portfolio weights.
            target_returns: (n_sim_days, n_j) 9:10-to-close returns.
            gap_returns: (n_sim_days, n_j) overnight gap returns.
            sim_dates: Simulation dates used to compute calendar days held.
            slip: One-way slippage fraction (bps/10000).
            financing_daily: Daily financing rate.
            borrow_daily: Daily borrow fee.
            reverse_daily: Daily reverse-fee fraction.
            alpha_long: Long overnight hold fraction.
            alpha_short: Short overnight hold fraction.
            side_leverage: Notional leverage multiplier.  Default 1.0 for
                ``research.backtest_v1.run_v1_backtest``; ``run_v2_backtest`` passes 1.5.
            oc_returns: Optional (n_sim_days, n_j) open-to-close returns.
                If provided, open-to-close gross/net series are computed.

        Returns:
            Dict of daily lists:
                gross_returns, net_returns, gross_returns_oc, net_returns_oc,
                costs, slip_costs, financing_costs, borrow_costs, reverse_costs,
                overnight_returns, gross_exps, turnover.
        """
        n_sim_days = len(weights)
        n_j = weights.shape[1]
        w_prev = np.zeros(n_j)

        # Calculate calendar days between trading dates to scale financing/borrow fees correctly.
        # For the last trading day, assume 1 calendar day as fallback.
        calendar_days = np.ones(n_sim_days)
        sim_dates_pd = pd.to_datetime(sim_dates)
        for i in range(n_sim_days - 1):
            calendar_days[i] = (sim_dates_pd[i + 1] - sim_dates_pd[i]).days

        gross_returns_list = []
        net_returns_list = []
        gross_returns_oc_list = []
        net_returns_oc_list = []
        cost_list = []
        slip_cost_list = []
        financing_cost_list = []
        borrow_cost_list = []
        reverse_cost_list = []
        overnight_ret_list = []
        gross_exp_list = []
        turnover_list = []

        include_oc = oc_returns is not None

        for i in range(n_sim_days):
            w_t = weights[i]
            r_target_t = target_returns[i]
            days_held = calendar_days[i]

            gross_ret = side_leverage * float(np.sum(w_t * r_target_t))
            gross_exp = float(np.sum(np.abs(w_t)))

            alpha_mask = np.where(w_t > 0, alpha_long, np.where(w_t < 0, alpha_short, 0.0))

            overnight_ret = 0.0
            if (alpha_long > 0 or alpha_short > 0) and i < n_sim_days - 1:
                r_gap_next = gap_returns[i + 1]
                overnight_ret = side_leverage * float(np.sum(alpha_mask * w_t * r_gap_next))

            turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)

            slip_cost = side_leverage * slip * (
                2.0 * np.sum((1.0 - alpha_mask) * np.abs(w_t))
                + np.sum(alpha_mask * np.abs(w_t - w_prev) / 2.0)
            )
            held_long = float(np.sum(alpha_mask * np.maximum(w_t, 0.0)))
            held_short = float(np.sum(alpha_mask * np.maximum(-w_t, 0.0)))
            fin_cost = side_leverage * held_long * financing_daily * days_held
            borrow_cost = side_leverage * held_short * borrow_daily * days_held
            reverse_cost = side_leverage * held_short * reverse_daily * days_held
            cost = slip_cost + fin_cost + borrow_cost + reverse_cost

            net_ret = gross_ret + overnight_ret - cost

            gross_returns_list.append(gross_ret + overnight_ret)
            net_returns_list.append(net_ret)
            cost_list.append(cost)
            slip_cost_list.append(slip_cost)
            financing_cost_list.append(fin_cost)
            borrow_cost_list.append(borrow_cost)
            reverse_cost_list.append(reverse_cost)
            overnight_ret_list.append(overnight_ret)
            gross_exp_list.append(gross_exp)
            turnover_list.append(turnover)

            if oc_returns is not None:
                r_oc_t = oc_returns[i]
                gross_ret_oc = side_leverage * float(np.sum(w_t * r_oc_t))
                net_ret_oc = gross_ret_oc - cost
                gross_returns_oc_list.append(gross_ret_oc)
                net_returns_oc_list.append(net_ret_oc)

            w_prev = w_t

        result = {
            "gross_returns": gross_returns_list,
            "net_returns": net_returns_list,
            "costs": cost_list,
            "slip_costs": slip_cost_list,
            "financing_costs": financing_cost_list,
            "borrow_costs": borrow_cost_list,
            "reverse_costs": reverse_cost_list,
            "overnight_returns": overnight_ret_list,
            "gross_exps": gross_exp_list,
            "turnover": turnover_list,
        }
        if include_oc:
            result["gross_returns_oc"] = gross_returns_oc_list
            result["net_returns_oc"] = net_returns_oc_list
        return result

    # ------------------------------------------------------------------
    # V2 backtest (ProductionV2 model — gap-adjusted distribution)
    # ------------------------------------------------------------------

    @classmethod
    def run_v2_backtest(
        cls,
        cfg: AppConfig | dict,
        gap_input_dir: Path | str | None,
        df_exec: pd.DataFrame,
        start_date: str = "2015-01-05",
        end_date: str = "latest",
        slippage_bps: float | None = None,
        overnight_alpha_long: float | None = None,
        overnight_alpha_short: float | None = None,
        buy_interest_annual: float | None = None,
        borrow_fee_annual: float | None = None,
        reverse_fee_bps: float | None = None,
        side_leverage: float | None = None,
        n_jobs: int = 1,
        overlay_model: MLOrderOverlayModel | None = None,
        overlay_model_dir: Path | str | None = None,
    ) -> dict:
        """Run a historical backtest using the V2 production model.

        Calls ``generate_v2_production_portfolio()`` for each trading date,
        loading per-date gap-adjusted distribution matrices from
        *gap_input_dir*.  The cost model is identical to
        ``research.backtest_v1.run_v1_backtest``.

        Args:
            cfg: Validated ``AppConfig`` or raw V2 production YAML dict.
            gap_input_dir: Path to a SQLite GapStore or a directory with
                ``mu_gap_{YYYYMMDD}.npy`` / ``omega_gap_{YYYYMMDD}.npy`` files.
                If None, every day will be a flat-position fallback.
            df_exec: Execution DataFrame.
            start_date: Backtest start date.
            end_date: Backtest end date ("latest" for last available).
            slippage_bps: Slippage bps one-way (defaults from cfg["costs"]).
            overnight_alpha_long: Long overnight hold fraction.
            overnight_alpha_short: Short overnight hold fraction.
            buy_interest_annual: Annual financing rate for longs.
            borrow_fee_annual: Annual borrow fee for shorts.
            reverse_fee_bps: Daily reverse stock lending fee (bps).
            side_leverage: Notional leverage applied to returns and costs,
                matching ``allocator.DEFAULT_SIDE_LEVERAGE`` in live trading.
                Gross exposure and turnover are reported at raw weight values.
            n_jobs: Number of parallel workers for per-date portfolio generation.
                1 = sequential. -1 = all cores.

        Returns:
            Dict with the same keys as ``research.backtest_v1.run_v1_backtest``,
            plus ``daily_fallback`` (bool series) and ``v2_summaries`` (list of
            per-date summary dicts).
        """
        app_config = (
            cfg if isinstance(cfg, AppConfig) else build_app_config_from_dict(cfg)
        )

        if overlay_model is None and overlay_model_dir is not None:
            overlay_model = load_overlay_model(Path(overlay_model_dir))
            logger.info("Loaded overlay model from %s", overlay_model_dir)

        cost_params = cls._resolve_v2_backtest_cost_params(
            app_config,
            slippage_bps,
            overnight_alpha_long,
            overnight_alpha_short,
            buy_interest_annual,
            borrow_fee_annual,
            reverse_fee_bps,
            side_leverage,
        )
        slip_bps = cost_params["slip_bps"]
        alpha_long = cost_params["alpha_long"]
        alpha_short = cost_params["alpha_short"]
        fin_annual = cost_params["fin_annual"]
        borrow_annual = cost_params["borrow_annual"]
        rev_bps = cost_params["rev_bps"]
        side_leverage = cost_params["side_leverage"]

        gap_dir: Path | None = Path(gap_input_dir) if gap_input_dir is not None else None

        logger.info(
            f"Starting V2 backtest: start={start_date}, gap_dir={gap_dir}, "
            f"slippage={slip_bps} bps, alpha_long={alpha_long}, alpha_short={alpha_short}, "
            f"financing={fin_annual*100:.2f}% ann, "
            f"borrow={borrow_annual*100:.2f}% ann, reverse={rev_bps:.1f} bps/day, "
            f"side_leverage={side_leverage}"
        )

        sim_dates, start_idx, end_idx = cls._resolve_sim_dates(df_exec, start_date, end_date, 0)
        sim_dates_slice = cast(pd.DatetimeIndex, sim_dates[start_idx : end_idx + 1])

        y_jp_target_arr, gap_returns_arr = cls._compute_target_and_gap_returns(
            df_exec, sim_dates, sim_dates_slice
        )

        n_j = len(JP_TICKERS)
        sre_weights, fallback_flags, v2_summaries = cls._generate_v2_weights(
            df_exec,
            app_config,
            gap_dir,
            sim_dates_slice,
            n_j,
            overlay_model,
            n_jobs,
        )

        sre_weights_df = pd.DataFrame(sre_weights, index=sim_dates_slice, columns=JP_TICKERS)

        # Cost parameters
        slip = slip_bps / 10000.0
        financing_daily = fin_annual / 365.0
        borrow_daily = borrow_annual / 365.0
        reverse_daily = rev_bps / 10000.0

        pnl = cls._simulate_daily_pnl(
            weights=sre_weights,
            target_returns=y_jp_target_arr,
            gap_returns=gap_returns_arr,
            sim_dates=sim_dates_slice,
            slip=slip,
            financing_daily=financing_daily,
            borrow_daily=borrow_daily,
            reverse_daily=reverse_daily,
            alpha_long=alpha_long,
            alpha_short=alpha_short,
            side_leverage=side_leverage,
        )

        return cls._assemble_v2_results(
            pnl,
            sre_weights_df,
            fallback_flags,
            v2_summaries,
            sim_dates_slice,
            alpha_long,
            alpha_short,
            side_leverage,
        )

    @staticmethod
    def _resolve_v2_backtest_cost_params(
        app_config: AppConfig,
        slippage_bps: float | None,
        overnight_alpha_long: float | None,
        overnight_alpha_short: float | None,
        buy_interest_annual: float | None,
        borrow_fee_annual: float | None,
        reverse_fee_bps: float | None,
        side_leverage: float | None,
    ) -> dict:
        """Resolve cost/financing and side-leverage parameters for run_v2_backtest."""
        # Prefer the V2 cost sub-model; fall back to the legacy StrategyConfig fields.
        v2_costs = getattr(app_config.v2, "costs", None)
        v2_costs = v2_costs or app_config.strategy
        strategy = app_config.strategy

        def _get(attr: str, prefer_v2: bool = True) -> Any:
            if prefer_v2 and v2_costs is not None and hasattr(v2_costs, attr):
                v = getattr(v2_costs, attr)
                if v is not None:
                    return v
            if hasattr(strategy, attr):
                return getattr(strategy, attr)
            return None

        def _resolve(override: Any, attr: str) -> Any:
            if override is not None:
                return override
            v2_v = _get(attr)
            if v2_v is not None:
                return v2_v
            return _get(attr, prefer_v2=False)

        slip_bps = _resolve(slippage_bps, "slippage_bps_per_side")
        if slip_bps is None:
            slip_bps = _resolve(slippage_bps, "slippage_bps")
        alpha_long = _resolve(overnight_alpha_long, "overnight_alpha_long")
        alpha_short = _resolve(overnight_alpha_short, "overnight_alpha_short")
        fin_annual = _resolve(buy_interest_annual, "buy_interest_annual")
        borrow_annual = _resolve(borrow_fee_annual, "borrow_fee_annual")
        rev_bps = _resolve(reverse_fee_bps, "reverse_fee_bps")

        if side_leverage is None:
            side_leverage = _resolve(None, "side_leverage")

        return {
            "slip_bps": slip_bps,
            "alpha_long": alpha_long,
            "alpha_short": alpha_short,
            "fin_annual": fin_annual,
            "borrow_annual": borrow_annual,
            "rev_bps": rev_bps,
            "side_leverage": side_leverage,
        }

    @staticmethod
    def _generate_v2_weights(
        df_exec: pd.DataFrame,
        app_config: AppConfig,
        gap_dir: Path | None,
        sim_dates_slice: pd.DatetimeIndex,
        n_j: int,
        overlay_model: MLOrderOverlayModel | None,
        n_jobs: int,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Generate V2 weights for each simulation date using the unified V2 model."""
        n_sim_days = len(sim_dates_slice)
        sre_weights = np.zeros((n_sim_days, n_j))
        fallback_flags = np.zeros(n_sim_days, dtype=bool)
        v2_summaries = cast(list[dict], [None] * n_sim_days)

        run_cfg = app_config.v2

        effective_gap_dir: Path | None = gap_dir

        # Build the BLPX model and the unified V2 decision model.
        blpx_model = ProductionBLPXModel(run_cfg.blpx)
        if n_jobs > 1:
            blpx_model.clear_caches()

        v2_model = ProductionV2Model(
            run_cfg,
            blpx_model=blpx_model,
            overlay_model=overlay_model,
        )

        lake = PITDataLake(df_exec)

        def _process_date(i_dt: tuple[int, pd.Timestamp]) -> tuple[int, np.ndarray, bool, dict]:
            i, dt = i_dt
            date_str = dt.strftime("%Y-%m-%d")
            try:
                snapshot = lake.get_snapshot(dt)
                result = v2_model.decide(
                    trade_date=date_str,
                    gap_input_dir=effective_gap_dir,
                    lake=lake,
                    snapshot=snapshot,
                    overlay_enabled=run_cfg.ml_overlay_enabled,
                    use_file_cache=True,
                )
                w = result["w_final"]
                fb = result["fallback"]["gap_data_missing"]
                summary = result.get("summary", {})
                return i, w, fb, summary
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                logger.warning("[%s] V2 generation failed: %s — flat position", date_str, e)
                return i, np.zeros(n_j), True, {"trade_date": date_str, "error": str(e)}
            except Exception as e:
                logger.error("[%s] Unexpected V2 generation error: %s", date_str, e)
                raise

        date_index_pairs = list(enumerate(sim_dates_slice))

        if n_jobs == 1 or n_sim_days <= 1:
            for pair in date_index_pairs:
                i, w, fb, summary = _process_date(pair)
                sre_weights[i] = w
                fallback_flags[i] = fb
                v2_summaries[i] = summary
                if fb:
                    logger.debug("[%s] V2 fallback (gap data missing)", sim_dates_slice[i].strftime("%Y-%m-%d"))
                if (i + 1) % 200 == 0:
                    logger.info("V2 backtest: processed %d/%d dates", i + 1, n_sim_days)
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                delayed(_process_date)(pair) for pair in date_index_pairs
            )
            for i, w, fb, summary in results:
                sre_weights[i] = w
                fallback_flags[i] = fb
                v2_summaries[i] = summary
                if fb:
                    logger.debug("[%s] V2 fallback (gap data missing)", sim_dates_slice[i].strftime("%Y-%m-%d"))
            logger.info("V2 backtest: processed %d/%d dates (parallel, n_jobs=%d)", n_sim_days, n_sim_days, n_jobs)

        return sre_weights, fallback_flags, v2_summaries

    @staticmethod
    def _assemble_v2_results(
        pnl: dict,
        sre_weights_df: pd.DataFrame,
        fallback_flags: np.ndarray,
        v2_summaries: list[dict],
        sim_dates_slice: pd.DatetimeIndex,
        alpha_long: float,
        alpha_short: float,
        side_leverage: float,
    ) -> dict:
        """Assemble the output dict for run_v2_backtest."""
        daily_returns_gross = pd.Series(pnl["gross_returns"], index=sim_dates_slice)
        daily_returns_net = pd.Series(pnl["net_returns"], index=sim_dates_slice)
        daily_costs = pd.Series(pnl["costs"], index=sim_dates_slice)
        daily_slip_costs = pd.Series(pnl["slip_costs"], index=sim_dates_slice)
        daily_financing_costs = pd.Series(pnl["financing_costs"], index=sim_dates_slice)
        daily_borrow_costs = pd.Series(pnl["borrow_costs"], index=sim_dates_slice)
        daily_reverse_costs = pd.Series(pnl["reverse_costs"], index=sim_dates_slice)
        daily_overnight_returns = pd.Series(pnl["overnight_returns"], index=sim_dates_slice)
        daily_gross_exps = pd.Series(pnl["gross_exps"], index=sim_dates_slice)
        daily_turnover = pd.Series(pnl["turnover"], index=sim_dates_slice)
        daily_fallback = pd.Series(fallback_flags, index=sim_dates_slice)

        wealth = (1.0 + daily_returns_net).cumprod()
        running_max = wealth.cummax()
        drawdown = (wealth / running_max) - 1.0

        n_fallback = int(fallback_flags.sum())
        n_sim_days = len(sim_dates_slice)
        logger.info(
            "V2 backtest done: %d days, %d fallback (%.1f%%)",
            n_sim_days, n_fallback, n_fallback / n_sim_days * 100 if n_sim_days > 0 else 0,
        )

        return {
            "weights": sre_weights_df,
            "daily_returns_gross": daily_returns_gross,
            "daily_returns": daily_returns_net,
            "daily_costs": daily_costs,
            "daily_slip_costs": daily_slip_costs,
            "daily_financing_costs": daily_financing_costs,
            "daily_borrow_costs": daily_borrow_costs,
            "daily_reverse_costs": daily_reverse_costs,
            "daily_overnight_returns": daily_overnight_returns,
            "daily_gross_exps": daily_gross_exps,
            "daily_turnover": daily_turnover,
            "daily_fallback": daily_fallback,
            "overnight_alpha_long": alpha_long,
            "overnight_alpha_short": alpha_short,
            "side_leverage": side_leverage,
            "equity_curve": wealth,
            "drawdown": drawdown,
            "v2_summaries": v2_summaries,
        }
