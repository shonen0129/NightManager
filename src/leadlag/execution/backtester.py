"""BacktestEngine — V2 production backtest engine.

The legacy V1 generic ``BaseModel`` backtest has been moved to
``research.backtest_v1.run_v1_backtest``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.compliance.v2_auditor import run_numerical_audit
from leadlag.config.schemas import AppConfig
from leadlag.core.pnl import simulate_daily_pnl
from leadlag.data import adr_features as adr_data
from leadlag.data import macro as macro_data
from leadlag.data.intraday_inputs import build_open_910_returns, compute_jp_target_returns
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import DecisionInputs, HistoricalInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.execution.config import build_app_config_from_dict
from leadlag.models.ml_order_overlay import MLOrderOverlayModel
from leadlag.models.v2.pit import load_pit_ir_history
from leadlag.reporting.metrics import compute_drawdown_series
from leadlag.runner.model_factory import build_v2_model_bundle
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint

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
            # searchsorted(..., side="right") gives the first index *after* end_dt;
            # subtract 1 to get the last trading day on or before end_dt. This
            # prevents a non-trading end_date from leaking the following business
            # day into the simulation.
            end_idx = int(sim_dates.searchsorted(end_dt, side="right")) - 1
            end_idx = max(0, min(end_idx, T - 1))
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
        open_910_returns: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 9:10-to-close target returns and overnight gap returns."""
        y_jp_target = compute_jp_target_returns(
            df_exec,
            JP_TICKERS,
            open_910_returns=open_910_returns,
        )
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
        """Compatibility adapter for the pure daily P&L calculator."""
        return simulate_daily_pnl(
            weights=weights,
            target_returns=target_returns,
            gap_returns=gap_returns,
            sim_dates=sim_dates,
            slip=slip,
            financing_daily=financing_daily,
            borrow_daily=borrow_daily,
            reverse_daily=reverse_daily,
            alpha_long=alpha_long,
            alpha_short=alpha_short,
            side_leverage=side_leverage,
            oc_returns=oc_returns,
        )

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
        decision_transform: Callable[[str, PortfolioDecision], PortfolioDecision] | None = None,
        historical_inputs: HistoricalInputs | None = None,
    ) -> dict:
        """Run a historical backtest using the V2 production model.

        Calls the canonical V2 model for each trading date,
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

        if historical_inputs is not None:
            expected_frame = PITDataLake(df_exec).df_exec
            supplied_frame = historical_inputs.to_frame()
            if dataframe_fingerprint(expected_frame) != dataframe_fingerprint(supplied_frame):
                raise ValueError(
                    "historical_inputs frame must match df_exec for a run-owned backtest"
                )

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

        # A backtest owns one 09:10 input frame for both realized P&L labels
        # and per-date model decisions.  Supplying it here prevents the target
        # calculation from reopening a different 5-minute cache.
        run_open_910_returns = (
            historical_inputs.open_910_returns
            if historical_inputs is not None
            else build_open_910_returns(df_exec, JP_TICKERS)
        )
        target_open_910_returns = run_open_910_returns
        if historical_inputs is not None and target_open_910_returns is None:
            # A supplied run with no intraday frame must remain explicit; the
            # pure target arithmetic then falls back to its existing jp_oc
            # labels instead of performing hidden adapter I/O.
            target_open_910_returns = pd.DataFrame(
                np.nan,
                index=df_exec.index,
                columns=JP_TICKERS,
            )
        y_jp_target_arr, gap_returns_arr = cls._compute_target_and_gap_returns(
            df_exec,
            sim_dates,
            sim_dates_slice,
            open_910_returns=target_open_910_returns,
        )

        n_j = len(JP_TICKERS)
        sre_weights, fallback_flags, v2_summaries = cls._generate_v2_weights(
            df_exec,
            app_config,
            gap_dir,
            sim_dates_slice,
            n_j,
            overlay_model,
            overlay_model_dir,
            n_jobs,
            decision_transform,
            historical_inputs,
            run_open_910_returns,
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
        # STRATEGY_SLIPPAGE_BPS must take precedence over both function argument
        # defaults and the config file, matching load_config_from_yaml behavior.
        env_slip = os.environ.get("STRATEGY_SLIPPAGE_BPS")
        if env_slip is not None and slippage_bps is None:
            slippage_bps = float(env_slip)

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
        overlay_model_dir: Path | str | None,
        n_jobs: int,
        decision_transform: Callable[[str, PortfolioDecision], PortfolioDecision] | None = None,
        historical_inputs: HistoricalInputs | None = None,
        open_910_returns: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Generate V2 weights for each simulation date using the unified V2 model."""
        n_sim_days = len(sim_dates_slice)
        sre_weights = np.zeros((n_sim_days, n_j))
        fallback_flags = np.zeros(n_sim_days, dtype=bool)
        v2_summaries = cast(list[dict], [None] * n_sim_days)

        effective_gap_dir: Path | None = gap_dir

        bundle = build_v2_model_bundle(
            app_config,
            overlay_model=overlay_model,
            overlay_model_dir=overlay_model_dir,
            clear_blpx_cache=n_jobs > 1,
        )
        v2_model = bundle.decision_model
        run_config = getattr(bundle, "run_config", app_config.v2)
        overlay_enabled = bool(getattr(bundle, "overlay_enabled", run_config.ml_overlay_enabled))

        lake = PITDataLake(df_exec)
        if historical_inputs is None:
            # Extract the intraday input once at the adapter boundary.  The model
            # receives this run-owned frame and cannot reopen the 5-minute cache
            # while computing a per-date decision.
            if open_910_returns is None:
                open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
            macro_prices = None
            if run_config.macro_kappa_enabled or run_config.macro_direction_enabled:
                try:
                    macro_prices = macro_data.load_macro_prices(
                        start=df_exec.index.min().strftime("%Y-%m-%d"),
                        end=(df_exec.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        period="max",
                    )
                except Exception as exc:
                    # Missing adapter data is represented explicitly in the
                    # snapshot; the typed model will apply its safe skip/fallback
                    # path without reopening the provider.
                    logger.warning("Failed to load run-owned macro prices: %s", exc)

            adr_features = None
            if overlay_enabled:
                try:
                    # The overlay selects the row for each trade date.  Loading
                    # the complete artifact once keeps the run deterministic and
                    # avoids a per-date filesystem read in model code.
                    adr_features = adr_data.load_adr_features()
                except Exception as exc:
                    logger.warning("Failed to load run-owned ADR features: %s", exc)

            pit_ir_history: dict[str, np.ndarray] | None = None
            pit_history_trade_dates: dict[str, np.ndarray] | None = None
            if effective_gap_dir is not None:
                pit_ir_history = {}
                pit_history_trade_dates = {}
                for dt in sim_dates_slice:
                    date_str = dt.strftime("%Y-%m-%d")
                    history_ir, _alerts, history_dates = load_pit_ir_history(
                        effective_gap_dir, date_str
                    )
                    pit_ir_history[date_str] = history_ir
                    pit_history_trade_dates[date_str] = history_dates

            rank_reversal_signals = None
            if run_config.cs_overlay_enabled:
                rank_reversal_signals = load_rank_reversal_frame(
                    effective_gap_dir,
                    sim_dates_slice,
                    file_pattern=run_config.cs_rank_reversal_file_pattern,
                )

            historical_observed_at_by_date = {
                dt.strftime("%Y-%m-%d"): {
                    "open_910_returns": f"{dt.date()} 09:10",
                    "macro_prices": f"{dt.date()} 09:00",
                    "adr_features": f"{dt.date()} 09:00",
                    "rank_reversal_signals": f"{dt.date()} 09:00",
                    "pit_ir_history": f"{dt.date()} 09:10",
                }
                for dt in sim_dates_slice
            }

            historical_inputs = HistoricalInputs(
                df_exec,
                source="backtest",
                open_910_returns=open_910_returns,
                macro_prices=macro_prices,
                adr_features_frame=adr_features,
                pit_ir_history=pit_ir_history,
                pit_history_trade_dates=pit_history_trade_dates,
                rank_reversal_signals=rank_reversal_signals,
                observed_at_by_date=historical_observed_at_by_date,
            )

        def _process_date(i_dt: tuple[int, pd.Timestamp]) -> tuple[int, np.ndarray, bool, dict]:
            i, dt = i_dt
            date_str = dt.strftime("%Y-%m-%d")
            try:
                # Every backtest decision is evaluated at the same 09:10 JST
                # cutoff used by the live bridge.  Passing a date-only value
                # would silently make the PIT contract stricter and would
                # leave source timestamps unspecified.
                decision_as_of = dt + pd.Timedelta(hours=9, minutes=10)
                snapshot = (
                    lake.get_execution_snapshot(decision_as_of, historical_inputs.open_910_returns)
                    if historical_inputs.open_910_returns is not None
                    and all(f"jp_open_trade_{ticker}" in df_exec for ticker in JP_TICKERS)
                    else lake.get_snapshot(decision_as_of)
                )
                sig_date = df_exec.loc[dt].get("sig_date") if "sig_date" in df_exec.columns else None
                if sig_date is not None and pd.isna(sig_date):
                    sig_date = None
                decision_inputs = DecisionInputs(
                    known=snapshot.to_known_inputs(
                        sig_date=sig_date,
                        observed_at={
                            "us_returns": f"{dt.date()} 09:00",
                            "jp_gap_returns": f"{dt.date()} 09:10",
                            "jp_betas": f"{dt.date()} 09:10",
                            "topix_night_return": f"{dt.date()} 09:10",
                            "current_prices": f"{dt.date()} 09:10",
                            "prev_closes": f"{dt.date()} 09:10",
                        },
                        source="backtest_pit",
                    ),
                    historical=historical_inputs,
                    gap_input_dir=effective_gap_dir,
                    use_file_cache=True,
                )
                result = v2_model.decide(
                    inputs=decision_inputs,
                    overlay_enabled=overlay_enabled,
                    use_file_cache=True,
                )
                # Research drivers inject their fitted transform explicitly.
                # A failed base decision must never be resurrected by an overlay.
                if decision_transform is not None and not (
                    result.fallback.get("gap_data_missing") or result.fallback.get("audit_failure")
                ):
                    result = decision_transform(date_str, result)
                    numerical = run_numerical_audit(result.w_final, result.scores, result.Omega_gap)
                    if numerical["status"] != "PASSED":
                        raise ValueError("Research decision transform failed numerical audit")
                    result = replace(result, numerical=numerical)
                w = result.w_final
                fb = (
                    result.fallback.get("gap_data_missing", False)
                    or result.fallback.get("audit_failure", False)
                )
                summary = result.summary
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
                    date_str = sim_dates_slice[i].strftime("%Y-%m-%d")
                    logger.debug("[%s] V2 fallback (gap data missing)", date_str)
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
                    date_str = sim_dates_slice[i].strftime("%Y-%m-%d")
                    logger.debug("[%s] V2 fallback (gap data missing)", date_str)
            logger.info(
                "V2 backtest: processed %d/%d dates (parallel, n_jobs=%d)",
                n_sim_days, n_sim_days, n_jobs,
            )

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
        drawdown = compute_drawdown_series(daily_returns_net)

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
