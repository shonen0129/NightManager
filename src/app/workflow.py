"""Workflow: orchestrates the trade decision and execution flow."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from config import N_US_ASSETS, N_JP_ASSETS
from data_loader import download_data, preprocess_data
from domain.models.types import (
    RiskConfig,
    StrategyConfig,
    TradeAction,
    TradeDecision,
    RiskReport,
)
from domain.signals import lead_lag as signals
from domain.portfolio import optimizer as portfolio
from domain.portfolio import allocator as capital_alloc
from domain.risk import metrics as risk_metrics
from infrastructure.storage.cache_repo import CacheRepository
from infrastructure.execution.engine import ExecutionEngine, build_orders_from_decision

logger = logging.getLogger(__name__)


class TradeWorkflow:
    """Orchestrates the full trade decision workflow.

    Steps:
    1. Load market data
    2. Build execution dataset
    3. Compute signal
    4. Build trade decision
    5. Run risk checks
    6. Allocate capital
    7. Submit orders (optional)
    """

    def __init__(
        self,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
        output_dir: str,
        cache_dir: str | None = None,
    ):
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.output_dir = output_dir
        self.cache_dir = cache_dir or os.path.join(output_dir, ".cache")
        self.cache_repo = CacheRepository(self.cache_dir)

    def run_decision(
        self,
        trade_date: pd.Timestamp,
        open_prices: dict[str, float],
        max_capital: float,
        api_client=None,
        dry_run: bool = False,
        gap_override: np.ndarray | None = None,
        topix_night_override: float | None = None,
    ) -> dict:
        """Run the full decision workflow.

        Args:
            trade_date: Trade date
            open_prices: Dict mapping ticker -> open price
            max_capital: Available capital in JPY
            api_client: Optional KabuClient for API submission
            dry_run: If True, simulate orders
            gap_override: Optional gap return override

        Returns:
            Dict with decision_df, risk_report, allocation, and execution results
        """
        # Step 1: Load data
        logger.info("[1/5] Loading market data...")
        data = download_data(beta_window=self.strategy_config.beta_window)
        df_exec = preprocess_data(data, beta_window=self.strategy_config.beta_window)

        # Step 2: Build signal
        logger.info("[2/5] Computing signal...")
        signal_result = self._compute_signal(
            df_exec,
            trade_date,
            gap_override,
            topix_night_override,
        )

        # Step 3: Build trade decision
        logger.info("[3/5] Building trade decision...")
        decision = self._build_decision(signal_result, trade_date)

        # Step 4: Risk checks
        logger.info("[4/5] Running risk checks...")
        hist_returns = self._get_hist_returns(df_exec, trade_date)
        risk_report = self._run_risk_checks(decision, hist_returns, max_capital)

        if risk_report.is_blocked:
            raise RuntimeError("Risk stop threshold breached. See [RISK-STOP] logs.")

        # Step 5: Allocate capital and build orders
        logger.info("[5/5] Allocating capital...")
        allocation = capital_alloc.allocate_capital(
            decision.weights,
            decision.tickers,
            open_prices,
            max_capital,
            max_net_exposure=self.risk_config.max_net_exposure,
        )

        # Build decision DataFrame
        decision_df = pd.DataFrame(
            {
                "ticker": decision.tickers,
                "open_price": [open_prices.get(tk, 0.0) for tk in decision.tickers],
                "signal": decision.signals,
                "weight": decision.weights,
                "action": [
                    a.value if isinstance(a, TradeAction) else a
                    for a in decision.actions
                ],
                "etf_amount": allocation.allocated_amounts,
                "quantity": allocation.quantities,
            }
        )

        # Submit orders if API client is provided
        execution_results = None
        if api_client is not None:
            engine = ExecutionEngine(api_client, dry_run=dry_run)
            orders = build_orders_from_decision(decision_df)
            execution_results = engine.submit_orders_batch(orders, self.output_dir)

        return {
            "decision_df": decision_df,
            "risk_report": risk_report,
            "allocation": allocation,
            "execution_results": execution_results,
            "decision": decision,
        }

    def _compute_signal(
        self,
        df_exec: pd.DataFrame,
        trade_date: pd.Timestamp,
        gap_override: np.ndarray | None,
        topix_night_override: float | None,
    ) -> dict:
        """Compute the signal for the trade date."""
        all_cc_cols = [
            c
            for c in df_exec.columns
            if c.startswith("us_cc_") or c.startswith("jp_cc_")
        ]
        all_returns = df_exec[all_cc_cols].values
        date_index = df_exec.index.values

        n_u = N_US_ASSETS
        n_j = N_JP_ASSETS
        cfg = self.strategy_config

        # Pre-compute baseline correlation
        c_full = signals.compute_baseline_correlation(
            all_returns,
            date_index,
            cfg.ewma_half_life,
        )

        # Build V0 static
        v0_static = signals.build_v3_static(n_u, n_j, cfg.include_v4_prior)
        base_vectors = signals.build_base_vectors(n_u, n_j)
        v1, v2 = base_vectors["v1"], base_vectors["v2"]

        # Find index for trade date
        try:
            idx = df_exec.index.get_loc(trade_date)
        except KeyError:
            # Use last row if trade_date not found
            idx = len(df_exec) - 1

        # Compute dispersion history
        dispersion_history = []
        gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
        beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]
        topix_night_series = (
            df_exec["topix_night_return"]
            if "topix_night_return" in df_exec.columns
            else None
        )
        for hist_i in range(max(0, idx - 60), idx):
            gap_hist = None
            if cfg.signal_mode == "gap_residual" and len(gap_cols) == n_j:
                gap_hist = np.nan_to_num(
                    df_exec.iloc[hist_i][gap_cols].values,
                    nan=0.0,
                    copy=True,
                ).astype(float, copy=False)
            betas_hist = None
            topix_night_hist = None
            if beta_cols and len(beta_cols) == n_j:
                betas_hist = np.asarray(
                    df_exec.iloc[hist_i][beta_cols].values,
                    dtype=float,
                )
            if topix_night_series is not None:
                topix_night_hist = float(topix_night_series.iloc[hist_i])
            sig = signals.compute_signal(
                all_returns,
                hist_i,
                n_u,
                cfg.corr_window,
                c_full,
                v0_static,
                v1,
                v2,
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
            )
            sig_signal = np.asarray(sig["signal"], dtype=float)
            disp = signals.compute_dispersion_indicator(
                sig_signal,
                cfg.q,
                n_j,
                cfg.dispersion_metric,
            )
            dispersion_history.append(disp)

        # Compute current signal
        gap_arr = (
            np.asarray(gap_override, dtype=float) if gap_override is not None else None
        )
        if (
            gap_arr is None
            and cfg.signal_mode == "gap_residual"
            and len(gap_cols) == n_j
        ):
            gap_arr = np.nan_to_num(
                df_exec.iloc[idx][gap_cols].values,
                nan=0.0,
                copy=True,
            ).astype(float, copy=False)
        betas_t = None
        topix_night_t = None
        if beta_cols and len(beta_cols) == n_j:
            betas_t = np.asarray(
                df_exec.iloc[idx][beta_cols].values,
                dtype=float,
            )
        if topix_night_series is not None:
            topix_night_t = float(topix_night_series.iloc[idx])
        if topix_night_override is not None:
            if (
                trade_date not in df_exec.index
                or topix_night_t is None
                or not np.isfinite(topix_night_t)
            ):
                topix_night_t = float(topix_night_override)
        result = signals.compute_signal(
            all_returns,
            idx,
            n_u,
            cfg.corr_window,
            c_full,
            v0_static,
            v1,
            v2,
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
        )

        result_dict: dict[str, object] = dict(result)
        result_dict["dispersion_history"] = dispersion_history
        result_dict["n_j"] = n_j
        return result_dict

    def _build_decision(
        self,
        signal_result: dict,
        trade_date: pd.Timestamp,
    ) -> TradeDecision:
        """Build TradeDecision from signal result."""
        cfg = self.strategy_config
        n_j = signal_result["n_j"]
        dispersion_history = signal_result["dispersion_history"]

        if cfg.signal_mode == "gap_tolerant":
            # Gap tolerant mode needs close/open prices
            decision = portfolio.compute_trade_decision(
                signal=signal_result["signal"],
                sigma_s=signal_result["sigma_s"],
                n_j=n_j,
                q=cfg.q,
                weight_mode=cfg.weight_mode,
                dispersion_filter=cfg.dispersion_filter,
                dispersion_metric=cfg.dispersion_metric,
                dispersion_history=dispersion_history,
                gap_tolerant=True,
                gamma=cfg.gamma,
            )
        else:
            decision = portfolio.compute_trade_decision(
                signal=signal_result["signal"],
                sigma_s=signal_result["sigma_s"],
                n_j=n_j,
                q=cfg.q,
                weight_mode=cfg.weight_mode,
                dispersion_filter=cfg.dispersion_filter,
                dispersion_metric=cfg.dispersion_metric,
                dispersion_history=dispersion_history,
            )

        # Apply gross exposure adjustment
        gross_adj = portfolio.adjust_gross_exposure(
            decision["weights"],
            self.risk_config.max_gross_exposure,
        )
        if gross_adj.was_adjusted:
            decision["weights"] = decision["weights"] * gross_adj.adjustment_factor
            logger.info(
                f"Gross auto-adjust applied: before={gross_adj.gross_before:.6f}, "
                f"after={gross_adj.gross_after:.6f}, factor={gross_adj.adjustment_factor:.6f}"
            )

        actions = portfolio.classify_actions(decision["weights"])
        tickers = [f"{t}.T" for t in range(1617, 1617 + n_j)]

        return TradeDecision(
            trade_date=trade_date,
            tickers=tickers,
            signals=decision["signal"],
            raw_weights=decision["raw_weights"],
            scale=decision["scale"],
            weights=decision["weights"],
            actions=[TradeAction(a) for a in actions],
            sigma_s=decision["sigma_s"],
            dispersion_indicator=decision["dispersion_indicator"],
            dispersion_metric=cfg.dispersion_metric,
        )

    def _get_hist_returns(
        self,
        df_exec: pd.DataFrame,
        trade_date: pd.Timestamp,
    ) -> pd.Series:
        """Get historical daily returns for VaR/ES."""
        hist_returns = self.cache_repo.read_daily_returns(trade_date)
        if hist_returns is not None:
            return hist_returns

        # No cache: run backtest to build history
        logger.info("No return cache found; running backtest for VaR/ES...")
        from backtest.runner import run_backtest_with_config

        bt_results = run_backtest_with_config(df_exec, self.strategy_config)
        self.cache_repo.write_daily_returns(bt_results)
        return bt_results.loc[bt_results.index < trade_date, "daily_return"]

    def _run_risk_checks(
        self,
        decision: TradeDecision,
        hist_returns: pd.Series,
        max_capital: float,
    ) -> RiskReport:
        """Run risk checks on the decision."""
        # Calculate allocated amounts (simplified: use weights * capital)
        buy_mask = np.array([a == TradeAction.BUY for a in decision.actions])
        sell_mask = np.array([a == TradeAction.SELL for a in decision.actions])
        total_buy = (
            float(np.sum(np.abs(decision.weights[buy_mask]))) * max_capital
            if max_capital > 0
            else 0.0
        )
        total_sell = (
            float(np.sum(np.abs(decision.weights[sell_mask]))) * max_capital
            if max_capital > 0
            else 0.0
        )

        return risk_metrics.evaluate_risk_checks(
            weights=decision.weights,
            total_buy_allocated=total_buy,
            total_sell_allocated=total_sell,
            max_capital=max_capital,
            hist_daily_returns=hist_returns,
            config=self.risk_config,
        )
