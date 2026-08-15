"""Next-Gen Lead-Lag Execution Pipeline.

Integrates:
  1. PIT Data Lake (Temporal Integrity)
  2. Pure On-Demand BLPX & Gap-Adjusted Distribution (< 10ms, Zero .npy Cache)
  3. Single-Stage Convex Optimization (cvxpy / SLSQP)
  4. Non-blocking Asynchronous Execution FSM (asyncio)

Acts as the modern, unified execution engine for the Lead-Lag quantitative strategy.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.broker.async_base import AsyncBrokerClient
from leadlag.config.schemas import AppConfig, ProductionV2RunConfig
from leadlag.core.convex_optimizer import (
    ConvexOptimizerConfig,
    OptimizationResult,
    optimize_portfolio_convex,
)
from leadlag.core.portfolio import get_rolling_pit_bin
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.async_fsm import AsyncExecutionEngine, ExecutionJournal
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.production_v2 import ProductionV2Model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NextGenDecisionResult:
    """Result container for a Next-Gen execution decision."""
    trade_date: str
    target_weights: dict[str, float]
    raw_weights_array: np.ndarray
    opt_result: OptimizationResult
    snapshot: MarketSnapshot
    journal: ExecutionJournal | None
    rule_d_multiplier: float
    rule_d_bin: str
    elapsed_seconds: float
    success: bool


class NextGenPipeline:
    """Next-Gen Production Pipeline Orchestrator."""

    def __init__(
        self,
        app_config: AppConfig,
        opt_config: ConvexOptimizerConfig | None = None,
    ) -> None:
        self.app_config = app_config
        self.v2_cfg: ProductionV2RunConfig = app_config.v2
        self.opt_config = opt_config or ConvexOptimizerConfig(
            lambda_risk=3.0,
            cost_bps=5.0,
            turnover_penalty=0.0001,
            max_single_weight=0.25,
            gross_target=self.v2_cfg.baseline_gross,
        )

        # Initialize core math models
        self.blpx_model = ProductionBLPXModel(app_config.model_dump())
        self.v2_model = ProductionV2Model(self.v2_cfg, blpx_model=self.blpx_model)
        self.fsm_engine = AsyncExecutionEngine(split_delay_seconds=1.0)

    def compute_decision(
        self,
        trade_date: str,
        lake: PITDataLake,
        w_prev: np.ndarray | None = None,
        pit_ir_history: list[float] | None = None,
    ) -> tuple[np.ndarray, OptimizationResult, float, str, MarketSnapshot]:
        """Compute optimal target weights synchronously without I/O blocking."""
        snapshot = lake.get_snapshot(trade_date)

        # 1. Compute on-demand distribution
        mu_gap, omega_gap = self.v2_model._compute_ondemand(
            trade_date=trade_date,
            df_exec=lake._df,
            current_prices=snapshot.current_prices,
            horizon=1,
        )

        # 2. Ex-ante IR & RuleD Dynamic Gross Scaling
        sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), 1e-8))
        raw_ir = float(np.mean(mu_gap / sigma_gap)) if len(mu_gap) > 0 else 0.0

        if pit_ir_history and len(pit_ir_history) >= self.v2_cfg.pit_rolling_window:
            bin_label, _, _, gross_mult = get_rolling_pit_bin(
                history_ir=np.array(pit_ir_history),
                current_ir=raw_ir,
                rolling_window=self.v2_cfg.pit_rolling_window,
                mult_low=self.v2_cfg.mult_low,
                mult_mid=self.v2_cfg.mult_mid,
                mult_high=self.v2_cfg.mult_high,
            )
        else:
            bin_label = "Medium"
            gross_mult = self.v2_cfg.fallback_multiplier

        # 3. Solve single-stage convex optimization
        opt_res = optimize_portfolio_convex(
            mu_gap=mu_gap,
            omega_gap=omega_gap,
            w_prev=w_prev,
            config=self.opt_config,
            gross_multiplier=gross_mult,
        )

        return opt_res.weights, opt_res, gross_mult, bin_label, snapshot

    async def run_daily_decision(
        self,
        trade_date: str,
        lake: PITDataLake,
        broker: AsyncBrokerClient,
        capital: float = 1_000_000.0,
        w_prev: np.ndarray | None = None,
        pit_ir_history: list[float] | None = None,
        submit_orders: bool = True,
    ) -> NextGenDecisionResult:
        """Run full daily decision pipeline with async order execution."""
        start_time = datetime.now()

        # Step 1: Compute optimal target weights
        weights, opt_res, gross_mult, bin_label, snapshot = self.compute_decision(
            trade_date=trade_date,
            lake=lake,
            w_prev=w_prev,
            pit_ir_history=pit_ir_history,
        )

        target_weights_dict = {tk: float(weights[i]) for i, tk in enumerate(JP_TICKERS)}

        # Step 2: Asynchronous execution
        journal: ExecutionJournal | None = None
        if submit_orders:
            journal = await self.fsm_engine.execute_portfolio(
                target_weights=weights,
                current_prices=snapshot.current_prices,
                total_capital=capital,
                broker=broker,
                trade_date=trade_date,
            )

        elapsed = (datetime.now() - start_time).total_seconds()
        success = journal.success if journal is not None else opt_res.converged

        return NextGenDecisionResult(
            trade_date=trade_date,
            target_weights=target_weights_dict,
            raw_weights_array=weights,
            opt_result=opt_res,
            snapshot=snapshot,
            journal=journal,
            rule_d_multiplier=gross_mult,
            rule_d_bin=bin_label,
            elapsed_seconds=elapsed,
            success=success,
        )
