"""Next-Gen Lead-Lag Execution Pipeline.

Integrates:
  1. PIT Data Lake (Temporal Integrity)
  2. Pure On-Demand BLPX & Gap-Adjusted Distribution (< 10ms, Zero .npy Cache)
  3. Single-Stage Convex Optimization (cvxpy / SLSQP)
  4. Non-blocking Asynchronous Execution FSM (asyncio)

Acts as the modern, unified execution engine for the Lead-Lag quantitative strategy.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

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
    raw_ir: float
    pit_history_count: int
    elapsed_seconds: float
    success: bool


class NextGenPipeline:
    """Next-Gen Production Pipeline Orchestrator."""

    def __init__(
        self,
        app_config: AppConfig,
        opt_config: ConvexOptimizerConfig | None = None,
        pit_ir_history_path: str | Path | None = None,
    ) -> None:
        self.app_config = app_config
        self.v2_cfg: ProductionV2RunConfig = app_config.v2
        self.opt_config: ConvexOptimizerConfig = opt_config if opt_config is not None else app_config.nextgen

        # Initialize core math models
        # NOTE: ProductionBLPXModel expects the v2 config (not the full AppConfig)
        # so that nested blpx/costs settings are resolved correctly.
        self.blpx_model = ProductionBLPXModel(self.v2_cfg.model_dump())
        self.v2_model = ProductionV2Model(self.v2_cfg, blpx_model=self.blpx_model)
        self.fsm_engine = AsyncExecutionEngine(split_delay_seconds=1.0)
        self._pit_ir_history_path: Path | None = (
            Path(pit_ir_history_path) if pit_ir_history_path else None
        )
        self._pit_records: list[tuple[date, float]] = self._load_pit_records()
        # In-memory history used only when no file path is configured.
        self._pit_ir_history: list[float] = []

    def _load_pit_records(self) -> list[tuple[date, float]]:
        """Load persisted PIT IR records from the configured path."""
        if self._pit_ir_history_path is None or not self._pit_ir_history_path.exists():
            return []
        records: list[tuple[date, float]] = []
        try:
            with self._pit_ir_history_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = datetime.strptime(row["trade_date"], "%Y-%m-%d").date()
                    records.append((d, float(row["raw_ir"])))
        except Exception:
            logger.warning("Failed to load PIT IR history from %s", self._pit_ir_history_path, exc_info=True)
        return records

    def _save_pit_records(self) -> None:
        """Persist PIT IR records to the configured path."""
        if self._pit_ir_history_path is None:
            return
        try:
            self._pit_ir_history_path.parent.mkdir(parents=True, exist_ok=True)
            with self._pit_ir_history_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["trade_date", "raw_ir"])
                writer.writeheader()
                for d, raw_ir in self._pit_records:
                    writer.writerow({"trade_date": d.isoformat(), "raw_ir": f"{raw_ir:.6f}"})
        except Exception:
            logger.warning("Failed to save PIT IR history to %s", self._pit_ir_history_path, exc_info=True)

    def compute_decision(
        self,
        trade_date: str,
        lake: PITDataLake,
        w_prev: np.ndarray | None = None,
        pit_ir_history: list[float] | None = None,
    ) -> tuple[np.ndarray, OptimizationResult, float, str, MarketSnapshot, float, list[float]]:
        """Compute optimal target weights synchronously without I/O blocking.

        Args:
            pit_ir_history: Optional externally-supplied history for PIT binning.
                Values must be on the same scale as this pipeline's ``raw_ir``
                (mean of mu_gap / sigma_gap). Passing V2 portfolio-IR history
                directly will misalign the percentiles.
        """
        snapshot = lake.get_snapshot(trade_date)

        # Sanity Guard: Check snapshot data validity (NaN, Inf, extreme outlier returns)
        is_valid, sanity_errors = snapshot.validate(max_abs_return=0.25)
        if not is_valid:
            logger.warning(
                "[%s] Snapshot sanity check failed with errors: %s. Falling back to flat position.",
                trade_date,
                "; ".join(sanity_errors),
            )
            n_j = len(JP_TICKERS)
            opt_result = OptimizationResult(
                weights=np.zeros(n_j),
                gross_exposure=0.0,
                net_exposure=0.0,
                ex_ante_return=0.0,
                ex_ante_vol=0.0,
                ex_ante_ir=0.0,
                turnover=float(np.sum(np.abs(w_prev))) if w_prev is not None else 0.0,
                converged=False,
                iterations=0,
                message=f"Sanity check failed: {sanity_errors}",
            )
            return (
                np.zeros(n_j),
                opt_result,
                0.0,
                "SanityFallback",
                snapshot,
                0.0,
                pit_ir_history if pit_ir_history is not None else [],
            )

        # 1. Compute gap-adjusted distribution (file cache first, on-demand fallback)
        mu_gap, omega_gap = self.v2_model.compute_distribution(
            trade_date=trade_date,
            df_exec=lake.df_exec,
            current_prices=snapshot.current_prices,
            horizon=1,
            use_file_cache=True,
        )

        # 2. Ex-ante IR & RuleD Dynamic Gross Scaling
        sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), 1e-8))
        raw_ir = float(np.mean(mu_gap / sigma_gap)) if len(mu_gap) > 0 else 0.0

        # Maintain PIT IR history.  Compute binning using the history **before**
        # appending the current raw_ir, so percentile thresholds are strictly historical.
        current_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        if pit_ir_history is not None:
            history = pit_ir_history
        elif self._pit_ir_history_path:
            history = [ir for d, ir in self._pit_records if d < current_date]
        else:
            history = self._pit_ir_history

        if history and len(history) >= self.v2_cfg.pit_rolling_window:
            bin_label, _, _, gross_mult = get_rolling_pit_bin(
                history_ir=np.array(history),
                current_ir=raw_ir,
                rolling_window=self.v2_cfg.pit_rolling_window,
                low_pct=self.v2_cfg.tertile_low_pct,
                high_pct=self.v2_cfg.tertile_high_pct,
                mult_low=self.v2_cfg.mult_low,
                mult_mid=self.v2_cfg.mult_mid,
                mult_high=self.v2_cfg.mult_high,
            )
        else:
            bin_label = "Medium"
            gross_mult = self.v2_cfg.fallback_multiplier

        if pit_ir_history is None:
            if self._pit_ir_history_path:
                self._pit_records.append((current_date, raw_ir))
                self._save_pit_records()
            else:
                self._pit_ir_history.append(raw_ir)

        # 3. Solve single-stage convex optimization
        opt_res = optimize_portfolio_convex(
            mu_gap=mu_gap,
            omega_gap=omega_gap,
            w_prev=w_prev,
            config=self.opt_config,
            gross_multiplier=gross_mult,
        )

        return opt_res.weights, opt_res, gross_mult, bin_label, snapshot, raw_ir, history

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
        """Run full daily decision pipeline with async order execution.

        Args:
            pit_ir_history: Optional externally-supplied history for PIT binning.
                Must be on the same scale as ``NextGenPipeline``'s internal
                ``raw_ir`` metric; passing V2 ``current_ir`` values directly
                will produce incorrect gross multipliers.
        """
        start_time = datetime.now()

        # Step 1: Compute optimal target weights
        weights, opt_res, gross_mult, bin_label, snapshot, raw_ir, history = self.compute_decision(
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
            raw_ir=raw_ir,
            pit_history_count=len(history),
            elapsed_seconds=elapsed,
            success=success,
        )
