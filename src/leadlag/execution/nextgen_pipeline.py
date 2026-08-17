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
import csv
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.broker.async_base import AsyncBrokerClient
from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.config.schemas import AppConfig, ProductionV2RunConfig
from leadlag.core.convex_optimizer import (
    ConvexOptimizerConfig,
    OptimizationResult,
    optimize_portfolio_convex,
)
from leadlag.core.market_calendar import previous_trading_day
from leadlag.core.portfolio import get_rolling_pit_bin, solve_baseline_style
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.tickers import JP_TICKERS, lot_size_for
from leadlag.execution.async_fsm import AsyncExecutionEngine, ExecutionJournal
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.production_v2 import VERSION, ProductionV2Model, _repair_and_adjust
from leadlag.reporting.formatter import print_text_orders as _print_text_orders

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
    mu_gap: np.ndarray | None = None
    sigma_gap: np.ndarray | None = None
    capital: float = 0.0
    leakage_audit: dict | None = None
    numerical_audit: dict | None = None


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
        self.fsm_engine = AsyncExecutionEngine(
            split_delay_seconds=self.opt_config.split_delay_seconds,
            order_timeout_seconds=self.opt_config.order_timeout_seconds,
            rate_limit_per_second=5.0,
            burst_limit=5,
        )
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

    def _compute_current_ir(
        self,
        mu_gap: np.ndarray,
        omega_gap: np.ndarray,
    ) -> float:
        """Compute V2-compatible portfolio ex-ante IR (net of expected cost).

        Uses the same baseline-style weights and cost formula as V2's
        ``_apply_pit_ruleD`` so that PIT binning is comparable.
        """
        n_j = len(mu_gap)
        if n_j == 0:
            return 0.0

        sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), 1e-8))
        scores = mu_gap / sigma_gap
        sorted_idx = np.argsort(scores)
        short_idx = sorted_idx[: self.v2_cfg.short_count]
        long_idx = sorted_idx[-self.v2_cfg.long_count :]
        w_pre = solve_baseline_style(
            scores,
            long_idx,
            short_idx,
            baseline_gross=self.v2_cfg.baseline_gross,
        )

        p_mean = float(np.dot(w_pre, mu_gap))
        p_var = float(np.dot(w_pre, np.dot(omega_gap, w_pre)))
        p_vol = float(np.sqrt(max(p_var, 0.0)))

        costs = getattr(self.v2_cfg, "costs", None)
        cost_bps_per_gross = getattr(costs, "cost_bps_per_gross", 10.0) if costs is not None else 10.0
        ex_ante_cost = self.v2_cfg.baseline_gross * (cost_bps_per_gross / 10000.0)

        if p_vol > 1e-8:
            return (p_mean - ex_ante_cost) / p_vol
        return 0.0

    def _audit_and_fallback(
        self,
        weights: np.ndarray,
        mu_gap: np.ndarray,
        omega_gap: np.ndarray,
        trade_date: str,
        snapshot: MarketSnapshot,
    ) -> tuple[np.ndarray, str, dict, dict]:
        """Run compliance/numerical audits and fall back to flat if required."""
        fallback = False
        message = ""

        # Numerical audit (finite weights, market neutral, gross, cov sanity)
        numerical = run_numerical_audit(weights, mu_gap, omega_gap)
        if numerical["status"] == "FAILED" and self.v2_cfg.fallback_on_audit_failure:
            logger.warning(
                "[%s] Numerical audit FAILED (%s). Falling back to flat position.",
                trade_date,
                numerical,
            )
            fallback = True
            message = f"Numerical audit FAILED: {numerical}"

        # Leakage audit
        # Signal date is the previous TSE trading day (US close / signal input date).
        sig_dt = previous_trading_day(pd.to_datetime(trade_date).to_pydatetime())
        sig_date = sig_dt.strftime("%Y-%m-%d")
        pit_dates = np.array([d.isoformat() for d, _ in self._pit_records])
        leakage = run_leakage_audit(
            sig_date=sig_date,
            trade_date=trade_date,
            gap_data_loaded=True,
            pit_history_trade_dates=pit_dates,
        )
        if leakage["status"] == "FAILED" and self.v2_cfg.fallback_on_audit_failure:
            logger.warning(
                "[%s] Leakage audit FAILED (%s). Falling back to flat position.",
                trade_date,
                leakage,
            )
            fallback = True
            message = f"{message}; Leakage audit FAILED: {leakage}".strip("; ")

        if fallback:
            return np.zeros_like(weights), message, leakage, numerical

        return weights, "", leakage, numerical

    def compute_decision(
        self,
        trade_date: str,
        lake: PITDataLake,
        w_prev: np.ndarray | None = None,
        pit_ir_history: list[float] | None = None,
        use_file_cache: bool = True,
    ) -> tuple[np.ndarray, OptimizationResult, float, str, MarketSnapshot, float, list[float], np.ndarray, np.ndarray, dict | None, dict | None]:
        """Compute optimal target weights synchronously without I/O blocking.

        Args:
            pit_ir_history: Optional externally-supplied history for PIT binning.
                Values must be on the same scale as this pipeline's ``raw_ir``
                (V2-style portfolio net IR). Passing V2 portfolio-IR history
                directly will produce consistent gross multipliers.
            use_file_cache: If True, prefer pre-computed gap matrices (production).
                Set to False for shadow runs that must be on-demand.
        """
        n_j = len(JP_TICKERS)
        snapshot = lake.get_snapshot(trade_date)

        # Sanity Guard: Check snapshot data validity (NaN, Inf, extreme outlier returns)
        is_valid, sanity_errors = snapshot.validate(max_abs_return=0.25)
        if not is_valid:
            logger.warning(
                "[%s] Snapshot sanity check failed with errors: %s. Falling back to flat position.",
                trade_date,
                "; ".join(sanity_errors),
            )
            opt_result = OptimizationResult(
                weights=np.zeros(n_j),
                gross_exposure=0.0,
                net_exposure=0.0,
                ex_ante_return=0.0,
                ex_ante_vol=0.0,
                ex_ante_ir=0.0,
                ex_ante_cost=0.0,
                ex_ante_net_return=0.0,
                ex_ante_net_ir=0.0,
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
                np.zeros(n_j),
                np.zeros(n_j),
                None,
                None,
            )

        try:
            # 1. Compute gap-adjusted distribution (file cache first, on-demand fallback)
            mu_gap, omega_gap = self.v2_model.compute_distribution(
                trade_date=trade_date,
                df_exec=lake.df_exec,
                current_prices=snapshot.current_prices,
                horizon=1,
                use_file_cache=use_file_cache,
            )
            # Ensure symmetry/PSD and apply the same macro adjustments as V2.
            date_str = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
            repair_alerts: list[str] = []
            mu_gap, omega_gap, _ = _repair_and_adjust(
                mu_gap,
                omega_gap,
                self.v2_cfg,
                date_str,
                n_j,
                repair_alerts,
                cache=self.v2_model._macro_price_cache,
            )
            sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), 1e-8))

            # 2. Ex-ante IR & RuleD Dynamic Gross Scaling (V2-compatible portfolio IR)
            raw_ir = self._compute_current_ir(mu_gap, omega_gap)

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

            # 3. Solve single-stage convex optimization
            opt_res = optimize_portfolio_convex(
                mu_gap=mu_gap,
                omega_gap=omega_gap,
                w_prev=w_prev,
                config=self.opt_config,
                gross_multiplier=gross_mult,
            )

            if not opt_res.converged:
                logger.warning(
                    "[%s] Convex optimization did not converge. Using flat fallback.",
                    trade_date,
                )
                return (
                    opt_res.weights,
                    opt_res,
                    gross_mult,
                    bin_label,
                    snapshot,
                    raw_ir,
                    history,
                    mu_gap,
                    sigma_gap,
                    None,
                    None,
                )

            # 4. Compliance audits (post-optimization, before persisting IR)
            audited_weights, audit_message, leakage, numerical = self._audit_and_fallback(
                opt_res.weights,
                mu_gap,
                omega_gap,
                trade_date,
                snapshot,
            )
            if not np.allclose(audited_weights, opt_res.weights, atol=1e-12):
                opt_result = OptimizationResult(
                    weights=np.zeros(n_j),
                    gross_exposure=0.0,
                    net_exposure=0.0,
                    ex_ante_return=0.0,
                    ex_ante_vol=0.0,
                    ex_ante_ir=0.0,
                    ex_ante_cost=0.0,
                    ex_ante_net_return=0.0,
                    ex_ante_net_ir=0.0,
                    turnover=float(np.sum(np.abs(w_prev))) if w_prev is not None else 0.0,
                    converged=False,
                    iterations=0,
                    message=f"Audit fallback: {audit_message}",
                )
                return (
                    np.zeros(n_j),
                    opt_result,
                    gross_mult,
                    bin_label,
                    snapshot,
                    raw_ir,
                    history,
                    mu_gap,
                    sigma_gap,
                    leakage,
                    numerical,
                )

            # 5. Persist PIT IR only after a successful audited decision
            if pit_ir_history is None:
                if self._pit_ir_history_path:
                    self._pit_records.append((current_date, raw_ir))
                    self._save_pit_records()
                else:
                    self._pit_ir_history.append(raw_ir)

            return (
                opt_res.weights,
                opt_res,
                gross_mult,
                bin_label,
                snapshot,
                raw_ir,
                history,
                mu_gap,
                sigma_gap,
                leakage,
                numerical,
            )

        except Exception as e:
            logger.error("[%s] Decision computation failed: %s. Falling back to flat.", trade_date, e, exc_info=True)
            opt_result = OptimizationResult(
                weights=np.zeros(n_j),
                gross_exposure=0.0,
                net_exposure=0.0,
                ex_ante_return=0.0,
                ex_ante_vol=0.0,
                ex_ante_ir=0.0,
                ex_ante_cost=0.0,
                ex_ante_net_return=0.0,
                ex_ante_net_ir=0.0,
                turnover=float(np.sum(np.abs(w_prev))) if w_prev is not None else 0.0,
                converged=False,
                iterations=0,
                message=f"Exception fallback: {e}",
            )
            return (
                np.zeros(n_j),
                opt_result,
                self.v2_cfg.fallback_multiplier,
                "Medium",
                snapshot,
                0.0,
                pit_ir_history if pit_ir_history is not None else [],
                np.zeros(n_j),
                np.zeros(n_j),
                None,
                None,
            )

    async def run_daily_decision(
        self,
        trade_date: str,
        lake: PITDataLake,
        broker: AsyncBrokerClient,
        capital: float = 1_000_000.0,
        w_prev: np.ndarray | None = None,
        pit_ir_history: list[float] | None = None,
        submit_orders: bool = True,
        use_file_cache: bool = True,
    ) -> NextGenDecisionResult:
        """Run full daily decision pipeline with async order execution.

        Args:
            pit_ir_history: Optional externally-supplied history for PIT binning.
                Must be on the same scale as ``NextGenPipeline``'s internal
                ``raw_ir`` metric (V2-style portfolio net IR).
            use_file_cache: If True, prefer pre-computed gap matrices (production).
                Set to False for on-demand shadow runs.
            submit_orders: If True, execute orders through the async broker.
        """
        start_time = datetime.now()

        # Step 1: Compute optimal target weights in a thread to avoid blocking
        # the asyncio event loop with CPU-heavy BLPX / SLSQP work.
        loop = asyncio.get_running_loop()
        (
            weights,
            opt_res,
            gross_mult,
            bin_label,
            snapshot,
            raw_ir,
            history,
            mu_gap,
            sigma_gap,
            leakage,
            numerical,
        ) = await loop.run_in_executor(
            None,
            self.compute_decision,
            trade_date,
            lake,
            w_prev,
            pit_ir_history,
            use_file_cache,
        )

        target_weights_dict = {tk: float(weights[i]) for i, tk in enumerate(JP_TICKERS)}

        # Step 2: Asynchronous execution
        # Do not submit orders if the decision is invalid, unaudited, or flat.
        journal: ExecutionJournal | None = None
        should_submit = (
            submit_orders
            and opt_res.converged
            and not bin_label == "SanityFallback"
            and snapshot.is_valid()
        )
        if should_submit:
            try:
                journal = await asyncio.wait_for(
                    self.fsm_engine.execute_portfolio(
                        target_weights=weights,
                        current_prices=snapshot.current_prices,
                        total_capital=capital,
                        broker=broker,
                        trade_date=trade_date,
                    ),
                    timeout=self.opt_config.execute_portfolio_timeout_seconds,
                )
            except TimeoutError:
                logger.error("[%s] Portfolio execution timed out.", trade_date)
            except Exception as e:
                logger.error("[%s] Portfolio execution failed: %s", trade_date, e, exc_info=True)

        elapsed = (datetime.now() - start_time).total_seconds()
        success = (
            (journal.success if journal is not None else not should_submit)
            and opt_res.converged
            and not bin_label == "SanityFallback"
        )

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
            mu_gap=mu_gap,
            sigma_gap=sigma_gap,
            capital=capital,
            leakage_audit=leakage,
            numerical_audit=numerical,
        )

    def _build_decision_df(self, result: NextGenDecisionResult) -> pd.DataFrame:
        """Build a V2-compatible decision DataFrame for text output and reporting."""
        rows = []
        for i, tk in enumerate(JP_TICKERS):
            w = result.target_weights.get(tk, 0.0)
            price = result.snapshot.current_prices.get(tk, 0.0)
            signal = (
                float(result.mu_gap[i] / max(result.sigma_gap[i], 1e-8))
                if result.mu_gap is not None and result.sigma_gap is not None
                else 0.0
            )
            if w > 1e-8:
                action = "BUY"
            elif w < -1e-8:
                action = "SELL"
            else:
                action = "HOLD"

            if price > 0 and abs(w) > 1e-8:
                lot = lot_size_for(tk)
                target_value = abs(w) * result.capital
                qty_raw = int(np.floor(target_value / price + 0.5))
                qty = (qty_raw // lot) * lot
                if qty <= 0:
                    qty = lot
                quantity = qty if w > 0 else -qty
            else:
                quantity = 0

            etf_amount = abs(quantity) * price
            rows.append({
                "ticker": tk,
                "open_price": price,
                "signal": signal,
                "weight": float(w),
                "action": action,
                "quantity": quantity,
                "etf_amount": etf_amount,
            })
        return pd.DataFrame(rows)

    def save_decision_artifacts(
        self,
        result: NextGenDecisionResult,
        output_root: Path | str,
        *,
        run_tag: str | None = None,
        live_dir: Path | str | None = None,
        dry_run: bool = False,
        text_output: bool = False,
    ) -> Path:
        """Persist Next-Gen decision artifacts (latest_weights.csv, summary.json, journal.json)."""
        out_path = Path(output_root)
        if run_tag:
            out_path = out_path / run_tag
        out_path = out_path / f"decision_{result.trade_date.replace('-', '')}"
        out_path.mkdir(parents=True, exist_ok=True)

        live_path = Path(live_dir) if live_dir else None

        if dry_run:
            logger.info("[DRY-RUN] Would write decision artifacts to: %s", out_path)
            if live_path:
                logger.info("[DRY-RUN] Would write live artifacts to: %s", live_path)
            if text_output:
                _print_text_orders(self._build_decision_df(result))
            return out_path

        # 1. Save latest_weights.csv (V2-compatible columns)
        weights_rows = []
        for i, tk in enumerate(JP_TICKERS):
            w = float(result.target_weights.get(tk, 0.0))
            side = "LONG" if w > 1e-8 else ("SHORT" if w < -1e-8 else "NEUTRAL")
            score = (
                float(result.mu_gap[i] / max(result.sigma_gap[i], 1e-8))
                if result.mu_gap is not None and result.sigma_gap is not None
                else 0.0
            )
            mu_gap = float(result.mu_gap[i]) if result.mu_gap is not None else 0.0
            sigma_gap = float(result.sigma_gap[i]) if result.sigma_gap is not None else 0.0
            weights_rows.append({
                "trade_date": result.trade_date,
                "ticker": tk,
                "weight": w,
                "side": side,
                "score": score,
                "mu_gap": mu_gap,
                "sigma_gap": sigma_gap,
                "ensemble_signal": score,
                "gross_multiplier": result.rule_d_multiplier,
                "pit_bin": result.rule_d_bin,
                "version": VERSION,
                "fallback_flag": 0 if result.success else 1,
            })

        weights_df = pd.DataFrame(weights_rows)
        weights_file = out_path / "latest_weights.csv"
        weights_df.to_csv(weights_file, index=False)

        # 1b. production_scores.csv
        score_rows = [
            {
                "trade_date": result.trade_date,
                "ticker": JP_TICKERS[i],
                "mu_gap": float(result.mu_gap[i]) if result.mu_gap is not None else 0.0,
                "sigma_gap": float(result.sigma_gap[i]) if result.sigma_gap is not None else 0.0,
                "mu_over_sigma_score": float(result.mu_gap[i] / max(result.sigma_gap[i], 1e-8))
                if result.mu_gap is not None and result.sigma_gap is not None
                else 0.0,
            }
            for i in range(len(JP_TICKERS))
        ]
        score_df = pd.DataFrame(score_rows)
        score_file = out_path / "production_scores.csv"
        score_df.to_csv(score_file, index=False)

        # 2. Save decision_summary.json
        summary_file = out_path / "decision_summary.json"
        summary_data = {
            "trade_date": result.trade_date,
            "success": result.success,
            "gross_exposure": result.opt_result.gross_exposure,
            "net_exposure": result.opt_result.net_exposure,
            "rule_d_multiplier": result.rule_d_multiplier,
            "rule_d_bin": result.rule_d_bin,
            "raw_ir": result.raw_ir,
            "ex_ante_return_bps": result.opt_result.ex_ante_return * 10000.0,
            "ex_ante_vol_bps": result.opt_result.ex_ante_vol * 10000.0,
            "ex_ante_ir": result.opt_result.ex_ante_ir,
            "turnover": result.opt_result.turnover,
            "elapsed_seconds": result.elapsed_seconds,
            "converged": result.opt_result.converged,
            "message": result.opt_result.message,
        }
        with summary_file.open("w") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # 3. PIT binning / audit files
        pit = {
            "trade_date": result.trade_date,
            "assigned_bin": result.rule_d_bin,
            "multiplier": result.rule_d_multiplier,
            "raw_ir": result.raw_ir,
            "pit_history_count": result.pit_history_count,
        }
        with (out_path / "pit_binning.json").open("w") as f:
            json.dump(pit, f, indent=2, ensure_ascii=False)

        if result.leakage_audit is not None:
            with (out_path / "leakage_audit.json").open("w") as f:
                json.dump(result.leakage_audit, f, indent=2, ensure_ascii=False)

        if result.numerical_audit is not None:
            with (out_path / "numerical_audit.json").open("w") as f:
                json.dump(result.numerical_audit, f, indent=2, ensure_ascii=False)

        all_passed = (
            result.success
            and (result.leakage_audit or {}).get("status") == "PASSED"
            and (result.numerical_audit or {}).get("status") == "PASSED"
        )
        production_audit = {
            "trade_date": result.trade_date,
            "all_passed": all_passed,
            "leakage": result.leakage_audit,
            "numerical": result.numerical_audit,
            "converged": result.opt_result.converged,
        }
        with (out_path / "production_audit.json").open("w") as f:
            json.dump(production_audit, f, indent=2, ensure_ascii=False)

        # 4. Save execution journal if available
        if result.journal is not None:
            journal_file = out_path / "execution_journal.json"
            journal_data = {
                "trade_date": result.journal.trade_date,
                "total_orders": result.journal.total_orders,
                "filled_orders": result.journal.filled_orders,
                "failed_orders": result.journal.failed_orders,
                "close_orders_count": result.journal.close_orders_count,
                "new_orders_count": result.journal.new_orders_count,
                "elapsed_seconds": result.journal.elapsed_seconds,
                "success": result.journal.success,
                "orders": [
                    {
                        "ticker": lc.order.ticker,
                        "side": lc.order.side.value if hasattr(lc.order.side, "value") else str(lc.order.side),
                        "quantity": lc.order.quantity,
                        "state": lc.state.name,
                        "message": lc.error_message,
                    }
                    for lc in result.journal.lifecycles
                ],
            }
            with journal_file.open("w") as f:
                json.dump(journal_data, f, indent=2, ensure_ascii=False)

        # 5. Mirror primary live artifacts to live_dir if configured
        if live_path:
            live_path.mkdir(parents=True, exist_ok=True)
            weights_df.to_csv(live_path / "latest_weights.csv", index=False)
            score_df.to_csv(live_path / "production_scores.csv", index=False)
            with (live_path / "decision_summary.json").open("w") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            with (live_path / "pit_binning.json").open("w") as f:
                json.dump(pit, f, indent=2, ensure_ascii=False)
            if result.leakage_audit is not None:
                with (live_path / "leakage_audit.json").open("w") as f:
                    json.dump(result.leakage_audit, f, indent=2, ensure_ascii=False)
            if result.numerical_audit is not None:
                with (live_path / "numerical_audit.json").open("w") as f:
                    json.dump(result.numerical_audit, f, indent=2, ensure_ascii=False)
            with (live_path / "production_audit.json").open("w") as f:
                json.dump(production_audit, f, indent=2, ensure_ascii=False)
            if result.journal is not None:
                with (live_path / "execution_journal.json").open("w") as f:
                    json.dump(journal_data, f, indent=2, ensure_ascii=False)

        if text_output:
            _print_text_orders(self._build_decision_df(result))

        logger.info("[%s] Saved decision artifacts to %s", result.trade_date, out_path)
        if live_path:
            logger.info("[%s] Mirrored live artifacts to %s", result.trade_date, live_path)
        return out_path
