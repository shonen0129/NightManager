"""Post-decision orchestration helpers.

This module was split from ``execution/helpers.py`` as part of P1-B1 to
isolate the end-of-day decision flow (gross adjustment, risk check,
capital allocation, order submission, and output writing) from standalone
output, broker, pricing, and risk helpers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.config.schemas import StrategyConfig as ProductionConfig
from leadlag.core import allocator as domain_allocator
from leadlag.execution.broker_ops import (
    OrderExecutionIncomplete,
    build_execution_plan,
    submit_orders_via_api,
)
from leadlag.execution.contracts import report_from_records
from leadlag.execution.output_ops import (
    save_daily_journal,
    save_decision_output,
    save_position_snapshot,
    save_wallet_snapshot,
)
from leadlag.execution.pricing import fetch_fill_prices
from leadlag.execution.risk_capital import (
    allocate_capital,
    auto_adjust_gross_exposure,
    run_risk_checks,
)
from leadlag.execution.state_store import ExecutionStateStore
from leadlag.reporting.formatter import (
    log_decision_summary as _log_decision_summary,
)
from leadlag.reporting.formatter import (
    print_risk_report as _print_risk_report,
)
from leadlag.reporting.formatter import (
    print_text_orders as _print_text_orders,
)

logger = logging.getLogger(__name__)


def _prepare_decision_df(
    decision: dict,
    config: ProductionConfig,
    manual_opens: dict,
    max_capital: float,
) -> tuple[pd.DataFrame, dict]:
    """Apply gross exposure adjustment, map actions, allocate, and build decision_df.

    Returns:
        (decision_df, capital_alloc)
    """
    adjusted = auto_adjust_gross_exposure(decision, config)
    # Preserve the adjusted state in the same decision dict so downstream helpers see it.
    decision.update(adjusted)

    actions_mapped: list[str] = []
    for side in decision["action"]:
        if side in ("LONG", "BUY"):
            actions_mapped.append("BUY")
        elif side in ("SHORT", "SELL"):
            actions_mapped.append("SELL")
        else:
            actions_mapped.append("HOLD")
    decision["action"] = actions_mapped

    capital_alloc = allocate_capital(
        decision,
        manual_opens,
        max_capital,
        max_net_exposure=config.max_net_exposure,
        side_leverage=getattr(config, "side_leverage", domain_allocator.DEFAULT_SIDE_LEVERAGE),
    )

    decision_df = pd.DataFrame(
        {
            "ticker": decision["tickers"],
            "open_price": [manual_opens[tk] for tk in decision["tickers"]],
            "signal": decision["signal"],
            "weight": decision["weight"],
            "action": decision["action"],
            "etf_amount": capital_alloc["allocated"],
            "quantity": capital_alloc["qty"],
        }
    )
    # Keep the trade date at the typed execution boundary without adding a
    # transient column to the public CSV schema.
    decision_df.attrs["trade_date"] = str(decision.get("trade_date", ""))

    return decision_df, capital_alloc


def _log_decision_allocations(
    decision_df: pd.DataFrame,
    capital_alloc: dict,
    max_capital: float,
    decision: dict,
) -> None:
    """Log the decision summary, gross adjustment, and target/allocated budgets."""
    _log_decision_summary(decision_df, decision)
    if decision.get("gross_adjusted", False):
        logger.info(
            "Gross auto-adjust applied: before=%.6f, after=%.6f, factor=%.6f",
            decision["gross_before"],
            decision["gross_after"],
            decision["gross_adjustment_factor"],
        )
    logger.info(
        "Equity capital (used for sizing): %s JPY "
        "(margin assumed: long+short notionals can exceed equity)",
        f"{max_capital:,.0f}",
    )

    buy_mask = decision_df["action"] == "BUY"
    sell_mask = decision_df["action"] == "SELL"
    total_buy_allocated = float(decision_df.loc[buy_mask, "etf_amount"].sum())
    total_sell_allocated = float(decision_df.loc[sell_mask, "etf_amount"].sum())
    total_gross_allocated = total_buy_allocated + total_sell_allocated
    total_net_allocated = total_buy_allocated - total_sell_allocated
    gross_budget = float(
        capital_alloc.get(
            "gross_budget", capital_alloc["buy_budget"] + capital_alloc["sell_budget"]
        )
    )

    logger.info("Target BUY budget: %s JPY", f"{capital_alloc['buy_budget']:,.0f}")
    logger.info("Target SELL budget: %s JPY", f"{capital_alloc['sell_budget']:,.0f}")
    logger.info("Allocated BUY notional: %s JPY", f"{total_buy_allocated:,.0f}")
    logger.info("Allocated SELL notional: %s JPY", f"{total_sell_allocated:,.0f}")
    logger.info("Allocated gross notional: %s JPY", f"{total_gross_allocated:,.0f}")
    logger.info("Allocated net notional: %s JPY", f"{total_net_allocated:,.0f}")
    logger.info("Unallocated gross budget: %s JPY", f"{gross_budget - total_gross_allocated:,.0f}")


def _run_risk_check_and_print(
    decision: dict,
    decision_df: pd.DataFrame,
    max_capital: float,
    hist_returns: pd.Series,
    config: ProductionConfig,
    current_positions: dict[str, int] | None = None,
) -> dict:
    """Compute allocated totals, run risk checks, print the report, and return it."""
    buy_mask = decision_df["action"] == "BUY"
    sell_mask = decision_df["action"] == "SELL"
    total_buy_allocated = float(decision_df.loc[buy_mask, "etf_amount"].sum())
    total_sell_allocated = float(decision_df.loc[sell_mask, "etf_amount"].sum())

    risk_report = run_risk_checks(
        decision=decision,
        total_buy_allocated=total_buy_allocated,
        total_sell_allocated=total_sell_allocated,
        max_capital=max_capital,
        hist_daily_returns=hist_returns,
        config=config,
    )
    _print_risk_report(risk_report)
    if risk_report["is_blocked"]:
        # A risk stop must prevent new risk, but it must not prevent reducing
        # already confirmed exposure.  A reversal is not reduction-only: it
        # closes one side and opens the other, so it remains blocked.
        reduction_only = (
            current_positions is not None
            and bool(current_positions)
            and _is_reduction_only(decision_df, current_positions)
        )
        if reduction_only:
            logger.warning(
                "[RISK-STOP] New exposure blocked; allowing reduction-only orders "
                "for confirmed existing positions."
            )
            return risk_report
        raise RuntimeError(
            "Risk stop threshold breached; order submission blocked. See [RISK-STOP] logs above."
        )
    return risk_report


def _is_reduction_only(
    decision_df: pd.DataFrame,
    current_positions: dict[str, int],
) -> bool:
    """Return whether a target can only reduce confirmed signed inventory."""
    for _, row in decision_df.iterrows():
        ticker = str(row["ticker"])
        target_qty = int(row.get("quantity", 0) or 0)
        action = str(row.get("action", "HOLD"))
        target = target_qty if action == "BUY" else -target_qty if action == "SELL" else 0
        current = int(current_positions.get(ticker, 0) or 0)
        if current == 0 and target != 0:
            return False
        if target != 0 and (target * current <= 0 or abs(target) > abs(current)):
            return False
    return True


def _write_decision_output_and_submit(
    decision_df: pd.DataFrame,
    decision: dict,
    output_dir: str | Path,
    text_output: bool,
    api_client: BrokerClient | None,
    current_positions: dict[str, int] | None,
    state_store: ExecutionStateStore | None = None,
    account_key: str = "default",
    strategy_key: str = "production_v2",
) -> str:
    """Write decision CSV, optionally print text, submit orders, and save journal.

    Returns:
        Path to the decision output CSV.
    """
    output_dir = Path(output_dir)
    logger.info("[4/4] Writing decision artifact...")
    out_path = save_decision_output(decision_df, output_dir, decision["trade_date"])
    logger.info("Decision saved: %s", out_path)

    if text_output:
        _print_text_orders(decision_df)

    if api_client is not None:
        order_error: OrderExecutionIncomplete | None = None
        execution_plan = build_execution_plan(decision_df, current_positions)
        try:
            order_summary = submit_orders_via_api(
                decision_df=decision_df,
                api_client=api_client,
                output_dir=output_dir,
                current_positions=current_positions,
                execution_plan=execution_plan,
                state_store=state_store,
                account_key=account_key,
                strategy_key=strategy_key,
            )
        except OrderExecutionIncomplete as exc:
            # Reconcile the in-memory summary before propagating the failure.
            # This also recovers the audit trail when the submitter's first
            # attempt to write it failed.
            order_summary = exc.summary
            order_error = exc
            order_summary["execution_error"] = str(exc)
            order_summary["execution_error_type"] = type(exc).__name__

        # --- Trade journal: collect post-execution data for model improvement ---
        api_log_path = os.path.join(output_dir, "api_execution_log.json")
        reconciliation_errors: list[str] = list(
            order_error.recording_errors if order_error is not None else []
        )

        def persist_order_summary() -> None:
            with open(api_log_path, "w", encoding="utf-8") as handle:
                json.dump(order_summary, handle, ensure_ascii=False, indent=2)

        # Fetch fill prices (約定価格) for slippage analysis
        from leadlag.broker.dry_run import DryRunBrokerClient

        if not isinstance(api_client, DryRunBrokerClient) and order_summary:
            all_results = (
                order_summary.get("buy_results", [])
                + order_summary.get("sell_results", [])
                + order_summary.get("close_results", [])
            )
            if all_results:
                try:
                    fetch_fill_prices(api_client, all_results)
                except Exception as exc:  # noqa: BLE001
                    reconciliation_errors.append(f"fill_prices: {exc}")
                    logger.exception("[JOURNAL] Fill-price reconciliation failed")
                # Re-save the enriched api_execution_log with fill data.
                try:
                    persist_order_summary()
                    logger.info("[JOURNAL] Fill prices enriched in api_execution_log.json")
                except Exception as exc:  # noqa: BLE001
                    reconciliation_errors.append(f"api_log: {exc}")
                    logger.exception("[JOURNAL] API execution log update failed")

        # Save position snapshot (建単価・評価単価・評価損益)
        try:
            pos_snapshot_path = save_position_snapshot(
                api_client, output_dir, label="decision", raise_on_error=True
            )
        except Exception as exc:  # noqa: BLE001
            pos_snapshot_path = None
            reconciliation_errors.append(f"positions: {exc}")
            logger.exception("[JOURNAL] Position reconciliation failed")

        # Save wallet snapshot (維持率・受入保証金)
        try:
            wallet_snapshot_path = save_wallet_snapshot(
                api_client, output_dir, label="decision", raise_on_error=True
            )
        except Exception as exc:  # noqa: BLE001
            wallet_snapshot_path = None
            reconciliation_errors.append(f"wallet: {exc}")
            logger.exception("[JOURNAL] Wallet reconciliation failed")

        # Save daily journal index
        try:
            save_daily_journal(
                output_dir=output_dir,
                decision_csv_path=out_path,
                api_execution_log_path=api_log_path,
                position_snapshot_path=pos_snapshot_path,
                wallet_snapshot_path=wallet_snapshot_path,
            )
        except Exception as exc:  # noqa: BLE001
            reconciliation_errors.append(f"daily_journal: {exc}")
            logger.exception("[JOURNAL] Daily journal save failed")

        # Completion is published only after every post-submission checkpoint.
        # A crash before this point leaves the run discoverable by recovery.
        if state_store is not None and order_summary.get("run_id"):
            try:
                from leadlag.execution.reconcile import verify_position_checkpoint

                if not isinstance(api_client, DryRunBrokerClient):
                    reconciliation_errors.extend(verify_position_checkpoint(
                        execution_plan.to_dict(),
                        order_summary.get("buy_results", []) + order_summary.get("sell_results", [])
                        + order_summary.get("close_results", []), pos_snapshot_path,
                    ))
                state_store.record_result_set(
                    str(order_summary["run_id"]), execution_plan,
                    order_summary.get("buy_results", []) + order_summary.get("sell_results", [])
                    + order_summary.get("close_results", []),
                )
                checkpoint_errors = reconciliation_errors + ([str(order_error)] if order_error is not None else [])
                state_store.record_reconciliation(
                    str(order_summary["run_id"]),
                    outcome="incomplete" if checkpoint_errors else "complete",
                    errors=checkpoint_errors,
                    references={
                        "api_execution_log": api_log_path,
                        "position_snapshot": str(pos_snapshot_path) if pos_snapshot_path else None,
                        "wallet_snapshot": str(wallet_snapshot_path) if wallet_snapshot_path else None,
                        "decision_csv": str(out_path),
                    },
                )
                if checkpoint_errors:
                    state_store.mark_reconciliation_required(
                        str(order_summary["run_id"]),
                        error="; ".join(checkpoint_errors),
                    )
                else:
                    state_store.mark_completed(str(order_summary["run_id"]))
            except Exception as exc:
                reconciliation_errors.append(f"state_store: {exc}")
                logger.exception("[JOURNAL] Failed to persist durable reconciliation state")

        if reconciliation_errors:
            order_summary["reconciliation_errors"] = reconciliation_errors
            all_results = order_summary.get("buy_results", []) + order_summary.get("sell_results", []) + order_summary.get("close_results", [])
            order_summary["execution_report"] = report_from_records(
                all_results, expected_orders=int(order_summary.get("expected_orders_count", len(all_results))),
                close_failed=bool(order_summary.get("close_failed", False)),
                recording_errors=order_summary.get("execution_log_errors", []),
                reconciliation_errors=reconciliation_errors,
            ).to_dict()
            try:
                persist_order_summary()
            except Exception as exc:
                reconciliation_errors.append(f"api_log: {exc}")

        if order_error is not None:
            raise order_error
        if reconciliation_errors:
            raise RuntimeError(
                "Post-decision reconciliation incomplete: "
                + "; ".join(reconciliation_errors)
            )

    return out_path


def execute_post_decision_flow(
    decision: dict,
    config: ProductionConfig,
    manual_opens: dict,
    max_capital: float,
    hist_returns: pd.Series,
    output_dir: str | Path,
    api_client: BrokerClient | None = None,
    text_output: bool = False,
    current_positions: dict[str, int] | None = None,
    state_store: ExecutionStateStore | None = None,
    account_key: str = "default",
    strategy_key: str = "production_v2",
) -> str:
    """Execute post-decision flow (gross adjustment, risk check, capital allocation, order submission, and output writing).

    The dry-run vs live behaviour is determined by the ``api_client`` type:
    pass a ``DryRunBrokerClient`` for simulated execution, or a
    ``KabuBrokerClient`` for live trading.

    Returns:
        Path to the decision output CSV.
    """
    decision_df, capital_alloc = _prepare_decision_df(
        decision, config, manual_opens, max_capital
    )

    _log_decision_allocations(decision_df, capital_alloc, max_capital, decision)

    _run_risk_check_and_print(
        decision,
        decision_df,
        max_capital,
        hist_returns,
        config,
        current_positions=current_positions,
    )

    out_path = _write_decision_output_and_submit(
        decision_df=decision_df,
        decision=decision,
        output_dir=output_dir,
        text_output=text_output,
        api_client=api_client,
        current_positions=current_positions,
        state_store=state_store,
        account_key=account_key,
        strategy_key=strategy_key,
    )

    return out_path

# ---------------------------------------------------------------------------
