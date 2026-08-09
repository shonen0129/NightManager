"""Output helpers.

This module was split from ``execution/helpers.py`` as part of P1-B1 to
isolate output directory construction, decision CSV/JSON writing, backtest
summary files, and daily trade journal snapshots from broker, pricing, risk,
and post-decision flow.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.execution.config import StrategyConfig as ProductionConfig
from leadlag.reporting.results_format import create_results_output_dir

logger = logging.getLogger(__name__)


def build_output_dir(
    output_root: str,
    run_tag: str | None,
    run_name: str,
) -> str:
    return create_results_output_dir(
        run_name=run_name,
        output_root=output_root,
        run_tag=run_tag,
        manifest_extra={"entry_point": "cli.py"},
    )


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------


def save_decision_output(
    decision_df: pd.DataFrame, output_dir: str | Path, trade_date: pd.Timestamp
) -> str:
    out_path = os.path.join(output_dir, f"decision_{trade_date.strftime('%Y%m%d')}.csv")
    decision_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path



def save_summary_files(
    results: pd.DataFrame,
    metrics: dict,
    config: ProductionConfig,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    results_path = os.path.join(output_dir, "daily_results.csv")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    summary_path = os.path.join(output_dir, "run_summary.json")

    results.to_csv(results_path, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    wealth = (1.0 + results["daily_return"]).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    cfg_dict = config.model_dump()

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "config": cfg_dict,
        "samples": int(len(results)),
        "first_trade_date": str(results.index.min().date()),
        "last_trade_date": str(results.index.max().date()),
        "final_wealth": float(wealth.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "output_files": {
            "daily_results": results_path,
            "metrics": metrics_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



def save_position_snapshot(
    api_client: BrokerClient,
    output_dir: str | Path,
    *,
    label: str = "decision",
    date_str: str | None = None,
) -> str | None:
    """Save current position snapshot with entry/evaluation prices.

    Saves a JSON file with per-position details including:
      - ticker, side, quantity, entry_price (建単価)
      - evaluation_price (評価単価), unrealized_pnl (評価損益)
      - margin costs (順日歩, 逆日歩, 貸株料)

    Args:
        api_client: BrokerClient instance
        output_dir: Directory to save the snapshot file
        label: Label for the filename (e.g. 'decision', 'close')
        date_str: Optional date string (YYYYMMDD). Defaults to today.

    Returns:
        Path to the saved file, or None if no positions or error.
    """
    try:
        positions = api_client.get_positions()
    except Exception as e:
        logger.warning("Failed to fetch positions for snapshot: %s", e)
        return None

    if not positions:
        logger.info("[JOURNAL] No open positions for snapshot.")
        return None

    snapshot: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "positions": [],
    }

    for pos in positions:
        extra = pos.extra or {}
        entry_price = pos.price
        eval_price = float(extra.get("sOrderHyoukaTanka", 0) or 0)
        unrealized_pnl = float(extra.get("sOrderGaisanHyoukaSoneki", 0) or 0)
        unrealized_pnl_pct = float(extra.get("sOrderGaisanHyoukaSonekiRitu", 0) or 0)

        snapshot["positions"].append({
            "ticker": pos.ticker,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": entry_price,
            "evaluation_price": eval_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "execution_id": pos.execution_id,
            "margin_trade_type": pos.margin_trade_type,
            "account_type": pos.account_type,
            "tategyoku_day": extra.get("sOrderTategyokuDay"),
            "tategyoku_kizitu_day": extra.get("sOrderTategyokuKizituDay"),
            "tategyoku_daikin": float(extra.get("sOrderTategyokuDaikin", 0) or 0),
            "tate_tesuryou": float(extra.get("sOrderTateTesuryou", 0) or 0),
            "jun_hibu": float(extra.get("sOrderZyunHibu", 0) or 0),
            "gyaku_hibu": float(extra.get("sOrderGyakuhibu", 0) or 0),
            "kasikaburyou": float(extra.get("sOrderKasikaburyou", 0) or 0),
            "hensai_kanou_suryou": extra.get("sOrderHensaiKanouSuryou"),
        })

    snapshot["position_count"] = len(snapshot["positions"])
    snapshot["total_unrealized_pnl"] = sum(p["unrealized_pnl"] for p in snapshot["positions"])

    filename_date = date_str or datetime.now().strftime('%Y%m%d')
    filename = f"positions_{label}_{filename_date}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    logger.info("[JOURNAL] Position snapshot saved: %s (%d positions, P&L=%s)",
                filepath, snapshot["position_count"],
                f"{snapshot['total_unrealized_pnl']:,.0f}")
    return filepath


def save_wallet_snapshot(
    api_client: BrokerClient,
    output_dir: str | Path,
    *,
    label: str = "decision",
    date_str: str | None = None,
) -> str | None:
    """Save wallet/balance snapshot with margin details.

    Saves cash_available, margin_available, 受入保証金, 維持率, 追証フラグ.

    Args:
        api_client: BrokerClient instance
        output_dir: Directory to save the snapshot file
        label: Label for the filename
        date_str: Optional date string (YYYYMMDD). Defaults to today.

    Returns:
        Path to the saved file, or None on error.
    """
    try:
        wallet = api_client.get_wallet()
    except Exception as e:
        logger.warning("Failed to fetch wallet for snapshot: %s", e)
        return None

    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "cash_available": wallet.cash_available,
        "margin_available": wallet.margin_available,
        "ukeire_hosyoukin": wallet.extra.get("ukeire_hosyoukin"),
        "hosyoukin_yoryoku": wallet.extra.get("hosyoukin_yoryoku"),
        "hosyoukin_ritu": wallet.extra.get("hosyoukin_ritu"),
        "sHosyouKinritu": wallet.extra.get("sHosyouKinritu"),
        "sOisyouHasseiFlg": wallet.extra.get("sOisyouHasseiFlg"),
        "sTatekaekinHasseiFlg": wallet.extra.get("sTatekaekinHasseiFlg"),
    }

    filename_date = date_str or datetime.now().strftime('%Y%m%d')
    filename = f"wallet_{label}_{filename_date}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    logger.info("[JOURNAL] Wallet snapshot saved: %s (margin=%s JPY, 維持率=%s%%)",
                filepath,
                f"{wallet.margin_available:,.0f}",
                snapshot.get("hosyoukin_ritu", "N/A"))
    return filepath


def save_daily_journal(
    output_dir: str | Path,
    decision_csv_path: str | None = None,
    api_execution_log_path: str | None = None,
    position_snapshot_path: str | None = None,
    wallet_snapshot_path: str | None = None,
    close_execution_log_path: str | None = None,
) -> str:
    """Save a daily journal index file that links all collected data.

    Creates a single JSON file per day that references all collected
    artifacts (decision, fills, positions, wallet, close) for easy
    retrospective analysis.

    Args:
        output_dir: Directory for the journal file
        decision_csv_path: Path to decision CSV
        api_execution_log_path: Path to API execution log JSON
        position_snapshot_path: Path to position snapshot JSON
        wallet_snapshot_path: Path to wallet snapshot JSON
        close_execution_log_path: Path to close execution log JSON

    Returns:
        Path to the journal index file.
    """
    journal_dir = os.path.join(os.path.dirname(output_dir), "trade_journal")
    os.makedirs(journal_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    journal: dict[str, Any] = {
        "date": date_str,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {},
    }

    for label, path in [
        ("decision_csv", decision_csv_path),
        ("api_execution_log", api_execution_log_path),
        ("position_snapshot", position_snapshot_path),
        ("wallet_snapshot", wallet_snapshot_path),
        ("close_execution_log", close_execution_log_path),
    ]:
        if path and os.path.exists(path):
            journal["artifacts"][label] = path

    journal_path = os.path.join(journal_dir, f"journal_{date_str}.json")
    with open(journal_path, "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)
    logger.info("[JOURNAL] Daily journal saved: %s", journal_path)
    return journal_path

