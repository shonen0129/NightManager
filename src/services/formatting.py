"""Output formatting services: text orders, decision summaries, risk reports."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def print_text_orders(decision_df: pd.DataFrame) -> None:
    """Print human-readable trade orders to stdout (Japanese text mode)."""
    print("\n" + "=" * 60)
    print(" 本日の全銘柄シグナル状況")
    print("=" * 60)

    # Sort signals descending
    all_signals = decision_df.sort_values(by="signal", ascending=False)
    for _, row in all_signals.iterrows():
        print(
            f" -> 銘柄: {row['ticker']:<6} | シグナル: {row['signal']:>7.4f} | 判定: {row['action']:<4}"
        )

    print("\n" + "=" * 60)
    print(" 本日の取引指示 (テキストモード)")
    print("=" * 60)

    buy_orders = decision_df[
        (decision_df["action"] == "BUY") & (decision_df["quantity"] > 0)
    ]
    sell_orders = decision_df[
        (decision_df["action"] == "SELL") & (decision_df["quantity"] > 0)
    ]

    print("\n[ 買い注文 (BUY) ]")
    if len(buy_orders) == 0:
        print(" -> 買い注文はありません")
    else:
        for _, row in buy_orders.iterrows():
            limit_str = ""
            if "limit_price" in row and pd.notna(row["limit_price"]):
                lp = row["limit_price"]
                lp_str = f"{lp:.1f}" if lp % 1 != 0 else f"{int(lp)}"
                limit_str = f"| 指値: {lp_str:>6} 円 "
            print(
                f" -> 銘柄: {row['ticker']:<6} | シグナル: {row['signal']:>7.4f} "
                f"{limit_str}| 数量: {int(row['quantity']):>4}株 | 概算: {int(row['etf_amount']):>8,} 円"
            )

    print("\n[ 売り注文 (SELL) ]")
    if len(sell_orders) == 0:
        print(" -> 売り注文はありません")
    else:
        for _, row in sell_orders.iterrows():
            limit_str = ""
            if "limit_price" in row and pd.notna(row["limit_price"]):
                lp = row["limit_price"]
                lp_str = f"{lp:.1f}" if lp % 1 != 0 else f"{int(lp)}"
                limit_str = f"| 指値: {lp_str:>6} 円 "
            print(
                f" -> 銘柄: {row['ticker']:<6} | シグナル: {row['signal']:>7.4f} "
                f"{limit_str}| 数量: {int(row['quantity']):>4}株 | 概算: {int(row['etf_amount']):>8,} 円"
            )
    print("\n" + "=" * 60 + "\n")


def log_decision_summary(decision_df: pd.DataFrame, decision: dict) -> None:
    """Log a structured summary of the trade decision."""
    buy_count = int((decision_df["action"] == "BUY").sum())
    sell_count = int((decision_df["action"] == "SELL").sum())
    hold_count = int((decision_df["action"] == "HOLD").sum())

    logger.info(
        f"Trade decision for {decision.get('trade_date', 'unknown')}: "
        f"top_signal={decision.get('top_signal', 'N/A')}"
    )
    logger.info(f"Positions: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")


def print_risk_report(report: dict) -> None:
    """Log risk check results."""
    logger.info("=== Risk Check ===")
    logger.info(f"Target net exposure: {report['target_net_exposure']:.4f}")
    logger.info(f"Target gross exposure: {report['target_gross_exposure']:.4f}")
    logger.info(f"Allocated net ratio: {report['allocated_net_ratio']:.4f}")
    logger.info(f"Allocated gross ratio: {report['allocated_gross_ratio']:.4f}")

    var_es = report["var_es"]
    if var_es["available"]:
        logger.info(
            "VaR/ES(99%%,250d): "
            f"VaR={var_es['var_loss']:.4%}, ES={var_es['es_loss']:.4%}"
        )
    else:
        logger.info(
            "VaR/ES(99%%,250d): skipped "
            f"(history={var_es['samples']}, required={var_es['window']})"
        )

    for msg in report["warning_breaches"]:
        logger.warning(f"[RISK-WARNING] {msg}")
    for msg in report["stop_breaches"]:
        logger.error(f"[RISK-STOP] {msg}")


def print_metrics(metrics: dict) -> None:
    """Log performance metrics summary."""
    logger.info("=== Performance Metrics ===")
    for key in ("AR", "RISK", "R/R", "MDD", "Total Return"):
        val = metrics.get(key)
        if val is not None:
            if key == "R/R":
                logger.info(f"  {key}: {val:.2f}")
            else:
                logger.info(f"  {key}: {val:.4%}")
