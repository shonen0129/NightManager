"""Build actual fill, position, cost, and exposure artifacts from local evidence.

This script reads the official Tachibana repayment CSV, the existing grouped
reconciliation, broker position/wallet snapshots, and local execution SQLite.
It does not access the broker API or mutate trading data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/Users/shonen/Downloads/Tachibana_取引記録.csv")

sys.path.insert(0, str(ROOT))
from tools.validation.reconcile_production_ledger import load_official_transactions  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_position_snapshots() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = ("20260729", "20260730", "20260731", "20260924", "20260925")
    exposures: list[dict[str, Any]] = []
    daily_pnl: list[dict[str, Any]] = []
    results = ROOT / "var" / "results"
    for date in dates:
        matches = sorted(results.glob(f"{date}_*_production_close_positions"))
        if not matches:
            continue
        folder = matches[-1]
        positions_path = next(iter(folder.glob("positions_close_*.json")), None)
        wallet_path = next(iter(folder.glob("wallet_close_*.json")), None)
        pnl_path = next(iter(folder.glob("daily_pnl_summary_*.json")), None)
        positions = read_json(positions_path) if positions_path else {}
        wallet = read_json(wallet_path) if wallet_path else {}
        pnl = read_json(pnl_path) if pnl_path else {}
        rows = positions.get("positions", [])

        long_value = 0.0
        short_value = 0.0
        gross = 0.0
        signed = 0.0
        for row in rows:
            quantity = int(row.get("quantity") or 0)
            mark = float(row.get("evaluation_price") or 0.0)
            value = quantity * mark
            gross += value
            if row.get("side") == "BUY":
                long_value += value
                signed += value
            elif row.get("side") == "SELL":
                short_value += value
                signed -= value

        collateral = wallet.get("ukeire_hosyoukin")
        exposures.append(
            {
                "date": date,
                "snapshot_time": positions.get("timestamp"),
                "position_count": positions.get("position_count", len(rows)),
                "long_position_count": sum(row.get("side") == "BUY" for row in rows),
                "short_position_count": sum(row.get("side") == "SELL" for row in rows),
                "long_market_value_jpy": round(long_value, 2),
                "short_abs_market_value_jpy": round(short_value, 2),
                "gross_market_value_jpy": round(gross, 2),
                "net_market_value_jpy": round(signed, 2),
                "net_to_gross_pct": round(100 * signed / gross, 4) if gross else 0.0,
                "gross_to_received_collateral": (
                    round(gross / float(collateral), 4) if collateral else None
                ),
                "cash_available_jpy": wallet.get("cash_available"),
                "margin_available_jpy": wallet.get("margin_available"),
                "received_collateral_jpy": collateral,
                "initial_margin_rate_pct": wallet.get("hosyoukin_ritu"),
                "total_unrealized_pnl_jpy": positions.get("total_unrealized_pnl"),
                "valuation_basis": "broker position quantity × broker evaluation_price",
            }
        )
        daily_pnl.append(
            {
                "date": date,
                "close_orders_count": pnl.get("close_orders_count"),
                "realized_count": pnl.get("realized_count"),
                "total_realized_pnl_jpy": pnl.get("total_realized_pnl"),
                "total_fees_jpy": pnl.get("total_fees"),
                "total_unrealized_pnl_jpy": pnl.get("total_unrealized_pnl"),
                "total_daily_pnl_jpy": pnl.get("total_daily_pnl"),
            }
        )
    return exposures, daily_pnl


def build(source_csv: Path) -> dict[str, Any]:
    recon = read_json(REPORT_DIR / "reconciliation.json")
    source_hash = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    if source_hash != recon["source_sha256"]:
        raise ValueError("The source CSV hash differs from the grouped reconciliation artifact")
    transactions = load_official_transactions(source_csv)
    all_groups = recon["groups"]
    local_groups = [group for group in all_groups if group["local_order_count"]]
    strict_groups = [group for group in local_groups if group["execution_price_match"]]

    # The source reconciliation groups on trade date/ticker/side. Without an
    # official order ID, equal aggregate quantity alone cannot prove an order
    # match, so report that limitation explicitly in the derived JSON too.
    recon["status"] = "DATA_EXTRACTED_ORDER_RECONCILIATION_INCOMPLETE"
    recon["decision"] = "KEEP_ORDER_AND_FEE_GAPS_EXPLICIT"
    recon["matching_basis"] = "trade_date + ticker + closing_side + aggregate quantity; no broker order ID"
    recon["coverage"]["order_id_level_match_count"] = 0
    recon["coverage"]["execution_price_exact_group_count"] = len(strict_groups)
    recon["coverage"]["execution_price_exact_share_count"] = sum(
        int(group["official_quantity"]) for group in strict_groups
    )
    recon["completion_basis"]["authoritative_ledger_group_match"] = False
    recon["completion_basis"]["authoritative_ledger_quantity_match"] = False
    recon["completion_basis"]["order_id_level_match"] = False
    recon["completion_basis"]["exact_price_match_for_all_groups"] = len(strict_groups) == len(local_groups)
    recon["unresolved"] = [
        "The official CSV has no broker order ID. Date/ticker/side/quantity group matches are aggregate crosswalks, not one-to-one order reconciliation.",
        "Local order logs record 8 filled shares out of 207 requested. The official CSV has 207 shares under those group keys, but cannot attribute the other 199 to specific order IDs.",
        f"Only {len(strict_groups)} of {len(local_groups)} groups ({sum(int(group['official_quantity']) for group in strict_groups)} shares) also match execution price exactly.",
        "The CSV has no itemized fee fields. Fee evidence is limited to fill-detail rows preserved in local execution logs.",
    ]
    (REPORT_DIR / "reconciliation.json").write_text(
        json.dumps(recon, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    transaction_path = REPORT_DIR / "official_repayment_transactions.csv"
    with transaction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        columns = [
            "repayment_index_in_parser_order",
            "trade_date",
            "ticker",
            "closing_side",
            "quantity_shares",
            "execution_price_jpy",
            "reported_pnl_or_settlement_jpy",
            "trade_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(transactions, start=1):
            writer.writerow(
                {
                    "repayment_index_in_parser_order": index,
                    "trade_date": row["trade_date"],
                    "ticker": row["ticker"],
                    "closing_side": row["side"],
                    "quantity_shares": row["quantity"],
                    "execution_price_jpy": row["price"],
                    "reported_pnl_or_settlement_jpy": row["reported_pnl_or_settlement"],
                    "trade_label": row["trade_label"],
                }
            )

    ledger_path = REPORT_DIR / "local_to_official_fill_crosswalk.csv"
    columns = [
        "trade_date",
        "ticker",
        "closing_side",
        "local_order_ids",
        "local_order_count",
        "requested_quantity_shares",
        "runtime_recorded_fill_quantity_shares",
        "official_group_quantity_shares",
        "group_quantity_match",
        "local_fill_prices_jpy",
        "official_fill_prices_jpy",
        "fill_price_match",
        "local_recorded_settlement_pnl_jpy",
        "official_reported_pnl_or_settlement_jpy",
        "pre_fee_price_pnl_from_official_close_jpy",
        "pnl_reconciliation_delta_jpy",
        "observed_fee_fields_jpy",
        "source_paths",
    ]
    with ledger_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for group in sorted(local_groups, key=lambda item: (item["trade_date"], item["ticker"], item["side"])):
            writer.writerow(
                {
                    "trade_date": group["trade_date"],
                    "ticker": group["ticker"],
                    "closing_side": group["side"],
                    "local_order_ids": json.dumps(group["local_order_ids"], ensure_ascii=False),
                    "local_order_count": group["local_order_count"],
                    "requested_quantity_shares": group["local_requested_quantity"],
                    "runtime_recorded_fill_quantity_shares": group["local_confirmed_quantity"],
                    "official_group_quantity_shares": group["official_quantity"],
                    "group_quantity_match": group["quantity_match"],
                    "local_fill_prices_jpy": json.dumps(group["local_observed_prices"], ensure_ascii=False),
                    "official_fill_prices_jpy": json.dumps(group["official_prices"], ensure_ascii=False),
                    "fill_price_match": group["execution_price_match"],
                    "local_recorded_settlement_pnl_jpy": group["local_reported_settlement_pnl"],
                    "official_reported_pnl_or_settlement_jpy": group["official_reported_pnl_or_settlement"],
                    "pre_fee_price_pnl_from_official_close_jpy": group["pre_fee_price_pnl_from_official_close"],
                    "pnl_reconciliation_delta_jpy": group["official_minus_pre_fee_price_pnl"],
                    "observed_fee_fields_jpy": json.dumps(group["local_fee_fields_observed"], ensure_ascii=False),
                    "source_paths": json.dumps(group["source_paths"], ensure_ascii=False),
                }
            )

    fee_totals: dict[str, float] = {}
    for group in strict_groups:
        for field, value in group.get("local_fee_fields_observed", {}).items():
            fee_totals[field] = fee_totals.get(field, 0.0) + float(value)
    strict_quantity = sum(group["official_quantity"] for group in strict_groups)
    strict_notional = sum(
        group["official_quantity"] * group["official_prices"][0]
        for group in strict_groups
        if group["official_prices"]
    )
    strict_recorded_pnl = sum(float(group["local_reported_settlement_pnl"] or 0.0) for group in strict_groups)
    strict_pre_fee_pnl = sum(
        float(group["pre_fee_price_pnl_from_official_close"])
        for group in strict_groups
        if group["pre_fee_price_pnl_from_official_close"] is not None
    )
    exposures, daily_pnl = build_position_snapshots()

    db_path = ROOT / "var/live/pipeline_data/execution/execution_state.sqlite"
    counts = {}
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    for table in ("execution_runs", "order_intents", "order_observations", "execution_reconciliations"):
        counts[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    db.close()
    broker_query_path = REPORT_DIR / "broker_requery.json"
    broker_query = read_json(broker_query_path) if broker_query_path.exists() else None

    total_repayment_turnover = sum(float(row["quantity"]) * float(row["price"]) for row in transactions)
    metrics = {
        "as_of": "2026-09-27",
        "official_source_path": str(source_csv),
        "official_source_sha256": recon["source_sha256"],
        "official_csv_repayment_transactions": len(transactions),
        "official_csv_repayment_shares": sum(int(row["quantity"]) for row in transactions),
        "official_csv_repayment_turnover_jpy": round(total_repayment_turnover, 2),
        "official_csv_repayment_groups": recon["coverage"]["official_group_count"],
        "local_close_orders": recon["coverage"]["local_close_order_count"],
        "local_close_groups": len(local_groups),
        "local_requested_shares": sum(group["local_requested_quantity"] for group in local_groups),
        "official_shares_for_local_group_keys": sum(group["official_quantity"] for group in local_groups),
        "runtime_recorded_filled_shares_for_local_group_keys": sum(group["local_confirmed_quantity"] for group in local_groups),
        "local_groups_matching_by_date_ticker_side_quantity": sum(bool(group["quantity_match"]) for group in local_groups),
        "local_groups_also_matching_execution_price": len(strict_groups),
        "strictly_price_matched_shares": strict_quantity,
        "strictly_price_matched_close_notional_jpy": round(strict_notional, 2),
        "strictly_matched_local_settlement_pnl_field_jpy": round(strict_recorded_pnl, 2),
        "strictly_matched_pre_fee_price_pnl_jpy": round(strict_pre_fee_pnl, 2),
        "strictly_matched_realized_pnl_vs_pre_fee_delta_jpy": round(strict_recorded_pnl - strict_pre_fee_pnl, 2),
        "strictly_matched_observed_fee_fields_jpy": fee_totals,
        "position_exposure_snapshots": exposures,
        "daily_pnl_summaries": daily_pnl,
        "execution_state_database_row_counts": counts,
        "broker_order_detail_requery": {
            "status": "HEALTH_CHECK_HTTP_404" if broker_query and broker_query.get("sanitization") else "NOT_RUN",
            "individual_order_detail_requests_sent": 0,
            "response_artifact": str(broker_query_path.relative_to(ROOT)) if broker_query else None,
        },
        "limitations": [
            "The official CSV has no broker order ID and its column is labelled 損益/受渡金額; values are preserved without treating them as net PnL.",
            "Date/ticker/side/quantity joins cover all 23 local groups, but only 7 groups (8 shares) also match execution price exactly.",
            "The local order log reports 8 filled shares out of 207 requested; 199 is a logging difference, not evidence that the remaining shares were unfilled.",
            "Observed fee fields on the 8-share strict scope total 47 JPY. This does not establish total fees on the remaining 199 shares or financing/borrow/reverse costs.",
            "Actual slippage cannot be derived without arrival-time executable quotes for these fills.",
            "Position value is quantity multiplied by the broker snapshot evaluation price, not liquidation proceeds at executable bid/ask.",
            "Margin available is reported as observed; no hypothetical buying power is inferred from it.",
            "Recent 2026-09-24 and 2026-09-25 snapshots are flat and cannot establish live active-portfolio neutrality.",
        ],
    }
    (REPORT_DIR / "actual_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def write_report(metrics: dict[str, Any]) -> None:
    exposures = metrics["position_exposure_snapshots"]
    lines = [
        "# 実約定台帳・実数量レビュー（2026-09-27）",
        "",
        f"対象CSV: `{metrics['official_source_path']}`。SHA-256: `{metrics['official_source_sha256']}`。CSV、既存ローカルログ、建玉/余力snapshot、execution SQLiteを読み取り照合した。読み取り専用の注文詳細照会は試みたが、API health checkがHTTP 404となり個別照会は送信されていない。再現コマンドは `python3 reports/20260927_actual_execution_review/build_actual_metrics.py`。",
        "",
        "## 約定台帳",
        "",
        f"公式取引CSVから返済取引 **{metrics['official_csv_repayment_transactions']:,}件 / {metrics['official_csv_repayment_shares']:,}株 / 売買代金 {metrics['official_csv_repayment_turnover_jpy']:,.0f}円**を行単位の[台帳](official_repayment_transactions.csv)にした。この代金はCSVにある返済取引分で、戦略全体の売買高や資金容量を表さない。",
        "",
        f"ローカル履歴は{metrics['local_close_orders']}注文・{metrics['local_close_groups']}グループ、要求数量207株。日付・銘柄・売買方向・数量の照合は23/23グループで合計207/207株だった。しかしこれは注文IDの一対一照合ではない。取引価格まで一致する厳密な照合は **{metrics['local_groups_also_matching_execution_price']}グループ・{metrics['strictly_price_matched_shares']}株**に留まる。注文ログの記録約定数量は8株で、残る199株はログとCSVの差であり、未約定とは断定できない。[注文ログとの照合表](local_to_official_fill_crosswalk.csv)・[全グループ照合結果](reconciliation.json)",
        "",
        "## 実費・損益",
        "",
        f"価格まで厳密に一致した7グループでは、約定代金 **{metrics['strictly_price_matched_close_notional_jpy']:,.0f}円**、記録上の決済損益 **{metrics['strictly_matched_local_settlement_pnl_field_jpy']:,.0f}円**、建値と約定価格からの費用控除前損益 **{metrics['strictly_matched_pre_fee_price_pnl_jpy']:,.0f}円**でした。両者の差は **{metrics['strictly_matched_realized_pnl_vs_pre_fee_delta_jpy']:,.0f}円**です。確認できた費用項目16円を差し引いても10円が未照合で、会計上の完全一致には至っていません。",
        "",
        f"この厳密な8株分の記録にある費用項目は合計 **{sum(metrics['strictly_matched_observed_fee_fields_jpy'].values()):,.0f}円**です。内訳: " + ", ".join(f"`{key}` {value:,.0f}円" for key, value in metrics["strictly_matched_observed_fee_fields_jpy"].items() if value),
        "",
        "この16円を全207株の費用へ外挿しない。公式CSVに注文IDと費用明細がなく、9:10時点の実行可能気配もこの約定日に揃わないため、全体のnet PnL・実測slippage・金利・貸株・逆日歩はまだ確定できない。CSVの「損益/受渡金額」列も意味を一意に定めず、その数値をnet PnLと解釈していない。",
        "",
        "## 実建玉・資金容量・市場中立性",
        "",
        "下表はブローカー建玉の実数量×評価価格による時価概算。net/grossは名目金額の差で、TOPIX等に対するβ中立性ではない。受入保証金に対するgrossは容量を見るための観測比率で、評価価格・口座保証金を使う簡易値です。",
        "",
        "| 日付 | 建玉数 | Long / Short時価 | Gross時価 | Net時価 | Net/Gross | Gross/受入保証金 | 新規信用余力 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in exposures:
        ratio = row["gross_to_received_collateral"]
        lines.append(
            f"| {row['date']} | {row['position_count']} | {row['long_market_value_jpy']:,.0f} / {row['short_abs_market_value_jpy']:,.0f}円 | {row['gross_market_value_jpy']:,.0f}円 | {row['net_market_value_jpy']:,.0f}円 | {row['net_to_gross_pct']:+.2f}% | {ratio:.2f}倍 | {row['margin_available_jpy']:,.0f}円 |"
        )
    lines += [
        "",
        "7月29日の実建玉では名目net/grossが−3.06%で、他の稼働スナップショットよりnet寄りでした。gross=2・net上限±0.05から導く2.5%比率を単純な目安として上回ります。ただし実建玉には前日からの保有が残り、評価価格での口座スナップショットはその日のモデルウェイトと一致しません。これは再確認すべき差であり、規約違反の確定判定ではありません。7月30・31日は−1.45%・−0.96%。稼働3日のgross/受入保証金は2.37〜2.78倍、新規信用余力は75,418〜205,278円でした。板厚・注文約定率・銘柄別貸株がなく、これを取引可能AUMとは認定できません。約定後のTOPIX βも、3日分の疎な建玉snapshotだけでは推定できません。",
        "",
        "9月24・25日は建玉0銘柄・gross/net 0円、現金余力11,102円、新規信用余力0円でした。これは当該日の口座がflatだった記録で、市場中立戦略の稼働時の実績を示しません。9月27日の収集結果も`MARKET_CLOSED`のため、新しい気配データはありません。",
        "",
        "## 次の照合に必要な証拠",
        "",
        "1. 注文ID・約定IDを含む取引CSVまたはbroker約定明細をローカルorder intentへ結合する。",
        "2. 次の実取引日の注文数量・部分約定・価格・全費用・建玉を、引け後に三者照合する。",
        "3. active positionsの日中bid/ask・板数量・当時の保証金・貸株可能量から、約定後net/gross・β・fill率・投下可能額を測る。",
        "",
        "JSON集計は[actual_metrics.json](actual_metrics.json)。",
        "",
    ]
    (REPORT_DIR / "actual_execution_cost_capacity_neutrality.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.csv if args.csv.is_absolute() else ROOT / args.csv
    metrics = build(source)
    write_report(metrics)
    print(
        json.dumps(
            {
                "official_repayment_transactions": metrics["official_csv_repayment_transactions"],
                "strictly_price_matched_groups": metrics["local_groups_also_matching_execution_price"],
                "strictly_matched_shares": metrics["strictly_price_matched_shares"],
                "strict_scope_recorded_pnl_jpy": metrics["strictly_matched_local_settlement_pnl_field_jpy"],
                "strict_scope_observed_fee_jpy": sum(metrics["strictly_matched_observed_fee_fields_jpy"].values()),
                "exposure_dates": [row["date"] for row in metrics["position_exposure_snapshots"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
