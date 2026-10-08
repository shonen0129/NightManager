#!/usr/bin/env python3
"""Audit and, when eligible, replay the research execution policy on saved LOB logs.

The command is read-only with respect to broker systems. It writes only the
research report directory and never builds a broker client or submits orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from research.execution_simulator import (
    PolicyConfig,
    assess_capture_record,
    build_order_intents,
    compare_results,
    replay_event_from_capture_record,
    simulate_adaptive_policy,
    simulate_market_baseline,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QUOTES = ROOT / "reports/20260923_profitability_order_2/quote_snapshots.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/20261007_execution_policy_simulator"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return records, [f"input file missing: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        if isinstance(item, dict):
            records.append(item)
        else:
            errors.append(f"line {line_number}: record is not an object")
    return records, errors


def _read_position_snapshot(path: Path) -> tuple[str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("positions"), dict):
        raise ValueError("positions file must contain trade_date and a signed positions object")
    trade_date = payload.get("trade_date")
    if not isinstance(trade_date, str):
        raise ValueError("positions file must contain trade_date")
    parsed_date = pd.Timestamp(trade_date).date().isoformat()
    positions: dict[str, int] = {}
    for ticker, quantity in payload["positions"].items():
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            raise ValueError(f"position quantity for {ticker} must be a signed integer")
        positions[str(ticker)] = quantity
    return parsed_date, positions


def _has_aware_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def build_audit(
    *,
    quote_path: Path,
    output_dir: Path,
    orders_path: Path | None = None,
    positions_path: Path | None = None,
    deadline_text: str | None = None,
    order_date_text: str | None = None,
) -> dict[str, Any]:
    records, read_errors = _read_jsonl(quote_path)
    checks = [assess_capture_record(record, required_tickers=JP_TICKERS) for record in records]
    eligible_by_time: dict[datetime, list[int]] = {}
    for index, item in enumerate(checks):
        if item["eligible_for_execution_replay"]:
            eligible_by_time.setdefault(datetime.fromisoformat(item["observed_at"]), []).append(index)
    for indices in eligible_by_time.values():
        if len(indices) > 1:
            for index in indices:
                checks[index]["eligible_for_execution_replay"] = False
                checks[index]["eligible_for_arrival"] = False
                checks[index]["reasons"].append("duplicate_replay_timestamp")
    eligible_indices = [i for i, item in enumerate(checks) if item["eligible_for_execution_replay"]]
    eligible_indices.sort(key=lambda i: datetime.fromisoformat(records[i]["observed_at"]))
    arrival_indices = [i for i in eligible_indices if checks[i]["eligible_for_arrival"]]
    comparison: dict[str, Any] | None = None
    simulation_status = "NOT_RUN"
    order_count: int | None = None
    deadline: datetime | None = None
    order_trade_date: str | None = order_date_text
    position_trade_date: str | None = None
    multiple_order_dates = False
    order_date_mismatch = False
    config = PolicyConfig()
    if orders_path is not None:
        decisions = pd.read_csv(orders_path)
        if "trade_date" in decisions.columns:
            dates = {
                pd.Timestamp(value).date().isoformat()
                for value in decisions["trade_date"].dropna().unique()
            }
            if len(dates) == 1:
                csv_order_date = next(iter(dates))
                if order_trade_date is None:
                    order_trade_date = csv_order_date
                elif order_trade_date != csv_order_date:
                    order_date_mismatch = True
            elif len(dates) > 1:
                multiple_order_dates = True
        if "pair_id" in decisions.columns:
            pair_ids = {
                str(row.ticker): str(row.pair_id)
                for row in decisions[["ticker", "pair_id"]].itertuples(index=False)
                if pd.notna(row.pair_id)
            }
        else:
            pair_ids = None
        current_positions: dict[str, int] | None = None
        if positions_path is not None:
            position_trade_date, current_positions = _read_position_snapshot(positions_path)
        intents = (
            build_order_intents(decisions, current_positions=current_positions, pair_ids=pair_ids)
            if current_positions is not None
            else []
        )
        order_count = len(intents) if current_positions is not None else None
        if not eligible_indices:
            simulation_status = "NOT_EVALUABLE_NO_ELIGIBLE_REPLAY_EVENTS"
        elif not arrival_indices:
            simulation_status = "NOT_EVALUABLE_NO_ELIGIBLE_ARRIVAL_BOOK"
        elif read_errors:
            simulation_status = "NOT_EVALUABLE_INPUT_READ_ERRORS"
        elif positions_path is None:
            simulation_status = "NOT_EVALUABLE_CURRENT_POSITIONS_MISSING"
        elif multiple_order_dates:
            simulation_status = "NOT_EVALUABLE_MULTIPLE_ORDER_DATES"
        elif not order_trade_date:
            simulation_status = "NOT_EVALUABLE_ORDER_DATE_MISSING"
        elif order_date_mismatch:
            simulation_status = "NOT_EVALUABLE_ORDER_DATE_MISMATCH"
        elif position_trade_date != order_trade_date:
            simulation_status = "NOT_EVALUABLE_POSITION_DATE_MISMATCH"
        elif not intents:
            simulation_status = "NOT_EVALUABLE_NO_NONZERO_ORDERS"
        elif not deadline_text:
            simulation_status = "NOT_RUN_DEADLINE_REQUIRED"
        else:
            deadline = datetime.fromisoformat(deadline_text)
            events = [replay_event_from_capture_record(records[i]) for i in eligible_indices]
            arrival_time = datetime.fromisoformat(records[arrival_indices[0]]["observed_at"])
            events = [event for event in events if event.timestamp >= arrival_time]
            if (
                len({event.timestamp.astimezone(ZoneInfo("Asia/Tokyo")).date() for event in events})
                != 1
            ):
                simulation_status = "NOT_EVALUABLE_MULTIPLE_SESSION_DATES"
            elif (
                events[0].timestamp.astimezone(ZoneInfo("Asia/Tokyo")).date().isoformat()
                != order_trade_date
            ):
                simulation_status = "NOT_EVALUABLE_ORDER_DATE_MISMATCH"
            elif deadline.tzinfo is None:
                simulation_status = "NOT_EVALUABLE_DEADLINE_TIMEZONE_MISSING"
            elif deadline < events[0].timestamp:
                simulation_status = "NOT_EVALUABLE_DEADLINE_BEFORE_ARRIVAL"
            elif deadline.astimezone(ZoneInfo("Asia/Tokyo")).date().isoformat() != order_trade_date:
                simulation_status = "NOT_EVALUABLE_DEADLINE_DATE_MISMATCH"
            elif any(
                not check["eligible_for_execution_replay"]
                and arrival_time <= datetime.fromisoformat(check["observed_at"]) <= deadline
                for check in checks
                if _has_aware_timestamp(check["observed_at"])
            ):
                simulation_status = "NOT_EVALUABLE_INVALID_REPLAY_EVENTS"
            else:
                baseline = simulate_market_baseline(
                    intents, events, deadline=deadline, config=config
                )
                adaptive = simulate_adaptive_policy(
                    intents, events, deadline=deadline, config=config
                )
                comparison = compare_results(baseline, adaptive)
                simulation_status = comparison["shortfall_delta_status"]
    elif arrival_indices:
        simulation_status = "NOT_RUN_ORDERS_REQUIRED"
    elif eligible_indices:
        simulation_status = "NOT_EVALUABLE_NO_ELIGIBLE_ARRIVAL_BOOK"
    else:
        simulation_status = "NOT_EVALUABLE_NO_ELIGIBLE_REPLAY_EVENTS"

    reasons: dict[str, int] = {}
    for check in checks:
        for reason in [*check["reasons"], *check["passive_evidence_reasons"]]:
            reasons[reason] = reasons.get(reason, 0) + 1
    has_complete_delta = bool(comparison and comparison["shortfall_delta_status"] == "COMPLETE")
    return {
        "study": "research_execution_policy_replay",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "read_only_broker": True,
        "broker_order_submission": False,
        "input_quotes": str(quote_path.relative_to(ROOT))
        if quote_path.is_relative_to(ROOT)
        else str(quote_path),
        "input_quotes_sha256": hashlib.sha256(quote_path.read_bytes()).hexdigest()
        if quote_path.exists()
        else None,
        "input_orders": str(orders_path.relative_to(ROOT))
        if orders_path and orders_path.is_relative_to(ROOT)
        else str(orders_path)
        if orders_path
        else None,
        "input_positions": str(positions_path.relative_to(ROOT))
        if positions_path and positions_path.is_relative_to(ROOT)
        else str(positions_path)
        if positions_path
        else None,
        "input_positions_sha256": hashlib.sha256(positions_path.read_bytes()).hexdigest()
        if positions_path and positions_path.exists()
        else None,
        "stored_quote_record_count": len(records),
        "eligible_replay_event_count": len(eligible_indices),
        "eligible_arrival_record_count": len(arrival_indices),
        "full_depth_record_count": sum(bool(item["full_five_level_depth"]) for item in checks),
        "true_0910_record_count": sum(bool(item["valid_09_10_capture"]) for item in checks),
        "passive_queue_evidence_record_count": sum(
            bool(item["queue_events_present"]) for item in checks
        ),
        "distinct_observed_times": sorted(
            {str(item.get("observed_at")) for item in records if item.get("observed_at")}
        ),
        "eligibility_reasons": reasons,
        "record_checks": checks,
        "orders_supplied": orders_path is not None,
        "requested_order_count": order_count,
        "order_trade_date": order_trade_date,
        "position_trade_date": position_trade_date,
        "deadline": deadline.isoformat() if deadline else deadline_text,
        "simulation_status": simulation_status,
        "policy_config": {
            "urgent_last_seconds": config.urgent_last_seconds,
            "wide_spread_bps": config.wide_spread_bps,
            "max_pair_net_ratio": config.max_pair_net_ratio,
            "adverse_selection_horizon_seconds": config.adverse_selection_horizon_seconds,
            "status": "initial design constants; uncalibrated",
        },
        "comparison": comparison,
        "metric_definitions": {
            "arrival_reference": "first valid 09:10:00-09:10:30 JST bid/ask midpoint",
            "side_adjusted_shortfall": "BUY: fill - reference; SELL: reference - fill",
            "delay_component": "side-adjusted fill-time midpoint movement from arrival midpoint",
            "spread_and_book_component": "side-adjusted fill price minus fill-time midpoint",
            "unfilled_opportunity_cost": "side-adjusted exact-deadline midpoint movement for known unfilled quantity",
            "adverse_selection": "side-adjusted midpoint movement from fill time to first observed event at/after configured horizon; diagnostic, not added twice to shortfall",
            "net_exposure": "known filled long minus short notional; peak includes each sequential simulated fill; event mids held through next mark or deadline; area is JPY-seconds; unresolved quantities make exposure incomplete",
            "depth_rule": "displayed five-level depth only; remainder is unresolved, never extrapolated",
            "passive_fill_rule": "complete queue-aware evidence must cover each preceding resting interval; a missing interval permanently leaves the pair unresolved and stops further execution",
        },
        "read_errors": read_errors,
        "paired_shortfall_delta_evaluable": has_complete_delta,
        "reproduction_command": (
            "PYTHONPATH=src .venv/bin/python src/research/scripts/experiments/"
            "replay_execution_policy.py"
        ),
    }


def write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comparison = payload.get("comparison")
    observed_times = ", ".join(payload.get("distinct_observed_times", [])) or "なし"
    if comparison:
        (output_dir / "baseline_vs_policy.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    lines = [
        "# 研究用執行ポリシー・板リプレイ",
        "",
        "## 評価仮説と対象",
        "",
        "仮説: spreadが広い、または執行側5段板のnotionalが注文量を覆わない場合、残り時間に余裕があれば同側最良気配で待機し、期限が近づいたら板内で成行性の高い注文へ切り替える。片側の約定でnetが偏る場合は、反対側を優先し、偏りを増やす側の次回許容量を縮める。",
        "",
        "baselineは`broker_ops.build_execution_plan()`が作る現行の成行注文計画。シミュレータは同関数と`execution/microstructure/`の板スキーマ・spread・depth計算を使う。broker clientの生成や注文・取消・再送は行わない。",
        "",
        "## リプレイデータの適格性",
        "",
        f"- 入力板ログ: `{payload['input_quotes']}`",
        f"- SHA-256: `{payload['input_quotes_sha256']}`",
        f"- 保存snapshot: {payload['stored_quote_record_count']}件、完全な5段板: {payload['full_depth_record_count']}件",
        f"- 観測時刻: {observed_times}",
        f"- 9:10適格snapshot: {payload['true_0910_record_count']}件、リプレイ適格イベント: {payload['eligible_replay_event_count']}件",
        f"- 受動約定のqueue証拠を含むsnapshot: {payload['passive_queue_evidence_record_count']}件",
        f"- 実行状態: **{payload['simulation_status']}**",
        "",
        "## baselineとの差",
        "",
    ]
    if comparison and payload["paired_shortfall_delta_evaluable"]:
        lines.extend(
            [
                f"- 成行baseline shortfall: {comparison['baseline']['total_shortfall_jpy']:.2f}円 ({comparison['baseline']['total_shortfall_bps']:.3f} bps)",
                f"- adaptive policy shortfall: {comparison['adaptive']['total_shortfall_jpy']:.2f}円 ({comparison['adaptive']['total_shortfall_bps']:.3f} bps)",
                f"- 差（adaptive − baseline）: {comparison['adaptive_minus_baseline_shortfall_jpy']:.2f}円",
                f"- baseline / adaptive 最大一時net: {comparison['baseline']['max_abs_net_exposure_jpy']:.2f}円 / {comparison['adaptive']['max_abs_net_exposure_jpy']:.2f}円",
                "",
                "詳細は`baseline_vs_policy.json`。",
            ]
        )
    else:
        lines.extend(
            [
                f"**差は算出できない。** 実行状態は `{payload['simulation_status']}`。入力の不足・不適格、または未確定の約定残量があるため、完全なshortfall差は報告しない。",
                "",
                "行別の理由は`data_inventory.json`。比較が実行できた場合の既知部分は`baseline_vs_policy.json`に保存するが、未確定量を含む合計値とは区別する。",
            ]
        )
    lines.extend(
        [
            "",
            "## 欠損と限界",
            "",
            "- 到着基準板は09:10:00–09:10:30 JSTに限定する。同一セッションの後続板は期限・markout用に保持する。板5段より深い残量はunresolvedとして残す。",
            "- quoteのtouchだけでは受動指値の約定とみなさない。各待機区間の完全な約定証拠が欠けたペアは、その後のクロスを停止し、後から完全なイベントが来てもunresolvedを解消しない。",
            "- net exposureは既知約定の数量を各イベントmidで評価し、両方式とも約定順のピークと期限までの絶対値積分を計算する。midは次の観測または期限まで据え置く近似で、イベント内の実時間順序は観測していない。未確定量があればnet_exposure_statusもINCOMPLETE_DATAとする。",
            "- execution shortfallは9:10 mid基準。約定分はfill時のdelayとspread/book costへ分解し、既知の未約定分は正確な期限midがある場合だけ機会費用化する。adverse selectionは別診断として報告し二重加算しない。",
            "- broker fee、税、金利、貸株料、逆日歩はこの価格shortfallに含めない。該当する約定明細・費用データもこのリプレイ入力に存在しない。",
            "- baseline注文は`broker_ops`のtarget/current差分から作る。比較には対象日と一致するsigned quantityの建玉snapshotが必要で、欠ければ空建玉を仮定せず停止する。",
            "- `wide_spread_bps`, `urgent_last_seconds`, `max_pair_net_ratio`は初期設計値で、探索・感度分析・OOS検証をしていない。データ不足のため差分検定、Deflated Sharpe、ExperimentRegistryへの性能試行登録は行っていない。",
            "- 保存ログの詳細な行別判定は`data_inventory.json`に記録した。",
            f"- 再現コマンド: `{payload['reproduction_command']}`",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", type=Path, default=DEFAULT_QUOTES)
    parser.add_argument(
        "--orders", type=Path, help="same-session decision CSV with ticker/action/quantity"
    )
    parser.add_argument(
        "--positions",
        type=Path,
        help="same-session JSON snapshot: {trade_date: YYYY-MM-DD, positions: {ticker: signed_shares}}",
    )
    parser.add_argument(
        "--deadline",
        help="timezone-aware ISO timestamp required for simulation, e.g. 2026-10-07T09:11:00+09:00",
    )
    parser.add_argument(
        "--order-date",
        help="order session date (YYYY-MM-DD), required if the CSV has no trade_date column",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    quote_path = args.quotes if args.quotes.is_absolute() else ROOT / args.quotes
    orders_path = (
        args.orders if args.orders is None or args.orders.is_absolute() else ROOT / args.orders
    )
    positions_path = (
        args.positions
        if args.positions is None or args.positions.is_absolute()
        else ROOT / args.positions
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_audit(
        quote_path=quote_path,
        output_dir=output_dir,
        orders_path=orders_path,
        positions_path=positions_path,
        deadline_text=args.deadline,
        order_date_text=args.order_date,
    )
    write_report(payload, output_dir)
    print(
        json.dumps(
            {
                "simulation_status": payload["simulation_status"],
                "stored_quote_record_count": payload["stored_quote_record_count"],
                "eligible_replay_event_count": payload["eligible_replay_event_count"],
                "report": str(output_dir / "report.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
