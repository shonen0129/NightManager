"""leadlag/reporting/daily_pnl_report.py — generate and optionally email a close P&L report.

Reads the artifacts written by ``close_all_positions`` / ``run_close_positions_mode``:

- ``close_execution_log.json``
- ``positions_close_YYYYMMDD.json``
- ``wallet_close_YYYYMMDD.json``
- ``trade_journal/journal_YYYYMMDD.json`` (for cross-references)

Computes ticket-level realized P&L from fill prices and entry prices, adds the
post-close unrealized P&L from residual positions, and writes a Markdown report.
Email delivery is opt-in via environment variables or CLI flags.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from leadlag.config.paths import results as _results_path

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_ROOT = str(_results_path())
DEFAULT_RUN_NAME = "production_close_positions"
DEFAULT_GMAIL_CREDENTIALS = "creds/credentials.json"
DEFAULT_GMAIL_SEND_TOKEN = "creds/token_gmail_send.json"


@dataclass(frozen=True)
class RealizedPnl:
    ticker: str
    close_side: str
    original_side: str
    quantity: int
    original_price: float
    fill_price: float
    fee: float
    realized_pnl: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_jpy(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.0f}"


def _find_latest_output_dir(results_root: str | Path, run_name: str = DEFAULT_RUN_NAME) -> Path | None:
    root = Path(results_root)
    if not root.exists():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir() and run_name in d.name]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _resolve_date_str(output_dir: Path, date_str: str | None = None) -> str:
    if date_str:
        return date_str
    m = re.search(r"(\d{8})", output_dir.name)
    if m:
        return m.group(1)
    return datetime.now().strftime("%Y%m%d")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_close_artifacts(
    output_dir: str | Path,
    date_str: str | None = None,
    snapshot_label: str = "close",
) -> tuple[dict, dict | None, dict | None]:
    output_dir = Path(output_dir)
    date_str = _resolve_date_str(output_dir, date_str)

    close_log = _load_json(output_dir / "close_execution_log.json") or {}
    position_snapshot = _load_json(output_dir / f"positions_{snapshot_label}_{date_str}.json") or None
    if position_snapshot is None:
        position_snapshot = _load_json(output_dir / f"positions_close_{date_str}.json") or None
    wallet_snapshot = _load_json(output_dir / f"wallet_{snapshot_label}_{date_str}.json") or None
    if wallet_snapshot is None:
        wallet_snapshot = _load_json(output_dir / f"wallet_close_{date_str}.json") or None

    return close_log, position_snapshot, wallet_snapshot


def compute_realized_pnl(close_results: Sequence[dict[str, Any]]) -> list[RealizedPnl]:
    """Compute realized P&L for each filled close order.

    ``close_results`` is expected to contain ``original_price`` and
    ``original_side`` (added by ``close_all_positions`` since 2026-07).
    Falls back to the close ``side`` if the original side is not present.
    """
    realized: list[RealizedPnl] = []
    for r in close_results:
        status = r.get("status")
        if status in ("FAILED", "SKIPPED", "SIMULATED"):
            continue
        fill_price = r.get("fill_price")
        if fill_price is None:
            continue

        original_price = _safe_float(r.get("original_price"))
        fill_price_f = _safe_float(fill_price)
        quantity = _safe_int(r.get("fill_quantity", r.get("quantity")))
        original_side = cast(str, r.get("original_side") or r.get("side"))
        close_side = cast(str, r.get("side", ""))
        fill_detail = r.get("fill_detail") or {}
        fee = _safe_float(fill_detail.get("sBaiBaiTesuryo"))

        if original_side == "BUY":
            pnl = (fill_price_f - original_price) * quantity - fee
        else:
            pnl = (original_price - fill_price_f) * quantity - fee

        realized.append(
            RealizedPnl(
                ticker=cast(str, r.get("ticker", "")),
                close_side=close_side,
                original_side=original_side,
                quantity=quantity,
                original_price=original_price,
                fill_price=fill_price_f,
                fee=fee,
                realized_pnl=pnl,
            )
        )
    return realized


def _build_markdown_report(
    date_str: str,
    close_log: dict,
    realized: list[RealizedPnl],
    position_snapshot: dict | None,
    wallet_snapshot: dict | None,
) -> str:
    total_realized = sum(r.realized_pnl for r in realized)
    total_fees = sum(r.fee for r in realized)
    filled_count = sum(1 for r in realized if r.fill_price > 0)
    pending_count = len([r for r in close_log.get("close_results", []) if r.get("fill_price") is None and r.get("status") not in ("FAILED", "SKIPPED")])

    positions = (position_snapshot or {}).get("positions", [])
    total_unrealized = _safe_float((position_snapshot or {}).get("total_unrealized_pnl"))
    total_daily_pnl = total_realized + total_unrealized

    lines: list[str] = []
    lines.append(f"# 日米ラグ 引け損益レポート — {date_str}")
    lines.append("")
    lines.append("## サマリー")
    lines.append(f"- 日付: `{date_str}`")
    lines.append(f"- クローズチケット数: {close_log.get('close_orders_count', 0)}")
    lines.append(f"- 約定済みチケット数: {filled_count}")
    lines.append(f"- 約定未取得 / 保留チケット数: {pending_count}")
    lines.append(f"- **実現損益合計: {_format_jpy(total_realized)} JPY**")
    lines.append(f"- **未実現損益合計: {_format_jpy(total_unrealized)} JPY**")
    lines.append(f"- **当日損益合計（実現＋未実現）: {_format_jpy(total_daily_pnl)} JPY**")
    lines.append(f"- 売買手数料・税合計: {_format_jpy(total_fees)} JPY")
    if close_log.get("dry_run"):
        lines.append("- **DRY RUN**: 本日のクローズはシミュレーションです")
    lines.append("")

    if realized:
        lines.append("## 実現損益チケット別")
        lines.append("")
        lines.append("| 銘柄 | 建玉Side | 引成Side | 数量 | 建単価 | 約定単価 | 手数料・税 | 実現損益 |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for r in sorted(realized, key=lambda x: x.ticker):
            lines.append(
                f"| {r.ticker} | {r.original_side} | {r.close_side} | {r.quantity:,} "
                f"| {_format_jpy(r.original_price)} | {_format_jpy(r.fill_price)} "
                f"| {_format_jpy(r.fee)} | **{_format_jpy(r.realized_pnl)}** |"
            )
        lines.append("")
        lines.append(f"**実現損益合計: {_format_jpy(total_realized)} JPY**")
        lines.append("")
    else:
        lines.append("## 実現損益チケット別")
        lines.append("")
        lines.append("_本日の実現損益はありません（約定済みクローズが0件です）。_")
        lines.append("")

    if positions:
        lines.append("## 未実現損益（残存ポジション）")
        lines.append("")
        lines.append("| 銘柄 | Side | 数量 | 建単価 | 評価単価 | 未実現損益 | 未実現% |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for p in sorted(positions, key=lambda x: x.get("ticker", "")):
            ticker = p.get("ticker", "")
            side = p.get("side", "")
            qty = _safe_int(p.get("quantity"))
            entry = _safe_float(p.get("entry_price"))
            eval_price = _safe_float(p.get("evaluation_price"))
            unrealized = _safe_float(p.get("unrealized_pnl"))
            unrealized_pct = _safe_float(p.get("unrealized_pnl_pct"))
            lines.append(
                f"| {ticker} | {side} | {qty:,} | {_format_jpy(entry)} "
                f"| {_format_jpy(eval_price)} | {_format_jpy(unrealized)} | {unrealized_pct:+.2f}% |"
            )
        lines.append("")
        lines.append(f"**未実現損益合計: {_format_jpy(total_unrealized)} JPY**")
        lines.append("")
    else:
        lines.append("## 未実現損益（残存ポジション）")
        lines.append("")
        lines.append("_引け後の残存ポジションはありません。_")
        lines.append("")

    if wallet_snapshot:
        lines.append("## ウォレット・証拠金状況")
        lines.append("")
        cash = _safe_float(wallet_snapshot.get("cash_available"))
        margin = _safe_float(wallet_snapshot.get("margin_available"))
        hosyoukin_ritu = wallet_snapshot.get("hosyoukin_ritu")
        oisyou = wallet_snapshot.get("sOisyouHasseiFlg")
        tatekaekin = wallet_snapshot.get("sTatekaekinHasseiFlg")
        lines.append(f"- 現金残高（概算）: {_format_jpy(cash)} JPY")
        lines.append(f"- 保証金余力（概算）: {_format_jpy(margin)} JPY")
        lines.append(f"- 維持率: {hosyoukin_ritu if hosyoukin_ritu is not None else 'N/A'}%")
        lines.append(f"- 追証発生フラグ: {oisyou if oisyou is not None else 'N/A'}")
        lines.append(f"- 立替金発生フラグ: {tatekaekin if tatekaekin is not None else 'N/A'}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*レポート自動生成時刻: " + datetime.now().isoformat(timespec="seconds") + "*")
    lines.append("")

    return "\n".join(lines)


def generate_close_pnl_report(
    output_dir: str | Path,
    date_str: str | None = None,
    snapshot_label: str = "close",
) -> tuple[str, dict[str, Any]]:
    """Generate a Markdown close P&L report and return (markdown, summary_dict)."""
    output_dir = Path(output_dir)
    date_str = _resolve_date_str(output_dir, date_str)

    close_log, position_snapshot, wallet_snapshot = load_close_artifacts(output_dir, date_str, snapshot_label)
    realized = compute_realized_pnl(close_log.get("close_results", []))

    total_realized = sum(r.realized_pnl for r in realized)
    total_fees = sum(r.fee for r in realized)
    total_unrealized = _safe_float((position_snapshot or {}).get("total_unrealized_pnl"))

    report_md = _build_markdown_report(
        date_str,
        close_log,
        realized,
        position_snapshot,
        wallet_snapshot,
    )

    summary = {
        "date": date_str,
        "output_dir": str(output_dir),
        "dry_run": bool(close_log.get("dry_run", False)),
        "realized_count": len(realized),
        "total_realized_pnl": total_realized,
        "total_fees": total_fees,
        "total_unrealized_pnl": total_unrealized,
        "total_daily_pnl": total_realized + total_unrealized,
        "close_orders_count": close_log.get("close_orders_count", 0),
    }

    return report_md, summary


def save_close_pnl_report(
    report_md: str,
    output_dir: str | Path,
    date_str: str,
) -> Path:
    """Write the Markdown report and a JSON summary to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"daily_pnl_report_{date_str}.md"
    md_path.write_text(report_md, encoding="utf-8")

    logger.info("Daily P&L report saved: %s", md_path)
    return md_path


def _read_env_recipients() -> list[str] | None:
    env = os.environ.get("LEADLAG_PNL_REPORT_RECIPIENTS", "")
    if not env:
        return None
    return [addr.strip() for addr in env.split(",") if addr.strip()]


def send_close_pnl_report_if_enabled(
    output_dir: str | Path | None = None,
    date_str: str | None = None,
    recipients: Sequence[str] | None = None,
    dry_run: bool | None = None,
    credentials_path: str | Path | None = None,
    token_path: str | Path | None = None,
    from_email: str | None = None,
    send_timeout: float = 30.0,
    snapshot_label: str = "close",
) -> dict[str, Any]:
    """Generate the close P&L report and email it if enabled.

    The default behavior is **dry-run / file only**. Email is only sent when:

    - ``LEADLAG_PNL_REPORT_SEND=1`` is set, and
    - at least one recipient is provided (``LEADLANG_PNL_REPORT_RECIPIENTS`` or
      the ``recipients`` argument).

    This prevents accidental email sending from untested or fresh environments.
    """
    if output_dir is None:
        output_dir = _find_latest_output_dir(DEFAULT_RESULTS_ROOT) or Path(DEFAULT_RESULTS_ROOT)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Close output directory not found: {output_dir}")

    resolved_date = _resolve_date_str(output_dir, date_str)

    # Determine dry-run and recipients from env if not explicitly passed.
    env_send = os.environ.get("LEADLAG_PNL_REPORT_SEND", "0").strip().lower() in ("1", "true", "yes")
    if dry_run is None:
        dry_run = not env_send

    env_recipients = _read_env_recipients()
    final_recipients: list[str] = []
    if recipients:
        final_recipients = [r.strip() for r in recipients if r.strip()]
    elif env_recipients:
        final_recipients = list(env_recipients)

    credentials_path = Path(credentials_path or os.environ.get("LEADLAG_GMAIL_CREDENTIALS", DEFAULT_GMAIL_CREDENTIALS))
    token_path = Path(token_path or os.environ.get("LEADLAG_GMAIL_SEND_TOKEN", DEFAULT_GMAIL_SEND_TOKEN))
    from_email = from_email or os.environ.get("LEADLAG_PNL_FROM_EMAIL")

    report_md, summary = generate_close_pnl_report(output_dir, resolved_date, snapshot_label)
    md_path = save_close_pnl_report(report_md, output_dir, resolved_date)

    summary_path = output_dir / f"daily_pnl_summary_{resolved_date}.json"
    _save_json(summary_path, summary)

    result: dict[str, Any] = {
        "date": resolved_date,
        "output_dir": str(output_dir),
        "report_path": str(md_path),
        "summary_path": str(summary_path),
        "dry_run": dry_run,
        "recipients": final_recipients,
        "sent": False,
        "message_id": None,
        "error": None,
        "summary": summary,
    }

    if dry_run or not final_recipients:
        logger.info(
            "[DRY RUN / NO RECIPIENTS] P&L report not sent. Recipients=%s, dry_run=%s",
            final_recipients,
            dry_run,
        )
        return result

    try:
        from leadlag.reporting.gmail_sender import GmailSender

        sender = GmailSender(
            credentials_path=credentials_path,
            token_path=token_path,
            from_email=from_email,
            send_timeout=send_timeout,
        )
        subject = f"【日米ラグ】引け損益レポート {resolved_date}"
        msg_id = sender.send(
            to=final_recipients,
            subject=subject,
            body=report_md,
            from_email=from_email,
            dry_run=False,
        )
        result["sent"] = True
        result["message_id"] = msg_id
        logger.info("P&L report sent to %s", final_recipients)
    except Exception as e:
        logger.exception("Failed to send P&L report: %s", e)
        result["error"] = str(e)

    return result

def _backfill_close_metadata(close_log: dict, pre_close_positions: list[dict]) -> None:
    """Fill in original_price/original_side for close results that lack them.

    Uses the pre-close position snapshot (``positions_close_YYYYMMDD.json``)
    to recover the entry price and original side. This makes P&L reports
    accurate even for close logs produced before ``close.py`` started saving
    ``original_price`` / ``original_side``.
    """
    positions_by_key: dict[tuple[str, str], dict] = {}
    for p in pre_close_positions:
        key = (p.get("ticker", ""), p.get("side", ""))
        positions_by_key.setdefault(key, p)

    for r in close_log.get("close_results", []):
        ticker = r.get("ticker")
        close_side = r.get("side")
        if not ticker or not close_side:
            continue

        # If both original price and side are already present, nothing to do.
        original_price = r.get("original_price")
        original_side = r.get("original_side")
        if original_price and original_side:
            continue

        # Original side is the opposite of the close-order side.
        inferred_original_side = "SELL" if close_side == "BUY" else "BUY"
        pos = positions_by_key.get((ticker, inferred_original_side))
        if pos is None:
            continue

        r["original_side"] = inferred_original_side
        entry = pos.get("entry_price") or pos.get("price")
        if entry:
            r["original_price"] = float(entry)


def send_post_close_pnl_report(
    output_dir: str | Path | None = None,
    date_str: str | None = None,
    api_client: Any | None = None,
    refresh_fills: bool = True,
    re_snapshot: bool = True,
    recipients: Sequence[str] | None = None,
    dry_run: bool | None = None,
    credentials_path: str | Path | None = None,
    token_path: str | Path | None = None,
    from_email: str | None = None,
    send_timeout: float = 30.0,
) -> dict[str, Any]:
    """Re-fetch fills and snapshots after the closing auction and send the P&L report.

    Intended to be called from a separate post-close job (e.g. 15:35-15:40)
    after the exchange close (15:30). It updates ``close_execution_log.json``
    with actual fill prices, takes a fresh residual-position snapshot, then
    falls through to ``send_close_pnl_report_if_enabled``.

    Args:
        output_dir: Close output directory. If None, the latest
            ``results/...production_close_positions`` directory is used.
        date_str: Date string ``YYYYMMDD``. If None, inferred from the directory.
        api_client: Broker client to re-fetch fills and snapshots. If None,
            the report is generated from existing files.
        refresh_fills: If True and ``api_client`` is provided, re-fetch fill
            prices for orders that are still unfilled.
        re_snapshot: If True and ``api_client`` is provided, take fresh
            position and wallet snapshots after the close.
        recipients, dry_run, credentials_path, token_path, from_email,
        send_timeout: Passed through to ``send_close_pnl_report_if_enabled``.
    """
    if output_dir is None:
        output_dir = _find_latest_output_dir(DEFAULT_RESULTS_ROOT) or Path(DEFAULT_RESULTS_ROOT)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Close output directory not found: {output_dir}")

    resolved_date = _resolve_date_str(output_dir, date_str)

    # Re-fetch fill prices for orders that have not yet recorded a fill.
    if refresh_fills and api_client is not None:
        from leadlag.execution.pricing import fetch_fill_prices

        close_log = _load_json(output_dir / "close_execution_log.json") or {}
        close_results = close_log.get("close_results", [])
        if close_results:
            try:
                fetch_fill_prices(api_client, close_results, wait_seconds=0.0)

                # Backfill original price/side from the pre-close snapshot if
                # the close log was produced before these fields were added.
                pre_close_snapshot = _load_json(
                    output_dir / f"positions_close_{resolved_date}.json"
                ) or {}
                _backfill_close_metadata(close_log, pre_close_snapshot.get("positions", []))

                _save_json(output_dir / "close_execution_log.json", close_log)
                logger.info("Refreshed fill prices in %s", output_dir / "close_execution_log.json")
            except Exception:
                logger.exception("Failed to refresh fill prices; continuing with existing log")

    # Take fresh post-close snapshots of residual positions and wallet.
    if re_snapshot and api_client is not None:
        from leadlag.execution.output_ops import save_position_snapshot, save_wallet_snapshot

        try:
            save_position_snapshot(api_client, str(output_dir), label="pnl", date_str=resolved_date)
        except Exception:
            logger.exception("Failed to save post-close position snapshot")
        try:
            save_wallet_snapshot(api_client, str(output_dir), label="pnl", date_str=resolved_date)
        except Exception:
            logger.exception("Failed to save post-close wallet snapshot")

    # Generate / send the report using the post-close snapshots if available.
    return send_close_pnl_report_if_enabled(
        output_dir=output_dir,
        date_str=resolved_date,
        recipients=recipients,
        dry_run=dry_run,
        credentials_path=credentials_path,
        token_path=token_path,
        from_email=from_email,
        send_timeout=send_timeout,
        snapshot_label="pnl",
    )

