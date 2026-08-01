#!/usr/bin/env python
"""Send or generate the daily close P&L report.

This script is designed to run as an independent post-close job (e.g. 15:35-15:40)
after the exchange close (15:30). It re-fetches fill prices and fresh residual
position/wallet snapshots before generating and optionally emailing the report.

Usage:

    # 1) Authorize Gmail send scope once (interactive):
    python tools/production/send_daily_close_pnl_report.py --authorize

    # 2) Dry-run from the latest close output directory
    python tools/production/send_daily_close_pnl_report.py

    # 3) Send to comma-separated recipients
    LEADLAG_PNL_REPORT_SEND=1 \
    LEADLAG_PNL_REPORT_RECIPIENTS=you@example.com \
    python tools/production/send_daily_close_pnl_report.py

    # 4) Override output directory and recipients from CLI
    python tools/production/send_daily_close_pnl_report.py \
        --output-dir results/20260731_145000_production_close_positions \
        --recipients you@example.com,you2@example.com \
        --send
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.helpers import build_api_client
from leadlag.reporting.daily_pnl_report import send_post_close_pnl_report
from leadlag.reporting.gmail_sender import authorize_gmail_send


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send daily close P&L report via Gmail API")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Close output directory (defaults to the latest results/ production_close_positions run)",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Trade date YYYYMMDD (defaults to the date in the output dir name or today)",
    )
    p.add_argument(
        "--recipients",
        default=None,
        help="Comma-separated recipient email addresses (overrides env)",
    )
    p.add_argument(
        "--send",
        action="store_true",
        default=None,
        help="Actually send the email (default: dry-run unless LEADLAG_PNL_REPORT_SEND=1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Force dry-run even if LEADLAG_PNL_REPORT_SEND is set",
    )
    p.add_argument(
        "--no-api",
        action="store_true",
        help="Skip broker API; generate the report from already-saved files only",
    )
    p.add_argument(
        "--api-url",
        default=None,
        help="Broker API URL (defaults to config or env)",
    )
    p.add_argument(
        "--api-token",
        default=None,
        help="Broker API token (defaults to config or env)",
    )
    p.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Use a dry-run broker client (do not call live APIs)",
    )
    p.add_argument(
        "--no-refresh-fills",
        action="store_true",
        help="Do not re-fetch fill prices from the broker",
    )
    p.add_argument(
        "--no-re-snapshot",
        action="store_true",
        help="Do not re-fetch position/wallet snapshots from the broker",
    )
    p.add_argument(
        "--credentials",
        default=None,
        help="Path to Gmail API credentials.json",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Path to Gmail send token.json",
    )
    p.add_argument(
        "--from-email",
        default=None,
        help="Sender display address (default: 'me')",
    )
    p.add_argument(
        "--authorize",
        action="store_true",
        help="Run the interactive Gmail send OAuth flow once and save the token",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Gmail API send timeout in seconds",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    if args.authorize:
        credentials = args.credentials or os.environ.get(
            "LEADLAG_GMAIL_CREDENTIALS", "creds/credentials.json"
        )
        token = args.token or os.environ.get("LEADLAG_GMAIL_SEND_TOKEN", "creds/token_gmail_send.json")
        authorize_gmail_send(
            credentials_path=Path(ROOT / credentials),
            token_path=Path(ROOT / token),
            from_email=args.from_email,
        )
        return 0

    recipients = None
    if args.recipients:
        recipients = [r.strip() for r in args.recipients.split(",") if r.strip()]

    # CLI --send / --dry-run override env, but --dry-run wins.
    dry_run: bool | None = None
    if args.dry_run:
        dry_run = True
    elif args.send is None:
        # Not specified on CLI; let the library decide from env.
        dry_run = None
    elif args.send:
        dry_run = False
    else:
        dry_run = True

    # Resolve credentials/token to absolute paths if relative.
    credentials_path = args.credentials
    if credentials_path:
        credentials_path = ROOT / credentials_path
    token_path = args.token
    if token_path:
        token_path = ROOT / token_path

    output_dir = args.output_dir
    if output_dir:
        output_dir = ROOT / output_dir

    # Build a broker client unless --no-api is specified.
    api_client = None
    if not args.no_api:
        try:
            api_client = build_api_client(args.api_url, args.api_token, args.api_dry_run)
        except Exception:
            logging.exception("Failed to build broker client; continuing with existing files only")

    try:
        # The scheduled post-close job re-fetches fills and snapshots, then sends.
        result = send_post_close_pnl_report(
            output_dir=output_dir,
            date_str=args.date,
            api_client=api_client,
            refresh_fills=not args.no_refresh_fills,
            re_snapshot=not args.no_re_snapshot,
            recipients=recipients,
            dry_run=dry_run,
            credentials_path=credentials_path,
            token_path=token_path,
            from_email=args.from_email,
            send_timeout=args.timeout,
        )
    finally:
        if api_client is not None:
            try:
                api_client.close()
            except Exception:
                logging.exception("Failed to close broker client")

    if result.get("error"):
        logging.error("P&L report failed: %s", result["error"])
        return 1

    logging.info("Report path: %s", result["report_path"])
    logging.info("Sent: %s", result["sent"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
