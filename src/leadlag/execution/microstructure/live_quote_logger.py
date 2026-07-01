from __future__ import annotations

import logging
import os
import time
from datetime import datetime, time as dt_time
from typing import Any
import pandas as pd

from leadlag.execution.order_book_schema import OrderBookSnapshot, from_api_price_response, validate_quote

logger = logging.getLogger(__name__)

def fetch_quote_snapshot(
    api_client: Any,
    tickers: list[str],
    enabled: bool = False
) -> list[OrderBookSnapshot]:
    """Fetches quote snapshots for all tickers.

    If enabled is False or api_client is None, returns stubs with lob_available=False
    and cost_source='not_configured'.
    """
    timestamp = datetime.now().isoformat()
    snapshots = []

    if not enabled or api_client is None:
        for ticker in tickers:
            snapshots.append(
                OrderBookSnapshot(
                    ticker=ticker,
                    timestamp=timestamp,
                    lob_available=False,
                    cost_source="not_configured"
                )
            )
        return snapshots

    try:
        # Strip '.T' for the Tachibana API request if necessary
        api_codes = [t.replace(".T", "") for t in tickers]
        raw_quotes = api_client.get_price(api_codes)
        
        # Build map of returned codes to quickly locate items
        quote_map = {item.get("sIssueCode"): item for item in raw_quotes if item.get("sIssueCode")}

        for ticker in tickers:
            code = ticker.replace(".T", "")
            api_item = quote_map.get(code)
            if api_item:
                snapshot = from_api_price_response(api_item, timestamp)
            else:
                snapshot = OrderBookSnapshot(
                    ticker=ticker,
                    timestamp=timestamp,
                    lob_available=False,
                    cost_source="api_error"
                )
            snapshots.append(snapshot)

    except Exception as e:
        logger.error(f"Error fetching quotes from API: {e}")
        for ticker in tickers:
            snapshots.append(
                OrderBookSnapshot(
                    ticker=ticker,
                    timestamp=timestamp,
                    lob_available=False,
                    cost_source="api_error"
                )
            )

    return snapshots


def append_to_parquet(snapshots: list[OrderBookSnapshot], output_path: str) -> None:
    """Appends order book snapshots to a parquet file."""
    if not snapshots:
        return

    records = [s.to_dict() for s in snapshots]
    df_new = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if os.path.exists(output_path):
        try:
            df_old = pd.read_parquet(output_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_parquet(output_path, index=False)
            logger.info(f"Appended {len(snapshots)} rows to {output_path}. Total rows: {len(df_combined)}")
        except Exception as e:
            logger.error(f"Failed to append to existing parquet {output_path}: {e}. Overwriting.")
            df_new.to_parquet(output_path, index=False)
    else:
        df_new.to_parquet(output_path, index=False)
        logger.info(f"Created new parquet file {output_path} with {len(snapshots)} rows.")


def log_quote_loop(
    api_client: Any,
    tickers: list[str],
    start_time: dt_time,
    end_time: dt_time,
    interval_sec: float = 1.0,
    output_path: str = "artifacts/sprint2c_lob_slippage/logs/quote_log.parquet",
    enabled: bool = False
) -> dict[str, int]:
    """Runs a time-windowed loop to log quotes.

    Executes loop during the time window [start_time, end_time].
    Returns execution statistics.
    """
    logger.info(f"Starting quote log loop. Schedule: {start_time} to {end_time}. Interval: {interval_sec}s. Enabled: {enabled}")
    
    lob_available_count = 0
    api_error_count = 0
    not_configured_count = 0
    total_iterations = 0

    while True:
        now_time = datetime.now().time()
        if now_time < start_time:
            # Wait until start time
            time.sleep(0.5)
            continue
        if now_time > end_time:
            # Past end time, stop
            break

        start_loop = time.time()
        snapshots = fetch_quote_snapshot(api_client, tickers, enabled=enabled)
        
        # Update statistics
        for s in snapshots:
            if s.lob_available:
                lob_available_count += 1
            elif s.cost_source == "api_error":
                api_error_count += 1
            elif s.cost_source == "not_configured":
                not_configured_count += 1

        append_to_parquet(snapshots, output_path)
        total_iterations += 1

        elapsed = time.time() - start_loop
        sleep_time = max(0.0, interval_sec - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info(f"Quote log loop finished. Iterations: {total_iterations}")
    return {
        "lob_available_count": lob_available_count,
        "api_error_count": api_error_count,
        "not_configured_count": not_configured_count,
        "total_iterations": total_iterations
    }
