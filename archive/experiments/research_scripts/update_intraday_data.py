#!/usr/bin/env python
"""update_intraday_data.py — accumulate intraday 1m/5m market data.

This script should be run periodically (e.g., daily or weekly via cron) to build 
a long-term historical database of 1-minute and 5-minute ETF prices.
Yahoo Finance limits 1m data to 7 days and 5m data to 60 days.
"""

import sys
import os
import logging

# Add src/ to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.downloader import update_intraday_cache

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting intraday data accumulation...")
    try:
        update_intraday_cache()
        logger.info("Intraday data accumulation completed successfully.")
    except Exception as e:
        logger.error("Error during intraday accumulation: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
