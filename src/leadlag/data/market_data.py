"""Market data services: open price fetching, gap computation, validation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER

logger = logging.getLogger(__name__)


def fetch_opens_from_google(
    tickers: list[str] | None = None,
    allow_missing: bool = False,
) -> dict[str, float]:
    """Fetch real-time JP ETF prices from Google Finance as open proxies."""
    import concurrent.futures

    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.114 Safari/537.36"
        )
    }
    opens: dict[str, float] = {}
    target_tickers = tickers if tickers is not None else JP_TICKERS
    if not target_tickers:
        return opens
    logger.info(
        "Fetching JP current real-time prices from Google Finance (mapped to open_price)..."
    )

    def _fetch_single(tk: str) -> tuple[str, float | None]:
        code = tk.replace(".T", "")
        url = f"https://www.google.com/finance/quote/{code}:TYO"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            price_div = soup.find("div", class_="N6SYTe")
            if not price_div:
                price_div = soup.find("span", class_="N6SYTe")
            if not price_div:
                price_div = soup.find("div", class_="YMlKec fxKbKc")
            if not price_div:
                for p in soup.find_all(name=None, attrs={"jsname": "Pdsbrc"}):
                    if "¥" in p.text:
                        price_div = p
                        break
            if not price_div:
                p_elements = soup.find_all(name=None, attrs={"jsname": "Pdsbrc"})
                if p_elements:
                    price_div = p_elements[-1]

            if price_div:
                price_text = price_div.text.replace("¥", "").replace(",", "").strip()
                price = float(price_text)
            else:
                logger.warning("Google Finance price element not found for %s", tk)
                return tk, None
            logger.debug(f"  Fetched {tk}: {price}")
            return tk, price
        except Exception as e:
            logger.error(f"Failed to fetch {tk} from Google: {e}")
            return tk, None

    failed: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_tk = {executor.submit(_fetch_single, tk): tk for tk in target_tickers}
        for future in concurrent.futures.as_completed(future_to_tk):
            tk, price = future.result()
            if price is None or not np.isfinite(float(price)) or float(price) <= 0.0:
                failed.append(tk)
            else:
                opens[tk] = float(price)

    if failed:
        failed_sorted = sorted(failed)
        message = "Failed to fetch valid open prices from Google Finance for: " + ", ".join(
            failed_sorted
        )
        if allow_missing:
            logger.warning(message)
        else:
            raise ValueError(message)

    return opens


def load_opens_from_csv(csv_path: str) -> dict[str, float]:
    """Load TOPIX-17 opens from CSV (columns: ticker, open_price)."""
    df = pd.read_csv(csv_path, dtype={"ticker": str, "open_price": float})
    if len(df) < len(JP_TICKERS):
        raise ValueError(f"CSV must have at least {len(JP_TICKERS)} rows, got {len(df)}")
    parsed: dict[str, float] = {}
    for _, row in df.iterrows():
        tk = row["ticker"].strip()
        if tk not in JP_TICKERS and tk != TOPIX_TICKER:
            raise ValueError(f"Unknown ticker in CSV: {tk}")
        parsed[tk] = float(row["open_price"])

    missing = [tk for tk in JP_TICKERS if tk not in parsed]
    if missing:
        raise ValueError(f"Missing opens in CSV for: {', '.join(missing)}")
    return parsed


def validate_manual_opens(manual_opens: dict[str, float]) -> None:
    """Validate that manual open prices cover all tickers and are positive."""
    missing = [tk for tk in JP_TICKERS if tk not in manual_opens]
    if missing:
        raise ValueError(f"Missing open prices for: {', '.join(missing)}")

    invalid = [
        tk
        for tk in JP_TICKERS
        if not np.isfinite(float(manual_opens.get(tk, np.nan))) or float(manual_opens[tk]) <= 0.0
    ]
    if invalid:
        raise ValueError(f"Non-positive or invalid open prices for: {', '.join(invalid)}")


def validate_topix_open(manual_opens: dict[str, float]) -> float:
    """Validate TOPIX proxy open price and return it."""
    if TOPIX_TICKER not in manual_opens:
        raise ValueError(f"Missing open price for {TOPIX_TICKER}")
    value = float(manual_opens.get(TOPIX_TICKER, np.nan))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid open price for {TOPIX_TICKER}: {value}")
    return value
