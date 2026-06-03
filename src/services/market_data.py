"""Market data services: open price fetching, gap computation, validation."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data_loader import JP_TICKERS, US_TICKERS, TOPIX_TICKER

logger = logging.getLogger(__name__)


def fetch_opens_from_google(
    tickers: Optional[list[str]] = None,
    allow_missing: bool = False,
) -> dict[str, float]:
    """Fetch real-time JP ETF prices from Google Finance as open proxies."""
    from bs4 import BeautifulSoup
    import requests
    import concurrent.futures

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
        "Fetching JP current real-time prices from Google Finance "
        "(mapped to open_price)..."
    )

    def _fetch_single(tk: str) -> tuple[str, Optional[float]]:
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
                for p in soup.find_all(attrs={"jsname": "Pdsbrc"}):
                    if "¥" in p.text:
                        price_div = p
                        break
            if not price_div:
                p_elements = soup.find_all(attrs={"jsname": "Pdsbrc"})
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
        message = (
            "Failed to fetch valid open prices from Google Finance for: "
            + ", ".join(failed_sorted)
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
        raise ValueError(
            f"CSV must have at least {len(JP_TICKERS)} rows, got {len(df)}"
        )
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
        if not np.isfinite(float(manual_opens.get(tk, np.nan)))
        or float(manual_opens[tk]) <= 0.0
    ]
    if invalid:
        raise ValueError(
            "Non-positive or invalid open prices for: " f"{', '.join(invalid)}"
        )


def validate_topix_open(manual_opens: dict[str, float]) -> float:
    """Validate TOPIX proxy open price and return it."""
    if TOPIX_TICKER not in manual_opens:
        raise ValueError(f"Missing open price for {TOPIX_TICKER}")
    value = float(manual_opens.get(TOPIX_TICKER, np.nan))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid open price for {TOPIX_TICKER}: {value}")
    return value


def normalize_to_tokyo_date(index: pd.Index) -> pd.DatetimeIndex:
    """Consistently normalize a DatetimeIndex to Tokyo timezone date.

    Args:
        index: pd.DatetimeIndex or pd.Index of datetime-like objects

    Returns:
        pd.DatetimeIndex normalized to midnight, timezone-naive
    """
    idx = pd.to_datetime(index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Tokyo")
    else:
        idx = idx.tz_convert("Asia/Tokyo")
    return idx.tz_localize(None).normalize()


def compute_gap_override(
    data: dict, trade_date: pd.Timestamp, manual_opens: dict[str, float]
) -> np.ndarray:
    """Compute gap overrides from raw data dict (standard path)."""
    jp_close = data["jp_close"].copy()
    jp_close.index = normalize_to_tokyo_date(jp_close.index)
    return compute_gap_from_jp_close(jp_close, trade_date, manual_opens)


def compute_gap_from_jp_close(
    jp_close: pd.DataFrame,
    trade_date: pd.Timestamp,
    manual_opens: dict[str, float],
) -> np.ndarray:
    """Compute gap override from a JP close DataFrame.

    Used by both the standard path (data dict) and the fast path
    (load_jp_close_from_cache) to share the same logic.
    """
    gaps = []
    for tk in JP_TICKERS:
        series = jp_close[tk].dropna()
        prev = series[series.index < trade_date]
        if len(prev) == 0:
            raise ValueError(
                f"Previous close not found for {tk} before {trade_date.date()}"
            )
        prev_close = float(prev.iloc[-1])
        open_price = float(manual_opens[tk])
        if prev_close <= 0:
            raise ValueError(f"Invalid previous close for {tk}: {prev_close}")
        gaps.append(open_price / prev_close - 1.0)

    return np.array(gaps, dtype=float)


def compute_topix_night_override(
    jp_close: pd.DataFrame,
    trade_date: pd.Timestamp,
    topix_open: float,
    topix_ticker: str = TOPIX_TICKER,
) -> float:
    """Compute TOPIX night return from cached close and current open."""
    jp_close = jp_close.copy()
    jp_close.index = normalize_to_tokyo_date(jp_close.index)

    if topix_ticker not in jp_close.columns:
        raise ValueError(f"TOPIX proxy close not found: {topix_ticker}")

    series = jp_close[topix_ticker].dropna()
    prev = series[series.index < trade_date]
    if len(prev) == 0:
        raise ValueError(
            f"Previous close not found for {topix_ticker} before {trade_date.date()}"
        )
    prev_close = float(prev.iloc[-1])
    if prev_close <= 0:
        raise ValueError(f"Invalid previous close for {topix_ticker}: {prev_close}")
    return float(topix_open) / prev_close - 1.0


def validate_us_returns_map(us_returns: dict[str, float]) -> dict[str, float]:
    """Validate and normalize US return map from API/cache."""
    missing: list[str] = []
    invalid: list[str] = []
    normalized: dict[str, float] = {}

    for tk in US_TICKERS:
        if tk not in us_returns:
            missing.append(tk)
            continue
        try:
            value = float(us_returns[tk])
        except (TypeError, ValueError):
            invalid.append(tk)
            continue
        if not np.isfinite(value):
            invalid.append(tk)
            continue
        normalized[tk] = value

    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if invalid:
            details.append(f"invalid={','.join(invalid)}")
        raise ValueError("Incomplete US returns: " + " | ".join(details))

    return normalized
