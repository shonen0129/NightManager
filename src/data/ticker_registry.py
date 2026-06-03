"""Ticker registry — single source of truth for all asset identifiers.

Defines the US ETF universe (signal source) and JP ETF universe (trade target),
plus conversion utilities between yfinance format and broker-specific codes.

All modules must import tickers and asset counts from here rather than
defining their own constants.

Examples::

    from data.ticker_registry import US_TICKERS, JP_TICKERS, N_US, N_JP, N_TOTAL
    from data.ticker_registry import yf_to_kabu, kabu_to_yf
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# US ETF universe — Select Sector SPDRs & Style ETFs (signal source, N_US = 15)
# Order is fixed and used as vector/matrix row indices throughout.
# ---------------------------------------------------------------------------
US_TICKERS: list[str] = [
    "XLB",   # 1  Materials
    "XLC",   # 2  Communication Services
    "XLE",   # 3  Energy
    "XLF",   # 4  Financials
    "XLI",   # 5  Industrials
    "XLK",   # 6  Information Technology
    "XLP",   # 7  Consumer Staples
    "XLRE",  # 8  Real Estate
    "XLU",   # 9  Utilities
    "XLV",   # 10 Health Care
    "XLY",   # 11 Consumer Discretionary
    "MTUM",  # 12 Momentum
    "VLUE",  # 13 Value
    "IUSG",  # 14 Growth
    "USMV",  # 15 Min Vol
]

# ---------------------------------------------------------------------------
# JP ETF universe — NEXT FUNDS TOPIX-17 series (trade target, N_JP = 17)
# Using yfinance format ("XXXX.T"). Order matches §3.2 of 運用方針書.
# ---------------------------------------------------------------------------
JP_TICKERS: list[str] = [f"{t}.T" for t in range(1617, 1634)]
# 1617.T 食品, 1618.T エネルギー資源, 1619.T 建設・資材, 1620.T 素材・化学,
# 1621.T 医薬品, 1622.T 自動車・輸送機, 1623.T 鉄鋼・非鉄, 1624.T 機械,
# 1625.T 電機・精密, 1626.T 情報通信・サービス, 1627.T 電力・ガス,
# 1628.T 運輸・物流, 1629.T 商社・卸売, 1630.T 小売, 1631.T 銀行,
# 1632.T 金融（除く銀行）, 1633.T 不動産

# TOPIX proxy ticker (used for beta computation and overnight return)
TOPIX_TICKER: str = "1306.T"

# Convenience list for download (JP ETFs + TOPIX proxy)
JP_TICKERS_WITH_TOPIX: list[str] = JP_TICKERS + [TOPIX_TICKER]

# ---------------------------------------------------------------------------
# Asset counts (computed from the above lists — do NOT hard-code elsewhere)
# ---------------------------------------------------------------------------
N_US: int = len(US_TICKERS)     # 15
N_JP: int = len(JP_TICKERS)     # 17
N_TOTAL: int = N_US + N_JP      # 32

# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by config.py and legacy imports)
# ---------------------------------------------------------------------------
N_US_ASSETS: int = N_US
N_JP_ASSETS: int = N_JP
N_TOTAL_ASSETS: int = N_TOTAL

# ---------------------------------------------------------------------------
# Lot-size overrides (broker-specific: some ETFs trade in units > 1)
# ---------------------------------------------------------------------------
LOT_SIZES: dict[str, int] = {
    "1629.T": 10,
    "1629": 10,   # kabu bare-code form
}


def lot_size_for(ticker: str) -> int:
    """Return the lot size for *ticker* (default 1 for most ETFs).

    Args:
        ticker: Ticker in any format ("1629.T", "1629", etc.)

    Returns:
        Lot size (minimum tradable unit in shares)
    """
    lot = LOT_SIZES.get(ticker)
    if lot is None and ticker.endswith(".T"):
        lot = LOT_SIZES.get(ticker.replace(".T", ""))
    return int(lot) if lot and lot >= 1 else 1


# ---------------------------------------------------------------------------
# Ticker format converters
# ---------------------------------------------------------------------------


def yf_to_kabu(ticker: str) -> str:
    """Convert yfinance format to kabu bare code.

    Examples::

        yf_to_kabu("1617.T") → "1617"
        yf_to_kabu("XLB")    → "XLB"   # US tickers unchanged
    """
    return ticker.replace(".T", "")


def kabu_to_yf(code: str) -> str:
    """Convert kabu bare code to yfinance format.

    Only appends ".T" for numeric codes (JP ETFs).

    Examples::

        kabu_to_yf("1617")  → "1617.T"
        kabu_to_yf("XLB")   → "XLB"
    """
    if code and not code.endswith(".T") and code.isdigit():
        return f"{code}.T"
    return code


def is_jp_ticker(ticker: str) -> bool:
    """Return True if *ticker* is a Japanese ETF."""
    clean = ticker.replace(".T", "")
    return clean.isdigit()


def is_us_ticker(ticker: str) -> bool:
    """Return True if *ticker* is a US ETF."""
    return ticker.upper() in US_TICKERS
