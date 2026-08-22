"""Japanese equity market calendar utilities.

Provides holiday detection for the Tokyo Stock Exchange (TSE) trading calendar.
Used by the CLI and scheduler to skip non-trading days automatically.

Primary strategy:
  1. Try ``jpholiday`` package (if installed) for authoritative holiday data.
  2. Fall back to a built-in static holiday table (updated annually).

Weekend check (Saturday/Sunday) is always applied regardless of strategy.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import cast

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in fallback holiday table (TSE closed days)
# Updated for 2025-2026. Update annually or install jpholiday for auto-updates.
# ---------------------------------------------------------------------------
_STATIC_HOLIDAYS: dict[int, set[date]] = {
    2025: {
        date(2025, 1, 1),    # 元日
        date(2025, 1, 2),    # 年末年始休場
        date(2025, 1, 3),    # 年末年始休場
        date(2025, 1, 13),   # 成人の日
        date(2025, 2, 11),   # 建国記念の日
        date(2025, 2, 23),   # 天皇誕生日
        date(2025, 2, 24),   # 振替休日
        date(2025, 3, 20),   # 春分の日
        date(2025, 4, 29),   # 昭和の日
        date(2025, 5, 3),    # 憲法記念日
        date(2025, 5, 4),    # みどりの日
        date(2025, 5, 5),    # こどもの日
        date(2025, 5, 6),    # 憲法記念日・みどりの日 振替休日
        date(2025, 7, 21),   # 海の日
        date(2025, 8, 11),   # 山の日
        date(2025, 9, 15),   # 敬老の日
        date(2025, 9, 23),   # 秋分の日
        date(2025, 10, 13),  # スポーツの日
        date(2025, 11, 3),   # 文化の日
        date(2025, 11, 23),  # 勤労感謝の日
        date(2025, 11, 24),  # 振替休日
        date(2025, 12, 31),  # 年末休業日
    },
    2026: {
        date(2026, 1, 1),    # 元日
        date(2026, 1, 2),    # 年末年始休場
        date(2026, 1, 3),    # 年末年始休場
        date(2026, 1, 12),   # 成人の日
        date(2026, 2, 11),   # 建国記念の日
        date(2026, 2, 23),   # 天皇誕生日
        date(2026, 3, 20),   # 春分の日
        date(2026, 4, 29),   # 昭和の日
        date(2026, 5, 3),    # 憲法記念日
        date(2026, 5, 4),    # みどりの日
        date(2026, 5, 5),    # こどもの日
        date(2026, 5, 6),    # 憲法記念日 振替休日
        date(2026, 7, 20),   # 海の日
        date(2026, 8, 11),   # 山の日
        date(2026, 9, 21),   # 敬老の日
        date(2026, 9, 22),   # 国民の休日（JPX休場）
        date(2026, 9, 23),   # 秋分の日
        date(2026, 10, 12),  # スポーツの日
        date(2026, 11, 3),   # 文化の日
        date(2026, 11, 23),  # 勤労感謝の日
        date(2026, 12, 31),  # 年末休業日
    },
    2027: {
        date(2027, 1, 1),    # 元日
        date(2027, 1, 2),    # 年末年始休場
        date(2027, 1, 3),    # 年末年始休場
        date(2027, 1, 11),   # 成人の日
        date(2027, 2, 11),   # 建国記念の日
        date(2027, 2, 23),   # 天皇誕生日
        date(2027, 3, 21),   # 春分の日
        date(2027, 3, 22),   # 振替休日
        date(2027, 4, 29),   # 昭和の日
        date(2027, 5, 3),    # 憲法記念日
        date(2027, 5, 4),    # みどりの日
        date(2027, 5, 5),    # こどもの日
        date(2027, 7, 19),   # 海の日
        date(2027, 8, 11),   # 山の日
        date(2027, 9, 20),   # 敬老の日
        date(2027, 9, 23),   # 秋分の日
        date(2027, 10, 11),  # スポーツの日
        date(2027, 11, 3),   # 文化の日
        date(2027, 11, 23),  # 勤労感謝の日
        date(2027, 12, 31),  # 年末休業日
    },
}


def _is_weekend(d: date) -> bool:
    """Return True if the given date is Saturday or Sunday."""
    return d.weekday() >= 5  # 5=Sat, 6=Sun


def _is_holiday_jpholiday(d: date) -> bool | None:
    """Try jpholiday package. Returns True/False, or None if unavailable."""
    try:
        import jpholiday

        return cast(bool, jpholiday.is_holiday(d))
    except ImportError:
        return None
    except Exception as e:
        logger.warning("jpholiday check failed for %s: %s", d, e)
        return None


def _is_holiday_static(d: date) -> bool:
    """Check built-in static holiday table."""
    holidays = _STATIC_HOLIDAYS.get(d.year)
    if holidays is None:
        logger.warning(
            "No static holiday table for year %d. "
            "Install jpholiday or update market_calendar.py.",
            d.year,
        )
        return False
    return d in holidays


def is_trading_day(d: date | datetime | None = None) -> bool:
    """Check if the given date is a TSE trading day.

    Args:
        d: Date to check. Defaults to today (JST).

    Returns:
        True if the date is a trading day (not weekend, not holiday).
    """
    if d is None:
        d = date.today()
    elif isinstance(d, datetime):
        d = d.date()

    if _is_weekend(d):
        return False

    # jpholiday covers public holidays; the static table covers JPX-specific
    # market closures that are not public holidays (e.g. 12/31, 1/2, 1/3) and
    # the exact JPX calendar for the years it includes. Use both sources.
    jp_hol = _is_holiday_jpholiday(d)
    if jp_hol:
        return False

    return not _is_holiday_static(d)


def is_market_closed(d: date | datetime | None = None) -> bool:
    """Inverse of is_trading_day. Returns True if market is closed."""
    return not is_trading_day(d)


def get_holiday_name(d: date | datetime | None = None) -> str | None:
    """Get the name of the holiday for the given date, if any.

    Returns None if it's a regular trading day or weekend.
    """
    if d is None:
        d = date.today()
    elif isinstance(d, datetime):
        d = d.date()

    if _is_weekend(d):
        return "Weekend"

    try:
        import jpholiday

        result = cast(str | None, jpholiday.is_holiday_name(d))
        if result:
            return result
    except ImportError:
        pass
    except Exception:
        pass

    return None


def previous_trading_day(d: date | datetime | None = None) -> date:
    """Return the nearest TSE trading day strictly before *d*.

    If *d* is None, defaults to today. Raises ValueError if a trading day
    cannot be found within a one-year defensive bound.
    """
    if d is None:
        d = date.today()
    elif isinstance(d, datetime):
        d = d.date()

    candidate = d - timedelta(days=1)
    for _ in range(366):
        if is_trading_day(candidate):
            return candidate
        candidate -= timedelta(days=1)

    raise ValueError(f"Could not find a TSE trading day before {d} within one year")


def next_trading_day(d: date | datetime | None = None) -> date:
    """Return the nearest TSE trading day strictly after *d*.

    If *d* is None, defaults to today. Raises ValueError if a trading day
    cannot be found within a one-year defensive bound.
    """
    if d is None:
        d = date.today()
    elif isinstance(d, datetime):
        d = d.date()

    candidate = d + timedelta(days=1)
    for _ in range(366):
        if is_trading_day(candidate):
            return candidate
        candidate += timedelta(days=1)

    raise ValueError(f"Could not find a TSE trading day after {d} within one year")


def count_tse_bdays(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count TSE trading days strictly between *start* and *end* (inclusive of end)."""
    count = 0
    d = start + timedelta(days=1)
    limit = 0
    while d <= end and limit < 1000:
        if is_trading_day(d.to_pydatetime()):
            count += 1
        d += timedelta(days=1)
        limit += 1
    return count
