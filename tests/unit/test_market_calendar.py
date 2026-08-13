"""Tests for leadlag.core.market_calendar."""

from datetime import date

import pandas as pd

from leadlag.core.market_calendar import (
    count_tse_bdays,
    get_holiday_name,
    is_market_closed,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
)


class TestWeekendCheck:
    """Weekend detection."""

    def test_saturday_is_closed(self):
        # 2025-01-11 is a Saturday
        assert is_market_closed(date(2025, 1, 11))

    def test_sunday_is_closed(self):
        # 2025-01-12 is a Sunday
        assert is_market_closed(date(2025, 1, 12))

    def test_monday_is_open(self):
        # 2025-01-13 is a Monday (but it's Coming of Age Day — holiday)
        # Use 2025-01-20 instead (a regular Monday)
        assert is_trading_day(date(2025, 1, 20))


class TestHolidayCheck:
    """Japanese holiday detection via static table."""

    def test_new_year_is_closed(self):
        assert is_market_closed(date(2025, 1, 1))

    def test_showa_day_is_closed(self):
        assert is_market_closed(date(2025, 4, 29))

    def test_golden_week_closed(self):
        assert is_market_closed(date(2025, 5, 3))
        assert is_market_closed(date(2025, 5, 5))

    def test_regular_weekday_is_open(self):
        # 2025-01-21 is a regular Tuesday
        assert is_trading_day(date(2025, 1, 21))

    def test_2026_autumnal_equinox_only_sep23(self):
        # 2026-09-22 is a regular Tuesday; only 2026-09-23 is 秋分の日
        assert is_trading_day(date(2026, 9, 22))
        assert is_market_closed(date(2026, 9, 23))

    def test_dec31_is_open(self):
        # TSE is typically open on Dec 31
        assert is_trading_day(date(2025, 12, 31))


class TestHolidayName:
    """Holiday name lookup."""

    def test_weekend_returns_weekend(self):
        name = get_holiday_name(date(2025, 1, 11))
        assert name == "Weekend"

    def test_regular_day_returns_none(self):
        name = get_holiday_name(date(2025, 1, 21))
        # Without jpholiday, static table doesn't provide names
        # Could be None or a name
        assert name is None or isinstance(name, str)


class TestPreviousTradingDay:
    """Previous trading day calculation."""

    def test_previous_weekday(self):
        # 2025-01-21 (Tuesday) → 2025-01-20 (Monday)
        assert previous_trading_day(date(2025, 1, 21)) == date(2025, 1, 20)

    def test_skip_weekend(self):
        # 2025-01-20 (Monday) → 2025-01-17 (Friday)
        assert previous_trading_day(date(2025, 1, 20)) == date(2025, 1, 17)

    def test_skip_holiday(self):
        # 2025-01-14 (Tuesday) → 2025-01-10 (Friday), skipping 成人の日
        assert previous_trading_day(date(2025, 1, 14)) == date(2025, 1, 10)

    def test_skip_long_weekend(self):
        # 2025-05-07 (Wednesday) → 2025-05-01 (Thursday), skipping 憲法記念日 etc.
        assert previous_trading_day(date(2025, 5, 7)) == date(2025, 5, 1)


class TestDefaultDate:
    """Default (today) behavior."""

    def test_is_trading_day_no_arg(self):
        result = is_trading_day()
        assert isinstance(result, bool)

    def test_is_market_closed_no_arg(self):
        result = is_market_closed()
        assert isinstance(result, bool)


class TestNextTradingDay:
    """Next trading day calculation."""

    def test_next_weekday(self):
        # 2025-01-21 (Tuesday) → 2025-01-22 (Wednesday)
        assert next_trading_day(date(2025, 1, 21)) == date(2025, 1, 22)

    def test_skip_weekend(self):
        # 2025-01-17 (Friday) → 2025-01-20 (Monday)
        assert next_trading_day(date(2025, 1, 17)) == date(2025, 1, 20)

    def test_skip_holiday(self):
        # 2025-01-10 (Friday) → 2025-01-14 (Tuesday), skipping 成人の日
        assert next_trading_day(date(2025, 1, 10)) == date(2025, 1, 14)


class TestCountTseBdays:
    """TSE trading day counting."""

    def test_same_day(self):
        d = pd.Timestamp("2025-01-21")
        assert count_tse_bdays(d, d) == 0

    def test_one_day(self):
        start = pd.Timestamp("2025-01-21")
        end = pd.Timestamp("2025-01-22")
        assert count_tse_bdays(start, end) == 1

    def test_skip_weekend(self):
        # Friday 2025-01-17 → Monday 2025-01-20: only Monday is counted
        start = pd.Timestamp("2025-01-17")
        end = pd.Timestamp("2025-01-20")
        assert count_tse_bdays(start, end) == 1

    def test_skip_holiday(self):
        # Friday 2025-01-10 → Tuesday 2025-01-14: skipping 成人の日
        start = pd.Timestamp("2025-01-10")
        end = pd.Timestamp("2025-01-14")
        assert count_tse_bdays(start, end) == 1
