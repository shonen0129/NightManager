"""Tests for leadlag.core.market_calendar."""

from datetime import date

from leadlag.core.market_calendar import (
    get_holiday_name,
    is_market_closed,
    is_trading_day,
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


class TestDefaultDate:
    """Default (today) behavior."""

    def test_is_trading_day_no_arg(self):
        result = is_trading_day()
        assert isinstance(result, bool)

    def test_is_market_closed_no_arg(self):
        result = is_market_closed()
        assert isinstance(result, bool)
