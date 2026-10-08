"""Market decisions must be invariant to the host's timezone."""

from __future__ import annotations

import argparse
import ast
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from leadlag import cli
from leadlag.core import market_calendar
from leadlag.data import market_data_cache
from leadlag.utils import timestamps


@pytest.fixture(params=["Asia/Tokyo", "UTC", "America/New_York"])
def market_clock(request, monkeypatch):
    """Freeze an absolute instant while preserving actual host TZ behavior."""
    if not hasattr(time, "tzset"):
        pytest.skip("host TZ regression requires time.tzset")
    with monkeypatch.context() as context:
        context.setenv("TZ", request.param)
        time.tzset()

        def freeze(instant: str) -> None:
            absolute = datetime.fromisoformat(instant).astimezone(UTC)

            class FrozenDatetime(datetime):
                @classmethod
                def now(cls, tz=None):
                    if tz is None:
                        return absolute.astimezone().replace(tzinfo=None)
                    return absolute.astimezone(tz)

            context.setattr(timestamps, "datetime", FrozenDatetime)

        yield freeze
    time.tzset()


def test_clock_returns_aware_jst_and_market_date(market_clock):
    market_clock("2025-01-19T23:10:00+00:00")
    assert timestamps.jst_now().isoformat() == "2025-01-20T08:10:00+09:00"
    assert timestamps.jst_today() == date(2025, 1, 20)


@pytest.mark.parametrize(
    ("instant", "phase"),
    [("2025-01-20T09:10:00+09:00", "decision"),
     ("2025-01-20T09:15:00+09:00", "close"),
     ("2025-01-20T15:30:00+09:00", "close")],
)
def test_daily_phase_uses_jst(market_clock, monkeypatch, instant, phase):
    market_clock(instant)
    calls = []
    monkeypatch.setattr(cli, "_handle_decision", lambda args: calls.append("decision") or 11)
    monkeypatch.setattr(cli, "_handle_close", lambda args: calls.append("close") or 12)
    assert cli._handle_daily(argparse.Namespace(decision_cutoff="09:15")) == (
        11 if phase == "decision" else 12
    )
    assert calls == [phase]


@pytest.mark.parametrize("command", ["decision", "daily", "close"])
@pytest.mark.parametrize("day", ["2025-01-20", "2025-01-19", "2025-01-13"])
def test_cli_market_day_skip_uses_jst(market_clock, monkeypatch, command, day):
    market_clock(f"{day}T09:10:00+09:00")
    calls = []
    for phase in ("decision", "daily", "close"):
        monkeypatch.setattr(cli, f"_handle_{phase}", lambda args: calls.append(args.command) or 13)
    is_open = day == "2025-01-20"
    assert cli.main([command]) == (13 if is_open else 0)
    assert calls == ([command] if is_open else [])


def test_calendar_implicit_today_is_jst(market_clock):
    market_clock("2025-01-20T08:10:00+09:00")
    assert market_calendar.is_trading_day()
    assert not market_calendar.is_market_closed()
    assert market_calendar.get_holiday_name() is None
    assert market_calendar.previous_trading_day() == date(2025, 1, 17)
    assert market_calendar.next_trading_day() == date(2025, 1, 21)
    # Explicit aware instants must use the same market date.
    assert market_calendar.is_trading_day(datetime.fromisoformat("2025-01-19T23:10:00+00:00"))


@pytest.mark.parametrize("last", ["2025-01-21", "2025-01-20T15:00:00+00:00"])
def test_stale_cache_boundary_rejected_in_all_host_timezones(market_clock, last):
    market_clock("2025-01-23T08:00:00+09:00")
    frame = pd.DataFrame({"value": [1]}, index=pd.DatetimeIndex([last]))
    with pytest.raises(RuntimeError, match="2 TSE trading days old"):
        market_data_cache._check_df_exec_staleness(frame, 1)


def test_future_and_holiday_cache_boundaries_use_jst(market_clock):
    market_clock("2025-01-14T08:00:00+09:00")
    frame = pd.DataFrame({"value": [1]}, index=pd.DatetimeIndex(["2025-01-10"]))
    market_data_cache._check_df_exec_staleness(frame, 1)  # weekend + Jan 13 holiday
    with pytest.raises(RuntimeError, match="1 TSE trading days old"):
        market_data_cache._check_df_exec_staleness(frame, 0)
    frame.index = pd.DatetimeIndex(["2025-01-15"])
    with pytest.raises(RuntimeError, match="in the future"):
        market_data_cache._check_df_exec_staleness(frame, 1)


@pytest.mark.parametrize("path", [
    "src/leadlag/cli.py", "src/leadlag/core/market_calendar.py",
    "src/leadlag/data/market_data_cache.py", "src/leadlag/utils/timestamps.py",
])
def test_market_clock_boundaries_reject_host_local_today(path):
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "today", f"host-local today in {path}:{node.lineno}"
            if node.func.attr == "now":
                assert node.args or node.keywords, f"naive now in {path}:{node.lineno}"
