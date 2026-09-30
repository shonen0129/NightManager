from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from leadlag.data.quote_snapshot import FrozenQuoteSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.reporting.ml_overlay_forward_outcome import (
    append_ml_overlay_forward_outcome,
    build_ml_overlay_forward_outcome,
    ensure_official_close_history_ready,
)

JST = ZoneInfo("Asia/Tokyo")


def _snapshot() -> FrozenQuoteSnapshot:
    quote_mid = {ticker: 100.0 for ticker in JP_TICKERS}
    return FrozenQuoteSnapshot(
        trade_date="2026-09-28",
        as_of=datetime(2026, 9, 28, 9, 10, 4, tzinfo=JST),
        request_started_at=datetime(2026, 9, 28, 9, 10, 3, tzinfo=JST),
        snapshot_id="quote-a",
        prices=quote_mid,
        observed_at={ticker: "2026-09-28T09:10:04+09:00" for ticker in JP_TICKERS},
        quote_rows={},
        source="tachibana:CLMMfdsGetMarketPrice",
        timestamp_source="local_response_receipt",
    )


def _history(close: str = "110.0") -> dict[str, list[dict[str, str]]]:
    return {
        ticker: [{"sDate": "20260928", "pDPP": close}]
        for ticker in JP_TICKERS
    }


def test_builds_complete_official_0910_to_close_outcome():
    outcome = build_ml_overlay_forward_outcome(
        trade_date="2026-09-28",
        input_digest="input-a",
        quote_snapshot=_snapshot(),
        close_history_by_ticker=_history(),
    )

    assert outcome["label_status"] == "complete"
    assert outcome["target_basis"] == "frozen_0910_quote_mid_to_official_close"
    assert outcome["target_source"] == "tachibana:CLMMfdsGetMarketPriceHistory:pDPP"
    assert set(outcome["target_returns"]) == set(JP_TICKERS)
    assert set(outcome["official_close_prices"]) == set(JP_TICKERS)
    assert outcome["target_returns"][JP_TICKERS[0]] == pytest.approx(0.1)


def test_build_rejects_missing_or_duplicate_session_close_rows():
    missing = _history()
    missing[JP_TICKERS[0]] = []
    with pytest.raises(ValueError, match="exactly one row"):
        build_ml_overlay_forward_outcome(
            trade_date="2026-09-28",
            input_digest="input-a",
            quote_snapshot=_snapshot(),
            close_history_by_ticker=missing,
        )

    duplicate = _history()
    duplicate[JP_TICKERS[0]].append({"sDate": "20260928", "pDPP": "111.0"})
    with pytest.raises(ValueError, match="exactly one row"):
        build_ml_overlay_forward_outcome(
            trade_date="2026-09-28",
            input_digest="input-a",
            quote_snapshot=_snapshot(),
            close_history_by_ticker=duplicate,
        )


def test_outcome_append_is_idempotent_and_refuses_label_replacement(tmp_path):
    output = tmp_path / "outcomes.jsonl"
    outcome = build_ml_overlay_forward_outcome(
        trade_date="2026-09-28",
        input_digest="input-a",
        quote_snapshot=_snapshot(),
        close_history_by_ticker=_history(),
    )

    assert append_ml_overlay_forward_outcome(output, outcome) is True
    assert append_ml_overlay_forward_outcome(output, outcome) is False

    corrected = build_ml_overlay_forward_outcome(
        trade_date="2026-09-28",
        input_digest="input-a",
        quote_snapshot=_snapshot(),
        close_history_by_ticker=_history("111.0"),
    )
    with pytest.raises(ValueError, match="refusing to replace"):
        append_ml_overlay_forward_outcome(output, corrected)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_official_close_history_cannot_be_read_before_refresh_time():
    with pytest.raises(ValueError, match="not ready"):
        ensure_official_close_history_ready(
            "2026-09-28",
            now=datetime(2026, 9, 29, 0, 59, 59, tzinfo=JST),
        )

    ensure_official_close_history_ready(
        "2026-09-28",
        now=datetime(2026, 9, 29, 1, 0, tzinfo=JST),
    )


def test_official_close_history_readiness_requires_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_official_close_history_ready(
            "2026-09-28", now=datetime(2026, 9, 29, 1, 0)
        )
