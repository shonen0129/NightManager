"""tests/unit/test_daily_pnl_report.py

Unit tests for close P&L report generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leadlag.broker.base import WalletInfo
from leadlag.reporting.daily_pnl_report import (
    compute_realized_pnl,
    generate_close_pnl_report,
    save_close_pnl_report,
    send_close_pnl_report_if_enabled,
    send_post_close_pnl_report,
)


def _make_close_result(
    *,
    ticker: str = "1570.T",
    side: str = "SELL",
    original_side: str = "BUY",
    quantity: int = 100,
    original_price: float | None = 1000.0,
    fill_price: float | None = 1050.0,
    fill_quantity: int | None = None,
    fee: float = 100.0,
    status: str = "SUBMITTED",
) -> dict:
    if fill_quantity is None:
        fill_quantity = quantity
    return {
        "ticker": ticker,
        "side": side,
        "original_side": original_side,
        "quantity": quantity,
        "original_price": original_price,
        "fill_price": fill_price,
        "fill_quantity": fill_quantity,
        "fill_status": "FULLY_FILLED" if fill_price is not None else "PENDING",
        "fill_detail": {"sBaiBaiTesuryo": fee},
        "status": status,
    }


def test_compute_realized_pnl_long_close():
    """Realized P&L for a long position closed at a higher price."""
    result = _make_close_result(
        side="SELL",
        original_side="BUY",
        quantity=100,
        original_price=1000.0,
        fill_price=1050.0,
        fee=100.0,
    )
    pnl_list = compute_realized_pnl([result])
    assert len(pnl_list) == 1
    assert pnl_list[0].realized_pnl == pytest.approx((1050.0 - 1000.0) * 100 - 100.0)
    assert pnl_list[0].ticker == "1570.T"


def test_compute_realized_pnl_short_close():
    """Realized P&L for a short position closed (buy back) at a lower price."""
    result = _make_close_result(
        side="BUY",
        original_side="SELL",
        quantity=100,
        original_price=1000.0,
        fill_price=950.0,
        fee=100.0,
    )
    pnl_list = compute_realized_pnl([result])
    assert len(pnl_list) == 1
    assert pnl_list[0].realized_pnl == pytest.approx((1000.0 - 950.0) * 100 - 100.0)


def test_compute_realized_pnl_skips_unfilled():
    """Unfilled orders should not contribute realized P&L."""
    result = _make_close_result(fill_price=None)
    pnl_list = compute_realized_pnl([result])
    assert len(pnl_list) == 0


def test_generate_close_pnl_report(tmp_path: Path):
    """Report generation loads artifacts and produces Markdown."""
    output_dir = tmp_path / "20260731_145000_production_close_positions"
    output_dir.mkdir()

    close_log = {
        "dry_run": False,
        "close_orders_count": 2,
        "close_results": [
            _make_close_result(
                ticker="1570.T",
                side="SELL",
                original_side="BUY",
                quantity=100,
                original_price=1000.0,
                fill_price=1050.0,
                fee=100.0,
            ),
            _make_close_result(
                ticker="1617.T",
                side="BUY",
                original_side="SELL",
                quantity=50,
                original_price=2000.0,
                fill_price=1950.0,
                fee=50.0,
            ),
        ],
    }
    positions = {
        "timestamp": "2026-07-31T14:55:00",
        "label": "close",
        "positions": [
            {
                "ticker": "1570.T",
                "side": "BUY",
                "quantity": 25,
                "entry_price": 1000.0,
                "evaluation_price": 1050.0,
                "unrealized_pnl": 1250.0,
                "unrealized_pnl_pct": 5.0,
            },
        ],
        "position_count": 1,
        "total_unrealized_pnl": 1250.0,
    }
    wallet = {
        "timestamp": "2026-07-31T14:55:00",
        "label": "close",
        "cash_available": 1_000_000.0,
        "margin_available": 500_000.0,
        "hosyoukin_ritu": 35.0,
        "sOisyouHasseiFlg": "0",
        "sTatekaekinHasseiFlg": "0",
    }

    (output_dir / "close_execution_log.json").write_text(
        json.dumps(close_log, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "positions_close_20260731.json").write_text(
        json.dumps(positions, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "wallet_close_20260731.json").write_text(
        json.dumps(wallet, ensure_ascii=False), encoding="utf-8"
    )

    report_md, summary = generate_close_pnl_report(output_dir, "20260731")

    # Realized: (1050-1000)*100 - 100 + (2000-1950)*50 - 50 = 4900 + 2450 = 7350
    assert summary["total_realized_pnl"] == pytest.approx(7350.0)
    assert summary["total_unrealized_pnl"] == pytest.approx(1250.0)
    assert summary["total_daily_pnl"] == pytest.approx(8600.0)
    assert summary["realized_count"] == 2
    assert "日米ラグ 引け損益レポート" in report_md
    assert "1570.T" in report_md
    assert "1617.T" in report_md


def test_save_close_pnl_report(tmp_path: Path):
    """Save a generated report to disk."""
    output_dir = tmp_path
    report_md = "# Test Report"
    path = save_close_pnl_report(report_md, output_dir, "20260731")
    assert path.exists()
    assert path.name == "daily_pnl_report_20260731.md"
    assert path.read_text(encoding="utf-8") == report_md


def test_send_close_pnl_report_if_enabled_dry_run(tmp_path: Path, monkeypatch):
    """Without env and without --send, the report is generated but not sent."""
    output_dir = tmp_path / "20260731_145000_production_close_positions"
    output_dir.mkdir()

    close_log = {
        "dry_run": True,
        "close_orders_count": 0,
        "close_results": [],
    }
    (output_dir / "close_execution_log.json").write_text(
        json.dumps(close_log, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "positions_close_20260731.json").write_text(
        json.dumps({"positions": [], "total_unrealized_pnl": 0.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "wallet_close_20260731.json").write_text(
        json.dumps({"cash_available": 0.0}, ensure_ascii=False), encoding="utf-8"
    )

    monkeypatch.delenv("LEADLAG_PNL_REPORT_SEND", raising=False)
    result = send_close_pnl_report_if_enabled(output_dir=output_dir, date_str="20260731")

    assert result["dry_run"] is True
    assert result["sent"] is False
    assert result["error"] is None
    assert Path(result["report_path"]).exists()
    assert Path(result["summary_path"]).exists()


def test_send_close_pnl_report_if_enabled_does_not_send_without_recipients(tmp_path: Path, monkeypatch):
    """Even with LEADLAG_PNL_REPORT_SEND=1, no email is sent without recipients."""
    output_dir = tmp_path / "20260731_145000_production_close_positions"
    output_dir.mkdir()

    close_log = {
        "dry_run": True,
        "close_orders_count": 0,
        "close_results": [],
    }
    (output_dir / "close_execution_log.json").write_text(
        json.dumps(close_log, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "positions_close_20260731.json").write_text(
        json.dumps({"positions": [], "total_unrealized_pnl": 0.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "wallet_close_20260731.json").write_text(
        json.dumps({"cash_available": 0.0}, ensure_ascii=False), encoding="utf-8"
    )

    monkeypatch.setenv("LEADLAG_PNL_REPORT_SEND", "1")
    monkeypatch.delenv("LEADLAG_PNL_REPORT_RECIPIENTS", raising=False)
    result = send_close_pnl_report_if_enabled(output_dir=output_dir, date_str="20260731")

    assert result["dry_run"] is False
    assert result["sent"] is False
    assert result["error"] is None


def test_send_post_close_pnl_report_refills_and_saves_pnl_snapshots(tmp_path: Path, monkeypatch):
    """Post-close report refreshes fills and saves pnl-labeled snapshots."""
    output_dir = tmp_path / "20260731_145000_production_close_positions"
    output_dir.mkdir()

    close_log = {
        "dry_run": False,
        "close_orders_count": 1,
        "close_results": [
            {
                "ticker": "1570.T",
                "side": "SELL",
                "original_side": "BUY",
                "quantity": 100,
                "original_price": 1000.0,
                "fill_price": None,
                "order_id": "ORDER-123",
                "status": "SUBMITTED",
            },
        ],
    }
    (output_dir / "close_execution_log.json").write_text(
        json.dumps(close_log, ensure_ascii=False), encoding="utf-8"
    )

    # Mock fetch_fill_prices to fill the missing price.
    def _mock_fetch_fill_prices(api_client, order_results, wait_seconds=0.0):
        for r in order_results:
            if r.get("fill_price") is None:
                r["fill_price"] = 1050.0
                r["fill_quantity"] = r.get("quantity", 100)
                r["fill_status"] = "FULLY_FILLED"
                r["fill_detail"] = {"sBaiBaiTesuryo": 100.0}

    monkeypatch.setattr(
        "leadlag.execution.pricing.fetch_fill_prices", _mock_fetch_fill_prices
    )

    # Minimal broker client stub.
    class _StubBroker:
        def get_positions(self):
            from leadlag.broker.base import Position

            return [
                Position(
                    ticker="1570.T",
                    side="BUY",
                    quantity=25,
                    price=1000.0,
                    execution_id="EXEC-1",
                    margin_trade_type=3,
                    account_type=4,
                    extra={},
                ),
            ]

        def get_wallet(self):
            return WalletInfo(
                cash_available=1_000_000.0,
                margin_available=500_000.0,
            )

        def close(self):
            pass

    api_client = _StubBroker()

    result = send_post_close_pnl_report(
        output_dir=output_dir,
        date_str="20260731",
        api_client=api_client,
        recipients=None,
        dry_run=True,
    )

    assert result["error"] is None
    assert Path(result["report_path"]).exists()

    # close_execution_log should have been updated with the fill.
    updated_log = json.loads((output_dir / "close_execution_log.json").read_text(encoding="utf-8"))
    assert updated_log["close_results"][0]["fill_price"] == 1050.0

    # pnl-labeled snapshots should have been written.
    assert (output_dir / "positions_pnl_20260731.json").exists()
    assert (output_dir / "wallet_pnl_20260731.json").exists()

    # The generated report should reflect the realized P&L.
    report_md = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "日米ラグ 引け損益レポート" in report_md
