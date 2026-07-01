import pytest
from leadlag.execution.microstructure.order_book_schema import OrderBookSnapshot, validate_quote, from_api_price_response
from leadlag.execution.microstructure.order_book_cost import (
    compute_mid_price,
    compute_quoted_spread_bps,
    compute_depth_jpy,
    estimate_market_order_fill_price,
    estimate_lob_slippage_bps,
    compute_order_to_depth_ratio,
    LobNotAvailable
)
from leadlag.execution.microstructure.slippage_model import compute_entry_cost_bps, compute_exit_cost_bps, CostSource
from leadlag.execution.microstructure.execution_constraints import apply_hard_rules, replace_unavailable_short


def test_order_book_snapshot_validation():
    # Valid LOB snapshot
    lob_snap = OrderBookSnapshot(
        ticker="1617.T",
        timestamp="2026-06-23T09:10:00",
        bid_price_1=1000.0,
        bid_size_1=100.0,
        ask_price_1=1002.0,
        ask_size_1=100.0,
        lob_available=True
    )
    assert validate_quote(lob_snap) is True

    # Invalid LOB snapshot (spread negative)
    invalid_lob = OrderBookSnapshot(
        ticker="1617.T",
        timestamp="2026-06-23T09:10:00",
        bid_price_1=1005.0,
        ask_price_1=1000.0,
        lob_available=True
    )
    assert validate_quote(invalid_lob) is False

    # Valid stub snapshot
    stub_snap = OrderBookSnapshot(
        ticker="1617.T",
        timestamp="2026-06-23T09:10:00",
        last_price=1000.0,
        lob_available=False
    )
    assert validate_quote(stub_snap) is True


def test_order_book_cost_calculations():
    # LOB snapshot setup
    snap = OrderBookSnapshot(
        ticker="1617.T",
        timestamp="2026-06-23T09:10:00",
        last_price=1001.0,
        bid_price_1=1000.0, bid_size_1=50,
        bid_price_2=998.0,  bid_size_2=100,
        bid_price_3=995.0,  bid_size_3=150,
        bid_price_4=990.0,  bid_size_4=200,
        bid_price_5=980.0,  bid_size_5=250,
        ask_price_1=1002.0, ask_size_1=50,
        ask_price_2=1004.0, ask_size_2=100,
        ask_price_3=1007.0, ask_size_3=150,
        ask_price_4=1012.0, ask_size_4=200,
        ask_price_5=1022.0, ask_size_5=250,
        lob_available=True
    )

    # Mid price
    assert compute_mid_price(snap) == 1001.0

    # Quoted spread: (1002 - 1000) / 1001 * 10000 = 19.9800 bps
    spread = compute_quoted_spread_bps(snap)
    assert pytest.approx(spread, 0.01) == 19.98

    # Depth BUY (asks) level 1: 1002 * 50 = 50100 JPY
    depth_buy = compute_depth_jpy(snap, "BUY", n_levels=1)
    assert depth_buy == 50100.0

    # Depth BUY level 2: 50100 + (1004 * 100) = 150500 JPY
    depth_buy_2 = compute_depth_jpy(snap, "BUY", n_levels=2)
    assert depth_buy_2 == 150500.0

    # Fill price check
    # Buy order of size 50100 JPY (exact size of ask level 1)
    fill_buy_1 = estimate_market_order_fill_price(snap, "BUY", 50100.0)
    assert fill_buy_1 == 1002.0

    # Buy order of size 100000 JPY (consuming level 1 and part of level 2)
    # level 1: 50100 JPY spent, qty = 50 shares
    # remaining: 49900 JPY spent at 1004, qty = 49.701195 shares
    # total spent = 100000 JPY, total qty = 99.701195
    # fill price = 100000 / 99.701195 = 1002.997
    fill_buy_2 = estimate_market_order_fill_price(snap, "BUY", 100000.0)
    assert pytest.approx(fill_buy_2, 0.01) == 1003.0

    # Slippage check
    slippage = estimate_lob_slippage_bps(snap, "BUY", 100000.0)
    expected_slip = (fill_buy_2 - 1001.0) / 1001.0 * 10000.0
    assert pytest.approx(slippage, 0.01) == expected_slip


def test_lob_not_available_exception():
    stub = OrderBookSnapshot(ticker="1617.T", timestamp="2026-06-23T09:10:00", last_price=1000.0, lob_available=False)
    
    with pytest.raises(LobNotAvailable):
        compute_quoted_spread_bps(stub)

    with pytest.raises(LobNotAvailable):
        compute_depth_jpy(stub, "BUY")


def test_slippage_model_integration():
    # Setup configs/params
    config = {
        "execution": {
            "max_quoted_spread_bps": 30.0,
            "max_estimated_slippage_bps": 20.0
        }
    }

    lob_snap = OrderBookSnapshot(
        ticker="1617.T",
        timestamp="2026-06-23T09:10:00",
        last_price=1001.0,
        bid_price_1=1000.0, bid_size_1=5000,
        ask_price_1=1002.0, ask_size_1=5000,
        lob_available=True,
        cost_source="lob_snapshot"
    )

    # LOB Entry cost
    # spread is 19.98bps, half is 9.99bps. Slippage is small because size is small.
    cost_bps, src = compute_entry_cost_bps(lob_snap, 100000.0, "BUY", 15.0, config)
    assert src == CostSource.LOB_SNAPSHOT
    assert cost_bps > 9.99

    # Exit cost is always fallback
    exit_cost_bps, src_exit = compute_exit_cost_bps(lob_snap, 100000.0, "BUY", 15.0, config)
    assert src_exit == CostSource.FIXED_SPREAD_FALLBACK
    assert exit_cost_bps == 7.5


def test_execution_constraints_hard_rules():
    config = {
        "execution": {
            "max_quoted_spread_bps": 15.0,
            "max_estimated_slippage_bps": 10.0,
            "min_depth_ratio_scale": 1.5,
            "lob_depth_levels": 5
        }
    }

    # Snapshot with high spread (20bps)
    snap_high_spread = OrderBookSnapshot(
        ticker="1617.T",
        timestamp="2026-06-23T09:10:00",
        last_price=1001.0,
        bid_price_1=1000.0, bid_size_1=1000,
        ask_price_1=1002.0, ask_size_1=1000,
        lob_available=True
    )

    # Spread rule should trigger skip
    decision = apply_hard_rules(snap_high_spread, "BUY", 10000.0, True, 0.0, config)
    assert decision.selected is False
    assert "SPREAD_EXCEEDS_CAP" in decision.skip_reason

    # Snapshot with low spread but thin book
    snap_thin_book = OrderBookSnapshot(
        ticker="1617.T",
        timestamp="2026-06-23T09:10:00",
        last_price=1000.5,
        bid_price_1=1000.0, bid_size_1=5,
        ask_price_1=1001.0, ask_size_1=5,
        lob_available=True
    )

    # Order size = 100,000 JPY. Level 1 ask depth = 1001 * 5 = 5005 JPY.
    # Order to depth ratio = 100000 / 5005 ≈ 20.0, which exceeds 1.5 limit.
    # Therefore, scale down should trigger
    decision_scale = apply_hard_rules(snap_thin_book, "BUY", 100000.0, True, 0.0, config)
    assert decision_scale.selected is True
    assert decision_scale.scale_factor < 1.0
    assert "ORDER_DEPTH_RATIO_EXCEEDS_LIMIT" in decision_scale.scale_reason


def test_short_replacements():
    # Initial candidates
    selected_shorts = ["1617.T", "1618.T", "1619.T"]
    
    # All borrowable pool
    available_shorts_pool = ["1618.T", "1620.T", "1621.T", "1622.T"]

    # 1617.T is not borrowable, 1619.T is not borrowable.
    # The output should replace 1617.T and 1619.T with 1620.T and 1621.T from the pool.
    final_shorts = replace_unavailable_short(selected_shorts, available_shorts_pool, max_shorts=3)
    
    assert "1618.T" in final_shorts
    assert "1620.T" in final_shorts
    assert "1621.T" in final_shorts
    assert "1617.T" not in final_shorts
    assert "1619.T" not in final_shorts
    assert len(final_shorts) == 3


def test_from_api_price_response_with_lob():
    """板情報カラム（pGBP/pGAP）が返却された場合に lob_available=True になること。"""
    api_item = {
        "sIssueCode": "1617",
        "pDPP": "1520.0",
        "pPRP": "1510.0",
        "pDOP": "1515.0",
        "pGBP1": "1519.0", "pGBV1": "200",
        "pGBP2": "1518.0", "pGBV2": "300",
        "pGBP3": "1517.0", "pGBV3": "500",
        "pGBP4": "1516.0", "pGBV4": "800",
        "pGBP5": "1514.0", "pGBV5": "1000",
        "pGAP1": "1521.0", "pGAV1": "150",
        "pGAP2": "1522.0", "pGAV2": "250",
        "pGAP3": "1524.0", "pGAV3": "400",
        "pGAP4": "1526.0", "pGAV4": "600",
        "pGAP5": "1530.0", "pGAV5": "900",
    }
    snapshot = from_api_price_response(api_item, timestamp="2026-06-23T09:10:00")

    assert snapshot.ticker == "1617.T"
    assert snapshot.lob_available is True
    assert snapshot.cost_source == "api_lob"
    assert snapshot.last_price == 1520.0

    # Bid side
    assert snapshot.bid_price_1 == 1519.0
    assert snapshot.bid_size_1 == 200.0
    assert snapshot.bid_price_5 == 1514.0
    assert snapshot.bid_size_5 == 1000.0

    # Ask side
    assert snapshot.ask_price_1 == 1521.0
    assert snapshot.ask_size_1 == 150.0
    assert snapshot.ask_price_5 == 1530.0
    assert snapshot.ask_size_5 == 900.0

    # Spread should be valid: bid < ask
    assert snapshot.bid_price_1 < snapshot.ask_price_1
    assert validate_quote(snapshot) is True


def test_from_api_price_response_without_lob():
    """板情報カラムが返却されない（LOBなし）場合は lob_available=False にフォールバック。"""
    api_item = {
        "sIssueCode": "1618",
        "pDPP": "920.5",
        "pPRP": "915.0",
    }
    snapshot = from_api_price_response(api_item, timestamp="2026-06-23T09:10:00")

    assert snapshot.ticker == "1618.T"
    assert snapshot.lob_available is False
    assert snapshot.cost_source == "fixed_spread_fallback"
    assert snapshot.last_price == 920.5

    # All LOB fields should be None
    assert snapshot.bid_price_1 is None
    assert snapshot.ask_price_1 is None
    assert validate_quote(snapshot) is True


def test_from_api_price_response_lob_zero_values():
    """板情報カラムがゼロ（"0" or "0.0000"）で返された場合は None として扱われ LOBなしになること。"""
    api_item = {
        "sIssueCode": "1619",
        "pDPP": "1100.0",
        "pGBP1": "0.0000", "pGBV1": "0",
        "pGAP1": "0.0000", "pGAV1": "0",
    }
    snapshot = from_api_price_response(api_item, timestamp="2026-06-23T09:10:00")

    assert snapshot.ticker == "1619.T"
    assert snapshot.lob_available is False
    assert snapshot.cost_source == "fixed_spread_fallback"
    assert snapshot.bid_price_1 is None
    assert snapshot.ask_price_1 is None

