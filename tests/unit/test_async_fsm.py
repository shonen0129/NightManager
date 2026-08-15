import asyncio
import numpy as np

from leadlag.broker.async_base import AsyncDryRunBrokerClient
from leadlag.broker.base import Position
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.async_fsm import (
    AsyncExecutionEngine,
    ExecutionJournal,
)


def test_async_dry_run_broker_lifecycle():
    """Test AsyncDryRunBrokerClient connect, position update, and disconnect."""
    async def _run():
        async with AsyncDryRunBrokerClient(simulated_latency_ms=5.0) as broker:
            wallet = await broker.get_wallet()
            assert wallet.cash_available == 10_000_000.0

            positions = await broker.get_positions()
            assert len(positions) == 0

    asyncio.run(_run())


def test_compute_order_deltas():
    """Test delta calculation separating close vs new orders."""
    engine = AsyncExecutionEngine()
    
    n_j = len(JP_TICKERS)
    target_weights = np.zeros(n_j)
    target_weights[0] = 0.20  # 1617.T long
    target_weights[1] = -0.20 # 1618.T short

    current_prices = {tk: 1000.0 for tk in JP_TICKERS}
    total_capital = 1_000_000.0

    # Existing positions: currently holding 1617.T SHORT (opposite) and 1619.T LONG
    existing_positions = [
        Position(ticker=JP_TICKERS[0], side="SELL", quantity=100, price=1000.0),
        Position(ticker=JP_TICKERS[2], side="BUY", quantity=100, price=1000.0),
    ]

    close_orders, new_orders = engine.compute_order_deltas(
        target_weights=target_weights,
        current_prices=current_prices,
        current_positions=existing_positions,
        total_capital=total_capital,
    )

    # Must close 1617.T (opposite) and 1619.T (not in target)
    assert len(close_orders) >= 1
    # Must open 1618.T and remaining 1617.T
    assert len(new_orders) >= 1


def test_async_portfolio_execution_end_to_end():
    """Test end-to-end asynchronous staged execution."""
    async def _run():
        engine = AsyncExecutionEngine(split_delay_seconds=0.01)
        
        n_j = len(JP_TICKERS)
        target_weights = np.zeros(n_j)
        target_weights[0] = 0.10  # Long 1617.T
        target_weights[1] = -0.10 # Short 1618.T
        
        # 1629.T is the large order ticker
        idx_1629 = JP_TICKERS.index("1629.T")
        target_weights[idx_1629] = 0.10

        current_prices = {tk: 1000.0 for tk in JP_TICKERS}
        total_capital = 1_000_000.0

        async with AsyncDryRunBrokerClient(simulated_latency_ms=5.0) as broker:
            journal = await engine.execute_portfolio(
                target_weights=target_weights,
                current_prices=current_prices,
                total_capital=total_capital,
                broker=broker,
                trade_date="2026-08-15",
            )

            assert isinstance(journal, ExecutionJournal)
            assert journal.success
            assert journal.total_orders == 3
            assert journal.filled_orders == 3
            assert journal.failed_orders == 0
            assert journal.elapsed_seconds > 0.0

            # Verify broker holds positions
            positions = await broker.get_positions()
            assert len(positions) == 3
            held_tickers = {p.ticker for p in positions}
            assert JP_TICKERS[0] in held_tickers
            assert JP_TICKERS[1] in held_tickers
            assert "1629.T" in held_tickers

    asyncio.run(_run())
