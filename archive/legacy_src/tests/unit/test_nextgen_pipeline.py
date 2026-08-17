"""Unit tests for the Next-Gen pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from leadlag.data.pit_lake import PITDataLake
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.nextgen_pipeline import NextGenPipeline


def _fake_compute_distribution(
    trade_date: str,
    df_exec: Any,
    current_prices: dict[str, float],
    horizon: int = 1,
    use_file_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n_j = 17
    rng = np.random.RandomState(hash(trade_date) % (2**31))
    mu_gap = rng.normal(0, 0.01, n_j)
    omega_gap = np.eye(n_j) * 0.0001
    return mu_gap, omega_gap


@pytest.fixture
def nextgen_app_config() -> Any:
    """Load production AppConfig for Next-Gen tests."""
    root = Path(__file__).resolve().parents[2]
    return load_config_from_yaml(str(root / "configs" / "production" / "production.yaml"))


def test_pit_ir_history_persistence(
    tmp_path: Path,
    synthetic_df_exec: Any,
    nextgen_app_config: Any,
    monkeypatch: Any,
) -> None:
    """PIT IR history must persist, remain strictly historical, and never leak same-day data."""
    history_path = tmp_path / "pit_ir_history.csv"

    pipeline = NextGenPipeline(
        nextgen_app_config,
        pit_ir_history_path=history_path,
    )
    monkeypatch.setattr(
        pipeline.v2_model,
        "compute_distribution",
        _fake_compute_distribution,
    )

    lake = PITDataLake(synthetic_df_exec)

    # Day 1: no prior history -> fallback Medium/1.0
    pipeline.compute_decision(trade_date="2015-01-05", lake=lake)
    assert history_path.exists()
    assert len(pipeline._pit_records) == 1

    # Day 2: history contains only day 1, still < rolling window for default config
    pipeline.compute_decision(trade_date="2015-01-06", lake=lake)
    assert len(pipeline._pit_records) == 2

    # Day 3: fresh pipeline reloads from file; current date 2015-01-07 should see
    # exactly two prior records (< current date).
    fresh_pipeline = NextGenPipeline(
        nextgen_app_config,
        pit_ir_history_path=history_path,
    )
    monkeypatch.setattr(
        fresh_pipeline.v2_model,
        "compute_distribution",
        _fake_compute_distribution,
    )

    records_before = [d for d, _ in fresh_pipeline._pit_records if d.isoformat() < "2015-01-07"]
    assert len(records_before) == 2

    # Re-running the same date must not allow same-day leakage
    fresh_pipeline.compute_decision(trade_date="2015-01-07", lake=lake)
    assert len([d for d, _ in fresh_pipeline._pit_records if d.isoformat() < "2015-01-07"]) == 2


def test_pit_ir_history_in_memory_no_file(
    synthetic_df_exec: Any,
    nextgen_app_config: Any,
    monkeypatch: Any,
) -> None:
    """Without a history file, the pipeline must keep in-memory history across calls."""
    pipeline = NextGenPipeline(nextgen_app_config)
    monkeypatch.setattr(
        pipeline.v2_model,
        "compute_distribution",
        _fake_compute_distribution,
    )
    lake = PITDataLake(synthetic_df_exec)

    pipeline.compute_decision(trade_date="2015-01-05", lake=lake)
    pipeline.compute_decision(trade_date="2015-01-06", lake=lake)
    assert len(pipeline._pit_ir_history) == 2
