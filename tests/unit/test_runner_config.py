"""tests/unit/test_runner_config.py

Unit tests for runner.config.ProductionConfig.
"""

from __future__ import annotations

import dataclasses

import pytest

from leadlag.execution.config import StrategyConfig as ProductionConfig


class TestProductionConfig:
    def test_default_k(self):
        cfg = ProductionConfig()
        assert isinstance(cfg.k, int)
        assert cfg.k > 0

    def test_default_max_gross_exposure(self):
        cfg = ProductionConfig()
        assert cfg.max_gross_exposure == pytest.approx(2.0)

    def test_default_max_net_exposure(self):
        cfg = ProductionConfig()
        assert cfg.max_net_exposure == pytest.approx(0.05)

    def test_default_var_confidence(self):
        cfg = ProductionConfig()
        assert 0.9 < cfg.var_confidence <= 1.0

    def test_start_date_is_string(self):
        cfg = ProductionConfig()
        assert isinstance(cfg.start_date, str)
        # Should be a valid ISO date
        from datetime import datetime
        dt = datetime.strptime(cfg.start_date, "%Y-%m-%d")
        assert dt.year >= 2010

    def test_is_frozen_immutable(self):
        from pydantic import ValidationError
        cfg = ProductionConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, ValidationError)):
            cfg.k = 99  # type: ignore[misc]

    def test_custom_k(self):
        cfg = ProductionConfig(k=4)
        assert cfg.k == 4

    def test_custom_start_date(self):
        cfg = ProductionConfig(start_date="2018-01-01")
        assert cfg.start_date == "2018-01-01"

    def test_risk_thresholds_consistency(self):
        cfg = ProductionConfig()
        # Warning should be lower than stop
        assert cfg.var_warning < cfg.var_stop
        assert cfg.es_warning < cfg.es_stop
        assert cfg.daily_loss_warning < cfg.daily_loss_stop
