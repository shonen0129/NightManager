"""One-step production runner for the V2 lead-lag pipeline.

This module provides a high-level ``ProductionRunner`` that wires the
BLOX model, the V2 portfolio model, and the optional ML order overlay into
a single ``run()`` call.  It is intended to be used by ``leadlag.cli`` and
by the backtest engine.
"""

from __future__ import annotations

import logging

from leadlag.config.schemas import AppConfig
from leadlag.data.pit_lake import validate_production_decision_inputs
from leadlag.domain.inputs import DecisionInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.runner.model_factory import build_v2_model_bundle

logger = logging.getLogger(__name__)

__all__ = ["ProductionRunner"]


class ProductionRunner:
    """High-level runner that orchestrates the full V2 decision flow."""

    def __init__(self, app_config: AppConfig) -> None:
        """Initialize the runner from a top-level application config.

        Args:
            app_config: A validated ``AppConfig``.
        """
        self.app_config = app_config

        bundle = build_v2_model_bundle(app_config)
        self._overlay_enabled = bundle.overlay_enabled
        self.model = bundle.decision_model

    def run(self, decision_inputs: DecisionInputs) -> PortfolioDecision:
        """Generate a decision from one versioned input contract."""
        if not isinstance(decision_inputs, DecisionInputs):
            raise TypeError("ProductionRunner.run requires DecisionInputs")
        validate_production_decision_inputs(decision_inputs)

        return self.model.decide(
            inputs=decision_inputs,
            overlay_enabled=self._overlay_enabled,
            use_file_cache=decision_inputs.use_file_cache,
        )
