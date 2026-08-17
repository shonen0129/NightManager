"""One-step production runner for the V2 lead-lag pipeline.

This module provides a high-level ``ProductionRunner`` that wires the
BLOX model, the V2 portfolio model, and the optional ML order overlay into
a single ``run()`` call.  It is intended to be used by ``leadlag.cli`` and
by the backtest engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)

__all__ = ["ProductionRunner", "RunnerInputs"]


@dataclass(frozen=True)
class RunnerInputs:
    """Inputs required for the production decision.

    Attributes:
        trade_date: Execution date in ``YYYY-MM-DD`` format.
        df_exec: Pre-processed execution DataFrame (index must contain trade_date).
        gap_input_dir: Directory containing the Step 2 gap-adjusted distribution
            cache (``mu_gap_YYYYMMDD.npy`` / ``omega_gap_YYYYMMDD.npy``), or
            ``None`` to force on-demand computation / flat fallback.
        current_prices: Mapping from JP tickers to 09:10 JPY prices.
        use_file_cache: Prefer pre-computed ``.npy`` file cache when available.
        previous_positions: Optional mapping of currently held share counts.
    """

    trade_date: str
    df_exec: pd.DataFrame
    gap_input_dir: Path | None
    current_prices: dict[str, float]
    use_file_cache: bool = True
    previous_positions: dict[str, int] | None = None


class ProductionRunner:
    """High-level runner that orchestrates the full V2 decision flow."""

    def __init__(self, app_config: Any) -> None:
        """Initialize the runner from a top-level application config.

        Args:
            app_config: A validated ``AppConfig`` (or compatible object) with a
                ``v2`` ``ProductionV2RunConfig`` attribute.
        """
        self.app_config = app_config

        # Resolve overlay config.  Prefer the flat v2 fields from the roadmap
        # (ml_overlay_enabled / ml_overlay_model_dir); fall back to the nested
        # ``ml_order_overlay`` field if the schema has not yet been flattened,
        # and finally to the top-level ``app_config.ml_order_overlay`` used by
        # the legacy schema.
        v2 = app_config.v2
        overlay_enabled: bool | None = getattr(v2, "ml_overlay_enabled", None)
        overlay_model_dir: str | Path | None = getattr(v2, "ml_overlay_model_dir", None)

        if overlay_enabled is None and hasattr(v2, "ml_order_overlay"):
            overlay_enabled = v2.ml_order_overlay.enabled
            overlay_model_dir = overlay_model_dir or v2.ml_order_overlay.model_dir

        if overlay_enabled is None and hasattr(app_config, "ml_order_overlay"):
            overlay_enabled = app_config.ml_order_overlay.enabled
            overlay_model_dir = overlay_model_dir or app_config.ml_order_overlay.model_dir

        if overlay_enabled and overlay_model_dir:
            from leadlag.models.ml_order_overlay import load_overlay_model

            overlay_model = load_overlay_model(Path(overlay_model_dir))
        else:
            overlay_model = None

        self._overlay_enabled = bool(overlay_enabled)

        # Build the BLPX model.  ``ProductionBLPXModel`` lives in the blpx module
        # once Phase 1 is fully landed; the import is deferred so the runner
        # package can still be imported before that file is in place.
        from leadlag.models.blpx import ProductionBLPXModel

        blpx_model = ProductionBLPXModel(app_config.v2.blpx)

        from leadlag.models.production_v2 import ProductionV2Model

        self.model = ProductionV2Model(
            app_config.v2,
            blpx_model=blpx_model,
            overlay_model=overlay_model,
        )

    def run(self, inputs: RunnerInputs) -> dict:
        """Generate the V2 portfolio decision for the supplied inputs.

        Args:
            inputs: A frozen ``RunnerInputs`` dataclass.

        Returns:
            Result dict from ``ProductionV2Model.decide``.

        Raises:
            ValueError: If ``df_exec`` is empty or ``trade_date`` is not in the
                index.
        """
        if inputs.df_exec is None or inputs.df_exec.empty:
            raise ValueError("df_exec must be a non-empty DataFrame")

        if inputs.trade_date not in inputs.df_exec.index:
            raise ValueError(
                f"trade_date {inputs.trade_date} not found in df_exec index"
            )

        # Allow missing prices to fall back to 0 gap per ticker rather than
        # aborting the whole decision.  _extract_gap_inputs treats missing
        # prices as 0 gap, which is the safest partial fallback when a single
        # broker price is unavailable.
        missing_prices = [t for t in JP_TICKERS if t not in inputs.current_prices]
        if missing_prices:
            logger.warning(
                "current_prices missing tickers %s; treating them as 0 gap",
                missing_prices,
            )

        current_prices = dict(inputs.current_prices)
        for t in missing_prices:
            current_prices[t] = 0.0

        return self.model.decide(
            trade_date=inputs.trade_date,
            gap_input_dir=inputs.gap_input_dir,
            df_exec=inputs.df_exec,
            current_prices=current_prices,
            overlay_enabled=self._overlay_enabled,
            use_file_cache=inputs.use_file_cache,
        )
