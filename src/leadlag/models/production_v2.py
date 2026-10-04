"""Production v2 portfolio construction module.

Public API: ProductionV2Model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.pit_lake import (
    validate_production_decision_inputs,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.domain.inputs import DecisionInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.v2.decision_engine import _decide as _v2_decide
from leadlag.models.v2.gap_io import compute_distribution as _v2_compute_distribution
from leadlag.utils.cache_manager import CacheManager
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger(__name__)


class ProductionV2Model:
    """Unified V2 production decision model."""

    def __init__(
        self,
        config: ProductionV2RunConfig,
        blpx_model: Any | None = None,
        overlay_model: Any | None = None,
    ) -> None:
        self.run_config = config
        self._raw_config: dict = self.run_config.model_dump()
        self._blpx_model = blpx_model
        self._overlay_model = overlay_model
        self.n_u = len(US_TICKERS)
        self.n_j = len(JP_TICKERS)
        self._cache_manager = CacheManager(
            CacheManager.config_hash_from_pydantic(self.run_config),
            maxsize=128,
        )
        self._macro_price_cache = self._cache_manager.namespace("macro_price")

    def decide(
        self,
        inputs: DecisionInputs,
        *,
        overlay_enabled: bool = True,
    ) -> PortfolioDecision:
        """Calculate from one explicit, versioned market-input contract."""
        validate_production_decision_inputs(inputs)
        return _v2_decide(
            self,
            trade_date=inputs.trade_date.strftime("%Y-%m-%d"),
            overlay_enabled=overlay_enabled,
            use_file_cache=inputs.use_file_cache,
            inputs=inputs,
        )

    def decide_from_cache(
        self,
        trade_date: str,
        gap_input_dir: str | Path | None,
    ) -> PortfolioDecision:
        """Replay a saved distribution, or return flat when it is unavailable.

        This artifact-only entry has no market history for on-demand computation
        or ML overlay application. Live calculation uses ``decide(inputs)``.
        """
        return _v2_decide(
            self,
            trade_date=normalize_jst_date(trade_date).strftime("%Y-%m-%d"),
            gap_input_dir=None if gap_input_dir is None else Path(gap_input_dir),
            overlay_enabled=False,
            use_file_cache=True,
        )

    def compute_distribution(
        self,
        trade_date: str,
        df_exec: pd.DataFrame,
        current_prices: dict[str, float],
        *,
        horizon: int = 1,
        mu_pattern: str | None = None,
        omega_pattern: str | None = None,
        use_file_cache: bool = True,
        snapshot: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _v2_compute_distribution(
            self,
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            horizon=horizon,
            mu_pattern=mu_pattern,
            omega_pattern=omega_pattern,
            use_file_cache=use_file_cache,
            snapshot=snapshot,
        )
